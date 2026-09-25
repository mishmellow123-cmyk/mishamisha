"""Fire: procedural HDR flames (torch/bonfire), distant beacons, sparks & embers, smoke.

Fire standard (director): sparks are tiny (0.6-1.5 px) with a hot head and a tapered
fading tail, steep size/brightness distribution, colour cools over life; slow embers
meander on curl noise. Bonfire: base as wide as the basket, 3-6 licking tongues that
split/merge/tear off, multi-scale domain warp, white-yellow core low, orange body,
darker broken red tips, flicker, height 2-3x basket width, smoke plume lit from below.
"""
import math

import numba as nb
import numpy as np

from core import cam_ray, fbm3, perlin3, splat_gauss, splat_disc, hash11


# ------------------------------------------------------------ colour ramp ---

@nb.njit(cache=True, fastmath=True, inline='always')
def bb(t):
    """numba copy of look.blackbody (5 stops)."""
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    if t < 0.30:
        w = t / 0.30
        return 0.35 + (0.90 - 0.35) * w, 0.02 + (0.12 - 0.02) * w, 0.00 + 0.01 * w
    elif t < 0.55:
        w = (t - 0.30) / 0.25
        return 0.90 + 0.10 * w, 0.12 + (0.36 - 0.12) * w, 0.01 + 0.03 * w
    elif t < 0.78:
        w = (t - 0.55) / 0.23
        return 1.0, 0.36 + (0.68 - 0.36) * w, 0.04 + (0.25 - 0.04) * w
    w = (t - 0.78) / 0.22
    return 1.0, 0.68 + (0.95 - 0.68) * w, 0.25 + (0.85 - 0.25) * w


@nb.njit(cache=True, fastmath=True, inline='always')
def sstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------ flame ---

@nb.njit(cache=True, fastmath=True)
def render_flame(img, cam, bx, by, bz, height, width, lean, t, seed, inten, st, x0, y0, x1, y1, alpha_out):
    """Flame on a camera-facing card at world depth bz, base centre (bx,by), in metres.
    width = half-width at the base. lean = horizontal tip offset in units of width.
    st: style [tongue_freq, turb, rise, core, top, bonfire(0..1), ragged, detail]
    Adds emission to img; accumulates flame opacity-ish mask into alpha_out (for occlusion
    of things behind thick flame cores)."""
    H = img.shape[0]
    W = img.shape[1]
    xa = max(0, x0)
    xb = min(W, x1)
    ya = max(0, y0)
    yb = min(H, y1)
    rs = st[2]
    sb = st[5]
    for j in range(ya, yb):
        for i in range(xa, xb):
            dx, dy, dz = cam_ray(cam, i + 0.5, j + 0.5)
            if dz <= 1e-6:
                continue
            th = (bz - cam[2]) / dz
            if th <= 0:
                continue
            X = cam[0] + th * dx - bx
            Y = cam[1] + th * dy - by
            u = X / width
            v = Y / height
            if v < -0.2 or v > 1.9:
                continue
            vc = v if v > 0.0 else 0.0
            uu = u - lean * vc ** 1.5
            if abs(uu) > 2.6 + 0.8 * v:
                continue
            # multi-scale domain warp (rising)
            w1 = fbm3(uu * 0.8 + seed, v * 1.1 - t * rs * 0.9, t * 0.33 + seed * 0.7, 3, 2.0, 0.5)
            w2 = fbm3(uu * 1.9 - seed, v * 2.3 - t * rs * 1.6, t * 0.55 + 5.0, 3, 2.0, 0.5)
            uw = uu + (st[1] * w1 + 0.45 * st[1] * w2) * (0.25 + v)
            vw = v + 0.10 * w2 * (0.3 + v)
            # envelope: teardrop (torch) -> wide base (bonfire)
            vv = vw if vw > 0.0 else 0.0
            one = 1.0 - vv
            if one < 0.0:
                one = 0.0
            hw_t = 1.75 * math.sqrt(vv + 0.03) * one ** 0.85
            hw_b = (1.0 - min(vv, 1.0) ** 0.9) ** 0.75 * (1.0 + 0.12 * w1) * (1.0 + 0.25 * sstep(0.25, 0.0, vv))
            hw = hw_t + (hw_b - hw_t) * sb
            if hw < 1e-3:
                hw = 1e-3
            edge = 1.0 - abs(uw) / hw
            # tongues: rising noise field with horizontal frequency -> 3-6 licking tongues
            tn = fbm3(uw * st[0] + seed * 3.1, vw * 1.3 - t * rs * 1.5, t * 0.45 + seed, 4, 2.0, 0.5)
            vtop = st[4] + 0.62 * tn * (0.6 + 0.4 * sb)
            tong = sstep(vtop + 0.06, vtop - 0.24, vw)
            # detached, torn-off fragments above the tongues
            frag = fbm3(uw * st[0] * 1.6 + 9.0, vw * 2.4 - t * rs * 2.2, t * 0.8 + seed, 3, 2.0, 0.5)
            fr = sstep(0.16, 0.36, frag) * sstep(vtop - 0.1, vtop + 0.2, vw) * sstep(1.5, 0.95, vw)
            body = sstep(-0.02, 0.28, edge)
            dens = body * max(tong, 0.75 * fr)
            if vw < 0.0:
                dens *= sstep(-0.14, 0.0, vw)
            if dens <= 0.002:
                continue
            # fine rising detail -> licks, gaps, internal structure
            d3 = fbm3(uw * 4.2, vw * 4.8 - t * rs * 3.2, t * 1.1 + seed, st[7] if st[7] > 0 else 2, 2.0, 0.5)
            dm = 0.58 + 1.15 * d3
            if dm < 0.0:
                dm = 0.0
            if dm > 1.3:
                dm = 1.3
            dens *= dm
            if dens <= 0.0:
                continue
            # temperature: hot low & central, cooler/broken tips
            core = math.exp(-(uw / 0.42) ** 2) * math.exp(-((vw - 0.10) / 0.28) ** 2) * st[3]
            T = dens * (0.80 - 0.46 * vv) + core * dens
            ragged = st[6] * sstep(0.45, 1.1, vw) * (0.5 + 0.5 * d3)
            T -= ragged * 0.30
            if T <= 0:
                continue
            if T > 1.25:
                T = 1.25
            cr, cg, cb = bb(0.15 + 0.85 * min(T, 1.0))
            e = inten * (T * T * T + 0.02 * T)
            img[j, i, 0] += cr * e
            img[j, i, 1] += cg * e
            img[j, i, 2] += cb * e
            a = min(1.0, T * 1.3)
            if a > alpha_out[j, i]:
                alpha_out[j, i] = a


def flame_bbox(cam, base, height, width, lean):
    """Screen bbox for a flame (with generous margins)."""
    pts = []
    for (dxm, dym) in [(-2.8, -0.3), (2.8, -0.3), (-2.8 + lean, 2.0), (2.8 + lean, 2.0), (lean, 2.0)]:
        pts.append([base[0] + dxm * width, base[1] + dym * height, base[2]])
    sx, sy, z = cam.project(np.array(pts))
    if np.any(z <= 0):
        return None
    x0, x1 = int(np.floor(sx.min())) - 2, int(np.ceil(sx.max())) + 2
    y0, y1 = int(np.floor(sy.min())) - 2, int(np.ceil(sy.max())) + 2
    if x1 < 0 or y1 < 0 or x0 > cam.W or y0 > cam.H:
        return None
    return x0, y0, x1, y1


TORCH_STYLE = np.array([2.4, 0.32, 2.2, 0.60, 0.74, 0.0, 0.5, 3], np.float64)
BONFIRE_STYLE = np.array([3.4, 0.45, 1.7, 0.70, 0.58, 1.0, 0.8, 3], np.float64)


def draw_flame(img, alpha, cam, base, height, width, lean, t, seed, inten, style):
    bbx = flame_bbox(cam, base, height, width, lean)
    if bbx is None:
        return
    x0, y0, x1, y1 = bbx
    render_flame(img, cam.params(), float(base[0]), float(base[1]), float(base[2]), float(height),
                 float(width), float(lean), float(t), float(seed), float(inten), style,
                 x0, y0, x1, y1, alpha)


# ---------------------------------------------------------------- flicker ---

def flicker(t, seed=0.0, amt=1.0):
    """Multiplicative light flicker around 1."""
    from core import fnoise1
    v = 0.55 * fnoise1(t * 5.1, seed, 3) + 0.30 * fnoise1(t * 11.7, seed + 3.0, 2) + 0.15 * fnoise1(t * 23.0, seed + 7.0, 1)
    return 1.0 + 0.16 * amt * v


def ignition(tr, overshoot=0.6, rise=0.45, settle=0.9):
    """Beacon ignition envelope (tr = seconds since ignition). Whoomp with overshoot."""
    if tr <= 0:
        return 0.0
    a = 1 - math.exp(-tr / (rise * 0.35))
    bump = overshoot * math.exp(-((tr - rise) / (0.35 * settle)) ** 2)
    return a * (1.0 + bump)


# ------------------------------------------------------- distant beacons ---

@nb.njit(cache=True, fastmath=True)
def splat_beacon(img, depth, x, y, z, hpx, energy, halo_px, halo_e, r, g, b):
    """Tiny distant beacon: 2-lobe vertical core + a soft warm halo, occluded by depth."""
    # core lobes
    s0 = max(0.45, 0.35 * hpx)
    _occ_gauss(img, depth, x, y - 0.25 * hpx, z, s0, energy * 0.6 * r, energy * 0.6 * g, energy * 0.6 * b)
    _occ_gauss(img, depth, x, y - 0.65 * hpx, z, s0 * 0.8, energy * 0.4 * r, energy * 0.4 * g * 0.8, energy * 0.4 * b * 0.6)
    if halo_e > 0:
        _occ_gauss(img, depth, x, y - 0.8 * hpx, z, halo_px, halo_e * r, halo_e * g * 0.75, halo_e * b * 0.5)


@nb.njit(cache=True, fastmath=True)
def _occ_gauss(img, depth, x, y, z, sigma, r, g, b):
    H = img.shape[0]
    W = img.shape[1]
    if sigma < 0.3:
        sigma = 0.3
    rad = int(math.ceil(sigma * 3.0))
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    if ix + rad < 0 or iy + rad < 0 or ix - rad >= W or iy - rad >= H:
        return
    inv = 1.0 / (2.0 * sigma * sigma)
    tot = 0.0
    for jj in range(iy - rad, iy + rad + 1):
        ddy = jj + 0.5 - y
        for ii in range(ix - rad, ix + rad + 1):
            ddx = ii + 0.5 - x
            tot += math.exp(-(ddx * ddx + ddy * ddy) * inv)
    k = 1.0 / tot
    for jj in range(max(0, iy - rad), min(H, iy + rad + 1)):
        ddy = jj + 0.5 - y
        for ii in range(max(0, ix - rad), min(W, ix + rad + 1)):
            dd = depth[jj, ii]
            vis = 1.0
            if dd < z * 0.98:
                vis = 0.0
            if vis <= 0:
                continue
            ddx = ii + 0.5 - x
            w = math.exp(-(ddx * ddx + ddy * ddy) * inv) * k * vis
            img[jj, ii, 0] += r * w
            img[jj, ii, 1] += g * w
            img[jj, ii, 2] += b * w


# -------------------------------------------------------------- particles ---

@nb.njit(cache=True, fastmath=True)
def _curl(px, py, pz, t, sc):
    """Cheap divergence-free-ish turbulence from potential noise (finite differences)."""
    e = 0.13
    x = px * sc
    y = py * sc
    z = pz * sc
    n1 = perlin3(x, y + e, z + t) - perlin3(x, y - e, z + t)
    n2 = perlin3(x + 31.4, y, z + e + t) - perlin3(x + 31.4, y, z - e + t)
    n3 = perlin3(x + e + 12.9, y, z + t) - perlin3(x - e + 12.9, y, z + t)
    n4 = perlin3(x + 12.9, y + e, z + t) - perlin3(x + 12.9, y - e, z + t)
    n5 = perlin3(x + e + 57.1, y, z + t) - perlin3(x - e + 57.1, y, z + t)
    n6 = perlin3(x + 57.1, y, z + e + t) - perlin3(x + 57.1, y, z - e + t)
    inv = 1.0 / (2 * e)
    return (n1 - n2) * inv, (n3 - n4) * inv, (n5 - n6) * inv


@nb.njit(cache=True, fastmath=True)
def step_particles(P, V, T, L, A, n, dt, t, wind, buoy, drag, turb, turb_sc, grav, cool):
    """In-place Euler step for n live particles. T = temperature (1 hot -> 0), L = life
    rate, A = alive flag."""
    for k in range(n):
        if A[k] == 0:
            continue
        cx, cy, cz = _curl(P[k, 0], P[k, 1], P[k, 2], t * 0.35, turb_sc)
        ax = drag * (wind[0] + turb * cx - V[k, 0])
        ay = drag * (wind[1] + turb * cy - V[k, 1]) + buoy * T[k] - grav
        az = drag * (wind[2] + turb * cz - V[k, 2])
        V[k, 0] += ax * dt
        V[k, 1] += ay * dt
        V[k, 2] += az * dt
        P[k, 0] += V[k, 0] * dt
        P[k, 1] += V[k, 1] * dt
        P[k, 2] += V[k, 2] * dt
        T[k] -= L[k] * dt * cool
        if T[k] <= 0.0:
            A[k] = 0


class ParticleSim:
    """Deterministic particle history for a whole shot (random frame access).

    emit(frame_time) -> list of (count, pos_fn, vel_fn) handled by the owner via
    a callback returning arrays. Stores per-frame snapshots for rendering with
    motion blur (positions at shutter open/close)."""

    def __init__(self, seed, cap=6000):
        self.rng = np.random.default_rng(seed)
        self.cap = cap
        self.P = np.zeros((cap, 3))
        self.V = np.zeros((cap, 3))
        self.T = np.zeros(cap)
        self.L = np.zeros(cap)
        self.A = np.zeros(cap, np.int64)
        self.S = np.zeros(cap)        # size/brightness class
        self.K = np.zeros(cap, np.int64)  # kind (0 spark, 1 ember)
        self.snap = {}

    def spawn(self, pos, vel, life, size, kind, T0=1.0):
        m = len(pos)
        free = np.nonzero(self.A == 0)[0][:m]
        m = len(free)
        if m == 0:
            return
        self.P[free] = pos[:m]
        self.V[free] = vel[:m]
        self.T[free] = T0 if np.isscalar(T0) else np.asarray(T0)[:m]
        self.L[free] = 1.0 / np.maximum(life[:m], 1e-3)
        self.A[free] = 1
        self.S[free] = size[:m]
        self.K[free] = kind[:m]

    def run(self, f0, f1, emitter, params, substeps=4, fps=24.0, shutter=0.5):
        """Simulate from frame f0 to f1 inclusive. emitter(sim, f, dt) spawns.
        params(f) -> dict(wind, buoy, drag, turb, turb_sc, grav, cool)."""
        dt = 1.0 / (fps * substeps)
        for f in range(f0, f1 + 1):
            for s in range(substeps):
                ft = f + s / substeps
                pr = params(ft)
                emitter(self, ft, dt)
                step_particles(self.P, self.V, self.T, self.L, self.A, self.cap, dt, ft / fps,
                               np.asarray(pr['wind'], np.float64), pr['buoy'], pr['drag'], pr['turb'],
                               pr['turb_sc'], pr['grav'], pr['cool'])
                if s == substeps - 1 - int(round(substeps * shutter)) + 1 or (substeps == 1):
                    pass
            idx = np.nonzero(self.A)[0]
            # store state at end of frame + velocity (for streaks)
            self.snap[f] = (self.P[idx].copy(), self.V[idx].copy(), self.T[idx].copy(),
                            self.S[idx].copy(), self.K[idx].copy())


@nb.njit(cache=True, fastmath=True)
def splat_spark(img, hx, hy, tx, ty, w_head, energy, r, g, b):
    """Tapered streak: bright head at (hx,hy), fading thin tail toward (tx,ty)."""
    L = math.sqrt((hx - tx) ** 2 + (hy - ty) ** 2)
    n = int(L / 0.4) + 1
    if n > 300:
        n = 300
    # weights ~ s^2 (s=1 at head), normalised
    tot = 0.0
    for q in range(n):
        s = 1.0 - q / n
        tot += s * s + 0.02
    for q in range(n):
        s = 1.0 - q / n
        w = (s * s + 0.02) / tot
        sig = w_head * (0.45 + 0.55 * s)
        splat_gauss(img, hx + (tx - hx) * (q / n), hy + (ty - hy) * (q / n), sig,
                    energy * w * r, energy * w * g, energy * w * b)


@nb.njit(cache=True, fastmath=True)
def render_sparks(img, cam, P, V, T, S, K, shutter, energy_scale, px_scale, focus, aperture,
                  ember_gain, spark_gain, depth_cut):
    """Project and splat particles. shutter in seconds (streak length = v*shutter).
    focus = focus distance (m); aperture = CoC px per (1/m) difference (0 = pinhole)."""
    H = img.shape[0]
    W = img.shape[1]
    f = cam[12]
    for k in range(P.shape[0]):
        x = P[k, 0] - cam[0]
        y = P[k, 1] - cam[1]
        z = P[k, 2] - cam[2]
        qx = cam[3] * x + cam[6] * y + cam[9] * z
        qy = cam[4] * x + cam[7] * y + cam[10] * z
        qz = cam[5] * x + cam[8] * y + cam[11] * z
        if qz < depth_cut:
            continue
        hx = cam[13] + f * qx / qz
        hy = cam[14] - f * qy / qz
        if hx < -60 or hy < -60 or hx > W + 60 or hy > H + 60:
            continue
        # tail point
        x2 = x - V[k, 0] * shutter
        y2 = y - V[k, 1] * shutter
        z2 = z - V[k, 2] * shutter
        qx2 = cam[3] * x2 + cam[6] * y2 + cam[9] * z2
        qy2 = cam[4] * x2 + cam[7] * y2 + cam[10] * z2
        qz2 = cam[5] * x2 + cam[8] * y2 + cam[11] * z2
        if qz2 < depth_cut:
            qz2 = depth_cut
        tx = cam[13] + f * qx2 / qz2
        ty = cam[14] - f * qy2 / qz2
        tt = T[k]
        if tt <= 0:
            continue
        # colour cools over life: white-yellow -> orange -> deep red -> gone
        cr, cg, cb = bb(0.25 + 0.75 * tt)
        br = S[k] * (tt ** 1.6) * (spark_gain if K[k] == 0 else ember_gain)
        # physical size falls with distance but never below ~0.6 px
        e = br * energy_scale * min(4.0, 3.0 / max(qz, 0.2))
        coc = 0.0
        if aperture > 0:
            coc = aperture * abs(1.0 / qz - 1.0 / focus)
        whead = px_scale * (0.30 + 0.25 * S[k] ** 0.5)
        if coc > 1.6:
            splat_disc(img, 0.5 * (hx + tx), 0.5 * (hy + ty), coc, e * cr, e * cg, e * cb, 1.0)
        else:
            splat_spark(img, hx, hy, tx, ty, max(whead, 0.3 + 0.5 * coc), e, cr, cg, cb)


# ------------------------------------------------------------------ smoke ---

@nb.njit(cache=True, fastmath=True)
def render_smoke(rgb, alpha, cam, bx, by, bz, t, seed, h_max, w0, grow, drift, rise,
                 opacity, lit_r, lit_g, lit_b, lit_len, amb_r, amb_g, amb_b, x0, y0, x1, y1):
    """Sparse, billowing dark smoke column on a card at depth bz (premultiplied rgb +
    alpha, to be composited OVER the scene). Lit warm only near the fire."""
    H = rgb.shape[0]
    W = rgb.shape[1]
    for j in range(max(0, y0), min(H, y1)):
        for i in range(max(0, x0), min(W, x1)):
            dx, dy, dz = cam_ray(cam, i + 0.5, j + 0.5)
            if dz <= 1e-6:
                continue
            th = (bz - cam[2]) / dz
            X = cam[0] + th * dx - bx
            Y = cam[1] + th * dy - by
            if Y < 0 or Y > h_max:
                continue
            # centreline bends downwind and wanders
            xc = drift * Y ** 1.15 + 0.25 * w0 * perlin3(Y * 0.6 / w0 - t * rise * 0.3, seed, t * 0.1)
            wd = w0 + grow * Y
            q = (X - xc) / wd
            if abs(q) > 2.2:
                continue
            env = math.exp(-q * q * 1.6) * sstep(0.0, 0.08 * h_max, Y) * sstep(h_max, 0.55 * h_max, Y)
            # billows advected upward
            sy = Y - t * rise
            n = fbm3(X * 1.3 / wd + seed, sy * 1.1 / wd, t * 0.12 + seed, 4, 2.0, 0.55)
            dens = env * sstep(-0.05, 0.35, n + 0.12 * env)
            if dens <= 0.003:
                continue
            a = opacity * dens
            lit = math.exp(-Y / lit_len)
            # underside lit: more light where density gradient faces down (cheap: low Y)
            cr = amb_r + lit_r * lit
            cg = amb_g + lit_g * lit
            cb = amb_b + lit_b * lit
            ao = 1.0 - a
            rgb[j, i, 0] = rgb[j, i, 0] * ao + cr * a
            rgb[j, i, 1] = rgb[j, i, 1] * ao + cg * a
            rgb[j, i, 2] = rgb[j, i, 2] * ao + cb * a
            alpha[j, i] = alpha[j, i] * ao + a


def add_glow(img, cam, pos, radius_m, inten, col=(1.0, 0.55, 0.22)):
    """Veiling glow around a bright source (air/smoke scattering + lens glare). Works even
    when the source is just outside the frame."""
    sx, sy, z = cam.project(np.asarray(pos, np.float64))
    if z <= 0.02:
        return
    rpx = cam.f * radius_m / z
    H, W = img.shape[:2]
    ys = (np.arange(H, dtype=np.float32) + 0.5 - sy)[:, None]
    xs = (np.arange(W, dtype=np.float32) + 0.5 - sx)[None, :]
    d2 = (xs * xs + ys * ys) / (rpx * rpx)
    g = inten / (1.0 + d2) ** 1.5
    img += g[..., None] * np.asarray(col, np.float32)
