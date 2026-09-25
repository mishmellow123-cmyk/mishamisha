"""Fire: procedural HDR flame, ignition envelope, deterministic sparks (motion-blurred streaks),
smoke puffs lit from below, and small distant beacon glows.

Everything is a pure function of time (no frame-to-frame state) so any frame renders alone.
"""
import math

import numpy as np
from numba import njit, prange

from .noise import fbm3, gnoise3, gnoise2, smoothstep, h01

# look.blackbody stops (t, r, g, b) -- kept identical so all departments share the ramp
_BB = np.array([[0.00, 0.35, 0.02, 0.00],
                [0.30, 0.90, 0.12, 0.01],
                [0.55, 1.00, 0.36, 0.04],
                [0.78, 1.00, 0.68, 0.25],
                [1.00, 1.00, 0.95, 0.85]])


@njit(inline='always', fastmath=True)
def bb(t):
    if t <= 0.0:
        return _BB[0, 1], _BB[0, 2], _BB[0, 3]
    if t >= 1.0:
        return _BB[4, 1], _BB[4, 2], _BB[4, 3]
    k = 0
    while k < 3 and t > _BB[k + 1, 0]:
        k += 1
    w = (t - _BB[k, 0]) / (_BB[k + 1, 0] - _BB[k, 0])
    return (_BB[k, 1] + (_BB[k + 1, 1] - _BB[k, 1]) * w,
            _BB[k, 2] + (_BB[k + 1, 2] - _BB[k, 2]) * w,
            _BB[k, 3] + (_BB[k + 1, 3] - _BB[k, 3]) * w)


FIRE_LIGHT = np.array([1.0, 0.52, 0.18])   # colour of firelight on surroundings (linear)


# ------------------------------------------------------------------ envelope ---

def ignite_env(t, t0, fps=24.0):
    """Ignition envelope at absolute time t (s) for ignition time t0 (s).
    Returns (size, intensity, light) multipliers. 0 before ignition."""
    u = (t - t0) * fps   # frames since ignition
    if u < -1.5:
        return 0.0, 0.0, 0.0
    if u < 0.0:
        # the catch: a small tongue one frame before the hit
        k = (u + 1.5) / 1.5
        return 0.22 * k, 0.8 * k, 0.25 * k
    # whoosh: fast rise with overshoot, settle
    rise = 1.0 - math.exp(-u / 1.6)
    over = 0.38 * math.exp(-((u - 5.0) / 4.5) ** 2)
    size = 0.55 + 0.45 * rise + over
    flash = 1.25 * math.exp(-u / 3.5)
    inten = 1.0 + flash
    light = 1.0 + 1.1 * math.exp(-u / 5.0)
    return size, inten, light


def flicker(t, seed=0, amt=1.0):
    """Slow organic flicker multiplier ~1 (for light intensity)."""
    s = seed * 1.7
    v = (0.06 * math.sin(2 * math.pi * 1.9 * t + s) + 0.04 * math.sin(2 * math.pi * 3.7 * t + 2.1 + s)
         + 0.03 * math.sin(2 * math.pi * 7.3 * t + 0.7 * s))
    return 1.0 + amt * v


# ---------------------------------------------------------------------- flame ---

# tunable flame look (index: meaning)
FLAME_P = np.array([
    0.55,   # 0 large warp amplitude
    1.1,    # 1 warp growth with height
    1.9,    # 2 taper (narrowing with height)
    0.33,   # 3 teardrop centre (qy)
    0.72,   # 4 teardrop vertical radius
    0.10,   # 5 erosion at base
    0.95,   # 6 erosion growth with height
    0.05,   # 7 bias
    1.25,   # 8 gain F->T
    0.24,   # 9 hot-core amount
    4.0,    # 10 emission power
    2.2,    # 11 detail noise x frequency (per Rb)
    0.55,   # 12 detail noise y frequency (per Rb)
    0.78,   # 13 body T scale (before core)
    0.55,   # 14 x squash in teardrop distance
])


@njit(parallel=True, fastmath=True, cache=True)
def _flame(img, depth, bx, by, ppm, zf, Hf, Rb, lean, t, seed, I, x0, x1, y0, y1, zbias, detail, alpha_mul,
           FP, debug):
    """Flame field in normalised coords q=(X/Rb, Y/Hf): a teardrop distance eroded by noise that is
    advected upward; erosion grows with height so the top breaks into tongues."""
    rise = 1.5 * math.sqrt(max(Hf, 0.05)) + 0.6          # m/s
    for py in prange(y0, y1):
        for px in range(x0, x1):
            if depth[py, px] < zf - zbias:
                continue
            X = (px + 0.5 - bx) / ppm
            Y = (by - (py + 0.5)) / ppm
            qy = Y / Hf
            if qy < -0.1 or qy > 1.7:
                continue
            qyc = max(qy, 0.0)
            xc = lean * qyc * qyc + 0.12 * Rb * math.sin(1.9 * t + 1.6 * qyc + seed) * qyc
            qx = (X - xc) / Rb
            w1 = fbm3(qx * 0.9, (Y - rise * t) / Rb * 0.45, t * 0.35 + seed, 3.0, seed)
            qxw = qx + FP[0] * w1 * (0.2 + FP[1] * qyc)
            sx = qxw * (1.0 + FP[2] * qyc * qyc)
            sy = (qy - FP[3]) / FP[4]
            r = math.sqrt(sx * sx * FP[14] + sy * sy)
            n = fbm3(qxw * FP[11], (Y - 1.25 * rise * t) / Rb * FP[12], t * 0.7 + 3.0 * seed, detail, seed + 29)
            n = 0.5 + 0.5 * n
            F = 1.0 - r - (1.0 - n) * (FP[5] + FP[6] * qyc) + FP[7]
            F *= smoothstep(-0.1, 0.03, qy)
            if F <= 0.0:
                continue
            T = min(F * FP[8], 1.0) * FP[13]
            core = math.exp(-(qx * 1.3) ** 2) * math.exp(-((qy - 0.2) / 0.3) ** 2)
            T = min(T + FP[9] * core * min(F * 3.0, 1.0), 1.0)
            if debug > 0:
                img[py, px, 0] += T
                img[py, px, 1] += T
                img[py, px, 2] += T
                continue
            r_, g_, b_ = bb(T)
            e_int = I * T ** FP[10] * alpha_mul
            img[py, px, 0] += r_ * e_int
            img[py, px, 1] += g_ * e_int
            img[py, px, 2] += b_ * e_int


def flame(img, depth, cam, base_world, Hf, Rb, t, seed=0, I=14.0, lean=0.0, zbias=0.5, detail=4.0,
          alpha=1.0, min_px=1.5, FP=None, debug=0):
    """Draw a flame whose base centre sits at world point base_world. Hf/Rb in metres."""
    if Hf <= 0.01 or I <= 0:
        return
    sx, sy, z = cam.project(np.asarray(base_world, np.float64))
    if z <= 0.1:
        return
    ppm = cam.f / z
    H, W = img.shape[:2]
    if Hf * ppm < min_px * 3:
        # too small for structure -> glow point
        glow(img, depth, sx, sy - 0.4 * Hf * ppm, max(Hf * ppm * 0.35, 0.8), I * 0.22 * (Hf * Rb * ppm * ppm) / 3.0,
             z, zbias)
        return
    x0 = int(max(0, sx - (Rb * 1.9 + abs(lean) * 1.8) * ppm - 2))
    x1 = int(min(W, sx + (Rb * 1.9 + abs(lean) * 1.8) * ppm + 2))
    y0 = int(max(0, sy - Hf * 1.8 * ppm - 2))
    y1 = int(min(H, sy + 0.15 * Hf * ppm + 2))
    if x1 <= x0 or y1 <= y0:
        return
    _flame(img, depth, float(sx), float(sy), float(ppm), float(z), float(Hf), float(Rb), float(lean), float(t),
           float(seed), float(I), x0, x1, y0, y1, float(zbias), float(detail), float(alpha),
           FLAME_P if FP is None else FP, int(debug))


@njit(fastmath=True)
def _glow(img, depth, sx, sy, rad, r, g, b, z, zbias):
    H, W = img.shape[0], img.shape[1]
    R = int(math.ceil(rad * 3.5)) + 1
    ix = int(sx)
    iy = int(sy)
    norm = 1.0 / (6.2832 * rad * rad)
    for py in range(iy - R, iy + R + 1):
        if py < 0 or py >= H:
            continue
        for px in range(ix - R, ix + R + 1):
            if px < 0 or px >= W:
                continue
            if depth[py, px] < z - zbias:
                continue
            dx = px + 0.5 - sx
            dy = py + 0.5 - sy
            w = math.exp(-(dx * dx + dy * dy) / (2 * rad * rad)) * norm
            img[py, px, 0] += r * w
            img[py, px, 1] += g * w
            img[py, px, 2] += b * w


def glow(img, depth, sx, sy, rad_px, energy, z=1e9, zbias=0.0, col=None):
    """Gaussian splat of total energy `energy` (sum over pixels) -- tiny distant fires."""
    if col is None:
        col = np.array([1.0, 0.62, 0.26])
    c = np.asarray(col) * energy
    _glow(img, depth, float(sx), float(sy), float(max(rad_px, 0.6)), float(c[0]), float(c[1]), float(c[2]),
          float(z), float(zbias))


@njit(parallel=True, fastmath=True)
def _halo(img, depth, sx, sy, sig, r, g, b, z, zbias, x0, x1, y0, y1, squash):
    for py in prange(y0, y1):
        for px in range(x0, x1):
            if depth[py, px] < z - zbias:
                continue
            dx = (px + 0.5 - sx) / sig
            dy = (py + 0.5 - sy) / (sig * squash)
            w = math.exp(-0.5 * (dx * dx + dy * dy))
            img[py, px, 0] += r * w
            img[py, px, 1] += g * w
            img[py, px, 2] += b * w


def halo(img, depth, sx, sy, sig_px, peak, z=1e9, zbias=0.0, col=None, squash=1.0):
    """Soft air-glow around a fire (peak radiance at centre)."""
    if col is None:
        col = FIRE_LIGHT
    H, W = img.shape[:2]
    R = sig_px * 3.2
    x0 = int(max(0, sx - R))
    x1 = int(min(W, sx + R))
    y0 = int(max(0, sy - R * squash))
    y1 = int(min(H, sy + R * squash))
    if x1 <= x0 or y1 <= y0:
        return
    c = np.asarray(col) * peak
    _halo(img, depth, float(sx), float(sy), float(sig_px), float(c[0]), float(c[1]), float(c[2]), float(z),
          float(zbias), x0, x1, y0, y1, float(squash))


# --------------------------------------------------------------------- sparks ---

class Sparks:
    """Deterministic spark system. Particles are spawned by schedule; each frame integrates every
    live particle from its birth, so frames are independent."""

    def __init__(self, seed, origin, t_ign, t_end, burst=350, rate=70.0, speed=(3.0, 9.0), spread=0.55,
                 life=(0.8, 2.4), wind=(0.8, 0.0, 0.0), buoy=7.0, updraft_h=3.0, drag=1.4, radius=0.55,
                 size=0.012, temp=(0.75, 1.0), I=6.0, burst_speed=1.6, turb=2.5, rate_ramp=0.4,
                 pre=0.0):
        rng = np.random.default_rng(seed)
        nb = int(burst)
        nc = int(max(0.0, (t_end - t_ign)) * rate) + 1
        n = nb + nc
        ts = np.empty(n)
        ts[:nb] = t_ign + rng.random(nb) ** 2 * 0.12
        # continuous emission, density ramping up after ignition
        u = np.sort(rng.random(nc))
        ts[nb:] = t_ign + 0.05 + u * (t_end - t_ign)
        ang = rng.random(n) * 2 * np.pi
        rr = np.sqrt(rng.random(n)) * radius
        p0 = np.stack([origin[0] + rr * np.cos(ang), origin[1] + rng.random(n) * 0.3,
                       origin[2] + rr * np.sin(ang)], 1)
        sp = rng.uniform(speed[0], speed[1], n)
        sp[:nb] *= burst_speed * (0.6 + 0.8 * rng.random(nb))
        th = rng.normal(size=n) * spread
        ph = rng.random(n) * 2 * np.pi
        vx = sp * np.sin(th) * np.cos(ph)
        vz = sp * np.sin(th) * np.sin(ph)
        vy = sp * np.cos(th)
        self.v0 = np.stack([vx, vy, vz], 1)
        self.p0 = p0
        self.ts = ts
        self.life = rng.uniform(life[0], life[1], n)
        self.life[:nb] *= 1.2
        self.temp = rng.uniform(temp[0], temp[1], n)
        self.size = size * (0.5 + rng.random(n) ** 2 * 1.6)
        self.ph = rng.random((n, 3)) * 2 * np.pi
        self.tw = rng.uniform(6.0, 22.0, n)
        self.params = np.array([wind[0], wind[1], wind[2], buoy, updraft_h, drag, turb, origin[1]], np.float64)
        self.I = I
        self.n = n

    def state(self, t, shutter=1.0 / 48.0, K=4):
        pts = np.zeros((self.n, K, 3))
        alive = np.zeros(self.n, np.bool_)
        age = np.zeros(self.n)
        _integrate(self.p0, self.v0, self.ts, self.life, self.ph, self.params, t, shutter, K, pts, alive, age)
        return pts, alive, age

    def render(self, img, depth, cam, t, gain=1.0, shutter=1.0 / 48.0, zbias=0.3, max_len_px=200.0):
        pts, alive, age = self.state(t, shutter)
        if not alive.any():
            return
        idx = np.nonzero(alive)[0]
        P = pts[idx]                          # (m, K, 3)
        sx, sy, z = cam.project(P)
        ok = (z > 0.2).all(axis=1)
        idx, sx, sy, z = idx[ok], sx[ok], sy[ok], z[ok]
        if len(idx) == 0:
            return
        a = age[idx] / self.life[idx]
        T = self.temp[idx] * (1.0 - 0.55 * a) * np.exp(-1.2 * a * a)
        tw = 0.65 + 0.35 * np.sin(self.tw[idx] * (t - self.ts[idx]) + self.ph[idx, 0])
        fade_in = np.clip((t - self.ts[idx]) / 0.05, 0, 1)
        fade_out = np.clip((1.0 - a) / 0.15, 0, 1)
        inten = self.I * gain * (T ** 3.0) * tw * fade_in * fade_out
        zc = z.mean(axis=1)
        rad = np.maximum(self.size[idx] * cam.f / zc, 0.5)
        col = bb_vec(T)
        _draw_streaks(img, depth, sx.astype(np.float64), sy.astype(np.float64), zc.astype(np.float64),
                      rad.astype(np.float64), col.astype(np.float64), inten.astype(np.float64), float(zbias),
                      float(max_len_px))


def bb_vec(T):
    T = np.clip(T, 0, 1)
    out = np.zeros((len(T), 3))
    for k in range(4):
        a, b = _BB[k], _BB[k + 1]
        w = np.clip((T - a[0]) / (b[0] - a[0]), 0, 1)[:, None]
        seg = (T >= a[0]) & (T <= b[0] + 1e-9)
        out = np.where(seg[:, None], a[1:] * (1 - w) + b[1:] * w, out)
    return out



@njit(fastmath=True)
def _integrate(p0, v0, ts, life, ph, prm, t, shutter, K, pts, alive, age):
    wx, wy, wz, buoy, uh, drag, turb, oy = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7]
    dt = 1.0 / 120.0
    t_open = t - 0.5 * shutter
    for k in range(p0.shape[0]):
        a_end = t + 0.5 * shutter - ts[k]
        if a_end <= 0.0 or (t_open - ts[k]) > life[k]:
            continue
        alive[k] = True
        age[k] = max(t - ts[k], 0.0)
        x, y, z = p0[k, 0], p0[k, 1], p0[k, 2]
        vx, vy, vz = v0[k, 0], v0[k, 1], v0[k, 2]
        tt = ts[k]
        ks = 0
        # sample times across the shutter
        for s in range(K):
            ts_s = t_open + shutter * s / (K - 1)
            while tt < ts_s:
                h = min(dt, ts_s - tt)
                hy = y - oy
                up = buoy * math.exp(-max(hy, 0.0) / uh)
                ax = drag * (wx - vx) + turb * math.sin(1.7 * y + 2.3 * tt + ph[k, 0]) \
                    + 0.5 * turb * math.sin(3.1 * z + 3.7 * tt + ph[k, 2])
                ay = up - 2.2 + drag * (wy - vy) * 0.5 + 0.4 * turb * math.sin(2.1 * x + 1.3 * tt + ph[k, 1])
                az = drag * (wz - vz) + turb * math.cos(1.9 * x + 2.9 * tt + ph[k, 1])
                vx += ax * h
                vy += ay * h
                vz += az * h
                x += vx * h
                y += vy * h
                z += vz * h
                tt += h
            if ts_s < ts[k]:
                pts[k, s, 0] = p0[k, 0]
                pts[k, s, 1] = p0[k, 1]
                pts[k, s, 2] = p0[k, 2]
            else:
                pts[k, s, 0] = x
                pts[k, s, 1] = y
                pts[k, s, 2] = z


@njit(fastmath=True)
def _draw_streaks(img, depth, sx, sy, z, rad, col, inten, zbias, max_len):
    H, W = img.shape[0], img.shape[1]
    n, K = sx.shape[0], sx.shape[1]
    for k in range(n):
        L = 0.0
        for s in range(K - 1):
            L += math.sqrt((sx[k, s + 1] - sx[k, s]) ** 2 + (sy[k, s + 1] - sy[k, s]) ** 2)
        if L > max_len:
            continue
        r = rad[k]
        # energy conservation: a streak spreads the same energy over its length
        e = inten[k] * (2.0 * r) / (L + 2.0 * r)
        if r < 0.5:
            e *= r / 0.5
            r = 0.5
        cr, cg, cb = col[k, 0] * e, col[k, 1] * e, col[k, 2] * e
        zk = z[k]
        xmin = sx[k, 0]
        xmax = sx[k, 0]
        ymin = sy[k, 0]
        ymax = sy[k, 0]
        for s in range(1, K):
            xmin = min(xmin, sx[k, s])
            xmax = max(xmax, sx[k, s])
            ymin = min(ymin, sy[k, s])
            ymax = max(ymax, sy[k, s])
        x0 = int(max(0.0, math.floor(xmin - r - 1)))
        x1 = int(min(W - 1.0, math.ceil(xmax + r + 1)))
        y0 = int(max(0.0, math.floor(ymin - r - 1)))
        y1 = int(min(H - 1.0, math.ceil(ymax + r + 1)))
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                if depth[py, px] < zk - zbias:
                    continue
                qx = px + 0.5
                qy = py + 0.5
                dmin = 1e9
                for s in range(K - 1):
                    ax = sx[k, s]
                    ay = sy[k, s]
                    bx = sx[k, s + 1] - ax
                    byy = sy[k, s + 1] - ay
                    l2 = bx * bx + byy * byy
                    if l2 < 1e-9:
                        hh = 0.0
                    else:
                        hh = ((qx - ax) * bx + (qy - ay) * byy) / l2
                        hh = min(max(hh, 0.0), 1.0)
                    ddx = qx - ax - bx * hh
                    ddy = qy - ay - byy * hh
                    dd = math.sqrt(ddx * ddx + ddy * ddy)
                    if dd < dmin:
                        dmin = dd
                cov = r + 0.5 - dmin
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                img[py, px, 0] += cr * cov
                img[py, px, 1] += cg * cov
                img[py, px, 2] += cb * cov


# ---------------------------------------------------------------------- smoke ---

class Smoke:
    """Billowing puffs rising from a fire, lit from below by it."""

    def __init__(self, seed, origin, t_start, t_end, rate=7.0, rise=1.6, wind=(0.9, 0.0, 0.0), life=4.5,
                 r0=0.35, growth=0.45, dens=1.0, jitter=0.25):
        rng = np.random.default_rng(seed)
        n = int(max(1, (t_end - t_start + life) * rate))
        self.ts = t_start - life + np.arange(n) / rate + rng.random(n) / rate * 0.8
        self.ts = self.ts[self.ts >= t_start - 0.001]
        n = len(self.ts)
        self.o = np.asarray(origin, np.float64)
        self.off = rng.normal(size=(n, 3)) * jitter
        self.off[:, 1] = np.abs(self.off[:, 1]) * 0.5
        self.rise = rise * (0.7 + 0.6 * rng.random(n))
        self.wind = np.asarray(wind, np.float64)
        self.life = life * (0.7 + 0.6 * rng.random(n))
        self.r0 = r0 * (0.7 + 0.6 * rng.random(n))
        self.growth = growth
        self.dens = dens * (0.6 + 0.8 * rng.random(n))
        self.seed = rng.integers(0, 10000, n)
        self.ph = rng.random(n) * 6.28

    def render(self, img, depth, cam, t, fire_pos, fire_I, amb=(0.004, 0.005, 0.008), albedo=0.5,
               light_scale=1.0, zbias=0.3, strength=1.0):
        age = t - self.ts
        ok = (age > 0) & (age < self.life)
        if not ok.any():
            return
        idx = np.nonzero(ok)[0]
        a = age[idx]
        # rise decelerates, wind shear grows with height
        up = self.rise[idx] * 1.4 * (1 - np.exp(-a / 1.4))
        pos = self.o[None, :] + self.off[idx] * (1 + 0.6 * a[:, None])
        pos[:, 1] += up + 0.25 * a
        shear = 0.35 + 0.65 * np.clip(up / 3.0, 0, 1)
        pos[:, 0] += self.wind[0] * a * shear + 0.18 * np.sin(1.3 * a + self.ph[idx])
        pos[:, 2] += self.wind[2] * a * shear
        rad = self.r0[idx] + self.growth * a ** 0.8
        life = self.life[idx]
        d = self.dens[idx] * np.clip(a / 0.35, 0, 1) * (1 - a / life) ** 1.5 * strength
        sx, sy, z = cam.project(pos)
        good = z > 0.3
        if not good.any():
            return
        order = np.argsort(-z[good])
        sel = np.nonzero(good)[0][order]
        ppm = cam.f / z[sel]
        _smoke(img, depth, sx[sel].astype(np.float64), sy[sel].astype(np.float64), z[sel].astype(np.float64),
               (rad[sel] * ppm).astype(np.float64), d[sel].astype(np.float64), pos[sel].astype(np.float64),
               a[sel].astype(np.float64), self.seed[idx][sel].astype(np.float64),
               np.asarray(fire_pos, np.float64), float(fire_I * light_scale), np.asarray(amb, np.float64),
               float(albedo), float(zbias), rad[sel].astype(np.float64))


@njit(fastmath=True)
def _smoke(img, depth, sx, sy, z, rpx, dens, pos, age, seeds, fp, fI, amb, albedo, zbias, rw):
    H, W = img.shape[0], img.shape[1]
    for k in range(sx.shape[0]):
        R = rpx[k] * 1.35
        x0 = int(max(0, sx[k] - R))
        x1 = int(min(W, sx[k] + R + 1))
        y0 = int(max(0, sy[k] - R))
        y1 = int(min(H, sy[k] + R + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        # light from the fire at the puff centre (inverse square, soft)
        dx = pos[k, 0] - fp[0]
        dy = pos[k, 1] - fp[1]
        dz = pos[k, 2] - fp[2]
        d2 = dx * dx + dy * dy + dz * dz
        Ef = fI / (1.0 + d2 * 0.35)
        sd = seeds[k]
        for py in range(y0, y1):
            for px in range(x0, x1):
                if depth[py, px] < z[k] - zbias:
                    continue
                u = (px + 0.5 - sx[k]) / rpx[k]
                v = (py + 0.5 - sy[k]) / rpx[k]
                q2 = u * u + v * v
                if q2 > 1.8:
                    continue
                nse = fbm3(u * 1.6 + sd, v * 1.6 + age[k] * 0.35, age[k] * 0.25 + sd * 0.1, 4.0, 17)
                dd = dens[k] * math.exp(-q2 * 2.0) * max(0.0, 0.55 + 1.3 * nse)
                if dd <= 1e-4:
                    continue
                tau = dd * 1.6
                a = 1.0 - math.exp(-tau)
                # lit from below: lower half of each puff faces the fire
                under = 0.35 + 0.65 * min(max(0.5 + 0.8 * v, 0.0), 1.0)
                L = albedo * Ef * under
                cr = amb[0] + L * 1.0
                cg = amb[1] + L * 0.45
                cb = amb[2] + L * 0.16
                img[py, px, 0] = img[py, px, 0] * (1 - a) + cr * a
                img[py, px, 1] = img[py, px, 1] * (1 - a) + cg * a
                img[py, px, 2] = img[py, px, 2] * (1 - a) + cb * a
