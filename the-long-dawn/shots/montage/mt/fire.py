"""Fire: tongue-based HDR bonfire flame, ignition envelope, deterministic sparks + embers
(motion-blurred, hot head / tapered tail), billowing smoke lit from below, heat shimmer, and small
distant beacon glows.

Everything is a pure function of time (no frame-to-frame state) so any frame renders alone.
"""
import math

import cv2
import numpy as np
from numba import njit, prange

from .noise import fbm3, gnoise3, smoothstep

# look.blackbody stops (t, r, g, b) -- identical so all departments share the ramp
_BB = np.array([[0.00, 0.35, 0.02, 0.00],
                [0.30, 0.90, 0.12, 0.01],
                [0.55, 1.00, 0.36, 0.04],
                [0.78, 1.00, 0.68, 0.25],
                [1.00, 1.00, 0.95, 0.85]])

FIRE_LIGHT = np.array([1.0, 0.50, 0.17])   # colour of firelight on surroundings (linear)


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


def bb_vec(T):
    T = np.clip(np.asarray(T, np.float64), 0, 1)
    out = np.zeros((len(T), 3))
    for k in range(4):
        a, b = _BB[k], _BB[k + 1]
        w = np.clip((T - a[0]) / (b[0] - a[0]), 0, 1)[:, None]
        seg = (T >= a[0]) & (T <= b[0] + 1e-9)
        out = np.where(seg[:, None], a[1:] * (1 - w) + b[1:] * w, out)
    return out


# ------------------------------------------------------------------ envelope ---

def ignite_env(t, t0, fps=24.0):
    """Ignition envelope at time t (s) for ignition t0 (s): (size, intensity, light) multipliers.
    A one-frame 'catch', then a whoomp that overshoots (~1.4x at +5 frames) and settles to a roar."""
    u = (t - t0) * fps   # frames since ignition
    if u < -1.5:
        return 0.0, 0.0, 0.0
    if u < 0.0:
        k = (u + 1.5) / 1.5
        return 0.25 * k, 0.9 * k, 0.3 * k
    rise = 1.0 - math.exp(-u / 1.4)
    over = 0.42 * math.exp(-((u - 5.0) / 4.0) ** 2) + 0.08 * math.exp(-((u - 13.0) / 4.0) ** 2)
    size = 0.5 + 0.5 * rise + over
    inten = 1.0 + 0.9 * math.exp(-u / 4.0)
    light = 1.0 + 1.2 * math.exp(-u / 5.0)
    return size, inten, light


def flicker(t, seed=0, amt=1.0):
    """Organic flicker multiplier ~1 (for the light a fire casts)."""
    s = seed * 1.7
    v = (0.07 * math.sin(2 * math.pi * 1.9 * t + s) + 0.05 * math.sin(2 * math.pi * 3.7 * t + 2.1 + s)
         + 0.035 * math.sin(2 * math.pi * 7.3 * t + 0.7 * s) + 0.02 * math.sin(2 * math.pi * 11.1 * t + s))
    return 1.0 + amt * v


def tongue_table(n, seed, spread=0.66):
    """Per-tongue constants: base x (Rb units), height factor, width factor, cycle period (s),
    phase, sway phase."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-spread, spread, n) + rng.normal(size=n) * 0.06
    centre = np.clip(1.0 - np.abs(xs) / max(spread, 1e-3), 0.0, 1.0)
    hf = 0.55 + 0.45 * centre ** 0.8 + rng.normal(size=n) * 0.05
    wf = 0.34 + 0.16 * centre + rng.random(n) * 0.06
    per = rng.uniform(0.42, 0.68, n)
    ph = rng.random(n)
    sw = rng.random(n) * 6.283
    return np.stack([xs, hf, wf, per, ph, sw], 1).astype(np.float64)


# ---------------------------------------------------------------------- flame ---

@njit(parallel=True, fastmath=True, cache=True)
def _bonfire(img, depth, bx, by, ppm, zf, Hf, Rb, lean, t, seed, I, x0, x1, y0, y1, zbias, TG, warp,
             debug):
    rise = 1.4 * math.sqrt(max(Hf, 0.05)) + 0.5          # m/s, upward advection of structure
    K = TG.shape[0]
    for py in prange(y0, y1):
        for px in range(x0, x1):
            if depth[py, px] < zf - zbias:
                continue
            X = (px + 0.5 - bx) / ppm
            Y = (by - (py + 0.5)) / ppm
            v = Y / Hf
            if v < -0.08 or v > 1.9:
                continue
            vc = max(v, 0.0)
            # --- multi-scale domain warp, growing with height (billow + finer curl)
            wa = fbm3(X / Rb * 0.7, (Y - rise * t) / Rb * 0.33, t * 0.45 + seed, 3.0, seed)
            wb = fbm3(X / Rb * 1.9 + 3.1, (Y - 1.3 * rise * t) / Rb * 0.8, t * 0.9 + seed, 3.0, seed + 7)
            Xw = X - lean * vc * vc - warp * Rb * (wa * (0.06 + 0.95 * vc) + 0.35 * wb * (0.04 + 0.7 * vc))
            Yw = Y + 0.12 * Rb * wb
            # --- tongues (each grows, then its tip tears off as a rising flamelet)
            dsum = 0.0
            for k in range(K):
                cyc = t / TG[k, 3] + TG[k, 4]
                ph = cyc - math.floor(cyc)
                hmax = Hf * TG[k, 1]
                grow = 0.6 + 0.5 * smoothstep(0.0, 0.72, ph)
                hk = hmax * grow
                xk = TG[k, 0] * Rb + 0.10 * Rb * math.sin(2.7 * t + TG[k, 5]) * (0.3 + vc)
                # tongues lean toward the centre as they rise (flames converge)
                xk *= 1.0 - 0.55 * min(max(Yw, 0.0) / Hf, 1.0)
                wk = Rb * TG[k, 2]
                dk = 0.0
                if Yw > -0.05 * Hf and Yw < hk:
                    s = max(Yw, 0.0) / hk
                    hw = wk * (1.0 - s) ** 0.62 * (1.0 + 0.45 * (1.0 - s) ** 3)
                    dk = 1.0 - abs(Xw - xk) / (hw + 1e-6)
                if ph > 0.7:
                    u = (ph - 0.7) / 0.3
                    ytop = hmax * 1.1
                    yb = ytop * 0.86 + u * TG[k, 3] * rise * 1.35
                    rb = wk * 0.55 * (1.0 - 0.75 * u) + 1e-4
                    ex = (Xw - xk * 0.5) / rb
                    ey = (Yw - yb) / (rb * 2.4)
                    blob = (1.0 - math.sqrt(ex * ex + ey * ey)) * (1.0 - u * u)
                    if blob > dk:
                        dk = blob
                if dk > 0.0:
                    dsum += dk * dk
            if dsum <= 0.0:
                continue
            dens = math.sqrt(dsum)
            # --- ragged edges: erosion by finer upward-advected noise
            nd = fbm3(Xw / Rb * 3.2, (Yw - 1.5 * rise * t) / Rb * 1.3, t * 1.3 + seed * 2.0, 3.0, seed + 29)
            F = dens - (0.5 - 0.5 * nd) * (0.22 + 0.55 * vc)
            F *= smoothstep(-0.06, 0.04, v)
            if F <= 0.0:
                continue
            # --- temperature: hot white-yellow core low, orange body, dark red broken tips
            core = math.exp(-(X / (0.62 * Rb)) ** 2) * math.exp(-((v - 0.1) / 0.24) ** 2)
            T = min(F * 1.6, 1.0) ** 0.75 * (0.86 - 0.36 * min(vc, 1.2)) + 0.26 * core * min(F * 4.0, 1.0)
            # internal brightness flicker (patches brighten/darken over time)
            fl = fbm3(X / Rb * 1.2, (Y - rise * t) / Rb * 0.6, t * 3.0 + seed, 2.0, seed + 41)
            T *= 0.9 + 0.22 * fl
            if T > 1.0:
                T = 1.0
            if T <= 0.0:
                continue
            if debug > 0:
                img[py, px, 0] += T
                img[py, px, 1] += T
                img[py, px, 2] += T
                continue
            r_, g_, b_ = bb(T)
            e = I * T ** 4.2
            img[py, px, 0] += r_ * e
            img[py, px, 1] += g_ * e
            img[py, px, 2] += b_ * e


def flame(img, depth, cam, base_world, Hf, Rb, t, seed=0, I=24.0, lean=0.0, zbias=0.5, tongues=5,
          warp=1.0, min_px=2.0, debug=0, TG=None):
    """Bonfire flame with base centre at world point base_world. Hf = flame height, Rb = base half-width
    (both metres). I = emission of the hottest core."""
    if Hf <= 0.01 or I <= 0:
        return
    sx, sy, z = cam.project(np.asarray(base_world, np.float64))
    if z <= 0.1:
        return
    ppm = cam.f / z
    H, W = img.shape[:2]
    if Hf * ppm < min_px * 4:
        # too small for structure: a flickering hot point
        e = I * 0.12 * (2 * Rb * ppm) * (Hf * ppm)
        glow(img, depth, sx, sy - 0.35 * Hf * ppm, max(0.35 * Hf * ppm, 0.7), e, z, zbias)
        return
    if TG is None:
        TG = tongue_table(tongues, seed)
    half = (Rb * 1.5 + abs(lean) * 1.3 + 0.25 * Hf) * ppm
    x0 = int(max(0, sx - half - 2))
    x1 = int(min(W, sx + half + 2))
    y0 = int(max(0, sy - Hf * 1.9 * ppm - 2))
    y1 = int(min(H, sy + 0.1 * Hf * ppm + 2))
    if x1 <= x0 or y1 <= y0:
        return
    _bonfire(img, depth, float(sx), float(sy), float(ppm), float(z), float(Hf), float(Rb), float(lean),
             float(t), float(seed), float(I), x0, x1, y0, y1, float(zbias), TG, float(warp), int(debug))


def shimmer(img, cam, base_world, Hf, Rb, t, amp_px=1.2, seed=0):
    """Heat haze: displace what is behind/above the flame (call before drawing the flame)."""
    sx, sy, z = cam.project(np.asarray(base_world, np.float64))
    if z <= 0.1:
        return
    ppm = cam.f / z
    H, W = img.shape[:2]
    x0 = int(max(0, sx - 1.6 * Rb * ppm - 0.3 * Hf * ppm))
    x1 = int(min(W, sx + 1.6 * Rb * ppm + 0.3 * Hf * ppm))
    y0 = int(max(0, sy - 3.0 * Hf * ppm))
    y1 = int(min(H, sy - 0.2 * Hf * ppm))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u = (xx - sx) / (Rb * ppm * 1.6 + 1e-6)
    v = (sy - yy) / (Hf * ppm)
    fade = np.clip(1 - np.abs(u), 0, 1) ** 1.5 * np.clip((v - 0.2) / 0.5, 0, 1) * np.clip((3.0 - v) / 1.2, 0, 1)
    sc = 0.06 * ppm
    k = 1.0 / max(sc, 1e-3)
    tt = t * 3.0
    dx = np.sin(xx * 0.9 * k + (yy + t * 2.2 * ppm) * 1.7 * k + seed) * 0.6 + np.sin((yy + t * 2.9 * ppm) * 2.9 * k + tt) * 0.4
    dy = np.sin((yy + t * 2.4 * ppm) * 2.1 * k + xx * 0.7 * k + 1.3 + seed) * 0.5
    mx = (xx + dx * amp_px * fade).astype(np.float32)
    my = (yy + dy * amp_px * fade).astype(np.float32)
    img[y0:y1, x0:x1] = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@njit(fastmath=True, cache=True)
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
        col = np.array([1.0, 0.60, 0.24])
    c = np.asarray(col) * energy
    _glow(img, depth, float(sx), float(sy), float(max(rad_px, 0.6)), float(c[0]), float(c[1]), float(c[2]),
          float(z), float(zbias))


@njit(parallel=True, fastmath=True, cache=True)
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
    """Deterministic sparks + slow embers. Each frame integrates every live particle from its birth
    (fixed 1/120 s steps), so frames are independent.

    Sparks: fast, short-lived, hot (white-yellow) cooling to orange -> deep red -> gone.
    Embers: slow, long-lived, meander upward on curl noise.
    Brightness distribution is steep: most are tiny and dim, a few are bright hero sparks."""

    def __init__(self, seed, origin, t_ign, t_end, burst=260, rate=45.0, ember_rate=10.0, speed=(2.0, 6.0),
                 burst_speed=(5.0, 13.0), spread=0.45, life=(0.5, 1.6), ember_life=(2.0, 4.0),
                 wind=(0.8, 0.0, 0.0), buoy=5.5, updraft_h=2.5, drag=1.3, radius=0.45, I=5.0, turb=1.6,
                 curl=1.4):
        rng = np.random.default_rng(seed)
        nb = int(burst)
        nc = int(max(0.0, t_end - t_ign) * rate) + 1
        ne = int(max(0.0, t_end - t_ign) * ember_rate) + 1
        n = nb + nc + ne
        ts = np.empty(n)
        kind = np.zeros(n, np.int64)          # 0 spark, 1 ember
        ts[:nb] = t_ign + rng.random(nb) ** 2 * 0.15
        ts[nb:nb + nc] = t_ign + 0.06 + np.sort(rng.random(nc)) * (t_end - t_ign)
        ts[nb + nc:] = t_ign + 0.1 + np.sort(rng.random(ne)) * (t_end - t_ign)
        kind[nb + nc:] = 1
        ang = rng.random(n) * 2 * np.pi
        rr = np.sqrt(rng.random(n)) * radius
        p0 = np.stack([origin[0] + rr * np.cos(ang), origin[1] + rng.random(n) * 0.5,
                       origin[2] + rr * np.sin(ang)], 1)
        sp = rng.uniform(speed[0], speed[1], n)
        sp[:nb] = rng.uniform(burst_speed[0], burst_speed[1], nb)
        sp[nb + nc:] = rng.uniform(0.3, 1.2, ne)
        th = rng.normal(size=n) * spread
        th[:nb] *= 1.25
        ph = rng.random(n) * 2 * np.pi
        v0 = np.stack([sp * np.sin(th) * np.cos(ph), sp * np.cos(th), sp * np.sin(th) * np.sin(ph)], 1)
        lf = rng.uniform(life[0], life[1], n)
        lf[nb + nc:] = rng.uniform(ember_life[0], ember_life[1], ne)
        u = rng.random(n)
        lum = 0.12 + 0.88 * u ** 5
        hero = rng.random(n) < 0.035
        lum[hero] *= 2.2
        lum[nb + nc:] *= 0.55
        T0 = rng.uniform(0.86, 1.0, n)
        T0[nb + nc:] = rng.uniform(0.6, 0.78, ne)
        self.p0, self.v0, self.ts, self.life = p0, v0, ts, lf
        self.kind = kind
        self.lum = lum
        self.T0 = T0
        self.wid = np.clip(0.3 + 0.45 * u ** 3 + 0.25 * hero, 0.3, 0.75)   # half-width px at 1920
        self.ph = rng.random((n, 3)) * 2 * np.pi
        self.tw = rng.uniform(5.0, 18.0, n)
        self.params = np.array([wind[0], wind[1], wind[2], buoy, updraft_h, drag, turb, origin[1], curl,
                                origin[0], origin[2]], np.float64)
        self.I = I
        self.n = n

    def state(self, t, shutter, K):
        pts = np.zeros((self.n, K, 3))
        alive = np.zeros(self.n, np.bool_)
        age = np.zeros(self.n)
        _integrate(self.p0, self.v0, self.ts, self.life, self.ph, self.kind, self.params, t, shutter, K, pts,
                   alive, age)
        return pts, alive, age

    def render(self, img, depth, cam, t, gain=1.0, shutter=1.0 / 48.0, zbias=0.3, max_len_px=260.0, K=5,
               res_scale=None):
        pts, alive, age = self.state(t, shutter, K)
        if not alive.any():
            return
        idx = np.nonzero(alive)[0]
        P = pts[idx]
        sx, sy, z = cam.project(P)
        ok = (z > 0.2).all(axis=1)
        idx, sx, sy, z = idx[ok], sx[ok], sy[ok], z[ok]
        if len(idx) == 0:
            return
        a = np.clip(age[idx] / self.life[idx], 0, 1)
        T = self.T0[idx] * (1.0 - 0.72 * a ** 0.9)
        tw = 0.7 + 0.3 * np.sin(self.tw[idx] * (t - self.ts[idx]) + self.ph[idx, 0])
        fade_in = np.clip((t - self.ts[idx]) / 0.04, 0, 1)
        fade_out = np.clip((1.0 - a) / 0.2, 0, 1)
        inten = self.I * gain * self.lum[idx] * (T / 0.9) ** 4.0 * tw * fade_in * fade_out
        zc = z.mean(axis=1)
        s = cam.W / 1920.0 if res_scale is None else res_scale
        rad = np.maximum(self.wid[idx] * s, 0.3)
        col = bb_vec(T)
        _draw_streaks(img, depth, sx.astype(np.float64), sy.astype(np.float64), zc.astype(np.float64),
                      rad.astype(np.float64), col.astype(np.float64), inten.astype(np.float64), float(zbias),
                      float(max_len_px * s))


@njit(fastmath=True, cache=True)
def _integrate(p0, v0, ts, life, ph, kind, prm, t, shutter, K, pts, alive, age):
    wx, wy, wz, buoy, uh, drag, turb, oy, curl = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7], prm[8]
    ox, oz = prm[9], prm[10]
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
        emb = kind[k] == 1
        for s in range(K):
            ts_s = t_open + shutter * s / (K - 1)
            while tt < ts_s:
                h = min(dt, ts_s - tt)
                hy = y - oy
                rx = x - ox
                rz = z - oz
                plume = math.exp(-(rx * rx + rz * rz) / (0.8 + 0.25 * max(hy, 0.0)))
                up = buoy * math.exp(-max(hy, 0.0) / uh) * (0.35 + 0.65 * plume)
                ax = drag * (wx - vx) + turb * math.sin(1.7 * y + 2.3 * tt + ph[k, 0])
                ay = up - 2.0 + drag * (wy - vy) * 0.4
                az = drag * (wz - vz) + turb * math.cos(1.9 * x + 2.9 * tt + ph[k, 1])
                if emb:
                    e = 0.15
                    q = 0.9
                    p_y1 = gnoise3(x * q, (y + e) * q, tt * 0.35 + ph[k, 2], 5)
                    p_y0 = gnoise3(x * q, (y - e) * q, tt * 0.35 + ph[k, 2], 5)
                    p_x1 = gnoise3((x + e) * q, y * q, tt * 0.35 + ph[k, 2], 5)
                    p_x0 = gnoise3((x - e) * q, y * q, tt * 0.35 + ph[k, 2], 5)
                    cx_ = (p_y1 - p_y0) / (2 * e)
                    cy_ = -(p_x1 - p_x0) / (2 * e)
                    ax += curl * 3.0 * (cx_ - 0.3 * vx)
                    ay += curl * 1.5 * cy_ + 1.6
                    az += curl * 2.0 * math.sin(1.3 * y + 1.1 * tt + ph[k, 2]) - 0.5 * vz
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


@njit(fastmath=True, cache=True)
def _draw_streaks(img, depth, sx, sy, z, rad, col, inten, zbias, max_len):
    """Polyline streak per particle: sample 0 = tail (shutter open), K-1 = head.
    Hot bright head, tapered fading tail; energy conserved along the length."""
    H, W = img.shape[0], img.shape[1]
    n, K = sx.shape[0], sx.shape[1]
    for k in range(n):
        L = 0.0
        for s in range(K - 1):
            L += math.sqrt((sx[k, s + 1] - sx[k, s]) ** 2 + (sy[k, s + 1] - sy[k, s]) ** 2)
        if L > max_len:
            continue
        r = rad[k]
        e = inten[k] * (2.0 * r + 0.6) / (L + 2.0 * r + 0.6)
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
                umin = 0.0
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
                        umin = (s + hh) / (K - 1)
                rr = r * (0.45 + 0.55 * umin)
                cov = rr + 0.5 - dmin
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                w = cov * (0.12 + 0.88 * umin ** 1.8) * 1.6
                img[py, px, 0] += cr * w
                img[py, px, 1] += cg * w
                img[py, px, 2] += cb * w


# ---------------------------------------------------------------------- smoke ---

class Smoke:
    """Dark billowing puffs rising from a fire; lit orange from beneath only near the fire."""

    def __init__(self, seed, origin, t_start, t_end, rate=5.0, rise=1.5, wind=(0.9, 0.0, 0.0), life=5.0,
                 r0=0.3, growth=0.42, dens=0.8, jitter=0.18):
        rng = np.random.default_rng(seed)
        n = int(max(1, (t_end - t_start + 0.5) * rate))
        self.ts = t_start + np.arange(n) / rate + rng.random(n) / rate * 0.8
        self.o = np.asarray(origin, np.float64)
        self.off = rng.normal(size=(n, 3)) * jitter
        self.off[:, 1] = np.abs(self.off[:, 1]) * 0.5
        self.rise = rise * (0.7 + 0.6 * rng.random(n))
        self.wind = np.asarray(wind, np.float64)
        self.life = life * (0.7 + 0.6 * rng.random(n))
        self.r0 = r0 * (0.7 + 0.6 * rng.random(n))
        self.growth = growth
        self.dens = dens * (0.5 + 0.9 * rng.random(n))
        self.seed = rng.integers(0, 10000, n)
        self.ph = rng.random(n) * 6.28
        self.spin = rng.normal(size=n) * 0.4

    def render(self, img, depth, cam, t, fire_pos, fire_I, amb=(0.003, 0.0035, 0.005), albedo=0.35,
               zbias=0.3, strength=1.0, light_falloff=0.9):
        age = t - self.ts
        ok = (age > 0) & (age < self.life)
        if not ok.any():
            return
        idx = np.nonzero(ok)[0]
        a = age[idx]
        up = self.rise[idx] * 1.6 * (1 - np.exp(-a / 1.6))
        pos = self.o[None, :] + self.off[idx] * (1 + 0.8 * a[:, None])
        pos[:, 1] += up + 0.3 * a
        shear = 0.3 + 0.7 * np.clip(up / 3.0, 0, 1)
        pos[:, 0] += self.wind[0] * a * shear + 0.15 * np.sin(1.3 * a + self.ph[idx])
        pos[:, 2] += self.wind[2] * a * shear
        rad = self.r0[idx] + self.growth * a ** 0.85
        life = self.life[idx]
        d = self.dens[idx] * np.clip(a / 0.4, 0, 1) * (1 - a / life) ** 1.6 * strength
        sx, sy, z = cam.project(pos)
        good = z > 0.3
        if not good.any():
            return
        sel = np.nonzero(good)[0][np.argsort(-z[good])]
        ppm = cam.f / z[sel]
        _smoke(img, depth, sx[sel].astype(np.float64), sy[sel].astype(np.float64), z[sel].astype(np.float64),
               (rad[sel] * ppm).astype(np.float64), d[sel].astype(np.float64), pos[sel].astype(np.float64),
               a[sel].astype(np.float64), self.seed[idx][sel].astype(np.float64),
               self.spin[idx][sel].astype(np.float64), np.asarray(fire_pos, np.float64), float(fire_I),
               np.asarray(amb, np.float64), float(albedo), float(zbias), float(light_falloff),
               rad[sel].astype(np.float64))


@njit(fastmath=True, cache=True)
def _smoke(img, depth, sx, sy, z, rpx, dens, pos, age, seeds, spin, fp, fI, amb, albedo, zbias, lfall, rw):
    H, W = img.shape[0], img.shape[1]
    for k in range(sx.shape[0]):
        R = rpx[k] * 1.5
        x0 = int(max(0, sx[k] - R))
        x1 = int(min(W, sx[k] + R + 1))
        y0 = int(max(0, sy[k] - R))
        y1 = int(min(H, sy[k] + R + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        sd = seeds[k]
        ca = math.cos(spin[k] * age[k])
        sa = math.sin(spin[k] * age[k])
        for py in range(y0, y1):
            for px in range(x0, x1):
                if depth[py, px] < z[k] - zbias:
                    continue
                u0 = (px + 0.5 - sx[k]) / rpx[k]
                v0 = (py + 0.5 - sy[k]) / rpx[k]
                q = math.sqrt(u0 * u0 + v0 * v0)
                if q > 1.5:
                    continue
                u = ca * u0 - sa * v0
                v = sa * u0 + ca * v0
                nb = fbm3(u * 1.3 + sd, v * 1.3, age[k] * 0.3 + sd * 0.01, 4.0, 17)
                edge = 0.78 + 0.5 * nb
                dd = dens[k] * smoothstep(edge, edge * 0.35, q)
                if dd <= 1e-4:
                    continue
                nd = fbm3(u * 2.7 - sd, v * 2.7, age[k] * 0.4, 3.0, 23)
                dd *= 0.55 + 0.9 * max(nd + 0.3, 0.0)
                a = 1.0 - math.exp(-dd * 1.8)
                wy = pos[k, 1] - v0 * rw[k]
                wx = pos[k, 0] + u0 * rw[k]
                dx = wx - fp[0]
                dy = wy - fp[1]
                dz = pos[k, 2] - fp[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                under = 0.4 + 0.6 * min(max(0.5 + 0.7 * v0, 0.0), 1.0)
                E = fI / (1.0 + (dist / lfall) ** 3) * under
                L = albedo * E
                cr = amb[0] + L * 1.0
                cg = amb[1] + L * 0.42
                cb = amb[2] + L * 0.13
                img[py, px, 0] = img[py, px, 0] * (1 - a) + cr * a
                img[py, px, 1] = img[py, px, 1] * (1 - a) + cg * a
                img[py, px, 2] = img[py, px, 2] * (1 - a) + cb * a
