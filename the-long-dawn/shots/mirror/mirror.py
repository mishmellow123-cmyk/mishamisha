"""C10 THE MIRROR (C bars 27-29, frames 2080-2319 -> renders/mirror_C). The homage to Galadriel's Mirror, from the book.

A hewn stone basin of dark water seen from above (4 deg off vertical, a slow crane push), stars in it. A breath
crosses the water (27 b2-b4) and its ripples carry in the lands burning (EMBERS-C's race, plates/fire.mp4).
On 28 b3 (f2200) a drop falls: for one breath (to f2248) the water shows a golden dawn (RUN-C's C24 illumination,
plates/dawn.jpg, the sun where the drop fell); then the returning ripples carry the fire back over it (to ~2286).
There is NO Eye in the water.

Optics (our own, numba): a dispersive height field (gravity-capillary w(k), exact per-mode propagation, the basin
wall as a reflecting mask) -> per-pixel normals; exact Fresnel; the reflection samples a star sky (gnomonic
texture); the refraction samples the vision on a plane D_V below the surface; the stone rim is a ray-marched
height field lit by the water's own glow, a faint cool key and the night sky. Linear HDR -> look.finish.

    python3 shots/mirror/mirror.py --frames 2080-2319 [--scale 1] [--ss 1.5] [--threads 8] [--out DIR] [--skip]
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
from numba import njit, prange, set_num_threads

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import look  # noqa: E402

F0, F1 = 2080, 2320                 # C10 = bars 27-29
FPS = 24.0
DROP_F = 2200                       # bar 28 b3: the drop strikes the water (barmap_C 'drop')
BACK_F = 2248                       # the golden dawn lasts one breath; the fire returns from here
BACK_END_F = 2288                   # the last gold goes out
BREATH = (2096, 2158)               # the breath crosses the water
R_W = 0.40                          # water radius (m)
WALL, Z_TOP, TOP, BULL = 0.028, 0.034, 0.078, 0.022
R_OUT = R_W + WALL + TOP + BULL     # outer lip
Z_GROUND = -0.95
D_V = 0.55                          # depth of the vision plane below the surface
P_DROP = np.array([0.095, 0.045])   # where the drop falls (off centre: no iris)
N_WATER = 1.333


def ts(f):
    return (f - F0) / FPS


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ camera ---

def camera(f, W, H):
    u = (f - F0) / (F1 - 1 - F0)
    e = u * u * u * (u * (u * 6 - 15) + 10)            # smootherstep push
    e = 0.35 * u + 0.65 * e
    hgt = 0.92 + (0.78 - 0.92) * e
    tx, ty = 0.125 + 0.015 * e, 0.012 + 0.016 * e
    tilt = math.radians(4.0)
    pos = np.array([tx, ty - hgt * math.tan(tilt), hgt])
    fwd = np.array([0.0, math.sin(tilt), -math.cos(tilt)])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    fpx = (H / 2.0) / math.tan(math.radians(27.0) / 2)
    return np.concatenate([pos, fwd, right, up]).astype(np.float64), fpx


# --------------------------------------------------------------- textures ---

def fbm(n, seed, base=4, octaves=7, pers=0.55):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    amp, tot, res = 1.0, 0.0, base
    for _ in range(octaves):
        if res > n // 2:
            break
        g = rng.standard_normal((res, res)).astype(np.float32)
        out += amp * cv2.resize(g, (n, n), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= pers
        res *= 2
    return out / tot


def radius_table(n=720, seed=3):
    """Hand-hewn: the water radius wanders ~0.6 % with angle (periodic)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(seed)
    r = np.ones(n)
    for k in range(2, 9):
        r += rng.normal(0, 0.0045 / k ** 0.8) * np.cos(k * th + rng.uniform(0, 6.28))
    return (R_W * r).astype(np.float64)


def stone_maps(res_m):
    """Height (m), albedo (linear RGB), wet factor over [-L, L]^2 around the basin."""
    L = R_OUT + 0.03
    n = int(2 * L / res_m) // 2 * 2
    xs = (np.arange(n) + 0.5) * (2 * L / n) - L
    X, Y = np.meshgrid(xs, -xs)                        # row 0 = +y (top)
    r = np.hypot(X, Y)
    th = np.mod(np.arctan2(Y, X), 2 * np.pi)
    rt = radius_table()
    rw = np.interp(th, np.linspace(0, 2 * np.pi, len(rt), endpoint=False), rt, period=2 * np.pi)
    rho = r - rw
    chip = fbm(n, 11, base=24, octaves=4)                # chipped outer lip
    lo = fbm(n, 12, base=6, octaves=6)
    mid = fbm(n, 13, base=48, octaves=5)
    hi = fbm(n, 14, base=220, octaves=3)
    z = np.zeros_like(r)
    u = np.clip(rho / WALL, 0, 1)
    z = np.where(rho >= 0, Z_TOP * (1 - (1 - u) ** 2.4), 0.0)
    top0 = WALL
    tt = np.clip((rho - top0) / TOP, 0, 1)
    z = np.where(rho >= top0, Z_TOP + 0.004 * np.sin(np.pi * tt), z)
    b0 = WALL + TOP + 0.004 * chip
    bb = np.clip((rho - b0) / BULL, 0, 1)
    z = np.where(rho >= b0, Z_TOP - BULL * (1 - np.sqrt(np.clip(1 - bb * bb, 0, 1))), z)
    edge = WALL + TOP + BULL + 0.004 * chip
    z = np.where(rho >= edge, Z_TOP - BULL - (rho - edge) * 40.0, z)
    z = np.maximum(z, -0.34)
    stone = (rho >= 0) & (rho < edge + 0.01)
    bump = 0.0011 * lo + 0.0006 * mid + 0.00022 * hi
    pits = np.clip(mid * 2.2 - 1.6, 0, 1) ** 2 * 0.0012
    # two old cracks across the lip
    crack = np.zeros_like(r)
    for a0, s in ((0.7, 1), (3.9, -1)):
        ang = a0 + 0.08 * lo * 6
        d = np.abs(np.mod(th - ang + np.pi, 2 * np.pi) - np.pi) * r
        crack += np.exp(-(d / 0.0007) ** 2) * (rho > 0.004)
    z = np.where(stone, z + bump - pits - 0.0014 * np.clip(crack, 0, 1), z)
    z = np.where(rho < 0, 0.0, z)
    z = np.where(rho >= edge + 0.012, Z_GROUND, z)
    # albedo
    base = np.array([0.20, 0.195, 0.18])[None, None, :] * (1 + 0.25 * lo[..., None] + 0.10 * hi[..., None])
    lich = smooth((fbm(n, 15, base=10, octaves=5) - 0.25) * 3.0)[..., None]
    base = base * (1 - 0.55 * lich) + np.array([0.30, 0.31, 0.22]) * 0.55 * lich
    rust = smooth((fbm(n, 16, base=40, octaves=4) - 0.55) * 5.0)[..., None]
    base = base * (1 - 0.5 * rust) + np.array([0.38, 0.24, 0.08]) * 0.5 * rust
    ao = np.clip(1.0 + 180.0 * (cv2.GaussianBlur(z.astype(np.float32), (0, 0), 6) - z), 0.35, 1.0)
    moss = smooth((1 - ao) * 2.5 + 0.35 * lo - 0.1)[..., None] * (rho[..., None] > 0.006)
    base = base * (1 - 0.7 * moss) + np.array([0.035, 0.05, 0.02]) * 0.7 * moss
    band = smooth(1 - rho / 0.016)[..., None] * (rho[..., None] >= 0)          # water-stained band
    base = base * (1 - 0.55 * band) + np.array([0.06, 0.075, 0.055]) * 0.55 * band
    base = base * ao[..., None]
    wet = (0.05 + 0.5 * band[..., 0]).astype(np.float32)
    par = np.array([-L, L, 2 * L / n, n], np.float64)
    return z.astype(np.float32), base.astype(np.float32), wet, par, rt


def star_sky(n=2048, span=1.25, seed=21):
    """Gnomonic (u, v) = (rx/rz, ry/rz) texture of the zenith sky: stars (a realistic magnitude spread, colours),
    a faint band of the galaxy and a little airglow. Linear radiance, scaled for a ~2 % Fresnel mirror."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n, 3), np.float32)
    ns = 1500
    ct = rng.uniform(math.cos(math.atan(span * 1.42)), 1.0, ns)            # uniform on the cap
    ph = rng.uniform(0, 2 * np.pi, ns)
    tn = np.sqrt(1 - ct * ct) / ct
    su, sv = tn * np.cos(ph), tn * np.sin(ph)
    mag = 6.8 - np.log10(1 + rng.random(ns) * (10 ** (0.36 * 6.8) - 1)) / 0.36
    flux = 10 ** (-0.4 * (mag - 1.0))
    temp = np.clip(rng.normal(6200, 2400, ns), 3000, 14000)
    tcol = np.stack([np.clip(1.0 + (6500 - temp) / 9000, 0.55, 1.3),
                     np.ones(ns), np.clip(1.0 + (temp - 6500) / 9000, 0.5, 1.35)], 1)
    px = (su / span * 0.5 + 0.5) * n
    py = (0.5 - sv / span * 0.5) * n
    sig = 0.85
    for i in range(ns):
        x, y = px[i], py[i]
        if not (3 <= x < n - 3 and 3 <= y < n - 3):
            continue
        x0, y0 = int(x) - 3, int(y) - 3
        gx = np.exp(-((np.arange(7) + x0 + 0.5 - x) ** 2) / (2 * sig * sig))
        gy = np.exp(-((np.arange(7) + y0 + 0.5 - y) ** 2) / (2 * sig * sig))
        img[y0:y0 + 7, x0:x0 + 7] += (gy[:, None] * gx[None, :])[..., None] * (flux[i] * tcol[i])[None, None, :]
    img *= 34.0 / (2 * np.pi * sig * sig)
    # the galaxy's band: a faint diagonal glow with dark lanes
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n - 0.5
    d = (xx * 0.8 - yy * 0.6 + 0.12)
    band = np.exp(-(d / 0.16) ** 2) * (0.6 + 0.8 * fbm(n, 22, base=6, octaves=6))
    lanes = np.clip(1 - 1.6 * np.clip(fbm(n, 23, base=10, octaves=5), 0, 1) * np.exp(-(d / 0.05) ** 2), 0.2, 1)
    img += (band * lanes)[..., None] * np.array([0.9, 0.92, 1.1], np.float32) * 0.9
    img += np.array([0.10, 0.14, 0.28], np.float32) * (1 + 0.6 * (xx * xx + yy * yy))[..., None]
    return img, np.array([span, n], np.float64)


# ------------------------------------------------------------------ the sim ---

class Water:
    """Linear gravity-capillary waves in the basin: exact per-mode propagation (w^2 = (gk + s/r k^3) tanh kh),
    viscous damping, the wall as a reflecting mask; forcing (the breath) and the drop's impulses go into v."""

    def __init__(self, N, rt, seed=5):
        from numpy.fft import rfft2, irfft2       # noqa (scipy below if present)
        try:
            import scipy.fft as sfft
            self.rfft = lambda a: sfft.rfft2(a, workers=-1)
            self.irfft = lambda a: sfft.irfft2(a, s=(N, N), workers=-1)
        except Exception:
            self.rfft, self.irfft = rfft2, (lambda a: irfft2(a, s=(N, N)))
        self.N, self.L = N, 2 * (R_W + 0.03)
        self.dx = self.L / N
        kx = 2 * np.pi * np.fft.fftfreq(N, self.dx)
        ky = 2 * np.pi * np.fft.rfftfreq(N, self.dx)
        K = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        self.K = K
        self.w = np.sqrt((9.81 * K + 7.28e-5 * K ** 3) * np.tanh(K * 0.12))
        self.gam = 0.55 + 2 * 2.4e-6 * K ** 2
        xs = (np.arange(N) + 0.5) * self.dx - self.L / 2
        self.X, self.Y = np.meshgrid(xs, -xs)
        r = np.hypot(self.X, self.Y)
        th = np.mod(np.arctan2(self.Y, self.X), 2 * np.pi)
        rw = np.interp(th, np.linspace(0, 2 * np.pi, len(rt), endpoint=False), rt, period=2 * np.pi)
        self.rho = r - rw
        self.mask = smooth(-self.rho / (2.5 * self.dx))
        self.eta = np.zeros((N, N))
        self.v = np.zeros((N, N))
        self.t = -1.2
        rng = np.random.default_rng(seed)
        # the breath's ripple spectrum: capillary band, aligned with the wind (from the left, a little down)
        self.wind = np.array([math.cos(-0.22), math.sin(-0.22)])
        ang = np.arctan2(np.fft.rfftfreq(N)[None, :] * 0 + ky[None, :], kx[:, None])
        align = np.cos(ang - math.atan2(self.wind[1], self.wind[0]))
        spec = np.exp(-((K - 650.0) / 330.0) ** 2) * np.clip(align, 0, 1) ** 3
        self.nb = spec * np.exp(2j * np.pi * rng.random(K.shape))
        amb = np.exp(-((K - 420.0) / 260.0) ** 2)
        self.na = amb * np.exp(2j * np.pi * rng.random(K.shape))
        self.pending = [(ts(DROP_F), 1.0), (ts(DROP_F) + 0.44, 0.33)]

    def breath_env(self, t):
        a, b = ts(BREATH[0]), ts(BREATH[1])
        s = -0.62 + 1.30 * smooth((t - a) / (b - a))
        x = self.X * self.wind[0] + self.Y * self.wind[1]
        y = -self.X * self.wind[1] + self.Y * self.wind[0]
        on = float(a - 0.2 < t < b + 0.3)
        return on * np.exp(-((x - s) / 0.075) ** 2) * (0.55 + 0.45 * np.cos(9.0 * y + 1.3)), s

    def impulse(self, x, y, amp):
        r2 = (self.X - x) ** 2 + (self.Y - y) ** 2
        s2 = 0.0028 ** 2
        self.v += -amp * 1.2 * (1 - r2 / (2 * s2)) * np.exp(-r2 / (2 * s2))

    def step(self, dt):
        t = self.t
        env, _ = self.breath_env(t)
        if env.any():
            f = self.irfft(self.nb * np.exp(-1j * self.w * t))
            self.v += dt * 8.0 * env * f / (np.abs(f).max() + 1e-9)
        fa = self.irfft(self.na * np.exp(-1j * self.w * t))
        self.v += dt * 0.045 * fa / (np.abs(fa).max() + 1e-9)
        for (tp, amp) in list(self.pending):
            if t >= tp:
                self.impulse(P_DROP[0], P_DROP[1], amp)
                self.pending.remove((tp, amp))
        eh, vh = self.rfft(self.eta), self.rfft(self.v)
        c, s = np.cos(self.w * dt), np.sin(self.w * dt)
        wsafe = np.where(self.w > 1e-9, self.w, 1.0)
        e2 = eh * c + vh * np.where(self.w > 1e-9, s / wsafe, dt)
        v2 = -eh * self.w * s + vh * c
        dmp = np.exp(-self.gam * dt)
        self.eta = self.irfft(e2 * dmp) * self.mask
        self.v = self.irfft(v2 * dmp) * self.mask
        self.t = t + dt

    def advance_to(self, t, sub=4):
        dt = 1.0 / (FPS * sub)
        while self.t < t - 1e-9:
            self.step(dt)

    def fields(self):
        e = self.eta
        gx = (np.roll(e, -1, 1) - np.roll(e, 1, 1)) / (2 * self.dx)
        gy = (np.roll(e, 1, 0) - np.roll(e, -1, 0)) / (2 * self.dx)      # row 0 = +y
        return gx.astype(np.float32), gy.astype(np.float32)


# ----------------------------------------------------------- reveal fields ---

def reveal(sim, f, slope):
    """Where the water shows the fire and where the dawn (0..1 on the sim grid). The ripples carry both edges."""
    t = ts(f)
    X, Y = sim.X, sim.Y
    n1 = sim.noise if hasattr(sim, 'noise') else None
    if n1 is None:
        sim.noise = fbm(sim.N, 31, base=5, octaves=5).astype(np.float64)
        n1 = sim.noise
    rip = np.clip(slope / 0.06, 0, 1.5)
    _, s = sim.breath_env(t)
    x = X * sim.wind[0] + Y * sim.wind[1]
    fire = smooth((s - 0.10 - x + 0.05 * n1 + 0.05 * rip) / 0.16) if t > ts(BREATH[0]) - 0.2 else np.zeros_like(X)
    d = np.hypot(X - P_DROP[0], Y - P_DROP[1])
    td = t - ts(DROP_F)
    dawn = np.zeros_like(X)
    if td > 0:
        rd = 0.70 * (1 - math.exp(-td / 0.42))
        dawn = smooth((rd - d + 0.035 * n1 + 0.04 * rip) / 0.10)
        tb = (f - BACK_F) / (BACK_END_F - BACK_F)
        if tb > 0:
            rb = 0.78 * (1 - smooth(tb) ** 0.9) - 0.05
            dawn = dawn * (1 - smooth((d - rb + 0.04 * n1 + 0.05 * rip) / 0.09))
    fire = fire * (1 - dawn)
    return fire.astype(np.float32), dawn.astype(np.float32)


# ------------------------------------------------------------------ kernel ---

@njit(inline='always', fastmath=True)
def bil(img, x, y, c):
    """bilinear sample of img[..., c] at continuous pixel coords (x right, y down), clamped."""
    h, w = img.shape[0], img.shape[1]
    x = min(max(x - 0.5, 0.0), w - 1.001)
    y = min(max(y - 0.5, 0.0), h - 1.001)
    i, j = int(y), int(x)
    fy, fx = y - i, x - j
    return ((img[i, j, c] * (1 - fx) + img[i, j + 1, c] * fx) * (1 - fy)
            + (img[i + 1, j, c] * (1 - fx) + img[i + 1, j + 1, c] * fx) * fy)


@njit(inline='always', fastmath=True)
def bil1(img, x, y):
    h, w = img.shape[0], img.shape[1]
    x = min(max(x - 0.5, 0.0), w - 1.001)
    y = min(max(y - 0.5, 0.0), h - 1.001)
    i, j = int(y), int(x)
    fy, fx = y - i, x - j
    return ((img[i, j] * (1 - fx) + img[i, j + 1] * fx) * (1 - fy)
            + (img[i + 1, j] * (1 - fx) + img[i + 1, j + 1] * fx) * fy)


@njit(inline='always', fastmath=True)
def fresnel(ci, n):
    ci = min(max(ci, 0.0), 1.0)
    st2 = (1 - ci * ci) / (n * n)
    if st2 >= 1:
        return 1.0
    ct = math.sqrt(1 - st2)
    rs = (ci - n * ct) / (ci + n * ct)
    rp = (n * ci - ct) / (n * ci + ct)
    return 0.5 * (rs * rs + rp * rp)


@njit(fastmath=True)
def shade_stone(x, y, z, hmap, amap, wet, mp, rimE, rt, key, keyc, skyc, dx, dy, dz):
    """Linear radiance of the stone at (x, y, z) seen along direction d."""
    L0, res = mp[0], mp[2]
    px = (x - L0) / res
    py = (mp[1] - y) / res
    e = 1.0
    hx = (bil1(hmap, px + e, py) - bil1(hmap, px - e, py)) / (2 * e * res)
    hy = (bil1(hmap, px, py - e) - bil1(hmap, px, py + e)) / (2 * e * res)
    nx, ny, nz = -hx, -hy, 1.0
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    ar, ag, ab = bil(amap, px, py, 0), bil(amap, px, py, 1), bil(amap, px, py, 2)
    wt = bil1(wet, px, py)
    # the water's glow: from the polar table, weighted by how much this facet faces the water (inward, down)
    r = math.sqrt(x * x + y * y) + 1e-9
    th = math.atan2(y, x)
    if th < 0:
        th += 2 * math.pi
    nth = rimE.shape[0]
    ft = th / (2 * math.pi) * nth
    i0 = int(ft) % nth
    i1 = (i0 + 1) % nth
    w1 = ft - int(ft)
    rw = rt[int(th / (2 * math.pi) * rt.shape[0]) % rt.shape[0]]
    rho = r - rw
    fac = (-(nx * x + ny * y) / r) * 0.85 + nz * 0.25
    fac = max(fac, 0.0) * math.exp(-max(rho, 0.0) / 0.035) * (1.0 - min(max((z - Z_TOP) / 0.01, 0.0), 1.0) * 0.0)
    fac *= min(1.0, max(0.0, 1.2 - rho / 0.11))
    gr = (rimE[i0, 0] * (1 - w1) + rimE[i1, 0] * w1) * fac
    gg = (rimE[i0, 1] * (1 - w1) + rimE[i1, 1] * w1) * fac
    gb = (rimE[i0, 2] * (1 - w1) + rimE[i1, 2] * w1) * fac
    kd = max(nx * key[0] + ny * key[1] + nz * key[2], 0.0)
    sk = 0.5 + 0.5 * nz
    cr = ar * (gr + keyc[0] * kd + skyc[0] * sk)
    cg = ag * (gg + keyc[1] * kd + skyc[1] * sk)
    cb = ab * (gb + keyc[2] * kd + skyc[2] * sk)
    # wet sheen: the glow and the sky in a glossy film near the waterline
    ci = -(dx * nx + dy * ny + dz * nz)
    fr = 0.02 + 0.98 * (1 - ci) ** 5
    cr += wt * fr * (gr * 1.6 + skyc[0] * 0.5)
    cg += wt * fr * (gg * 1.6 + skyc[1] * 0.5)
    cb += wt * fr * (gb * 1.6 + skyc[2] * 0.5)
    return cr, cg, cb


@njit(parallel=True, fastmath=True, cache=True)
def render_kernel(out, kind, cam, fpx, gx, gy, sp, wf, wd, hmap, amap, wet, mp, rimE, rt,
                  fire, fp, dawn, dpar, sky, skp, key, keyc, skyc, deep):
    H, W = out.shape[0], out.shape[1]
    cx, cy, cz = cam[0], cam[1], cam[2]
    for i in prange(H):
        for j in range(W):
            vx = (j + 0.5 - W / 2) / fpx
            vy = -(i + 0.5 - H / 2) / fpx
            dx = cam[3] + vx * cam[6] + vy * cam[9]
            dy = cam[4] + vx * cam[7] + vy * cam[10]
            dz = cam[5] + vx * cam[8] + vy * cam[11]
            dl = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx, dy, dz = dx / dl, dy / dl, dz / dl
            # march the stone height field from above the rim down to the outer wall's foot
            z = 0.045
            t = (z - cz) / dz
            hit = 0
            x, y = cx + t * dx, cy + t * dy
            hz = 0.0
            r0 = math.sqrt(x * x + y * y)
            tb = (-0.345 - cz) / dz
            xb, yb = cx + tb * dx, cy + tb * dy
            rb = math.sqrt(xb * xb + yb * yb)
            if (r0 < R_W - 0.03 and rb < R_W - 0.03):
                hit = 1                                    # straight to the water
                t = -cz / dz
                x, y, z = cx + t * dx, cy + t * dy, 0.0
            elif min(r0, rb) > R_OUT + 0.03 and abs(r0 - rb) < 0.02:
                hit = 3
            else:
                step = 0.0015
                zz = z
                while zz > -0.345:
                    tt = (zz - cz) / dz
                    xx, yy = cx + tt * dx, cy + tt * dy
                    h = bil1(hmap, (xx - mp[0]) / mp[2], (mp[1] - yy) / mp[2])
                    if zz <= h + 1e-6:
                        lo_, hi_ = zz, zz + step
                        for _ in range(7):
                            m = 0.5 * (lo_ + hi_)
                            tm = (m - cz) / dz
                            if m <= bil1(hmap, (cx + tm * dx - mp[0]) / mp[2], (mp[1] - cy - tm * dy) / mp[2]):
                                lo_ = m
                            else:
                                hi_ = m
                        zz = 0.5 * (lo_ + hi_)
                        tt = (zz - cz) / dz
                        x, y, z = cx + tt * dx, cy + tt * dy, zz
                        rr = math.sqrt(x * x + y * y)
                        th = math.atan2(y, x)
                        if th < 0:
                            th += 2 * math.pi
                        rw = rt[int(th / (2 * math.pi) * rt.shape[0]) % rt.shape[0]]
                        hit = 1 if (rr < rw and zz < 0.0015) else 2
                        if hit == 1:
                            t = -cz / dz
                            x, y, z = cx + t * dx, cy + t * dy, 0.0
                        break
                    zz -= step
                if hit == 0:
                    hit = 3
            kind[i, j] = hit
            if hit == 2:
                cr, cg, cb = shade_stone(x, y, z, hmap, amap, wet, mp, rimE, rt, key, keyc, skyc, dx, dy, dz)
                out[i, j, 0], out[i, j, 1], out[i, j, 2] = cr, cg, cb
                continue
            if hit == 3:
                out[i, j, 0] = 0.0
                out[i, j, 1] = 0.0
                out[i, j, 2] = 0.0
                continue
            # ---- water
            sx = (x - sp[0]) / sp[2]
            sy = (sp[1] - y) / sp[2]
            nx = -bil1(gx, sx, sy)
            ny = -bil1(gy, sx, sy)
            nl = math.sqrt(nx * nx + ny * ny + 1.0)
            nx, ny, nz = nx / nl, ny / nl, 1.0 / nl
            ci = -(dx * nx + dy * ny + dz * nz)
            F = fresnel(ci, N_WATER)
            rx, ry, rz = dx + 2 * ci * nx, dy + 2 * ci * ny, dz + 2 * ci * nz
            # reflection: the stone lip near the wall, else the stars
            rr = math.sqrt(x * x + y * y)
            refl_r, refl_g, refl_b = 0.0, 0.0, 0.0
            got = False
            if rr > R_W - 0.035 and rz > 0.05:
                zz = 0.0005
                while zz < 0.042:
                    tt = zz / rz
                    xx, yy = x + tt * rx, y + tt * ry
                    h = bil1(hmap, (xx - mp[0]) / mp[2], (mp[1] - yy) / mp[2])
                    if h >= zz and h > 0.0008:
                        refl_r, refl_g, refl_b = shade_stone(xx, yy, h, hmap, amap, wet, mp, rimE, rt, key, keyc,
                                                             skyc, rx, ry, -rz)
                        got = True
                        break
                    zz += 0.0012
            if not got and rz > 0.05:
                u = rx / rz
                v = ry / rz
                n = skp[1]
                px = (u / skp[0] * 0.5 + 0.5) * n
                py = (0.5 - v / skp[0] * 0.5) * n
                refl_r, refl_g, refl_b = bil(sky, px, py, 0), bil(sky, px, py, 1), bil(sky, px, py, 2)
            # refraction into the vision
            eta = 1.0 / N_WATER
            k = 1 - eta * eta * (1 - ci * ci)
            ct = math.sqrt(max(k, 0.0))
            tx = eta * dx + (eta * ci - ct) * nx
            ty = eta * dy + (eta * ci - ct) * ny
            tz = eta * dz + (eta * ci - ct) * nz
            s = D_V / max(-tz, 0.2)
            qx, qy = x + tx * s, y + ty * s
            a_f = bil1(wf, sx, sy)
            a_d = bil1(wd, sx, sy)
            vr, vg, vb = deep[0], deep[1], deep[2]
            if a_f > 0.001:
                fx_ = ((qx - fp[0]) / fp[2] + 0.5) * fire.shape[1]
                fy_ = (0.5 - (qy - fp[1]) / fp[3]) * fire.shape[0]
                ew = min(fx_ / (0.04 * fire.shape[1]), (fire.shape[1] - fx_) / (0.04 * fire.shape[1]),
                         fy_ / (0.06 * fire.shape[0]), (fire.shape[0] - fy_) / (0.06 * fire.shape[0]), 1.0)
                ew = max(ew, 0.0) * a_f
                vr += bil(fire, fx_, fy_, 0) * ew
                vg += bil(fire, fx_, fy_, 1) * ew
                vb += bil(fire, fx_, fy_, 2) * ew
            if a_d > 0.001:
                ux = ((qx - dpar[0]) / dpar[2] + 0.5) * dawn.shape[1]
                uy = (0.5 - (qy - dpar[1]) / dpar[3]) * dawn.shape[0]
                vr += bil(dawn, ux, uy, 0) * a_d
                vg += bil(dawn, ux, uy, 1) * a_d
                vb += bil(dawn, ux, uy, 2) * a_d
            th = math.atan2(y, x)
            if th < 0:
                th += 2 * math.pi
            rw = rt[int(th / (2 * math.pi) * rt.shape[0]) % rt.shape[0]]
            edge = min(max((rw - rr) / 0.05, 0.0), 1.0)
            edge = 0.15 + 0.85 * edge * edge * (3 - 2 * edge)
            tr = (1 - F) * edge
            if not got:                                    # a bright vision drowns the stars' reflections
                ks = 1.0 / (1.0 + 7.0 * (vr + vg + vb))
                refl_r *= ks
                refl_g *= ks
                refl_b *= ks
            out[i, j, 0] = F * refl_r + tr * vr
            out[i, j, 1] = F * refl_g + tr * vg
            out[i, j, 2] = F * refl_b + tr * vb


# -------------------------------------------------------------- the shot ---

class Mirror:
    def __init__(self, scale=1.0, ss=1.0):
        self.scale, self.ss = scale, ss
        self.W = int(round(1920 * scale * ss))
        self.H = int(round(804 * scale * ss))
        res = 0.0006 / max(scale * ss, 0.35)
        self.hmap, self.amap, self.wet, self.mp, self.rt = stone_maps(max(res, 0.0005))
        n_sim = 768 if scale * ss >= 0.9 else 512
        self.sim = Water(n_sim, self.rt)
        self.sp = np.array([-self.sim.L / 2, self.sim.L / 2, self.sim.dx], np.float64)
        self.sky, self.skp = star_sky(2048 if scale * ss >= 0.9 else 1536)
        cap = cv2.VideoCapture(os.path.join(HERE, 'plates', 'fire.mp4'))
        fr = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            fr.append(im[..., ::-1].copy())
        assert len(fr) == 120, f'fire plate: {len(fr)} frames'
        self.fire_u8 = fr
        d = cv2.imread(os.path.join(HERE, 'plates', 'dawn.jpg'))[..., ::-1]
        d = cv2.GaussianBlur(look.srgb_to_linear(d.astype(np.float32) / 255.0), (0, 0), 1.6)
        h, w = d.shape[:2]
        yy = (np.arange(h, dtype=np.float32) / h)[:, None, None]
        land = np.clip((yy - 0.36) / 0.10, 0, 1)
        lum = d.mean(2, keepdims=True)
        d = lum + 1.6 * (d - lum)
        d = d * (1.0 - 0.68 * land) * np.array([1.15, 0.90, 0.55], np.float32)
        sx, sy = 0.2705 * w, 0.358 * h                     # the sun (RUN-C f2480)
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        rs = np.hypot(X - sx, Y - sy) / w
        d += (np.exp(-(rs / 0.008) ** 2) * 1.6 + np.exp(-(rs / 0.06) ** 2) * 0.3)[..., None] * \
            np.array([1.0, 0.78, 0.45], np.float32)
        self.dawn = np.ascontiguousarray(d * 0.62)
        self.sun_uv = (0.2705, 0.358)
        self.key = np.array([-0.45, 0.35, 0.82]) / np.linalg.norm([-0.45, 0.35, 0.82])
        self.keyc = np.array([0.11, 0.135, 0.20])
        self.skyc = np.array([0.018, 0.024, 0.045])
        self.deep = np.array([0.0012, 0.0019, 0.0028])

    def fire_frame(self, f):
        s = min(max((f - F0) * 0.5, 0.0), 118.999)
        i = int(s)
        a = s - i
        im = self.fire_u8[i].astype(np.float32) * (1 - a) + self.fire_u8[i + 1].astype(np.float32) * a
        lin = look.srgb_to_linear(im / 255.0)
        lin = cv2.GaussianBlur(lin, (0, 0), 1.8)
        grow = 2.6 + 1.4 * smooth((f - 2150) / 150.0)
        return np.ascontiguousarray(lin * grow * np.array([1.05, 0.92, 0.85], np.float32))

    def render(self, f):
        sim = self.sim
        sim.advance_to(ts(f))
        gx, gy = sim.fields()
        slope = np.hypot(gx, gy)
        wf, wd = reveal(sim, f, slope)
        cam, fpx = camera(f, self.W, self.H)
        fire = self.fire_frame(f)
        u = (f - F0) / (F1 - F0)
        fh = 1.02 * (1.0 - 0.10 * u)
        fp = np.array([0.02, 0.03 + 0.03 * u, fh * fire.shape[1] / fire.shape[0], fh], np.float64)
        dh = 1.25
        dw = dh * self.dawn.shape[1] / self.dawn.shape[0]
        # the sun sits under the drop (seen through a flat surface from the drop-time camera)
        qx, qy = self.drop_q()
        dpar = np.array([qx - (self.sun_uv[0] - 0.5) * dw, qy + (self.sun_uv[1] - 0.5) * dh, dw, dh], np.float64)
        rimE = self.rim_glow(wf, wd, fire, fp, dpar)
        out = np.zeros((self.H, self.W, 3), np.float32)
        kind = np.zeros((self.H, self.W), np.uint8)
        render_kernel(out, kind, cam, fpx, gx, gy, self.sp, wf, wd, self.hmap, self.amap, self.wet, self.mp,
                      rimE, self.rt, fire, fp, self.dawn, dpar, self.sky, self.skp, self.key, self.keyc,
                      self.skyc, self.deep)
        # the ground far below: near-black, out of focus
        g = (kind == 3)
        if g.any():
            gt = self.ground(cam, fpx)
            out[g] = gt[g]
        self.drop_fx(out, f, cam, fpx)
        if self.ss != 1.0:
            out = cv2.resize(out, (int(round(1920 * self.scale)), int(round(804 * self.scale))),
                             interpolation=cv2.INTER_AREA)
        return out

    def drop_q(self):
        cam, fpx = camera(DROP_F, self.W, self.H)
        p = np.array([P_DROP[0], P_DROP[1], 0.0])
        d = p - cam[:3]
        d /= np.linalg.norm(d)
        eta = 1 / N_WATER
        ci = -d[2]
        ct = math.sqrt(1 - eta * eta * (1 - ci * ci))
        t = eta * d + (eta * ci - ct) * np.array([0, 0, 1.0])
        s = D_V / -t[2]
        return p[0] + t[0] * s, p[1] + t[1] * s

    def rim_glow(self, wf, wd, fire, fp, dpar, nth=360):
        """The water's own light on the lip: per angle, the vision's radiance averaged over the water near
        that side (flat surface), plus a share of the whole disc."""
        sim = self.sim
        n = 48
        xs = np.linspace(-R_W, R_W, n)
        X, Y = np.meshgrid(xs, -xs)
        inside = np.hypot(X, Y) < R_W * 0.97
        px = ((X - self.sp[0]) / self.sp[2]).astype(np.float32)
        py = ((self.sp[1] - Y) / self.sp[2]).astype(np.float32)
        a_f = cv2.remap(wf, px, py, cv2.INTER_LINEAR)
        a_d = cv2.remap(wd, px, py, cv2.INTER_LINEAR)
        fx = (((X - fp[0]) / fp[2] + 0.5) * fire.shape[1]).astype(np.float32)
        fy = ((0.5 - (Y - fp[1]) / fp[3]) * fire.shape[0]).astype(np.float32)
        fs = cv2.GaussianBlur(fire, (0, 0), 6)
        Lf = cv2.remap(fs, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        dxm = (((X - dpar[0]) / dpar[2] + 0.5) * self.dawn.shape[1]).astype(np.float32)
        dym = ((0.5 - (Y - dpar[1]) / dpar[3]) * self.dawn.shape[0]).astype(np.float32)
        Ld = cv2.remap(cv2.GaussianBlur(self.dawn, (0, 0), 8), dxm, dym, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
        L = (Lf * a_f[..., None] + Ld * a_d[..., None]) * inside[..., None]
        th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        E = np.zeros((nth, 3), np.float64)
        pts = np.stack([X[inside], Y[inside]], 1)
        Lin = L[inside]
        mean = Lin.mean(0) if len(Lin) else np.zeros(3)
        for k in range(nth):
            e = np.array([math.cos(th[k]), math.sin(th[k])]) * R_W
            d2 = ((pts - e) ** 2).sum(1)
            w = 1.0 / (d2 + 0.012)
            E[k] = (Lin * w[:, None]).sum(0) / w.sum()
        return np.ascontiguousarray(3.0 * E + 1.0 * mean)

    def ground(self, cam, fpx):
        H, W = self.H, self.W
        if not hasattr(self, '_gtex'):
            n = 1024
            t = 0.5 + 0.5 * fbm(n, 41, base=8, octaves=6)
            leaves = np.clip(fbm(n, 42, base=200, octaves=2) * 3 - 2.4, 0, 1)
            col = (np.array([0.024, 0.028, 0.02]) * t[..., None] + np.array([0.05, 0.036, 0.012]) * leaves[..., None])
            self._gtex = col.astype(np.float32)
        jj, ii = np.meshgrid(np.arange(W) + 0.5 - W / 2, -(np.arange(H) + 0.5 - H / 2))
        vx, vy = jj / fpx, ii / fpx
        d = cam[3:6][None, None, :] + vx[..., None] * cam[6:9] + vy[..., None] * cam[9:12]
        t = (Z_GROUND - cam[2]) / d[..., 2]
        gx = cam[0] + t * d[..., 0]
        gy = cam[1] + t * d[..., 1]
        u = ((gx / 3.0 + 0.5) * 1024).astype(np.float32)
        v = ((0.5 - gy / 3.0) * 1024).astype(np.float32)
        g = cv2.remap(self._gtex, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return cv2.GaussianBlur(g, (0, 0), 6 * self.scale * self.ss)

    def drop_fx(self, out, f, cam, fpx):
        """The falling drop's glint and its rising reflection meeting at the strike (f2200), then the splash's
        sparkles (the crown, the jet) lit by the new dawn."""
        H, W = out.shape[:2]

        def proj(p):
            d = np.asarray(p) - cam[:3]
            zc = d @ cam[3:6]
            return (W / 2 + fpx * (d @ cam[6:9]) / zc, H / 2 - fpx * (d @ cam[9:12]) / zc)

        def seg(p0, p1, rad, col):
            x0, y0 = p0
            x1, y1 = p1
            n = int(max(abs(x1 - x0), abs(y1 - y0)) / max(rad * 0.5, 0.5)) + 2
            r = int(rad * 3) + 2
            for k in range(n):
                a = k / (n - 1)
                x, y = x0 + (x1 - x0) * a, y0 + (y1 - y0) * a
                xi, yi = int(x), int(y)
                if not (r <= xi < W - r and r <= yi < H - r):
                    continue
                yy, xx = np.mgrid[yi - r:yi + r + 1, xi - r:xi + r + 1]
                g = np.exp(-((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) / (2 * rad * rad))
                out[yi - r:yi + r + 1, xi - r:xi + r + 1] += g[..., None] * (np.asarray(col) / n)

        tt = ts(f) - ts(DROP_F)
        g = 9.81
        v0 = 6.2                                     # impact speed (m/s)
        px, py = P_DROP
        sc = self.scale * self.ss
        for k in range(2):                           # the drop and its reflection
            t0, t1 = tt - 1 / 48.0, tt                # 180-degree shutter
            if t1 > 0:
                t1 = 0.0
            if t0 >= 0 or t1 <= t0:
                continue
            z0 = -v0 * t0 - 0.5 * g * t0 * t0
            z1 = -v0 * t1 - 0.5 * g * t1 * t1
            if z0 > 0.9:
                continue
            s = 1 if k == 0 else -1
            a, b = proj((px, py, s * z0)), proj((px, py, s * z1))
            col = (np.array([4.0, 3.1, 2.0]) if k == 0 else np.array([0.35, 0.28, 0.18])) * 60 * sc
            seg(a, b, 1.3 * sc, col)
        if 0 <= tt < 0.34:                           # the crown's droplets, then the jet's bead
            rng = np.random.default_rng(int(f))
            c = proj((px, py, 0.0))
            k = np.exp(-tt / 0.08)
            for m in range(14):
                ang = rng.uniform(0, 2 * np.pi)
                rr = (0.004 + 0.05 * tt) * rng.uniform(0.7, 1.1)
                p = proj((px + rr * math.cos(ang), py + rr * math.sin(ang), 0.004))
                seg(p, p, 0.9 * sc, np.array([3.0, 2.3, 1.2]) * 9 * k * sc)
            jet = max(0.0, math.sin(min(tt / 0.30, 1.0) * math.pi))
            p = proj((px, py, 0.02 * jet))
            seg(p, p, 1.6 * sc, np.array([3.5, 2.8, 1.6]) * 14 * jet * sc)


def frames_arg(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'renders', 'mirror_C'))
    ap.add_argument('--exposure', type=float, default=1.35)
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    if a.threads:
        set_num_threads(a.threads)
        cv2.setNumThreads(a.threads)
    frames = frames_arg(a.frames)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    m = Mirror(a.scale, a.ss)
    for f in frames:
        if a.skip and os.path.exists(look.find_frame(out_dir, f)):
            continue
        t0 = time.time()
        hdr = m.render(f)
        img = look.finish(hdr, exposure=a.exposure, bloom_strength=0.07, bloom_threshold=0.9,
                          vignette_amount=0.28, lift=0.003)
        look.save_png(look.frame_path(out_dir, f), img)
        print(f'mirror {f} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
