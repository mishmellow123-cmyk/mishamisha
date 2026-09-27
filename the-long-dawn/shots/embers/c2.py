"""EMBERS-C2: three of cut C's embers shots, in C's own frame numbers (BIBLE_V3 "REVISION 1 . LOCKED BEAT SHEETS",
cut C, with the DIRECTOR'S H5 CALLS). Output: render.py FRAMES --cut C3 -> renders/embers_C3 (render.py routes these
frames here; every other C3 frame stays EMBERS-C's c3.TimelineC3).

    C9   1920-2080  E12  THE EYE ONTO NOTHING        c_eye.EyeShot
    C11  2320-2480  E8-C THE GRASP THAT CANNOT HOLD  c_grasp.GraspShot
    C13a 2720-2840  E13a THE RING FALLS (part a)     c_fall.FallShot   (RUN-C owns part b from 2840)

Shared here: the bar grid, the router, and the 2.5-D storm/cloud layer kernels (billboard layers at depths, lit by one
light, composited back to front in post; they parallax with the camera because each layer is a plane in the world).
"""
import math

import numpy as np
from numba import njit, prange

from core import perlin3, smoothstep

BAR, BEAT = 80, 20


def bar(n, b=1.0):
    return int(round((n - 1) * BAR + (b - 1) * BEAT))


T_EYE, T_SLIT, T_EYE_END = bar(25), bar(26), bar(27)                               # 1920, 2000, 2080
T_GRASP, T_CLOSE, T_CRACK, T_SLIP, T_GRASP_END = bar(30), bar(30, 3), bar(31), bar(31, 3), bar(32)
T_FALL, T_FALL_END = bar(35), bar(36, 3)                                             # 2720, 2840
SHOTS2 = [(T_EYE, T_EYE_END, 'eye'), (T_GRASP, T_GRASP_END, 'grasp'), (T_FALL, T_FALL_END, 'fall')]
ENABLED = {'eye'}           # shots not yet rebuilt here stay on EMBERS-C's timeline (its stand-ins)


def mine(t):
    for a, b, name in SHOTS2:
        if a <= t < b:
            return name if name in ENABLED else None
    return None


class Router:
    """render.py's scene for --cut C3: EMBERS-C2's frames go to their shot; everything else to EMBERS-C's timeline"""

    def __init__(self, base):
        self.base = base
        self._shots = {}

    def _get(self, name):
        if name not in self._shots:
            if name == 'eye':
                import c_eye
                self._shots[name] = c_eye.EyeShot()
            elif name == 'grasp':
                import c_grasp
                self._shots[name] = c_grasp.GraspShot()
            else:
                import c_fall
                self._shots[name] = c_fall.FallShot()
        return self._shots[name]

    def pick(self, t):
        name = mine(t)
        return self.base if name is None else self._get(name)

    def camera(self, t):
        return self.pick(t).camera(t)

    def render_opts(self, f):
        return self.pick(f).render_opts(f)

    def emit(self, ctx):
        return self.pick(ctx.t).emit(ctx)

    def post(self, ctx, hdr):
        return self.pick(ctx.t).post(ctx, hdr)

    def finish_opts(self, f):
        return self.pick(f).finish_opts(f)

    def __getattr__(self, k):                 # anything else (towers, light, ...) is the base timeline's
        return getattr(self.base, k)


# ============================================================ plane layers ===

def plane_coords(cam, W, H, C, ex, ey, n, scale):
    """per pixel: the point where the pixel's ray meets the plane through C spanned by ex, ey (normal n), in units
    of `scale` along ex, ey -> X, Y (H, W) float32; and a mask of rays that hit it in front of the lens"""
    f = cam.f_px(W)
    u = (np.arange(W, dtype=np.float64) - (W - 1) / 2.0) / f
    v = -(np.arange(H, dtype=np.float64) - (H - 1) / 2.0) / f
    R = cam.R
    # ray direction (world) = right * u + up * v + fwd
    dx = R[0][None, None, :] * u[None, :, None] + R[1][None, None, :] * v[:, None, None] + R[2][None, None, :]
    den = dx @ n
    num = float((C - cam.pos) @ n)
    s = num / np.where(np.abs(den) < 1e-9, 1e-9, den)
    ok = s > 0
    P = cam.pos[None, None, :] + dx * s[..., None] - C[None, None, :]
    X = (P @ ex) / scale
    Y = (P @ ey) / scale
    return X.astype(np.float32), Y.astype(np.float32), ok


@njit(fastmath=True, cache=True, inline='always')
def _ss(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(fastmath=True, cache=True, inline='always')
def _fbm(x, y, z, octv, lac, gain):
    s = 0.0
    a = 1.0
    nrm = 0.0
    for o in range(octv):
        s += a * perlin3(x, y, z)
        nrm += a
        x *= lac
        y *= lac
        z *= lac
        a *= gain
    return s / nrm


@njit(fastmath=True, cache=True, inline='always')
def _storm_density(x, y, t, pr):
    """density of the storm at plane point (x, y) (units of the Eye's radius) at time t.
    pr: 0 seed, 1 freq, 2 draw (the texture's contraction: >1 = drawn in toward the Eye; it never turns), 3 warp,
        4 form (0 storm .. 1 Eye), 5 hole r0, 6 hole soft,
        7 torus r_in, 8 torus r_out, 9 bias, 10 evolve rate, 11 push (outward pressure wave), 12 core fill"""
    r = math.sqrt(x * x + y * y) + 1e-6
    # drawn in: the texture contracts toward the Eye (advected along log r only; nothing turns: no spiral)
    xs = x * pr[2]
    ys = y * pr[2]
    # the pressure wave pushes the billows outward
    k = 1.0 - pr[11] * math.exp(-((r - 1.4) / 0.9) ** 2) * 0.18
    xs *= k
    ys *= k
    f = pr[1]
    z = t * pr[10] + pr[0]
    # turbulence: domain warp
    wx = _fbm(xs * f * 0.7 + 3.1, ys * f * 0.7 - 1.7, z * 0.8, 3, 2.03, 0.5)
    wy = _fbm(xs * f * 0.7 - 7.3, ys * f * 0.7 + 4.9, z * 0.8 + 11.0, 3, 2.03, 0.5)
    xw = xs + pr[3] * wx
    yw = ys + pr[3] * wy
    n = _fbm(xw * f, yw * f, z, 5, 2.03, 0.52)
    # billows: rounded tops (abs noise folded in at a larger scale)
    b = 1.0 - 2.0 * abs(perlin3(xw * f * 0.5 + 9.1, yw * f * 0.5 - 2.2, z * 0.7 + 4.0))
    n = 0.72 * n + 0.28 * b * 0.6
    # the torus: where the storm gathers round the Eye (and the whole sky before it forms)
    tor = _ss(pr[7] - 0.5, pr[7] + 0.35, r) * (1.0 - _ss(pr[8] - 1.2, pr[8] + 1.5, r))
    env = pr[12] * (1.0 - pr[4]) + tor * (0.55 + 0.45 * pr[4]) + 0.25
    d = _ss(-0.04, 0.22, n + pr[9]) * env
    # the Eye draws the storm back from itself: a ragged hole that opens as it forms
    hr = pr[5] * (1.0 + 0.2 * perlin3(x / r * 2.3 + 5.0, y / r * 2.3, z * 1.3 + 2.0)
                  + 0.09 * perlin3(x / r * 7.0 - 3.0, y / r * 7.0, z * 2.0 + 9.0))
    hole = _ss(hr - pr[6], hr + pr[6], r)
    return d * (1.0 - pr[4] + pr[4] * hole)


@njit(parallel=True, fastmath=True, cache=True)
def storm_layer(X, Y, ok, t, pr, lp, out_rgb, out_a):
    """one billboard layer of the storm round the Eye.
    lp: 0 light gain, 1 light radius, 2 edge light, 3 transmitted (back-lit) light, 4 sigma (opacity),
        5 below glow (the race's fires under the storm), 6 depth offset of the layer behind the Eye (+: behind),
        7 eye brightness (0..1+), 8 edge step, 9 body light, 10 flare (the slit's opening)"""
    H, W = X.shape
    for j in prange(H):
        for i in range(W):
            if not ok[j, i]:
                out_rgb[j, i, 0] = 0.0
                out_rgb[j, i, 1] = 0.0
                out_rgb[j, i, 2] = 0.0
                out_a[j, i] = 0.0
                continue
            x = X[j, i]
            y = Y[j, i]
            d = _storm_density(x, y, t, pr)
            if d < 1e-4:
                out_rgb[j, i, 0] = 0.0
                out_rgb[j, i, 1] = 0.0
                out_rgb[j, i, 2] = 0.0
                out_a[j, i] = 0.0
                continue
            r = math.sqrt(x * x + y * y) + 1e-6
            # lit edge: the side of a billow that faces the Eye (denser here than a step toward it)
            st = lp[8]
            d2 = _storm_density(x - x / r * st, y - y / r * st, t, pr)
            edge = d - d2
            if edge < 0.0:
                edge = 0.0
            edge = min(edge * 4.0, 1.0)
            dz = lp[6]
            rr = math.sqrt(r * r + dz * dz)
            I = lp[0] * lp[7] / (1.0 + (rr / lp[1]) ** 2) ** 2.0 * (1.0 + 0.8 * lp[10])
            # thin cloud glows with light through it (back-lit layers); thick cloud is dark
            thin = d * math.exp(-d * 2.2)
            L = I * (lp[9] * d + lp[2] * edge + lp[3] * thin)
            # colour: gold-orange near the Eye, crimson farther, the dark red of smoke far out
            u = min(rr / 2.4, 1.0)
            cr = 1.0
            cg = 0.5 * (1.0 - u) + 0.1 * u
            cb = 0.17 * (1.0 - u) + 0.035 * u
            # the race's fires glowing up under the storm (from below the frame): only its underside
            gl = lp[5] * _ss(-1.0, -2.8, y) * d
            out_rgb[j, i, 0] = L * cr + gl * 0.9
            out_rgb[j, i, 1] = L * cg + gl * 0.16
            out_rgb[j, i, 2] = L * cb + gl * 0.05
            out_a[j, i] = 1.0 - math.exp(-lp[4] * d)
