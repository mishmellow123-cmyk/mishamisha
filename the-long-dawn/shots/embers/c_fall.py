"""EMBERS-C2 . E13a THE RING FALLS, part a (cut C, C13, C frames 2720-2840, bars 35-36 b2).

"Out of the black a gold glint tumbles slowly, its letters faintly awake, down through cloud." RUN-C takes part b at
2840 (bar 36 b3): over a moonlit range drawn in ink it streaks like a falling star.

    2720  black (C12's black runs on); the band turns into the moon's light: a glint, then its gold; letters faint
    2724  far below, a moonlit deck of cloud comes up out of the dark; we fall with the band, looking down on it
    2760  a thin veil of high wisps; the band falls through it
    2764  we slow; the band does not: it falls away from us, smaller, toward the deck
    2792  it goes into the cloud: a warm glow sinking in the moonlit tops, the cloud round it lit gold, then fading
    2830  we come down to the tops ourselves (the mist starts to close over the lens) -> cut to RUN-C

The band is ringsolid's canonical Ring (EMBERS-C's), always three-quarter to the lens (face-on it reads as a halo);
moonlight is the only light it catches, with a warm fill so the gold stays gold; the letters are awake. The clouds
are horizontal billboard layers (`_moon_layer`: billowed fbm lit on the moon's side, silver-edged, dark in the folds)
seen from above; the band's warmth lights them from inside once it is in them.
"""
import math

import numpy as np
from numba import njit, prange

from core import Camera, smoothstep, smootherstep, lerp, catmull, perlin3
import ringsolid as RS
import c2
from c2 import T_FALL, T_FALL_END

RING_W = 1.3
MOON = np.array([-0.45, 0.8, 0.4])
MOON = MOON / np.linalg.norm(MOON)
C_MOON = np.array([0.62, 0.72, 0.9])
DECK_Y = -80.0                                   # the deck's top (the band goes in at ~2792)
WISP_Y = -40.0
T_IN = 2792.0
# the camera: (frame offset, height, horizontal offset from the band's line, hfov)
CAM_KEYS = [(-20, 90.0, 52.0, 42.0), (0, 70.0, 52.0, 42.0), (40, 32.0, 50.0, 42.0), (70, 8.0, 46.0, 41.0),
            (95, -30.0, 40.0, 40.0), (119, -64.0, 34.0, 40.0), (140, -80.0, 30.0, 40.0)]
H_CAM = np.array([0.55, 0.0, -0.83])             # from the band toward the lens (horizontal); the moon is ahead, high
H_CAM = H_CAM / np.linalg.norm(H_CAM)


# ============================================================ motion ===

def ring_pos(t):
    x = t - T_FALL
    return np.array([0.03 * x + 0.9 * math.sin(0.02 * x), -0.92 * x - 0.0028 * x * x, 0.02 * x])


def ring_rot(t):
    """tumbling slowly; its hole kept roughly three-quarter to the lens (never a flat halo)"""
    x = t - T_FALL
    a = 0.045 * x + 0.9
    b = 0.028 * x + 0.4
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Rz = np.array([[cb, -sb, 0], [sb, cb, 0], [0, 0, 1]])
    return Rz @ Rx


def cam_state(t):
    x = t - T_FALL
    y, hoff, hf = catmull(x, [(k[0], np.array(k[1:])) for k in CAM_KEYS])
    r = ring_pos(t)
    base = np.array([r[0], 0.0, r[2]])
    pos = base + H_CAM * hoff + np.array([0.0, y, 0.0])
    # look at the band until it is in the cloud, then at where it went in (and a little below: its glow)
    k = float(smoothstep(T_IN - 14, T_IN + 12, t))
    rin = ring_pos(T_IN)
    tgt = r * (1 - k) + (rin + np.array([0.0, -4.0, 0.0])) * k
    return pos, tgt, float(hf)


def dark(t):
    """out of the black: the deck comes up out of the dark as the band turns into the moonlight"""
    return float(smoothstep(T_FALL + 4, T_FALL + 44, t))


def glint(t):
    return float(smoothstep(T_FALL, T_FALL + 8, t))


# ============================================================ clouds ===

@njit(fastmath=True, cache=True, inline='always')
def _ss(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(fastmath=True, cache=True, inline='always')
def _dens(x, y, z, pr):
    f = pr[1]
    wx = c2._fbm(x * f * 0.6 + 2.3, y * f * 0.6 - 1.1, z * 0.004 + pr[0], 3, 2.03, 0.5)
    wy = c2._fbm(x * f * 0.6 - 5.2, y * f * 0.6 + 3.7, z * 0.004 + 8.0 + pr[0], 3, 2.03, 0.5)
    xw = x * f + 1.5 * wx
    yw = y * f + 1.5 * wy
    n = c2._fbm(xw, yw, pr[0] + z * 0.003, 6, 2.03, 0.52)
    b = 1.0 - 2.0 * abs(perlin3(xw * 0.45 + 9.1, yw * 0.45 - 2.2, pr[0] * 0.7))
    n = 0.72 * n + 0.28 * 0.6 * b
    d = _ss(-0.03, 0.2, n + pr[2])
    # the wake: where the band went in, the tops are torn open and slowly close
    if pr[4] > 0.0:
        dx = x - pr[5]
        dy = y - pr[6]
        d *= 1.0 - pr[4] * math.exp(-(dx * dx + dy * dy) / (pr[7] * pr[7]))
    return d


@njit(parallel=True, fastmath=True, cache=True)
def _moon_layer(X, Y, ok, t, pr, lp, rp, out_rgb, out_a):
    """a horizontal cloud layer seen from above.
    pr: 0 seed, 1 freq, 2 bias, 3 lit, 4 wake depth, 5-6 wake centre, 7 wake radius
    lp: 0 moon gain, 1-2 the moon's in-plane direction, 3 sigma (opacity), 4 the band's warmth, 5 the band's warmth
        spread; rp: the band's in-plane position (x, y) and its depth under the tops (>0 inside)"""
    H, W = X.shape
    for j in prange(H):
        for i in range(W):
            out_rgb[j, i, 0] = 0.0
            out_rgb[j, i, 1] = 0.0
            out_rgb[j, i, 2] = 0.0
            out_a[j, i] = 0.0
            if not ok[j, i]:
                continue
            x = X[j, i]
            y = Y[j, i]
            d = _dens(x, y, t, pr)
            if d < 1e-4:
                continue
            st = 2.5
            d2 = _dens(x + lp[1] * st, y + lp[2] * st, t, pr)
            lit = d - d2                                   # the side of a billow that faces the moon
            lit = min(max(lit * 2.2 + 0.25, 0.0), 1.0)
            thin = d * math.exp(-d * 2.4)
            L = lp[0] * pr[3] * (0.12 * d + 0.75 * lit * d + 0.35 * thin)
            # the band's warmth inside the cloud: it spreads as it sinks and fades
            dx = x - rp[0]
            dy = y - rp[1]
            sp = lp[5] + 0.9 * rp[2]
            rg = lp[4] * math.exp(-(dx * dx + dy * dy) / (sp * sp)) * (0.35 + 0.65 * d)
            out_rgb[j, i, 0] = L * 0.62 + rg * 1.0 + 0.002 * d
            out_rgb[j, i, 1] = L * 0.72 + rg * 0.52 + 0.0025 * d
            out_rgb[j, i, 2] = L * 0.9 + rg * 0.16 + 0.0045 * d
            out_a[j, i] = 1.0 - math.exp(-lp[3] * d)


@njit(parallel=True, fastmath=True, cache=True)
def _veil(W, H, t, k, out_rgb, out_a):
    """the tops closing over the lens"""
    for j in prange(H):
        for i in range(W):
            x = (i - W * 0.5) / H * 2.6
            y = (j - H * 0.5) / H * 2.6 + t * 0.03
            n = c2._fbm(x * 1.2, y * 1.2, t * 0.02, 4, 2.03, 0.5)
            m = _ss(-0.3, 0.4, n + 0.9 * (k - 0.5))
            out_rgb[j, i, 0] = m * 0.03 * k
            out_rgb[j, i, 1] = m * 0.036 * k
            out_rgb[j, i, 2] = m * 0.05 * k
            out_a[j, i] = min(m * k * 1.2, 1.0)


# =============================================================== shot ===

class FallShot:
    def __init__(self):
        pass

    def camera(self, t):
        pos, tgt, hf = cam_state(t)
        return Camera(pos, tgt, hfov=hf, focus=float(np.linalg.norm(ring_pos(t) - pos)), aperture=0.0)

    def render_opts(self, f):
        return dict(bokeh_pow=0.0, bokeh_cap=1.0, near=0.05)

    def emit(self, ctx):
        pass

    def env(self, t):
        e = RS.Env(above=(0.003, 0.0035, 0.006), horizon=(0.004, 0.0045, 0.007), below=(0.002, 0.0025, 0.004))
        g = glint(t)
        e.lobe(MOON, C_MOON * 7.0 * g, 700.0)                      # the moon: a small hard body
        e.lobe(MOON, C_MOON * 0.12 * g, 5.0)                       # its halo in the thin air
        e.lobe([0.0, -1.0, 0.0], C_MOON * 0.07 * dark(t), 1.4)     # the moonlit deck below
        e.lobe([0.2, 0.5, 0.3], [0.2, 0.12, 0.045], 0.5)           # warm fill: the gold stays gold, never olive
        return e

    def post(self, ctx, hdr):
        t = ctx.t
        W, H = ctx.fr.W, ctx.fr.H
        cam = ctx.cam
        dk = dark(t)
        rpos = ring_pos(t)
        up = np.array([0.0, 1.0, 0.0])
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 0.0, 1.0])
        img = np.zeros((H, W, 3), np.float32)
        # ---- the band (three instants across the shutter)
        acc = None
        for q in range(3):
            tq = ctx.t0 + (ctx.t1 - ctx.t0) * q / 2.0
            cq = self.camera(tq)
            st = RS.RingState()
            st.letters = (0.55 + 0.12 * math.sin(0.21 * tq)) * (0.4 + 0.6 * glint(tq))
            st.glow = 0.015
            rgb, a, d = RS.render(cq, W, H, ring_rot(tq), ring_pos(tq), RING_W, st, self.env(tq))
            if acc is None:
                acc = [rgb, a]
            else:
                acc[0] += rgb
                acc[1] += a
        rrgb, ra = acc[0] / 3.0, acc[1] / 3.0
        # ---- clouds (from above): the deck, and a thin veil of high wisps
        rin = ring_pos(T_IN)
        layers = []
        for yk, seed, fq, bias, sig, lit, wake in ((DECK_Y - 36.0, 29.0, 0.012, 0.05, 1.8, 0.45, False),
                                                   (DECK_Y, 3.0, 0.018, 0.12, 3.0, 1.0, True),
                                                   (WISP_Y, 17.0, 0.03, -0.3, 1.3, 0.7, False)):
            if cam.pos[1] <= yk + 0.5:
                continue
            X, Y, ok = c2.plane_coords(cam, W, H, np.array([0.0, yk, 0.0]), ex, ey, up, 1.0)
            wk = 0.75 * float(smoothstep(T_IN - 2, T_IN + 6, t)) * (1.0 - 0.5 * float(smoothstep(T_IN + 10, T_FALL_END, t))) \
                if wake else 0.0
            pr = np.array([seed, fq, bias, lit, wk, rin[0], rin[2], 3.5 + 0.12 * max(t - T_IN, 0.0)], np.float64)
            mdx, mdz = MOON[0], MOON[2]
            nn = math.hypot(mdx, mdz)
            inside = rpos[1] < yk
            depth = (yk - rpos[1]) if inside else 0.0
            warm = 0.0
            if wake:
                # the band's warmth in the tops: from the moment it goes in, sinking and fading
                warm = 0.9 * float(smoothstep(T_IN - 3, T_IN + 2, t)) * math.exp(-max(t - T_IN, 0.0) / 26.0)
            lp = np.array([0.3 * dk, mdx / nn, mdz / nn, sig, warm, 2.5], np.float64)
            rp = np.array([rpos[0], rpos[2], depth], np.float64)
            rgb = np.empty((H, W, 3), np.float32)
            a = np.empty((H, W), np.float32)
            _moon_layer(X, Y, ok, float(t), pr, lp, rp, rgb, a)
            layers.append((yk, rgb * dk, a * dk))
        for yk, rgb, a in layers:                          # far to near; the band where it is among them
            if yk < rpos[1]:
                img = img * (1.0 - a[..., None]) + rgb
        img = img * (1.0 - ra[..., None]) + rrgb
        for yk, rgb, a in layers:
            if yk >= rpos[1]:
                img = img * (1.0 - a[..., None]) + rgb
        # ---- we come down to the tops ourselves
        kv = float(smoothstep(DECK_Y + 30.0, DECK_Y + 6.0, cam.pos[1])) * dk
        if kv > 0.01:
            vr = np.empty((H, W, 3), np.float32)
            va = np.empty((H, W), np.float32)
            _veil(W, H, float(t), kv, vr, va)
            img = img * (1.0 - va[..., None]) + vr
        return hdr + img

    def finish_opts(self, f):
        return dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.9, streak_strength=0.0, vignette_amount=0.3)
