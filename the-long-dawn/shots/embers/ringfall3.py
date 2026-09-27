"""EMBERS-C: E13a THE RING FALLS (C13, C frames 2720-2839), before RUN-C's falling star over the ink range (2840).

    2720-2758  out of the black a gold glint: the Ring tumbling slowly, close, catching a cold light from above; its
               letters faintly awake (deep orange-red)
    2758-2826  we fall with it into cloud: moonlit cloud streams up past the lens (three layers, parallax), their
               tops silvered, their bellies dark; the Ring turns over and over in the grey
    2826-2839  out of the cloud's underside into clear night: we slow, it does not; it drops away below, a small
               gold glint falling (HANDOVER to RUN-C: at 2839 the Ring is a ~20 px glint just below frame centre,
               falling straight down)

Own renderer: ringsolid's canonical band (motion-blurred, lit by a moon lobe and the cloud) over a screen-space
cloud shader (domain-warped fbm, lit from above through a density-gradient normal: billows, not smudges).
"""
import math

import numpy as np
from numba import njit, prange

import look
from core import Camera, smoothstep, smootherstep, lerp, perlin3
import ringsolid as RS

T0, T1 = 2720, 2840
T_CLOUD0, T_CLOUD1 = 2752, 2826       # inside the cloud
T_DROP = 2822                         # the camera slows; the Ring falls on
RING_W = 1.0
MOON = np.array([-0.35, 0.86, -0.37]) / np.linalg.norm([-0.35, 0.86, -0.37])


@njit(fastmath=True, cache=True, inline='always')
def _fbm(x, y, z, octaves):
    a = 0.5
    s = 0.0
    for o in range(octaves):
        s += a * perlin3(x, y, z)
        x = x * 2.03 + 1.7
        y = y * 2.03 - 3.1
        z = z * 2.03 + 0.9
        a *= 0.5
    return s


@njit(fastmath=True, cache=True, inline='always')
def _ss(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(parallel=True, fastmath=True, cache=True)
def _clouds(out, W, H, t, scroll, amount, seed):
    """three layers of moonlit cloud streaming up past the lens (screen space; scroll = how far we have fallen)"""
    for iy in prange(H):
        for ix in range(W):
            x = ix / W * 3.2
            y = iy / W * 3.2
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            trans = 1.0
            for layer in range(3):
                k = 3 - layer                       # far (1) -> near (3): nearer layers stream faster, bigger
                sc = 0.55 + 0.35 * layer
                sp = 0.6 + 0.9 * layer
                u = x * sc + seed * 3.7 + 5.1 * layer
                v = y * sc + scroll * sp
                wx = _fbm(u * 0.8, v * 0.8, 1.3 * layer + 0.01 * t, 2)
                wy = _fbm(u * 0.8 + 4.0, v * 0.8 - 2.0, 0.7 * layer + 0.01 * t, 2)
                n = _fbm(u + 0.8 * wx, v + 0.8 * wy, 2.0 * layer, 5)
                dens = _ss(-0.02, 0.28, n) * amount
                if dens <= 0.001:
                    continue
                # lit from above: the density falls off upward on a lit top (a billow's silver lining)
                n_up = _fbm(u + 0.8 * wx, v + 0.8 * wy - 0.035, 2.0 * layer, 5)
                lit = _ss(-0.06, 0.06, n - n_up)                 # 1 where the cloud thins upward: a lit top
                L = (0.035 + 0.16 * lit) * (0.6 + 0.2 * layer)
                a = dens * (0.35 + 0.2 * layer)
                acc_r += trans * a * L * 0.78
                acc_g += trans * a * L * 0.88
                acc_b += trans * a * L * 1.05
                trans *= 1.0 - a
            out[iy, ix, 0] = out[iy, ix, 0] * trans + acc_r
            out[iy, ix, 1] = out[iy, ix, 1] * trans + acc_g
            out[iy, ix, 2] = out[iy, ix, 2] * trans + acc_b


def fall_depth(t):
    """how far the camera has fallen (world units): steady, then it slows at the cloud's underside"""
    x = t - T0
    y = 0.55 * x
    if t > T_DROP:
        d = t - T_DROP
        y = 0.55 * (T_DROP - T0) + 0.55 * d - 0.5 * 0.03 * min(d, 18.0) ** 2
    return y


def ring_state(t):
    """(Rot, centre in camera-following coordinates, scale) of the tumbling Ring"""
    x = t - T0
    a1 = 0.035 * x + 0.4
    a2 = 0.021 * x + 1.1
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    Rx = np.array([[1, 0, 0], [0, c1, -s1], [0, s1, c1]])
    Rz = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])
    Rot = Rz @ Rx
    C = np.array([0.35 * math.sin(0.02 * x), -fall_depth(t) - 0.4 * math.sin(0.013 * x), 0.0])
    if t > T_DROP:
        d = t - T_DROP
        C = C + np.array([0.0, -0.5 * 0.05 * d * d, -0.25 * d])            # it falls on, away from us
    return Rot, C, RING_W


def camera(t):
    y = -fall_depth(t)
    pos = np.array([0.0, y + 1.2, 9.0])
    tgt = np.array([0.0, y - 0.6, 0.0])
    if t > T_DROP:
        d = t - T_DROP
        tgt = tgt + np.array([0.0, -0.08 * d * d * 0.05, 0.0]) * 4.0          # tilt down after it
    return Camera(pos, tgt, hfov=40.0, focus=9.0, aperture=0.0)


def cloud_amount(t):
    return float(smoothstep(T_CLOUD0 - 10, T_CLOUD0 + 12, t)) * (1.0 - float(smoothstep(T_CLOUD1 - 10, T_CLOUD1 + 6, t)))


def render(f, scale=1.0):
    """-> HDR frame (H, W, 3)"""
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    hdr = np.zeros((H, W, 3), np.float32)
    t = float(f)
    # the night above the cloud, and the clear night below it (a trace of moonlight in the air)
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    sky = (0.004 + 0.012 * (1 - y)) * float(smoothstep(T0, T0 + 30, t))
    hdr += sky[..., None] * np.array([0.7, 0.8, 1.0], np.float32)
    # the Ring (five instants: it tumbles)
    acc = None
    n = 5
    for k in range(n):
        tq = t - 0.25 + 0.5 * (k + 0.5) / n
        cam = camera(tq)
        Rot, C, sc = ring_state(tq)
        st = RS.RingState()
        st.letters = 0.75 * (0.8 + 0.2 * math.sin(0.1 * tq))
        st.glow = 0.03
        st.glow_col = np.array([1.0, 0.75, 0.35])
        cl = cloud_amount(tq)
        env = RS.Env(above=(0.1 + 0.25 * cl, 0.12 + 0.27 * cl, 0.16 + 0.3 * cl),
                     horizon=(0.03 + 0.12 * cl, 0.035 + 0.13 * cl, 0.05 + 0.16 * cl),
                     below=(0.01, 0.012, 0.018))
        env.lobe(MOON, np.array([5.0, 5.6, 6.5]), 90.0)                         # the moon: the glint
        env.lobe(MOON * np.array([1, 0.6, 1]), np.array([0.25, 0.3, 0.4]), 6.0)
        rgb, a, d = RS.render(cam, W, H, Rot, C, sc, st, env)
        if acc is None:
            acc = [rgb, a]
        else:
            acc[0] += rgb
            acc[1] += a
    rgb, a = acc[0] / n, acc[1] / n
    # the cloud streams past: behind the Ring where it is far, over it where it is near (the near layer passes in
    # front now and then)
    amt = cloud_amount(t)
    if amt > 0:
        _clouds(hdr, W, H, t, fall_depth(t) * 0.09, amt, 1.3)
    return hdr * (1 - a[..., None]) + rgb


def finish(hdr):
    return look.finish(hdr, exposure=1.0, bloom_strength=0.12, bloom_threshold=0.7, vignette_amount=0.3)
