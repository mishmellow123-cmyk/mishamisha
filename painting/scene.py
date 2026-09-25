"""
The still life: a glass sphere on a stone parapet, an opened letter, the
landscape beyond. A small ray tracer -- refraction with dispersion, Fresnel
reflection, soft sun shadows, and a caustic traced photon by photon: the
sphere is a ball lens, and it gathers the evening sun into one bright point.

Units: the sphere has radius 1. y is up, z runs from the viewer into the
picture, x to the right.
"""
import math

import numpy as np
from scipy import ndimage

import landscape as L

SPHERE_C = np.array([0.0, 1.0, 0.0])
SPHERE_R = 1.0
IOR = (1.512, 1.521, 1.534)                  # red, green, blue: glass disperses
GLASS_ABS = np.array([0.10, 0.035, 0.075])   # a faint green, as old glass has
ZF, ZB = -1.9, 1.3                           # parapet: front face, far edge
CORNICE = 0.14                               # the top slab overhangs the face
SUN = L.SUN / np.linalg.norm(L.SUN)
SUN_RADIUS = math.radians(0.9)
SUN_E = np.array([2.6, 2.3, 1.85])           # evening sun
FILL_DIR = np.array([-0.55, 0.35, -0.76])
FILL_DIR = FILL_DIR / np.linalg.norm(FILL_DIR)
FILL_E = np.array([0.155, 0.13, 0.105])      # warm light from the loggia behind us
COLS = [(-3.35, 0.55), (3.35, 0.55)]
COL_R, PLINTH_A, PLINTH_H = 0.40, 0.55, 0.30
# the letter: lying on the parapet, its last third hanging over the edge toward us
LX0, LX1 = 0.34, 1.56
LZ_FAR = -0.70
LZ_HANG = ZF - CORNICE - 0.004
LY_BOTTOM = -0.56
L_TOP = LZ_FAR - LZ_HANG
L_TOTAL = L_TOP - LY_BOTTOM


def nrm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Intersections (vectorised over rays)
# ---------------------------------------------------------------------------
def hit_sphere(o, d, c=SPHERE_C, r=SPHERE_R):
    oc = o - c
    b = np.sum(oc * d, -1)
    cc = np.sum(oc * oc, -1) - r * r
    disc = b * b - cc
    t = np.full(len(o), np.inf)
    ok = disc > 0
    sq = np.sqrt(np.maximum(disc, 0))
    t0 = -b - sq
    t1 = -b + sq
    tt = np.where(t0 > 1e-6, t0, t1)
    ok &= tt > 1e-6
    t[ok] = tt[ok]
    return t


def hit_top(o, d):
    """The parapet's upper surface, y = 0, between its front and far edges."""
    t = np.full(len(o), np.inf)
    ok = d[:, 1] < -1e-9
    tt = -o[:, 1] / np.where(ok, d[:, 1], -1)
    p = o + d * tt[:, None]
    ok &= (tt > 1e-6) & (p[:, 2] >= ZF - CORNICE) & (p[:, 2] <= ZB)
    t[ok] = tt[ok]
    return t


def hit_front(o, d):
    """The face below the cornice (z = ZF), and the cornice's own front edge."""
    t = np.full(len(o), np.inf)
    ok = d[:, 2] > 1e-9
    # cornice front: z = ZF - CORNICE, y in [-0.16, 0]
    tt = (ZF - CORNICE - o[:, 2]) / np.where(ok, d[:, 2], 1)
    p = o + d * tt[:, None]
    okc = ok & (tt > 1e-6) & (p[:, 1] <= 0) & (p[:, 1] >= -0.16)
    t[okc] = tt[okc]
    tt2 = (ZF - o[:, 2]) / np.where(ok, d[:, 2], 1)
    p2 = o + d * tt2[:, None]
    okf = ok & (tt2 > 1e-6) & (p2[:, 1] < -0.16) & ~okc
    t[okf] = tt2[okf]
    return t


def hit_columns(o, d):
    t = np.full(len(o), np.inf)
    which = np.full(len(o), -1)
    for k, (cx, cz) in enumerate(COLS):
        # shaft
        ox, oz = o[:, 0] - cx, o[:, 2] - cz
        a = d[:, 0] ** 2 + d[:, 2] ** 2
        b = ox * d[:, 0] + oz * d[:, 2]
        c = ox * ox + oz * oz - COL_R ** 2
        disc = b * b - a * c
        ok = (disc > 0) & (a > 1e-12)
        tt = (-b - np.sqrt(np.maximum(disc, 0))) / np.where(a > 1e-12, a, 1)
        y = o[:, 1] + d[:, 1] * tt
        ok &= (tt > 1e-6) & (y > PLINTH_H)
        better = ok & (tt < t)
        t[better], which[better] = tt[better], k
        # the base's ring moulding (a short, wider drum)
        rr = COL_R + 0.09
        c2 = ox * ox + oz * oz - rr ** 2
        disc2 = b * b - a * c2
        ok2 = (disc2 > 0) & (a > 1e-12)
        tt2 = (-b - np.sqrt(np.maximum(disc2, 0))) / np.where(a > 1e-12, a, 1)
        y2 = o[:, 1] + d[:, 1] * tt2
        ok2 &= (tt2 > 1e-6) & (y2 > PLINTH_H) & (y2 < PLINTH_H + 0.13)
        better = ok2 & (tt2 < t)
        t[better], which[better] = tt2[better], k
        # its top: a flat ring
        tcap = (PLINTH_H + 0.13 - o[:, 1]) / np.where(np.abs(d[:, 1]) > 1e-9, d[:, 1], 1e-9)
        pc = o + d * tcap[:, None]
        rc = np.hypot(pc[:, 0] - cx, pc[:, 2] - cz)
        okc = (tcap > 1e-6) & (rc < rr) & (rc > COL_R) & (d[:, 1] < 0)
        better = okc & (tcap < t)
        t[better], which[better] = tcap[better], k + 20
        # plinth (a box)
        lo = np.array([cx - PLINTH_A, 0.0, cz - PLINTH_A])
        hi = np.array([cx + PLINTH_A, PLINTH_H, cz + PLINTH_A])
        inv = 1.0 / np.where(np.abs(d) > 1e-12, d, 1e-12)
        t1, t2 = (lo - o) * inv, (hi - o) * inv
        tmin = np.max(np.minimum(t1, t2), -1)
        tmax = np.min(np.maximum(t1, t2), -1)
        okb = (tmax >= tmin) & (tmin > 1e-6)
        better = okb & (tmin < t)
        t[better], which[better] = tmin[better], k + 10
    return t, which


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def fbm2(x, y, octaves, seed):
    return L.fbm(x, y, octaves, seed=seed)


def stone_albedo(u, v, seed=0):
    """Warm limestone: a fine grain, soft clouding, the odd thin vein."""
    base = np.array([0.50, 0.445, 0.37])
    cloud = fbm2(u * 1.1, v * 1.1, 4, seed + 1)
    grain = fbm2(u * 55, v * 55, 2, seed + 4)
    warp = fbm2(u * 0.6, v * 0.6, 3, seed + 3)
    vein = np.abs(fbm2(u * 0.9 + 1.6 * warp, v * 0.9 - 1.1 * warp, 5, seed + 2))
    vein = smoothstep(0.012, 0.0, vein) * 0.6
    a = base[None] * (1 + 0.10 * cloud + 0.035 * grain)[:, None]
    a = a * (1 - 0.14 * vein[:, None])
    a *= np.array([1.0, 1.0, 1.0]) + np.array([0.02, 0.0, -0.03]) * cloud[:, None]
    return a


class Maps:
    """Everything precomputed: the world outside, the caustic, the letter."""

    def __init__(self, env_hi, env_wide, caustic, letter_tex, inscription):
        self.env_hi, self.env_wide = env_hi, env_wide
        self.caustic = caustic
        self.letter = letter_tex
        self.inscription = inscription


def sample_env(maps, d, channel=None):
    """The landscape and sky beyond the loggia, or the loggia behind us.
    channel=3 returns how much of each ray is open sky instead."""
    az = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
    el = np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
    sl = slice(0, 3) if channel is None else slice(channel, channel + 1)
    out = np.zeros((len(d), 3 if channel is None else 1))
    back = d[:, 2] <= 0.02
    if channel is None:
        out[back] = room(d[back])
    fwd = ~back
    img, (az0, az1, el0, el1, ppd) = maps.env_hi
    inside = fwd & (az > az0 + 0.1) & (az < az1 - 0.1) & (el > el0 + 0.1) & (el < el1 - 0.1)
    out[inside] = bilinear(img[..., sl], (az[inside] - az0) * ppd - 0.5, (el1 - el[inside]) * ppd - 0.5)
    rest = fwd & ~inside
    img, (az0, az1, el0, el1, ppd) = maps.env_wide
    a = np.clip(az[rest], az0, az1)
    e = np.clip(el[rest], el0, el1)
    out[rest] = bilinear(img[..., sl], (a - az0) * ppd - 0.5, (el1 - e) * ppd - 0.5)
    return out


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def room(d):
    """The loggia behind the viewer: dim, warm, with a window to the left."""
    az = np.degrees(np.arctan2(d[:, 0], d[:, 2]))            # |az| > 90 here
    el = np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
    col = np.array([0.045, 0.036, 0.028])[None] * (0.7 + 0.3 * smoothstep(-40, 40, el))[:, None]
    win = smoothstep(0.0, 1.2, 9.0 - np.abs(az + 142.0)) * smoothstep(0.0, 1.2, 16.0 - np.abs(el - 12.0))
    mull = (np.abs(az + 142.0) < 0.5) | (np.abs(el - 12.0) < 0.5)
    win = win * (~mull)
    col = col + np.array([1.9, 1.75, 1.5])[None] * win[:, None]
    return col


def sun_visibility(p, soft=True):
    """Fraction of the sun's disc not hidden by the sphere or the columns."""
    v = SPHERE_C[None] - p
    dist = np.linalg.norm(v, axis=-1)
    cosang = np.sum(v * SUN[None], -1) / dist
    ang = np.arccos(np.clip(cosang, -1, 1))
    alpha = np.arcsin(np.clip(SPHERE_R / dist, 0, 1))
    vis = smoothstep(alpha - SUN_RADIUS, alpha + SUN_RADIUS, ang)
    tc, _ = hit_columns(p + SUN[None] * 1e-4, np.broadcast_to(SUN, p.shape).copy())
    vis = vis * np.isinf(tc)
    return vis


def shade_top(maps, p, d):
    """The parapet's top, the letter lying on it, and the gathered light."""
    x, z = p[:, 0], p[:, 2]
    alb = stone_albedo(x * 1.0, z * 1.0)
    n = np.tile(np.array([0.0, 1.0, 0.0]), (len(p), 1))
    # the letter
    u = (x - LX0) / (LX1 - LX0)
    v = (LZ_FAR - z) / L_TOTAL
    on = (u > 0) & (u < 1) & (z > LZ_HANG) & (z < LZ_FAR)
    if on.any():
        alb[on] = letter_albedo(maps, u[on], v[on])
        # it was folded in three: the panels don't quite lie flat
        panel_tilt = np.where(v[on] < 1 / 3, -0.06, np.where(v[on] < 2 / 3, 0.035, -0.02))
        n[on] = nrm(np.stack([np.zeros_like(panel_tilt), np.ones_like(panel_tilt), panel_tilt], -1))
    near_x = np.clip(np.maximum(LX0 - x, x - LX1), 0, None)
    near_z = np.clip(z - LZ_FAR, 0, None)
    ao = np.where(on, 1.0, 1 - 0.35 * np.exp(-np.hypot(near_x, near_z) / 0.025) * (z < LZ_FAR + 0.05) * (x > LX0 - 0.05) * (x < LX1 + 0.05))
    # sphere contact
    rc = np.hypot(x - SPHERE_C[0], z - SPHERE_C[2])
    ao = ao * (1 - 0.55 * np.exp(-(rc / 0.22) ** 2))
    ndl = np.clip(n @ SUN, 0, 1)
    vis = sun_visibility(p)
    E = SUN_E[None] * (ndl * vis)[:, None]
    E = E + caustic_at(maps, x, z)
    sky_amb = np.array([0.30, 0.34, 0.42]) * 0.55
    E = E + sky_amb[None] * ao[:, None] + FILL_E[None] * max(0.0, float(FILL_DIR @ np.array([0, 1, 0]))) * ao[:, None]
    col = alb * E
    # polished stone: a little of the sky in the far part of the parapet
    fres = 0.04 + 0.96 * (1 - np.clip(-d[:, 1], 0, 1)) ** 5
    col = col + 0.25 * fres[:, None] * sample_env(maps, d * np.array([1, -1, 1])) * (~on)[:, None]
    return col


def letter_albedo(maps, u, v):
    tex = maps.letter
    th, tw = tex.shape[:2]
    return bilinear(tex, u * (tw - 1), v * (th - 1))


def hit_hanging(o, d):
    """The part of the letter hanging down in front of the cornice."""
    t = np.full(len(o), np.inf)
    ok = d[:, 2] > 1e-9
    tt = (LZ_HANG - o[:, 2]) / np.where(ok, d[:, 2], 1)
    p = o + d * tt[:, None]
    ok &= (tt > 1e-6) & (p[:, 0] > LX0) & (p[:, 0] < LX1) & (p[:, 1] < 0.004) & (p[:, 1] > LY_BOTTOM)
    t[ok] = tt[ok]
    return t


def shade_hanging(maps, p):
    u = (p[:, 0] - LX0) / (LX1 - LX0)
    v = (L_TOP - p[:, 1]) / L_TOTAL
    alb = letter_albedo(maps, u, v)
    n = np.array([0.0, 0.0, -1.0])
    # it curls a little as it goes over the edge
    top = np.exp(-(-p[:, 1]) / 0.05)
    E = FILL_E * 2.6 * max(0.0, float(n @ FILL_DIR)) + np.array([0.10, 0.095, 0.09])
    E = E[None] * (1 - 0.25 * top)[:, None] + (SUN_E * 0.35)[None] * top[:, None]
    # light bounced up from the sunlit stone below the sphere
    E = E + np.array([0.06, 0.05, 0.04])[None]
    return alb * E


def caustic_at(maps, x, z):
    img, (x0, x1, z0, z1) = maps.caustic
    h, w = img.shape[:2]
    fx = (x - x0) / (x1 - x0) * (w - 1)
    fz = (z1 - z) / (z1 - z0) * (h - 1)
    ok = (fx >= 0) & (fx < w - 1) & (fz >= 0) & (fz < h - 1)
    out = np.zeros((len(x), 3))
    if ok.any():
        out[ok] = bilinear(img, fx[ok], fz[ok])
    # as a painter would: the halo of aberrated light held down, so the
    # point where the sun is gathered stays the brightest thing on the stone
    return 1.1 * (1 - np.exp(-out / 5.0)) + 0.08 * np.maximum(out - 50.0, 0.0)


def shade_front(maps, p, d, is_cornice):
    """The parapet's face, with the motto cut into it."""
    x, y = p[:, 0], p[:, 1]
    alb = stone_albedo(x * 1.0 + 7.0, y * 1.0 - 3.0, seed=10) * 0.95
    n = np.tile(np.array([0.0, 0.0, -1.0]), (len(p), 1))
    ao = np.ones(len(p))
    # cornice shadow: the overhang shades the face just below it
    below = ~is_cornice
    ao[below] *= 1 - 0.55 * np.exp(-np.maximum(-0.16 - y[below], 0) / 0.07)
    # the carved inscription (a height field on the face)
    ins, (ix0, ix1, iy0, iy1) = maps.inscription
    h_, w_ = ins.shape[:2]
    fx = (x - ix0) / (ix1 - ix0) * (w_ - 1)
    fy = (iy1 - y) / (iy1 - iy0) * (h_ - 1)
    inside = below & (fx > 1) & (fx < w_ - 2) & (fy > 1) & (fy < h_ - 2)
    if inside.any():
        dep = bilinear(ins[..., None], fx[inside], fy[inside])[:, 0]
        gx = (bilinear(ins[..., None], fx[inside] + 1, fy[inside]) - bilinear(ins[..., None], fx[inside] - 1, fy[inside]))[:, 0]
        gy = (bilinear(ins[..., None], fx[inside], fy[inside] + 1) - bilinear(ins[..., None], fx[inside], fy[inside] - 1))[:, 0]
        k = 4.0
        nn = np.stack([gx * k, -gy * k, -np.ones_like(gx)], -1)       # grooves tilt the face
        n[inside] = nrm(nn)
        ao[inside] *= 1 - 0.65 * dep
        alb[inside] *= (1 - 0.35 * dep)[:, None]
    ndf = np.clip(n @ FILL_DIR, 0, 1)
    # the hanging letter shades the face behind it from the loggia's light
    tfill = (LZ_HANG - p[:, 2]) / FILL_DIR[2]
    q = p + FILL_DIR[None] * tfill[:, None]
    sh = (smoothstep(LX0 - 0.03, LX0 + 0.03, q[:, 0]) * smoothstep(LX1 + 0.03, LX1 - 0.03, q[:, 0])
          * smoothstep(LY_BOTTOM - 0.03, LY_BOTTOM + 0.03, q[:, 1]) * (q[:, 1] < 0.0) * ~is_cornice)
    ndf = ndf * (1 - 0.8 * sh)
    E = FILL_E[None] * 2.6 * ndf[:, None]
    E = E + np.array([0.09, 0.085, 0.08])[None] * ao[:, None]
    col = alb * E
    # the cornice's rounded lip catches the sky
    col[is_cornice] = alb[is_cornice] * (FILL_E * 2.8 + np.array([0.30, 0.30, 0.32]) * 0.6)
    return col


def shade_column(maps, p, d, which):
    col = np.zeros((len(p), 3))
    for k, (cx, cz) in enumerate(COLS):
        m = which == k
        if m.any():
            n = p[m] - np.array([cx, 0, cz])
            n[:, 1] = 0
            n = nrm(n)
            alb = stone_albedo(np.arctan2(n[:, 0], n[:, 2]) * 2, p[m, 1] * 1.3, seed=20 + k) * np.array([0.72, 0.76, 0.82])
            flute = 0.85 + 0.15 * np.cos(np.arctan2(n[:, 0], n[:, 2]) * 24)
            E = SUN_E * np.clip(n @ SUN, 0, 1)[:, None] + FILL_E * 1.6 * np.clip(n @ FILL_DIR, 0, 1)[:, None] + 0.05
            col[m] = alb * E * flute[:, None]
        m = which == k + 20
        if m.any():
            alb = stone_albedo(p[m, 0] * 2, p[m, 2] * 2, seed=40 + k) * np.array([0.72, 0.76, 0.82])
            E = SUN_E * max(0.0, float(SUN[1])) + FILL_E * 1.6 * max(0.0, float(FILL_DIR[1])) + 0.05
            col[m] = alb * E[None]
        m = which == k + 10
        if m.any():
            alb = stone_albedo(p[m, 0] * 2, p[m, 2] * 2 + p[m, 1], seed=30 + k) * np.array([0.72, 0.76, 0.82])
            q = p[m] - np.array([cx, 0, cz])
            n = np.zeros_like(q)
            ax = np.argmax(np.abs(q / np.array([PLINTH_A, PLINTH_H * 2, PLINTH_A])), -1)
            n[np.arange(len(q)), ax] = np.sign(q[np.arange(len(q)), ax])
            E = SUN_E * np.clip(n @ SUN, 0, 1)[:, None] + FILL_E * 1.6 * np.clip(n @ FILL_DIR, 0, 1)[:, None] + 0.05
            col[m] = alb * E
    return col


def trace_scene(maps, o, d):
    """Radiance along rays that do not meet the glass."""
    t_top = hit_top(o, d)
    t_front = hit_front(o, d)
    t_col, which = hit_columns(o, d)
    t_hang = hit_hanging(o, d)
    t = np.minimum(np.minimum(np.minimum(t_top, t_front), t_col), t_hang)
    out = np.zeros((len(o), 3))
    m = (t == t_hang) & np.isfinite(t)
    if m.any():
        out[m] = shade_hanging(maps, o[m] + d[m] * t[m, None])
    t_front = np.where(t == t_hang, np.inf, t_front)
    t_top = np.where(t == t_hang, np.inf, t_top)
    t_col = np.where(t == t_hang, np.inf, t_col)
    miss = np.isinf(t)
    if miss.any():
        out[miss] = sample_env(maps, d[miss])
    m = (t == t_top) & ~miss
    if m.any():
        p = o[m] + d[m] * t[m, None]
        out[m] = shade_top(maps, p, d[m])
    m = (t == t_front) & ~miss & (t != t_top)
    if m.any():
        p = o[m] + d[m] * t[m, None]
        corn = p[:, 2] < ZF - CORNICE * 0.5
        out[m] = shade_front(maps, p, d[m], corn)
    m = (t == t_col) & ~miss & (t != t_top) & (t != t_front)
    if m.any():
        p = o[m] + d[m] * t[m, None]
        out[m] = shade_column(maps, p, d[m], which[m])
    return out


def fresnel(cosi, n1, n2):
    """Unpolarised Fresnel reflectance."""
    sint = n1 / n2 * np.sqrt(np.maximum(0, 1 - cosi ** 2))
    tir = sint >= 1
    cost = np.sqrt(np.maximum(0, 1 - sint ** 2))
    rs = ((n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost)) ** 2
    rp = ((n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi)) ** 2
    return np.where(tir, 1.0, 0.5 * (rs + rp))


def refract(d, n, eta):
    cosi = -np.sum(d * n, -1)
    k = 1 - eta ** 2 * (1 - cosi ** 2)
    t = eta * d + (eta * cosi - np.sqrt(np.maximum(k, 0)))[:, None] * n
    return nrm(t), k < 0


def through_glass(o, d, ior):
    """Enter the sphere at o (on its surface) heading d; return exit point,
    exit direction, path length inside, and the two Fresnel transmissions."""
    n_in = nrm(o - SPHERE_C[None])
    cos_in = np.clip(-np.sum(d * n_in, -1), 0, 1)
    t_in = 1 - fresnel(cos_in, 1.0, ior)
    d1, _ = refract(d, n_in, 1.0 / ior)
    # far side of the sphere
    oc = o - SPHERE_C[None]
    b = np.sum(oc * d1, -1)
    c = np.sum(oc * oc, -1) - SPHERE_R ** 2
    tt = -b + np.sqrt(np.maximum(b * b - c, 0))
    p2 = o + d1 * tt[:, None]
    n_out = nrm(p2 - SPHERE_C[None])
    cos_out = np.clip(np.sum(d1 * n_out, -1), 0, 1)
    t_out = 1 - fresnel(cos_out, ior, 1.0)
    d2, tir = refract(d1, -n_out, ior)
    return p2, d2, tt, t_in * t_out, d1, n_out


def shade_glass(maps, o, d, p):
    """Everything seen in (and on) the glass."""
    n = nrm(p - SPHERE_C[None])
    cosi = np.clip(-np.sum(d * n, -1), 0, 1)
    R = fresnel(cosi, 1.0, IOR[1])
    refl_d = nrm(d - 2 * np.sum(d * n, -1)[:, None] * n)
    refl = trace_scene(maps, p + n * 1e-4, refl_d)
    # the sun itself, reflected: a small hard highlight
    spec = np.clip(refl_d @ SUN, 0, 1)
    refl += SUN_E[None] * 60.0 * (spec > math.cos(SUN_RADIUS * 1.2))[:, None] * 1.0
    refl += SUN_E[None] * 2.0 * np.exp(-(np.arccos(np.clip(spec, -1, 1)) / math.radians(2.5)) ** 2)[:, None]
    out = refl * R[:, None]
    for ch, ior in enumerate(IOR):
        p2, d2, path, T, d1, n_out = through_glass(p, d, ior)
        c = trace_scene(maps, p2 + d2 * 1e-4, d2)[:, ch]
        # the sun seen through the lens
        s = np.clip(d2 @ SUN, 0, 1)
        c = c + SUN_E[ch] * 40.0 * (s > math.cos(SUN_RADIUS))
        # one reflection inside the glass, faint
        out[:, ch] += c * T * np.exp(-GLASS_ABS[ch] * path)
    return out


# ---------------------------------------------------------------------------
# The caustic: follow the sun's light through the lens, photon by photon
# ---------------------------------------------------------------------------
def trace_caustic(extent=(-2.6, 3.6, -2.1, 1.3), res=0.004, n_photons=12_000_000, seed=3):
    x0, x1, z0, z1 = extent
    w = int((x1 - x0) / res)
    h = int((z1 - z0) / res)
    img = np.zeros((h, w, 3))
    rng = np.random.default_rng(seed)
    # an orthonormal frame around the sun direction
    s = SUN
    a = nrm(np.cross(s, np.array([0, 1.0, 0]))[None])[0]
    b = np.cross(s, a)
    area = math.pi * SPHERE_R ** 2
    chunk = 1_000_000
    for ch, ior in enumerate(IOR):
        done = 0
        while done < n_photons:
            m = min(chunk, n_photons - done)
            done += m
            r = SPHERE_R * np.sqrt(rng.uniform(0, 1, m))
            th = rng.uniform(0, 2 * np.pi, m)
            # jitter within the sun's disc
            jr = SUN_RADIUS * np.sqrt(rng.uniform(0, 1, m))
            jt = rng.uniform(0, 2 * np.pi, m)
            dirs = -(s[None] + (np.cos(jt) * jr)[:, None] * a[None] + (np.sin(jt) * jr)[:, None] * b[None])
            dirs = nrm(dirs)
            origin = SPHERE_C[None] + s[None] * 5 + (r * np.cos(th))[:, None] * a[None] + (r * np.sin(th))[:, None] * b[None]
            t = hit_sphere(origin, dirs)
            ok = np.isfinite(t)
            p = origin[ok] + dirs[ok] * t[ok, None]
            p2, d2, path, T, _, _ = through_glass(p, dirs[ok], ior)
            down = d2[:, 1] < -1e-6
            tt = -p2[down, 1] / d2[down, 1]
            q = p2[down] + d2[down] * tt[:, None]
            e = (T * np.exp(-GLASS_ABS[ch] * path))[down]
            fx = (q[:, 0] - x0) / res
            fz = (z1 - q[:, 2]) / res
            inb = (fx >= 0) & (fx < w - 1) & (fz >= 0) & (fz < h - 1)
            ix = fx[inb].astype(int)
            iz = fz[inb].astype(int)
            np.add.at(img[..., ch], (iz, ix), e[inb])
        # photons carry the flux of the sphere's cross-section; the plane
        # receives it at the sun's elevation
        flux_per = SUN_E[ch] * area / n_photons
        img[..., ch] *= flux_per / (res * res)
    img = np.stack([ndimage.gaussian_filter(img[..., c], 0.6) for c in range(3)], -1)
    return img, extent
