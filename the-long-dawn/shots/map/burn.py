"""X1 · the burn-throughs: a page browns outward from a point, chars, and burns open onto the next shot;
reversed (mode 1) it heals: the hole closes and the scorch fades back into clean paper.

A burn is a 16-float parameter block evaluated per pixel in page cm (numba), so it is exact at any
resolution and a pure function of time:

    0 on        1-2 origin (u, v) page cm      3 t_start (s)     4 speed (cm/s^p)    5 p (growth exponent)
    6 noise amplitude (fraction of radius)     7 noise frequency (1/cm)             8 seed
    9 brown width (cm)   10 char width (cm)    11 glowing-edge width (cm)            12 mode (0 burn, 1 heal)
    13 heat lead (s): browning starts this long before the hole opens               14 t_pivot (heal)
    15 aspect (stretch along v: > 1 runs faster up and down the page)

The front is the level set d(p) = R(t) of a ragged distance: |p - o| warped by two octaves of noise and
nibbled at fine scale, so the edge is torn, not a circle; ahead of it the paper browns (heat), then a band
of char, then a thin line of glowing embers with hot spots that crawl; behind it, the hole.
"""
import math

import numpy as np
from numba import njit

from noise import fbm, gnoise


def edge_params(side, depth, pivot, p=1.2, seed=12, brown=1.4, char=0.2, edge=0.03, PW=20.0, PH=29.0, start=0.0):
    """Mode 2: the book's outer edges burned `depth` cm in (fore-edge, head and foot; never the gutter), holding
    until `start` and healing to clean paper by `pivot` seconds. side: 'R' (fore-edge at u = PW) or 'L'."""
    b = np.zeros(16)
    b[:] = (1.0, PW, PH, start, depth, p, 0.35, 0.6, seed, brown, char, edge, 2.0, 1.0, pivot, 1.0 if side == 'R' else -1.0)
    return b


def sweep_params(direction, t_start, dur, span=(-4.0, 34.0), seed=31, brown=1.6, char=0.3, edge=0.035):
    """Mode 3: an ember edge sweeping across the page along `direction` (page cm, unit), from span[0] to span[1]
    (projections of page points on the direction) over dur seconds, leaving parchment behind it; ahead of the
    edge is the shot before (the hole)."""
    b = np.zeros(16)
    d = np.asarray(direction, np.float64)
    d = d / np.linalg.norm(d)
    b[:] = (1.0, d[0], d[1], t_start, span[0], span[1], 0.45, 0.35, seed, brown, char, edge, 3.0, dur, 0.0, 1.0)
    return b


def hold_params(origin, t_start, r_max, tau, amp=0.45, freq=0.35, seed=4, brown=2.6, char=0.32, edge=0.035, lead=1.3):
    """Mode 4: burn open from a point, the hole growing quickly and easing out toward r_max (the frame's edge),
    where the burnt rim stays, glowing like a hearth's embers."""
    b = np.zeros(16)
    b[:] = (1.0, origin[0], origin[1], t_start, r_max, tau, amp, freq, seed, brown, char, edge, 4.0, lead, 0.0, 1.0)
    return b


def params(origin, t_start, speed=1.2, p=1.6, amp=0.35, freq=0.45, seed=1, brown=1.6, char=0.22, edge=0.06,
           mode=0, lead=0.8, pivot=0.0, aspect=1.0):
    b = np.zeros(16)
    b[:] = (1.0, origin[0], origin[1], t_start, speed, p, amp, freq, seed, brown, char, edge, mode, lead, pivot,
            aspect)
    return b


@njit(cache=True)
def radius(bf, t):
    """Front radius (cm) at time t (negative before the hole opens)."""
    dt = t - bf[3]
    if bf[12] > 3.5:
        if dt <= 0.0:
            return -1.0 + dt * 0.1
        return bf[4] * (1.0 - math.exp(-dt / bf[5]))
    if bf[12] > 0.5:
        dt = bf[14] - t          # heal: time runs backward from the pivot
    if dt <= 0.0:
        return -1.0 + dt * 0.1
    return bf[4] * dt ** bf[5]


@njit(cache=True)
def dist(bf, u, v):
    """The ragged distance from the origin (cm)."""
    du = u - bf[1]
    dv = (v - bf[2]) / bf[15]
    r = math.sqrt(du * du + dv * dv)
    f = bf[7]
    s = int(bf[8])
    n1 = fbm(u * f, v * f, 301 + s, 3, 2.0, 0.5)
    n2 = gnoise(u * f * 7.0, v * f * 7.0, 303 + s)
    n3 = gnoise(u * f * 26.0, v * f * 26.0, 305 + s)
    return r * (1.0 + bf[6] * n1) + 0.3 * bf[6] * n1 + 0.11 * n2 + 0.035 * n3


@njit(cache=True)
def field(bf, u, v, t):
    """(brown, char, hole, edge, pool) at page cm (u, v), time t."""
    if bf[0] <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if bf[12] > 2.5 and bf[12] < 3.5:
        return sweep_field(bf, u, v, t)
    if bf[12] > 1.5 and bf[12] < 2.5:
        return edge_field(bf, u, v, t)
    R = radius(bf, t)
    d = dist(bf, u, v)
    # heat runs ahead: the browning disc grows from `lead` seconds before the hole
    dt = t - bf[3] if (bf[12] < 0.5 or bf[12] > 3.5) else bf[14] - t
    heat = min(max((dt + bf[13]) / max(bf[13], 1e-3), 0.0), 1.0)
    Rb = max(R, 0.0) + bf[9] * heat * (0.6 + 0.4 * heat)
    sdf = d - max(R, -0.5)
    brown = 0.0
    if Rb > 0.0 and d < Rb:
        x = 1.0 - (d - max(R, 0.0)) / max(Rb - max(R, 0.0), 1e-3)
        x = min(max(x, 0.0), 1.0)
        brown = x * x * (0.55 + 0.45 * heat)
    char = 0.0
    hole = 0.0
    edge = 0.0
    pool = 0.0
    if R > 0.0:
        w = bf[10]
        if sdf < w:
            x = 1.0 - max(sdf, 0.0) / w
            char = x ** 0.7
        a = 0.012
        hole = min(max(0.5 - sdf / a, 0.0), 1.0)
        we = bf[11]
        if sdf > -0.02 and sdf < 5.0 * we:
            # the burning line is uneven: some stretches blaze, some are only a dull red thread
            hot = 0.5 + 0.5 * gnoise(u * 5.0 - t * 0.8, v * 5.0 + t * 0.6, 311 + int(bf[8]))
            hot = max(hot, 0.0) ** 1.6
            spark = gnoise(u * 38.0, v * 38.0 + t * 2.5, 313)
            e = math.exp(-((sdf - 0.25 * we) / we) ** 2)
            edge = e * (0.12 + 1.1 * hot) * (1.0 + 1.2 * max(spark - 0.35, 0.0))
        # glowing threads in the fresh char, fading as it cools
        if sdf > 0.0 and sdf < w:
            cr = gnoise(u * 22.0, v * 22.0, 317 + int(bf[8]))
            edge += 0.25 * max(cr - 0.45, 0.0) * (1.0 - sdf / w) ** 2
        # the embers' light on the paper around the edge
        pool = math.exp(-max(sdf, 0.0) / 0.5) * (1.0 - hole)
    return brown, char, hole, edge, pool


@njit(cache=True)
def edge_field(bf, u, v, t):
    """Mode 2: scorched outer edges that heal. The burned margin's depth shrinks to nothing at the pivot; no
    glowing edge (the fire is long out), only char, a torn margin and the brown of old heat."""
    PW, PH = bf[1], bf[2]
    fe = (PW - u) if bf[15] > 0 else u
    d = min(fe, min(v, PH - v))
    s = int(bf[8])
    f = bf[7]
    d += 0.45 * fbm(u * f, v * f, 331 + s, 3, 2.0, 0.5) + 0.08 * gnoise(u * f * 8.0, v * f * 8.0, 333 + s) + 0.3
    k = min(max((bf[14] - t) / max(bf[14] - bf[3], 1e-3), 0.0), 1.0)
    R = bf[4] * k ** bf[5]
    if R <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    sdf = d - R
    brown = 0.0
    wb = bf[9] * min(R / 0.6, 1.0)
    if sdf < wb:
        x = 1.0 - max(sdf, 0.0) / max(wb, 1e-3)
        brown = x * x
    char = 0.0
    w = bf[10] * min(R / 0.4, 1.0)
    if sdf < w:
        char = (1.0 - max(sdf, 0.0) / max(w, 1e-3)) ** 0.7
    # where the margin is burned away we see the scorched leaf beneath, not a hole: render it as deep char
    gone = min(max(0.5 - sdf / 0.012, 0.0), 1.0)
    char = max(char, 0.85 * gone)
    return brown, char, 0.0, 0.0, 0.0


@njit(cache=True)
def sweep_field(bf, u, v, t):
    """Mode 3: an ember edge sweeps across; behind it the parchment (scorched fresh, cooling to clean), ahead of
    it the hole onto the shot before."""
    x = (t - bf[3]) / max(bf[13], 1e-3)
    x = min(max(x, 0.0), 1.0)
    front = bf[4] + (bf[5] - bf[4]) * (x * x * (3.0 - 2.0 * x))
    s = int(bf[8])
    f = bf[7]
    d = u * bf[1] + v * bf[2]
    d += 0.9 * fbm(u * f, v * f, 341 + s, 3, 2.0, 0.5) + 0.12 * gnoise(u * f * 7.0, v * f * 7.0, 343 + s) + \
        0.035 * gnoise(u * f * 26.0, v * f * 26.0, 345 + s)
    sdf = front - d                      # > 0 behind the edge (parchment), < 0 ahead (the shot before)
    hole = min(max(0.5 - sdf / 0.012, 0.0), 1.0)
    char = 0.0
    brown = 0.0
    edge = 0.0
    pool = 0.0
    if sdf > -0.02:
        w = bf[10]
        if sdf < w:
            char = (1.0 - max(sdf, 0.0) / w) ** 0.8
        if sdf < bf[9]:
            y = 1.0 - max(sdf, 0.0) / bf[9]
            brown = y * y
        we = bf[11]
        if sdf < 5.0 * we:
            hot = 0.5 + 0.5 * gnoise(u * 5.0 - t * 0.8, v * 5.0 + t * 0.6, 351 + s)
            hot = max(hot, 0.0) ** 1.6
            e = math.exp(-((sdf - 0.25 * we) / we) ** 2)
            edge = e * (0.15 + 1.1 * hot)
        pool = math.exp(-max(sdf, 0.0) / 0.5) * (1.0 - hole)
    return brown, char, hole, edge, pool


# ============================================================ v2 (PAGES-C) ===
# The book's paper burns, v2 (PAGES-C, 28 Sep; review/C_PAGES.md #1). A v2 block is 20 floats: the 16 above, then
# 16 = 1 (the v2 flag), 17 = raggedness gain. v1 blocks (16 floats; MAP-L2's road.py) keep `field` unchanged.
# The front is torn at every scale down to the fibre with an ABSOLUTE raggedness (so a small hole is never born round:
# the paper breaks through at a few points and they merge); behind the hole's edge a thin beaded ember line with hot
# spots that crawl, a smouldering band, a crinkled char band of uneven width whose curled lip catches the fire, ember
# flecks dying in the fresh char, then the toasted scorch and clean paper.

def v2(b, rag=1.0, creep=0.0):
    """Extend a 16-float block to v2 (creep: a hold's slow drift outward after it rests, cm/s^2)."""
    out = np.zeros(20)
    out[:16] = b
    out[16] = 1.0
    out[17] = rag
    out[18] = creep
    return out


@njit(cache=True)
def is_v2(bf):
    return bf.shape[0] > 17 and bf[16] > 0.5


@njit(cache=True)
def radius2(bf, t):
    """v2 radius: a hold (mode 4) may creep on outward after it has eased to rest (bf[18], cm/s^2), so a rim that
    glowed at the frame's edge leaves it (C5: 'for a moment it is only light')."""
    R = radius(bf, t)
    if bf.shape[0] > 18 and bf[12] > 3.5 and bf[18] > 0.0:
        x = t - bf[3] - 2.0 * bf[5]
        if x > 0.0:
            R += bf[18] * x * x
    return R


@njit(cache=True)
def _rag(u, v, s, k):
    """Absolute raggedness of a paper burn front (cm), amplitude falling with scale down to the fibre."""
    return k * (0.26 * gnoise(u * 2.6, v * 2.6, 361 + s) + 0.11 * gnoise(u * 9.0, v * 9.0, 363 + s) +
                0.045 * gnoise(u * 27.0, v * 27.0, 365 + s) + 0.018 * gnoise(u * 70.0, v * 70.0, 367 + s) +
                0.008 * gnoise(u * 170.0, v * 170.0, 369 + s))


@njit(cache=True)
def dist2(bf, u, v):
    du = u - bf[1]
    dv = (v - bf[2]) / bf[15]
    r = math.sqrt(du * du + dv * dv)
    s = int(bf[8])
    n1 = fbm(u * bf[7], v * bf[7], 301 + s, 3, 2.0, 0.5)
    return r * (1.0 + bf[6] * n1) + 0.3 * bf[6] * n1 + _rag(u, v, s, bf[17])


@njit(cache=True)
def _bands(bf, u, v, t, sdf, w, embers, still):
    """The structure outside a hole's edge (sdf > 0 is paper): (char, hole, edge, lip, fleck, crk)."""
    s = int(bf[8])
    wv = max(w * (0.7 + 0.6 * (0.5 + gnoise(u * 3.1, v * 3.1, 371 + s))), 1e-4)   # the char band's width wanders
    char = 0.0
    if sdf < wv:
        x = (max(sdf, 0.0) - 0.55 * wv) / (0.45 * wv)
        x = min(max(x, 0.0), 1.0)
        char = 1.0 - x * x * (3.0 - 2.0 * x)
    hole = min(max(0.5 - sdf / 0.008, 0.0), 1.0)
    # crinkle: carbon wrinkled across the band (stretched along the edge by the noise's anisotropy of scale)
    crk = 0.5 + 0.5 * gnoise(u * 34.0, v * 34.0, 381 + s) + 0.3 * gnoise(u * 90.0, v * 90.0, 383 + s)
    crk = min(max(crk, 0.0), 1.0)
    # the curled lip at the hole's edge catches the fire
    lip = math.exp(-((sdf - 0.02) / 0.014) ** 2) * (0.55 + 0.9 * max(gnoise(u * 18.0, v * 18.0, 373 + s), 0.0))
    edge = 0.0
    fleck = 0.0
    if embers > 0.0 and sdf > -0.01:
        hot = 0.5 + 0.5 * gnoise(u * 6.0 - t * 0.9, v * 6.0 + t * 0.7, 311 + s)
        hot = max(hot, 0.0) ** 1.6
        bead = max(gnoise(u * 24.0 + t * 1.3, v * 24.0 - t * 0.8, 375 + s) + 0.18, 0.0) * 2.2
        core = math.exp(-((sdf - 0.006) / 0.0075) ** 2)
        halo = math.exp(-max(sdf, 0.0) / 0.05)
        br = 1.0 - 0.45 * still * (0.5 + 0.5 * math.sin(t * 1.7 + u * 0.9 + v * 0.4))     # embers at rest breathe
        edge = (core * (0.25 + 1.6 * hot) * (0.45 + bead) + 0.3 * halo * hot) * br * embers
        if sdf > 0.0 and sdf < wv:
            fn = gnoise(u * 60.0, v * 60.0, 377 + s)
            tw = 0.5 + 0.5 * gnoise(u * 11.0, t * 2.2, 379 + s)
            fleck = max(fn - 0.3, 0.0) * 6.0 * math.exp(-sdf / 0.09) * (0.35 + 0.65 * tw) * embers
    return char, hole, edge, lip, fleck, crk


@njit(cache=True)
def sweep_front(bf, t):
    x = (t - bf[3]) / max(bf[13], 1e-3)
    x = min(max(x, 0.0), 1.0)
    return bf[4] + (bf[5] - bf[4]) * (x * x * (3.0 - 2.0 * x))


@njit(cache=True)
def sweep_d(bf, u, v):
    """The sweep's ragged distance along its direction (the front is the level set sweep_d = sweep_front)."""
    s = int(bf[8])
    return u * bf[1] + v * bf[2] + 0.9 * fbm(u * bf[7], v * bf[7], 341 + s, 3, 2.0, 0.5) + 1.6 * _rag(u, v, s, bf[17])


@njit(cache=True)
def sweep_pt(bf, lat, t):
    """(u, v) of the sweep front at lateral position `lat` (cm along the front) at time t (bisection)."""
    dx, dy = bf[1], bf[2]
    ex, ey = -dy, dx
    front = sweep_front(bf, t)
    lo = front - 5.0
    hi = front + 5.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if sweep_d(bf, lat * ex + mid * dx, lat * ey + mid * dy) < front:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return lat * ex + mid * dx, lat * ey + mid * dy


@njit(cache=True)
def field2(bf, u, v, t):
    """v2: (brown, char, hole, edge, pool, lip, fleck, crk) at page cm (u, v), time t."""
    z = 0.0
    if bf[0] <= 0.0:
        return z, z, z, z, z, z, z, z
    s = int(bf[8])
    if bf[12] > 2.5 and bf[12] < 3.5:                  # sweep
        x = (t - bf[3]) / max(bf[13], 1e-3)
        x = min(max(x, 0.0), 1.0)
        front = sweep_front(bf, t)
        d = sweep_d(bf, u, v)
        sdf = front - d
        if sdf < -0.02:
            return z, z, 1.0, z, z, z, z, z
        brown = 0.0
        if sdf < bf[9]:
            y = 1.0 - max(sdf, 0.0) / bf[9]
            brown = y * y
        mv = 1.0 if x < 1.0 else 0.0
        char, hole, edge, lip, fleck, crk = _bands(bf, u, v, t, sdf, bf[10], 1.0, 1.0 - mv)
        pool = math.exp(-max(sdf, 0.0) / 0.5) * (1.0 - hole)
        return brown, char, hole, edge, pool, lip, fleck, crk
    if bf[12] > 1.5 and bf[12] < 2.5:                  # scorched outer edges that heal
        PW, PH = bf[1], bf[2]
        fe = (PW - u) if bf[15] > 0 else u
        d = min(fe, min(v, PH - v))
        d += 0.45 * fbm(u * bf[7], v * bf[7], 331 + s, 3, 2.0, 0.5) + 0.8 * _rag(u, v, s, bf[17]) + 0.3
        k = min(max((bf[14] - t) / max(bf[14] - bf[3], 1e-3), 0.0), 1.0)
        R = bf[4] * k ** bf[5]
        if R <= 0.0:
            return z, z, z, z, z, z, z, z
        sdf = d - R
        brown = 0.0
        wb = bf[9] * min(R / 0.6, 1.0)
        if sdf < wb:
            x = 1.0 - max(sdf, 0.0) / max(wb, 1e-3)
            brown = x * x
        char, hole, edge, lip, fleck, crk = _bands(bf, u, v, t, sdf, bf[10] * min(R / 0.4, 1.0), 0.0, 1.0)
        # where the margin is burned away we see the scorched leaf beneath (deep char), not a hole
        char = max(char, 0.9 * hole)
        return brown, char, 0.0, 0.0, 0.0, lip * (1.0 - hole), 0.0, crk
    R = radius2(bf, t)
    d = dist2(bf, u, v)
    dt = t - bf[3] if (bf[12] < 0.5 or bf[12] > 3.5) else bf[14] - t
    heat = min(max((dt + bf[13]) / max(bf[13], 1e-3), 0.0), 1.0)
    Rb = max(R, 0.0) + bf[9] * heat * (0.6 + 0.4 * heat)
    sdf = d - max(R, -0.5)
    brown = 0.0
    if Rb > 0.0 and d < Rb:
        x = 1.0 - (d - max(R, 0.0)) / max(Rb - max(R, 0.0), 1e-3)
        x = min(max(x, 0.0), 1.0)
        brown = x * x * (0.55 + 0.45 * heat)
    if R <= 0.0:
        return brown, z, z, z, z, z, z, z
    still = 0.0
    if bf[12] > 3.5:                                   # hold: the front eases to rest at the frame's edge
        still = min(max((dt / max(bf[5], 1e-3) - 2.0) / 2.0, 0.0), 1.0)
    char, hole, edge, lip, fleck, crk = _bands(bf, u, v, t, sdf, bf[10], 1.0, still)
    pool = math.exp(-max(sdf, 0.0) / 0.5) * (1.0 - hole)
    return brown, char, hole, edge, pool, lip, fleck, crk


@njit(cache=True)
def front_r(bf, ang, t):
    """Radius (cm) of the radial front along the ray at angle `ang` from the origin at time t (bisection on dist2);
    -1 before the hole opens. Used to seed the smoke on the burning edge."""
    R = radius2(bf, t)
    if R <= 0.0:
        return -1.0
    ca = math.cos(ang)
    sa = math.sin(ang) * bf[15]
    lo = 0.0
    hi = 3.0 * R + 1.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if dist2(bf, bf[1] + ca * mid, bf[2] + sa * mid) < R:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
