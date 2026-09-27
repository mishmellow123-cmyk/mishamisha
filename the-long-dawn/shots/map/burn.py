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


def edge_params(side, depth, pivot, p=1.2, seed=12, brown=1.4, char=0.2, edge=0.03, PW=20.0, PH=29.0):
    """Mode 2: the book's outer edges burned `depth` cm in (fore-edge, head and foot; never the gutter), healing
    to clean paper by `pivot` seconds. side: 'R' (fore-edge at u = PW) or 'L' (fore-edge at u = 0)."""
    b = np.zeros(16)
    speed = depth / max(pivot, 1e-3) ** p
    b[:] = (1.0, PW, PH, 0.0, speed, p, 0.35, 0.6, seed, brown, char, edge, 2.0, 1.0, pivot, 1.0 if side == 'R' else -1.0)
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
    if bf[12] > 1.5:
        return edge_field(bf, u, v, t)
    R = radius(bf, t)
    d = dist(bf, u, v)
    # heat runs ahead: the browning disc grows from `lead` seconds before the hole
    dt = t - bf[3] if bf[12] < 0.5 else bf[14] - t
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
    R = bf[4] * max(bf[14] - t, 0.0) ** bf[5]
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
