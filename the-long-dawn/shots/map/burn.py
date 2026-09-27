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
    return r * (1.0 + bf[6] * n1) + 0.25 * bf[6] * n1 + 0.05 * n2 + 0.012 * n3


@njit(cache=True)
def field(bf, u, v, t):
    """(brown, char, hole, edge, pool) at page cm (u, v), time t."""
    if bf[0] <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
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
        if sdf > -0.02 and sdf < 4.0 * we:
            hot = 0.55 + 0.45 * gnoise(u * 6.0 - t * 0.9, v * 6.0 + t * 0.7, 311 + int(bf[8]))
            spark = gnoise(u * 40.0, v * 40.0 + t * 3.0, 313)
            e = math.exp(-((sdf - 0.3 * we) / we) ** 2)
            edge = e * hot * (1.0 + 0.8 * max(spark - 0.3, 0.0))
        # the embers' light on the paper around the edge
        pool = math.exp(-max(sdf, 0.0) / 0.7) * (1.0 - hole)
    return brown, char, hole, edge, pool
