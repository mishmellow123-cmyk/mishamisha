"""Numba noise for the MOUNTAIN department (hash-based gradient noise, fBm, ridged)."""
import math

import numpy as np
from numba import njit


@njit(inline='always', cache=True)
def ihash(a):
    a = a & 0xFFFFFFFF
    a = ((a ^ 61) ^ (a >> 16)) & 0xFFFFFFFF
    a = (a + (a << 3)) & 0xFFFFFFFF
    a = a ^ (a >> 4)
    a = (a * 0x27D4EB2D) & 0xFFFFFFFF
    a = a ^ (a >> 15)
    return a


@njit(inline='always', cache=True)
def hash2(i, j, seed):
    return ihash((i * 73856093) ^ (j * 19349663) ^ (seed * 83492791))


@njit(inline='always', cache=True)
def hash3(i, j, k, seed):
    return ihash((i * 73856093) ^ (j * 19349663) ^ (k * 83492791) ^ (seed * 2654435761))


@njit(inline='always', cache=True)
def h01(i, j, seed):
    return hash2(i, j, seed) * (1.0 / 4294967296.0)


@njit(inline='always', cache=True)
def fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(inline='always', cache=True)
def _g2(h, x, y):
    a = (h & 255) * (2.0 * math.pi / 256.0)
    return math.cos(a) * x + math.sin(a) * y


@njit(cache=True)
def gnoise2(x, y, seed):
    """2D gradient noise with 256 gradient directions, ~[-1, 1]."""
    fx = math.floor(x)
    fy = math.floor(y)
    i = int(fx)
    j = int(fy)
    rx = x - fx
    ry = y - fy
    u = fade(rx)
    v = fade(ry)
    n00 = _g2(hash2(i, j, seed), rx, ry)
    n10 = _g2(hash2(i + 1, j, seed), rx - 1.0, ry)
    n01 = _g2(hash2(i, j + 1, seed), rx, ry - 1.0)
    n11 = _g2(hash2(i + 1, j + 1, seed), rx - 1.0, ry - 1.0)
    a = n00 + (n10 - n00) * u
    b = n01 + (n11 - n01) * u
    return 1.41 * (a + (b - a) * v)


@njit(cache=True)
def fbm2(x, y, octaves, seed, lac=2.0, gain=0.5):
    s = 0.0
    a = 1.0
    norm = 0.0
    f = 1.0
    n = int(math.ceil(octaves))
    for o in range(n):
        w = 1.0
        if o == n - 1:
            w = octaves - (n - 1) if octaves < n else 1.0
        # rotate each octave to kill lattice alignment
        ca = math.cos(0.5 * o)
        sa = math.sin(0.5 * o)
        xx = (x * ca - y * sa) * f
        yy = (x * sa + y * ca) * f
        s += w * a * gnoise2(xx, yy, seed + 31 * o)
        norm += w * a
        a *= gain
        f *= lac
    return s / max(norm, 1e-9)


@njit(cache=True)
def ridged2(x, y, octaves, seed, lac=2.0, gain=0.5, sharp=2.0):
    """Ridged multifractal in [0, 1] (1 = crest). Higher octaves are damped where the
    lower ones are low (valleys smoother than crests)."""
    s = 0.0
    a = 1.0
    norm = 0.0
    f = 1.0
    wgt = 1.0
    n = int(math.ceil(octaves))
    for o in range(n):
        w = 1.0
        if o == n - 1:
            w = octaves - (n - 1) if octaves < n else 1.0
        ca = math.cos(0.7 * o + 0.3)
        sa = math.sin(0.7 * o + 0.3)
        xx = (x * ca - y * sa) * f
        yy = (x * sa + y * ca) * f
        r = 1.0 - abs(gnoise2(xx, yy, seed + 57 * o))
        r = r ** sharp
        r *= wgt
        wgt = min(max(r * 1.5, 0.0), 1.0)
        s += w * a * r
        norm += w * a
        a *= gain
        f *= lac
    return s / max(norm, 1e-9)


@njit(inline='always', cache=True)
def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def vnoised(x, y, seed):
    """Value noise in [-1, 1] with analytic derivatives (quintic). Returns (v, dv/dx, dv/dy)."""
    fx = math.floor(x)
    fy = math.floor(y)
    i = int(fx)
    j = int(fy)
    ux = x - fx
    uy = y - fy
    u = ux * ux * ux * (ux * (ux * 6.0 - 15.0) + 10.0)
    v = uy * uy * uy * (uy * (uy * 6.0 - 15.0) + 10.0)
    du = 30.0 * ux * ux * (ux * (ux - 2.0) + 1.0)
    dv = 30.0 * uy * uy * (uy * (uy - 2.0) + 1.0)
    a = h01(i, j, seed)
    b = h01(i + 1, j, seed)
    c = h01(i, j + 1, seed)
    d = h01(i + 1, j + 1, seed)
    k1 = b - a
    k2 = c - a
    k4 = a - b - c + d
    val = -1.0 + 2.0 * (a + k1 * u + k2 * v + k4 * u * v)
    return val, 2.0 * du * (k1 + k4 * v), 2.0 * dv * (k2 + k4 * u)


@njit(cache=True)
def dfbm(x, y, octaves, seed, damp=1.0):
    """iq-style 'eroded' fBm: each octave is damped by the accumulated slope, so crests are
    sharp and valleys smooth. Returns roughly [-1, 1]."""
    a = 0.0
    b = 1.0
    sx = 0.0
    sy = 0.0
    px = x
    py = y
    norm = 0.0
    n = int(math.ceil(octaves))
    for o in range(n):
        w = 1.0
        if o == n - 1:
            w = octaves - (n - 1) if octaves < n else 1.0
        vv, dx, dy = vnoised(px, py, seed + 13 * o)
        sx += dx
        sy += dy
        a += w * b * vv / (1.0 + damp * (sx * sx + sy * sy))
        norm += w * b
        b *= 0.5
        # rotate + scale (iq's m2 = [0.8,-0.6;0.6,0.8] * 2)
        qx = 1.6 * px - 1.2 * py
        qy = 1.2 * px + 1.6 * py
        px = qx
        py = qy
    return a / max(norm, 1e-9) * 1.6
