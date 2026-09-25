"""Numba noise primitives (hash-based, no tables needed at runtime).

All functions are deterministic and seedable. Gradient noise returns roughly [-1, 1].
"""
import math

import numpy as np
from numba import njit

_G2X = np.array([1.0, -1.0, 0.0, 0.0, 0.70710678, -0.70710678, 0.70710678, -0.70710678])
_G2Y = np.array([0.0, 0.0, 1.0, -1.0, 0.70710678, 0.70710678, -0.70710678, -0.70710678])


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
def hash2i(i, j, seed):
    return ihash(i * 73856093 ^ j * 19349663 ^ int(seed) * 83492791)


@njit(inline='always', cache=True)
def hash3i(i, j, k, seed):
    return ihash(i * 73856093 ^ j * 19349663 ^ k * 83492791 ^ int(seed) * 2654435761)


@njit(inline='always', cache=True)
def h01(i, j, seed):
    """Hash of an integer lattice point -> float in [0, 1)."""
    return hash2i(i, j, seed) * (1.0 / 4294967296.0)


@njit(inline='always', cache=True)
def h01_3(i, j, k, seed):
    return hash3i(i, j, k, seed) * (1.0 / 4294967296.0)


@njit(inline='always', cache=True)
def fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(inline='always', cache=True)
def vnoise2(x, y, seed):
    """Value noise in [-1, 1]."""
    fx = math.floor(x)
    fy = math.floor(y)
    i = int(fx)
    j = int(fy)
    u = fade(x - fx)
    v = fade(y - fy)
    a = h01(i, j, seed)
    b = h01(i + 1, j, seed)
    c = h01(i, j + 1, seed)
    d = h01(i + 1, j + 1, seed)
    return 2.0 * (a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v) - 1.0


@njit(inline='always', cache=True)
def gnoise2(x, y, seed):
    """2D gradient (Perlin-style) noise, approx [-1, 1]."""
    fx = math.floor(x)
    fy = math.floor(y)
    i = int(fx)
    j = int(fy)
    rx = x - fx
    ry = y - fy
    u = fade(rx)
    v = fade(ry)
    h00 = hash2i(i, j, seed) & 7
    h10 = hash2i(i + 1, j, seed) & 7
    h01_ = hash2i(i, j + 1, seed) & 7
    h11 = hash2i(i + 1, j + 1, seed) & 7
    n00 = _G2X[h00] * rx + _G2Y[h00] * ry
    n10 = _G2X[h10] * (rx - 1.0) + _G2Y[h10] * ry
    n01 = _G2X[h01_] * rx + _G2Y[h01_] * (ry - 1.0)
    n11 = _G2X[h11] * (rx - 1.0) + _G2Y[h11] * (ry - 1.0)
    nx0 = n00 + (n10 - n00) * u
    nx1 = n01 + (n11 - n01) * u
    return 1.41421356 * (nx0 + (nx1 - nx0) * v)


@njit(inline='always', cache=True)
def _grad3(h, x, y, z):
    h = h & 15
    u = x if h < 8 else y
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    else:
        v = z
    r = u if (h & 1) == 0 else -u
    r += v if (h & 2) == 0 else -v
    return r


@njit(inline='always', cache=True)
def gnoise3(x, y, z, seed):
    """3D gradient noise (Perlin improved-style), approx [-1, 1]."""
    fx = math.floor(x)
    fy = math.floor(y)
    fz = math.floor(z)
    i = int(fx)
    j = int(fy)
    k = int(fz)
    rx = x - fx
    ry = y - fy
    rz = z - fz
    u = fade(rx)
    v = fade(ry)
    w = fade(rz)
    n000 = _grad3(hash3i(i, j, k, seed), rx, ry, rz)
    n100 = _grad3(hash3i(i + 1, j, k, seed), rx - 1.0, ry, rz)
    n010 = _grad3(hash3i(i, j + 1, k, seed), rx, ry - 1.0, rz)
    n110 = _grad3(hash3i(i + 1, j + 1, k, seed), rx - 1.0, ry - 1.0, rz)
    n001 = _grad3(hash3i(i, j, k + 1, seed), rx, ry, rz - 1.0)
    n101 = _grad3(hash3i(i + 1, j, k + 1, seed), rx - 1.0, ry, rz - 1.0)
    n011 = _grad3(hash3i(i, j + 1, k + 1, seed), rx, ry - 1.0, rz - 1.0)
    n111 = _grad3(hash3i(i + 1, j + 1, k + 1, seed), rx - 1.0, ry - 1.0, rz - 1.0)
    nx00 = n000 + (n100 - n000) * u
    nx10 = n010 + (n110 - n010) * u
    nx01 = n001 + (n101 - n001) * u
    nx11 = n011 + (n111 - n011) * u
    nxy0 = nx00 + (nx10 - nx00) * v
    nxy1 = nx01 + (nx11 - nx01) * v
    return 0.95 * (nxy0 + (nxy1 - nxy0) * w)


@njit(inline='always', cache=True)
def fbm2(x, y, octaves, seed):
    """fBm of gradient noise; octaves may be fractional (last octave faded). ~[-1,1]."""
    s = 0.0
    a = 0.5
    norm = 0.0
    n = int(octaves)
    frac = octaves - n
    for o in range(n + 1):
        wgt = a if o < n else a * frac
        if wgt <= 0.0:
            break
        s += wgt * gnoise2(x, y, seed + o * 131)
        norm += a
        # rotate ~37 degrees and scale by 2.03 to break lattice alignment
        nx = 1.6 * x - 1.2 * y
        ny = 1.2 * x + 1.6 * y
        x = nx * 1.015 + 17.13
        y = ny * 1.015 - 9.71
        a *= 0.5
    return s / max(norm, 1e-6)


@njit(inline='always', cache=True)
def fbm3(x, y, z, octaves, seed):
    s = 0.0
    a = 0.5
    norm = 0.0
    n = int(octaves)
    frac = octaves - n
    for o in range(n + 1):
        wgt = a if o < n else a * frac
        if wgt <= 0.0:
            break
        s += wgt * gnoise3(x, y, z, seed + o * 131)
        norm += a
        x = x * 2.03 + 11.7
        y = y * 2.03 - 5.3
        z = z * 2.03 + 3.1
        a *= 0.5
    return s / max(norm, 1e-6)


@njit(inline='always', cache=True)
def ridged2(x, y, octaves, seed, gain=2.0):
    """Musgrave ridged multifractal, returns ~[0, 1+]."""
    s = 0.0
    a = 0.5
    w = 1.0
    norm = 0.0
    n = int(octaves)
    frac = octaves - n
    for o in range(n + 1):
        wgt = 1.0 if o < n else frac
        if wgt <= 0.0:
            break
        r = 1.0 - abs(gnoise2(x, y, seed + o * 131))
        r = r * r
        r *= w
        w = min(max(r * gain, 0.0), 1.0)
        s += wgt * r * a
        norm += a
        nx = 1.6 * x - 1.2 * y
        ny = 1.2 * x + 1.6 * y
        x = nx * 1.015 + 17.13
        y = ny * 1.015 - 9.71
        a *= 0.5
    return s / max(norm, 1e-6)


@njit(inline='always', cache=True)
def smoothstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(inline='always', cache=True)
def clamp(x, a, b):
    return a if x < a else (b if x > b else x)


@njit(inline='always', cache=True)
def mix(a, b, t):
    return a + (b - a) * t
