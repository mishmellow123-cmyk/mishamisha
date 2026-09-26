"""Deterministic 2-D gradient noise + fBm (numba), evaluated at map coordinates so any tile at any
resolution agrees with every other."""
import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline='always')
def _hash(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


@njit(cache=True, inline='always')
def _grad(ix, iy, seed, dx, dy):
    h = _hash(ix, iy, seed)
    a = (h & 0xFFFF) / 65536.0 * 6.283185307179586
    return math.cos(a) * dx + math.sin(a) * dy


@njit(cache=True, inline='always')
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True)
def gnoise(x, y, seed):
    """Gradient noise in about [-0.7, 0.7]."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    u = _fade(fx)
    v = _fade(fy)
    n00 = _grad(ix, iy, seed, fx, fy)
    n10 = _grad(ix + 1, iy, seed, fx - 1.0, fy)
    n01 = _grad(ix, iy + 1, seed, fx, fy - 1.0)
    n11 = _grad(ix + 1, iy + 1, seed, fx - 1.0, fy - 1.0)
    a = n00 + u * (n10 - n00)
    b = n01 + u * (n11 - n01)
    return a + v * (b - a)


@njit(cache=True)
def vnoise(x, y, seed):
    """Value noise in [0, 1]."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    u = _fade(fx)
    v = _fade(fy)
    a = (_hash(ix, iy, seed) & 0xFFFFFF) / 16777216.0
    b = (_hash(ix + 1, iy, seed) & 0xFFFFFF) / 16777216.0
    c = (_hash(ix, iy + 1, seed) & 0xFFFFFF) / 16777216.0
    d = (_hash(ix + 1, iy + 1, seed) & 0xFFFFFF) / 16777216.0
    return (a + u * (b - a)) + v * ((c + u * (d - c)) - (a + u * (b - a)))


@njit(cache=True)
def fbm(x, y, seed, octaves, lac, gain):
    s = 0.0
    amp = 1.0
    norm = 0.0
    f = 1.0
    for o in range(octaves):
        s += amp * gnoise(x * f, y * f, seed + o * 1013)
        norm += amp
        amp *= gain
        f *= lac
    return s / norm


@njit(cache=True, parallel=True)
def fbm_grid(x0, y0, dx, dy, H, W, freq, seed, octaves, lac, gain):
    """fBm on a pixel grid: pixel (i, j) centre at (x0 + (j+.5) dx, y0 - (i+.5) dy)."""
    out = np.empty((H, W), np.float32)
    for i in prange(H):
        y = (y0 - (i + 0.5) * dy) * freq
        for j in range(W):
            x = (x0 + (j + 0.5) * dx) * freq
            out[i, j] = fbm(x, y, seed, octaves, lac, gain)
    return out


@njit(cache=True, parallel=True)
def warped_grid(x0, y0, dx, dy, H, W, freq, seed, octaves, warp, wfreq):
    """Domain-warped fBm (organic blotches)."""
    out = np.empty((H, W), np.float32)
    for i in prange(H):
        yy = y0 - (i + 0.5) * dy
        for j in range(W):
            xx = x0 + (j + 0.5) * dx
            qx = fbm(xx * wfreq, yy * wfreq, seed + 77, 3, 2.0, 0.5)
            qy = fbm(xx * wfreq + 5.2, yy * wfreq + 1.3, seed + 91, 3, 2.0, 0.5)
            out[i, j] = fbm((xx + warp * qx) * freq, (yy + warp * qy) * freq, seed, octaves, 2.0, 0.5)
    return out


@njit(cache=True, parallel=True)
def fibre_grid(x0, y0, dx, dy, H, W, freq, seed, stretch):
    """Anisotropic streaky noise: fibres whose direction wanders slowly across the sheet."""
    out = np.empty((H, W), np.float32)
    for i in prange(H):
        yy = y0 - (i + 0.5) * dy
        for j in range(W):
            xx = x0 + (j + 0.5) * dx
            a = 3.0 * gnoise(xx * 0.05, yy * 0.05, seed + 5)
            ca = math.cos(a)
            sa = math.sin(a)
            u = (ca * xx + sa * yy) * freq
            v = (-sa * xx + ca * yy) * freq * stretch
            out[i, j] = 0.6 * gnoise(u, v, seed) + 0.4 * gnoise(u * 2.1, v * 2.3, seed + 3)
    return out


def wobble1d(s, seed, wavelength, amp):
    """Smooth 1-D hand tremor along arc length s (numpy, cheap): sum of a few random sines."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(s, dtype=np.float64)
    tot = 0.0
    for k in range(4):
        f = 2 * np.pi / wavelength * (0.6 + 0.8 * rng.random()) * (1.7 ** k)
        a = 1.0 / (1.6 ** k)
        out += a * np.sin(f * s + rng.random() * 2 * np.pi)
        tot += a
    return out / tot * amp
