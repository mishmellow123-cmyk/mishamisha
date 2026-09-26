"""The cloud sea: a billowy (cauliflower) top surface near z = 0 (numba).

Heights = long swells + multi-scale Worley 'puffs' (a rounded dome per cell), domain-warped.
"""
import math

import numpy as np
from numba import njit, prange

from .noise import fbm2, smoothstep, hash2

CLOUD_Z = 0.0


@njit(inline='always', cache=True)
def _lod(scale, fp):
    return min(max(math.log2(max(scale / max(fp * 2.0, 1e-3), 1.0)), 1.0), 8.0)


@njit(cache=True)
def worley(x, y, seed):
    ix = math.floor(x)
    iy = math.floor(y)
    best = 1e9
    for i in range(-1, 2):
        for j in range(-1, 2):
            cx = int(ix) + i
            cy = int(iy) + j
            h = hash2(cx, cy, seed)
            px = cx + 0.1 + 0.8 * ((h & 0xFFFF) / 65535.0)
            py = cy + 0.1 + 0.8 * ((h >> 16) / 65535.0)
            d = (x - px) ** 2 + (y - py) ** 2
            if d < best:
                best = d
    return math.sqrt(best)


@njit(inline='always', cache=True)
def puff(x, y, scale, seed):
    d = worley(x / scale, y / scale, seed)
    q = max(1.0 - d / 0.78, 0.0)
    return q ** 0.55


@njit(cache=True)
def cloud_height(x, y, fp):
    # warp so the puffs are not on a lattice-looking field
    wx = 120.0 * fbm2(x / 900.0 + 1.1, y / 900.0, 2.0, 311)
    wy = 120.0 * fbm2(x / 900.0 - 3.7, y / 900.0 + 2.3, 2.0, 312)
    xx = x + wx
    yy = y + wy
    # long swells of the deck
    a = 24.0 * fbm2(x / 3000.0 + 0.7, y / 3000.0 - 0.2, _lod(3000.0, fp), 301)
    # cauliflower lumps at three scales (faded out below the sample footprint)
    b = 0.0
    if fp < 180.0:
        b += 34.0 * (puff(xx, yy, 460.0, 302) - 0.45) * smoothstep(180.0, 60.0, fp)
    if fp < 50.0:
        b += 13.0 * (puff(xx * 1.03 + 17.0, yy, 140.0, 303) - 0.45) * smoothstep(50.0, 18.0, fp)
    if fp < 16.0:
        b += 4.5 * (puff(xx, yy * 1.05 - 9.0, 42.0, 304) - 0.45) * smoothstep(16.0, 6.0, fp)
    return CLOUD_Z + a + b


@njit(parallel=True, cache=True)
def eval_points(X, Y, FP):
    n = X.shape[0]
    out = np.empty(n)
    for k in prange(n):
        out[k] = cloud_height(X[k], Y[k], FP[k])
    return out


@njit(parallel=True, cache=True)
def raster(nx, ny, x0, y0, dx):
    g = np.empty((ny, nx), np.float64)
    for j in prange(ny):
        for i in range(nx):
            g[j, i] = cloud_height(x0 + i * dx, y0 + j * dx, dx)
    return g
