"""Numba primitives for the ACCORD renderer: math, noise, SDFs, texture sampling."""
import math

import numpy as np
from numba import njit

FM = dict(fastmath=True, cache=True)


@njit(inline='always', **FM)
def clamp(x, a, b):
    return a if x < a else (b if x > b else x)


@njit(inline='always', **FM)
def sstep(a, b, x):
    t = (x - a) / (b - a)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


@njit(inline='always', **FM)
def mix(a, b, t):
    return a + (b - a) * t


# ------------------------------------------------------------------- noise ---

@njit(inline='always', **FM)
def hash2i(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) * (1.0 / 16777215.0)


@njit(inline='always', **FM)
def vnoise2(x, y, seed):
    """Value noise in [0,1], quintic interpolation."""
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    ix = int(ix)
    iy = int(iy)
    ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    a = hash2i(ix, iy, seed)
    b = hash2i(ix + 1, iy, seed)
    c = hash2i(ix, iy + 1, seed)
    d = hash2i(ix + 1, iy + 1, seed)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy


@njit(**FM)
def fbm2(x, y, seed, octaves, lac, gain, fp):
    """Band-limited fbm. fp = pixel footprint in the same units as x,y (fades octaves
    whose wavelength < ~2*fp). Returns ~[-0.5,0.5]."""
    s = 0.0
    amp = 0.5
    freq = 1.0
    norm = 0.0
    for o in range(octaves):
        wl = 1.0 / freq
        w = sstep(1.0 * fp, 3.0 * fp, wl)
        if w <= 0.0:
            break
        s += amp * w * (vnoise2(x * freq + o * 17.3, y * freq - o * 9.7, seed + o) - 0.5)
        norm += amp
        amp *= gain
        freq *= lac
    return s


@njit(inline='always', **FM)
def tex3(n3, x, y, z):
    """Trilinear sample of tileable 3D noise texture n3 (S^3), coords in texels."""
    S = n3.shape[0]
    fx = math.floor(x)
    fy = math.floor(y)
    fz = math.floor(z)
    tx = x - fx
    ty = y - fy
    tz = z - fz
    ix = int(fx) % S
    iy = int(fy) % S
    iz = int(fz) % S
    jx = (ix + 1) % S
    jy = (iy + 1) % S
    jz = (iz + 1) % S
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    tz = tz * tz * (3 - 2 * tz)
    c00 = n3[ix, iy, iz] + (n3[jx, iy, iz] - n3[ix, iy, iz]) * tx
    c10 = n3[ix, jy, iz] + (n3[jx, jy, iz] - n3[ix, jy, iz]) * tx
    c01 = n3[ix, iy, jz] + (n3[jx, iy, jz] - n3[ix, iy, jz]) * tx
    c11 = n3[ix, jy, jz] + (n3[jx, jy, jz] - n3[ix, jy, jz]) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    return c0 + (c1 - c0) * tz


def make_noise3(S=64, seed=7):
    rng = np.random.default_rng(seed)
    return rng.random((S, S, S)).astype(np.float32)


# -------------------------------------------------------------------- SDFs ---

@njit(inline='always', **FM)
def sd_ellipsoid(x, y, z, rx, ry, rz):
    k0 = math.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
    k1 = math.sqrt((x / (rx * rx)) ** 2 + (y / (ry * ry)) ** 2 + (z / (rz * rz)) ** 2)
    if k1 < 1e-9:
        return -min(rx, min(ry, rz))
    return k0 * (k0 - 1.0) / k1


@njit(inline='always', **FM)
def sd_capsule(px, py, pz, ax, ay, az, bx, by, bz, r):
    pax = px - ax
    pay = py - ay
    paz = pz - az
    bax = bx - ax
    bay = by - ay
    baz = bz - az
    h = (pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz + 1e-12)
    h = clamp(h, 0.0, 1.0)
    dx = pax - bax * h
    dy = pay - bay * h
    dz = paz - baz * h
    return math.sqrt(dx * dx + dy * dy + dz * dz) - r


@njit(inline='always', **FM)
def sd_capsule_r(px, py, pz, ax, ay, az, bx, by, bz, ra, rb):
    """Capsule with linearly varying radius (cheap approx of round cone)."""
    pax = px - ax
    pay = py - ay
    paz = pz - az
    bax = bx - ax
    bay = by - ay
    baz = bz - az
    h = (pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz + 1e-12)
    h = clamp(h, 0.0, 1.0)
    dx = pax - bax * h
    dy = pay - bay * h
    dz = paz - baz * h
    return math.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * h)


@njit(inline='always', **FM)
def smin(a, b, k):
    h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


# ---------------------------------------------------------- mip textures ---

@njit(inline='always', **FM)
def _bilerp4(flat, off, n, u, v, out):
    """u,v in texel units of this level (0..n). Accumulates 4-channel bilinear sample into out (weighted)."""
    x = u - 0.5
    y = v - 0.5
    fx = math.floor(x)
    fy = math.floor(y)
    tx = x - fx
    ty = y - fy
    ix = int(fx)
    iy = int(fy)
    if ix < 0:
        ix = 0
        tx = 0.0
    if iy < 0:
        iy = 0
        ty = 0.0
    if ix > n - 2:
        ix = n - 2
        tx = 1.0
    if iy > n - 2:
        iy = n - 2
        ty = 1.0
    i00 = off + iy * n + ix
    i01 = i00 + 1
    i10 = i00 + n
    i11 = i10 + 1
    w00 = (1 - tx) * (1 - ty)
    w01 = tx * (1 - ty)
    w10 = (1 - tx) * ty
    w11 = tx * ty
    for c in range(4):
        out[c] = flat[i00, c] * w00 + flat[i01, c] * w01 + flat[i10, c] * w10 + flat[i11, c] * w11


@njit(**FM)
def tex_sample(flat, offs, sizes, R, x, y, lod, out, tmp):
    """Trilinear sample of the table map at world (x,y). lod = mip level (float)."""
    nl = offs.shape[0]
    if lod < 0.0:
        lod = 0.0
    if lod > nl - 1.001:
        lod = nl - 1.001
    l0 = int(lod)
    fr = lod - l0
    n0 = sizes[l0]
    u = (x + R) / (2.0 * R) * n0
    v = (R - y) / (2.0 * R) * n0
    _bilerp4(flat, offs[l0], n0, u, v, out)
    if fr > 1e-3:
        n1 = sizes[l0 + 1]
        _bilerp4(flat, offs[l0 + 1], n1, u * 0.5, v * 0.5, tmp)
        for c in range(4):
            out[c] = out[c] * (1 - fr) + tmp[c] * fr


@njit(inline='always', **FM)
def grid_sample(g, x0, y0, cell, x, y):
    """Bilinear sample of a 2D grid g covering [x0, x0+n*cell) x [y0, ...), row = y index."""
    n = g.shape[0]
    m = g.shape[1]
    u = (x - x0) / cell - 0.5
    v = (y - y0) / cell - 0.5
    if u < 0 or v < 0 or u > m - 1.001 or v > n - 1.001:
        return 0.0
    iu = int(u)
    iv = int(v)
    tu = u - iu
    tv = v - iv
    a = g[iv, iu]
    b = g[iv, iu + 1]
    c = g[iv + 1, iu]
    d = g[iv + 1, iu + 1]
    return (a * (1 - tu) + b * tu) * (1 - tv) + (c * (1 - tu) + d * tu) * tv
