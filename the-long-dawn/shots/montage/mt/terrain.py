"""Column-coherent heightfield ray marcher + shading helpers.

Camera has no roll and tilt is a lens shift, so every screen column is a vertical plane through
the eye and the hit distance grows monotonically going up a column: march each column once,
bottom row to top row, re-using the previous row's bracket.

Height functions have the signature  h = hfun(x, z, fp, P)  where fp is the pixel footprint in
metres at that distance (for LOD / anti-aliasing) and P is a float64 parameter array.
"""
import math

import numpy as np
from numba import njit, prange

INF = 1e30


@njit(parallel=True, fastmath=True)
def march(hfun, P, C, d0, dmax, rel, kgap, hmax, nbis, out_d):
    """C = cam.params(); writes horizontal hit distance per pixel (INF = sky)."""
    cxw, cyw, czw = C[0], C[1], C[2]
    fx, fz, rx, rz = C[3], C[4], C[5], C[6]
    f, cx, cyy = C[7], C[8], C[9]
    H, W = out_d.shape
    pix = 1.0 / f
    for i in prange(W):
        xo = i + 0.5 - cx
        dxh = fx * f + rx * xo
        dzh = fz * f + rz * xo
        hl = math.sqrt(dxh * dxh + dzh * dzh)
        dxh /= hl
        dzh /= hl
        pixh = 1.0 / hl
        da = d0
        ha = hfun(cxw + dxh * da, czw + dzh * da, da * pix, P)
        missed = False
        for j in range(H - 1, -1, -1):
            if missed:
                out_d[j, i] = INF
                continue
            s = (cyy - (j + 0.5)) * pixh
            if cyw + s * da < ha:
                out_d[j, i] = da
                continue
            d = da
            h = ha
            hit = False
            while d < dmax:
                gap = cyw + s * d - h
                step = rel * d
                g = kgap * gap
                if g > step:
                    step = g
                dn = d + step
                hn = hfun(cxw + dxh * dn, czw + dzh * dn, dn * pix, P)
                yn = cyw + s * dn
                if yn < hn:
                    lo = d
                    hi = dn
                    hlo = h
                    for _k in range(nbis):
                        mid = 0.5 * (lo + hi)
                        hm = hfun(cxw + dxh * mid, czw + dzh * mid, mid * pix, P)
                        if cyw + s * mid < hm:
                            hi = mid
                        else:
                            lo = mid
                            hlo = hm
                    out_d[j, i] = 0.5 * (lo + hi)
                    da = lo
                    ha = hlo
                    hit = True
                    break
                if yn > hmax and s >= 0.0:
                    break
                d = dn
                h = hn
            if not hit:
                missed = True
                out_d[j, i] = INF


@njit(inline='always', fastmath=True)
def ray_of(C, i, j):
    """Normalised world ray direction for (sub)pixel centre (i+0.5, j+0.5) plus its horizontal
    length factor. Returns dx, dy, dz, slope, dxh, dzh."""
    fx, fz, rx, rz = C[3], C[4], C[5], C[6]
    f, cx, cyy = C[7], C[8], C[9]
    xo = i + 0.5 - cx
    dxh = fx * f + rx * xo
    dzh = fz * f + rz * xo
    hl = math.sqrt(dxh * dxh + dzh * dzh)
    vy = cyy - (j + 0.5)
    n = math.sqrt(hl * hl + vy * vy)
    return dxh / n, vy / n, dzh / n, vy / hl, dxh / hl, dzh / hl


@njit(inline='always', fastmath=True)
def normal_at(hfun, P, x, z, e):
    h0 = hfun(x, z, e, P)
    hx = hfun(x + e, z, e, P)
    hz = hfun(x, z + e, e, P)
    nx = -(hx - h0)
    ny = e
    nz = -(hz - h0)
    inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx * inv, ny * inv, nz * inv, h0


@njit(inline='always', fastmath=True)
def soft_shadow(hfun, P, x, y, z, lx, ly, lz, t0, tmax, nsteps, k, fp):
    """iq-style soft shadow toward a directional light; geometric step growth."""
    res = 1.0
    t = t0
    grow = (tmax / t0) ** (1.0 / nsteps)
    for _s in range(nsteps):
        px = x + lx * t
        py = y + ly * t
        pz = z + lz * t
        h = hfun(px, pz, fp + t * 0.004, P)
        dh = py - h
        r = k * dh / t
        if r < res:
            res = r
        if res < 0.0:
            return 0.0
        t *= grow
    if res > 1.0:
        res = 1.0
    return res * res * (3.0 - 2.0 * res)


@njit(inline='always', fastmath=True)
def height_fog_tau(dist, y0, y1, a, b):
    """Optical depth of density a*exp(-b*y) along a straight segment of length dist
    from height y0 to y1."""
    dy = y1 - y0
    e0 = math.exp(-b * y0)
    if abs(dy) < 1e-3:
        return a * dist * e0
    e1 = math.exp(-b * y1)
    return a * dist * (e0 - e1) / (b * dy)


@njit(inline='always', fastmath=True)
def inscatter_point(ox, oy, oz, dx, dy, dz, tmax, lx, ly, lz, r0):
    """Integral over t in [0,tmax] of 1/(|o + d t - l|^2 + r0^2) dt  (d normalised)."""
    qx = lx - ox
    qy = ly - oy
    qz = lz - oz
    b = qx * dx + qy * dy + qz * dz
    c2 = qx * qx + qy * qy + qz * qz - b * b
    h2 = c2 + r0 * r0
    if h2 < 1e-8:
        h2 = 1e-8
    h = math.sqrt(h2)
    return (math.atan((tmax - b) / h) - math.atan(-b / h)) / h
