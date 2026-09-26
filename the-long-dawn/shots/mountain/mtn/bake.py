"""Nested height grids + per-vertex bakes (numba).

Grids: a list of (g, x0, y0, dx) from finest to coarsest; `hgrid` samples the finest grid that
contains the point (bilinear).

Bakes:
  sun_vis   : soft visibility of a distant light (moon or sun) by marching the horizon along its
              azimuth (Earth curvature included); penumbra from the disc size
  horizon   : the horizon elevation angle toward an azimuth (for animated sunrise)
  snow      : snow cover from slope (at a few metres' scale) and concavity
"""
import math

import numpy as np
from numba import njit, prange

R_EARTH = 6.371e6


@njit(inline='always', cache=True)
def _bil(g, fx, fy):
    ny, nx = g.shape
    i = int(fx)
    j = int(fy)
    u = fx - i
    v = fy - j
    return (g[j, i] * (1 - u) * (1 - v) + g[j, i + 1] * u * (1 - v) + g[j + 1, i] * (1 - u) * v
            + g[j + 1, i + 1] * u * v)


@njit(cache=True)
def hgrid(G0, M0, G1, M1, G2, M2, x, y):
    """Sample the finest grid containing (x, y). M = (x0, y0, dx). Returns -1e9 outside all."""
    fx = (x - M0[0]) / M0[2]
    fy = (y - M0[1]) / M0[2]
    if fx >= 0 and fy >= 0 and fx < G0.shape[1] - 1 and fy < G0.shape[0] - 1:
        return _bil(G0, fx, fy)
    fx = (x - M1[0]) / M1[2]
    fy = (y - M1[1]) / M1[2]
    if fx >= 0 and fy >= 0 and fx < G1.shape[1] - 1 and fy < G1.shape[0] - 1:
        return _bil(G1, fx, fy)
    fx = (x - M2[0]) / M2[2]
    fy = (y - M2[1]) / M2[2]
    if fx >= 0 and fy >= 0 and fx < G2.shape[1] - 1 and fy < G2.shape[0] - 1:
        return _bil(G2, fx, fy)
    return -1e9


@njit(parallel=True, cache=True)
def horizon_angle(X, Y, Z, az, G0, M0, G1, M1, G2, M2, tmax, t0=0.8, growth=1.045):
    """Max elevation angle (radians) of the terrain seen from each (X, Y, Z) toward bearing
    az (radians, 0 = +y north, + = east), up to distance tmax. Z = the point's height."""
    n = X.shape[0]
    out = np.empty(n)
    dx = math.sin(az)
    dy = math.cos(az)
    for k in prange(n):
        x0 = X[k]
        y0 = Y[k]
        z0 = Z[k] + 0.05
        best = -1.5
        t = t0
        while t < tmax:
            h = hgrid(G0, M0, G1, M1, G2, M2, x0 + dx * t, y0 + dy * t)
            if h > -1e8:
                e = (h - z0 - t * t / (2.0 * R_EARTH)) / t
                if e > best:
                    best = e
            t *= growth
            t += 0.3
        out[k] = math.atan(best)
    return out


@njit(parallel=True, cache=True)
def slope_curv(X, Y, G0, M0, G1, M1, G2, M2, r):
    """Slope (rise/run) and Laplacian-ish concavity at scale r (metres)."""
    n = X.shape[0]
    S = np.empty(n)
    C = np.empty(n)
    for k in prange(n):
        x = X[k]
        y = Y[k]
        hc = hgrid(G0, M0, G1, M1, G2, M2, x, y)
        hx1 = hgrid(G0, M0, G1, M1, G2, M2, x + r, y)
        hx0 = hgrid(G0, M0, G1, M1, G2, M2, x - r, y)
        hy1 = hgrid(G0, M0, G1, M1, G2, M2, x, y + r)
        hy0 = hgrid(G0, M0, G1, M1, G2, M2, x, y - r)
        gx = (hx1 - hx0) / (2 * r)
        gy = (hy1 - hy0) / (2 * r)
        S[k] = math.sqrt(gx * gx + gy * gy)
        C[k] = (hx1 + hx0 + hy1 + hy0 - 4 * hc) / (r * r)     # > 0 concave (hollow)
    return S, C


@njit(cache=True)
def visible(G0, M0, G1, M1, G2, M2, cx, cy, cz, px, py, pz, ref_x, ref_y, clear=4.0):
    """True if the segment camera -> point clears the land (Earth curvature about ref)."""
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    L = math.sqrt(dx * dx + dy * dy)
    if L < 1.0:
        return True
    n = int(min(max(L / 8.0, 40.0), 4000.0))
    for k in range(1, n):
        t = k / n
        # sample densely near both ends (geometric spacing)
        t = t * t * (3.0 - 2.0 * t)
        x = cx + dx * t
        y = cy + dy * t
        z = cz + dz * t
        if (1.0 - t) * L < 25.0:
            continue
        h = hgrid(G0, M0, G1, M1, G2, M2, x, y)
        h -= ((x - ref_x) ** 2 + (y - ref_y) ** 2) / (2.0 * R_EARTH)
        if h > z - clear:
            return False
    return True


@njit(cache=True)
def local_maxima(g, r, hmin):
    """Indices (j, i) of cells that are the maximum within radius r cells and above hmin."""
    ny, nx = g.shape
    out = []
    for j in range(r, ny - r):
        for i in range(r, nx - r):
            v = g[j, i]
            if v < hmin:
                continue
            ok = True
            for a in range(-r, r + 1):
                for b in range(-r, r + 1):
                    if (a != 0 or b != 0) and g[j + a, i + b] >= v:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append((j, i))
    return out
