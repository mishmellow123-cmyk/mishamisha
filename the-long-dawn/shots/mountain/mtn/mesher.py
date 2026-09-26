"""Adaptive, camera-aware triangle meshes of heightfields.

Points are taken from nested regular grids (spacing s_min * 2^L) so that each region gets the
coarsest grid finer than the target spacing s(p) = clamp(k * dist_to_path(p), s_min, s_max); the
union is Delaunay-triangulated (scipy), which grades the transitions without cracks.
Points outside every camera's frustum (plus margin) are dropped.
"""
import json
import math
import os

import numpy as np
from numba import njit, prange
from scipy.spatial import Delaunay, cKDTree

R_EARTH = 6.371e6


def dist_to_path(X, Y, path_xy):
    """Horizontal distance from points to the polyline of camera positions (dense enough)."""
    tree = cKDTree(path_xy)
    d, _ = tree.query(np.stack([X, Y], 1), k=1)
    return d


def visible_mask(X, Y, Z, cams, margin_deg=8.0, near_keep=60.0):
    """cams: list of (pos[3], fwd[3], right[3], up[3], tan_hx, tan_hy). Keep a point if inside any
    frustum (angles enlarged by margin) or within near_keep metres of a camera."""
    keep = np.zeros(X.shape[0], bool)
    P = np.stack([X, Y, Z], 1)
    tm = math.tan(math.radians(margin_deg))
    for (c, f, r, u, thx, thy) in cams:
        d = P - c
        zf = d @ f
        xr = d @ r
        yu = d @ u
        ok = zf > 1.0
        lim_x = (thx + tm) * np.maximum(zf, 1.0)
        lim_y = (thy + tm) * np.maximum(zf, 1.0)
        # generous in y below the frame (hidden terrain still needed for slopes) handled by margin
        ok &= (np.abs(xr) < lim_x) & (np.abs(yu) < lim_y + 0.0)
        near = np.einsum('ij,ij->i', d, d) < near_keep ** 2
        keep |= ok | near
    return keep


def point_set(bounds, path_xy, k, s_min, s_max, max_level=14, cull=None):
    """Nested grid points. bounds = (x0, y0, x1, y1). Level L (spacing s) only covers the
    annulus where the target spacing is in [s, 2s), so each level is generated inside the path's
    bounding box grown by 2s/k. `cull(X, Y)` -> bool keep mask (e.g. frustum test)."""
    x0, y0, x1, y1 = bounds
    px0, py0 = path_xy.min(0)
    px1, py1 = path_xy.max(0)
    tree = cKDTree(path_xy)
    pts = []
    s = s_min
    L = 0
    while s <= s_max * 1.0001 and L <= max_level:
        last = s * 2 > s_max * 1.0001
        grow = 1e12 if last else 2 * s / k
        bx0, by0 = max(x0, px0 - grow), max(y0, py0 - grow)
        bx1, by1 = min(x1, px1 + grow), min(y1, py1 + grow)
        gx = np.arange(math.ceil(bx0 / s) * s, bx1 + 1e-6, s)
        gy = np.arange(math.ceil(by0 / s) * s, by1 + 1e-6, s)
        rows = max(1, 2_000_000 // max(len(gx), 1))
        for c0 in range(0, len(gy), rows):
            GX, GY = np.meshgrid(gx, gy[c0:c0 + rows])
            X = GX.ravel()
            Y = GY.ravel()
            if cull is not None:
                m = cull(X, Y)
                X = X[m]
                Y = Y[m]
            if len(X) == 0:
                continue
            d, _ = tree.query(np.stack([X, Y], 1), k=1)
            target = np.clip(k * d, s_min, s_max)
            # this level owns points whose target spacing is in [s, 2s) (finest level owns < 2s_min)
            if L == 0:
                sel = target < 2 * s
            elif last:
                sel = target >= s
            else:
                sel = (target >= s) & (target < 2 * s)
            pts.append(np.stack([X[sel], Y[sel], np.full(sel.sum(), s)], 1))
        s *= 2
        L += 1
    P = np.concatenate(pts, 0)
    return P


@njit(parallel=True, cache=True)
def _cull_nb(X, Y, C, F, R, U, TX, TY, zs, near2):
    """Keep (x, y) if the vertical segment z in [zs[0], zs[-1]] intersects any camera frustum
    (exact: each frustum plane bounds z; the bounds must leave a non-empty interval), or
    passes within sqrt(near2) of a camera."""
    n = X.shape[0]
    keep = np.zeros(n, np.bool_)
    nc = C.shape[0]
    zlo0 = zs[0]
    zhi0 = zs[zs.shape[0] - 1]
    for i in prange(n):
        x = X[i]
        y = Y[i]
        ok = False
        for c in range(nc):
            dx = x - C[c, 0]
            dy = y - C[c, 1]
            h2 = dx * dx + dy * dy
            # near sphere: closest z on the segment to the camera
            zc = min(max(C[c, 2], zlo0), zhi0)
            dzc = zc - C[c, 2]
            if h2 + dzc * dzc < near2:
                ok = True
                break
            lo = zlo0
            hi = zhi0
            good = True
            for pl in range(5):
                # plane normal a (inside: a . d < b)
                if pl == 0:
                    ax = -F[c, 0]; ay = -F[c, 1]; az = -F[c, 2]; bb = -1.0
                elif pl == 1:
                    ax = R[c, 0] - TX[c] * F[c, 0]; ay = R[c, 1] - TX[c] * F[c, 1]
                    az = R[c, 2] - TX[c] * F[c, 2]; bb = 0.0
                elif pl == 2:
                    ax = -R[c, 0] - TX[c] * F[c, 0]; ay = -R[c, 1] - TX[c] * F[c, 1]
                    az = -R[c, 2] - TX[c] * F[c, 2]; bb = 0.0
                elif pl == 3:
                    ax = U[c, 0] - TY[c] * F[c, 0]; ay = U[c, 1] - TY[c] * F[c, 1]
                    az = U[c, 2] - TY[c] * F[c, 2]; bb = 0.0
                else:
                    ax = -U[c, 0] - TY[c] * F[c, 0]; ay = -U[c, 1] - TY[c] * F[c, 1]
                    az = -U[c, 2] - TY[c] * F[c, 2]; bb = 0.0
                rhs = bb - ax * dx - ay * dy + az * C[c, 2]
                if abs(az) < 1e-9:
                    if 0.0 >= rhs:
                        good = False
                        break
                elif az > 0.0:
                    hi = min(hi, rhs / az)
                else:
                    lo = max(lo, rhs / az)
                if lo >= hi:
                    good = False
                    break
            if good:
                ok = True
                break
        keep[i] = ok
    return keep


def frustum_cull_xy(cams, zlo, zhi, margin_deg=10.0, near_keep=80.0, nz=5):
    """Returns cull(X, Y): keep if the vertical segment (zlo..zhi) at (X, Y) intersects any
    camera frustum (tested at a few heights)."""
    zs = np.linspace(zlo, zhi, nz)
    tm = math.tan(math.radians(margin_deg))
    C = np.array([c[0] for c in cams], float)
    F = np.array([c[1] for c in cams], float)
    R = np.array([c[2] for c in cams], float)
    U = np.array([c[3] for c in cams], float)
    TX = np.array([c[4] for c in cams], float) + tm
    TY = np.array([c[5] for c in cams], float) + tm

    def cull(X, Y):
        return _cull_nb(np.ascontiguousarray(X, float), np.ascontiguousarray(Y, float), C, F, R, U,
                        TX, TY, zs, near_keep * near_keep)
    return cull


def triangulate(P, max_ratio=3.5):
    """Delaunay; drops triangles whose longest edge is much longer than the local spacing
    (P[:, 2]) - those only bridge culled gaps / the convex hull."""
    tri = Delaunay(P[:, :2], qhull_options='Qbb Qc Qz Q12')
    T = tri.simplices.astype(np.int32)
    a, b, c = P[T[:, 0], :2], P[T[:, 1], :2], P[T[:, 2], :2]
    L = np.maximum(np.maximum(np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1)),
                   np.linalg.norm(c - a, axis=1))
    s = np.maximum(np.maximum(P[T[:, 0], 2], P[T[:, 1], 2]), P[T[:, 2], 2])
    return T[L < max_ratio * s]


def save_mesh(path, V, T, attrs=None, meta=None):
    """Binary mesh: <path>.json header + <path>.bin payload (float32 verts, int32 tris, float32 attrs)."""
    attrs = attrs or {}
    meta = dict(meta or {})
    meta['nv'] = int(V.shape[0])
    meta['nt'] = int(T.shape[0])
    meta['attrs'] = list(attrs.keys())
    with open(path + '.bin', 'wb') as fh:
        fh.write(np.ascontiguousarray(V, np.float32).tobytes())
        fh.write(np.ascontiguousarray(T, np.int32).tobytes())
        for kname in attrs:
            fh.write(np.ascontiguousarray(attrs[kname], np.float32).tobytes())
    with open(path + '.json', 'w') as fh:
        json.dump(meta, fh)
