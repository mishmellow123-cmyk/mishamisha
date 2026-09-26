"""KARST geometry (venv side): sandstone/limestone pillars as detailed r(theta, z) meshes + where
trees cling (crowns, ledges, cracks).

A pillar's radius around its (slightly wandering) axis:
  r = R(z) * S(theta)  * (1 + lumps)  - flutes - cracks - bedding notches + fine noise
S = rounded-rectangle (superellipse) section (Zhangjiajie columns are joint-bounded), flutes = narrow
vertical grooves with wandering phase, cracks = a few deep clefts that wander and end, bedding =
horizontal notches/overhangs every 6-14 m. The top closes with a lumpy dome (the crown).
Output: binary mesh (figures.write_mesh format; quads, 4th index -1 = triangle) + a per-vertex
'mat' channel = vegetation affinity (0 rock .. 1 crown / ledge / crack).
"""
import math
import os
import struct
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'shots', 'montage'))
from mt.noise import fbm3, gnoise3  # noqa: E402


@njit(inline="always", fastmath=True, cache=True)
def _block_r(ct, st, ox, oy, a, b, rot, n_):
    """Distance from the pillar axis along direction (ct, st) to a superellipse block boundary
    (block centre (ox, oy), half-sizes a, b, rotation rot, exponent n_). The axis must be inside."""
    cr, sr = math.cos(rot), math.sin(rot)
    lo, hi = 0.0, 4.0 * (a + b + abs(ox) + abs(oy))
    for _ in range(22):
        t = 0.5 * (lo + hi)
        x = t * ct - ox
        y = t * st - oy
        u = (cr * x + sr * y) / a
        v = (-sr * x + cr * y) / b
        if abs(u) ** n_ + abs(v) ** n_ < 1.0:
            lo = t
        else:
            hi = t
    return 0.5 * (lo + hi)


@njit(parallel=True, fastmath=True, cache=True)
def _pillar_r(TH, Z, prm, LB, seed, out_r, out_veg):
    """Radius + vegetation affinity on the (theta, z) grid.
    LB rows (lobes = joint-bounded sub-columns): (theta_c, half_width, radius, ztop, taper, edge_sharp).
    r = max over live lobes of a flat-topped angular plateau -> column faces with deep clefts between."""
    nz, nt = TH.shape
    R0, zb, zt, lump, fine, Lb, stri, core = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7]
    nl = LB.shape[0]
    for j in prange(nz):
        z = Z[j, 0]
        u = (z - zb) / max(zt - zb, 1e-3)
        bulge = 1.0 + 0.05 * math.sin(6.283 * z / 53.0 + seed) + 0.03 * math.sin(6.283 * z / 21.0 + 2.0 * seed)
        zz = z / Lb + 0.4 * gnoise3(z * 0.035, seed, 0.0, 3)
        fz = zz - math.floor(zz)
        nstr = max(0.0, gnoise3(math.floor(zz) * 1.7, seed, 2.0, 4))
        notch = math.exp(-((fz - 0.5) / 0.05) ** 2) * nstr
        for i in range(nt):
            th = TH[j, i]
            ct = math.cos(th)
            st = math.sin(th)
            r = core * R0
            ledge = 0.0
            for k in range(nl):
                ztk = LB[k, 3]
                top = min(1.0, max(0.0, (ztk - z) / (0.35 * R0)))
                if top <= 0.0:
                    continue
                topf = math.sqrt(top * (2.0 - top))
                d = math.atan2(math.sin(th - LB[k, 0]), math.cos(th - LB[k, 0])) / LB[k, 1]
                wob = 0.08 * gnoise3(z / 23.0, k * 3.3, seed, 31)
                d += wob
                pl = 1.0 / (1.0 + abs(d) ** LB[k, 5])
                face = LB[k, 2] * (1.0 - 0.12 * d * d) * pl * (1.0 - LB[k, 4] * u)
                rk = R0 * face * (0.25 + 0.75 * topf)
                if rk > r:
                    r = rk
                if top < 1.0:
                    ledge = max(ledge, (1.0 - top) * pl)
            r *= bulge * (1.0 + lump * fbm3(ct * 1.3 + seed, st * 1.3, z / 36.0, 4.0, 7))
            sv = fbm3(th * R0 * 0.22, z * 0.028, seed * 0.37, 4.0, 9)
            r -= stri * abs(sv) * R0
            r -= 0.018 * R0 * notch
            # fractured stone: hashed block offsets in (arc length, height) at two scales, soft edges
            sarc = th * R0
            blk = 0.0
            for sc_i in range(2):
                bw = 3.2 if sc_i == 0 else 1.1
                bh = 2.4 if sc_i == 0 else 0.9
                amp = 0.42 if sc_i == 0 else 0.16
                row = math.floor(z / bh + 0.13 * gnoise3(sarc * 0.1, seed, sc_i * 5.0, 41))
                col = math.floor(sarc / bw + 0.5 * (row % 2) + 0.2 * gnoise3(z * 0.1, seed, sc_i * 7.0, 43))
                hsh = math.sin(row * 12.9898 + col * 78.233 + seed * 3.7 + sc_i * 17.0) * 43758.5453
                rnd = hsh - math.floor(hsh)
                fu = sarc / bw + 0.5 * (row % 2) - col
                fv = z / bh - row
                fu = fu - math.floor(fu)
                fv = fv - math.floor(fv)
                edge = min(min(fu, 1.0 - fu) * bw, min(fv, 1.0 - fv) * bh)
                soft = min(1.0, edge / 0.18)
                blk += amp * (rnd - 0.5) * soft - 0.05 * amp * (1.0 - soft)
            r += blk
            r += fine * fbm3(ct * r * 0.5, st * r * 0.5, z * 0.5, 3.0, 23)
            out_r[j, i] = max(r, 0.12 * R0)
            out_veg[j, i] = min(1.0, 0.6 * notch + 1.2 * ledge + 2.0 * max(0.0, abs(sv) - 0.3))


def lobes_for(R0, zb, zt, rng, n=None):
    """3-6 sub-columns around the axis; two reach the top, the others stop lower (ledges)."""
    n = n or int(rng.integers(3, 7))
    th0 = rng.uniform(0, 2 * math.pi)
    gaps = rng.uniform(0.7, 1.3, n)
    ths = th0 + np.cumsum(gaps) / gaps.sum() * 2 * math.pi
    L = []
    for k in range(n):
        hw = (2 * math.pi / n) * rng.uniform(0.32, 0.5)
        rad = rng.uniform(0.72, 1.08)
        if k < 2:
            ztop = zt
        else:
            ztop = zt - rng.uniform(0.08, 0.55) * (zt - zb) * 0.5
        L.append((ths[k], hw, rad, ztop, rng.uniform(0.04, 0.2), rng.uniform(5.0, 9.0)))
    return np.array(L, np.float64)


def pillar(cx, cy, R0, zb, zt, seed, nt=256, nz=None, lobes=None, lump=0.07, fine=0.1, bed=11.0, stri=0.035,
           core=0.55, lean=(0.0, 0.0), cap=0.22):
    """Returns (V float32 Nx3, Q int32 Mx4 (-1 = tri), veg float32 N)."""
    rng = np.random.default_rng(int(abs(seed) * 1000) % (2 ** 31))
    H = zt - zb
    if nz is None:
        nz = max(24, int(nt * H / (2 * math.pi * R0) * 1.1))
    BL = lobes if lobes is not None else lobes_for(R0, zb, zt, rng)
    prm = np.array([R0, zb, zt, lump, fine, bed, stri, core], np.float64)
    th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
    z = np.linspace(zb, zt, nz)
    TH, Z = np.meshgrid(th, z)
    R = np.empty_like(TH)
    VEG = np.empty_like(TH)
    _pillar_r(TH, Z, prm, BL, float(seed % 97), R, VEG)
    ox = cx + lean[0] * (Z - zb) / max(H, 1) + 0.03 * R0 * np.sin(Z / 31.0 + seed)
    oy = cy + lean[1] * (Z - zb) / max(H, 1) + 0.03 * R0 * np.cos(Z / 27.0 + 2 * seed)
    X = ox + R * np.cos(TH)
    Y = oy + R * np.sin(TH)
    V = [np.stack([X, Y, Z], -1).reshape(-1, 3)]
    veg = [VEG.reshape(-1)]
    nc = max(6, nz // 14)
    rt_top = R[-1]
    ctop = zt + cap * R0
    for c in range(1, nc + 1):
        t = c / nc
        a = t * math.pi / 2
        rr = rt_top * math.cos(a)
        zz = zt + (ctop - zt) * math.sin(a)
        bump = 0.07 * R0 * np.sin(th * 3 + seed) * np.sin(th * 5 + 2 * seed) * math.sin(a)
        V.append(np.stack([ox[-1] + rr * np.cos(th), oy[-1] + rr * np.sin(th), zz + bump + 0 * th], -1))
        veg.append(np.full(nt, 1.0))
    V.append(np.array([[ox[-1, 0], oy[-1, 0], ctop + 0.04 * R0]]))
    veg.append(np.array([1.0]))
    V = np.vstack(V).astype(np.float32)
    veg = np.concatenate(veg).astype(np.float32)
    rows = nz + nc
    Q = []
    i = np.arange(nt)
    i2 = (i + 1) % nt
    for j in range(rows - 1):
        a = j * nt
        b = (j + 1) * nt
        Q.append(np.stack([a + i, a + i2, b + i2, b + i], 1))
    pole = rows * nt
    a = (rows - 1) * nt
    Q.append(np.stack([a + i, a + i2, np.full(nt, pole), np.full(nt, -1)], 1))
    Q = np.vstack(Q).astype(np.int32)
    return V, Q, veg


def upness(V, Q):
    """Per-vertex upward-facing factor from face normals (for ledge vegetation)."""
    Qt = Q.copy()
    tri = Qt[:, 3] < 0
    Qt[tri, 3] = Qt[tri, 2]
    p0, p1, p2, p3 = V[Qt[:, 0]], V[Qt[:, 1]], V[Qt[:, 2]], V[Qt[:, 3]]
    n = np.cross(p2 - p0, p3 - p1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    acc = np.zeros(len(V))
    cnt = np.zeros(len(V))
    for k in range(4):
        np.add.at(acc, Qt[:, k], n[:, 2])
        np.add.at(cnt, Qt[:, k], 1)
    return (acc / np.maximum(cnt, 1)).astype(np.float32)


def write_mesh(path, V, Q, M):
    with open(path, 'wb') as f:
        f.write(struct.pack('<ii', len(V), len(Q)))
        f.write(np.ascontiguousarray(V, np.float32).tobytes())
        f.write(np.ascontiguousarray(Q, np.int32).tobytes())
        f.write(np.ascontiguousarray(M, np.float32).tobytes())


def tree_points(V, Q, veg, up, rng, dens_crown=0.5, dens_ledge=0.08, zmin=-1e9, max_n=4000):
    """Where trees grow: crown vertices (dense), ledge/crack vertices (sparse). Returns rows of
    (x, y, z, yaw, scale, tilt_x, tilt_y, variant)."""
    out = []
    cand = np.nonzero(((veg > 0.95) | ((up > 0.35) & (veg > 0.2)) | (veg > 0.55)) & (V[:, 2] > zmin))[0]
    for vi in cand:
        crown = veg[vi] > 0.95
        p = dens_crown if crown else dens_ledge * (0.5 + up[vi])
        if rng.random() > p:
            continue
        x, y, z = V[vi]
        s = rng.uniform(0.6, 1.25) * (1.0 if crown else 0.8)
        if crown:
            tx, ty = rng.normal(0, 0.08, 2)
        else:
            # clinging: lean outward from the pillar axis, then up
            tx, ty = rng.normal(0, 0.25, 2)
        out.append((x, y, z - 0.2, rng.uniform(0, 6.283), s, tx, ty, int(rng.integers(0, 6))))
        if len(out) >= max_n:
            break
    return out
