"""B's figure renderer (RUN-B-3), shared by every B shot that shows her, the travellers, the child or the cairn
(the vigil, the reveal, the hand-back, the climb), so she is the same person all film.

It draws mt.figure's 2-D SDF puppets (FG.Drawing: same primitives, same groups) with B's light:
  * NO UNIFORM OUTLINE HALO (director, H5): mt.figure lights every silhouette edge when a light is behind the figure
    (the sticker / Jawa read). Here the rim is lit only where the edge's outward normal faces the light's direction on
    screen, so a low sun behind-left rims the upper-left edges and the rest stays a soft silhouette.
  * WOOL TRANSLUCENCY: thin wool on the sun side glows in its own colour (the red shawl reads red against the sun).
  * CLOTH FOLDS: cloaks carry soft vertical folds (ambient occlusion in the troughs, splaying toward the hem), so the
    silhouette has weight instead of a flat fill.
  * SNOW-TOPPED STONES: materials with a snow amount catch the sky on their upper faces.
Same call as FG.render; `mats` defaults to MB (bset.M padded to B's layout + B's extra rows 22..).

    import bfig; bfig.render(img, zb, cam, drawing, feet_world, lights, amb=..., t=t)
"""
import math
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_MT = os.path.join(os.path.dirname(HERE), 'montage')          # mt (the shared puppet kit)
if _MT not in sys.path:
    sys.path.insert(0, _MT)

from mt import figure as FG            # noqa: E402
from mt.noise import gnoise2           # noqa: E402

# B's material layout: mt.figure's 11 columns +
#   11 translucency (0..1) | 12-14 transmitted tint | 15 fold depth (0..1) | 16 fold frequency (1/m) | 17 snow on top
#   18 lichen (0..1: pale grey-green and ochre crust in small blotches, weathered stone)
NCOLB = 19


def pad(M):
    M = np.asarray(M, np.float64)
    out = np.zeros((M.shape[0], NCOLB))
    out[:, :min(M.shape[1], NCOLB)] = M[:, :NCOLB]
    return out


def _mb():
    import bset as BS
    M = pad(BS.M)
    extra = pad(np.array([
        [0.74, 0.75, 0.80, 0.5, 0.0, 8.0, 0.04, 0, 0, 0, 0],      # 22 old snow (a drift, a cap)
        [0.17, 0.165, 0.155, 1.0, 0.1, 10.0, 0.05, 0, 0, 0, 0],   # 23 pale lichened granite
        [0.012, 0.011, 0.011, 0.0, 0.0, 8.0, 0.30, 0, 0, 0, 0],   # 24 the dark gaps inside a cairn
    ]))
    M = np.vstack([M, extra])
    # the red woven shawl (13) and its weave (20): translucent wool, a red glow on the sun side
    for r in (13, 20):
        M[r, 11] = 0.55
        M[r, 12:15] = [1.0, 0.16, 0.07]
    # the grey-brown wool cloak (14) and a traveller's coat: heavy vertical folds
    M[14, 11] = 0.10
    M[14, 12:15] = [0.9, 0.55, 0.35]
    M[14, 15] = 0.34
    M[14, 16] = 7.5
    M[21, 15] = 0.20
    M[21, 16] = 7.5
    # granite: snow lies on the stones' upper faces
    for r in (16, 17, 23):
        M[r, 17] = 0.85
        M[r, 18] = 0.55
        M[r, 6] = 0.13                      # rounded fieldstones: a broad pillow, never a flat facet
    return M


_MB = None


def MB():
    global _MB
    if _MB is None:
        _MB = _mb()
    return _MB


SNOW, PALE_STONE, GAP = 22, 23, 24
STONE = 4        # B's extra primitive: an irregular, faceted stone (no internal seams; see add_stone)


def add_stone(d, c, r, elong=1.4, ang=0.0, facet=0.8, seed=0, k=0.0, mat=16):
    """Append one irregular stone to Drawing d (current group): a random convex polygon (5-7 corners, softened by
    a few harmonics) of half-length r*elong along `ang`, half-height r. Row: 1 cx | 2 cy | 3-4 the long axis tip |
    5 b (half-height) | 6 facet | 7 bounding radius | 10 seed (fuzz stays 0)."""
    cx, cy = float(c[0]), float(c[1])
    a = r * elong
    tip = (cx + a * math.cos(ang), cy + a * math.sin(ang))
    d._add(STONE, [cx, cy, tip[0], tip[1], r, facet, 1.35 * max(a, r)], k, 0.0, float(seed % 100000), mat)
    return d


@njit(inline='always', fastmath=True)
def _hash01(i, j):
    a = (i * 73856093) ^ (j * 19349663)
    a = a & 0xFFFFFFFF
    a = ((a ^ 61) ^ (a >> 16)) & 0xFFFFFFFF
    a = (a + (a << 3)) & 0xFFFFFFFF
    a = a ^ (a >> 4)
    a = (a * 0x27D4EB2D) & 0xFFFFFFFF
    a = a ^ (a >> 15)
    return a * (1.0 / 4294967296.0)


@njit(inline='always', fastmath=True)
def _sd_stone(x, y, cx, cy, tx, ty, b, facet, seed):
    ax = tx - cx
    ay = ty - cy
    al = math.sqrt(ax * ax + ay * ay) + 1e-9
    ux = ax / al
    uy = ay / al
    px = x - cx
    py = y - cy
    u = (px * ux + py * uy) / al
    v = (-px * uy + py * ux) / b
    rho = math.sqrt(u * u + v * v) + 1e-9
    th = math.atan2(v, u)
    # the polygon's radius at angle th (corners at jittered angles, radii 0.80-1.12)
    n = 5 + int(_hash01(seed, 1) * 3.0)
    t0 = _hash01(seed, 2) * 6.2832
    rel = (th - t0) % 6.2832
    k = int(rel / (6.2832 / n))
    if k >= n:
        k = n - 1
    ta = t0 + 6.2832 * (k + 0.30 * (_hash01(seed, 10 + k) - 0.5)) / n
    tb = t0 + 6.2832 * (k + 1 + 0.30 * (_hash01(seed, 10 + (k + 1) % n) - 0.5)) / n
    ra = 0.80 + 0.32 * _hash01(seed, 30 + k)
    rb = 0.80 + 0.32 * _hash01(seed, 30 + (k + 1) % n)
    th2 = t0 + rel
    den = ra * math.sin(th2 - ta) + rb * math.sin(tb - th2)
    rp = ra * rb * math.sin(tb - ta) / den if abs(den) > 1e-6 else 0.5 * (ra + rb)
    rp = min(max(rp, 0.5), 1.2)
    # softened by harmonics (weathered, never a perfect polygon)
    rh = (1.0 + (0.05 + 0.07 * _hash01(seed, 55)) * math.cos(2.0 * th + 6.2832 * _hash01(seed, 51))
          + (0.03 + 0.05 * _hash01(seed, 56)) * math.cos(3.0 * th + 6.2832 * _hash01(seed, 52))
          + 0.025 * math.cos(5.0 * th + 6.2832 * _hash01(seed, 53)))
    if v < -0.35:                                      # a flatter underside: it rests on the stone below
        rh *= 1.0 - 0.10 * min((-v - 0.35) / 0.4, 1.0)
    r = rp * facet + rh * (1.0 - facet) * 0.96
    return (rho - r) * min(al, b) * 0.95


@njit(inline='always', fastmath=True)
def _prim_b(P, i, x, y, seed):
    if int(P[i, 0]) == STONE:
        return _sd_stone(x, y, P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6], int(P[i, 10]) + 7 * seed)
    return FG._prim(P, i, x, y, seed)


@njit(inline='always', fastmath=True)
def _group_sdf_b(P, g0, g1, x, y, seed):
    d = 1e9
    dm = 1e9
    mat = 0
    for i in range(g0, g1):
        di = _prim_b(P, i, x, y, seed)
        if di < dm:
            dm = di
            mat = int(P[i, 11])
        k = P[i, 8]
        if k > 0.0 and d < 1e8:
            h = max(k - abs(d - di), 0.0) / k
            d = min(d, di) - h * h * k * 0.25
        else:
            d = min(d, di)
    return d, mat


@njit(parallel=True, fastmath=True, cache=True)
def _render(img, depth, P, gidx, M, L, amb, fx, fy, ppm, zf, x0, x1, y0, y1, seed, t, fogT, fogc, flipx,
            write_depth, zbias, rim_gain):
    ng = gidx.shape[0] - 1
    e = 0.6 / ppm
    for py in prange(y0, y1):
        for px in range(x0, x1):
            if depth[py, px] < zf - zbias:
                continue
            X = (px + 0.5 - fx) / ppm
            if flipx:
                X = -X
            Y = (fy - (py + 0.5)) / ppm
            for g in range(ng):
                a0 = gidx[g]
                a1 = gidx[g + 1]
                d, mat = _group_sdf_b(P, a0, a1, X, Y, seed)
                cov = 0.5 - d * ppm
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                dx1, _m = _group_sdf_b(P, a0, a1, X + e, Y, seed)
                dy1, _m = _group_sdf_b(P, a0, a1, X, Y + e, seed)
                gx = dx1 - d
                gy = dy1 - d
                gl = math.sqrt(gx * gx + gy * gy) + 1e-12
                gx /= gl
                gy /= gl
                if flipx:
                    gx = -gx
                thick = M[mat, 6]
                di = max(-d, 0.0)
                u = min(di / thick, 1.0)
                nz = math.sqrt(max(1.0 - (1.0 - u) * (1.0 - u), 0.0))
                sl = math.sqrt(max(1.0 - nz * nz, 0.0))
                nx = gx * sl
                ny = gy * sl
                Xs = X if not flipx else -X
                ar = M[mat, 0]
                ag = M[mat, 1]
                ab = M[mat, 2]
                # snow lying on the upper faces (stones): a cap a few cm deep wherever the nearest edge is above
                sn = M[mat, 17]
                if sn > 0.0:
                    capd = 0.035 + 0.02 * (0.5 + 0.5 * gnoise2(Xs * 6.0, Y * 6.0, seed + 5))
                    w = sn * min(max((gy - 0.20) / 0.35, 0.0), 1.0) * min(max((capd - di) / 0.012, 0.0), 1.0)
                    ar += (0.78 - ar) * w
                    ag += (0.79 - ag) * w
                    ab += (0.84 - ab) * w
                # lichen and weathering on stone: blotches of pale grey-green and ochre crust, mottled grey
                lc = M[mat, 18]
                if lc > 0.0:
                    n1 = gnoise2(Xs * 26.0, Y * 26.0, seed + 21) + 0.5 * gnoise2(Xs * 9.0, Y * 9.0, seed + 22)
                    li = lc * min(max((n1 - 0.18) / 0.22, 0.0), 1.0) * (1.0 - min(max(gy, 0.0) * 2.0 * (1.0 - u), 1.0) * sn)
                    och = 0.5 + 0.5 * gnoise2(Xs * 5.0 + 3.0, Y * 5.0, seed + 23)
                    lr = 0.30 + 0.12 * och
                    lg = 0.31 + 0.02 * och
                    lb = 0.24 - 0.10 * och
                    mot = 0.82 + 0.36 * (0.5 + 0.5 * gnoise2(Xs * 4.0, Y * 4.0, seed + 24))
                    ar = (ar * mot) * (1.0 - li) + lr * li
                    ag = (ag * mot) * (1.0 - li) + lg * li
                    ab = (ab * mot) * (1.0 - li) + lb * li
                # cloth folds: soft vertical troughs splaying toward the hem (ambient occlusion)
                ao = 1.0
                fd = M[mat, 15]
                if fd > 0.0:
                    fq = M[mat, 16]
                    ph = 1.6 * gnoise2(Xs * 1.3, Y * 0.9, seed + 11) + 0.9 * Y * fq * 0.15
                    wv = 0.5 + 0.5 * math.sin(Xs * fq * 6.2832 / (1.0 + 0.25 * max(0.9 - Y, 0.0)) + ph)
                    ao = 1.0 - fd * wv * wv * (0.4 + 0.6 * u)
                rim_k = M[mat, 3]
                spk = M[mat, 4]
                spp = M[mat, 5]
                sky = (0.55 + 0.45 * max(ny, 0.0)) * ao
                cr = amb[0] * ar * sky
                cg = amb[1] * ag * sky
                cb = amb[2] * ab * sky
                edge = (1.0 - nz) ** 2.5
                for li in range(L.shape[0]):
                    if L[li, 0] < 0.5:
                        lx = L[li, 1] - Xs
                        ly = L[li, 2] - Y
                        lz = L[li, 3]
                        d2 = lx * lx + ly * ly + lz * lz
                        E = L[li, 7] / (d2 + L[li, 8] * L[li, 8])
                        inv = 1.0 / math.sqrt(d2 + 1e-12)
                        lx *= inv
                        ly *= inv
                        lz *= inv
                    else:
                        lx = L[li, 1]
                        ly = L[li, 2]
                        lz = L[li, 3]
                        E = L[li, 7]
                    ndl = nx * lx + ny * ly + nz * lz
                    diff = max(ndl, 0.0)
                    wrap = max((ndl + 0.35) / 1.35, 0.0)
                    # the rim: ONLY on edges whose outward normal faces the light across the screen
                    lxy = math.sqrt(lx * lx + ly * ly)
                    face = min(max((gx * lx + gy * ly) / max(lxy, 0.25), 0.0), 1.0)
                    back = max(-lz, 0.0) ** 0.5
                    rim = edge * face ** 1.5 * (wrap + 1.4 * back) * rim_k * rim_gain
                    # wool translucency: thin wool toward the light glows in its own colour
                    tr = M[mat, 11]
                    trans = 0.0
                    if tr > 0.0:
                        trans = tr * back * face * (1.0 - nz) ** 0.8
                    spec = 0.0
                    if spk > 0.0:
                        hz = lz + 1.0
                        hx = lx
                        hy = ly
                        hl = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
                        nh = max((nx * hx + ny * hy + nz * hz) / hl, 0.0)
                        spec = spk * nh ** spp
                    k_ = E * (0.85 * diff + 0.15 * wrap) * (0.35 + 0.65 * ao)
                    cr += L[li, 4] * (ar * k_ + E * (rim * 0.10 + spec * 0.05 + trans * 0.16 * M[mat, 12]))
                    cg += L[li, 5] * (ag * k_ + E * (rim * 0.10 + spec * 0.05 + trans * 0.16 * M[mat, 13]))
                    cb += L[li, 6] * (ab * k_ + E * (rim * 0.10 + spec * 0.05 + trans * 0.16 * M[mat, 14]))
                if M[mat, 7] + M[mat, 8] + M[mat, 9] > 0.0:
                    fq = M[mat, 10]
                    n = gnoise2(X * fq + t * 1.5, Y * fq, 77) * 0.6 + 0.4 * gnoise2(X * fq * 2.3, Y * fq * 2.3 + t * 2.5, 78)
                    em = max(n + 0.25, 0.0) ** 2 * 2.5 * u
                    cr += M[mat, 7] * em
                    cg += M[mat, 8] * em
                    cb += M[mat, 9] * em
                cr = cr * fogT + fogc[0]
                cg = cg * fogT + fogc[1]
                cb = cb * fogT + fogc[2]
                img[py, px, 0] = img[py, px, 0] * (1.0 - cov) + cr * cov
                img[py, px, 1] = img[py, px, 1] * (1.0 - cov) + cg * cov
                img[py, px, 2] = img[py, px, 2] * (1.0 - cov) + cb * cov
                if write_depth and cov > 0.5:
                    depth[py, px] = zf


def render(img, depth, cam, drawing, feet_world, lights_world, amb=(0.01, 0.012, 0.02), mats=None, seed=0,
           t=0.0, fog_T=1.0, fog_col=(0, 0, 0), flipx=False, write_depth=True, zbias=0.05, emissive_gain=1.0,
           rim_gain=1.0):
    """FG.render's call with B's light (see the module docstring). mats: None = MB(); a table with fewer columns is
    padded (no translucency, folds or snow for its rows)."""
    A = drawing.array() if isinstance(drawing, FG.Drawing) else drawing
    if len(A) == 0:
        return
    Pw = np.asarray(feet_world, np.float64)
    sx, sy, z = cam.project(Pw)
    if z <= 0.1:
        return
    ppm = cam.f / z
    groups = A[:, 12]
    order = np.argsort(groups, kind='stable')
    A = A[order]
    g = A[:, 12]
    starts = np.nonzero(np.r_[True, g[1:] != g[:-1]])[0]
    gidx = np.r_[starts, len(A)].astype(np.int64)
    xmin, xmax, ymin, ymax = (drawing.bounds() if isinstance(drawing, FG.Drawing) else FG._bounds_arr(A))
    if flipx:
        xmin, xmax = -xmax, -xmin
    H, W = img.shape[:2]
    x0 = int(max(0, math.floor(sx + xmin * ppm - 2)))
    x1 = int(min(W, math.ceil(sx + xmax * ppm + 2)))
    y0 = int(max(0, math.floor(sy - ymax * ppm - 2)))
    y1 = int(min(H, math.ceil(sy - ymin * ppm + 2)))
    if x1 <= x0 or y1 <= y0:
        return
    L = []
    for lw in lights_world:
        row = np.zeros(9)
        if 'pos' in lw:
            d = np.asarray(lw['pos'], np.float64) - Pw
            row[0] = 0
        else:
            d = np.asarray(lw['dir'], np.float64)
            d = d / np.linalg.norm(d)
            row[0] = 1
        row[1] = d[0] * cam.right[0] + d[2] * cam.right[2]
        row[2] = d[1]
        row[3] = -(d[0] * cam.fwd[0] + d[2] * cam.fwd[2])
        row[4:7] = lw['col']
        row[7] = lw['I']
        row[8] = lw.get('r0', 0.3)
        L.append(row)
    L = np.array(L, np.float64) if L else np.zeros((0, 9))
    if mats is None:
        M = MB().copy()
    else:
        M = pad(mats)
        base = MB()
        if M.shape[0] < base.shape[0]:
            M = np.vstack([M, base[M.shape[0]:]])
    M[:, 7:10] *= emissive_gain
    _render(img, depth, A, gidx, M, L, np.asarray(amb, np.float64), float(sx), float(sy), float(ppm), float(z),
            x0, x1, y0, y1, int(seed), float(t), float(fog_T), np.asarray(fog_col, np.float64), bool(flipx),
            bool(write_depth), float(zbias), float(rim_gain))
