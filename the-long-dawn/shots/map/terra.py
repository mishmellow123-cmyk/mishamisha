"""THE INVENTED WORLD (MAP-L2): the continent cut C's map is drawn from. It replaces every real-world input the map
had (Natural Earth coasts, the Earth normal map, Blue Marble biomes and lakes, hand-traced real rivers and ranges).

Everything is in map degrees (X east, Y north) on the same sheet as before (geo.MAP_*), so the engine (the camera,
the bake, the hand, the parchment, the fire relay, the pen-line Road) is untouched. Nothing here is a real place,
and nothing is Middle-earth: the large shapes are designed below, and everything finer is grown from them.

* LAND: a designed silhouette (control polygons, gulfs and firths as sea capsules, island seeds) as a signed distance,
  plus domain-warped fractal noise, so the coasts are intricate at every scale the camera sees; the ranges push the
  land out where they meet the sea (headlands, island chains).
* RELIEF (elevation units as the old proxy: lowland < 1, hills 1-2.5, ranges 3-9): the ranges are designed spines
  (height and width along them; passes where the height drops) raised with a sharp crest and ridged-noise spurs; the
  High Moor at the heart (the ring of stones) is a broad, open upland that belongs to no one; hill country from noise.
* WATER: rivers from flow accumulation (priority-flood filled, D8 steepest descent, weighted by rainfall), traced
  from their heads to the sea, width from the flow; lakes where the fill is deep and wide.
* CLIMATE: rain from the great western sea, carried east and wrung out by the heights (a rain shadow): forest where
  it is wet, conifers high and in the north, a desert far inland in the lee, open plains between.

Cached in renders/map_C/cache/terra_<VER>.npz (a pure function of the constants below).
    python terra.py              # build + print a summary
    python terra.py preview      # a coloured design preview of the sheet and of the camera region (scratchpad)
"""
import math
import os
import sys
import time

import cv2
import numpy as np
from numba import njit, prange

import geo
from noise import fbm, gnoise

VER = 1
D = 0.1                      # hydrology / relief grid (map degrees per cell)
DF = 0.05                    # coast grid
X0, X1 = geo.MAP_X0, geo.MAP_X1
Y0, Y1 = geo.MAP_Y0, geo.MAP_Y1

# ================================================================== DESIGN ===
# The heart of the map: the ring of stones on the High Moor.
RING = (0.0, 18.0)
MOOR = dict(c=(0.0, 18.0), r=6.0, h=2.2)
# Her beacon: the high knee of her range (the ink Run's range), where it bends.
BEACON = (13.0, 8.0)

# Land silhouettes (closed polygons, map degrees), the signed distance of their union is the land's skeleton.
HEART = [  # the great continent; its west coast faces the great western sea
    (-17, 64), (-21, 56), (-18, 49), (-23, 43), (-19, 37), (-25, 30), (-23, 24), (-18, 20), (-21, 11), (-27, 6),
    (-25, -1), (-19, -7), (-12, -12), (-6, -19), (4, -23), (16, -21), (27, -26), (38, -24), (47, -30), (58, -27),
    (66, -33), (74, -28), (84, -24), (95, -27), (104, -20), (109, -9), (116, -3), (113, 8), (121, 17), (118, 28),
    (126, 36), (121, 47), (111, 55), (104, 66), (92, 71), (78, 76), (63, 81), (48, 79), (35, 83), (22, 78),
    (10, 74), (-2, 71), (-10, 68)]
SOUTHLAND = [  # across the southern sea
    (28, -48), (40, -44), (55, -47), (70, -43), (86, -46), (100, -52), (108, -60), (60, -64), (30, -62), (22, -55)]
FARWEST = [  # a far shore across the great western sea, at the sheet's edge
    (-178, 42), (-166, 47), (-158, 40), (-161, 28), (-154, 17), (-160, 6), (-170, -4), (-178, -8)]
EASTISLE = [(140, 22), (149, 30), (155, 22), (151, 9), (143, 12)]
POLYS = [HEART, SOUTHLAND, FARWEST, EASTISLE]

# Sea capsules cut into the land: (x0, y0, x1, y1, half-width). Firths, a bay, a gulf.
SEAS = [
    (-19.0, 33.2, -8.5, 33.9, 1.1),      # the long firth in the north-west
    (-23.0, 15.8, -14.5, 16.4, 3.6),     # the western bay, where the great river meets the sea
    (60.0, -30.0, 63.0, -18.0, 3.2),     # a southern gulf
    (104.0, 30.0, 114.0, 31.0, 2.4),     # an eastern firth
    (40.0, 76.0, 44.0, 66.0, 2.0),       # a northern sound
]
# Island seeds: (x, y, radius). Chains off the headlands, a scatter in the bay, a large isle offshore.
ISLES = [
    (-29.5, 28.6, 1.3), (-32.5, 29.9, 0.9), (-35.2, 31.2, 1.6), (-38.6, 30.4, 0.8), (-41.0, 32.6, 1.1),
    (-31.0, 11.0, 2.3), (-34.5, 8.2, 1.1), (-36.4, 14.4, 0.8),
    (-20.2, 17.4, 0.45), (-18.6, 14.6, 0.35),
    (-29.0, 1.0, 0.9), (-31.8, -3.0, 1.4), (-27.4, -7.2, 0.7),
    (-60.0, 22.0, 6.0), (-56.0, 12.0, 3.0), (-72.0, 30.0, 2.2), (-80.0, 8.0, 3.0), (-95.0, 24.0, 1.8),
    (-110.0, 10.0, 2.6), (-125.0, 30.0, 1.6), (-130.0, 0.0, 2.2), (-45.0, -20.0, 3.4), (-48.0, 50.0, 3.5),
    (10.0, -30.0, 2.0), (126.0, -12.0, 3.0), (132.0, 58.0, 4.0), (-20.0, 72.0, 2.5),
]

# Ranges: spines of (x, y, crest height, half-width). The crest height drops where a pass crosses.
RANGES = {
    # HER RANGE (the ink Run's range): the west arm climbs from the south-west to the knee (her beacon), the east
    # arm runs east-south-east from it, long and high, off the sheet's heart toward the east.
    'hers_w': [(-7.0, -6.0, 1.6, 2.0), (-3.0, -3.0, 3.8, 2.6), (1.5, 0.5, 5.4, 2.8), (6.0, 3.6, 6.4, 3.0),
               (9.5, 6.0, 7.4, 3.2), (13.0, 8.0, 8.6, 3.4)],
    'hers_e': [(13.0, 8.0, 8.6, 3.4), (16.5, 7.4, 7.8, 3.3), (20.0, 6.4, 7.2, 3.2), (23.5, 5.6, 7.6, 3.3),
               (27.0, 5.0, 7.0, 3.3), (30.0, 4.9, 3.2, 2.6), (33.0, 5.2, 6.8, 3.2), (37.5, 6.0, 7.6, 3.4),
               (43.0, 7.8, 7.0, 3.4), (50.0, 10.0, 6.6, 3.2), (58.0, 12.6, 6.2, 3.0), (67.0, 14.0, 5.8, 3.0),
               (76.0, 17.6, 4.8, 2.8), (84.0, 22.0, 3.4, 2.4), (90.0, 26.0, 1.8, 2.0)],
    # the northern range, along the far side of the plains; a pass north of the moor
    'north': [(-14.0, 40.0, 2.4, 2.2), (-8.5, 38.8, 4.8, 2.6), (-2.5, 38.4, 5.6, 2.8), (3.0, 37.6, 2.6, 2.4),
              (8.0, 38.6, 5.4, 2.8), (14.0, 40.4, 6.0, 3.0), (20.0, 43.0, 5.2, 2.8), (26.0, 46.4, 4.2, 2.6),
              (31.0, 50.8, 2.4, 2.2)],
    # the western coast hills, broken, down the headland and toward the bay
    'coast': [(-21.0, 30.5, 2.0, 1.8), (-18.0, 27.5, 3.4, 2.0), (-15.5, 24.0, 3.0, 2.0), (-14.0, 21.0, 1.6, 1.8)],
    # a short range in the south-east of the camera's sheet, across the southern lowlands
    'south': [(24.0, -8.0, 1.8, 2.0), (30.0, -10.5, 4.8, 2.6), (37.0, -12.0, 5.6, 2.8), (44.0, -12.5, 4.6, 2.6),
              (50.0, -15.0, 2.0, 2.0)],
    # beyond the camera: the far north, the east, the south lands, the far west
    'farnorth': [(20.0, 58.0, 2.0, 2.4), (32.0, 60.0, 6.0, 3.2), (46.0, 58.0, 6.8, 3.2), (60.0, 60.0, 6.2, 3.2),
                 (74.0, 64.0, 5.0, 3.0), (86.0, 66.0, 2.0, 2.4)],
    'fareast': [(92.0, 44.0, 2.0, 2.4), (98.0, 34.0, 5.8, 3.0), (104.0, 22.0, 6.4, 3.2), (106.0, 8.0, 5.6, 3.0),
                (102.0, -6.0, 4.0, 2.6), (96.0, -16.0, 2.0, 2.2)],
    'southland': [(34.0, -54.0, 2.0, 2.4), (50.0, -52.0, 5.6, 3.0), (66.0, -50.0, 6.4, 3.2), (82.0, -52.0, 5.2, 3.0),
                  (96.0, -56.0, 2.2, 2.4)],
    'farwest': [(-172.0, 36.0, 2.0, 2.2), (-166.0, 26.0, 4.6, 2.8), (-168.0, 12.0, 4.2, 2.8), (-174.0, 2.0, 2.0, 2.2)],
    'westisle': [(-63.0, 25.0, 2.4, 1.8), (-60.0, 21.0, 4.0, 2.0), (-57.0, 17.0, 2.6, 1.8)],
}
# hill country (x, y, radius, strength): low rolling uplands where the fire finds hills to stand on
HILLS = [(-10.0, 8.0, 5.0, 1.0), (-12.0, 26.0, 5.5, 0.9), (22.0, 20.0, 6.0, 0.9), (35.0, 30.0, 7.0, 0.8),
         (-6.0, -12.0, 6.0, 0.9), (8.0, -12.0, 5.0, 0.8), (48.0, 32.0, 8.0, 0.8), (62.0, 30.0, 8.0, 0.9),
         (70.0, -12.0, 8.0, 0.9), (80.0, 50.0, 8.0, 0.8), (6.0, 56.0, 7.0, 0.8), (-6.0, 50.0, 5.0, 0.8)]
# basins (x, y, rx, ry, depth): depressions the fill turns into lakes
BASINS = [(19.0, 15.0, 4.2, 1.3, 1.1), (31.0, 25.5, 2.8, 2.0, 0.9), (-7.0, 30.0, 2.0, 1.3, 0.8),
          (56.0, 38.0, 3.2, 2.2, 1.0), (40.0, 58.0, 2.6, 1.6, 0.8)]

RIVER_T = 1500.0               # a channel drains at least this much rain (cells x rainfall)
WIND = 12.0                  # the rain comes from the great western sea, a little south of west (degrees)


# ============================================================== geometry ===

def grid(d, x0=X0, x1=X1, y0=Y0, y1=Y1):
    W = int(round((x1 - x0) / d))
    H = int(round((y1 - y0) / d))
    xs = x0 + (np.arange(W) + 0.5) * d
    ys = y1 - (np.arange(H) + 0.5) * d
    return xs, ys


def poly_sdf(d):
    """Signed distance (map degrees, + inside) to the union of the land polygons, on grid d."""
    xs, ys = grid(d)
    H, W = len(ys), len(xs)
    m = np.zeros((H, W), np.uint8)
    for P in POLYS:
        P = np.asarray(P, np.float64)
        px = (P[:, 0] - X0) / d - 0.5
        py = (Y1 - P[:, 1]) / d - 0.5
        cv2.fillPoly(m, [np.round(np.stack([px, py], 1) * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    sdf = (din - dout) * d
    # rounder corners: a light blur of the distance
    return cv2.GaussianBlur(sdf.astype(np.float32), (0, 0), 1.2 / d)


def _seg_d(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    t = ((px - ax) * abx + (py - ay) * aby) / max(abx * abx + aby * aby, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - ax - abx * t, py - ay - aby * t), t


def range_fields(xs, ys):
    """Per range: distance to its spine and the crest height / half-width interpolated along it. Returns the
    max-combined crest profile (height at the crest line) and a corridor weight."""
    X, Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    H = np.zeros(X.shape, np.float32)
    Wd = np.zeros(X.shape, np.float32)
    for name, pts in RANGES.items():
        P = np.asarray(pts, np.float64)
        # densify the spine (Catmull-Rom) so the crest bends smoothly
        Q = _catmull(P, 12)
        bx0, bx1 = Q[:, 0].min() - 3 * Q[:, 3].max(), Q[:, 0].max() + 3 * Q[:, 3].max()
        by0, by1 = Q[:, 1].min() - 3 * Q[:, 3].max(), Q[:, 1].max() + 3 * Q[:, 3].max()
        ix = np.where((xs >= bx0) & (xs <= bx1))[0]
        iy = np.where((ys >= by0) & (ys <= by1))[0]
        if not len(ix) or not len(iy):
            continue
        sx = X[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
        sy = Y[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
        best = np.full(sx.shape, 1e9, np.float32)
        hh = np.zeros(sx.shape, np.float32)
        ww = np.ones(sx.shape, np.float32)
        for k in range(len(Q) - 1):
            dd, t = _seg_d(sx, sy, Q[k, 0], Q[k, 1], Q[k + 1, 0], Q[k + 1, 1])
            m = dd < best
            best = np.where(m, dd, best)
            hh = np.where(m, Q[k, 2] + (Q[k + 1, 2] - Q[k, 2]) * t, hh)
            ww = np.where(m, Q[k, 3] + (Q[k + 1, 3] - Q[k, 3]) * t, ww)
        u = best / ww
        prof = np.where(u < 1.0, (1.0 - u) ** 1.6, 0.0) * hh
        sub = H[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
        np.maximum(sub, prof.astype(np.float32), out=sub)
        wsub = Wd[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
        np.maximum(wsub, np.clip(1.25 - u / 1.6, 0, 1).astype(np.float32) * (hh > 2.5), out=wsub)
    return H, Wd


def _catmull(P, n):
    """Catmull-Rom through the rows of P (any number of columns), n samples per span."""
    P = np.asarray(P, np.float64)
    if len(P) < 3:
        u = np.linspace(0, 1, n + 1)[:, None]
        return P[0] * (1 - u) + P[1] * u
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(Q[-2])
    return np.array(out)


# =========================================================== noise kernels ===

@njit(cache=True, parallel=True)
def _coast_noise(xs, ys, fine):
    """The coast's fractal: domain-warped fBm in map degrees of 'inlandness'. fine=True returns only the octaves
    finer than ~1.5 degrees (evaluated on the fine coast grid), fine=False only the coarser ones."""
    H, W = len(ys), len(xs)
    out = np.empty((H, W), np.float32)
    for i in prange(H):
        y = ys[i]
        for j in range(W):
            x = xs[j]
            wx = fbm(x * 0.035, y * 0.035, 11, 3, 2.0, 0.5) * 16.0
            wy = fbm(x * 0.035 + 7.1, y * 0.035 - 3.3, 12, 3, 2.0, 0.5) * 16.0
            xx = x + wx
            yy = y + wy
            if fine:
                v = 1.3 * fbm(xx * 0.55, yy * 0.55, 23, 2, 2.1, 0.55) + 0.55 * fbm(xx * 1.7, yy * 1.7, 24, 2, 2.1, 0.55) \
                    + 0.2 * gnoise(xx * 4.6, yy * 4.6, 25)
            else:
                v = 7.5 * fbm(xx * 0.05, yy * 0.05, 21, 2, 2.0, 0.5) + 3.6 * fbm(xx * 0.17, yy * 0.17, 22, 2, 2.0, 0.5)
            out[i, j] = v
    return out


@njit(cache=True, parallel=True)
def _relief_noise(xs, ys, out_r, out_h):
    """Ridged noise for spurs (0..1, sharp crests) and rolling noise for hills (-1..1)."""
    H, W = len(ys), len(xs)
    for i in prange(H):
        y = ys[i]
        for j in range(W):
            x = xs[j]
            wx = fbm(x * 0.12, y * 0.12, 41, 2, 2.0, 0.5) * 2.2
            wy = fbm(x * 0.12 + 3.7, y * 0.12 + 9.1, 42, 2, 2.0, 0.5) * 2.2
            s = 0.0
            a = 1.0
            f = 0.55
            nrm = 0.0
            for o in range(4):
                n = 1.0 - abs(gnoise((x + wx) * f, (y + wy) * f, 51 + o * 7) * 1.45)
                s += a * n * n
                nrm += a
                a *= 0.5
                f *= 2.1
            out_r[i, j] = s / nrm
            out_h[i, j] = fbm(x * 0.28, y * 0.28, 61, 4, 2.0, 0.55) * 1.6


# ================================================================ relief ===

def relief(d):
    """(S, E, E_range): the land field S (+ land, map degrees of inlandness), elevation E and the range part."""
    xs, ys = grid(d)
    X, Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    sdf = poly_sdf(d)
    S = sdf + _coast_noise(xs, ys, False)
    for (ax, ay, bx, by, w) in SEAS:
        dd, _ = _seg_d(X, Y, ax, ay, bx, by)
        S -= np.clip(w - dd, 0, None) * 2.4 + 3.0 * np.exp(-(dd / w) ** 2)
    for (cx, cy, r) in ISLES:
        dd = np.hypot(X - cx, Y - cy)
        S = np.maximum(S, (r - dd) * 1.6 + 0.8 * np.exp(-(dd / r) ** 2) * r)
    Hc, Wc = range_fields(xs, ys)
    ridg = np.empty(X.shape, np.float32)
    roll = np.empty(X.shape, np.float32)
    _relief_noise(xs, ys, ridg, roll)
    # the ranges: a sharp crest along the spine, spurs from the ridged noise off it
    Er = Hc * (0.5 + 0.62 * ridg) + 0.9 * Wc * np.maximum(ridg - 0.55, 0) * Hc / 3.0
    # the ranges push the land out where they meet the sea (headlands, island chains)
    S = S + 0.55 * Er
    inland = np.clip(sdf / 10.0, 0, 1)
    E = 0.25 + 0.6 * inland ** 0.7 + 0.25 * roll * (0.4 + 0.6 * inland)
    for (cx, cy, r, k) in HILLS:
        dd = np.hypot(X - cx, Y - cy) / r
        E += k * np.clip(1 - dd, 0, 1) ** 1.3 * (1.2 + 0.9 * (ridg - 0.5) + 0.6 * roll)
    # the High Moor: a broad, open upland; flat on top, a gentle rim of downs
    mc, mr, mh = MOOR['c'], MOOR['r'], MOOR['h']
    dm = np.hypot(X - mc[0], (Y - mc[1]) * 1.1) / mr
    top = 1.0 - np.clip((dm - 0.55) / 0.45, 0, 1)
    top = top * top * (3 - 2 * top)
    E += mh * top + 0.35 * np.exp(-((dm - 0.78) / 0.12) ** 2) * (0.7 + 0.6 * ridg) + 0.12 * roll * top
    for (cx, cy, rx, ry, dep) in BASINS:
        dd = np.hypot((X - cx) / rx, (Y - cy) / ry)
        E -= dep * np.clip(1 - dd * dd, 0, 1)
    E = E + Er
    return S.astype(np.float32), E.astype(np.float32), Er.astype(np.float32), sdf


# ============================================================== hydrology ===

@njit(cache=True)
def _heap_push(hk, hv, n, k, v):
    i = n
    hk[i] = k
    hv[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = hk[i], hk[p]
        hv[p], hv[i] = hv[i], hv[p]
        i = p
    return n + 1


@njit(cache=True)
def _heap_pop(hk, hv, n):
    k, v = hk[0], hv[0]
    n -= 1
    hk[0] = hk[n]
    hv[0] = hv[n]
    i = 0
    while True:
        l = 2 * i + 1
        if l >= n:
            break
        c = l
        if l + 1 < n and hk[l + 1] < hk[l]:
            c = l + 1
        if hk[i] <= hk[c]:
            break
        hk[c], hk[i] = hk[i], hk[c]
        hv[c], hv[i] = hv[i], hv[c]
        i = c
    return k, v, n


@njit(cache=True)
def priority_flood(E, land, eps):
    """Depression-filled elevation (every land cell drains to the sea) and the D8 receiver of every land cell
    (-1: flows straight into the sea)."""
    H, W = E.shape
    F = E.copy()
    done = np.zeros((H, W), np.bool_)
    N = H * W
    hk = np.empty(N, np.float64)
    hv = np.empty(N, np.int64)
    n = 0
    di = (-1, -1, -1, 0, 0, 1, 1, 1)
    dj = (-1, 0, 1, -1, 1, -1, 0, 1)
    for i in range(H):
        for j in range(W):
            if not land[i, j]:
                done[i, j] = True
                continue
            edge = i == 0 or j == 0 or i == H - 1 or j == W - 1
            if not edge:
                for q in range(8):
                    if not land[i + di[q], j + dj[q]]:
                        edge = True
                        break
            if edge:
                done[i, j] = True
                n = _heap_push(hk, hv, n, F[i, j], i * W + j)
    while n > 0:
        k, v, n = _heap_pop(hk, hv, n)
        i = v // W
        j = v % W
        for q in range(8):
            a = i + di[q]
            b = j + dj[q]
            if a < 0 or b < 0 or a >= H or b >= W or done[a, b]:
                continue
            done[a, b] = True
            if F[a, b] <= F[i, j] + eps:
                F[a, b] = F[i, j] + eps
            n = _heap_push(hk, hv, n, F[a, b], a * W + b)
    rec = np.full(N, -1, np.int64)
    for i in range(H):
        for j in range(W):
            if not land[i, j]:
                continue
            best = 0.0
            bq = -1
            sea = False
            for q in range(8):
                a = i + di[q]
                b = j + dj[q]
                if a < 0 or b < 0 or a >= H or b >= W:
                    sea = True
                    continue
                if not land[a, b]:
                    sea = True
                    continue
                dist = 1.4142135 if (di[q] != 0 and dj[q] != 0) else 1.0
                s = (F[i, j] - F[a, b]) / dist
                if s > best:
                    best = s
                    bq = q
            if bq >= 0:
                rec[i * W + j] = (i + di[bq]) * W + (j + dj[bq])
            elif sea:
                rec[i * W + j] = -1
            else:
                rec[i * W + j] = -1
    return F, rec


@njit(cache=True)
def accumulate(F, rec, wgt, land):
    H, W = F.shape
    order = np.argsort(-F.ravel())
    A = wgt.ravel().copy()
    lf = land.ravel()
    for k in range(len(order)):
        c = order[k]
        if not lf[c]:
            continue
        r = rec[c]
        if r >= 0:
            A[r] += A[c]
    return A.reshape(H, W)


def rain(E, land, d):
    """Rain carried from the great western sea (a little south of west), wrung out by the heights. Returns the
    rainfall per cell, smoothed (0..~1.5)."""
    a = math.radians(WIND)
    H, W = E.shape
    # rotate so the wind blows along +x
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), -WIND, 1.0)
    Er = cv2.warpAffine(E, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Lr = cv2.warpAffine(land.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    P = _rain_sweep(Er, Lr, d)
    Minv = cv2.invertAffineTransform(M)
    P = cv2.warpAffine(P, Minv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    P = cv2.GaussianBlur(P, (0, 0), 1.2 / d)
    return np.clip(P, 0, 4.0) * land


@njit(cache=True, parallel=True)
def _rain_sweep(E, L, d):
    """March each row with the wind: the air takes up the sea's moisture, and over land it rains out a little
    everywhere and much more where the ground rises into it. Rainfall is in units of the wet lowland's."""
    H, W = E.shape
    P = np.zeros((H, W), np.float32)
    base = 0.0026 * d / 0.1
    for i in prange(H):
        m = 1.0
        for j in range(W):
            if L[i, j] < 0.5:
                m += (1.0 - m) * 0.05 * d / 0.1
                continue
            up = 0.0
            if j > 0:
                up = max(E[i, j] - E[i, j - 1], 0.0) / d
            r = m * (base + 0.0012 * d / 0.1 * max(E[i, j] - 1.0, 0.0) + 0.0105 * d / 0.1 * min(up, 6.0))
            r = min(r, m)
            m -= r
            P[i, j] = r / base
    return P


def rivers_from(F, rec, A, land, lake, d, thresh, min_len):
    """Trace the channel network: every channel cell belongs to one polyline, the main stem through each
    confluence continuing and each tributary ending where it joins. Returns [(xy (N,2), flow (N,))]."""
    H, W = F.shape
    xs, ys = grid(d)
    ch = (A > thresh) & land
    idx = np.where(ch.ravel())[0]
    chs = set(idx.tolist())
    ups = {}
    Af = A.ravel()
    for c in idx:
        r = rec[c]
        if r >= 0 and r in chs:
            ups.setdefault(int(r), []).append(int(c))
    main_up = {r: max(u, key=lambda q: Af[q]) for r, u in ups.items()}
    heads = [int(c) for c in idx if int(c) not in ups]
    out = []
    for h in heads:
        path = [h]
        c = h
        while True:
            r = int(rec[c])
            if r < 0 or r not in chs:
                break
            path.append(r)
            if main_up.get(r) != c:
                break                       # a tributary ends where it joins the larger stream
            c = r
        P = np.array([(xs[p % W], ys[p // W]) for p in path], np.float64)
        fl = Af[path]
        if len(P) < 2:
            continue
        L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
        if L < min_len:
            continue
        out.append((P, fl, int(rec[path[-1]]) < 0))
    return out


# ================================================================== build ===

def _objs(items):
    a = np.empty(len(items), dtype=object)
    for i, v in enumerate(items):
        a[i] = v
    return a


_W = None


def world():
    """The invented world (cached): rasters on the D grid and the vector data for the drawing."""
    global _W
    if _W is not None:
        return _W
    path = os.path.join(geo.CACHE, f'terra_{VER}.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        _W = {k: z[k] for k in z.files}
        _W['coast'] = list(_W['coast'])
        _W['lakes'] = list(_W['lakes'])
        _W['rivers'] = list(_W['rivers'])
        return _W
    t0 = time.time()
    global _SC
    _SC = relief(D)
    S, E, Er, sdf = _SC
    land = S > 0
    # keep the small bits of land that are real islands, drop specks
    n, lab, st, _ = cv2.connectedComponentsWithStats(land.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = st[1:, 4] < 4
    land &= ~small[lab]
    E = np.where(land, np.maximum(E, 0.05), 0.0).astype(np.float32)
    print('relief', round(time.time() - t0, 1), 's', flush=True)
    P = rain(E, land, D)
    F, rec = priority_flood(E.astype(np.float64), land, 1e-5)
    depth = (F - E) * land
    lake = (depth > 0.06) & land
    n, lab, st, _ = cv2.connectedComponentsWithStats(lake.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= 45
    lake = keep[lab]
    wgt = (0.25 + P).astype(np.float64) * land
    A = accumulate(F, rec, wgt, land)
    print('hydrology', round(time.time() - t0, 1), 's', flush=True)
    rv = rivers_from(F, rec, A, land, lake, D, thresh=RIVER_T, min_len=1.5)
    print('rivers', len(rv), round(time.time() - t0, 1), 's', flush=True)
    # ruggedness: slope of the relief (elevation units per degree, scaled like the old normal-map proxy)
    Es = cv2.GaussianBlur(E, (0, 0), 0.25 / D)
    gy, gx = np.gradient(Es, D)
    rug = np.clip(np.hypot(gx, gy) * 0.1, 0, 2.0) * land
    coast = coast_rings()
    lakes = lake_rings(lake)
    _W = dict(S=S, E=E, Er=Er, land=land, rain=P.astype(np.float32), acc=A.astype(np.float32), rug=rug.astype(np.float32),
              lake=lake, coast=coast, lakes=lakes, rivers=rv)
    np.savez_compressed(path, S=S, E=E, Er=Er, land=land, rain=P.astype(np.float32), acc=A.astype(np.float32),
                        rug=rug.astype(np.float32), lake=lake,
                        coast=_objs(coast), lakes=_objs(lakes), rivers=_objs(rv))
    print('world', round(time.time() - t0, 1), 's', flush=True)
    return _W


def coast_field(xs, ys, d_coarse=D):
    """The land field S on an arbitrary fine grid: the coarse design (sdf + large noise + ranges + seas + isles,
    bilinearly up-sampled) plus the fine fractal octaves evaluated in place."""
    S, _, _, _ = _coarse_S()
    xc, yc = grid(d_coarse)
    fx = ((xs - xc[0]) / d_coarse).astype(np.float32)
    fy = ((yc[0] - ys) / d_coarse).astype(np.float32)
    mx, my = np.meshgrid(fx, fy)
    Sc = cv2.remap(S, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Sc + _coast_noise(xs, ys, True)


_SC = None


def _coarse_S():
    global _SC
    if _SC is None:
        _SC = relief(D)
    return _SC


def _samp(A, px, py):
    return geo.remap_points(A, px, py)


def coast_rings():
    """Coast rings (map XY, + is_hole) traced on the fine grid, refined onto the zero level set."""
    xs, ys = grid(DF)
    out = []
    band = 400
    Sf = np.empty((len(ys), len(xs)), np.float32)
    for a in range(0, len(ys), band):
        b = min(len(ys), a + band)
        Sf[a:b] = coast_field(xs, ys[a:b])
    m = (Sf > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = st[1:, 4] < 10
    m[small[lab]] = 0
    del lab
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for k, c in enumerate(cs):
        if len(c) < 12:
            continue
        p = c[:, 0, :].astype(np.float64)
        # onto the zero crossing: two Newton steps on the bilinear field
        for _ in range(2):
            f = _samp(Sf, p[:, 0], p[:, 1])
            ggx = 0.5 * (_samp(Sf, p[:, 0] + 1, p[:, 1]) - _samp(Sf, p[:, 0] - 1, p[:, 1]))
            ggy = 0.5 * (_samp(Sf, p[:, 0], p[:, 1] + 1) - _samp(Sf, p[:, 0], p[:, 1] - 1))
            g2 = ggx * ggx + ggy * ggy + 1e-6
            step = np.clip(f / g2, -0.8, 0.8)
            p[:, 0] -= step * ggx
            p[:, 1] -= step * ggy
        xy = np.stack([xs[0] + p[:, 0] * DF, ys[0] - p[:, 1] * DF], 1)
        hole = hier[0][k][3] >= 0
        out.append((xy, bool(hole)))
    return out


def lake_rings(lake):
    """Lake outlines (map XY) from the lake cells, smoothed on a 4x grid."""
    xs, ys = grid(D)
    big = cv2.resize(lake.astype(np.uint8) * 255, (lake.shape[1] * 4, lake.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), 2.5)
    cs, _ = cv2.findContours((big > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 16:
            continue
        p = c[:, 0, :].astype(np.float64)
        sx = (p[:, 0] + 0.5) / 4.0 - 0.5
        sy = (p[:, 1] + 0.5) / 4.0 - 0.5
        out.append(np.stack([xs[0] + sx * D, ys[0] - sy * D], 1))
    return out


# ============================================================ the preview ===

def preview(out_dir):
    """A coloured design sheet: hypsometric tint, rain, rivers, lakes, coasts; the ring and her beacon marked."""
    W_ = world()
    E, land, P = W_['E'], W_['land'], W_['rain']
    H, W = E.shape
    img = np.zeros((H, W, 3), np.float32)
    sea = np.array([0.55, 0.62, 0.66])
    low = np.array([0.80, 0.78, 0.62])
    wet = np.array([0.52, 0.64, 0.42])
    dry = np.array([0.90, 0.80, 0.58])
    hi = np.array([0.52, 0.42, 0.34])
    snow = np.array([0.92, 0.9, 0.88])
    r = np.clip(P / 0.9, 0, 1)[..., None]
    base = dry * (1 - r) + wet * r
    e = E[..., None]
    col = np.where(e < 2.5, base * (1 - e / 2.5) + low * (e / 2.5) * 0.3 + base * (e / 2.5) * 0.7,
                   np.where(e < 6.0, base * (1 - (e - 2.5) / 3.5) + hi * ((e - 2.5) / 3.5),
                            hi * (1 - np.clip((e - 6) / 3, 0, 1)) + snow * np.clip((e - 6) / 3, 0, 1)))
    img = np.where(land[..., None], col, sea)
    # hillshade
    gy, gx = np.gradient(cv2.GaussianBlur(E, (0, 0), 1.0), D)
    sh = np.clip(1.0 + (-gx * 0.7 + gy * 0.7) * 0.08, 0.55, 1.35)
    img = img * np.where(land, sh, 1.0)[..., None]
    img = np.where(W_['lake'][..., None], np.array([0.45, 0.55, 0.62]), img)
    im8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    sc = 1.0 / D

    def px(x, y):
        return int(round((x - X0) * sc)), int(round((Y1 - y) * sc))
    for P_, fl, _ in W_['rivers']:
        w = np.clip(np.sqrt(fl / RIVER_T) * 0.8, 1, 4)
        for k in range(len(P_) - 1):
            cv2.line(im8, px(*P_[k]), px(*P_[k + 1]), (160, 90, 40), int(round(w[k])), cv2.LINE_AA)
    for xy, hole in W_['coast']:
        pts = np.stack([(xy[:, 0] - X0) * sc, (Y1 - xy[:, 1]) * sc], 1)
        cv2.polylines(im8, [np.round(pts * 16).astype(np.int32)], True, (30, 30, 30), 1, cv2.LINE_AA, shift=4)
    cv2.circle(im8, px(*RING), 6, (0, 0, 200), 2, cv2.LINE_AA)
    cv2.drawMarker(im8, px(*BEACON), (0, 140, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    os.makedirs(out_dir, exist_ok=True)
    small = cv2.resize(im8, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, 'terra_sheet.jpg'), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    a, b = px(-45, 50)
    c, d = px(50, -15)
    cv2.imwrite(os.path.join(out_dir, 'terra_heart.jpg'), im8[b:d, a:c], [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('preview ->', out_dir)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'preview':
        preview(sys.argv[2] if len(sys.argv) > 2 else os.path.join(geo.CACHE, 'preview'))
    else:
        w = world()
        print('land cells', int(w['land'].sum()), 'rivers', len(w['rivers']), 'lakes', len(w['lakes']),
              'coast rings', len(w['coast']))
