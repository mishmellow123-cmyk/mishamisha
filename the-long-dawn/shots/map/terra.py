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

VER = geo.WORLD_VER
D = 0.1                      # hydrology / relief grid (map degrees per cell)
DF = 0.05                    # coast grid
X0, X1 = geo.MAP_X0, geo.MAP_X1
Y0, Y1 = geo.MAP_Y0, geo.MAP_Y1

# ================================================================== DESIGN ===
# The heart of the map: the ring of stones on the High Moor, at the centre of the sheet.
RING = (0.0, 18.0)
MOOR = dict(c=(0.0, 18.0), r=5.2, h=1.6)
# Her beacon: the high knee of her range (the ink Run's range), where it bends.
BEACON = (13.0, 8.0)

# Land silhouettes (closed polygons, map degrees). The great continent fills the east of the sheet and runs off
# its north, east and south edges; its west coast faces the great western sea. A far shore at the west edge.
HEART = [
    (-6, 106), (-12, 95), (-9, 86), (-16, 78), (-22, 70), (-19, 61), (-24, 53), (-20, 47),
    (-19, 43.5), (-17, 40), (-20, 37.5), (-26, 38.5), (-32, 36.5), (-35, 32.5), (-33.5, 28.5),   # the hook
    (-30, 29.5), (-26, 31.5), (-22, 31.5), (-20, 28), (-24, 24), (-22, 20.5),                  # the bulge
    (-17, 18.5), (-15, 14.8), (-19.5, 11.5),                                                   # the western bay
    (-26, 8.5), (-31, 5), (-36, 1), (-37.5, -3), (-32, -3.5), (-25, -4.5), (-19, -8),          # the long cape
    (-12, -13), (-6, -18), (3, -21), (11, -19), (19, -24), (27, -23), (35, -28), (42, -30),
    (47, -25), (52, -18), (58, -14), (65, -17), (70, -25), (72, -38), (76, -52), (74, -72),     # the southern gulf
    (125, -72), (192, -72), (192, 110),
    (64, 110), (61, 88), (56, 77), (47, 73), (38, 75), (32, 83), (28, 110),                     # the northern sound
    (-6, 110)]
FARWEST = [(-196, 58), (-178, 52), (-168, 45), (-171, 33), (-163, 22), (-167, 10), (-176, 1), (-182, -6), (-196, -8)]
SEA_ISLE = [(-66, 31), (-58, 33), (-51, 27), (-52, 18), (-57, 11), (-64, 13), (-68, 21)]     # the great western isle
POLYS = [HEART, FARWEST, SEA_ISLE]

# Sea capsules cut into the land: (x0, y0, x1, y1, half-width). The firth under the hook, the western bay's inner
# water, a gulf on the south coast, sounds in the far east and north.
SEAS = [
    (-17.8, 16.4, -15.6, 16.5, 1.3),     # the western bay's inner water, where the west river meets the sea
    (8.0, -21.0, 9.0, -15.0, 1.6),       # a south-coast inlet
    (120.0, 40.0, 150.0, 44.0, 3.0),     # an eastern sound
    (96.0, -40.0, 110.0, -30.0, 4.0),    # a southern bay beyond the gulf
    (140.0, -20.0, 170.0, -26.0, 5.0),
]
# Island seeds: (x, y, radius). Grown by the same fractal as the coasts, so none is round.
ISLES = [
    (-30.5, 22.0, 1.0), (-33.5, 19.5, 1.5), (-37.0, 23.0, 0.8), (-32.5, 15.5, 0.7), (-40.0, 18.0, 1.2),
    (-41.0, 5.0, 1.6), (-43.5, 0.5, 0.9), (-39.0, -7.0, 1.2), (-45.0, 9.5, 0.7),
    (-38.0, 37.0, 1.3), (-41.0, 33.0, 0.8), (-28.5, 31.2, 0.4), (-16.9, 17.3, 0.3),
    (-80.0, 6.0, 2.2), (-88.0, 30.0, 1.6), (-100.0, 14.0, 2.4), (-112.0, 34.0, 1.4), (-118.0, 4.0, 2.0),
    (-130.0, 22.0, 3.0), (-142.0, 40.0, 1.6), (-148.0, 8.0, 2.2), (-80.0, 44.0, 2.4), (-95.0, -12.0, 3.0),
    (-60.0, -20.0, 2.6), (-120.0, -30.0, 2.0), (-70.0, 60.0, 3.0), (-40.0, 64.0, 2.0), (-30.0, 80.0, 3.4),
    (-5.0, -30.0, 1.8), (20.0, -36.0, 2.2), (58.0, -40.0, 2.4), (62.0, -55.0, 2.0),
]

# Ranges: spines of (x, y, crest height, half-width). The crest height drops where a pass crosses.
RANGES = {
    # HER RANGE (the ink Run's range): the west arm climbs from the south-west to the knee (her beacon), the east
    # arm runs east-south-east from it, long and high, then swings north-east across the continent.
    'hers_w': [(-8.0, -7.0, 1.8, 1.8), (-4.0, -3.4, 4.0, 2.2), (0.5, 0.4, 5.6, 2.4), (5.0, 3.6, 6.4, 2.5),
               (9.0, 6.2, 7.6, 2.6), (13.0, 8.0, 8.8, 2.8)],
    'hers_e': [(13.0, 8.0, 8.8, 2.8), (16.5, 7.3, 8.0, 2.7), (20.0, 6.4, 7.4, 2.6), (23.5, 5.6, 7.8, 2.7),
               (27.0, 5.0, 7.0, 2.7), (30.0, 4.9, 3.0, 2.0), (33.0, 5.2, 6.8, 2.6), (37.5, 6.0, 7.8, 2.8),
               (43.0, 7.8, 7.2, 2.8), (50.0, 10.4, 6.6, 2.7), (57.0, 13.6, 6.0, 2.6), (64.0, 18.0, 6.4, 2.7),
               (70.0, 24.0, 5.4, 2.6), (75.0, 31.0, 3.0, 2.2), (80.0, 38.0, 5.8, 2.6), (86.0, 46.0, 6.2, 2.7),
               (93.0, 53.0, 5.0, 2.5), (100.0, 58.0, 2.0, 2.0)],
    # the northern range, along the far side of the plains; a pass north of the moor
    'north': [(-13.0, 33.5, 2.0, 1.7), (-8.0, 33.2, 4.2, 2.1), (-2.5, 33.6, 5.0, 2.2), (2.5, 33.2, 2.2, 1.9),
              (7.5, 33.6, 4.8, 2.2), (13.0, 34.8, 5.4, 2.3), (19.0, 36.8, 4.8, 2.2), (25.0, 39.6, 4.0, 2.1),
              (31.0, 43.0, 2.4, 1.8)],
    # the hills down the long south-west cape
    'cape': [(-19.0, 5.0, 1.4, 1.6), (-24.0, 2.6, 2.8, 1.8), (-29.0, 0.6, 3.2, 1.8), (-34.0, -1.4, 1.8, 1.6)],
    # the hook's own hills
    'hook': [(-20.0, 38.0, 1.4, 1.3), (-26.0, 36.8, 2.8, 1.5), (-30.5, 34.4, 2.6, 1.4), (-32.0, 30.8, 1.4, 1.2)],
    # beyond the camera: the south, the far north, the east, the far west and the western isle
    'south': [(6.0, -11.0, 1.8, 1.8), (14.0, -13.0, 4.8, 2.3), (22.0, -15.0, 5.4, 2.4), (30.0, -17.0, 4.4, 2.2),
              (37.0, -21.0, 1.8, 1.8)],
    'farnorth': [(8.0, 62.0, 2.0, 2.0), (16.0, 66.0, 5.6, 2.6), (24.0, 70.0, 6.0, 2.6), (28.0, 78.0, 4.8, 2.4),
                 (26.0, 88.0, 2.0, 2.0)],
    'eastwall': [(110.0, 70.0, 2.0, 2.0), (116.0, 58.0, 6.0, 2.8), (122.0, 44.0, 6.6, 2.9), (126.0, 30.0, 6.0, 2.8),
                 (124.0, 16.0, 5.0, 2.6), (118.0, 4.0, 2.0, 2.0)],
    'southeast': [(84.0, -20.0, 2.0, 2.0), (94.0, -14.0, 5.4, 2.6), (106.0, -12.0, 6.2, 2.8), (120.0, -16.0, 5.6, 2.7),
                  (134.0, -8.0, 4.4, 2.4), (146.0, 2.0, 2.0, 2.0)],
    'farwest': [(-176.0, 44.0, 2.0, 1.8), (-172.0, 34.0, 4.4, 2.4), (-173.0, 20.0, 4.0, 2.4), (-178.0, 6.0, 1.8, 1.8)],
    'westisle': [(-63.0, 27.0, 2.4, 1.6), (-59.0, 22.0, 4.0, 1.9), (-58.0, 16.0, 2.6, 1.6)],
}
# hill country (x, y, radius, strength): low rolling uplands where the fire finds hills to stand on
HILLS = [(-10.0, 5.0, 4.0, 0.7), (-12.0, 24.0, 4.0, 0.6), (15.0, 23.5, 4.0, 0.7), (36.0, 29.0, 5.0, 0.7),
         (28.0, 14.0, 4.0, 0.6), (6.0, -7.0, 4.5, 0.7), (40.0, 18.0, 4.0, 0.6),
         (-3.0, -12.0, 5.0, 0.8), (26.0, -4.0, 4.0, 0.7), (48.0, 34.0, 7.0, 0.8), (62.0, 32.0, 7.0, 0.9),
         (86.0, 12.0, 8.0, 0.8), (80.0, -8.0, 7.0, 0.8), (100.0, 30.0, 8.0, 0.8), (140.0, 30.0, 9.0, 0.8),
         (150.0, 60.0, 9.0, 0.8), (70.0, 60.0, 8.0, 0.8), (0.0, 58.0, 7.0, 0.8), (-12.0, 50.0, 5.0, 0.7),
         (160.0, -40.0, 9.0, 0.8), (110.0, -50.0, 8.0, 0.8)]
# deserts (x, y, rx, ry): far from the sea and in the lee, stippled; the one the camera sees lies north-east
DESERTS = [(33.0, 25.0, 12.0, 8.5), (92.0, 32.0, 20.0, 12.0), (140.0, 10.0, 16.0, 10.0), (110.0, -40.0, 14.0, 9.0)]
# river valleys cut to a steady fall (x, y) from source to mouth, half-width: the great rivers run in them
VALLEYS = [
    ('west', [(17.0, 13.4), (10.0, 12.0), (4.0, 11.4), (-2.0, 11.8), (-7.0, 13.2), (-11.5, 15.2), (-16.4, 16.4)], 1.6),
    ('north', [(3.0, 28.8), (-3.0, 27.6), (-9.0, 28.4), (-15.0, 29.4), (-21.6, 30.4)], 1.3),
    ('south', [(3.0, -3.0), (-2.0, -7.0), (-5.0, -12.0), (-7.5, -17.5)], 1.2),
]
# basins (x, y, rx, ry, depth): the few lakes (the knee's lake, the desert's salt lake, a tarn in the north-west)
BASINS = [(19.4, 14.2, 3.0, 1.0, 1.2), (31.0, 24.0, 2.3, 1.5, 1.0), (-9.5, 22.5, 1.2, 0.7, 0.7),
          (58.0, 42.0, 3.4, 2.0, 1.2), (96.0, 40.0, 3.0, 2.0, 1.0), (140.0, 50.0, 3.0, 2.0, 1.0),
          (10.0, 60.0, 2.4, 1.4, 0.9)]

# the map's few words, in the book hand (bake.labels): each one word along a curve (the letters' up is the curve's
# left): the great western sea, her range, the northern range, the desert
LABELS = [
    dict(pts=[(-27.2, 9.0), (-27.4, 13.0), (-28.4, 17.5)], n=6, xh=0.42, gap=1.0, seed=31),
    dict(pts=[(17.5, 2.6), (23.5, 1.7), (30.0, 1.6)], n=5, xh=0.42, gap=0.9, seed=32),
    dict(pts=[(-7.0, 36.2), (-0.5, 36.8), (5.5, 36.6)], n=6, xh=0.4, gap=0.9, seed=33),
    dict(pts=[(27.0, 31.0), (33.0, 31.6), (39.0, 30.6)], n=5, xh=0.45, gap=1.3, seed=34),
]

RIVER_T = 1800.0             # a channel drains at least this much rain (cells x rainfall)
WIND = 12.0                  # the rain comes from the great western sea, a little south of west (degrees)
REACH = 42.0                 # map degrees of land over which the sea's moisture fades to a third
WRING = 0.24                 # how much of the moisture every unit of climb wrings out


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
def _coast_noise(xs, ys, band):
    """The coast's fractal, in map degrees of 'inlandness', on a domain-warped plane. band 0: the coarse octaves
    (bays and capes tens of degrees across), 1: the middle ones (a few degrees), 2: the fine ones (the pen's
    scale, evaluated on the fine coast grid)."""
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
            if band == 0:
                v = 12.0 * fbm(xx * 0.028, yy * 0.028, 21, 2, 2.0, 0.5) + 5.0 * fbm(xx * 0.085, yy * 0.085, 22, 2, 2.0, 0.5)
            elif band == 1:
                v = 2.4 * fbm(xx * 0.26, yy * 0.26, 23, 2, 2.1, 0.55) + 1.0 * fbm(xx * 0.75, yy * 0.75, 24, 2, 2.1, 0.55)
            else:
                v = 0.42 * fbm(xx * 2.0, yy * 2.0, 25, 2, 2.1, 0.55) + 0.14 * gnoise(xx * 5.5, yy * 5.5, 26)
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

CAM_BOX = (-48.0, 52.0, -16.0, 46.0)     # the region the camera ever sees (+ margin): the design holds here


def relief(d):
    """(S, E, Er, sdf): the land field S (+ land, map degrees of inlandness), elevation E, the ranges' part Er,
    and the silhouettes' signed distance."""
    xs, ys = grid(d)
    X, Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    sdf = poly_sdf(d)
    # the coarse fractal is gentler where the camera looks, so the designed coast (the hook, the bay, the cape)
    # keeps its shape there; the middle and fine octaves make it intricate everywhere
    bx0, bx1, by0, by1 = CAM_BOX
    inbox = (np.clip((X - bx0) / 12.0, 0, 1) * np.clip((bx1 - X) / 12.0, 0, 1) *
             np.clip((Y - by0) / 12.0, 0, 1) * np.clip((by1 - Y) / 12.0, 0, 1))
    n0 = _coast_noise(xs, ys, 0)
    n1 = _coast_noise(xs, ys, 1)
    S = sdf + n0 * (1.0 - 0.7 * inbox) + n1
    for (ax, ay, bx, by, w) in SEAS:
        dd, _ = _seg_d(X, Y, ax, ay, bx, by)
        S -= np.clip(w - dd, 0, None) * 2.4 + 3.0 * np.exp(-(dd / w) ** 2)
    rs = np.random.default_rng(404)
    for (cx, cy, r) in ISLES:
        # a low dome, drawn out along a random bearing, that the middle octaves bite into: no isle is round
        a, el = rs.uniform(0, math.pi), rs.uniform(0.55, 1.0)
        u = ((X - cx) * math.cos(a) + (Y - cy) * math.sin(a)) / (r / el)
        v = (-(X - cx) * math.sin(a) + (Y - cy) * math.cos(a)) / (r * el)
        q = np.hypot(u, v)
        Si = r * (1.0 - q * q) + 0.95 * r * n1 / 2.4
        S = np.maximum(S, np.where(q < 2.2, Si, -1e3))
    Hc, Wc = range_fields(xs, ys)
    ridg = np.empty(X.shape, np.float32)
    roll = np.empty(X.shape, np.float32)
    _relief_noise(xs, ys, ridg, roll)
    # the ranges: a sharp crest along the spine, spurs from the ridged noise off it
    Er = Hc * (0.52 + 0.58 * ridg) + 0.8 * Wc * np.maximum(ridg - 0.55, 0) * Hc / 3.0
    # the ranges push the land out where they meet the sea (headlands, island chains)
    S = S + 0.5 * Er
    # lowland: rising gently from the coast, a little roll
    E = 0.2 + 1.15 * (1.0 - np.exp(-np.maximum(sdf, 0.0) / 9.0)) + 0.1 * roll
    for (cx, cy, r, k) in HILLS:
        dd = np.hypot(X - cx, Y - cy) / r
        E += k * np.clip(1 - dd, 0, 1) ** 1.2 * (1.25 + 0.9 * (ridg - 0.5) + 0.5 * roll)
    # the High Moor: a broad, open, lobed upland; flat on top, a gentle rim of downs
    mc, mr, mh = MOOR['c'], MOOR['r'], MOOR['h']
    dm = np.hypot(X - mc[0], (Y - mc[1]) * 1.12) / mr * (1.0 + 0.16 * n1 / 2.4)
    top = 1.0 - np.clip((dm - 0.5) / 0.5, 0, 1)
    top = top * top * (3 - 2 * top)
    E += mh * top + 0.3 * np.exp(-((dm - 0.82) / 0.1) ** 2) * (0.6 + 0.8 * ridg) + 0.06 * roll * top
    E = E + Er
    for (cx, cy, rx, ry, dep) in BASINS:
        dd = np.hypot((X - cx) / rx, (Y - cy) / ry)
        E -= dep * np.clip(1 - dd * dd, 0, 1)
    # the great rivers' valleys, cut to a steady fall from source to mouth
    for (name, pts, w) in VALLEYS:
        Q = _catmull(np.asarray(pts, np.float64), 10)
        L = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
        x0_, x1_ = Q[:, 0].min() - 3 * w, Q[:, 0].max() + 3 * w
        y0_, y1_ = Q[:, 1].min() - 3 * w, Q[:, 1].max() + 3 * w
        ix = np.where((xs >= x0_) & (xs <= x1_))[0]
        iy = np.where((ys >= y0_) & (ys <= y1_))[0]
        sx = X[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
        sy = Y[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
        best = np.full(sx.shape, 1e9, np.float32)
        sl = np.zeros(sx.shape, np.float32)
        for k in range(len(Q) - 1):
            dd, t = _seg_d(sx, sy, Q[k, 0], Q[k, 1], Q[k + 1, 0], Q[k + 1, 1])
            m = dd < best
            best = np.where(m, dd, best)
            sl = np.where(m, L[k] + t * (L[k + 1] - L[k]), sl)
        u = sl / L[-1]
        floor = 1.25 * (1.0 - u) + 0.12 * u
        sub = E[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
        carve = floor + 0.35 * (best / w) ** 2
        E[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1] = np.where(best < 2.2 * w, np.minimum(sub, carve), sub)
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
            h = (a * 73856093) ^ (b * 19349663)
            h = (h ^ (h >> 13)) * 1274126177
            e = eps * (0.2 + 1.6 * ((h & 0xFFFF) / 65536.0))
            if F[a, b] <= F[i, j] + e:
                F[a, b] = F[i, j] + e
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
    """The air from the great western sea (a little south of west): it carries the sea's moisture inland, losing
    it with distance (REACH) and wringing it out wherever the ground climbs (WRING). Returns (moist 0..1: what the
    air still carries, smoothed: it decides woods, plains and desert; rain: the relative rainfall that feeds the
    rivers)."""
    H, W = E.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), -WIND, 1.0)
    Er = cv2.warpAffine(E, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Lr = cv2.warpAffine(land.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    Mo, Pr = _moist_sweep(Er, Lr, d, REACH, WRING)
    Minv = cv2.invertAffineTransform(M)
    Mo = cv2.warpAffine(Mo, Minv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Pr = cv2.warpAffine(Pr, Minv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Mo = cv2.GaussianBlur(Mo, (0, 0), 1.5 / d)
    Pr = cv2.GaussianBlur(Pr, (0, 0), 0.8 / d)
    return (np.clip(Mo, 0, 1) * land).astype(np.float32), (np.clip(Pr, 0, 4.0) * land).astype(np.float32)


@njit(cache=True, parallel=True)
def _moist_sweep(E, L, d, reach, wring):
    H, W = E.shape
    Mo = np.zeros((H, W), np.float32)
    Pr = np.zeros((H, W), np.float32)
    for i in prange(H):
        m = 1.0
        for j in range(W):
            if L[i, j] < 0.5:
                m += (1.0 - m) * min(1.0, 0.08 * d / 0.1)
                Mo[i, j] = m
                continue
            climb = 0.0
            if j > 0:
                climb = max(E[i, j] - E[i, j - 1], 0.0)
            m *= math.exp(-d / reach - wring * climb)
            Mo[i, j] = m
            Pr[i, j] = m * (0.6 + 1.6 * min(climb / d, 3.0))
    return Mo, Pr


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
    MO, P = rain(E, land, D)
    F, rec = priority_flood(E.astype(np.float64), land, 1e-5)
    depth = (F - E) * land
    lake = (depth > 0.05) & land
    n, lab, st, _ = cv2.connectedComponentsWithStats(lake.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= 900                       # a great natural lake
    xs_, ys_ = grid(D)
    for (cx, cy, rx, ry, dep) in BASINS:              # the designed ones
        i = int(round((Y1 - cy) / D - 0.5))
        j = int(round((cx - X0) / D - 0.5))
        if 0 <= i < lab.shape[0] and 0 <= j < lab.shape[1] and lab[i, j] > 0:
            keep[lab[i, j]] = True
    lake = keep[lab]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lake = cv2.morphologyEx(lake.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool) & land
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
    _W = dict(S=S, E=E, Er=Er, land=land, rain=P.astype(np.float32), moist=MO, acc=A.astype(np.float32),
              rug=rug.astype(np.float32), lake=lake, coast=coast, lakes=lakes, rivers=rv)
    np.savez_compressed(path, S=S, E=E, Er=Er, land=land, rain=P.astype(np.float32), moist=MO, acc=A.astype(np.float32),
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
    return Sc + _coast_noise(xs, ys, 2)


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
