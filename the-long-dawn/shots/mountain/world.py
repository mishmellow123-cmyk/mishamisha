"""The MOUNTAIN world: story geography shared by every script (pure python + numpy).

World units metres; x east, y north, z up; cloud-sea top ~ z = 0.
The run flies north from the shepherd's summit (S0) and the signal races north-north-east along
the main range (the 'chain spine') to the horizon.
"""
import math

import numpy as np

LAND_VERSION = 'v17'          # bump when the land function changes (grid cache key)
from world_bl import MOON_AZ, MOON_EL, SUN_AZ

# land parameters (scaled-domain units; see mtn.land.base_height / horns)
#  0 BASE, 1 body power, 2 body amp, 3 arete amp, 4 gully amp, 5 horn cell, 6 horn amp,
#  7 horn profile power, 8 horn height-dist power, 9 spine boost, 10 horn mask t0,
#  11 horn floor (fraction of horns outside the high body)
LAND_PARAMS = np.array([-1157.0, 1.8, 2600.0, 500.0, 30.0, 3000.0, 3600.0, 1.45, 1.15, 0.30, 0.2, 0.3])

# story places (x, y)
MARKS = {
    'S0': (0.0, 0.0),          # the shepherd's summit (his beacon, 1480)
    'B3': (40.0, 470.0),       # the great pyre on the ridge we skim (1580)
    'B1': (-700.0, 3000.0),    # 1540
    'B2': (2150.0, 3350.0),    # 1560
    'B4': (-300.0, 5800.0),    # 1600
    'B5': (1400.0, 10000.0),   # 1620
    'B6': (3800.0, 17000.0),   # 1640
    'B7': (8500.0, 31000.0),   # 1660
}

# the main range the chain follows (polyline, metres)
SPINE = np.array([
    [-900.0, 4200.0], [-300.0, 5800.0], [600.0, 8000.0], [1400.0, 10000.0], [2500.0, 13500.0],
    [3800.0, 17000.0], [5600.0, 23000.0], [8500.0, 31000.0], [11500.0, 42000.0],
    [14000.0, 56000.0], [16000.0, 75000.0], [17000.0, 100000.0], [17500.0, 130000.0],
])


def peaks_table():
    """Designed land edits.
    kind 0 = designed horn peak (x, y, summit height, radius, 0, 0, 0)
    kind 2 = broad gaussian bump/depression (x, y, amp, radius, 2, 0, 0)
    kind 1 = flat pad for a cairn (x, y, pad height, radius, 1, 0, 0).
    Horns placed for the run's composition are read from cache/design_peaks.json."""
    rows = [
        (0.0, 2100.0, -380.0, 1100.0, 2, 0, 0),     # open the basin north of the pyre ridge
    ]
    import json
    import os
    dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'design_peaks.json')
    if os.path.exists(dp):
        for r in json.load(open(dp)):
            kind = r.get('kind', 0)
            rows.append((r['x'], r['y'], r['H'], r['R'], kind, 0, 0))
            if kind == 0:
                MARKS[r['name']] = (r['x'], r['y'])
    rows += [
        # cairn pads, pinned where v1 found them (so the camera path never shifts); col 5 = 1:
        # truncate the summit into a small rocky plateau 5 m below its natural top
        (7.5, -4.0, 0.0, 10.0, 1, 1, 1),            # S0 (the shepherd's beacon)
        (29.5, 467.5, 0.0, 9.0, 1, 1, 1),           # B3 (the great pyre)
    ]
    return np.array(rows, np.float64)


def reset_args():
    global _ARGS
    _ARGS = None


def spine_array():
    return SPINE.copy()


# ------------------------------------------------------------------ story zone ---
# The shepherd's summit (S0) and the great-pyre ridge (B3) are designed horn peaks; the natural
# land's own fine detail is added on top, and the zone blends into the natural land.
ZONE = np.array([150.0, 450.0, 1150.0, 1750.0, 220.0])      # cx, cy, r_in, r_out, detail fp


def zone_table():
    """Story zone horns: x, y, H (m above the cloud top), R, n_faces, orientation, power,
    stretch (< 1 elongates the horn along its orientation).
    Profile z = -160 + (H + 160)(1 - q)^power, q = polygonal distance / R."""
    rows = [
        # the shepherd's summit and its satellites
        (0.0, 0.0, 330.0, 700.0, 4, 0.96, 1.7, 0.8),
        (-300.0, -300.0, 240.0, 600.0, 3, 1.10, 1.5, 0.7),
        (360.0, -230.0, 225.0, 600.0, 3, 0.20, 1.5, 0.7),
        # the great-pyre ridge (E-W, bearing ~ -0.15 rad): elongated horns -> a serrated crest;
        # the pyre knoll at (35, 470)
        (-560.0, 575.0, 245.0, 520.0, 3, -0.15, 1.35, 0.45),
        (-320.0, 535.0, 212.0, 480.0, 4, -0.15, 1.35, 0.45),
        (-130.0, 505.0, 178.0, 450.0, 3, -0.15, 1.4, 0.5),
        (35.0, 470.0, 208.0, 420.0, 4, 0.96, 1.4, 0.75),
        (240.0, 448.0, 160.0, 450.0, 3, -0.15, 1.4, 0.5),
        (470.0, 425.0, 200.0, 500.0, 4, -0.15, 1.35, 0.45),
        (730.0, 380.0, 165.0, 520.0, 3, -0.15, 1.35, 0.45),
    ]
    return np.array(rows, np.float64)


_ARGS = None


def land_args():
    """Arguments for mtn.land.height/eval_points/raster after (x, y, fp). Pads are moved to the
    real local summit near their design point and set to its height (cached)."""
    global _ARGS
    if _ARGS is not None:
        return _ARGS
    from mtn import land
    P = peaks_table()
    SP, Z, ZC = spine_array(), zone_table(), ZONE.copy()
    LP = LAND_PARAMS.copy()
    pad_rows = [k for k in range(len(P)) if P[k, 4] == 1]
    P0 = P.copy()
    P0[pad_rows, 4] = 9          # disabled
    names = {0: 'S0', 1: 'B3'}
    for n, k in enumerate(pad_rows):
        x, y = P[k, 0], P[k, 1]
        if P[k, 6] != 1:
            for rad, step in ((40.0, 4.0), (6.0, 0.5)):
                xs = np.arange(x - rad, x + rad + 1e-6, step)
                X, Y = np.meshgrid(xs, np.arange(y - rad, y + rad + 1e-6, step))
                H = land.eval_points(X.ravel(), Y.ravel(), np.full(X.size, 2.0), P0, SP, Z, ZC, LP)
                i = int(np.argmax(H))
                x, y = X.ravel()[i], Y.ravel()[i]
        P[k, 0], P[k, 1] = x, y
        hn = land.eval_points(np.array([x]), np.array([y]), np.array([1.0]), P0, SP, Z, ZC, LP)[0]
        P[k, 2] = hn - (5.0 if P[k, 5] > 0 else 0.3)
        MARKS[names.get(n, 'pad%d' % n)] = (float(x), float(y))
    _ARGS = (P, SP, Z, ZC, LP)
    return _ARGS


def ground(x, y, fp=1.0):
    from mtn import land
    return float(land.eval_points(np.array([float(x)]), np.array([float(y)]), np.array([fp]), *land_args())[0])
