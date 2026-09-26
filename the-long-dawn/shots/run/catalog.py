"""Summit catalogue of the run's world (foothills + s1's islands) for placing beacons.
   python catalog.py  -> writes shots/run/summits.npy (x, y, z, prominence)"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world as WD      # noqa: E402
import run as RN        # noqa: E402


def grid(x0, z0, dx, n):
    H = np.zeros((n, n))
    WD.height_grid(x0, z0, dx, n, n, RN.CR, 63.0, H)
    return H


def peaks(H, x0, z0, dx, win, floor):
    mx = ndi.maximum_filter(H, size=win)
    mn = ndi.minimum_filter(H, size=win * 2 + 1)
    out = []
    for j, i in np.argwhere((H == mx) & (H > floor)):
        out.append((x0 + i * dx, H[j, i], z0 + j * dx, H[j, i] - mn[j, i]))
    return out


def main():
    cat = []
    # near: the foothills (8 m grid, 3 km square in front of the summit)
    cat += peaks(grid(-1500.0, -300.0, 8.0, 400), -1500.0, -300.0, 8.0, 9, -560.0)
    # far: s1's islands (60 m grid, 40 km square)
    far = peaks(grid(-20000.0, -4000.0, 60.0, 667), -20000.0, -4000.0, 60.0, 9, -560.0)
    cat += [p for p in far if abs(p[0]) > 1500 or p[2] > 2700 or p[2] < -300]
    cat = np.array(cat)
    np.save(os.path.join(HERE, 'summits.npy'), cat)
    print(len(cat), 'summits')


if __name__ == '__main__':
    main()
