"""Build a preview terrain mesh for one camera and write a camera json (venv python)."""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtn import land, mesher
import world

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE, exist_ok=True)


def look_basis(pos, target, roll=0.0):
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u


if __name__ == '__main__':
    pos = np.array([float(v) for v in os.environ.get('CAMPOS', '0,-1500,700').split(',')])
    tgt = np.array([float(v) for v in os.environ.get('CAMTGT', '0,6000,200').split(',')])
    hfov = float(os.environ.get('HFOV', 60))
    f, r, u = look_basis(pos, tgt)
    thx = math.tan(math.radians(hfov / 2))
    thy = thx * 804 / 1920
    t0 = time.time()
    path = pos[None, :2]
    R = 60000.0
    cull = mesher.frustum_cull_xy([(pos, f, r, u, thx, thy)], -300.0, 2200.0, 10.0)
    K = float(os.environ.get('K', 0.004))
    P = mesher.point_set((pos[0] - R, pos[1] - R, pos[0] + R, pos[1] + R), path, K, 2.0, 800.0, cull=cull)
    print('points', P.shape, time.time() - t0)
    t0 = time.time()
    Z = land.eval_points(P[:, 0].copy(), P[:, 1].copy(), P[:, 2] * 0.7, *world.land_args())
    d2 = (P[:, 0] - pos[0]) ** 2 + (P[:, 1] - pos[1]) ** 2
    Z -= d2 / (2 * land.R_EARTH)
    print('heights', time.time() - t0, Z.min(), Z.max())
    # thin out terrain buried well under the cloud sea (keep a 4x coarser lattice there)
    s_ = P[:, 2]
    on4 = (np.abs(np.round(P[:, 0] / (4 * s_)) * 4 * s_ - P[:, 0]) < 1e-3) & (np.abs(np.round(P[:, 1] / (4 * s_)) * 4 * s_ - P[:, 1]) < 1e-3)
    keep = (Z > -150.0) | on4
    P = P[keep]; Z = Z[keep]
    print('after cloud thinning', P.shape)
    t0 = time.time()
    T = mesher.triangulate(P)
    print('tris', T.shape, time.time() - t0)
    V = np.stack([P[:, 0], P[:, 1], Z], 1)
    mesher.save_mesh(os.path.join(CACHE, 'preview_land'), V, T, {})
    cam = dict(pos=pos.tolist(), target=tgt.tolist(), hfov=hfov)
    json.dump(cam, open(os.path.join(CACHE, 'preview_cam.json'), 'w'))
