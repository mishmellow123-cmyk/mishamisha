"""Terrain explorer. python explore_land.py NAME key=val ... [emerge=0.3]
keys: base,bpow,bamp,aamp,gamp,hcell,hamp,hpow,hdist,spine,hmask,hfloor (LAND_PARAMS order)
Builds quick meshes for two fixed test cameras and renders them (half res)."""
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world
from mtn import land, campath

KEYS = ['base', 'bpow', 'bamp', 'aamp', 'gamp', 'hcell', 'hamp', 'hpow', 'hdist', 'spine', 'hmask',
        'hfloor']
S = '/private/tmp/claude-501/-Users-mishasalahshoor/d71369b9-7742-4da4-abf2-0e5d5e46099b/scratchpad'


def cam(pos, yaw, pitch, hfov=58.0, frame=0.0):
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    r, u = campath.basis_from(fwd, 0.0)
    q = campath.mat_to_quat(np.stack([r, u, -fwd], 1))
    return dict(frame=frame, loc=list(map(float, pos)), quat=q.tolist(), hfov=hfov,
                fwd=fwd.tolist(), right=r.tolist(), up=u.tolist())


TEST_CAMS = [cam((180, 900, 1100), 10, -6, 58, 1), cam((-40, 250, 230), 12, 1, 62, 2),
             cam((-60, -260, 360), 3, -3, 62, 3)]

if __name__ == '__main__':
    name = sys.argv[1]
    emerge = None
    LP = world.LAND_PARAMS.copy()
    for kv in sys.argv[2:]:
        k, v = kv.split('=')
        if k == 'emerge':
            emerge = float(v)
        else:
            LP[KEYS.index(k)] = float(v)
    world.LAND_PARAMS = LP
    world.LAND_VERSION = 'x' + name
    # set BASE so `emerge` of the natural land is above the cloud top (zone/bumps off)
    SP = world.spine_array()
    E = np.zeros((0, 7))
    xs = np.linspace(-15000, 15000, 120)
    ys = np.linspace(-3000, 30000, 130)
    X, Y = np.meshgrid(xs, ys)
    if emerge is not None:
        LP0 = LP.copy()
        LP0[0] = 0.0
        g = land.eval_points(X.ravel(), Y.ravel(), np.full(X.size, 150.0), E, SP, np.zeros((0, 8)),
                             world.ZONE.copy(), LP0)
        LP[0] = -np.percentile(g, 100 * (1 - emerge)) / land.LS
    world.LAND_PARAMS = LP
    world._ARGS = None
    print(name, 'BASE', round(LP[0]), 'LP', LP.round(2).tolist())
    import build_mesh as BM
    frames = []
    for c in TEST_CAMS:
        frames.append(c)
    json.dump(dict(frames=frames), open(os.path.join(HERE, 'cache', 'x%s_cam.json' % name), 'w'))
    cams = BM.cams_from_frames(frames)
    BM.build('x' + name, 0.008, 2.5, {'moon': (world.MOON_AZ, world.MOON_EL, 0.35)}, cams=cams,
             quick=True, cloud_k=2.0)
    subprocess.run(['python', 'landmap.py', os.path.join(S, 'xmap_%s.png' % name), '-15000', '-3000',
                    '30000', '600'], env=dict(os.environ, XLP=json.dumps(LP.tolist())))
