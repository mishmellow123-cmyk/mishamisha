"""Build the terrain and cloud-sea meshes for a shot's camera path, with baked attributes.

python build_mesh.py run|dawn [--k 0.0035] [--smin 0.6]
Reads cache/<shot>_cam.json; writes cache/<shot>_land.{json,bin} and cache/<shot>_cloud.*
Grids of the land (camera independent) are cached in cache/grids.npz.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mtn import land, mesher, bake, cloud
import world

CACHE = os.path.join(HERE, 'cache')
R_EARTH = 6.371e6


def grids(force=False, quick=False, tag=None):
    tag = tag or world.LAND_VERSION
    path = os.path.join(CACHE, 'grids_%s%s.npz' % (tag, '_q' if quick else ''))
    if os.path.exists(path) and not force:
        z = np.load(path)
        return [(z['g%d' % i], z['m%d' % i]) for i in range(3)]
    args = world.land_args()
    if quick:
        specs = [(world.ZONE[0], world.ZONE[1], 2600.0, 1040),
                 (1000.0, 12000.0, 44000.0, 1100),
                 (0.0, 60000.0, 520000.0, 1040)]
    else:
        specs = [  # (cx, cy, size, n)
            (world.ZONE[0], world.ZONE[1], 2600.0, 2080),
            (1000.0, 12000.0, 44000.0, 2200),
            (0.0, 60000.0, 520000.0, 2080),
        ]
    out = {}
    res = []
    for i, (cx, cy, size, n) in enumerate(specs):
        t0 = time.time()
        dx = size / (n - 1)
        x0, y0 = cx - size / 2, cy - size / 2
        g = land.raster(n, n, x0, y0, dx, *args)
        m = np.array([x0, y0, dx])
        out['g%d' % i] = g
        out['m%d' % i] = m
        res.append((g, m))
        print('grid', i, 'dx %.2f' % dx, '%.1fs' % (time.time() - t0), flush=True)
    np.savez(path, **out)
    return res


def cams_of(shot):
    cj = json.load(open(os.path.join(CACHE, '%s_cam.json' % shot)))
    return cams_from_frames(cj['frames'])


def cams_from_frames(fr):
    cams = []
    for c in fr:
        thx = math.tan(math.radians(c['hfov'] / 2))
        cams.append((np.array(c['loc']), np.array(c['fwd']), np.array(c['right']),
                     np.array(c['up']), thx, thx * 804 / 1920))
    return cams, np.array([c['loc'] for c in fr])


def curvature(X, Y, ref):
    return ((X - ref[0]) ** 2 + (Y - ref[1]) ** 2) / (2 * R_EARTH)


def build(shot, k, smin, light_dirs, sun_az=None, cams=None, quick=False, cloud_k=2.4):
    t0 = time.time()
    G = grids(quick=quick)
    (G0, M0), (G1, M1), (G2, M2) = G
    if cams is None:
        cams, locs = cams_of(shot)
    else:
        cams, locs = cams
    ref = locs.mean(0)
    sub = cams[::3] if len(cams) > 12 else cams
    path_xy = locs[:, :2]
    zlo, zhi = -250.0, 3200.0
    cull = mesher.frustum_cull_xy(sub, zlo, zhi, 12.0, 120.0)
    R = 260000.0
    P = mesher.point_set((ref[0] - R, ref[1] - R, ref[0] + R, ref[1] + R), path_xy, k, smin, 900.0,
                         cull=cull)
    print('land candidates', len(P), '%.1fs' % (time.time() - t0), flush=True)
    args = world.land_args()
    Z = land.eval_points(P[:, 0].copy(), P[:, 1].copy(), P[:, 2] * 0.7, *args)
    # thin the land buried under the cloud (keep a 4x coarser lattice)
    s_ = P[:, 2]
    on4 = ((np.abs(np.round(P[:, 0] / (4 * s_)) * 4 * s_ - P[:, 0]) < 1e-3)
           & (np.abs(np.round(P[:, 1] / (4 * s_)) * 4 * s_ - P[:, 1]) < 1e-3))
    keep = (Z > -70.0) | on4
    P, Z = P[keep], Z[keep]
    P[Z <= -70.0, 2] *= 4.0            # their true spacing (for the long-edge filter)
    print('land points', len(P), '%.1fs' % (time.time() - t0), flush=True)
    T = mesher.triangulate(P)
    print('land tris', len(T), '%.1fs' % (time.time() - t0), flush=True)
    X, Y = P[:, 0].copy(), P[:, 1].copy()
    attrs = {}
    for name, (az, el, pen) in light_dirs.items():
        ha = bake.horizon_angle(X, Y, Z, math.radians(az), G0, M0, G1, M1, G2, M2, 60000.0)
        e = math.radians(el)
        attrs[name] = np.clip((e - ha) / (2 * math.radians(pen)) + 0.5, 0, 1).astype(np.float32)
        print('bake', name, '%.1fs' % (time.time() - t0), flush=True)
    if sun_az is not None:
        attrs['sunhor'] = bake.horizon_angle(X, Y, Z, math.radians(sun_az), G0, M0, G1, M1, G2, M2,
                                             260000.0).astype(np.float32)
        print('bake sunhor', '%.1fs' % (time.time() - t0), flush=True)
    # snow from the land's own structure: slope at two scales, gully concavity at two scales
    s1, c1 = bake.slope_curv(X, Y, G0, M0, G1, M1, G2, M2, 4.0)
    s2, c2 = bake.slope_curv(X, Y, G0, M0, G1, M1, G2, M2, 22.0)
    s3, c3 = bake.slope_curv(X, Y, G0, M0, G1, M1, G2, M2, 60.0)
    th = np.degrees(np.arctan(0.5 * s1 + 0.5 * s2))
    snow = np.clip((70.0 - th) / 24.0, 0, 1)
    snow = snow * snow * (3 - 2 * snow)
    steep = np.clip((th - 30.0) / 20.0, 0, 1)
    cc = 0.6 * np.tanh(c2 / 0.010) + 0.4 * np.tanh(c3 / 0.004)
    snow += steep * (0.55 * np.clip(cc, 0, 1) - 0.45 * np.clip(-cc, 0, 1))
    # the story summits (the cairn pads) are snow domes: the fire lights snow, not black rock
    Pw = world.land_args()[0]
    for pr in Pw[Pw[:, 4] == 1]:
        dd = np.hypot(X - pr[0], Y - pr[1])
        w = np.clip((45.0 - dd) / 30.0, 0, 1)
        snow = np.maximum(snow, 0.62 * w * w * (3 - 2 * w))
    attrs['snow'] = np.clip(snow, 0, 1).astype(np.float32)
    attrs['slope'] = th.astype(np.float32)
    attrs['cz'] = cloud.eval_points(X, Y, np.full(len(X), 60.0)).astype(np.float32)
    # sky visibility (large-scale ambient occlusion) from 6 horizon directions
    ao = np.zeros(len(X))
    for az in np.linspace(0, 2 * math.pi, 6, endpoint=False):
        ha = bake.horizon_angle(X, Y, Z, az, G0, M0, G1, M1, G2, M2, 450.0, 1.0, 1.08)
        ao += np.cos(np.clip(ha, 0, math.pi / 2)) ** 2
    attrs['ao'] = (ao / 6).astype(np.float32)
    print('bake ao', '%.1fs' % (time.time() - t0), flush=True)
    Zc = Z - curvature(X, Y, ref)
    V = np.stack([X, Y, Zc], 1)
    mesher.save_mesh(os.path.join(CACHE, '%s_land' % shot), V, T, attrs,
                     dict(ref=ref.tolist(), k=k, smin=smin))
    print('land saved', V.shape, '%.1fs' % (time.time() - t0), flush=True)
    del P, Z, T, V

    # ------------------------------------------------------------ cloud sea ---
    cull = mesher.frustum_cull_xy(sub, -80.0, 80.0, 12.0, 120.0)
    Pc = mesher.point_set((ref[0] - R, ref[1] - R, ref[0] + R, ref[1] + R), path_xy, cloud_k * k,
                          max(smin * 4, 3.0), 2500.0, cull=cull)
    Xc, Yc = Pc[:, 0].copy(), Pc[:, 1].copy()
    Zc0 = cloud.eval_points(Xc, Yc, Pc[:, 2] * 0.7)
    Zt = land.eval_points(Xc, Yc, np.maximum(Pc[:, 2], 6.0), *args)
    clear = Zc0 - Zt
    s_ = Pc[:, 2]
    on4 = ((np.abs(np.round(Pc[:, 0] / (4 * s_)) * 4 * s_ - Pc[:, 0]) < 1e-3)
           & (np.abs(np.round(Pc[:, 1] / (4 * s_)) * 4 * s_ - Pc[:, 1]) < 1e-3))
    keep = (clear > -40.0) | on4
    Xc, Yc, Zc0, clear, Pc = Xc[keep], Yc[keep], Zc0[keep], clear[keep], Pc[keep]
    Pc[clear <= -40.0, 2] *= 4.0
    Tc = mesher.triangulate(Pc)
    print('cloud', len(Xc), 'pts', len(Tc), 'tris', '%.1fs' % (time.time() - t0), flush=True)
    cat = {'clear': clear.astype(np.float32)}
    # cloud height grids (fine near the path, coarse far) for the deck's self-shadowing
    pc = locs.mean(0)
    C0 = cloud.raster(1100, 1100, pc[0] - 5500, pc[1] - 3000, 10.0)
    CM0 = np.array([pc[0] - 5500, pc[1] - 3000, 10.0])
    C1 = cloud.raster(1000, 1000, pc[0] - 40000, pc[1] - 20000, 80.0)
    CM1 = np.array([pc[0] - 40000, pc[1] - 20000, 80.0])
    cz_min = np.full((2, 2), -1e9)
    CMx = np.array([0.0, 0.0, 1.0])
    for name, (az, el, pen) in light_dirs.items():
        ha = bake.horizon_angle(Xc, Yc, Zc0, math.radians(az), G0, M0, G1, M1, G2, M2, 90000.0)
        hc = bake.horizon_angle(Xc, Yc, Zc0 + 0.5, math.radians(az), C0, CM0, C1, CM1, cz_min, CMx, 6000.0, 2.0, 1.06)
        ha = np.maximum(ha, hc)
        e = math.radians(el)
        cat[name] = np.clip((e - ha) / (2 * math.radians(pen + 1.2)) + 0.5, 0, 1).astype(np.float32)
    # hollows of the deck (for a soft occlusion term)
    sC, cC = bake.slope_curv(Xc, Yc, C0, CM0, C1, CM1, cz_min, CMx, 25.0)
    cat['hollow'] = np.clip(cC / 0.03, -1, 1).astype(np.float32)
    if sun_az is not None:
        cat['sunhor'] = bake.horizon_angle(Xc, Yc, Zc0, math.radians(sun_az), G0, M0, G1, M1, G2,
                                           M2, 260000.0).astype(np.float32)
    Vc = np.stack([Xc, Yc, Zc0 - curvature(Xc, Yc, ref)], 1)
    mesher.save_mesh(os.path.join(CACHE, '%s_cloud' % shot), Vc, Tc, cat, dict(ref=ref.tolist()))
    print('cloud saved', '%.1fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('shot')
    ap.add_argument('--k', type=float, default=0.0035)
    ap.add_argument('--smin', type=float, default=0.6)
    ap.add_argument('--grids', action='store_true', help='rebuild grids')
    ap.add_argument('--cloudk', type=float, default=2.4)
    a = ap.parse_args()
    if a.grids:
        grids(True)
    if a.shot == 'run':
        build('run', a.k, a.smin, {'moon': (world.MOON_AZ, world.MOON_EL, 0.35)}, cloud_k=a.cloudk)
    elif a.shot == 'dawn':
        build('dawn', a.k, a.smin, {}, sun_az=world.SUN_AZ, cloud_k=a.cloudk)
