"""DAWN in the east (cut C; v2 frames 2400-2655): camera, sun timing, eagles, smoke plumes.

python shot_dawn.py [check]  -> cache/dawn_cam.json, cache/dawn_sun.json, cache/dawn_eagles.json,
                                cache/dawn_beacons.json (the night's fires, now smoking)

The same world as THE BEACON RUN, the morning after. The camera hangs high over the cloud sea
looking east along SUN_AZ and rises slowly. The sun's elevation is solved from the camera's own
horizon toward the sun so that its upper limb clears the eastern ridge exactly at 2400 (the
burst), then climbs ~0.33 deg/s (time-lapse) so the light floods down the peaks and across the
cloud sea. A flight of great eagles crosses the light 2430-2560.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mtn import campath, land, bake
import world
from world_bl import SUN_AZ

CACHE = os.path.join(HERE, 'cache')
F0, F1 = 2400, 2655
HANDLE = 3
FRAMES = np.arange(F0 - HANDLE, F1 + HANDLE + 1)
W, H = 1920, 804
SUN_R = math.radians(0.42)          # sun disc radius (drawn ~1.6x real for presence)
SUN_RATE = math.radians(0.33) / 24.0  # rise per frame

# camera: high over the cloud sea south-east of the run, looking east; slow crane up + drift
CAM0 = np.array([2600.0, 2400.0, 640.0])
CAM1 = np.array([2860.0, 2440.0, 820.0])
YAW0, YAW1 = SUN_AZ - 7.0, SUN_AZ - 4.0     # the sun sits a little right of centre
PITCH0, PITCH1 = 2.2, 1.2
HFOV = 50.0


def ease(t):
    return t * t * (3 - 2 * t)


def cam_frames():
    out = []
    for f in FRAMES:
        u = (f - F0) / (F1 - F0)
        uc = min(max(u, -0.05), 1.05)
        # the crane: a slow start (the burst holds), then a steady rise that settles
        e = 0.5 * (uc + ease(min(max(uc, 0), 1)))
        p = CAM0 + (CAM1 - CAM0) * e
        yaw = math.radians(YAW0 + (YAW1 - YAW0) * e)
        pitch = math.radians(PITCH0 + (PITCH1 - PITCH0) * e)
        # breathing: tiny drift so nothing is static
        t = f / 24.0
        yaw += math.radians(0.12 * math.sin(0.31 * t + 1.0))
        pitch += math.radians(0.08 * math.sin(0.43 * t))
        fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch),
                        math.sin(pitch)])
        roll = math.radians(0.25 * math.sin(0.27 * t + 2.0))
        r, uvec = campath.basis_from(fwd, roll)
        q = campath.mat_to_quat(np.stack([r, uvec, -fwd], 1))
        out.append(dict(frame=float(f), loc=p.tolist(), quat=q.tolist(), hfov=HFOV,
                        fwd=fwd.tolist(), right=r.tolist(), up=uvec.tolist(), speed=0.0,
                        yaw=math.degrees(yaw), pitch=math.degrees(pitch),
                        roll=math.degrees(roll)))
    return out


def sun_track(cams, G, ref):
    """Sun elevation per frame: its top limb meets the camera's horizon at F0."""
    (G0, M0), (G1, M1), (G2, M2) = G
    byf = {int(c['frame']): c for c in cams}
    c = np.array(byf[F0]['loc'])
    hz = bake.horizon_angle(np.array([c[0]]), np.array([c[1]]), np.array([c[2] - 0.05]),
                            math.radians(SUN_AZ), G0, M0, G1, M1, G2, M2, 260000.0)[0]
    e0 = hz - SUN_R + math.radians(0.03)       # a sliver of the limb shows on the hit
    track = {int(f): float(e0 + (f - F0) * SUN_RATE) for f in FRAMES}
    return track, float(hz)


def eagles():
    """Six great eagles in a loose echelon crossing right -> left in front of the light. Heights
    are set by the angle above/below the view axis so they pass through the sun's band."""
    rng = np.random.default_rng(7)
    out = []
    base_t0 = 2430
    angles = [2.6, 1.5, 0.3, -0.9, 3.4, -0.3]          # deg relative to the view axis
    for k in range(6):
        dist = 300.0 + 60.0 * k + rng.uniform(-25, 25)
        t0 = base_t0 + 10 * k + rng.uniform(-3, 3)
        dur = 150 + rng.uniform(-10, 10)
        span = 11.0 + rng.uniform(-1.5, 2.0)            # great eagles
        height = dist * math.tan(math.radians(angles[k]))
        out.append(dict(id=k, dist=dist, t0=t0, dur=dur, span=span, height=height,
                        phase=float(rng.uniform(0, 1)), flap_period=float(rng.uniform(36, 44)),
                        glide=float(rng.uniform(0.25, 0.45))))
    return out


def plumes(cams, ref, G, n_want=6):
    """The night's beacons seen from the dawn camera: natural summits in view (3-30 km), visible,
    spread across the frame. They still smoke."""
    import shot_run as SR
    byf = {int(c['frame']): c for c in cams}
    (G0, M0), (G1, M1), (G2, M2) = G
    c0 = byf[F0]
    c1 = byf[F1]
    cl = np.array(c0['loc'])
    C = SR.summit_candidates(G)
    cand = []
    for (x, y, z) in C:
        d = math.hypot(x - cl[0], y - cl[1])
        if not (2500 < d < 32000) or z < 150:
            continue
        zc = z - ((x - ref[0]) ** 2 + (y - ref[1]) ** 2) / (2 * 6.371e6)
        p0 = SR.project(c0, (x, y, zc))
        p1 = SR.project(c1, (x, y, zc))
        if p0 is None or p1 is None:
            continue
        if not (80 < p0[0] < W - 80 and 60 < p0[1] < H - 60 and 80 < p1[0] < W - 80):
            continue
        # keep the sun's neighbourhood clear (the burst) and the eagles' band readable
        if abs(p0[0] - W * 0.58) < 170 and p0[1] < H * 0.55:
            continue
        if not bake.visible(G0, M0, G1, M1, G2, M2, cl[0], cl[1], cl[2], x, y, zc + 3.0,
                            ref[0], ref[1]):
            continue
        cand.append((z / (d ** 0.35), x, y, z, p0[0], d))
    cand.sort(reverse=True)
    out = []
    for sc, x, y, z, sx, d in cand:
        if any(abs(sx - o['sx']) < 170 for o in out):
            continue
        out.append(dict(name='D%d' % len(out), pos=(float(x), float(y), float(z)), t=-10000,
                        scale=1.3 + d / 20000.0, sx=float(sx)))
        if len(out) >= n_want:
            break
    return out


if __name__ == '__main__':
    import build_mesh as BM
    cams = cam_frames()
    campath.save(os.path.join(CACHE, 'dawn_cam.json'), cams, dict(F0=F0, F1=F1))
    G = BM.grids()
    ref = np.array([c['loc'] for c in cams]).mean(0)
    track, hz = sun_track(cams, G, ref)
    json.dump(dict(az=SUN_AZ, elev=track, horizon=hz, radius=SUN_R),
              open(os.path.join(CACHE, 'dawn_sun.json'), 'w'), indent=0)
    json.dump(eagles(), open(os.path.join(CACHE, 'dawn_eagles.json'), 'w'), indent=1)
    P = plumes(cams, ref, G)
    json.dump(P, open(os.path.join(CACHE, 'dawn_beacons.json'), 'w'), indent=1)
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        print('camera horizon toward the sun at 2400: %.2f deg' % math.degrees(hz))
        print('sun elevation 2400 %.2f  2500 %.2f  2655 %.2f deg' % tuple(
            math.degrees(track[f]) for f in (2400, 2500, 2655)))
        print('smoking beacons in view:', [b['name'] for b in P])
