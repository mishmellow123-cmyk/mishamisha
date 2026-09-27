"""A14 · THE BEACON RUN (R3, cut A): A cut frames 3920-4239 (bars 50-53) -> renders/beaconrun_A/.

The cut follows the desert figure's look into the depth of the frame. We come in over A's knife-edge ranges on H1's
moonlit night, heading toward the cold glow (az -30), which still breathes once a bar beyond the far ranges. Under
the cloud sea the red patches of the race throb on every beat. The fires that caught with hers (A13) are already
burning on far ridges. Now the watch links across the ranges ahead of the camera, every two beats, seven times,
from bar 50 b3 to bar 53 b3. Near and far alternate, so the far answer is heard on the second fire, bar 51 b1.
The camera is a slow, high glider on a long lens. It never overtakes a fire, every fire stays ahead of it, and
there is always one in frame. It descends and settles behind the seventh fire, the nearest, which catches on bar
53 b3 beside the one who lit it. A15 R16 THE WATCHERS (RUN-A3) holds from this last position: it imports
end_cam(), FIRES, CR and light() from here.

Shared night (nighta.py, RUN-A3's kit): the glow, the red under the cloud and the fire recipe are the same in
A13-A15.

  farm look-dev: python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/beaconrun_a_look.json --test K
  local test:    python3 shots/run/beaconrun_a.py --frames 3960,4200 --scale 0.5 --out tests
  env look-dev:  A14_CAM="x,y,z,yaw,pitch,hfov" (a static camera for every frame), A14_LABELS=1 (catalog summit
                 indices and the chain's numbers drawn on the frame)
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import falsedawn as FD      # noqa: E402  (A's great sierra, FD_RANGE, and the glow model through nighta)
import nighta as NA         # noqa: E402
from mt import sky as SK    # noqa: E402

FPS = 24.0
FR0, NFR = 3920, 320
BAR, BEAT = 80, 20
IGN = [3960 + 40 * k for k in range(7)]          # bar 50 b3 ... bar 53 b3, every two beats
SHUTTER = 0.35

CR = FD.terrain_rows()                           # the Run world + A's great knife-edge sierra (A2's landform)
CAT = np.load(os.path.join(HERE, 'summits.npy'))

# ------------------------------------------------------------------ the light (A15 imports this) ---


def light(f=None):
    """H1's moonlit night (world.night_light) with the crossing's terrain fixes: the anti-streak snow noise (Q[18]),
    snow held on steeper ground, and more aerial depth so the ranges step back in veils."""
    Lk, amb, S, fogp, Q = WD.night_light()
    Q = Q.copy()
    Q[18] = 1.0
    Q[3] = 0.30
    Q[4] = 0.54
    fogp = fogp.copy()
    fogp[0] = 1.0e-4
    return Lk, amb, S, fogp, Q


# ------------------------------------------------------------------ the chain (world-fixed) ---
# (x, y, z, f_ign, size, seed): nighta's fire rows, picked by _local_logs/runA/pick_chain.py and committed as
# beaconrun_a_chain.npy. The seventh summit is RUN-A3's best brink (-1004, -508): level where its lighter stands,
# falling away beyond it toward the glow. Every fire is SEEN (with a 1 m sightline margin) from both the glider's
# start and the plate (A15). The near chain comes in toward us (1: 1.07 km, 3: 530 m, 5: 190 m, 7: here), and so
# does the far chain answering it (2: 18.5 km, the far answer on bar 51 b1; 4: 9.3 km, high on the great sierra;
# 6: 5.2 km).
CHAIN = np.load(os.path.join(HERE, 'beaconrun_a_chain.npy'))
FIRES = CHAIN                                    # A15's name for the seven (ignition order)
_LITF = os.path.join(HERE, 'reveal_a_fires.npy')  # A13's fires (RUN-A3): already burning, far off (her cluster)
LIT = np.load(_LITF) if os.path.exists(_LITF) else np.zeros((0, 6))
LT_MAX = 3000.0                                  # a fire's pool on the snow only matters near the camera


def extra():
    """RUN-A3's EXTRA watch-fires on the ridges between (built against CHAIN and END_CAM); empty until they land."""
    try:
        import watchers_a as WA
        E = np.asarray(WA.extra_fires(), np.float64).reshape(-1, 6)
    except Exception:
        E = np.zeros((0, 6))
    return E


def fires_table():
    return np.vstack([LIT, extra(), CHAIN])


# ------------------------------------------------------------------ the camera ---
# A slow crane-glide: from 75 m over the land behind the seventh summit, forward and down along the plate's axis,
# settling at the seventh lighter's eye (A15's plate, watchers_a.plate) by 4230 and holding there. The seventh
# summit rises into frame as we come down. Pitch opens from looking down over the ranges to the plate's -1.8.
T_SETTLE = 4230
BACK0, UP0 = 120.0, 75.0
PITCH0, HFOV0 = -6.5, 44.0
_PLATE = None


def end_cam():
    """(pos, yaw, pitch, hfov) at cut 4239 = A15's plate (RUN-A3's watchers_a.plate at the seventh fire)."""
    global _PLATE
    if _PLATE is None:
        import watchers_a as WA
        pos, yaw, pitch, hfov = WA.plate(CHAIN[6])
        _PLATE = (np.asarray(pos, np.float64), float(yaw), float(pitch), float(hfov))
    return _PLATE


END_CAM = True                                   # A15 reads end_cam() (watchers_a.camera checks this flag)


def _ease(u):
    """0..1 with a gentle start speed (a moving cut) that settles to rest: 1 - (1 - u)^1.6."""
    u = min(max(u, 0.0), 1.0)
    return 1.0 - (1.0 - u) ** 1.6


def _cam_env():
    s = os.environ.get('A14_CAM')
    if not s:
        return None
    return [float(a) for a in s.split(',')]


def camera(f, W=1920, H=804):
    ov = _cam_env()
    if ov is not None:
        x, y, z, yaw, pitch, hfov = ov
        return RC.RCam(np.array([x, y, z]), yaw, pitch, 0.0, hfov, W, H)
    pos1, yaw, pitch1, hfov1 = end_cam()
    fw = np.array([math.sin(math.radians(yaw)), 0.0, math.cos(math.radians(yaw))])
    pos0 = pos1 - fw * BACK0 + np.array([0.0, UP0, 0.0])
    e = _ease((f - FR0) / float(T_SETTLE - FR0))
    # the height comes down a little later than the travel (a glider flares at the end): no dive at the start
    eh = _ease(((f - FR0) / float(T_SETTLE - FR0)) ** 1.15)
    pos = pos0 + (pos1 - pos0) * np.array([e, eh, e])
    pitch = PITCH0 + (pitch1 - PITCH0) * _ease(((f - FR0) / float(T_SETTLE - FR0)) ** 1.3)
    hfov = HFOV0 + (hfov1 - HFOV0) * e
    return RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)


# ------------------------------------------------------------------ the frame ---
_STARS = None


def stars():
    global _STARS
    if _STARS is None:
        _STARS = SK.make_stars(14000, 101, lum_scale=7.0)
    return _STARS


def render(f, scale=1.0, ss=1.5, mblur=True):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = camera(f, W, H)
    fr = PI.Frame(tcam, ss)
    FT = fires_table()
    near = np.hypot(FT[:, 0] - tcam.pos[0], FT[:, 2] - tcam.pos[2]) < LT_MAX
    LT = NA.lt_rows(FT[near], f)
    PL = NA.pl_rows(FT, max_dist=6000.0, cam_pos=tcam.pos)
    Lk, amb, S, fogp, Q = light(f)
    scam = fr.src
    C = scam.params()
    t = f / FPS
    Pp = np.array([scam.pos[0], scam.pos[2], t, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(Pp, CR, C, 0.2, 90000.0, 0.0035, 0.35, 700.0, 9, D)
    fr.img = np.zeros((scam.H, scam.W, 3), np.float32)
    fr.zb = np.zeros((scam.H, scam.W), np.float32)
    fr.dist = np.zeros((scam.H, scam.W), np.float32)
    WD.shade(C, D, Pp, CR, S, LT, Lk, Q, amb, fogp, fr.img, fr.zb, fr.dist, PL)
    # the red under the cloud, throbbing on the race's beat
    WD.cloud_glow(C, D, Pp, CR, NA.ug_rows(f), fogp, fr.img)
    # the cold glow beyond the ranges, breathing once a bar; it puts out the stars near it
    kill = np.zeros(fr.dist.shape, np.float32)
    skl = NA.skyline(scam.pos, CR, WD)
    NA.glow_pass(fr.img, fr.dist, kill, C, NA.glow_gp(f), skl[0], skl[1], skl[2], NA.HAZE_K, NA.HAZE_D)
    pxs = PI.src_scale(fr)
    SK.splat_stars(fr.img, scam, stars(), kill, t=t, gain=ss * ss, scale=pxs)
    # the lighters (RUN-A3's watchers_a): after the stars, BEFORE the fires, so each is a silhouette against its fire
    import watchers_a as WA
    WA.draw_figures(fr.img, fr.zb, scam, f, LT, (Lk, amb, S, fogp, Q), CH=CHAIN)
    NA.fires_layer(fr.img, fr.zb, scam, FT, f, pxs, fogp=fogp, wmod=WD)     # each dimmed by the air in front
    # a near catch throws a short burst of orange sparks that arc and fall (H5): fires 5 and 7
    for k in (4, 6):
        x, y, z, fi, sz, sd = CHAIN[k]
        if fi <= f < fi + 40 and np.hypot(x - scam.pos[0], z - scam.pos[2]) < 400.0:
            NA.ignition_sparks(np.array([x, y, z]), fi, sz, sd).render(fr.img, fr.zb, scam, t, gain=1.0, zbias=0.3)
    img, zb, di = PI.to_target(fr)
    if mblur and _cam_env() is None:
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        r = tcam.ray(u + 0.5, v + 0.5)
        zt = (di * (tcam.f / np.linalg.norm(r, axis=-1))).astype(np.float32)
        zt[di > 1e8] = 1e9
        V = RC.velocity(lambda ff: camera(ff, W, H), f, tcam, di, shutter=SHUTTER)
        img = RC.motion_blur(img, V, zt, max_r=int(40 * scale) + 8)
    return img, tcam, di


FINISH = dict(exposure=1.0, bloom_strength=0.07, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)


def labels(img8, tcam, di):
    """Look-dev overlay: visible catalog summits (index, distance km) and the chain's numbers."""
    import cv2
    H, W = img8.shape[:2]
    s = W / 1920.0
    P = CAT[:, :3] + np.array([0.0, 2.0, 0.0])
    u, v, z = tcam.project(P)
    for k in np.argsort(-z):
        if z[k] < 60 or not (0 <= u[k] < W and 0 <= v[k] < H) or CAT[k, 3] < 40:
            continue
        d = float(np.linalg.norm(P[k] - tcam.pos))
        if di[int(v[k]), int(u[k])] < d - max(40.0, 0.02 * d):
            continue
        cv2.circle(img8, (int(u[k]), int(v[k])), max(2, int(3 * s)), (0, 200, 255), 1)
        cv2.putText(img8, f'{k}:{d / 1000:.1f}', (int(u[k]) + 4, int(v[k]) - 3), 0, 0.33 * max(s, 0.6),
                    (0, 255, 255), 1)
    for n, row in enumerate(CHAIN):
        uu, vv, zz = tcam.project(row[:3] + np.array([0.0, 2.0, 0.0]))
        if zz > 0 and 0 <= uu < W and 0 <= vv < H:
            cv2.putText(img8, str(n + 1), (int(uu) - 5, int(vv) - 10), 0, 0.6 * max(s, 0.6), (80, 80, 255), 2)
    return img8


CUT0 = FR0


def _work(args):
    frames, scale, ss, od, threads = args
    import numba                        # NUMBA_NUM_THREADS is fixed once numba runs (the farm sets it per node)
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    import cv2
    cv2.setNumThreads(1)
    look = PI.look
    for f in frames:
        t0 = time.time()
        hdr, tcam, di = render(f, scale, ss)
        img = look.finish(hdr, **FINISH)
        if os.environ.get('A14_LABELS'):
            im8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
            im8 = labels(im8, tcam, di)
            img = im8[..., ::-1].astype(np.float32) / 255.0
        look.save_png(look.frame_path(od, f), img)
        print(f'A14 frame {f} {time.time() - t0:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    base = os.path.join(PI.CM.ROOT, 'renders', 'beaconrun_A')
    out = base if a.out is None else os.path.join(base, a.out)
    if a.range:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1))
    else:
        frames = [int(x) for x in (a.frames or '4200').split(',')]
    bad = [f for f in frames if not FR0 <= f < FR0 + NFR]
    if bad:
        raise SystemExit(f'frames outside A14 ({FR0}-{FR0 + NFR - 1}): {bad[:5]}')
    os.makedirs(out, exist_ok=True)
    todo = [f for f in frames if not (a.skip and os.path.exists(PI.look.frame_path(out, f)))]
    if a.procs <= 1:
        _work((todo, a.scale, a.ss, out, a.threads))
    else:
        import multiprocessing as mp
        chunks = [todo[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(c, a.scale, a.ss, out, 1) for c in chunks if c])


if __name__ == '__main__':
    main()
