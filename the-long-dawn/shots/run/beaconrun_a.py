"""A14 · THE BEACON RUN (R3, cut A): A cut frames 3920-4239 (bars 50-53) -> renders/beaconrun_A/.

The cut follows the desert figure's look into the depth of the frame. We come in over A's knife-edge ranges on H1's
moonlit night, heading toward the cold glow (az -30), which still breathes once a bar beyond the far ranges. Under
the cloud sea the red patches of the race throb on every beat. The fires that caught with hers (A13) are already
burning on far ridges. Now the watch links across the ranges ahead of the camera, every two beats, seven times,
from bar 50 b3 to bar 53 b3. Near and far alternate, so the far answer is heard on the second fire, bar 51 b1.
The camera is a slow, high glider on a long lens. It never overtakes a fire, every fire stays ahead of it, and
there is always one in frame. It descends and settles behind the seventh fire, the nearest, which catches on bar
53 b3 beside the one who lit it. A15 R16 THE WATCHERS (RUN-A3) holds from this last position: it imports
END_CAM, FIRES, CR and light() from here.

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
# (x, y, z, f_ign, size, seed): nighta's fire rows. Filled by the look-dev pass; see NOTES (RUN-A-L) for the picks.
CHAIN = np.zeros((0, 6))
LIT = np.zeros((0, 6))                           # fires that caught with hers (A13), already burning here


def fires_table():
    return np.vstack([LIT, CHAIN]) if len(LIT) else CHAIN


FIRES = CHAIN                                    # A15's name for the seven (ignition order)

# ------------------------------------------------------------------ the camera ---
# keys: (cut frame, x, y, z, yaw, pitch, hfov); PCHIP through the keys, eased to rest at the end
CAM_KEYS = [(3920, 60.0, 168.0, 480.0, -30.0, -6.0, 40.0),
            (4239, 60.0, 168.0, 480.0, -30.0, -6.0, 40.0)]
_PCH = None


def _cam_env():
    s = os.environ.get('A14_CAM')
    if not s:
        return None
    v = [float(a) for a in s.split(',')]
    return v


def camera(f, W=1920, H=804):
    ov = _cam_env()
    if ov is not None:
        x, y, z, yaw, pitch, hfov = ov
        return RC.RCam(np.array([x, y, z]), yaw, pitch, 0.0, hfov, W, H)
    global _PCH
    if _PCH is None:
        from scipy.interpolate import PchipInterpolator
        K = np.array(CAM_KEYS, np.float64)
        _PCH = [PchipInterpolator(K[:, 0], K[:, j]) for j in range(1, 7)]
    ff = min(max(float(f), CAM_KEYS[0][0]), CAM_KEYS[-1][0])
    x, y, z, yaw, pitch, hfov = [float(p(ff)) for p in _PCH]
    return RC.RCam(np.array([x, y, z]), yaw, pitch, 0.0, hfov, W, H)


END_CAM = None                                   # (pos, yaw, pitch, hfov) at cut 4239, set with the keys


def end_cam():
    c = camera(FR0 + NFR - 1)
    return c.pos.copy(), c.yaw_d, c.pitch_d, c.hfov_d


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
    LT = NA.lt_rows(FT, f)
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
    NA.fires_layer(fr.img, fr.zb, scam, FT, f, pxs, fogp=fogp, wmod=WD)     # each dimmed by the air in front
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
