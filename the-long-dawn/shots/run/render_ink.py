"""C · THE LIVING INK (R12): renders the Run world's AOVs and separated fire / prop layers for the ink pass
(`inkpass.py`). Additive: world.py, render_run.py, beacons.py and dawn.py are imported, never changed.

Shots (C only; REVISION 1: three ink shots):
  reveal   C17  her fire, small on the vast range: a slow pull-back from behind her summit (local frames 0-239)
  scroll   C18  the lateral run: the camera trucks sideways at beacon height like a scroll unrolling; seven
                beacons catch as they come into view (local frames 0-287)
  illum    R15  the illumination: dawn_C's camera and sun (v2 frames 2400-2655), ink filled with colour where
                the sun touches

  python render_ink.py --shot scroll --frames 0,140 --scale 0.25            # AOV caches -> renders/run_c_tests/aov
  python render_ink.py --shot scroll --range 130-153 --scale 0.5 --threads 4
Caches are float16/32 npz (see save_cache); `inkpass.py` turns them into frames.
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

import pipe as PI           # noqa: E402
import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import run as RN            # noqa: E402
import beacons as BC        # noqa: E402
import ink_aov as IA        # noqa: E402
from mt import fire as F    # noqa: E402
import fire2 as F2          # noqa: E402

import cv2                  # noqa: E402

FPS = 24.0
ROOT = PI.CM.ROOT
OUT = os.path.join(ROOT, 'renders', 'run_c_tests')
HER = np.array([-5449.7, 282.7, 31997.3])          # her beacon (beacons.npy row 0 = S1.summit() - 3 m)

# ------------------------------------------------------------------ cameras ---


def _smoother(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


# C17 THE REVEAL: behind her summit, looking out over the range toward the shepherd (bearing ~170 deg);
# a slow pull-back and rise from over her shoulder to a wide where her fire is small (240 frames).
REV_YAW = 170.0
REV_N = 240
REV_KEYS = dict(back=(70.0, 430.0), up=(16.0, 66.0), pitch=(-7.5, -3.2), hfov=(44.0, 46.0))


_REV = {}


def reveal_set():
    """Her summit as RUN-B dresses it (keeper.py: the snow shelf over the rock fin, the NE arete), so the shared
    reveal (R2) shows one summit in every cut; her fire sits on the shelf where B's beacon stands."""
    if not _REV:
        import keeper as KP
        _REV['CR'] = np.ascontiguousarray(KP.CR_B)
        _REV['fire'] = np.array(KP.BEACON, np.float64)
    return _REV


def reveal_camera(frame, W=1920, H=804):
    fire = reveal_set()['fire']
    u = _smoother(frame / (REV_N - 1))
    a = math.radians(REV_YAW)
    fwd = np.array([math.sin(a), 0.0, math.cos(a)])
    back = REV_KEYS['back'][0] + (REV_KEYS['back'][1] - REV_KEYS['back'][0]) * u
    up = REV_KEYS['up'][0] + (REV_KEYS['up'][1] - REV_KEYS['up'][0]) * u
    pos = fire + np.array([0.0, 3.0 + up, 0.0]) - fwd * back
    pitch = REV_KEYS['pitch'][0] + (REV_KEYS['pitch'][1] - REV_KEYS['pitch'][0]) * u
    hfov = REV_KEYS['hfov'][0] + (REV_KEYS['hfov'][1] - REV_KEYS['hfov'][0]) * u
    return RC.RCam(pos, REV_YAW, pitch, 0.0, hfov, W, H)


# C18 THE SCROLL: a lateral truck at beacon height. The camera looks along SC_YAW and travels to its right;
# constant speed with short eases, no roll, no bank: a scroll unrolling, not a glider.
SC_YAW = 22.0
SC_N = 320                         # bars 49-52 of cut C (C17), 13.3 s
SC_CATCH = [40, 80, 120, 160, 200, 240, 280]      # bar 49 b3 .. bar 52 b3: one beacon every two beats
_a = math.radians(SC_YAW)
_FWD = np.array([math.sin(_a), 0.0, math.cos(_a)])
_RIGHT = np.array([math.cos(_a), 0.0, -math.sin(_a)])
SC_C0 = np.array([-16789.0, 0.0, 12762.0])        # the line past the western chain (summits 1.3-5 km off)
SC_P0 = SC_C0 - _FWD * 400.0 + _RIGHT * (-900.0) + np.array([0.0, 110.0, 0.0])
SC_LEN = 3400.0                    # metres travelled over the shot (~255 m/s: a peak 3 km off takes ~9 s to cross)
SC_PITCH = -2.5
SC_HFOV = 44.0


def scroll_s(frame):
    """Distance along the truck (m): eased in over 1 s, constant, eased out over 1 s."""
    T = SC_N - 1
    e = 24.0
    v = 1.0 / (T - e)                  # normalised speed so s(T) = 1
    f = min(max(frame, 0.0), T)
    if f < e:
        s = 0.5 * v * f * f / e
    elif f > T - e:
        g = T - f
        s = 1.0 - 0.5 * v * g * g / e
    else:
        s = 0.5 * v * e + v * (f - e)
    return SC_LEN * s


def scroll_camera(frame, W=1920, H=804):
    a = math.radians(SC_YAW)
    right = np.array([math.cos(a), 0.0, -math.sin(a)])
    pos = SC_P0 + right * scroll_s(frame)
    return RC.RCam(pos, SC_YAW, SC_PITCH, 0.0, SC_HFOV, W, H)


_SCB = None


def scroll_beacons():
    """The seven beacons of C17, one per locked catch frame (SC_CATCH): at each catch the chosen summit is on the
    unrolling side of the frame (0.55-0.88 W), 1.3-5.5 km off, standing clear of the cloud and visible, and no
    summit is used twice; depths alternate where the catalogue allows. Her own beacon burns on the horizon from
    the start if it is in view (where the call came from). Each fire is seated on the true apex.
    Rows as beacons.npy: x y z ign kind idx dist."""
    global _SCB
    if _SCB is not None:
        return _SCB
    S = np.load(os.path.join(HERE, 'summits.npy'))
    S = S[(S[:, 3] >= 180.0) & (S[:, 1] >= WD.CLOUD_Y + 300.0)]
    rows = []
    c0 = scroll_camera(0)
    hx, hy, hz = c0.project(HER + np.array([0.0, 1.6, 0.0]))
    if hz > 0 and 0.03 < hx / 1920.0 < 0.97 and BC._visible(c0, HER + np.array([0, 3.0, 0]), RN.CR, 10.0):
        rows.append([HER[0], HER[1], HER[2], -400.0, 1, 0, float(np.linalg.norm(HER - c0.pos))])
    used = []
    last_d = None
    for k, f in enumerate(SC_CATCH):
        c = scroll_camera(f)
        placed = False
        for (ulo, uhi, dlo, dhi) in ((0.55, 0.88, 1300.0, 5500.0), (0.45, 0.94, 1100.0, 9000.0)):
            if placed:
                break
            cands = []
            for q in range(len(S)):
                if q in used:
                    continue
                x, y, z, prom = S[q]
                P = np.array([x, y + 1.0, z])
                sx, sy, zc = c.project(P)
                if zc <= 0:
                    continue
                u = sx / 1920.0
                if u < ulo or u > uhi or sy < 0.16 * 804 or sy > 0.78 * 804:
                    continue
                d = float(np.linalg.norm(P - c.pos))
                if d < dlo or d > dhi:
                    continue
                score = -3.0 * abs(u - 0.70) + min(prom, 900.0) / 900.0 - 0.4 * abs(math.log(d / 3000.0))
                if last_d is not None:
                    score += 0.35 * min(abs(math.log(d / last_d)), 0.7)        # alternate near and far
                cands.append((score, q, P, d))
            cands.sort(key=lambda r: -r[0])
            for score, q, P, d in cands[:12]:
                if BC._visible(c, P + np.array([0, 2.5, 0]), RN.CR, 10.0):
                    sm = WD.summit_near(P[0], P[2], RN.CR, rad=45.0, n=15)
                    rows.append([sm[0], sm[1], sm[2], float(f), 1, len(rows), d])
                    used.append(q)
                    last_d = d
                    placed = True
                    break
    _SCB = np.array(rows)
    return _SCB


# ------------------------------------------------------------------ shots ---

class Shot:
    def __init__(self, name):
        self.name = name
        self.CR = RN.CR
        if name == 'reveal':
            self.cam = reveal_camera
            rs = reveal_set()
            self.CR = rs['CR']
            f = rs['fire']
            self.B = np.array([[f[0], f[1], f[2], -400.0, 1, 0, 0.0]])
        elif name == 'scroll':
            self.cam = scroll_camera
            self.B = scroll_beacons()
        elif name == 'illum':
            import dawn as DN
            self.DN = DN
            self.cam = DN.camera
            self.B = np.zeros((0, 7))
        else:
            raise ValueError(name)

    def light(self, frame):
        if self.name == 'illum':
            DN = self.DN
            Lk = np.r_[DN.light_dir(frame), DN.sun_col(frame)]
            Lk_, amb, S, fogp, Q = WD.night_light()
            Q = Q.copy()
            Q[0] = 2.1 * DN.light_gain(frame)
            Q[1] = 2.2
            Q[2] = 8.0
            fogc = PI.CM.lin('#7288BE') * 0.33
            fogp = np.array([5.0e-5, 1 / 1500.0, 2.0e-4, 1 / 150.0, 1.2, fogc[0], fogc[1], fogc[2]])
            return Lk, Q, fogp
        Lk, amb, S, fogp, Q = WD.night_light()
        if self.name == 'reveal':
            # the illustrator's key from the page's left (the moon is behind this camera, which would leave every
            # face front-lit and the drawing without form); C only, and invisible as 'moonlight' in ink
            a = math.radians(REV_YAW - 80.0)
            e = math.radians(28.0)
            Lk = Lk.copy()
            Lk[0:3] = [math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)]
        return Lk, Q, fogp

    def lights_steady(self, frame):
        """Fire point lights WITHOUT flicker (the ink's tone must not flicker)."""
        LT = []
        for b in self.B:
            size, inten, light = BC.env(frame, b[3])
            if light <= 0:
                continue
            LT.append([b[0], b[1] + 2.2, b[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2], 60.0 * light, 0.8])
        return np.array(LT, np.float64).reshape(-1, 8)


def fire_layer(img, zb, scam, frame, B, pxs):
    BC.draw_source(img, zb, scam, frame, B, pxs)


def render_aov(shot, frame, scale=0.5, ss=2.0):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = shot.cam(frame, W, H)
    fr = PI.Frame(tcam, ss)
    scam = fr.src
    C = scam.params()
    P = np.array([scam.pos[0], scam.pos[2], frame / FPS, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(P, shot.CR, C, 0.2, 90000.0, 0.0035, 0.35, 900.0, 9, D)
    Lk, Q, fogp = shot.light(frame)
    LT = shot.lights_steady(frame)
    PL = BC.platforms(shot.B) if len(shot.B) else np.zeros((0, 4))
    A = np.zeros((scam.H, scam.W, IA.NA), np.float32)
    IA.shade_aov(C, D, P, shot.CR, LT, Lk, Q, fogp, PL, A)
    pxs = PI.src_scale(fr)
    # fire layer on black and on grey: emission (premultiplied) and coverage (smoke, soot)
    zb0 = np.ascontiguousarray(A[..., 15])
    g = 0.5
    Eb = np.zeros((scam.H, scam.W, 3), np.float32)
    Eg = np.full((scam.H, scam.W, 3), g, np.float32)
    # (H5) the fires and the dawn's smoke are DRAWN by the ink pass (inkpass.ink_flames / ink_plumes); these layers
    # stay empty
    alpha = np.clip(1.0 - (Eg - Eb).mean(axis=2) / g, 0.0, 1.0)
    # warp to the (supersampled) target camera; AOVs nearest, layers linear
    mx, my = RC.warp_maps(fr.t_ss, scam)
    At = np.concatenate([RC.warp(np.ascontiguousarray(A[..., c:c + 4]), mx, my, cv2.INTER_NEAREST)
                         for c in range(0, IA.NA, 4)], axis=2)
    Et = RC.warp(Eb, mx, my, cv2.INTER_LINEAR)
    at = RC.warp(alpha.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    tss = fr.t_ss
    cam = dict(pos=tss.pos, fwd=tss.fwd, right=tss.right, up=tss.up, f=tss.f, cx=tss.cx, cy=tss.cy, W=tss.W,
               H=tss.H, hfov=tss.hfov_d, yaw=tss.yaw_d, pitch=tss.pitch_d, roll=tss.roll_d, t=frame / FPS)
    # beacon screen positions and envelopes (for the gold leaf)
    bl = []
    for b in shot.B:
        size, inten, light = BC.env(frame, b[3])
        base = np.array([b[0], b[1] + 1.6, b[2]])
        sx, sy, z = tss.project(base)
        bl.append([sx, sy, z, b[3], size, inten, light, b[5], b[4]])
    # camera velocity for an optional ink motion blur (target, supersampled)
    V = RC.velocity(lambda f: shot.cam(f, tss.W, tss.H), frame, tss, At[..., 1], shutter=0.35)
    return dict(A=At, E=Et, a=at, cam=cam, bl=np.array(bl, np.float64).reshape(-1, 9), V=V, Lk=Lk,
                frame=frame, shot=shot.name, scale=scale, ss=ss)


def cache_path(shot, frame, scale):
    return os.path.join(OUT, 'aov', f'{shot}_{int(round(scale * 100)):03d}_{frame:05d}.npz')


def save_cache(p, R):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    A = R['A']
    f32 = [1, 2, 4, 5]                     # dist, x, z, h stay float32 (world anchoring)
    keep16 = [c for c in range(A.shape[2]) if c not in f32]
    tmp = p + '.tmp.npz'
    np.savez_compressed(tmp, A32=A[..., f32].astype(np.float32), A16=A[..., keep16].astype(np.float16),
                        E=R['E'].astype(np.float16), a=R['a'].astype(np.float16), V=R['V'].astype(np.float16),
                        bl=R['bl'], Lk=R['Lk'], cam=np.array([R['cam']], dtype=object),
                        meta=np.array([R['frame'], R['scale'], R['ss']]), shot=np.array([R['shot']]))
    os.replace(tmp, p)


def load_cache(p):
    Z = np.load(p, allow_pickle=True)
    A32, A16 = Z['A32'], Z['A16']
    H, W = A32.shape[:2]
    A = np.zeros((H, W, IA.NA), np.float32)
    f32 = [1, 2, 4, 5]
    keep16 = [c for c in range(IA.NA) if c not in f32]
    A[..., f32] = A32
    A[..., keep16] = A16.astype(np.float32)
    return dict(A=A, E=Z['E'].astype(np.float32), a=Z['a'].astype(np.float32), V=Z['V'].astype(np.float32),
                bl=Z['bl'], Lk=Z['Lk'], cam=Z['cam'][0], frame=int(Z['meta'][0]), scale=float(Z['meta'][1]),
                ss=float(Z['meta'][2]), shot=str(Z['shot'][0]))


def work(args):
    shot_name, frames, scale, ss, threads, skip = args
    os.environ['NUMBA_NUM_THREADS'] = str(threads)
    import numba
    numba.set_num_threads(threads)
    cv2.setNumThreads(1)
    shot = Shot(shot_name)
    for f in frames:
        p = cache_path(shot_name, f, scale)
        if skip and os.path.exists(p):
            continue
        t0 = time.time()
        R = render_aov(shot, f, scale, ss)
        save_cache(p, R)
        print(f'{shot_name} {f} {time.time() - t0:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shot', default='scroll')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--ss', type=float, default=2.0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--beacons', action='store_true', help='print the scroll beacon table and exit')
    a = ap.parse_args()
    if a.beacons:
        np.set_printoptions(suppress=True, precision=1, linewidth=150)
        print(scroll_beacons())
        return
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1))
    work((a.shot, frames, a.scale, a.ss, a.threads, a.skip))


if __name__ == '__main__':
    main()
