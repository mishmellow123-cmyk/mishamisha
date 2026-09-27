"""B5 . THE ROAR . THE REVEAL (R2-B) on B's own landform: frames 1360-1519 of cut B (bars 18-19).

On the roar (bar 18 b1, the cut out of H1-B's hands and tinder) we are close behind her, a dark hooded shape against
the fire as it roars up; the camera pulls back and rises (fast, then settling) until she is a few pixels beside a
single warm point on her rounded summit, above a moonlit silver cloud sea, the far ranges in layers and the Milky Way
over them. No red anywhere but her shawl and the fire. Nothing answers yet.
Per-frame G-buffer (the camera moves): bworld.build with one moon channel.

  python shots/run/reveal.py --frames 1360,1400,1460,1519 --scale 0.25
  python shots/run/reveal.py --range 1360-1519 --scale 1.0 --ss 1.5 --out renders/reveal_B --procs 4 --skip
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bworld as BW         # noqa: E402
import bset as BS           # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import fire2 as F2          # noqa: E402
import keeper as KP         # noqa: E402  (generic star trails)
import vigil as VG          # noqa: E402  (the night's moon, sky wheel, band)
from mt import fire as F, figure as FG   # noqa: E402
from mt.noise import smoothstep   # noqa: E402

CM = PI.CM
look = PI.look
lin = CM.lin
CR = BW.CR_B
FPS = 24.0
TESTS = os.path.join(CM.ROOT, 'renders', 'run_b_tests')
F0, F1 = 1360, 1520

HER = BS.STAND
_fd = BS.BEACON - HER
FACE_AZ = math.degrees(math.atan2(_fd[0], _fd[2]))        # she faces the fire
BACK_AZ = FACE_AZ + 180.0


def _ease_out(u):
    u = min(max(u, 0.0), 1.0)
    return 1.0 - (1.0 - u) ** 3


def camera(frame, W, H):
    u = _ease_out((frame - F0) / float(F1 - 1 - F0))
    # start: 7 m behind her at eye height; end: 300 m out (WNW of the top), 40 m above it
    p0 = HER + BS.dirxz(BACK_AZ) * 7.0 + np.array([0.0, 1.5, 0.0])
    p1 = BS.TOP + BS.dirxz(BACK_AZ - 8.0) * 300.0 + np.array([0.0, 40.0, 0.0])
    # a curved path: out first, then up
    pos = p0 + (p1 - p0) * np.array([u, u ** 1.6, u])
    tgt0 = BS.BEACON + np.array([0.0, 0.9, 0.0])
    tgt1 = BS.TOP + np.array([0.0, 1.0, 0.0])
    tgt = tgt0 + (tgt1 - tgt0) * u
    hfov = 46.0 + 6.0 * u
    d = tgt - pos
    bear = math.degrees(math.atan2(d[0], d[2]))
    el = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
    fpx = 0.5 * W / math.tan(math.radians(hfov) * 0.5)
    # the target sits a little right of centre and low (the sky and the band get the room)
    su, sv = 0.5 + 0.10 * u, 0.55 + 0.18 * u
    yaw = bear - math.degrees(math.atan((su - 0.5) * W / fpx))
    pitch = el + math.degrees(math.atan((sv - 0.5) * H / fpx))
    return RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)


class Reveal:
    def __init__(self, scale=0.25, ss=1.5):
        self.scale, self.ss = scale, ss
        self.W, self.H = int(round(1920 * scale)), int(round(804 * scale))
        az, el = VG.moon_at(VG.F0)
        self.moon = BS.moon_vec(el, az)

    def render(self, f):
        t = f / FPS
        tc = camera(f, self.W, self.H)
        fr = PI.Frame(tc, self.ss)
        scam = fr.src
        P = np.array([scam.pos[0], scam.pos[2], t, 0.0])
        G = BW.build(scam, P, CR, None, dmax=180000.0, moons=[self.moon], mk=10.0)
        LP, SN, amb, fogp = BS.night_params(self.moon, 1.0 / scam.f)
        LP[32], LP[33], LP[34] = 0, 0, 0.0
        roar = math.exp(-((f - F0 - 4) / 14.0) ** 2)
        lv = 1.05 + 0.9 * roar
        LP[36] = 9.0 * lv * F.flicker(t, 3)
        LP[37:40] = BS.BEACON + np.array([0.0, 1.3, 0.0])
        LP[40:43] = np.array(F.FIRE_LIGHT)
        LP[43] = 40.0
        img = np.zeros((scam.H, scam.W, 3), np.float32)
        BW.shade(G, LP, SN, amb, fogp, float(scam.pos[1]), img)
        dist = G[..., BW.G_DIST].astype(np.float32)
        zb = dist.copy()
        sky = (dist > 1e8).astype(np.float32)
        ang = VG.sky_angle(VG.F0) - math.radians(VG.STAR_DEG / (VG.F1 - VG.F0)) * (VG.F0 - f)
        BW.add_band(G, KP._rotmat(KP.POLE, ang), VG.BAND, 0.05, 1.0, img)
        KP.draw_stars(img, scam, sky, ang - 0.0008, ang, 1.6 * self.ss * self.ss, K=3)
        # the summit: the cairn, the beacon roaring up, her
        md = self.moon
        lights = [dict(pos=BS.BEACON + np.array([0, 1.3, 0]), col=F.FIRE_LIGHT, I=1.8 * lv * F.flicker(t, 3), r0=0.5),
                  dict(dir=md, col=lin('#A7BCE0'), I=0.5)]
        amb_f = lin('#27335E') * 0.45
        FG.render(img, zb, scam, BS.rubble_cairn(), BS.CAIRN, lights, amb=amb_f, mats=BS.M, t=t, write_depth=True,
                  zbias=0.3)
        back, front, fb = BS.beacon_base()
        FG.render(img, zb, scam, back, BS.BEACON, lights, amb=amb_f, mats=BS.M, t=t, emissive_gain=1.0,
                  write_depth=True, zbias=0.3)
        base = BS.BEACON + np.array([0.0, fb, 0.0])
        F2.flame(img, zb, scam, base, 1.0 + 0.9 * roar, 0.44, t, seed=4, I=14.0 * lv, lean=0.15, zbias=0.5, tongues=6,
                 warp=1.3)
        bx, by, bz = scam.project(base + np.array([0, 0.6, 0]))
        F2.halo(img, zb, bx, by, 5.0 * scam.f / bz, 0.006 * lv, z=bz, zbias=3.0)
        FG.render(img, zb, scam, front, BS.BEACON, lights, amb=amb_f, mats=BS.M, t=t, write_depth=False, zbias=0.3)
        # her: weight back from the roar, then standing, looking at her fire
        d, _ = BS.person('look', age=0.85, shawl=True, staff=True, wind=0.5)
        if f < F0 + 30:
            d.rotate(-0.10 * math.exp(-((f - F0 - 6) / 12.0) ** 2), pivot=(0.0, 0.0))
        FG.render(img, zb, scam, d, HER, lights, amb=amb_f, mats=BS.M, t=t, write_depth=False, zbias=0.3)
        fr.img, fr.zb, fr.dist = img, zb, dist
        out, _, _ = PI.to_target(fr)
        return out


FINISH = dict(exposure=1.15, bloom_strength=0.06, bloom_threshold=1.2, streak_strength=0.0, vignette_amount=0.22)


def _work(args):
    frames, scale, ss, out = args
    shot = Reveal(scale, ss)
    for f in frames:
        t0 = time.time()
        img = look.finish(shot.render(f), **FINISH)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.2f}s', flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--range', default=None)
    ap.add_argument('--frames', default=None)
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default='rev_t')
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    out = os.path.join(CM.ROOT, a.out) if a.out.startswith('renders/') else os.path.join(TESTS, a.out)
    os.makedirs(out, exist_ok=True)
    if a.range:
        s0, s1 = a.range.split('-')
        frames = list(range(int(s0), int(s1) + 1))
    else:
        frames = [int(x) for x in a.frames.split(',')]
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.find_frame(out, f))]   # find_frame never returns None
    if a.procs <= 1:
        _work((frames, a.scale, a.ss, out))
    else:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(frames[i::a.procs], a.scale, a.ss, out) for i in range(a.procs)])


if __name__ == '__main__':
    main()
