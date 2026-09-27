"""B5 . THE ROAR . THE REVEAL (R2-B) on B's own landform: frames 1360-1519 of cut B (bars 18-19); the edit cuts in
from H1-B's roar at 1372.

A held beat on the roar (her dark shape by the fire as it roars up and settles, a slight drift back), then from the
cello's CALL (bar 18 b3) one long pull back and up in log distance (20 m -> 130 m) that settles on the last wide by
bar 19 b3.5: moonlit silver, her summit low left with the fire a single warm point, the cloud sea to the horizon
with its rock islands, and the Milky Way rising from the horizon right of her summit. No red anywhere but her shawl
and the fire. Nothing answers yet. The sky is the vigil's own (frozen here: this is real time), so the vigil's
first frame (1520) continues it. Per-frame G-buffer (bworld.build with one moon channel).

  python shots/run/reveal.py --frames 1372,1400,1440,1480,1519 --scale 0.25
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
import bfig as BF           # noqa: E402  (RUN-B-3's B figure light: sun-side rim, wool, stones)
import bprops as BP         # noqa: E402  (RUN-B-3's cairn3 / child3, shared with the hand-back)
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
BAND_GAIN = 0.26             # the Milky Way over her summit (director: stronger; it sits by the moon's glow)

HER = BS.STAND
_fd = BS.BEACON - HER
FACE_AZ = math.degrees(math.atan2(_fd[0], _fd[2]))        # she faces the fire
BACK_AZ = FACE_AZ + 180.0
F_HOLD, F_SETTLE = 1398, 1506        # the pull starts on the cello's CALL (bar 18 b3 = 1400), settles by bar 19 b3.5
START = dict(az=BACK_AZ + 30.0, dist=20.0, up=2.5, hfov=44.0, uv=(0.58, 0.60))   # behind her LEFT shoulder: the
# fire stands to her left, as in H1-B's last roar frame (straight behind her, the flame rose out of her head)
END = dict(az=250.0, dist=130.0, up=20.0, hfov=50.0, uv=(0.42, 0.86))   # the crest low: sky + band over it   # at 700 m level the broad summit
# plateau read as a sand dune (so did 240 m); at ~130 m, a little above, it reads as a snowy summit crest (stony boss, the cairn,
# her, the fire) against the silver cloud sea, and the sky above the horizon keeps half the frame for the band.
# WSW of the top looking ENE: the ground beyond drops straight to the cloud sea, the fire sits just right of her,
# and the last wide shares the vigil's axis (its locked frame is the same view from 55 m: the cut is a push-in).


def _smoother(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def pull(frame):
    """0 at the roar .. 1 on the wide: a slight drift through the held beat, then the long pull."""
    drift = 0.04 * min(max((frame - F0) / float(F_HOLD - F0), 0.0), 1.0)
    return drift + (1.0 - drift) * _smoother((frame - F_HOLD) / float(F_SETTLE - F_HOLD))


def camera(frame, W, H):
    s = pull(frame)
    dist = START['dist'] * (END['dist'] / START['dist']) ** s
    daz = (END['az'] - START['az'] + 540.0) % 360.0 - 180.0
    az = START['az'] + daz * s
    up = START['up'] + (END['up'] - START['up']) * s ** 1.3
    piv = BS.BEACON + (BS.TOP - BS.BEACON) * s
    pos = piv + BS.dirxz(az) * dist + np.array([0.0, up, 0.0])
    tgt = (BS.BEACON + np.array([0.0, 0.9, 0.0])) * (1.0 - s) + (BS.TOP + np.array([0.0, 1.0, 0.0])) * s
    hfov = START['hfov'] + (END['hfov'] - START['hfov']) * s
    su = START['uv'][0] + (END['uv'][0] - START['uv'][0]) * s
    sv = START['uv'][1] + (END['uv'][1] - START['uv'][1]) * s
    d = tgt - pos
    bear = math.degrees(math.atan2(d[0], d[2]))
    el = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
    fpx = 0.5 * W / math.tan(math.radians(hfov) * 0.5)
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
        BS.summit_snow(G)                       # old snow on her summit (the frost speckle read as a beach)
        LP, SN, amb, fogp = BS.night_params(self.moon, 1.0 / scam.f, horizon_match=True)
        LP[32], LP[33], LP[34] = 0, 0, 0.0
        # the roar: up on bar 18 b1, still roaring at the cut from H1-B (1372), settling over ~2 s
        roar = smoothstep(F0 - 2, F0 + 4, f) * math.exp(-max(f - F0 - 14, 0) / 22.0)
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
        VG.draw_sky(img, G, scam, sky, f, 1.0, self.ss, th=VG.theta(VG.F0), band_gain=BAND_GAIN)
        # the summit: the cairn, the beacon roaring up, her
        md = self.moon
        lights = [dict(pos=BS.BEACON + np.array([0, 1.3, 0]), col=F.FIRE_LIGHT, I=1.8 * lv * F.flicker(t, 3), r0=0.5),
                  dict(dir=md, col=lin('#A7BCE0'), I=0.5)]
        amb_f = lin('#27335E') * 0.45
        BF.render(img, zb, scam, BP.cairn3(), BS.CAIRN, lights, amb=amb_f * 2.6, t=t, write_depth=True,
                  zbias=0.3)
        back, front, fb = BS.beacon_base()
        BF.render(img, zb, scam, back, BS.BEACON, lights, amb=amb_f, t=t, emissive_gain=1.0,
                  write_depth=True, zbias=0.3)
        base = BS.BEACON + np.array([0.0, fb, 0.0])
        F2.flame(img, zb, scam, base, 1.0 + 0.9 * roar, 0.44, t, seed=4, I=14.0 * lv, lean=0.15, zbias=0.5, tongues=6,
                 warp=1.3)
        bx, by, bz = scam.project(base + np.array([0, 0.6, 0]))
        F2.halo(img, zb, bx, by, 5.0 * scam.f / bz, 0.006 * lv, z=bz, zbias=3.0)
        # far away the fire must still read as a warm POINT (director): a distance-sized core and a small halo
        far = smoothstep(40.0, 260.0, bz)
        if far > 0.0:
            F2.glow(img, zb, bx, by, 1.1 * self.ss, 26.0 * far * lv * self.ss * self.ss, z=bz, zbias=bz * 0.02,
                    col=np.array([1.0, 0.62, 0.26]))
            F2.halo(img, zb, bx, by, 5.0 * self.ss, 0.05 * far * lv, z=bz, zbias=bz * 0.02,
                    col=np.array([1.0, 0.50, 0.18]))
        BF.render(img, zb, scam, front, BS.BEACON, lights, amb=amb_f, t=t, write_depth=False, zbias=0.3)
        # her: weight back from the roar, then standing, looking at her fire
        d, _ = BS.person2('look', age=0.9, shawl=True, staff=True, wind=0.5)
        if f < F0 + 30:
            d.rotate(-0.10 * math.exp(-((f - F0 - 6) / 12.0) ** 2), pivot=(0.0, 0.0))
        BF.render(img, zb, scam, d, HER, lights, amb=amb_f, t=t, write_depth=False, zbias=0.3)
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
