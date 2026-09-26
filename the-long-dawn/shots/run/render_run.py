"""THE BEACON RUN frame renderer + driver.

  python render_run.py --frames 1520,1580 --scale 0.4 --out tests/t1     # test stills
  python render_run.py --range 1520-1679 --procs 3                       # final -> renders/run_v2/
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
from mt import sky as SK, fire as F, props as PR, figure as FG   # noqa: E402
import fire2 as F2          # noqa: E402

look = PI.look
FPS = 24.0
PYRE_IGN = 1580
SHEP_FIRE = np.array([0.35, 1.12 + 0.7, 16.6])        # s1's cairn fire (lit at 1480), behind the camera at 1520
SHUTTER = 0.35                                          # frames (126 degrees)
FINISH = dict(exposure=1.0, bloom_strength=0.07, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)

_C = {}


def _cached(key, fn):
    if key not in _C:
        _C[key] = fn()
    return _C[key]


def _stars():
    return _cached('stars', lambda: SK.make_stars(14000, 101, lum_scale=7.0))


def _pyre_props():
    return _cached('pyre', lambda: PR.cairn(seed=9, height=1.45, base_w=2.0, top_w=1.35, basket=True, basket_h=0.62,
                                            basket_w=1.7))


def _pyre_base():
    back, front, fb, rb = _pyre_props()
    feet = RN.PYRE_XZ + np.array([0.0, -0.2, 0.0])
    return feet, feet + np.array([0.0, fb, 0.0]), rb


def _pyre_tongues():
    """Six tongues, well separated (gaps between them), two tall ones off-centre."""
    def mk():
        TG = F.tongue_table(6, 23, spread=0.8)
        TG[:, 2] *= 0.78                  # narrower tongues -> dark gaps between them
        TG[[1, 4], 1] *= 1.18             # two tall ones
        return TG
    return _cached('ptg', mk)


def _wind():
    """Summit gale at the pyre: blowing across the flight line, from the tower toward the camera's track."""
    h = math.radians(RN.heading(RN.PYRE_PASS))
    right = np.array([math.cos(h), 0.0, -math.sin(h)])
    return right * 6.0 + np.array([0.0, 0.0, 0.0])


def _pyre_sparks():
    def mk():
        _, base, rb = _pyre_base()
        sp = F.Sparks(77, base + np.array([0, 0.4, 0]), PYRE_IGN / FPS, 1690 / FPS, burst=180, rate=150,
                        ember_rate=36, speed=(3.0, 8.0), burst_speed=(4.0, 9.0), spread=0.22, radius=0.8,
                        wind=tuple(_wind()), I=13.0, buoy=6.5, updraft_h=4.0)
        sp.T0 *= 0.80                     # born orange-yellow, not white
        return sp
    return _cached('psparks', mk)


def _pyre_smoke():
    def mk():
        _, base, rb = _pyre_base()
        w = _wind()
        return F.Smoke(78, base + np.array([0, 3.2, 0]), PYRE_IGN / FPS + 0.15, 1690 / FPS, rate=6.0,
                       wind=(w[0], 0.0, w[2]), rise=2.0, life=5.0, r0=0.6, growth=0.9, dens=0.5)
    return _cached('psmoke', mk)


def pyre_light(frame):
    size, inten, light = BC.env(frame, PYRE_IGN)
    if light <= 0:
        return None
    _, base, _ = _pyre_base()
    fl = F.flicker(frame / FPS, 5)
    return [base[0], base[1] + 2.6, base[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2], 48.0 * light * fl, 2.6]


def draw_pyre_source(img, zb, scam, frame, pxs, moon_dir):
    t = frame / FPS
    size, inten, light = BC.env(frame, PYRE_IGN)
    back, front, fb, rb = _pyre_props()
    feet, base, _ = _pyre_base()
    sx, sy, z = scam.project(feet)
    if z <= 2.0 or sx < -800 or sx > img.shape[1] + 800:
        return
    fl = F.flicker(t, 5)
    fire_I = 70.0 * light * fl
    lights = [dict(dir=moon_dir, col=PI.CM.lin('#9DB4D9'), I=0.55 * 0.7)]
    if fire_I > 0:
        lights.append(dict(pos=base + np.array([0, 0.6, 0]), col=F.FIRE_LIGHT, I=fire_I * 0.25, r0=0.4))
    amb = PI.CM.lin('#27335E') * 0.35 * 1.2
    FG.render(img, zb, scam, back, feet, lights, amb=amb, t=t, emissive_gain=min(light, 1.5) * 1.3, write_depth=True,
              zbias=0.3)
    if size > 0:
        _pyre_smoke().render(img, zb, scam, t, base + np.array([0, 1.2, 0]), 1.6 * light * fl, albedo=0.24,
                             amb=(0.004, 0.005, 0.009), zbias=0.6)
        F2.flame(img, zb, scam, base + np.array([0.0, -0.15, 0.0]), 4.8 * size, 1.15, t, seed=21, I=13.0 * inten,
                 lean=0.55 * size, zbias=0.6, TG=_pyre_tongues(), warp=1.4)
        ppm = scam.f / z
        F2.halo(img, zb, sx, sy - 2.0 * ppm * size, 6.0 * ppm, 0.028 * light * fl, z=z, zbias=2.0)
    FG.render(img, zb, scam, front, feet, lights, amb=amb, t=t, write_depth=False, zbias=0.3)


def draw_sparks_target(img, zt, sp, frame, W, H, gain=1.0, K=5, zbias=0.4):
    """Sparks streaked through the MOVING camera: sample s of the shutter is projected with the camera at
    that instant, so fast camera moves smear sparks correctly."""
    t = frame / FPS
    shutter = SHUTTER / FPS
    pts, alive, age = sp.state(t, shutter, K)
    if not alive.any():
        return
    idx = np.nonzero(alive)[0]
    P = pts[idx]
    sx = np.zeros((len(idx), K))
    sy = np.zeros((len(idx), K))
    zc = np.zeros((len(idx), K))
    for s in range(K):
        fs = frame + (-0.5 + s / (K - 1)) * SHUTTER
        c = RN.camera(fs, W, H)
        a, b, z = c.project(P[:, s])
        sx[:, s], sy[:, s], zc[:, s] = a, b, z
    ok = (zc > 0.3).all(axis=1)
    idx, sx, sy, zc = idx[ok], sx[ok], sy[ok], zc[ok]
    if len(idx) == 0:
        return
    a = np.clip(age[idx] / sp.life[idx], 0, 1)
    T = sp.T0[idx] * (1.0 - 0.72 * a ** 0.9)
    tw = 0.7 + 0.3 * np.sin(sp.tw[idx] * (t - sp.ts[idx]) + sp.ph[idx, 0])
    fade_in = np.clip((t - sp.ts[idx]) / 0.04, 0, 1)
    fade_out = np.clip((1.0 - a) / 0.2, 0, 1)
    inten = sp.I * gain * sp.lum[idx] * (T / 0.9) ** 4.0 * tw * fade_in * fade_out
    s_ = W / 1920.0
    rad = np.maximum(sp.wid[idx] * s_ * np.clip(30.0 / zc.mean(axis=1), 1.0, 3.0), 0.3)
    col = F2.ramp_vec(T)
    F._draw_streaks(img, zt, sx.astype(np.float64), sy.astype(np.float64), zc.mean(axis=1).astype(np.float64),
                    rad.astype(np.float64), col.astype(np.float64), inten.astype(np.float64), float(zbias),
                    float(900.0 * s_))


def render(frame, scale=1.0, ss=1.5, mblur=True):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = RN.camera(frame, W, H)
    fr = PI.Frame(tcam, ss)
    B = BC.place()
    extra = [[SHEP_FIRE[0], SHEP_FIRE[1], SHEP_FIRE[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2],
              7.5 * F.flicker(frame / FPS, 3), 0.5]]
    pl = pyre_light(frame)
    if pl is not None:
        extra.append(pl)
    LT = BC.lights(frame, B, extra)
    light = WD.night_light()
    PL = BC.platforms(B, extra=[[RN.PYRE_XZ[0], RN.PYRE_XZ[1], RN.PYRE_XZ[2], 3.4]])
    PI.render_terrain(fr, frame, RN.CR, light, LT, PL=PL)
    scam = fr.src
    pxs = PI.src_scale(fr)
    mask = (fr.dist > 1e8).astype(np.float32)
    SK.splat_stars(fr.img, scam, _stars(), mask, t=frame / FPS, gain=ss * ss, scale=pxs)
    BC.draw_source(fr.img, fr.zb, scam, frame, B, pxs)
    draw_pyre_source(fr.img, fr.zb, scam, frame, pxs, WD.MOON_DIR)
    img, zb, di = PI.to_target(fr)
    # target-space view depth from the Euclidean distance
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    r = tcam.ray(u + 0.5, v + 0.5)
    zt = (di * (tcam.f / np.linalg.norm(r, axis=-1))).astype(np.float32)
    zt[di > 1e8] = 1e9
    if mblur:
        V = RC.velocity(lambda f: RN.camera(f, W, H), frame, tcam, di, shutter=SHUTTER)
        img = RC.motion_blur(img, V, zt, max_r=int(40 * scale) + 8)
    size, _, _ = BC.env(frame, PYRE_IGN)
    if size > 0:
        draw_sparks_target(img, zt, _pyre_sparks(), frame, W, H)
    return img


def work(args):
    frames, scale, out, threads, ss = args
    os.environ['NUMBA_NUM_THREADS'] = str(threads)
    import numba
    numba.set_num_threads(threads)
    import cv2
    cv2.setNumThreads(1)
    times = []
    for f in frames:
        p = look.frame_path(out, f)
        t0 = time.time()
        hdr = render(f, scale=scale, ss=ss)
        img = look.finish(hdr, **FINISH)
        look.save_png(p, img)
        times.append(time.time() - t0)
        print(f'frame {f} {times[-1]:.1f}s', flush=True)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--skip', action='store_true', help='skip frames that already exist')
    a = ap.parse_args()
    root = PI.CM.ROOT
    base = os.path.join(root, 'renders', 'run_v2')
    out = base if a.out is None else os.path.join(base, a.out)
    os.makedirs(out, exist_ok=True)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        s, e = (a.range or '1520-1679').split('-')
        frames = list(range(int(s), int(e) + 1, a.step))
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.frame_path(out, f))]
    BC.place()          # build the chain once (cached to beacons.npy) before forking
    t0 = time.time()
    if a.procs <= 1:
        times = work((frames, a.scale, out, a.threads, a.ss))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            res = pool.map(work, [(c, a.scale, out, 1, a.ss) for c in chunks])
        times = [x for r in res for x in r]
    print(f'total {time.time() - t0:.1f}s, mean {sum(times) / max(len(times), 1):.1f}s/frame')


if __name__ == '__main__':
    main()
