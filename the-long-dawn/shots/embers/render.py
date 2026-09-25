"""EMBERS renderer driver.

usage:
  python render.py FRAMES [--scale 0.5] [--out DIR]
FRAMES: "300-1199" (inclusive), "300,340,480", or "300-400:10" (step)
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import look  # noqa: E402
from core import Frame, Camera, smoothstep, window  # noqa: E402

OUT = os.path.join(ROOT, 'renders', 'embers')

# shots: [start, end) -- the shutter never straddles a cut
SHOTS = [(300, 880), (880, 960), (960, 1040), (1040, 1200)]

TEXT = [(340, 440), (490, 550), (565, 635), (660, 730), (820, 900), (1055, 1195)]


def band_k(t):
    k = 0.0
    for a, b in TEXT:
        k = max(k, float(window(t, a, b, 10, 10)))
    return 0.62 * k


class Ctx:
    pass


_scene = None


def scene():
    global _scene
    if _scene is None:
        import timeline
        _scene = timeline.Timeline()
    return _scene


def render_frame(f, scale=1.0, outdir=OUT, save=True, verbose=True):
    sc = scene()
    t_start = time.time()
    s0, s1 = [s for s in SHOTS if s[0] <= f < s[1]][0]
    t0 = max(f - 0.25, s0)
    t1 = min(f + 0.25, s1 - 0.02)
    fr = Frame(scale)
    ctx = Ctx()
    ctx.f, ctx.t, ctx.t0, ctx.t1, ctx.fr, ctx.scale = f, float(f), t0, t1, fr, scale
    ctx.cam0 = sc.camera(t0)
    ctx.cam1 = sc.camera(t1)
    cm = sc.camera(float(f))
    ctx.cam = cm
    fr.set(focus=cm.focus, aperture=cm.aperture, band=(560, 700, band_k(f)),
           **sc.render_opts(f))
    sc.emit(ctx)
    hdr = fr.resolve()
    hdr = sc.post(ctx, hdr)
    if not np.isfinite(hdr).all():
        print(f'warning: non-finite values in frame {f}', flush=True)
        hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)
    fin = sc.finish_opts(f)
    img = look.finish(hdr, **fin)
    if save:
        look.save_png(look.frame_path(outdir, f), img)
    if verbose:
        print(f'frame {f} scale {scale} {time.time() - t_start:.2f}s', flush=True)
    return img


def parse_frames(s):
    out = []
    for part in s.split(','):
        step = 1
        if ':' in part:
            part, st = part.split(':')
            step = int(st)
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1, step))
        else:
            out.append(int(part))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('frames')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in parse_frames(a.frames):
        render_frame(f, a.scale, a.out)
