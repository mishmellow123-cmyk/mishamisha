"""EMBERS renderer driver.

usage:
  python render.py FRAMES [--cut A|B|C] [--scale 0.5] [--out DIR]
FRAMES: "300-1199" (inclusive), "300,340,480", or "300-400:10" (step)
--cut: A (default) -> renders/embers_v2, B -> renders/embers_B (no text band), C -> renders/embers_C (Tolkien).
The delivered v1 frames in renders/embers are never written by default.
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

OUT_V1 = os.path.join(ROOT, 'renders', 'embers')
OUTS = {'A': os.path.join(ROOT, 'renders', 'embers_v2'), 'B': os.path.join(ROOT, 'renders', 'embers_B'),
        'C': os.path.join(ROOT, 'renders', 'embers_C')}

# shots: [start, end) -- the shutter never straddles a cut
SHOTS = [(300, 880), (880, 960), (960, 1040), (1040, 1200)]

# text windows (v2 frames = src frames here) per cut, from the edit's current titles (director, framing review).
# The band y~560-700 is calmed during these. Lines on black (1060-1186) sit mid-frame; kept for completeness.
TEXT = {
    'A': [(340, 440), (490, 565), (580, 648), (668, 738), (800, 866), (1060, 1186)],
    'B': [],
    'C': [(340, 440), (490, 565), (628, 695), (705, 770), (780, 834), (1060, 1186)],
}


def band_k(t):
    import variant
    if not variant.text_band():
        return 0.0
    k = 0.0
    for a, b in TEXT[variant.CUT]:
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


def render_frame(f, scale=1.0, outdir=None, save=True, verbose=True):
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
        import variant
        outdir = outdir or OUTS[variant.CUT]
        assert os.path.abspath(outdir) != os.path.abspath(OUT_V1), 'refusing to overwrite the delivered v1 frames'
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
    ap.add_argument('--cut', default='A', choices=['A', 'B', 'C'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    import variant
    variant.set_cut(a.cut)
    out = a.out or OUTS[a.cut]
    os.makedirs(out, exist_ok=True)
    for f in parse_frames(a.frames):
        render_frame(f, a.scale, out)
