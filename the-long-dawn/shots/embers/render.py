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
OUT_ALT = os.path.join(ROOT, 'renders', 'embers_v2_alt_codedtowers')
OUT_ALT3 = os.path.join(ROOT, 'renders', 'embers_A3_alt_codedtowers')
OUTS = {'A': os.path.join(ROOT, 'renders', 'embers_v2'), 'B': os.path.join(ROOT, 'renders', 'embers_B'),
        'C': os.path.join(ROOT, 'renders', 'embers_C'),
        'A3': os.path.join(ROOT, 'renders', 'embers_A3'),     # v3: A's own frames (BIBLE_V3 locked beat sheet)
        'C3': os.path.join(ROOT, 'renders', 'embers_C3')}

# shots: [start, end) -- the shutter never straddles a cut
SHOTS = [(300, 880), (880, 960), (960, 1040), (1040, 1200)]
SHOTS_V3 = {'A3': [(560, 1040), (1040, 1760), (1760, 1840), (1840, 2468), (2468, 2640), (2640, 2800), (2800, 3120),
                   (4400, 4880)],
            'C3': [(1040, 1680), (1920, 2080), (2320, 2480), (2720, 2840)]}

# text windows (v2 frames = src frames here) per cut, from the edit's current titles (director, framing review).
# The band y~560-700 is calmed during these. Lines on black (1060-1186) sit mid-frame; kept for completeness.
TEXT = {
    'A': [(340, 440), (490, 565), (580, 648), (668, 738), (800, 866), (1060, 1186)],
    'B': [],
    'C': [(340, 440), (490, 572), (656, 722), (744, 804), (1052, 1190)],   # text red-team retime (19:20Z)
    # v3, A's own frames (locked sheet): T2, T4, T5, T6a/b, T7, T8, T9 (T10 is centred on black)
    'A3': [(700, 840), (1140, 1270), (1460, 1610), (1640, 1800), (1848, 1944), (1960, 2060), (2670, 2790)],
    'C3': [(1056, 1190), (1446, 1550)],     # T5b, T6 (T8 sits on black)
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
        import variant
        if variant.CUT == 'A3':
            import a3
            _scene = a3.TimelineA3()
        elif variant.CUT == 'C3':
            import c3
            _scene = c3.TimelineC3()
            import c2                   # EMBERS-C2: C9 THE EYE, C11 THE GRASP, C13a THE RING FALLS (c2.SHOTS2)
            _scene = c2.Router(_scene)
        else:
            import timeline
            _scene = timeline.Timeline()
    return _scene


def render_frame(f, scale=1.0, outdir=None, save=True, verbose=True):
    sc = scene()
    t_start = time.time()
    import variant
    s0, s1 = [s for s in SHOTS_V3.get(variant.CUT, SHOTS) if s[0] <= f < s[1]][0]
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
        outdir = outdir or ((OUT_ALT3 if variant.CUT == 'A3' else OUT_ALT) if variant.TOWERS_ALT else OUTS[variant.CUT])
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
    ap.add_argument('--cut', default='A', choices=['A', 'B', 'C', 'A3', 'C3'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--alt-towers', action='store_true',
                    help='A only: the "coded pair" alternate (giants as pagoda + obelisk) -> embers_v2_alt_codedtowers')
    a = ap.parse_args()
    import variant
    variant.set_cut(a.cut)
    variant.TOWERS_ALT = bool(a.alt_towers)
    assert not (a.alt_towers and a.cut not in ('A', 'A3')), 'the coded-towers alternate exists for cut A only'
    out = a.out or ((OUT_ALT3 if a.cut == 'A3' else OUT_ALT) if a.alt_towers else OUTS[a.cut])
    os.makedirs(out, exist_ok=True)
    for f in parse_frames(a.frames):
        render_frame(f, a.scale, out)
