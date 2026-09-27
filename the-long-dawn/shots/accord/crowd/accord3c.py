"""accord3 with the crowd installed (ACCORD-CROWD's driver; the same frames as accord3.py, plus the crowd).

  python shots/accord/crowd/accord3c.py range --frames 4480,4540-4545 --outdir renders/accord_C [--scale 0.5]
  python shots/accord/crowd/accord3c.py still 4480 [--scale 0.25] [--window x0,y0,w,h]  -> renders/accord_C/tests/crowd/
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.path.dirname(HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
for p in (HERE, ACC):
    if p not in sys.path:
        sys.path.insert(0, p)

import accord3 as A3  # noqa: E402
import crowd3 as CR  # noqa: E402
from scene3 import look  # noqa: E402

CR.install(A3)


def frames_of(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def render(f, scale, window=None):
    t0 = time.time()
    hdr, info = A3.render_frame(float(f), scale, window=window)
    img = A3.finish(hdr, float(f))
    return img, time.time() - t0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['still', 'range'])
    ap.add_argument('a', nargs='?')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--scale', type=float, default=None)
    ap.add_argument('--window', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    import cv2
    cv2.setNumThreads(1)
    if args.cmd == 'still':
        win = tuple(float(v) for v in args.window.split(',')) if args.window else None
        sc = args.scale if args.scale is not None else (1.0 if win else 0.5)
        img, dt = render(float(args.a), sc, win)
        d = os.path.join(A3.OUT, 'tests', 'crowd')
        w = '' if win is None else '_w' + '-'.join(str(int(v)) for v in win)
        p = os.path.join(d, f'crowd_{int(float(args.a))}{args.tag}{w}_s{sc}.png')
        look.save_png(p, img)
        print(f'{p}  {dt:.1f}s', flush=True)
    else:
        frames = frames_of(args.frames) if args.frames else [int(args.a)]
        sc = args.scale if args.scale is not None else 1.0
        dest = args.outdir or A3.OUT
        dest = dest if os.path.isabs(dest) else os.path.join(A3.SC.ROOT, dest)
        os.makedirs(dest, exist_ok=True)
        for f in frames:
            p = look.frame_path(dest, f)
            if (os.path.exists(p) or os.path.exists(p[:-4] + '.jpg')) and not args.force:
                continue
            img, dt = render(f, sc)
            look.save_png(p, img)
            print(f'frame {f} {dt:.1f}s', flush=True)
