"""Render GLOBE frames.

  NUMBA_NUM_THREADS=2 python3 shots/globe/render.py answers 1752 1935            # deliver
  NUMBA_NUM_THREADS=2 python3 shots/globe/render.py dawn 2232 2495 --step 1
  python3 shots/globe/render.py answers 1760 1920 --step 40 --scale 0.5 --test   # WIP stills
  python3 shots/globe/render.py --finalize        # preview.mp4 + contact.png from delivered frames

Frames: renders/globe/f_%05d.png (global numbering). Tests: renders/globe/tests/<shot>_<frame>.png
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'lib'))
os.environ.setdefault('NUMBA_NUM_THREADS', '2')

import look  # noqa: E402

OUT = os.environ.get('LONGDAWN_GLOBE_OUT') or os.path.join(ROOT, 'renders', 'globe')   # e.g. renders/globe_B
TESTS = os.path.join(OUT, 'tests')
RANGES = {'answers': (1752, 1935), 'dawn': (2232, 2495)}


def finalize():
    import tempfile
    parts = []
    tmpd = tempfile.mkdtemp(dir=TESTS)
    for name, (a, b) in RANGES.items():
        mp = os.path.join(tmpd, f'{name}.mp4')
        look.preview_mp4(OUT, mp, a, b + 1)
        parts.append(mp)
    lst = os.path.join(tmpd, 'list.txt')
    with open(lst, 'w') as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                    '-c', 'copy', os.path.join(OUT, 'preview.mp4')], check=True)
    frames = [1752, 1760, 1776, 1792, 1808, 1824, 1840, 1856, 1872, 1888, 1904, 1919,
              2232, 2238, 2240, 2244, 2250, 2270, 2300, 2330, 2360, 2400, 2440, 2479]
    look.contact_sheet(OUT, frames, os.path.join(OUT, 'contact.png'), cols=4, thumb_w=480)
    print('preview + contact written')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shot', nargs='?')
    ap.add_argument('start', nargs='?', type=int)
    ap.add_argument('end', nargs='?', type=int)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--frames', default='')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--finalize', action='store_true')
    a = ap.parse_args()
    if a.finalize:
        finalize()
        return
    import shots
    shot = shots.SHOTS[a.shot]
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        frames = list(range(a.start, a.end + 1, a.step))
    os.makedirs(TESTS, exist_ok=True)
    for fr in frames:
        if a.test:
            path = os.path.join(TESTS, f'{a.shot}{a.tag}_{fr:05d}.png')
        else:
            path = look.frame_path(OUT, fr)
            if a.scale != 1.0:
                raise SystemExit('deliverable frames must be full resolution')
        if a.skip_existing and os.path.exists(path):
            continue
        t0 = time.time()
        img = shot.render(float(fr), a.scale)
        look.save_png(path, img)
        print(f'{a.shot} {fr} {time.time() - t0:.1f}s -> {os.path.relpath(path, ROOT)}', flush=True)


if __name__ == '__main__':
    main()
