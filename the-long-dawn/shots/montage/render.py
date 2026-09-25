"""MONTAGE render driver.

  python render.py s1 --frames 1440,1466,1480 --scale 0.5 --out tests/s1      # test stills
  python render.py s1 --range 1440-1519 --scale 1.0                           # final frames
  python render.py all --scale 1.0 --procs 2                                  # everything

Frames are written with GLOBAL numbering to renders/montage/f_%05d.png (or --out subdir).
CPU etiquette: --procs N workers x 1 numba thread each (default 1 worker x 2 threads).
"""
import argparse
import importlib
import os
import sys
import time

SHOTS = {
    's1': ('s1_peak', 1440, 1519),
    's2': ('s2_desert', 1520, 1579),
    's3': ('s3_ice', 1580, 1639),
    's4': ('s4_karst', 1640, 1679),
    's5': ('s5_city', 1680, 1719),
    's6': ('s6_sea', 1720, 1767),
}


def shot_of(frame):
    for k, (m, a, b) in SHOTS.items():
        if a <= frame <= b:
            return k
    raise ValueError(frame)


def work(args):
    frames, scale, out, threads = args
    os.environ['NUMBA_NUM_THREADS'] = str(threads)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import numba
    numba.set_num_threads(threads)
    import cv2
    cv2.setNumThreads(1 if threads == 1 else 2)
    import common as CM
    look = CM.look
    mods = {}
    times = []
    for f in frames:
        k = shot_of(f)
        if k not in mods:
            mods[k] = importlib.import_module(SHOTS[k][0])
        m = mods[k]
        t0 = time.time()
        hdr = m.render(f, scale=scale)
        img = look.finish(hdr, **m.FINISH)
        look.save_png(look.frame_path(out, f), img)
        dt = time.time() - t0
        times.append(dt)
        print(f'frame {f} ({k}) {dt:.1f}s', flush=True)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shot')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--sheet', action='store_true')
    ap.add_argument('--cols', type=int, default=4)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..', '..'))
    base = os.path.join(root, 'renders', 'montage')
    out = base if a.out is None else os.path.join(base, a.out)
    os.makedirs(out, exist_ok=True)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    elif a.range:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1, a.step))
    elif a.shot == 'all':
        frames = list(range(1440, 1768, a.step))
    else:
        _, s, e = SHOTS[a.shot]
        frames = list(range(s, e + 1, a.step))
    t0 = time.time()
    if a.procs <= 1:
        times = work((frames, a.scale, out, 2))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        ctx = mp.get_context('spawn')
        with ctx.Pool(a.procs) as pool:
            res = pool.map(work, [(c, a.scale, out, 1) for c in chunks])
        times = [x for r in res for x in r]
    print(f'total {time.time() - t0:.1f}s, mean {sum(times) / max(len(times), 1):.1f}s/frame')
    if a.sheet:
        sys.path.insert(0, here)
        import common as CM
        name = os.path.join(out, 'sheet_' + '_'.join(str(f) for f in frames[:1]) + f'_{len(frames)}.png')
        CM.look.contact_sheet(out, frames, name, cols=a.cols, thumb_w=int(1920 * a.scale) if len(frames) <= 2
                              else min(640, int(1920 * a.scale)))
        print('sheet', name)


if __name__ == '__main__':
    main()
