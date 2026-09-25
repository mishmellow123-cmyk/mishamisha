#!/usr/bin/env python3
"""HILLS renderer CLI.

  python3 shots/hills/render.py --shot intro --frames 0-359 [--step 1] [--scale 1.0]
                                [--out renders/hills] [--workers 2] [--skip-existing]

Frames are global frame numbers; output f_%05d.png via look.save_png. Each worker is
a separate process with NUMBA_NUM_THREADS=1 (CPU etiquette: <= 2 cores total).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_frames(spec, step):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1, step))
        else:
            out.append(int(part))
    return out


def make_shot(name):
    if name == 'intro':
        import intro
        return intro.Intro()
    if name == 'beacon':
        import beacon
        return beacon.FirstBeacon()
    if name == 'coda':
        import coda
        return coda.Coda()
    raise ValueError(name)


def worker(args):
    name, frames, scale, out, skip, wid = args
    os.environ['NUMBA_NUM_THREADS'] = '1'
    sys.path.insert(0, HERE)
    import cv2
    cv2.setNumThreads(1)
    from core import look
    shot = make_shot(name)
    t0 = time.time()
    done = 0
    for f in frames:
        path = look.frame_path(out, f)
        if skip and os.path.exists(path):
            continue
        t1 = time.time()
        img = shot.render(f, scale)
        look.save_png(path, img)
        done += 1
        print(f'[w{wid}] {name} f{f} {time.time() - t1:.1f}s', flush=True)
    return done, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shot', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', default=os.path.join(HERE, '..', '..', 'renders', 'hills'))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--skip-existing', action='store_true')
    a = ap.parse_args()
    frames = parse_frames(a.frames, a.step)
    os.makedirs(a.out, exist_ok=True)
    nw = max(1, min(a.workers, 2, len(frames)))
    chunks = [frames[i::nw] for i in range(nw)]
    jobs = [(a.shot, c, a.scale, a.out, a.skip_existing, i) for i, c in enumerate(chunks)]
    t0 = time.time()
    if nw == 1:
        res = [worker(jobs[0])]
    else:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(nw) as pool:
            res = pool.map(worker, jobs)
    n = sum(r[0] for r in res)
    dt = time.time() - t0
    print(f'rendered {n} frames in {dt:.0f}s ({dt / max(1, n) * nw:.1f}s/frame/worker)')


if __name__ == '__main__':
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    sys.path.insert(0, HERE)
    main()
