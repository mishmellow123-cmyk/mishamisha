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
    if name == 'beacon_v3':               # BIBLE_V3: the flint take re-keyed (B7: gloves, the head a rim-lit silhouette)
        import beacon                     # AND on the locked sheets' master timing (src 1200-1555, strike 1 = 1236)
        beacon.V3_REKEY = True
        beacon.use_master_timing()
        beacon.apply_h5_calls()           # the director's H5 calls (07:10Z): wardrobe, sparks, flinch, lift, basket
        return beacon.FirstBeacon()
    if name == 'beacon_v3_roar2':         # HEROINE-L: beacon_v3 + the roar re-polish (differs from beacon_v3 from src 1476)
        import beacon
        beacon.V3_REKEY = True
        beacon.use_master_timing()
        beacon.apply_h5_calls()
        beacon.apply_roar2()
        return beacon.FirstBeacon()
    if name == 'beacon_v3_afix':          # A-FIX (28 Sep, film A only): beacon_v3_roar2 + apply_afix(): no strike halo,
        import beacon                     # no "A" woodpile, no lurch, dark scarf till the catch, pale breath, the roar
        beacon.V3_REKEY = True            # ramped, no pull-back, a slow push-in. src numbering -> renders/h1_A
        beacon.use_master_timing()
        beacon.apply_h5_calls()
        beacon.apply_roar2()
        beacon.apply_afix()
        return beacon.FirstBeacon()
    if name == 'h1c':                     # HEROINE-L: C14's H1-C (hands and flint only), frames in C NUMBERING:
        import beacon                     # C 2960-2999 = src 1216-1255 (strike 1 = C 2980), C 3150-3359 = src
        beacon.V3_REKEY = True            # 1266-1475 (strike 3 = C 3178, the catch C 3316; the roar C 3360 is
        beacon.use_master_timing()        # MONTAGE-3D's fire test). The find (C 3000-3149) lies between.
        beacon.apply_h5_calls()
        beacon.apply_h1c()
        return H1CShot(beacon.FirstBeacon())
    if name == 'beacon_v3_pre_h5':        # the master-timing take as rendered 05:09Z (before the H5 calls)
        import beacon
        beacon.V3_REKEY = True
        beacon.use_master_timing()
        return beacon.FirstBeacon()
    if name == 'beacon_v3_rekey':         # the re-key alone on the accepted v2b timing (renders/hills_v3, 00:42Z)
        import beacon
        beacon.V3_REKEY = True
        return beacon.FirstBeacon()
    if name in ('deadember', 'find', 'firetest'):   # BIBLE_V3 H5 / H2 close-ups (frames = seconds x 24 in the cut)
        import heroine_v3
        return heroine_v3.SHOTS[name]()
    if name == 'coda':
        import coda
        return coda.Coda()
    raise ValueError(name)


class H1CShot:
    """C frames -> the master take's src frames (BIBLE_V3 locked sheet C14: the find lies between strikes 2 and 3)."""
    SEGS = ((2960, 3000, -1744), (3150, 3360, -1884))

    def __init__(self, shot):
        self.shot = shot

    def src(self, f):
        for a, b, off in self.SEGS:
            if a <= f < b:
                return f + off
        raise ValueError(f'C frame {f} is not an H1-C frame (2960-2999, 3150-3359)')

    def render(self, f, scale):
        return self.shot.render(self.src(f), scale)


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
