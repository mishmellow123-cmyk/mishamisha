"""s1 (FAR PEAK, 1440-1519) re-rendered with the Run's fire so the shepherd's fire and the Run's match
(picture critic: s1's flame tip read magenta). shots/montage/ is untouched: its s1_peak.render() runs as is,
with mt.fire's flame, spark colour ramp and air-glow colour swapped for fire2's (orange -> yellow -> white ramp,
no crimson end; the flame body absorbs what is behind it, so the blue sky never shows through the tips).

  python shots/run/s1_v2.py --frames 1500 --scale 0.5 --out tests/s1v2     # test (under renders/run_v2/tests/)
  python shots/run/s1_v2.py --range 1440-1519 --procs 4 --skip             # final -> renders/montage_v2/
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MONT = os.path.abspath(os.path.join(HERE, '..', 'montage'))
sys.path.insert(0, MONT)
sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '1')


def _patch():
    from mt import fire as F
    import fire2 as F2
    if getattr(F, '_v2_patched', False):
        return
    orig_halo = F.halo

    def flame(img, depth, cam, base_world, Hf, Rb, t, seed=0, I=24.0, lean=0.0, zbias=0.5, tongues=5, warp=1.0,
              min_px=2.0, debug=0, TG=None):
        F2.flame(img, depth, cam, base_world, Hf, Rb, t, seed=seed, I=I, lean=lean, zbias=zbias, tongues=tongues,
                 warp=warp, min_px=min_px, TG=TG)

    def halo(img, depth, sx, sy, sig_px, peak, z=1e9, zbias=0.0, col=None, squash=1.0):
        orig_halo(img, depth, sx, sy, sig_px, peak, z=z, zbias=zbias, col=F2.GLOW_COL if col is None else col,
                  squash=squash)

    F.flame = flame
    import numpy as np
    F.bb_vec = lambda T: F2.ramp_vec(np.asarray(T, np.float64) * 0.85)   # sparks: born orange-yellow, never crimson
    F.halo = halo
    F._v2_patched = True


def work(args):
    frames, scale, out, threads = args
    os.environ['NUMBA_NUM_THREADS'] = str(threads)
    import numba
    numba.set_num_threads(threads)
    _patch()
    import common as CM
    import s1_peak as S1
    look = CM.look
    for f in frames:
        t0 = time.time()
        img = look.finish(S1.render(f, scale=scale), **S1.FINISH)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default='1440-1519')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    root = os.path.abspath(os.path.join(HERE, '..', '..'))
    out = os.path.join(root, 'renders', 'montage_v2') if a.out is None else \
        os.path.join(root, 'renders', 'run_v2', a.out)
    os.makedirs(out, exist_ok=True)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1))
    if a.skip:
        frames = [f for f in frames if not os.path.exists(os.path.join(out, f'f_{f:05d}.png'))]
    if a.procs <= 1:
        work((frames, a.scale, out, a.threads))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(work, [(c, a.scale, out, 1) for c in chunks])


if __name__ == '__main__':
    main()
