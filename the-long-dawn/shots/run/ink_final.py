"""C · THE LIVING INK: production driver (cloud). Per frame: the Run world's AOVs (render_ink.py) at 2x -> the ink
pass (inkpass.py), drawn at 3840x1608 -> downsampled (INTER_AREA) to a final 1920x804 frame. No caches are written. For THE ILLUMINATION the dawn's colour plate is
rendered in the same worker (dawn.py's terrain and smoke, WITHOUT the eagles: C cuts them) and laid in where the sun
touches.

  python shots/run/ink_final.py --shot scroll --range 0-319 --procs 4 --skip      # -> renders/runC_scroll/ (C17)
  python shots/run/ink_final.py --shot reveal --range 0-239 --procs 4 --skip      # -> renders/runC_reveal/ (C16)
  python shots/run/ink_final.py --shot illum --range 2398-2877 --procs 4 --skip   # -> renders/runC_illum/ (C24 local 0-479)
  python shots/run/ink_final.py --shot scroll --frames 152 --scale 0.5 --out tests/t   # a quick test
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '1')


def illum_plate(frame, scale, ss, Wt, Ht):
    """The dawn in colour (display-referred, linearised), no eagles, at the AOV resolution. It only feeds a wash
    blurred by ~6 px, so it is rendered at half scale without supersampling (about a ninth of the cost)."""
    scale, ss = 0.5 * scale, 1.0
    import cv2
    import dawn as DN
    import pipe as PI
    look = PI.look
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = DN.camera(frame, W, H)
    fr = PI.Frame(tcam, ss)
    DN.render_terrain(fr, frame)                # (the smoke is drawn in ink; the plate only gives the wash its colour)
    img, zb, di = PI.to_target(fr)
    srgb = look.finish(img, **DN.FINISH)
    lin = look.srgb_to_linear(srgb)
    if lin.shape[:2] != (Ht, Wt):
        lin = cv2.resize(lin, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    return lin.astype(np.float32)


def work(args):
    shot_name, frames, scale, ss, out, threads = args
    os.environ['NUMBA_NUM_THREADS'] = str(threads)
    import numba
    numba.set_num_threads(threads)
    import cv2
    cv2.setNumThreads(1)
    import render_ink as RI
    import inkpass as IP
    look = IP.look
    shot = RI.Shot(shot_name)
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    for f in frames:
        t0 = time.time()
        R = RI.render_aov(shot, f, scale, ss)
        plate = None
        if shot_name == 'illum':
            Ht, Wt = R['A'].shape[:2]
            plate = illum_plate(f, scale, ss, Wt, Ht)
        img, _ = IP.compose(R, B=shot.B, plate=plate, CR=shot.CR)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        look.save_png(look.frame_path(out, f), img)
        print(f'{shot_name} {f} {time.time() - t0:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shot', required=True, choices=['reveal', 'scroll', 'illum'])
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=2.0, help='supersampling: the page is drawn at 2x and downsampled (H5)')
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default=None, help='sub-folder of renders/runC_<shot>/ (tests)')
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    import pipe as PI
    look = PI.look
    base = os.path.join(PI.CM.ROOT, 'renders', f'runC_{a.shot}')
    out = base if a.out is None else os.path.join(base, a.out)
    os.makedirs(out, exist_ok=True)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1))
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.frame_path(out, f))]
    if a.shot == 'scroll':
        import render_ink as RI
        RI.scroll_beacons()                     # deterministic; built once before forking
    t0 = time.time()
    if a.procs <= 1:
        work((a.shot, frames, a.scale, a.ss, out, a.threads))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(work, [(a.shot, c, a.scale, a.ss, out, 1) for c in chunks])
    print(f'total {time.time() - t0:.1f}s for {len(frames)} frames')


if __name__ == '__main__':
    main()
