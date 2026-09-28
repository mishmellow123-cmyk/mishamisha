"""C v5 paired promise, absolute C2880..3119.

The existing reveal's wide camera poses160..239 are stretched over this shot,
so both fires stay small. The distant site is summit catalogue row886, refined
onto its actual terrain apex. Both use the unchanged ink-fire envelope at C2880.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from watch_c_v5 import write_image

FIRST, LAST = 2880, 3119
CAMERA_FIRST, CAMERA_LAST = 160., 239.
FAR_SUMMIT = 886
IGNITION = 0.


def local_frame(frame):
    if not FIRST <= frame <= LAST:
        raise ValueError(f'Paired reveal frame {frame} outside {FIRST}..{LAST}')
    return int(frame) - FIRST


def camera_frame(local):
    return CAMERA_FIRST + (CAMERA_LAST - CAMERA_FIRST) * np.clip(local / (LAST - FIRST), 0., 1.)


def camera(local, W=1920, H=804):
    import render_ink as RI
    return RI.reveal_camera(camera_frame(local), W, H)


def make_shot():
    import render_ink as RI
    shot = RI.Shot('reveal')
    shot.cam = camera
    cat = np.load(Path(__file__).with_name('summits.npy'))
    p = cat[FAR_SUMMIT, :3]
    far = RI.WD.summit_near(p[0], p[2], shot.CR, rad=45., n=15)
    near = shot.B[0].copy()
    near[3] = IGNITION
    shot.B = np.array([near, [*far, IGNITION, 1., 1., np.linalg.norm(far-camera(0).pos)]], dtype=np.float64)
    return shot


def render(frame, shot, scale=.5, ss=2.):
    import cv2
    import render_ink as RI
    import inkpass as IP
    local = local_frame(frame)
    aov = RI.render_aov(shot, local, scale, ss)
    aov['kpx'] = scale * ss
    rgb, _ = IP.compose(aov, B=shot.B, CR=shot.CR)
    width, height = int(round(1920*scale)), int(round(804*scale))
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
        raise ValueError('Invalid paired reveal render')
    return rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs = ap.add_mutually_exclusive_group(required=True)
    fs.add_argument('--frames')
    fs.add_argument('--range')
    ap.add_argument('--scale', type=float, default=1.)
    ap.add_argument('--ss', type=float, default=2.)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--format', choices=('png','jpg'), default='jpg')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if not np.isfinite([a.scale,a.ss]).all() or min(a.scale,a.ss)<=0 or a.threads<1:
        ap.error('scale, ss and threads must be finite and positive')
    try:
        if a.frames:
            frames = [int(f) for f in a.frames.split(',')]
        else:
            lo, hi = (int(f) for f in a.range.split('-'))
            frames = list(range(lo, hi + 1))
        if not frames or len(frames)!=len(set(frames)):
            raise ValueError('Supply distinct frames or a nonempty increasing range')
        for f in frames: local_frame(f)
    except (ValueError,IndexError) as exc:
        ap.error(str(exc))
    a.out.mkdir(parents=True,exist_ok=True)
    paths = [a.out/f'f_{f:05d}.{a.format}' for f in frames]
    if any(p.exists() for p in paths): ap.error('Output frame exists; select a fresh directory')
    receipt = dict(status='initializing',scale=a.scale,ss=a.ss,
                   camera_source_frames=[float(camera_frame(local_frame(f))) for f in frames],
                   catch_frame=FIRST,far_summit_row=FAR_SUMMIT,
                   frames=[])
    rp = a.out/f'receipt_{frames[0]:05d}_{frames[-1]:05d}.json'
    if rp.exists(): ap.error('Receipt exists; select a fresh directory')
    rp.write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        os.environ.setdefault('NUMBA_NUM_THREADS',str(a.threads))
        import numba
        import cv2
        shot = make_shot()
        cv2.setNumThreads(0)
        numba.set_num_threads(min(a.threads,numba.config.NUMBA_NUM_THREADS))
        receipt.update(status='running',beacons=shot.B.tolist(),
                       numba_threads=numba.get_num_threads(),cv_threads=cv2.getNumThreads())
        for f,p in zip(frames,paths):
            t=time.perf_counter();rgb=render(f,shot,a.scale,a.ss);write_image(p,rgb)
            row=dict(frame=f,file=p.name,shape=list(rgb.shape),seconds=time.perf_counter()-t,
                     sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            receipt['frames'].append(row);rp.write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        receipt['status']='complete'
    except BaseException:
        receipt['status']='failed';receipt['traceback']=traceback.format_exc();raise
    finally:
        rp.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__': main()
