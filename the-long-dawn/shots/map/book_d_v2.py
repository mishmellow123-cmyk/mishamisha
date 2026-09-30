"""Explicit D-v2 book shots; every input and output frame is a v2 D frame.

The v1 entry point remains reproducible. The optional title is its unchanged
render with an absolute-frame offset, rather than a new lighting revision.
"""
import argparse
import json
import math
import os
import resource
import sys
import time

for _pool in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_pool, '1')

import numpy as np
import book_c as BC
import book_c_v5 as V
import book_d as D1
import lastleaf as L

RANGES = {'oldfire': (5840, 6080), 'lastleaf': (8320, 8640), 'title': (8880, 9120)}


def phase(shot, frame):
    start, end = RANGES[shot]
    if (isinstance(frame, bool) or not isinstance(frame, (int, np.integer))
            or not start <= frame < end):
        raise ValueError(f'{shot} accepts integer v2 D frames {start}..{end-1}')
    return (frame-start)/(end-start-1)


class RetainedTitle:
    def __init__(self, W=1920, H=804, ppc=90, gap_degrees=50.):
        self.W, self.H = W, H
        self.source = D1.DSpread('title', W, H, ppc, gap_degrees)

    def frame(self, frame):
        phase('title', frame)
        return self.source.frame(frame-2000)


def make_renderer(shot, scale=1., ppc=90, gap_degrees=50., caption_hold=False):
    if shot not in RANGES:
        raise ValueError('unknown D-v2 book shot')
    if caption_hold and shot != 'oldfire':
        raise ValueError('caption_hold is only defined for oldfire')
    if not math.isfinite(scale) or not 0 < scale <= 1 or int(804*scale) < 1:
        raise ValueError('scale must produce positive dimensions, at most native')
    if not isinstance(ppc, int) or isinstance(ppc, bool) or ppc <= 0:
        raise ValueError('ppc must be a positive integer')
    width, height = int(1920*scale), int(804*scale)
    if shot == 'oldfire':
        from oldfire_v2 import OldFireV2
        return OldFireV2(width, height, ppc, caption_hold=caption_hold)
    if shot == 'lastleaf':
        from lastleaf_v2 import LastLeafV2
        return LastLeafV2(width, height, ppc, gap_degrees)
    return RetainedTitle(width, height, ppc, gap_degrees)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shot', required=True, choices=tuple(RANGES))
    parser.add_argument('--frames', required=True, help='absolute v2 D frame list or inclusive range')
    parser.add_argument('--out', required=True, type=L.output_dir)
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--ppc', type=int, default=90)
    parser.add_argument('--gap-degrees', type=float, default=50.)
    parser.add_argument('--caption-hold', action='store_true',
                        help='oldfire only: retain the fall/flare, complete caption by D5983')
    args = parser.parse_args(argv)
    if args.caption_hold and args.shot != 'oldfire':
        parser.error('--caption-hold requires --shot oldfire')
    frames = BC.frames_of(args.frames)
    try:
        if not frames:
            raise ValueError('empty frame range')
        for frame in frames:
            phase(args.shot, frame)
    except ValueError as exc:
        parser.error(str(exc))
    renderer = make_renderer(args.shot, args.scale, args.ppc, args.gap_degrees, args.caption_hold)
    matte = args.out.with_name(args.out.name+'_matte')
    for frame in frames:
        started = time.perf_counter()
        hdr, alpha = renderer.frame(frame)
        if not np.isfinite(hdr).all() or not np.isfinite(alpha).all():
            raise RuntimeError('non-finite D-v2 book frame')
        rgb = BC.look.finish(hdr, exposure=1.15, bloom_strength=.06,
                             bloom_threshold=1.2, vignette_amount=.32)
        V.save(args.out/f'f_{frame:05d}.png', rgb, 'png')
        V.save(matte/f'f_{frame:05d}.png', np.repeat(alpha[..., None], 3, -1), 'png')
        print(json.dumps(dict(shot=args.shot, frame=frame, seconds=time.perf_counter()-started,
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
              (1 if sys.platform == 'darwin' else 1024), width=renderer.W,
              height=renderer.H, ppc=args.ppc)), flush=True)


if __name__ == '__main__':
    main()
