#!/usr/bin/env python3
"""Render opt-in false-dawn variants to an explicit directory; no farm submission.

Frames are absolute cut-D frames and ranges are inclusive. Farm manifests use
plain --range A-B so farm.py can split them; local probes may add :STRIDE.
Each JSON receipt reports render/finish/save time and process peak RSS.
"""
import argparse
from dataclasses import asdict
import importlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import tempfile
import time

for _name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parents[1]
RANGES = {'brink': (4080, 4239), 'twofires': (4560, 5039),
          'watch': (7040, 7359), 'truedawn': (7360, 7839)}
LENGTHS = {variant: last - first + 1 for variant, (first, last) in RANGES.items()}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=LENGTHS, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--range', dest='frame_range', metavar='A-B[:STRIDE]')
    mode.add_argument('--frames', metavar='A,B,C')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--beat-offset', type=int, default=0)
    ap.add_argument('--pair-frame', type=int, default=4320, help='absolute D frame when both crowns ignite')
    ap.add_argument('--cascade-frame', type=int, default=4760, help='earliest absolute D frame of the approaching catches')
    return ap


def selected_frames(args):
    shot_first, shot_last = RANGES[args.variant]
    if args.frame_range is not None:
        match = re.fullmatch(r'(\d+)-(\d+)(?::(\d+))?', args.frame_range)
        if not match:
            raise ValueError('--range must be an inclusive A-B[:STRIDE] range')
        first, last = map(int, match.groups()[:2])
        stride = int(match.group(3) or 1)
        if first > last or stride < 1:
            raise ValueError('--range requires A <= B and a positive stride')
        # Validate endpoints before allocating a user-specified range.
        if first < shot_first or last > shot_last:
            raise ValueError('Frame range is outside this variant\'s absolute D frames')
        return list(range(first, last + 1, stride))
    if not args.frames or not re.fullmatch(r'\d+(?:,\d+)*', args.frames):
        raise ValueError('--frames must be a comma-separated list of frame numbers')
    frames = [int(part) for part in args.frames.split(',')]
    if min(frames) < shot_first or max(frames) > shot_last:
        raise ValueError('Frame list is outside this variant\'s absolute D frames')
    if len(frames) != len(set(frames)):
        raise ValueError('--frames must not repeat a frame')
    return frames


def validate_options(args):
    if not math.isfinite(args.scale) or not 0 < args.scale <= 1 or round(804 * args.scale) < 1:
        raise ValueError('--scale must produce positive dimensions, at most native')
    if not math.isfinite(args.ss) or not 0 < args.ss <= 2:
        raise ValueError('--ss must be finite and satisfy 0 < ss <= 2')
    if min(round(round(804 * args.scale) * args.ss), round(round(1920 * args.scale) * args.ss)) < 1:
        raise ValueError('--ss must produce positive supersampled dimensions')
    if args.pair_frame < 0 or args.cascade_frame <= args.pair_frame:
        raise ValueError('Ignition frames must satisfy 0 <= pair-frame < cascade-frame')


def output_directory(out, variant):
    """Allow external probes and dedicated farm directories; never owner renders.

    Check resolved ancestry as well as the supplied path: an absolute path or a
    second symlink can otherwise reach the target of the worktree's renders link.
    """
    out = Path(out)
    if not out.is_absolute():
        out = ROOT / out
    resolved = out.resolve()
    renders = ROOT / 'renders'
    if renders.is_symlink() and resolved.is_relative_to(renders.resolve()):
        raise ValueError('Refusing output inside the linked owner renders directory')
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise ValueError('Output must be a real directory')
    if resolved.is_relative_to(ROOT.resolve()):
        expected = (ROOT / 'renders' / f'falsedawn_{variant}').resolve()
        if resolved != expected or any(p.is_symlink() for p in (out, out.parent)):
            raise ValueError('In-repository output must be renders/falsedawn_<variant>')
    return resolved


def load_renderer():
    sys.path.insert(0, str(ROOT / 'shots' / 'run'))
    sys.path.append(str(ROOT / 'lib'))
    module = importlib.import_module('glowvars')
    if module.RANGES != RANGES or module.LENGTHS != LENGTHS:
        raise RuntimeError('CLI frame ranges disagree with glowvars.RANGES or LENGTHS')
    import cv2
    import numba
    cv2.setNumThreads(0)
    numba.set_num_threads(1)
    return module


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def save_frame(look, path, rgb):
    """Use the Run world's dithered writer, then publish without overwriting."""
    with tempfile.TemporaryDirectory(prefix='.glowvars-', dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        look.save_png(str(staged), rgb)
        os.link(staged, path)


def checked_rgb(module, frame, args, config):
    import numpy as np
    hdr = module.render(frame, variant=args.variant, scale=args.scale, ss=args.ss, config=config)
    shape = (round(804 * args.scale), round(1920 * args.scale), 3)
    if hdr.shape != shape or not np.isfinite(hdr).all():
        raise ValueError('Unexpected dimensions or non-finite HDR output')
    rgb = module.FD.PI.look.finish(hdr, **module.FD.FINISH)
    if rgb.shape != shape or not np.isfinite(rgb).all():
        raise ValueError('Unexpected dimensions or non-finite finished output')
    return rgb


def main(argv=None):
    args = parser().parse_args(argv)
    frames = selected_frames(args)
    validate_options(args)
    out = output_directory(args.out, args.variant)
    paths = [out / f'f_{frame:05d}.png' for frame in frames]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError('A requested output frame already exists; use a fresh probe directory')
    # Safety checks precede imports: renderer imports may initialize caches.
    module = load_renderer()
    config = module.Config(beat_offset=args.beat_offset, pair_frame=args.pair_frame,
                           cascade_frame=args.cascade_frame)
    out.mkdir(parents=True, exist_ok=True)
    for frame, path in zip(frames, paths):
        started = time.perf_counter()
        rgb = checked_rgb(module, frame, args, config)
        save_frame(module.FD.PI.look, path, rgb)
        print(json.dumps(dict(variant=args.variant, frame=frame, path=str(path),
                              scale=args.scale, ss=args.ss, config=asdict(config),
                              seconds=time.perf_counter() - started,
                              process_peak_rss_bytes=peak_rss_bytes())), flush=True)
        del rgb


if __name__ == '__main__':
    main()
