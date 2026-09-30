"""Opt-in cut-D shot renderer; no routing changes to accepted renders.

LD_OPEN_RING=1 python shots/embers/render_d.py --frames 2960-3199 --shot brink --out renders/embers_D_brink
"""
import argparse
import importlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SHOTS = {'inscription': (1680, 2079, 'd_inscription'), 'brink': (2960, 3199, 'd_vision'),
         'vision': (3200, 3439, 'd_vision'), 'gap': (3440, 3519, 'd_vision'),
         'crowns': (4240, 4559, 'd_crowns')}


def parse_frames(value):
    frames = []
    for part in value.split(','):
        span, _, stride = part.partition(':')
        step = int(stride or 1)
        if step <= 0:
            raise ValueError('Frame stride must be positive')
        first, sep, last = span.partition('-')
        a, b = int(first), int(last) if sep else int(first)
        if b < a:
            raise ValueError('Reversed frame range')
        frames.extend(range(a, b + 1, step))
    if not frames or len(set(frames)) != len(frames):
        raise ValueError('Empty or duplicate frames')
    return frames


def output_path(shot, out):
    """External probe dirs are allowed; local farm outputs must use new names.

    Refuse the workstation's read-only renders symlink, including callers that
    supplied its resolved target. A farm clone has an ordinary renders folder.
    """
    path = Path(out).absolute()
    renders = ROOT / 'renders'
    resolved = path.resolve()
    if renders.is_symlink() and (resolved == renders.resolve() or renders.resolve() in resolved.parents):
        raise ValueError('The shared renders symlink is read-only')
    if path == renders or renders in path.parents:
        expected = renders / f'embers_D_{shot}'
        if path != expected or path.is_symlink():
            raise ValueError('Only the matching new cut-D farm directory is allowed')
    elif ROOT == resolved or ROOT in resolved.parents:
        raise ValueError('Local probes must live outside the repository')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', nargs='?')
    parser.add_argument('--frames', dest='named_frames', help='Farm-rewritable frame list')
    parser.add_argument('--shot', required=True, choices=SHOTS)
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--out', required=True)
    parser.add_argument('--measure', help='Append per-frame seconds and process peak RSS JSONL')
    args = parser.parse_args(argv)
    if bool(args.frames) == bool(args.named_frames):
        parser.error('Supply exactly one positional or --frames argument')
    frames = parse_frames(args.frames or args.named_frames)
    lo, hi, name = SHOTS[args.shot]
    if any(not lo <= f <= hi for f in frames):
        parser.error(f'{args.shot} requires D{lo}–{hi}')
    if not 0 < args.scale <= 1:
        parser.error('scale must be in (0, 1]')
    out = output_path(args.shot, args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = importlib.import_module(name)
    import look
    for frame in frames:
        started = time.perf_counter()
        image = renderer.render_frame(frame, scale=args.scale, save=False, verbose=False)
        seconds = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin':
            peak *= 1024
        look.save_png(look.frame_path(str(out), frame), image)
        measurement = dict(shot=args.shot, frame=frame, scale=args.scale,
                           seconds=seconds, peak_rss_bytes=peak,
                           shape=list(image.shape), first_frame_in_process=(frame == frames[0]))
        print(json.dumps(measurement), flush=True)
        if args.measure:
            with open(args.measure, 'a') as stream:
                stream.write(json.dumps(measurement) + '\n')


if __name__ == '__main__':
    main()
