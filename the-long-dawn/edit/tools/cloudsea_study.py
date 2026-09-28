"""Full-frame, geometry-only A cloud-sea comparison; production shots remain off.

Imports renderers only after CLI validation and thread limits. A2 is rendered in
shot numbering and finished with its own FINISH; A13 returns its finished frame.
"""
import argparse
from contextlib import contextmanager
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'shots' / 'run'
RANGES = {'A2': (80, 559), 'A13': (3680, 3799)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shot', required=True, choices=tuple(RANGES))
    parser.add_argument('--frames', required=True, help='comma-separated CUT frame numbers')
    parser.add_argument('--relief', type=float, default=0.0, help='geometry strength in [0,1]; default off')
    parser.add_argument('--scale', type=float, default=0.5, help='output scale; default 960x402')
    parser.add_argument('--threads', type=int, default=1, choices=range(1, 5))
    parser.add_argument('--out', required=True, type=Path, help='absolute study output directory')
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='absolute the-long-dawn source tree; defaults to this checkout')
    args = parser.parse_args(argv)
    if not args.out.is_absolute():
        parser.error('--out must be absolute')
    if not args.source_root.is_absolute() or not (args.source_root / 'shots' / 'run' / 'world.py').is_file():
        parser.error('--source-root must be an absolute the-long-dawn source tree')
    if not math.isfinite(args.relief) or not 0.0 <= args.relief <= 1.0:
        parser.error('--relief must be finite and between 0 and 1')
    if not math.isfinite(args.scale) or not 0.01 <= args.scale <= 1.0:
        parser.error('--scale must be finite and between .01 and 1')
    try:
        args.frames = [int(value.strip()) for value in args.frames.split(',')]
    except ValueError:
        parser.error('--frames must contain comma-separated integer cut frames')
    if len(set(args.frames)) != len(args.frames):
        parser.error('--frames must not contain duplicates')
    low, high = RANGES[args.shot]
    if any(frame < low or frame > high for frame in args.frames):
        parser.error(f'{args.shot} cut frames must be between {low} and {high}')
    return args


@contextmanager
def relief_inputs(world, strength, march_depths=None):
    """Set runtime P[3] at the real Python entry points, never in a JIT global.

    Each call receives a copy: no caller-owned array or lighting parameter is
    changed. Internal compiled march/shadow calls read that same runtime value.
    """
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError('relief strength must be finite and in [0,1]')
    originals = {name: getattr(world, name) for name in ('march', 'shade', 'cloud_glow')}
    calls = {name: [] for name in originals}

    def wrap(name, index):
        original = originals[name]

        @functools.wraps(original)
        def call(*args, **kwargs):
            import numpy as np
            values = list(args)
            p = values[index] if len(values) > index else kwargs['P']
            p = np.array(p, dtype=np.float64, copy=True)
            if p.shape != (4,):
                raise ValueError(f'{name} expected four world parameters, got {p.shape}')
            p[3] = strength
            if len(values) > index:
                values[index] = p
            else:
                kwargs['P'] = p
            calls[name].append(p.tolist())
            result = original(*values, **kwargs)
            if name == 'march' and march_depths is not None:
                depth = values[9] if len(values) > 9 else kwargs['out_d']
                depth = np.ascontiguousarray(depth)
                march_depths.append(dict(shape=list(depth.shape), dtype=str(depth.dtype),
                    sha256=hashlib.sha256(memoryview(depth).cast('B')).hexdigest()))
            return result

        return call

    try:
        for name, index in (('march', 0), ('shade', 2), ('cloud_glow', 2)):
            setattr(world, name, wrap(name, index))
        yield calls
    finally:
        for name, original in originals.items():
            setattr(world, name, original)


def shot_adapter(shot):
    """Return the actual world module and a complete, finished-frame renderer."""
    sys.path.insert(0, str(RUN))
    if shot == 'A2':
        import falsedawn as module

        def render(frame, scale):
            # Both conditions get the skyline for this frame's actual camera.
            module._SKL = None
            return module.PI.look.finish(module.render(frame - module.CUT0, 'arc', scale, 1.5), **module.FINISH)

        return module.PI.WD, render, 'falsedawn.render(cut-80, arc, scale, ss=1.5) + FINISH'
    if shot == 'A13':
        table_path = RUN / 'reveal_a_fires.npy'
        fire_table = validate_a13_fires(table_path)
        import reveal_a as module
        if Path(module.FIRES_NPY).resolve() != table_path.resolve():
            raise RuntimeError('A13 module and selected source root disagree about the fire table')
        module._FIRES = fire_table
        world = module._orig_s1_world()['WD']

        def render(frame, scale):
            module.NA._SKL.clear()
            return module.render(frame, scale)

        return world, render, 'reveal_a.render(cut, scale); includes FirstBeacon foreground and finish'
    raise ValueError(f'unsupported shot: {shot}')


def validate_a13_fires(path):
    """Require the distant-fire input that A13's ordinary CLI preflights."""
    import numpy as np
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'A13 study requires the existing fire table: {path}')
    table = np.load(path, allow_pickle=False)
    if (not isinstance(table, np.ndarray) or table.ndim != 2 or table.shape[1] != 6
            or table.shape[0] == 0 or not np.issubdtype(table.dtype, np.number)
            or np.issubdtype(table.dtype, np.complexfloating) or not np.isfinite(table).all()):
        raise ValueError(f'A13 fire table must be a nonempty, finite, real N×6 array: {path}')
    return table


def save_png(path, srgb):
    """Deterministic 8-bit RGB quantization, no random dither; checked atomic write."""
    import cv2
    import numpy as np
    pixels = np.asarray(srgb)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or not np.isfinite(pixels).all():
        raise ValueError('finished frame must be finite HxWx3 RGB')
    quantized = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
    ok, encoded = cv2.imencode('.png', quantized[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f'PNG encoding failed for {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as fh:
            tmp = Path(fh.name)
            fh.write(encoded.tobytes())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return quantized


def source_record(world, shot=None):
    paths = [RUN / 'world.py', RUN / 'falsedawn.py', RUN / 'pipe.py', RUN / 'reveal_a.py',
             ROOT / 'shots' / 'hills' / 'beacon.py', Path(__file__), Path(world.__file__)]
    if shot == 'A13':
        paths.append(RUN / 'reveal_a_fires.npy')
    hashes = {}
    for path in paths:
        try:
            role = str(path.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            role = 'driver/cloudsea_study.py'
        hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True, check=True)
    return {'git_head': revision.stdout.strip(), 'source_sha256': hashes}


def write_manifest(path, data):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def main(argv=None):
    global ROOT, RUN
    args = parse_args(argv)
    ROOT = args.source_root.resolve()
    RUN = ROOT / 'shots' / 'run'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    os.environ['NUMBA_NUM_THREADS'] = str(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists() or any((args.out / f'f_{frame:05d}.png').exists() for frame in args.frames):
        raise FileExistsError('use a new output directory for each shot/condition')
    import cv2
    import numba
    import numpy as np
    cv2.setNumThreads(1)
    numba.set_num_threads(args.threads)
    world, render, recipe = shot_adapter(args.shot)
    cap = getattr(world, 'CLOUD_RELIEF_MAX', None)
    if cap is None and args.relief > 0.0:
        raise ValueError('selected source does not provide the cloud relief API; only --relief 0 is valid')
    manifest = dict(shot=args.shot, cut_frames=args.frames, relief=args.relief, relief_cap_m=cap,
                    scale=args.scale, threads=args.threads, full_frame=True, recipe=recipe,
                    quantization='round-to-nearest RGB uint8; no dither',
                    skyline_cache='cleared before each frame in both conditions',
                    status='running', frames=[], **source_record(world, args.shot))
    write_manifest(manifest_path, manifest)
    try:
        for frame in args.frames:
            start = time.perf_counter()
            march_depths = []
            with relief_inputs(world, args.relief, march_depths) as calls:
                result = render(frame, args.scale)
            if not calls['march'] or not calls['shade']:
                raise RuntimeError('study did not intercept the actual marcher and shader')
            expected = (round(804 * args.scale), round(1920 * args.scale), 3)
            if result.shape != expected:
                raise ValueError(f'frame shape {result.shape} differs from requested {expected}')
            path = args.out / f'f_{frame:05d}.png'
            pixels = save_png(path, result)
            elapsed = time.perf_counter() - start
            manifest['frames'].append(dict(cut_frame=frame, path=str(path), seconds=elapsed,
                shape=list(result.shape), world_inputs=calls, march_depths=march_depths,
                finished_rgb_sha256=hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest(),
                quantized_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                png_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            write_manifest(manifest_path, manifest)
            print(f'{args.shot} cut {frame}, relief={args.relief:g}: {elapsed:.1f}s -> {path}', flush=True)
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        write_manifest(manifest_path, manifest)
        raise
    manifest['status'] = 'complete'
    write_manifest(manifest_path, manifest)


if __name__ == '__main__':
    main()
