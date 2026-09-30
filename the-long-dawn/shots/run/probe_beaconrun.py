"""Single-frame, half-size candidate evidence; run through tonight's onepy lock.

Writes only the explicit report directory. Delivered renders are read-only.
Warm = R>110, R>1.25G, G>1.2B on finished uint8 RGB; area means pixels,
not a physical flame measurement. Components use eight-connected neighbours.
"""
import argparse
import faulthandler
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

for key in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parents[2] / 'outputs' / 'claude-owner-night' / 'codex'


def warm_components(image):
    import cv2
    import numpy as np
    a = image.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    warm = ((r > 110) & (r > 1.25 * g) & (g > 1.2 * b)).astype(np.uint8)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(warm, 8)
    return warm, [dict(area=int(s[4]), box=s[:4].tolist(), centre=c.tolist())
                  for s, c in zip(stats[1:], centres[1:]) if s[4] >= 3]


def write_image(path, image):
    import cv2
    temporary = path.with_name(path.stem + '.tmp.png')
    if not cv2.imwrite(str(temporary), image[..., ::-1]):
        raise RuntimeError(f'Failed to save {path}')
    temporary.replace(path)


def finished_bytes(pre, finisher, kind, option, frame):
    """Match assemble.Ctx.picture/frame's float film input and byte output."""
    import numpy as np
    picture = finisher(pre.astype(np.float32) / 255.,
                       dict(stem=f'cand_{kind}_{option}'), 'A', frame)
    return (np.clip(picture, 0, 1) * 255 + .5).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--accepted-only', action='store_true')
    parser.add_argument('--verify-existing', action='store_true')
    parser.add_argument('--low-memory-jit', action='store_true',
                        help='compile only the figure shader without optimization/parallel lowering; same scene code')
    args = parser.parse_args()
    print('probe process started', flush=True)
    faulthandler.dump_traceback_later(120, repeat=True)
    import cv2
    import numpy as np
    cv2.setNumThreads(0)
    out = OUT.resolve()
    if not out.is_dir() or (ROOT / 'renders').resolve() in out.parents:
        raise ValueError('Evidence must be outside delivered renders')
    if args.verify_existing:
        from PIL import Image
        for path in sorted(out.glob('beaconrun_*.png')):
            with Image.open(path) as im:
                im.verify()
            print('verified', path.name, flush=True)
        return
    sys.path.insert(0, str(ROOT / 'edit'))
    import assemble as AS
    AS._init('A', None, .5, True, True)
    from stage import Finisher
    fin = Finisher()
    print('assembler ready', flush=True)
    for f in (int(x) for x in args.frames.split(',')):
        start = time.monotonic()
        accepted = AS._CTX.frame(f)
        write_image(out / f'beaconrun_accepted_{f}.png', accepted)
        _, components = warm_components(accepted)
        record = dict(frame=f, scale=.5, accepted_components=components)
        del accepted
        gc.collect()
        if not args.accepted_only:
            import beaconrun_candidates as X
            B, W = X.modules()
            if args.low_memory_jit and not hasattr(W.SP, '_probe_original_render'):
                import numba
                W.SP._probe_original_render = W.SP.render
                # Recompile the IDENTICAL Python body, without changing its
                # geometry/shading/sampling. Keep this alternative out of the
                # accepted JIT cache and leave terrain kernels optimized.
                simple_sdf = numba.njit(parallel=False, fastmath=True, cache=False)(W.SP.render.py_func)

                def low_memory_figures(*a):
                    previous = os.environ.get('NUMBA_OPT')
                    os.environ['NUMBA_OPT'] = '0'
                    numba.core.config.reload_config()
                    print('figure shader: unoptimized sequential compilation', flush=True)
                    try:
                        return simple_sdf(*a)
                    finally:
                        if previous is None:
                            os.environ.pop('NUMBA_OPT', None)
                        else:
                            os.environ['NUMBA_OPT'] = previous
                        numba.core.config.reload_config()
                W.SP.render = low_memory_figures
            kind = 'beaconrun' if f < 4240 else 'watchers'
            print('render', f, flush=True)
            faulthandler.dump_traceback_later(120, repeat=True)
            try:
                rgb = X.render(f, kind=kind, candidate=X.OPTION, scale=.5)
            finally:
                faulthandler.cancel_dump_traceback_later()
            pre = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            del rgb
            # The master reads q95 4:4:4 JPEGs; emulate that quantization before
            # the identical film finish. Local rasterization remains half-size.
            ok, encoded = cv2.imencode('.jpg', pre[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95,
                                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            pre = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]
            candidate = finished_bytes(pre, fin, kind, X.OPTION, f)
            write_image(out / f'beaconrun_candidate_{f}.png', candidate)
            warm, components = warm_components(candidate)
            record['candidate_components'] = components
            with X.linked_scene() as chain:
                camera = B.camera(f, 960, 402) if f < 4240 else W.camera(f, 960, 402)
                x, y, z = camera.project(chain[:, :3])
                record['candidate_bases'] = np.c_[x, y, z].tolist()
            camera = B.camera(f, 960, 402) if f < 4240 else W.camera(f, 960, 402)
            x, y, z = camera.project(B.CHAIN[:, :3])
            record['accepted_bases'] = np.c_[x, y, z].tolist()
            del candidate, pre, warm, encoded
            gc.collect()
        record['seconds'] = time.monotonic() - start
        record['numba_opt'] = os.environ.get('NUMBA_OPT', '3')
        record['figure_jit'] = 'sequential-opt0-uncached' if args.low_memory_jit else 'accepted'
        record['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        path = out / f'beaconrun_probe_{f}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(path)
        print(json.dumps(dict(frame=f, seconds=record['seconds'],
                              peak_rss_bytes=record['peak_rss_bytes'],
                              largest_candidate_components=sorted(
                                  [c['area'] for c in record.get('candidate_components', [])], reverse=True)[:8])),
              flush=True)


if __name__ == '__main__':
    main()
