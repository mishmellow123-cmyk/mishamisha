"""Compare actual renderer depth and deterministic finished RGB at alternative march ceilings.

Run each shot in a fresh process; the legacy render modules share import names and global caches.
No scene file is changed. All outputs require an explicit absolute directory. A13 and beacon test
their actual background world layers (before foreground compositing), as labelled in the report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def wait_for_load(limit):
    while os.getloadavg()[0] >= limit:
        print(f'PAUSED: one-minute load {os.getloadavg()[0]:.2f} >= {limit:g}', flush=True)
        time.sleep(20)


def adapter(shot, scale):
    sys.path.insert(0, str(ROOT / 'shots/run'))
    import numpy as np
    if shot in ('A2', 'A11', 'A15', 'A18', 'bluehour'):
        import world as WD
        import pipe as PI
        if shot == 'A2':
            import falsedawn as M
            render = lambda f: PI.look.finish(M.render(f - M.CUT0, 'arc', scale, 1.5), **M.FINISH)
            reset = lambda: setattr(M, '_SKL', None)
        elif shot == 'A11':
            import stars_a as M
            import falsedawn as FD
            render = lambda f: PI.look.finish(M.render(f, scale, 1.5), **M.FINISH)
            def reset():
                M._ST.clear()
                FD._SKL = None
        elif shot == 'A15':
            import watchers_a as M
            render = lambda f: M.finish(M.render(f, scale=scale, ss=1.5))
            reset = M.NA._SKL.clear
        elif shot == 'A18':
            import crossing as M
            render = lambda f: PI.look.finish(M.render(f - M.CUT0, scale, 1.5, variant='main', trail=True), **M.FINISH)
            # Prime completed terrain before planting figures in either condition.
            reset = M.wfs
        else:
            import bluehour as M
            render = lambda f: PI.look.finish(M.render(f - M.CUT0, scale, 1.5), **M.FINISH)
            reset = lambda: None
        return WD, render, reset, 'finished full frame', Path(M.__file__)
    if shot in ('A13', 'beacon'):
        import reveal_a as RA
        if not Path(RA.FIRES_NPY).is_file():
            raise FileNotFoundError(RA.FIRES_NPY)
        BK = RA.BK
        WD = RA._orig_s1_world()['WD']
        if shot == 'A13':
            def render(f):
                src = f - RA.SRC_OFF
                cam = RA.camera(src, scale)[0]
                return RA.look.finish(RA.world_layer(cam, src, src / 24.0, 1.0))
            source = Path(RA.__file__)
        else:
            def render(f):
                cam = BK.camera(f, scale)[0]
                return RA.look.finish(BK.world_layer(cam, f, f / 24.0, 1.0))
            source = Path(BK.__file__)
        return WD, render, lambda: None, 'world layer; reveal gain 1; diagnostic finish; foreground excluded', source
    if shot == 'bh_probe':
        import bh_probe as M
        M.X.wfs()                         # same initialization as its CLI
        def render(f):
            # Labels report depth and therefore would contaminate the RGB metric.
            original = M.cv2.putText
            try:
                M.cv2.putText = lambda image, *a, **kw: image
                return M.view(f, cloud=f == 3)
            finally:
                M.cv2.putText = original
        return M.WD, render, lambda: None, 'full-resolution probe; depth labels suppressed', Path(M.__file__)
    raise ValueError(shot)


def bbox(mask):
    import numpy as np
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None


def compare_depth(a, b):
    import numpy as np
    old, new = a['depth'], b['depth']
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('nonfinite depth buffer')
    if old.shape != new.shape or not np.array_equal(a['camera'], b['camera']):
        raise ValueError('camera/depth shape changed between conditions')
    changed = old != new
    both = (old < 1e29) & (new < 1e29)
    lost_sky = (old >= 1e29) & (new < 1e29)
    return dict(shape=list(old.shape), pixels=int(old.size), changed=int(changed.sum()),
                sky_to_terrain=int(lost_sky.sum()), terrain_to_sky=int(((old < 1e29) & (new >= 1e29)).sum()),
                changed_bbox_source=bbox(changed), recovered_bbox_source=bbox(lost_sky),
                max_finite_depth_delta_m=float(np.abs(old[both] - new[both]).max()) if both.any() else None,
                camera=a['camera'].tolist(),
                caller_hmax=a['caller_hmax'], baseline_hmax=a['effective_hmax'], diagnostic_hmax=b['effective_hmax'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shot', choices=['A2', 'A11', 'A15', 'A18', 'A13', 'bluehour', 'beacon', 'bh_probe'], required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--threads', type=int, choices=range(1, 5), default=2)
    ap.add_argument('--max-load', type=float, default=8.0)
    args = ap.parse_args()
    if not args.out.is_absolute():
        ap.error('--out must be absolute')
    if not 0 < args.scale <= 1 or not args.max_load > 0:
        ap.error('--scale must be in (0,1] and --max-load must be positive')
    frames = [int(f) for f in args.frames.split(',')]
    if args.shot == 'bh_probe' and any(f not in (2, 3) for f in frames):
        ap.error('bh_probe supports views 2/3 only; view 1 is a top-down map without a marcher')
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = str(args.threads if name == 'NUMBA_NUM_THREADS' else 1)
    wait_for_load(args.max_load)
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
    print(f'Loading {args.shot} adapter (first use can compile render kernels)', flush=True)
    WD, render, reset, scope, source = adapter(args.shot, args.scale)
    march = WD.march
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(shot=args.shot, scope=scope, scale=args.scale, threads=args.threads,
                  head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  lookdev_overrides={k: os.environ.get(k) for k in ('W15_CAM', 'W15_P7', 'W15_OWN_CAM', 'A14_CAM', 'NIGHT_CLOUD')},
                  rgb_metric='max absolute finished RGB channel difference; uint8 uses round(255*clip(RGB,0,1)), no dither',
                  coordinates='bounding boxes inclusive, [xmin,ymin,xmax,ymax]; source and output spaces labelled separately',
                  coverage='all source rows and columns in every actual march call', frames=[])
    report['initialization'] = 'A18 terrain completed before either condition; A2/A11 skyline reset per condition; isolated frames, not farm cache replay'
    for frame in frames:
        conditions = [('700', 700.0), ('3000', 3000.0)]
        if args.shot in ('A2', 'A11', 'A15'):
            conditions.append(('current', None))
        results = {}
        for label, override in conditions:
            wait_for_load(args.max_load)
            print(f'{args.shot} {frame} {label}: initializing condition', flush=True)
            reset()
            calls = []
            def wrapped(*pos, **kw):
                if kw:
                    raise ValueError('unexpected keyword march arguments; review adapter')
                call = list(pos)
                caller = float(call[7])
                if override is not None:
                    call[7] = override
                march(*call)
                calls.append(dict(depth=call[9].copy(), camera=call[2].copy(), caller_hmax=caller,
                                  effective_hmax=float(call[7])))
                print(f'  march pass {len(calls)} captured: {call[9].shape}', flush=True)
            WD.march = wrapped
            began = time.monotonic()
            try:
                np.random.seed(9272026)
                picture = np.asarray(render(frame))
            finally:
                WD.march = march
            if not calls or not np.isfinite(picture).all():
                raise RuntimeError('missing march coverage or nonfinite rendered output')
            quantized = np.rint(np.clip(picture, 0, 1) * 255).astype(np.uint8)
            target = args.out / f'{args.shot}_{frame}_{label}.png'
            if not cv2.imwrite(str(target), quantized[..., ::-1]):
                raise IOError(target)
            results[label] = (picture.copy(), quantized, calls)
            print(f'{args.shot} {frame} {label}: {time.monotonic()-began:.1f}s, {len(calls)} march passes', flush=True)
        diagnostic, qdiag, dcalls = results['3000']
        item = dict(frame=frame, width=int(diagnostic.shape[1]), height=int(diagnostic.shape[0]), comparisons={})
        for label in ('700', 'current'):
            if label not in results:
                continue
            picture, quantized, calls = results[label]
            if len(calls) != len(dcalls):
                raise RuntimeError('march pass count changed')
            diff = np.abs(picture.astype(np.float64) - diagnostic)
            qdiff = np.abs(quantized.astype(np.int16) - qdiag.astype(np.int16))
            item['comparisons'][label] = dict(max_rgb_float=float(diff.max()), max_rgb_uint8=int(qdiff.max()),
                    changed_rgb_pixels=int(np.any(qdiff > 0, axis=2).sum()), rgb_bbox_output=bbox(np.any(qdiff > 0, axis=2)),
                    depth=[compare_depth(a, b) for a, b in zip(calls, dcalls)])
        report['frames'].append(item)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
