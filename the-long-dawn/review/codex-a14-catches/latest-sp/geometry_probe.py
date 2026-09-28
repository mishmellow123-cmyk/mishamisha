"""Compare actual watcher scenes across two sdfppl revisions; never render pixels.

Example: NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B probe_sp_compatibility.py --repo ../night-a14 --output sp-compatibility
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True

import cv2
import numpy as np
import numba


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def array_info(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=digest(np.ascontiguousarray(array).tobytes()), finite=bool(np.isfinite(array).all()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old-ref', default='7ab77932ebe7fbd143638980ef706ded90bdc2ab')
    parser.add_argument('--new-ref', default='742bcae')
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = repo / 'the-long-dawn/shots/run'
    relative = 'the-long-dawn/shots/run/sdfppl.py'
    sources = {}
    for label, ref in [('old', args.old_ref), ('new', args.new_ref)]:
        sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', ref], text=True).strip()
        data = subprocess.check_output(['git', '-C', str(repo), 'show', f'{sha}:{relative}'])
        path = output / f'sdfppl_{label}.py'
        path.write_bytes(data)
        sources[label] = dict(commit=sha, path=str(path), sha256=digest(data))
    # Actual shot modules, committed input tables and actual terrain evaluation.
    # No call to render, draw_figures, or either SDF pixel kernel is made.
    sys.path.insert(0, str(run))
    print('Importing actual watcher/terrain modules (no pixel renderer).', flush=True)
    import watchers_a as WA
    BR = WA.br()
    cv2.setNumThreads(0)  # look.py sets this during import; override after it.
    numba.set_num_threads(1)
    modules = {label: load_module(f'sp_compat_{label}', info['path']) for label, info in sources.items()}
    original = WA.SP
    original_overrides = {key: os.environ.pop(key, None) for key in ('W15_P7', 'W15_CAM', 'W15_OWN_CAM', 'A14_CAM')}
    frames = []
    try:
        for frame in (4202, 4208, 4239, 4240, 4399):
            arrays = {}
            for label, module in modules.items():
                WA.SP = module
                arrays[label] = WA.figures_scene(frame).arrays()
                np.savez(output / f'geometry-{label}-{frame}.npz', primitives=arrays[label][0], objects=arrays[label][1])
            item = dict(frame=frame, arrays={})
            for index, name in enumerate(('primitives', 'objects')):
                old, new = arrays['old'][index], arrays['new'][index]
                same_shape = old.shape == new.shape
                item['arrays'][name] = dict(old=array_info(old), new=array_info(new),
                    exact_equal=bool(np.array_equal(old, new)),
                    differing_values=int(np.count_nonzero(old != new)) if same_shape else None,
                    max_abs_difference=float(np.max(np.abs(old - new))) if same_shape and old.size else None)
            item['scene_rows'] = {}
            for label, (P, O) in arrays.items():
                def counts(column):
                    values, ns = np.unique(column, return_counts=True)
                    return {str(float(v)): int(n) for v, n in zip(values, ns)}
                boxes = P[P[:, 0] == 1]
                item['scene_rows'][label] = dict(
                    primitive_types=counts(P[:, 0]), materials=counts(P[:, 9]),
                    box_rough_amplitudes=counts(boxes[:, 14]),
                    box_rough_frequencies=counts(boxes[:, 15]),
                    object_emission=counts(O[:, 9]),
                    object_horn_flags=counts(O[:, 14]),
                    object_flame_height=counts(O[:, 15]))
            frames.append(item)
            print(f'frame {frame}: ' + ', '.join(f'{name} exact={values["exact_equal"]}'
                  for name, values in item['arrays'].items()), flush=True)
    finally:
        WA.SP = original
        for key, value in original_overrides.items():
            if value is not None:
                os.environ[key] = value
    # Legacy v2 reach bug: no renderer, just constructing a tiny primitive scene.
    legacy = []
    for label, module in modules.items():
        for cloth, reaching, carrying in ((False, True, False), (False, False, False),
                                         (True, True, False), (False, True, True)):
            scene = module.Scene()
            controls = dict(cloth=cloth, reach=np.array([0., .5, .5]) if reaching else None,
                carry=(np.array([-.2, .8, .3]), np.array([.2, .8, .3])) if carrying else None)
            result = dict(version=label, cloth=cloth, reach=reaching, carry=carrying)
            try:
                module.traveller(scene, np.array([0., .93, 0.]), np.array([0., 0., 1.]),
                                 np.array([-.13, .07, 0.]), np.array([.13, .07, 0.]), (.03, .026, .022), **controls)
                result.update(status='ok', primitives=array_info(scene.arrays()[0]))
            except Exception as exc:
                result.update(status='error', error_type=type(exc).__name__, message=str(exc),
                              traceback=traceback.format_exc())
            legacy.append(result)
    source_files = ['watchers_a.py', 'beaconrun_a.py', 'world.py', 'falsedawn.py', 'nighta.py',
                    'hearth_a.py', 'fire_near_a.py', 'beaconrun_a_chain.npy', 'watchers_a_extra.npy',
                    'watchers_a_figs.npy']
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), repo=str(repo),
        worktree_head=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        probe_sha256=digest(Path(__file__).read_bytes()), sdfppl_sources=sources,
        unchanged_scene_sources_sha256={name: digest((run / name).read_bytes()) for name in source_files},
        numba_threads=numba.get_num_threads(), opencv_sequential_requested=True,
        opencv_threads_reported=cv2.getNumThreads(), jit_disabled=bool(numba.config.DISABLE_JIT),
        scene_inputs=dict(chain=array_info(WA.chain()), extra_fires=array_info(WA.table()[0]),
                          extra_figures=array_info(WA.table()[1]), terrain=array_info(BR.CR)),
        scope='Actual WA.figures_scene with actual ground and tables; WA.SP module swapped. No pixel render.',
        frames=frames, all_arrays_exact=all(a['exact_equal'] for f in frames for a in f['arrays'].values()),
        legacy_reach_probe=legacy,
        limits='Geometry/object arrays only. Does not execute SP.render or establish final-pixel shader parity.')
    path = output / 'receipt.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Receipt: {path}', flush=True)
    print('Legacy results: ' + str([(x['version'], x['cloth'], x['reach'], x['carry'], x['status']) for x in legacy]), flush=True)
    if not report['all_arrays_exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
