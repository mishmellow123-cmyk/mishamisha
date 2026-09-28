"""Compile only the latest primitive dispatcher, using a saved actual watcher box.

No scene rebuild or pixel renderer. Run with the isolated validation Python.
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
import time

for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '0'
sys.dont_write_bytecode = True

import cv2
import numba
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    here = Path(__file__).resolve().parent
    run = repo / 'the-long-dawn/shots/run'
    source = run / 'sdfppl.py'
    pinned = '70d102b148ff225947f252f3183aca9503a6d4fd'
    raw = source.read_bytes()
    committed = subprocess.check_output(['git', '-C', str(repo), 'show',
                                       pinned + ':the-long-dawn/shots/run/sdfppl.py'])
    assert raw == committed, 'Current shader differs from the audited production source'
    sys.path.insert(0, str(repo / 'the-long-dawn/shots/montage'))
    spec = importlib.util.spec_from_file_location('latest_sp_primitive_smoke', source)
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    cv2.setNumThreads(0)
    numba.set_num_threads(1)
    assert not numba.config.DISABLE_JIT
    array_path = here / 'geometry-extracts/geometry-new-4208.npz'
    with np.load(array_path) as arrays:
        primitives = arrays['primitives']
    box_index = int(np.flatnonzero(primitives[:, 0] == 1)[0])
    box = primitives[box_index].copy()
    assert box[14] == 0 and box[15] == 0
    rough = box.copy()
    rough[14], rough[15] = 0.03, 7.0
    offsets = np.array([[0., 0., 0.], [.05, .08, -.03], [.15, .12, .10], [-.2, -.1, .07]])
    rows = []
    started = time.monotonic()
    print('Compiling latest SP._prim and evaluating four saved-box sample points.', flush=True)
    for point in box[1:4] + offsets:
        px, py, pz = map(float, point)
        ordinary = float(sp._prim(px, py, pz, box))
        sd_box = float(sp._sd_box(px, py, pz, box))
        displaced = float(sp._prim(px, py, pz, rough))
        fq = float(rough[15])
        noise = (sp.gnoise3(px*fq, py*fq, pz*fq, 41)
                 + .5*sp.gnoise3(px*fq*2.3+1.7, py*fq*2.3, pz*fq*2.3-.9, 42))
        expected = .5*(sd_box + float(rough[14])*noise)
        assert np.isfinite([ordinary, sd_box, displaced, expected]).all()
        assert ordinary == sd_box
        assert np.isclose(displaced, expected, rtol=1e-12, atol=1e-12)
        rows.append(dict(point=point.tolist(), default=ordinary, sd_box=sd_box,
                         rough=displaced, rough_formula=float(expected),
                         rough_differs_from_default=displaced != ordinary))
    assert any(row['rough_differs_from_default'] for row in rows)
    assert sp._prim.nopython_signatures and sp._sd_box.nopython_signatures
    assert not sp.render.signatures, 'Full pixel renderer must remain uncompiled'
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Compiled primitive smoke only; no SP.render or pixel comparison.',
                  pinned_shader_commit=pinned,
                  worktree_head=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                  source_path=str(source), source_sha256=sha(raw),
                  probe_sha256=sha(Path(__file__).read_bytes()),
                  saved_geometry_path=str(array_path), saved_geometry_sha256=sha(array_path.read_bytes()),
                  saved_geometry_frame=4208, primitive_index=box_index,
                  original_primitive=box.tolist(), copied_rough_primitive=rough.tolist(),
                  numba_version=numba.__version__, numpy_version=np.__version__,
                  jit_disabled=bool(numba.config.DISABLE_JIT), numba_threads=numba.get_num_threads(),
                  opencv_sequential_requested=True, opencv_threads_reported=cv2.getNumThreads(),
                  elapsed_seconds=time.monotonic()-started,
                  primitive_nopython_signatures=[str(s) for s in sp._prim.nopython_signatures],
                  render_compiled_signatures=[str(s) for s in sp.render.signatures],
                  all_default_equal_sd_box=True, all_rough_match_formula=True, all_finite=True,
                  samples=rows,
                  limits='Four points on one actual watcher box; copied positive roughness exercises typing and formula. No full-render typing, pixel equivalence, silhouette quality, or exhaustive primitive correctness claim.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'PASS: {args.out}; elapsed {report["elapsed_seconds"]:.3f}s', flush=True)


if __name__ == '__main__':
    main()
