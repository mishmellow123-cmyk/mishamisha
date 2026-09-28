"""Portable candidate RUN-suite runner; one thread and a fresh private cache.

Adapted from run_suite_original.py after the recorded run. Tests, source pin,
thread settings and no-full-render assertions are unchanged.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tree', type=Path, default=Path(__file__).resolve().parents[4],
                    help='Repository root (defaults to the checkout containing this script).')
parser.add_argument('--out', type=Path, required=True,
                    help='New output directory; must not already exist.')
args = parser.parse_args()
TREE = args.tree.resolve()
HERE = args.out.resolve()
assert not HERE.exists(), 'Use a new output directory; preserve earlier receipts.'
HERE.mkdir(parents=True)
RUN = TREE / 'the-long-dawn/shots/run'
CACHE = HERE / 'suite-cache'
assert not CACHE.exists(), 'Preserve prior evidence; use a new private cache for a new cold run.'
CACHE.mkdir()
for key in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)
os.environ.pop('NUMBA_DISABLE_JIT', None)
sys.dont_write_bytecode = True
started = time.perf_counter()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source = {str(path.relative_to(TREE)): sha(path) for path in sorted(RUN.rglob('*.py'))}
assert source['the-long-dawn/shots/run/sdfppl.py'] == '9140e4ccd060db8895a2952ea5aab74a3c444bda12e29fd43ab8ec12909b2855'
receipt = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               pid=os.getpid(), tree=str(TREE),
               head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TREE, text=True).strip(),
               source_sha256=source, runner_sha256=sha(__file__), cache=str(CACHE),
               cache_initially_empty=True, status='running')

def save():
    (HERE / 'suite-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

save()
import cv2
import numba
cv2.setNumThreads(0)
numba.set_num_threads(1)
suite = unittest.defaultTestLoader.discover(str(RUN / 'tests'), pattern='test_*.py')
# look.py changes OpenCV settings on import; enforce sequential execution after
# discovery has loaded every test and its source dependencies.
cv2.setNumThreads(0)
numba.set_num_threads(1)
import sdfppl as SP
assert not SP.render.signatures
receipt.update(discovered_tests=suite.countTestCases(), python=sys.version,
               executable=sys.executable,
               versions={key: importlib.metadata.version(key) for key in ('numpy', 'numba', 'llvmlite', 'opencv-python-headless')},
               jit_disabled=numba.config.DISABLE_JIT, numba_threads=numba.get_num_threads(),
               opencv_requested_threads=0, opencv_reported_threads=cv2.getNumThreads(),
               environment={key: os.environ[key] for key in ('NUMBA_NUM_THREADS', 'NUMBA_CACHE_DIR', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
save()
print(json.dumps({key: receipt[key] for key in ('head', 'discovered_tests', 'jit_disabled', 'numba_threads', 'opencv_reported_threads', 'cache')}), flush=True)
test_start = time.perf_counter()
result = unittest.TextTestRunner(verbosity=2).run(suite)
receipt.update(status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
               failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
               unittest_elapsed_seconds=time.perf_counter() - test_start,
               total_elapsed_seconds=time.perf_counter() - started,
               sp_render_signatures=[str(signature) for signature in SP.render.signatures],
               noise_signatures=[str(signature) for signature in SP.gnoise3.signatures],
               source_unchanged=all(sha(TREE / path) == digest for path, digest in source.items()),
               final_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TREE, text=True).strip())
save()
assert receipt['source_unchanged'], 'Source changed during verification'
assert not SP.render.signatures, 'This bounded suite must not compile the full figure renderer'
print(json.dumps({key: receipt[key] for key in ('status', 'tests_run', 'failures', 'errors', 'skipped', 'unittest_elapsed_seconds', 'total_elapsed_seconds', 'source_unchanged', 'sp_render_signatures')}), flush=True)
raise SystemExit(not result.wasSuccessful())
