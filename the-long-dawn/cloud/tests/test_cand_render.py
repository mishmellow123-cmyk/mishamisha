"""Candidate adapter contracts and opt-in, supervised comparisons with ADOPTION.md.

The four real-render tests are disabled unless LD_CAND_RENDER_SLOW=1. Each runs
in a fresh process, against copied source/assets in a temporary project, and
compares arrays with the literal ld_candidate recipe extracted from the document.
No accepted render directory may be written, including by imported renderers.
"""
import ast
import contextlib
import gc
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / 'cloud' / 'cand_render.py'
MEMORY_LIMIT = 1_400_000_000
SCALE = .05
CASES = {'trap': 2479, 'cold': 3816, 'deep': 4360, 'pen': 5560}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def recipe_tree():
    text = (ROOT / 'ADOPTION.md').read_text()
    source = text.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    return ast.parse(source, filename='ADOPTION.md:ld_candidate')


def recipe_literal(name):
    assignment = next(node for node in recipe_tree().body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == name
                              for t in node.targets))
    return ast.literal_eval(assignment.value)


def recipe_options():
    return recipe_literal('options')


def refuse_real_render_writes():
    forbidden = (ROOT / 'renders').resolve()

    def under_renders(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        path = Path(os.fsdecode(value)).resolve()
        return path == forbidden or forbidden in path.parents

    def audit(event, args):
        paths = ()
        if event == 'open':
            path, mode, flags = args
            writable = (isinstance(mode, str) and any(c in mode for c in 'wax+'))
            writable |= bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                                       os.O_TRUNC | os.O_APPEND))
            paths = (path,) if writable else ()
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod',
                       'os.chown', 'os.truncate', 'os.utime'):
            paths = args[:1]
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            paths = args[:2]
        if any(under_renders(path) for path in paths):
            raise RuntimeError(f'test refused write to real renders: {event} {paths}')

    sys.addaudithook(audit)


def copy_fixture(destination):
    """Copy renderer Python and tracked assets, never existing renders/caches."""
    for relative in ('shots/embers', 'shots/map', 'lib'):
        for source in (ROOT / relative).rglob('*.py'):
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    result = subprocess.run(
        ['git', 'ls-files', '-z', '--', 'the-long-dawn/assets/ring',
         'the-long-dawn/assets/fonts'], cwd=ROOT.parent, check=True,
        stdout=subprocess.PIPE)
    assets = [ROOT.parent / os.fsdecode(p) for p in result.stdout.split(b'\0') if p]
    if not assets:
        raise RuntimeError('No tracked ring/font assets found for isolated fixture')
    for source in assets:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak * 1024


def start_memory_watchdog(kind):
    """Poll our own high-water RSS; this also works when sandboxed ps is denied.

    Native work that holds the GIL can delay this Python thread. The parent's
    final high-water assertion also rejects a peak missed before child exit.
    """
    def watch():
        while True:
            peak = peak_rss_bytes()
            if peak > MEMORY_LIMIT:
                print(json.dumps(dict(kind=kind, memory_limit_exceeded=True,
                                      process_peak_rss_bytes=peak,
                                      memory_limit_bytes=MEMORY_LIMIT)), flush=True)
                os._exit(86)
            time.sleep(.05)

    threading.Thread(target=watch, daemon=True, name='candidate-memory-watchdog').start()


def reference_child(kind, fixture):
    start_memory_watchdog(kind)
    refuse_real_render_writes()
    os.chdir(fixture)
    family = 'embers' if kind in ('trap', 'cold') else 'map'
    sys.path.insert(0, str(fixture / 'shots' / family))
    sys.path.append(str(fixture / 'lib'))
    import numpy as np
    import cv2
    import numba

    names = {'threads', 'checked', 'book_output', 'build'}
    functions = [node for node in recipe_tree().body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise RuntimeError('ADOPTION reference functions changed; review the extraction')
    namespace = dict(root=fixture, kind=kind, scale=SCALE,
                     W=int(1920 * SCALE), H=int(804 * SCALE),
                     page_kind=kind in ('deep', 'pen'), np=np, cv2=cv2,
                     numba=numba, importlib=importlib, hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 'ADOPTION.md:literal-reference', 'exec'), namespace)
    option, frame = recipe_options()[kind], CASES[kind]
    original = namespace['build']('candidate', option)
    expected = namespace['checked'](original, frame)
    # Retain only the tiny result arrays while releasing the first scene.
    del original
    gc.collect()
    wrapper = load_module(WRAPPER, 'cand_render_reference_target')
    wrapper.ROOT = fixture
    actual_render = wrapper.build(kind, 'candidate', option, scale=SCALE)
    actual = wrapper.checked(actual_render, frame, scale=SCALE)
    if actual.keys() != expected.keys():
        raise AssertionError(f'{kind}: keys differ: {actual.keys()} != {expected.keys()}')
    proofs = {}
    for key in expected:
        if actual[key].dtype != expected[key].dtype:
            raise AssertionError(f'{kind}/{key}: dtype differs')
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f'{kind}/{key}')
        proofs[key] = dict(shape=list(actual[key].shape), dtype=actual[key].dtype.str,
                           sha256=hashlib.sha256(actual[key].tobytes()).hexdigest())
    print(json.dumps(dict(kind=kind, option=option, frame=frame, scale=SCALE,
                          literal_recipe_equal=True, arrays=proofs)), flush=True)


def run_supervised(kind):
    with tempfile.TemporaryDirectory(prefix=f'cand-test-{kind}-') as temporary:
        base = Path(temporary)
        fixture = base / 'project'
        copy_fixture(fixture)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                   NUMBA_CACHE_DIR=str(base / 'numba-cache'), NUMBA_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1')
        with tempfile.TemporaryFile(mode='w+b') as output:
            child = subprocess.Popen(
                [sys.executable, '-B', str(Path(__file__).resolve()),
                 '--reference-child', kind, str(fixture)],
                cwd=fixture, env=env, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True)
            failure = None
            deadline = time.monotonic() + float(os.environ.get('LD_CAND_TEST_TIMEOUT', '900'))
            try:
                while child.poll() is None:
                    if time.monotonic() > deadline:
                        failure = f'{kind}: reference comparison timed out'
                        break
                    time.sleep(.1)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            output.seek(0)
            text = output.read().decode('utf-8', 'replace')
        print(json.dumps(dict(kind=kind, memory_monitor='child ru_maxrss watchdog',
                              memory_limit_bytes=MEMORY_LIMIT)), flush=True)
        print(text, end='', flush=True)
        if failure or child.returncode:
            raise AssertionError(failure or f'{kind}: child exited {child.returncode}\n{text[-8000:]}')
        reports = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
        if not any(row.get('literal_recipe_equal') for row in reports):
            raise AssertionError(f'{kind}: child emitted no comparison result')
        peak_report = next((row for row in reports if 'process_peak_rss_bytes' in row), None)
        if peak_report is None or peak_report['process_peak_rss_bytes'] > MEMORY_LIMIT:
            raise AssertionError(f'{kind}: missing or excessive process peak RSS: {peak_report}')


class CandidateAdapterTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = load_module(WRAPPER, 'cand_render_contract_target')

    def test_non_native_cli_render_rejected_before_build(self):
        with mock.patch.object(self.wrapper, 'build') as build:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    status = self.wrapper.main([
                        'trap', 'front_smoke_near', '--range', '2479-2479',
                        '--out', 'renders/cand_trap_front_smoke_near', '--scale', '.05'])
                except (SystemExit, ValueError, RuntimeError) as error:
                    status = error.code if isinstance(error, SystemExit) else 1
            self.assertNotIn(status, (None, 0))
            build.assert_not_called()

    def test_equality_rejects_changed_rgb(self):
        import numpy as np
        arrays = [dict(rgb=np.zeros((40, 96, 3), dtype=np.float32)),
                  dict(rgb=np.ones((40, 96, 3), dtype=np.float32))]
        with mock.patch.object(self.wrapper, 'build'), \
             mock.patch.object(self.wrapper, 'checked', side_effect=arrays):
            with self.assertRaises((RuntimeError, ValueError, AssertionError)):
                self.wrapper.equal('trap', 2479, scale=SCALE)

    def test_equality_rejects_keys_dtype_hdr_and_alpha_changes(self):
        import numpy as np
        expected = dict(rgb=np.zeros((40, 96, 3), dtype=np.float32),
                        hdr=np.zeros((40, 96, 3), dtype=np.float32),
                        alpha=np.zeros((40, 96), dtype=np.float32))
        for change in ('missing_key', 'extra_key', 'dtype', 'hdr', 'alpha'):
            with self.subTest(change=change):
                actual = {key: value.copy() for key, value in expected.items()}
                if change == 'missing_key':
                    del actual['alpha']
                elif change == 'extra_key':
                    actual['unexpected'] = actual['alpha'].copy()
                elif change == 'dtype':
                    actual['rgb'] = actual['rgb'].astype(np.float64)
                else:
                    actual[change].flat[0] = 1
                message = ('array keys' if change.endswith('_key') else
                           'rgb' if change == 'dtype' else change)
                with mock.patch.object(self.wrapper, 'build') as build, \
                     mock.patch.object(self.wrapper, 'checked', side_effect=[expected, actual]):
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.wrapper.equal('deep', 4360, scale=SCALE)
                    self.assertEqual(build.call_count, 2)

    def test_linked_output_directories_rejected_before_build(self):
        for linked in ('renders', 'candidate', 'matte', 'dangling_candidate'):
            with self.subTest(linked=linked), tempfile.TemporaryDirectory(
                    prefix='cand-link-test-') as temporary:
                base = Path(temporary)
                fixture, outside = base / 'project', base / 'outside'
                fixture.mkdir()
                outside.mkdir()
                renders = fixture / 'renders'
                kind, option, frame = 'deep', 'leaned_ladders', 4360
                relative = Path('renders') / f'cand_{kind}_{option}'
                if linked == 'renders':
                    renders.symlink_to(outside, target_is_directory=True)
                else:
                    renders.mkdir()
                    name = relative.name + ('_matte' if linked == 'matte' else '')
                    target = base / 'absent' if linked == 'dangling_candidate' else outside
                    (renders / name).symlink_to(target, target_is_directory=True)
                with mock.patch.object(self.wrapper, 'ROOT', fixture), \
                     mock.patch.object(self.wrapper, 'build') as build:
                    with self.assertRaises(ValueError):
                        self.wrapper.output_dirs(kind, option, relative)
                    with self.assertRaises(ValueError):
                        self.wrapper.main([kind, option, '--range', f'{frame}-{frame}',
                                           '--out', str(relative)])
                    build.assert_not_called()
                self.assertEqual(list(outside.iterdir()), [])
                self.assertFalse((base / 'absent').exists())

    def test_checked_rejects_wrong_shape_and_nonfinite_auxiliary_arrays(self):
        import numpy as np
        rgb = np.zeros((40, 96, 3), dtype=np.float32)
        cases = [dict(rgb=rgb[:, :-1]),
                 dict(rgb=rgb, hdr=np.full_like(rgb, np.nan)),
                 dict(rgb=rgb, alpha=np.full((40, 96), np.inf, dtype=np.float32))]
        for result in cases:
            with self.subTest(keys=tuple(result)):
                with self.assertRaises((RuntimeError, ValueError, AssertionError)):
                    self.wrapper.checked(lambda frame: result, 2479, scale=SCALE)

    def test_save_png_retains_exact_eight_bit_pixels(self):
        import numpy as np
        from PIL import Image
        pixels = np.array([[[0, 17, 255], [128, 254, 1]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory(prefix='cand-png-test-') as temporary:
            target = Path(temporary) / 'f_02479.png'
            self.wrapper.save_png(target, pixels.astype(np.float32) / 255)
            with Image.open(target) as image:
                self.assertEqual(image.format, 'PNG')
                np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_save_png_refuses_overwrite_and_preserves_first_bytes(self):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix='cand-existing-png-') as temporary:
            directory = Path(temporary)
            target = directory / 'f_02479.png'
            self.wrapper.save_png(target, np.zeros((2, 3, 3), dtype=np.float32))
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                self.wrapper.save_png(target, np.ones((2, 3, 3), dtype=np.float32))
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(directory.iterdir()), [target])

    def test_each_manifest_is_rewritable_into_three_units(self):
        farm = load_module(ROOT / 'cloud/farm.py', 'cand_farm_contract_target')
        options = {**recipe_options(), 'crossing': 'both'}
        ranges = recipe_literal('ranges')
        probes = dict(re.findall(r'`ld_candidate equal (\w+) (\d+)`',
                                 (ROOT / 'ADOPTION.md').read_text()))
        self.assertEqual(len(options), 9)
        self.assertEqual(set(probes), set(options))
        for kind, option in options.items():
            with self.subTest(kind=kind):
                path = ROOT / 'cloud/jobs' / f'cand_{kind}_{option}.json'
                args = farm.parse_run_args([str(path), '--nodes', '3', '--dry-run'])
                job = farm.Job(str(path), args)
                self.assertFalse(job.whole)
                self.assertEqual(job.skipped_setup, [])
                setup_args = []
                for command in job.setup:
                    tokens = shlex.split(command)
                    self.assertIn('cloud/cand_render.py', tokens)
                    setup_args.append(tokens[tokens.index('cloud/cand_render.py') + 1:])
                self.assertEqual(setup_args, [[kind, '--prepare'],
                                             [kind, '--equal', probes[kind], '--scale', '1']])
                first, last = ranges[kind]
                expected_frames = list(range(first, last + 1))
                picture = f'renders/cand_{kind}_{option}'
                expected_dirs = [picture] + ([picture + '_matte']
                                             if kind in ('deep', 'pen', 't1') else [])
                self.assertEqual([o['out_dir'] for o in job.outputs], expected_dirs)
                for output in job.outputs:
                    self.assertEqual(output['frames'], expected_frames)
                command_frames = []
                for command in job.spec['render']:
                    tokens = shlex.split(command)
                    command_args = tokens[tokens.index('cloud/cand_render.py') + 1:]
                    self.assertEqual(command_args[:2], [kind, option])
                    self.assertEqual(command_args[command_args.index('--out') + 1], picture)
                    command_frames.extend(farm.LaneCmd(command).seq)
                self.assertEqual(sorted(command_frames), expected_frames)
                units = farm.plan_units([job], args)
                self.assertEqual(len(units), 3)
                by_output = {o['key']: [] for o in job.outputs}
                for unit in units:
                    unit_frames = sorted({f for output in unit['outputs'] for f in output['frames']})
                    rendered_frames = [f for item in unit['items']
                                       for f in farm.LaneCmd(item['cmd']).seq]
                    self.assertEqual(sorted(rendered_frames), unit_frames)
                    for output in unit['outputs']:
                        by_output[output['key']].extend(output['frames'])
                for output in job.outputs:
                    actual = by_output[output['key']]
                    self.assertEqual(sorted(actual), output['frames'])
                    self.assertEqual(len(actual), len(set(actual)))
                    self.assertEqual(Path(output['out_dir']).parent, Path('renders'))
                    self.assertTrue(Path(output['out_dir']).name.startswith(f'cand_{kind}_{option}'))


@unittest.skipUnless(os.environ.get('LD_CAND_RENDER_SLOW') == '1',
                     'real rendering requires LD_CAND_RENDER_SLOW=1')
class CandidateRecipeTests(unittest.TestCase):
    def test_trap(self):
        run_supervised('trap')

    def test_cold(self):
        run_supervised('cold')

    def test_deep(self):
        run_supervised('deep')

    def test_pen(self):
        run_supervised('pen')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--reference-child':
        try:
            reference_child(sys.argv[2], Path(sys.argv[3]))
        finally:
            print(json.dumps(dict(process_peak_rss_bytes=peak_rss_bytes())), flush=True)
    else:
        unittest.main()
