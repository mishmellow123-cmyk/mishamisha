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

    def test_scroll_source_range_and_run_family(self):
        self.assertEqual(self.wrapper.RANGES['scroll'], (0, 319))
        self.assertEqual(self.wrapper.OPTIONS['scroll'], 'night-fire')
        self.assertEqual(self.wrapper.ASSETS['scroll'], ('shots/run/summits.npy',))
        self.wrapper.validate_frames('scroll', 0, 319)
        for first, last in ((-1, 0), (0, 320), (3120, 3439)):
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                self.wrapper.validate_frames('scroll', first, last)
        with mock.patch.object(sys, 'path', []):
            self.wrapper.imports('scroll')
            self.assertEqual(sys.path[0], str(ROOT / 'shots/run'))
            self.assertEqual(self.wrapper._FAMILY, 'run')

    def test_scroll_routes_preserve_source_frame_and_default_is_accepted(self):
        module, source = mock.Mock(), mock.Mock()
        shot = source.Shot.return_value
        module.scroll.render.return_value = 'original'
        module.render.return_value = 'render'
        module.render_variants.return_value = {'accepted': 'shared'}
        with mock.patch.object(self.wrapper, 'imports'), \
             mock.patch.dict(sys.modules, beacon_night_candidates=module, render_ink=source):
            for route, result in (('original', 'original'), ('default', 'render'),
                                  ('shared', 'shared'), ('candidate', 'render')):
                with self.subTest(route=route):
                    module.reset_mock()
                    source.reset_mock()
                    option = 'night-fire' if route == 'candidate' else None
                    render = self.wrapper.build('scroll', route, option, scale=.5)
                    self.assertEqual(render(152), {'rgb': result})
                    source.Shot.assert_called_once_with('scroll')
                    module.scroll.make_shot.assert_not_called()
                    if route == 'original':
                        module.scroll.render.assert_called_once_with(152, shot, scale=.5, ss=2.)
                        module.render.assert_not_called()
                    elif route == 'shared':
                        module.render_variants.assert_called_once_with(
                            152, shot, kind='scroll', scale=.5, ss=2., variants=('accepted',))
                    else:
                        kwargs = dict(kind='scroll', scale=.5, ss=2.)
                        if route == 'candidate':
                            kwargs['variant'] = 'night-fire'
                        module.render.assert_called_once_with(152, shot, **kwargs)
            with self.assertRaisesRegex(ValueError, 'Unsupported candidate'):
                self.wrapper.build('scroll', 'candidate', 'night-wisp')

    def test_scroll_cli_routes_source_frames_and_picture_before_build(self):
        class StopBeforeRendering(Exception):
            pass

        with tempfile.TemporaryDirectory(prefix='cand-scroll-cli-') as temporary:
            fixture = Path(temporary)
            picture = fixture / 'renders/cand_scroll_night-fire'
            with mock.patch.object(self.wrapper, 'ROOT', fixture), \
                 mock.patch.object(self.wrapper, 'build', side_effect=StopBeforeRendering) as build:
                with self.assertRaises(StopBeforeRendering):
                    self.wrapper.main([
                        'scroll', 'night-fire', '--range', '0-319',
                        '--out', 'renders/cand_scroll_night-fire'])
                build.assert_called_once_with('scroll', 'candidate', 'night-fire')
            self.assertEqual(list((fixture / 'renders').iterdir()), [picture])
            self.assertEqual(list(picture.iterdir()), [])

    def test_scroll_equality_checks_default_and_shared_routes(self):
        import numpy as np
        accepted = dict(rgb=np.zeros((2, 3, 3), dtype=np.float32))
        with mock.patch.object(self.wrapper, 'build') as build, \
             mock.patch.object(self.wrapper, 'checked', return_value=accepted), \
             contextlib.redirect_stdout(io.StringIO()):
            receipt = self.wrapper.equal('scroll', 152, scale=.5)
        self.assertEqual(build.call_args_list, [
            mock.call('scroll', 'original', scale=.5),
            mock.call('scroll', 'default', scale=.5),
            mock.call('scroll', 'shared', scale=.5)])
        self.assertEqual(set(receipt['proofs']), {'default', 'shared'})
        changed = dict(rgb=accepted['rgb'].copy())
        changed['rgb'].flat[0] = 1
        with mock.patch.object(self.wrapper, 'build'), \
             mock.patch.object(self.wrapper, 'checked', side_effect=[accepted, accepted, changed]):
            with self.assertRaisesRegex(RuntimeError, 'scroll shared rgb frame 152'):
                self.wrapper.equal('scroll', 152, scale=.5)

    def test_crossing_cap_preserves_existing_routes_and_candidate_layers(self):
        self.assertEqual(self.wrapper.OPTIONS['crossing'], 'both')
        crossing, rock, renderer = mock.Mock(), mock.Mock(), mock.Mock()
        crossing.load_renderer.return_value = renderer
        crossing.render_cut.return_value = renderer.render.return_value = mock.sentinel.hdr
        renderer.FINISH = {'exposure': 1.0}
        renderer.PI.look.finish.return_value = mock.sentinel.rgb
        cases = [('original', None), ('default', None), ('shared', None),
                 ('candidate', 'both'), ('candidate', 'both_decal'),
                 ('candidate', 'both_decal_cap')]
        with mock.patch.object(self.wrapper, 'imports'), \
             mock.patch.dict(sys.modules, crossing_candidates=crossing,
                             crossing_rock_candidates=rock):
            for route, option in cases:
                with self.subTest(route=route, option=option):
                    crossing.reset_mock()
                    rock.reset_mock()
                    renderer.reset_mock()
                    result = self.wrapper.build('crossing', route, option, scale=.5)(5043)
                    self.assertEqual(result, dict(hdr=mock.sentinel.hdr, rgb=mock.sentinel.rgb))
                    common = dict(scale=.5, ss=1.5, variant='main', trail=True)
                    if route == 'original':
                        renderer.render.assert_called_once_with(163, **common)
                        crossing.render_cut.assert_not_called()
                    else:
                        expected = dict(renderer=renderer, **common)
                        if route != 'default':
                            modifier = crossing.render_cut.call_args.kwargs['rock_modifier']
                            expected.update(
                                rope=('snow_clearance' if option == 'both' else
                                      'snow_decal' if option else 'accepted'),
                                rock='low_shoulders' if option else 'accepted',
                                rock_modifier=modifier if option else None)
                            if option:
                                modifier(mock.sentinel.scene, renderer, mock.sentinel.cfg)
                                rock.apply_to_scene.assert_called_once_with(
                                    mock.sentinel.scene, renderer, mock.sentinel.cfg,
                                    candidate='low_shoulders')
                        if option == 'both_decal_cap':
                            expected['terrain'] = 'round_cap'
                        crossing.render_cut.assert_called_once_with(5043, **expected)
                        renderer.render.assert_not_called()
                    renderer.PI.look.finish.assert_called_once_with(
                        mock.sentinel.hdr, **renderer.FINISH)

    def test_crossing_cap_cli_uses_separate_output_before_build(self):
        class StopBeforeRendering(Exception):
            pass

        with tempfile.TemporaryDirectory(prefix='cand-crossing-cap-cli-') as temporary:
            fixture = Path(temporary)
            picture = fixture / 'renders/cand_crossing_both_decal_cap'
            with mock.patch.object(self.wrapper, 'ROOT', fixture), \
                 mock.patch.object(self.wrapper, 'build', side_effect=StopBeforeRendering) as build:
                with self.assertRaises(StopBeforeRendering):
                    self.wrapper.main([
                        'crossing', 'both_decal_cap', '--range', '4880-5839',
                        '--out', 'renders/cand_crossing_both_decal_cap'])
                build.assert_called_once_with('crossing', 'candidate', 'both_decal_cap')
                build.reset_mock()
                with self.assertRaisesRegex(ValueError, 'Use renders/cand_'):
                    self.wrapper.main([
                        'crossing', 'both_decal_cap', '--range', '4880-5839',
                        '--out', 'renders/cand_crossing_both_decal'])
                build.assert_not_called()
            self.assertEqual(list((fixture / 'renders').iterdir()), [picture])
            self.assertEqual(list(picture.iterdir()), [])

    def test_t1_held_camera_requires_explicit_candidate_option(self):
        self.assertEqual(self.wrapper.OPTIONS['t1'], 'current-words')
        renderer = mock.Mock()
        module = mock.Mock()
        module.make_renderer.return_value = renderer
        with mock.patch.object(self.wrapper, 'imports'), \
             mock.patch.object(self.wrapper.importlib, 'import_module', return_value=module), \
             mock.patch.object(self.wrapper, 'book_output', return_value='output'):
            default = self.wrapper.build('t1', 'default')
            module.make_renderer.assert_called_once_with(scale=1.0, with_fire=False)
            module.make_renderer.reset_mock()
            candidate = self.wrapper.build('t1', 'candidate', 'current-words-held')
            module.make_renderer.assert_called_once_with(
                'current-words-held', scale=1.0, with_fire=False)
            self.assertTrue(renderer.no_burn)
            self.assertEqual(candidate(527), 'output')
            renderer.frame.assert_called_once_with(527)
            self.assertTrue(callable(default))

    def test_t1_held_cli_routes_picture_and_matte_before_build(self):
        class StopBeforeRendering(Exception):
            pass

        with tempfile.TemporaryDirectory(prefix='cand-t1-held-cli-') as temporary:
            fixture = Path(temporary)
            picture = fixture / 'renders/cand_t1_current-words-held'
            matte = picture.with_name(picture.name + '_matte')
            with mock.patch.object(self.wrapper, 'ROOT', fixture), \
                 mock.patch.object(self.wrapper, 'build', side_effect=StopBeforeRendering) as build:
                with self.assertRaises(StopBeforeRendering):
                    self.wrapper.main([
                        't1', 'current-words-held', '--range', '320-559',
                        '--out', 'renders/cand_t1_current-words-held'])
                build.assert_called_once_with('t1', 'candidate', 'current-words-held')
            self.assertTrue(picture.is_dir())
            self.assertTrue(matte.is_dir())
            self.assertEqual(list(picture.iterdir()), [])
            self.assertEqual(list(matte.iterdir()), [])

    def test_t1_held_manifest_preserves_current_words_setup(self):
        directory = ROOT / 'cloud/jobs'
        original = json.loads((directory / 'cand_t1_current-words.json').read_text())
        held = json.loads((directory / 'cand_t1_current-words-held.json').read_text())
        self.assertEqual(held['setup'], original['setup'])
        for key in ('branch', 'frames', 'shape', 'ship'):
            self.assertEqual(held[key], original[key])

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
        options = list({**recipe_options(), 'crossing': 'both'}.items())
        options.append(('t1', 'current-words-held'))
        options.append(('scroll', 'night-fire'))
        options.append(('crossing', 'both_decal_cap'))
        ranges = recipe_literal('ranges')
        ranges['scroll'] = (0, 319)
        probes = dict(re.findall(r'`ld_candidate equal (\w+) (\d+)`',
                                 (ROOT / 'ADOPTION.md').read_text()))
        probes['scroll'] = '152'
        self.assertEqual(len(options), 12)
        self.assertEqual(set(probes), {kind for kind, _ in options})
        for kind, option in options:
            with self.subTest(kind=kind, option=option):
                path = ROOT / 'cloud/jobs' / f'cand_{kind}_{option}.json'
                args = farm.parse_run_args([str(path), '--nodes', '3', '--dry-run'])
                job = farm.Job(str(path), args)
                if option == 'both_decal_cap':
                    self.assertEqual(job.spec['branch'], 'claude/owner-night-20260929')
                    self.assertEqual(job.spec['shape'], [804, 1920])
                    self.assertEqual(job.spec['ship'], 'jpg')
                    self.assertEqual(job.spec['name'], 'cand_crossing_both_decal_cap')
                self.assertFalse(job.whole)
                self.assertEqual(job.skipped_setup, [])
                if kind == 'scroll':
                    self.assertEqual(job.name, 'cand_scroll_night-fire')
                    self.assertEqual(job.shape, (804, 1920))
                    self.assertEqual(job.spec['ship'], 'jpg')
                    self.assertEqual(job.spec['branch'], 'claude/owner-night-20260929')
                setup_args = []
                for command in job.setup:
                    tokens = shlex.split(command)
                    self.assertIn('cloud/cand_render.py', tokens)
                    setup_args.append(tokens[tokens.index('cloud/cand_render.py') + 1:])
                probe = '5043' if option == 'both_decal_cap' else probes[kind]
                self.assertEqual(setup_args, [[kind, '--prepare'],
                                             [kind, '--equal', probe, '--scale', '1']])
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
                    if kind == 'scroll':
                        self.assertEqual(unit['branch'], 'claude/owner-night-20260929')
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
