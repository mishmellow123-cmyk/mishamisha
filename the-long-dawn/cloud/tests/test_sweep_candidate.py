"""Sweep adapter contracts with stubs; no delivered frame or renderer is loaded."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
KIND, OPTION = 'sweep', 'soft-entry'
OUTPUT = 'renders/cand_sweep_soft-entry'
JOB = ROOT / 'cloud/jobs/cand_sweep_soft-entry.json'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SweepCandidateTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = load_module(ROOT / 'cloud/cand_render.py', 'sweep_adapter_contract')

    def test_range_is_exact_changed_span_in_absolute_c_frames(self):
        self.assertEqual(self.wrapper.RANGES[KIND], (1680, 1687))
        self.assertEqual(self.wrapper.OPTIONS[KIND], OPTION)
        self.assertIn(KIND, self.wrapper.PAGES)
        self.wrapper.validate_frames(KIND, 1680, 1687)
        for first, last in ((1679, 1680), (1687, 1688), (1681, 1680)):
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                self.wrapper.validate_frames(KIND, first, last)
        with mock.patch.object(sys, 'path', list(sys.path)):
            self.wrapper.imports(KIND)
            self.assertEqual(self.wrapper._FAMILY, 'map')
            self.assertEqual(sys.path[0], str(ROOT / 'shots/map'))

    def test_routes_preserve_rgb_matte_and_require_explicit_opt_in(self):
        result = dict(rgb=object(), alpha=object())
        renderer = mock.Mock(return_value=result)
        module = SimpleNamespace(render_original=mock.Mock(return_value=result),
                                 make_renderer=mock.Mock(return_value=renderer))
        cases = [('original', None), ('default', None), ('shared', None), ('candidate', OPTION)]
        for route, option in cases:
            module.render_original.reset_mock()
            module.make_renderer.reset_mock()
            renderer.reset_mock()
            with self.subTest(route=route), mock.patch.object(self.wrapper, 'imports'), \
                 mock.patch.dict(sys.modules, sweep_entry_candidates=module), \
                 mock.patch.object(self.wrapper, 'book_output') as finish:
                actual = self.wrapper.build(KIND, route, option, scale=.5)(1683)
                self.assertIs(actual, result)
                finish.assert_not_called()
                if route == 'original':
                    module.render_original.assert_called_once_with(1683, scale=.5)
                    module.make_renderer.assert_not_called()
                    renderer.assert_not_called()
                else:
                    module.render_original.assert_not_called()
                    args = () if route == 'default' else ('accepted' if route == 'shared' else OPTION,)
                    module.make_renderer.assert_called_once_with(*args, scale=.5)
                    renderer.assert_called_once_with(1683)

    def test_unsupported_candidate_fails_before_import(self):
        for option in (None, 'accepted', 'unknown'):
            with self.subTest(option=option), mock.patch.object(self.wrapper, 'imports') as imports:
                with self.assertRaisesRegex(ValueError, 'Unsupported candidate'):
                    self.wrapper.build(KIND, 'candidate', option)
                imports.assert_not_called()

    def test_prepare_requires_pinned_inputs_and_prints_receipt(self):
        receipt = {'manifest_sha256': 'test-fixture', 'files': {}}
        module = SimpleNamespace(validate_inputs=mock.Mock(return_value=receipt))
        with mock.patch.object(self.wrapper, 'writable_renders'), \
             mock.patch.object(self.wrapper, 'imports') as imports, \
             mock.patch.object(self.wrapper.sys, 'version_info', (3, 12)), \
             mock.patch.object(self.wrapper.importlib.metadata, 'version',
                               side_effect=self.wrapper.STACK.__getitem__), \
             mock.patch.dict(sys.modules, sweep_entry_candidates=module):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.wrapper.prepare(KIND)
            module.validate_inputs.assert_called_once_with()
            imports.assert_called_once_with(KIND)
            self.assertEqual(json.loads(output.getvalue())['inputs'], receipt)
            module.validate_inputs.side_effect = RuntimeError('source hash mismatch')
            with self.assertRaisesRegex(RuntimeError, 'source hash mismatch'), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.wrapper.prepare(KIND)

    def test_equality_checks_default_shared_and_opaque_matte(self):
        import numpy as np
        accepted = dict(rgb=np.zeros((1, 1, 3), np.float32), alpha=np.ones((1, 1), np.float32))
        with mock.patch.object(self.wrapper, 'build') as build, \
             mock.patch.object(self.wrapper, 'checked', return_value=accepted), \
             contextlib.redirect_stdout(io.StringIO()):
            receipt = self.wrapper.equal(KIND, 1680, scale=.5)
        self.assertEqual(build.call_args_list, [mock.call(KIND, route, scale=.5)
                                               for route in ('original', 'default', 'shared')])
        self.assertEqual(set(receipt['proofs']), {'default', 'shared'})
        for route in ('default', 'shared'):
            for key in ('rgb', 'alpha'):
                changed = {name: array.copy() for name, array in accepted.items()}
                changed[key].flat[0] = 1 - changed[key].flat[0]
                outputs = [accepted, changed, accepted] if route == 'default' else [accepted, accepted, changed]
                with self.subTest(route=route, key=key), \
                     mock.patch.object(self.wrapper, 'build'), \
                     mock.patch.object(self.wrapper, 'checked', side_effect=outputs), \
                     self.assertRaisesRegex(RuntimeError, f'sweep {route} {key} frame 1680'):
                    self.wrapper.equal(KIND, 1680, scale=.5)

    def test_cli_creates_isolated_picture_and_matte_before_build(self):
        class StopBeforeRendering(Exception):
            pass

        with tempfile.TemporaryDirectory(prefix='sweep-cli-contract-') as temporary:
            fixture = Path(temporary)
            with mock.patch.object(self.wrapper, 'ROOT', fixture), \
                 mock.patch.object(self.wrapper, 'build', side_effect=StopBeforeRendering) as build:
                with self.assertRaises(StopBeforeRendering):
                    self.wrapper.main([KIND, OPTION, '--range', '1680-1687', '--out', OUTPUT])
                build.assert_called_once_with(KIND, 'candidate', OPTION)
                build.reset_mock()
                with self.assertRaisesRegex(ValueError, 'Use renders/cand_'):
                    self.wrapper.main([KIND, OPTION, '--range', '1680-1687',
                                       '--out', 'renders/book_C_ft'])
                build.assert_not_called()
            expected = [fixture / OUTPUT, fixture / (OUTPUT + '_matte')]
            self.assertEqual(sorted((fixture / 'renders').iterdir()), sorted(expected))
            for directory in expected:
                self.assertEqual(list(directory.iterdir()), [])

    def test_all_cli_modes_refuse_owner_renders_link_before_import(self):
        with tempfile.TemporaryDirectory(prefix='sweep-link-contract-') as temporary:
            base = Path(temporary)
            fixture, outside = base / 'project', base / 'outside'
            fixture.mkdir()
            outside.mkdir()
            (fixture / 'renders').symlink_to(outside, target_is_directory=True)
            commands = ([KIND, '--prepare'], [KIND, '--equal', '1680', '--scale', '1'],
                        [KIND, OPTION, '--range', '1680-1687', '--out', OUTPUT])
            for command in commands:
                with self.subTest(command=command), mock.patch.object(self.wrapper, 'ROOT', fixture), \
                     mock.patch.object(self.wrapper, 'build',
                                       side_effect=AssertionError('build reached after owner symlink')) as build, \
                     mock.patch.object(self.wrapper, 'equal',
                                       side_effect=AssertionError('equal reached after owner symlink')) as equal, \
                     mock.patch.object(self.wrapper.importlib.metadata, 'version',
                                       side_effect=AssertionError('stack check reached after owner symlink')) as version:
                    with self.assertRaisesRegex(ValueError, 'linked renders'):
                        self.wrapper.main(command)
                    build.assert_not_called()
                    equal.assert_not_called()
                    version.assert_not_called()
            self.assertEqual(list(outside.iterdir()), [])

    def test_manifest_and_farm_split_cover_exactly_eight_frames_and_both_outputs(self):
        farm = load_module(ROOT / 'cloud/farm.py', 'sweep_farm_contract')
        args = farm.parse_run_args([str(JOB), '--nodes', '3', '--dry-run'])
        job = farm.Job(str(JOB), args)
        self.assertFalse(job.whole)
        self.assertEqual(job.skipped_setup, [])
        self.assertEqual(job.spec['name'], 'cand_sweep_soft-entry')
        self.assertEqual(job.spec['branch'], 'claude/owner-night-20260929')
        self.assertEqual(job.shape, (804, 1920))
        self.assertEqual(job.spec['ship'], 'jpg')
        expected_frames = list(range(1680, 1688))
        expected_dirs = [OUTPUT, OUTPUT + '_matte']
        self.assertEqual(job.spec['frames'], '1680-1687')
        self.assertEqual(job.spec['out_dir'], OUTPUT)
        self.assertEqual([output['out_dir'] for output in job.outputs], expected_dirs)
        for output in job.outputs:
            self.assertEqual(output['frames'], expected_frames)
        environment = {'NUMBA_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                       'MKL_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1'}
        commands = []
        for command in job.setup + job.spec['render']:
            tokens = shlex.split(command)
            index = tokens.index('python3')
            self.assertEqual(set(tokens[:index]), environment)
            self.assertEqual(tokens[index + 1], 'cloud/cand_render.py')
            commands.append(tokens[index + 2:])
        self.assertEqual(commands, [[KIND, '--prepare'], [KIND, '--equal', '1680', '--scale', '1'],
                                    [KIND, OPTION, '--range', '1680-1687', '--out', OUTPUT]])
        units = farm.plan_units([job], args)
        self.assertEqual(len(units), 3)
        landed = {directory: [] for directory in expected_dirs}
        rendered = []
        for unit in units:
            self.assertEqual(unit['branch'], job.spec['branch'])
            self.assertEqual([output['out_dir'] for output in unit['outputs']], expected_dirs)
            frames = [frame for item in unit['items'] for frame in farm.LaneCmd(item['cmd']).seq]
            self.assertEqual(len(frames), len(set(frames)))
            rendered.extend(frames)
            for output in unit['outputs']:
                self.assertEqual(sorted(frames), output['frames'])
                landed[output['out_dir']].extend(output['frames'])
        self.assertEqual(sorted(rendered), expected_frames)
        for frames in landed.values():
            self.assertEqual(sorted(frames), expected_frames)


if __name__ == '__main__':
    unittest.main()
