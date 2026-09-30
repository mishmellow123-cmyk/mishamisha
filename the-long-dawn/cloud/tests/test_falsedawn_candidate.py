"""False-dawn farm contracts; renderer calls are mocked and no frames are rendered."""
import importlib.util
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
KIND = 'falsedawn'
OPTION = 'clear_high_deck'
OUTPUT = 'renders/cand_falsedawn_clear_high_deck'
JOB = ROOT / 'cloud/jobs/cand_falsedawn_clear_high_deck.json'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FalseDawnCandidateTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = load_module(ROOT / 'cloud/cand_render.py',
                                   'falsedawn_adapter_contract')

    def test_cut_frame_range_and_run_family(self):
        self.assertEqual(self.wrapper.RANGES[KIND], (80, 559))
        self.assertEqual(self.wrapper.OPTIONS[KIND], OPTION)
        self.wrapper.validate_frames(KIND, 80, 559)
        for first, last in ((79, 80), (559, 560), (212, 211)):
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                self.wrapper.validate_frames(KIND, first, last)
        with mock.patch.object(sys, 'path', list(sys.path)):
            self.wrapper.imports(KIND)
            self.assertEqual(self.wrapper._FAMILY, 'run')
            self.assertEqual(sys.path[0], str(ROOT / 'shots/run'))

    def test_only_explicit_candidate_changes_renderer_arguments(self):
        hdr, rgb = object(), object()
        finish = mock.Mock(return_value=rgb)
        renderer = SimpleNamespace(
            CUT0=80, FINISH={'exposure': object(), 'vignette_amount': object()},
            render=mock.Mock(return_value=hdr),
            PI=SimpleNamespace(look=SimpleNamespace(finish=finish)))
        for route in ('original', 'default', 'shared', 'candidate'):
            with self.subTest(route=route), \
                 mock.patch.object(self.wrapper, 'imports'), \
                 mock.patch.dict(sys.modules, falsedawn=renderer):
                result = self.wrapper.build(KIND, route, OPTION, scale=.5)(212)
                expected = dict(design='arc', scale=.5, ss=1.5)
                if route == 'candidate':
                    expected['sky_candidate'] = OPTION
                renderer.render.assert_called_once_with(132, **expected)
                finish.assert_called_once_with(hdr, **renderer.FINISH)
                self.assertEqual(set(result), {'hdr', 'rgb'})
                self.assertIs(result['hdr'], hdr)
                self.assertIs(result['rgb'], rgb)
                renderer.render.reset_mock()
                finish.reset_mock()

    def test_unsupported_candidate_rejected_before_import(self):
        for option in (None, 'accepted', 'unknown'):
            with self.subTest(option=option), \
                 mock.patch.object(self.wrapper, 'imports') as imports, \
                 mock.patch.dict(sys.modules, falsedawn=SimpleNamespace()):
                with self.assertRaisesRegex(ValueError, 'Unsupported candidate'):
                    self.wrapper.build(KIND, 'candidate', option)
                imports.assert_not_called()

    def test_render_cli_creates_only_candidate_output_before_build(self):
        class StopBeforeRendering(Exception):
            pass

        with tempfile.TemporaryDirectory(prefix='falsedawn-cli-contract-') as temporary:
            fixture = Path(temporary)
            with mock.patch.object(self.wrapper, 'ROOT', fixture), \
                 mock.patch.object(self.wrapper, 'build', side_effect=StopBeforeRendering) as build:
                with self.assertRaises(StopBeforeRendering):
                    self.wrapper.main([KIND, OPTION, '--range', '80-559', '--out', OUTPUT])
                build.assert_called_once_with(KIND, 'candidate', OPTION)
            self.assertEqual(list((fixture / 'renders').iterdir()), [fixture / OUTPUT])
            self.assertEqual(list((fixture / OUTPUT).iterdir()), [])

    def test_all_cli_modes_refuse_owner_renders_link_before_import(self):
        with tempfile.TemporaryDirectory(prefix='falsedawn-link-contract-') as temporary:
            base = Path(temporary)
            fixture, outside = base / 'project', base / 'outside'
            fixture.mkdir()
            outside.mkdir()
            (fixture / 'renders').symlink_to(outside, target_is_directory=True)
            commands = ([KIND, '--prepare'], [KIND, '--equal', '212', '--scale', '1'],
                        [KIND, OPTION, '--range', '80-559', '--out', OUTPUT])
            for command in commands:
                with self.subTest(command=command), \
                     mock.patch.object(self.wrapper, 'ROOT', fixture), \
                     mock.patch.object(self.wrapper, 'build',
                                       side_effect=AssertionError('Build reached before link refusal')) as build, \
                     mock.patch.object(self.wrapper, 'equal',
                                       side_effect=AssertionError('Equality reached before link refusal')) as equal, \
                     mock.patch.object(self.wrapper.importlib.metadata, 'version',
                                       side_effect=AssertionError('Stack inspection reached before link refusal')) as version:
                    with self.assertRaisesRegex(ValueError, 'linked renders'):
                        self.wrapper.main(command)
                    build.assert_not_called()
                    equal.assert_not_called()
                    version.assert_not_called()
            self.assertEqual(list(outside.iterdir()), [])

    def test_manifest_keeps_native_candidate_isolated_and_splittable(self):
        farm = load_module(ROOT / 'cloud/farm.py', 'falsedawn_farm_contract')
        args = farm.parse_run_args([str(JOB), '--nodes', '3', '--dry-run'])
        job = farm.Job(str(JOB), args)
        self.assertFalse(job.whole)
        self.assertEqual(job.skipped_setup, [])
        self.assertEqual(job.spec['name'], 'cand_falsedawn_clear_high_deck')
        self.assertEqual(job.spec['branch'], 'claude/owner-night-20260929')
        self.assertEqual(job.shape, (804, 1920))
        self.assertEqual(job.spec['ship'], 'jpg')
        self.assertEqual([output['out_dir'] for output in job.outputs], [OUTPUT])
        expected_frames = list(range(80, 560))
        self.assertEqual(job.outputs[0]['frames'], expected_frames)
        expected_environment = {
            'NUMBA_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
            'MKL_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1'}
        commands = []
        for command in job.setup + job.spec['render']:
            tokens = shlex.split(command)
            index = tokens.index('python3')
            self.assertEqual(set(tokens[:index]), expected_environment)
            self.assertEqual(tokens[index + 1], 'cloud/cand_render.py')
            commands.append(tokens[index + 2:])
        self.assertEqual(commands, [
            [KIND, '--prepare'], [KIND, '--equal', '212', '--scale', '1'],
            [KIND, OPTION, '--range', '80-559', '--out', OUTPUT]])
        units = farm.plan_units([job], args)
        self.assertEqual(len(units), 3)
        landed, rendered = [], []
        for unit in units:
            self.assertEqual(unit['branch'], job.spec['branch'])
            self.assertEqual(len(unit['outputs']), 1)
            self.assertEqual(unit['outputs'][0]['out_dir'], OUTPUT)
            unit_frames = unit['outputs'][0]['frames']
            command_frames = [frame for item in unit['items']
                              for frame in farm.LaneCmd(item['cmd']).seq]
            self.assertEqual(sorted(command_frames), unit_frames)
            landed.extend(unit_frames)
            rendered.extend(command_frames)
        self.assertEqual(sorted(landed), expected_frames)
        self.assertEqual(sorted(rendered), expected_frames)
        self.assertEqual(len(landed), len(set(landed)))


if __name__ == '__main__':
    unittest.main()
