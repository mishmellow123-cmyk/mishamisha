"""In-step adapter contracts and failing controls, without rendering a frame."""
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
KINDS = {'cold': (3816, 3999), 'unfinished': (4000, 4239)}
ASSETS = ('assets/ring/inscription_outer.png', 'assets/ring/inscription_inner.png',
          'music/v3/barmap_C5P2.json')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstepCandidateTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = load_module(ROOT / 'cloud/cand_render.py', 'instep_adapter_contract')

    def test_ranges_and_options_preserve_lead24_and_use_embers(self):
        self.assertEqual(self.wrapper.OPTIONS['cold'], 'lead24')
        self.assertEqual(self.wrapper.MORE['cold'], ('in-step',))
        self.assertEqual(self.wrapper.OPTIONS['unfinished'], 'in-step')
        for kind, (first, last) in KINDS.items():
            self.assertEqual(self.wrapper.RANGES[kind], (first, last))
            self.wrapper.validate_frames(kind, first, last)
            for bounds in ((first - 1, last), (first, last + 1), (last, first)):
                with self.subTest(kind=kind, bounds=bounds), self.assertRaises(ValueError):
                    self.wrapper.validate_frames(kind, *bounds)
            self.wrapper._FAMILY = None
            with mock.patch.object(sys, 'path', list(sys.path)):
                self.wrapper.imports(kind)
                self.assertEqual(self.wrapper._FAMILY, 'embers')
                self.assertEqual(sys.path[0], str(ROOT / 'shots/embers'))

    def test_default_off_routes_and_accepted_leadin(self):
        for kind in KINDS:
            for route, option in (('original', None), ('default', None),
                                  ('shared', None), ('candidate', 'in-step'),
                                  *((('candidate', 'lead24'),) if kind == 'cold' else ())):
                lead, base, candidate = mock.Mock(), mock.Mock(), mock.Mock()
                candidate.base = base
                constructors = (lead.Scene, base.Scene, candidate.Scene)
                for index, constructor in enumerate(constructors):
                    constructor.return_value.frame.return_value = ('pixels', index)
                modules = {'c_v5_cold_leadin': lead, 'c_v5_instep': candidate}
                with self.subTest(kind=kind, route=route, option=option), \
                     mock.patch.object(self.wrapper, 'imports'), mock.patch.dict(sys.modules, modules):
                    render = self.wrapper.build(kind, route, option, scale=.5)
                    actual = render(KINDS[kind][0])
                if route == 'original':
                    index, args, kwargs = (0, ('lead24',), {}) if kind == 'cold' else (1, (), {})
                elif route == 'default':
                    index, args, kwargs = 2, (), {}
                elif route == 'shared':
                    index, args, kwargs = 2, (), {'variant': 'accepted'}
                else:
                    index, args, kwargs = (0 if option == 'lead24' else 2), (), {'variant': option}
                self.assertEqual(actual, {'rgb': ('pixels', index)})
                for other, constructor in enumerate(constructors):
                    if other == index:
                        constructor.assert_called_once_with(*args, **kwargs)
                        constructor.return_value.frame.assert_called_once_with(KINDS[kind][0], scale=.5)
                    else:
                        constructor.assert_not_called()

    def test_invalid_candidates_fail_before_import(self):
        for kind in KINDS:
            for option in (None, 'accepted', 'unknown', *(('lead24',) if kind == 'unfinished' else ())):
                with self.subTest(kind=kind, option=option), \
                     mock.patch.object(self.wrapper, 'imports') as imports, \
                     self.assertRaisesRegex(ValueError, 'Unsupported candidate'):
                    self.wrapper.build(kind, 'candidate', option)
                imports.assert_not_called()

    def test_all_three_prechange_equality_proofs_and_each_failing_control(self):
        import numpy as np
        accepted = {'rgb': np.zeros((1, 1, 3), np.float32)}
        for frame in (3816, 3840, 3847):
            calls = [mock.call('cold', route, scale=.5) for route in ('original', 'default', 'shared')]
            calls.append(mock.call('cold', 'candidate', 'in-step', scale=.5))
            with mock.patch.object(self.wrapper, 'build') as build, \
                 mock.patch.object(self.wrapper, 'checked', return_value=accepted), \
                 contextlib.redirect_stdout(io.StringIO()):
                receipt = self.wrapper.equal('cold', frame, .5, candidate='in-step')
            self.assertEqual(build.call_args_list, calls)
            self.assertEqual(receipt['candidate'], 'in-step')
            self.assertEqual(set(receipt['proofs']), {'default', 'shared', 'candidate'})
            for index, route in enumerate(('default', 'shared', 'candidate'), start=1):
                for failure in ('pixel', 'dtype', 'key'):
                    changed = {'rgb': accepted['rgb'].copy()}
                    if failure == 'pixel':
                        changed['rgb'].flat[0] = 1
                    elif failure == 'dtype':
                        changed['rgb'] = changed['rgb'].astype(np.float64)
                    else:
                        changed['alpha'] = np.ones((1, 1), np.float32)
                    outputs = [accepted] * 4
                    outputs[index] = changed
                    with self.subTest(frame=frame, route=route, failure=failure), \
                         mock.patch.object(self.wrapper, 'build'), \
                         mock.patch.object(self.wrapper, 'checked', side_effect=outputs), \
                         self.assertRaisesRegex(RuntimeError, f'cold {route}'):
                        self.wrapper.equal('cold', frame, .5, candidate='in-step')

    def test_candidate_equality_rejects_changed_domain_and_wrong_option(self):
        for kind, frame, option in (('cold', 3848, 'in-step'), ('cold', 3847, 'lead24'),
                                    ('unfinished', 4000, 'in-step')):
            with mock.patch.object(self.wrapper, 'build') as build, \
                 self.assertRaisesRegex(ValueError, 'before C3848'):
                self.wrapper.equal(kind, frame, candidate=option)
            build.assert_not_called()

    def test_unfinished_equality_retains_default_and_shared_accepted(self):
        import numpy as np
        accepted = {'rgb': np.zeros((1, 1, 3), np.float32)}
        with mock.patch.object(self.wrapper, 'build') as build, \
             mock.patch.object(self.wrapper, 'checked', return_value=accepted), \
             contextlib.redirect_stdout(io.StringIO()):
            receipt = self.wrapper.equal('unfinished', 4000, .5)
        self.assertEqual(build.call_args_list, [mock.call('unfinished', route, scale=.5)
                                               for route in ('original', 'default', 'shared')])
        self.assertEqual(set(receipt['proofs']), {'default', 'shared'})

    def test_prepare_requires_each_input_including_score(self):
        for kind in KINDS:
            self.assertEqual(self.wrapper.ASSETS[kind], ASSETS)
            for missing in (None, *ASSETS):
                with tempfile.TemporaryDirectory(prefix='instep-prepare-') as temporary:
                    fixture = Path(temporary)
                    for name in ASSETS:
                        if name != missing:
                            path = fixture / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(b'input fixture')
                    with mock.patch.object(self.wrapper, 'ROOT', fixture), \
                         mock.patch.object(sys, 'version_info', (3, 12)), \
                         mock.patch.object(self.wrapper.importlib.metadata, 'version',
                                           side_effect=self.wrapper.STACK.__getitem__), \
                         contextlib.redirect_stdout(io.StringIO()) as output:
                        if missing:
                            with self.assertRaisesRegex(RuntimeError, 'Required tracked input is missing'):
                                self.wrapper.prepare(kind)
                        else:
                            self.wrapper.prepare(kind)
                            self.assertEqual(set(json.loads(output.getvalue())['assets']), set(ASSETS))

    def test_cli_equality_explicit_candidate_and_mode_rejection(self):
        with mock.patch.object(self.wrapper, 'writable_renders'), \
             mock.patch.object(self.wrapper, 'equal') as equal, \
             contextlib.redirect_stdout(io.StringIO()):
            self.wrapper.main(['cold', '--equal', '3816', '--candidate', 'in-step', '--scale', '.5'])
        equal.assert_called_once_with('cold', 3816, .5, candidate='in-step')
        for argv in (['cold', '--prepare', '--candidate', 'in-step'],
                     ['cold', 'in-step', '--candidate', 'in-step', '--range', '3816-3999',
                      '--out', 'renders/cand_cold_in-step']):
            with mock.patch.object(self.wrapper, 'build') as build, \
                 mock.patch.object(self.wrapper, 'prepare') as prepare, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.wrapper.main(argv)
            build.assert_not_called()
            prepare.assert_not_called()

    def test_isolated_outputs_and_all_modes_reject_owner_link_before_import(self):
        class StopBeforeRendering(Exception):
            pass

        for kind, (first, last) in KINDS.items():
            output = f'renders/cand_{kind}_in-step'
            commands = ([kind, '--prepare'], [kind, '--equal', str(first), '--scale', '1'],
                        [kind, 'in-step', '--range', f'{first}-{last}', '--out', output])
            if kind == 'cold':
                commands += ([kind, '--equal', '3816', '--candidate', 'in-step', '--scale', '1'],)
            with tempfile.TemporaryDirectory(prefix='instep-isolation-') as temporary:
                fixture = Path(temporary) / 'project'
                outside = Path(temporary) / 'owner'
                fixture.mkdir()
                outside.mkdir()
                with mock.patch.object(self.wrapper, 'ROOT', fixture), \
                     mock.patch.object(self.wrapper, 'build', side_effect=StopBeforeRendering) as build:
                    with self.assertRaises(StopBeforeRendering):
                        self.wrapper.main(commands[2])
                    build.assert_called_once_with(kind, 'candidate', 'in-step')
                    self.assertEqual(list((fixture / output).iterdir()), [])
                    build.reset_mock()
                    for bad in ('renders/embers_C5_unfinished', 'renders/cand_cold_lead24',
                                str(outside / Path(output).name)):
                        with self.assertRaises(ValueError):
                            self.wrapper.main([kind, 'in-step', '--range', f'{first}-{last}', '--out', bad])
                    build.assert_not_called()
                (fixture / output).rmdir()
                (fixture / 'renders').rmdir()
                (fixture / 'renders').symlink_to(outside, target_is_directory=True)
                for command in commands:
                    with mock.patch.object(self.wrapper, 'ROOT', fixture), \
                         mock.patch.object(self.wrapper, 'build') as build, \
                         mock.patch.object(self.wrapper, 'equal') as equal, \
                         mock.patch.object(self.wrapper.importlib.metadata, 'version') as version, \
                         self.assertRaisesRegex(ValueError, 'linked renders'):
                        self.wrapper.main(command)
                    build.assert_not_called()
                    equal.assert_not_called()
                    version.assert_not_called()
                self.assertEqual(list(outside.iterdir()), [])

    def test_farm_jobs_partition_exact_frames_and_preserve_setup_and_branch(self):
        farm = load_module(ROOT / 'cloud/farm.py', 'instep_farm_contract')
        environment = {'NUMBA_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                       'MKL_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1'}
        for kind, (first, last) in KINDS.items():
            output = f'renders/cand_{kind}_in-step'
            path = ROOT / 'cloud/jobs' / f'cand_{kind}_in-step.json'
            args = farm.parse_run_args([str(path), '--nodes', '3', '--dry-run'])
            job = farm.Job(str(path), args)
            self.assertEqual(job.spec['name'], f'cand_{kind}_in-step')
            self.assertEqual(job.spec['branch'], 'claude/owner-night-20260929')
            self.assertEqual(job.spec['frames'], f'{first}-{last}')
            self.assertEqual(job.spec['out_dir'], output)
            self.assertEqual(job.shape, (804, 1920))
            self.assertEqual(job.spec['ship'], 'jpg')
            self.assertFalse(job.whole)
            self.assertEqual(job.skipped_setup, [])
            commands = []
            for command in job.setup + job.spec['render']:
                tokens = shlex.split(command)
                index = tokens.index('python3')
                self.assertEqual(set(tokens[:index]), environment)
                self.assertEqual(tokens[index + 1], 'cloud/cand_render.py')
                commands.append(tokens[index + 2:])
            expected = [[kind, '--prepare']]
            expected += ([[kind, '--equal', str(frame), '--candidate', 'in-step', '--scale', '1']
                          for frame in (3816, 3840, 3847)] if kind == 'cold' else
                         [[kind, '--equal', '4000', '--scale', '1']])
            expected += [[kind, 'in-step', '--range', f'{first}-{last}', '--out', output]]
            self.assertEqual(commands, expected)
            units = farm.plan_units([job], args)
            self.assertEqual(len(units), 3)
            rendered = []
            for unit in units:
                self.assertEqual(unit['branch'], job.spec['branch'])
                self.assertEqual(unit['setup'], job.setup)
                frames = [frame for item in unit['items'] for frame in farm.LaneCmd(item['cmd']).seq]
                self.assertEqual(len(frames), len(set(frames)))
                self.assertEqual(len(unit['outputs']), 1)
                self.assertEqual(unit['outputs'][0]['out_dir'], output)
                self.assertEqual(unit['outputs'][0]['frames'], sorted(frames))
                rendered.extend(frames)
            self.assertEqual(sorted(rendered), list(range(first, last + 1)))


if __name__ == '__main__':
    unittest.main()
