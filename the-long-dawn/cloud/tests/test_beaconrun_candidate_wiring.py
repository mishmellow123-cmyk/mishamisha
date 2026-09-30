"""A14/A15 farm contracts, using stubs: no renderer or farm request is started."""
import contextlib
import importlib.util
import io
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
KINDS = {'beaconrun': (3920, 4239), 'watchers': (4240, 4399)}
ASSETS = ('shots/run/summits.npy', 'shots/run/beaconrun_a_chain.npy',
          'shots/run/reveal_a_fires.npy', 'shots/run/watchers_a_extra.npy',
          'shots/run/watchers_a_figs.npy')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_wrapper():
    return load_module(ROOT / 'cloud/cand_render.py', 'beaconrun_adapter_test')


class BeaconrunCandidateWiringTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = load_wrapper()

    def test_both_kinds_load_run_family(self):
        for kind in KINDS:
            self.wrapper._FAMILY = None
            with mock.patch.object(sys, 'path', list(sys.path)):
                self.wrapper.imports(kind)
                self.assertEqual(self.wrapper._FAMILY, 'run')
                self.assertEqual(sys.path[0], str(ROOT / 'shots/run'))

    def test_routes_preserve_default_and_explicit_opt_in(self):
        for kind, (first, _) in KINDS.items():
            base, watcher, candidate = mock.Mock(), mock.Mock(), mock.Mock()
            base.render.return_value = ('a14-hdr', 'camera', 'distance')
            base.FINISH = {'exposure': 1.0}
            base.PI.look.finish.return_value = 'a14-rgb'
            watcher.render.return_value = 'a15-hdr'
            watcher.finish.return_value = 'a15-rgb'
            candidate.render.return_value = 'candidate-rgb'
            modules = {'beaconrun_a': base, 'watchers_a': watcher,
                       'beaconrun_candidates': candidate}
            with mock.patch.dict(sys.modules, modules), mock.patch.object(self.wrapper, 'imports'):
                result = self.wrapper.build(kind, 'original', scale=.5)(first)
                if kind == 'beaconrun':
                    self.assertEqual(result, {'rgb': 'a14-rgb'})
                    base.render.assert_called_once_with(first, scale=.5, ss=1.5)
                    base.PI.look.finish.assert_called_once_with('a14-hdr', exposure=1.0)
                else:
                    self.assertEqual(result, {'rgb': 'a15-rgb'})
                    watcher.render.assert_called_once_with(first, scale=.5, ss=1.5)
                    watcher.finish.assert_called_once_with('a15-hdr')
                candidate.render.assert_not_called()
                for route, option, keywords in (
                        ('default', None, {}), ('shared', None, {'candidate': 'accepted'}),
                        ('candidate', 'linked-fires', {'candidate': 'linked-fires'})):
                    self.assertEqual(self.wrapper.build(kind, route, option, scale=.5)(first),
                                     {'rgb': 'candidate-rgb'})
                    candidate.render.assert_called_once_with(first, kind=kind, scale=.5, ss=1.5, **keywords)
                    candidate.render.reset_mock()

    def test_equality_checks_default_and_shared(self):
        import numpy as np
        for kind, (first, _) in KINDS.items():
            rgb = np.zeros((1, 1, 3), np.float32)
            with mock.patch.object(self.wrapper, 'build') as build, \
                 mock.patch.object(self.wrapper, 'checked', return_value={'rgb': rgb}), \
                 contextlib.redirect_stdout(io.StringIO()):
                proof = self.wrapper.equal(kind, first, scale=.5)
            self.assertEqual(build.call_args_list, [
                mock.call(kind, route, scale=.5) for route in ('original', 'default', 'shared')])
            self.assertEqual(set(proof['proofs']), {'default', 'shared'})

    def test_frame_boundaries_and_candidate_validation(self):
        for kind, (first, last) in KINDS.items():
            self.assertEqual(self.wrapper.RANGES[kind], (first, last))
            self.wrapper.validate_frames(kind, first, last)
            for bad in ((first - 1, last), (first, last + 1), (last, first)):
                with self.assertRaises(ValueError):
                    self.wrapper.validate_frames(kind, *bad)
            with mock.patch.object(self.wrapper, 'imports') as imports:
                for option in (None, 'accepted', 'night-fire'):
                    with self.assertRaises(ValueError):
                        self.wrapper.build(kind, 'candidate', option)
                imports.assert_not_called()

    def test_prepare_requires_every_shared_input(self):
        for kind in KINDS:
            self.assertEqual(self.wrapper.ASSETS[kind], ASSETS)
            for missing in ASSETS:
                with tempfile.TemporaryDirectory(prefix='beaconrun-assets-') as temporary:
                    root = Path(temporary)
                    for name in ASSETS:
                        if name != missing:
                            target = root / name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(b'tracked input stub')
                    with mock.patch.object(self.wrapper, 'ROOT', root), \
                         mock.patch.object(sys, 'version_info', (3, 12)), \
                         mock.patch.object(self.wrapper.importlib.metadata, 'version',
                                           side_effect=self.wrapper.STACK.__getitem__), \
                         self.assertRaisesRegex(RuntimeError, 'Required tracked input is missing'):
                        self.wrapper.prepare(kind)

    def test_paired_jobs_partition_exact_shot_ranges(self):
        farm = load_module(ROOT / 'cloud/farm.py', 'beaconrun_farm_test')
        covered = []
        for kind, (first, last) in KINDS.items():
            path = ROOT / 'cloud/jobs' / f'cand_{kind}_linked-fires.json'
            args = farm.parse_run_args([str(path), '--nodes', '3', '--dry-run'])
            job = farm.Job(str(path), args)
            self.assertEqual(job.spec['branch'], 'claude/owner-night-20260929')
            self.assertEqual(job.shape, (804, 1920))
            self.assertEqual(job.spec['ship'], 'jpg')
            self.assertEqual(job.skipped_setup, [])
            setup = [shlex.split(command) for command in job.setup]
            setup = [tokens[tokens.index('cloud/cand_render.py') + 1:] for tokens in setup]
            self.assertEqual(setup, [[kind, '--prepare'], [kind, '--equal',
                              str(last if kind == 'beaconrun' else first), '--scale', '1']])
            output = f'renders/cand_{kind}_linked-fires'
            self.assertEqual([(o['out_dir'], o['frames']) for o in job.outputs],
                             [(output, list(range(first, last + 1)))])
            tokens = shlex.split(job.spec['render'][0])
            self.assertEqual(tokens[tokens.index('cloud/cand_render.py') + 1:],
                             [kind, 'linked-fires', '--range', f'{first}-{last}', '--out', output])
            units = farm.plan_units([job], args)
            self.assertEqual(len(units), 3)
            rendered = [frame for unit in units for item in unit['items']
                        for frame in farm.LaneCmd(item['cmd']).seq]
            self.assertEqual(sorted(rendered), list(range(first, last + 1)))
            self.assertEqual(len(rendered), len(set(rendered)))
            covered.extend(rendered)
        self.assertEqual(sorted(covered), list(range(3920, 4400)))

    def test_cli_output_isolation_before_renderer_import(self):
        class StopBeforeRender(Exception):
            pass

        for kind, (first, last) in KINDS.items():
            with tempfile.TemporaryDirectory(prefix='beaconrun-output-') as temporary:
                root = Path(temporary)
                accepted = root / 'renders' / ('beaconrun_A_catches3' if kind == 'beaconrun'
                                               else 'watchers_A_catches3')
                accepted.mkdir(parents=True)
                marker = accepted / 'untouched'
                marker.write_bytes(b'accepted take')
                out = f'renders/cand_{kind}_linked-fires'
                with mock.patch.object(self.wrapper, 'ROOT', root), \
                     mock.patch.object(self.wrapper, 'build', side_effect=StopBeforeRender) as build:
                    for wrong in (str(accepted), 'renders/cand_other_linked-fires'):
                        with self.assertRaises(ValueError):
                            self.wrapper.main([kind, 'linked-fires', '--range',
                                               f'{first}-{last}', '--out', wrong])
                        build.assert_not_called()
                    with self.assertRaises(StopBeforeRender):
                        self.wrapper.main([kind, 'linked-fires', '--range',
                                           f'{first}-{last}', '--out', out])
                    build.assert_called_once_with(kind, 'candidate', 'linked-fires')
                self.assertEqual(marker.read_bytes(), b'accepted take')
                self.assertEqual(list((root / out).iterdir()), [])
            with tempfile.TemporaryDirectory(prefix='beaconrun-linked-') as temporary:
                root = Path(temporary) / 'project'
                target = Path(temporary) / 'owner-renders'
                root.mkdir()
                target.mkdir()
                (root / 'renders').symlink_to(target, target_is_directory=True)
                with mock.patch.object(self.wrapper, 'ROOT', root), \
                     mock.patch.object(self.wrapper, 'build') as build, \
                     self.assertRaises(ValueError):
                    self.wrapper.main([kind, 'linked-fires', '--range',
                                       f'{first}-{last}', '--out', out])
                build.assert_not_called()
                self.assertEqual(list(target.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
