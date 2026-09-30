"""CLI safety, renderer wiring and farm splitting; no actual rendering or farm calls."""
import contextlib
from dataclasses import field, make_dataclass
import importlib.util
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = load_module(ROOT / 'cloud/glowvars_render.py', 'glowvars_cli_contract')
FARM = load_module(ROOT / 'cloud/farm.py', 'glowvars_farm_contract')
Config = make_dataclass('Config', [('beat_offset', int), ('pair_frame', int),
                                 ('cascade_frame', int), ('camera_frame', int, field(default=420))])


class GlowvarsCliTests(unittest.TestCase):
    def args(self, *options):
        return CLI.parser().parse_args(['--variant', 'brink', '--out', 'renders/falsedawn_brink', *options])

    def test_range_endpoints_stride_and_sparse_frame_order(self):
        self.assertEqual(CLI.selected_frames(self.args('--range', '4080-4239')), list(range(4080, 4240)))
        self.assertEqual(CLI.selected_frames(self.args('--range', '4080-4086:3')), [4080, 4083, 4086])
        self.assertEqual(CLI.selected_frames(self.args('--range', '4086-4086')), [4086])
        self.assertEqual(CLI.selected_frames(self.args('--frames', '4239,4080,4100')), [4239, 4080, 4100])

    def test_cut_d_ranges_and_defaults_keep_crowns_lit_before_twofires(self):
        expected = {'brink': (4080, 4239), 'twofires': (4560, 5039),
                    'watch': (7040, 7359), 'truedawn': (7360, 7839)}
        self.assertEqual(CLI.RANGES, expected)
        self.assertEqual(CLI.LENGTHS, {'brink': 160, 'twofires': 480, 'watch': 320, 'truedawn': 480})
        for variant, (first, last) in expected.items():
            with self.subTest(variant=variant):
                args = CLI.parser().parse_args(['--variant', variant, '--frames', f'{first},{last}',
                                               '--out', f'renders/falsedawn_{variant}'])
                self.assertEqual(CLI.selected_frames(args), [first, last])
                self.assertEqual(args.pair_frame, 4320)
                self.assertEqual(args.cascade_frame, 4760)
                self.assertEqual(args.beat_offset, 0)
                self.assertFalse(hasattr(args, 'camera_frame'))
                self.assertFalse(hasattr(args, 'pair_x'))
                self.assertFalse(hasattr(args, 'settle_start'))
                self.assertFalse(hasattr(args, 'settle_end'))
                for bad in (first - 1, last + 1, 0):
                    args.frames = str(bad)
                    with self.assertRaises(ValueError):
                        CLI.selected_frames(args)
        self.assertLess(args.pair_frame, expected['twofires'][0])
        self.assertEqual(expected['watch'][1] + 1, expected['truedawn'][0])

    def test_missing_required_arguments_and_conflicting_selection_rejected(self):
        argv = ['--variant', 'brink', '--range', '4080-4081', '--out', 'renders/falsedawn_brink']
        cases = [argv[2:], argv[:2] + argv[4:], argv[:4], argv + ['--frames', '4080'],
                 ['--variant', 'accepted', '--range', '4080-4081', '--out', 'elsewhere'],
                 argv + ['--settle-start', '4800'], argv + ['--settle-end', '4960'],
                 argv + ['--camera-frame', '420'], argv + ['--pair-x', '.3', '.7']]
        for options in cases:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit), mock.patch.object(CLI, 'load_renderer') as load:
                try:
                    CLI.main(options)
                finally:
                    load.assert_not_called()

    def test_invalid_frames_and_options_rejected_before_renderer_import(self):
        bad_frames = [('--range', '4084-4083'), ('--range', '4080-4240'), ('--range', '4080-4081:0'),
                      ('--range=-1-4',), ('--range', '4080-999999999999999999999'),
                      ('--range', '4080-4081:1.5'), ('--range', '4079-4239'), ('--range', '0-159'),
                      ('--range', '3120-3279'), ('--frames', '3120'),
                      ('--frames', '4240'), ('--frames', '4079'), ('--frames', '4080,4080'),
                      ('--frames', '4080,-1'), ('--frames', '4080,'), ('--frames', '')]
        bad_options = [('--scale', '0'), ('--scale', 'nan'), ('--scale', 'inf'),
                       ('--scale', '1.1'), ('--scale', '0.0001'), ('--ss', 'nan'),
                       ('--ss', '0'), ('--ss', '2.1'), ('--pair-frame', '-1'), ('--cascade-frame', '4320')]
        for options in bad_frames + [('--frames', '4080', *opt) for opt in bad_options]:
            with self.subTest(options=options), mock.patch.object(CLI, 'load_renderer') as load:
                with self.assertRaises(ValueError):
                    CLI.main(['--variant', 'brink', '--out', 'renders/falsedawn_brink', *options])
                load.assert_not_called()

    def test_safe_external_output_allowed_despite_linked_owner_renders(self):
        with tempfile.TemporaryDirectory(prefix='glowvars-path-') as temporary:
            fixture = Path(temporary)
            project, owner, reports = fixture / 'project', fixture / 'owner', fixture / 'reports'
            project.mkdir()
            owner.mkdir()
            (project / 'renders').symlink_to(owner, target_is_directory=True)
            with mock.patch.object(CLI, 'ROOT', project):
                self.assertEqual(CLI.output_directory(reports, 'brink'), reports.resolve())
            self.assertFalse(reports.exists())
            self.assertEqual(list(owner.iterdir()), [])

    def test_positive_ss_that_rounds_a_source_dimension_to_zero_is_rejected(self):
        for scale, ss in (('1', '0.000001'), ('0.001', '0.5')):
            with self.subTest(scale=scale, ss=ss), mock.patch.object(CLI, 'load_renderer') as load:
                with self.assertRaisesRegex(ValueError, 'positive supersampled dimensions'):
                    CLI.main(['--variant', 'brink', '--frames', '4080', '--out', 'renders/falsedawn_brink',
                              '--scale', scale, '--ss', ss])
                load.assert_not_called()
        # The one-pixel target remains supported when its source is nonempty.
        CLI.validate_options(self.args('--frames', '4080', '--scale', '.001', '--ss', '1'))

    def test_owner_renders_rejected_by_relative_absolute_and_alias_paths(self):
        with tempfile.TemporaryDirectory(prefix='glowvars-owner-') as temporary:
            fixture = Path(temporary)
            project, owner = fixture / 'project', fixture / 'owner'
            project.mkdir()
            owner.mkdir()
            (project / 'renders').symlink_to(owner, target_is_directory=True)
            (fixture / 'alias').symlink_to(owner, target_is_directory=True)
            targets = [Path('renders/falsedawn_brink'), owner / 'probe', fixture / 'alias/probe',
                       owner, project / 'renders/probe/../falsedawn_brink']
            with mock.patch.object(CLI, 'ROOT', project), mock.patch.object(CLI, 'load_renderer') as load:
                for target in targets:
                    with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'linked owner'):
                        CLI.main(['--variant', 'brink', '--frames', '4080', '--out', str(target)])
                load.assert_not_called()
            self.assertEqual(list(owner.iterdir()), [])

    def test_wrong_repository_directory_and_output_symlinks_rejected(self):
        with tempfile.TemporaryDirectory(prefix='glowvars-directory-') as temporary:
            fixture = Path(temporary)
            project, external = fixture / 'project', fixture / 'external'
            project.mkdir()
            external.mkdir()
            (project / 'renders').mkdir()
            (fixture / 'linked').symlink_to(external, target_is_directory=True)
            (fixture / 'dangling').symlink_to(fixture / 'missing', target_is_directory=True)
            (fixture / 'file').write_text('preserve')
            targets = [Path('renders/falsedawn_A'), Path('renders/falsedawn_twofires'),
                       Path('shots/run'), fixture / 'linked', fixture / 'dangling', fixture / 'file']
            with mock.patch.object(CLI, 'ROOT', project), mock.patch.object(CLI, 'load_renderer') as load:
                for target in targets:
                    with self.subTest(target=target), self.assertRaises(ValueError):
                        CLI.main(['--variant', 'brink', '--frames', '4080', '--out', str(target)])
                load.assert_not_called()
                self.assertEqual(CLI.output_directory('renders/falsedawn_brink', 'brink'),
                                 (project / 'renders/falsedawn_brink').resolve())
            self.assertFalse((fixture / 'missing').exists())
            self.assertEqual((fixture / 'file').read_text(), 'preserve')

    def test_existing_frame_or_dangling_link_rejected_before_import(self):
        for dangling in (False, True):
            with self.subTest(dangling=dangling), tempfile.TemporaryDirectory(prefix='glowvars-existing-') as temporary:
                out = Path(temporary)
                frame = out / 'f_04080.png'
                if dangling:
                    frame.symlink_to(out / 'absent.png')
                else:
                    frame.write_bytes(b'original')
                with mock.patch.object(CLI, 'load_renderer') as load, self.assertRaises(FileExistsError):
                    try:
                        CLI.main(['--variant', 'brink', '--frames', '4080', '--out', str(out)])
                    finally:
                        load.assert_not_called()

    def test_main_passes_explicit_config_and_reports_paths_time_and_rss(self):
        renderer = mock.Mock(Config=Config)
        with tempfile.TemporaryDirectory(prefix='glowvars-main-') as temporary:
            stdout = io.StringIO()
            out = Path(temporary) / 'probe'
            with mock.patch.object(CLI, 'load_renderer', return_value=renderer), \
                 mock.patch.object(CLI, 'checked_rgb', return_value=mock.sentinel.rgb) as render, \
                 mock.patch.object(CLI, 'save_frame') as save, \
                 mock.patch.object(CLI, 'peak_rss_bytes', return_value=123456), \
                 mock.patch.object(CLI.time, 'perf_counter', side_effect=[10., 12., 15., 18.]), \
                 contextlib.redirect_stdout(stdout):
                CLI.main(['--variant', 'twofires', '--frames', '4564,4567', '--out', str(out),
                          '--scale', '.5', '--ss', '1', '--beat-offset', '-20',
                          '--pair-frame', '4360', '--cascade-frame', '4800'])
            receipts = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual([r['frame'] for r in receipts], [4564, 4567])
            self.assertEqual([r['seconds'] for r in receipts], [2., 3.])
            self.assertEqual([r['process_peak_rss_bytes'] for r in receipts], [123456, 123456])
            config = dict(beat_offset=-20, pair_frame=4360, cascade_frame=4800, camera_frame=420)
            for i, frame in enumerate((4564, 4567)):
                self.assertEqual(receipts[i]['config'], config)
                self.assertEqual(receipts[i]['path'], str(out.resolve() / f'f_{frame:05d}.png'))
                self.assertEqual(render.call_args_list[i].args[1], frame)
                self.assertEqual(render.call_args_list[i].args[3].camera_frame, 420)
                self.assertFalse(hasattr(render.call_args_list[i].args[3], 'pair_x'))
                self.assertEqual(save.call_args_list[i].args,
                                 (renderer.FD.PI.look, out.resolve() / f'f_{frame:05d}.png', mock.sentinel.rgb))

    def test_finish_uses_accepted_grade_and_rejects_bad_arrays(self):
        import numpy as np
        args = self.args('--frames', '4080', '--scale', '.05')
        hdr = np.zeros((40, 96, 3), dtype=np.float32)
        renderer = mock.Mock()
        renderer.FD.FINISH = {'exposure': 1.05}
        renderer.render.return_value = hdr
        renderer.FD.PI.look.finish.return_value = hdr
        self.assertIs(CLI.checked_rgb(renderer, 4080, args, mock.sentinel.config), hdr)
        renderer.render.assert_called_once_with(4080, variant='brink', scale=.05, ss=1.5,
                                                config=mock.sentinel.config)
        renderer.FD.PI.look.finish.assert_called_once_with(hdr, exposure=1.05)
        for bad in (hdr[:, :-1], np.full_like(hdr, np.nan)):
            for stage in ('hdr', 'rgb'):
                with self.subTest(stage=stage), self.assertRaises(ValueError):
                    renderer.render.return_value = bad if stage == 'hdr' else hdr
                    renderer.FD.PI.look.finish.return_value = bad if stage == 'rgb' else hdr
                    CLI.checked_rgb(renderer, 4080, args, mock.sentinel.config)

    def test_checked_dimensions_match_renderer_rounding_at_fractional_scale(self):
        import numpy as np
        args = self.args('--frames', '4080', '--scale', '.333')
        hdr = np.zeros((268, 639, 3), dtype=np.float32)
        renderer = mock.Mock()
        renderer.FD.FINISH = {}
        renderer.render.return_value = hdr
        renderer.FD.PI.look.finish.return_value = hdr
        self.assertIs(CLI.checked_rgb(renderer, 4080, args, mock.sentinel.config), hdr)
        renderer.render.return_value = hdr[:-1]
        with self.assertRaises(ValueError):
            CLI.checked_rgb(renderer, 4080, args, mock.sentinel.config)

    def test_save_publishes_complete_frame_and_refuses_overwrite(self):
        def write(path, rgb):
            Path(path).write_bytes(rgb)

        look = mock.Mock()
        look.save_png.side_effect = write
        with tempfile.TemporaryDirectory(prefix='glowvars-save-') as temporary:
            target = Path(temporary) / 'f_04080.png'
            CLI.save_frame(look, target, b'first')
            with self.assertRaises(FileExistsError):
                CLI.save_frame(look, target, b'second')
            self.assertEqual(target.read_bytes(), b'first')
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_load_refuses_renderer_cli_range_disagreement(self):
        renderer = mock.Mock(RANGES=CLI.RANGES, LENGTHS={**CLI.LENGTHS, 'brink': 159})
        with mock.patch.object(CLI.importlib, 'import_module', return_value=renderer), \
             mock.patch.object(CLI.sys, 'path', list(CLI.sys.path)), \
             self.assertRaisesRegex(RuntimeError, 'frame ranges disagree'):
            CLI.load_renderer()

    def test_load_refuses_shifted_ranges_even_with_equal_lengths(self):
        renderer = mock.Mock(RANGES={**CLI.RANGES, 'brink': (4081, 4240)}, LENGTHS=CLI.LENGTHS)
        with mock.patch.object(CLI.importlib, 'import_module', return_value=renderer), \
             mock.patch.object(CLI.sys, 'path', list(CLI.sys.path)), \
             self.assertRaisesRegex(RuntimeError, 'frame ranges disagree'):
            CLI.load_renderer()

    def test_peak_rss_normalizes_platform_units(self):
        with mock.patch.object(CLI.resource, 'getrusage', return_value=mock.Mock(ru_maxrss=42)):
            for platform, expected in (('darwin', 42), ('linux', 42 * 1024)):
                with self.subTest(platform=platform), mock.patch.object(CLI.sys, 'platform', platform):
                    self.assertEqual(CLI.peak_rss_bytes(), expected)


class GlowvarsFarmTests(unittest.TestCase):
    def test_jobs_split_with_exact_once_coverage_and_cli_compatible_commands(self):
        for variant, (first, last) in CLI.RANGES.items():
            with self.subTest(variant=variant):
                path = ROOT / 'cloud/jobs' / f'falsedawn_{variant}.json'
                args = FARM.parse_run_args([str(path), '--nodes', '3', '--dry-run'])
                job = FARM.Job(str(path), args)
                self.assertFalse(job.whole)
                self.assertEqual(job.spec['branch'], 'claude/owner-night-20260929')
                self.assertEqual(job.spec['shape'], [804, 1920])
                self.assertEqual(job.spec['ship'], 'jpg')
                self.assertEqual(job.spec['out_dir'], f'renders/falsedawn_{variant}')
                self.assertEqual(job.outputs[0]['frames'], list(range(first, last + 1)))
                self.assertEqual(job.skipped_setup, [])
                units = job.units(3, 8)
                self.assertEqual(len(units), 3)
                all_frames = []
                for unit in units:
                    self.assertEqual(unit['branch'], job.spec['branch'])
                    unit_frames = []
                    for item in unit['items']:
                        tokens = shlex.split(item['cmd'])
                        cli_args = CLI.parser().parse_args(tokens[tokens.index('cloud/glowvars_render.py') + 1:])
                        CLI.validate_options(cli_args)
                        self.assertEqual(cli_args.variant, variant)
                        self.assertEqual(cli_args.scale, 1)
                        self.assertEqual(cli_args.ss, 1.5)
                        self.assertEqual(str(cli_args.out), job.spec['out_dir'])
                        unit_frames.extend(CLI.selected_frames(cli_args))
                    self.assertEqual(sorted(unit_frames), unit['outputs'][0]['frames'])
                    all_frames.extend(unit_frames)
                self.assertEqual(sorted(all_frames), list(range(first, last + 1)))
                self.assertEqual(len(all_frames), len(set(all_frames)))


if __name__ == '__main__':
    unittest.main()
