"""D output safety and farm contracts, without renderer or farm execution."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import re
import shlex
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'shots/embers'))
import d_output as O


def farm_lane_parser():
    """Compile the actual farm command parser without importing its network CLI."""
    tree = ast.parse((ROOT / 'cloud/farm.py').read_text())
    constants = {'RX', 'RX_STEP'}
    definitions = {'frames_of', 'compact', 'LaneCmd'}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))
             or (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definitions)]
    scope = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual_farm_parser', 'exec'), scope)
    return scope['LaneCmd']


def d_frame_parser():
    """Use the actual renderer's frame syntax without constructing a scene."""
    tree = ast.parse((ROOT / 'shots/embers/c_d.py').read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'frames')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual_D_frame_parser', 'exec'), scope)
    return scope['frames']


class DFarmJobs(unittest.TestCase):
    THREADS = ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PYTHONDONTWRITEBYTECODE')

    def setUp(self):
        self.jobs = {shot: json.loads((ROOT / 'cloud/jobs' / f'embers_D_{shot}.json').read_text())
                     for shot in O.D.SHOTS}
        self.LaneCmd = farm_lane_parser()
        self.parse_frames = d_frame_parser()

    def test_ranges_cover_current_v2_shots_and_survive_actual_farm_rewriting(self):
        for shot, job in self.jobs.items():
            first, end = O.D.shot_range(shot)

            def oracle(data):
                self.assertEqual(data['frames'], f'{first}-{end - 1}')
                actual = []
                for command in data['render']:
                    lane = self.LaneCmd(command)
                    actual.extend(lane.seq)
                    selected = [lane.seq[0], lane.seq[len(lane.seq) // 2], lane.seq[-1]]
                    rewritten_frames = []
                    for piece in lane.runs(selected):
                        rewritten = lane.render(piece)
                        reparsed = self.LaneCmd(rewritten)
                        self.assertEqual(reparsed.seq, piece)
                        tokens = shlex.split(rewritten)
                        renderer_frames = self.parse_frames(tokens[tokens.index('--range') + 1])
                        self.assertEqual(renderer_frames, piece)
                        rewritten_frames.extend(renderer_frames)
                    self.assertEqual(rewritten_frames, selected)
                self.assertEqual(sorted(actual), list(range(first, end)))

            with self.subTest(shot=shot):
                oracle(job)
                for mutant in ({**job, 'render': []},
                               {**job, 'render': job['render'] + job['render']},
                               {**job, 'frames': f'{first}-{end}'},
                               {**job, 'render': [job['render'][0].replace(f'{first}-{end - 1}', f'{first + 1}-{end - 1}')]}):
                    with self.assertRaises(AssertionError):
                        oracle(mutant)

    def test_names_branch_shape_and_measurement_scope_are_explicit(self):
        for shot, job in self.jobs.items():
            first, end = O.D.shot_range(shot)

            def oracle(data):
                self.assertEqual(data['name'], 'embers_D_' + shot)
                self.assertEqual(data['branch'], 'claude/owner-night-20260929')
                self.assertEqual(data['shape'], [804, 1920])
                self.assertEqual(data['ship'], 'jpg')
                note = data['note']
                self.assertIn('native 1920x804', note)
                self.assertIn('Scene.frame only, excluding scene initialization and PNG encoding', note)
                self.assertIn('RSS is cumulative process high-water', note)
                self.assertIn('not farm throughput estimates', note)
                self.assertNotRegex(note, r'(?<!\w)/(?:[^\s/]+/)+')
                samples = re.findall(r'D(\d+): (\d+\.\d{6}) s/frame, (\d+) bytes peak RSS', note)
                self.assertEqual(len(samples), 2)
                for frame, seconds, rss in samples:
                    self.assertTrue(first <= int(frame) < end)
                    self.assertGreater(float(seconds), 0.)
                    self.assertGreater(int(rss), 0)

            with self.subTest(shot=shot):
                oracle(job)
                for field, wrong in (('name', 'accepted'), ('branch', 'main'), ('shape', [402, 960]),
                                     ('ship', 'mp4'), ('note', 'unmeasured'),
                                     ('note', job['note'] + ' /tmp/nonportable-render-root'),
                                     ('note', job['note'].replace('Scene.frame only, excluding scene initialization and PNG encoding', 'total process runtime')),
                                     ('note', job['note'].replace('RSS is cumulative process high-water', 'RSS is per-frame'))):
                    with self.assertRaises(AssertionError):
                        oracle({**job, field: wrong})

    def assert_environment(self, tokens, enabled):
        for key, value in [('LD_OPEN_RING', str(enabled)), *((key, '1') for key in self.THREADS)]:
            self.assertEqual([token for token in tokens if token.startswith(key + '=')], [f'{key}={value}'])

    def test_render_commands_opt_in_and_target_only_matching_new_native_directories(self):
        for shot, job in self.jobs.items():
            first, end = O.D.shot_range(shot)
            output = 'renders/embers_D_' + shot

            def oracle(data):
                self.assertEqual(data['out_dir'], output)
                self.assertEqual(len(data['render']), 1)
                tokens = shlex.split(data['render'][0])
                self.assert_environment(tokens, 1)
                script = tokens.index('python3')
                self.assertEqual(tokens[script + 1:], [
                    'shots/embers/c_d.py', shot, '--range', f'{first}-{end - 1}',
                    '--out', output, '--format', 'jpg', '--scale', '1'])

            with self.subTest(shot=shot):
                oracle(job)
                for old, new in (('LD_OPEN_RING=1', 'LD_OPEN_RING=0'),
                                 ('LD_OPEN_RING=1', 'LD_OPEN_RING=1 LD_OPEN_RING=0'),
                                 ('NUMBA_NUM_THREADS=1', 'NUMBA_NUM_THREADS=2'),
                                 (output, 'renders/embers_C3'),
                                 ('--scale 1', '--scale .5'),
                                 ('--format jpg', '--format png'),
                                 (f'c_d.py {shot}', 'c_d.py wrong')):
                    mutant = deepcopy(job)
                    mutant['render'] = [command.replace(old, new) for command in mutant['render']]
                    with self.assertRaises(AssertionError):
                        oracle(mutant)
                with self.assertRaises(AssertionError):
                    oracle({**job, 'out_dir': 'renders/accepted'})

    def test_setup_only_verifies_the_pinned_stack_and_assets(self):
        def oracle(data):
            self.assertEqual(len(data['setup']), 1)
            tokens = shlex.split(data['setup'][0])
            self.assert_environment(tokens, 0)
            script = tokens.index('python3')
            self.assertEqual(tokens[script + 1:], ['cloud/cand_render.py', 'trap', '--prepare'])

        for shot, job in self.jobs.items():
            with self.subTest(shot=shot):
                oracle(job)
                for mutant in ({**job, 'setup': []},
                               {**job, 'setup': [job['setup'][0].replace('LD_OPEN_RING=0', 'LD_OPEN_RING=1')]},
                               {**job, 'setup': [job['setup'][0].replace('--prepare', '--range 2320-2320')]},
                               {**job, 'setup': [job['setup'][0] + ' --out renders/embers_C3']}):
                    with self.assertRaises(AssertionError):
                        oracle(mutant)


class DOutputIsolation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / 'repo'
        self.root.mkdir()
        self.renders = self.root / 'renders'
        self.renders.mkdir()

    def assert_good_routes(self, output_dir):
        for shot in O.D.SHOTS:
            expected = self.renders / ('embers_D_' + shot)
            self.assertEqual(output_dir(shot, Path('renders') / expected.name, self.root), expected)
            self.assertEqual(output_dir(shot, expected, self.root), expected)
            external = self.base / 'lookdev' / shot
            self.assertEqual(output_dir(shot, external, self.root), external)

    def assert_bad_routes(self, output_dir):
        for output in ('renders/embers_C3', 'renders/embers_D_race', 'elsewhere',
                       'renders/../renders/embers_D_trap', '../escape/embers_D_trap',
                       self.root, self.root / 'accepted',
                       self.renders / '..' / 'renders' / 'embers_D_trap'):
            with self.assertRaises(ValueError, msg=str(output)):
                output_dir('trap', output, self.root)
        with self.assertRaises(ValueError):
            output_dir('not-a-shot', self.base / 'outside', self.root)

    def test_only_new_matching_repo_directories_or_explicit_external_outputs(self):
        self.assert_good_routes(O.output_dir)
        self.assert_bad_routes(O.output_dir)
        with self.assertRaises(AssertionError):
            self.assert_good_routes(lambda shot, out, root: Path(root) / 'renders' / 'wrong')
        with self.assertRaises(AssertionError):
            self.assert_bad_routes(lambda shot, out, root: Path(out))

    def assert_symlinks_rejected(self, output_dir):
        external = self.base / 'outside'
        external.mkdir(exist_ok=True)
        output = self.renders / 'embers_D_trap'
        output.symlink_to(external, target_is_directory=True)
        try:
            with self.assertRaises(ValueError):
                output_dir('trap', output, self.root)
        finally:
            output.unlink()
        parent = self.base / 'linked-parent'
        parent.symlink_to(external, target_is_directory=True)
        try:
            with self.assertRaises(ValueError):
                output_dir('trap', parent / 'child', self.root)
        finally:
            parent.unlink()
        self.renders.rmdir()
        self.renders.symlink_to(external, target_is_directory=True)
        try:
            # External output does not remove the constructor's cache hazard.
            with self.assertRaises(ValueError):
                output_dir('trap', external / 'lookdev', self.root)
        finally:
            self.renders.unlink()
            self.renders.mkdir()

    def test_symlink_outputs_parents_and_render_cache_root_are_rejected(self):
        self.assert_symlinks_rejected(O.output_dir)
        with self.assertRaises(AssertionError):
            self.assert_symlinks_rejected(lambda shot, out, root: Path(out))

    def assert_files_rejected(self, output_dir):
        file = self.renders / 'embers_D_trap'
        file.write_bytes(b'not a directory')
        try:
            with self.assertRaises(ValueError):
                output_dir('trap', file, self.root)
        finally:
            file.unlink()
        self.renders.rmdir()
        self.renders.write_bytes(b'not a directory')
        try:
            with self.assertRaises(ValueError):
                output_dir('trap', self.base / 'outside', self.root)
        finally:
            self.renders.unlink()
            self.renders.mkdir()

    def test_file_outputs_and_non_directory_cache_root_are_rejected(self):
        self.assert_files_rejected(O.output_dir)
        with self.assertRaises(AssertionError):
            self.assert_files_rejected(lambda shot, out, root: Path(out))


class DAtomicPublication(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.rgb = np.array([[[0., .5, 1.], [1.2, -.1, .4]],
                             [[.7, .2, .9], [.15, .8, .3]]], np.float32)

    def assert_atomic(self, publish, suffix):
        destination = self.directory / ('f_02800' + suffix)
        real_link = O.os.link
        observed = []

        def observe_complete(temporary, target):
            self.assertFalse(Path(target).exists())
            self.assertEqual(Path(temporary).parent, Path(target).parent)
            with Image.open(temporary) as decoded:
                decoded.load()
                self.assertEqual(decoded.size, (2, 2))
                self.assertEqual(decoded.mode, 'RGB')
                self.assertEqual(decoded.format, 'PNG' if suffix == '.png' else 'JPEG')
            observed.append(True)
            return real_link(temporary, target)

        try:
            with patch.object(O.os, 'link', side_effect=observe_complete):
                publish(destination, self.rgb)
            self.assertEqual(observed, [True])
            self.assertEqual(list(self.directory.iterdir()), [destination])
            with Image.open(destination) as decoded:
                decoded.load()
                if suffix == '.png':
                    np.testing.assert_array_equal(np.array(decoded),
                                                  np.rint(np.clip(self.rgb, 0., 1.) * 255).astype(np.uint8))
        finally:
            destination.unlink(missing_ok=True)

    @staticmethod
    def direct_write(path, rgb):
        Image.fromarray(np.rint(np.clip(rgb, 0., 1.) * 255).astype(np.uint8)).save(path)

    def test_complete_encoding_precedes_visibility_for_both_formats(self):
        for suffix in ('.png', '.jpg'):
            with self.subTest(format=suffix):
                self.assert_atomic(O.publish, suffix)
                with self.assertRaises(AssertionError):
                    self.assert_atomic(self.direct_write, suffix)

    def assert_collision_preserves_bytes(self, publish, suffix):
        destination = self.directory / ('f_02800' + suffix)
        original = b'existing frame must survive'
        destination.write_bytes(original)
        try:
            with self.assertRaises(FileExistsError):
                publish(destination, self.rgb)
            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(list(self.directory.iterdir()), [destination])
        finally:
            destination.unlink(missing_ok=True)

    def test_name_collision_does_not_replace_existing_bytes_or_leak_temporary(self):
        for suffix in ('.png', '.jpg'):
            with self.subTest(format=suffix):
                self.assert_collision_preserves_bytes(O.publish, suffix)
                with self.assertRaises(AssertionError):
                    self.assert_collision_preserves_bytes(self.direct_write, suffix)

    def test_dangling_frame_symlink_is_also_an_occupied_name(self):
        def oracle(publish):
            destination = self.directory / 'f_02800.png'
            future_target = self.directory / 'untouched.png'
            destination.symlink_to(future_target)
            try:
                with self.assertRaises(FileExistsError):
                    publish(destination, self.rgb)
                self.assertTrue(destination.is_symlink())
                self.assertFalse(future_target.exists())
                self.assertEqual(list(self.directory.iterdir()), [destination])
            finally:
                destination.unlink(missing_ok=True)
                future_target.unlink(missing_ok=True)

        oracle(O.publish)
        with self.assertRaises(AssertionError):
            oracle(self.direct_write)

    def test_failed_encoding_removes_temporary_and_publishes_nothing(self):
        def oracle(publish):
            with patch.object(Image.Image, 'save', side_effect=OSError('synthetic encoder failure')):
                with self.assertRaises(OSError):
                    publish(self.directory / 'f_02800.png', self.rgb)
            self.assertEqual(list(self.directory.iterdir()), [])

        oracle(O.publish)
        def leaks_temporary(path, rgb):
            (self.directory / '.broken.tmp').write_bytes(b'partial frame')
            raise OSError('synthetic encoder failure')
        try:
            with self.assertRaises(AssertionError):
                oracle(leaks_temporary)
        finally:
            (self.directory / '.broken.tmp').unlink(missing_ok=True)

    def test_nonfinite_or_malformed_images_and_unknown_format_fail(self):
        def oracle(publish):
            for image in (np.full((2, 2, 3), np.nan), np.zeros((2, 2)), np.zeros((0, 2, 3))):
                with self.assertRaises(ValueError):
                    publish(self.directory / 'f_02800.png', image)
            with self.assertRaises(ValueError):
                publish(self.directory / 'f_02800.webp', self.rgb)
            self.assertEqual(list(self.directory.iterdir()), [])

        oracle(O.publish)
        with self.assertRaises(AssertionError):
            oracle(lambda path, rgb: None)


if __name__ == '__main__':
    unittest.main()
