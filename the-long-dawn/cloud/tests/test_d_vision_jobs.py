"""Cut-D job coverage, real farm parser compatibility and output isolation."""
import ast
from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CASES = {'inscription': (1680, 2079), 'brink': (2960, 3199),
         'vision': (3200, 3439), 'gap': (3440, 3519), 'crowns': (4240, 4559)}
THREADS = ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def lane_parser():
    tree = ast.parse((ROOT / 'cloud/farm.py').read_text())
    names = {'RX', 'RX_STEP'}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                                  for t in node.targets)) or
             (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and
              node.name in ('frames_of', 'compact', 'LaneCmd'))]
    scope = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'farm_command_contract', 'exec'), scope)
    return scope['LaneCmd']


def load_driver():
    spec = importlib.util.spec_from_file_location('cut_d_test_driver', ROOT / 'shots/embers/render_d.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CutDJobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = {shot: json.loads((ROOT / 'cloud/jobs' / f'embers_D_{shot}.json').read_text())
                    for shot in CASES}
        cls.LaneCmd = lane_parser()

    def test_four_interleaved_lanes_cover_every_frame_once_and_can_be_rewritten(self):
        for shot, (first, last) in CASES.items():
            def oracle(job):
                self.assertEqual(job['frames'], f'{first}-{last}')
                self.assertEqual(len(job['render']), 4)
                seen = []
                for offset, command in enumerate(job['render']):
                    lane = self.LaneCmd(command)
                    self.assertEqual(lane.seq, list(range(first + offset, last + 1, 4)))
                    seen.extend(lane.seq)
                    sample = lane.seq[2:5]
                    self.assertEqual(self.LaneCmd(lane.render(sample)).seq, sample)
                self.assertEqual(sorted(seen), list(range(first, last + 1)))
            job = self.jobs[shot]
            with self.subTest(shot=shot):
                oracle(job)
                missing = {**job, 'render': job['render'][:-1]}
                duplicate = {**job, 'render': [job['render'][0]] * 4}
                positional = {**job, 'render': [cmd.replace('--frames ', '') for cmd in job['render']]}
                for wrong in (missing, duplicate, positional,
                              {**job, 'frames': f'{first}-{last + 1}'}):
                    with self.assertRaises((AssertionError, ValueError)):
                        oracle(wrong)

    def test_native_shape_branch_and_isolated_output_names(self):
        for shot, job in self.jobs.items():
            def oracle(data):
                self.assertEqual(data['name'], f'embers_D_{shot}')
                self.assertEqual(data['branch'], 'claude/owner-night-20260929')
                self.assertEqual(data['shape'], [804, 1920])
                self.assertEqual(data['out_dir'], f'renders/embers_D_{shot}')
            with self.subTest(shot=shot):
                oracle(job)
                for key, value in (('name', 'accepted'), ('branch', 'main'),
                                   ('shape', [402, 960]), ('out_dir', 'renders/embers_C3')):
                    with self.assertRaises(AssertionError):
                        oracle({**job, key: value})

    def test_explicit_shot_output_and_single_thread_limits(self):
        for shot, job in self.jobs.items():
            def oracle(data):
                for command in data['render']:
                    words = shlex.split(command)
                    self.assertIn('python3', words)
                    self.assertIn('LD_OPEN_RING=1', words)
                    self.assertIn('shots/embers/render_d.py', words)
                    self.assertEqual(words[words.index('--shot') + 1], shot)
                    self.assertEqual(words[words.index('--out') + 1], f'renders/embers_D_{shot}')
                    for key in THREADS:
                        self.assertEqual(words.count(f'{key}=1'), 1)
            with self.subTest(shot=shot):
                oracle(job)
                for original, replacement in (('LD_OPEN_RING=1', 'LD_OPEN_RING=0'), ('python3', 'python2'),
                                               ('shots/embers/render_d.py', 'shots/embers/render.py'),
                                               (f'--shot {shot}', '--shot accepted'),
                                               (f'renders/embers_D_{shot}', 'renders/accepted'),
                                               *((f'{key}=1', f'{key}=2') for key in THREADS)):
                    wrong = deepcopy(job)
                    wrong['render'] = [cmd.replace(original, replacement) for cmd in job['render']]
                    with self.assertRaises(AssertionError):
                        oracle(wrong)


class CutDDriverGuards(unittest.TestCase):
    def test_direct_entrypoints_reject_accepted_outputs_before_constructing_scene(self):
        load_driver()  # Install the same local import path as the production CLI.
        import d_crowns
        import d_vision
        for module, frame in ((d_crowns, 4320), (d_vision, 3360)):
            def oracle(render):
                with self.assertRaises(ValueError):
                    render(frame, outdir=ROOT / 'renders' / 'embers_A3', save=True)
            with self.subTest(module=module.__name__):
                oracle(module.render_frame)
                with self.assertRaises(AssertionError):
                    oracle(lambda *args, **kwargs: None)

    def test_v2_preroll_and_shot_keep_absolute_filenames_without_translation(self):
        import numpy as np
        driver = load_driver()
        import look
        calls = []
        def render(frame, **kwargs):
            calls.append(frame)
            return np.zeros((2, 4, 3), np.float32)
        renderer = SimpleNamespace(render_frame=render)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(driver.importlib, 'import_module', return_value=renderer), \
             redirect_stdout(io.StringIO()):
            driver.main(['--frames', '1680,1759,1760,2079', '--shot', 'inscription', '--out', temporary])
            def oracle(frames, files):
                self.assertEqual(frames, [1680, 1759, 1760, 2079])
                self.assertEqual(files, ['f_01680.png', 'f_01759.png', 'f_01760.png', 'f_02079.png'])
            files = sorted(p.name for p in Path(temporary).glob('*.png'))
            oracle(calls, files)
            with self.assertRaises(AssertionError):
                oracle([f - 1040 for f in calls], files)
            with self.assertRaises(SystemExit):
                driver.main(['--frames', '640', '--shot', 'inscription', '--out', temporary])

    def test_frame_parser_rejects_duplicate_reversed_and_zero_stride_requests(self):
        driver = load_driver()
        def oracle(parse):
            self.assertEqual(parse('2240-2252:4,2255'), [2240, 2244, 2248, 2252, 2255])
            for invalid in ('2240,2240', '2250-2240', '2240-2250:0', '2240-2250:-1', ''):
                with self.assertRaises(ValueError):
                    parse(invalid)
        oracle(driver.parse_frames)
        with self.assertRaises(AssertionError):
            oracle(lambda value: [2240, 2244, 2248, 2252, 2255])

    def test_output_guard_allows_only_matching_new_farm_dirs_and_external_probes(self):
        driver = load_driver()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root, external = base / 'repo', base / 'probe'
            (root / 'renders').mkdir(parents=True)
            external.mkdir()
            with patch.object(driver, 'ROOT', root):
                def oracle(output):
                    for shot in CASES:
                        target = root / 'renders' / f'embers_D_{shot}'
                        self.assertEqual(output(shot, target), target)
                        self.assertEqual(output(shot, external), external)
                        for bad in (root / 'renders', root / 'renders' / 'accepted',
                                    root / 'renders' / 'embers_D_wrong', root / 'local_probe'):
                            with self.assertRaises(ValueError):
                                output(shot, bad)
                oracle(driver.output_path)
                with self.assertRaises(AssertionError):
                    oracle(lambda shot, path: Path(path))

    def test_output_guard_rejects_shared_render_symlinks_and_direct_targets(self):
        driver = load_driver()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root, shared = base / 'repo', base / 'shared'
            root.mkdir()
            shared.mkdir()
            (root / 'renders').symlink_to(shared, target_is_directory=True)
            with patch.object(driver, 'ROOT', root):
                def oracle(output):
                    for path in (root / 'renders' / 'embers_D_vision', shared / 'embers_D_vision'):
                        with self.assertRaises(ValueError):
                            output('vision', path)
                oracle(driver.output_path)
                with self.assertRaises(AssertionError):
                    oracle(lambda shot, path: Path(path))
            (root / 'renders').unlink()
            (root / 'renders').mkdir()
            target = root / 'renders' / 'embers_D_vision'
            target.symlink_to(shared, target_is_directory=True)
            with patch.object(driver, 'ROOT', root), self.assertRaises(ValueError):
                driver.output_path('vision', target)


if __name__ == '__main__':
    unittest.main()
