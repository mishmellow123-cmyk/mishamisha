"""Open-ring job contracts and output isolation; no renderer or farm calls."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import posixpath
import re
import shlex
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'embers_C3_forging_open': (1040, 1439, 'renders/embers_C3_open', None, None),
    'embers_C3_race_open': (1440, 1679, 'renders/embers_C3_open', None, None),
    'embers_C5_trap_open': (2320, 2639, 'renders/embers_C5_trap_open', 'trap', 'front_smoke_near'),
    'cand_cold_instep_open': (3816, 3999, 'renders/cand_cold_instep_open', 'cold', 'in-step'),
    'cand_unfinished_instep_open': (4000, 4239, 'renders/cand_unfinished_instep_open', 'unfinished', 'in-step'),
}
THREADS = ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def load_adapter():
    spec = importlib.util.spec_from_file_location('openring_candidate_adapter', ROOT / 'cloud/cand_render.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lane_parser():
    """Use the real farm command parser without importing its daemon/CLI."""
    tree = ast.parse((ROOT / 'cloud/farm.py').read_text())
    names = {'RX', 'RX_STEP'}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
             or (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ('frames_of', 'compact', 'LaneCmd'))]
    scope = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'farm_command_contract', 'exec'), scope)
    return scope['LaneCmd']


class OpenRingJobs(unittest.TestCase):
    def setUp(self):
        self.jobs = {name: json.loads((ROOT / 'cloud/jobs' / (name + '.json')).read_text()) for name in CASES}
        self.LaneCmd = lane_parser()

    def rejects(self, oracle, mutant):
        with self.assertRaises(AssertionError):
            oracle(mutant)

    def test_frames_tile_each_requested_shot_and_remain_farm_rewritable(self):
        for name, (first, last, _, _, _) in CASES.items():
            def oracle(job):
                self.assertEqual(job['frames'], f'{first}-{last}')
                actual = []
                for command in job['render']:
                    lane = self.LaneCmd(command)
                    actual.extend(lane.seq)
                    sample = lane.seq[len(lane.seq) // 2:len(lane.seq) // 2 + 2]
                    self.assertEqual(self.LaneCmd(lane.render(sample)).seq, sample)
                self.assertEqual(sorted(actual), list(range(first, last + 1)))
            with self.subTest(name=name):
                job = self.jobs[name]
                oracle(job)
                missing = deepcopy(job)
                missing['render'] = []
                self.rejects(oracle, missing)
                duplicate = deepcopy(job)
                duplicate['render'].append(job['render'][0])
                self.rejects(oracle, duplicate)
                wrong_span = deepcopy(job)
                wrong_span['frames'] = f'{first}-{last + 1}'
                self.rejects(oracle, wrong_span)

    def test_branch_native_shape_and_names_are_explicit(self):
        for name, job in self.jobs.items():
            def oracle(data):
                self.assertEqual(data['name'], name)
                self.assertEqual(data['branch'], 'claude/owner-night-20260929')
                self.assertEqual(data['shape'], [804, 1920])
                self.assertEqual(data['ship'], 'jpg')
                self.assertNotRegex(data['note'], r'(?<!\w)/(?:[^\s/]+/)+')
            with self.subTest(name=name):
                oracle(job)
                for field, wrong in (('name', 'accepted'), ('branch', 'main'), ('shape', [402, 960]), ('ship', 'mp4'), ('note', '/tmp/nonportable-render-root')):
                    self.rejects(oracle, {**job, field: wrong})

    def test_render_flags_threads_outputs_and_accepted_variants(self):
        for name, (_, _, out, kind, option) in CASES.items():
            def oracle(job):
                self.assertEqual(job['out_dir'], out)
                for command in job['render']:
                    tokens = shlex.split(command)
                    for value in ('LD_OPEN_RING=1', 'PYTHONDONTWRITEBYTECODE=1', *(f'{key}=1' for key in THREADS)):
                        self.assertEqual(tokens.count(value), 1)
                    self.assertIn('--out', tokens)
                    given = tokens[tokens.index('--out') + 1]
                    cwd = tokens[1] if tokens[:1] == ['cd'] else '.'
                    self.assertEqual(posixpath.normpath(posixpath.join(cwd, given)), out)
                    if kind is None:
                        self.assertEqual(tokens[tokens.index('--cut') + 1], 'C3')
                    else:
                        index = tokens.index('cloud/cand_render.py')
                        self.assertEqual(tokens[index + 1:index + 3], [kind, option])
            with self.subTest(name=name):
                job = self.jobs[name]
                oracle(job)
                for original, replacement in (('LD_OPEN_RING=1', 'LD_OPEN_RING=0'), ('NUMBA_NUM_THREADS=1', 'NUMBA_NUM_THREADS=2'), (out, 'renders/accepted')):
                    mutant = deepcopy(job)
                    mutant['render'] = [cmd.replace(original, replacement) for cmd in mutant['render']]
                    self.rejects(oracle, mutant)
                self.rejects(oracle, {**job, 'out_dir': 'renders/accepted'})
                if kind:
                    mutant = deepcopy(job)
                    mutant['render'] = [cmd.replace(f'{kind} {option}', f'{kind} accepted') for cmd in mutant['render']]
                    self.rejects(oracle, mutant)

    def test_candidate_setup_uses_disabled_baseline_equality(self):
        for name, (_, _, _, kind, _) in CASES.items():
            if kind is None:
                continue
            def oracle(job):
                commands = [shlex.split(cmd) for cmd in job['setup']]
                self.assertTrue(any('--prepare' in tokens for tokens in commands))
                equal = [tokens for tokens in commands if '--equal' in tokens]
                expected = [3816, 3840, 3847] if kind == 'cold' else [2479 if kind == 'trap' else 4000]
                self.assertEqual([int(tokens[tokens.index('--equal') + 1]) for tokens in equal], expected)
                for tokens in commands:
                    self.assertIn('LD_OPEN_RING=0', tokens)
                    self.assertNotIn('LD_OPEN_RING=1', tokens)
                for tokens in equal:
                    self.assertEqual(tokens[tokens.index('--scale') + 1], '1')
                    if kind == 'cold':
                        self.assertEqual(tokens[tokens.index('--candidate') + 1], 'in-step')
                    else:
                        self.assertNotIn('--candidate', tokens)
            with self.subTest(name=name):
                job = self.jobs[name]
                oracle(job)
                self.rejects(oracle, {**job, 'setup': []})
                self.rejects(oracle, {**job, 'setup': [cmd.replace('LD_OPEN_RING=0', 'LD_OPEN_RING=1') for cmd in job['setup']]})

    def test_adapter_routes_only_opted_in_matching_new_directories(self):
        module = load_adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'renders'
            directory.mkdir()
            with patch.object(module, 'ROOT', root), patch.object(module, 'writable_renders', return_value=directory):
                def oracle(output_dirs):
                    for name, (_, _, out, kind, option) in CASES.items():
                        if kind is None:
                            continue
                        old = f'renders/cand_{kind}_{option}'
                        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
                            self.assertEqual(output_dirs(kind, option, out), [root / out])
                            with self.assertRaises(ValueError):
                                output_dirs(kind, option, old)
                            with self.assertRaises(ValueError):
                                output_dirs(kind, option, 'renders/cand_wrong_open')
                            wrong_kind = 'cold' if kind != 'cold' else 'unfinished'
                            with self.assertRaises(ValueError):
                                output_dirs(wrong_kind, 'in-step', out)
                        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}):
                            self.assertEqual(output_dirs(kind, option, old), [root / old])
                            with self.assertRaises(ValueError):
                                output_dirs(kind, option, out)
                oracle(module.output_dirs)
                self.rejects(oracle, lambda kind, option, out: [root / out])

    def test_adapter_keeps_parent_and_symlink_escape_guards(self):
        module = load_adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'renders'
            directory.mkdir()
            outside = root / 'outside'
            outside.mkdir()
            with patch.object(module, 'ROOT', root), patch.object(module, 'writable_renders', return_value=directory), \
                 patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
                def oracle(output_dirs):
                    for _, (_, _, out, kind, option) in CASES.items():
                        if kind is None:
                            continue
                        basename = Path(out).name
                        with self.assertRaises(ValueError):
                            output_dirs(kind, option, outside / basename)
                        target = directory / basename
                        target.symlink_to(outside, target_is_directory=True)
                        try:
                            with self.assertRaises(ValueError):
                                output_dirs(kind, option, target)
                        finally:
                            target.unlink()
                oracle(module.output_dirs)
                self.rejects(oracle, lambda kind, option, out: [Path(out)])


if __name__ == '__main__':
    unittest.main()
