"""Actual shared-world checks for the paired-reveal amendment."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import reveal_pair_c_v5 as pair
import render_ink as RI


class PairedReveal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shot=pair.make_shot()

    def test_cut_has240_frames_and_uses_only_original_wide_camera_poses(self):
        self.assertEqual([pair.local_frame(f) for f in range(2880,3120)],list(range(240)))
        self.assertEqual(pair.camera_frame(0),160.)
        self.assertEqual(pair.camera_frame(239),239.)
        for f in (2879,3120):
            with self.assertRaises(ValueError):pair.local_frame(f)

    def test_both_actual_fire_envelopes_catch_at_the_same_frame(self):
        B=self.shot.B
        self.assertEqual(B.shape,(2,7))
        np.testing.assert_array_equal(B[:,3],[0.,0.])
        for f in (-2.,0.,1.,5.,23.,239.):
            np.testing.assert_array_equal(RI.BC.env(f,B[0,3]),RI.BC.env(f,B[1,3]))
        self.assertEqual(RI.BC.env(-2.,0.),(0.,0.,0.))
        self.assertGreater(RI.BC.env(0.,0.)[0],0.)

    def test_near_original_is_unchanged_and_far_sits_on_catalogue_peak(self):
        original=RI.Shot('reveal')
        self.assertEqual(original.B[0,3],-400.)
        np.testing.assert_array_equal(self.shot.B[0,:3],original.B[0,:3])
        cat=np.load(Path(pair.__file__).with_name('summits.npy'))
        far=self.shot.B[1,:3]
        self.assertLess(np.linalg.norm(far[[0,2]]-cat[pair.FAR_SUMMIT,[0,2]]),64.)
        self.assertGreater(np.linalg.norm(far-self.shot.B[0,:3]),5000.)

    def test_sites_are_separate_inside_frame_and_visible_in_real_terrain(self):
        sites=self.shot.B[:,:3]
        for f in (0,40,80,120,160,200,239):
            cam=pair.camera(f)
            u,v,z=cam.project(sites+np.array([0.,1.6,0.]))
            self.assertTrue(np.all(z>0.))
            self.assertTrue(np.all((u>192)&(u<1728)&(v>80)&(v<764)))
            self.assertGreater(abs(u[0]-u[1]),1920*.17)
            for site in sites:
                self.assertTrue(RI.BC._visible(cam,site+np.array([0.,2.5,0.]),self.shot.CR,f/24.))


class FreshProcess(unittest.TestCase):
    def test_two_frames_spawn_sequential_single_frame_children_with_all_options(self):
        with tempfile.TemporaryDirectory() as out:
            argv = ['driver', '--frames', '2880,2881', '--scale', '.25', '--ss', '1.5',
                    '--threads', '2', '--format', 'png', '--out', out, '--fresh-process']
            inherited = []

            def child(*args, **kwargs):
                inherited.append(os.environ.get('LD_INK_CREST_REFINE'))

            with (mock.patch.object(sys, 'argv', argv),
                  mock.patch.dict(os.environ, LD_INK_CREST_REFINE='1'),
                  mock.patch.object(pair.subprocess, 'run', side_effect=child) as run,
                  mock.patch.object(pair, 'make_shot') as setup):
                pair.main()
            expected = [mock.call([sys.executable, str(Path(pair.__file__).resolve()),
                                  '--frames', str(frame), '--scale', '0.25', '--ss', '1.5',
                                  '--threads', '2', '--format', 'png', '--out', out], check=True)
                        for frame in (2880, 2881)]
            self.assertEqual(run.call_args_list, expected)
            self.assertEqual(inherited, ['1', '1'])
            setup.assert_not_called()
            for call in run.call_args_list:
                self.assertNotIn('--fresh-process', call.args[0])

    def test_failed_child_stops_the_batch_and_propagates_failure(self):
        failure = subprocess.CalledProcessError(17, ['render-child'])
        with tempfile.TemporaryDirectory() as out:
            argv = ['driver', '--range', '2880-2882', '--out', out, '--fresh-process']
            with (mock.patch.object(sys, 'argv', argv),
                  mock.patch.object(pair.subprocess, 'run', side_effect=[None, failure]) as run):
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    pair.main()
            self.assertIs(error.exception, failure)
            self.assertEqual(run.call_count, 2)
            self.assertEqual([call.args[0][3] for call in run.call_args_list], ['2880', '2881'])
            self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))

    def test_any_existing_output_is_rejected_before_starting_children(self):
        with tempfile.TemporaryDirectory() as out:
            existing = Path(out) / 'f_02881.png'
            existing.write_bytes(b'prior frame')
            argv = ['driver', '--frames', '2880,2881', '--format', 'png', '--out', out,
                    '--fresh-process']
            with (mock.patch.object(sys, 'argv', argv),
                  mock.patch.object(pair.subprocess, 'run') as run,
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit) as error:
                    pair.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
            self.assertEqual(existing.read_bytes(), b'prior frame')

    def test_direct_default_stays_in_process_and_receipt_only_marks_enabled_fix(self):
        for flag in (None, '0', '1'):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as out:
                argv = ['driver', '--frames', '2880,2881', '--out', out]
                with mock.patch.dict(os.environ):
                    if flag is None:
                        os.environ.pop('LD_INK_CREST_REFINE', None)
                    else:
                        os.environ['LD_INK_CREST_REFINE'] = flag
                    with (mock.patch.object(sys, 'argv', argv),
                          mock.patch.object(pair.subprocess, 'run') as run,
                          mock.patch.object(pair, 'make_shot', side_effect=RuntimeError('setup reached'))):
                        with self.assertRaisesRegex(RuntimeError, 'setup reached'):
                            pair.main()
                run.assert_not_called()
                receipt = json.loads((Path(out) / 'receipt_02880_02881.json').read_text())
                self.assertEqual(receipt['status'], 'failed')
                if flag == '1':
                    self.assertIs(receipt['ink_crest_refine'], True)
                else:
                    self.assertNotIn('ink_crest_refine', receipt)


class CrestFixFarmRecipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        cls.job = json.loads((root / 'cloud/jobs/runC_reveal_pair_v5_fix.json').read_text())
        # Read the real command splitter without importing farm side effects.
        tree = ast.parse((root / 'cloud/farm.py').read_text())
        nodes = [node for node in tree.body
                 if (isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and node.name in ('frames_of', 'LaneCmd'))
                 or (isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id in ('RX', 'RX_STEP')
                             for target in node.targets))]
        scope = {'re': re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'farm_contract', 'exec'), scope)
        cls.LaneCmd = scope['LaneCmd']
        cls.frames_of = staticmethod(scope['frames_of'])

    def assert_fix_recipe(self, job):
        self.assertEqual(job['branch'], 'claude/owner-night-20260929')
        self.assertEqual(job['out_dir'], 'renders/runC_reveal_pair_v5_fix')
        self.assertEqual(job['shape'], [804, 1920])
        self.assertEqual(job['ship'], 'jpg')
        lanes = [self.LaneCmd(command) for command in job['render']]
        self.assertEqual(len(lanes), 4)
        self.assertEqual([len(lane.seq) for lane in lanes], [60] * 4)
        frames = [frame for lane in lanes for frame in lane.seq]
        self.assertEqual(sorted(frames), list(range(2880, 3120)))
        self.assertEqual(self.frames_of(job['frames']), sorted(frames))
        for lane in lanes:
            subset = lane.seq[:3]
            rewritten = lane.render(subset)
            self.assertEqual(self.LaneCmd(rewritten).seq, subset)
            for command in (lane.cmd, rewritten):
                tokens = shlex.split(command)
                self.assertIn('LD_INK_CREST_REFINE=1', tokens)
                self.assertIn('--fresh-process', tokens)
                self.assertIn('shots/run/reveal_pair_c_v5.py', tokens)
                self.assertEqual(tokens[tokens.index('--out') + 1], job['out_dir'])
                self.assertEqual(tokens[tokens.index('--format') + 1], 'png')
                self.assertEqual(tokens[tokens.index('--scale') + 1], '1')
                self.assertEqual(tokens[tokens.index('--ss') + 1], '2')

    def test_all_240_frames_have_one_lane_and_split_commands_keep_fix_options(self):
        self.assert_fix_recipe(self.job)

    def test_recipe_guards_reject_disabled_fix_overlap_and_original_output(self):
        for before, after in (('LD_INK_CREST_REFINE=1', 'LD_INK_CREST_REFINE=0'),
                              (' --fresh-process', ''),
                              ('2940-2999', '2880-2939'),
                              ('renders/runC_reveal_pair_v5_fix', 'renders/runC_reveal_pair_v5')):
            with self.subTest(mutation=before):
                changed = json.loads(json.dumps(self.job).replace(before, after))
                self.assertNotEqual(changed, self.job)
                with self.assertRaises(AssertionError):
                    self.assert_fix_recipe(changed)


if __name__=='__main__':unittest.main()
