"""Pin the watch's absolute-frame and shared-catalogue boundaries without rendering."""
from pathlib import Path
import ast
import json
import re
import shlex
import contextlib
import io
import tempfile
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import watch_c_v5 as watch


class WatchContract(unittest.TestCase):
    def test_full_cut_maps_to_continuous_240_existing_poses(self):
        self.assertEqual([watch.source_frame(f) for f in range(4480, 4720)], list(range(80, 320)))
        for f in (4479, 4720):
            with self.assertRaises(ValueError):
                watch.source_frame(f)

    def test_all_beacons_lit_without_mutating_shared_sites_or_ids(self):
        original = np.array([[1, 2, 3, 40, 1, 5, 900], [7, 8, 9, 280, 1, 6, 1800.]], dtype=np.float64)
        before = original.copy()
        changed = watch.settled_beacons(original)
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(changed[:, [0, 1, 2, 4, 5, 6]], original[:, [0, 1, 2, 4, 5, 6]])
        np.testing.assert_array_equal(changed[:, 3], [-400, -400])
        changed[0, 0] = 99
        np.testing.assert_array_equal(original, before)

    def test_invalid_catalogue_fails_before_rendering(self):
        for invalid in (np.empty((0, 7)), np.ones((2, 6)), np.ones(7)):
            with self.assertRaises(ValueError):
                watch.settled_beacons(invalid)

    def test_farm_recipe_can_split_the_cut_and_emits_runner_pngs(self):
        root = Path(__file__).resolve().parents[3]
        job = json.loads((root / 'cloud/jobs/runC_watch_v5.json').read_text())
        tree = ast.parse((root / 'cloud/farm.py').read_text())
        nodes = [node for node in tree.body if
                 isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ('frames_of', 'LaneCmd') or
                 isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('RX', 'RX_STEP') for t in node.targets)]
        scope = {'re': re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'farm_contract', 'exec'), scope)
        lane = scope['LaneCmd'](job['render'][0])
        self.assertEqual(lane.seq, list(range(4480, 4720)))
        self.assertEqual(scope['LaneCmd'](lane.render([4600, 4601, 4602])).seq, [4600, 4601, 4602])
        self.assertEqual(lane.threads, 4)
        tokens = shlex.split(lane.cmd)
        self.assertEqual(tokens[tokens.index('--format') + 1], 'png')
        self.assertEqual(tokens[tokens.index('--out') + 1], job['out_dir'])
        self.assertEqual(job['ship'], 'jpg')
        self.assertEqual(job['shape'], [804, 1920])

    def test_setup_failure_records_failure_and_prior_receipt_cannot_survive_a_retry(self):
        import reveal_pair_c_v5 as pair
        for driver, first in ((watch,4480),(pair,2880)):
            with self.subTest(driver=driver.__name__), tempfile.TemporaryDirectory() as out:
                args=['driver','--frames',str(first),'--out',out]
                with mock.patch.object(sys,'argv',args), mock.patch.object(driver,'make_shot',side_effect=RuntimeError('setup failed')):
                    with self.assertRaisesRegex(RuntimeError,'setup failed'):
                        driver.main()
                receipt=Path(out)/f'receipt_{first:05d}_{first:05d}.json'
                old=receipt.read_text()
                self.assertEqual(json.loads(old)['status'],'failed')
                self.assertEqual(json.loads(old)['frames'],[])
                with mock.patch.object(sys,'argv',args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        driver.main()
                self.assertEqual(error.exception.code,2)
                self.assertEqual(receipt.read_text(),old)


if __name__ == '__main__':
    unittest.main()
