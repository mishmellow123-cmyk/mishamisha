"""Geometry contracts only; perceptual approval requires the rendered study."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import book_c_v5_deep_candidates as C
import book_c_v5 as V
import v5_inkpages as P
import pen


class DeepCandidates(unittest.TestCase):
    def test_default_dispatch_is_the_original_renderer(self):
        # Avoid page allocations: assert the real routing arguments, not a
        # second wrapper implementation that could silently regrade the image.
        with patch.object(V, 'PagesV5', autospec=True) as original:
            for kwargs in ({}, {'candidate': 'accepted'}):
                self.assertIs(C.make_renderer(**kwargs), original.return_value)
                original.assert_called_with('deep_abandoned', 960, 402, 110)
            C.make_renderer('accepted', scale=1.)
            original.assert_called_with('deep_abandoned', 1920, 804, 110)

    def test_rendering_camera_light_and_grade_path_are_inherited(self):
        for method in ('frame', 'shot_deep_abandoned', 'finish_layer', 'light'):
            self.assertIs(getattr(C.CandidatePagesV5, method), getattr(V.PagesV5, method))
        self.assertIs(C.CandidateDeep.build, P.AbandonedDeep.build)
        self.assertIs(C.CandidateDeep._miner, P.AbandonedDeep._miner)

    def test_every_inherited_mine_stroke_and_rng_result_stays_exact(self):
        original = P.AbandonedDeep()
        expected = original.build('ink', .6, 9.6).pack()
        for name in C.CANDIDATES[1:]:
            artist = C.CandidateDeep(name)
            actual = artist.build('ink', .6, 9.6).pack()
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=f'{name}: {key}')
            np.testing.assert_array_equal(artist.vein, original.vein)
            self.assertEqual(artist.halls, original.halls)

    def test_additions_are_deterministic_finite_ink_and_do_not_mutate_inputs(self):
        for name in C.CANDIDATES[1:]:
            artist = C.CandidateDeep(name)
            vein, halls = artist.vein.copy(), copy.deepcopy(artist.halls)
            state = np.random.get_state()
            first = artist.additions().pack()
            second = artist.additions().pack()
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
                self.assertTrue(np.isfinite(first[key]).all(), key)
            np.testing.assert_array_equal(artist.vein, vein)
            self.assertEqual(artist.halls, halls)
            after = np.random.get_state()
            self.assertEqual(state[0], after[0])
            np.testing.assert_array_equal(state[1], after[1])
            self.assertEqual(state[2:], after[2:])
            self.assertEqual(set(first['L']), {pen.INK})
            self.assertTrue((first['R'] > 0).all())
            self.assertLess(first['R'].max(), .065)
            self.assertGreater(first['P'][:, 0].min(), 2.3)
            self.assertLess(first['P'][:, 0].max(), 17.3)
            self.assertGreater(first['P'][:, 1].min(), 0.)
            self.assertLess(first['P'][:, 1].max(), 26.6)

    def test_ladder_rails_rungs_clear_vaults_walls_and_pillars(self):
        for name in C.CANDIDATES[1:]:
            artist = C.CandidateDeep(name)
            self.assertEqual(len(artist.ladders()), len(artist.halls))
            for ladder in artist.ladders():
                hall = artist.halls[ladder.hall]
                self.assertAlmostEqual(ladder.bottom_y, hall['y1']-.035)
                self.assertGreaterEqual(len(list(ladder.rungs())), 2)
                self.assertLessEqual(abs(ladder.top_x-ladder.bottom_x), .26+1e-12)
                for side in (-1, 1):
                    rail = ladder.rail(side)
                    q = np.linspace(0., 1., 101)[:, None]
                    xy = rail[0] + q*(rail[1]-rail[0])
                    self.assertTrue(artist.void(hall, xy[:, 0], xy[:, 1]).all())
                    self.assertGreater((xy[:, 1]-artist.arch_y(hall, xy[:, 0])).min(), .055)
                    self.assertGreater((xy[:, 0]-hall['x0']).min(), .075)
                    self.assertGreater((hall['x1']-xy[:, 0]).min(), .075)
                for rung in ladder.rungs():
                    np.testing.assert_allclose(rung[:, 1], rung[0, 1])
                    self.assertAlmostEqual(rung[1, 0]-rung[0, 0], 2*ladder.half_width)
                    self.assertTrue(artist.void(hall, rung[:, 0], rung[:, 1]).all())

    def test_lamp_has_round_bail_and_reservoir_at_local_surface(self):
        for name in C.CANDIDATES[1:]:
            artist = C.CandidateDeep(name)
            paths = artist.lamp_paths()
            bail, reservoir = paths[0], paths[7]
            np.testing.assert_allclose(bail[0], bail[-1])
            np.testing.assert_allclose(reservoir[0], reservoir[-1])
            self.assertEqual(len(bail), 65)
            self.assertGreater(np.ptp(bail[:, 0]), .6)
            self.assertGreater(np.ptp(bail[:, 1]), .7)
            x = float(artist.vein[0, 0])-1.35
            self.assertAlmostEqual(reservoir[:, 1].max(), float(artist.surface(x))-.035)
            xy = np.concatenate(paths)
            self.assertLess(np.ptp(xy[:, 0]), 1.4)
            self.assertLess(np.ptp(xy[:, 1]), 1.93)

    def test_study_paths_reject_accepted_finals_and_symlink_aliases(self):
        root = Path.home()/'ldfarm'/'out'
        self.assertEqual(C.validate_output_dir(root/'cand_deep_round_bail'),
                         (root/'cand_deep_round_bail').resolve())
        for path in (root/'book_C5_deep_abandoned', root/'cand_deep_',
                     root/'cand_deep_round_bail'/'nested', Path('cand_deep_test')):
            with self.assertRaises(ValueError):
                C.validate_output_dir(path)
        with tempfile.TemporaryDirectory() as d:
            link = Path(d)/'cand_deep_alias'
            link.symlink_to(root/'book_C5_deep_abandoned')
            with self.assertRaises(ValueError):
                C.validate_output_dir(link)

    def test_invalid_options_fail_before_renderer_construction(self):
        for scale in (0., -1., np.nan, np.inf, .00001):
            with self.assertRaises(ValueError):
                C.make_renderer(scale=scale)
        with self.assertRaises(ValueError):
            C.make_renderer('unknown')


if __name__ == '__main__':
    unittest.main()
