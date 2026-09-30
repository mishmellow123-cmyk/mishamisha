"""Freeze the inscription -> forging hand-off at D v2 frame 2080.

These literals are the published ``openring.forging-opening.v1`` contract in
``O/codex/openring/forging_open_contract.json``. Keep them independent of scene
constants: deriving the expectation from the renderer would hide a moved cut.
Only the deterministic layout is constructed; no tower meshes are loaded.
"""
from copy import copy
import unittest
from unittest.mock import patch

import numpy as np

from test_c_d import C, scene_for


FRAME = 2080
CAMERA_POSITION = [-12.896215704688116, 18.0, 149.44459716061385]
CAMERA_TARGET = [0.0, 13.0, 0.0]
CAMERA_HFOV = 62.0
CAMERA_FOCUS = 152.5909564816998
CAMERA_APERTURE = 0.012
BAND_CENTRE = [0.0, 38.0, 0.0]
BAND_ROTATION = [
    [-0.6872480717197039, -0.5744632015400474, 0.44461457240390556],
    [-0.6655996975676453, 0.7431448254773944, -0.06864991597979037],
    [-0.29097616829460116, -0.343114847307038, -0.8930873815266119],
]
BAND_SCALE = 2.2
BAND_GAP_DEGREES = 120.0
BAND_THETA_RANGE = [-0.5999999999999996, 3.588790204786391]
FIRE_POSITION = [0.0, -14.0, 0.0]
FIRE_TIP = [0.0, -7.5, 0.0]


class ForgingOpeningContract(unittest.TestCase):
    def rejects(self, oracle, *args):
        with self.assertRaises(AssertionError):
            oracle(*args)

    def assert_camera(self, camera):
        np.testing.assert_allclose(camera.pos, CAMERA_POSITION, rtol=0., atol=1e-12)
        np.testing.assert_allclose(camera.target, CAMERA_TARGET, rtol=0., atol=1e-12)
        self.assertEqual(camera.hfov, CAMERA_HFOV)
        self.assertAlmostEqual(camera.focus, CAMERA_FOCUS, places=12)
        self.assertEqual(camera.aperture, CAMERA_APERTURE)

    def test_actual_opening_camera_and_lens_match_published_contract(self):
        camera = scene_for('forging').camera(FRAME)
        self.assert_camera(camera)
        for field, wrong in (
            ('pos', camera.pos + [0., 0., .01]),
            ('target', camera.target + [0., .01, 0.]),
            ('hfov', camera.hfov + .01),
            ('focus', camera.focus + .01),
            ('aperture', camera.aperture + .001),
        ):
            with self.subTest(field=field):
                mutated = copy(camera)
                setattr(mutated, field, wrong)
                self.rejects(self.assert_camera, mutated)

    def assert_band(self, pose):
        rotation, centre, scale = pose
        np.testing.assert_allclose(rotation, BAND_ROTATION, rtol=0., atol=1e-12)
        np.testing.assert_allclose(centre, BAND_CENTRE, rtol=0., atol=1e-12)
        self.assertEqual(scale, BAND_SCALE)

    def test_actual_opening_band_pose_matches_published_contract(self):
        pose = scene_for('forging').ring_frame(FRAME)
        self.assert_band(pose)
        rotation, centre, scale = pose
        moved_rotation = rotation.copy()
        moved_rotation[0, 0] += .001
        for mutated in (
            (moved_rotation, centre, scale),
            (rotation, centre + [0., .01, 0.], scale),
            (rotation, centre, scale + .001),
        ):
            self.rejects(self.assert_band, mutated)

    def test_open_band_is_already_present_at_the_frozen_cut(self):
        def oracle():
            self.assertEqual(C.D.SHOTS['forging'].d_start, FRAME)
            self.assertEqual(C.D.gap_deg(FRAME), BAND_GAP_DEGREES)
            np.testing.assert_allclose(C.D.theta_range(FRAME), BAND_THETA_RANGE,
                                       rtol=0., atol=1e-12)
        oracle()
        with patch.object(C.D, 'gap_deg', return_value=119.):
            self.rejects(oracle)
        with patch.object(C.D, 'theta_range', return_value=(-.6, 3.6)):
            self.rejects(oracle)

    def test_central_fire_root_and_tip_match_published_contract(self):
        def oracle():
            root = C.c3.FIRE_ROOT
            tip = root + [0., C.c3.HF, 0.]
            np.testing.assert_allclose(root, FIRE_POSITION, rtol=0., atol=1e-12)
            np.testing.assert_allclose(tip, FIRE_TIP, rtol=0., atol=1e-12)
        oracle()
        with patch.object(C.c3, 'FIRE_ROOT', C.c3.FIRE_ROOT + [.01, 0., 0.]):
            self.rejects(oracle)
        with patch.object(C.c3, 'HF', C.c3.HF + .01):
            self.rejects(oracle)

    def test_holdout_first_live_contact_is_exactly_d5496(self):
        def oracle(scene):
            self.assertEqual(C.D.HOLDOUT_STRIKE, 5496)
            for frame, contact in ((5496. - .001, False), (5496., True)):
                points, energy, owners = scene.spark_paths(frame)
                live = (energy > 0.) & (owners == C.HOLDOUT)
                self.assertTrue(live.any())
                distances = np.linalg.norm(
                    points[live, None, :] - scene.ends(frame)[None, :, :], axis=2
                ).min(axis=1)
                if contact:
                    self.assertLess(distances.min(), 1e-5)
                else:
                    self.assertGreater(distances.min(), 1e-5)

        scene = scene_for('holdout')
        oracle(scene)
        delayed = scene_for('holdout')
        delayed.spark_delay[C.HOLDOUT] += .1
        self.rejects(oracle, delayed)


if __name__ == '__main__':
    unittest.main()
