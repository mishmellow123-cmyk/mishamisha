"""Legacy reaching-hand construction; no terrain, JIT kernel or pixel render."""
from pathlib import Path
import sys
import unittest

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(RUN.parent / 'montage'))
import sdfppl as SP


class LegacyReachTests(unittest.TestCase):
    def figure(self, cloth=False, **kwargs):
        scene = SP.Scene()
        result = SP.traveller(scene, np.array([.1, .93, -.2]), np.array([0., 0., 1.]),
                              np.array([-.13, .07, 0.]), np.array([.13, .07, 0.]),
                              (.03, .026, .022), cloth=cloth, **kwargs)
        primitives, objects = scene.arrays()
        self.assertTrue(np.isfinite(primitives).all())
        self.assertTrue(np.isfinite(objects).all())
        return result, primitives, objects

    def assert_glove_at(self, primitives, hand):
        # SDF gloves are the real degenerate round-cones emitted by arm().
        gloves = primitives[(primitives[:, 0] == 0.) &
                            np.all(primitives[:, 1:4] == primitives[:, 4:7], axis=1), 1:4]
        self.assertTrue(np.any(np.linalg.norm(gloves - hand, axis=1) < 1e-10))

    def test_default_legacy_reach_retains_original_length_and_glove_endpoint(self):
        # Before 60e2f13 the legacy branch used the constant 0.62*h; it has no
        # reach_len parameter. Exercise both lantern-side choices of free arm.
        target = np.array([.3, 1.1, 1.4])
        for height in (.8, 1., 1.2):
            for lantern_side in (-1., 0., 1.):
                with self.subTest(height=height, lantern_side=lantern_side):
                    result, primitives, _ = self.figure(reach=target, h=height, lantern_side=lantern_side)
                    shoulder = result['shoulders'][1 if lantern_side < 0 else 0]
                    direction = target - shoulder
                    expected = shoulder + direction / (np.linalg.norm(direction) + 1e-9) * .62 * height
                    np.testing.assert_allclose(result['reach_hand'], expected, rtol=0., atol=1e-12)
                    self.assert_glove_at(primitives, expected)

    def test_legacy_hand_meets_a_target_at_its_reach_distance(self):
        resting, _, _ = self.figure()
        target = resting['shoulders'][0] + np.array([0., -.62, 0.])
        result, primitives, _ = self.figure(reach=target)
        np.testing.assert_allclose(result['reach_hand'], target, rtol=0., atol=1e-8)
        self.assert_glove_at(primitives, result['reach_hand'])

    def test_default_cloth_path_keeps_custom_reach_length(self):
        target = np.array([.3, 1.1, 1.4])
        for length in (.30, .62, .80):
            with self.subTest(length=length):
                result, primitives, _ = self.figure(cloth=True, reach=target, reach_len=length)
                shoulder = result['shoulders'][0]
                direction = target - shoulder
                expected = shoulder + direction / (np.linalg.norm(direction) + 1e-9) * length
                np.testing.assert_allclose(result['reach_hand'], expected, rtol=0., atol=1e-12)
                self.assert_glove_at(primitives, expected)

    def test_carry_targets_take_precedence_over_reach_in_both_paths(self):
        hands = (np.array([-.2, .8, .3]), np.array([.2, .8, .3]))
        for cloth in (False, True):
            with self.subTest(cloth=cloth):
                _, original, objects = self.figure(cloth=cloth, carry=hands)
                result, reaching, reaching_objects = self.figure(cloth=cloth, carry=hands,
                                                                 reach=np.array([.3, 1.1, 1.4]))
                np.testing.assert_array_equal(reaching, original)
                np.testing.assert_array_equal(reaching_objects, objects)
                self.assertNotIn('reach_hand', result)
                for hand in hands:
                    self.assert_glove_at(reaching, hand)

    def test_legacy_without_reach_and_staff_override_still_construct(self):
        result, _, _ = self.figure()
        self.assertNotIn('reach_hand', result)
        staff = np.array([.4, 0., .4])
        _, original, objects = self.figure(staff_tip=staff)
        result, reaching, reaching_objects = self.figure(staff_tip=staff, reach=np.array([.3, 1.1, 1.4]))
        np.testing.assert_array_equal(reaching, original)
        np.testing.assert_array_equal(reaching_objects, objects)
        self.assertNotIn('reach_hand', result)


if __name__ == '__main__':
    unittest.main()
