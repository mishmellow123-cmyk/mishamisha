"""Synthetic depth checks only; importing the audit does not import render modules."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


AUDIT_PATH = Path(__file__).resolve().parents[1] / 'terrain_ceiling_audit.py'
SPEC = importlib.util.spec_from_file_location('terrain_ceiling_audit', AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

SKY = 1e30
CAMERA = np.array([60.0, 172.0, 480.0, 0.5, 0.8, 0.8, -0.5,
                   900.0, 486.0, 215.0, 972.0, 430.0])


def pass_result(depth, ceiling=700.0, camera=None):
    return dict(depth=np.asarray(depth, dtype=np.float64),
                camera=CAMERA.copy() if camera is None else camera.copy(),
                caller_hmax=700.0, effective_hmax=ceiling)


class BoundingBoxTests(unittest.TestCase):
    def test_empty_mask_has_no_box(self):
        self.assertIsNone(AUDIT.bbox(np.zeros((3, 5), dtype=bool)))

    def test_single_pixel_box_uses_xy_and_inclusive_endpoints(self):
        mask = np.zeros((3, 5), dtype=bool)
        mask[2, 4] = True
        self.assertEqual(AUDIT.bbox(mask), [4, 2, 4, 2])

    def test_disconnected_changes_keep_full_extent(self):
        mask = np.zeros((4, 7), dtype=bool)
        mask[0, 5] = True
        mask[3, 1] = True
        self.assertEqual(AUDIT.bbox(mask), [1, 0, 5, 3])


class DepthComparisonTests(unittest.TestCase):
    def test_unchanged_depth_is_clean_even_when_ceilings_differ(self):
        depth = [[SKY, 4.0], [9.0, SKY]]
        report = AUDIT.compare_depth(pass_result(depth), pass_result(depth, 3000.0))
        self.assertEqual(report['shape'], [2, 2])
        self.assertEqual(report['pixels'], 4)
        self.assertEqual(report['changed'], 0)
        self.assertEqual(report['sky_to_terrain'], 0)
        self.assertEqual(report['terrain_to_sky'], 0)
        self.assertIsNone(report['changed_bbox_source'])
        self.assertIsNone(report['recovered_bbox_source'])
        self.assertEqual(report['max_finite_depth_delta_m'], 0.0)
        self.assertEqual(report['camera'], CAMERA.tolist())
        self.assertEqual(report['caller_hmax'], 700.0)
        self.assertEqual(report['baseline_hmax'], 700.0)
        self.assertEqual(report['diagnostic_hmax'], 3000.0)

    def test_known_changes_detect_recovered_geometry_and_finite_depth_error(self):
        # Includes the exact 1e29 sky boundary. The 3 m finite-depth change
        # must determine the finite maximum, never the sky sentinel delta.
        before = pass_result([[SKY, SKY, 12.0, 30.0],
                              [5.0, SKY, SKY, 1e29],
                              [8.0, SKY, 4.0, SKY]])
        after = pass_result([[SKY, 19.0, 15.0, SKY],
                             [7.0, SKY, 2.0, 10.0],
                             [8.0, SKY, 4.0, SKY]], 3000.0)
        old_depth, new_depth = before['depth'].copy(), after['depth'].copy()
        report = AUDIT.compare_depth(before, after)
        self.assertEqual(report['pixels'], 12)
        self.assertEqual(report['changed'], 6)
        self.assertEqual(report['sky_to_terrain'], 3)
        self.assertEqual(report['terrain_to_sky'], 1)
        self.assertEqual(report['changed_bbox_source'], [0, 0, 3, 1])
        self.assertEqual(report['recovered_bbox_source'], [1, 0, 3, 1])
        self.assertEqual(report['max_finite_depth_delta_m'], 3.0)
        np.testing.assert_array_equal(before['depth'], old_depth)
        np.testing.assert_array_equal(after['depth'], new_depth)

    def test_direction_is_preserved_when_conditions_are_reversed(self):
        before = pass_result([[SKY, 10.0, SKY]])
        after = pass_result([[3.0, SKY, 7.0]], 3000.0)
        forward = AUDIT.compare_depth(before, after)
        reverse = AUDIT.compare_depth(after, before)
        self.assertEqual((forward['sky_to_terrain'], forward['terrain_to_sky']), (2, 1))
        self.assertEqual((reverse['sky_to_terrain'], reverse['terrain_to_sky']), (1, 2))
        self.assertEqual(forward['recovered_bbox_source'], [0, 0, 2, 0])
        self.assertEqual(reverse['recovered_bbox_source'], [1, 0, 1, 0])
        self.assertIsNone(forward['max_finite_depth_delta_m'])
        self.assertIsNone(reverse['max_finite_depth_delta_m'])

    def test_sky_sentinel_change_is_not_recovered_geometry(self):
        report = AUDIT.compare_depth(pass_result([[SKY]]), pass_result([[1e29]], 3000.0))
        self.assertEqual(report['changed'], 1)
        self.assertEqual(report['sky_to_terrain'], 0)
        self.assertEqual(report['terrain_to_sky'], 0)
        self.assertEqual(report['changed_bbox_source'], [0, 0, 0, 0])
        self.assertIsNone(report['recovered_bbox_source'])
        self.assertIsNone(report['max_finite_depth_delta_m'])

    def test_shape_mismatch_rejected_even_with_equal_pixel_count(self):
        before = pass_result(np.full((2, 3), SKY))
        after = pass_result(np.full((3, 2), SKY), 3000.0)
        with self.assertRaisesRegex(ValueError, 'camera/depth shape changed'):
            AUDIT.compare_depth(before, after)

    def test_camera_mismatch_rejected_even_with_identical_depth(self):
        before = pass_result([[SKY, 10.0]])
        moved_camera = CAMERA.copy()
        moved_camera[8] += 1.0  # A one-pixel source crop is a different ray set.
        after = pass_result([[SKY, 10.0]], 3000.0, camera=moved_camera)
        with self.assertRaisesRegex(ValueError, 'camera/depth shape changed'):
            AUDIT.compare_depth(before, after)

    def test_nonfinite_depth_rejected_in_either_condition(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            for reverse in (False, True):
                with self.subTest(bad=bad, reverse=reverse):
                    pair = [pass_result([[bad]]), pass_result([[SKY]], 3000.0)]
                    with self.assertRaisesRegex(ValueError, 'nonfinite depth'):
                        AUDIT.compare_depth(*(pair[::-1] if reverse else pair))


if __name__ == '__main__':
    unittest.main()
