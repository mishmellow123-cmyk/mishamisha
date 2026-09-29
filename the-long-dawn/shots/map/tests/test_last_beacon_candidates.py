"""Pure candidate contracts; no atlas generation or renderer construction."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import last_beacon as B
import last_beacon_candidates as C


class TerritoryCandidates(unittest.TestCase):
    def beacon_maps(self):
        return C.territory_maps(B.BEACONS[:, 0], B.BEACONS[:, 1], 1., 0.)

    def test_default_is_direct_original_without_candidate_fields(self):
        image = np.linspace(0, 1, 72, dtype=np.float32).reshape(4, 6, 3)
        with patch.object(B, 'LastBeacon', autospec=True) as constructor:
            constructor.return_value.render_c.return_value = image
            with patch.object(C, 'projected_territories', side_effect=AssertionError('default reached candidate')):
                self.assertIs(C.render(3724), image)
                self.assertIs(C.render(3724, candidate='accepted'), image)
            self.assertEqual(constructor.call_count, 2)
            constructor.return_value.render_c.assert_called_with(3724, .5)
        self.assertIs(C.composite(image, 3724, None, None, 'accepted'), image)

    def test_partition_matches_fixed_map_space_samples_and_event_contract(self):
        # Labels recorded from the accepted partition before candidate changes.
        # Off-beacon points catch joint drift of the delegate and its caller;
        # calling B.kingdom_at again would compare that function with itself.
        points = np.array([(-35, -5), (-35, 30), (-15, -5), (-15, 18),
                           (-15, 35), (0, -5), (0, 12), (0, 33),
                           (15, -5), (15, 15), (15, 35), (40, -5),
                           (40, 20), (40, 38), (8, 10), (-5, 0)], float)
        state = C.territory_maps(points[:, 0], points[:, 1], 1., 0.)
        np.testing.assert_array_equal(state['labels'], [1, 2, 1, 2, 2, 7, 4, 3, 7, 0, 5, 6, 6, 5, 0, 4])
        np.testing.assert_array_equal(self.beacon_maps()['labels'], np.arange(8))
        self.assertEqual((B.F0, B.F1, B.LAST), (3440, 3840, 4))
        np.testing.assert_array_equal(B.IGNITION, [3432, 3645, 3466, 3678, 3784, 3582, 3716, 3521])

    def test_one_dark_hold_and_last_catch_follow_original_exactly(self):
        state = self.beacon_maps()
        for variant in C.VARIANTS[1:]:
            for f in range(3724, 3785):
                field = C.warmth_field(f, state, variant)
                self.assertEqual(field[B.LAST], 0.)
                np.testing.assert_array_equal(np.delete(field, B.LAST), 1.)
            last = [C.warmth_field(f, state, variant)[B.LAST] for f in range(3784, 3793)]
            self.assertTrue(all(a < b for a, b in zip(last, last[1:])))
            # Eight-frame catch, with quarter/half/three-quarter milestones.
            np.testing.assert_allclose(last, [0., .04296875, .15625, .31640625, .5,
                                             .68359375, .84375, .95703125, 1.], atol=3e-8)
            for f in (3792, 3816, 3839):
                np.testing.assert_array_equal(C.warmth_field(f, state, variant), 1.)

    def test_water_and_lakes_never_receive_territory_wash(self):
        x, y = B.BEACONS[:, 0], B.BEACONS[:, 1]
        for land, lake in ((0., 0.), (1., 1.), (.90, 0.)):
            state = C.territory_maps(x, y, land, lake)
            for variant in C.VARIANTS[1:]:
                np.testing.assert_array_equal(C.warmth_field(3839, state, variant), 0.)

    def test_feathering_cannot_leak_into_dark_neighbour(self):
        labels = np.zeros((17, 25), np.uint8)
        labels[:, 12:] = B.LAST
        state = dict(labels=labels, dry=np.ones_like(labels, bool), falloff=np.ones_like(labels, np.float32))
        for variant in C.VARIANTS[1:]:
            f = C.warmth_field(3783, state, variant, 2.5)
            np.testing.assert_array_equal(f[:, 12:], 0.)
            self.assertGreater(f[:, :10].min(), 0.)
            after = C.warmth_field(3792, state, variant, 2.5)
            self.assertGreater(after[:, 15:].min(), 0.)

    def test_falloff_varies_with_distance_but_stays_in_actual_region(self):
        x = np.array([13., 14., 16.]); y = np.array([8., 8., 8.])
        state = C.territory_maps(x, y, 1., 0.)
        np.testing.assert_array_equal(state['labels'], 0)
        self.assertEqual(state['falloff'][0], 1.)
        self.assertTrue(np.all(np.diff(state['falloff']) < 0))
        self.assertTrue(np.all((state['falloff'] >= .45) & (state['falloff'] <= 1.)))
        wash = C.warmth_field(3724, state, 'territory-wash')
        pool = C.warmth_field(3724, state, 'beacon-falloff')
        self.assertTrue(np.any(pool < wash))

    def test_composite_is_deterministic_bounded_and_preserves_inputs_and_ink(self):
        image = np.full((12, 20, 3), .65, np.float32)
        image[4:7, 3:7] = .01
        labels = np.zeros((12, 20), np.uint8); labels[:, 10:] = B.LAST
        maps = dict(labels=labels, dry=np.ones_like(labels, bool), falloff=np.ones_like(labels, np.float32)*.8)
        mask = np.zeros((12, 20), np.float32); mask[1:3, 2:4] = 1
        originals = (image.copy(), labels.copy(), mask.copy())
        results = []
        for v in C.VARIANTS[1:]:
            first = C.composite(image, 3783, maps, mask, v)
            second = C.composite(image, 3783, maps, mask, v)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first[mask == 1], image[mask == 1])
            self.assertTrue(np.isfinite(first).all())
            self.assertGreaterEqual(first.min(), 0); self.assertLessEqual(first.max(), 1)
            self.assertLess(first[5, 5].max(), .025)
            self.assertGreater(first[8, 5].mean(), first[8, 15].mean())
            results.append(first)
        self.assertTrue(np.any(results[0] != results[1]))
        for expected, actual in zip(originals, (image, labels, mask)):
            np.testing.assert_array_equal(expected, actual)

    def test_invalid_frame_scale_and_candidate_stop_before_constructing(self):
        with patch.object(B, 'LastBeacon', side_effect=AssertionError('constructed')):
            for f in (3439, 3840, 3724.5):
                with self.assertRaises(ValueError): C.render(f)
            for scale in (0, -1, 2, np.nan, np.inf, .00001):
                with self.assertRaises(ValueError): C.render(3724, scale=scale)
            with self.assertRaises(ValueError): C.render(3724, candidate='unknown')

    def test_missing_world_refuses_before_any_geography_sample_or_bake(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(B.geo, 'CACHE', directory), patch.object(B.geo, 'sample') as sample:
                with self.assertRaisesRegex(RuntimeError, 'may not generate'):
                    C.projected_territories(3724, 20, 12)
                sample.assert_not_called()

    def test_default_variant_batch_renders_once_and_skips_geography(self):
        image = np.ones((12, 20, 3), np.float32)*.5
        with patch.object(B, 'LastBeacon', autospec=True) as constructor:
            shot = constructor.return_value
            shot.render_c.return_value = image
            with patch.object(C, 'projected_territories', side_effect=AssertionError('candidate path')):
                images, fields = C.render_variants(3724, shot, variants=('accepted',), return_fields=True)
            self.assertIs(images['accepted'], image)
            self.assertIsNone(fields)
            shot.render_c.assert_called_once_with(3724, .5)


if __name__ == '__main__':
    unittest.main()
