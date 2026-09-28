"""Near-flame volume contracts; NUMBA_DISABLE_JIT=1 runs without compilation.

These synthetic checks do not establish the finished shot's visual quality.
Run the same suite with JIT enabled when the render queue allocates compute.
"""
from pathlib import Path
import sys
import types
import unittest

import cv2
import numpy as np

cv2.setNumThreads(0)
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(RUN.parent / 'montage'))
import fire_near_a as FN


class NearFireTests(unittest.TestCase):
    seed = 314

    def render(self, depth=1e9, t=2., base=None, camera_offset=None, background=0., **kwargs):
        base = np.zeros(3) if base is None else np.asarray(base, np.float64)
        camera_offset = np.zeros(3) if camera_offset is None else np.asarray(camera_offset, np.float64)
        C = np.array([0., .46, -4., 0., 1., 1., 0., 35., 3.5, 3.5, 7., 7.])
        C[:3] += camera_offset
        camera = types.SimpleNamespace(params=lambda: C)
        image = np.full((7, 7, 3), background, np.float32)
        zb = np.full((7, 7), depth, np.float32)
        FN.draw(image, zb, camera, base, 1., .5, t, seed=self.seed, **kwargs)
        np.testing.assert_array_equal(zb, np.full_like(zb, depth))
        return image

    def test_real_volume_emits_finite_warm_light_and_is_deterministic(self):
        first = self.render()
        np.testing.assert_array_equal(first, self.render())
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(float(first.max()), 0.)
        lit = first[..., 0] > 0.
        self.assertTrue(np.all(first[..., 0][lit] > first[..., 1][lit]))
        self.assertTrue(np.all(first[..., 1][lit] > first[..., 2][lit]))

    def test_actual_scene_depth_can_hide_or_cut_through_the_volume(self):
        full = self.render()
        partial = self.render(depth=4.)
        hidden = self.render(depth=2.)
        np.testing.assert_array_equal(hidden, np.zeros_like(hidden))
        self.assertGreater(float(partial.sum()), 0.)
        self.assertLess(float(partial.sum()), float(full.sum()))

    def test_oblique_ray_uses_horizontal_forward_depth(self):
        # This downward ray reaches visible flame before the z=0 scene plane.
        # Treating its 2 m forward depth as ray distance clips before any flame.
        C = np.array([0., 3., -2., 0., 1., 1., 0., 10., 3.5, -12.5, 7., 1.])
        image = np.zeros((1, 7, 3), np.float32)
        zb = np.full((1, 7), 2., np.float32)
        FN.draw(image, zb, types.SimpleNamespace(params=lambda: C), [0., 0., 0.],
                1., .5, 2., seed=self.seed)
        self.assertGreater(float(image.sum()), 0.)
        np.testing.assert_array_equal(zb, np.full_like(zb, 2.))

    def test_translation_of_camera_and_fuel_preserves_pixels(self):
        shift = np.array([1000., -200., -550.])
        np.testing.assert_allclose(self.render(base=shift, camera_offset=shift), self.render(), atol=2e-6, rtol=0.)

    def test_extinguished_and_zero_transmission_draws_are_noops(self):
        for kwargs in ({'I': 0.}, {'strength': 0.}, {'trans': 0.}):
            np.testing.assert_array_equal(self.render(**kwargs), np.zeros((7, 7, 3), np.float32))

    def test_transmission_is_applied_once(self):
        background = .12
        full = self.render(background=background)
        quarter = self.render(background=background, trans=.25)
        np.testing.assert_allclose(quarter - background, (full - background) * .25, rtol=1e-6, atol=1e-7)

    def test_time_is_continuous_and_changes_the_field_without_reseeding_fuel(self):
        fuel = FN._fuel(self.seed).copy()
        points = [(x, .4, z) for x in np.linspace(-.4, .4, 7) for z in np.linspace(-.4, .4, 7)]
        def sample(t):
            tongues = FN._tongues(self.seed, t)
            values = np.array([FN.field(x, y, z, 1., .5, t, self.seed, 0., 0., tongues) for x, y, z in points])
            values[:, 1] *= values[:, 0]  # Heat contribution also vanishes in empty air.
            return values
        a, near, later = sample(2.), sample(2. + 1e-6), sample(2.2)
        self.assertGreater(float(a[:, 0].max()), 0.)
        self.assertGreater(np.count_nonzero(a[:, 0] == 0.), 0)
        self.assertLess(float(np.max(np.abs(near - a))), 1e-3)
        self.assertGreater(float(np.max(np.abs(later - a))), float(np.max(np.abs(near - a))) * 100.)
        np.testing.assert_array_equal(FN._fuel(self.seed), fuel)

    def test_irregular_fuel_seats_stay_stationary_while_flame_changes(self):
        first, later = FN._tongues(self.seed, 2.), FN._tongues(self.seed, 3.)
        stationary = [0, 1, 5, 7, 8, 9]
        np.testing.assert_array_equal(first[:, stationary], later[:, stationary])
        self.assertGreater(np.ptp(first[:, 5]), 0.)
        self.assertGreater(np.ptp(first[:, 6]), 0.)
        self.assertFalse(np.array_equal(first[:, [2, 6]], later[:, [2, 6]]))
        # Emission never precedes an individual patch's physical seat.
        for row in first:
            y = row[5] * .5
            single = row[None, :]
            d, _ = FN.field(row[0] * .5, y - 1e-5, row[1] * .5,
                            1., .5, 2., self.seed, 0., 0., single)
            self.assertEqual(d, 0.)

    def test_roi_keeps_visible_tip_when_base_is_outside_the_image(self):
        C = np.array([0., 0., -4., 0., 1., 1., 0., 30., 10., 40., 20., 20.])
        low, high = FN._bounds(4., .4, np.zeros(2))
        rect = FN._rect(C, low, high, 20, 20)
        self.assertGreater(rect[2], rect[0])
        self.assertGreater(rect[3], rect[1])

    def test_invalid_parameters_fail_before_rendering(self):
        with self.assertRaises(ValueError):
            self.render(trans=float('nan'))
        with self.assertRaises(ValueError):
            self.render(lean=(1., 2., 3.))


if __name__ == '__main__':
    unittest.main()
