"""Hearth scalar/synthetic contracts; use NUMBA_DISABLE_JIT=1 for cheap checks.

Repeat without that flag, when allocated compute, to validate Numba compilation.
Neither mode renders a full shot or imports the world/figure modules.
"""
from pathlib import Path
import sys
import types
import unittest

import cv2
import numpy as np

cv2.setNumThreads(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hearth_a as HE


class HearthTests(unittest.TestCase):
    def setUp(self):
        self.hearth = HE.Hearth([0., 0., 0.], seed=7070)

    def test_build_is_deterministic_and_samples_world_ground(self):
        translated = HE.Hearth([120., 50., -80.], seed=7070, ground=lambda x, z: 50.)
        np.testing.assert_array_equal(translated.stones, self.hearth.stones)
        np.testing.assert_array_equal(translated.planes, self.hearth.planes)
        np.testing.assert_array_equal(translated.pits, self.hearth.pits)
        other = HE.Hearth([0., 0., 0.], seed=7071)
        self.assertFalse(np.array_equal(other.stones, self.hearth.stones))
        self.assertTrue(np.isfinite(self.hearth.stones).all())
        self.assertTrue(np.isfinite(self.hearth.planes).all())

    def test_stones_are_flattened_and_loose_stones_partly_sunk(self):
        s, p = self.hearth.stones, self.hearth.planes
        # Half-thickness is smaller than the lateral supports, including rubble.
        self.assertTrue(np.all(p[:, 6, 3] < p[:, :6, 3].max(axis=1)))
        loose = s[:, 8] == 2.
        self.assertGreater(int(loose.sum()), 0)
        self.assertTrue(np.all(s[loose, 1] - p[loose, 6, 3] < s[loose, 7]))
        self.assertGreater(np.ptp(s[:18, 4]), .01)

    def test_fire_seat_is_above_actual_central_rubble_surface(self):
        h = self.hearth
        y = h.fire_base_offset
        above, _ = HE.scene_distance(0., y, 0., h.stones, h.planes, h.pits)
        below, _ = HE.scene_distance(0., y - .010, 0., h.stones, h.planes, h.pits)
        self.assertGreater(above, 0.)
        self.assertLess(below, 0.)
        self.assertLess(y, .20)

    def test_rough_site_culls_stairs_without_moving_surviving_stones(self):
        ground = lambda x, z: .8 * x - .4 * z
        full = HE.Hearth([0., 0., 0.], seed=7070, ground=ground)
        broken = HE.Hearth([0., 0., 0.], seed=7070, ground=ground, ground_band=(-.30, .12))
        self.assertLess(len(broken.stones), len(full.stones))
        self.assertTrue(np.all((broken.stones[:, 7] >= -.30) & (broken.stones[:, 7] <= .12)))
        for k, row in enumerate(broken.stones):
            original = np.flatnonzero(full.stones[:, 10] == row[10])
            self.assertEqual(len(original), 1)
            j = original[0]
            np.testing.assert_array_equal(row, full.stones[j])
            np.testing.assert_array_equal(broken.planes[k], full.planes[j])
            np.testing.assert_array_equal(broken.pits[k], full.pits[j])
            # Each stone intersects terrain beneath its centre. No retained
            # upper course can float after removal of a supporting neighbour.
            self.assertLess(HE.stone_distance(row[0], row[7], row[2], k,
                                             broken.stones, broken.planes, broken.pits), 0.)
        y = broken.fire_base_offset
        self.assertGreater(HE.scene_distance(0., y, 0., broken.stones, broken.planes, broken.pits)[0], 0.)

    def test_sloped_ground_cannot_embed_the_fire_in_neighbouring_rubble(self):
        # This seed/slope put stone 27 above the flat central cap: using only
        # that cap's top left the old nominal flame base 5.6 mm inside the union.
        h = HE.Hearth([0., 0., 0.], seed=7074, ground=lambda x, z: .8 * (.6 * x + .8 * z))
        y = h.fire_base_offset
        above, _ = HE.scene_distance(0., y, 0., h.stones, h.planes, h.pits)
        below, _ = HE.scene_distance(0., y - .010, 0., h.stones, h.planes, h.pits)
        self.assertGreater(above, 0.)
        self.assertLess(below, 0.)

    def test_bounds_contain_each_stone_and_distant_points_are_outside(self):
        h = self.hearth
        dist = np.linalg.norm(h.stones[:, :3] - h.bound_centre, axis=1) + h.stones[:, 3]
        self.assertTrue(np.all(dist <= h.bound_radius + .01))
        d, _ = HE.scene_distance(5., 5., 5., h.stones, h.planes, h.pits)
        self.assertGreater(d, 1.)

    def test_physical_pit_removes_solid_without_affecting_unrelated_point(self):
        h = self.hearth
        k = 0
        pit = h.pits[k, 0]
        original = h.pits.copy()
        no_pits = original.copy()
        no_pits[:, :, 3] = 0.
        # Probe just below this top-plane pit's centre along the stone's up axis.
        n = h.planes[k, 6, :3]
        point = pit[:3] - n * pit[3] * .9
        uncut = HE.stone_distance(*point, k, h.stones, h.planes, no_pits)
        cut = HE.stone_distance(*point, k, h.stones, h.planes, original)
        self.assertGreater(cut, uncut)
        self.assertGreater(cut, 0.)

    def test_light_visibility_detects_a_real_blocker(self):
        h = self.hearth
        row = h.stones[0]
        self.assertEqual(HE.visibility(row[0], row[1] + 2., row[2], 0., -1., 0., 4., h.stones, h.planes), 0.)
        self.assertEqual(HE.visibility(4., 2., 4., 0., 1., 0., 2., h.stones, h.planes), 1.)

    def synthetic_draw(self, depth):
        C = np.array([0., .5, -4., 0., 1., 1., 0., 30., 1.5, -1.5, 3., 3.])
        camera = types.SimpleNamespace(params=lambda: C)
        img = np.zeros((3, 3, 3), np.float32)
        zb = np.full((3, 3), depth, np.float32)
        lights = np.array([[0., .6, 0., 1., .4, .1, 2., .3]])
        lk = np.array([-.6, .8, 0., .4, .5, .7])
        fog = np.array([0., .001, 0., .01, 0., 0., 0., 0.])
        q = np.zeros(24)
        q[0] = .4
        HE.draw(img, zb, camera, self.hearth, lights, (lk, np.ones(3) * .1, None, fog, q), ground_shadow=False)
        return img, zb

    def test_tracer_draws_stone_and_writes_real_forward_depth(self):
        image, depth = self.synthetic_draw(1e9)
        self.assertGreater(np.count_nonzero(depth < 1e8), 0)
        self.assertGreater(float(image.max()), 0.)
        self.assertTrue(np.isfinite(image).all())
        self.assertTrue(np.all(depth[depth < 1e8] > 2.))
        self.assertTrue(np.all(depth[depth < 1e8] < 6.))

    def test_foreground_depth_prevents_any_hearth_paint(self):
        image, depth = self.synthetic_draw(.1)
        np.testing.assert_array_equal(image, np.zeros_like(image))
        np.testing.assert_array_equal(depth, np.full_like(depth, .1))

    def ground_pass(self, point, depth=4., lights=None):
        h = self.hearth
        x, y, z = point
        C = np.array([x, y + 1., z - 4., 0., 1., 1., 0., 4., .5, -.5, 1., 1.])
        image = np.full((1, 1, 3), .5, np.float32)
        zb = np.full((1, 1), depth, np.float32)
        if lights is None:
            lights = np.empty((0, 8))
        HE._ground_shadow(image, zb, C, h.stones, h.planes, h.pits, lights,
                          np.zeros(7), np.zeros(3), h.scale, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(zb, np.full_like(zb, depth))
        return image

    def test_ground_pass_preserves_outside_sky_and_foreground_pixels(self):
        for point, depth in (((5., 0., 0.), 4.), ((0., 0., 0.), 1e9), ((0., 0., 0.), .1)):
            np.testing.assert_array_equal(self.ground_pass(point, depth), np.full((1, 1, 3), .5, np.float32))

    def test_ground_pass_contact_is_finite_without_light_and_blocked_light_darkens(self):
        unlit = self.ground_pass((0., 0., 0.))
        self.assertTrue(np.isfinite(unlit).all())
        self.assertTrue(np.all((unlit > 0.) & (unlit < .5)))
        lit = self.ground_pass((0., 0., 0.), lights=np.array([[0., 1., 0., 1., .5, .2, 2., .2]]))
        self.assertTrue(np.all(lit < unlit))

    def test_bad_ground_and_scale_fail_early(self):
        with self.assertRaises(ValueError):
            HE.Hearth([0., 0., 0.], scale=0.)
        with self.assertRaises(ValueError):
            HE.Hearth([0., 0., 0.], ground=lambda x, z: float('nan'))


if __name__ == '__main__':
    unittest.main()
