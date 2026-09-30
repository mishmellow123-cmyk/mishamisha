"""Absolute-frame closure and camera-derived placement, including red controls."""
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'lib')]
import openring as O
import c3
import c_v5 as C


class GapSchedule(unittest.TestCase):
    def assert_monotone(self, gap):
        samples = gap(np.arange(1200.0, 4240.01, 0.25))
        self.assertTrue(np.all(np.diff(samples) <= 1e-10))
        self.assertGreaterEqual(float(np.min(samples)), 30.0 - 1e-10)
        self.assertLessEqual(float(np.max(samples)), 360.0)

    def assert_holds(self, gap):
        for (_, end, target), (start, _, _) in zip(O.GAP_SCHEDULE, O.GAP_SCHEDULE[1:]):
            values = gap(np.linspace(end, start, 17))
            np.testing.assert_allclose(values, target, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(gap(np.arange(3792.0, 8000.0, 0.5)), 30.0,
                                   atol=1e-10, rtol=0.0)

    def test_absolute_schedule_milestones_and_perpetual_pause(self):
        self.assert_monotone(O.gap_deg)
        self.assert_holds(O.gap_deg)
        for f, gap in ((1199, 360), (1200, 360), (1305, 120), (1680, 80),
                       (2080, 40), (2640, 34), (3784, 34), (3792, 30),
                       (3816, 30), (3848, 30), (4240, 30), (1e8, 30)):
            self.assertAlmostEqual(O.gap_deg(f), gap)
        self.assertEqual(O.FORGE_END, 1305.0)
        self.assertTrue(all(start < end for start, end, _ in O.GAP_SCHEDULE))
        self.assertTrue(all(a[1] < b[0] for a, b in zip(O.GAP_SCHEDULE, O.GAP_SCHEDULE[1:])))

    def test_red_controls_reject_reopening_crawl_and_post_pause_closure(self):
        wrong = list(O.GAP_SCHEDULE)
        start, end, _ = wrong[7]
        wrong[7] = (start, end, 125.0)
        with patch.object(O, 'GAP_SCHEDULE', tuple(wrong)):
            with self.assertRaises(AssertionError):
                self.assert_monotone(O.gap_deg)
        frames = [1200.0] + [end for _, end, _ in O.GAP_SCHEDULE]
        gaps = [360.0] + [gap for _, _, gap in O.GAP_SCHEDULE]
        with self.assertRaises(AssertionError):
            self.assert_holds(lambda f: np.interp(f, frames, gaps))
        with patch.object(O, 'GAP_SCHEDULE', O.GAP_SCHEDULE + ((4100., 4105., 29.),)):
            with self.assertRaises(AssertionError):
                self.assert_holds(O.gap_deg)

    def test_strokes_follow_existing_forge_and_race_beat_lists(self):
        starts = np.array([start for start, _, _ in O.GAP_SCHEDULE])
        np.testing.assert_array_equal(starts[starts < 1440], c3.STROKES)
        np.testing.assert_array_equal(starts[(starts >= 1440) & (starts < 1680)], c3.RACE_BEATS)
        for start, end, _ in O.GAP_SCHEDULE:
            self.assertGreater(O.advance_rate((start + end) / 2.0), 0.0)
            self.assertEqual(O.advance_rate(start), 0.0)
            self.assertEqual(O.advance_rate(end), 0.0)
            self.assertEqual(O.advance_rate(end + 1), 0.0)
            f = start + 0.37 * (end - start)
            numerical = -(O.gap_deg(f + 1e-4) - O.gap_deg(f - 1e-4)) / 2e-4
            self.assertAlmostEqual(O.advance_rate(f), numerical, places=6)

    def test_gap_bisector_is_fixed_as_both_ends_advance(self):
        frames = np.arange(1200.0, 4240.0, 0.5)
        start, end = O.theta_range(frames)
        np.testing.assert_allclose((start + end) * .5, O.GAP_CENTRE + np.pi)
        np.testing.assert_allclose(end - start,
                                   np.maximum(np.deg2rad(360.0 - O.gap_deg(frames)), O.MIN_ARC_RADIANS))
        self.assertTrue(np.all(np.diff(start) <= 1e-10))
        self.assertTrue(np.all(np.diff(end) >= -1e-10))
        self.assertTrue(np.all(end - start < 2 * np.pi))
        self.assertGreater(float((end - start).min()), 0.)
        self.assertAlmostEqual(O.theta_range(O.FORGE_END)[0], O.THETA_START)

    def test_drops_use_only_extant_arc_including_array_birth_times(self):
        frames = np.array([1200., 1201., 1203., 1282., 1581., 4120.])[:, None]
        fractions = np.linspace(0.0, 1.0, 31)[None, :]
        th0, th1 = O.theta_range(frames)
        theta = O.arc_theta(frames, fractions)
        self.assertTrue(np.all(theta >= th0))
        self.assertTrue(np.all(theta <= th1 + 1e-10))
        self.assertTrue(np.all(theta[1:, 0] > th0[1:, 0]))
        self.assertTrue(np.all(theta[1:, -1] < th1[1:, 0]))
        self.assertAlmostEqual(O.arc_theta(1580, .5), sum(O.theta_range(1580)) / 2.0)
        # The old closed-ring mapping fails on the same birth-time population.
        with self.assertRaises(AssertionError):
            self.assertTrue(np.all(th0 + fractions * 2 * np.pi <= th1 + 1e-10))

    def test_inscription_cannot_cross_gap_and_has_half_written_edge(self):
        for f in (1305, 1602, 2483, 3816, 4120):
            a, b = O.theta_range(f)
            mask = O.write(f)
            theta = np.array([a + .1, b - .1, b, b + .01, a + 2 * np.pi - .01])
            np.testing.assert_allclose(mask(theta), [1., 1., .5, 0., 0.], atol=1e-10)
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(np.ones_like(theta), mask(theta))


class OptInAndHeat(unittest.TestCase):
    def test_flag_requires_explicit_one_and_is_not_cached(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(O.enabled())
            for flag in ('', '0', 'true', 'yes', '01'):
                os.environ['LD_OPEN_RING'] = flag
                self.assertFalse(O.enabled())
            os.environ['LD_OPEN_RING'] = '1'
            self.assertTrue(O.enabled())
            os.environ['LD_OPEN_RING'] = '0'
            self.assertFalse(O.enabled())

    def test_both_ends_reheat_with_advance_rate_and_body_stays_gold(self):
        for f in (1562.5, 1602.5, 2502.5):
            a, b = O.theta_range(f)
            heat = O.heat(f)(np.array([a, b, (a + b) / 2]))
            self.assertAlmostEqual(heat[0], heat[1])
            self.assertGreater(heat[0], .82)
            self.assertLess(heat[2], .001)
        endpoint_heat = lambda f: O.heat(f)(O.theta_range(f)[0])
        self.assertGreater(endpoint_heat(1602.5), endpoint_heat(1615))
        self.assertGreater(endpoint_heat(1602.5), endpoint_heat(2502.5))

    def assert_cools_and_keeps_warmth(self, temperatures):
        self.assertGreater(temperatures[0], .8)
        self.assertTrue(np.all(np.diff(temperatures) <= 1e-12))
        self.assertGreater(temperatures[-1], .05)
        self.assertLess(temperatures[-1], .2)

    def test_pause_cools_monotonically_to_faint_residual_warmth(self):
        temperatures = np.array([O.heat(f)(O.theta_range(f)[0]) for f in range(3848, 4240)])
        self.assert_cools_and_keeps_warmth(temperatures)
        self.assertAlmostEqual(temperatures[-1], .13)
        for wrong in (np.zeros(392), np.ones(392), temperatures[::-1]):
            with self.assertRaises(AssertionError):
                self.assert_cools_and_keeps_warmth(wrong)


class Placement(unittest.TestCase):
    def assert_readable(self, rotation, centre_at, camera_at, frames):
        for f in frames:
            centre, camera = centre_at(f), camera_at(f)
            self.assertGreater(O.gap_facing(rotation, camera, centre, f), .9, f)
            self.assertGreater(float(O.cap_facing(rotation, camera, centre, f).min()), .1, f)
            view = camera - centre
            axis = abs(float(view @ rotation[:, 1] / np.linalg.norm(view)))
            self.assertGreater(axis, .2, f)
            self.assertLess(axis, .85, f)

    def test_forging_race_share_one_pose_all_frames(self):
        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}):
            reference = 1330.
            raw = c3.ring_frame
            base, centre, _ = raw(reference)
            rotation = O.placement(base, c3.cam_c(reference)[0], centre, reference)
            np.testing.assert_allclose(rotation[:, 1], base[:, 1], atol=1e-14)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
            centre_at = lambda f: raw(f)[1]
            camera_at = lambda f: c3.cam_c(f)[0]
            self.assert_readable(rotation, centre_at, camera_at, range(1205, 1680))
            for f in range(1200, 1680):
                self.assertGreater(O.gap_facing(rotation, camera_at(f), centre_at(f), f), .9)
            with self.assertRaises(AssertionError):
                self.assert_readable(rotation @ O._roll(np.pi), centre_at, camera_at, [1602])
            # The first stroke starts as a nearly zero-length seed; its two
            # cut normals are almost opposite before the first arc is formed.
            # Test gap placement there, and cap readability from the 40-degree arc.

    def test_trap_and_quiet_pose_cover_every_frame_including_leadin(self):
        scene = C.Scene.__new__(C.Scene)
        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}):
            for a, b, reference in ((2320, 2640, 2480), (3816, 4240, 4000)):
                base, centre, _ = scene.ring_frame(reference)
                rotation = O.placement(base, scene.camera(reference).pos, centre, reference)
                self.assert_readable(rotation, lambda f: scene.ring_frame(f)[1],
                                     lambda f: scene.camera(f).pos, range(a, b))

    def test_camera_on_axis_rejects_undefined_roll(self):
        with self.assertRaises(ValueError):
            O.placement(np.eye(3), [0., 1., 0.], [0., 0., 0.], 1600)

    @staticmethod
    def forging_camera(f):
        pos, target, hfov = c3.cam_c(f)
        return c3.Camera(pos, target, hfov=hfov)

    def assert_projected_band(self, frame_at, camera_at, section_at, frames):
        """Inspect both cut outlines AND intervening metal, in native pixels."""
        for f in frames:
            rotation, centre, scale = frame_at(f)
            camera = camera_at(f)
            ends = O.theta_range(f)
            theta, psi = np.meshgrid(np.linspace(*ends, 97), np.linspace(-np.pi, np.pi, 49))
            points, _ = C.RS.local_points(theta.ravel(), psi.ravel(), sec=section_at(f))
            x, y, z = camera.project(centre + scale * points @ rotation.T, 1920, 804)
            self.assertGreater(float(z.min()), 0., f)
            self.assertGreaterEqual(min(float(x.min()), 1920. - float(x.max()),
                                        float(y.min()), 804. - float(y.max())), 12., f)
            self.assertGreater(O.gap_facing(rotation, camera.pos, centre, f), .9, f)
            view = camera.pos - centre
            axis = abs(float(view @ rotation[:, 1] / np.linalg.norm(view)))
            self.assertGreater(axis, .2, f)
            self.assertLess(axis, .85, f)
            if f < 1205:
                continue  # a nearly zero-length seed has almost opposite cut normals
            # View each cap from its OWN position, not the ring's centre.
            a, b = ends
            cap_points = np.array([[C.RS.R_MID * math.cos(a), 0., C.RS.R_MID * math.sin(a)],
                                   [C.RS.R_MID * math.cos(b), 0., C.RS.R_MID * math.sin(b)]])
            cap_normals = np.array([[math.sin(a), 0., -math.cos(a)],
                                    [-math.sin(b), 0., math.cos(b)]]) @ rotation.T
            directions = camera.pos - (centre + scale * cap_points @ rotation.T)
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            self.assertGreater(float(np.sum(cap_normals * directions, axis=1).min()), .1, f)

    def test_actual_open_band_and_both_ends_fit_every_native_frame(self):
        scene = C.Scene.__new__(C.Scene)
        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            self.assert_projected_band(c3.ring_frame, self.forging_camera, c3.ring_sec,
                                       range(1200, 1680))
            for a, b in ((2320, 2640), (3816, 4240)):
                self.assert_projected_band(scene.ring_frame, scene.camera, lambda f: (1., 1.), range(a, b))

    def test_red_control_catches_crop_despite_valid_gap_and_cap_angles(self):
        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}):
            raw = c3.ring_frame
            base, centre, _ = raw(1330.)
            rotation = O.placement(base, c3.cam_c(1330.)[0], centre, 1330.)
            # The original scale has a readable, camera-facing gap but projects
            # beyond the image. Even 80% scale clears both caps and crops metal.
            for multiplier in (1., .8):
                def wrong_frame(f):
                    _, centre, scale = raw(f)
                    return rotation, centre, scale * multiplier
                self.assert_readable(rotation, lambda f: raw(f)[1],
                                     lambda f: c3.cam_c(f)[0], [1339])
                with self.assertRaises(AssertionError):
                    self.assert_projected_band(wrong_frame, self.forging_camera, c3.ring_sec, [1339])


if __name__ == '__main__':
    unittest.main()
