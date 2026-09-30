"""Forging effects originate on each birth's actual metal; no raster/JIT."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'lib')]
import c3 as C
import openring as O


class ForgeBirths(unittest.TestCase):
    def effects(self, front=(), strokes=()):
        fx = C.ForgeFX()
        fx.th_u = np.array([0., .5, 1.])
        fx.th_j = np.zeros((3, 3))
        fx.th_E = np.ones(3)
        fx.f_t0 = np.array(front, dtype=float)
        fx.f_v = np.zeros((len(front), 3))
        fx.f_life = np.full(len(front), 20.)
        fx.f_E = np.ones(len(front))
        fx.s_t0 = np.array(strokes, dtype=float)
        fx.s_th = np.linspace(.3, 5.8, len(strokes))
        fx.s_v = fx.s_up = np.zeros(len(strokes))
        fx.s_life = np.full(len(strokes), 20.)
        fx.s_E = np.ones(len(strokes))
        return fx

    def emit(self, fx, f):
        pos, target, hfov = C.cam_c(f)
        camera = C.Camera(pos, target, hfov=hfov)
        records = []
        def record(p0, p1, radius, *args, **kwargs):
            records.append((p0.copy(), p1.copy(), radius))
        ctx = SimpleNamespace(t=f, t0=f, t1=f, cam=camera, cam0=camera, cam1=camera,
                              fr=SimpleNamespace(splat=record))
        fx.emit(ctx, C.FIRE_ROOT + np.array([0., C.HF, 0.]))
        return records

    @staticmethod
    def physical_points(times, angles, pose_time=None):
        points = []
        for birth, theta in zip(times, angles):
            rotation, centre, scale = C.ring_frame(birth if pose_time is None else pose_time)
            local, _ = C.RS.local_points(np.array([theta]), np.zeros(1), sec=C.ring_state(birth).sec)
            points.append(centre + scale * local[0] @ rotation.T)
        return np.array(points)

    def test_vector_births_match_rendered_section_and_flag_off_is_exact(self):
        times = np.array([1202., 1242., 1283., 1304.])
        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            angles = O.theta_range(times)[1]
            expected = self.physical_points(times, angles)
            got, _ = C.ForgeFX.band_point(times, angles)
            np.testing.assert_allclose(got, expected, rtol=0., atol=1e-12)
            with patch.object(C, 'ring_sec', return_value=(1., 1.)):
                wrong, _ = C.ForgeFX.band_point(times, angles)
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(wrong, expected, rtol=0., atol=1e-12)
        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}):
            rotation, centre, scale = C._closed_ring_frame(1242.)
            local, normals = C.RS.local_points(angles, np.zeros(len(angles)))
            actual_points, actual_normals = C.ForgeFX.band_point(1242., angles)
            np.testing.assert_array_equal(actual_points, centre + scale * local @ rotation.T)
            np.testing.assert_array_equal(actual_normals, normals @ rotation.T)

    def test_actual_thread_and_glare_end_at_thin_band_surface(self):
        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            f = 1242.
            expected = self.physical_points([f], [O.theta_range(f)[1]])[0]
            records = self.emit(self.effects(), f)
            self.assertEqual(len(records), 2)
            np.testing.assert_allclose(records[0][0][-1], expected, atol=1e-12)
            np.testing.assert_allclose(records[1][0][0], expected, atol=1e-12)
            with patch.object(C, 'ring_sec', return_value=(1., 1.)):
                wrong = self.emit(self.effects(), f)
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(wrong[0][0][-1], expected, atol=1e-12)

    def test_front_particle_position_is_independent_of_other_live_particles(self):
        def oracle():
            alone = self.emit(self.effects(front=[1261.]), 1262.)[-1][0][0]
            alongside = self.emit(self.effects(front=[1254., 1261.]), 1262.)[-1][0][1]
            np.testing.assert_allclose(alongside, alone, rtol=0., atol=1e-12)
        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            oracle()
            original = C.ForgeFX.band_point
            def first_birth(t, theta, psi=0., **kwargs):
                return original(np.asarray(t).ravel()[0], theta, psi, **kwargs)
            with patch.object(C.ForgeFX, 'band_point', staticmethod(first_birth)):
                with self.assertRaises(AssertionError):
                    oracle()

    def test_actual_anvil_sparks_keep_birth_section_when_riding_current_pose(self):
        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            f = 1262.
            fx = self.effects(strokes=[1243., 1261.])
            angles = O.arc_theta(fx.s_t0, fx.s_th / (2 * np.pi))
            expected = self.physical_points(fx.s_t0, angles, pose_time=f)
            expected[:, 1] -= .012 * (f - fx.s_t0) ** 2
            actual = self.emit(fx, f)[-1][0]
            np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)
            original = C.ForgeFX.band_point
            def wrong_section(t, theta, psi=0., **kwargs):
                return original(1240., theta, psi, **kwargs)
            with patch.object(C.ForgeFX, 'band_point', staticmethod(wrong_section)):
                wrong = self.emit(fx, f)[-1][0]
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(wrong, expected, rtol=0., atol=1e-12)


if __name__ == '__main__':
    unittest.main()
