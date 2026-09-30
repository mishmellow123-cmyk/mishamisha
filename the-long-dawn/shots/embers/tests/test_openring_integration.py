"""Open-band wiring contracts; no geometry construction, rasterization or JIT.

Each oracle is also run against a deliberately broken result or implementation.
Rendered flag-off byte hashes are captured by the lane's separate render probe.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'lib')]
import c3
import c_v5 as V
import c_v5_instep as I
import openring as OR


class OpenRingIntegration(unittest.TestCase):
    def rejects(self, oracle, mutant):
        with self.assertRaises(AssertionError):
            oracle(mutant)

    def v5_scene(self):
        scene = V.Scene.__new__(V.Scene)
        scene.towers = SimpleNamespace(k_all=0)
        return scene

    def instep_scene(self):
        with patch.object(V.Scene, '__init__', return_value=None):
            return I.Scene('in-step')

    def test_flag_off_keeps_legacy_pose_and_theta_exactly(self):
        frames = (1200., 1208.5, 1238., 1306., 1439.75, 1440., 1582., 1679.75)

        def pose_oracle(frame_fn):
            for f in frames:
                for got, wanted in zip(frame_fn(f), c3._closed_ring_frame(f)):
                    np.testing.assert_array_equal(got, wanted)

        def theta_oracle(theta_fn):
            for f in frames:
                k = float(c3.smootherstep(1202., 1238., f))
                self.assertEqual(theta_fn(f), (-.6, -.6 + max(k, 1e-3) * 2 * np.pi))

        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}), \
             patch.object(OR, 'placement', side_effect=AssertionError('open placement on closed route')):
            pose_oracle(c3.ring_frame)
            theta_oracle(c3.ring_theta_range)
            self.rejects(pose_oracle, lambda f: (np.eye(3), np.zeros(3), c3.RING_W))
            self.rejects(theta_oracle, lambda f: (-.6, -.6 + 4 * np.pi / 3))
            scene = self.v5_scene()

            def v5_oracle(frame_fn):
                for f in (2320., 2480., 2639., 3816., 3848., 4000., 4239.):
                    clock = 1580. + .12 * ((f - 2320) if f < 3000 else 320. + min(f, 4216) - 3840.)
                    expected = c3._closed_ring_frame(clock)
                    rot, centre, size = frame_fn(f)
                    np.testing.assert_array_equal(rot, expected[0])
                    np.testing.assert_array_equal(centre, V.RING_C)
                    self.assertEqual(size, expected[2])

            v5_oracle(scene.ring_frame)
            self.rejects(v5_oracle, lambda f: c3._closed_ring_frame(f))

    def capture_v5_ring(self, scene, f):
        """Run the real layer through a tiny recording raster sink."""
        seen = []
        ctx = SimpleNamespace(t=f, fr=SimpleNamespace(W=3, H=2), cam=scene.camera(f))

        def raster(cam, w, h, rot, centre, scale, state, env, **kwargs):
            seen.append((state, kwargs))
            return np.ones((h, w, 3), np.float32), np.ones((h, w)), np.ones((h, w))

        with patch.object(V.RS, 'render', raster), \
             patch.object(V.RS, 'merge_occluder', return_value=None), \
             patch.object(V.RS, 'visibility', return_value=np.ones((2, 3))):
            scene.ring_layer(ctx)
        self.assertEqual(len(seen), 1)
        return seen[0]

    def test_v5_flag_gates_arc_caps_letters_and_absolute_frame_heat(self):
        scene = self.v5_scene()

        def open_oracle(result, frame=2480.):
            state, kwargs = result
            self.assertEqual(kwargs.get('th_range'), OR.theta_range(frame))
            self.assertTrue(state.end_caps)
            self.assertTrue(state.leading_glyph)
            angles = np.array(OR.theta_range(frame))
            np.testing.assert_array_equal(state.heat(angles), OR.heat(frame)(angles))

        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            for frame in (2480., 3816., 4120.):
                open_oracle(self.capture_v5_ring(scene, frame), frame)
            state, kwargs = self.capture_v5_ring(scene, 2480.)
            self.rejects(open_oracle, (state, {}))
            self.rejects(open_oracle, (state, {**kwargs, 'th_range': OR.theta_range(1580.)}))

        def closed_oracle(result):
            state, kwargs = result
            self.assertEqual(kwargs, {})
            self.assertFalse(getattr(state, 'end_caps', False))
            self.assertFalse(getattr(state, 'leading_glyph', False))
            self.assertEqual(state.letters, .48)
            self.assertEqual(state.glow, .24)
            np.testing.assert_array_equal(state.heat(np.array([0., 1.])), [.31, .31])

        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}):
            closed = self.capture_v5_ring(scene, 2480.)
            closed_oracle(closed)
            self.rejects(closed_oracle, (closed[0], {'th_range': (0., 5.)}))

    def test_c3_open_state_reaches_each_shutter_sample(self):
        seen = []
        f = 1582.
        camera = self.v5_scene().camera(f)
        ctx = SimpleNamespace(t=f, t0=f - .25, t1=f + .25,
                              fr=SimpleNamespace(W=3, H=2))
        timeline = SimpleNamespace(camera=lambda t: camera)

        def raster(cam, w, h, rot, centre, scale, state, env, **kwargs):
            seen.append((state, kwargs))
            return np.ones((h, w, 3), np.float32), np.ones((h, w)), np.ones((h, w))

        def oracle(records):
            self.assertEqual(len(records), 3)
            for (state, kwargs), t in zip(records, (f - .25, f, f + .25)):
                self.assertEqual(kwargs['th_range'], OR.theta_range(t))
                self.assertTrue(state.end_caps)
                self.assertTrue(state.leading_glyph)
                angles = np.array(OR.theta_range(t))
                np.testing.assert_array_equal(state.heat(angles), OR.heat(t)(angles))
                a, b = OR.theta_range(t)
                np.testing.assert_array_equal(state.write(np.array([a - .01, (a + b) / 2, b + .01])), [0., 1., 0.])

        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}), \
             patch.object(c3, 'ring_env', return_value=object()), \
             patch.object(c3.RS, 'render', raster), \
             patch.object(c3.RS, 'merge_occluder', return_value=None), \
             patch.object(c3.RS, 'visibility', return_value=np.ones((2, 3))):
            c3.ring_layer(timeline, ctx)
            oracle(seen)
            self.rejects(oracle, seen[:1])
            self.rejects(oracle, [(state, {'th_range': (0., 2 * np.pi)}) for state, _ in seen])
        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}):
            state = c3.ring_state(f)
            self.assertIsNone(state.heat)
            self.assertFalse(getattr(state, 'end_caps', False))
            self.assertFalse(getattr(state, 'leading_glyph', False))

    def test_c3_gold_origins_are_on_metal_at_detachment(self):
        rain = c3.GoldRain.__new__(c3.GoldRain)
        rain.t0 = np.array([1450., 1562., 1602., 1662.])
        rain.th = np.array([0., np.pi / 2, np.pi, 2 * np.pi - .01])

        def oracle(starts):
            for index, t in enumerate(rain.t0):
                rotation, centre, size = c3.ring_frame(t)
                theta = OR.arc_theta(t, np.array([rain.th[index] / (2 * np.pi)]))
                local, _ = c3.RS.local_points(theta, np.zeros(1))
                np.testing.assert_allclose(starts[index], centre + size * local[0] @ rotation.T, atol=1e-12)

        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            starts = rain.start(rain.t0)
            oracle(starts)
            with patch.object(OR, 'arc_theta', side_effect=lambda t, fractions: 2 * np.pi * np.asarray(fractions)):
                wrong = rain.start(rain.t0)
            self.rejects(oracle, wrong)
        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}):
            for index, t in enumerate(rain.t0):
                rotation, centre, size = c3._closed_ring_frame(t)
                local, _ = c3.RS.local_points(np.array([rain.th[index]]), np.zeros(1))
                np.testing.assert_array_equal(rain.start(rain.t0)[index], centre + size * local[0] @ rotation.T)

    def test_v5_drop_origins_stay_on_the_arc_at_their_absolute_birth_times(self):
        scene = self.v5_scene()
        scene.drop_phase = np.array([.1, .3, .5])
        scene.drop_th = np.array([0., np.pi, 2 * np.pi - .01])
        scene.drop_tower = np.array([0, 1, 2])
        scene.towers = SimpleNamespace(top=lambda i, t: np.array([i, 0., 0.]))
        ctx = SimpleNamespace(t=2480., t0=2479.75, t1=2480.25, cam=scene.camera(2480.),
                              cam0=scene.camera(2479.75), cam1=scene.camera(2480.25),
                              fr=SimpleNamespace(splat=lambda *a, **k: None))
        original = V.RS.local_points

        def collect():
            seen = []
            def record(theta, psi, *args):
                seen.append(np.asarray(theta).copy())
                return original(theta, psi, *args)
            with patch.object(V.RS, 'local_points', record):
                scene.drops(ctx)
            return seen

        def oracle(samples):
            self.assertEqual(len(samples), 2)
            for theta, t in zip(samples, (ctx.t0, ctx.t1)):
                phase = (scene.drop_phase + (t - 2320.) / 52.) % 1.
                births = t - 52. * phase
                np.testing.assert_array_equal(theta, OR.arc_theta(births, scene.drop_th / (2 * np.pi)))

        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            oracle(collect())
            self.rejects(oracle, [scene.drop_th, scene.drop_th])
            moving_origins = [OR.arc_theta(t, scene.drop_th / (2 * np.pi)) for t in (ctx.t0, ctx.t1)]
            self.rejects(oracle, moving_origins)
        with patch.dict(os.environ, {'LD_OPEN_RING': '0'}):
            for theta in collect():
                np.testing.assert_array_equal(theta, scene.drop_th)

    def test_instep_shared_gold_uses_open_rim_for_origins_and_ceiling(self):
        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            scene = self.instep_scene()
            scene.towers = SimpleNamespace(k_all=5)
            targets = np.array([[i, 0., 0.] for i in range(5)])
            with patch.object(I, 'window_targets', return_value=(targets, tuple(range(5)))) as targeter:
                scene._shared_drops()
            rotation, centre, size = scene.ring_frame(4000.)
            theta = OR.arc_theta(4000., np.arange(10) / 10.)
            rim, _ = V.RS.local_points(theta, np.zeros(10))
            starts = centre + size * rim @ rotation.T

            def ceiling_oracle(ceiling):
                self.assertAlmostEqual(ceiling, starts[:, 1].min() - .25)

            ceiling_oracle(targeter.call_args.kwargs['ceiling'])
            self.rejects(ceiling_oracle, centre[1] - 3.)

            def origins_oracle(result):
                positions, u = result
                end = targets[scene.shared_tower]
                expected = starts + (end - starts) * u[:, None]
                expected[:, 1] = starts[:, 1] + (end[:, 1] - starts[:, 1]) * u * u
                np.testing.assert_allclose(positions, expected, atol=1e-12)
                np.testing.assert_array_equal(np.bincount(scene.shared_tower, minlength=5), [2] * 5)

            origins_oracle(scene.shared_positions(4120.))
            self.rejects(origins_oracle, (np.zeros((10, 3)), np.zeros(10)))

    def test_open_pose_has_no_pop_at_contiguous_edit_boundaries(self):
        with patch.dict(os.environ, {'LD_OPEN_RING': '1'}):
            scene = self.instep_scene()

            def oracle(frame_fn, boundary):
                left = frame_fn(boundary - 1e-4)
                right = frame_fn(boundary + 1e-4)
                for a, b in zip(left, right):
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-4)

            oracle(c3.ring_frame, 1440.)
            oracle(scene.ring_frame, 4000.)
            for f in (3816., 3840., 3848., 3999., 4000., 4120., 4239.):
                np.testing.assert_array_equal(scene.ring_frame(f)[0], scene.ring_frame(4000.)[0])
            def broken(frame):
                rot, centre, size = scene.ring_frame(frame)
                return (rot + (np.eye(3) if frame >= 4000 else 0), centre, size)
            with self.assertRaises(AssertionError):
                oracle(broken, 4000.)


if __name__ == '__main__':
    unittest.main()
