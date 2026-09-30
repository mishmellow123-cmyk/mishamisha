"""The study lamps illuminate the real letter mask; absent lamps preserve C pixels."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'shots' / 'embers'), str(ROOT / 'lib')]
import d_lamplight as L
import ringsolid as R
from core import Camera


class ReadingLightTests(unittest.TestCase):
    def rejects(self, oracle, value):
        with self.assertRaises(AssertionError):
            oracle(value)

    def setUp(self):
        self.th = np.linspace(0., 2. * np.pi, 4096, endpoint=False)
        self.ps = np.zeros_like(self.th)
        self.camera = Camera((6., 2., -4.), (0., 0., 0.))
        self.dark = R.Env(above=(0., 0., 0.), horizon=(0., 0., 0.), below=(0., 0., 0.))

    def shade(self, state, leading=None):
        return R.shade(self.th, self.ps, self.camera, 4096., 1, np.eye(3), np.zeros(3),
                       1., state, self.dark, leading_theta=leading)

    def test_default_none_and_zero_power_preserve_actual_shader_exactly(self):
        state = R.RingState()
        self.assertIsNone(state.reading_light)
        for letters in (0., .62):
            state.letters = letters
            state.glow = .08
            state.heat = lambda th: .2 + .7 * np.sin(th) ** 2
            state.write = lambda th: np.clip(th / (2 * np.pi), 0., 1.)
            state.reading_light = None
            expected = self.shade(state, leading=5.4)
            state.reading_light = L.ReadingLights([[3., 1., 0.]], powers=0.)
            def oracle(result):
                np.testing.assert_array_equal(result, expected)
            oracle(self.shade(state, leading=5.4))
            self.rejects(oracle, expected + .01)

    def test_locality_inverse_square_and_backfaces_with_uniform_negative(self):
        points = np.array([[0., 0., 0.], [0., -3., 0.], [0., 0., 0.]])
        normals = np.array([[0., 1., 0.], [0., 1., 0.], [0., -1., 0.]])
        lamp = L.ReadingLights([[0., 1., 0.]], powers=3., radius=.4)
        def oracle(callback):
            result = callback(points, normals)
            self.assertGreater(float(result[0].sum()), float(result[1].sum()) * 10.)
            np.testing.assert_array_equal(result[2], np.zeros(3))
            np.testing.assert_allclose(result[0] / result[1], (16. + .16) / (1. + .16))
        oracle(lamp)
        self.rejects(oracle, lambda p, n: np.ones_like(p))
        self.rejects(oracle, lambda p, n: lamp(p, np.abs(n)))

    def test_lamp_color_and_power_are_linear_and_zero_at_coincident_sample(self):
        points = np.array([[0., 0., 0.], [0., 1., 0.]])
        normals = np.tile([0., 1., 0.], (2, 1))
        lamp = L.ReadingLights([[0., 1., 0.]], powers=2.)
        dim = L.ReadingLights([[0., 1., 0.]], powers=1.)
        np.testing.assert_allclose(lamp(points, normals), 2. * dim(points, normals), rtol=0., atol=0.)
        np.testing.assert_array_equal(lamp(points, normals)[1], np.zeros(3))
        ratio = lamp(points, normals)[0] / lamp(points, normals)[0, 0]
        np.testing.assert_allclose(ratio, L.GOLD_WHITE)
        self.assertGreater(float(ratio[2]), .6)

    def test_real_canonical_glyph_mask_receives_the_stronger_response(self):
        # Recover the actual mip-filtered mask from the renderer's existing emissive path.
        emissive = R.RingState()
        emissive.letters = 1.
        texture = self.shade(emissive)[:, 0]
        self.assertGreater(np.count_nonzero(texture > .1), 10)
        self.assertGreater(np.count_nonzero(texture < .001), 10)
        state = R.RingState()
        state.reading_light = lambda p, n: np.broadcast_to(L.GOLD_WHITE, p.shape)
        expected = (R.READING_BODY_GAIN * R.F0_GOLD[None, :] * L.GOLD_WHITE[None, :]
                    + R.READING_GROOVE_GAIN * texture[:, None] * R.READING_GROOVE_TINT[None, :]
                    * L.GOLD_WHITE[None, :]).astype(np.float32)
        def oracle(result):
            np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-7)
            self.assertGreater(float(result[texture > .1].mean()), float(result[texture < .001].mean()) * 1.8)
        result = self.shade(state)
        oracle(result)
        self.rejects(oracle, np.broadcast_to(result.mean(axis=0), result.shape))

    def test_moving_lamp_moves_the_illuminated_band_and_letters(self):
        positions = ([[R.R_OUT + 1., .35, 0.]], [[-R.R_OUT - 1., .35, 0.]])
        state = R.RingState()
        near_a = np.cos(self.th) > .96
        near_b = np.cos(self.th) < -.96
        def oracle(factory):
            images = []
            for pos in positions:
                state.reading_light = factory(pos)
                images.append(self.shade(state))
            self.assertGreater(float(images[0][near_a].sum()), float(images[0][near_b].sum()) + 1.)
            self.assertGreater(float(images[1][near_b].sum()), float(images[1][near_a].sum()) + 1.)
        oracle(lambda pos: L.ReadingLights(pos, powers=4.))
        self.rejects(oracle, lambda pos: L.ReadingLights(positions[0], powers=4.))

    def test_callback_rejects_invalid_irradiance(self):
        state = R.RingState()
        for callback in (lambda p, n: np.zeros(len(p)), lambda p, n: -np.ones_like(p),
                         lambda p, n: np.full_like(p, np.nan)):
            state.reading_light = callback
            with self.assertRaisesRegex(ValueError, 'reading_light'):
                self.shade(state)


if __name__ == '__main__':
    unittest.main()
