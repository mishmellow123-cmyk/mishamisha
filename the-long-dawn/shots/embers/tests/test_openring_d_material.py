"""D's optional worked metal: preserved Round 1 pixels and physical material invariants.

Tiny real-raster probes cover the public renderer; analytic probes test a continuous shallow surface,
reflected-light energy and uniform heat, including deliberately broken controls for each contract.
"""
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'shots' / 'embers'), str(ROOT / 'lib')]
import ringsolid as R
from core import Camera


class WorkedCaps(unittest.TestCase):
    def rejects(self, oracle, value):
        with self.assertRaises(AssertionError):
            oracle(value)

    def state(self):
        st = R.RingState()
        st.end_caps = st.leading_glyph = True
        st.letters, st.glow, st.exposure = 1.1, .09, .7
        st.heat = lambda th: .4 + .3 * np.sin(th) ** 2
        return st

    def draw(self, st, arc=(.1, 5.4)):
        cam = Camera((5, 2, -2), (0, 0, 0), hfov=60)
        env = R.Env().lobe((.1, -.5, .86), (3., 2., 1.), 80.)
        return R.render(cam, 128, 80, np.eye(3), np.zeros(3), 1., st, env,
                        ss=1, nt=64, npp=20, th_range=arc)

    def test_default_off_pins_round1_capped_and_inscribed_render(self):
        # Saved Round 1 source measured with the prescribed validation interpreter, including cap shading.
        expected = {
            (.1, 5.4): ('1e5a681843ce6a574c2307dd92ca69a06ead156d40a6fe173ecf2bce4d81349e',
                       'ffcc756d2c8dcf2b98c6565e74e3df536c0ef5a2e4adb2fe773ec9c2da482e8c',
                       '0c2cd256505b5293f4de312007a3e927e4ffbb1fd82513416d7dc7976ca2866d'),
            (.1, 4.3): ('62905c5ad007b7734abe2f7d2096b9f9caa89ce83b6f5034ae8b4dba5cd57372',
                       '961c6ecdb31f4efad0680e40efa0f5aed80354a9e9906e9b82c5445321368d09',
                       'ec16caf7a1e447d0fb80f02f1b1dcadc0277b5917239ab5ffe7b53614f836f32'),
        }
        st = self.state()
        self.assertFalse(st.worked_caps)
        for arc, hashes in expected.items():
            def oracle(arrays):
                self.assertEqual(tuple(hashlib.sha256(a.tobytes()).hexdigest() for a in arrays), hashes)
            result = self.draw(st, arc)
            oracle(result)
            broken = [a.copy() for a in result]
            broken[0][0, 0, 0] = 1.
            self.rejects(oracle, broken)

    def test_opt_in_changes_cap_light_but_preserves_geometry_and_body(self):
        st = self.state()
        off = self.draw(st)
        st.worked_caps = True
        on = self.draw(st)
        def oracle(result):
            np.testing.assert_array_equal(result[1], off[1])
            np.testing.assert_array_equal(result[2], off[2])
            self.assertGreater(np.count_nonzero(np.abs(result[0] - off[0]) > 1e-5), 8)
        oracle(on)
        self.rejects(oracle, off)
        broken = [a.copy() for a in on]
        broken[1][0, 0] = 1.
        self.rejects(oracle, broken)
        broken = [a.copy() for a in on]
        broken[2][0, 0] = 0.
        self.rejects(oracle, broken)
        st.worked_caps = False
        closed_off = self.draw(st, None)
        st.worked_caps = True
        closed_on = self.draw(st, None)
        for a, b in zip(closed_off, closed_on):
            np.testing.assert_array_equal(a, b)

    def test_relief_is_shallow_nonflat_and_from_a_continuous_height_field(self):
        rr, yy = np.meshgrid(np.linspace(-.2, .2, 37), np.linspace(-.47, .47, 43))
        rr, yy = rr.ravel(), yy.ravel()
        def oracle(field):
            for sec in ((1., 1.), (.4, .7)):
                r, y = rr * sec[0], yy * sec[1]
                gr, gy = field(r, y, sec)
                slope = np.hypot(gr, gy)
                self.assertGreater(slope.max(), .002)
                self.assertLess(slope.max(), .08)
                eps = 1e-6
                dr_dy = (field(r, y + eps, sec)[0] - field(r, y - eps, sec)[0]) / (2 * eps)
                dy_dr = (field(r + eps, y, sec)[1] - field(r - eps, y, sec)[1]) / (2 * eps)
                self.assertLess(float(np.max(np.abs(dr_dy - dy_dr))), 2e-5)
        oracle(R._worked_cap_gradients)
        self.rejects(oracle, lambda r, y, sec: (np.zeros_like(r), np.zeros_like(y)))
        def too_deep(r, y, sec):
            return tuple(20 * a for a in R._worked_cap_gradients(r, y, sec))
        self.rejects(oracle, too_deep)
        def incoherent(r, y, sec):
            gr, gy = R._worked_cap_gradients(r, y, sec)
            return gr + .02 * y, gy
        self.rejects(oracle, incoherent)

    def test_material_coordinates_are_repeatable_and_filter_subpixel_relief(self):
        r, y = np.linspace(-.19, .19, 200), np.linspace(-.44, .44, 200)
        fine = R._worked_cap_gradients(r, y)
        pieces = [R._worked_cap_gradients(r[s], y[s]) for s in (slice(0, 71), slice(71, 135), slice(135, None))]
        for k in (0, 1):
            np.testing.assert_array_equal(fine[k], np.concatenate([part[k] for part in pieces]))
        def oracle(field):
            sharp = np.hypot(*field(r, y, footprint=0.)).mean()
            broad = np.hypot(*field(r, y, footprint=.008)).mean()
            filtered = np.hypot(*field(r, y, footprint=.2)).mean()
            self.assertGreater(sharp, .001)
            self.assertLess(broad, sharp * .45)
            self.assertLess(filtered, sharp * .15)
        oracle(R._worked_cap_gradients)
        self.rejects(oracle, lambda r, y, footprint=0.: R._worked_cap_gradients(r, y, footprint=0.))
        def broad_blurred_marks(r, y, footprint=0.):
            attenuation = np.exp(-.5 * (14 * footprint) ** 2)
            return .015 * np.sin(14 * r) * attenuation, .015 * np.sin(14 * y) * attenuation
        self.rejects(oracle, broad_blurred_marks)

    def test_white_heat_has_no_pores_or_separately_emitted_rim(self):
        rr, yy = np.meshgrid(np.linspace(-.5 * R.THICK, .5 * R.THICK, 31), np.linspace(-.5, .5, 43))
        r, y = rr.ravel(), yy.ravel()
        cam = Camera((5, 2, -2), (0, 0, 0))
        st = R.RingState()
        st.worked_caps = True
        st.heat = lambda th: np.full_like(th, .96)
        dark = R.Env(above=(0., 0., 0.), horizon=(0., 0., 0.), below=(0., 0., 0.))
        args = (r, y, 5.4, 1., cam, np.eye(3), np.zeros(3), 1., st, dark)
        def oracle(shade):
            result = shade(*args)
            # The cap must supply its own white heat even in a dark environment, with no band glare.
            self.assertGreater(float(result.mean()), 1.8)
            np.testing.assert_allclose(result, np.broadcast_to(result[0], result.shape), atol=1e-7)
            self.assertGreater(float(result[0, 2] / result[0, 0]), .65)
        oracle(R.shade_cap)
        self.rejects(oracle, lambda *args: .25 * R.shade_cap(*args))
        def pores(*args):
            result = R.shade_cap(*args)
            return result * (1 - .2 * np.sin(80 * args[0]) ** 2)[:, None]
        self.rejects(oracle, pores)
        def rim(*args):
            result = R.shade_cap(*args)
            edge = np.abs(args[1]) > .45
            result[edge] += .3
            return result
        self.rejects(oracle, rim)

    def test_cold_cap_reflects_supplied_gold_light_without_creating_energy(self):
        class Uniform:
            def __init__(self, value):
                self.value = np.asarray(value)
            def radiance(self, directions, positions):
                return np.broadcast_to(self.value, directions.shape)
        r, y = np.linspace(-.18, .18, 120), np.linspace(-.4, .4, 120)
        cam = Camera((5, 2, -2), (0, 0, 0))
        st = R.RingState()
        st.worked_caps = True
        args = (r, y, 5.4, 1., cam, np.eye(3), np.zeros(3), 1., st)
        def oracle(shade):
            dark = shade(*args, Uniform((0., 0., 0.)))
            np.testing.assert_array_equal(dark, np.zeros_like(dark))
            white = shade(*args, Uniform((1., 1., 1.)))
            self.assertTrue(np.all(white <= 1.))
            self.assertTrue(np.all(white >= R.F0_GOLD - 1e-7))
            self.assertTrue(np.all(white[:, 0] >= white[:, 1]))
            self.assertTrue(np.all(white[:, 1] > white[:, 2]))
            low = shade(*args, Uniform((.1, .2, .3)))
            high = shade(*args, Uniform((.2, .4, .6)))
            np.testing.assert_allclose(high, 2 * low, rtol=0., atol=1e-7)
        oracle(R.shade_cap)
        self.rejects(oracle, lambda *args: R.shade_cap(*args) + .1)
        self.rejects(oracle, lambda *args: R.shade_cap(*args) * 1.5)


if __name__ == '__main__':
    unittest.main()
