"""Open-band section/glyph contracts, with broken controls and pre-change byte pins.

The tiny raster probes exercise the real depth buffer; no whole scene is constructed.
"""
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'shots' / 'embers'), str(ROOT / 'lib'), str(ROOT)]
import ringsolid as R
from core import Camera


class Caps(unittest.TestCase):
    def rejects(self, oracle, value):
        with self.assertRaises(AssertionError):
            oracle(value)

    def test_disabled_bytes_match_before_change_for_closed_and_open_band(self):
        # Original renderer snapshot measured in the prescribed validation environment (NumPy 2.3.5,
        # OpenCV 5.0.0), including actual asset sampling and heat. RGB differs across OpenCV environments;
        # before/after equality was also measured independently in the production render environment.
        expected = {
            None: ('8519274aad0080248471946340cee49e094030d412ee1ab0613dafb338a39cc8',
                   '258daed8d55500bf2f42932c121d9c44aed74d2e6b9ecc5df7121aefae0c0dd0',
                   '82e8fe5ca7cc71c3dab6133badffe492cd04c755e2e2740d695fab489430b4b9'),
            (.1, 5.4): ('047f89419a2f41f0c74625b602777480df94412808d59c0f3788e19225d04624',
                       '8e847067a36e2f30199c973906ab57d682ad55946b91a8d47c89edea542feb53',
                       '385d5b0c53de826b5453369578922c693aee99330c740fbc5631b394f9514843'),
        }
        cam = Camera((4, 3, 7), (0, 0, 0), hfov=47)
        st = R.RingState()
        st.glow, st.letters = .11, .8
        st.heat = lambda th: .4 + .3 * np.sin(th) ** 2
        self.assertFalse(st.end_caps)
        self.assertFalse(st.leading_glyph)
        for arc, hashes in expected.items():
            result = R.render(cam, 128, 80, np.eye(3), np.zeros(3), 1., st, R.Env(),
                              ss=1, nt=64, npp=20, th_range=arc)
            def oracle(arrays):
                self.assertEqual(tuple(hashlib.sha256(a.tobytes()).hexdigest() for a in arrays), hashes)
            oracle(result)
            changed = [a.copy() for a in result]
            changed[0][0, 0, 0] = np.nextafter(changed[0][0, 0, 0], np.float32(1))
            self.rejects(oracle, changed)

    def test_cap_boundary_and_center_are_the_band_section(self):
        def oracle(builder):
            for theta in (.1, 5.4):
                for sec in ((1., 1.), (.35, .6)):
                    points, r, y = builder(theta, 24, sec)
                    band, _ = R.local_points(np.full(25, theta), np.linspace(-np.pi, np.pi, 25), sec)
                    np.testing.assert_allclose(points[1], band, rtol=0., atol=1e-14)
                    np.testing.assert_allclose(points[0], np.tile([R.R_MID * np.cos(theta), 0.,
                                                                   R.R_MID * np.sin(theta)], (25, 1)), atol=1e-14)
                    self.assertEqual(np.max(np.abs(r[0])), 0.)
                    self.assertEqual(np.max(np.abs(y[0])), 0.)
        oracle(R.cap_points)
        def hollow(theta, npp, sec):
            points, r, y = R.cap_points(theta, npp, sec)
            points[0] = points[1]  # A tube wall without a section face.
            return points, r, y
        self.rejects(oracle, hollow)

    def test_both_sections_fill_and_depth_occlude_the_open_tube(self):
        cam = Camera((5, 2, -2), (0, 0, 0), hfov=60)
        arc = (.1, 5.4)
        W, H = 256, 160
        st = R.RingState()
        st.glow = .15
        def draw(caps):
            st.end_caps = caps
            return R.render(cam, W, H, np.eye(3), np.zeros(3), 1., st, R.Env(),
                            ss=1, nt=160, npp=32, th_range=arc)
        opened, capped = draw(False), draw(True)
        def oracle(result):
            for theta in arc:
                centre = np.array([R.R_MID * np.cos(theta), 0., R.R_MID * np.sin(theta)])
                u, v, _ = cam.project(centre[None, :], W, H)
                x, y = int(round(u[0])), int(round(v[0]))
                n = np.array([-np.sin(theta), 0., np.cos(theta)])
                ray = np.array([(x - (W - 1) / 2) / cam.f_px(W),
                                -(y - (H - 1) / 2) / cam.f_px(W), 1.]) @ cam.R
                wanted_depth = np.dot(centre - cam.pos, n) / np.dot(ray, n)
                self.assertEqual(result[1][y, x], 1.)
                self.assertAlmostEqual(result[2][y, x], wanted_depth, places=5)
                self.assertLess(result[2][y, x], opened[2][y, x])
        oracle(capped)
        self.rejects(oracle, opened)
        # Closed rings must never receive artificial end faces or leading-edge lettering.
        st.end_caps = st.leading_glyph = False
        closed = R.render(cam, W, H, np.eye(3), np.zeros(3), 1., st, R.Env(), ss=1, nt=64, npp=20)
        st.end_caps = st.leading_glyph = True
        closed_opt = R.render(cam, W, H, np.eye(3), np.zeros(3), 1., st, R.Env(), ss=1, nt=64, npp=20)
        for a, b in zip(closed, closed_opt):
            np.testing.assert_array_equal(a, b)

    def test_cap_heat_samples_both_endpoint_angles_and_changes_radiance(self):
        cam = Camera((5, 2, -2), (0, 0, 0))
        r, y = np.linspace(-.15, .15, 40), np.linspace(-.3, .3, 40)
        st = R.RingState()
        def oracle(shade):
            for theta, side in ((.1, -1.), (5.4, 1.)):
                args = (r, y, theta, side, cam, np.eye(3), np.zeros(3), 1., st, R.Env())
                st.heat = None
                cold = R.shade_cap(*args)
                angles = []
                st.heat = lambda th: (angles.append(th.copy()) or np.ones(len(th)))
                hot = shade(*args)
                np.testing.assert_array_equal(angles, [np.full(len(r), theta)])
                self.assertGreater(hot.mean(), cold.mean() + .15)
                st.heat = None
        oracle(R.shade_cap)
        self.rejects(oracle, lambda *args: np.zeros((len(r), 3)))
        def dark(*args):
            return R.shade_cap(*args) * 0.
        self.rejects(oracle, dark)

    def test_face_heat_is_uniform_and_reflection_detail_is_restrained(self):
        cam = Camera((5, 2, -2), (0, 0, 0))
        r, y = np.linspace(-.14, .14, 200), np.linspace(-.29, .29, 200)
        st = R.RingState()
        dark_env = R.Env(above=(0., 0., 0.), horizon=(0., 0., 0.), below=(0., 0., 0.))
        st.heat = lambda th: np.ones_like(th)
        args = (r, y, 5.4, 1., cam, np.eye(3), np.zeros(3), 1., st, dark_env)
        def uniform_temperature(shade):
            col = shade(*args)
            # The face interior has one temperature. Patterned emission made it look porous.
            np.testing.assert_allclose(col, np.broadcast_to(col[0], col.shape), atol=1e-7)
        uniform_temperature(R.shade_cap)
        def mottled(*args):
            col = R.shade_cap(*args)
            return col * (1 - .25 * np.sin(70 * args[0]) ** 2)[:, None]
        self.rejects(uniform_temperature, mottled)
        st.heat = None
        env = R.Env().lobe((.1, -.5, .86), (3., 2., 1.), 80.)
        args = (r, y, 5.4, 1., cam, np.eye(3), np.zeros(3), 1., st, env)
        def reflected_relief(shade):
            col = shade(*args)
            display = col / (1 + col)
            contrast = float(np.ptp(display[:, 0]))
            self.assertGreater(contrast, .015)
            self.assertLess(contrast, .15)
        reflected_relief(R.shade_cap)
        def featureless(*args):
            real = R.shade_cap(*args)
            return np.broadcast_to(real.mean(axis=0), real.shape)
        self.rejects(reflected_relief, featureless)
        # The rejected mosaic pass used .25 normal slope; both too-flat and too-coarse controls must fail.
        coarse = R._CAP_FACETS.copy()
        coarse[:, 2:] = .5 + (coarse[:, 2:] - .5) * (.25 / .04)
        with patch.object(R, '_CAP_FACETS', coarse):
            self.rejects(reflected_relief, R.shade_cap)

    def test_cap_respects_existing_nearer_geometry_in_shared_depth_buffer(self):
        x = np.array([[0., 3.], [0., 3.]])
        y = np.array([[0., 0.], [3., 3.]])
        param = np.zeros((2, 2))
        def oracle(raster):
            zb = np.ones((4, 4))
            tb, pb, fb = np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4), np.uint8)
            raster(x, y, np.full((2, 2), 2.), param, param, zb, tb, pb, fb, 2)
            np.testing.assert_array_equal(zb, np.ones((4, 4)))
            np.testing.assert_array_equal(fb, np.zeros((4, 4)))
            raster(x, y, np.full((2, 2), .5), param, param, zb, tb, pb, fb, 2)
            self.assertEqual(zb[1, 1], .5)
            self.assertEqual(fb[1, 1], 2)
        oracle(R._raster)
        def painter(*args):
            args[5][:] = np.inf  # A separate additive cap layer discards the band's nearer surface.
            R._raster(*args)
        self.rejects(oracle, painter)

    def test_half_glyph_uses_canonical_ink_and_does_not_slide_settled_letters(self):
        from shots.montage3d import ring_script as script
        ins = R.inscription()
        for face, prm in (('outer', script.OUTER), ('inner', script.INNER)):
            first = script.words_fixed(prm['seed'], 1)[0][0]
            mid, advance = ins.leading_glyph(face)
            self.assertEqual(advance, script.ALPHABET[first][0] * script.EM_FRAC)
            img = ins.lv[face][0]
            h, w = img.shape
            mass = img[:, :int(advance * h)].sum(axis=0, dtype=np.float64)
            column = int(mid * w)
            def oracle(x):
                self.assertLessEqual(abs(mass[:x + 1].sum() / mass.sum() - .5), .035)
            oracle(column)
            self.rejects(oracle, 0)
        cam = Camera((5, 2, -2), (0, 0, 0))
        st = R.RingState()
        st.letters = 1.5
        theta = np.linspace(4.7, 5.1, 300)
        psi = np.linspace(-.7, .7, 300)
        args = (theta, psi, cam, 500., 2, np.eye(3), np.zeros(3), 1., st, R.Env())
        settled = R.shade(*args)
        np.testing.assert_array_equal(settled, R.shade(*args, leading_theta=5.4))
        # A disabled write sweep must suppress the forming glyph as well as the accepted inscription.
        st.write = lambda th: np.zeros_like(th)
        edge_theta = np.full(300, 5.4)
        edge_args = (edge_theta, psi, cam, 500., 2, np.eye(3), np.zeros(3), 1., st, R.Env())
        np.testing.assert_array_equal(R.shade(*edge_args), R.shade(*edge_args, leading_theta=5.4))
        st.write = None
        without = R.shade(*edge_args)
        formed = R.shade(*edge_args, leading_theta=5.4)
        def changes_edge(image):
            self.assertGreater(np.count_nonzero(np.abs(image - without) > .01), 15)
        changes_edge(formed)
        self.rejects(changes_edge, without)


if __name__ == '__main__':
    unittest.main()
