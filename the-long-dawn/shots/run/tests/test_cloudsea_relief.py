"""Numerical relief contracts. Compiles small Numba kernels; no full shot renders.

Run only when the shared machine has a compute slot (NUMBA_NUM_THREADS=1).
"""
import ast
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
import world as WD


class CloudReliefTests(unittest.TestCase):
    def test_bound_strength_and_footprint(self):
        positive = []
        for x in (-8000.25, -360.0, -0.001, 0.0, 360.0, 19531.7):
            for z in (-13000.0, -125.0, 0.0, 999.7):
                for fp in (0.01, 8.0, 30.0, 75.0, 180.0, 300.0):
                    value = WD.cloud_relief(x, z, fp, 1.0)
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, WD.CLOUD_RELIEF_MAX)
                    self.assertEqual(WD.cloud_relief(x, z, fp, 0.0), 0.0)
                    self.assertEqual(WD.cloud_relief(x, z, fp, -1.0), 0.0)
                    self.assertEqual(WD.cloud_relief(x, z, fp, 5.0), value)
                    self.assertAlmostEqual(WD.cloud_relief(x, z, fp, 0.5), value / 2, places=12)
                    if fp >= 180.0:
                        self.assertEqual(value, 0.0)
                    positive.append(value)
        self.assertGreater(max(positive), 20.0)  # a permanently disabled helper must fail

    def test_lobes_cross_cell_boundaries_continuously(self):
        for scale, seed in ((360.0, 413), (125.0, 827), (45.0, 1291)):
            for cell in (-4, -1, 0, 3):
                x = cell * scale
                left = WD._cloud_lobes(x - 1e-5, 0.371 * scale, 1.0, scale, seed)
                right = WD._cloud_lobes(x + 1e-5, 0.371 * scale, 1.0, scale, seed)
                self.assertAlmostEqual(left, right, delta=1e-5)

    def test_default_off_is_the_existing_cloud_top_and_holes(self):
        empty = np.zeros((0, WD.NCR))
        hole = WD.hole_row(100.0, -200.0, 500.0, 220.0)
        valley = WD.valley_row(np.array([0., 0., 0.]), np.array([500., 0., 0.]), -800., -900.)
        tables = (empty, np.array([hole]), np.array([valley, hole]))
        for cr in tables:
            for x, z, fp, t in ((100., -200., .05, 0.), (400., 80., 4., 15.), (5000., -2500., 50., 100.)):
                expected = WD.h_cloud(x, z, fp, t) - WD.cloud_thin(x, z, cr)
                self.assertEqual(WD.h_cloud_cr(x, z, fp, t, cr), expected)
                self.assertEqual(WD.h_cloud_cr(x, z, fp, t, cr, 0.0), expected)
                enabled = WD.h_cloud_cr(x, z, fp, t, cr, 1.0)
                self.assertAlmostEqual(enabled - expected, WD.cloud_relief(x, z, fp, 1.0), places=10)

    def cloud_point(self):
        cr = np.zeros((0, WD.NCR))
        for x in np.arange(-24000., 24001., 4000.):
            for z in np.arange(-12000., 12001., 4000.):
                base = WD.h_cloud(x, z, 1.0, 0.0)
                if (max(WD.h_near(x, z, 1.0), WD.h_rock(x, z, 1.0, cr)) < base - 50.0
                        and WD.cloud_relief(x, z, 1.0, 1.0) > 25.0):
                    return float(x), float(z), cr, base
        self.fail('no cloud-visible point found: negative control has no denominator')

    def test_runtime_flag_off_on_off_uses_one_compiled_signature(self):
        x, z, cr, base = self.cloud_point()
        p = np.array([x, z, 0., 0.])
        off = WD.hfun(x, z, 1.0, p, cr)
        signatures = tuple(WD.hfun.signatures)
        self.assertEqual(off, base)
        p[3] = 1.0
        on = WD.hfun(x, z, 1.0, p, cr)
        self.assertGreater(on, off + 25.0)
        self.assertAlmostEqual(on - off, WD.cloud_relief(x, z, 1.0, 1.0), places=10)
        p[3] = 0.0
        self.assertEqual(WD.hfun(x, z, 1.0, p, cr), off)
        self.assertEqual(tuple(WD.hfun.signatures), signatures)

    def test_shadow_sees_enabled_geometry_and_returns_to_off(self):
        x, z, cr, base = self.cloud_point()
        p = np.array([x, z, 0., 0.])

        def shadow():
            # A vertical light ray starts above the old cloud but inside a lobe.
            return WD.soft_shadow(p, cr, x, base + 10., z, 0., 1., 0., 0.1, 100., 10, 6., 1.)

        off = shadow()
        signatures = tuple(WD.soft_shadow.signatures)
        self.assertEqual(off, 1.0)
        p[3] = 1.0
        self.assertEqual(shadow(), 0.0)
        p[3] = 0.0
        self.assertEqual(shadow(), off)
        self.assertEqual(tuple(WD.soft_shadow.signatures), signatures)

    def test_actual_march_depth_changes_on_and_returns_exactly_off(self):
        x, z, cr, base = self.cloud_point()
        p = np.array([x, z - 100., 0., 0.])
        camera = np.array([x, base + 100., z - 100., 0., 1., 1., 0., 100., .5, -99.5, 1., 1.])

        def march():
            depth = np.zeros((1, 1))
            WD.march(p, cr, camera, .2, 1000., .0035, .35, 700., 9, depth)
            return depth

        off = march()
        signatures = tuple(WD.march.signatures)
        self.assertGreater(off[0, 0], 0.2)
        self.assertLess(off[0, 0], 1e29)
        p[3] = 1.0
        on = march()
        self.assertLess(on[0, 0], off[0, 0])
        p[3] = 0.0
        np.testing.assert_array_equal(march(), off)
        self.assertEqual(tuple(WD.march.signatures), signatures)

    def test_three_parameter_api_matches_four_parameter_off_in_all_passes(self):
        x, z, cr, base = self.cloud_point()
        p3 = np.array([x, z - 100., 0.])
        p4 = np.r_[p3, 0.]
        self.assertEqual(WD.hfun(x, z, 1., p3, cr), WD.hfun(x, z, 1., p4, cr))
        # A real downward ray from 100 m above the original cloud neighbourhood.
        camera = np.array([x, base + 100., z - 100., 0., 1., 1., 0., 100., .5, -99.5, 1., 1.])
        depths = []
        for p in (p3, p4):
            depth = np.zeros((1, 1))
            WD.march(p, cr, camera, .2, 1000., .0035, .35, 700., 9, depth)
            depths.append(depth)
        np.testing.assert_array_equal(*depths)
        self.assertLess(depths[0][0, 0], 1e29)
        lk, amb, sky, fog, q = WD.night_light()
        ug = np.array([[x, z, 500., 1., .3, .1, 100., .4]])
        results = []
        for p in (p3, p4):
            image = np.zeros((1, 1, 3), np.float32)
            zb, distance = np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32)
            WD.shade(camera, depths[0], p, cr, sky, np.zeros((0, 8)), lk, q, amb, fog,
                     image, zb, distance, np.zeros((0, 4)))
            before_glow = image.copy()
            WD.cloud_glow(camera, depths[0], p, cr, ug, fog, image)
            self.assertGreater(float(image.sum()), float(before_glow.sum()))
            results.append((before_glow, image, zb, distance))
        for a, b in zip(*results):
            np.testing.assert_array_equal(a, b)

    def test_generated_hills_world_inherits_runtime_relief(self):
        # Execute beacon's actual generator without importing its simulations.
        source = RUN.parent / 'hills' / 'beacon.py'
        tree = ast.parse(source.read_text())
        generator = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == '_world_keep_module')
        code = compile(ast.Module(body=[generator], type_ignores=[]), str(source), 'exec')
        scope = {'__file__': str(source), 'KEEP': (60., 70., 160.),
                 'peaks': types.SimpleNamespace(OFFX=0., OFFZ=0.)}
        exec(code, scope)
        with tempfile.TemporaryDirectory() as cache:
            with mock.patch.dict(sys.modules, {'core': types.SimpleNamespace(CACHE_DIR=cache)}):
                # The generator registers world_keep for Numba cache resolution.
                previous = sys.modules.get('world_keep')
                try:
                    generated = scope['_world_keep_module']()
                    cr = np.zeros((0, WD.NCR))
                    args = (7000., -8000., 1., 0., cr)
                    off = generated.h_cloud_cr(*args, 0.)
                    self.assertEqual(off, WD.h_cloud_cr(*args, 0.))
                    self.assertEqual(generated.h_cloud_cr(*args, 1.), WD.h_cloud_cr(*args, 1.))
                    self.assertGreater(generated.h_cloud_cr(*args, 1.), off)
                    self.assertEqual(generated.h_cloud_cr(*args, 0.), off)
                finally:
                    if previous is None:
                        sys.modules.pop('world_keep', None)
                    else:
                        sys.modules['world_keep'] = previous


if __name__ == '__main__':
    unittest.main()
