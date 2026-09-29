"""Small Pen-study invariants; no book tracing or scene rendering here."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import book_c_v5_pen_candidates as C
import book_c_v5 as V


class PenCandidates(unittest.TestCase):
    def test_default_is_original_renderer_without_candidate_operations(self):
        with patch.object(V, 'PagesV5', autospec=True) as original:
            for kwargs in ({}, {'candidate': 'accepted'}):
                self.assertIs(C.make_renderer(**kwargs), original.return_value)
                original.assert_called_with('pen', 960, 402, 110)
            C.make_renderer(scale=1.)
            original.assert_called_with('pen', 1920, 804, 110)

    def test_frame_range_cache_and_light_methods_are_inherited(self):
        for method in ('frame', 'book', 'once', 'light'):
            self.assertIs(getattr(C.CandidatePagesV5, method), getattr(V.PagesV5, method))
        for frame in (5439, 5680):
            renderer = object.__new__(C.CandidatePagesV5)
            renderer.shot = 'pen'
            with self.assertRaises(ValueError):
                renderer.frame(frame)

    def test_gutter_term_is_local_smooth_shaded_paper_only(self):
        coords = np.array([0., .4, .8, 1.6, 3.19, 3.2, 4.])
        G = np.zeros((4, len(coords), 16))
        G[..., 0] = C.B.M_PAGE_R
        G[..., 2] = coords
        G[1, :, 2] = -coords
        G[1, :, 0] = C.B.M_PAGE_L
        G[2, :, 0] = C.B.M_LEATHER
        G[3, :, 11] = 1.
        original_G = G.copy()
        out = np.zeros((4, len(coords), 3), np.float32)
        # Exercise the complete local-light math without compiling or tracing
        # a scene. Procedural paper itself is an unchanged accepted dependency.
        with patch.object(C.B, 'paper', return_value=(.8, .69, .47, 0.)):
            C._gutter_bounce.py_func(out, G, 20., 29., 13., np.array([1.9, .98, .46]))
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((out[:2, :5] > 0).all())
        self.assertGreater(out[0, 2, 0], out[0, 0, 0])
        self.assertLess(out[0, 4, 0], .00002)
        np.testing.assert_array_equal(out[0], out[1])
        np.testing.assert_array_equal(out[:, 5:], 0.)
        np.testing.assert_array_equal(out[2:], 0.)
        np.testing.assert_array_equal(G, original_G)

    def test_gutter_copy_preserves_inputs(self):
        original = np.ones((2, 3, 3), np.float32)
        G = np.zeros((2, 3, 16))
        bk = SimpleNamespace(PW=20., PH=29., seed=13)
        light = SimpleNamespace(col=np.array([1., .5, .2]))
        def mutate(out, *args):
            out[0, 0] += .1
        with patch.object(C, '_gutter_bounce', side_effect=mutate):
            changed = C.soften_spine(original, G, bk, light)
        self.assertFalse(np.shares_memory(original, changed))
        np.testing.assert_array_equal(original, 1.)
        self.assertGreater(changed[0, 0, 0], original[0, 0, 0])

    def test_nib_response_is_finite_and_mark_tip_are_darker(self):
        position = np.zeros((4, 3))
        normal = np.tile([0., 0., 1.], (4, 1))
        uv = np.array([[.93, np.pi/2], [.900, np.pi/2], [1., np.pi/2], [.93, 3*np.pi/2]])
        before = [a.copy() for a in (position, normal, uv)]
        cam = SimpleNamespace(pos=np.array([0., -1., 1.]))
        light = SimpleNamespace(pos=np.array([-1., 1., 2.]), col=np.array([1.9, .988, .456]))
        rgb = C.metal_nib_rgb(position, normal, uv, cam, light)
        self.assertTrue(np.isfinite(rgb).all())
        self.assertTrue((rgb >= 0).all())
        self.assertTrue((rgb[1] < .1*rgb[0]).all())
        np.testing.assert_allclose(rgb[2], .25*rgb[0])
        np.testing.assert_array_equal(rgb[0], rgb[3])
        for a, b in zip((position, normal, uv), before):
            np.testing.assert_array_equal(a, b)

    def test_visible_nib_repaint_preserves_depth_shadow_and_other_materials(self):
        cam = C.B.Cam([0., -10., 10.], [0., 0., 1.], 50., 48, 24)
        light = C.B.Light([-10., 10., 20.])
        verts = np.array([[-3., 0., 2.], [-1., 0., 2.], [-2., 1., 2.],
                          [1., 0., 2.], [3., 0., 2.], [2., 1., 2.]])
        mesh = (verts, np.tile([0., 0., 1.], (6, 1)),
                np.tile([.93, np.pi/2], (6, 1)), np.array([[0, 1, 2], [3, 4, 5]]), np.array([0, 2]))
        before_mesh = [a.copy() for a in mesh]
        G = np.zeros((24, 48, 16)); G[..., 1] = 100.
        bk = SimpleNamespace(params=np.zeros(12), ck=np.zeros((2, 2)))
        with patch.object(C.B, 'height', return_value=(0., C.B.M_PAGE_R)):
            accepted, depth = C.PEN.composite(np.ones((24, 48, 3), np.float32), G, bk, cam, light, mesh)
        original, original_depth = accepted.copy(), depth.copy()
        result = C.replace_visible_nib(accepted, G, depth, mesh, cam, light)
        sl, _, z, inside = C.PEN._pixels(cam, verts, mesh[3][1])
        mask = np.zeros((24, 48), bool)
        mask[sl] = inside & (z == depth[sl]) & (z < G[sl][..., 1])
        self.assertGreater(mask.sum(), 0)
        self.assertTrue(np.any(result[mask] != original[mask]))
        np.testing.assert_array_equal(result[~mask], original[~mask])
        np.testing.assert_array_equal(depth, original_depth)
        np.testing.assert_array_equal(accepted, original)
        for a, b in zip(mesh, before_mesh):
            np.testing.assert_array_equal(a, b)
        # A page in front hides every triangle; no material may paint through it.
        occluded = G.copy(); occluded[..., 1] = 1.
        hidden = C.replace_visible_nib(original, occluded, np.ones_like(depth), mesh, cam, light)
        np.testing.assert_array_equal(hidden, original)

    def test_candidate_setup_and_original_contact_shadow_calls_match(self):
        original_book, original_mesh = object(), object()
        traces = []
        def run(method, candidate):
            calls = []
            receiver = SimpleNamespace(W=960, H=402, blank=object(), candidate=candidate)
            receiver.book = lambda *a, **k: (calls.append(('book', a, k)), original_book)[1]
            receiver.once = lambda key, fn: (calls.append(('once', key)), original_mesh)[1]
            receiver.light = lambda *a, **k: (calls.append(('light', a, k)), object())[1]
            def render(book, cam, light, left, right, t, **kwargs):
                calls.append(('render', book, cam.pos.copy(), cam.target.copy(), cam.hfov, cam.dof_k, t, kwargs))
                return np.ones((2, 3, 3), np.float32), np.ones((2, 3)), np.zeros((2, 3, 16))
            def composite(hdr, G, book, cam, light, mesh):
                self.assertIs(book, original_book); self.assertIs(mesh, original_mesh)
                calls.append(('original_pen_composite',))
                return hdr, np.ones((2, 3))
            def dof(hdr, depth, cam, amount):
                calls.append(('dof', amount, depth.dtype.str))
                return hdr
            with patch.object(C.B, 'render', side_effect=render), patch.object(C.PEN, 'composite', side_effect=composite), \
                 patch.object(C.B, 'dof', side_effect=dof), patch.object(C, 'soften_spine', side_effect=lambda h, *a: h), \
                 patch.object(C, 'replace_visible_nib', side_effect=lambda h, *a: h):
                method(receiver, 5., 5560)
            traces.append(calls)
        run(V.PagesV5.shot_pen, 'accepted')
        for candidate in C.CANDIDATES[1:]:
            run(C.CandidatePagesV5.shot_pen, candidate)
        for trace in traces[1:]:
            self.assertEqual(len(trace), len(traces[0]))
            for a, b in zip(traces[0], trace):
                if a[0] == 'render':
                    self.assertEqual(a[:2], b[:2])
                    np.testing.assert_array_equal(a[2], b[2]); np.testing.assert_array_equal(a[3], b[3])
                    self.assertEqual(a[4:], b[4:])
                else:
                    self.assertEqual(a, b)

    def test_invalid_options_and_accepted_output_paths_are_rejected(self):
        for scale in (0., -1., np.nan, np.inf, .00001):
            with self.assertRaises(ValueError): C.make_renderer(scale=scale)
        for ppc in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): C.make_renderer(ppc=ppc)
        with self.assertRaises(ValueError): C.make_renderer('other')
        with self.assertRaises(ValueError): C.CandidatePagesV5('accepted')
        root = Path.home()/'ldfarm/out'
        self.assertEqual(C.validate_output_dir(root/'cand_pen_test'), (root/'cand_pen_test').resolve())
        for path in (root/'book_C5_pen', root/'cand_pen_', root/'cand_pen_test/nested', Path('cand_pen_test')):
            with self.assertRaises(ValueError): C.validate_output_dir(path)
        with tempfile.TemporaryDirectory() as temp:
            alias = Path(temp)/'cand_pen_alias'; alias.symlink_to(root/'book_C5_pen')
            with self.assertRaises(ValueError): C.validate_output_dir(alias)


if __name__ == '__main__':
    unittest.main()
