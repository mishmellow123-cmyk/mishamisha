"""Tiny geometry/contracts only: no production imports, JIT or image renders."""
import ast
import hashlib
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
import crossing_candidates as CC


class RopeGeometryTests(unittest.TestCase):
    def test_module_import_has_no_render_or_thread_effects(self):
        code = ('import sys, os; before=dict(os.environ); sys.path.insert(0,sys.argv[1]); '
                'import crossing_candidates; '
                'assert not {"numpy","numba","cv2","crossing"}.intersection(sys.modules); '
                'assert before==dict(os.environ)')
        subprocess.run([sys.executable, '-B', '-c', code, str(RUN)], check=True,
                       capture_output=True, text=True)

    def test_frozen_crossing_source_identity(self):
        self.assertEqual(hashlib.sha256((RUN / 'crossing.py').read_bytes()).hexdigest(),
                         CC.BASELINE_SHA256)

    def test_clear_terrain_preserves_every_point_and_attachments(self):
        points = CC.sag_points((0, 1, 0), (4, 1.2, 0))
        solved = CC.clear_rope(points, lambda q: [-2.0] * len(q))
        self.assertEqual(points, solved.points)
        self.assertEqual(solved.max_lift, 0.0)

    def test_sag_penetration_negative_control_and_restored_clearance(self):
        points = CC.sag_points((0, 0.4, 0), (5, 0.4, 0))
        calls = []
        def ground(q):
            calls.append(q)
            return [0.0] * len(q)
        solved = CC.clear_rope(points, ground)
        self.assertLess(solved.original_minimum_sampled_clearance, 0.0)
        self.assertGreaterEqual(solved.minimum_sampled_clearance, 0.03 - 1e-12)
        self.assertGreater(solved.max_lift, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(solved.points[0], points[0])
        self.assertEqual(solved.points[-1], points[-1])
        self.assertEqual([(p[0], p[2]) for p in solved.points], [(p[0], p[2]) for p in points])

    def test_between_vertex_bump_defeats_vertex_only_clamp(self):
        points = ((0., .15, 0.), (.5, .15, 0.), (1., .15, 0.))
        def height(x):
            return .25 * max(0., 1. - abs(x - .25) / .10)
        solved = CC.clear_rope(points, lambda q: [height(p[0]) for p in q], probe_spacing=.01)
        # Every original vertex clears snow, yet the actual first chord crosses it.
        self.assertTrue(all(p[1] >= height(p[0]) + .03 for p in points))
        self.assertLess(.15 - height(.25), 0.)
        self.assertGreater(solved.points[1][1], points[1][1])
        for i in range(1001):
            x = i / 1000
            k = min(int(x * 2), 1)
            a, b = solved.points[k:k + 2]
            u = (x - a[0]) / (b[0] - a[0])
            self.assertGreaterEqual((1-u)*a[1] + u*b[1] - height(x), .03 - 1e-10)
        self.assertEqual(solved.points[0], points[0])
        self.assertEqual(solved.points[-1], points[-1])

    def test_last_attachment_segment_is_also_checked(self):
        pts = ((0., .15, 0.), (.5, .15, 0.), (1., .15, 0.))
        solved = CC.clear_rope(pts, lambda q: [.25 * max(0., 1-abs(p[0]-.75)/.1) for p in q],
                               probe_spacing=.01)
        self.assertGreater(solved.points[1][1], .15)
        self.assertEqual(solved.points[-1], pts[-1])
        self.assertGreaterEqual(solved.minimum_sampled_clearance, .03-1e-12)

    def test_bad_attachment_or_terrain_is_explicit_failure(self):
        pts = ((0., .02, 0.), (.5, .1, 0.), (1., .1, 0.))
        with self.assertRaisesRegex(ValueError, 'attachment'):
            CC.clear_rope(pts, lambda q: [0.] * len(q))
        for heights in (lambda q: [], lambda q: [math.nan] * len(q)):
            with self.assertRaisesRegex(ValueError, 'heights'):
                CC.clear_rope(pts, heights)
        with self.assertRaisesRegex(ValueError, 'budget'):
            CC.clear_rope(pts, lambda q: [0.] * len(q), max_probes=4)


class OriginalPassContractTests(unittest.TestCase):
    def test_sag_and_pass_style_match_actual_original_on_clear_terrain(self):
        import numpy as np
        source = ast.parse((RUN / 'crossing.py').read_text())
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'draw_rope')
        captured, original_calls, candidate_calls = [], [], []
        class Cam:
            f = 100.
            def project(self, points):
                captured.append(points.copy())
                return points[:, 0] * 10, points[:, 1] * 10, points[:, 2] + 10
        def capture(target):
            return lambda *args: target.append(args[2:])
        namespace = dict(np=np, math=math, UP=np.array([0., 1., 0.]),
                         _rope_seg=capture(original_calls))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUN/'crossing.py'), 'exec'), namespace)
        waists = [(0., .8, 0.), (3., 1.1, .2), (4., 1., .7)]
        lights = [(1., 2., 2., .6, .4, .2, 2., .12)]
        args = (object(), object(), Cam(), waists, lights, None, .7)
        namespace['draw_rope'](*args)
        renderer = SimpleNamespace(np=np, ground_many=lambda q: [-2.] * len(q),
                                   _rope_seg=capture(candidate_calls))
        CC.draw_clear_rope(renderer, *args)
        self.assertEqual(len(original_calls), 26)
        self.assertEqual(len(candidate_calls), len(original_calls))
        for a, b in zip(captured[:2], captured[2:]):
            np.testing.assert_allclose(a, b, atol=1e-14, rtol=0)
        for a, b in zip(original_calls, candidate_calls):
            for old, new in zip(a, b):
                np.testing.assert_allclose(old, new, atol=1e-13, rtol=0)


class RenderHookTests(unittest.TestCase):
    def renderer(self):
        return SimpleNamespace(render=mock.Mock(return_value=object()),
                               draw_rope=mock.Mock(), build_scene=mock.Mock())

    def test_accepted_calls_original_directly_with_exact_return_and_no_hooks(self):
        cr = self.renderer()
        old_rope, old_scene = cr.draw_rope, cr.build_scene
        result = CC.render_cut(5040, renderer=cr)
        self.assertIs(result, cr.render.return_value)
        cr.render.assert_called_once_with(160, scale=.5, ss=1.5, variant='main', trail=True)
        self.assertIs(cr.draw_rope, old_rope)
        self.assertIs(cr.build_scene, old_scene)
        old_rope.assert_not_called()
        old_scene.assert_not_called()

    def test_rope_only_changes_rope_hook_and_always_restores(self):
        cr = self.renderer()
        old_rope, old_scene = cr.draw_rope, cr.build_scene
        def failing_render(*args, **kwargs):
            self.assertIsNot(cr.draw_rope, old_rope)
            self.assertIs(cr.build_scene, old_scene)
            raise RuntimeError('controlled failure')
        cr.render.side_effect = failing_render
        with self.assertRaisesRegex(RuntimeError, 'controlled failure'):
            CC.render_cut(5584, rope='snow_clearance', renderer=cr)
        self.assertIs(cr.draw_rope, old_rope)
        self.assertIs(cr.build_scene, old_scene)

    def test_rock_hook_preserves_tuple_and_composes_independently(self):
        for rope in CC.ROPE_OPTIONS:
            cr = self.renderer()
            old_rope, old_scene = cr.draw_rope, cr.build_scene
            scene, replacement, cfg = object(), object(), object()
            remaining = tuple(object() for _ in range(4))
            old_scene.return_value = (scene, *remaining)
            modifier = mock.Mock(return_value=replacement)
            def render(*args, **kwargs):
                changed = cr.build_scene(1.25, cfg)
                self.assertIs(changed[0], replacement)
                self.assertEqual(changed[1:], remaining)
                self.assertEqual(cr.draw_rope is old_rope, rope == 'accepted')
                return changed
            cr.render.side_effect = render
            CC.render_cut(5584, rope=rope, rock='explicit_study', rock_modifier=modifier, renderer=cr)
            modifier.assert_called_once_with(scene, cr, cfg)
            old_scene.assert_called_once_with(1.25, cfg)
            self.assertIs(cr.draw_rope, old_rope)
            self.assertIs(cr.build_scene, old_scene)

    def test_invalid_options_do_not_load_renderer(self):
        with mock.patch.object(CC, 'load_renderer') as loader:
            for kwargs in ({'frame':4879}, {'frame':5840}, {'frame':5040.5},
                           {'frame':5040,'rope':'unknown'}, {'frame':5040,'rock':'missing'},
                           {'frame':5040,'terrain':'unknown'},
                           {'frame':5040,'rock_modifier':lambda *a:None}):
                with self.assertRaises(ValueError):
                    CC.render_cut(**kwargs)
            loader.assert_not_called()

    def test_missing_cache_fails_before_any_production_import(self):
        with mock.patch.object(Path, 'is_file', return_value=False), \
             mock.patch.object(CC.importlib, 'import_module') as importer:
            with self.assertRaisesRegex(FileNotFoundError, 'regeneration'):
                CC.load_renderer()
            importer.assert_not_called()


class TerrainCapHookTests(unittest.TestCase):
    """Table routing only; no renderer import, frame load or JIT."""

    def renderer(self):
        import numpy as np
        base = np.arange(5 * 22, dtype=np.float64).reshape(5, 22)
        base[3, 12], base[3, 16] = -1., 0.
        rows = np.vstack([base, np.zeros((2, 22))])
        world = SimpleNamespace(march=mock.Mock(), shade=mock.Mock(),
                                cloud_glow=mock.Mock(), heights=mock.Mock())
        return SimpleNamespace(np=np, CR0=base, CR=rows, WD=world,
                               render=mock.Mock(), build_scene=mock.Mock(),
                               draw_rope=mock.Mock(), render_figures=mock.Mock(),
                               _PATH=object(), _KEEP={'accepted': object()})

    def test_only_owned_flag_changes_without_mutating_source(self):
        cr = self.renderer()
        before = cr.CR.copy()
        actual = CC.round_cap_rows(cr, cr.CR)
        expected = before.copy()
        expected[3, 16] = 1.
        cr.np.testing.assert_array_equal(actual, expected)
        cr.np.testing.assert_array_equal(cr.CR, before)
        self.assertFalse(cr.np.shares_memory(actual, cr.CR))

    def test_rejects_mismatched_or_already_modified_row(self):
        cr = self.renderer()
        for column in (0, 6, 12, 16):
            rows = cr.CR.copy()
            rows[3, column] += 1.
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'ridge 3'):
                CC.round_cap_rows(cr, rows)
        for rows in (cr.CR[:3], cr.CR[:, :16], cr.CR.ravel()):
            with self.assertRaises(ValueError):
                CC.round_cap_rows(cr, rows)

    def test_all_draw_passes_compose_with_rope_and_rock_and_preserve_placement(self):
        for rope in CC.ROPE_OPTIONS:
            for rock in ('accepted', 'study'):
                cr = self.renderer()
                world = vars(cr.WD).copy()
                old = {name: getattr(cr, name) for name in
                       ('CR', 'CR0', '_PATH', '_KEEP', 'draw_rope', 'build_scene', 'render_figures')}
                scene, cfg, replacement = object(), object(), object()
                rest = tuple(object() for _ in range(4))
                cr.build_scene.return_value = (scene, *rest)
                modifier = mock.Mock(return_value=replacement) if rock != 'accepted' else None

                def render(*args, **kwargs):
                    for name, index in (('march', 1), ('shade', 3), ('cloud_glow', 3)):
                        argv = [object() for _ in range(index + 3)]
                        argv[index] = cr.CR
                        value = getattr(cr.WD, name)(*argv)
                        self.assertIs(value, world[name].return_value)
                        forwarded = world[name].call_args.args
                        for k, item in enumerate(argv):
                            if k != index:
                                self.assertIs(forwarded[k], item)
                        expected = cr.CR.copy()
                        expected[3, 16] = 1.
                        cr.np.testing.assert_array_equal(forwarded[index], expected)
                    self.assertIs(cr.WD.heights, world['heights'])
                    result = cr.build_scene(1., cfg)
                    self.assertIs(result[0], replacement if modifier else scene)
                    self.assertEqual(result[1:], rest)
                    if rope == 'snow_decal':
                        fr = SimpleNamespace(zb=cr.np.ones((2, 2)))
                        cr.render_figures(fr)
                        old['render_figures'].assert_called_once_with(fr)
                        with mock.patch.object(CC, 'draw_decal_rope') as decal:
                            cr.draw_rope('rope-arguments')
                            cr.np.testing.assert_array_equal(decal.call_args.args[1]['zt'], fr.zb)
                    return scene

                cr.render.side_effect = render
                result = CC.render_cut(5043, renderer=cr, terrain='round_cap', rope=rope,
                                       rock=rock, rock_modifier=modifier)
                self.assertIs(result, scene)
                for name, value in old.items():
                    self.assertIs(getattr(cr, name), value)
                for name, value in world.items():
                    self.assertIs(getattr(cr.WD, name), value)
                self.assertEqual(cr.CR[3, 16], 0.)

    def test_world_hooks_restore_after_render_and_installation_failures(self):
        for failure in ('render', 'installation', 'row'):
            cr = self.renderer()
            if failure == 'installation':
                del cr.WD.cloud_glow
            world = vars(cr.WD).copy()
            old_rope, old_scene = cr.draw_rope, cr.build_scene
            if failure == 'render':
                cr.render.side_effect = RuntimeError('controlled failure')
            elif failure == 'row':
                cr.render.side_effect = lambda *a, **k: cr.WD.march(None, cr.CR[:3])
            with self.assertRaises((RuntimeError, AttributeError, ValueError)):
                CC.render_cut(5043, renderer=cr, terrain='round_cap', rope='snow_decal')
            for name, value in world.items():
                self.assertIs(getattr(cr.WD, name), value)
            self.assertIs(cr.draw_rope, old_rope)
            self.assertIs(cr.build_scene, old_scene)


class RopeDecalTests(unittest.TestCase):
    """snow_decal (29 Sep): a rope over rippled snow reads as one line; a figure in front still hides it."""

    def draw(self, zb, zt):
        import numpy as np
        img = np.ones((20, 60, 3), np.float32)
        CC._rope_seg_decal(np, img, zb, zt, 5.0, 10.0, 6.0, 55.0, 10.0, 7.0, 1.5, (0.0, 0.0, 0.0), 1.0)
        return img[10, 5:55, 0]

    def rippled(self):
        """Terrain behind the rope (8 m) with a ripple every third column 15 cm in FRONT of the rope's own depth there:
        inside the decal margin (0.06 + 0.025 z, ~0.21-0.24 m here), outside the original 5 cm one."""
        import numpy as np
        zb = np.full((20, 60), 8.0, np.float32)
        x = np.arange(60)
        zr = 6.0 + (np.clip((x + 0.5 - 5.0) / 50.0, 0, 1))
        zb[:, ::3] = (zr - 0.15)[::3]
        return zb, zr

    def test_ripples_in_front_within_the_margin_do_not_cut_the_line(self):
        zb, _ = self.rippled()
        line = self.draw(zb, zb.copy())
        self.assertTrue((line < 0.5).all(), line)

    def test_the_mean_depth_rasterizer_would_have_cut_it(self):
        # the defect this replaces: one depth per sub-segment (its mean, 6.5) and a 5 cm margin
        zb, _ = self.rippled()
        hidden = zb[10, 5:55] < 6.5 - 0.05
        self.assertGreater(int(hidden.sum()), 5)

    def test_a_figure_in_front_still_hides_the_rope(self):
        import numpy as np
        zt = np.full((20, 60), 9.0, np.float32)
        zb = zt.copy()
        zb[:, 20:30] = 5.8                    # a figure 0.2+ m in front of the rope, over terrain further back
        line = self.draw(zb, zt)
        self.assertTrue((line[16:24] > 0.99).all(), line[16:24])
        self.assertTrue((line[:12] < 0.5).all())

    def test_far_terrain_ridge_in_front_hides_it(self):
        import numpy as np
        zb = np.full((20, 60), 9.0, np.float32)
        zb[:, 20:30] = 4.0                    # a snow crest 2+ m in front: beyond any decal margin
        line = self.draw(zb, zb.copy())
        self.assertTrue((line[16:24] > 0.99).all())


if __name__ == '__main__':
    unittest.main()
