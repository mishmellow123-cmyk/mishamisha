"""Light tests of default-off, semantic separation and the new glyph geometry."""
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import beacon_night_candidates as night


class NightCandidates(unittest.TestCase):
    def test_accepted_delegates_to_each_original_driver(self):
        for kind, frame, driver in (('reveal', 2880, night.reveal), ('watch', 4480, night.watch)):
            with self.subTest(kind=kind), mock.patch.object(driver, 'render', return_value='original') as render:
                with mock.patch.dict(sys.modules, {'inkpass': None}):
                    shot = object()
                    self.assertEqual(night.render(frame, shot, kind=kind), 'original')
                render.assert_called_once_with(frame, shot, scale=.5, ss=2.)

    def test_shared_accepted_compose_and_resize_match_each_canonical_driver(self):
        """Exercise real driver/resize code with synthetic AOV and compose output."""
        import cv2

        yy, xx = np.mgrid[:23, :41]
        field = ((7*yy + 13*xx) % 17).astype(np.float32)/16.
        composed = np.stack((field, field[::-1], field[:, ::-1]), axis=-1)
        # Noninteger shrink ratios expose AREA -> LINEAR drift (2x can hide it).
        self.assertFalse(np.array_equal(cv2.resize(composed, (19, 8), interpolation=cv2.INTER_AREA),
                                        cv2.resize(composed, (19, 8), interpolation=cv2.INTER_LINEAR)))
        shot = types.SimpleNamespace(B=object(), CR=object())
        for kind, frame, driver, source in (('reveal', 2881, night.reveal, 1),
                                            ('watch', 4481, night.watch, 81)):
            with self.subTest(kind=kind):
                ri = types.SimpleNamespace(render_aov=mock.Mock(side_effect=lambda *a: {}))
                ip = types.SimpleNamespace(compose=mock.Mock(side_effect=lambda *a, **k: (composed.copy(), {})))
                with mock.patch.dict(sys.modules, {'inkpass': ip, 'render_ink': ri}):
                    expected = driver.render(frame, shot, scale=.01, ss=1.75)
                    actual = night.render_variants(frame, shot, kind=kind, variants=('accepted',),
                                                   scale=.01, ss=1.75)['accepted']
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(ri.render_aov.call_args_list,
                                 [mock.call(shot, source, .01, 1.75)]*2)
                self.assertEqual(ip.compose.call_args_list,
                                 [mock.call({'kpx': .0175}, B=shot.B, CR=shot.CR)]*2)

    def test_source_clocks_and_bad_inputs(self):
        self.assertEqual(night._source(2880, 'reveal'), 0)
        self.assertEqual(night._source(3119, 'reveal'), 239)
        self.assertEqual(night._source(4480, 'watch'), 80)
        self.assertEqual(night._source(4719, 'watch'), 319)
        for args in ((2879, 'reveal', 'accepted', .5, 2.), (4720, 'watch', 'accepted', .5, 2.),
                     (2880, 'bad', 'accepted', .5, 2.), (2880, 'reveal', 'bad', .5, 2.),
                     (2880, 'reveal', 'accepted', float('nan'), 2.),
                     (2880, 'reveal', 'accepted', .5, 0.)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                night._validate(*args)

    def test_geometry_is_tall_asymmetric_and_moves_without_noise_reseeding(self):
        yy, xx = np.mgrid[-10:140, -70:71]
        alpha, core = night._fire_fields(xx, yy, 80., 1., 10, 1.)
        rows, cols = np.where(alpha > .5)
        self.assertGreater(rows.max() - rows.min(), 2 * (cols.max() - cols.min()))
        self.assertFalse(np.array_equal(alpha, alpha[:, ::-1]))
        self.assertFalse(np.array_equal(alpha, night._fire_fields(xx, yy, 80., 1.2, 10, 1.)[0]))
        np.testing.assert_array_equal(alpha, night._fire_fields(xx, yy, 80., 1., 10, 1.)[0])
        self.assertTrue(np.all(core <= alpha))
        self.assertTrue(np.isfinite(alpha).all())
        self.assertGreater(core.max(), .99)
        # A round-ended root occupies multiple rows rather than one baseline.
        lows = [np.where(alpha[:, x] > .5)[0].min() for x in range(alpha.shape[1]) if np.any(alpha[:, x] > .5)]
        self.assertGreater(len(set(lows)), 4)

    def test_wisp_is_optional_restrained_and_above_the_flame(self):
        yy, xx = np.mgrid[-10:230, -70:71]
        empty = night._wisp_field(xx, yy, 80., 1., 10, 1., .5)
        self.assertEqual(float(empty.max()), 0.)
        actual = night._wisp_field(xx, yy, 80., 1., 10, 1., 1.)
        self.assertGreater(float(actual.max()), 0.)
        self.assertLessEqual(float(actual.max()), .24)
        self.assertFalse(np.any(actual[yy < 80.] > 0))

    def test_night_page_dims_warm_ground_without_a_colour_key(self):
        paper = np.full((20, 20, 3), .8, np.float32)
        paper[10, 10] = [1., .9, .2]
        result = night._night_page(paper)
        self.assertLess(float(result[10, 10] @ night.LUMA), float(paper[10, 10] @ night.LUMA))
        self.assertLess(result[0].mean(), result[-1].mean())
        self.assertLess(float(result.max()), .8)

    def test_same_core_radiance_for_different_sized_fires(self):
        page = np.full((64, 128, 3), .8, np.float32)
        core = np.zeros((64, 128), np.float32)
        core[28:37, 25:34] = 1.
        core[31:34, 94:97] = 1.
        colour = np.broadcast_to(night._linear(night.ORANGE), page.shape).copy()
        result, glow = night._finish(page, core, colour, core, np.zeros_like(core), 1.)
        np.testing.assert_array_equal(result[32, 29], result[32, 95])
        np.testing.assert_allclose(result[32, 29], night.CORE, atol=1e-6)
        self.assertGreater(float(result[32, 95] @ night.LUMA), float(result[0, 0] @ night.LUMA))
        self.assertGreater(float(glow[29, 95]), 0.)

    def fake_ip(self, fail=False, visible=None):
        ip = types.SimpleNamespace()
        zero = np.zeros((8, 8), np.float32)
        ip.flame_glyph = lambda *args: (zero, np.zeros((8, 8, 3), np.float32), zero)
        ip.fire_soft = lambda *a, **k: np.ones((8, 8), np.float32)
        called = []

        def flames(*args, **kwargs):
            yy, xx = np.mgrid[:8, :8]
            called.append(ip.flame_glyph)
            alpha, colour, ink = ip.flame_glyph(xx - 4, 7 - yy, 5., 1., 10, 3, 1., .75, np.ones((8, 8)), 1.)
            if visible is not None:
                alpha = np.where(visible, alpha, 0.)
            return alpha, colour, ink, zero

        def compose(aov, **kwargs):
            fa, _, _, _ = ip.ink_flames()
            self.assertFalse(fa.any())  # Candidate fire never enters the graded page.
            if fail:
                raise RuntimeError('composition failed')
            return np.full((8, 8, 3), .8, np.float32), {}

        ip.ink_flames, ip.compose = flames, compose
        return ip, called

    def test_injected_ink_pass_routes_occluded_masks_and_restores_globals(self):
        """Sentinel layers prove routing; this fake does not test real occlusion."""
        visible = np.ones((8, 8), bool)
        visible[:, :2] = False
        alpha = dict(outer=np.full((8, 8), .75, np.float32),
                     core=np.full((8, 8), .4, np.float32),
                     smoke=np.full((8, 8), .2, np.float32))
        colour = np.full((8, 8, 3), .31, np.float32)
        page = np.full((8, 8, 3), .8, np.float32)
        final = np.full((8, 8, 3), .12, np.float32)
        glow = np.full((8, 8), .17, np.float32)

        def glyph(mode):
            return lambda *args: (alpha[mode].copy(), colour.copy(), np.zeros((8, 8), np.float32))

        for variant, passes in (('night-fire', 2), ('night-wisp', 3)):
            ip, called = self.fake_ip(visible=visible)
            saved = dict(vars(ip))
            captured = {}
            with (mock.patch.dict(sys.modules, {'inkpass': ip}),
                  mock.patch.object(night, '_glyph', side_effect=glyph) as glyph_factory,
                  mock.patch.object(night, '_finish', return_value=(final, glow)) as finish):
                with night._style(variant, captured):
                    self.assertAlmostEqual(float(ip.fire_soft().max()), .35, places=6)
                    result, _ = ip.compose({'kpx': 1.})
            self.assertIs(result, final)
            self.assertEqual(len(called), passes)
            modes = ['outer', 'core'] + (['smoke'] if variant == 'night-wisp' else [])
            self.assertEqual(glyph_factory.call_args_list, [mock.call(mode) for mode in modes])
            self.assertEqual(set(captured), {'flame_alpha', 'core_alpha', 'smoke_alpha', 'glow'})
            expected = [np.where(visible, alpha['outer'], 0.), np.where(visible, alpha['core'], 0.),
                        np.where(visible, alpha['smoke'], 0.) if passes == 3 else np.zeros((8, 8), np.float32)]
            for key, value in zip(('flame_alpha', 'core_alpha', 'smoke_alpha'), expected):
                np.testing.assert_array_equal(captured[key], value)
            np.testing.assert_array_equal(captured['glow'], glow)
            finish.assert_called_once()
            for actual, value in zip(finish.call_args.args[:-1], (page, expected[0], colour, expected[1], expected[2])):
                np.testing.assert_array_equal(actual, value)
            self.assertEqual(finish.call_args.args[-1], 1.)
            for name, original in saved.items():
                self.assertIs(getattr(ip, name), original)

    def test_failure_restores_functions_and_releases_guard(self):
        ip, _ = self.fake_ip(fail=True)
        saved = dict(vars(ip))
        with mock.patch.dict(sys.modules, {'inkpass': ip}):
            with self.assertRaisesRegex(RuntimeError, 'composition failed'):
                with night._guard(), night._style('night-fire', {}):
                    ip.compose({'kpx': 1.})
        for name, original in saved.items():
            self.assertIs(getattr(ip, name), original)
        with night._guard():
            with self.assertRaisesRegex(RuntimeError, 'sequentially'):
                night.render(2880, object())

    def test_variants_share_one_aov_and_return_geometric_review_masks(self):
        ip, _ = self.fake_ip()
        ri = types.SimpleNamespace(render_aov=mock.Mock(return_value={'kpx': .02}))
        shot = types.SimpleNamespace(B=np.zeros((2, 7)), CR=object())
        with mock.patch.dict(sys.modules, {'inkpass': ip, 'render_ink': ri}):
            images, masks = night.render_variants(4480, shot, kind='watch', scale=.01, return_layers=True)
        ri.render_aov.assert_called_once_with(shot, 80, .01, 2.)
        self.assertEqual(set(images), set(night.VARIANTS))
        self.assertEqual(masks['accepted'], {})
        for variant in ('night-fire', 'night-wisp'):
            self.assertEqual(images[variant].shape, (8, 19, 3))
            self.assertEqual(masks[variant]['core_alpha'].shape, (8, 19))
            self.assertTrue(np.isfinite(images[variant]).all())


if __name__ == '__main__':
    unittest.main()
