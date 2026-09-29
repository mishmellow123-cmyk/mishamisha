"""Light contract tests; no ray march, Numba compile, or frame audit here."""
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reveal_pair_candidates as candidates


class RevealCandidates(unittest.TestCase):
    def setUp(self):
        self.shot = types.SimpleNamespace(B=np.zeros((2, 7)), CR=object())

    def fake_inkpass(self, fail=False):
        ip = types.SimpleNamespace()
        ip.GOLD = np.array([214., 164., 62.])
        ip.s2l = lambda colour: np.asarray(colour) / 255.
        self.glyph_calls = []
        self.glyph_output = (np.full((2, 2), .8), np.full((2, 2, 3), .5), np.full((2, 2), .2))

        def glyph(*args):
            self.glyph_calls.append(args)
            return self.glyph_output

        ip.flame_glyph = glyph
        ip.flame_scale = lambda z: 7. if z > 2500 else 1.

        def flames(seed=10):
            scales = (ip.flame_scale(300.), ip.flame_scale(3000.))
            result = ip.flame_glyph(np.zeros((2, 2)), np.zeros((2, 2)), 20., 0., seed, 3, 1., .75, np.ones((2, 2)), 1.)
            if fail:
                raise RuntimeError('deliberate ink failure')
            return scales, result

        ip.ink_flames = flames
        return ip

    def test_default_delegates_without_loading_or_patching_inkpass(self):
        marker = object()
        with mock.patch.object(candidates.pair, 'render', return_value=marker) as render:
            with mock.patch.dict(sys.modules, {'inkpass': None}):
                self.assertIs(candidates.render(2880, self.shot), marker)
        render.assert_called_once_with(2880, self.shot, scale=.5, ss=2.)

    def test_shared_accepted_compose_and_resize_match_canonical_driver(self):
        """Real driver/resize code, synthetic AOV and compose output; no march."""
        import cv2

        yy, xx = np.mgrid[:23, :41]
        field = ((7*yy + 13*xx) % 17).astype(np.float32)/16.
        composed = np.stack((field, field[::-1], field[:, ::-1]), axis=-1)
        self.assertFalse(np.array_equal(cv2.resize(composed, (19, 8), interpolation=cv2.INTER_AREA),
                                        cv2.resize(composed, (19, 8), interpolation=cv2.INTER_LINEAR)))
        ri = types.SimpleNamespace(render_aov=mock.Mock(side_effect=lambda *a: {}))
        ip = types.SimpleNamespace(compose=mock.Mock(side_effect=lambda *a, **k: (composed.copy(), {})))
        with mock.patch.dict(sys.modules, {'inkpass': ip, 'render_ink': ri}):
            expected = candidates.pair.render(2881, self.shot, scale=.01, ss=1.75)
            actual = candidates.render_variants(2881, self.shot, variants=('accepted',),
                                                 scale=.01, ss=1.75)['accepted']
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(ri.render_aov.call_args_list, [mock.call(self.shot, 1, .01, 1.75)]*2)
        self.assertEqual(ip.compose.call_args_list,
                         [mock.call({'kpx': .0175}, B=self.shot.B, CR=self.shot.CR)]*2)

    def test_near_glyph_is_exactly_the_original_result_and_call(self):
        for variant in ('far-ink', 'far-scale'):
            with self.subTest(variant=variant):
                ip = self.fake_inkpass()
                originals = (ip.flame_glyph, ip.flame_scale, ip.ink_flames)
                with mock.patch.dict(sys.modules, {'inkpass': ip}):
                    with candidates._candidate_style(variant):
                        scales, result = ip.ink_flames(seed=3)
                        self.assertIs(result, self.glyph_output)
                        self.assertEqual(self.glyph_calls[0][7], .75)
                        self.assertEqual(scales[0], 1.)
                    self.assertEqual((ip.flame_glyph, ip.flame_scale, ip.ink_flames), originals)

    def test_far_ink_keeps_height_and_uses_existing_palette(self):
        ip = self.fake_inkpass()
        with mock.patch.dict(sys.modules, {'inkpass': ip}):
            with candidates._candidate_style('far-ink'):
                scales, (alpha, colour, _) = ip.ink_flames()
        self.assertEqual(scales, (1., 7.))
        self.assertAlmostEqual(self.glyph_calls[0][7], .75 * 1.35)
        np.testing.assert_allclose(alpha, .88)
        np.testing.assert_allclose(colour, np.broadcast_to(.5 * .68 + ip.s2l(ip.GOLD) * .32, colour.shape))

    def test_far_scale_only_changes_distant_drawing_not_terrain_light_scale(self):
        ip = self.fake_inkpass()
        original = ip.flame_scale
        with mock.patch.dict(sys.modules, {'inkpass': ip}):
            with candidates._candidate_style('far-scale'):
                self.assertIs(ip.flame_scale, original)
                scales, _ = ip.ink_flames()
                self.assertEqual(scales[0], 1.)
                self.assertAlmostEqual(scales[1], 9.8)
                self.assertIs(ip.flame_scale, original)
        self.assertAlmostEqual(self.glyph_calls[0][7], .75 * 1.12)

    def test_exception_restores_all_globals(self):
        ip = self.fake_inkpass(fail=True)
        originals = (ip.flame_glyph, ip.flame_scale, ip.ink_flames)
        with mock.patch.dict(sys.modules, {'inkpass': ip}):
            with self.assertRaisesRegex(RuntimeError, 'deliberate ink failure'):
                with candidates._candidate_style('far-scale'):
                    ip.ink_flames()
        self.assertEqual((ip.flame_glyph, ip.flame_scale, ip.ink_flames), originals)

    def test_bad_variant_or_scale_rejected_before_render(self):
        with mock.patch.object(candidates.pair, 'render') as render:
            for values in ({'variant': 'unknown'}, {'scale': float('nan')}, {'ss': float('inf')},
                           {'scale': 0}, {'ss': -1}, {'scale': 1e-10}):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    candidates.render(2880, self.shot, **values)
            render.assert_not_called()
        for variants in ((), ('accepted', 'accepted'), ('missing',)):
            with self.assertRaises(ValueError):
                candidates.render_variants(2880, self.shot, variants=variants)

    def test_overlap_rejected_and_lock_released_after_failure(self):
        with candidates._render_guard():
            with self.assertRaisesRegex(RuntimeError, 'sequentially'):
                candidates.render(2880, self.shot)
        with mock.patch.object(candidates.pair, 'render', side_effect=RuntimeError('render failure')):
            with self.assertRaisesRegex(RuntimeError, 'render failure'):
                candidates.render(2880, self.shot)
        with candidates._render_guard():
            pass

    def test_variants_reuse_one_aov_and_preserve_shot_sites(self):
        ip = self.fake_inkpass()
        original_sites = self.shot.B.copy()
        aov = {'A': np.zeros((2, 2, 16))}
        ri = types.SimpleNamespace(render_aov=mock.Mock(return_value=aov))
        seen = []

        def compose(actual_aov, shot, scale):
            self.assertIs(actual_aov, aov)
            self.assertIs(shot, self.shot)
            self.assertEqual(actual_aov['kpx'], 1.)
            seen.append(ip.ink_flames())
            return len(seen)

        with mock.patch.dict(sys.modules, {'inkpass': ip, 'render_ink': ri}):
            with mock.patch.object(candidates, '_compose', side_effect=compose):
                result = candidates.render_variants(2881, self.shot, variants=('accepted', 'far-ink', 'far-scale'))
        self.assertEqual(result, {'accepted': 1, 'far-ink': 2, 'far-scale': 3})
        ri.render_aov.assert_called_once_with(self.shot, 1, .5, 2.)
        self.assertEqual(seen[0][0], (1., 7.))
        self.assertAlmostEqual(seen[2][0][1], 9.8)
        np.testing.assert_array_equal(self.shot.B, original_sites)


class HaloWash(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((120, 320, 3), .8, dtype=np.float32)
        self.ink = np.full((120, 320), .6, dtype=np.float32)
        self.protected = np.zeros((120, 320), bool)
        self.aov = {
            'A': np.zeros((120, 320, 16)), 'frame': 12, 'kpx': 1.,
            'cam': dict(pos=np.zeros(3), fwd=np.array([0., 0., 1.]),
                        right=np.array([1., 0., 0.]), up=np.array([0., 1., 0.]),
                        f=100., cx=160., cy=60.),
        }
        self.aov['A'][..., 1] = 1000.
        self.beacons = np.array([[-90., 0., 100., 0., 1., 0., 0.], [90., 0., 100., 0., 1., 1., 0.]])
        self.ip = types.SimpleNamespace(
            GOLD_HI=np.array([252., 226., 150.]), INK=np.array([40., 27., 20.]),
            s2l=lambda c: np.asarray(c) / 255., flame_scale=lambda z: 1.,
            look=types.SimpleNamespace(linear_to_srgb=lambda c: c))
        self.bc = types.SimpleNamespace(env=lambda frame, ignition: (1., 1., 1.))

    def apply(self):
        with mock.patch.dict(sys.modules, {'inkpass': self.ip, 'beacons': self.bc}):
            return candidates._halo_wash(self.rgb, self.aov, self.beacons, self.ink, self.protected)

    def test_both_beacons_receive_the_same_local_treatment(self):
        actual = self.apply()
        np.testing.assert_array_equal(actual[:, 22:118], actual[:, 202:298])
        self.assertFalse(np.array_equal(actual, self.rgb))
        np.testing.assert_array_equal(self.rgb, np.full_like(self.rgb, .8))

    def test_pixels_outside_compact_discs_and_flames_are_exactly_preserved(self):
        self.protected[51:61, 66:76] = True
        actual = self.apply()
        yy, xx = np.mgrid[:120, :320]
        cy = 60. - 1.6 - .42 * 6.5
        support = (((xx + .5 - 70.) ** 2 + (yy + .5 - cy) ** 2 < 48. ** 2)
                   | ((xx + .5 - 250.) ** 2 + (yy + .5 - cy) ** 2 < 48. ** 2))
        np.testing.assert_array_equal(actual[~support], self.rgb[~support])
        np.testing.assert_array_equal(actual[self.protected], self.rgb[self.protected])

    def test_unlit_or_occluded_beacons_do_not_change_the_page(self):
        self.bc.env = lambda frame, ignition: (0., 0., 0.)
        np.testing.assert_array_equal(self.apply(), self.rgb)
        self.bc.env = lambda frame, ignition: (1., 1., 1.)
        self.aov['A'][..., 1] = 2.
        np.testing.assert_array_equal(self.apply(), self.rgb)

    def test_paper_warms_and_existing_ink_deepens(self):
        self.ink[:] = 0.
        paper = self.apply()
        self.assertGreater(paper[56, 70, 0], self.rgb[56, 70, 0])
        self.assertLess(paper[56, 70, 2], self.rgb[56, 70, 2])
        self.ink[:] = 1.
        reinforced = self.apply()
        self.assertTrue(np.all(reinforced[56, 70] < self.rgb[56, 70]))

    def test_halo_context_protects_fire_layers_and_restores_after_failure(self):
        fire = np.zeros((120, 320), np.float32)
        fire[55, 70] = .8
        self.ip.ink_flames = lambda *a, **k: (fire, np.zeros_like(self.rgb), fire, fire)

        def compose(aov, **kwargs):
            self.ip.ink_flames()
            return self.rgb, {'ink': self.ink}

        self.ip.compose = compose
        original_compose, original_flames = self.ip.compose, self.ip.ink_flames
        with mock.patch.dict(sys.modules, {'inkpass': self.ip, 'beacons': self.bc}):
            with candidates._candidate_style('halo-wash'):
                result, _ = self.ip.compose(self.aov, B=self.beacons)
                np.testing.assert_array_equal(result[55, 70], self.rgb[55, 70])
            self.assertIs(self.ip.compose, original_compose)
            self.assertIs(self.ip.ink_flames, original_flames)
            with mock.patch.object(candidates, '_halo_wash', side_effect=RuntimeError('halo failure')):
                with self.assertRaisesRegex(RuntimeError, 'halo failure'):
                    with candidates._candidate_style('halo-wash'):
                        self.ip.compose(self.aov, B=self.beacons)
            self.assertIs(self.ip.compose, original_compose)
            self.assertIs(self.ip.ink_flames, original_flames)


if __name__ == '__main__':
    unittest.main()
