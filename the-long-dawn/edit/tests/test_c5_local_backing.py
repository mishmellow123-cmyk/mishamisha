"""C's optional backing follows the real caption envelope and stays local.

Synthetic pictures only; contrast uses the production glyph-core/ring metric.
"""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
sys.path.insert(0, str(EDIT / 'tools'))
import c5_caption_backdrop as BD  # noqa: E402
import titles  # noqa: E402


def caption(**kw):
    return dict(dict(id='BACKING_TEST', row=20, line='They left the gold in the ground.',
                     f_in=10, f_out=130, set='ink', x=960, y=402), **kw)


def picture(scale, ground):
    return np.full((round(titles.H * scale), round(titles.W * scale), 3), ground, np.float32)


def draw(row, scale, ground, frame, cut='C'):
    return titles.TextV3(cut, row, scale).draw(picture(scale, ground), frame)


class LocalBackingTests(unittest.TestCase):
    def test_table_and_parts_pass_backing_into_the_renderer(self):
        parts = (dict(line='They left the gold', f_in=10, f_out=60),
                 dict(line='in the ground.', f_in=60, f_out=130, backing=0.8),
                 dict(line='They left the gold', f_in=130, f_out=170, backing=0))
        row = caption(backing=0.35, parts=parts)
        with mock.patch.object(titles, 'C5_TEXT', [row]):
            self.assertEqual(titles.text_table('C')[0]['backing'], 0.35)
            lines = titles.lines_v3('C', 0.5)
        self.assertEqual([line.backing for line in lines], [0.35, 0.8, 0])
        for line, strength in zip(lines, [0.35, 0.8, 0]):
            with self.subTest(line=line.id):
                f = line.f_in + 30
                backed = line.draw(picture(0.5, 0.3), f)
                bare = titles.TextV3('C', caption(id=line.id, line=line.text,
                                                f_in=line.f_in, f_out=line.f_out), 0.5)
                same = np.array_equal(backed, bare.draw(picture(0.5, 0.3), f))
                self.assertEqual(same, strength == 0)

    def test_invalid_strength_is_rejected_before_rendering(self):
        for value in (-0.01, 1.01, np.nan, np.inf, -np.inf, '0.5', None, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'backing'):
                titles.TextV3('C', caption(backing=value), 0.5)

    def test_omitted_and_zero_backing_have_identical_pixels(self):
        for scale in (0.5, 1.0):
            for kind in ('ink', 'fire'):
                for frame in (9, 10, 16, 40, 119, 129, 130):
                    with self.subTest(scale=scale, kind=kind, frame=frame):
                        base = caption(set=kind)
                        np.testing.assert_array_equal(draw(base, scale, 0.45, frame),
                                                      draw(dict(base, backing=0), scale, 0.45, frame))

    def test_default_ink_preserves_the_existing_composite(self):
        for scale in (0.5, 1.0):
            line = titles.TextV3('C', caption(), scale)
            for frame in (16, 40, 129):
                with self.subTest(scale=scale, frame=frame):
                    expected = picture(scale, 0.3)
                    sub = expected[line.y0:line.y0 + line.h, line.x0:line.x0 + line.w]
                    # Legacy draw() copies the envelope into float32 `big` before compositing; reproduce that
                    # boundary here so this is an exact-pixel comparison even when _ink returns float64.
                    a = np.asarray(line._ink(frame), dtype=np.float32)[..., None]
                    sub[:] = sub * (1 - a) + titles.IRON * a
                    np.testing.assert_array_equal(line.draw(picture(scale, 0.3), frame), expected)

    def test_other_cuts_ignore_c_backing(self):
        for cut in ('A', 'B'):
            with self.subTest(cut=cut):
                row = caption(set='lower')
                np.testing.assert_array_equal(draw(row, 0.5, 0.45, 40, cut),
                                              draw(dict(row, backing=1), 0.5, 0.45, 40, cut))

    def test_backing_stays_inside_the_existing_padded_region(self):
        for scale in (0.5, 1.0):
            for kind, ground in (('ink', 0.3), ('fire', 1.0)):
                with self.subTest(scale=scale, kind=kind):
                    row = caption(set=kind)
                    line = titles.TextV3('C', row, scale)
                    plain = draw(row, scale, ground, 40)
                    backed = draw(dict(row, backing=1), scale, ground, 40)
                    pad = int(40 * scale) + 4
                    outside = np.ones(plain.shape[:2], bool)
                    outside[max(0, line.y0 - pad):line.y0 + line.h + pad,
                            max(0, line.x0 - pad):line.x0 + line.w + pad] = False
                    self.assertFalse(np.array_equal(backed, plain))
                    np.testing.assert_array_equal(backed[outside], plain[outside])

    def test_backing_clears_a_real_low_contrast_glyph_ring_case(self):
        for kind, ground in (('ink', 0.3), ('fire', 1.0)):
            with self.subTest(kind=kind):
                row = caption(set=kind)
                line = titles.TextV3('C', row, 1.0)
                core, ring = BD.glyph_masks(line)
                slices = BD.band_slices(line)
                plain = BD.contrasts(draw(row, 1.0, ground, 40), core, ring, slices)[1]
                backed = BD.contrasts(draw(dict(row, backing=1), 1.0, ground, 40), core, ring, slices)[1]
                self.assertLess(plain, 4.5, f'{kind} control is not low contrast: {plain}')
                self.assertGreater(backed, 4.5, f'{kind} backing did not clear the floor: {backed}')

    def test_slate_backing_preserves_contrast_for_parchment_white_ink(self):
        row = caption()
        line = titles.TextV3('C', row, 1.0)
        core, ring = BD.glyph_masks(line)
        slices = BD.band_slices(line)
        plain = line.draw(picture(1.0, 0.12), 40, dark_ground=True)
        backed = titles.TextV3('C', dict(row, backing=1), 1.0).draw(
            picture(1.0, 0.12), 40, dark_ground=True)
        self.assertTrue(np.all(backed[ring] <= plain[ring]))
        self.assertLess(float(backed[ring].mean()), float(plain[ring].mean()))
        self.assertGreaterEqual(BD.contrasts(backed, core, ring, slices)[1],
                                BD.contrasts(plain, core, ring, slices)[1])

    def test_paper_backing_preserves_rgb_ratios_and_caps_clipped_channels(self):
        line = titles.TextV3('C', caption(backing=1), 1.0)
        _, ring = BD.glyph_masks(line)
        alpha = np.zeros(ring.shape, np.float32)
        alpha[line.y0:line.y0 + line.h, line.x0:line.x0 + line.w] = line.alpha
        bare_paper = ring & (alpha == 0)
        self.assertTrue(bare_paper.any())
        source = picture(1.0, np.array([0.16, 0.24, 0.32], np.float32))
        backed = line.draw(source.copy(), 40)
        gain = backed[bare_paper] / source[bare_paper]
        self.assertTrue(np.all(gain > 1))
        np.testing.assert_allclose(gain[:, 0], gain[:, 1], rtol=1e-6, atol=0)
        np.testing.assert_allclose(gain[:, 1], gain[:, 2], rtol=1e-6, atol=0)
        bright = line.draw(picture(1.0, np.array([0.85, 0.7, 0.6], np.float32)), 40)
        self.assertTrue(np.any(bright[bare_paper] == 1))
        self.assertGreaterEqual(float(bright.min()), 0)
        self.assertLessEqual(float(bright.max()), 1)

    def test_paper_apron_edges_are_zero_without_changing_glyph_or_ring(self):
        for scale in (0.5, 1.0):
            with self.subTest(scale=scale):
                row = caption(backing=1)
                line = titles.TextV3('C', row, scale)
                source = picture(scale, 0.3)
                plain = draw(dict(row, backing=0), scale, 0.3, 40)
                backed = line.draw(source.copy(), 40)
                pad = int(40 * scale) + 4
                x0, y0 = line.x0 - pad, line.y0 - pad
                x1, y1 = line.x0 + line.w + pad, line.y0 + line.h + pad
                reference = source.copy()
                region = reference[y0:y1, x0:x1]
                alpha = np.zeros(region.shape[:2], np.float32)
                alpha[pad:pad + line.h, pad:pad + line.w] = line._ink(40)
                # The previous Gaussian wash, without any edge taper, is the interior-pixel reference.
                radius = round(24 * scale)
                spread = titles.cv2.dilate(alpha, titles.cv2.getStructuringElement(
                    titles.cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
                wash = titles.cv2.GaussianBlur(spread, (0, 0), 16 * scale)
                region[:] = np.minimum(region * (1 + wash[..., None]), 1.0)
                region[:] = region * (1 - alpha[..., None]) + titles.IRON * alpha[..., None]
                boundary = np.zeros(source.shape[:2], bool)
                boundary[y0, x0:x1] = boundary[y1 - 1, x0:x1] = True
                boundary[y0:y1, x0] = boundary[y0:y1, x1 - 1] = True
                self.assertTrue(np.any(reference[boundary] != plain[boundary]))
                np.testing.assert_array_equal(backed[boundary], plain[boundary])
                with mock.patch.object(BD, 'H', source.shape[0]), mock.patch.object(BD, 'W', source.shape[1]):
                    core, ring = BD.glyph_masks(line)
                glyphs = np.zeros_like(core)
                glyphs[line.y0:line.y0 + line.h, line.x0:line.x0 + line.w] = line.alpha > 0
                self.assertTrue(core.any() and ring.any())
                np.testing.assert_array_equal(backed[glyphs | ring], reference[glyphs | ring])

    def test_backing_arrives_with_the_letters_and_leaves_with_them(self):
        for kind, ground, early in (('ink', 0.3, 16), ('fire', 1.0, 11)):
            with self.subTest(kind=kind):
                row = caption(set=kind)
                line = titles.TextV3('C', row, 1.0)
                for frame in (9, 10, 130, 131):
                    np.testing.assert_array_equal(draw(dict(row, backing=1), 1.0, ground, frame),
                                                  draw(row, 1.0, ground, frame))
                early_delta = draw(dict(row, backing=1), 1.0, ground, early) - draw(row, 1.0, ground, early)
                steady_delta = draw(dict(row, backing=1), 1.0, ground, 40) - draw(row, 1.0, ground, 40)
                late_delta = draw(dict(row, backing=1), 1.0, ground, 129) - draw(row, 1.0, ground, 129)
                self.assertGreater(float(np.abs(early_delta).sum()), 0)
                self.assertLess(float(np.abs(early_delta).sum()), float(np.abs(steady_delta).sum()))
                self.assertLess(float(np.abs(late_delta).sum()), float(np.abs(steady_delta).sum()))
                # The unwritten tail must not acquire the backing of the completed caption.
                tail = line.x0 + round(0.8 * line.w)
                np.testing.assert_array_equal(early_delta[:, tail:], 0)


if __name__ == '__main__':
    unittest.main()
