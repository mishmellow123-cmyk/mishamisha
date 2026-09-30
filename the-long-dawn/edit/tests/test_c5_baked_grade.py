"""Fixed baked-caption grades on synthetic pixels and exported-schema fixtures."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import caption_grade as CG  # noqa: E402


OUTLINE = [[800, 320], [1120, 320], [1120, 480], [800, 480]]
SHOULDER = dict(kind='shadow_shoulder', low=0.04, high=0.14, gain=1.3)
EXPOSURE = dict(kind='blackpoint_gain', blackpoint=0.0, gain=2.5)


def decode(value):
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def encode(value):
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


class BakedGradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.temp) / 'bands.json'
        self.enterContext(mock.patch.object(CG, 'DATA_PATH', self.path))
        self.data = dict(version=1, dimensions=[1920, 804],
                         band=dict(inner_margin_pixels=24, feather_pixels=240,
                                   boundary_stride_source_texels=8, raster_shift_bits=8),
                         source_hashes={}, bands={})
        for ident, f0, f1, stem, off in [('R02', 400, 540, 'cand_t1_current-words-held', 0),
                                        ('title', 5748, 5880, 'book_C', 1280)]:
            self.data['bands'][ident] = dict(frame_start=f0, frame_end_exclusive=f1, source_stem=stem, source_off=off,
                                            outlines={str(f): OUTLINE for f in range(f0, f1)})
        self.spec = dict(id='R02', f0=400, f1=540, full0=430, full1=528,
                         source_stem='cand_t1_current-words-held', source_off=0, curve=SHOULDER)
        self.write_data()

    def write_data(self):
        raw = json.dumps(self.data, sort_keys=True).encode()
        self.path.write_bytes(raw)
        self.spec['band_sha256'] = hashlib.sha256(raw).hexdigest()

    def apply(self, image, frame=450, **kw):
        return CG.apply(image, frame, dict(self.spec, **kw),
                        dict(stem=self.spec['source_stem'], off=self.spec['source_off'], mode='exact'))

    def test_zero_strength_is_exact_identity_without_geometry_io(self):
        image = np.array([[[0.17, 0.31, 0.47]]], np.float32)
        self.path.unlink()
        for frame in (399, 400, 539, 540):
            with self.subTest(frame=frame):
                self.assertIs(self.apply(image, frame), image)

    def test_fixed_curves_match_independent_linear_formula(self):
        rgb = np.array([[[encode(v)] * 3 for v in (0.02, 0.08, 0.14, 0.25, 0.6)]], np.float32)
        alpha = np.array([[1, 1, 0.25, 0.5, 1]], np.float32)
        for curve in (SHOULDER, EXPOSURE):
            expected = np.empty_like(rgb)
            for i, pixel in enumerate(rgb[0]):
                linear = [decode(float(channel)) for channel in pixel]
                lum = sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722)))
                if curve['kind'] == 'shadow_shoulder':
                    u = min(1, max(0, (lum - 0.04) / 0.10))
                    target_lum = 1.3 * lum * (3 * u * u - 2 * u * u * u)
                else:
                    target_lum = 2.5 * lum
                for channel, value in enumerate(linear):
                    target = min(1, max(0, value * target_lum / lum))
                    expected[0, i, channel] = encode(value + float(alpha[0, i]) * (target - value))
            with self.subTest(curve=curve['kind']):
                np.testing.assert_allclose(CG._grade(rgb, alpha, curve), expected, rtol=0, atol=3e-7)

    def test_chroma_ratios_clipping_and_zero_band_pixels(self):
        rgb = np.array([[[0.12, 0.2, 0.3], [0.9, 0.3, 0.1], [0.13, 0.41, 0.73], [0, 0, 0]]], np.float32)
        alpha = np.array([[1, 1, 0, 1]], np.float32)
        out = CG._grade(rgb, alpha, EXPOSURE)
        gains = [decode(float(out[0, 0, c])) / decode(float(rgb[0, 0, c])) for c in range(3)]
        np.testing.assert_allclose(gains, [2.5] * 3, rtol=0, atol=2e-6)
        self.assertAlmostEqual(float(out[0, 1, 0]), 1, places=6)
        self.assertGreaterEqual(float(out.min()), 0)
        self.assertLessEqual(float(out.max()), 1)
        np.testing.assert_array_equal(out[0, 2:], rgb[0, 2:])

    def test_band_margin_feather_and_local_bounds_scale_with_width(self):
        for scale in (0.5, 1.0):
            with self.subTest(scale=scale):
                image = np.full((round(804 * scale), round(1920 * scale), 3), 0.6, np.float32)
                original = image.copy()
                out = self.apply(image)
                np.testing.assert_array_equal(image, original)
                box, alpha = CG._band(OUTLINE, image.shape)
                x0, y0, x1, y1 = box
                y = round(400 * scale) - y0
                for native_x, expected in ((1144, 1), (1264, 0.5), (1384, 0)):
                    self.assertAlmostEqual(float(alpha[y, round(native_x * scale) - x0]), expected, places=6)
                outside = np.ones(image.shape[:2], bool)
                outside[y0:y1, x0:x1] = alpha == 0
                np.testing.assert_array_equal(out[outside], image[outside])
                self.assertGreater(float(out[round(400 * scale), round(960 * scale), 0]), 0.6)

    def test_envelopes_keep_declared_holds_and_ease_both_edges(self):
        for f0, f1, a, b in ((400, 540, 430, 528), (5748, 5880, 5808, 5847)):
            spec = dict(f0=f0, f1=f1, full0=a, full1=b)
            for frame in (f0 - 1, f0, f1 - 1, f1):
                self.assertEqual(CG.strength(frame, spec), 0)
            self.assertTrue(all(CG.strength(f, spec) == 1 for f in range(a, b + 1)))
            rising = [CG.strength(f, spec) for f in range(f0, a + 1)]
            falling = [CG.strength(f, spec) for f in range(b, f1)]
            self.assertTrue(np.all(np.diff(rising) > 0))
            self.assertTrue(np.all(np.diff(falling) < 0))
            self.assertLess(rising[1], 1 / (a - f0))
            self.assertLess(falling[-2], 1 / (f1 - 1 - b))

    def test_missing_geometry_frame_fails_even_before_that_frame_is_used(self):
        del self.data['bands']['R02']['outlines']['401']
        self.write_data()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.apply(np.full((402, 960, 3), 0.3, np.float32))

    def test_source_mismatch_fails_loudly(self):
        image = np.full((402, 960, 3), 0.3, np.float32)
        for source in (None, dict(stem='wrong-stem', off=0, mode='exact')):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'source stem'):
                CG.apply(image, 450, self.spec, source)

    def test_same_stem_wrong_offset_or_remapped_take_fails_loudly(self):
        image = np.full((402, 960, 3), 0.3, np.float32)
        for change in (dict(off=1), dict(crop=(0, 0, 0.5, 1)), dict(mode='video'), dict(video='another.mp4')):
            source = dict(stem=self.spec['source_stem'], off=0, mode='exact')
            source.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'offset or geometry mapping'):
                CG.apply(image, 450, self.spec, source)

    def test_checksum_rejects_wrong_pin_and_file_edits_after_successful_load(self):
        image = np.full((402, 960, 3), 0.3, np.float32)
        self.apply(image)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.apply(image, band_sha256='0' * 64)
        self.path.write_bytes(self.path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.apply(image)

    def test_invalid_polygon_is_rejected_before_use(self):
        for outline in ([[0, 0]] * 4, [[0, 0], [1, 0], [1, float('nan')], [0, 1]]):
            self.data['bands']['R02']['outlines']['402'] = outline
            self.write_data()
            with self.subTest(outline=outline), self.assertRaisesRegex(ValueError, 'polygon'):
                self.apply(np.full((402, 960, 3), 0.3, np.float32))

    def test_delivery_source_contains_exact_geometry_bytes_and_changes_with_them(self):
        before = CG.source()
        self.assertIn(self.path.read_text(), before)
        self.assertIn(self.spec['band_sha256'], before)
        self.path.write_bytes(self.path.read_bytes() + b' ')
        self.assertNotEqual(CG.source(), before)


if __name__ == '__main__':
    unittest.main()
