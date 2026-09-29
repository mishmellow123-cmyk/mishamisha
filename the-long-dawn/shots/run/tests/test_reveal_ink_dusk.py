"""Small-array contracts for the image-only, default-off dusk study."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reveal_ink_dusk as D


class InkDusk(unittest.TestCase):
    def test_default_is_bit_exact_independent_copy(self):
        for dtype in (np.float32, np.float64):
            source = np.random.default_rng(23).random((7, 9, 3)).astype(dtype)
            baseline = source.copy()
            actual = D.grade(source)
            self.assertEqual(actual.dtype, source.dtype)
            self.assertEqual(actual.tobytes(), source.tobytes())
            self.assertFalse(np.shares_memory(actual, source))
            np.testing.assert_array_equal(source, baseline)

    def test_fixed_key_separates_measured_paper_and_luminous_gold(self):
        source = np.array([[[.86, .765, .588], [.90, .76, .44],
                            [.31, .16, .00], [.80, .80, .80]]])
        key = D.warmth_key(source)
        self.assertEqual(key[0, 0], 0.)
        self.assertEqual(key[0, 1], 1.)
        self.assertEqual(key[0, 2], 0.)
        self.assertEqual(key[0, 3], 0.)
        actual = D.grade(source, 'ink-dusk')
        np.testing.assert_array_equal(actual[0, 1], source[0, 1])

    def test_paper_top_reduces_45_to_65_percent_with_lighter_foreground(self):
        paper = np.broadcast_to([.86, .765, .588], (5, 4, 3)).copy()
        actual = D.grade(paper, 'ink-dusk')
        ratio = (actual @ D.LUMA)/(paper @ D.LUMA)
        self.assertTrue(np.all((ratio[0] >= .35) & (ratio[0] <= .55)))
        self.assertTrue(np.all(np.diff(ratio[:, 0]) > 0))
        self.assertLess(ratio[-1, 0], .8)
        self.assertLess(np.ptp(actual[0, 0]), np.ptp(paper[0, 0]))

    def test_linework_remains_darker_without_blur(self):
        source = np.broadcast_to([.86, .765, .588], (5, 5, 3)).copy()
        source[:, 2] = [.57, .47, .33]
        actual = D.grade(source, 'ink-dusk')
        self.assertTrue(np.all((actual @ D.LUMA)[:, 2] < (actual @ D.LUMA)[:, 1]))
        np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
        np.testing.assert_array_equal(actual[:, 3], actual[:, 4])

    def test_grade_is_finite_bounded_deterministic_and_preserves_source(self):
        source = np.random.default_rng(11).random((11, 13, 3)).astype(np.float32)
        before = source.copy()
        actual = D.grade(source, 'ink-dusk')
        self.assertEqual(actual.shape, source.shape)
        self.assertEqual(actual.dtype, source.dtype)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.all((actual >= 0.) & (actual <= 1.)))
        np.testing.assert_array_equal(actual, D.grade(source, 'ink-dusk'))
        np.testing.assert_array_equal(source, before)

    def test_no_exposure_pumping_when_unrelated_pixels_change(self):
        source = np.broadcast_to([.86, .765, .588], (7, 8, 3)).copy()
        changed = source.copy()
        changed[1, 1] = [.90, .76, .44]
        a, b = D.grade(source, 'ink-dusk'), D.grade(changed, 'ink-dusk')
        mask = np.ones((7, 8), bool)
        mask[1, 1] = False
        np.testing.assert_array_equal(a[mask], b[mask])

    def test_invalid_inputs_fail_instead_of_silently_normalizing(self):
        for bad in (np.zeros((2, 2), float), np.zeros((0, 2, 3), float),
                    np.zeros((2, 2, 4), float), np.zeros((2, 2, 3), np.uint8),
                    np.full((2, 2, 3), np.nan), np.full((2, 2, 3), 1.01),
                    np.full((2, 2, 3), -.01)):
            with self.assertRaises(ValueError):
                D.grade(bad, 'ink-dusk')
        with self.assertRaises(ValueError):
            D.grade(np.zeros((2, 2, 3)), 'typo')


if __name__ == '__main__':
    unittest.main()
