import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sepia_night as S


class SepiaNight(unittest.TestCase):
    def test_default_exact_and_independent(self):
        image = np.random.default_rng(51).random((10, 12, 3), dtype=np.float32)
        saved = image.copy()
        result = S.grade(image)
        np.testing.assert_array_equal(result, image)
        self.assertFalse(np.shares_memory(result, image))
        S.grade(image, 'sepia-night')
        np.testing.assert_array_equal(image, saved)

    def test_bright_core_and_orange_tongue_are_unchanged(self):
        image = np.array([[[1., .965, .875], [1., .47, .085], [.9, .9, .9]]], np.float32)
        np.testing.assert_array_equal(S.grade(image, 'sepia-night'), image)

    def test_dark_neutral_page_is_warm_with_order_preserved(self):
        image = np.repeat(np.linspace(0, .5, 32, dtype=np.float32)[:, None, None], 3, axis=2)
        result = S.grade(image, 'sepia-night')
        self.assertTrue(np.all(result[..., 0] >= result[..., 1]))
        self.assertTrue(np.all(result[..., 1] >= result[..., 2]))
        self.assertTrue(np.all(np.diff(result, axis=0) > 0))
        self.assertEqual(result.dtype, image.dtype)

    def test_invalid_input_is_rejected(self):
        for image in (np.zeros((2, 2), np.float32), np.zeros((2, 2, 3), np.uint8),
                      np.full((1, 1, 3), np.nan), np.full((1, 1, 3), 1.01)):
            with self.assertRaises(ValueError): S.grade(image, 'sepia-night')
        with self.assertRaises(ValueError): S.grade(np.zeros((1, 1, 3)), 'unknown')


if __name__ == '__main__': unittest.main()
