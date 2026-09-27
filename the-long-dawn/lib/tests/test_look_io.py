"""PNG write contract: real encode/decode plus injected encoder/filesystem failures.

    python -m unittest discover -s lib/tests -p 'test_look_io.py' -v

LOOK_UNDER_TEST may point at an earlier look.py for a negative regression check.
"""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np


SOURCE = Path(os.environ.get('LOOK_UNDER_TEST') or Path(__file__).resolve().parents[1] / 'look.py')
SPEC = importlib.util.spec_from_file_location('look_under_test', SOURCE)
look = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(look)


class SavePngTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='look-io-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'frame.png'
        self.image = np.linspace(0.0, 1.0, 8 * 12 * 3, dtype=np.float32).reshape(8, 12, 3)

    def assert_directory(self, expected):
        self.assertEqual({p.name for p in self.root.iterdir()}, set(expected))

    def test_real_png_roundtrip_preserves_shape_rgb_and_dither_range(self):
        self.path.write_bytes(b'old destination')
        self.assertIsNone(look.save_png(self.path, self.image))
        decoded = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, self.image.shape)
        self.assertEqual(decoded.dtype, np.uint8)
        expected = np.round(self.image * 255).astype(np.int16)
        self.assertLessEqual(np.abs(decoded[..., ::-1].astype(np.int16) - expected).max(), 1)
        self.assert_directory(['frame.png'])

    def test_false_encoder_result_preserves_destination_and_cleans_partial(self):
        for old_destination in (False, True):
            with self.subTest(old_destination=old_destination):
                if old_destination:
                    self.path.write_bytes(b'old destination')
                def failed_encode(path, *args):
                    Path(path).write_bytes(b'partial encoded bytes')
                    return False
                with mock.patch.object(look.cv2, 'imwrite', side_effect=failed_encode):
                    with self.assertRaisesRegex(OSError, 'PNG encoding failed.*frame.png'):
                        look.save_png(self.path, self.image)
                if old_destination:
                    self.assertEqual(self.path.read_bytes(), b'old destination')
                else:
                    self.assertFalse(self.path.exists())
                self.assert_directory(['frame.png'] if old_destination else [])

    def test_encoder_exception_preserves_destination_and_cleans_partial(self):
        self.path.write_bytes(b'old destination')
        error = cv2.error('injected encoder exception')
        def failed_encode(path, *args):
            Path(path).write_bytes(b'partial encoded bytes')
            raise error
        with mock.patch.object(look.cv2, 'imwrite', side_effect=failed_encode):
            with self.assertRaises(cv2.error) as raised:
                look.save_png(self.path, self.image)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.path.read_bytes(), b'old destination')
        self.assert_directory(['frame.png'])

    def test_replace_failure_preserves_destination_and_cleans_encoded_temp(self):
        self.path.write_bytes(b'old destination')
        error = PermissionError('injected rename failure')
        with mock.patch.object(look.os, 'replace', side_effect=error):
            with self.assertRaises(PermissionError) as raised:
                look.save_png(self.path, self.image)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.path.read_bytes(), b'old destination')
        self.assert_directory(['frame.png'])

    def test_cleanup_failure_does_not_mask_false_encoder_result(self):
        self.path.write_bytes(b'old destination')
        def failed_encode(path, *args):
            Path(path).write_bytes(b'partial encoded bytes')
            return False
        with mock.patch.object(look.cv2, 'imwrite', side_effect=failed_encode), \
                mock.patch.object(look.os, 'remove', side_effect=PermissionError('injected cleanup failure')):
            with self.assertRaisesRegex(OSError, 'PNG encoding failed.*frame.png'):
                look.save_png(self.path, self.image)
        self.assertEqual(self.path.read_bytes(), b'old destination')

    def test_cleanup_failure_does_not_mask_encoder_exception(self):
        self.path.write_bytes(b'old destination')
        error = cv2.error('injected encoder exception')
        def failed_encode(path, *args):
            Path(path).write_bytes(b'partial encoded bytes')
            raise error
        with mock.patch.object(look.cv2, 'imwrite', side_effect=failed_encode), \
                mock.patch.object(look.os, 'remove', side_effect=PermissionError('injected cleanup failure')):
            with self.assertRaises(cv2.error) as raised:
                look.save_png(self.path, self.image)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.path.read_bytes(), b'old destination')


if __name__ == '__main__':
    unittest.main()
