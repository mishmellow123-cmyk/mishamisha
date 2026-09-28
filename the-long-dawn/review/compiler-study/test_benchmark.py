"""Lightweight negative controls for comparison gates; no renderer imports or JIT."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import benchmark as B


class ComparisonTests(unittest.TestCase):
    def pair(self, directory, changed=False, altered_input=False):
        dirs = [Path(directory) / name for name in ('reference', 'candidate')]
        for index, folder in enumerate(dirs):
            folder.mkdir()
            image = np.zeros((2, 3, 3), np.float32)
            if index and changed:
                image[1, 2, 0] = .125
            path = folder / 'image.npy'
            np.save(path, image)
            image_info = {'sha256': B.sha(image.tobytes())}
            call = dict(name='probe', repeat=0, call=0, input_hashes=['same'], seconds=1.,
                        img=image_info, zb={'sha256': 'same-depth'},
                        depth_changed_mask={'sha256': 'same-mask'}, depth_changed_pixels=2)
            if index and altered_input:
                call['input_hashes'] = ['changed-input']
            receipt = dict(status='complete', settings={'mode': 'material'}, versions={'numba': 'test'},
                phase='cold', python='test', platform='test', opencv_version='test', warm_repeats_exact=True,
                harness_sha256='same-harness', numba_threads=1, opencv_reported_threads=1,
                reference_commit='61cf28b', source_sha256={B.SP_PATH: str(index), 'mt/noise.py': 'same'},
                sp_calls=[call], outputs={'image': dict(file='image.npy', file_sha256=B.sha(path.read_bytes()))})
            B.write_json(folder / 'receipt.json', receipt)
        return SimpleNamespace(reference=dirs[0], candidate=dirs[1], output=Path(directory) / 'comparison.json')

    def compare(self, args):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exited:
            B.compare(args)
        return exited.exception.code

    def test_exact_pair_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.pair(temp)
            self.assertEqual(self.compare(args), 0)
            self.assertTrue(json.loads(args.output.read_text())['all_exact'])

    def test_known_float_pixel_change_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.pair(temp, changed=True)
            self.assertEqual(self.compare(args), 1)
            result = json.loads(args.output.read_text())['outputs'][0]
            self.assertEqual(result['changed_values'], 1)
            self.assertEqual(result['max_abs_difference'], .125)

    def test_different_inputs_are_rejected_before_pixel_verdict(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.pair(temp, altered_input=True)
            with self.assertRaisesRegex(AssertionError, 'Inputs differ'):
                B.compare(args)
            self.assertFalse(args.output.exists())

    def test_unrelated_source_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.pair(temp)
            path = args.candidate / 'receipt.json'
            receipt = json.loads(path.read_text())
            receipt['source_sha256']['mt/noise.py'] = 'changed'
            B.write_json(path, receipt)
            with self.assertRaises(AssertionError):
                B.compare(args)

    def test_cache_reload_detects_changed_output(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.pair(temp)
            cold = json.loads((args.reference/'receipt.json').read_text())
            current = copy.deepcopy(cold)
            B.assert_cache_reload(cold, current)
            current['outputs']['image']['file_sha256'] = 'different'
            with self.assertRaisesRegex(AssertionError, 'Cache reload changed outputs'):
                B.assert_cache_reload(cold, current)

    def test_changed_depth_mask_fails_even_with_matching_rgb(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.pair(temp)
            path = args.candidate/'receipt.json'
            record = json.loads(path.read_text())
            record['sp_calls'][0]['depth_changed_mask']['sha256'] = 'changed-mask'
            B.write_json(path, record)
            self.assertEqual(self.compare(args), 1)

    def test_changed_depth_fails_even_with_matching_rgb(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.pair(temp)
            path = args.candidate/'receipt.json'
            record = json.loads(path.read_text())
            record['sp_calls'][0]['zb']['sha256'] = 'changed-depth'
            B.write_json(path, record)
            self.assertEqual(self.compare(args), 1)


if __name__ == '__main__':
    unittest.main()
