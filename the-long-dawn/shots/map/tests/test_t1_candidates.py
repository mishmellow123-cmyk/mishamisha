"""T1 callback/mask contracts; no scene renders, page textures or Numba calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import book_c_t1_candidates as C


def small_channels(line):
    """Apply the original algorithm to a small canvas without page allocation."""
    local = copy.copy(line)
    local.r0 = local.c0 = 2
    height, width = local.a.shape
    return local, np.zeros((height+4, width+4, 7), np.float32)


class T1Candidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        C.cv2.setNumThreads(0)

    def test_default_is_the_original_book3_constructor(self):
        with patch.object(C.BC, 'Book3', autospec=True) as original:
            for kwargs in ({}, {'candidate': 'accepted'}):
                self.assertIs(C.make_renderer(**kwargs), original.return_value)
                original.assert_called_with(960, 402, with_fire=False)
            C.make_renderer(scale=1., with_fire=True)
            original.assert_called_with(1920, 804, with_fire=True)

    def test_camera_scene_material_light_and_dispatch_are_inherited(self):
        for name in ('frame', 'mountain', 'shot_mountain', 'cam_mountain', 'light',
                     'finish_layer', 'shot_deep', 'shot_last_pages', 'tex_text'):
            self.assertIs(getattr(C.CandidateBook3, name), getattr(C.BC.Book3, name))
        self.assertIs(C.TextInkLine.apply, C.IL.InkLine.apply)
        self.assertIs(C.TextInkLine.active, C.IL.InkLine.active)

    def test_legacy_words_reproduce_every_original_mask_exactly(self):
        for ppc in (45, 110):
            original = C.IL.InkLine('T1', ppc)
            private = C.TextInkLine(C.IL.LINES['T1'][0], ppc)
            for name in ('f_in', 'f_out', 'ppc', 'c0', 'r0'):
                self.assertEqual(getattr(original, name), getattr(private, name))
            for name in ('a', 'xx', 'yy', 'noise', 'dens'):
                np.testing.assert_array_equal(getattr(original, name), getattr(private, name))
            for frame in (399, 400, 404, 412, 424, 430, 535, 536, 539, 540):
                a, ca = small_channels(original)
                b, cb = small_channels(private)
                a.apply(ca, frame); b.apply(cb, frame)
                np.testing.assert_array_equal(ca, cb)

    def test_new_words_fit_the_original_caption_band_and_timing(self):
        original = C.IL.InkLine('T1', 110)
        line = C.make_line('current-words')
        self.assertEqual(line.text, 'In the old story, a Dark Lord forges a Ring to rule the world.')
        self.assertEqual((line.f_in, line.f_out, line.ppc), (400, 540, 110))
        self.assertEqual(line.r0, original.r0)
        x0, y0, x1, y1 = C.CAPTION_BAND_CM
        self.assertGreaterEqual(line.c0, x0*110)
        self.assertGreaterEqual(line.r0, y0*110)
        self.assertLessEqual(line.c0+line.a.shape[1], x1*110)
        self.assertLessEqual(line.r0+line.a.shape[0], y1*110)
        self.assertTrue(np.isfinite(line.a).all())
        self.assertTrue(np.isfinite(line.dens).all())

    def test_write_on_and_effective_dissolve_match_original_algorithm(self):
        line = C.make_line('current-words')
        energies = {}
        for frame in (399, 400, 404, 412, 424, 430, 528, 529, 535, 536, 539, 540):
            local, channels = small_channels(line)
            local.apply(channels, frame)
            energies[frame] = float(channels[..., 0].sum())
            self.assertTrue((channels[..., 2:] == 0).all())
        self.assertEqual(energies[399], 0)
        self.assertEqual(energies[400], 0)
        self.assertLess(energies[404], energies[412])
        self.assertLess(energies[412], energies[430])
        for frame in (528, 529, 535):
            self.assertEqual(energies[frame], energies[430])
        self.assertLess(energies[536], energies[535])
        self.assertLess(energies[539], energies[536])
        self.assertEqual(energies[540], 0)

    def test_no_t1_omits_callback_without_constructing_blank_text(self):
        renderer = C.make_renderer('no-t1')
        with patch.object(C.IL, 'InkLine', side_effect=AssertionError('T1 must not be constructed')):
            for frame in (0, 399, 400, 412, 430, 539, 540, 6814):
                self.assertIsNone(renderer.ink_line('T1', frame, 110))
        self.assertEqual(renderer.cache, {})

    def test_t14_delegates_with_original_key_and_arrays(self):
        for name in C.CANDIDATES[1:]:
            renderer = C.make_renderer(name)
            with patch.object(C.BC.Book3, 'ink_line', autospec=True) as original:
                result = renderer.ink_line('T14', 6814, 100)
                self.assertIs(result, original.return_value)
                original.assert_called_once_with(renderer, 'T14', 6814, 100)
            self.assertIsNotNone(renderer.ink_line('T14', 6814, 100))
            line = renderer.cache[('inkline', 'T14', 100)]
            reference = C.IL.InkLine('T14', 100)
            for key in ('a', 'xx', 'yy', 'noise', 'dens'):
                np.testing.assert_array_equal(getattr(line, key), getattr(reference, key))

    def test_candidates_are_instance_local_and_leave_global_inputs_unchanged(self):
        lines = copy.deepcopy(C.IL.LINES)
        original_class = C.IL.InkLine
        state = np.random.get_state()
        first, second = C.make_renderer('current-words'), C.make_renderer('no-t1')
        self.assertIsNotNone(first.ink_line('T1', 430, 110))
        self.assertIsNone(second.ink_line('T1', 430, 110))
        a = first.cache[('candidate-T1', 'current-words', 110)]
        b = C.make_line('current-words')
        for name in ('a', 'xx', 'yy', 'noise', 'dens'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        self.assertEqual(C.IL.LINES, lines)
        self.assertIs(C.IL.InkLine, original_class)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0]); np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        self.assertEqual(first.books, {})
        self.assertEqual(second.books, {})

    def test_inactive_candidate_returns_none_and_reuses_its_mask(self):
        renderer = C.make_renderer('current-words')
        self.assertIsNone(renderer.ink_line('T1', 399, 110))
        key = ('candidate-T1', 'current-words', 110)
        line = renderer.cache[key]
        self.assertIsNotNone(renderer.ink_line('T1', 400, 110))
        self.assertIsNone(renderer.ink_line('T1', 540, 110))
        self.assertIs(renderer.cache[key], line)

    def test_cross_process_hash_seeds_leave_masks_grain_and_applied_ink_identical(self):
        script = '''
import copy, hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import book_c_t1_candidates as C
import numpy as np
C.cv2.setNumThreads(0)
def signature(line):
    values = {name: hashlib.sha256(getattr(line, name).tobytes()).hexdigest()
              for name in ('a', 'xx', 'yy', 'noise', 'dens')}
    local = copy.copy(line)
    local.r0 = local.c0 = 2
    h, w = local.a.shape
    chan = np.zeros((h+4, w+4, 7), np.float32)
    local.apply(chan, local.f_in+12)
    values['applied'] = hashlib.sha256(chan.tobytes()).hexdigest()
    values['placement'] = [line.r0, line.c0]
    return values
original = signature(C.IL.InkLine('T1', 110))
legacy = signature(C.TextInkLine(C.IL.LINES['T1'][0], 110))
assert original == legacy, 'T1 wrapper differs from current original InkLine'
assert C.make_renderer('no-t1').ink_line('T1', 412, 110) is None
assert type(C.make_renderer()) is C.BC.Book3
print(json.dumps(dict(salted_hash=hash('T1'), arrays=dict(
    original=original, legacy=legacy, current=signature(C.make_line('current-words')),
    t14=signature(C.IL.InkLine('T14', 100))))))
'''
        records = []
        for seed in ('0', '1'):
            env = os.environ.copy()
            env.update(PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE='1',
                       NUMBA_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
            run = subprocess.run([sys.executable, '-B', '-c', script, str(HERE)],
                                 env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, msg=f'PYTHONHASHSEED={seed}: {run.stderr}')
            records.append(json.loads(run.stdout))
        self.assertNotEqual(records[0]['salted_hash'], records[1]['salted_hash'])
        self.assertEqual(records[0]['arrays'], records[1]['arrays'])

    def test_candidates_do_not_require_a_hash_seed_environment_setting(self):
        with patch.dict(os.environ):
            os.environ.pop('PYTHONHASHSEED', None)
            self.assertIsNotNone(C.make_renderer('current-words').ink_line('T1', 412, 110))
            self.assertIsNone(C.make_renderer('no-t1').ink_line('T1', 412, 110))

    def test_invalid_candidate_scale_and_text_fail_before_scene_allocation(self):
        for name in ('unknown', ''):
            with self.assertRaises(ValueError):
                C.make_renderer(name)
        for scale in (0, -1, np.nan, np.inf, 1.1, .00001):
            with self.assertRaises(ValueError):
                C.make_renderer(scale=scale)
        for text in ('', ' '):
            with self.assertRaises(ValueError):
                C.TextInkLine(text, 110)


if __name__ == '__main__':
    unittest.main()
