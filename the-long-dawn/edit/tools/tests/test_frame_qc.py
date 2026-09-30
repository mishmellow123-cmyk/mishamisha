"""Synthetic frame-QC contracts; thresholds are review heuristics, not film calibration.

Run with python -B -m unittest discover -s edit/tools/tests -p test_frame_qc.py -v.
FRAME_QC_UNDER_TEST can select a deliberately broken copy for a negative control.
All image fixtures are tiny; no production media or renderer is required.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

for _name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import cv2
import numpy as np

cv2.setNumThreads(1)
SOURCE = Path(os.environ.get('FRAME_QC_UNDER_TEST') or Path(__file__).resolve().parents[1] / 'frame_qc.py')
SPEC = importlib.util.spec_from_file_location('frame_qc_under_test', SOURCE)
qc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qc
SPEC.loader.exec_module(qc)


class FrameQCTests(unittest.TestCase):
    W, H = 64, 32

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='frame-qc-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = qc.Config(expected_width=self.W, expected_height=self.H, thumbnail_width=self.W)

    def gradient(self, offset=0):
        row = np.linspace(40, 120, self.W).astype(np.uint8)
        gray = np.tile(row[None, :], (self.H, 1)).astype(np.int16) + offset
        return np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2)

    def write(self, frame, image=None, extension='png'):
        path = self.root / f'f_{frame:05d}.{extension}'
        image = self.gradient() if image is None else image
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def analyze(self, start, end, **kwargs):
        options = dict(film='A', config=self.config, joins=[])
        options.update(kwargs)
        return qc.analyze(self.root, start, end, **options)

    def frames(self, report):
        return {f['frame']: f for f in report['frames']}

    def transitions(self, report):
        return {(t['from'], t['to']): t for t in report['transitions']}

    def test_missing_corrupt_and_wrong_dimensions_are_distinct(self):
        self.write(0)
        (self.root / 'f_00002.png').write_bytes(b'not an image')
        self.write(3, np.zeros((self.H - 1, self.W, 3), np.uint8))
        self.write(4, self.gradient(4), extension='jpg')
        report = self.analyze(0, 4)
        frames = self.frames(report)
        self.assertEqual({n: f['status'] for n, f in frames.items()},
                         {0: 'ok', 1: 'missing', 2: 'decode_error', 3: 'wrong_dimensions', 4: 'ok'})
        self.assertEqual((frames[3]['width'], frames[3]['height']), (self.W, self.H - 1))
        self.assertEqual(report['transitions'], [])
        self.assertEqual(report['coverage']['expected_frames'], 5)
        self.assertEqual(report['coverage']['frame_status_counts'],
                         dict(ok=2, missing=1, decode_error=1, wrong_dimensions=1, unsupported_format=0))
        self.assertEqual(report['coverage']['compared_adjacent_pairs'], 0)
        self.assertEqual(report['coverage']['skipped_adjacent_pairs'], 4)
        # The report must remain ordinary strict JSON even for failures.
        json.dumps(report, allow_nan=False)

    def test_default_dimensions_remain_delivery_dimensions(self):
        config = qc.Config()
        self.assertEqual((config.expected_width, config.expected_height), (1920, 804))
        self.write(0)
        report = self.analyze(0, 0, config=config)
        self.assertEqual(self.frames(report)[0]['status'], 'wrong_dimensions')

    def test_d_joins_are_explicit_and_do_not_select_another_films_join(self):
        self.write(0)
        self.write(1, self.gradient(2))
        joins = [{'film': c, 'from': 0, 'to': 1, 'label': f'{c} fixture'} for c in ('C', 'D')]
        report = self.analyze(0, 1, film='D', joins=joins)
        self.assertEqual(report['film'], 'D')
        self.assertEqual([j['label'] for j in report['joins']], ['D fixture'])
        self.assertTrue(report['transitions'][0]['declared_join'])
        # Wrong-cut metadata is a negative fixture: it must not exempt D's transition.
        unselected = self.analyze(0, 1, film='D', joins=joins[:1])
        self.assertEqual(unselected['joins'], [])
        self.assertFalse(unselected['transitions'][0]['declared_join'])

    def test_d_cli_uses_d_join_records(self):
        self.write(0)
        self.write(1, self.gradient(2))
        custom = self.root / 'd_joins.json'
        custom.write_text(json.dumps([{'film': 'D', 'from': 0, 'to': 1, 'label': 'D fixture'}]))
        output = self.root / 'd_qc.json'
        completed = subprocess.run([sys.executable, '-B', str(SOURCE), str(self.root), '--range', '0-1',
                                    '--film', 'D', '--width', str(self.W), '--height', str(self.H),
                                    '--json', str(output), '--joins', str(custom)],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report['film'], 'D')
        self.assertEqual(report['joins'][0]['status'], 'measured')

    def test_black_constant_and_near_constant_frames_have_review_flags(self):
        self.write(0, np.zeros((self.H, self.W, 3), np.uint8))
        self.write(1, np.full((self.H, self.W, 3), 110, np.uint8))
        near = np.full((self.H, self.W, 3), 110, np.uint8)
        near[0, 0] = 111
        self.write(2, near)
        self.write(3, self.gradient())
        frames = self.frames(self.analyze(0, 3))
        self.assertIn('black', frames[0]['flags'])
        self.assertIn('constant', frames[1]['flags'])
        self.assertIn('near_constant', frames[2]['flags'])
        self.assertNotIn('constant', frames[2]['flags'])
        self.assertFalse(set(frames[3]['flags']) & {'black', 'constant', 'near_constant'})

    def test_exact_hold_is_detected_without_mislabeling_a_changed_frame(self):
        image = self.gradient()
        first = self.write(0, image)
        second = self.root / 'f_00001.png'
        self.assertTrue(cv2.imwrite(str(second), image.copy(), [cv2.IMWRITE_PNG_COMPRESSION, 9]))
        self.assertNotEqual(first.read_bytes(), second.read_bytes())
        self.write(2, self.gradient(1))
        pairs = self.transitions(self.analyze(0, 2))
        self.assertTrue(pairs[(0, 1)]['exact_duplicate'])
        self.assertFalse(pairs[(1, 2)]['exact_duplicate'])

    def test_16_bit_grayscale_keeps_low_bit_changes_and_strict_json(self):
        gray = self.gradient()[..., 0].astype(np.uint16) * 257
        self.write(0, gray)
        self.write(1, gray.copy())
        self.write(2, gray + 1)
        report = self.analyze(0, 2)
        for frame in report['frames']:
            self.assertEqual(frame['status'], 'ok')
            self.assertEqual((frame['channels'], frame['dtype']), (1, 'uint16'))
            self.assertFalse(set(frame['flags']) & {'black', 'constant', 'near_constant'})
        pairs = self.transitions(report)
        self.assertTrue(pairs[(0, 1)]['exact_duplicate'])
        self.assertFalse(pairs[(1, 2)]['exact_duplicate'])
        self.assertGreater(pairs[(1, 2)]['luminance_delta'], 0)
        json.dumps(report, allow_nan=False)

    def test_transparent_rgba_is_flagged_and_alpha_changes_prevent_exact_hold(self):
        rgba = np.concatenate((self.gradient(), np.zeros((self.H, self.W, 1), np.uint8)), axis=2)
        self.write(0, rgba)
        rgba[..., 3] = 255
        self.write(1, rgba)
        report = self.analyze(0, 1)
        frames = self.frames(report)
        self.assertEqual(frames[0]['status'], 'ok')
        self.assertEqual(frames[0]['channels'], 4)
        self.assertIn('transparent', frames[0]['flags'])
        self.assertNotIn('transparent', frames[1]['flags'])
        self.assertFalse(report['transitions'][0]['exact_duplicate'])
        # The report's luminance/structure metrics explicitly ignore alpha.
        self.assertEqual(report['transitions'][0]['luminance_delta'], 0)
        self.assertEqual(report['transitions'][0]['structure_delta'], 0)
        json.dumps(report, allow_nan=False)

    def test_smooth_exposure_change_is_not_a_pop_or_hold(self):
        for n in range(12):
            self.write(n, self.gradient(n))
        report = self.analyze(0, 11)
        self.assertEqual(len(report['transitions']), 11)
        for pair in report['transitions']:
            self.assertFalse(pair['exact_duplicate'])
            self.assertEqual(pair['pop_candidates'], [])

    def test_exposure_flash_trips_luminance_review_on_both_edges(self):
        for n in range(12):
            self.write(n, self.gradient(n + (100 if n == 5 else 0)))
        pairs = self.transitions(self.analyze(0, 11))
        for edge in ((4, 5), (5, 6)):
            self.assertIn('luminance', pairs[edge]['pop_candidates'])
            self.assertGreater(pairs[edge]['luminance_delta'], self.config.pop_luma_floor)

    def test_same_brightness_structure_jump_trips_structure_review(self):
        for n in range(12):
            image = self.gradient(n)
            if n == 5:
                image = image[:, ::-1].copy()
            self.write(n, image)
        pairs = self.transitions(self.analyze(0, 11))
        for edge in ((4, 5), (5, 6)):
            self.assertIn('structure', pairs[edge]['pop_candidates'])
            self.assertNotIn('luminance', pairs[edge]['pop_candidates'])
            self.assertLess(pairs[edge]['luminance_delta'], 0.01)

    def test_missing_or_corrupt_frames_do_not_create_bridging_comparisons(self):
        self.write(0)
        self.write(2, self.gradient(2))
        (self.root / 'f_00003.jpg').write_bytes(b'broken jpeg')
        self.write(4, self.gradient(4))
        self.write(5, self.gradient(5))
        self.assertEqual(set(self.transitions(self.analyze(0, 5))), {(4, 5)})

    def test_gap_also_splits_outlier_neighbor_population(self):
        for n in range(11):
            if n != 5:
                self.write(n, self.gradient(n))
        pairs = self.transitions(self.analyze(0, 10))
        self.assertEqual(len(pairs), 8)
        for edge in ((0, 1), (3, 4), (6, 7), (9, 10)):
            for metric in ('luminance', 'structure'):
                baseline = pairs[edge]['baseline'][metric]
                self.assertEqual(baseline['count'], 3)
                self.assertEqual(baseline['method'], 'absolute_floor_only')

    def test_corrupt_preferred_png_is_not_hidden_by_a_valid_jpeg(self):
        (self.root / 'f_00000.png').write_bytes(b'corrupt preferred image')
        self.write(0, extension='jpg')
        report = self.analyze(0, 0)
        self.assertEqual(self.frames(report)[0]['status'], 'decode_error')
        self.assertTrue(self.frames(report)[0]['path'].endswith('.png'))

    def test_declared_cut_is_measured_but_excluded_from_pops_and_neighbor_baselines(self):
        for n in range(10):
            image = self.gradient(n)
            if n >= 5:
                image = image[:, ::-1].copy()
            self.write(n, image)
        join = {'film': 'A', 'from': 4, 'to': 5, 'label': 'synthetic continuity cut'}
        report = self.analyze(0, 9, joins=[join])
        pairs = self.transitions(report)
        self.assertTrue(pairs[(4, 5)]['declared_join'])
        self.assertGreater(pairs[(4, 5)]['structure_delta'], self.config.pop_structure_floor)
        self.assertEqual(pairs[(4, 5)]['pop_candidates'], [])
        measured = report['joins'][0]
        self.assertEqual(measured['status'], 'measured')
        self.assertAlmostEqual(measured['metrics']['structure_delta'], pairs[(4, 5)]['structure_delta'])
        # Each five-frame side has four transitions. The candidate itself is not its own baseline,
        # and the other side of the declared cut must not enter the neighbor population.
        for edge in ((0, 1), (3, 4), (5, 6), (8, 9)):
            for metric in ('luminance', 'structure'):
                self.assertEqual(pairs[edge]['baseline'][metric]['count'], 3)
        self.assertFalse(any(t['pop_candidates'] for t in report['transitions']))

    def test_custom_joins_filter_by_film_and_replace_defaults(self):
        self.write(0)
        self.write(1, self.gradient(1))
        joins = [{'film': 'A', 'from': 0, 'to': 1, 'label': 'A custom'},
                 {'film': 'B', 'from': 0, 'to': 1, 'label': 'B custom'}]
        report = self.analyze(0, 1, film='B', joins=joins)
        self.assertEqual([(j['film'], j['label']) for j in report['joins']], [('B', 'B custom')])
        self.assertTrue(report['transitions'][0]['declared_join'])

    def test_shipped_join_data_contains_exact_handover_pairs_and_filters_films(self):
        expected = {'A': {(4879, 4880), (3679, 3680)}, 'B': {(1519, 1520), (3839, 3840)}}
        for film, pairs in expected.items():
            report = self.analyze(0, 0, film=film, joins=None)
            self.assertEqual({(j['from'], j['to']) for j in report['joins']}, pairs)
            self.assertTrue(all(j['film'] == film for j in report['joins']))
            self.assertTrue(all(j['status'] == 'outside_range' for j in report['joins']))
        unselected = qc.analyze(self.root, 0, 0, config=self.config)
        self.assertEqual(unselected['joins'], [])

    def test_join_missing_endpoint_is_unavailable_with_no_fabricated_metric(self):
        self.write(0)
        report = self.analyze(0, 1, joins=[{'film': 'A', 'from': 0, 'to': 1, 'label': 'missing destination'}])
        join = report['joins'][0]
        self.assertEqual(join['status'], 'unavailable')
        self.assertIsNone(join['metrics'])
        self.assertEqual({e['frame']: e['status'] for e in join['endpoints']}, {0: 'ok', 1: 'missing'})
        self.assertEqual(report['transitions'], [])

    def test_intersecting_join_reads_its_other_endpoint_with_separate_coverage(self):
        self.write(0)
        self.write(1, self.gradient(1))
        report = self.analyze(1, 1, joins=[{'film': 'A', 'from': 0, 'to': 1, 'label': 'outside start'}])
        join = report['joins'][0]
        self.assertEqual(join['status'], 'measured')
        self.assertIsNotNone(join['metrics'])
        self.assertEqual({e['frame']: e['status'] for e in join['endpoints']},
                         {0: 'ok', 1: 'ok'})
        self.assertEqual({e['frame']: e['in_requested_range'] for e in join['endpoints']},
                         {0: False, 1: True})
        self.assertEqual([f['frame'] for f in report['join_endpoint_frames']], [0])
        self.assertEqual([f['frame'] for f in report['frames']], [1])
        self.assertEqual(report['coverage']['expected_frames'], 1)
        self.assertEqual(report['coverage']['extra_join_endpoint_frames'], 1)
        self.assertEqual(report['transitions'], [])

    def test_intersecting_join_missing_extra_endpoint_is_unavailable(self):
        self.write(1)
        report = self.analyze(1, 1, joins=[{'film': 'A', 'from': 0, 'to': 1, 'label': 'missing extra'}])
        join = report['joins'][0]
        self.assertEqual(join['status'], 'unavailable')
        self.assertIsNone(join['metrics'])
        self.assertEqual({e['frame']: e['status'] for e in join['endpoints']}, {0: 'missing', 1: 'ok'})
        self.assertEqual(report['coverage']['frame_status_counts']['missing'], 0)
        self.assertEqual(report['join_endpoint_frames'][0]['status'], 'missing')

    def test_join_entirely_outside_range_remains_unmeasured_even_if_files_exist(self):
        for n in range(3):
            self.write(n, self.gradient(n))
        report = self.analyze(2, 2, joins=[{'film': 'A', 'from': 0, 'to': 1, 'label': 'outside'}])
        self.assertEqual(report['joins'][0]['status'], 'outside_range')
        self.assertIsNone(report['joins'][0]['metrics'])
        self.assertEqual(report['join_endpoint_frames'], [])
        self.assertEqual([f['frame'] for f in report['frames']], [2])

    def test_cli_writes_parseable_json_and_a_short_text_summary(self):
        self.write(0)
        self.write(2, np.zeros((self.H, self.W, 3), np.uint8))
        output = self.root / 'qc.json'
        completed = subprocess.run([sys.executable, '-B', str(SOURCE), str(self.root), '--range', '0-2',
                                    '--film', 'A', '--width', str(self.W), '--height', str(self.H),
                                    '--json', str(output)], capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(self.frames(report)[1]['status'], 'missing')
        self.assertIn('black', self.frames(report)[2]['flags'])
        summary = completed.stdout.lower()
        self.assertIn('missing', summary)
        self.assertIn('black', summary)
        self.assertLess(len(summary.splitlines()), 35)
        json.dumps(report, allow_nan=False)

    def test_cli_custom_joins_replace_defaults_and_extra_joins_append(self):
        for n in range(3):
            self.write(n, self.gradient(n))
        custom = self.root / 'custom.json'
        extra = self.root / 'extra.json'
        custom.write_text(json.dumps([{'film': 'A', 'from': 0, 'to': 1, 'label': 'custom'}]))
        extra.write_text(json.dumps([{'film': 'A', 'from': 1, 'to': 2, 'label': 'extra'}]))
        output = self.root / 'qc.json'
        completed = subprocess.run([sys.executable, '-B', str(SOURCE), str(self.root), '--range', '0-2',
                                    '--film', 'A', '--width', str(self.W), '--height', str(self.H),
                                    '--json', str(output), '--joins', str(custom), '--extra-joins', str(extra)],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(output.read_text())
        self.assertEqual({j['label'] for j in report['joins']}, {'custom', 'extra'})
        self.assertTrue(all(j['status'] == 'measured' for j in report['joins']))
        self.assertTrue(all(t['declared_join'] for t in report['transitions']))

    def test_cli_review_flags_do_not_turn_a_completed_scan_into_an_input_failure(self):
        for n in range(2):
            self.write(n, np.zeros((self.H, self.W, 3), np.uint8))
        output = self.root / 'qc.json'
        completed = subprocess.run([sys.executable, '-B', str(SOURCE), str(self.root), '--range', '0-1',
                                    '--width', str(self.W), '--height', str(self.H), '--json', str(output)],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(output.read_text())
        self.assertIn('black', self.frames(report)[0]['flags'])
        self.assertTrue(report['transitions'][0]['exact_duplicate'])

    def test_cli_unavailable_extra_join_endpoint_returns_failure_with_honest_coverage(self):
        self.write(1)
        joins = self.root / 'joins.json'
        joins.write_text(json.dumps([{'film': 'A', 'from': 0, 'to': 1, 'label': 'missing extra'}]))
        output = self.root / 'qc.json'
        completed = subprocess.run([sys.executable, '-B', str(SOURCE), str(self.root), '--range', '1-1',
                                    '--film', 'A', '--width', str(self.W), '--height', str(self.H),
                                    '--json', str(output), '--joins', str(joins)],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report['coverage']['frame_status_counts']['ok'], 1)
        self.assertEqual(report['coverage']['frame_status_counts']['missing'], 0)
        self.assertEqual(report['coverage']['joins_unavailable'], 1)
        self.assertEqual(report['join_endpoint_frames'][0]['status'], 'missing')
        self.assertIn('1 unavailable', completed.stdout)

    def test_nonfinite_threshold_is_rejected_by_api_and_cli(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                qc.Config(pop_luma_floor=value)
        output = self.root / 'qc.json'
        completed = subprocess.run([sys.executable, '-B', str(SOURCE), str(self.root), '--range', '0-0',
                                    '--json', str(output), '--pop-luma-floor', 'nan'],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 2)
        self.assertIn('finite', completed.stderr)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
