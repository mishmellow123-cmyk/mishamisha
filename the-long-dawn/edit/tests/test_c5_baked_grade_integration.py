"""Baked-caption grade dispatch and delivery dependencies, with synthetic pixels.

No delivered frame, book renderer, LUT or production geometry file is read.
"""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS  # noqa: E402
import caption_grade as CG  # noqa: E402
import deliver as D  # noqa: E402


WINDOWS = (
    dict(id='R02', kind='caption_grade', f0=400, f1=540, full0=430, full1=528,
         source_stem='cand_t1_current-words-held', source_off=0,
         curve=dict(kind='shadow_shoulder', low=0.04, high=0.14, gain=1.3)),
    dict(id='title', kind='caption_grade', f0=5748, f1=5880, full0=5808, full1=5847,
         source_stem='book_C', source_off=1280,
         curve=dict(kind='blackpoint_gain', blackpoint=0.0, gain=2.5)),
)


class BakedGradeIntegrationTests(unittest.TestCase):
    def test_wrapper_finishes_once_before_grading_and_supports_unfinished_preview(self):
        raw = np.full((2, 3, 3), 0.2, np.float32)
        source = dict(stem='synthetic-source')
        for finishing in (True, False):
            with self.subTest(finishing=finishing):
                events = []

                def finish(image, src, cut, frame):
                    events.append('finish')
                    self.assertIs(image, raw)
                    self.assertIs(src, source)
                    self.assertEqual((cut, frame), ('C', 450))
                    return image + np.float32(0.1)

                def grade(image, frame, spec, src):
                    events.append('grade')
                    expected = raw + np.float32(0.1) if finishing else raw
                    np.testing.assert_array_equal(image, expected)
                    self.assertEqual(frame, 450)
                    self.assertIs(spec, WINDOWS[0])
                    self.assertIs(src, source)
                    return image * np.float32(0.5)

                with mock.patch.object(CG, 'apply', side_effect=grade):
                    out = AS._tk_caption_grade(raw, source, 450, WINDOWS[0], finish if finishing else None, 'C')
                self.assertEqual(events, ['finish', 'grade'] if finishing else ['grade'])
                np.testing.assert_array_equal(out, (raw + np.float32(0.1) if finishing else raw) * np.float32(0.5))

    def test_wrapper_rejects_other_cuts_before_finishing_or_grading(self):
        image = np.zeros((2, 3, 3), np.float32)
        for cut in ('A', 'B'):
            with self.subTest(cut=cut), mock.patch.object(CG, 'apply') as grade:
                finish = mock.Mock()
                with self.assertRaisesRegex(ValueError, 'cut C'):
                    AS._tk_caption_grade(image, {}, 450, WINDOWS[0], finish, cut)
                finish.assert_not_called()
                grade.assert_not_called()

    def test_dispatcher_uses_both_windows_and_preserves_neighbors_and_other_cuts(self):
        raw_reads, finish_calls, grade_calls = [], [], []
        plate = np.full((2, 3, 3), 0.2, np.float32)

        class Context:
            def __init__(self, cut, variant, scale, clean):
                self.cut = cut

            def picture(self, frame):
                raw_reads.append((self.cut, frame))
                return plate.copy(), {'sec': 'synthetic-shot'}, 'synthetic', {'stem': 'synthetic-source'}

        class Finisher:
            def __call__(self, image, source, cut, frame):
                finish_calls.append((cut, frame))
                return image + np.float32(0.1)

        def grade(image, frame, spec, source):
            grade_calls.append((frame, spec['id']))
            np.testing.assert_array_equal(image, plate + np.float32(0.1))
            return image + np.float32(0.1 * CG.strength(frame, spec))

        windows = deepcopy(WINDOWS)
        fake_stage = SimpleNamespace(Finisher=lambda look: Finisher())
        cases = [(window, frame, active, amount) for window in windows for frame, active, amount in (
            (window['f0'] - 1, False, 0), (window['f0'], True, 0),
            (window['full0'], True, 1), (window['full1'], True, 1),
            (window['f1'] - 1, True, 0), (window['f1'], False, 0))]
        with mock.patch.object(AS, 'Ctx', Context), mock.patch.object(AS, '_CTX', None), \
                mock.patch.dict(AS.EDL.TRANS, {'A': [], 'B': [], 'C': windows, 'D': deepcopy(windows)}), \
                mock.patch.dict(sys.modules, stage=fake_stage), mock.patch.object(CG, 'apply', side_effect=grade):
            for cut in ('A', 'B', 'C', 'D'):
                AS._init(cut, None, 0.5, True, True)
                for window, frame, active, amount in cases:
                    with self.subTest(cut=cut, frame=frame):
                        raw_reads.clear()
                        finish_calls.clear()
                        grade_calls.clear()
                        out, _, status, _ = AS._CTX.picture(frame)
                        enabled = cut in ('C', 'D') and active
                        self.assertEqual(raw_reads, [(cut, frame)])
                        self.assertEqual(finish_calls, [(cut, frame)])
                        self.assertEqual(grade_calls, [(frame, window['id'])] if enabled else [])
                        expected = plate + np.float32(0.1)
                        if enabled:
                            expected = expected + np.float32(0.1 * amount)
                        np.testing.assert_array_equal(out, expected)
                        self.assertEqual(' + caption_grade' in status, enabled)

    def assert_only_baked_shots_rekey(self, change):
        with tempfile.TemporaryDirectory() as folder:
            helper = Path(folder) / 'synthetic_grade.py'
            geometry = Path(folder) / 'synthetic_bands.json'
            helper.write_text('# synthetic helper identity\n')
            geometry.write_text('{"synthetic_geometry": 1}\n')
            code = {k: 'synthetic-code' for k in ('frame', 'text', 'slate', 'x2', 'title')}

            def keys():
                out = {}
                with mock.patch.object(D, '_TRANS_CODE', []):
                    for cut in ('A', 'B', 'C', 'D'):
                        for index, shot in enumerate(AS.EDL.EDL[cut]):
                            take = shot['takes'][0] if shot['takes'] else None
                            plan = dict(kind='take' if take else shot['kind'], take=take,
                                        have=shot['f1'] - shot['f0'], alt=False)
                            out[(cut, index)] = D.segment_key(cut, None, D.PROFILES['master'], index, shot,
                                                            plan, code, [])
                return out

            expected = {(cut, index) for cut in ('C', 'D') for index, shot in enumerate(AS.EDL.EDL[cut])
                        if shot['sec'] in ('C3', 'C23')}
            self.assertEqual(len(expected), 4, 'Both C and D must cover both baked-caption shots')
            with mock.patch.object(CG, '__file__', str(helper)), mock.patch.object(CG, 'DATA_PATH', geometry), \
                    mock.patch.dict(AS.EDL.TRANS, {'A': [], 'B': [], 'C': deepcopy(WINDOWS), 'D': deepcopy(WINDOWS)}), \
                    mock.patch.object(D, 'frame_sources', side_effect=lambda *args: []), \
                    mock.patch.object(D, '_stat', return_value='synthetic-source-stat'):
                before = keys()
                changed_file = helper if change == 'helper' else geometry
                changed_file.write_bytes(changed_file.read_bytes() + b'\n')
                after = keys()
            self.assertEqual({key for key in before if before[key] != after[key]}, expected,
                             'Helper and geometry contents must rekey C3/C23 in C and D only; other shots retain keys')

    def test_helper_source_changes_rekey_only_c3_and_c23(self):
        self.assert_only_baked_shots_rekey('helper')

    def test_geometry_content_changes_rekey_only_c3_and_c23(self):
        self.assert_only_baked_shots_rekey('geometry')


if __name__ == '__main__':
    unittest.main()
