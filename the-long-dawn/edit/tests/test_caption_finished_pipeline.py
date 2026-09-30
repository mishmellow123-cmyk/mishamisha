"""Caption measurement/search must see the master's FINISH and EDIT picture stages.

Synthetic pixels only: the source is pale, FINISH makes it dark, and an EDIT window
restores pale paper on the right. The real assemble._init and its wrappers compose
these stages; neither source frames nor LUTs are read.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
sys.path.insert(0, str(EDIT / 'tools'))
import assemble as AS  # noqa: E402
import c5_caption_backdrop as BD  # noqa: E402


W, H = 192, 96
PALE, DARK = 0.875, 0.25


class SyntheticCtx:
    def __init__(self, cut, variant, scale, clean):
        self.cut, self.W, self.H = cut, W, H

    def picture(self, f):
        return np.full((H, W, 3), PALE, np.float32), {}, 'synthetic', 'synthetic'


class SyntheticFinisher:
    def __init__(self, look):
        pass

    def __call__(self, img, src, cut, f):
        return np.full_like(img, DARK)


def synthetic_transition(img, src, f, window, fin, cut):
    img = fin(img, src, cut, f) if fin else img.copy()
    img[:, W // 2:] = PALE
    return img


class SyntheticCaption:
    """A solid rectangular glyph, large enough for the actual slice/ring metrics."""
    def __init__(self, cut, row, scale):
        self.w, self.h, self.size = 32, 16, 16
        self.x0, self.y0 = row['x'] - self.w // 2, row['y'] - self.h // 2
        self.alpha = np.ones((self.h, self.w), np.float32)
        self.bands = [(0, self.h, 0, self.w)]

    def draw(self, img, f):
        img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w] = BD.titles.IRON


class FinishedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(id=rid, set='ink', f_in=0, f_out=40, x=x, y=36)
                     for rid, x in [('LEFT', 44), ('RIGHT', 140)]]
        self.enterContext(mock.patch.object(BD, 'W', W))
        self.enterContext(mock.patch.object(BD, 'H', H))
        self.enterContext(mock.patch.object(BD.titles, 'TextV3', SyntheticCaption))
        self.enterContext(mock.patch.object(BD, 'delivered_rows',
                          lambda ids=None: [r for r in self.rows if not ids or r['id'] in ids]))
        self.enterContext(mock.patch.object(AS, 'Ctx', SyntheticCtx))
        self.enterContext(mock.patch.object(AS, '_CTX', None))
        self.enterContext(mock.patch.dict(sys.modules, {'stage': SimpleNamespace(Finisher=SyntheticFinisher)}))
        window = dict(kind='synthetic_caption', f0=0, f1=40)
        self.enterContext(mock.patch.dict(AS.EDL.TRANS, {'C': [window]}))
        self.enterContext(mock.patch.dict(AS.TKINDS_SHOT, {'synthetic_caption': synthetic_transition}))

    @staticmethod
    def expected_contrast(ground):
        return BD.ratio(float(BD.rel_lum(np.full(3, ground, np.float32))),
                        float(BD.rel_lum(BD.titles.IRON)))

    def test_measure_reads_both_finished_and_transitioned_pixels(self):
        results = {r['id']: r for r in BD.measure()}
        for rid, ground in [('LEFT', DARK), ('RIGHT', PALE)]:
            with self.subTest(rid=rid):
                result = results[rid]
                self.assertEqual(result['measured'], 2)
                self.assertAlmostEqual(result['worst_frame']['worst'], self.expected_contrast(ground), places=5)
        self.assertTrue(any(f.startswith('LOW-CONTRAST') for f in results['LEFT']['flags']))
        self.assertEqual(results['RIGHT']['flags'], [])

    def test_search_moves_from_dark_finished_paper_to_the_edit_windows_pale_side(self):
        result = BD.search('LEFT', step=1, top=1, grid=16, keep=4)
        best = result['candidates'][0]
        self.assertEqual(result['frames'], 2)
        self.assertGreaterEqual(best['x'], W // 2 + 24)
        self.assertAlmostEqual(best['contrast_est'], self.expected_contrast(PALE), places=5)
        self.assertGreaterEqual(best['contrast_est'], BD.CR_FLOOR)


if __name__ == '__main__':
    unittest.main()
