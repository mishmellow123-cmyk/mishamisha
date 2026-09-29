"""C5 caption placement: a caption broken over two lines keeps its words and writes in reading order, a one-line
caption renders as before, and edit/tools/c5_caption_backdrop.py tells a clean backdrop from a busy or a dark one.
Synthetic frames only (no render is read).

Run: python -B -m unittest discover -s the-long-dawn/edit/tests -p 'test_*.py'
"""
from pathlib import Path
import sys
import unittest

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
sys.path.insert(0, str(EDIT / 'tools'))
import c5_caption_backdrop as BD  # noqa: E402
import titles  # noqa: E402

WORDS = 'They left the gold in the ground.'
BREAK = ('They left the gold', 'in the ground.')


def ink_row(**kw):
    return dict(dict(id='T', row=18, line=WORDS, f_in=0, f_out=140, set='ink', x=310, y=300), **kw)


class CaptionBlockTests(unittest.TestCase):
    def test_a_break_keeps_the_words_and_stacks_two_lines(self):
        one = titles.TextV3('C', ink_row(), 1.0)
        two = titles.TextV3('C', ink_row(lines=BREAK), 1.0)
        self.assertEqual(two.text, WORDS)
        self.assertEqual(len(one.bands), 1)
        self.assertEqual(len(two.bands), 2)
        (a0, a1, _, _), (b0, b1, _, _) = two.bands
        self.assertGreater(b0, a0 + 0.9 * two.size)                          # the second line sits below the first
        self.assertLess(two.w, 0.7 * one.w)
        self.assertEqual((two.x0 + two.w / 2.0, two.y0 + two.h / 2.0), (310.0, 300.0))
        with self.assertRaises(ValueError):                                   # a break may not change the words
            titles.TextV3('C', ink_row(lines=('They left the gold', 'in the ground')), 1.0)

    def test_the_ink_writes_the_first_line_before_the_second(self):
        ln = titles.TextV3('C', ink_row(lines=BREAK), 1.0)
        (a0, a1, _, _), (b0, b1, _, _) = ln.bands
        core = ln.alpha > 0.9
        written = []
        for t in (6, 18, 30):
            a = ln._ink(t)
            written.append(tuple(float((a[y0:y1] > 0.45)[core[y0:y1]].mean()) for y0, y1 in ((a0, a1), (b0, b1))))
        self.assertGreater(written[0][0], 0.3)                               # early: the first line is being written
        self.assertLess(written[0][1], 0.05)                                 # and the second not yet begun
        self.assertGreater(written[1][1], 0.3)                               # later: the second line follows
        self.assertTrue(all(v > 0.99 for v in written[2]))                  # by frame 30 both are written

    def test_a_one_line_caption_renders_exactly_as_before(self):
        ln = titles.TextV3('C', ink_row(), 1.0)
        a, stag, w, h = titles.render_line(WORDS, titles.EBG_ITALIC, 54, 500, 0.015)
        self.assertIs(ln.alpha, a)
        self.assertIs(ln.stag, stag)
        self.assertEqual((ln.w, ln.h), (w, h))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        np.testing.assert_array_equal(ln.xn, xx / (w - 1))
        np.testing.assert_array_equal(ln.yy, yy)


class FakeCtx:
    def __init__(self, img):
        self.img = img

    def picture(self, f):
        return self.img.copy(), None, 'synthetic', None


class BackdropToolTests(unittest.TestCase):
    def measure(self, img, **kw):
        return BD.measure_row(FakeCtx(img), ink_row(x=960, y=402, **kw), step=11)

    def test_plain_paper_is_clean(self):
        r = self.measure(np.full((804, 1920, 3), 0.85, np.float32))
        self.assertEqual(r['flags'], [])
        self.assertLess(r['median']['edge'], 0.001)
        self.assertGreater(r['worst_frame']['worst'], 7)

    def test_hatching_is_busy(self):
        img = np.full((804, 1920, 3), 0.85, np.float32)
        img[::7] = 0.35                                                       # an engraver's hatching
        img[:, ::9] = 0.35
        r = self.measure(img)
        self.assertTrue(any(f.startswith('BUSY (edge') for f in r['flags']), r['flags'])
        self.assertTrue(any(f.startswith('BUSY (texture') for f in r['flags']), r['flags'])

    def test_ink_on_dark_paper_is_low_contrast(self):
        r = self.measure(np.full((804, 1920, 3), 0.3, np.float32))
        self.assertTrue(any(f.startswith('LOW-CONTRAST') for f in r['flags']), r['flags'])
        self.assertLess(r['worst_frame']['worst'], BD.CR_FLOOR)

    def test_a_dark_band_under_one_word_fails_the_worst_slice_not_the_mean(self):
        img = np.full((804, 1920, 3), 0.85, np.float32)
        ln = titles.TextV3('C', ink_row(x=960, y=402), 1.0)
        x0, y0, x1, y1 = BD.glyph_extent(ln)
        img[:, x1 - (x1 - x0) // 8:x1 + 20] = 0.25                            # the last eighth of the line in shadow
        r = self.measure(img)
        self.assertGreater(r['median']['contrast'], BD.CR_FLOOR)
        self.assertLess(r['median']['worst'], BD.CR_FLOOR)


if __name__ == '__main__':
    unittest.main()
