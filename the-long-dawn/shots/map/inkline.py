"""PAGES-C (28 Sep): C's ink lines written INTO the page (T1 on the Mountain's caption band, T14 on the first healed
blank verso), so they sit on the paper in perspective, under the hearth, in iron-gall ink, instead of floating over
the picture as flat type. EDIT draws no T1/T14 (T7 stays EDIT's: the camera travels down the Deep while it is up).

The same text table as EDIT (music/v3/barmap_C.json via edit/titles.py; frames in/out below) and the same letter
(EB Garamond Italic, the book's printed hand for our lines), written on by a pen-shaped wipe over the first 24 frames
(a slanted nib front, wet and glossy at the nib, a little bleed) and dissolving out over the last 12 (eroding, as
EDIT's did). The ink's density follows a dip cycle along the line (dark after the dip, paler toward the end).
"""
import os
import zlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, '..', '..', 'assets', 'fonts', 'EBGaramond-Italic.ttf')

# id: (text, C frame in, C frame out, page centre u (cm), baseline v (cm), font size (cm))
LINES = {
    'T1': ('In the old story, a Dark Lord forges the Ring in secret.', 400, 540, 9.8, 13.72, 0.52),
    'T14': ('The last pages were left for us.', 6790, 6930, 8.7, 13.4, 1.1),   # 1.5x; off the gutter
}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class InkLine:
    """One line, rastered once at a page texture's ppc; `apply(chan, f)` writes frame f's state into the channels."""

    def __init__(self, key, ppc):
        text, self.f_in, self.f_out, uc, vb, size = LINES[key]
        self.ppc = ppc
        px = max(8, int(round(size * ppc)))
        font = ImageFont.truetype(FONT, px)
        asc, desc = font.getmetrics()
        w = int(np.ceil(font.getlength(text))) + 2 * px
        h = asc + desc + px
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).text((px, px // 2), text, font=font, fill=255)
        a = np.asarray(img, np.float32) / 255.0
        ys, xs = np.nonzero(a > 0.01)
        a = a[max(ys.min() - 2, 0):ys.max() + 3, max(xs.min() - 2, 0):xs.max() + 3]
        self.a = a
        base_row = (px // 2 + asc) - max(ys.min() - 2, 0)          # the baseline inside the crop
        self.c0 = int(round(uc * ppc - a.shape[1] / 2))
        self.r0 = int(round(vb * ppc - base_row))
        hh, ww = a.shape
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        self.xx, self.yy = xx, yy
        rng = np.random.default_rng(zlib.crc32(str(key).encode()) & 0xffff)   # not hash(): salted per process
        n = cv2.resize(rng.random((max(2, hh // 6), max(2, ww // 6))).astype(np.float32), (ww, hh))
        self.noise = cv2.GaussianBlur(n, (0, 0), 1.5)
        # the dip cycle: dark after each dip (every ~9 cm of writing), paler as the pen runs dry; a little grain
        s = xx / max(ppc, 1.0)
        cyc = (s % 9.0) / 9.0
        self.dens = (1.0 - 0.28 * cyc) * (0.9 + 0.1 * self.noise)

    def active(self, f):
        return self.f_in <= f < self.f_out

    def apply(self, chan, f):
        if not self.active(f):
            return
        t = f - self.f_in
        hh, ww = self.a.shape
        edge = 0.35 * self.a.shape[0]
        pos = -edge + (ww + 2 * edge) * min(t / 24.0, 1.0)
        front = pos - (self.xx + 0.35 * (self.yy - hh / 2))
        cov = np.clip(front / edge + 0.5, 0, 1)
        ink = self.a * cov
        wet = np.zeros_like(ink)
        if t < 30:                                          # wet at the nib: a little bleed, glossy while fresh
            band = np.exp(-(front / (1.4 * edge)) ** 2)
            ink = np.maximum(ink, cv2.GaussianBlur(self.a, (0, 0), 0.9) * band * 0.5 * cov)
            wet = np.clip(1.0 - front / (6.0 * edge), 0, 1) * ink
        u = np.clip((self.f_out - f) / 12.0, 0, 1)          # 1 -> 0 over the dissolve
        if u < 1:
            ink = ink * _smooth((u * 1.25 - self.noise * 0.25) / 0.2) * _smooth(u / 0.35)
            wet = wet * u
        ink = ink * self.dens
        H, W = chan.shape[:2]
        r0, c0 = self.r0, self.c0
        r1, c1 = min(H, r0 + hh), min(W, c0 + ww)
        R0, C0 = max(r0, 0), max(c0, 0)
        if r1 <= R0 or c1 <= C0:
            return
        sub = (slice(R0 - r0, r1 - r0), slice(C0 - c0, c1 - c0))
        chan[R0:r1, C0:c1, 0] = np.maximum(chan[R0:r1, C0:c1, 0], ink[sub])
        chan[R0:r1, C0:c1, 1] = np.maximum(chan[R0:r1, C0:c1, 1], wet[sub])
