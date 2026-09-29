"""Default-off T1 wording studies in the unchanged original book renderer.

``make_renderer()`` returns the original Book3. Opt-ins alter only the T1
page-texture callback: current-words uses the current provisional R02 wording
in the existing hand; no-t1 leaves that caption band empty. Every camera,
illustration, light, material, T14 line and frame dispatch stays inherited.

Current wording/timing was read from edit/titles.py at owner integration commit
df7a3aed38153158c1631074375cea10b36419ab. That EDIT table still says in_picture:
no-t1 does NOT automatically add an EDIT caption. Adoption would require EDIT
to choose its setting; this module never edits or imports EDIT.

Ink grain follows the owner's CRC32 seed fix in b893905, independently of
PYTHONHASHSEED. Keep original page ppc110 and use Book3's original HDR grade.
All first-half images produced here are new engine reconstructions; original
delivered book_C frames were not available for comparison when prepared.

The archived C430 study used the previous hash('T1') seed with PYTHONHASHSEED=0.
Its images do not validate this updated grain. This revision has mask/callback
tests only; no new T1 render was authorized after that study's memory-gate failure.
"""
import zlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import book_c as BC
import inkline as IL


CANDIDATES = ('accepted', 'current-words', 'no-t1')
CAPTION_SOURCE_COMMIT = 'df7a3aed38153158c1631074375cea10b36419ab'
INK_SEED_SOURCE_COMMIT = 'b8939059dda08d77bbe8d7399896791168c8b786'
CURRENT_WORDS = 'In the old story, a Dark Lord forges a Ring to rule the world.'
REPRESENTATIVE_FRAMES = (404, 412, 430, 528)
CAPTION_BAND_CM = (2.3, 12.8, 17.3, 14.2)


class TextInkLine(IL.InkLine):
    """Original InkLine raster construction with an instance-owned text value.

    The original constructor accepts only a global table key. Keeping its small
    construction algorithm here avoids any transient mutation of IL.LINES;
    apply() and active() remain the original methods. A legacy-word equality
    test pins every mask/noise/density array against the original constructor.
    """
    def __init__(self, text, ppc):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('InkLine needs visible text; use no-t1 to omit it')
        if not np.isfinite(ppc) or ppc <= 0:
            raise ValueError('ppc must be finite and positive')
        _, self.f_in, self.f_out, uc, vb, size = IL.LINES['T1']
        self.text = text
        self.ppc = ppc
        px = max(8, int(round(size*ppc)))
        font = ImageFont.truetype(IL.FONT, px)
        asc, desc = font.getmetrics()
        w = int(np.ceil(font.getlength(text)))+2*px
        h = asc+desc+px
        image = Image.new('L', (w, h), 0)
        ImageDraw.Draw(image).text((px, px//2), text, font=font, fill=255)
        a = np.asarray(image, np.float32)/255.
        ys, xs = np.nonzero(a > .01)
        a = a[max(ys.min()-2, 0):ys.max()+3, max(xs.min()-2, 0):xs.max()+3]
        self.a = a
        base_row = (px//2+asc)-max(ys.min()-2, 0)
        self.c0 = int(round(uc*ppc-a.shape[1]/2))
        self.r0 = int(round(vb*ppc-base_row))
        hh, ww = a.shape
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        self.xx, self.yy = xx, yy
        rng = np.random.default_rng(zlib.crc32(str('T1').encode()) & 0xffff)
        noise = cv2.resize(rng.random((max(2, hh//6), max(2, ww//6))).astype(np.float32), (ww, hh))
        self.noise = cv2.GaussianBlur(noise, (0, 0), 1.5)
        s = xx/max(ppc, 1.)
        cycle = (s % 9.)/9.
        self.dens = (1.-.28*cycle)*(.9+.1*self.noise)


def make_line(candidate='accepted', ppc=110):
    """Lightweight T1 mask construction; no scene or page texture allocation."""
    if candidate not in CANDIDATES:
        raise ValueError('unknown T1 candidate')
    if candidate == 'accepted':
        return IL.InkLine('T1', ppc)
    return TextInkLine(CURRENT_WORDS, ppc) if candidate == 'current-words' else None


class CandidateBook3(BC.Book3):
    def __init__(self, candidate, W=1920, H=804, with_fire=False):
        if candidate not in CANDIDATES[1:]:
            raise ValueError('unknown opt-in T1 candidate')
        super().__init__(W, H, with_fire=with_fire)
        self.t1_candidate = candidate

    def ink_line(self, key, f, ppc):
        if key != 'T1':
            return super().ink_line(key, f, ppc)
        if self.t1_candidate == 'no-t1':
            return None
        line = self.once(('candidate-T1', self.t1_candidate, ppc),
                         lambda: make_line(self.t1_candidate, ppc))
        if not line.active(f):
            return None
        return lambda channels: line.apply(channels, f)


def make_renderer(candidate='accepted', scale=.5, with_fire=False):
    """Book3 API: frame(absolute_C_frame) returns (HDR, alpha), or None.

    Original mountain page ppc remains110; it is not an output-scale parameter.
    No blanket frame guard is added, so every other Book3 shot retains dispatch.
    """
    if candidate not in CANDIDATES:
        raise ValueError('unknown T1 candidate')
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('scale must be finite and in (0,1]')
    width, height = int(1920*scale), int(804*scale)
    if min(width, height) < 1:
        raise ValueError('scale produces an empty frame')
    if candidate == 'accepted':
        return BC.Book3(width, height, with_fire=with_fire)
    return CandidateBook3(candidate, width, height, with_fire=with_fire)
