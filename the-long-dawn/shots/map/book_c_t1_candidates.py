"""Default-off T1 wording and camera studies in the original book renderer.

``make_renderer()`` returns the original Book3. current-words alters the T1
page-texture callback with the current provisional R02 wording in the existing
hand; no-t1 leaves that caption band empty. Illustration, light, material,
T14 line and frame dispatch stay inherited.
current-words-held additionally tilts the Mountain camera toward the caption,
then rejoins the original camera at C557; the other options keep their cameras.

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
and camera-projection tests; no new T1 render was run in this geometry-only lane.
"""
import zlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import book_c as BC
import inkline as IL


CANDIDATES = ('accepted', 'current-words', 'no-t1', 'current-words-held')
CAPTION_SOURCE_COMMIT = 'df7a3aed38153158c1631074375cea10b36419ab'
INK_SEED_SOURCE_COMMIT = 'b8939059dda08d77bbe8d7399896791168c8b786'
CURRENT_WORDS = 'In the old story, a Dark Lord forges a Ring to rule the world.'
REPRESENTATIVE_FRAMES = (404, 412, 430, 528)
CAPTION_BAND_CM = (2.3, 12.8, 17.3, 14.2)

# C3: arrive at 2.1 s, settle toward the sentence by C450, lift toward the
# drawing from C512, and rejoin at C557 (including the original final velocity).
# At native resolution the glyph/bleed support bottom stays at y < 757 through
# C527, versus original row 812 there; see t1_camera_projection.py. The 0.025
# rad tilt keeps a reserve beyond the required 40 px and the measured 1.35 px
# original-projection residual. No dolly or lens change accompanies the tilt.
HELD_START = 2.1
HELD_SETTLE = (450 - 320) / 24.0
HELD_RELEASE = (512 - 320) / 24.0
HELD_END = (557 - 320) / 24.0
HELD_MAX_TILT = 0.025


def held_tilt(t):
    """Downward angle in radians; C2 correction with compact time support."""
    if t <= HELD_START or t >= HELD_END:
        return 0.0

    def ease(x):
        x = np.clip(x, 0.0, 1.0)
        return x*x*x*(10.0 + x*(-15.0 + 6.0*x))

    return float(HELD_MAX_TILT * ease((t-HELD_START)/(HELD_SETTLE-HELD_START)) *
                 (1.0-ease((t-HELD_RELEASE)/(HELD_END-HELD_RELEASE))))


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
    return None if candidate == 'no-t1' else TextInkLine(CURRENT_WORDS, ppc)


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


class HeldWordsBook3(CandidateBook3):
    """Same ink and push-in; one eased tilt follows the sentence, then leaves it."""

    def cam_mountain(self, bk, t):
        original = super().cam_mountain(bk, t)
        angle = held_tilt(t)
        if angle == 0.0:
            return original
        # Rotate the sightline in its vertical plane at fixed target distance.
        # Retain the original focus/aperture controls as well as its lens.
        direction = original.f*np.cos(angle) - original.u*np.sin(angle)
        cam = BC.B.Cam(original.pos, original.pos + original.dist*direction,
                       original.hfov, self.W, self.H, focus=original.focus,
                       fstop=original.fstop)
        cam.dof_k = original.dof_k
        return cam


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
    cls = HeldWordsBook3 if candidate == 'current-words-held' else CandidateBook3
    return cls(candidate, width, height, with_fire=with_fire)
