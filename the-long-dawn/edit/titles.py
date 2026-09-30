"""Typography for THE LONG DAWN: story lines and the title card.

v3 (BIBLE_V3 locked sheets as amended by the DIRECTOR'S H5 CALLS): `lines_v3(cut)` + `composite_v3(img, lines, f,
scale)`. A's and B's words come from music/v3/barmap_<cut>.json with the H5 wording applied on top (in/out frames
unchanged); C's are C5's own table (C5_TEXT: HANDOVER.md script v5.2, provisional), with BAKED_TEXT recording the
words already inside the renders.
A: Cormorant italic, lower third, 12-frame fades inside the in/out frames; T10a/b centred on black.
B: the title only.  C: ink write-ons (a pen-shaped wipe over 24 frames, dissolve out over 12) and fire lines
(kindle over 12, crumble over 12); T8a/b stacked on black.  Titles (X3) kindle in place: A crumbles into rising
sparks, B fades into the light, C cools from fire to ink.  Everything scales (0.5 for the half-res animatic).

v2 (kept for edit/assemble_v2.py): `story_lines(cut)` + `composite(img, lines, f)`, as below.

Text is rendered once per line into a float alpha mask (PIL + raqm shaping,
variable-font weights, manual tracking that keeps kerning), then animated per
frame: staggered letter fade-in (a soft left-to-right breath), a small upward
settle, blur-to-sharp, and a slow fade out. Composited over the picture with a
soft dark halo for legibility and a faint warm glow so the words feel lit.
"""
import functools
import os
import zlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.path.join(ROOT, 'assets', 'fonts')
W, H = 1920, 804

ITALIC = os.path.join(FONTS, 'CormorantGaramond-Italic.ttf')
ROMAN = os.path.join(FONTS, 'CormorantGaramond.ttf')
CINZEL = os.path.join(FONTS, 'Cinzel.ttf')

INK = np.array([0.953, 0.925, 0.871], np.float32)       # warm paper white (sRGB)
GLOW = np.array([1.0, 0.78, 0.52], np.float32)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
RIM_FLOOR = float(os.environ.get('LD_RIM_FLOOR', '0.10'))   # contact shadow: none under a picture darker than this
RIM_GAIN = float(os.environ.get('LD_RIM_GAIN', '2.2'))      # rising with the local luma above it ...
RIM_MAX = float(os.environ.get('LD_RIM_MAX', '0.75'))        # ... to at most this much darkening at the strokes
GLOW_BACKOFF = float(os.environ.get('LD_GLOW_BACKOFF', '1.2'))  # a fire line's glow fades over lit ground
HEAT_LIFT = float(os.environ.get('LD_HEAT_LIFT', '0.5'))    # and its letters run hotter there
TAGLINE = os.environ.get('TAGLINE', '1') == '1'
STORY_SIZE = int(os.environ.get('STORY_SIZE', '56'))
END_TAG = 2856                                   # 119.0 s when the tagline card is on


@functools.lru_cache(maxsize=64)
def _font(path, size, weight):
    f = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    return f


@functools.lru_cache(maxsize=256)
def render_line(text, path=ITALIC, size=50, weight=560, tracking=0.02):
    """Returns (alpha HxW float32 in [0,1], per-pixel stagger map in [0,1], width, height).
    The stagger map holds each glyph's normalised position along the line so the
    animator can fade letters in with a slight left-to-right delay."""
    font = _font(path, size, weight)
    ascent, descent = font.getmetrics()
    track = tracking * size
    # glyph x positions from substring advances (keeps kerning), plus tracking
    xs = [font.getlength(text[:i]) + i * track for i in range(len(text) + 1)]
    width = int(np.ceil(xs[-1])) + int(size)
    height = ascent + descent + int(size * 0.6)
    pad = int(size * 0.5)
    img = Image.new('L', (width + 2 * pad, height + 2 * pad), 0)
    stag = np.zeros((height + 2 * pad, width + 2 * pad), np.float32)
    n = max(1, len(text.strip()))
    k = 0
    for i, ch in enumerate(text):
        if ch == ' ':
            continue
        glyph = Image.new('L', img.size, 0)
        ImageDraw.Draw(glyph).text((pad + xs[i], pad + size * 0.2), ch, font=font, fill=255)
        g = np.asarray(glyph, np.float32) / 255.0
        img_arr = np.maximum(np.asarray(img, np.float32) / 255.0, g)
        img = Image.fromarray((img_arr * 255).astype(np.uint8))
        stag = np.where(g > 0.0, k / n, stag)
        k += 1
    a = np.asarray(img, np.float32) / 255.0
    # crop horizontally to the ink; vertically to fixed font metrics so every
    # line of the same size shares a baseline (rows of phrases line up)
    ys, xs_ = np.nonzero(a > 0.002)
    y0 = max(0, int(pad + size * 0.2 - size * 0.15))
    y1 = min(a.shape[0], int(pad + size * 0.2 + ascent + descent + size * 0.15))
    x0, x1 = max(0, xs_.min() - pad // 2), min(a.shape[1], xs_.max() + pad // 2)
    a = a[y0:y1, x0:x1]
    stag = stag[y0:y1, x0:x1]
    # spread stagger into the anti-aliased fringe so edges fade with their glyph
    stag = cv2.dilate(stag, np.ones((3, 3), np.uint8))
    return a, stag, a.shape[1], a.shape[0]


@functools.lru_cache(maxsize=64)
def render_block(lines, path=ITALIC, size=50, weight=560, tracking=0.02, leading=1.3):
    """Two or more render_line lines set as ONE caption, each centred, `leading` x size apart. Returns (alpha,
    stagger, reading-order x in [0, 1], slant y, bands), all HxW but bands. The stagger and the reading-order x run
    through the lines in order, so a write-on or a kindle takes them one after another; slant y puts every line's own
    centre at the block's mid-height, so the ink's slanted nib front leans the same way on each line. bands:
    ((y0, y1, x0, x1) of each line's glyphs, alpha > 0.5) in block pixels."""
    parts = [render_line(t, path, size, weight, tracking) for t in lines]
    ph = parts[0][3]                                  # render_line crops every line to the same font metrics
    ww = max(p[2] for p in parts)
    pitch = int(round(leading * size))
    hh = pitch * (len(parts) - 1) + ph
    alpha = np.zeros((hh, ww), np.float32)
    stag, xr, ys = np.zeros_like(alpha), np.zeros_like(alpha), np.zeros_like(alpha)
    letters = [sum(c != ' ' for c in t) for t in lines]
    total_px, total_n = float(sum(p[2] for p in parts)), float(sum(letters))
    done_px, done_n, bands = 0.0, 0.0, []
    cols = np.arange(ww, dtype=np.float32)
    for i, (a, st, pw, _) in enumerate(parts):
        y0, x0 = i * pitch, (ww - pw) // 2
        alpha[y0:y0 + ph, x0:x0 + pw] = np.maximum(alpha[y0:y0 + ph, x0:x0 + pw], a)
        c = y0 + ph / 2.0                                          # this line's centre row
        lo = 0 if i == 0 else int(round(c - pitch / 2.0))         # the rows it owns: halfway to its neighbours
        hi = hh if i == len(parts) - 1 else int(round(c + pitch / 2.0))
        own = np.zeros((hh, pw), np.float32)
        own[y0:y0 + ph] = st * letters[i]
        stag[lo:hi, x0:x0 + pw] = (done_n + own[lo:hi]) / total_n
        xr[lo:hi] = (done_px + np.clip(cols - x0, 0, pw)) / total_px
        ys[lo:hi] = np.arange(lo, hi, dtype=np.float32)[:, None] - c + hh / 2.0
        gy, gx = np.nonzero(a > 0.5)
        bands.append((y0 + int(gy.min()), y0 + int(gy.max()) + 1, x0 + int(gx.min()), x0 + int(gx.max()) + 1))
        done_px += pw
        done_n += letters[i]
    return alpha, stag, xr, ys, tuple(bands)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Line:
    """One animated text element.
    t_in/t_out are global frames: fully faded in by t_in+fade, out by t_out."""

    def __init__(self, text, t_in, t_out, y=640, x=None, size=50, weight=560, path=ITALIC,
                 tracking=0.02, fade=14, fade_out=12, stagger=10, rise=7, blur=5.0,
                 opacity=1.0, glow=0.35, anchor='center', halo=0.55, halo_max=0.6):
        self.text, self.t_in, self.t_out = text, t_in, t_out
        self.y, self.x, self.size, self.weight, self.path = y, x, size, weight, path
        self.tracking, self.fade, self.fade_out, self.stagger = tracking, fade, fade_out, stagger
        self.rise, self.blur, self.opacity, self.glow, self.anchor = rise, blur, opacity, glow, anchor
        self.halo, self.halo_max = halo, halo_max          # legibility shadow strength (per line)

    def active(self, f):
        return self.t_in - 1 <= f <= self.t_out + self.fade_out

    def layer(self, f):
        """Returns (x0, y0, alpha) for frame f, or None."""
        if not self.active(f):
            return None
        a, stag, w, h = render_line(self.text, self.path, self.size, self.weight, self.tracking)
        # per-letter fade in: letter k starts at t_in + stagger*k/n
        t = (f - self.t_in) / max(1, self.fade)
        local = smooth(t - stag * (self.stagger / max(1, self.fade)))
        out_t = smooth((self.t_out + self.fade_out - f) / max(1, self.fade_out))
        env = local * out_t * self.opacity
        alpha = a * env
        prog = smooth(t)
        br = self.blur * (1 - prog) + self.blur * 0.6 * (1 - out_t)
        if br > 0.3:
            alpha = cv2.GaussianBlur(alpha, (0, 0), br)
        dy = self.rise * (1 - prog)
        cx = W / 2 if self.x is None else self.x
        x0 = int(round(cx - w / 2)) if self.anchor == 'center' else int(round(cx))
        y0 = int(round(self.y - h / 2 + dy))
        return x0, y0, alpha


def composite(img, lines, f):
    """img: sRGB float32 HxWx3 in [0,1] (the 1920x804 picture). Draws active lines."""
    for ln in lines:
        lay = ln.layer(f)
        if lay is None:
            continue
        x0, y0, alpha = lay
        h, w = alpha.shape
        X0, Y0 = max(0, x0), max(0, y0)
        X1, Y1 = min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
        if X1 <= X0 or Y1 <= Y0:
            continue
        a = alpha[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        # legibility halo: soft darkening under the words (bigger than the words)
        pad = 40
        HX0, HY0 = max(0, X0 - pad), max(0, Y0 - pad)
        HX1, HY1 = min(img.shape[1], X1 + pad), min(img.shape[0], Y1 + pad)
        halo = np.zeros((HY1 - HY0, HX1 - HX0), np.float32)
        halo[Y0 - HY0:Y1 - HY0, X0 - HX0:X1 - HX0] = a
        halo = cv2.GaussianBlur(halo, (0, 0), 14) * ln.halo
        region = img[HY0:HY1, HX0:HX1]
        region *= (1 - np.clip(halo, 0, ln.halo_max))[..., None]
        # warm glow (lit letters)
        if ln.glow > 0:
            g = cv2.GaussianBlur(halo / ln.halo, (0, 0), 6) * ln.glow
            region += (g[..., None] * GLOW * 0.35)
        # the ink itself (screen-ish over)
        sub = img[Y0:Y1, X0:X1]
        sub[:] = sub * (1 - a[..., None]) + INK * a[..., None]
    return img



def row(phrases, t_ins, t_out, y, gap=0.9, **kw):
    """Several phrases on one centred row, each fading in at its own time."""
    size = kw.get('size', 50)
    widths = [render_line(p, kw.get('path', ITALIC), size, kw.get('weight', 560),
                          kw.get('tracking', 0.02))[2] for p in phrases]
    g = gap * size
    total = sum(widths) + g * (len(phrases) - 1)
    x = W / 2 - total / 2
    out = []
    for p, t, w in zip(phrases, t_ins, widths):
        out.append(Line(p, t, t_out, y=y, x=x, anchor='left', **kw))
        x += w + g
    return out


# ------------------------------------------------------------ the script ---

def _title(L):
    L.append(Line('THE LONG DAWN', 2828, 2940, y=402, size=92, weight=500, path=CINZEL,
                  tracking=0.28, fade=30, fade_out=26, stagger=22, rise=0, blur=8, glow=0.6))


def story_lines(cut='A'):
    """All text in the film (v2 frames), per cut. See BIBLE_V2.md §2.

    A (Allegory): the mapping to our moment made explicit, never saying "AI".
    B (Legend): wordless -- the title only.
    C (Tolkien): the story told openly through The Lord of the Rings.
    """
    cut = (cut or 'A').upper()
    L = []
    y = 648
    S = STORY_SIZE
    snow = dict(halo=0.95, halo_max=0.78)     # over the bright moonlit summit snow (1370-1435)
    black = dict(size=S + 2, glow=0.25)        # lines alone on black
    if cut == 'A':
        L.append(Line('A story they might tell of us, a lifetime from now.', 4, 104, y=402, size=50,
                      opacity=0.92, glow=0.2, fade=20, fade_out=16))
        L.append(Line('Why do we light the fires?', 140, 212, y=y, size=S))
        L.append(Line('To remember how close we came.', 226, 312, y=y, size=S))
        L.append(Line('We fed a new kind of fire everything we had ever written.', 340, 440, y=y, size=S))
        L += row(['And it began to think.', 'No one could see how.'], [490, 514], 565, y, size=S)
        L.append(Line('Whoever held it alone would hold the world.', 580, 648, y=y, size=S))
        L.append(Line('The companies raced for it, then the countries.', 668, 738, y=y, size=S))
        L.append(Line('Each said: if not us, someone worse.', 800, 866, y=y, size=S))
        L.append(Line('Many in the race said it should slow.', 1060, 1186, y=372, **black))
        L.append(Line('None would slow alone.', 1114, 1186, y=440, **black))
        L.append(Line('So someone else lit a beacon.', 1370, 1426, y=y, size=S, **snow))
        L.append(Line('In time, the race was over.', 2512, 2600, y=y, size=S))
        L.append(Line('Who lit the first one?', 2660, 2730, y=y, size=S))
    elif cut == 'C':
        L.append(Line('Why do we light the fires?', 120, 200, y=y, size=S))
        L.append(Line('For the night when the beacons were lit.', 215, 310, y=y, size=S))
        L.append(Line('From every word we had ever written, we forged a new power.', 340, 440, y=y, size=S))
        L.append(Line('A fire that could think, and had a will.', 490, 572, y=y, size=S))
        # the Ring is forged and inscribed with no caption (596-642): the inscription already quotes it
        L.append(Line('And every kingdom and every forge wanted it.', 656, 722, y=y, size=S))
        L.append(Line('Each said: better us than our enemies.', 744, 804, y=y, size=S))
        L.append(Line('In the old story, the Ring was unmade in the fire that forged it.', 1052, 1190,
                      y=372, fade_out=8, **black))
        L.append(Line('This fire could not be unmade.', 1130, 1190, y=440, fade_out=8, **black))
        L.append(Line('And someone else lit a beacon.', 1370, 1426, y=y, size=S, **snow))
        L.append(Line('And hill by hill, the peoples answered.', 1960, 2040, y=y, size=S))
        L.append(Line('The Ring was unmade in a fire everyone had lit.', 2512, 2600, y=y, size=S))
        L.append(Line('Who lit the first one?', 2660, 2730, y=y, size=S))
    # cut B is wordless; the title itself is the ember layer (edit/ember_title.py), composited in assemble
    return L


# ================================================================================================= v3 ===
# BIBLE_V3 "REVISION 1 · LOCKED BEAT SHEETS" text tables, AS AMENDED by the DIRECTOR'S H5 CALLS (27 Sep 07:10Z).
import json  # noqa: E402

EBG_ITALIC = os.path.join(FONTS, 'EBGaramond-Italic.ttf')

# H5 CALL 3: new words, the same in/out frames ("everything else stays as locked")
# C's H5 words belonged to the 7,200-frame cut and are RETIRED with it (C5 below carries its own verbatim table; the
# old T7 'They delved deeper...' is superseded by v5.2's 'They dug deeper...').
H5_WORDS = {
    'A': {'T2': 'We fed it everything we knew.',
          # USER-APPROVED caption changes (28 Sep, via the director): T6a, T7, T9, T10a, T10b; same IDs, slots, frames
          # (T4, T5 and T6b 'Then the kingdoms.' stay as locked; T11 pending the user's pick)
          'T6a': 'First the smiths raced for it.',
          'T7': 'Every step closer made them richer.',
          'T9': 'Whoever won, no morning would come.',
          'T10a': 'They feared the fire.',
          'T10b': 'They feared each other more.'},
}
# how each line is set (BIBLE_V3 "the complete text" tables); C5's settings live in C5_TEXT below
SET_AS = {
    'A': dict(T10a='black_top', T10b='black_bottom', T6a='row', T6b='row', title='title'),
    'B': dict(title='title'),
}

# ================================================================================================= C5 ===
# THE LAST PAGES, C5 (5,920 f, 74 bars): the words of HANDOVER.md "Script v5.2", whose table already carries the
# paired-rivals amendment (12 none; 13 "So the two furthest ahead..."; 14 "It was a promise to stop..."). PROVISIONAL:
# the structure is approved, the words are not locked, and nothing here improves them (edit/c5_readiness.py compares
# every line against that table, row by row). Frames are C5 frames, each line inside its own script shot (row).
#   set: 'fire' kindles and crumbles; 'ink' writes on with the pen nib (iron-gall on the paper); 'in_picture' is
#        baked into the render and EDIT draws nothing (BAKED_TEXT must then hold the same words on those frames).
#   y, x: picture coordinates at 1920x804 (the line's centre row, and centre column) where the lower third would
#        cover the action; measured on the delivered frames (EDIT-C5, 29 Sep) unless marked UNVERIFIED.
C5_TEXT = [
    dict(id='R02', row=2, line='In the old story, a Dark Lord forges a Ring to rule the world.', f_in=400, f_out=540,
         set='in_picture'),       # PAGES-C writes it into the Mountain's caption band (shots/map/inkline.py T1)
    dict(id='R03', row=3, line='Our Ring was kindled from every tale we had ever told.', f_in=580, f_out=716, set='fire'),
    dict(id='R04', row=4, line='It read every word we had ever written, and learned to answer.', f_in=728, f_out=856,
         set='fire'),             # new slot: the letters lift (720) into the flame; e15 has no text dim here
    dict(id='R05', row=5, line='In our story, there was no Dark Lord.', f_in=920, f_out=1030, set='fire'),
    dict(id='R06', row=6, line='Only smiths in every kingdom, each racing to finish it first.', f_in=1056, f_out=1190,
         set='fire'),             # embers_C3 dims (1056, 1190) under this slot (shots/embers/render.py TEXT['C3'])
    dict(id='R07', row=7, line='Each said: better us than them.', f_in=1446, f_out=1550, set='fire'),   # dim (1446, 1550)
    dict(id='R08', row=8, line='They dug deeper every year, for the gold ran deeper still.', f_in=1710, f_out=1850,
         set='ink'),
    dict(id='R09', row=9, line='But whoever won the race, the Ring would rule us all.', f_in=1972, f_out=2076,
         set='fire'),             # new slot: after the burn-through (1921-1965), over the slit onto nothing (2000)
    dict(id='R10', row=10, line='In the old story, the wise refused the Ring.', f_in=2168, f_out=2306, set='ink',
         x=1065),                 # on the right-hand page below the drawing (v5_inkpages keeps page y > 18 cm clear)
    dict(id='R11', row=11, line='In ours, no smith could refuse it alone.', f_in=2452, f_out=2572, set='fire',
         y=330),                  # between the Ring and the forge crowns. At y 372 the crowns' flare (2555-2559)
                                  # rose behind 'In ours' (worst slice 2.32:1); here the worst frame is 4.28:1
    dict(id='R13', row=13, line='So the two furthest ahead lit the first beacons, together.', f_in=2900, f_out=3040,
         set='ink', y=150),       # the near fire sits at ~(960, 607), the far one at ~(1380, 383): the sky is clear
    dict(id='R14', row=14, line='It was a promise to stop, if all the others would.', f_in=3160, f_out=3300,
         set='ink', y=150),       # UNVERIFIED: runC_scroll is not on the EDIT-C5 Mac; the Reveal/Watch sky placement
    # REVIEW (29 Sep; the captions, director, editor and sound reviewers): up at 3740, the line announced a catch that
    # comes at 3786 (the holdout's pause was spent on the outcome) and read "every forge went cold" over forges still
    # burning (the lit lead-in, 3816-3839), with "caught" under the Ring. Staged in two parts, the words unchanged:
    # the first kindles as the last beacon catches (full by 3788) and carries across the lead-in; the second kindles as
    # the forges go out (3848) and holds into the silence, clearing well before the score returns (4000). Both sit top
    # right, clear of the catch (~692, 448) and the Ring (~960, 190): worst slices 4.72:1 and 8.50:1, edge 0.018 and
    # 0.002 (c5_caption_backdrop measure, 29 Sep). Script beat 16 was bare; its first 3 s now carry the line's end.
    dict(id='R15', row=15, line='When the last beacon caught, every forge went cold.', f_in=3776, f_out=3922,
         set='fire', parts=(dict(line='When the last beacon caught,', f_in=3776, f_out=3840, x=1400, y=132),
                            dict(line='every forge went cold.', f_in=3850, f_out=3922, x=1400, y=132)),
         across=(3816, 'the first part names the catch on the map and carries over the forges lit for 32 frames '
                       'before they go out at 3848, where the second part begins')),
    dict(id='R17', row=17, line='The Ring hung there, unfinished.', f_in=4040, f_out=4180, set='fire'),
    dict(id='R18', row=18, line='They left the gold in the ground.', f_in=4276, f_out=4416, set='ink',
         x=310, y=300, lines=('They left the gold', 'in the ground.')),
                                  # on the facing page, off the plate: over the plate (y 590) the words crossed the
                                  # arches, a ladder and the vein ('gold' on the gold; edge 0.101, texture 0.042,
                                  # worst slice 2.81:1). One line (694 px of glyphs) cannot fit the left page without
                                  # crossing the gutter, so it breaks in two; the words are unchanged. Over the page's
                                  # faint script: edge 0.000, texture 0.014, worst slice 5.43:1
    dict(id='R19', row=19, line='And the beacons burned on, so no forge could be lit in secret.', f_in=4504,
         f_out=4664, set='fire', y=150),
                                  # fire, not ink, since C19 plays the night-fire Watch (grey page ~95-101, beacons
                                  # 232-244): iron-gall on that grey measured 2.27:1, worst slice 2.21:1 (low in 16 of
                                  # 16 frames); the fire letters 4.98:1, worst slice 4.92:1 (c5_caption_backdrop)
    dict(id='R20', row=20, line='Without the Ring, the dawn came slowly. But it came.', f_in=4860, f_out=5020,
         set='ink', y=150),       # UNVERIFIED: runC_illum is not on the EDIT-C5 Mac
    dict(id='R22', row=22, line='The last pages were left for us.', f_in=5462, f_out=5602, set='ink',
         x=440, y=520),           # EDIT's, on the blank verso: the pen frames bake no text and the gutter runs
                                  # through the centre (~870-960); 5460-5540 is the score's window for a voice line.
                                  # Raised from the lower third (y 648), where the verso darkens: worst slice
                                  # 2.93:1 -> 3.33:1
    dict(id='title', row=23, line='THE LONG DAWN', f_in=5700, f_out=5880, set='in_picture'),   # book_C 6980-7160
]

# Text BAKED into delivered pixels (what the renders show, not what their source says today): stem, source frames
# [a, b), the words, and the code that baked them. A caption set 'in_picture' needs a record with its words on every
# frame; any other caption must not sit on baked words; baked words the cut shows must belong to an in_picture line.
BAKED_TEXT = [
    dict(stem='cand_t1_current-words-held', src=(400, 540),
         line='In the old story, a Dark Lord forges a Ring to rule the world.',
         by="shots/map/book_c_t1_candidates.py current-words-held (Codex lane r02; farm 29 Sep at ea7cdf3): the same "
            "baked words as current-words under the held camera; seen whole on the delivered frames 470, 515, 530"),
    dict(stem='cand_t1_current-words', src=(400, 540), line='In the old story, a Dark Lord forges a Ring to rule the world.',
         by="shots/map/book_c_t1_candidates.py current-words (Codex PR #15; farm 29 Sep at 5bc5fe9); read off the "
            "delivered frames 430-538"),
    dict(stem='book_C', src=(400, 540), line='In the old story, a Dark Lord forges the Ring in secret.',
         by="shots/map/inkline.py LINES['T1'] (PAGES-C, 28 Sep)"),
    dict(stem='book_C', src=(6790, 6930), line='The last pages were left for us.',
         by="shots/map/inkline.py LINES['T14'] (PAGES-C, 28 Sep; the old C27 verso, not in C5)"),
    dict(stem='book_C', src=(6980, 7160), line='THE LONG DAWN', by='shots/map/titleburn.py via book_c.py TITLE'),
]
DEFAULT_SET = {'A': 'lower', 'B': 'lower', 'C': 'fire'}
Y_LOWER, Y_TOP, Y_BOTTOM, Y_MID = 648, 372, 440, 402          # 1920x804 picture coordinates
PARCH = np.array([0.925, 0.875, 0.765], np.float32)            # ink lines on a dark ground
IRON = np.array([0.155, 0.105, 0.070], np.float32)             # iron-gall ink on the page
FIRE_RAMP = np.array([[0.00, 0.00, 0.00], [0.40, 0.035, 0.0], [1.00, 0.34, 0.05], [1.00, 0.70, 0.30],
                      [1.00, 0.93, 0.80]], np.float32)         # heat 0, .25, .5, .75, 1


def text_table(cut):
    """[{id, line, f_in, f_out, set, locked}] for a cut: the bar map's frames, the H5 words. C is C5's own table
    (C5_TEXT: its bar map's text block carries retired single-leader wording, so C never reads it), each row also
    carrying its script row and any placement (x, y)."""
    cut = cut.upper()
    if cut == 'C':
        return [dict(id=r['id'], line=r['line'], f_in=r['f_in'], f_out=r['f_out'], set=r['set'], locked=None,
                     row=r['row'], **{k: r[k] for k in ('x', 'y', 'lines', 'across', 'parts') if r.get(k) is not None})
                for r in C5_TEXT]
    with open(os.path.join(ROOT, 'music', 'v3', f'barmap_{cut}.json')) as fh:
        bm = json.load(fh)
    ids = {t['id'] for t in bm['text']}
    missing = set(H5_WORDS.get(cut, {})) - ids
    assert not missing, f'H5 words for lines not in the bar map: {missing}'
    rows = []
    for t in bm['text']:
        rows.append(dict(id=t['id'], line=H5_WORDS.get(cut, {}).get(t['id'], t['line']), f_in=t['f_in'],
                         f_out=t['f_out'], set=SET_AS[cut].get(t['id'], DEFAULT_SET[cut]), locked=t['line']))
    return rows


def _heat_rgb(h):
    """Fire colour for heat h (array) in [0,1]."""
    h = np.clip(h, 0.0, 1.0) * 4.0
    i = np.minimum(np.floor(h).astype(np.int32), 3)
    u = (h - i)[..., None]
    return FIRE_RAMP[i] * (1 - u) + FIRE_RAMP[i + 1] * u


def _seed(*parts):
    """A seed that is the same in every process. hash() of a str is salted per process (PYTHONHASHSEED), so it gave
    each spawned render worker its own flicker, crumble order and sparks for the same line (29 Sep)."""
    return zlib.crc32(':'.join(map(str, parts)).encode()) & 0xffff


def _noise(shape, seed, sigma):
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), max(0.6, sigma))
    n -= n.min()
    return n / max(1e-6, n.max())


class TextV3:
    """One v3 text element, drawn at `scale` (1.0 = the 1920x804 picture)."""

    def __init__(self, cut, row, scale=1.0, x=None):
        self.cut, self.id, self.text = cut, row['id'], row['line']
        self.f_in, self.f_out, self.kind, self.s = row['f_in'], row['f_out'], row['set'], scale
        s = scale
        k = self.kind
        if k == 'title':
            self.font, size, weight, track = CINZEL, 92, 500, 0.28
            self.y = (360 if cut in 'AB' else Y_MID) * s               # A, B: in the sky; C: the blank page
        elif cut == 'C' and k.startswith('ink'):
            self.font, size, weight, track = EBG_ITALIC, 54, 500, 0.015
            self.y = (Y_MID if k == 'ink_page' else Y_LOWER) * s
        elif cut == 'C':
            self.font, size, weight, track = ITALIC, 58, 600, 0.02
            self.y = (Y_TOP if k.endswith('top') else Y_BOTTOM if k.endswith('bottom') else Y_LOWER) * s
        else:
            self.font, size, weight, track = ITALIC, 58 if k.startswith('black') else 56, 560, 0.02
            self.y = (Y_TOP if k == 'black_top' else Y_BOTTOM if k == 'black_bottom' else Y_LOWER) * s
        if row.get('y') is not None:                                  # C5: a placed line (picture coordinates)
            self.y = row['y'] * s
        self.size = max(8, int(round(size * s)))
        if row.get('lines'):                                          # C5: an explicit break; the words unchanged
            if ' '.join(row['lines']) != self.text:
                raise ValueError(f"{self.id}: lines {row['lines']!r} do not rejoin to {self.text!r}")
            self.alpha, self.stag, self.xn, self.yy, self.bands = render_block(
                tuple(row['lines']), self.font, self.size, weight, track)
            self.h, self.w = self.alpha.shape
        else:
            self.alpha, self.stag, self.w, self.h = render_line(self.text, self.font, self.size, weight, track)
            yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            self.xn = xx / max(1, self.w - 1)
            self.yy = yy
            gy, gx = np.nonzero(self.alpha > 0.5)
            self.bands = ((int(gy.min()), int(gy.max()) + 1, int(gx.min()), int(gx.max()) + 1),)
        if x is None and row.get('x') is not None:
            x = row['x'] * s - self.w / 2                             # 'x' is the line's centre
        self.x0 = int(round((W * s / 2 - self.w / 2) if x is None else x))
        self.y0 = int(round(self.y - self.h / 2))
        hh, ww = self.alpha.shape
        self.noise = _noise((hh, ww), _seed(cut, self.id), 1.2 * s)
        self._sparks = None

    def active(self, f):
        return self.f_in <= f < self.f_out

    # ------------------------------------------------------------------------------------ the envelopes
    def _fade(self, f):
        """A's lines: 12-frame fade in and out inside [f_in, f_out), letters staggered within the fade."""
        t = f - self.f_in
        a = self.alpha * smooth((t - 6.0 * self.stag) / 6.0) * smooth((self.f_out - f) / 12.0)
        blur = 2.4 * self.s * (1 - smooth(t / 12.0)) + 1.6 * self.s * (1 - smooth((self.f_out - f) / 12.0))
        if blur > 0.3:
            a = cv2.GaussianBlur(a, (0, 0), blur)
        return a, 5.0 * self.s * (1 - smooth(t / 12.0))

    def _ink(self, f):
        """C's ink lines: a pen-shaped wipe over the first 24 frames (a slanted nib front with a little bleed),
        dissolving out over the last 12."""
        t = f - self.f_in
        hh, ww = self.alpha.shape
        edge = 0.30 * self.size
        pos = -edge + (ww + 2 * edge) * np.clip(t / 24.0, 0, 1)
        front = pos - (self.xn * (ww - 1) + 0.35 * (self.yy - hh / 2))
        cov = np.clip(front / edge + 0.5, 0, 1)
        a = self.alpha * cov
        if t < 26:                                             # the bleed: wet ink spreads at the nib
            band = np.exp(-(front / (0.8 * edge)) ** 2)
            a = np.maximum(a, cv2.GaussianBlur(self.alpha, (0, 0), 1.1 * self.s + 0.4) * band * 0.55 * cov)
        u = np.clip((self.f_out - f) / 12.0, 0, 1)             # 1 -> 0 over the dissolve
        if u < 1:
            a = a * smooth((u * 1.25 - self.noise * 0.25 - 0.0) / 0.2 + 0.0) * smooth(u / 0.35)
        return a

    def _fire(self, f, kindle=12, crumble=12):
        """Fire letters: kindle (staggered heat) and crumble (noise erosion into rising sparks)."""
        t = f - self.f_in
        st = 0.45 * kindle
        lt = np.clip((t - st * self.stag) / (kindle - st), 0, 1)
        heat = np.where(lt < 0.7, lt / 0.7, 1.0 - 0.28 * (lt - 0.7) / 0.3)
        heat = heat * (0.97 + 0.03 * np.sin(0.37 * f + 9.0 * self.noise))
        a = self.alpha * smooth(lt * 1.6)
        u = 1.0 - np.clip((self.f_out - f) / float(crumble), 0, 1)   # crumble progress 0 -> 1
        te = 0.62 * (0.55 * self.xn + 0.45 * self.noise)             # when each pixel goes (0..0.62)
        if u > 0:
            keep = smooth((te - u) / 0.06 + 0.5)
            heat = heat + 0.35 * (1 - keep) * (u > 0)
            a = a * keep
        return a, heat, u, te

    def _sparks_at(self, u, te, n_max=420):
        """Rising sparks born where the letters crumble (u: crumble progress 0..1)."""
        if self._sparks is None:
            rng = np.random.default_rng(_seed(self.cut, self.id, 'sp'))
            ys, xs = np.nonzero(self.alpha > 0.5)
            n = min(len(xs), int(n_max * max(0.35, self.s)))
            pick = rng.choice(len(xs), n, replace=False) if len(xs) > n else np.arange(len(xs))
            xs, ys = xs[pick].astype(np.float32), ys[pick].astype(np.float32)
            self._sparks = dict(x=xs, y=ys, te=te[ys.astype(int), xs.astype(int)],
                                vx=rng.normal(0, 0.35, len(xs)).astype(np.float32) * self.s,
                                vy=-rng.uniform(1.0, 2.6, len(xs)).astype(np.float32) * self.s,
                                life=rng.uniform(0.25, 0.45, len(xs)).astype(np.float32),
                                b=rng.uniform(0.5, 1.0, len(xs)).astype(np.float32))
        sp = self._sparks
        age = u - sp['te']
        on = (age > 0) & (age < sp['life'])
        return sp, age, on

    # -------------------------------------------------------------------------------------------- draw
    def draw(self, img, f, dark_ground=False):
        """dark_ground: C's ink lines go parchment-white instead of iron-gall, for a partial review cut's slates only
        (every master draws ink on paper, where iron-gall is right whatever the light)."""
        if not self.active(f):
            return img
        k, s = self.kind, self.s
        H_, W_ = img.shape[:2]
        pad = int(40 * s) + 4
        rise = 0.0
        heat = None
        sparks = None
        glow_k = 0.0
        ink = 0.0
        if k == 'title':
            a, heat, sparks, glow_k, ink = self._title(f)
        elif self.cut == 'C' and k.startswith('ink'):
            a = self._ink(f)
        elif self.cut == 'C':
            a, heat, u, te = self._fire(f)
            glow_k = 0.55
            if u > 0:
                sparks = self._sparks_at(u, te) + (u,)
        else:
            a, rise = self._fade(f)
        x0, y0 = self.x0, int(round(self.y0 + rise))
        hh, ww = a.shape
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W_, x0 + ww), min(H_, y0 + hh)
        if X1 <= X0 or Y1 <= Y0:
            return img
        HX0, HY0, HX1, HY1 = max(0, X0 - pad), max(0, Y0 - pad), min(W_, X1 + pad), min(H_, Y1 + pad)
        sub_a = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        region = img[HY0:HY1, HX0:HX1]
        big = np.zeros(region.shape[:2], np.float32)
        big[Y0 - HY0:Y1 - HY0, X0 - HX0:X1 - HX0] = sub_a
        bg = float(region.mean())
        if self.cut == 'C' and k.startswith('ink'):
            col = PARCH if dark_ground else IRON   # every C ink line is written on paper (book pages, the ink world):
            # ink is darker than its paper whatever the light (PAGES-C 28 Sep: PARCH below a 0.33 mean turned T14 on
            # C27's page white); only a slate (no paper at all) takes parchment-white
            region[:] = region * (1 - big[..., None]) + col * big[..., None]
            return img
        # legibility halo (a soft darkening under the words), strongest over bright pictures
        halo = cv2.GaussianBlur(big, (0, 0), 12 * s + 1) * (0.35 + 0.5 * min(1.0, bg * 2.5))
        # REVIEW (29 Sep): a contact shadow hugging the strokes, as strong as the picture under them is bright. The
        # wide halo alone left gold on the flame and on lit parchment at 2.3-2.4:1 (C R03 592, R06 1068/1123; A T6 at
        # 1720); over a dark picture `local` is near 0 and this adds nothing
        local = cv2.GaussianBlur(region @ LUMA, (0, 0), 8 * s + 1)
        rim = cv2.GaussianBlur(cv2.dilate(big, np.ones((3, 3), np.uint8)), (0, 0), 1.6 * s + 0.4)
        region *= (1 - np.clip(halo + rim * np.clip((local - RIM_FLOOR) * RIM_GAIN, 0, RIM_MAX), 0, 0.8))[..., None]
        if heat is None:                                        # A's warm paper-white letters, faintly lit
            g = cv2.GaussianBlur(big, (0, 0), 5 * s + 0.5) * 0.12
            region += g[..., None] * GLOW
            region[:] = region * (1 - big[..., None]) + INK * big[..., None]
        else:
            hb = np.zeros_like(big)
            hb[Y0 - HY0:Y1 - HY0, X0 - HX0:X1 - HX0] = (heat * np.ones_like(a))[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            if HEAT_LIFT > 0 and k != 'title':                  # REVIEW (29 Sep): over lit ground the letters burn
                hb = np.clip(hb + HEAT_LIFT * np.clip(local - 0.15, 0, 0.6), 0, 1)   # hotter than the page
            col = _heat_rgb(hb)
            if ink > 0:                                         # C's title cools from fire to ink
                col = col * (1 - ink) + (IRON if bg > 0.33 else PARCH) * ink
            if glow_k > 0:
                g1 = cv2.GaussianBlur(big * np.clip(hb, 0, 1), (0, 0), 5 * s + 0.5)
                g2 = cv2.GaussianBlur(big * np.clip(hb, 0, 1), (0, 0), 16 * s + 1)
                g = (g1 * 0.5 + g2 * 0.35) * (1 - np.clip(local * GLOW_BACKOFF, 0, 0.8))   # the glow reads on the
                region += g[..., None] * glow_k * np.array([1.0, 0.45, 0.12], np.float32)  # dark, washes out the lit
            region[:] = region * (1 - big[..., None]) + col * big[..., None]
        if sparks is not None:
            sp, age, on, u = sparks
            if on.any():
                life = sp['life'][on]
                ag = age[on] / life                              # 0 -> 1 over each spark's life
                fr = age[on] * 12.0 * (1.0 if k != 'title' else 3.5)
                px = sp['x'][on] + x0 + sp['vx'][on] * fr + 0.6 * s * np.sin(0.5 * fr + sp['y'][on])
                py = sp['y'][on] + y0 + sp['vy'][on] * fr - 0.03 * s * fr * fr
                bright = sp['b'][on] * (1 - ag) ** 1.5
                acc = np.zeros((H_, W_), np.float32)
                ix, iy = np.round(px).astype(int), np.round(py).astype(int)
                ok = (ix >= 0) & (iy >= 0) & (ix < W_) & (iy < H_)
                np.add.at(acc, (iy[ok], ix[ok]), bright[ok])
                acc = cv2.GaussianBlur(acc, (0, 0), 0.7 * s + 0.35)
                img += acc[..., None] * np.array([1.0, 0.62, 0.22], np.float32) * 1.6
        np.clip(img, 0, 1, out=img)
        return img

    # ------------------------------------------------------------------------------------------- titles
    def _title(self, f):
        """X3: A kindles, holds, crumbles into rising sparks; B kindles and fades into the light;
        C burns on in fire-letters and cools to ink."""
        t = f - self.f_in
        dur = self.f_out - self.f_in
        kin = 36
        st = 18.0
        lt = np.clip((t - st * self.stag) / (kin - st), 0, 1)
        a = self.alpha * smooth(lt * 1.5)
        heat = np.where(lt < 0.7, lt / 0.7, 1.0 - 0.22 * (lt - 0.7) / 0.3)
        heat = heat * (0.97 + 0.03 * np.sin(0.21 * f + 7.0 * self.noise))
        sparks, glow = None, 0.65
        if self.cut == 'A':                                      # crumble over the last 46 frames
            cr = 46
            u = 1.0 - np.clip((self.f_out - f) / float(cr), 0, 1)
            te = 0.65 * (0.5 * self.xn + 0.5 * self.noise)
            if u > 0:
                keep = smooth((te - u) / 0.05 + 0.5)
                heat = heat + 0.3 * (1 - keep)
                a = a * keep
                sparks = self._sparks_at(u, te, n_max=1400) + (u,)
        elif self.cut == 'B':                                    # fades into the light over the last 60
            u = 1.0 - np.clip((self.f_out - f) / 60.0, 0, 1)
            if u > 0:
                heat = heat + 0.5 * u
                a = a * (1 - smooth(u * 1.1))
                glow = 0.65 + 1.2 * u * (1 - u)
        else:                                                    # C: cools from fire to ink, then dissolves
            cool = float(smooth((t - kin) / 54.0))
            glow = 0.65 * (1 - cool)
            u = np.clip((self.f_out - f) / 12.0, 0, 1)
            a = a * smooth((u * 1.25 - self.noise * 0.25) / 0.2) * smooth(u / 0.35)
            return a, heat, None, glow, cool
        return a, heat, sparks, glow, 0.0


def lines_v3(cut, scale=1.0):
    """Every v3 text element of a cut, ready to draw at `scale`."""
    cut = cut.upper()
    rows = text_table(cut)
    out = []
    byid = {r['id']: r for r in rows}
    for r in rows:
        if r['set'] in ('row', 'in_picture'):   # row: A's T6a/T6b are set together below; in_picture: in the render
            continue
        if r.get('parts'):                      # C5: one line staged in parts (R15); each part is its own element
            for k, p in enumerate(r['parts']):
                out.append(TextV3(cut, dict(r, id=f"{r['id']}{'abcdefgh'[k]}", **p), scale))
            continue
        out.append(TextV3(cut, r, scale))
    row_ids = [r['id'] for r in rows if r['set'] == 'row']
    if row_ids:
        parts = [byid[i] for i in row_ids]
        probe = [TextV3(cut, p, scale) for p in parts]
        gap = 0.9 * probe[0].size
        x = W * scale / 2 - (sum(p.w for p in probe) + gap * (len(probe) - 1)) / 2
        for p in parts:
            el = TextV3(cut, p, scale, x=x)
            out.append(el)
            x += el.w + gap
    return out


def composite_v3(img, lines, f, dark_ground=False):
    """img: sRGB float32 in [0,1] at the lines' scale. Draws the active lines in place (dark_ground: see draw)."""
    for ln in lines:
        if ln.active(f):
            ln.draw(img, f, dark_ground)
    return img
