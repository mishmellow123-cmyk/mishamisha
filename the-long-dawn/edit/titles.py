"""Typography for THE LONG DAWN: story lines, the makers' plea, the title card.

Text is rendered once per line into a float alpha mask (PIL + raqm shaping,
variable-font weights, manual tracking that keeps kerning), then animated per
frame: staggered letter fade-in (a soft left-to-right breath), a small upward
settle, blur-to-sharp, and a slow fade out. Composited over the picture with a
soft dark halo for legibility and a faint warm glow so the words feel lit.
"""
import functools
import os

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

def story_lines():
    """All text in the film (global frames). See BIBLE.md §2."""
    L = []
    y = 648
    S = STORY_SIZE                     # story-line size (50 in the cloud cut; 56 reads on a phone)
    L.append(Line('Why do we light the fires?', 120, 200, y=y, size=S))
    L.append(Line('To remember the night the whole world answered.', 215, 310, y=y, size=S))
    L.append(Line('In the age of the Kindling, we made a new kind of fire.', 340, 440, y=y, size=S))
    L.append(Line('A fire that could think.', 490, 550, y=y, size=S))
    L.append(Line('Whoever held it alone would hold the world.', 565, 635, y=y, size=S))
    L.append(Line('So the kingdoms raced.', 660, 730, y=y, size=S))
    L.append(Line('Each said: if we stop, they win.', 820, 900, y=y, size=S))
    # the makers' plea, on black, centred
    L.append(Line('Then even its makers said:', 1055, 1190, y=350, size=48, opacity=0.8, glow=0.2))
    L += row(['Slow us down.', 'All of us.', 'Together.'], [1095, 1125, 1155], 1190, y=432,
             size=62, weight=600, stagger=6)
    # over bright moonlit summit snow: a deeper legibility shadow
    L.append(Line('And on a cold mountain, someone lit a beacon.', 1370, 1435, y=y, size=S,
                  halo=0.95, halo_max=0.78))
    L += row(['Rivals.', 'Strangers.', 'Enemies.'], [1522, 1542, 1562], 1600, y=y, stagger=5, size=S)
    L.append(Line('They answered anyway.', 1622, 1700, y=y, size=S))
    L.append(Line('They agreed on little — but they agreed on this:', 1922, 1998, y=y, size=S))
    L.append(Line('It did not end our differences.', 2270, 2345, y=y, size=S))
    L.append(Line('It ended the race.', 2352, 2440, y=y, weight=600, size=S))
    L.append(Line('Who lit the first one?', 2500, 2570, y=y, size=S))
    # title
    if TAGLINE:
        # title holds over the sky while it fades to black; then the last line stands
        # alone on black, in the silence after the final chord (the film runs to END_TAG)
        L.append(Line('THE LONG DAWN', 2662, 2750, y=402, size=92, weight=500, path=CINZEL,
                      tracking=0.28, fade=30, fade_out=24, stagger=22, rise=0, blur=8, glow=0.6))
        L.append(Line('The Kindling is now.', 2798, 2838, y=402, size=60, weight=520,
                      fade=22, fade_out=14, stagger=10, glow=0.25))
    else:
        L.append(Line('THE LONG DAWN', 2668, 2780, y=402, size=92, weight=500, path=CINZEL,
                      tracking=0.28, fade=30, fade_out=26, stagger=22, rise=0, blur=8, glow=0.6))
    return L
