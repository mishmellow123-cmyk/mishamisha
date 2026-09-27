"""THE RING'S INSCRIPTION (canonical for every lane: MONTAGE-3D, HEROINE, EMBERS, ACCORD-v3).

Our own invented script, the SCRIPT OF FIRE. It is not Tengwar and not the films' inscription, and it copies no
living script: there is no broad nib, no marks above the line, no dots, no headline and no Latin, Greek or
Arabic forms. The book, not the films, is its guide: "fine lines, finer than the finest pen-strokes ... lines
of fire that seemed to form the letters of a flowing script", outside AND inside the band.

Its logic, the one idea every letter follows: each letter is the OUTLINE OF A SMALL FLAME drawn in one fine
line (up one side from the base to the tip, down the other), heavier at the base and a hairline at the tip.
Letters differ the way flames do: height, lean, a tip bent back or forward into a small hook, two tips, a side
left open, a tail dropping below the line, a small flame at the foot of a large one. There is never a line
inside a flame (a lens with a slit would read as an eye). Within a word most letters are joined base to base by
a low hairline, so a word runs as one line of fire. The inscription is fixed (one sequence of letters, never
re-randomised) and means nothing: no real text in any language.

    python ring_script.py specimen out.png        # the alphabet + the two inscription lines, for review
    python ring_script.py publish                 # writes the canonical assets to the-long-dawn/assets/ring/

Assets (published): assets/ring/inscription_outer.png and inscription_inner.png, 16-bit grey, white letters on
black, u (columns) = once round the band, v (rows) = across the band's width; assets/ring/inscription.json holds the
mapping (which way u runs, where v=0 is, the proportions of the letters to the band).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ASSET_DIR = os.path.join(ROOT, 'assets', 'ring')

# ------------------------------------------------------------------------------------------ geometry ---
# Proportions of a line of script, in x-heights (em).
ASC = 1.85          # a tall lick's tip
DSC = -0.85         # a deep fall's point
SLANT = 0.10        # a slight forward lean
W_HAIR = 0.034      # the joining hairline
GAP = 0.20          # space between letters of a word (em)
WORD_GAP = 0.80     # space between words (em)


def _bez(p0, p1, p2, p3, n=48):
    u = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * p3


class Pen:
    """Draws tapered strokes (width per vertex) into a float canvas by stamping anti-aliased quads + discs.
    Coordinates are canvas pixels (x right, y down)."""

    def __init__(self, W, H):
        import cv2
        self.cv2 = cv2
        self.img = np.zeros((H, W), np.uint8)

    def stroke(self, P, widths):
        cv2 = self.cv2
        P = np.asarray(P, np.float64)
        wd = np.asarray(widths, np.float64)
        sh = 16.0
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            d = b - a
            L = math.hypot(d[0], d[1])
            if L < 1e-9:
                continue
            nx, ny = -d[1] / L, d[0] / L
            ra, rb = 0.5 * wd[i], 0.5 * wd[i + 1]
            quad = np.array([[a[0] + nx * ra, a[1] + ny * ra], [b[0] + nx * rb, b[1] + ny * rb],
                             [b[0] - nx * rb, b[1] - ny * rb], [a[0] - nx * ra, a[1] - ny * ra]])
            cv2.fillConvexPoly(self.img, np.round(quad * sh).astype(np.int32), 255, cv2.LINE_AA, 4)
            if ra > 0.35:
                cv2.circle(self.img, (int(round(a[0] * sh)), int(round(a[1] * sh))), int(round(ra * sh)), 255, -1,
                           cv2.LINE_AA, 4)


# ------------------------------------------------------------------------------------------ alphabet ---
# Every letter is the OUTLINE OF A SMALL FLAME drawn in one fine line: up one side from the base to the tip and
# down the other. Letters differ the way flames do: height, lean, a tip bent back or forward, two tips, a side
# left open, a tail that drops below the line, a small flame at the foot of a large one. There is never a line
# inside a flame (a lens with a slit would read as an eye). Within a word most letters are joined base to base by
# a low hairline, so the word runs as one line of fire.
PEN0, PEN1 = 0.095, 0.030           # stroke width at the base and at the tip (em)


def flame(x0, h=1.0, w=0.24, lean=0.12, wig=0.05, open_l=0.0, open_r=0.0, curl=0.0, tail=0.0, twin=0.0, n=90,
          belly=0.0, flick=0.0, spur=0.0, spur_at=0.62):
    """One flame outline with its left base at (x0, 0). Returns [(P, widths), ...] and its right base point.
    open_l / open_r: the fraction of that side (from the base) left undrawn. curl: the tip bends on into a small
    open hook (>0 back to the left, <0 forward). tail: the right side runs on below the line and sweeps back.
    twin: a second, lower tip beside the first (the outline dips to a notch between them)."""
    t = np.linspace(0.0, 1.0, n)
    half = w * np.sin(np.pi * t ** 0.55) * (1.0 - t) ** 1.45 / 0.43
    # the sides waver, each on its own phase, more toward the tip (a flame, never a droplet)
    ph = 7.3 * h + 11.0 * lean + 5.0 * w
    wav = 0.035 * np.clip(t / 0.35, 0, 1)
    # a flickering tip: the top third of the flame bends aside (flick > 0 to the right)
    c = x0 + w + lean * t ** 2 + wig * np.sin(2.0 * np.pi * t) * t + flick * np.clip((t - 0.62) / 0.38, 0, 1) ** 2
    # an uneven belly: one side fuller than the other (belly > 0: the right side)
    hl = half * (1.0 - belly * np.sin(np.pi * np.clip(t / 0.8, 0, 1))) + wav * np.sin(3.0 * np.pi * t + ph) * (1 - t)
    hr = half * (1.0 + belly * np.sin(np.pi * np.clip(t / 0.8, 0, 1))) + wav * np.sin(3.4 * np.pi * t + ph + 1.9) * (1 - t)
    hl, hr = np.maximum(hl, 0.0), np.maximum(hr, 0.0)
    # a spur: a small tongue licking off the right side at spur_at of the height (pointed, rising)
    if spur:
        tri = np.clip(1.0 - np.abs(t - spur_at) / 0.09, 0.0, 1.0) ** 1.6
        hr = hr + spur * tri
    left = np.stack([c - hl, h * t], 1)
    right = np.stack([c + hr, h * t + (0.35 * spur * h * np.clip(1.0 - np.abs(t - spur_at) / 0.09, 0.0, 1.0) ** 1.6
                                        if spur else 0.0)], 1)
    out = []
    if twin:
        # the right half becomes a second flame: from the notch up to a lower tip and down to the base
        h2 = h * (1.0 - 0.38 * twin)
        t2 = np.linspace(0.0, 1.0, n // 2)
        k = int(0.55 * n)
        notch = right[k]
        tip2 = np.array([c[-1] + 0.36 * w + lean * 0.4, h2])
        up = _bez(notch, notch + [0.05, 0.10 * h], tip2 + [-0.06, -0.22 * h2], tip2, n // 3)
        down = _bez(tip2, tip2 + [0.10, -0.20 * h2], right[int(0.25 * n)] + [0.26 * w + 0.1, 0.1], right[0] +
                    [0.28 * w, 0.0], n // 2)
        P = np.vstack([left[int(open_l * n):], right[k:][::-1], up[1:], down[1:]])
        end = P[-1]
    else:
        P = np.vstack([left[int(open_l * n):], right[int(open_r * n):][::-1]])
        end = right[0]
    if curl:
        # break the outline at the tip: the left side runs on past the tip into a small open hook
        L = left[int(open_l * n):]
        d = L[-1] - L[-6]
        a0 = math.atan2(d[1], d[0])
        r = 0.085 * abs(curl) * max(0.8, h)
        sg = 1.0 if curl > 0 else -1.0
        cc = L[-1] + r * np.array([math.cos(a0 + sg * math.pi / 2), math.sin(a0 + sg * math.pi / 2)])
        ang = a0 - sg * math.pi / 2 + sg * np.linspace(0.0, 1.55 * math.pi, 22)[1:]
        hook = np.stack([cc[0] + r * np.cos(ang), cc[1] + r * np.sin(ang)], 1)
        R = right[int(open_r * n):int(0.93 * n)][::-1]
        out.append((np.vstack([L, hook]), None))
        P = R
    if tail:
        e = P[-1]
        tl = _bez(e, e + [0.10, -0.30 * tail], e + [-0.02, -0.80 * tail], e + [-0.34, -0.82 * tail], 30)
        P = np.vstack([P, tl[1:]])
        end = None
    out.append((P, None))
    res = []
    for Q, _ in out:
        y = Q[:, 1]
        wd = PEN1 + (PEN0 - PEN1) * np.clip(1.0 - np.clip(y, 0.0, None) / 1.6, 0.0, 1.0) ** 1.4
        wd = np.where(y < 0.0, PEN0 * (1.0 + 0.6 * y).clip(0.25, 1.0), wd)
        res.append((Q, wd))
    return res, end


def hair(a, b, dip=0.06):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    P = _bez(a, (a[0] + 0.35 * (b[0] - a[0]), a[1] - dip), (b[0] - 0.35 * (b[0] - a[0]), b[1] - dip), b, 24)
    t = np.linspace(0.0, 1.0, len(P))
    return P, W_HAIR * (0.8 + 0.4 * np.sin(t * math.pi))


# name: (advance in em, [flame kwargs, ...] with x offsets, joins onward?)
ALPHABET = {
    # a small flame, full on the right, its tip flicking back to the left
    'ka': (0.58, [dict(dx=0.0, h=1.02, w=0.25, lean=0.08, wig=0.07, belly=0.35, flick=-0.22)], True),
    # a tall flame leaning into the wind in a long S, its tip thrown forward
    'ta': (0.64, [dict(dx=0.0, h=1.70, w=0.22, lean=0.34, wig=0.16, belly=-0.2, flick=0.14)], True),
    # a flame whose tip bends back over itself into a hook
    'ri': (0.60, [dict(dx=0.0, h=1.22, w=0.23, lean=0.16, wig=0.06, belly=-0.3, curl=1.0)], True),
    # a tall flame leaning back, a tongue licking off its right side
    'lo': (0.64, [dict(dx=0.0, h=1.55, w=0.21, lean=-0.16, wig=-0.08, belly=0.25, spur=0.13, spur_at=0.58,
                       flick=-0.08)], True),
    # a low flame lying in the wind
    'mu': (0.70, [dict(dx=0.0, h=0.76, w=0.28, lean=0.30, wig=0.08, belly=0.2, flick=0.22)], True),
    'se': (0.58, [dict(dx=0.0, h=1.02, w=0.24, lean=0.08, belly=0.2, flick=-0.08, tail=1.0)], False),
    'na': (0.80, [dict(dx=0.0, h=1.22, w=0.23, lean=0.10, belly=-0.15, flick=-0.06, twin=1.0)], True),
    'vo': (0.56, [dict(dx=0.0, h=1.08, w=0.24, lean=0.10, belly=0.25, flick=0.1, open_l=0.42)], False),
    'di': (0.58, [dict(dx=0.0, h=1.12, w=0.24, lean=0.14, belly=-0.25, flick=-0.1, open_r=0.48)], False),
    'he': (0.86, [dict(dx=0.0, h=1.30, w=0.22, lean=0.12, belly=0.2, flick=-0.1),
                  dict(dx=0.40, h=0.62, w=0.15, lean=0.16, flick=0.06)], True),
    'wa': (0.62, [dict(dx=0.0, h=0.92, w=0.24, lean=0.46, wig=0.10, belly=0.2, curl=-0.8)], True),
    'zu': (0.88, [dict(dx=0.0, h=0.66, w=0.16, lean=0.02, flick=0.08),
                  dict(dx=0.30, h=1.40, w=0.22, lean=0.16, belly=-0.2, curl=0.9)], True),
}
ORDER = ['ka', 'ta', 'ri', 'lo', 'mu', 'se', 'na', 'vo', 'di', 'he', 'wa', 'zu']


def glyph(name, x):
    """Strokes of one letter with its left base at x; returns (strokes, first base point, last base point or None)."""
    adv, parts, joins = ALPHABET[name]
    strokes, first, last = [], None, None
    for k in parts:
        kw = dict(k)
        dx = kw.pop('dx')
        st, end = flame(x + dx, **kw)
        if first is None:
            first = st[0][0][0]
        strokes += st
        last = end
    return strokes, first, (last if joins else None)


def words_fixed(seed, n_words, rng_words=(2, 6)):
    """The fixed sequence of letters. Letter frequencies follow a gentle Zipf so a few letters recur the way
    letters do in a real language (a random draw of equal letters reads as ornament, not writing)."""
    rng = np.random.default_rng(seed)
    k = len(ORDER)
    p = 1.0 / (np.arange(k) + 1.8)
    p = p / p.sum()
    order = list(rng.permutation(ORDER))
    words = []
    for _ in range(n_words):
        n = int(rng.integers(rng_words[0], rng_words[1] + 1))
        w = []
        for _ in range(n):
            g = order[int(rng.choice(k, p=p))]
            while w and g == w[-1] and g in ('ta', 'lo', 'na', 'he', 'zu', 'se', 'vo', 'di'):
                g = order[int(rng.choice(k, p=p))]
            w.append(g)
        words.append(w)
    return words


def layout(words, extra_gap=0.0):
    """Returns the strokes of a line [(P (n,2) em, widths)] and its length in em."""
    out = []
    x = 0.0
    for w in words:
        prev = None
        for g in w:
            st, first, last = glyph(g, x)
            if prev is not None and first is not None:
                out.append(hair(prev, first, 0.05))
            out += st
            prev = last
            x += ALPHABET[g][0] + GAP
        x += WORD_GAP - GAP + extra_gap
    return out, x


def draw_line(strokes, W, H, base_frac=0.64, em_frac=0.30, ss=3, wrap=True, x0_em=0.0):
    """Render strokes (em units) to a coverage image (H x W float32 0..1) at the letters' own aspect: 1 em =
    em_frac * H pixels both ways, the baseline at base_frac of the height (rows from the top). wrap: strokes
    crossing an edge also draw on the other side (a seamless band)."""
    import cv2
    Ws, Hs = W * ss, H * ss
    em = em_frac * Hs
    pen = Pen(Ws, Hs)
    base = base_frac * Hs
    for P, wd in strokes:
        u, v = P[:, 0] - x0_em, P[:, 1]
        X = (u + SLANT * v) * em
        Y = base - v * em
        for off in ((0.0, -float(Ws), float(Ws)) if wrap else (0.0,)):
            pen.stroke(np.stack([X + off, Y], 1), wd * em)
    return cv2.resize(pen.img.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)


def fitted_words(seed, length_em):
    """Words from the fixed sequence until the line fills length_em, with the slack shared out between the
    word gaps (never by stretching a letter)."""
    words = words_fixed(seed, 200)
    for n in range(1, len(words) + 1):
        _, L = layout(words[:n])
        if L > length_em:
            n -= 1
            break
    words = words[:max(n, 1)]
    _, L = layout(words)
    return words, (length_em - L) / len(words)


# Canonical inscription parameters. Outer face: one line round; inner face: its own line (the book: "outside and
# inside"). The x-height is 0.30 of the band's width; a tall flame reaches ~0.52 of it above the baseline.
OUTER = dict(seed=1937)
INNER = dict(seed=1954)
EM_FRAC, BASE_FRAC = 0.30, 0.66


def strip(W, H, face='outer'):
    """The canonical strip for one face at any resolution: coverage (H x W) in 0..1. u = column / W runs once
    round the band in the direction the script is read; v = row / H across the band width, row 0 at the band's
    +axis edge. W / H must be the face's circumference / band width, so the letters keep their shapes."""
    prm = OUTER if face == 'outer' else INNER
    L = W / (EM_FRAC * H)
    words, extra = fitted_words(prm['seed'], L)
    strokes, _ = layout(words, extra)
    return draw_line(strokes, W, H, BASE_FRAC, EM_FRAC)


# ------------------------------------------------------------------------------------------- outputs ---

def specimen(path, W=3400):
    """Review sheet: the alphabet large, then the outer and inner lines unrolled (each in three rows)."""
    import cv2
    rows = []
    strokes, x = [], 0.4
    for g in ORDER:
        st, _, _ = glyph(g, x)
        strokes += st
        x += ALPHABET[g][0] + 0.75
    em_px = W / x
    rows.append(draw_line(strokes, W, int(em_px * 3.1), base_frac=0.66, em_frac=1 / 3.1, wrap=False))
    for face in ('outer', 'inner'):
        Hs = 300
        Wf = int(round(Hs * 2 * math.pi * (0.0117 if face == 'outer' else 0.0094) / 0.0052))
        cov = strip(Wf, Hs, face)
        k = int(math.ceil(Wf / W))
        for i in range(k):
            part = cov[:, i * W:(i + 1) * W]
            if part.shape[1] < W:
                part = np.pad(part, ((0, 0), (0, W - part.shape[1])))
            rows.append(part)
    img = np.vstack([np.pad(r, ((10, 10), (0, 0))) for r in rows])
    cv2.imwrite(path, (255 - np.clip(img, 0, 1) * 255).astype(np.uint8))
    return path


def publish():
    import cv2
    os.makedirs(ASSET_DIR, exist_ok=True)
    R_IN, THICK, WIDTH = 0.0094, 0.0023, 0.0052          # MONTAGE-3D's band (m); any band scales the same way
    H = 640
    for face, r in (('outer', R_IN + THICK), ('inner', R_IN)):
        W = int(round(H * 2 * math.pi * r / WIDTH / 16.0)) * 16
        cov = strip(W, H, face)
        cv2.imwrite(os.path.join(ASSET_DIR, f'inscription_{face}.png'), (np.clip(cov, 0, 1) * 65535).astype(np.uint16))
        print(face, W, H)
    spec = dict(
        script='THE SCRIPT OF FIRE (invented; no real alphabet, no real text). Generator: '
               'shots/montage3d/ring_script.py (strip(W, H, face) at any resolution).',
        files=dict(outer='inscription_outer.png', inner='inscription_inner.png'),
        pixel='16-bit grey: 65535 = letter (full coverage), 0 = bare metal. Use as coverage/mask; the colour and '
              'strength of the glow are each lane\'s own (awake in fire: deep orange-red core, never white).',
        u='columns: once round the band, left to right = the direction the script is read; the strip wraps '
          'seamlessly at u=0/1. OUTER face: u = angle/360 counter-clockwise seen from the +axis end, so it reads '
          'left to right seen from outside with +axis up. INNER face: u = 1 - angle/360 (clockwise seen from '
          '+axis), so it too reads left to right seen from inside, never mirrored.',
        v='rows: across the band width; row 0 = the +axis edge (the top edge when the band stands axis-up). '
          'The letters are upright when row 0 is up.',
        proportions=dict(x_height_of_width=EM_FRAC, baseline_from_top=BASE_FRAC,
                         tallest_letter_above_baseline_of_width=round(1.72 * EM_FRAC, 3),
                         deepest_tail_below_baseline_of_width=round(0.82 * EM_FRAC, 3),
                         stroke_at_base_of_width=round(PEN0 * EM_FRAC, 4),
                         stroke_at_tip_of_width=round(PEN1 * EM_FRAC, 4)),
        aspect=dict(outer_w_over_h=round(2 * math.pi * (R_IN + THICK) / WIDTH, 4),
                    inner_w_over_h=round(2 * math.pi * R_IN / WIDTH, 4),
                    note='for another band, keep the strip height = band width and stretch u round the '
                         'circumference; the letters keep their shapes when W/H = circumference/width'),
    )
    json.dump(spec, open(os.path.join(ASSET_DIR, 'inscription.json'), 'w'), indent=1)
    specimen(os.path.join(ASSET_DIR, 'inscription_specimen.png'))


if __name__ == '__main__':
    if sys.argv[1] == 'specimen':
        print(specimen(sys.argv[2]))
    elif sys.argv[1] == 'publish':
        publish()
