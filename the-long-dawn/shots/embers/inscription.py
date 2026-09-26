"""Cut C: an invented flowing script for the Ring's inscription (ORIGINAL marks: no real text, not Tengwar, and
deliberately unlike any living script — no baseline-joined dotted letters, no headline, no Latin forms).

A "script of flames": letters are built from pointed ogee arches, tall flame-shaped loops that rise above the
x-height, descenders that sweep back to the left, small flame-ticks floating above; the letters of a word are
joined by a hairline running at MID-height. Drawn with a broad-nib pen (fixed nib angle) for thick/thin contrast,
slanted forward. The strip is sampled into points (like the glyph atlas) and wrapped round the Ring.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

X = 1.0          # x-height
ASC = 2.3        # ascender
DSC = -1.15      # descender
SLANT = 0.16     # a slight forward lean (a strong slant reads as Latin italic)

# Glyphs: lists of strokes; each stroke is a list of (x, y) knots (Catmull-Rom through them) + advance width.
# Every form is chosen to be unlike Latin, Greek, Cyrillic, Arabic, Hebrew, Indic scripts and Tengwar letters.
GLYPHS = {
    # a tall stem crowned by a crozier curl (a fern's head) — no letter of any living script
    'sa': dict(w=0.8, s=[[(0.3, -0.1), (0.32, 1.2), (0.36, 1.95), (0.58, 2.25), (0.72, 2.05), (0.58, 1.86),
                          (0.46, 1.98)]]),
    # a tight spiral whose tail rises into a curl above the x-height
    'sp': dict(w=1.0, s=[[(0.42, 0.45), (0.52, 0.55), (0.46, 0.66), (0.3, 0.6), (0.24, 0.38), (0.4, 0.14),
                          (0.64, 0.18), (0.76, 0.5), (0.8, 1.05), (0.95, 1.4), (1.05, 1.3)]]),
    # a flame standing on a short stem
    'fs': dict(w=0.7, s=[[(0.35, -0.05), (0.36, 0.55)],
                         [(0.36, 0.55), (0.12, 0.85), (0.14, 1.25), (0.36, 1.75), (0.56, 1.25), (0.58, 0.85),
                          (0.36, 0.55)]]),
    # a tilde-wave with a curled foot
    'tw': dict(w=1.1, s=[[(0.0, 0.45), (0.22, 0.78), (0.48, 0.6), (0.68, 0.32), (0.95, 0.48), (1.05, 0.72)],
                         [(0.5, 0.5), (0.46, 0.05), (0.62, -0.12), (0.75, 0.02)]]),
    # a moon-crescent with pointed horns and a mark in its lap
    'mc': dict(w=1.15, s=[[(0.0, 1.0), (0.12, 0.35), (0.55, 0.02), (0.98, 0.35), (1.1, 1.0)],
                          [(0.5, 0.42), (0.58, 0.5), (0.52, 0.6)]]),
    # a descending stem hooking back to the left, a small ring riding its shoulder
    'dh': dict(w=0.85, s=[[(0.4, 1.0), (0.42, 0.0), (0.36, -0.8), (0.12, -1.12), (-0.14, -0.98)],
                          [(0.6, 1.02), (0.78, 1.16), (0.86, 0.98), (0.72, 0.86), (0.6, 1.02)]]),
    # a looped cross
    'lx': dict(w=1.0, s=[[(0.0, 0.0), (0.4, 0.55), (0.62, 0.95), (0.5, 1.12), (0.36, 0.95), (0.6, 0.55),
                          (1.0, 0.0)],
                         [(0.18, 0.7), (0.5, 0.78), (0.82, 0.72)]]),
    # a tall stem hooking over to the left at the top, the foot curling right
    'hs': dict(w=0.75, s=[[(0.0, 1.65), (0.1, 2.15), (0.35, 2.3), (0.46, 2.0), (0.45, 0.6), (0.48, 0.05),
                           (0.62, -0.08), (0.78, 0.08)]]),
    # a fork whose arms curl outward
    'vf': dict(w=1.05, s=[[(0.5, -0.05), (0.5, 0.3), (0.3, 0.72), (0.08, 0.95), (0.0, 0.8), (0.1, 0.66)],
                          [(0.5, 0.3), (0.72, 0.72), (0.94, 0.95), (1.02, 0.8), (0.92, 0.66)]]),
    # a single tall flame tongue
    'ft': dict(w=0.85, s=[[(0.05, 0.1), (0.2, 0.5), (0.55, 0.95), (0.62, 1.45), (0.38, 1.95), (0.3, 1.55),
                           (0.46, 1.1), (0.3, 0.6), (0.45, 0.12), (0.85, 0.08)]]),
    # a pair of pointed almonds lying on their side, bridged
    'aa': dict(w=1.2, s=[[(0.0, 0.5), (0.25, 0.85), (0.5, 0.5), (0.25, 0.15), (0.0, 0.5)],
                         [(0.5, 0.5), (0.72, 0.62), (0.9, 0.5)],
                         [(0.9, 0.5), (1.05, 0.72), (1.2, 0.5), (1.05, 0.28), (0.9, 0.5)]]),
    # a deep loop below the line with a curl above
    'dl': dict(w=0.9, s=[[(0.1, 0.9), (0.45, 0.55), (0.5, -0.5), (0.3, -1.05), (0.12, -0.62), (0.5, -0.2),
                          (0.9, 0.1)]]),
}
MARKS = {
    'curl': [[(0.0, 0.0), (0.1, 0.12), (0.2, 0.04), (0.32, 0.14)]],
    'ring': [[(0.08, 0.0), (0.16, 0.06), (0.1, 0.15), (0.02, 0.08), (0.08, 0.0)]],
    'tri': [[(0.0, 0.0), (0.02, 0.03)], [(0.14, 0.0), (0.16, 0.03)], [(0.07, 0.12), (0.09, 0.15)]],
    'hook': [[(0.0, 0.0), (0.22, 0.02), (0.3, 0.14), (0.22, 0.2)]],
}
TALL = {'sa', 'fs', 'hs', 'ft', 'sp'}


def _catmull(pts, n=24):
    P = np.asarray(pts, np.float64)
    if len(P) < 2:
        return P
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        t2, t3 = t * t, t * t * t
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2][None, :])
    return np.vstack(out)


def text_sequence(rng, n_words):
    """random words of 2-5 invented letters (no meaning, no real text)"""
    keys = list(GLYPHS.keys())
    words = []
    for _ in range(n_words):
        k = int(rng.integers(2, 6))
        w = []
        prev = None
        for _ in range(k):
            g = keys[int(rng.integers(0, len(keys)))]
            while g == prev:
                g = keys[int(rng.integers(0, len(keys)))]
            w.append(g)
            prev = g
        words.append(w)
    return words


def render_strip(seed=7, n_words=11, em=160, pad=0.9, nib=0.16, thin=0.028, angle=math.radians(38), length=None):
    """Returns (mask float32 HxW in 0..1, em px, baseline y px). The strip is one line of the inscription.
    length: if given, words are added until the line is at least this long (em units)."""
    rng = np.random.default_rng(seed)
    words = text_sequence(rng, 200 if length else n_words)
    strokes, hair = [], []
    x = 0.4
    for wi, w in enumerate(words):
        prev_exit = None
        wstart = x
        for li, g in enumerate(w):
            G = GLYPHS[g]
            sx = 1.0 + rng.uniform(-0.06, 0.06)
            sy = 1.0 + rng.uniform(-0.05, 0.05)
            glyph = []
            for st in G['s']:
                glyph.append([(x + px * sx, py * sy) for px, py in st])
            strokes += glyph
            # letters stand apart; now and then a fine hairline flows from one into the next at mid-height
            ent = glyph[0][0]
            if prev_exit is not None and rng.random() < 0.55:
                hair.append([prev_exit, ((prev_exit[0] + ent[0]) / 2, 0.5 + rng.uniform(-0.05, 0.1)), ent])
            prev_exit = glyph[-1][-1]
            # a mark above some letters (never above the tall ones)
            if g not in TALL and rng.random() < 0.6:
                mk = list(MARKS.keys())[int(rng.integers(0, len(MARKS)))]
                mx = x + G['w'] * sx * rng.uniform(0.3, 0.6)
                my = 1.3 + rng.uniform(0.0, 0.2)
                for st in MARKS[mk]:
                    strokes.append([(mx + a * 1.4, my + b * 1.4) for a, b in st])
            x += G['w'] * sx + 0.22
        x += pad                                     # word gap
        if length and x >= length - 0.6:
            break
    W = int((x + 0.6) * em)
    H = int((ASC - DSC + 0.9) * em)
    base = int((ASC + 0.45) * em)
    ss = 3                                           # supersample
    img = Image.new('L', (W * ss, H * ss), 0)
    dr = ImageDraw.Draw(img)
    ca, sa = math.cos(angle), math.sin(angle)

    def stamp(path, width, thinw):
        P = _catmull(path, 28)
        P = P.copy()
        P[:, 0] = P[:, 0] + SLANT * P[:, 1]
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            # broad nib: a thin rectangle at a fixed angle swept along the path -> quad per segment
            nx, ny = ca * width / 2, sa * width / 2
            tx, ty = -sa * thinw / 2, ca * thinw / 2
            quad = []
            for (px, py), (qx, qy) in (((a[0] - nx - tx, a[1] - ny - ty), (a[0] + nx + tx, a[1] + ny + ty)),
                                       ((b[0] + nx + tx, b[1] + ny + ty), (b[0] - nx - tx, b[1] - ny - ty))):
                quad += [(px, py), (qx, qy)]
            poly = [((u) * em * ss, (base / em - v) * em * ss) for u, v in quad]
            dr.polygon(poly, fill=255)
    for st in strokes:
        stamp(st, nib, thin)
    for h in hair:
        stamp(h, nib * 0.18, thin * 0.8)
    img = img.resize((W, H), Image.LANCZOS)
    return np.asarray(img, np.float32) / 255.0, em, base


def strip_points(mask, n, rng, em, base):
    """sample n points on the inked area; returns (u in 0..1 along the strip, v in em units, y up from baseline)"""
    ys, xs = np.nonzero(mask > 0.35)
    w = mask[ys, xs]
    p = w / w.sum()
    i = rng.choice(len(xs), size=n, p=p)
    u = (xs[i] + rng.random(n)) / mask.shape[1]
    v = (base - (ys[i] + rng.random(n))) / em
    return u.astype(np.float64), v.astype(np.float64)


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'inscription.png'
    m, em, base = render_strip(seed=7)
    m2, _, _ = render_strip(seed=19)
    k = min(m.shape[1], 4200)
    img = np.vstack([m[:, :k], m2[:, :k]])
    Image.fromarray((img * 255).astype(np.uint8)).save(out)
    print(out, m.shape)
