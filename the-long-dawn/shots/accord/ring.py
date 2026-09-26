"""Variant C (the Tolkien cut): the Ring lying in the hearth fire at the centre of the stone table.

A plain, heavy gold band (mythic scale, ~0.37 m across) resting a little tilted on the coals.
Its outer and inner faces carry an inscription in an INVENTED flowing script: every glyph is
generated here from strokes of a broad-nib pen (stems with hooked heads, slanted loops, tails,
waves, and marks above the line) - no real alphabet and no real text. The letters kindle when the
merged fire takes the Ring (2001-2016), then breathe slowly; the white flare swallows it.

    python shots/accord/ring.py preview out.png      # the unrolled inscription, for review
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(ROOT, 'renders', 'accord_cache')

# geometry (metres)
R_OUT = 0.186                    # outer radius of the band
R_IN = 0.138                     # inner radius
BAND = 0.084                     # band width (along the ring's axis)
ROUND = 0.013                    # rounding of the band's edges
TILT = math.radians(27.0)        # it lies askew on the coals
TILT_AZ = math.radians(-35.0)    # direction of the downhill side
CENTRE = (0.035, -0.028)         # a little off the hearth's centre
Z_REST = 0.510                   # centre height (the bowl's ash surface is ~0.45-0.48 here)

# inscription strip: u runs once around the band, v across the band width
STRIP_MM = 0.40                  # mm per texel at the outer face
STRIP_H = int(round(BAND * 1000 / STRIP_MM))
STRIP_W = int(round(2 * math.pi * R_OUT * 1000 / STRIP_MM))
X_HEIGHT = 0.30                  # of the band width
SEED = 1914

RP_N = 32


# ------------------------------------------------------------------ script ---

class Pen:
    """Broad-nib pen on a supersampled canvas; coordinates in canvas px, nib angle fixed."""

    def __init__(self, img, nib_w, nib_h, angle=math.radians(38.0)):
        self.img = img
        self.w, self.h = nib_w, nib_h
        self.ca, self.sa = math.cos(angle), math.sin(angle)

    def _nib(self, x, y, scale=1.0):
        w, h = self.w * scale, self.h * scale
        pts = np.array([[-w, -h], [w, -h], [w, h], [-w, h]], np.float64)
        R = np.array([[self.ca, -self.sa], [self.sa, self.ca]])
        p = pts @ R.T + [x, y]
        cv2.fillConvexPoly(self.img, np.round(p * 16).astype(np.int32), 255, cv2.LINE_AA, 4)

    def stroke(self, pts, taper=(1.0, 1.0)):
        pts = np.asarray(pts, np.float64)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = seg.sum() + 1e-9
        acc = 0.0
        for k in range(len(seg)):
            n = max(1, int(seg[k] / (0.35 * self.h)))
            for j in range(n):
                a = j / n
                p = pts[k] * (1 - a) + pts[k + 1] * a
                u = (acc + seg[k] * a) / L
                sc = taper[0] + (taper[1] - taper[0]) * u
                sc *= min(1.0, 0.35 + 3.0 * min(u, 1 - u))       # stroke entry/exit thin out
                self._nib(p[0], p[1], sc)
            acc += seg[k]

    def dot(self, x, y, r):
        cv2.circle(self.img, (int(round(x * 16)), int(round(y * 16))), int(round(r * 16)), 255, -1, cv2.LINE_AA, 4)


class _NullPen:
    """Dry-run pen (layout pass): draws nothing."""

    def stroke(self, pts, taper=(1.0, 1.0)):
        pass

    def dot(self, x, y, r):
        pass


def _bez(p0, p1, p2, p3, n=40):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * np.asarray(p0) + 3 * (1 - u) ** 2 * u * np.asarray(p1) \
        + 3 * (1 - u) * u * u * np.asarray(p2) + u ** 3 * np.asarray(p3)


def _glyph(pen, rng, x, base, xh, slant):
    """Draw one invented glyph with its left edge at x; returns its advance (in x-heights).
    The vocabulary is deliberately un-Latin: arches whose legs curl outward, flame-like S stems with
    a coiled head, U-bowls with a rising tail, small coils, double bowls, and long under-sweeps."""
    def P(u, v):          # glyph space (u right, v up, in x-heights) -> canvas, with slant
        return (x + (u + slant * v) * xh, base - v * xh)

    B = _bez
    kind = rng.choice(['arch', 'flame', 'bowl', 'coil', 'twin', 'sweep', 'crescent'],
                      p=[0.20, 0.17, 0.17, 0.12, 0.12, 0.10, 0.12])
    if kind == 'arch':          # a wide arch; the left leg ends in a curl, the right sweeps out
        tall = rng.random() < 0.35
        top = 1.55 if tall else 1.0
        pen.stroke(B(P(0.05, 0.05), P(-0.02, 0.55 * top), P(0.10, top), P(0.45, top)))
        pen.stroke(B(P(0.45, top), P(0.80, top), P(0.92, 0.55), P(0.95, 0.1)))
        pen.stroke(B(P(0.95, 0.1), P(0.97, -0.12), P(1.18, -0.12), P(1.25, 0.10), 20), taper=(0.9, 0.4))
        pen.stroke(B(P(0.05, 0.05), P(0.08, -0.18), P(-0.18, -0.2), P(-0.2, 0.02), 20), taper=(0.9, 0.4))
        adv = 1.3
    elif kind == 'flame':       # a tall S stem with a coiled head
        pen.stroke(B(P(0.10, -0.05), P(0.55, 0.45), P(-0.05, 1.05), P(0.30, 1.55)))
        t = np.linspace(0, 1.6 * np.pi, 30)
        c = P(0.40, 1.52)
        r = 0.20 * xh * (1 - 0.35 * t / (1.6 * np.pi))
        pen.stroke(np.stack([c[0] - r * np.cos(t), c[1] - r * np.sin(t)], -1), taper=(1.0, 0.5))
        adv = 0.75
    elif kind == 'bowl':        # a U bowl with a tail rising high on the right
        pen.stroke(B(P(0.0, 0.95), P(-0.05, 0.1), P(0.65, -0.15), P(0.75, 0.55)))
        pen.stroke(B(P(0.75, 0.55), P(0.80, 1.0), P(0.95, 1.4), P(1.15, 1.55), 24), taper=(1.0, 0.35))
        adv = 1.05
    elif kind == 'coil':        # a small coil that unwinds into a high flick
        t = np.linspace(0, 1.6 * np.pi, 34)
        c = P(0.38, 0.42)
        r = 0.34 * xh * (1 - 0.5 * t / (1.6 * np.pi))
        pen.stroke(np.stack([c[0] - r * np.cos(t + 0.3), c[1] + r * np.sin(t + 0.3)], -1)[::-1], taper=(0.6, 1.0))
        pen.stroke(B(P(0.05, 0.30), P(0.0, 0.75), P(0.20, 1.20), P(0.55, 1.35), 24), taper=(1.0, 0.3))
        adv = 0.85
    elif kind == 'twin':        # two bowls opening upward, joined, with a flag
        for k in range(2):
            o = 0.55 * k
            pen.stroke(B(P(o, 0.8), P(o + 0.02, 0.0), P(o + 0.45, 0.0), P(o + 0.5, 0.75), 28))
        pen.stroke(B(P(1.05, 0.75), P(1.05, 1.1), P(1.2, 1.25), P(1.35, 1.1), 16), taper=(0.9, 0.4))
        adv = 1.35
    elif kind == 'sweep':       # a short stem that sweeps long under the line
        pen.stroke(B(P(0.25, 1.0), P(0.30, 0.5), P(0.35, 0.1), P(0.25, -0.35), 24))
        pen.stroke(B(P(0.25, -0.35), P(0.15, -0.7), P(0.9, -0.75), P(1.5, -0.35), 30), taper=(1.0, 0.3))
        adv = 0.75
    else:                       # crescent: a bow open to the left, hung from a stem with a flag
        pen.stroke(B(P(0.70, 1.45), P(0.72, 0.9), P(0.74, 0.4), P(0.70, -0.05), 26))
        pen.stroke(B(P(0.70, 1.0), P(0.05, 1.05), P(0.0, 0.05), P(0.62, 0.12), 30))
        pen.stroke(B(P(0.70, 1.45), P(0.85, 1.6), P(1.0, 1.5), P(1.02, 1.32), 12), taper=(0.8, 0.4))
        adv = 1.0
    # marks above the line (dense, as the script's vowels might be)
    m = rng.random()
    hi = 1.75 if kind in ('flame', 'bowl') else 1.3
    if m < 0.26:
        pen.dot(*P(0.45, hi), 0.10 * xh)
    elif m < 0.42:
        for k in range(3):
            pen.dot(*P(0.2 + 0.22 * k, hi + (0.12 if k == 1 else 0.0)), 0.07 * xh)
    elif m < 0.58:
        pen.stroke(B(P(0.1, hi), P(0.3, hi + 0.3), P(0.55, hi - 0.25), P(0.8, hi + 0.05), 20), taper=(0.6, 0.6))
    elif m < 0.68:
        pen.stroke(B(P(0.25, hi - 0.05), P(0.45, hi + 0.35), P(0.65, hi + 0.25), P(0.6, hi + 0.05), 16),
                   taper=(0.7, 0.4))
    return adv


def inscription(width, height, ss=3):
    """The ring's inscription as a coverage strip (height x width, float32 0..1): flowing words of
    invented glyphs joined along the line, fitted exactly once around the band."""
    W, H = width * ss, height * ss
    rng = np.random.default_rng(SEED)
    xh = X_HEIGHT * H
    base = 0.62 * H
    slant = 0.07
    # lay out words first (dry run on a scratch canvas) so they fit the circumference evenly
    words = []
    total = 0.0
    while True:
        n = int(rng.integers(3, 7))
        seed = int(rng.integers(1 << 30))
        pen = _NullPen()
        r2 = np.random.default_rng(seed)
        adv = sum(_glyph(pen, r2, -1e4, 0.0, 0.0, slant) + 0.12 for _ in range(n))
        if total + adv + 1.0 > W / xh - 1.0 and words:
            break
        words.append((n, seed, adv))
        total += adv
    gap = (W / xh - total) / len(words)
    img = np.zeros((H, W), np.uint8)
    pen = Pen(img, 0.115 * xh, 0.030 * xh)
    x = 0.5 * gap * xh
    for (n, seed, adv) in words:
        r2 = np.random.default_rng(seed)
        for g in range(n):
            a = _glyph(pen, r2, x, base, xh, slant)
            x0 = x
            x += (a + 0.12) * xh
            if g < n - 1 and r2.random() < 0.6:              # a joining hairline along the line
                pen.stroke(_bez((x0 + a * 0.85 * xh, base + 0.02 * xh), (x0 + (a + 0.05) * xh, base + 0.14 * xh),
                                (x - 0.2 * xh, base + 0.14 * xh), (x + 0.02 * xh, base - 0.02 * xh), 16),
                           taper=(0.45, 0.45))
        x += gap * xh
    cov = cv2.resize(img.astype(np.float32) / 255.0, (width, height), interpolation=cv2.INTER_AREA)
    return cov


def _mips(a, n=5):
    out = [a]
    for _ in range(n - 1):
        a = cv2.resize(a, (max(1, a.shape[1] // 2), max(1, a.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        out.append(a)
    return out


def load():
    """(flat texture, sizes[nl,2]) of the inscription mip chain (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f'ring_inscription_{SEED}_v2.npz')
    if os.path.exists(path):
        z = np.load(path)
        return z['flat'], z['sizes']
    cov = inscription(STRIP_W, STRIP_H)
    lv = _mips(cov)
    sizes = np.array([[l.shape[0], l.shape[1]] for l in lv], np.int64)
    flat = np.concatenate([l.reshape(-1) for l in lv]).astype(np.float32)
    np.savez(path, flat=flat, sizes=sizes)
    return flat, sizes


# ------------------------------------------------------------ per frame ---

def _rot():
    """World -> ring-local rotation (the ring's axis is local z)."""
    ax = np.array([math.cos(TILT_AZ + math.pi / 2), math.sin(TILT_AZ + math.pi / 2), 0.0])   # tilt axis (horizontal)
    c, s = math.cos(TILT), math.sin(TILT)
    x, y, z = ax
    Rw = np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                   [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                   [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])   # local -> world
    return Rw.T


def ring_params(t):
    """RP array for the shader (RP[0] = 0 when there is no Ring: variants A, B)."""
    import scene as SC
    RP = np.zeros(RP_N, np.float64)
    if SC.VARIANT != 'C':
        return RP
    RP[0] = 1.0
    RP[1], RP[2], RP[3] = CENTRE[0], CENTRE[1], Z_REST
    RP[4:13] = _rot().reshape(-1)
    RP[13] = 0.5 * (R_OUT + R_IN)
    RP[14] = 0.5 * (R_OUT - R_IN)
    RP[15] = 0.5 * BAND
    RP[16] = ROUND
    RP[17] = math.hypot(R_OUT, 0.5 * BAND) + 0.01          # bounding sphere radius
    RP[18] = 2 * math.pi * R_OUT                            # strip length (m)
    RP[19] = BAND
    return RP


def fire_hollow(t):
    """0..1: how far the hearth fire's hot core opens around the Ring (C only). The flare closes it."""
    import scene as SC
    return 1.0 - SC.smooth(SC.ramp(t, SC.FLARE_T0 - 2, SC.FLARE_T1 - 4))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'preview':
        cov = inscription(STRIP_W, STRIP_H)
        im = (255 - np.clip(cov, 0, 1) * 255).astype(np.uint8)
        # show the strip in 4 rows
        q = im.shape[1] // 4
        rows = [im[:, k * q:(k + 1) * q] for k in range(4)]
        cv2.imwrite(sys.argv[2], np.vstack([np.pad(r, ((6, 6), (0, 0)), constant_values=200) for r in rows]))
        print('strip', cov.shape)
