"""C · THE RED BOOK shots (P1, P2, X1): the pages, the cameras, LETTERS TO FIRE, stills and frames.

    python redbook.py stills --out ../../review/v3/map_stills      # the risk-test stills (pencil and ink)
    python redbook.py x1 --frames 0,20,40,60,80,100,120,140 --out DIR   # LETTERS TO FIRE test frames
    python redbook.py frames <shot> --range A-B --out renders/book_C   # delivery frames (shot-local numbering)

Shots are timed in shot-local seconds (t = (frame - base) / 24), so the bar map can retime them without
renumbering. Every frame is a pure function of its frame number. The book layer is written premultiplied
(with the fire layer added on top) plus a matte, so EDIT comps it over the next shot: out = rgb + (1 - a) * next.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import look  # noqa: E402
import pen  # noqa: E402
import pages as PG  # noqa: E402
import book as B  # noqa: E402
import burn as BURN  # noqa: E402
import fire as FIRE  # noqa: E402
from pen import INK, GILT, PENCIL, Strokes, hand, line, catmull  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FPS = 24


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return np.asarray(a, np.float64) + (np.asarray(b, np.float64) - np.asarray(a, np.float64)) * t


# ======================================================= small ink drawings ===

def sketch_mountain(S, x, y, s, seed, fire=True):
    """A small marginal mountain (width ~2 s cm) with smoke, drawn quickly in the margin."""
    rng = np.random.default_rng(seed)
    L = np.array([[x - s, y], [x - 0.45 * s, y - 0.55 * s], [x - 0.12 * s, y - 1.05 * s]])
    R = np.array([[x + 0.12 * s, y - 1.05 * s], [x + 0.5 * s, y - 0.5 * s], [x + s, y]])
    line(S, L, 0.018, seed, lift=(5, 6))
    line(S, R, 0.022, seed + 1, lift=(5, 6))
    line(S, np.array([[x - 0.12 * s, y - 1.05 * s], [x, y - 0.98 * s], [x + 0.12 * s, y - 1.05 * s]]), 0.016, seed + 2)
    for q in np.linspace(0.1, 0.9, 9):
        p0 = R[0] + (R[-1] - R[0]) * q
        p1 = p0 + np.array([-0.18 * s * (1 - q), 0.35 * s * (1 - q) + 0.1 * s])
        pp, rr, dd = hand(np.array([p0, p1]), 0.008, int(rng.integers(1 << 30)), thin_end=0.2)
        S.add(pp, rr, dd)
    if fire:
        for m in range(3):
            c = np.array([[x + 0.02 * s, y - 1.12 * s], [x + (0.1 + 0.25 * m) * s, y - (1.5 + 0.18 * m) * s],
                          [x + (0.35 + 0.4 * m) * s, y - (1.6 + 0.1 * m) * s]])
            line(S, catmull(c, 8), 0.012, seed + 10 + m, lift=(5, 6))
    line(S, np.array([[x - 1.3 * s, y + 0.02], [x + 1.4 * s, y + 0.01]]), 0.012, seed + 20, lift=(5, 6))


def sketch_ring(S, x, y, r, seed, gilt=False):
    a = np.linspace(0, 2 * np.pi, 70)
    for sc in (1.0, 0.72):
        P = np.column_stack([x + r * sc * np.cos(a), y + 0.8 * r * sc * np.sin(a)])
        pp, rr, dd = hand(P, 0.012, seed + int(sc * 10), taper=(0, 0), thin_end=0.9, slow=(1.0, 0.003))
        S.add(pp, rr, dd)
    if gilt:
        P = np.column_stack([x + 0.86 * r * np.cos(a), y + 0.8 * 0.86 * r * np.sin(a)])
        S.add(P, np.full(len(P), 0.13 * r), np.ones(len(P)), layer=GILT)


def sketch_ship(S, x, y, s, seed):
    """A small grey ship on a low sea: a long hull with a swan-high prow, one mast, a sail, a few waves."""
    rng = np.random.default_rng(seed)
    hull = catmull([(x - 1.3 * s, y - 0.25 * s), (x - 0.9 * s, y + 0.12 * s), (x, y + 0.2 * s), (x + 0.9 * s, y + 0.1 * s),
                    (x + 1.25 * s, y - 0.3 * s), (x + 1.35 * s, y - 0.75 * s), (x + 1.18 * s, y - 0.85 * s)], 8)
    line(S, hull, 0.02, seed, lift=(6, 7))
    line(S, np.array([[x - 1.3 * s, y - 0.25 * s], [x + 1.28 * s, y - 0.3 * s]]), 0.014, seed + 1, lift=(6, 7))
    mast = np.array([[x - 0.05 * s, y - 0.25 * s], [x - 0.02 * s, y - 1.8 * s]])
    line(S, mast, 0.016, seed + 2)
    sail = catmull([(x - 0.02 * s, y - 1.7 * s), (x + 0.55 * s, y - 1.2 * s), (x + 0.62 * s, y - 0.55 * s),
                    (x - 0.02 * s, y - 0.45 * s)], 8)
    line(S, sail, 0.014, seed + 3, lift=(6, 7))
    for q in np.linspace(0.15, 0.85, 6):
        p0 = np.array([x - 0.02 * s, y - (1.7 - 1.2 * q) * s])
        p1 = p0 + np.array([0.5 * s * math.sin(q * math.pi) + 0.05 * s, 0.02 * s])
        pp, rr, dd = hand(np.array([p0, p1]), 0.007, int(rng.integers(1 << 30)), thin_end=0.25)
        S.add(pp, rr, dd)
    for m in range(4):
        yy = y + (0.35 + 0.18 * m) * s
        xx = x - 1.6 * s + 0.3 * m * s
        wv = np.array([[xx + 0.4 * k * s, yy + 0.05 * s * math.sin(k * 2.1)] for k in range(8 - m)])
        line(S, catmull(wv, 6), 0.009, seed + 10 + m, lift=(4, 6))


# =================================================================== pages ===

class Page:
    """Strokes of one page and its texture at a time t. `burning` strokes (the text of X1a) get per-stroke
    gains for ink, fire and ghost from a LettersToFire."""

    def __init__(self, S, ppc, seed=0):
        self.S = S
        self.pk = S.pack()
        self.ppc = ppc
        self.tex = B.PageTex(PG.PW, PG.PH, ppc)

    def texture(self, t, gains=None):
        tx = self.tex
        tx.chan[:] = 0
        pk = self.pk
        H, W, ppc = tx.H, tx.W, self.ppc
        g_ink = None if gains is None else gains[0]
        C, Wt = pen.raster(pk, t, ppc, H, W, INK, dry=0.9, gain=g_ink)
        tx.chan[..., 0] = C
        tx.chan[..., 1] = Wt
        C, _ = pen.raster(pk, t, ppc, H, W, GILT, gain=g_ink)
        tx.chan[..., 2] = np.clip(C, 0, 1)
        C, _ = pen.raster(pk, t, ppc, H, W, PENCIL)
        tx.chan[..., 3] = C
        if gains is not None:
            C, _ = pen.raster(pk, 1e9, ppc, H, W, INK, gain=gains[1])
            tx.chan[..., 4] = C
            C, _ = pen.raster(pk, 1e9, ppc, H, W, INK, gain=gains[2])
            tx.chan[..., 5] = np.clip(C, 0, 1)
        return tx.build()


def initial(S, x, y, size, seed, glyph='lp'):
    """The book's one illuminated initial: a letter of our script in gold leaf on a small ruled square, with
    fine penwork tendrils curling down the margin (page cm; (x, y) is the square's top-left)."""
    rng = np.random.default_rng(seed)
    for inset, w in ((0.0, 0.02), (0.07, 0.009)):
        a, b, c, d = x + inset, y + inset, x + size - inset, y + size - inset
        for (p0, p1) in (((a - 0.03, b), (c + 0.03, b)), ((c, b - 0.03), (c, d + 0.03)), ((c + 0.03, d), (a - 0.03, d)),
                         ((a, d + 0.03), (a, b - 0.03))):
            pp, rr, dd = hand(np.array([p0, p1]), w, int(rng.integers(1 << 30)), slow=(5.0, 0.004), taper=(0.02, 0.02),
                              thin_end=0.6)
            S.add(pp, rr, dd)
    # the letter, broad-nibbed in gold, and a fine ink spine that makes it read against the gilt
    G = pen.GLYPHS[glyph]
    ys = [py for st in G['s'] for _, py in pen._knots(st)]
    em = min((size - 0.42) / G['w'], (size - 0.42) / max(max(ys) - min(ys), 0.5))
    hx = x + 0.5 * size - 0.5 * G['w'] * em
    base = y + 0.5 * size + 0.5 * (max(ys) + min(ys)) * em          # the letter centred in its square
    hg = pen.Hand(seed=seed + 1, xh=em, nib=0.55, thin=0.08, layer=GILT)
    hg.write_word(S, [glyph], hx, base)
    hi = pen.Hand(seed=seed + 1, xh=em, nib=0.05, thin=0.02, layer=INK)
    hi.write_word(S, [glyph], hx, base)
    # penwork: little spirals from the corners and a tendril down the outer margin
    for (cx, cy, r0) in ((x - 0.05, y + size + 0.1, 0.16), (x + size + 0.08, y - 0.02, 0.12), (x - 0.12, y + 0.35, 0.1)):
        a = np.linspace(0, 4.4, 40) + rng.uniform(0, 6)
        rr_ = r0 * np.linspace(1.0, 0.2, 40)
        pp, rr, dd = hand(np.column_stack([cx + rr_ * np.cos(a), cy + rr_ * np.sin(a)]), 0.008, int(rng.integers(1 << 30)),
                          thin_end=0.3, taper=(0.05, 0.2))
        S.add(pp, rr, dd)
    ten = catmull([(x - 0.08, y + size + 0.25), (x - 0.25, y + size + 0.9), (x - 0.12, y + size + 1.6), (x - 0.3, y + size + 2.4)], 10)
    pp, rr, dd = hand(ten, 0.009, int(rng.integers(1 << 30)), thin_end=0.2, taper=(0.1, 0.5))
    S.add(pp, rr, dd)
    for q in range(4):
        i = int(len(ten) * (0.2 + 0.2 * q))
        c0 = ten[i]
        a = np.linspace(0, 3.6, 24) + (0 if q % 2 else np.pi)
        rr_ = 0.12 * np.linspace(1.0, 0.25, 24)
        sg = 1 if q % 2 else -1
        pp, rr, dd = hand(np.column_stack([c0[0] + sg * rr_ * np.cos(a) + sg * 0.12, c0[1] + rr_ * np.sin(a)]), 0.007,
                          int(rng.integers(1 << 30)), thin_end=0.3)
        S.add(pp, rr, dd)
    # a single gilt dot at each tendril's end, as the penman finishes
    for (dx, dy) in ((x - 0.3, y + size + 2.4), (x + size + 0.08, y - 0.02)):
        S.add(np.array([[dx, dy], [dx + 0.001, dy]]), np.array([0.035, 0.035]), np.array([1.0, 1.0]), layer=GILT)


def leaves_last(seed=5):
    """P1: the last written spread. Left: a full page with a mountain sketched in the margin and a small ring
    by a paragraph; right: the tale ends two-thirds down, a small ship below it, then nothing."""
    L, R = Strokes(), Strokes()
    box = (3.0, 2.9, 16.8, 26.4)          # verso: the wider margin is the outer (left) one

    def skipL(li, xa, xb):
        if 3 <= li <= 8:
            return [(xa - 1, xa + 2.9)]
        if 20 <= li <= 22:
            return [(xb - 1.7, xb + 1)]
        return []
    PG.text_page(L, seed, lines=38, box=box, skip=skipL)
    sketch_mountain(L, box[0] + 1.25, 2.9 + 0.62 * 8.6, 0.85, seed + 100)
    sketch_ring(L, box[2] - 0.8, 2.9 + 0.62 * 21.3, 0.42, seed + 200)
    boxR = (3.2, 2.9, 17.0, 26.4)

    def skipR(li, xa, xb):
        if li == 14:                            # the last chapter begins: a blank line, then the gilt initial
            return [(xa - 1, xb + 1)]
        if 15 <= li <= 17:
            return [(xa - 1, xa + 1.95)]
        return []
    recs = PG.text_page(R, seed + 1, lines=21, box=boxR, last_frac=0.42, skip=skipR)
    initial(R, boxR[0] + 0.02, 2.9 + 0.62 * 14.45, 1.72, seed + 50, glyph='lp')
    yl = recs[-1]['base']
    sketch_ship(R, 10.1, yl + 2.6, 0.75, seed + 300)
    # the tale's closing mark: a small flourish under the last line
    fl = catmull([(8.6, yl + 0.55), (9.5, yl + 0.42), (10.2, yl + 0.62), (10.9, yl + 0.45), (11.6, yl + 0.55)], 8)
    line(R, fl, 0.014, seed + 400, lift=(8, 9))
    return L, R, yl


def mountain_leaves(seed=11, mode='ink', t0=0.0, t1=8.0):
    """P2 THE MOUNTAIN: an early recto with the plate, the caption band left clear (T1), and the tale below;
    the verso opposite is text. Returns (L, R, words_R) where words_R are the recto's words (for X1a)."""
    mt = PG.Mountain(seed)
    R = mt.build(mode, t0, t1)
    h = pen.Hand(seed=seed + 7, xh=0.2)
    k0 = len(R)
    h.write_block(R, 3.2, 15.05, 13.8, 18, 0.62, last_frac=None)
    words = [(a + 0, b + 0, xl, xr, base, li) for (a, b, xl, xr, base, li) in h.words]
    L = Strokes()
    PG.text_page(L, seed + 3, lines=38, box=(3.0, 2.9, 16.8, 26.4))
    return L, R, words, mt


# ================================================================== x1: fire ===

class LettersToFire:
    """X1a: the letters glow in reading order (accelerating), lift off as sparks that stream into a flame
    standing on the page; the flame's heat browns the paper and burns it open onto the next shot."""

    def __init__(self, book, words, F, side='R', t_glow=(0.2, 2.6), seed=3, spp=4):
        self.book, self.words, self.F, self.side = book, words, np.asarray(F, np.float64), side
        rng = np.random.default_rng(seed)
        n = len(words)
        order = np.argsort([w[5] * 100 + w[2] for w in words])       # reading order: line, then x
        rank = np.empty(n)
        rank[order] = np.arange(n)
        # kindle times accelerate through the text
        self.tk = t_glow[0] + (t_glow[1] - t_glow[0]) * (rank / max(n - 1, 1)) ** 0.62
        self.tk += rng.uniform(-0.04, 0.04, n)
        self.glow_d = rng.uniform(0.35, 0.6, n)                       # time glowing before it lifts
        self.lift = self.tk + self.glow_d
        # sparks sampled on each word's strokes
        self.sp = []
        for wi, (a, b, xl, xr, base, li) in enumerate(words):
            m = max(3, int(spp * (xr - xl) / 0.9))
            u = rng.uniform(xl, xr, m)
            v = base - rng.uniform(0.0, 0.3, m)
            for k in range(m):
                self.sp.append((wi, u[k], v[k]))
        self.sp = np.array(self.sp)
        N = len(self.sp)
        self.sp_t0 = self.lift[self.sp[:, 0].astype(int)] + rng.uniform(0.0, 0.3, N)
        self.sp_d = rng.uniform(0.9, 1.8, N)
        P0 = book.page_to_world(side, self.sp[:, 1], self.sp[:, 2])
        Fw = book.page_to_world(side, np.array([self.F[0]]), np.array([self.F[1]]))[0]
        self.Fw = Fw
        self.P0 = P0 + np.array([0, 0, 0.02])
        up = rng.uniform(0.6, 2.2, N)
        lat = rng.normal(0, 0.8, (N, 2))
        self.P1 = self.P0 + np.column_stack([lat * 0.6, up])
        ang = rng.uniform(0, 2 * np.pi, N)
        rad = rng.uniform(0.8, 2.6, N)
        self.P2 = Fw + np.column_stack([rad * np.cos(ang), rad * np.sin(ang), rng.uniform(1.5, 3.4, N)])
        self.P3 = Fw + np.column_stack([rng.normal(0, 0.15, (N, 2)), rng.uniform(0.2, 1.4, N)])
        self.seed = seed
        self.heat = rng.uniform(0.6, 1.0, N)

    def gains(self, t, nstrokes):
        """Per-stroke gains (ink, fire, ghost) at time t for a page with nstrokes strokes."""
        gi = np.ones(nstrokes)
        gf = np.zeros(nstrokes)
        gg = np.zeros(nstrokes)
        for wi, (a, b, xl, xr, base, li) in enumerate(self.words):
            tk, tl = self.tk[wi], self.lift[wi]
            if t < tk:
                continue
            # the glow rises in the ink, flickers, and flares as the letter lifts
            e = smooth((t - tk) / 0.25) * (1.0 - smooth((t - tl) / 0.2))
            fl = 1.0 + 0.25 * math.sin(t * 23.0 + wi * 1.7) + 0.15 * math.sin(t * 41.0 + wi)
            gf[a:b] = 0.55 * e * fl + 0.9 * math.exp(-((t - tl) / 0.08) ** 2)
            gi[a:b] = 1.0 - smooth((t - tl + 0.05) / 0.2)
            gg[a:b] = 0.4 * smooth((t - tl) / 0.3)
        return gi, gf, gg

    def sparks(self, t, dt=0.018):
        """World positions of the live sparks now and a shutter earlier, their brightness."""
        tau = (t - self.sp_t0) / self.sp_d
        live = (tau > 0) & (tau < 1)
        if not live.any():
            z = np.zeros((0, 3))
            return z, z, np.zeros(0), np.zeros(0)

        def at(tt):
            s = np.clip(tt, 0, 1)[:, None]
            a = s ** 1.25
            b = 1 - a
            P = (b ** 3) * self.P0 + 3 * (b ** 2) * a * self.P1 + 3 * b * (a ** 2) * self.P2 + (a ** 3) * self.P3
            # curl: a little swirl that grows then settles
            w = np.sin(np.pi * s) * 0.35
            ph = self.sp_d[:, None] * 7.0
            P = P + w * np.column_stack([np.sin(tt * 9.0 + ph[:, 0]), np.cos(tt * 8.0 + ph[:, 0] * 1.3), 0 * tt])
            return P
        Pn = at(tau)[live]
        Pp = at(tau - dt / self.sp_d)[live]
        tl = tau[live]
        br = self.heat[live] * np.clip(tl / 0.08, 0, 1) * (1.0 - smooth_arr((tl - 0.8) / 0.2))
        temp = 0.78 - 0.3 * tl
        return Pn, Pp, br, temp

    def flame_h(self, t):
        """The flame's height (cm): fed by the arriving sparks."""
        arr = np.sum(t > self.sp_t0 + 0.85 * self.sp_d) / max(len(self.sp_t0), 1)
        k = smooth((t - 1.0) / 0.6)
        return (0.3 + 3.2 * arr ** 0.6) * k


def smooth_arr(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


@njit(cache=True, parallel=True)
def streaks(img, x0, y0, x1, y1, r, g, b, sig):
    """Additive anti-aliased streaks (a spark's travel during the shutter), Gaussian across, per row band."""
    H, W = img.shape[0], img.shape[1]
    nb = 64
    band = (H + nb - 1) // nb
    for bnd in prange(nb):
        r0 = bnd * band
        r1 = min(H, r0 + band)
        for k in range(len(x0)):
            s = sig[k]
            m = 3.0 * s + 1.0
            if max(y0[k], y1[k]) + m < r0 or min(y0[k], y1[k]) - m > r1:
                continue
            xa = int(max(0, math.floor(min(x0[k], x1[k]) - m)))
            xb = int(min(W - 1, math.ceil(max(x0[k], x1[k]) + m)))
            ya = int(max(r0, math.floor(min(y0[k], y1[k]) - m)))
            yb = int(min(r1 - 1, math.ceil(max(y0[k], y1[k]) + m)))
            dx = x1[k] - x0[k]
            dy = y1[k] - y0[k]
            L2 = dx * dx + dy * dy
            L = math.sqrt(L2) + 1e-6
            norm = 1.0 / (1.0 + L / (2.5 * s))          # energy spread along a longer streak
            for i in range(ya, yb + 1):
                for j in range(xa, xb + 1):
                    px = j + 0.5 - x0[k]
                    py = i + 0.5 - y0[k]
                    t = 0.0
                    if L2 > 1e-9:
                        t = min(max((px * dx + py * dy) / L2, 0.0), 1.0)
                    qx = px - t * dx
                    qy = py - t * dy
                    w = math.exp(-(qx * qx + qy * qy) / (2 * s * s)) * norm * (0.4 + 0.6 * t)
                    img[i, j, 0] += w * r[k]
                    img[i, j, 1] += w * g[k]
                    img[i, j, 2] += w * b[k]


def fire_layer(cam, x1, t, W, H):
    """Sparks and the flame as an additive HDR layer, plus the flame's light (for the page) and its screen base."""
    img = np.zeros((H, W, 3), np.float32)
    Pn, Pp, br, temp = x1.sparks(t)
    if len(Pn):
        a, za = cam.project(Pn)
        b, zb = cam.project(Pp)
        col = look.blackbody(np.clip(temp, 0, 1)) * (br * 4.0)[:, None]
        sig = np.clip(0.55 * cam.F / za * 0.012, 0.6, 2.2)
        streaks(img, b[:, 0].astype(np.float64), b[:, 1].astype(np.float64), a[:, 0].astype(np.float64),
                a[:, 1].astype(np.float64), col[:, 0].astype(np.float64), col[:, 1].astype(np.float64),
                col[:, 2].astype(np.float64), sig.astype(np.float64))
    h = x1.flame_h(t)
    xl = None
    if h > 0.02:
        base = x1.Fw + np.array([0, 0, 0.02])
        top = base + np.array([0, 0, h])
        (bx, by), zb_ = cam.project(base)
        (tx, ty), _ = cam.project(top)
        hp = math.hypot(tx - bx, ty - by)
        ux, uy = (tx - bx) / max(hp, 1e-6), (ty - by) / max(hp, 1e-6)
        fr = t * FPS
        for k, (dxk, sc, wd, g, lean) in enumerate(((0.0, 1.0, 1.35, 1.0, 0.0), (-0.32, 0.5, 0.8, 0.55, -0.28),
                                                     (0.3, 0.44, 0.75, 0.5, 0.3))):
            ox = dxk * h * 0.5
            bb, _ = cam.project(base + np.array([ox, 0, 0]))
            ca, sa = math.cos(lean), math.sin(lean)
            FIRE.flame(img, float(bb[0]), float(bb[1]), float(hp * sc), float(ux * ca - uy * sa), float(ux * sa + uy * ca),
                       float(fr * 1.3 + 11 * k), int(7 + 13 * k), float(1.5 * g), float(wd))
        # the ember bed at its foot
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = max(hp * 0.16, 2.0)
        bed = np.exp(-(((xx - bx) / (rr * 1.6)) ** 2 + ((yy - by) / (rr * 0.55)) ** 2))
        img += bed[..., None] * np.array([2.2, 0.55, 0.08], np.float32)
        pw = 1.6 * h * (1.0 + 0.12 * math.sin(t * 17.0) + 0.08 * math.sin(t * 29.0))
        xl = np.array([[base[0], base[1], base[2] + 0.6 * h, pw * 1.0, pw * 0.5, pw * 0.16]])
    return img, xl


# ================================================================== shots ===

class Shot:
    """A book shot: geometry, pages and cameras, rendered to (rgb premultiplied + fire, alpha)."""

    def __init__(self, name, W=1920, H=804, mode='ink'):
        self.name, self.W, self.H = name, W, H
        self.glow = None
        self.x1 = None
        self.burnR = None
        self.plate_col = np.array([0.0, 0.0, 0.0])
        if name == 'set':                  # P1 prologue: the last written spread
            self.book = B.Book(TL=2.6, TR=1.4, seed=3)
            L, R, self.yl = leaves_last()
            self.pL, self.pR = Page(L, 70), Page(R, 70)
            self.light = B.Light((-55, 55, 38), (1.0, 0.66, 0.36), power=2.2, radius=9)
        elif name in ('mountain', 'x1'):
            self.book = B.Book(TL=0.6, TR=2.8, seed=5)
            L, R, words, self.mt = mountain_leaves(mode=mode)
            self.pL, self.pR = Page(L, 40), Page(R, 110)
            self.light = B.Light((-50, 48, 36), (1.0, 0.66, 0.36), power=2.2, radius=9)
            if name == 'x1':
                F = (10.1, 19.2)
                self.x1 = LettersToFire(self.book, words, F)
                self.burnR = BURN.params(F, t_start=3.5, speed=1.5, p=1.8, amp=0.45, freq=0.4, seed=4,
                                         brown=2.6, char=0.32, edge=0.035, lead=1.3)
        elif name == 'deep':
            self.book = B.Book(TL=1.6, TR=2.2, seed=9)
            self.deep = PG.Deep()
            R = self.deep.build(mode, 0.0, 8.0)
            self.glow = self.deep.glow
            L = Strokes()
            PG.text_page(L, 31, lines=38, box=(3.0, 2.9, 16.8, 26.4))
            self.pL, self.pR = Page(L, 40), Page(R, 110)
            self.light = B.Light((-50, 48, 36), (1.0, 0.66, 0.36), power=2.2, radius=9)

    def camera(self, t, key=None):
        W, H = self.W, self.H
        bk = self.book
        if self.name == 'set':
            keys = {
                'wide': ((-2.0, -58.0, 44.0), (0.5, 1.0, 1.0), 40.0, None, 0.0),
                'drift': ((-12.0, -17.0, 13.0), (-10.0, 3.0, 2.0), 30.0, None, 5.0),
                'sheaf': ((34.0, -26.0, 14.0), (14.0, -2.0, 1.2), 34.0, None, 3.0),
            }
            pos, tgt, fov, foc, k = keys[key or 'wide']
            cam = B.Cam(pos, tgt, fov, W, H)
            cam.dof_k = k
            return cam
        if self.name in ('mountain', 'x1'):
            fx, fy = self.mt.ring
            if self.name == 'mountain' or key == 'plate':
                tgt = bk.page_to_world('R', np.array([9.8]), np.array([8.2]))[0]
                pos = tgt + np.array([-0.6, -17.5, 16.0])
                cam = B.Cam(pos, tgt, 38.0, W, H)
                cam.dof_k = 2.0
                return cam
            # x1: from the caption band down to the text, pushing in slowly toward the flame
            F = self.x1.Fw
            a = smooth(t / 6.5)
            tgt = F + np.array([0.0, 1.2 - 0.9 * a, 0.0])
            pos = tgt + lerp((-0.5, -16.5, 17.0), (-0.3, -12.0, 12.5), a)
            cam = B.Cam(pos, tgt, 40.0, W, H)
            cam.dof_k = 3.0
            return cam
        if self.name == 'deep':
            v = 4.0 if key is None else float(key)
            tgt = bk.page_to_world('R', np.array([9.6]), np.array([v]))[0]
            pos = tgt + np.array([-0.4, -13.5, 13.0])
            cam = B.Cam(pos, tgt, 40.0, W, H)
            cam.dof_k = 2.5
            return cam

    def render(self, t, key=None, draw_t=1e9, plate=None):
        cam = self.camera(t, key)
        gains = None
        if self.x1 is not None:
            gains = self.x1.gains(t, len(self.pR.S))
        tL = self.pL.texture(draw_t)
        tR = self.pR.texture(draw_t, gains)
        fire, xl = (None, None)
        if self.x1 is not None:
            fire, xl = fire_layer(cam, self.x1, t, self.W, self.H)
        fl = B.flicker(t, 3, 0.09)
        L = B.Light(self.light.pos + np.array([1.5 * math.sin(t * 1.3), 0.8 * math.sin(t * 0.9), 0.5 * math.sin(t * 2.1)]),
                    self.light.col / np.max(self.light.col) * np.max(self.light.col) * fl, 1.0, self.light.radius)
        L.col = self.light.col * fl
        if self.glow is not None and key is not None and float(key) > 14.0:
            # the red glow waking at the bottom of the page, and the heat browning the paper round it
            gw = self.book.page_to_world('R', np.array([self.glow[0]]), np.array([self.glow[1]]))[0]
            gi = 5.0 * (1.0 + 0.15 * math.sin(t * 13.0) + 0.1 * math.sin(t * 31.0))
            gl = np.array([[gw[0], gw[1], gw[2] + 0.35, 1.6 * gi, 0.2 * gi, 0.025 * gi]])
            xl = gl if xl is None else np.vstack([xl, gl])
            if self.burnR is None:
                self.burnR = BURN.params(self.glow, t_start=1e6, brown=2.4, lead=1e7, seed=6)
        hdr, alpha, G = B.render(self.book, cam, L, tL, tR, t, burnR=self.burnR, xlights=xl)
        hdr = B.dof(hdr, G[..., 1].astype(np.float32), cam, cam.dof_k * self.W / 1920.0)
        if fire is not None:
            hdr = hdr + fire
        return hdr, alpha, cam


# ================================================================ delivery ===

# shot-local numbering: src frame = base + round(t * 24); durations follow the v3 beat sheet until the bar map locks
SHOTS = {
    'prologue': (1000, 320),     # C2  THE RED BOOK: the last written leaves, the drift, the blank sheaf
    'mountain': (2000, 240),     # C3  INK PAGE: THE MOUNTAIN, drawn on (T1 in the caption band, EDIT's)
    'x1': (3000, 160),           # C4  LETTERS TO FIRE: the letters glow, lift as sparks, a flame burns the page open
    'deep': (4000, 280),         # C9  INK PAGE: THE DEEP (heals in from the race, burns out into the Eye)
    'epilogue': (5000, 1040),    # C27-C30 THE LAST PAGES: plenty, the Havens, blank leaves, the page for the title
}

# the epilogue's clock (shot seconds)
EP_TURNS = ((9.4, 1.7), (23.3, 1.6), (28.6, 1.6))       # (start, duration) of each page turn
EP_HEAL = 31.0                                          # the scorched edges are whole again by here


class Epilogue:
    """C27-C30: the red book again, the hearth low. THE YEAR OF PLENTY draws itself on the recto; the leaf turns:
    THE HAVENS on the verso (the ship slides west off the page, the coast's flame glyphs kindle one by one, a
    small light answers at the stern, and in the roundel her bound hand holds up the lamp); the leaf turns: blank;
    and the next; the scorched edges have healed; the blank page waits for the title."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.book = B.Book(TL=2.8, TR=1.2, seed=13)
        self.light = B.Light((-55, 50, 30), (1.0, 0.52, 0.24), power=1.9, radius=10)
        L0 = Strokes()
        PG.text_page(L0, 81, lines=38, box=(3.0, 2.9, 16.8, 26.4))
        self.tex_text = Page(L0, 45).texture(1e9)
        self.plenty = PG.Plenty()
        self.pPlenty = Page(self.plenty.build('ink', 0.5, 8.5), 90)
        self.tex_plenty_done = None
        self.hv = PG.Havens()
        self.hv_static = self.hv.build('ink', 0, 0)
        self.hv_ppc = 90
        self.tex_blank = Page(Strokes(), 10).texture(1e9)
        base = B.PageTex(PG.PW, PG.PH, self.hv_ppc)
        pk = self.hv_static.pack()
        C, _ = pen.raster(pk, 1e9, self.hv_ppc, base.H, base.W, INK)
        self.hv_ink = C
        # fires: the flame glyphs, filled, and the stern light and the lamp in the roundel
        self.hv_fire = Strokes()
        self.fire_t = []
        for m, (fx, fy, hh) in enumerate(self.hv.fires):
            pts = np.array([[fx, fy - 0.05], [fx + 0.01, fy - 0.75 * hh]])
            self.hv_fire.add(pts, np.array([0.1, 0.05]), np.array([1.0, 1.0]), layer=INK)
            self.fire_t.append(12.0 + 1.1 * m)
        lx, ly = self.hv.lamp_flame
        self.hv_fire.add(np.array([[lx, ly + 0.05], [lx + 0.005, ly - 0.08]]), np.array([0.06, 0.03]), np.array([1.0, 1.0]))
        self.fire_t.append(18.2)
        self.hv_fire_pk = self.hv_fire.pack()

    def phase(self, t):
        """Which leaf is turning (index, phi) or (None, 0)."""
        for k, (a, d) in enumerate(EP_TURNS):
            if a <= t < a + d:
                x = (t - a) / d
                return k, x * x * (3 - 2 * x)
        return None, 0.0

    def havens_tex(self, t):
        tx = B.PageTex(PG.PW, PG.PH, self.hv_ppc)
        dx = 10.5 * smooth((t - 11.5) / 11.0) ** 1.3 if t > 11.5 else 0.0
        ship, stern = self.hv.ship_strokes(dx)
        pk = ship.pack()
        C, _ = pen.raster(pk, 1e9, self.hv_ppc, tx.H, tx.W, INK)
        occ = self.hv.ship_mask(dx, self.hv_ppc, tx.H, tx.W)
        tx.chan[..., 0] = np.maximum(self.hv_ink * (1.0 - occ), C)
        # the stern light moves with the ship
        fs = Strokes()
        fs.extend(self.hv_fire)
        fs.add(np.array([[stern[0], stern[1] + 0.02], [stern[0] + 0.004, stern[1] - 0.05]]), np.array([0.05, 0.03]),
               np.array([1.0, 1.0]))
        ft = self.fire_t + [17.0]
        g = np.array([smooth((t - a) / 0.5) * (0.85 + 0.15 * math.sin(t * (17 + 3 * i) + i)) for i, a in enumerate(ft)])
        F, _ = pen.raster(fs.pack(), 1e9, self.hv_ppc, tx.H, tx.W, INK, gain=g)
        tx.chan[..., 4] = F
        return tx.build(), dx, stern, g

    def frame(self, t):
        k, phi = self.phase(t)
        leaf = None
        xl = []
        if t < EP_TURNS[0][0]:
            texL = self.tex_text
            texR = self.pPlenty.texture(t)
        elif k == 0:
            if self.tex_plenty_done is None:
                self.tex_plenty_done = self.pPlenty.texture(1e9)
            hvt, dx, stern, g = self.havens_tex(t)
            texL, texR = self.tex_text, self.tex_blank
            leaf = (phi, self.tex_plenty_done, hvt)
        else:
            hvt, dx, stern, g = self.havens_tex(t)
            texL = hvt if (k is None and t < EP_TURNS[2][0]) or k in (None, 1) else self.tex_blank
            if t >= EP_TURNS[1][0] + EP_TURNS[1][1]:
                texL = self.tex_blank
            texR = self.tex_blank
            if k in (1, 2):
                leaf = (phi, self.tex_blank, self.tex_blank)
            if texL is hvt:
                for i, (fx, fy, hh) in enumerate(self.hv.fires):
                    if g[i] > 0.01:
                        wp = self.book.page_to_world('L', np.array([fx]), np.array([fy - 0.3]))[0]
                        xl.append([wp[0], wp[1], wp[2] + 0.6, 0.35 * g[i], 0.12 * g[i], 0.03 * g[i]])
        burnL = BURN.edge_params('L', 0.9, EP_HEAL, seed=21)
        burnR = BURN.edge_params('R', 0.9, EP_HEAL, seed=22)
        return texL, texR, leaf, (np.array(xl) if xl else None), burnL, burnR

    def camera(self, t):
        bk = self.book
        W, H = self.W, self.H
        pl = bk.page_to_world('R', np.array([9.8]), np.array([11.0]))[0]
        hv = bk.page_to_world('L', np.array([10.0]), np.array([10.5]))[0]
        sp = np.array([0.0, 0.0, bk.zg])
        bl = bk.page_to_world('R', np.array([10.0]), np.array([13.0]))[0]
        # C27 the plenty page with its scorched fore-edge in frame; C28 over to the Havens; C29 back to see the
        # spread (the turns, the healed edges); C30 in to the blank recto for the title
        keys_t = [(0.0, pl + [1.2, 0, 0]), (9.0, pl + [1.0, 0, 0]), (12.0, hv + [0.5, -1.0, 0]), (22.3, hv + [0.2, -1.5, 0]),
                  (25.5, sp + [0, -0.5, 0]), (33.0, sp + [2.5, -0.5, 0]), (43.3, bl)]
        keys_o = [(0.0, (-2.0, -30.0, 28.0)), (9.0, (-1.8, -29.0, 27.0)), (12.0, (1.0, -31.0, 29.0)),
                  (22.3, (0.8, -30.0, 28.0)), (25.5, (0.0, -52.0, 46.0)), (33.0, (0.5, -50.0, 44.0)),
                  (43.3, (0.0, -30.0, 28.0))]
        tgt = keyed(keys_t, t)
        pos = tgt + keyed(keys_o, t)
        cam = B.Cam(pos, tgt, 38.0, W, H)
        cam.dof_k = 2.0
        return cam


def catrom(keys, t):
    """Catmull-Rom through (t, vector) keys with an eased global parameter (still at both ends)."""
    ts = np.array([k[0] for k in keys], np.float64)
    vs = [np.asarray(k[1], np.float64) for k in keys]
    u = smooth((t - ts[0]) / (ts[-1] - ts[0]))
    tt = ts[0] + u * (ts[-1] - ts[0])
    i = int(np.clip(np.searchsorted(ts, tt) - 1, 0, len(ts) - 2))
    f = (tt - ts[i]) / (ts[i + 1] - ts[i])
    p0, p1, p2, p3 = vs[max(i - 1, 0)], vs[i], vs[i + 1], vs[min(i + 2, len(vs) - 1)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)


def keyed(keys, t):
    """Keys with a hold at each: smoothstep within each interval (the camera settles on every key)."""
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return np.asarray(keys[0][1], np.float64)
    if t >= ts[-1]:
        return np.asarray(keys[-1][1], np.float64)
    i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
    u = smooth((t - ts[i]) / (ts[i + 1] - ts[i]))
    return lerp(keys[i][1], keys[i + 1][1], u)


class Delivery:
    """Frames of the four book shots (P1, P2 x2, X1a) as pure functions of the src frame number."""

    def __init__(self, W=1920, H=804):
        self.W, self.H = W, H
        self.cache = {}

    def shot(self, name):
        if name not in self.cache:
            if name == 'epilogue':
                self.cache[name] = Epilogue(self.W, self.H)
                return self.cache[name]
            if name == 'prologue':
                sh = Shot('set', self.W, self.H)
            elif name == 'mountain':
                sh = Shot('mountain', self.W, self.H)
                L, R, words, sh.mt = mountain_leaves(mode='ink', t0=0.5, t1=8.5)
                sh.pR = Page(R, sh.pR.ppc)
            elif name == 'x1':
                sh = Shot('x1', self.W, self.H)
            else:
                sh = Shot('deep', self.W, self.H)
                sh.pR = Page(sh.deep.build('ink', 0.3, 8.6), sh.pR.ppc)
            self.cache[name] = sh
        return self.cache[name]

    def camera(self, name, sh, t):
        W, H = self.W, self.H
        bk = sh.book
        if name == 'prologue':
            keys_p = [(0.0, (-12.0, -17.0, 13.0)), (6.5, (1.0, -20.0, 15.5)), (13.33, (34.0, -26.0, 14.0))]
            keys_t = [(0.0, (-10.0, 3.0, 2.0)), (6.5, (5.0, 1.0, 2.0)), (13.33, (14.0, -2.0, 1.2))]
            cam = B.Cam(catrom(keys_p, t), catrom(keys_t, t), 30.0 + 4.0 * smooth(t / 13.33), W, H)
            cam.dof_k = 5.0 - 2.0 * smooth(t / 13.33)
            return cam
        if name == 'mountain':
            tgt = bk.page_to_world('R', np.array([9.8]), np.array([8.2]))[0]
            pos0 = tgt + np.array([-0.6, -17.5, 16.0])
            pos = pos0 + (tgt - pos0) * 0.07 * smooth(t / 10.0) + np.array([0.4 * smooth(t / 10.0), 0, 0])
            cam = B.Cam(pos, tgt, 38.0, W, H)
            cam.dof_k = 2.0
            return cam
        if name == 'x1':
            # from the plate's framing (the end of C3) down to the tale, then in toward the flame
            tgtA = bk.page_to_world('R', np.array([9.8]), np.array([8.2]))[0]
            posA = tgtA + np.array([-0.6, -17.5, 16.0]) + (np.array([-0.6, -17.5, 16.0]) * -0.07) + np.array([0.4, 0, 0])
            F = sh.x1.Fw
            a = smooth(t / 2.6)
            b = smooth((t - 2.6) / 3.9)
            tgtB = F + np.array([0.0, 1.2, 0.0])
            posB = tgtB + np.array([-0.5, -16.5, 17.0])
            tgtC = F + np.array([0.0, 0.3, 0.0])
            posC = tgtC + np.array([-0.3, -12.0, 12.5])
            tgt = lerp(lerp(tgtA, tgtB, a), tgtC, b)
            pos = lerp(lerp(posA, posB, a), posC, b)
            cam = B.Cam(pos, tgt, 38.0 + 2.0 * a, W, H)
            cam.dof_k = 2.0 + 1.0 * a
            return cam
        # the deep: descend the page after the vein
        v = 3.6 + 18.6 * smooth((t - 0.4) / 9.0) ** 1.15
        xv = float(np.interp(v + 1.0, sh.deep.vein[:, 1], sh.deep.vein[:, 0]))
        xv = 0.55 * xv + 0.45 * 9.6
        tgt = bk.page_to_world('R', np.array([xv]), np.array([v]))[0]
        pos = tgt + np.array([-0.4, -13.5, 13.0])
        cam = B.Cam(pos, tgt, 40.0, W, H)
        cam.dof_k = 2.5
        return cam

    def render(self, name, f):
        base, n = SHOTS[name]
        t = (f - base) / FPS
        sh = self.shot(name)
        if name == 'epilogue':
            cam = sh.camera(t)
            texL, texR, leaf, xl, burnL, burnR = sh.frame(t)
            fl = B.flicker(t, 5, 0.12)
            L = B.Light(sh.light.pos + np.array([1.2 * math.sin(t * 1.1), 0.6 * math.sin(t * 0.8), 0.4 * math.sin(t * 1.9)]),
                        (1, 1, 1), 1.0, sh.light.radius)
            L.col = sh.light.col * fl
            hdr, alpha, G = B.render(sh.book, cam, L, texL, texR, t, burnL=burnL, burnR=burnR, xlights=xl, leaf=leaf)
            hdr = B.dof(hdr, G[..., 1].astype(np.float32), cam, cam.dof_k * self.W / 1920.0)
            return hdr, alpha, cam, t
        cam = self.camera(name, sh, t)
        draw_t = 1e9 if name in ('prologue', 'x1') else t
        burnR = sh.burnR
        xl = None
        if name == 'deep':
            gx, gy = sh.deep.glow
            if t < 3.0:
                # it burns back into parchment: the hole left by the last shot closes from the edges in
                burnR = BURN.params((10.0, 6.5), t_start=0.0, speed=2.2, p=1.3, amp=0.5, freq=0.3, seed=8, brown=1.8,
                                    char=0.3, edge=0.035, mode=1, pivot=1.3, lead=0.6)
            else:
                # the red glow wakes, the paper browns and smokes, and the glow burns through (into the Eye)
                burnR = BURN.params((gx, gy - 0.2), t_start=9.9, speed=3.0, p=1.9, amp=0.45, freq=0.35, seed=9,
                                    brown=3.4, char=0.3, edge=0.035, lead=3.2)
            gw = sh.book.page_to_world('R', np.array([gx]), np.array([gy]))[0]
            gi = 6.0 * smooth((t - 6.0) / 3.5) * (1.0 + 0.15 * math.sin(t * 13.0) + 0.1 * math.sin(t * 31.0))
            if gi > 0:
                xl = np.array([[gw[0], gw[1], gw[2] + 0.35, 1.6 * gi, 0.2 * gi, 0.025 * gi]])
        gains = sh.x1.gains(t, len(sh.pR.S)) if sh.x1 is not None else None
        tL = sh.pL.texture(1e9)
        tR = sh.pR.texture(draw_t, gains)
        fire = None
        if sh.x1 is not None:
            fire, xl2 = fire_layer(cam, sh.x1, t, self.W, self.H)
            xl = xl2 if xl is None else (xl if xl2 is None else np.vstack([xl, xl2]))
        fl = B.flicker(t, 3, 0.09)
        L = B.Light(sh.light.pos + np.array([1.5 * math.sin(t * 1.3), 0.8 * math.sin(t * 0.9), 0.5 * math.sin(t * 2.1)]),
                    (1, 1, 1), 1.0, sh.light.radius)
        L.col = sh.light.col * fl
        hdr, alpha, G = B.render(sh.book, cam, L, tL, tR, t, burnR=burnR, xlights=xl)
        hdr = B.dof(hdr, G[..., 1].astype(np.float32), cam, cam.dof_k * self.W / 1920.0)
        if fire is not None:
            hdr = hdr + fire
        return hdr, alpha, cam, t


def deliver(names, frames, out, matte_out, W=1920, H=804):
    D = Delivery(W, H)
    for f in frames:
        name = [k for k, (b, n) in SHOTS.items() if b <= f < b + n]
        if not name or (names and name[0] not in names):
            continue
        t0 = time.time()
        hdr, alpha, cam, t = D.render(name[0], f)
        rgb = look.finish(hdr, exposure=1.15, bloom_strength=0.06, bloom_threshold=1.2, vignette_amount=0.32)
        # the matte rides with the colour: EDIT comps out = rgb + (1 - a) * next (both display-referred)
        look.save_png(look.frame_path(out, f), rgb)
        look.save_png(look.frame_path(matte_out, f), np.repeat(np.clip(alpha, 0, 1)[..., None], 3, -1))
        print(f, name[0], '%.2fs' % t, '%.1fs' % (time.time() - t0), flush=True)


def comp(hdr, alpha, plate=None):
    """Stand-in comp: the book layer (premultiplied + fire) over the next shot's plate."""
    if plate is None:
        return hdr
    return hdr + (1.0 - alpha)[..., None] * plate


def finish(hdr, exposure=1.15):
    return look.finish(hdr, exposure=exposure, bloom_strength=0.06, bloom_threshold=1.2, vignette_amount=0.32)


def save(path, srgb, q=92):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out = (np.clip(srgb, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8)
    if path.endswith('.jpg'):
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, q])
    else:
        cv2.imwrite(path, out)


def ember_plate(W, H, cam, x1, t):
    """A stand-in for the next shot (the fire in the dark): warm black with the flame's glow."""
    p = np.zeros((H, W, 3), np.float32)
    (bx, by), _ = cam.project(x1.Fw)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - bx, (yy - by) * 1.2) / W
    g = np.exp(-d / 0.22)[..., None]
    return p + g * np.array([0.05, 0.018, 0.004], np.float32) + np.array([0.004, 0.003, 0.0025], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what')
    ap.add_argument('--frames', default='')
    ap.add_argument('--out', default=os.path.join(ROOT, 'review', 'v3', 'map_tests'))
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--key', default=None)
    a = ap.parse_args()
    W, H = int(1920 * a.scale), int(804 * a.scale)
    if a.what == 'x1':
        sh = Shot('x1', W, H)
        for f in [int(x) for x in a.frames.split(',')]:
            t0 = time.time()
            t = f / FPS
            hdr, alpha, cam = sh.render(t)
            plate = ember_plate(W, H, cam, sh.x1, t)
            img = finish(comp(hdr, alpha, plate))
            save(os.path.join(a.out, f'x1_{f:04d}.jpg'), img)
            print(f, '%.1fs' % (time.time() - t0), flush=True)
    elif a.what == 'frames':
        fr = []
        for part in a.frames.split(','):
            if '-' in part:
                x, y = part.split('-')
                fr += list(range(int(x), int(y) + 1))
            elif part:
                fr.append(int(part))
        out = a.out if a.out != os.path.join(ROOT, 'review', 'v3', 'map_tests') else os.path.join(ROOT, 'renders', 'book_C')
        deliver([], fr, out, out.rstrip('/') + '_matte', W, H)
    elif a.what == 'stills':
        jobs = [('set', 'ink', ['wide', 'drift', 'sheaf']), ('mountain', 'ink', ['plate']), ('mountain', 'pencil', ['plate']),
                ('deep', 'ink', ['5.5', '21.0']), ('deep', 'pencil', ['5.5'])]
        for name, mode, keys in jobs:
            sh = Shot(name, W, H, mode)
            for key in keys:
                t0 = time.time()
                hdr, alpha, cam = sh.render(0.0, key)
                save(os.path.join(a.out, f'{name}_{mode}_{key}.jpg'), finish(hdr))
                print(name, mode, key, '%.1fs' % (time.time() - t0), flush=True)
    else:
        sh = Shot(a.what, W, H)
        for f in [float(x) for x in (a.frames or '0').split(',')]:
            t0 = time.time()
            key = a.key
            if a.what == 'deep' and key is not None:
                key = float(key)
            hdr, alpha, cam = sh.render(f, key)
            save(os.path.join(a.out, f'{a.what}_{a.key or "k"}_{int(f * 100):05d}.jpg'), finish(hdr))
            print(a.what, a.key, f, '%.1fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
