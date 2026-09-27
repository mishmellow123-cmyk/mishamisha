"""THE RING ON THE PAGE: the Ring close-up FALLBACK (H5: "an ink ring drawn on the book's page, not the numba ring").

C is the book: the Ring is drawn in it, a band of gold leaf in pen and ink lying in three-quarter view, its outer face
carrying the canonical inscription (`assets/ring`, the script of fire: one inscription on both rings) and its inner
face, seen through the hole, the inner line. It lies on a page by a drawn hearth fire, living ink flames with gold
laid in them and a white heart.

C22 THE UNMAKING on its beats (5360-5519 -> renders/book_C, via book_c.py):
    5360  the drawn fire burns, its heart white
    5366  the Ring drops into it, crisp (a fall of seven frames, a small squash as it lands)
    ...   it slumps in the heat, the gold dull red, darker
    5420  bar 68 b4: its letters flare once, still letters
    5440  bar 69 b1: they go out
    ...   it runs into a bead (the hole closes; never a glowing donut)
    5490  the hearth flares; white by 5519 (bar 70 opens in white)
Stills (the find / the fire test fallbacks): `python ringpage.py stills --out DIR`.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import pen  # noqa: E402
import book as B  # noqa: E402
import pages as PG  # noqa: E402
import redbook as RB  # noqa: E402
from noise import gnoise  # noqa: E402
from pen import Strokes  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ASSET = os.path.join(ROOT, 'assets', 'ring')
T0 = 5360                      # C22's first frame
FPS = 24.0
smooth = RB.smooth

_STRIPS = {}


def strip(face):
    """The canonical inscription strip (coverage 0..1), rows across the band, columns once round it."""
    if face not in _STRIPS:
        im = cv2.imread(os.path.join(ASSET, 'inscription_%s.png' % face), cv2.IMREAD_UNCHANGED).astype(np.float32)
        im /= 65535.0
        im = cv2.resize(im, (im.shape[1] // 3, im.shape[0] // 3), interpolation=cv2.INTER_AREA)
        _STRIPS[face] = np.ascontiguousarray(im)
    return _STRIPS[face]


# ================================================================ the ring ===

@njit(cache=True, inline='always')
def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


@njit(cache=True)
def _inv_bilinear(px, py, ax, ay, bx, by, cx, cy, dx, dy):
    """(u, v) of p in the bilinear patch a(0,0) b(1,0) c(1,1) d(0,1); u < -1 when outside."""
    ex, ey = bx - ax, by - ay
    fx, fy = dx - ax, dy - ay
    gx, gy = ax - bx + cx - dx, ay - by + cy - dy
    hx, hy = px - ax, py - ay
    k2 = _cross(gx, gy, fx, fy)
    k1 = _cross(ex, ey, fx, fy) + _cross(hx, hy, gx, gy)
    k0 = _cross(hx, hy, ex, ey)
    if abs(k2) < 1e-10:
        if abs(k1) < 1e-12:
            return -9.0, -9.0
        v = -k0 / k1
        den = ex + gx * v
        u = (hx - fx * v) / den if abs(den) > 1e-12 else (hy - fy * v) / (ey + gy * v + 1e-12)
        return u, v
    w = k1 * k1 - 4.0 * k0 * k2
    if w < 0.0:
        return -9.0, -9.0
    w = math.sqrt(w)
    for sg in (-1.0, 1.0):
        v = (-k1 + sg * w) / (2.0 * k2)
        den = ex + gx * v
        if abs(den) > 1e-12:
            u = (hx - fx * v) / den
        else:
            u = (hy - fy * v) / (ey + gy * v + 1e-12)
        if -1e-6 <= u <= 1.0 + 1e-6 and -1e-6 <= v <= 1.0 + 1e-6:
            return u, v
    return -9.0, -9.0


@njit(cache=True, inline='always')
def _radius(P, th):
    R, wob, wph = P[2], P[6], P[7]
    return R * (1.0 + wob * (0.6 * math.sin(3.0 * th + wph) + 0.4 * math.sin(5.0 * th - 1.3 * wph)))


@njit(cache=True)
def _corner(P, part, th, s):
    """Page cm of ring-space (theta, s) on a part: 0 back inner face, 1 back rim, 2 front rim, 3 front outer face."""
    cx, cy, k, h, tr, pool = P[0], P[1], P[3], P[4], P[5], P[8]
    ax, ay, hb = P[12], P[13], P[14]
    Ro = _radius(P, th)
    Ri = max(Ro - tr, 0.0)
    st = math.sin(th)
    # the hole narrows toward a slit and slides back as the gold runs forward (never a round hole closing: no donut)
    ix = cx + Ri * ax * math.cos(th)
    iy = cy + k * Ri * ay * st - hb
    if part == 0:
        return ix, iy + s * h * ay
    if part == 3:
        sw = h * (1.0 + pool * max(st, 0.0) ** 4) * (1.0 - 0.6 * (1.0 - ay) * (1.0 - abs(st)))
        return cx + Ro * math.cos(th), cy + k * Ro * st + s * sw
    ox_ = cx + Ro * math.cos(th)
    oy_ = cy + k * Ro * st
    return ix + (ox_ - ix) * s, iy + (oy_ - iy) * s


@njit(cache=True)
def _sample(S, u, v):
    H, W = S.shape
    x = (u - math.floor(u)) * W - 0.5
    y = min(max(v, 0.0), 1.0) * (H - 1)
    ix = int(math.floor(x))
    iy = min(int(y), H - 2)
    fx = x - ix
    fy = y - iy
    i0 = ix % W
    i1 = (ix + 1) % W
    return ((S[iy, i0] * (1 - fx) + S[iy, i1] * fx) * (1 - fy) + (S[iy + 1, i0] * (1 - fx) + S[iy + 1, i1] * fx) * fy)


@njit(cache=True)
def draw_ring(ch, ox, oy, ppc, P, SO, SI, glow, engr, heat, nth=360, ns=4):
    """Paint the ring into the channel window ch (H x W x 6 over page cm [ox, ..] x [oy, ..]) back to front,
    overwriting: gilt with form hatching, the inscription cut in it (engr) or awake in fire (glow), heat in the gold.
    P: cx, cy, R, k, h, tr, wob, wph, pool, uo, ui."""
    H, W = ch.shape[0], ch.shape[1]
    uo, ui = P[9], P[10]
    for part in range(4):
        th0 = math.pi if part < 2 else 0.0
        for a in range(nth // 2):
            ta = th0 + math.pi * a / (nth // 2)
            tb = th0 + math.pi * (a + 1) / (nth // 2)
            for b in range(ns):
                sa = b / ns
                sb = (b + 1) / ns
                ax, ay = _corner(P, part, ta, sa)
                bx, by = _corner(P, part, tb, sa)
                cx_, cy_ = _corner(P, part, tb, sb)
                dx, dy = _corner(P, part, ta, sb)
                x0 = int(math.floor((min(ax, bx, cx_, dx) - ox) * ppc))
                x1 = int(math.ceil((max(ax, bx, cx_, dx) - ox) * ppc))
                y0 = int(math.floor((min(ay, by, cy_, dy) - oy) * ppc))
                y1 = int(math.ceil((max(ay, by, cy_, dy) - oy) * ppc))
                if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                    continue
                for i in range(max(y0, 0), min(y1 + 1, H)):
                    py = oy + (i + 0.5) / ppc
                    for j in range(max(x0, 0), min(x1 + 1, W)):
                        px = ox + (j + 0.5) / ppc
                        u, v = _inv_bilinear(px, py, ax, ay, bx, by, cx_, cy_, dx, dy)
                        if u < -1e-6 or u > 1.0 + 1e-6 or v < -1e-6 or v > 1.0 + 1e-6:
                            continue
                        th = ta + (tb - ta) * u
                        s = sa + (sb - sa) * v
                        st = math.sin(th)
                        arc = _radius(P, th) * th
                        let = 0.0
                        if part == 3:
                            tone = 0.5 * (1.0 - abs(st)) ** 1.5 + 0.12 * s
                            let = _sample(SO, uo + (math.pi - th) / (2 * math.pi), s)
                        elif part == 0:
                            tone = 0.55 + 0.15 * s
                            let = _sample(SI, ui + (th - math.pi) / (2 * math.pi), s)
                        else:
                            tone = 0.08 + 0.1 * (1.0 - abs(st))
                        # hatching that follows the form, only where it turns into shadow: fine pen lines across
                        # the band, a little uneven, closer and heavier toward the silhouette
                        sp = 0.15
                        fr = arc / sp + 0.1 * math.sin(s * 5.0 + th * 17.0)
                        fr = fr - math.floor(fr)
                        duty = min(max(tone - 0.22, 0.0) * 1.1, 0.42) if part == 3 else min(max(tone - 0.35, 0.0) * 0.9, 0.2)
                        aa = 0.7 / (ppc * sp)
                        dd = 0.5 * duty - abs(fr - 0.5)
                        hat = min(max(dd / aa + 0.5, 0.0), 1.0) if duty > 0.02 and part != 1 and part != 2 else 0.0
                        cut = let * engr
                        dark = max(cut, 0.85 * hat)
                        ch[i, j, 0] = max(dark, 0.18 * tone)
                        ch[i, j, 1] = 0.0
                        ch[i, j, 2] = (1.0 - 0.92 * dark) * (1.0 - 0.3 * tone)
                        ch[i, j, 3] = 0.0
                        pool_k = 0.35 + 0.65 * s if part == 3 else 0.5
                        ch[i, j, 4] = let * glow + heat * pool_k * (1.0 - dark)
                        ch[i, j, 5] = 0.0


@njit(cache=True)
def draw_bead(ch, ox, oy, ppc, cx, cy, rx, ry, heat):
    """A bead of gold: a small dome in gilt, shaded to the lower right, its highlight high on the left, a dull heat
    pooled low in it; hatched a little where it turns away."""
    H, W = ch.shape[0], ch.shape[1]
    i0 = max(int((cy - ry - oy) * ppc) - 1, 0)
    i1 = min(int((cy + ry - oy) * ppc) + 2, H)
    j0 = max(int((cx - rx - ox) * ppc) - 1, 0)
    j1 = min(int((cx + rx - ox) * ppc) + 2, W)
    for i in range(i0, i1):
        b = (oy + (i + 0.5) / ppc - cy) / ry
        for j in range(j0, j1):
            a = (ox + (j + 0.5) / ppc - cx) / rx
            d = a * a + b * b
            if d > 1.0:
                continue
            nz = math.sqrt(1.0 - d)
            tone = min(max(0.1 + 0.62 * (0.55 * b + 0.3 * a + 0.4 * (1.0 - nz)), 0.0), 0.85)
            hl = math.exp(-((a + 0.34) ** 2 + (b + 0.42) ** 2) / 0.018)
            fr = (a * rx * 1.0 + b * ry * 0.3) / 0.12
            fr = fr - math.floor(fr)
            duty = min(max(tone - 0.35, 0.0) * 1.2, 0.35)
            hat = 1.0 if (duty > 0.02 and abs(fr - 0.5) < 0.5 * duty) else 0.0
            ch[i, j, 0] = max(0.85 * hat, 0.2 * tone) * (1.0 - hl)
            ch[i, j, 1] = 0.25 * hl
            ch[i, j, 2] = (1.0 - 0.9 * 0.85 * hat) * (1.0 - 0.35 * tone) * (1.0 - hl) + hl
            ch[i, j, 3] = 0.0
            ch[i, j, 4] = heat * (0.3 + 0.7 * max(b, 0.0)) * (1.0 - 0.8 * hl)
            ch[i, j, 5] = 0.0


def ring_outline(S, P, seed, width=0.02):
    """The pen's outline of the ring: both edges of the rim, the front face's foot and its two sides, and the foot
    of the inner face where the hole shows it."""
    cx, cy, R, k, h, tr, wob, wph, pool = P[:9]
    th = np.linspace(0, 2 * np.pi, 241)
    Ro = R * (1 + wob * (0.6 * np.sin(3 * th + wph) + 0.4 * np.sin(5 * th - 1.3 * wph)))
    Ri = np.maximum(Ro - tr, 0.0)
    ax, ay, hb = P[12], P[13], P[14]
    E_o = np.column_stack([cx + Ro * np.cos(th), cy + k * Ro * np.sin(th)])
    E_i = np.column_stack([cx + Ri * ax * np.cos(th), cy + k * Ri * ay * np.sin(th) - hb])
    pen.line(S, E_o, width, seed, smooth=0, lift=(3.0, 6.0))
    if Ri.max() * ay > 0.03:
        pen.line(S, E_i, width * 0.8, seed + 1, smooth=0, lift=(2.0, 5.0))
    f = th <= np.pi
    st = np.sin(th[f])
    foot = E_o[f] + np.column_stack([np.zeros(f.sum()), h * (1 + pool * np.maximum(st, 0) ** 4)
                                     * (1.0 - 0.6 * (1.0 - ay) * (1.0 - np.abs(st)))])
    pen.line(S, foot, width * 1.15, seed + 2, smooth=0, lift=(3.0, 6.0))
    for q in (0, -1):
        side = np.array([E_o[f][q], foot[q]])
        pen.line(S, side, width, seed + 3 + q, smooth=0)
    # the inner face's foot, where it shows through the hole
    if Ri.max() > 0.3 and ay > 0.6:
        b = th > np.pi
        tb = th[b]
        Rib = Ri[b]
        fx = cx + Rib * np.cos(tb)
        fy = cy + k * Rib * np.sin(tb) + h
        yf = cy + k * Rib * np.sqrt(np.clip(1 - ((fx - cx) / np.maximum(Rib, 1e-6)) ** 2, 0, 1))
        vis = fy < yf - 0.02
        run = []
        for vv, x_, y_ in zip(vis, fx, fy):
            if vv:
                run.append((x_, y_))
            elif len(run) > 3:
                pen.line(S, np.array(run), width * 0.7, seed + 9, smooth=0)
                run = []
            else:
                run = []
        if len(run) > 3:
            pen.line(S, np.array(run), width * 0.7, seed + 9, smooth=0)


# ================================================================ the fire ===

def tongue(bx, by, H, W, lean, t, k):
    """One drawn flame tongue (closed outline, page cm, y down): a broad body that narrows into a licking S-curve and
    a curled tip, alive (it sways, stretches and breathes)."""
    s = np.linspace(0.0, 1.0, 26)
    ph = 1.7 * k
    Hh = H * (1.0 + 0.08 * math.sin(t * 2.1 + ph) + 0.05 * math.sin(t * 5.3 + 2 * ph))
    sway = W * (0.55 * np.sin(2.6 * s - t * 2.7 + ph) * s ** 1.4 + 0.25 * np.sin(5.1 * s - t * 4.3 + 2 * ph) * s ** 2)
    curl = W * 0.9 * np.sign(math.sin(ph + 0.5)) * np.clip((s - 0.72) / 0.28, 0, 1) ** 2
    cxl = bx + lean * s * Hh + sway + curl
    cyl = by - s * Hh
    wd = W * (1.0 - s) ** 0.9 * (0.75 + 0.25 * np.sin(np.pi * np.minimum(s * 1.6, 1.0)))
    left = np.column_stack([cxl - wd, cyl])
    right = np.column_stack([cxl + wd, cyl])[::-1]
    return np.vstack([left, right])


def gn_(x, k):
    return np.array([gnoise(float(a), 0.37 * k, 61) for a in np.atleast_1d(x)])


class RingPage:
    """The right page with the drawing (text above and below it), re-drawn per frame in a window."""

    WIN = (2.8, 6.6, 17.8, 21.6)          # page cm (x0, y0, x1, y1) of the drawing
    FIRE = (10.2, 18.4)                   # the ember bed's centre
    BED = (10.2, 17.3)                    # where the ring lies in the fire

    def __init__(self, seed=7, ppc=130):
        self.ppc = ppc
        S = Strokes()

        def skip(li, xa, xb):
            return [(xa - 1, xb + 1)] if 5 <= li <= 30 else []
        PG.text_page(S, seed, lines=38, box=(3.2, 2.9, 17.0, 26.4), skip=skip)
        self.base = RB.Page(S, ppc).texture(1e9).chan.copy()
        self.tex = B.PageTex(PG.PW, PG.PH, ppc)
        rng = np.random.default_rng(seed)
        fx, fy = self.FIRE
        self.tongues = []
        for k, (dx, Hk, Wk) in enumerate(((-2.3, 2.3, 0.62), (-1.35, 3.4, 0.8), (-0.35, 4.3, 0.95), (0.7, 3.8, 0.9),
                                          (1.65, 2.9, 0.72), (2.45, 2.0, 0.55), (-0.9, 2.6, 0.6), (1.2, 2.4, 0.6))):
            self.tongues.append((fx + dx + rng.normal(0, 0.08), fy - 0.1, Hk * rng.uniform(0.92, 1.06), Wk,
                                 rng.normal(0, 0.05) - 0.03 * dx))
        self.front = [(fx - 1.55, fy + 0.25, 1.4, 0.42, 0.05), (fx + 1.35, fy + 0.3, 1.15, 0.38, -0.06),
                      (fx - 0.1, fy + 0.35, 0.95, 0.34, 0.02)]
        self.embers = [(fx + (k - 5) * 0.52 + rng.normal(0, 0.08), fy + 0.2 + rng.normal(0, 0.1),
                        rng.uniform(0.2, 0.32)) for k in range(11)]

    # ------------------------------------------------------------ state ---
    @staticmethod
    def state(t):
        """The ring's shape and light at shot time t (s from 5360)."""
        bx, by = RingPage.BED
        t_d, t_i = 0.2, 0.55
        y_top = 8.2
        if t < t_d:
            cy = y_top - 6.0                                 # above, out of frame
        elif t < t_i:
            q = (t - t_d) / (t_i - t_d)
            cy = y_top + (by - y_top) * q * q               # a crisp fall, quickening
        else:
            cy = by
        land = math.exp(-max(t - t_i, 0.0) / 0.07) * (t > t_i)
        m1 = smooth((t - t_i) / 2.8)
        m2 = smooth((t - 3.4) / 2.0)
        mb = min(m2 / 0.85, 1.0)
        R = 3.3 * (1.0 - 0.72 * mb ** 0.8)
        tr = 0.46
        k = (0.4 - 0.07 * m1) * (1.0 - 0.1 * land) + 0.24 * mb
        h = (1.32 - 0.25 * m1) * (1.0 - 0.1 * land) * (1.0 - 0.74 * mb)
        ax = 1.0 - 0.55 * mb
        ay = max(1.0 - 1.35 * mb, 0.0) ** 1.2
        hb = 0.5 * k * R * mb
        wob = 0.035 * m1 * (1.0 - 0.5 * m2) + 0.012 * m2
        pool = 0.25 * m1 * (1.0 - m2)
        cy_ = cy + 0.35 * m2                                # the bead settles low in the bed
        heat = 0.04 + 0.26 * m1 + 0.14 * math.sin(math.pi * m2) - 0.12 * smooth((t - 5.0) / 0.8)
        # the letters: cut in the gold, flare once on 68 b4, out on 69 b1
        tf, to = (5420 - T0) / FPS, (5440 - T0) / FPS
        glow = 0.0
        if t >= tf - 0.04:
            glow = 1.35 * smooth((t - tf + 0.04) / 0.08) * (1.0 - smooth((t - to + 0.12) / 0.12))
            glow *= 1.0 + 0.08 * math.sin(t * 41.0) + 0.05 * math.sin(t * 23.0)
        engr = 1.0 - smooth((t - to + 0.02) / 0.16)
        return (np.array([bx, cy_, R, k, h, tr, wob, 1.3 + 0.4 * t, pool, 0.12, 0.61, m2, ax, ay, hb]), glow, engr,
                max(heat, 0.0))

    # ------------------------------------------------------------ paint ---
    def texture(self, t, white=0.0, find=False, pristine=False, awake=None):
        """The page at shot time t; pristine: the ring as it lies before any heat (the find, the fire test), awake:
        its letters' glow (the fire test: awake in fire, not even warm)."""
        ppc = self.ppc
        ch = self.tex.chan
        ch[:] = self.base
        x0, y0, x1, y1 = self.WIN
        i0, i1, j0, j1 = int(y0 * ppc), int(y1 * ppc), int(x0 * ppc), int(x1 * ppc)
        win = np.ascontiguousarray(ch[i0:i1, j0:j1])
        oy, ox = i0 / ppc, j0 / ppc
        Hh, Ww = win.shape[:2]
        P, glow, engr, heat = self.state(0.9 if pristine else t)
        if awake is not None:
            glow, engr, heat = awake, 1.0, 0.0
        if not find:
            flare = 1.0 + 0.5 * math.exp(-max(t - 0.55, 0.0) / 0.2) * (t > 0.55) + 18.0 * white
            self._fire(win, ox, oy, t, self.tongues, flare, seed=11)
            self._embers(win, ox, oy, t, flare)
        if P[11] >= 0.85:
            # the bead: a drop of gold sitting in the embers (the hole long closed; never a glowing ring)
            rx = P[2]
            ry = P[3] * P[2] + 0.5 * P[4]
            draw_bead(win, ox, oy, float(ppc), float(P[0]), float(P[1] + 0.5 * P[4]), float(rx), float(ry), float(heat))
            S = Strokes()
            a = np.linspace(0, 2 * np.pi, 61)
            pen.line(S, np.column_stack([P[0] + rx * np.cos(a), P[1] + 0.5 * P[4] + ry * np.sin(a)]), 0.026, 78, smooth=0,
                     lift=(2.0, 4.0))
            C, _ = pen.raster(S.pack(), 1e9, ppc, Hh, Ww, pen.INK, ox=ox, oy=oy)
            np.maximum(win[..., 0], C, out=win[..., 0])
            win[..., 2] *= 1.0 - 0.9 * np.clip(C, 0, 1)
        elif P[1] > y0 - 3.0:
            draw_ring(win, ox, oy, float(ppc), P, strip('outer'), strip('inner'), float(glow), float(engr),
                      float(heat))
            S = Strokes()
            ring_outline(S, P, 77)
            C, _ = pen.raster(S.pack(), 1e9, ppc, Hh, Ww, pen.INK, ox=ox, oy=oy)
            np.maximum(win[..., 0], C, out=win[..., 0])
            win[..., 2] *= 1.0 - 0.9 * np.clip(C, 0, 1)
        if not find and t > 0.6:
            self._fire(win, ox, oy, t, self.front, 1.0 + 18.0 * white, seed=31, grow=smooth((t - 0.6) / 0.5))
        ch[i0:i1, j0:j1] = win
        return self.tex.build()

    def _fire(self, win, ox, oy, t, tongues, flare, seed, grow=1.0):
        ppc = self.ppc
        Hh, Ww = win.shape[:2]
        gilt = np.zeros((Hh, Ww), np.float32)
        core = np.zeros((Hh, Ww), np.float32)
        S = Strokes()
        for k, (bx, by, Hk, Wk, lean) in enumerate(tongues):
            O = tongue(bx, by, Hk * grow, Wk, lean, t, k + seed)
            Q = np.round((O - [ox, oy]) * ppc * 16).astype(np.int32)
            cv2.fillPoly(gilt, [Q], 1.0, lineType=cv2.LINE_AA, shift=4)
            I = tongue(bx, by + 0.05, Hk * grow * 0.55, Wk * 0.45, lean, t + 0.05, k + seed)
            QI = np.round((I - [ox, oy]) * ppc * 16).astype(np.int32)
            cv2.fillPoly(core, [QI], 1.0, lineType=cv2.LINE_AA, shift=4)
            pen.line(S, O, 0.016, seed * 7 + k, smooth=0, lift=(2.0, 4.0))
        C, _ = pen.raster(S.pack(), 1e9, ppc, Hh, Ww, pen.INK, ox=ox, oy=oy)
        m = gilt > 0
        fl = 1.0 + 0.1 * math.sin(t * 13.0) + 0.06 * math.sin(t * 29.0 + 1.0)
        # overwrite what lies behind the flame: gold laid in the tongue, its heart glowing, the pen's outline
        win[..., 0] = np.where(m, np.maximum(C, 0.06 * gilt), np.maximum(win[..., 0], C))
        win[..., 2] = np.where(m, 0.55 * gilt * (1 - 0.9 * C), win[..., 2])
        win[..., 4] = np.where(m, (0.1 * gilt + 0.4 * core * fl) * flare, win[..., 4])
        win[..., 3] = np.where(m, 0.0, win[..., 3])

    def _embers(self, win, ox, oy, t, flare):
        ppc = self.ppc
        Hh, Ww = win.shape[:2]
        S = Strokes()
        glow = np.zeros((Hh, Ww), np.float32)
        for k, (x, y, r) in enumerate(self.embers):
            a = np.linspace(0, 2 * np.pi, 9)[:-1] + 0.3 * k
            rr = r * (1 + 0.18 * np.sin(3 * a + k))
            O = np.column_stack([x + 1.4 * rr * np.cos(a), y + 0.55 * rr * np.sin(a)])
            pen.line(S, O, 0.014, 300 + k, smooth=6, closed=True)
            Q = np.round((O - [ox, oy]) * ppc * 16).astype(np.int32)
            cv2.fillPoly(glow, [Q], 1.0, lineType=cv2.LINE_AA, shift=4)
        C, _ = pen.raster(S.pack(), 1e9, ppc, Hh, Ww, pen.INK, ox=ox, oy=oy)
        fl = 1.0 + 0.15 * math.sin(t * 7.0) + 0.1 * math.sin(t * 17.0)
        win[..., 0] = np.maximum(win[..., 0], C)
        win[..., 4] = np.maximum(win[..., 4], 0.9 * glow * fl * flare)
        win[..., 2] = np.maximum(win[..., 2], 0.3 * glow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['stills'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'review', 'v3'))
    ap.add_argument('--scale', type=float, default=1.0)
    a = ap.parse_args()
    import book_c as BC
    import look
    b3 = BC.Book3(int(1920 * a.scale), int(804 * a.scale))
    for name in ('find', 'test'):
        hdr, alpha = b3.ring_still(name)
        rgb = look.finish(hdr, exposure=1.15, bloom_strength=0.06, bloom_threshold=1.2, vignette_amount=0.32)
        p = os.path.join(a.out, 'map_ring_page_%s.png' % name)
        look.save_png(p, rgb)
        print(p)


if __name__ == '__main__':
    main()
