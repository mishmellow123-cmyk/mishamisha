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

The last written leaf opts into `RingLeaf(gap_degrees=30)`: a static, open Ring at the page head,
with two cut ends and the canonical inscription unfinished at its leading end. `RingPage(open_band=True)`
also exposes the open geometry to the existing page scene. All default RingPage output remains unchanged.
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
def draw_ring(ch, ox, oy, ppc, P, SO, SI, glow, engr, heat, nth=360, ns=4, gap_degrees=0.0):
    """Paint the ring into the channel window ch (H x W x 6 over page cm [ox, ..] x [oy, ..]) back to front,
    overwriting: gilt with form hatching, the inscription cut in it (engr) or awake in fire (glow), heat in the gold.
    P: cx, cy, R, k, h, tr, wob, wph, pool, uo, ui.
    gap_degrees=0 takes the original closed-band path without changing its arithmetic."""
    if not math.isfinite(gap_degrees) or gap_degrees < 0.0 or gap_degrees >= 180.0:
        raise ValueError('gap_degrees must be finite and in [0, 180)')
    if gap_degrees > 0.0:
        _draw_open_ring(ch, ox, oy, ppc, P, SO, SI, glow, engr, heat, gap_degrees, nth, ns)
        return
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


def ring_outline(S, P, seed, width=0.02, gap_degrees=0.0):
    """The pen's outline of the ring: both edges of the rim, the front face's foot and its two sides, and the foot
    of the inner face where the hole shows it."""
    validate_gap(gap_degrees)
    if gap_degrees > 0.0:
        _open_outline(S, P, seed, width, gap_degrees)
        return
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


# =========================================================== open gold band ===

def validate_gap(gap_degrees):
    """Return a finite opening in degrees; 0 selects the exact legacy closed band."""
    if not math.isfinite(gap_degrees) or not 0.0 <= gap_degrees < 180.0:
        raise ValueError('gap_degrees must be finite and in [0, 180)')
    return float(gap_degrees)


@njit(cache=True)
def _half_glyph_u(S):
    """Mid-column of the first substantial ink run in the canonical strip, with ink on both sides.

    This aligns the leading metal end through an existing glyph rather than inventing another letter.
    The strip's empty inter-letter columns define the run; no asset pixel is changed.
    """
    start = -1
    for j in range(S.shape[1]):
        ink = np.max(S[:, j]) > 0.1
        if ink and start < 0:
            start = j
        if not ink and start >= 0:
            if j - start >= 4:
                return (0.5 * (start + j - 1) + 0.5) / S.shape[1]
            start = -1
    return 0.0


@njit(cache=True)
def _open_vertex(P, part, th, s):
    """Projected x/y and viewer depth. The top is z=0; the band's foot is z=-h.

    Both sides of each cut use these same corners, so the cut cannot float off the rim or the foot.
    """
    x, y = _corner(P, part, th, s)
    r = _radius(P, th)
    if part == 0:
        depth = (r - P[5]) * math.sin(th) - P[3] * s * P[4] * P[13]
    elif part == 3:
        _, top = _corner(P, 3, th, 0.0)
        depth = r * math.sin(th) - P[3] * (y - top)
    else:
        depth = (r - P[5] * (1.0 - s)) * math.sin(th)
    return x, y, depth


@njit(cache=True)
def _open_quad(ch, depth, ox, oy, ppc, Q, part, ta, tb, sa, sb, P, SO, SI, uo, glow, engr, heat, plate=False):
    """Depth-tested bilinear surface, including the inner wall revealed through the gap."""
    H, W = ch.shape[:2]
    x0 = max(int(math.floor((np.min(Q[:, 0]) - ox) * ppc)), 0)
    x1 = min(int(math.ceil((np.max(Q[:, 0]) - ox) * ppc)) + 1, W)
    y0 = max(int(math.floor((np.min(Q[:, 1]) - oy) * ppc)), 0)
    y1 = min(int(math.ceil((np.max(Q[:, 1]) - oy) * ppc)) + 1, H)
    for i in range(y0, y1):
        py = oy + (i + 0.5) / ppc
        for j in range(x0, x1):
            px = ox + (j + 0.5) / ppc
            u, v = _inv_bilinear(px, py, Q[0, 0], Q[0, 1], Q[1, 0], Q[1, 1],
                                 Q[2, 0], Q[2, 1], Q[3, 0], Q[3, 1])
            if not (-1e-6 <= u <= 1.0 + 1e-6 and -1e-6 <= v <= 1.0 + 1e-6):
                continue
            d = ((1-u)*Q[0, 2] + u*Q[1, 2])*(1-v) + ((1-u)*Q[3, 2] + u*Q[2, 2])*v
            if d < depth[i, j]:
                continue
            depth[i, j] = d
            th = ta + (tb-ta)*u
            s = sa + (sb-sa)*v
            st = math.sin(th)
            let = 0.0
            if part == 3:
                tone = 0.5*(1.0-abs(st))**1.5 + 0.12*s
                let = _sample(SO, uo + (math.pi-th)/(2*math.pi), s)
            elif part == 0:
                tone = 0.55 + 0.15*s
                let = _sample(SI, P[10] + (th-math.pi)/(2*math.pi), s)
            elif part == 4:
                # Fine diagonal tool marks across a flat cut, darker than the rolled rim.
                tone = 0.33 + 0.14*v
            else:
                tone = 0.08 + 0.1*(1.0-abs(st))
            arc = _radius(P, th)*th
            fr = (px + 0.32*py)/0.105 if part == 4 else arc/0.15 + 0.1*math.sin(s*5.0+th*17.0)
            fr -= math.floor(fr)
            duty = 0.11 if part == 4 else (min(max(tone-0.22, 0.0)*1.1, 0.42) if part == 3
                                                   else min(max(tone-0.35, 0.0)*0.9, 0.2))
            aa = 0.7/(ppc*(0.105 if part == 4 else 0.15))
            hat = min(max((0.5*duty-abs(fr-0.5))/aa+0.5, 0.0), 1.0) if part != 1 and duty > .02 else 0.0
            dark = max(let*engr, 0.85*hat)
            ch[i, j, 0] = max(dark, 0.18*tone)
            ch[i, j, 1] = 0.0
            ch[i, j, 2] = (1.0-0.92*dark)*(1.0-0.3*tone)
            if plate:
                # Gold leaf belongs chiefly to the narrow rolled rim. A small wash on the
                # upper-left shoulder leaves the inscription, inner hatching and cut faces in ink.
                gilt = .72 if part == 1 else 0.0
                if part == 3:
                    gilt = .38*math.exp(-((th-2.55)/.50)**4)*max(1.0-s/.68, 0.0)**1.4
                ch[i, j, 2] *= gilt
            ch[i, j, 3] = 0.0
            ch[i, j, 4] = let*glow + heat*(0.35+0.65*s if part == 3 else 0.5)*(1.0-dark)
            ch[i, j, 5] = 0.0


@njit(cache=True)
def _draw_open_ring(ch, ox, oy, ppc, P, SO, SI, glow, engr, heat, gap_degrees, nth=360, ns=4, plate=False):
    half = gap_degrees*math.pi/360.0
    lo, hi = math.pi/2-half, math.pi/2+half
    uo = _half_glyph_u(SO) - (math.pi-lo)/(2*math.pi)
    depth = np.full(ch.shape[:2], -1e30)
    # The visible inner back wall, all remaining top rim, and the two front shoulders.
    # Intervals end exactly on the cut, independently of the angular mesh resolution.
    for part, a0, a1 in ((0, math.pi, 2*math.pi), (1, hi, 2*math.pi), (1, 0.0, lo),
                        (3, 0.0, lo), (3, hi, math.pi)):
        n = max(1, int(math.ceil((a1-a0)*nth/(2*math.pi))))
        for a in range(n):
            ta, tb = a0+(a1-a0)*a/n, a0+(a1-a0)*(a+1)/n
            for b in range(ns):
                sa, sb = b/ns, (b+1)/ns
                Q = np.array((_open_vertex(P, part, ta, sa), _open_vertex(P, part, tb, sa),
                              _open_vertex(P, part, tb, sb), _open_vertex(P, part, ta, sb)))
                _open_quad(ch, depth, ox, oy, ppc, Q, part, ta, tb, sa, sb, P, SO, SI, uo, glow, engr, heat, plate)
    for end in range(2 if gap_degrees > 0.0 else 0):
        th = lo if end == 0 else hi
        it = _open_vertex(P, 0, th, 0.0)
        ot = _open_vertex(P, 3, th, 0.0)
        ib = _open_vertex(P, 0, th, 1.0)
        ob = _open_vertex(P, 3, th, 1.0)
        Q = np.array((it, ot, ob, ib))
        _open_quad(ch, depth, ox, oy, ppc, Q, 4, th, th, 0.0, 1.0, P, SO, SI, uo, glow, engr, heat, plate)


def _open_outline(S, P, seed, width, gap_degrees):
    half = math.radians(gap_degrees)*0.5
    lo, hi = math.pi/2-half, math.pi/2+half
    th = np.linspace(hi, 2*math.pi+lo, 241)
    for part, s, w in ((3, 0., width), (0, 0., width*.8)):
        pts = np.array([_corner(P, part, a, s) for a in th])
        pen.line(S, pts, w, seed+part, smooth=0, lift=(3., 6.))
    for idx, (a0, a1) in enumerate(((0., lo), (hi, math.pi))):
        aa = np.linspace(a0, a1, 81)
        pen.line(S, np.array([_corner(P, 3, a, 1.) for a in aa]), width*1.15,
                 seed+10+idx, smooth=0, lift=(3., 6.))
    for idx, a in enumerate((0., math.pi)):
        pen.line(S, np.array([_corner(P, 3, a, s) for s in (0., 1.)]), width, seed+12+idx, smooth=0)
    for idx, a in enumerate((lo, hi)):
        # Separate closed pen contours make both flat metal ends legible at small sizes.
        pts = np.array([_corner(P, part, a, s) for part, s in ((0, 0.), (3, 0.), (3, 1.), (0, 1.), (0, 0.))])
        pen.line(S, pts, width, seed+15+idx, smooth=0, lift=(2., 5.))
    # Back inner foot: hide it wherever a surviving front rim/wall lies closer to the viewer.
    run = []
    for a in np.linspace(math.pi, 2*math.pi, 181):
        x, y = _corner(P, 0, a, 1.)
        r = max(_radius(P, a)-P[5], 1e-6)
        front = math.acos(np.clip((x-P[0])/(r*P[12]), -1., 1.))
        _, front_top = _corner(P, 0, front, 0.)
        visible = y < front_top-.02 or lo < front < hi
        if visible:
            run.append((x, y))
        else:
            if len(run) > 3:
                pen.line(S, np.array(run), width*.7, seed+20, smooth=0)
            run = []
    if len(run) > 3:
        pen.line(S, np.array(run), width*.7, seed+20, smooth=0)


class RingLeaf:
    """A static drawing alone on a blank leaf; an explicit opt-in to the last-page artwork.

    `texture(t)` is compatible with redbook.Page. Placement and radius are page centimetres.
    There is no fire, wet ink or write-on; the facing leaf can remain entirely blank.
    """

    def __init__(self, ppc=70, seed=77, gap_degrees=30.0, center=(10.2, 7.2), radius=3.3):
        validate_gap(gap_degrees)
        if not math.isfinite(ppc) or ppc <= 0 or not math.isfinite(radius) or radius <= 0:
            raise ValueError('ppc and radius must be finite and positive')
        if len(center) != 2 or not np.isfinite(center).all():
            raise ValueError('center must be two finite page coordinates')
        self.ppc, self.gap_degrees = ppc, gap_degrees
        q = radius/3.3
        self.params = np.array([*center, radius, .4, 1.32*q, .46*q, 0., 1.3, 0., .12, .61, 0., 1., 1., 0.])
        self.tex = B.PageTex(PG.PW, PG.PH, ppc)
        # Raster only the drawing window; the rest of the leaf is exactly blank.
        j0 = max(0, int((center[0]-radius-.12)*ppc))
        j1 = min(self.tex.W, int(math.ceil((center[0]+radius+.12)*ppc)))
        i0 = max(0, int((center[1]-.4*radius-.12)*ppc))
        i1 = min(self.tex.H, int(math.ceil((center[1]+.4*radius+1.32*q+.12)*ppc)))
        if i1 <= i0 or j1 <= j0:
            raise ValueError('the Ring drawing must intersect the page')
        ch = np.zeros((i1-i0, j1-j0, B.NCH), np.float32)
        ox, oy = j0/ppc, i0/ppc
        draw_ring(ch, ox, oy, float(ppc), self.params, strip('outer'), strip('inner'), 0., 1., 0.,
                  gap_degrees=float(gap_degrees))
        S = Strokes()
        ring_outline(S, self.params, seed, gap_degrees=gap_degrees)
        ink, _ = pen.raster(S.pack(), 1e9, ppc, ch.shape[0], ch.shape[1], pen.INK, ox=ox, oy=oy)
        np.maximum(ch[..., 0], ink, out=ch[..., 0])
        ch[..., 2] *= 1.-.9*np.clip(ink, 0., 1.)
        self.tex.chan[i0:i1, j0:j1] = ch
        self.tex.build()

    def texture(self, t=0.0):
        return self.tex


class RingPlate:
    """The opt-in final-page plate, with the book's double rules and three lines of its hand.

    The 8 cm Ring occupies 40% of a 20 cm leaf. Its top ellipse has a .75 minor/major
    ratio before the book camera adds perspective. Gold is confined to the top rim and
    a small shoulder wash; canonical letters and form hatching use the normal ink channel.
    The untouched lower half of the leaf continues onto the facing blank page.
    """

    BOX = (3.3, 4.7, 16.7, 13.3)
    BLANK_FROM = 13.5

    def __init__(self, ppc=70, seed=77, gap_degrees=30.0, center=(10., 8.3), radius=4., aspect=.75):
        self.gap_degrees = validate_gap(gap_degrees)
        if not math.isfinite(ppc) or ppc <= 0.0:
            raise ValueError('ppc must be finite and positive')
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError('radius must be finite and positive')
        if not math.isfinite(aspect) or not .6 <= aspect <= 1.0:
            raise ValueError('the plate ellipse aspect must be in [.6, 1]')
        if len(center) != 2 or not np.isfinite(center).all():
            raise ValueError('center must be two finite page coordinates')
        self.ppc = ppc
        self.params = np.array([*center, radius, aspect, 1.25*radius/4., .44*radius/4.,
                                0., 1.3, 0., .12, .61, 0., 1., 1., 0.])
        x0, y0, x1, y1 = self.BOX
        if (center[0]-radius < x0+.3 or center[0]+radius > x1-.3
                or center[1]-radius*aspect < y0+.3
                or center[1]+radius*aspect+self.params[4] > y1-.3):
            raise ValueError('the Ring must fit inside the plate rules with a .3 cm margin')
        self.tex = B.PageTex(PG.PW, PG.PH, ppc)
        # The plate window is the only substantial temporary raster; the lower page stays zero.
        i1 = min(self.tex.H, int(math.ceil(self.BLANK_FROM*ppc)))
        ch = np.zeros((i1, self.tex.W, B.NCH), np.float32)
        _draw_open_ring(ch, 0., 0., float(ppc), self.params, strip('outer'), strip('inner'),
                        0., 1., 0., self.gap_degrees, plate=True)
        S = Strokes()
        ring_outline(S, self.params, seed, width=.023, gap_degrees=self.gap_degrees)
        PG.frame_rules(S, self.BOX, seed+31)
        hand = pen.Hand(seed=seed+41, xh=.20, nib=.20, thin=.03, dens=.88)
        hand.write_block(S, x0, 2.9, x1-x0, 3, .62, last_frac=.78)
        self.strokes = S
        ink, _ = pen.raster(S.pack(), 1e9, ppc, ch.shape[0], ch.shape[1], pen.INK)
        np.maximum(ch[..., 0], ink, out=ch[..., 0])
        ch[..., 2] *= 1.-.9*np.clip(ink, 0., 1.)
        self.tex.chan[:i1] = ch
        self.tex.build()

    def texture(self, t=0.0):
        return self.tex


# ================================================================ the fire ===

def tongue_geom(bx, by, H, W, lean, t, k):
    """A drawn flame tongue's centreline and half-width (page cm, y down): a broad body that narrows into a licking
    S-curve and a curled tip, alive (it sways, stretches and breathes)."""
    s = np.linspace(0.0, 1.0, 26)
    ph = 1.7 * k
    Hh = H * (1.0 + 0.08 * math.sin(t * 2.1 + ph) + 0.05 * math.sin(t * 5.3 + 2 * ph))
    sway = W * (0.55 * np.sin(2.6 * s - t * 2.7 + ph) * s ** 1.4 + 0.25 * np.sin(5.1 * s - t * 4.3 + 2 * ph) * s ** 2)
    curl = W * 0.9 * np.sign(math.sin(ph + 0.5)) * np.clip((s - 0.72) / 0.28, 0, 1) ** 2
    cxl = bx + lean * s * Hh + sway + curl
    cyl = by - s * Hh
    wd = W * (1.0 - s) ** 0.9 * (0.75 + 0.25 * np.sin(np.pi * np.minimum(s * 1.6, 1.0)))
    return s, cxl, cyl, wd


def tongue(bx, by, H, W, lean, t, k):
    """One drawn flame tongue as a closed outline."""
    s, cxl, cyl, wd = tongue_geom(bx, by, H, W, lean, t, k)
    left = np.column_stack([cxl - wd, cyl])
    right = np.column_stack([cxl + wd, cyl])[::-1]
    return np.vstack([left, right])


def tongue_lines(bx, by, H, W, lean, t, k, n=5):
    """The engraver's lines inside a tongue: strokes that follow its flow from the base and converge on its tip,
    the outer ones stopping short (MAP-L: a drawn flame, not a paper cut-out)."""
    s, cxl, cyl, wd = tongue_geom(bx, by, H, W, lean, t, k)
    out = []
    for f in np.linspace(-0.62, 0.62, n):
        smax = 0.9 - 0.42 * abs(f)
        m = (s >= 0.05) & (s <= smax)
        if m.sum() >= 3:
            out.append(np.column_stack([cxl[m] + f * wd[m], cyl[m]]))
    return out


def gn_(x, k):
    return np.array([gnoise(float(a), 0.37 * k, 61) for a in np.atleast_1d(x)])


class RingPage:
    """The right page with the drawing (text above and below it), re-drawn per frame in a window."""

    WIN = (2.8, 6.6, 17.8, 21.6)          # page cm (x0, y0, x1, y1) of the drawing
    FIRE = (10.2, 18.4)                   # the ember bed's centre
    BED = (10.2, 17.3)                    # where the ring lies in the fire

    def __init__(self, seed=7, ppc=130, *, open_band=False, gap_degrees=30.0):
        self.gap_degrees = validate_gap(gap_degrees) if open_band else 0.0
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
                      float(heat), gap_degrees=self.gap_degrees)
            S = Strokes()
            ring_outline(S, P, 77, gap_degrees=self.gap_degrees)
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
        heart = np.zeros((Hh, Ww), np.float32)
        for k, (bx, by, Hk, Wk, lean) in enumerate(tongues):
            O = tongue(bx, by, Hk * grow, Wk, lean, t, k + seed)
            Q = np.round((O - [ox, oy]) * ppc * 16).astype(np.int32)
            cv2.fillPoly(gilt, [Q], 1.0, lineType=cv2.LINE_AA, shift=4)
            Hq = tongue(bx, by + 0.03, Hk * grow * 0.62, Wk * 0.72, lean, t + 0.03, k + seed)
            QH = np.round((Hq - [ox, oy]) * ppc * 16).astype(np.int32)
            cv2.fillPoly(heart, [QH], 1.0, lineType=cv2.LINE_AA, shift=4)
            I = tongue(bx, by + 0.05, Hk * grow * 0.45, Wk * 0.42, lean, t + 0.05, k + seed)
            QI = np.round((I - [ox, oy]) * ppc * 16).astype(np.int32)
            cv2.fillPoly(core, [QI], 1.0, lineType=cv2.LINE_AA, shift=4)
            pen.line(S, O, 0.016, seed * 7 + k, smooth=0, lift=(2.0, 4.0))
            for j, L_ in enumerate(tongue_lines(bx, by, Hk * grow, Wk, lean, t, k + seed)):
                pen.line(S, L_, 0.0085, seed * 7 + 100 * k + j, smooth=0, lift=(3.0, 5.0), dens=0.85)
        C, _ = pen.raster(S.pack(), 1e9, ppc, Hh, Ww, pen.INK, ox=ox, oy=oy)
        m = gilt > 0
        fl = 1.0 + 0.1 * math.sin(t * 13.0) + 0.06 * math.sin(t * 29.0 + 1.0)
        # overwrite what lies behind the flame: the pen's outline and flow lines, gold laid only in the heart, the
        # heart glowing
        win[..., 0] = np.where(m, np.maximum(C * (1.0 - 0.45 * core), 0.04 * gilt), np.maximum(win[..., 0], C))
        win[..., 2] = np.where(m, 0.6 * heart * (1 - 0.85 * C), win[..., 2])
        win[..., 4] = np.where(m, (0.07 * gilt + 0.18 * heart + 0.5 * core * fl) * flare, win[..., 4])
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
