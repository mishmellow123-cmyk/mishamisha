"""THE RED BOOK: pen, pencil and gilt on a page.

Page space is centimetres, x to the right and y DOWN from the head of the page, so row = y * ppc and
column = x * ppc of a page array. A drawing is a list of strokes, each drawn on in time (t0..t1, seconds of
shot time) along its arc length the way a pen moves. Layers are max-combined (a pen line over a pen line
is still one line), and a fresh stroke stays wet (darker, glossy) for a moment before it dries.

Also here: the book hand (an invented broad-nib hand of its own, below, with the dip cycle of a real pen:
dark after each dip, paler as it runs dry), hatching that follows form and tone,
stipple, and the construction lines of a pencil underdrawing.
"""
import math
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'embers'))
from noise import wobble1d  # noqa: E402

INK, PENCIL, GILT, FIRE, RUBRIC = 0, 1, 2, 3, 4      # RUBRIC: the rubricator's vermilion (MAP-L)


# ================================================================ strokes ===

class Strokes:
    """Polylines in page cm with a radius and a density per vertex, a draw window (t0, t1) and a layer."""

    def __init__(self):
        self.P, self.R, self.D, self.T0, self.T1, self.L, self.G, self.N = [], [], [], [], [], [], [], []

    def add(self, pts, rad, dens=1.0, t0=0.0, t1=0.0, layer=INK, gain=1.0, nib=(0.0, 0.0)):
        """nib: half-vector (cm) of a broad nib swept along the path (thick/thin, sharp terminals); (0, 0) is
        a round pen whose radius is rad."""
        pts = np.asarray(pts, np.float64)
        n = len(pts)
        if n < 2:
            return
        self.P.append(pts)
        self.R.append(np.broadcast_to(np.asarray(rad, np.float64), (n,)).copy())
        self.D.append(np.broadcast_to(np.asarray(dens, np.float64), (n,)).copy())
        self.T0.append(float(t0))
        self.T1.append(float(t1))
        self.L.append(int(layer))
        self.G.append(float(gain))
        self.N.append((float(nib[0]), float(nib[1])))

    def extend(self, o):
        for k in ('P', 'R', 'D', 'T0', 'T1', 'L', 'G', 'N'):
            getattr(self, k).extend(getattr(o, k))

    def __len__(self):
        return len(self.P)

    def retime(self, t0, t1, order=None, overlap=0.0, weight='length'):
        """Give strokes consecutive draw windows filling [t0, t1] in the given order (default: as added);
        each stroke's share is proportional to its length. overlap > 0 lets the next stroke start before
        the previous one ends (a quick hand), as a fraction of the stroke's own window."""
        idx = list(range(len(self.P))) if order is None else list(order)
        if not idx:
            return
        L = np.array([arclen(self.P[i])[-1] for i in idx]) + 0.05
        if weight != 'length':
            L = np.ones_like(L)
        span = (t1 - t0) / (L.sum() * (1.0 - overlap) + L[-1] * overlap)
        t = t0
        for k, i in enumerate(idx):
            d = L[k] * span
            self.T0[i], self.T1[i] = t, t + d
            t += d * (1.0 - overlap)

    def pack(self):
        if not self.P:
            z = np.zeros(0)
            return dict(P=np.zeros((0, 2)), R=z, D=z, S=z, st=np.zeros(1, np.int64), T0=z, T1=z,
                        L=np.zeros(0, np.int64), G=z, bb=np.zeros((0, 4)), N=np.zeros((0, 2)))
        st = np.zeros(len(self.P) + 1, np.int64)
        st[1:] = np.cumsum([len(p) for p in self.P])
        S = []
        bb = np.zeros((len(self.P), 4))
        for k, p in enumerate(self.P):
            s = arclen(p)
            S.append(s / max(s[-1], 1e-9))
            r = self.R[k].max() + math.hypot(*self.N[k])
            bb[k] = (p[:, 0].min() - r, p[:, 1].min() - r, p[:, 0].max() + r, p[:, 1].max() + r)
        return dict(P=np.concatenate(self.P), R=np.concatenate(self.R), D=np.concatenate(self.D),
                    S=np.concatenate(S), st=st, T0=np.array(self.T0), T1=np.array(self.T1),
                    L=np.array(self.L, np.int64), G=np.array(self.G), bb=bb, N=np.array(self.N, np.float64))


@njit(cache=True)
def _seg(C, Wt, r0, r1, ax, ay, bx, by, ra, rb, da, db, wa, wb):
    H, W = C.shape
    rmax = max(ra, rb, 0.5) + 1.0
    x0 = int(math.floor(min(ax, bx) - rmax))
    x1 = int(math.ceil(max(ax, bx) + rmax))
    y0 = int(math.floor(min(ay, by) - rmax))
    y1 = int(math.ceil(max(ay, by) + rmax))
    if x1 < 0 or x0 >= W or y1 < r0 or y0 >= r1:
        return
    x0 = max(x0, 0)
    x1 = min(x1, W - 1)
    y0 = max(y0, r0)
    y1 = min(y1, r1 - 1)
    dx = bx - ax
    dy = by - ay
    L2 = dx * dx + dy * dy
    for i in range(y0, y1 + 1):
        py = i + 0.5 - ay
        for j in range(x0, x1 + 1):
            px = j + 0.5 - ax
            t = 0.0
            if L2 > 1e-12:
                t = (px * dx + py * dy) / L2
                t = min(max(t, 0.0), 1.0)
            qx = px - t * dx
            qy = py - t * dy
            d = math.sqrt(qx * qx + qy * qy)
            r = ra + (rb - ra) * t
            k = 1.0
            if r < 0.5:
                k = r / 0.5
                r = 0.5
            c = r - d + 0.5
            if c <= 0.0:
                continue
            if c > 1.0:
                c = 1.0
            v = c * k * (da + (db - da) * t)
            if v > C[i, j]:
                C[i, j] = v
            w = c * (wa + (wb - wa) * t)
            if w > Wt[i, j]:
                Wt[i, j] = w


@njit(cache=True, inline='always')
def _dseg(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    L2 = dx * dx + dy * dy
    t = 0.0
    if L2 > 1e-12:
        t = min(max(((px - ax) * dx + (py - ay) * dy) / L2, 0.0), 1.0)
    qx = px - ax - t * dx
    qy = py - ay - t * dy
    return math.sqrt(qx * qx + qy * qy)


@njit(cache=True)
def _quad(C, Wt, r0, r1, ax, ay, bx, by, nx, ny, ra, rb, da, db, wa, wb):
    """A broad nib (half-vector n) swept from a to b: the parallelogram a-n, a+n, b+n, b-n, grown by the
    hairline radius; anti-aliased by its exact distance."""
    H, W = C.shape
    th = max(ra, rb)
    k = 1.0
    if th < 0.5:
        k = th / 0.5
        th = 0.5
    qx = (ax - nx, ax + nx, bx + nx, bx - nx)
    qy = (ay - ny, ay + ny, by + ny, by - ny)
    m = th + 1.5
    x0 = int(math.floor(min(min(qx[0], qx[1]), min(qx[2], qx[3])) - m))
    x1 = int(math.ceil(max(max(qx[0], qx[1]), max(qx[2], qx[3])) + m))
    y0 = int(math.floor(min(min(qy[0], qy[1]), min(qy[2], qy[3])) - m))
    y1 = int(math.ceil(max(max(qy[0], qy[1]), max(qy[2], qy[3])) + m))
    if x1 < 0 or x0 >= W or y1 < r0 or y0 >= r1:
        return
    x0 = max(x0, 0)
    x1 = min(x1, W - 1)
    y0 = max(y0, r0)
    y1 = min(y1, r1 - 1)
    ex = bx - ax
    ey = by - ay
    L2 = ex * ex + ey * ey
    for i in range(y0, y1 + 1):
        py = i + 0.5
        for j in range(x0, x1 + 1):
            px = j + 0.5
            dmin = 1e9
            pos = 0
            neg = 0
            for e in range(4):
                e2 = (e + 1) % 4
                d = _dseg(px, py, qx[e], qy[e], qx[e2], qy[e2])
                if d < dmin:
                    dmin = d
                cr = (qx[e2] - qx[e]) * (py - qy[e]) - (qy[e2] - qy[e]) * (px - qx[e])
                if cr > 0:
                    pos += 1
                elif cr < 0:
                    neg += 1
            sd = -dmin if (pos == 0 or neg == 0) else dmin
            c = th - sd + 0.5
            if c <= 0.0:
                continue
            if c > 1.0:
                c = 1.0
            t = 0.0
            if L2 > 1e-12:
                t = min(max(((px - ax) * ex + (py - ay) * ey) / L2, 0.0), 1.0)
            v = c * k * (da + (db - da) * t)
            if v > C[i, j]:
                C[i, j] = v
            w = c * (wa + (wb - wa) * t)
            if w > Wt[i, j]:
                Wt[i, j] = w


@njit(cache=True, parallel=True)
def _raster(C, Wt, P, R, D, S, st, T0, T1, LY, G, NB, bb, t, layer, ppc, ox, oy, dry, nb):
    H, W = C.shape
    band = (H + nb - 1) // nb
    for b in prange(nb):
        r0 = b * band
        r1 = min(H, r0 + band)
        if r0 >= r1:
            continue
        for s in range(len(st) - 1):
            if LY[s] != layer or t < T0[s] or G[s] <= 0.0:
                continue
            if (bb[s, 3] - oy) * ppc < r0 - 4 or (bb[s, 1] - oy) * ppc > r1 + 4:
                continue
            if (bb[s, 2] - ox) * ppc < -4 or (bb[s, 0] - ox) * ppc > W + 4:
                continue
            dur = T1[s] - T0[s]
            f = 1.0
            if dur > 1e-9:
                f = min(1.0, (t - T0[s]) / dur)
            g = G[s]
            for k in range(st[s], st[s + 1] - 1):
                s0 = S[k]
                s1 = S[k + 1]
                if s0 > f:
                    break
                ax = (P[k, 0] - ox) * ppc
                ay = (P[k, 1] - oy) * ppc
                bx = (P[k + 1, 0] - ox) * ppc
                by = (P[k + 1, 1] - oy) * ppc
                ra = R[k] * ppc
                rb = R[k + 1] * ppc
                da = D[k] * g
                db = D[k + 1] * g
                tip = False
                if s1 > f:
                    u = (f - s0) / max(s1 - s0, 1e-9)
                    bx = ax + (bx - ax) * u
                    by = ay + (by - ay) * u
                    rb = ra + (rb - ra) * u
                    db = da + (db - da) * u
                    s1 = f
                    tip = True
                wa = 0.0
                wb = 0.0
                if dry > 0.0:
                    wa = max(0.0, 1.0 - (t - (T0[s] + s0 * dur)) / dry)
                    wb = max(0.0, 1.0 - (t - (T0[s] + s1 * dur)) / dry)
                nx = NB[s, 0] * ppc
                ny = NB[s, 1] * ppc
                if nx != 0.0 or ny != 0.0:
                    _quad(C, Wt, r0, r1, ax, ay, bx, by, nx, ny, ra, rb, da, db, wa, wb)
                else:
                    _seg(C, Wt, r0, r1, ax, ay, bx, by, ra, rb, da, db, wa, wb)
                if tip:           # the bead of wet ink at the nib
                    rr = max(rb * 1.3, 0.45 * math.sqrt(nx * nx + ny * ny))
                    _seg(C, Wt, r0, r1, bx, by, bx + 1e-3, by, rr, rr, db, db, 1.0, 1.0)


def raster(pk, t, ppc, H, W, layer=INK, ox=0.0, oy=0.0, dry=0.0, gain=None):
    """Rasterise the strokes of one layer drawn by time t into (coverage*density, wetness) float32 HxW
    arrays covering page cm [ox, ox + W/ppc] x [oy, oy + H/ppc]. gain overrides the per-stroke gains."""
    C = np.zeros((H, W), np.float32)
    Wt = np.zeros((H, W), np.float32)
    if len(pk['T0']) == 0:
        return C, Wt
    G = pk['G'] if gain is None else np.asarray(gain, np.float64)
    _raster(C, Wt, pk['P'], pk['R'], pk['D'], pk['S'], pk['st'], pk['T0'], pk['T1'], pk['L'], G, pk['N'], pk['bb'],
            float(t), int(layer), float(ppc), float(ox), float(oy), float(dry), 96)
    return C, Wt


# ============================================================ polylines ===

def resample(p, step):
    p = np.asarray(p, np.float64)
    d = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, 1))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] <= 1e-9:
        return p[:1].repeat(2, 0)
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def arclen(p):
    d = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, 1))
    return np.concatenate([[0], np.cumsum(d)])


def normals(p):
    d = np.gradient(p, axis=0)
    L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    return np.stack([-d[:, 1] / L, d[:, 0] / L], 1)


def catmull(pts, n=16, closed=False):
    P = np.asarray(pts, np.float64)
    if len(P) < 2:
        return P
    if closed:
        P = np.vstack([P[-1:], P, P[:2]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    out.append(P[-2][None, :])
    return np.vstack(out)


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def hand(pts, width, seed, step=0.012, slow=(2.6, 0.012), fast=(0.28, 0.0025), taper=(0.05, 0.12),
         dens=1.0, pool=0.3, thin_end=0.25, press=0.15):
    """A pen line through pts (page cm): hand tremor along the normal, a pressure profile that swells in
    the middle and thins at the ends, ink pooling where the nib lands and lifts.
    Returns (pts, radius, density)."""
    p = resample(pts, step)
    s = arclen(p)
    L = max(s[-1], 1e-6)
    nrm = normals(p)
    off = wobble1d(s, seed, slow[0], slow[1]) + wobble1d(s, seed + 7, fast[0], fast[1])
    p = p + nrm * off[:, None]
    a = _ss(s / max(taper[0], 1e-4)) if taper[0] > 0 else np.ones_like(s)
    b = _ss((L - s) / max(taper[1], 1e-4)) if taper[1] > 0 else np.ones_like(s)
    prof = thin_end + (1.0 - thin_end) * np.sqrt(a * b)
    swell = 1.0 + press * wobble1d(s, seed + 11, max(0.8, L * 0.7), 1.0)
    rad = 0.5 * width * prof * swell
    d = dens * (1.0 + pool * np.exp(-s / 0.035) + 0.6 * pool * np.exp(-(L - s) / 0.03))
    return p, rad, np.minimum(d, 1.25)


def broad_nib(p, nib, thin, angle):
    """Width of a broad nib held at `angle` (radians, page space with y down) along a path."""
    d = np.gradient(p, axis=0)
    phi = np.arctan2(d[:, 1], d[:, 0])
    return nib * np.abs(np.sin(phi - angle)) + thin


# ================================================================ script ===

# THE BOOK HAND (H5): a human calligraphic hand of its own, and deliberately not the Ring's (the script of fire,
# `assets/ring`, is the fire's own: our writing goes into the fire, and what comes out is a script no one can
# read). Its letters stand apart and are built from a small palette of broad-nib strokes: the lying wave, the
# open wedge, the low heavy fall, the fork of a feather's tail, the small coil, the lozenge and the nib's own
# lozenge dot. It has no stems, bowls, arches, cups or hooks, no marks over its letters, and nothing drawn from
# Latin, Greek, Cyrillic, Tengwar or runes (the H5 critic read l, u and y in the first hand).
def coil(cx, cy, r, turns=1.1, a0=0.0, cw=True, n=8, inward=False):
    """A small spiral from its rim (at angle a0) winding inward (inward=True) or the reverse."""
    s = -1.0 if cw else 1.0
    pts = []
    for k in range(n + 1):
        u = k / n
        a = a0 + s * u * turns * 2 * math.pi
        rr = r * (1.0 - 0.72 * u)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts if inward else pts[::-1]


def lozenge(x0, y0, w, h, n=3):
    """Sharp lozenge from its left corner, anticlockwise; n knots per side keep the sides straight."""
    C = [(x0, y0), (x0 + w / 2, y0 - h / 2), (x0 + w, y0), (x0 + w / 2, y0 + h / 2), (x0, y0)]
    pts = []
    for (ax, ay), (bx, by) in zip(C[:-1], C[1:]):
        for k in range(n):
            u = k / n
            pts.append((ax + (bx - ax) * u, ay + (by - ay) * u))
    pts.append(C[-1])
    return pts


def dot(x, y):
    return [(x, y), (x + 0.05, y + 0.035)]


def fine(pts, k=0.55):
    return (pts, k)


BOOK = {
    # a lying wave
    'wa': dict(w=1.1, s=[[(0.0, 0.4), (0.14, 0.56), (0.34, 0.54), (0.56, 0.3), (0.78, 0.18), (0.98, 0.3), (1.08, 0.46)]]),
    # a scroll lying on the line: one end rolled up, the other rolled down
    'lc': dict(w=1.3, s=[coil(0.14, 0.34, 0.13, 1.0, math.pi * 1.5, cw=True) + [(0.36, 0.2), (0.66, 0.16), (0.96, 0.2)]
                         + coil(1.12, 0.06, 0.13, 1.0, math.pi * 0.5, cw=True, inward=True)]),
    # a small coil and a long low stroke out of it, ending in the nib's dot
    'cw': dict(w=1.3, s=[coil(0.16, 0.42, 0.13, 1.1, math.pi * 1.9, cw=False) + [(0.44, 0.34), (0.8, 0.3), (1.08, 0.26)],
                         dot(1.16, 0.24)]),
    # a heavy low fall with a hairline entry and a flick at its foot
    'hk': dict(w=0.95, s=[[(0.0, 0.6), (0.18, 0.66), (0.4, 0.44), (0.7, 0.12), (0.86, 0.02), (0.98, 0.08)]]),
    # an open wedge, a coil at its upper tip
    'wl': dict(w=1.05, s=[coil(0.9, 0.66, 0.1, 1.0, math.pi * 0.2, cw=False) + [(0.52, 0.52), (0.06, 0.34), (0.52, 0.16),
                                                                                 (1.0, 0.02)]]),
    # an open wedge with the nib's lozenge in its mouth
    'wt': dict(w=1.05, s=[[(1.0, 0.68), (0.5, 0.52), (0.04, 0.35), (0.5, 0.18), (1.0, 0.02)], dot(0.56, 0.33)]),
    # a feather's tail: a nib dot, a long flat stroke forking at its end
    'fk': dict(w=1.2, s=[dot(0.0, 0.34), [(0.14, 0.36), (0.5, 0.4), (0.86, 0.36)],
                         [(0.84, 0.37), (0.98, 0.54), (1.14, 0.62)], [(0.84, 0.35), (0.98, 0.18), (1.16, 0.1)]]),
    # two lozenges bridged by a hairline
    'll': dict(w=1.4, s=[fine(lozenge(0.0, 0.34, 0.56, 0.66)), [(0.56, 0.34), (0.7, 0.38), (0.84, 0.34)],
                         fine(lozenge(0.84, 0.34, 0.5, 0.56))]),
    # a lozenge pierced by a wave
    'lp': dict(w=1.35, s=[[(0.0, 0.3), (0.2, 0.42), (0.42, 0.34), (0.66, 0.26), (0.9, 0.36), (1.12, 0.46), (1.32, 0.38)],
                          fine(lozenge(0.36, 0.34, 0.6, 0.74))]),
    # a lozenge sitting on a flat leaf
    'ls': dict(w=1.2, s=[[(0.0, 0.08), (0.3, 0.02), (0.6, 0.0), (0.9, 0.02), (1.18, 0.08)], fine(lozenge(0.3, 0.42, 0.58, 0.68))]),
    # a heavy fall from a hairline entry, through the line, ending in the nib's dot (descends)
    'df': dict(w=1.35, s=[[(0.0, 0.58), (0.14, 0.66), (0.42, 0.46), (0.8, 0.06), (1.12, -0.3), (1.24, -0.44)],
                          dot(1.28, -0.54)]),
    # a coil with a short stroke forking out of it forward, like a bud
    'cf': dict(w=1.2, s=[coil(0.16, 0.34, 0.14, 1.1, 0.0, cw=False) + [(0.44, 0.36), (0.74, 0.36)],
                         [(0.72, 0.37), (0.86, 0.56), (1.02, 0.64)], [(0.72, 0.35), (0.88, 0.16), (1.04, 0.1)]]),
    # a wave whose end rolls under itself
    'wv': dict(w=1.15, s=[[(0.0, 0.34), (0.2, 0.54), (0.44, 0.46), (0.7, 0.28), (0.92, 0.3), (1.04, 0.44)]
                          + coil(0.9, 0.3, 0.14, 1.0, math.pi * 0.2, cw=True, inward=True)]),
    # the plain lozenge
    'lo': dict(w=0.75, s=[fine(lozenge(0.0, 0.34, 0.7, 0.7))]),
    # an open wedge whose lower arm rolls into a coil
    'wc': dict(w=1.15, s=[[(1.0, 0.7), (0.5, 0.52), (0.04, 0.35), (0.5, 0.16), (0.9, 0.04)]
                          + coil(1.0, 0.14, 0.12, 1.0, math.pi * 1.4, cw=False, inward=True)]),
    # a lozenge, and from its forward corner a heavy fall through the line to the nib's dot (descends)
    'dh': dict(w=1.45, s=[fine(lozenge(0.0, 0.5, 0.36, 0.4)), [(0.36, 0.5), (0.62, 0.3), (0.96, -0.08), (1.26, -0.4)],
                          dot(1.32, -0.5)]),
}
GLYPHS = BOOK
TALL = set()
KEYS = sorted(GLYPHS.keys())
# letter frequencies: a real language leans on a few letters
_FREQ = np.array([{'wa': 1.6, 'hk': 1.4, 'cw': 1.3, 'lc': 1.3, 'wv': 1.2, 'wl': 0.8, 'wt': 0.6, 'wc': 0.6, 'll': 0.5,
                   'lo': 0.5, 'df': 0.7, 'dh': 0.6}.get(k, 1.0) for k in KEYS])
_FREQ = _FREQ / _FREQ.sum()
# no two waves side by side (they would read as m or n), nor two wedges
FAMILY = {'wa': 'wave', 'wv': 'wave', 'cw': 'wave', 'lc': 'wave', 'lp': 'wave', 'wl': 'wedge', 'wt': 'wedge',
          'wc': 'wedge', 'hk': 'fall', 'df': 'fall', 'dh': 'fall'}
JOIN_P = 0.18              # now and then a hairline carries one letter into the next
MARK_P = 0.0               # no marks over the letters (they read as Latin accents)
MARKS = {}


def _knots(st):
    return st[0] if isinstance(st, tuple) else st


def _nibk(st):
    return st[1] if isinstance(st, tuple) else 1.0


class Hand:
    """A scribe: writes lines of the invented script with a broad nib and a dip pen's ink cycle."""

    def __init__(self, seed=1, xh=0.2, nib=0.2, thin=0.03, angle=-35.0, slant=0.07, gap=0.14, space=0.7,
                 dens=0.95, layer=INK):
        self.rng = np.random.default_rng(seed)
        self.xh, self.nib, self.thin = xh, nib, thin
        self.angle = math.radians(angle)          # page space (y down): -38 deg == the Ring's 38 deg nib
        self.slant, self.gap, self.space, self.dens, self.layer = slant, gap, space, dens, layer
        self.ink = 1.0                            # ink left in the pen
        self.run = int(self.rng.integers(20, 36))
        self.seed = seed
        self.words = []            # (first stroke, end stroke, x_left, x_right, baseline, line)
        self.line_no = 0

    def word(self, n=None):
        r = self.rng
        n = n or int(r.choice([1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7], 1)[0])
        w, prev = [], None
        for _ in range(n):
            g = KEYS[int(r.choice(len(KEYS), p=_FREQ))]
            while g == prev or (prev is not None and FAMILY.get(g, g) == FAMILY.get(prev, prev)):
                g = KEYS[int(r.choice(len(KEYS), p=_FREQ))]
            w.append(g)
            prev = g
        return w

    def width(self, w):
        return sum(GLYPHS[g]['w'] for g in w) + self.gap * (len(w) - 1)

    def _dip(self):
        self.run -= 1
        if self.run <= 0:
            self.ink = 1.0
            self.run = int(self.rng.integers(20, 36))
        else:
            self.ink = max(0.66, self.ink - self.rng.uniform(0.008, 0.014))

    def write_word(self, S, w, x, base, t0=0.0, t1=0.0, scale=1.0):
        """Write word w (glyph keys) with its left end at page (x, base); returns the x of its right end.
        Timing: the letters of the word are drawn in turn across [t0, t1]."""
        r = self.rng
        em = self.xh * scale
        sl = self.slant
        strokes, hair = [], []
        cx = 0.0
        prev_exit = None
        marks = []
        for g in w:
            G = GLYPHS[g]
            sx = 1.0 + r.uniform(-0.05, 0.05)
            sy = 1.0 + r.uniform(-0.05, 0.05)
            glyph = [[(cx + px * sx, py * sy) for px, py in _knots(st)] for st in G['s']]
            nibs = [_nibk(st) for st in G['s']]
            ent = glyph[0][0]
            if prev_exit is not None and r.random() < JOIN_P:
                hair.append([prev_exit, ((prev_exit[0] + ent[0]) / 2, 0.5 + r.uniform(-0.05, 0.12)), ent])
            prev_exit = glyph[-1][-1]
            strokes.append((glyph, self.ink, nibs))
            if g not in TALL and r.random() < MARK_P:
                mk = list(MARKS.keys())[int(r.integers(0, len(MARKS)))]
                mx = cx + G['w'] * sx * r.uniform(0.3, 0.6)
                my = 1.3 + r.uniform(0.0, 0.2)
                marks.append(([[(mx + a * 1.3, my + b * 1.3) for a, b in st] for st in MARKS[mk]], self.ink))
            cx += G['w'] * sx + self.gap
            self._dip()
        n_all = sum(len(gl) for gl, _, _ in strokes) + len(hair) + sum(len(m) for m, _ in marks)
        dt = (t1 - t0) / max(n_all, 1)
        k = 0

        def emit(path, nib, thin, ink, th=None):
            nonlocal k
            P = catmull(path, 10)
            P = np.column_stack([x + (P[:, 0] + sl * P[:, 1]) * em, base - P[:, 1] * em])
            if len(P) < 2:
                return
            s = arclen(P)
            Ls = max(s[-1], 1e-6)
            # a nib lands a little heavy and lifts clean
            d = self.dens * ink * (1.0 + 0.22 * np.exp(-s / (0.35 * em)) + 0.1 * np.exp(-(Ls - s) / (0.2 * em)))
            ta = t0 + k * dt
            hn = 0.5 * nib * em
            S.add(P, 0.5 * thin * em, np.minimum(d, 1.2), ta, ta + dt, self.layer,
                  nib=(hn * math.cos(self.angle), hn * math.sin(self.angle)))
            k += 1

        k_first = len(S)
        for gl, ink, nibs in strokes:
            for st, nk in zip(gl, nibs):
                emit(st, self.nib * nk, self.thin, ink)
        for h in hair:
            emit(h, self.nib * 0.16, self.thin * 0.8, 0.8 * self.ink)
        for m, ink in marks:
            for st in m:
                emit(st, self.nib * 0.8, self.thin, ink)
        xr = x + (cx - self.gap) * em
        self.words.append((k_first, len(S), x, xr, base, self.line_no))
        return xr

    def write_block(self, S, x0, y0, width, n_lines, lead, t0=0.0, t1=0.0, last_frac=None, indent_first=False,
                    skip=None, wavy=0.03):
        """Lines of text in the box [x0, x0 + width] starting with the first baseline at y0, `lead` apart.
        The last line runs to last_frac of the width (a paragraph's end). skip(line, x_left, x_right) may
        return a list of (xa, xb) intervals to leave clear (a drawing in the text). Returns line records."""
        r = self.rng
        em = self.xh
        recs = []
        tl = (t1 - t0) / max(n_lines, 1)
        for li in range(n_lines):
            self.line_no = li
            base = y0 + li * lead + r.uniform(-0.012, 0.012)
            lw = width if (last_frac is None or li < n_lines - 1) else width * last_frac
            xa = x0 + (1.6 * em if (indent_first and li == 0) else 0.0)
            clear = skip(li, xa, x0 + lw) if skip else []
            # collect words until the line is full, then justify by the word spaces
            segs = []
            lo = xa
            for (ca, cb) in sorted(clear) + [(x0 + lw, x0 + lw)]:
                if ca > lo + 0.3:
                    segs.append((lo, ca - 0.15))
                lo = max(lo, cb + 0.15)
            words_line = []
            for (sa, sb) in segs:
                ws, cur = [], 0.0
                while True:
                    w = self.word()
                    ww = self.width(w) * em
                    need = ww if not ws else cur + self.space * em + ww
                    if need > sb - sa:
                        break
                    ws.append(w)
                    cur = need
                if not ws:
                    continue
                full = (last_frac is None or li < n_lines - 1) or len(segs) > 1
                extra = (sb - sa - cur) / max(len(ws) - 1, 1) if (full and len(ws) > 1) else 0.0
                extra = min(extra, 1.2 * em)
                words_line.append((sa, ws, extra))
            nw = sum(len(ws) for _, ws, _ in words_line)
            k = 0
            xr = xa
            for sa, ws, extra in words_line:
                xx = sa
                for w in ws:
                    ta = t0 + li * tl + tl * k / max(nw, 1)
                    tb = t0 + li * tl + tl * (k + 1) / max(nw, 1)
                    bb = base + wavy * math.sin(xx * 0.9 + li * 1.7) * 0.3
                    xr = self.write_word(S, w, xx, bb, ta, tb)
                    xx = xr + self.space * em + extra
                    k += 1
            recs.append(dict(base=base, x_end=xr))
        return recs


# ============================================================== hatching ===

def hatch(S, inside, tone, bbox, angle, spacing, thr, width, seed, dens=0.9, seg=(0.6, 2.2), jitter=0.25,
          gap=0.08, step=0.01, layer=INK, overshoot=0.03, wob=0.006, taper=(0.03, 0.12), curve=None,
          min_len=0.05):
    """Parallel hatching at `angle` (degrees, page space) and `spacing` (cm) inside bbox (x0, y0, x1, y1):
    lines are kept where inside(x, y) and tone(x, y) > thr, broken into strokes of seg (min, max) cm with
    small gaps, like a hand laying one line after another. curve(u, v) may bend the lines (returns an
    offset along the line normal in cm). Returns the number of strokes added."""
    rng = np.random.default_rng(seed)
    a = math.radians(angle)
    d = np.array([math.cos(a), math.sin(a)])
    nrm = np.array([-d[1], d[0]])
    x0, y0, x1, y1 = bbox
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    R = 0.5 * math.hypot(x1 - x0, y1 - y0) + 0.1
    offs = np.arange(-R, R, spacing) + rng.uniform(0, spacing)
    u = np.arange(-R, R, step)
    n = 0
    for o in offs:
        oj = o + rng.normal(0, jitter * spacing * 0.3)
        vv = np.full_like(u, oj)
        if curve is not None:
            vv = vv + curve(u, oj)
        px = cx + u * d[0] + vv * nrm[0]
        py = cy + u * d[1] + vv * nrm[1]
        ok = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        if not ok.any():
            continue
        m = np.zeros_like(ok)
        m[ok] = inside(px[ok], py[ok]) & (tone(px[ok], py[ok]) > thr + rng.normal(0, 0.02))
        if not m.any():
            continue
        # intervals of m
        dm = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        starts = np.nonzero(dm == 1)[0]
        ends = np.nonzero(dm == -1)[0] - 1
        for sa, sb in zip(starts, ends):
            ua, ub = u[sa] - rng.uniform(-overshoot, overshoot), u[sb] + rng.uniform(-overshoot, overshoot)
            while ub - ua > min_len:
                ln = rng.uniform(*seg)
                uc = min(ub, ua + ln)
                if ub - uc < 0.3 * seg[0]:
                    uc = ub
                k = np.linspace(ua, uc, max(3, int((uc - ua) / 0.03)))
                vk = np.full_like(k, oj) if curve is None else oj + curve(k, oj)
                pts = np.column_stack([cx + k * d[0] + vk * nrm[0], cy + k * d[1] + vk * nrm[1]])
                p, rr, dd = hand(pts, width * rng.uniform(0.85, 1.15), int(rng.integers(1 << 30)),
                                 slow=(2.0, wob), fast=(0.25, wob * 0.35), taper=taper,
                                 dens=dens * rng.uniform(0.85, 1.05), pool=0.18, thin_end=0.2)
                S.add(p, rr, dd, layer=layer)
                n += 1
                ua = uc + rng.uniform(0.3, 1.0) * gap
    return n


def stipple(S, inside, tone, bbox, density, rad, seed, dens=0.85, layer=INK):
    """Dots whose probability follows tone (dots per cm^2 at tone 1)."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bbox
    n = int(density * (x1 - x0) * (y1 - y0))
    if n <= 0:
        return
    px = rng.uniform(x0, x1, n)
    py = rng.uniform(y0, y1, n)
    keep = inside(px, py) & (rng.random(n) < tone(px, py))
    for x, y in zip(px[keep], py[keep]):
        r = rad * rng.uniform(0.6, 1.3)
        a = rng.uniform(0, 6.28)
        pts = np.array([[x, y], [x + 0.35 * r * math.cos(a), y + 0.35 * r * math.sin(a)]])
        S.add(pts, r, dens * rng.uniform(0.75, 1.05), layer=layer)


def line(S, pts, width, seed, dens=0.95, layer=INK, lift=(1.5, 4.0), smooth=8, closed=False, **kw):
    """A contour drawn as a few pen strokes (the pen lifts every lift=(min, max) cm and overlaps a little)."""
    rng = np.random.default_rng(seed)
    P = catmull(pts, smooth, closed) if smooth else np.asarray(pts, np.float64)
    P = resample(P, 0.02)
    s = arclen(P)
    L = s[-1]
    a = 0.0
    k = 0
    while a < L - 1e-3:
        b = min(L, a + rng.uniform(*lift))
        if L - b < 0.4 * lift[0]:
            b = L
        m = (s >= a - 1e-9) & (s <= b + 1e-9)
        seg = P[m]
        if len(seg) >= 2:
            p, r, d = hand(seg, width * rng.uniform(0.9, 1.1), seed * 31 + k, dens=dens * rng.uniform(0.9, 1.04),
                           **kw)
            S.add(p, r, d, layer=layer)
        k += 1
        a = b - (rng.uniform(0.0, 0.04) if b < L else 0.0)
    return L
