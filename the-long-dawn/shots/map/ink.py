"""Ink rasterisation in map space.

A tile covers map X from x0 to the right and Y from y0 downwards at `ppd` pixels per map degree.
Strokes are polylines with a radius and an ink density per vertex (map units), rasterised as
anti-aliased capsules combined with max() inside the ink layer, so joints are round and overlaps
don't double-darken (a pen line over a pen line is still one pen line).
"""
import numpy as np
from numba import njit, prange


class Strokes:
    """Accumulates polylines: points (N,2) map XY, radius (N,), density (N,)."""

    def __init__(self):
        self.P, self.R, self.D, self.lens = [], [], [], []

    def add(self, pts, rad, dens=1.0):
        pts = np.asarray(pts, np.float64)
        n = len(pts)
        if n < 2:
            return
        self.P.append(pts)
        self.R.append(np.broadcast_to(np.asarray(rad, np.float64), (n,)).copy())
        self.D.append(np.broadcast_to(np.asarray(dens, np.float64), (n,)).copy())
        self.lens.append(n)

    def extend(self, other):
        self.P += other.P
        self.R += other.R
        self.D += other.D
        self.lens += other.lens

    def arrays(self):
        if not self.P:
            z = np.zeros((0, 2))
            return z, np.zeros(0), np.zeros(0), np.zeros(1, np.int64)
        st = np.zeros(len(self.lens) + 1, np.int64)
        st[1:] = np.cumsum(self.lens)
        return np.concatenate(self.P), np.concatenate(self.R), np.concatenate(self.D), st

    def bbox(self):
        P = np.concatenate(self.P)
        return P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()


@njit(cache=True)
def _seg(A, ax, ay, bx, by, ra, rb, da, db):
    H, W = A.shape
    rmax = max(ra, rb, 0.5) + 1.0
    x0 = int(np.floor(min(ax, bx) - rmax))
    x1 = int(np.ceil(max(ax, bx) + rmax))
    y0 = int(np.floor(min(ay, by) - rmax))
    y1 = int(np.ceil(max(ay, by) + rmax))
    if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
        return
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, W - 1)
    y1 = min(y1, H - 1)
    dx = bx - ax
    dy = by - ay
    L2 = dx * dx + dy * dy
    for i in range(y0, y1 + 1):
        py = i - ay
        for j in range(x0, x1 + 1):
            px = j - ax
            if L2 > 1e-12:
                t = (px * dx + py * dy) / L2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
            else:
                t = 0.0
            qx = px - t * dx
            qy = py - t * dy
            d = np.sqrt(qx * qx + qy * qy)
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
            if v > A[i, j]:
                A[i, j] = v


@njit(cache=True)
def _draw(A, px, py, rad, dens, starts):
    for s in range(len(starts) - 1):
        for k in range(starts[s], starts[s + 1] - 1):
            _seg(A, px[k], py[k], px[k + 1], py[k + 1], rad[k], rad[k + 1], dens[k], dens[k + 1])


def draw(A, strokes, x0, y0, ppd, width_scale=1.0):
    """Rasterise Strokes into ink layer A (float32 HxW) for a tile at (x0, y0, ppd)."""
    P, R, D, st = strokes.arrays() if isinstance(strokes, Strokes) else strokes
    if len(P) == 0:
        return
    px = (P[:, 0] - x0) * ppd - 0.5
    py = (y0 - P[:, 1]) * ppd - 0.5
    H, W = A.shape
    # cull whole strokes outside the tile (cheap bbox test per stroke)
    _draw(A, px, py, R * ppd * width_scale, D, st)


@njit(cache=True)
def _dots(A, px, py, rad, dens):
    for k in range(len(px)):
        _seg(A, px[k], py[k], px[k] + 1e-3, py[k], rad[k], rad[k], dens[k], dens[k])


def dots(A, pts, rad, dens, x0, y0, ppd):
    px = (pts[:, 0] - x0) * ppd - 0.5
    py = (y0 - pts[:, 1]) * ppd - 0.5
    H, W = A.shape
    m = (px > -5) & (px < W + 5) & (py > -5) & (py < H + 5)
    _dots(A, px[m], py[m], rad[m] * ppd, dens[m])


# ------------------------------------------------------------ helpers ---

def chaikin(p, n=2, closed=False):
    """Corner-cutting smoothing."""
    p = np.asarray(p, np.float64)
    for _ in range(n):
        if closed:
            q = np.roll(p, -1, 0)
        else:
            q = p[1:]
            p0 = p[:-1]
        if closed:
            a = 0.75 * p + 0.25 * q
            b = 0.25 * p + 0.75 * q
            p = np.empty((2 * len(a), 2))
            p[0::2] = a
            p[1::2] = b
        else:
            a = 0.75 * p0 + 0.25 * q
            b = 0.25 * p0 + 0.75 * q
            mid = np.empty((2 * len(a), 2))
            mid[0::2] = a
            mid[1::2] = b
            p = np.concatenate([p[:1], mid, p[-1:]])
    return p


def resample(p, step, closed=False):
    """Resample a polyline at roughly uniform arc-length spacing `step`."""
    p = np.asarray(p, np.float64)
    if closed:
        p = np.concatenate([p, p[:1]])
    d = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, 1))
    s = np.concatenate([[0], np.cumsum(d)])
    L = s[-1]
    if L <= 0:
        return p[:1].repeat(2, 0)
    n = max(2, int(np.ceil(L / step)) + 1)
    t = np.linspace(0, L, n)
    x = np.interp(t, s, p[:, 0])
    y = np.interp(t, s, p[:, 1])
    out = np.stack([x, y], 1)
    if closed:
        out = out[:-1]
    return out


def arclen(p):
    d = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, 1))
    return np.concatenate([[0], np.cumsum(d)])
