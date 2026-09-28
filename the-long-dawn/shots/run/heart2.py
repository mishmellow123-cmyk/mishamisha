"""THE HEART OF LIGHT, contract v2: the A17 -> A18 match cut's shared core (EMBERS-A3's 4879 | the crossing's 4880).

v1 (kept exactly as the base): centre (959.5, 401.5) at A cut 4879/4880; ice-white (0.80, 0.92, 1.00); gaussians of
sigma 10 / 26 / 114 px with peaks 30 / 0.9 / 0.05 at full resolution.
v2 adds, identically on both sides of the cut, as functions of A's CUT frame f only (no noise calls, so 4879 -> 4880
is continuous):
  flicker  I(f) = 1 + 0.06 sin(2 pi f 7.3/24) + 0.04 sin(2 pi f 11.9/24 + 1.3) + 0.03 sin(2 pi f 3.1/24 + 2.2),
           multiplying the core's peak only (the 26 / 114 px glows stay steady);
  flame    the core stretched upright (sigma_x 9, sigma_y 12 px), its centre 3 px above the heart, its upper half
           licking sideways by dx(f) = 2.0 sin(2 pi f 1.3/24 + 0.7) px (a smooth shear from the centre up);
  edge     a faint warm halo (1.00, 0.72, 0.40), sigma 18 px, peak 0.10, under the white.
Every size scales with `u` = the heart's core sigma in the caller's pixels (10 at full resolution on 4879/4880;
15 in a 1.5x supersampled source buffer), so the same call works in any buffer and as a camera draws back.

  draw(img, zb, sx, sy, u, f_cut, col, I=1.0, z=1e9, zbias=0.4)   # all four terms, additive, depth-tested
  core_weight(dx, dy, u, f_cut)                                     # the flame core's unit-peak profile (numpy)
"""
import math

import numpy as np
from numba import njit, prange

EDGE_COL = np.array([1.00, 0.72, 0.40])


def flicker(f_cut):
    return 1.0 + 0.06 * math.sin(2 * math.pi * f_cut * 7.3 / 24.0) \
        + 0.04 * math.sin(2 * math.pi * f_cut * 11.9 / 24.0 + 1.3) \
        + 0.03 * math.sin(2 * math.pi * f_cut * 3.1 / 24.0 + 2.2)


def lick(f_cut):
    """The flame's sideways lick at the top, in units of the core sigma (2 px at u = 10)."""
    return 0.2 * math.sin(2 * math.pi * f_cut * 1.3 / 24.0 + 0.7)


def core_weight(dx, dy, u, f_cut):
    """Unit-peak flame profile at pixel offsets (dx right, dy DOWN) from the heart centre, for a core of sigma u."""
    dx = np.asarray(dx, np.float64)
    dy = np.asarray(dy, np.float64)
    sxg, syg = 0.9 * u, 1.2 * u
    v = -(dy + 0.3 * u)                              # height above the flame's centre (3 px above the heart)
    c = np.clip(v / syg, 0.0, 1.0)
    sh = lick(f_cut) * u * c * c * (3.0 - 2.0 * c)
    return np.exp(-0.5 * (((dx - sh) / sxg) ** 2 + (v / syg) ** 2))


@njit(parallel=True, fastmath=True, cache=True)
def _flame(img, zb, sx, sy, u, lk, r, g, b, z, zbias, x0, x1, y0, y1):
    sxg = 0.9 * u
    syg = 1.2 * u
    cy = sy - 0.3 * u
    for py in prange(y0, y1):
        for px in range(x0, x1):
            if zb[py, px] < z - zbias:
                continue
            v = cy - (py + 0.5)
            c = min(max(v / syg, 0.0), 1.0)
            sh = lk * u * c * c * (3.0 - 2.0 * c)
            a = (px + 0.5 - sx - sh) / sxg
            bq = v / syg
            w = math.exp(-0.5 * (a * a + bq * bq))
            img[py, px, 0] += r * w
            img[py, px, 1] += g * w
            img[py, px, 2] += b * w


@njit(parallel=True, fastmath=True, cache=True)
def _gauss(img, zb, sx, sy, sig, r, g, b, z, zbias, x0, x1, y0, y1):
    for py in prange(y0, y1):
        for px in range(x0, x1):
            if zb[py, px] < z - zbias:
                continue
            a = (px + 0.5 - sx) / sig
            bq = (py + 0.5 - sy) / sig
            w = math.exp(-0.5 * (a * a + bq * bq))
            img[py, px, 0] += r * w
            img[py, px, 1] += g * w
            img[py, px, 2] += b * w


def _box(img, sx, sy, R):
    H, W = img.shape[:2]
    return int(max(0, sx - R)), int(min(W, sx + R + 1)), int(max(0, sy - R)), int(min(H, sy + R + 1))


def draw(img, zb, sx, sy, u, f_cut, col, I=1.0, z=1e9, zbias=0.4):
    """The heart at (sx, sy) in img's pixels (pixel centres at +0.5), core sigma u px; col = the heart's colour
    (ice-white at the cut; the crossing warms it); I = the caller's slow breath (1 = the contract's peaks)."""
    col = np.asarray(col, np.float64)
    u = max(float(u), 0.6)
    # the warm edge first (under the white): 1.8 u
    s = 1.8 * u
    x0, x1, y0, y1 = _box(img, sx, sy, 3.2 * s)
    if x1 > x0 and y1 > y0:
        c = EDGE_COL * 0.10 * I
        _gauss(img, zb, float(sx), float(sy), s, c[0], c[1], c[2], float(z), float(zbias), x0, x1, y0, y1)
    # the steady glows: 2.6 u (peak 0.9) and 11.4 u (peak 0.05)
    for k, pk, zb_ in ((2.6, 0.9, zbias), (11.4, 0.05, zbias + 0.2)):
        s = k * u
        x0, x1, y0, y1 = _box(img, sx, sy, 3.2 * s)
        if x1 > x0 and y1 > y0:
            c = col * pk * I
            _gauss(img, zb, float(sx), float(sy), s, c[0], c[1], c[2], float(z), float(zb_), x0, x1, y0, y1)
    # the flame core: peak 30 x the flicker
    x0, x1, y0, y1 = _box(img, sx, sy - 0.3 * u, 3.6 * 1.2 * u + 0.3 * u)
    if x1 > x0 and y1 > y0:
        c = col * 30.0 * I * flicker(f_cut)
        _flame(img, zb, float(sx), float(sy), u, lick(f_cut), c[0], c[1], c[2], float(z), float(zbias),
               x0, x1, y0, y1)
