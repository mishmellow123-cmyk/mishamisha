"""Fire v2 for the Run / Dawn / s1 re-render: montage's tongue bonfire with two fixes the picture critic asked
for:
  * the ramp goes orange -> yellow -> white; its cool end is a deep ORANGE, never crimson (a crimson tip added
    to the blue night sky reads magenta);
  * the flame body is partly opaque (soot): it absorbs what is behind it in proportion to its density, so
    the sky's blue never shows through the tips.
Also an air-glow colour that goes warm-grey (not mauve) over a blue sky, and a spark colour ramp to match.
"""
import math

import numpy as np
from numba import njit, prange

from mt.noise import fbm3, smoothstep
from mt import fire as F

RAMP = np.array([[0.00, 0.58, 0.15, 0.025],
                 [0.30, 0.95, 0.30, 0.050],
                 [0.55, 1.00, 0.50, 0.110],
                 [0.78, 1.00, 0.73, 0.300],
                 [1.00, 1.00, 0.94, 0.800]])
GLOW_COL = np.array([1.0, 0.66, 0.32])


@njit(inline='always', fastmath=True)
def ramp(t):
    if t <= 0.0:
        return RAMP[0, 1], RAMP[0, 2], RAMP[0, 3]
    if t >= 1.0:
        return RAMP[4, 1], RAMP[4, 2], RAMP[4, 3]
    k = 0
    while k < 3 and t > RAMP[k + 1, 0]:
        k += 1
    w = (t - RAMP[k, 0]) / (RAMP[k + 1, 0] - RAMP[k, 0])
    return (RAMP[k, 1] + (RAMP[k + 1, 1] - RAMP[k, 1]) * w,
            RAMP[k, 2] + (RAMP[k + 1, 2] - RAMP[k, 2]) * w,
            RAMP[k, 3] + (RAMP[k + 1, 3] - RAMP[k, 3]) * w)


def ramp_vec(T):
    T = np.clip(np.asarray(T, np.float64), 0, 1)
    out = np.zeros((len(T), 3))
    for k in range(4):
        a, b = RAMP[k], RAMP[k + 1]
        w = np.clip((T - a[0]) / (b[0] - a[0]), 0, 1)[:, None]
        seg = (T >= a[0]) & (T <= b[0] + 1e-9)
        out = np.where(seg[:, None], a[1:] * (1 - w) + b[1:] * w, out)
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _bonfire(img, depth, bx, by, ppm, zf, Hf, Rb, lean, t, seed, I, x0, x1, y0, y1, zbias, TG, warp, absorb, ux, uy):
    """(ux, uy) = screen direction of world-up (unit; (0, -1) is straight up the image)."""
    rise = 1.4 * math.sqrt(max(Hf, 0.05)) + 0.5
    K = TG.shape[0]
    for py in prange(y0, y1):
        for px in range(x0, x1):
            if depth[py, px] < zf - zbias:
                continue
            qx = (px + 0.5 - bx) / ppm
            qy = (py + 0.5 - by) / ppm
            # rotate into the flame frame (Y along world-up)
            Y = qx * ux + qy * uy
            X = qx * (-uy) + qy * ux
            v = Y / Hf
            if v < -0.08 or v > 1.9:
                continue
            vc = max(v, 0.0)
            wa = fbm3(X / Rb * 0.7, (Y - rise * t) / Rb * 0.33, t * 0.45 + seed, 3.0, seed)
            wb = fbm3(X / Rb * 1.9 + 3.1, (Y - 1.3 * rise * t) / Rb * 0.8, t * 0.9 + seed, 3.0, seed + 7)
            Xw = X - lean * vc * vc - warp * Rb * (wa * (0.06 + 0.95 * vc) + 0.35 * wb * (0.04 + 0.7 * vc))
            Yw = Y + 0.12 * Rb * wb
            dsum = 0.0
            for k in range(K):
                cyc = t / TG[k, 3] + TG[k, 4]
                ph = cyc - math.floor(cyc)
                hmax = Hf * TG[k, 1]
                grow = 0.6 + 0.5 * smoothstep(0.0, 0.72, ph)
                hk = hmax * grow
                xk = TG[k, 0] * Rb + 0.10 * Rb * math.sin(2.7 * t + TG[k, 5]) * (0.3 + vc)
                xk *= 1.0 - 0.55 * min(max(Yw, 0.0) / Hf, 1.0)
                wk = Rb * TG[k, 2]
                dk = 0.0
                if Yw > -0.05 * Hf and Yw < hk:
                    s = max(Yw, 0.0) / hk
                    hw = wk * (1.0 - s) ** 0.62 * (1.0 + 0.45 * (1.0 - s) ** 3)
                    dk = 1.0 - abs(Xw - xk) / (hw + 1e-6)
                if ph > 0.7:
                    u = (ph - 0.7) / 0.3
                    ytop = hmax * 1.1
                    yb = ytop * 0.86 + u * TG[k, 3] * rise * 1.35
                    rb = wk * 0.55 * (1.0 - 0.75 * u) + 1e-4
                    ex = (Xw - xk * 0.5) / rb
                    ey = (Yw - yb) / (rb * 2.4)
                    blob = (1.0 - math.sqrt(ex * ex + ey * ey)) * (1.0 - u * u) * 1.3
                    if blob > dk:
                        dk = blob
                if dk > 0.0:
                    dsum += dk * dk
            if dsum <= 0.0:
                continue
            dens = math.sqrt(dsum)
            nd = fbm3(Xw / Rb * 3.2, (Yw - 1.5 * rise * t) / Rb * 1.3, t * 1.3 + seed * 2.0, 3.0, seed + 29)
            Fd = dens - (0.5 - 0.5 * nd) * (0.22 + 0.55 * vc)
            Fd *= smoothstep(-0.06, 0.04, v)
            if Fd <= 0.0:
                continue
            core = math.exp(-(X / (0.62 * Rb)) ** 2) * math.exp(-((v - 0.1) / 0.24) ** 2)
            Tb = 1.0 - math.exp(-Fd * 2.2)
            T = Tb * (0.68 - 0.30 * min(vc, 1.2)) + 0.30 * core * min(Fd * 4.0, 1.0)
            fl = fbm3(X / Rb * 1.2, (Y - rise * t) / Rb * 0.6, t * 3.0 + seed, 2.0, seed + 41)
            T *= 0.9 + 0.22 * fl
            if T > 1.0:
                T = 1.0
            if T <= 0.0:
                continue
            a = absorb * (1.0 - math.exp(-Fd * 3.0))
            r_, g_, b_ = ramp(T)
            e = I * T ** 4.2
            img[py, px, 0] = img[py, px, 0] * (1.0 - a) + r_ * e
            img[py, px, 1] = img[py, px, 1] * (1.0 - a) + g_ * e
            img[py, px, 2] = img[py, px, 2] * (1.0 - a) + b_ * e


def flame(img, depth, cam, base_world, Hf, Rb, t, seed=0, I=24.0, lean=0.0, zbias=0.5, tongues=5, warp=1.0,
          min_px=2.0, absorb=0.7, TG=None, up=None):
    """As mt.fire.flame (base centre at base_world; Hf height, Rb base half-width, metres). `up` = screen
    angle of world-up (radians, 0 = image up); None for lens-shift cameras (verticals vertical)."""
    if Hf <= 0.01 or I <= 0:
        return
    sx, sy, z = cam.project(np.asarray(base_world, np.float64))
    if z <= 0.1:
        return
    ppm = cam.f / z
    H, W = img.shape[:2]
    if Hf * ppm < min_px * 4:
        e = I * 0.12 * (2 * Rb * ppm) * (Hf * ppm)
        F.glow(img, depth, sx, sy - 0.35 * Hf * ppm, max(0.35 * Hf * ppm, 0.7), e, z, zbias, col=GLOW_COL)
        return
    if TG is None:
        TG = F.tongue_table(tongues, seed)
    ang = 0.0 if up is None else up
    ux, uy = math.sin(ang), -math.cos(ang)
    R = (Rb * 1.5 + abs(lean) * 1.3 + 0.25 * Hf + Hf * 1.9 * abs(math.sin(ang))) * ppm
    top_x = sx + ux * Hf * 1.9 * ppm
    top_y = sy + uy * Hf * 1.9 * ppm
    x0 = int(max(0, min(sx, top_x) - R - 2))
    x1 = int(min(W, max(sx, top_x) + R + 2))
    y0 = int(max(0, min(sy, top_y) - Rb * 1.5 * ppm - 2))
    y1 = int(min(H, max(sy, top_y) + 0.1 * Hf * ppm + Rb * ppm + 2))
    if x1 <= x0 or y1 <= y0:
        return
    _bonfire(img, depth, float(sx), float(sy), float(ppm), float(z), float(Hf), float(Rb), float(lean), float(t),
             float(seed), float(I), x0, x1, y0, y1, float(zbias), TG, float(warp), float(absorb), float(ux), float(uy))


AURA_COL = np.array([1.0, 0.50, 0.17])


def halo(img, depth, sx, sy, sig_px, peak, z=1e9, zbias=0.0, squash=1.0, col=None):
    F.halo(img, depth, sx, sy, sig_px, peak, z=z, zbias=zbias, col=GLOW_COL if col is None else col, squash=squash)


def glow(img, depth, sx, sy, rad_px, energy, z=1e9, zbias=0.0, col=None):
    F.glow(img, depth, sx, sy, rad_px, energy, z=z, zbias=zbias, col=GLOW_COL if col is None else col)
