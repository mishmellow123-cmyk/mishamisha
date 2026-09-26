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


# ------------------------------------------------------------ small fires ---
# A distant fire is the same bonfire, only smaller: it is rendered into a supersampled scratch buffer
# (emission + opacity), box-filtered down and composited, so a 3-12 px fire keeps its tongues, its hot base
# and its ragged, flickering tip instead of collapsing into a round dot.

@njit(parallel=True, fastmath=True, cache=True)
def _bonfire_ea(E, A, bx, by, ppm, Hf, Rb, lean, t, seed, I, TG, warp, absorb, ux, uy):
    """_bonfire into an empty buffer: emission E (h, w, 3) and opacity A (h, w). (ux, uy) = screen direction
    of world-up (unit; (0, -1) is straight up the image)."""
    rise = 1.4 * math.sqrt(max(Hf, 0.05)) + 0.5
    K = TG.shape[0]
    Hh, Ww = A.shape
    for py in prange(Hh):
        for px in range(Ww):
            qx = (px + 0.5 - bx) / ppm
            qy = (py + 0.5 - by) / ppm
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
            r_, g_, b_ = ramp(T)
            e = I * T ** 4.2
            E[py, px, 0] = r_ * e
            E[py, px, 1] = g_ * e
            E[py, px, 2] = b_ * e
            A[py, px] = absorb * (1.0 - math.exp(-Fd * 3.0))


@njit(fastmath=True, cache=True)
def _composite(img, depth, E, A, x0, y0, z, zbias):
    h, w = A.shape
    H, W = img.shape[0], img.shape[1]
    for j in range(h):
        py = y0 + j
        if py < 0 or py >= H:
            continue
        for i in range(w):
            px = x0 + i
            if px < 0 or px >= W:
                continue
            if depth[py, px] < z - zbias:
                continue
            a = A[j, i]
            img[py, px, 0] = img[py, px, 0] * (1.0 - a) + E[j, i, 0]
            img[py, px, 1] = img[py, px, 1] * (1.0 - a) + E[j, i, 1]
            img[py, px, 2] = img[py, px, 2] * (1.0 - a) + E[j, i, 2]


def flame_small(img, depth, cam, base_world, Hf, Rb, t, seed=0, I=24.0, lean=0.0, zbias=0.5, TG=None, tongues=5,
                warp=1.0, absorb=0.6, target_px=26.0):
    """A flame of any size (use it below ~40 px): rendered at `target_px` tall in a scratch buffer, then
    area-filtered to its true size. Returns the screen (sx, sy, height_px) of the flame."""
    if Hf <= 0.0 or I <= 0:
        return None
    sx, sy, z = cam.project(np.asarray(base_world, np.float64))
    if z <= 0.1:
        return None
    ppm = cam.f / z
    hpx = Hf * ppm
    H, W = img.shape[:2]
    if hpx < 0.25:
        hpx = 0.25
    k = int(min(max(math.ceil(target_px / hpx), 1), 48))
    if TG is None:
        TG = F.tongue_table(tongues, seed)
    # scratch bbox (output pixels), then k x supersampled
    halfw = (Rb * 1.6 + abs(lean) * 1.3 + 0.28 * Hf) * ppm + 1.5
    x0 = int(math.floor(sx - halfw))
    x1 = int(math.ceil(sx + halfw))
    y0 = int(math.floor(sy - 1.95 * hpx - 1.5))
    y1 = int(math.ceil(sy + 0.12 * hpx + 1.5))
    if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
        return (sx, sy, hpx)
    w, h = x1 - x0, y1 - y0
    E = np.zeros((h * k, w * k, 3), np.float32)
    A = np.zeros((h * k, w * k), np.float32)
    _bonfire_ea(E, A, float((sx - x0) * k), float((sy - y0) * k), float(ppm * k), float(Hf), float(Rb), float(lean),
                float(t), float(seed), float(I), TG, float(warp), float(absorb), 0.0, -1.0)
    if k > 1:
        E = E.reshape(h, k, w, k, 3).mean(axis=(1, 3))
        A = A.reshape(h, k, w, k).mean(axis=(1, 3))
    _composite(img, depth, E.astype(np.float32), A.astype(np.float32), x0, y0, float(z), float(zbias))
    return (sx, sy, hpx)


def flame_sprite(sx, sy, ppm, Hf, Rb, t, seed=0, I=24.0, lean=0.0, TG=None, warp=1.0, absorb=0.6, ang=0.0,
                 target_px=64.0):
    """Render a flame (base at screen (sx, sy), ppm px/m, world-up at screen angle `ang`, 0 = image up) into
    a small scratch sprite at output resolution, supersampled and area-filtered. Returns (E, A, x0, y0)."""
    hpx = max(Hf * ppm, 0.25)
    k = int(min(max(math.ceil(target_px / hpx), 1), 48))
    ux, uy = math.sin(ang), -math.cos(ang)
    ext = 1.95 * hpx
    halfw = (Rb * 1.6 + abs(lean) * 1.3 + 0.28 * Hf) * ppm + 1.5
    tx, ty = sx + ux * ext, sy + uy * ext
    x0 = int(math.floor(min(sx, tx) - halfw - 1))
    x1 = int(math.ceil(max(sx, tx) + halfw + 1))
    y0 = int(math.floor(min(sy, ty) - halfw - 1))
    y1 = int(math.ceil(max(sy, ty) + halfw + 1))
    w, h = x1 - x0, y1 - y0
    E = np.zeros((h * k, w * k, 3), np.float32)
    A = np.zeros((h * k, w * k), np.float32)
    _bonfire_ea(E, A, float((sx - x0) * k), float((sy - y0) * k), float(ppm * k), float(Hf), float(Rb), float(lean),
                float(t), float(seed), float(I), TG, float(warp), float(absorb), float(ux), float(uy))
    if k > 1:
        E = E.reshape(h, k, w, k, 3).mean(axis=(1, 3))
        A = A.reshape(h, k, w, k).mean(axis=(1, 3))
    return np.ascontiguousarray(E, np.float32), np.ascontiguousarray(A, np.float32), x0, y0


@njit(fastmath=True, cache=True)
def composite_blur(img, depth, E, A, x0, y0, offs, z, zbias):
    """Composite a sprite (E, A; top-left at x0, y0) as the average of K copies shifted by offs (K, 2) px:
    an exact shutter-sampled motion blur of a rigid sprite, done after the frame's vector blur so small
    fires keep their shape instead of being smeared into bars by the reconstruction filter."""
    h, w = A.shape
    H, W = img.shape[0], img.shape[1]
    K = offs.shape[0]
    mnx = 0.0
    mxx = 0.0
    mny = 0.0
    mxy = 0.0
    for s in range(K):
        mnx = min(mnx, offs[s, 0])
        mxx = max(mxx, offs[s, 0])
        mny = min(mny, offs[s, 1])
        mxy = max(mxy, offs[s, 1])
    X0 = max(int(math.floor(x0 + mnx)) - 1, 0)
    X1 = min(int(math.ceil(x0 + w + mxx)) + 1, W)
    Y0 = max(int(math.floor(y0 + mny)) - 1, 0)
    Y1 = min(int(math.ceil(y0 + h + mxy)) + 1, H)
    inv = 1.0 / K
    for py in range(Y0, Y1):
        for px in range(X0, X1):
            if depth[py, px] < z - zbias:
                continue
            er = 0.0
            eg = 0.0
            eb = 0.0
            aa = 0.0
            for s in range(K):
                fx = px + 0.5 - x0 - offs[s, 0] - 0.5
                fy = py + 0.5 - y0 - offs[s, 1] - 0.5
                ix = int(math.floor(fx))
                iy = int(math.floor(fy))
                if ix < -1 or iy < -1 or ix >= w or iy >= h:
                    continue
                ax = fx - ix
                ay = fy - iy
                for dy in range(2):
                    yy = iy + dy
                    if yy < 0 or yy >= h:
                        continue
                    wy = ay if dy == 1 else 1.0 - ay
                    for dx in range(2):
                        xx = ix + dx
                        if xx < 0 or xx >= w:
                            continue
                        wgt = wy * (ax if dx == 1 else 1.0 - ax)
                        er += E[yy, xx, 0] * wgt
                        eg += E[yy, xx, 1] * wgt
                        eb += E[yy, xx, 2] * wgt
                        aa += A[yy, xx] * wgt
            if aa <= 0.0 and er <= 0.0:
                continue
            a = aa * inv
            img[py, px, 0] = img[py, px, 0] * (1.0 - a) + er * inv
            img[py, px, 1] = img[py, px, 1] * (1.0 - a) + eg * inv
            img[py, px, 2] = img[py, px, 2] * (1.0 - a) + eb * inv


def tongues_for(n, seed, Hf, spread=0.8, narrow=0.78, tall=(1, 4)):
    """A bonfire's tongue table (the pyre's recipe: separated tongues with dark gaps, two tall ones off
    centre) for a fire of height Hf (m); big fires breathe slower (period ~ Hf^0.35)."""
    TG = F.tongue_table(n, seed, spread=spread)
    TG[:, 2] *= narrow
    for i in tall:
        if i < n:
            TG[i, 1] *= 1.18
    TG[:, 3] *= max(Hf / 4.0, 1.0) ** 0.35
    return TG


# ------------------------------------------------------------------ plumes ---
# Smoke from a beacon: a buoyant column of soft puffs. The first puffs are thrown up by the whoomp of the
# catch (a rising cap), the rest climb more slowly and lean downwind. Each puff is lit orange from below by
# its fire (inverse-cube falloff), grey-blue on its moon side, and sits in the same air as the terrain
# (fogged by the transmittance of its distance).

class Plume:
    def __init__(self, seed, base, Hf, t_ign, t_end=1690 / 24.0, wind=(0.35, 0.0, 0.08), rate=7.0, life=4.0,
                 dens=0.55, burst=0.45):
        rng = np.random.default_rng(seed)
        self.base = np.asarray(base, np.float64)
        self.H = float(Hf)
        n = int(max(1, (t_end - t_ign) * rate))
        self.ts = t_ign + 0.06 + np.arange(n) / rate + rng.random(n) / rate * 0.7
        self.off = rng.normal(size=(n, 3)) * np.array([0.16, 0.08, 0.16])
        self.w0 = rng.uniform(0.9, 1.4, n)
        self.w0[self.ts < t_ign + burst] *= 1.9             # the catch throws the first puffs high
        self.winf = rng.uniform(0.22, 0.38, n)
        self.life = life * rng.uniform(0.75, 1.25, n)
        self.r0 = rng.uniform(0.13, 0.22, n)
        self.gr = rng.uniform(0.16, 0.30, n)
        self.dens = dens * rng.uniform(0.55, 1.3, n)
        self.seed = rng.integers(0, 10000, n)
        self.spin = rng.normal(size=n) * 0.35
        self.wind = np.asarray(wind, np.float64)             # drift, in flame heights per second
        self.t_ign = t_ign

    def render(self, img, depth, cam, t, fire_I, moon2d, moon_I, amb, fogc, tr, zbias, albedo=0.22,
               fire_col=(1.0, 0.42, 0.13), moon_col=(0.34, 0.45, 0.70)):
        a = t - self.ts
        ok = (a > 0) & (a < self.life)
        if not ok.any():
            return
        idx = np.nonzero(ok)[0]
        a = a[idx]
        H = self.H
        tau = 0.6
        rise = H * (self.winf[idx] * a + (self.w0[idx] - self.winf[idx]) * tau * (1.0 - np.exp(-a / tau)))
        pos = self.base[None, :] + self.off[idx] * H * (1.0 + 0.9 * a[:, None])
        pos[:, 1] += 0.55 * H + rise
        shear = np.clip(rise / (1.5 * H), 0.25, 1.0)
        pos[:, 0] += self.wind[0] * H * a * shear
        pos[:, 2] += self.wind[2] * H * a * shear
        rad = H * (self.r0[idx] + self.gr[idx] * a)
        life = self.life[idx]
        d = self.dens[idx] * np.clip(a / 0.25, 0, 1) * np.clip(1.0 - a / life, 0, 1) ** 1.5
        sx, sy, z = cam.project(pos)
        good = z > 0.5
        if not good.any():
            return
        sel = np.nonzero(good)[0][np.argsort(-z[good])]
        ppm = cam.f / z[sel]
        fire = self.base + np.array([0.0, 0.35 * H, 0.0])
        _plume(img, depth, sx[sel].astype(np.float64), sy[sel].astype(np.float64), z[sel].astype(np.float64),
               (rad[sel] * ppm).astype(np.float64), d[sel].astype(np.float64), pos[sel].astype(np.float64),
               a[sel].astype(np.float64), self.seed[idx][sel].astype(np.float64),
               self.spin[idx][sel].astype(np.float64), fire, float(fire_I), float(0.8 * H),
               rad[sel].astype(np.float64), np.asarray(moon2d, np.float64), float(moon_I), np.asarray(amb, np.float64),
               np.asarray(fogc, np.float64), float(tr), float(albedo), float(zbias), np.asarray(fire_col, np.float64),
               np.asarray(moon_col, np.float64))


@njit(fastmath=True, cache=True)
def _plume(img, depth, sx, sy, z, rpx, dens, pos, age, seeds, spin, fp, fI, lfall, rw, m2, mI, amb, fogc, tr,
           albedo, zbias, fcol, mcol):
    H, W = img.shape[0], img.shape[1]
    for k in range(sx.shape[0]):
        R = rpx[k] * 1.5
        x0 = int(max(0, sx[k] - R))
        x1 = int(min(W, sx[k] + R + 1))
        y0 = int(max(0, sy[k] - R))
        y1 = int(min(H, sy[k] + R + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        sd = seeds[k]
        ca = math.cos(spin[k] * age[k])
        sa = math.sin(spin[k] * age[k])
        for py in range(y0, y1):
            for px in range(x0, x1):
                if depth[py, px] < z[k] - zbias:
                    continue
                u0 = (px + 0.5 - sx[k]) / rpx[k]
                v0 = (py + 0.5 - sy[k]) / rpx[k]
                q = math.sqrt(u0 * u0 + v0 * v0)
                if q > 1.5:
                    continue
                u = ca * u0 - sa * v0
                v = sa * u0 + ca * v0
                nb = fbm3(u * 1.3 + sd, v * 1.3, age[k] * 0.3 + sd * 0.01, 4.0, 17)
                edge = 0.78 + 0.5 * nb
                dd = dens[k] * smoothstep(edge, edge * 0.35, q)
                if dd <= 1e-4:
                    continue
                nd = fbm3(u * 2.7 - sd, v * 2.7, age[k] * 0.4, 3.0, 23)
                dd *= 0.55 + 0.9 * max(nd + 0.3, 0.0)
                a = (1.0 - math.exp(-dd * 1.1)) * (0.35 + 0.65 * tr)
                wy = pos[k, 1] - v0 * rw[k]
                wx = pos[k, 0] + u0 * rw[k]
                dx = wx - fp[0]
                dy = wy - fp[1]
                dz = pos[k, 2] - fp[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                under = 0.4 + 0.6 * min(max(0.5 + 0.7 * v0, 0.0), 1.0)
                E = fI / (1.0 + (dist / lfall) ** 3) * under
                # the moon side of each billow is lit, the far side falls into the column's own shade
                lit = 0.30 + 0.70 * min(max(0.5 + 0.6 * (u0 * m2[0] + v0 * m2[1]), 0.0), 1.0)
                cr = amb[0] + albedo * (E * fcol[0] + mI * lit * mcol[0])
                cg = amb[1] + albedo * (E * fcol[1] + mI * lit * mcol[1])
                cb = amb[2] + albedo * (E * fcol[2] + mI * lit * mcol[2])
                cr = cr * tr + fogc[0] * (1.0 - tr)
                cg = cg * tr + fogc[1] * (1.0 - tr)
                cb = cb * tr + fogc[2] * (1.0 - tr)
                img[py, px, 0] = img[py, px, 0] * (1 - a) + cr * a
                img[py, px, 1] = img[py, px, 1] * (1 - a) + cg * a
                img[py, px, 2] = img[py, px, 2] * (1 - a) + cb * a
