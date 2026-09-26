"""EMBERS core: camera, numba point-splat renderer, noise, easing.

Conventions
-----------
* World: Y up. Camera space: x right, y up, z forward (depth).
* Screen: u = cx + f*x/z, v = cy - f*y/z ; pixel centres at integer coords.
* Particle energy E is expressed for the FULL-RES frame (1920x804): a particle
  of E=1 adds a total of 1.0 (linear HDR, summed over its footprint) per channel
  weighted by its colour. At render scale s the kernel multiplies by s^2 so
  test renders keep the same overall exposure.
"""
import math
import os

import cv2
import numpy as np
from numba import njit, prange

NT = int(os.environ.get('NUMBA_NUM_THREADS', '2'))
FULL_W, FULL_H = 1920, 804


# ----------------------------------------------------------------- easing ---

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = clamp01((np.asarray(x, np.float64) - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def smootherstep(e0, e1, x):
    t = clamp01((np.asarray(x, np.float64) - e0) / (e1 - e0))
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_in_out(t):
    t = clamp01(t)
    return np.where(t < 0.5, 4 * t ** 3, 1 - (-2 * t + 2) ** 3 / 2)


def ease_in_out_sine(t):
    t = clamp01(t)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def ease_out(t, p=3.0):
    t = clamp01(t)
    return 1 - (1 - t) ** p


def ease_in(t, p=3.0):
    t = clamp01(t)
    return t ** p


def ease_out_expo(t, k=10.0):
    t = clamp01(t)
    return (1 - np.exp(-k * t)) / (1 - math.exp(-k))


def ease_out_back(t, s=1.4):
    t = clamp01(t) - 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


def window(t, a, b, fade_in, fade_out):
    """1 inside [a,b], smooth ramps of given lengths outside."""
    return smoothstep(a - fade_in, a, t) * (1 - smoothstep(b, b + fade_out, t))


def track(t, keys, ease=ease_in_out_sine):
    """Piecewise eased interpolation through [(frame, value), ...]; value may be array."""
    ks = keys
    if t <= ks[0][0]:
        return np.asarray(ks[0][1], np.float64)
    for (f0, v0), (f1, v1) in zip(ks[:-1], ks[1:]):
        if t <= f1:
            u = float(ease((t - f0) / (f1 - f0)))
            return np.asarray(v0, np.float64) * (1 - u) + np.asarray(v1, np.float64) * u
    return np.asarray(ks[-1][1], np.float64)


def catmull(t, keys):
    """Catmull-Rom (centripetal-ish uniform) through keyed points [(frame, vec)], C1 smooth.
    Time is reparametrised per segment with smoothstep-free linear; use for camera paths."""
    fr = np.array([k[0] for k in keys], np.float64)
    P = np.array([k[1] for k in keys], np.float64)
    if t <= fr[0]:
        return P[0].copy()
    if t >= fr[-1]:
        return P[-1].copy()
    i = int(np.searchsorted(fr, t) - 1)
    i = max(0, min(i, len(fr) - 2))
    u = (t - fr[i]) / (fr[i + 1] - fr[i])
    p0 = P[i - 1] if i > 0 else 2 * P[i] - P[i + 1]
    p1, p2 = P[i], P[i + 1]
    p3 = P[i + 2] if i + 2 < len(P) else 2 * P[i + 1] - P[i]
    # tangent scaling for non-uniform key spacing
    d1 = fr[i + 1] - fr[i]
    d0 = fr[i] - fr[i - 1] if i > 0 else d1
    d2 = fr[i + 2] - fr[i + 1] if i + 2 < len(fr) else d1
    m1 = (p2 - p0) / (d0 + d1) * d1
    m2 = (p3 - p1) / (d1 + d2) * d1
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * p1 + (u3 - 2 * u2 + u) * m1 +
            (-2 * u3 + 3 * u2) * p2 + (u3 - u2) * m2)


# ----------------------------------------------------------------- camera ---

class Camera:
    def __init__(self, pos, target, up=(0, 1, 0), hfov=50.0, roll=0.0,
                 focus=None, aperture=0.0):
        self.pos = np.asarray(pos, np.float64)
        self.target = np.asarray(target, np.float64)
        fwd = self.target - self.pos
        fwd /= np.linalg.norm(fwd)
        up = np.asarray(up, np.float64)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(fwd, np.array([0, 0, 1.0]))
        right /= np.linalg.norm(right)
        upv = np.cross(right, fwd)
        if roll:
            c, s = math.cos(roll), math.sin(roll)
            right, upv = c * right + s * upv, -s * right + c * upv
        self.R = np.stack([right, upv, fwd])        # world -> camera rows
        self.hfov = hfov
        self.focus = focus if focus is not None else float(np.linalg.norm(self.target - self.pos))
        self.aperture = aperture

    def f_px(self, W):
        return (W / 2.0) / math.tan(math.radians(self.hfov) / 2.0)

    def params(self, W, H):
        a = np.zeros(16, np.float64)
        a[0:9] = self.R.reshape(-1)
        a[9:12] = self.pos
        a[12] = self.f_px(W)
        a[13] = (W - 1) / 2.0
        a[14] = (H - 1) / 2.0
        return a

    def project(self, P, W, H):
        """numpy projection (for 2D effects): returns u, v, z."""
        d = (np.asarray(P, np.float64) - self.pos) @ self.R.T
        f = self.f_px(W)
        z = d[..., 2]
        u = (W - 1) / 2.0 + f * d[..., 0] / np.maximum(z, 1e-6)
        v = (H - 1) / 2.0 - f * d[..., 1] / np.maximum(z, 1e-6)
        return u, v, z


# ---------------------------------------------------------------- splats ---

@njit(fastmath=True, cache=True, inline='always')
def _disc(buf, x, y, r, e, cr, cg, cb):
    H = buf.shape[0]
    W = buf.shape[1]
    edge = 0.5 + 0.06 * r
    ext = r + edge
    ix0 = int(math.floor(x - ext))
    ix1 = int(math.ceil(x + ext))
    iy0 = int(math.floor(y - ext))
    iy1 = int(math.ceil(y + ext))
    if ix1 < 0 or iy1 < 0 or ix0 >= W or iy0 >= H:
        return
    inv2e = 1.0 / (2.0 * edge)
    if r < 5.0:
        s = 0.0
        for yy in range(iy0, iy1 + 1):
            dy = yy - y
            for xx in range(ix0, ix1 + 1):
                dx = xx - x
                w = (r + edge - math.sqrt(dx * dx + dy * dy)) * inv2e
                if w > 0.0:
                    s += w if w < 1.0 else 1.0
        if s <= 1e-9:
            return
        k = e / s
    else:
        k = e / (math.pi * r * r)
    ys = iy0 if iy0 > 0 else 0
    ye = iy1 if iy1 < H - 1 else H - 1
    xs = ix0 if ix0 > 0 else 0
    xe = ix1 if ix1 < W - 1 else W - 1
    for yy in range(ys, ye + 1):
        dy = yy - y
        for xx in range(xs, xe + 1):
            dx = xx - x
            w = (r + edge - math.sqrt(dx * dx + dy * dy)) * inv2e
            if w > 0.0:
                if w > 1.0:
                    w = 1.0
                w *= k
                buf[yy, xx, 0] += w * cr
                buf[yy, xx, 1] += w * cg
                buf[yy, xx, 2] += w * cb


@njit(fastmath=True, cache=True, inline='always')
def _gauss(buf, x, y, r, e, cr, cg, cb):
    """Soft gaussian puff; r ~ 2 sigma."""
    H = buf.shape[0]
    W = buf.shape[1]
    sig = 0.5 * r
    if sig < 0.5:
        sig = 0.5
    ext = 2.5 * sig
    ix0 = int(math.floor(x - ext))
    ix1 = int(math.ceil(x + ext))
    iy0 = int(math.floor(y - ext))
    iy1 = int(math.ceil(y + ext))
    if ix1 < 0 or iy1 < 0 or ix0 >= W or iy0 >= H:
        return
    a = -0.5 / (sig * sig)
    k = e / (2.0 * math.pi * sig * sig)
    ys = iy0 if iy0 > 0 else 0
    ye = iy1 if iy1 < H - 1 else H - 1
    xs = ix0 if ix0 > 0 else 0
    xe = ix1 if ix1 < W - 1 else W - 1
    for yy in range(ys, ye + 1):
        dy = yy - y
        for xx in range(xs, xe + 1):
            dx = xx - x
            w = k * math.exp(a * (dx * dx + dy * dy))
            buf[yy, xx, 0] += w * cr
            buf[yy, xx, 1] += w * cg
            buf[yy, xx, 2] += w * cb


@njit(parallel=True, fastmath=True, cache=True)
def splat_points(P0, P1, RW, E, COL, cam0, cam1, prm, B0, B1, B2, B3, profile):
    """Project + splat N particles (motion-blurred between P0@cam0 and P1@cam1)."""
    N = P0.shape[0]
    T = B0.shape[0]
    focus = prm[0]
    aper = prm[1]
    rmin = prm[2]
    bpow = prm[3]
    bcap = prm[4]
    near = prm[5]
    fog0 = prm[6]
    fogl = prm[7]
    by0 = prm[8]
    by1 = prm[9]
    bk = prm[10]
    bsoft = prm[11]
    escale = prm[12]
    rmax = prm[13]
    maxsub = int(prm[14])
    zref = prm[15]
    f0 = cam0[12]
    f1 = cam1[12]
    cx = cam0[13]
    cy = cam0[14]
    Wf = cam0[13] * 2 + 1
    Hf = cam0[14] * 2 + 1
    chunk = (N + T - 1) // T
    for th in prange(T):
        b0 = B0[th]
        b1 = B1[th]
        b2 = B2[th]
        b3 = B3[th]
        i0 = th * chunk
        i1 = min(N, i0 + chunk)
        for i in range(i0, i1):
            e = E[i]
            if e <= 0.0:
                continue
            dx = P0[i, 0] - cam0[9]
            dy = P0[i, 1] - cam0[10]
            dz = P0[i, 2] - cam0[11]
            z0 = cam0[6] * dx + cam0[7] * dy + cam0[8] * dz
            if z0 < near:
                continue
            x0 = cam0[0] * dx + cam0[1] * dy + cam0[2] * dz
            y0 = cam0[3] * dx + cam0[4] * dy + cam0[5] * dz
            dx = P1[i, 0] - cam1[9]
            dy = P1[i, 1] - cam1[10]
            dz = P1[i, 2] - cam1[11]
            z1 = cam1[6] * dx + cam1[7] * dy + cam1[8] * dz
            if z1 < near:
                continue
            x1 = cam1[0] * dx + cam1[1] * dy + cam1[2] * dz
            y1 = cam1[3] * dx + cam1[4] * dy + cam1[5] * dz
            u0 = cx + f0 * x0 / z0
            v0 = cy - f0 * y0 / z0
            u1 = cx + f1 * x1 / z1
            v1 = cy - f1 * y1 / z1
            z = 0.5 * (z0 + z1)
            f = 0.5 * (f0 + f1)
            rp = RW[i] * f / z
            coc = 0.5 * aper * f * abs(1.0 / z - 1.0 / focus)
            rf2 = rp * rp + rmin * rmin
            r = math.sqrt(rf2 + coc * coc)
            # quick cull (generous)
            um = u0 if u0 < u1 else u1
            uM = u1 if u0 < u1 else u0
            vm = v0 if v0 < v1 else v1
            vM = v1 if v0 < v1 else v0
            if uM + r + 2 < 0 or vM + r + 2 < 0 or um - r - 2 > Wf or vm - r - 2 > Hf:
                continue
            e *= escale
            if zref > 0.0:
                e *= (zref / z) * (zref / z)
            if z < 2.0 * near:
                e *= (z - near) / near
            if z > fog0:
                e *= math.exp(-(z - fog0) / fogl)
            rf = math.sqrt(rf2)
            if bpow > 0.0 and r > rf * 1.02:
                g = (r / rf) ** bpow
                if g > bcap:
                    g = bcap
                e *= g
            if bk > 0.0:
                vc = 0.5 * (v0 + v1)
                a = (vc - (by0 - bsoft)) / (2 * bsoft)
                a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
                a = a * a * (3 - 2 * a)
                b = (vc - (by1 - bsoft)) / (2 * bsoft)
                b = 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)
                b = b * b * (3 - 2 * b)
                e *= 1.0 - bk * a * (1.0 - b)
            if r > rmax:
                r = rmax
            lvl = 0
            sc = 1.0
            rl = r
            while rl > 9.0 and lvl < 3:
                lvl += 1
                sc *= 0.5
                rl *= 0.5
            el = e * sc * sc
            ddx = (u1 - u0) * sc
            ddy = (v1 - v0) * sc
            L = math.sqrt(ddx * ddx + ddy * ddy)
            sp = 0.6 * rl
            if sp < 0.4:
                sp = 0.4
            n = int(L / sp) + 1
            if n > maxsub:
                n = maxsub
                rr = L / maxsub / 0.6
                if rr > rl:
                    rl = rr
            el /= n
            cr = COL[i, 0]
            cg = COL[i, 1]
            cb = COL[i, 2]
            bx = (u0 + 0.5) * sc - 0.5
            byy = (v0 + 0.5) * sc - 0.5
            for k in range(n):
                tk = (k + 0.5) / n
                xk = bx + ddx * tk
                yk = byy + ddy * tk
                if profile == 0:
                    if lvl == 0:
                        _disc(b0, xk, yk, rl, el, cr, cg, cb)
                    elif lvl == 1:
                        _disc(b1, xk, yk, rl, el, cr, cg, cb)
                    elif lvl == 2:
                        _disc(b2, xk, yk, rl, el, cr, cg, cb)
                    else:
                        _disc(b3, xk, yk, rl, el, cr, cg, cb)
                else:
                    if lvl == 0:
                        _gauss(b0, xk, yk, rl, el, cr, cg, cb)
                    elif lvl == 1:
                        _gauss(b1, xk, yk, rl, el, cr, cg, cb)
                    elif lvl == 2:
                        _gauss(b2, xk, yk, rl, el, cr, cg, cb)
                    else:
                        _gauss(b3, xk, yk, rl, el, cr, cg, cb)


# ------------------------------------------------------------- occlusion ---
#
# v2: the towers are solid. An occluder pass rasterises their surface points into a small depth / coverage / id
# pyramid (level 0 = half the frame's level-0 resolution, i.e. the size of the frame's level-1 buffer); every
# other splat then tests each pixel it touches: behind the occluder -> scaled by (1 - alpha). A tower's own
# points use a larger bias against their own id (so a surface never shadows itself at grazing angles) and the
# small one against the other towers.

@njit(fastmath=True, cache=True, inline='always')
def _vis(ZB, AB, IDB, ox, oy, z, bo, bs, myid):
    zb = ZB[oy, ox]
    if z <= zb + bo:
        return 1.0
    if IDB[oy, ox] == myid and z <= zb + bs:
        return 1.0
    return 1.0 - AB[oy, ox]


@njit(fastmath=True, cache=True, inline='always')
def _disc_z(buf, x, y, r, e, cr, cg, cb, ZB, AB, IDB, fo, z, bo, bs, myid):
    H = buf.shape[0]
    W = buf.shape[1]
    Ho = ZB.shape[0]
    Wo = ZB.shape[1]
    edge = 0.5 + 0.06 * r
    ext = r + edge
    ix0 = int(math.floor(x - ext))
    ix1 = int(math.ceil(x + ext))
    iy0 = int(math.floor(y - ext))
    iy1 = int(math.ceil(y + ext))
    if ix1 < 0 or iy1 < 0 or ix0 >= W or iy0 >= H:
        return
    inv2e = 1.0 / (2.0 * edge)
    if r < 5.0:
        s = 0.0
        for yy in range(iy0, iy1 + 1):
            dy = yy - y
            for xx in range(ix0, ix1 + 1):
                dx = xx - x
                w = (r + edge - math.sqrt(dx * dx + dy * dy)) * inv2e
                if w > 0.0:
                    s += w if w < 1.0 else 1.0
        if s <= 1e-9:
            return
        k = e / s
    else:
        k = e / (math.pi * r * r)
    ys = iy0 if iy0 > 0 else 0
    ye = iy1 if iy1 < H - 1 else H - 1
    xs = ix0 if ix0 > 0 else 0
    xe = ix1 if ix1 < W - 1 else W - 1
    for yy in range(ys, ye + 1):
        dy = yy - y
        oy = int((yy + 0.5) * fo)
        if oy > Ho - 1:
            oy = Ho - 1
        for xx in range(xs, xe + 1):
            dx = xx - x
            w = (r + edge - math.sqrt(dx * dx + dy * dy)) * inv2e
            if w > 0.0:
                if w > 1.0:
                    w = 1.0
                ox = int((xx + 0.5) * fo)
                if ox > Wo - 1:
                    ox = Wo - 1
                w *= k * _vis(ZB, AB, IDB, ox, oy, z, bo, bs, myid)
                buf[yy, xx, 0] += w * cr
                buf[yy, xx, 1] += w * cg
                buf[yy, xx, 2] += w * cb


@njit(fastmath=True, cache=True, inline='always')
def _gauss_z(buf, x, y, r, e, cr, cg, cb, ZB, AB, IDB, fo, z, bo, bs, myid):
    H = buf.shape[0]
    W = buf.shape[1]
    Ho = ZB.shape[0]
    Wo = ZB.shape[1]
    sig = 0.5 * r
    if sig < 0.5:
        sig = 0.5
    ext = 2.5 * sig
    ix0 = int(math.floor(x - ext))
    ix1 = int(math.ceil(x + ext))
    iy0 = int(math.floor(y - ext))
    iy1 = int(math.ceil(y + ext))
    if ix1 < 0 or iy1 < 0 or ix0 >= W or iy0 >= H:
        return
    a = -0.5 / (sig * sig)
    k = e / (2.0 * math.pi * sig * sig)
    ys = iy0 if iy0 > 0 else 0
    ye = iy1 if iy1 < H - 1 else H - 1
    xs = ix0 if ix0 > 0 else 0
    xe = ix1 if ix1 < W - 1 else W - 1
    for yy in range(ys, ye + 1):
        dy = yy - y
        oy = int((yy + 0.5) * fo)
        if oy > Ho - 1:
            oy = Ho - 1
        for xx in range(xs, xe + 1):
            dx = xx - x
            ox = int((xx + 0.5) * fo)
            if ox > Wo - 1:
                ox = Wo - 1
            w = k * math.exp(a * (dx * dx + dy * dy)) * _vis(ZB, AB, IDB, ox, oy, z, bo, bs, myid)
            buf[yy, xx, 0] += w * cr
            buf[yy, xx, 1] += w * cg
            buf[yy, xx, 2] += w * cb


@njit(parallel=True, fastmath=True, cache=True)
def splat_points_z(P0, P1, RW, E, COL, cam0, cam1, prm, B0, B1, B2, B3, profile,
                   Z0, A0, I0, Z1, A1, I1, Z2, A2, I2, ob, myid):
    """splat_points with a per-pixel occlusion test against the occluder pyramid (see above).
    ob = (generic bias abs, rel, same-id bias abs, rel)."""
    N = P0.shape[0]
    T = B0.shape[0]
    focus = prm[0]
    aper = prm[1]
    rmin = prm[2]
    bpow = prm[3]
    bcap = prm[4]
    near = prm[5]
    fog0 = prm[6]
    fogl = prm[7]
    by0 = prm[8]
    by1 = prm[9]
    bk = prm[10]
    bsoft = prm[11]
    escale = prm[12]
    rmax = prm[13]
    maxsub = int(prm[14])
    zref = prm[15]
    f0 = cam0[12]
    f1 = cam1[12]
    cx = cam0[13]
    cy = cam0[14]
    Wf = cam0[13] * 2 + 1
    Hf = cam0[14] * 2 + 1
    chunk = (N + T - 1) // T
    for th in prange(T):
        b0 = B0[th]
        b1 = B1[th]
        b2 = B2[th]
        b3 = B3[th]
        i0 = th * chunk
        i1 = min(N, i0 + chunk)
        for i in range(i0, i1):
            e = E[i]
            if e <= 0.0:
                continue
            dx = P0[i, 0] - cam0[9]
            dy = P0[i, 1] - cam0[10]
            dz = P0[i, 2] - cam0[11]
            z0 = cam0[6] * dx + cam0[7] * dy + cam0[8] * dz
            if z0 < near:
                continue
            x0 = cam0[0] * dx + cam0[1] * dy + cam0[2] * dz
            y0 = cam0[3] * dx + cam0[4] * dy + cam0[5] * dz
            dx = P1[i, 0] - cam1[9]
            dy = P1[i, 1] - cam1[10]
            dz = P1[i, 2] - cam1[11]
            z1 = cam1[6] * dx + cam1[7] * dy + cam1[8] * dz
            if z1 < near:
                continue
            x1 = cam1[0] * dx + cam1[1] * dy + cam1[2] * dz
            y1 = cam1[3] * dx + cam1[4] * dy + cam1[5] * dz
            u0 = cx + f0 * x0 / z0
            v0 = cy - f0 * y0 / z0
            u1 = cx + f1 * x1 / z1
            v1 = cy - f1 * y1 / z1
            z = 0.5 * (z0 + z1)
            f = 0.5 * (f0 + f1)
            rp = RW[i] * f / z
            coc = 0.5 * aper * f * abs(1.0 / z - 1.0 / focus)
            rf2 = rp * rp + rmin * rmin
            r = math.sqrt(rf2 + coc * coc)
            um = u0 if u0 < u1 else u1
            uM = u1 if u0 < u1 else u0
            vm = v0 if v0 < v1 else v1
            vM = v1 if v0 < v1 else v0
            if uM + r + 2 < 0 or vM + r + 2 < 0 or um - r - 2 > Wf or vm - r - 2 > Hf:
                continue
            e *= escale
            if zref > 0.0:
                e *= (zref / z) * (zref / z)
            if z < 2.0 * near:
                e *= (z - near) / near
            if z > fog0:
                e *= math.exp(-(z - fog0) / fogl)
            rf = math.sqrt(rf2)
            if bpow > 0.0 and r > rf * 1.02:
                g = (r / rf) ** bpow
                if g > bcap:
                    g = bcap
                e *= g
            if bk > 0.0:
                vc = 0.5 * (v0 + v1)
                a = (vc - (by0 - bsoft)) / (2 * bsoft)
                a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
                a = a * a * (3 - 2 * a)
                b = (vc - (by1 - bsoft)) / (2 * bsoft)
                b = 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)
                b = b * b * (3 - 2 * b)
                e *= 1.0 - bk * a * (1.0 - b)
            if r > rmax:
                r = rmax
            lvl = 0
            sc = 1.0
            rl = r
            while rl > 9.0 and lvl < 3:
                lvl += 1
                sc *= 0.5
                rl *= 0.5
            el = e * sc * sc
            ddx = (u1 - u0) * sc
            ddy = (v1 - v0) * sc
            L = math.sqrt(ddx * ddx + ddy * ddy)
            sp = 0.6 * rl
            if sp < 0.4:
                sp = 0.4
            n = int(L / sp) + 1
            if n > maxsub:
                n = maxsub
                rr = L / maxsub / 0.6
                if rr > rl:
                    rl = rr
            el /= n
            cr = COL[i, 0]
            cg = COL[i, 1]
            cb = COL[i, 2]
            bx = (u0 + 0.5) * sc - 0.5
            byy = (v0 + 0.5) * sc - 0.5
            for k in range(n):
                tk = (k + 0.5) / n
                xk = bx + ddx * tk
                yk = byy + ddy * tk
                zk = z0 + (z1 - z0) * tk
                bo = ob[0] + ob[1] * zk
                bs = ob[2] + ob[3] * zk
                if profile == 0:
                    if lvl == 0:
                        _disc_z(b0, xk, yk, rl, el, cr, cg, cb, Z0, A0, I0, 0.5, zk, bo, bs, myid)
                    elif lvl == 1:
                        _disc_z(b1, xk, yk, rl, el, cr, cg, cb, Z0, A0, I0, 1.0, zk, bo, bs, myid)
                    elif lvl == 2:
                        _disc_z(b2, xk, yk, rl, el, cr, cg, cb, Z1, A1, I1, 1.0, zk, bo, bs, myid)
                    else:
                        _disc_z(b3, xk, yk, rl, el, cr, cg, cb, Z2, A2, I2, 1.0, zk, bo, bs, myid)
                else:
                    if lvl == 0:
                        _gauss_z(b0, xk, yk, rl, el, cr, cg, cb, Z0, A0, I0, 0.5, zk, bo, bs, myid)
                    elif lvl == 1:
                        _gauss_z(b1, xk, yk, rl, el, cr, cg, cb, Z0, A0, I0, 1.0, zk, bo, bs, myid)
                    elif lvl == 2:
                        _gauss_z(b2, xk, yk, rl, el, cr, cg, cb, Z1, A1, I1, 1.0, zk, bo, bs, myid)
                    else:
                        _gauss_z(b3, xk, yk, rl, el, cr, cg, cb, Z2, A2, I2, 1.0, zk, bo, bs, myid)


@njit(fastmath=True, cache=True)
def occ_raster(P, NRM, AREA, RW, TID, cam, focus, aper, near, ZB, CV, IDB):
    """Occluder pass (single thread): front-facing surface points -> min depth ZB, coverage CV (sum of each
    point's projected patch area spread over its (defocused) disc) and the id of the nearest point."""
    Ho = ZB.shape[0]
    Wo = ZB.shape[1]
    f = cam[12]
    cx = cam[13]
    cy = cam[14]
    for i in range(P.shape[0]):
        dx = P[i, 0] - cam[9]
        dy = P[i, 1] - cam[10]
        dz = P[i, 2] - cam[11]
        z = cam[6] * dx + cam[7] * dy + cam[8] * dz
        if z < near:
            continue
        dl = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
        ndv = -(NRM[i, 0] * dx + NRM[i, 1] * dy + NRM[i, 2] * dz) / dl
        if ndv <= 0.0:
            continue
        x = cam[0] * dx + cam[1] * dy + cam[2] * dz
        y = cam[3] * dx + cam[4] * dy + cam[5] * dz
        u = cx + f * x / z
        v = cy - f * y / z
        rp = RW[i] * f / z
        coc = 0.5 * aper * f * abs(1.0 / z - 1.0 / focus)
        r = math.sqrt(rp * rp + coc * coc + 0.35 * 0.35)
        apx = AREA[i] * (f / z) * (f / z)
        edge = 0.5 + 0.06 * r
        ext = r + edge
        ix0 = int(math.floor(u - ext))
        ix1 = int(math.ceil(u + ext))
        iy0 = int(math.floor(v - ext))
        iy1 = int(math.ceil(v + ext))
        if ix1 < 0 or iy1 < 0 or ix0 >= Wo or iy0 >= Ho:
            continue
        inv2e = 1.0 / (2.0 * edge)
        s = 0.0
        for yy in range(iy0, iy1 + 1):
            ddy = yy - v
            for xx in range(ix0, ix1 + 1):
                ddx = xx - u
                w = (r + edge - math.sqrt(ddx * ddx + ddy * ddy)) * inv2e
                if w > 0.0:
                    s += w if w < 1.0 else 1.0
        if s <= 1e-9:
            continue
        k = apx / s
        tid = TID[i]
        ys = iy0 if iy0 > 0 else 0
        ye = iy1 if iy1 < Ho - 1 else Ho - 1
        xs = ix0 if ix0 > 0 else 0
        xe = ix1 if ix1 < Wo - 1 else Wo - 1
        for yy in range(ys, ye + 1):
            ddy = yy - v
            for xx in range(xs, xe + 1):
                ddx = xx - u
                w = (r + edge - math.sqrt(ddx * ddx + ddy * ddy)) * inv2e
                if w > 0.0:
                    if w > 1.0:
                        w = 1.0
                    CV[yy, xx] += w * k
                    if w > 0.5 and z < ZB[yy, xx]:
                        ZB[yy, xx] = z
                        IDB[yy, xx] = tid


@njit(fastmath=True, cache=True)
def occ_fill(ZB, IDB, ZO, IO):
    """one pass of hole filling: an empty pixel takes the nearest depth (and id) among its 8 neighbours"""
    H = ZB.shape[0]
    W = ZB.shape[1]
    for y in range(H):
        for x in range(W):
            ZO[y, x] = ZB[y, x]
            IO[y, x] = IDB[y, x]
            if ZB[y, x] < 1e29:
                continue
            best = 1e30
            bid = -1
            n = 0
            for dy in range(-1, 2):
                yy = y + dy
                if yy < 0 or yy >= H:
                    continue
                for dx in range(-1, 2):
                    xx = x + dx
                    if xx < 0 or xx >= W or (dx == 0 and dy == 0):
                        continue
                    z = ZB[yy, xx]
                    if z < 1e29:
                        n += 1
                        if z < best:
                            best = z
                            bid = IDB[yy, xx]
            if n >= 3:
                ZO[y, x] = best
                IO[y, x] = bid


def build_occluder(fr, cam, P, NRM, AREA, RW, TID, near=0.3, gain=1.25, amax=0.985):
    """Rasterise occluder points for frame `fr` (camera `cam` at the frame's mid time) and attach the pyramid."""
    H1 = -(-fr.H // 2)
    W1 = -(-fr.W // 2)
    ZB = np.full((H1, W1), 1e30, np.float32)
    CV = np.zeros((H1, W1), np.float32)
    IDB = np.full((H1, W1), -1, np.int32)
    c = cam.params(W1, H1)
    occ_raster(np.ascontiguousarray(P, np.float64), np.ascontiguousarray(NRM, np.float64),
               np.ascontiguousarray(AREA, np.float64), np.ascontiguousarray(RW, np.float64),
               np.ascontiguousarray(TID, np.int32), c, float(fr.prm[0]), float(fr.prm[1]), near, ZB, CV, IDB)
    ZO, IO = np.empty_like(ZB), np.empty_like(IDB)
    for _ in range(2):
        occ_fill(ZB, IDB, ZO, IO)
        ZB, ZO = ZO, ZB
        IDB, IO = IO, IDB
    # alpha follows the (defocused) coverage linearly, so a blurred silhouette fades like the light it hides
    CV = cv2.GaussianBlur(CV, (0, 0), 1.0)
    A = np.clip(gain * CV, 0.0, amax).astype(np.float32)
    Zs, As, Is = [ZB], [A], [IDB]
    for _ in range(2):
        z, a, i = Zs[-1], As[-1], Is[-1]
        h2, w2 = -(-z.shape[0] // 2), -(-z.shape[1] // 2)
        zp = np.full((h2 * 2, w2 * 2), 1e30, np.float32)
        ap = np.zeros((h2 * 2, w2 * 2), np.float32)
        ip = np.full((h2 * 2, w2 * 2), -1, np.int32)
        zp[:z.shape[0], :z.shape[1]] = z
        ap[:z.shape[0], :z.shape[1]] = a
        ip[:z.shape[0], :z.shape[1]] = i
        zq = zp.reshape(h2, 2, w2, 2).transpose(0, 2, 1, 3).reshape(h2, w2, 4)
        iq = ip.reshape(h2, 2, w2, 2).transpose(0, 2, 1, 3).reshape(h2, w2, 4)
        am = np.argmin(zq, axis=2)
        Zs.append(np.ascontiguousarray(np.take_along_axis(zq, am[..., None], 2)[..., 0]))
        Is.append(np.ascontiguousarray(np.take_along_axis(iq, am[..., None], 2)[..., 0]))
        As.append(np.ascontiguousarray(ap.reshape(h2, 2, w2, 2).mean(axis=(1, 3)), np.float32))
    fr.occ = (Zs, As, Is)
    return fr.occ


# ------------------------------------------------------------------ noise ---

_rs = np.random.RandomState(7)
_p = _rs.permutation(256).astype(np.int64)
PERM = np.concatenate([_p, _p]).astype(np.int64)


@njit(fastmath=True, cache=True, inline='always')
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(fastmath=True, cache=True, inline='always')
def _grad(h, x, y, z):
    h = h & 15
    u = x if h < 8 else y
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    else:
        v = z
    a = u if (h & 1) == 0 else -u
    b = v if (h & 2) == 0 else -v
    return a + b


@njit(fastmath=True, cache=True)
def perlin3(x, y, z):
    fx = math.floor(x)
    fy = math.floor(y)
    fz = math.floor(z)
    X = int(fx) & 255
    Y = int(fy) & 255
    Z = int(fz) & 255
    x -= fx
    y -= fy
    z -= fz
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)
    A = PERM[X] + Y
    AA = PERM[A] + Z
    AB = PERM[A + 1] + Z
    B = PERM[X + 1] + Y
    BA = PERM[B] + Z
    BB = PERM[B + 1] + Z
    g1 = _grad(PERM[AA], x, y, z)
    g2 = _grad(PERM[BA], x - 1, y, z)
    g3 = _grad(PERM[AB], x, y - 1, z)
    g4 = _grad(PERM[BB], x - 1, y - 1, z)
    g5 = _grad(PERM[AA + 1], x, y, z - 1)
    g6 = _grad(PERM[BA + 1], x - 1, y, z - 1)
    g7 = _grad(PERM[AB + 1], x, y - 1, z - 1)
    g8 = _grad(PERM[BB + 1], x - 1, y - 1, z - 1)
    l1 = g1 + u * (g2 - g1)
    l2 = g3 + u * (g4 - g3)
    l3 = g5 + u * (g6 - g5)
    l4 = g7 + u * (g8 - g7)
    m1 = l1 + v * (l2 - l1)
    m2 = l3 + v * (l4 - l3)
    return m1 + w * (m2 - m1)


@njit(parallel=True, fastmath=True, cache=True)
def _vnoise(P, freq, off, octaves, out):
    N = P.shape[0]
    for i in prange(N):
        x = P[i, 0] * freq + off[0]
        y = P[i, 1] * freq + off[1]
        z = P[i, 2] * freq + off[2]
        a = 1.0
        nx = 0.0
        ny = 0.0
        nz = 0.0
        for o in range(octaves):
            nx += a * perlin3(x, y, z)
            ny += a * perlin3(x + 31.416, y - 17.13, z + 5.91)
            nz += a * perlin3(x - 11.37, y + 43.71, z - 29.23)
            x *= 2.03
            y *= 2.03
            z *= 2.03
            a *= 0.5
        out[i, 0] = nx
        out[i, 1] = ny
        out[i, 2] = nz


def vnoise(P, freq, off=(0.0, 0.0, 0.0), octaves=2):
    """Coherent 3D vector noise (each component ~[-1,1]) at points P (N,3)."""
    P = np.ascontiguousarray(P, np.float64)
    out = np.empty_like(P)
    _vnoise(P, float(freq), np.asarray(off, np.float64), int(octaves), out)
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _snoise(P, freq, off, octaves, out):
    N = P.shape[0]
    for i in prange(N):
        x = P[i, 0] * freq + off[0]
        y = P[i, 1] * freq + off[1]
        z = P[i, 2] * freq + off[2]
        a = 1.0
        s = 0.0
        for o in range(octaves):
            s += a * perlin3(x, y, z)
            x *= 2.03
            y *= 2.03
            z *= 2.03
            a *= 0.5
        out[i] = s


def snoise(P, freq, off=(0.0, 0.0, 0.0), octaves=2):
    P = np.ascontiguousarray(P, np.float64)
    out = np.empty(P.shape[0], np.float64)
    _snoise(P, float(freq), np.asarray(off, np.float64), int(octaves), out)
    return out


# ----------------------------------------------------------------- frame ---

class Frame:
    """Multi-resolution accumulation buffers for one frame."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.W = int(round(FULL_W * scale))
        self.H = int(round(FULL_H * scale))
        self.bufs = []
        for l in range(4):
            hl = -(-self.H // (1 << l))
            wl = -(-self.W // (1 << l))
            self.bufs.append(np.zeros((NT, hl, wl, 3), np.float32))
        self.prm = np.zeros(16, np.float64)
        self.set(focus=10.0, aperture=0.0)
        self.occ = None                                   # occluder pyramid (build_occluder), None = off
        self.occ_bias = np.array([0.35, 0.012, 1.4, 0.05], np.float64)

    def set(self, focus=10.0, aperture=0.0, rmin=0.72, bokeh_pow=0.55, bokeh_cap=5.0,
            near=0.15, fog_start=1e9, fog_len=1e9, band=None, rmax=None, maxsub=96, zref=10.0):
        p = self.prm
        p[0] = focus
        p[1] = aperture
        p[2] = rmin
        p[3] = bokeh_pow
        p[4] = bokeh_cap
        p[5] = near
        p[6] = fog_start
        p[7] = fog_len
        if band is None:
            p[10] = 0.0
        else:
            y0, y1, k = band
            p[8] = y0 * self.scale
            p[9] = y1 * self.scale
            p[10] = k
            p[11] = 45.0 * self.scale
        p[12] = self.scale * self.scale
        p[13] = (rmax if rmax is not None else 110.0) * self.scale
        p[14] = maxsub
        p[15] = zref

    def splat(self, P0, P1, RW, E, COL, cam0, cam1, profile=0, zref=None, rmax=None, occ=True, myid=-1):
        if len(E) == 0:
            return
        if zref is not None or rmax is not None:
            keep = self.prm.copy()
            if zref is not None:
                self.prm[15] = zref
            if rmax is not None:
                self.prm[13] = rmax * self.scale
            try:
                self.splat(P0, P1, RW, E, COL, cam0, cam1, profile, occ=occ, myid=myid)
            finally:
                self.prm[:] = keep
            return
        P0 = np.ascontiguousarray(P0, np.float32)
        P1 = np.ascontiguousarray(P1, np.float32)
        RW = np.ascontiguousarray(np.broadcast_to(np.asarray(RW, np.float32), (len(E),)))
        E = np.ascontiguousarray(E, np.float32)
        COL = np.ascontiguousarray(np.broadcast_to(np.asarray(COL, np.float32), (len(E), 3)))
        c0 = cam0.params(self.W, self.H)
        c1 = cam1.params(self.W, self.H)
        if self.occ is not None and occ:
            Zs, As, Is = self.occ
            splat_points_z(P0, P1, RW, E, COL, c0, c1, self.prm,
                           self.bufs[0], self.bufs[1], self.bufs[2], self.bufs[3], profile,
                           Zs[0], As[0], Is[0], Zs[1], As[1], Is[1], Zs[2], As[2], Is[2],
                           self.occ_bias, int(myid))
            return
        splat_points(P0, P1, RW, E, COL, c0, c1, self.prm,
                     self.bufs[0], self.bufs[1], self.bufs[2], self.bufs[3], profile)

    def resolve(self):
        out = self.bufs[0].sum(axis=0)
        for l in range(1, 4):
            b = self.bufs[l].sum(axis=0)
            if not b.any():
                continue
            s = 1 << l
            up = cv2.resize(b, (b.shape[1] * s, b.shape[0] * s), interpolation=cv2.INTER_LINEAR)
            out += up[:self.H, :self.W]
        return out


# --------------------------------------------------------------- helpers ---

def rng(seed):
    return np.random.default_rng(seed)


def fib_sphere(n, seed=0):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i + seed
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)


def rand_dirs(r, n):
    v = r.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def axis_angle(axis, ang):
    """Rodrigues for many: axis (N,3) unit, ang (N,) -> (N,3,3)."""
    axis = np.asarray(axis, np.float64)
    ang = np.asarray(ang, np.float64)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(ang), np.sin(ang)
    C = 1 - c
    R = np.empty(ang.shape + (3, 3))
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R
