"""THE RED BOOK (cut C, P1/P2/X1): an old red-bound book lying open on a table by a hearth, as a 2.5-D field.

World units are centimetres: x to the right (east), y away from the reader (north), z up; the table is z = 0
and the spine runs along y at x = 0. The book is a union of height columns (table, leather boards, the two
page blocks with their curved gutter, the spine and headbands at the head and tail), ray-marched per pixel.
The top leaf of each block carries a page texture (pen.py rasters in a mip pyramid); the paper itself is
procedural in page coordinates, so it is filtered by the pixel's own footprint and never shimmers.

Light: the hearth, a warm, flickering, soft point light low at one side (per shot), with soft shadows and a
dim warm room reflected in the gilt; a faint cool fill; optional extra point lights (a fire on the page).
Materials: aged paper, iron-gall ink (brown where thin, near black where it pools; glossy while wet), gold
leaf (a crinkled metal that catches the fire), graphite, red leather worn brown at the edges with gilt
rolls, silk headbands, a dark oak table.
"""
import math
import os
import sys

import cv2
import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
from noise import fbm, gnoise  # noqa: E402
import burn as BURN  # noqa: E402

M_NONE, M_TABLE, M_LEATHER, M_PAGE_L, M_PAGE_R, M_EDGE, M_SPINE, M_BAND = 0, 1, 2, 3, 4, 5, 6, 7
M_LEAF_F, M_LEAF_B = 8, 9      # a turning leaf: its front (the recto it was) and its back (the next verso)
NCH = 7          # page texture channels: ink, wet, gilt, pencil, fire (glowing ink), ghost (scorched letters),
                 # rubric (vermilion: the illuminated initial's ground and the rubricated line; MAP-L)
NG = 16          # G-buffer channels


# ================================================================ geometry ===

class Book:
    """Dimensions (cm). TL, TR: thickness of the left and right page blocks (open near the end: TL > TR)."""

    def __init__(self, PW=20.0, PH=29.0, TL=2.6, TR=1.4, bt=0.4, sq=0.75, seed=3, cockle_amp=0.07):
        self.PW, self.PH, self.TL, self.TR, self.bt, self.sq = PW, PH, TL, TR, bt, sq
        self.zg = bt + 0.5 * min(TL, TR) + 0.3           # the bottom of the gutter
        self.xr = 2.4 + 0.6 * max(TL, TR)                   # the width of the roll out of the gutter
        self.dome = 0.35
        self.seed = seed
        self.da = 0.0025
        self.a_tab = np.arange(0.0, PW + 2.0, self.da)
        self.uL = self._arc(TL)
        self.uR = self._arc(TR)
        self.feL = float(np.interp(PW, self.uL, self.a_tab))     # where the top leaf ends (horizontal)
        self.feR = float(np.interp(PW, self.uR, self.a_tab))
        self.params = np.array([PW, PH, TL, TR, bt, sq, self.zg, self.xr, self.dome, self.feL, self.feR,
                                0.8 + 0.3 * max(TL, TR)], np.float64)
        self.ck = cockle(self, cockle_amp, seed + 2)

    def _arc(self, T):
        a = self.a_tab
        z = np.array([_top(x, T, self.bt, self.zg, self.xr, self.dome, self.PW) for x in a])
        d = np.sqrt(np.diff(a) ** 2 + np.diff(z) ** 2)
        return np.concatenate([[0], np.cumsum(d)])

    def zmax(self):
        return self.bt + max(self.TL, self.TR) + self.dome + 0.6

    def page_to_world(self, side, u, v):
        """Page cm (u from the page's left edge, v down from its head) -> world (x, y, z) on the top leaf."""
        u = np.asarray(u, np.float64)
        v = np.asarray(v, np.float64)
        if side == 'R':
            a = np.interp(u, self.uR, self.a_tab)
            x = a
            T = self.TR
        else:
            a = np.interp(self.PW - u, self.uL, self.a_tab)
            x = -a
            T = self.TL
        y = 0.5 * self.PH - v
        z = np.array([_top(float(aa), T, self.bt, self.zg, self.xr, self.dome, self.PW)
                      for aa in np.atleast_1d(a)]).reshape(np.shape(a))
        return np.stack([x, y, z], -1)


@njit(cache=True)
def _top(a, T, bt, zg, xr, dome, PW):
    """Height of the top leaf of a block of thickness T at distance a from the spine (before the fore-edge)."""
    zt = bt + T
    if a < xr:
        q = 1.0 - a / xr
        r = math.sqrt(max(1.0 - q * q, 0.0))
        r = r * (0.82 + 0.18 * r)
        return zg + (zt + dome - zg) * r
    s = (a - xr) / (0.42 * PW)
    return zt + dome * math.exp(-s * s)


@njit(cache=True)
def height(x, y, P, ck):
    """Scene height and material at (x, y). P: Book.params; ck: the cockle grid (61 x 41)."""
    PW, PH, TL, TR, bt, sq = P[0], P[1], P[2], P[3], P[4], P[5]
    zg, xr, dome, feL, feR, J = P[6], P[7], P[8], P[9], P[10], P[11]
    z = 0.0
    m = M_TABLE
    ay = abs(y)
    ax = abs(x)
    Xo = PW + sq
    Yo = 0.5 * PH + sq
    if ax < Xo and ay < Yo and ax > 0.2:
        d = min(Xo - ax, Yo - ay)
        rr = 0.14
        zb = bt
        if d < rr:
            q = 1.0 - d / rr
            zb = bt * (0.35 + 0.65 * math.sqrt(max(1.0 - q * q, 0.0)))
        if zb > z:
            z = zb
            m = M_LEATHER
    if ay < 0.5 * PH:
        if ax < PW:
            T = TL if x < 0 else TR
            fe = feL if x < 0 else feR
            zt = _top(ax, T, bt, zg, xr, dome, PW)
            gx = (x + PW) / (2.0 * PW) * 40.0
            gy = (y + 0.5 * PH) / PH * 60.0
            ix = min(max(int(gx), 0), 39)
            iy = min(max(int(gy), 0), 59)
            fx = min(max(gx - ix, 0.0), 1.0)
            fy = min(max(gy - iy, 0.0), 1.0)
            c = (ck[iy, ix] * (1 - fx) + ck[iy, ix + 1] * fx) * (1 - fy) + \
                (ck[iy + 1, ix] * (1 - fx) + ck[iy + 1, ix + 1] * fx) * fy
            zt += c
            mm = M_PAGE_L if x < 0 else M_PAGE_R
            if ax > fe:
                s = (ax - fe) / max(PW - fe, 1e-3)
                zt = bt + (zt - bt) * max(0.0, 1.0 - s) ** 0.8
                mm = M_EDGE
            eh = 0.5 * PH - ay
            if eh < 0.06:
                zt = bt + (zt - bt) * max(eh / 0.06, 0.0) ** 0.35
                mm = M_EDGE
            if zt > z:
                z = zt
                m = mm
    else:
        if ay < 0.5 * PH + sq and ax < J + 0.35:
            q = ax / (J + 0.35)
            za = bt + (zg - bt - 0.1) * math.sqrt(max(1.0 - q * q, 0.0))
            mm = M_SPINE
            e = ay - 0.5 * PH
            if e < 0.22 and ax < J:
                hb = zg - 0.18 + 0.1 * math.sqrt(max(1.0 - ((e - 0.11) / 0.11) ** 2, 0.0))
                if hb > za:
                    za = hb
                    mm = M_BAND
            if za > z:
                z = za
                m = mm
    return z, m


def cockle(book, amp=0.07, seed=5):
    """Low waves in the leaves (61 x 41 grid over x in [-PW, PW] and the page height)."""
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 61))
    c = np.zeros_like(X)
    for _ in range(5):
        fx, fy = rng.uniform(1.5, 5.0), rng.uniform(1.0, 4.0)
        c += rng.uniform(0.4, 1.0) * np.sin(fx * X * 3.1 + rng.uniform(0, 6)) * np.sin(fy * Y * 3.1 + rng.uniform(0, 6))
    near = 0.35 + 0.65 * (np.exp(-np.abs(X) * 8) + np.clip((np.abs(X) - 0.8) / 0.2, 0, 1))
    return (amp * c / 2.5 * near).astype(np.float64)


# ================================================================ camera ===

class Cam:
    """Pinhole camera at pos looking at target; hfov in degrees; roll in degrees; focus distance (cm)."""

    def __init__(self, pos, target, hfov, W, H, roll=0.0, focus=None, fstop=None):
        self.pos = np.asarray(pos, np.float64)
        self.target = np.asarray(target, np.float64)
        self.W, self.H, self.hfov = W, H, hfov
        f = self.target - self.pos
        self.dist = float(np.linalg.norm(f))
        f /= np.linalg.norm(f)
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        if roll:
            a = math.radians(roll)
            r, u = r * math.cos(a) + u * math.sin(a), -r * math.sin(a) + u * math.cos(a)
        self.f, self.r, self.u = f, r, u
        self.F = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)
        self.focus = self.dist if focus is None else focus
        self.fstop = fstop

    def project(self, P):
        d = np.asarray(P, np.float64) - self.pos
        xc, yc, zc = d @ self.r, d @ self.u, d @ self.f
        zc = np.maximum(zc, 1e-6)
        return np.stack([self.W / 2.0 + self.F * xc / zc, self.H / 2.0 - self.F * yc / zc], -1), zc


# ============================================================ page textures ===

class PageTex:
    """A page's rasters (NCH channels at ppc px/cm) as a mip pyramid packed for the shader."""

    def __init__(self, PW, PH, ppc, levels=7):
        self.PW, self.PH, self.ppc, self.levels = PW, PH, ppc, levels
        self.H, self.W = int(round(PH * ppc)), int(round(PW * ppc))
        self.chan = np.zeros((self.H, self.W, NCH), np.float32)

    def build(self):
        lv = [self.chan]
        for _ in range(self.levels - 1):
            lv.append(cv2.pyrDown(lv[-1]))
        meta = np.zeros((len(lv), 3), np.int64)
        off = 0
        for k, a in enumerate(lv):
            meta[k] = (off, a.shape[0], a.shape[1])
            off += a.size
        data = np.empty(off, np.float32)
        for k, a in enumerate(lv):
            data[meta[k, 0]:meta[k, 0] + a.size] = a.reshape(-1)
        self.data, self.meta = data, meta
        return self


def blank_tex(PW, PH):
    return PageTex(PW, PH, 2.0, levels=1).build()


@njit(cache=True, inline='always')
def _tex_level(data, meta, L, x, y, out, w):
    off, H, W = meta[L, 0], meta[L, 1], meta[L, 2]
    s = W / meta[0, 2]
    x = x * s - 0.5
    y = y * s - 0.5
    x = min(max(x, 0.0), W - 1.001)
    y = min(max(y, 0.0), H - 1.001)
    ix = int(x)
    iy = int(y)
    fx = x - ix
    fy = y - iy
    b00 = off + (iy * W + ix) * NCH
    b01 = b00 + NCH
    b10 = b00 + W * NCH
    b11 = b10 + NCH
    for c in range(NCH):
        out[c] += w * ((data[b00 + c] * (1 - fx) + data[b01 + c] * fx) * (1 - fy) +
                       (data[b10 + c] * (1 - fx) + data[b11 + c] * fx) * fy)


@njit(cache=True)
def tex(data, meta, ppc, u, v, fp, out):
    """Trilinear sample at page cm (u, v) for a footprint fp (cm per pixel)."""
    for c in range(NCH):
        out[c] = 0.0
    nl = meta.shape[0]
    lod = math.log2(max(fp * ppc, 1e-6)) - 0.3
    lod = min(max(lod, 0.0), nl - 1.0)
    L = int(lod)
    f = lod - L
    _tex_level(data, meta, L, u * ppc, v * ppc, out, 1.0 - f)
    if f > 0.0 and L + 1 < nl:
        _tex_level(data, meta, L + 1, u * ppc, v * ppc, out, f)


# ================================================================== trace ===

@njit(cache=True)
def _arc_u(a, tab, da):
    x = a / da
    i = int(x)
    if i >= len(tab) - 1:
        return tab[-1]
    f = x - i
    return tab[i] * (1 - f) + tab[i + 1] * f


@njit(cache=True)
def leaf_hit(ox, oy, oz, dx, dy, dz, LX, LZ, halfH, tmax):
    """The nearest crossing (t, leaf, s index + fraction) of a ray with the turning leaves, sheets extruded along y
    whose cross-sections are the polylines LX[k], LZ[k]. Returns (t, leaf, k + u) or (-1, -1, 0)."""
    best = -1.0
    bs = 0.0
    bl = -1
    for q in range(LX.shape[0]):
        for k in range(LX.shape[1] - 1):
            ex = LX[q, k + 1] - LX[q, k]
            ez = LZ[q, k + 1] - LZ[q, k]
            den = dx * ez - dz * ex
            if abs(den) < 1e-12:
                continue
            qx = LX[q, k] - ox
            qz = LZ[q, k] - oz
            t = (qx * ez - qz * ex) / den
            u = (qx * dz - qz * dx) / den
            if u < 0.0 or u > 1.0 or t <= 1e-4 or t >= tmax:
                continue
            if abs(oy + dy * t) > halfH:
                continue
            if best < 0.0 or t < best:
                best = t
                bs = k + u
                bl = q
    return best, bl, bs


@njit(cache=True, parallel=True)
def trace_kernel(G, P, ck, uL, uR, da, cam_pos, cam_r, cam_u, cam_f, F, W, H, zmax, L_pos, L_rad, table_far,
                 LX, LZ):
    """G-buffer: 0 mat, 1 depth, 2-4 position, 5-7 normal, 8 u, 9 v, 10 footprint (cm/px), 11 shadow,
    12 ao, 13-15 view dir."""
    PW, PH = P[0], P[1]
    for i in prange(H):
        for j in range(W):
            sx = (j + 0.5 - W / 2.0) / F
            sy = -(i + 0.5 - H / 2.0) / F
            dx = cam_f[0] + sx * cam_r[0] + sy * cam_u[0]
            dy = cam_f[1] + sx * cam_r[1] + sy * cam_u[1]
            dz = cam_f[2] + sx * cam_r[2] + sy * cam_u[2]
            nd = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= nd
            dy /= nd
            dz /= nd
            ox, oy, oz = cam_pos[0], cam_pos[1], cam_pos[2]
            G[i, j, 13] = dx
            G[i, j, 14] = dy
            G[i, j, 15] = dz
            G[i, j, 0] = M_NONE
            G[i, j, 1] = 1e5
            if dz >= -1e-6:
                continue
            t0 = max(0.0, (zmax - oz) / dz)
            t1 = (0.0 - oz) / dz
            hit = False
            th = t1
            hxy = math.sqrt(dx * dx + dy * dy)
            step = min(0.02 / max(hxy, 1e-3), 0.06 / max(-dz, 1e-3))
            step = max(step, 0.004)
            tp = t0
            tc = t0
            while tc < t1:
                tc = min(tc + step, t1)
                hc, mc = height(ox + dx * tc, oy + dy * tc, P, ck)
                if oz + dz * tc - hc <= 1e-7 or tc >= t1:
                    a = tp
                    b = tc
                    for _ in range(16):
                        mid = 0.5 * (a + b)
                        hm, mm = height(ox + dx * mid, oy + dy * mid, P, ck)
                        if oz + dz * mid - hm > 0.0:
                            a = mid
                        else:
                            b = mid
                    th = b
                    hit = True
                    break
                tp = tc
            px = ox + dx * th
            py = oy + dy * th
            if LX.shape[0] > 0:
                tl, lq, sl = leaf_hit(ox, oy, oz, dx, dy, dz, LX, LZ, 0.5 * PH, th if (hit and py <= table_far) else 1e9)
                if tl > 0.0:
                    k = int(sl)
                    ex = LX[lq, k + 1] - LX[lq, k]
                    ez = LZ[lq, k + 1] - LZ[lq, k]
                    el = math.sqrt(ex * ex + ez * ez) + 1e-12
                    nx = -ez / el
                    nz = ex / el
                    side = M_LEAF_F
                    if nx * dx + nz * dz > 0.0:          # we see the back of the leaf
                        nx = -nx
                        nz = -nz
                        side = M_LEAF_B
                    qx = ox + dx * tl
                    qy = oy + dy * tl
                    qz = oz + dz * tl
                    G[i, j, 0] = side
                    G[i, j, 1] = tl * (dx * cam_f[0] + dy * cam_f[1] + dz * cam_f[2])
                    G[i, j, 2] = qx
                    G[i, j, 3] = qy
                    G[i, j, 4] = qz
                    G[i, j, 5] = nx
                    G[i, j, 6] = 0.0
                    G[i, j, 7] = nz
                    G[i, j, 10] = tl / F / math.sqrt(max(abs(dx * nx + dz * nz), 0.1))
                    s_arc = sl / (LX.shape[1] - 1) * P[0]
                    G[i, j, 8] = s_arc if side == M_LEAF_F else P[0] - s_arc
                    G[i, j, 9] = 0.5 * PH - qy
                    # light: the leaf's own side toward the hearth, or light through the paper
                    lx = L_pos[0] - qx
                    lz = L_pos[2] - qz
                    ly = L_pos[1] - qy
                    ld = math.sqrt(lx * lx + ly * ly + lz * lz)
                    tb, _, _ = leaf_hit(qx + nx * 1e-3, qy, qz + nz * 1e-3, lx / ld, ly / ld, lz / ld, LX, LZ, 0.5 * PH, 1e9)
                    G[i, j, 11] = 1.0 if tb < 0.0 else 0.2
                    G[i, j, 12] = 1.0
                    continue
            if not hit or py > table_far:
                continue
            pz, m = height(px, py, P, ck)
            e = 0.003
            hx1, _ = height(px + e, py, P, ck)
            hx0, _ = height(px - e, py, P, ck)
            hy1, _ = height(px, py + e, P, ck)
            hy0, _ = height(px, py - e, P, ck)
            nx = -(hx1 - hx0) / (2 * e)
            ny = -(hy1 - hy0) / (2 * e)
            nn = math.sqrt(nx * nx + ny * ny + 1.0)
            nx /= nn
            ny /= nn
            nz = 1.0 / nn
            G[i, j, 0] = m
            G[i, j, 1] = th * (dx * cam_f[0] + dy * cam_f[1] + dz * cam_f[2])
            G[i, j, 2] = px
            G[i, j, 3] = py
            G[i, j, 4] = pz
            G[i, j, 5] = nx
            G[i, j, 6] = ny
            G[i, j, 7] = nz
            cosv = abs(dx * nx + dy * ny + dz * nz)
            G[i, j, 10] = th / F / math.sqrt(max(cosv, 0.1))
            if m == M_PAGE_R:
                G[i, j, 8] = _arc_u(px, uR, da)
                G[i, j, 9] = 0.5 * PH - py
            elif m == M_PAGE_L:
                G[i, j, 8] = PW - _arc_u(-px, uL, da)
                G[i, j, 9] = 0.5 * PH - py
            elif m == M_EDGE:
                G[i, j, 8] = pz            # leaf index runs with height
                G[i, j, 9] = 0.5 * PH - py
            # soft shadow toward the hearth
            lx = L_pos[0] - px
            ly = L_pos[1] - py
            lz = L_pos[2] - pz
            ld = math.sqrt(lx * lx + ly * ly + lz * lz)
            lx /= ld
            ly /= ld
            lz /= ld
            sh = 1.0
            if lz > 0.0:
                tt = 0.02
                zs = pz + 2e-3
                while zs + lz * tt < zmax and tt < 80.0:
                    hq, _ = height(px + lx * tt, py + ly * tt, P, ck)
                    dh = zs + lz * tt - hq
                    pw_ = L_rad * tt / ld + 0.002
                    s = 0.5 + 0.5 * dh / pw_
                    if s < sh:
                        sh = max(s, 0.0)
                    if sh <= 0.0:
                        break
                    tt += min(max(0.015, 0.4 * abs(dh)) + 0.004 * tt, 0.2)
                sh = sh * sh * (3 - 2 * sh)
            else:
                sh = 0.0
            if LX.shape[0] > 0 and sh > 0.0:
                tb, _, _ = leaf_hit(px, py, pz + 2e-3, lx, ly, lz, LX, LZ, 0.5 * PH, 1e9)
                if tb > 0.0:
                    sh *= 0.12 + 0.1 * min(tb / 6.0, 1.0)       # a thin leaf: the shadow is soft and warm
            G[i, j, 11] = sh
            # ambient occlusion from the local field
            hs = 0.0
            for k in range(8):
                a = k * 0.785398 + 0.3
                for rr in (0.35, 1.1):
                    hq, _ = height(px + rr * math.cos(a), py + rr * math.sin(a), P, ck)
                    hs += max(hq - pz, 0.0) / rr
            G[i, j, 12] = 1.0 / (1.0 + 0.22 * hs)


# ================================================================ shading ===

@njit(cache=True, inline='always')
def _ss(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline='always')
def _ggx(nh, a2):
    d = nh * nh * (a2 - 1.0) + 1.0
    return a2 / (math.pi * d * d + 1e-9)


@njit(cache=True)
def _hash01(ix, iy, s):
    h = (ix * 374761393 + iy * 668265263 + s * 1274126177) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / 16777216.0


@njit(cache=True)
def _fox(u, v, sd):
    """Foxing: sparse round brown spots with soft, irregular rims (a jittered point process)."""
    cs = 1.3
    gx = u / cs
    gy = v / cs
    ix0 = int(math.floor(gx))
    iy0 = int(math.floor(gy))
    f = 0.0
    for iy in range(iy0 - 1, iy0 + 2):
        for ix in range(ix0 - 1, ix0 + 2):
            if _hash01(ix, iy, sd + 71) > 0.075:
                continue
            cx = (ix + _hash01(ix, iy, sd + 73)) * cs
            cy = (iy + _hash01(ix, iy, sd + 79)) * cs
            r = 0.03 + 0.14 * _hash01(ix, iy, sd + 83) ** 3
            d = math.sqrt((u - cx) ** 2 + (v - cy) ** 2)
            d *= 1.0 + 0.35 * gnoise(u * 12.0, v * 12.0, sd + 85)
            if d < 2.5 * r:
                a = math.exp(-(d / r) ** 2) * (0.35 + 0.65 * _hash01(ix, iy, sd + 89))
                ring = 0.35 * math.exp(-((d - r) / (0.25 * r)) ** 2)
                f = max(f, a + ring * a)
    return min(f, 1.0)


@njit(cache=True)
def paper(u, v, fp, PW, PH, seed, age, fore_right):
    """Aged paper albedo (linear RGB) and a height detail at page cm (u, v), filtered by footprint fp."""
    sd = int(seed)
    big = fbm(u * 0.08 + seed * 3.1, v * 0.08, 11 + sd, 3, 2.0, 0.5)
    mid = fbm(u * 0.55, v * 0.55 + seed, 23 + sd, 3, 2.1, 0.55)
    t = 0.55 * big + 0.3 * mid
    fib = 0.0
    if fp < 0.03:
        a = 2.0 * gnoise(u * 0.3, v * 0.3, 41 + sd)
        ca, sa = math.cos(a), math.sin(a)
        fu = (ca * u + sa * v) * 36.0
        fv = (-sa * u + ca * v) * 4.5
        k = _ss(0.03, 0.01, fp)
        fib = k * (0.55 * gnoise(fu, fv, 43 + sd) + 0.45 * gnoise(u * 85.0, v * 85.0, 47 + sd))
    de = min(min(u, PW - u), min(v, PH - v))
    tone = age * (0.5 * math.exp(-de / 0.7) + 0.3 * math.exp(-de / 3.0))
    fox = age * _fox(u, v, sd)
    # thumbed corner: the foot of the fore-edge
    uc = (PW - u) if fore_right else u
    gr = age * 0.5 * math.exp(-(uc * uc) / 18.0 - ((PH - v) ** 2) / 30.0) * (0.7 + 0.3 * big)
    r = 0.80 + 0.07 * t + 0.015 * fib - 0.16 * tone - 0.24 * fox - 0.14 * gr
    g = 0.69 + 0.07 * t + 0.015 * fib - 0.22 * tone - 0.32 * fox - 0.2 * gr
    b = 0.47 + 0.06 * t + 0.015 * fib - 0.24 * tone - 0.35 * fox - 0.22 * gr
    return r, g, b, fib


@njit(cache=True, parallel=True)
def shade_kernel(out, alpha, G, P, cam_pos, L_pos, L_col, L_rad, fill_dir, fill_col, amb_col, env_col,
                 texL, metaL, ppcL, texR, metaR, ppcR, seedL, seedR, age, burnL, burnR, xl, t,
                 texF, metaF, ppcF, texB, metaB, ppcB):
    """Colour (linear HDR) and coverage of the book layer. burnL/burnR: burn fields per page (burn.py);
    xl: extra point lights (n x 6: position, colour) without shadows (a fire on the page)."""
    H, W = out.shape[0], out.shape[1]
    PW, PH = P[0], P[1]
    for i in prange(H):
        tv = np.zeros(NCH, np.float64)
        for j in range(W):
            m = int(G[i, j, 0])
            dx, dy, dz = G[i, j, 13], G[i, j, 14], G[i, j, 15]
            if m == M_NONE:
                # the dark room beyond the table
                lx = L_pos[0] - cam_pos[0]
                ly = L_pos[1] - cam_pos[1]
                ll = math.sqrt(lx * lx + ly * ly) + 1e-6
                gl = max(0.0, (dx * lx + dy * ly) / ll / max(math.sqrt(dx * dx + dy * dy), 1e-6))
                for c in range(3):
                    out[i, j, c] = env_col[c] * (0.03 + 0.25 * gl ** 8)
                alpha[i, j] = 1.0
                continue
            px, py, pz = G[i, j, 2], G[i, j, 3], G[i, j, 4]
            nx, ny, nz = G[i, j, 5], G[i, j, 6], G[i, j, 7]
            fp = G[i, j, 10]
            sh = G[i, j, 11]
            ao = G[i, j, 12]
            vx, vy, vz = -dx, -dy, -dz
            # albedo, detail normal and material response
            ar, ag, ab = 0.1, 0.1, 0.1
            rough = 0.6
            spec = 0.02
            metal = 0.0
            mr, mg, mb = 1.0, 0.77, 0.34
            er, eg, eb = 0.0, 0.0, 0.0          # emission
            cov = 1.0
            wrap = 0.0
            trans = 0.0
            if m == M_PAGE_L or m == M_PAGE_R or m == M_LEAF_F or m == M_LEAF_B:
                u, v = G[i, j, 8], G[i, j, 9]
                right = m == M_PAGE_R or m == M_LEAF_F
                sd = seedR if right else seedL
                if m == M_LEAF_F or m == M_LEAF_B:
                    sd = sd + 5.0
                pr, pg, pb, fib = paper(u, v, fp, PW, PH, sd, age, right)
                if m == M_PAGE_R:
                    tex(texR, metaR, ppcR, u, v, fp, tv)
                elif m == M_PAGE_L:
                    tex(texL, metaL, ppcL, u, v, fp, tv)
                elif m == M_LEAF_F:
                    tex(texF, metaF, ppcF, u, v, fp, tv)
                else:
                    tex(texB, metaB, ppcB, u, v, fp, tv)
                if m == M_LEAF_F or m == M_LEAF_B:
                    trans = 0.3
                ink = min(tv[0], 1.25)
                wet = tv[1]
                gilt = min(tv[2], 1.0)
                pencil = min(tv[3], 1.0)
                fire = tv[4]
                ghost = min(tv[5], 1.0)
                rub = min(tv[6], 1.0)
                # paper tooth: ink sits in the valleys, graphite catches on the peaks
                tooth = 0.5 + 0.5 * fib
                # iron-gall ink: warm brown where thin, near black where it pools
                d = min(ink * (0.9 + 0.2 * tooth), 1.0)
                k = d ** 0.85
                ir = 0.16 - 0.14 * min(ink, 1.0)
                ig = 0.085 - 0.075 * min(ink, 1.0)
                ib = 0.04 - 0.033 * min(ink, 1.0)
                ar = pr * (1 - k) + ir * k
                ag = pg * (1 - k) + ig * k
                ab = pb * (1 - k) + ib * k
                # vermilion (rubric): an opaque, slightly chalky red laid over the ink it shares a letter with
                if rub > 0.0:
                    kr = min(rub * (0.92 + 0.16 * tooth), 1.0) ** 0.8
                    ar = ar * (1 - kr) + 0.43 * kr
                    ag = ag * (1 - kr) + 0.052 * kr
                    ab = ab * (1 - kr) + 0.026 * kr
                # graphite
                gp = pencil * (0.75 + 0.5 * (1.0 - tooth))
                gp = min(gp, 0.85)
                ar = ar * (1 - gp) + 0.085 * gp
                ag = ag * (1 - gp) + 0.085 * gp
                ab = ab * (1 - gp) + 0.095 * gp
                # scorched ghosts of letters that burned away
                if ghost > 0.0:
                    ar *= 1.0 - 0.55 * ghost
                    ag *= 1.0 - 0.7 * ghost
                    ab *= 1.0 - 0.8 * ghost
                spec = 0.03 + 0.25 * wet * d + 0.05 * gp
                rough = 0.55 - 0.4 * wet * d - 0.2 * gp
                wrap = 0.25
                if wet > 0.0:
                    ar *= 1.0 - 0.3 * wet * d
                    ag *= 1.0 - 0.3 * wet * d
                    ab *= 1.0 - 0.3 * wet * d
                # gold leaf: crinkled, burnished metal
                if gilt > 0.0:
                    cr = gnoise(u * 60.0, v * 60.0, 91) + 0.5 * gnoise(u * 150.0, v * 150.0, 93)
                    cr2 = gnoise(u * 60.0 + 7.0, v * 60.0, 95) + 0.5 * gnoise(u * 150.0 + 3.0, v * 150.0, 97)
                    nx = nx + 0.22 * cr * gilt
                    ny = ny + 0.22 * cr2 * gilt
                    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
                    nx /= nn
                    ny /= nn
                    nz /= nn
                    metal = gilt
                    rough = rough * (1 - gilt) + 0.22 * gilt
                # burn fields: browning, char, the hole, the glowing edge
                bf = burnR if m == M_PAGE_R else burnL
                if (m == M_PAGE_R or m == M_PAGE_L) and bf[0] > 0.0:
                    brown, char, hole, edge, pool = BURN.field(bf, u, v, t)
                    ar = ar * (1 - 0.62 * brown) * (1 - 0.96 * char)
                    ag = ag * (1 - 0.82 * brown) * (1 - 0.97 * char)
                    ab = ab * (1 - 0.95 * brown) * (1 - 0.98 * char)
                    cov = 1.0 - hole
                    er += edge * 4.2 + pool * ar * 0.6
                    eg += edge * 1.25 + pool * ag * 0.3
                    eb += edge * 0.18 + pool * ab * 0.1
                if fire > 0.0:
                    # awake in fire, like the Ring's letters: a deep orange-red core, never white
                    er += fire * 2.7
                    eg += fire * 0.64
                    eb += fire * 0.1
            elif m == M_EDGE:
                # leaf edges: one line per leaf, a little uneven, toned and dusty
                lv = G[i, j, 8] * 70.0 + 0.6 * gnoise(G[i, j, 9] * 0.4, G[i, j, 8] * 20.0, 5)
                ln = 0.5 + 0.5 * math.cos(6.2832 * lv)
                k = _ss(0.2, 1.0, fp * 70.0)            # leaves melt into tone when small on screen
                lt = (1 - k) * ln + k * 0.5
                # the quires: every sixteen leaves a darker seam, and each quire sits a little proud or shy
                qv = G[i, j, 8] * 70.0 / 16.0 + 0.3 * gnoise(G[i, j, 9] * 0.2, 3.0, 9)
                qs = math.exp(-((qv - math.floor(qv) - 0.5) / 0.06) ** 2)
                kq = _ss(0.3, 1.2, fp * 70.0 / 16.0)
                lt = min(1.0, lt + 0.9 * qs * (1 - kq))
                # striation that survives in a wide shot (H5): bands of leaves of slightly different tone, some
                # standing proud, wandering a little along the edge
                bv = G[i, j, 8] * 12.0 + 0.35 * gnoise(G[i, j, 9] * 0.25, G[i, j, 8] * 3.0, 11)
                bi = math.floor(bv)
                bt_ = (bi * 0.6180339887) % 1.0
                bf_ = bv - bi
                band = (bt_ - 0.5) * 0.9 + 0.35 * math.exp(-((bf_ - 0.5) / 0.1) ** 2) * (bt_ > 0.7)
                kb = _ss(0.6, 1.6, fp * 12.0)
                lt = min(max(lt + 0.45 * band * (1 - kb), 0.0), 1.0)
                dust = 0.5 + 0.5 * fbm(G[i, j, 9] * 0.6, G[i, j, 8] * 3.0, 7, 3, 2.0, 0.5)
                ar = (0.62 - 0.22 * lt) * (0.85 + 0.15 * dust)
                ag = (0.50 - 0.2 * lt) * (0.85 + 0.15 * dust)
                ab = (0.32 - 0.14 * lt) * (0.85 + 0.15 * dust)
                wrap = 0.2
            elif m == M_LEATHER or m == M_SPINE:
                gr = fbm(px * 3.0, py * 3.0, 31, 4, 2.2, 0.55)
                gr2 = gnoise(px * 22.0, py * 22.0, 33)
                edge_d = min(PW + P[5] - abs(px), 0.5 * PH + P[5] - abs(py))
                wear = _ss(0.25, 0.0, edge_d) * (0.6 + 0.4 * gr) + 0.25 * _ss(0.3, 0.7, gr)
                if m == M_SPINE:
                    wear = 0.3 + 0.3 * gr
                ar = 0.28 * (1 - wear) + 0.22 * wear + 0.03 * gr2
                ag = 0.03 * (1 - wear) + 0.075 * wear
                ab = 0.02 * (1 - wear) + 0.045 * wear
                spec = 0.045
                rough = 0.42 + 0.2 * wear
                nx += 0.05 * gr2
                ny += 0.05 * gnoise(px * 22.0 + 5.0, py * 22.0, 35)
                nn = math.sqrt(nx * nx + ny * ny + nz * nz)
                nx /= nn
                ny /= nn
                nz /= nn
                # gilt rolls on the squares: two fine fillets and a line of small dots between them
                if m == M_LEATHER and pz > P[4] * 0.9:
                    dd = edge_d
                    g1 = _ss(0.012, 0.0, abs(dd - 0.12))
                    g2 = _ss(0.009, 0.0, abs(dd - 0.33))
                    along = px if (PW + P[5] - abs(px)) > (0.5 * PH + P[5] - abs(py)) else py
                    dots = _ss(0.03, 0.0, abs(dd - 0.225)) * _ss(0.55, 0.9, 0.5 + 0.5 * math.cos(along * 41.0))
                    gl = max(g1, max(g2, dots)) * (1.0 - 0.7 * _ss(0.2, 0.8, wear + 0.3 * gr2))
                    if gl > 0.0:
                        metal = gl
                        rough = rough * (1 - gl) + 0.3 * gl
            elif m == M_BAND:
                st = 0.5 + 0.5 * math.sin(px * 90.0)
                ar = 0.25 * st + 0.08
                ag = 0.12 * st + 0.02
                ab = 0.03
                spec = 0.08
                rough = 0.35
            else:  # table: dark oak
                w1 = fbm(px * 0.05, py * 0.9, 51, 4, 2.0, 0.5)
                ring = 0.5 + 0.5 * math.sin((py + 6.0 * w1) * 2.4 + 3.0 * fbm(px * 0.02, py * 0.3, 53, 2, 2.0, 0.5))
                ar = 0.035 + 0.02 * ring
                ag = 0.017 + 0.01 * ring
                ab = 0.009 + 0.005 * ring
                spec = 0.03
                rough = 0.45
            # ------------------------------------------------ lighting
            lx = L_pos[0] - px
            ly = L_pos[1] - py
            lz = L_pos[2] - pz
            ld = math.sqrt(lx * lx + ly * ly + lz * lz)
            lx /= ld
            ly /= ld
            lz /= ld
            fall = (60.0 * 60.0) / (ld * ld)
            ndl = nx * lx + ny * ly + nz * lz
            dif = max((ndl + wrap) / (1.0 + wrap), 0.0)
            hx = lx + vx
            hy = ly + vy
            hz = lz + vz
            hn = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
            nh = max((nx * hx + ny * hy + nz * hz) / hn, 0.0)
            ndv = max(nx * vx + ny * vy + nz * vz, 1e-3)
            a2 = max(rough * rough, 0.002) ** 2
            spd = _ggx(nh, a2) * 0.25 / max(ndv, 0.2) * max(ndl, 0.0)
            spd = min(spd, 60.0)
            ndf = max(nx * fill_dir[0] + ny * fill_dir[1] + nz * fill_dir[2], 0.0)
            # a dim warm room reflected in metal: brighter toward the hearth
            rx = 2.0 * (nx * vx + ny * vy + nz * vz) * nx - vx
            ry = 2.0 * (nx * vx + ny * vy + nz * vz) * ny - vy
            rz = 2.0 * (nx * vx + ny * vy + nz * vz) * nz - vz
            envk = 0.04 + 0.5 * max(rx * lx + ry * ly + rz * lz, 0.0) ** 6 + 0.06 * max(rz, 0.0)
            for c in range(3):
                alb = (ar, ag, ab)[c]
                mc = (mr, mg, mb)[c]
                Lc = L_col[c] * fall * sh
                d_ = alb * (1.0 - metal) * (Lc * dif + fill_col[c] * ndf + amb_col[c] * ao)
                if trans > 0.0 and ndl < 0.0:
                    # light through the thin leaf from behind: a warm glow in the paper
                    d_ += trans * alb * L_col[c] * fall * (-ndl) * (0.6 + 0.4 * alb)
                s_ = (spec * (1.0 - metal) + mc * metal) * Lc * spd
                # gold leaf: a mirror of the warm room plus the broad scatter of crinkled, burnished leaf
                e_ = metal * mc * (env_col[c] * envk * ao * 2.2 + 0.55 * Lc * max(ndl, 0.0) + 0.9 * amb_col[c] * ao)
                # extra lights (a fire on the page): diffuse only
                x_ = 0.0
                for q in range(xl.shape[0]):
                    qx = xl[q, 0] - px
                    qy = xl[q, 1] - py
                    qz = xl[q, 2] - pz
                    qd2 = qx * qx + qy * qy + qz * qz + 0.04
                    qd = math.sqrt(qd2)
                    nq = max((nx * qx + ny * qy + nz * qz) / qd, 0.0)
                    x_ += xl[q, 3 + c] * nq / qd2
                d_ += alb * (1.0 - metal * 0.6) * x_ + metal * mc * x_ * 0.5
                out[i, j, c] = d_ + s_ + e_ + (er, eg, eb)[c]
            alpha[i, j] = cov
            if cov < 1.0:           # premultiplied: the glowing edge stays, the paper goes
                for c in range(3):
                    out[i, j, c] = (out[i, j, c] - (er, eg, eb)[c]) * cov + (er, eg, eb)[c]


# ================================================================ render ===

class Light:
    def __init__(self, pos, col=(1.0, 0.5, 0.2), power=1.0, radius=8.0):
        self.pos = np.asarray(pos, np.float64)
        self.col = np.asarray(col, np.float64) * power
        self.radius = radius


def flicker(t, seed=0, amt=0.1):
    """Hearth flicker: a slow breath and a quicker waver (t in seconds)."""
    return 1.0 + amt * (0.6 * gnoise(t * 0.7, seed * 1.3, 201) + 0.3 * gnoise(t * 2.3, seed, 203) +
                        0.25 * gnoise(t * 6.1, seed, 205))


def render(book, cam, light, texL, texR, t, fill=None, amb=(0.012, 0.009, 0.007), env=(0.9, 0.5, 0.22),
           burnL=None, burnR=None, xlights=None, age=1.0, table_far=40.0, leaf=None):
    """One frame of the book: returns (hdr HxWx3, alpha HxW, G-buffer). leaf = (phi, texF, texB) turns the
    right-hand leaf over: phi 0 (lying on the right) .. 1 (lying on the left); texR is then the page beneath it.
    phi may be a list (a riffle: several leaves in flight, all with the faces texF/texB)."""
    W, H = cam.W, cam.H
    G = np.zeros((H, W, NG), np.float64)
    phis = []
    if leaf is not None:
        ph = leaf[0] if np.ndim(leaf[0]) else [leaf[0]]
        phis = [p for p in ph if 0.0 < p < 1.0]
    if phis:
        cs = [leaf_curve(book, p) for p in phis]
        LX = np.stack([c[0] for c in cs])
        LZ = np.stack([c[1] for c in cs])
        texF, texB = leaf[1], leaf[2]
    else:
        LX = LZ = np.zeros((0, 2))
        texF = texB = texR
    trace_kernel(G, book.params, book.ck, book.uL, book.uR, book.da, cam.pos, cam.r, cam.u, cam.f, cam.F,
                 W, H, book.zmax(), light.pos, light.radius, table_far, LX, LZ)
    out = np.zeros((H, W, 3), np.float64)
    alpha = np.ones((H, W), np.float64)
    fd = np.array([0.35, -0.5, 0.8]) if fill is None else np.asarray(fill[0], np.float64)
    fd = fd / np.linalg.norm(fd)
    fc = np.array([0.010, 0.012, 0.016]) if fill is None else np.asarray(fill[1], np.float64)
    bL = np.zeros(16) if burnL is None else burnL
    bR = np.zeros(16) if burnR is None else burnR
    xl = np.zeros((0, 6)) if xlights is None else np.asarray(xlights, np.float64).reshape(-1, 6)
    shade_kernel(out, alpha, G, book.params, cam.pos, light.pos, light.col, light.radius, fd, fc,
                 np.asarray(amb, np.float64), np.asarray(env, np.float64),
                 texL.data, texL.meta, float(texL.ppc), texR.data, texR.meta, float(texR.ppc),
                 float(book.seed), float(book.seed + 17), float(age), bL, bR, xl, float(t),
                 texF.data, texF.meta, float(texF.ppc), texB.data, texB.meta, float(texB.ppc))
    return out.astype(np.float32), alpha.astype(np.float32), G


def leaf_curve(book, phi, N=160, lag=0.55):
    """Cross-section (x, z) of the right-hand leaf turning over, at progress phi: every part of the leaf rotates
    from the slope of the right block's top to the mirrored slope of the left block's, the part near the spine
    leading and the free edge lagging, so the leaf lifts, arches and curls over, then settles."""
    s = np.linspace(0.0, book.PW, N + 1)
    aR = np.interp(s, book.uR, book.a_tab)
    zR = np.array([_top(a, book.TR, book.bt, book.zg, book.xr, book.dome, book.PW) for a in aR])
    angR = np.arctan2(np.gradient(zR), np.gradient(aR))
    aL = np.interp(s, book.uL, book.a_tab)
    zL = np.array([_top(a, book.TL, book.bt, book.zg, book.xr, book.dome, book.PW) for a in aL])
    angL = np.pi - np.arctan2(np.gradient(zL), np.gradient(aL))
    g = np.clip(phi * (1.0 + lag) - lag * (s / book.PW), 0.0, 1.0)
    g = g * g * (3 - 2 * g)
    ang = angR * (1 - g) + angL * g
    ds = np.diff(s)
    x = np.concatenate([[0.0], np.cumsum(np.cos(0.5 * (ang[1:] + ang[:-1])) * ds)])
    z = book.zg + np.concatenate([[0.0], np.cumsum(np.sin(0.5 * (ang[1:] + ang[:-1])) * ds)]) + 0.004
    return x.astype(np.float64), z.astype(np.float64)


def dof(hdr, depth, cam, strength):
    """Thin-lens depth of field as a layered blur: CoC (px) = strength * |1 - focus / depth|."""
    if strength <= 0.05:
        return hdr
    coc = np.clip(strength * np.abs(1.0 - cam.focus / np.maximum(depth, 1.0)), 0, strength * 1.2).astype(np.float32)
    levels = [0.0, 1.0, 2.2, 4.5, 9.0, 18.0]
    levels = [l for l in levels if l <= max(coc.max(), 1.0) * 1.05] + []
    if len(levels) < 2:
        return hdr
    blurs = [hdr] + [cv2.GaussianBlur(hdr, (0, 0), s) for s in levels[1:]]
    out = np.zeros_like(hdr)
    wsum = np.zeros(coc.shape, np.float32)
    for k, lv in enumerate(levels):
        lo = levels[k - 1] if k > 0 else None
        hi = levels[k + 1] if k + 1 < len(levels) else None
        w = np.zeros_like(coc)
        if lo is None:
            w = np.clip(1 - coc / hi, 0, 1)
        elif hi is None:
            w = np.clip((coc - lo) / (lv - lo), 0, 1)
        else:
            w = np.where(coc < lv, np.clip((coc - lo) / (lv - lo), 0, 1), np.clip((hi - coc) / (hi - lv), 0, 1))
        out += blurs[k] * w[..., None]
        wsum += w
    return out / np.maximum(wsum, 1e-6)[..., None]
