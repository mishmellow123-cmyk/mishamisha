"""Merged-silhouette renderer for the festival-hill figures (v2, CODA dept, Sep 2026).

v1 shaded every puppet Group on its own, so each limb segment got its own rim and bevel: the
Elder and the Child read as jointed wooden dolls. Here a figure (or several figures that
overlap) is ONE shape:

  1. every Group is rasterised to its own distance field (smooth union inside the group);
  2. the fields are unioned (small smooth-min, so hairline gaps close) BEFORE any lighting,
     and the depth below the outline is the exact distance transform of that union mask, so
     seams between primitives (ribbon quads, a sleeve against the coat) never light up;
  3. rim light (the sky backlight and the fire rims) comes only from that OUTER contour;
     the sky rim takes its colour and strength from what is really behind each edge (the
     blurred background), so an edge against the bright sky glows and one against the dark
     hill almost vanishes;
  4. materials only change the interior colour, rim strength and edge softness. The scarf is
     the only lit cloth: diffuse firelight on its own rounded shape (the wrap) or on the
     smoothed face normals of the twisting tail, with a 2-D shadow march through the
     silhouette toward each light (the head shadows the tail).

Groups are puppet.Group objects; extra attributes (set by figures2.G2) are read with getattr
defaults, so plain Groups work too. Output: premultiplied RGB + alpha patch, like Figure.render.
"""
import math

import cv2
import numba as nb
import numpy as np

from core import perlin3
from puppet import local_grid, _sd_prim, _smin

GP = 24          # floats per group parameter row


def group_params(g):
    """Pack a Group's material into a row of GP floats."""
    r = np.zeros(GP)
    r[0:3] = g.albedo
    r[3] = getattr(g, 'mode', 1.0 if g.diffuse > 0 else 0.0)
    r[4] = g.sheen
    r[5] = g.sky_rim
    r[6] = g.soft
    r[7] = g.diffuse
    r[8] = max(1e-4, g.bevel)
    r[9:12] = g.tint
    r[12] = getattr(g, 'aa', 1.0)              # edge softness (1 = crisp, >1 = softer, px)
    r[13] = g.translucent
    em = np.zeros(3) if g.emissive is None else np.asarray(g.emissive, np.float64)
    r[14:17] = em
    r[17] = getattr(g, 'tex', 0.0)             # 1 = wool mottling
    r[18] = getattr(g, 'tex_scale', 0.012)     # metres
    r[19] = getattr(g, 'tex_amp', 0.0)
    r[20] = getattr(g, 'fuzz', 0.0)            # contour noise amplitude (metres)
    r[21] = getattr(g, 'fuzz_scale', 0.006)    # contour noise feature size (metres)
    r[22] = getattr(g, 'shadow', 1.0)          # 2-D shadow march for its diffuse light
    r[23] = getattr(g, 'union', 1.0)           # 0 = not part of the silhouette union
    return r


def casts(g):
    """Does this group cast 2-D shadows? (thin things - torch shaft, fingers, wisps, fringe - don't:
    in 3-D the flame is a volume and light wraps round them.)"""
    return getattr(g, 'cast', 1.0) > 0.5


# ------------------------------------------------------------------ raster ---

@nb.njit(cache=True, fastmath=True)
def raster_prims(D, PI, LX, LY, rows, verts, ranges, row0):
    """Smooth union of one group's primitives into D (h,w); PI = index (row0 + p) of the nearest
    primitive (for per-primitive attributes such as the scarf tail's face normals)."""
    Dp = np.full(D.shape, 10.0)
    for p in range(rows.shape[0]):
        j0 = ranges[p, 0]
        j1 = ranges[p, 1]
        i0 = ranges[p, 2]
        i1 = ranges[p, 3]
        k = rows[p, 1]
        for j in range(j0, j1):
            for i in range(i0, i1):
                d = _sd_prim(rows[p], verts, LX[j, i], LY[j, i])
                D[j, i] = _smin(D[j, i], d, k)
                if d < Dp[j, i]:
                    Dp[j, i] = d
                    PI[j, i] = row0 + p


@nb.njit(cache=True, fastmath=True)
def _fbm2n(x, y, z, octaves):
    s = 0.0
    a = 1.0
    n = 0.0
    for o in range(octaves):
        s += a * perlin3(x, y, z)
        n += a
        a *= 0.5
        x = x * 2.03 + 17.3
        y = y * 2.03 - 9.1
    return s / n


@nb.njit(cache=True, fastmath=True)
def apply_fuzz(D, LX, LY, G):
    """Contour noise (wool/hair fibres) on the fields that ask for it."""
    ng = D.shape[0]
    h = D.shape[1]
    w = D.shape[2]
    for g in range(ng):
        amp = G[g, 20]
        if amp <= 0.0:
            continue
        sc = 1.0 / max(1e-4, G[g, 21])
        for j in range(h):
            for i in range(w):
                d = D[g, j, i]
                if d > 0.02 or d < -0.02:
                    continue
                n = _fbm2n(LX[j, i] * sc, LY[j, i] * sc, g * 3.7 + 0.5, 2)
                D[g, j, i] = d + amp * n


@nb.njit(cache=True, fastmath=True)
def union_field(D, G, kunion):
    ng = D.shape[0]
    h = D.shape[1]
    w = D.shape[2]
    U = np.full((h, w), 10.0)
    for j in range(h):
        for i in range(w):
            m = 10.0
            for g in range(ng):
                if G[g, 23] > 0.5:
                    m = _smin(m, D[g, j, i], kunion)
            U[j, i] = m
    return U


@nb.njit(cache=True, fastmath=True)
def _bilin(U, x, y):
    h = U.shape[0]
    w = U.shape[1]
    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    if x > w - 1.001:
        x = w - 1.001
    if y > h - 1.001:
        y = h - 1.001
    i = int(x)
    j = int(y)
    fx = x - i
    fy = y - j
    a = U[j, i] * (1 - fx) + U[j, i + 1] * fx
    b = U[j + 1, i] * (1 - fx) + U[j + 1, i + 1] * fx
    return a * (1 - fy) + b * fy


@nb.njit(cache=True, fastmath=True)
def shadow2d(U, ppm, x0, y0, lx, ly, soft_k, floor):
    """Soft 2-D shadow on the card: march from pixel (x0,y0) toward the light's pixel position
    (lx,ly) through the union field U (metres, <0 inside). The start region is escaped first
    (a lit surface is not its own occluder) and the first few pixels past it never count, so
    the surface itself cannot soft-shadow its own edge; re-entering the silhouette = shadow."""
    dx = lx - x0
    dy = ly - y0
    L = math.sqrt(dx * dx + dy * dy)
    if L < 1.0:
        return 1.0
    dx /= L
    dy /= L
    h = U.shape[0]
    w = U.shape[1]
    t = 0.0
    esc = 0.06 * ppm
    while t < esc and t < L:
        x = x0 + dx * t
        y = y0 + dy * t
        if x < 0 or y < 0 or x > w - 1 or y > h - 1:
            return 1.0
        if _bilin(U, x, y) > 0.0:
            break
        t += 1.0
    if t >= esc:
        return 1.0          # deep inside a non-casting part (a hand on the shaft): unshadowed
    t0 = t
    t += 3.0
    res = 1.0
    for it in range(80):
        if t >= L - 1.0:
            break
        x = x0 + dx * t
        y = y0 + dy * t
        if x < 0 or y < 0 or x > w - 1 or y > h - 1:
            break
        d = _bilin(U, x, y) * ppm            # pixels
        if d < -0.5:
            return floor
        tt = t - t0
        s = soft_k * d / tt
        if s < res:
            res = s
        if res <= 0.0:
            return floor
        t += max(0.8, d)
    if res < 0.0:
        res = 0.0
    return floor + (1.0 - floor) * res


@nb.njit(cache=True, fastmath=True)
def shade_sil(D, DG, U, UC, DEP, NB, PI, prow, LX, LY, ppm, G, lights, lpx, nl, amb_top, amb_bot, back,
              out_rgb, out_a):
    """D (ng,h,w) group fields (metres); DG (ng,h,w) per-group depth below the group's outline
    (exact, for the lit wrap's bevel); U (h,w) union field; DEP (h,w) exact depth below the
    union's outer contour; NB (h,w,3) smoothed face normals of mode-2 cloth; G (ng,GP) params;
    lights (nl,8) [x,y,z(toward camera),r,g,b,d0,_] card-local; lpx (nl,2) lights in patch px;
    back (h,w,3) light behind each pixel (sky rim + wool glow)."""
    ng = D.shape[0]
    h = D.shape[1]
    w = D.shape[2]
    rw = max(0.0016, 1.1 / ppm)
    rw_soft = max(0.010, 3.0 / ppm)
    for j in range(h):
        for i in range(w):
            du = U[j, i]
            any_near = du * ppm < 3.0
            if not any_near:
                for g in range(ng):
                    if G[g, 23] < 0.5 and D[g, j, i] * ppm < 3.0:
                        any_near = True
                if not any_near:
                    continue
            x = LX[j, i]
            y = LY[j, i]
            il = i - 1 if i > 0 else i
            ir = i + 1 if i < w - 1 else i
            ju = j - 1 if j > 0 else j
            jd = j + 1 if j < h - 1 else j
            ddx = LX[j, ir] - LX[j, il]
            ddy = LY[ju, i] - LY[jd, i]
            cr = 0.0
            cg = 0.0
            cb = 0.0
            A = 0.0
            sh_ = 0.0
            sk_ = 0.0
            so_ = 0.0
            t0_ = 0.0
            t1_ = 0.0
            t2_ = 0.0
            for g in range(ng):
                d = D[g, j, i]
                aa = G[g, 12]
                c = 0.5 - d * ppm / aa
                if c <= 0.0:
                    continue
                if c > 1.0:
                    c = 1.0
                ar = G[g, 0]
                ag = G[g, 1]
                ab = G[g, 2]
                if G[g, 17] > 1.5 and PI[j, i] >= 0:
                    # knitted rib running along the ribbon (in the tail's own frame) + stitch grain
                    p_ = PI[j, i]
                    cx_ = prow[p_, 4]
                    cy_ = prow[p_, 5]
                    tx_ = prow[p_, 6]
                    ty_ = prow[p_, 7]
                    hwd = max(1e-4, prow[p_, 9])
                    wv = ((x - cx_) * (-ty_) + (y - cy_) * tx_) / max(1e-4, prow[p_, 8]) * hwd
                    sv = (x - cx_) * tx_ + (y - cy_) * ty_
                    rib = 0.5 + 0.5 * math.cos(math.pi * 2.0 * wv / 0.0155)
                    st_ = 0.5 + 0.5 * math.cos(math.pi * 2.0 * (sv / 0.0068 + 0.5 * wv / 0.0155))
                    tx = _fbm2n(x / G[g, 18], y / G[g, 18], 5.0 + g * 1.3, 2)
                    m = 0.72 + 0.32 * rib + 0.06 * st_ + 0.5 * G[g, 19] * tx
                    if m < 0.2:
                        m = 0.2
                    ar *= m
                    ag *= m
                    ab *= m
                elif G[g, 17] > 0.5:
                    tx = _fbm2n(x / G[g, 18], y / G[g, 18], 5.0 + g * 1.3, 3)
                    m = 1.0 + G[g, 19] * tx
                    if m < 0.2:
                        m = 0.2
                    ar *= m
                    ag *= m
                    ab *= m
                mode = int(G[g, 3] + 0.5)
                nx = 0.0
                ny = 0.0
                nz = 1.0
                if mode == 2:
                    nx = NB[j, i, 0]
                    ny = NB[j, i, 1]
                    nz = NB[j, i, 2]
                else:
                    # outward direction = -grad(exact depth): seams between primitives never tilt it
                    gx = 0.0
                    gy = 0.0
                    if abs(ddx) > 1e-9:
                        gx = -(DG[g, j, ir] - DG[g, j, il]) / ddx
                    if abs(ddy) > 1e-9:
                        gy = -(DG[g, ju, i] - DG[g, jd, i]) / ddy
                    gn = math.sqrt(gx * gx + gy * gy)
                    if gn > 1e-9:
                        gx /= gn
                        gy /= gn
                    e = DG[g, j, i] / G[g, 8]
                    if e > 1.0:
                        e = 1.0
                    if e < 0.0:
                        e = 0.0
                    oe = 1.0 - e
                    nz = math.sqrt(max(0.0, 1.0 - oe * oe)) + 0.08
                    nx = gx * oe
                    ny = gy * oe
                    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
                    nx /= nn
                    ny /= nn
                    nz /= nn
                up = 0.5 + 0.5 * ny
                r = ar * (amb_top[0] * up + amb_bot[0] * (1 - up))
                gg = ag * (amb_top[1] * up + amb_bot[1] * (1 - up))
                b = ab * (amb_top[2] * up + amb_bot[2] * (1 - up))
                if G[g, 7] > 0.0 and (mode == 1 or mode == 2):
                    for q in range(nl):
                        vx = lights[q, 0] - x
                        vy = lights[q, 1] - y
                        vz = lights[q, 2]
                        dist = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-6
                        vx /= dist
                        vy /= dist
                        vz /= dist
                        lam = nx * vx + ny * vy + nz * vz
                        if mode == 2:
                            lam = abs(lam) * 0.85 + 0.15 * max(0.0, lam)
                        if lam <= 0.0:
                            continue
                        att = 1.0 / (1.0 + (dist / lights[q, 6]) ** 2)
                        if G[g, 7] * lam * att * lights[q, 3] < 1e-4:
                            continue
                        shd = 1.0
                        if G[g, 22] > 0.5:
                            shd = shadow2d(UC, ppm, i, j, lpx[q, 0], lpx[q, 1], 3.0, 0.22)
                        k_ = G[g, 7] * lam * att * shd
                        r += ar * lights[q, 3] * k_
                        gg += ag * lights[q, 4] * k_
                        b += ab * lights[q, 5] * k_
                if G[g, 13] > 0.0:
                    r += G[g, 13] * back[j, i, 0] * ar * 1.5
                    gg += G[g, 13] * back[j, i, 1] * ag * 1.5
                    b += G[g, 13] * back[j, i, 2] * ab * 1.5
                r += G[g, 14]
                gg += G[g, 15]
                b += G[g, 16]
                ao = 1.0 - c
                cr = cr * ao + r * c
                cg = cg * ao + gg * c
                cb = cb * ao + b * c
                sh_ = sh_ * ao + G[g, 4] * c
                sk_ = sk_ * ao + G[g, 5] * c
                so_ = so_ * ao + G[g, 6] * c
                t0_ = t0_ * ao + G[g, 9] * c
                t1_ = t1_ * ao + G[g, 10] * c
                t2_ = t2_ * ao + G[g, 11] * c
                A = A * ao + c
            if A <= 1e-4:
                continue
            sh_ /= A
            sk_ /= A
            so_ /= A
            t0_ /= A
            t1_ /= A
            t2_ /= A
            # ---- rims: from the UNION's outer contour only
            depth = DEP[j, i]
            rs = math.exp(-depth / rw)
            rsoft = math.exp(-depth / rw_soft)
            rr = 0.0
            rg = 0.0
            rb = 0.0
            if rsoft > 0.004:
                ugx = 0.0
                ugy = 0.0
                if abs(ddx) > 1e-9:
                    ugx = (min(U[j, ir], 0.5) - min(U[j, il], 0.5)) / ddx
                if abs(ddy) > 1e-9:
                    ugy = (min(U[ju, i], 0.5) - min(U[jd, i], 0.5)) / ddy
                un = math.sqrt(ugx * ugx + ugy * ugy)
                if un > 1e-9:
                    ugx /= un
                    ugy /= un
                for q in range(nl):
                    vx = lights[q, 0] - x
                    vy = lights[q, 1] - y
                    vz = lights[q, 2]
                    dist = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-6
                    pl = math.sqrt(vx * vx + vy * vy) + 1e-6
                    rimd = (ugx * vx + ugy * vy) / pl
                    if rimd <= 0.0:
                        continue
                    att = 1.0 / (1.0 + (dist / lights[q, 6]) ** 2)
                    k_ = sh_ * rs * rimd * rimd + so_ * rsoft * rimd
                    if k_ * att * lights[q, 3] < 1e-4:
                        continue
                    shd = shadow2d(UC, ppm, i, j, lpx[q, 0], lpx[q, 1], 4.0, 0.15)
                    k_ *= att * shd
                    rr += lights[q, 3] * k_ * t0_
                    rg += lights[q, 4] * k_ * t1_
                    rb += lights[q, 5] * k_ * t2_
                sr = sk_ * rs * (0.40 + 0.60 * max(0.0, ugy))
                rr += back[j, i, 0] * sr
                rg += back[j, i, 1] * sr
                rb += back[j, i, 2] * sr
            out_rgb[j, i, 0] = out_rgb[j, i, 0] * (1 - A) + cr + rr * A
            out_rgb[j, i, 1] = out_rgb[j, i, 1] * (1 - A) + cg + rg * A
            out_rgb[j, i, 2] = out_rgb[j, i, 2] * (1 - A) + cb + rb * A
            out_a[j, i] = out_a[j, i] * (1 - A) + A


def exact_depth(F, ppm):
    """Depth (metres) below the outline of the shape F<0: the larger of the SDF's own inside
    distance (accurate near the edge) and the exact distance transform (accurate inside,
    immune to seams between abutting primitives)."""
    m = (F < 0.0).astype(np.uint8)
    if not m.any():
        return np.zeros(F.shape)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float64)
    return np.maximum(np.maximum(-F, 0.0), (dt - 0.5) / ppm)


class Silhouette:
    """Ordered groups on one camera-facing card, shaded as ONE silhouette.
    origin = world position of local (0,0); card plane z = origin.z."""

    def __init__(self, origin, groups, kunion=0.006):
        self.origin = np.asarray(origin, np.float64)
        self.groups = [g for g in groups if g is not None and len(g.rows)]
        self.kunion = kunion

    def local_bbox(self, pad=0.05):
        bbs = np.concatenate([g.arrays()[2] for g in self.groups], 0)
        return bbs[:, 0].min() - pad, bbs[:, 1].min() - pad, bbs[:, 2].max() + pad, bbs[:, 3].max() + pad

    def render(self, cam, lights, amb_top, amb_bot, back=(0.0, 0.0, 0.0), blur_px=0.0, t=0.0, bg=None,
               rim_k=2.4, rim_max=0.6, bg_sigma=5.0):
        """bg: the frame so far (linear HDR); if given, the sky rim at each edge is the light
        behind it (blurred bg * rim_k, capped) instead of the constant `back`."""
        if not self.groups:
            return None
        x0l, y0l, x1l, y1l = self.local_bbox()
        o = self.origin
        corners = np.array([[o[0] + x0l, o[1] + y0l, o[2]], [o[0] + x1l, o[1] + y0l, o[2]],
                            [o[0] + x0l, o[1] + y1l, o[2]], [o[0] + x1l, o[1] + y1l, o[2]]])
        sx, sy, z = cam.project(corners)
        if np.any(z <= 0.05):
            return None
        pad = int(4 + 3 * blur_px)
        X0 = max(0, int(np.floor(sx.min())) - pad)
        X1 = min(cam.W, int(np.ceil(sx.max())) + pad)
        Y0 = max(0, int(np.floor(sy.min())) - pad)
        Y1 = min(cam.H, int(np.ceil(sy.max())) + pad)
        if X1 <= X0 or Y1 <= Y0:
            return None
        h, w = Y1 - Y0, X1 - X0
        camp = cam.params()
        LX = np.empty((h, w), np.float64)
        LY = np.empty((h, w), np.float64)
        local_grid(LX, LY, camp, o[0], o[1], o[2], Y0, X0)
        ppm = cam.f / max(1e-6, float(np.mean(z)))
        ng = len(self.groups)
        D = np.full((ng, h, w), 10.0, np.float64)
        PI = np.full((h, w), -1, np.int64)
        G = np.zeros((ng, GP))
        prow_list = []
        row0 = 0
        for gi, g in enumerate(self.groups):
            rows, verts, bbs = g.arrays()
            G[gi] = group_params(g)
            kk = np.maximum(rows[:, 1], g.k) + 3.0 / ppm + G[gi, 20] * 2
            cx = np.stack([bbs[:, 0] - kk, bbs[:, 2] + kk], 1) + o[0]
            cy = np.stack([bbs[:, 1] - kk, bbs[:, 3] + kk], 1) + o[1]
            P = np.stack([np.stack([cx[:, a], cy[:, b], np.full(len(cx), o[2])], 1)
                          for a in (0, 1) for b in (0, 1)], 1)
            psx, psy, pz = cam.project(P.reshape(-1, 3))
            psx = psx.reshape(-1, 4)
            psy = psy.reshape(-1, 4)
            ranges = np.zeros((len(rows), 4), np.int64)
            ranges[:, 0] = np.clip(np.floor(psy.min(1)) - Y0 - 2, 0, h)
            ranges[:, 1] = np.clip(np.ceil(psy.max(1)) - Y0 + 3, 0, h)
            ranges[:, 2] = np.clip(np.floor(psx.min(1)) - X0 - 2, 0, w)
            ranges[:, 3] = np.clip(np.ceil(psx.max(1)) - X0 + 3, 0, w)
            PIg = np.full((h, w), -1, np.int64)
            raster_prims(D[gi], PIg, LX, LY, rows, verts, ranges, row0)
            pa = getattr(g, 'prim_attr', None)
            pr = np.zeros((len(rows), 10))
            if pa is not None and len(pa) == len(rows):
                pa = np.asarray(pa, np.float64).reshape(len(rows), -1)
                pr[:, :pa.shape[1]] = pa[:, :10]
            prow_list.append(pr)
            if G[gi, 3] > 1.5:
                sel = PIg >= 0
                PI[sel & (D[gi] < 0.004)] = PIg[sel & (D[gi] < 0.004)]
            row0 += len(rows)
        prow = np.concatenate(prow_list, 0) if prow_list else np.zeros((1, 10))
        apply_fuzz(D, LX, LY, G)
        U = union_field(D, G, self.kunion)
        GC_ = G.copy()
        for gi, g in enumerate(self.groups):
            if not casts(g):
                GC_[gi, 23] = 0.0
        UC = union_field(D, GC_, self.kunion)
        DEP = exact_depth(U, ppm)
        DG = np.zeros_like(D)
        NB = np.zeros((h, w, 3))
        NB[..., 2] = 1.0
        for gi in range(ng):
            if int(round(G[gi, 3])) == 1 and G[gi, 7] > 0:
                DG[gi] = np.where(D[gi] < 0, exact_depth(D[gi], ppm), -D[gi])
            else:
                DG[gi] = -D[gi]
        if (G[:, 3] > 1.5).any():
            # smoothed per-pixel face normals of the twisting tail (no banding between quads)
            m = PI >= 0
            if m.any():
                nb = np.zeros((h, w, 3))
                idx = PI[m]
                nb[m] = prow[idx, 1:4] * prow[idx, 0:1]
                wgt = m.astype(np.float64)
                s = max(1.0, 0.012 * ppm * 0.35)
                nb = cv2.GaussianBlur(nb, (0, 0), s)
                wgt = cv2.GaussianBlur(wgt, (0, 0), s)
                nb = nb / np.maximum(wgt, 1e-6)[..., None]
                nn = np.linalg.norm(nb, axis=2, keepdims=True)
                NB = np.where(nn > 1e-6, nb / np.maximum(nn, 1e-9), NB)
        L = np.asarray(lights, np.float64).reshape(-1, 8).copy()
        lpx = np.zeros((len(L), 2))
        if len(L):
            wp = np.stack([L[:, 0], L[:, 1], np.full(len(L), o[2])], 1)
            lsx, lsy, lz = cam.project(wp)
            lpx[:, 0] = lsx - X0 - 0.5
            lpx[:, 1] = lsy - Y0 - 0.5
        L[:, 0] -= o[0]
        L[:, 1] -= o[1]
        L[:, 2] = o[2] - L[:, 2]
        if bg is not None:
            sub = np.ascontiguousarray(bg[Y0:Y1, X0:X1, :3], np.float32)
            sig = max(1.0, bg_sigma * cam.scale)
            sub = cv2.GaussianBlur(sub, (0, 0), sig)
            bk = np.minimum(sub.astype(np.float64) * rim_k, rim_max)
        else:
            bk = np.empty((h, w, 3))
            bk[:] = np.asarray(back, np.float64)
        rgb = np.zeros((h, w, 3), np.float64)
        alpha = np.zeros((h, w), np.float64)
        shade_sil(D, DG, U, UC, DEP, NB, PI, prow, LX, LY, ppm, G, L, lpx, len(L), np.asarray(amb_top, np.float64),
                  np.asarray(amb_bot, np.float64), bk, rgb, alpha)
        rgb = rgb.astype(np.float32)
        alpha = alpha.astype(np.float32)
        if blur_px > 0.3:
            rgb = cv2.GaussianBlur(rgb, (0, 0), blur_px)
            alpha = cv2.GaussianBlur(alpha, (0, 0), blur_px)
        return Y0, X0, rgb, alpha
