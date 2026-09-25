"""2D SDF puppets for figures and props.

A drawing is a list of primitives in *local metres* (x right on screen, y up, origin at the feet /
base). Primitives with the same `group` are smooth-unioned into one silhouette; groups are painted
in order (so a cairn is many stone groups, a figure is one or two groups).

Shading uses a pseudo-3D normal built from the SDF (edge -> lateral, interior -> facing camera),
so fires beside/behind a figure produce proper rim light and fires in front light the face.
Lights are given in figure-local 3D coordinates (x right, y up, z toward the camera).
"""
import math

import numpy as np
from numba import njit, prange

from .noise import gnoise2, fbm2, gnoise3

# primitive row layout
# 0 type | 1..7 geometry | 8 smooth k | 9 fuzz amp | 10 fuzz freq | 11 material | 12 group
CAPSULE, ELLIPSE, TRAP, TRI = 0, 1, 2, 3
NCOL = 13

# material row: 0-2 albedo | 3 rim | 4 spec | 5 spec pow | 6 thickness | 7-9 emissive | 10 ember noise freq
MATS = np.array([
    [0.030, 0.028, 0.030, 0.9, 0.0, 8.0, 0.14, 0, 0, 0, 0],     # 0 dark cloth
    [0.070, 0.060, 0.052, 1.6, 0.0, 8.0, 0.05, 0, 0, 0, 0],     # 1 fur / wool fuzz
    [0.110, 0.066, 0.048, 1.0, 0.2, 10.0, 0.07, 0, 0, 0, 0],    # 2 skin
    [0.030, 0.034, 0.030, 1.0, 1.4, 14.0, 0.12, 0, 0, 0, 0],    # 3 oilskin (glossy)
    [0.050, 0.032, 0.020, 0.6, 0.0, 8.0, 0.03, 0, 0, 0, 0],     # 4 wood
    [0.035, 0.036, 0.040, 0.8, 1.6, 24.0, 0.02, 0, 0, 0, 0],    # 5 iron
    [0.018, 0.016, 0.015, 1.2, 0.3, 12.0, 0.05, 0, 0, 0, 0],    # 6 hair
    [0.030, 0.028, 0.026, 1.3, 0.0, 6.0, 0.035, 0, 0, 0, 0],    # 7 stone (dark, rim-lit edges)
    [0.050, 0.030, 0.018, 0.4, 0.0, 8.0, 0.05, 1.0, 0.30, 0.05, 28.0],   # 8 glowing log / embers
    [0.020, 0.020, 0.024, 0.9, 0.3, 10.0, 0.14, 0, 0, 0, 0],    # 9 dark synthetic (hoodie)
    [0.300, 0.300, 0.320, 0.4, 0.3, 8.0, 0.30, 0, 0, 0, 0],     # 10 snow / pale stone
    [0.060, 0.058, 0.062, 0.6, 0.8, 16.0, 0.30, 0, 0, 0, 0],    # 11 painted steel (drum / tank)
    [0.008, 0.007, 0.007, 0.0, 0.0, 8.0, 0.30, 0.8, 0.22, 0.03, 9.0],   # 12 cairn chinks (glow when lit)
], np.float64)


class Drawing:
    def __init__(self):
        self.rows = []
        self.group = 0

    def new_group(self):
        self.group += 1
        return self

    def _add(self, typ, g, k, fuzz, ff, mat):
        r = np.zeros(NCOL)
        r[0] = typ
        r[1:1 + len(g)] = g
        r[8] = k
        r[9] = fuzz
        r[10] = ff
        r[11] = mat
        r[12] = self.group
        self.rows.append(r)

    def capsule(self, p0, p1, r0, r1=None, k=0.0, mat=0, fuzz=0.0, ff=40.0):
        self._add(CAPSULE, [p0[0], p0[1], p1[0], p1[1], r0, r0 if r1 is None else r1], k, fuzz, ff, mat)

    def ellipse(self, c, rx, ry, ang=0.0, k=0.0, mat=0, fuzz=0.0, ff=40.0):
        self._add(ELLIPSE, [c[0], c[1], rx, ry, ang], k, fuzz, ff, mat)

    def trap(self, pb, pt, wb, wt, rnd=0.0, k=0.0, mat=0, fuzz=0.0, ff=40.0):
        self._add(TRAP, [pb[0], pb[1], pt[0], pt[1], wb, wt, rnd], k, fuzz, ff, mat)

    def tri(self, a, b, c, rnd=0.0, k=0.0, mat=0, fuzz=0.0, ff=40.0):
        self._add(TRI, [a[0], a[1], b[0], b[1], c[0], c[1], rnd], k, fuzz, ff, mat)

    def chain(self, pts, r0, r1, k=0.0, mat=0, fuzz=0.0, ff=40.0):
        n = len(pts)
        for i in range(n - 1):
            ra = r0 + (r1 - r0) * i / max(n - 1, 1)
            rb = r0 + (r1 - r0) * (i + 1) / max(n - 1, 1)
            self.capsule(pts[i], pts[i + 1], ra, rb, k=k, mat=mat, fuzz=fuzz, ff=ff)

    def array(self):
        return np.array(self.rows, np.float64) if self.rows else np.zeros((0, NCOL))

    def bounds(self):
        A = self.array()
        xs, ys = [], []
        for r in A:
            t = int(r[0])
            if t == CAPSULE:
                m = max(r[5], r[6]) + r[9] * 2
                xs += [r[1] - m, r[1] + m, r[3] - m, r[3] + m]
                ys += [r[2] - m, r[2] + m, r[4] - m, r[4] + m]
            elif t == ELLIPSE:
                m = max(r[3], r[4]) + r[9] * 2
                xs += [r[1] - m, r[1] + m]
                ys += [r[2] - m, r[2] + m]
            elif t == TRAP:
                m = max(r[5], r[6]) + r[7] + r[9] * 2
                xs += [r[1] - m, r[1] + m, r[3] - m, r[3] + m]
                ys += [r[2] - m, r[2] + m, r[4] - m, r[4] + m]
            else:
                m = r[7] + r[9] * 2
                xs += [r[1] - m, r[1] + m, r[3] - m, r[3] + m, r[5] - m, r[5] + m]
                ys += [r[2] - m, r[2] + m, r[4] - m, r[4] + m, r[6] - m, r[6] + m]
        return min(xs), max(xs), min(ys), max(ys)


# ------------------------------------------------------------------ SDFs ---

@njit(inline='always', fastmath=True)
def _sd_capsule(px, py, ax, ay, bx, by, ra, rb):
    px -= ax
    py -= ay
    bx -= ax
    by -= ay
    h = bx * bx + by * by
    if h < 1e-12:
        return math.sqrt(px * px + py * py) - max(ra, rb)
    qx = (px * by - py * bx) / h
    qy = (px * bx + py * by) / h
    qx = abs(qx)
    b = ra - rb
    cx = math.sqrt(max(h - b * b, 1e-12))
    cy = b
    k = cx * qy - cy * qx
    m = cx * qx + cy * qy
    n = qx * qx + qy * qy
    if k < 0.0:
        return math.sqrt(h * n) - ra
    elif k > cx:
        return math.sqrt(h * (n + 1.0 - 2.0 * qy)) - rb
    return m - ra


@njit(inline='always', fastmath=True)
def _sd_ellipse(px, py, cx, cy, rx, ry, ang):
    dx = px - cx
    dy = py - cy
    c = math.cos(ang)
    s = math.sin(ang)
    x = c * dx + s * dy
    y = -s * dx + c * dy
    k0 = math.sqrt((x / rx) ** 2 + (y / ry) ** 2)
    k1 = math.sqrt((x / (rx * rx)) ** 2 + (y / (ry * ry)) ** 2)
    if k1 < 1e-9:
        return -min(rx, ry)
    return k0 * (k0 - 1.0) / k1


@njit(inline='always', fastmath=True)
def _sd_trap(px, py, bx, by, tx, ty, wb, wt, rnd):
    ux = tx - bx
    uy = ty - by
    L = math.sqrt(ux * ux + uy * uy) + 1e-12
    ux /= L
    uy /= L
    mx = 0.5 * (bx + tx)
    my = 0.5 * (by + ty)
    he = 0.5 * L
    qx = (px - mx) * uy - (py - my) * ux
    qy = (px - mx) * ux + (py - my) * uy
    r1 = wb
    r2 = wt
    k1x = r2
    k1y = he
    k2x = r2 - r1
    k2y = 2.0 * he
    qx = abs(qx)
    rr = r1 if qy < 0.0 else r2
    cax = qx - min(qx, rr)
    cay = abs(qy) - he
    t = ((k1x - qx) * k2x + (k1y - qy) * k2y) / (k2x * k2x + k2y * k2y)
    t = min(max(t, 0.0), 1.0)
    cbx = qx - k1x + k2x * t
    cby = qy - k1y + k2y * t
    s = -1.0 if (cbx < 0.0 and cay < 0.0) else 1.0
    return s * math.sqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby)) - rnd


@njit(inline='always', fastmath=True)
def _sd_tri(px, py, ax, ay, bx, by, cx, cy, rnd):
    e0x = bx - ax
    e0y = by - ay
    e1x = cx - bx
    e1y = cy - by
    e2x = ax - cx
    e2y = ay - cy
    v0x = px - ax
    v0y = py - ay
    v1x = px - bx
    v1y = py - by
    v2x = px - cx
    v2y = py - cy
    t0 = min(max((v0x * e0x + v0y * e0y) / (e0x * e0x + e0y * e0y), 0.0), 1.0)
    t1 = min(max((v1x * e1x + v1y * e1y) / (e1x * e1x + e1y * e1y), 0.0), 1.0)
    t2 = min(max((v2x * e2x + v2y * e2y) / (e2x * e2x + e2y * e2y), 0.0), 1.0)
    p0x = v0x - e0x * t0
    p0y = v0y - e0y * t0
    p1x = v1x - e1x * t1
    p1y = v1y - e1y * t1
    p2x = v2x - e2x * t2
    p2y = v2y - e2y * t2
    s = 1.0 if (e0x * e2y - e0y * e2x) > 0 else -1.0
    d0 = p0x * p0x + p0y * p0y
    d1 = p1x * p1x + p1y * p1y
    d2 = p2x * p2x + p2y * p2y
    c0 = s * (v0x * e0y - v0y * e0x)
    c1 = s * (v1x * e1y - v1y * e1x)
    c2 = s * (v2x * e2y - v2y * e2x)
    d = min(d0, min(d1, d2))
    c = min(c0, min(c1, c2))
    sg = 1.0 if c > 0.0 else -1.0
    return -math.sqrt(d) * sg - rnd


@njit(inline='always', fastmath=True)
def _prim(P, i, x, y, seed):
    t = int(P[i, 0])
    if t == 0:
        d = _sd_capsule(x, y, P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6])
    elif t == 1:
        d = _sd_ellipse(x, y, P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5])
    elif t == 2:
        d = _sd_trap(x, y, P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6], P[i, 7])
    else:
        d = _sd_tri(x, y, P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6], P[i, 7])
    fz = P[i, 9]
    if fz > 0.0 and d < 3.0 * fz:
        ff = P[i, 10]
        d += fz * (gnoise2(x * ff, y * ff, seed + i) * 0.7 + 0.3 * gnoise2(x * ff * 2.7, y * ff * 2.7, seed + 3 * i))
    return d


@njit(inline='always', fastmath=True)
def _group_sdf(P, g0, g1, x, y, seed):
    d = 1e9
    dm = 1e9
    mat = 0
    for i in range(g0, g1):
        di = _prim(P, i, x, y, seed)
        if di < dm:
            dm = di
            mat = int(P[i, 11])
        k = P[i, 8]
        if k > 0.0 and d < 1e8:
            h = max(k - abs(d - di), 0.0) / k
            d = min(d, di) - h * h * k * 0.25
        else:
            d = min(d, di)
    return d, mat


@njit(parallel=True, fastmath=True, cache=True)
def _render(img, depth, P, gidx, M, L, amb, fx, fy, ppm, zf, x0, x1, y0, y1, seed, t, fogT, fogc, flipx,
            write_depth, zbias):
    ng = gidx.shape[0] - 1
    e = 0.6 / ppm
    for py in prange(y0, y1):
        for px in range(x0, x1):
            if depth[py, px] < zf - zbias:
                continue
            X = (px + 0.5 - fx) / ppm
            if flipx:
                X = -X
            Y = (fy - (py + 0.5)) / ppm
            for g in range(ng):
                a0 = gidx[g]
                a1 = gidx[g + 1]
                d, mat = _group_sdf(P, a0, a1, X, Y, seed)
                cov = 0.5 - d * ppm
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                dx1, _m = _group_sdf(P, a0, a1, X + e, Y, seed)
                dy1, _m = _group_sdf(P, a0, a1, X, Y + e, seed)
                gx = dx1 - d
                gy = dy1 - d
                gl = math.sqrt(gx * gx + gy * gy) + 1e-12
                gx /= gl
                gy /= gl
                if flipx:
                    gx = -gx
                thick = M[mat, 6]
                di = max(-d, 0.0)
                u = min(di / thick, 1.0)
                nz = math.sqrt(max(1.0 - (1.0 - u) * (1.0 - u), 0.0))
                sl = math.sqrt(max(1.0 - nz * nz, 0.0))
                nx = gx * sl
                ny = gy * sl
                Xs = X if not flipx else -X
                ar = M[mat, 0]
                ag = M[mat, 1]
                ab = M[mat, 2]
                rim_k = M[mat, 3]
                spk = M[mat, 4]
                spp = M[mat, 5]
                cr = amb[0] * ar * (0.6 + 0.4 * ny)
                cg = amb[1] * ag * (0.6 + 0.4 * ny)
                cb = amb[2] * ab * (0.6 + 0.4 * ny)
                edge = (1.0 - nz) ** 2.5
                for li in range(L.shape[0]):
                    if L[li, 0] < 0.5:
                        lx = L[li, 1] - Xs
                        ly = L[li, 2] - Y
                        lz = L[li, 3]
                        d2 = lx * lx + ly * ly + lz * lz
                        E = L[li, 7] / (d2 + L[li, 8] * L[li, 8])
                        inv = 1.0 / math.sqrt(d2 + 1e-12)
                        lx *= inv
                        ly *= inv
                        lz *= inv
                    else:
                        lx = L[li, 1]
                        ly = L[li, 2]
                        lz = L[li, 3]
                        E = L[li, 7]
                    ndl = nx * lx + ny * ly + nz * lz
                    diff = max(ndl, 0.0)
                    # wrap for cloth + rim at grazing edges + halo when the light is behind
                    wrap = max((ndl + 0.35) / 1.35, 0.0)
                    rim = edge * (wrap + 1.4 * max(-lz, 0.0) ** 0.5) * rim_k
                    spec = 0.0
                    if spk > 0.0:
                        hz = lz + 1.0
                        hx = lx
                        hy = ly
                        hl = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
                        nh = max((nx * hx + ny * hy + nz * hz) / hl, 0.0)
                        spec = spk * nh ** spp
                    k_ = E * (0.85 * diff + 0.15 * wrap)
                    cr += L[li, 4] * (ar * k_ + E * (rim * 0.10 + spec * 0.05))
                    cg += L[li, 5] * (ag * k_ + E * (rim * 0.10 + spec * 0.05))
                    cb += L[li, 6] * (ab * k_ + E * (rim * 0.10 + spec * 0.05))
                if M[mat, 7] + M[mat, 8] + M[mat, 9] > 0.0:
                    fq = M[mat, 10]
                    n = gnoise3(X * fq, Y * fq, t * 1.5, 77) * 0.6 + 0.4 * gnoise3(X * fq * 2.3, Y * fq * 2.3, t * 2.5, 78)
                    em = max(n + 0.25, 0.0) ** 2 * 2.5 * u
                    cr += M[mat, 7] * em
                    cg += M[mat, 8] * em
                    cb += M[mat, 9] * em
                # atmospheric fog over the figure
                cr = cr * fogT + fogc[0]
                cg = cg * fogT + fogc[1]
                cb = cb * fogT + fogc[2]
                img[py, px, 0] = img[py, px, 0] * (1.0 - cov) + cr * cov
                img[py, px, 1] = img[py, px, 1] * (1.0 - cov) + cg * cov
                img[py, px, 2] = img[py, px, 2] * (1.0 - cov) + cb * cov
                if write_depth and cov > 0.5:
                    depth[py, px] = zf


def render(img, depth, cam, drawing, feet_world, lights_world, amb=(0.01, 0.012, 0.02), mats=None, seed=0,
           t=0.0, fog_T=1.0, fog_col=(0, 0, 0), flipx=False, write_depth=True, zbias=0.05, emissive_gain=1.0):
    """Render a Drawing standing at world point feet_world, facing the camera.
    lights_world: list of dicts {pos:(x,y,z) or dir:(x,y,z), col:(r,g,b), I, r0}."""
    A = drawing.array() if isinstance(drawing, Drawing) else drawing
    if len(A) == 0:
        return
    Pw = np.asarray(feet_world, np.float64)
    sx, sy, z = cam.project(Pw)
    if z <= 0.1:
        return
    ppm = cam.f / z
    # group index table (primitives must be contiguous per group; keep order)
    groups = A[:, 12]
    order = np.argsort(groups, kind='stable')
    A = A[order]
    g = A[:, 12]
    starts = np.nonzero(np.r_[True, g[1:] != g[:-1]])[0]
    gidx = np.r_[starts, len(A)].astype(np.int64)
    # bounds
    xmin, xmax, ymin, ymax = (drawing.bounds() if isinstance(drawing, Drawing) else _bounds_arr(A))
    if flipx:
        xmin, xmax = -xmax, -xmin
    H, W = img.shape[:2]
    x0 = int(max(0, math.floor(sx + xmin * ppm - 2)))
    x1 = int(min(W, math.ceil(sx + xmax * ppm + 2)))
    y0 = int(max(0, math.floor(sy - ymax * ppm - 2)))
    y1 = int(min(H, math.ceil(sy - ymin * ppm + 2)))
    if x1 <= x0 or y1 <= y0:
        return
    L = []
    for lw in lights_world:
        row = np.zeros(9)
        if 'pos' in lw:
            d = np.asarray(lw['pos'], np.float64) - Pw
            row[0] = 0
            row[1] = d[0] * cam.right[0] + d[2] * cam.right[2]
            row[2] = d[1]
            row[3] = -(d[0] * cam.fwd[0] + d[2] * cam.fwd[2])
        else:
            d = np.asarray(lw['dir'], np.float64)
            d = d / np.linalg.norm(d)
            row[0] = 1
            row[1] = d[0] * cam.right[0] + d[2] * cam.right[2]
            row[2] = d[1]
            row[3] = -(d[0] * cam.fwd[0] + d[2] * cam.fwd[2])
        if flipx:
            row[1] = -row[1]
        row[4:7] = lw['col']
        row[7] = lw['I']
        row[8] = lw.get('r0', 0.3)
        L.append(row)
    L = np.array(L, np.float64) if L else np.zeros((0, 9))
    M = (MATS if mats is None else mats).copy()
    M[:, 7:10] *= emissive_gain
    _render(img, depth, A, gidx, M, L, np.asarray(amb, np.float64), float(sx), float(sy), float(ppm), float(z),
            x0, x1, y0, y1, int(seed), float(t), float(fog_T), np.asarray(fog_col, np.float64), bool(flipx),
            bool(write_depth), float(zbias))


def _bounds_arr(A):
    d = Drawing()
    d.rows = list(A)
    return d.bounds()


# ------------------------------------------------------------------ cloth ---

def verlet_chain(anchor_fn, n, seg, t0, t1, wind_fn, dt=1 / 96.0, gravity=-9.8, drag=2.2, iters=4,
                 init_dir=(0.0, -1.0), stiffness=0.0):
    """Simulate a hanging chain (figure-local metres) from t0 to t1 with a moving anchor.
    anchor_fn(t) -> (x, y). wind_fn(t, y) -> (wx, wy) wind velocity. Returns (n,2) positions."""
    a = np.asarray(anchor_fn(t0), np.float64)
    dvec = np.asarray(init_dir, np.float64)
    dvec /= np.linalg.norm(dvec)
    p = a[None, :] + dvec[None, :] * seg * np.arange(n)[:, None]
    prev = p.copy()
    t = t0
    steps = max(1, int(math.ceil((t1 - t0) / dt)))
    h = (t1 - t0) / steps if steps > 0 else dt
    for _ in range(steps):
        t += h
        a = np.asarray(anchor_fn(t), np.float64)
        vel = (p - prev) / h
        acc = np.zeros_like(p)
        acc[:, 1] += gravity
        w = np.array([wind_fn(t, p[i, 1]) for i in range(n)])
        # aerodynamic drag toward the wind; stronger on the free end
        wk = drag * (0.4 + 0.6 * np.arange(n) / max(n - 1, 1))[:, None]
        acc += wk * (w - vel)
        newp = p + (p - prev) * 0.985 + acc * h * h
        prev = p
        p = newp
        p[0] = a
        for _it in range(iters):
            dd = p[1:] - p[:-1]
            L = np.linalg.norm(dd, axis=1) + 1e-9
            corr = (L - seg) / L
            c = dd * corr[:, None] * 0.5
            p[1:] -= c
            p[:-1] += c
            p[0] = a
            if stiffness > 0 and n > 2:
                # bending stiffness: pull toward straight continuation
                mid = 0.5 * (p[2:] + p[:-2])
                p[1:-1] += (mid - p[1:-1]) * stiffness
                p[0] = a
    return p
