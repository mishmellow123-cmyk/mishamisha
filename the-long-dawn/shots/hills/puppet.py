"""2-D SDF puppets on camera-facing cards, lit in 2.5-D.

A Figure is a set of Groups drawn in order. Each Group is a smooth union of SDF
primitives (ellipse, uneven capsule, polygon, box, ring) in the card's local metres
(x right, y up, origin on the ground). Rasterisation is per-primitive bounding box,
so cost ~ covered area. Each pixel intersects its camera ray with the card plane
(exact under pitch/roll). Lighting: pillow normal from the SDF, point lights with
falloff, rim sheen, cool sky ambient; the result is premultiplied RGBA.
"""
import math

import cv2
import numba as nb
import numpy as np

from core import cam_ray

P_ELL, P_CONE, P_POLY, P_BOX, P_RING = 0, 1, 2, 3, 4
NP = 12  # floats per primitive row


# ------------------------------------------------------------ primitives ---

class Group:
    def __init__(self, name, albedo, k=0.02, bevel=0.05, sheen=0.6, rough=1.0, emissive=None,
                 sky_rim=0.35, translucent=0.0, per_prim=False, tint=(1.0, 0.9, 0.8), soft=0.0,
                 diffuse=0.0):
        self.name = name
        self.albedo = np.asarray(albedo, np.float64)
        self.k = k
        self.bevel = bevel
        self.sheen = sheen
        self.sky_rim = sky_rim
        self.translucent = translucent
        self.emissive = emissive
        self.per_prim = per_prim
        self.tint = np.asarray(tint, np.float64)
        self.soft = soft
        self.diffuse = diffuse
        self.rows = []
        self.verts = []
        self.bbs = []

    def _add(self, typ, k, params, bb):
        row = np.zeros(NP)
        row[0] = typ
        row[1] = self.k if k is None else k
        row[2:2 + len(params)] = params
        self.rows.append(row)
        self.bbs.append(bb)

    def ellipse(self, c, rx, ry=None, rot=0.0, k=None):
        ry = rx if ry is None else ry
        r = max(rx, ry)
        self._add(P_ELL, k, [c[0], c[1], rx, ry, math.radians(rot)], (c[0] - r, c[1] - r, c[0] + r, c[1] + r))

    circle = ellipse

    def cone(self, a, b, ra, rb, k=None):
        r = max(ra, rb)
        self._add(P_CONE, k, [a[0], a[1], b[0], b[1], ra, rb],
                  (min(a[0], b[0]) - r, min(a[1], b[1]) - r, max(a[0], b[0]) + r, max(a[1], b[1]) + r))

    def chain(self, pts, radii, k=None):
        pts = np.asarray(pts, np.float64)
        radii = np.broadcast_to(np.asarray(radii, np.float64), (len(pts),))
        for i in range(len(pts) - 1):
            self.cone(pts[i], pts[i + 1], radii[i], radii[i + 1], k)

    def poly(self, pts, r=0.0, k=None):
        pts = np.asarray(pts, np.float64)
        s = len(self.verts)
        self.verts.extend(pts.tolist())
        self._add(P_POLY, k, [s, len(pts), r], (pts[:, 0].min() - r, pts[:, 1].min() - r,
                                                  pts[:, 0].max() + r, pts[:, 1].max() + r))

    def box(self, c, hx, hy, rot=0.0, rnd=0.0, k=None):
        r = math.hypot(hx, hy) + rnd
        self._add(P_BOX, k, [c[0], c[1], hx, hy, math.radians(rot), rnd], (c[0] - r, c[1] - r, c[0] + r, c[1] + r))

    def ring(self, c, rx, ry, thick, rot=0.0, k=None):
        r = max(rx, ry) + thick
        self._add(P_RING, k, [c[0], c[1], rx, ry, math.radians(rot), thick], (c[0] - r, c[1] - r, c[0] + r, c[1] + r))

    def arrays(self):
        rows = np.array(self.rows, np.float64) if self.rows else np.zeros((0, NP))
        verts = np.array(self.verts, np.float64) if self.verts else np.zeros((1, 2))
        return rows, verts, np.array(self.bbs, np.float64).reshape(-1, 4)


# ------------------------------------------------------------ SDF kernels ---

@nb.njit(cache=True, fastmath=True, inline='always')
def _smin(a, b, k):
    if k <= 0.0:
        return a if a < b else b
    h = k - abs(a - b)
    if h < 0.0:
        h = 0.0
    h /= k
    m = a if a < b else b
    return m - h * h * k * 0.25


@nb.njit(cache=True, fastmath=True, inline='always')
def _sd_ellipse(x, y, a, b):
    # IQ approximation
    k0 = math.sqrt((x / a) ** 2 + (y / b) ** 2)
    k1 = math.sqrt((x / (a * a)) ** 2 + (y / (b * b)) ** 2)
    if k1 < 1e-9:
        return -min(a, b)
    return k0 * (k0 - 1.0) / k1


@nb.njit(cache=True, fastmath=True, inline='always')
def _sd_cone(px, py, ax, ay, bx, by, ra, rb):
    dx = bx - ax
    dy = by - ay
    h = math.sqrt(dx * dx + dy * dy)
    if h < 1e-7:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2) - max(ra, rb)
    ux = dx / h
    uy = dy / h
    lx = px - ax
    ly = py - ay
    qy = lx * ux + ly * uy
    qx = abs(-lx * uy + ly * ux)
    b = (ra - rb) / h
    if abs(b) >= 0.999:
        d1 = math.sqrt(qx * qx + qy * qy) - ra
        d2 = math.sqrt(qx * qx + (qy - h) ** 2) - rb
        return min(d1, d2)
    a = math.sqrt(1.0 - b * b)
    k = -b * qx + a * qy
    if k < 0.0:
        return math.sqrt(qx * qx + qy * qy) - ra
    if k > a * h:
        return math.sqrt(qx * qx + (qy - h) ** 2) - rb
    return qx * a + qy * b - ra


@nb.njit(cache=True, fastmath=True)
def _sd_poly(px, py, verts, s, n):
    d = (px - verts[s, 0]) ** 2 + (py - verts[s, 1]) ** 2
    sg = 1.0
    j = s + n - 1
    for i in range(s, s + n):
        ex = verts[j, 0] - verts[i, 0]
        ey = verts[j, 1] - verts[i, 1]
        wx = px - verts[i, 0]
        wy = py - verts[i, 1]
        ee = ex * ex + ey * ey
        t = 0.0
        if ee > 1e-12:
            t = (wx * ex + wy * ey) / ee
            if t < 0.0:
                t = 0.0
            if t > 1.0:
                t = 1.0
        bx = wx - ex * t
        by = wy - ey * t
        dd = bx * bx + by * by
        if dd < d:
            d = dd
        c1 = py >= verts[i, 1]
        c2 = py < verts[j, 1]
        c3 = ex * wy > ey * wx
        if (c1 and c2 and c3) or ((not c1) and (not c2) and (not c3)):
            sg = -sg
        j = i
    return sg * math.sqrt(d)


@nb.njit(cache=True, fastmath=True)
def _sd_prim(row, verts, x, y):
    typ = int(row[0])
    if typ == 0:
        c = math.cos(row[6])
        s = math.sin(row[6])
        lx = x - row[2]
        ly = y - row[3]
        rx = c * lx + s * ly
        ry = -s * lx + c * ly
        return _sd_ellipse(rx, ry, row[4], row[5])
    elif typ == 1:
        return _sd_cone(x, y, row[2], row[3], row[4], row[5], row[6], row[7])
    elif typ == 2:
        return _sd_poly(x, y, verts, int(row[2]), int(row[3])) - row[4]
    elif typ == 3:
        c = math.cos(row[6])
        s = math.sin(row[6])
        lx = x - row[2]
        ly = y - row[3]
        rx = abs(c * lx + s * ly) - row[4]
        ry = abs(-s * lx + c * ly) - row[5]
        ox = max(rx, 0.0)
        oy = max(ry, 0.0)
        return math.sqrt(ox * ox + oy * oy) + min(max(rx, ry), 0.0) - row[7]
    else:
        c = math.cos(row[6])
        s = math.sin(row[6])
        lx = x - row[2]
        ly = y - row[3]
        rx = c * lx + s * ly
        ry = -s * lx + c * ly
        return abs(_sd_ellipse(rx, ry, row[4], row[5])) - row[7] * 0.5


@nb.njit(cache=True, fastmath=True)
def local_grid(LX, LY, cam, ox, oy, oz, j0, i0):
    """Per-pixel intersection of camera rays with the card plane z=oz; local coords
    relative to (ox, oy)."""
    h = LX.shape[0]
    w = LX.shape[1]
    for j in range(h):
        for i in range(w):
            dx, dy, dz = cam_ray(cam, i0 + i + 0.5, j0 + j + 0.5)
            if dz <= 1e-7:
                LX[j, i] = 1e6
                LY[j, i] = 1e6
                continue
            t = (oz - cam[2]) / dz
            LX[j, i] = cam[0] + t * dx - ox
            LY[j, i] = cam[1] + t * dy - oy


@nb.njit(cache=True, fastmath=True)
def raster_group(D, Dp, LX, LY, rows, verts, ranges):
    """D = smooth union (coverage); Dp = distance to the nearest single primitive (for
    per-primitive shading of stones/logs)."""
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


@nb.njit(cache=True, fastmath=True)
def shade_group(D, N, LX, LY, ppm, alb, lights, nl, amb_top, amb_bot, bevel, sheen, sky_rim,
                translucent, emis, back, tint, soft, diffuse, out_rgb, out_a):
    """Backlit-silhouette shading.
    * interiors: albedo * faint ambient (reads as #05060B-ish)
    * sky backlight rim: a thin rim of the bright sky colour (`back`) around the outline,
      strongest on edges facing up/sideways (sky_rim)
    * light rim: a thin warm rim on edges facing each light (sheen), plus an optional soft
      band (soft) and lambert (diffuse) for materials that should glow (the scarf).
    lights: (n, 8) [lx, ly, lz, r, g, b, d0, wrap] card-local metres (lz toward camera)."""
    h = D.shape[0]
    w = D.shape[1]
    rw = max(0.0022, 1.0 / ppm)
    rw_soft = max(0.006, 2.5 / ppm)
    for j in range(h):
        for i in range(w):
            d = D[j, i]
            cov = 0.5 - d * ppm
            if cov <= 0.0:
                continue
            if cov > 1.0:
                cov = 1.0
            il = i - 1 if i > 0 else i
            ir = i + 1 if i < w - 1 else i
            ju = j - 1 if j > 0 else j
            jd = j + 1 if j < h - 1 else j
            ddx = LX[j, ir] - LX[j, il]
            ddy = LY[ju, i] - LY[jd, i]
            gx = 0.0
            gy = 0.0
            if abs(ddx) > 1e-9:
                gx = (min(N[j, ir], 1.0) - min(N[j, il], 1.0)) / ddx
            if abs(ddy) > 1e-9:
                gy = (min(N[ju, i], 1.0) - min(N[jd, i], 1.0)) / ddy
            gn = math.sqrt(gx * gx + gy * gy)
            if gn > 1e-9:
                gx /= gn
                gy /= gn
            depth = -N[j, i]
            if depth < 0.0:
                depth = 0.0
            rs = math.exp(-depth / rw)
            rsoft = math.exp(-depth / rw_soft)
            e = depth / bevel
            if e > 1.0:
                e = 1.0
            oe = 1.0 - e
            nz = math.sqrt(max(0.0, 1.0 - oe * oe)) + 0.05
            nx = gx * oe
            ny = gy * oe
            nn = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= nn
            ny /= nn
            nz /= nn
            x = LX[j, i]
            y = LY[j, i]
            r = 0.0
            g = 0.0
            b = 0.0
            for q in range(nl):
                vx = lights[q, 0] - x
                vy = lights[q, 1] - y
                vz = lights[q, 2]
                dist = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-6
                vx /= dist
                vy /= dist
                vz /= dist
                att = 1.0 / (1.0 + (dist / lights[q, 6]) ** 2)
                pl = math.sqrt(vx * vx + vy * vy) + 1e-6
                rimd = (gx * vx + gy * vy) / pl
                if rimd < 0.0:
                    rimd = 0.0
                rim = sheen * rs * rimd * rimd + soft * rsoft * rimd
                lam = 0.0
                if diffuse > 0.0:
                    lam = nx * vx + ny * vy + nz * vz
                    if lam < 0.0:
                        lam = 0.0
                    lam *= diffuse
                tr = 0.0
                if translucent > 0.0 and vz < 0.0:
                    tr = translucent * (-vz)
                lr = lights[q, 3] * att
                lg = lights[q, 4] * att
                lb = lights[q, 5] * att
                r += lr * (alb[0] * (lam + tr) + rim * tint[0])
                g += lg * (alb[1] * (lam + tr) + rim * tint[1])
                b += lb * (alb[2] * (lam + tr) + rim * tint[2])
            up = 0.5 + 0.5 * ny
            r += alb[0] * (amb_top[0] * up + amb_bot[0] * (1 - up))
            g += alb[1] * (amb_top[1] * up + amb_bot[1] * (1 - up))
            b += alb[2] * (amb_top[2] * up + amb_bot[2] * (1 - up))
            # sky backlight rim (thin), stronger on upward/sideways edges
            sr = sky_rim * rs * (0.45 + 0.55 * max(0.0, gy))
            r += back[0] * sr
            g += back[1] * sr
            b += back[2] * sr
            r += emis[0] + translucent * back[0] * alb[0] * 1.5
            g += emis[1] + translucent * back[1] * alb[1] * 1.5
            b += emis[2] + translucent * back[2] * alb[2] * 1.5
            ao = 1.0 - cov
            out_rgb[j, i, 0] = out_rgb[j, i, 0] * ao + r * cov
            out_rgb[j, i, 1] = out_rgb[j, i, 1] * ao + g * cov
            out_rgb[j, i, 2] = out_rgb[j, i, 2] * ao + b * cov
            out_a[j, i] = out_a[j, i] * ao + cov


# ------------------------------------------------------------------ figure ---

class Figure:
    """Ordered groups on one card. origin = world position of local (0,0); the card plane
    is z = origin.z (camera-facing for cameras looking roughly along +z)."""

    def __init__(self, origin, groups):
        self.origin = np.asarray(origin, np.float64)
        self.groups = [g for g in groups if g is not None and len(g.rows)]

    def local_bbox(self, pad=0.05):
        bbs = np.concatenate([g.arrays()[2] for g in self.groups], 0)
        return bbs[:, 0].min() - pad, bbs[:, 1].min() - pad, bbs[:, 2].max() + pad, bbs[:, 3].max() + pad

    def render(self, cam, lights, amb_top, amb_bot, blur_px=0.0, back=(0.0, 0.0, 0.0)):
        """Returns (y0, x0, rgb_premul, alpha) screen-space patch, or None."""
        x0l, y0l, x1l, y1l = self.local_bbox()
        o = self.origin
        corners = np.array([[o[0] + x0l, o[1] + y0l, o[2]], [o[0] + x1l, o[1] + y0l, o[2]],
                            [o[0] + x0l, o[1] + y1l, o[2]], [o[0] + x1l, o[1] + y1l, o[2]]])
        sx, sy, z = cam.project(corners)
        if np.any(z <= 0.05):
            return None
        pad = int(3 + 3 * blur_px)
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
        rgb = np.zeros((h, w, 3), np.float64)
        alpha = np.zeros((h, w), np.float64)
        L = np.asarray(lights, np.float64).reshape(-1, 8).copy()
        L[:, 0] -= o[0]
        L[:, 1] -= o[1]
        L[:, 2] = o[2] - L[:, 2]      # world z -> toward-camera positive
        for g in self.groups:
            rows, verts, bbs = g.arrays()
            # pixel ranges of each primitive (project local bbox corners)
            kk = np.maximum(rows[:, 1], g.k) + 1.5 / ppm
            cx = np.stack([bbs[:, 0] - kk, bbs[:, 2] + kk], 1) + o[0]
            cy = np.stack([bbs[:, 1] - kk, bbs[:, 3] + kk], 1) + o[1]
            P = np.stack([np.stack([cx[:, a], cy[:, b], np.full(len(cx), o[2])], 1)
                          for a in (0, 1) for b in (0, 1)], 1)  # (n,4,3)
            psx, psy, pz = cam.project(P.reshape(-1, 3))
            psx = psx.reshape(-1, 4)
            psy = psy.reshape(-1, 4)
            ranges = np.zeros((len(rows), 4), np.int64)
            ranges[:, 0] = np.clip(np.floor(psy.min(1)) - Y0 - 1, 0, h)
            ranges[:, 1] = np.clip(np.ceil(psy.max(1)) - Y0 + 2, 0, h)
            ranges[:, 2] = np.clip(np.floor(psx.min(1)) - X0 - 1, 0, w)
            ranges[:, 3] = np.clip(np.ceil(psx.max(1)) - X0 + 2, 0, w)
            D = np.full((h, w), 10.0, np.float64)
            Dp = np.full((h, w), 10.0, np.float64)
            raster_group(D, Dp, LX, LY, rows, verts, ranges)
            em = np.zeros(3) if g.emissive is None else np.asarray(g.emissive, np.float64)
            shade_group(D, Dp if g.per_prim else D, LX, LY, ppm, g.albedo, L, len(L), np.asarray(amb_top, np.float64),
                        np.asarray(amb_bot, np.float64), g.bevel, g.sheen, g.sky_rim, g.translucent,
                        em, np.asarray(back, np.float64), g.tint, g.soft, g.diffuse, rgb, alpha)
        rgb = rgb.astype(np.float32)
        alpha = alpha.astype(np.float32)
        if blur_px > 0.3:
            rgb = cv2.GaussianBlur(rgb, (0, 0), blur_px)
            alpha = cv2.GaussianBlur(alpha, (0, 0), blur_px)
        return Y0, X0, rgb, alpha


# ------------------------------------------------------------------ verlet ---

@nb.njit(cache=True, fastmath=True)
def verlet_steps(P, Pp, rest, pin_idx, pins, nsteps, dt, grav, wind, drag, iters, damp):
    """P,Pp: (n,2) current/previous. pins: (nsteps, npin, 2) target positions per step for
    the pinned indices. wind: (nsteps, n, 2) air velocity at each node per step."""
    n = P.shape[0]
    npin = pin_idx.shape[0]
    for s in range(nsteps):
        for k in range(n):
            vx = (P[k, 0] - Pp[k, 0]) * damp
            vy = (P[k, 1] - Pp[k, 1]) * damp
            Pp[k, 0] = P[k, 0]
            Pp[k, 1] = P[k, 1]
            # aerodynamic drag toward the wind velocity
            ax = drag * (wind[s, k, 0] - vx / dt)
            ay = drag * (wind[s, k, 1] - vy / dt) - grav
            P[k, 0] += vx + ax * dt * dt
            P[k, 1] += vy + ay * dt * dt
        for q in range(npin):
            P[pin_idx[q], 0] = pins[s, q, 0]
            P[pin_idx[q], 1] = pins[s, q, 1]
        for it in range(iters):
            for k in range(n - 1):
                dx = P[k + 1, 0] - P[k, 0]
                dy = P[k + 1, 1] - P[k, 1]
                L = math.sqrt(dx * dx + dy * dy) + 1e-9
                c = (L - rest[k]) / L
                w0 = 0.5
                w1 = 0.5
                for q in range(npin):
                    if pin_idx[q] == k:
                        w0 = 0.0
                        w1 = 1.0
                    if pin_idx[q] == k + 1:
                        w1 = 0.0
                        w0 = 1.0
                P[k, 0] += dx * c * w0
                P[k, 1] += dy * c * w0
                P[k + 1, 0] -= dx * c * w1
                P[k + 1, 1] -= dy * c * w1
            # bending stiffness (keep second neighbours apart)
            for k in range(n - 2):
                dx = P[k + 2, 0] - P[k, 0]
                dy = P[k + 2, 1] - P[k, 1]
                L = math.sqrt(dx * dx + dy * dy) + 1e-9
                mn = (rest[k] + rest[k + 1]) * 0.55
                if L < mn:
                    c = (L - mn) / L * 0.5
                    P[k, 0] += dx * c * 0.5
                    P[k, 1] += dy * c * 0.5
                    P[k + 2, 0] -= dx * c * 0.5
                    P[k + 2, 1] -= dy * c * 0.5
            for q in range(npin):
                P[pin_idx[q], 0] = pins[s, q, 0]
                P[pin_idx[q], 1] = pins[s, q, 1]


class Chain:
    """A pinned verlet chain simulated over a frame range; positions cached per frame."""

    def __init__(self, n, seg, anchor_fn, dir0, grav=9.8, drag=6.0, iters=6, damp=0.985,
                 substeps=8, fps=24.0):
        self.n = n
        self.seg = seg
        self.anchor_fn = anchor_fn      # f -> (x, y) of node 0 (and optional node 1 dir)
        self.rest = np.full(n - 1, seg)
        self.dir0 = np.asarray(dir0, np.float64) / np.linalg.norm(dir0)
        self.grav, self.drag, self.iters, self.damp = grav, drag, iters, damp
        self.sub = substeps
        self.fps = fps
        self.cache = {}

    def simulate(self, f0, f1, wind_fn, pin_fn=None, prewarm=48):
        """wind_fn(f, pts) -> (n,2) air velocity; pin_fn(f) -> (npin idx list, (npin,2))."""
        a = np.asarray(self.anchor_fn(f0 - prewarm), np.float64)
        P = np.array([a + self.dir0 * self.seg * i for i in range(self.n)], np.float64)
        Pp = P.copy()
        dt = 1.0 / (self.fps * self.sub)
        for f in range(f0 - prewarm, f1 + 1):
            ns = self.sub
            winds = np.zeros((ns, self.n, 2))
            if pin_fn is None:
                idx = np.array([0], np.int64)
                pins = np.zeros((ns, 1, 2))
                for s in range(ns):
                    pins[s, 0] = self.anchor_fn(f + s / ns)
            else:
                idx, _ = pin_fn(f)
                idx = np.asarray(idx, np.int64)
                pins = np.zeros((ns, len(idx), 2))
                for s in range(ns):
                    pins[s] = pin_fn(f + s / ns)[1]
            for s in range(ns):
                winds[s] = wind_fn(f + s / ns, P)
            verlet_steps(P, Pp, self.rest, idx, pins, ns, dt, self.grav, winds, self.drag,
                         self.iters, self.damp)
            if f >= f0:
                self.cache[f] = P.copy()
        return self

    def at(self, f):
        f = int(round(f))
        if f in self.cache:
            return self.cache[f]
        ks = sorted(self.cache)
        return self.cache[min(ks, key=lambda k: abs(k - f))]
