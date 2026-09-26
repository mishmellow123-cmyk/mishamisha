"""The festival beacon's cairn, v2 (CODA dept, Sep 2026): irregular, flat-bedded dry stone.

v1 drew rounded boxes with a bevel all round - "rounded pillows". Here each stone is an
irregular polygon laid in courses (flat beds, broken ends, chipped corners, staggered joints,
a few pinning stones, a through-capstone), with dark gaps into the hearting behind. Shading is
per stone face (a random tilt), plus a narrow hard chamfer along every edge, so the edges that
face a light catch a crisp line (the fire above lights the top arrises: horizontal bright lines,
the look of dry stone at night) while the undersides fall into shadow. Weathered-stone albedo
with grain and lichen. The silhouette (stones + hearting) also gets the sky rim from what is
behind it, like the figures.

The basket, the stacked wood and every dimension that other code uses (top, bk_bot, bk_top,
bk_rb, bk_rt, fire_base) are identical to characters.Cairn, so the fire - and the ember title
that rises from it - stay exactly where they were.
"""
import math

import cv2
import numba as nb
import numpy as np

import characters as ch
from core import perlin3
from puppet import local_grid, _sd_poly, Figure
from silhouette import exact_depth, _fbm2n


class DryStoneCairn:
    def __init__(self, seed=7, base_hw=0.40, top_hw=0.285):
        old = ch.Cairn(seed=seed)
        self.old = old
        self.height = old.height
        self.top = old.top
        self.bk_bot, self.bk_top = old.bk_bot, old.bk_top
        self.bk_rb, self.bk_rt = old.bk_rb, old.bk_rt
        self.bars = old.bars
        self.logs = old.logs
        rng = np.random.default_rng(seed * 101 + 5)
        cap_h = 0.062
        stones = []            # (verts (n,2), normal (3,), albedo (3,), seed)

        def hw_at(y):
            return base_hw + (top_hw - base_hw) * (y / self.top)

        def stone(x0, x1, yb, yt, tilt, jag=1.0):
            """Flat-bedded stone: flat top/bottom (slightly tilted), broken ends, chipped corners."""
            L = x1 - x0
            H = yt - yb
            cx = 0.5 * (x0 + x1)
            tb = math.tan(math.radians(tilt))
            tt = math.tan(math.radians(tilt + rng.normal(0, 1.5)))
            yb0 = yb + (x0 - cx) * tb
            yb1 = yb + (x1 - cx) * tb
            yt0 = yt + (x0 - cx) * tt
            yt1 = yt + (x1 - cx) * tt
            c = lambda: rng.uniform(0.15, 0.45) * min(H, 0.05) * jag     # chipped corner size
            e = lambda: rng.uniform(-0.25, 0.25) * H * jag                # broken end wander
            pts = [(x0 + c() * 0.6, yb0),
                   (x0 + 0.35 * L + rng.uniform(-0.05, 0.05) * L, yb + (x0 + 0.35 * L - cx) * tb + rng.uniform(-0.003, 0.002)),
                   (x1 - c() * 0.7, yb1),
                   (x1 + e() * 0.35, yb1 + c()),
                   (x1 - abs(e()) * 0.4, 0.5 * (yb1 + yt1) + e() * 0.4),
                   (x1 + e() * 0.3, yt1 - c()),
                   (x1 - c(), yt1),
                   (x0 + 0.55 * L + rng.uniform(-0.1, 0.1) * L, yt + (x0 + 0.55 * L - cx) * tt + rng.uniform(-0.004, 0.003)),
                   (x0 + c(), yt0),
                   (x0 - e() * 0.3, yt0 - c()),
                   (x0 + abs(e()) * 0.4, 0.5 * (yb0 + yt0) + e() * 0.3),
                   (x0 - e() * 0.3, yb0 + c() * 0.8)]
            v = np.array(pts, np.float64)
            n = np.array([rng.normal(0, 0.16), rng.normal(0.06, 0.12), 0.0])
            n[2] = math.sqrt(max(0.2, 1.0 - n[0] ** 2 - n[1] ** 2))
            tone = rng.uniform(0.75, 1.25)
            warm = rng.uniform(-0.08, 0.08)
            alb = np.array([0.105 * (1 + warm), 0.100, 0.090 * (1 - warm)]) * tone
            stones.append((v, n / np.linalg.norm(n), alb, float(rng.uniform(0, 100))))

        y = -0.012
        row = 0
        while y < self.top - cap_h - 0.035:
            chh = rng.uniform(0.050, 0.105)
            if y + chh > self.top - cap_h - 0.004:
                chh = self.top - cap_h - 0.004 - y
            hw = hw_at(y + 0.5 * chh)
            x = -hw - rng.uniform(-0.012, 0.028)
            x += rng.uniform(0.0, 0.10) if row % 2 else 0.0
            if x > -hw + 0.02:
                # a small end stone first so the course reaches the face
                stone(-hw - rng.uniform(0.0, 0.02), x - 0.006, y + 0.003, y + chh * rng.uniform(0.75, 0.95),
                      rng.normal(0, 2.0))
            first = True
            while x < hw - 0.03:
                L = rng.uniform(0.11, 0.30)
                if first and rng.random() < 0.3:
                    L = rng.uniform(0.28, 0.40)               # a long through-stone now and then
                first = False
                x1 = min(x + L, hw + rng.uniform(-0.01, 0.025))
                if hw - x1 < 0.05:
                    x1 = hw + rng.uniform(-0.005, 0.025)
                sh = chh * rng.uniform(0.78, 1.0)
                yb = y + rng.uniform(0.0, 0.006)
                stone(x, x1, yb, yb + sh, rng.normal(0, 2.2))
                # pinning stone in the gap under a thinner stone
                if sh < chh * 0.86 and rng.random() < 0.45 and x1 - x > 0.12:
                    px = x + rng.uniform(0.2, 0.8) * (x1 - x)
                    pw = rng.uniform(0.03, 0.06)
                    stone(px - pw / 2, px + pw / 2, yb + sh + 0.003, y + chh - 0.001, rng.normal(0, 6), jag=0.5)
                x = x1 + rng.uniform(0.005, 0.014)
            y += chh + rng.uniform(0.004, 0.009)
            row += 1
        # the capstone: one long flat slab, slightly overhanging, its top at self.top
        stone(-top_hw - 0.045, top_hw + 0.055, self.top - cap_h, self.top, 0.6, jag=0.6)
        self.stones = stones
        # the woodpile in the basket: split logs with flat-cut ends in a teepee, cross logs poking
        # through the bars, kindling at the top (polygons, not capsules)
        wr = np.random.default_rng(seed * 7 + 3)

        def log(a, b, r, bark=0.18):
            a = np.asarray(a, np.float64)
            b = np.asarray(b, np.float64)
            d = b - a
            L = np.linalg.norm(d)
            u = d / (L + 1e-12)
            nrm = np.array([-u[1], u[0]])
            m = 6
            side1 = []
            side2 = []
            for q in range(m + 1):
                s_ = q / m
                c = a + d * s_
                w1 = r * (1.0 + bark * wr.normal(0, 0.5))
                w2 = r * (0.75 + bark * wr.normal(0, 0.5))      # split face: flatter on one side
                side1.append(c + nrm * w1)
                side2.append(c - nrm * w2)
            cut0 = wr.uniform(-0.35, 0.35) * r
            cut1 = wr.uniform(-0.35, 0.35) * r
            side1[0] = side1[0] + u * cut0
            side2[0] = side2[0] - u * cut0
            side1[-1] = side1[-1] + u * cut1
            side2[-1] = side2[-1] - u * cut1
            return np.array(side1 + side2[::-1])

        wood = []
        yb, yt = self.bk_bot, self.bk_top
        for i in range(6):
            side = -1 if i % 2 == 0 else 1
            xb = side * wr.uniform(0.07, 0.21)
            xt = -side * wr.uniform(-0.01, 0.05)
            wood.append(log((xb, yb + 0.03), (xt, yt + wr.uniform(0.07, 0.16)), wr.uniform(0.022, 0.034)))
        for i in range(3):
            y_ = yb + 0.08 + i * 0.075 + wr.uniform(-0.015, 0.015)
            x0_ = -(self.bk_rb + (self.bk_rt - self.bk_rb) * (y_ - yb) / (yt - yb)) - wr.uniform(0.00, 0.06)
            x1_ = -x0_ + wr.uniform(-0.03, 0.05)
            wood.append(log((x0_, y_), (x1_, y_ + wr.uniform(-0.05, 0.05)), wr.uniform(0.024, 0.032)))
        for i in range(5):
            x_ = wr.uniform(-0.12, 0.12)
            ang = wr.uniform(-40, 40)
            Lk = wr.uniform(0.10, 0.20)
            a_ = np.array([x_, yt - wr.uniform(0.02, 0.10)])
            b_ = a_ + Lk * np.array([-math.sin(math.radians(ang)), math.cos(math.radians(ang))])
            wood.append(log(a_, b_, wr.uniform(0.006, 0.010), bark=0.3))
        self.wood = wood
        # hearting (the dark core behind the gaps), inset from the face outline
        k = 0.022
        self.core = np.array([(-base_hw + k, -0.01), (base_hw - k, -0.01), (top_hw - k, self.top - 0.02),
                              (-top_hw + k, self.top - 0.02)], np.float64)
        # packed polygons
        nv = [len(s[0]) for s in stones]
        self.pstart = np.cumsum([0] + nv[:-1]).astype(np.int64)
        self.pn = np.array(nv, np.int64)
        self.pverts = np.concatenate([s[0] for s in stones], 0)
        self.pnorm = np.array([s[1] for s in stones])
        self.palb = np.array([s[2] for s in stones])
        self.pseed = np.array([s[3] for s in stones])
        self.pbb = np.array([[s[0][:, 0].min(), s[0][:, 1].min(), s[0][:, 0].max(), s[0][:, 1].max()] for s in stones])

    # ---- same API as characters.Cairn for the basket + wood (the old puppet groups)
    def basket_groups(self, x=0.0, only_iron=False):
        """The iron basket + stacked wood as silhouette groups (rims only on the outer contour)."""
        g = self.old.groups(x=x)
        out = []
        if not only_iron:
            w = ch.G('wood', 'wood', k=0.004)
            for poly in self.wood:
                w.poly(poly + np.array([x, 0.0]), r=0.002, k=0.004)
            g = [w] + [gg for gg in g if gg.name == 'basket']
        for gg in g:
            if gg.name not in ('basket', 'wood') or (only_iron and gg.name != 'basket'):
                continue
            gg.sheen = 0.40 if gg.name == 'basket' else 0.32
            gg.soft = 0.03
            gg.sky_rim = 0.5
            gg.cast = 0.0
            gg.aa = 1.0
            out.append(gg)
        return out

    def fire_base(self, x=0.0):
        return np.array([x, self.bk_bot + 0.10])

    def render(self, cam, origin, lights, amb_top, bg=None, rim_k=2.2, rim_max=0.5):
        """Stones + hearting on the card at `origin` (base centre, world). lights (n,8) world
        [x,y,z,r,g,b,d0,_]. Returns (Y0, X0, rgb premult, alpha) or None."""
        o = np.asarray(origin, np.float64)
        x0l, y0l = self.pbb[:, 0].min() - 0.05, -0.05
        x1l, y1l = self.pbb[:, 2].max() + 0.05, self.top + 0.05
        corners = np.array([[o[0] + x0l, o[1] + y0l, o[2]], [o[0] + x1l, o[1] + y0l, o[2]],
                            [o[0] + x0l, o[1] + y1l, o[2]], [o[0] + x1l, o[1] + y1l, o[2]]])
        sx, sy, z = cam.project(corners)
        if np.any(z <= 0.05):
            return None
        X0 = max(0, int(np.floor(sx.min())) - 3)
        X1 = min(cam.W, int(np.ceil(sx.max())) + 3)
        Y0 = max(0, int(np.floor(sy.min())) - 3)
        Y1 = min(cam.H, int(np.ceil(sy.max())) + 3)
        if X1 <= X0 or Y1 <= Y0:
            return None
        h, w = Y1 - Y0, X1 - X0
        LX = np.empty((h, w))
        LY = np.empty((h, w))
        local_grid(LX, LY, cam.params(), o[0], o[1], o[2], Y0, X0)
        ppm = cam.f / max(1e-6, float(np.mean(z)))
        # pixel bboxes of the stones
        bb = self.pbb
        P = np.stack([np.stack([bb[:, 0] - 0.01 + o[0], bb[:, 1] - 0.01 + o[1], np.full(len(bb), o[2])], 1),
                      np.stack([bb[:, 2] + 0.01 + o[0], bb[:, 3] + 0.01 + o[1], np.full(len(bb), o[2])], 1)], 1)
        psx, psy, _ = cam.project(P.reshape(-1, 3))
        psx = psx.reshape(-1, 2)
        psy = psy.reshape(-1, 2)
        ranges = np.zeros((len(bb), 4), np.int64)
        ranges[:, 0] = np.clip(np.floor(psy.min(1)) - Y0 - 2, 0, h)
        ranges[:, 1] = np.clip(np.ceil(psy.max(1)) - Y0 + 3, 0, h)
        ranges[:, 2] = np.clip(np.floor(psx.min(1)) - X0 - 2, 0, w)
        ranges[:, 3] = np.clip(np.ceil(psx.max(1)) - X0 + 3, 0, w)
        SD = np.full((h, w), 10.0)
        SID = np.full((h, w), -1, np.int64)
        raster_stones(SD, SID, LX, LY, self.pverts, self.pstart, self.pn, ranges)
        DC = np.full((h, w), 10.0)
        core_field(DC, LX, LY, self.core)
        U = np.minimum(SD, DC)
        DEP = exact_depth(U, ppm)
        L = np.asarray(lights, np.float64).reshape(-1, 8).copy()
        L[:, 0] -= o[0]
        L[:, 1] -= o[1]
        L[:, 2] = o[2] - L[:, 2]
        if bg is not None:
            sub = np.ascontiguousarray(bg[Y0:Y1, X0:X1, :3], np.float32)
            sub = cv2.GaussianBlur(sub, (0, 0), max(1.0, 5.0 * cam.scale))
            bk = np.minimum(sub.astype(np.float64) * rim_k, rim_max)
        else:
            bk = np.zeros((h, w, 3))
        rgb = np.zeros((h, w, 3))
        a = np.zeros((h, w))
        shade_stones(SD, SID, DC, U, DEP, LX, LY, ppm, self.pnorm, self.palb, self.pseed, L, len(L),
                     np.asarray(amb_top, np.float64), bk, rgb, a)
        return Y0, X0, rgb.astype(np.float32), a.astype(np.float32)


@nb.njit(cache=True, fastmath=True)
def raster_stones(SD, SID, LX, LY, verts, pstart, pn, ranges):
    for p in range(pstart.shape[0]):
        for j in range(ranges[p, 0], ranges[p, 1]):
            for i in range(ranges[p, 2], ranges[p, 3]):
                d = _sd_poly(LX[j, i], LY[j, i], verts, pstart[p], pn[p])
                if d < SD[j, i]:
                    SD[j, i] = d
                    SID[j, i] = p


@nb.njit(cache=True, fastmath=True)
def core_field(DC, LX, LY, core):
    h = DC.shape[0]
    w = DC.shape[1]
    for j in range(h):
        for i in range(w):
            DC[j, i] = _sd_poly(LX[j, i], LY[j, i], core, 0, core.shape[0])


@nb.njit(cache=True, fastmath=True)
def shade_stones(SD, SID, DC, U, DEP, LX, LY, ppm, pnorm, palb, pseed, lights, nl, amb_top, back,
                 out_rgb, out_a):
    h = SD.shape[0]
    w = SD.shape[1]
    bev = max(0.0045, 1.4 / ppm)            # the hard chamfer on every arris
    rw = max(0.0016, 1.1 / ppm)
    for j in range(h):
        for i in range(w):
            u = U[j, i]
            cov = 0.5 - u * ppm
            if cov <= 0.0:
                continue
            if cov > 1.0:
                cov = 1.0
            x = LX[j, i]
            y = LY[j, i]
            il = i - 1 if i > 0 else i
            ir = i + 1 if i < w - 1 else i
            ju = j - 1 if j > 0 else j
            jd = j + 1 if j < h - 1 else j
            ddx = LX[j, ir] - LX[j, il]
            ddy = LY[ju, i] - LY[jd, i]
            # hearting: near-black, faint ambient, darker deeper in the gap
            d = SD[j, i]
            cr = 0.0018 * amb_top[0] * 4.0
            cg = 0.0018 * amb_top[1] * 4.0
            cb = 0.0018 * amb_top[2] * 4.0
            sc = 0.5 - d * ppm
            if sc > 0.0:
                if sc > 1.0:
                    sc = 1.0
                p = SID[j, i]
                # this stone's own edge direction (only if the neighbours are the same stone)
                gx = 0.0
                gy = 0.0
                if SID[j, ir] == p and SID[j, il] == p and abs(ddx) > 1e-9:
                    gx = (SD[j, ir] - SD[j, il]) / ddx
                if SID[ju, i] == p and SID[jd, i] == p and abs(ddy) > 1e-9:
                    gy = (SD[ju, i] - SD[jd, i]) / ddy
                gn = math.sqrt(gx * gx + gy * gy)
                if gn > 1e-9:
                    gx /= gn
                    gy /= gn
                depth = -d
                if depth < 0.0:
                    depth = 0.0
                nx = pnorm[p, 0]
                ny = pnorm[p, 1]
                nz = pnorm[p, 2]
                # hard chamfer: inside `bev` of the arris the surface turns ~55 deg toward the edge
                ch_ = 1.0 - depth / bev
                if ch_ > 0.0:
                    k = 1.4 * ch_ ** 0.5
                    nx += gx * k
                    ny += gy * k
                    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
                    nx /= nn
                    ny /= nn
                    nz /= nn
                # weathered stone: grain + lichen blotches + darker toward the chipped edges
                sd_ = pseed[p]
                g1 = _fbm2n(x * 38.0 + sd_, y * 38.0, sd_ * 0.31, 3)
                g2 = _fbm2n(x * 140.0, y * 140.0 + sd_, 3.3, 2)
                li = _fbm2n(x * 14.0 - sd_, y * 14.0, 7.7, 2)
                tex = 0.82 + 0.35 * g1 + 0.14 * g2
                if li > 0.28:
                    tex *= 0.72
                if tex < 0.3:
                    tex = 0.3
                ar = palb[p, 0] * tex
                ag = palb[p, 1] * tex
                ab = palb[p, 2] * tex
                # twilight sky on grey stone (faces + the upward arrises catch it): the courses stay
                # faintly legible even when the fire is far
                up = 0.5 + 0.5 * ny
                r = ar * amb_top[0] * (1.2 + 3.4 * up)
                g = ag * amb_top[1] * (1.2 + 3.4 * up)
                b = ab * amb_top[2] * (1.2 + 3.4 * up)
                for q in range(nl):
                    vx = lights[q, 0] - x
                    vy = lights[q, 1] - y
                    vz = lights[q, 2]
                    dist = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-6
                    lam = (nx * vx + ny * vy + nz * vz) / dist
                    if lam <= 0.0:
                        continue
                    att = 1.0 / (1.0 + (dist / lights[q, 6]) ** 2)
                    r += ar * lights[q, 3] * lam * att
                    g += ag * lights[q, 4] * lam * att
                    b += ab * lights[q, 5] * lam * att
                # contact shadow just below the stone above (the bed joint) and in narrow gaps
                occ = 1.0
                if gy < -0.5 and depth < 2.5 * bev:
                    occ = 0.55 + 0.45 * depth / (2.5 * bev)
                r *= occ
                g *= occ
                b *= occ
                cr = cr * (1.0 - sc) + r * sc
                cg = cg * (1.0 - sc) + g * sc
                cb = cb * (1.0 - sc) + b * sc
            # sky rim on the cairn's outer outline only
            dep = DEP[j, i]
            rs = math.exp(-dep / rw)
            if rs > 0.01:
                ugx = 0.0
                ugy = 0.0
                if abs(ddx) > 1e-9:
                    ugx = (min(U[j, ir], 0.5) - min(U[j, il], 0.5)) / ddx
                if abs(ddy) > 1e-9:
                    ugy = (min(U[ju, i], 0.5) - min(U[jd, i], 0.5)) / ddy
                un = math.sqrt(ugx * ugx + ugy * ugy)
                if un > 1e-9:
                    ugy /= un
                sr = 0.45 * rs * (0.4 + 0.6 * max(0.0, ugy))
                cr += back[j, i, 0] * sr
                cg += back[j, i, 1] * sr
                cb += back[j, i, 2] * sr
            out_rgb[j, i, 0] = cr * cov
            out_rgb[j, i, 1] = cg * cov
            out_rgb[j, i, 2] = cb * cov
            out_a[j, i] = cov
