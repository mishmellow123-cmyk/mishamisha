"""v2: eight towers MADE OF EMBERS (one per kingdom), each architecturally distinct.

v1 drew the towers as points along box edges (they read as CAD wireframes). v2 builds each one as a solid
surface of embers with architectural structure:

    CRUST  the facade itself: a dense skin of dim ember-light (area-sampled; it also feeds the occluder,
           so the towers are opaque masses that hide what is behind them)
    SEAM   floor lines and bay joints: fine seams of fire in the crust
    EDGE   the architectural edges (corners, tier and eave edges, ribs): burning lines
    WIN    windows of fire: dense glowing cells (only lit ones get points; unlit ones are dark openings,
           marked on the crust points that lie inside them)

Every tower is built in local coordinates with its TOP at y = HMAX; the body continues down to YBOT (< 0)
so a tower that has surged higher than HMAX still stands on the ground. Local +x faces the centre.
Per point:  p (N,3) local position, n (N,3) outward normal, a area (world units^2; a line's length x width),
            kind, key (window id for WIN / crust inside a window, else -1), hot (intrinsic heat 0..1).
"""
import math

import numpy as np
from numba import njit, prange

HMAX = 66.0
YBOT = -26.0

CRUST, SEAM, EDGE, WIN, CRACK = 0, 1, 2, 3, 4


@njit(fastmath=True, cache=True, inline='always')
def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = h & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x7FEB352D) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x846CA68B) & 0xFFFFFFFF
    h ^= h >> 16
    return h


@njit(parallel=True, fastmath=True, cache=True)
def _worley(P, seed, F1, F2, CID):
    """3D cellular noise (cell units): nearest / second-nearest feature distance and the nearest cell's id"""
    for i in prange(P.shape[0]):
        x = P[i, 0]
        y = P[i, 1]
        z = P[i, 2]
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        iz = int(math.floor(z))
        f1 = 1e9
        f2 = 1e9
        c1 = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cx = ix + dx
                    cy = iy + dy
                    cz = iz + dz
                    h1 = _hash3(cx, cy, cz, seed)
                    h2 = _hash3(cx, cy, cz, seed + 17)
                    h3 = _hash3(cx, cy, cz, seed + 31)
                    fx = cx + (h1 & 1023) / 1024.0
                    fy = cy + (h2 & 1023) / 1024.0
                    fz = cz + (h3 & 1023) / 1024.0
                    d2 = (x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2
                    if d2 < f1:
                        f2 = f1
                        f1 = d2
                        c1 = h1
                    elif d2 < f2:
                        f2 = d2
        F1[i] = math.sqrt(f1)
        F2[i] = math.sqrt(f2)
        CID[i] = c1 & 0xFFFF


def joint_field(u, v, style, rng):
    """fire in the joints of the facade, in facade coordinates (u across, v up; world units).
    style: ('masonry', course height, block length) -> staggered courses of blocks (the ancient towers)
           ('grid', floor height, bay width)        -> the mullion grid of a curtain wall (the modern ones)
    returns (intensity 0..1, block id)"""
    kind, ch, bl = style
    w = 0.05
    k = np.floor(v / ch)
    if kind == 'masonry':
        # every course shifts by half a block, plus a little irregularity per course
        jit = ((k * 0.618034) % 1.0) * 0.3 * bl
        uu = u + (k % 2) * 0.5 * bl + jit
    else:
        uu = u
    fv = v - k * ch
    dv = np.minimum(fv, ch - fv)
    j = np.floor(uu / bl)
    fu = uu - j * bl
    du = np.minimum(fu, bl - fu)
    jv = np.clip(1.0 - dv / w, 0, 1) ** 1.5
    ju = np.clip(1.0 - du / (w * (0.85 if kind == 'masonry' else 1.0)), 0, 1) ** 1.5
    inten = np.maximum(jv, ju * (0.55 if kind == 'masonry' else 0.75))
    blk = ((k * 7919 + j * 104729) % 1009) / 1009.0
    return inten, blk


def crack_field(p, cell=(1.6, 0.8, 1.6), seed=11, width=0.07):
    """intensity (0..1) of a network of cracks on the crust: cells stretched along the courses (stone, not
    skin), a finer second network inside the blocks, and a per-block heat (so some seams roar)"""
    n = len(p)
    q = np.ascontiguousarray(p / np.asarray(cell, np.float64))
    F1, F2, cid = np.empty(n), np.empty(n), np.empty(n, np.int64)
    _worley(q, seed, F1, F2, cid)
    c1 = 1.0 - np.clip((F2 - F1) / width, 0, 1) ** 0.8
    q2 = np.ascontiguousarray(p / (np.asarray(cell, np.float64) * 0.42))
    G1, G2, cid2 = np.empty(n), np.empty(n), np.empty(n, np.int64)
    _worley(q2, seed + 7, G1, G2, cid2)
    c2 = 1.0 - np.clip((G2 - G1) / (width * 1.1), 0, 1) ** 0.8
    blk = (cid % 997) / 997.0
    return np.clip(np.maximum(c1 * (0.45 + 0.55 * blk), 0.45 * c2 * blk), 0, 1)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


class TB:
    def __init__(self, seed, dc=80.0, dw=900.0, dl=40.0, win_p=0.36, kc=5.0, cell=(2.6, 1.8, 2.6),
                 style=('masonry', 0.55, 1.2)):
        self.r = np.random.default_rng(seed)
        self.dc, self.dw, self.dl = dc, dw, dl
        self.kc, self.cell, self.seed = kc, cell, seed
        self.style = style
        self.win_p = win_p
        self.P, self.N, self.A, self.K, self.KEY, self.HOT = [], [], [], [], [], []
        self.nwin = 0

    # ---------------------------------------------------------------- basics
    def add(self, kind, p, n, a, key=-1, hot=0.5):
        p = np.asarray(p, np.float64).reshape(-1, 3)
        m = len(p)
        if m == 0:
            return
        self.P.append(p)
        self.N.append(np.broadcast_to(np.asarray(n, np.float64), (m, 3)).copy())
        self.A.append(np.broadcast_to(np.asarray(a, np.float64), (m,)).copy())
        self.K.append(np.full(m, kind, np.int8))
        self.KEY.append(np.broadcast_to(np.asarray(key, np.int32), (m,)).copy())
        self.HOT.append(np.broadcast_to(np.asarray(hot, np.float64), (m,)).copy())

    def line(self, a, b, kind=EDGE, n=None, width=0.07, hot=0.6, dens=None, jit=0.012):
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        d = (dens if dens is not None else self.dl) * (1.4 if kind == EDGE else 1.0)
        m = max(2, int(L * d))
        t = self.r.random(m)
        p = a + (b - a) * t[:, None] + self.r.normal(0, jit, (m, 3))
        if n is None:
            n = np.zeros((m, 3))
            n[:, 0], n[:, 2] = p[:, 0], p[:, 2]
            n = _unit(n + np.array([1e-6, 0, 0]))
        self.add(kind, p, n, L * width / m, -1, hot)

    def polyline(self, pts, kind=EDGE, nrm=None, width=0.07, hot=0.6, closed=False):
        pts = np.asarray(pts, float)
        k = len(pts) if closed else len(pts) - 1
        for i in range(k):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            nn = None if nrm is None else _unit(np.asarray(nrm, float))
            self.line(a, b, kind, nn, width, hot)

    def new_windows(self, m):
        ids = np.arange(self.nwin, self.nwin + m)
        self.nwin += m
        return ids

    def win_points(self, corners_fn, m_each, key, n):
        pass

    # --------------------------------------------------------------- prisms
    def prism(self, apo, y0, y1, nsides=4, rot=0.0, twist=None, win=None, edges=True, floors=False,
              edge_hot=0.7, dc=None, joints=2, cracks=True, style=None):
        """Side faces of a (tapering, optionally twisted) prism with a regular nsides-gon section.
        apo(y): apothem at height y (vectorised).  win: dict(fh, bw, ww, wh, off, p) window grid."""
        r = self.r
        dc = self.dc if dc is None else dc
        tn = np.tan(np.pi / nsides)
        ys = np.linspace(y0, y1, 256)
        area = np.trapezoid(nsides * 2 * apo(ys) * tn, ys)
        m = int(area * dc * (1.0 + (self.kc if cracks else 0.0)))
        if m < 10:
            return
        y = r.uniform(y0, y1, m)
        a_ = apo(y)
        keep = r.random(m) < a_ / apo(ys).max()
        y = y[keep]
        a_ = a_[keep]
        m = len(y)
        j = r.integers(0, nsides, m)
        s = r.uniform(-1, 1, m)
        tw = twist(y) if twist is not None else np.zeros_like(y)
        phi = rot + 2 * np.pi * j / nsides + tw
        c, sn = np.cos(phi), np.sin(phi)
        half = a_ * tn
        x = a_ * c - s * half * sn
        z = a_ * sn + s * half * c
        dady = np.gradient(apo(ys), ys)
        slope = np.interp(y, ys, dady)
        n = _unit(np.stack([c, -slope, sn], 1))
        p = np.stack([x, y, z], 1)
        key = np.full(m, -1, np.int32)
        wid0 = None
        if win is not None:
            fh, bw, ww, wh, off, pl = (win['fh'], win['bw'], win['ww'], win['wh'], win.get('off', 0.0),
                                       win.get('p', self.win_p))
            # window grid on each face: floor k, bay b
            kf = np.floor((y - y0 - off) / fh)
            yc = y0 + off + (kf + 0.5) * fh
            width = 2 * apo(yc) * tn
            nb = np.maximum(np.floor(width / bw), 1)
            bb = np.floor((s + 1) / 2 * nb)
            sc = -1 + (2 * bb + 1) / nb
            inx = np.abs(s - sc) * apo(yc) * tn < ww / 2
            iny = np.abs(y - yc) < wh / 2
            nf = int(np.ceil((y1 - y0) / fh)) + 1
            nbmax = int(np.floor(2 * apo(ys).max() * tn / bw)) + 1
            wid = ((j * nf + kf.astype(int)) * nbmax + bb.astype(int))
            wid0 = self.nwin
            inside = inx & iny & (kf >= 0)
            key = np.where(inside, wid0 + wid, -1).astype(np.int32)
            self.nwin += nsides * nf * nbmax
            # lit windows: dense cells
            lit = r.random(nsides * nf * nbmax) < pl
            for jj in range(nsides):
                for k in range(nf):
                    ycw = y0 + off + (k + 0.5) * fh
                    if ycw - wh / 2 < y0 or ycw + wh / 2 > y1:
                        continue
                    aw = float(apo(np.array([ycw]))[0])
                    wdt = 2 * aw * tn
                    nbk = max(int(np.floor(wdt / bw)), 1)
                    for b in range(nbk):
                        w_id = (jj * nf + k) * nbmax + b
                        if not lit[w_id]:
                            continue
                        scw = -1 + (2 * b + 1) / nbk
                        mw = max(6, int(ww * wh * self.dw))
                        ss = scw + r.uniform(-1, 1, mw) * (ww / 2) / (aw * tn)
                        yy = ycw + r.uniform(-1, 1, mw) * wh / 2
                        aa = apo(yy)
                        tww = twist(yy) if twist is not None else np.zeros_like(yy)
                        ph = rot + 2 * np.pi * jj / nsides + tww
                        cc, snn = np.cos(ph), np.sin(ph)
                        xw = (aa + 0.01) * cc - ss * aa * tn * snn
                        zw = (aa + 0.01) * snn + ss * aa * tn * cc
                        nw = _unit(np.stack([cc, -np.interp(yy, ys, dady), snn], 1))
                        self.add(WIN, np.stack([xw, yy, zw], 1), nw, ww * wh / mw, wid0 + w_id, 0.5)
        uf = (j * 2.0 + (s + 1.0)) * half
        self.crust_and_cracks(p, n, key, dc, 0.3, cracks, (uf, y), style)
        # seams: floor lines + bay joints (on the face), corner edges
        if win is not None and floors:
            fh, bw, off = win['fh'], win['bw'], win.get('off', 0.0)
            for k in range(int(np.ceil((y1 - y0) / fh)) + 1):
                yf = y0 + off + k * fh
                if yf < y0 - 1e-6 or yf > y1 + 1e-6:
                    continue
                aw = float(apo(np.array([yf]))[0])
                tww = float(twist(np.array([yf]))[0]) if twist is not None else 0.0
                for jj in range(nsides):
                    ph = rot + 2 * np.pi * jj / nsides + tww
                    cc, snn = np.cos(ph), np.sin(ph)
                    ctr = np.array([aw * cc, yf, aw * snn])
                    tvec = np.array([-snn, 0, cc]) * aw * tn
                    self.line(ctr - tvec, ctr + tvec, SEAM, np.array([cc, 0, snn]), 0.05, 0.55)
                    # bay joints up to the next floor
                    wdt = 2 * aw * tn
                    nbk = max(int(np.floor(wdt / bw)), 1)
                    yt = min(yf + fh, y1)
                    if yt - yf < 0.2:
                        continue
                    for b in range(1, nbk):
                        if joints <= 0 or b % joints:
                            continue
                        sb = -1 + 2 * b / nbk
                        pa = ctr + tvec * sb
                        at = float(apo(np.array([yt]))[0])
                        twt = float(twist(np.array([yt]))[0]) if twist is not None else 0.0
                        pht = rot + 2 * np.pi * jj / nsides + twt
                        pb = np.array([at * np.cos(pht), yt, at * np.sin(pht)]) + \
                            np.array([-np.sin(pht), 0, np.cos(pht)]) * at * tn * sb
                        self.line(pa, pb, SEAM, np.array([cc, 0, snn]), 0.035, 0.3)
        if edges:
            yy = np.linspace(y0, y1, max(8, int((y1 - y0) * 3)))
            aa = apo(yy)
            tww = twist(yy) if twist is not None else np.zeros_like(yy)
            for jj in range(nsides):
                ph = rot + 2 * np.pi * (jj + 0.5) / nsides + tww      # corner direction
                rr = aa / np.cos(np.pi / nsides)
                pts = np.stack([rr * np.cos(ph), yy, rr * np.sin(ph)], 1)
                for i in range(len(pts) - 1):
                    nn = np.array([np.cos(ph[i]), 0.0, np.sin(ph[i])])
                    self.line(pts[i], pts[i + 1], EDGE, nn, 0.08, edge_hot)

    def ngon_ring(self, apo, y, nsides=4, rot=0.0, twist=0.0, kind=EDGE, width=0.08, hot=0.7, nrm=None):
        """closed horizontal edge around an n-gon of apothem apo at height y"""
        rr = apo / np.cos(np.pi / nsides)
        pts = [np.array([rr * np.cos(rot + twist + 2 * np.pi * (j + 0.5) / nsides), y,
                         rr * np.sin(rot + twist + 2 * np.pi * (j + 0.5) / nsides)]) for j in range(nsides)]
        for j in range(nsides):
            a, b = pts[j], pts[(j + 1) % nsides]
            mid = 0.5 * (a + b)
            nn = _unit(np.array([mid[0], 0.0, mid[2]])) if nrm is None else nrm
            self.line(a, b, kind, nn, width, hot)

    def ngon_slab(self, apo_out, apo_in, y, nsides=4, rot=0.0, up=1.0, dc=None):
        """horizontal surface between two concentric n-gons (a terrace / roof / floor plate)"""
        dc = self.dc if dc is None else dc
        tn = np.tan(np.pi / nsides)
        area = nsides * (apo_out ** 2 - apo_in ** 2) * tn
        m = int(area * dc)
        if m < 4:
            return
        r = self.r
        # sample in the circumscribed square then keep inside the ring
        R = apo_out / np.cos(np.pi / nsides)
        q = r.uniform(-R, R, (m * 3, 2))
        ang = np.arctan2(q[:, 1], q[:, 0]) - rot
        sec = np.mod(ang, 2 * np.pi / nsides) - np.pi / nsides
        apd = np.hypot(q[:, 0], q[:, 1]) * np.cos(sec)          # distance along the sector's apothem
        ok = (apd < apo_out) & (apd > apo_in)
        q = q[ok][:m]
        p = np.stack([q[:, 0], np.full(len(q), y), q[:, 1]], 1)
        self.add(CRUST, p, np.array([0.0, up, 0.0]), 1.0 / dc, -1, 0.25)

    # ------------------------------------------------------------ revolution
    def revolve(self, rf, y0, y1, win=None, nbay=16, seams=False, dc=None, hot=0.3, meridians=0, cracks=True,
                style=None):
        r = self.r
        dc = self.dc if dc is None else dc
        ys = np.linspace(y0, y1, 400)
        rs = np.maximum(rf(ys), 0.0)
        drs = np.gradient(rs, ys)
        dl = np.sqrt(1 + drs ** 2)
        area = np.trapezoid(2 * np.pi * rs * dl, ys)
        m = int(area * dc * (1.0 + (self.kc if cracks else 0.0)))
        if m < 10:
            return
        w = rs * dl
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        y = np.interp(r.random(m), cdf, ys)
        rr = np.maximum(rf(y), 0.0)
        dr = np.interp(y, ys, drs)
        th = r.uniform(0, 2 * np.pi, m)
        c, s = np.cos(th), np.sin(th)
        p = np.stack([rr * c, y, rr * s], 1)
        n = _unit(np.stack([c, -dr, s], 1))
        key = np.full(m, -1, np.int32)
        if win is not None:
            fh, ww, wh, off, pl = win['fh'], win['ww'], win['wh'], win.get('off', 0.0), win.get('p', self.win_p)
            kf = np.floor((y - y0 - off) / fh)
            yc = y0 + off + (kf + 0.5) * fh
            bb = np.floor(th / (2 * np.pi) * nbay)
            thc = (bb + 0.5) * 2 * np.pi / nbay
            rc = np.maximum(rf(yc), 1e-3)
            inside = (np.abs(th - thc) * rc < ww / 2) & (np.abs(y - yc) < wh / 2) & (kf >= 0)
            nf = int(np.ceil((y1 - y0) / fh)) + 1
            wid0 = self.nwin
            self.nwin += nf * nbay
            key = np.where(inside, wid0 + kf.astype(int) * nbay + bb.astype(int), -1).astype(np.int32)
            lit = r.random(nf * nbay) < pl
            for k in range(nf):
                ycw = y0 + off + (k + 0.5) * fh
                if ycw - wh / 2 < y0 or ycw + wh / 2 > y1:
                    continue
                rcw = float(rf(np.array([ycw]))[0])
                if rcw < 0.3:
                    continue
                for b in range(nbay):
                    if not lit[k * nbay + b]:
                        continue
                    mw = max(6, int(ww * wh * self.dw))
                    tt = (b + 0.5) * 2 * np.pi / nbay + r.uniform(-1, 1, mw) * (ww / 2) / rcw
                    yy = ycw + r.uniform(-1, 1, mw) * wh / 2
                    r2 = np.maximum(rf(yy), 0.0) + 0.01
                    d2 = np.interp(yy, ys, drs)
                    pw = np.stack([r2 * np.cos(tt), yy, r2 * np.sin(tt)], 1)
                    nw = _unit(np.stack([np.cos(tt), -d2, np.sin(tt)], 1))
                    self.add(WIN, pw, nw, ww * wh / mw, wid0 + k * nbay + b, 0.5)
            if seams:
                for k in range(nf + 1):
                    yf = y0 + off + k * fh
                    if yf < y0 or yf > y1:
                        continue
                    self.circle(float(rf(np.array([yf]))[0]) + 0.005, yf, SEAM, 0.05, 0.5)
        self.crust_and_cracks(p, n, key, dc, hot, cracks, (th * np.maximum(rr, 0.3), y), style)
        if meridians:
            for b in range(meridians):
                a = 2 * np.pi * b / meridians
                yy = np.linspace(y0, y1, 60)
                r2 = np.maximum(rf(yy), 0.0) + 0.005
                pts = np.stack([r2 * np.cos(a), yy, r2 * np.sin(a)], 1)
                for i in range(len(pts) - 1):
                    self.line(pts[i], pts[i + 1], SEAM, None, 0.04, 0.4)

    def circle(self, rad, y, kind=EDGE, width=0.07, hot=0.6, nrm_up=0.0):
        if rad < 0.02:
            return
        m = max(8, int(2 * np.pi * rad * self.dl * (1.4 if kind == EDGE else 1.0)))
        a = self.r.uniform(0, 2 * np.pi, m)
        p = np.stack([rad * np.cos(a), np.full(m, y), rad * np.sin(a)], 1) + self.r.normal(0, 0.01, (m, 3))
        n = _unit(np.stack([np.cos(a), np.full(m, nrm_up), np.sin(a)], 1))
        self.add(kind, p, n, 2 * np.pi * rad * width / m, -1, hot)

    def annulus(self, r0, r1, y, up=1.0, dc=None, hot=0.25):
        dc = self.dc if dc is None else dc
        m = int(np.pi * (r1 * r1 - r0 * r0) * dc)
        if m < 4:
            return
        rr = np.sqrt(self.r.uniform(r0 * r0, r1 * r1, m))
        a = self.r.uniform(0, 2 * np.pi, m)
        p = np.stack([rr * np.cos(a), np.full(m, y), rr * np.sin(a)], 1)
        self.add(CRUST, p, np.array([0.0, up, 0.0]), 1.0 / dc, -1, hot)

    def beam(self, a, b, rad, hot=0.4, dc=None, edge=True):
        """solid cylindrical beam (truss member): ember crust around it + a burning core line"""
        dc = self.dc if dc is None else dc
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        d /= L
        ref = np.array([0, 1.0, 0]) if abs(d[1]) < 0.9 else np.array([1.0, 0, 0])
        e1 = _unit(np.cross(d, ref))
        e2 = np.cross(d, e1)
        m = max(8, int(2 * np.pi * rad * L * dc * 1.6))
        t = self.r.random(m)
        th = self.r.uniform(0, 2 * np.pi, m)
        nrm = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
        p = a + d * (t * L)[:, None] + nrm * rad
        self.add(CRUST, p, nrm, 2 * np.pi * rad * L / m, -1, hot)
        if edge:
            self.line(a, b, EDGE, None, rad * 0.9, hot)

    def crust_and_cracks(self, p, n, key, dc, hot=0.3, cracks=True, uv=None, style=None):
        """the first 1/(1+kc) of the samples become crust; the rest are candidates for the fire in the joints
        (masonry / mullion grid in facade coordinates uv, plus sparse large fissures), kept with probability =
        intensity: importance sampling, so the points crowd onto the lines and draw them continuously.
        CRACK 'hot' holds the per-block heat (some joints roar, some smoulder)."""
        m = len(p)
        mc = int(round(m / (1.0 + self.kc))) if cracks else m
        self.add(CRUST, p[:mc], n[:mc], 1.0 / dc, key[:mc], hot)
        if not cracks or mc >= m:
            return
        pc, nc, kk = p[mc:], n[mc:], key[mc:]
        style = style or self.style
        fis = crack_field(pc, self.cell, 11 + self.seed, 0.05) * 0.5
        if uv is not None and style is not None:
            ji, blk = joint_field(uv[0][mc:], uv[1][mc:], style, self.r)
        else:
            ji, blk = np.zeros(len(pc)), np.zeros(len(pc))
        inten = np.maximum(ji, fis)
        keep = (self.r.random(len(pc)) < inten) & (kk < 0)
        # 'hot' = how close to the joint's centre line (the core of the seam burns hottest)
        self.add(CRACK, pc[keep], nc[keep], 1.0 / (dc * self.kc), -1, inten[keep])

    def done(self):
        out = dict(p=np.concatenate(self.P), n=np.concatenate(self.N), a=np.concatenate(self.A),
                   kind=np.concatenate(self.K), key=np.concatenate(self.KEY), hot=np.concatenate(self.HOT))
        out['nwin'] = self.nwin
        # shuffle once so any prefix of each kind is a uniform subsample (level of detail)
        idx = self.r.permutation(len(out['p']))
        for k in ('p', 'n', 'a', 'kind', 'key', 'hot'):
            out[k] = out[k][idx]
        return out


# =================================================================== designs ===

def needle_spire(seed=0):
    """Octagonal gothic needle: tapering shaft with burning ribs, slit windows, a cone crown and a needle."""
    b = TB(seed, dc=95.0, style=('masonry', 0.48, 0.95))
    top = HMAX
    body = top - 12.0

    def apo(y):
        return 1.62 - 0.46 * np.clip(y / body, 0, 1)
    b.prism(apo, YBOT, body, 8, rot=np.pi / 8, win=dict(fh=0.95, bw=0.55, ww=0.18, wh=0.5, off=0.3, p=0.5),
            edge_hot=0.8)
    a1 = float(apo(np.array([body]))[0])
    b.ngon_ring(a1 + 0.08, body, 8, rot=np.pi / 8, hot=0.9)
    b.ngon_slab(a1 + 0.08, 0.0, body, 8, rot=np.pi / 8)
    # crown: an octagonal cone (faces + ribs) and a needle
    b.prism(lambda y: np.interp(y, [body, body + 5.0], [a1 * 0.92, 0.2]), body, body + 5.0, 8, rot=np.pi / 8,
            edge_hot=0.9)
    b.line([0, body + 4.6, 0], [0, top, 0], EDGE, None, 0.06, 1.0, dens=70)
    for k in range(4):
        a = 2 * np.pi * k / 4 + 0.4
        b.line([a1 * 1.1 * np.cos(a), body, a1 * 1.1 * np.sin(a)],
               [0.1 * np.cos(a), body + 7.0, 0.1 * np.sin(a)], EDGE, None, 0.05, 0.8)
    return b.done()


def ziggurat(seed=1):
    """Stepped temple-mountain: tiers of dark masonry, a colonnade of fire in every tier, burning terraces."""
    b = TB(seed, dc=110.0, win_p=0.3, style=('masonry', 0.55, 1.3))
    top = HMAX
    th = 2.2
    shrine = 1.8
    y = top - shrine
    b.prism(lambda yy: np.full_like(yy, 0.8), y, top, 4, rot=0.0,
            win=dict(fh=shrine, bw=0.8, ww=0.36, wh=1.0, off=0.0, p=1.0), floors=False, edge_hot=0.9)
    b.ngon_ring(0.8, top, 4, hot=0.9)
    b.ngon_slab(0.8, 0.0, top, 4)
    k = 0
    prev = 0.8
    while y > YBOT:
        hw = 1.3 + 0.75 * min(k, 7)
        y0 = y - th
        b.prism(lambda yy, hw=hw: np.full_like(yy, hw), y0, y, 4, rot=0.0,
                win=dict(fh=th / 2, bw=0.62, ww=0.2, wh=0.52, off=0.0), floors=False, edge_hot=0.35)
        # the tier's crown edge (terrace lip) and base course
        b.ngon_ring(hw + 0.02, y, 4, hot=0.6)
        b.ngon_ring(hw + 0.02, y0 + 0.28, 4, kind=SEAM, width=0.05, hot=0.45)
        if hw > prev + 0.05:
            b.ngon_slab(hw, prev, y, 4)            # terrace
        # the central stair on the inner (+x) face
        for ys in np.arange(y0 + 0.15, y, 0.28):
            b.line([hw + 0.02, ys, -0.55], [hw + 0.02, ys, 0.05], SEAM, np.array([1.0, 0, 0]), 0.04, 0.6,
                   dens=60)
        prev = hw
        y = y0
        k += 1
    return b.done()


def lattice_mast(seed=2):
    """Triangular truss mast: burning beams, platforms, a cabin of fire near the top, the antenna."""
    b = TB(seed, dc=90.0, style=None, kc=0.0)
    top = HMAX
    mast = top - 5.0

    def hw(y):
        return 0.45 + 2.3 * (1 - np.clip(y / mast, 0, 1)) ** 1.3 + 0.02 * np.maximum(-y, 0)
    legs = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    ys = np.arange(YBOT, mast + 0.01, 2.2)

    def leg(a, y):
        r = float(hw(np.array([y]))[0])
        return np.array([r * np.cos(a), y, r * np.sin(a)])
    for a in legs:
        for i in range(len(ys) - 1):
            b.beam(leg(a, ys[i]), leg(a, ys[i + 1]), 0.13, hot=0.55)
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        for j in range(3):
            a0, a1 = legs[j], legs[(j + 1) % 3]
            b.beam(leg(a0, y0), leg(a1, y1), 0.06, hot=0.35, edge=True)
            b.beam(leg(a1, y0), leg(a0, y1), 0.06, hot=0.35, edge=True)
            b.beam(leg(a0, y1), leg(a1, y1), 0.07, hot=0.45)
    # platforms (triangular plates with a burning rim) and a cabin
    for yp in (14.0, 30.0, 44.0):
        r = float(hw(np.array([yp]))[0]) + 0.5
        b.ngon_slab(r * 0.5, 0.0, yp, 3, rot=0.0)
        b.ngon_ring(r * 0.5, yp, 3, rot=0.0, hot=0.9)
    cab_y = mast - 4.0
    b.revolve(lambda y: np.full_like(y, 1.05), cab_y - 1.6, cab_y + 1.6,
              win=dict(fh=1.05, ww=0.34, wh=0.62, off=0.0, p=0.85), nbay=10, dc=140)
    b.circle(1.08, cab_y + 1.6, EDGE, 0.08, 0.9)
    b.circle(1.08, cab_y - 1.6, EDGE, 0.08, 0.9)
    b.annulus(0.0, 1.05, cab_y + 1.6, 1.0)
    b.line([0, mast, 0], [0, top, 0], EDGE, None, 0.05, 1.0, dens=60)
    for y in (mast - 3.0, mast + 1.5):
        b.circle(0.9, y, EDGE, 0.05, 0.7)
    return b.done()


def ringed_cylinder(seed=3):
    """A drum tower girdled by ring balconies whose lips burn; a band of windows between each pair; a dome."""
    b = TB(seed, dc=85.0, style=('masonry', 0.65, 1.15))
    top = HMAX
    body = top - 3.0
    R = 2.0
    rf = lambda y: np.full_like(y, R)
    b.revolve(rf, YBOT, body, win=dict(fh=1.3, ww=0.2, wh=0.5, off=0.0, p=0.5), nbay=30, seams=False)
    for y in np.arange(YBOT + 1.3, body, 2.6):
        b.annulus(R, R + 0.95, y, 1.0, dc=60)
        b.annulus(R, R + 0.95, y - 0.18, -1.0, dc=40)
        b.circle(R + 0.95, y, EDGE, 0.1, 0.85)
        b.circle(R + 0.95, y - 0.18, EDGE, 0.06, 0.55)
    b.revolve(lambda y: np.sqrt(np.maximum(R * R - (y - body) ** 2, 0.0)) + 0.01, body, body + R * 0.999,
              meridians=8, dc=110)
    b.line([0, body + R, 0], [0, top + 1.5, 0], EDGE, None, 0.05, 1.0, dens=60)
    return b.done()


def twisted(seed=4):
    """A square prism twisting a turn and a half as it rises: spiralling burning corners, twisted window grid."""
    b = TB(seed, dc=90.0, style=('grid', 0.7, 0.55))
    top = HMAX
    body = top - 4.0

    def apo(y):
        return 1.9 - 0.7 * np.clip(y / body, 0, 1)

    def tw(y):
        return 1.3 * np.pi * np.clip(y / body, -0.4, 1)
    b.prism(apo, YBOT, body, 4, rot=0.0, twist=tw, win=dict(fh=0.7, bw=0.55, ww=0.2, wh=0.36, off=0.0, p=0.5),
            edge_hot=0.85)
    h = float(apo(np.array([body]))[0])
    a = float(tw(np.array([body]))[0])
    b.ngon_ring(h, body, 4, twist=a, hot=0.9)
    # crown: a pyramid (faces + ridges)
    b.prism(lambda y: np.interp(y, [body, top], [h, 0.02]), body, top, 4, rot=0.0, twist=lambda y: np.full_like(y, a),
            edge_hot=0.95)
    return b.done()


def pagoda(seed=5):
    """Tiered pagoda: lit screens in every storey, dark tiled roofs with burning upturned eaves, a finial."""
    b = TB(seed, dc=85.0, win_p=0.5, style=('grid', 1.25, 0.7))
    top = HMAX
    tier = 3.4
    y = top - 4.5
    b.line([0, y, 0], [0, top, 0], EDGE, None, 0.06, 1.0, dens=60)
    for yy in np.linspace(y + 0.6, top - 0.6, 6):
        b.circle(0.24, yy, EDGE, 0.06, 0.8)
    k = 0
    r = b.r
    while y > YBOT:
        body_hw = 1.4 + 0.08 * min(k, 6)
        roof_hw = body_hw + 1.6
        y0 = y - tier
        # roof: flared skirt from the body top outward and down, corners upturned
        n = int(4 * 2 * roof_hw * 1.9 * b.dc * 0.9)
        s = r.random(n)
        side = r.integers(0, 4, n)
        t = r.uniform(-1, 1, n)
        hwv = body_hw + (roof_hw - body_hw) * s
        droop = -0.9 * s + 0.75 * s ** 3 * np.abs(t) ** 4
        yy = y - 0.1 + droop
        x = np.where(side == 0, hwv, np.where(side == 1, -hwv, t * hwv))
        z = np.where(side == 2, hwv, np.where(side == 3, -hwv, t * hwv))
        out = np.stack([np.where(side == 0, 1.0, np.where(side == 1, -1.0, 0.0)), np.full(n, 1.6),
                        np.where(side == 2, 1.0, np.where(side == 3, -1.0, 0.0))], 1)
        b.add(CRUST, np.stack([x, yy, z], 1), _unit(out), 4 * 2 * roof_hw * 1.9 / n, -1, 0.2)
        # the roof's underside (seen from below: dark, lit by the storey's glow)
        m2 = n // 2
        s2 = r.random(m2)
        side2 = r.integers(0, 4, m2)
        t2 = r.uniform(-1, 1, m2)
        hw2 = body_hw + (roof_hw - body_hw) * s2
        yy2 = y - 0.22 - 0.9 * s2 + 0.75 * s2 ** 3 * np.abs(t2) ** 4
        x2 = np.where(side2 == 0, hw2, np.where(side2 == 1, -hw2, t2 * hw2))
        z2 = np.where(side2 == 2, hw2, np.where(side2 == 3, -hw2, t2 * hw2))
        b.add(CRUST, np.stack([x2, yy2, z2], 1), np.array([0.0, -1.0, 0.0]), 4 * 2 * roof_hw * 1.9 / m2 * 0.5,
              -1, 0.3)
        # eave edge (dense, burning) with upturned corners, and the four hip ridges
        m = int(8 * roof_hw * b.dl * 1.5)
        t = r.uniform(-1, 1, m)
        side = r.integers(0, 4, m)
        ye = y - 0.1 - 0.9 + 0.75 * np.abs(t) ** 4
        x = np.where(side == 0, roof_hw, np.where(side == 1, -roof_hw, t * roof_hw))
        z = np.where(side == 2, roof_hw, np.where(side == 3, -roof_hw, t * roof_hw))
        ne = _unit(np.stack([x, np.full(m, 0.3), z], 1))
        b.add(EDGE, np.stack([x, ye, z], 1), ne, 8 * roof_hw * 0.09 / m, -1, 0.95)
        for c in range(4):
            ca = np.pi / 4 + c * np.pi / 2
            p0 = np.array([body_hw * np.sqrt(2) * np.cos(ca), y - 0.1, body_hw * np.sqrt(2) * np.sin(ca)])
            p1 = np.array([roof_hw * np.sqrt(2) * np.cos(ca), y - 0.1 - 0.9 + 0.75, roof_hw * np.sqrt(2) * np.sin(ca)])
            b.line(p0, p1, EDGE, None, 0.06, 0.8)
        # body storey with lit screens
        if y - 0.9 > y0 + 0.05:
            b.prism(lambda q, h=body_hw: np.full_like(q, h), y0, y - 0.9, 4, rot=0.0,
                    win=dict(fh=(y - 0.9 - y0) / 2, bw=0.7, ww=0.36, wh=0.62, off=0.0), floors=False, edge_hot=0.6)
            b.ngon_ring(body_hw + 0.02, y0 + 0.05, 4, kind=SEAM, width=0.06, hot=0.55)
        y = y0
        k += 1
    return b.done()


def pod_tower(seed=6):
    """A slender shaft carrying a great sphere (a lit window band round its equator), a small sphere, a needle."""
    b = TB(seed, dc=95.0, style=('grid', 0.78, 0.7))
    top = HMAX
    pod_y = top - 11.0
    R = 3.1
    shaft = 0.9
    b.revolve(lambda y: np.full_like(y, shaft), YBOT, pod_y - R + 0.3,
              win=dict(fh=3.0, ww=0.22, wh=1.6, off=0.4, p=0.5), nbay=6, seams=False)
    for y in np.arange(YBOT + 2.0, pod_y - R, 6.0):
        b.circle(shaft + 0.03, y, EDGE, 0.07, 0.8)
    for k in range(6):
        a = 2 * np.pi * (k + 0.5) / 6
        b.line([shaft * np.cos(a), YBOT, shaft * np.sin(a)], [shaft * np.cos(a), pod_y - R + 0.3, shaft * np.sin(a)],
               SEAM, None, 0.04, 0.4)
    sph = lambda y: np.sqrt(np.maximum(R * R - (y - pod_y) ** 2, 0)) + 0.01
    b.revolve(sph, pod_y - R * 0.999, pod_y + R * 0.999,
              win=dict(fh=0.78, ww=0.36, wh=0.5, off=R * 0.999 - 1.17 - 0.0, p=0.8), nbay=28, seams=False,
              meridians=12, dc=120)
    for lat in np.linspace(-0.85, 0.85, 9):
        b.circle(R * np.cos(np.arcsin(lat)) + 0.01, pod_y + R * lat, SEAM, 0.05, 0.55)
    r2 = 1.35
    y2 = pod_y + R + 2.4
    b.revolve(lambda y: np.sqrt(np.maximum(r2 * r2 - (y - y2) ** 2, 0)) + 0.01, y2 - r2 * 0.999, y2 + r2 * 0.999,
              meridians=6, dc=140)
    b.circle(r2 + 0.01, y2, EDGE, 0.06, 0.9)
    b.revolve(lambda y: np.full_like(y, 0.28), pod_y + R - 0.2, y2 - r2 + 0.1, dc=140)
    b.line([0, y2 + r2, 0], [0, top + 2.0, 0], EDGE, None, 0.05, 1.0, dens=60)
    return b.done()


def blade(seed=7):
    """A tall curved blade: lens-shaped section narrowing and leaning to a sharp tip, a glass grid of fire."""
    b = TB(seed, dc=85.0, win_p=0.34, style=('grid', 0.62, 0.52))
    top = HMAX
    r = b.r

    def sect(y):
        u = np.clip(y / top, 0, 1)
        L = 3.4 * (1 - u ** 1.6) + 0.05 + 0.03 * np.maximum(-y, 0)
        th = 0.55 * (1 - u) + 0.04
        lean = 2.2 * u ** 2.2
        return L, th, lean
    # faces (both sides of the lens): u across the face (-1..1), y
    ys = np.linspace(YBOT, top, 400)
    L_, th_, le_ = sect(ys)
    area = np.trapezoid(2 * 2 * L_ * 1.05, ys)
    m = int(area * b.dc * (1 + b.kc))
    y = r.uniform(YBOT, top, m)
    L, th, lean = sect(y)
    keep = r.random(m) < L / L_.max()
    y, L, th, lean = y[keep], L[keep], th[keep], lean[keep]
    m = len(y)
    s = r.uniform(-1, 1, m)
    side = np.where(r.random(m) < 0.5, -1.0, 1.0)
    x = s * L + lean
    zz = side * th * np.sqrt(np.maximum(1 - s * s, 0))
    dzds = -side * th * s / np.maximum(np.sqrt(np.maximum(1 - s * s, 1e-4)), 1e-2) / np.maximum(L, 1e-3)
    nrm = _unit(np.stack([-dzds * 0.3, np.zeros(m), side], 1))
    # window grid: floors 0.8, bays by fraction of the section
    fh, nbay = 0.62, 13
    kf = np.floor((y - YBOT) / fh)
    yc = YBOT + (kf + 0.5) * fh
    bb = np.floor((s + 1) / 2 * nbay)
    sc = -1 + (2 * bb + 1) / nbay
    Lc, thc, lc = sect(yc)
    inside = (np.abs(s - sc) * Lc < 0.11) & (np.abs(y - yc) < 0.17) & (np.abs(sc) < 0.9)
    nf = int(np.ceil((top - YBOT) / fh)) + 1
    wid0 = b.nwin
    b.nwin += 2 * nf * nbay
    sidx = (side > 0).astype(int)
    key = np.where(inside, wid0 + (sidx * nf + kf.astype(int)) * nbay + bb.astype(int), -1).astype(np.int32)
    b.crust_and_cracks(np.stack([x, y, zz], 1), nrm, key, b.dc, 0.3, True, (s * L, y), ('grid', fh, 2 * 3.4 / nbay))
    lit = r.random(2 * nf * nbay) < b.win_p
    for sd in (0, 1):
        sgn = 1.0 if sd else -1.0
        for k in range(nf):
            ycw = YBOT + (k + 0.5) * fh
            if ycw > top - 5.0:
                continue
            Lw, tw_, lw = [float(np.ravel(v)[0]) for v in sect(np.array([ycw]))]
            for bi in range(nbay):
                scw = -1 + (2 * bi + 1) / nbay
                if abs(scw) >= 0.9 or not lit[(sd * nf + k) * nbay + bi]:
                    continue
                mw = max(6, int(0.22 * 0.34 * b.dw))
                ss = scw + r.uniform(-1, 1, mw) * 0.11 / Lw
                yy = ycw + r.uniform(-1, 1, mw) * 0.17
                L2, t2, l2 = sect(yy)
                xw = ss * L2 + l2
                zw = sgn * (t2 * np.sqrt(np.maximum(1 - ss * ss, 0)) + 0.01)
                b.add(WIN, np.stack([xw, yy, zw], 1), np.array([0.0, 0.0, sgn]), 0.22 * 0.34 / mw,
                      wid0 + (sd * nf + k) * nbay + bi, 0.5)
    # floor seams across the faces
    for yy in np.arange(YBOT, top - 2.0, fh):
        Lf, tf, lf = [float(np.ravel(v)[0]) for v in sect(np.array([yy]))]
        for sgn in (-1.0, 1.0):
            ss = np.linspace(-1, 1, 24)
            pts = np.stack([ss * Lf + lf, np.full(24, yy), sgn * tf * np.sqrt(np.maximum(1 - ss * ss, 0))], 1)
            for i in range(23):
                b.line(pts[i], pts[i + 1], SEAM, np.array([0.0, 0.0, sgn]), 0.045, 0.45)
    # leading / trailing edges (burning), dense
    for sd in (-1, 1):
        L, th, lean = sect(ys)
        pts = np.stack([sd * L + lean, ys, np.zeros_like(ys)], 1)
        for i in range(len(pts) - 1):
            b.line(pts[i], pts[i + 1], EDGE, np.array([float(sd), 0.0, 0.0]), 0.09, 0.85)
    return b.done()


BUILDERS = [needle_spire, ziggurat, lattice_mast, ringed_cylinder, twisted, pagoda, pod_tower, blade]
NAMES = ['needle spire', 'ziggurat', 'lattice mast', 'ringed cylinder', 'twisted prism', 'pagoda', 'pod tower',
         'blade']


def build_all():
    return [f(i) for i, f in enumerate(BUILDERS)]


if __name__ == '__main__':
    import time
    t0 = time.time()
    tot = 0
    for nm, t in zip(NAMES, build_all()):
        k = t['kind']
        tot += len(k)
        print('%-16s %8d  crust %7d seam %7d edge %7d win %7d crack %7d (nwin %d)  y %.1f..%.1f' % (
            nm, len(k), (k == 0).sum(), (k == 1).sum(), (k == 2).sum(), (k == 3).sum(), (k == 4).sum(), t['nwin'],
            t['p'][:, 1].min(), t['p'][:, 1].max()))
    print('total', tot, '%.1fs' % (time.time() - t0))
