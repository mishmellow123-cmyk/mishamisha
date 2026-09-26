"""MAP (cut C): THE WORLD ANSWERS, told on a map. v2 frames 1920-2087 -> renders/map_C/.

    python render.py --frames 1920,1960,2000,2040,2080 --scale 0.5 --test   # review stills
    python render.py 1920 2087                                            # delivery frames
    python render.py --sheet                                              # review sheet (map_C.jpg)

Every frame is a pure function of its v2 frame number.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import look  # noqa: E402
import geo  # noqa: E402
import fire  # noqa: E402
from webmap import MapWeb, T0  # noqa: E402
from answer import Answer, T_CUT  # noqa: E402
from noise import gnoise  # noqa: E402

cv2.setNumThreads(2)
OUT = os.path.join(geo.ROOT, 'renders', 'map_C')
FIRST, LAST = 1920, 2087
TEXT_IN, TEXT_OUT = 1960, 2040          # cut C's line ("And all the peoples answered.") sits over the map here


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def smoother(x):
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (x * 6 - 15) + 10)


def lerp(a, b, t):
    return a + (b - a) * t


def spline(t, keys):
    """Catmull-Rom through (frame, value) keys."""
    ts = [k[0] for k in keys]
    vs = [k[1] for k in keys]
    if t <= ts[0]:
        return vs[0]
    if t >= ts[-1]:
        return vs[-1]
    i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
    u = (t - ts[i]) / (ts[i + 1] - ts[i])
    p0 = vs[max(i - 1, 0)]
    p1 = vs[i]
    p2 = vs[i + 1]
    p3 = vs[min(i + 2, len(vs) - 1)]
    # scale tangents for non-uniform key spacing
    t0 = ts[max(i - 1, 0)]
    t3 = ts[min(i + 2, len(ts) - 1)]
    d1 = (p2 - p0) / max(ts[i + 1] - t0, 1e-9) * (ts[i + 1] - ts[i])
    d2 = (p3 - p1) / max(t3 - ts[i], 1e-9) * (ts[i + 1] - ts[i])
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p1 + (u3 - 2 * u2 + u) * d1 + (-2 * u3 + 3 * u2) * p2 + (u3 - u2) * d2


# ================================================================ camera ===

class Cam:
    """Perspective camera over the table (the map plane z = 0, units = map degrees)."""

    def __init__(self, tx, ty, width, tilt, heading, W, H, hfov=38.0):
        self.W, self.H = W, H
        th, ps = math.radians(tilt), math.radians(heading)
        self.F = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)
        D = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)
        hd = np.array([math.sin(ps), math.cos(ps), 0.0])
        T = np.array([tx, ty, 0.0])
        self.pos = T - D * math.sin(th) * hd + np.array([0.0, 0.0, D * math.cos(th)])
        f = T - self.pos
        self.f = f / np.linalg.norm(f)
        self.r = np.array([math.cos(ps), -math.sin(ps), 0.0])
        self.u = np.cross(self.r, self.f)
        R = np.stack([self.r, self.u, self.f])
        K = np.array([[self.F, 0, W / 2.0], [0, -self.F, H / 2.0], [0, 0, 1.0]])
        M = np.array([[1.0, 0, -self.pos[0]], [0, 1.0, -self.pos[1]], [0, 0, -self.pos[2]]])
        self.Hm = K @ R @ M                 # map (X, Y, 1) -> screen (homogeneous)
        self.Hinv = np.linalg.inv(self.Hm)

    def project(self, P):
        """P (N,3) world -> screen (N,2), depth (N,)."""
        d = np.asarray(P, np.float64) - self.pos[None, :]
        xc = d @ self.r
        yc = d @ self.u
        zc = d @ self.f
        zc = np.maximum(zc, 1e-6)
        return np.stack([self.W / 2.0 + self.F * xc / zc, self.H / 2.0 - self.F * yc / zc], 1), zc

    def scale(self, P):
        """Screen px per map unit near world points (isotropic approximation)."""
        uv, z = self.project(P)
        return self.F / z

    def up2d(self, P, h=0.5):
        """Screen direction (unit) and length (px per unit height) of world-up at points P."""
        a, _ = self.project(P)
        Q = np.array(P, np.float64)
        Q[:, 2] += h
        b, _ = self.project(Q)
        d = b - a
        L = np.linalg.norm(d, axis=1)
        return d / np.maximum(L, 1e-9)[:, None], L / h

    def screen_to_map(self, sx, sy):
        p = self.Hinv @ np.stack([sx, sy, np.ones_like(sx)])
        return p[0] / p[2], p[1] / p[2]


# ================================================================ sheet ===

class Sheet:
    """The baked map pyramid, sampled through the camera's plane homography (trilinear)."""

    def __init__(self, tag):
        import bake
        self.levels = []
        L = 0
        while os.path.exists(bake.level_path(tag, L)):
            self.levels.append(np.load(bake.level_path(tag, L), mmap_mode='r'))
            L += 1
        if not self.levels:
            raise SystemExit(f'no baked pyramid "{tag}" - run bake.py pyramid')
        self.ppd0 = self.levels[0].shape[1] / (geo.TEX_X1 - geo.TEX_X0)
        self.lut = look.srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)
        rel = np.load(os.path.join(geo.CACHE, 'relief_4.npy'))
        rel = cv2.GaussianBlur(rel, (0, 0), 1.0)
        gy, gx = np.gradient(rel, 1.0 / 4.0)
        # gradient in map axes: X right, Y up (rows go down)
        self.nrm = np.dstack([-gx, gy]).astype(np.float32)
        self.nrm_ppd = 4.0

    def _A(self, ppd):
        return np.array([[ppd, 0, -geo.TEX_X0 * ppd - 0.5], [0, -ppd, geo.TEX_Y1 * ppd - 0.5], [0, 0, 1.0]])

    def lod(self, cam, step=16, bias=0.0):
        W, H = cam.W, cam.H
        ys = np.arange(0, H + step, step, dtype=np.float64)
        xs = np.arange(0, W + step, step, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        X0, Y0 = cam.screen_to_map(gx.ravel(), gy.ravel())
        X1, Y1 = cam.screen_to_map(gx.ravel() + 1, gy.ravel())
        X2, Y2 = cam.screen_to_map(gx.ravel(), gy.ravel() + 1)
        a = np.hypot(X1 - X0, Y1 - Y0)
        b = np.hypot(X2 - X0, Y2 - Y0)
        fp = np.sqrt(np.maximum(a, b) * np.minimum(a, b)) ** 0.5 * np.maximum(a, b) ** 0.5
        lod = np.log2(np.maximum(fp * self.ppd0, 1e-6)) + bias
        lod = lod.reshape(gx.shape).astype(np.float32)
        return cv2.resize(lod, (W, H), interpolation=cv2.INTER_LINEAR)[:H, :W]

    def warp_level(self, cam, L):
        lev = self.levels[L]
        ppd = self.ppd0 / (2 ** L)
        Hh, Ww = lev.shape[:2]
        T = self._A(ppd) @ cam.Hinv
        c = T @ np.array([[0, cam.W, cam.W, 0], [0, 0, cam.H, cam.H], [1, 1, 1, 1.0]])
        u, v = c[0] / c[2], c[1] / c[2]
        u0 = int(max(math.floor(u.min()) - 3, 0))
        v0 = int(max(math.floor(v.min()) - 3, 0))
        u1 = int(min(math.ceil(u.max()) + 4, Ww))
        v1 = int(min(math.ceil(v.max()) + 4, Hh))
        if u1 <= u0 or v1 <= v0:
            return np.zeros((cam.H, cam.W, 3), np.uint8)
        crop = np.ascontiguousarray(lev[v0:v1, u0:u1])
        Tc = np.array([[1, 0, -u0], [0, 1, -v0], [0, 0, 1.0]]) @ T
        return cv2.warpPerspective(crop, Tc, (cam.W, cam.H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(22, 14, 10))

    def sample(self, cam, bias=-0.25):
        lod = self.lod(cam, bias=bias)
        nL = len(self.levels)
        lod = np.clip(lod, 0.0, nL - 1.0)
        L0 = int(math.floor(float(lod.min())))
        L1 = int(math.ceil(float(lod.max())))
        out = np.zeros((cam.H, cam.W, 3), np.float32)
        for L in range(L0, min(L1, nL - 1) + 1):
            w = np.clip(1.0 - np.abs(lod - L), 0.0, 1.0)
            if w.max() <= 0:
                continue
            img = self.warp_level(cam, L)
            out += w[..., None] * self.lut[img]
        return out

    def normals(self, cam):
        A = self._A(self.nrm_ppd)
        T = A @ cam.Hinv
        n2 = cv2.warpPerspective(self.nrm, T, (cam.W, cam.H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0))
        nz = np.ones(n2.shape[:2], np.float32)
        n = np.dstack([n2, nz])
        return n / np.linalg.norm(n, axis=2, keepdims=True)


# ============================================================ fire light ===

class LightGrid:
    """Irradiance cast on the paper by the fires: sources splatted into a map-space grid and
    spread by a sum of Gaussians (a small flame's light pool), then warped to the screen."""

    KERNEL = ((0.45, 0.5), (1.4, 0.35), (4.0, 0.16), (10.0, 0.06))

    def __init__(self, cam, cells=420):
        W, H = cam.W, cam.H
        xs = np.array([0, W, W, 0, W / 2, W / 2, 0, W], np.float64)
        ys = np.array([0, 0, H, H, 0, H, H / 2, H / 2], np.float64)
        X, Y = cam.screen_to_map(xs, ys)
        m = 10.0
        self.X0, self.X1 = X.min() - m, X.max() + m
        self.Y0, self.Y1 = Y.min() - m, Y.max() + m
        self.cell = (self.X1 - self.X0) / cells
        self.nx = cells
        self.ny = int(math.ceil((self.Y1 - self.Y0) / self.cell)) + 1
        self.G = np.zeros((self.ny, self.nx, 3), np.float32)
        self.cam = cam

    def add(self, X, Y, pw):
        """Sources at map (X, Y) with power pw (N,3)."""
        if len(X) == 0:
            return
        gx = (np.asarray(X) - self.X0) / self.cell - 0.5
        gy = (self.Y1 - np.asarray(Y)) / self.cell - 0.5
        ix = np.floor(gx).astype(np.int64)
        iy = np.floor(gy).astype(np.int64)
        fx = gx - ix
        fy = gy - iy
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            jx = ix + dx
            jy = iy + dy
            ok = (jx >= 0) & (jx < self.nx) & (jy >= 0) & (jy < self.ny)
            np.add.at(self.G, (jy[ok], jx[ok]), (pw[ok] * w[ok][:, None]).astype(np.float32))

    def irradiance(self):
        E = np.zeros_like(self.G)
        for sig, wt in self.KERNEL:
            s = sig / self.cell
            if s < 0.35:
                E += self.G * (wt / (2 * math.pi * sig * sig)) * (self.cell ** 2 / 1.0)
            else:
                E += cv2.GaussianBlur(self.G, (0, 0), s) * (wt / (2 * math.pi * sig * sig)) * (self.cell ** 2)
        A = np.array([[1 / self.cell, 0, -self.X0 / self.cell - 0.5], [0, -1 / self.cell, self.Y1 / self.cell - 0.5], [0, 0, 1.0]])
        T = A @ self.cam.Hinv
        return cv2.warpPerspective(E, T, (self.cam.W, self.cam.H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# ================================================================ colours ===

def lin(h):
    return look.hexrgb(h).astype(np.float64)


PAL = None


def pal():
    global PAL
    if PAL is None:
        PAL = dict(white=lin('#FFF4DC'), hot=lin('#FFC873'), gold=lin('#FFA93A'), amber=lin('#EE7E1C'),
                   deep=lin('#C2410C'), ember=lin('#8E2A08'), light=np.array([1.0, 0.62, 0.3]))
    return PAL


def fire_col(tau):
    """Colour of a burned thread tau frames after the fire passed (linear, unit-ish)."""
    p = pal()
    tau = np.asarray(tau, np.float64)[..., None]
    a = np.clip(tau / 2.5, 0, 1)
    b = np.clip((tau - 2.5) / 10.0, 0, 1)
    c = np.clip((tau - 12.0) / 30.0, 0, 1)
    col = p['hot'] * (1 - a) + p['gold'] * a
    col = col * (1 - b) + p['amber'] * b
    col = col * (1 - c) + p['deep'] * c
    return col


# ================================================================== shot ===

class Shot:
    def __init__(self, tag='full'):
        self.sheet = Sheet(tag)
        self.web = MapWeb()
        w = self.web
        # fire, not fibre: no new threads after T_CUT, no loop-closing cross-links
        self.akeep, t_cut = w.cut(T_CUT)
        w.t_ign = t_cut
        w.t_ign[0] = T0 - 60.0              # the first beacon already burns (small) as the shot opens
        self.ans = Answer(w.P, t_cut)
        # per-arc inverse of the head motion: frame at which the head passed u
        self.tpass = []
        for k in range(len(w.ai)):
            ts = np.linspace(w.tl[k], w.ta[k], 48)
            us = np.array([w.head_u(k, tt) for tt in ts])
            us[-1] = 1.0
            self.tpass.append((us, ts))
        rng = np.random.default_rng(3)
        self.nphase = rng.random(len(w.P)) * 1000.0
        self.aphase = rng.random(len(w.ai)) * 2 * np.pi
        self.aper = rng.uniform(55.0, 110.0, len(w.ai))
        self.gseed = rng.integers(0, 1 << 20, len(w.ai))
        # every beacon on the sheet: the web's (lit before the cut) and the answering ones
        lit = np.where(np.isfinite(w.t_ign))[0]
        a = self.ans
        self.bP = np.concatenate([w.P[lit], a.P])
        self.bt = np.concatenate([w.t_ign[lit], a.t_ign])
        kd = w.kind[lit]
        # which of the web's beacons stand on a height, a shore or a river (they keep burning)
        import bake as _bake
        import features as _ft
        gb = _bake.glyph_bank()
        mi = np.where(gb['kind'] == 0)[0]
        summ = np.array([gb['sP'][gb['sO'][gb['gS'][g]]:gb['sO'][gb['gS'][g] + 1]].max(axis=0) for g in mi])
        from scipy.spatial import cKDTree as _T
        d_peak = _T(summ).query(w.P[lit])[0]
        riv = np.concatenate([R for R, _, _ in _ft.river_lines()])
        d_riv = _T(riv).query(w.P[lit])[0]
        cst = np.concatenate([R for R, h in _ft.coast_lines() if not h])
        d_cst = _T(cst).query(w.P[lit])[0]
        on_peak = d_peak < 1.4
        on_feat = on_peak | (d_riv < 0.6) | (d_cst < 0.6)
        base = np.where(on_peak, 1.3, np.where(on_feat, 0.9, 0.85))
        abase = a.size * np.select([a.kind == 0, a.kind == 1, a.kind == 2], [1.0, 0.9, 0.85], 0.85)
        self.bsize = np.concatenate([base * rng.uniform(0.8, 1.2, len(lit)), abase])
        self.bsize[np.where(lit == 0)[0]] = 2.0 / 0.45     # the first beacon: 2 map-degrees of flame
        # relay fires out in the open burn down once they have passed the fire on;
        # the beacons on heights, shores and rivers keep burning
        self.bdwindle = np.concatenate([~on_feat & (lit != 0), np.zeros(len(a.P), bool)])
        self.borg = np.concatenate([lit == 0, np.zeros(len(a.P), bool)])
        self.bphase = np.concatenate([self.nphase[lit], a.phase])
        self.bid = np.concatenate([lit, 5000 + np.arange(len(a.P))])
        self._spark_bank(rng)

    # ------------------------------------------------------------ camera ---
    def camera(self, t, W, H):
        # log-width keys: close on the Himalaya, then a steady crane up and back to the whole sheet
        lw = spline(t, [(1905, math.log(42.0)), (1920, math.log(42.0)), (1950, math.log(54.0)),
                        (1985, math.log(120.0)), (2025, math.log(290.0)), (2060, math.log(368.0)),
                        (2100, math.log(372.0))])
        tx = spline(t, [(1905, 76.0), (1920, 76.0), (1950, 73.0), (1985, 52.0), (2025, 14.0), (2060, 0.5), (2100, 0.0)])
        ty = spline(t, [(1905, 25.5), (1920, 25.5), (1950, 24.5), (1985, 22.0), (2025, 16.0), (2060, 12.5), (2100, 12.0)])
        tilt = spline(t, [(1905, 46.0), (1920, 46.0), (1950, 43.0), (1985, 35.0), (2025, 22.0), (2060, 13.0), (2100, 12.0)])
        head = spline(t, [(1905, -7.0), (1920, -7.0), (1950, -6.0), (1985, -3.5), (2025, -1.0), (2060, 0.0), (2100, 0.0)])
        return Cam(tx, ty, math.exp(lw), tilt, head, W, H)

    def calm(self, t):
        """Text window: how much to hush the band y 515..745 of 804 (0..1)."""
        return smooth((t - (TEXT_IN - 10)) / 10.0) * (1 - smooth((t - TEXT_OUT - 2) / 10.0))

    def flare(self, t):
        """The first beacon's flare: rises 1920->1923, settles by ~1945."""
        if t < T0:
            return 0.0
        a = smooth((t - T0) / 3.0)
        return a * math.exp(-max(t - T0 - 3.0, 0.0) / 9.0)

    # ------------------------------------------------------------ sparks ---
    def _spark_bank(self, rng):
        """Deterministic sparks: shed along every burning thread, burst at every ignition."""
        w = self.web
        K, U, LIFE, V = [], [], [], []
        for k in range(len(w.ai)):
            if w.ak[k] == 1 or not self.akeep[k]:
                continue
            n = int(max(3, w.plen[k] / 0.45))
            K.append(np.full(n, k))
            U.append(np.sort(rng.random(n)))
            LIFE.append(rng.uniform(5.0, 13.0, n))
            V.append(np.column_stack([rng.normal(0, 0.022, n), rng.normal(0, 0.022, n), rng.uniform(0.03, 0.075, n)]))
        self.sk = np.concatenate(K)
        self.su = np.concatenate(U)
        self.slife = np.concatenate(LIFE)
        self.sv = np.concatenate(V)
        self.st = np.zeros(len(self.sk))
        for k in np.unique(self.sk):
            m = self.sk == k
            us, ts = self.tpass[k]
            self.st[m] = np.interp(self.su[m], us, ts)
        self.sP = np.zeros((len(self.sk), 2))
        for k in np.unique(self.sk):
            m = self.sk == k
            path = w.paths[k]
            uu = np.linspace(0, 1, len(path))
            self.sP[m, 0] = np.interp(self.su[m], uu, path[:, 0])
            self.sP[m, 1] = np.interp(self.su[m], uu, path[:, 1])
        # ignition bursts at every beacon (and a steady fountain from the first one)
        n = 9
        nb = len(self.bP)
        self.bi = np.repeat(np.arange(nb), n)
        self.bdt = rng.uniform(0.0, 3.0, nb * n)
        self.blife = rng.uniform(8.0, 18.0, nb * n)
        ang = rng.random(nb * n) * 2 * np.pi
        sp = rng.uniform(0.02, 0.06, nb * n)
        self.bv = np.column_stack([np.cos(ang) * sp, np.sin(ang) * sp, rng.uniform(0.05, 0.12, nb * n)])
        m = 260
        self.oi_t = T0 - 30.0 + np.sort(rng.random(m)) * 200.0
        self.oi_life = rng.uniform(14.0, 32.0, m)
        ang = rng.random(m) * 2 * np.pi
        sp = rng.uniform(0.004, 0.025, m)
        self.oi_v = np.column_stack([np.cos(ang) * sp, np.sin(ang) * sp, rng.uniform(0.035, 0.08, m)])
        # the flare throws a fountain
        m2 = 70
        self.of_dt = rng.uniform(0.5, 5.0, m2)
        self.of_life = rng.uniform(12.0, 30.0, m2)
        ang = rng.random(m2) * 2 * np.pi
        sp = rng.uniform(0.03, 0.12, m2)
        self.of_v = np.column_stack([np.cos(ang) * sp, np.sin(ang) * sp, rng.uniform(0.08, 0.2, m2)])

    def draw_sparks(self, cam, t, emis, scale, hush):
        w = self.web
        p = pal()
        segs0, segs1, I = [], [], []

        def fly(P0, v, age, life):
            # rise, drift, slow down (air drag); a little curl
            a = age
            drag = (1.0 - np.exp(-a / 6.0)) * 6.0
            pos = P0 + v * drag[:, None]
            pos[:, 2] += 0.004 * a * a / (1.0 + 0.1 * a)
            return pos

        # thread sparks
        age = t - self.st
        m = (age >= 0) & (age < self.slife)
        if m.any():
            P0 = np.column_stack([self.sP[m], np.full(m.sum(), 0.02)])
            a = age[m]
            segs0.append(fly(P0, self.sv[m], np.maximum(a - 0.9, 0), self.slife[m]))
            segs1.append(fly(P0, self.sv[m], a, self.slife[m]))
            I.append(5.0 * (1 - a / self.slife[m]) ** 1.5)
        # ignition bursts
        tb = self.bt[self.bi] + self.bdt
        age = t - tb
        m = (age >= 0) & (age < self.blife) & (~self.borg[self.bi])
        if m.any():
            P0 = np.column_stack([self.bP[self.bi[m]], np.full(m.sum(), 0.15)])
            a = age[m]
            segs0.append(fly(P0, self.bv[m], np.maximum(a - 0.9, 0), self.blife[m]))
            segs1.append(fly(P0, self.bv[m], a, self.blife[m]))
            I.append(4.0 * (1 - a / self.blife[m]) ** 1.5)
        # the first beacon: a steady stream, and the fountain of the flare
        O = np.array([w.P[0, 0], w.P[0, 1], 0.5])
        age = t - self.oi_t
        m = (age >= 0) & (age < self.oi_life)
        if m.any():
            a = age[m]
            P0 = np.repeat(O[None, :], m.sum(), 0)
            segs0.append(fly(P0, self.oi_v[m], np.maximum(a - 0.9, 0), self.oi_life[m]))
            segs1.append(fly(P0, self.oi_v[m], a, self.oi_life[m]))
            I.append(3.5 * (1 - a / self.oi_life[m]) ** 1.3)
        age = t - (T0 + self.of_dt)
        m = (age >= 0) & (age < self.of_life)
        if m.any():
            a = age[m]
            P0 = np.repeat(O[None, :], m.sum(), 0)
            segs0.append(fly(P0, self.of_v[m], np.maximum(a - 0.9, 0), self.of_life[m]))
            segs1.append(fly(P0, self.of_v[m], a, self.of_life[m]))
            I.append(7.0 * (1 - a / self.of_life[m]) ** 1.3)
        if not segs0:
            return
        A = np.concatenate(segs0)
        B = np.concatenate(segs1)
        I = np.concatenate(I)
        ua, za = cam.project(A)
        ub, zb = cam.project(B)
        n = len(I)
        xs = np.empty(2 * n)
        ys = np.empty(2 * n)
        xs[0::2] = ua[:, 0]
        xs[1::2] = ub[:, 0]
        ys[0::2] = ua[:, 1]
        ys[1::2] = ub[:, 1]
        sc = cam.F / zb * scale
        sig = np.repeat(np.clip(0.012 * sc, 0.45, 1.2), 2)
        g = np.repeat(np.clip(0.012 * sc / 0.45, 0.25, 1.0), 2)
        if hush is not None:
            g = g * np.repeat(hush(ub[:, 1]), 2)
        col = np.outer(np.repeat(I, 2) * g, p['gold'] * np.array([1.0, 0.85, 0.6]))
        st = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
        fire.add_lines(emis, xs, ys, col, sig, st, np.ones(2 * n))

    # -------------------------------------------------------------- fire ---
    def draw_web(self, cam, t, emis, scorch, grid, scale, hush):
        w = self.web
        p = pal()
        act = np.where(self.akeep & (w.tl <= t))[0]
        xs, ys, cols, sig, starts = [], [], [], [], [0]
        sxs, sys_, scol, ssig, sst = [], [], [], [], [0]
        heads = []
        lx, ly, lp = [], [], []
        for k in act:
            uh = w.head_u(k, t)
            if uh <= 0:
                continue
            path = w.paths[k]
            n = len(path)
            m = int(math.ceil(uh * (n - 1)))
            u = np.linspace(0, 1, n)[:m + 1]
            u[-1] = min(u[-1], uh)
            if len(u) < 2:
                continue
            uu = np.linspace(0, 1, n)
            P2 = np.stack([np.interp(u, uu, path[:, 0]), np.interp(u, uu, path[:, 1])], 1)
            hh = np.interp(u, uu, w.heights[k])
            us, ts = self.tpass[k]
            tau = t - np.interp(u, us, ts)
            leap = w.ak[k] == 1
            cross = w.ak[k] == 3
            L = w.plen[k]
            gain = 0.7 if cross else 1.0
            if not leap:
                P3 = np.column_stack([P2, np.full(len(u), 0.02)])
                uv, z = cam.project(P3)
                sc = cam.F / z * scale
                # embers are uneven along the line and breathe slowly
                sd = float(self.gseed[k] % 997)
                gran = np.array([0.55 + 0.6 * gnoise(uj * L / 0.22 + sd, t * 0.06, 31) for uj in u])
                gran = np.clip(gran, 0.15, 1.2)
                # the burn cools: a hot front, embers, then only the scorch in the paper
                hot = 2.4 * np.exp(-tau / 1.3)
                ember = (1.2 * np.exp(-tau / 8.0) + 0.3 * np.exp(-tau / 38.0)) * gran
                I = (hot + ember) * gain
                col = fire_col(tau) * I[:, None]
                wd = (0.018 + 0.022 * np.exp(-tau / 4.0)) * sc
                g = np.minimum(1.0, (wd / 0.6) ** 0.45)
                xs.append(uv[:, 0]); ys.append(uv[:, 1]); cols.append(col * g[:, None]); sig.append(np.maximum(wd, 0.6))
                starts.append(starts[-1] + len(u))
                # scorch: a brown line and a singed halo develop behind the fire
                s_amt = 0.7 * np.clip((tau - 0.5) / 10.0, 0, 1) ** 0.7 * (0.75 if cross else 1.0)
                s_amt = s_amt * np.minimum(1.0, (0.045 * sc / 0.5) ** 0.5)
                sxs.append(uv[:, 0]); sys_.append(uv[:, 1]); scol.append(np.repeat(s_amt[:, None], 3, 1))
                ssig.append(np.maximum(0.045 * sc, 0.5)); sst.append(sst[-1] + len(u))
                sxs.append(uv[:, 0]); sys_.append(uv[:, 1]); scol.append(np.repeat(0.22 * s_amt[:, None], 3, 1))
                ssig.append(np.maximum(0.16 * sc, 0.6)); sst.append(sst[-1] + len(u))
                # light cast by the fresh part of the thread onto the paper
                sel = np.arange(0, len(u), max(1, len(u) // max(2, int(L / 0.7))))
                lx.append(P2[sel, 0]); ly.append(P2[sel, 1])
                lw_ = (0.5 * np.exp(-tau[sel] / 3.0) + 0.1 * np.exp(-tau[sel] / 20.0)) * gain
                lp.append(lw_[:, None] * p['light'][None, :] * (L / max(len(sel), 1)))
                if uh < 1.0:
                    heads.append((P2[-1], 0.0, False, k))
            else:
                # a spark flies over the sea; behind it a comet tail; after landing the arc fades
                P3 = np.column_stack([P2, hh + 0.02])
                uv, z = cam.project(P3)
                sc = cam.F / z * scale
                back = (u[-1] - u) * L
                if uh < 1.0:
                    I = 14.0 * np.exp(-back / 0.8) + 2.2 * np.exp(-back / 4.0)
                    heads.append((P2[-1], hh[-1], True, k))
                else:
                    age = t - w.ta[k]
                    I = (3.0 * np.exp(-back / 6.0) + 0.8) * math.exp(-age / 6.0)
                col = fire_col(np.maximum(tau, 0) * 0.4) * I[:, None]
                wd = 0.03 * sc
                g = np.minimum(1.0, (wd / 0.6) ** 0.45)
                xs.append(uv[:, 0]); ys.append(uv[:, 1]); cols.append(col * g[:, None]); sig.append(np.maximum(wd, 0.6))
                starts.append(starts[-1] + len(u))
                # below it on the paper: a dotted sea-route of embers that scorch
                G3 = np.column_stack([P2, np.full(len(u), 0.02)])
                guv, gz = cam.project(G3)
                gsc = cam.F / gz * scale
                dash = np.clip((np.sin(u * L / 0.6 * 2 * np.pi) - 0.1) * 3.0, 0, 1)
                gt = tau - 3.0
                gI = (4.0 * np.exp(-np.maximum(gt, 0) / 4.0) + 0.45) * (gt > 0) * dash
                gw = 0.022 * gsc
                gg = np.minimum(1.0, (gw / 0.6) ** 0.45)
                xs.append(guv[:, 0]); ys.append(guv[:, 1]); cols.append(fire_col(np.maximum(gt, 0)) * (gI * gg)[:, None])
                sig.append(np.maximum(gw, 0.6)); starts.append(starts[-1] + len(u))
                s_amt = 0.55 * np.clip((gt - 1.0) / 12.0, 0, 1) * dash
                sxs.append(guv[:, 0]); sys_.append(guv[:, 1]); scol.append(np.repeat(s_amt[:, None], 3, 1))
                ssig.append(np.maximum(0.045 * gsc, 0.5)); sst.append(sst[-1] + len(u))
        if xs:
            X = np.concatenate(xs).astype(np.float64)
            Y = np.concatenate(ys).astype(np.float64)
            C = np.concatenate(cols).astype(np.float64)
            S = np.concatenate(sig).astype(np.float64)
            if hush is not None:
                C = C * hush(Y)[:, None]
            st = np.asarray(starts, np.int64)
            one = np.ones(len(X))
            fire.add_lines(emis, X, Y, C, S, st, one)
            fire.add_lines(emis, X, Y, C * 0.03, S * 4.0, st, one)
        if sxs:
            fire.add_lines(scorch, np.concatenate(sxs).astype(np.float64), np.concatenate(sys_).astype(np.float64),
                           np.concatenate(scol).astype(np.float64), np.concatenate(ssig).astype(np.float64),
                           np.asarray(sst, np.int64), np.ones(sum(len(a) for a in sxs)))
        if lx:
            grid.add(np.concatenate(lx), np.concatenate(ly), np.concatenate(lp))
        self.draw_heads(cam, t, emis, grid, heads, scale, hush)
        self.draw_nodes(cam, t, emis, scorch, grid, scale, hush)
        self.draw_sparks(cam, t, emis, scale, hush)

    def draw_heads(self, cam, t, emis, grid, heads, scale, hush):
        if not heads:
            return
        p = pal()
        P = np.array([[h[0][0], h[0][1], h[1] + 0.03] for h in heads])
        uv, z = cam.project(P)
        sc = cam.F / z * scale
        leap = np.array([h[2] for h in heads])
        mul = np.ones(len(P)) if hush is None else hush(uv[:, 1])
        pk = np.where(leap, 18.0, 7.0) * mul
        fire.splat_points(emis, uv[:, 0].copy(), uv[:, 1].copy(), np.outer(pk, p['hot']), np.maximum(0.03 * sc, 0.55))
        fire.splat_points(emis, uv[:, 0].copy(), uv[:, 1].copy(), np.outer(pk * 0.03, p['gold']), np.maximum(0.25 * sc, 1.0))
        # a little licking flame rides the front of each burning thread
        up, upl = cam.up2d(P)
        for i in range(len(heads)):
            if leap[i]:
                continue
            hp = 0.3 * upl[i] * scale
            if hp > 1.5:
                fire.flame(emis, uv[i, 0], uv[i, 1], hp, up[i, 0], up[i, 1], t * 1.9, int(heads[i][3]) * 7 + 3,
                           2.4 * mul[i] * min(1.0, (hp - 1.5) / 1.0), 0.9)
        grid.add(P[~leap, 0], P[~leap, 1], np.outer(np.full((~leap).sum(), 1.4), p['light']))
        # leaping sparks light the sea beneath them (height-dependent pool)
        self.leap_light = [(P[i], 2.5) for i in range(len(P)) if leap[i]]

    def draw_nodes(self, cam, t, emis, scorch, grid, scale, hush):
        """Every lit beacon: a small living flame of its own size, a pool of light, soot."""
        p = pal()
        m = self.bt <= t
        if not m.any():
            return
        lit = np.where(m)[0]
        age = t - self.bt[lit]
        org = self.borg[lit]
        size = self.bsize[lit]
        # relay fires burn down after they have passed the fire on
        dw = self.bdwindle[lit]
        size = np.where(dw, size * (1.0 - 0.8 * np.clip((age - 12.0) / 36.0, 0, 1)), size)
        grow = np.clip(age / 7.0, 0, 1)
        grow = grow * grow * (3 - 2 * grow)
        flare = np.exp(-age / 4.0) * np.clip(age / 1.5, 0, 1)
        fo = self.flare(t)
        flare = np.where(org, 1.5 * fo, flare)
        ph = self.bphase[lit]
        fl = 1.0 + 0.12 * np.sin(t * 0.5 + ph) + 0.07 * np.sin(t * 1.3 + 2 * ph)
        h0 = 0.45 * size
        hgt = h0 * (grow * fl + 0.6 * flare)
        P = np.column_stack([self.bP[lit], np.zeros(len(lit))])
        uv, z = cam.project(P)
        sc = cam.F / z * scale
        up, upl = cam.up2d(P)
        # physical height on the paper, but never smaller on screen than a readable flame
        floor = 2.2 * np.minimum(size, 2.4) ** 1.5 * scale * (grow * fl + 0.6 * flare)
        hp = np.maximum(hgt * upl * scale, floor)
        mul = np.ones(len(lit)) if hush is None else hush(uv[:, 1])
        # flash of ignition
        fk = np.exp(-age / 2.0) * (age >= 0) * (~org)
        fire.splat_points(emis, uv[:, 0].copy(), uv[:, 1].copy(), np.outer(6.0 * fk * mul * np.minimum(size, 1.3), p['hot']),
                          np.maximum(0.05 * sc, 0.55))
        fire.splat_points(emis, uv[:, 0].copy(), uv[:, 1].copy(),
                          np.outer(0.7 * np.exp(-age / 6.0) * (~org) * mul + 0.25 * org * (1 + fo), p['gold']),
                          np.maximum(0.45 * sc, 1.0))
        # flame sprites (a crossfade to a dot only for the very smallest)
        xf = np.clip((hp - 1.4) / 0.8, 0.0, 1.0)
        for i in np.where(xf > 0.0)[0]:
            g = (1.5 if org[i] else 2.4) * mul[i] * xf[i] * min(1.0, 0.55 + 0.45 * size[i])
            fire.flame(emis, uv[i, 0], uv[i, 1], hp[i], up[i, 0], up[i, 1], t + ph[i],
                       int(self.bid[lit[i]]) * 13 + 1, g, 1.3 if org[i] else 1.0)
        sm = xf < 1.0
        if sm.any():
            # the smallest fires are points of light, as bright as they are big; a relay that
            # has burned down is a dull red ember
            c = uv[sm]
            q = grow[sm] * fl[sm] * mul[sm] * (1.0 - xf[sm]) * np.minimum(size[sm], 1.2) ** 1.5
            ember = np.clip((0.55 - size[sm]) / 0.3, 0, 1)[:, None]
            col = p['gold'][None, :] * (1 - ember) + p['ember'][None, :] * 0.6 * ember
            fire.splat_points(emis, c[:, 0].copy(), c[:, 1].copy(), 3.2 * q[:, None] * col, np.full(sm.sum(), 0.55))
        # soot ring under every beacon
        fire.splat_points(scorch, uv[:, 0].copy(), uv[:, 1].copy(), np.outer(0.3 * grow, np.ones(3)), np.maximum(0.2 * sc, 0.6))
        pw = (hgt / 0.45) * (1.0 + 2.0 * flare) * fl
        pw = np.where(org, pw * 5.0, pw)
        grid.add(self.bP[lit, 0], self.bP[lit, 1], np.outer(2.6 * pw, p['light']))

    # ------------------------------------------------------------- frame ---
    def hush_fn(self, t, H):
        k = self.calm(t)
        if k <= 0:
            return None
        # full hush over y 555..705 of 804 (the words), easing in/out over ~110 px either side
        y0, y1, soft = 555.0 / 804 * H, 705.0 / 804 * H, 110.0 / 804 * H

        def f(y):
            y = np.asarray(y, np.float64)
            a = np.clip((y - (y0 - soft)) / soft, 0, 1) * np.clip(((y1 + soft) - y) / soft, 0, 1)
            a = a * a * (3 - 2 * a)
            return 1.0 - 0.7 * k * a
        return f

    def render_hdr(self, t, scale=1.0):
        W, H = int(round(1920 * scale)), int(round(804 * scale))
        cam = self.camera(t, W, H)
        alb = self.sheet.sample(cam)
        nrm = self.sheet.normals(cam)
        emis = np.zeros((H, W, 3), np.float32)
        scorch = np.zeros((H, W, 3), np.float32)
        grid = LightGrid(cam)
        self.leap_light = []
        hush = self.hush_fn(t, H)
        self.draw_web(cam, t, emis, scorch, grid, scale, hush)
        # the paper browns where the fire has passed
        s = np.clip(scorch[..., :1], 0, 1)
        burnt = np.array([0.32, 0.18, 0.09], np.float32)
        alb = alb * (1 - s) + alb * burnt * s
        # map positions of pixels (for the room light)
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float64)
        X, Y = cam.screen_to_map((gx + 0.5).ravel(), (gy + 0.5).ravel())
        X = X.reshape(H, W).astype(np.float32)
        Y = Y.reshape(H, W).astype(np.float32)
        # room light: a hearth beyond the far side of the table, low and raking, flickering
        Lk = np.array([-40.0, 230.0, 110.0])
        dx = Lk[0] - X
        dy = Lk[1] - Y
        dz = np.full_like(X, Lk[2])
        dl = np.sqrt(dx * dx + dy * dy + dz * dz)
        ndl = (nrm[..., 0] * dx + nrm[..., 1] * dy + nrm[..., 2] * dz) / dl
        ndl0 = dz / dl
        rel = np.clip(ndl / np.maximum(ndl0, 1e-3), 0.0, 3.0)
        # the hearth's pool on the table; in the last second it gathers toward the centre of frame
        # (composed in frame: light falls from the upper left early on, and gathers to the centre)
        gath = smooth((t - 2000.0) / 80.0)
        sx0 = W * lerp(0.36, 0.5, gath)
        sy0 = H * lerp(0.18, 0.46, gath)
        sgp = W * lerp(0.36, 0.34, gath)
        px_ = (np.arange(W) + 0.5 - sx0)
        py_ = (np.arange(H) + 0.5 - sy0) * 1.7
        pool = 0.05 + 0.95 * np.exp(-(py_[:, None] ** 2 + px_[None, :] ** 2) / (2 * sgp ** 2)).astype(np.float32)
        flick = 1.0 + 0.035 * math.sin(t * 0.41) + 0.025 * math.sin(t * 1.13 + 1.0) + 0.02 * gnoise(t * 0.2, 0.5, 9)
        # the hearth burns down as the world's own fire takes over
        room = 0.52 * flick * lerp(1.0, 0.42, smooth((t - 2005.0) / 55.0)) * (1.0 - 0.3 * smooth((t - 2060.0) / 27.0))
        room *= lerp(0.8, 1.0, smooth((t - 1935.0) / 30.0))
        E = (room * rel * pool)[..., None] * np.array([1.0, 0.6, 0.3], np.float32)[None, None, :]
        E += np.array([0.014, 0.024, 0.05], np.float32)[None, None, :]             # cool night fill (moonlight)
        # the first beacon's flare floods the peaks around it, then settles to a steady pool
        fo = self.flare(t)
        ox, oy = self.web.P[0]
        r2 = (X - ox) ** 2 + (Y - oy) ** 2
        big = (0.34 * fo) * np.exp(-r2 / (2 * 6.5 ** 2)) + (0.10 + 0.22 * fo) * np.exp(-r2 / (2 * 2.2 ** 2))
        big *= 1.0 - 0.6 * smooth((t - 1990.0) / 40.0)
        E += big[..., None] * np.array([1.0, 0.62, 0.3], np.float32)[None, None, :]
        Ef = grid.irradiance()
        for (Pl, pw) in self.leap_light:
            h = max(Pl[2], 0.3)
            r2 = (X - Pl[0]) ** 2 + (Y - Pl[1]) ** 2
            Ef += (pw * h / (r2 + h * h) ** 1.5)[..., None] * pal()['light'][None, None, :].astype(np.float32)
        if hush is not None:
            Ef *= hush((np.arange(H) + 0.5))[:, None, None].astype(np.float32)
        hdr = alb * (E + Ef) + emis
        return self.dof(hdr, cam, t), cam

    def dof(self, hdr, cam, t):
        """Tilt-shift depth of field while the camera is low over the table (focus: the target)."""
        K = 3.2 * (cam.W / 1920.0) * (1.0 - smooth((t - 1950.0) / 45.0))
        if K < 0.25:
            return hdr
        H, W = hdr.shape[:2]
        ys = (np.arange(H) + 0.5)
        X, Y = cam.screen_to_map(np.full(H, W / 2.0), ys)
        z = (np.column_stack([X, Y, np.zeros(H)]) - cam.pos[None, :]) @ cam.f
        Xc, Yc = cam.screen_to_map(np.array([W / 2.0]), np.array([H * 0.42]))
        zf = float((np.array([Xc[0], Yc[0], 0.0]) - cam.pos) @ cam.f)
        coc = np.clip(np.abs(1.0 - zf / z) * K * 6.0, 0.0, K)        # blur sigma (px) per row
        levels = [0.0, 0.5 * K, K]
        blurs = [hdr] + [cv2.GaussianBlur(hdr, (0, 0), s) for s in levels[1:]]
        out = np.zeros_like(hdr)
        for i in range(len(levels)):
            if i == 0:
                wgt = np.clip(1.0 - coc / levels[1], 0, 1)
            elif i == len(levels) - 1:
                wgt = np.clip((coc - levels[i - 1]) / (levels[i] - levels[i - 1]), 0, 1)
            else:
                wgt = np.clip(1.0 - np.abs(coc - levels[i]) / (levels[i] - levels[i - 1]), 0, 1)
            out += blurs[i] * wgt[:, None, None].astype(np.float32)
        return out

    def exposure(self, t):
        return 1.5

    def render(self, t, scale=1.0):
        hdr, cam = self.render_hdr(t, scale)
        vig = 0.3 + 0.25 * smooth((t - 2050.0) / 37.0)
        return look.finish(hdr, exposure=self.exposure(t), bloom_strength=0.075, bloom_threshold=1.0,
                           vignette_amount=vig)


def review_sheet(frames=None, out=None, tw=640):
    """Contact sheet of delivered frames -> ~/mishamisha/_local_logs/review/map_C.jpg"""
    frames = frames or [1920, 1924, 1936, 1948, 1960, 1972, 1984, 1996, 2008, 2020, 2032, 2044, 2056, 2068, 2078, 2087]
    out = out or os.path.join(geo.ROOT, '..', '_local_logs', 'review', 'map_C.jpg')
    tiles = []
    for f in frames:
        im = cv2.imread(look.frame_path(OUT, f))
        if im is None:
            im = np.full((804, 1920, 3), 40, np.uint8)
        th = int(round(tw * im.shape[0] / im.shape[1]))
        t = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(f), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, str(f), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('start', nargs='?', type=int)
    ap.add_argument('end', nargs='?', type=int)
    ap.add_argument('--frames', default=None)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--tag', default='full')
    ap.add_argument('--out', default=None)
    ap.add_argument('--sheet', action='store_true')
    a = ap.parse_args()
    if a.sheet:
        return review_sheet()
    shot = Shot(a.tag)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        frames = list(range(a.start, a.end + 1))
    out = a.out or (os.path.join(geo.ROOT, '..', '_local_logs', 'review', 'map_tests') if a.test else OUT)
    os.makedirs(out, exist_ok=True)
    for f in frames:
        t0 = time.time()
        img = shot.render(float(f), a.scale)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
