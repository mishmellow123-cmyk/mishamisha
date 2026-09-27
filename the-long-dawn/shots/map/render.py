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
from relay import Relay, T0  # noqa: E402
from noise import gnoise  # noqa: E402

cv2.setNumThreads(2)
OUT = os.path.join(geo.ROOT, 'renders', 'map_C')
FIRST, LAST = 1920, 2087
TEXT_IN, TEXT_OUT = 1960, 2040          # cut C's line ("And hill by hill, the peoples answered.") sits over the map here
POOL = 1.0                              # the answering fires' light on the paper


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def smooth_arr(x):
    x = np.clip(x, 0.0, 1.0)
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
        self.width = width
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

    def __init__(self, cam, cells=420, ps=1.0):
        self.ps = ps                        # pool scale: the pools grow as the camera pulls back
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
        # G holds power per cell: blurring spreads it (the blur kernel sums to one), and dividing by the
        # cell area gives irradiance per square map-degree, whatever the cell size
        E = np.zeros_like(self.G)
        for sig, wt in self.KERNEL:
            s = sig * self.ps / self.cell
            if s < 0.35:
                E += self.G * (wt / self.cell ** 2)
            else:
                E += cv2.GaussianBlur(self.G, (0, 0), s) * (wt / self.cell ** 2)
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
        PAL = dict(gold=lin('#FFA93A'), light=np.array([1.0, 0.62, 0.3]))
    return PAL


# ================================================================== shot ===

class Shot:
    def __init__(self, tag='full'):
        self.sheet = Sheet(tag)
        r = Relay()
        self.P = r.P
        self.t_ign = r.t_ign.copy()
        self.t_ign[0] = T0 - 60.0          # the first beacon already burns (small) as the shot opens
        n = len(self.P)
        self.org = np.arange(n) == 0
        self.size = r.size.copy()
        self.size[0] = 2.0 / 0.45          # the first beacon: 2 map-degrees of flame
        # fire to fire: brightness about 3:1 across the settled fires (a wider spread than the relay's own),
        # and each settles to its own level
        self.gain = 1.0 + (r.gain - 1.0) * 1.5
        self.rest = 0.28 + 0.3 * ((r.phase * 0.618034) % 1.0)
        self.phase = r.phase.copy()
        self.phase[0] = np.random.default_rng(3).random() * 1000.0     # the first beacon's flame, as before
        self._spark_bank(np.random.default_rng(3))

    # ------------------------------------------------------------ camera ---
    def camera(self, t, W, H):
        # log-width keys: close on the Himalaya, then a slow crane up and back to the region, which
        # it never leaves: the fire runs off the frame on every side
        lw = spline(t, [(1905, math.log(42.0)), (1920, math.log(42.0)), (1950, math.log(54.0)),
                        (1985, math.log(66.0)), (2025, math.log(73.5)), (2060, math.log(76.5)),
                        (2100, math.log(79.5))])
        tx = spline(t, [(1905, 76.0), (1920, 76.0), (1950, 73.0), (1985, 69.0), (2025, 66.0), (2060, 65.0), (2100, 64.0)])
        ty = spline(t, [(1905, 25.5), (1920, 25.5), (1950, 24.5), (1985, 25.0), (2025, 26.5), (2060, 27.0), (2100, 27.0)])
        tilt = spline(t, [(1905, 46.0), (1920, 46.0), (1950, 43.0), (1985, 38.0), (2025, 33.0), (2060, 31.0), (2100, 30.5)])
        head = spline(t, [(1905, -7.0), (1920, -7.0), (1950, -6.0), (1985, -4.5), (2025, -3.5), (2060, -3.0), (2100, -3.0)])
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
        """Deterministic sparks: a burst as every fire catches, and the first beacon's stream."""
        n = 7
        nb = len(self.P)
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
        p = pal()
        segs0, segs1, I = [], [], []

        def fly(P0, v, age):
            # rise, drift, slow down (air drag); a little curl
            drag = (1.0 - np.exp(-age / 6.0)) * 6.0
            pos = P0 + v * drag[:, None]
            pos[:, 2] += 0.004 * age * age / (1.0 + 0.1 * age)
            return pos

        # a burst as each fire catches
        age = t - (self.t_ign[self.bi] + self.bdt)
        m = (age >= 0) & (age < self.blife) & (~self.org[self.bi])
        if m.any():
            P0 = np.column_stack([self.P[self.bi[m]], np.full(m.sum(), 0.15)])
            a = age[m]
            segs0.append(fly(P0, self.bv[m], np.maximum(a - 0.9, 0)))
            segs1.append(fly(P0, self.bv[m], a))
            I.append(4.0 * float(np.clip(54.0 / cam.width, 0.35, 1.0)) * (1 - a / self.blife[m]) ** 1.5)
        # the first beacon: a steady stream, and the fountain of the flare
        O = np.array([self.P[0, 0], self.P[0, 1], 0.5])
        for tt, life, v, k in ((self.oi_t, self.oi_life, self.oi_v, 3.5), (T0 + self.of_dt, self.of_life, self.of_v, 7.0)):
            age = t - tt
            m = (age >= 0) & (age < life)
            if m.any():
                a = age[m]
                P0 = np.repeat(O[None, :], m.sum(), 0)
                segs0.append(fly(P0, v[m], np.maximum(a - 0.9, 0)))
                segs1.append(fly(P0, v[m], a))
                I.append(k * (1 - a / life[m]) ** 1.3)
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

    # ------------------------------------------------------------- fires ---
    def draw_fires(self, cam, t, emis, scorch, grid, scale, hush):
        """Every lit fire: a living flame on its summit that flares as it catches and then settles,
        its own pool of light on the paper, and the singe under it. No threads, no dots."""
        p = pal()
        lit = np.where(self.t_ign <= t)[0]
        if not len(lit):
            return
        age = t - self.t_ign[lit]
        org = self.org[lit]
        size = self.size[lit]
        ph = self.phase[lit]
        catch = smooth_arr(age / 2.5)
        flare = np.exp(-age / 6.0) * np.clip(age / 1.2, 0, 1)
        fo = self.flare(t)
        flare = np.where(org, 1.5 * fo, flare)
        # the newest fires flare; the older ones settle to a lower, steady burn
        rest = self.rest[lit]
        settle = np.where(org, 1.0, rest + (1.0 - rest) * np.exp(-age / 24.0))
        fl = 1.0 + 0.12 * np.sin(t * 0.5 + ph) + 0.07 * np.sin(t * 1.3 + 2 * ph)
        body = catch * settle * fl
        hgt = np.where(org, 0.45, 0.8) * size * (body + 0.8 * flare)
        P = np.column_stack([self.P[lit], np.zeros(len(lit))])
        uv, z = cam.project(P)
        sc = cam.F / z * scale
        up, upl = cam.up2d(P)
        # physical height on the paper, but never smaller on screen than a readable flame
        floor = 19.0 * scale * np.minimum(size, 2.0) ** 0.8 * (0.4 * catch + 0.6 * body + 0.7 * flare)
        hp = np.maximum(hgt * upl * scale, floor)
        mul = np.ones(len(lit)) if hush is None else hush(uv[:, 1])
        W, H = cam.W, cam.H
        vis = (uv[:, 0] > -80) & (uv[:, 0] < W + 80) & (uv[:, 1] > -80) & (uv[:, 1] < H + 120)
        for i in np.where(vis & (hp > 0.3))[0]:
            if org[i]:
                g, cool = 1.5 * mul[i], 0.0
            else:
                g = 1.15 * self.gain[lit[i]] * (0.55 + 0.45 * settle[i]) * (1.0 + 1.2 * flare[i]) * catch[i] * mul[i]
                cool = (1.0 - settle[i]) / 0.7          # burned down: lower and redder
            fire.flame(emis, uv[i, 0], uv[i, 1], hp[i], up[i, 0], up[i, 1], t + ph[i],
                       int(lit[i]) * 13 + 1, g, 1.3 if org[i] else 1.6, cool)
        # the first beacon's own soft glow (as it always had)
        o = np.where(org)[0]
        if len(o):
            fire.splat_points(emis, uv[o, 0].copy(), uv[o, 1].copy(), np.outer(0.25 * (1 + fo) * mul[o], p['gold']),
                              np.maximum(0.45 * sc[o], 1.0))
        # the singe under every fire
        fire.splat_points(scorch, uv[:, 0].copy(), uv[:, 1].copy(), np.outer(0.3 * catch, np.ones(3)),
                          np.maximum(0.2 * sc, 0.6))
        # each fire's own pool of light on the paper (the first beacon's pool is its flare light, below)
        # a fire throws its light wide as it catches; settled, it lights only the summit it stands on
        pw = size * self.gain[lit] * (0.9 * body + 2.2 * flare)
        a = ~org
        grid.add(self.P[lit[a], 0], self.P[lit[a], 1], np.outer(POOL * pw[a], p['light']))

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
        grid = LightGrid(cam, ps=float(np.clip((cam.width / 54.0) ** 0.5, 1.0, 1.2)))
        hush = self.hush_fn(t, H)
        self.draw_fires(cam, t, emis, scorch, grid, scale, hush)
        self.draw_sparks(cam, t, emis, scale, hush)
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
        # (it burns down further than it used to: at the end the map is warm only where fire touches it)
        room = 0.52 * flick * lerp(1.0, 0.17, smooth((t - 1995.0) / 65.0)) * (1.0 - 0.3 * smooth((t - 2060.0) / 27.0))
        room *= lerp(0.8, 1.0, smooth((t - 1935.0) / 30.0))
        room *= self.room_gain(t)
        E = (room * rel * pool)[..., None] * np.array([1.0, 0.6, 0.3], np.float32)[None, None, :]
        moon = lerp(1.0, 4.4, smooth((t - 1995.0) / 60.0)) * self.moon_gain(t)
        E += moon * np.array([0.014, 0.024, 0.05], np.float32)[None, None, :]      # cool night fill (moonlight)
        # the first beacon's flare floods the peaks around it, then settles to a steady pool
        fo = self.flare(t)
        ox, oy = self.P[0]
        r2 = (X - ox) ** 2 + (Y - oy) ** 2
        big = (0.34 * fo) * np.exp(-r2 / (2 * 6.5 ** 2)) + (0.10 + 0.22 * fo) * np.exp(-r2 / (2 * 2.2 ** 2))
        big *= 1.1 * (1.0 - 0.4 * smooth((t - 1990.0) / 40.0))     # (the first fire stays the source)
        E += big[..., None] * np.array([1.0, 0.62, 0.3], np.float32)[None, None, :]
        Ef = grid.irradiance()
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

    def room_gain(self, t):
        """(hook) the hearth beyond the table, relative to its keyed burn-down"""
        return 1.0

    def moon_gain(self, t):
        """(hook) the cool night fill, relative to its keys"""
        return 1.0

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
