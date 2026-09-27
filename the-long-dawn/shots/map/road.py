"""C18 · THE MAP ANSWERS · THE ROAD (P4) on the locked bar map: C frames 4160-4480 -> renders/map_C/.

map_C revision 3 (the relay of fires along the ranges; `render.py`, `relay.py`) retimed from its v2 168 frames onto
C18's four bars, with THE ROAD (REV 1): one slow dotted route walks from her beacon's glyph (the first fire, on the
high Himalaya) west along the ranges to a ring of stones drawn where the great ranges meet (the Pamir knot, a
crossroads and no one's capital), and reaches it on bar 56 b3 (4440) while the beacons race. The camera ends
centred on the drawn ring, a little pushed in, for the match to THE COUNCIL's ring of stones (C19, 4480).

    python road.py frames --frames 4160-4479            # delivery (renders/map_C, C numbering)
    python road.py frames --frames 4170,4300,4440,4479 --scale 0.5 --out DIR
    python road.py x1 --center 960,300 --frames 4150-4185 --out renders/x1_map_C
        # X1 onto the map: the seventh beacon's bloom burns through the drawn sky. Writes the burn's glow (RGB) and
        # a keep-matte (1 = the ink run still shows); EDIT: out = run*keep + map*(1-keep) + glow

T10 (4190-4320) sits over the lower third, which the relay hushes as before.
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
import render as MAP  # noqa: E402
import pen  # noqa: E402
import burn as BURN  # noqa: E402

F0, F1 = 4160, 4480
ARRIVE = 4440                                   # bar 56 b3: the road reaches the ring
T10 = (4190, 4320)
RING_LL = (38.4, 73.2)                          # the Pamir knot (lat, lon)
ROUTE_LL = [(27.99, 86.93), (28.3, 84.9), (28.9, 82.6), (29.9, 80.6), (31.2, 78.7), (32.6, 77.2), (34.1, 76.0),
            (35.4, 75.0), (36.5, 74.5), (37.6, 73.8), (38.25, 73.3)]


def v2(f):
    """C frame -> map_C's v2 clock (1920..2087)."""
    return 1920.0 + (f - F0) * (167.0 / (F1 - 1 - F0))


def smooth(x):
    return MAP.smooth(x)


class RoadShot(MAP.Shot):
    def __init__(self, tag='full'):
        super().__init__(tag)
        rx, ry = geo.ll2map(*RING_LL)
        self.ring = np.array([float(rx), float(ry)])
        pts = np.array([geo.ll2map(la, lo) for la, lo in ROUTE_LL], np.float64)
        pts[0] = self.P[0]                                  # from her beacon's glyph itself
        c = pen.catmull(pts, 10)
        c = pen.resample(c, 0.02)
        rng = np.random.default_rng(17)
        n = pen.normals(c)
        s = pen.arclen(c)
        c = c + n * (0.12 * np.sin(s * 1.7) + 0.05 * np.sin(s * 4.3))[:, None]     # a road bends with the land
        self.route = c
        self.rs = pen.arclen(c)
        # dashes along it, each a short pull of the pen
        L = self.rs[-1]
        self.dash = []
        q = 0.0
        while q < L - 0.2:
            a, b = q, min(q + 0.2 + 0.04 * rng.random(), L)
            self.dash.append((a, b))
            q = b + 0.14 + 0.03 * rng.random()
        # the ring of stones: eleven stones round a flat one
        self.stones = []
        for k in range(11):
            a = k * 2 * math.pi / 11 + rng.normal(0, 0.06)
            r = 0.52 + rng.normal(0, 0.03)
            self.stones.append((self.ring[0] + r * math.cos(a), self.ring[1] + 0.9 * r * math.sin(a),
                                0.06 + 0.02 * rng.random(), rng.uniform(0, math.pi)))
        self.f = F0

    def calm(self, t):
        a, b = v2(T10[0]), v2(T10[1])
        return smooth((t - (a - 6)) / 6.0) * (1 - smooth((t - b - 1) / 6.0))

    def camera(self, t, W, H):
        # map_C rev 3's crane, ending centred on the ring of stones and pushed in a little, for the match
        k = smooth((t - 2010.0) / 77.0)
        tx, ty = self._tx(t), self._ty(t)
        tx += (self.ring[0] - tx) * k
        ty += (self.ring[1] - ty) * k
        w = self._width(t) * (1.0 - 0.72 * smooth((t - 2040.0) / 47.0) ** 1.3)
        return MAP.Cam(tx, ty, w, self._tilt(t), self._head(t), W, H)

    # map_C rev 3's own keys (v2 clock), so the push can be layered on them
    def _width(self, t):
        return math.exp(MAP.spline(t, [(1905, math.log(42.0)), (1920, math.log(42.0)), (1950, math.log(54.0)),
                                       (1985, math.log(66.0)), (2025, math.log(73.5)), (2060, math.log(76.5)),
                                       (2100, math.log(79.5))]))

    def _tx(self, t):
        return MAP.spline(t, [(1905, 76.0), (1920, 76.0), (1950, 73.0), (1985, 69.0), (2025, 66.0), (2060, 65.0), (2100, 64.0)])

    def _ty(self, t):
        return MAP.spline(t, [(1905, 25.5), (1920, 25.5), (1950, 24.5), (1985, 25.0), (2025, 26.5), (2060, 27.0), (2100, 27.0)])

    def _tilt(self, t):
        return MAP.spline(t, [(1905, 46.0), (1920, 46.0), (1950, 43.0), (1985, 38.0), (2025, 33.0), (2060, 31.0), (2100, 30.5)])

    def _head(self, t):
        return MAP.spline(t, [(1905, -7.0), (1920, -7.0), (1950, -6.0), (1985, -4.5), (2025, -3.5), (2060, -3.0), (2100, -3.0)])

    # -------------------------------------------------------------- ink ---
    def ink_layer(self, cam, f):
        """Coverage of the road's dashes drawn by frame f and of the ring of stones (screen, 0..1)."""
        H, W = cam.H, cam.W
        C = np.zeros((H, W), np.float32)
        Wt = np.zeros((H, W), np.float32)
        # the road walks at a steady, slow pace: from the map's reveal to bar 56 b3
        prog = np.clip((f - (F0 + 8)) / (ARRIVE - (F0 + 8)), 0.0, 1.0)
        reach = prog * self.rs[-1]
        sc = float(cam.scale(np.array([[self.ring[0], self.ring[1], 0.0]]))[0])      # px per map degree
        rad = max(0.035 * sc, 0.75)
        for (a, b) in self.dash:
            if a >= reach:
                break
            bb = min(b, reach)
            m = (self.rs >= a) & (self.rs <= bb)
            seg = self.route[m]
            if len(seg) < 2:
                continue
            uv, _ = cam.project(np.column_stack([seg, np.zeros(len(seg))]))
            fresh = np.clip(1.0 - (reach - bb) / 0.6, 0, 1)                  # the newest dash is still wet
            for k in range(len(uv) - 1):
                pen._seg(C, Wt, 0, H, uv[k, 0], uv[k, 1], uv[k + 1, 0], uv[k + 1, 1], rad, rad, 0.9, 0.9, fresh, fresh)
        for (x, y, r, ang) in self.stones:
            uv, _ = cam.project(np.array([[x - r * math.cos(ang), y - r * math.sin(ang), 0.0],
                                          [x + r * math.cos(ang), y + r * math.sin(ang), 0.0]]))
            rr = max(0.45 * r * sc, 0.9)
            pen._seg(C, Wt, 0, H, uv[0, 0], uv[0, 1], uv[1, 0], uv[1, 1], rr, rr, 0.92, 0.92, 0.0, 0.0)
        uv, _ = cam.project(np.array([[self.ring[0] - 0.1, self.ring[1], 0.0], [self.ring[0] + 0.1, self.ring[1], 0.0]]))
        rr = max(0.07 * sc, 1.0)
        pen._seg(C, Wt, 0, H, uv[0, 0], uv[0, 1], uv[1, 0], uv[1, 1], rr, rr, 0.85, 0.85, 0.0, 0.0)
        return C, Wt

    def render_c(self, f, scale=1.0):
        t = v2(f)
        self.f = f
        orig = self.sheet.sample

        def sample(cam, bias=-0.25):
            alb = orig(cam, bias)
            C, Wt = self.ink_layer(cam, f)
            k = np.clip(C, 0, 1)[..., None]
            ink = np.array([0.02, 0.012, 0.008], np.float32)
            return alb * (1 - k) + ink * k
        self.sheet.sample = sample
        try:
            hdr, cam = self.render_hdr(t, scale)
        finally:
            self.sheet.sample = orig
        vig = 0.3 + 0.25 * smooth((t - 2050.0) / 37.0)
        return look.finish(hdr, exposure=self.exposure(t), bloom_strength=0.075, bloom_threshold=1.0, vignette_amount=vig)


# ================================================================ x1 matte ===

def x1_screen(center, frames, out, W=1920, H=804, t_open=None, span=1.1):
    """The burn-through onto the map as a screen-space layer: glow (premultiplied RGB) and keep-matte."""
    cx, cy = center
    s = 60.0                                       # screen px per burn unit
    first = frames[0]
    t0 = (first + 6) / 24.0 if t_open is None else t_open
    bf = BURN.params((cx / s, cy / s), t_start=t0, speed=14.0 / span ** 1.5, p=1.5, amp=0.45, freq=0.35, seed=12,
                     brown=2.2, char=0.3, edge=0.05, lead=0.25)
    os.makedirs(out, exist_ok=True)
    os.makedirs(out.rstrip('/') + '_matte', exist_ok=True)
    ys, xs = np.mgrid[0:H, 0:W]
    for f in frames:
        t = f / 24.0
        glow, keep = _x1_field(bf, xs.astype(np.float64) / s, ys.astype(np.float64) / s, t)
        rgb = look.finish(glow, exposure=1.0, bloom_strength=0.05, bloom_threshold=1.0, vignette_amount=0.0, lift=0.0)
        look.save_png(look.frame_path(out, f), rgb)
        look.save_png(look.frame_path(out.rstrip('/') + '_matte', f), np.repeat(keep[..., None], 3, -1))
        print('x1', f)


def _x1_field(bf, U, V, t):
    H, W = U.shape
    glow = np.zeros((H, W, 3), np.float32)
    keep = np.ones((H, W), np.float32)
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def run(U, V, glow, keep, bf, t):
        for i in prange(U.shape[0]):
            for j in range(U.shape[1]):
                brown, char, hole, edge, pool = BURN.field(bf, U[i, j], V[i, j], t)
                keep[i, j] = (1.0 - hole) * (1.0 - 0.9 * char)
                glow[i, j, 0] = edge * 4.2
                glow[i, j, 1] = edge * 1.25
                glow[i, j, 2] = edge * 0.18
    run(U, V, glow, keep, bf, t)
    return glow, keep


def frames_of(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['frames', 'x1'])
    ap.add_argument('--frames', default='')
    ap.add_argument('--out', default=os.path.join(geo.ROOT, 'renders', 'map_C'))
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--center', default='960,300')
    a = ap.parse_args()
    fr = frames_of(a.frames)
    if a.what == 'x1':
        cx, cy = (float(v) for v in a.center.split(','))
        x1_screen((cx, cy), fr, a.out)
        return
    sh = RoadShot('full')
    for f in fr:
        t0 = time.time()
        img = sh.render_c(f, a.scale)
        look.save_png(look.frame_path(a.out, f), img)
        print(f, '%.1fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
