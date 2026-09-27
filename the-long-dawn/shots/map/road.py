"""C18 · THE MAP ANSWERS · THE ROAD (P4) on the locked bar map: C frames 4160-4480 -> renders/map_C/.
MAP-L2: drawn on the INVENTED world (terra.py); no real place is on the sheet.

The map opens on her range (the ink Run's range): her beacon burns on its high knee and the Run's seven fires stand
lit along its east arm, the seventh still blooming where C17's burn-through opens the sheet. The relay (`relay.py`)
carries the fire on along the ranges, down the great rivers and along the coasts, off the frame on every side. And
THE ROAD (REV 1): one fine pen line leaves her beacon's glyph, comes down off the range, crosses the valley and its
river, and climbs to the ring of stones on the High Moor at the heart of the map (a high place that belongs to no
one), reaching it on bar 56 b3 (4440) while the beacons race. The camera cranes up and back from the chain, west
over the land to the great western sea, and ends centred on the drawn ring, a little pushed in, for the match to
THE COUNCIL's ring of stones (C19, 4480).

    python road.py frames --frames 4160-4479            # delivery (renders/map_C, C numbering)
    python road.py frames --frames 4170,4300,4440,4479 --scale 0.5 --out DIR
    python road.py x1 --center 1130,485 --frames 4150-4185 --out renders/x1_map_C
        # X1 onto the map: the seventh beacon's bloom (RUN-C's C17 seventh at local 310-319 stands at 1130,485)
        # burns through the drawn sky. Writes the burn's glow (RGB) and a keep-matte (1 = the ink run still shows);
        # EDIT: out = run*keep + map*(1-keep) + glow

T10 (4190-4320) sits over the lower third, which the relay hushes as before.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import look  # noqa: E402
import geo  # noqa: E402
import render as MAP  # noqa: E402
import pen  # noqa: E402
import burn as BURN  # noqa: E402
import terra  # noqa: E402

F0, F1 = 4160, 4480
ARRIVE = 4440                                   # bar 56 b3: the road reaches the ring
T10 = (4190, 4320)
SEVENTH_SCREEN = (1130.0, 485.0)                 # C17's seventh beacon on screen as the burn-through opens
OPEN_B = (560.0, 360.0)                          # her beacon on screen at the open (the Road leaves it up-left)
_OPEN = None


def opening_key():
    """The opening framing (v2 1905-1920), solved from the relay: the Run's seventh fire exactly where C17's
    burn-through opens (SEVENTH_SCREEN) and her beacon at OPEN_B, at the keyed tilt. Returns (tx, ty, w, head)."""
    global _OPEN
    if _OPEN is not None:
        return _OPEN or None
    from scipy.optimize import least_squares
    import relay
    try:
        r = relay.Relay()
    except Exception as e:                          # before the glyphs exist (the design sketches)
        print('opening_key: no relay yet:', e)
        _OPEN = False
        return None
    B = r.P[0]
    S7 = r.P[r.chain[-1]]
    tilt = CAM_KEYS[0][4]

    def res(p):
        tx, ty, lw, hd = p
        cam = MAP.Cam(tx, ty, math.exp(lw), tilt, hd, 1920, 804)
        uv, _ = cam.project(np.array([[S7[0], S7[1], 0.0], [B[0], B[1], 0.0]]))
        return np.concatenate([(uv[0] - SEVENTH_SCREEN) / 100.0, (uv[1] - OPEN_B) / 100.0])
    k0 = CAM_KEYS[0]
    sol = least_squares(res, [k0[1], k0[2], math.log(k0[3]), k0[5]])
    tx, ty, lw, hd = sol.x
    _OPEN = (float(tx), float(ty), float(math.exp(lw)), float(hd))
    return _OPEN


def _key(t, col, log=False):
    keys = list(CAM_KEYS)
    ok = opening_key()
    if ok:
        tx, ty, w, hd = ok
        keys[0] = (keys[0][0], tx, ty, w, keys[0][4], hd)
        keys[1] = (keys[1][0], tx, ty, w, keys[1][4], hd)
    ks = [(k[0], math.log(k[col]) if log else k[col]) for k in keys]
    v = MAP.spline(t, ks)
    return math.exp(v) if log else v


def camera_at(t, ring, W=1920, H=804):
    """The crane (v2 clock): from the Run's chain on her range, up and back and west over the land to the great
    western sea; from 2010 it drifts to the ring of stones, and from 2040 it pushes in on it for the match to C19."""
    k = smooth((t - 2010.0) / 77.0)
    tx, ty = _key(t, 1), _key(t, 2)
    tx += (ring[0] - tx) * k
    ty += (ring[1] - ty) * k
    w = _key(t, 3, True) * (1.0 - 0.72 * smooth((t - 2040.0) / 47.0) ** 1.3)
    return MAP.Cam(tx, ty, w, _key(t, 4), _key(t, 5), W, H)


def v2(f):
    """C frame -> map_C's v2 clock (1920..2087)."""
    return 1920.0 + (f - F0) * (167.0 / (F1 - 1 - F0))


def smooth(x):
    return MAP.smooth(x)


def smooth_arr(x):
    return MAP.smooth_arr(x)


class RoadShot(MAP.Shot):
    def __init__(self, tag='full'):
        super().__init__(tag)
        self.ring = np.array(terra.RING, np.float64)
        c = road_path(self.P[0], self.ring)                 # from her beacon's glyph itself
        rng = np.random.default_rng(17)
        n = pen.normals(c)
        s = pen.arclen(c)
        taper = np.clip(s / 0.8, 0, 1) * np.clip((s[-1] - s) / 0.8, 0, 1)
        c = c + n * ((0.08 * np.sin(s * 1.7) + 0.03 * np.sin(s * 4.3)) * taper)[:, None]   # a road bends with the land
        self.route = c
        self.rs = pen.arclen(c)
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
        return camera_at(t, self.ring, W, H)

    # -------------------------------------------------------------- ink ---
    def ink_layer(self, cam, f):
        """Coverage of the road drawn by frame f and of the ring of stones (screen, 0..1)."""
        H, W = cam.H, cam.W
        C = np.zeros((H, W), np.float32)
        Wt = np.zeros((H, W), np.float32)
        # the road walks at a steady, slow pace: from the map's reveal to bar 56 b3
        prog = np.clip((f - (F0 + 8)) / (ARRIVE - (F0 + 8)), 0.0, 1.0)
        reach = prog * self.rs[-1]
        sc = float(cam.scale(np.array([[self.ring[0], self.ring[1], 0.0]]))[0])      # px per map degree
        self.pen_line(C, Wt, cam, self.route, self.rs, reach, sc, seed=3)
        self.stones_layer(C, cam, sc)
        return C, Wt

    def stones_layer(self, C, cam, sc):
        """The ring of stones as drawn stones: each a small irregular outline with a wash of ink in it and its
        shadow side hatched, round one flat stone."""
        import cv2
        H, W = cam.H, cam.W
        wash = np.zeros((H, W), np.float32)
        dry = np.zeros((H, W), np.float32)
        rw = max(0.012 * sc, 0.6)
        stones = list(self.stones) + [(self.ring[0], self.ring[1], 0.13, 0.3)]
        for q, (x, y, r, ang) in enumerate(stones):
            rng = np.random.default_rng(100 + q)
            n = 9
            th = np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.uniform(0, 1)
            rr = r * (1.0 + rng.normal(0, 0.1, n))
            ex = 1.35 if q < len(self.stones) else 1.6
            px = x + rr * ex * np.cos(th) * math.cos(ang) - rr * np.sin(th) * math.sin(ang)
            py = y + rr * ex * np.cos(th) * math.sin(ang) + rr * np.sin(th) * math.cos(ang)
            uv, _ = cam.project(np.column_stack([px, py, np.zeros(n)]))
            O = pen.catmull(np.vstack([uv, uv[:1]]), 4)
            cv2.fillPoly(wash, [np.round(O * 16).astype(np.int32)], 0.3, lineType=cv2.LINE_AA, shift=4)
            for k in range(len(O) - 1):
                w = rw * (1.0 + 0.5 * max(0.0, math.sin(k / (len(O) - 1) * 2 * math.pi + 1.0)))   # heavier below
                pen._seg(C, dry, 0, H, O[k, 0], O[k, 1], O[k + 1, 0], O[k + 1, 1], w, w, 0.92, 0.92, 0.0, 0.0)
        np.maximum(C, wash, out=C)

    def pen_line(self, C, Wt, cam, route, rs, reach, sc, seed=0, weight=1.0, start=0.0, r_in=None):
        """THE ROAD (H5): one fine pen line drawn from `start` up to `reach` (map degrees of arc), its weight
        swelling and thinning with the hand, the last half-degree still wet, a bead of ink at the nib while it
        moves. r_in: (centre, r0, r1) fades the line in from r0 to r1 about a centre (out of the ring)."""
        H = cam.H
        if reach <= start:
            return
        m = (rs <= reach) & (rs >= start)
        seg = route[m]
        ss = rs[m]
        if reach < rs[-1]:
            tip = np.array([np.interp(reach, rs, route[:, 0]), np.interp(reach, rs, route[:, 1])])
            seg = np.vstack([seg, tip])
            ss = np.concatenate([ss, [reach]])
        if len(seg) < 2:
            return
        uv, _ = cam.project(np.column_stack([seg, np.zeros(len(seg))]))
        ph = seed * 1.7
        press = 1.0 + 0.22 * np.sin(ss * 2.3 + ph) + 0.12 * np.sin(ss * 7.1 + 2 * ph)
        rad = np.maximum(0.024 * sc * weight * press, 0.6)
        wet = np.clip(1.0 - (reach - ss) / 0.5, 0.0, 1.0)
        den = 0.9 + 0.08 * wet
        if r_in is not None:
            c0, r0, r1 = r_in
            d = np.hypot(seg[:, 0] - c0[0], seg[:, 1] - c0[1])
            den = den * np.clip((d - r0) / (r1 - r0), 0, 1)
        for k in range(len(uv) - 1):
            if den[k] <= 0 and den[k + 1] <= 0:
                continue
            pen._seg(C, Wt, 0, H, uv[k, 0], uv[k, 1], uv[k + 1, 0], uv[k + 1, 1], rad[k], rad[k + 1],
                     den[k], den[k + 1], wet[k], wet[k + 1])
        if reach < rs[-1] and den[-1] > 0:
            r = 1.4 * rad[-1]
            pen._seg(C, Wt, 0, H, uv[-1, 0], uv[-1, 1], uv[-1, 0] + 1e-3, uv[-1, 1], r, r, den[-1], den[-1], 1.0, 1.0)

    # ----------------------------------------------------- the beacons ---
    def glyph_frame(self, cam, t):
        """Screen placement of every lit fire's glyph: (index, base x, y, up x, y, height px, catch)."""
        lit = np.where(self.t_ign <= t)[0]
        if not len(lit):
            return []
        catch = smooth_arr((t - self.t_ign[lit]) / 2.5)
        P = np.column_stack([self.P[lit], np.zeros(len(lit))])
        uv, z = cam.project(P)
        up, _ = cam.up2d(P)
        sc = cam.F / z
        out = []
        for i in range(len(lit)):
            x0, y0 = uv[i]
            if x0 < -60 or x0 > cam.W + 60 or y0 < -80 or y0 > cam.H + 80 or catch[i] <= 0.02:
                continue
            org = bool(self.org[lit[i]])
            hp = max((0.95 if org else 0.55) * min(self.size[lit[i]], 2.2) * sc[i], 9.0 if org else 6.5)
            out.append((lit[i], x0, y0, up[i, 0], up[i, 1], hp, catch[i], sc[i], org))
        return out

    def flame_glyph(self, C, Wt, gold, x0, y0, ux, uy, hp, catch, sc, ph, heavy=1.0, stack=True):
        """One flame drawn in the pen: a tall S-tongue between two shorter ones, outlined at the terrain's line
        weight (drawn on as it catches), shell gold laid inside; over a low stack of two strokes."""
        import cv2
        H = C.shape[0]
        rx, ry = -uy, ux
        lean = 0.1 * math.sin(ph)
        wd = 0.34 * hp

        def P2(a, b):
            return (x0 + a * wd * rx + b * hp * ux, y0 + a * wd * ry + b * hp * uy)
        ring = [(-0.85, 0.0), (-1.0, 0.22), (-0.78, 0.48), (-0.9, 0.7), (-0.52, 0.5), (-0.36, 0.72),
                (-0.02 + lean, 1.0), (0.12 + lean, 0.78), (0.42, 0.55), (0.82, 0.74), (0.8, 0.42),
                (1.0, 0.2), (0.85, 0.0)]
        O = pen.catmull(np.array([P2(a, b) for a, b in ring]), 5)
        n_draw = max(2, int(round(len(O) * min(1.0, catch * 1.4))))
        rr = max(0.016 * sc * heavy, 0.65)
        for k in range(n_draw - 1):
            pen._seg(C, Wt, 0, H, O[k, 0], O[k, 1], O[k + 1, 0], O[k + 1, 1], rr, rr, 0.9, 0.9, 0.0, 0.0)
        if stack:
            for a, b, c, d in ((-0.95, -0.02, 0.95, -0.02), (-0.55, -0.17, 0.6, -0.17)):
                A_, B_ = P2(a, b), P2(c, d)
                pen._seg(C, Wt, 0, H, A_[0], A_[1], B_[0], B_[1], rr * 0.9, rr * 0.9, 0.9, 0.9, 0.0, 0.0)
        else:                                                     # a hearth: the curb of its stones
            Q = pen.catmull(np.array([P2(-1.25, 0.12), P2(-0.7, -0.08), P2(0.0, -0.13), P2(0.7, -0.08), P2(1.25, 0.12)]), 5)
            for k in range(min(len(Q), n_draw) - 1):
                pen._seg(C, Wt, 0, H, Q[k, 0], Q[k, 1], Q[k + 1, 0], Q[k + 1, 1], rr, rr, 0.9, 0.9, 0.0, 0.0)
        if catch > 0.3:
            g = min(1.0, (catch - 0.3) / 0.5)
            cv2.fillPoly(gold, [np.round(O * 16).astype(np.int32)], g, lineType=cv2.LINE_AA, shift=4)

    def beacon_glyphs(self, cam, t):
        """C's beacons hand-inked (H5): every lit fire a flame drawn in the pen with shell gold laid inside it.
        Returns (ink coverage, gold wash) in screen space."""
        H, W = cam.H, cam.W
        C = np.zeros((H, W), np.float32)
        Wt = np.zeros((H, W), np.float32)
        gold = np.zeros((H, W), np.float32)
        for (i, x0, y0, ux, uy, hp, catch, sc, org) in self.glyph_frame(cam, t):
            self.flame_glyph(C, Wt, gold, x0, y0, ux, uy, hp, catch, sc, self.phase[i], heavy=1.3 if org else 1.0)
        return C, np.clip(gold, 0, 1)

    def beacon_fade(self, t):
        return 1.0

    def draw_fires(self, cam, t, emis, scorch, grid, scale, hush):
        """The living fire inside each drawn flame (H5, no emoji halo): the flame is the glyph's own size and
        stands inside its outline, catching bright and settling low; the singe under it; a small pool."""
        p = MAP.pal()
        gl = self.glyph_frame(cam, t)
        if not gl:
            return
        fade = self.beacon_fade(t)
        xs, ys, pw, sg = [], [], [], []
        for (i, x0, y0, ux, uy, hp, catch, sc, org) in gl:
            age = t - self.t_ign[i]
            flare = math.exp(-age / 6.0) * min(age / 1.2, 1.0)
            if org:
                flare = 1.5 * self.flare(t)
            rest = self.rest[i]
            settle = 1.0 if org else rest + (1.0 - rest) * math.exp(-age / 24.0)
            fl = 1.0 + 0.12 * math.sin(t * 0.5 + self.phase[i]) + 0.07 * math.sin(t * 1.3 + 2 * self.phase[i])
            body = catch * (0.55 + 0.45 * settle) * fl
            g = (1.2 if org else 0.95 * self.gain[i]) * body * (1.0 + 0.8 * flare) * fade
            if hush is not None:
                g *= float(hush(np.array([y0]))[0])
            h = hp * (0.72 + 0.12 * flare) * (0.6 + 0.4 * catch)
            MAP.fire.flame(emis, x0 + 0.02 * hp * uy, y0 - 0.02 * hp * ux, h, ux, uy, t + self.phase[i],
                           int(i) * 13 + 1, g, 1.25 if org else 1.35, (1.0 - settle) / 0.7 if not org else 0.0)
            xs.append(x0)
            ys.append(y0)
            sg.append(max(0.2 * sc * scale, 0.6))
            pw.append((i, 0.3 * catch))
        idx = np.array([q[0] for q in pw])
        MAP.fire.splat_points(scorch, np.array(xs), np.array(ys), np.outer([q[1] for q in pw], np.ones(3)), np.array(sg))
        lit = idx[~self.org[idx]]
        age = t - self.t_ign[lit]
        body = smooth_arr(age / 2.5) * (0.55 + 0.45 * self.rest[lit])
        flare = np.exp(-age / 6.0) * np.clip(age / 1.2, 0, 1)
        pw_ = self.size[lit] * self.gain[lit] * (0.9 * body + 2.2 * flare) * fade
        grid.add(self.P[lit, 0], self.P[lit, 1], np.outer(POOL_C * pw_, p['light']))

    # ------------------------------------------------------------ frame ---
    def clock(self, f):
        return v2(f)

    def render_c(self, f, scale=1.0):
        t = self.clock(f)
        self.f = f
        orig = self.sheet.sample

        def sample(cam, bias=-0.25):
            alb = orig(cam, bias)
            Cb, gold = self.beacon_glyphs(cam, t)
            C, Wt = self.ink_layer(cam, f)
            C2, gold2 = self.extra_glyphs(cam, t, f)
            g = np.clip(0.85 * np.maximum(gold, gold2), 0, 1)[..., None]
            alb = alb * (1 - g) + GOLD_ALB * g
            k = np.clip(np.maximum(np.maximum(C, Cb), C2), 0, 1)[..., None]
            return alb * (1 - k) + INK * k
        self.sheet.sample = sample
        try:
            hdr, cam = self.render_hdr(t, scale)
        finally:
            self.sheet.sample = orig
        return self.finish(hdr, t, f)

    def extra_glyphs(self, cam, t, f):
        return 0.0, 0.0

    def finish(self, hdr, t, f):
        vig = 0.3 + 0.25 * smooth((t - 2050.0) / 37.0)
        return look.finish(hdr, exposure=self.exposure(t), bloom_strength=0.06, bloom_threshold=1.0, vignette_amount=vig)


INK = np.array([0.02, 0.012, 0.008], np.float32)
GOLD_ALB = np.array([0.95, 0.66, 0.24], np.float32)       # shell gold laid in the glyph
POOL_C = 0.4                                             # the fires' pools on the paper (rev 3 had 1.0)
# (v2 frame, target x, target y, width, tilt, heading): open on the chain, crane up and back and west
CAM_KEYS = [(1905, 20.5, 3.2, 44.0, 46.0, -6.0), (1920, 20.5, 3.2, 44.0, 46.0, -6.0), (1950, 16.0, 6.5, 54.0, 43.0, -5.0),
            (1985, 9.0, 10.0, 66.0, 39.0, -4.0), (2025, 4.5, 12.0, 74.0, 37.0, -3.5), (2060, 3.5, 12.5, 77.0, 32.0, -3.0),
            (2100, 2.5, 12.5, 80.0, 25.0, -3.0)]


# ============================================================ C23 bar 71 ===

F71 = (5600, 5680)          # bar 71: the burn-through to the map (bar 70 is ACCORD's council plate)
BREATH = 5673               # the 275 ms breath before the sunrise (C24, 5680)
RUN = (5603.0, 5663.0)      # the flames leave the council fire, and the last hearth is reached


def cost_grid(cx, cy, span, step):
    """The walking cost of the invented land around (cx, cy): slope and height cost, rivers are forded (a
    penalty, so a road crosses them and does not follow them), open water is not crossed. Returns xs, ys, X, Y,
    cost, land."""
    xs = np.arange(cx - span[0], cx + span[0] + 1e-9, step)
    ys = np.arange(cy - span[1], cy + span[1] + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    Fw = geo.fields()
    land = geo.sample(Fw['land'].astype(np.float32), X, Y) * (1.0 - geo.sample(Fw['lake'].astype(np.float32), X, Y))
    elev = geo.sample('E', X, Y)
    rug = geo.sample('rug', X, Y)
    acc = geo.sample('acc', X, Y)
    river = np.clip(np.log(np.maximum(acc, 1.0) / terra.RIVER_T) + 0.6, 0, 1.5)
    # a road keeps off a river's banks and crosses it where it must (a ford), never running along it
    import cv2
    wet = (acc > terra.RIVER_T).astype(np.uint8)
    dr = cv2.distanceTransform(1 - wet, cv2.DIST_L2, 3) * step
    bank = np.exp(-(dr / 0.45) ** 2)
    cost = 1.0 + 2.6 * np.clip(rug, 0, 1.5) + 0.22 * np.clip(elev, 0, 9) + 3.0 * river + 2.2 * bank + 80.0 * (land < 0.5)
    return xs, ys, X, Y, cost, land


def _graph(cost, step):
    from scipy.sparse import coo_matrix
    ny, nx = cost.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    ra, rb, rw = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[0:ny - dy, max(0, -dx):nx - max(0, dx)]
        b = idx[dy:ny, max(0, dx):nx - max(0, -dx)]
        L = step * math.hypot(dx, dy)
        ca = cost.ravel()[a.ravel()]
        cb = cost.ravel()[b.ravel()]
        ra.append(a.ravel())
        rb.append(b.ravel())
        rw.append(L * 0.5 * (ca + cb))
    G = coo_matrix((np.concatenate(rw), (np.concatenate(ra), np.concatenate(rb))), shape=(nx * ny, nx * ny)).tocsr()
    return idx, G


def road_path(a, b, step=0.1):
    """THE ROAD's line from her beacon (a) to the ring (b): the cheapest walk over the invented land (down off the
    range by its gentlest fall, across the valley, a ford, up onto the moor), smoothed as a hand draws it."""
    from scipy.sparse.csgraph import dijkstra
    cx, cy = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    span = (abs(a[0] - b[0]) / 2 + 6.0, abs(a[1] - b[1]) / 2 + 6.0)
    xs, ys, X, Y, cost, _ = cost_grid(cx, cy, span, step)
    idx, G = _graph(cost, step)
    ia = int(idx[int(round((a[1] - ys[0]) / step)), int(round((a[0] - xs[0]) / step))])
    ib = int(idx[int(round((b[1] - ys[0]) / step)), int(round((b[0] - xs[0]) / step))])
    dist, pred = dijkstra(G, directed=False, indices=ia, return_predecessors=True)
    path = [ib]
    while path[-1] != ia and path[-1] >= 0:
        path.append(int(pred[path[-1]]))
    path = path[::-1]
    XYf = np.column_stack([X.ravel(), Y.ravel()])
    P = XYf[path]
    P[0], P[-1] = a, b
    keep = list(range(0, len(P), 6))
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    c = pen.catmull(P[keep], 10)
    return pen.resample(c, 0.02)


def road_tree(ring, n_dest=22, seed=71, step=0.2, span=(27.0, 20.0)):
    """The roads home: the shortest-path tree over the drawn land from the ring of stones to n_dest hearth
    places in the lowlands (low ground, gentle slopes, never across open water), so the roads share their trunks,
    fork and go round the ranges by the passes. Returns (chains, hearths): chains are dicts of polyline `c`,
    arc `rs`, start distance `d0`, served hearth count `n`; hearths are (x, y, arrival distance)."""
    from scipy.sparse.csgraph import dijkstra
    xs, ys, X, Y, cost, land = cost_grid(ring[0], ring[1], span, step)
    elev = geo.sample('E', X, Y)
    rug = geo.sample('rug', X, Y)
    idx, G = _graph(cost, step)
    ny, nx = X.shape
    src = int(idx[int(round((ring[1] - ys[0]) / step)), int(round((ring[0] - xs[0]) / step))])
    dist, pred = dijkstra(G, directed=False, indices=src, return_predecessors=True)
    dist = dist.reshape(ny, nx)
    # hearth places: lowland, on land, spread round the ring
    rng = np.random.default_rng(seed)
    D = np.hypot(X - ring[0], Y - ring[1])
    good = (land > 0.97) & (elev < 1.8) & (rug < 0.33) & (D > 4.0) & (D < 21.0) & np.isfinite(dist)
    ang = np.arctan2(Y - ring[1], X - ring[0])
    chosen = []
    order = rng.permutation(n_dest * 2)
    for k in order:
        a0 = -math.pi + 2 * math.pi * (k + 0.5) / (n_dest * 2)
        m = good & (np.abs(np.angle(np.exp(1j * (ang - a0)))) < math.pi / (n_dest * 2))
        if not m.any():
            continue
        cand = np.argwhere(m)
        sc_ = elev[m] * 0.3 + rug[m] * 2.0 + rng.random(len(cand)) * 1.2 + np.abs(D[m] - rng.uniform(6, 17)) * 0.25
        for j in np.argsort(sc_):
            q = (X[tuple(cand[j])], Y[tuple(cand[j])])
            if all(math.hypot(q[0] - c[0], q[1] - c[1]) > 3.0 for c in chosen):
                chosen.append((q[0], q[1], int(idx[tuple(cand[j])])))
                break
        if len(chosen) >= n_dest:
            break
    # the tree: union of the paths
    children = {}
    dests = set()
    for (_, _, node) in chosen:
        dests.add(node)
        path = [node]
        while path[-1] != src:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        for u, v in zip(path[:-1], path[1:]):
            children.setdefault(u, set()).add(v)
    XYf = np.column_stack([X.ravel(), Y.ravel()])
    chains = []

    def serve(u):
        n = 1 if u in dests else 0
        for v in children.get(u, ()):
            n += serve(v)
        return n
    served = {}

    def walk(u, d0):
        for v in sorted(children.get(u, ())):
            nodes = [u, v]
            while v not in dests and len(children.get(v, ())) == 1:
                v = next(iter(children[v]))
                nodes.append(v)
            P = XYf[nodes]
            keep = list(range(0, len(P), 3))
            if keep[-1] != len(P) - 1:
                keep.append(len(P) - 1)
            c = pen.catmull(P[keep], 6) if len(keep) > 2 else P[keep]
            c = pen.resample(c, 0.025)
            s_ = pen.arclen(c)
            taper = np.clip(s_ / 0.6, 0, 1) * np.clip((s_[-1] - s_) / 0.6, 0, 1)
            c = c + pen.normals(c) * (0.035 * np.sin(s_ * 3.1 + len(chains)) * taper)[:, None]
            rs = pen.arclen(c)
            n = served.setdefault(v, serve(v))
            chains.append(dict(c=c, rs=rs, d0=d0, n=n, end=v, dest=v in dests))
            walk(v, d0 + rs[-1])
    walk(src, 0.0)
    hearths = [(ch['c'][-1, 0], ch['c'][-1, 1], ch['d0'] + ch['rs'][-1]) for ch in chains if ch['dest']]
    return chains, hearths


class FireRemains(RoadShot):
    """C23 bar 71, THE FIRE REMAINS (H5). Burned through from the council plate onto the map: the fire on the
    council stone goes out along the roads, a small moving flame at the head of each fine pen line, sharing the
    trunks and forking round the ranges by the passes and down to the lowlands; where each road arrives a
    hearth is drawn and kindles, gold laid in it. The Road, reversed. The war-beacons on the summits settle low
    as the hearths take the light; the camera draws back from the ring."""

    def __init__(self, tag='full', n=22, seed=71):
        super().__init__(tag)
        self.chains, self.hearths = road_tree(self.ring, n, seed)
        self.Lmax = max(h[2] for h in self.hearths)

    def clock(self, f):
        return 2087.0 + (f - F71[0]) * 0.5            # the fires burn on, settled; nothing new catches

    def front(self, f):
        """How far along the roads (map degrees from the council stone) the flames have come by frame f."""
        u = np.clip((f - RUN[0]) / (RUN[1] - RUN[0]), 0, 1)
        return self.Lmax * u ** 1.35

    def kk(self, f):
        return smooth((f - (F71[0] + 2)) / (F71[1] - F71[0] - 2))

    def camera(self, t, W, H):
        k = self.kk(self.f)
        w = _key(2087.0, 3, True) * 0.28 * (1.0 + 1.05 * k)
        return MAP.Cam(self.ring[0], self.ring[1], w, _key(2087.0, 4) + 3.0 * k, _key(2087.0, 5), W, H)

    def room_gain(self, t):
        return 1.0 + 1.6 * smooth((self.f - 5618.0) / 45.0)       # warm and steady: the room's hearth returns

    def moon_gain(self, t):
        return 1.0 - 0.3 * smooth((self.f - 5618.0) / 45.0)

    def beacon_fade(self, t):
        return 1.0 - 0.45 * smooth((self.f - 5612.0) / 40.0)       # the war-beacons settle low

    def ink_layer(self, cam, f):
        C, Wt = RoadShot.ink_layer(self, cam, F1)                   # the ring, and the Road that came to it
        sc = float(cam.scale(np.array([[self.ring[0], self.ring[1], 0.0]]))[0])
        fr = self.front(f)
        for ch in self.chains:
            reach = fr - ch['d0']
            if reach <= 0:
                continue
            w = 0.62 + 0.22 * math.log2(ch['n'] + 1)
            self.pen_line(C, Wt, cam, ch['c'], ch['rs'], min(reach, ch['rs'][-1]), sc, seed=len(ch['c']), weight=w,
                          r_in=(self.ring, 0.62, 0.82) if ch['d0'] == 0.0 else None)
        return C, Wt

    def extra_glyphs(self, cam, t, f):
        """The hearths drawn where the roads arrive."""
        H, W = cam.H, cam.W
        C = np.zeros((H, W), np.float32)
        Wt = np.zeros((H, W), np.float32)
        gold = np.zeros((H, W), np.float32)
        fr = self.front(f)
        # the fire on the council stone, where the Ring was (the match from the council plate)
        x0, y0, ux, uy, hp, sc = self.council(cam)
        self.flame_glyph(C, Wt, gold, x0, y0, ux, uy, hp, 1.0, sc, 0.9, heavy=1.25, stack=False)
        for q, (x, y, d) in enumerate(self.hearths):
            if fr < d:
                continue
            age = (fr - d) / max(self.Lmax / (RUN[1] - RUN[0]), 1e-6)          # frames since it arrived
            catch = smooth(age / 7.0)
            P = np.array([[x, y, 0.0]])
            uv, z = cam.project(P)
            up, _ = cam.up2d(P)
            sc = float(cam.F / z[0])
            hp = max(0.42 * sc, 7.0)
            self.flame_glyph(C, Wt, gold, uv[0, 0], uv[0, 1], up[0, 0], up[0, 1], hp, catch, sc, 1.3 * q + 0.5,
                             heavy=0.9, stack=False)
        return C, np.clip(gold, 0, 1)

    def council(self, cam):
        P = np.array([[self.ring[0], self.ring[1] - 0.04, 0.0]])
        uv, z = cam.project(P)
        up, _ = cam.up2d(P)
        sc = float(cam.F / z[0])
        return uv[0, 0], uv[0, 1], up[0, 0], up[0, 1], max(0.62 * sc, 12.0), sc

    def draw_fires(self, cam, t, emis, scorch, grid, scale, hush):
        RoadShot.draw_fires(self, cam, t, emis, scorch, grid, scale, hush)
        f = self.f
        p = MAP.pal()
        fr = self.front(f)
        vel = self.Lmax / (RUN[1] - RUN[0])
        x0, y0, ux, uy, hp, sc = self.council(cam)
        MAP.fire.flame(emis, float(x0), float(y0), float(0.74 * hp), float(ux), float(uy), float(t * 2.0), 977,
                       1.3, 1.3, 0.0)
        pts, gains = [], [(self.ring[0], self.ring[1], 0.9)]
        # the moving flames: one at the head of every road still being drawn
        for ch in self.chains:
            reach = fr - ch['d0']
            if reach <= 0 or reach >= ch['rs'][-1]:
                continue
            x = np.interp(reach, ch['rs'], ch['c'][:, 0])
            y = np.interp(reach, ch['rs'], ch['c'][:, 1])
            k = np.clip(np.hypot(x - self.ring[0], y - self.ring[1]) / 0.5, 0.3, 1.0)
            pts.append((x, y, 1.0, k, len(ch['c'])))
        # the hearths, kindling where they arrive
        for q, (x, y, d) in enumerate(self.hearths):
            if fr < d:
                continue
            catch = smooth((fr - d) / vel / 7.0)
            pts.append((x, y, 0.0, catch, 500 + q))
        for (x, y, moving, k, sd) in pts:
            P = np.array([[x, y, 0.0]])
            uv, z = cam.project(P)
            up, _ = cam.up2d(P)
            sc = float(cam.F / z[0])
            if moving:
                h, g = max(0.3 * sc, 6.0) * k, 1.1
            else:
                h, g = max(0.42 * sc, 7.0) * 0.72 * (0.5 + 0.5 * k), 0.95 * k
            MAP.fire.flame(emis, float(uv[0, 0]), float(uv[0, 1]), float(h), float(up[0, 0]), float(up[0, 1]),
                           float(t * 2.0 + 0.7 * sd), int(900 + sd), float(g), 1.3, 0.0)
            gains.append((x, y, 0.3 * g * (0.6 if moving else 1.0)))
        if gains:
            G = np.array(gains)
            grid.add(G[:, 0], G[:, 1], np.outer(G[:, 2], p['light']))

    def finish(self, hdr, t, f):
        dim = 1.0 - 0.2 * smooth((f - (BREATH - 5)) / 8.0)       # the breath: the map settles, a touch darker
        return look.finish(hdr * dim, exposure=self.exposure(t), bloom_strength=0.06, bloom_threshold=1.0,
                           vignette_amount=0.5)


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


@njit(parallel=True, cache=True)
def _x1_kernel(U, V, glow, keep, bf, t):
    for i in prange(U.shape[0]):
        for j in range(U.shape[1]):
            brown, char, hole, edge, pool = BURN.field(bf, U[i, j], V[i, j], t)
            keep[i, j] = (1.0 - hole) * (1.0 - 0.9 * char)
            glow[i, j, 0] = edge * 4.2
            glow[i, j, 1] = edge * 1.25
            glow[i, j, 2] = edge * 0.18


def _x1_field(bf, U, V, t):
    H, W = U.shape
    glow = np.zeros((H, W, 3), np.float32)
    keep = np.ones((H, W), np.float32)
    _x1_kernel(U, V, glow, keep, bf, t)
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
    ap.add_argument('--center', default='%d,%d' % SEVENTH_SCREEN)
    a = ap.parse_args()
    fr = frames_of(a.frames)
    if a.what == 'x1':
        cx, cy = (float(v) for v in a.center.split(','))
        x1_screen((cx, cy), fr, a.out)
        return
    sh = None
    fr_ = None
    for f in fr:
        t0 = time.time()
        if F71[0] <= f < F71[1]:
            fr_ = fr_ or FireRemains('full')
            img = fr_.render_c(f, a.scale)
        else:
            sh = sh or RoadShot('full')
            img = sh.render_c(f, a.scale)
        look.save_png(look.frame_path(a.out, f), img)
        print(f, '%.1fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
