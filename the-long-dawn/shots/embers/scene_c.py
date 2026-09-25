"""EMBERS part C: the world as a cracking ember globe (880-960), the GRASP (960-1040),
and the SILENCE (1040-1199)."""
import json
import math
import os

import cv2
import numpy as np

import look
from core import (clamp01, smoothstep, smootherstep, ease_out, ease_in, ease_in_out, lerp, vnoise,
                  rng, rand_dirs, catmull, Camera)
import scene_b as B

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
C_CORE = look.hexrgb(look.PALETTE['mind_core'])
C_ICE = look.hexrgb(look.PALETTE['mind_ice'])
C_GOLD = look.hexrgb(look.PALETTE['mind_gold'])
C_RED = look.hexrgb(look.PALETTE['race_red'])
C_CRIMSON = look.hexrgb(look.PALETTE['race_crimson'])


# =================================================================== GLOBE ===

def land_mask(W=2048, H=1024):
    d = json.load(open(os.path.join(ROOT, 'assets', 'data', 'ne_110m_land.geojson')))
    m = np.zeros((H, W), np.uint8)
    for f in d['features']:
        for ring in f['geometry']['coordinates'][:1]:
            pts = np.array([[(lon + 180) / 360 * W, (90 - lat) / 180 * H] for lon, lat in ring], np.float64)
            cv2.fillPoly(m, [np.round(pts * 8).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=3)
        for ring in f['geometry']['coordinates'][1:]:
            pts = np.array([[(lon + 180) / 360 * W, (90 - lat) / 180 * H] for lon, lat in ring], np.float64)
            cv2.fillPoly(m, [np.round(pts * 8).astype(np.int32)], 0, lineType=cv2.LINE_8, shift=3)
    return m


def sph_to_ll(p):
    lat = np.degrees(np.arcsin(np.clip(p[:, 1], -1, 1)))
    lon = np.degrees(np.arctan2(-p[:, 2], p[:, 0]))
    return lat, lon


def lookup(mask, p):
    H, W = mask.shape
    lat, lon = sph_to_ll(p)
    x = np.clip(((lon + 180) / 360 * W).astype(int), 0, W - 1)
    y = np.clip(((90 - lat) / 180 * H).astype(int), 0, H - 1)
    return mask[y, x]


class Globe:
    def __init__(self, seed=202):
        r = rng(seed)
        mask = land_mask()
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        P = rand_dirs(r, 900000)
        onland = lookup(mask, P) > 0
        self.land = P[onland][:260000]
        Q = rand_dirs(r, 1400000)
        oncoast = lookup(edge, Q) > 0
        self.coast = Q[oncoast][:60000]
        self.ocean = P[~onland][:26000]
        # plates: Voronoi on the sphere
        self.seeds = rand_dirs(r, 16)
        # cracks: points near Voronoi edges (jagged with noise)
        C = rand_dirs(r, 1600000)
        Cj = C + vnoise(C * 3.0, 1.0, (0, 0, 0), 2) * 0.06
        Cj /= np.linalg.norm(Cj, axis=1, keepdims=True)
        d = np.arccos(np.clip(Cj @ self.seeds.T, -1, 1))
        ds = np.sort(d, axis=1)
        crack = (ds[:, 1] - ds[:, 0]) < 0.018
        self.crack = C[crack]
        # crack activation: spreads from several origins (the race is everywhere)
        origins = rand_dirs(r, 5)
        dg = np.arccos(np.clip(self.crack @ origins.T, -1, 1)).min(axis=1)
        self.t_act = 882.0 + 52.0 * dg / np.pi * 2.2 + r.uniform(-3, 3, len(self.crack))
        self.cell = {}
        for name in ('land', 'coast', 'ocean', 'crack'):
            pts = getattr(self, name)
            self.cell[name] = np.argmax(pts @ self.seeds.T, axis=1)
        # fire spilling out of the cracks
        n = 60000
        self.sp_i = r.integers(0, len(self.crack), n)
        self.sp_ph = r.random(n)
        self.sp_v = r.uniform(0.004, 0.012, n)
        self.sp_E = r.lognormal(0, 0.6, n)
        self.rnd = {k: r.random(len(getattr(self, k))) for k in ('land', 'coast', 'ocean', 'crack')}
        print('globe: land', len(self.land), 'coast', len(self.coast), 'crack', len(self.crack))

    def rot(self, t):
        a = 2.3 + 0.0045 * (t - 880)
        tilt = math.radians(20)
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
        return Rx @ Ry

    def split(self, t):
        return 0.075 * float(ease_in_out((t - 922) / 30.0))

    def world(self, name, t, radius=1.0):
        pts = getattr(self, name)
        sp = self.split(t)
        off = self.seeds[self.cell[name]] * sp
        return (pts * radius + off) @ self.rot(t).T

    def emit(self, ctx):
        t = ctx.t
        cam = ctx.cam
        vdir = cam.pos / np.linalg.norm(cam.pos)       # globe at origin
        heat = float(smoothstep(935, 959, t))
        flare = float(smoothstep(950, 959, t))

        def facing(P):
            n = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
            c = n @ vdir
            return smoothstep(-0.05, 0.25, c)
        # land embers
        P0 = self.world('land', ctx.t0)
        P1 = self.world('land', ctx.t1)
        f = facing(P1)
        rnd = self.rnd['land']
        e = (0.55 + 0.45 * np.sin(0.9 * t + 40 * rnd)) * (0.6 + 0.8 * rnd) * 1.1 * f * (1 + 2 * heat)
        col = look.blackbody(0.3 + 0.18 * rnd)
        ctx.fr.splat(P0, P1, 0.0015, e, col, ctx.cam0, ctx.cam1)
        # coastlines: brighter, gold
        P0 = self.world('coast', ctx.t0)
        P1 = self.world('coast', ctx.t1)
        f = facing(P1)
        e = 2.2 * f * (1 + heat)
        ctx.fr.splat(P0, P1, 0.0015, e, look.blackbody(0.55), ctx.cam0, ctx.cam1)
        # ocean: sparse deep-red embers
        P0 = self.world('ocean', ctx.t0)
        P1 = self.world('ocean', ctx.t1)
        f = facing(P1)
        ctx.fr.splat(P0, P1, 0.0015, 0.5 * f, C_CRIMSON, ctx.cam0, ctx.cam1)
        # cracks: hot front, then glowing seams (and the molten core showing through as plates part)
        act = smoothstep(self.t_act, self.t_act + 3.0, t)
        front = np.exp(-np.maximum(t - self.t_act, 0) / 5.0) * act
        P0 = self.world('crack', ctx.t0)
        P1 = self.world('crack', ctx.t1)
        f = facing(P1)
        e = (2.6 * act + 9.0 * front) * f * (1 + 2.5 * flare)
        col = look.blackbody(np.clip(0.62 + 0.35 * front + 0.2 * heat, 0, 1))
        ctx.fr.splat(P0, P1, 0.0015, e, col, ctx.cam0, ctx.cam1)
        sp = self.split(t)
        if sp > 0.002:
            Pc = (self.crack * 0.965) @ self.rot(t).T
            fc = facing(Pc)
            ec = act * fc * 6.0 * (sp / 0.075) * (1 + 3 * flare)
            ctx.fr.splat(Pc, Pc, 0.002, ec, look.blackbody(0.9), ctx.cam0, ctx.cam1)
        # fire spilling out of the cracks, rising off the surface
        i = self.sp_i
        base = self.crack[i]
        ok = t > self.t_act[i] + 2

        def spill(tq):
            k = (self.sp_ph + self.sp_v * (tq - 880)) % 1.0
            p = base * (1.0 + 0.2 * k[:, None] ** 1.2)
            w = vnoise(p * 4.0 + np.array([0, 0.02 * tq, 0]), 1.0, (0, 0, 0), 1)
            p = p + w * (0.02 + 0.05 * k)[:, None]
            off = self.seeds[self.cell['crack'][i]] * self.split(tq)
            return (p + off) @ self.rot(tq).T, k
        S0, k0 = spill(ctx.t0)
        S1, k1 = spill(ctx.t1)
        f = smoothstep(-0.3, 0.2, (S1 / np.linalg.norm(S1, axis=1, keepdims=True)) @ vdir)
        e = self.sp_E * (1 - k1) ** 1.5 * 3.0 * ok * f * (k1 >= k0) * (1 + 2 * heat)
        col = look.blackbody(np.clip(0.85 - 0.5 * k1, 0.2, 1))
        ctx.fr.splat(S0, S1, 0.002, e, col, ctx.cam0, ctx.cam1)
        # halo of the burning world
        H = np.zeros((1, 3))
        ctx.fr.splat(H, H, np.array([1.9]), np.array([2600.0 * (0.4 + heat + 3 * flare)]),
                     np.array([C_RED * 0.7 + look.blackbody(0.6) * 0.3]), ctx.cam0, ctx.cam1, profile=1)


def cam_globe(t):
    u = (t - 880) / 80.0
    d = lerp(3.9, 3.05, float(ease_in_out(u)))
    a = 0.25 - 0.12 * u
    pos = np.array([d * math.sin(a), 0.42 + 0.1 * u, d * math.cos(a)])
    tgt = np.array([0.0, -0.3 + 0.08 * u, 0.0])
    return pos, tgt


# ==================================================================== HAND ===

# right hand, local frame: wrist at origin, fingers +Y, palm faces +Z, thumb on -X
FINGERS = {  # name: (mcp x, mcp y, segment lengths, radii)
    'index': (-0.150, 0.445, (0.20, 0.12, 0.09), (0.043, 0.038, 0.033)),
    'middle': (-0.050, 0.458, (0.22, 0.135, 0.095), (0.045, 0.040, 0.034)),
    'ring': (0.050, 0.446, (0.21, 0.13, 0.09), (0.042, 0.037, 0.032)),
    'little': (0.143, 0.418, (0.165, 0.10, 0.085), (0.036, 0.032, 0.028)),
}
FNAMES = ['index', 'middle', 'ring', 'little']
SPREAD_OPEN = {'index': -0.26, 'middle': -0.07, 'ring': 0.1, 'little': 0.3}
SPREAD_REST = {'index': -0.08, 'middle': -0.02, 'ring': 0.03, 'little': 0.1}


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class HandRig:
    """Capsule hand: palm (rounded slab), 4 fingers x 3 phalanges, thumb x 3, forearm.
    Surface points are sampled once per capsule in the capsule's own frame and posed by FK."""

    def __init__(self, seed=303, dens=1.0):
        self.r = rng(seed)
        self.parts = []          # list of dicts: name, local pts (unit capsule frame), normals
        self.dens = dens
        # palm: rounded box points (local, fixed to the hand frame)
        self.palm = self._palm(int(26000 * dens))
        self.fore = self._capsule_pts(1.25, 0.125, 0.155, int(16000 * dens))
        self.caps = {}
        for f in FNAMES:
            segs = []
            for k in range(3):
                L = FINGERS[f][2][k]
                r0 = FINGERS[f][3][k]
                r1 = FINGERS[f][3][k] * 0.9 if k < 2 else FINGERS[f][3][k] * 0.8
                segs.append(self._capsule_pts(L, r0, r1, int((5200 if k == 0 else 3800) * dens * L / 0.15)))
            self.caps[f] = segs
        tl = (0.19, 0.14, 0.11)
        tr = (0.058, 0.05, 0.042)
        self.thumb = [self._capsule_pts(tl[k], tr[k], tr[k] * 0.9, int(5200 * dens * tl[k] / 0.15))
                      for k in range(3)]
        self.tl = tl
        # thenar eminence (thumb muscle) as an ellipsoid patch
        n = int(5000 * dens)
        d = rand_dirs(self.r, n)
        self.thenar = (d * np.array([0.085, 0.13, 0.06]) + np.array([-0.1, 0.16, 0.05]), d)

    def _capsule_pts(self, L, r0, r1, n):
        """points on a capsule from (0,0,0) to (0,L,0), radius r0 -> r1 (with hemispherical caps)."""
        r = self.r
        n = max(n, 50)
        y = r.uniform(-r0 * 0.9, L + r1 * 0.9, n)
        a = r.uniform(0, 2 * np.pi, n)
        rad = np.interp(y, [0, L], [r0, r1])
        cap0 = y < 0
        cap1 = y > L
        rad = np.where(cap0, np.sqrt(np.maximum(r0 ** 2 - y ** 2, 0)), rad)
        rad = np.where(cap1, np.sqrt(np.maximum(r1 ** 2 - (y - L) ** 2, 0)), rad)
        p = np.stack([rad * np.cos(a), y, rad * np.sin(a)], 1)
        nrm = np.stack([np.cos(a), np.zeros(n), np.sin(a)], 1)
        nrm[cap0] = np.stack([rad[cap0] * np.cos(a[cap0]), y[cap0], rad[cap0] * np.sin(a[cap0])], 1)
        nrm[cap1] = np.stack([rad[cap1] * np.cos(a[cap1]), y[cap1] - L, rad[cap1] * np.sin(a[cap1])], 1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        # knuckle-ish ridge lines give the phalanges definition
        return p, nrm

    def _palm(self, n):
        r = self.r
        # superellipsoid-ish slab: x in [-0.19,0.19] (wider at the knuckles), y in [0.02,0.45], z thickness
        u = r.uniform(-1, 1, n)
        v = r.uniform(0, 1, n)
        side = r.choice([-1.0, 1.0], n)
        yy = 0.02 + 0.43 * v
        hw = 0.165 + 0.045 * v
        x = u * hw
        th = 0.065 * (1 - 0.5 * np.abs(u) ** 4) * (1 - 0.3 * (1 - v) ** 2)
        z = side * th
        # rounded edges: move some points onto the rim
        rim = r.random(n) < 0.25
        ang = r.uniform(-np.pi / 2, np.pi / 2, n)
        x = np.where(rim, np.sign(u) * (hw + 0.02 * np.cos(ang)), x)
        z = np.where(rim, th * np.sin(ang) * 1.2, z)
        # cupped palm: the centre sinks slightly
        z = z - 0.02 * (1 - u ** 2) * np.sin(np.pi * v) * (side > 0)
        p = np.stack([x, yy, z], 1)
        nrm = np.stack([np.where(rim, np.sign(u) * np.cos(ang), 0), np.zeros(n),
                        np.where(rim, np.sin(ang), side)], 1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        return p, nrm

    def pose(self, flex, spread, thumb_open):
        """flex: dict name -> (a1,a2,a3) radians; spread: dict name -> rad; thumb_open in [0,1].
        Returns (points, normals, part-id) in the hand frame (units of hand length)."""
        P, N, ID = [], [], []
        P.append(self.palm[0])
        N.append(self.palm[1])
        ID.append(np.zeros(len(self.palm[0]), int))
        P.append(self.thenar[0])
        N.append(self.thenar[1])
        ID.append(np.zeros(len(self.thenar[0]), int))
        fp, fn = self.fore
        P.append(np.stack([fp[:, 0], -fp[:, 1] + 0.05, fp[:, 2]], 1))
        N.append(np.stack([fn[:, 0], -fn[:, 1], fn[:, 2]], 1))
        ID.append(np.full(len(fp), 9))
        for fi, f in enumerate(FNAMES):
            mx, my, lens, _ = FINGERS[f]
            Rm = _rz(-spread[f])
            p0 = np.array([mx, my, 0.0])
            R = Rm.copy()
            for k in range(3):
                R = R @ _rx(flex[f][k])
                cp, cn = self.caps[f][k]
                P.append(cp @ R.T + p0)
                N.append(cn @ R.T)
                ID.append(np.full(len(cp), 1 + fi))
                p0 = p0 + R @ np.array([0, lens[k], 0])
        # thumb: from CMC, direction interpolates open (abducted) -> opposed (across the palm)
        cmc = np.array([-0.12, 0.07, 0.035])
        d_open = np.array([-0.62, 0.62, 0.48])
        d_closed = np.array([0.25, 0.62, 0.74])
        d = lerp(d_open, d_closed, thumb_open)
        d /= np.linalg.norm(d)
        # frame whose +Y is d
        y = d
        x = np.cross(y, np.array([0, 0, 1.0]))
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
        R = np.stack([x, y, z], 1)
        p0 = cmc
        bend = 0.25 + 0.55 * thumb_open
        for k in range(3):
            if k > 0:
                R = R @ _rx(bend)
            cp, cn = self.thumb[k]
            P.append(cp @ R.T + p0)
            N.append(cn @ R.T)
            ID.append(np.full(len(cp), 5))
            p0 = p0 + R @ np.array([0, self.tl[k], 0])
        return np.concatenate(P), np.concatenate(N), np.concatenate(ID)


def hand_pose_at(t):
    """Keyframed hand performance: rise (relaxed) -> reach (open, spread) -> grasp (closed at 1036)."""
    reach = float(smoothstep(990, 1016, t))
    close = float(ease_in((t - 1024) / 12.0, 2.2))
    flex, spread = {}, {}
    rest = (0.28, 0.35, 0.2)
    openp = (0.05, 0.06, 0.03)
    closed = {'index': (1.05, 1.45, 0.95), 'middle': (1.1, 1.5, 1.0), 'ring': (1.15, 1.5, 1.0),
              'little': (1.2, 1.45, 0.95)}
    for f in FNAMES:
        a = [lerp(rest[k], openp[k], reach) for k in range(3)]
        a = [lerp(a[k], closed[f][k], close) for k in range(3)]
        flex[f] = tuple(a)
        s = lerp(SPREAD_REST[f], SPREAD_OPEN[f], reach)
        spread[f] = lerp(s, SPREAD_REST[f] * 0.3, close)
    thumb = lerp(lerp(0.3, 0.0, reach), 0.95, close)
    return flex, spread, thumb


HAND_L = 80.0


def hand_transform(t):
    """World placement of the hand (rotation + wrist position), relative to the crown centre."""
    C = B.crown_centre(960.0)
    rise = float(ease_out((t - 960) / 46.0, 2.4))
    approach = float(ease_in_out((t - 996) / 40.0))
    # final: palm in front of the crown so the fingers wrap it
    Wf = C + np.array([0.0, -0.56 * HAND_L, -0.11 * HAND_L])
    W0 = Wf + np.array([10.0, -95.0, -30.0])
    Wm = Wf + np.array([4.0, -22.0, -16.0])
    W = lerp(lerp(W0, Wm, rise), Wf, approach)
    # orientation: fingers lean toward the crown as it reaches
    lean = lerp(-0.35, 0.12, rise) + 0.1 * approach
    yaw = lerp(0.25, 0.0, rise)
    R = _ry(yaw) @ _rx(lean)
    return R, W


class Hand:
    def __init__(self):
        self.rig = HandRig()
        self.rnd = None

    def world(self, t):
        flex, spread, th = hand_pose_at(t)
        p, n, pid = self.rig.pose(flex, spread, th)
        R, W = hand_transform(t)
        return (p * HAND_L) @ R.T + W, n @ R.T, pid

    def emit(self, ctx):
        t = ctx.t
        if t < 960 or t >= 1040:
            return
        P0, _, _ = self.world(ctx.t0)
        P1, N1, pid = self.world(ctx.t1)
        if self.rnd is None or len(self.rnd) != len(P1):
            rr = rng(5)
            self.rnd = rr.random(len(P1))
            self.ph = rr.uniform(0, 2 * np.pi, len(P1))
        C = B.crown_centre(960.0)
        # ember body (deep red), lit rim from the crown (gold/white), back faces dimmer
        L = C - P1
        dL = np.linalg.norm(L, axis=1)
        lam = np.maximum((N1 * L).sum(1) / np.maximum(dL, 1e-6), 0)
        vd = ctx.cam.pos - P1
        vd /= np.linalg.norm(vd, axis=1, keepdims=True)
        facing = (N1 * vd).sum(1)
        front = 0.25 + 0.75 * smoothstep(-0.2, 0.3, facing)
        rim = np.exp(-np.abs(facing) / 0.25)                    # silhouette edges glow
        fl = 1 + 0.4 * np.sin(1.1 * t + self.ph) * np.sin(0.41 * t + 2 * self.ph)
        close = float(smoothstep(1020, 1036, t))
        e_body = (0.5 + 0.5 * self.rnd) * fl * 1.5 * front
        e_rim = rim * 2.0 * (0.4 + 0.6 * self.rnd)
        e_lit = lam * 520.0 / (1 + (dL / 14.0) ** 2) * (1 + 3 * close)
        forearm_fade = np.where(pid == 9, smoothstep(-1.2 * HAND_L, -0.2 * HAND_L,
                                                      ((P1 - hand_transform(t)[1]) @ hand_transform(t)[0][:, 1])), 1.0)
        cb = C_RED * 0.55 + C_CRIMSON * 0.45
        colE = (cb[None, :] * e_body[:, None] + look.blackbody(0.55)[None, :] * e_rim[:, None] +
                (C_GOLD * 0.7 + C_CORE * 0.3)[None, :] * e_lit[:, None])
        colE *= (forearm_fade * smoothstep(960, 972, t))[:, None]
        ctx.fr.splat(P0, P1, 0.05, np.ones(len(P1)), colE, ctx.cam0, ctx.cam1, zref=120.0)


def cam_grasp(t):
    C = B.crown_centre(960.0)
    u = (t - 960) / 80.0
    push = float(ease_in_out(u))
    a = B.ALPHA_C - 0.2 - 1.15
    d = lerp(200.0, 165.0, push)
    pos = C + np.array([d * math.cos(a), lerp(-26.0, -18.0, push), d * math.sin(a)])
    tgt = C + np.array([0.0, lerp(-24.0, -10.0, push), 0.0])
    # riser: a slow shake building to the close
    sh = 0.0
    if t > 1000:
        k = (t - 1000) / 36.0
        sh = 0.5 * k * k
        pos = pos + sh * np.array([math.sin(3.1 * t), math.sin(4.3 * t), 0])
    return pos, tgt


# ================================================================= SILENCE ===

class LastEmber:
    """One small ember drifting down in black, dimming, dying by ~1150 (lower third stays clear)."""

    def __init__(self):
        r = rng(77)
        self.n_smoke = 260
        self.sm_d = r.normal(0, 1, (self.n_smoke, 3))
        self.sm_ph = r.random(self.n_smoke)
        self.sm_v = r.uniform(0.004, 0.01, self.n_smoke)

    def pos(self, t):
        u = t - 1040
        x = 0.35 + 0.18 * math.sin(u * 0.045) + 0.04 * math.sin(u * 0.13)
        y = 1.05 - 0.0068 * u - 0.000012 * u * u
        z = -6.0 + 0.3 * math.sin(u * 0.03)
        return np.array([x, y, z])

    def life(self, t):
        u = t - 1040
        base = math.exp(-u / 70.0)
        fl = 1 + 0.35 * math.sin(u * 0.9) * math.sin(u * 0.23)
        # a last flare, then out
        flare = 0.8 * math.exp(-((u - 98) / 3.0) ** 2)
        die = 1 - smoothstep(100, 110, u)
        return max(base * fl + flare, 0) * die

    def emit(self, ctx):
        t = ctx.t
        if t < 1040:
            return
        lv = self.life(t)
        if lv > 0.001:
            p0 = self.pos(ctx.t0)[None, :]
            p1 = self.pos(ctx.t1)[None, :]
            temp = 0.45 + 0.35 * lv
            e = np.array([60.0 * lv])
            ctx.fr.splat(p0, p1, 0.006, e, look.blackbody(temp)[None, :], ctx.cam0, ctx.cam1)
            H = np.array([self.pos(t)])
            ctx.fr.splat(H, H, np.array([0.12]), np.array([90.0 * lv]), look.blackbody(temp - 0.1)[None, :],
                         ctx.cam0, ctx.cam1, profile=1)
        # a thin wisp of smoke after it dies (barely there)
        u = t - 1150
        if 0 < u < 45:
            base = self.pos(1150.0)
            k = (self.sm_ph + u * self.sm_v * 6) % 1.0
            P = base + np.array([0, 1.0, 0]) * (k * 0.5)[:, None] + self.sm_d * (0.02 + 0.08 * k)[:, None]
            e = (1 - k) * 0.25 * (1 - u / 45.0) * math.exp(-u / 20.0)
            ctx.fr.splat(P, P, 0.03, e, np.array([0.5, 0.35, 0.3]), ctx.cam0, ctx.cam1, profile=1)


def cam_silence(t):
    return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -10.0])
