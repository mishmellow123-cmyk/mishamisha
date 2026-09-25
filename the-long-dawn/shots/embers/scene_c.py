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
    d = lerp(10.2, 8.6, float(ease_in_out(u)))
    a = 0.25 - 0.12 * u
    pos = np.array([d * math.sin(a), 1.2 + 0.2 * u, d * math.cos(a)])
    tgt = np.array([0.0, -0.18, 0.0])
    return pos, tgt


# ==================================================================== HAND ===

# right hand, local frame (units of hand length L = wrist crease -> middle fingertip):
# wrist at origin, fingers +Y, palm faces +Z, back of hand -Z, thumb on -X
FINGERS = {  # name: (mcp x, mcp y, phalanx lengths, joint radii at mcp/pip/dip/tip)
    'index': (-0.128, 0.455, (0.205, 0.122, 0.090), (0.046, 0.040, 0.035, 0.029)),
    'middle': (-0.042, 0.468, (0.222, 0.138, 0.095), (0.048, 0.042, 0.036, 0.030)),
    'ring': (0.044, 0.457, (0.208, 0.130, 0.090), (0.045, 0.039, 0.034, 0.028)),
    'little': (0.125, 0.428, (0.166, 0.100, 0.084), (0.040, 0.034, 0.030, 0.025)),
}
FNAMES = ['index', 'middle', 'ring', 'little']
SPREAD_OPEN = {'index': -0.24, 'middle': -0.06, 'ring': 0.1, 'little': 0.28}
SPREAD_REST = {'index': -0.07, 'middle': -0.02, 'ring': 0.03, 'little': 0.09}


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
    """Anatomical point-sampled hand: palm with thenar/hypothenar pads and knuckle ridge,
    4 fingers x 3 tapered phalanges with joint bulges, 3-segment opposing thumb, wrist, forearm.
    Each part is sampled once in its own frame; pose() applies forward kinematics."""

    def __init__(self, seed=303, dens=1.0):
        self.r = rng(seed)
        r = self.r
        self.palm = self._palm(int(42000 * dens))
        self.fore = self._forearm(int(30000 * dens))
        self.caps = {}
        for f in FNAMES:
            lens, rad = FINGERS[f][2], FINGERS[f][3]
            self.caps[f] = [self._phalanx(lens[k], rad[k], rad[k + 1], int(52000 * dens * lens[k] * rad[k] / 0.0092),
                                          tip=(k == 2)) for k in range(3)]
        self.tl = (0.20, 0.145, 0.115)
        tr = (0.062, 0.050, 0.043, 0.034)
        self.thumb = [self._phalanx(self.tl[k], tr[k], tr[k + 1], int(52000 * dens * self.tl[k] * tr[k] / 0.0092),
                                    tip=(k == 2)) for k in range(3)]

    def _phalanx(self, L, r0, r1, n, tip=False):
        """capsule-like segment from joint (0,0,0) to (0,L,0): tapered, with bulging joints."""
        r = self.r
        n = max(n, 200)
        y = r.uniform(-r0 * 0.6, L + (r1 * 0.95 if tip else r1 * 0.4), n)
        a = r.uniform(0, 2 * np.pi, n)
        u = np.clip(y / L, 0, 1)
        rad = (r0 + (r1 - r0) * u) * (1 + 0.1 * np.exp(-(y / (0.2 * L)) ** 2) - 0.06 * np.sin(np.pi * u))
        # slightly flattened on the palm side
        ell = np.stack([np.cos(a), np.sin(a) * np.where(np.sin(a) > 0, 0.86, 1.0)], 1)
        c0 = y < 0
        rad = np.where(c0, np.sqrt(np.maximum(r0 ** 2 - y ** 2, 0)), rad)
        c1 = y > L
        rad = np.where(c1, np.sqrt(np.maximum(r1 ** 2 - (y - L) ** 2, 0)), rad)
        p = np.stack([rad * ell[:, 0], y, rad * ell[:, 1]], 1)
        nrm = np.stack([ell[:, 0], np.zeros(n), ell[:, 1]], 1)
        nrm[c0, 1] = y[c0] / r0
        nrm[c1, 1] = (y[c1] - L) / r1
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        knuckle = np.exp(-(y / (0.22 * L)) ** 2)          # glow at the proximal joint of each phalanx
        return p, nrm, knuckle

    def _palm(self, n):
        r = self.r
        m = int(n * 0.8)
        u = r.uniform(-1, 1, m)                 # across (thumb side -1)
        v = r.uniform(0, 1, m)                  # wrist 0 -> knuckles 1
        yy = 0.01 + 0.45 * v
        hw = 0.14 + 0.07 * v ** 0.8             # broader toward the knuckles
        x = u * hw
        side = np.where(r.random(m) < 0.5, -1.0, 1.0)      # -1 back of hand, +1 palm
        th = 0.058 * np.sqrt(np.maximum(1 - np.abs(u) ** 6, 0.05))
        # pads on the palm side
        thenar = 0.05 * np.exp(-(((x + 0.11) / 0.07) ** 2 + ((yy - 0.15) / 0.11) ** 2))
        hypo = 0.03 * np.exp(-(((x - 0.12) / 0.06) ** 2 + ((yy - 0.2) / 0.13) ** 2))
        pads = 0.02 * np.exp(-((yy - 0.42) / 0.035) ** 2)                 # the fleshy ridge under the fingers
        cup = -0.018 * (1 - u ** 2) * np.sin(np.pi * v)
        z_palm = th + thenar + hypo + pads + cup
        # back of the hand: tendons + knuckle ridge
        knuckle_ridge = 0.028 * np.exp(-((yy - 0.445) / 0.03) ** 2) * (0.6 + 0.4 * np.cos(u * np.pi * 3.5) ** 2)
        tendons = 0.006 * np.cos(u * np.pi * 3.5) ** 8 * v
        z_back = -(th + knuckle_ridge + tendons)
        z = np.where(side > 0, z_palm, z_back)
        p = np.stack([x, yy, z], 1)
        nrm = np.stack([np.zeros(m), np.zeros(m), side], 1)
        kn = np.where(side < 0, np.exp(-((yy - 0.445) / 0.035) ** 2), 0.0)
        # rim around the palm edge
        k = n - m
        v2 = r.uniform(0, 1, k)
        yy2 = 0.01 + 0.45 * v2
        hw2 = 0.14 + 0.07 * v2 ** 0.8
        sgn = np.where(r.random(k) < 0.5, -1.0, 1.0)
        ang = r.uniform(-np.pi / 2, np.pi / 2, k)
        x2 = sgn * (hw2 + 0.045 * np.cos(ang))
        z2 = 0.058 * np.sin(ang)
        # thumb-side web is thicker (thenar)
        x2 = np.where(sgn < 0, x2 - 0.02 * np.exp(-((yy2 - 0.15) / 0.12) ** 2), x2)
        p2 = np.stack([x2, yy2, z2], 1)
        n2 = np.stack([sgn * np.cos(ang), np.zeros(k), np.sin(ang)], 1)
        return (np.concatenate([p, p2]), np.concatenate([nrm, n2]), np.concatenate([kn, np.zeros(k)]))

    def _forearm(self, n):
        r = self.r
        y = -r.uniform(0.0, 1.5, n) ** 1.0
        a = r.uniform(0, 2 * np.pi, n)
        t = np.clip(-y / 1.5, 0, 1)
        ax = 0.13 + 0.06 * t ** 0.7            # half-width
        az = 0.075 + 0.05 * t ** 0.7           # half-depth
        # wrist narrowing
        ax = ax * (1 - 0.12 * np.exp(-((y + 0.05) / 0.08) ** 2))
        p = np.stack([ax * np.cos(a), y, az * np.sin(a)], 1)
        nrm = np.stack([np.cos(a) / ax, np.zeros(n), np.sin(a) / az], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        return p, nrm, np.zeros(n)

    def pose(self, flex, spread, thumb_close):
        P, N, ID, KN = [], [], [], []
        for part, pid in ((self.palm, 0), (self.fore, 9)):
            P.append(part[0])
            N.append(part[1])
            KN.append(part[2])
            ID.append(np.full(len(part[0]), pid))
        for fi, f in enumerate(FNAMES):
            mx, my, lens, _ = FINGERS[f]
            R = _rz(-spread[f])
            p0 = np.array([mx, my, 0.0])
            for k in range(3):
                R = R @ _rx(flex[f][k])
                cp, cn, ck = self.caps[f][k]
                P.append(cp @ R.T + p0)
                N.append(cn @ R.T)
                KN.append(ck)
                ID.append(np.full(len(cp), 1 + fi))
                p0 = p0 + R @ np.array([0, lens[k], 0])
        # thumb: CMC on the radial side near the wrist; swings from abducted to opposed
        cmc = np.array([-0.1, 0.075, 0.02])
        d_open = np.array([-0.66, 0.62, 0.42])
        d_closed = np.array([0.2, 0.56, 0.8])
        d = lerp(d_open, d_closed, thumb_close)
        d /= np.linalg.norm(d)
        y = d
        x = np.cross(y, np.array([0, 0, 1.0]))
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
        R = np.stack([x, y, z], 1)
        p0 = cmc
        bend = 0.18 + 0.6 * thumb_close
        for k in range(3):
            if k > 0:
                R = R @ _rx(bend)
            cp, cn, ck = self.thumb[k]
            P.append(cp @ R.T + p0)
            N.append(cn @ R.T)
            KN.append(ck)
            ID.append(np.full(len(cp), 5))
            p0 = p0 + R @ np.array([0, self.tl[k], 0])
        return np.concatenate(P), np.concatenate(N), np.concatenate(ID), np.concatenate(KN)


def hand_pose_at(t):
    """rise relaxed (slight curl) -> reach open and spread (~995-1012) -> close 1015-1036 (fist on the crown)."""
    reach = float(smoothstep(988, 1012, t))
    close = float(ease_in((t - 1015) / 21.0, 1.7))
    flex, spread = {}, {}
    rest = (0.3, 0.38, 0.22)
    openp = (0.1, 0.14, 0.08)
    closed = {'index': (1.2, 1.55, 0.95), 'middle': (1.25, 1.6, 1.0), 'ring': (1.3, 1.6, 1.0),
              'little': (1.35, 1.55, 0.95)}
    for fi, f in enumerate(FNAMES):
        # the little finger leads the close slightly, like a real grip
        cf = float(np.clip(close * (1.0 + 0.06 * (fi - 1.5)), 0, 1))
        a = [lerp(rest[k], openp[k], reach) for k in range(3)]
        a = [lerp(a[k], closed[f][k], cf) for k in range(3)]
        flex[f] = tuple(a)
        s = lerp(SPREAD_REST[f], SPREAD_OPEN[f], reach)
        spread[f] = lerp(s, SPREAD_REST[f] * 0.2, cf)
    thumb = lerp(lerp(0.35, 0.02, reach), 0.9, close)
    return flex, spread, thumb


HAND_L = 60.0
GRASP_AZ = B.ALPHA_C - 0.24


def grasp_dir():
    """horizontal unit vector from the grasp camera toward the crown"""
    a = GRASP_AZ
    return -np.array([math.cos(a), 0.0, math.sin(a)])


def hand_transform(t):
    """World placement of the hand: rotation (local->world) and wrist position.
    Back of the hand faces the camera; the crown is beyond the palm (a backlit silhouette)."""
    C = B.crown_centre(960.0)
    dz = grasp_dir()
    up = np.array([0.0, 1.0, 0.0])
    dx = np.cross(up, dz)
    dx /= np.linalg.norm(dx)
    R0 = np.stack([dx, up, dz], 1)
    rise = float(ease_out((t - 960) / 52.0, 2.0))
    approach = float(ease_in_out((t - 1004) / 32.0))
    Wf = C - up * (0.6 * HAND_L) - dz * (0.16 * HAND_L)
    W0 = Wf + np.array([0.0, -80.0, 0.0]) - dz * 10.0 + dx * 10.0
    Wm = Wf + np.array([0.0, -20.0, 0.0]) - dz * 6.0 + dx * 3.0
    W = lerp(lerp(W0, Wm, rise), Wf, approach)
    lean = lerp(-0.35, -0.1, rise) + 0.2 * approach
    R = R0 @ _rx(lean) @ _rz(lerp(0.18, 0.02, rise))
    return R, W


class Hand:
    def __init__(self):
        self.rig = HandRig()
        self.rnd = None
        r = rng(8)
        self.ns = 16000
        self.s_idx = None
        self.s_age = r.uniform(0, 1, self.ns)
        self.s_v = r.normal(0, 1, (self.ns, 3)) * 0.35 + np.array([0, 1.0, 0])
        self.s_E = r.lognormal(0, 0.6, self.ns)

    def world(self, t):
        flex, spread, th = hand_pose_at(t)
        p, n, pid, kn = self.rig.pose(flex, spread, th)
        R, W = hand_transform(t)
        return (p * HAND_L) @ R.T + W, n @ R.T, pid, kn

    def emit(self, ctx, fr_hand, fr_cov):
        t = ctx.t
        if t < 960 or t >= 1040:
            return
        P0, _, _, _ = self.world(ctx.t0)
        P1, N1, pid, kn = self.world(ctx.t1)
        npts = len(P1)
        if self.rnd is None or len(self.rnd) != npts:
            rr = rng(5)
            self.rnd = rr.random(npts)
            self.ph = rr.uniform(0, 2 * np.pi, npts)
            self.s_idx = rr.integers(0, npts, self.ns)
        C = B.crown_centre(960.0)
        L = C - P1
        dL = np.linalg.norm(L, axis=1)
        lam = np.maximum((N1 * L).sum(1) / np.maximum(dL, 1e-6), 0)
        vd = ctx.cam.pos - P1
        vd /= np.linalg.norm(vd, axis=1, keepdims=True)
        facing = (N1 * vd).sum(1)
        rim = np.exp(-np.abs(facing) / 0.2)                     # silhouette edges glow
        interior = smoothstep(0.15, 0.7, facing)                # the broad faces toward us: darker
        fl = 1 + 0.5 * np.sin(1.1 * t + self.ph) * np.sin(0.41 * t + 2 * self.ph)
        close = float(smoothstep(1015, 1036, t))
        near = smoothstep(40.0, 8.0, dL)                        # parts that approach the crown
        e_body = (0.3 + 0.7 * self.rnd) * fl * 0.5 * (1 - 0.55 * interior)
        e_edge = rim * (0.9 + 0.6 * self.rnd) * 1.1
        e_kn = kn * (0.8 + 0.8 * self.rnd) * 1.6 * fl
        # the crown's cold light rims whatever turns toward it (fingertips from above as they approach)
        e_cold = lam * (0.2 + 0.8 * rim) * near * (2.2 + 5.0 * close)
        c_body = C_CRIMSON * 0.55 + C_RED * 0.35 + look.blackbody(0.35) * 0.1
        c_edge = look.blackbody(0.52)
        c_kn = look.blackbody(0.62)
        c_cold = C_ICE * 0.6 + C_CORE * 0.4
        colE = (c_body[None, :] * e_body[:, None] + c_edge[None, :] * e_edge[:, None] +
                c_kn[None, :] * e_kn[:, None] + c_cold[None, :] * e_cold[:, None])
        R, W = hand_transform(t)
        along = (P1 - W) @ R[:, 1]
        fade = np.where(pid == 9, smoothstep(-1.45 * HAND_L, -0.5 * HAND_L, along), 1.0)
        fade = fade * smoothstep(960, 964, t)
        colE *= fade[:, None]
        fr_hand.splat(P0, P1, 0.035, np.ones(npts), colE, ctx.cam0, ctx.cam1, zref=0.0)
        fr_cov.splat(P0, P1, 0.035, fade * 0.8, np.ones(3), ctx.cam0, ctx.cam1, zref=0.0)
        # sparks shedding from the moving hand (trail: born at the surface in the recent past)
        ages = self.s_age * 14.0
        tb = t - ages
        Ps = np.empty((self.ns, 3))
        for b0 in range(0, 15, 3):
            m = (ages >= b0) & (ages < b0 + 3)
            if m.any():
                Pb, _, _, _ = self.world(t - (b0 + 1.5))
                Ps[m] = Pb[self.s_idx[m]]
        drift = self.s_v * ages[:, None] * 0.35
        S1 = Ps + drift
        S0 = S1 - self.s_v * 0.35 * 0.5
        mov = float(smoothstep(962, 975, t)) * (1 - 0.6 * close)
        es = self.s_E * (1 - self.s_age) ** 2 * 14.0 * mov
        ctx.fr.splat(S0, S1, 0.03, es, look.blackbody(0.7 - 0.3 * self.s_age), ctx.cam0, ctx.cam1, zref=60.0)


def cam_grasp(t):
    """Low, looking up at the storm's eye; the hand rises from the bottom edge (slow, huge)."""
    C = B.crown_centre(960.0)
    u = (t - 960) / 80.0
    push = float(ease_in_out(u))
    a = GRASP_AZ
    d = lerp(128.0, 112.0, push)
    pos = np.array([d * math.cos(a), lerp(8.0, 12.0, push), d * math.sin(a)])
    tgt = C + np.array([0.0, lerp(-26.0, -20.0, push), 0.0])
    if t > 1004:
        k = (t - 1004) / 32.0
        sh = 0.3 * k * k
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
