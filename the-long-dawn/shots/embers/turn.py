"""EMBERS v3, cut A (EMBERS-A3): A16 TOWERS IN THE LIGHT and A17 THE FIRE, SEEN, on A's own frames (a3.py routes
every frame from 4400 here). One take, 4400-4879.

A16 4400-4720 (bars 56-59). The cut follows the watchers' look into the embers: from outside the ring of forges, low,
looking in past their backs, which are lit warm for the first time by small fires on the far ridges (clusters of one to
five on ridgelines at every distance and height, with long dark gaps: never a ring). 4400 and 4420: the last two surges;
4440 they stop. 4480: the two giants' shutters open first, on the faces turned to each other, from the bottom up in a
rising wave; their light crosses the pit past the fire and lands on the other's face. From 4560 the smaller forges
follow, one per beat, the light spreading round the ring from the giants (3, 7, 1, 5, then 4 and 0, which meet). The
gilding cools and by 4700 the rim holds (no more crumbs off the lip).
A17 4720-4880 (bars 60-61). The camera walks on, down between giant 2 and forge 1, to the rim; the fire, calm, its body
clear. Small lights come down to the rim from the towers and from the hills alike. From 4800 the fire gathers into one
small, intense heart, centred at (960, 402) on 4879 for THE CROSSING's match cut (the lantern's heart: ice-white
(0.80, 0.92, 1.00), core ~10 px, glow ~26 px, faint halo ~114 px); the world it lit darkens round it to the night.

Look: A5's thinking fire and A6's charcoal forge-stacks (approved); THE EDGE's crater (edge.py, EMBERS-2) calming.
Hooks over a3/edge's schedule (installed on import, t >= 4400 only): shutters, back light, fire centre/scale/power, the
crater as occluder. Farm look-dev: TURN_CAMS='{"4400": [r, az, y, ty, hf], ...}' overrides the camera per frame.
"""
import json
import math
import os

import numpy as np

import look
from core import Camera, smoothstep, smootherstep, lerp, catmull, rng, vnoise
import scene_b as B
import a3 as A
import edge

BEAT = 20.0
T_LIGHT, T_STOP, T_OPEN_G, T_OPEN_O = A.T_LIGHT, A.T_STOP, A.T_OPEN_GIANTS, A.T_OPEN_OTHERS
T_HOLDS, T_SEEN, T_HEART, T_END = A.T_HOLDS, A.T_SEEN, A.T_HEART, A.T_END
GIANTS = tuple(A.GIANTS)
OPEN_ORDER = (3, 7, 1, 5, 4, 0)       # one per beat from 4560: from the giants' neighbours round to the two that meet
FIRE_Y = 0.0                           # calm, the fire hangs free over the pit again, as in A5/A6 (THE EDGE had
                                       # drawn it down into the crater's mouth)
C_WIN = look.blackbody(0.64)           # a forge's own light, its shutters open
C_RIDGE = look.blackbody(0.5)          # the far fires' light
C_LAMP = look.blackbody(0.6)           # the small lights
C_HEART = np.array([0.80, 0.92, 1.00])  # THE CROSSING's heart_col at its first frame
HEART_PX = (10.0, 26.0, 114.0)         # the crossing's heart: core, glow, halo sigmas (px) at 4880


# ================================================================ schedule hooks (t >= 4400 only)

def t_open(i):
    if i in GIANTS:
        return float(T_OPEN_G)
    if i in OPEN_ORDER:
        return float(T_OPEN_O + BEAT * OPEN_ORDER.index(i))
    return 1e9


def opened(i, t):
    """0..1: how far tower i's shutters are open over its whole facade (the wave climbs it)"""
    return float(smoothstep(0.0, 30.0 if i in GIANTS else 18.0, t - t_open(i)))


def gather(t):
    """0 -> 1 as the fire gathers into its heart (bar 61)"""
    return float(smootherstep(T_HEART, T_END - 12, t))


def world_light(t):
    """the fire's light on the world: it draws back into the heart"""
    return 1.0 - 0.72 * float(smootherstep(T_HEART + 10, T_END - 8, t))


_ORIG = A.__dict__.setdefault('_turn_orig', {})


def _orig(name):
    if name not in _ORIG:
        _ORIG[name] = getattr(A.A3Sched, name)
    return _ORIG[name]


def _shutter(self, i, t, pl):
    if t < T_LIGHT:
        return _orig('shutter')(self, i, t, pl)
    # scene_b's own window light stays at the race's level until the tower opens; Shutters draws the open windows
    return (0.04 if i in GIANTS else 0.3) * (1.0 - opened(i, t))


def _back_light(self, t):
    return _orig('back_light')(self, t) if t < T_LIGHT else 0.0      # TowerLight lights the backs


def _fire_centre(self, t):
    if t < T_LIGHT:
        return _orig('fire_centre')(self, t)
    return np.array([0.0, FIRE_Y, 0.0])


def _fire_scale(self, t):
    if t < T_LIGHT:
        return _orig('fire_scale')(self, t)
    return 2.0 * (1.0 - 0.86 * gather(t))


def _power(self, t):
    if t < T_LIGHT:
        return _orig('power')(self, t)
    s_ = _fire_scale(self, t)
    g = gather(t)
    # the flame's surface brightens as it draws in, and hands its light over to the heart (Heart) at the end
    return s_ ** 1.6 * (1.0 + 1.8 * g) * (1.0 - 0.9 * float(smoothstep(T_HEART + 40, T_END - 4, t)))


def _extra_occluders(self, towers, ctx):
    if ctx.t < T_LIGHT:
        return _orig('extra_occluders')(self, towers, ctx)
    cr = getattr(ctx, 'tl_crater', None)
    if cr is None:
        return None
    P, N, Ar = cr.occluders(ctx.t)
    return (P.astype(np.float64), N.astype(np.float64), Ar.astype(np.float64),
            np.full(len(P), edge.CRATER_ID, np.int32))


for _n, _f in (('shutter', _shutter), ('back_light', _back_light), ('fire_centre', _fire_centre),
               ('fire_scale', _fire_scale), ('power', _power), ('extra_occluders', _extra_occluders)):
    _orig(_n)
    setattr(A.A3Sched, _n, _f)


# ================================================================ camera

def _polar(r, a, y):
    return np.array([r * math.cos(a), y, r * math.sin(a)])


# one take: (frame, radius, azimuth, height, target y, hfov). From outside the ring behind giant 2 and forge 1 (the
# watchers' look), a slow push; A17 walks on, down through the gap between them (az ~1.53) to the rim.
CAM = [
    (4400, 64.0, 1.620, 9.0, 3.0, 64.0),
    (4480, 50.0, 1.640, 8.0, 2.0, 63.0),
    (4560, 42.0, 1.660, 8.0, 1.0, 62.0),
    (4640, 38.0, 1.670, 7.0, 1.0, 61.0),
    (4720, 35.0, 1.660, 6.0, 0.5, 60.0),
    (4780, 27.0, 1.600, 0.0, 0.0, 57.0),
    (4840, 21.0, 1.550, -6.0, 0.0, 53.0),
    (4880, 17.5, 1.530, -10.5, 0.0, 50.0),
]
_LAB = {}
if os.environ.get('TURN_CAMS'):
    _LAB = {int(k): v for k, v in json.loads(os.environ['TURN_CAMS']).items()}


def heart_pos(t):
    return B.crown_centre(t) + np.array([0.0, 0.5 * B.fire_radius(t), 0.0])


def camera(tl, t):
    f = int(round(t))
    if f in _LAB:
        r, az, y, ty, hf = _LAB[f][:5]
        pos = _polar(r, az, y)
        tgt = np.array([0.0, ty, 0.0])
        return Camera(pos, tgt, hfov=hf, focus=float(np.linalg.norm(pos - np.array([0.0, FIRE_Y, 0.0]))),
                      aperture=0.03)
    k = [(fr, np.array([r, az, y, ty, hf])) for fr, r, az, y, ty, hf in CAM]
    r, az, y, ty, hf = catmull(t, k)
    # a walker's step on every beat once the camera walks (A17), very small
    w = float(smoothstep(T_SEEN - 10, T_SEEN + 30, t)) * (1.0 - float(smoothstep(T_END - 40, T_END - 10, t)))
    ph = math.pi * (t - T_SEEN) / BEAT
    y = y + 0.07 * w * abs(math.sin(ph))
    az = az + 0.0012 * w * math.sin(0.5 * ph)
    pos = _polar(r, az, y)
    tgt = np.array([0.0, ty, 0.0])
    # the heart is centred for the match cut: the target settles on it
    k_h = float(smootherstep(T_HEART - 20, T_END - 30, t))
    tgt = lerp(tgt, heart_pos(t), k_h)
    focus = float(np.linalg.norm(heart_pos(t) - pos))
    ap = lerp(0.03, 0.05, float(smoothstep(T_SEEN, T_END, t)))
    return Camera(pos, tgt, hfov=hf, focus=focus, aperture=ap)


def render_opts(tl, f):
    return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=45.0, fog_len=70.0, near=0.3)


def finish_opts(tl, f):
    return dict(exposure=1.0, bloom_strength=0.15, bloom_threshold=0.7, streak_strength=0.0, vignette_amount=0.25)


# ================================================================ helpers

def _splat_nofog(ctx, *a, **kw):
    """splat without the arena's depth fog (the far ridges are 60-400 units out)"""
    p = ctx.fr.prm
    keep = p.copy()
    p[6] = 1e9
    p[7] = 1e9
    try:
        ctx.fr.splat(*a, **kw)
    finally:
        p[:] = keep


def _hash01(k, s=0):
    k = np.asarray(k, np.int64)
    h = (k * 73856093 + 19349663 * (s + 1)) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def _face_dir(tw, i):
    """unit horizontal vector from tower i's axis toward the fire (the giants: toward each other)"""
    b = tw.base(i)
    d = -np.array([b[0], 0.0, b[2]])
    return d / np.linalg.norm(d)


# ================================================================ the far ridges and their fires

class RidgeWorld:
    """A16/A17: the land round the arena, dark all through the race, and the small fires on its ridges. Clusters of
    one to five fires on ridgelines at every distance and height, with long dark gaps (never a ring). Each fire is a
    short flickering flame with a glow in the air round it; the rock round it catches its light, and the crest shows
    as a thin line of glints only near the fires."""

    # (azimuth from the arena, distance, crest height above the ground, fires)
    CLUSTERS = [(0.30, 150, 34, 3), (0.95, 75, 9, 2), (1.40, 260, 70, 4), (1.62, 95, 14, 1), (1.95, 170, 40, 3),
                (2.55, 320, 95, 2), (3.30, 120, 22, 4), (3.75, 210, 55, 1), (4.35, 90, 12, 3), (4.62, 280, 78, 5),
                (4.98, 140, 30, 2), (5.30, 360, 110, 3), (5.62, 65, 7, 1), (6.05, 230, 60, 2)]

    def __init__(self, seed=4401):
        r = rng(seed)
        P, S, PH, D = [], [], [], []
        crest, cE = [], []
        for ca, cd, ch, nf in self.CLUSTERS:
            ca = ca + r.normal(0, 0.03)
            spread = 0.03 + 0.05 * r.random()
            # the crest: a knife-edge ridgeline through the cluster (sharp peaks and notches)
            aa = np.linspace(ca - 5 * spread - 0.06, ca + 5 * spread + 0.06, 90)
            u = (aa - ca) / (5 * spread + 0.06)
            ridge = ch * (1.0 - 0.45 * u * u) + 0.08 * ch * (np.abs(np.sin(aa * 57.0 + ca)) - 0.5) \
                + 0.05 * ch * np.sin(aa * 133.0 + 2 * ca)
            dd = cd * (1.0 + 0.04 * np.sin(aa * 23.0 + ca))
            cp = np.stack([dd * np.cos(aa), B.GROUND + ridge, dd * np.sin(aa)], 1)
            for j in range(nf):
                a = ca + r.normal(0, spread)
                jj = int(np.clip(np.searchsorted(aa, a), 0, len(aa) - 1))
                p = cp[jj] + np.array([0.0, -r.uniform(0.0, 0.12) * ch * (j > 0), 0.0])   # on the crest or a shoulder
                P.append(p)
                S.append(r.uniform(0.6, 1.8) * (0.5 if j > 0 and r.random() < 0.4 else 1.0))
                PH.append(r.uniform(0, 2 * np.pi))
                D.append(cd)
            crest.append(cp)
            cE.append(np.full(len(cp), 1.0))
        self.p = np.array(P)
        self.s = np.array(S)
        self.ph = np.array(PH)
        self.d = np.array(D)
        self.n = len(self.p)
        self.crest = np.concatenate(crest)
        # the flames: a few dozen particles each in a short tongue
        m = 36
        self.fl = r.normal(0, 1, (self.n, m, 3)) * np.array([0.28, 1.0, 0.28])
        self.fl[:, :, 1] = np.abs(self.fl[:, :, 1])
        # the rock round each fire: points scattered on the slope below and beside it, lit by it
        k = 70
        off = r.normal(0, 1, (self.n, k, 3)) * np.array([1.0, 0.35, 1.0])
        off[:, :, 1] = -np.abs(off[:, :, 1])
        self.rock = off
        self.rockE = r.lognormal(0, 0.6, (self.n, k))
        # the backs' light: each fire as a direction from the arena, weighted (a compressed falloff)
        v = self.p - np.array([0.0, 10.0, 0.0])
        self.dirs = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.w = self.s * (90.0 / self.d) ** 1.3

    def flicker(self, t):
        return 0.78 + 0.22 * np.sin(1.3 * t + self.ph) * np.sin(0.37 * t + 2.0 * self.ph)

    def emit(self, ctx):
        t = ctx.t
        k = float(smoothstep(T_LIGHT - 2, T_LIGHT + 12, t))
        f = self.flicker(t)
        sz = self.s * (0.45 + 0.004 * self.d)                         # far fires a little larger (they are bonfires)
        m = self.fl.shape[1]
        Pf = (self.p[:, None, :] + self.fl * sz[:, None, None]).reshape(-1, 3)
        dist = np.repeat(self.d, m)
        e = np.repeat(self.s * f * k, m) * np.exp(-self.fl[:, :, 1].reshape(-1) / 1.1) * 30.0 * (60.0 / dist) ** 1.25
        col = look.blackbody(np.clip(0.52 + 0.14 * np.exp(-self.fl[:, :, 1].reshape(-1)), 0, 1))
        _splat_nofog(ctx, Pf, Pf, 0.1 * np.repeat(sz, m), e, col, ctx.cam0, ctx.cam1, zref=0.0)
        # a glow in the air round each fire
        G = self.p + np.array([0.0, 0.6, 0.0]) * sz[:, None]
        eg = self.s * f * k * 900.0 * (60.0 / self.d) ** 1.25
        _splat_nofog(ctx, G, G, 2.6 * sz, eg, C_RIDGE, ctx.cam0, ctx.cam1, profile=1, zref=0.0, rmax=200.0)
        # the rock it lights
        kk = self.rock.shape[1]
        Pr = (self.p[:, None, :] + self.rock * (5.0 * sz)[:, None, None]).reshape(-1, 3)
        rd = np.linalg.norm(self.rock, axis=2).reshape(-1)
        er = (self.rockE * np.repeat((self.s * f * k)[:, None], kk, 1)).reshape(-1) * np.exp(-rd / 1.2) \
            * 5.0 * (60.0 / np.repeat(self.d, kk)) ** 1.25
        _splat_nofog(ctx, Pr, Pr, 0.12 * np.repeat(sz, kk), er, look.blackbody(0.42), ctx.cam0, ctx.cam1, zref=0.0)
        # the crest: glints only near the fires
        near = np.zeros(len(self.crest))
        for j in range(self.n):
            dj = np.linalg.norm(self.crest - self.p[j], axis=1)
            near = np.maximum(near, self.s[j] * f[j] * np.exp(-dj / (0.05 * self.d[j] + 6.0)))
        dc = np.linalg.norm(self.crest, axis=1)
        ec = near * k * 3.0 * (60.0 / dc) ** 1.25
        _splat_nofog(ctx, self.crest, self.crest, 0.08, ec, look.blackbody(0.45), ctx.cam0, ctx.cam1, zref=0.0)

    def irradiance(self, N, t):
        """the far fires' light on surfaces of normals N (a warm key from every cluster; the backs catch it)"""
        w = self.w * self.flicker(t)
        return (np.clip(N @ self.dirs.T.astype(N.dtype), 0.0, 1.0) * w.astype(N.dtype)).sum(1)


# ================================================================ light on the towers' crust

class TowerLight:
    """A16/A17: light the forge-stacks never had. Their backs catch the far fires (irregular: brighter toward the
    clusters); once a tower opens, its light falls on the others' faces, and the two giants, black toward each other
    all through the race, are lit by each other for the first time. The light finds the STRUCTURE (the approved
    charcoal look): edges, bands, seams and joints catch it as thin warm lines, the crust only an even, dim sheen
    (a lit crust with any grain reads as leopard print). Drawn as its own layer over the towers' points."""

    # per kind: (gain on the far fires' light, gain on the open towers' light, splat width, min geo term, face lo, hi)
    KINDS = {0: (0.018, 0.035, None, 0.05, -0.02, 0.1),
             2: (0.30, 0.55, 0.03, 0.6, -0.5, -0.1),
             1: (0.22, 0.45, 0.022, 0.25, -0.02, 0.12),
             4: (0.16, 0.35, 0.03, 0.15, -0.02, 0.12)}

    def emit(self, ctx, tl, rw):
        tw = tl.towers
        if tw.cur is None:
            return
        t = ctx.t
        cam = ctx.cam
        cpos = cam.pos
        fwd = cam.R[2]
        fpx = cam.f_px(1920)
        kb = float(smoothstep(T_LIGHT - 2, T_LIGHT + 16, t))
        # the open facades as vertical line lights: samples up each opened tower's fire-facing face
        src, srcI, srcN, srcT = [], [], [], []
        for i in range(8):
            if t < t_open(i) or tw.cur[i] is None:
                continue
            d = _face_dir(tw, i)
            b = tw.base(i)
            top = B.GROUND + tw.height(i, t)
            ys = np.linspace(B.GROUND + 3.0, top - 3.0, 7)
            hk = np.clip((t - t_open(i) - 0.32 * (ys - B.GROUND) * (1.0 if i in GIANTS else 0.5)) / 8.0, 0.0, 1.0)
            for y_, k_ in zip(ys, hk):
                if k_ > 0:
                    src.append(np.array([b[0], y_, b[2]]) + d * 3.2)
                    srcI.append(k_ * (1.6 if i in GIANTS else 1.0))
                    srcN.append(d)
                    srcT.append(i)
        src, srcI, srcN, srcT = np.array(src), np.array(srcI), np.array(srcN), np.array(srcT)
        for i in range(8):
            c = tw.cur[i]
            if c is None:
                continue
            G = tw.G[i]
            other = srcT != i if len(src) else np.zeros(0, bool)
            S_ = src[other].astype(np.float32) if other.any() else None
            if S_ is not None:
                I_ = srcI[other].astype(np.float32)
                D_ = srcN[other].astype(np.float32)
            for kind, (gb, go, rwk, gmin, f0, f1) in self.KINDS.items():
                pt = c['parts'][kind]
                idx = pt['idx']
                if len(idx) == 0:
                    continue
                P, N = pt['P'], pt['N']
                V = cpos[None, :] - P
                dist = np.linalg.norm(V, axis=1)
                ndv = (N * V).sum(1) / np.maximum(dist, 1e-6)
                yl = P[:, 1] - B.GROUND
                face = smoothstep(f0, f1, ndv) * smoothstep(-0.3, 0.4, yl)
                vis = face > 0.01
                if not vis.any():
                    continue
                P, N, ndv, face = P[vis], N[vis], ndv[vis], face[vis]
                sel = idx[vis]
                rnd = G[kind]['rnd'][sel]
                if kind == 0:
                    grain = 0.9 + 0.2 * rnd                         # an even sheen: no blotches at any scale
                else:
                    grain = (0.55 + 0.9 * G[kind]['nz2'][sel]) * (0.8 + 0.4 * rnd)
                Nf = N.astype(np.float32)
                Eb = rw.irradiance(Nf, t) * (gb * kb) * grain
                colE = np.outer(Eb, C_RIDGE)
                if S_ is not None:
                    Eo = np.zeros(len(P), np.float32)
                    for c0 in range(0, len(P), 20000):
                        Pc = P[c0:c0 + 20000].astype(np.float32)
                        Nc = Nf[c0:c0 + 20000]
                        L = S_[None, :, :] - Pc[:, None, :]
                        dL = np.sqrt((L * L).sum(2))
                        Lh = L / np.maximum(dL, 1e-6)[:, :, None]
                        lam = np.clip((Nc[:, None, :] * Lh).sum(2), 0, 1) if kind != 2 else 0.5 + 0.5 * np.clip(
                            (Nc[:, None, :] * Lh).sum(2), -1, 1)
                        emi = np.clip(-(D_[None, :, :] * Lh).sum(2), 0, 1) ** 0.7
                        Eo[c0:c0 + 20000] = (lam * emi * I_[None, :] / (1.0 + (dL / 24.0) ** 2)).sum(1)
                    colE = colE + np.outer(Eo * go * grain, C_WIN)
                colE = colE * face[:, None]
                z = np.maximum((P - cpos[None, :]) @ fwd, 0.3)
                a = G[kind]['a'][sel] / pt['q']
                geo = np.clip(ndv, gmin, 1.0) if kind != 2 else np.full(len(P), gmin)
                pa = a * geo * (fpx / z) ** 2
                colE = colE * pa[:, None]
                E = colE.max(1)
                m = E > 1e-7
                if not m.any():
                    continue
                rwid = np.sqrt(a / np.pi) * 1.7 if rwk is None else np.full(len(P), rwk)
                ctx.fr.splat(pt['P0'][vis][m], pt['P1'][vis][m], rwid[m], E[m], colE[m] / E[m][:, None], ctx.cam0,
                             ctx.cam1, zref=0.0, myid=i, profile=1)


# ================================================================ the shutters

class Shutters:
    """A16: the windows on the faces turned to the fire (the giants: to each other) open from the bottom up in a
    rising wave: each shutter swings open over a few frames with a breath of flare, then burns steady (a hearth
    inside, not a screen). Every window cell opens, lit or not in the race."""

    def emit(self, ctx, tl):
        tw = tl.towers
        if tw.cur is None:
            return
        t = ctx.t
        cam = ctx.cam
        cpos = cam.pos
        fwd = cam.R[2]
        fpx = cam.f_px(1920)
        for i in range(8):
            if t < t_open(i) - 1 or tw.cur[i] is None:
                continue
            d = _face_dir(tw, i)
            G = tw.G[i]
            # a tower with no windows at all (the ALT's obelisk) opens along its great joints instead
            blind = len(G[3]['p']) == 0 and not (G[0]['key'] >= 0).any()
            for kind in ((4, 1) if blind else (0, 3)):
                pt = tw.cur[i]['parts'][kind]
                idx = pt['idx']
                if len(idx) == 0:
                    continue
                key = G[kind]['key'][idx]
                win = key >= 0
                if blind:
                    key = (np.floor(G[kind]['p'][idx][:, 1] / 3.0)).astype(np.int64) + 1000   # one course per 'window'
                    win = np.ones(len(idx), bool)
                if kind == 0 and not win.any():
                    continue
                P, N = pt['P'], pt['N']
                facing = smoothstep(0.12, 0.5, N[:, 0] * d[0] + N[:, 2] * d[2])
                sel = win & (facing > 0.01) & (P[:, 1] > B.GROUND)
                if not sel.any():
                    continue
                P, N, facing, key = P[sel], N[sel], facing[sel], key[sel]
                V = cpos[None, :] - P
                dist = np.linalg.norm(V, axis=1)
                ndv = (N * V).sum(1) / np.maximum(dist, 1e-6)
                ok = ndv > 0.0
                if not ok.any():
                    continue
                y = P[:, 1] - B.GROUND
                jit = 6.0 * (_hash01(key, 3 + i) - 0.5)
                tw_ = t_open(i) + 0.32 * y * (1.0 if i in GIANTS else 0.5) + jit
                x = t - tw_
                sw = smoothstep(0.0, 5.0, x)
                flare = 1.0 + 0.9 * np.exp(-np.maximum(x, 0.0) / 6.0) * (x > 0)
                lvk = 0.55 + 0.45 * _hash01(key, 11)
                wf = 1.0 + 0.1 * np.sin(0.21 * t + 6.2832 * _hash01(key, 7)) * np.sin(0.07 * t + 3.0 * _hash01(key, 5))
                L = 2.4 * sw * flare * lvk * wf * facing * (1.0 if kind == 0 else 1.25)
                col = look.blackbody(np.clip(0.6 + 0.08 * _hash01(key, 13) + 0.05 * (flare - 1.0), 0, 1))
                z = np.maximum((P - cpos[None, :]) @ fwd, 0.3)
                a = G[kind]['a'][idx][sel] / pt['q']
                pa = a * np.clip(ndv, 0.08, 1.0) * (fpx / z) ** 2
                colE = col * (L * pa * ok)[:, None]
                E = colE.max(1)
                m = E > 1e-7
                if not m.any():
                    continue
                rwid = np.sqrt(a / np.pi) * (1.5 if kind == 0 else 1.15)
                ctx.fr.splat(pt['P0'][sel][m], pt['P1'][sel][m], rwid[m], E[m], colE[m] / E[m][:, None], ctx.cam0,
                             ctx.cam1, zref=0.0, myid=i, profile=1)


# ================================================================ the light crossing the pit

class Beams:
    """A16: the light out of the open shutters, seen in the pit's hot air: from the two giants it crosses the pit past
    the fire to the other (reaching it in about a third of a beat); from each smaller forge, once it opens, a softer
    one toward the fire. Soft drifting motes in the haze, never rays."""

    def __init__(self, seed=4403):
        r = rng(seed)
        n = 26000
        self.s = r.random(n)                 # along the beam (0 at the facade)
        self.u = r.normal(0, 1, n)           # across
        self.v = r.random(n)                 # up the facade
        self.E = r.lognormal(0, 0.5, n)
        self.rw = r.uniform(0.5, 1.3, n)

    def emit(self, ctx, tl):
        tw = tl.towers
        t = ctx.t
        for i in range(8):
            o = opened(i, t)
            if t < t_open(i) or tw.cur[i] is None:
                continue
            d = _face_dir(tw, i)
            side = np.array([-d[2], 0.0, d[0]])
            b = tw.base(i)
            h = tw.height(i, t)
            gi = i in GIANTS
            L = 30.0 if gi else 13.0
            y0, y1 = B.GROUND + 1.0, B.GROUND + (h - 4.0 if gi else min(h - 4.0, 30.0))
            yy = y0 + (y1 - y0) * self.v
            x = t - t_open(i) - 0.32 * (yy - B.GROUND) * (1.0 if gi else 0.5)
            reach = np.clip(x * 4.0, 0.0, L)                  # the light's front crosses in ~8 frames
            ss = self.s * L
            on = (ss < reach) & (x > 0)
            if not on.any():
                continue
            wid = 2.4 + 0.12 * ss
            P = (b + d * 3.0)[None, :] + d[None, :] * ss[:, None] + side[None, :] * (self.u * wid)[:, None]
            P[:, 1] = yy
            drift = np.array([0.0, 0.02 * t, 0.0])
            hz = 0.5 + 0.5 * np.clip(vnoise(P * 0.18 + drift, 1.0, (3.0, 1.0, 7.0), 2)[:, 0] * 1.6, -1, 1)
            e = self.E * hz * np.exp(-0.5 * self.u ** 2) * (1.0 - 0.55 * self.s) * (6.0 if gi else 2.0) \
                * np.clip(x / 10.0, 0, 1) * on
            ctx.fr.splat(P[on], P[on], self.rw[on], e[on], C_WIN * 0.85 + np.array([0.15, 0.13, 0.1]), ctx.cam0,
                         ctx.cam1, profile=1, zref=30.0)


# ================================================================ A17: the small lights

class SmallLights:
    """A17: small lights come down to the rim, from the towers and from the hills alike. From the towers they come
    down the fire-facing faces from the open windows (a steady descent) and step to the lip; from the hills they
    walk down in loose strings from the far fires across the plain. At the rim they stop, in knots and gaps (never
    an even ring), and hold, bobbing a little."""

    def __init__(self, tl, rw, seed=4405):
        r = rng(seed)
        tw = tl.towers
        paths, t0, spd, E = [], [], [], []
        # from the towers: ~9 each, from the window heights down the face to the lip
        for i in range(8):
            d = _face_dir(tw, i)
            side = np.array([-d[2], 0.0, d[0]])
            b = tw.base(i)
            h = tw.height(i, 4700.0)
            for j in range(9 if i not in GIANTS else 12):
                s_ = r.uniform(-2.2, 2.2)
                top = B.GROUND + r.uniform(4.0, min(h - 4.0, 40.0))
                p0 = b + d * 3.4 + side * s_
                p0 = np.array([p0[0], top, p0[2]])
                p1 = np.array([p0[0], B.GROUND + 0.9, p0[2]])
                a = math.atan2(p0[2], p0[0]) + r.normal(0, 0.05)
                rr = edge.rim_r(np.array([a]))[0] + 0.6
                p2 = np.array([rr * math.cos(a), B.GROUND + 0.9, rr * math.sin(a)])
                paths.append([p0, p1, p2])
                t0.append(t_open(i) + r.uniform(40.0, 230.0))
                spd.append(r.uniform(0.18, 0.3))
                E.append(r.lognormal(0, 0.3))
        # from the hills: strings from the near fires, walking across the plain
        near = [j for j in range(rw.n) if rw.d[j] < 130]
        for j in near:
            pf = rw.p[j]
            a0 = math.atan2(pf[2], pf[0])
            for q in range(10):
                a = a0 + r.normal(0, 0.06)
                rr = edge.rim_r(np.array([a]))[0] + 0.6 + r.uniform(0, 1.5)
                p2 = np.array([rr * math.cos(a), B.GROUND + 0.9, rr * math.sin(a)])
                p0 = pf + np.array([r.normal(0, 1.5), -1.0, r.normal(0, 1.5)])
                mid = p0 * 0.45 + p2 * 0.55
                mid[1] = B.GROUND + 0.9 + 0.1 * (p0[1] - B.GROUND)
                paths.append([p0, mid, p2])
                t0.append(T_OPEN_G + 30.0 + 26.0 * q + r.uniform(0, 18.0))
                spd.append(r.uniform(0.34, 0.5))
                E.append(r.lognormal(0, 0.3))
        self.paths = np.array(paths)
        self.t0 = np.array(t0)
        self.spd = np.array(spd)
        self.E = np.array(E)
        seg = np.linalg.norm(np.diff(self.paths, axis=1), axis=2)
        self.seg = seg
        self.len = seg.sum(1)
        self.ph = r.uniform(0, 2 * np.pi, len(self.E))

    def pos(self, tq):
        dist = np.clip((tq - self.t0) * self.spd, 0.0, self.len)
        # ease into the stop at the rim
        u = dist / np.maximum(self.len, 1e-6)
        u = np.where(u > 0.85, 0.85 + 0.15 * smoothstep(0.85, 1.0, u), u)
        dist = u * self.len
        a = self.seg[:, 0]
        k1 = np.clip(dist / np.maximum(a, 1e-6), 0, 1)
        k2 = np.clip((dist - a) / np.maximum(self.seg[:, 1], 1e-6), 0, 1)
        p = np.where((dist <= a)[:, None], self.paths[:, 0] + (self.paths[:, 1] - self.paths[:, 0]) * k1[:, None],
                     self.paths[:, 1] + (self.paths[:, 2] - self.paths[:, 1]) * k2[:, None])
        bob = 0.06 * np.sin(0.31 * tq + self.ph) * (dist > 0)
        return p + np.stack([np.zeros_like(bob), bob, np.zeros_like(bob)], 1), dist > 0

    def emit(self, ctx):
        t = ctx.t
        P0, _ = self.pos(ctx.t0)
        P1, on = self.pos(ctx.t1)
        if not on.any():
            return
        fl = 0.85 + 0.15 * np.sin(0.9 * t + self.ph) * np.sin(0.23 * t + 2 * self.ph)
        e = self.E * fl * 16.0 * on
        _splat_nofog(ctx, P0[on], P1[on], 0.05, e[on], C_LAMP, ctx.cam0, ctx.cam1, zref=12.0)
        _splat_nofog(ctx, P1[on], P1[on], 0.55, e[on] * 6.0, C_LAMP * 0.8, ctx.cam0, ctx.cam1, profile=1,
                     zref=12.0)


# ================================================================ A17: the heart

class Heart:
    """A17 bar 61: the fire gathers into one small, intense heart (THE CROSSING's lantern heart at 4880): an ice-white
    core, a glow and a faint halo, at the crossing's sizes; the flame's own filaments and motes draw in round it."""

    def emit(self, ctx):
        t = ctx.t
        g = gather(t)
        if g <= 0.0:
            return
        H = heart_pos(t)
        cam = ctx.cam
        dz = float(np.linalg.norm(H - cam.pos))
        f = cam.f_px(1920)
        px = dz / f                                   # world units per full-res pixel at the heart
        br = 1.0 + 0.1 * math.sin(2 * math.pi * (t - A.T_IGN) / 80.0 - math.pi / 2)
        k = g ** 1.5 * br
        sig = np.array(HEART_PX) * px
        peak = np.array([30.0, 0.9, 0.05])
        Ee = 2 * np.pi * (np.array(HEART_PX) ** 2) * peak * k
        Hs = np.repeat(H[None, :], 3, 0)
        # r ~ 2 sigma for the gaussian splat
        ctx.fr.splat(Hs, Hs, 2.0 * sig, Ee, np.repeat(C_HEART[None, :], 3, 0), ctx.cam0, ctx.cam1, profile=1,
                     zref=0.0, rmax=520.0, occ=False)


# ================================================================ emit

def _get(tl, name, make):
    return tl._get(name, make)


def emit_haze(ctx, wl):
    """a3.emit_haze with the fire's light on the world (it draws back into the heart)"""
    t = ctx.t
    C = B.crown_centre(t) + np.array([0.0, 2.0 * A.SCHED.fire_scale(t), 0.0])
    pw = B.fire_power(t)
    col = B.C_GOLD * 0.55 + np.array([1.0, 0.62, 0.32]) * 0.45
    red = B.redness(t)
    col = col * (1 - 0.7 * red) + B.C_RED * 0.7 * red
    H = C[None, :]
    ctx.fr.splat(H, H, np.array([6.0]), np.array([900.0]) * pw ** 0.5 * wl, col[None, :], ctx.cam0, ctx.cam1,
                 profile=1)


def emit(tl, ctx):
    t = ctx.t
    ctx.tl = tl
    lp, lc, lpw = tl.light(t)
    wl = world_light(t)
    lpw = lpw * wl
    cr = edge._crater(tl)
    ctx.tl_crater = cr
    tl.towers.prepare(ctx)
    tl.dust.emit(ctx)
    tl.smoke.emit(ctx, lp, lc, lpw)
    cr.emit(ctx, lp)
    tl.towers.emit(ctx, lp, lc, lpw)
    rw = _get(tl, 't_ridge', RidgeWorld)
    _get(tl, 't_light', TowerLight).emit(ctx, tl, rw)
    _get(tl, 't_shutters', Shutters).emit(ctx, tl)
    tl.tembers.emit(ctx)
    tl.tsmoke.emit(ctx, lp, lc, lpw)
    tl.sparks.emit(ctx)
    g = gather(t)
    keep_t, keep_d = list(B.FLAME_TONGUES), B.MIND_TONGUE_DIM
    try:
        if g > 0:
            th, H, W, lean, om, ph = B.FLAME_TONGUES[0]
            B.FLAME_TONGUES[0] = (th, H * (1.0 - 0.72 * g), W * (1.0 - 0.3 * g), lean * (1.0 - g), om, ph)
            B.MIND_TONGUE_DIM = keep_d + (1.0 - keep_d) * g
        tl.fire.emit(ctx)
    finally:
        B.FLAME_TONGUES[:] = keep_t
        B.MIND_TONGUE_DIM = keep_d
    emit_haze(ctx, wl)
    if t < T_HEART + 30:
        tl.fsparks.emit(ctx)
    tl._get('gold_runs', edge.GoldRuns).emit(ctx, tl)
    tl._get('pit_embers', edge.PitEmbers).emit(ctx, 0.8 - 0.5 * float(smoothstep(T_STOP, T_HOLDS, t)))
    rw.emit(ctx)
    _get(tl, 't_beams', Beams).emit(ctx, tl)
    if t >= T_OPEN_G + 20:
        _get(tl, 't_lamps', lambda: SmallLights(tl, rw)).emit(ctx)
    if t >= T_HEART:
        _get(tl, 't_heart', Heart).emit(ctx)


def post(tl, ctx, hdr):
    return edge.post(tl, ctx, hdr)
