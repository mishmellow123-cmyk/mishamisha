"""EMBERS part A (300-480): rising embers -> glyphs -> spiral -> the point."""
import os

import numpy as np

import look
from core import (clamp01, smoothstep, smootherstep, ease_out, ease_in, lerp, vnoise,
                  rng, rand_dirs, catmull)
import glyphs as G

CACHE = os.path.join(os.path.dirname(__file__), '..', '..', 'renders', 'embers', 'cache')

C_CORE = look.hexrgb(look.PALETTE['mind_core'])
C_ICE = look.hexrgb(look.PALETTE['mind_ice'])
C_GOLD = look.hexrgb(look.PALETTE['mind_gold'])
C_EMBER = look.hexrgb(look.PALETTE['ember'])

EMITTER = np.array([0.0, -1.3, 2.4])


def _warp_table():
    """Global time warp for the embers: s(t) = integral of rate; they slow as they become letters."""
    ts = np.arange(200.0, 520.0, 0.05)
    rate = 1.0 - 0.82 * smootherstep(308.0, 344.0, ts)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * 0.05)])
    return ts, s


_WT, _WS = _warp_table()


def warp(t):
    return np.interp(t, _WT, _WS)


class Embers:
    """Pure embers from the torch (continuity with INTRO), 240..370."""

    def __init__(self, n=9000, seed=1):
        r = rng(seed)
        self.n = n
        self.tb = r.uniform(236.0, 324.0, n)                  # birth frame
        self.sb = warp(self.tb)
        a = r.uniform(0, 2 * np.pi, n)
        rad = 0.35 * np.sqrt(r.random(n))
        self.p0 = EMITTER + np.stack([rad * np.cos(a), r.normal(0, 0.1, n), rad * np.sin(a)], 1)
        sp = r.uniform(0.16, 0.34, n)                        # units / frame (warped time)
        dirs = np.stack([r.normal(0, 0.24, n), np.ones(n), r.normal(0.26, 0.2, n)], 1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        self.v = dirs * sp[:, None]
        self.T0 = r.uniform(0.62, 0.95, n)
        self.E0 = r.lognormal(0.0, 0.6, n) * 22.0
        self.life = r.uniform(70, 150, n)                     # in warped frames
        self.flk = r.uniform(0.15, 0.5, n)
        self.ph = r.uniform(0, 2 * np.pi, n)
        self.wamp = r.uniform(0.4, 1.3, n)
        self.rw = r.uniform(0.008, 0.024, n)

    def pos(self, t):
        s = warp(t)                       # scalar or per-particle array
        age = np.maximum(s - self.sb, 0.0)
        p = self.p0 + self.v * age[:, None] + np.array([0, 0.0012, 0]) * (age ** 2)[:, None]
        s_arr = np.broadcast_to(s, age.shape)
        q = p + np.stack([0 * s_arr, -0.02 * s_arr, 0.013 * s_arr], 1) / 0.33
        w = vnoise(q, 0.33, (0.0, 0.0, 0.0), 2)
        p = p + w * (self.wamp * np.minimum(age / 40.0, 1.0) ** 1.5)[:, None]
        return p, age

    def emit(self, ctx):
        t = ctx.t
        if t > 372 or t < 236:
            return
        p0, _ = self.pos(ctx.t0)
        p1, age = self.pos(ctx.t1)
        alive = (t >= self.tb) & (age < self.life)
        x = np.clip(age / self.life, 0, 1)
        temp = self.T0 * (1 - 0.55 * x)
        fl = 1 + self.flk * np.sin(0.9 * t + self.ph) * np.sin(0.37 * t + 2 * self.ph)
        e = self.E0 * np.clip((t - self.tb) / 3.0, 0, 1) * (1 - x) ** 0.8 * fl
        e *= 1 - smoothstep(335, 372, t)          # the embers die away once letters have formed
        e = np.where(alive, e, 0.0)
        col = look.blackbody(temp)
        ctx.fr.splat(p0, p1, self.rw, e, col, ctx.cam0, ctx.cam1)


class TorchGlow:
    """The warm glow of the torch flame below frame (fades out)."""

    def emit(self, ctx):
        t = ctx.t
        k = 1 - smoothstep(300, 334, t)
        if k <= 0:
            return
        P = np.array([EMITTER + [0, -0.9, 0], EMITTER + [0, -0.2, 0.2], EMITTER + [0, -1.6, 0]])
        e = np.array([30000.0, 9000.0, 40000.0]) * k
        col = np.array([look.blackbody(0.75), look.blackbody(0.6), look.blackbody(0.85)])
        rw = np.array([2.2, 3.2, 1.2])
        ctx.fr.splat(P, P, rw, e, col, ctx.cam0, ctx.cam1, profile=1)


# ------------------------------------------------------------------ glyphs ---

class Glyphs:
    """Thousands of glyphs: born from embers / drifting in; spiral; compress to a point."""

    def __init__(self, n=4200, n_ember=380, seed=3):
        r = rng(seed)
        A = G.load(os.path.join(CACHE, 'glyphs.npz'))
        self.A = A
        ng = len(A['off'])
        catw = np.array([G.CAT_W.get(G.CATS[c], 1.0) for c in A['cat']], np.float64)
        # per-category normalisation so each category's total weight = CAT_W
        cat_count = np.bincount(A['cat'], minlength=len(G.CATS)).astype(np.float64)
        pw = catw / cat_count[A['cat']]
        pw /= pw.sum()
        self.n = n
        self.proto = r.choice(ng, size=n, p=pw)
        self.hero = np.zeros(n, bool)
        # hero instances: pick from hero-flagged prototypes, placed near the camera path
        heroes = np.nonzero(A['hero'])[0]
        nh = 30
        self.proto[:nh] = r.choice(heroes, nh, replace=False)
        self.hero[:nh] = True
        texts = list(A['text'])
        self.passes = [p for p in HERO_PASSES if p[0] in texts]
        for i, p in enumerate(self.passes):
            self.proto[i] = texts.index(p[0])
        # sizes (em in world units)
        s = r.lognormal(np.log(0.24), 0.3, n)
        s[:nh] = r.uniform(0.3, 0.46, nh)
        self.s = s
        # ---- home positions: uniform in a big flattened ellipsoid (deep field of letters)
        d = rand_dirs(r, n)
        rad = 34.0 * r.random(n) ** (1 / 3.0)
        home = d * rad[:, None] * np.array([1.3, 0.55, 1.0])
        home[:, 1] += 2.0
        # keep the lens clear: push non-heroes away from the camera corridor
        corr = np.array([0.6, 1.0, 8.2])
        dv = home - corr
        dd = np.linalg.norm(dv, axis=1)
        push = dd < 4.0
        home[push] = corr + dv[push] / dd[push, None] * (4.0 + 3.0 * r.random(push.sum()))[:, None]
        # heroes: in front of the camera corridor, above the text band
        hz = r.uniform(1.0, 5.2, nh)
        hx = r.choice([-1, 1], nh) * r.uniform(0.7, 3.4, nh)
        hy = r.uniform(1.3, 3.6, nh)
        home[:nh] = np.stack([hx, hy, hz], 1)
        # passes: placed so they are exactly in focus (d = HERO_FOCUS) at their moment
        hv = []
        for i, (txt, th, sx, sy, sz) in enumerate(self.passes):
            pc, tc = cam_a(th)
            fw = (tc - pc) / np.linalg.norm(tc - pc)
            rt = np.cross(fw, [0, 1.0, 0])
            rt /= np.linalg.norm(rt)
            upv = np.cross(rt, fw)
            tn = np.tan(np.radians(HFOV_A(th)) / 2)
            d = fw + sx * tn * rt + sy * tn * upv
            home[i] = pc + d / np.linalg.norm(d) * HERO_FOCUS / np.dot(d / np.linalg.norm(d), fw)
            s[i] = sz
            # velocity: pass the lens on its own side ~26 frames after the moment
            pe, te = cam_a(th + 26)
            fe = (te - pe) / np.linalg.norm(te - pe)
            re = np.cross(fe, [0, 1.0, 0])
            re /= np.linalg.norm(re)
            ue = np.cross(re, fe)
            goal = pe + np.sign(sx) * 1.7 * re + 1.0 * ue + 0.6 * fe
            hv.append((goal - home[i]) / 26.0)
        self.hv = np.array(hv).reshape(-1, 3)
        self.th = np.array([p[1] for p in self.passes], np.float64)
        self.home = home
        # ---- arrival (drift in from all directions)
        self.arr = r.uniform(318.0, 392.0, n)
        self.arr_len = r.uniform(40.0, 80.0, n)
        for i, p in enumerate(self.passes):
            self.arr[i] = p[1] - 44.0
            self.arr_len[i] = 20.0
        self.din = rand_dirs(r, n) * r.uniform(6.0, 16.0, n)[:, None]
        self.din += (home / np.maximum(np.linalg.norm(home, axis=1, keepdims=True), 1e-3)) * 6.0
        # ---- ember-born glyphs: follow ember paths first
        self.n_ember = n_ember
        ie = np.arange(nh, nh + n_ember)
        self.ie = ie
        self.eb = Embers(n=n_ember, seed=seed + 100)
        self.eb.tb = r.uniform(278.0, 310.0, n_ember)
        self.eb.sb = warp(self.eb.tb)
        self.eb.v *= 1.15
        self.t_open = np.full(n, -1.0)
        self.t_open[ie] = r.uniform(320.0, 342.0, n_ember)
        self.is_ember = np.zeros(n, bool)
        self.is_ember[ie] = True
        s[ie] = r.lognormal(np.log(0.15), 0.2, n_ember)
        # ---- look
        self.T = r.uniform(0.5, 0.82, n)
        self.T[:nh] = r.uniform(0.62, 0.8, nh)
        self.E = r.lognormal(-0.3, 0.7, n)
        self.tw_f = r.uniform(0.05, 0.22, n)
        self.tw_p = r.uniform(0, 2 * np.pi, n)
        self.tw_a = r.uniform(0.1, 0.55, n)
        self.tilt = r.normal(0, 0.22, (n, 2))
        self.tilt[:nh] *= 0.4
        self.tumf = r.uniform(0.004, 0.02, (n, 2))
        self.tump = r.uniform(0, 2 * np.pi, (n, 2))
        self.wamp = r.uniform(0.15, 0.6, n)
        self.wamp[:nh] = 0.08
        self.din[:len(self.passes)] *= 0.25
        # ---- spiral
        rr = np.linalg.norm(home[:, [0, 2]], axis=1)
        self.sp_start = 396.0 + 12.0 * clamp01(rr / 30.0) + r.uniform(-3, 3, n)
        self.sp_end = 468.0 + r.uniform(-7, 2, n) - 4 * (1 - clamp01(rr / 30.0))
        self.sp_k = r.uniform(1.8, 2.2, n)
        self.arm = r.integers(0, 3, n) * (2 * np.pi / 3) + r.normal(0, 0.18, n)
        # ---- points per glyph
        cnt = A['cnt'][self.proto]
        npt = np.where(s > 0.3, 700, np.where(s > 0.2, 170, 60))
        npt[:nh] = 1100
        textlen = np.array([len(A['text'][p]) for p in self.proto])
        npt = np.where(textlen > 1, (npt * np.minimum(1 + 0.45 * (textlen - 1), 3.2)).astype(int), npt)
        npt = np.minimum(npt, cnt)
        self.npt = npt
        gid = np.repeat(np.arange(n), npt)
        loc = np.concatenate([A['pts'][A['off'][p]:A['off'][p] + k] for p, k in zip(self.proto, npt)])
        self.gid = gid
        self.loc = loc.astype(np.float64)
        self.pe = (1.0 / npt)[gid]                       # energy share per point
        self.prand = r.random(len(gid))
        print('glyphs: instances', n, 'points', len(gid))

    # centre of each glyph before the spiral
    def centre_pre(self, t):
        n = self.n
        ta = np.broadcast_to(np.asarray(t, np.float64), (n,))
        u = ease_out(clamp01((ta - self.arr) / self.arr_len), 3.0)
        c = self.home + self.din * (1 - u)[:, None]
        # ember-born: follow ember path (time-warped, slowing)
        pe, _ = self.eb.pos(ta[self.ie])
        c[self.ie] = pe
        q = self.home + np.stack([0.004 * ta, 0.006 * ta, -0.003 * ta], 1) / 0.21
        w = vnoise(q, 0.21, (0.0, 0.0, 0.0), 2)
        c = c + w * self.wamp[:, None] * 1.3
        npass = len(self.passes)
        if npass:
            c[:npass] = self.home[:npass] + self.hv * (ta[:npass] - self.th)[:, None] + \
                (w[:npass] * 0.06)
        return c

    def state(self, t):
        """Glyph centres, scale, spiral progress at time t."""
        c = self.centre_pre(t)
        # spiral transform
        ts = self.sp_start
        u = ease_in(clamp01((t - ts) / (self.sp_end - ts)), 1.7)
        act = u > 0
        if act.any():
            cs = self.centre_pre_frozen()
            q = cs[act]
            rq = np.linalg.norm(q[:, [0, 2]], axis=1)
            th = np.arctan2(q[:, 2], q[:, 0])
            h = q[:, 1]
            ua = np.minimum(u[act], 0.9995)
            rr = rq * (1 - ua) ** 1.25
            # pull toward 3 logarithmic arms as the spiral forms (galaxy of writing)
            tharm = self.arm[act] + 1.1 * np.log(np.maximum(rq, 0.5))
            dth = np.angle(np.exp(1j * (tharm - th)))
            th = th + dth * smoothstep(0.0, 0.22, ua)
            th2 = th + self.sp_k[act] * np.log(1.0 / (1.0 - 0.995 * ua)) + 0.35 * ua
            hh = h * (1 - smoothstep(0.0, 0.5, ua)) * (1 - ua)
            sp = np.stack([rr * np.cos(th2), hh, rr * np.sin(th2)], 1)
            # residual motion fades out
            resid = c[act] - q
            c[act] = sp + resid * (1 - ua)[:, None] ** 2
        return c, u

    _frozen = None

    def centre_pre_frozen(self):
        if self._frozen is None:
            # each glyph's pre-spiral centre at its own spiral start time
            self._frozen = self.centre_pre(self.sp_start)
        return self._frozen

    def points(self, t, campos):
        c, u = self.state(t)
        # scale: ember-born glyphs open from nothing; spiral shrinks them
        sc = self.s.copy()
        op = self.is_ember
        g = clamp01((t - self.t_open[op]) / 12.0)
        sc[op] *= ease_out(g, 2.0) * 0.98 + 0.02
        sc *= np.maximum((1 - u) ** 1.7, 0.02)
        # billboard basis toward camera with tilt/tumble
        fwd = campos[None, :] - c
        fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-6)
        up0 = np.array([0.0, 1.0, 0.0])
        right = np.cross(up0[None, :], fwd)
        right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-6)
        up = np.cross(fwd, right)
        amp = np.where(self.hero, 0.08, 0.2)
        a = np.clip(self.tilt[:, 0] + amp * np.sin(self.tumf[:, 0] * t + self.tump[:, 0]), -0.42, 0.42)
        a = a + 2.2 * u ** 1.5
        b = np.clip(self.tilt[:, 1] + 1.5 * amp * np.sin(self.tumf[:, 1] * t + self.tump[:, 1]), -0.6, 0.6)
        ca, sa = np.cos(a), np.sin(a)
        r2 = ca[:, None] * right + sa[:, None] * up
        u2 = -sa[:, None] * right + ca[:, None] * up
        cb = np.cos(b)
        r2 = r2 * cb[:, None] + fwd * np.sin(b)[:, None]      # yaw tilt (foreshortening)
        gid = self.gid
        lx = self.loc[:, 0] * sc[gid]
        ly = self.loc[:, 1] * sc[gid]
        P = c[gid] + r2[gid] * lx[:, None] + u2[gid] * ly[:, None]
        return P, u, sc

    def emit(self, ctx):
        t = ctx.t
        if t < 300 or t > 486:
            return
        P0, _, _ = self.points(ctx.t0, ctx.cam0.pos)
        P1, u, sc = self.points(ctx.t1, ctx.cam1.pos)
        gid = self.gid
        # glyph-level brightness
        fade_arr = smoothstep(self.arr, self.arr + 22, t)
        fade_arr[self.is_ember] = smoothstep(self.t_open[self.is_ember] - 4, self.t_open[self.is_ember] + 6, t)
        tw = 1 + self.tw_a * np.sin(self.tw_f * t * 6.28 / 6 + self.tw_p)
        heat = smoothstep(0.55, 1.0, u)
        absorb = 1 - smoothstep(0.9, 0.99, u)
        shrink = np.maximum((1 - u) ** 1.7, 0.02)
        eg = self.E * tw * fade_arr * (1 + 1.5 * heat) * absorb * (0.2 + 0.8 * shrink)
        eg *= 3000.0 * self.s ** 2 * (1 + 0.8 * smoothstep(400, 440, t))  # ~ area (seen at z=10)
        eg[self.hero] *= 1.6
        T = lerp(self.T, 0.86, heat)
        # ember-born glyphs keep ember heat as they open
        col_g = look.blackbody(T)
        # blend toward the mind palette as they compress
        mind = C_CORE * 0.6 + C_ICE * 0.4
        hm = smoothstep(0.8, 1.0, u)[:, None]
        col_g = col_g * (1 - hm * 0.6) + mind[None, :] * hm * 0.6
        e = eg[gid] * self.pe * (0.75 + 0.5 * self.prand)
        col = col_g[gid]
        rw = sc[gid] * 0.018
        ctx.fr.splat(P0, P1, rw, e, col, ctx.cam0, ctx.cam1)
        self.last_u = u


class ThePoint:
    """The blinding point that everything compresses into (430..484)."""

    def __init__(self, glyphs):
        self.g = glyphs
        r = rng(9)
        self.n = 1500
        self.d = rand_dirs(r, self.n) * (r.random(self.n) ** 2)[:, None]
        self.ph = r.uniform(0, 6.28, self.n)

    def level(self, t):
        # fraction of glyph energy absorbed by t (smooth, monotone)
        g = self.g
        u = ease_in(clamp01((t - g.sp_start) / (g.sp_end - g.sp_start)), 1.7)
        return float(np.mean(smoothstep(0.9, 0.995, u)))

    def emit(self, ctx):
        t = ctx.t
        if t < 404 or t >= 484:
            return
        lv = self.level(t)
        breath = 1 - 0.25 * smoothstep(471, 479, t) + 0.6 * smoothstep(478.5, 480.0, t)
        I = 200 + 16000 * lv ** 1.3 * breath
        tremble = 0.02 * np.sin(np.array([37.0, 51.0, 43.0]) * t)
        rad = 0.05 + 0.04 * lv
        P = self.d * rad + tremble
        e = np.full(self.n, I / self.n)
        col = np.broadcast_to(C_CORE, (self.n, 3))
        ctx.fr.splat(P, P, 0.004, e, col, ctx.cam0, ctx.cam1)
        # galactic bulge: grows with the mean spiral progress
        g = self.g
        um = ease_in(clamp01((t - g.sp_start) / (g.sp_end - g.sp_start)), 1.7).mean()
        bul = 1 - smoothstep(472, 481, t)
        Hb = np.zeros((3, 3))
        ctx.fr.splat(Hb, Hb, np.array([4.0 * (1 - um) + 0.5, 2.0 * (1 - um) + 0.3, 0.8]),
                     np.array([9000.0, 6000.0, 3000.0]) * um ** 1.5 * bul,
                     np.array([C_GOLD, look.blackbody(0.8), C_ICE]), ctx.cam0, ctx.cam1, profile=1)
        # halo
        H = np.zeros((2, 3))
        ctx.fr.splat(H, H, np.array([0.3, 1.2]), np.array([1.5, 1.2]) * I,
                     np.array([C_ICE, C_GOLD]), ctx.cam0, ctx.cam1, profile=1)


# ------------------------------------------------------------------ camera ---

CAM_A = [
    (300, (0.0, -1.2, 9.4), (0.0, 2.0, 0.0)),
    (340, (0.2, 0.2, 8.8), (0.0, 2.6, -1.0)),
    (372, (0.5, 1.2, 7.4), (0.2, 2.4, -3.0)),
    (392, (0.9, 2.2, 6.9), (0.2, 1.6, -3.0)),
    (430, (5.0, 14.0, 27.0), (0.0, -1.5, 0.0)),
    (455, (4.0, 10.0, 19.5), (0.0, -1.6, 0.0)),
    (470, (3.1, 7.6, 14.2), (0.0, -1.4, 0.0)),
    (484, (2.9, 7.2, 13.6), (0.0, -1.35, 0.0)),
]

HERO_FOCUS = 5.0
# (text, moment, screen x, screen y (fractions of half-width), em size)
HERO_PASSES = [
    ('火', 346, -0.46, 0.22, 0.36),
    ('\U0001D11E', 353, 0.50, 0.16, 0.36),
    ('كلمة', 360, -0.12, 0.30, 0.30),
    ('π', 367, 0.60, 0.30, 0.34),
    ('ज्ञान', 374, -0.58, 0.10, 0.28),
    ('A', 381, 0.22, 0.26, 0.34),
    ('ACGT', 388, -0.30, 0.33, 0.26),
    ('∞', 395, 0.56, 0.08, 0.34),
]


def HFOV_A(t):
    return float(lerp(50, 46, smoothstep(300, 480, t)))


def cam_a(t):
    pos = catmull(t, [(k[0], k[1]) for k in CAM_A])
    tgt = catmull(t, [(k[0], k[2]) for k in CAM_A])
    return pos, tgt
