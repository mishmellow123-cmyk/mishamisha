"""EMBERS v3, cut A (EMBERS-A2): A3 INTO THE LIGHT . GLYPHS (560-960) and A4 THE POINT (960-1040), on A's frames.

    560  bar 8 b1   the push has gone into the false dawn's glow: the frame is white (ice-white, its colour)
    560-640         the white recedes, drawn back toward the upper middle where the fire will be born
    640  bar 9 b1   the first letters appear in what is left of it: the word for light in many scripts
    660-880         letters of every script drift in from every side (the frame's edges, past the lens, out of the
                    depth) in slow currents, and gather into a deep, uneven cloud of what we wrote. T2 (700-840)
                    sits in the lower third, which stays quiet
    880  bar 12 b1  the cloud begins to turn: one heavy swirl about a tilted axis, seen obliquely, the inner letters
                    faster (they keep their angular momentum), so the cloud shears into streams by itself -- no
                    drawn arms, no bulge, no disc seen face-on (H5: no screensaver). The letters stay letters,
                    shrinking and heating toward the mind's ice-white as they fall in
    960  bar 13 b1  A4: the spiral collapses to a blinding point
    1020 bar 13 b4  one beat of breath: the point holds, trembling
    1040            A5's ignition opens on a soft white disc centred at px (964, 238): the point lands exactly there

The letters are point clouds sampled from real font glyphs (assets/glyphs/atlas_v3.npz, committed: the approved v2
atlas less the Om and the Eye of Horus), splatted as ember-light, like everything in the embers act.
"""
import math
import os

import numpy as np

import look
from core import Camera, smoothstep, smootherstep, lerp, rng, vnoise, ease_out, ease_in, clamp01

T_WHITE = 560
T_LETTERS = 640
T_SPIRAL = 880
T_POINT = 960
T_BREATH = 1020
T_IGN = 1040
POINT_PX = (964.0, 238.0)          # A5 f1040's ignition disc (measured on the approved render)
ATLAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'glyphs', 'atlas_v3.npz')

C_CORE = look.hexrgb(look.PALETTE['mind_core'])
C_ICE = look.hexrgb(look.PALETTE['mind_ice'])
C_GOLD = look.hexrgb(look.PALETTE['mind_gold'])
C_WHITE = np.array([0.80, 0.92, 1.00])              # the false dawn's cold glow (falsedawn.py) and the lantern's heart

# relative frequency of each script among the letters (glyphs.CAT_W, DNA thinned)
CAT_W = {'latin': 12, 'greek': 6, 'cyrillic': 6, 'arabic': 9, 'hebrew': 5, 'devanagari': 7,
         'cjk': 11, 'kana': 3, 'hangul': 5, 'geez': 4, 'tamil': 4, 'math': 6, 'formula': 1.5, 'music': 3,
         'music2': 3, 'dna': 1.5, 'code': 5, 'cuneiform': 1.5, 'hiero': 2, 'runic': 2,
         'phoenician': 1, 'linearb': 1, 'georgian': 1.5, 'armenian': 1.5, 'thai': 1.5,
         'bengali': 1.5, 'tibetan': 1, 'cherokee': 1, 'syllabics': 1, 'tifinagh': 1,
         'khmer': 1, 'sinhala': 1, 'gujarati': 1, 'telugu': 1, 'myanmar': 1, 'yi': 1, 'syriac': 1}
# the first letters, appearing in the white: the word for light (and a few of the oldest marks)
FIRST = ['光', 'نور', 'φῶς', 'אור', 'свет', 'ज्ञान', '빛', 'ብርሃን', 'ஒளி', 'A', 'ᚠ', '∞', '\U0001D11E', 'ع']


def load_atlas():
    z = np.load(ATLAS)
    A = {k: z[k] for k in z.files}
    A['pts'] = A['pts'].astype(np.float32)
    A['text'] = [str(t) for t in A['text']]
    A['cats'] = [str(c) for c in A['cats']]
    return A


# ============================================================================ the camera

CAM_D = [(560, 36.0), (700, 31.0), (880, 26.0), (960, 22.5), (1040, 20.0)]
GATHER_Y = 336.0            # where the gathering sits on screen until the swirl (the lower third stays quiet)


def _dist(t):
    ts = [k[0] for k in CAM_D]
    return float(np.interp(t, ts, [k[1] for k in CAM_D]))


def camera(tl, t):
    """a slow push toward the gathering (continuing the false dawn's push), arcing a little so the cloud's depth
    shows; the gathering point sits at POINT_PX all through, so the point lands where the ignition begins"""
    u = float(np.clip((t - T_WHITE) / (T_IGN - T_WHITE), 0, 1))
    D = _dist(t)
    az = lerp(0.30, -0.06, float(smootherstep(0.0, 1.0, u)))
    el = 0.12 + 0.2 * float(smootherstep(T_SPIRAL - 40, T_POINT + 20, t))      # rising to see the eddy at a low angle
    pos = np.array([D * math.sin(az) * math.cos(el), D * math.sin(el), D * math.cos(az) * math.cos(el)])
    hf = lerp(50.0, 46.0, float(smoothstep(0.55, 1.0, u)))
    cam = Camera(pos, np.zeros(3), hfov=hf, focus=D, aperture=lerp(0.07, 0.035, u))
    # tilt so that the gathering point sits at POINT_PX (above the frame's centre, as the fire will)
    f = cam.f_px(1920)
    py = lerp(GATHER_Y, POINT_PX[1], float(smootherstep(T_SPIRAL + 20, T_POINT + 40, t)))
    a = math.atan((401.5 - py) / f)
    b = math.atan((POINT_PX[0] - 959.5) / f)
    right, up, fwd = cam.R
    fwd2 = fwd * math.cos(a) - up * math.sin(a)
    up2 = up * math.cos(a) + fwd * math.sin(a)
    fwd3 = fwd2 * math.cos(b) - right * math.sin(b)
    right3 = right * math.cos(b) + fwd2 * math.sin(b)
    cam.R = np.stack([right3, up2, fwd3])
    return cam


def render_opts(tl, f):
    return dict(bokeh_pow=0.35, bokeh_cap=2.2, fog_start=60.0, fog_len=60.0, near=0.25)


# ============================================================================ the letters

class Letters:
    def __init__(self, n=2300, seed=8080):
        r = rng(seed)
        A = load_atlas()
        self.A = A
        cats = A['cats']
        ncat = len(cats)
        w = np.array([CAT_W.get(cats[c], 1.0) for c in A['cat']], np.float64)
        cnt = np.bincount(A['cat'], minlength=ncat).astype(np.float64)
        w = w / cnt[A['cat']]
        single = np.array([len(t) == 1 for t in A['text']])
        w = w * np.where(single, 1.0, 0.35)                      # mostly single letters; words are rarer
        w /= w.sum()
        proto = r.choice(len(w), size=n, p=w)
        # the first letters (in the white)
        first = [A['text'].index(x) for x in FIRST if x in A['text']]
        nf = len(first)
        proto[:nf] = first
        self.nf = nf
        s = r.lognormal(math.log(0.3), 0.32, n)
        s[:nf] = r.uniform(0.62, 0.9, nf)
        big = r.random(n) < 0.05                                 # a few larger ones in the cloud
        s[big] *= 1.6
        # ---- the cloud (homes): clumps (phrases that arrived together, on one current) + a diffuse field
        home = np.zeros((n, 3))
        clump = np.full(n, -1)
        ncl = 15
        cc = r.normal(0, 1, (ncl, 3)) * np.array([3.6, 2.5, 3.2])
        cs = r.uniform(0.7, 1.9, ncl)
        idx = np.arange(nf, n)
        inc = r.random(len(idx)) < 0.66
        ci = r.integers(0, ncl, len(idx))
        for j, i in enumerate(idx):
            if inc[j]:
                k = ci[j]
                clump[i] = k
                home[i] = cc[k] + r.normal(0, 1, 3) * cs[k] * np.array([1.25, 0.95, 1.0])
            else:
                d = r.normal(0, 1, 3)
                d /= np.linalg.norm(d)
                home[i] = d * 7.0 * r.random() ** 0.5 * np.array([1.3, 0.86, 1.15])
        # the first letters: spread round the gathering point, near its depth, in focus
        fa = r.permutation(np.linspace(0, 2 * np.pi, nf, endpoint=False)) + r.uniform(-0.35, 0.35, nf)
        fr = r.uniform(0.8, 4.2, nf) ** 1.0
        home[:nf] = np.stack([fr * np.cos(fa) * 1.35, fr * np.sin(fa) * 0.62 + r.normal(0, 0.35, nf),
                              r.uniform(-3.0, 3.0, nf)], 1)
        self.home = home
        # ---- where each comes from: the frame's sides, past the lens (behind the camera), out of the depth
        sides = np.array([[-1, 0.15, 0.1], [1, 0.1, 0.05], [0.15, 1, 0.1], [-0.1, -1, 0.1], [0.25, 0.1, 1.0],
                          [-0.3, 0.15, 1.0], [0.1, 0.1, -1.0], [-0.8, 0.7, -0.2], [0.8, -0.6, 0.3], [0.7, 0.8, 0.4]])
        sides = sides / np.linalg.norm(sides, axis=1, keepdims=True)
        sw = np.array([1.3, 1.3, 0.9, 0.7, 1.0, 1.0, 0.6, 0.8, 0.8, 0.8])
        cl_side = r.choice(len(sides), ncl, p=sw / sw.sum())
        side = r.choice(len(sides), n, p=sw / sw.sum())
        side[clump >= 0] = cl_side[clump[clump >= 0]]
        dirs = sides[side] + r.normal(0, 0.18, (n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        dist = r.uniform(22.0, 38.0, n)
        self.orig = home + dirs * dist[:, None]
        # a sideways bow on each path (they arc in on currents, never straight at the centre)
        b = np.cross(dirs, r.normal(0, 1, (n, 3)))
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
        self.bow = b * (dist * r.uniform(0.08, 0.25, n))[:, None]
        # ---- arrival: currents come in waves from bar 9 b2 to bar 11 b4; a clump's letters travel together
        t_arr = r.uniform(655.0, 850.0, n) ** 1.0
        cl_t = r.uniform(655.0, 820.0, ncl)
        m = clump >= 0
        t_arr[m] = cl_t[clump[m]] + r.normal(0, 10.0, m.sum())
        self.t_arr = t_arr
        self.dur = r.uniform(80.0, 150.0, n)
        self.t_arr[:nf] = T_LETTERS - 14 + np.arange(nf) * 4.0 + r.uniform(0, 4, nf)   # 626-686: appear in place
        self.dur[:nf] = 1.0
        self.orig[:nf] = home[:nf]
        self.bow[:nf] = 0.0
        # ---- look
        self.s = s
        self.T = r.uniform(0.54, 0.8, n)
        self.T[:nf] = r.uniform(0.7, 0.82, nf)
        self.E = r.lognormal(-0.25, 0.55, n)
        self.E[:nf] = 1.6
        self.tilt = r.normal(0, 0.2, (n, 2))
        self.tumf = r.uniform(0.004, 0.018, (n, 2))
        self.tump = r.uniform(0, 2 * np.pi, (n, 2))
        self.tw_f = r.uniform(0.05, 0.2, n)
        self.tw_p = r.uniform(0, 2 * np.pi, n)
        self.tw_a = r.uniform(0.1, 0.4, n)
        self.wamp = r.uniform(0.25, 0.8, n)
        self.wamp[:nf] = 0.2
        # ---- the swirl (bar 12 b1 on): it catches one side of the cloud first, and the outer letters last
        ang = np.arctan2(home[:, 2], home[:, 0])
        rr = np.linalg.norm(home, axis=1)
        self.ts = T_SPIRAL + 22.0 * ((ang / (2 * np.pi) + 0.35) % 1.0) + 10.0 * clamp01(rr / 20.0) + r.uniform(0, 6, n)
        self.te = np.minimum(self.ts + r.uniform(78.0, 104.0, n) + 12.0 * clamp01(rr / 20.0), T_BREATH - 6)
        # its angle (per letter): integrate an angular speed that grows as the radius falls (kept momentum)
        self.tg = np.arange(T_SPIRAL - 2.0, T_IGN + 2.0, 0.25)
        U = self.u_of(self.tg[None, :], self.ts[:, None], self.te[:, None])
        om = OMEGA0 * np.minimum((1.0 - 0.985 * U) ** -2.1, OMEGA_MAX / OMEGA0)
        th = np.cumsum(om * 0.25, axis=1)
        self.theta = th.astype(np.float32)
        # ---- points per glyph (a little more than the eye needs at its largest)
        cnt_a = A['cnt'][proto]
        tl_ = np.array([len(A['text'][p]) for p in proto])
        npt = np.where(s > 0.5, 520, np.where(s > 0.3, 240, 110))
        npt = np.where(tl_ > 1, (npt * np.minimum(1 + 0.45 * (tl_ - 1), 2.6)).astype(int), npt)
        npt[:nf] = 700 * np.minimum(1 + 0.4 * (tl_[:nf] - 1), 2.0)
        npt = np.minimum(npt, cnt_a)
        self.proto = proto
        self.npt = npt
        self.gid = np.repeat(np.arange(n), npt)
        self.loc = np.concatenate([A['pts'][A['off'][p]:A['off'][p] + k] for p, k in zip(proto, npt)]).astype(np.float64)
        self.pe = (1.0 / npt)[self.gid]
        self.prand = r.random(len(self.gid))
        self.n = n
        print('glyphs3: letters', n, 'points', len(self.gid), 'first', nf)

    @staticmethod
    def u_of(t, ts, te):
        return ease_in(np.clip((t - ts) / (te - ts), 0.0, 1.0), 1.8)

    def centre_pre(self, t):
        """before the swirl: each letter's journey in on its current, then a slow drift in the cloud"""
        n = self.n
        ta = np.broadcast_to(np.asarray(t, np.float64), (n,))
        x = np.clip((ta - self.t_arr) / self.dur, 0.0, 1.0)
        k = ease_out(x, 2.4)
        c = self.orig + (self.home - self.orig) * k[:, None] + self.bow * np.sin(np.pi * k)[:, None]
        q = self.home * 0.2 + np.stack([0.004 * ta, 0.006 * ta, -0.003 * ta], 1)
        w = vnoise(q, 1.0, (0.0, 0.0, 0.0), 2)
        return c + w * self.wamp[:, None]

    def state(self, t):
        c = self.centre_pre(t)
        u = self.u_of(t, self.ts, self.te)
        act = u > 0
        if act.any():
            th = np.array([np.interp(t, self.tg, self.theta[i]) for i in np.nonzero(act)[0]]) if act.sum() < 64 else \
                self._theta_at(t)[act]
            v = c[act]
            h = v @ AXIS
            w = v - h[:, None] * AXIS[None, :]
            cw = np.cross(AXIS[None, :], w)
            cth, sth = np.cos(th)[:, None], np.sin(th)[:, None]
            w2 = w * cth + cw * sth
            ua = u[act]
            gr = (1.0 - ua) ** 1.25 * (1.0 + 0.12 * np.sin(np.pi * np.minimum(ua * 2.5, 1.0)))
            gh = (1.0 - ua) ** 1.5
            # the swirl's own unevenness: a turbulence that dies as the letters fall in
            tq = w2 * 0.12 + np.array([0.01 * t, 0.0, -0.013 * t])
            turb = vnoise(tq, 1.0, (4.4, 1.1, 2.7), 2) * (1.6 * (1.0 - ua) * np.minimum(ua * 6.0, 1.0))[:, None]
            c[act] = w2 * gr[:, None] + (h * gh)[:, None] * AXIS[None, :] + turb
        return c, u

    def _theta_at(self, t):
        i = np.clip((t - self.tg[0]) / 0.25, 0, len(self.tg) - 1.001)
        i0 = int(i)
        f = i - i0
        return self.theta[:, i0] * (1 - f) + self.theta[:, i0 + 1] * f

    def points(self, t, campos):
        c, u = self.state(t)
        sc = self.s * np.maximum((1.0 - u) ** 0.65, 0.015)
        fwd = campos[None, :] - c
        fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-6)
        right = np.cross(np.array([0.0, 1.0, 0.0])[None, :], fwd)
        right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-6)
        up = np.cross(fwd, right)
        a = np.clip(self.tilt[:, 0] + 0.18 * np.sin(self.tumf[:, 0] * t + self.tump[:, 0]), -0.45, 0.45) + 1.6 * u ** 1.6
        b = np.clip(self.tilt[:, 1] + 0.3 * np.sin(self.tumf[:, 1] * t + self.tump[:, 1]), -0.7, 0.7)
        ca, sa = np.cos(a), np.sin(a)
        r2 = ca[:, None] * right + sa[:, None] * up
        u2 = -sa[:, None] * right + ca[:, None] * up
        r2 = r2 * np.cos(b)[:, None] + fwd * np.sin(b)[:, None]
        g = self.gid
        lx = self.loc[:, 0] * sc[g]
        ly = self.loc[:, 1] * sc[g]
        P = c[g] + r2[g] * lx[:, None] + u2[g] * ly[:, None]
        return P, u, sc

    def absorbed(self, t):
        u = self.u_of(t, self.ts, self.te)
        return float(np.mean(smoothstep(0.86, 0.995, u)))

    def emit(self, ctx):
        t = ctx.t
        P0, _, _ = self.points(ctx.t0, ctx.cam0.pos)
        P1, u, sc = self.points(ctx.t1, ctx.cam1.pos)
        g = self.gid
        # appear: the first letters fade up in place in the white; the rest are born out of the dark far off
        # and brighten as they come in
        x = np.clip((t - self.t_arr) / self.dur, 0.0, 1.0)
        born = smoothstep(self.t_arr - 30.0, self.t_arr + 25.0, t)
        born[:self.nf] = smoothstep(self.t_arr[:self.nf], self.t_arr[:self.nf] + 18.0, t)
        tw = 1.0 + self.tw_a * np.sin(self.tw_f * t + self.tw_p)
        heat = smoothstep(0.72, 1.0, u)
        gone = 1.0 - smoothstep(0.9, 0.995, u)
        eg = self.E * tw * born * (0.55 + 0.45 * x) * (1.0 + 2.2 * heat) * gone * np.maximum(1.0 - u, 0.0) ** 1.15
        eg = eg * LETTER_E * self.s ** 2
        T = lerp(self.T, 0.9, heat)
        col = look.blackbody(T)
        hm = smoothstep(0.82, 1.0, u)[:, None]
        col = col * (1 - 0.75 * hm) + (C_CORE * 0.55 + C_ICE * 0.45)[None, :] * 0.75 * hm
        e = eg[g] * self.pe * (0.75 + 0.5 * self.prand)
        t2 = float(smoothstep(686, 704, t)) * (1.0 - float(smoothstep(836, 856, t)))
        if t2 > 0:
            H, W = ctx.fr.H, ctx.fr.W
            _, v, _ = ctx.cam.project(self.state(t)[0], W, H)
            low = smoothstep(0.6 * H, 0.72 * H, v)
            eg2 = 1.0 - 0.7 * t2 * low
            e = e * eg2[g]
        m = e > 1e-7
        ctx.fr.splat(P0[m], P1[m], (sc[g] * 0.02)[m], e[m], col[g][m], ctx.cam0, ctx.cam1)


AXIS = np.array([0.06, 1.0, 0.16]) / np.linalg.norm([0.06, 1.0, 0.16])   # nearly vertical: the swirl is seen
                                                                          # almost edge-on (parallax, never arms)
OMEGA0 = 0.017                 # rad/frame at the swirl's start
OMEGA_MAX = 0.14
LETTER_E = 2600.0


# ============================================================================ the point

class ThePoint:
    def __init__(self, letters, seed=909):
        self.L = letters
        r = rng(seed)
        self.n = 1200
        self.d = r.normal(0, 1, (self.n, 3)) * (r.random(self.n) ** 1.5)[:, None]

    def level(self, t):
        return self.L.absorbed(t)

    def emit(self, ctx):
        t = ctx.t
        if t < T_SPIRAL + 30:
            return
        lv = self.level(t)
        # the breath (bar 13 b4): it holds, shrinks a little, trembles, then swells for the ignition
        br = 1.0 - 0.18 * float(smoothstep(T_BREATH, T_BREATH + 8, t)) + 0.5 * float(smoothstep(T_IGN - 5, T_IGN, t))
        I = (60.0 + 26000.0 * lv ** 1.5) * br
        trem = 0.012 * np.sin(np.array([37.0, 51.0, 43.0]) * t) * float(smoothstep(T_BREATH, T_BREATH + 6, t))
        rad = 0.04 + 0.05 * lv
        P = self.d * rad + trem
        ctx.fr.splat(P, P, 0.004, np.full(self.n, I / self.n), np.broadcast_to(C_CORE, (self.n, 3)), ctx.cam0, ctx.cam1)
        H = np.zeros((3, 3))
        ctx.fr.splat(H, H, np.array([0.14, 0.5, 1.8]), np.array([0.9, 0.45, 0.2]) * I,
                     np.array([C_CORE, C_ICE * 0.6 + C_CORE * 0.4, C_GOLD * 0.5 + C_ICE * 0.5]),
                     ctx.cam0, ctx.cam1, profile=1)


# ============================================================================ timeline hooks

def _letters(tl):
    return tl._get('g3_letters', Letters)


def emit(tl, ctx):
    L = _letters(tl)
    L.emit(ctx)
    tl._get('g3_point', lambda: ThePoint(L)).emit(ctx)


def white_k(t):
    """the white: full at 560, drawn back toward the gathering point, gone by ~690"""
    return 1.0 - float(smoothstep(T_WHITE, T_WHITE + 110, t))


def post(tl, ctx, hdr):
    t = ctx.t
    if t < T_WHITE + 130:
        H, W = hdr.shape[:2]
        u0, v0 = POINT_PX[0] * W / 1920.0, GATHER_Y * H / 804.0
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - u0) / W) ** 2 + ((yy - v0) / W) ** 2)
        sr = float(smoothstep(T_WHITE + 4, T_WHITE + 72, t))
        R = 0.2 + 1.45 * (1.0 - sr) ** 1.6                          # the glow's radius (x frame width)
        amp = 1.0 - float(smoothstep(T_WHITE + 36, T_WHITE + 118, t))
        g = np.exp(-(d / R) ** 1.25) * amp
        full = (1.0 - float(smoothstep(T_WHITE, T_WHITE + 14, t)))
        g = np.maximum(g, full)[..., None]
        hdr = hdr * (1.0 - 0.6 * np.clip(g, 0, 1)) + (C_WHITE * 6.0).astype(np.float32) * g
    return hdr


def finish_opts(tl, f):
    bloom = 0.13
    if f >= T_POINT:
        bloom = 0.16
    return dict(exposure=1.0, bloom_strength=bloom, bloom_threshold=0.7, streak_strength=0.0, vignette_amount=0.25)
