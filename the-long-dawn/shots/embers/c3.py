"""EMBERS v3, cut C on its own timeline (BIBLE_V3 "REVISION 1 . LOCKED BEAT SHEETS", cut C). Frames are C's own.

    C6  1040-1440  THE FORGING        the forge-towers of every realm rise round the gold fire (many, none taller, no
                                      pair); from 1200 its light is drawn out and beaten into a band that cools from
                                      white through yellow to gold, throwing sparks; 1320 the inscription burns up;
                                      1360 the Ring rises above the towers, and the fire still burns beneath it
    C7  1440-1680  THE RACE           a hush: the Ring hangs over the towers and the first gold falls slowly from it
                                      into the nearest windows; from 1560 the towers surge on every beat, gold rains
                                      into their windows, walls of red rise between them
    C9  1920-2080  THE EYE ONTO NOTHING   (the v2 src machinery, time-warped: src 830-879) the storm resolves into a
                                      lidless Eye over all the towers, every tower leaning in; 2000 its slit opens for
                                      the first time, onto empty black
    C11 2320-2480  THE GRASP THAT CANNOT HOLD  (src 960-1039, time-warped) the claw descends, closes on the Ring at
                                      2360, the crust glows and cracks from 2400, the band slips through at 2440

THE FORGING and THE RACE run on a schedule (scene_b.SCHED / tolkien.SCHED, in C's frames); THE EYE and THE GRASP
render the v2 src scene at warped times (SCHED off), so their looks are the verified v3 ones.
"""
import math

import numpy as np

import look
from core import Camera, smoothstep, smootherstep, lerp, catmull, rng, vnoise
import scene_b as B
import tolkien as TK

BAR, BEAT = 80, 20


def bar(n, b=1.0):
    return int(round((n - 1) * BAR + (b - 1) * BEAT))


T_FIRE = bar(11)          # 800: the fire catches (E15)
T_TOWERS = bar(14)        # 1040
T_FORGE = bar(16)         # 1200
T_WRITE = bar(17, 3)      # 1320
T_RISE = bar(18)          # 1360
T_RACE = bar(19)          # 1440
T_SURGE = bar(20, 3)      # 1560
T_RACE_END = bar(22)      # 1680
T_EYE = bar(25)           # 1920
T_SLIT = bar(26)          # 2000
T_EYE_END = bar(27)       # 2080
T_GRASP = bar(30)         # 2320
T_CLOSE = bar(30, 3)      # 2360
T_CRACK = bar(31)         # 2400
T_SLIP = bar(31, 3)       # 2440
T_GRASP_END = bar(32)     # 2480

SHOTS = [(T_TOWERS, T_RACE_END), (T_EYE, T_EYE_END), (T_GRASP, T_GRASP_END)]
C_GOLDEN = TK.C_GOLD


# ================================================================== schedule ===

class C3Sched:
    ign = float(T_FIRE)
    end = float(T_RACE_END)
    tower_rise = float(T_TOWERS)
    vortex_t0 = 1e9
    beats = [float(x) for x in range(T_SURGE, T_RACE_END + 1, BEAT)]
    anvil = [float(x) for x in range(T_FORGE, T_WRITE + 1, BEAT)]
    ring_t0 = float(T_FORGE)
    walls_t0 = float(T_SURGE + 18)
    # C's fire is gold and calm (born from the book): a warm white heart, a pale gold body, a deep gold edge
    mind_cols = (np.array([1.0, 0.95, 0.84]), look.hexrgb('#FFD98A') * 0.75 + np.array([1.0, 0.95, 0.84]) * 0.25,
                 look.hexrgb('#F7B23E'))

    def redness(self, t):
        return 0.55 * float(smoothstep(T_SURGE, T_RACE_END + 40, t))

    def beat_pulse(self, t):
        for tb in self.beats:
            x = t - tb
            if 0 <= x < 20:
                return math.exp(-x / 3.0)
        return 0.0

    def anvil_pulse(self, t):
        for tb in self.anvil:
            x = t - tb
            if 0 <= x < 20:
                return math.exp(-x / 2.5)
        return 0.0

    def race(self, t):
        return float(smoothstep(T_SURGE - 20, T_RACE_END, t))

    def fire_centre(self, t):
        return np.array([0.0, 0.0, 0.0])

    def fire_scale(self, t):
        return 1.0

    def power(self, t):
        # its light is drawn out into the band (dims a little while it is forged), then it burns on beneath the Ring
        return 1.0 - 0.3 * float(smoothstep(T_FORGE, T_FORGE + 20, t)) * (1 - float(smoothstep(T_WRITE, T_RISE, t)))

    def forged(self, t):
        return 0.0

    def vortex_tilt(self, t):
        return np.eye(3)

    def vortex_grow(self, t):
        return 0.0

    def tower_lean(self, t):
        return 0.0

    def tower_rise_glow(self, t):
        return 1.0 - 0.7 * float(smoothstep(T_TOWERS + 60, T_TOWERS + 110, t))

    def tower_grow(self, t):
        return 1.0 + 0.8 * self.race(t)

    def tower_extra(self, towers, i, t):
        return 0.0

    # --- the Ring (tolkien.SCHED)
    def ring_on(self, t):
        return T_FORGE - 6 <= t < T_RACE_END

    def ring_y(self, t):
        return 2.8 + 33.0 * float(smootherstep(T_RISE, T_RACE, t)) + 6.0 * float(smoothstep(T_SURGE, T_RACE_END, t))

    def ring_frame(self, t):
        az, dep = cam_az_dep(t)
        tilt = 0.42 + 0.25 * float(smootherstep(T_RISE, T_RACE, t))
        return B._rot_about(az, tilt), np.array([0.0, self.ring_y(t), 0.0])


SCHED = C3Sched()


def use_sched():
    """C3's schedule on (THE FORGING, THE RACE)"""
    B.SCHED = SCHED
    B.IGN = SCHED.ign
    B.BEATS = list(SCHED.beats)
    TK.SCHED = SCHED
    TK.T_FORGE0, TK.T_FORGE1 = float(T_FORGE), float(T_FORGE + 26)
    TK.T_WRITE0, TK.T_WRITE1 = float(T_WRITE), float(T_WRITE + 26)
    TK.T_EYE0, TK.T_EYE1 = 1e9, 1e9 + 22


def use_src():
    """the v2 src scene (THE EYE, THE GRASP, rendered at warped times)"""
    B.SCHED = None
    B.IGN = 480.0
    B.BEATS = [640 + 20 * k for k in range(16)]
    TK.SCHED = None
    TK.T_FORGE0, TK.T_FORGE1 = 596.0, 622.0
    TK.T_WRITE0, TK.T_WRITE1 = 616.0, 642.0
    TK.T_EYE0, TK.T_EYE1 = 836.0, 858.0


# ============================================================ time warps ===

EYE_WARP = [(T_EYE, 834.0), (T_SLIT - 14, 851.0), (T_SLIT, 857.0), (T_EYE_END, 879.4)]
# src grasp: scene_c's v3 constants are cut for this -- src 960-1040 at half speed (close 980 = C 2360, cracks 1000 =
# C 2400, slip 1020 = C 2440)
GRASP_WARP = [(T_GRASP, 960.0), (T_GRASP_END, 1040.0)]


def warp(t, table):
    xs = [a for a, b in table]
    ys = [b for a, b in table]
    return float(np.interp(t, xs, ys))


# ================================================================ cameras ===

AC = B.ALPHA_C

CAM_C6 = [  # (frame, radius, azimuth offset, height, target y, hfov)
    (1040, 16.0, 0.3, 0.5, 1.2, 50.0),          # the gold fire alone (E15 hands over here)
    (1110, 40.0, 0.18, 5.0, 7.0, 54.0),         # the forges of every realm rise round it
    (1180, 44.0, 0.08, 8.0, 9.0, 54.0),
    (1235, 24.0, 0.03, 4.0, 3.0, 54.0),         # its light drawn out and beaten into a band
    (1270, 21.0, 0.01, 4.0, 3.0, 54.0),
    (1330, 20.0, -0.02, 4.5, 4.0, 54.0),        # the inscription burns up out of the metal
    (1380, 26.0, -0.05, 8.0, 14.0, 58.0),       # the Ring rises above the towers; the fire burns on beneath it
    (1440, 34.0, -0.06, 16.0, 30.0, 62.0),
    (1520, 35.0, -0.1, 20.0, 33.0, 64.0),       # the hush: the Ring over the towers, the first gold falling
    (1580, 35.0, -0.14, 22.0, 35.0, 66.0),      # the race
    (1680, 34.0, -0.2, 26.0, 39.0, 68.0),
]


def cam_c(t):
    k = [(f, np.array([r, a, y, ty, hf])) for f, r, a, y, ty, hf in CAM_C6]
    r, a, y, ty, hf = catmull(t, k)
    pos = np.array([r * math.cos(AC + a), y, r * math.sin(AC + a)])
    tgt = np.array([0.0, ty, 0.0])
    sh = np.zeros(3)
    for tb in SCHED.beats:
        x = t - tb
        if 0 <= x < 14:
            sh += 0.25 * math.exp(-x / 3.0) * np.array([math.sin(2.1 * x + tb), math.sin(2.9 * x + 0.5 * tb), 0.0])
    return pos + sh * 0.5, tgt + sh, float(hf)


def cam_az_dep(t):
    pos, tgt, hf = cam_c(t)
    C = np.array([0.0, SCHED.ring_y(t) if t >= T_FORGE - 6 else 0.0, 0.0])
    return math.atan2(pos[2], pos[0]), math.atan2(C[1] - pos[1], math.hypot(pos[0], pos[2]))


# ============================================================= gold rain ===

class GoldRain:
    """C7: gold falls from the Ring into the towers' windows -- a few slow drops in the hush, then a rain on every
    surge; each drop lands in a window and makes it flare"""

    def __init__(self, tl, seed=611):
        r = rng(seed)
        self.tl = tl
        n = 1400
        slow = r.random(n) < 0.12
        self.t0 = np.where(slow, r.uniform(T_RACE + 10, T_SURGE, n),
                           np.array(SCHED.beats)[r.integers(0, len(SCHED.beats), n)] + r.uniform(-2, 12, n))
        self.dur = np.where(slow, r.uniform(55, 80, n), r.uniform(26, 40, n))
        self.th = r.uniform(0, 2 * np.pi, n)
        self.tw = r.integers(0, 8, n)
        self.hy = r.uniform(0.35, 0.8, n)
        self.E = r.lognormal(0, 0.4, n) * np.where(slow, 1.6, 1.0)
        self.jit = r.normal(0, 1, (n, 2))

    def pos(self, t):
        tw = self.tl.towers
        Rot, C = SCHED.ring_frame(t)
        start = C + (np.stack([np.cos(self.th), np.zeros_like(self.th), np.sin(self.th)], 1) * TK.R_RACE) @ Rot.T
        base = np.array([tw.base(i) for i in range(8)])[self.tw]
        h = np.array([tw.height(i, t) for i in range(8)])[self.tw]
        inward = -base / np.maximum(np.linalg.norm(base[:, [0, 2]], axis=1, keepdims=True), 1e-6)
        inward[:, 1] = 0.0
        end = base + np.stack([np.zeros(len(h)), h * self.hy, np.zeros(len(h))], 1) + inward * 2.6
        end[:, 0] += self.jit[:, 0] * 0.8
        end[:, 2] += self.jit[:, 1] * 0.8
        u = np.clip((t - self.t0) / self.dur, 0, 1)
        P = start + (end - start) * u[:, None] + np.array([0.0, 1.0, 0.0]) * (6.0 * u * (1 - u))[:, None]
        return P, u, end

    def emit(self, ctx):
        t = ctx.t
        if t < T_RACE or t >= T_RACE_END:
            return
        P0, _, _ = self.pos(ctx.t0)
        P1, u, end = self.pos(ctx.t1)
        fly = (t >= self.t0) & (u < 1)
        if fly.any():
            e = self.E * 3.0 * fly * (0.6 + 0.4 * u)
            ctx.fr.splat(P0[fly], P1[fly], 0.05, e[fly], C_GOLDEN, ctx.cam0, ctx.cam1, zref=30.0)
        # the windows flare where it lands
        land = t - (self.t0 + self.dur)
        fl = (land >= 0) & (land < 16)
        if fl.any():
            e = self.E * 60.0 * np.exp(-land / 5.0) * fl
            ctx.fr.splat(end[fl], end[fl], 0.8, e[fl], C_GOLDEN * 0.8 + look.blackbody(0.6) * 0.2, ctx.cam0,
                         ctx.cam1, profile=1, zref=30.0)


# =============================================================== timeline ===

import timeline as TL  # noqa: E402


class TimelineC3(TL.Timeline):
    """THE FORGING and THE RACE on the schedule; THE EYE and THE GRASP via a src Timeline at warped times"""

    def __init__(self):
        use_sched()
        super().__init__()
        self._src = None

    gold_rain = property(lambda s: s._get('gold_rain', lambda: GoldRain(s)))

    def src(self):
        if self._src is None:
            use_src()
            self._src = TL.Timeline()
        return self._src

    def mode(self, t):
        return 'sched' if t < T_RACE_END else 'src'

    def camera(self, t):
        if self.mode(t) == 'src':
            use_src()
            return self.src().camera(self.srct(t))
        use_sched()
        pos, tgt, hf = cam_c(t)
        Rot, C = SCHED.ring_frame(t)
        focus = float(np.linalg.norm((C if t >= T_FORGE else np.zeros(3)) - pos))
        return Camera(pos, tgt, hfov=hf, focus=focus, aperture=float(lerp(0.06, 0.12, smoothstep(1100, 1300, t))))

    @staticmethod
    def srct(t):
        if T_EYE - 1 <= t < T_EYE_END + 1:
            return warp(t, EYE_WARP)
        return warp(t, GRASP_WARP)

    def render_opts(self, f):
        if self.mode(f) == 'src':
            return self.src().render_opts(int(round(self.srct(f))))
        return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=45.0, fog_len=70.0, near=0.3)

    def emit(self, ctx):
        t = ctx.t
        if self.mode(t) == 'src':
            use_src()
            f, t0, t1 = ctx.t, ctx.t0, ctx.t1
            ctx.t, ctx.t0, ctx.t1 = self.srct(f), self.srct(t0), self.srct(t1)
            ctx.f = int(round(ctx.t))
            try:
                self.src().emit(ctx)
            finally:
                ctx.c3 = (f, t0, t1)
            return
        use_sched()
        lp, lc, lpw = self.light(t)
        self.towers.prepare(ctx)
        self.dust.emit(ctx)
        self.smoke.emit(ctx, lp, lc, lpw)
        self.towers.emit(ctx, lp, lc, lpw)
        self.tembers.emit(ctx)
        self.tsmoke.emit(ctx, lp, lc, lpw)
        self.walls.emit(ctx)
        self.sparks.emit(ctx)
        self.fire.emit(ctx)
        self.fsparks.emit(ctx)
        self.ring.emit(ctx)
        self.gold_rain.emit(ctx)

    def post(self, ctx, hdr):
        if hasattr(ctx, 'c3'):
            return self.src().post(ctx, hdr)
        return hdr

    def finish_opts(self, f):
        if self.mode(f) == 'src':
            return self.src().finish_opts(int(round(self.srct(f))))
        return dict(exposure=1.0, bloom_strength=0.15, bloom_threshold=0.7, streak_strength=0.0,
                    vignette_amount=0.25)
