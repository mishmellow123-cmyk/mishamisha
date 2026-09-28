"""EMBERS v3, cut A on its own timeline (BIBLE_V3 "REVISION 1 . LOCKED BEAT SHEETS", cut A). Frames are A's own.

    A5  1040-1440  IGNITION . THE PROMISE   the thinking fire owns the frame; its light rolls out and the dark answers
                                            with a valley drawn in embers to the horizon (E3), gone by 1420
    A6  1440-1840  THE TOWERS . TWO GIANTS  forge-stacks rise round the fire, lit only on the face turned to it;
                                            from 1730 two giants outgrow the rest; 1760 the camera slips behind one
    A7  1840-2400  THE EDGE                 (edge.py) the crater, the gilding, the crumbling rim, one steady orbit
    A8  2400-2640  THE BRINK                (edge.py) the torn vortex, the rim gives way, a crown falls, over the rim
    A9  2640-2800  THE DEAD VALLEY          the promise's own points, grey and unlit, under ash
    A10 2800-3120  BLACK . THE EMBER        one ember drifts down and hangs; it does not die
    A16 4400-4720  TOWERS IN THE LIGHT      (edge.py) far ridge fires; the giants' shutters open first
    A17 4720-4880  THE FIRE, SEEN           (edge.py) down to the calm fire, which gathers into one heart

Nothing here changes the v2 src timeline: scene_b's functions consult scene_b.SCHED only when this module sets it.
"""
import math

import numpy as np

import look
from core import Camera, smoothstep, smootherstep, lerp, catmull, rng, vnoise, ease_in_out, ease_out
import scene_b as B

BAR, BEAT = 80, 20


def bar(n, b=1.0):
    """first frame of bar n (1-based), beat b (1-based, fractional allowed)"""
    return int(round((n - 1) * BAR + (b - 1) * BEAT))


T_IGN = bar(14)             # 1040 ignition
T_T4 = (1140, 1270)
T_PROMISE = bar(17)         # 1280 its light rolls out
T_PROMISE_GONE = bar(18, 4)  # 1420
T_TOWERS = bar(19)          # 1440
T_GIANTS = bar(22, 3.5)     # 1730 the forges give way to the kingdoms: two grow tallest
T_BEHIND = bar(23)          # 1760 the camera slips behind one giant
T_EDGE = bar(24)            # 1840
T_CRUMBLE = bar(26)         # 2000
T_BRINK = bar(31)           # 2400
T_RIM_GIVES = bar(31, 3)    # 2440
T_CROWN = bar(32)           # 2480
T_TIP = bar(32, 3)          # 2520
T_WHITE = bar(34)           # 2640
T_DEAD = bar(34, 2)         # 2660
T_BLACK = bar(36)           # 2800
T_EMBER_END = bar(40)       # 3120
T_LIGHT = bar(56)           # 4400
T_STOP = bar(56, 3)         # 4440 the surges stop
T_OPEN_GIANTS = bar(57)     # 4480
T_OPEN_OTHERS = bar(58)     # 4560
T_HOLDS = bar(59, 3)        # 4700
T_SEEN = bar(60)            # 4720
T_HEART = bar(61)           # 4800
T_END = bar(62)             # 4880

GIANTS = (2, 6)
PROMISE_GAIN = 3.0
GIANT_TOP = 42.0            # both giants grow to this crown height (y) over 1730-1795; the rest stand at 12-26


# ================================================================== schedule ===

class A3Sched:
    """what scene_b's functions return on A's own timeline"""
    ign = float(T_IGN)
    end = float(T_END)
    tower_rise = float(T_TOWERS)
    vortex_t0 = float(T_BRINK)
    beats = [float(x) for x in range(T_EDGE, T_WHITE, BEAT)] + [float(T_LIGHT), float(T_LIGHT + BEAT)]
    smoke_gain = 0.45           # thinner smoke off the tops, so the silhouettes and the giants read
    surge_scale = 0.28          # 40 surges in A (v2 had 16): each one smaller, so the towers stay on the rim

    def haze(self, t):
        """0..1: the air of the arena lit by the fire (a backdrop the towers stand out against)"""
        if t >= T_WHITE and t < T_LIGHT:
            return 0.0
        return float(smoothstep(T_PROMISE + 60, T_TOWERS + 60, t))

    def dust_k(self, t):
        return float(smoothstep(T_TOWERS, T_TOWERS + 80, t))

    def redness(self, t):
        if t >= T_LIGHT:
            return 0.55 * (1 - float(smoothstep(T_STOP, T_HOLDS, t)))
        return 0.6 * float(smoothstep(T_EDGE, T_BRINK, t)) + 0.25 * float(smoothstep(T_BRINK, T_WHITE, t))

    def beat_pulse(self, t):
        for tb in self.beats:
            x = t - tb
            if 0 <= x < 20:
                return math.exp(-x / 3.0)
        return 0.0

    def race(self, t):
        """0 -> 1 as the race heats (tower embers, smoke glow, tower surge brightness)"""
        if t >= T_LIGHT:
            return 1.0 - 0.7 * float(smoothstep(T_STOP, T_HOLDS, t))
        return float(smoothstep(T_GIANTS, T_EDGE + 300, t))

    def fire_centre(self, t):
        y = 0.0
        if t >= T_EDGE and t < T_LIGHT:
            # EMBERS-2: as the ground falls away it settles into the crater's mouth, just above the lip, so the towers
            # lean in over the edge to reach it; at the brink it swells up out of the pit
            y = -9.0 * float(smootherstep(T_EDGE + 6, T_EDGE + 90, t)) + 5.0 * float(smoothstep(T_BRINK, T_BRINK + 60, t))
        return np.array([0.0, y, 0.0])

    def fire_scale(self, t):
        """the fire's size (x the A5 flame). As its light rolls out (the promise) it swells to twice A5's size, so
        it stays the brightest thing in the frame once the towers rise round it; it swells again over the edge"""
        if t < T_LIGHT:
            s_ = 1.0 + 1.0 * float(smootherstep(T_PROMISE, T_TOWERS, t))
            # EMBERS-2: it settles smaller into the crater's mouth as the ground falls away (the bowl and the far
            # towers must read round it), then swells out of the pit at the brink
            s_ += (-0.6 * float(smootherstep(T_EDGE + 6, T_EDGE + 70, t)) + 0.3 * float(smoothstep(T_EDGE + 70, T_BRINK, t))
                   + 2.6 * float(smoothstep(T_BRINK, T_WHITE, t)))
            return s_
        # the calm fire after the turn, gathering into one small heart for the match cut
        return 2.0 * (1.0 - 0.72 * float(smootherstep(T_HEART, T_END - 8, t)))

    def power(self, t):
        """surface brightness is kept as the fire grows (energy ~ its area), plus the race's own heat"""
        k = self.fire_scale(t) ** 1.6
        if t < T_EDGE:
            return k
        if t < T_LIGHT:
            return k * (1.0 + 0.5 * float(smoothstep(T_EDGE, T_BRINK, t)) + 1.2 * float(smoothstep(T_BRINK, T_WHITE, t)))
        return k * (1.0 + 2.2 * float(smootherstep(T_HEART, T_END - 8, t)))     # a small, intense heart

    def vortex_tilt(self, t):
        return np.eye(3)

    def vortex_grow(self, t):
        if t < T_BRINK or t >= T_LIGHT:
            return 0.0
        return float(smootherstep(T_BRINK - 20, T_BRINK + 150, t))

    def tower_lean(self, t):
        if t >= T_LIGHT:
            return 0.07 - 0.03 * float(smoothstep(T_STOP, T_HOLDS, t))
        return 0.07 * float(smoothstep(T_EDGE, T_EDGE + 400, t)) + 0.02 * float(smoothstep(T_BRINK, T_WHITE, t))

    def tower_rise_glow(self, t):
        return 1.0 - 0.7 * float(smoothstep(T_TOWERS + 60, T_TOWERS + 110, t))

    def tower_grow(self, t):
        return 1.0 + 0.8 * self.race(t)

    def tower_extra(self, towers, i, t):
        if i in GIANTS:
            grow = GIANT_TOP - (B.GROUND + towers.h_rise[i])
            return grow * float(smootherstep(T_GIANTS, T_GIANTS + 50, t))
        return 0.0


SCHED = A3Sched()


def install():
    """point scene_b at A's timeline (call before the scene's elements are built)"""
    B.SCHED = SCHED
    B.IGN = SCHED.ign
    B.BEATS = list(SCHED.beats)


# ============================================================= THE PROMISE ===

def valley_h(x, z):
    """the promised valley (world units): a broad floor round the fire, a river winding away to the horizon, hills
    rising beyond with terraced flanks"""
    r = np.hypot(x, z)
    hills = 26.0 * smoothstep(70.0, 260.0, r) * (0.65 + 0.35 * np.sin(0.012 * x + 1.3) * np.cos(0.009 * z - 0.4))
    swell = 4.0 * np.sin(0.035 * x + 0.7) * np.sin(0.03 * z + 1.9) * smoothstep(30.0, 90.0, r)
    return B.GROUND - 1.5 + hills + swell


RIVER_AZ = B.ALPHA_C - 0.21 + np.pi + 0.12       # it winds away from the fire, into the promise camera's view


def river_xz(s):
    """centreline of the river, s in 0..1 from just beyond the fire to the horizon (it leaves at azimuth ~2.0)"""
    a = RIVER_AZ + 0.35 * np.sin(5.0 * s) + 0.15 * np.sin(13.0 * s + 1.0)
    r = 26.0 + 330.0 * s
    return np.stack([r * np.cos(a), r * np.sin(a)], -1)


class Promise:
    """E3: for one breath the dark answers the fire's light with a valley drawn in embers to the horizon --
    terraces on the hills, orchards in rows, a river, roofs -- warm, blooming as the light rolls out, gone by 1420.
    The same points, grey and unlit, are the dead valley (E4)."""

    def __init__(self, seed=515):
        r = rng(seed)
        P, K, E = [], [], []

        def add(p, kind, e):
            P.append(p)
            K.append(np.full(len(p), kind, np.int8))
            E.append(np.broadcast_to(np.asarray(e, float), (len(p),)).copy())
        # 0 terraces: contour lines on the hill flanks (dense, broken)
        # contour lines found by bisection along rays from the fire (the flanks rise monotonically)
        for lv in np.arange(3.0, 26.0, 1.9):
            na = 2600
            aa = RIVER_AZ + r.uniform(-1.5, 1.5, na)
            lo, hi = np.full(na, 80.0), np.full(na, 430.0)
            f = lambda rad: valley_h(rad * np.cos(aa), rad * np.sin(aa)) - B.GROUND - lv
            ok = (f(lo) < 0) & (f(hi) > 0)
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                up = f(mid) > 0
                hi = np.where(up, mid, hi)
                lo = np.where(up, lo, mid)
            rad = 0.5 * (lo + hi)[ok]
            a_ = aa[ok]
            gap = np.sin(a_ * 23.0 + lv * 1.7) > -0.55               # the terraces break here and there
            rad, a_ = rad[gap], a_[gap]
            x, z = rad * np.cos(a_), rad * np.sin(a_)
            add(np.stack([x, valley_h(x, z) + 0.2 + r.normal(0, 0.05, len(x)), z], 1), 0, 1.0)
        # 1 orchards: patches of trees in rows on the valley floor
        for _ in range(30):
            a = RIVER_AZ + r.uniform(-1.4, 1.4)
            d = r.uniform(34, 170)
            cx, cz = d * math.cos(a), d * math.sin(a)
            ang = r.uniform(0, np.pi)
            nr, nc = r.integers(4, 9), r.integers(5, 12)
            sp = r.uniform(2.6, 3.4)
            for i in range(nr):
                for j in range(nc):
                    u = (i - nr / 2) * sp
                    v = (j - nc / 2) * sp
                    tx = cx + u * math.cos(ang) - v * math.sin(ang)
                    tz = cz + u * math.sin(ang) + v * math.cos(ang)
                    m = 16
                    q = r.normal(0, 1, (m, 3)) * np.array([0.35, 0.3, 0.35])
                    ty = float(valley_h(np.array([tx]), np.array([tz]))[0]) + 1.4
                    add(np.stack([tx + q[:, 0], ty + q[:, 1], tz + q[:, 2]], 1), 1, 0.9)
        # 2 the river: two banks and the glinting water between, winding away to the horizon
        s = r.random(30000)
        c = river_xz(s)
        eps = 1e-3
        dcs = river_xz(np.clip(s + eps, 0, 1)) - river_xz(np.clip(s - eps, 0, 1))
        nrm = np.stack([-dcs[:, 1], dcs[:, 0]], 1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        w = 1.2 + 4.0 * s
        side = r.choice([-1.0, 1.0, 0.0, 0.0], len(s))
        off = np.where(side == 0, r.uniform(-0.8, 0.8, len(s)), side * (1 + r.normal(0, 0.05, len(s)))) * w * 0.5
        bx = c[:, 0] + nrm[:, 0] * off
        bz = c[:, 1] + nrm[:, 1] * off
        by = valley_h(bx, bz) * 0 + B.GROUND - 1.7
        add(np.stack([bx, by, bz], 1), 2, np.where(side == 0, 0.25, 0.5))
        # 3 roofs: gable outlines in small villages by the river and on the floor
        for _ in range(18):
            if r.random() < 0.6:
                c0 = river_xz(np.array([r.uniform(0.02, 0.45)]))[0] + r.normal(0, 9, 2)
            else:
                a = RIVER_AZ + r.uniform(-1.3, 1.3)
                c0 = np.array([math.cos(a), math.sin(a)]) * r.uniform(40, 140)
            for _ in range(r.integers(3, 9)):
                hx, hz = c0 + r.normal(0, 6, 2)
                ang = r.uniform(0, np.pi)
                L, Wd, Hh = r.uniform(3, 5.5), r.uniform(2.2, 3.4), r.uniform(1.4, 2.2)
                y0 = float(valley_h(np.array([hx]), np.array([hz]))[0])
                m = 160
                u = r.uniform(-0.5, 0.5, m) * L
                side_ = r.choice([-1.0, 1.0], m)
                v = r.random(m)
                # the two roof planes (ridge along u) and the gable ends
                px = u
                pz = side_ * (1 - v) * Wd * 0.5
                py = 1.8 + v * Hh
                ca, sa = math.cos(ang), math.sin(ang)
                add(np.stack([hx + px * ca - pz * sa, y0 + py, hz + px * sa + pz * ca], 1), 3, 1.1)
                wq = r.normal(0, 1, (10, 3)) * np.array([0.12, 0.12, 0.12])
                add(np.stack([hx + wq[:, 0], y0 + 0.9 + wq[:, 1], hz + wq[:, 2]], 1), 4, 2.4)
        self.P = np.concatenate(P)
        self.K = np.concatenate(K)
        self.E = np.concatenate(E) * r.lognormal(0, 0.35, len(self.P))
        self.r = np.hypot(self.P[:, 0], self.P[:, 2])
        self.ph = r.uniform(0, 2 * np.pi, len(self.P))
        print('promise: points', len(self.P))

    def emit(self, ctx, mode='promise'):
        t = ctx.t
        if mode == 'promise':
            if t < T_PROMISE or t >= T_PROMISE_GONE + 12:
                return
            # the light rolls out from the fire (a front at ~9 units/frame), then all of it fades by 1420
            front = (t - T_PROMISE) * 9.0
            lit = smoothstep(front, front - 40.0, self.r)
            fade = 1.0 - float(smoothstep(T_PROMISE + 70, T_PROMISE_GONE, t))
            fl = 0.8 + 0.2 * np.sin(0.15 * t + self.ph)
            cam = ctx.cam
            z = np.maximum((self.P - cam.pos) @ cam.R[2], 1.0)
            e = self.E * lit * fade * fl * 1.9 * (1.0 + 0.35 * (self.K == 3)) * np.clip(40.0 / z, 0.0, 2.5) * PROMISE_GAIN
            T = 0.44 + 0.1 * (self.K == 3) + 0.06 * (self.K == 1) - 0.08 * (self.K == 2) + 0.16 * (self.K == 4)
            col = look.blackbody(T)
            col = col * 0.8 + look.hexrgb(look.PALETTE['mind_gold']) * 0.2
            m = e > 1e-5
            ctx.fr.splat(self.P[m], self.P[m], 0.09, e[m], col[m], ctx.cam0, ctx.cam1, zref=0.0)
        else:
            # the dead valley: the same points, grey and unlit, crumbling into the black on the last beat
            if t < T_DEAD - 4 or t >= T_BLACK:
                return
            appear = smoothstep(self.r * 0.02 - 4.0 + T_DEAD, self.r * 0.02 + 8.0 + T_DEAD, t)
            crumble = smoothstep(T_BLACK - 22 + 10 * np.sin(self.ph), T_BLACK - 4, t)
            drop = np.stack([np.zeros_like(self.r), -6.0 * crumble ** 2, np.zeros_like(self.r)], 1)
            cam = ctx.cam
            z = np.maximum((self.P - cam.pos) @ cam.R[2], 1.0)
            e = self.E * appear * (1 - crumble) * 0.22 * np.clip(40.0 / z, 0.0, 2.5) * PROMISE_GAIN
            g = np.array([0.55, 0.55, 0.57])
            m = e > 1e-6
            ctx.fr.splat(self.P[m] + drop[m], self.P[m] + drop[m], 0.09, e[m], np.broadcast_to(g, (int(m.sum()), 3)),
                         ctx.cam0, ctx.cam1, zref=0.0)


# ================================================================ cameras ===

def _polar(r, a, y):
    return np.array([r * math.cos(a), y, r * math.sin(a)])


AC = B.ALPHA_C
TH_G2 = 2 * np.pi * 2 / 8 + B.TOWER_ANG0          # giant 2's azimuth

CAM_A5 = [  # (frame, radius, azimuth offset from ALPHA_C, height, target y, hfov)
    (1040, 13.9, 0.43, 7.2, -1.35, 46.0),       # the v2 ignition camera (continuity with the point)
    (1072, 21.0, 0.26, 2.2, 0.9, 48.0),         # the fire blooms and the camera draws back from it ...
    (1110, 26.0, 0.12, -1.2, 0.9, 50.0),        # ... until it owns the frame: ~38% of its height, upper middle
    (1210, 26.8, 0.0, -1.5, 0.9, 50.0),
    (1272, 27.2, -0.1, -1.3, 0.9, 50.0),
    (1330, 34.0, -0.15, 3.5, 1.8, 56.0),        # its light rolls out: up and back to see the land answer
    (1380, 48.0, -0.20, 11.0, 4.5, 62.0),       # ... to the horizon (the fire swells to twice its size)
    (1425, 52.0, -0.22, 10.0, 5.0, 62.0),
    (1470, 50.0, -0.16, 5.0, 7.0, 66.0),        # the forges rise round it: down to the fire's own height
    # A-FIX (the race was legible only in the captions: the forges' burning crowns and the giants' growth were all
    # above the frame): the camera draws back as they rise until the whole ring and its crowns race in view
    (1540, 62.0, -0.08, 2.0, 12.0, 68.0),
    (1620, 78.0, 0.02, 0.0, 17.0, 66.0),        # T5: the ring of forges, every crown burning, round the fire
    (1690, 92.0, -0.04, -2.0, 21.0, 66.0),      # T6a: the smiths race
    (1730, 100.0, -0.08, -4.0, 25.0, 66.0),     # T6b: two giants shoot up past the rest; the camera tilts up with them
    (1759, 104.0, -0.12, -5.0, 28.0, 66.0),
]

# bar 23 b1 (1760): a CUT to behind giant 2, which slips across the fire until it eclipses it: a black mass fringed
# with glare, the other giant lost beyond the fire. Ends on THE EDGE's first camera (edge.py: r 31, ORB_A0).
CAM_A6B = [
    (1760, 36.0, TH_G2 - AC - 0.24, 13.0, 5.0, 58.0),
    (1800, 33.0, TH_G2 - AC - 0.14, 11.5, 4.0, 57.0),
    (1840, 31.0, TH_G2 - AC - 0.11, 10.0, 3.0, 56.0),
]


def cam_a5a6(t):
    keys = CAM_A6B if t >= T_BEHIND else CAM_A5
    k = [(f, np.array([r, a, y, ty, hf])) for f, r, a, y, ty, hf in keys]
    r, a, y, ty, hf = catmull(t, k)
    pos = _polar(r, AC + a, y)
    tgt = np.array([0.0, ty, 0.0]) + 0.0 * pos
    return pos, tgt, float(hf)


def emit_haze(ctx):
    """the arena's air, lit by the fire: two broad soft glows round it (the near towers occlude them, so they stand
    out as silhouettes; the giants rise into the lit air)"""
    t = ctx.t
    hz = SCHED.haze(t)
    if hz <= 0:
        return
    C = B.crown_centre(t) + np.array([0.0, 2.0 * SCHED.fire_scale(t), 0.0])
    pw = B.fire_power(t)
    col = B.C_GOLD * 0.55 + np.array([1.0, 0.62, 0.32]) * 0.45
    red = B.redness(t)
    col = col * (1 - 0.7 * red) + B.C_RED * 0.7 * red
    H = np.array([C, C + np.array([0.0, 6.0, 0.0])])
    ctx.fr.splat(H[:1], H[:1], np.array([6.0]), np.array([900.0]) * hz * pw ** 0.5,
                 col[None, :], ctx.cam0, ctx.cam1, profile=1)     # a small warm core; the lit smoke is the air
    # the glare: a bright, tight bloom round the fire that a tower standing in front of it cuts into a fringe
    # (A6's end: behind the giant, "a black mass fringed with glare")
    g = float(smoothstep(T_BEHIND, T_BEHIND + 20, t)) * (1.0 - float(smoothstep(T_EDGE + 40, T_EDGE + 100, t)))
    if g > 0:
        Cg = B.crown_centre(t) + np.array([0.0, 2.4, 0.0])
        ctx.fr.splat(Cg[None, :], Cg[None, :], np.array([9.0]), np.array([2.0e5 * g]) * pw ** 0.5,
                     (B.C_CORE * 0.5 + B.C_GOLD * 0.5)[None, :], ctx.cam0, ctx.cam1, profile=1, rmax=1400.0)


# =============================================================== timeline ===

import timeline as TL  # noqa: E402

T_GLYPHS = bar(8)           # 560 A3 INTO THE LIGHT . GLYPHS (glyphs3.py; A4 THE POINT 960-1040 too)


def _mine(t):
    """EMBERS-A2's shots own modules: A3/A4 (glyphs3), A9/A10 (aftermath), A16/A17 (turn); None = the rest"""
    if t < T_IGN:
        import glyphs3
        return glyphs3
    if T_WHITE <= t < T_LIGHT:
        import aftermath
        return aftermath
    if t >= T_LIGHT:
        import turn
        return turn
    return None


class TimelineA3(TL.Timeline):
    def __init__(self):
        install()
        import edge  # noqa: F401  (adds THE EDGE's schedule hooks: gilding, lean, the falling crown)
        import turn  # noqa: F401  (EMBERS-A2: A16's hooks over edge's for t >= T_LIGHT: shutters, ridge light)
        super().__init__()

    promise = property(lambda s: s._get('promise', Promise))

    def camera(self, t):
        m = _mine(t)
        if m is not None:
            return m.camera(self, t)
        if t < T_EDGE:
            pos, tgt, hf = cam_a5a6(t)
            focus = float(np.linalg.norm(B.crown_centre(t) - pos))
            ap = float(lerp(0.06, 0.012, smoothstep(T_TOWERS, T_TOWERS + 100, t)))  # the towers stay crisp (no bokeh blobs)
            return Camera(pos, tgt, hfov=hf, focus=focus, aperture=ap)
        import edge
        return edge.camera(self, t)

    def render_opts(self, f):
        m = _mine(f)
        if m is not None:
            return m.render_opts(self, f)
        if T_PROMISE - 10 <= f < T_TOWERS + 40 or T_DEAD - 30 <= f < T_BLACK:
            return dict(bokeh_pow=0.25, bokeh_cap=1.6, fog_start=260.0, fog_len=320.0, near=0.3)
        if f >= T_BLACK and f < T_LIGHT:
            return dict(bokeh_pow=0.0, bokeh_cap=1.0, near=0.05)
        return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=45.0, fog_len=70.0, near=0.3)

    def emit(self, ctx):
        t = ctx.t
        m = _mine(t)
        if m is not None:
            return m.emit(self, ctx)
        if T_BLACK <= t < T_LIGHT:
            import edge
            return edge.emit_black(self, ctx)
        if T_WHITE <= t < T_BLACK:
            self.promise.emit(ctx, mode='dead')
            import edge
            return edge.emit_ash(self, ctx)
        lp, lc, lpw = self.light(t)
        if t >= T_EDGE - 2:
            import edge
            ctx.tl_crater = edge._crater(self)       # the crater joins the occluder (edge._extra_occluders)
        self.towers.prepare(ctx)
        self.dust.emit(ctx)
        self.smoke.emit(ctx, lp, lc, lpw)
        if t >= T_EDGE:
            import edge
            edge.emit_before_towers(self, ctx, lp, lc, lpw)
        self.towers.emit(ctx, lp, lc, lpw)
        self.tembers.emit(ctx)
        self.tsmoke.emit(ctx, lp, lc, lpw)
        if T_TOWERS <= t < T_WHITE:                   # A-FIX: the forges' throats roar, and throw sparks on the beat
            import race_afix
            self._get('throat_fire', lambda: race_afix.ThroatFire(self.towers, T_TOWERS + 30, T_WHITE,
                                                                   giants=GIANTS)).emit(ctx)
        self.sparks.emit(ctx)
        # (H5) no vortex in A: THE BRINK is the fire's own updraft (edge.Updraft) fed by embers off the tower tops
        self.fire.emit(ctx)
        emit_haze(ctx)
        self.fsparks.emit(ctx)
        self.shock.emit(ctx)
        self.promise.emit(ctx, mode='promise')
        if t >= T_EDGE:
            import edge
            edge.emit_after(self, ctx, lp, lc, lpw)

    def post(self, ctx, hdr):
        m = _mine(ctx.t)
        if m is not None:
            return m.post(self, ctx, hdr)
        if ctx.t >= T_EDGE - 40:
            import edge
            return edge.post(self, ctx, hdr)
        return hdr

    def finish_opts(self, f):
        m = _mine(f)
        if m is not None:
            return m.finish_opts(self, f)
        streak = 0.0             # v3 (H5): no anamorphic lens streak at the ignition
        bloom = 0.15
        if T_BLACK <= f < T_LIGHT:
            bloom = 0.1
        return dict(exposure=1.0, bloom_strength=bloom, bloom_threshold=0.7, streak_strength=streak,
                    vignette_amount=0.25)
