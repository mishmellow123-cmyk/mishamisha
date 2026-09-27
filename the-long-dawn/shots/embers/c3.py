"""EMBERS-C, cut C on its own timeline (BIBLE_V3 "REVISION 1 . LOCKED BEAT SHEETS", cut C, with the H5 calls).
Frames are C's own. Output: render.py FRAMES --cut C3 -> renders/embers_C3.

    C6  1040-1440  THE FORGING   the gold fire alone (E15 hands over); from 1040 the forge-towers of every realm rise
                                 round it (many, none taller than the rest, no pair; A's approved forge family:
                                 charcoal crust, banded stacks, glowing throats); from 1200 (bar 16) its light is
                                 drawn out and beaten into a band on the anvil strokes (every beat to 1300): a
                                 white-hot front runs round, it cools through yellow to gold; 1320 (17 b3) the
                                 canonical inscription burns up out of the metal; 1360 (bar 18) the Ring rises above
                                 the towers, the fire burning on beneath it
    C7  1440-1680  THE RACE      a hush: the Ring hangs over the towers and the first gold falls slowly from it into
                                 the nearest windows; from 1560 (20 b3) the towers surge on every beat, gold rains
                                 from the Ring into their windows, walls of red rise between them
    C9  1920-2080  THE EYE       (still the v2 src machinery, time-warped: to be rebuilt without the spiral)
    C11 2320-2480  THE GRASP     (still the v2 src machinery, time-warped: to be rebuilt without the spiral)

THE RING is ringsolid's solid canonical band (not tolkien's splat hoop); the fire is cflame's one natural gold flame
(not the A-style thinking fire); the towers are scene_b's forge family, laid out by C (layout_towers).
"""
import math

import numpy as np

import look
from core import Camera, smoothstep, smootherstep, lerp, catmull, rng, vnoise, snoise
import scene_b as B
import tolkien as TK
import ringsolid as RS
import cflame as CF

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

G = B.GROUND                              # -14: the ground
HF = 6.5                                  # the fire's height (world units)
FIRE_ROOT = np.array([0.0, G, 0.0])       # it burns on the ground at the centre of the forges
RING_W = 1.3                              # the band's width (world units): outer R 2.93, thickness 0.58
# E15 hands over at 1040 on MAP's camera (book_c.cam_letters at t = 20 s): (-0.3, -11.5, 12.0) cm from the heart
# (z up), looking at the heart + (0, 0.2, 0), hfov 38, over a flame 3.3 cm x 0.9 (calm). The same flame here is
# FLAME_S x 0.9 = HF tall, so that view scales by K_E15; the camera's azimuth IS the forging's reference azimuth, and
# the forges are laid out round it (the gap between forges 0 and 1 is where the camera starts).
FLAME_H = 0.9                             # cflame height fraction (MAP's calm flame is 0.9 of its full 3.3 cm)
FLAME_S = HF / FLAME_H
K_E15 = FLAME_S / 3.3
AZ0 = math.atan2(11.5, -0.3)              # the camera's azimuth at 1040 (embers x-z plane: MAP (x, y, z) -> (x, z, -y))
E15_R = K_E15 * math.hypot(0.3, 11.5)     # 1040: horizontal distance ...
E15_Y = K_E15 * 12.0                      # ... height above the flame's root ...
E15_TGT = np.array([0.0, 0.0, -0.2 * K_E15])   # ... and the target's offset from the root (MAP's +0.2 cm north)
STROKES = [float(x) for x in range(T_FORGE, T_WRITE, BEAT)]         # the anvil: 1200 .. 1300, every beat
RACE_BEATS = [float(x) for x in range(T_SURGE, T_RACE_END, BEAT)]   # 1560 .. 1660


# ================================================================== schedule ===

class C3Sched:
    """what scene_b's functions return on C's timeline (THE FORGING and THE RACE)"""
    ign = float(T_FIRE)
    end = float(T_RACE_END)
    tower_rise = float(T_TOWERS)
    vortex_t0 = 1e9
    beats = list(RACE_BEATS)                  # the towers surge on these (scene_b.BEATS)
    pulses = STROKES + RACE_BEATS             # ... and their throats flare on these (the anvil too)
    ring_t0 = float(T_FORGE)
    walls_t0 = float(T_SURGE + 12)
    smoke_gain = 0.6
    surge_scale = 1.0
    mind_cols = (np.array([1.0, 0.95, 0.84]), look.hexrgb('#FFD98A') * 0.75 + np.array([1.0, 0.95, 0.84]) * 0.25,
                 look.hexrgb('#F7B23E'))

    def haze(self, t):
        return float(smoothstep(T_TOWERS + 40, T_TOWERS + 160, t))

    def dust_k(self, t):
        return float(smoothstep(T_TOWERS + 20, T_TOWERS + 120, t))

    def redness(self, t):
        return 0.55 * float(smoothstep(T_SURGE, T_RACE_END + 40, t))

    def beat_pulse(self, t):
        for tb in self.pulses:
            x = t - tb
            if 0 <= x < 20:
                return math.exp(-x / (2.5 if tb < T_WRITE else 3.0)) * (0.7 if tb < T_WRITE else 1.0)
        return 0.0

    def race(self, t):
        return float(smoothstep(T_SURGE - 20, T_RACE_END, t))

    def fire_centre(self, t):
        return FIRE_ROOT + np.array([0.0, 0.4 * HF, 0.0])

    def fire_scale(self, t):
        return 1.0

    def power(self, t):
        return fire_bright(t)

    def forged(self, t):
        return 0.0

    def vortex_tilt(self, t):
        return np.eye(3)

    def vortex_grow(self, t):
        return 0.0

    def tower_lean(self, t):
        return 0.0

    def tower_rise_glow(self, t):
        return 1.0 - 0.7 * float(smoothstep(T_TOWERS + 70, T_TOWERS + 130, t))

    def tower_grow(self, t):
        return 1.0 + 0.8 * self.race(t)

    def tower_extra(self, towers, i, t):
        return 0.0


SCHED = C3Sched()


def fire_bright(t):
    """the flame's brightness: its light drawn out into the band (1200-1240), burning on beneath the Ring"""
    k = 1.0 - 0.35 * float(smoothstep(T_FORGE, T_FORGE + 16, t)) * (1.0 - float(smoothstep(T_FORGE + 60, T_WRITE, t)))
    return k * (1.0 + 0.05 * math.sin(0.23 * t) + 0.03 * math.sin(0.61 * t + 1.0))


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


# ================================================================ the towers ===

def layout_towers(tw):
    """C: the forge-towers of every realm round the fire, many, none taller than the rest, no pair. Eight forges
    ring the fire; ten more stand in a far ring between them. Heights are all alike (no tower leads), and every
    beat each surges by about the same (the lead passes round, never to two)."""
    r = rng(9031)
    k, n = tw.k, tw.k_all
    ns = n - k
    a0 = AZ0 - np.pi / k                                      # the camera's gap is between forges 0 and 1
    ang_f = a0 + 2 * np.pi * np.arange(k) / k + r.uniform(-0.09, 0.09, k)
    rad_f = 17.5 + r.uniform(-0.9, 0.9, k)
    ang_s = a0 + np.pi / k + np.pi / ns + 2 * np.pi * np.arange(ns) / ns + r.uniform(-0.12, 0.12, ns)
    rad_s = r.uniform(31.0, 39.0, ns)
    tw.ang = np.concatenate([ang_f, ang_s])
    tw.rad = np.concatenate([rad_f, rad_s])
    tw.rot = [(-a + np.pi) for a in tw.ang]
    # heights: near 21-24.5, far a little taller so their crowns show over the near ring (no single tallest)
    tw.h_rise = np.concatenate([r.uniform(21.0, 24.5, k), r.uniform(24.0, 28.0, ns)])
    # they rise from the dark one after another, all within two bars
    order = r.permutation(n)
    tw.t_rise = T_TOWERS + 8.0 + 120.0 * (order / max(n - 1, 1)) ** 0.9 + r.uniform(-4, 4, n)
    nb = len(RACE_BEATS)
    J = r.uniform(0.9, 1.3, (n, nb))
    lead = r.permutation(np.tile(np.arange(n), 2))[:nb]     # the lead passes round: one tower per beat, never a pair
    for b_, l_ in enumerate(lead):
        J[l_, b_] += 0.7
    tw.J = J
    tw.dly = r.uniform(0.0, 2.5, n)
    return tw


# ================================================================== the ring ===

def ring_frame(t):
    """(Rot, centre, scale) of the Ring. Forged standing on edge above the flame, a three-quarter view to the lens;
    it turns slowly (the letters travel); from 1360 it rises above the towers and tilts to hang over them."""
    cam_az = cam_azimuth(t)
    rise = float(smootherstep(T_RISE, T_RACE + 30, t))
    y = G + RING_Y0 + (RING_Y1 - RING_Y0) * rise + 3.0 * float(smoothstep(T_SURGE, T_RACE_END, t))
    y += 0.12 * math.sin(0.05 * t)                      # it breathes in the heat
    C = np.array([0.0, y, 0.0])
    # axis: horizontal, turned ~40 degrees off the line of sight (so its hole and its thickness both show) ->
    # tilted up toward vertical as it rises (it hangs over the towers, never face-on like a halo)
    yaw = cam_az + math.radians(40.0) + 0.0035 * (t - T_FORGE)
    tilt = math.radians(90.0) - math.radians(48.0) * rise
    Rt = B._rot_about(yaw, tilt)
    spin = 0.006 * (t - T_FORGE)
    c, s_ = math.cos(spin), math.sin(spin)
    Rs = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
    return Rt @ Rs, C, RING_W


def ring_on(t):
    return T_FORGE <= t < T_RACE_END


FRONT_T = (T_FORGE + 2.0, T_FORGE + 38.0)          # the white-hot front runs once round the circle
SHAPE_STROKES = [x for x in STROKES if x >= T_FORGE + 40]   # once the circle is closed, these beat it into a band
RING_Y0 = HF + 3.8                                # the band is forged just above the flame's tip ...
RING_Y1 = 38.0                                    # ... and rises to hang over the crowns (above ground)


def ring_theta_range(t):
    """(th0, th1): how much of the band exists yet (the front runs round from th0)"""
    k = float(smootherstep(FRONT_T[0], FRONT_T[1], t))
    th0 = -0.6
    return th0, th0 + max(k, 1e-3) * 2 * np.pi


def ring_heat(t):
    """fn(theta) -> 0..1: the thread is laid white-hot at the running front and glows yellow-orange behind it while
    it is beaten; every stroke re-heats it for a moment; after the last stroke it cools through orange and dull red
    to gold (cold by ~1322, when the letters burn up out of the metal)"""
    th0, th1 = ring_theta_range(t)
    strike = 0.0
    for tb in STROKES:
        x = t - tb
        if 0 <= x < 18:
            strike = max(strike, math.exp(-x / 4.0))
    beaten = float(smoothstep(T_FORGE - 2, T_FORGE + 6, t)) * (1.0 - float(smoothstep(T_WRITE - 24, T_WRITE + 2, t)))
    cooling = 1.0 - float(smoothstep(T_WRITE - 24, T_WRITE + 2, t))

    def f(th):
        frac = np.clip(((th - th0) % (2 * np.pi)) / (2 * np.pi), 0, 1)
        laid = FRONT_T[0] + (FRONT_T[1] - FRONT_T[0]) * frac                  # when the front passed here
        age = np.maximum(t - laid, 0.0)
        front = np.exp(-age / 5.0)                                          # white just behind the front
        h = 0.6 * beaten + 0.4 * front + 0.3 * strike * cooling
        return np.clip(h, 0, 1)
    return f


def ring_sec(t):
    """the band's section scale: a drawn thread of light, beaten out stroke by stroke into the full band"""
    k = sum(float(smoothstep(tb, tb + 5.0, t)) for tb in SHAPE_STROKES) / max(len(SHAPE_STROKES), 1)
    return (0.3 + 0.7 * k, 0.22 + 0.78 * k)


def ring_write(t):
    """fn(theta) -> 0..1: the letters burn up out of the metal in a sweep round the band from 1320"""
    def f(th):
        frac = ((th + 0.6) % (2 * np.pi)) / (2 * np.pi)
        tw = T_WRITE + 26.0 * frac
        return smoothstep(-1.0, 3.0, t - tw)
    return f


def ring_letters(t):
    """the letters' brightness: they flare as they burn up, then settle; faintly awake over the race"""
    if t < T_WRITE - 2:
        return 0.0
    flare = math.exp(-max(t - (T_WRITE + 14), 0.0) / 16.0)
    settle = 1.0 - 0.8 * float(smoothstep(T_RISE, T_RACE + 20, t))
    return 2.4 * (0.5 + 0.5 * flare) * settle


def ring_env(tl, t):
    """what the gold mirrors: the fire below, the forge throats around, the lit smoke above, the red of the race"""
    red = B.redness(t)
    e = RS.Env(above=(0.05, 0.024, 0.01), horizon=(0.32, 0.15, 0.055), below=(0.22, 0.09, 0.03))
    fb = fire_bright(t)
    e.point(FIRE_ROOT + np.array([0.0, 0.45 * HF, 0.0]), np.array([30.0, 19.0, 8.0]) * fb, 1.6)
    tw = tl.towers
    for i in range(tw.k_all):
        h = tw.height(i, t)
        if h < 3.0:
            continue
        top = tw.base(i) + np.array([0.0, h - 0.6, 0.0])
        pulse = 1.0 + 0.9 * SCHED.beat_pulse(t - tw.dly[i])
        c = np.array([3.4, 1.35, 0.4]) * (1 - 0.5 * red) + np.array([3.2, 0.5, 0.15]) * (0.5 * red)
        e.point(top, c * pulse * float(smoothstep(0.6, 1.0, h / max(tw.h_rise[i], 1.0))), 1.8)
    e.lobe([0.0, 1.0, 0.0], np.array([0.1, 0.045, 0.018]) * (1 + 0.8 * red), 1.2)
    if red > 0:
        e.lobe([0.0, -0.2, 1.0], np.array([0.25, 0.03, 0.01]) * red, 1.0)
        e.lobe([0.0, -0.2, -1.0], np.array([0.25, 0.03, 0.01]) * red, 1.0)
    return e


def ring_state(t):
    st = RS.RingState()
    st.heat = ring_heat(t) if t < T_WRITE + 10 else None
    st.hammer = 1.0 - float(smoothstep(T_WRITE - 10, T_WRITE + 4, t))
    st.sec = ring_sec(t)
    st.write = ring_write(t)
    st.letters = ring_letters(t)
    st.glow = 0.05
    st.exposure = 1.0
    st.alpha = 1.0
    return st


def ring_layer(tl, ctx):
    """the solid Ring for this frame (three instants across the shutter), joined to the occluder"""
    t = ctx.t
    if not ring_on(t):
        return None
    W, H = ctx.fr.W, ctx.fr.H
    ts = [ctx.t0, 0.5 * (ctx.t0 + ctx.t1), ctx.t1]
    acc = None
    for tq in ts:
        cam = tl.camera(tq)
        Rot, C, sc = ring_frame(tq)
        rgb, a, d = RS.render(cam, W, H, Rot, C, sc, ring_state(tq), ring_env(tl, tq),
                              th_range=ring_theta_range(tq))
        if acc is None:
            acc = [rgb, a, d]
        else:
            acc[0] += rgb
            acc[1] += a
            acc[2] = np.minimum(acc[2], d)
    rgb, a, d = acc[0] / 3.0, acc[1] / 3.0, acc[2]
    before = RS.merge_occluder(ctx.fr, a, d)
    vis = RS.visibility(before, d, H, W)
    return rgb * vis[..., None]


# ============================================================ the forging ===

class ForgeFX:
    """THE FORGING's light: the thread of light drawn out of the flame's tip to the white-hot front as it runs round
    (1200-1240), the front's own glare and sparks, and a burst of sparks off the band on every anvil stroke"""

    def __init__(self, seed=707):
        r = rng(seed)
        self.nt = 700
        self.th_u = np.sort(r.random(self.nt))
        self.th_j = r.normal(0, 1, (self.nt, 3))
        self.th_E = r.lognormal(0, 0.3, self.nt)
        # sparks per stroke
        per = 260
        ns = len(STROKES) * per
        self.s_t0 = np.repeat(np.array(STROKES), per) + r.uniform(0.0, 2.5, ns)
        self.s_th = r.uniform(0, 2 * np.pi, ns)
        self.s_psi = r.uniform(-np.pi, np.pi, ns)
        self.s_v = r.uniform(0.18, 0.55, ns)
        self.s_up = r.uniform(0.0, 0.25, ns)
        self.s_life = r.uniform(7.0, 17.0, ns)
        self.s_E = r.lognormal(0, 0.55, ns)
        # sparks streaming off the front as it runs
        nf = 900
        self.f_t0 = r.uniform(FRONT_T[0], FRONT_T[1], nf)
        self.f_v = r.normal(0, 1, (nf, 3))
        self.f_life = r.uniform(6.0, 14.0, nf)
        self.f_E = r.lognormal(0, 0.5, nf)

    @staticmethod
    def band_point(t, th, psi=0.0):
        Rot, C, sc = ring_frame(t)
        Pl, Nl = RS.local_points(np.atleast_1d(th), np.atleast_1d(np.full(np.shape(np.atleast_1d(th)), psi)))
        return C[None, :] + sc * Pl @ Rot.T, Nl @ Rot.T

    def emit(self, ctx, flame_tip):
        t = ctx.t
        if not (T_FORGE - 2 <= t < T_WRITE + 20):
            return
        # --- the thread: from the flame's tip up to the white-hot front
        k = float(smoothstep(FRONT_T[0] - 2, FRONT_T[0] + 4, t)) * (1.0 - float(smoothstep(FRONT_T[1] - 4, FRONT_T[1] + 8, t)))
        if k > 0:
            def thread(tq):
                th0, th1 = ring_theta_range(tq)
                F, _ = self.band_point(tq, th1, 0.0)
                F = F[0]
                A = flame_tip
                ctrl = A + np.array([0.0, 2.2, 0.0]) + 0.35 * (F - A)
                u = self.th_u[:, None]
                P = (1 - u) ** 2 * A + 2 * u * (1 - u) * ctrl + u ** 2 * F
                wob = 0.07 * np.sin(np.pi * u) * self.th_j
                return P + wob
            P0 = thread(ctx.t0)
            P1 = thread(ctx.t1)
            fpx = ctx.cam.f_px(1920)
            z = max(float((P1.mean(0) - ctx.cam.pos) @ ctx.cam.R[2]), 1.0)
            E = self.th_E * k * 3.5 * (0.5 + 0.5 * self.th_u) * (fpx / z) * 0.02
            col = np.array([1.0, 0.9, 0.7])
            ctx.fr.splat(P0, P1, 0.03, E, col, ctx.cam0, ctx.cam1, profile=1)
        # --- the front: a white glare where the band is being laid
        if FRONT_T[0] <= t < FRONT_T[1] + 4:
            th0, th1 = ring_theta_range(t)
            F, _ = self.band_point(t, th1, 0.0)
            fpx = ctx.cam.f_px(1920)
            z = max(float((F[0] - ctx.cam.pos) @ ctx.cam.R[2]), 1.0)
            g = 1.0 - float(smoothstep(FRONT_T[1] - 2, FRONT_T[1] + 4, t))
            ctx.fr.splat(F, F, np.array([0.5]), np.array([60.0 * g * (fpx / z) ** 2 * 0.002]),
                         np.array([[1.0, 0.95, 0.85]]), ctx.cam0, ctx.cam1, profile=1)
        # sparks off the running front
        a = t - self.f_t0
        m = (a >= 0) & (a < self.f_life)
        if m.any():
            idx = np.nonzero(m)[0]

            def fpos(tq):
                aa = np.maximum(tq - self.f_t0[idx], 0.0)
                # where the front was when each spark left it
                frac = (self.f_t0[idx] - FRONT_T[0]) / (FRONT_T[1] - FRONT_T[0])
                th0 = -0.6
                P, N = self.band_point(self.f_t0[idx][0], th0 + frac * 2 * np.pi, 0.0)
                return P + (N * 0.25 + self.f_v[idx] * 0.12) * aa[:, None] + np.array([0.0, -0.006, 0.0]) * (aa ** 2)[:, None]
            S0 = fpos(ctx.t0)
            S1 = fpos(ctx.t1)
            u = (t - self.f_t0[idx]) / self.f_life[idx]
            es = self.f_E[idx] * (1 - u) ** 1.5 * 2.2
            cs = look.blackbody(np.clip(0.95 - 0.45 * u, 0.4, 1.0))
            ctx.fr.splat(S0, S1, 0.012, es, cs, ctx.cam0, ctx.cam1)
        # --- the anvil strokes: sparks off the band, short, orange, falling and curving
        a = t - self.s_t0
        m = (a >= 0) & (a < self.s_life)
        if m.any():
            idx = np.nonzero(m)[0]
            tb = self.s_t0[idx]

            def spos(tq):
                P, N = self.band_point(T_FORGE + 40.0, self.s_th[idx], 0.0)
                Rot, C, sc = ring_frame(tq)
                Rot0, C0, _ = ring_frame(T_FORGE + 40.0)
                P = C + (P - C0) @ Rot0 @ Rot.T                     # ride with the band
                N = N @ Rot0 @ Rot.T
                aa = np.maximum(tq - tb, 0.0)
                v = N * self.s_v[idx][:, None] + np.array([0.0, 1.0, 0.0]) * self.s_up[idx][:, None]
                return P + v * aa[:, None] + np.array([0.0, -0.012, 0.0]) * (aa ** 2)[:, None]
            S0 = spos(ctx.t0)
            S1 = spos(ctx.t1)
            u = (t - tb) / self.s_life[idx]
            cold = float(smoothstep(T_FORGE + 60, T_WRITE, t))
            es = self.s_E[idx] * (1 - u) ** 1.4 * (3.0 - 1.4 * cold)
            cs = look.blackbody(np.clip(0.9 - 0.5 * u - 0.1 * cold, 0.35, 1.0))
            ctx.fr.splat(S0, S1, 0.01, es, cs, ctx.cam0, ctx.cam1)


# ============================================================= gold rain ===

class GoldRain:
    """C7: molten gold falls from the Ring into the towers' windows. In the hush a few slow drops, one and then
    another, each landing in a window that flares; from 1560 on every surge a rain of drops from all round the
    band into every forge. Each drop: a bright bead with a short tail; each landing: a flare and a splash."""

    G_ = 0.014                       # gravity (units / frame^2)

    def __init__(self, tl, seed=611):
        r = rng(seed)
        self.tl = tl
        tw = tl.towers
        slow_t = np.sort(r.uniform(T_RACE + 18, T_SURGE - 6, 13))
        fast_t = np.concatenate([np.full(34, tb) + r.uniform(-4, 10, 34) for tb in RACE_BEATS])
        self.t0 = np.concatenate([slow_t, fast_t])
        n = len(self.t0)
        self.slow = np.arange(n) < len(slow_t)
        self.dur = np.where(self.slow, r.uniform(34, 46, n), r.uniform(18, 28, n))
        self.th = r.uniform(0, 2 * np.pi, n)
        # the hush feeds the nearest forges (toward the lens); the race feeds all of them
        near = self._near_forges(tw)
        self.tw_i = np.where(self.slow, near[r.integers(0, len(near), n)], r.integers(0, tw.k_all, n))
        self.hy = np.where(self.slow, r.uniform(0.78, 0.93, n), r.uniform(0.6, 0.95, n))
        self.side = r.uniform(-0.6, 0.6, n)
        self.E = r.lognormal(0, 0.3, n) * np.where(self.slow, 1.5, 1.0)
        self.sz = np.where(self.slow, 1.4, 1.0) * r.uniform(0.8, 1.2, n)

    @staticmethod
    def _near_forges(tw):
        az = AZ0 + 0.35
        d = np.abs((tw.ang[:tw.k] - (az + np.pi) + np.pi) % (2 * np.pi) - np.pi)
        return np.argsort(d)[:4]

    def target(self, t_land):
        tw = self.tl.towers
        out = np.zeros((len(self.t0), 3))
        for i in np.unique(self.tw_i):
            m = self.tw_i == i
            b = tw.base(i)
            h = np.array([tw.height(i, x) for x in t_land[m]])
            inward = -np.array([b[0], 0.0, b[2]]) / max(math.hypot(b[0], b[2]), 1e-6)
            side = np.array([-inward[2], 0.0, inward[0]])
            out[m] = (b[None, :] + np.outer(h * self.hy[m], [0, 1, 0]) + inward[None, :] * 1.9
                      + side[None, :] * self.side[m][:, None])
        return out

    def start(self, t_det):
        """the drop leaves the band's lower edge, at its own place round the band"""
        out = np.zeros((len(self.t0), 3))
        for k_ in range(len(self.t0)):
            Rot, C, sc = ring_frame(t_det[k_])
            Pl, _ = RS.local_points(np.array([self.th[k_]]), np.array([0.0]))
            p = C + sc * Pl[0] @ Rot.T
            out[k_] = p
        return out

    def emit(self, ctx):
        t = ctx.t
        if t < T_RACE or t >= T_RACE_END:
            return
        live = (t >= self.t0 - 1) & (t < self.t0 + self.dur + 18)
        if not live.any():
            return
        if not hasattr(self, '_A'):
            self._A = self.start(self.t0)
            self._B = self.target(self.t0 + self.dur)
        A, Bt = self._A, self._B
        fpx = ctx.cam.f_px(1920)

        def pos(tq):
            u = np.clip((tq - self.t0) / self.dur, 0.0, 1.0)
            T_ = self.dur
            # ballistic: leaves the band with the velocity that lands it in its window at t0 + dur
            v0 = (Bt - A) / T_[:, None] + np.array([0.0, 0.5 * self.G_, 0.0]) * T_[:, None]
            s = u * T_
            return A + v0 * s[:, None] - np.array([0.0, 0.5 * self.G_, 0.0]) * (s ** 2)[:, None], u
        P0, u0 = pos(ctx.t0)
        P1, u1 = pos(ctx.t1)
        fly = live & (t >= self.t0) & (u1 < 1.0)
        if fly.any():
            z = np.maximum((P1[fly] - ctx.cam.pos) @ ctx.cam.R[2], 1.0)
            e = self.E[fly] * 5.0 * self.sz[fly] * (0.5 + 0.5 * smoothstep(0.0, 0.2, u1[fly])) * (fpx / z) ** 2 * 0.004
            core = np.array([1.0, 0.9, 0.62])
            ctx.fr.splat(P0[fly], P1[fly], 0.07 * self.sz[fly], e, core, ctx.cam0, ctx.cam1, zref=0.0)
            # its tail: where it was a frame and two frames ago
            for lag, g in ((1.2, 0.45), (2.6, 0.2)):
                Pa, _ = pos(ctx.t0 - lag)
                ctx.fr.splat(Pa[fly], P0[fly], 0.05 * self.sz[fly], e * g, C_GOLDEN, ctx.cam0, ctx.cam1, zref=0.0)
        # the windows flare where it lands, and a little splash
        land = t - (self.t0 + self.dur)
        fl = live & (land >= 0) & (land < 18)
        if fl.any():
            z = np.maximum((Bt[fl] - ctx.cam.pos) @ ctx.cam.R[2], 1.0)
            e = self.E[fl] * 40.0 * np.exp(-land[fl] / 5.0) * self.sz[fl] * (fpx / z) ** 2 * 0.004
            ctx.fr.splat(Bt[fl], Bt[fl], 0.9 * self.sz[fl], e, C_GOLDEN * 0.75 + np.array([1.0, 0.85, 0.5]) * 0.25,
                         ctx.cam0, ctx.cam1, profile=1, zref=0.0)


# ============================================================ the ground ===

def occ_vis(fr, z, H, W):
    """visibility (H, W) of a surface at camera depth z against the frame's occluder (towers + the Ring)"""
    if fr.occ is None:
        return np.ones((H, W), np.float32)
    import cv2
    Z, A = fr.occ[0][0], fr.occ[1][0]
    Zf = cv2.resize(Z, (Z.shape[1] * 2, Z.shape[0] * 2), interpolation=cv2.INTER_NEAREST)[:H, :W]
    Af = cv2.resize(A, (A.shape[1] * 2, A.shape[0] * 2), interpolation=cv2.INTER_LINEAR)[:H, :W]
    zz = z if np.ndim(z) == 0 else z
    return np.where(zz <= Zf + 0.3, 1.0, 1.0 - Af).astype(np.float32)


class GroundPool:
    """the fire lights the ground it burns on: a warm pool fading into the dark, cinders and ash in it; a screen-space
    shader of the ground plane (hidden behind the towers and the Ring), not a field of points"""

    def __init__(self, seed=515):
        import cv2
        r = rng(seed)
        n = 512
        tex = np.zeros((n, n), np.float32)
        for k, (cells, amp) in enumerate(((8, 0.5), (32, 0.3), (128, 0.2))):
            g = r.random((cells, cells)).astype(np.float32)
            tex += amp * cv2.resize(g, (n, n), interpolation=cv2.INTER_CUBIC)
        spk = (r.random((n, n)) > 0.985).astype(np.float32) * r.uniform(0.5, 1.5, (n, n)).astype(np.float32)
        self.tex = tex
        self.spk = cv2.GaussianBlur(spk, (0, 0), 0.8)
        self.span = 24.0                        # the texture covers [-span, span]^2 of ground

    def draw(self, out, cam, fr, scale, t):
        import cv2
        H, W = out.shape[:2]
        fb = fire_bright(t) * float(smoothstep(T_TOWERS, T_TOWERS + 50, t))   # (E15's fire burns in the black)
        if fb <= 0:
            return
        f = cam.f_px(W)
        xs = ((np.arange(W, dtype=np.float32) - (W - 1) / 2.0) / f)[None, :]
        ys = (((H - 1) / 2.0 - np.arange(H, dtype=np.float32)) / f)[:, None]
        R = cam.R
        dy = xs * R[0, 1] + ys * R[1, 1] + R[2, 1]
        hit = dy < -1e-4
        if not hit.any():
            return
        sd = np.where(hit, (G - cam.pos[1]) / np.where(hit, dy, -1.0), 0.0)            # camera depth to the ground
        px = cam.pos[0] + sd * (xs * R[0, 0] + ys * R[1, 0] + R[2, 0])
        pz = cam.pos[2] + sd * (xs * R[0, 2] + ys * R[1, 2] + R[2, 2])
        d = np.sqrt(px * px + pz * pz)
        near = hit & (d < self.span)
        if not near.any():
            return
        u = ((px / self.span) * 0.5 + 0.5) * (self.tex.shape[1] - 1)
        v = ((pz / self.span) * 0.5 + 0.5) * (self.tex.shape[0] - 1)
        um = np.where(near, u, 0).astype(np.float32)
        vm = np.where(near, v, 0).astype(np.float32)
        tx = cv2.remap(self.tex, um, vm, cv2.INTER_LINEAR)
        sp = cv2.remap(self.spk, um, vm, cv2.INTER_LINEAR)
        pool = 1.0 / (1.0 + (d / 2.8) ** 2)                                           # lit by the fire
        emb = np.exp(-d / 1.6)                                                         # the ember bed at its root
        L = fb * (0.26 * pool * (0.35 + 0.9 * tx) + 0.9 * emb * (0.4 + 0.8 * tx) + 0.5 * pool * sp)
        L = np.where(near, L, 0.0) * smoothstep(self.span, self.span * 0.6, d)
        vis = occ_vis(fr, sd, H, W)
        L = (L * vis).astype(np.float32)
        out[..., 0] += L * 1.0
        out[..., 1] += L * 0.5
        out[..., 2] += L * 0.2


# ================================================================ cameras ===

CAM_C6 = [  # (frame, radius, azimuth offset, height above ground, target height above ground, hfov)
    (1040, E15_R, 0.00, E15_Y, 0.0, 38.0),  # the gold fire alone: E15's (MAP's) view, 46 degrees above
    (1066, 24.0, 0.01, 15.0, 2.5, 42.0),    # descending as the first forges rise
    (1105, 27.0, 0.03, 4.5, 7.0, 50.0),     # low, outside the forges' ring, in a gap: they rise round the fire
    (1150, 28.0, 0.05, 3.5, 9.0, 54.0),
    (1195, 25.0, 0.07, 5.0, 10.0, 50.0),
    (1240, 18.0, 0.09, 7.5, 10.5, 42.0),    # pushing in through the gap to the band being beaten
    (1290, 14.0, 0.11, 9.0, 10.4, 36.0),
    (1330, 11.5, 0.12, 9.6, 10.2, 32.0),    # the letters burn up out of the metal
    (1360, 12.5, 0.13, 9.0, 12.0, 36.0),
    (1400, 20.0, 0.15, 8.0, 24.0, 52.0),    # the Ring rises above the towers: pull back and tilt up with it
    (1445, 28.0, 0.17, 14.0, 31.0, 60.0),
    (1500, 33.0, 0.19, 19.0, 31.5, 62.0),   # the hush: the Ring over the crowns
    (1560, 34.0, 0.21, 20.0, 32.0, 64.0),
    (1620, 34.5, 0.23, 20.5, 33.0, 65.0),   # the race
    (1680, 35.0, 0.25, 21.0, 34.0, 66.0),
]


def _cam_keys(t):
    k = [(f, np.array([r, a, y, ty, hf])) for f, r, a, y, ty, hf in CAM_C6]
    return catmull(t, k)


def cam_azimuth(t):
    r, a, y, ty, hf = _cam_keys(t)
    return AZ0 + a


def cam_c(t):
    r, a, y, ty, hf = _cam_keys(t)
    pos = np.array([r * math.cos(AZ0 + a), G + y, r * math.sin(AZ0 + a)])
    tgt = np.array([0.0, G + ty, 0.0]) + E15_TGT * (1.0 - float(smootherstep(T_TOWERS, T_TOWERS + 60, t)))
    sh = np.zeros(3)
    for tb in SCHED.pulses:
        x = t - tb
        if 0 <= x < 14:
            g = 0.12 if tb < T_WRITE else 0.25
            sh += g * math.exp(-x / 3.0) * np.array([math.sin(2.1 * x + tb), math.sin(2.9 * x + 0.5 * tb), 0.0])
    return pos + sh * 0.5, tgt + sh, float(hf)


def cam_az_dep(t):
    pos, tgt, hf = cam_c(t)
    Rot, C, sc = ring_frame(t)
    return math.atan2(pos[2], pos[0]), math.atan2(C[1] - pos[1], math.hypot(pos[0], pos[2]))


# ============================================================ time warps ===

EYE_WARP = [(T_EYE, 834.0), (T_SLIT - 14, 851.0), (T_SLIT, 857.0), (T_EYE_END, 879.4)]
GRASP_WARP = [(T_GRASP, 960.0), (T_GRASP_END, 1040.0)]


def warp(t, table):
    xs = [a for a, b in table]
    ys = [b for a, b in table]
    return float(np.interp(t, xs, ys))


# =============================================================== timeline ===

import timeline as TL  # noqa: E402


class TimelineC3(TL.Timeline):
    """THE FORGING and THE RACE on C's schedule; THE EYE and THE GRASP via a src Timeline at warped times"""

    def __init__(self):
        use_sched()
        super().__init__()
        self._src = None

    def _towers(self):
        tw = B.Towers()
        return layout_towers(tw)

    towers = property(lambda s: s._get('towers', s._towers))
    csparks = property(lambda s: s._get('csparks', CF.Sparks))
    forge = property(lambda s: s._get('forge', ForgeFX))
    gold_rain = property(lambda s: s._get('gold_rain', lambda: GoldRain(s)))
    ground = property(lambda s: s._get('ground', GroundPool))

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
        if t >= T_FORGE - 20:
            Rot, C, sc = ring_frame(t)
            k = float(smoothstep(T_FORGE - 20, T_FORGE + 30, t))
            focus = lerp(float(np.linalg.norm(FIRE_ROOT + np.array([0, 0.4 * HF, 0]) - pos)),
                         float(np.linalg.norm(C - pos)), k)
        else:
            focus = float(np.linalg.norm(FIRE_ROOT + np.array([0, 0.4 * HF, 0]) - pos))
        return Camera(pos, tgt, hfov=hf, focus=focus, aperture=0.03)

    @staticmethod
    def srct(t):
        if T_EYE - 1 <= t < T_EYE_END + 1:
            return warp(t, EYE_WARP)
        return warp(t, GRASP_WARP)

    def light(self, t):
        red = B.redness(t)
        col = np.array([1.0, 0.76, 0.4])
        col = col * (1 - 0.6 * red) + look.hexrgb(look.PALETTE['race_red']) * 0.6 * red
        return FIRE_ROOT + np.array([0.0, 0.4 * HF, 0.0]), col, 60.0 * fire_bright(t)

    def render_opts(self, f):
        if self.mode(f) == 'src':
            return self.src().render_opts(int(round(self.srct(f))))
        return dict(bokeh_pow=0.25, bokeh_cap=1.6, fog_start=60.0, fog_len=80.0, near=0.3)

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
        ctx.ring_rgb = ring_layer(self, ctx)            # the Ring hides what is behind it (occluder)
        self.dust.emit(ctx)
        self.smoke.emit(ctx, lp, lc, lpw)
        self.towers.emit(ctx, lp, lc, lpw)
        self.tembers.emit(ctx)
        self.tsmoke.emit(ctx, lp, lc, lpw)
        self.walls.emit(ctx)
        # (no white spark sprays in C: the surges show in the throats, the climbing heat and the gold rain)
        self.csparks.emit(ctx, FIRE_ROOT, HF, bright=fire_bright(t), amount=1.0)
        tip = FIRE_ROOT + np.array([0.0, 0.92 * HF, 0.0])
        self.forge.emit(ctx, tip)
        self.gold_rain.emit(ctx)

    def post(self, ctx, hdr):
        if hasattr(ctx, 'c3'):
            return self.src().post(ctx, hdr)
        t = ctx.t
        p = ctx.fr.prm
        H, W = hdr.shape[:2]
        band = None
        if p[10] > 0:                                       # calm the text band as the splats are calmed
            y = np.arange(H, dtype=np.float32)
            a = np.clip((y - (p[8] - p[11])) / (2 * p[11]), 0, 1)
            a = a * a * (3 - 2 * a)
            b = np.clip((y - (p[9] - p[11])) / (2 * p[11]), 0, 1)
            b = b * b * (3 - 2 * b)
            band = (1.0 - p[10] * a * (1.0 - b))[:, None, None].astype(np.float32)
        lay = np.zeros_like(hdr)
        cam = ctx.cam
        # the ground the fire burns on (hidden behind the towers and the Ring)
        self.ground.draw(lay, cam, ctx.fr, ctx.scale, t)
        # the flame (cflame: the same flame E15 hands over at 1039)
        pts = np.stack([FIRE_ROOT, FIRE_ROOT + np.array([0.0, HF, 0.0]), FIRE_ROOT + np.array([0.0, 0.4 * HF, 0.0])])
        u, v, z = cam.project(pts, 1920, 804)
        if z[0] > 0.3 and z[1] > 0.3:
            vis = occ_vis(ctx.fr, float(z[2]), H, W)
            CF.draw(lay, (u[0], v[0]), (u[1], v[1]), t, bright=fire_bright(t), calm=1.0 - 0.5 * SCHED.race(t),
                    vis=vis, scale=ctx.scale)
        rgb = getattr(ctx, 'ring_rgb', None)
        if rgb is not None:
            lay = lay + rgb
        if band is not None:
            lay = lay * band
        return hdr + lay

    def finish_opts(self, f):
        if self.mode(f) == 'src':
            return self.src().finish_opts(int(round(self.srct(f))))
        return dict(exposure=1.0, bloom_strength=0.14, bloom_threshold=0.7, streak_strength=0.0,
                    vignette_amount=0.25)
