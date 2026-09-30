"""EMBERS part B (480-880): ignition, the thinking fire, towers, crown, the race, the vortex."""
import math
import os

import numpy as np

import look
import variant
from core import (clamp01, smoothstep, smootherstep, ease_out, ease_in, ease_in_out, ease_out_back,
                  ease_out_expo, lerp, vnoise, snoise, rng, rand_dirs, fib_sphere, catmull)
import towers as TW

C_CORE = look.hexrgb(look.PALETTE['mind_core'])
C_ICE = look.hexrgb(look.PALETTE['mind_ice'])
C_GOLD = look.hexrgb(look.PALETTE['mind_gold'])
C_RED = look.hexrgb(look.PALETTE['race_red'])
C_CRIMSON = look.hexrgb(look.PALETTE['race_crimson'])
C_EMBER = look.hexrgb(look.PALETTE['ember'])

GROUND = -14.0
TOWER_ANG0 = 0.35
IGN = 480.0
BEATS = [640 + 20 * k for k in range(16)]          # 640 .. 940

# v3: a cut's own schedule (a3.py), in that cut's frames (BIBLE_V3 rev. 1 locked beat sheets). None = the v2 src
# timeline, which every function below keeps exactly. When set, IGN and BEATS are replaced by its values too.
SCHED = None


# ------------------------------------------------------------ timelines ---

def redness(t):
    """0 -> 1 palette bleed toward crimson during the race (decisive by ~780)."""
    if SCHED is not None:
        return SCHED.redness(t)
    return float(smoothstep(646, 780, t))


def beat_pulse(t):
    """1 on each race beat, decaying fast (for brightness pulses)."""
    if SCHED is not None:
        return SCHED.beat_pulse(t)
    if t < 640 or t >= 960:
        return 0.0
    x = (t - 640) % 20.0
    return math.exp(-x / 3.0)


STORM_LIFT = 16.0


def crown_centre(t):
    if SCHED is not None:
        return SCHED.fire_centre(t)
    y = 20.0 * smootherstep(548, 634, t) + 28.0 * smoothstep(640, 820, t) + 6.0 * smoothstep(820, 900, t)
    if t < 880:
        # v2: the storm climbs clear of the towers' crowns as it grows (they are solid now and would hide its
        # eye); the grasp (960+) keeps v1's geometry
        y += STORM_LIFT * smootherstep(804, 852, t)
    return np.array([0.0, float(y), 0.0])


# v3 (picture red team: "a candle"): while it is a flame, the thinking fire breathes once a bar (80 frames at 72 BPM),
# a slow, perfectly even swell of size, light and tongues together -- too regular for a flame -- and its own motion
# runs on a slower clock (FLAME_CLOCK). Everything v3 adds is weighted by flame_w(), so the crown / Ring are as before.
BREATH_P = 80.0
FLAME_CLOCK = 0.55
MIND_BODY_DIM = 0.55      # the flame's body is dimmed by this much so its filaments show through it
MIND_TONGUE_DIM = 0.55     # translucent tongues
MIND_GLOW = 0.6           # the flame's halo
MIND_STRETCH = 1.25       # a taller body (v2 1.05), so the tongues crown a body rather than make the flame
MIND_NARROW = 0.8         # v3 (H5): a broad body under the one tongue (0.62 made a torch / a candle)
MIND_SOFT = 0.03         # softer splats while it is a flame: a luminous body, not grain
MIND_RIM_GOLD = 0.95      # gold on the silhouette only
MIND_RIM_E = 4.0          # ... a little brighter there
C_MIND_ICE = C_CORE * 0.45 + C_ICE * 0.55                        # ice-WHITE: a pale body, never a gas-blue one
C_MIND_EDGE = C_GOLD * 0.6 + np.array([1.0, 0.72, 0.30]) * 0.4    # a deeper gold than mind_gold (never orange)
MIND_FIL = 0.9            # ... and its filaments brightened by this much
MIND_FIL_UP = 1.55        # the tree fills the body (v2 2.1 drew it out into vertical streaks)


def breath(t):
    """-1..1, the fire's even breath (lowest at ignition, fullest 40 frames later, once a bar)"""
    return math.sin(2 * math.pi * (t - IGN) / BREATH_P - math.pi / 2)


def fire_radius(t):
    grow = float(ease_out_expo((t - IGN) / 22.0, 6.0))
    breath_ = 1 + (0.07 + 0.05 * flame_w(t)) * breath(t)
    r = 1.3 * grow * breath_
    if SCHED is not None:
        r *= SCHED.fire_scale(t)
    return r


def crown_morph(t):
    if SCHED is not None:
        return 0.0                       # v3 A: no crown -- the fire itself is the prize
    return float(smootherstep(562, 628, t))


def _rot_about(a, tau):
    """rotation by tau about the horizontal axis perpendicular to azimuth a (tips +Y toward azimuth a)"""
    axis = np.array([math.sin(a), 0.0, -math.cos(a)])
    c, s_ = math.cos(tau), math.sin(tau)
    x, y, z = axis
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s_, x * z * C + y * s_],
                     [y * x * C + z * s_, c + y * y * C, y * z * C - x * s_],
                     [z * x * C - y * s_, z * y * C + x * s_, c + z * z * C]])


def cam_az_dep(t):
    """azimuth of the (unshaken) race camera and its depression below the crown (rad, > 0: camera below)"""
    k = [(f, np.array([r, a, y, ty])) for f, r, a, y, ty in _cam_keys()]
    r, a, y, ty = catmull(t, k)
    pos = _polar(r, ALPHA_C + a, y)
    C = crown_centre(t)
    return math.atan2(pos[2], pos[0]), math.atan2(C[1] - pos[1], math.hypot(pos[0] - C[0], pos[2] - C[2]))


def crown_tilt(t):
    """Rotation (3x3) tilting the crown's axis toward the camera (hero angle for a floating ring).
    v2: from 798 the ring keeps turning its face toward the lens as the storm is born (v1 flattened it and the
    ring passed edge-on); it faces the lens by ~842."""
    if SCHED is not None:
        return np.eye(3)
    tau = 0.38 * crown_morph(t)
    a = ALPHA_C
    if t >= 960:
        return _rot_about(ALPHA_C - 1.35, 0.42)
    if t >= 798:
        az, dep = cam_az_dep(t)
        s2 = float(smootherstep(798, 842, t))
        a = lerp(ALPHA_C, az, s2)
        tau = lerp(tau, math.pi / 2 + dep, s2)
    return _rot_about(a, tau)


VORTEX_VIEW = math.radians(38.0)


def vortex_tilt(t):
    """v2: the storm's disc turns its underside to the lens as it grows, so it is seen at ~38 degrees from the
    first frame of its growth to the cut (v1: nearly edge-on around 805-825, a flat bright smear)."""
    if SCHED is not None:
        return SCHED.vortex_tilt(t)
    if t < 790 or t >= 960:
        return np.eye(3)
    az, dep = cam_az_dep(t)
    beta = float(smoothstep(790, 812, t)) * max(VORTEX_VIEW - dep, 0.0)
    return _rot_about(az, -beta)


KINGDOM_ROT_GRASP = 0.64


def kingdom_rot(t):
    """v2: for the grasp shot (after the hard cut at 960) the ring of towers is turned about the centre axis so a
    gap between two slender towers lies in front of the storm's eye (solid towers would otherwise stand across it:
    the eye backlights the hand, and in cut C the Ring hangs there). Rigid rotation, world -> world."""
    if t < 960 or SCHED is not None:
        return None
    c, s = math.cos(KINGDOM_ROT_GRASP), math.sin(KINGDOM_ROT_GRASP)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _krot(P, t):
    R = kingdom_rot(t)
    if R is None:
        return P
    P = np.asarray(P)
    return P @ R.T.astype(P.dtype)


def forged(t):
    """cut C: 0 -> 1 as the thinking fire is forged into the Ring (the fire's own structure gives way)"""
    if SCHED is not None:
        return SCHED.forged(t) if hasattr(SCHED, 'forged') else 0.0
    if not variant.tolkien():
        return 0.0
    return float(smoothstep(596, 626, t))


def eye_k(t):
    """cut C: the Eye resolving in the storm's centre"""
    if not variant.tolkien() or t >= 880:
        return 0.0
    return float(smootherstep(830, 856, t))


def storm_fade(t):
    """the crown's flames (tines, tongues, spark column) are drawn into the storm as it is born"""
    if t >= 960:
        return 1.0
    return 1.0 - float(smoothstep(802, 830, t))


def fire_power(t):
    """overall luminous power of the fire/crown (drives tower lighting too)."""
    if t < IGN:
        return 0.0
    p = 1.0 + 3.0 * math.exp(-(t - IGN) / 8.0)
    p *= 1 + (0.12 + 0.16 * flame_w(t)) * breath(t)
    if SCHED is not None:
        return p * SCHED.power(t) * (1 + 0.45 * beat_pulse(t))
    p *= 1 + 0.5 * smoothstep(640, 900, t)
    p *= 1 + 0.45 * beat_pulse(t)
    return p


# ------------------------------------------------------------ the fire ---

def _build_filaments(seed=5, n_roots=18):
    r = rng(seed)
    P, S, M, D = [], [], [], []
    roots = fib_sphere(n_roots, seed=0.7)

    def grow(p, d, s, m, depth, length):
        step = 0.008
        n = int(length / step)
        for i in range(n):
            d = d + r.normal(0, 0.16, 3)
            rp = np.linalg.norm(p)
            if rp > 0.78:
                d = d - p / rp * (rp - 0.78) * 6.0
            d /= np.linalg.norm(d)
            p = p + d * step
            s += step
            k = 3 if depth == 0 else 2
            P.append(p + r.normal(0, 0.004, (k, 3)))
            S.append(np.full(k, s))
            M.append(np.full(k, m))
            D.append(np.full(k, depth))
            if depth < 3 and r.random() < 0.05 - 0.01 * depth:
                nd = d + r.normal(0, 0.9, 3)
                nd /= np.linalg.norm(nd)
                grow(p.copy(), nd, s, m, depth + 1, length * r.uniform(0.3, 0.55))
    for m, d0 in enumerate(roots):
        grow(d0 * 0.06, d0.copy(), 0.0, m, 0, r.uniform(0.7, 0.95))
    return (np.concatenate(P), np.concatenate(S), np.concatenate(M).astype(int),
            np.concatenate(D).astype(int))


# v2 (critic note M2): the thinking fire must read as a FLAME, never a hanging bulb: a pointed tip inside the frame,
# three asymmetric licking tongues, the white core up in the body (the base reaches down below it), filaments
# reaching out past the body so no smooth envelope edge shows, a glow drawn up the flame. All of this belongs to the
# fire before it opens into the crown / Ring: it is weighted by (1 - crown_morph), so the race is unchanged.
FLAME_TONGUES = [  # base azimuth, height above the orb centre, width, lean, flicker rate, phase
    (0.0, 2.75, 0.5, 0.08, 0.155, 0.0),       # v3 (H5): ONE tapering tongue on the axis -- never a fork
]


def flame_w(t):
    return 1.0 - crown_morph(t)


def flame_shape(p, fw, stretch_up=1.05, reach=1.0, down=1.2, narrow=0.62, lift=0.0):
    """orb-coordinate points -> the flame body (a narrow teardrop, the core about a third of the way up):
    the base drawn down and rounded below the core, the top thinned toward the tongues. fw: 1 flame, 0 unchanged."""
    if fw <= 0:
        return p
    x, y, z = p[:, 0], p[:, 1] + lift * fw, p[:, 2]
    yb = np.where(y < 0, y * (1.0 + (down - 1.0) * fw), y * (1.0 + (stretch_up - 1.0) * fw))
    taper = 1.0 - fw * (0.32 * smoothstep(-0.1, -0.95, yb) + 0.55 * smoothstep(-0.1, 1.45, yb))
    k = taper * (1.0 + (narrow * reach - 1.0) * fw)
    return np.stack([x * k, yb, z * k], 1)


def tongue_axis(kt, s, t, fw):
    """centreline of flame tongue kt (array) at progress s in [0,1] (orb units): rises from the upper body, leans and
    sways, a travelling S-wave licks up it"""
    T = np.array(FLAME_TONGUES)
    th, H, W, lean, om, ph = [T[kt, j] for j in range(6)]
    # v3: the tongue rises and sinks on the breath (no flicker), with only a slow drift of its own
    Ht = H * (1.0 + 0.16 * breath(t) + 0.035 * np.sin(om * t + ph))
    y0 = 0.55
    y = y0 + s * (Ht - y0)
    if len(T) == 1:
        # v3 (H5): one tongue on the axis, drawn to a single point; it leans a little on a slow, even sway and a
        # gentle S-wave climbs it (too even for a flame), so its tip never splits
        lean_a = 0.021 * t + ph
        lick = 0.06 * s * (1.0 - 0.65 * s) * np.sin(2 * np.pi * 1.1 * s - om * 1.4 * t + ph)
        px = lean * s * s * np.cos(lean_a) + lick
        pz = lean * s * s * np.sin(0.8 * lean_a + 1.0) + 0.7 * lick * np.cos(0.5 * t * om)
        w = W * (1.0 - s) ** 1.15 * (1.0 + 0.06 * np.sin(1.3 * om * t + 4.0 * s + ph))
        return np.stack([px, y, pz], 1), w
    rb = 0.2 * (1.0 - 0.55 * s)
    sway = th + 0.3 * np.sin(0.035 * t + ph)
    lick = 0.13 * s * np.sin(2 * np.pi * 1.4 * s - om * 1.6 * t + ph)
    px = rb * np.cos(th) + lean * s * s * np.cos(sway) - lick * np.sin(th)
    pz = rb * np.sin(th) + lean * s * s * np.sin(sway) + lick * np.cos(th)
    w = W * (1.0 - s) ** 0.72 * (1.0 + 0.18 * np.sin(1.3 * om * t + 5.0 * s + ph))
    return np.stack([px, y, pz], 1), w


def _mind_cols():
    """the thinking fire's (core, body, edge) colours: A's mind palette, or a schedule's own (C: gold)"""
    if SCHED is not None and hasattr(SCHED, 'mind_cols'):
        return SCHED.mind_cols
    return C_CORE, C_MIND_ICE, C_MIND_EDGE


def _side(cam, C):
    """unit horizontal vector across the line of sight (screen-horizontal at C): silhouettes lie along it"""
    v = np.asarray(C, np.float64) - cam.pos
    s_ = np.array([-v[2], 0.0, v[0]])
    return s_ / max(np.linalg.norm(s_), 1e-9)


def _rim(q, s_hat, nb=28):
    """0..1: how close an orb-space point lies to the flame body's LEFT/RIGHT silhouette as seen from the camera
    (lateral offset over the body's envelope radius at that height)"""
    lat = np.abs(q[:, 0] * s_hat[0] + q[:, 2] * s_hat[2])
    rho = np.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2)
    y = q[:, 1]
    lo, hi = np.percentile(y, 1), np.percentile(y, 99.5)
    b = np.clip(((y - lo) / max(hi - lo, 1e-6) * nb).astype(int), 0, nb - 1)
    env = np.zeros(nb)
    for i in range(nb):
        m = b == i
        env[i] = np.percentile(rho[m], 96) if m.sum() > 20 else 0.3
    env = np.convolve(np.pad(env, 1, mode='edge'), np.ones(3) / 3, 'valid')
    return smoothstep(0.58, 0.86, lat / np.maximum(env[b], 1e-3))


class MindFire:
    """The thinking fire: toroidal flow, gold flame tongues, white core, neural filaments.
    Morphs into a floating crown/ring from ~562."""

    def __init__(self, seed=21):
        r = rng(seed)
        n = 110000
        self.n = n
        rc = r.uniform(0.22, 0.62, n)
        a = np.minimum(r.uniform(0.07, 0.46, n), rc + 0.04)
        self.rc, self.a = rc, a
        self.b = a * r.uniform(1.1, 1.7, n)
        self.y0 = r.normal(0, 0.05, n)
        self.th0 = r.uniform(0, 2 * np.pi, n)
        self.om = 0.075 / np.sqrt(a / 0.3)
        self.ph0 = r.uniform(0, 2 * np.pi, n)
        self.omp = r.uniform(0.006, 0.02, n)
        self.E = r.lognormal(0, 0.5, n)
        self.fl = r.uniform(0, 2 * np.pi, n)
        # flame tongues (gold edges)
        m = 42000
        self.m = m
        d = rand_dirs(r, m)
        d[:, 1] = np.where(d[:, 1] < -0.45, -d[:, 1], d[:, 1])
        self.td = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.tph = r.random(m)
        self.tv = r.uniform(0.018, 0.04, m)
        self.tE = r.lognormal(0, 0.5, m)
        # core
        k = 7000
        self.k = k
        self.cd = rand_dirs(r, k) * (r.random(k) ** 1.5)[:, None]
        # filaments
        fp, fs, fm, fd = _build_filaments()
        self.fp, self.fs, self.fm, self.fdep = fp, fs, fm, fd
        self.nroot = fm.max() + 1
        self.pv = r.uniform(0.022, 0.04, (self.nroot, 4))
        self.pp = r.uniform(0, 10, (self.nroot, 4))
        self.pdir = r.choice([1.0, 1.0, -1.0], (self.nroot, 4))
        r2 = rng(seed + 900)
        T = np.array(FLAME_TONGUES)
        wt = T[:, 1] * T[:, 2]
        self.kt = r2.choice(len(T), m, p=wt / wt.sum())
        a_ = r2.uniform(0, 2 * np.pi, m)
        rr_ = np.sqrt(r2.random(m))
        self.tu = np.stack([rr_ * np.cos(a_), rr_ * np.sin(a_)], 1)
        # v3 (H5: "visible inner structure, embers and filaments, so it's a fire, not a ghost"): inner embers,
        # slow motes climbing through the body and up the tongue on the fire's calm clock
        r3 = rng(seed + 1700)
        q = 1100
        self.mo_ph = r3.random(q)
        self.mo_v = r3.uniform(0.0035, 0.008, q)
        self.mo_u = r3.normal(0, 1, (q, 2)) * 0.62
        self.mo_E = r3.lognormal(0, 0.55, q)
        self.mo_tw = r3.uniform(0, 2 * np.pi, q)
        print('mindfire: flow', n, 'tongues', m, 'filament pts', len(fp))

    # --- warp from orb coordinates (unit ball) to the ring/crown
    @staticmethod
    def warp(p, m, R, t):
        """p: points in units of the orb radius (centred). Returns world offsets."""
        if m <= 0:
            return p * R
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        rho = np.sqrt(x * x + z * z)
        phi = np.arctan2(z, x)
        Rr = 5.0 + 0.6 * smoothstep(640, 800, t)
        rho2 = lerp(rho * R, Rr + (rho - 0.45) * 0.62 * 1.5, m)
        y2 = lerp(y * R, y * 0.4 * 1.4, m)
        phi2 = phi + m * 0.012 * (t - 562)
        return np.stack([rho2 * np.cos(phi2), y2, rho2 * np.sin(phi2)], 1)

    def motes(self, t):
        """orb-space positions of the inner embers: up the body, then up the tongue's own axis"""
        k = (self.mo_ph + self.mo_v * FLAME_CLOCK * (t - IGN)) % 1.0
        T = np.array(FLAME_TONGUES)
        Ht = T[0, 1] * (1.0 + 0.16 * breath(t))
        y = -0.35 + k * (Ht - 0.1 + 0.35)
        rad = np.interp(y, [-0.35, 0.1, 0.6, 1.3, 2.0, Ht], [0.18, 0.4, 0.4, 0.3, 0.13, 0.0])
        s_ = np.clip((y - 0.55) / max(Ht - 0.55, 1e-3), 0, 1)
        ax, _ = tongue_axis(np.zeros(len(k), int), s_, t, 1.0)
        a = smoothstep(0.35, 0.75, y)
        x = ax[:, 0] * a + self.mo_u[:, 0] * rad
        z = ax[:, 2] * a + self.mo_u[:, 1] * rad
        return np.stack([x, y, z], 1)

    def flow_pts(self, t):
        tt = t - IGN
        th = self.th0 - self.om * tt
        rho = self.rc + self.a * np.cos(th)
        y = self.y0 + self.b * np.sin(th)
        y = np.where(y > 0, y * 1.55, y * 0.85)
        rho = rho * (1 - 0.42 * smoothstep(0.0, 1.3, y))          # teardrop: narrow toward the tip
        ph = self.ph0 + self.omp * tt
        p = np.stack([rho * np.cos(ph), y, rho * np.sin(ph)], 1)
        w = vnoise(p * 1.0 + np.array([0.0, -0.045 * tt, 0.0]), 1.7, (0.0, 0.0, 0.0), 2)
        amp = 0.06 + 0.16 * smoothstep(0.0, 1.3, y)
        return p + w * amp[:, None] + np.array([0, 0.1, 0]) * (smoothstep(0.3, 1.3, y) * w[:, 1] ** 2)[:, None]

    def emit(self, ctx):
        t = ctx.t
        if t < IGN or t >= (960 if SCHED is None else SCHED.end):
            return
        R0, R1 = fire_radius(ctx.t0), fire_radius(ctx.t1)
        m = crown_morph(t)
        C0, C1 = crown_centre(ctx.t0), crown_centre(ctx.t1)
        red = redness(t)
        pw = fire_power(t)
        # --- flow particles
        q0 = self.flow_pts(ctx.t0)
        q1 = self.flow_pts(ctx.t1)
        fw = flame_w(t)
        if fw > 0:
            # a looser edge: the outer flow breaks up (no smooth envelope)
            wv = vnoise(q1 * 1.4 + np.array([0.0, -0.09 * FLAME_CLOCK * (t - IGN), 0.0]), 1.3, (5.0, 1.0, 2.0), 2)
            edge = smoothstep(0.45, 1.0, np.linalg.norm(q1, axis=1))
            q0 = q0 + wv * (0.09 * fw * edge)[:, None]              # v3: calmer (v2 0.16)
            q1 = q1 + wv * (0.09 * fw * edge)[:, None]
        yc = 0.08 + 0.3 * fw                                           # the hot zone sits up in the body
        d = np.sqrt(q1[:, 0] ** 2 + q1[:, 2] ** 2 + ((q1[:, 1] - yc) / (1.0 + 0.6 * fw)) ** 2)
        low = smoothstep(0.1, -0.6, q1[:, 1]) * fw                   # a cooler, dimmer base
        q0 = flame_shape(q0, flame_w(ctx.t0), stretch_up=MIND_STRETCH, lift=0.32, narrow=MIND_NARROW)
        q1 = flame_shape(q1, flame_w(ctx.t1), stretch_up=MIND_STRETCH, lift=0.32, narrow=MIND_NARROW)
        s_hat = _side(ctx.cam, C1)
        P0 = C0 + self.warp(q0, crown_morph(ctx.t0), R0, ctx.t0) @ crown_tilt(ctx.t0).T
        P1 = C1 + self.warp(q1, crown_morph(ctx.t1), R1, ctx.t1) @ crown_tilt(ctx.t1).T
        if fw > 0:
            # v3 (red team: a candle): the MIND palette, not a flame's. White core, an ice-white body that stays
            # ice-white to the top (no blue base, no gold crown), gold only on its outer edge
            c1 = lerp(smoothstep(0.08, 0.42, d), smoothstep(0.05, 0.62, d), fw)[:, None]
            rim_b = _rim(q1, s_hat)
            c2 = lerp(smoothstep(0.5, 0.95, d), MIND_RIM_GOLD * rim_b, fw)[:, None]
        else:
            c1 = smoothstep(0.08, 0.42, d)[:, None]
            c2 = smoothstep(0.5, 0.95, d)[:, None]
        m_core, m_ice, m_edge = _mind_cols()
        gold = C_GOLD * (1 - 0.55 * red) + C_RED * 0.55 * red
        if fw > 0:
            gold = gold * (1 - fw) + m_edge * fw
        ice = C_ICE * (1 - fw) + m_ice * fw
        col = m_core * (1 - c1) + ice * c1
        col = col * (1 - c2) + gold * c2
        fl = 1 + lerp(0.35, 0.1, fw) * np.sin(0.7 * t + self.fl)     # v3: no sparkle while it is a flame
        e = self.E * fl * (1.9 - 1.1 * d) * 0.55 * pw
        if fw > 0:
            # a calm, translucent body (dimmer, so the filaments inside it show); a slightly dimmer base
            e = e * (1.0 - 0.45 * low) * (1.0 - MIND_BODY_DIM * fw) * (1.0 + MIND_RIM_E * fw * rim_b)
        e *= 1.0 + 0.6 * m          # ring is thinner: keep it bright
        e *= (1.0 - 0.85 * forged(t)) * (1.0 - eye_k(t))   # cut C: the fire becomes the band (a faint aura stays)
        ctx.fr.splat(P0, P1, 0.004 + MIND_SOFT * fw, e, col, ctx.cam0, ctx.cam1)
        # --- core
        cj = self.cd * 0.2 + 0.01 * np.sin(np.array([31.0, 47.0, 53.0]) * t)
        if fw > 0:
            cj = cj * np.array([1.0 - 0.3 * fw, 1.0 + 0.9 * fw, 1.0 - 0.3 * fw]) + np.array([0.0, 0.5 * fw, 0.0])
        if m < 1:
            Pc0 = C0 + self.warp(cj, crown_morph(ctx.t0), R0, ctx.t0) @ crown_tilt(ctx.t0).T
            Pc1 = C1 + self.warp(cj, crown_morph(ctx.t1), R1, ctx.t1) @ crown_tilt(ctx.t1).T
            ec = np.full(self.k, 1.8 * pw * (1 - 0.5 * m) * (1 - 0.4 * fw))
            ctx.fr.splat(Pc0, Pc1, 0.003, ec, C_CORE, ctx.cam0, ctx.cam1)
        # --- tongues: rise from the upper surface, cool to gold/orange
        kk0 = (self.tph + self.tv * (ctx.t0 - IGN)) % 1.0
        kk1 = (self.tph + self.tv * (ctx.t1 - IGN)) % 1.0
        wrap_ok = kk1 >= kk0
        # v3: the flame's tongues run on the slow clock (the crown's plume keeps its own)
        kf0 = (self.tph + self.tv * FLAME_CLOCK * (ctx.t0 - IGN)) % 1.0
        kf1 = (self.tph + self.tv * FLAME_CLOCK * (ctx.t1 - IGN)) % 1.0
        wrap_f = kf1 >= kf0

        def tongue(kk, tq, kf):
            base = self.td * 0.92
            base = base * np.array([1.0, 1.3, 1.0])
            conv = 1 - 0.8 * kk * (1 - crown_morph(tq))
            base = base * np.stack([conv, np.ones_like(conv), conv], 1)
            hgt = (0.9 + 1.9 * np.clip(self.td[:, 1], 0, 1)) * (1 - crown_morph(tq)) + 2.6 * crown_morph(tq)
            p = base + np.array([0, 1.0, 0]) * (hgt * kk ** 1.35)[:, None]
            w = vnoise(p + np.array([0, -0.06 * (tq - IGN), 0]), 2.0, (0, 0, 0), 2)
            p_old = p + w * (0.06 + 0.3 * kk)[:, None]
            fwq = flame_w(tq)
            if fwq <= 0:
                return p_old
            # v2: three asymmetric tongues licking up out of the body (v1's symmetric plume read as a bulb's neck)
            sq = kf ** 0.85
            ax, wd = tongue_axis(self.kt, sq, tq, fwq)
            th = np.array(FLAME_TONGUES)[self.kt, 0]
            e1 = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], 1)
            e2 = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1)
            pn = ax + wd[:, None] * (self.tu[:, :1] * e1 + self.tu[:, 1:] * e2 * 0.8)
            wn = vnoise(pn * 1.6 + np.array([0.0, -0.11 * FLAME_CLOCK * (tq - IGN), 0.0]), 1.5, (0, 0, 0), 2)
            pn = pn + wn * (0.02 + 0.08 * sq)[:, None]             # v3: laminar, not turbulent
            return lerp(p_old, pn, fwq)
        T0 = C0 + self.warp(tongue(kk0, ctx.t0, kf0), crown_morph(ctx.t0), R0, ctx.t0) @ crown_tilt(ctx.t0).T
        T1 = C1 + self.warp(tongue(kk1, ctx.t1, kf1), crown_morph(ctx.t1), R1, ctx.t1) @ crown_tilt(ctx.t1).T
        et = self.tE * (1 - kk1) ** 1.6 * smoothstep(0.0, 0.08, kk1) * 2.4 * pw * wrap_ok * (0.3 + 0.7 * storm_fade(t))
        tcol = look.blackbody(0.9 - 0.45 * kk1)
        if fw > 0:
            # v3: the flame's tongues: white at the root, ice along the tongue, gold only on the outer edge
            etf = self.tE * (1 - kf1) ** 1.3 * smoothstep(0.0, 0.08, kf1) * 2.4 * pw * wrap_f
            th_ = np.array(FLAME_TONGUES)[self.kt, 0]
            lat = np.abs(self.tu[:, 0] * (-np.sin(th_) * s_hat[0] + np.cos(th_) * s_hat[2])
                         + self.tu[:, 1] * 0.8 * (np.cos(th_) * s_hat[0] + np.sin(th_) * s_hat[2]))
            gq = np.clip(MIND_RIM_GOLD * smoothstep(0.58, 0.82, lat) + 0.25 * smoothstep(0.75, 1.0, kf1), 0, 1)[:, None]
            tm = m_core * (1 - smoothstep(0.0, 0.55, kf1))[:, None] + m_ice * smoothstep(0.0, 0.55, kf1)[:, None]
            tm = tm * (1 - gq) + m_edge * gq
            etf = etf * (1.0 + MIND_RIM_E * gq[:, 0])
            et = lerp(et, etf, fw)
            tcol = tcol * (1 - fw) + tm * fw
        et = et * (1.0 - 0.9 * forged(t)) * (1.0 - MIND_TONGUE_DIM * flame_w(t))
        tcol = tcol * (1 - 0.35 * red) + C_RED * 0.35 * red * np.ones_like(tcol)
        ctx.fr.splat(T0, T1, 0.004 + MIND_SOFT * 1.4 * fw, et, tcol, ctx.cam0, ctx.cam1)
        # --- filaments + pulses
        grow = smoothstep(0.0, 1.0, (t - IGN) / 26.0) * 1.3
        vis = self.fs < grow
        s = self.fs
        L = 1.4

        def pulses(tc):
            pu = np.zeros(len(s))
            for j in range(4):
                sp = (tc * self.pv[self.fm, j] + self.pp[self.fm, j]) % (L + 0.6)
                sp = np.where(self.pdir[self.fm, j] > 0, sp, L - sp)
                pu += np.exp(-((s - sp) / 0.035) ** 2)
            return pu
        pulse = pulses(t - IGN)
        base = 0.55 + 0.35 * (self.fdep == 0)
        if fw > 0:
            # v3 (red team): the thought stays visible in a calm body -- slow pulses, and on every breath one
            # wave of light climbs out from the core through the whole tree at once (too regular for a flame)
            ph_b = ((t - IGN) / BREATH_P) % 1.0                 # climbs from the core as the breath swells
            wave = np.exp(-((s - (ph_b * 2.4 - 0.15)) / 0.07) ** 2) * (1.0 - 0.5 * ph_b)
            pulse = lerp(pulse, pulses(FLAME_CLOCK * (t - IGN)) + 0.45 * wave, fw)
        ef = (base + 10.0 * pulse) * vis * 1.3 * pw * (1.0 - forged(t))
        if fw > 0:
            ef = ef * (1.0 - fw * smoothstep(0.22, -0.12, self.fp[:, 1]))   # no filament coil at the base
            ef = ef * (1.0 + (MIND_FIL - 1.0) * fw)
        fcol = C_ICE * (1 - np.minimum(pulse, 1))[:, None] + C_CORE * np.minimum(pulse, 1)[:, None]
        fq0 = flame_shape(self.fp, flame_w(ctx.t0), stretch_up=MIND_FIL_UP, reach=1.3, down=1.0, lift=0.22)
        fq1 = flame_shape(self.fp, flame_w(ctx.t1), stretch_up=MIND_FIL_UP, reach=1.3, down=1.0, lift=0.22)
        F0 = C0 + self.warp(fq0, crown_morph(ctx.t0), R0, ctx.t0) @ crown_tilt(ctx.t0).T
        F1 = C1 + self.warp(fq1, crown_morph(ctx.t1), R1, ctx.t1) @ crown_tilt(ctx.t1).T
        ctx.fr.splat(F0, F1, 0.002, ef, fcol, ctx.cam0, ctx.cam1)       # v3 (locked sheet): filaments below notice
        if fw > 0:
            M0 = C0 + self.motes(ctx.t0) * R0 @ crown_tilt(ctx.t0).T
            Mq = self.motes(ctx.t1)
            M1 = C1 + Mq * R1 @ crown_tilt(ctx.t1).T
            k = ((self.mo_ph + self.mo_v * FLAME_CLOCK * (ctx.t1 - IGN)) % 1.0)
            k0 = ((self.mo_ph + self.mo_v * FLAME_CLOCK * (ctx.t0 - IGN)) % 1.0)
            ok = k >= k0
            life = smoothstep(0.0, 0.12, k) * (1.0 - smoothstep(0.7, 1.0, k))
            tw = 0.8 + 0.2 * np.sin(0.2 * t + self.mo_tw)
            em = self.mo_E * life * tw * ok * 7.0 * pw * fw * smoothstep(IGN + 4, IGN + 30, t)
            cm = C_CORE * (1 - 0.6 * k)[:, None] + (C_GOLD * 0.8 + C_CORE * 0.2) * (0.6 * k)[:, None]
            ctx.fr.splat(M0[ok], M1[ok], 0.005, em[ok], cm[ok], ctx.cam0, ctx.cam1)
        # --- glow (volumetric halo around the fire)
        H = np.array([C1, C1, C1 + [0, 0.4, 0]])
        rr = np.array([1.2, 3.0, 7.0]) * (1 + 1.5 * m)
        eh = np.array([500.0, 500.0, 420.0]) * pw * smoothstep(IGN, IGN + 6, t)
        hc = np.array([C_CORE, C_ICE * 0.6 + C_GOLD * 0.4, C_GOLD * (1 - red) + C_RED * red])
        if fw > 0:
            R_ = fire_radius(t)
            up = crown_tilt(t)[:, 1]
            Hf = np.array([C1 + up * (0.6 * R_), C1 + up * (1.05 * R_), C1 + up * (1.6 * R_), C1 + up * (0.7 * R_)])
            rf = np.array([0.8, 1.5, 1.9, 6.0])
            ef = np.array([200.0, 300.0, 170.0, 360.0]) * pw * smoothstep(IGN, IGN + 6, t) * MIND_GLOW
            cf = np.array([C_CORE, C_ICE * 0.75 + C_GOLD * 0.25, C_ICE * 0.55 + C_GOLD * 0.45,
                           (C_ICE * 0.5 + C_GOLD * 0.5) * (1 - red) + C_RED * red])       # v3: mind palette
            H = np.vstack([H, Hf])
            rr = np.concatenate([rr, rf])
            eh = np.concatenate([eh * (1 - fw), ef * fw])
            hc = np.vstack([hc, cf])
        fg = forged(t)
        if fg > 0:
            eh = eh * (1 - 0.65 * fg) * (1 - 0.8 * eye_k(t))
            gc = np.array([C_GOLD, C_GOLD, C_GOLD * (1 - red) + C_RED * red] + [C_GOLD] * (len(hc) - 3))
            hc = hc * (1 - fg) + gc * fg
        ctx.fr.splat(H, H, rr, eh, hc, ctx.cam0, ctx.cam1, profile=1)


class FireSparks:
    """A column of sparks rising from the thinking fire (and from the ring later)."""

    def __init__(self, seed=34):
        r = rng(seed)
        n = 5000
        self.n = n
        self.ph = r.random(n)
        self.v = r.uniform(0.006, 0.014, n)
        self.a = r.uniform(0, 2 * np.pi, n)
        self.sp = r.normal(0, 1, (n, 2))
        self.E = r.lognormal(0, 0.7, n)
        self.flame_sub = rng(seed + 500).random(n) < (0.04 if SCHED is None else 0.012)   # v2: only a few lift off

    def pts(self, t):
        C = crown_centre(t)
        R = fire_radius(t)
        m = crown_morph(t)
        k = (self.ph + self.v * (t - IGN)) % 1.0
        h = 1.6 * R + k * 14.0
        spread = 0.25 + 2.2 * k
        Rr = 5.0 * m
        x = (Rr + self.sp[:, 0] * spread * 0.4) * np.cos(self.a) * m + self.sp[:, 0] * spread * (1 - m)
        z = (Rr + self.sp[:, 0] * spread * 0.4) * np.sin(self.a) * m + self.sp[:, 1] * spread * (1 - m)
        y = h * (1 - 0.8 * m) + m * (1.0 + k * 9.0)
        fw = flame_w(t)
        if fw > 0:
            # v2: from the three tongue tips, a thin drifting plume (v1's column read as the bulb's cord)
            kt = (np.arange(self.n) % len(FLAME_TONGUES))
            tip, _ = tongue_axis(kt, np.full(self.n, 0.97), t, fw)
            spr = 0.2 + 0.9 * k
            xf = tip[:, 0] * R + self.sp[:, 0] * spr
            zf = tip[:, 2] * R + self.sp[:, 1] * spr
            yf = tip[:, 1] * R + k * 2.2                    # a few sparks lift off the tips and die (the old column
                                                             # climbed out of the frame and read as the bulb's neck)
            x = lerp(x, xf, fw)
            y = lerp(y, yf, fw)
            z = lerp(z, zf, fw)
        P = C + np.stack([x, y, z], 1) @ crown_tilt(t).T
        w = vnoise(P * 0.3 + np.array([0, -0.03 * t, 0]), 0.6, (0, 0, 0), 1)
        return P + w * (0.3 + 1.2 * k)[:, None], k

    def emit(self, ctx):
        t = ctx.t
        if t < IGN + 6 or t >= (960 if SCHED is None else SCHED.end):
            return
        P0, k0 = self.pts(ctx.t0)
        P1, k = self.pts(ctx.t1)
        ok = k >= k0
        e = self.E * (1 - k) ** 1.5 * 9.0 * smoothstep(IGN + 6, IGN + 20, t) * ok * storm_fade(t)
        fw = flame_w(t)
        e = e * np.where(self.flame_sub, 1.0, 1.0 - fw) * (1.0 - fw * smoothstep(0.35, 0.8, k))
        red = redness(t)
        col = look.blackbody(0.92 - 0.45 * k)
        if fw > 0:          # v3: pale gold-white motes off the tongue tips while it is a flame, not orange sparks
            cm = C_CORE * 0.35 + C_GOLD * 0.65
            col = col * (1 - fw) + cm * fw
        col = col * (1 - 0.4 * red) + C_RED * 0.4 * red
        ctx.fr.splat(P0, P1, 0.004, e, col, ctx.cam0, ctx.cam1)


class Crown:
    """Flame tines of the crown (appear as the fire opens into the ring)."""

    def __init__(self, seed=33):
        r = rng(seed)
        self.nt = 9
        n = 36000
        self.n = n
        self.tine = r.integers(0, self.nt, n)
        self.u = r.random(n)
        self.v = r.normal(0, 1, (n, 2))
        self.ph = r.random(n)
        self.sp = r.uniform(0.02, 0.05, n)
        self.E = r.lognormal(0, 0.4, n)
        # v2 (critic note m7): not nine equal triangles (a tiara) but irregular licking tongues over a low fringe
        r2 = rng(seed + 900)
        nt = 13
        base = (np.arange(nt) + 0.5) / nt * 2 * np.pi
        self.ta = base + r2.uniform(-0.2, 0.2, nt)                      # uneven spacing
        self.tH = r2.uniform(0.9, 3.1, nt) ** 1.0                         # some tall, some low
        self.tH[r2.permutation(nt)[:3]] *= 1.25
        self.tW = r2.uniform(0.28, 0.62, nt)
        self.tL = r2.uniform(0.25, 0.7, nt) * np.where(r2.random(nt) < 0.5, -1.0, 1.0)
        self.tom = r2.uniform(0.22, 0.45, nt)                             # licking rhythm (rad/frame)
        self.tph = r2.uniform(0, 2 * np.pi, nt)
        wgt = self.tH * self.tW
        self.kt = r2.choice(nt, n, p=wgt / wgt.sum())
        self.fringe = r2.random(n) < 0.3
        self.fa = r2.uniform(0, 2 * np.pi, n)                             # fringe: anywhere round the ring
        self.fH = r2.uniform(0.25, 0.7, n)

    def pts(self, t):
        m = crown_morph(t)
        C = crown_centre(t)
        Rr = 5.0 + 0.6 * smoothstep(640, 800, t)
        rot = m * 0.012 * (t - 562)
        k = (self.ph + self.sp * t) % 1.0          # particles stream up each tongue
        j = self.kt
        om, ph = self.tom[j], self.tph[j]
        # each tongue licks: its height breathes on its own rhythm, it sways, a wave runs up it, the tip bends over
        H = self.tH[j] * (1.0 + 0.28 * np.sin(om * t + ph) + 0.13 * np.sin(2.3 * om * t + 1.7 * ph))
        H = np.where(self.fringe, self.fH * (1.0 + 0.35 * np.sin(0.4 * t + 9.0 * self.fa)), H)
        h = H * m
        y = k * h
        sway = self.tL[j] * k * k * np.sin(0.06 * t + ph) + 0.16 * k * np.sin(2 * np.pi * 1.3 * k - 1.5 * om * t + ph)
        a = np.where(self.fringe, self.fa, self.ta[j] + sway / Rr) + rot
        W = np.where(self.fringe, 0.35, self.tW[j])
        w = W * (1 - k) ** 0.65 * (1.0 + 0.25 * np.sin(1.2 * om * t + 4.0 * k + ph)) + 0.02
        lean_r = np.where(self.fringe, 0.0, 0.35 * k * k * np.sin(0.05 * t + 2.0 * ph))
        x = (Rr + self.v[:, 0] * w * 0.5 + lean_r)
        z = self.v[:, 1] * w
        P = np.stack([x * np.cos(a) - z * np.sin(a), y + 0.25, x * np.sin(a) + z * np.cos(a)], 1)
        wn = vnoise(P * 0.8 + np.array([0.0, -0.09 * t, 0.0]), 1.0, (3.0, 0.0, 1.0), 2)
        P = P + wn * (0.04 + 0.3 * k)[:, None] * m
        return C + P @ crown_tilt(t).T, k

    def emit(self, ctx):
        t = ctx.t
        m = crown_morph(t)
        if m <= 0.02 or t >= 960 or variant.tolkien():       # cut C: a plain band, no crown of flame
            return
        P0, _ = self.pts(ctx.t0)
        P1, k = self.pts(ctx.t1)
        red = redness(t)
        e = self.E * (1 - k) ** 1.2 * 3.4 * smoothstep(0.35, 0.9, m) * fire_power(t) * storm_fade(t)
        e = e * np.where(self.fringe, 1.2, 1.55)          # v2: the wider tongues need a little more light
        col = look.blackbody(0.95 - 0.35 * k)
        col = col * (1 - 0.4 * red) + C_RED * 0.4 * red
        ctx.fr.splat(P0, P1, 0.004, e, col, ctx.cam0, ctx.cam1)


class Shockwave:
    def __init__(self, seed=44):
        r = rng(seed)
        n = 26000
        self.n = n
        d = rand_dirs(r, n)
        d[:, 1] *= 0.55
        self.d = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.v = r.uniform(0.35, 1.5, n) * r.choice([1.0, 1.0, 0.45], n)
        self.tau = r.uniform(5.0, 11.0, n)
        self.E = r.lognormal(0, 0.6, n)

    def pts(self, t):
        a = max(t - IGN, 0.0)
        rr = self.v * self.tau * (1 - np.exp(-a / self.tau))
        return self.d * rr[:, None] + np.array([0, 0.012, 0]) * a * a

    def emit(self, ctx):
        t = ctx.t
        if t < IGN or t > IGN + 40:
            return
        if SCHED is not None:
            # v3 (H5: "a radial ember starfield" at the ignition): the point simply catches -- a soft flash
            a = t - IGN
            if a < 10:
                H = np.zeros((2, 3))
                ctx.fr.splat(H, H, np.array([1.6, 5.0]), np.array([9.0e4, 4.0e4]) * math.exp(-a / 3.0),
                             np.array([C_CORE, C_ICE]), ctx.cam0, ctx.cam1, profile=1)
            return
        P0 = self.pts(ctx.t0)
        P1 = self.pts(ctx.t1)
        a = t - IGN
        e = self.E * 16.0 * math.exp(-a / 9.0)
        col = look.blackbody(np.clip(0.98 - a / 40.0, 0.4, 1.0))
        col = np.broadcast_to(col, (self.n, 3)) * 1.0
        ctx.fr.splat(P0, P1, 0.003, e, col, ctx.cam0, ctx.cam1)
        if a < 8:
            H = np.zeros((2, 3))
            ctx.fr.splat(H, H, np.array([2.0, 7.0]), np.array([3.0e5, 2.0e5]) * math.exp(-a / 2.2),
                         np.array([C_CORE, C_ICE]), ctx.cam0, ctx.cam1, profile=1)


# ------------------------------------------------------------- towers ---

class TowersV1:
    """v1 towers (points along edges; kept for reference only)."""
    def __init__(self, seed=55):
        r = rng(seed)
        self.T = TW.build_all()
        k = len(self.T)
        self.k = k
        self.ang = 2 * np.pi * np.arange(k) / k + TOWER_ANG0
        self.rad = 18.0 + r.uniform(-1.2, 1.2, k)
        self.t_rise = 520 + np.array([0, 9, 4, 14, 6, 11, 2, 8], float)
        # near the camera (k=0,1) tall and looming; the far side lower so the crown floats clear
        self.h_rise = np.array([40.0, 38.0, 35.0, 27.0, 28.5, 26.0, 27.5, 36.0])
        # surge amounts per beat (leap-frogging race)
        J = r.uniform(1.6, 2.8, (k, len(BEATS)))
        lead = r.permutation(np.tile(np.arange(k), 2))[:len(BEATS)]
        for b, l in enumerate(lead):
            J[l, b] += r.uniform(3.0, 4.5)
        # the far side (shorter at 640) races harder to catch up
        J[3:7] *= 1.25
        self.J = J
        self.dly = r.uniform(0, 2.5, k)
        # per-point attributes
        self.rnd = [r.random(len(t['p'])) for t in self.T]
        self.win_on = [r.random(len(t['p'])) < 0.8 for t in self.T]
        self.flk = [r.uniform(0, 2 * np.pi, len(t['p'])) for t in self.T]
        # facing: rotate each tower so its local +x faces the centre
        self.rot = [(-a + np.pi) for a in self.ang]

    def height(self, i, t):
        h = self.h_rise[i] * float(ease_out((t - self.t_rise[i]) / 80.0, 2.6))
        for b, tb in enumerate(BEATS):
            x = (t - tb - self.dly[i]) / 5.0
            if x > 0:
                h += self.J[i, b] * float(ease_out_back(x, 1.6))
        return h

    def base(self, i):
        a = self.ang[i]
        return np.array([self.rad[i] * math.cos(a), GROUND, self.rad[i] * math.sin(a)])

    def top(self, i, t):
        return self.base(i) + np.array([0, self.height(i, t), 0])

    def world(self, i, p, h):
        c, s = math.cos(self.rot[i]), math.sin(self.rot[i])
        x = c * p[:, 0] - s * p[:, 2]
        z = s * p[:, 0] + c * p[:, 2]
        b = self.base(i)
        return np.stack([x + b[0], p[:, 1] - TW.HMAX + h + b[1], z + b[2]], 1)

    def emit(self, ctx, light_pos, light_col, light_pow):
        t = ctx.t
        if t < 515 or t >= 1040:
            return
        red = redness(t)
        for i in range(self.k):
            T = self.T[i]
            h0, h1 = self.height(i, ctx.t0), self.height(i, ctx.t1)
            if h1 <= 0.05:
                continue
            vis = T['p'][:, 1] > TW.HMAX - h1 - 0.3
            p = T['p'][vis]
            P0 = self.world(i, p, h0)
            P1 = self.world(i, p, h1)
            kind = T['kind'][vis]
            c, s = math.cos(self.rot[i]), math.sin(self.rot[i])
            nl = T['n'][vis]
            nw = np.stack([c * nl[:, 0] - s * nl[:, 2], nl[:, 1], s * nl[:, 0] + c * nl[:, 2]], 1)
            yl = P1[:, 1] - GROUND          # height above ground
            rnd = self.rnd[i][vis]
            # base ember glow: brighter edges, faint surfaces, dim toward the ground
            e_base = np.where(kind == 1, 1.0, 0.55) * (0.5 + 0.5 * rnd)
            e_base *= 0.35 + 0.65 * smoothstep(0.0, 14.0, yl)
            fk = self.flk[i][vis]
            fl = 1 + 0.45 * np.sin(1.3 * t + fk) * np.sin(0.37 * t + 2.0 * fk)
            Tb = 0.27 + 0.13 * rnd
            cb = look.blackbody(Tb)
            cb = cb * (1 - 0.85 * red) + (C_RED * 0.6 + C_CRIMSON * 0.4) * 0.85 * red
            bp = beat_pulse(t - self.dly[i])
            e_base = e_base * fl * 2.3 * (1 + 1.6 * bp) * (1 + 0.5 * smoothstep(640, 880, t))
            # fire light (inner faces)
            L = light_pos - P1
            dL = np.linalg.norm(L, axis=1)
            lam = np.maximum((nw * L).sum(1) / np.maximum(dL, 1e-6), 0.0)
            lam = np.where(kind == 1, 0.1 + 0.9 * lam, lam) ** 0.8
            e_lit = light_pow * lam / (1 + (dL / 16.0) ** 2) * (1.0 - 0.35 * red)
            # emergence front: hot line where the tower leaves the ground
            front = np.exp(-yl / 0.6) * 5.0 * (1 - smoothstep(610, 650, t) * 0.7)
            # windows
            win = (kind == 2) & self.win_on[i][vis]
            e_win = np.where(win, 26.0 * (0.7 + 0.3 * np.sin(0.2 * t + 7 * rnd)), 0.0)
            wcol = look.blackbody(0.72 - 0.15 * red)
            colE = (cb * e_base[:, None] + light_col[None, :] * e_lit[:, None] +
                    look.blackbody(0.8)[None, :] * front[:, None] + wcol[None, :] * e_win[:, None])
            E = np.ones(len(p))
            E[kind == 2] = np.where(win[kind == 2], 1.0, 0.0) + 1e-3
            ctx.fr.splat(P0, P1, 0.012, E, colE, ctx.cam0, ctx.cam1)


C_MOLTEN = np.array([1.0, 0.74, 0.28])
C_GOLD = np.array([1.0, 0.8, 0.42])
C_GOLD_HOT = np.array([1.0, 0.93, 0.72])


def _gold_runs(pl, t, g, fside, face):
    """v3 A (THE EDGE): on every surge liquid gold pours from the crown and runs down the fire-facing face in
    rivulets, each with a bright bead at its head, staggered like drips; behind them the face stays gilt (a gleaming
    skin that thickens as the race goes on). Radiance, added to the crust's own."""
    import bisect
    dtop = 66.0 - pl[:, 1]
    beats = getattr(SCHED, 'beats', None) or []
    j = bisect.bisect_right(beats, t) - 1
    age = (t - beats[j]) if j >= 0 else 99.0
    n = vnoise(np.stack([pl[:, 2] * 1.25, pl[:, 0] * 0.55, np.zeros(len(pl), pl.dtype)], 1), 1.0, (7.1, 0.0, 3.3), 2)[:, 0]
    riv = smoothstep(0.12, 0.42, n)
    v = 1.0 + 1.4 * smoothstep(-0.4, 0.4, vnoise(np.stack([pl[:, 2] * 0.45, pl[:, 0] * 0.3, np.zeros(len(pl), pl.dtype)], 1),
                                                   1.0, (2.3, 5.0, 1.7), 1)[:, 0])
    front = v * age
    behind = smoothstep(0.6, -0.6, dtop - front)
    fresh = math.exp(-age / 16.0)
    head = np.exp(-((dtop - front) / 1.1) ** 2) * fresh
    skin = 0.22 + 0.3 * smoothstep(-0.2, 0.3, n)
    run = riv * (0.35 + 1.1 * behind * (0.45 + 0.55 * fresh))
    fs = np.clip(fside, 0, 2.5) if np.ndim(fside) else fside
    k = g * fs * face
    return C_GOLD[None, :] * (k * (skin + run))[:, None] + C_GOLD_HOT[None, :] * (k * 2.6 * riv * head)[:, None]


def _line_mod(towers, i, kind, pt, t):
    """v3 A (EMBERS-A3, A16/A17): a schedule may break the towers' lines of fire (edges, seams, joints) into irregular
    lengths so they read as ember-lit stone, not strings of lights; 1 when it doesn't"""
    if SCHED is not None and hasattr(SCHED, 'line_mod'):
        return SCHED.line_mod(towers, i, kind, pt, t)
    return 1.0


def _tw_colours(T):
    return look.blackbody(np.clip(T, 0.0, 1.0))


FIRE_SIDE_BOOST = 0.9      # v3 (A): the fire-facing faces burn this much brighter than v2's crust ...
FIRE_SIDE_LIT = 2.6        # ... and catch this much more of the fire's light
def _tower_cache(TW2, giants, alt):
    """the tower geometry is deterministic: cache it on disk, keyed by towers2.py's source and the switches"""
    import hashlib
    import os
    import pickle
    src = open(TW2.__file__, 'rb').read()
    key = hashlib.sha1(src + repr((tuple(giants), bool(alt))).encode()).hexdigest()[:16]
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'renders', 'embers_A3', 'cache')
    f = os.path.join(d, 'towers_%s.pkl' % key)
    if os.path.exists(f):
        with open(f, 'rb') as fh:
            return pickle.load(fh)
    T, S = TW2.build_all(giants, alt=alt), TW2.build_skyline()
    os.makedirs(d, exist_ok=True)
    tmp = f + '.%d' % os.getpid()
    with open(tmp, 'wb') as fh:
        pickle.dump((T, S), fh, protocol=4)
    os.replace(tmp, f)
    return T, S


GIANT_SURGE = 2.0          # v3: extra height per beat for the two giants (from 660) ...
GIANT_LEAD = 1.8           # ... and a lead on alternate beats (they leap-frog each other)

# LD_SURGE_EASE=1 (A7 THE EDGE re-render; the owner: A's sequence is "jumpy and unpolished"): each beat's jump J in
# Towers.height gets a soft attack. The accepted curve, ease_out_back((t - beat - delay) / 5, 1.6), puts 69% of J
# into the first frame after the onset and sweeps 40% of J inside one half-frame shutter. The eased one starts at zero
# slope, reaches 21% of J one frame in, peaks at 1.05 J 3.5 frames after the onset (was 1.09 J at 2.95, so the
# highest frame is the same one or the next at every onset phase) and holds exactly J from 7 frames (was 5); its
# largest one-frame step is 44% of J, its largest half-frame sweep 22%. Read once, here; off (the default) takes the
# accepted expression unchanged, so a render made without the flag is the render made before it existed.
SURGE_EASE = os.environ.get('LD_SURGE_EASE', '0') == '1'
SURGE_PEAK, SURGE_OVER, SURGE_SETTLE = 3.5, 0.05, 7.0     # frames after the onset; the overshoot as a share of J


def surge_ease(d):
    """share of a surge's jump reached d frames after its onset (LD_SURGE_EASE): a smoothstep rise to 1 + SURGE_OVER
    at SURGE_PEAK, a smoothstep settle to exactly 1.0 at SURGE_SETTLE; zero slope at the onset, C1 throughout"""
    if d <= 0.0:
        return 0.0
    if d >= SURGE_SETTLE:
        return 1.0
    if d < SURGE_PEAK:
        u = d / SURGE_PEAK
        return (1.0 + SURGE_OVER) * u * u * (3.0 - 2.0 * u)
    u = (d - SURGE_PEAK) / (SURGE_SETTLE - SURGE_PEAK)
    return 1.0 + SURGE_OVER * (1.0 - u * u * (3.0 - 2.0 * u))


# LD_TOWER_PULSE_EASE=1 (A7 re-render; owner, 30 Sep): LD_SURGE_EASE's eased height measured no better on A7's plates
# (per-beat peak one-frame MAD 12.38 -> 11.97): the one-frame lurch is each tower's LIGHT. On its onset frame a tower's
# brightness steps to (1 + 0.9 beat_pulse) x (1 + 2.5 heat band), up to 6.65x, both at full strength at once. With the
# flag both rise over PULSE_ATTACK frames (smoothstep, half-way one frame in) and then decay exactly as before, so the
# flash keeps its brightness and lands PULSE_ATTACK frames after its onset. Read once, here; off (the default) takes the
# accepted expressions unchanged.
TOWER_PULSE_EASE = os.environ.get('LD_TOWER_PULSE_EASE', '0') == '1'
PULSE_ATTACK = 2.0


def pulse_attack(x):
    """share of a tower's beat flash reached x frames after its onset (LD_TOWER_PULSE_EASE)"""
    if x <= 0.0:
        return 0.0
    if x >= PULSE_ATTACK:
        return 1.0
    u = x / PULSE_ATTACK
    return u * u * (3.0 - 2.0 * u)


def tower_pulse(t):
    """beat_pulse for a tower's light under LD_TOWER_PULSE_EASE: the same 1 -> exp(-x/3) decay, reached over
    PULSE_ATTACK frames and decaying from there (BEATS is the schedule's beats once a3.install() has run)"""
    for tb in BEATS:
        x = t - tb
        if 0.0 <= x < 20.0 + PULSE_ATTACK:
            return pulse_attack(x) * math.exp(-max(x - PULSE_ATTACK, 0.0) / 3.0)
    return 0.0


class Towers:
    """v2: eight towers MADE OF EMBERS (geometry: towers2.py). Solid masses: their crust feeds an occluder that
    hides whatever stands behind them. Ember crust with slow heat patches, seams of fire, burning edges and
    rims, windows of fire (lit cells flicker; unlit ones are dark openings), heat at the base, a heat wave that
    climbs each tower on every surge, the crown-fire lighting the inner faces, the palette bleeding to crimson,
    tops thinning into smoke. Motion (rise, surges, leap-frogging) is exactly v1's."""

    ZF = 15.0             # full point density closer than this (world units); LOD beyond
    ALB = np.array([1.0, 0.62, 0.4], np.float32)       # warm albedo of the crust for the fire's light

    def __init__(self, seed=55):
        r = rng(seed)
        k = 8
        self.k = k
        self.ang = 2 * np.pi * np.arange(k) / k + TOWER_ANG0
        self.rad = 18.0 + r.uniform(-1.2, 1.2, k)
        self.t_rise = (520 if SCHED is None else SCHED.tower_rise) + np.array([0, 9, 4, 14, 6, 11, 2, 8], float)
        self.h_rise = np.array([40.0, 38.0, 35.0, 27.0, 28.5, 26.0, 27.5, 36.0])
        J = r.uniform(1.6, 2.8, (k, len(BEATS)))
        lead = r.permutation(np.tile(np.arange(k), 2))[:len(BEATS)]
        for b, l in enumerate(lead):
            J[l, b] += r.uniform(3.0, 4.5)
        J[3:7] *= 1.25
        self.J = J
        self.dly = r.uniform(0, 2.5, k)
        import towers2 as TW2
        self.TW2 = TW2
        # v3 (director, "TWO GIANTS, UNCODED"; A only): two forges on opposite sides of the fire outgrow every
        # other tower as the race escalates, leap-frogging each other on alternate beats
        self.giants = variant.giants()
        T, S = _tower_cache(TW2, self.giants, variant.TOWERS_ALT)
        if SCHED is None:
            for g_ in self.giants:
                for b in range(1, len(BEATS)):
                    J[g_, b] += GIANT_SURGE + (GIANT_LEAD if (b + (g_ == self.giants[1])) % 2 else 0.0)
        # v3 (BIBLE_V3 rev. 1, B6 "many, not two"): the eight nearest the fire are forge-stacks; behind them a far
        # skyline of many traditions (towers2.SKYLINE), spread round the ring except the camera's own sector and
        # the gap behind the fire, rising a little after the forges and surging less. Walls, sparks, embers and
        # smoke stay with the eight (self.k); drawing and occlusion cover all (self.k_all).
        if variant.CUT == 'A3':
            S = []          # v3 A: the eight forges alone, so the ring and the two giants read (C keeps the skyline)
        ns = len(S)
        rs = rng(seed + 77)
        free = []                        # azimuths clear of the camera's sector and of the gap behind the fire
        for a in np.linspace(0, 2 * np.pi, 720, endpoint=False):
            dc_ = abs((a - ALPHA_C + np.pi) % (2 * np.pi) - np.pi)
            db_ = abs((a - ALPHA_C) % (2 * np.pi) - np.pi)
            if dc_ > 1.0 and db_ > 0.16:
                free.append(a)
        free = np.array(free)
        pick = free[np.linspace(0, len(free) - 1, max(ns, 1)).round().astype(int)][:ns] + rs.uniform(-0.05, 0.05, ns)
        self.k_all = k + ns
        self.ang = np.concatenate([self.ang, pick])
        self.rad = np.concatenate([self.rad, rs.uniform(33.0, 46.0, ns)])
        self.t_rise = np.concatenate([self.t_rise, self.t_rise[0] + 4 + rs.uniform(0, 26, ns)])
        self.h_rise = np.concatenate([self.h_rise, rs.uniform(24.0, 44.0, ns)])
        self.J = np.concatenate([self.J, rs.uniform(1.0, 2.2, (ns, len(BEATS)))])
        if SCHED is not None and hasattr(SCHED, 'surge_scale'):
            self.J = self.J * SCHED.surge_scale
        self.dly = np.concatenate([self.dly, rs.uniform(0, 4.0, ns)])
        self.rot = [(-a + np.pi) for a in self.ang]
        T = T + S
        rr = rng(seed + 1000)
        self.G = []
        for i, t in enumerate(T):
            g = {}
            for kind in range(5):
                m = t['kind'] == kind
                d = dict(p=t['p'][m].astype(np.float32), n=t['n'][m].astype(np.float32),
                         a=t['a'][m].astype(np.float32), key=t['key'][m].astype(np.int32),
                         hot=t['hot'][m].astype(np.float32))
                nn = int(m.sum())
                d['rnd'] = rr.random(nn).astype(np.float32)
                pl = t['p'][m]
                d['nz'] = (0.5 + 0.5 * np.clip(snoise(pl, 0.23, (3.7 * i, 1.3, 2.9), 2) * 1.6, -1, 1)).astype(np.float32)
                d['nz2'] = (0.5 + 0.5 * np.clip(snoise(pl, 1.1, (9.1, 4.4 * i, 0.3), 2) * 1.6, -1, 1)).astype(np.float32)
                # heat streaks: patches stretched up the facade (fire climbs)
                d['nzv'] = (0.5 + 0.5 * np.clip(snoise(pl * np.array([1.0, 0.22, 1.0]), 0.45, (1.7 * i, 0.0, 5.3), 2)
                                                * 1.7, -1, 1)).astype(np.float32)
                g[kind] = d
            nw = max(int(t['nwin']), 1)
            lv = rr.random(nw) ** 1.8                   # most openings smoulder, a few roar
            g['wlv'] = (0.12 + 0.88 * lv).astype(np.float32)
            g['wT'] = (0.36 + 0.3 * lv + rr.uniform(-0.04, 0.04, nw)).astype(np.float32)
            g['wph'] = rr.uniform(0, 2 * np.pi, nw).astype(np.float32)
            g['wfr'] = rr.uniform(0.04, 0.22, nw).astype(np.float32)
            self.G.append(g)
        self.cur = None
        self.lean = 0.0
        print('towers v3: forges', ' '.join(str(sum(len(g[kk]['p']) for kk in range(5))) for g in self.G[:k]),
              '| skyline', sum(sum(len(g[kk]['p']) for kk in range(5)) for g in self.G[k:]))

    # ------------------------------------------------------------- motion (v1)
    def height(self, i, t):
        h = self.h_rise[i] * float(ease_out((t - self.t_rise[i]) / 80.0, 2.6))
        for b, tb in enumerate(BEATS):
            x = (t - tb - self.dly[i]) / 5.0
            if x > 0:
                h += self.J[i, b] * (surge_ease(t - tb - self.dly[i]) if SURGE_EASE else float(ease_out_back(x, 1.6)))
        if SCHED is not None:
            h += SCHED.tower_extra(self, i, t)
        return h

    def base(self, i):
        a = self.ang[i]
        return np.array([self.rad[i] * math.cos(a), GROUND, self.rad[i] * math.sin(a)])

    def top(self, i, t):
        return self.base(i) + np.array([0, self.height(i, t), 0])

    def world(self, i, p, h):
        c, s = math.cos(self.rot[i]), math.sin(self.rot[i])
        b = self.base(i)
        lean = self.lean[i] if np.ndim(self.lean) else self.lean
        if lean != 0.0:
            # v3 (C): every tower leans toward the Eye; (A) toward the fire over the crater (a shear toward +x)
            p = p.copy()
            p[:, 0] = p[:, 0] + (p[:, 1] - np.float32(self.TW2.HMAX - h)) * np.float32(lean)
        x = c * p[:, 0] - s * p[:, 2] + b[0]
        z = s * p[:, 0] + c * p[:, 2] + b[2]
        y = p[:, 1] + np.float32(- self.TW2.HMAX + h + b[1])
        return np.stack([x, y, z], 1)

    def nworld(self, i, n):
        c, s = math.cos(self.rot[i]), math.sin(self.rot[i])
        return np.stack([c * n[:, 0] - s * n[:, 2], n[:, 1], s * n[:, 0] + c * n[:, 2]], 1)

    def heat_wave(self, i, t, yl):
        """a band of heat that climbs the tower after each surge"""
        w = np.zeros_like(yl)
        for tb in BEATS:
            x = t - tb - self.dly[i]
            if 0.0 <= x < 26.0:
                yw = -2.0 + 7.5 * x
                w += (pulse_attack(x) if TOWER_PULSE_EASE else 1.0) * math.exp(-x / 11.0) * np.exp(-((yl - yw) / 3.2) ** 2)
        return w

    # ------------------------------------------------------------- per frame
    def prepare(self, ctx):
        """World positions of the visible, level-of-detail points of every tower; builds the occluder."""
        t = ctx.t
        self.cur = None
        self.lean = 0.09 * float(smootherstep(826, 862, t)) if (variant.tolkien() and t < 880) else 0.0
        if SCHED is not None:
            self.lean = SCHED.tower_lean(t)
            if t < SCHED.tower_rise - 5 or t >= SCHED.end:
                return
        elif t < 515 or t >= 1040:
            return
        cam = ctx.cam
        cur = []
        oP, oN, oA, oI = [], [], [], []
        for i in range(self.k_all):
            h0, h1, h = self.height(i, ctx.t0), self.height(i, ctx.t1), self.height(i, t)
            if h1 <= 0.05:
                cur.append(None)
                continue
            b = self.base(i)
            if kingdom_rot(t) is not None:
                b = kingdom_rot(t) @ b
            # distance from the lens to the tower (axis segment base..top), less its half-width
            q0 = b + np.array([0, 0.0, 0])
            q1 = b + np.array([0, h, 0])
            d = q1 - q0
            u = float(np.clip(np.dot(cam.pos - q0, d) / max(np.dot(d, d), 1e-9), 0, 1))
            zmin = max(float(np.linalg.norm(cam.pos - (q0 + d * u))) - 4.0, 1.0)
            qs = float(np.clip((self.ZF / zmin) ** 2, 0.06, 1.0))
            ql = float(np.clip(self.ZF / zmin, 0.2, 1.0))
            ylim = self.TW2.HMAX - h1 - 0.3
            parts = {}
            for kind in range(5):
                g = self.G[i][kind]
                q = qs if kind in (0, 3, 4) else ql
                n = int(len(g['p']) * q)
                p = g['p'][:n]
                m = p[:, 1] > ylim
                idx = np.nonzero(m)[0]
                pl = p[idx]
                parts[kind] = dict(idx=idx, q=q, P0=_krot(self.world(i, pl, h0), ctx.t0),
                                   P1=_krot(self.world(i, pl, h1), ctx.t1), P=_krot(self.world(i, pl, h), t),
                                   N=_krot(self.nworld(i, g['n'][idx]), t), pl=pl)
                if SCHED is not None and hasattr(SCHED, 'tower_post'):
                    pp = parts[kind]
                    if hasattr(SCHED, 'tower_post_n'):
                        pp['N'] = SCHED.tower_post_n(self, i, pp['N'], pp['P'], t)     # (the falling crown turns)
                    pp['P0'] = SCHED.tower_post(self, i, pp['P0'], ctx.t0)
                    pp['P1'] = SCHED.tower_post(self, i, pp['P1'], ctx.t1)
                    pp['P'] = SCHED.tower_post(self, i, pp['P'], t)
            cr = parts[0]
            if len(cr['idx']):
                a = self.G[i][0]['a'][cr['idx']] / cr['q']
                oP.append(cr['P'])
                oN.append(cr['N'])
                oA.append(a)
                oI.append(np.full(len(a), i, np.int32))
            cur.append(dict(h=h, parts=parts))
        self.cur = cur
        if SCHED is not None and hasattr(SCHED, 'extra_occluders'):
            ex = SCHED.extra_occluders(self, ctx)          # v3 A: THE EDGE's crater wall and plain are solid too
            if ex is not None:
                oP.append(ex[0])
                oN.append(ex[1])
                oA.append(ex[2])
                oI.append(ex[3])
        if oP:
            A = np.concatenate(oA)
            from core import build_occluder
            build_occluder(ctx.fr, cam, np.concatenate(oP), np.concatenate(oN), A,
                           np.sqrt(A / np.pi) * 2.2, np.concatenate(oI), near=ctx.fr.prm[5])

    def emit(self, ctx, light_pos, light_col, light_pow):
        t = ctx.t
        if self.cur is None:
            return
        cam = ctx.cam
        cpos = cam.pos.astype(np.float32)
        fwd = cam.R[2].astype(np.float32)
        lpos = np.asarray(light_pos, np.float32)
        fpx = cam.f_px(1920)
        red = redness(t)
        C_CR = (C_CRIMSON * 0.75 + C_RED * 0.25)
        lcol = (light_col * self.ALB).astype(np.float32)
        rise = 1 - 0.7 * float(smoothstep(610, 650, t))
        grow = 1 + 0.8 * float(smoothstep(640, 820, t))
        FS = variant.fire_side_only()
        if SCHED is not None:
            rise, grow = SCHED.tower_rise_glow(t), SCHED.tower_grow(t)
        elif t >= 960:
            grow *= 0.6            # the grasp: the towers stand back in the storm's dark; the hand leads
        for i in range(self.k_all):
            c = self.cur[i]
            if c is None:
                continue
            G = self.G[i]
            bp = tower_pulse(t - self.dly[i]) if TOWER_PULSE_EASE else beat_pulse(t - self.dly[i])
            surge = (1 + 0.9 * bp) * grow
            for kind in (0, 4, 1, 2, 3):
                pt = c['parts'][kind]
                idx = pt['idx']
                if len(idx) == 0:
                    continue
                g = G[kind]
                P, N = pt['P'], pt['N']
                V = cpos[None, :] - P
                dist = np.linalg.norm(V, axis=1)
                ndv = (N * V).sum(1) / np.maximum(dist, 1e-6)
                z = np.maximum((P - cpos[None, :]) @ fwd, 0.3)
                yl = P[:, 1] - GROUND
                dtop = self.TW2.HMAX - pt['pl'][:, 1]
                a = g['a'][idx] / pt['q']
                rnd = g['rnd'][idx]
                nz = g['nz'][idx]
                nz2 = g['nz2'][idx]
                hot = g['hot'][idx]
                ph = rnd * 6.2832
                fl = 1 + 0.33 * np.sin(1.3 * t + ph) * np.sin(0.37 * t + 2.0 * ph)
                ylc = np.maximum(yl, 0)
                base = 0.55 * np.exp(-ylc / 6.0) + 0.45 * np.exp(-ylc / 26.0)   # hot below, darkening upward
                front = np.exp(-ylc / 0.5) * rise
                frac = np.clip(ylc / max(c['h'], 1.0), 0, 1)
                vgr = 0.28 + 0.72 * (1.0 - frac) ** 1.25                        # the tops recede into the dark
                nzv = g['nzv'][idx]
                far = np.clip(24.0 / z, 0.3, 1.0)                               # fine joints simplify with distance
                wave = self.heat_wave(i, t, yl)
                ftop = 0.35 + 0.65 * smoothstep(0.0, 7.0, dtop)
                crown = 0.0
                if FS:
                    # v3 A (H5 "ember life on the fire side of every tower"; the crowns must read): the fire-facing
                    # side stays lit to the top, the crenellated crown is crisp, and each stack's own throat glows
                    # up through its parapet (the forge's fire, seen from every side: not the thinking fire's light)
                    ftop = 0.85 + 0.15 * smoothstep(0.0, 7.0, dtop)
                    vgr = 0.55 + 0.45 * (1.0 - frac)
                    crown = np.exp(-np.maximum(dtop - 0.25, 0.0) / 1.1) * (0.75 + 0.25 * np.sin(2.1 * t + 7.0 * ph))
                keep = ftop * smoothstep(-0.3, 0.4, yl)              # thin at the very top; nothing underground
                heat = (1 + 2.5 * wave) * surge
                gild, dark = 0.0, 0.0
                if SCHED is not None and hasattr(SCHED, 'gild'):
                    gild, dark = SCHED.gild(i, t)
                    heat = heat * (1.0 - 0.9 * dark)
                if FS:
                    # v3 (A): lit only on the face turned to the fire, black toward the others; the fire side is
                    # alive (director: brighter joints and vents, heat shimmer, crust that catches the light)
                    Lf = lpos[None, :] - P
                    fside = smoothstep(-0.3, 0.35, (N * Lf).sum(1) / np.maximum(np.linalg.norm(Lf, axis=1), 1e-6))
                    shim = 1.0 + 0.24 * fside * np.sin(0.9 * yl - 0.33 * t + 6.0 * nz + 3.0 * rnd)
                    fside = fside * (1.0 + FIRE_SIDE_BOOST * fside) * shim
                    # the fire's light dies out above the ring (director: the giants' faces toward each other are
                    # black, shutters shut: each only a shape lost in the other's glare); soot blackens every top
                    yw = P[:, 1]
                    if i in self.giants:
                        fside = fside * (1.0 - smoothstep(6.0, 24.0, yw))
                    else:
                        fside = fside * (1.0 - 0.8 * smoothstep(16.0, 34.0, yw))
                    soot = smoothstep(0.62, 0.97, frac) if kind in (0, 4, 1) else 0.0
                    fside = fside * (1.0 - 0.6 * soot)
                else:
                    fside = 1.0
                if kind == 0:                                        # the crust: a glowing coal, ash patches, rims
                    key = g['key'][idx]
                    opening = np.where(key >= 0, 0.12, 1.0)
                    mz = SCHED.masonry(i, pt['pl'], t) if (SCHED is not None and hasattr(SCHED, 'masonry')) else None
                    if mz is not None:                                # (C3: ashlar stones, no window-cell grid)
                        opening = 1.0 - mz[1] * (1.0 - opening)
                    ash = smoothstep(0.2, 0.85, 0.35 * nz + 0.65 * nzv)  # glowing coal vs cooler ash, streaked upward
                    grain = 0.85 + 0.3 * rnd
                    L = 0.09 * (0.3 + 1.4 * ash) * (0.8 + 0.4 * nz2) * grain * fl * opening * (1 + 3.0 * base) * heat
                    if FS:
                        # director: CHARCOAL. A dark crust with fine live ember points; the seams, joints and edges
                        # carry the fire (no blotches at any scale)
                        # (the crust points are big surface discs: any per-point glow reads as spots, so the crust
                        # stays an even charcoal with a fine grain; the fire lives in the joints, seams and edges)
                        L = 0.03 * (0.8 + 0.4 * rnd) * fl * opening * (1 + 2.0 * base) * heat
                    rim = np.clip(1.0 - ndv / 0.35, 0, 1) ** 2 * (ndv > 0)
                    rimL = 0.35 * rim * (0.6 + 0.6 * nz2) * fl * heat
                    L = ((L + rimL) * vgr + 2.2 * front) * fside + (0.8 if FS else 0.25) * rimL * vgr * (1 - fside)   # a glare fringe
                    L = L + 0.35 * crown
                    T = 0.27 + 0.1 * ash + 0.18 * base + 0.2 * front
                    col = _tw_colours(T)
                    col = col * (1 - 0.8 * red) + C_CR * 0.8 * red
                    Lv = lpos[None, :] - P
                    dL = np.linalg.norm(Lv, axis=1)
                    lam = np.clip((N * Lv).sum(1) / np.maximum(dL, 1e-6), 0, 1)
                    lit = 0.006 * light_pow * lam ** 1.3 / (1 + (dL / 16.0) ** 2) * (1 - 0.4 * red) * opening
                    if FS:
                        lit = lit * FIRE_SIDE_LIT * 0.5 * (0.75 + 0.5 * rnd)    # charcoal: a dark sheen, fine grain
                    lit = lit * np.clip(hot / 0.3, 0, 1) ** 2          # roofs (hot 0.2) stay dark tile
                    if gild > 0 or dark > 0:
                        # (A, THE EDGE) the gilded wear gold, not the fire's light; the farthest go dark
                        lit = lit * (1.0 - 0.6 * min(gild, 1.0)) * (1.0 - 0.9 * dark)
                    if SCHED is not None and hasattr(SCHED, 'edge_joint_k'):
                        lit = lit * SCHED.edge_joint_k(i, t)   # (A-FIX: the fire-lit crust sheen read as gold speckle)
                    face = smoothstep(-0.02, 0.1, ndv) * keep
                    colE = (col * L[:, None] + lcol[None, :] * lit[:, None]) * face[:, None]
                    if SCHED is not None and hasattr(SCHED, 'back_light'):
                        bl = SCHED.back_light(t)
                        if bl > 0:
                            out = np.stack([P[:, 0], np.zeros(len(P)), P[:, 2]], 1)
                            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)
                            lb = np.clip((N * out).sum(1), 0, 1) ** 1.5 * (0.6 + 0.8 * nz2)
                            colE = colE + np.array([1.0, 0.62, 0.33])[None, :] * (bl * lb * face)[:, None]
                    if gild > 0 and not getattr(SCHED, 'gold_overlay', False):
                        colE = colE + _gold_runs(pt['pl'], t, gild, fside, face)
                    if mz is not None:
                        colE = colE * mz[0][:, None]
                    self._splat(ctx, i, pt, colE, a, np.clip(ndv, 0.05, 1.0), z, fpx, np.sqrt(a / np.pi) * 1.7)
                    continue
                if kind == 4:                                        # fire in the joints (masonry / mullions / fissures)
                    # soft crevices of fire: a hot core line fading into the stone; slow patches of heat
                    patch = 0.2 + 1.3 * smoothstep(0.25, 0.9, 0.4 * nz + 0.6 * nzv) ** 1.5
                    if FS:
                        patch = 0.55 + 0.25 * patch      # fine lines of fire, not patches (director: charcoal)
                        hot = 0.55 + 0.45 * hot
                    L = (0.6 * far * (0.2 + 0.8 * hot ** 2) * patch * (0.8 + 0.4 * nz2) * fl * (1 + 3.0 * base) * heat
                         * vgr + 2.0 * front) * fside + 0.9 * crown
                    if SCHED is not None and hasattr(SCHED, 'joint_k'):
                        L = L * SCHED.joint_k(i)         # (C3: the facade's regular joint grid reads as offices)
                    if SCHED is not None and hasattr(SCHED, 'edge_joint_k'):
                        L = L * SCHED.edge_joint_k(i, t)   # (A-FIX, A: darker forges at THE EDGE so the gilding reads)
                    T = 0.34 + 0.16 * nz + 0.1 * hot + 0.22 * base
                    col = _tw_colours(T)
                    col = col * (1 - 0.7 * red) + C_RED * 0.7 * red
                    face = smoothstep(-0.02, 0.12, ndv) * keep
                    face = face * _line_mod(self, i, kind, pt, t)     # v3 A16/A17 (EMBERS-A3): broken lines of fire
                    self._splat(ctx, i, pt, col * (L * face)[:, None], a, np.clip(ndv, 0.15, 1.0), z, fpx,
                                np.full(len(idx), 0.03, np.float32))
                    continue
                if kind == 3:                                        # windows of fire
                    key = g['key'][idx]
                    lv = G['wlv'][key]
                    wf = 1 + 0.25 * np.sin(G['wfr'][key] * t * 6.2832 / 6.0 + G['wph'][key]) \
                        + 0.12 * np.sin(1.9 * t + G['wph'][key] * 3.0)
                    L = 1.1 * lv * wf * (0.8 + 0.4 * rnd) * (1 + 2.2 * base) * (1 + 2.0 * wave) * surge * (0.3 + 0.7 * vgr) \
                        * fside
                    if SCHED is not None and hasattr(SCHED, 'shutter'):
                        L = L * SCHED.shutter(i, t, pt['pl'])
                    if FS:
                        # the roaring throats (the only windows that face the sky) always burn, seen from anywhere
                        throat = (N[:, 1] > 0.9) & (dtop < 1.5)
                        L = np.where(throat, 2.4 * wf * (0.8 + 0.4 * rnd) * surge, L)
                    T = G['wT'][key] + 0.12 * base - 0.1 * red + 0.05 * (rnd - 0.5)
                    col = _tw_colours(T)
                    col = col * (1 - 0.35 * red) + (C_RED * 0.7 + look.blackbody(0.55) * 0.3) * 0.35 * red
                    face = smoothstep(-0.02, 0.12, ndv) * keep
                    self._splat(ctx, i, pt, col * (L * face)[:, None], a, np.clip(ndv, 0.08, 1.0), z, fpx,
                                np.sqrt(a / np.pi) * 1.15)
                    continue
                if kind == 1:                                        # seams of fire (floors, joints)
                    seg = 0.15 + 1.3 * nz2 ** 2.5
                    L = (0.3 * hot * seg * fl * (1 + 4.0 * base) * heat * vgr + 2.0 * front) * fside + 0.9 * crown
                    if SCHED is not None and hasattr(SCHED, 'seam_k'):
                        L = L * SCHED.seam_k(i)          # (C3: a lit grid of seams reads as offices at night)
                    T = 0.46 + 0.14 * nz2 + 0.2 * base
                    col = _tw_colours(T)
                    col = col * (1 - 0.7 * red) + C_RED * 0.7 * red
                    face = smoothstep(-0.02, 0.12, ndv) * keep
                    face = face * _line_mod(self, i, kind, pt, t)     # v3 A16/A17 (EMBERS-A3): broken lines of fire
                    self._splat(ctx, i, pt, col * (L * face)[:, None], a, np.clip(ndv, 0.25, 1.0), z, fpx,
                                np.full(len(idx), 0.022, np.float32))
                    continue
                # burning edges (corners, tier lips, eaves, ribs); they catch the crown-fire too
                L = (0.2 * hot * (0.35 + 0.9 * nz2) * fl * (1 + 3.5 * base) * heat * (0.5 + 0.5 * far) * vgr + 2.5 * front) \
                    * fside + 1.6 * crown
                T = 0.5 + 0.12 * nz2 + 0.2 * base
                col = _tw_colours(T)
                col = col * (1 - 0.7 * red) + C_RED * 0.7 * red
                Lv = lpos[None, :] - P
                dL = np.linalg.norm(Lv, axis=1)
                lam = np.clip((N * Lv).sum(1) / np.maximum(dL, 1e-6), 0, 1)
                lit = 0.014 * light_pow * lam / (1 + (dL / 16.0) ** 2) * (1 - 0.4 * red)
                face = smoothstep(-0.5, -0.1, ndv) * keep
                face = face * _line_mod(self, i, kind, pt, t)     # v3 A16/A17 (EMBERS-A3): broken lines of fire
                colE = (col * L[:, None] + lcol[None, :] * lit[:, None]) * face[:, None]
                self._splat(ctx, i, pt, colE, a, np.full(len(idx), 0.6, np.float32), z, fpx,
                            np.full(len(idx), 0.03, np.float32))

    def _splat(self, ctx, i, pt, colE, a, geo, z, fpx, rw):
        """radiance (colE, HDR) -> per-point energy = radiance x projected pixel area (full-res pixels)"""
        pa = a * geo * (fpx / z) ** 2
        colE = colE * pa[:, None]
        E = colE.max(1)
        m = E > 1e-7
        if not m.any():
            return
        ctx.fr.splat(pt['P0'][m], pt['P1'][m], rw[m], E[m], colE[m] / E[m][:, None], ctx.cam0, ctx.cam1,
                     zref=0.0, myid=i, profile=1)


class TowerEmbers:
    """v2: embers shed continuously off the towers' burning edges and seams: born on the surface (where the tower
    stood at the moment of birth, so a surging tower leaves its embers behind), drifting up and out, cooling."""

    def __init__(self, towers, seed=121, per=3200):
        r = rng(seed)
        self.tw = towers
        k = towers.k
        self.k = k
        P, Nn = [], []
        for i in range(k):
            e = towers.G[i][2]
            c = towers.G[i][4]
            ne = int(per * (0.65 if len(c['p']) else 1.0))
            ie = r.integers(0, len(e['p']), ne)
            P.append(e['p'][ie])
            Nn.append(e['n'][ie])
            if len(c['p']):
                ic = r.integers(0, len(c['p']), per - ne)
                P.append(c['p'][ic])
                Nn.append(c['n'][ic])
        self.sp = np.concatenate(P).astype(np.float64)
        self.sn = np.concatenate(Nn).astype(np.float64)
        N = len(self.sp)
        self.ti = np.repeat(np.arange(k), per)
        self.life = r.uniform(12.0, 36.0, N)
        self.ph = r.random(N)
        self.v = r.normal(0, 1, (N, 3)) * np.array([0.035, 0.02, 0.035]) + np.array([0.0, 0.06, 0.0])
        self.E = r.lognormal(0, 0.7, N)
        self.tt = np.arange(500.0, 1045.0, 0.5) if SCHED is None else np.arange(SCHED.tower_rise - 20.0,
                                                                                    SCHED.end + 5.0, 0.5)
        self.H = np.array([[towers.height(i, x) for x in self.tt] for i in range(k)])

    def pts(self, t):
        age = ((t - 500.0) / self.life + self.ph) % 1.0 * self.life
        tb = t - age
        hb = np.empty(len(age))
        for i in range(self.k):
            m = self.ti == i
            hb[m] = np.interp(tb[m], self.tt, self.H[i])
        # birth position: the source on the tower at its height back then
        c = np.cos(np.array(self.tw.rot))[self.ti]
        s = np.sin(np.array(self.tw.rot))[self.ti]
        bx = np.array([self.tw.base(i)[0] for i in range(self.k)])[self.ti]
        bz = np.array([self.tw.base(i)[2] for i in range(self.k)])[self.ti]
        p = self.sp
        n = self.sn
        B = np.stack([c * p[:, 0] - s * p[:, 2] + bx, p[:, 1] - self.tw.TW2.HMAX + hb + GROUND,
                      s * p[:, 0] + c * p[:, 2] + bz], 1)
        nw = np.stack([c * n[:, 0] - s * n[:, 2], n[:, 1], s * n[:, 0] + c * n[:, 2]], 1)
        a = age[:, None]
        P = B + nw * (0.15 + 0.02 * a) + self.v * a + np.array([0.0, 0.0011, 0.0]) * a * a
        alive = (p[:, 1] > self.tw.TW2.HMAX - hb) & (hb > 0.5)
        self._last = (B, nw, p[:, 1] - (self.tw.TW2.HMAX - hb), hb)
        return P, age, alive

    def emit(self, ctx):
        t = ctx.t
        if SCHED is not None:
            if t < SCHED.tower_rise + 2 or t >= SCHED.end:
                return
        elif t < 522 or t >= 1040:
            return
        P0, _, _ = self.pts(ctx.t0)
        P1, age, alive = self.pts(ctx.t1)
        P0, P1 = _krot(P0, ctx.t0), _krot(P1, ctx.t1)
        w = vnoise(P1 * 0.25 + np.array([0.0, -0.04 * t, 0.0]), 0.5, (0, 0, 0), 1)
        P0 = P0 + w * (0.05 * age)[:, None]
        P1 = P1 + w * (0.05 * age)[:, None]
        u = age / self.life
        red = redness(t)
        e = self.E * (1 - u) ** 1.6 * smoothstep(0.0, 2.0, age) * 5.0 * alive \
            * (1 + 0.6 * (smoothstep(640, 820, t) if SCHED is None else SCHED.race(t)))
        e = e * (1 + 0.8 * np.array([beat_pulse(t - self.tw.dly[i]) for i in range(self.k)])[self.ti])
        if variant.fire_side_only():
            # v3 (A): embers leave the fire-facing faces and stream off the tops, not off the black backs
            B, nw, yrel, hb = self._last
            Lf = crown_centre(t)[None, :] - B
            fs = smoothstep(-0.2, 0.4, (nw * Lf).sum(1) / np.maximum(np.linalg.norm(Lf, axis=1), 1e-6))
            e = e * (0.15 + 1.6 * fs) * (1.0 + 2.2 * smoothstep(0.72, 0.98, yrel / np.maximum(hb, 1.0)))
        col = look.blackbody(np.clip(0.78 - 0.45 * u, 0.2, 1))
        col = col * (1 - 0.45 * red) + C_RED * 0.45 * red
        if SCHED is not None and hasattr(SCHED, 'near_fade'):
            e = e * SCHED.near_fade(ctx, P1)          # (A, THE EDGE) no big bright streaks across the lens
        ctx.fr.splat(P0, P1, 0.006, e, col, ctx.cam0, ctx.cam1, zref=28.0)


class TowerSmoke:
    """v2: smoke rolling off every tower's top (the tops are lost in it), lit from below by the tower's heat and
    by the crown-fire; left behind as the tower surges."""

    def __init__(self, towers, seed=131, per=420):
        r = rng(seed)
        self.tw = towers
        k = towers.k
        self.k = k
        N = k * per
        self.ti = np.repeat(np.arange(k), per)
        self.life = r.uniform(55.0, 95.0, N)
        self.ph = r.random(N)
        self.off = r.normal(0, 1, (N, 3)) * np.array([1.0, 0.5, 1.0])
        self.E = r.lognormal(0, 0.45, N) * 0.3
        self.tt = np.arange(500.0, 1045.0, 0.5) if SCHED is None else np.arange(SCHED.tower_rise - 20.0,
                                                                                    SCHED.end + 5.0, 0.5)
        self.H = np.array([[towers.height(i, x) for x in self.tt] for i in range(k)])
        self.hw = np.array([2.0, 3.5, 1.5, 2.2, 1.6, 2.2, 2.4, 2.2])     # rough width of each tower's crown

    def emit(self, ctx, light_pos, light_col, light_pow):
        t = ctx.t
        if SCHED is not None:
            if t < SCHED.tower_rise + 10 or t >= SCHED.end:
                return
        elif t < 530 or t >= 1040:
            return
        age = ((t - 500.0) / self.life + self.ph) % 1.0 * self.life
        tb = t - age
        u = age / self.life
        top = np.empty(len(age))
        now = np.empty(len(age))
        for i in range(self.k):
            m = self.ti == i
            top[m] = np.interp(tb[m], self.tt, self.H[i])
            now[m] = self.tw.height(i, t)
        base = np.array([self.tw.base(i) for i in range(self.k)])[self.ti]
        hw = self.hw[self.ti]
        P = base + np.stack([np.zeros_like(age), top - 1.5, np.zeros_like(age)], 1)
        P = P + np.array([0.0, 1.0, 0.0]) * (0.13 * age)[:, None] + self.off * (hw * 0.5 + 0.05 * age)[:, None]
        P = P + vnoise(P * 0.08 + np.array([0.0, -0.01 * t, 0.003 * t]), 0.4, (0, 0, 0), 1) * (0.5 + 0.04 * age)[:, None]
        P = _krot(P, t)
        rad = hw * 0.35 + 0.035 * age
        cam = ctx.cam
        z = np.maximum((P - cam.pos) @ cam.R[2], 1.0)
        fpx = cam.f_px(1920)
        rpx = rad * fpx / z
        # lit from below by the tower's own heat (falls off above its current top) and by the crown-fire
        above = np.maximum(P[:, 1] - (GROUND + now), 0.0)
        glow = np.exp(-above / 7.0) * (0.6 + 0.4 * (smoothstep(640, 800, t) if SCHED is None else SCHED.race(t)))
        d = np.linalg.norm(P - light_pos, axis=1)
        lit = light_pow * 0.0005 / (1 + (d / 14.0) ** 2)
        red = redness(t)
        warm = look.blackbody(0.42) * (1 - 0.6 * red) + (C_CRIMSON * 0.6 + C_RED * 0.4) * 0.6 * red
        L = (0.012 * glow)[:, None] * warm[None, :] + (lit[:, None] * (light_col * 0.5 + warm * 0.5)[None, :])
        L = L * (self.E * smoothstep(0.0, 0.15, u) * (1 - u) ** 1.2
                 * (smoothstep(530, 560, t) if SCHED is None else smoothstep(SCHED.tower_rise + 10,
                                                                            SCHED.tower_rise + 40, t)
                    * getattr(SCHED, 'smoke_gain', 1.0)))[:, None]
        E = L.max(1) * np.pi * rpx ** 2
        m = E > 1e-6
        if not m.any():
            return
        ctx.fr.splat(P[m], P[m], rad[m], E[m], L[m] / np.maximum(L[m].max(1), 1e-9)[:, None], ctx.cam0, ctx.cam1,
                     profile=1, zref=0.0, rmax=300.0)


class Sparks:
    """Spark sprays at every surge (and at emergence)."""

    def __init__(self, towers, seed=66):
        r = rng(seed)
        self.tw = towers
        rows = []
        per = 700
        for b, tb in enumerate(BEATS):
            for i in range(towers.k):
                n = per
                t0 = tb + towers.dly[i] + r.uniform(0, 3.0, n)
                rows.append((i, b, t0, r))
        self.n_per = per
        k = towers.k
        nb = len(BEATS)
        N = k * nb * per
        self.N = N
        self.ti = np.repeat(np.tile(np.arange(k), nb), per)
        self.bi = np.repeat(np.repeat(np.arange(nb), k), per)
        self.tb = np.array(BEATS, float)[self.bi] + towers.dly[self.ti] + r.uniform(0, 3.5, N)
        self.u = r.random(N)                     # height fraction within the top of the tower
        self.base_spark = r.random(N) < 0.22       # some burst from the base where the tower leaves the ground
        self.a = r.uniform(0, 2 * np.pi, N)
        up = r.uniform(0.12, 0.5, N)
        out = r.uniform(0.02, 0.2, N)
        self.v = np.stack([out * np.cos(self.a), up, out * np.sin(self.a)], 1) + r.normal(0, 0.04, (N, 3))
        self.life = r.uniform(16, 38, N)
        self.E = r.lognormal(0, 0.6, N)
        self.k = 0.06
        self.g = np.array([0, -0.012, 0])
        # birth positions (depend on tower height at birth: precompute)
        P = np.zeros((N, 3))
        for i in range(k):
            m = self.ti == i
            tops = np.array([towers.top(i, tt)[1] for tt in self.tb[m]])
            base = towers.base(i)
            rr = 1.4 + 1.5 * r.random(m.sum())
            P[m, 0] = base[0] + rr * np.cos(self.a[m])
            P[m, 2] = base[2] + rr * np.sin(self.a[m])
            P[m, 1] = tops - 8.0 * self.u[m] ** 2
        bs = self.base_spark
        rr2 = 2.0 + 3.0 * r.random(N)
        for i in range(k):
            m = (self.ti == i) & bs
            base = towers.base(i)
            P[m, 0] = base[0] + rr2[m] * np.cos(self.a[m])
            P[m, 2] = base[2] + rr2[m] * np.sin(self.a[m])
            P[m, 1] = GROUND + 0.3
        self.v[bs, 1] *= 0.6
        self.v[bs, 0] *= 2.5
        self.v[bs, 2] *= 2.5
        # later beats: more energetic
        self.E *= 1 + 0.12 * self.bi
        self.p0 = P

    def pts(self, t):
        tau = np.maximum(t - self.tb, 0.0)
        ek = (1 - np.exp(-self.k * tau)) / self.k
        v = self.v + self.g / self.k
        return self.p0 + v * ek[:, None] - (self.g / self.k) * tau[:, None], tau

    def emit(self, ctx):
        t = ctx.t
        if SCHED is not None:
            if t < SCHED.beats[0] - 2 or t >= SCHED.end:
                return
        elif t < 638 or t >= 1000:
            return
        alive = (t >= self.tb) & (t < self.tb + self.life)
        if not alive.any():
            return
        idx = np.nonzero(alive)[0]
        P0, _ = self.pts(ctx.t0)
        P1, tau = self.pts(ctx.t1)
        P0, P1 = _krot(P0, ctx.t0), _krot(P1, ctx.t1)
        x = np.clip(tau[idx] / self.life[idx], 0.0, 1.0)
        e = self.E[idx] * 34.0 * (1 - x) ** 1.3
        red = redness(t)
        col = look.blackbody(np.clip(0.95 - 0.6 * x, 0.2, 1.0))
        col = col * (1 - 0.3 * red) + C_RED * 0.3 * red
        if SCHED is not None and hasattr(SCHED, 'near_fade'):
            e = e * SCHED.near_fade(ctx, P1[idx])
        ctx.fr.splat(P0[idx], P1[idx], 0.005, e, col, ctx.cam0, ctx.cam1, zref=28.0)


class Walls:
    """Radial walls of rising red embers dividing the kingdoms (from 680)."""

    def __init__(self, towers, seed=77):
        r = rng(seed)
        k = towers.k
        per = 22000
        N = k * per
        self.N = N
        self.w = np.repeat(np.arange(k), per)
        self.ang = towers.ang[:k] + np.pi / k + 0.16      # offset so no wall is seen exactly edge-on (the forges only)
        self.u = r.random(N)
        self.ph = r.random(N)
        self.v = r.uniform(0.004, 0.012, N)
        self.off = r.normal(0, 1, N)
        self.E = r.lognormal(0, 0.5, N)
        self.t0w = 658.0 if SCHED is None else getattr(SCHED, 'walls_t0', 1e9)
        self.start = self.t0w + 14 * r.random(k)

    def height(self, t, w):
        base = 37.0 * ease_out((t - self.start[w]) / 30.0, 2.5)
        grow = 0.0
        for b, tb in enumerate(BEATS):
            if tb >= self.t0w + 22:
                x = (t - tb - 3.0) / 6.0
                if x > 0:
                    grow += 3.0 * float(ease_out_back(x, 1.3))
        return base + grow

    def pts(self, t):
        a = self.ang[self.w]
        rr = 5.0 + 16.0 * self.u
        H = np.array([self.height(t, i) for i in range(len(self.ang))])[self.w]
        k = (self.ph + self.v * t) % 1.0
        y = GROUND + k * H
        lat = self.off * 0.18 + 0.5 * np.sin(0.35 * rr + 0.08 * t + self.w)
        P = np.stack([rr * np.cos(a) - lat * np.sin(a), y, rr * np.sin(a) + lat * np.cos(a)], 1)
        wv = vnoise(P * 0.5 + np.array([0, -0.02 * t, 0]), 0.35, (0, 0, 0), 1)
        return P + wv * np.array([0.6, 0.3, 0.6]), k, H

    def emit(self, ctx):
        t = ctx.t
        if t < self.t0w or t >= (1040 if SCHED is None else SCHED.end):
            return
        P0, _, _ = self.pts(ctx.t0)
        P1, k, H = self.pts(ctx.t1)
        P0, P1 = _krot(P0, ctx.t0), _krot(P1, ctx.t1)
        on = t >= self.start[self.w]
        crest = np.exp(-((1 - k) / 0.04) ** 2) * 2.0
        e = self.E * (0.35 + 0.65 * (1 - k) ** 0.6 + crest) * 2.8 * on * (1 + 0.8 * beat_pulse(t - 3))
        e *= smoothstep(0, 10, t - self.start[self.w])
        col = C_RED * (1 - k)[:, None] + C_CRIMSON * k[:, None]
        ctx.fr.splat(P0, P1, 0.006, e, col, ctx.cam0, ctx.cam1, zref=30.0)


class Smoke:
    """Faint smoke lit from within by the fire (and red in the race)."""

    def __init__(self, seed=88):
        r = rng(seed)
        n = 5000
        self.n = n
        a = r.uniform(0, 2 * np.pi, n)
        rr = 2.0 + 34.0 * r.random(n) ** 0.7
        self.p = np.stack([rr * np.cos(a), r.uniform(GROUND, 30.0, n), rr * np.sin(a)], 1)
        if SCHED is not None and hasattr(SCHED, 'haze'):
            # v3 A: the arena's air: more smoke, higher (the giants rise into it) and wider (past the ring)
            r2 = rng(seed + 7)
            m = 7000
            a2 = r2.uniform(0, 2 * np.pi, m)
            rr2 = 4.0 + 52.0 * r2.random(m) ** 0.8
            self.p = np.vstack([self.p, np.stack([rr2 * np.cos(a2), r2.uniform(GROUND, 58.0, m), rr2 * np.sin(a2)], 1)])
            self.n = n = n + m
        self.rw = r.uniform(1.6, 4.2, n)
        self.vy = r.uniform(0.004, 0.02, n)
        self.E = r.lognormal(0, 0.5, n)
        if n > 5000:
            r3 = rng(seed + 8)
            self.rw[5000:] = r3.uniform(3.0, 7.5, n - 5000)
            self.E[5000:] = r3.lognormal(0, 0.6, n - 5000)

    def emit(self, ctx, light_pos, light_col, light_pow):
        t = ctx.t
        if t < IGN or t >= (1040 if SCHED is None else SCHED.end):
            return
        p = self.p.copy()
        p[:, 1] += self.vy * (t - IGN)
        p[:, 1] = GROUND + (p[:, 1] - GROUND) % 72.0
        w = vnoise(p * 0.1 + np.array([0, 0, 0.004 * t]), 0.3, (0, 0, 0), 1)
        p = p + w * 3.0
        d = np.linalg.norm(p - light_pos, axis=1)
        lit = light_pow * 5.0 / (1 + (d / 3.2) ** 2) ** 2
        if SCHED is not None and hasattr(SCHED, 'haze'):
            # the fire lights the whole arena's air, falling off slowly: warm smoke the towers stand out against
            lit = lit + light_pow * 0.07 * SCHED.haze(t) / (1 + (d / 20.0) ** 2) * (1.0 + 2.5 * smoothstep(8.0, 38.0, p[:, 1]))
        red = redness(t)
        amb = 1.2 * red * np.exp(-np.maximum(p[:, 1] - GROUND, 0) / 25.0)
        warm = light_col * 0.4 + look.blackbody(0.6) * 0.6
        colE = warm[None, :] * lit[:, None] + (C_CRIMSON * 0.7 + C_RED * 0.3)[None, :] * amb[:, None]
        colE *= self.E[:, None]
        ctx.fr.splat(p, p, self.rw, np.ones(self.n), colE, ctx.cam0, ctx.cam1, profile=1, zref=0.0,
                     rmax=420.0)


class Dust:
    """Faint floating ash/embers everywhere: depth and parallax."""

    def __init__(self, seed=99):
        r = rng(seed)
        n = 16000
        self.n = n
        a = r.uniform(0, 2 * np.pi, n)
        rr = 60.0 * np.sqrt(r.random(n))
        self.p = np.stack([rr * np.cos(a), r.uniform(GROUND, 90.0, n), rr * np.sin(a)], 1)
        self.E = r.lognormal(0, 0.8, n)
        self.T = r.uniform(0.3, 0.6, n)
        self.ph = r.uniform(0, 6.28, n)

    def emit(self, ctx):
        t = ctx.t
        if t < IGN or t >= (1040 if SCHED is None else SCHED.end):
            return

        def pts(tq):
            q = self.p + np.array([0, 0.01, 0]) * (tq - IGN)
            return q + vnoise(self.p * 0.05 + np.array([0.003 * tq, 0, 0]), 0.5, (0, 0, 0), 1) * 1.5
        red = redness(t)
        e = self.E * 0.5 * (0.6 + 0.4 * np.sin(0.1 * t + self.ph))
        if SCHED is not None and hasattr(SCHED, 'dust_k'):
            k = SCHED.dust_k(t)          # v3 A: the fire alone in a clean black; the air fills as the towers rise
            if k <= 0.0:
                return
            e = e * k
        col = look.blackbody(self.T) * (1 - 0.5 * red) + C_RED * 0.5 * red
        ctx.fr.splat(pts(ctx.t0), pts(ctx.t1), 0.01, e, col, ctx.cam0, ctx.cam1, zref=0.0)


class Vortex:
    """800-960: the crown balloons into a vast unstable vortex (the fire outgrows the hands that fed it)."""

    def __init__(self, fire, seed=111):
        r = rng(seed)
        n = 380000
        self.n = n
        self.rf = 4.0 + 86.0 * r.random(n) ** 1.25
        narm = 4
        self.arm = r.integers(0, narm, n) * 2 * np.pi / narm + r.normal(0, 0.2, n)
        self.h = r.normal(0, 1, n)
        self.E = r.lognormal(0, 0.6, n)
        self.ph = r.uniform(0, 2 * np.pi, n)
        self.jit = r.normal(0, 1, (n, 3))
        # neural filaments grown to storm scale (flattened into the disc)
        self.fp = fire.fp.copy()
        self.fs, self.fm, self.fdep = fire.fs, fire.fm, fire.fdep
        self.nroot = fire.nroot
        self.pv = r.uniform(0.02, 0.035, (self.nroot, 3))
        self.pp = r.uniform(0, 10, (self.nroot, 3))
        self.flick = r.uniform(0, 2 * np.pi, self.nroot)

    def grow(self, t):
        if SCHED is not None:
            return SCHED.vortex_grow(t)
        return float(smootherstep(798, 866, t))

    def spin(self, rr, t):
        om = 0.9 * (rr / 10.0) ** -1.1
        return om * max(t - (800.0 if SCHED is None else SCHED.vortex_t0), 0.0) * 0.1

    def pts(self, t, C):
        g = self.grow(t)
        rr = 4.0 + (self.rf - 4.0) * g ** 0.8
        th = self.arm - 1.25 * np.log(rr / 10.0) + self.spin(rr, t)
        funnel = -8.0 * g * np.exp(-rr / 11.0)
        if SCHED is not None or t < 880 or (variant.tolkien() and t >= 960):
            funnel = funnel + 8.0 * g * math.exp(-4.0 / 11.0)   # v2: the funnel's throat sits on the crown / Ring
        thick = (0.4 + 0.05 * rr) * self.h
        P = np.stack([rr * np.cos(th), funnel + thick, rr * np.sin(th)], 1)
        P += self.jit * (0.3 + 0.03 * rr)[:, None]
        return C + P @ vortex_tilt(t).T, rr

    def fil_pts(self, t, C):
        g = self.grow(t)
        R = 3.0 + 52.0 * g
        q = self.fp
        rho = np.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2) + 1e-6
        rr = np.sqrt(rho ** 2 + q[:, 1] ** 2) * R
        ang = np.arctan2(q[:, 2], q[:, 0]) + self.spin(rr, t) * 0.8
        y = q[:, 1] * 4.0 - 8.0 * g * np.exp(-rr / 11.0)
        if SCHED is not None or t < 880 or (variant.tolkien() and t >= 960):
            y = y + 8.0 * g * math.exp(-4.0 / 11.0)
        return C + np.stack([rr * np.cos(ang), y, rr * np.sin(ang)], 1) @ vortex_tilt(t).T

    def emit(self, ctx):
        t = ctx.t
        g = self.grow(t)
        if g <= 0.001 or t >= (1040 if SCHED is None else SCHED.end):
            return
        C0, C1 = crown_centre(ctx.t0), crown_centre(ctx.t1)
        P0, _ = self.pts(ctx.t0, C0)
        P1, rr = self.pts(ctx.t1, C1)
        x = clamp01((rr - 4.0) / 80.0)
        ek = eye_k(t)
        core = C_CORE * (1 - ek) + look.blackbody(0.72) * ek
        col = core * (1 - smoothstep(0.0, 0.06, x))[:, None]
        col += C_GOLD * (smoothstep(0.0, 0.06, x) * (1 - smoothstep(0.06, 0.25, x)))[:, None]
        col += look.blackbody(0.55) * (smoothstep(0.06, 0.25, x) * (1 - smoothstep(0.25, 0.5, x)))[:, None]
        col += C_RED * (smoothstep(0.25, 0.5, x) * (1 - smoothstep(0.5, 0.9, x)))[:, None]
        col += C_CRIMSON * smoothstep(0.5, 0.9, x)[:, None]
        # instability: travelling brightness waves, beat surges, flicker
        wave = 0.55 + 0.45 * np.sin(0.45 * rr - 0.5 * t + self.ph * 0.3)
        surge = (1 + 0.6 * beat_pulse(t)) * (1 - 0.6 * (smoothstep(956, 972, t) if SCHED is None else 0.0))
        e = self.E * wave * (1.6 + 3.5 * np.exp(-rr / 9.0) * (1 - 0.75 * ek)) * 1.3 * g * surge * (1 - 0.5 * ek * np.exp(-rr / 14.0))
        if SCHED is not None and hasattr(SCHED, 'vortex_shape'):
            e, P0, P1 = SCHED.vortex_shape(self, t, e, P0, P1, rr)
        ctx.fr.splat(P0, P1, 0.02, e, col, ctx.cam0, ctx.cam1, zref=60.0)
        # the thinking filaments stretched across the storm, pulses racing out along them
        F0 = self.fil_pts(ctx.t0, C0)
        F1 = self.fil_pts(ctx.t1, C1)
        pulse = np.zeros(len(self.fs))
        for j in range(3):
            sp = ((t - 800) * self.pv[self.fm, j] + self.pp[self.fm, j]) % 1.8
            pulse += np.exp(-((self.fs - sp) / 0.05) ** 2)
        fl = 0.5 + 0.5 * np.sin(1.7 * t + self.flick[self.fm]) * np.sin(0.63 * t + 2 * self.flick[self.fm])
        ef = (0.25 + 5.0 * pulse) * fl * 22.0 * g * (self.fdep <= 2) * (1 - 0.6 * (smoothstep(956, 972, t) if SCHED is None else 0.0))
        fcol = C_ICE * 0.6 + C_CORE * 0.4
        ctx.fr.splat(F0, F1, 0.03, ef, fcol, ctx.cam0, ctx.cam1, zref=60.0)


# ------------------------------------------------------------- camera ---

ALPHA_C = TOWER_ANG0 + np.pi / 8 + 0.05   # camera azimuth: in a gap; the opposite gap is behind the crown


def _polar(r, a, y):
    return np.array([r * math.cos(a), y, r * math.sin(a)])


CAM_B = [  # (frame, radius, azimuth offset, height, target y)
    (480, 13.9, 0.43, 7.2, -1.35),
    (500, 16.6, 0.16, -2.9, 0.5),       # v3 (red team): back in and tilted up at the fire, which fills ~40% of
    (520, 16.2, -0.08, -3.3, 0.6),      # the frame height, its tip inside the frame, above the caption
    (548, 17.2, -0.05, -2.5, 1.0),
    (580, 23.0, -0.02, 8.0, 6.5),
    (610, 30.0, 0.00, 15.5, 14.0),
    (640, 36.0, 0.00, 21.5, 19.0),
    (700, 35.0, -0.06, 23.0, 25.0),
    (760, 34.0, -0.14, 33.0, 36.5),
    (800, 34.0, -0.2, 40.0, 44.0),
    (840, 50.0, -0.22, 30.0, 60.0),      # v2: tilts up after the storm as it climbs clear of the towers
    (880, 60.0, -0.24, 26.0, 64.0),
]


# v3 (C only): the Eye is centred over the whole ring of towers -- the camera swings back so the gap opposite it lies
# behind the Eye (v2 kept -0.22 rad and the Eye sat on one tower's apex)
CAM_B_C = [k if k[0] < 800 else {800: (800, 34.0, -0.12, 40.0, 44.0), 840: (840, 60.0, 0.0, 36.0, 58.0),
                                  880: (880, 68.0, 0.0, 34.0, 62.0)}[k[0]] for k in CAM_B]


def _cam_keys():
    return CAM_B_C if variant.tolkien() else CAM_B


def cam_b(t):
    k = [(f, np.array([r, a, y, ty])) for f, r, a, y, ty in _cam_keys()]
    v = catmull(t, k)
    r, a, y, ty = v
    pos = _polar(r, ALPHA_C + a, y)
    tgt = np.array([0.0, ty, 0.0])
    # beat shake in the race
    sh = np.zeros(3)
    for tb in BEATS:
        x = t - tb
        if 0 <= x < 14:
            amp = 0.4 * math.exp(-x / 3.0) * (0.6 + 0.4 * smoothstep(640, 800, t))
            sh += amp * np.array([math.sin(2.1 * x + tb), math.sin(2.9 * x + 0.5 * tb), 0.0])
    return pos + sh * 0.5, tgt + sh
