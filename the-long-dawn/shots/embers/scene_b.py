"""EMBERS part B (480-880): ignition, the thinking fire, towers, crown, the race, the vortex."""
import math

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


# ------------------------------------------------------------ timelines ---

def redness(t):
    """0 -> 1 palette bleed toward crimson during the race (decisive by ~780)."""
    return float(smoothstep(646, 780, t))


def beat_pulse(t):
    """1 on each race beat, decaying fast (for brightness pulses)."""
    if t < 640 or t >= 960:
        return 0.0
    x = (t - 640) % 20.0
    return math.exp(-x / 3.0)


STORM_LIFT = 16.0


def crown_centre(t):
    y = 20.0 * smootherstep(548, 634, t) + 28.0 * smoothstep(640, 820, t) + 6.0 * smoothstep(820, 900, t)
    if t < 880:
        # v2: the storm climbs clear of the towers' crowns as it grows (they are solid now and would hide its
        # eye); the grasp (960+) keeps v1's geometry
        y += STORM_LIFT * smootherstep(804, 852, t)
    return np.array([0.0, float(y), 0.0])


def fire_radius(t):
    grow = float(ease_out_expo((t - IGN) / 22.0, 6.0))
    breath = 1 + 0.07 * math.sin(2 * math.pi * (t - IGN) / 80.0 - math.pi / 2)
    return 1.3 * grow * breath


def crown_morph(t):
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
    k = [(f, np.array([r, a, y, ty])) for f, r, a, y, ty in CAM_B]
    r, a, y, ty = catmull(t, k)
    pos = _polar(r, ALPHA_C + a, y)
    C = crown_centre(t)
    return math.atan2(pos[2], pos[0]), math.atan2(C[1] - pos[1], math.hypot(pos[0] - C[0], pos[2] - C[2]))


def crown_tilt(t):
    """Rotation (3x3) tilting the crown's axis toward the camera (hero angle for a floating ring).
    v2: from 798 the ring keeps turning its face toward the lens as the storm is born (v1 flattened it and the
    ring passed edge-on); it faces the lens by ~842."""
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
    if t < 960:
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
    p *= 1 + 0.12 * math.sin(2 * math.pi * (t - IGN) / 80.0 - math.pi / 2)
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
    (0.25, 2.30, 0.27, 0.26, 0.31, 0.0),
    (2.45, 1.90, 0.22, 0.34, 0.39, 1.9),
    (4.30, 1.62, 0.19, 0.40, 0.47, 3.6),
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
    Ht = H * (1.0 + 0.1 * np.sin(om * t + ph) + 0.05 * np.sin(2.7 * om * t + 2.0 * ph))
    y0 = 0.72
    y = y0 + s * (Ht - y0)
    rb = 0.2 * (1.0 - 0.55 * s)
    sway = th + 0.45 * np.sin(0.09 * t + ph)
    lick = 0.13 * s * np.sin(2 * np.pi * 1.4 * s - om * 1.6 * t + ph)
    px = rb * np.cos(th) + lean * s * s * np.cos(sway) - lick * np.sin(th)
    pz = rb * np.sin(th) + lean * s * s * np.sin(sway) + lick * np.cos(th)
    w = W * (1.0 - s) ** 0.72 * (1.0 + 0.18 * np.sin(1.3 * om * t + 5.0 * s + ph))
    return np.stack([px, y, pz], 1), w


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
        if t < IGN or t >= 960:
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
            wv = vnoise(q1 * 1.4 + np.array([0.0, -0.09 * (t - IGN), 0.0]), 1.3, (5.0, 1.0, 2.0), 2)
            edge = smoothstep(0.45, 1.0, np.linalg.norm(q1, axis=1))
            q0 = q0 + wv * (0.16 * fw * edge)[:, None]
            q1 = q1 + wv * (0.16 * fw * edge)[:, None]
        yc = 0.08 + 0.3 * fw                                           # the hot zone sits up in the body
        d = np.sqrt(q1[:, 0] ** 2 + q1[:, 2] ** 2 + ((q1[:, 1] - yc) / (1.0 + 0.6 * fw)) ** 2)
        hi = smoothstep(0.4, 1.1, q1[:, 1]) * fw                    # the upper body burns gold into the tongues
        low = smoothstep(0.1, -0.6, q1[:, 1]) * fw                   # a cooler, dimmer base
        q0 = flame_shape(q0, flame_w(ctx.t0), lift=0.32)
        q1 = flame_shape(q1, flame_w(ctx.t1), lift=0.32)
        P0 = C0 + self.warp(q0, crown_morph(ctx.t0), R0, ctx.t0) @ crown_tilt(ctx.t0).T
        P1 = C1 + self.warp(q1, crown_morph(ctx.t1), R1, ctx.t1) @ crown_tilt(ctx.t1).T
        if fw > 0:
            # a flame: a small white-hot core up in the body, ice-blue inner light, gold edges running up the tongues
            c1 = lerp(smoothstep(0.08, 0.42, d), smoothstep(0.04, 0.28, d), fw)[:, None]
            c2 = np.maximum(lerp(smoothstep(0.5, 0.95, d), smoothstep(0.42, 0.85, d), fw), hi)[:, None]
        else:
            c1 = smoothstep(0.08, 0.42, d)[:, None]
            c2 = smoothstep(0.5, 0.95, d)[:, None]
        gold = C_GOLD * (1 - 0.55 * red) + C_RED * 0.55 * red
        if fw > 0:
            gold = gold * (1 - 0.55 * fw) + look.blackbody(0.63) * (0.55 * fw)     # a deeper gold at the flame's edges
        col = C_CORE * (1 - c1) + C_ICE * c1
        col = col * (1 - c2) + gold * c2
        fl = 1 + 0.35 * np.sin(0.7 * t + self.fl)
        e = self.E * fl * (1.9 - 1.1 * d) * 0.55 * pw
        if fw > 0:
            e = e * (1.0 - 0.8 * low) * (1.0 - 0.25 * fw)
            col = col * (1.0 - 0.55 * low[:, None]) + (C_ICE * 0.55 + C_GOLD * 0.15) * (0.55 * low[:, None])
        e *= 1.0 + 0.6 * m          # ring is thinner: keep it bright
        e *= (1.0 - 0.85 * forged(t)) * (1.0 - eye_k(t))   # cut C: the fire becomes the band (a faint aura stays)
        ctx.fr.splat(P0, P1, 0.004, e, col, ctx.cam0, ctx.cam1)
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

        def tongue(kk, tq):
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
            sq = kk ** 0.85
            ax, wd = tongue_axis(self.kt, sq, tq, fwq)
            th = np.array(FLAME_TONGUES)[self.kt, 0]
            e1 = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], 1)
            e2 = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1)
            pn = ax + wd[:, None] * (self.tu[:, :1] * e1 + self.tu[:, 1:] * e2 * 0.8)
            wn = vnoise(pn * 1.6 + np.array([0.0, -0.11 * (tq - IGN), 0.0]), 1.5, (0, 0, 0), 2)
            pn = pn + wn * (0.03 + 0.14 * sq)[:, None]
            return lerp(p_old, pn, fwq)
        T0 = C0 + self.warp(tongue(kk0, ctx.t0), crown_morph(ctx.t0), R0, ctx.t0) @ crown_tilt(ctx.t0).T
        T1 = C1 + self.warp(tongue(kk1, ctx.t1), crown_morph(ctx.t1), R1, ctx.t1) @ crown_tilt(ctx.t1).T
        et = self.tE * (1 - kk1) ** 1.6 * smoothstep(0.0, 0.08, kk1) * 2.4 * pw * wrap_ok * (0.3 + 0.7 * storm_fade(t))
        et = et * (1.0 - 0.9 * forged(t)) * (1.0 + 0.35 * flame_w(t))
        tcol = look.blackbody(0.9 - 0.45 * kk1)
        tcol = tcol * (1 - 0.35 * red) + C_RED * 0.35 * red * np.ones_like(tcol)
        ctx.fr.splat(T0, T1, 0.004, et, tcol, ctx.cam0, ctx.cam1)
        # --- filaments + pulses
        grow = smoothstep(0.0, 1.0, (t - IGN) / 26.0) * 1.3
        vis = self.fs < grow
        s = self.fs
        pulse = np.zeros(len(s))
        L = 1.4
        for j in range(4):
            sp = ((t - IGN) * self.pv[self.fm, j] + self.pp[self.fm, j]) % (L + 0.6)
            sp = np.where(self.pdir[self.fm, j] > 0, sp, L - sp)
            pulse += np.exp(-((s - sp) / 0.035) ** 2)
        base = 0.55 + 0.35 * (self.fdep == 0)
        ef = (base + 10.0 * pulse) * vis * 1.3 * pw * (1.0 - forged(t))
        if fw > 0:
            ef = ef * (1.0 - 0.96 * fw * smoothstep(0.08, -0.32, self.fp[:, 1]))   # no filament coil at the base
            ef = ef * (1.0 - 0.3 * fw * (1.0 - np.minimum(pulse, 1.0)))              # the pulses carry the thought
        fcol = C_ICE * (1 - np.minimum(pulse, 1))[:, None] + C_CORE * np.minimum(pulse, 1)[:, None]
        fq0 = flame_shape(self.fp, flame_w(ctx.t0), stretch_up=2.1, reach=1.3, down=1.0, lift=0.22)
        fq1 = flame_shape(self.fp, flame_w(ctx.t1), stretch_up=2.1, reach=1.3, down=1.0, lift=0.22)
        F0 = C0 + self.warp(fq0, crown_morph(ctx.t0), R0, ctx.t0) @ crown_tilt(ctx.t0).T
        F1 = C1 + self.warp(fq1, crown_morph(ctx.t1), R1, ctx.t1) @ crown_tilt(ctx.t1).T
        ctx.fr.splat(F0, F1, 0.002, ef, fcol, ctx.cam0, ctx.cam1)
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
            ef = np.array([200.0, 300.0, 170.0, 360.0]) * pw * smoothstep(IGN, IGN + 6, t)
            cf = np.array([C_CORE, C_ICE * 0.5 + C_GOLD * 0.5, C_GOLD, C_GOLD * (1 - red) + C_RED * red])
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
            kt = (np.arange(self.n) % 3)
            tip, _ = tongue_axis(kt, np.full(self.n, 0.97), t, fw)
            spr = 0.15 + 1.6 * k
            xf = tip[:, 0] * R + self.sp[:, 0] * spr
            zf = tip[:, 2] * R + self.sp[:, 1] * spr
            yf = tip[:, 1] * R + k * 6.5
            x = lerp(x, xf, fw)
            y = lerp(y, yf, fw)
            z = lerp(z, zf, fw)
        P = C + np.stack([x, y, z], 1) @ crown_tilt(t).T
        w = vnoise(P * 0.3 + np.array([0, -0.03 * t, 0]), 0.6, (0, 0, 0), 1)
        return P + w * (0.3 + 1.2 * k)[:, None], k

    def emit(self, ctx):
        t = ctx.t
        if t < IGN + 6 or t >= 960:
            return
        P0, k0 = self.pts(ctx.t0)
        P1, k = self.pts(ctx.t1)
        ok = k >= k0
        e = self.E * (1 - k) ** 1.5 * 9.0 * smoothstep(IGN + 6, IGN + 20, t) * ok * storm_fade(t)
        e = e * (1.0 - 0.7 * flame_w(t))
        red = redness(t)
        col = look.blackbody(0.92 - 0.45 * k)
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


def _tw_colours(T):
    return look.blackbody(np.clip(T, 0.0, 1.0))


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
        self.t_rise = 520 + np.array([0, 9, 4, 14, 6, 11, 2, 8], float)
        self.h_rise = np.array([40.0, 38.0, 35.0, 27.0, 28.5, 26.0, 27.5, 36.0])
        J = r.uniform(1.6, 2.8, (k, len(BEATS)))
        lead = r.permutation(np.tile(np.arange(k), 2))[:len(BEATS)]
        for b, l in enumerate(lead):
            J[l, b] += r.uniform(3.0, 4.5)
        J[3:7] *= 1.25
        self.J = J
        self.dly = r.uniform(0, 2.5, k)
        self.rot = [(-a + np.pi) for a in self.ang]
        import towers2 as TW2
        self.TW2 = TW2
        T = TW2.build_all()
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
        print('towers v2:', ' '.join(str(sum(len(g[kk]['p']) for kk in range(5))) for g in self.G))

    # ------------------------------------------------------------- motion (v1)
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
        b = self.base(i)
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
                w += math.exp(-x / 11.0) * np.exp(-((yl - yw) / 3.2) ** 2)
        return w

    # ------------------------------------------------------------- per frame
    def prepare(self, ctx):
        """World positions of the visible, level-of-detail points of every tower; builds the occluder."""
        t = ctx.t
        self.cur = None
        if t < 515 or t >= 1040:
            return
        cam = ctx.cam
        cur = []
        oP, oN, oA, oI = [], [], [], []
        for i in range(self.k):
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
            cr = parts[0]
            if len(cr['idx']):
                a = self.G[i][0]['a'][cr['idx']] / cr['q']
                oP.append(cr['P'])
                oN.append(cr['N'])
                oA.append(a)
                oI.append(np.full(len(a), i, np.int32))
            cur.append(dict(h=h, parts=parts))
        self.cur = cur
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
        if t >= 960:
            grow *= 0.6            # the grasp: the towers stand back in the storm's dark; the hand leads
        for i in range(self.k):
            c = self.cur[i]
            if c is None:
                continue
            G = self.G[i]
            bp = beat_pulse(t - self.dly[i])
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
                keep = ftop * smoothstep(-0.3, 0.4, yl)              # thin at the very top; nothing underground
                heat = (1 + 2.5 * wave) * surge
                if kind == 0:                                        # the crust: a glowing coal, ash patches, rims
                    key = g['key'][idx]
                    opening = np.where(key >= 0, 0.12, 1.0)
                    ash = smoothstep(0.2, 0.85, 0.35 * nz + 0.65 * nzv)  # glowing coal vs cooler ash, streaked upward
                    grain = 0.85 + 0.3 * rnd
                    L = 0.09 * (0.3 + 1.4 * ash) * (0.8 + 0.4 * nz2) * grain * fl * opening * (1 + 3.0 * base) * heat
                    rim = np.clip(1.0 - ndv / 0.35, 0, 1) ** 2 * (ndv > 0)
                    L = (L + 0.35 * rim * (0.6 + 0.6 * nz2) * fl * heat) * vgr + 2.2 * front
                    T = 0.27 + 0.1 * ash + 0.18 * base + 0.2 * front
                    col = _tw_colours(T)
                    col = col * (1 - 0.8 * red) + C_CR * 0.8 * red
                    Lv = lpos[None, :] - P
                    dL = np.linalg.norm(Lv, axis=1)
                    lam = np.clip((N * Lv).sum(1) / np.maximum(dL, 1e-6), 0, 1)
                    lit = 0.006 * light_pow * lam ** 1.3 / (1 + (dL / 16.0) ** 2) * (1 - 0.4 * red) * opening
                    lit = lit * np.clip(hot / 0.3, 0, 1) ** 2          # roofs (hot 0.2) stay dark tile
                    face = smoothstep(-0.02, 0.1, ndv) * keep
                    colE = (col * L[:, None] + lcol[None, :] * lit[:, None]) * face[:, None]
                    self._splat(ctx, i, pt, colE, a, np.clip(ndv, 0.05, 1.0), z, fpx, np.sqrt(a / np.pi) * 1.7)
                    continue
                if kind == 4:                                        # fire in the joints (masonry / mullions / fissures)
                    # soft crevices of fire: a hot core line fading into the stone; slow patches of heat
                    patch = 0.2 + 1.3 * smoothstep(0.25, 0.9, 0.4 * nz + 0.6 * nzv) ** 1.5
                    L = 0.6 * far * (0.2 + 0.8 * hot ** 2) * patch * (0.8 + 0.4 * nz2) * fl * (1 + 3.0 * base) * heat \
                        * vgr + 2.0 * front
                    T = 0.34 + 0.16 * nz + 0.1 * hot + 0.22 * base
                    col = _tw_colours(T)
                    col = col * (1 - 0.7 * red) + C_RED * 0.7 * red
                    face = smoothstep(-0.02, 0.12, ndv) * keep
                    self._splat(ctx, i, pt, col * (L * face)[:, None], a, np.clip(ndv, 0.15, 1.0), z, fpx,
                                np.full(len(idx), 0.03, np.float32))
                    continue
                if kind == 3:                                        # windows of fire
                    key = g['key'][idx]
                    lv = G['wlv'][key]
                    wf = 1 + 0.25 * np.sin(G['wfr'][key] * t * 6.2832 / 6.0 + G['wph'][key]) \
                        + 0.12 * np.sin(1.9 * t + G['wph'][key] * 3.0)
                    L = 1.1 * lv * wf * (0.8 + 0.4 * rnd) * (1 + 2.2 * base) * (1 + 2.0 * wave) * surge * (0.3 + 0.7 * vgr)
                    T = G['wT'][key] + 0.12 * base - 0.1 * red + 0.05 * (rnd - 0.5)
                    col = _tw_colours(T)
                    col = col * (1 - 0.35 * red) + (C_RED * 0.7 + look.blackbody(0.55) * 0.3) * 0.35 * red
                    face = smoothstep(-0.02, 0.12, ndv) * keep
                    self._splat(ctx, i, pt, col * (L * face)[:, None], a, np.clip(ndv, 0.08, 1.0), z, fpx,
                                np.sqrt(a / np.pi) * 1.15)
                    continue
                if kind == 1:                                        # seams of fire (floors, joints)
                    seg = 0.15 + 1.3 * nz2 ** 2.5
                    L = 0.3 * hot * seg * fl * (1 + 4.0 * base) * heat * vgr + 2.0 * front
                    T = 0.46 + 0.14 * nz2 + 0.2 * base
                    col = _tw_colours(T)
                    col = col * (1 - 0.7 * red) + C_RED * 0.7 * red
                    face = smoothstep(-0.02, 0.12, ndv) * keep
                    self._splat(ctx, i, pt, col * (L * face)[:, None], a, np.clip(ndv, 0.25, 1.0), z, fpx,
                                np.full(len(idx), 0.022, np.float32))
                    continue
                # burning edges (corners, tier lips, eaves, ribs); they catch the crown-fire too
                L = 0.2 * hot * (0.35 + 0.9 * nz2) * fl * (1 + 3.5 * base) * heat * (0.5 + 0.5 * far) * vgr + 2.5 * front
                T = 0.5 + 0.12 * nz2 + 0.2 * base
                col = _tw_colours(T)
                col = col * (1 - 0.7 * red) + C_RED * 0.7 * red
                Lv = lpos[None, :] - P
                dL = np.linalg.norm(Lv, axis=1)
                lam = np.clip((N * Lv).sum(1) / np.maximum(dL, 1e-6), 0, 1)
                lit = 0.014 * light_pow * lam / (1 + (dL / 16.0) ** 2) * (1 - 0.4 * red)
                face = smoothstep(-0.5, -0.1, ndv) * keep
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
        self.tt = np.arange(500.0, 1045.0, 0.5)
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
        return P, age, alive

    def emit(self, ctx):
        t = ctx.t
        if t < 522 or t >= 1040:
            return
        P0, _, _ = self.pts(ctx.t0)
        P1, age, alive = self.pts(ctx.t1)
        P0, P1 = _krot(P0, ctx.t0), _krot(P1, ctx.t1)
        w = vnoise(P1 * 0.25 + np.array([0.0, -0.04 * t, 0.0]), 0.5, (0, 0, 0), 1)
        P0 = P0 + w * (0.05 * age)[:, None]
        P1 = P1 + w * (0.05 * age)[:, None]
        u = age / self.life
        red = redness(t)
        e = self.E * (1 - u) ** 1.6 * smoothstep(0.0, 2.0, age) * 5.0 * alive * (1 + 0.6 * smoothstep(640, 820, t))
        e = e * (1 + 0.8 * np.array([beat_pulse(t - self.tw.dly[i]) for i in range(self.k)])[self.ti])
        col = look.blackbody(np.clip(0.78 - 0.45 * u, 0.2, 1))
        col = col * (1 - 0.45 * red) + C_RED * 0.45 * red
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
        self.tt = np.arange(500.0, 1045.0, 0.5)
        self.H = np.array([[towers.height(i, x) for x in self.tt] for i in range(k)])
        self.hw = np.array([2.0, 3.5, 1.5, 2.2, 1.6, 2.2, 2.4, 2.2])     # rough width of each tower's crown

    def emit(self, ctx, light_pos, light_col, light_pow):
        t = ctx.t
        if t < 530 or t >= 1040:
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
        glow = np.exp(-above / 7.0) * (0.6 + 0.4 * smoothstep(640, 800, t))
        d = np.linalg.norm(P - light_pos, axis=1)
        lit = light_pow * 0.0005 / (1 + (d / 14.0) ** 2)
        red = redness(t)
        warm = look.blackbody(0.42) * (1 - 0.6 * red) + (C_CRIMSON * 0.6 + C_RED * 0.4) * 0.6 * red
        L = (0.012 * glow)[:, None] * warm[None, :] + (lit[:, None] * (light_col * 0.5 + warm * 0.5)[None, :])
        L = L * (self.E * smoothstep(0.0, 0.15, u) * (1 - u) ** 1.2 * smoothstep(530, 560, t))[:, None]
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
        if t < 638 or t >= 1000:
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
        self.ang = towers.ang + np.pi / k + 0.16      # offset so no wall is seen exactly edge-on
        self.u = r.random(N)
        self.ph = r.random(N)
        self.v = r.uniform(0.004, 0.012, N)
        self.off = r.normal(0, 1, N)
        self.E = r.lognormal(0, 0.5, N)
        self.start = 658 + 14 * r.random(k)

    def height(self, t, w):
        base = 37.0 * ease_out((t - self.start[w]) / 30.0, 2.5)
        grow = 0.0
        for b, tb in enumerate(BEATS):
            if tb >= 680:
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
        if t < 658 or t >= 1040:
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
        self.rw = r.uniform(1.6, 4.2, n)
        self.vy = r.uniform(0.004, 0.02, n)
        self.E = r.lognormal(0, 0.5, n)

    def emit(self, ctx, light_pos, light_col, light_pow):
        t = ctx.t
        if t < IGN or t >= 1040:
            return
        p = self.p.copy()
        p[:, 1] += self.vy * (t - IGN)
        p[:, 1] = GROUND + (p[:, 1] - GROUND) % 70.0
        w = vnoise(p * 0.1 + np.array([0, 0, 0.004 * t]), 0.3, (0, 0, 0), 1)
        p = p + w * 3.0
        d = np.linalg.norm(p - light_pos, axis=1)
        lit = light_pow * 5.0 / (1 + (d / 3.2) ** 2) ** 2
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
        if t < IGN or t >= 1040:
            return

        def pts(tq):
            q = self.p + np.array([0, 0.01, 0]) * (tq - IGN)
            return q + vnoise(self.p * 0.05 + np.array([0.003 * tq, 0, 0]), 0.5, (0, 0, 0), 1) * 1.5
        red = redness(t)
        e = self.E * 0.5 * (0.6 + 0.4 * np.sin(0.1 * t + self.ph))
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
        return float(smootherstep(798, 866, t))

    def spin(self, rr, t):
        om = 0.9 * (rr / 10.0) ** -1.1
        return om * max(t - 800.0, 0.0) * 0.1

    def pts(self, t, C):
        g = self.grow(t)
        rr = 4.0 + (self.rf - 4.0) * g ** 0.8
        th = self.arm - 1.25 * np.log(rr / 10.0) + self.spin(rr, t)
        funnel = -8.0 * g * np.exp(-rr / 11.0)
        if t < 880 or (variant.tolkien() and t >= 960):
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
        if t < 880 or (variant.tolkien() and t >= 960):
            y = y + 8.0 * g * math.exp(-4.0 / 11.0)
        return C + np.stack([rr * np.cos(ang), y, rr * np.sin(ang)], 1) @ vortex_tilt(t).T

    def emit(self, ctx):
        t = ctx.t
        g = self.grow(t)
        if g <= 0.001 or t >= 1040:
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
        surge = (1 + 0.6 * beat_pulse(t)) * (1 - 0.6 * smoothstep(956, 972, t))
        e = self.E * wave * (1.6 + 3.5 * np.exp(-rr / 9.0) * (1 - 0.75 * ek)) * 1.3 * g * surge * (1 - 0.5 * ek * np.exp(-rr / 14.0))
        ctx.fr.splat(P0, P1, 0.02, e, col, ctx.cam0, ctx.cam1, zref=60.0)
        # the thinking filaments stretched across the storm, pulses racing out along them
        F0 = self.fil_pts(ctx.t0, C0)
        F1 = self.fil_pts(ctx.t1, C1)
        pulse = np.zeros(len(self.fs))
        for j in range(3):
            sp = ((t - 800) * self.pv[self.fm, j] + self.pp[self.fm, j]) % 1.8
            pulse += np.exp(-((self.fs - sp) / 0.05) ** 2)
        fl = 0.5 + 0.5 * np.sin(1.7 * t + self.flick[self.fm]) * np.sin(0.63 * t + 2 * self.flick[self.fm])
        ef = (0.25 + 5.0 * pulse) * fl * 22.0 * g * (self.fdep <= 2) * (1 - 0.6 * smoothstep(956, 972, t))
        fcol = C_ICE * 0.6 + C_CORE * 0.4
        ctx.fr.splat(F0, F1, 0.03, ef, fcol, ctx.cam0, ctx.cam1, zref=60.0)


# ------------------------------------------------------------- camera ---

ALPHA_C = TOWER_ANG0 + np.pi / 8 + 0.05   # camera azimuth: in a gap; the opposite gap is behind the crown


def _polar(r, a, y):
    return np.array([r * math.cos(a), y, r * math.sin(a)])


CAM_B = [  # (frame, radius, azimuth offset, height, target y)
    (480, 13.9, 0.43, 7.2, -1.35),
    (500, 17.4, 0.16, 2.3, 0.0),        # v2: ~15% further back (and a wider lens, timeline.camera): the whole
    (520, 17.8, -0.08, 2.2, 0.1),       # flame, its tip inside the frame, above the caption
    (548, 19.6, -0.05, 2.7, 0.6),
    (580, 23.0, -0.02, 8.0, 6.5),
    (610, 30.0, 0.00, 15.5, 14.0),
    (640, 36.0, 0.00, 21.5, 19.0),
    (700, 35.0, -0.06, 23.0, 25.0),
    (760, 34.0, -0.14, 33.0, 36.5),
    (800, 34.0, -0.2, 40.0, 44.0),
    (840, 50.0, -0.22, 30.0, 60.0),      # v2: tilts up after the storm as it climbs clear of the towers
    (880, 60.0, -0.24, 26.0, 64.0),
]


def cam_b(t):
    k = [(f, np.array([r, a, y, ty])) for f, r, a, y, ty in CAM_B]
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
