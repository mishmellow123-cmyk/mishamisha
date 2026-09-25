"""EMBERS part B (480-880): ignition, the thinking fire, towers, crown, the race, the vortex."""
import math

import numpy as np

import look
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
IGN = 480.0
BEATS = [640 + 20 * k for k in range(16)]          # 640 .. 940


# ------------------------------------------------------------ timelines ---

def redness(t):
    """0 -> 1 palette bleed toward crimson during the race."""
    return float(smoothstep(648, 800, t))


def crown_centre(t):
    y = 10.0 * smootherstep(552, 632, t) + 38.0 * ease_in_out((t - 640) / 300.0)
    return np.array([0.0, float(y), 0.0])


def fire_radius(t):
    grow = float(ease_out_expo((t - IGN) / 22.0, 6.0))
    breath = 1 + 0.07 * math.sin(2 * math.pi * (t - IGN) / 80.0 - math.pi / 2)
    return 1.7 * grow * breath


def crown_morph(t):
    return float(smootherstep(562, 628, t))


def fire_power(t):
    """overall luminous power of the fire/crown (drives tower lighting too)."""
    if t < IGN:
        return 0.0
    p = 1.0 + 3.0 * math.exp(-(t - IGN) / 8.0)
    p *= 1 + 0.12 * math.sin(2 * math.pi * (t - IGN) / 80.0 - math.pi / 2)
    p *= 1 + 0.5 * smoothstep(640, 900, t)
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
        m = 26000
        self.m = m
        d = rand_dirs(r, m)
        d[:, 1] = np.abs(d[:, 1]) * 0.9 + 0.1 * d[:, 1]
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
        Rr = 3.2 + 0.6 * smoothstep(640, 800, t)
        rho2 = lerp(rho * R, Rr + (rho - 0.45) * 0.62 * 1.4, m)
        y2 = lerp(y * R, y * 0.45 * 1.4, m)
        phi2 = phi + m * 0.035 * (t - 562)
        return np.stack([rho2 * np.cos(phi2), y2, rho2 * np.sin(phi2)], 1)

    def flow_pts(self, t):
        tt = t - IGN
        th = self.th0 - self.om * tt
        rho = self.rc + self.a * np.cos(th)
        y = self.y0 + self.b * np.sin(th)
        y = np.where(y > 0, y * 1.25, y * 0.9)
        ph = self.ph0 + self.omp * tt
        p = np.stack([rho * np.cos(ph), y, rho * np.sin(ph)], 1)
        w = vnoise(p * 1.0 + np.array([0.0, -0.03 * tt, 0.0]), 1.6, (0.0, 0.0, 0.0), 2)
        return p + w * 0.07

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
        P0 = C0 + self.warp(q0, crown_morph(ctx.t0), R0, ctx.t0)
        P1 = C1 + self.warp(q1, crown_morph(ctx.t1), R1, ctx.t1)
        d = np.linalg.norm(q1, axis=1)
        c1 = smoothstep(0.08, 0.42, d)[:, None]
        c2 = smoothstep(0.5, 0.95, d)[:, None]
        gold = C_GOLD * (1 - 0.55 * red) + C_RED * 0.55 * red
        col = C_CORE * (1 - c1) + C_ICE * c1
        col = col * (1 - c2) + gold * c2
        fl = 1 + 0.35 * np.sin(0.7 * t + self.fl)
        e = self.E * fl * (1.9 - 1.1 * d) * 0.55 * pw
        e *= 1.0 + 0.6 * m          # ring is thinner: keep it bright
        ctx.fr.splat(P0, P1, 0.004, e, col, ctx.cam0, ctx.cam1)
        # --- core
        cj = self.cd * 0.2 + 0.01 * np.sin(np.array([31.0, 47.0, 53.0]) * t)
        if m < 1:
            Pc0 = C0 + self.warp(cj, crown_morph(ctx.t0), R0, ctx.t0)
            Pc1 = C1 + self.warp(cj, crown_morph(ctx.t1), R1, ctx.t1)
            ec = np.full(self.k, 1.8 * pw * (1 - 0.5 * m))
            ctx.fr.splat(Pc0, Pc1, 0.003, ec, C_CORE, ctx.cam0, ctx.cam1)
        # --- tongues: rise from the upper surface, cool to gold/orange
        kk0 = (self.tph + self.tv * (ctx.t0 - IGN)) % 1.0
        kk1 = (self.tph + self.tv * (ctx.t1 - IGN)) % 1.0
        wrap_ok = kk1 >= kk0

        def tongue(kk, tq):
            base = self.td * (0.9 + 0.12 * kk[:, None])
            p = base + np.array([0, 1.0, 0]) * ((0.95 + 1.6 * crown_morph(tq)) * kk ** 1.6)[:, None]
            w = vnoise(p + np.array([0, -0.05 * (tq - IGN), 0]), 2.2, (0, 0, 0), 2)
            return p + w * (0.08 + 0.2 * kk)[:, None]
        T0 = C0 + self.warp(tongue(kk0, ctx.t0), crown_morph(ctx.t0), R0, ctx.t0)
        T1 = C1 + self.warp(tongue(kk1, ctx.t1), crown_morph(ctx.t1), R1, ctx.t1)
        et = self.tE * (1 - kk1) ** 2 * smoothstep(0.0, 0.08, kk1) * 2.6 * pw * wrap_ok
        tcol = look.blackbody(0.88 - 0.4 * kk1)
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
        ef = (base + 10.0 * pulse) * vis * 1.3 * pw
        fcol = C_ICE * (1 - np.minimum(pulse, 1))[:, None] + C_CORE * np.minimum(pulse, 1)[:, None]
        F0 = C0 + self.warp(self.fp, crown_morph(ctx.t0), R0, ctx.t0)
        F1 = C1 + self.warp(self.fp, crown_morph(ctx.t1), R1, ctx.t1)
        ctx.fr.splat(F0, F1, 0.002, ef, fcol, ctx.cam0, ctx.cam1)
        # --- glow (volumetric halo around the fire)
        H = np.array([C1, C1, C1 + [0, 0.4, 0]])
        rr = np.array([1.2, 3.0, 7.0]) * (1 + 1.5 * m)
        eh = np.array([500.0, 500.0, 420.0]) * pw * smoothstep(IGN, IGN + 6, t)
        hc = np.array([C_CORE, C_ICE * 0.6 + C_GOLD * 0.4, C_GOLD * (1 - red) + C_RED * red])
        ctx.fr.splat(H, H, rr, eh, hc, ctx.cam0, ctx.cam1, profile=1)


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

    def pts(self, t):
        m = crown_morph(t)
        C = crown_centre(t)
        Rr = 3.2 + 0.6 * smoothstep(640, 800, t)
        rot = m * 0.035 * (t - 562)
        k = (self.ph + self.sp * t) % 1.0          # particles stream up each tine
        a = 2 * np.pi * self.tine / self.nt + rot
        h = (2.1 + 0.3 * np.sin(0.13 * t + self.tine)) * m
        y = k * h
        w = 0.5 * (1 - k) ** 0.9 + 0.03
        x = (Rr + self.v[:, 0] * w * 0.5)
        z = self.v[:, 1] * w
        P = np.stack([x * np.cos(a) - z * np.sin(a), y + 0.25, x * np.sin(a) + z * np.cos(a)], 1)
        return C + P, k

    def emit(self, ctx):
        t = ctx.t
        m = crown_morph(t)
        if m <= 0.02 or t >= 960:
            return
        P0, _ = self.pts(ctx.t0)
        P1, k = self.pts(ctx.t1)
        red = redness(t)
        e = self.E * (1 - k) ** 1.2 * 3.4 * smoothstep(0.35, 0.9, m) * fire_power(t)
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

class Towers:
    def __init__(self, seed=55):
        r = rng(seed)
        self.T = TW.build_all()
        k = len(self.T)
        self.k = k
        self.ang = 2 * np.pi * np.arange(k) / k + 0.35
        self.rad = 18.0 + r.uniform(-1.5, 1.5, k)
        self.t_rise = 520 + np.array([0, 9, 4, 14, 6, 11, 2], float)
        self.h_rise = np.array([19.0, 16.0, 21.0, 18.0, 22.0, 17.0, 20.0])
        # surge amounts per beat (leap-frogging race)
        J = r.uniform(1.4, 2.6, (k, len(BEATS)))
        lead = r.integers(0, k, len(BEATS))
        for b, l in enumerate(lead):
            J[l, b] += r.uniform(1.5, 2.5)
        self.J = J
        self.dly = r.uniform(0, 2.5, k)
        # per-point attributes
        self.rnd = [r.random(len(t['p'])) for t in self.T]
        self.win_on = [r.random(len(t['p'])) < 0.8 for t in self.T]
        self.flk = [r.uniform(0, 2 * np.pi, len(t['p'])) for t in self.T]
        # facing: rotate each tower so its local +x faces the centre
        self.rot = [(-a + np.pi) for a in self.ang]

    def height(self, i, t):
        h = self.h_rise[i] * float(ease_out((t - self.t_rise[i]) / 70.0, 3.0))
        for b, tb in enumerate(BEATS):
            x = (t - tb - self.dly[i]) / 6.0
            if x > 0:
                h += self.J[i, b] * float(ease_out_back(x, 1.3))
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
            e_base = np.where(kind == 1, 1.0, 0.45) * (0.5 + 0.5 * rnd)
            e_base *= 0.35 + 0.65 * smoothstep(0.0, 14.0, yl)
            fl = 1 + 0.25 * np.sin(0.45 * t + self.flk[i][vis])
            Tb = 0.4 + 0.14 * rnd
            cb = look.blackbody(Tb)
            cb = cb * (1 - 0.6 * red) + (C_RED * 0.7 + C_CRIMSON * 0.3) * 0.6 * red
            e_base = e_base * fl * 4.5
            # fire light (inner faces)
            L = light_pos - P1
            dL = np.linalg.norm(L, axis=1)
            lam = np.maximum((nw * L).sum(1) / np.maximum(dL, 1e-6), 0.0)
            lam = np.where(kind == 1, 0.35 + 0.65 * lam, lam)
            e_lit = light_pow * lam / (1 + (dL / 14.0) ** 2) * 1.6
            # emergence front: hot line where the tower leaves the ground
            front = np.exp(-yl / 0.6) * 5.0 * (1 - smoothstep(610, 650, t) * 0.7)
            # windows
            win = (kind == 2) & self.win_on[i][vis]
            e_win = np.where(win, 9.0 * (0.7 + 0.3 * np.sin(0.2 * t + 7 * rnd)), 0.0)
            wcol = look.blackbody(0.72 - 0.15 * red)
            colE = (cb * e_base[:, None] + light_col[None, :] * e_lit[:, None] +
                    look.blackbody(0.8)[None, :] * front[:, None] + wcol[None, :] * e_win[:, None])
            E = np.ones(len(p))
            E[kind == 2] = np.where(win[kind == 2], 1.0, 0.0) + 1e-3
            ctx.fr.splat(P0, P1, 0.012, E, colE, ctx.cam0, ctx.cam1)


class Sparks:
    """Spark sprays at every surge (and at emergence)."""

    def __init__(self, towers, seed=66):
        r = rng(seed)
        self.tw = towers
        rows = []
        per = 420
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
        x = tau[idx] / self.life[idx]
        e = self.E[idx] * 26.0 * (1 - x) ** 1.3
        red = redness(t)
        col = look.blackbody(np.clip(0.95 - 0.6 * x, 0.2, 1.0))
        col = col * (1 - 0.3 * red) + C_RED * 0.3 * red
        ctx.fr.splat(P0[idx], P1[idx], 0.005, e, col, ctx.cam0, ctx.cam1)


class Walls:
    """Radial walls of rising red embers dividing the kingdoms (from 680)."""

    def __init__(self, towers, seed=77):
        r = rng(seed)
        k = towers.k
        per = 16000
        N = k * per
        self.N = N
        self.w = np.repeat(np.arange(k), per)
        self.ang = towers.ang + np.pi / k
        self.u = r.random(N)
        self.ph = r.random(N)
        self.v = r.uniform(0.004, 0.012, N)
        self.off = r.normal(0, 1, N)
        self.E = r.lognormal(0, 0.5, N)
        self.start = 676 + 14 * r.random(k)

    def height(self, t, w):
        base = 6.0 * ease_out((t - self.start[w]) / 40.0, 2.0)
        grow = 0.0
        for b, tb in enumerate(BEATS):
            if tb >= 680:
                x = (t - tb - 3.0) / 8.0
                if x > 0:
                    grow += 1.9 * float(ease_out_back(x, 1.0))
        return base + grow

    def pts(self, t):
        a = self.ang[self.w]
        rr = 4.0 + 19.0 * self.u
        H = np.array([self.height(t, i) for i in range(len(self.ang))])[self.w]
        k = (self.ph + self.v * t) % 1.0
        y = GROUND + k * H
        lat = self.off * 0.18 + 0.5 * np.sin(0.35 * rr + 0.08 * t + self.w)
        P = np.stack([rr * np.cos(a) - lat * np.sin(a), y, rr * np.sin(a) + lat * np.cos(a)], 1)
        wv = vnoise(P * 0.5 + np.array([0, -0.02 * t, 0]), 0.35, (0, 0, 0), 1)
        return P + wv * np.array([0.6, 0.3, 0.6]), k, H

    def emit(self, ctx):
        t = ctx.t
        if t < 676 or t >= 1040:
            return
        P0, _, _ = self.pts(ctx.t0)
        P1, k, H = self.pts(ctx.t1)
        on = t >= self.start[self.w]
        crest = np.exp(-((1 - k) / 0.04) ** 2) * 2.0
        e = self.E * (0.35 + 0.65 * (1 - k) ** 0.6 + crest) * 3.0 * on
        e *= smoothstep(0, 10, t - self.start[self.w])
        col = C_RED * (1 - k)[:, None] + C_CRIMSON * k[:, None]
        ctx.fr.splat(P0, P1, 0.006, e, col, ctx.cam0, ctx.cam1)


class Smoke:
    """Faint smoke lit from within by the fire (and red in the race)."""

    def __init__(self, seed=88):
        r = rng(seed)
        n = 5000
        self.n = n
        a = r.uniform(0, 2 * np.pi, n)
        rr = 2.0 + 34.0 * r.random(n) ** 0.7
        self.p = np.stack([rr * np.cos(a), r.uniform(GROUND, 30.0, n), rr * np.sin(a)], 1)
        self.rw = r.uniform(3.0, 7.0, n)
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
        lit = light_pow * 2.2 / (1 + (d / 5.0) ** 2) ** 1.5
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
    """800-960: the crown balloons into a vast unstable vortex."""

    def __init__(self, seed=111):
        r = rng(seed)
        n = 260000
        self.n = n
        self.rf = 3.5 + 75.0 * r.random(n) ** 1.3
        narm = 4
        self.arm = r.integers(0, narm, n) * 2 * np.pi / narm + r.normal(0, 0.22, n)
        self.h = r.normal(0, 1, n)
        self.E = r.lognormal(0, 0.6, n)
        self.ph = r.uniform(0, 2 * np.pi, n)
        self.jit = r.normal(0, 1, (n, 3))

    def grow(self, t):
        return float(smootherstep(796, 872, t))

    def pts(self, t, C):
        g = self.grow(t)
        rr = 3.5 + (self.rf - 3.5) * g ** 0.8
        om = 0.9 * (rr / 10.0) ** -1.1
        th = self.arm - 1.2 * np.log(rr / 10.0) + om * (t - 800) * 0.12
        funnel = -9.0 * g * np.exp(-rr / 14.0)
        thick = (0.4 + 0.06 * rr) * self.h
        P = np.stack([rr * np.cos(th), funnel + thick, rr * np.sin(th)], 1)
        P += self.jit * (0.3 + 0.03 * rr)[:, None]
        return C + P, rr

    def emit(self, ctx):
        t = ctx.t
        g = self.grow(t)
        if g <= 0.001 or t >= 1040:
            return
        C0, C1 = crown_centre(ctx.t0), crown_centre(ctx.t1)
        P0, _ = self.pts(ctx.t0, C0)
        P1, rr = self.pts(ctx.t1, C1)
        x = clamp01((rr - 3.5) / 70.0)
        col = C_CORE * (1 - smoothstep(0.0, 0.08, x))[:, None]
        col += C_GOLD * (smoothstep(0.0, 0.08, x) * (1 - smoothstep(0.08, 0.3, x)))[:, None]
        col += look.blackbody(0.55) * (smoothstep(0.08, 0.3, x) * (1 - smoothstep(0.3, 0.6, x)))[:, None]
        col += C_RED * (smoothstep(0.3, 0.6, x) * (1 - smoothstep(0.6, 1.0, x)))[:, None]
        col += C_CRIMSON * smoothstep(0.6, 1.0, x)[:, None]
        # instability: travelling brightness waves + flicker
        wave = 0.6 + 0.4 * np.sin(0.5 * rr - 0.45 * t + self.ph * 0.3)
        e = self.E * wave * (3.0 + 10.0 * np.exp(-rr / 8.0)) * 1.6 * g
        ctx.fr.splat(P0, P1, 0.02, e, col, ctx.cam0, ctx.cam1)


# ------------------------------------------------------------- camera ---

TOWER_ANG0 = 0.35
ALPHA_C = TOWER_ANG0 + np.pi / 7 + 0.13          # camera azimuth: in a gap, off the wall line


def _polar(r, a, y):
    return np.array([r * math.cos(a), y, r * math.sin(a)])


CAM_B = [  # (frame, radius, azimuth offset, height, target y)
    (480, 13.9, 0.43, 7.2, -1.35),
    (500, 15.5, 0.16, 3.4, -1.2),
    (520, 16.5, -0.18, 3.6, -1.0),
    (556, 21.0, -0.12, 6.0, 1.8),
    (596, 27.0, -0.04, 12.5, 6.0),
    (640, 30.0, 0.00, 16.5, 8.6),
    (720, 31.0, 0.07, 25.0, 17.0),
    (800, 30.0, 0.14, 42.0, 31.0),
    (840, 54.0, 0.24, 64.0, 39.0),
    (880, 86.0, 0.30, 88.0, 43.0),
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
            amp = 0.22 * math.exp(-x / 3.5) * (0.5 + 0.5 * smoothstep(640, 800, t))
            sh += amp * np.array([math.sin(2.1 * x + tb), math.sin(2.9 * x + 0.5 * tb), 0.0])
    return pos + sh * 0.5, tgt + sh
