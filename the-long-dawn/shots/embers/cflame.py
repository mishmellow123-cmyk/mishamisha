"""EMBERS-C: C's fire -- ONE natural flame, gold and calm (E15's catch, C5's fire alone, E5-C and E11).

Flame-local units: the root at the origin, +Y up, height 1; the caller places and scales it (`Flame.emit(ctx, C,
scale, ...)`). Stateless: every frame is a pure function of t, so any frame renders alone.

Each particle is gas born on the root disc that rises for one breath of the flame: buoyant (it quickens as it
climbs), drawn in toward the axis (the flame necks to ONE tip), wrinkled by a turbulent field that travels up the
flame (faint at the root, stronger toward the tip, where small flamelets break off and die), and bent by a slow
draught. Its heat falls with age and away from the axis: a white-gold root, a gold body, deep orange at the rim
and the tip, red where the flamelets die. A soft body glow, and a few calm sparks off the tip. Never a fork.

Brightness is set as SURFACE brightness (each particle's energy follows its share of the flame's projected area),
so the flame looks the same whether it is 60 px or 600 px tall.
"""
import math

import numpy as np

from core import rng, vnoise, snoise, smoothstep

C_ROOT = np.array([1.0, 0.93, 0.76])
C_BODY = np.array([1.0, 0.75, 0.33])
C_RIM = np.array([1.0, 0.47, 0.12])
C_DEEP = np.array([0.86, 0.21, 0.045])

RB = 0.16            # root radius (flame heights)
RISE = 30.0          # frames for gas to climb from the root to the tip (calm)


def heat_col(h):
    """0..1 heat -> colour: deep red -> orange rim -> gold body -> white-gold root"""
    h = np.clip(h, 0.0, 1.0)[:, None]
    a = np.clip(h / 0.3, 0, 1)
    b = np.clip((h - 0.3) / 0.35, 0, 1)
    c = np.clip((h - 0.65) / 0.35, 0, 1)
    col = C_DEEP + (C_RIM - C_DEEP) * a
    col = col + (C_BODY - C_RIM) * b
    col = col + (C_ROOT - C_BODY) * c
    return col


class Flame:
    def __init__(self, seed=801, n=90000, rise=RISE, wide=1.0):
        r = rng(seed)
        self.n = n
        self.rise = rise
        self.wide = wide
        rr = np.sqrt(r.random(n))
        self.r0 = rr
        self.a0 = r.uniform(0, 2 * np.pi, n)
        self.ph = r.random(n)
        self.sp = r.uniform(0.82, 1.22, n)
        # gas near the axis reaches the tip; the rim burns out lower (so the flame tapers to one point)
        self.reach = (1.0 - 0.45 * rr ** 2) * r.uniform(0.88, 1.05, n)
        self.E = r.lognormal(0, 0.3, n)
        self.sz = r.uniform(0.75, 1.3, n)
        self.flk = r.random(n)
        # calm sparks off the tip
        ns = 420
        self.s_ph = r.random(ns)
        self.s_v = r.uniform(0.006, 0.011, ns)
        self.s_a = r.uniform(0, 2 * np.pi, ns)
        self.s_d = r.normal(0, 1, (ns, 2))
        self.s_E = r.lognormal(0, 0.6, ns)

    # ------------------------------------------------------------------ gas
    def lean(self, t):
        """the draught: a slow lean of the upper flame (flame units per unit height^2)"""
        return np.array([0.06 * math.sin(0.021 * t + 0.3) + 0.028 * math.sin(0.057 * t + 1.7), 0.0,
                         0.045 * math.sin(0.017 * t + 2.2) + 0.02 * math.sin(0.049 * t + 0.4)])

    def gas(self, t, calm=1.0):
        """positions (flame units), cycle position u, heat, emission weight"""
        u = (t * self.sp / self.rise + self.ph) % 1.0
        y = self.reach * (0.2 * u + 0.8 * u ** 1.45)
        g = (1.0 + 0.62 * np.sin(0.5 * np.pi * np.clip(u / 0.3, 0, 1))) * (1.0 - u) ** 0.66
        rho = RB * self.wide * self.r0 * g
        tw = self.a0 + 0.35 * u
        P = np.stack([rho * np.cos(tw), y, rho * np.sin(tw)], 1)
        # turbulence travelling up the flame: wrinkles climb it; stronger toward the tip
        amp = (0.016 + (0.075 + 0.05 * (1 - calm)) * y ** 1.7)
        Q = P * np.array([1.0, 0.8, 1.0]) + np.array([0.0, -0.045 * t, 0.013 * t])
        D = vnoise(Q, 2.3, (1.7, 3.1, 0.2), 2)
        P = P + D * amp[:, None] * np.array([1.0, 0.45, 1.0])
        # flamelets: near the tip a few tear off and rise on their own
        tear = (self.flk > 0.72) & (u > 0.62)
        if tear.any():
            k = np.clip((u - 0.62) / 0.38, 0, 1)
            P[tear, 1] += 0.18 * k[tear] ** 1.5
            P[tear, 0] += 0.05 * np.sin(9.0 * self.flk[tear] + 0.11 * t) * k[tear]
        P = P + self.lean(t)[None, :] * (y ** 2)[:, None]
        heat = (1.0 - u) ** 1.1 * (1.0 - 0.5 * self.r0 ** 2)
        heat = np.where(tear, heat * 0.6, heat)
        w = self.E * smoothstep(0.0, 0.05, u) * (1.0 - smoothstep(0.78, 1.0, u))
        w = np.where(tear, w * 0.7, w)
        return P, u, heat, w

    # ----------------------------------------------------------------- emit
    def emit(self, ctx, C, scale, bright=1.0, height=1.0, calm=1.0, fr=None, glow=1.0, sparks=1.0, Rot=None,
             zref=0.0, myid=-1):
        """splat the flame rooted at world point C, `scale` world units tall (x height, 0..1, as it catches)"""
        if height <= 0.002 or bright <= 0:
            return
        fr = fr if fr is not None else ctx.fr
        cam = ctx.cam
        f = cam.f_px(1920)
        z = max(float((np.asarray(C) - cam.pos) @ cam.R[2]), 0.05)
        px = f * scale * height / z                         # the flame's height on screen, px
        n = int(np.clip(px * 150, 6000, self.n))
        sl = slice(0, n)
        Rm = np.eye(3) if Rot is None else Rot

        def world(t):
            P, u, heat, w = self.gas(t, calm)
            P = P[sl]
            P = P * np.array([height ** 0.85, height, height ** 0.85])
            return np.asarray(C) + (P * scale) @ Rm.T, u[sl], heat[sl], w[sl]
        P0, u0, _, _ = world(ctx.t0)
        P1, u1, heat, w = world(ctx.t1)
        ok = u1 >= u0                                      # skip gas reborn inside the shutter
        # surface brightness: each particle carries its share of the flame's projected area
        area = 0.42 * (scale * height) ** 2 * (f / z) ** 2 / n
        E = bright * area * w * (0.2 + 0.8 * heat ** 0.85) * 1.6 * ok
        col = heat_col(heat * (0.85 + 0.15 * height))
        rw = 0.028 * scale * max(height, 0.2) * self.sz[sl]
        fr.splat(P0, P1, rw, E, col, ctx.cam0, ctx.cam1, profile=1, zref=zref, myid=myid)
        # the body's glow (a soft light round it)
        if glow > 0:
            Cg = np.asarray(C) + (np.array([0.0, 0.33 * height, 0.0]) * scale) @ Rm.T
            H = np.array([Cg, Cg])
            rg = np.array([0.34, 0.9]) * scale * max(height, 0.25)
            eg = np.array([2.6, 1.3]) * glow * bright * (f * scale * max(height, 0.25) / z) ** 2 * 0.02
            fr.splat(H, H, rg, eg, np.array([[1.0, 0.72, 0.34], [1.0, 0.55, 0.2]]), ctx.cam0, ctx.cam1, profile=1,
                     zref=zref, myid=myid)
        # a few calm sparks off the tip
        if sparks > 0 and height > 0.5:
            def spk(t):
                k = (self.s_ph + self.s_v * t) % 1.0
                tip = np.array([0.0, 0.92 * height, 0.0]) + self.lean(t) * height ** 2
                y = tip[1] + k * 1.6
                spr = 0.03 + 0.28 * k
                x = tip[0] + self.s_d[:, 0] * spr + 0.05 * np.sin(6.0 * k + self.s_a)
                zz = tip[2] + self.s_d[:, 1] * spr
                Ps = np.stack([x, y, zz], 1)
                Ps = Ps + vnoise(Ps + np.array([0, -0.02 * t, 0]), 1.3, (5.0, 1.0, 2.0), 1) * (0.1 * k)[:, None]
                return np.asarray(C) + (Ps * scale) @ Rm.T, k
            S0, k0 = spk(ctx.t0)
            S1, k1 = spk(ctx.t1)
            okk = k1 >= k0
            es = self.s_E * (1 - k1) ** 1.6 * 0.9 * sparks * bright * okk * smoothstep(0.0, 0.06, k1)
            cs = heat_col(0.75 - 0.6 * k1)
            fr.splat(S0, S1, 0.0035 * scale, es, cs, ctx.cam0, ctx.cam1, zref=zref, myid=myid)

    def light(self, C, scale, bright=1.0, height=1.0):
        """(position, colour, power) of the flame as a light for the towers"""
        return (np.asarray(C) + np.array([0.0, 0.35 * scale * height, 0.0]), np.array([1.0, 0.76, 0.4]),
                bright * height)
