"""A-FIX (28 Sep): THE RACE made legible (film A only; hooked from a3.TimelineA3.emit).

The towers are forges, so their throats burn. Every forge roars at its crown and throws a burst of sparks on each
beat of the race (72 BPM: every 20 frames), and the smoke off it is lit from below. From bar 22 b3.5 the two giants'
throats roar hardest as they outgrow the rest. The stranger test: "gold-spotted pillars" becomes "furnaces racing".
Pure numpy, modelled on scene_b.TowerSmoke (same splat call, same kingdom rotation).
"""
import math

import numpy as np

import look
import scene_b as B
from core import rng, smoothstep, vnoise

BEAT = 20.0


class ThroatFire:
    def __init__(self, towers, t0, t1, giants=(), seed=4411, per=300, spk=90):
        r = rng(seed)
        self.tw = towers
        self.k = towers.k
        self.t0, self.t1 = float(t0), float(t1)
        self.giants = set(giants)
        k = self.k
        n = k * per
        self.fi = np.repeat(np.arange(k), per)                  # flame particles: short, rising, cooling
        self.flife = r.uniform(6.0, 15.0, n)
        self.fph = r.random(n)
        self.foff = r.normal(0, 1, (n, 3)) * np.array([1.0, 0.25, 1.0])
        self.fv = r.uniform(0.35, 0.8, n)
        self.fE = r.lognormal(0, 0.35, n)
        m = k * spk                                             # beat sparks: thrown up and out, falling
        self.si = np.repeat(np.arange(k), spk)
        self.sv = r.normal(0, 1, (m, 3)) * np.array([0.35, 0.0, 0.35]) + np.array([0.0, 1.0, 0.0]) * r.uniform(0.6, 1.3, m)[:, None]
        self.slife = r.uniform(8.0, 18.0, m)
        self.sE = r.lognormal(0, 0.5, m)
        self.hw = np.array([2.0, 3.5, 1.5, 2.2, 1.6, 2.2, 2.4, 2.2])[:k] if k <= 8 else np.full(k, 2.0)

    def _tops(self, t):
        return np.array([self.tw.top(i, t) for i in range(self.k)])

    def emit(self, ctx):
        t = ctx.t
        if not (self.t0 <= t < self.t1):
            return
        on = float(smoothstep(self.t0, self.t0 + 40.0, t))
        race = B.SCHED.race(t) if B.SCHED is not None else 1.0
        gro = np.array([1.0 + (0.8 * float(smoothstep(1730.0, 1780.0, t)) if i in self.giants else 0.0)
                        for i in range(self.k)])
        tops = self._tops(t)
        beat_age = t % BEAT                                     # on the beat (the grid: beats at multiples of 20)
        surge = 1.0 + 0.9 * math.exp(-beat_age / 4.0)          # every beat the throats roar up
        cam = ctx.cam
        fpx = cam.f_px(1920)
        P_all, R_all, E_all, C_all, V_all = [], [], [], [], []
        # --- the flames in the throats
        age = ((t / self.flife + self.fph) % 1.0) * self.flife
        u = age / self.flife
        g = gro[self.fi]
        top = tops[self.fi]
        hw = self.hw[self.fi] * g
        P = top + np.stack([self.foff[:, 0] * hw * 0.28 * (1 - 0.5 * u),
                            -0.3 + self.fv * age * surge * g * (0.8 + 0.6 * race),
                            self.foff[:, 2] * hw * 0.28 * (1 - 0.5 * u)], 1)
        P = P + vnoise(P * 0.35 + np.array([0.0, -0.08 * t, 0.0]), 0.25, (0, 0, 0), 1) * (0.2 + 1.2 * u)[:, None]
        rad = (0.28 + 0.45 * u) * (0.7 + 0.3 * g)
        temp = 0.78 - 0.45 * u                                  # forge-orange at the throat, red above (the thinking
                                                                # fire stays the only ice-white thing in the frame)
        col = np.array([look.blackbody(float(x)) for x in np.round(temp, 2)])
        L = (1.0 * self.fE * (1 - u) ** 1.4 * smoothstep(0.0, 0.12, u) * on * g * (0.7 + 0.3 * surge))
        P_all.append(P); R_all.append(rad); E_all.append(L); C_all.append(col); V_all.append(np.zeros_like(P))
        # --- the sparks thrown on the beat
        sage = beat_age + (t // BEAT) * 0.0
        if sage < 18.0:
            v = self.sv * (0.55 + 0.45 * gro[self.si])[:, None] * (0.8 + 0.5 * race)
            Ps = tops[self.si] + v * sage + np.array([0.0, -0.012, 0.0]) * sage * sage
            us = np.clip(sage / self.slife, 0.0, 1.0)
            Ls = 2.0 * self.sE * (1 - us) ** 1.5 * (us < 1.0) * on * gro[self.si]
            cs = np.array([look.blackbody(0.8)] * len(Ps))
            P_all.append(Ps); R_all.append(np.full(len(Ps), 0.18)); E_all.append(Ls); C_all.append(cs)
            V_all.append(v * 0.9)
        P = np.concatenate(P_all)
        V = np.concatenate(V_all)
        rad = np.concatenate(R_all)
        L = np.concatenate(E_all)
        col = np.concatenate(C_all)
        P0 = B._krot(P - 0.5 * V, t)
        P1 = B._krot(P + 0.5 * V, t)
        z = np.maximum((0.5 * (P0 + P1) - cam.pos) @ cam.R[2], 1.0)
        rpx = rad * fpx / z
        E = L * np.pi * rpx ** 2
        m = E > 1e-6
        if not m.any():
            return
        ctx.fr.splat(P0[m], P1[m], rad[m], E[m], col[m] / np.maximum(col[m].max(1), 1e-9)[:, None], ctx.cam0,
                     ctx.cam1, profile=1, zref=0.0, rmax=300.0)
