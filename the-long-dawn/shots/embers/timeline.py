"""Timeline: which elements are alive when, the camera, and per-frame finish settings."""
import math

import numpy as np

import look
from core import Camera, smoothstep, lerp
import scene_a as A
import scene_b as B


class Timeline:
    def __init__(self):
        self.embers = A.Embers()
        self.glow = A.TorchGlow()
        self.glyphs = A.Glyphs()
        self.point = A.ThePoint(self.glyphs)
        self.fire = B.MindFire()
        self.crown = B.Crown()
        self.shock = B.Shockwave()
        self.towers = B.Towers()
        self.sparks = B.Sparks(self.towers)
        self.walls = B.Walls(self.towers)
        self.smoke = B.Smoke()
        self.dust = B.Dust()
        self.vortex = B.Vortex()

    # ---------------------------------------------------------------- camera
    def camera(self, t):
        if t < 480:
            pos, tgt = A.cam_a(t)
            # rack: the ember column (~8) while embers become letters, then the passing heroes
            focus = lerp(lerp(8.0, A.HERO_FOCUS, float(smoothstep(338, 350, t))), np.linalg.norm(pos),
                         float(smoothstep(398, 432, t)))
            ap = lerp(0.065, 0.04, float(smoothstep(392, 440, t)))
            return Camera(pos, tgt, hfov=A.HFOV_A(t), focus=focus, aperture=ap)
        if t < 880:
            pos, tgt = B.cam_b(t)
            C = B.crown_centre(t)
            focus = float(np.linalg.norm(C - pos))
            hf = float(lerp(46.0, 44.0, smoothstep(480, 520, t)))
            hf = float(lerp(hf, 56.0, smoothstep(540, 620, t)))
            hf = float(lerp(hf, 60.0, smoothstep(800, 870, t)))
            ap = float(lerp(0.06, 0.12, smoothstep(520, 600, t)))
            return Camera(pos, tgt, hfov=hf, focus=focus, aperture=ap)
        return Camera((0, 0, 10), (0, 0, 0))

    def render_opts(self, f):
        if f < 480:
            fs = float(lerp(16.0, 45.0, smoothstep(405, 430, f)))
            return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=fs, fog_len=22.0, near=0.45)
        if f < 880:
            return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=45.0, fog_len=70.0, near=0.3)
        return dict(bokeh_pow=0.3, bokeh_cap=2.0)

    def light(self, t):
        red = B.redness(t)
        col = (look.hexrgb(look.PALETTE['mind_gold']) * 0.6 + look.hexrgb(look.PALETTE['mind_ice']) * 0.4)
        col = col * (1 - 0.5 * red) + look.hexrgb(look.PALETTE['race_red']) * 0.5 * red
        return B.crown_centre(t), col, 60.0 * B.fire_power(t)

    def emit(self, ctx):
        t = ctx.t
        if t < 481:
            self.glow.emit(ctx)
            self.embers.emit(ctx)
            self.glyphs.emit(ctx)
            self.point.emit(ctx)
        if 480 <= t < 880:
            lp, lc, lpw = self.light(t)
            self.dust.emit(ctx)
            self.smoke.emit(ctx, lp, lc, lpw)
            self.towers.emit(ctx, lp, lc, lpw)
            self.walls.emit(ctx)
            self.sparks.emit(ctx)
            self.vortex.emit(ctx)
            self.fire.emit(ctx)
            self.crown.emit(ctx)
            self.shock.emit(ctx)

    def post(self, ctx, hdr):
        f = ctx.t
        if f < 336:
            # warm light of the torch flame just below frame (continuity with INTRO)
            k = float(1 - smoothstep(300, 336, f))
            H, W = hdr.shape[:2]
            y, x = np.mgrid[0:H, 0:W].astype(np.float32)
            x = (x - W * 0.5) / (W * 0.30)
            y = (y - H * 1.08) / (H * 0.42)
            g = np.exp(-(x * x + y * y))[..., None]
            hdr += g * np.array([1.0, 0.42, 0.10], np.float32) * (0.9 * k)
        return hdr

    def finish_opts(self, f):
        streak = 0.0
        if 455 <= f < 500:
            streak = 0.05 * float(smoothstep(455, 478, f)) * (1 - float(smoothstep(484, 500, f)))
        bloom = 0.12
        if f >= 480:
            bloom = 0.15
        return dict(exposure=1.0, bloom_strength=bloom, bloom_threshold=0.7,
                    streak_strength=streak, vignette_amount=0.25)
