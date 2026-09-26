"""Timeline: which elements are alive when, the camera, and per-frame finish settings."""
import math

import numpy as np

import look
from core import Camera, smoothstep, lerp
import scene_a as A
import scene_b as B
import scene_c as C


class Timeline:
    def __init__(self):
        self._cache = {}
        self.ember = C.LastEmber()

    def _get(self, name, make):
        if name not in self._cache:
            self._cache[name] = make()
        return self._cache[name]

    # lazily built elements (each render worker only builds what its frame range needs)
    embers = property(lambda s: s._get('embers', A.Embers))
    glow = property(lambda s: s._get('glow', A.TorchGlow))
    glyphs = property(lambda s: s._get('glyphs', A.Glyphs))
    point = property(lambda s: s._get('point', lambda: A.ThePoint(s.glyphs)))
    fire = property(lambda s: s._get('fire', B.MindFire))
    crown = property(lambda s: s._get('crown', B.Crown))
    fsparks = property(lambda s: s._get('fsparks', B.FireSparks))
    shock = property(lambda s: s._get('shock', B.Shockwave))
    towers = property(lambda s: s._get('towers', B.Towers))
    sparks = property(lambda s: s._get('sparks', lambda: B.Sparks(s.towers)))
    tembers = property(lambda s: s._get('tembers', lambda: B.TowerEmbers(s.towers)))
    tsmoke = property(lambda s: s._get('tsmoke', lambda: B.TowerSmoke(s.towers)))
    walls = property(lambda s: s._get('walls', lambda: B.Walls(s.towers)))
    smoke = property(lambda s: s._get('smoke', B.Smoke))
    dust = property(lambda s: s._get('dust', B.Dust))
    vortex = property(lambda s: s._get('vortex', lambda: B.Vortex(s.fire)))
    globe = property(lambda s: s._get('globe', C.Globe))
    ring = property(lambda s: s._get('ring', lambda: __import__('tolkien').Ring()))
    eye = property(lambda s: s._get('eye', lambda: __import__('tolkien').Eye()))
    hand = property(lambda s: s._get('hand', C.Hand))

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
            Cc = B.crown_centre(t)
            focus = float(np.linalg.norm(Cc - pos))
            hf = float(lerp(46.0, 52.0, smoothstep(480, 505, t)))    # v2: wider while it is a flame (v1: 44)
            hf = float(lerp(hf, 56.0, smoothstep(540, 620, t)))
            hf = float(lerp(hf, 66.0, smoothstep(640, 700, t)))
            hf = float(lerp(hf, 78.0, smoothstep(800, 850, t)))
            ap = float(lerp(0.06, 0.12, smoothstep(520, 600, t)))
            return Camera(pos, tgt, hfov=hf, focus=focus, aperture=ap)
        if t < 960:
            pos, tgt = C.cam_globe(t)
            return Camera(pos, tgt, hfov=40.0, focus=float(np.linalg.norm(pos)) - 0.8, aperture=0.004)
        if t < 1040:
            pos, tgt = C.cam_grasp(t)
            Cc = B.crown_centre(960.0)
            return Camera(pos, tgt, hfov=62.0, focus=float(np.linalg.norm(Cc - pos)), aperture=0.25)
        pos, tgt = C.cam_silence(t)
        return Camera(pos, tgt, hfov=50.0, focus=6.0, aperture=0.03)

    def render_opts(self, f):
        if f < 480:
            fs = float(lerp(16.0, 45.0, smoothstep(405, 430, f)))
            return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=fs, fog_len=22.0, near=0.45)
        if f < 880:
            return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=45.0, fog_len=70.0, near=0.3)
        if f < 960:
            return dict(bokeh_pow=0.2, bokeh_cap=1.5, zref=9.0, near=0.05)
        if f < 1040:
            return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=45.0, fog_len=70.0, near=1.0)
        return dict(bokeh_pow=0.0, bokeh_cap=1.0, near=0.05)

    def light(self, t):
        red = B.redness(t)
        col = (look.hexrgb(look.PALETTE['mind_gold']) * 0.8 + look.hexrgb(look.PALETTE['mind_ice']) * 0.2)
        col = col * (1 - 0.8 * red) + look.hexrgb(look.PALETTE['race_red']) * 0.8 * red
        return B.crown_centre(t), col, 60.0 * B.fire_power(t)

    def emit(self, ctx):
        t = ctx.t
        if t < 481:
            self.glow.emit(ctx)
            self.embers.emit(ctx)
            self.glyphs.emit(ctx)
            self.point.emit(ctx)
        if 480 <= t < 880 or 960 <= t < 1040:
            lp, lc, lpw = self.light(t)
            self.towers.prepare(ctx)        # v2: solid towers -> occluder for everything splatted after this
            self.dust.emit(ctx)
            self.smoke.emit(ctx, lp, lc, lpw)
            self.towers.emit(ctx, lp, lc, lpw)
            self.tembers.emit(ctx)
            self.tsmoke.emit(ctx, lp, lc, lpw)
            self.walls.emit(ctx)
            self.sparks.emit(ctx)
            self.vortex.emit(ctx)
            self.fire.emit(ctx)
            self.crown.emit(ctx)
            self.fsparks.emit(ctx)
            self.shock.emit(ctx)
            import variant
            if variant.tolkien():                      # cut C: the Ring (race + storm, and in the grasp), the Eye
                if 560 <= t < 880 or 960 <= t < 1040:
                    self.ring.emit(ctx)
                if 830 <= t < 880:
                    self.eye.emit(ctx)
        if 880 <= t < 960:
            self.globe.emit(ctx)
        if 960 <= t < 1040:
            from core import Frame
            ctx.fr_hand = Frame(ctx.scale)
            ctx.fr_cov = Frame(ctx.scale)
            ctx.fr_rim = Frame(ctx.scale)
            for fr_ in (ctx.fr_hand, ctx.fr_cov, ctx.fr_rim):
                fr_.prm[:] = ctx.fr.prm
                fr_.prm[6] = 1e9      # the hand keeps its own radiance and a solid silhouette: no depth fog
            self.hand.emit(ctx, ctx.fr_hand, ctx.fr_cov)
        if t >= 1040:
            self.ember.emit(ctx)

    def post(self, ctx, hdr):
        f = ctx.t
        H, W = hdr.shape[:2]
        if 830 <= f < 880 and hasattr(ctx, 'fr_eye'):
            # cut C: the slit of darkness, then the Eye's own fire on top
            import tolkien
            hdr = tolkien.Eye.darkness(ctx, hdr) + ctx.fr_eye.resolve()
        if f < 336:
            # warm light of the torch flame just below frame (continuity with INTRO)
            k = float(1 - smoothstep(300, 336, f))
            y, x = np.mgrid[0:H, 0:W].astype(np.float32)
            x = (x - W * 0.5) / (W * 0.30)
            y = (y - H * 1.08) / (H * 0.42)
            g = np.exp(-(x * x + y * y))[..., None]
            hdr += g * np.array([1.0, 0.42, 0.10], np.float32) * (0.9 * k)
        if 960 <= f < 1040 and hasattr(ctx, 'fr_hand'):
            import cv2
            cov = ctx.fr_cov.resolve()[..., 0]
            cov = cv2.GaussianBlur(cov, (0, 0), 1.5 * ctx.scale + 0.5)
            alpha = 1.0 - np.exp(-cov * 1.2)
            hdr = hdr * (1.0 - 0.985 * alpha[..., None]) + ctx.fr_hand.resolve()
            # rim light survives only near the outer silhouette (blurred coverage < 1 there)
            ab = cv2.GaussianBlur(alpha, (0, 0), 7.0 * ctx.scale)
            outer = np.clip((1.0 - ab) * 2.4, 0.0, 1.0)
            hdr += ctx.fr_rim.resolve() * (0.18 + 0.82 * outer)[..., None]
        if 1035 <= f < 1040:
            # white-red flash as the fingers close (1036-1039)
            k = {1035: 0.03, 1036: 0.14, 1037: 0.38, 1038: 0.7, 1039: 1.0}[int(f)]
            u, v, z = ctx.cam.project(B.crown_centre(960.0)[None, :], W, H)
            y, x = np.mgrid[0:H, 0:W].astype(np.float32)
            d2 = ((x - u[0]) ** 2 + (y - v[0]) ** 2) / (W * W)
            red = np.exp(-d2 / 0.09)[..., None] * np.array([3.0, 0.35, 0.12], np.float32)
            white = np.exp(-d2 / 0.012)[..., None] * np.array([6.0, 5.5, 5.0], np.float32)
            hdr += (red + white) * (6.0 * k) + np.array([0.9, 0.12, 0.05], np.float32) * (2.5 * k)
        return hdr

    def finish_opts(self, f):
        streak = 0.0
        if 455 <= f < 500:
            streak = 0.05 * float(smoothstep(455, 478, f)) * (1 - float(smoothstep(484, 500, f)))
        bloom = 0.12
        if f >= 480:
            bloom = 0.15
        if 880 <= f < 960:
            bloom = 0.14
        if f >= 1040:
            bloom = 0.1
        if 1030 <= f < 1040:
            streak = 0.06
        return dict(exposure=1.0, bloom_strength=bloom, bloom_threshold=0.7,
                    streak_strength=streak, vignette_amount=0.25)
