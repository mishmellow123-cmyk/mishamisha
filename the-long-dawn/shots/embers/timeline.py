"""Timeline: which elements are alive when, the camera, and per-frame finish settings."""
import numpy as np

from core import Camera, smoothstep, lerp, catmull
import scene_a as A


class Timeline:
    def __init__(self):
        self.embers = A.Embers()
        self.glow = A.TorchGlow()
        self.glyphs = A.Glyphs()
        self.point = A.ThePoint(self.glyphs)

    # ---------------------------------------------------------------- camera
    def camera(self, t):
        if t < 484:
            pos, tgt = A.cam_a(t)
            # focus: near hero glyphs early, the spiral centre later
            fd = float(np.linalg.norm(tgt - pos))
            focus = lerp(lerp(7.0, A.HERO_FOCUS, float(smoothstep(318, 336, t))), np.linalg.norm(pos),
                         float(smoothstep(398, 432, t)))
            ap = lerp(0.12, 0.05, float(smoothstep(392, 440, t)))
            return Camera(pos, tgt, hfov=A.HFOV_A(t),
                          focus=focus, aperture=ap)
        return Camera((0, 0, 10), (0, 0, 0))

    def render_opts(self, f):
        if f < 484:
            fs = float(lerp(16.0, 45.0, smoothstep(405, 430, f)))
            return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=fs, fog_len=22.0)
        return dict(bokeh_pow=0.45, bokeh_cap=3.0)

    def emit(self, ctx):
        t = ctx.t
        if t < 484:
            self.glow.emit(ctx)
            self.embers.emit(ctx)
            self.glyphs.emit(ctx)
            self.point.emit(ctx)

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
            streak = 0.05 * float(smoothstep(455, 478, f))
        return dict(exposure=1.0, bloom_strength=0.12, bloom_threshold=0.7,
                    streak_strength=streak, vignette_amount=0.25)
