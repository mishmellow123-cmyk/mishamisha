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
            focus = lerp(5.5, np.linalg.norm(pos), float(smoothstep(392, 430, t)))
            ap = lerp(0.16, 0.05, float(smoothstep(392, 440, t)))
            return Camera(pos, tgt, hfov=float(lerp(50, 46, smoothstep(300, 480, t))),
                          focus=focus, aperture=ap)
        return Camera((0, 0, 10), (0, 0, 0))

    def render_opts(self, f):
        return dict(bokeh_pow=0.6, bokeh_cap=6.0)

    def emit(self, ctx):
        t = ctx.t
        if t < 484:
            self.glow.emit(ctx)
            self.embers.emit(ctx)
            self.glyphs.emit(ctx)
            self.point.emit(ctx)

    def post(self, ctx, hdr):
        return hdr

    def finish_opts(self, f):
        streak = 0.0
        if 455 <= f < 500:
            streak = 0.05 * float(smoothstep(455, 478, f))
        return dict(exposure=1.0, bloom_strength=0.12, bloom_threshold=0.7,
                    streak_strength=streak, vignette_amount=0.25)
