"""EMBERS v3, cut A (EMBERS-A2): A3 INTO THE LIGHT . GLYPHS and A4 THE POINT (stub)."""
import numpy as np
from core import Camera


def camera(tl, t):
    return Camera(np.array([0.0, 0.0, 10.0]), np.zeros(3), hfov=50.0)


def render_opts(tl, f):
    return dict(bokeh_pow=0.3, bokeh_cap=2.0, near=0.3)


def emit(tl, ctx):
    pass


def post(tl, ctx, hdr):
    return hdr


def finish_opts(tl, f):
    return dict(exposure=1.0, bloom_strength=0.12, bloom_threshold=0.7, streak_strength=0.0, vignette_amount=0.25)
