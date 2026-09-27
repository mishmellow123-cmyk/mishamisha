"""EMBERS v3, cut A (EMBERS-A2): A16 TOWERS IN THE LIGHT and A17 THE FIRE, SEEN (stub: delegates to edge.py)."""
import numpy as np

import a3 as A
import edge


def camera(tl, t):
    return edge.camera(tl, t)


def render_opts(tl, f):
    return dict(bokeh_pow=0.3, bokeh_cap=2.0, fog_start=45.0, fog_len=70.0, near=0.3)


def emit(tl, ctx):
    t = ctx.t
    lp, lc, lpw = tl.light(t)
    ctx.tl_crater = edge._crater(tl)
    tl.towers.prepare(ctx)
    tl.dust.emit(ctx)
    tl.smoke.emit(ctx, lp, lc, lpw)
    edge.emit_before_towers(tl, ctx, lp, lc, lpw)
    tl.towers.emit(ctx, lp, lc, lpw)
    tl.tembers.emit(ctx)
    tl.tsmoke.emit(ctx, lp, lc, lpw)
    tl.sparks.emit(ctx)
    tl.fire.emit(ctx)
    A.emit_haze(ctx)
    tl.fsparks.emit(ctx)
    tl.shock.emit(ctx)
    edge.emit_after(tl, ctx, lp, lc, lpw)


def post(tl, ctx, hdr):
    return edge.post(tl, ctx, hdr)


def finish_opts(tl, f):
    return dict(exposure=1.0, bloom_strength=0.15, bloom_threshold=0.7, streak_strength=0.0, vignette_amount=0.25)
