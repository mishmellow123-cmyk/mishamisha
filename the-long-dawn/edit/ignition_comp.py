"""A4/A5's bounded collapse and ignition. Inputs are finished sRGB plates.

The EDL opts into these treatments; other cuts keep their existing compositor.
Source samples are read one at a time, and captions stay on the cut-frame clock.
"""
import math

import numpy as np


def ease(a, b, f):
    u = float(np.clip((f - a) / (b - a), 0, 1))
    return u * u * (3 - 2 * u)


def linear(x):
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4).astype(np.float32)


def srgb(x):
    x = np.maximum(x, 0)
    return np.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055).astype(np.float32)


def collapse_source(f, t):
    """Hermite retime: enter at native speed and arrive at the point at zero speed."""
    start, land = t['f0'], t['land']
    end = t['source_end']
    u = float(np.clip((f - start) / (land - start), 0, 1))
    m = (land - start) / (end - start)
    return start + (end - start) * ((m - 2) * u ** 3 + (3 - 2 * m) * u ** 2 + m * u)


def collapse(f, t, read):
    """Five source-frame shutter taps soften angular aliasing as motion settles.

    The shutter eases in from an exact native first frame. Its width contracts
    with the arrival, so the held point has no residual glyph ghosts.
    """
    source = collapse_source(f, t)
    mix = ease(t['f0'], t['f0'] + 3, f)
    radius = t.get('shutter', 2.0) * (1 - ease(t['land'] - 12, t['land'], f))
    weights = np.asarray([1, 2, 3, 2, 1], np.float32) / 9
    coefficients = {}
    for offset, weight in zip(np.linspace(-radius, radius, 5), weights):
        pos = float(np.clip(source + offset, t['f0'], t['source_end']))
        lo = int(math.floor(pos))
        for g, w in ((lo, 1 - (pos - lo)), (min(lo + 1, t['source_end']), pos - lo)):
            coefficients[g] = coefficients.get(g, 0) + float(weight) * w
    out = None
    for g, weight in coefficients.items():
        if weight <= 0:
            continue
        sample = read(g)
        if sample is None:
            return None  # Let the dispatcher preserve its normal missing-source fallback.
        img = linear(sample) * weight
        out = img if out is None else out + img
    if mix < 1:
        native = read(f)
        if native is None:
            return None
        if mix == 0:
            return native
        out = linear(native) * (1 - mix) + out * mix
    return np.clip(srgb(out), 0, 1)


def swell(o, i, d, f, t, k):
    """Carry the disc into the moving flame, with a shared smooth light envelope.

    Sample the disc in linear light; the incoming flame's own bright pixels
    locate its moving body. A gradually narrowing, rising halo joins them.
    """
    if f >= t['f1'] - 1:
        return i
    h, w = d.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = np.asarray(t['disc'], np.float32) * k
    rd = np.hypot(xx - dx, yy - dy)
    bins = rd.astype(np.int32).ravel()
    count = np.maximum(np.bincount(bins), 1)
    dl = linear(d)
    base_d = np.median(dl, axis=(0, 1))
    profile = np.stack([np.bincount(bins, dl[..., c].ravel()) / count for c in range(3)], 1)
    profile = np.maximum(profile - base_d, 0)
    profile *= np.clip(1.2 - np.arange(len(profile)) / (t['R'] * k), 0, 1)[:, None]

    def disc(cx, cy, sx, sy):
        radius = np.hypot((xx - cx) / sx, (yy - cy) / sy)
        return np.stack([np.interp(radius, np.arange(len(profile)), profile[:, c], right=0)
                         for c in range(3)], -1).astype(np.float32)

    if f < t['cut']:
        u = ease(t['f0'], t['cut'], f)
        cx, cy = (np.asarray(t['p0']) * (1 - u) + np.asarray(t['disc']) * u) * k
        scale = t['s0'] + (1 - t['s0']) * u
        ol = linear(o)
        light = disc(cx, cy, scale, scale) * u
        return np.clip(srgb(np.maximum(ol, base_d + light)), 0, 1)

    v = ease(t['cut'], t['f1'] - 1, f)
    il = linear(i)
    base_i = np.median(il, axis=(0, 1))
    # Locate the luminous flame body only in its central column, excluding grain.
    lum = i.mean(2)
    weight = np.maximum(lum - .55, 0) * (np.abs(xx - dx) < 150 * k)
    total = float(weight.sum())
    fx = float((weight * xx).sum() / total) if total > 1e-6 else dx
    fy = float((weight * yy).sum() / total) if total > 1e-6 else dy
    cx, cy = dx * (1 - v) + fx * v, dy * (1 - v) + fy * v
    scale = 1 - .72 * v
    halo = disc(cx, cy, scale, scale * (1 + .8 * math.sin(math.pi * v))) * (1 - v) ** 1.4
    flame = np.maximum(il - base_i, 0) * ease(t['cut'], t['cut'] + 14, f)
    base = base_d * (1 - v) + base_i * v
    return np.clip(srgb(base + np.maximum(flame, halo)), 0, 1)
