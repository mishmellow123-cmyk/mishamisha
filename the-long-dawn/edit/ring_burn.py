"""Opt-in drawn-Ring transition using the book's x1burn v2 field, evaluated without layer files.

The field and finish are the same as shots/map/x1burn.py. Coordinates are native 1920x804 pixels;
preview samples cover the same native screen. RGB keep carries scorch/char, cover carries the hole,
and the display-sRGB glow is additive. No frames are written by this module.
"""
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _engine():
    sys.path.insert(0, str(ROOT / 'shots' / 'map'))
    import x1burn
    return x1burn


def source():
    """All pixel-producing dependencies, also usable by metadata-only delivery planning."""
    paths = [Path(__file__), ROOT/'shots/map/x1burn.py', ROOT/'shots/map/burn.py',
             ROOT/'shots/map/noise.py', ROOT/'lib/look.py']
    return '\n'.join(p.read_text() for p in paths)


def layers(frame, spec, width, height):
    """Return (glow RGB, keep RGB, cover scalar), all float32 in display coordinates."""
    engine = _engine()
    cx, cy = spec['center']
    unit = engine.UNIT
    bf = engine.BURN.v2(engine.BURN.params(
        (cx/unit, cy/unit), t_start=spec['t_open']/24., speed=spec['speed'],
        p=1.5, amp=.45, freq=.35, seed=spec.get('seed', 12), brown=1.6,
        char=.25, edge=.035, lead=.3))
    ys, xs = np.mgrid[0:height, 0:width]
    u = ((xs.astype(np.float64)+.5)*(1920./width))/unit
    v = ((ys.astype(np.float64)+.5)*(804./height))/unit
    glow = np.zeros((height, width, 3), np.float32)
    keep = np.ones_like(glow)
    cover = np.ones((height, width), np.float32)
    engine._kernel(u, v, glow, keep, cover, bf, frame/24.)
    glow = engine.look.finish(glow, exposure=1., bloom_strength=.05, bloom_threshold=1.,
                              vignette_amount=0., lift=0.)
    return glow, np.clip(keep, 0., 1.), cover


def composite(outgoing, incoming, frame, spec):
    if frame == spec['f0']:
        return outgoing.copy()
    glow, keep, cover = layers(frame, spec, outgoing.shape[1], outgoing.shape[0])
    return outgoing*keep + incoming*(1.-cover[..., None]) + glow


def screen_transform(image, frame, spec):
    """Explicit stand-in camera: align an interpolated source anchor to a moving destination.

    Uniform scale preserves the flame's shape/colour; a constant film-black border extends its background.
    Only takes carrying this spec are transformed. The future camera-aligned D plate omits it entirely.
    """
    u = float(np.clip((frame-spec['f0'])/(spec['f1']-spec['f0']), 0., 1.))
    e = u*u*(3.-2.*u)
    a = float(np.clip((frame-spec['source_f0'])/(spec['source_f1']-spec['source_f0']), 0., 1.))
    anchor = np.asarray(spec['anchor0'])*(1.-a) + np.asarray(spec['anchor1'])*a
    target = np.asarray(spec['target0'])*(1.-e) + np.asarray(spec['target1'])*e
    zoom = spec['scale0']*(1.-e) + spec['scale1']*e
    h, w = image.shape[:2]
    offset = (target-zoom*anchor)*np.asarray((w/1920., h/804.))
    matrix = np.asarray([[zoom, 0., offset[0]], [0., zoom, offset[1]]], np.float32)
    return cv2.warpAffine(image, matrix, (w,h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(spec['border']))
