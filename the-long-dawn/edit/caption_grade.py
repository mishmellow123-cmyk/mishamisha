"""Post-finish grades for C's baked inscriptions, using checked screen-space bands.

The exported polygons replace runtime camera/font reconstruction. Curves and
band rasterization match captionsC_baked_grade_probe.py at native resolution.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

DATA_PATH = Path(__file__).with_name('data') / 'c5_caption_bands.json'
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def source():
    """Delivery cache source: this module and the exact current geometry bytes."""
    raw = DATA_PATH.read_bytes()
    return Path(__file__).read_text() + '\nBAND_SHA256=' + hashlib.sha256(raw).hexdigest() + '\n' + raw.decode('utf-8')


def strength(frame, spec):
    """Zero at both window endpoints; full0/full1 are inclusive hold endpoints."""
    f0, f1, a, b = (spec[k] for k in ('f0', 'f1', 'full0', 'full1'))
    if not f0 < a <= b < f1 - 1:
        raise ValueError('Invalid caption-grade envelope')
    if frame <= f0 or frame >= f1 - 1:
        return 0.0
    enter = np.clip((frame - f0) / (a - f0), 0.0, 1.0)
    leave = np.clip((f1 - 1 - frame) / (f1 - 1 - b), 0.0, 1.0)
    return float(min(enter * enter * (3 - 2 * enter), leave * leave * (3 - 2 * leave)))


@lru_cache(maxsize=2)
def _load(raw):
    data = json.loads(raw)
    if data.get('version') != 1 or data.get('dimensions') != [1920, 804]:
        raise ValueError('Unsupported caption-band schema or dimensions')
    band = data.get('band', {})
    expected = dict(inner_margin_pixels=24, feather_pixels=240, boundary_stride_source_texels=8, raster_shift_bits=8)
    if any(band.get(k) != v for k, v in expected.items()):
        raise ValueError('Caption-band geometry settings differ from the measured probe')
    for ident, item in data['bands'].items():
        frames = set(map(str, range(item['frame_start'], item['frame_end_exclusive'])))
        if not frames or set(item['outlines']) != frames:
            raise ValueError(f'{ident}: incomplete caption-band frame coverage')
        for frame, outline in item['outlines'].items():
            xy = np.asarray(outline, np.float64)
            if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4 or not np.isfinite(xy).all()
                    or abs(cv2.contourArea(xy.astype(np.float32))) <= 1e-6):
                raise ValueError(f'{ident} frame {frame}: invalid caption-band polygon')
    return data


def _band(outline, shape):
    height, width = shape[:2]
    scale = width / 1920.0
    if height != round(804 * scale):
        raise ValueError('Caption-grade image must preserve the 1920x804 aspect ratio')
    # JSON stores pixel-index centres; resize those centres, including the half-pixel offset.
    outline = np.asarray(outline, np.float64)
    if scale != 1.0:
        outline = (outline + 0.5) * scale - 0.5
    inner, feather = 24.0 * scale, 240.0 * scale
    pad = inner + feather + 2.0 * scale
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    x0, y0 = max(0, int(np.floor(lo[0] - pad))), max(0, int(np.floor(lo[1] - pad)))
    x1, y1 = min(width, int(np.ceil(hi[0] + pad)) + 1), min(height, int(np.ceil(hi[1] + pad)) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Caption band is outside the picture')
    interior = np.zeros((y1 - y0, x1 - x0), np.uint8)
    points = np.rint((outline - [x0, y0]) * 256).astype(np.int32)
    cv2.fillPoly(interior, [points], 1, lineType=cv2.LINE_8, shift=8)
    if not interior.any():
        raise ValueError('Caption band has no interior pixels')
    distance = cv2.distanceTransform(1 - interior, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    u = np.clip((inner + feather - distance) / feather, 0, 1)
    return (x0, y0, x1, y1), (u * u * (3 - 2 * u)).astype(np.float32)


def _grade(rgb, alpha, curve):
    """Probe's linear-luminance curve and linear-light blend, without diagnostics."""
    kind, gain = curve['kind'], curve['gain']
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Caption-grade gain must be finite and positive')
    if kind == 'blackpoint_gain':
        black = curve['blackpoint']
        if not np.isfinite(black) or not 0 <= black < 1:
            raise ValueError('Invalid caption-grade blackpoint')
        if gain == 1 and black == 0:
            return rgb.copy()
    elif kind == 'shadow_shoulder':
        if not (np.isfinite(curve['low']) and np.isfinite(curve['high'])
                and 0 <= curve['low'] < curve['high'] <= 1):
            raise ValueError('Invalid caption-grade shoulder')
    else:
        raise ValueError(f'Unknown caption-grade curve: {kind}')
    x = np.clip(rgb, 0.0, 1.0)
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)
    lum = linear @ LUMA
    if kind == 'blackpoint_gain':
        target_lum = gain * np.maximum(lum - curve['blackpoint'], 0.0)
    else:
        u = np.clip((lum - curve['low']) / (curve['high'] - curve['low']), 0, 1)
        target_lum = gain * lum * (u * u * (3 - 2 * u))
    ratio = np.divide(target_lum, lum, out=np.zeros_like(lum), where=lum > 1e-12)
    target = linear * ratio[..., None]
    mixed = linear + alpha[..., None] * (np.clip(target, 0, 1) - linear)
    out = rgb.copy()
    active = alpha > 0
    if kind == 'shadow_shoulder' and gain == 1.0:
        active = active & (lum < curve['high'])
    x = np.maximum(mixed[active], 0.0)
    out[active] = np.where(x <= 0.0031308, 12.92 * x,
                           1.055 * x ** (1.0 / 2.4) - 0.055).astype(np.float32)
    return out


def apply(image, frame, spec, source):
    """Grade finished float32 RGB pixels; fail loudly on an unregistered active frame."""
    amount = strength(frame, spec)
    if amount == 0:
        return image
    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Caption grade expects an HxWx3 float32 image')
    # Reread before every use: a cached successful load must not hide an edited geometry file.
    raw = DATA_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['band_sha256']:
        raise ValueError('Caption-band checksum mismatch')
    data = _load(raw)
    try:
        band = data['bands'][spec['id']]
    except KeyError as exc:
        raise ValueError(f'No caption band for {spec["id"]}') from exc
    if (source is None or source.get('stem') != spec['source_stem']
            or band['source_stem'] != spec['source_stem']):
        raise ValueError('Caption-band source stem mismatch')
    # Harvested plates retain the registered geometry's timeline. This explicit offset maps the new cut's
    # frame to that timeline; its source mapping must still name the same physical source frame.
    band_off = spec.get('band_frame_off', 0)
    if isinstance(band_off, bool) or not isinstance(band_off, int):
        raise ValueError('Caption-band frame offset must be an integer')
    if (source.get('off') != spec['source_off'] or band['source_off'] != spec['source_off'] - band_off
            or source.get('mode') != 'exact' or source.get('crop') is not None or source.get('video') is not None
            or source.get('hold') is not None or source.get('clamp') is not None
            or source.get('screen_transform') is not None):
        raise ValueError('Caption-band source offset or geometry mapping mismatch')
    if (band['frame_start'], band['frame_end_exclusive']) != (spec['f0'] + band_off, spec['f1'] + band_off):
        raise ValueError('Caption-band frame coverage differs from grade window')
    box, alpha = _band(band['outlines'][str(frame + band_off)], image.shape)
    x0, y0, x1, y1 = box
    out = image.copy()
    out[y0:y1, x0:x1] = _grade(image[y0:y1, x0:x1], alpha * amount, spec['curve'])
    return out
