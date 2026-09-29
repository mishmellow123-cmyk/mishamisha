"""Cheap optional warm-page comparison for already-rendered night-fire images.

This is a page-colour study, not a new fire extraction method. It tints only
dark, nearly neutral pixels; bright cores and saturated orange tongues retain
their input values. Renderer masks on probe frames can verify that protection.
The default is an exact copy. It does not change timing, shapes or geometry.
"""
import numpy as np


def _smooth(a, b, values):
    u = np.clip((values - a) / (b - a), 0., 1.)
    return u * u * (3. - 2. * u)


def grade(image, variant='accepted'):
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype.kind != 'f':
        raise ValueError('Expected a floating RGB image')
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError('Expected finite display RGB in [0,1]')
    if variant not in ('accepted', 'sepia-night'):
        raise ValueError(f'Unknown page tint: {variant}')
    if variant == 'accepted':
        return rgb.copy()
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    luma = rgb @ np.asarray([.2126, .7152, .0722], dtype=rgb.dtype)
    amount = (1.-_smooth(.055, .12, chroma)) * (1.-_smooth(.52, .70, luma))
    multiplier = np.asarray([.97, .72, .49], dtype=rgb.dtype)
    tint = rgb * multiplier
    return rgb + (tint-rgb) * amount[..., None]
