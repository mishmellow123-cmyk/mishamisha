"""Default-off, image-only dusk study for the accepted ink landscape.

``grade(rgb)`` copies the accepted float RGB array exactly. Opt in with
``grade(rgb, variant='ink-dusk')``. Inputs and outputs are display-referred
sRGB floats in [0, 1]; this is a post grade, not a lighting simulation.
There are no shot, frame, camera, beacon, or region-of-interest inputs.

The fixed warmth key was measured on decoded accepted 960x402 Reveal JPEGs
C2880, C2886 and C2960: the sky's largest R-B was .2902; the C2960 near/far
detail regions reached .4745/.4549 (95th percentiles .4196/.3882). Thus .30
starts protection outside the measured paper range, with full warmth
protection at .38. A .50-.69 display-luma gate excludes dark warm ink.
These thresholds identify colour, not semantic fire: other sufficiently
bright gold in Beacon Run or Watch will also be protected. Review that
continuity before adopting the grade for those shots.

The unprotected page gets a fixed cool, partly desaturated ink tone and a
smooth vertical gain of .44 at the top to .69 at the foreground. This
preserves local line variation without a blur, object-location mask or per-frame
histogram fit. Warm, luminous source pixels are blended back unchanged;
the existing source glow is protected by the same key, with no added halo.
"""
import numpy as np


VARIANTS = ('accepted', 'ink-dusk')
LUMA = np.array([.2126, .7152, .0722], dtype=np.float64)
TINT = np.array([.87, .95, 1.08], dtype=np.float64)
WARMTH_LOW, WARMTH_HIGH = .30, .38
LUMA_LOW, LUMA_HIGH = .50, .69


def _image(rgb):
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 1:
        raise ValueError('expected a nonempty H x W x 3 RGB image')
    if not np.issubdtype(a.dtype, np.floating):
        raise ValueError('expected floating-point sRGB in [0, 1]')
    if not np.isfinite(a).all() or np.any((a < 0.) | (a > 1.)):
        raise ValueError('RGB values must be finite and in [0, 1]')
    return a


def _smoothstep(low, high, x):
    u = np.clip((x-low)/(high-low), 0., 1.)
    return u*u*(3.-2.*u)


def _key(a):
    warmth = _smoothstep(WARMTH_LOW, WARMTH_HIGH, a[..., 0]-a[..., 2])
    luminous = _smoothstep(LUMA_LOW, LUMA_HIGH, a @ LUMA)
    return warmth*luminous


def warmth_key(rgb):
    """Return the fixed image-only protection key, for study diagnostics."""
    return _key(_image(rgb).astype(np.float64, copy=False))


def grade(rgb, variant='accepted'):
    """Return a new same-shaped, same-dtype image; accepted is bit-exact."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown ink grade {variant!r}; choose {VARIANTS}')
    source = _image(rgb)
    if variant == 'accepted':
        return source.copy()
    a = source.astype(np.float64, copy=False)
    key = _key(a)
    luma = a @ LUMA
    dusk = (.55*a + .45*luma[..., None])*TINT
    y = np.linspace(0., 1., a.shape[0])
    gain = .44 + .25*_smoothstep(0., 1., y)
    dusk *= gain[:, None, None]
    result = dusk*(1.-key[..., None]) + a*key[..., None]
    result = np.clip(result, 0., 1.).astype(source.dtype)
    # Pin complete protection to source bytes, including narrower float dtypes.
    result[key == 1.] = source[key == 1.]
    return result
