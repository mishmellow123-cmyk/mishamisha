"""Default-off, shared Reveal/Watch night-fire illustration studies.

Accepted calls delegate to the frozen drivers. Candidates retain their camera,
terrain, catalogue and fire clocks. Three unequal curved tongues replace the
flat rooted glyph; the original ink_flames pass still projects and occludes
them. A second invocation of that pass carries geometric core alpha, rather
than trying to recover fire from a colour key. The optional third pass carries
a restrained smoke wisp. Lit ground is dimmed separately from emitted fire.

``render_variants`` ray-marches once per frame and composes sequentially. Its
optional ``return_layers=True`` returns (images, masks), with output-resolution
flame/core/smoke/glow arrays for candidate diagnostics. None are colour keys.
No functions are patched outside an exception-safe context. Use one render
thread per process; simultaneous calls into inkpass by other code are unsafe.
"""
from contextlib import contextmanager
import math
import threading

import numpy as np

import reveal_pair_c_v5 as reveal
import watch_c_v5 as watch


VARIANTS = ('accepted', 'night-fire', 'night-wisp')
KINDS = ('reveal', 'watch')
ORANGE = np.array([1.0, .47, .085], np.float32)
CORE = np.array([1.0, .965, .875], np.float32)
WISP = np.array([.66, .68, .70], np.float32)
LUMA = np.array([.2126, .7152, .0722], np.float32)
TINT = np.array([.87, .95, 1.08], np.float32)
_LOCK = threading.Lock()


def _linear(rgb):
    rgb = np.asarray(rgb, np.float32)
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4).astype(np.float32)


def _srgb(rgb):
    rgb = np.clip(np.asarray(rgb, np.float32), 0., 1.)
    return np.where(rgb <= .0031308, 12.92 * rgb,
                    1.055 * rgb ** (1. / 2.4) - .055).astype(np.float32)


def _driver(kind):
    if kind not in KINDS:
        raise ValueError(f'Unknown beacon shot {kind!r}; choose {KINDS}')
    return reveal if kind == 'reveal' else watch


def _source(frame, kind):
    driver = _driver(kind)
    return driver.local_frame(frame) if kind == 'reveal' else driver.source_frame(frame)


def _validate(frame, kind, variant, scale, ss):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown night-fire study {variant!r}; choose {VARIANTS}')
    if not np.isfinite([scale, ss]).all() or min(scale, ss) <= 0:
        raise ValueError('scale and ss must be finite and positive')
    if min(round(1920 * scale), round(804 * scale)) < 1:
        raise ValueError('scale must produce a nonempty image')
    return _source(frame, kind)


@contextmanager
def _guard():
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Night-fire candidate calls must run sequentially')
    try:
        yield
    finally:
        _LOCK.release()


def _curve_sdf(x, y, cx, cy, widths):
    """Distance to tapered, round-ended pen curves; no common flat baseline."""
    result = np.full(x.shape, np.inf)
    for j in range(len(cx) - 1):
        dx, dy = cx[j + 1] - cx[j], cy[j + 1] - cy[j]
        u = np.clip(((x - cx[j]) * dx + (y - cy[j]) * dy) / max(dx * dx + dy * dy, 1e-12), 0., 1.)
        distance = np.hypot(x - cx[j] - u * dx, y - cy[j] - u * dy)
        result = np.minimum(result, distance - (widths[j] * (1. - u) + widths[j + 1] * u))
    return result


def _fire_fields(PX, PY, Hp, t, seed, kpx):
    """Outer/core coverage of three uneven, swaying tongues in page pixels.

    A taller main tongue and two unequal shoulders deliberately avoid the
    symmetric three-point crown. The root is made by overlapping rounded
    strokes; there is no ellipse or straight horizontal base.
    """
    x, y = PX / Hp, PY / Hp
    s = np.linspace(0., 1., 28)
    outer = np.full(x.shape, np.inf)
    inner = np.full(x.shape, np.inf)
    phase = .731 * seed
    # Offset, height, root width, lean, phase. These are one illustration rule
    # for every site; seed supplies temporal phase, not emitted brightness.
    for offset, height, width, lean, ph in (
            (-.035, 1.26, .135, .105, 0.),
            (-.145, .80, .105, -.075, 1.7),
            (.135, .61, .105, .16, 4.1)):
        flicker = 1. + .065 * math.sin(2. * math.pi * .83 * t + phase + ph)
        bend = .060 * s ** 1.5 * np.sin(2. * math.pi * .68 * t - 2.8 * s + phase + ph)
        cx = offset * (1. - .45 * s) + lean * s ** 1.3 + bend
        cy = -.035 + height * flicker * s
        widths = width * (1. - s) ** 1.12 * (.88 + .30 * np.sin(math.pi * s))
        outer = np.minimum(outer, _curve_sdf(x, y, cx, cy, widths))
        inner = np.minimum(inner, _curve_sdf(x, y, cx * .53, cy * .64, widths * .57))
    outer *= Hp
    inner = np.maximum(inner * Hp, outer)
    alpha = np.clip(.5 - outer / kpx, 0., 1.)
    # Slightly fuller anti-aliasing of the small geometric core keeps both
    # sites' radiance represented at the same output scale; no per-site key.
    core = np.minimum(alpha, np.clip(.7 - inner / kpx, 0., 1.))
    return alpha.astype(np.float32), core.astype(np.float32)


def _wisp_field(PX, PY, Hp, t, seed, kpx, size):
    if size <= .6:
        return np.zeros(PX.shape, np.float32)
    s = np.linspace(0., 1., 35)
    phase = .731 * seed
    cx = (.12 + .19 * s + .075 * s * np.sin(5. * s - .65 * t + phase)) * Hp
    cy = (1.12 + .93 * s) * Hp
    widths = .38 * kpx * (1. - .8 * s)
    sd = _curve_sdf(PX, PY, cx, cy, widths)
    taper = np.clip((2.10 - PY / Hp) / .9, 0., 1.)
    return (np.clip(.5 - sd / kpx, 0., 1.) * taper * .24
            * min((size - .6) / .3, 1.)).astype(np.float32)


def _glyph(mode):
    def draw(PX, PY, Hp, t, seed, n, kpx, pw, grain, size=1.):
        if mode == 'smoke':
            alpha = _wisp_field(PX, PY, Hp, t, seed, kpx, size)
        else:
            alpha, core = _fire_fields(PX, PY, Hp, t, seed, kpx)
            alpha = core if mode == 'core' else alpha
        # Pigment variation is confined to the orange tongues. The core is
        # assigned its shared near-white radiance later, after the page grade.
        colour = np.broadcast_to(_linear(ORANGE), (*PX.shape, 3)).copy()
        colour *= (.97 + .03 * grain)[..., None]
        return alpha, colour, np.zeros(PX.shape, np.float32)
    return draw


def _night_page(rgb):
    """The ink-dusk gain/tint, with no warmth protection or colour classifier."""
    a = np.asarray(rgb, np.float32)
    luma = a @ LUMA
    result = (.55 * a + .45 * luma[..., None]) * TINT
    y = np.linspace(0., 1., a.shape[0], dtype=np.float32)
    gain = .44 + .25 * y * y * (3. - 2. * y)
    return np.clip(result * gain[:, None, None], 0., 1.)


def _finish(page, flame, colour, core, smoke, kpx):
    """Composite geometric masks over the separately graded page in linear RGB."""
    import cv2

    rgb = _linear(_night_page(page))
    # Two soft scales of emission, both derived from actual occluded core
    # coverage. No inferred points, colour selection, or screen-space rings.
    glow = np.clip(2.6 * cv2.GaussianBlur(core, (0, 0), 2.3 * kpx)
                   + 1.2 * cv2.GaussianBlur(core, (0, 0), 8.0 * kpx), 0., 1.)
    rgb += glow[..., None] * _linear(np.array([1., .48, .12])) * .30
    rgb = rgb * (1. - smoke[..., None]) + _linear(WISP) * smoke[..., None]
    rgb = rgb * (1. - flame[..., None]) + colour * flame[..., None]
    rgb = rgb * (1. - core[..., None]) + _linear(CORE) * core[..., None]
    return _srgb(rgb), glow.astype(np.float32)


@contextmanager
def _style(variant, captured):
    if variant == 'accepted':
        yield
        return
    import inkpass as IP

    saved = {name: getattr(IP, name) for name in ('compose', 'ink_flames', 'flame_glyph', 'fire_soft')}
    layers = {}

    def flames(*args, **kwargs):
        try:
            IP.flame_glyph = _glyph('outer')
            outer = saved['ink_flames'](*args, **kwargs)
            layers['flame_alpha'], layers['colour'] = outer[0], outer[1]
            del outer
            IP.flame_glyph = _glyph('core')
            layers['core_alpha'] = saved['ink_flames'](*args, **kwargs)[0]
            if variant == 'night-wisp':
                IP.flame_glyph = _glyph('smoke')
                layers['smoke_alpha'] = saved['ink_flames'](*args, **kwargs)[0]
            else:
                layers['smoke_alpha'] = np.zeros_like(layers['core_alpha'])
            # Compose the page without flame paint, so the night grade cannot
            # dim the actual fire, and recompositing cannot leave a gold ghost.
            zero = np.zeros_like(layers['flame_alpha'])
            return zero, np.zeros_like(layers['colour']), zero, zero
        finally:
            IP.flame_glyph = saved['flame_glyph']

    def fire_soft(*args, **kwargs):
        # Avoid the accepted wash's saturated gold cap. Its geometry/falloff is
        # still the shared source, but illumination of paper is kept subdued.
        return saved['fire_soft'](*args, **kwargs) * .35

    def compose(aov, prm=None, B=None, plate=None, CR=None):
        layers.clear()
        page, debug = saved['compose'](aov, prm=prm, B=B, plate=plate, CR=CR)
        if not layers:
            return _night_page(page), debug
        image, glow = _finish(page, layers['flame_alpha'], layers['colour'], layers['core_alpha'],
                              layers['smoke_alpha'], aov.get('kpx', page.shape[1] / 1920.))
        captured.update({name: layers[name] for name in ('flame_alpha', 'core_alpha', 'smoke_alpha')})
        captured['glow'] = glow
        return image, debug

    IP.compose, IP.ink_flames, IP.fire_soft = compose, flames, fire_soft
    try:
        yield
    finally:
        for name, original in saved.items():
            setattr(IP, name, original)


def _resize(rgb, scale):
    import cv2

    width, height = round(1920 * scale), round(804 * scale)
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
        raise ValueError('Invalid beacon night render')
    return rgb


def render(frame, shot, kind='reveal', variant='accepted', scale=.5, ss=2.):
    """One explicit candidate, or the untouched accepted driver by default."""
    _validate(frame, kind, variant, scale, ss)
    with _guard(), _style(variant, {}):
        return _driver(kind).render(frame, shot, scale=scale, ss=ss)


def render_variants(frame, shot, kind='reveal', variants=VARIANTS, scale=.5, ss=2., return_layers=False):
    """One AOV per frame; optional output-resolution semantic masks for review."""
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError('Supply distinct, nonempty variants')
    for variant in variants:
        source = _validate(frame, kind, variant, scale, ss)
    import cv2
    import inkpass as IP
    import render_ink as RI

    with _guard():
        aov = RI.render_aov(shot, source, scale, ss)
        aov['kpx'] = scale * ss
        images, masks = {}, {}
        for variant in variants:
            captured = {}
            with _style(variant, captured):
                rgb, _ = IP.compose(aov, B=shot.B, CR=shot.CR)
            images[variant] = _resize(rgb, scale)
            if return_layers:
                shape = (round(1920 * scale), round(804 * scale))
                masks[variant] = {name: cv2.resize(mask, shape, interpolation=cv2.INTER_AREA)
                                  for name, mask in captured.items()}
        return (images, masks) if return_layers else images
