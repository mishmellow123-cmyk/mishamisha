"""Default-off studies of the distant C5 Reveal beacon's legibility.

The accepted entry point delegates to reveal_pair_c_v5.render unchanged. The
studies retain that shot's terrain, camera, sites, ignition and fire clock:
``far-ink`` puts more pigment and pen into the existing distant glyph;
``far-scale`` draws that glyph 1.4 times larger, with a smaller pen adjustment.
Neither changes the light cast on the terrain or alters the near fire.
``halo-wash`` instead puts the same compact warm paper wash and reinforcement
of existing ink around both fires, leaving their glyphs unchanged. These are
comparison candidates, not adopted replacements.

Use one rendering thread in a process. The ink pass exposes module-level
functions; a context temporarily replaces them and restores them even if a
compose fails. Public calls reject overlap/re-entry instead of racing. Other
code must not call inkpass directly concurrently with these candidate calls.
``render_variants`` ray-marches once, then composes each requested variant
sequentially from the same AOV. The retained 2026-09-29 reveal-v2-probe receipt
records actual float-array equality between the driver, default render and
shared-AOV accepted output at C2960 (960x402, ss=2), for candidate source SHA256
a6343cbecec6fc5c5e31fe339582991f7e458e7b41f14e3943cadb5cb15cb1ed.
That sampled frozen run does not establish all-frame equivalence or visual
approval. Lightweight tests separately compare the current compose/resize
path with the actual driver using synthetic AOV/compose output, without a march.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import threading

import numpy as np

import reveal_pair_c_v5 as pair


@dataclass(frozen=True)
class Style:
    height: float = 1.0
    pen: float = 1.0
    coverage: float = 1.0
    gold_mix: float = 0.0


STYLES = {
    'accepted': Style(),
    'far-ink': Style(pen=1.35, coverage=1.10, gold_mix=0.32),
    'far-scale': Style(height=1.4, pen=1.12, coverage=1.04, gold_mix=0.12),
    'halo-wash': Style(),
}
FAR_SEED = 10  # inkpass: int(b[5]) * 7 + 3; the paired far site has id 1.
HALO_RADIUS = 48.0  # Page pixels at 1920 wide; identical for both sites.
_RENDER_LOCK = threading.Lock()


def _validate(variant, scale, ss):
    if variant not in STYLES:
        raise ValueError(f'Unknown Reveal study {variant!r}; choose {tuple(STYLES)}')
    if not np.isfinite([scale, ss]).all() or min(scale, ss) <= 0:
        raise ValueError('scale and ss must be finite and positive')
    if min(round(1920 * scale), round(804 * scale)) < 1:
        raise ValueError('scale must produce a nonempty output image')
    style = STYLES[variant]
    if (not np.isfinite([style.height, style.pen, style.coverage, style.gold_mix]).all()
            or min(style.height, style.pen, style.coverage) <= 0
            or not 0 <= style.gold_mix <= 1):
        raise ValueError('Candidate style values must be finite and in range')


@contextmanager
def _render_guard():
    if not _RENDER_LOCK.acquire(blocking=False):
        raise RuntimeError('Reveal candidate calls must run sequentially in one process')
    try:
        yield
    finally:
        _RENDER_LOCK.release()


@contextmanager
def _candidate_style(variant):
    """Patch only the fire drawing pass, leaving the terrain-light pass intact."""
    if variant == 'accepted':
        yield
        return
    if variant == 'halo-wash':
        with _halo_style():
            yield
        return
    import inkpass as IP

    style = STYLES[variant]
    original_glyph = IP.flame_glyph
    original_scale = IP.flame_scale
    original_flames = IP.ink_flames

    def glyph(PX, PY, Hp, t, seed, n, kpx, pw, grain, size=1.0):
        if seed != FAR_SEED:
            return original_glyph(PX, PY, Hp, t, seed, n, kpx, pw, grain, size)
        alpha, colour, ink = original_glyph(
            PX, PY, Hp, t, seed, n, kpx, pw * style.pen, grain, size)
        # Existing GOLD pigment, confined to the original glyph's coverage.
        # There is no screen-space blob or added emission around the flame.
        colour = colour * (1.0 - style.gold_mix) + IP.s2l(IP.GOLD) * style.gold_mix
        return np.clip(alpha * style.coverage, 0, 1), colour, ink

    def scale(z):
        # This branch is used only inside ink_flames, never by fire_soft.
        return original_scale(z) * (style.height if z > 2500.0 else 1.0)

    def flames(*args, **kwargs):
        IP.flame_glyph = glyph
        IP.flame_scale = scale
        try:
            return original_flames(*args, **kwargs)
        finally:
            IP.flame_glyph = original_glyph
            IP.flame_scale = original_scale

    IP.ink_flames = flames
    try:
        yield
    finally:
        IP.flame_glyph = original_glyph
        IP.flame_scale = original_scale
        IP.ink_flames = original_flames


def _halo_wash(rgb, aov, beacons, ink, protected):
    """Local paper intervention only; compact support and unchanged fire pixels.

    Both sites use the same 48-page-pixel radius, coefficients and envelope.
    This is explicitly an illustration study, not a world-lighting simulation.
    """
    import beacons as BC
    import inkpass as IP

    if beacons is None or not len(beacons):
        return rgb
    out = rgb.copy()
    height, width = rgb.shape[:2]
    kpx = aov.get('kpx', width / 1920.0)
    radius = HALO_RADIUS * kpx
    cam = aov['cam']
    warm = IP.s2l(IP.GOLD_HI)
    ink_colour = IP.s2l(IP.INK)
    for beacon in beacons:
        size, _, light = BC.env(aov['frame'], beacon[3])
        if size <= .02 or light <= 0:
            continue
        base = np.asarray(beacon[:3]) + np.array([0., 1.6, 0.])
        delta = base - cam['pos']
        z = float(delta @ cam['fwd'])
        if z <= 1:
            continue
        fd = float(np.linalg.norm(delta))
        sx = cam['cx'] + cam['f'] * float(delta @ cam['right']) / z
        sy = cam['cy'] - cam['f'] * float(delta @ cam['up']) / z
        hp = 6.5 * IP.flame_scale(z) * min(size, 1.15) * cam['f'] / z
        # Centre the wash on the existing flame, not the stone below its root.
        sy -= .42 * hp
        x0, x1 = max(int(np.floor(sx - radius)), 0), min(int(np.ceil(sx + radius)), width)
        y0, y1 = max(int(np.floor(sy - radius)), 0), min(int(np.ceil(sy + radius)), height)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r2 = ((xx + .5 - sx) ** 2 + (yy + .5 - sy) ** 2) / radius ** 2
        field = np.clip(1. - r2, 0., 1.) ** 3
        # Use the same foreground-depth test as the accepted fire pass. A fire
        # hidden by a nearer ridge must not advertise itself through that ridge.
        visible = aov['A'][y0:y1, x0:x1, 1] > fd - max(3., .004 * fd)
        field *= np.clip(light, 0., 1.) * visible
        active = (field > 0) & ~protected[y0:y1, x0:x1]
        if not active.any():
            continue
        patch = out[y0:y1, x0:x1]
        linear = IP.s2l(patch * 255.)
        existing_ink = np.clip(ink[y0:y1, x0:x1], 0., 1.)
        dark = .18 * field * existing_ink
        linear = linear * (1. - dark[..., None]) + ink_colour * dark[..., None]
        wash = .32 * field * (1. - .85 * existing_ink)
        linear = linear * (1. - wash[..., None]) + warm * wash[..., None]
        changed = IP.look.linear_to_srgb(np.clip(linear, 0., 1.)).astype(rgb.dtype)
        # Preserve every pixel outside the compact support and every pixel of
        # the accepted flames/inked smoke exactly, including float rounding.
        patch[active] = changed[active]
    return out


@contextmanager
def _halo_style():
    import inkpass as IP

    original_compose, original_flames = IP.compose, IP.ink_flames
    protected = None

    def flames(*args, **kwargs):
        nonlocal protected
        result = original_flames(*args, **kwargs)
        protected = (result[0] > 0) | (result[2] > 0)
        return result

    def compose(aov, prm=None, B=None, plate=None, CR=None):
        nonlocal protected
        protected = None
        rgb, layers = original_compose(aov, prm=prm, B=B, plate=plate, CR=CR)
        if protected is None:
            return rgb, layers
        return _halo_wash(rgb, aov, B, layers['ink'], protected), layers

    IP.ink_flames, IP.compose = flames, compose
    try:
        yield
    finally:
        IP.ink_flames, IP.compose = original_flames, original_compose


def _compose(aov, shot, scale):
    import cv2
    import inkpass as IP

    rgb, _ = IP.compose(aov, B=shot.B, CR=shot.CR)
    width, height = int(round(1920 * scale)), int(round(804 * scale))
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
        raise ValueError('Invalid paired Reveal candidate render')
    return rgb


def render(frame, shot, variant='accepted', scale=.5, ss=2.):
    """Render one explicit study; the default is the accepted renderer itself."""
    _validate(variant, scale, ss)
    pair.local_frame(frame)
    with _render_guard():
        with _candidate_style(variant):
            return pair.render(frame, shot, scale=scale, ss=ss)


def render_variants(frame, shot, variants=('accepted', 'far-ink', 'far-scale', 'halo-wash'), scale=.5, ss=2.):
    """Return a variant->RGB mapping using one unchanged terrain AOV per frame.

    The accepted compose follows pair.render's exact compose/resize steps;
    it does not call pair.render because that would repeat the ray march.
    """
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError('Supply a nonempty sequence of distinct Reveal variants')
    for variant in variants:
        _validate(variant, scale, ss)
    local = pair.local_frame(frame)
    import render_ink as RI

    with _render_guard():
        aov = RI.render_aov(shot, local, scale, ss)
        aov['kpx'] = scale * ss
        results = {}
        for variant in variants:
            with _candidate_style(variant):
                # compose makes its own dictionary and replacement AOV; it does
                # not mutate this shared input (inkpass.py:compose).
                results[variant] = _compose(aov, shot, scale)
        return results
