"""Default-off territorial light studies for C3440..3839.

The accepted result is the original LastBeacon.render_c() result. Opt-in
variants grade that result as a night page and warm each lit beacon's *actual*
territory. Borders use the original kingdom_at() partition, not new polygons;
coasts and lakes come from the existing source-pinned terra cache. No world or
atlas is baked here. Default rendering does not calculate candidate fields.

``render_variants(frame, shot, scale=.5)`` renders the accepted image once.
It returns accepted/territory-wash/beacon-falloff images, all display RGB.
Calls are sequential: the accepted renderer changes its per-instance sheet
sampler temporarily. Construct the original shot only against a prevalidated
existing atlas/world cache. These studies do not adopt a sequence-wide grade.
"""
from pathlib import Path

import cv2
import numpy as np

import last_beacon as BASE


VARIANTS = ('accepted', 'territory-wash', 'beacon-falloff')
LUMA = np.array([.2126, .7152, .0722], np.float32)
NIGHT_TINT = np.array([.87, .95, 1.08], np.float32)
WARM_LIGHT = np.array([.23, .11, .027], np.float32)


def _validate(frame, candidate, scale):
    if candidate not in VARIANTS:
        raise ValueError('unknown Last Beacon candidate')
    if not isinstance(frame, (int, np.integer)) or not BASE.F0 <= frame < BASE.F1:
        raise ValueError('Last Beacon accepts integer C3440..3839 only')
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('scale must be finite and in (0,1]')
    if min(round(1920*scale), round(804*scale)) < 1:
        raise ValueError('scale produces an empty image')


def make_shot(candidate='accepted'):
    """Return the unchanged original shot; candidates compose via render()."""
    if candidate not in VARIANTS:
        raise ValueError('unknown Last Beacon candidate')
    return BASE.LastBeacon()


def territory_maps(X, Y, land, lake):
    """Pure map-space classification from the same partition used for borders."""
    X, Y, land, lake = np.broadcast_arrays(X, Y, land, lake)
    if not all(np.isfinite(a).all() for a in (X, Y, land, lake)):
        raise ValueError('map coordinates and coast fields must be finite')
    labels = BASE.kingdom_at(X, Y)
    dry = (land >= .95) & (lake <= .05)
    falloff = np.empty(X.shape, np.float32)
    # A broad pool still reaches the entire territory. Its floor avoids
    # converting the experiment back to eight tiny isolated marks.
    for i, (bx, by) in enumerate(BASE.BEACONS):
        own = labels == i
        d2 = (X[own]-bx)**2 + (Y[own]-by)**2
        falloff[own] = .45 + .55*np.exp(-d2/(2*8.0**2))
    return dict(labels=labels, dry=dry, falloff=falloff)


def warmth_field(frame, maps, candidate='territory-wash', feather_pixels=0.):
    """Each region follows only its own unchanged catch curve; never a neighbour."""
    _validate(frame, candidate, .5)
    labels, dry = maps['labels'], maps['dry']
    if labels.shape != dry.shape or np.any(labels > 7) or np.any(labels < 0):
        raise ValueError('invalid kingdom labels or land-mask shape')
    if not np.isfinite(feather_pixels) or feather_pixels < 0:
        raise ValueError('feather width must be finite and nonnegative')
    if candidate == 'accepted':
        return np.zeros(labels.shape, np.float32)
    coverage = np.zeros(labels.shape, np.float32)
    catches = BASE.catch_at(frame)
    for i, catch in enumerate(catches):
        if catch <= 0:
            continue
        own = (labels == i) & dry
        if feather_pixels:
            if labels.ndim != 2:
                raise ValueError('edge feathering requires a 2D map')
            distance = cv2.distanceTransform(own.astype(np.uint8), cv2.DIST_L2, 3)
            edge = np.clip(distance/feather_pixels, 0, 1)
            coverage += own.astype(np.float32)*edge*float(catch)
        else:
            coverage[own] = catch
    if candidate == 'beacon-falloff':
        coverage *= maps['falloff']
    return coverage


def projected_territories(frame, width, height):
    """Read the existing world; refuse a cache miss rather than trigger a bake."""
    cam = BASE.camera_at(frame, width, height)
    world = Path(BASE.geo.CACHE)/f'terra_{BASE.geo.WORLD_VER}.npz'
    if not world.is_file():
        raise RuntimeError('Missing prepared terra cache; candidate may not generate it: '+str(world))
    gy, gx = np.mgrid[0:height, 0:width].astype(np.float64)
    X, Y = cam.screen_to_map((gx+.5).ravel(), (gy+.5).ravel())
    X, Y = X.reshape(height, width), Y.reshape(height, width)
    land = BASE.geo.sample('land', X, Y)
    lake = BASE.geo.sample('lake', X, Y)
    return territory_maps(X, Y, land, lake)


def _night_page(rgb):
    """Same neutral night-page gain/tint rule as the Reveal/Watch study.

    This is deliberately independent of RUN imports: their modules have names
    that collide with this map engine. The accepted temporal light drain stays
    in the underlying rendering and is read as nightfall in these variants.
    """
    luma = rgb@LUMA
    page = (.55*rgb+.45*luma[..., None])*NIGHT_TINT
    y = np.linspace(0., 1., rgb.shape[0], dtype=np.float32)
    gain = .44+.25*y*y*(3.-2.*y)
    return np.clip(page*gain[:, None, None], 0., 1.)


def composite(rgb, frame, maps, flame_mask, candidate='territory-wash'):
    """Add a paper wash in linear light, retaining ink and geometric flames."""
    _validate(frame, candidate, .5)
    if candidate == 'accepted':
        return rgb  # Identical array/values; no grade, copy or mask computation.
    image = np.asarray(rgb, np.float32)
    if (image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all()
            or np.any(image < 0) or np.any(image > 1)):
        raise ValueError('candidate requires finite display RGB in [0,1]')
    mask = np.asarray(flame_mask, np.float32)
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError('flame mask must be finite coverage matching the image')
    if maps['labels'].shape != image.shape[:2]:
        raise ValueError('territory map does not match image')
    coverage = warmth_field(frame, maps, candidate, feather_pixels=1.25*image.shape[1]/960)
    # Reflectance response leaves dark engraved lines dark rather than filling
    # the whole kingdom with an opaque infographic colour block.
    response = np.clip((image@LUMA-.025)/.70, 0, 1)**.65
    linear = BASE.look.srgb_to_linear(_night_page(image))
    linear = linear + coverage[..., None]*response[..., None]*WARM_LIGHT
    warmed = BASE.look.linear_to_srgb(linear)
    # This is the original renderer's actual glyph coverage, not a colour key
    # guessed from the composite. Preserve its values over the darker page.
    return np.clip(warmed*(1-mask[..., None])+image*mask[..., None], 0, 1).astype(np.float32)


def render_variants(frame, shot, scale=.5, variants=VARIANTS, return_fields=False):
    variants = tuple(variants)
    if not variants or len(variants) != len(set(variants)):
        raise ValueError('variants must be a nonempty distinct sequence')
    for variant in variants:
        _validate(frame, variant, scale)
    accepted = shot.render_c(frame, scale)
    results = {'accepted': accepted} if 'accepted' in variants else {}
    requested = [v for v in variants if v != 'accepted']
    fields = None
    if requested:
        height, width = accepted.shape[:2]
        fields = projected_territories(frame, width, height)
        cam = BASE.camera_at(frame, width, height)
        _, flame_mask = shot.beacon_glyphs(cam, shot.clock(frame))
        for variant in requested:
            results[variant] = composite(accepted, frame, fields, flame_mask, variant)
        fields = dict(fields, flame_mask=flame_mask,
                      warmth={v: warmth_field(frame, fields, v, 1.25*width/960) for v in requested})
    return (results, fields) if return_fields else results


def render(frame, shot=None, candidate='accepted', scale=.5):
    _validate(frame, candidate, scale)
    if shot is None:
        shot = make_shot(candidate)
    if candidate == 'accepted':
        return shot.render_c(frame, scale)
    return render_variants(frame, shot, scale, (candidate,))[candidate]
