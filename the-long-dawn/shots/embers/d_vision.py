"""Cut D's conditional Eye, bracketed by a working, open Ring.

Explicit entry points only; the accepted A/C drivers never import this module.
All times are D frames. The gap aperture is measured between the projected
solid end faces; it is not the unbounded connected black region outside a ring.
"""
import math
import time
from functools import lru_cache

import cv2
import numpy as np

import c3
import c2
import c_d
import openring_d
import glyphs3
import d_glare
import ringsolid as RS
import look
from core import Camera, smoothstep, smootherstep

RING_C = c_d.RING_C.copy()
RING_SIZE = 2.2
GAP_DEGREES = 55.
AZ = c3.AZ0 + .082
RIGHT = np.array([math.sin(AZ), 0., -math.cos(AZ)])
FRONT = np.array([math.cos(AZ), 0., math.sin(AZ)])
UP = np.array([0., 1., 0.])
OBLIQUE = math.radians(38.)
ROT = np.column_stack((math.cos(OBLIQUE) * RIGHT - math.sin(OBLIQUE) * FRONT,
                       math.sin(OBLIQUE) * RIGHT + math.cos(OBLIQUE) * FRONT, -UP))
ICE = np.array([.73, .9, 1.], np.float32)
SHOTS = {'brink': (2960, 3200), 'vision': (3200, 3440), 'gap': (3440, 3520)}
CLOSE = 3232
SEAM_END = 3264
FORMED = 3296
PUSH = 3344
LAST_EYE = 3439
FIRST_GAP = 3440


class CircularInscription:
    """Carry the same attested script line round the vision's complete band."""
    def __init__(self):
        import d_inscription
        self.source = d_inscription.inscription()
        self.lv = self.source.lv
        # The opening shot stops at the ink midpoint of its final glyph. Its
        # texture contains the rest; use the actual complete ink bounds here.
        bounds = []
        for face, sign in (('outer', -1.), ('inner', 1.)):
            strip = self.lv[face][0]
            columns = np.flatnonzero(strip.max(axis=0) > 1e-6)
            anchor = d_inscription.TH0 / (2. * math.pi)
            theta = anchor + (sign * (columns + .5) / strip.shape[1] - anchor) % 1.
            bounds.append((theta.min() - 1. / strip.shape[1],
                           theta.max() + 1. / strip.shape[1]))
        self.offset = min(b[0] for b in bounds)
        self.extent = max(b[1] for b in bounds) - self.offset

    def sample(self, face, u, v, footprint):
        sign = -1. if face == 'outer' else 1.
        phase = (sign * u) % 1.
        mapped = sign * (self.offset + self.extent * phase)
        return self.source.sample(face, mapped, v, footprint * self.extent)


@lru_cache(maxsize=1)
def inscription():
    return CircularInscription()


def state(f):
    """Closure is conditional; the full Eye reads for 48 frames before approach."""
    if not any(a <= f < b for a, b in SHOTS.values()):
        raise ValueError('D brink/vision/gap frame required')
    vision = 3200 <= f < 3440
    close = float(smootherstep(3202., CLOSE, f)) if vision else 0.
    form = float(smootherstep(3256., FORMED, f)) if vision else 0.
    opening = float(smootherstep(3284., FORMED, f)) if vision else 0.
    return dict(gap=GAP_DEGREES * (1. - close), closed=vision and f >= CLOSE,
                seam=float(smoothstep(CLOSE, SEAM_END, f)) if vision else 0.,
                eye=form, opening=opening,
                push=float(smootherstep(PUSH, LAST_EYE, f)) if vision else 0.,
                lean_leaders=.085 + .135 * float(smootherstep(3272., 3300., f)) if vision else
                (.085 if f < 3200 else 0.),
                lean_others=.04 + .11 * float(smootherstep(3294., 3320., f)) if vision else
                (.04 if f < 3200 else 0.))


def theta_range(f):
    half = math.radians(state(f)['gap']) / 2.
    return half, 2. * math.pi - half


def endpoints(f):
    th = np.array(theta_range(f))
    p = np.stack((RS.R_MID * np.cos(th), np.zeros(2), RS.R_MID * np.sin(th)), 1)
    return RING_C + RING_SIZE * p @ ring_rotation(f).T


def _camera(distance, target, hfov=46., direction=FRONT):
    target = np.asarray(target, np.float64)
    return Camera(target + distance * direction, target, hfov=hfov,
                  focus=distance, aperture=.012)


def aperture_bounds(cam):
    """Intersection in x, open interval in y between the two cut faces (pixels)."""
    boxes = []
    half = math.radians(GAP_DEGREES) / 2.
    for theta in (half, 2. * math.pi - half):
        p, _, _ = RS.cap_points(theta, 256)
        world = RING_C + RING_SIZE * p.reshape(-1, 3) @ ROT.T
        u, v, _ = cam.project(world, 1920, 804)
        boxes.append((float(u.min()), float(v.min()), float(u.max()), float(v.max())))
    lower, upper = sorted(boxes, key=lambda b: b[1], reverse=True)
    return np.array([max(lower[0], upper[0]), upper[3], min(lower[2], upper[2]), lower[1]])


@lru_cache(maxsize=1)
def gap_camera():
    target = endpoints(FIRST_GAP).mean(axis=0)
    direction = math.cos(math.radians(20.)) * FRONT + math.sin(math.radians(20.)) * RIGHT
    # Translation in the image plane preserves verticality and converges the
    # cap-bound aperture (rather than just its two centres) to screen centre.
    for _ in range(5):
        cam = _camera(30. * RING_SIZE / 3.5, target, direction=direction)
        b = aperture_bounds(cam)
        delta = (b[:2] + b[2:]) / 2. - np.array([959.5, 401.5])
        z = float((endpoints(FIRST_GAP).mean(axis=0) - cam.pos) @ cam.R[2])
        target = target + cam.R[0] * delta[0] * z / cam.f_px(1920) - UP * delta[1] * z / cam.f_px(1920)
    return _camera(30. * RING_SIZE / 3.5, target, direction=direction)


def slit_profile(y, half_height=.88):
    """Parallel cut edges with short flat chamfers; no organic pupil noise."""
    return np.clip((half_height - np.abs(y)) / (.12 * half_height), 0., 1.)


@lru_cache(maxsize=1)
def match_geometry():
    b = aperture_bounds(gap_camera())
    width, height = b[2:] - b[:2]
    hh = .88
    radius_px = height / (2. * hh)
    radius_world = RS.R_IN * RING_SIZE
    fp = _camera(1., RING_C).f_px(1920)
    distance = fp * radius_world / radius_px
    # The eye's horizontal axis is oblique. At Y=0 perspective is symmetric
    # only about X=0 for a face-on disc, so final Eye uses camera-facing basis.
    width_unit = width / (2. * radius_px)
    return dict(bounds=b.tolist(), centre=((b[:2] + b[2:]) / 2.).tolist(),
                width=float(width), height=float(height), half_height=hh,
                half_width=float(width_unit), radius_px=float(radius_px),
                eye_distance=float(distance))


def camera(f):
    if f >= FIRST_GAP:
        pull = float(smootherstep(FIRST_GAP, 3520., f))
        close = gap_camera()
        wide = _camera(115., RING_C - 13. * UP)
        pos = close.pos * (1 - pull) + wide.pos * pull
        tgt = close.target * (1 - pull) + wide.target * pull
        return Camera(pos, tgt, hfov=46., focus=np.linalg.norm(pos - tgt), aperture=.012)
    if f < 3200:
        q = float(smootherstep(2960., 3199., f))
        return _camera(135. - 20. * q, RING_C - (15. - 2. * q) * UP)
    q = float(smootherstep(PUSH, LAST_EYE, f))
    formation = float(smootherstep(3200., FORMED, f))
    drift = float(smootherstep(FORMED, PUSH, f))
    return _camera((115. - 2. * formation - .6 * drift) * (1 - q)
                   + match_geometry()['eye_distance'] * q,
                   RING_C - (13. - .1 * formation - .1 * drift) * (1 - q) * UP)


@lru_cache(maxsize=1)
def race_rotation():
    """Same physical pose as c_d.Scene('race'), expressed with gap centre zero."""
    az = c3.AZ0 + .14
    reference_pos = np.array([79. * math.cos(az), c_d.B.GROUND + 48., 79. * math.sin(az)])
    base = c3._closed_ring_frame(1580.)[0]
    source = openring_d.placement(base, reference_pos, RING_C, 2520.)
    theta = openring_d.GAP_CENTRE
    roll = np.array([[math.cos(theta), 0., -math.sin(theta)],
                     [0., 1., 0.], [math.sin(theta), 0., math.cos(theta)]])
    return source @ roll


def ring_rotation(f):
    if f < 3200:
        q = float(smootherstep(2960., 3199., f))
        # Orthogonal projection of the blend retains a rigid, full-width band.
        u, _, vt = np.linalg.svd((1. - q) * race_rotation() + q * ROT)
        return u @ vt
    q = float(smootherstep(PUSH, LAST_EYE, f)) if f < FIRST_GAP else 0.
    angle = OBLIQUE * (1 - q)
    return np.column_stack((math.cos(angle) * RIGHT - math.sin(angle) * FRONT,
                            math.sin(angle) * RIGHT + math.cos(angle) * FRONT, -UP))


def ring_layer(f, cam, W, H):
    stt = state(f)
    th0, th1 = theta_range(f)
    st = RS.RingState()
    st.end_caps = True
    st.worked_caps = True
    st.inscription = inscription()
    st.hammer = .16
    st.glow = .085
    st.letters = .65 + .8 * stt['seam']
    st.letters_col = np.array([1., .85, .47])
    pulse = math.exp(-((f - 2960) % 20) / 4.)
    seam_th = (2. * math.pi * stt['seam']) % (2. * math.pi)
    def heat(th):
        d = np.minimum(np.abs(th - th0), np.abs(th1 - th))
        hot = (.61 + .32 * pulse) * np.exp(-d / .19)
        if stt['closed']:
            dist = np.abs((th - seam_th + math.pi) % (2. * math.pi) - math.pi)
            hot = .22 + .77 * np.exp(-(dist / .12) ** 2) * (1 - stt['eye'])
        return hot
    st.heat = heat
    def write(th):
        # Only the material inside the original 55-degree missing interval
        # kindles under the travelling weld; existing lettering stays awake.
        old_half = math.radians(GAP_DEGREES) / 2.
        missing = (th < old_half) | (th > 2. * math.pi - old_half)
        if not stt['closed']:
            passed = np.zeros_like(th)
        elif stt['seam'] >= 1.:
            passed = np.ones_like(th)
        else:
            passed = smoothstep(-.035, .055, 2. * math.pi * stt['seam'] - th)
        return np.where(missing, passed, 1.)
    st.write = write
    env = RS.Env(above=(.18, .19, .21), horizon=(.42, .35, .23), below=(.10, .07, .035))
    env.point(RING_C - 26. * UP, np.array([32., 24., 12.]), 9.)
    env.lobe(FRONT - UP * .5, np.array([.85, .88, .92]), 5.)
    rgb, a, depth = RS.render(cam, W, H, ring_rotation(f), RING_C, RING_SIZE, st, env,
                              th_range=(th0, th1), ss=2)
    if 3200 <= f < FIRST_GAP:
        # The conditional Eye keeps the band in a gold/white palette; the
        # working forge's red heat ramp remains available outside this shot.
        rgb[..., 1] = np.maximum(rgb[..., 1], .75 * rgb[..., 0])
        rgb[..., 2] = np.maximum(rgb[..., 2], .30 * rgb[..., 0])
    # A white weld travels over the extant metal, preserving its material detail.
    if stt['closed'] and stt['seam'] < 1.:
        white = np.clip(rgb.max(axis=2) - .8, 0., None)
        rgb = rgb + white[..., None] * ICE * .65
    return rgb, a, depth


def glare_layer(f, cam, W, H, frame=None):
    """The adjacent race's aperture-confined glare, pulsing only on the brink."""
    if not 2960 <= f < 3200:
        return np.zeros((H, W, 3), np.float32)
    caps = np.concatenate([RS.cap_points(theta, 24)[0][1] for theta in theta_range(f)])
    caps = RING_C + RING_SIZE * caps @ ring_rotation(f).T
    ends = endpoints(f)
    visibility = None
    if frame is not None:
        depth = float(np.min((ends - cam.pos) @ cam.R[2]))
        visibility = c3.occ_vis(frame, depth - .65, H, W)
    return d_glare.render(cam, W, H, ends, caps, openring_d.glare(f),
                          math.exp(-((f - 2960.) % 20.) / 4.), visibility)


@lru_cache(maxsize=1)
def glyph_bank():
    """The exact attested characters written on the band, one per atlas script.

    Masks preserve source point positions and aspect ratios. They are never
    procedurally drawn letters; no random marks supplement the source atlas.
    """
    atlas = glyphs3.load_atlas()
    source = inscription().source
    masks = []
    for idx in source.indices:
        pts = atlas['pts'][atlas['off'][idx]:atlas['off'][idx] + atlas['cnt'][idx]].copy()
        pts -= (pts.min(axis=0) + pts.max(axis=0)) / 2.
        span = np.maximum(np.ptp(pts, axis=0), 1e-6)
        scale = 88. / max(span)
        size = np.maximum(np.ceil(span * scale).astype(int) + 8, 9)
        mask = np.zeros((size[1], size[0]), np.float32)
        xx = np.clip(np.rint(pts[:, 0] * scale + (size[0] - 1) / 2.).astype(int), 0, size[0] - 1)
        yy = np.clip(np.rint(-pts[:, 1] * scale + (size[1] - 1) / 2.).astype(int), 0, size[1] - 1)
        np.add.at(mask, (yy, xx), 1.)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.)
        mask /= max(float(np.quantile(mask[mask > 0.], .92)), 1e-9)
        masks.append(np.clip(mask, 0., 1.))
    return dict(masks=tuple(masks), categories=tuple(atlas['cats'][int(atlas['cat'][i])] for i in source.indices),
                indices=tuple(source.indices), texts=tuple(source.texts))


def glyph_streams(f, late_slit=False):
    """Material-plane positions for 37 strands flowing from rim toward slit."""
    bank = glyph_bank()
    count = len(bank['categories'])
    strand, rank = np.meshgrid(np.arange(count), np.arange(12), indexing='ij')
    theta = 2. * np.pi * (strand.ravel() + .35) / count
    travel = (rank.ravel() / 12. + (f - 3256.) * .0048) % 1.
    hh, hw = match_geometry()['half_height'], match_geometry()['half_width']
    start = np.column_stack((.965 * np.cos(theta), .965 * np.sin(theta)))
    opening = float(smootherstep(PUSH, PUSH + 36., f)) if late_slit else state(f)['opening']
    finish = np.column_stack((np.where(np.cos(theta) >= 0., 1., -1.) * (hw + .05) * opening,
                              .78 * hh * np.sin(theta)))
    # A small directional bend joins the rim to the two straight cut edges.
    xy = start * (1. - travel[:, None]) + finish * travel[:, None]
    xy[:, 1] += .04 * np.sin(theta * 3.) * np.sin(np.pi * travel)
    category = (strand.ravel() + 7 * rank.ravel()) % count
    intensity = (.68 + .32 * np.cos(theta + f * .025) ** 2)
    intensity *= smoothstep(0., .045, travel) * (1. - smoothstep(.94, 1., travel))
    intensity *= 1. - smoothstep(state(f)['eye'] - .055, state(f)['eye'] + .025, travel)
    return xy, category, intensity


def glyph_canvas(f, size=1536, late_slit=False):
    """Render discrete writing into a flat emissive plane, leaving black gaps."""
    bank = glyph_bank()
    xy, categories, intensity = glyph_streams(f, late_slit=late_slit)
    ink = np.zeros((size, size), np.float32)
    for point, cat, strength in zip(xy, categories, intensity):
        if strength <= 0.:
            continue
        mask = bank['masks'][cat]
        extent = max(5, int(round(.061 * size / 2.)))
        width = max(2, int(round(extent * mask.shape[1] / max(mask.shape))))
        height = max(2, int(round(extent * mask.shape[0] / max(mask.shape))))
        glyph = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
        centre = (point * [1., -1.] + 1.) * ((size - 1) / 2.)
        x, y = np.rint(centre - [width / 2., height / 2.]).astype(int)
        x0, x1, y0, y1 = max(x, 0), min(x + width, size), max(y, 0), min(y + height, size)
        if x1 > x0 and y1 > y0:
            ink[y0:y1, x0:x1] += glyph[y0-y:y1-y, x0-x:x1-x] * strength
    return ink


def eye_layer(f, cam, W, H, late_slit=False):
    """Ice-white writing from the Ring; a black manufactured aperture at centre."""
    st = state(f)
    formed = st['eye']
    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    if formed <= 0.:
        return rgb, alpha, alpha.copy()
    rotation = ring_rotation(f)
    n = rotation[:, 1]
    ey = UP
    ex = np.cross(ey, n)
    ex /= np.linalg.norm(ex)
    X, Y, ok = c2.plane_coords(cam, W, H, RING_C, ex, ey, n, RS.R_IN * RING_SIZE)
    canvas = glyph_canvas(f, late_slit=late_slit)
    size = canvas.shape[0]
    ink = cv2.remap(canvas, ((X + 1.) * (size - 1.) / 2.).astype(np.float32),
                    ((1. - Y) * (size - 1.) / 2.).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    radius = np.hypot(X, Y)
    disk = (1. - smoothstep(.985, 1., radius)) * ok
    front = 1. - 1.12 * formed
    flood = smoothstep(front - .06, front + .04, radius) * disk
    opening = float(smootherstep(PUSH, PUSH + 36., f)) if late_slit else st['opening']
    match = match_geometry()
    edge = match['half_width'] * opening * slit_profile(Y, match['half_height'])
    aa = 1. / max(cam.f_px(W) * RS.R_IN * RING_SIZE / np.linalg.norm(cam.pos - RING_C), 1.)
    void = (1. - smoothstep(edge - aa, edge + aa, np.abs(X)))
    void *= (1. - smoothstep(match['half_height'] - aa, match['half_height'] + aa, np.abs(Y)))
    void *= opening * disk
    # The iris has no skin, fibres, veins, corneal highlight or shader noise.
    rgb = ink[..., None] * ICE * (3.1 * flood * (1. - void))[..., None]
    alpha = flood * .94
    alpha = np.maximum(alpha, void)
    return rgb.astype(np.float32), alpha.astype(np.float32), void.astype(np.float32)


_scene = None


def render_frame(f, scale=1., outdir=None, save=True, verbose=True, late_slit=False):
    global _scene
    started = time.perf_counter()
    st = state(f)
    if not 0. < scale <= 1.:
        raise ValueError('scale must be in (0, 1]')
    if save:
        if outdir is None:
            raise ValueError('Explicit new output directory required')
        from render_d import output_path
        shot = next(name for name, (a, b) in SHOTS.items() if a <= f < b)
        outdir = output_path(shot, outdir)
    import d_forges
    if _scene is None:
        _scene = d_forges.Scene(phase='race', world_frame=2959.)
    cam = camera(float(f))
    ctx = d_forges.make_context(f, scale, cam)
    W, H = ctx.fr.W, ctx.fr.H
    lean = {i: st['lean_leaders'] if i in d_forges.LEADERS else st['lean_others']
            for i in range(_scene.towers.k_all)}
    with _scene.world(lean=lean, beacon_power=0.):
        _scene.prepare(ctx)
        ring, alpha, depth = ring_layer(f, cam, W, H)
        before = RS.merge_occluder(ctx.fr, alpha, depth)
        ring *= RS.visibility(before, depth, H, W)[..., None]
        _scene.emit(ctx, RING_C, np.array([1., .65, .31]), 90.)
        if f < 3200 or f >= FIRST_GAP:
            _scene.hammer_sparks(ctx, endpoints(f))
            _scene.gold(ctx, (ring_rotation(f), RING_C, RING_SIZE), theta_range(f))
        hdr = ctx.fr.resolve() + glare_layer(f, cam, W, H, ctx.fr)
    eye, a, void = eye_layer(f, cam, W, H, late_slit=late_slit)
    hdr = hdr * (1 - a[..., None]) + eye + ring
    if f < FIRST_GAP:
        y = np.arange(H, dtype=np.float32)[:, None, None] / H
        hdr *= 1 - .72 * smoothstep(.74, .92, y)
    img = look.finish(hdr, exposure=.88, bloom_strength=.15, bloom_threshold=.8,
                      streak_strength=.015, vignette_amount=.12)
    if save:
        outdir.mkdir(parents=True, exist_ok=True)
        look.save_png(look.frame_path(str(outdir), f), img)
    if verbose:
        print(f'D {f}: {time.perf_counter() - started:.6f}s', flush=True)
    return img
