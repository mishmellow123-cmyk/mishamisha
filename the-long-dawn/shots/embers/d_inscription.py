"""Cut D1680–2079: live thinking fire drawn into a multilingual open band.

The explicit entry point is the opt-in. The final pose/camera are read directly
from the adjoining cut-D forging renderer; accepted A/C routes are unchanged.
"""
from functools import lru_cache
from pathlib import Path
import math
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'lib'), str(HERE)]

import cv2
import numpy as np

import c_d as FORGE
import d_thinking_fire as FIRE
import glyphs3
import look
import ringsolid as RS
from core import Camera

START, FORM_START, STOP, END = 1680, 1760, 1952, 2080
PULL_START, PULL_END = STOP, 2064
TH0 = float(FORGE.D.theta_range(END)[0])
EXTENT = math.radians(240.)
CENTRE = FORGE.RING_C - np.array([0., 8., 0.])
# Projected source illustration centre in Mountain C559, including book cockle.
PAGE_RING = np.array([997.5120369856625, 90.54896042341699])
PAGE_RING_WIDTH = 59.210102030642474


def ease(a, b, x):
    q = np.clip((np.asarray(x) - a) / (b - a), 0., 1.)
    return q * q * q * (q * (q * 6. - 15.) + 10.)


def theta_range(f):
    """Fixed trailing end and one moving front; hold exactly at a 120° gap."""
    return TH0, TH0 + .018 + (EXTENT - .018) * float(ease(FORM_START, STOP, f))


def gap_deg(f):
    a, b = theta_range(f)
    return 360. - math.degrees(b - a)


def _basis(theta):
    return (np.array([math.cos(theta), 0., math.sin(theta)]),
            np.array([-math.sin(theta), 0., math.cos(theta)]))


@lru_cache(maxsize=1)
def handoff_scene():
    """Read scene geometry without initializing towers or any disk cache."""
    scene = object.__new__(FORGE.Scene)
    scene.shot = 'forging'
    reference = FORGE.RACE_START + 120.
    scene.rotation = FORGE.D.placement(FORGE.c3._closed_ring_frame(1580.)[0],
                                      scene.camera(reference).pos, FORGE.RING_C, reference)
    return scene


def ring_frame(f):
    return handoff_scene().ring_frame(END)


def camera(f):
    """The lens rides the writing edge, then withdraws as the edge arrests."""
    _, theta = theta_range(f)
    radial, tangent = _basis(theta)
    rot, centre, size = ring_frame(f)
    edge = centre + size * (RS.R_OUT * radial) @ rot.T
    near_target = edge + size * (-.24 * tangent - np.array([0., .16, 0.])) @ rot.T
    near_pos = edge + size * (2.65 * radial + .85 * tangent + np.array([0., .58, 0.])) @ rot.T
    far = handoff_scene().camera(END)
    if f >= PULL_END:
        return far
    p = float(ease(PULL_START, PULL_END, f))
    pos = near_pos * (1 - p) + far.pos * p
    target = near_target * (1 - p) + far.target * p
    return Camera(pos, target, up=rot[:, 1] * (1 - p) + np.array([0., 1., 0.]) * p,
                  hfov=42. * (1 - p) + far.hfov * p,
                  focus=np.linalg.norm(edge - pos) * (1 - p) + far.focus * p,
                  aperture=.016 * (1 - p) + far.aperture * p)


class AtlasInscription(RS.Inscription):
    """One row, one attested atlas item per category, plus a final half glyph.

    Glyph points come directly from glyphs3's committed font atlas. The strip
    is material-fixed: writing reveals it by adding metal, never by scrolling
    the texture. Its last glyph's ink-mass median lies on the final cut plane.
    """

    def __init__(self):
        atlas = glyphs3.load_atlas()
        cats = atlas['cats']
        self.categories = tuple(cats)
        self.indices = []
        for ci, cat in enumerate(cats):
            choices = np.flatnonzero(atlas['cat'] == ci)
            choices = [int(i) for i in choices if atlas['cnt'][i] > 0]
            if not choices:
                raise ValueError(f'Atlas category has no ink: {cat}')
            # A short item leaves room for every script in a single line.
            self.indices.append(min(choices, key=lambda i: (len(atlas['text'][i]), i)))
        final = atlas['text'].index('光')
        self.indices.append(final)
        self.texts = tuple(atlas['text'][i] for i in self.indices)
        self.lv = {}
        self.leading = {}
        h, w = 128, 4096
        pitch = (EXTENT - .07) / (len(self.indices) - 1)
        self.centres = np.linspace(TH0 + .07, TH0 + EXTENT, len(self.indices))
        self.final_ink_fraction = None
        for face, direction in (('outer', -1.), ('inner', 1.)):
            img = np.zeros((h, w), np.float32)
            for j, (idx, theta) in enumerate(zip(self.indices, self.centres)):
                pts = atlas['pts'][atlas['off'][idx]:atlas['off'][idx] + atlas['cnt'][idx]].copy()
                ink_width = float(np.ptp(pts[:, 0]))
                ink_height = float(np.ptp(pts[:, 1]))
                size = min(.31 / max(ink_height, 1e-6),
                           pitch * RS.R_OUT * .78 / max(ink_width, 1e-6))
                px = pts[:, 0] * size
                py = pts[:, 1] * size
                if j == len(self.indices) - 1:
                    px -= np.median(px)
                    self.final_ink_fraction = float(np.mean(px <= 0.))
                    self.leading[face] = ((direction * theta / (2. * np.pi)) % 1.,
                                          float(np.ptp(px)) * 1.18)
                u = direction * theta / (2. * np.pi) + px / (2. * np.pi * RS.R_OUT)
                v = .5 - py
                xx = np.rint((u % 1.) * w).astype(int) % w
                yy = np.clip(np.rint(v * (h - 1)).astype(int), 0, h - 1)
                np.add.at(img, (yy, xx), 1.)
            # The atlas samples filled font masks. A subpixel reconstruction
            # joins their points into the same strokes, without drawing new text.
            img = cv2.GaussianBlur(img, (0, 0), .90)
            img = np.clip(img * 1.45, 0., 1.)
            lv = [img]
            while lv[-1].shape[0] > 8:
                old = lv[-1]
                lv.append(cv2.resize(old, (old.shape[1] // 2, old.shape[0] // 2),
                                     interpolation=cv2.INTER_AREA))
            self.lv[face] = lv


@lru_cache(maxsize=1)
def inscription():
    return AtlasInscription()


class FrontInscription:
    """Use the edge-contrast hook without moving any material-space lettering."""

    def __init__(self, texture, theta):
        self.texture, self.theta = texture, theta
        self.lv = texture.lv

    def sample(self, face, u, v, texels_per_sample):
        return self.texture.sample(face, u, v, texels_per_sample)

    def leading_glyph(self, face):
        sign = -1. if face == 'outer' else 1.
        # ringsolid adds sign * (sample_theta - leading_theta) / tau to this.
        # Cancellation therefore gives the original material UV everywhere.
        return (sign * self.theta / (2. * np.pi)) % 1., .29


def ring_state(f):
    st = RS.RingState()
    st.end_caps = True
    st.cap_heat_gain = 20.
    st.leading_glyph = True
    st.inscription = FrontInscription(inscription(), theta_range(f)[1])
    st.letters = 1.25
    st.letters_col = np.array([1., .25, .024])
    st.glow = .065
    st.hammer = .24
    a, b = theta_range(f)

    def heat(theta):
        x = np.asarray(theta)
        lead = np.exp(-((b - x) / .028) ** 2)
        tail = .94 * np.exp(-((x - a) / .026) ** 2)
        wake = .13 * np.exp(-np.maximum(b - x, 0.) / .30)
        return np.clip(np.maximum(lead, tail) + wake, 0., 1.)

    st.heat = heat
    st.write = lambda theta: np.ones_like(theta)
    return st


class ForgeEnvironment(RS.Env):
    """Polished metal reflects long heated sources with a narrow azimuth."""

    def radiance(self, reflected, points):
        light = super().radiance(reflected, points)
        azimuth = np.arctan2(reflected[:, 2], reflected[:, 0])
        strips = np.zeros(len(points))
        for az in (.22, 2.08, 3.48):
            strips += np.exp(480. * (np.cos(azimuth - az) - 1.))
        height = reflected[:, 1]
        broken_source = (.16 + .84 * np.exp(-((height + .32) / .12) ** 2)
                         + .48 * np.exp(-((height - .19) / .09) ** 2))
        strips *= broken_source
        return light + strips[:, None] * np.array([2.3, 1.65, .62])


def environment(f):
    env = ForgeEnvironment(above=(.008, .006, .003), horizon=(.14, .063, .012), below=(.07, .025, .003))
    env.point((0., -.45, 0.), (3.4, 1.6, .32), .70)
    env.lobe((-.2, 1., .3), (.40, .27, .10), 7.)
    env.lobe((.7, .2, -.6), (1.1, .76, .28), 48.)
    return env


def _depth_of_field(rgb, alpha, depth, focus, scale, strength):
    """Composite two defocus levels without smearing the sharp cut section."""
    if strength <= 0:
        return rgb, alpha
    distance = np.where(np.isfinite(depth), np.abs(depth - focus), 0.)
    k = (ease(.20, 1.2, distance) * strength).astype(np.float32)
    sigma = max(2.1 * scale, .4)
    blurred_rgb = cv2.GaussianBlur(rgb, (0, 0), sigma)
    blurred_a = cv2.GaussianBlur(alpha, (0, 0), sigma)
    # Blur the weighting too so defocused silhouettes gain an outside halo.
    k = np.maximum(k, cv2.GaussianBlur(k, (0, 0), sigma))
    return rgb * (1 - k[..., None]) + blurred_rgb * k[..., None], alpha * (1 - k) + blurred_a * k


def preroll_fire(f):
    # Place the bright body at the source ring centre, not the flame root.
    close = float(ease(START, FORM_START - 1, f))
    height = PAGE_RING_WIDTH * 3.55 * (1 - close) + 650. * close
    centre = PAGE_RING * (1 - close) + np.array([960., 402.]) * close
    root = centre + np.array([0., .402 * height])
    return root, root - np.array([0., height])


def front_fire(f):
    cam = camera(f)
    rot, centre, size = ring_frame(f)
    radial, tangent = _basis(theta_range(f)[1])
    edge = centre + size * (RS.R_OUT * radial) @ rot.T
    points = np.stack([edge - .36 * size * rot[:, 1], edge + .62 * size * rot[:, 1]])
    x, y, z = cam.project(points, 1920, 804)
    root, tip = np.array([x[0], y[0]]), np.array([x[1], y[1]])
    entry = float(ease(FORM_START, FORM_START + 40, f))
    old_root, old_tip = preroll_fire(FORM_START)
    return old_root * (1 - entry) + root * entry, old_tip * (1 - entry) + tip * entry


def draw_fire(hdr, f, scale=1., alpha=None):
    if f < FORM_START:
        root, tip = preroll_fire(f)
        return FIRE.draw(hdr, root, tip, f, scale=scale)
    # The front itself carries A's ice-white fire; its gold rim sheds motes.
    root, tip = front_fire(f)
    active = 1. - float(ease(STOP - 22., STOP, f))
    FIRE.draw(hdr, root, tip, f, bright=active, scale=scale)
    # The same living source remains below the band as the lens withdraws.
    pull = float(ease(PULL_START, PULL_END, f))
    if pull > 0:
        points = np.stack([FORGE.c3.FIRE_ROOT, FORGE.c3.FIRE_ROOT + [0., FORGE.c3.HF, 0.]])
        x, y, z = camera(f).project(points, 1920, 804)
        if min(z) > 0:
            FIRE.draw(hdr, (x[0], y[0]), (x[1], y[1]), f, bright=1.1 * pull,
                      scale=scale, vis=None if alpha is None else 1. - alpha)
    return hdr


def _output_dir(outdir):
    if outdir is None:
        raise ValueError('Pass an explicit new output directory for this opt-in shot')
    requested = Path(outdir).absolute()
    dest = requested.resolve()
    renders = ROOT / 'renders'
    accepted = renders.resolve()
    if renders.is_symlink() and (requested == renders or renders in requested.parents
                                or dest == accepted or accepted in dest.parents):
        raise ValueError('The shared renders symlink is read-only')
    # Only the new shot name is allowed below renders; never an accepted plate. Judge the path as requested as well as
    # resolved: on the owner's machine renders/ is a real folder whose accepted stems (renders/embers_A3, ...) are
    # symlinks into the farm's store, so a resolved-only check lets them through.
    if (requested == renders or renders in requested.parents) and requested != renders / 'embers_D_inscription':
        raise ValueError('Output would enter an accepted renders directory')
    if dest == accepted or (accepted in dest.parents and dest != accepted / 'embers_D_inscription'):
        raise ValueError('Output would enter an accepted renders directory')
    return dest


def finish_opts(f):
    """Join c_d.Scene.frame's display grade by the sixteen-frame handoff."""
    pull = float(ease(PULL_START, PULL_END, f))
    old = dict(exposure=.79, bloom_strength=.035, bloom_threshold=1.2,
               streak_strength=.004, vignette_amount=.12, lift=.001)
    final = dict(exposure=1.05, bloom_strength=.08, bloom_threshold=1.1,
                 streak_strength=0., vignette_amount=.20, lift=.004)
    return {key: value * (1 - pull) + final[key] * pull for key, value in old.items()}


def render_frame(f, scale=1., outdir=None, save=True, verbose=True):
    """Return finished float32 display RGB; saving requires an explicit directory."""
    if not START <= f < END:
        raise ValueError(f'Inscription frame outside [{START}, {END}): {f}')
    if not 0 < scale <= 1.:
        raise ValueError('Scale must be greater than zero and no larger than native')
    dest = _output_dir(outdir) if save else None
    started = time.perf_counter()
    w, h = round(1920 * scale), round(804 * scale)
    hdr = np.zeros((h, w, 3), np.float32)
    alpha = None
    if f >= FORM_START:
        cam = camera(f)
        rgb = alpha = depth = None
        for sample in (max(float(f) - .20, FORM_START), min(float(f) + .20, END - 1)):
            rot, centre, size = ring_frame(sample)
            color, cover, z = RS.render(camera(sample), w, h, rot, centre, size,
                                        ring_state(sample), environment(sample),
                                        th_range=theta_range(sample), ss=2, nt=1000, npp=64)
            if rgb is None:
                rgb, alpha, depth = color * .5, cover * .5, z
            else:
                rgb += color * .5
                alpha += cover * .5
                depth = np.minimum(depth, z)
        pull = float(ease(PULL_START, PULL_END, f))
        rgb, alpha = _depth_of_field(rgb, alpha, depth, cam.focus, scale, .45 * (1. - pull))
        entry = float(ease(FORM_START, FORM_START + 28., f))
        hdr += rgb * entry
        alpha *= entry
    draw_fire(hdr, f, scale, alpha)
    img = look.finish(hdr, **finish_opts(f))
    if not np.isfinite(img).all():
        raise ValueError(f'Non-finite inscription output at D{f}')
    if save:
        dest.mkdir(parents=True, exist_ok=True)
        # Deterministic lossless stills preserve the renderer output for inspection.
        output = dest / f'f_{int(f):05d}.png'
        if not cv2.imwrite(str(output), np.rint(np.clip(img, 0., 1.)[..., ::-1] * 255).astype(np.uint8),
                           [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise OSError(f'Could not save frame D{f}')
    if verbose:
        print(f'inscription D{f} scale={scale:g} seconds={time.perf_counter() - started:.6f}', flush=True)
    return img.astype(np.float32, copy=False)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('frames', help='Comma-separated D frame numbers')
    p.add_argument('--scale', type=float, default=1.)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    for frame in args.frames.split(','):
        render_frame(int(frame), args.scale, args.out)
