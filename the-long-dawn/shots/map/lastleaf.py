"""Opt-in last written leaf, on C22's camera and resting-pen spread.

Local frames 0..239. No accepted shot or EDL dispatch is changed. The left
leaf holds the open Ring; the right remains unwritten unless --title is set.
Two shadowed light passes retain the hearth at frame left and admit a lower,
rose dawn from that side (the incoming C20 illumination also enters left).
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

for _pool in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_pool, '1')

import cv2
import numpy as np
import book_c as BC
import book as B
import book_c_v5 as V
import book_c_v5_pen_candidates as C
import ringpage as RP
import v5_pen as PEN

cv2.setNumThreads(0)
FRAMES = 240
HEARTH_POS = (-55., 50., 30.)
DAWN_POS = (-90., 38., 13.)
PLATE_DAWN_POS = (-48., -80., 27.)
PLATE_REFLECTION_POS = (-22.5463, 50.8598, 36.8330)


def light_state(t):
    """Continuous envelopes; seconds 0..10. Hearth retains a living ember."""
    q = BC.smooth(t / 10.)
    return 1.9 * (1. - .77 * q), .75 + 4.75 * q


def plate_light_state(t):
    q = BC.smooth(t / 10.)
    return 1.35 * (1. - .82 * q), 1.0 + 3.8 * q


class LastLeaf(C.CandidatePagesV5):
    def __init__(self, W=1920, H=804, ppc=70, gap_degrees=30., title=False, plate=False):
        super().__init__('soft_spine_metal', W, H, ppc)
        self.gap_degrees = RP.validate_gap(gap_degrees)
        self.with_title = bool(title)
        self.plate = bool(plate)

    def layout(self, t):
        bk = self.book(2.8, 1.2, seed=13)
        target = np.array([.6, 1.9 if self.plate else .2, 2.0])
        cam = B.Cam(target + np.array([-.5, -39., 39.]) *
                    (1.09 if self.plate else 1.) * (1. - .025 * BC.smooth(t / 10.)),
                    target, 38., self.W, self.H)
        cam.dof_k = 2.
        return bk, cam

    def leaf(self):
        if self.plate:
            return self.once('last_written_plate', lambda: RP.RingPlate(
                ppc=self.ppc, gap_degrees=self.gap_degrees).texture())
        return self.once('last_written_leaf', lambda: RP.RingLeaf(
            ppc=self.ppc, gap_degrees=self.gap_degrees,
            center=(10.2, 7.2), radius=3.3).texture())

    def title(self):
        from titleburn import TitleBurn
        # t_out beyond the shot holds the cooled ink through the last frame.
        return self.once('last_title', lambda: TitleBurn(3.8, 12., ppc=self.ppc))

    def _pass(self, bk, cam, light, left, right, t, *, dawn=False, xl=None):
        q = BC.smooth(t / 10.)
        kw = dict(amb=(0., 0., 0.), env=(0., 0., 0.),
                  fill=((.2, -.1, 1.), (.026*q, .025*q, .040*q))) if dawn else {}
        if self.plate:
            kw['burnished_gilt'] = True
            if dawn:
                kw['fill'] = ((.2, -.1, 1.), (.060*q, .050*q, .064*q))
                # A weak rose room reflection catches the smooth leaf from
                # beyond the book. Its power follows the dawn; ink masks it.
                strength = plate_light_state(t)[1] / 4.8
                kw['burnished_reflection'] = (*PLATE_REFLECTION_POS,
                                              .024*strength, .0204*strength,
                                              .0204*strength, 3.)
        hdr, alpha, G = B.render(bk, cam, light, left, right, t,
                                 xlights=xl, fire_k=float(xl is not None), **kw)
        if not dawn:
            hdr = C.soften_spine(hdr, G, bk, light)
        if self.plate:
            import lastleaf_pen as resting_pen
        else:
            resting_pen = PEN
        mesh = self.once('physical_pen', lambda: resting_pen.geometry(bk))
        hdr, depth = resting_pen.composite(hdr, G, bk, cam, light, mesh)
        hdr = C.replace_visible_nib(hdr, G, depth, mesh, cam, light)
        if dawn:
            # The candidate nib has a fixed room reflection. Keep it once,
            # in the hearth pass; all other pen terms are linear in light.col.
            dark = B.Light(light.pos, power=0., radius=light.radius)
            reflected = C.replace_visible_nib(np.zeros_like(hdr), G, depth,
                                              mesh, cam, dark)
            hdr -= reflected
        return hdr, alpha, depth

    def frame(self, f):
        if isinstance(f, bool) or not isinstance(f, (int, np.integer)) or not 0 <= f < FRAMES:
            raise ValueError('last leaf accepts local integer frames 0..239')
        t = f / 24.
        bk, cam = self.layout(t)
        hp, dp = plate_light_state(t) if self.plate else light_state(t)
        hearth = self.light(HEARTH_POS, hp, t, seed=5, amt=.14,
                           col=(1., .52, .24), radius=10.)
        dawn = B.Light(PLATE_DAWN_POS if self.plate else DAWN_POS,
                       col=(1., .68, .75) if self.plate else (1., .62, .82),
                       power=dp, radius=2.4 if self.plate else 1.3)
        left = self.leaf()
        title = self.title() if self.with_title else None
        right = title.texture(t) if title else self.blank
        xl = title.lights(t, bk) if title else None
        hdr, alpha, depth = self._pass(bk, cam, hearth, left, right, t, xl=xl)
        # Emission belongs in one pass. The additive dawn uses only the cold
        # channels while the title is burning, then restores the live texture.
        if title:
            fire = right.chan[..., 4].copy()
            right.chan[..., 4] = 0.
            right.build()
        daylight, _, _ = self._pass(bk, cam, dawn, left, right, t, dawn=True)
        if title:
            right.chan[..., 4] = fire
            right.build()
        hdr += daylight
        hdr = B.dof(hdr, depth.astype(np.float32), cam, cam.dof_k * self.W / 1920.)
        if title:
            sparks = title.sparks(t, bk, cam, self.W, self.H)
            if sparks is not None:
                hdr += sparks
        return hdr, alpha


def make_renderer(open_leaf=False, scale=1., ppc=70, gap_degrees=30., title=False, plate=False):
    """Omitted opt-in returns the exact accepted soft-spine/metal renderer."""
    if not math.isfinite(scale) or not 0 < scale <= 1 or int(804 * scale) < 1:
        raise ValueError('scale must produce positive dimensions, at most native')
    if not isinstance(ppc, int) or isinstance(ppc, bool) or ppc <= 0:
        raise ValueError('ppc must be a positive integer')
    if not open_leaf:
        if title or plate:
            raise ValueError('title and plate require the open-leaf opt-in')
        return C.CandidatePagesV5('soft_spine_metal', int(1920 * scale), int(804 * scale), ppc)
    return LastLeaf(int(1920 * scale), int(804 * scale), ppc, gap_degrees, title, plate)


def output_dir(value):
    """A farm may own renders/; a lane's shared renders symlink is read-only."""
    path = Path(value).expanduser()
    shared = Path(__file__).resolve().parents[2] / 'renders'
    if shared.is_symlink():
        for destination in (path, path.with_name(path.name + '_matte')):
            if destination.resolve().is_relative_to(shared.resolve()):
                raise ValueError('shared renders is read-only; choose a lane output directory')
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--open-leaf', action='store_true', required=True)
    ap.add_argument('--frames', required=True, help='local frames, inclusive range or comma list')
    ap.add_argument('--out', required=True, type=output_dir)
    ap.add_argument('--scale', type=float, default=1.)
    ap.add_argument('--ppc', type=int, default=70)
    ap.add_argument('--gap-degrees', type=float, default=30.)
    ap.add_argument('--title', action='store_true')
    ap.add_argument('--plate', action='store_true', help='Round 2 illustrated plate, burnished gilt, resting pen and cross-spread dawn')
    a = ap.parse_args(argv)
    fs = BC.frames_of(a.frames)
    if not fs or any(f < 0 or f >= FRAMES for f in fs):
        ap.error('frames must be 0..239')
    renderer = make_renderer(True, a.scale, a.ppc, a.gap_degrees, a.title, a.plate)
    matte = a.out.with_name(a.out.name + '_matte')
    for f in fs:
        start = time.perf_counter()
        hdr, alpha = renderer.frame(f)
        if not np.isfinite(hdr).all() or not np.isfinite(alpha).all():
            raise RuntimeError('non-finite last-leaf output')
        rgb = BC.look.finish(hdr, exposure=1.15, bloom_strength=.06,
                             bloom_threshold=1.2, vignette_amount=.32)
        V.save(a.out / f'f_{f:05d}.png', rgb, 'png')
        V.save(matte / f'f_{f:05d}.png', np.repeat(alpha[..., None], 3, -1), 'png')
        print(json.dumps(dict(frame=f, seconds=time.perf_counter()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                           (1 if sys.platform == 'darwin' else 1024),
            width=renderer.W, height=renderer.H, ppc=a.ppc)), flush=True)


if __name__ == '__main__':
    main()
