"""Explicit cut-D book shots; input and output frame numbers are D frames.

Accepted C shots and the Round 2 lastleaf entry point are unchanged. The old
fire is D4800..4959, last written leaf D6320..6639, and title D6880..7119.
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

import numpy as np
import book as B
import book_c as BC
import book_c_v5 as V
import book_c_v5_pen_candidates as C
import lastleaf as L
import lastleaf_pen as PEN


RANGES = {'oldfire': (4800, 4960), 'lastleaf': (6320, 6640), 'title': (6880, 7120)}
FPS = 24


def phase(shot, frame):
    """Inclusive first/last images sample both endpoints of the shot move."""
    start, end = RANGES[shot]
    if (isinstance(frame, bool) or not isinstance(frame, (int, np.integer))
            or not start <= frame < end):
        raise ValueError(f'{shot} accepts integer D frames {start}..{end-1}')
    return (frame-start)/(end-start-1)


def dawn_state(q):
    """Approved rose illumination, eased over the complete 320-frame shot."""
    eased = BC.smooth(q)
    return 1.35*(1.-.82*eased), 1.+3.8*eased


def gilt_reflection(q):
    """A bright rose reflection travels slowly across the smooth gilt rim.

    The centroid moves within the storyteller's window reflection; this is
    authored reflected illumination, not a claim about solar motion in 13 s.
    """
    strength = dawn_state(q)[1]/4.8
    sweep = 2.*BC.smooth(q)-1.
    position = np.asarray(L.PLATE_REFLECTION_POS)+sweep*np.array([6., 0., 2.])
    return (*position, .024*strength, .0204*strength,
            .0204*strength, 3.)


class DSpread(L.LastLeaf):
    def __init__(self, shot='lastleaf', W=1920, H=804, ppc=90, gap_degrees=50.):
        if shot not in ('lastleaf', 'title'):
            raise ValueError('DSpread supports lastleaf or title')
        super().__init__(W, H, ppc, gap_degrees, title=shot == 'title', plate=True)
        self.shot = shot

    def layout_at(self, frame):
        q = phase(self.shot, frame)
        # The approved 2.5% move now occupies all 320 frames. The title uses
        # C X3's measured move onto the recto, with this spread's dawn light.
        if self.shot == 'lastleaf':
            return self.layout((239/FPS)*q)
        bk, _ = self.layout(239/FPS)
        source_frame = 6960+(frame-RANGES['title'][0])
        source_time = (source_frame-BC.SHOTS['last_pages'][0])/FPS
        return bk, self.cam_last_pages(bk, source_time)

    def title(self):
        from titleburn import TitleBurn
        # Match X3's 20-frame lead-in, then hold the cooled ink through D7119.
        return self.once('D_title', lambda: TitleBurn(20/FPS, 12., ppc=self.ppc))

    def _draw_pass(self, bk, cam, light, left, right, clock, q, *, dawn=False, xl=None):
        kwargs = {'burnished_gilt': True}
        if dawn:
            eased = BC.smooth(q)
            kwargs.update(amb=(0., 0., 0.), env=(0., 0., 0.),
                          fill=((.2, -.1, 1.), (.060*eased, .050*eased, .064*eased)),
                          burnished_reflection=gilt_reflection(q))
        hdr, alpha, G = B.render(bk, cam, light, left, right, clock,
                                 xlights=xl, fire_k=float(xl is not None), **kwargs)
        if not dawn:
            hdr = C.soften_spine(hdr, G, bk, light)
        mesh = self.once('physical_pen', lambda: PEN.geometry(bk))
        hdr, depth = PEN.composite(hdr, G, bk, cam, light, mesh)
        hdr = C.replace_visible_nib(hdr, G, depth, mesh, cam, light)
        if dawn:
            dark = B.Light(light.pos, power=0., radius=light.radius)
            hdr -= C.replace_visible_nib(np.zeros_like(hdr), G, depth, mesh, cam, dark)
        return hdr, alpha, depth

    def frame(self, frame):
        q = phase(self.shot, frame)
        bk, cam = self.layout_at(frame)
        light_q = q if self.shot == 'lastleaf' else 1.
        clock = (frame-RANGES['lastleaf'][0])/FPS
        hp, dp = dawn_state(light_q)
        hearth = self.light(L.HEARTH_POS, hp, clock, seed=5, amt=.14,
                           col=(1., .52, .24), radius=10.)
        dawn = B.Light(L.PLATE_DAWN_POS, col=(1., .68, .75), power=dp, radius=2.4)
        left = self.leaf()
        title = self.title() if self.with_title else None
        title_time = (frame-RANGES[self.shot][0])/FPS
        right = title.texture(title_time) if title else self.blank
        xl = title.lights(title_time, bk) if title else None
        hdr, alpha, depth = self._draw_pass(bk, cam, hearth, left, right, clock, light_q, xl=xl)
        fire = right.chan[..., 4].copy() if title else None
        try:
            if title:
                right.chan[..., 4] = 0.
                right.build()
            daylight, _, _ = self._draw_pass(bk, cam, dawn, left, right, clock, light_q, dawn=True)
        finally:
            if title:
                right.chan[..., 4] = fire
                right.build()
        hdr += daylight
        hdr = B.dof(hdr, depth.astype(np.float32), cam, cam.dof_k*self.W/1920.)
        if title:
            sparks = title.sparks(title_time, bk, cam, self.W, self.H)
            if sparks is not None:
                hdr += sparks
        return hdr, alpha


def make_renderer(shot, scale=1., ppc=90, gap_degrees=50.):
    if shot not in RANGES:
        raise ValueError('unknown D book shot')
    if not math.isfinite(scale) or not 0 < scale <= 1 or int(804*scale) < 1:
        raise ValueError('scale must produce positive dimensions, at most native')
    if not isinstance(ppc, int) or isinstance(ppc, bool) or ppc <= 0:
        raise ValueError('ppc must be a positive integer')
    width, height = int(1920*scale), int(804*scale)
    if shot == 'oldfire':
        from oldfire import OldFire
        return OldFire(W=width, H=height, ppc=ppc)
    return DSpread(shot, width, height, ppc, gap_degrees)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shot', required=True, choices=tuple(RANGES))
    parser.add_argument('--frames', required=True, help='absolute D frame list or inclusive range')
    parser.add_argument('--out', required=True, type=L.output_dir)
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--ppc', type=int, default=90)
    parser.add_argument('--gap-degrees', type=float, default=50.)
    args = parser.parse_args(argv)
    frames = BC.frames_of(args.frames)
    try:
        if not frames:
            raise ValueError('empty frame range')
        for frame in frames:
            phase(args.shot, frame)
    except ValueError as exc:
        parser.error(str(exc))
    renderer = make_renderer(args.shot, args.scale, args.ppc, args.gap_degrees)
    matte = args.out.with_name(args.out.name+'_matte')
    for frame in frames:
        started = time.perf_counter()
        hdr, alpha = renderer.frame(frame)
        if not np.isfinite(hdr).all() or not np.isfinite(alpha).all():
            raise RuntimeError('non-finite D book frame')
        rgb = BC.look.finish(hdr, exposure=1.15, bloom_strength=.06,
                             bloom_threshold=1.2, vignette_amount=.32)
        V.save(args.out/f'f_{frame:05d}.png', rgb, 'png')
        V.save(matte/f'f_{frame:05d}.png', np.repeat(alpha[..., None], 3, -1), 'png')
        print(json.dumps(dict(shot=args.shot, frame=frame, seconds=time.perf_counter()-started,
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
              (1 if sys.platform == 'darwin' else 1024), width=renderer.W,
              height=renderer.H, ppc=args.ppc)), flush=True)


if __name__ == '__main__':
    main()
