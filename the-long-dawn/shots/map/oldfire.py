"""Opt-in D21: the closed Ring is unmade inside the drawn Mountain.

Absolute D frames 4800..4959, at 24 fps. This module does not register a shot
or alter a C renderer. The Mountain is the accepted seed-11 C3 illustration;
its three small Ring strokes are replaced by the canonical closed RingPage
unmaking, scaled into the throat. The caption uses C's baked, pen-shaped ink
wipe with wetness; neither that caption path nor this shot draws a visible pen.
"""
import os

for _pool in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_pool, '1')

import cv2
import numpy as np

import book as B
import book_c as BC
import book_c_t1_candidates as T1
import pages as PG
import pen
import redbook as RB
import ringpage as RP


START = 4800
END = 4960
FPS = 24
CAPTION = 'In the old story, the Ring is unmade in the fire that forged it.'
RING_SCALE = .17
RING_BED = np.array([10.62, 4.53])
FALL_START = .55


def local_time(frame):
    if (isinstance(frame, bool) or not isinstance(frame, (int, np.integer))
            or not START <= frame < END):
        raise ValueError('OldFire accepts absolute integer D frames 4800..4959')
    return (frame - START) / FPS


def unmaking_time(t):
    """Hold the complete closed Ring above the crater, then run its C motion.

    Landing is at local 0.90 s; the canonical letter flare starts near 2.81 s.
    The final bead holds in the same fire without C's white-out.
    """
    return .2 + max(float(t) - FALL_START, 0.)


def mountain_strokes():
    """Retain every accepted Mountain stroke except the original small Ring.

    The public schedule parameter tags only the Ring's time window. It does
    not change seeded geometry. Fail if that component's three-stroke
    structure changes, rather than erasing an arbitrary rectangle of ink.
    """
    mountain = PG.Mountain(11)
    schedule = dict(PG.Mountain.SCHED_C3, ring=(-2., -1.))
    strokes = mountain.build('ink', 1.1, 8.6, sched=schedule)
    removed = [i for i, (a, b) in enumerate(zip(strokes.T0, strokes.T1))
               if a < 0. and b < 0.]
    if len(removed) != 3 or sorted(strokes.L[i] for i in removed) != [pen.INK, pen.INK, pen.GILT]:
        raise RuntimeError('Mountain Ring stroke contract changed')
    removed = set(removed)
    for field in ('P', 'R', 'D', 'T0', 'T1', 'L', 'G', 'N'):
        values = getattr(strokes, field)
        setattr(strokes, field, [value for i, value in enumerate(values) if i not in removed])
    # The same invented book hand below the picture as accepted C3.
    pen.Hand(seed=18, xh=.2).write_block(strokes, 3.2, 15.05, 13.8, 18, .62)
    return mountain, strokes


def mapped_point(point):
    return RING_BED + RING_SCALE * (np.asarray(point) - RP.RingPage.BED)


def overlay_patch(receiver, patch):
    """Opaque gilt/flame replaces underlying ink; uncovered paper is retained.

    The canonical patch contains its own ink, gilt and fire. Combining every
    channel with max would let the Mountain's hatching show through the metal.
    Fine outline coverage outside the material is the usual ink max-composite.
    """
    opaque = (patch[..., 2] > 0.) | (patch[..., 4] > 0.)
    np.maximum(receiver, patch, out=receiver)
    receiver[opaque] = patch[opaque]


def foreground_growth(t):
    """Let the near tongues settle only after the canonical bead has formed."""
    melted = RP.RingPage.state(t)[0][11]
    return 1. - .78 * BC.smooth((melted - .85) / .15)


class MountainRing(RP.RingPage):
    """Canonical closed unmaking; near fire settles and its bead gains a contour.

    The earlier fall, slump and letter flare are byte-identical to RingPage.
    The Ring state and both geometry painters remain inherited unchanged.
    """

    def _fire(self, win, ox, oy, t, tongues, flare, seed, grow=1.):
        if tongues is self.front:
            grow *= foreground_growth(t)
        super()._fire(win, ox, oy, t, tongues, flare, seed, grow)
        if tongues is self.front:
            self._bead_contour(win, ox, oy, t)

    def _bead_contour(self, win, ox, oy, t):
        state = self.state(t)[0]
        strength = BC.smooth((state[11] - .85) / .15)
        if strength <= 0.:
            return
        # RingPage's canonical ellipse, with a book-scale pen edge. Its .026
        # source contour becomes subpixel after the Mountain mapping. The
        # added .08 contour changes neither the ellipse nor the filled bead.
        cx, cy = state[0], state[1] + .5 * state[4]
        rx, ry = state[2], state[3] * state[2] + .5 * state[4]
        angle = np.linspace(0., 2. * np.pi, 61)
        strokes = pen.Strokes()
        pen.line(strokes, np.column_stack((cx + rx * np.cos(angle),
                                          cy + ry * np.sin(angle))),
                 .08, 78, smooth=0, lift=(2., 4.))
        ink, _ = pen.raster(strokes.pack(), 1e9, self.ppc, *win.shape[:2],
                            pen.INK, ox=ox, oy=oy)
        ink = np.clip(ink * strength, 0., 1.)
        np.maximum(win[..., 0], ink, out=win[..., 0])
        win[..., 2] *= 1. - .9 * ink
        # Only the narrow ink contour masks emission. The fire's interior,
        # the bead's gold and its source geometry are left in place.
        win[..., 4] *= 1. - .96 * ink


class OldFire(BC.Book3):
    """Independent renderer with Book3's HDR/alpha frame contract."""

    def __init__(self, W=1920, H=804, ppc=110):
        if (any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
                for v in (W, H, ppc)) or ppc > 110):
            raise ValueError('OldFire needs positive integer dimensions and ppc <= 110')
        cv2.setNumThreads(1)
        super().__init__(int(W), int(H))
        self.ppc = int(ppc)

    def layout(self, t):
        book = self.book(.6, 2.8, seed=3)
        target = book.page_to_world('R', np.array([9.8]), np.array([8.8]))[0]
        offset = np.array([-.6, -25.5, 25.]) * (1. - .025 * BC.smooth(t / ((END-START)/FPS)))
        camera = B.Cam(target + offset, target, 38., self.W, self.H)
        camera.dof_k = 2.
        return book, camera

    def caption(self):
        def make():
            line = T1.TextInkLine(CAPTION, self.ppc)
            line.f_in, line.f_out = START, END + 12
            return line
        return self.once('oldfire_caption', make)

    def mountain_page(self):
        def make():
            _, strokes = mountain_strokes()
            page = RB.Page(strokes, self.ppc)
            texture = page.texture(1e9)
            return texture, texture.chan.copy()
        return self.once('oldfire_mountain', make)

    def ring_page(self):
        # Raster density follows its scale on the destination page. A full
        # 130-ppc RingPage would allocate a large page only to reduce it sixfold.
        return self.once('oldfire_closed_ring', lambda: MountainRing(
            ppc=self.ppc * RING_SCALE, open_band=False))

    def fire_patch(self, t):
        ring = self.ring_page()
        if ring.gap_degrees != 0.:
            raise ValueError('The old story requires the canonical closed Ring')
        x0, y0, x1, y1 = ring.WIN
        i0, i1 = int(y0 * ring.ppc), int(y1 * ring.ppc)
        j0, j1 = int(x0 * ring.ppc), int(x1 * ring.ppc)
        # Nine samples across a 180-degree shutter for the short fall; only
        # the small drawing patch is sampled, not nine complete book renders.
        offsets = ((np.arange(9) + .5) / 9. / 48. - 1. / 96.
                   if FALL_START < t < FALL_START + .42 else (0.,))
        patch = None
        for dt in offsets:
            texture = ring.texture(unmaking_time(t + dt))
            sample = texture.chan[i0:i1, j0:j1]
            if patch is None:
                patch = sample.copy()
            else:
                patch += sample
        patch /= len(offsets)
        origin = mapped_point((j0 / ring.ppc, i0 / ring.ppc))
        col, row = np.rint(origin * self.ppc).astype(int)
        return row, col, patch

    def texture(self, frame):
        t = local_time(frame)
        texture, base = self.mountain_page()
        texture.chan[:] = base
        row, col, patch = self.fire_patch(t)
        height, width = patch.shape[:2]
        receiver = texture.chan[row:row+height, col:col+width]
        if receiver.shape != patch.shape:
            raise ValueError('OldFire drawing patch extends outside its page')
        overlay_patch(receiver, patch)
        self.caption().apply(texture.chan, frame)
        return texture.build()

    def frame(self, frame):
        t = local_time(frame)
        book, camera = self.layout(t)
        right = self.texture(frame)
        left = self.tex_text(41)
        light = self.light((-50., 48., 36.), 2.2, t)
        fire_uv = mapped_point(RP.RingPage.FIRE)
        fire_world = book.page_to_world('R', fire_uv[:1], fire_uv[1:])[0]
        # A small, local ember bounce connects the drawn fire to its parchment.
        strength = B.flicker(t * 1.6, 5, .14)
        fire_light = np.array([[*fire_world[:2], fire_world[2] + .7,
                                .10 * strength, .045 * strength, .012 * strength]])
        return self.finish_layer(book, camera, light, left, right, t,
                                 xl=fire_light, fire_k=1., texS=left, st=.05)
