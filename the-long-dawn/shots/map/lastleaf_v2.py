"""Explicit D-v2 leaf: interrupted handwriting between two warm C book shots.

The accepted Round 2 entry point and the D-v1 dawn remain unchanged.
"""
import math
import os

for _pool in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_pool, '1')

import numpy as np
import book as B
import book_c as BC
import book_c_v5_pen_candidates as C
import lastleaf_v2_assets as A

START, END = 8320, 8640
FPS = 24
# Calibrated against separately reported clear-paper patches, not a pooled
# spread average. The gains compensate the new cameras and verso/recto placement.
START_GAIN = np.array([0.6925597046726039, 0.6921291662862642, 0.6928128851207653])
END_GAIN = np.array([0.9417331668200033, 0.9394046696763936, 0.9427851223991631])


def phase(frame):
    if (isinstance(frame, bool) or not isinstance(frame, (int, np.integer))
            or not START <= frame < END):
        raise ValueError('LastLeafV2 accepts integer D frames 8320..8639')
    return (frame-START)/(END-START-1)


def neighbour_lights():
    """Actual C6399 plenty and candidate C5440 hearth states."""
    scene = BC.Book3(1, 1)
    t = 239/FPS
    power = 1.9*(1.+.3*(1.-BC.smooth((t-9.)/2.)))
    options = dict(seed=5, amt=.14, col=(1., .52, .24), radius=10.)
    return (scene.light((-55., 50., 30.), power, t, **options),
            scene.light((-55., 50., 30.), 1.9, 0., **options))


def bridge_light(q):
    first, last = neighbour_lights()
    k = BC.smooth(q)
    # Small living-hearth variation vanishes at both measured cut endpoints.
    breath = 1.+(B.hearth(q*(END-START)/FPS, 5, .04)-1.)*math.sin(math.pi*q)**2
    gain = START_GAIN*(1.-k)+END_GAIN*k
    return B.Light(first.pos*(1.-k)+last.pos*k,
                   (first.col*(1.-k)+last.col*k)*breath*gain, radius=10.)


def gilt_reflection():
    """A stationary warm room reflection; camera motion moves its sheen."""
    return (-22.5463, 50.8598, 36.8330, .030, .019, .0085, 3.)


def bezier(first, middle, last, q):
    control = 2.*middle-.5*(first+last)
    return first*(1.-q)**2+control*(2.*q*(1.-q))+last*q*q


class LastLeafV2(C.CandidatePagesV5):
    def __init__(self, W=1920, H=804, ppc=90, gap_degrees=50.):
        super().__init__('soft_spine_metal', W, H, ppc)
        self.gap_degrees = A.RP.validate_gap(gap_degrees)

    def leaf(self):
        return self.once('interrupted_leaf', lambda: A.leaf(self.ppc, self.gap_degrees))

    def layout_at(self, frame):
        q = BC.smooth(phase(frame))
        bk = self.book(2.8, 1.2, seed=13)
        ring = bk.page_to_world('L', np.array([10.]), np.array([8.3]))[0]
        first = ring+np.array([0., -1.3, 0.])
        middle = np.array([5.0, 5.4, 3.0])
        last = np.array([6.0, 3.2, 2.0])
        target = bezier(first, middle, last, q)
        # Reach the break early, then keep travelling toward the recto. An
        # unconstrained three-point quadratic overshoots and backs up late.
        exponent = math.log((last[0]-middle[0])/(last[0]-first[0]))/math.log(.5)
        target[0] = last[0]-(last[0]-first[0])*(1.-q)**exponent
        offset = bezier(np.array([-.5, -28.5, 28.5]),
                        np.array([-.5, -27., 27.]),
                        np.array([-.5, -36., 36.]), q)
        cam = B.Cam(target+offset, target, 38., self.W, self.H)
        cam.dof_k = 2.
        return bk, cam

    def frame(self, frame):
        q = phase(frame)
        bk, cam = self.layout_at(frame)
        light = bridge_light(q)
        hdr, alpha, G = B.render(bk, cam, light, self.leaf(), self.blank,
                                 (frame-START)/FPS, burnished_gilt=True,
                                 burnished_reflection=gilt_reflection())
        # Plenty has no added spine bounce; the outgoing C22 candidate does.
        bounced = C.soften_spine(hdr, G, bk, light)
        hdr += BC.smooth(q)*(bounced-hdr)
        mesh = self.once('C22_pen_at_break', lambda: A.geometry(bk))
        hdr, depth = A.composite(hdr, G, bk, cam, light, mesh)
        self._G, self._depth = G, depth
        return B.dof(hdr, depth.astype(np.float32), cam, cam.dof_k*self.W/1920.), alpha
