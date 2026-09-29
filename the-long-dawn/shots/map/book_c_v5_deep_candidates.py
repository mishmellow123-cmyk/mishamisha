"""Default-off Deep studies: two combined lamp-and-ladder alternatives.

``make_renderer()`` returns the *original* PagesV5('deep_abandoned', ...).
Both ``round_bail`` and ``leaned_ladders`` replace the accepted lantern and
every added ladder; their names do not denote isolated equipment changes.
``round_bail`` pairs the rounded lamp with wider, more widely spaced ladder
rungs. Relative to that option, ``leaned_ladders`` widens the entire lamp by
15%, reduces its cap rise, and further widens, spaces and inclines the ladders.
The returned renderer has the accepted ``frame(absolute_C_frame) -> (hdr, alpha)`` API;
callers must use book_c_v5's existing grade and save functions. This module
does not write files, change production dispatch, or bake textures on import.
Use validate_output_dir() before a study runner writes any frame.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import book_c_v5 as V
import pen
import v5_inkpages as P


CANDIDATES = ('accepted', 'round_bail', 'leaned_ladders')


def validate_output_dir(path):
    """Study destinations are new cand_deep_* siblings, never final stems."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError('study output must be absolute')
    path = path.resolve()
    root = (Path.home() / 'ldfarm' / 'out').resolve()
    if path.parent != root or not path.name.startswith('cand_deep_'):
        raise ValueError('use a new cand_deep_* directory under the configured candidate output root')
    if path == root / 'cand_deep_':
        raise ValueError('study directory needs a candidate name')
    return path


@dataclass(frozen=True)
class Ladder:
    """Nominal rail endpoints in page cm; y increases toward the floor."""
    hall: int
    top_x: float
    bottom_x: float
    top_y: float
    bottom_y: float
    half_width: float
    rung_pitch: float

    def rail(self, side):
        return np.array([[self.top_x + side * self.half_width, self.top_y],
                         [self.bottom_x + side * self.half_width, self.bottom_y]])

    def rungs(self):
        for y in np.arange(self.top_y + .14, self.bottom_y - .09, self.rung_pitch):
            q = (y - self.top_y) / (self.bottom_y - self.top_y)
            x = self.top_x + q * (self.bottom_x - self.top_x)
            yield np.array([[x - self.half_width, y], [x + self.half_width, y]])


class CandidateDeep(P.AbandonedDeep):
    """Keep the accepted mine/miner RNG contract; replace lamp and ladders together.

    Both options retain the inherited mine, gold and miner-removal behavior.
    They replace the accepted additions with a curved lamp at its own local
    surface height and ladders fitted within the end bays below the vaults.
    Neither option isolates a lamp change from a ladder change.
    """
    def __init__(self, candidate='round_bail', seed=23):
        if candidate not in CANDIDATES[1:]:
            raise ValueError('unknown Deep candidate')
        super().__init__(seed)
        self.candidate = candidate

    def lamp_paths(self):
        """An unlit cage lamp: oval bail, domed cap, curved cage, round reservoir.

        Both options replace the accepted pitched-roof lantern. ``round_bail``
        uses the coordinates below unchanged; ``leaned_ladders`` scales every
        horizontal offset by 1.15 and reduces cap rise from .19 to .13 page cm.
        That 15% widening is relative to ``round_bail``, not the accepted lamp.
        The x centre matches the accepted lantern. The reservoir's bottom is
        .035 page cm above the surface at that x; accepted uses the shaft-mouth
        surface height for its base instead.
        No pointed roof, rectangular handle, flame, or new gold is introduced.
        """
        wide = self.candidate == 'leaned_ladders'
        scale_x = 1.15 if wide else 1.
        lx = float(self.vein[0, 0]) - 1.35
        ly = float(self.surface(lx)) - .035

        def arc(cx, cy, rx, ry, start=0., end=2*np.pi, count=65):
            a = np.linspace(start, end, count)
            return np.column_stack([cx + rx*np.cos(a), cy + ry*np.sin(a)])

        cap_rise = .13 if wide else .19
        paths = [
            arc(0., -1.43, .33, .38),                 # open oval carrying bail
            arc(0., -1.04, .53, cap_rise, np.pi, 2*np.pi),
            np.array([[-.53, -1.04], [.53, -1.04]]),  # thin cap underside
            # The sloped shoulders and waist avoid a door/window rectangle.
            np.array([[-.36, -.98], [-.43, -.88], [-.43, -.72], [-.34, -.31]]),
            np.array([[ .36, -.98], [ .43, -.88], [ .43, -.72], [ .34, -.31]]),
            np.array([[-.24, -.98], [-.28, -.74], [-.20, -.31]]),
            np.array([[ .24, -.98], [ .28, -.74], [ .20, -.31]]),
            arc(0., -.165, .43, .165),                # rounded fuel reservoir
            arc(0., -.30, .34, .055, np.pi, 2*np.pi),
        ]
        return [q*np.array([scale_x, 1.]) + [lx, ly] for q in paths]

    def ladders(self):
        """Replace every added ladder in both options; dimensions are page cm.

        Accepted rails are .24 apart with .17 rung pitch. ``round_bail`` uses
        .44 separation, .30 pitch and up to .06 horizontal top-to-bottom lean;
        ``leaned_ladders`` uses .54, .36 and up to .26 respectively. Bay clipping
        can reduce the lean. Both move rails clear of walls and pillar plinths
        and put the top cross-section below the actual vaulted ceiling.
        """
        wide = self.candidate == 'leaned_ladders'
        half = .27 if wide else .22
        pitch = .36 if wide else .30
        lean = .26 if wide else .06
        result = []
        for k, h in enumerate(self.halls):
            left = bool(k % 2)
            supports = self.supports(h)
            # First/last bay only. Clearance includes rail ink radius and the
            # interior pillar's plinth, so wider rails do not become pillars.
            a, b = supports[:2] if left else supports[-2:]
            wall_margin = .09
            pillar_margin = h['pw']*.725 + .065
            low = a + (wall_margin if left else pillar_margin) + half
            high = b - (pillar_margin if left else wall_margin) - half
            if low > high:
                raise ValueError('candidate ladder does not fit its mine bay')
            old_foot = h['x0']+.45 if left else h['x1']-.31
            bottom = float(np.clip(old_foot, low, high))
            top = float(np.clip(bottom + (lean if left else -lean), low, high))
            # Put the full top cross-section below the actual vaulted ceiling.
            xs = np.linspace(top-half-.05, top+half+.05, 41)
            ya = max(h['yc']+.48, float(self.arch_y(h, xs).max())+.095)
            yb = h['y1']-.035
            if ya + .35 >= yb:
                raise ValueError('candidate ladder has insufficient clear height')
            result.append(Ladder(k, top, bottom, ya, yb, half, pitch))
        return tuple(result)

    def additions(self):
        strokes = pen.Strokes()
        for q, points in enumerate(self.lamp_paths()):
            # Explicit sampled curves already have their intended curvature.
            P.contour(strokes, points, .040, 11950+q, smooth=0)
        for ladder in self.ladders():
            for j, side in enumerate((-1, 1)):
                P.contour(strokes, ladder.rail(side), .036,
                          12970+ladder.hall*100+j, smooth=0)
            for j, rung in enumerate(ladder.rungs()):
                P.contour(strokes, rung, .033, 12980+ladder.hall*100+j, smooth=0)
        return strokes


class CandidatePagesV5(V.PagesV5):
    def __init__(self, candidate, W=1920, H=804, ppc=110):
        if candidate not in CANDIDATES[1:]:
            raise ValueError('unknown Deep candidate')
        self.candidate = candidate
        super().__init__('deep_abandoned', W, H, ppc)

    def abandoned(self):
        dp = CandidateDeep(self.candidate)
        strokes = dp.build('ink', .6, 9.6)
        strokes.extend(dp.additions())
        return dp, V.RB.Page(strokes, self.ppc).texture(1e9)


def make_renderer(candidate='accepted', scale=.5, ppc=110):
    """Factory matching accepted CLI dimensions; ppc remains the accepted 110.

    Construction allocates the accepted blank page but does not bake the mine.
    The original frame guard limits both paths to C4240 through C4479.
    """
    if candidate not in CANDIDATES:
        raise ValueError('unknown Deep candidate')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    W, H = int(1920*scale), int(804*scale)
    if min(W, H) < 1:
        raise ValueError('scale produces an empty frame')
    if candidate == 'accepted':
        return V.PagesV5('deep_abandoned', W, H, ppc)
    return CandidatePagesV5(candidate, W, H, ppc)
