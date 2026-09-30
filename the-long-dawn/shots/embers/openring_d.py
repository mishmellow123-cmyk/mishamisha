"""Explicit D-frame facade for the open band; importing it changes no C code.

The geometry, inscription-stop and placement call signatures match ``openring``.
Only this module owns D offsets and the D gap/heat/glare score. A caller must
choose this module explicitly; ``openring`` and its C schedule remain unchanged.
Shot ranges are start-inclusive/end-exclusive; frame files retain cut D v2 numbers.
"""
from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np

import openring as _C


@dataclass(frozen=True)
class Shot:
    d_start: int
    d_end: int
    c_start: int

    @property
    def offset(self):
        return self.d_start - self.c_start


# The sole C -> D v2 offset table. Forging trims 80 frames from C's source shot;
# race extends 80 frames past C's source shot;
# unfinished extends 80 frames too. Consumers may clamp a camera or pose, but
# must not silently clamp the event clock returned by legacy_frame(). Holdout's
# late-trap C clock is an authored staging choice, not a pre-existing C insert.
SHOTS = {
    'forging': Shot(2080, 2400, 1040),
    'race': Shot(2400, 2720, 1440),
    'trap': Shot(3760, 4080, 2320),
    'holdout': Shot(5480, 5520, 2600),
    'instep': Shot(5680, 5840, 3840),
    'unfinished': Shot(6080, 6400, 4000),
}
# Locked to cut D v2's EDL interval; only the legacy staging clock is authored.
HOLDOUT_PROVISIONAL = False


def legacy_frame(shot, frame):
    """Convert D frame(s) to the explicit shot's unbounded legacy C clock."""
    value = np.asarray(frame, dtype=float) - SHOTS[shot].offset
    return float(value) if value.ndim == 0 else value


def shot_range(shot):
    """Return the D [start, end) range, suitable for ``range(*shot_range(...))``."""
    timing = SHOTS[shot]
    return timing.d_start, timing.d_end


def _d_event(shot, start, end):
    offset = SHOTS[shot].offset
    return start + offset, end + offset


# Actual source intervals, not measurements inferred from a rendered image.
# "sink" is the inherited power dim; C never physically lowers that tower:
# c_v5.forge_level: sink2420..2438, surge2480..2492, return2540..2552,
# leaders2580..2598. Forges.height: surge2480..2538, return2540..2578,
# leaders2580..2636. Schedule.tower_post: leader lean2580..2630.
# c_v5_instep.START/SETTLED = 3848/3872; R1 openring.COOL_END = 3912.
EVENTS = {
    'sink': _d_event('trap', 2420, 2438),
    'surge': _d_event('trap', 2480, 2492),
    'surge_height': _d_event('trap', 2480, 2538),
    'return': _d_event('trap', 2540, 2552),
    'return_height': _d_event('trap', 2540, 2578),
    'leaders': _d_event('trap', 2580, 2598),
    'leaders_height': _d_event('trap', 2580, 2636),
    'leaders_lean': _d_event('trap', 2580, 2630),
    'instep_ease': _d_event('instep', 3848, 3872),
    'ends_cool': _d_event('instep', 3848, 3912),
}
PAUSE_START, SETTLED = EVENTS['instep_ease']
_, COOL_END = EVENTS['ends_cool']
# The v2 EDL places first contact sixteen frames into the holdout insert.
HOLDOUT_STRIKE = SHOTS['holdout'].d_start + 16

THETA_START = _C.THETA_START
GAP_CENTRE = _C.GAP_CENTRE
FORGE_SCALE = _C.FORGE_SCALE
INITIAL_GAP_DEG = 120.0

def _stroke(shot, offset, target):
    start = float(SHOTS[shot].d_start + offset)
    return start, start + 5., float(target)


# One row is one 5-frame stroke: start, end, remaining gap degrees. All intervals
# between rows are exact holds. The band already exists at the forging's first
# frame. The race accelerates from 40- to 20- to 10-frame stroke spacing. The
# final five approach strokes occur under the Deep, before the brink shot.
# Author offsets relative to the sole shot table so an EDL move cannot leave
# the heat, particle strikes or angular schedule on the previous cut's clock.
GAP_SCHEDULE = (
    _stroke('forging', 160, 118),
    _stroke('forging', 180, 116),
    _stroke('forging', 200, 114),
    _stroke('forging', 220, 112),
    _stroke('forging', 240, 110),
    _stroke('forging', 260, 108),
    _stroke('race', 0, 106),
    _stroke('race', 40, 104),
    _stroke('race', 80, 102),
    _stroke('race', 120, 100),
    _stroke('race', 160, 98),
    _stroke('race', 200, 96),
    _stroke('race', 220, 94),
    _stroke('race', 240, 92),
    _stroke('race', 260, 90),
    _stroke('race', 270, 88),
    _stroke('race', 280, 86),
    _stroke('race', 290, 84),
    _stroke('race', 300, 82),
    _stroke('race', 310, 80),
    _stroke('race', 360, 75),
    _stroke('race', 400, 70),
    _stroke('race', 440, 65),
    _stroke('race', 480, 60),
    _stroke('race', 520, 55),
    (float(HOLDOUT_STRIKE), float(HOLDOUT_STRIKE + 5), 50.),
)


def _ease(u):
    u = np.clip(u, 0., 1.)
    return u * u * (3. - 2. * u)


def gap_deg(f):
    """Remaining D gap; forging opens at 120 degrees and 50 survives the pause."""
    frames = np.asarray(f, dtype=float)
    gap = np.full_like(frames, INITIAL_GAP_DEG)
    previous = INITIAL_GAP_DEG
    for start, end, target in GAP_SCHEDULE:
        gap -= (previous - target) * _ease((frames - start) / (end - start))
        previous = target
    return float(gap) if gap.ndim == 0 else gap


def advance_rate(f):
    """Positive degrees per D frame, exactly zero between hammer strokes."""
    frames = np.asarray(f, dtype=float)
    rate = np.zeros_like(frames)
    previous = INITIAL_GAP_DEG
    for start, end, target in GAP_SCHEDULE:
        u = np.clip((frames - start) / (end - start), 0., 1.)
        rate += (previous - target) * 6. * u * (1. - u) / (end - start)
        previous = target
    return float(rate) if rate.ndim == 0 else rate


def theta_range(f):
    """Both ends approach the unchanged, camera-facing bisector."""
    extent = np.deg2rad(360. - gap_deg(f))
    back = GAP_CENTRE + np.pi
    return back - extent * .5, back + extent * .5


def arc_theta(f, fractions, inset_deg=1.5):
    """Map scalar/vector birth fractions onto existing metal only."""
    th0, th1 = theta_range(f)
    extent = th1 - th0
    pad = np.minimum(math.radians(inset_deg), .1 * extent)
    return th0 + pad + np.clip(np.asarray(fractions), 0., 1.) * (extent - 2. * pad)


def write(f):
    """Preserve the canonical inscription stop: half a glyph at the leading end."""
    th0, th1 = theta_range(f)

    def coverage(theta):
        distance = (np.asarray(theta) - th0) % (2. * np.pi)
        remaining = th1 - th0 - distance
        return np.where(remaining >= -1e-10,
                        np.clip(.5 + remaining / math.radians(1.5), 0., 1.), 0.)

    return coverage


def _cool_fraction(f):
    return float(_ease((f - PAUSE_START) / (COOL_END - PAUSE_START)))


def heat(f):
    """Both cut ends stay white-hot in the race, then cool to unlit gold.

    The renderer's metal shader turns scalar heat into emission colour. The
    body remains reflective gold; no heat travels round or across the gap.
    """
    th0, th1 = theta_range(f)
    race_a, race_b = shot_range('race')
    race = float(_ease((f - race_a) / (race_b - race_a)))
    work = min(advance_rate(f) / .6, 1.)
    hot = min(.88 + .12 * race + .08 * work, 1.)
    cooled = _cool_fraction(f)
    end_heat = hot * (1. - cooled) + .13 * cooled
    width = math.radians(5. + 4. * race + 2. * work)

    def temperature(theta):
        th = np.asarray(theta)
        a = np.abs((th - th0 + np.pi) % (2. * np.pi) - np.pi)
        b = np.abs((th - th1 + np.pi) % (2. * np.pi) - np.pi)
        return end_heat * np.exp(-(np.minimum(a, b) / width) ** 2)

    return temperature


class Glare(NamedTuple):
    """Native-pixel fringe outside the cap aperture and linear HDR gain."""
    radius_px: float
    gain: float


def glare(f):
    """Glare obscures racing ends; radius and energy clear by ``COOL_END``.

    This is a score, not the bloom implementation. ``d_glare.render`` measures
    the gap's projected end faces and adds this small fringe around them.
    Radius and energy grow independently of the true angular gap. Stroke
    pulses modulate the renderer separately, so this envelope is monotonic.
    """
    a, b = shot_range('race')
    race = float(_ease((f - a) / (b - a)))
    clear = 1. - _cool_fraction(f)
    return Glare((4. + 14. * race) * clear, (.5 + 1.7 * race) * clear)


# Public placement remains the exact implementation used by the C facade.
placement = _C.placement
gap_facing = _C.gap_facing


def cap_facing(rotation, camera_pos, centre, f):
    """The legacy diagnostic with the D endpoints and unchanged world pose."""
    view = (np.asarray(camera_pos) - np.asarray(centre)) @ np.asarray(rotation)
    view = view / max(np.linalg.norm(view), 1e-10)
    a, b = theta_range(f)
    return np.array([view[0] * math.sin(a) - view[2] * math.cos(a),
                     -view[0] * math.sin(b) + view[2] * math.cos(b)])
