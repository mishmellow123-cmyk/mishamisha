"""Opt-in open band on absolute C frames, shared by the C3 and C5 drivers.

The schedule below is the sole authoring table for closure. A row is one short
stroke: (start frame, end frame, remaining gap in degrees). Between rows both
ends hold exactly still. Geometry, drops, inscription and heat all read it.
Importing this module changes no renderer state; callers gate it with enabled().
"""
import math
import os

import numpy as np


THETA_START = -0.6
INITIAL_GAP_DEG = 360.0
PAUSE_START = 3848.0           # c_v5_instep.START: the forges settle together
COOL_END = 3912.0
FORGE_SCALE = 0.75            # fixed C3 size: keeps the full band inside its original camera move

# C3.STROKES and C3.RACE_BEATS are on the 20-frame score beat. The later
# progression is authored here on that same beat, with a final stroke on the
# actual last catch (shots/map/last_beacon.py: C3784..3791, all lit by C3792).
# Edit the target gaps here; do not put a second schedule in a shot driver.
GAP_SCHEDULE = (
    (1200.0, 1205.0, 320.0),
    (1220.0, 1225.0, 280.0),
    (1240.0, 1245.0, 240.0),
    (1260.0, 1265.0, 200.0),
    (1280.0, 1285.0, 160.0),
    (1300.0, 1305.0, 120.0),
    (1560.0, 1565.0, 120.0 - 40.0 / 6.0),
    (1580.0, 1585.0, 120.0 - 80.0 / 6.0),
    (1600.0, 1605.0, 100.0),
    (1620.0, 1625.0, 120.0 - 160.0 / 6.0),
    (1640.0, 1645.0, 120.0 - 200.0 / 6.0),
    (1660.0, 1665.0, 80.0),
    (1920.0, 1925.0, 75.0),
    (1940.0, 1945.0, 70.0),
    (1960.0, 1965.0, 65.0),
    (1980.0, 1985.0, 60.0),
    (2000.0, 2005.0, 55.0),
    (2020.0, 2025.0, 50.0),
    (2040.0, 2045.0, 45.0),
    (2060.0, 2065.0, 40.0),
    (2480.0, 2485.0, 39.25),
    (2500.0, 2505.0, 38.5),
    (2520.0, 2525.0, 37.75),
    (2540.0, 2545.0, 37.0),
    (2560.0, 2565.0, 36.25),
    (2580.0, 2585.0, 35.5),
    (2600.0, 2605.0, 34.75),
    (2620.0, 2625.0, 34.0),
    (3784.0, 3792.0, 30.0),
)
FORGE_END = max(end for start, end, _ in GAP_SCHEDULE if start < 1440.0)
_FORGED_GAP = next(gap for _, end, gap in GAP_SCHEDULE if end == FORGE_END)
GAP_CENTRE = THETA_START - math.radians(_FORGED_GAP) * 0.5
MIN_ARC_RADIANS = 1e-3


def enabled():
    """Only the explicitly requested value opts existing renders in."""
    return os.environ.get('LD_OPEN_RING') == '1'


def _ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def gap_deg(f):
    """Remaining gap, with short stroke advances and exact intervening holds."""
    frames = np.asarray(f, dtype=float)
    gap = np.full_like(frames, INITIAL_GAP_DEG)
    previous = INITIAL_GAP_DEG
    for start, end, target in GAP_SCHEDULE:
        gap -= (previous - target) * _ease((frames - start) / (end - start))
        previous = target
    return float(gap) if gap.ndim == 0 else gap


def advance_rate(f):
    """Positive angular advance in degrees/frame; exactly zero off a stroke."""
    frames = np.asarray(f, dtype=float)
    rate = np.zeros_like(frames)
    previous = INITIAL_GAP_DEG
    for start, end, target in GAP_SCHEDULE:
        u = np.clip((frames - start) / (end - start), 0.0, 1.0)
        rate += (previous - target) * 6.0 * u * (1.0 - u) / (end - start)
        previous = target
    return float(rate) if rate.ndim == 0 else rate


def theta_range(f):
    """Both ends advance about a fixed gap bisector; th1 is the writing front.

    Symmetric formation keeps the gap camera-facing without rotating the band
    while it is forged. The initial 0.001-radian seed avoids a degenerate mesh
    on the stroke's first shutter sample; it is less than 0.06 degrees long.
    """
    extent = np.maximum(np.deg2rad(360.0 - gap_deg(f)), MIN_ARC_RADIANS)
    back = GAP_CENTRE + np.pi
    return back - extent * 0.5, back + extent * 0.5


def arc_theta(f, fractions, inset_deg=1.5):
    """Map drop fractions onto existing metal, inset from both cut edges.

    Frame and fraction arrays follow NumPy broadcasting; a vanishing first arc
    shrinks the inset with it, so early births cannot cross the leading edge.
    """
    th0, th1 = theta_range(f)
    extent = th1 - th0
    pad = np.minimum(math.radians(inset_deg), 0.1 * extent)
    return th0 + pad + np.clip(np.asarray(fractions), 0.0, 1.0) * (extent - 2.0 * pad)


def write(f):
    """Inscription coverage on existing metal; the leading edge is half-written.

    The canonical inscription texture remains the glyph source. This mask clips
    its last stroke at the cut face, and cannot paint letters across the gap.
    """
    th0, th1 = theta_range(f)

    def coverage(theta):
        distance = (np.asarray(theta) - th0) % (2.0 * np.pi)
        remaining = th1 - th0 - distance
        return np.where(remaining >= -1e-10,
                        np.clip(0.5 + remaining / math.radians(1.5), 0.0, 1.0), 0.0)

    return coverage


def _recent_rate(f):
    """Residual heat from recent angular work; a faster stroke has more heat."""
    impulse = advance_rate(f)
    previous = INITIAL_GAP_DEG
    for start, end, target in GAP_SCHEDULE:
        if f >= start:
            age = max(float(f) - end, 0.0)
            mean_rate = (previous - target) / (end - start)
            impulse = max(impulse, mean_rate * math.exp(-age / 7.0))
        previous = target
    return float(impulse)


def heat(f):
    """Return ringsolid's heat(theta): both ends reheat as closure advances.

    White heat stays local to the cut faces. The race's hot ends cool through
    the shader's orange range after the shared pause, retaining faint warmth.
    """
    th0, th1 = theta_range(f)
    active = float(_ease((f - GAP_SCHEDULE[0][0]) / 3.0))
    race = float(_ease((f - 1440.0) / 120.0))
    work = min(_recent_rate(f) / (40.0 / 6.0 / 5.0), 1.0)
    hot = active * min(0.48 + 0.34 * race + 0.52 * work, 1.0)
    cooled = float(_ease((f - PAUSE_START) / (COOL_END - PAUSE_START)))
    end_heat = hot * (1.0 - cooled) + 0.13 * cooled
    width = math.radians(5.0 + 4.0 * work)

    def temperature(theta):
        th = np.asarray(theta)
        a = np.abs((th - th0 + np.pi) % (2.0 * np.pi) - np.pi)
        b = np.abs((th - th1 + np.pi) % (2.0 * np.pi) - np.pi)
        near_end = np.exp(-(np.minimum(a, b) / width) ** 2)
        return end_heat * near_end

    return temperature


def _roll(angle):
    """Local +Y-axis roll that increases the material's polar angle."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def placement(base_rot, camera_pos, centre, reference_frame):
    """Roll a reference pose so its gap bisector faces that shot's camera.

    Pass fixed reference inputs throughout a shot. This changes only roll about
    the band's own axis: it preserves the authored inclination and avoids a
    per-frame camera chase. C3's contiguous forging/race move shares one pose.
    """
    rot = np.asarray(base_rot, dtype=float)
    view = (np.asarray(camera_pos) - np.asarray(centre)) @ rot
    if np.linalg.norm(view[[0, 2]]) < 1e-10:
        raise ValueError('A camera on the band axis cannot define gap-facing roll')
    camera_angle = math.atan2(view[2], view[0])
    return rot @ _roll(camera_angle - GAP_CENTRE)


def gap_facing(rotation, camera_pos, centre, f):
    """Cosine between gap bisector and camera, projected into the ring plane."""
    view = (np.asarray(camera_pos) - np.asarray(centre)) @ np.asarray(rotation)
    radial = np.linalg.norm(view[[0, 2]])
    if radial < 1e-10:
        return 0.0
    return float((view[0] * math.cos(GAP_CENTRE) + view[2] * math.sin(GAP_CENTRE)) / radial)


def cap_facing(rotation, camera_pos, centre, f):
    """Centre-view cosine for each cut normal (positive faces the camera).

    This is an angular placement check; screen-space occlusion and the finite
    perspective offset of a cut face still require a rendered inspection.
    """
    view = (np.asarray(camera_pos) - np.asarray(centre)) @ np.asarray(rotation)
    view = view / max(np.linalg.norm(view), 1e-10)
    a, b = theta_range(f)
    return np.array([view[0] * math.sin(a) - view[2] * math.cos(a),
                     -view[0] * math.sin(b) + view[2] * math.cos(b)])
