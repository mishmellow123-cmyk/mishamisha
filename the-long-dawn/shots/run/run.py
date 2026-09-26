"""THE BEACON RUN (v2 frames 1520-1679, all cuts).

Hard cut from the shepherd's beacon: the camera leaps off his summit and glides out over the foothill
range that falls from it (s1's ridged range at a smaller scale, held under s1's summit-lip sight line so
s1 never saw it), riding the updraft off the cliff rather than diving into the valley, while beacons
flare on the beats on summits around and ahead. At 1580 a great pyre on a tower beside the track
ignites 40 m ahead; we sweep past it at eye level (~1594) through its sparks, then climb away while the
operator eases in (54 -> 42 degrees) as the great peak catches at 1600, pans with the signal as it leaps to
1620 and races away, and zooms on in (-> 29 degrees) to stack the ranges for the last beats and the chains
running to the horizon.

Camera = flight dynamics: horizontal speed / heading keys integrated into a ground track; altitude = a
terrain-following upper envelope (limited vertical acceleration) + clearance and a cruise floor, then a
climb; bank from the turn rate (coordinated turn, scaled for a camera mount, stabilised to level once the
long lens is on); the operator's look is a lagged (critically damped) blend of the flight direction and
what he is framing: the pyre before the pass, then a world-anchored pan across the beats with the horizon
held at a fixed screen height through the zoom.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world as WD          # noqa: E402
import rcam as RC           # noqa: E402

from mt.cam import keys, smoother, smooth   # noqa: E402

START, END = 1520, 1679
FPS = 24.0
G = 9.81

# ------------------------------------------------------------------ flight ---
SPEED = [(1500, 40.0), (1520, 42.0), (1540, 55.0), (1560, 60.0), (1580, 67.0), (1594, 68.0), (1606, 58.0),
         (1626, 47.0), (1650, 41.0), (1679, 37.0), (1700, 35.0)]              # horizontal speed (m/s)
HEADING = [(1500, 8.0), (1522, 8.0), (1552, 28.0), (1583, 28.0), (1608, 19.0), (1636, 9.0), (1679, 4.0),
           (1700, 3.0)]
P0 = np.array([1.5, 3.5, 20.5])          # at START: just over the shepherd's summit lip
CLEAR = 24.0                             # metres above the terrain envelope under the track
A_DOWN = 20.0                            # m/s^2 the camera may fall away after a crest
A_UP = 11.0                              # m/s^2 it may pull up ahead of a crest
# cruise floor before the pass: off the lip the glider rides the cliff's updraft instead of diving into the
# valley (80-100 m over the valley floor), then settles to eye level with the pyre
FLOOR = [(1500, 4.6), (1520, 4.6), (1532, 7.5), (1550, 11.5), (1558, 11.5), (1575, 5.0), (1600, 4.5)]
CLIMB0 = 1594.0                          # the climb begins (just after the pass)
VZ = [(1594, 0.0), (1602, 26.0), (1614, 88.0), (1636, 125.0), (1656, 112.0), (1679, 84.0), (1700, 76.0)]


_PCH = {}


def pch(frame, kf):
    """Monotone cubic (PCHIP) through keyframes: continuous rates, no stop at every key."""
    k = id(kf)
    if k not in _PCH:
        from scipy.interpolate import PchipInterpolator
        _PCH[k] = PchipInterpolator([a for a, _ in kf], [b for _, b in kf], extrapolate=True)
    lo, hi = kf[0][0], kf[-1][0]
    return float(_PCH[k](min(max(frame, lo), hi)))


def heading(frame):
    return pch(frame, HEADING)


def _track():
    df = 0.125
    fs = np.arange(1500.0, 1700.0 + 1e-9, df)
    xz = np.zeros((len(fs), 2))
    i0 = int(round((START - 1500.0) / df))
    xz[i0] = (P0[0], P0[2])
    dt = df / FPS

    def v(f):
        s_ = pch(f, SPEED)
        h = math.radians(heading(f))
        return np.array([s_ * math.sin(h), s_ * math.cos(h)])

    for i in range(i0, len(fs) - 1):
        xz[i + 1] = xz[i] + v(fs[i] + 0.5 * df) * dt
    for i in range(i0, 0, -1):
        xz[i - 1] = xz[i] - v(fs[i] - 0.5 * df) * dt
    return fs, xz


def _fly(CRx):
    fs, xz = _track()
    H = np.full(len(fs), -1e3)
    for i, f in enumerate(fs):
        h = math.radians(heading(f))
        lx, lz = -math.cos(h), math.sin(h)
        for l in (-18.0, -9.0, 0.0, 9.0, 18.0):
            H[i] = max(H[i], WD.ground(xz[i, 0] + lx * l, xz[i, 1] + lz * l, CRx, 0.5))
    t = fs / FPS
    dtm = t[:, None] - t[None, :]
    A = np.where(dtm >= 0.0, A_DOWN, A_UP)
    clr = CLEAR - (CLEAR - 3.5) * np.clip((1532.0 - fs) / 12.0, 0.0, 1.0)     # skim the lip at the start
    env = np.max(H[None, :] + clr[None, :] - 0.5 * A * dtm * dtm, axis=1)
    i0 = int(round((START - 1500.0) / 0.125))
    lip = P0[1] - 0.5 * A_DOWN * np.maximum(t - t[i0], 0.0) ** 2 - 4.0 * np.maximum(t - t[i0], 0.0)
    lip[:i0] = P0[1]
    floor = np.array([pch(f, FLOOR) for f in fs])
    y = np.maximum(np.maximum(env, lip), floor)
    k = np.exp(-0.5 * (np.arange(-24, 25) / 8.0) ** 2)
    k /= k.sum()
    y = np.convolve(np.pad(y, 24, mode='edge'), k, mode='same')[24:-24]
    y = np.maximum(y, H + 0.6 * clr)
    ic = int(round((CLIMB0 - 1500.0) / 0.125))
    yc = y.copy()
    for i in range(ic + 1, len(fs)):
        yc[i] = yc[i - 1] + pch(fs[i] - 0.0625, VZ) * 0.125 / FPS
    w = np.array([smoother((f - CLIMB0) / 12.0) for f in fs])
    y = np.maximum(y * (1 - w) + yc * w, np.where(fs >= CLIMB0, yc, y))
    return fs, np.stack([xz[:, 0], y, xz[:, 1]], 1), H


_PATH = None


def path(frame):
    fs, pos, _ = _PATH
    return np.array([np.interp(frame, fs, pos[:, k]) for k in range(3)])


# bank = coordinated-turn bank x this factor: a glider feel before the pass, a stabilised (gimbal) head once
# the long lens is on
BANK_K = [(1500, 0.13), (1590, 0.13), (1612, 0.05), (1700, 0.05)]


def _bank_target(frame):
    h = 0.5
    dpsi = math.radians(heading(frame + h) - heading(frame - h)) / (2 * h / FPS)
    v = pch(frame, SPEED)
    return pch(frame, BANK_K) * math.degrees(math.atan(v * dpsi / G))


_BANK = None


def bank(frame):
    """Coordinated-turn bank (scaled for a camera mount, + = right), rolled in ahead of the turn (5-frame lead)
    through a critically damped filter (a heavy airframe does not snap-roll)."""
    global _BANK
    if _BANK is None:
        df = 0.25
        fs = np.arange(1480.0, 1700.0 + 1e-9, df)
        w = 2 * math.pi * 0.9
        b = vb = 0.0
        out = np.zeros(len(fs))
        dt = df / FPS
        for i, f in enumerate(fs):
            tb = _bank_target(f + 5.0)
            ab = w * w * (tb - b) - 2 * w * vb
            vb += ab * dt
            b += vb * dt
            out[i] = b
        _BANK = (fs, out)
    fs, out = _BANK
    return float(np.interp(frame, fs, out))


def _left(frame):
    h = math.radians(heading(frame))
    return np.array([-math.cos(h), 0.0, math.sin(h)])


# ------------------------------------------------------------------ the set ---
PYRE_PASS = 1594.0                        # closest approach
PYRE_LAT = 14.0                           # metres to the left of the track at the pass
PYRE_BELOW = 4.5                          # the pyre summit sits this far below the eye at the pass
PYRE_RAISE = 0.0                         # extra height given to the chosen summit (narrow)


def _range(pyre_xz=(0.0, 0.0), boost=0.0):
    return WD.range_row(scale=420.0, amp=170.0, ox=3.7, oz=1.3, seed=21, k=0.0, below_track=0.0,
                        pyre=pyre_xz, pyre_boost=boost, rise=0.0, spine_az=28.0, spine_w=90.0, spine_boost=0.35,
                        spine_len=1000.0)


def _fwd(frame):
    h = math.radians(heading(frame))
    return np.array([math.sin(h), 0.0, math.cos(h)])


def build_set():
    """Fly the range, then stand the pyre tower beside the pass: a rugged pinnacle rising from the gulf on
    the LEFT (SFX pan at 1580 is left), its summit shelf just below the eye line."""
    global _PATH
    CR0 = np.array([_range()])
    _PATH = _fly(CR0)
    c = path(PYRE_PASS)
    q = c + _left(PYRE_PASS) * PYRE_LAT
    top = c[1] - PYRE_BELOW
    ang = math.radians(heading(PYRE_PASS))
    row = WD.crag_row(q[0], q[2], top, L=10.0, s_hi=3.4, s_lo=2.0, aniso=1.7, ang=ang + 0.6, seed=17, k=4.0,
                      detail=0.42, shelf=1.6, nf=3)
    CR1 = np.array([CR0[0], row])
    sm = WD.summit_near(q[0], q[2], CR1, rad=6.0, n=13)
    return CR1, sm, (PYRE_PASS, 1.0, 0.0, 0.0)


CR, PYRE_XZ, PYRE_INFO = build_set()


# ------------------------------------------------------------------ the look ---
# before the pass: a steady world-anchored swing right with the turn (half the old peak pan rate, so the
# first beacons are not smeared), leaning toward the pyre as it comes up (its bearing blended in)
PRE_YAW = [(1500, 8.0), (1510, 7.35), (1520, 6.7), (1528, 8.6), (1540, 12.6), (1552, 16.3), (1564, 19.6),
           (1574, 21.8), (1584, 22.8), (1600, 23.0), (1700, 23.0)]
PYRE_W = [(1500, 0.0), (1562, 0.0), (1578, 0.40), (1586, 0.50), (1608, 0.0), (1700, 0.0)]   # lean toward the pyre
LOOK_PITCH = [(1500, -6.0), (1520, -6.0), (1540, -7.0), (1565, -4.5), (1585, -2.0), (1600, -1.5), (1625, -4.5),
              (1679, -9.5), (1700, -9.5)]
# after the pass: a world-anchored pan with the signal (each beat lands on its SFX side: 1600 right, 1620
# left of centre, 1640 right, 1660 just left of centre, the chains racing away on the left)
WORLD_YAW = [(1500, 9.5), (1588, 9.5), (1596, 6.2), (1604, 1.8), (1614, -3.2), (1624, -7.8), (1634, -11.4),
             (1644, -14.2), (1654, -16.6), (1666, -18.6), (1679, -20.0), (1700, -21.0)]
POST_W = [(1500, 0.0), (1588, 0.0), (1600, 1.0), (1700, 1.0)]                   # blend pre -> post look
PITCH_W = [(1500, 0.0), (1586, 0.0), (1614, 1.0), (1700, 1.0)]                  # the tilt down, unhurried
HORIZON_Y = 0.235                        # post-pass: the sea-of-cloud horizon held at this fraction of height
# the lens: 54 degrees until the pyre has gone by; then the operator eases in to 42 (the great peak, the 1620
# beacon and the start of its chain share one frame: the signal spreading across the range), and once the
# chain has raced away he zooms on in to 29 degrees, stacking the ranges to the horizon for the last beats.
# Each stage: a brisk start and a long gentle landing, in log focal length.
HFOV0, HFOVM, HFOV1 = 54.0, 42.0, 29.0
ZOOM1 = (1590.0, 1597.0, 1614.0)         # start, peak rate, end
ZOOM2 = (1632.0, 1642.0, 1666.0)


def _f_of(hfov, W=1920):
    return 0.5 * W / math.tan(math.radians(hfov) * 0.5)


_ZOOM = {}


def _ramp(frame, z):
    """0..1: velocity ramps up linearly to the peak, then eases to 0 with a quadratic landing (C1)."""
    if z not in _ZOOM:
        a, p, b = z
        fs = np.linspace(a, b, 2001)
        v = np.where(fs < p, (fs - a) / (p - a), (1.0 - (fs - p) / (b - p)) ** 2)
        c = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(fs))])
        _ZOOM[z] = (fs, c / c[-1])
    fs, c = _ZOOM[z]
    return float(np.interp(frame, fs, c))


def hfov(frame):
    f0, fm, f1 = _f_of(HFOV0), _f_of(HFOVM), _f_of(HFOV1)
    lf = math.log(f0) + math.log(fm / f0) * _ramp(frame, ZOOM1) + math.log(f1 / fm) * _ramp(frame, ZOOM2)
    return math.degrees(2.0 * math.atan(960.0 / math.exp(lf)))


def _horizon_pitch(frame, p):
    """Pitch that puts the cloud-sea horizon at HORIZON_Y of the frame height for the current lens."""
    f = _f_of(hfov(frame))
    dip = math.degrees(math.sqrt(2.0 * max(p[1] - WD.CLOUD_Y, 1.0) / WD.R_EARTH))
    return -(math.degrees(math.atan((0.5 - HORIZON_Y) * 804.0 / f)) + dip)


def _target_look(frame):
    p = path(frame)
    b = math.degrees(math.atan2(PYRE_XZ[0] - p[0], PYRE_XZ[2] - p[2]))
    y0 = pch(frame, PRE_YAW)
    rel = (b - y0 + 180.0) % 360.0 - 180.0
    rel = 34.0 * math.tanh(rel / 34.0)          # never chase it past the frame edge
    yaw_pre = y0 + pch(frame, PYRE_W) * rel
    pitch_pre = pch(frame, LOOK_PITCH)
    w = smoother(pch(frame, POST_W))
    yaw = yaw_pre * (1 - w) + pch(frame, WORLD_YAW) * w
    wp = smoother(pch(frame, PITCH_W))
    pitch = pitch_pre * (1 - wp) + _horizon_pitch(frame, p) * wp
    return yaw, pitch


_LOOK = None


def _smooth_look():
    """The operator's heavy head: critically damped follow of the target look (deterministic)."""
    df = 0.25
    fs = np.arange(1480.0, 1700.0 + 1e-9, df)
    w = 2 * math.pi * 1.1           # natural frequency (rad/s)
    y, p = _target_look(fs[0])
    vy = vp = 0.0
    out = np.zeros((len(fs), 2))
    dt = df / FPS
    for i, f in enumerate(fs):
        ty, tp = _target_look(f)
        ay = w * w * (ty - y) - 2 * w * vy
        ap = w * w * (tp - p) - 2 * w * vp
        vy += ay * dt
        vp += ap * dt
        y += vy * dt
        p += vp * dt
        out[i] = (y, p)
    return fs, out


def look(frame):
    global _LOOK
    if _LOOK is None:
        _LOOK = _smooth_look()
    fs, o = _LOOK
    return float(np.interp(frame, fs, o[:, 0])), float(np.interp(frame, fs, o[:, 1]))


def camera(frame, W=1920, H=804):
    pos = path(frame)
    yaw, pitch = look(frame)
    roll = bank(frame)
    return RC.RCam(pos, yaw, pitch, roll, hfov(frame), W, H)
