"""THE BEACON RUN (v2 frames 1520-1679, all cuts).

Hard cut from the shepherd's beacon: the camera leaps off his summit and runs low over the foothill
range that falls from it (s1's ridged range at a smaller scale, held under s1's summit-lip sight line so
s1 never saw it), terrain-following like a heavy glider -- over crests, through cols -- while beacons
flare on the beats on summits around and ahead. At 1580 a great pyre on a tower beside the track
ignites 75 m ahead; we sweep past it at eye level (1600) through its sparks, then pull up and rise to
reveal the chain racing away across the islands of the cloud sea to the horizon.

Camera = flight dynamics: horizontal speed / heading keys integrated into a ground track; altitude = a
terrain-following upper envelope (limited vertical acceleration) + clearance, then a zoom climb; bank from
the turn rate (coordinated turn, scaled for a camera mount); the operator's look is a lagged (critically
damped) blend of the flight direction and what he is framing (the pyre).
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
SPEED = [(1500, 40.0), (1520, 42.0), (1550, 92.0), (1590, 86.0), (1606, 84.0), (1640, 80.0), (1665, 66.0),
         (1679, 58.0), (1700, 54.0)]                            # horizontal speed (m/s)
HEADING = [(1500, 8.0), (1522, 8.0), (1552, 28.0), (1581, 28.0), (1626, -10.0), (1679, -19.0), (1700, -20.0)]
P0 = np.array([1.5, 3.5, 20.5])          # at START: just over the shepherd's summit lip
CLEAR = 24.0                             # metres above the terrain envelope under the track
A_DOWN = 20.0                            # m/s^2 the camera may fall away after a crest
A_UP = 11.0                              # m/s^2 it may pull up ahead of a crest
CLIMB0 = 1592.0                          # the zoom climb begins
CLIMB = [(1592, 0.0), (1604, 16.0), (1618, 44.0), (1652, 44.0), (1668, 30.0), (1679, 22.0), (1700, 18.0)]


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
    y = np.maximum(env, lip)
    k = np.exp(-0.5 * (np.arange(-24, 25) / 8.0) ** 2)
    k /= k.sum()
    y = np.convolve(np.pad(y, 24, mode='edge'), k, mode='same')[24:-24]
    y = np.maximum(y, H + 0.6 * clr)
    ic = int(round((CLIMB0 - 1500.0) / 0.125))
    yc = y.copy()
    for i in range(ic + 1, len(fs)):
        s_ = pch(fs[i] - 0.0625, SPEED)
        g = math.radians(pch(fs[i] - 0.0625, CLIMB))
        yc[i] = yc[i - 1] + s_ * math.tan(g) * 0.125 / FPS
    w = np.array([smoother((f - CLIMB0) / 12.0) for f in fs])
    y = np.maximum(y * (1 - w) + yc * w, np.where(fs >= CLIMB0, yc, y))
    return fs, np.stack([xz[:, 0], y, xz[:, 1]], 1), H


_PATH = None


def path(frame):
    fs, pos, _ = _PATH
    return np.array([np.interp(frame, fs, pos[:, k]) for k in range(3)])


def _bank_target(frame):
    h = 0.5
    dpsi = math.radians(heading(frame + h) - heading(frame - h)) / (2 * h / FPS)
    v = pch(frame, SPEED)
    return 0.22 * math.degrees(math.atan(v * dpsi / G))


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
PYRE_PASS = 1592.0                        # closest approach
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


def _pick_pyre(CR0):
    """The most prominent natural summit 20-40 m beside the track around the pass (left preferred), not far
    below the eye line."""
    best = None
    for f in np.arange(1598.0, 1609.0, 2.0):
        c = path(f)
        for side in (1.0,):
            for lat in (14.0, 18.0, 22.0, 26.0, 30.0):
                q = c + _left(f) * lat * side
                sm = WD.summit_near(q[0], q[2], CR0, rad=9.0, n=11)
                ring = np.mean([WD.ground(sm[0] + 16.0 * math.cos(a), sm[2] + 16.0 * math.sin(a), CR0)
                                for a in np.linspace(0, 2 * math.pi, 9)[:-1]])
                prom = sm[1] - ring
                dxz = math.hypot(sm[0] - c[0], sm[2] - c[2])
                if dxz < 12.0 or dxz > 32.0:
                    continue
                score = prom + (10.0 if side > 0 else 0.0) - 0.35 * abs(sm[1] - (c[1] - PYRE_BELOW)) \
                    - 0.4 * abs(f - PYRE_PASS)
                if best is None or score > best[0]:
                    best = (score, sm, f, side, prom)
    return best


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
LOOK_YAW = [(1500, 0.0), (1530, -2.0), (1560, -3.0), (1600, -6.0), (1640, -4.0), (1700, -3.0)]
PYRE_W = [(1500, 0.0), (1562, 0.0), (1578, 0.40), (1586, 0.50), (1608, 0.0), (1700, 0.0)]   # lean toward the pyre
LOOK_PITCH = [(1500, -6.0), (1520, -6.0), (1540, -7.0), (1565, -4.5), (1585, -2.0), (1600, -1.5), (1625, -4.5),
              (1679, -9.5), (1700, -9.5)]
HFOV = 54.0


def _target_look(frame):
    hd = heading(frame)
    p = path(frame)
    b = math.degrees(math.atan2(PYRE_XZ[0] - p[0], PYRE_XZ[2] - p[2]))
    rel = (b - hd + 180.0) % 360.0 - 180.0
    rel = 34.0 * math.tanh(rel / 34.0)          # never chase it past the frame edge
    yaw = hd + pch(frame, LOOK_YAW) + pch(frame, PYRE_W) * rel
    pitch = pch(frame, LOOK_PITCH)
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
    return RC.RCam(pos, yaw, pitch, roll, HFOV, W, H)
