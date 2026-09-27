"""THE CROSSING (cut A, R6; A's showpiece #2). 960 frames: A cut frames 4880-5839 (renders/crossing_A/, EDIT-v3).

Match cut on the heart of light: the same light, now inside a great lantern of iron and glass hung between two
poles on the shoulders of two hooded bearers. The camera draws back and out over the void, never getting ahead
of the lantern, to a side-on wide: a knife-edge arete above the cloud sea, about forty people roped together
along it, each with a small orange lantern, between two watch-fires. The line walks at the pace of the great
lantern's pool of light (0.45 m/s, a step on every beat at 72 BPM), and so does the camera.

Two clocks. The people keep real time. The sky keeps a second, faster clock that eases in once the camera has
drawn back: the stars wheel about the pole and draw trails, the moon sets behind the arete, the watch-fires
burn low and are fed, the red under the cloud goes out patch by patch, the lantern's fire warms from ice-white
toward gold, and the path behind the line keeps a faint trail of light. By the end it is the dark hour before
the blue hour.

  python crossing.py --frames 0,240,480 --numbering shot --scale 0.5 --out t1   # test stills -> renders/crossing_A/t1/
  python crossing.py --range 4880-5839 --procs 4 --skip                          # finals, A cut frames -> crossing_A/
  --variant few   the fallback test: 12 walkers, larger (camera closer)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import fire2 as F2          # noqa: E402
import sdfppl as SP         # noqa: E402
from mt import sky as SK, fire as F   # noqa: E402


def smoothstep(a, b, x):
    """numpy smoothstep (scalars or arrays; mt.noise's is numba-scalar only)."""
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    r = t * t * (3.0 - 2.0 * t)
    return float(r) if r.ndim == 0 else r


FPS = 24.0
NFR = 960
lin = PI.CM.lin
UP = np.array([0.0, 1.0, 0.0])

# ------------------------------------------------------------------ the set frame ---
C0 = np.array([-5200.0, 14400.0])            # a clear basin of the cloud sea, islands to the north
YAW_E = -30.0                                # walking direction = the world's sunrise azimuth ("east")
_a = math.radians(YAW_E)
E3 = np.array([math.sin(_a), 0.0, math.cos(_a)])     # east (along the arete, the way they walk)
S3 = np.array([math.cos(_a), 0.0, -math.sin(_a)])    # south (the camera's side; right when facing east)
LAT = 6.0                                    # latitude: the pole sits low in the north, well inside the wide,
                                             # above the great range (it was 8: the pole sat on the frame's top edge)
_nh = -S3
POLE = np.array([_nh[0] * math.cos(math.radians(LAT)), math.sin(math.radians(LAT)),
                 _nh[2] * math.cos(math.radians(LAT))])


def uv(u, v, y=0.0):
    p = C0 + E3[[0, 2]] * u + S3[[0, 2]] * v
    return np.array([p[0], y, p[1]])


# crest knots (u along the arete, v lateral wander, crest height)
KNOTS = [(-330.0, 0.0, 188.0), (-235.0, 7.0, 158.0), (-150.0, -3.0, 142.0), (-70.0, 4.0, 131.0),
         (-10.0, -2.0, 126.0), (60.0, 3.0, 133.0), (150.0, -4.0, 158.0), (255.0, 0.0, 206.0)]
# the four watch-fires the line passes (A18: bars 64, 67, 70, 73 = shot frames 160, 400, 640, 880). The first stands
# on a shoulder beside the path, so the bearers walk past it; the others stand on pinnacles beyond the arete, each
# placed on the camera's line of sight through the lantern at its bar, so the lantern passes in front of it
PASS_FRAMES = (160, 400, 640, 880)
PASS_BEYOND = (None, 70.0, 160.0, 300.0)       # metres beyond the lantern along that line of sight


def build_arete():
    rows = []
    for k in range(len(KNOTS) - 1):
        a, b = KNOTS[k], KNOTS[k + 1]
        pa, pb = uv(a[0], a[1], a[2]), uv(b[0], b[1], b[2])
        # detail 0.14 (was 0.24): the deep gullies read as combed streaks on the steep flanks at close range
        rows.append(WD.ridge_row(pa, pb, wl=15.0, wr=13.0, seed=41 + 3 * k, k=6.0, detail=0.14, slope=1.9))
    for (u, top, seed, ang) in ((-338.0, 196.0, 7, 0.4), (262.0, 214.0, 11, -0.3)):
        p = uv(u, 0.0)
        rows.append(WD.crag_row(p[0], p[2], top, L=16.0, s_hi=2.4, s_lo=1.35, aniso=1.4, ang=math.radians(YAW_E) + ang,
                                seed=seed, k=7.0, detail=0.3, shelf=2.0, nf=4))
    return np.array(rows)


def _knot_y(u):
    us = [k[0] for k in KNOTS]
    ys = [k[2] for k in KNOTS]
    return float(np.interp(u, us, ys))


CR0 = build_arete()
CR0[:, 14] = 0.95                                # crags and ridges: the safe early-out bound (world, v3 flag)
CR0[CR0[:, 12] >= 0.0, 13] *= 2.5                # crags: a cutoff radius beyond which every flank is under the cloud
CR = CR0                                         # replaced by CR0 + the watch-fire sites + the great range (below)

# ------------------------------------------------------------------ A's landform: the great range ---
# The islands north of the arete are a field of small, uniform needles (a terrain tell in the wides). One great
# knife-edge range stands across the view 7 km out, under the celestial pole: a serrated crest with one dominant,
# asymmetric summit left of the pole, stepping down to the right into the open cloud sea. A second, older range
# 18 km out on the right stands pale in the haze. Aerial depth: dark near islands, the great range, the far
# range, the sky. (u, v) are the arete frame's metres: u east along the line, v south, the wide's camera at
# about (-11, 62), the pole due north of it (u = -11).
GR_V = -6900.0
GR_U0 = -1120.0                                  # the dominant summit (about 9 deg left of the pole in the wide)
GR_KNOTS = [(-2700, 250), (-2350, 360), (-2000, 330), (-1700, 420), (-1400, 385), (-1150, 470), (-900, 440),
            (-650, 500), (-420, 465), (-250, 540), (-40, 560), (180, 505), (400, 450), (650, 468), (900, 400),
            (1150, 372), (1400, 300), (1650, 248), (1950, 120), (2250, -160)]
FR_V = -17500.0
FR_KNOTS = [(1900, 150), (2400, 420), (3000, 520), (3600, 470), (4300, 600), (5000, 540), (5700, 575),
            (6400, 470), (7200, 380), (8000, 150)]


def build_ranges():
    rows = []
    crest = []
    rng = np.random.default_rng(606)
    for knots, v0, u0, sd in ((GR_KNOTS, GR_V, GR_U0, 300), (FR_KNOTS, FR_V, 0.0, 400)):
        pts = []
        for du, y in knots:
            # the crest wanders (never a straight wall) and bows away from the camera at its ends
            vv = v0 + rng.uniform(-140.0, 140.0) - 4.0e-5 * du * du
            pts.append(uv(u0 + du, vv, y))
        far = v0 < -10000.0
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            rows.append(WD.ridge_row(a, b, wl=(70.0 if far else 46.0), wr=(64.0 if far else 42.0), seed=sd + k,
                                     k=12.0, detail=(0.16 if far else 0.22), slope=(1.45 if far else 1.75)))
            rows[-1][14] = 0.95
        crest += pts if not far else []
        # faceted spires of unequal height, width and lean on the high points; one dominant, asymmetric summit
        hi = [k for k in range(1, len(pts) - 1) if pts[k][1] > max(pts[k - 1][1], pts[k + 1][1])]
        main = max(hi, key=lambda k: pts[k][1])
        for k in hi:
            if k != main and rng.random() < 0.4:
                continue
            big = (k == main) and not far
            rows.append(WD.crag_row(pts[k][0] + rng.uniform(-50.0, 50.0), pts[k][2] + rng.uniform(-50.0, 50.0),
                                    pts[k][1] + (70.0 if big else rng.uniform(12.0, 45.0)),
                                    L=(110.0 if big else rng.uniform(25.0, 55.0)),
                                    s_hi=(2.3 if big else rng.uniform(2.4, 3.4)), s_lo=1.7,
                                    aniso=(2.1 if big else rng.uniform(1.2, 2.0)),
                                    ang=math.radians(YAW_E) + rng.uniform(-0.6, 0.6), seed=sd + 50 + k,
                                    k=(26.0 if big else 14.0), detail=0.34, shelf=0.0, nf=(4 if big else 3)))
            rows[-1][14] = 0.95
            rows[-1][13] *= 2.5          # every flank reaches the cloud inside the cutoff (no walls)
    return np.array(rows), crest


GR_ROWS, GR_CREST = build_ranges()


def ground_many(P, fp=0.02, CRx=None):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    out = np.zeros(len(P))
    WD.heights(P[:, 0].copy(), P[:, 2].copy(), fp, CR if CRx is None else CRx, out)
    return out


# ------------------------------------------------------------------ the path on the crest ---
_PATH = None


def _build_path():
    us = np.arange(-300.0, 230.0, 0.5)
    vs = np.linspace(-14.0, 14.0, 113)
    UU, VV = np.meshgrid(us, vs, indexing='ij')
    pts = C0[None, None, :] + E3[[0, 2]] * UU[..., None] + S3[[0, 2]] * VV[..., None]
    h = np.zeros(UU.size)
    WD.heights(pts[..., 0].ravel().copy(), pts[..., 1].ravel().copy(), 0.02, CR0, h)
    h = h.reshape(UU.shape)
    vbest = vs[np.argmax(h, axis=1)]
    from scipy.ndimage import gaussian_filter1d
    v = gaussian_filter1d(vbest, 6.0, mode='nearest')          # sigma 3 m
    P = np.stack([uv(u, vv) for u, vv in zip(us, v)])
    y = ground_many(P, CRx=CR0)
    y = np.maximum(gaussian_filter1d(y, 1.2, mode='nearest'), y - 0.04)
    P[:, 1] = y
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    return s, P


def path():
    global _PATH
    if _PATH is None:
        _PATH = _build_path()
    return _PATH


def at(sq):
    """Point on the walking path at arc length sq (vectorised) and the walking direction there."""
    s, P = path()
    sq = np.asarray(sq, np.float64)
    p = np.stack([np.interp(sq, s, P[:, k]) for k in range(3)], -1)
    q = np.stack([np.interp(sq + 0.6, s, P[:, k]) for k in range(3)], -1)
    w = q - p
    w[..., 1] = 0.0
    w /= np.linalg.norm(w, axis=-1, keepdims=True) + 1e-9
    return p, w


def s_of_u(u):
    s, P = path()
    uu = (P[:, [0, 2]] - C0) @ E3[[0, 2]]
    return float(np.interp(u, uu, s))


# ------------------------------------------------------------------ the line ---
SPEED = 0.45                                  # m/s: a step on every beat (72 BPM), 0.375 m steps
S_L0 = None                                   # arc length of the lantern at frame 0 (set from u = -6 m)


def variant_cfg(variant):
    if variant == 'few':
        return dict(n=12, gap=4.2, jit=0.5, r1=52.0, dT=12.0, hfov1=46.0, h1=1.6, r0=9.5)
    return dict(n=40, gap=2.6, jit=0.35, r1=64.0, dT=17.0, hfov1=48.0, h1=2.2, r0=9.5)


def _hh(i, j=0):
    a = (i * 73856093 ^ j * 19349663) & 0xFFFFFFFF
    a = ((a ^ 61) ^ (a >> 16)) & 0xFFFFFFFF
    a = (a * 0x27D4EB2D) & 0xFFFFFFFF
    return ((a ^ (a >> 15)) & 0xFFFFFFFF) / 4294967296.0


def lantern_s(t):
    global S_L0
    if S_L0 is None:
        S_L0 = s_of_u(-6.0)
    return S_L0 + SPEED * t


def line_s(t, cfg):
    """Arc lengths of: front bearer, rear bearer, then the roped walkers. The spacing breathes (an accordion:
    people pause and close up on a knife-edge), never faster than the lantern."""
    sl = lantern_s(t)
    out = [sl + 1.3, sl - 1.3]
    cur = sl - 1.3 - 2.4
    for k in range(cfg['n']):
        T = 11.0 + 9.0 * _hh(k, 1)
        g = cfg['gap'] + cfg['jit'] * (_hh(k, 2) - 0.5) * 2.0 + 0.32 * math.sin(2 * math.pi * t / T + 6.28 * _hh(k, 3))
        out.append(cur)
        cur -= g
    return np.array(out)


STANCE = 0.62                                    # fraction of the cycle a foot is planted (a slow, careful walk)


def stride_of(i):
    """Stride (m) of figure i (0, 1 = the bearers: short, loaded steps)."""
    return 0.62 if i < 2 else 0.75 * (0.92 + 0.16 * _hh(i, 5))


def foot_cycle(s, S, ph, delta):
    """Where one foot is along the path (arc length) and how high it is lifted, for a body at arc s. A planted foot
    stays exactly where it was put (no sliding); it is under the body at mid-stance and swings to the next plant."""
    v = s / S + ph + delta
    k = math.floor(v)
    c = v - k
    sp = S * (k + 0.5 * STANCE - ph - delta)
    if c < STANCE:
        return sp, 0.0
    u = (c - STANCE) / (1.0 - STANCE)
    return sp + S * u * u * (3.0 - 2.0 * u), 0.10 * math.sin(math.pi * u)


def walk_phase(s, i):
    """Cycle phase (radians) of figure i at arc s, for the cloak's swing."""
    return 2 * math.pi * (s / stride_of(i) + _hh(i, 6))


def plant(figs):
    """Feet and pelvises for figures [(arc s, index i, height h, crouch m)]: each foot planted on the real terrain
    (one height query for all), the pelvis lowered wherever a leg could not otherwise reach its foot."""
    feet = []
    for (s_, i, h, cr) in figs:
        S = stride_of(i)
        ph = _hh(i, 6)
        for delta, side in ((0.0, 1.0), (0.5, -1.0)):
            sf, lift = foot_cycle(s_, S, ph, delta)
            feet.append((sf, side, lift, h))
    sfs = np.array([f[0] for f in feet])
    P, Wd = at(sfs)
    lat = np.cross(UP[None, :], Wd)
    Q = P + lat * np.array([f[1] * 0.10 * f[3] for f in feet])[:, None]
    gy = ground_many(Q)
    Q[:, 1] = np.maximum(gy, P[:, 1] - 0.05) + np.array([0.075 * f[3] + f[2] for f in feet])
    out = []
    for n_, (s_, i, h, cr) in enumerate(figs):
        pb, wb = at(s_)
        aL, aR = Q[2 * n_], Q[2 * n_ + 1]
        latb = np.cross(UP, wb)
        pel = pb + UP * (0.93 * h - cr)
        for a, side in ((aL, 1.0), (aR, -1.0)):
            hip = pel + latb * side * 0.085 * h
            dxz = math.hypot(a[0] - hip[0], a[2] - hip[2])
            L = 0.91 * h * 0.985
            pel[1] = min(pel[1], a[1] + math.sqrt(max(L * L - dxz * dxz, 0.0)))
        out.append((pel, aL, aR, wb))
    return out


# ------------------------------------------------------------------ the two clocks ---
SKY_DEG = 46.0                                # the sky turns ~3 hours over the shot


T_WHEEL = 320 / FPS                              # bar 66 b1: the sky begins to wheel (T13 lands with it)
T_WIDE = 560 / FPS                               # bar 69 b1: the camera has settled; star trails may open


def _rate(t):
    return smoothstep(T_WHEEL, T_WHEEL + 6.0, t) * (1.0 - 0.35 * smoothstep(31.0, 40.0, t))


_CLK = None


def sky_angle(t):
    """Degrees the sky has turned by shot time t (s): eases in after the draw-back begins."""
    global _CLK
    if _CLK is None:
        ts = np.linspace(0.0, NFR / FPS + 1.0, 2001)
        r = np.array([_rate(x) for x in ts])
        c = np.r_[0.0, np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(ts))]
        c *= SKY_DEG / np.interp(NFR / FPS, ts, c)
        _CLK = (ts, c)
    return float(np.interp(t, _CLK[0], _CLK[1]))


def rot_pole(d, ang_deg):
    th = math.radians(ang_deg)
    d = np.asarray(d, np.float64)
    c, s_ = math.cos(th), math.sin(th)
    return d * c + np.cross(POLE, d) * s_ + np.outer(d @ POLE, POLE).reshape(d.shape) * (1 - c) if d.ndim > 1 else \
        d * c + np.cross(POLE, d) * s_ + POLE * (d @ POLE) * (1 - c)


_mz = math.radians(-166.0)
_mel = math.radians(36.0)
MOON0 = np.array([math.cos(_mel) * math.sin(_mz), math.sin(_mel), math.cos(_mel) * math.cos(_mz)])


def moon_dir(t):
    return rot_pole(MOON0, sky_angle(t))


def heart_col(t):
    """The lantern's fire: ice-white warming toward gold on the sky clock."""
    u = smoothstep(4.0, SKY_DEG, sky_angle(t))
    c0 = np.array([0.80, 0.92, 1.00])
    c1 = np.array([1.00, 0.80, 0.48])
    return c0 * (1 - u) + c1 * u


def breath(t):
    return 1.0 + 0.10 * math.sin(2 * math.pi * t / (80.0 / FPS))      # once a bar


# ------------------------------------------------------------------ camera ---
def _ss(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


LANT_K = 0.85                                 # the great lantern: ~0.9 m from foot to finial ring
RING = (0.30 + 0.33) * LANT_K                 # heart -> finial ring


def heart_pos(t, cfg=None):
    sl = lantern_s(t)
    p, w = at(sl)
    # poles on the bearers' shoulders ~1.47 m over the path; crossbar +0.03, a 0.10 m hook, then the ring
    return p + UP * (1.50 - 0.10 - RING) + _swing(t) * w


def _swing(t):
    return 0.035 * math.sin(2 * math.pi * 0.6 * t - 0.8)


def camera(frame, W=1920, H=804, cfg=None):
    cfg = cfg or variant_cfg('main')
    t = frame / FPS
    L = heart_pos(t)
    # bar 63 b1 (3.33 s): the draw-back begins (a cubic ease); it settles on bar 69 b1 (23.33 s), in time for the
    # narrowest stretch; the swing from behind to side-on runs alongside it
    e1 = _ss((t - 80 / FPS) / 20.0)
    e2 = _ss((t - 80 / FPS) / 19.0)
    r = cfg['r0'] * (cfg['r1'] / cfg['r0']) ** e1
    # side-on from the first frame, a little behind the lantern and below the crest, so the bearers stand in
    # profile on the crest's skyline against the stars, feet on the snow; the draw-back rises to crest level
    beta = math.radians(72.0 + 4.0 * e2)
    h = -0.9 + (cfg['h1'] + 0.9) * e1 ** 1.3
    pos = L + (-E3 * math.cos(beta) + S3 * math.sin(beta)) * r + UP * h
    hfov = 38.0 + (cfg['hfov1'] - 38.0) * e1
    # frame the lantern directly: centred for the match cut, easing to 80% across the frame in the wide (the line
    # trails away to the left), a little below the centre so the sky has room to turn
    d = L - pos
    x_frac = 0.5 + 0.30 * e1
    off = math.degrees(math.atan(math.tan(math.radians(hfov) * 0.5) * (2.0 * x_frac - 1.0)))
    yaw = math.degrees(math.atan2(d[0], d[2])) - off
    pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2]))) + 3.0 * e1
    return RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)


# ------------------------------------------------------------------ the red under the cloud ---
_UG = None


def _ug_table():
    global _UG
    if _UG is None:
        rng = np.random.default_rng(606)
        rows = []
        for k in range(26):
            # the cloud sea still seen from the arete: the basin in front of the great range (2-6 km), and the
            # open distance to its right (east of north) and beyond the frame's left edge
            if k % 3 == 1:
                dist = 2000.0 + 4200.0 * rng.random()
                ang = math.radians(rng.uniform(-40.0, 40.0))
                radk = 0.35
            else:
                dist = 10000.0 * (4.4 ** rng.random())                # 10 - 44 km
                ang = math.radians(rng.uniform(0.0, 50.0) if rng.random() < 0.75 else rng.uniform(-60.0, -28.0))
                radk = 1.0
            dirv = -S3 * math.cos(ang) + E3 * math.sin(ang)
            c = uv(0.0, 0.0) + dirv * dist
            rad = rng.uniform(1400.0, 3600.0) * (0.6 + 0.4 * dist / 30000.0) * radk
            out_at = rng.uniform(655.0, 945.0) / FPS if k >= 3 else rng.uniform(1000.0, 1300.0) / FPS
            rows.append([c[0], c[2], rad, out_at, rng.uniform(0.6, 1.0), rng.uniform(0, 6.28)])
        _UG = np.array(rows)
    return _UG


def underglow(t):
    """Patch table for WD.cloud_glow at shot time t: steady until bar 70 b1 (f640), then each patch goes out at its
    own time, with a last flicker."""
    T = _ug_table()
    out = []
    col = np.array([1.0, 0.34, 0.07]) * 1.4      # orange-red: crimson over blue moonlit cloud reads magenta
    for x, z, rad, a_out, g, ph in T:
        live = 1.0 - smoothstep(a_out - 1.4, a_out, t)
        if live <= 0.0:
            continue
        fl = 1.0 + 0.25 * math.sin(7.0 * t + ph) * smoothstep(a_out - 3.0, a_out - 1.0, t) * live
        I = g * live * fl
        out.append([x, z, rad, col[0] * I, col[1] * I, col[2] * I, 900.0, 0.35])
    return np.array(out, np.float64).reshape(-1, 8)


# ------------------------------------------------------------------ watch-fires ---
def _fire_sites():
    """The four watch-fires and the rock that carries each (CR rows). Built from the main variant's camera."""
    cfg = variant_cfg('main')
    sites = []
    rows = []
    for k, (f, D) in enumerate(zip(PASS_FRAMES, PASS_BEYOND)):
        t = f / FPS
        L = heart_pos(t)
        if D is None:
            # a rock shoulder just beyond the crest (away from the camera), level with the lantern at its bar:
            # the bearers pass in front of its fire as backlit silhouettes
            p, w = at(lantern_s(t))
            north = np.cross(UP, w)
            q = p + north * 2.6 + w * 0.3
            top = p[1] - 0.40
            rows.append(WD.crag_row(q[0], q[2], top, L=3.0, s_hi=2.2, s_lo=1.7, aniso=1.25,
                                    ang=math.radians(YAW_E), seed=31, k=2.5, detail=0.2, shelf=1.2, nf=4))
            sites.append(np.array([q[0], top, q[2]]))
        else:
            cam = camera(f, cfg=cfg).pos
            d = L - cam
            d /= np.linalg.norm(d)
            q = L + d * D
            top = q[1] - 0.9
            rows.append(WD.crag_row(q[0], q[2], top, L=10.0 + 3.0 * k, s_hi=2.8, s_lo=1.55, aniso=1.3,
                                    ang=math.radians(YAW_E + 25.0 * k), seed=50 + 7 * k, k=6.0, detail=0.24,
                                    shelf=1.5, nf=4))
            sites.append(np.array([q[0], top, q[2]]))
    return sites, np.array(rows)


def _finish_set():
    global CR, _WF
    sites, rows = _fire_sites()
    rows[:, 14] = 0.95
    rows[:, 13] *= 2.5
    CR = np.vstack([CR0, rows, GR_ROWS])
    # each fire stands on the highest point of its rock's summit shelf
    out = []
    for q in sites:
        best = None
        for dv in np.linspace(-1.6, 1.6, 9):
            for du in np.linspace(-1.6, 1.6, 9):
                c = q + np.array([du, 0.0, dv])
                c[1] = ground_many(c)[0]
                if best is None or c[1] > best[1]:
                    best = c
        out.append(best)
    _WF = out


_WF = None


def wfs():
    if _WF is None:
        _finish_set()
    return _WF


def wf_burn(t, k):
    """Watch-fire k: burns low on the sky clock and is fed (a flare) every ~13 degrees of sky; before the sky
    wheels it simply burns."""
    a = sky_angle(t) + 4.0 * k
    period = 13.0
    ph = (a % period) / period
    since = ph * period
    b = 0.32 + 0.68 * math.exp(-since / 5.5)
    w = smoothstep(0.0, 2.0, sky_angle(t))
    return (1.0 - w) * 0.85 + w * b, ph if w > 0.5 else 0.0


_FAR = None


def far_fires():
    """Summit watch-fires on the islands north of the arete (the watch kept all around): cached to disk."""
    global _FAR
    if _FAR is None:
        fn = os.path.join(HERE, 'crossing_fires.npy')
        if os.path.exists(fn):
            _FAR = np.load(fn)
        else:
            n = 260
            dx = 70.0
            xs = C0[0] + (np.arange(n) - n / 2) * dx
            zs = C0[1] + (np.arange(n) - n / 2) * dx
            XX, ZZ = np.meshgrid(xs, zs)
            h = np.zeros(XX.size)
            WD.heights(XX.ravel().copy(), ZZ.ravel().copy(), 35.0, np.zeros((0, WD.NCR)), h)
            h = h.reshape(XX.shape)
            from scipy import ndimage as ndi
            mx = ndi.maximum_filter(h, size=15)
            pts = []
            for j, i in np.argwhere((h == mx) & (h > -330.0)):
                p = np.array([XX[j, i], h[j, i], ZZ[j, i]])
                d = p - uv(0.0, 0.0)
                north = -(d[[0, 2]] @ S3[[0, 2]])
                if north < 1500.0:
                    continue
                pts.append([p[0], p[1], p[2], h[j, i] - np.min(h[max(0, j - 8):j + 9, max(0, i - 8):i + 9])])
            _FAR = np.array(pts)
            np.save(fn, _FAR)
    return _FAR


_RF = None


def range_fires():
    """Three watch-fires on the great range's crest (in notches and on a shoulder, never on the summit spire)."""
    global _RF
    if _RF is None:
        out = []
        for kn in (4, 8, 16):
            p = GR_CREST[kn]
            best = None
            for du in np.linspace(-30.0, 30.0, 7):
                for dv in np.linspace(-30.0, 30.0, 7):
                    q = np.array([p[0] + du, 0.0, p[2] + dv])
                    q[1] = ground_many(q, fp=0.5)[0]
                    if best is None or q[1] > best[1]:
                        best = q
            out.append([best[0], best[1], best[2], 50.0])
        _RF = np.array(out)
    return _RF


# ------------------------------------------------------------------ star trails ---
_STARS = None


def stars():
    global _STARS
    if _STARS is None:
        _STARS = SK.make_stars(14000, 101, lum_scale=7.0)
    return _STARS


@njit(fastmath=True, cache=True)
def _aa_line(img, zb, x0, y0, x1, y1, z, w, r, g, b, zbias):
    """Additive anti-aliased line (width w px, energy per unit length ~ r,g,b), depth tested at z."""
    H, W = img.shape[0], img.shape[1]
    dx = x1 - x0
    dy = y1 - y0
    L = math.sqrt(dx * dx + dy * dy)
    n = int(L / 0.7) + 1
    rad = max(w, 0.5)
    for k in range(n + 1):
        u = k / n
        px = x0 + dx * u
        py = y0 + dy * u
        seg = L / (n + 1) if L > 0 else 1.0
        ix0 = int(px - rad - 1)
        iy0 = int(py - rad - 1)
        for yy in range(iy0, iy0 + int(2 * rad + 3)):
            if yy < 0 or yy >= H:
                continue
            for xx in range(ix0, ix0 + int(2 * rad + 3)):
                if xx < 0 or xx >= W:
                    continue
                if zb[yy, xx] < z - zbias:
                    continue
                ddx = xx + 0.5 - px
                ddy = yy + 0.5 - py
                q = math.exp(-(ddx * ddx + ddy * ddy) / (2.0 * rad * rad * 0.36)) * seg / (2.5 * rad * 0.6)
                img[yy, xx, 0] += r * q
                img[yy, xx, 1] += g * q
                img[yy, xx, 2] += b * q


@njit(fastmath=True, cache=True)
def _trails(img, mask, SX, SY, LUM, COL, sig):
    """Star trails: SX, SY (N, M) screen positions along each star's arc (oldest -> newest), LUM (N, M) energy per
    segment; splatted as a chain of soft segments, sky pixels only."""
    H, W = img.shape[0], img.shape[1]
    N, M = SX.shape
    for k in range(N):
        for m in range(M - 1):
            x0, y0, x1, y1 = SX[k, m], SY[k, m], SX[k, m + 1], SY[k, m + 1]
            if (x0 < -4 and x1 < -4) or (y0 < -4 and y1 < -4) or (x0 > W + 4 and x1 > W + 4) or (y0 > H + 4 and y1 > H + 4):
                continue
            dx = x1 - x0
            dy = y1 - y0
            L = math.sqrt(dx * dx + dy * dy)
            n = int(L / 0.6) + 1
            e = LUM[k, m] / n
            rad = int(math.ceil(2.5 * sig))
            for q in range(n):
                u = (q + 0.5) / n
                px = x0 + dx * u
                py = y0 + dy * u
                ix = int(math.floor(px))
                iy = int(math.floor(py))
                norm = e / (6.2832 * sig * sig)
                for yy in range(iy - rad, iy + rad + 1):
                    if yy < 0 or yy >= H:
                        continue
                    for xx in range(ix - rad, ix + rad + 1):
                        if xx < 0 or xx >= W:
                            continue
                        m2 = mask[yy, xx]
                        if m2 <= 0.0:
                            continue
                        ddx = xx + 0.5 - px
                        ddy = yy + 0.5 - py
                        wgt = math.exp(-(ddx * ddx + ddy * ddy) / (2 * sig * sig)) * norm * m2
                        img[yy, xx, 0] += wgt * COL[k, 0]
                        img[yy, xx, 1] += wgt * COL[k, 1]
                        img[yy, xx, 2] += wgt * COL[k, 2]


T_TRAIL0 = T_WHEEL                               # the trails open with the wheel (bar 66 b1, T13)
EXPO_MAX = 36.0                                  # degrees of arc a trail keeps (then its oldest end moves on)


def draw_sky(fr, t, pxs):
    """Stars on the sky clock: points until the sky begins to wheel (bar 66 b1), then trails that grow into long
    concentric arcs about the pole, which is inside the frame from that bar on (so no trail is ever a loose dash:
    gate, 'rain / warp'). The newest end is the star; the arc fades with age."""
    scam = fr.src
    st = stars()
    a = sky_angle(t)
    mask = (fr.dist > 1e8).astype(np.float32)
    expo = float(np.clip(a - sky_angle(T_TRAIL0), 0.0, EXPO_MAX))
    M = 2 if expo < 0.05 else int(min(10 + 1.8 * expo, 72))
    angs = a - expo * (1.0 - np.linspace(0.0, 1.0, M))
    D = st['dir']
    # rotate every star for every sample (vectorised Rodrigues)
    c = np.cos(np.radians(angs))[None, :, None]
    s_ = np.sin(np.radians(angs))[None, :, None]
    Dd = D[:, None, :]
    cr = np.cross(POLE[None, None, :], Dd)
    dp = (D @ POLE)[:, None, None]
    R = Dd * c + cr * s_ + POLE[None, None, :] * dp * (1 - c)
    x = R[..., 0] * scam.right[0] + R[..., 2] * scam.right[2]
    z = R[..., 0] * scam.fwd[0] + R[..., 2] * scam.fwd[2]
    y = R[..., 1]
    ok = (z > 0.05).all(axis=1) & (y > -0.05).any(axis=1)
    x, y, z = x[ok], y[ok], z[ok]
    sx = scam.cx + scam.f * x / z
    sy = scam.cyy - scam.f * y / z
    lum = (st['lum'][ok] / 7.0) ** 0.62 * 7.0 * 0.30 * fr.ss * fr.ss
    ext = np.clip(y / 0.12, 0, 1) ** 1.5
    tw = 1.0 + 0.25 * np.sin(t * 9.0 + st['phase'][ok]) * np.minimum(lum / 0.05, 1.0)
    head = lum * tw * ext[:, -1]
    col = st['col'][ok]
    sig = max(0.55 * pxs, 0.45)
    if expo < 0.05:
        SK._splat(fr.img, sx[:, -1].astype(np.float64), sy[:, -1].astype(np.float64), head.astype(np.float64), col,
                  np.full(len(head), sig), mask)
        return
    # the head (current position) stays a point; the arc behind it is the exposure, fading with age. Energy per
    # segment grows with its length on screen (^0.75), so a trail far from the pole is not a faint scratch while
    # one near it burns (a real stack keeps a little of that: the inner trails are the brighter)
    wage = np.exp(-(1.0 - np.linspace(0.0, 1.0, M)) * 1.25)[None, :]
    segl = np.hypot(np.diff(sx, axis=1), np.diff(sy, axis=1))
    seg_e = lum[:, None] * ext[:, :-1] * wage[:, :-1] * 0.10 * np.maximum(segl, 0.3) ** 0.75 \
        * (0.35 + 0.65 * min(expo / 4.0, 1.0))
    _trails(fr.img, mask, sx.astype(np.float64), sy.astype(np.float64), seg_e.astype(np.float64), col,
            float(sig))
    SK._splat(fr.img, sx[:, -1].astype(np.float64), sy[:, -1].astype(np.float64), (head * 0.6).astype(np.float64),
              col, np.full(len(head), sig), mask)


# ------------------------------------------------------------------ lights ---
SMALL_COL = np.array([1.0, 0.52, 0.20])      # the people's own fires: orange


def small_lantern_I(k, t):
    return 0.50 * (0.85 + 0.3 * _hh(k, 9)) * F.flicker(t, 100 + k, 0.5)


def night(t):
    Lk, amb, S, fogp, Q = WD.night_light()
    md = moon_dir(t)
    el = math.degrees(math.asin(md[1]))
    mf = smoothstep(-1.0, 7.0, el)
    Lk = Lk.copy()
    Lk[0:3] = md
    warm = 1.0 - smoothstep(2.0, 14.0, el)
    Lk[3:6] = Lk[3:6] * (1 - 0.35 * warm) + np.array([0.75, 0.55, 0.40]) * 0.35 * warm
    Q = Q.copy()
    Q[0] = 0.55 * mf
    Q[18] = 1.0                                  # anti-streak snow noise on the steep flanks (world.shade flag)
    Q[3] = 0.30                                  # snow holds on steeper ground: the upper flanks read as fluted
    Q[4] = 0.54                                  # snow, not black-and-white combed stripes
    fogp = fogp.copy()
    fogp[0] = 1.0e-4                             # aerial depth: near islands dark, the great range 7 km out half
                                                 # veiled, the far range and the needles beyond it pale
    amb = amb * (0.55 + 0.45 * mf)
    S = SK.sky_params(zenith='#070B1C', horizon='#2A3866', moon_dir=md, moon_radius_deg=0.8,
                      halo_I=0.025 * mf, halo_w=0.22, halo2_I=0.012 * (0.3 + 0.7 * mf), halo2_w=0.7,
                      horizon_glow=0.25 * (0.6 + 0.4 * mf), gain=0.75 + 0.25 * mf)
    return (Lk, amb, S, fogp, Q), mf, md


# ------------------------------------------------------------------ the wind on the arete ---
# A steady headwind down the arete from the east-north-east (they walk into it), with gusts that travel along the
# line, so no two cloaks move together. The cloaks stream to the lee (west, a little toward the camera), the hems
# lift and flutter, the watch-fires lean downwind (screen left in every framing of this take).
WIND_DIR = -E3 * 0.93 + S3 * 0.37
WIND_DIR = WIND_DIR / np.linalg.norm(WIND_DIR)
WIND0 = 0.17                                     # metres of billow at the hem in the mean wind
FIRE_LEAN = -0.26


def wind_at(t, s, k=0):
    """Wind vector (world, toward the lee; |v| = hem billow in m) at shot time t for a figure at arc s, and a
    flutter phase for figure k."""
    g = 0.74 + 0.26 * math.sin(2 * math.pi * (t / 3.3 - s / 17.0)) \
        + 0.13 * math.sin(2 * math.pi * (t / 1.15 - s / 6.0) + 1.0 + 3.0 * _hh(k, 31))
    return WIND_DIR * WIND0 * g, 2 * math.pi * 1.35 * t * (0.85 + 0.3 * _hh(k, 32)) + 6.28 * _hh(k, 33)


# ------------------------------------------------------------------ the scene at time t ---
def _shoulders(pel, w, h, lean):
    """The same shoulder points SP.traveller builds (left, right), so the poles can rest on them."""
    sd = np.cross(UP, w)
    fl = w * math.sin(lean) + UP * math.cos(lean)
    chest = pel + fl * 0.47 * h
    return chest + UP * 0.02 * h + sd * 0.19 * h, chest + UP * 0.02 * h - sd * 0.19 * h


_KEEP = {}


def keeper_spot(k):
    """Where fire k's keeper stands: on the camera side of the fire (so the fire backlights them and never
    lights a face), on the highest footing 0.7-1.1 m from it."""
    if k not in _KEEP:
        p = wfs()[k]
        cam = camera(PASS_FRAMES[k], cfg=variant_cfg('main')).pos
        d = cam - p
        d[1] = 0.0
        d /= np.linalg.norm(d)
        if k == 0:
            # the near fire sits just beyond the crest: its keeper kneels on its west side, in profile, facing it
            # (below the poles as the bearers pass, and never front-lit)
            _, w = at(lantern_s(PASS_FRAMES[0] / FPS))
            d = -w.copy()
        side = np.cross(UP, d)
        best = None
        for r in (0.7, 0.85, 1.0, 1.1):
            for o in (-0.35, 0.0, 0.35):
                q = p + d * r + side * o
                q[1] = ground_many(q)[0]
                if best is None or q[1] > best[1] + 0.05:
                    best = q
        _KEEP[k] = (best, -d)
    return _KEEP[k]


def build_scene(t, cfg):
    sc = SP.Scene()
    S_ = line_s(t, cfg)
    n = len(S_)
    hts = [1.0, 1.03] + [0.93 + 0.14 * _hh(k, 14) for k in range(cfg['n'])]
    # the bearers walk loaded: knees a little bent, shorter steps (stride_of)
    planted = plant([(S_[i], i, hts[i], 0.04 if i < 2 else 0.0) for i in range(n)])
    lights = []
    lant = []
    waists = []
    hc = heart_col(t)
    br = breath(t)
    # ---- the great lantern on its poles, carried by two cloaked bearers
    lean_b = 0.14
    (pf, aLf, aRf, wf_), (pr, aLr, aRr, wr_) = planted[0], planted[1]
    fL, fR = _shoulders(pf, wf_, 1.0, lean_b)
    rL, rR = _shoulders(pr, wr_, 1.03, lean_b)
    lat = np.cross(UP, wf_)
    poles = ((fL + UP * 0.075 + wf_ * 0.35, rL + UP * 0.075 - wr_ * 0.35),
             (fR + UP * 0.075 + wf_ * 0.35, rR + UP * 0.075 - wr_ * 0.35))
    carry_f = (fL + UP * 0.05 + wf_ * 0.13, fR + UP * 0.05 + wf_ * 0.13)
    carry_r = (rL + UP * 0.05 + wr_ * 0.13, rR + UP * 0.05 + wr_ * 0.13)
    ph_f = walk_phase(S_[0], 0)
    ph_r = walk_phase(S_[1], 1)
    wf_v, fl_f = wind_at(t, S_[0], 0)
    wr_v, fl_r = wind_at(t, S_[1], 1)
    SP.traveller(sc, pf, wf_, aLf, aRf, (0.022, 0.018, 0.016), h=1.0, lean=lean_b, hem=0.17, cloak=(0.20, 0.30),
                 folds=11, fold_depth=0.030, fold_phase=0.8 * math.sin(ph_f) + 1.1, sway=0.025 * math.sin(ph_f),
                 carry=carry_f, carry_side=1.0, free_swing=0.06 * math.sin(ph_f), peak=True, wind=wf_v, flutter=fl_f)
    SP.traveller(sc, pr, wr_, aLr, aRr, (0.030, 0.030, 0.036), h=1.03, lean=lean_b, hem=0.17, cloak=(0.20, 0.30),
                 folds=12, fold_depth=0.030, fold_phase=0.8 * math.sin(ph_r) + 2.3, sway=0.025 * math.sin(ph_r),
                 carry=carry_r, carry_side=-1.0, free_swing=0.06 * math.sin(ph_r), peak=False, wind=wr_v, flutter=fl_r)
    sc.begin(rgb=(0.1, 0.07, 0.05))
    for a_, b_ in poles:
        sc.cone(a_, b_, 0.028, 0.028, 2, 0.0)
    mid = 0.25 * (poles[0][0] + poles[0][1] + poles[1][0] + poles[1][1])
    sc.cone(mid + lat * 0.30, mid - lat * 0.30, 0.022, 0.022, 2, 0.0)                     # crossbar
    heart = mid - UP * (0.10 + RING) + _swing(t) * wf_
    sc.cone(mid, heart + UP * RING, 0.008, 0.008, 1, 0.0)                                # the hook
    sc.end()
    SP.great_lantern(sc, heart, math.radians(YAW_E) + 0.3, hc, 0.9 * br, scale=LANT_K)
    lights.append([heart[0], heart[1], heart[2], hc[0], hc[1], hc[2], 2.6 * br, 0.35])
    waists += [0.5 * (pf + pr), pr]          # the rope is tied in at the rear bearer's waist
    # ---- the roped walkers: hooded wool cloaks and coats, packs, staffs; no two alike
    pal = [(0.07, 0.035, 0.028), (0.04, 0.045, 0.06), (0.06, 0.05, 0.03), (0.035, 0.045, 0.035), (0.05, 0.05, 0.05),
           (0.075, 0.04, 0.03), (0.055, 0.042, 0.038)]
    for k in range(cfg['n']):
        i = k + 2
        pel, aL, aR, w = planted[i]
        ph = walk_phase(S_[i], i)
        lside = 1.0 if _hh(k, 16) < 0.45 else -1.0
        # most wear knee-length hooded wool coats (the legs and the walk show); one in five a long cloak
        long_ = (0.20 + 0.06 * _hh(k, 17), 0.26 + 0.04 * _hh(k, 19)) if _hh(k, 24) < 0.2 else \
            (0.42 + 0.12 * _hh(k, 17), 0.215 + 0.04 * _hh(k, 19))
        wk_v, flk = wind_at(t, S_[i], i)
        tip = None
        if _hh(k, 12) < 0.35:
            q, wq = at(S_[i] + 0.5 + 0.12 * math.sin(ph))
            q = q - np.cross(UP, wq) * (0.22 * lside)
            q[1] = ground_many(q)[0]
            tip = q
        o = SP.traveller(sc, pel, w, aL, aR, pal[int(_hh(k, 11) * len(pal))], h=hts[i], lean=0.07 + 0.07 * _hh(k, 15),
                         hem=long_[0], cloak=(0.19 + 0.025 * _hh(k, 18), long_[1]),
                         folds=int(8 + 5 * _hh(k, 20)), fold_depth=0.020 + 0.014 * _hh(k, 21),
                         fold_phase=0.8 * math.sin(ph) + 6.28 * _hh(k, 22), sway=0.03 * math.sin(ph),
                         pack=_hh(k, 13) < 0.7, staff_tip=tip, lantern_side=lside,
                         lantern_swing=0.03 * math.sin(2 * ph - 0.6), free_swing=0.08 * math.sin(ph),
                         peak=_hh(k, 23) < 0.6, wind=wk_v * (0.8 + 0.4 * _hh(k, 34)), flutter=flk)
        waists.append(o['waist'])
        I = small_lantern_I(k, t)
        bc = SP.small_lantern(sc, o['lantern'], SMALL_COL, 2.2 * I / 0.3)
        lights.append([bc[0], bc[1], bc[2], SMALL_COL[0], SMALL_COL[1], SMALL_COL[2], I, 0.12])
        lant.append((bc, I))
    # ---- the watch-fires: a ring of stones at the near one; a hooded keeper at each, backlit, feeding it
    for k, p in enumerate(wfs()):
        b, ph = wf_burn(t, k)
        if k == 0:
            sc.begin(rgb=(0.07, 0.07, 0.075))
            rng = np.random.default_rng(70 + k)
            for m in range(13):
                a_ = 2 * math.pi * m / 13 + rng.uniform(-0.15, 0.15)
                q = p + np.array([math.cos(a_), 0.0, math.sin(a_)]) * (0.50 + rng.uniform(-0.04, 0.05))
                hs = rng.uniform(0.6, 1.0)
                sc.box(q + UP * 0.05 * hs, (0.09 * hs, 0.07 * hs, 0.08 * hs), yaw=a_ + rng.uniform(-0.5, 0.5),
                       pitch=rng.uniform(-0.3, 0.3), rnd=0.025, mat=5, k=0.0)
            sc.end()
        feed = smoothstep(0.86, 0.93, ph) * (1.0 - smoothstep(0.97, 1.0, ph))
        kp, wk = keeper_spot(k)
        side = np.cross(UP, wk)
        aL = kp + side * 0.12 + UP * 0.07
        aR = kp - side * 0.12 + wk * 0.08 + UP * 0.07
        kneel = 0.34 if k == 0 else 0.0
        pel = kp + UP * (0.93 * 0.98 - 0.30 * feed - kneel)
        kw_v, kfl = wind_at(t, 0.0, 60 + k)
        SP.traveller(sc, pel, wk, aL, aR, (0.03, 0.025, 0.02), h=0.98, lean=0.10 + 0.55 * feed + 0.3 * (k == 0), hem=0.20,
                     cloak=(0.20, 0.29), folds=10, fold_depth=0.028, fold_phase=1.3 * k, peak=True,
                     reach=(p + UP * 0.25) if feed > 0.2 else None, wind=kw_v * (0.7 if k == 0 else 1.0), flutter=kfl)
        fl = F.flicker(t, 30 + k)
        lights.append([p[0], p[1] + 0.9, p[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2], 9.0 * b * fl, 0.8])
    return sc, lights, lant, waists, heart


# ------------------------------------------------------------------ 2-D passes in source space ---
def draw_rope(img, zb, scam, waists, lights, md, mf):
    W = np.array(waists)
    L = np.array(lights)
    for a, b in zip(W[:-1], W[1:]):
        d = np.linalg.norm(b - a)
        sag = 0.28 + 0.12 * math.sin(d * 3.1)
        u = np.linspace(0, 1, 14)
        pts = a[None] + (b - a)[None] * u[:, None] - UP[None] * (4 * sag * u * (1 - u))[:, None]
        sx, sy, z = scam.project(pts)
        if (z < 0.3).any():
            continue
        # light on the rope: lanterns nearby (inverse square) + the moon
        cols = np.zeros((len(pts), 3))
        for li in L:
            l2 = np.sum((pts - li[None, :3]) ** 2, axis=1)
            cols += li[3:6][None] * (li[6] / (l2 + li[7] ** 2))[:, None] * 0.6
        cols += np.array([0.42, 0.52, 0.72])[None] * 0.25 * mf
        cols *= np.array([0.16, 0.12, 0.08])[None] * 0.35
        for m in range(len(pts) - 1):
            zz = 0.5 * (z[m] + z[m + 1])
            wpx = max(0.012 * scam.f / zz, 0.5)
            a_ = min(0.012 * scam.f / zz / 0.5, 1.0) ** 1.5
            c = 0.5 * (cols[m] + cols[m + 1]) * a_
            # rope: dark line (occluding) plus its lit colour
            _rope_seg(img, zb, sx[m], sy[m], sx[m + 1], sy[m + 1], zz, wpx, c, a_ * 0.85)


@njit(fastmath=True, cache=True)
def _rope_seg_k(img, zb, x0, y0, x1, y1, z, w, r, g, b, alpha):
    H, W = img.shape[0], img.shape[1]
    dx = x1 - x0
    dy = y1 - y0
    L = math.sqrt(dx * dx + dy * dy) + 1e-9
    xmin = int(min(x0, x1) - w - 2)
    xmax = int(max(x0, x1) + w + 2)
    ymin = int(min(y0, y1) - w - 2)
    ymax = int(max(y0, y1) + w + 2)
    for yy in range(max(ymin, 0), min(ymax, H)):
        for xx in range(max(xmin, 0), min(xmax, W)):
            if zb[yy, xx] < z - 0.05:
                continue
            px = xx + 0.5 - x0
            py = yy + 0.5 - y0
            u = min(max((px * dx + py * dy) / (L * L), 0.0), 1.0)
            ex = px - dx * u
            ey = py - dy * u
            dd = math.sqrt(ex * ex + ey * ey)
            cov = min(max(0.5 * w + 0.5 - dd, 0.0), 1.0) * alpha
            if cov <= 0.0:
                continue
            img[yy, xx, 0] = img[yy, xx, 0] * (1 - cov) + r * cov / max(alpha, 1e-3)
            img[yy, xx, 1] = img[yy, xx, 1] * (1 - cov) + g * cov / max(alpha, 1e-3)
            img[yy, xx, 2] = img[yy, xx, 2] * (1 - cov) + b * cov / max(alpha, 1e-3)


def _rope_seg(img, zb, x0, y0, x1, y1, z, w, c, alpha):
    _rope_seg_k(img, zb, float(x0), float(y0), float(x1), float(y1), float(z), float(w), float(c[0]), float(c[1]),
                float(c[2]), float(alpha))


def draw_trail(img, zb, scam, t, cfg, gain):
    """The faint trail of light the line leaves on the path behind it (opens with the long exposure)."""
    if gain <= 0.0:
        return
    # only BEHIND the last walker, on the trodden snow: between the walkers a lit line reads as a wire
    S_ = line_s(t, cfg)
    s_tail = S_[-1]
    ss = np.arange(s_tail - 150.0, s_tail - 0.8, 0.4)
    dlt = s_tail - ss
    I = np.exp(-dlt / 45.0) * smoothstep(1.0, 5.0, dlt)
    P, _ = at(ss)
    P = P + UP * 0.05
    sx, sy, z = scam.project(P)
    col = np.array([1.0, 0.62, 0.30])
    for m in range(len(ss) - 1):
        if z[m] < 0.5 or I[m] <= 1e-4:
            continue
        e = gain * 0.5 * (I[m] + I[m + 1])
        w = max(0.22 * scam.f / z[m], 0.6)
        _aa_line(img, zb, float(sx[m]), float(sy[m]), float(sx[m + 1]), float(sy[m + 1]), float(z[m]), float(w),
                 float(col[0] * e), float(col[1] * e), float(col[2] * e), 0.4)


def draw_heart(img, zb, scam, heart, t, hc, br, pxs):
    """The heart of light in the lantern: an intense core, a soft glow filling the glass, and a few slow filaments."""
    sx, sy, z = scam.project(heart)
    if z < 0.2:
        return
    ppm = scam.f / z
    I = 1.0 * br
    F.halo(img, zb, sx, sy, max(0.035 * ppm, 0.6), 30.0 * I, z=z, zbias=0.4, col=hc)
    F.halo(img, zb, sx, sy, max(0.09 * ppm, 0.9), 0.9 * I, z=z, zbias=0.4, col=hc)
    F.halo(img, zb, sx, sy, max(0.40 * ppm, 1.5), 0.05 * I, z=z, zbias=0.6, col=hc)
    if ppm < 40:
        return
    # filaments: slow curling threads of light around the core (the thinking fire, calm)
    rng = np.random.default_rng(5)
    for k in range(7):
        a0 = rng.uniform(0, 6.28)
        tilt = rng.uniform(-0.9, 0.9)
        rr = rng.uniform(0.05, 0.11)
        n = 28
        pts = []
        for m in range(n):
            u = m / (n - 1)
            a = a0 + 2.2 * u + 0.15 * t * (0.6 + 0.4 * k / 7)
            r = rr * (0.4 + 0.6 * math.sin(math.pi * u))
            p = heart + np.array([math.cos(a) * r, (u - 0.5) * 0.16 + tilt * 0.03 * math.sin(a), math.sin(a) * r])
            pts.append(p)
        pts = np.array(pts)
        a_, b_, zz = scam.project(pts)
        for m in range(n - 1):
            e = 0.9 * I * math.sin(math.pi * m / (n - 1)) ** 2
            _aa_line(img, zb, float(a_[m]), float(b_[m]), float(a_[m + 1]), float(b_[m + 1]), float(zz[m]),
                     max(0.004 * ppm, 0.5), float(hc[0] * e), float(hc[1] * e), float(hc[2] * e), 0.3)


def draw_small_glows(img, zb, scam, lant):
    for bc, I in lant:
        sx, sy, z = scam.project(bc)
        if z < 0.3:
            continue
        ppm = scam.f / z
        F.halo(img, zb, sx, sy, max(0.05 * ppm, 0.55), 22.0 * I, z=z, zbias=0.3, col=SMALL_COL)
        F.halo(img, zb, sx, sy, max(0.35 * ppm, 1.2), 0.05 * I, z=z, zbias=0.5, col=SMALL_COL)


def draw_fires(img, zb, scam, t):
    for k, p in enumerate(wfs()):
        b, ph = wf_burn(t, k)
        fl = F.flicker(t, 30 + k)
        F2.flame(img, zb, scam, p + UP * 0.12, 1.5 * b, 0.42, t, seed=40 + k, I=12.0 * (0.5 + 0.5 * b),
                 lean=FIRE_LEAN, zbias=0.5)
        sx, sy, z = scam.project(p + UP * 0.6)
        if z > 0.5:
            ppm = scam.f / z
            F2.halo(img, zb, sx, sy, 2.2 * ppm, 0.02 * b * fl, z=z, zbias=2.0)
    # far watch-fires on the islands (and three on the great range's crest): warm points and a soft aura, burning
    # low and fed on the sky clock
    FF = far_fires()
    RF = range_fires()
    if len(RF):
        FF = np.vstack([FF, RF]) if len(FF) else RF
    if len(FF):
        a = sky_angle(t)
        for k, (x, y, z_, pr) in enumerate(FF):
            p = np.array([x, y + 3.0, z_])
            sx, sy, z = scam.project(p)
            if z < 50.0 or sx < -20 or sy < -20 or sx > img.shape[1] + 20 or sy > img.shape[0] + 20:
                continue
            per = 9.0 + 6.0 * _hh(k, 21)
            since = ((a + per * _hh(k, 22)) % per)
            b = 0.35 + 0.65 * math.exp(-since / 3.5)
            fl = F.flicker(t, 200 + k, 0.6)
            dist = np.linalg.norm(p - scam.pos)
            tr = math.exp(-dist * 6.0e-5)
            F2.glow(img, zb, sx, sy, 0.8, 1.6 * b * fl * tr * (2500.0 / max(dist, 800.0)) ** 1.2, z=z, zbias=40.0)
            F2.halo(img, zb, sx, sy, 3.5, 0.010 * b * tr, z=z, zbias=40.0, col=F2.AURA_COL)


# ------------------------------------------------------------------ frame ---
FINISH = dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.9, streak_strength=0.0, vignette_amount=0.25)


def render(frame, scale=1.0, ss=1.5, variant='main', trail=True):
    cfg = variant_cfg(variant)
    t = frame / FPS
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = camera(frame, W, H, cfg)
    fr = PI.Frame(tcam, ss)
    light, mf, md = night(t)
    sc, lights, lant, waists, heart = build_scene(t, cfg)
    PL = [[p[0], p[1], p[2], 1.2] for p in wfs()]
    scam = fr.src
    C = scam.params()
    Pp = np.array([scam.pos[0], scam.pos[2], t, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(Pp, CR, C, 0.2, 90000.0, 0.0035, 0.35, 700.0, 9, D)
    Lk_, amb_, S_, fogp_, Q_ = light
    fr.img = np.zeros((scam.H, scam.W, 3), np.float32)
    fr.zb = np.zeros((scam.H, scam.W), np.float32)
    fr.dist = np.zeros((scam.H, scam.W), np.float32)
    WD.shade(C, D, Pp, CR, S_, np.array(lights, np.float64).reshape(-1, 8), Lk_, Q_, amb_, fogp_, fr.img, fr.zb,
             fr.dist, np.array(PL, np.float64).reshape(-1, 4))
    pxs = PI.src_scale(fr)
    # the red under the cloud (post-pass on the cloud deck, same depth)
    UG = underglow(t)
    if len(UG):
        WD.cloud_glow(C, D, Pp, CR, UG, fogp_, fr.img)
    draw_sky(fr, t, pxs)
    # 3-D people, the lanterns, the poles, the stone rings
    Pr, Ob = sc.arrays()
    Lk = light[0]
    moon = np.array([md[0], md[1], md[2], Lk[3], Lk[4], Lk[5], 0.55 * mf])
    SP.render(fr.img, fr.zb, C, Pr, Ob, np.array(lights, np.float64), moon, light[1] * 1.3, light[3], scam.pos[1])
    a = sky_angle(t)
    if trail:
        draw_trail(fr.img, fr.zb, scam, t, cfg, 0.10 * smoothstep(4.0, 16.0, a))
    draw_rope(fr.img, fr.zb, scam, waists, lights, md, mf)
    draw_fires(fr.img, fr.zb, scam, t)
    draw_small_glows(fr.img, fr.zb, scam, lant)
    draw_heart(fr.img, fr.zb, scam, heart, t, heart_col(t), breath(t), pxs)
    img, zb, di = PI.to_target(fr)
    return img


CUT0 = 4880                                      # A18 starts on A's cut frame 4880 (bar 62): files use cut frames


def work(args):
    frames, scale, out, threads, ss, variant, off = args
    # never rewrite NUMBA_NUM_THREADS here: the farm exports it (the node's CPU budget) and numba refuses a change
    # once its pool is up; only narrow the pool
    import numba
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    import cv2
    cv2.setNumThreads(1)
    look = PI.look
    for f in frames:
        t0 = time.time()
        img = look.finish(render(f - off, scale=scale, ss=ss, variant=variant), **FINISH)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--variant', default='main')
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--numbering', choices=('cut', 'shot'), default='cut',
                    help="cut (default, EDIT-v3's delivery convention): frames are A's cut frames 4880-5839; "
                         "shot: 0-959 (tests)")
    a = ap.parse_args()
    off = CUT0 if a.numbering == 'cut' else 0
    base = os.path.join(PI.CM.ROOT, 'renders', 'crossing_A')
    out = base if a.out is None else os.path.join(base, a.out)
    os.makedirs(out, exist_ok=True)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        s, e = (a.range or f'{off}-{off + NFR - 1}').split('-')
        frames = list(range(int(s), int(e) + 1, a.step))
    bad = [f for f in frames if not off <= f < off + NFR]
    if bad:
        raise SystemExit(f'frames outside the shot ({off}-{off + NFR - 1} in {a.numbering} numbering): {bad[:5]}')
    if a.skip:
        frames = [f for f in frames if not os.path.exists(PI.look.frame_path(out, f))]
    far_fires()              # build the island fire table once (cached to crossing_fires.npy) before forking
    path()
    if a.procs <= 1:
        work((frames, a.scale, out, a.threads, a.ss, a.variant, off))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(work, [(c, a.scale, out, 1, a.ss, a.variant, off) for c in chunks])


if __name__ == '__main__':
    main()
