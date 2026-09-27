"""THE CROSSING (cut A, R6; A's showpiece #2). 960 frames, shot-local numbering 0-959 (renders/crossing_A/).

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

  python crossing.py --frames 0,240,480,720,959 --scale 0.5 --out t1       # stills -> renders/crossing_A/t1/
  python crossing.py --range 0-959 --procs 4 --skip                         # full  -> renders/crossing_A/
  --variant few   the fallback test: 12 walkers, larger (camera closer)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from numba import njit, prange

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
LAT = 8.0                                    # latitude: the pole sits low in the north, inside the wide
_nh = -S3
POLE = np.array([_nh[0] * math.cos(math.radians(LAT)), math.sin(math.radians(LAT)),
                 _nh[2] * math.cos(math.radians(LAT))])


def uv(u, v, y=0.0):
    p = C0 + E3[[0, 2]] * u + S3[[0, 2]] * v
    return np.array([p[0], y, p[1]])


# crest knots (u along the arete, v lateral wander, crest height)
KNOTS = [(-330.0, 0.0, 188.0), (-235.0, 7.0, 158.0), (-150.0, -3.0, 142.0), (-70.0, 4.0, 131.0),
         (-10.0, -2.0, 126.0), (60.0, 3.0, 133.0), (150.0, -4.0, 158.0), (255.0, 0.0, 206.0)]
WF_U = (-128.0, 52.0)                        # the two watch-fires of this stretch (behind, ahead)


def build_set():
    rows = []
    for k in range(len(KNOTS) - 1):
        a, b = KNOTS[k], KNOTS[k + 1]
        pa, pb = uv(a[0], a[1], a[2]), uv(b[0], b[1], b[2])
        rows.append(WD.ridge_row(pa, pb, wl=15.0, wr=13.0, seed=41 + 3 * k, k=6.0, detail=0.24, slope=1.9))
    for (u, top, seed, ang) in ((-338.0, 196.0, 7, 0.4), (262.0, 214.0, 11, -0.3)):
        p = uv(u, 0.0)
        rows.append(WD.crag_row(p[0], p[2], top, L=16.0, s_hi=2.4, s_lo=1.35, aniso=1.4, ang=math.radians(YAW_E) + ang,
                                seed=seed, k=7.0, detail=0.3, shelf=2.0, nf=4))
    # the watch-fire steps: a small rock shoulder beside the crest (camera side) for each fire
    for u in WF_U:
        cr = _knot_y(u)
        p = uv(u, 5.5)
        rows.append(WD.crag_row(p[0], p[2], cr - 1.2, L=3.5, s_hi=2.0, s_lo=1.6, aniso=1.2, ang=math.radians(YAW_E),
                                seed=int(abs(u)) + 3, k=2.5, detail=0.25, shelf=1.4, nf=4))
    return np.array(rows)


def _knot_y(u):
    us = [k[0] for k in KNOTS]
    ys = [k[2] for k in KNOTS]
    return float(np.interp(u, us, ys))


CR = build_set()


def ground_many(P, fp=0.02):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    out = np.zeros(len(P))
    WD.heights(P[:, 0].copy(), P[:, 2].copy(), fp, CR, out)
    return out


# ------------------------------------------------------------------ the path on the crest ---
_PATH = None


def _build_path():
    us = np.arange(-300.0, 230.0, 0.5)
    vs = np.linspace(-14.0, 14.0, 113)
    UU, VV = np.meshgrid(us, vs, indexing='ij')
    pts = C0[None, None, :] + E3[[0, 2]] * UU[..., None] + S3[[0, 2]] * VV[..., None]
    h = np.zeros(UU.size)
    WD.heights(pts[..., 0].ravel().copy(), pts[..., 1].ravel().copy(), 0.02, CR, h)
    h = h.reshape(UU.shape)
    vbest = vs[np.argmax(h, axis=1)]
    from scipy.ndimage import gaussian_filter1d
    v = gaussian_filter1d(vbest, 6.0, mode='nearest')          # sigma 3 m
    P = np.stack([uv(u, vv) for u, vv in zip(us, v)])
    y = ground_many(P)
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
        return dict(n=12, gap=4.2, jit=0.5, r1=52.0, dT=12.0, hfov1=46.0, h1=1.6, r0=5.5)
    return dict(n=40, gap=2.6, jit=0.35, r1=85.0, dT=22.0, hfov1=48.0, h1=2.5, r0=5.5)


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


def walk_phase(sk, k):
    # phase follows the distance actually walked, so the feet never slide
    stride = 0.75 * (0.92 + 0.16 * _hh(k, 5))
    return 2 * math.pi * (sk / stride + _hh(k, 6))


# ------------------------------------------------------------------ the two clocks ---
SKY_DEG = 46.0                                # the sky turns ~3 hours over the shot


def _rate(t):
    return smoothstep(7.0, 17.0, t) * (1.0 - 0.35 * smoothstep(31.0, 40.0, t))


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
    e1 = _sm((t - 2.5) / 27.0)
    e2 = _sm((t - 2.0) / 26.0)
    r = cfg['r0'] * (cfg['r1'] / cfg['r0']) ** e1
    beta = math.radians(14.0 + 60.0 * e2)
    h = 1.25 + (cfg['h1'] - 1.25) * e1 ** 1.3
    pos = L + (-E3 * math.cos(beta) + S3 * math.sin(beta)) * r + UP * h
    T = L - E3 * cfg['dT'] * e1 - UP * 0.3 * e1
    d = T - pos
    yaw = math.degrees(math.atan2(d[0], d[2]))
    pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2]))) + 5.0 * e1
    hfov = 32.0 + (cfg['hfov1'] - 32.0) * e1
    return RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)


# ------------------------------------------------------------------ the red under the cloud ---
_UG = None


def _ug_table():
    global _UG
    if _UG is None:
        rng = np.random.default_rng(606)
        rows = []
        for k in range(26):
            # the visible cloud sea from the arete is the band under the horizon beyond ~15 km
            dist = 14000.0 * (3.4 ** rng.random())                    # 14 - 48 km out, north of the arete
            ang = math.radians(rng.uniform(-48.0, 48.0))
            dirv = -S3 * math.cos(ang) + E3 * math.sin(ang)
            c = uv(0.0, 0.0) + dirv * dist
            rad = rng.uniform(1400.0, 3600.0) * (0.6 + 0.4 * dist / 30000.0)
            out_at = rng.uniform(6.0, 44.0) if k >= 3 else rng.uniform(47.0, 60.0)
            rows.append([c[0], c[2], rad, out_at, rng.uniform(0.6, 1.0), rng.uniform(0, 6.28)])
        _UG = np.array(rows)
    return _UG


def underglow(t):
    """Patch table for WD.cloud_glow at shot time t: each patch dies on the sky clock, with a last flicker."""
    a = sky_angle(t)
    T = _ug_table()
    out = []
    col = np.array([1.0, 0.16, 0.035]) * 1.5
    for x, z, rad, a_out, g, ph in T:
        live = 1.0 - smoothstep(a_out - 3.5, a_out, a)
        if live <= 0.0:
            continue
        fl = 1.0 + 0.25 * math.sin(7.0 * t + ph) * smoothstep(a_out - 6.0, a_out - 2.0, a) * live
        I = g * live * fl
        out.append([x, z, rad, col[0] * I, col[1] * I, col[2] * I, 900.0, 0.35])
    return np.array(out, np.float64).reshape(-1, 8)


# ------------------------------------------------------------------ watch-fires ---
def wf_positions():
    out = []
    for u in WF_U:
        p = uv(u, 5.5)
        # the top of the little rock step
        best = None
        for dv in np.linspace(-2.5, 2.5, 11):
            for du in np.linspace(-2.5, 2.5, 11):
                q = uv(u + du, 5.5 + dv)
                q[1] = ground_many(q)[0]
                if best is None or q[1] > best[1]:
                    best = q
        out.append(best)
    return out


_WF = None


def wfs():
    global _WF
    if _WF is None:
        _WF = wf_positions()
    return _WF


def wf_burn(t, k):
    """Watch-fire k: burns low on the sky clock and is fed (a flare) every ~13 degrees of sky."""
    a = sky_angle(t) + 5.0 * k
    period = 13.0
    ph = (a % period) / period
    since = ph * period
    b = 0.32 + 0.68 * math.exp(-since / 5.5)
    return b, ph


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


def draw_sky(fr, t, pxs):
    """Stars on the sky clock: points at first, then arcs about the pole as the long exposure opens."""
    scam = fr.src
    st = stars()
    a = sky_angle(t)
    mask = (fr.dist > 1e8).astype(np.float32)
    expo = min(a, 10.0) * smoothstep(0.0, 6.0, a)         # degrees of arc in each trail
    M = 2 if expo < 0.05 else int(8 + 16 * min(expo / 10.0, 1.0))
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
    # the head (current position) stays a point; the arc behind it is the exposure, fading with age
    wage = np.exp(-(1.0 - np.linspace(0.0, 1.0, M)) * 1.6)[None, :]
    seg_e = lum[:, None] * ext * wage * 0.45 * (0.4 + 0.6 * min(expo / 8.0, 1.0))
    _trails(fr.img, mask, sx.astype(np.float64), sy.astype(np.float64), seg_e[:, :-1].astype(np.float64), col,
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
    amb = amb * (0.55 + 0.45 * mf)
    S = SK.sky_params(zenith='#070B1C', horizon='#2A3866', moon_dir=md, moon_radius_deg=0.8,
                      halo_I=0.025 * mf, halo_w=0.22, halo2_I=0.012 * (0.3 + 0.7 * mf), halo2_w=0.7,
                      horizon_glow=0.25 * (0.6 + 0.4 * mf), gain=0.75 + 0.25 * mf)
    return (Lk, amb, S, fogp, Q), mf, md


# ------------------------------------------------------------------ the scene at time t ---
def build_scene(t, cfg):
    sc = SP.Scene()
    S_ = line_s(t, cfg)
    P, Wd = at(S_)
    lights = []
    lant = []
    waists = []
    hc = heart_col(t)
    br = breath(t)
    # the two bearers and the great lantern
    pf, pr = P[0], P[1]
    w0 = Wd[0]
    lat = np.cross(UP, w0)
    ph_f = walk_phase(S_[0], 0)
    ph_r = walk_phase(S_[1], 1)
    bob_f = 0.018 * math.cos(2 * ph_f)
    bob_r = 0.018 * math.cos(2 * ph_r)
    shf = pf + UP * (1.47 + bob_f)
    shr = pr + UP * (1.47 + bob_r)
    tips = []
    for side in (1.0, -1.0):
        a_ = shf + lat * side * 0.26 + w0 * 0.35
        b_ = shr + lat * side * 0.26 - w0 * 0.35
        tips.append((a_, b_))
    fr_carry = (shf + lat * 0.24 + w0 * 0.12 - UP * 0.03, shf - lat * 0.24 + w0 * 0.12 - UP * 0.03)
    rr_carry = (shr + lat * 0.24 + w0 * 0.12 - UP * 0.03, shr - lat * 0.24 + w0 * 0.12 - UP * 0.03)
    out_f = SP.walker(sc, pf, w0, ph_f, (0.06, 0.045, 0.035), carry=fr_carry, pack=False, height=1.0, lean=0.1,
                     carry_side=1.0)
    out_r = SP.walker(sc, pr, Wd[1], ph_r, (0.04, 0.042, 0.05), carry=rr_carry, pack=False, height=1.03, lean=0.1,
                     carry_side=-1.0)
    sc.begin(rgb=(0.1, 0.07, 0.05))
    for a_, b_ in tips:
        sc.cone(a_, b_, 0.028, 0.028, 2, 0.0)
    mid = 0.5 * (shf + shr)
    sc.cone(mid + lat * 0.3 + UP * 0.03, mid - lat * 0.3 + UP * 0.03, 0.022, 0.022, 2, 0.0)   # crossbar
    heart = heart_pos(t)
    sc.cone(mid + UP * 0.03, heart + UP * RING, 0.008, 0.008, 1, 0.0)  # the hook
    sc.end()
    SP.great_lantern(sc, heart, math.radians(YAW_E) + 0.3, hc, 0.9 * br, scale=LANT_K)
    lights.append([heart[0], heart[1], heart[2], hc[0], hc[1], hc[2], 2.6 * br, 0.35])
    waists += [out_f['waist'], out_r['waist']]
    # the roped walkers
    pal = [(0.07, 0.035, 0.028), (0.04, 0.045, 0.06), (0.06, 0.05, 0.03), (0.035, 0.045, 0.035), (0.05, 0.05, 0.05),
           (0.075, 0.04, 0.03)]
    for k in range(cfg['n']):
        i = k + 2
        ph = walk_phase(S_[i], i)
        o = SP.walker(sc, P[i], Wd[i], ph, pal[int(_hh(k, 11) * len(pal))], lantern='hand',
                      staff=_hh(k, 12) < 0.35, pack=_hh(k, 13) < 0.8, height=0.93 + 0.14 * _hh(k, 14),
                      lean=0.06 + 0.08 * _hh(k, 15), lantern_side=1.0 if _hh(k, 16) < 0.45 else -1.0)
        waists.append(o['waist'])
        top = o['lantern']
        I = small_lantern_I(k, t)
        bc = SP.small_lantern(sc, top, SMALL_COL, 2.2 * I / 0.3)
        lights.append([bc[0], bc[1], bc[2], SMALL_COL[0], SMALL_COL[1], SMALL_COL[2], I, 0.12])
        lant.append((bc, I))
    # the watch-fires: stone rings + a keeper
    for k, p in enumerate(wfs()):
        b, ph = wf_burn(t, k)
        sc.begin(rgb=(0.07, 0.07, 0.075))
        rng = np.random.default_rng(70 + k)
        for m in range(9):
            a_ = 2 * math.pi * m / 9 + rng.uniform(-0.2, 0.2)
            q = p + np.array([math.cos(a_), 0.0, math.sin(a_)]) * 0.55
            sc.box(q + UP * 0.08, (0.16, 0.11, 0.12), yaw=a_, rnd=0.05, mat=5, k=0.0)
        sc.end()
        feed = smoothstep(0.86, 0.93, ph) * (1.0 - smoothstep(0.97, 1.0, ph))
        kp = p + S3 * 0.9 - E3 * 0.8
        wk = (p - kp)
        wk[1] = 0
        wk /= np.linalg.norm(wk)
        o = SP.walker(sc, kp + UP * 0.0, wk, 0.4, (0.08, 0.06, 0.05), pack=False, height=0.98,
                      lean=0.1 + 0.9 * feed)
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
    S_ = line_s(t, cfg)
    s_head = S_[1]
    ss = np.arange(s_head - 260.0, s_head - 0.5, 0.5)
    I = np.zeros(len(ss))
    for sk in S_[1:]:
        dlt = sk - ss
        I += np.where(dlt > 0, smoothstep(0.6, 3.2, dlt) * np.exp(-dlt / 55.0), 0.0)
    I = I / 8.0
    P, _ = at(ss)
    P = P + UP * 0.72
    sx, sy, z = scam.project(P)
    col = np.array([1.0, 0.62, 0.30])
    for m in range(len(ss) - 1):
        if z[m] < 0.5 or I[m] <= 1e-4:
            continue
        e = gain * 0.5 * (I[m] + I[m + 1])
        w = max(0.05 * scam.f / z[m], 0.6)
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
        F2.flame(img, zb, scam, p + UP * 0.12, 1.5 * b, 0.42, t, seed=40 + k, I=12.0 * (0.5 + 0.5 * b), lean=0.12,
                 zbias=0.5)
        sx, sy, z = scam.project(p + UP * 0.6)
        if z > 0.5:
            ppm = scam.f / z
            F2.halo(img, zb, sx, sy, 2.2 * ppm, 0.02 * b * fl, z=z, zbias=2.0)
    # far watch-fires on the islands: warm points and a soft aura, burning low and fed on the sky clock
    FF = far_fires()
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
        draw_trail(fr.img, fr.zb, scam, t, cfg, 0.06 * smoothstep(4.0, 16.0, a))
    draw_rope(fr.img, fr.zb, scam, waists, lights, md, mf)
    draw_fires(fr.img, fr.zb, scam, t)
    draw_small_glows(fr.img, fr.zb, scam, lant)
    draw_heart(fr.img, fr.zb, scam, heart, t, heart_col(t), breath(t), pxs)
    img, zb, di = PI.to_target(fr)
    return img


def work(args):
    frames, scale, out, threads, ss, variant = args
    os.environ['NUMBA_NUM_THREADS'] = str(threads)
    import numba
    numba.set_num_threads(threads)
    import cv2
    cv2.setNumThreads(1)
    look = PI.look
    for f in frames:
        t0 = time.time()
        img = look.finish(render(f, scale=scale, ss=ss, variant=variant), **FINISH)
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
    a = ap.parse_args()
    base = os.path.join(PI.CM.ROOT, 'renders', 'crossing_A')
    out = base if a.out is None else os.path.join(base, a.out)
    os.makedirs(out, exist_ok=True)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        s, e = (a.range or f'0-{NFR - 1}').split('-')
        frames = list(range(int(s), int(e) + 1, a.step))
    if a.skip:
        frames = [f for f in frames if not os.path.exists(PI.look.frame_path(out, f))]
    far_fires()              # build the island fire table once (cached to crossing_fires.npy) before forking
    path()
    if a.procs <= 1:
        work((frames, a.scale, out, a.threads, a.ss, a.variant))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(work, [(c, a.scale, out, 1, a.ss, a.variant) for c in chunks])


if __name__ == '__main__':
    main()
