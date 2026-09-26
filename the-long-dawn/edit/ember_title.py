"""THE LONG DAWN -- the title card, formed from embers (v2 frames 2800-2966, all three cuts).

The bookend of the KINDLING (EMBERS 300-340, torch embers slowing into letters). As the answering
fires ignite across the valley, sparks drift up from them (and, once the crane has lifted the
valley away, from behind our hill and the frame edge), ride a rising curl-noise flow and slow into
a loose hover under the sky; then -- staggered gently left to right -- each one glides the last
few dozen pixels into its place in THE LONG DAWN: Cinzel 92, tracking 0.28, centred at y=402,
i.e. exactly the mask titles._title() draws (titles.render_line is reused, read-only). By ~2880
the title is crisp warm-gold light built of fine embers, with a living flicker and a soft halo.
It holds; from ~2924 the letters loosen back into embers that rise, cool and die by ~2960 as the
picture goes to black.

How it is made
* Every letter ember has an exact target on the glyph, a landing frame and a hover point
  scattered around (mostly below) that target. Its rising path is integrated BACKWARD in time
  from the hover point through the flow (buoyant rise that slows into the hover, curl noise, the
  plate's wind, a partial coupling to the crane's screen motion), so it arrives on schedule
  having ridden coherent air currents. The glide is a curved blend from that drifting path onto
  the target with zero velocity at both ends: quick to start, long and soft to settle.
  Embers that come into view in open sky are launched from a real answering fire (world
  positions from the hills department through a replica of the CODA crane camera; at most one
  spark per fire every few frames) or fade in from the valley haze; the rest rise from behind our
  hilltop and the frame edge.
* The glyph is the exact rasterised mask. Every pixel has a reveal frame (letter stagger, lower
  parts first, soft patches) and a release frame (left to right, tops first); the embers use the
  same fields, so the mask fills in exactly where and when they land and gives its light back to
  them when they leave. Inside: a fine flickering stipple of embers, thick stems a little hotter
  than hairlines, slow heat drifting through the strokes, a soft halo.
* Loosening: the landed embers plus ~2200 more that live invisibly in the glyph (their light is
  the glyph's) leave as the release front passes, each from rest with its own small drift on top
  of the shared updraft, so the letters crumble into fine embers that lift, cool from gold to
  ember red and die.
* Moving embers are tapered hot-head streaks (shutter motion blur); defocused ones are soft bokeh
  discs. Far sparks are hidden behind our hilltop and behind the cairn and the two figures.

Output: renders/title/t_%05d.exr -- linear-light RGB (half float), 1920x804, additive, v2
numbering 2800-2966. The director composites it after picture() and before grain:

    img_srgb = look.linear_to_srgb(soft_clip(look.srgb_to_linear(img_srgb) + layer(f)))

soft_clip: identity up to 0.8, exponential shoulder to 1.0 above (see soft_clip() / composite()).

    python edit/ember_title.py                        # render 2800-2966 with 2 workers
    python edit/ember_title.py --range 2840 2890      # part of it
    python edit/ember_title.py --review               # composite sheet -> _local_logs/review/title.jpg
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')     # before cv2 touches any EXR
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, 'lib'), os.path.join(ROOT, 'edit')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import look  # noqa: E402
import titles  # noqa: E402
import ember_title_fires as FD  # noqa: E402

W, H = look.W, look.H
F0, F1 = 2800, 2966                      # v2 frames covered (inclusive)
OUT_DIR = os.path.join(ROOT, 'renders', 'title')
REVIEW = os.path.join(os.path.dirname(ROOT), '_local_logs', 'review', 'title.jpg')
TEXT = 'THE LONG DAWN'
TITLE_Y = 402

# --- the choreography (v2 frames) -------------------------------------------------------------
TA0, TA1 = 2851.0, 2869.0                # mean landing frame of the first / last letter
A_SPREAD = 16.0                          # landing spread inside one letter (frames)
REL0, REL1 = 2927.0, 2938.0              # mean release frame of the first / last letter
R_SPREAD = 8.0
END = 2961.0                             # every ember is dark by here
BIRTH_MIN = 2807.0                       # the first answering fires are alight by here
SHUT = 0.6                               # shutter (frames) for the streaks
T0_GRID, T1_GRID, DT = 2776.0, 2970.0, 0.25
FIRE_GAP = 3.0                           # min frames between two sparks launched by one fire

GOLD = look.hexrgb(look.PALETTE['accord_gold'])
PALE = look.hexrgb(look.PALETTE['accord_pale'])
AMBER = look.hexrgb(look.PALETTE['accord_amber'])


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ the CODA crane camera ---

def crane(fv2):
    """Replica of shots/hills/coda.py camera() for the wide (C2), at v2 frame fv2 (src = v2-160).
    Returns (pos, R (world = R @ cam, columns right/up/fwd), f_px)."""
    f = float(fv2) - 160.0
    c = float(smoothstep(2662.0, 2800.0, f))
    c = c * c * (3 - 2 * c)
    pos = np.array([1.25, 0.05 + 13.0 * c, 5.2 - 2.0 * c])
    pitch = math.radians(1.55 + 12.5 * c)
    yaw = math.radians(2.0)
    fpx = (W / 2) / math.tan(math.radians(50.0 + 6.0 * c) / 2)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = np.array([sy_ * cp, sp, cy_ * cp])
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    return pos, np.stack([right, up, fwd], axis=1), fpx


def project(P, fv2):
    """World points (...,3) -> screen (x, y, depth) in full-res pixels at v2 frame fv2."""
    pos, R, fpx = crane(fv2)
    q = (np.asarray(P, np.float64) - pos) @ R
    z = np.where(np.abs(q[..., 2]) < 1e-9, 1e-9, q[..., 2])
    return W / 2 + fpx * q[..., 0] / z, H / 2 - fpx * q[..., 1] / z, z


def _dir_flow(x, y, fv2, eps=0.25):
    """Screen velocity (px/frame) of very distant world points seen at (x, y): the crane."""
    _, R0, f0 = crane(fv2 - eps)
    _, R1, f1 = crane(fv2 + eps)
    d = np.stack([(x - W / 2) / f0, -(y - H / 2) / f0, np.ones_like(x)], -1) @ R0.T
    q = d @ R1
    x1 = W / 2 + f1 * q[..., 0] / q[..., 2]
    y1 = H / 2 - f1 * q[..., 1] / q[..., 2]
    return (x1 - x) / (2 * eps), (y1 - y) / (2 * eps)


CF_X = np.linspace(-400.0, W + 400.0, 15)
CF_Y = np.linspace(-400.0, H + 500.0, 11)
CF_T0, CF_DT = T0_GRID - 2.0, 1.0
CF_NT = int((T1_GRID + 4.0 - CF_T0) / CF_DT) + 1


def _camflow_table():
    gx, gy = np.meshgrid(CF_X, CF_Y)
    tab = np.zeros((CF_NT, len(CF_Y), len(CF_X), 2), np.float64)
    for k in range(CF_NT):
        u, v = _dir_flow(gx, gy, CF_T0 + k * CF_DT)
        tab[k, :, :, 0] = u
        tab[k, :, :, 1] = v
    return tab


# crest silhouette (our hilltop) and the figures' boxes: far sparks are hidden behind them
_CREST_P = np.stack([FD.CREST_X0 + FD.CREST_DX * np.arange(len(FD.CREST_H)), np.array(FD.CREST_H),
                     np.full(len(FD.CREST_H), FD.CREST_Z)], 1)


def crest_y(xs, fv2):
    """Screen y of our hilltop's silhouette at screen columns xs (below it = hidden)."""
    sx, sy, _ = project(_CREST_P, fv2)
    o = np.argsort(sx)
    return np.interp(xs, sx[o], sy[o])


def _ground(x):
    return float(np.interp(x, _CREST_P[:, 0], _CREST_P[:, 1]))


# cairn, child, elder in the wide (coda.py: X_CAIRN=2.20, xs(C2) = (3.22, 2.74)); world boxes (x0,x1,y0,y1)
_BOXES_W = [(2.20 - 0.50, 2.20 + 0.50, _ground(2.20) - 0.2, _ground(2.20) + 1.30),
            (2.74 - 0.24, 2.74 + 0.22, _ground(2.74) - 0.2, _ground(2.74) + 1.08),
            (3.22 - 0.26, 3.22 + 0.27, _ground(3.22) - 0.2, _ground(3.22) + 1.70)]


def occluder_boxes(fv2):
    """Screen rectangles (x0, x1, y0, y1) of the cairn and the two figures at frame fv2."""
    out = []
    for x0, x1, y0, y1 in _BOXES_W:
        P = np.array([[x0, y0, FD.CREST_Z], [x1, y0, FD.CREST_Z], [x0, y1, FD.CREST_Z], [x1, y1, FD.CREST_Z]])
        sx, sy, _ = project(P, fv2)
        out.append((sx.min(), sx.max(), sy.min(), sy.max()))
    return np.array(out, np.float64)


FIRES = np.array(FD.FIRES, np.float64)          # x, y, z, f_ign(v2), size, kind
CAIRN = int(np.nonzero(FIRES[:, 5] == 2)[0][0])


def fire_screen(fv2):
    """Screen positions of every fire at frame fv2 -> (sx, sy, lit, visible)."""
    sx, sy, z = project(FIRES[:, :3], fv2)
    lit = FIRES[:, 3] <= fv2
    vis = (z > 0) & (sx > -20) & (sx < W + 20) & (sy > -20) & (sy < H - 2)
    vis &= sy < crest_y(sx, fv2) - 2.0
    bx = occluder_boxes(fv2)
    far = FIRES[:, 5] != 2
    for x0, x1, y0, y1 in bx:
        vis &= ~(far & (sx > x0) & (sx < x1) & (sy > y0) & (sy < y1))
    return sx, sy, lit, vis


def hidden_at(x, y, fv2, cache={}):
    """Is screen point (x, y) behind our hilltop / the figures (or off the bottom) at fv2?"""
    key = round(float(fv2) * 2) / 2
    if key not in cache:
        cache[key] = (crest_y(np.arange(W, dtype=np.float64), key), occluder_boxes(key))
    cr, bx = cache[key]
    if x < 0 or x >= W or y > H - 1:
        return True
    if y > np.interp(x, np.arange(W), cr) - 1:
        return True
    for b in bx:
        if b[0] < x < b[1] and b[2] < y < b[3]:
            return True
    return False


# ------------------------------------------------------------------------- curl noise flow ---

_PERM = np.random.default_rng(20260926).permutation(256).astype(np.int64)
_PERM = np.concatenate([_PERM, _PERM])
OCT_L = np.array([330.0, 150.0, 64.0])          # px
OCT_V = np.array([1.35, 0.70, 0.30])            # px/frame (speed scale per octave)
OCT_T = np.array([70.0, 36.0, 18.0])            # frames (pattern evolution)
ADV = 1.6                                       # the eddies rise with the air (px/frame)


@njit(cache=True, fastmath=True, inline='always')
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, fastmath=True, inline='always')
def _grad(h, x, y, z):
    h = h & 15
    u = x if h < 8 else y
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    else:
        v = z
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


@njit(cache=True, fastmath=True)
def _perlin3(x, y, z, P):
    fx = math.floor(x)
    fy = math.floor(y)
    fz = math.floor(z)
    X = int(fx) & 255
    Y = int(fy) & 255
    Z = int(fz) & 255
    x -= fx
    y -= fy
    z -= fz
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)
    A = P[X] + Y
    AA = P[A] + Z
    AB = P[A + 1] + Z
    B = P[X + 1] + Y
    BA = P[B] + Z
    BB = P[B + 1] + Z
    g000 = _grad(P[AA], x, y, z)
    g100 = _grad(P[BA], x - 1, y, z)
    g010 = _grad(P[AB], x, y - 1, z)
    g110 = _grad(P[BB], x - 1, y - 1, z)
    g001 = _grad(P[AA + 1], x, y, z - 1)
    g101 = _grad(P[BA + 1], x - 1, y, z - 1)
    g011 = _grad(P[AB + 1], x, y - 1, z - 1)
    g111 = _grad(P[BB + 1], x - 1, y - 1, z - 1)
    a0 = g000 + u * (g100 - g000)
    a1 = g010 + u * (g110 - g010)
    a2 = g001 + u * (g101 - g001)
    a3 = g011 + u * (g111 - g011)
    b0 = a0 + v * (a1 - a0)
    b1 = a2 + v * (a3 - a2)
    return b0 + w * (b1 - b0)


@njit(cache=True, fastmath=True)
def _psi(x, y, t, P, OL, OV, OT, adv):
    s = 0.0
    for o in range(OL.shape[0]):
        L = OL[o]
        s += OV[o] * L * _perlin3(x / L + 17.31 * o + 0.5, (y + adv * t) / L + 5.13 * o + 0.5,
                                  t / OT[o] + 3.71 * o, P)
    return s


@njit(cache=True, fastmath=True)
def _curl(x, y, t, P, OL, OV, OT, adv):
    """Divergence-free 2-D flow: the curl of a 3-octave Perlin stream function."""
    e = 0.8
    dpy = (_psi(x, y + e, t, P, OL, OV, OT, adv) - _psi(x, y - e, t, P, OL, OV, OT, adv)) / (2 * e)
    dpx = (_psi(x + e, y, t, P, OL, OV, OT, adv) - _psi(x - e, y, t, P, OL, OV, OT, adv)) / (2 * e)
    return dpy, -dpx


@njit(cache=True, fastmath=True)
def _camflow(x, y, t, CF, cfp):
    """Trilinear lookup of the crane's screen flow (points at infinity)."""
    nt, ny, nx = CF.shape[0], CF.shape[1], CF.shape[2]
    ft = min(max((t - cfp[4]) / cfp[5], 0.0), nt - 1.001)
    fy = min(max((y - cfp[2]) / cfp[3], 0.0), ny - 1.001)
    fx = min(max((x - cfp[0]) / cfp[1], 0.0), nx - 1.001)
    it, iy, ix = int(ft), int(fy), int(fx)
    at, ay, ax = ft - it, fy - iy, fx - ix
    u = 0.0
    v = 0.0
    for dt_ in range(2):
        wt = at if dt_ else 1 - at
        for dy_ in range(2):
            wy = ay if dy_ else 1 - ay
            for dx_ in range(2):
                wx = ax if dx_ else 1 - ax
                w = wt * wy * wx
                u += w * CF[it + dt_, iy + dy_, ix + dx_, 0]
                v += w * CF[it + dt_, iy + dy_, ix + dx_, 1]
    return u, v


@njit(cache=True, fastmath=True)
def _v_rise(x, y, t, pr, P, OL, OV, OT, adv, CF, cfp):
    """Forward-time velocity of a letter ember before its glide: launched at birth (pr[1]),
    rising on the flow and slowing into its hover (reached at pr[0]); after that it only drifts.
    pr = [tg, tb, Ur, Uh, Td, UB, UBT, TS, WD, CC]."""
    tau = pr[0] - t
    age = t - pr[1]
    if age < 0.0:
        age = 0.0
    launch = pr[2] * pr[5] * math.exp(-age / pr[6])
    if tau > 0.0:
        U = pr[3] + (pr[2] - pr[3]) * (1.0 - math.exp(-tau / pr[4]))
        c = (tau - 4.0) / 18.0
        c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
        c = c * c * (3 - 2 * c) * pr[9]
    else:
        U = pr[3]
        c = 0.0
    cx, cy = _curl(x, y, t, P, OL, OV, OT, adv)
    fu, fv = _camflow(x, y, t, CF, cfp)
    return pr[7] * cx + pr[8] + c * fu, pr[7] * cy - U - launch + c * fv


@njit(cache=True, fastmath=True)
def _v_free(x, y, t, pr, P, OL, OV, OT, adv, CF, cfp):
    """Forward-time velocity of a free / released ember.
    pr = [t0 (birth or release), U0, UB, UBT, Gr (start-from-rest time, 0 = none), TS, WD, CC,
          KX, KY, KT (its own drift, decaying over KT: neighbours part company)]."""
    age = t - pr[0]
    if age <= 0.0:
        return 0.0, 0.0
    g = 1.0
    if pr[4] > 0.0:
        q = age / pr[4]
        g = 1.0 - math.exp(-q * q)
    rise = pr[1] * (1.0 + pr[2] * math.exp(-age / pr[3]))
    kd = math.exp(-age / pr[10])
    cx, cy = _curl(x, y, t, P, OL, OV, OT, adv)
    fu, fv = _camflow(x, y, t, CF, cfp)
    return (g * (pr[5] * cx + pr[6] + pr[8] * kd) + pr[7] * fu,
            g * (pr[5] * cy - rise + pr[9] * kd) + pr[7] * fv)


@njit(cache=True, fastmath=True)
def _paths(OUT, X0, Y0, T0, PR, kind, t0, dt, P, OL, OV, OT, adv, CF, cfp):
    """OUT[k, i] = position at grid time t0 + k*dt of a particle that is at (X0, Y0) at time T0.
    kind 0: letter ember carrier (_v_rise; backward before T0, forward drift after).
    kind 1: free / released ember (_v_free; forward after T0, held at the start before)."""
    K = OUT.shape[0]
    N = OUT.shape[1]
    for i in range(N):
        pr = PR[i]
        k0 = int(math.ceil((T0[i] - t0) / dt - 1e-9))      # first grid index with t >= T0
        # before T0
        x = X0[i]
        y = Y0[i]
        tc = T0[i]
        for k in range(min(k0, K) - 1, -1, -1):
            if kind == 1:
                OUT[k, i, 0] = X0[i]
                OUT[k, i, 1] = Y0[i]
                continue
            tk = t0 + k * dt
            h = tc - tk
            u1, v1 = _v_rise(x, y, tc, pr, P, OL, OV, OT, adv, CF, cfp)
            u2, v2 = _v_rise(x - 0.5 * h * u1, y - 0.5 * h * v1, tc - 0.5 * h, pr, P, OL, OV, OT, adv, CF, cfp)
            x -= h * u2
            y -= h * v2
            tc = tk
            OUT[k, i, 0] = x
            OUT[k, i, 1] = y
        # from T0 on
        x = X0[i]
        y = Y0[i]
        tc = T0[i]
        for k in range(max(k0, 0), K):
            tk = t0 + k * dt
            h = tk - tc
            if h > 0.0:
                if kind == 0:
                    u1, v1 = _v_rise(x, y, tc, pr, P, OL, OV, OT, adv, CF, cfp)
                    u2, v2 = _v_rise(x + 0.5 * h * u1, y + 0.5 * h * v1, tc + 0.5 * h, pr, P, OL, OV, OT, adv,
                                     CF, cfp)
                else:
                    u1, v1 = _v_free(x, y, tc, pr, P, OL, OV, OT, adv, CF, cfp)
                    u2, v2 = _v_free(x + 0.5 * h * u1, y + 0.5 * h * v1, tc + 0.5 * h, pr, P, OL, OV, OT, adv,
                                     CF, cfp)
                x += h * u2
                y += h * v2
                tc = tk
            OUT[k, i, 0] = x
            OUT[k, i, 1] = y


# ------------------------------------------------------------------------------- splatting ---

@njit(cache=True, fastmath=True)
def _gauss(buf, x, y, sig, e, cr, cg, cb, x0, y0):
    """Energy-exact gaussian (normalised by its discrete sum: no sub-pixel twinkle)."""
    Hh = buf.shape[0]
    Ww = buf.shape[1]
    ext = int(math.ceil(3.0 * sig)) + 1
    ix = int(math.floor(x + 0.5)) - x0
    iy = int(math.floor(y + 0.5)) - y0
    lx = x - x0
    ly = y - y0
    if ix + ext < 0 or iy + ext < 0 or ix - ext >= Ww or iy - ext >= Hh:
        return
    a = -0.5 / (sig * sig)
    s = 0.0
    for yy in range(iy - ext, iy + ext + 1):
        dy = yy - ly
        for xx in range(ix - ext, ix + ext + 1):
            dx = xx - lx
            s += math.exp(a * (dx * dx + dy * dy))
    if s <= 1e-12:
        return
    k = e / s
    for yy in range(max(iy - ext, 0), min(iy + ext, Hh - 1) + 1):
        dy = yy - ly
        for xx in range(max(ix - ext, 0), min(ix + ext, Ww - 1) + 1):
            dx = xx - lx
            w = k * math.exp(a * (dx * dx + dy * dy))
            buf[yy, xx, 0] += w * cr
            buf[yy, xx, 1] += w * cg
            buf[yy, xx, 2] += w * cb


@njit(cache=True, fastmath=True)
def _disc(buf, x, y, r, e, cr, cg, cb):
    """Soft bokeh disc of radius r (a faintly brighter rim, like a real lens), energy e."""
    Hh = buf.shape[0]
    Ww = buf.shape[1]
    edge = 0.9
    ext = r + edge
    ix0 = int(math.floor(x - ext))
    ix1 = int(math.ceil(x + ext))
    iy0 = int(math.floor(y - ext))
    iy1 = int(math.ceil(y + ext))
    if ix1 < 0 or iy1 < 0 or ix0 >= Ww or iy0 >= Hh:
        return
    s = 0.0
    for yy in range(iy0, iy1 + 1):
        dy = yy - y
        for xx in range(ix0, ix1 + 1):
            dx = xx - x
            d = math.sqrt(dx * dx + dy * dy)
            w = (r + edge - d) / (2 * edge)
            if w > 0.0:
                w = min(w, 1.0) * (0.85 + 0.3 * (d / r) ** 4 if d < r else 1.15)
                s += w
    if s <= 1e-12:
        return
    k = e / s
    for yy in range(max(iy0, 0), min(iy1, Hh - 1) + 1):
        dy = yy - y
        for xx in range(max(ix0, 0), min(ix1, Ww - 1) + 1):
            dx = xx - x
            d = math.sqrt(dx * dx + dy * dy)
            w = (r + edge - d) / (2 * edge)
            if w > 0.0:
                w = min(w, 1.0) * (0.85 + 0.3 * (d / r) ** 4 if d < r else 1.15) * k
                buf[yy, xx, 0] += w * cr
                buf[yy, xx, 1] += w * cg
                buf[yy, xx, 2] += w * cb


@njit(cache=True, fastmath=True)
def _vis(x, y, occ, crest, boxes):
    if occ == 0:
        return 1.0
    ix = int(x)
    if ix < 0:
        ix = 0
    if ix > crest.shape[0] - 1:
        ix = crest.shape[0] - 1
    v = (crest[ix] - y) / 1.5 + 0.5
    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    for b in range(boxes.shape[0]):
        if x > boxes[b, 0] and x < boxes[b, 1] and y > boxes[b, 2] and y < boxes[b, 3]:
            return 0.0
    return v


@njit(cache=True, fastmath=True)
def _splat(buf, X0, Y0, X1, Y1, E, COL, RAD, COC, OCC, crest, boxes):
    """Tapered hot-head streaks (tail X0,Y0 -> head X1,Y1) or motion-blurred bokeh discs."""
    for i in range(X0.shape[0]):
        e = E[i]
        if e <= 1e-6:
            continue
        x0 = X0[i]
        y0 = Y0[i]
        dx = X1[i] - x0
        dy = Y1[i] - y0
        L = math.sqrt(dx * dx + dy * dy)
        if max(x0, x0 + dx) < -40 or min(x0, x0 + dx) > buf.shape[1] + 40:
            continue
        if max(y0, y0 + dy) < -40 or min(y0, y0 + dy) > buf.shape[0] + 40:
            continue
        cr = COL[i, 0]
        cg = COL[i, 1]
        cb = COL[i, 2]
        c = COC[i]
        if c > 2.2:
            n = int(L / (0.45 * c)) + 1
            for k in range(n):
                u = (k + 0.5) / n
                xs = x0 + dx * u
                ys = y0 + dy * u
                vv = _vis(xs, ys, OCC[i], crest, boxes)
                if vv > 0.0:
                    _disc(buf, xs, ys, c, e * vv / n, cr, cg, cb)
        else:
            s0 = math.sqrt(RAD[i] * RAD[i] + 0.25 * c * c)
            n = int(L / 0.3) + 1
            if n > 160:
                n = 160
            ws = 0.0
            for k in range(n):
                u = (k + 1.0) / n
                ws += (0.06 + u) ** 2.4
            for k in range(n):
                u = (k + 1.0) / n
                xs = x0 + dx * u
                ys = y0 + dy * u
                vv = _vis(xs, ys, OCC[i], crest, boxes)
                if vv <= 0.0:
                    continue
                w = (0.06 + u) ** 2.4 / ws
                _gauss(buf, xs, ys, s0 * (0.42 + 0.58 * u), e * w * vv, cr, cg, cb, 0, 0)


@njit(cache=True, fastmath=True)
def _stipple(buf, X, Y, B, sig, x0, y0):
    for i in range(X.shape[0]):
        if B[i] > 0.0:
            _gauss(buf, X[i], Y[i], sig, B[i], 1.0, 1.0, 1.0, x0, y0)


@njit(cache=True, fastmath=True)
def _noise_grid_nb(X, Y, z, P):
    out = np.empty(X.shape)
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            out[j, i] = _perlin3(X[j, i], Y[j, i], z, P)
    return out


def _noise_grid(X, Y, z):
    return _noise_grid_nb(np.ascontiguousarray(X), np.ascontiguousarray(Y), float(z), _PERM) * 2.0


# ------------------------------------------------------------------------------ the glyphs ---

def _blur_noise(rng, shape, sigma):
    n = rng.standard_normal(shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return (n - n.mean()) / (n.std() + 1e-9)


def glide_ease(u):
    """0 -> 1 with zero velocity at both ends; quick to start, long and soft to settle."""
    u = np.clip(u, 0.0, 1.0)
    return (1.0 - (1.0 - u) ** 3) ** 2


class Title:
    """Everything that does not change per frame: glyph fields, particles and their paths."""

    def __init__(self, n_letter=750, n_release=2200, n_free=440, n_bokeh=14, seed=2880):
        rng = np.random.default_rng(seed)
        a, stag, mw, mh = titles.render_line(TEXT, titles.CINZEL, 92, 500, 0.28)
        self.mask = a.astype(np.float32)
        self.mx0 = int(round(W / 2 - mw / 2))
        self.my0 = int(round(TITLE_Y - mh / 2))
        mh, mw = self.mask.shape
        n_chars = max(1, len(TEXT.strip()))
        letter = np.full(self.mask.shape, -1, np.int32)
        ink = self.mask > 0.0
        letter[ink] = np.round(stag[ink] * n_chars).astype(np.int32)
        self.nl = int(letter.max()) + 1
        ys, xs = np.nonzero(self.mask > 0.5)
        self.cap0, self.cap1 = ys.min(), ys.max()
        cxs = np.array([xs[letter[ys, xs] == k].mean() for k in range(self.nl)])
        ln = (cxs - cxs.min()) / (cxs.max() - cxs.min())
        if (ink & (letter < 0)).any():
            yy, xx = np.nonzero(ink & (letter < 0))
            for y_, x_ in zip(yy, xx):
                letter[y_, x_] = int(np.argmin(np.abs(cxs - x_)))
        lab = np.clip(letter, 0, None)
        self.letter = letter
        yrel = (np.arange(mh, dtype=np.float32)[:, None] - self.cap0) / max(1, self.cap1 - self.cap0)
        xrel = np.arange(mw, dtype=np.float32)[None, :]
        wid = np.array([max(8.0, np.ptp(xs[letter[ys, xs] == k])) for k in range(self.nl)])
        xin = (xrel - cxs[lab]) / wid[lab]
        # landing frames: letter stagger; inside a letter soft patches, lower parts first
        nA = _blur_noise(rng, self.mask.shape, 5.0)
        base = TA0 + (TA1 - TA0) * ln[lab]
        self.A = (base + A_SPREAD * (0.30 * np.tanh(0.8 * nA) - 0.28 * (yrel - 0.5)
                                     + 0.18 * xin)).astype(np.float32)
        # release frames: left to right, tops first, patchy
        nR = _blur_noise(rng, self.mask.shape, 6.0)
        self.R = (REL0 + (REL1 - REL0) * ln[lab] + R_SPREAD * (0.32 * np.tanh(0.8 * nR)
                                                              + 0.34 * (yrel - 0.5) + 0.12 * xin)).astype(np.float32)
        # heat: thick stems hotter than hairlines (distance to the edge, from a 4x raster)
        big = cv2.resize(self.mask, (mw * 4, mh * 4), interpolation=cv2.INTER_CUBIC)
        dt = cv2.distanceTransform((big > 0.5).astype(np.uint8), cv2.DIST_L2, 5) / 4.0
        dt = cv2.resize(dt, (mw, mh), interpolation=cv2.INTER_AREA)
        self.core = np.clip(dt / 3.6, 0, 1).astype(np.float32)
        # glyph colour (linear): amber edges -> gold -> a paler gold heart in the stems
        c = self.core[..., None]
        c16 = np.clip(c * 1.6, 0, 1)
        col = AMBER * 1.05 * (1 - c16) + GOLD * c16
        col = col * (1 - 0.25 * c ** 2) + PALE * 0.25 * c ** 2
        self.col = col.astype(np.float32)
        self.inten = (0.74 + 0.26 * self.core).astype(np.float32)
        self._init_stipple(rng)
        self.used = {}                                     # fire -> launch frames (rate limit)
        self._init_particles(rng, n_letter, n_release, n_free, n_bokeh)

    # -------------------------------------------------------------------------------------
    def _sample_mask(self, rng, n, lo=0.25):
        w = np.clip(self.mask - lo, 0, None).ravel()
        idx = rng.choice(w.size, size=n, p=w / w.sum())
        ty, tx = np.divmod(idx, self.mask.shape[1])
        return tx, ty

    def _init_stipple(self, rng):
        n = 11000
        tx, ty = self._sample_mask(rng, n, 0.05)
        self.st_x = (tx + rng.uniform(-0.5, 0.5, n)).astype(np.float64)
        self.st_y = (ty + rng.uniform(-0.5, 0.5, n)).astype(np.float64)
        self.st_b = rng.lognormal(0.0, 0.55, n)
        self.st_w = rng.uniform(0.07, 0.30, n)
        self.st_ph = rng.uniform(0, 2 * np.pi, n)
        self.st_a = rng.uniform(0.25, 0.6, n)
        self.st_A = self.A[ty, tx]
        self.st_R = self.R[ty, tx]
        acc = np.zeros(self.mask.shape + (3,), np.float32)
        _stipple(acc, self.st_x, self.st_y, self.st_b, 0.75, 0, 0)
        self.st_norm = float((acc[..., 0] * self.mask).sum() / self.mask.sum())

    # -------------------------------------------------------------------------------------
    def _take_fire(self, rng, t, x0, y0, far_only=False, sigma_x=120.0):
        """Pick a lit, visible fire near/below screen point (x0, y0) at frame t that has not
        launched a spark within FIRE_GAP frames. Returns (index, sx, sy) or None."""
        sx, sy, lit, vis = fire_screen(t)
        ok = lit & vis & (FIRES[:, 3] <= t - 2.0)
        if far_only:
            ok &= FIRES[:, 5] != 2
        for j in np.nonzero(ok)[0]:
            if any(abs(t - u) < FIRE_GAP for u in self.used.get(int(j), ())):
                ok[j] = False
        if x0 is None:
            w = np.where(ok, FIRES[:, 4], 0.0)
        else:
            dx = sx - x0
            dy = sy - y0
            w = np.exp(-(dx / sigma_x) ** 2) * np.exp(-(np.maximum(0.0, -dy) / 20.0) ** 2) * np.exp(-(dy / 240.0) ** 2)
            w = np.where(ok, w * FIRES[:, 4], 0.0)
        if w.sum() < 1e-4:
            return None
        j = int(rng.choice(len(w), p=w / w.sum()))
        self.used.setdefault(j, []).append(t)
        return j, sx[j], sy[j]

    def _init_particles(self, rng, nL, nR, nF, nB):
        K = int(round((T1_GRID - T0_GRID) / DT)) + 1
        self.K = K
        self.tt = T0_GRID + DT * np.arange(K)
        CF = _camflow_table()
        cfp = np.array([CF_X[0], CF_X[1] - CF_X[0], CF_Y[0], CF_Y[1] - CF_Y[0], CF_T0, CF_DT], np.float64)
        args = (T0_GRID, DT, _PERM, OCT_L, OCT_V, OCT_T, ADV, CF, cfp)
        self.nL, self.nR, self.nF, self.nB = nL, nR, nF, nB

        # ---------------- free embers first: they are the sparks visibly rising from the fires
        tbF = self._free_births(rng, nF)
        fx = np.zeros(nF)
        fy = np.zeros(nF)
        self.f_src = np.zeros(nF, np.int64)          # 0 below frame/hill, 1 far fire, 2 the cairn
        for i in np.argsort(tbF):
            t = tbF[i]
            got = self._take_fire(rng, t, None, None) if rng.random() < 0.85 else None
            if got is not None:
                j, sx, sy = got
                if j == CAIRN:
                    fx[i], fy[i] = sx + rng.normal(0, 14), sy - rng.uniform(40, 140)
                    self.f_src[i] = 2
                else:
                    fx[i], fy[i] = sx + rng.normal(0, 1.2), sy - rng.uniform(1, 3)
                    self.f_src[i] = 1
            else:
                fx[i], fy[i] = rng.uniform(-60, W + 60), H + rng.uniform(8, 50)
        PRF = np.stack([tbF, rng.uniform(1.3, 2.8, nF), rng.uniform(0.8, 2.4, nF), rng.uniform(5.0, 12.0, nF),
                        np.zeros(nF), rng.uniform(0.8, 1.6, nF), rng.uniform(0.3, 1.0, nF),
                        rng.uniform(0.35, 0.8, nF), np.zeros(nF), np.zeros(nF), np.ones(nF)], 1)
        Fp = np.zeros((K, nF, 2))
        _paths(Fp, fx, fy, tbF, PRF, 1, *args)
        self.f_tb = tbF

        # ---------------- letter embers
        tx, ty = self._sample_mask(rng, nL, 0.3)
        TX = self.mx0 + tx + rng.uniform(-0.5, 0.5, nL)
        TY = self.my0 + ty + rng.uniform(-0.5, 0.5, nL)
        TA = self.A[ty, tx] + rng.normal(0, 0.6, nL)
        TR = self.R[ty, tx] + rng.normal(0, 0.6, nL)
        # hover points: mostly below the target (the embers rise into their places), a few beside it
        side = rng.random(nL) < 0.10
        HX = TX + np.where(side, np.clip(rng.normal(0, 140, nL), -300, 300), np.clip(rng.normal(0, 90, nL), -220, 220))
        HY = TY + np.where(side, rng.normal(0, 40, nL), np.minimum(40 + rng.exponential(70, nL), 300))
        GG = np.clip(8.0 + np.hypot(HX - TX, HY - TY) / 6.0, 12.0, 32.0)     # gentle approach speeds
        GG = np.minimum(GG, TA - BIRTH_MIN - 14.0)
        TG = TA - GG
        DR = rng.uniform(16.0, 44.0, nL)
        TB = np.minimum(TG - 12.0, np.maximum(TG - DR, BIRTH_MIN + rng.uniform(0, 12, nL)))
        PRL = np.stack([TG, TB, rng.uniform(2.8, 4.6, nL), rng.uniform(1.0, 2.2, nL), rng.uniform(6.0, 12.0, nL),
                        rng.uniform(0.5, 1.3, nL), rng.uniform(3.0, 6.0, nL), rng.uniform(0.6, 1.2, nL),
                        rng.uniform(0.2, 0.7, nL), rng.uniform(0.4, 0.8, nL)], 1)
        C = np.zeros((K, nL, 2))
        _paths(C, HX, HY, TG, PRL, 0, *args)
        self.l_TA, self.l_TB, self.l_TG, self.l_TR = TA, TB, TG, TR
        self.l_T = np.stack([TX, TY], 1)
        # births in open sky: a spark launched by a real fire (rate-limited) or a fade-in from the haze
        self.l_birth = np.zeros(nL, np.int64)        # 0 hidden, 1 far fire, 2 cairn, 3 haze fade-in
        tt = self.tt
        for i in np.argsort(TB):
            tb = TB[i]
            kb = (tb - T0_GRID) / DT
            k0 = int(np.floor(kb))
            a = kb - k0
            x0 = C[k0, i, 0] * (1 - a) + C[k0 + 1, i, 0] * a
            y0 = C[k0, i, 1] * (1 - a) + C[k0 + 1, i, 1] * a
            if hidden_at(x0, y0, tb):
                continue
            got = self._take_fire(rng, tb, x0, y0, far_only=tb < 2852) if rng.random() < 0.55 else None
            if got is None:
                self.l_birth[i] = 3
                continue
            j, sx, sy = got
            if j == CAIRN:
                ox, oy = sx + rng.normal(0, 12) - x0, sy - rng.uniform(40, 120) - y0
                self.l_birth[i] = 2
            else:
                ox, oy = sx + rng.normal(0, 1.2) - x0, sy - rng.uniform(1, 3) - y0
                self.l_birth[i] = 1
            B = float(np.clip(0.5 * (TG[i] - tb), 6.0, 16.0))
            s = 1 - smoothstep(tb, tb + B, tt)
            s[tt < tb] = 1.0
            C[:, i, 0] += s * ox
            C[:, i, 1] += s * oy
        # glide: a curved blend from the drifting carrier onto the target (zero velocity at both ends)
        k = np.clip(np.round((TG - T0_GRID) / DT).astype(int), 0, K - 1)
        H0 = C[k, np.arange(nL)]
        d0 = self.l_T - H0
        perp = np.stack([-d0[:, 1], d0[:, 0]], 1)
        kappa = rng.normal(0, 0.22, nL)
        u = (tt[:, None] - TG[None, :]) / (TA - TG)[None, :]
        s = glide_ease(u)
        Lp = C * (1 - s)[..., None] + self.l_T[None] * s[..., None] + (kappa * np.sin(np.pi * s))[..., None] * perp[None]
        Lp = np.where((tt[:, None] >= TA[None, :])[..., None], self.l_T[None], Lp)
        # release: forward from the glyph, starting from rest, lifting and swirling away
        Rp = np.zeros((K, nL, 2))
        _paths(Rp, TX, TY, TR, self._release_params(rng, TR), 1, *args)
        rel = tt[:, None] >= TR[None, :]
        Lp[rel] = Rp[rel]

        # ---------------- release-only embers: part of the glyph's light until it loosens
        rx, ry = self._sample_mask(rng, nR, 0.3)
        RX = self.mx0 + rx + rng.uniform(-0.5, 0.5, nR)
        RY = self.my0 + ry + rng.uniform(-0.5, 0.5, nR)
        TRr = self.R[ry, rx] + rng.normal(0, 0.6, nR)
        Rq = np.zeros((K, nR, 2))
        _paths(Rq, RX, RY, TRr, self._release_params(rng, TRr), 1, *args)
        self.r_TR = TRr

        # ---------------- bokeh (foreground embers, out of focus; they sweep down as we rise)
        tbB = rng.uniform(2828.0, 2905.0, nB)
        bx = rng.uniform(60, W - 60, nB)
        by = rng.uniform(470, 780, nB)
        PRB = np.stack([tbB, rng.uniform(0.8, 1.6, nB), np.zeros(nB), np.ones(nB), np.zeros(nB),
                        rng.uniform(0.5, 1.0, nB), rng.uniform(0.2, 0.7, nB), rng.uniform(1.2, 1.6, nB),
                        np.zeros(nB), np.zeros(nB), np.ones(nB)], 1)
        Bp = np.zeros((K, nB, 2))
        _paths(Bp, bx, by, tbB, PRB, 1, *args)

        self.POS = np.concatenate([Lp, Rq, Fp, Bp], 1).astype(np.float32)
        N = nL + nR + nF + nB
        o = nL + nR
        self.occ = np.zeros(N, np.int64)
        self.occ[:nL] = np.where(self.l_birth == 2, 0, 1)
        self.occ[o:o + nF] = np.where(self.f_src == 2, 0, 1)
        # per-particle look
        self.temp0 = rng.uniform(0.84, 0.96, N)
        self.cool = rng.uniform(8.0, 16.0, N)
        self.flk_w = rng.uniform(0.15, 0.5, N)
        self.flk_ph = rng.uniform(0, 2 * np.pi, N)
        self.flk_a = rng.uniform(0.05, 0.18, N)
        self.rad = rng.uniform(0.55, 0.85, N)
        self.coc0 = rng.uniform(0.2, 2.0, N)
        # letter embers drift out of focus until they glide home (the title racks into focus)
        self.coc0[:nL] = np.clip(rng.lognormal(np.log(3.2), 0.55, nL), 1.0, 11.0)
        area = float(self.mask.sum())
        e_mean = float((self.inten * self.mask).sum() / area)
        eL = rng.lognormal(0.0, 0.6, nL)
        self.l_E = 0.28 * e_mean * area / nL * eL / eL.mean()      # the glyph itself carries most of the light
        self.l_GG = GG
        # on release the glyph's light goes back into the embers (~75% of it; sparks read brighter)
        eR = rng.lognormal(0.0, 0.45, nL + nR)
        self.e_rel = 0.75 * e_mean * area / (nL + nR) * eR / eR.mean()
        self.r_life = rng.uniform(10.0, 16.0, nL + nR)
        self.f_life = rng.uniform(30.0, 80.0, nF)
        self.f_E = rng.lognormal(np.log(2.2), 0.6, nF)
        self.b_tb = tbB
        self.b_E = rng.uniform(8.0, 22.0, nB)
        self.b_coc = rng.uniform(6.0, 12.0, nB)
        self.b_life = rng.uniform(40.0, 70.0, nB)

    def _release_params(self, rng, TR):
        """Loosening: from rest, a quick lift, strong swirl, and each ember's own drift."""
        n = len(TR)
        ang = -0.5 * np.pi + rng.normal(0, 0.85, n)           # mostly up, fanning out
        m = rng.uniform(0.8, 3.4, n)
        return np.stack([TR, rng.uniform(1.8, 6.4, n), np.zeros(n), np.ones(n), rng.uniform(2.5, 5.0, n),
                         rng.uniform(1.6, 2.8, n), rng.uniform(0.2, 0.7, n), rng.uniform(0.1, 0.3, n),
                         m * np.cos(ang), m * np.sin(ang), rng.uniform(10.0, 26.0, n)], 1)

    def _free_births(self, rng, n):
        """Birth frames of the free embers: sparse early, a swell through the gathering, a thin
        trickle through the hold, none once the letters start to loosen."""
        ts = np.arange(BIRTH_MIN, 2920.0, 0.5)
        dens = (smoothstep(BIRTH_MIN, 2830, ts) * (1.0 - 0.8 * smoothstep(2862, 2884, ts)))
        dens = np.maximum(dens, 1e-4) * (1 - smoothstep(2908, 2920, ts))
        cdf = np.cumsum(dens)
        cdf /= cdf[-1]
        return np.interp(rng.random(n), cdf, ts)

    # ------------------------------------------------------------------------ per frame ---
    def pos_at(self, t):
        k = (t - T0_GRID) / DT
        k0 = int(np.clip(np.floor(k), 0, self.K - 2))
        a = float(np.clip(k - k0, 0.0, 1.0))
        return self.POS[k0] * (1 - a) + self.POS[k0 + 1] * a

    def particles(self, f):
        """-> tail xy, head xy, energy, colour, radius, coc for frame f."""
        nL, nR, nF, nB = self.nL, self.nR, self.nF, self.nB
        head = self.pos_at(f)
        tail = self.pos_at(f - SHUT)
        N = head.shape[0]
        E = np.zeros(N)
        T = np.zeros(N)
        coc = self.coc0.copy()
        fl = 1.0 + self.flk_a * np.sin(self.flk_w * f + self.flk_ph) * np.sin(0.37 * self.flk_w * f + 2 * self.flk_ph)
        end = 1.0 - smoothstep(END - 12.0, END, f)
        # --- letter embers
        ta, tb, tg, tr = self.l_TA, self.l_TB, self.l_TG, self.l_TR
        age = f - tb
        haze = self.l_birth == 3
        born = np.where(haze, 0.8 * smoothstep(0.0, 9.0, age), smoothstep(0.0, 2.0, age))
        cooled = np.exp(-np.maximum(age, 0) / self.cool[:nL])
        rk = smoothstep(-4.0, self.l_GG, f - tg)                         # re-kindles as it glides home
        temp_fl = 0.58 + (self.temp0[:nL] - 0.58) * cooled
        temp_fl = temp_fl * (1 - rk) + 0.775 * rk
        e_fl = ((0.40 + 0.60 * cooled) * (1 - rk) + rk) * born * (f < ta)
        hand = (f >= ta) * (1 - smoothstep(ta - 1.0, ta + 3.0, f)) * (f < tr - 3.0)   # melts into the glyph
        sg = glide_ease((f - tg) / (ta - tg))
        coc[:nL] = self.coc0[:nL] * (1.0 - sg) ** 1.4 * (f < ta)
        E[:nL] = self.l_E * (e_fl + hand) * (1.0 + 0.12 * coc[:nL])   # a soft disc still reads as bright as a point
        T[:nL] = np.where(f < ta, temp_fl, 0.775)
        # --- release: letter embers + release-only embers leave the glyph, lift, cool and die
        s = slice(0, nL + nR)
        ra = f - np.concatenate([tr, self.r_TR])
        rel = smoothstep(-3.0, 2.0, ra) * np.exp(-(np.maximum(ra - 1.0, 0.0) / self.r_life) ** 1.6)
        on = ra >= -3.0
        E[s] = np.where(on, self.e_rel * rel, E[s])
        T[s] = np.where(on, 0.80 - 0.47 * smoothstep(0.0, 1.2 * self.r_life, ra), T[s])
        coc[s] = np.where(on, 0.3 + 1.6 * smoothstep(0.0, 18.0, ra), coc[s])
        E[:nL + nR] *= fl[:nL + nR] * end
        # --- free embers
        s = slice(nL + nR, nL + nR + nF)
        a = f - self.f_tb
        x = np.clip(a / self.f_life, 0, 1)
        E[s] = self.f_E * smoothstep(0.0, 2.0, a) * (1 - x) ** 1.3 * (a >= 0) * fl[s] * end
        T[s] = self.temp0[s] - 0.5 * x
        # --- bokeh
        s = slice(nL + nR + nF, N)
        a = f - self.b_tb
        x = np.clip(a / self.b_life, 0, 1)
        E[s] = self.b_E * np.sin(np.pi * x) ** 1.5 * (a >= 0) * (0.85 + 0.15 * fl[s]) * end
        T[s] = 0.68 + 0.06 * np.sin(self.flk_ph[s])
        coc[s] = self.b_coc
        col = look.blackbody(T).astype(np.float64)
        return tail, head, E, col, self.rad, coc

    def glyph(self, f):
        """The formed-glyph light in the mask box (mh x mw x 3, linear), or None."""
        A, R = self.A, self.R
        rev = smoothstep(A - 3.0, A + 2.5, f)
        dis = 1.0 - smoothstep(R - 2.0, R + 3.0, f)
        on = (rev * dis).astype(np.float32)
        if on.max() <= 0:
            return None
        b = self.st_b * (1.0 + self.st_a * np.sin(self.st_w * f + self.st_ph))
        b = b * smoothstep(self.st_A - 1.0, self.st_A + 3.0, f) * (1 - smoothstep(self.st_R - 2.0, self.st_R + 2.0, f))
        acc = np.zeros(self.mask.shape + (3,), np.float32)
        _stipple(acc, self.st_x, self.st_y, b, 0.75, 0, 0)
        st = acc[..., 0] / self.st_norm
        tex = 0.66 + 0.34 * st
        heat = self._heat(f)
        breath = 1.0 + 0.025 * math.sin(2 * math.pi * f / 70.0) + 0.015 * math.sin(2 * math.pi * f / 31.0 + 1.3)
        flash = np.exp(-np.maximum(f - A, 0.0) / 5.0) * smoothstep(A - 1.0, A + 1.0, f)
        crumble = smoothstep(R - 7.0, R - 1.0, f) * dis
        I = self.inten * tex * heat * breath * (1.0 + 0.5 * flash + 0.3 * crumble)
        lum = (self.mask * on * I).astype(np.float32)
        hot = np.clip(0.45 * flash + 0.2 * crumble, 0, 1)[..., None].astype(np.float32)
        col = self.col * (1 - hot) + PALE * hot
        return lum[..., None] * col

    def _heat(self, f):
        mh, mw = self.mask.shape
        gy, gx = np.mgrid[0:mh:6, 0:mw:6].astype(np.float64)
        n = np.zeros(gx.shape)
        for (L, a, sp) in ((70.0, 0.07, 0.012), (26.0, 0.04, 0.03)):
            n += a * _noise_grid(gx / L + 3.0 * f * sp, gy / L - f * sp, f * sp * 0.7)
        n = cv2.resize(n.astype(np.float32), (mw, mh), interpolation=cv2.INTER_CUBIC)
        return 1.0 + n


# -------------------------------------------------------------------------------- the layer ---

_TITLE = None


def get_title():
    global _TITLE
    if _TITLE is None:
        _TITLE = Title()
    return _TITLE


def halo(img):
    """Soft warm halo: a tight glow and two wider ones, built on a pyramid (cheap)."""
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    g1 = cv2.GaussianBlur(small, (0, 0), 1.6)
    g2 = cv2.GaussianBlur(small, (0, 0), 6.0)
    q = cv2.resize(small, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    g3 = cv2.GaussianBlur(q, (0, 0), 3.2)
    g1 = cv2.resize(g1, (W, H), interpolation=cv2.INTER_LINEAR)
    g2 = cv2.resize(g2, (W, H), interpolation=cv2.INTER_LINEAR)
    g3 = cv2.resize(g3, (W, H), interpolation=cv2.INTER_LINEAR)
    tint = np.array([1.0, 0.80, 0.55], np.float32)
    return 0.30 * g1 + 0.16 * g2 * tint + 0.09 * g3 * tint * np.array([1.0, 0.85, 0.7], np.float32)


def render(f):
    """The linear-light additive title layer for v2 frame f (HxWx3 float32)."""
    T = get_title()
    buf = np.zeros((H, W, 3), np.float32)
    tail, head, E, col, rad, coc = T.particles(f)
    crest = crest_y(np.arange(W, dtype=np.float64), f).astype(np.float64)
    boxes = occluder_boxes(f)
    _splat(buf, tail[:, 0].astype(np.float64), tail[:, 1].astype(np.float64), head[:, 0].astype(np.float64),
           head[:, 1].astype(np.float64), E.astype(np.float64), col, rad.astype(np.float64),
           coc.astype(np.float64), T.occ, crest, boxes)
    g = T.glyph(f)
    if g is not None:
        mh, mw = T.mask.shape
        buf[T.my0:T.my0 + mh, T.mx0:T.mx0 + mw] += g
    return buf + halo(buf)


# ------------------------------------------------------------------------- compositing / io ---

def soft_clip(x, knee=0.8):
    """Identity below `knee`, exponential shoulder to 1.0 above (C1 at the knee)."""
    x = np.maximum(np.asarray(x, np.float32), 0.0)
    s = 1.0 - knee
    return np.where(x <= knee, x, knee + s * (1.0 - np.exp(-(x - knee) / s))).astype(np.float32)


def frame_path(f):
    return os.path.join(OUT_DIR, f't_{f:05d}.exr')


def layer(f):
    """The title layer for v2 frame f: HxWx3 float32 linear light (additive), or None outside
    2800-2966. Loads renders/title/t_%05d.exr (renders it on the fly if the file is missing)."""
    if f < F0 or f > F1:
        return None
    p = frame_path(f)
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED) if os.path.exists(p) else None
    if img is None:
        return render(f)
    return np.ascontiguousarray(img[..., ::-1]).astype(np.float32)


def composite(img_srgb, f):
    """img_srgb: the 1920x804 picture, display sRGB float32 in [0,1] (after picture(), before
    titles.composite / finish). Returns the picture with the ember title added in linear light."""
    lay = layer(f)
    if lay is None:
        return img_srgb
    return look.linear_to_srgb(soft_clip(look.srgb_to_linear(img_srgb) + lay))


def write(f, img):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = frame_path(f)[:-4] + '.tmp.exr'
    ok = cv2.imwrite(tmp, np.ascontiguousarray(img[..., ::-1]).astype(np.float32),
                     [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
                      cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
    if not ok:
        raise RuntimeError('could not write ' + tmp)
    os.replace(tmp, frame_path(f))


def _work(frames):
    cv2.setNumThreads(1)
    t0 = time.time()
    get_title()
    t1 = time.time()
    for f in frames:
        write(f, render(f))
    return len(frames), t1 - t0, time.time() - t1


def render_range(a, b, workers=2):
    frames = list(range(a, b + 1))
    t0 = time.time()
    if workers <= 1:
        n, tp, tr = _work(frames)
        print(f'{n} frames: setup {tp:.1f}s, render {tr:.1f}s')
    else:
        from multiprocessing import get_context
        chunks = [frames[i::workers] for i in range(workers)]
        with get_context('spawn').Pool(workers) as pool:
            for n, tp, tr in pool.imap_unordered(_work, chunks):
                print(f'  worker: {n} frames, setup {tp:.1f}s, render {tr:.1f}s', flush=True)
    print(f'rendered {a}-{b} in {time.time() - t0:.1f}s -> {OUT_DIR}')


# -------------------------------------------------------------------------------- review ---

def plate(f):
    """The real background (CODA, hills src = v2 - 160), display sRGB float32."""
    p = look.frame_path(os.path.join(ROOT, 'renders', 'hills'), f - 160)
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    return img[..., ::-1].astype(np.float32) / 255.0


def edit_fade(f, total=2968):
    """assemble.picture()'s fade to black (so the review matches the film)."""
    x = float(np.clip((total - 3 - f) / 50.0, 0, 1))
    return x * x * (3 - 2 * x)


def review(frames=(2810, 2830, 2850, 2865, 2880, 2905, 2935, 2950), out=REVIEW, cols=2, tw=960, fade=True):
    tiles = []
    for f in frames:
        bg = plate(f)
        if fade:
            bg = look.linear_to_srgb(look.srgb_to_linear(bg) * edit_fade(f))
        img = composite(bg, f)
        t = cv2.resize((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), (tw, int(tw * H / W)),
                       interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(f), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.vstack(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--range', nargs=2, type=int, default=[F0, F1])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--review', action='store_true', help='write the composite review sheet')
    a = ap.parse_args(argv)
    if a.review:
        print(review())
    else:
        render_range(a.range[0], a.range[1], a.workers)


if __name__ == '__main__':
    # run under the module's real name: numba's on-disk cache and the worker pool both refer to
    # functions by module name, and the edit imports this file as `ember_title`
    import ember_title as _self
    _self.main()
