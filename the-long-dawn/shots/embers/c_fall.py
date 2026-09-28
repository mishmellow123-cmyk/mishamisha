"""EMBERS-C4 . E13a THE RING FALLS, part a (cut C, C13, C frames 2720-2840, bars 35-36 b2). v2.

"Out of the black a gold glint tumbles slowly, its letters faintly awake, down through cloud." RUN-C takes part b at
2840 (bar 36 b3, renders/runC_ringfall): over a moonlit ink range the star enters at frame upper-left (248, 31) as a
~3 px warm-white dot and streaks down-right at ~29 deg. This shot hands over to that exact spot, size and direction.

    2720  black (C12's black runs on); the band turns into the moon's light: a glint, then its gold; letters faint
    2724  far below, a moonlit sea of cloud and a thin high layer of wisps come up out of the dark; we fall with the band
    2760  we slow to a hover; the band does not: it drops away below us, and we tilt down after it (past the vertical,
          the frame slowly rolling), the wisps' lit tops sliding up past it
    2800  it goes down through the wisps: they glow warm round it and part where it went through
    2833  it has become a glint in the frame's upper-left corner; the camera settles
    2839  the glint (~5 px, warm white core) at (240, 28), sinking down-right toward the sea -> RUN-C's star at 2840

Geometry: the camera is solved per frame so that the band lands on a designed screen path (`ring_px`); the band
itself falls straight down in the world, so its motion against the clouds is honest (down, turning down-right as
the frame rolls). The clouds are horizontal planes seen from above: the deck is a height field of billows (|noise|
octaves: round tops, sharp creases) lit by a low moon with self-shadowing, cavity darkening and haze, octaves faded
by the pixel footprint; the wisps are a sparse translucent layer the band lights from inside. Both are motion-
blurred by reprojection when the camera turns.
"""
import math

import cv2
import numpy as np
from numba import njit, prange

from core import Camera, smoothstep, perlin3
import ringsolid as RS
import c2
from c2 import T_FALL, T_FALL_END

FW, FH = 1920, 804                                   # the screen path is designed at full resolution
HFOV = 42.0
RING_W = 1.3
OFF = np.array([-0.55 * 52.0, -70.0, 0.83 * 52.0])   # the band from the lens while we fall together (the kept opening)
V0 = 6.0                                             # our common fall (units per frame) until T_LET
T_LET = 2760.0
BRAKE = 40.0                                         # frames for us to come to a hover
DROP_A, DROP_K = 36.0, 0.055                         # the band falls away from us: exp-like (it shrinks steadily)
PS0 = (640.0, 262.0)                                 # the band on screen while we fall together (lead room down-right)
PS1 = (220.0, 17.0)                                  # the corner it settles into as we tilt after it
PS2 = (240.0, 28.0)                                  # 2839: just behind RUN-C's first star position (248, 31)
T_S0, T_S1 = 2762.0, 2833.0
PHI = math.radians(-60.0)                            # the frame's slow roll during the tilt (so it sinks down-right)
C_MOON = np.array([0.62, 0.72, 0.9])
DECK_Y = -6500.0                                     # the sea of cloud, far below (the band never reaches it here)
WISP_Y = -1100.0                                     # the thin high layer the band goes through (~2812)
DIAM = 2.0 * RS.R_OUT * RING_W                       # the band's outer diameter (RING_W is its WIDTH)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


H0 = _unit([OFF[0], 0.0, OFF[2]])


def _roty(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


def _ss(a, b, x):
    return float(smoothstep(a, b, x))


# ============================================================ motion ===

def cam_pos(t):
    x = t - T_FALL
    xl = T_LET - T_FALL
    if x <= xl:
        y = -V0 * x
    else:
        k = min(x - xl, BRAKE)
        y = -V0 * xl - V0 * (k - k * k / (2.0 * BRAKE))
    return np.array([0.0, y, 0.0])


def drop(t):
    """how far the band has fallen away below us since we let it go: it keeps our old speed while we brake, and
    falls away ever faster (5 %/frame at the end: 168 px at 2760 -> ~5 px at 2839)"""
    y = t - T_LET
    if y <= 0.0:
        return 0.0
    k = min(y, BRAKE)
    return DROP_A * (math.exp(DROP_K * y) - 1.0 - DROP_K * y) + V0 * (k * k / (2.0 * BRAKE) + (y - k))


def cross_t():
    """when the band goes through the thin layer"""
    a, b = T_LET, T_FALL_END
    for _ in range(40):
        m = 0.5 * (a + b)
        if ring_pos(m)[1] > WISP_Y:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def ring_pos(t):
    return cam_pos(t) + OFF + np.array([0.0, -drop(t), 0.0])


def ring_px(t):
    """the band's designed screen position (full-res px)"""
    k = _ss(T_S0, T_S1, t)
    p = np.array(PS0) * (1.0 - k) + np.array(PS1) * k
    if t > T_S1:
        k2 = (t - T_S1) / (2839.0 - T_S1)
        p = np.array(PS1) + (np.array(PS2) - np.array(PS1)) * min(k2, 1.2) ** 1.3
    return p


def up_hint(t):
    return _roty(H0, PHI * _ss(2758.0, 2836.0, t))


def ring_rot(t):
    """tumbling slowly; its hole kept roughly three-quarter to the lens (never a flat halo)"""
    x = t - T_FALL
    a = 0.045 * x + 0.9
    b = 0.028 * x + 0.4
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Rz = np.array([[cb, -sb, 0], [sb, cb, 0], [0, 0, 1]])
    return Rz @ Rx


_CAM = {}


def solve_cam(t):
    """the camera at t: at cam_pos(t), oriented (no roll beyond up_hint) so the band lands on ring_px(t)"""
    key = round(float(t), 4)
    c = _CAM.get(key)
    if c is not None:
        return c
    pos = cam_pos(t)
    ring = ring_pos(t)
    ps = ring_px(t)
    up = up_hint(t)
    f = (FW / 2.0) / math.tan(math.radians(HFOV) / 2.0)
    tgt = ring.copy()
    for _ in range(60):
        cam = Camera(pos, tgt, up=up, hfov=HFOV)
        u, v, z = cam.project(ring, FW, FH)
        eu, ev = ps[0] - float(u), ps[1] - float(v)
        if abs(eu) + abs(ev) < 0.005:
            break
        D = float(np.linalg.norm(tgt - pos))
        tgt = tgt - cam.R[0] * eu * D / f + cam.R[1] * ev * D / f
    c = Camera(pos, tgt, up=up, hfov=HFOV, focus=float(np.linalg.norm(ring - pos)), aperture=0.0)
    _CAM[key] = c
    return c


def dark(t):
    """out of the black: the clouds come up out of the dark as the band turns into the moonlight"""
    return _ss(T_FALL + 4, T_FALL + 44, t)


def glint(t):
    return _ss(T_FALL, T_FALL + 8, t)


def moon_dir():
    """the moon sits off the FINAL frame's upper-left (as in RUN-C's R13b), 30 degrees up"""
    c = solve_cam(2839.0)
    d = -c.R[0] + c.R[1]
    d = np.array([d[0], 0.0, d[2]])
    d = d / np.linalg.norm(d)
    e = math.radians(30.0)
    return np.array([d[0] * math.cos(e), math.sin(e), d[2] * math.cos(e)])


# ============================================================ clouds ===
# Both layers are ray-marched volumes (a slab of cloud, not a painted plane): the sea's tops are billowed in 3-D
# (|noise| octaves: rounded bulges with creases between), lit by the moon with a shadow march and a multiple-
# scattering softening, brighter toward the tops (sky light); the wisps are a thin, eroded, sparse slab.

@njit(fastmath=True, cache=True, inline='always')
def _ssj(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(fastmath=True, cache=True, inline='always')
def _hash01(i, j):
    h = (i * 374761393 + j * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


@njit(fastmath=True, cache=True, inline='always')
def _sea_top(x, z):
    """the sea's slow heaps and hollows: the height of its top above DECK_Y, before the billows"""
    u = x * 3.85e-4
    v = z * 3.85e-4
    return 150.0 + 230.0 * (0.62 * perlin3(u, v, 1.3) + 0.3 * perlin3(u * 2.07 + 5.1, v * 2.07 - 3.3, 4.2)
                            + 0.14 * perlin3(u * 4.3 - 1.7, v * 4.3 + 8.2, 7.7))


@njit(fastmath=True, cache=True, inline='always')
def _billow(x, y, z, noct):
    """billows on the tops in 3-D: |noise| octaves (0 .. ~1.3)"""
    f = 1.0 / 420.0
    a = 1.0
    s = 0.0
    for o in range(noct):
        s += a * abs(perlin3(x * f + 3.1 * o, y * f * 1.25 - 1.7 * o, z * f + 5.3 * o))
        a *= 0.5
        f *= 2.02
    return s


@njit(fastmath=True, cache=True, inline='always')
def _sea_rho(x, hy, z, noct):
    """the sea's density (0..1) at (x, height above DECK_Y, z), and its local surface height"""
    top = _sea_top(x, z)
    if hy > top + 170.0:
        return 0.0, top + 170.0
    b = _billow(x, hy, z, noct)
    surf = top + 175.0 * (b - 0.33)
    return min(max((surf - hy) / 45.0, 0.0), 1.0), surf


@njit(parallel=True, fastmath=True, cache=True)
def _sea_vol(Dirs, cp, pr, L, out_rgb):
    """ray-march the sea of cloud. pr: 0 DECK_Y, 1 sigma, 2 moon E, 3 sky E, 4 step, 5 octaves, 6 haze distance,
    7 phase (moon), 8 frame seed; L: the direction to the moon"""
    H, W, _ = Dirs.shape
    dy0 = pr[0]
    sig = pr[1]
    noct = int(pr[5])
    for j in prange(H):
        for i in range(W):
            dx = Dirs[j, i, 0]
            dy = Dirs[j, i, 1]
            dz = Dirs[j, i, 2]
            hz_r = 0.010
            hz_g = 0.014
            hz_b = 0.026
            if dy > -1e-3:
                out_rgb[j, i, 0] = hz_r
                out_rgb[j, i, 1] = hz_g
                out_rgb[j, i, 2] = hz_b
                continue
            s = (dy0 + 560.0 - cp[1]) / dy
            if s < 0.0:
                s = 0.0
            step = pr[4]
            s += step * _hash01(i + int(pr[8]) * 7919, j)
            T = 1.0
            Lr = 0.0
            Lg = 0.0
            Lb = 0.0
            for k in range(220):
                px = cp[0] + dx * s
                hy = cp[1] + dy * s - dy0
                pz = cp[2] + dz * s
                if hy < -250.0:
                    break
                rho, surf = _sea_rho(px, hy, pz, noct)
                if rho > 0.001:
                    # the moon: a short shadow march (fewer octaves)
                    tl = 0.0
                    qs = 10.0
                    ls = 16.0
                    for m in range(5):
                        rq, _ = _sea_rho(px + L[0] * qs, hy + L[1] * qs, pz + L[2] * qs, 2)
                        tl += rq * ls
                        qs += ls
                        ls *= 2.0
                    tau = sig * tl
                    em = (math.exp(-tau) + 0.45 * math.exp(-0.3 * tau) + 0.2 * math.exp(-0.1 * tau)) / 1.65
                    hf = min(max((hy + 250.0) / (surf + 250.0), 0.0), 1.0)
                    amb = pr[3] * (0.2 + 0.8 * hf * hf)
                    moon = pr[2] * em * pr[7]
                    a = 1.0 - math.exp(-sig * rho * step)
                    Lr += T * a * (moon * 0.58 + amb * 0.3)
                    Lg += T * a * (moon * 0.7 + amb * 0.45)
                    Lb += T * a * (moon * 0.95 + amb * 1.0)
                    T *= 1.0 - a
                    if T < 0.015:
                        break
                s += step
            # what little shows through is the dark below; the long air between is a cold haze
            Lr += T * 0.004
            Lg += T * 0.005
            Lb += T * 0.009
            hz = 1.0 - math.exp(-s / pr[6])
            out_rgb[j, i, 0] = Lr * (1.0 - hz) + hz * hz_r
            out_rgb[j, i, 1] = Lg * (1.0 - hz) + hz * hz_g
            out_rgb[j, i, 2] = Lb * (1.0 - hz) + hz * hz_b


@njit(fastmath=True, cache=True, inline='always')
def _puff_rho(x, y, z, pr):
    """small clouds high over the sea (sparse), one kept where the band goes through. pr: 0 layer mid y,
    1 coverage bias, 2 half-thickness, 3-6 the tear (segment x0 z0 x1 z1), 7 tear radius, 8 tear depth,
    14-15 the crossing (x, z)"""
    hy = (y - pr[0]) / pr[2]
    if hy <= -1.0 or hy >= 1.0:
        return 0.0
    u = x / 420.0
    v = z / 420.0
    c = 0.65 * perlin3(u + 5.3, v - 2.9, 3.1) + 0.35 * perlin3(u * 2.1 - 1.2, v * 2.1 + 4.4, 6.6) + pr[1]
    ddx = (x - pr[14]) / 150.0
    ddz = (z - pr[15]) / 150.0
    c += 0.62 * math.exp(-(ddx * ddx + ddz * ddz))
    if c < 0.0:
        return 0.0
    prof = (1.0 - hy * hy) if hy > 0.0 else (1.0 - hy * hy * hy * hy)       # domed tops, flatter bases
    b = 0.6 * abs(perlin3(x / 95.0, y / 80.0, z / 95.0)) + 0.4 * abs(perlin3(x / 41.0 + 3.0, y / 36.0, z / 41.0 - 2.0))
    d = _ssj(0.0, 0.16, c * 1.7 * prof - 0.42 * b)
    if d <= 0.0:
        return 0.0
    # the band punched through here (the hole stays open along the line we watch it down)
    ax = pr[5] - pr[3]
    az = pr[6] - pr[4]
    ll = ax * ax + az * az
    tt = ((x - pr[3]) * ax + (z - pr[4]) * az) / max(ll, 1e-6)
    tt = min(max(tt, 0.0), 1.0)
    qx = x - (pr[3] + ax * tt)
    qz = z - (pr[4] + az * tt)
    d *= 1.0 - pr[8] * math.exp(-(qx * qx + qz * qz) / (pr[7] * pr[7]))
    return d


@njit(parallel=True, fastmath=True, cache=True)
def _puff_vol(Dirs, cp, pr, L, rp, out_rgb, out_a):
    """ray-march the small high clouds: moonlit with a short shadow march, the band's warmth inside the one it goes
    through. pr: 0-8, 14-15 as _puff_rho; 9 sigma, 10 moon E, 11 warm E, 12 warm radius, 13 frame seed;
    L: to the moon; rp: the band (or where it went through)"""
    H, W, _ = Dirs.shape
    for j in prange(H):
        for i in range(W):
            out_rgb[j, i, 0] = 0.0
            out_rgb[j, i, 1] = 0.0
            out_rgb[j, i, 2] = 0.0
            out_a[j, i] = 0.0
            dy = Dirs[j, i, 1]
            if dy > -1e-3:
                continue
            s0 = (pr[0] + pr[2] - cp[1]) / dy
            s1 = (pr[0] - pr[2] - cp[1]) / dy
            if s1 <= 0.0:
                continue
            if s0 < 0.0:
                s0 = 0.0
            n = 14
            ds = (s1 - s0) / n
            s = s0 + ds * _hash01(i + int(pr[13]) * 131, j + 17)
            T = 1.0
            Lr = 0.0
            Lg = 0.0
            Lb = 0.0
            for k in range(n):
                px = cp[0] + Dirs[j, i, 0] * s
                py = cp[1] + dy * s
                pz = cp[2] + Dirs[j, i, 2] * s
                rho = _puff_rho(px, py, pz, pr)
                if rho > 0.001:
                    tl = 0.0
                    qs = 6.0
                    ls = 10.0
                    for m in range(4):
                        tl += _puff_rho(px + L[0] * qs, py + L[1] * qs, pz + L[2] * qs, pr) * ls
                        qs += ls
                        ls *= 2.0
                    tau = pr[9] * tl
                    em = (math.exp(-tau) + 0.45 * math.exp(-0.3 * tau) + 0.2 * math.exp(-0.1 * tau)) / 1.65
                    hf = 0.5 + 0.5 * (py - pr[0]) / pr[2]
                    r2 = ((px - rp[0]) ** 2 + (py - rp[1]) ** 2 + (pz - rp[2]) ** 2) / (pr[12] * pr[12])
                    wm = pr[11] * (math.exp(-r2) + 0.1 / (1.0 + 3.0 * r2))
                    moon = pr[10] * em
                    amb = 0.1 * (0.3 + 0.7 * hf)
                    a = 1.0 - math.exp(-pr[9] * rho * ds)
                    Lr += T * a * (moon * 0.58 + amb * 0.3 + wm * 1.0)
                    Lg += T * a * (moon * 0.7 + amb * 0.45 + wm * 0.5)
                    Lb += T * a * (moon * 0.95 + amb * 1.0 + wm * 0.16)
                    T *= 1.0 - a
                    if T < 0.01:
                        break
                s += ds
            out_rgb[j, i, 0] = Lr
            out_rgb[j, i, 1] = Lg
            out_rgb[j, i, 2] = Lb
            out_a[j, i] = 1.0 - T


def _rays(cam, W, H):
    f = cam.f_px(W)
    u = (np.arange(W, dtype=np.float64) - (W - 1) / 2.0) / f
    v = -(np.arange(H, dtype=np.float64) - (H - 1) / 2.0) / f
    R = cam.R
    d = R[0][None, None, :] * u[None, :, None] + R[1][None, None, :] * v[:, None, None] + R[2][None, None, :]
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return d


def _plane_hit(cam, Dirs, y):
    s = (y - cam.pos[1]) / np.minimum(Dirs[..., 1], -1e-6)
    return (cam.pos[0] + Dirs[..., 0] * s).astype(np.float32), (cam.pos[2] + Dirs[..., 2] * s).astype(np.float32)


def _mblur(rgb, a, X, Y, y, cam0, cam1, cam, W, H, n_max=10):
    """shutter blur of a layer by reprojection of its plane (the camera turns fast during the tilt)"""
    P = np.stack([X.astype(np.float64), np.full(X.shape, y), Y.astype(np.float64)], -1)
    u0, v0, _ = cam0.project(P, W, H)
    u1, v1, _ = cam1.project(P, W, H)
    uu, vv, _ = cam.project(P, W, H)
    d0x, d0y = (u0 - uu).astype(np.float32), (v0 - vv).astype(np.float32)
    d1x, d1y = (u1 - uu).astype(np.float32), (v1 - vv).astype(np.float32)
    span = float(np.percentile(np.hypot(d1x - d0x, d1y - d0y), 95))
    if not np.isfinite(span) or span < 1.2:
        return rgb, a
    n = int(min(max(math.ceil(span / 2.5), 2), n_max))
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    acc = np.zeros_like(rgb)
    acca = np.zeros_like(a)
    for k in range(n):
        s = (k + 0.5) / n
        mx = gx - (d0x + (d1x - d0x) * s)
        my = gy - (d0y + (d1y - d0y) * s)
        acc += cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acca += cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n, acca / n


# =============================================================== shot ===

class FallShot:
    def __init__(self):
        self.moon = moon_dir()
        # the tear in the thin layer: from where the band went through to where our last look down crosses it
        self.tc = cross_t()
        r = ring_pos(self.tc)
        c_end = cam_pos(2839.0)
        r_end = ring_pos(2839.0)
        k = (WISP_Y - c_end[1]) / (r_end[1] - c_end[1])
        e = c_end + (r_end - c_end) * k
        self.tear = (float(r[0]), float(r[2]), float(e[0]), float(e[2]))

    def camera(self, t):
        return solve_cam(t)

    def render_opts(self, f):
        return dict(bokeh_pow=0.0, bokeh_cap=1.0, near=0.05)

    def emit(self, ctx):
        pass

    def env(self, t):
        e = RS.Env(above=(0.003, 0.0035, 0.006), horizon=(0.004, 0.0045, 0.007), below=(0.002, 0.0025, 0.004))
        g = glint(t)
        e.lobe(self.moon, C_MOON * 7.0 * g, 700.0)                 # the moon: a small hard body
        e.lobe(self.moon, C_MOON * 0.12 * g, 5.0)                  # its halo in the thin air
        e.lobe([0.0, -1.0, 0.0], C_MOON * 0.07 * dark(t), 1.4)     # the moonlit sea below
        e.lobe([0.2, 0.5, 0.3], [0.2, 0.12, 0.045], 0.5)           # warm fill: the gold stays gold, never olive
        return e

    def post(self, ctx, hdr):
        t = ctx.t
        W, H = ctx.fr.W, ctx.fr.H
        cam = ctx.cam
        dk = dark(t)
        rpos = ring_pos(t)
        img = np.zeros((H, W, 3), np.float32)
        f = cam.f_px(W)
        seed = float(int(t) % 997)
        Dirs = None
        # ---- the sea of cloud, far below
        if dk > 0.0:
            Dirs = _rays(cam, W, H)
            ph = self.phase(cam)
            pr = np.array([DECK_Y, 0.05, 0.64, 0.1, 15.0, 5.0, 60000.0, ph, seed], np.float64)
            drgb = np.empty((H, W, 3), np.float32)
            _sea_vol(Dirs, cam.pos.astype(np.float64), pr, self.moon.astype(np.float64), drgb)
            drgb = cv2.GaussianBlur(drgb, (0, 0), 0.7 * W / FW)            # the march's last grain (cloud is soft)
            X, Y = _plane_hit(cam, Dirs, DECK_Y + 250.0)
            drgb, _ = _mblur(drgb, np.ones((H, W), np.float32), X, Y, DECK_Y + 250.0, ctx.cam0, ctx.cam1, cam, W, H)
            img = drgb * dk
        # ---- small clouds high over the sea (always below us): the band goes down through one
        wl = None
        if dk > 0.0:
            warm = 1.4 * (_ss(self.tc - 5.0, self.tc, t) if t < self.tc else math.exp(-(t - self.tc) / 6.0))
            rp_w = rpos if t < self.tc else ring_pos(self.tc)
            cx = ring_pos(self.tc)
            pr = np.array([WISP_Y, -0.34, 60.0, self.tear[0], self.tear[1], self.tear[2], self.tear[3], 26.0,
                           0.97 * _ss(self.tc - 2.0, self.tc + 3.0, t), 0.03, 0.62, warm, 14.0, seed,
                           float(cx[0]), float(cx[2])], np.float64)
            wrgb = np.empty((H, W, 3), np.float32)
            wa = np.empty((H, W), np.float32)
            _puff_vol(Dirs, cam.pos.astype(np.float64), pr, self.moon.astype(np.float64), rp_w.astype(np.float64),
                      wrgb, wa)
            wrgb = cv2.GaussianBlur(wrgb, (0, 0), 0.6 * W / FW)
            wa = cv2.GaussianBlur(wa, (0, 0), 0.6 * W / FW)
            X, Y = _plane_hit(cam, Dirs, WISP_Y)
            wrgb, wa = _mblur(wrgb, wa, X, Y, WISP_Y, ctx.cam0, ctx.cam1, cam, W, H)
            wl = (wrgb * dk, wa * dk)
        # ---- the band (three instants across the shutter)
        acc = None
        for q in range(3):
            tq = ctx.t0 + (ctx.t1 - ctx.t0) * q / 2.0
            cq = self.camera(tq)
            st = RS.RingState()
            st.letters = (0.55 + 0.12 * math.sin(0.21 * tq)) * (0.4 + 0.6 * glint(tq))
            st.glow = 0.015
            rgb, a, d = RS.render(cq, W, H, ring_rot(tq), ring_pos(tq), RING_W, st, self.env(tq))
            if acc is None:
                acc = [rgb, a]
            else:
                acc[0] += rgb
                acc[1] += a
        rrgb, ra = acc[0] / 3.0, acc[1] / 3.0
        # ---- the glint: once the band is too small to show its hole it reads as a warm star (RUN-C's star at
        #      2840: a ~3 px warm-white core in a soft warm halo)
        u, v, z = cam.project(rpos, W, H)
        diam = DIAM * f / max(float(z), 1e-3) * (FW / W)
        kg = _ss(15.0, 7.0, diam) * glint(t)
        if kg > 0.0 and z > 0:
            sc = W / FW
            gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            r2 = ((gx - float(u)) ** 2 + (gy - float(v)) ** 2) / (sc * sc)
            core = np.exp(-r2 / (2.0 * 0.85 ** 2))
            halo = np.exp(-r2 / (2.0 * 2.6 ** 2))
            tw = 1.0 + 0.08 * math.sin(1.7 * t) * math.sin(0.61 * t)
            gl = (core[..., None] * np.array([2.7, 2.35, 1.6], np.float32)
                  + halo[..., None] * np.array([0.34, 0.2, 0.07], np.float32)) * (kg * tw)
            rrgb = rrgb + gl.astype(np.float32)
        # ---- composite, far to near
        k_over = float(np.clip((WISP_Y + 60.0 - rpos[1]) / 120.0, 0.0, 1.0))    # how much of the layer is over it
        if wl is not None:
            img = img * (1.0 - wl[1][..., None] * (1.0 - k_over)) + wl[0] * (1.0 - k_over)
        img = img * (1.0 - ra[..., None]) + rrgb
        if wl is not None:
            img = img * (1.0 - wl[1][..., None] * k_over) + wl[0] * k_over
        return hdr + img

    def phase(self, cam):
        """the moon's phase for the clouds seen from this camera (mostly back-scatter from up here)"""
        c = float(np.dot(self.moon, cam.R[2]))
        hg = lambda g: (1.0 - g * g) / (1.0 + g * g - 2.0 * g * c) ** 1.5
        return 0.6 * hg(0.35) + 0.4 * hg(-0.25)

    def finish_opts(self, f):
        return dict(exposure=1.0, bloom_strength=0.06, bloom_threshold=1.2, streak_strength=0.0, vignette_amount=0.25)
