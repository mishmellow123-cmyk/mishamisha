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
G_RING = 0.084                                       # the band keeps falling and gathers speed
PS0 = (640.0, 262.0)                                 # the band on screen while we fall together (lead room down-right)
PS1 = (220.0, 17.0)                                  # the corner it settles into as we tilt after it
PS2 = (240.0, 28.0)                                  # 2839: just behind RUN-C's first star position (248, 31)
T_S0, T_S1 = 2762.0, 2833.0
PHI = math.radians(-60.0)                            # the frame's slow roll during the tilt (so it sinks down-right)
C_MOON = np.array([0.62, 0.72, 0.9])
DECK_Y = -2700.0                                     # the sea of cloud, far below
WISP_Y = -600.0                                      # the thin high layer the band goes through (~2800)


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
    """how far the band has fallen away below us since we let it go"""
    y = t - T_LET
    if y <= 0.0:
        return 0.0
    k = min(y, BRAKE)
    return 0.5 * G_RING * y * y + V0 * (k * k / (2.0 * BRAKE) + (y - k))


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

@njit(fastmath=True, cache=True, inline='always')
def _ssj(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(fastmath=True, cache=True, inline='always')
def _sea_h(u, v, nob):
    """the sea's top height at (u, v) (units of the biggest billow's wavelength). Round billows with sharp creases
    (|noise| octaves, each turned ~37 deg), on slow swells; nob = octaves to use (fractional: the last fades)"""
    m = 0.62 * perlin3(u * 0.21 + 7.1, v * 0.21 - 3.3, 1.9) + 0.38 * perlin3(u * 0.47 - 2.2, v * 0.47 + 8.4, 4.4)
    h = 0.55 * m
    a = 1.0
    uu = u
    vv = v
    for o in range(6):
        w = nob - o
        if w <= 0.0:
            break
        if w > 1.0:
            w = 1.0
        n = abs(perlin3(uu + 3.7 * o, vv - 1.3 * o, 0.61 * o + 2.0))
        if o == 0:
            n = math.sqrt(n)                     # the big domes: rounder tops, the creases stay sharp
        h += w * a * n
        a *= 0.47
        u2 = 0.8 * uu - 0.6 * vv
        vv = (0.6 * uu + 0.8 * vv) * 2.03
        uu = u2 * 2.03
    return h


@njit(parallel=True, fastmath=True, cache=True)
def _deck(X, Y, ok, DIST, FPW, pr, L, out_rgb):
    """the sea of cloud from above. pr: 0 freq (1/wavelength), 1 relief, 2 moon E, 3 sky E, 4 haze distance,
    5 gap level, 6 moon tan(elevation); L: the moon's horizontal direction (lx, lz)"""
    H, W = X.shape
    f = pr[0]
    A = pr[1]
    tanE = pr[6]
    lx = L[0]
    lz = L[1]
    for j in prange(H):
        for i in range(W):
            if not ok[j, i]:
                out_rgb[j, i, 0] = 0.0
                out_rgb[j, i, 1] = 0.0
                out_rgb[j, i, 2] = 0.0
                continue
            u = X[j, i] * f
            v = Y[j, i] * f
            fp = FPW[j, i] * f                                   # the pixel's footprint in wavelengths
            # octaves: keep those whose wavelength spans > ~3 px
            nob = math.log(max(0.5 / (3.0 * fp), 1e-6)) / math.log(2.03) + 1.0
            if nob > 6.0:
                nob = 6.0
            if nob < 0.6:
                nob = 0.6
            h0 = _sea_h(u, v, nob)
            e = max(fp, 0.004)
            hx = _sea_h(u + e, v, nob)
            hy = _sea_h(u, v + e, nob)
            gx = A * (hx - h0) / e
            gy = A * (hy - h0) / e
            nn = 1.0 / math.sqrt(gx * gx + gy * gy + 1.0)
            nx = -gx * nn
            ny = nn
            nz = -gy * nn
            # the moon: wrapped diffuse (soft, as cloud is), and the billows' shadows (march toward the moon)
            ly = tanE / math.sqrt(1.0 + tanE * tanE)
            lh = 1.0 / math.sqrt(1.0 + tanE * tanE)
            ndl = nx * lx * lh + ny * ly + nz * lz * lh
            diff = (ndl + 0.35) / 1.35
            if diff < 0.0:
                diff = 0.0
            sh = 1.0
            nsb = min(nob, 3.0)
            z0 = A * _sea_h(u, v, nsb) + 0.004
            s = 0.045
            for k in range(6):
                hs = A * _sea_h(u + lx * s, v + lz * s, nsb)
                occ = (hs - (z0 + s * tanE)) / 0.03
                if occ > 0.0:
                    sh *= 1.0 - 0.8 * min(occ, 1.0)
                s *= 1.62
            # creases and the low places between the billows hold the dark
            cav = _ssj(pr[5] - 0.25, pr[5] + 0.35, h0)
            hi = _ssj(pr[5], pr[5] + 1.0, h0)
            moon = pr[2] * diff * sh * (0.55 + 0.45 * hi) * (0.35 + 0.65 * cav)
            sky = pr[3] * (0.45 + 0.55 * ny) * (0.25 + 0.75 * cav)
            r = moon * 0.62 + sky * 0.5
            g = moon * 0.72 + sky * 0.62
            b = moon * 0.9 + sky * 1.0
            # the air between: a cold haze thickens with distance
            hz = 1.0 - math.exp(-DIST[j, i] / pr[4])
            out_rgb[j, i, 0] = r * (1.0 - hz) + hz * 0.010
            out_rgb[j, i, 1] = g * (1.0 - hz) + hz * 0.014
            out_rgb[j, i, 2] = b * (1.0 - hz) + hz * 0.026


@njit(fastmath=True, cache=True, inline='always')
def _wisp_d(u, v, pr):
    n = c2._fbm(u + 5.3, v - 2.9, 3.1, 4, 2.03, 0.5)
    b = abs(perlin3(u * 2.3 + 1.1, v * 2.3 + 7.7, 0.4))
    d = _ssj(0.0, 0.32, n + pr[1] - 0.25 * b)
    # the band tore them open where it went through (and they part slowly)
    dx = u / pr[0] - pr[4]
    dz = v / pr[0] - pr[5]
    d *= 1.0 - pr[7] * math.exp(-(dx * dx + dz * dz) / (pr[6] * pr[6]))
    return d


@njit(parallel=True, fastmath=True, cache=True)
def _wisps(X, Y, ok, pr, L, rp, out_rgb, out_a):
    """a sparse, thin, high layer seen from above. pr: 0 freq, 1 coverage bias, 2 sigma, 3 moon E,
    4-5 the tear (x, z), 6 tear radius, 7 tear depth, 8 the band's warmth E, 9 warmth radius, 10 plane y;
    L: moon (lx, lz) horizontal; rp: the band (x, y, z)"""
    H, W = X.shape
    f = pr[0]
    for j in prange(H):
        for i in range(W):
            out_rgb[j, i, 0] = 0.0
            out_rgb[j, i, 1] = 0.0
            out_rgb[j, i, 2] = 0.0
            out_a[j, i] = 0.0
            if not ok[j, i]:
                continue
            u = X[j, i] * f
            v = Y[j, i] * f
            d = _wisp_d(u, v, pr)
            if d < 1e-4:
                continue
            d2 = _wisp_d(u + L[0] * 0.05, v + L[1] * 0.05, pr)
            lit = min(max((d - d2) * 3.0 + 0.45, 0.0), 1.0)          # the moon-facing side of each wisp
            thick = 1.0 - math.exp(-2.2 * d)
            Lm = pr[3] * (0.3 + 0.7 * lit) * (0.35 + 0.65 * thick)
            # the band's warmth inside them
            dx = X[j, i] - rp[0]
            dy = pr[10] - rp[1]
            dz = Y[j, i] - rp[2]
            r2 = (dx * dx + dy * dy + dz * dz) / (pr[9] * pr[9])
            wm = pr[8] * (math.exp(-r2) + 0.18 / (1.0 + 3.0 * r2)) * (0.3 + 0.7 * thick)
            out_rgb[j, i, 0] = (Lm * 0.62 + wm * 1.0) * d
            out_rgb[j, i, 1] = (Lm * 0.72 + wm * 0.5) * d
            out_rgb[j, i, 2] = (Lm * 0.9 + wm * 0.16) * d
            out_a[j, i] = 1.0 - math.exp(-pr[2] * d)


def _plane(cam, W, H, y):
    X, Y, ok = c2.plane_coords(cam, W, H, np.array([0.0, y, 0.0]), np.array([1.0, 0.0, 0.0]),
                               np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), 1.0)
    return X, Y, ok


def _mblur(rgb, a, X, Y, y, cam0, cam1, cam, W, H, n_max=10):
    """shutter blur of a plane layer by reprojection (the camera turns fast during the tilt)"""
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
        mh = np.array([self.moon[0], self.moon[2]])
        mh = mh / np.linalg.norm(mh)
        tanE = self.moon[1] / math.hypot(self.moon[0], self.moon[2])
        # ---- the sea of cloud, far below
        if dk > 0.0:
            X, Y, ok = _plane(cam, W, H, DECK_Y)
            dx = X - cam.pos[0]
            dz = Y - cam.pos[2]
            dy = DECK_Y - cam.pos[1]
            dist = np.sqrt(dx * dx + dz * dz + dy * dy).astype(np.float32)
            cosi = np.abs(dy) / np.maximum(dist, 1e-3)
            fpw = (dist / (f * np.maximum(cosi, 0.12))).astype(np.float32)
            pr = np.array([1.0 / 520.0, 0.3, 0.46, 0.05, 9000.0, 0.62, tanE], np.float64)
            drgb = np.empty((H, W, 3), np.float32)
            _deck(X, Y, ok, dist, fpw, pr, mh, drgb)
            da = ok.astype(np.float32)
            drgb, da = _mblur(drgb, da, X, Y, DECK_Y, ctx.cam0, ctx.cam1, cam, W, H)
            img = drgb * dk
        # ---- the thin high layer (only while it is below us)
        wl = None
        if cam.pos[1] > WISP_Y + 1.0 and dk > 0.0:
            X, Y, ok = _plane(cam, W, H, WISP_Y)
            tear_t = T_LET + 40.0                      # ~ when the band goes through
            rt = ring_pos(tear_t)
            opened = _ss(tear_t - 2.0, tear_t + 5.0, t)
            warm = 2.2 * math.exp(-((rpos[1] - WISP_Y) / 38.0) ** 2) * glint(t)
            pr = np.array([1.0 / 150.0, -0.08, 1.4, 0.36, rt[0], rt[2], 34.0, 0.92 * opened, warm, 34.0, WISP_Y],
                          np.float64)
            wrgb = np.empty((H, W, 3), np.float32)
            wa = np.empty((H, W), np.float32)
            _wisps(X, Y, ok, pr, mh, rpos.astype(np.float64), wrgb, wa)
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
        # ---- the glint: once the band is small it reads as a warm star (RUN-C's star at 2840: ~3 px warm white
        #      core, a soft warm halo)
        u, v, z = cam.project(rpos, W, H)
        size = RING_W * f / max(float(z), 1e-3) * (FW / W)
        kg = _ss(16.0, 6.0, size) * glint(t)
        if kg > 0.0 and z > 0:
            sc = W / FW
            gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            r2 = ((gx - float(u)) ** 2 + (gy - float(v)) ** 2) / (sc * sc)
            core = np.exp(-r2 / (2.0 * 0.85 ** 2))
            halo = np.exp(-r2 / (2.0 * 2.6 ** 2))
            tw = 1.0 + 0.08 * math.sin(1.7 * t) * math.sin(0.61 * t)
            gl = (core[..., None] * np.array([2.6, 2.35, 1.9], np.float32)
                  + halo[..., None] * np.array([0.34, 0.2, 0.07], np.float32)) * (kg * tw)
            rrgb = rrgb + gl.astype(np.float32)
        # ---- composite, far to near
        above = rpos[1] > WISP_Y
        if wl is not None and above:
            img = img * (1.0 - wl[1][..., None]) + wl[0]
        img = img * (1.0 - ra[..., None]) + rrgb
        if wl is not None and not above:
            img = img * (1.0 - wl[1][..., None]) + wl[0]
        return hdr + img

    def finish_opts(self, f):
        return dict(exposure=1.0, bloom_strength=0.06, bloom_threshold=1.2, streak_strength=0.0, vignette_amount=0.25)
