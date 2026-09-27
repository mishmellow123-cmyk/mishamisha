"""EMBERS-C: C's fire -- ONE natural flame, gold and calm (E15's catch, C5's fire alone, E5-C and E11).

A screen-space procedural flame, drawn about its own axis (the root's screen point -> the projection of the point
straight above it), so it is the same flame from any camera and at any size (surface brightness is independent of
its size on screen). The first particle version read as a glowing drop; a real flame is a shape with a life:

* a teardrop body with a rounded root, tapering to ONE tip (never a fork: the tip region is a single tongue whose
  outline is torn by noise, so small flamelets break off it and die, but no two tongues stand side by side);
* domain-warped, upward-advected fbm: licks climb the flame, the outline wavers more toward the tip, the body carries
  brighter sheets and darker gaps;
* a heat field: a white-gold core low in the body, gold, orange at the rim, deep red at the torn tips;
* a slow lean in the draught; a soft glow round it.

Every frame is a pure function of (frame, root, axis, height): E15 (C 800-1039) and C6 (from 1040) draw the same
flame, so the handover is continuous by construction. `Sparks` adds a few calm sparks off the tip (3-D splats).
"""
import math

import numpy as np
from numba import njit, prange

from core import rng, vnoise, smoothstep, perlin3


@njit(fastmath=True, cache=True, inline='always')
def _fbm(x, y, z, octaves):
    a = 0.5
    s = 0.0
    for o in range(octaves):
        s += a * perlin3(x, y, z)
        x = x * 2.03 + 1.7
        y = y * 2.03 - 3.1
        z = z * 2.03 + 0.9
        a *= 0.5
    return s


@njit(fastmath=True, cache=True, inline='always')
def _ss(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(fastmath=True, cache=True, inline='always')
def _ramp(T, out):
    """temperature 0..1 -> colour (deep red, orange, gold, white-gold)"""
    if T < 0.25:
        k = T / 0.25
        out[0] = 0.72 * k
        out[1] = 0.17 * k
        out[2] = 0.035 * k
    elif T < 0.5:
        k = (T - 0.25) / 0.25
        out[0] = 0.72 + 0.28 * k
        out[1] = 0.17 + 0.31 * k
        out[2] = 0.035 + 0.085 * k
    elif T < 0.78:
        k = (T - 0.5) / 0.28
        out[0] = 1.0
        out[1] = 0.48 + 0.3 * k
        out[2] = 0.12 + 0.22 * k
    else:
        k = (T - 0.78) / 0.22
        if k > 1.0:
            k = 1.0
        out[0] = 1.0
        out[1] = 0.78 + 0.16 * k
        out[2] = 0.34 + 0.46 * k


@njit(parallel=True, fastmath=True, cache=True)
def _flame(out, VIS, x0, y0, ax, ay, H, t, calm, bright, lean, seed, xlo, xhi, ylo, yhi, glow):
    """draw one flame into out (h, w, 3) float32 (additive), within [xlo, xhi) x [ylo, yhi); VIS (h, w): visibility"""
    px = -ay
    py = ax
    vs = 0.052 + 0.02 * (1.0 - calm)           # how fast the licks climb (flame heights per frame)
    rise = t * vs
    wig = 1.25 - 0.3 * calm
    col = np.zeros(3)
    for iy in prange(ylo, yhi):
        c = np.zeros(3)
        for ix in range(xlo, xhi):
            vis = VIS[iy, ix]
            if vis <= 0.0:
                continue
            dx = ix - x0
            dy = iy - y0
            s = (dx * ax + dy * ay) / H
            w = (dx * px + dy * py) / H
            # the soft glow round the body (drawn wherever we look)
            gs = s - 0.33
            g2 = (gs * gs) / 0.16 + (w * w) / 0.07
            gl = glow * math.exp(-g2 * 1.4)
            if s < -0.3 or s > 1.7 or w > 0.75 or w < -0.75:
                if gl > 1e-4:
                    out[iy, ix, 0] += 1.0 * gl * bright * vis
                    out[iy, ix, 1] += 0.55 * gl * bright * vis
                    out[iy, ix, 2] += 0.2 * gl * bright * vis
                continue
            sc = s if s > 0.0 else 0.0
            # large licks and the draught
            n1 = _fbm(w * 2.1 + seed, s * 1.7 - rise, t * 0.021 + seed * 0.37, 3)
            n2 = _fbm(w * 5.3 - seed, s * 4.2 - rise * 1.7, t * 0.047 + 11.3, 3)
            amp = (0.018 + 0.19 * sc ** 1.6) * wig
            ww = w - lean * sc * sc + amp * n1 * 1.7
            ss = s + 0.11 * n2 * sc
            # the body: rounded root, widest low, one tapering tip
            if ss < -0.11:
                hw = 0.0
            else:
                a = (ss + 0.11) / 0.32
                if a > 1.0:
                    a = 1.0
                b = 1.0 - ss
                if b < 0.0:
                    b = 0.0
                hw = 0.205 * math.sqrt(a) * b ** 0.82
            edge = hw * (1.0 + 0.3 * n2)
            aw = abs(ww)
            dens = 0.0
            if edge > 1e-4 and aw < edge:
                dens = _ss(edge, edge * 0.42, aw)
            tip = 1.0 - _ss(0.58, 1.06, ss + 0.3 * n1 + 0.08 * n2)
            root = _ss(-0.14, -0.03, ss)
            dens = dens * tip * root
            if dens > 1e-4:
                core = 0.0
                if hw > 1e-4:
                    q = ww / (0.5 * hw + 1e-3)
                    om = 1.0 - sc
                    if om < 0.0:
                        om = 0.0
                    core = math.exp(-q * q) * om ** 1.6 * _ss(-0.1, 0.12, ss)
                tex = 0.72 + 0.62 * n2
                heat = dens * (1.0 - 0.52 * sc) * tex
                if heat < 0.0:
                    heat = 0.0
                T = 0.18 + 0.62 * heat + 0.34 * core - 0.2 * sc
                if T > 1.0:
                    T = 1.0
                if T < 0.0:
                    T = 0.0
                _ramp(T, c)
                E = bright * (1.15 * heat ** 1.25 + 2.6 * core * dens) * vis
                out[iy, ix, 0] += c[0] * E
                out[iy, ix, 1] += c[1] * E
                out[iy, ix, 2] += c[2] * E
            if gl > 1e-4:
                out[iy, ix, 0] += 1.0 * gl * bright * vis
                out[iy, ix, 1] += 0.55 * gl * bright * vis
                out[iy, ix, 2] += 0.2 * gl * bright * vis


def lean_at(t):
    """the draught's slow lean of the upper flame (flame heights, signed, screen-across)"""
    return 0.07 * math.sin(0.019 * t + 0.4) + 0.035 * math.sin(0.053 * t + 2.1)


def draw(hdr, root_px, tip_px, t, bright=1.0, calm=1.0, vis=None, glow=0.06, seed=3.7, scale=1.0):
    """draw the flame into hdr (H, W, 3) float32 in place. root_px / tip_px: FULL-RES screen points of the root and
    of the nominal tip (the flame's height); scale: the frame's render scale. vis: (H, W) visibility or None."""
    Hh, Ww = hdr.shape[:2]
    x0, y0 = root_px[0] * scale, root_px[1] * scale
    dx, dy = (tip_px[0] - root_px[0]) * scale, (tip_px[1] - root_px[1]) * scale
    Hpx = math.hypot(dx, dy)
    if Hpx < 1.5 or bright <= 0:
        return hdr
    ax, ay = dx / Hpx, dy / Hpx
    ext = 1.8 * Hpx
    xlo = int(max(math.floor(x0 - ext), 0))
    xhi = int(min(math.ceil(x0 + ext), Ww))
    ylo = int(max(math.floor(y0 - ext), 0))
    yhi = int(min(math.ceil(y0 + ext), Hh))
    if xhi <= xlo or yhi <= ylo:
        return hdr
    if vis is None:
        vis = np.ones((Hh, Ww), np.float32)
    _flame(hdr, vis, float(x0), float(y0), float(ax), float(ay), float(Hpx), float(t), float(calm), float(bright),
           float(lean_at(t)), float(seed), xlo, xhi, ylo, yhi, float(glow))
    return hdr


class Sparks:
    """a few calm sparks off the tip, short and orange, curving as they rise and die (3-D splats)"""

    def __init__(self, seed=802, n=320):
        r = rng(seed)
        self.ph = r.random(n)
        self.v = r.uniform(0.007, 0.012, n)
        self.a = r.uniform(0, 2 * np.pi, n)
        self.d = r.normal(0, 1, (n, 2))
        self.E = r.lognormal(0, 0.6, n)

    def emit(self, ctx, root, height, bright=1.0, amount=1.0):
        """root: world point of the flame's root; height: the flame's height (world units)"""
        if amount <= 0 or height <= 0:
            return
        from look import blackbody

        def pos(t):
            k = (self.ph + self.v * t) % 1.0
            y = 0.9 * height + k * 1.5 * height
            spr = 0.02 * height + 0.22 * height * k
            x = self.d[:, 0] * spr + 0.05 * height * np.sin(6.0 * k + self.a)
            z = self.d[:, 1] * spr
            P = np.asarray(root)[None, :] + np.stack([x, y, z], 1)
            P = P + vnoise(P * (1.0 / max(height, 1e-3)) + np.array([0, -0.02 * t, 0]), 1.3, (5.0, 1.0, 2.0), 1) \
                * (0.12 * height * k)[:, None]
            return P, k
        P0, k0 = pos(ctx.t0)
        P1, k1 = pos(ctx.t1)
        ok = k1 >= k0
        e = self.E * (1 - k1) ** 1.6 * 6.0 * amount * bright * ok * smoothstep(0.0, 0.08, k1)
        col = blackbody(np.clip(0.72 - 0.35 * k1, 0.3, 1.0))
        ctx.fr.splat(P0, P1, 0.004 * height, e, col, ctx.cam0, ctx.cam1)
