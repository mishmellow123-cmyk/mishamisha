"""PAGES-C (28 Sep): smoke off a burning page (C4-C5, C8-C9), for the book layer.

Thin wisps rise from the ember front. Every wisp is the lineage of one emitter on the front: a puff is born on a
fixed clock (every DT seconds, from the emitter's own phase) at the front's position AT ITS BIRTH, and rises, leans
downwind and grows turbulent with age. At any frame the live puffs of one emitter, joined in birth order, are one
filament from the edge up into the dark, so the smoke is a pure function of time (no state, no pops between farm
chunks). The filaments are splatted as a density field and laid OVER the premultiplied book layer:
    rgb' = a * colour + (1 - a) * rgb,   matte' = 1 - (1 - a) * (1 - matte),   a = amax * (1 - exp(-density))

Emitters: 'radial' (a burn opening from a point: BURN.front_r along fixed angles), 'sweep' (an ember edge crossing
the page: BURN.sweep_pt at fixed lateral positions), and 'pre' (the paper smoking over a hot spot before it burns
through: a small disc round the burn's origin, rising with the browning's heat). Burn blocks are the book's v2
blocks, evaluated in the post's world frame (u = x, v = PH / 2 - y).
"""
import math

import numpy as np
from numba import njit, prange

import burn as BURN
import book as B
from noise import gnoise

DT = 1.0 / 48.0         # puff clock (s)
LIFE = 2.2              # a puff's life (s)


@njit(cache=True)
def _hash(k, s):
    h = (k * 374761393 + s * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / 16777216.0


@njit(cache=True)
def _puffs(bf, mode, t, M, PH, P, ck, pre_r, out):
    """Fill out[k, n] = (x, y, z, width_cm, density) for emitter k, puff n (n = 0 youngest). density 0 = no puff."""
    NA = out.shape[1]
    for k in prange(M):
        ph = _hash(k, 7) * DT
        ang = 2.0 * math.pi * (k + 0.5 * _hash(k, 11)) / M
        lat = -14.0 + 40.0 * (k + 0.5 * _hash(k, 13)) / M
        pr = pre_r * math.sqrt(_hash(k, 17))
        n0 = math.floor((t - ph) / DT)
        for n in range(NA):
            tau = ph + (n0 - n) * DT                 # this puff's birth time
            age = t - tau
            out[k, n, 4] = 0.0
            if age < 0.0 or age > LIFE:
                continue
            strength = 0.0
            u = 0.0
            v = 0.0
            if mode == 0:                            # radial: on the front, along a fixed angle
                r = BURN.front_r(bf, ang, tau)
                if r > 0.0:
                    r2 = BURN.front_r(bf, ang, tau + 0.05)
                    sp = max(r2 - r, 0.0) / 0.05     # the front's speed there (cm/s)
                    strength = 0.3 + 0.7 * sp / (sp + 1.5)
                    u = bf[1] + math.cos(ang) * r
                    v = bf[2] + math.sin(ang) * r * bf[15]
                elif pre_r > 0.0:                    # before it opens: the paper smokes over the hot spot
                    dt = tau - bf[3]
                    heat = min(max((dt + bf[13]) / max(bf[13], 1e-3), 0.0), 1.0)
                    strength = 0.55 * heat * heat
                    u = bf[1] + math.cos(ang) * pr
                    v = bf[2] + math.sin(ang) * pr * bf[15]
            else:                                    # sweep: on the moving edge, at a fixed lateral position
                x = (tau - bf[3]) / max(bf[13], 1e-3)
                if x > 0.0:
                    u, v = BURN.sweep_pt(bf, lat, tau)
                    strength = 1.0 if x < 1.0 else max(0.0, 1.0 - (x - 1.0) * bf[13] / 0.6)
            if strength <= 0.0:
                continue
            wx = u
            wy = 0.5 * PH - v
            z0, _m = B.height(wx, wy, P, ck)
            # rise, lean downwind, and turbulence that grows with age (each puff its own path, the whole
            # filament swaying slowly with the birth clock)
            A = 0.12 + 1.1 * age ** 1.35
            q1 = gnoise(k * 1.713 + age * 0.9, tau * 0.35, 431)
            q2 = gnoise(k * 1.713 + 5.1, age * 0.9 + tau * 0.35, 433)
            out[k, n, 0] = wx + 0.55 * age + A * q1
            out[k, n, 1] = wy + 0.25 * age + A * q2
            out[k, n, 2] = z0 + 0.05 + 0.9 * age + 1.1 * age * age
            out[k, n, 3] = 0.035 + 0.24 * age
            fl = 0.55 + 0.45 * (0.5 + gnoise(k * 0.37, tau * 3.0, 435))
            out[k, n, 4] = strength * fl * (1.0 - math.exp(-age / 0.12)) * math.exp(-age / 1.1)


@njit(cache=True, parallel=True)
def _splat(D, x0, y0, x1, y1, d0, d1, s0, s1):
    """Density along segments: a Gaussian cross-section whose width and density vary along each segment."""
    H, W = D.shape
    nb = 64
    band = (H + nb - 1) // nb
    for bnd in prange(nb):
        r0 = bnd * band
        r1 = min(H, r0 + band)
        for k in range(len(x0)):
            m = 3.0 * max(s0[k], s1[k]) + 1.0
            if max(y0[k], y1[k]) + m < r0 or min(y0[k], y1[k]) - m > r1:
                continue
            xa = int(max(0.0, math.floor(min(x0[k], x1[k]) - m)))
            xb = int(min(W - 1.0, math.ceil(max(x0[k], x1[k]) + m)))
            ya = int(max(float(r0), math.floor(min(y0[k], y1[k]) - m)))
            yb = int(min(r1 - 1.0, math.ceil(max(y0[k], y1[k]) + m)))
            dx = x1[k] - x0[k]
            dy = y1[k] - y0[k]
            L2 = dx * dx + dy * dy
            for i in range(ya, yb + 1):
                for j in range(xa, xb + 1):
                    px = j + 0.5 - x0[k]
                    py = i + 0.5 - y0[k]
                    tt = 0.0
                    if L2 > 1e-9:
                        tt = min(max((px * dx + py * dy) / L2, 0.0), 1.0)
                    qx = px - tt * dx
                    qy = py - tt * dy
                    s = s0[k] + (s1[k] - s0[k]) * tt
                    d = d0[k] + (d1[k] - d0[k]) * tt
                    D[i, j] += d * math.exp(-(qx * qx + qy * qy) / (2.0 * s * s))


def density(bf, mode, t, cam, bk, W, H, M=72, pre_r=0.0):
    """The smoke's density field (H x W) at time t for a v2 burn block (mode 'radial' or 'sweep')."""
    NA = int(LIFE / DT) + 2
    out = np.zeros((M, NA, 5), np.float64)
    _puffs(np.asarray(bf, np.float64), 0 if mode == 'radial' else 1, float(t), M, float(bk.PH), bk.params, bk.ck,
           float(pre_r), out)
    a = out[:, :-1]
    b = out[:, 1:]
    ok = (a[..., 4] > 0) & (b[..., 4] > 0)
    if not ok.any():
        return np.zeros((H, W), np.float32)
    A = a[ok]
    Bq = b[ok]
    pa, za = cam.project(A[:, :3])
    pb, zb = cam.project(Bq[:, :3])
    good = (za > 1.0) & (zb > 1.0)
    # a filament whose two puffs landed far apart (the front jumped) is not drawn as one long streak
    far = np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1]) > 0.08 * W
    good &= ~far
    sa = np.clip(A[:, 3] * cam.F / za, 0.6, 60.0)
    sb = np.clip(Bq[:, 3] * cam.F / zb, 0.6, 60.0)
    # density per unit of screen length: a thin wisp reads as a line, a wide one is thinner per pixel
    da = A[:, 4] * 0.9 / np.sqrt(sa)
    db = Bq[:, 4] * 0.9 / np.sqrt(sb)
    D = np.zeros((H, W), np.float32)
    g = np.where(good)[0]
    if len(g):
        _splat(D, pa[g, 0].astype(np.float64), pa[g, 1].astype(np.float64), pb[g, 0].astype(np.float64),
               pb[g, 1].astype(np.float64), da[g].astype(np.float64), db[g].astype(np.float64),
               sa[g].astype(np.float64), sb[g].astype(np.float64))
    return D


def over(hdr, alpha, D, colour, amax=0.5, k=1.0):
    """Lay the smoke over the premultiplied layer (in place)."""
    a = (amax * (1.0 - np.exp(-k * D))).astype(np.float32)
    c = np.asarray(colour, np.float32)
    hdr *= (1.0 - a)[..., None]
    hdr += a[..., None] * c
    alpha[:] = 1.0 - (1.0 - a) * (1.0 - alpha)
    return hdr, alpha
