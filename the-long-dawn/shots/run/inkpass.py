"""C · THE LIVING INK ILLUSTRATION: the ink pass (H5 revision). Turns a render_ink.py AOV set into a frame drawn in
the old book's manner: parchment fixed to the page; engraved hatching that FOLLOWS THE FORM (cross-contour lines
wrapping each peak, a diagonal crosshatch in the shadows); pen outlines whose weight follows the light (heavy in
shadow, thin and broken where lit, lost in a fire's light); the cloud sea in ruled contours and stipple; a sepia
wash on the macro shadow planes; the beacons hand-inked at the terrain's pen weight with a gold wash (no halo:
the catch blooms as gold wash on the rock the fire lights); at dawn (C24) a watercolour wash that spreads where
the light goes, an inked sun and inked smoke. Render at 2x and downsample (ink_final.py does).

TEMPORAL COHERENCE (the risk): every mark is a function of WORLD position, never of the screen.
  * Hatch lines are iso-lines of world-anchored fields (the height, and planes fixed to the shot's world axes),
    at NESTED intervals chosen by their own screen spacing. A line's rank comes from its birth level against the
    continuous scale, so as the camera moves lines only fade in between lines already there, never pop.
  * Dashes and pen pressure are world-anchored noise band-passed to the stroke's screen size.
  * Tone comes from the macro form (normals smoothed at a world footprint) and never flickers; firelight enters
    it steadily (a smooth falloff with distance from each fire).
  * Cloud contours are iso-height lines at nested intervals (every other line of a level is the level above).

  python inkpass.py --shot scroll --frames 168 --scale 0.5 --tag _t      # cache -> renders/run_c_tests/<shot>_t/
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ink_aov as IA        # noqa: E402
import run as RN            # noqa: E402
from mt.noise import gnoise2, fbm2, h01_3, smoothstep   # noqa: E402

look = IA.WD.CM.look
ROOT = IA.WD.CM.ROOT
OUT = os.path.join(ROOT, 'renders', 'run_c_tests')

# palette (sRGB 0-255, MAP's parchment and iron-gall ink)
PAPER = np.array([226, 204, 160])
PAPER_DARK = np.array([196, 160, 108])
INK = np.array([40, 27, 20])
INK_LIGHT = np.array([104, 72, 46])
WASH = np.array([150, 112, 72])
GOLD_HI = np.array([252, 226, 150])
GOLD = np.array([214, 164, 62])
GOLD_LO = np.array([128, 84, 26])
BOLE = np.array([150, 58, 36])


def s2l(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


S0 = 0.25              # finest stroke spacing (m); level L spacing = S0 * 2^L
LMAX = 15              # coarsest level (8 km cells)


# ------------------------------------------------------------ the nested seeds ---

@njit(inline='always', fastmath=True)
def _jit(M, I, J, salt):
    return 0.15 + 0.7 * h01_3(I, J, M, salt), 0.15 + 0.7 * h01_3(I, J, M + 101, salt)


@njit(inline='always', fastmath=True)
def seed_of(L, i, j, salt):
    """World (x, z) and birth level of the seed of cell (i, j) at level L (nested: a cell inherits its parent's
    seed if that seed lies inside it, else gets a new jittered one)."""
    sM = S0 * 2.0 ** LMAX
    sh = LMAX - L
    I = i >> sh
    J = j >> sh
    a, b = _jit(LMAX, I, J, salt)
    px = (I + a) * sM
    pz = (J + b) * sM
    birth = LMAX
    for M in range(LMAX - 1, L - 1, -1):
        sM = S0 * 2.0 ** M
        sh = M - L
        I = i >> sh
        J = j >> sh
        if math.floor(px / sM) == I and math.floor(pz / sM) == J:
            continue
        a, b = _jit(M, I, J, salt)
        px = (I + a) * sM
        pz = (J + b) * sM
        birth = M
    return px, pz, birth


# ------------------------------------------------------------ cross-contour hatching (H5) ---
# The engraver's line: parallel lines that FOLLOW THE FORM, not dashes down the fall line. Each family is the set of
# iso-lines of a world-anchored scalar field, spaced on the page by the field's own screen gradient:
#   F1  contours of the height (horizontal planes): lines that wrap every peak and curve with each face
#   F2  tilted planes (h cos b - psi sin b; psi = plan coordinate along the shot's fixed world direction): the
#       diagonal crosshatch, laid only in the shadows
# Intervals are nested (every other line of level L is a line of level L+1). A line's rank comes from its BIRTH
# level relative to the continuous scale, so coarse lines are drawn in the light tones and the in-between lines
# fill in as the tone darkens; as the camera moves, a line fades in or out, it never pops. Dashes and pressure
# come from world-anchored noise, band-passed to the stroke's screen size.

H0 = 0.5               # finest line interval (m of the field); level L interval = H0 * 2^L
D0 = 0.25              # finest dash-noise scale (m)


@njit(inline='always')
def _ctz(k):
    """Trailing zero bits of |k| (k != 0), capped at 8."""
    k = abs(k)
    n = 0
    while (k & 1) == 0 and n < 8:
        k >>= 1
        n += 1
    return n


@njit(inline='always', fastmath=True)
def _wnoise(x, z, mpx, Lpx, seed, oa, ob):
    """World-anchored noise whose features stay ~Lpx page pixels long: two octaves (4x apart), each fixed to the
    world, cross-faded by the continuous scale (mpx = metres per 1920-px at this point)."""
    lo = 0.5 * math.log2(max(mpx * Lpx / D0, 1e-6))
    o = math.floor(lo)
    b = lo - o
    s0 = D0 * 4.0 ** o
    s1 = 4.0 * s0
    n0 = gnoise2(x / s0 + oa, z / s0 + ob, seed)
    n1 = gnoise2(x / s1 + ob, z / s1 + oa, seed + 1)
    return (n0 * (1.0 - b) + n1 * b) / math.sqrt((1.0 - b) * (1.0 - b) + b * b)


@njit(inline='always', fastmath=True)
def _lines(phi, gx, gy, x, z, mpx, T, kpx, sp, seed, wmul, gapk, shift):
    """Coverage of one iso-line family at this pixel. phi: the field; (gx, gy): its screen gradient (field units
    per ss px); T: the family's tone; sp: the densest spacing (1920-px)."""
    g = math.sqrt(gx * gx + gy * gy)
    if g < 1e-9:
        return 0.0
    lam = math.log2(max(sp * kpx * g / H0, 1.0))
    L = int(math.floor(lam))
    dh = H0 * 2.0 ** L
    q = phi / dh + shift                            # shift: 0, or the boiling control's re-deal of positions
    k = int(math.floor(q + 0.5))
    dpx = abs(q - k) * dh / g                       # distance to the nearest line of level L (ss px)
    M = L + (8 if k == 0 else _ctz(k))              # the line's birth level
    c = M - lam                                     # coarseness relative to the current scale
    wl = min(max(1.0 + c, 0.0), 1.0)                # finer than the current level: fading out
    if wl <= 0.0:
        return 0.0
    lid = k * (1 << L)                              # the line's identity (its field value / H0), level-free
    r1 = h01_3(lid & 1048575, lid >> 20, seed, 5)
    r2 = h01_3(lid & 1048575, lid >> 20, seed, 7)
    rank = 0.92 * min(max(1.0 - c / 2.6, 0.0), 1.0) + 0.10 * (r1 - 0.5)
    pres = (T - rank) / 0.10
    if pres <= 0.0:
        return 0.0
    if pres > 1.0:
        pres = 1.0
    # dashes: gaps where the world-anchored noise is low (more gaps in the light), tapering the pen at each gap
    nd = _wnoise(x, z, mpx, 120.0, seed + 3, 37.1 * r1, 53.7 * r2)
    thr = gapk * (-0.70 + 0.35 * (1.0 - min(T, 1.0)))
    gap = smoothstep(thr, thr + 0.30, nd)
    npr = _wnoise(x, z, mpx, 28.0, seed + 7, 11.3 * r2, 71.9 * r1)
    w = kpx * wmul * (0.35 + 0.80 * min(T, 1.2)) * (0.85 + 0.30 * npr) * gap
    if w < 0.06 * kpx:
        return 0.0
    cv = 0.5 * w + 0.5 - dpx
    if cv <= 0.0:
        return 0.0
    if cv > 1.0:
        cv = 1.0
    if w < 1.0:
        cv *= w                                     # thin: fainter, never thinner than a pixel
    return cv * pres * wl * (0.85 + 0.15 * r2)


@njit(parallel=True, fastmath=True, cache=True)
def xhatch(A, T, G, hs, prm, out):
    """A: AOV (H, W, 16); T: darkness (H, W); G (H, W, 6): screen gradients (m per ss px) of h, psi and chi
    (psi, chi: plan coordinates along the shot's fixed right and forward directions).
    prm: kpx, sp1, sp2, cos b, sin b, rx, rz, salt, f, t2, kappa, fx, fz. out (H, W, 2): F1, F2 coverage.
    F1 = h - kappa chi: contours of planes rising away from the camera, so every ring is seen as if from a little
    above and wraps its peak (an arc under each summit), even from beacon height."""
    H, W = T.shape
    kpx = prm[0]
    sp1 = prm[1]
    sp2 = prm[2]
    cb = prm[3]
    sb = prm[4]
    rx = prm[5]
    rz = prm[6]
    so = int(prm[7])
    f = prm[8]
    t2 = prm[9]
    kap = prm[10]
    fx = prm[11]
    fz = prm[12]
    shf = 0.0 if so == 0 else 0.3 + 0.4 * ((so * 0.6180339887) % 1.0)     # the control moves every line
    for j in prange(H):
        for i in range(W):
            out[j, i, 0] = 0.0
            out[j, i, 1] = 0.0
            if int(A[j, i, 0] + 0.5) >= 2:
                continue
            Tt = T[j, i]
            if Tt <= 0.02:
                continue
            gx = G[j, i, 0]
            gy = G[j, i, 1]
            if not (gx == gx):
                continue
            x = A[j, i, 2]
            z = A[j, i, 4]
            h = hs[j, i]
            mpx = A[j, i, 1] / f * kpx                  # metres per 1920-px here
            chi = x * fx + z * fz
            ph1 = h - kap * chi
            T1 = Tt if Tt < prm[13] else prm[13] + 0.25 * (Tt - prm[13])     # contours stop densifying at mid-tone
            out[j, i, 0] = _lines(ph1, gx - kap * G[j, i, 4], gy - kap * G[j, i, 5], x, z, mpx, T1, kpx, sp1,
                                  11 + so, 1.0, 1.0, shf)
            if Tt > t2:
                psi = x * rx + z * rz
                ph = h * cb - psi * sb
                qx = gx * cb - G[j, i, 2] * sb
                qy = gy * cb - G[j, i, 3] * sb
                T2 = (Tt - t2) / (1.0 - t2) * 1.15
                out[j, i, 1] = _lines(ph, qx, qy, x, z, mpx, T2, kpx, sp2, 29 + so, 0.70, 0.8, shf)


@njit(parallel=True, fastmath=True, cache=True)
def macro_ndl(A, Lk, CR, t, f, kpx, K, Ks, out, hs):
    """n.l of the land's MACRO form: the height's normal smoothed at ~K page pixels (world-anchored functions of
    position, footprint by distance). The illustrator's planes of light and shadow, not every gully.
    hs: the height smoothed at ~Ks page pixels (the contour lines' field: clean lines, not pixel wiggles)."""
    H, W = out.shape
    for j in prange(H):
        for i in range(W):
            out[j, i] = 0.0
            surf = int(A[j, i, 0] + 0.5)
            hs[j, i] = A[j, i, 5]
            if surf >= 2:
                continue
            x = A[j, i, 2]
            z = A[j, i, 4]
            sc = A[j, i, 1] / f * kpx * K
            if sc < 1.0:
                sc = 1.0
            if sc > 600.0:
                sc = 600.0
            gx, gz = IA.grad_at(x, z, sc, CR, surf, t)
            nn = 1.0 / math.sqrt(gx * gx + gz * gz + 1.0)
            out[j, i] = (-gx * Lk[0] + Lk[1] - gz * Lk[2]) * nn
            hs[j, i] = IA.height_at(x, z, A[j, i, 1] / f * kpx * Ks, CR, surf, t)


def sgrad(F, D):
    """Screen gradient of a world-anchored field F (per px) that never differences across a depth break: at each
    pixel take the one-sided difference toward the neighbour whose depth is closer (the same surface)."""
    F = F.astype(np.float64)
    D = D.astype(np.float64)
    out = []
    for ax in (1, 0):
        Fp = np.roll(F, -1, axis=ax)
        Fm = np.roll(F, 1, axis=ax)
        Dp = np.roll(D, -1, axis=ax)
        Dm = np.roll(D, 1, axis=ax)
        jp = np.abs(Dp - D) / np.maximum(D, 1e-3)
        jm = np.abs(Dm - D) / np.maximum(D, 1e-3)
        g = np.where(jp <= jm, Fp - F, F - Fm)
        bad = np.minimum(jp, jm) > 0.08
        g[bad] = np.nan
        out.append(g)
    gx, gy = out
    return gx.astype(np.float32), gy.astype(np.float32)


SHOT_YAW = dict(scroll=22.0, reveal=170.0, illum=-19.0)     # each shot's fixed world direction for F2
XSP1 = 4.0             # F1's densest spacing (1920-px)
XSP2 = 4.5             # F2's
XB = math.radians(45.0)   # F2's tilt: diagonals rising to the right on a face
XT2 = 0.30              # F2 (the diagonal shadow hatching) is laid where the tone passes this
XT1MAX = 0.62           # F1 (contours) densify no further than this tone: the shadows are carried by F2
XKAP = 0.36             # F1's tilt toward the camera (tan 20 deg)
MACRO_K = 40.0          # the macro form's smoothing (page pixels)
HSMOOTH = 5.0           # the contour field's smoothing footprint (page pixels): 2.5 left a pixel-scale tremor
                        # on every line at 1:1 (the fractal crag), which read as a filter, not a burin


# ------------------------------------------------------------ cloud sea ---

@njit(parallel=True, fastmath=True, cache=True)
def cloud(A, Tc, gs, cam, prm, out):
    """Contours of the cloud's height at nested intervals + stipple in the shade. gs: |screen gradient of h|
    (m per ss px). out (H, W, 2): contour cov, stipple cov."""
    H, W = Tc.shape
    kpx = prm[0]
    csp = prm[3]            # contour spacing target (1920-px)
    dsp = prm[4]            # stipple spacing target (1920-px)
    f = cam[12]
    for j in prange(H):
        for i in range(W):
            out[j, i, 0] = 0.0
            out[j, i, 1] = 0.0
            if int(A[j, i, 0] + 0.5) != 2:
                continue
            h = A[j, i, 5]
            g = max(gs[j, i], 1e-4)
            # contour interval: every other line of level L is a line of level L+1
            H0 = 0.5
            lam = math.log2(max(csp * kpx * g / H0, 1.0))
            L = int(math.floor(lam))
            al = lam - L
            dh = H0 * 2.0 ** L
            q = h / dh
            k = math.floor(q + 0.5)
            dpx = abs(q - k) * dh / g
            odd = (int(k) & 1) == 1
            wl = 1.0 - al if odd else 1.0
            Tt = Tc[j, i]
            w = kpx * (0.6 + 0.9 * Tt)
            c = 0.5 * w + 0.5 - dpx
            if c > 0.0:
                c = min(c, 1.0) * min(w, 1.0) * wl * (0.55 + 0.45 * Tt) * smoothstep(0.06, 0.30, Tt)
                # broken lines: gaps anchored to the world
                br = fbm2(A[j, i, 2] / (dh * 9.0), A[j, i, 4] / (dh * 9.0), 2.0, 57)
                c *= smoothstep(-0.55, -0.25, br)
                out[j, i, 0] = c
            # stipple: nested world dots, denser in the troughs and shadows
            x = A[j, i, 2]
            z = A[j, i, 4]
            fpx = A[j, i, 1] / f
            ls = math.log2(max(dsp * kpx * fpx / S0, 1.0))
            Ls = int(math.floor(ls))
            als = ls - Ls
            if Ls > LMAX - 1:
                continue
            sL = S0 * 2.0 ** Ls
            ci = int(math.floor(x / sL))
            cj = int(math.floor(z / sL))
            best = 0.0
            for dj in range(-1, 2):
                for di in range(-1, 2):
                    px, pz, birth = seed_of(Ls, ci + di, cj + dj, 43 + int(prm[5]))
                    wl2 = 1.0 if birth > Ls else 1.0 - als
                    if wl2 <= 0.0:
                        continue
                    rank = h01_3(ci + di, cj + dj, Ls + 3, 47)
                    pres = (Tt * 1.2 - 0.25 - rank) / 0.12
                    if pres <= 0.0:
                        continue
                    ex = (x - px) / fpx
                    ez = (z - pz) / fpx
                    # dots are round on the page: distance in screen px (foreshortening ~ isotropic enough
                    # for dots of one pixel)
                    d = math.sqrt(ex * ex + ez * ez)
                    r = kpx * (0.55 + 0.35 * h01_3(ci + di, cj + dj, Ls + 5, 49))
                    c2 = r + 0.5 - d
                    if c2 > 0.0:
                        c2 = min(c2, 1.0) * min(pres, 1.0) * wl2
                        if c2 > best:
                            best = c2
            out[j, i, 1] = best


# ------------------------------------------------------------ the page ---

@njit(parallel=True, fastmath=True, cache=True)
def paper(H, W, kpx, out):
    """Parchment fixed to the page (1920-px units): warped tone, blotches, fibres, tooth. out (H, W, 2):
    tone t (0 light .. 1 dark), grain g."""
    for i in prange(H):
        for j in range(W):
            x = (j + 0.5) / kpx / 40.0
            y = (i + 0.5) / kpx / 40.0
            qx = fbm2(x * 0.06, y * 0.06, 3.0, 1177)
            qy = fbm2(x * 0.06 + 5.2, y * 0.06 + 1.3, 3.0, 1191)
            big = fbm2((x + 6.0 * qx) * 0.09, (y + 6.0 * qy) * 0.09, 4.0, 101)
            mid = fbm2(x * 0.5, y * 0.5, 4.0, 202)
            tt = 0.38 + 0.55 * big + 0.22 * mid
            g = 0.9 * fbm2(x * 2.2, y * 2.2, 4.0, 303)
            a = 3.0 * gnoise2(x * 0.05, y * 0.05, 409)
            ca = math.cos(a)
            sa = math.sin(a)
            u = (ca * x + sa * y) * 14.0
            v = (-sa * x + ca * y) * 1.6
            g += 0.30 * gnoise2(u, v, 404) + 0.20 * gnoise2(u * 2.3, v * 2.1, 406) + 0.55 * gnoise2(x * 30.0, y * 30.0, 405)
            out[i, j, 0] = tt
            out[i, j, 1] = g


# ------------------------------------------------------------ compose ---

def darkness(A, fire_steady, ndl_m=None):
    """Illustration value, decided like an illustrator: the side turned from the light is shadow (dense
    hatching), cast shadow adds a middle tone, bare rock a light texture, lit snow is bare paper.
    0 = paper, 1 = full hatching, >0.85 cross-hatching."""
    surf = A[..., 0]
    snow = np.clip(A[..., 9], 0, 1)
    ndl = A[..., 10]
    shd = np.clip(A[..., 11], 0, 1)
    sf = np.clip((0.40 - ndl) / 0.60, 0, 1)
    sf = sf * sf * (3 - 2 * sf)                      # the shadow side
    sm = sf
    if ndl_m is not None:
        sm = np.clip((0.35 - ndl_m) / 0.45, 0, 1)
        sm = sm * sm * (3 - 2 * sm)
        sf = 0.72 * sm + 0.28 * sf                   # the macro plane decides; the fine normal textures
    cs = (1 - shd) * (1 - sf)                        # cast shadow on the lit side
    rock = 1 - snow
    T = 1.05 * sf + 0.30 * cs + 0.20 * rock
    T = T * (1 - 0.55 * fire_steady)                 # fire light lightens the ink (it is gilded instead)
    tr = np.clip(A[..., 12], 0, 1)
    ap = 0.30 + 0.70 * tr ** 0.8                     # aerial perspective: far strokes sparse and light
    apl = 0.45 + 0.55 * tr ** 0.8                    # (land tone: mid distances keep their shadows)
    Tw = np.clip(sm * (1 - 0.9 * fire_steady) * apl, 0, 1)    # the wash follows the macro planes only
    T = np.clip(T, 0, 1.3) * apl
    T[surf >= 2.5] = 0.0
    Tw[surf >= 1.5] = 0.0
    # cloud value: lit tops bare, troughs and shadows dark
    cl = A[..., 14]
    csh = np.clip(A[..., 11], 0, 1)
    Vc = np.clip(0.18 + 0.75 * cl * (0.4 + 0.6 * csh), 0, 1)
    Tc = np.clip(1.0 - Vc, 0, 1) * (0.35 + 0.65 * tr)
    Tc[surf != 2] = 0.0
    return T.astype(np.float32), Tc.astype(np.float32), ap, Tw.astype(np.float32)


@njit(cache=True)
def _splat_lines(b, w, dens, out):
    H, W = b.shape
    for j in range(H):
        for i in range(W):
            if b[j, i] <= 0.0:
                continue
            r = 0.5 * w[j, i] + 1.0
            R = int(math.ceil(r))
            for dj in range(-R, R + 1):
                y = j + dj
                if y < 0 or y >= H:
                    continue
                for di in range(-R, R + 1):
                    x = i + di
                    if x < 0 or x >= W:
                        continue
                    d = math.sqrt(dj * dj + di * di)
                    c = 0.5 * w[j, i] + 0.5 - d
                    if c <= 0.0:
                        continue
                    if c > 1.0:
                        c = 1.0
                    c *= b[j, i] * dens[j, i]
                    if c > out[y, x]:
                        out[y, x] = c


@njit(parallel=True, fastmath=True, cache=True)
def line_mod(A, ndl_m, fire, kpx, f, out):
    """The pen's pressure along the outlines, from the world under the pen: heavier where the form turns into
    shadow, light and broken where it is lit (the lost edge), dissolving into a fire's light; pressure and pen
    lifts are world-anchored. out (H, W, 2): width factor, presence."""
    H, W = ndl_m.shape
    for j in prange(H):
        for i in range(W):
            out[j, i, 0] = 1.0
            out[j, i, 1] = 1.0
            surf = int(A[j, i, 0] + 0.5)
            if surf >= 3:
                continue
            x = A[j, i, 2]
            z = A[j, i, 4]
            mpx = A[j, i, 1] / f * kpx
            if surf == 2:
                lit = 0.5
            else:
                sh = A[j, i, 11]
                if sh < 0.0:
                    sh = 0.0
                if sh > 1.0:
                    sh = 1.0
                lit = smoothstep(0.05, 0.45, ndl_m[j, i]) * (0.35 + 0.65 * sh)
            pr = _wnoise(x, z, mpx, 90.0, 71, 3.1, 7.7)
            lift = _wnoise(x, z, mpx, 170.0, 73, 5.3, 1.9)
            out[j, i, 0] = (1.25 - 0.60 * lit) * (0.80 + 0.35 * pr)
            thr = -0.80 + 0.50 * lit
            out[j, i, 1] = smoothstep(thr, thr + 0.28, lift) * (1.0 - 0.85 * fire[j, i])


def outlines(A, kpx, mod=None):
    """Ink lines at depth breaks, on the near side: heavier for nearer and bigger breaks; the skyline always.
    Width in 1920-px: ~2.3 near, ~0.8 far; far lines also lighter. mod: line_mod's pressure and pen lifts."""
    d = A[..., 1].astype(np.float64)
    sky = d > 1e8
    dd = np.where(sky, 1e9, d)
    r = np.ones_like(dd)
    for sh in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        n = np.roll(dd, sh, axis=(0, 1))
        r = np.maximum(r, n / np.maximum(dd, 1e-3))
    b = np.clip((r - 1.05) / 0.25, 0, 1) ** 0.7
    b[sky] = 0.0
    lg = np.log10(np.maximum(dd, 1.0))
    near = np.clip((4.6 - lg) / 2.2, 0.0, 1.0)             # 1 inside ~250 m, 0 beyond ~40 km
    brk = np.clip(np.log10(np.maximum(r, 1.0)) / 0.6, 0.0, 1.0)   # 1 for a 4x depth jump
    w = kpx * (0.6 + 1.3 * near + 0.35 * brk)
    dens = (0.45 + 0.55 * near) * (0.7 + 0.3 * brk)
    if mod is not None:
        w = w * mod[..., 0]
        b = b * mod[..., 1]
    out = np.zeros(d.shape, np.float32)
    _splat_lines(b.astype(np.float32), w.astype(np.float32), dens.astype(np.float32), out)
    # a touch of softening: the line's centre sits on the pixel grid; this (at 2x, before the downsample) hides
    # the steps
    return cv2.GaussianBlur(out, (0, 0), 0.30 * kpx)


def screen_grad(h, surf):
    gy, gx = np.gradient(h.astype(np.float64))
    g = np.sqrt(gx * gx + gy * gy)
    m = surf == 2
    # at the cloud's own edges the difference crosses surfaces: fall back to a median of the neighbourhood
    g = np.where(m, g, 0.0)
    return np.minimum(g, 50.0).astype(np.float32)


def cam_vec(c):
    return np.array([c['pos'][0], c['pos'][1], c['pos'][2], c['fwd'][0], c['fwd'][1], c['fwd'][2],
                     c['right'][0], c['right'][1], c['right'][2], c['up'][0], c['up'][1], c['up'][2],
                     c['f'], c['cx'], c['cy']], np.float64)


SALT = 0                # a non-zero offset re-deals every stroke (the boiling control)
FLAME_SCALE = 7.0      # far beacons are drawn larger than life (an illustrator's licence): ~45 m flames


def flame_scale(z):
    """True size near (her own fire in the reveal), up to FLAME_SCALE for beacons kilometres off."""
    u = min(max((z - 300.0) / 2200.0, 0.0), 1.0)
    return 1.0 + (FLAME_SCALE - 1.0) * u * u * (3 - 2 * u)


# ------------------------------------------------------------ the beacons, hand-inked (H5) ---
# Each fire is DRAWN: swaying teardrop tongues outlined in iron-gall ink at the terrain's own pen weight, a gold wash
# laid inside (vermilion at the root, a pale inner flame), a thread of inked smoke. No halo in the sky and no
# icon: the catch blooms as gold wash on the rock the fire lights (see compose). Everything is analytic
# (anti-aliased from true distances) and depth-tested against the drawn land.

def _seg_dist(px, py, cx, cy):
    """Distance from points (px, py) (N,) to a polyline (cx, cy) (M,); returns distance and the curve parameter
    (0-1) of the nearest point."""
    ax, ay = cx[:-1][None, :], cy[:-1][None, :]
    bx, by = cx[1:][None, :], cy[1:][None, :]
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-12)
    t = np.clip(((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / L2, 0, 1)
    qx = ax + t * dx - px[:, None]
    qy = ay + t * dy - py[:, None]
    d2 = qx * qx + qy * qy
    k = np.argmin(d2, axis=1)
    M = len(cx) - 1
    return np.sqrt(d2[np.arange(len(px)), k]), (k + t[np.arange(len(px)), k]) / M


def _smin(a, b, k):
    """Polynomial smooth minimum (rounded unions)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def flame_tongues(t, seed, n):
    """The bonfire's tongues (27 Sep, v3): n slender tongues rising from a broad root, several of them tall (a fire,
    not one horn), each leaning with the common wind, splaying outward, bent in a gentle curve (never a hook: a hook
    closes a loop that reads as an eye) and swaying as a slow wave runs up it. Returns [(cu, cv, w)] in units of the
    flame's height, base centre at the origin, y up."""
    rng = np.random.default_rng(seed)
    wind = 0.09 + 0.06 * rng.uniform()                        # one common wind: every fire leans the same way
    spread = 0.26
    bs = np.linspace(-spread, spread, n) + 0.03 * rng.standard_normal(n)
    wm = 1.55 if n <= 3 else (1.2 if n <= 5 else 1.0)          # few tongues on a small glyph, and fuller
    s = np.linspace(0.0, 1.0, 22)
    out = []
    for rank_, i in enumerate(np.argsort(np.abs(bs))):
        b = bs[i]
        cen = 1.0 - min(abs(b) / (spread + 0.05), 1.0)
        h = 1.0 if rank_ == 0 else (0.40 + 0.48 * cen) * (0.80 + 0.35 * rng.uniform()) * (1.0 + 0.30 * b / spread)
        side = 1.0 if b >= 0 else -1.0
        l = wind + 0.22 * b + 0.05 * rng.standard_normal()        # (0.45 splayed the outer tongues into a crown)
        w = (0.085 + 0.035 * cen) * wm * (0.85 + 0.30 * rng.uniform())
        ph = rng.uniform(0, 6.283)
        fr = 0.55 + 0.35 * rng.uniform()
        hh = h * (1.0 + 0.07 * math.sin(6.283 * 0.8 * fr * t + ph))
        bend = (0.035 + 0.035 * rng.uniform()) * (side if rank_ else (1.0 if rng.uniform() < 0.5 else -1.0))
        cu = b + l * s ** 1.3 + bend * np.sin(math.pi * s) * s
        cu = cu + 0.07 * hh * s ** 1.4 * np.sin(6.283 * fr * t - 3.0 * s + ph)
        out.append((cu, hh * s, w))
    return out


def flame_sdf(px, py, H, t, seed, n, chunk=60000):
    """Signed distances (page px, <0 inside) of a drawn bonfire of height H, base centre at the origin, y up: the
    outer flame, and the inner flame (the same tongues at 58% height and 62% width, sitting on the same root: the
    illustrator's flame-within-a-flame). Evaluated in chunks (a near fire covers ~10^5-10^6 px)."""
    T = flame_tongues(t, seed, n)
    so = np.empty(px.shape)
    si = np.empty(px.shape)
    for c0 in range(0, len(px), chunk):
        X = px[c0:c0 + chunk] / H
        Y = py[c0:c0 + chunk] / H
        res = []
        for (sx_, sy_, sc) in ((1.0, 1.0, 1.0), (0.62, 0.58, 0.60)):
            sd = np.full(X.shape, 1e9)
            for cu, cv, w in T:
                d, u = _seg_dist(X / sx_, Y / sy_, cu, cv)
                prof = w * np.clip(1.0 - u, 0, 1) ** 1.3 * (1.0 + 1.2 * u)
                sd = _smin(sd, (d - prof) * H * sc, 0.022 * H * sc)
            eb = np.sqrt((X / (0.30 * sx_)) ** 2 + ((Y - 0.02) / (0.05 * sy_)) ** 2)
            sd = _smin(sd, (eb - 1.0) * 0.05 * H * sc, 0.02 * H * sc)
            res.append(sd)
        so[c0:c0 + chunk], si[c0:c0 + chunk] = res
    return so, si


def _fill_holes(sd, r):
    """Pockets of 'outside' enclosed by the flame (two tongues touching twice) would be outlined as small rings -
    eyes. They are filled, with a margin r px so no ring is drawn round them."""
    out = (sd >= 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(out, connectivity=4)
    Hh, Ww = sd.shape
    hole = np.zeros(sd.shape, np.uint8)
    for k in range(1, n):
        x, y, w, h = st[k, 0], st[k, 1], st[k, 2], st[k, 3]
        if x > 0 and y > 0 and x + w < Ww and y + h < Hh:
            hole[lab == k] = 1
    if not hole.any():
        return sd
    rr = int(math.ceil(r))
    hole = cv2.dilate(hole, np.ones((2 * rr + 1, 2 * rr + 1), np.uint8)) > 0
    return np.where(hole & (sd < rr + 1), np.minimum(sd, -(rr + 1.0)), sd)


def smoke_line(px, py, H, t, seed, k):
    """Distance (page px) to one inked smoke strand rising from the flame's tip, curling and drifting; and its
    parameter (0 at the tip, 1 at its end)."""
    rng = np.random.default_rng(seed * 31 + k)
    ph = rng.uniform(0, 6.283)
    s = np.linspace(0.0, 1.0, 40)
    drift = (0.30 + 0.20 * rng.uniform()) * (1 if k % 2 == 0 else 0.6)
    amp = 0.14 + 0.08 * rng.uniform()
    cu = (0.02 + drift * s ** 1.5 + amp * s * np.sin(6.283 * (1.1 * s - 0.22 * t) + ph)) * H
    cv = (0.92 + 0.08 * k + 1.05 * s) * H          # a short curl above the tip (the long threads read as scratches)
    return _seg_dist(px, py, cu, cv)


def flame_glyph(PX, PY, Hp, t, seed, n, kpx, pw, grain, size=1.0):
    """One fire drawn on the page. PX, PY (h, w): page px from the flame's base centre (y up); grain (h, w): the
    paper's tooth in 0-1. Returns wash alpha, wash colour (linear, (h, w, 3)) and ink alpha - before any
    occlusion. The pen: the outer line at the terrain's weight, pressing and lifting; the inner flame's line lighter
    and more broken. The wash: warm gold laid flat, the inner flame paler, the root deeper, pooling at the rim."""
    sh = PX.shape
    so, si = flame_sdf(PX.ravel(), PY.ravel(), Hp, t, seed, n)
    so = _fill_holes(so.reshape(sh), pw + 1.0)
    si = _fill_holes(si.reshape(sh), pw + 1.0)
    si = np.maximum(si, so)                                   # the inner flame stays inside the outer
    inside = np.clip(0.5 - so / kpx, 0, 1)
    core = np.clip(0.5 - si / (1.5 * kpx), 0, 1)
    v = np.clip(PY / Hp, 0, 1)
    root = np.clip((0.26 - v) / 0.26, 0, 1) ** 1.3
    pool = np.exp(-(np.maximum(-so, 0.0) / (0.03 * Hp + 0.8 * kpx)) ** 2) * inside
    g_hi, g_mid, g_lo = s2l(GOLD_HI), s2l(GOLD), s2l(GOLD_LO)
    col = g_mid[None, None] * 0.66 + g_hi[None, None] * 0.34
    col = col * (1 - 0.72 * core[..., None]) + (g_hi[None, None] * 0.80 + g_mid[None, None] * 0.20) * (0.72 * core[..., None])
    col = col * (1 - 0.40 * root[..., None]) + g_mid[None, None] * (0.40 * root[..., None])
    col = col * (1 - 0.30 * pool[..., None]) + g_lo[None, None] * (0.30 * pool[..., None])
    col = col * (0.95 + 0.10 * grain)[..., None]
    a = inside * (0.74 + 0.14 * grain)
    lift = np.sin(PX / Hp * 9.0 + seed) * np.sin(PY / Hp * 7.0 + 2 * seed)
    pen = np.clip((lift + 0.75) / 0.3, 0, 1)
    prs = 0.75 + 0.25 * np.sin(PY / Hp * 11.0 + 0.7 * seed)
    ln = np.clip(0.5 * pw * prs + 0.5 - np.abs(so), 0, 1) * (0.45 + 0.55 * pen)
    lni = np.zeros_like(ln)
    if Hp > 24.0 * kpx:
        brk2 = np.clip((np.sin(PY / Hp * 13.0 + 1.7 * seed) + 0.25) / 0.3, 0, 1)
        lni = np.clip(0.35 * pw + 0.5 - np.abs(si), 0, 1) * 0.50 * brk2 * np.clip(-so / (1.5 * kpx), 0, 1)
    fuel = np.zeros_like(ln)
    if Hp > 45.0 * kpx:                       # a few crossed billets under a near fire, behind the flame
        X, Y = PX / Hp, PY / Hp
        for (ax_, ay_, bx_, by_) in ((-0.36, -0.03, 0.30, 0.04), (-0.30, 0.04, 0.36, -0.04), (-0.12, -0.06, 0.14, -0.01)):
            dx_, dy_ = bx_ - ax_, by_ - ay_
            tt_ = np.clip(((X - ax_) * dx_ + (Y - ay_) * dy_) / (dx_ * dx_ + dy_ * dy_), 0, 1)
            dd_ = np.sqrt((X - ax_ - tt_ * dx_) ** 2 + (Y - ay_ - tt_ * dy_) ** 2) * Hp
            fuel = np.maximum(fuel, np.clip(0.5 * 2.2 * pw + 0.5 - dd_, 0, 1) * (1 - 0.6 * np.abs(tt_ - 0.5)))
        fuel = fuel * np.clip(1.0 - inside * 1.5, 0, 1)
    sm = np.zeros_like(ln)
    if size > 0.6 and Hp > 9.0 * kpx:        # a curl of smoke or two, thin and broken, once the fire is up
        for k in range(2 if Hp > 40.0 * kpx else 1):
            dd, u = smoke_line(PX.ravel(), PY.ravel(), Hp, t, seed, k)
            dd, u = dd.reshape(sh), u.reshape(sh)
            wv = pw * 0.8 * (1.0 - 0.8 * u)
            brk = 0.5 + 0.5 * np.sin(6.283 * (u * (3.0 + k) - 0.35 * t) + seed)
            sm = np.maximum(sm, np.clip(0.5 * wv + 0.5 - dd, 0, 1) * np.clip(1.0 - u, 0, 1) ** 1.1
                            * (brk > 0.25) * 0.45 * min((size - 0.6) / 0.3, 1.0))
    return a, col, np.maximum.reduce([ln, lni, sm, fuel])


def ink_flames(R, B, kpx, pp):
    """The fires drawn on the page. Returns wash alpha, wash colour (linear), ink alpha and the ink's farness, (H, W)."""
    import beacons as BC
    A = R['A']
    H, W = A.shape[:2]
    c = R['cam']
    frame = R['frame']
    t = frame / 24.0
    fa = np.zeros((H, W), np.float32)
    fc = np.zeros((H, W, 3), np.float32)
    la = np.zeros((H, W), np.float32)
    lf = np.zeros((H, W), np.float32)             # the pen's farness (aerial perspective) where it drew
    dist = A[..., 1]
    rows = []
    for b in B:
        size, inten, light = BC.env(frame, b[3])
        if size <= 0.02:
            continue
        base = np.array([b[0], b[1] + 1.6, b[2]])
        d = base - c['pos']
        z = float(d @ c['fwd'])
        if z <= 1.0:
            continue
        sx = c['cx'] + c['f'] * float(d @ c['right']) / z
        sy = c['cy'] - c['f'] * float(d @ c['up']) / z
        rows.append((float(np.linalg.norm(d)), z, sx, sy, size, b))
    rows.sort(key=lambda r: -r[0])                       # far to near
    for fd, z, sx, sy, size, b in rows:
        Hf = 6.5 * flame_scale(z) * min(size, 1.15)
        Hp = Hf * c['f'] / z                              # flame height, page px (AOV res)
        if Hp < 2.0 * kpx:
            continue
        near = min(max((Hp / kpx - 20.0) / 120.0, 0.0), 1.0)
        pw = kpx * (0.75 + 0.55 * near)                   # the terrain's pen weight, a touch more near
        x0 = int(max(sx - 0.9 * Hp - 4, 0))
        x1 = int(min(sx + 1.2 * Hp + 5, W))
        y0 = int(max(sy - 2.3 * Hp - 4, 0))
        y1 = int(min(sy + 0.15 * Hp + 5, H))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        PX = xx + 0.5 - sx
        PY = sy - (yy + 0.5)
        seed = int(b[5]) * 7 + 3
        n = 3 if Hp < 22.0 * kpx else (5 if Hp < 70.0 * kpx else 7)
        grain = np.clip(pp[y0:y1, x0:x1, 1] * 0.5 + 0.5, 0, 1)
        a, col, l_all = flame_glyph(PX, PY, Hp, t, seed, n, kpx, pw, grain, size)
        # the land in front hides the fire (a nearer ridge); the fire's own summit does not
        vis = (dist[y0:y1, x0:x1] > fd - max(3.0, 0.004 * fd)).astype(np.float32)
        vis = cv2.GaussianBlur(vis, (0, 0), 0.6 * kpx) if min(vis.shape) > 3 else vis
        a = a * vis
        fa[y0:y1, x0:x1] = np.maximum(fa[y0:y1, x0:x1], a)
        m = (a > 0)[..., None]
        fc[y0:y1, x0:x1] = np.where(m, col, fc[y0:y1, x0:x1])
        l_all = l_all * vis
        iy, ix = int(min(max(sy + 2.0 * kpx, 0), H - 1)), int(min(max(sx, 0), W - 1))
        farf = float(np.clip(1.0 - A[iy, ix, 12], 0.0, 1.0)) * 0.7
        upd = l_all > la[y0:y1, x0:x1]
        lf[y0:y1, x0:x1] = np.where(upd, farf, lf[y0:y1, x0:x1])
        la[y0:y1, x0:x1] = np.maximum(la[y0:y1, x0:x1], l_all)
    return fa, fc, la, lf


def fire_soft(A, B, frame, campos):
    """The fires' light on the land as a painter lays it: a smooth falloff with distance from each fire (world
    position), growing with the catch. The AOV's firelight follows the fine normal and bands on stratified
    rock; this does not."""
    import beacons as BC
    g = np.zeros(A.shape[:2], np.float32)
    if B is None or not len(B):
        return g
    x, h, z = A[..., 2], A[..., 5], A[..., 4]
    land = A[..., 0] < 1.5
    for b in B:
        size, inten, light = BC.env(frame, b[3])
        if light <= 0:
            continue
        d2 = (x - b[0]) ** 2 + (h - b[1] - 1.5) ** 2 + (z - b[2]) ** 2
        Rr = 9.0 * flame_scale(float(np.linalg.norm(np.asarray(b[:3]) - campos)))
        g = np.maximum(g, (light / (1.0 + d2 / (Rr * Rr)) ** 1.5).astype(np.float32))
    return np.where(land, g, 0.0).astype(np.float32)


HFIELD_SIG = 1.5        # the line-placing fields' smoothing along the surface (page px)
WASH_LO, WASH_HI, WASH_BLUR = 0.40, 0.62, 1.2     # the sepia shadow wash's ramp on the macro shadow, and its blur
                                                  # (27 Sep farm A/B: 0.30/0.85/2.5 read as grey smudges; firmer edges)
if os.environ.get('RUNC_WASH'):                     # look-dev A/B on the farm: RUNC_WASH=lo,hi,blur
    WASH_LO, WASH_HI, WASH_BLUR = [float(v) for v in os.environ['RUNC_WASH'].split(',')]


@njit(parallel=True, fastmath=True, cache=True)
def _surf_blur(F, D, land, sig, out):
    """Separable Gaussian of the fields F (H, W, K) along the land's surface: a neighbour counts only if it is land
    at nearly the same depth (a silhouette is never crossed). Non-land pixels are copied."""
    H, W, K = F.shape
    r = int(math.ceil(2.0 * sig))
    tmp = np.empty_like(F)
    inv = 1.0 / (2.0 * sig * sig)
    for j in prange(H):
        for i in range(W):
            d0 = D[j, i]
            if land[j, i] < 0.5:
                for k in range(K):
                    tmp[j, i, k] = F[j, i, k]
                continue
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            ws = 0.0
            for o in range(-r, r + 1):
                q = i + o
                if q < 0 or q >= W or land[j, q] < 0.5:
                    continue
                dd = (D[j, q] - d0) / (0.03 * d0)
                if dd > 1.0 or dd < -1.0:
                    continue
                wgt = math.exp(-o * o * inv) * (1.0 - dd * dd)
                acc0 += wgt * F[j, q, 0]
                acc1 += wgt * F[j, q, 1]
                acc2 += wgt * F[j, q, 2]
                ws += wgt
            tmp[j, i, 0] = acc0 / ws
            tmp[j, i, 1] = acc1 / ws
            tmp[j, i, 2] = acc2 / ws
    for j in prange(H):
        for i in range(W):
            d0 = D[j, i]
            if land[j, i] < 0.5:
                for k in range(K):
                    out[j, i, k] = F[j, i, k]
                continue
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            ws = 0.0
            for o in range(-r, r + 1):
                q = j + o
                if q < 0 or q >= H or land[q, i] < 0.5:
                    continue
                dd = (D[q, i] - d0) / (0.03 * d0)
                if dd > 1.0 or dd < -1.0:
                    continue
                wgt = math.exp(-o * o * inv) * (1.0 - dd * dd)
                acc0 += wgt * tmp[q, i, 0]
                acc1 += wgt * tmp[q, i, 1]
                acc2 += wgt * tmp[q, i, 2]
                ws += wgt
            out[j, i, 0] = acc0 / ws
            out[j, i, 1] = acc1 / ws
            out[j, i, 2] = acc2 / ws


def _crest_hatch_aa(A, hat, kpx):
    """Soften hatch sampling near a recovered crest, without filtering the page or outlines.

    The C13 march refinement restores the surface but leaves hatch contrast changes
    above the nearby paper/ink baseline. Limit this appearance correction to land
    around the nearer-side pixels of depth breaks: full strength within 8 page
    pixels, fading to zero at 16. The 0.7-page-pixel filter uses the existing
    surface-depth weights, keeping the two sides separate at a clear depth step.
    These settings are checked on C2990..3013; they are not a temporal guarantee
    for every camera. Call only under the explicit crest-refinement opt-in.
    """
    D = np.ascontiguousarray(A[..., 1])
    land = (A[..., 0] < 1.5).astype(np.float32)
    edge = (land > 0) & (cv2.dilate(D, np.ones((3, 3), np.uint8)) > D * 1.05)
    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip((16.0 * kpx - distance) / (8.0 * kpx), 0, 1) * land
    weight = weight * weight * (3 - 2 * weight)
    # _surf_blur operates on exactly three channels; the hatch has two.
    fields = np.zeros((*hat.shape[:2], 3), np.float32)
    fields[..., :2] = hat
    smoothed = np.empty_like(fields)
    _surf_blur(fields, D, land, 0.7 * kpx, smoothed)
    return hat + (smoothed[..., :2] - hat) * weight[..., None]


@njit(parallel=True, cache=True)
def _deteeth(A, w, near, out):
    """The ray march at a grazing crest hits in some pixel columns and misses in the next, so a far ridge's crest
    comes back as a comb of 1-4 px columns (land, cloud, land ...) that the outliner would draw as organ-pipe bars.
    A running median of the depth along each row (2w+1 taps) keeps every edge and every smooth run and removes
    only runs of w pixels or fewer; each outlier pixel takes all its channels from the nearest pixel in reach whose
    depth is the median's. Pixels nearer than `near` m (the summit she stands on) are left alone."""
    H, W, C = A.shape
    for j in prange(H):
        buf = np.empty(2 * w + 1, np.float64)
        for i in range(W):
            for c in range(C):
                out[j, i, c] = A[j, i, c]
        # pass 1 (erode first): a sliver of w px or fewer that stands NEARER than both its neighbours is a crest the
        # march only half-sampled; it takes the far side's data (each half from its own side). A median alone
        # filled the densest comb solid and stood a flat-topped chimney on the ridge.
        i = 1
        while i < W - 1:
            d = A[j, i, 1]
            if d >= near and A[j, i - 1, 1] > d * 1.05:
                k = i + 1
                while k < W and k - i <= w and abs(A[j, k, 1] - d) <= 0.10 * d:
                    k += 1
                if k < W and k - i <= w and A[j, k, 1] > A[j, k - 1, 1] * 1.05:
                    for q in range(i, k):
                        s = i - 1 if 2 * (q - i) < (k - i) else k
                        for c in range(C):
                            out[j, q, c] = A[j, s, c]
                    i = k
                    continue
            i += 1
        # pass 2: a running median of the eroded depth fills the thin far slits left inside a solid face
        row = out[j].copy()
        for i in range(w, W - w):
            d = row[i, 1]
            if d < near:
                continue
            for q in range(2 * w + 1):
                buf[q] = row[i - w + q, 1]
            buf.sort()
            med = buf[w]
            if abs(d - med) <= 0.04 * med:
                continue
            src = -1
            for o in range(1, w + 1):
                if abs(row[i - o, 1] - med) <= 0.02 * med:
                    src = i - o
                    break
                if abs(row[i + o, 1] - med) <= 0.02 * med:
                    src = i + o
                    break
            if src >= 0:
                for c in range(C):
                    out[j, i, c] = row[src, c]


def deteeth(A, kpx):
    out = np.empty_like(A)
    _deteeth(A, max(3, int(round(2.5 * kpx))), 300.0, out)
    return out


def compose(R, prm=None, B=None, plate=None, CR=None):
    R = dict(R)
    R['A'] = deteeth(R['A'], R.get('kpx', R['A'].shape[1] / 1920.0))
    A = R['A']
    H, W = A.shape[:2]
    kpx = R.get('kpx', W / 1920.0)            # (a crop at production scale passes its true kpx)
    cam = cam_vec(R['cam'])
    t = R['cam']['t']
    fire_st = np.clip(fire_soft(A, B, R['frame'], np.asarray(R['cam']['pos'], np.float64)) / 0.5, 0, 1)        # the fires' light, steady and smooth
    ndl_m = np.zeros((H, W), np.float32)
    hsm = np.zeros((H, W), np.float32)
    macro_ndl(A, np.asarray(R['Lk'][:3], np.float64), RN.CR if CR is None else CR, t, cam[12], kpx, MACRO_K, HSMOOTH,
              ndl_m, hsm)
    T, Tc, ap, Tw = darkness(A, fire_st, ndl_m)
    # band-limit the tone to the stroke scale: the per-pixel normal and close-range snow alias as the pixel grid
    # slides over the land, and a stroke near its rank threshold would flicker; a screen blur of a world-anchored
    # field is shift-invariant, so the smoothed tone still travels with the terrain
    land = (A[..., 0] < 1.5).astype(np.float32)
    sig = 1.8 * kpx
    T = cv2.GaussianBlur(T * land, (0, 0), sig) / np.maximum(cv2.GaussianBlur(land, (0, 0), sig), 1e-3) * land
    cm = (A[..., 0] == 2).astype(np.float32)
    Tc = cv2.GaussianBlur(Tc * cm, (0, 0), sig) / np.maximum(cv2.GaussianBlur(cm, (0, 0), sig), 1e-3) * cm
    T = T.astype(np.float32)
    Tc = Tc.astype(np.float32)
    p = np.array([kpx, 6.2, t, 10.0, 3.6, SALT], np.float64)
    hat = np.zeros((H, W, 2), np.float32)
    D = A[..., 1]
    # (1:1 crops) every hatch line carried a pixel-scale sawtooth: on rough rock the hit point's plan position (x, z)
    # jumps from pixel to pixel (micro-occlusion), and the cross-contour field inherits it. The fields that PLACE
    # the lines (height, x, z) are smoothed along the surface (depth-aware, never across a silhouette) at ~1.5 page
    # px: a screen blur of world-anchored fields is shift-invariant, so the lines still ride the rock.
    Fh = np.ascontiguousarray(np.stack([hsm, A[..., 2], A[..., 4]], axis=2).astype(np.float32))
    Fb = np.empty_like(Fh)
    _surf_blur(Fh, D.astype(np.float32), (A[..., 0] < 1.5).astype(np.float32), HFIELD_SIG * kpx, Fb)
    hsb = np.ascontiguousarray(Fb[..., 0])
    Ah = np.ascontiguousarray(np.stack([A[..., 0], A[..., 1], Fb[..., 1], A[..., 3], Fb[..., 2]], axis=2))
    ghx, ghy = sgrad(hsb, D)
    ya = math.radians(SHOT_YAW.get(R.get('shot', ''), 0.0))
    rx, rz = math.cos(ya), -math.sin(ya)
    gpx, gpy = sgrad(Ah[..., 2].astype(np.float64) * rx + Ah[..., 4].astype(np.float64) * rz, D)
    fx, fz = math.sin(ya), math.cos(ya)
    gcx, gcy = sgrad(Ah[..., 2].astype(np.float64) * fx + Ah[..., 4].astype(np.float64) * fz, D)
    G = np.ascontiguousarray(np.stack([ghx, ghy, gpx, gpy, gcx, gcy], axis=2))
    xp = np.array([kpx, XSP1, XSP2, math.cos(XB), math.sin(XB), rx, rz, SALT, cam[12], XT2, XKAP, fx, fz, XT1MAX],
                  np.float64)
    xhatch(Ah, T, G, hsb, xp, hat)
    if os.environ.get('LD_INK_CREST_REFINE') == '1':
        hat = _crest_hatch_aa(Ah, hat, kpx)
    gs = screen_grad(A[..., 5], np.rint(A[..., 0]))
    cl = np.zeros((H, W, 2), np.float32)
    cloud(A, Tc, gs, cam, p, cl)
    lm = np.zeros((H, W, 2), np.float32)
    line_mod(A, ndl_m, fire_st.astype(np.float32), kpx, cam[12], lm)
    ol = outlines(A, kpx, lm)
    pp = np.zeros((H, W, 2), np.float32)
    paper(H, W, kpx, pp)
    fl_a = fl_c = fl_l = None
    if B is not None and len(B):
        R = dict(R)
        R['E'] = np.zeros_like(R['E'])          # the fires are drawn (ink_flames), not rendered
        R['a'] = np.zeros_like(R['a'])
        fl_a, fl_c, fl_l, fl_f = ink_flames(R, B, kpx, pp)
    pl_l = pl_w = None
    if plate is not None:
        R = dict(R)
        R['E'] = np.zeros_like(R['E'])
        R['a'] = np.zeros_like(R['a'])          # the dawn's smoke is drawn (ink_plumes), not rendered
        pl_l, pl_w = ink_plumes(R, kpx)
    # ---- the page
    tt = np.clip(pp[..., 0], 0, 1)[..., None]
    base = s2l(PAPER)[None, None] * (1 - 0.55 * tt) + s2l(PAPER_DARK)[None, None] * (0.55 * tt)
    base = base * (1.0 + 0.06 * pp[..., 1])[..., None]
    # ---- sepia wash in the shadows (a brush, soft-edged)
    wash = np.clip((Tw - WASH_LO) / (WASH_HI - WASH_LO), 0, 1)
    wash = wash * wash * (3 - 2 * wash) * 0.26 + np.clip(Tc - 0.45, 0, 1) * 0.25
    wash = cv2.GaussianBlur(wash.astype(np.float32), (0, 0), WASH_BLUR * kpx)
    wcol = base * (s2l(WASH) / s2l(PAPER))[None, None]
    rgb = base * (1 - wash[..., None]) + wcol * wash[..., None]
    # ---- the catch's bloom: a gold wash laid on the rock the fire lights (world-anchored: the light's own reach),
    # soft, taking the paper's grain at its edge and pooling a little darker where it dried
    landm = (A[..., 0] < 1.5).astype(np.float32)
    fs = fire_st * landm
    g0 = np.clip((fs - 0.15) / 0.45, 0, 1)
    g_land = np.clip((g0 - 0.06 + 0.05 * pp[..., 1]) / 0.60, 0, 1) * landm
    g_land = cv2.GaussianBlur(g_land, (0, 0), 0.7 * kpx)
    tide = np.exp(-((g_land - 0.10) / 0.07) ** 2) * (g_land > 0.02) * 0.35
    gold_a = np.clip(0.48 * g_land + tide * 0.4, 0, 0.6)
    gcol = s2l(GOLD)[None, None] * (0.92 + 0.16 * np.clip(pp[..., 1] * 0.5 + 0.5, 0, 1))[..., None]
    gcol = gcol * (1 - 0.25 * tide[..., None]) + s2l(GOLD_LO)[None, None] * (0.25 * tide[..., None])
    rgb = rgb * (1 - gold_a[..., None]) + gcol * gold_a[..., None]
    gl = np.zeros((H, W), np.float32) if fl_a is None else fl_a
    fill = None
    if plate is not None:
        fill, rgb = illum_fill(R, plate, rgb, pp, kpx)
    # ---- ink: hatching, cross-hatching, cloud lines and stipple, outlines
    clear = 1.0 - 0.85 * gl
    ink_a = np.maximum.reduce([hat[..., 0] * clear, hat[..., 1] * 0.9 * clear, cl[..., 0] * clear,
                               cl[..., 1] * 0.9 * clear, ol * (1.0 - 0.35 * gl)])
    # the paper's tooth takes more or less ink at a stroke's edge
    ink_a = np.clip(ink_a * (0.92 + 0.12 * pp[..., 1]), 0, 1)
    far = np.clip(1.0 - ap, 0, 1)[..., None]                       # far lines lighter, browner
    far = np.where(ol[..., None] > hat[..., :1], far * 0.6, far)
    icol = s2l(INK)[None, None] * (1 - far) + s2l(INK_LIGHT)[None, None] * far
    ink_a = ink_a * (1 - 0.45 * far[..., 0])
    if fill is not None:
        ink_a = ink_a * (1 - 0.08 * fill)                          # coloured ground: the line work stays
    rgb = rgb * (1 - ink_a[..., None]) + icol * ink_a[..., None]
    if fill is not None:
        rgb = sun_ink(R, rgb, pp, kpx)
    if pl_l is not None:
        rgb = rgb * (1 - pl_w[..., None]) + (rgb * (s2l(WASH) / s2l(PAPER))[None, None] * 0.85) * pl_w[..., None]
        pcol = s2l(INK)[None, None] * 0.25 + s2l(INK_LIGHT)[None, None] * 0.75
        rgb = rgb * (1 - pl_l[..., None]) + pcol * pl_l[..., None]
    # ---- the fires, drawn: the gold wash, then the pen
    if fl_a is not None:
        rgb = rgb * (1 - fl_a[..., None]) + fl_c * fl_a[..., None]
        fcl = s2l(INK)[None, None] * (1 - fl_f[..., None]) + s2l(INK_LIGHT)[None, None] * fl_f[..., None]
        fla = 0.88 * fl_l * (1.0 - 0.35 * fl_f)
        rgb = rgb * (1 - fla[..., None]) + fcl * fla[..., None]
    E = R['E']
    lum = E @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    mx = np.maximum(E.max(axis=2), 1e-5)
    fa = 1.0 - np.exp(-5.0 * lum)
    fcol = E / mx[..., None]                       # full saturation: vermilion, orange, yellow by temperature
    fcol = fcol ** 1.25 * 0.92
    hot = np.clip((lum - 1.5) / 4.0, 0, 1)[..., None]
    fcol = fcol * (1 - hot) + s2l(GOLD_HI)[None, None] * 1.05 * hot          # the hottest core burnished gold
    rgb = rgb * (1 - fa[..., None]) + fcol * fa[..., None]
    # smoke / soot coverage: a thin sepia wash
    sa = np.clip(R['a'] - fa * 0.5, 0, 1) * 0.55
    rgb = rgb * (1 - sa[..., None]) + (base * (s2l(WASH) / s2l(PAPER))[None, None] * 0.8) * sa[..., None]
    a_s = cv2.GaussianBlur(np.clip(R['a'] - fa, 0, 1).astype(np.float32), (0, 0), 0.8 * kpx)
    gy, gx = np.gradient(a_s)
    gsm = np.sqrt(gx * gx + gy * gy) + 1e-4
    line = np.clip(0.5 * 0.8 * kpx + 0.5 - np.abs(a_s - 0.10) / gsm, 0, 1) * np.clip(gsm / 0.01, 0, 1) * 0.6
    rgb = rgb * (1 - line[..., None]) + s2l(INK_LIGHT)[None, None] * line[..., None]
    out = look.linear_to_srgb(np.clip(rgb, 0, 1))
    return out.astype(np.float32), dict(T=T, hat=hat, cl=cl, ol=ol, ink=ink_a)


def _sun_px(R):
    import dawn as DN
    c = R['cam']
    d = DN.sun_dir(R['frame'])
    x = d @ c['right']
    y = d @ c['up']
    z = d @ c['fwd']
    return c['cx'] + c['f'] * x / z, c['cy'] - c['f'] * y / z, c['f']


def _page_noise(H, W, kpx, scale_px, seed):
    """Low-frequency noise fixed to the page (1920-px units), for the irregular edges of a painted wash."""
    h, w = max(H // 16, 4), max(W // 16, 4)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    k = 16.0 / kpx / scale_px
    n = np.zeros((h, w), np.float32)
    for o, amp in ((1.0, 1.0), (2.1, 0.5), (4.3, 0.25)):
        n += amp * np.sin(xs * k * o * 6.283 / 7.0 + seed + o) * np.sin(ys * k * o * 6.283 / 5.0 + 2.1 * seed + 1.7 * o)
    return cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC) / 1.75


def illum_fill(R, plate, rgb, pp, kpx):
    """THE ILLUMINATION: the colour is laid where the sun's light goes, and spreads as the light spreads. In the
    sky, a wash floods out from the sun along the horizon first, then up (the dawn's own light: brighter sky is
    reached sooner), with a wide, irregular, feathered edge and a faint dried tide-line; there is never a mask
    edge. On the land, only what the sun actually touches (its rims, the lit cloud sea) takes colour, as the wash
    reaches it; the faces turned from the sun stay ink, with a faint cool wash late. plate: the dawn in colour
    (display-referred, linearised, H x W x 3)."""
    import dawn as DN
    A = R['A']
    H, W = A.shape[:2]
    frame = R['frame']
    surf = A[..., 0]
    gain = DN.light_gain(frame)
    sx, sy, f = _sun_px(R)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    daz = np.degrees(np.arctan((xx - sx) / f))
    delv = np.degrees(np.arctan((yy - sy) / f))
    ang = np.sqrt((daz / 1.7) ** 2 + delv ** 2)                  # the dawn spreads along the horizon first
    pb = cv2.GaussianBlur(plate, (0, 0), 10.0 * kpx)
    Y = pb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    Ys = cv2.GaussianBlur(Y, (0, 0), 24.0 * kpx)
    Yn = np.clip(Ys / max(float(np.percentile(Ys, 99.5)), 1e-4), 0, 1)
    u = np.clip((frame - 2402.0) / 318.0, 0, 1)                 # C24: complete as the line settles home (local 320)
    reach = 2.0 + 44.0 * (1.0 - (1.0 - u) ** 1.6)                # degrees reached so far (the frame is ~23 deg a side)
    edge_n = _page_noise(H, W, kpx, 260.0, 3.0) * 3.5 + 1.2 * pp[..., 1]
    m = ang - 9.0 * Yn + edge_n                                  # brighter light is reached sooner
    front = np.clip((reach - m) / 9.0, 0, 1)                     # a wide feather: ~9 deg
    front = front * front * (3 - 2 * front)
    # what the light touches
    litl = np.clip(A[..., 10], 0, 1) * np.clip(A[..., 11], 0, 1) * gain
    cl = np.clip(A[..., 14] * (0.35 + 0.65 * np.clip(A[..., 11], 0, 1)) * gain / 1.1, 0, 1)
    rimY = np.clip((Y - 0.12) / 0.25, 0, 1)
    touch = np.where(surf >= 2.5, 1.0, np.where(surf == 2, np.clip(1.3 * cl - 0.05, 0, 1),
                                                 np.clip(1.8 * litl - 0.05, 0, 1)))
    touch = cv2.GaussianBlur(touch.astype(np.float32), (0, 0), 1.5 * kpx)
    M = np.clip(front * touch, 0, 1)
    # the colour: the dawn's hue in broad washes (its detail blurred away), value eased toward the paper
    pc = cv2.GaussianBlur(plate, (0, 0), 22.0 * kpx)
    Yc = pc @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    chroma = pc / np.maximum(Yc, 1e-4)[..., None]
    chroma = np.clip(chroma, 0, 3) ** 0.8
    # blue on this yellow paper would turn green: the cool washes are laid as lilac (a violet with red in it)
    chroma[..., 0] = np.maximum(chroma[..., 0], 0.92 * chroma[..., 2])
    chroma[..., 1] = np.minimum(chroma[..., 1], 0.5 * (chroma[..., 0] + chroma[..., 2]))
    chroma = chroma / np.maximum(chroma @ np.array([0.2126, 0.7152, 0.0722], np.float32), 1e-4)[..., None]
    near_sun = np.clip(1.0 - ang / 7.0, 0, 1)
    tint = chroma * (1 - near_sun[..., None]) + (s2l(GOLD_HI) / s2l(GOLD_HI).mean())[None, None] * near_sun[..., None]
    gran = 0.97 + 0.06 * np.clip(pp[..., 1] * 0.5 + 0.5, 0, 1)            # pigment settling in the tooth
    tide = 1.0 - 0.07 * np.exp(-((front - 0.18) / 0.08) ** 2) * (surf >= 2.5)
    paper_n = rgb / s2l(PAPER)[None, None]
    washed = paper_n * s2l(PAPER)[None, None] * (0.55 + 0.45 * np.clip(tint * 0.95, 0, 1.25)) * (gran * tide)[..., None]
    washed = washed * (1.0 + 0.10 * near_sun[..., None])
    out = rgb * (1 - M[..., None]) + washed * M[..., None]
    # a faint cool wash on the faces turned from the sun, late, so the whole page ends coloured
    late = np.clip((frame - 2402.0 - 200.0) / 120.0, 0, 1)
    shade = np.clip(1.0 - touch, 0, 1) * (surf < 1.5) * late * 0.10
    cool = s2l(np.array([150, 150, 176]))
    out = out * (1 - shade[..., None]) + (out * cool[None, None] / s2l(PAPER)[None, None]) * shade[..., None]
    # gold leaf where the light is strongest: the sun's path on the cloud sea, laid as the wash arrives there
    gl = np.clip((Y - 0.32) / 0.25, 0, 1) * np.clip((10.0 - ang) / 6.0, 0, 1) * (surf == 2) * M
    gl = np.clip((gl - 0.35 + 0.12 * pp[..., 1]) / 0.25, 0, 1)
    gl = cv2.GaussianBlur(gl.astype(np.float32), (0, 0), 0.9 * kpx)
    sheen = np.clip(1.0 - ang / 10.0, 0, 1)
    gcol = s2l(GOLD)[None, None] * (1 - sheen[..., None]) + s2l(GOLD_HI)[None, None] * sheen[..., None]
    out = out * (1 - 0.75 * gl[..., None]) + gcol * (0.75 * gl[..., None])
    return M, out


def ink_plumes(R, kpx):
    """C24's smoke rising from every summit that held a beacon, drawn: two thin strands curling up each plume's own
    column (dawn.py's plumes: straight up, then bent downwind at the inversion) with a faint sepia wash between
    them, hidden by any land in front. Returns line alpha, wash alpha (H, W)."""
    import dawn as DN
    A = R['A']
    H, W = A.shape[:2]
    c = R['cam']
    t = R['frame'] / 24.0
    la = np.zeros((H, W), np.float32)
    wa = np.zeros((H, W), np.float32)
    dist = A[..., 1]
    pos, fwd, right, up = [np.asarray(c[k], np.float64) for k in ('pos', 'fwd', 'right', 'up')]
    wn = DN.WIND / np.linalg.norm(DN.WIND)
    rng = np.random.default_rng(5)
    for (x, y, z, kind, k) in DN._plume_sources():
        dd0 = float(np.linalg.norm(np.array([x, y, z]) - pos))
        lic = 1.0 + 1.6 * min(max((dd0 - 1500.0) / 10000.0, 0.0), 1.0)      # an illustrator's licence, as the flames
        Hc = (230.0 if kind == 1 else 150.0) * lic
        sgm = np.linspace(0.0, 1.0, 36)
        hgt = sgm * Hc
        top = np.clip((sgm - 0.72) / 0.28, 0, 1)
        drift = hgt * 0.25 + top * Hc * 0.9
        P = np.stack([x + wn[0] * drift, y + 2.0 + hgt - top * Hc * 0.12, z + wn[2] * drift], 1)
        rr = (1.5 + 0.09 * hgt / lic + top * 14.0) * lic
        d = P - pos
        zc = d @ fwd
        if zc.min() < 30.0:
            continue
        sx = c['cx'] + c['f'] * (d @ right) / zc
        sy = c['cy'] - c['f'] * (d @ up) / zc
        rp = rr * c['f'] / zc
        if sx.max() < -20 or sx.min() > W + 20 or sy.max() < -20 or sy.min() > H + 20:
            continue
        if (sy[0] - sy.min()) < 5.0 * kpx:
            continue
        dmin = float(np.linalg.norm(d, axis=1).min())
        x0 = int(max(sx.min() - rp.max() - 4, 0))
        x1 = int(min(sx.max() + rp.max() + 5, W))
        y0 = int(max(sy.min() - rp.max() - 4, 0))
        y1 = int(min(sy.max() + 4, H))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        qx, qy = (xx + 0.5).ravel(), (yy + 0.5).ravel()
        vis = (dist[y0:y1, x0:x1] > dmin - max(10.0, 0.004 * dmin)).astype(np.float32)
        # the column's own wash
        dc, uc = _seg_dist(qx, qy, sx, sy)
        rw = np.interp(uc, sgm, rp)
        fade = np.clip(uc / 0.06, 0, 1) * (1.0 - uc) ** 0.6
        wash = np.clip(1.0 - dc / np.maximum(rw, 0.8 * kpx), 0, 1) ** 1.5 * fade * 0.22
        wa[y0:y1, x0:x1] = np.maximum(wa[y0:y1, x0:x1], wash.reshape(yy.shape) * vis)
        # two strands winding round the column, the curl running up it
        ph = rng.uniform(0, 6.283)
        nx_, ny_ = np.gradient(sy), -np.gradient(sx)
        nn = np.maximum(np.sqrt(nx_ ** 2 + ny_ ** 2), 1e-6)
        nx_, ny_ = nx_ / nn, ny_ / nn
        # (1:1 crops) two dark strands either side of a thin column read as a staple or a crack: a thin plume is
        # ONE pale strand winding across its column; a broad one keeps two
        for side in ((-1.0, 1.0) if rp.max() > 3.0 * kpx else (0.0,)):
            wob = side * 0.55 + (0.45 if side else 0.85) * np.sin(6.283 * (2.2 * sgm - 0.12 * t) + ph + (0 if side >= 0 else 2.3))
            ux = sx + nx_ * rp * wob
            uy = sy + ny_ * rp * wob
            dl, ul = _seg_dist(qx, qy, ux, uy)
            pw = kpx * (0.75 - 0.35 * ul)
            brk = (np.sin(6.283 * (ul * 5.0 - 0.2 * t) + ph + side) > -0.65)
            ln = (np.clip(0.5 * pw + 0.5 - dl, 0, 1) * brk * np.clip(ul / 0.04, 0, 1) * (1.0 - ul) ** 0.9 * 0.60
                  * np.clip((0.80 - ul) / 0.18, 0, 1))      # the strands give out before the plume bends over
            la[y0:y1, x0:x1] = np.maximum(la[y0:y1, x0:x1], ln.reshape(yy.shape) * vis)
    return la, wa


def sun_ink(R, rgb, pp, kpx):
    """The sun as the old books draw it: a pen circle, a little uneven, with a lift where the pen began; inside,
    the paper under a pale gold wash that pools at the rim; around it, fine ruled rays of broken, uneven length.
    The drawn skyline cuts it as it rises."""
    H, W = rgb.shape[:2]
    sx, sy, f = _sun_px(R)
    rad = f * math.tan(math.radians(0.75))
    if sx < -5 * rad or sx > W + 5 * rad:
        return rgb
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx + 0.5 - sx, yy + 0.5 - sy
    r = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)
    sky = (R['A'][..., 0] >= 2.5).astype(np.float32)
    sky = cv2.GaussianBlur(sky, (0, 0), 0.6 * kpx)
    rr = rad * (1.0 + 0.014 * np.sin(3 * th + 0.7) + 0.008 * np.sin(7 * th + 2.0))
    inside = np.clip(rr - r + 0.5, 0, 1) * sky
    pool = np.exp(-((rr - r) / (0.12 * rad + kpx)) ** 2) * inside
    wash = inside * 0.42 + pool * 0.25
    gcol = s2l(GOLD_HI)[None, None] * (0.94 + 0.12 * np.clip(pp[..., 1] * 0.5 + 0.5, 0, 1))[..., None]
    out = rgb * (1 - wash[..., None]) + gcol * wash[..., None]
    out = out * (1 - 0.35 * pool[..., None]) + s2l(GOLD)[None, None] * (0.35 * pool[..., None])
    # the pen circle, lifted near the top left where it began and ended
    gap = np.clip((np.abs(np.angle(np.exp(1j * (th + 2.2)))) - 0.10) / 0.12, 0, 1)
    pw = 0.80 * kpx * (0.85 + 0.30 * np.sin(2.0 * th + 1.1) ** 2)         # the pen presses and eases round it
    ring = np.clip(0.5 * pw + 0.5 - np.abs(r - rr), 0, 1) * gap * sky * 0.80
    # the rays: fine ruled lines, alternately long and short, never touching the disc
    N = 44
    k = np.floor((th + math.pi) / (2 * math.pi) * N + 0.5)
    tk = k / N * 2 * math.pi - math.pi
    rng = np.random.default_rng(7)
    jit = rng.uniform(-0.25, 0.25, N + 1)
    lens = np.where(np.arange(N + 1) % 2 == 0, 1.5, 0.75) * rng.uniform(0.7, 1.25, N + 1)
    kk = k.astype(np.int64) % N
    tk = tk + jit[kk] * (2 * math.pi / N)
    dperp = np.abs(np.sin(th - tk)) * r
    r0 = 1.35 * rad
    r1 = r0 + lens[kk] * rad
    along = np.clip((r - r0) / np.maximum(r1 - r0, 1e-3), 0, 1)
    onray = (r > r0) & (r < r1)
    brk = (np.sin(r / rad * 9.0 + kk * 1.3) > -0.55)
    pr = 0.55 * kpx * (1.0 - 0.6 * along)
    ray = np.clip(0.5 * pr + 0.5 - dperp, 0, 1) * onray * brk * (1.0 - along) ** 0.6 * 0.75 * sky
    ink = np.maximum(ring, ray)
    icol = s2l(INK)[None, None] * 0.35 + s2l(INK_LIGHT)[None, None] * 0.65
    return out * (1 - 0.80 * ink[..., None]) + icol * (0.80 * ink[..., None])


def render_file(shot, frame, scale, outscale=None, tag=''):
    import render_ink as RI
    R = RI.load_cache(RI.cache_path(shot, frame, scale))
    plate = None
    if shot == 'illum':
        pl = look.find_frame(os.path.join(ROOT, 'renders', 'dawn_C'), frame)
        im = cv2.imread(pl)[..., ::-1].astype(np.float32) / 255.0
        H, W = R['A'].shape[:2]
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        plate = look.srgb_to_linear(im)
    S = RI.Shot(shot)
    img, dbg = compose(R, B=S.B, plate=plate, CR=S.CR)
    H, W = img.shape[:2]
    if outscale is not None:
        W2, H2 = int(round(1920 * outscale)), int(round(804 * outscale))
        if (W2, H2) != (W, H):
            img = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)
    d = os.path.join(OUT, shot + tag)
    os.makedirs(d, exist_ok=True)
    look.save_png(look.frame_path(d, frame), img)
    # the world-anchored ink coverage (for the boiling check: reproject frame to frame through the camera)
    di = os.path.join(OUT, shot + tag + '_ink')
    os.makedirs(di, exist_ok=True)
    cv2.imwrite(look.frame_path(di, frame), np.clip(dbg['ink'] * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return img, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shot', default='scroll')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--out', type=float, default=None, help='output scale (default: the AOV resolution)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--salt', type=int, default=0, help='re-deal the strokes (boiling control)')
    a = ap.parse_args()
    global SALT
    SALT = a.salt
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1))
    for f in frames:
        t0 = time.time()
        render_file(a.shot, f, a.scale, a.out, a.tag)
        print(f'ink {a.shot} {f} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
