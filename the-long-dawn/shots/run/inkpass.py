"""C · THE LIVING INK ILLUSTRATION (R12): the ink pass. Turns a render_ink.py AOV cache into a frame drawn in the
old book's manner: parchment (fixed to the page), iron-gall hatching down the fall line, outlines at depth breaks,
the cloud sea in contour lines and stipple, a sepia wash in the shadows, fires in full colour with gold leaf.

TEMPORAL COHERENCE (the risk): every mark is a function of WORLD position, never of the screen.
  * Hatch strokes hang on seeds of a NESTED jittered grid in world x-z (levels 2^L x S0 m): each level keeps
    every seed of the level above and adds new ones between, so as the camera nears a slope strokes only ever
    fade in between the strokes already there (a tonal-art-map hierarchy, in world space).
  * A stroke's direction is the fall line of the height field smoothed at the level's own scale; its length is
    proportional to that scale; its pen width is in screen pixels (a pen, not a texture).
  * Density follows tone by a fixed rank per seed, so a stroke only comes or goes when the light on its slope
    changes (fire light enters the tone WITHOUT flicker).
  * Cloud contours are iso-height lines at nested intervals (every other line of a level is the level above).

  python inkpass.py --shot scroll --frames 140 --scale 0.5            # cache -> renders/run_c_tests/<shot>/
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


# ------------------------------------------------------------ hatching kernel ---

@njit(inline='always', fastmath=True)
def _kv(Px, Py, Pz, tx, tz, cam):
    """Screen pixels per metre along the horizontal world direction (tx, 0, tz) at world point P."""
    dx = Px - cam[0]
    dy = Py - cam[1]
    dz = Pz - cam[2]
    Xc = dx * cam[6] + dy * cam[7] + dz * cam[8]
    Yc = dx * cam[9] + dy * cam[10] + dz * cam[11]
    Zc = dx * cam[3] + dy * cam[4] + dz * cam[5]
    if Zc < 1e-3:
        return 0.0
    tcx = tx * cam[6] + tz * cam[8]
    tcy = tx * cam[9] + tz * cam[11]
    tcz = tx * cam[3] + tz * cam[5]
    sx = cam[12] / Zc * (tcx - Xc * tcz / Zc)
    sy = cam[12] / Zc * (tcy - Yc * tcz / Zc)
    return math.sqrt(sx * sx + sy * sy)


@njit(inline='always', fastmath=True)
def _family(x, z, Py, gx, gz, L, alpha, sl, T, cam, salt, rot, kpx, wmul, tgain, t0, CR):
    """Coverage (0-1) of one stroke family at world (x, z). (gx, gz) = unit downhill direction; the family's
    strokes run at `rot` radians from it."""
    cr = math.cos(rot)
    sr = math.sin(rot)
    ux = gx * cr - gz * sr
    uz = gx * sr + gz * cr
    vx = -uz
    vz = ux
    kv = _kv(x, Py, z, vx, vz, cam)
    if kv <= 0.0:
        return 0.0
    sL = S0 * 2.0 ** L
    ell = 1.45 * sl                       # half length (m, in plan)
    R = int(math.ceil(1.2 * ell / sL + 0.5))
    if R > 5:
        R = 5
    ci = int(math.floor(x / sL))
    cj = int(math.floor(z / sL))
    hx = 0.5 * sL * (abs(vx) + abs(vz))   # a cell's half extent across / along the stroke
    hu = 0.5 * sL * (abs(ux) + abs(uz))
    tol = 3.0 * kpx / kv
    cov = 0.0
    for dj in range(-R, R + 1):
        for di in range(-R, R + 1):
            ccx = (ci + di + 0.5) * sL - x
            ccz = (cj + dj + 0.5) * sL - z
            if abs(ccx * vx + ccz * vz) > hx + tol + 0.06 * sl:
                continue
            if abs(ccx * ux + ccz * uz) > hu + 1.25 * ell:
                continue
            px, pz, birth = seed_of(L, ci + di, cj + dj, salt)
            w_lod = 1.0 if birth > L else 1.0 - alpha
            if w_lod <= 0.0:
                continue
            ex = x - px
            ez = z - pz
            v = ex * vx + ez * vz
            if abs(v) * kv > 3.0 * kpx:
                continue
            u = ex * ux + ez * uz
            r1 = h01_3(ci + di, cj + dj, L + 7, salt + 13)
            r2 = h01_3(ci + di, cj + dj, L + 9, salt + 17)
            lenk = 0.75 + 0.45 * r2
            tt = (u / (ell * lenk) + 1.0) * 0.5          # 0 at the uphill end, 1 at the downhill end
            if tt <= 0.0 or tt >= 1.0:
                continue
            # rank: a stroke is drawn once the tone is darker than its rank
            rank = h01_3(ci + di, cj + dj, L + 3, salt + 5)
            pres = (T * tgain - rank * 0.92) / 0.10
            if pres <= 0.0:
                continue
            if pres > 1.0:
                pres = 1.0
            # the pen: pressure at the top, lifting toward the bottom; a slight tremor
            taper = smoothstep(0.0, 0.12, tt) * (1.0 - 0.85 * tt ** 1.8)
            wob = 0.05 * sl * math.sin(6.2831853 * (tt * (1.1 + r1) + r1 * 3.7))
            w = kpx * wmul * (0.6 + 1.2 * min(T, 1.2)) * taper
            if w < 0.08 * kpx:
                continue
            d = abs(v + wob) * kv
            c = 0.5 * w + 0.5 - d
            if c <= 0.0:
                continue
            if c > 1.0:
                c = 1.0
            k = 1.0
            if w < 1.0:
                k = w                           # thin: fainter, not thinner than a pixel
            c = c * k * pres * w_lod * (0.82 + 0.18 * r1)
            if c > cov:
                cov = c
    return cov


@njit(parallel=True, fastmath=True, cache=True)
def hatch(A, T, cam, CR, prm, out):
    """A: AOV (H, W, 16) target-space; T: darkness (H, W); out: (H, W, 3) = fall-line cov, cross cov, level."""
    H, W = T.shape
    kpx = prm[0]            # ss pixels per 1920-px
    sp_px = prm[1]          # target stroke spacing (1920-px)
    t = prm[2]
    f = cam[12]
    for j in prange(H):
        for i in range(W):
            out[j, i, 0] = 0.0
            out[j, i, 1] = 0.0
            out[j, i, 2] = 0.0
            surf = int(A[j, i, 0] + 0.5)
            if surf == 3 or surf == 2:
                continue
            if T[j, i] <= 0.02:
                continue
            x = A[j, i, 2]
            Py = A[j, i, 3]
            z = A[j, i, 4]
            dist = A[j, i, 1]
            fpx = dist / f                       # m per ss pixel
            lam = math.log2(max(sp_px * kpx * fpx / S0, 1.0))
            L = int(math.floor(lam))
            alpha = lam - L
            if L > LMAX - 1:
                L = LMAX - 1
                alpha = 1.0
            sl = S0 * 2.0 ** lam
            # fall line at the level's own scale (blend of L and L+1: continuous in lam)
            gx0, gz0 = IA.grad_at(x, z, S0 * 2.0 ** L, CR, surf, t)
            gx1, gz1 = IA.grad_at(x, z, S0 * 2.0 ** (L + 1), CR, surf, t)
            gx = gx0 * (1.0 - alpha) + gx1 * alpha
            gz = gz0 * (1.0 - alpha) + gz1 * alpha
            gm = math.sqrt(gx * gx + gz * gz)
            if gm < 1e-6:
                continue
            steep = smoothstep(0.06, 0.35, gm)           # no fall line on the flat
            ux = -gx / gm
            uz = -gz / gm
            Tt = T[j, i]
            so = int(prm[5])
            c1 = _family(x, z, Py, ux, uz, L, alpha, sl, Tt * steep, cam, 11 + so, 0.0, kpx, 1.0, 1.25, t, CR)
            c2 = 0.0
            if Tt > 0.85:
                c2 = _family(x, z, Py, ux, uz, L, alpha, sl * 0.9, (Tt - 0.83) * 2.6, cam, 29 + so, 0.95, kpx, 0.8,
                             1.0, t, CR)
            out[j, i, 0] = c1
            out[j, i, 1] = c2
            out[j, i, 2] = lam


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
            g += 0.55 * gnoise2(u, v, 404) + 0.35 * gnoise2(u * 2.3, v * 2.1, 406) + 0.45 * gnoise2(x * 30.0, y * 30.0, 405)
            out[i, j, 0] = tt
            out[i, j, 1] = g


# ------------------------------------------------------------ compose ---

def darkness(A, fire_steady):
    """Illustration value, decided like an illustrator: the side turned from the light is shadow (dense
    hatching), cast shadow adds a middle tone, bare rock a light texture, lit snow is bare paper.
    0 = paper, 1 = full hatching, >0.85 cross-hatching."""
    surf = A[..., 0]
    snow = np.clip(A[..., 9], 0, 1)
    ndl = A[..., 10]
    shd = np.clip(A[..., 11], 0, 1)
    sf = np.clip((0.40 - ndl) / 0.60, 0, 1)
    sf = sf * sf * (3 - 2 * sf)                      # the shadow side
    cs = (1 - shd) * (1 - sf)                        # cast shadow on the lit side
    rock = 1 - snow
    T = 0.80 * sf + 0.28 * cs + 0.22 * rock
    T = T * (1 - 0.9 * fire_steady)                  # fire light clears the ink (it is gilded instead)
    tr = np.clip(A[..., 12], 0, 1)
    ap = 0.30 + 0.70 * tr ** 0.8                     # aerial perspective: far strokes sparse and light
    T = np.clip(T, 0, 1.3) * ap
    T[surf >= 2.5] = 0.0
    # cloud value: lit tops bare, troughs and shadows dark
    cl = A[..., 14]
    csh = np.clip(A[..., 11], 0, 1)
    Vc = np.clip(0.18 + 0.75 * cl * (0.4 + 0.6 * csh), 0, 1)
    Tc = np.clip(1.0 - Vc, 0, 1) * (0.35 + 0.65 * tr)
    Tc[surf != 2] = 0.0
    return T.astype(np.float32), Tc.astype(np.float32), ap


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


def outlines(A, kpx):
    """Ink lines at depth breaks, on the near side: heavier for nearer and bigger breaks; the skyline always.
    Width in 1920-px: ~2.3 near, ~0.8 far; far lines also lighter."""
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
    out = np.zeros(d.shape, np.float32)
    _splat_lines(b.astype(np.float32), w.astype(np.float32), dens.astype(np.float32), out)
    return out


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


def gold_leaf(H, W, kpx, bl, frame, fcam=None, fps=24.0):
    """Gold laid on the page around each lit fire: an irregular aureole that blooms outward at the catch (like
    leaf pressed onto size), fixed to the fire, not to the screen. Returns coverage, sheen and rim (H, W)."""
    cov = np.zeros((H, W), np.float32)
    sheen = np.zeros((H, W), np.float32)
    rim = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = fcam if fcam is not None else 1920.0 * kpx / 2.0 / math.tan(math.radians(22.0))
    for b in bl:
        sx, sy, z, ign, size, inten, light, idx, kind = b
        if light <= 0 or z <= 1.0:
            continue
        u = (frame - ign)                          # frames since the catch
        bloom = 1.0 - math.exp(-max(u, 0.0) / 4.0)
        over = 0.12 * math.exp(-((u - 7.0) / 5.0) ** 2)
        ppm = f / z
        Hf = 6.5 * flame_scale(z)
        R = max(0.85 * Hf * ppm, 8.0 * kpx) * (bloom + over)
        R = min(R, 150.0 * kpx)
        if R < 0.5:
            continue
        cy = sy - 0.40 * Hf * ppm
        x0, x1 = int(max(sx - 1.6 * R, 0)), int(min(sx + 1.6 * R + 1, W))
        y0, y1 = int(max(cy - 1.6 * R, 0)), int(min(cy + 1.6 * R + 1, H))
        if x1 <= x0 or y1 <= y0:
            continue
        dx = (xx[y0:y1, x0:x1] - sx) / R
        dy = (yy[y0:y1, x0:x1] - cy) / R
        rr = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(dy, dx)
        # the leaf's torn edge, fixed to the fire
        edge = 1.0 + 0.10 * np.sin(3 * ang + idx * 1.7) + 0.06 * np.sin(7 * ang + idx * 0.3) \
            + 0.035 * np.sin(13 * ang + idx) + 0.02 * np.sin(29 * ang + 2.0 * idx)
        aa = 1.2 / R
        fpx = Hf * ppm * max(size, 0.3) / kpx               # flame height on the page (1920-px)
        near_fade = min(max((110.0 - fpx) / 60.0, 0.0), 1.0)
        c = np.clip((edge - rr) / aa, 0, 1) * near_fade
        cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1], c)
        # burnish: brighter toward the fire, a sweep of light across the leaf
        sh = np.clip(1.0 - rr / edge, 0, 1) ** 1.5 * 0.8 + 0.25 * np.clip(0.5 + 0.5 * np.cos(2.0 * (ang - 0.9)), 0, 1)
        sheen[y0:y1, x0:x1] = np.maximum(sheen[y0:y1, x0:x1], sh * c)
        rm = np.exp(-((edge - rr) / (1.6 * aa)) ** 2) * (rr < edge + aa) * near_fade
        rim[y0:y1, x0:x1] = np.maximum(rim[y0:y1, x0:x1], rm)
    return cov, sheen, rim


SALT = 0                # a non-zero offset re-deals every stroke (the boiling control)
FLAME_SCALE = 5.0      # far beacons are drawn larger than life (an illustrator's licence): ~32 m flames


def flame_scale(z):
    """True size near (her own fire in the reveal), up to FLAME_SCALE for beacons kilometres off."""
    u = min(max((z - 300.0) / 2200.0, 0.0), 1.0)
    return 1.0 + (FLAME_SCALE - 1.0) * u * u * (3 - 2 * u)


def fire_target(R, B):
    """The fires, drawn in the target camera at the AOV resolution: flames (fire2) and smoke only, no additive
    halos (the gold leaf is the glow). Returns premultiplied emission and coverage (smoke, soot)."""
    import rcam as RC
    import fire2 as F2
    import beacons as BC
    c = R['cam']
    A = R['A']
    H, W = A.shape[:2]
    cam = RC.RCam(c['pos'], c['yaw'], c['pitch'], c['roll'], c['hfov'], W, H)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    ray = cam.ray(u + 0.5, v + 0.5)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    zt = (A[..., 1] * (ray @ cam.fwd)).astype(np.float32)
    zt[A[..., 1] > 1e8] = 1e9
    frame = R['frame']
    t = frame / 24.0
    g = 0.5
    Eb = np.zeros((H, W, 3), np.float32)
    Eg = np.full((H, W, 3), g, np.float32)
    kpx = W / 1920.0
    order = np.argsort([-np.linalg.norm(b[:3] - cam.pos) for b in B])
    for img in (Eb, Eg):
        zb = zt.copy()
        for i in order:
            b = B[i]
            size, inten, light = BC.env(frame, b[3])
            if size <= 0:
                continue
            base = np.array([b[0], b[1] + 1.6, b[2]])
            sx, sy, z = cam.project(base)
            if z <= 1.0 or sx < -300 or sx > W + 300 or sy < -300 or sy > H + 300:
                continue
            ppm = cam.f / z
            zbias = max(3.0, 0.004 * z)
            fs_ = flame_scale(z)
            Hf = 6.5 * fs_ * size
            fl = 1.0 + 0.0 * t
            if z < 6000.0:
                BC._smoke(b).render(img, zb, cam, t, base + np.array([0, 0.8 * Hf, 0]), 2.5 * light,
                                    albedo=0.25, amb=(0.004, 0.005, 0.009), zbias=zbias)
            F2.flame(img, zb, cam, base, Hf, 1.7 * fs_, t, seed=int(b[5]) * 7 + 3,
                     I=30.0 * inten, lean=0.25 * size, zbias=zbias, tongues=7)
            if Hf * ppm < 3.0 * kpx:
                F2.glow(img, zb, sx, sy - 0.3 * Hf * ppm, 0.95 * kpx, 6.5 * light * kpx * kpx, z=z, zbias=zbias,
                        col=np.array([1.0, 0.62, 0.26]))
    alpha = np.clip(1.0 - (Eg - Eb).mean(axis=2) / g, 0.0, 1.0)
    return Eb, alpha


def beacon_rows(R, B):
    """Screen position and ignition envelope of every beacon for this frame (as render_ink's bl rows)."""
    import beacons as BC
    c = R['cam']
    rows = []
    for b in B:
        size, inten, light = BC.env(R['frame'], b[3])
        d = np.array([b[0], b[1] + 1.6, b[2]]) - c['pos']
        z = float(d @ c['fwd'])
        zz = max(z, 1e-6)
        rows.append([c['cx'] + c['f'] * float(d @ c['right']) / zz, c['cy'] - c['f'] * float(d @ c['up']) / zz, z,
                     b[3], size, inten, light, b[5], b[4]])
    return np.array(rows, np.float64).reshape(-1, 9)


def compose(R, prm=None, B=None, plate=None, CR=None):
    A = R['A']
    H, W = A.shape[:2]
    kpx = W / 1920.0
    cam = cam_vec(R['cam'])
    t = R['cam']['t']
    fire_st = np.clip(A[..., 13] / 0.05, 0, 1) ** 0.7              # steady firelight (no flicker)
    T, Tc, ap = darkness(A, fire_st)
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
    hat = np.zeros((H, W, 3), np.float32)
    hatch(A, T, cam, RN.CR if CR is None else CR, p, hat)
    gs = screen_grad(A[..., 5], np.rint(A[..., 0]))
    cl = np.zeros((H, W, 2), np.float32)
    cloud(A, Tc, gs, cam, p, cl)
    ol = outlines(A, kpx)
    pp = np.zeros((H, W, 2), np.float32)
    paper(H, W, kpx, pp)
    if B is not None and len(B):
        R = dict(R)
        R['E'], R['a'] = fire_target(R, B)
    if plate is not None:
        R = dict(R)
        R['E'] = np.zeros_like(R['E'])          # the dawn smoke's colour comes from the plate where the light is
    # ---- the page
    tt = np.clip(pp[..., 0], 0, 1)[..., None]
    base = s2l(PAPER)[None, None] * (1 - 0.55 * tt) + s2l(PAPER_DARK)[None, None] * (0.55 * tt)
    base = base * (1.0 + 0.06 * pp[..., 1])[..., None]
    # ---- sepia wash in the shadows (a brush, soft-edged)
    wash = np.clip((T - 0.35) / 0.6, 0, 1) * 0.30 + np.clip(Tc - 0.45, 0, 1) * 0.25
    wash = cv2.GaussianBlur(wash.astype(np.float32), (0, 0), 1.6 * kpx)
    wcol = base * (s2l(WASH) / s2l(PAPER))[None, None]
    rgb = base * (1 - wash[..., None]) + wcol * wash[..., None]
    # ---- gold where the firelight falls on the land (steady; its edge is the light's own falloff)
    g_land = np.clip((fire_st - 0.25) / 0.5, 0, 1) * (A[..., 0] < 1.5)
    g_land = cv2.GaussianBlur(g_land.astype(np.float32), (0, 0), 0.8 * kpx)
    gl, sh, grim = gold_leaf(H, W, kpx, R['bl'] if B is None else beacon_rows(R, B), R['frame'], R['cam']['f'])
    gold_a = np.clip(np.maximum(gl, 0.75 * g_land), 0, 1)
    gcol = s2l(GOLD)[None, None] * (1 - sh[..., None]) + s2l(GOLD_HI)[None, None] * sh[..., None]
    gcol = gcol * (0.85 + 0.3 * np.clip(pp[..., 1] * 0.5 + 0.5, 0, 1))[..., None]
    rgb = rgb * (1 - gold_a[..., None]) + gcol * gold_a[..., None]
    # a fine burnished line at the leaf's edge
    rgb = rgb * (1 - 0.55 * grim[..., None]) + s2l(GOLD_LO)[None, None] * (0.55 * grim[..., None])
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
        ink_a = ink_a * (1 - 0.35 * fill)                          # coloured ground: the line work lightens a touch
    rgb = rgb * (1 - ink_a[..., None]) + icol * ink_a[..., None]
    if fill is not None:
        rgb = sun_leaf(R, rgb, pp, kpx)
    # ---- the fires: full colour, laid on like gouache (opacity from their own light)
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


def illum_fill(R, plate, rgb, pp, kpx):
    """THE ILLUMINATION: wherever the sun's light touches, the page takes the dawn's colour like a watercolour
    wash (translucent: the paper and the line work stay). The wash is a wet front spreading from the sun, faster
    along the horizon than up the sky, its edge lobed and bled along the paper's grain, with a faint tide-line
    where it dried. It lays on the sky, on the cloud sea where the sun reaches it, and on the lit rims; the faces
    turned from the sun stay ink. plate: the dawn in colour (display-referred, linearised, H x W x 3)."""
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
    ang = np.sqrt((daz / 1.25) ** 2 + np.maximum(-delv, 0.0) ** 2 + (np.maximum(delv, 0.0) * 0.8) ** 2)
    # lobes of the wet front: slow noise in the frame's angular coordinates (the camera barely moves)
    lob = np.zeros((H, W), np.float32)
    small = np.zeros((max(H // 8, 2), max(W // 8, 2)), np.float32)
    ys, xs = np.mgrid[0:small.shape[0], 0:small.shape[1]].astype(np.float32)
    for k, (fq, amp) in enumerate(((0.9, 1.0), (2.1, 0.45), (4.3, 0.2))):
        ph = 1.7 * k + 0.3
        small += amp * np.sin(xs / small.shape[1] * 6.283 * fq + ph + np.sin(ys / small.shape[0] * 6.283 * fq * 0.8 + 2 * ph))
    lob = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    u = np.clip((frame - 2402.0) / 320.0, 0, 1)                # C24: complete as the line settles home (local 320)
    reach = 1.0 + 34.0 * u ** 1.4                           # degrees reached so far (the frame is ~23 deg to a side)
    front = reach * (1.0 + 0.16 * lob) - ang
    # what the light touches: the sky; the cloud sea by its own sun-lit term; the lit rims of the land
    cl = np.clip(A[..., 14] * (0.35 + 0.65 * np.clip(A[..., 11], 0, 1)) * gain / 1.1, 0, 1)
    litl = np.clip(A[..., 10], 0, 1) * np.clip(A[..., 11], 0, 1) * gain
    pb = cv2.GaussianBlur(plate, (0, 0), 6.0 * kpx)
    Y = pb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    rim = np.clip((Y - 0.10) / 0.20, 0, 1)
    touch = np.where(surf >= 2.5, 1.0, np.where(surf == 2, 0.35 + 0.65 * cl, np.maximum(litl * 1.5, rim)))
    Mraw = np.clip(front / 7.0, 0, 1) * touch
    # the edge bleeds along the paper (page-fixed grain), never a clean mask edge
    M = np.clip((Mraw - 0.25 + 0.14 * pp[..., 1]) / 0.35, 0, 1)
    M = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 1.2 * kpx)
    # translucent, softened colour: the dawn's hue with its saturation eased, its value compressed
    chroma = pb / np.maximum(Y, 1e-4)[..., None]
    chroma = np.clip(chroma, 0, 4) ** 0.85
    chroma = chroma / np.maximum(chroma @ np.array([0.2126, 0.7152, 0.0722], np.float32), 1e-4)[..., None]
    val = np.clip(np.sqrt(np.clip(Y, 0, 1)) * 1.2, 0.40, 0.98)
    tint = chroma * val[..., None]
    gran = 0.94 + 0.12 * np.clip(pp[..., 1] * 0.5 + 0.5, 0, 1)            # pigment settling in the tooth
    edge = np.clip(front / 7.0, 0, 1)
    tide = 1.0 - 0.10 * np.exp(-((edge - 0.12) / 0.07) ** 2) * (surf >= 2.5)   # a faint dried edge in the sky
    paper_n = rgb / s2l(PAPER)[None, None]
    washed = paper_n * (0.25 + 0.75 * tint) * (gran * tide)[..., None]
    out = rgb * (1 - M[..., None]) + washed * M[..., None]
    # gold leaf where the light is strongest: the sun's path on the cloud sea, laid as the wash arrives there
    near_sun = np.clip((10.0 - ang) / 6.0, 0, 1)
    gl = np.clip((Y - 0.32) / 0.25, 0, 1) * near_sun * (surf == 2) * M
    gl = np.clip((gl - 0.35 + 0.18 * pp[..., 1]) / 0.25, 0, 1)
    gl = cv2.GaussianBlur(gl.astype(np.float32), (0, 0), 0.7 * kpx)
    sheen = np.clip(1.0 - ang / 10.0, 0, 1)
    gcol = s2l(GOLD)[None, None] * (1 - sheen[..., None]) + s2l(GOLD_HI)[None, None] * sheen[..., None]
    out = out * (1 - 0.85 * gl[..., None]) + gcol * (0.85 * gl[..., None])
    return M, out


def sun_leaf(R, rgb, pp, kpx):
    """The sun as a burnished gold-leaf disc, tooled with faint rings and inked at its rim, cut by the drawn
    skyline as it rises."""
    H, W = rgb.shape[:2]
    sx, sy, f = _sun_px(R)
    rad = f * math.tan(math.radians(0.75))
    if sx < -3 * rad or sx > W + 3 * rad:
        return rgb
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / rad
    sky = (R['A'][..., 0] >= 2.5).astype(np.float32)
    sky = cv2.GaussianBlur(sky, (0, 0), 0.6 * kpx)
    c = np.clip((1.0 - r) * rad / 1.2, 0, 1) * sky
    ang = np.arctan2(yy - sy, xx - sx)
    sh = np.clip(0.55 + 0.45 * np.cos(ang + 2.3), 0, 1) * np.clip(1.0 - r, 0, 1) ** 0.4    # burnish toward the upper left
    tool = 1.0 - 0.07 * (0.5 + 0.5 * np.cos(r * 6.283 * 4.0)) * (r < 0.85)                  # tooled rings
    gcol = s2l(GOLD)[None, None] * (1 - sh[..., None]) + s2l(GOLD_HI)[None, None] * sh[..., None]
    gcol = gcol * (tool * (0.92 + 0.16 * np.clip(pp[..., 1] * 0.5 + 0.5, 0, 1)))[..., None]
    out = rgb * (1 - c[..., None]) + gcol * c[..., None]
    ring = np.exp(-((r - 1.0) * rad / (0.8 * kpx)) ** 2) * sky
    out = out * (1 - 0.7 * ring[..., None]) + s2l(INK)[None, None] * (0.7 * ring[..., None])
    return out


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
