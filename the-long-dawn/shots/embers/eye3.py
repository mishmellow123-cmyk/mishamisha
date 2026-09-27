"""EMBERS-C: E12 THE EYE ONTO NOTHING (C9, C frames 1920-2079). Our own design, from the book's words, not the
films' image: "rimmed with fire ... glazed, yellow as a cat's, watchful and intent, and the black slit of its pupil
opened on a pit, a window into nothing."

    1920-1960  THE STORM   (MAP burns the Deep's page open onto it, 1920-1991) dark cloud over the forges, lit red
                           from below by their throats; it is drawn in toward one place above the towers (log-polar
                           inflow: it gathers, it never turns: no spiral)
    1940-1975  the rim     a ring of fire forms in the cloud, licking outward; every tower leans toward it
    1955-1995  the iris    fibres of flame, amber-yellow and glazed, drawn slowly inward to a closed seam
    2000       THE SLIT    (bar 26 b1, C's loudest moment) the seam opens onto empty black: its white-hot lips part,
                           the inner wall of the pit shows for a hand's breadth and then nothing; no glow, no star,
                           no one behind it. It does not blink. It holds.

The sky (cloud, rim, iris, slit) is a screen-space shader of a plane facing the lens (behind the towers: they
occlude it through the splat occluder). The towers, their embers and smoke are scene_b's, leaning in.
"""
import math

import numpy as np
from numba import njit, prange

from core import perlin3, smoothstep, smootherstep

T0 = 1920
T_RIM0, T_RIM1 = 1938, 1972
T_IRIS0, T_IRIS1 = 1952, 1994
T_SLIT = 2000
T_END = 2080
EYE_Y = 46.0          # above the ground (the forges' crowns stand at ~27-36)
EYE_R = 8.8           # the iris radius (world units)


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
def _fire(T, out):
    """temperature 0..1 -> fire colour (deep red, orange, amber, yellow-white); unit peak"""
    if T < 0.3:
        k = T / 0.3
        out[0] = 0.55 * k
        out[1] = 0.08 * k
        out[2] = 0.02 * k
    elif T < 0.6:
        k = (T - 0.3) / 0.3
        out[0] = 0.55 + 0.45 * k
        out[1] = 0.08 + 0.34 * k
        out[2] = 0.02 + 0.06 * k
    elif T < 0.85:
        k = (T - 0.6) / 0.25
        out[0] = 1.0
        out[1] = 0.42 + 0.32 * k
        out[2] = 0.08 + 0.14 * k
    else:
        k = (T - 0.85) / 0.15
        if k > 1.0:
            k = 1.0
        out[0] = 1.0
        out[1] = 0.74 + 0.2 * k
        out[2] = 0.22 + 0.5 * k


@njit(parallel=True, fastmath=True, cache=True)
def _sky(out, VIS, CR, CP, f, W, H, C, n, eu, es, R, t, storm, rim, iris, slit_w, flare, red):
    """out (H, W, 3) += the sky. CR: camera rows (3, 3) world->camera; CP camera position; f focal px;
    C eye centre, n its normal (toward the lens), eu / es its up / side axes; R the iris radius.
    Angular noise is taken on circles (cos, sin) so nothing has a seam; time only advects along log r (the
    storm and the fibres are drawn INWARD; nothing turns)."""
    for iy in prange(H):
        c = np.zeros(3)
        for ix in range(W):
            vis = VIS[iy, ix]
            if vis <= 0.0:
                continue
            xc = (ix - (W - 1) * 0.5) / f
            yc = ((H - 1) * 0.5 - iy) / f
            dx = CR[0, 0] * xc + CR[1, 0] * yc + CR[2, 0]
            dy = CR[0, 1] * xc + CR[1, 1] * yc + CR[2, 1]
            dz = CR[0, 2] * xc + CR[1, 2] * yc + CR[2, 2]
            dn = dx * n[0] + dy * n[1] + dz * n[2]
            if dn > -1e-6:
                continue
            s = ((C[0] - CP[0]) * n[0] + (C[1] - CP[1]) * n[1] + (C[2] - CP[2]) * n[2]) / dn
            if s <= 0.0:
                continue
            px = CP[0] + s * dx - C[0]
            py = CP[1] + s * dy - C[1]
            pz = CP[2] + s * dz - C[2]
            lx = (px * es[0] + py * es[1] + pz * es[2]) / R
            ly = (px * eu[0] + py * eu[1] + pz * eu[2]) / R
            r = math.sqrt(lx * lx + ly * ly) + 1e-6
            ca = lx / r
            sa = ly / r
            lr = math.log(r)
            cr = 0.0
            cg = 0.0
            cb = 0.0
            # ---------------------------------------------------------------- the storm
            g = _ss(4.0, 1.4, r) * storm
            ux = lx * 0.3
            uy = ly * 0.3 - t * 0.004
            wx = _fbm(ux + 3.1, uy, t * 0.006, 2)
            wy = _fbm(ux - 7.7, uy + 1.3, t * 0.006 + 4.0, 2)
            cl_far = _fbm(ux + 0.9 * wx, uy + 0.9 * wy, t * 0.009, 4)
            zk = 0.3 * math.exp(0.006 * (t - 1920.0))                                  # drawn inward: the
            cl_in = _fbm(lx * zk + 5.0 + 0.6 * wx, ly * zk + 0.6 * wy, 7.0 + t * 0.004, 4)   # texture contracts
            cl = cl_far * (1.0 - g) + cl_in * g
            dens = _ss(-0.3, 0.4, cl)
            below = _ss(0.6, -3.0, ly)                     # the forges' red from below
            L = dens * (0.03 + 0.2 * below * below)
            cr += L * 1.0
            cg += L * 0.13
            cb += L * 0.04
            # the Eye's own light on the nearest cloud: its edges catch fire-orange
            eyel = rim * _ss(2.2, 1.05, r) * _ss(0.9, 1.1, r)
            E2 = dens * eyel * eyel * 0.5
            cr += E2
            cg += E2 * 0.36
            cb += E2 * 0.08
            # ------------------------------------------------------------------ the rim of fire (ragged)
            if rim > 0.0:
                lick = _fbm(ca * 2.2 + 11.0, sa * 2.2, (lr - t * 0.02) * 5.0, 3)
                lick2 = _fbm(ca * 6.0 - 3.0, sa * 6.0, (lr - t * 0.03) * 9.0 + 7.0, 2)
                edge = 1.0 + 0.1 * lick
                reach = 0.05 + 0.3 * _ss(-0.05, 0.5, lick + 0.4 * lick2)
                inner = _ss(edge - 0.07, edge - 0.01, r)
                outer = _ss(edge + reach, edge, r)
                F = rim * inner * outer * (0.55 + 0.6 * lick2 + 0.3 * lick)
                if F > 0.0:
                    T = 0.35 + 0.45 * outer * _ss(edge + reach, edge + 0.02, r) + 0.25 * lick2
                    if T > 1.0:
                        T = 1.0
                    _fire(T, c)
                    k = F * 2.4 * (1.0 + 0.9 * flare)
                    cr += c[0] * k
                    cg += c[1] * k
                    cb += c[2] * k
            # --------------------------------------------------------------------- the iris
            if iris > 0.0 and r < 1.06:
                # the stroma: wavy fibres (the angle itself wanders), crypts, a collarette; drawn slowly inward
                fl = lr + t * 0.018
                wav = 0.55 * _fbm(ca * 1.5 + 2.0, sa * 1.5, fl * 2.2, 2)
                ca2 = math.cos(wav) * ca - math.sin(wav) * sa
                sa2 = math.sin(wav) * ca + math.cos(wav) * sa
                fib = _fbm(ca2 * 6.0 + 21.0, sa2 * 6.0, fl * 1.3, 3)
                fib2 = _fbm(ca2 * 15.0 - 5.0, sa2 * 15.0 + 2.0, fl * 2.4, 2)
                crypt = _ss(0.12, 0.34, _fbm(lx * 3.2 + 9.0, ly * 3.2, fl * 0.6, 2))   # dark hollows
                strand = _ss(-0.35, 0.45, fib + 0.45 * fib2) * (1.0 - 0.75 * crypt)
                limb = _ss(1.04, 0.84, r) * (1.0 - 0.5 * math.exp(-((r - 0.85) / 0.06) ** 2))
                colla = math.exp(-((r - 0.42 - 0.04 * fib) / 0.1) ** 2)                 # the collarette
                heat = 0.38 + 0.45 * _ss(0.95, 0.2, r) + 0.35 * colla
                strand = strand * (0.55 + 0.45 * _ss(0.12, 0.38, r)) + 0.35 * colla * _ss(0.4, 0.1, r)
                T = 0.18 + 0.62 * strand * heat + 0.12 * heat
                if T > 1.0:
                    T = 1.0
                _fire(T, c)
                I = iris * limb * (0.12 + 1.9 * strand * heat) * (1.0 + 0.6 * flare)
                cr += c[0] * I
                cg += c[1] * I
                cb += c[2] * I
            # --------------------------------------------------------------------- the slit
            if iris > 0.0 and r < 1.0:
                h0 = 0.86
                if ly > -h0 and ly < h0:
                    q = 1.0 - (ly / h0) * (ly / h0)
                    rag = 1.0 + 0.22 * _fbm(ly * 7.0, t * 0.05, 3.3, 3)
                    w = slit_w * q ** 0.6 * rag + 0.003 * q
                    ax = abs(lx)
                    if ax < w:
                        # the pit: its inner wall catches the lips' light for a hand's breadth, then nothing
                        dep = (w - ax) / (0.02 + 0.22 * slit_w)
                        wall = math.exp(-dep * 3.2) * _ss(0.0, 0.02, slit_w) * iris
                        cr = wall * 0.45
                        cg = wall * 0.12
                        cb = wall * 0.03
                    else:
                        lipn = _fbm(ly * 16.0 + 40.0, t * 0.08, 1.0, 2)
                        lw = (0.005 + 0.022 * slit_w) * (0.7 + 0.9 * _ss(-0.4, 0.4, lipn))
                        lip = math.exp(-((ax - w) / lw) ** 2) * q ** 0.8 * (0.55 + 0.9 * _ss(-0.3, 0.5, lipn))
                        L = lip * iris * (1.6 + 3.0 * flare)
                        cr += L * 1.0
                        cg += L * 0.74
                        cb += L * 0.36
            out[iy, ix, 0] += cr * vis
            out[iy, ix, 1] += cg * vis
            out[iy, ix, 2] += cb * vis


def state(t):
    """(storm, rim, iris, slit half-width (iris radii), flare, red) at C frame t"""
    storm = float(smoothstep(T0 - 10, T0 + 30, t))
    rim = float(smoothstep(T_RIM0, T_RIM1, t))
    iris = float(smoothstep(T_IRIS0, T_IRIS1, t))
    op = float(smootherstep(T_SLIT, T_SLIT + 9, t))
    slit_w = 0.004 + 0.15 * op + 0.012 * float(smoothstep(T_SLIT + 9, T_END, t))
    flare = math.exp(-max(t - T_SLIT, 0.0) / 9.0) * float(t >= T_SLIT - 0.5)
    return storm, rim, iris, slit_w, flare, 0.7


def eye_centre():
    return np.array([0.0, -14.0 + EYE_Y, 0.0])


def draw(out, cam, t, vis=None):
    """draw the sky (storm + Eye) into out (H, W, 3) float32, in place"""
    H, W = out.shape[:2]
    C = eye_centre()
    n = cam.pos - C
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 1.0, 0.0])
    eu = up - (up @ n) * n
    eu /= np.linalg.norm(eu)
    es = np.cross(eu, n)
    storm, rim, iris, slit_w, flare, red = state(t)
    if vis is None:
        vis = np.ones((H, W), np.float32)
    _sky(out, vis, np.ascontiguousarray(cam.R, np.float64), np.ascontiguousarray(cam.pos, np.float64),
         float(cam.f_px(W)), W, H, C, n, eu, es, float(EYE_R), float(t), storm, rim, iris, slit_w, flare, red)
    return out


CAM = [  # (frame, radius, height above ground, target height above ground, hfov, azimuth offset)
    (1920, 74.0, 20.0, 37.0, 64.0, -0.10),     # the storm over the forges
    (1960, 70.0, 20.0, 38.0, 62.0, -0.07),
    (2000, 60.0, 21.0, 40.0, 58.0, -0.04),     # the slit opens
    (2080, 54.0, 21.0, 41.0, 54.0, -0.02),     # it holds; we are drawn in
]


def camera(t, az0):
    from core import Camera, catmull
    k = [(f_, np.array([r, y, ty, hf, a])) for f_, r, y, ty, hf, a in CAM]
    r, y, ty, hf, a = catmull(t, k)
    pos = np.array([r * math.cos(az0 + a), -14.0 + y, r * math.sin(az0 + a)])
    tgt = np.array([0.0, -14.0 + ty, 0.0])
    if t >= T_SLIT:
        x = t - T_SLIT
        sh = 0.35 * math.exp(-x / 6.0)
        pos = pos + sh * np.array([math.sin(3.1 * x), math.sin(2.3 * x + 1.0), 0.0])
    return Camera(pos, tgt, hfov=float(hf), focus=float(np.linalg.norm(eye_centre() - pos)), aperture=0.0)
