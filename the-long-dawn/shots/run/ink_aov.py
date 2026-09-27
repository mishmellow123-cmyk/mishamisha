"""AOV passes of the Run world for C's LIVING INK ILLUSTRATION (R12). Additive: world.py is untouched.

`shade_aov` walks exactly the surface logic of `world.shade` (surface choice, fine normal, s1's snow rule plus the
close-range snow, the key light's soft shadow, aerial perspective), but writes the components instead of a colour:

  0 surf (0 near summit patch, 1 rock, 2 cloud sea, 3 sky)   1 dist (Euclidean m; 1e9 sky)
  2 x  3 y (as rendered: height minus Earth curvature)  4 z   5 h (raw height, no curvature: world-anchored)
  6 nx 7 ny 8 nz (fine normal)   9 snow   10 n.l (key)   11 key shadow (cloud: its own soft shadow)
  12 fog transmittance   13 firelight (sum of E x luminance, the caller's intensities, no flicker)
  14 cloud light (wrap x forward scatter x trough; 0 on rock)   15 view depth along the source axis (z-buffer)

`grad_at` gives the ink pass a height gradient at a chosen world scale (the stroke's), from the same height
functions, so hatching directions are functions of world position and level only (never of the camera)."""
import math
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import world as WD                       # noqa: E402
from mt.noise import gnoise2, smoothstep  # noqa: E402

NA = 16
CLOUD_Y = WD.CLOUD_Y
R_EARTH = WD.R_EARTH


@njit(parallel=True, fastmath=True, cache=True)
def shade_aov(C, D, P, CR, LT, Lk, Q, fogp, PL, A):
    H, W = D.shape
    mx, my, mz = Lk[0], Lk[1], Lk[2]
    for j in prange(H):
        for i in range(W):
            fxx, fzz, rxx, rzz = C[3], C[4], C[5], C[6]
            f, cx, cyy = C[7], C[8], C[9]
            xo = i + 0.5 - cx
            dxh = fxx * f + rxx * xo
            dzh = fzz * f + rzz * xo
            hl = math.sqrt(dxh * dxh + dzh * dzh)
            vy = cyy - (j + 0.5)
            nn = math.sqrt(hl * hl + vy * vy)
            dx = dxh / nn
            dy = vy / nn
            dz = dzh / nn
            sl = vy / hl
            dxh /= hl
            dzh /= hl
            d = D[j, i]
            for c in range(NA):
                A[j, i, c] = 0.0
            if d >= 1e29:
                A[j, i, 0] = 3.0
                A[j, i, 1] = 1e9
                A[j, i, 12] = 0.0
                A[j, i, 15] = 1e9
                continue
            x = C[0] + dxh * d
            z = C[2] + dzh * d
            dist = d * math.sqrt(1.0 + sl * sl)
            fp = dist / C[7]
            e = max(fp * 0.8, 0.003)
            ddx = x - P[0]
            ddz = z - P[1]
            curv = (ddx * ddx + ddz * ddz) / (2.0 * R_EARTH)
            hn = WD.h_near(x, z, fp)
            hf = WD.h_rock(x, z, fp, CR)
            hc = WD.h_cloud(x, z, fp, P[2])
            surf = 0
            if hf > hn:
                surf = 1
            if hc > max(hn, hf):
                surf = 2
            if surf == 0:
                h0 = hn
                hx = WD.h_near(x + e, z, fp)
                hz = WD.h_near(x, z + e, fp)
            elif surf == 1:
                h0 = hf
                hx = WD.h_rock(x + e, z, fp, CR)
                hz = WD.h_rock(x, z + e, fp, CR)
            else:
                h0 = hc
                hx = WD.h_cloud(x + e, z, fp, P[2])
                hz = WD.h_cloud(x, z + e, fp, P[2])
            nx = -(hx - h0)
            ny = e
            nz = -(hz - h0)
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= inv
            ny *= inv
            nz *= inv
            ndl = nx * mx + ny * my + nz * mz
            yw = h0 - curv
            snow = 0.0
            shd = 1.0
            fire = 0.0
            cl = 0.0
            if surf == 2:
                cosk = dx * mx + dy * my + dz * mz
                fwd = 1.0 + Q[1] * max(cosk, 0.0) ** Q[2]
                wrap = max((ndl + 0.8) / 1.8, 0.0)
                shd = WD.soft_shadow(P, CR, x, yw + 5.0, z, mx, my, mz, 40.0, 14000.0, 13, 6.0, fp)
                trough = smoothstep(CLOUD_Y - 120.0, CLOUD_Y + 110.0, h0)
                tk = (1.0 - Q[8]) + Q[8] * trough
                cl = wrap * fwd * tk
            else:
                es = max(fp * 6.0, 45.0)
                if surf == 0:
                    hsx = WD.h_near(x + es, z, fp * 4.0)
                    hsz = WD.h_near(x, z + es, fp * 4.0)
                else:
                    hsx = WD.h_rock(x + es, z, fp * 4.0, CR)
                    hsz = WD.h_rock(x, z + es, fp * 4.0, CR)
                nsx = -(hsx - h0) / es
                nsz = -(hsz - h0) / es
                nsy = 1.0 / math.sqrt(nsx * nsx + nsz * nsz + 1.0)
                if surf == 0:
                    sn = gnoise2(x * 0.8, z * 0.8, 71) * 0.12 + gnoise2(x * 3.1, z * 3.1, 72) * 0.05
                    snow = smoothstep(0.45, 0.70, ny + sn)
                else:
                    sn = gnoise2(x / 380.0, z / 380.0, 73) * 0.18 + gnoise2(x / 95.0, z / 95.0, 74) * 0.06
                    snow = smoothstep(0.38, 0.58, nsy + sn)
                    wf = smoothstep(3.0, 0.4, fp)
                    if wf > 0.0:
                        em = max(fp * 6.0, 3.0)
                        hmx = WD.h_rock(x + em, z, fp * 2.0, CR)
                        hmz = WD.h_rock(x, z + em, fp * 2.0, CR)
                        nmx = -(hmx - h0) / em
                        nmz = -(hmz - h0) / em
                        nmy = 1.0 / math.sqrt(nmx * nmx + nmz * nmz + 1.0)
                        sf = gnoise2(x / 9.0, z / 9.0, 75) * 0.06 + gnoise2(x * 0.9, z * 0.9, 76) * 0.035
                        sv = 0.42 * ny + 0.33 * nmy + 0.25 * nsy + sn * 0.6 + sf
                        sneat = smoothstep(Q[3], Q[4], sv)
                        snow = snow + (sneat - snow) * wf
                for pi in range(PL.shape[0]):
                    qx = x - PL[pi, 0]
                    qz = z - PL[pi, 2]
                    qd = math.sqrt(qx * qx + qz * qz)
                    if qd < PL[pi, 3] * 1.6 and abs(yw - PL[pi, 1]) < 6.0:
                        snow *= smoothstep(PL[pi, 3] * 0.7, PL[pi, 3] * 1.6,
                                           qd + 0.4 * PL[pi, 3] * gnoise2(x * 1.3, z * 1.3, 91))
                if ndl > 0.0:
                    t0 = max(fp * 1.5, 0.08)
                    nst = 20 if dist < 2500.0 else 14
                    shd = WD.soft_shadow(P, CR, x, yw + max(fp * 1.2, 0.03), z, mx, my, mz, t0, 12000.0, nst,
                                         Q[11], fp)
                else:
                    shd = 0.0
                for li in range(LT.shape[0]):
                    lx = LT[li, 0] - x
                    ly = LT[li, 1] - yw
                    lz = LT[li, 2] - z
                    l2 = lx * lx + ly * ly + lz * lz
                    if l2 > LT[li, 6] * 4.0e4:
                        continue
                    ll = math.sqrt(l2) + 1e-9
                    ndf = (nx * lx + ny * ly + nz * lz) / ll
                    if ndf > 0.0:
                        E = LT[li, 6] * ndf / (l2 + LT[li, 7] * LT[li, 7])
                        fire += E * (0.2126 * LT[li, 3] + 0.7152 * LT[li, 4] + 0.0722 * LT[li, 5])
            yc = C[1]
            tau = WD.height_fog_tau(dist, yc, yw, fogp[0], fogp[1])
            tau += WD.height_fog_tau(dist, yc - CLOUD_Y, yw - CLOUD_Y, fogp[2], fogp[3])
            A[j, i, 0] = surf
            A[j, i, 1] = dist
            A[j, i, 2] = x
            A[j, i, 3] = yw
            A[j, i, 4] = z
            A[j, i, 5] = h0
            A[j, i, 6] = nx
            A[j, i, 7] = ny
            A[j, i, 8] = nz
            A[j, i, 9] = snow
            A[j, i, 10] = ndl
            A[j, i, 11] = shd
            A[j, i, 12] = math.exp(-tau)
            A[j, i, 13] = fire
            A[j, i, 14] = cl
            A[j, i, 15] = d * (dxh * C[3] + dzh * C[4])


@njit(inline='always', fastmath=True)
def height_at(x, z, fp, CR, surf, t):
    if surf == 0:
        return WD.h_near(x, z, fp)
    if surf == 2:
        return WD.h_cloud(x, z, fp, t)
    return WD.h_rock(x, z, fp, CR)


@njit(inline='always', fastmath=True)
def grad_at(x, z, s, CR, surf, t):
    """Central-difference gradient of the surface's height at world scale s (footprint s/2)."""
    fp = 0.5 * s
    hx1 = height_at(x + s, z, fp, CR, surf, t)
    hx0 = height_at(x - s, z, fp, CR, surf, t)
    hz1 = height_at(x, z + s, fp, CR, surf, t)
    hz0 = height_at(x, z - s, fp, CR, surf, t)
    return (hx1 - hx0) / (2.0 * s), (hz1 - hz0) / (2.0 * s)
