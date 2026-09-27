"""Relighting a locked frame whose sun moves (B's DUSK and the settled HAND-BACK).

For a locked camera the world is fixed; only the sun's elevation changes (its azimuth is held). March every pixel
once into a G-buffer: surface point, normals, snow, albedo variation, and c0 = the smallest angular clearance of the
ray toward the sun's azimuth over the terrain + cloud sea (incl. the Earth's curvature) at elevation 0. The clearance
at elevation e is then c(e) = c0 + sin(e) (the ray's ground track does not change for |e| < 3 deg), so each frame is
a cheap relight: sun visibility smoothstep(-r, r, c), sun colour through the grazing path (Kasten-Young air mass,
Rayleigh + aerosol), sky, ambient, fog. Same world functions as world.py (read-only).
"""
import math
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import world as WD          # noqa: E402
from mt.noise import gnoise2, smoothstep   # noqa: E402

CLOUD_Y = WD.CLOUD_Y
R_EARTH = WD.R_EARTH

# G-buffer channels
G_FLAG, G_DIST, G_X, G_Y, G_Z, G_NX, G_NY, G_NZ, G_SNOW, G_RV, G_C0, G_DX, G_DY, G_DZ, G_TROUGH, G_FP = range(16)
NG = 16


@njit(inline='always', fastmath=True)
def airmass(c):
    cd = max(c * 57.29578, 0.0)
    return 1.0 / (math.sin(math.radians(cd)) + 0.50572 * (cd + 6.07995) ** -1.6364)


@njit(inline='always', fastmath=True)
def clearance0(P, CR, x, y, z, hx, hz, fp, nsteps, tmax):
    """min over t of (y - h(t)) / t along the horizontal direction (hx, hz): the clearance at elevation 0."""
    t = max(fp * 2.0, 0.5)
    grow = (tmax / t) ** (1.0 / nsteps)
    c = 1.0
    for _s in range(nsteps):
        h = WD.hfun(x + hx * t, z + hz * t, fp + t * 0.002, P, CR)
        a = (y - h) / t
        if a < c:
            c = a
        t *= grow
    return c


@njit(inline='always', fastmath=True)
def clearance0_fine(P, CR, x, y, z, hx, hz, fp, kstep, smin, tmax):
    """As clearance0, with steps proportional to distance (kstep * t, at least smin metres): resolves a narrow
    shoulder a few km out that geometric steps skip over."""
    t = max(fp * 2.0, 0.5)
    c = 1.0
    while t < tmax:
        h = WD.hfun(x + hx * t, z + hz * t, fp + t * 0.002, P, CR)
        a = (y - h) / t
        if a < c:
            c = a
        t += max(kstep * t, smin)
    return c


@njit(parallel=True, fastmath=True, cache=True)
def gbuffer(C, D, P, CR, hx, hz, nsteps, tmax, snow_bias, G, kstep=0.0):
    H, W = D.shape
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
            G[j, i, G_DX] = dx
            G[j, i, G_DY] = dy
            G[j, i, G_DZ] = dz
            sl = vy / hl
            dxh /= hl
            dzh /= hl
            d = D[j, i]
            if d >= 1e29:
                G[j, i, G_FLAG] = 0.0
                G[j, i, G_DIST] = 1e9
                continue
            x = C[0] + dxh * d
            z = C[2] + dzh * d
            dist = d * math.sqrt(1.0 + sl * sl)
            fp = dist / C[7]
            e = max(fp * 0.8, 0.003)
            ddx = x - P[0]
            ddz = z - P[1]
            curv = (ddx * ddx + ddz * ddz) / (2.0 * R_EARTH)
            hf = WD.h_rock(x, z, fp, CR)
            hc = WD.h_cloud(x, z, fp, P[2])
            cloud = hc > hf
            if cloud:
                h0 = hc
                hxx = WD.h_cloud(x + e, z, fp, P[2])
                hzz = WD.h_cloud(x, z + e, fp, P[2])
            else:
                h0 = hf
                hxx = WD.h_rock(x + e, z, fp, CR)
                hzz = WD.h_rock(x, z + e, fp, CR)
            nx = -(hxx - h0)
            ny = e
            nz = -(hzz - h0)
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= inv
            ny *= inv
            nz *= inv
            yw = h0 - curv
            snow = 0.0
            rv = 1.0
            if not cloud:
                es = max(fp * 6.0, 45.0)
                hsx = WD.h_rock(x + es, z, fp * 4.0, CR)
                hsz = WD.h_rock(x, z + es, fp * 4.0, CR)
                nsx = -(hsx - h0) / es
                nsz = -(hsz - h0) / es
                nsy = 1.0 / math.sqrt(nsx * nsx + nsz * nsz + 1.0)
                sn = gnoise2(x / 380.0, z / 380.0, 73) * 0.18 + gnoise2(x / 95.0, z / 95.0, 74) * 0.06
                snow = smoothstep(0.30, 0.52, nsy + sn + snow_bias + 0.05 * gnoise2(x / 30.0, z / 30.0, 75))
                # couloirs and ribs down the fall line: snow gullies and rock ribs on the steep faces
                gl = math.sqrt(nsx * nsx + nsz * nsz) + 1e-6
                ux = -nsz / gl
                uz = nsx / gl
                ac = x * ux + z * uz
                al = (x * nsx + z * nsz) / gl
                gul = gnoise2(ac / 9.0, al / 70.0, 77) * 0.6 + gnoise2(ac / 3.5, al / 25.0, 78) * 0.4
                steep = smoothstep(0.35, 0.8, gl)
                snow = min(max(snow + steep * (0.55 * gul + 0.05), 0.0), 1.0)
                # near: ledges hold snow, ribs shed it (world.py's near rule)
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
                    snow = snow + (smoothstep(0.40, 0.60, sv) - snow) * wf
                if fp < 20.0:
                    rv = 0.75 + 0.5 * (0.5 + 0.5 * gnoise2(x / 7.0 + h0 / 3.0, z / 7.0, 81))
                    rv *= 0.85 + 0.3 * (0.5 + 0.5 * gnoise2(h0 * 0.35, x / 40.0 + z / 40.0, 82))
                    rv = 1.0 + (rv - 1.0) * smoothstep(20.0, 4.0, fp)
            if kstep > 0.0:
                c0 = clearance0_fine(P, CR, x, yw + max(fp * 1.5, 0.3), z, hx, hz, fp, kstep, 4.0, tmax)
            else:
                c0 = clearance0(P, CR, x, yw + max(fp * 1.5, 0.3), z, hx, hz, fp, nsteps, tmax)
            G[j, i, G_FLAG] = 2.0 if cloud else 1.0
            G[j, i, G_DIST] = dist
            G[j, i, G_X] = x
            G[j, i, G_Y] = yw
            G[j, i, G_Z] = z
            G[j, i, G_NX] = nx
            G[j, i, G_NY] = ny
            G[j, i, G_NZ] = nz
            G[j, i, G_SNOW] = snow
            G[j, i, G_RV] = rv
            G[j, i, G_C0] = c0
            G[j, i, G_TROUGH] = smoothstep(CLOUD_Y - 120.0, CLOUD_Y + 110.0, h0) if cloud else 0.0
            G[j, i, G_FP] = fp


# SKY kind 0 = dusk (anti-solar: Earth's shadow + Belt of Venus), 1 = dawn (dawn.py: aureole, gold band, disc)
# SKP (dusk): 0-2 anti-solar horizontal dir | 3 shadow top el | 4-6 zenith | 7-9 horizon | 10-12 shadow col
#             13-15 belt col | 16 belt width | 17 gain
# SD (dawn): dawn.py's layout


@njit(inline='always', fastmath=True)
def dusk_sky(dx, dy, dz, SK):
    el = math.asin(min(max(dy, -1.0), 1.0))
    hl = math.sqrt(dx * dx + dz * dz) + 1e-9
    ca = (dx * SK[0] + dz * SK[2]) / hl
    wa = (0.5 + 0.5 * ca) ** 1.5
    u = smoothstep(0.0, 0.55, max(el, 0.0)) ** 0.5
    r = SK[7] + (SK[4] - SK[7]) * u
    g = SK[8] + (SK[5] - SK[8]) * u
    b = SK[9] + (SK[6] - SK[9]) * u
    es = SK[3]
    sh = (1.0 - smoothstep(es - 0.012, es + 0.014, el)) * wa
    belt = math.exp(-((el - es - 0.035) / SK[16]) ** 2) * wa * (1.0 - sh)
    r = r * (1.0 - sh) + SK[10] * sh + SK[13] * belt
    g = g * (1.0 - sh) + SK[11] * sh + SK[14] * belt
    b = b * (1.0 - sh) + SK[12] * sh + SK[15] * belt
    return r * SK[17], g * SK[17], b * SK[17]


@njit(inline='always', fastmath=True)
def dawn_sky(dx, dy, dz, SD, pix_ang):
    sx, sy, sz = SD[0], SD[1], SD[2]
    el = math.asin(min(max(dy, -1.0), 1.0))
    hl = math.sqrt(dx * dx + dz * dz) + 1e-9
    shl = math.sqrt(sx * sx + sz * sz) + 1e-9
    caz = (dx * sx + dz * sz) / (hl * shl)
    daz = math.acos(min(max(caz, -1.0), 1.0))
    warm = math.exp(-(daz / SD[20]) ** 2)
    hr = SD[14] + (SD[11] - SD[14]) * warm
    hg = SD[15] + (SD[12] - SD[15]) * warm
    hb = SD[16] + (SD[13] - SD[16]) * warm
    u = smoothstep(0.0, 0.10 + 0.08 * warm, max(el, 0.0)) ** 0.75
    r = hr + (SD[8] - hr) * u
    g = hg + (SD[9] - hg) * u
    b = hb + (SD[10] - hb) * u
    band = math.exp(-max(el, 0.0) / SD[19]) * warm
    r += 0.9 * band * SD[11]
    g += 0.9 * band * SD[12]
    b += 0.9 * band * SD[13]
    cosg = dx * sx + dy * sy + dz * sz
    ang = math.acos(min(max(cosg, -1.0), 1.0))
    au = SD[17] * math.exp(-ang / 0.045) + SD[18] * math.exp(-ang / 0.28)
    r += au * SD[3]
    g += au * SD[4]
    b += au * SD[5]
    R = SD[7]
    if ang < R + 2.0 * pix_ang:
        cov = min(max((R - ang) / pix_ang + 0.5, 0.0), 1.0)
        rr = min((ang / R) ** 2, 1.0)
        limb = 0.62 + 0.38 * math.sqrt(1.0 - rr)
        I = SD[6] * cov * limb
        r += I * 1.0
        g += I * 0.93
        b += I * 0.80
    return r, g, b


# LP (light): 0-2 sun dir | 3 sun I | 4-6 Rayleigh tau | 7 aerosol tau | 8 disc radius (visibility penumbra)
#   9 cloud albedo | 10 cloud amb gain | 11 belt fill (west faces) | 12 sheen | 13-15 glow dir | 16-18 glow col
#   19 sky kind (0 dusk / 1 dawn) | 20 pix_ang | 21 rock warmth (0 dusk grey / 1 dawn granite) | 22 sun warm tint
#   23 night key I | 24-26 night key dir | 27-29 night key col | 30 hero red floor | 31 hero penumbra (rad)
NLP = 32


@njit(parallel=True, fastmath=True, cache=True)
def shade(G, LP, SKY, amb, fogp, cam_y, out):
    H, W = G.shape[0], G.shape[1]
    lx, ly, lz = LP[0], LP[1], LP[2]
    se = ly                                          # sin(e)
    rsun = LP[8]
    for j in prange(H):
        for i in range(W):
            dx = G[j, i, G_DX]
            dy = G[j, i, G_DY]
            dz = G[j, i, G_DZ]
            flag = G[j, i, G_FLAG]
            if flag == 0.0:
                if LP[19] < 0.5:
                    r, g, b = dusk_sky(dx, dy, dz, SKY)
                else:
                    r, g, b = dawn_sky(dx, dy, dz, SKY, LP[20])
                out[j, i, 0] = r
                out[j, i, 1] = g
                out[j, i, 2] = b
                continue
            nx = G[j, i, G_NX]
            ny = G[j, i, G_NY]
            nz = G[j, i, G_NZ]
            ndl = nx * lx + ny * ly + nz * lz
            c = G[j, i, G_C0] + se
            hero = flag == 1.0 and G[j, i, G_TROUGH] > 0.5     # tagged hero pixels (her summit): a crisp, vivid end
            rs = LP[31] if (hero and LP[31] > 0.0) else rsun
            vis = smoothstep(-rs, rs, c)
            sr = 0.0
            sg = 0.0
            sb = 0.0
            if vis > 0.0:
                m = airmass(c)
                sr = vis * math.exp(-(LP[4] + LP[7]) * m)
                sg = vis * math.exp(-(LP[5] + LP[7]) * m)
                sb = vis * math.exp(-(LP[6] + LP[7]) * m)
                if hero and LP[30] > 0.0:
                    sr = max(sr, vis * LP[30])
                    sg = max(sg, vis * LP[30] * 0.28)
                    sb = max(sb, vis * LP[30] * 0.20)
            gw = max((nx * LP[13] + ny * LP[14] + nz * LP[15]) * 0.5 + 0.5, 0.0) ** 1.5
            # night key (moon) for the night side of a transition, off when 0
            nk = LP[23] * max(nx * LP[24] + ny * LP[25] + nz * LP[26], 0.0)
            if flag == 2.0:
                wrap = max((ndl + 0.6) / 1.6, 0.0)
                ca = LP[9]
                sky = 0.55 + 0.45 * ny
                cr = ca * (LP[3] * sr * wrap + amb[0] * LP[10] * sky + LP[16] * gw + nk * LP[27])
                cg = ca * (LP[3] * sg * wrap + amb[1] * LP[10] * sky + LP[17] * gw + nk * LP[28])
                cb = ca * (LP[3] * sb * wrap + amb[2] * LP[10] * sky + LP[18] * gw + nk * LP[29])
                tk = 0.55 + 0.45 * G[j, i, G_TROUGH]
                cr *= tk
                cg *= tk
                cb *= tk
            else:
                snow = G[j, i, G_SNOW]
                rv = G[j, i, G_RV]
                w = LP[21]
                a_r = (0.055 + 0.015 * w) * rv
                a_g = (0.056 + 0.006 * w) * rv
                a_b = (0.062 - 0.004 * w) * rv
                ar = a_r + (0.82 - a_r) * snow
                ag = a_g + (0.86 - a_g) * snow
                ab = a_b + (0.96 - a_b) * snow
                dif = max(ndl, 0.0)
                skyl = 0.55 + 0.45 * ny
                west = max(nx * lx + nz * lz, 0.0) / (math.sqrt(lx * lx + lz * lz) + 1e-9)
                belt = LP[11] * west * (1.0 - min(sr * 4.0, 1.0))
                cr = ar * (LP[3] * sr * dif + amb[0] * skyl + belt * 1.0 + LP[16] * gw + nk * LP[27])
                cg = ag * (LP[3] * sg * dif + amb[1] * skyl + belt * 0.55 + LP[17] * gw + nk * LP[28])
                cb = ab * (LP[3] * sb * dif + amb[2] * skyl + belt * 0.62 + LP[18] * gw + nk * LP[29])
                if snow > 0.0 and dif > 0.0 and vis > 0.0:
                    hx_ = lx - dx
                    hy_ = ly - dy
                    hz_ = lz - dz
                    hl_ = math.sqrt(hx_ * hx_ + hy_ * hy_ + hz_ * hz_) + 1e-9
                    nh = max((nx * hx_ + ny * hy_ + nz * hz_) / hl_, 0.0)
                    sp = snow * LP[12] * nh ** 20
                    cr += sp * LP[3] * sr
                    cg += sp * LP[3] * sg
                    cb += sp * LP[3] * sb
            dist = G[j, i, G_DIST]
            yw = G[j, i, G_Y]
            tau = WD.height_fog_tau(dist, cam_y, yw, fogp[0], fogp[1])
            tau += WD.height_fog_tau(dist, cam_y - CLOUD_Y, yw - CLOUD_Y, fogp[2], fogp[3])
            tr = math.exp(-tau)
            cosv = dx * lx + dy * ly + dz * lz
            wv = max(cosv, 0.0) ** fogp[8]
            ph = 1.0 + fogp[4] * wv
            fr = (fogp[5] * (1.0 - wv) + fogp[9] * wv) * ph
            fg = (fogp[6] * (1.0 - wv) + fogp[10] * wv) * ph
            fb = (fogp[7] * (1.0 - wv) + fogp[11] * wv) * ph
            out[j, i, 0] = cr * tr + fr * (1.0 - tr)
            out[j, i, 1] = cg * tr + fg * (1.0 - tr)
            out[j, i, 2] = cb * tr + fb * (1.0 - tr)


def build(scam, P, CR, sun_az, nsteps=72, tmax=160000.0, snow_bias=0.0, dmax=160000.0, hmax=2500.0, kstep=0.0):
    """March the locked source camera and fill the G-buffer (H, W, NG) float32. kstep > 0: the fine clearance march."""
    C = scam.params()
    D = np.zeros((scam.H, scam.W))
    WD.march(P, CR, C, 0.2, dmax, 0.0035, 0.35, hmax, 9, D)
    a = math.radians(sun_az)
    G = np.zeros((scam.H, scam.W, NG), np.float32)
    gbuffer(C, D, P, CR, math.sin(a), math.cos(a), int(nsteps), float(tmax), float(snow_bias), G, float(kstep))
    return G


def sun_vec(el_deg, az_deg):
    e, a = math.radians(el_deg), math.radians(az_deg)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
