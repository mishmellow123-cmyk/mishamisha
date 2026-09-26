"""FIRST BEACON only: a true 3-D moonlit Himalaya for the reveal (replaces the pyramid ridge
cards, which read as stage flats). Imported by beacon.py alone; INTRO/CODA still use land.py.

* Far range: ridged-multifractal heightfield (domain-warped, massif envelope, sharpened peaks)
  sampled once on a POLAR grid around the summit (azimuth x log-range), so every cell is about
  one screen pixel wide at any distance and the octave count is chosen from that footprint
  (prefiltered: no shimmer). Earth curvature applied at render time.
* Cloud sea: its own polar top-surface grid; thins where peaks pierce it (soft skirts).
* Moon: from the left and a little behind camera; soft cast shadows precomputed per cell.
* Summit: fine cartesian grid (10 cm): flat top for the cairn, a serrated rocky arete
  falling away left and right, a calm wind-packed snow face toward camera (T9 band).
* Render: per-pixel ray march, column-coherent (hit distance grows up a column), 2x2
  supersampled, exponential height haze toward the sky's own horizon colour.
Grids are deterministic and cached in shots/hills/cache/.
"""
import math
import os

import cv2
import numba as nb
import numpy as np

from core import CACHE_DIR, fbm2, perlin2
from sky import sky_rad

VERSION = 7
R_EARTH = 6371000.0
_AZ, _EL = math.radians(-112.0), math.radians(23.0)
MOON_L = np.array([math.sin(_AZ) * math.cos(_EL), math.sin(_EL), math.cos(_AZ) * math.cos(_EL)])

TH0, TH1, NTH = math.radians(-44.0), math.radians(44.0), 4096
R0, R1, NR = 110.0, 170000.0, 2048
DTH = (TH1 - TH0) / (NTH - 1)
LR0 = math.log(R0)
DLR = (math.log(R1) - LR0) / (NR - 1)
SX0, SZ0, SD, SNX, SNZ = -70.0, -60.0, 0.1, 1401, 1201
CLOUD_Y = -2150.0
SHN = 2           # shadow grid decimation


@nb.njit(cache=True, fastmath=True, inline='always')
def _sst(a, b, x):
    t = (x - a) / (b - a)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@nb.njit(cache=True, fastmath=True)
def _ridged(x, z, octf):
    nfull = int(octf)
    frac = octf - nfull
    s = 1.0 - abs(perlin2(x, z)) * 1.3
    s = s * s
    res = s
    amp = 1.0
    nmax = nfull + (1 if frac > 1e-3 else 0)
    for o in range(1, nmax):
        x = x * 2.07 + 13.1
        z = z * 2.07 - 7.7
        w = min(max(s * 1.8, 0.0), 1.0)
        s = 1.0 - abs(perlin2(x, z)) * 1.3
        s = s * s * w
        amp *= 0.5
        k = amp if o < nfull else amp * (frac if frac > 1e-3 else 1.0)
        res += s * k
    return res


@nb.njit(cache=True, fastmath=True)
def far_height(x, z, foot):
    S = 7800.0
    wx = fbm2(x / 21000.0 + 3.1, z / 21000.0 - 1.7, 3, 2.0, 0.5)
    wz = fbm2(x / 21000.0 - 4.3, z / 21000.0 + 2.9, 3, 2.0, 0.5)
    octf = math.log2(max(S / max(foot * 1.5, 1e-3), 1.0))
    octf = min(max(octf, 1.0), 12.0)
    r = _ridged(x / S + 0.6 * wx + 0.31, z / S + 0.6 * wz - 1.73, octf)
    env = fbm2(x / 26000.0 + 0.7, z / 26000.0 - 0.4, 3, 2.0, 0.5)
    e = _sst(-0.40, 0.35, env)
    rn = max(r - 0.5, 0.0) / 1.25          # median ~0.36, 99.5th pct ~1.0
    rr = math.sqrt(x * x + z * z)
    far = 1.0 - 0.5 * _sst(30000.0, 110000.0, rr)   # distant ranges sink into the haze (no cap-height wall)
    h = -2950.0 + (1500.0 + 1300.0 * e) * far * rn ** 2.2
    # every summit stays below hers (rarely active soft cap ~ -240 m)
    if h > -520.0:
        h = -520.0 + 280.0 * math.tanh((h + 520.0) / 280.0)
    near = _sst(300.0, 1900.0, rr)
    return -3300.0 + (h + 3300.0) * near


@nb.njit(cache=True, fastmath=True)
def cloud_height(x, z, foot):
    o1 = min(max(math.log2(max(2600.0 / max(foot * 1.5, 1e-3), 1.0)), 1.0), 6.0)
    n = fbm2(x / 2600.0 + 1.3, z / 2600.0 - 0.2, int(o1), 2.0, 0.5)
    o2 = min(max(math.log2(max(650.0 / max(foot * 1.5, 1e-3), 1.0)), 1.0), 5.0)
    m = 1.0 - abs(fbm2(x / 650.0 - 2.1, z / 650.0 + 0.8, int(o2), 2.0, 0.5))
    return CLOUD_Y + 170.0 * n + 55.0 * m


@nb.njit(cache=True, fastmath=True)
def _build_polar(H, C):
    for j in range(NR):
        r = math.exp(LR0 + DLR * j)
        foot = r * DTH
        for i in range(NTH):
            th = TH0 + DTH * i
            x = r * math.sin(th)
            z = r * math.cos(th)
            H[j, i] = far_height(x, z, foot)
            if (i % 2 == 0) and (j % 2 == 0):
                C[j // 2, i // 2] = cloud_height(x, z, foot * 2.0)


@nb.njit(cache=True, fastmath=True, inline='always')
def polar_h(G, x, z, dec):
    r2 = x * x + z * z
    if r2 < R0 * R0:
        return -1e9
    th = math.atan2(x, z)
    fi = (th - TH0) / (DTH * dec)
    fj = (0.5 * math.log(r2) - LR0) / (DLR * dec)
    ni = G.shape[1]
    nj = G.shape[0]
    if fi < 0.0 or fj < 0.0 or fi >= ni - 1 or fj >= nj - 1:
        return -1e9
    i = int(fi)
    j = int(fj)
    a = fi - i
    b = fj - j
    h0 = G[j, i] + (G[j, i + 1] - G[j, i]) * a
    h1 = G[j + 1, i] + (G[j + 1, i + 1] - G[j + 1, i]) * a
    return h0 + (h1 - h0) * b


@nb.njit(cache=True, fastmath=True, inline='always')
def summit_h(S, x, z):
    fi = (x - SX0) / SD
    fj = (z - SZ0) / SD
    if fi < 0.0 or fj < 0.0 or fi >= S.shape[1] - 1 or fj >= S.shape[0] - 1:
        return -1e9
    i = int(fi)
    j = int(fj)
    a = fi - i
    b = fj - j
    h0 = S[j, i] + (S[j, i + 1] - S[j, i]) * a
    h1 = S[j + 1, i] + (S[j + 1, i + 1] - S[j + 1, i]) * a
    return h0 + (h1 - h0) * b


@nb.njit(cache=True, fastmath=True)
def _shadow_grid(H, C, L, SH):
    lx = L[0]
    ly = L[1]
    lz = L[2]
    for j in range(SH.shape[0]):
        r = math.exp(LR0 + DLR * j * SHN)
        for i in range(SH.shape[1]):
            th = TH0 + DTH * i * SHN
            x = r * math.sin(th)
            z = r * math.cos(th)
            h = max(H[j * SHN, i * SHN], C[min(j * SHN // 2, C.shape[0] - 1), min(i * SHN // 2, C.shape[1] - 1)])
            y = h - r * r / (2.0 * R_EARTH) + 3.0
            k = 1.0
            s = 30.0
            while s < 40000.0:
                qx = x + lx * s
                qz = z + lz * s
                qy = y + ly * s
                hq = max(polar_h(H, qx, qz, 1.0), polar_h(C, qx, qz, 2.0))
                if hq < -1e8:
                    break
                d = qy - (hq - (qx * qx + qz * qz) / (2.0 * R_EARTH))
                kk = d / (0.06 * s)
                if kk < k:
                    k = kk
                    if k < -0.5:
                        break
                s *= 1.08
            SH[j, i] = _sst(-0.35, 0.3, k)


@nb.njit(cache=True, fastmath=True)
def _build_summit(S, M):
    for j in range(SNZ):
        z = SZ0 + SD * j
        for i in range(SNX):
            x = SX0 + SD * i
            zc = 0.30 + 0.0045 * x * x
            dl = math.log(1.0 + math.exp(1.6 * (-0.9 - x))) / 1.6
            dr = math.log(1.0 + math.exp(1.6 * (x - 1.9))) / 1.6
            side = dl + dr
            hc = -0.010 * (x - 0.5) ** 2 - 0.26 * dl ** 1.25 - 0.34 * dr ** 1.3
            # serrated arete: rock teeth grow away from the top
            g = (1.0 - math.exp(-side / 2.5)) * math.exp(-abs(z - zc) / 2.2)
            tooth = _ridged(x / 2.1 + 5.3, z / 3.0 + 1.1, 4.0)
            hc += g * (0.9 * tooth - 0.35) * (1.0 + 0.04 * side)
            if z < zc:
                df = zc - z
                h = hc - 0.10 * df - 0.021 * df * df - 0.02 * (x - 0.5) ** 2 * df / (df + 2.5)
            else:
                db = z - zc
                h = hc - 0.55 * db - 0.09 * db * db
            # rock outcrops: on the flanks and low on the face, kept off the calm central face
            dfc = max(0.0, zc - z)
            m = _sst(6.0, 12.0, side) * (0.4 + 0.6 * math.exp(-dfc / 6.0)) + _sst(13.0, 20.0, dfc) * _sst(2.0, 8.0, abs(x - 0.5))
            m += _sst(0.0, 3.0, z - zc) * _sst(2.0, 5.0, side)
            m = min(m, 1.0) * _sst(-0.35, 0.25, fbm2(x / 4.5 + 2.2, z / 4.5 - 0.7, 3, 2.0, 0.5))
            rk = 0.0
            if m > 0.0:
                q = _ridged(x / 3.2 - 1.9, z / 3.2 + 4.4, 5.0)
                rk = m * max(0.0, q - 0.55) * 2.4
            # wind-packed snow: soft sastrugi, very low contrast
            sn = 0.035 * perlin2(x / 1.4 + 0.3, z / 0.45 - 0.2) + 0.05 * perlin2(x / 5.0, z / 3.0 + 9.1)
            sn += 0.35 * perlin2(x / 7.0 + 1.7, z / 9.0 - 3.3) * _sst(2.0, 6.0, max(side, max(0.0, zc - z)))
            top = _sst(3.0, 0.8, math.sqrt((x - 0.45) ** 2 * 0.35 + (z - 0.1) ** 2 * 1.6))
            sn *= 1.0 - top
            S[j, i] = h + rk + sn
            M[j, i] = min(1.0, rk * 3.0)


_CACHE = {}


def grids():
    """(H far heights, C cloud tops, SH moon visibility, S summit heights, M summit rock mask)."""
    if 'g' in _CACHE:
        return _CACHE['g']
    path = os.path.join(CACHE_DIR, f'beacon_peaks_v{VERSION}.npz')
    if os.path.exists(path):
        d = np.load(path)
        g = (d['H'], d['C'], d['SH'], d['S'], d['M'])
    else:
        H = np.zeros((NR, NTH), np.float32)
        C = np.zeros((NR // 2, NTH // 2), np.float32)
        _build_polar(H, C)
        SH = np.zeros((NR // SHN, NTH // SHN), np.float32)
        _shadow_grid(H, C, MOON_L, SH)
        S = np.zeros((SNZ, SNX), np.float32)
        M = np.zeros((SNZ, SNX), np.float32)
        _build_summit(S, M)
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + f'.{os.getpid()}.tmp.npz'
        np.savez(tmp, H=H, C=C, SH=SH, S=S, M=M)
        os.replace(tmp, path)
        g = (H, C, SH, S, M)
    _CACHE['g'] = g
    return g


# ------------------------------------------------------------------ render ---

@nb.njit(cache=True, fastmath=True, inline='always')
def _surface(H, C, S, x, z):
    """(terrain height incl. curvature, cloud-top height incl. curvature, is_summit)"""
    curv = (x * x + z * z) / (2.0 * R_EARTH)
    hs = summit_h(S, x, z)
    hf = polar_h(H, x, z, 1.0) - curv
    hc = polar_h(C, x, z, 2.0) - curv
    if hs > hf:
        return hs, hc, 1
    return hf, hc, 0


@nb.njit(cache=True, fastmath=True)
def _fog_tau(cy, dy, t, P):
    # exponential height haze: sigma(y) = P0 + P1*exp(-(y-P2)/P3), 6-pt Gauss-Legendre
    xs = (0.0337652429, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571)
    ws = (0.0856622462, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.0856622462)
    acc = 0.0
    for k in range(6):
        s = t * xs[k]
        y = cy + dy * s + s * s / (2.0 * R_EARTH)
        acc += ws[k] * (P[0] + P[1] * math.exp(-max(y - P[2], -3.0 * P[3]) / P[3]))
    return acc * t


@nb.njit(cache=True, fastmath=True)
def render_terrain(out, alpha, cam, H, C, SH, S, M, sp, P, lights, gains, t):
    """out HxWx3 premultiplied (at the supersampled resolution of `cam`), alpha HxW.
    P: [fog s0, s1, y0, hs, moon I, moon r,g,b, amb r,g,b, fog_el, cloud_alb, snow_thr,
        haze_moon, rock_alb] ; lights: (N,8) x,y,z, r,g,b, radius, _ ; gains: [far, summit]"""
    Hh = out.shape[0]
    W = out.shape[1]
    lx = MOON_L[0]
    ly = MOON_L[1]
    lz = MOON_L[2]
    fpx = cam[12]
    for i in range(W):
        # fog colour for this column: sky near the horizon along the column's azimuth
        dx0 = cam[3] * ((i + 0.5 - cam[13]) / fpx) + cam[5]
        dz0 = cam[9] * ((i + 0.5 - cam[13]) / fpx) + cam[11]
        hn = math.sqrt(dx0 * dx0 + dz0 * dz0) + 1e-9
        ce = math.cos(P[14])
        fr, fg, fb = sky_rad(dx0 / hn * ce, math.sin(P[14]), dz0 / hn * ce, sp)
        # moon-side haze glow
        mph = max(0.0, -(dx0 / hn) * lx - (dz0 / hn) * lz)
        hz = 1.0 + P[17] * mph
        fr = P[11] * fr * hz + P[12] * P[5] * (0.5 + mph)
        fg = P[11] * fg * hz + P[12] * P[6] * (0.5 + mph)
        fb = P[11] * fb * hz + P[12] * P[7] * (0.5 + mph)
        tprev = 0.05
        for j in range(Hh - 1, -1, -1):
            x = (i + 0.5 - cam[13]) / fpx
            y = -(j + 0.5 - cam[14]) / fpx
            dx = cam[3] * x + cam[4] * y + cam[5]
            dy = cam[6] * x + cam[7] * y + cam[8]
            dz = cam[9] * x + cam[10] * y + cam[11]
            nrm = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
            dx *= nrm
            dy *= nrm
            dz *= nrm
            if dy > 0.02:
                break
            cx = cam[0]
            cy = cam[1]
            cz = cam[2]
            tt = tprev
            t0 = tt
            hit = 0
            kind = 0
            while tt < 260000.0:
                px = cx + dx * tt
                py = cy + dy * tt
                pz = cz + dz * tt
                hh, hc, ks = _surface(H, C, S, px, pz)
                if py < hh:
                    hit = 1
                    kind = ks
                    break
                if py < hc:
                    hit = 2
                    break
                t0 = tt
                top = max(hh, hc)
                st = max(0.0035 * tt, 0.012)
                if top > -1e8:
                    st = max(st, min(0.4 * (py - top), 0.05 * tt + 5.0))
                elif tt < R0:
                    st = max(st, 0.25 * tt)
                if hh < -1e8 and hc < -1e8 and tt > 1000.0:
                    break
                tt += st
            if hit == 0:
                break
            # refine the crossing
            a = t0
            b = tt
            for _ in range(10):
                m = 0.5 * (a + b)
                px = cx + dx * m
                py = cy + dy * m
                pz = cz + dz * m
                hh, hc, ks = _surface(H, C, S, px, pz)
                lim = hh if hit == 1 else max(hh, hc)
                if py < lim:
                    b = m
                else:
                    a = m
            th_ = b
            tprev = max(0.05, th_ * 0.985)
            px = cx + dx * th_
            py = cy + dy * th_
            pz = cz + dz * th_
            hh, hc, ks = _surface(H, C, S, px, pz)
            if hit == 1:
                kind = ks
            foot = th_ / fpx
            cr = 0.0
            cg = 0.0
            cb = 0.0
            cloud_a = 0.0
            ccr = 0.0
            ccg = 0.0
            ccb = 0.0
            tsurf = th_
            if hit == 2:
                # cloud top: soft wrap light, forward-scatter sheen, troughs darker
                e = max(foot * 2.0, 25.0)
                c0 = polar_h(C, px, pz, 2.0)
                cxp = polar_h(C, px + e, pz, 2.0)
                czp = polar_h(C, px, pz + e, 2.0)
                nx = -(cxp - c0)
                nz = -(czp - c0)
                ny = e
                inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
                nx *= inv
                ny *= inv
                nz *= inv
                ndl = nx * lx + ny * ly + nz * lz
                wrap = max((ndl + 0.6) / 1.6, 0.0)
                vis = 0.25 + 0.75 * polar_h(SH, px, pz, float(SHN))
                tro = _sst(CLOUD_Y - 150.0, CLOUD_Y + 120.0, c0)
                tex = 0.75 + 0.55 * fbm2(px / 900.0 + 0.02 * t, pz / 900.0, 4, 2.0, 0.5)
                k = P[15] * (0.45 + 0.55 * tro) * tex
                ccr = k * (P[4] * P[5] * wrap * vis + P[8] * 1.3)
                ccg = k * (P[4] * P[6] * wrap * vis + P[9] * 1.3)
                ccb = k * (P[4] * P[7] * wrap * vis + P[10] * 1.3)
                # peaks pierce the cloud: thin where terrain is close beneath the top
                depth_below = c0 - (hh + (px * px + pz * pz) / (2.0 * R_EARTH))
                cloud_a = 1.0 - math.exp(-max(depth_below, 0.0) / 70.0)
                if cloud_a < 0.985:
                    # find the terrain under the thin cloud
                    t2 = th_
                    found = 0
                    while t2 < th_ + 6000.0:
                        t2 += max(0.003 * t2, 2.0)
                        qx = cx + dx * t2
                        qy = cy + dy * t2
                        qz = cz + dz * t2
                        h2, c2, k2 = _surface(H, C, S, qx, qz)
                        if qy < h2:
                            found = 1
                            break
                    if found == 1:
                        px = qx
                        py = qy
                        pz = qz
                        hh = h2
                        kind = k2
                        tsurf = t2
                        foot = t2 / fpx
                    else:
                        cloud_a = 1.0
                else:
                    cloud_a = 1.0
            if hit == 1 or cloud_a < 1.0:
                if kind == 1:
                    e = max(foot * 1.5, SD)
                    h0 = summit_h(S, px, pz)
                    hxp = summit_h(S, px + e, pz)
                    hxm = summit_h(S, px - e, pz)
                    hzp = summit_h(S, px, pz + e)
                    hzm = summit_h(S, px, pz - e)
                else:
                    e = max(foot * 1.5, 6.0)
                    h0 = polar_h(H, px, pz, 1.0)
                    hxp = polar_h(H, px + e, pz, 1.0)
                    hxm = polar_h(H, px - e, pz, 1.0)
                    hzp = polar_h(H, px, pz + e, 1.0)
                    hzm = polar_h(H, px, pz - e, 1.0)
                if hxp < -1e8:
                    hxp = h0
                if hxm < -1e8:
                    hxm = h0
                if hzp < -1e8:
                    hzp = h0
                if hzm < -1e8:
                    hzm = h0
                nx = -(hxp - hxm)
                nz = -(hzp - hzm)
                ny = 2.0 * e
                inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
                nx *= inv
                ny *= inv
                nz *= inv
                ndl = nx * lx + ny * ly + nz * lz
                if kind == 1:
                    rock = M[min(max(int((pz - SZ0) / SD + 0.5), 0), SNZ - 1), min(max(int((px - SX0) / SD + 0.5), 0), SNX - 1)]
                    rock = max(rock, _sst(0.62, 0.50, ny) * 0.9)
                    vis = 1.0
                    snoise = 0.0
                else:
                    # smoother normal for the snow-hold decision (snow sits where the big form is gentle)
                    es = max(foot * 6.0, 40.0)
                    sxp = polar_h(H, px + es, pz, 1.0)
                    szp = polar_h(H, px, pz + es, 1.0)
                    if sxp < -1e8:
                        sxp = h0
                    if szp < -1e8:
                        szp = h0
                    gx = (sxp - h0) / es
                    gz = (szp - h0) / es
                    nys = 1.0 / math.sqrt(1.0 + gx * gx + gz * gz)
                    snoise = 0.16 * perlin2(px / 420.0, pz / 420.0) + 0.07 * perlin2(px / 110.0 + 3.3, pz / 110.0)
                    rock = 1.0 - _sst(P[16] - 0.07, P[16] + 0.07, 0.55 * ny + 0.45 * nys + snoise)
                    vis = polar_h(SH, px, pz, float(SHN))
                    if vis < -1e8:
                        vis = 1.0
                snow = 1.0 - rock
                ar = P[18] * rock + 0.80 * snow
                ag = P[18] * 1.02 * rock + 0.86 * snow
                ab = P[18] * 1.12 * rock + 0.98 * snow
                dif = max(ndl, 0.0) * vis
                skyl = 0.45 + 0.55 * ny
                cr = ar * (P[4] * P[5] * dif + P[8] * skyl)
                cg = ag * (P[4] * P[6] * dif + P[9] * skyl)
                cb = ab * (P[4] * P[7] * dif + P[10] * skyl)
                if kind == 1:
                    for q in range(lights.shape[0]):
                        vx = lights[q, 0] - px
                        vy = lights[q, 1] - hh
                        vz = lights[q, 2] - pz
                        d2 = vx * vx + vy * vy + vz * vz
                        dd = math.sqrt(d2) + 1e-6
                        nd = (nx * vx + ny * vy + nz * vz) / dd
                        if nd > 0.0:
                            E = (0.35 + 0.65 * nd) / (1.0 + d2 / (lights[q, 6] * lights[q, 6]))
                            cr += ar * E * lights[q, 3]
                            cg += ag * E * lights[q, 4]
                            cb += ab * E * lights[q, 5]
            # combine terrain under thin cloud
            if hit == 2:
                if cloud_a >= 1.0:
                    cr = ccr
                    cg = ccg
                    cb = ccb
                    tsurf = th_
                else:
                    cr = cr * (1.0 - cloud_a) + ccr * cloud_a
                    cg = cg * (1.0 - cloud_a) + ccg * cloud_a
                    cb = cb * (1.0 - cloud_a) + ccb * cloud_a
                    tsurf = th_
            # aerial perspective
            tau = _fog_tau(cy, dy, tsurf, P)
            tr = math.exp(-tau)
            cr = cr * tr + fr * (1.0 - tr)
            cg = cg * tr + fg * (1.0 - tr)
            cb = cb * tr + fb * (1.0 - tr)
            gn = gains[1] if (kind == 1 and hit == 1) else gains[0]
            out[j, i, 0] = cr * gn
            out[j, i, 1] = cg * gn
            out[j, i, 2] = cb * gn
            alpha[j, i] = 1.0


# fog / light parameters (linear radiance, same scale as the rest of the shot)
PARAMS = np.array([
    1.0 / 110000.0,      # 0 fog sigma floor
    1.0 / 6000.0,        # 1 haze sigma at y0
    -2250.0,             # 2 y0 (just above the cloud sea)
    900.0,               # 3 haze scale height
    0.95,                # 4 moon intensity
    0.55, 0.70, 0.95,    # 5-7 moonlight colour (#9DB4D9-ish, linear)
    0.010, 0.016, 0.040, # 8-10 sky ambient
    2.2,                 # 11 haze gain on the sky's horizon colour
    0.030, 0.0,          # 12 moonlit in-scatter, 13 unused
    math.radians(1.6),   # 14 fog colour elevation
    0.42,                # 15 cloud albedo
    0.22,                # 16 snow threshold on normal.y
    0.6,                 # 17 haze brightening toward the moon side
    0.07,                # 18 rock albedo
], np.float64)


def render(cam, sky_packed, t, lights, far_gain, summit_gain, ss=2, params=None):
    """Premultiplied rgb + alpha at cam resolution."""
    from core import Camera
    H, C, SH, S, M = grids()
    big = Camera(cam.pos, yaw=cam.yaw, pitch=cam.pitch, roll=cam.roll, f=cam.f_full, scale=cam.scale * ss)
    out = np.zeros((big.H, big.W, 3), np.float32)
    al = np.zeros((big.H, big.W), np.float32)
    render_terrain(out, al, big.params(), H, C, SH, S, M, sky_packed,
                   PARAMS if params is None else params, np.asarray(lights, np.float64).reshape(-1, 8),
                   np.array([far_gain, summit_gain], np.float64), float(t))
    if ss != 1:
        out = cv2.resize(out, (cam.W, cam.H), interpolation=cv2.INTER_AREA)
        al = cv2.resize(al, (cam.W, cam.H), interpolation=cv2.INTER_AREA)
    return out, al
