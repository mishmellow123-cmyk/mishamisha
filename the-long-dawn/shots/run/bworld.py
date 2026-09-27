"""B's own world (THE VIGIL): old, broad, glaciated massifs above the cloud sea.

H5 CALLS s5: each film owns its landform. A keeps s1's knife-edge ranges (world.py, untouched); B gets this:
  * her massif: a broad dome with a rounded, stony summit (the shelf for the beacon, the cairn and the figures),
    lobed spurs and cirque bays, and the long NE ridge (a broad, round-crested snow ridge: the way up and down);
  * art-directed neighbours (rows of CR_B): broad domes and whalebacks placed for the three frames (DUSK, the
    vigil's locked frame, the hand-back's sunrise shoulder);
  * the regional field: rounded massifs from a convex transfer of warped fBm (steep glacial lower flanks, gentle
    domed tops), cut into ranges by U-shaped troughs that the cloud sea fills, so the far ranges stand as layered
    silhouettes. No ridged noise anywhere: no needles, no triangles.
Shading lives in a G-buffer (one march per camera) that is relit per frame: the sun's clearance toward a fixed
azimuth (DUSK / sunrise), K moon-visibility channels (the vigil's moving moon), snow that follows the form in
horizontal bands (no fall-line streaks). Same cloud sea, curvature and fog as world.py (read-only).
"""
import math
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import world as W0          # noqa: E402  (constants, cloud sea, fog; never modified)
import s1_peak as S1        # noqa: E402
from mt.noise import fbm2, gnoise2, smoothstep   # noqa: E402

CM = W0.CM
R_EARTH = W0.R_EARTH
CLOUD_Y = W0.CLOUD_Y
MOON_DIR = W0.MOON_DIR.copy()
height_fog_tau = W0.height_fog_tau

_S = S1.summit()
TX = float(_S[0] - 8.0)                 # her summit (same point keeper.py uses)
TZ = float(_S[2] + 1.0)
TOP_Y = 301.5
RIDGE_AZ = 40.0                         # the NE ridge
NCR = 16
VERSION = 'b2'                           # bump when the terrain changes (cache keys)

# row layout (type in col 12)
#  dome  (0): 0 cx | 1 cz | 2 top | 3 R (reaches the cloud) | 4 p profile | 5 aniso | 6 angle | 7 seed | 8 lobe
#             9 smooth k | 10 detail | 11 bay depth | 12 type 0 | 13 reach | 14 top rounding | 15 -
#  ridge (1): 0 ax | 1 az | 2 ay | 3 bx | 4 bz | 5 by | 6 crest radius | 7 seed | 8 flank slope | 9 smooth k
#             10 detail | 11 - | 12 type 1 | 13 reach | 14 - | 15 -


def dome_row(cx, cz, top, R, p=1.7, aniso=1.0, ang=0.0, seed=1, k=60.0, detail=0.30, bay=0, rnd=0.03,
             cq0=0.0, keep=0.0):
    """dome: 0 cx | 1 cz | 2 top | 3 R | 4 p | 5 aniso | 6 angle | 7 seed | 8 first cirque angle (rad) | 9 smooth k
    10 cirque depth (x Dm) | 11 cirque count | 12 type 0 | 13 reach | 14 top rounding."""
    r = np.zeros(NCR)
    r[:16] = [cx, cz, top, R, p, aniso, math.radians(ang), seed, cq0, k, detail, bay, 0.0, 0.0, rnd, keep]
    r[13] = 1.6 * R * max(aniso, 1.0)
    return r


def ridge_row(a, b, crest=22.0, seed=41, slope=0.85, k=25.0, detail=1.0):
    """A broad, round-crested ridge from a=(x, y, z) to b=(x, y, z)."""
    r = np.zeros(NCR)
    r[:11] = [a[0], a[2], a[1], b[0], b[2], b[1], crest, seed, slope, k, detail]
    r[12] = 1.0
    r[13] = (max(a[1], b[1]) - (CLOUD_Y - 300.0)) / slope + 60.0
    return r


def polar(az_deg, r):
    a = math.radians(az_deg)
    return TX + r * math.sin(a), TZ + r * math.cos(a)


@njit(inline='always', fastmath=True)
def _lod(scale, fp, lo, hi):
    o = math.log2(max(scale / max(fp * 2.0, 1e-4), 1.0))
    return min(max(o, lo), hi)


@njit(inline='always', fastmath=True)
def smax(a, b, k):
    if k <= 0.0:
        return max(a, b)
    h = max(k - abs(a - b), 0.0) / k
    return max(a, b) + h * h * k * 0.25


# ------------------------------------------------------------------ terrain ---
CELL = 8200.0                           # one procedural massif per cell (most cells), beyond her own neighbourhood
FLOOR = CLOUD_Y - 480.0                 # the land under the cloud sea
QUIET = 6200.0                          # no procedural massif centred within this of her summit (art-directed)


@njit(inline='always', fastmath=True)
def _h01(i, j, k):
    a = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791)
    a = a & 0xFFFFFFFF
    a = ((a ^ 61) ^ (a >> 16)) & 0xFFFFFFFF
    a = (a + (a << 3)) & 0xFFFFFFFF
    a = a ^ (a >> 4)
    a = (a * 0x27D4EB2D) & 0xFFFFFFFF
    a = a ^ (a >> 15)
    return a * (1.0 / 4294967296.0)


@njit(inline='always', fastmath=True)
def smin(a, b, k):
    return -smax(-a, -b, k)


@njit(inline='always', fastmath=True)
def bowl(a, b, hc, Rc, Dm, sf, kb, kl):
    """A cirque in its own frame (a: along the fall line, + downhill; b: across): a floor tilted gently downhill,
    a steep back wall uphill and steep side walls; downhill it opens (the massif falls faster than the floor)."""
    ua = max(-a / Rc, 0.0)
    ub = abs(b) / Rc
    return hc - sf * a + kb * Dm * ua * ua + kl * Dm * ub ** 2.4


@njit(inline='always', fastmath=True)
def soft_ridged2(x, y, octaves, seed, eps, gain, pers):
    """Ridged multifractal with ROUNDED crests (1 - sqrt(n^2 + eps) instead of 1 - |n|) and a low persistence:
    real ridge-and-cirque networks, worn old. ~[0, 1]."""
    s = 0.0
    a = 0.5
    w = 1.0
    norm = 0.0
    n = int(octaves)
    frac = octaves - n
    e2 = eps
    for o in range(n + 1):
        wgt = 1.0 if o < n else frac
        if wgt <= 0.0:
            break
        g = gnoise2(x, y, seed + o * 131)
        r = 1.0 - math.sqrt(g * g + e2) + math.sqrt(e2)
        r = r * r
        r *= w
        w = min(max(r * gain, 0.0), 1.0)
        s += wgt * r * a
        norm += a
        nx = 1.6 * x - 1.2 * y
        ny = 1.2 * x + 1.6 * y
        x = nx * 1.015 + 17.13
        y = ny * 1.015 - 9.71
        a *= pers
        e2 *= 0.55
    return s / max(norm, 1e-6)


@njit(inline='always', fastmath=True)
def massif(u, v, top, R, p, rnd, ncq, cq0, seed, cqd, x, z, fp, keep):
    """An art-directed massif in its own (u, v) frame: a rounded summit dome whose flanks steepen with depth, carved
    by the ranges' own soft-ridged spurs and cirque-headed valleys (none within `keep` of the top; cqd = their
    relief as a fraction of the massif's height)."""
    r = math.sqrt(u * u + v * v)
    Dm = top - FLOOR
    xr = r / R
    xr = math.sqrt(xr * xr + rnd * rnd) - rnd
    h = top - Dm * xr ** p
    if cqd > 0.0:
        S = 0.55 * R
        o = _lod(S, fp, 1.0, 8.0)
        rg = soft_ridged2(x / S + seed * 0.37, z / S - seed * 0.23, o, seed + 300, 0.03, 1.9, 0.46)
        h += cqd * Dm * (rg - 0.46) * smoothstep(max(keep, 0.08 * R), 0.42 * R, r)
    return h


@njit(fastmath=True, cache=True)
def field(x, z, fp, hcur):
    """B's ranges: broad old massifs from a soft-ridged multifractal (rounded crests, cirque-headed valleys), their
    tops compressed into broad rounded summits, clustered into ranges with cloud-filled basins between (layering)."""
    ow = _lod(16000.0, fp, 1.0, 3.0)
    wx = fbm2(x / 16000.0 + 2.1, z / 16000.0 - 1.3, ow, 201)
    wz = fbm2(x / 16000.0 - 0.7, z / 16000.0 + 3.3, ow, 202)
    S = 7600.0
    o = _lod(S, fp, 1.0, 9.0)
    r = soft_ridged2(x / S + 0.55 * wx, z / S + 0.55 * wz, o, 261, 0.045, 1.9, 0.42)
    oe = _lod(26000.0, fp, 1.0, 2.0)
    env = smoothstep(-0.32, 0.30, fbm2(x / 26000.0 + 0.9, z / 26000.0 - 3.1, oe, 262))
    hr = r * (0.50 + 0.50 * env)
    # ~30% of the land stands above the cloud; the tops are compressed into broad rounded summits (old, worn)
    u = hr - 0.63
    if u > 0.0:
        h = CLOUD_Y + 3291.0 * (1.0 - math.exp(-1.5 * u)) / 1.5
    else:
        h = CLOUD_Y + 3291.0 * u
    # the cloud bay she looks over (NNE of her) and her own neighbourhood kept low under her massif
    dx = x - (TX + 5200.0 * 0.4226)
    dz = z - (TZ + 5200.0 * 0.9063)
    h -= 520.0 * math.exp(-(dx * dx + dz * dz) / (2.0 * 3600.0 * 3600.0))
    return h


@njit(fastmath=True, cache=True)
def uplift(x, z, CR, k):
    """Type 3 rows: a broad raise of the ranges (her massif, the sunrise shoulder): 0 cx | 1 cz | 2 A | 3 sigma
    | 5 aniso | 6 angle."""
    dx = x - CR[k, 0]
    dz = z - CR[k, 1]
    ca = math.cos(CR[k, 6])
    sa = math.sin(CR[k, 6])
    u = dx * ca + dz * sa
    v = (-dx * sa + dz * ca) * CR[k, 5]
    q = (u * u + v * v) / (2.0 * CR[k, 3] * CR[k, 3])
    if q > 12.0:
        return 0.0
    return CR[k, 2] * math.exp(-q)


@njit(fastmath=True, cache=True)
def dome(x, z, fp, CR, k, hcur):
    """An art-directed massif row (her neighbours; her own massif is row 0)."""
    cx = CR[k, 0]
    cz = CR[k, 1]
    dx = x - cx
    dz = z - cz
    reach = CR[k, 13]
    d2 = dx * dx + dz * dz
    if d2 > reach * reach:
        return -1e5
    top = CR[k, 2]
    R = CR[k, 3]
    p = CR[k, 4]
    an = CR[k, 5]
    rb = math.sqrt(d2) / max(an, 1.0)
    xb = rb / R
    if top - (top - FLOOR) * max(xb - 0.02, 0.0) ** p < hcur - CR[k, 9]:
        return -1e5
    ang = CR[k, 6]
    ca = math.cos(ang)
    sa = math.sin(ang)
    u = dx * ca + dz * sa
    v = (-dx * sa + dz * ca) * an
    return massif(u, v, top, R, p, CR[k, 14], int(CR[k, 11]), CR[k, 8], int(CR[k, 7]), CR[k, 10], x, z, fp,
                  CR[k, 15])


@njit(fastmath=True, cache=True)
def bridge(x, z, fp, CR, k, hcur):
    """Broad round-crested ridge a->b: crest(t) - s*(sqrt(d^2 + a^2) - a), a = s * crest radius."""
    ax = CR[k, 0]
    az = CR[k, 1]
    bx = CR[k, 3]
    bz = CR[k, 4]
    ddx = bx - ax
    ddz = bz - az
    L2 = ddx * ddx + ddz * ddz
    t = ((x - ax) * ddx + (z - az) * ddz) / L2
    t = min(max(t, 0.0), 1.0)
    ex = x - (ax + t * ddx)
    ez = z - (az + t * ddz)
    d0 = math.sqrt(ex * ex + ez * ez)
    if d0 > CR[k, 13]:
        return -1e5
    s = CR[k, 8]
    a = s * CR[k, 6]
    ya = CR[k, 2]
    yb = CR[k, 5]
    yc = ya + (yb - ya) * t ** 1.3
    ub = yc - s * (math.sqrt(max(d0 - 30.0, 0.0) ** 2 + a * a) - a) + 25.0
    if ub < hcur - CR[k, 9]:
        return -1e5
    seed = int(CR[k, 7])
    ow = _lod(260.0, fp, 1.0, 3.0)
    wx = x + 22.0 * fbm2(x / 260.0 + 1.3, z / 260.0, ow, seed)
    wz = z + 22.0 * fbm2(x / 260.0 - 4.1, z / 260.0 + 2.2, ow, seed + 1)
    t = ((wx - ax) * ddx + (wz - az) * ddz) / L2
    t = min(max(t, 0.0), 1.0)
    ex = wx - (ax + t * ddx)
    ez = wz - (az + t * ddz)
    d = math.sqrt(ex * ex + ez * ez)
    yc = ya + (yb - ya) * t ** 1.3
    L = math.sqrt(L2)
    yc += 7.0 * math.sin(t * L / 150.0 + seed) * smoothstep(0.03, 0.2, t)
    side = ddx * ez - ddz * ex
    sk = s * (1.0 + 0.2 * (1.0 if side > 0.0 else -1.0))
    h = yc - sk * (math.sqrt(d * d + a * a) - a) - 0.0012 * d * d
    return h


@njit(fastmath=True, cache=True)
def knoll(x, z, fp):
    """Her summit's top: a rounded boss (frost-shattered rock and old snow) that falls away on every side, flat
    enough in its middle for the beacon, the cairn and two figures; low boulders round its rim."""
    dx = x - TX
    dz = z - TZ
    d2 = dx * dx + dz * dz
    if d2 > 140.0 * 140.0:
        return -1e5
    d = math.sqrt(d2)
    h = TOP_Y - 0.030 * d ** 1.62
    o = _lod(5.0, fp, 0.0, 4.0)
    if o > 0.0:
        rim = smoothstep(4.0, 9.0, d) * (1.0 - smoothstep(30.0, 60.0, d))
        h += 0.55 * rim * (fbm2(x / 5.0 + 0.3, z / 5.0 - 1.7, o, 301) + 0.2)
        o2 = _lod(1.4, fp, 0.0, 4.0)
        if o2 > 0.0:
            h += 0.08 * fbm2(x / 1.4, z / 1.4, o2, 302) * (0.3 + 0.7 * smoothstep(2.5, 6.0, d))
    return h


@njit(fastmath=True, cache=True)
def h_rock(x, z, fp, CR):
    up = 0.0
    for k in range(CR.shape[0]):
        if CR[k, 12] > 2.5:
            up += uplift(x, z, CR, k)
    h = field(x, z, fp, FLOOR) + up
    for k in range(CR.shape[0]):
        if CR[k, 12] < 0.5:
            hc = dome(x, z, fp, CR, k, h)
            if hc > -1e4:
                h = smax(h, hc, CR[k, 9])
    for k in range(CR.shape[0]):
        if CR[k, 12] > 1.5 and CR[k, 12] < 2.5:
            ddx = x - CR[k, 0]
            ddz = z - CR[k, 1]
            if ddx * ddx + ddz * ddz < CR[k, 13] * CR[k, 13]:
                ca_ = math.sin(CR[k, 4])
                sa_ = math.cos(CR[k, 4])
                aa = ddx * ca_ + ddz * sa_
                bb = -ddx * sa_ + ddz * ca_
                hb = bowl(aa, bb, CR[k, 2], CR[k, 3], CR[k, 5], CR[k, 6], CR[k, 7], CR[k, 8])
                h = smin(h, hb, CR[k, 9])
    for k in range(CR.shape[0]):
        if CR[k, 12] > 0.5 and CR[k, 12] < 1.5:
            hc = bridge(x, z, fp, CR, k, h)
            if hc > -1e4:
                h = smax(h, hc, CR[k, 9])
    hk = knoll(x, z, fp)
    if hk > -1e4:
        h = smax(h, hk, 4.0)
    os_ = _lod(420.0, fp, 0.0, 2.0)
    if os_ > 0.0:
        h += 2.0 * fbm2(x / 420.0 - 1.1, z / 420.0 + 4.4, os_, 232)
    return h


@njit(fastmath=True, cache=True)
def h_cloud(x, z, fp, t):
    return W0.h_cloud(x, z, fp, t)


@njit(fastmath=True, cache=True)
def hfun(x, z, fp, P, CR):
    """World height incl. Earth curvature relative to (P[0], P[1]); P[2] = time (s)."""
    dx = x - P[0]
    dz = z - P[1]
    curv = (dx * dx + dz * dz) / (2.0 * R_EARTH)
    h = h_rock(x, z, fp, CR)
    hc = h_cloud(x, z, fp, P[2])
    if hc > h:
        h = hc
    return h - curv


# ------------------------------------------------------------------ marcher ---

@njit(parallel=True, fastmath=True, cache=True)
def march(P, CR, C, d0, dmax, rel, kgap, hmax, nbis, out_d):
    """Column-coherent heightfield march (world.march with B's height function)."""
    cxw, cyw, czw = C[0], C[1], C[2]
    fx, fz, rx, rz = C[3], C[4], C[5], C[6]
    f, cx, cyy = C[7], C[8], C[9]
    H, W = out_d.shape
    pix = 1.0 / f
    for i in prange(W):
        xo = i + 0.5 - cx
        dxh = fx * f + rx * xo
        dzh = fz * f + rz * xo
        hl = math.sqrt(dxh * dxh + dzh * dzh)
        dxh /= hl
        dzh /= hl
        pixh = 1.0 / hl
        da = d0
        ha = hfun(cxw + dxh * da, czw + dzh * da, da * pix, P, CR)
        missed = False
        for j in range(H - 1, -1, -1):
            if missed:
                out_d[j, i] = 1e30
                continue
            s = (cyy - (j + 0.5)) * pixh
            if cyw + s * da < ha:
                out_d[j, i] = da
                continue
            d = da
            h = ha
            hit = False
            while d < dmax:
                gap = cyw + s * d - h
                step = rel * d
                g = kgap * gap
                if g > step:
                    step = g
                dn = d + step
                hn = hfun(cxw + dxh * dn, czw + dzh * dn, dn * pix, P, CR)
                yn = cyw + s * dn
                if yn < hn:
                    lo = d
                    hi = dn
                    hlo = h
                    for _k in range(nbis):
                        mid = 0.5 * (lo + hi)
                        hm = hfun(cxw + dxh * mid, czw + dzh * mid, mid * pix, P, CR)
                        if cyw + s * mid < hm:
                            hi = mid
                        else:
                            lo = mid
                            hlo = hm
                    out_d[j, i] = 0.5 * (lo + hi)
                    da = lo
                    ha = hlo
                    hit = True
                    break
                if yn > hmax and s >= 0.0:
                    break
                d = dn
                h = hn
            if not hit:
                missed = True
                out_d[j, i] = 1e30


# ------------------------------------------------------------------ G-buffer ---
G_FLAG, G_DIST, G_X, G_Y, G_Z, G_NX, G_NY, G_NZ, G_SNOW, G_RV, G_C0, G_DX, G_DY, G_DZ, G_TROUGH, G_FP = range(16)
NG = 16          # + K moon-visibility channels appended by build(..., moons=K dirs)


@njit(inline='always', fastmath=True)
def airmass(c):
    cd = max(c * 57.29578, 0.0)
    return 1.0 / (math.sin(math.radians(cd)) + 0.50572 * (cd + 6.07995) ** -1.6364)


@njit(inline='always', fastmath=True)
def clearance0(P, CR, x, y, z, hx, hz, fp, nsteps, tmax):
    t = max(fp * 2.0, 0.5)
    grow = (tmax / t) ** (1.0 / nsteps)
    c = 1.0
    for _s in range(nsteps):
        h = hfun(x + hx * t, z + hz * t, fp + t * 0.002, P, CR)
        a = (y - h) / t
        if a < c:
            c = a
        t *= grow
    return c


@njit(inline='always', fastmath=True)
def clearance0_fine(P, CR, x, y, z, hx, hz, fp, kstep, smin, tmax):
    t = max(fp * 2.0, 0.5)
    c = 1.0
    while t < tmax:
        h = hfun(x + hx * t, z + hz * t, fp + t * 0.002, P, CR)
        a = (y - h) / t
        if a < c:
            c = a
        t += max(kstep * t, smin)
    return c


@njit(inline='always', fastmath=True)
def soft_vis(P, CR, x, y, z, lx, ly, lz, fp, nsteps, tmax, k):
    """Soft visibility toward a direction (moon): 1 lit, 0 in shadow."""
    res = 1.0
    t = max(fp * 2.0, 0.4)
    grow = (tmax / t) ** (1.0 / nsteps)
    for _s in range(nsteps):
        h = hfun(x + lx * t, z + lz * t, fp + t * 0.003, P, CR)
        dh = y + ly * t - h
        r = k * dh / t
        if r < res:
            res = r
        if res < 0.0:
            return 0.0
        t *= grow
    res = min(res, 1.0)
    return res * res * (3.0 - 2.0 * res)


@njit(parallel=True, fastmath=True, cache=True)
def gbuffer(C, D, P, CR, hx, hz, nsteps, tmax, snow_bias, G, kstep, MD, mk):
    """Fill G (H, W, NG + len(MD)). hx, hz: the sun's azimuth for the clearance channel (nsteps 0 = skip).
    MD: (K, 3) moon directions -> soft visibility channels NG..NG+K-1 (k = softness)."""
    H, W = D.shape
    K = MD.shape[0]
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
            G[j, i, G_DX] = dxh / nn
            G[j, i, G_DY] = vy / nn
            G[j, i, G_DZ] = dzh / nn
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
            hf = h_rock(x, z, fp, CR)
            hc = h_cloud(x, z, fp, P[2])
            cloud = hc > hf
            if cloud:
                h0 = hc
                hxx = h_cloud(x + e, z, fp, P[2])
                hzz = h_cloud(x, z + e, fp, P[2])
            else:
                h0 = hf
                hxx = h_rock(x + e, z, fp, CR)
                hzz = h_rock(x, z + e, fp, CR)
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
                # the form's own slope (a coarser normal) decides snow; a finer one adds ledges near the camera
                es = max(fp * 5.0, 12.0)
                hsx = h_rock(x + es, z, fp * 3.0, CR)
                hsz = h_rock(x, z + es, fp * 3.0, CR)
                nsx = -(hsx - h0) / es
                nsz = -(hsz - h0) / es
                nsy = 1.0 / math.sqrt(nsx * nsx + nsz * nsz + 1.0)
                # thresholds vary in broad patches and in HORIZONTAL bands (strata): never along the fall line
                band = gnoise2(h0 / 40.0 + x / 1400.0, z / 1400.0 - h0 / 600.0, 74)
                patch = gnoise2(x / 1100.0, z / 1100.0, 73)
                sv = nsy + 0.035 * patch + 0.03 * band + snow_bias
                snow = smoothstep(0.56, 0.66, sv)
                wf = smoothstep(4.0, 0.5, fp)
                if wf > 0.0:
                    sf = 0.05 * gnoise2(x / 3.0, z / 3.0, 75) + 0.03 * gnoise2(x * 1.1, z * 1.1, 76)
                    sv2 = 0.55 * ny + 0.45 * nsy + 0.05 * patch + sf + snow_bias
                    snow = snow + (smoothstep(0.80, 0.90, sv2) - snow) * wf
                # her stony summit: frost-shattered rock, old snow only in the hollows
                dtx = x - TX
                dtz = z - TZ
                dtop = math.sqrt(dtx * dtx + dtz * dtz)
                if dtop < 40.0:
                    # frost and old snow in the grain of the stones (fine), never blotches
                    fine = gnoise2(x / 0.45, z / 0.45, 91) * 0.6 + gnoise2(x / 0.17, z / 0.17, 92) * 0.4
                    frost = 0.18 + 0.30 * smoothstep(-0.2, 0.6, fine) + 0.25 * smoothstep(0.93, 0.99, ny)
                    snow = snow + (min(frost, 0.75) - snow) * (1.0 - smoothstep(18.0, 40.0, dtop))
                # rock tone: strata (horizontal) and broad patches
                rv = 0.80 + 0.35 * (0.5 + 0.5 * gnoise2(h0 / 7.0 + x / 300.0, z / 300.0, 81))
                rv *= 0.88 + 0.24 * (0.5 + 0.5 * patch)
                # snow tone: wind-packed vs fresh, very broad
                if snow > 0.0:
                    G[j, i, G_TROUGH] = 0.5 + 0.5 * gnoise2(x / 1600.0 + 3.0, z / 1600.0, 83)
            c0 = 1.0
            if nsteps > 0:
                if kstep > 0.0:
                    c0 = clearance0_fine(P, CR, x, yw + max(fp * 1.5, 0.3), z, hx, hz, fp, kstep, 4.0, tmax)
                else:
                    c0 = clearance0(P, CR, x, yw + max(fp * 1.5, 0.3), z, hx, hz, fp, nsteps, tmax)
            for m in range(K):
                G[j, i, NG + m] = soft_vis(P, CR, x, yw + max(fp * 1.5, 0.3), z, MD[m, 0], MD[m, 1], MD[m, 2],
                                           fp, 56, 40000.0, mk)
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
            if cloud:
                G[j, i, G_TROUGH] = smoothstep(CLOUD_Y - 120.0, CLOUD_Y + 110.0, h0)
            G[j, i, G_FP] = fp


def build(scam, P, CR, sun_az=None, nsteps=72, tmax=160000.0, snow_bias=0.0, dmax=180000.0, hmax=2500.0,
          kstep=0.0, moons=None, mk=10.0):
    """March the source camera and fill the G-buffer (H, W, NG + K) float32."""
    C = scam.params()
    D = np.zeros((scam.H, scam.W))
    march(P, CR, C, 0.2, dmax, 0.0035, 0.35, hmax, 9, D)
    MD = np.zeros((0, 3)) if moons is None else np.asarray(moons, np.float64).reshape(-1, 3)
    G = np.zeros((scam.H, scam.W, NG + MD.shape[0]), np.float32)
    if sun_az is None:
        hx, hz, ns = 0.0, 1.0, 0
    else:
        a = math.radians(sun_az)
        hx, hz, ns = math.sin(a), math.cos(a), int(nsteps)
    gbuffer(C, D, P, CR, hx, hz, ns, float(tmax), float(snow_bias), G, float(kstep), MD, float(mk))
    return G


def sun_vec(el_deg, az_deg):
    e, a = math.radians(el_deg), math.radians(az_deg)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


# ------------------------------------------------------------------ skies + shading ---
# SKY kind 0 = dusk: 0-2 anti-solar horizontal dir | 3 shadow top el | 4-6 zenith | 7-9 horizon | 10-12 shadow col
#   13-15 belt col | 16 belt width | 17 gain | 18 shadow edge softness (rad)
# SKY kind 1 = dawn: 0-2 sun dir | 3-5 aureole col | 6 disc I | 7 disc radius | 8-10 zenith | 11-13 horizon warm
#   14-16 horizon cool | 17 aureole near | 18 aureole wide | 19 band scale | 20 warm az width
# SKY kind 2 = night: 0-2 moon dir | 3-5 zenith | 6-8 horizon | 9 moon halo I | 10 halo width | 11-13 halo col
#   14 moon disc I | 15 moon radius | 16 horizon band height | 17 gain | 18-20 dawn tint (grey) | 21 dawn amount
#   22-24 east dir (the dawn side) | 25 east glow width

@njit(inline='always', fastmath=True)
def dusk_sky(dx, dy, dz, SK):
    el = math.asin(min(max(dy, -1.0), 1.0))
    hl = math.sqrt(dx * dx + dz * dz) + 1e-9
    ca = (dx * SK[0] + dz * SK[2]) / hl
    wa = (0.5 + 0.5 * ca) ** 1.2
    u = smoothstep(0.0, 0.60, max(el, 0.0)) ** 0.55
    r = SK[7] + (SK[4] - SK[7]) * u
    g = SK[8] + (SK[5] - SK[8]) * u
    b = SK[9] + (SK[6] - SK[9]) * u
    es = SK[3]
    sw = SK[18]
    # the Earth's shadow: a soft-topped band (never a seam), strongest opposite the sun
    sh = (1.0 - smoothstep(es - sw, es + sw, el)) * (0.35 + 0.65 * wa)
    belt = math.exp(-((el - es - 0.6 * sw) / SK[16]) ** 2) * (0.25 + 0.75 * wa) * (1.0 - 0.7 * sh)
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
    u = smoothstep(0.0, 0.16 + 0.10 * warm, max(el, 0.0)) ** 0.7
    r = hr + (SD[8] - hr) * u
    g = hg + (SD[9] - hg) * u
    b = hb + (SD[10] - hb) * u
    band = math.exp(-max(el, 0.0) / SD[19]) * warm
    r += 0.6 * band * SD[11]
    g += 0.6 * band * SD[12]
    b += 0.6 * band * SD[13]
    cosg = dx * sx + dy * sy + dz * sz
    ang = math.acos(min(max(cosg, -1.0), 1.0))
    au = SD[17] * math.exp(-ang / 0.035) + SD[18] * math.exp(-ang / 0.22)
    r += au * SD[3]
    g += au * SD[4]
    b += au * SD[5]
    R = SD[7]
    if ang < R + 2.0 * pix_ang:
        cov = min(max((R - ang) / pix_ang + 0.5, 0.0), 1.0)
        rr = min((ang / R) ** 2, 1.0)
        limb = 0.55 + 0.45 * math.sqrt(1.0 - rr)
        I = SD[6] * cov * limb
        r += I * 1.0
        g += I * 0.86
        b += I * 0.66
    return r, g, b


@njit(inline='always', fastmath=True)
def night_sky(dx, dy, dz, SN, pix_ang):
    el = math.asin(min(max(dy, -1.0), 1.0))
    u = smoothstep(-0.02, SN[16], el) ** 0.6
    r = SN[6] + (SN[3] - SN[6]) * u
    g = SN[7] + (SN[4] - SN[7]) * u
    b = SN[8] + (SN[5] - SN[8]) * u
    # the east greys (dawn amount), from the east horizon up
    if SN[21] > 0.0:
        hl = math.sqrt(dx * dx + dz * dz) + 1e-9
        ca = (dx * SN[22] + dz * SN[24]) / hl
        wa = math.exp(-((1.0 - ca) / SN[25]) ** 2)
        dw = SN[21] * (0.35 + 0.65 * wa) * math.exp(-max(el, 0.0) / 0.35)
        r += dw * SN[18]
        g += dw * SN[19]
        b += dw * SN[20]
    cosg = dx * SN[0] + dy * SN[1] + dz * SN[2]
    ang = math.acos(min(max(cosg, -1.0), 1.0))
    halo = SN[9] * math.exp(-ang / SN[10])
    r += halo * SN[11]
    g += halo * SN[12]
    b += halo * SN[13]
    R = SN[15]
    if SN[14] > 0.0 and ang < R + 2.0 * pix_ang:
        cov = min(max((R - ang) / pix_ang + 0.5, 0.0), 1.0)
        r += SN[14] * cov * 0.92
        g += SN[14] * cov * 0.95
        b += SN[14] * cov * 1.0
    return r * SN[17], g * SN[17], b * SN[17]


# LP (light): 0-2 sun dir | 3 sun I | 4-6 Rayleigh tau | 7 aerosol tau | 8 sun penumbra (rad) | 9 cloud albedo
#   10 cloud amb gain | 11 belt fill (faces toward the sunset) | 12 sheen | 13-15 glow dir | 16-18 glow col
#   19 sky kind (0 dusk / 1 dawn / 2 night) | 20 pix_ang | 21 rock warmth | 22 snow blue (night) | 23 moon I
#   24-26 moon dir | 27-29 moon col | 30 hero red floor | 31 hero penumbra | 32 moon vis ch a (-1 none) | 33 ch b
#   34 a->b mix | 35 moon wrap on cloud | 36 fire I | 37-39 fire pos | 40-42 fire col | 43 fire reach (m)
#   44 sun on (1) | 45 cloud fwd-scatter toward moon
NLP = 48


@njit(parallel=True, fastmath=True, cache=True)
def shade(G, LP, SKY, amb, fogp, cam_y, out):
    H, W = G.shape[0], G.shape[1]
    lx, ly, lz = LP[0], LP[1], LP[2]
    se = ly
    rsun = LP[8]
    ma = int(LP[32])
    mb = int(LP[33])
    mmix = LP[34]
    kind = LP[19]
    for j in prange(H):
        for i in range(W):
            dx = G[j, i, G_DX]
            dy = G[j, i, G_DY]
            dz = G[j, i, G_DZ]
            flag = G[j, i, G_FLAG]
            if flag == 0.0:
                if kind < 0.5:
                    r, g, b = dusk_sky(dx, dy, dz, SKY)
                elif kind < 1.5:
                    r, g, b = dawn_sky(dx, dy, dz, SKY, LP[20])
                else:
                    r, g, b = night_sky(dx, dy, dz, SKY, LP[20])
                out[j, i, 0] = r
                out[j, i, 1] = g
                out[j, i, 2] = b
                continue
            nx = G[j, i, G_NX]
            ny = G[j, i, G_NY]
            nz = G[j, i, G_NZ]
            # the sun, through its grazing path (clearance channel)
            sr = 0.0
            sg = 0.0
            sb = 0.0
            vis = 0.0
            ndl = nx * lx + ny * ly + nz * lz
            if LP[44] > 0.5:
                c = G[j, i, G_C0] + se
                hero = flag == 1.0 and G[j, i, G_TROUGH] > 1.5
                rs = LP[31] if (hero and LP[31] > 0.0) else rsun
                vis = smoothstep(-rs, rs, c)
                if vis > 0.0:
                    m = airmass(c)
                    sr = vis * math.exp(-(LP[4] + LP[7]) * m)
                    sg = vis * math.exp(-(LP[5] + LP[7]) * m)
                    sb = vis * math.exp(-(LP[6] + LP[7]) * m)
                    if hero and LP[30] > 0.0:
                        sr = max(sr, vis * LP[30])
                        sg = max(sg, vis * LP[30] * 0.30)
                        sb = max(sb, vis * LP[30] * 0.34)
                        ndl = max(ndl, 0.55)
            gw = max((nx * LP[13] + ny * LP[14] + nz * LP[15]) * 0.5 + 0.5, 0.0) ** 1.5
            # the moon (visibility interpolated between two precomputed channels)
            mv = 1.0
            if ma >= 0:
                va = G[j, i, NG + ma]
                vb = G[j, i, NG + mb] if mb >= 0 else va
                mv = va + (vb - va) * mmix
            mdl = nx * LP[24] + ny * LP[25] + nz * LP[26]
            # fire light (a point light at the beacon: inverse square, soft reach)
            fr_ = 0.0
            if LP[36] > 0.0:
                px = G[j, i, G_X] - LP[37]
                py = G[j, i, G_Y] - LP[38]
                pz = G[j, i, G_Z] - LP[39]
                d2 = px * px + py * py + pz * pz
                if d2 < LP[43] * LP[43]:
                    dd = math.sqrt(d2) + 1e-6
                    fdl = -(nx * px + ny * py + nz * pz) / dd
                    fr_ = LP[36] * max(fdl * 0.8 + 0.2, 0.0) / (d2 + 1.0) * (1.0 - smoothstep(0.5 * LP[43], LP[43], dd))
            if flag == 2.0:
                wrap = max((ndl + 0.6) / 1.6, 0.0)
                ca = LP[9]
                sky = 0.55 + 0.45 * ny
                mw = max((mdl + LP[35]) / (1.0 + LP[35]), 0.0) * mv
                cr = ca * (LP[3] * sr * wrap + amb[0] * LP[10] * sky + LP[16] * gw + LP[23] * mw * LP[27])
                cg = ca * (LP[3] * sg * wrap + amb[1] * LP[10] * sky + LP[17] * gw + LP[23] * mw * LP[28])
                cb = ca * (LP[3] * sb * wrap + amb[2] * LP[10] * sky + LP[18] * gw + LP[23] * mw * LP[29])
                tk = 0.55 + 0.45 * G[j, i, G_TROUGH]
                cr *= tk
                cg *= tk
                cb *= tk
                if LP[45] > 0.0:
                    fw = max(dx * LP[24] + dy * LP[25] + dz * LP[26], 0.0) ** 6
                    cr += LP[45] * fw * LP[27] * mv
                    cg += LP[45] * fw * LP[28] * mv
                    cb += LP[45] * fw * LP[29] * mv
            else:
                snow = G[j, i, G_SNOW]
                rv = G[j, i, G_RV]
                w = LP[21]
                a_r = (0.060 + 0.018 * w) * rv
                a_g = (0.060 + 0.008 * w) * rv
                a_b = (0.066 - 0.004 * w) * rv
                tone = G[j, i, G_TROUGH] if G[j, i, G_TROUGH] <= 1.0 else 0.5
                sa = 0.80 + 0.06 * tone
                ar = a_r + (sa * 0.99 - a_r) * snow
                ag = a_g + (sa * 1.00 - a_g) * snow
                ab = a_b + (sa * (1.04 + LP[22]) - a_b) * snow
                dif = max(ndl, 0.0)
                skyl = 0.55 + 0.45 * ny
                west = max(nx * lx + nz * lz, 0.0) / (math.sqrt(lx * lx + lz * lz) + 1e-9)
                belt = LP[11] * west * (1.0 - min(sr * 4.0, 1.0))
                mk_ = LP[23] * max(mdl, 0.0) * mv
                cr = ar * (LP[3] * sr * dif + amb[0] * skyl + belt * 1.0 + LP[16] * gw + mk_ * LP[27] + fr_ * LP[40])
                cg = ag * (LP[3] * sg * dif + amb[1] * skyl + belt * 0.60 + LP[17] * gw + mk_ * LP[28] + fr_ * LP[41])
                cb = ab * (LP[3] * sb * dif + amb[2] * skyl + belt * 0.70 + LP[18] * gw + mk_ * LP[29] + fr_ * LP[42])
                if snow > 0.0 and dif > 0.0 and vis > 0.0:
                    hx_ = lx - dx
                    hy_ = ly - dy
                    hz_ = lz - dz
                    hl_ = math.sqrt(hx_ * hx_ + hy_ * hy_ + hz_ * hz_) + 1e-9
                    nh = max((nx * hx_ + ny * hy_ + nz * hz_) / hl_, 0.0)
                    sp = snow * LP[12] * nh ** 16
                    cr += sp * LP[3] * sr
                    cg += sp * LP[3] * sg
                    cb += sp * LP[3] * sb
            dist = G[j, i, G_DIST]
            yw = G[j, i, G_Y]
            tau = height_fog_tau(dist, cam_y, yw, fogp[0], fogp[1])
            tau += height_fog_tau(dist, cam_y - CLOUD_Y, yw - CLOUD_Y, fogp[2], fogp[3])
            tr = math.exp(-tau)
            cosv = dx * fogp[12] + dy * fogp[13] + dz * fogp[14]
            wv = max(cosv, 0.0) ** fogp[8]
            ph = 1.0 + fogp[4] * wv
            fr = (fogp[5] * (1.0 - wv) + fogp[9] * wv) * ph
            fg = (fogp[6] * (1.0 - wv) + fogp[10] * wv) * ph
            fb = (fogp[7] * (1.0 - wv) + fogp[11] * wv) * ph
            out[j, i, 0] = cr * tr + fr * (1.0 - tr)
            out[j, i, 1] = cg * tr + fg * (1.0 - tr)
            out[j, i, 2] = cb * tr + fb * (1.0 - tr)


# ------------------------------------------------------------------ utilities ---

@njit(parallel=True, fastmath=True, cache=True)
def height_grid(x0, z0, dx, nx, nz, CR, out):
    for j in prange(nz):
        for i in range(nx):
            out[j, i] = h_rock(x0 + i * dx, z0 + j * dx, dx, CR)


def ground(x, z, CR, fp=0.05):
    return h_rock(float(x), float(z), fp, CR)


def summit_near(x, z, CR, rad=60.0, n=25):
    best = (-1e9, x, z)
    for _it in range(3):
        for xx in np.linspace(best[1] - rad, best[1] + rad, n):
            for zz in np.linspace(best[2] - rad, best[2] + rad, n):
                h = ground(xx, zz, CR)
                if h > best[0]:
                    best = (h, xx, zz)
        rad *= 0.25
    return np.array([best[1], best[0], best[2]])


# ------------------------------------------------------------------ B's set ---
_R0 = polar(RIDGE_AZ, 30.0)
_R1 = polar(RIDGE_AZ + 4.0, 1250.0)
_R2 = polar(RIDGE_AZ + 9.0, 2900.0)


def cirque_row(az_open, r_centre, floor, Rc, Dm=1430.0, sf=0.12, kb=0.9, kl=0.6, k=25.0, from_top=True):
    """A cirque carved into whatever is there: centred r_centre from her summit along az_open (its fall line)."""
    cx, cz = polar(az_open, r_centre)
    r = np.zeros(NCR)
    r[:10] = [cx, cz, floor, Rc, math.radians(az_open), Dm, sf, kb, kl, k]
    r[12] = 2.0
    r[13] = 1.6 * Rc + 200.0
    return r


def uplift_row(cx, cz, A, sigma, aniso=1.0, ang=0.0):
    r = np.zeros(NCR)
    r[:7] = [cx, cz, A, sigma, 0.0, aniso, math.radians(ang)]
    r[12] = 3.0
    return r


def set_rows():
    rows = [
        # her massif: a broad old dome, its flanks carved by the ranges' spurs and valleys, the summit left round
        dome_row(TX + 120.0, TZ - 260.0, TOP_Y - 13.0, 2700.0, p=1.6, aniso=1.2, ang=-35.0, seed=7, k=40.0,
                 detail=0.30, rnd=0.04, keep=110.0),
        # the NE ridge (the way up and down): round-crested snow, into the cloud at ~2.9 km
        ridge_row((_R0[0], TOP_Y - 3.0, _R0[1]), (_R1[0], 40.0, _R1[1]), crest=30.0, seed=41, slope=0.75, k=30.0),
        ridge_row((_R1[0], 40.0, _R1[1]), (_R2[0], -760.0, _R2[1]), crest=40.0, seed=43, slope=0.9, k=45.0),
    ]
    # the sunrise shoulder: a broad whaleback ESE, a little higher than her summit (she sees the sun last)
    sx, sz = polar(124.0, 3000.0)
    rows.append(dome_row(sx, sz, TOP_Y + 18.0, 2300.0, p=1.7, aniso=1.7, ang=124.0 - 90.0, seed=11, k=120.0,
                         detail=0.30, rnd=0.05, keep=250.0))
    return np.array(rows)


CR_B = set_rows()
