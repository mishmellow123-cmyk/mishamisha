"""The shepherd's world, extended for flight (THE BEACON RUN and DAWN C).

Same ranges, cloud sea, moon, sky and fog as `shots/montage/s1_peak.py` (its height functions are
imported unchanged, same seeds), plus:
  * art-directed crags (CR rows) that rise where s1's camera could not see (below its summit lip),
    so the camera has something to skim past; each is smooth-maxed into the ranges;
  * close-range detail: fine rock ribs, multi-scale snow (ledges hold snow, ribs shed it),
    rock albedo variation, finer cloud billows at close range;
  * one shading kernel for both shots: a directional key light (moon at night, sun at dawn) with
    long soft shadows, sky ambient, fire point lights, snow sheen, cloud forward-scatter,
    aerial perspective + cloud-top mist.
Everything is a pure function of (frame, pixel).
"""
import math
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
MONT = os.path.abspath(os.path.join(HERE, '..', 'montage'))
if MONT not in sys.path:
    sys.path.insert(0, MONT)

import common as CM                      # noqa: E402
import s1_peak as S1                     # noqa: E402
from mt import sky as SK                 # noqa: E402
from mt.noise import fbm2, ridged2, gnoise2, smoothstep   # noqa: E402

R_EARTH = 6.371e6
CLOUD_Y = S1.CLOUD_Y
MOON_DIR = S1.MOON_DIR.copy()
NEAR_R = 250.0                           # the shepherd's summit patch (world-anchored at the origin)

# crag row (a faceted glacial horn): 0 cx | 1 cz | 2 top | 3 L profile length | 4 s_hi (slope at the top)
#   5 s_lo (slope far down) | 6 aniso | 7 angle | 8 seed | 9 smooth-k | 10 detail amp | 11 summit shelf
#   12 n facets | 13 influence radius (derived)
NCR = 22


def ridge_row(a, b, wl=40.0, wr=40.0, seed=31, k=8.0, detail=0.30, slope=1.35, p=1.35):
    """A knife-edge ridge segment from a=(x, y, z) to b=(x, y, z) (crest heights at the ends), flanks falling
    at `slope` at distance w (left / right of a->b), steepening with exponent p. Row type flag: col 12 = -1.
    Layout: 0 ax | 1 az | 2 ay | 3 bx | 4 bz | 5 by | 6 wl | 7 wr | 8 seed | 9 k | 10 detail | 11 slope
    | 12 -1 | 13 reach."""
    r = np.zeros(NCR)
    r[:12] = [a[0], a[2], a[1], b[0], b[2], b[1], wl, wr, seed, k, detail, slope]
    r[12] = -1.0
    top = max(a[1], b[1])
    r[13] = min(600.0, 2.0 * max(wl, wr) * ((top - (CLOUD_Y - 150.0)) / (slope * max(wl, wr))) ** (1.0 / p) + 40.0)
    r[13] = max(r[13], 80.0)
    return r


def range_row(scale=700.0, amp=560.0, ox=0.0, oz=0.0, seed=21, k=0.0, lip_slope=0.1334, margin=12.0,
              origin=(0.35, 2.5, 1.0), reach=4200.0, below_track=40.0, pyre=(0.0, 0.0), pyre_boost=0.0,
              rise=0.12, spine_az=28.0, spine_w=90.0, spine_boost=0.0, spine_len=1000.0):
    """The shepherd's foothills: s1's ridged range (same recipe as S1.h_far) at a smaller scale, its summit
    envelope held under s1's summit-lip sight line (so s1 never saw it). Row type flag col 12 = -2.
    Layout: 0 ox | 1 oz | 2 scale | 3 amp | 4 origin x | 5 origin z | 6 origin y | 7 lip slope | 8 seed
    | 9 k | 10 margin | 11 reach | 12 -2 | 13 reach."""
    r = np.zeros(NCR)
    r[:12] = [ox, oz, scale, amp, origin[0], origin[2], origin[1], lip_slope, seed, k, margin, reach]
    r[12] = -2.0
    r[13] = reach
    r[14:18] = [below_track, pyre[0], pyre[1], pyre_boost]
    r[18:22] = [math.radians(spine_az), spine_w, spine_boost, spine_len]
    r[13] = rise
    return r


def track_row(a, b):
    """One segment of the flight track (a, b = (x, y, z)); the foothills are shaped relative to it.
    Row type flag col 12 = -3 (not a height primitive itself)."""
    r = np.zeros(NCR)
    r[:6] = [a[0], a[2], a[1], b[0], b[2], b[1]]
    r[12] = -3.0
    return r


def crag_row(cx, cz, top, L=40.0, s_hi=2.4, s_lo=1.05, aniso=1.0, ang=0.0, seed=1, k=10.0, detail=0.28,
             shelf=0.0, nf=4):
    r = np.zeros(NCR)
    r[:13] = [cx, cz, top, L, s_hi, s_lo, aniso, ang, seed, k, detail, shelf, nf]
    reach = (top - (CLOUD_Y - 150.0)) / (s_lo * 0.8)
    r[13] = (reach + shelf + 2.0 * L) * max(aniso, 1.0)
    return r


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


@njit(inline='always', fastmath=True)
def _hh(i, j):
    # hash of two small ints -> [0,1)
    a = (i * 73856093) ^ (j * 19349663)
    a = a & 0xFFFFFFFF
    a = ((a ^ 61) ^ (a >> 16)) & 0xFFFFFFFF
    a = (a + (a << 3)) & 0xFFFFFFFF
    a = a ^ (a >> 4)
    a = (a * 0x27D4EB2D) & 0xFFFFFFFF
    a = a ^ (a >> 15)
    return a * (1.0 / 4294967296.0)


@njit(inline='always', fastmath=True)
def _facet_m(u, v, n, seed):
    m = -1e9
    for j in range(n):
        a = 6.283185307 * (j + 0.35 * (_hh(seed, j * 3 + 1) - 0.5)) / n
        sj = 0.62 + 0.85 * _hh(seed, j * 3 + 2)
        mj = sj * (u * math.cos(a) + v * math.sin(a))
        if mj > m:
            m = mj
    return m


@njit(inline='always', fastmath=True)
def _drop(m, L, shi, slo):
    return slo * m + (shi - slo) * L * (1.0 - math.exp(-m / L))


@njit(fastmath=True, cache=True)
def crag(x, z, fp, CR, k, hcur):
    """Faceted horn; returns -1e5 when it cannot rise above hcur - smooth-k (envelope bound)."""
    cx = CR[k, 0]
    cz = CR[k, 1]
    dx = x - cx
    dz = z - cz
    if dx * dx + dz * dz > CR[k, 13] * CR[k, 13]:
        return -1e5
    top = CR[k, 2]
    L = CR[k, 3]
    shi = CR[k, 4]
    slo = CR[k, 5]
    an = CR[k, 7]
    ca = math.cos(an)
    sa = math.sin(an)
    u = dx * ca + dz * sa
    v = (-dx * sa + dz * ca) / CR[k, 6]
    seed = int(CR[k, 8])
    n = int(CR[k, 12])
    shelf = CR[k, 11]
    det = CR[k, 10]
    # envelope bound (warp can move u,v by <= 0.2 L; detail adds <= 0.55 det min(m, 2L))
    m0 = _facet_m(u, v, n, seed) - shelf
    mb = max(m0 - 0.34 * L * 1.5 * 1.414, 0.0)
    ub = top - _drop(mb, L, shi, slo) + 0.6 * det * min(max(m0, 0.0) + 0.4 * L, 2.0 * L)
    if ub < hcur - CR[k, 9]:
        return -1e5
    # irregular facets: a gentle domain warp
    o1 = _lod(1.3 * L, fp, 1.0, 4.0)
    u += 0.34 * L * fbm2(x / (1.3 * L) + 3.7, z / (1.3 * L), o1, seed)
    v += 0.34 * L * fbm2(x / (1.3 * L) - 1.3, z / (1.3 * L) + 5.1, o1, seed + 1)
    m = max(_facet_m(u, v, n, seed) - shelf, 0.0)
    h = top - _drop(m, L, shi, slo)
    # strata: gently tilted, wavy ledges (treads hold snow, risers shed it)
    T = 0.22 * L
    tw = 0.22 * smoothstep(0.0, 0.3 * L, m) * smoothstep(0.5 * T, 0.12 * T, fp) * (0.5 + 0.5 * gnoise2(x / L, z / L, seed + 9))
    if tw > 0.0:
        ow = _lod(0.8 * L, fp, 1.0, 4.0)
        off = 0.35 * T * fbm2(x / (0.8 * L), z / (0.8 * L), ow, seed + 5) + 0.05 * (x - z)
        q = (h + off) / T
        fq = q - math.floor(q)
        ht = (math.floor(q) + smoothstep(0.55, 1.0, fq)) * T - off
        h += tw * (ht - h)
    # ribs, gullies, broken steps: ridged noise, zero on the shelf, growing with depth below the top
    o = _lod(0.55 * L, fp, 1.0, 11.0)
    rg = ridged2(x / (0.55 * L) + seed * 0.37, z / (0.55 * L) - seed * 0.11, o, seed + 2)
    rg2 = ridged2(x / (0.19 * L) - 2.1, z / (0.19 * L) + 0.7, _lod(0.19 * L, fp, 1.0, 10.0), seed + 3)
    a = det * min(m + 0.25 * shelf, 2.0 * L)
    h += a * ((rg - 0.45) + 0.35 * (rg2 - 0.45))
    if shelf > 0.0 and m < 0.8 * L:
        # the summit: broken slabs and boulders, not a table
        o3 = _lod(3.0, fp, 1.0, 7.0)
        h += 0.55 * fbm2(x * 0.45, z * 0.45, o3, seed + 4) * smoothstep(0.8 * L, 0.2 * L, m)
        o4 = _lod(1.2, fp, 1.0, 6.0)
        h += 0.35 * (ridged2(x * 0.9 + 1.7, z * 0.9 - 0.3, o4, seed + 6) - 0.45) * smoothstep(0.8 * L, 0.1 * L, m)
    return h


@njit(fastmath=True, cache=True)
def ridge(x, z, fp, CR, k, hcur):
    ax = CR[k, 0]
    az = CR[k, 1]
    bx = CR[k, 3]
    bz = CR[k, 4]
    dx = bx - ax
    dz = bz - az
    L2 = dx * dx + dz * dz
    t = ((x - ax) * dx + (z - az) * dz) / L2
    t = min(max(t, 0.0), 1.0)
    ex = x - (ax + t * dx)
    ez = z - (az + t * dz)
    d0 = math.sqrt(ex * ex + ez * ez)
    reach = CR[k, 13]
    if d0 > reach:
        return -1e5
    seed = int(CR[k, 8])
    yc = CR[k, 2] + t * (CR[k, 5] - CR[k, 2])
    slope = CR[k, 11]
    det = CR[k, 10]
    p = 1.35
    # envelope bound before paying for noise (warp <= 9 m)
    w0 = CR[k, 6] if (dx * ez - dz * ex) > 0.0 else CR[k, 7]
    db = max(d0 - 9.0, 0.0)
    ub = yc - slope * db * (db / w0) ** (p - 1.0) + 0.6 * det * min(slope * db + 6.0, 90.0) + 4.0
    if ub < hcur - CR[k, 9]:
        return -1e5
    # wandering crest: warp the query point
    ow = _lod(60.0, fp, 1.0, 4.0)
    wx = x + 9.0 * fbm2(x / 70.0 + 1.3, z / 70.0, ow, seed)
    wz = z + 9.0 * fbm2(x / 70.0 - 4.1, z / 70.0 + 2.2, ow, seed + 1)
    t = ((wx - ax) * dx + (wz - az) * dz) / L2
    t = min(max(t, 0.0), 1.0)
    ex = wx - (ax + t * dx)
    ez = wz - (az + t * dz)
    d = math.sqrt(ex * ex + ez * ez)
    w = CR[k, 6] if (dx * ez - dz * ex) > 0.0 else CR[k, 7]
    yc = CR[k, 2] + t * (CR[k, 5] - CR[k, 2])
    drop = slope * d * (d / w) ** (p - 1.0)
    h = yc - drop
    # strata ledges
    T = 5.5
    tw = 0.5 * smoothstep(0.0, 6.0, drop) * smoothstep(0.5 * T, 0.12 * T, fp)
    if tw > 0.0:
        off = 2.0 * fbm2(x / 30.0, z / 30.0, _lod(30.0, fp, 1.0, 4.0), seed + 5) + 0.04 * (x + z)
        q = (h + off) / T
        fq = q - math.floor(q)
        ht = (math.floor(q) + smoothstep(0.55, 1.0, fq)) * T - off
        h += tw * (ht - h)
    # ribs, gullies, outcrops (the ranges' ridged character at the ridge's scale)
    o = _lod(22.0, fp, 1.0, 11.0)
    rg = ridged2(x / 22.0 + seed * 0.37, z / 22.0 - seed * 0.11, o, seed + 2)
    rg2 = ridged2(x / 7.0 - 2.1, z / 7.0 + 0.7, _lod(7.0, fp, 1.0, 10.0), seed + 3)
    a = det * min(drop + 6.0, 90.0)
    h += a * ((rg - 0.45) + 0.35 * (rg2 - 0.45))
    # the crest itself: a narrow, broken snow arete (small-scale roughness)
    if d < 6.0:
        o3 = _lod(2.0, fp, 1.0, 6.0)
        h += 0.5 * fbm2(x * 0.5, z * 0.5, o3, seed + 4) * (1.0 - d / 6.0)
    return h


@njit(fastmath=True, cache=True)
def near_range(x, z, fp, CR, k, hcur):
    dx = x - CR[k, 4]
    dz = z - CR[k, 5]
    d = math.sqrt(dx * dx + dz * dz)
    if d > CR[k, 11]:
        return -1e5
    lip = CR[k, 6] - CR[k, 7] * d - CR[k, 10]          # s1's summit-lip sight line (minus margin)
    az = abs(math.atan2(dx, dz))
    wo = smoothstep(0.34, 0.56, az)                     # 0 inside s1's frustum, 1 outside
    cap = lip + 2000.0 * wo
    amp = CR[k, 3]
    # reference level relative to the flight track (nearest track segment): the camera flies ~40 m above
    # it; beyond 40 m to the right the range climbs, beyond 30 m to the left it falls to the gulf
    bd = 1e18
    ty = 0.0
    tl = 0.0
    for kk in range(CR.shape[0]):
        if CR[kk, 12] > -2.5:
            continue
        ax = CR[kk, 0]
        azz = CR[kk, 1]
        sx = CR[kk, 3] - ax
        sz = CR[kk, 4] - azz
        L2 = sx * sx + sz * sz + 1e-9
        t = min(max(((x - ax) * sx + (z - azz) * sz) / L2, 0.0), 1.0)
        ex = x - (ax + t * sx)
        ez = z - (azz + t * sz)
        dd = ex * ex + ez * ez
        if dd < bd:
            bd = dd
            ty = CR[kk, 2] + t * (CR[kk, 5] - CR[kk, 2])
            tl = math.sqrt(dd) * (1.0 if (sx * ez - sz * ex) < 0.0 else -1.0)    # + = right of the track
    if bd < 1e17 and CR[k, 14] > 0.0:
        ref = ty - CR[k, 14] + 0.9 * max(tl - 40.0, 0.0) - 0.55 * max(-tl - 30.0, 0.0)
    else:
        ref = lip - 60.0 * (1.0 - wo) + CR[k, 13] * d * wo
    # far from the track: settle toward the lip-line reference so the foothills stay a range
    ref = min(ref, cap + 30.0)
    if min(ref + 0.5 * amp, cap) < hcur - 1.0 or ref < CLOUD_Y - 250.0:
        return -1e5
    s = CR[k, 2]
    seed = int(CR[k, 8])
    o = _lod(s, fp, 1.0, 13.0)
    wx = fbm2(x / (2.0 * s) + 5.1 + CR[k, 0], z / (2.0 * s) + CR[k, 1], 3.0, seed + 40)
    wz = fbm2(x / (2.0 * s) - 2.3 + CR[k, 0], z / (2.0 * s) + 1.9 + CR[k, 1], 3.0, seed + 41)
    r = ridged2(x / s + 0.31 + CR[k, 0] + 0.55 * wx, z / s - 1.73 + CR[k, 1] + 0.55 * wz, o, seed)
    m = fbm2(x / (3.2 * s) + 0.7 + CR[k, 0], z / (3.2 * s) - 0.4 + CR[k, 1], 3.0, seed + 42)
    prom = 0.55 + 0.45 * smoothstep(-0.35, 0.35, m)
    f = prom * r ** 1.45 / 0.72
    if CR[k, 20] > 0.0:
        # the spine: an art-directed high crest along the flight line (boosts the ridged field, keeps its notches)
        sa = math.sin(CR[k, 18])
        ca = math.cos(CR[k, 18])
        along = dx * sa + dz * ca
        lat = -dx * ca + dz * sa
        ac = min(max(along, 0.0), CR[k, 21])
        ld = math.sqrt(lat * lat + (along - ac) ** 2)
        f += CR[k, 20] * math.exp(-(ld / CR[k, 19]) ** 2)
    if f > 1.0:
        f = 1.0 + 0.35 * math.tanh((f - 1.0) / 0.35)      # mild: summits stay peaked, no monsters
    h = ref - amp * (1.0 - f)
    # the pyre summit: a natural peak of the range, boosted (col 15-17 = pyre x, z, boost m; radius 55 m)
    if CR[k, 17] != 0.0:
        px = x - CR[k, 15]
        pz = z - CR[k, 16]
        h += CR[k, 17] * math.exp(-(px * px + pz * pz) / (2.0 * 16.0 * 16.0))
    # a soft ceiling under the camera: nothing within 18 m of the track rises closer than 14 m below it
    if bd < 1e17 and CR[k, 14] > 0.0 and abs(tl) < 40.0:
        ceil = ty - 14.0 + 1.2 * max(abs(tl) - 18.0, 0.0)
        if h > ceil - 30.0:
            h = ceil - 5.0 * math.log1p(math.exp(min((ceil - h) / 5.0, 50.0)))
    # the hard cap inside s1's frustum: a narrow soft min (only the rare summit touches it)
    if h > cap - 40.0:
        kk = 8.0
        h = cap - kk * math.log1p(math.exp(min((cap - h) / kk, 50.0)))
    # fade the foothills out toward their reach
    h -= 400.0 * smoothstep(0.75 * CR[k, 11], CR[k, 11], d)
    return h


@njit(fastmath=True, cache=True)
def h_rock(x, z, fp, CR):
    h = S1.h_far(x, z, fp)
    for k in range(CR.shape[0]):
        if CR[k, 12] < -2.5:
            continue
        if CR[k, 12] < -1.5:
            hc = near_range(x, z, fp, CR, k, h)
        elif CR[k, 12] < 0.0:
            hc = ridge(x, z, fp, CR, k, h)
        else:
            hc = crag(x, z, fp, CR, k, h)
        if hc > -1e4:
            h = smax(h, hc, CR[k, 9])
    return h


@njit(fastmath=True, cache=True)
def h_cloud(x, z, fp, t):
    h = S1.h_cloud(x, z, fp, t)
    # close range: finer cauliflower billows (only resolved when the camera is near; s1 never saw them)
    if fp < 1.0:
        o = _lod(40.0, fp, 0.0, 4.0)
        if o > 0.0:
            b = fbm2(x / 38.0 - t * 0.02, z / 38.0, o, 37)
            h += 7.0 * (1.0 - abs(b)) * smoothstep(1.0, 0.3, fp)
    return h


@njit(inline='always', fastmath=True)
def _billow(x, y, octaves, seed):
    """fBm of |gradient noise|: rounded tops with sharp creases between them (cloud tops)."""
    s = 0.0
    a = 0.5
    norm = 0.0
    n = int(octaves)
    frac = octaves - n
    for o in range(n + 1):
        w = a if o < n else a * frac
        if w <= 0.0:
            break
        s += w * abs(gnoise2(x, y, seed + o * 131))
        norm += a
        nx = 1.6 * x - 1.2 * y
        ny = 1.2 * x + 1.6 * y
        x = nx * 1.015 + 17.13
        y = ny * 1.015 - 9.71
        a *= 0.5
    return s / max(norm, 1e-6)


@njit(fastmath=True, cache=True)
def h_cloud_r(x, z, fp, t, relief):
    """The Run's cloud sea: s1's layers plus, when relief > 0 (metres), a field of rounded billow tops with
    creases between them (|noise| fBm from ~560 m down), so the low moon models it into lit domes and
    shadowed hollows instead of a flat grey floor. relief = 0 is exactly h_cloud (DAWN_C, HEROINE)."""
    h = h_cloud(x, z, fp, t)
    if relief > 0.0:
        o = _lod(640.0, fp, 1.0, 3.0)
        b = _billow(x / 640.0 - t * 0.004 + 3.3, z / 640.0 + 1.7, o, 38)
        # the tops swell where the big layer is already high (a sea of cloud heaps up, it does not dimple)
        m = 0.5 + 0.5 * fbm2(x / 3100.0 + 0.4, z / 3100.0 - 2.2, 2.0, 39)
        h += relief * (b - 0.30) * (0.55 + 0.9 * m)
    return h


@njit(inline='always', fastmath=True)
def h_near(x, z, fp):
    if x * x + z * z < NEAR_R * NEAR_R:
        return S1.h_near(x, z, fp)
    return -1e5


@njit(fastmath=True, cache=True)
def hfun(x, z, fp, P, CR):
    """World height incl. Earth curvature relative to the camera (P[0], P[1]); P[2] = time (s)."""
    dx = x - P[0]
    dz = z - P[1]
    curv = (dx * dx + dz * dz) / (2.0 * R_EARTH)
    h = h_near(x, z, fp)
    hf = h_rock(x, z, fp, CR)
    if hf > h:
        h = hf
    hc = h_cloud_r(x, z, fp, P[2], P[3])
    if hc > h:
        h = hc
    return h - curv


# ------------------------------------------------------------------ marcher ---

@njit(parallel=True, fastmath=True, cache=True)
def march(P, CR, C, d0, dmax, rel, kgap, hmax, nbis, out_d):
    """Column-coherent heightfield march (see mt/terrain.py) with the run's height function."""
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


@njit(inline='always', fastmath=True)
def soft_shadow(P, CR, x, y, z, lx, ly, lz, t0, tmax, nsteps, k, fp):
    res = 1.0
    t = t0
    grow = (tmax / t0) ** (1.0 / nsteps)
    for _s in range(nsteps):
        h = hfun(x + lx * t, z + lz * t, fp + t * 0.004, P, CR)
        dh = y + ly * t - h
        r = k * dh / t
        if r < res:
            res = r
        if res < 0.0:
            return 0.0
        t *= grow
    if res > 1.0:
        res = 1.0
    return res * res * (3.0 - 2.0 * res)


# ------------------------------------------------------------------- shading ---
# Q (shading knobs):
#  0 key intensity | 1 cloud fwd-scatter gain | 2 cloud fwd-scatter power | 3 snow lo | 4 snow hi
#  5 cloud shadow floor | 6 alpenglow (0 night) | 7 sheen | 8 cloud trough dark | 9 cloud albedo
# 10 rock albedo scale | 11 key-light shadow k (far) | 12 fog in-scatter toward key: gain
# 13 cloud ambient gain | 14 sun-disc (0 = moon) | 15 cloud self-shadow strength | 16 bounce fill


@njit(parallel=True, fastmath=True, cache=True)
def shade(C, D, P, CR, S, LT, Lk, Q, amb, fogp, out, zbuf, dist_out, PL):
    H, W = D.shape
    mx, my, mz = Lk[0], Lk[1], Lk[2]
    Ik = Q[0]
    pix_ang = 1.0 / C[7]
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
            if d >= 1e29 and Q[17] > 0.0 and dy < 0.0:
                # below the horizon but nothing hit within dmax: the ray ends in the haze over the far cloud
                # sea (not in "sky below the horizon", which drew an inverted range along the horizon)
                cosv = dx * mx + dy * my + dz * mz
                ph = 1.0 + fogp[4] * max(cosv, 0.0) ** 4
                out[j, i, 0] = fogp[5] * ph
                out[j, i, 1] = fogp[6] * ph
                out[j, i, 2] = fogp[7] * ph
                zbuf[j, i] = 1e9
                dist_out[j, i] = 1e9
                continue
            if d >= 1e29:
                r, g, b = SK.sky_base(dx, dy, dz, S)
                mr, mg, mb = SK.moon_disk(dx, dy, dz, S, pix_ang)
                out[j, i, 0] = r + mr
                out[j, i, 1] = g + mg
                out[j, i, 2] = b + mb
                zbuf[j, i] = 1e9
                dist_out[j, i] = 1e9
                continue
            x = C[0] + dxh * d
            z = C[2] + dzh * d
            dist = d * math.sqrt(1.0 + sl * sl)
            fp = dist / C[7]
            e = max(fp * 0.8, 0.003)
            ddx = x - P[0]
            ddz = z - P[1]
            curv = (ddx * ddx + ddz * ddz) / (2.0 * R_EARTH)
            hn = h_near(x, z, fp)
            hf = h_rock(x, z, fp, CR)
            hc = h_cloud_r(x, z, fp, P[2], P[3])
            surf = 0
            if hf > hn:
                surf = 1
            if hc > max(hn, hf):
                surf = 2
            if surf == 0:
                h0 = hn
                hx = h_near(x + e, z, fp)
                hz = h_near(x, z + e, fp)
            elif surf == 1:
                h0 = hf
                hx = h_rock(x + e, z, fp, CR)
                hz = h_rock(x, z + e, fp, CR)
            else:
                h0 = hc
                if Q[17] > 0.0:
                    # cloud is translucent: light diffuses through the small lumps, so shade it by the
                    # normal of its larger forms (a wide difference) -- soft heaps, not a rough rug
                    e = max(e, 22.0)
                hx = h_cloud_r(x + e, z, fp, P[2], P[3])
                hz = h_cloud_r(x, z + e, fp, P[2], P[3])
            nx = -(hx - h0)
            ny = e
            nz = -(hz - h0)
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= inv
            ny *= inv
            nz *= inv
            ndl = nx * mx + ny * my + nz * mz
            yw = h0 - curv
            if surf == 2:
                # cloud sea: wrap light + forward scatter toward the key light, peak shadows, troughs
                cosk = dx * mx + dy * my + dz * mz
                fwd = 1.0 + Q[1] * max(cosk, 0.0) ** Q[2]
                wk = 0.8 if Q[17] <= 0.0 else Q[21]
                wrap = max((ndl + wk) / (1.0 + wk), 0.0)
                shc = soft_shadow(P, CR, x, yw + 5.0, z, mx, my, mz, 40.0, 14000.0, 13, 6.0, fp)
                shc = Q[5] + (1.0 - Q[5]) * shc
                ca = Q[9]
                cr = ca * (Ik * Lk[3] * wrap * fwd * shc + amb[0] * Q[13])
                cg = ca * (Ik * Lk[4] * wrap * fwd * shc + amb[1] * Q[13])
                cb = ca * (Ik * Lk[5] * wrap * fwd * shc + amb[2] * Q[13])
                trough = smoothstep(CLOUD_Y - 120.0, CLOUD_Y + 110.0, h0)
                tk = (1.0 - Q[8]) + Q[8] * trough
                cr *= tk
                cg *= tk
                cb *= tk
                if Q[17] > 0.0:
                    # beacons low over the cloud sea light its tops (soft: the light enters the cloud)
                    for li in range(LT.shape[0]):
                        lx = LT[li, 0] - x
                        ly = LT[li, 1] - yw
                        lz = LT[li, 2] - z
                        l2 = lx * lx + ly * ly + lz * lz
                        if l2 > LT[li, 6] * 4.0e4:
                            continue
                        ll = math.sqrt(l2) + 1e-9
                        ndf = max((nx * lx + ny * ly + nz * lz) / ll + 0.6, 0.0) / 1.6
                        E = LT[li, 6] * ndf / (l2 + LT[li, 7] * LT[li, 7]) * ca
                        cr += E * LT[li, 3]
                        cg += E * LT[li, 4]
                        cb += E * LT[li, 5]
            else:
                # snow / rock. Far: exactly s1's rule (45 m-smoothed slope + 380/95 m noise). Near: ledges
                # hold snow and ribs shed it (fine normal + 3 m slope + the far tendency).
                es = max(fp * 6.0, 45.0)
                if surf == 0:
                    hsx = h_near(x + es, z, fp * 4.0)
                    hsz = h_near(x, z + es, fp * 4.0)
                else:
                    hsx = h_rock(x + es, z, fp * 4.0, CR)
                    hsz = h_rock(x, z + es, fp * 4.0, CR)
                nsx = -(hsx - h0) / es
                nsz = -(hsz - h0) / es
                nsy = 1.0 / math.sqrt(nsx * nsx + nsz * nsz + 1.0)
                if surf == 0:
                    sn = gnoise2(x * 0.8, z * 0.8, 71) * 0.12 + gnoise2(x * 3.1, z * 3.1, 72) * 0.05
                    snow = smoothstep(0.45, 0.70, ny + sn)
                elif Q[17] > 0.0:
                    # the Run's rule: the mountain's FORM decides. Slope from a smooth central difference at
                    # the snow scale (the fine relief no longer speckles it), convexity (ridges, shoulders and
                    # ledges hold snow, gullies and steep faces go bare), and only a little noise, in domains
                    # sheared by height so nothing streaks down a steep face.
                    hsx2 = h_rock(x - es, z, fp * 4.0, CR)
                    hsz2 = h_rock(x, z - es, fp * 4.0, CR)
                    hsc = h_rock(x, z, fp * 4.0, CR)
                    gx = (hsx - hsx2) / (2.0 * es)
                    gz = (hsz - hsz2) / (2.0 * es)
                    nsy = 1.0 / math.sqrt(gx * gx + gz * gz + 1.0)
                    conv = -(hsx + hsx2 + hsz + hsz2 - 4.0 * hsc) / (2.0 * es)
                    conv = min(max(conv, -1.0), 1.0)
                    xs = x + 0.61 * h0
                    zs = z - 0.37 * h0
                    sn = (gnoise2(xs / 380.0, zs / 380.0, 73) * 0.18 + gnoise2(xs / 95.0, zs / 95.0, 74) * 0.06) * Q[22]
                    snow = smoothstep(0.38 - Q[23], 0.58 - Q[23], nsy + Q[20] * conv + sn)
                    # nothing holds on a face steeper than ~65 deg, however convex (no bright snow curtains
                    # down the walls of the towers)
                    snow *= smoothstep(0.30, 0.46, nsy)
                    wf = smoothstep(3.0, 0.4, fp)
                    if wf > 0.0:
                        em = max(fp * 6.0, 3.0)
                        hmx = h_rock(x + em, z, fp * 2.0, CR)
                        hmz = h_rock(x, z + em, fp * 2.0, CR)
                        hmx2 = h_rock(x - em, z, fp * 2.0, CR)
                        hmz2 = h_rock(x, z - em, fp * 2.0, CR)
                        gmx = (hmx - hmx2) / (2.0 * em)
                        gmz = (hmz - hmz2) / (2.0 * em)
                        nmy = 1.0 / math.sqrt(gmx * gmx + gmz * gmz + 1.0)
                        cm = -(hmx + hmx2 + hmz + hmz2 - 4.0 * h0) / (2.0 * em)
                        cm = min(max(cm, -1.0), 1.0)
                        sf = gnoise2(xs / 9.0, zs / 9.0, 75) * 0.05 + gnoise2(xs * 0.9, zs * 0.9, 76) * 0.02
                        sv = 0.16 * ny + 0.38 * nmy + 0.46 * nsy + 0.5 * Q[20] * (conv + cm) + sn * 0.5 + sf
                        sneat = smoothstep(Q[3] - Q[23], Q[4] - Q[23], sv)
                        # the steepness cap on a ~12 m slope (the form of a wall, not its ribs: a rib-scale
                        # cap streaks the faces)
                        e12 = max(fp * 6.0, 12.0)
                        g1 = (h_rock(x + e12, z, fp * 3.0, CR) - h_rock(x - e12, z, fp * 3.0, CR)) / (2.0 * e12)
                        g2 = (h_rock(x, z + e12, fp * 3.0, CR) - h_rock(x, z - e12, fp * 3.0, CR)) / (2.0 * e12)
                        n12 = 1.0 / math.sqrt(g1 * g1 + g2 * g2 + 1.0)
                        sneat *= smoothstep(0.30, 0.46, min(n12, nsy))
                        snow = snow + (sneat - snow) * wf
                else:
                    sn = gnoise2(x / 380.0, z / 380.0, 73) * 0.18 + gnoise2(x / 95.0, z / 95.0, 74) * 0.06
                    snow = smoothstep(0.38, 0.58, nsy + sn)
                    wf = smoothstep(3.0, 0.4, fp)
                    if wf > 0.0:
                        em = max(fp * 6.0, 3.0)
                        hmx = h_rock(x + em, z, fp * 2.0, CR)
                        hmz = h_rock(x, z + em, fp * 2.0, CR)
                        nmx = -(hmx - h0) / em
                        nmz = -(hmz - h0) / em
                        nmy = 1.0 / math.sqrt(nmx * nmx + nmz * nmz + 1.0)
                        sf = gnoise2(x / 9.0, z / 9.0, 75) * 0.06 + gnoise2(x * 0.9, z * 0.9, 76) * 0.035
                        sv = 0.42 * ny + 0.33 * nmy + 0.25 * nsy + sn * 0.6 + sf
                        sneat = smoothstep(Q[3], Q[4], sv)
                        snow = snow + (sneat - snow) * wf
                # rock albedo: slabs and strata, a little lighter on the ribs
                rv = 1.0
                if fp < 20.0:
                    rv = 0.75 + 0.5 * (0.5 + 0.5 * gnoise2(x / 7.0 + h0 / 3.0, z / 7.0, 81))
                    rv *= 0.85 + 0.3 * (0.5 + 0.5 * gnoise2(h0 * 0.35, x / 40.0 + z / 40.0, 82))
                    rv = 1.0 + (rv - 1.0) * smoothstep(20.0, 4.0, fp)
                    if fp < 0.05:
                        rv *= 0.8 + 0.4 * (0.5 + 0.5 * gnoise2(x * 1.1, z * 1.1 + h0 * 0.8, 83)) * smoothstep(0.05, 0.01, fp) \
                            + 0.4 * (1.0 - smoothstep(0.05, 0.01, fp)) * 0.5
                # beacon platforms: cleared, dark rock. A negative radius marks a bare tower instead: its walls
                # (from 6 m above the row's y to 140 m below) are too steep to hold snow.
                for pi in range(PL.shape[0]):
                    qx = x - PL[pi, 0]
                    qz = z - PL[pi, 2]
                    qd = math.sqrt(qx * qx + qz * qz)
                    if PL[pi, 3] < 0.0:
                        rr = -PL[pi, 3]
                        if qd < rr and yw < PL[pi, 1] + 6.0 and yw > PL[pi, 1] - 140.0:
                            snow *= 0.15 + 0.85 * smoothstep(0.6 * rr, rr, qd)
                        continue
                    if qd < PL[pi, 3] * 1.6 and abs(yw - PL[pi, 1]) < 6.0:
                        snow *= smoothstep(PL[pi, 3] * 0.7, PL[pi, 3] * 1.6,
                                           qd + 0.4 * PL[pi, 3] * gnoise2(x * 1.3, z * 1.3, 91))
                ra = Q[10] * rv
                ar = 0.055 * ra + (0.80 - 0.055 * ra) * snow
                ag = 0.056 * ra + (0.86 - 0.056 * ra) * snow
                ab = 0.062 * ra + (0.98 - 0.062 * ra) * snow
                shd = 1.0
                if ndl > 0.0:
                    t0 = max(fp * 1.5, 0.08)
                    nst = 20 if dist < 2500.0 else 14
                    shd = soft_shadow(P, CR, x, yw + max(fp * 1.2, 0.03), z, mx, my, mz, t0, 12000.0, nst,
                                      Q[11], fp)
                dif = max(ndl, 0.0) * shd
                skyl = 0.55 + 0.45 * ny
                # bounce from the key-lit cloud sea / snowfields below (faces turned sideways or down)
                bnc = Q[16] * Ik * (0.5 - 0.5 * ny) * (0.6 + 0.4 * max(-(nx * mx + nz * mz), 0.0))
                cr = ar * (Ik * Lk[3] * dif + amb[0] * skyl + Lk[3] * bnc)
                cg = ag * (Ik * Lk[4] * dif + amb[1] * skyl + Lk[4] * bnc)
                cb = ab * (Ik * Lk[5] * dif + amb[2] * skyl + Lk[5] * bnc)
                if snow > 0.0:
                    hx_ = mx - dx
                    hy_ = my - dy
                    hz_ = mz - dz
                    hl_ = math.sqrt(hx_ * hx_ + hy_ * hy_ + hz_ * hz_) + 1e-9
                    nh = max((nx * hx_ + ny * hy_ + nz * hz_) / hl_, 0.0)
                    sp = snow * Q[7] * nh ** 18 * shd
                    cr += sp * Ik * Lk[3]
                    cg += sp * Ik * Lk[4]
                    cb += sp * Ik * Lk[5]
                # fire light (point lights, inverse square)
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
                        cr += ar * E * LT[li, 3]
                        cg += ag * E * LT[li, 4]
                        cb += ab * E * LT[li, 5]
                if Q[17] > 0.0:
                    # the Run: where a mountain meets the sea of cloud, moonlit wisps wrap its foot (the cloud
                    # tops' own light, torn into streaks that lift with height) instead of a hard waterline
                    dyc = h0 - hc
                    if dyc < 65.0:
                        # layered like real wisps: the noise runs mostly with height (horizontal streaks)
                        wn = fbm2(x / 110.0 + h0 / 14.0 - P[2] * 0.05, z / 110.0 - h0 / 20.0, 3.0, 93)
                        mm = (1.0 - smoothstep(0.0, 58.0, dyc)) ** 1.5 * (0.45 + 0.55 * smoothstep(-0.4, 0.4, wn)) * 0.8
                        if mm > 0.0:
                            cosk = dx * mx + dy * my + dz * mz
                            fwd = 1.0 + Q[1] * max(cosk, 0.0) ** Q[2]
                            wk = Q[21]
                            wr = (my + wk) / (1.0 + wk)
                            ca = Q[9]
                            mr = ca * (Ik * Lk[3] * wr * fwd * 0.8 + amb[0] * Q[13])
                            mg = ca * (Ik * Lk[4] * wr * fwd * 0.8 + amb[1] * Q[13])
                            mb = ca * (Ik * Lk[5] * wr * fwd * 0.8 + amb[2] * Q[13])
                            cr += (mr - cr) * mm
                            cg += (mg - cg) * mm
                            cb += (mb - cb) * mm
            # atmosphere: aerial perspective + low mist over the cloud sea
            yc = C[1]
            tau = height_fog_tau(dist, yc, yw, fogp[0], fogp[1])
            tau += height_fog_tau(dist, yc - CLOUD_Y, yw - CLOUD_Y, fogp[2], fogp[3])
            if Q[18] > 0.0:
                # the Run: a deeper haze layer over the cloud sea, so each range stands paler than the last
                tau += height_fog_tau(dist, yc - CLOUD_Y, yw - CLOUD_Y, Q[18], Q[19])
            tr = math.exp(-tau)
            cosv = dx * mx + dy * my + dz * mz
            ph = 1.0 + fogp[4] * max(cosv, 0.0) ** 4
            out[j, i, 0] = cr * tr + fogp[5] * ph * (1.0 - tr)
            out[j, i, 1] = cg * tr + fogp[6] * ph * (1.0 - tr)
            out[j, i, 2] = cb * tr + fogp[7] * ph * (1.0 - tr)
            zbuf[j, i] = d * (dxh * C[3] + dzh * C[4])
            dist_out[j, i] = dist


@njit(inline='always', fastmath=True)
def height_fog_tau(dist, y0, y1, a, b):
    dy = y1 - y0
    e0 = math.exp(-b * y0)
    if abs(dy) < 1e-3:
        return a * dist * e0
    e1 = math.exp(-b * y1)
    return a * dist * (e0 - e1) / (b * dy)


# ---------------------------------------------------------------- utilities ---

@njit(parallel=True, fastmath=True, cache=True)
def height_grid(x0, z0, dx, nx, nz, CR, t, out):
    P = np.array([x0, z0, t, 0.0])
    for j in prange(nz):
        for i in range(nx):
            x = x0 + i * dx
            z = z0 + j * dx
            h = h_near(x, z, dx)
            hf = h_rock(x, z, dx, CR)
            if hf > h:
                h = hf
            out[j, i] = h


def ground(x, z, CR, fp=0.05):
    """Terrain height (no cloud, no curvature) at a world point."""
    return max(h_near(float(x), float(z), fp), h_rock(float(x), float(z), fp, CR))


def summit_near(x, z, CR, rad=60.0, n=25):
    """Local maximum of the terrain near (x, z): returns (x, y, z)."""
    best = (-1e9, x, z)
    for _it in range(3):
        xs = np.linspace(best[1] - rad, best[1] + rad, n)
        zs = np.linspace(best[2] - rad, best[2] + rad, n)
        for xx in xs:
            for zz in zs:
                h = ground(xx, zz, CR)
                if h > best[0]:
                    best = (h, xx, zz)
        rad *= 0.25
    return np.array([best[1], best[0], best[2]])


def night_light():
    """s1's moon, ambient, sky, fog (identical values)."""
    moon_col = CM.lin('#9DB4D9')
    Lk = np.r_[MOON_DIR, moon_col]
    amb = CM.lin('#27335E') * 0.35
    S = SK.sky_params(zenith='#070B1C', horizon='#2A3866', moon_dir=MOON_DIR, moon_radius_deg=0.8,
                      halo_I=0.025, halo_w=0.22, halo2_I=0.012, halo2_w=0.7, horizon_glow=0.25, gain=1.0)
    fogc = CM.lin('#2E3D66') * 0.95
    fogp = np.array([5.0e-5, 1 / 1500.0, 2.2e-4, 1 / 140.0, 1.5, fogc[0], fogc[1], fogc[2]])
    Q = np.zeros(24)
    Q[0] = 0.55        # moon intensity (s1 Im)
    Q[1] = 1.6
    Q[2] = 4.0
    Q[3] = 0.40
    Q[4] = 0.60
    Q[5] = 0.35
    Q[7] = 0.25
    Q[8] = 0.55
    Q[9] = 0.9
    Q[10] = 1.0
    Q[11] = 12.0
    Q[13] = 1.5
    Q[16] = 0.10
    return Lk, amb, S, fogp, Q


CLOUD_RELIEF = 60.0          # the Run: amplitude (m) of the billowed cloud tops (P[3] of the marcher / shader)


def run_light():
    """night_light() with the Run's look switched on (Q[17..23]); s1's world is night_light() unchanged.
    For shots that must continue the Run (e.g. the FIRST BEACON reveal plate), use this + CLOUD_RELIEF."""
    Lk, amb, S, fogp, Q = night_light()
    Q = Q.copy()
    Q[17] = 1.0              # misses below the horizon end in haze; firelight on cloud tops; the new snow rule
    Q[18] = 7.0e-5           # haze layer over the cloud sea: density at the cloud top (1/m) ...
    Q[19] = 1.0 / 450.0      # ... and 1 / its scale height (m)
    Q[20] = 0.35             # snow: convexity gain (ridges and ledges hold snow, gullies go bare)
    Q[21] = 0.35             # cloud: wrap (s1: 0.8) -- the low moon models the billows
    Q[22] = 1.0 / 3.0        # snow: noise scale vs s1's (the critic: cut the noise terms to a third)
    Q[23] = 0.06             # snow: threshold shift (a little more snow on steep ground)
    return Lk, amb, S, fogp, Q

