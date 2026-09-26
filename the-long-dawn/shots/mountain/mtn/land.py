"""The mountain world's height function (analytic, numba) and its rasterisation.

World units: metres. x = east, y = north, z = up. The cloud-sea top sits near z = 0.

height(x, y, fp, P, SP) : fp = footprint (metres) of the sample; octaves finer than ~fp are
dropped so a coarse LOD is an alias-free, smoother version of the same land.
P  = designed summits table (see world.peaks_table)
SP = the main range's spine polyline (world.SPINE)

Structure:
  base    : warped, slope-damped fBm (sharp crests, smooth valleys) + ridged aretes on the high
            ground, scaled by a massif envelope that is raised along the main range
  designed: explicit summits placed for the story, blended in with a smooth max
  gullies : multi-octave slope-aligned erosion noise (stripes that run downhill and branch)
"""
import math

import numpy as np
from numba import njit, prange

from .noise import gnoise2, fbm2, ridged2, smoothstep, hash2, dfbm

R_EARTH = 6.371e6
BASE = -1416.0
AMP = 3700.0
LS = 0.5          # land scale: the natural relief is shrunk uniformly (x, y, z) by this


@njit(inline='always', cache=True)
def _lod(scale, fp):
    """Number of octaves for a feature of `scale` metres at footprint fp."""
    return min(max(math.log2(max(scale / max(fp * 2.0, 1e-3), 1.0)), 1.0), 14.0)


@njit(cache=True)
def spine_dist(x, y, SP):
    best = 1e18
    for k in range(SP.shape[0] - 1):
        ax = SP[k, 0]
        ay = SP[k, 1]
        bx = SP[k + 1, 0]
        by = SP[k + 1, 1]
        vx = bx - ax
        vy = by - ay
        t = ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy)
        t = min(max(t, 0.0), 1.0)
        dx = x - ax - t * vx
        dy = y - ay - t * vy
        d = dx * dx + dy * dy
        if d < best:
            best = d
    return math.sqrt(best)


@njit(cache=True)
def _erosion_cell(px, py, dx, dy):
    """Clayjohn/Fewes erosion kernel: returns (value, dvalue/dpx, dvalue/dpy) of stripes along
    direction perpendicular to (dx, dy)."""
    ix = math.floor(px)
    iy = math.floor(py)
    fx = px - ix
    fy = py - iy
    va = 0.0
    vx = 0.0
    vy = 0.0
    wt = 0.0
    TWO_PI = 2.0 * math.pi
    for i in range(-2, 2):
        for j in range(-2, 2):
            hh = hash2(int(ix) - i, int(iy) - j, 911)
            hx = ((hh & 0xFFFF) / 65535.0) * 0.5
            hy = ((hh >> 16) / 65535.0) * 0.5
            ppx = fx + i - hx
            ppy = fy + j - hy
            d = ppx * ppx + ppy * ppy
            w = math.exp(-d * 2.0)
            wt += w
            mag = ppx * dx + ppy * dy
            va += math.cos(mag * TWO_PI) * w
            s = -math.sin(mag * TWO_PI) * w
            vx += s * dx
            vy += s * dy
    return va / wt, vx / wt, vy / wt


@njit(cache=True)
def horns(x, y, fp, LP):
    """Sum of warped polygonal pyramids (3-4 faces) scattered on a jittered grid: horn peaks with
    aretes where faces meet and cols where neighbours overlap. Scaled-domain units."""
    C = LP[5]
    A = LP[6]
    pw = LP[7]
    ix = math.floor(x / C)
    iy = math.floor(y / C)
    wx = 0.16 * C * fbm2(x / (0.6 * C) + 9.1, y / (0.6 * C), 3.0, 401)
    wy = 0.16 * C * fbm2(x / (0.6 * C) - 2.7, y / (0.6 * C) + 5.3, 3.0, 402)
    h = 0.0
    for i in range(-1, 2):
        for j in range(-1, 2):
            cx = int(ix) + i
            cy = int(iy) + j
            h1 = hash2(cx, cy, 777)
            h2 = hash2(cx, cy, 778)
            u1 = (h1 & 0xFFFF) / 65535.0
            u2 = (h1 >> 16) / 65535.0
            u3 = (h2 & 0xFFFF) / 65535.0
            u4 = ((h2 >> 16) & 0xFF) / 255.0
            u5 = ((h2 >> 24) & 0xFF) / 255.0
            hr = u3 ** LP[8]
            if hr < 0.03:
                continue
            px = (cx + 0.15 + 0.7 * u1) * C
            py = (cy + 0.15 + 0.7 * u2) * C
            R = C * (0.7 + 0.5 * u4) * (0.75 + 0.5 * hr)
            dx = x + wx - px
            dy = y + wy - py
            if dx * dx + dy * dy > R * R * 2.2:
                continue
            nf = 3 if (h1 & 3) != 0 else 4
            th0 = u5 * 2.0 * math.pi
            dp = -1e18
            for k in range(nf):
                a = th0 + 2.0 * math.pi * k / nf
                v = dx * math.cos(a) + dy * math.sin(a)
                if v > dp:
                    dp = v
            dp /= math.cos(math.pi / nf)
            dc = math.sqrt(dx * dx + dy * dy)
            d = 0.88 * dp + 0.12 * dc
            q = d / R
            if q < 1.0:
                hp = A * hr * (1.0 - q) ** pw
                # smooth max: distinct summits with sharp cols between neighbours
                kk = 0.07 * A
                if hp > h - kk:
                    if hp > h + kk:
                        h = hp
                    else:
                        t = (hp - h + kk) / (2.0 * kk)
                        h = h + (hp - h) * t * t * (3.0 - 2.0 * t) + kk * 0.25 * t * (1.0 - t)
    return h


@njit(cache=True)
def base_height(x, y, fp, P, SP, LP):
    """Base terrain (no gullies). LP = land parameter vector (see world.LAND_PARAMS)."""
    X0 = x
    Y0 = y
    x = x / LS
    y = y / LS
    fp = fp / LS
    # domain warp (large)
    wx = fbm2(x / 9000.0 + 3.3, y / 9000.0 - 1.1, 3.0, 101)
    wy = fbm2(x / 9000.0 - 7.1, y / 9000.0 + 2.9, 3.0, 102)
    xw = x + 2200.0 * wx
    yw = y + 2200.0 * wy
    # massif envelope: where ranges are high; the main range along the spine
    m = fbm2(x / 20000.0 + 0.3, y / 20000.0 - 0.8, 3.0, 103)
    env = 0.7 + 0.3 * smoothstep(-0.5, 0.5, m)
    ds = spine_dist(LS * (x + 900.0 * wx), LS * (y + 900.0 * wy), SP) / LS
    env += LP[9] * math.exp(-(ds / 2500.0) ** 2)
    # massive eroded body (slope-damped fBm, raised to a power so summits stand apart)
    o2 = _lod(6500.0, fp)
    t = 0.5 + 0.5 * dfbm(xw / 6500.0 + 3.1, yw / 6500.0 - 7.7, o2, 105, 2.0)
    t = min(max(t, 0.0), 1.0) ** LP[1]
    # aretes on the high ground
    o = _lod(1400.0, fp)
    r = ridged2(xw / 1400.0, yw / 1400.0, o, 104, 2.05, 0.5, 1.8)
    hi = smoothstep(0.35, 0.70, t)
    hm = smoothstep(LP[10], LP[10] + 0.4, t)
    h = LP[0] + env * (LP[2] * t + LP[3] * (r - 0.45) * hi)
    h += env * (LP[11] + (1.0 - LP[11]) * hm) * horns(xw, yw, fp, LP)
    h *= LS
    x = X0
    y = Y0
    # designed summits (kind 0: horn x, y, H, R) and broad bumps/depressions (kind 2)
    n = P.shape[0]
    for k in range(n):
        kind = P[k, 4]
        if kind != 0.0 and kind != 2.0:
            continue
        ddx = x - P[k, 0]
        ddy = y - P[k, 1]
        R = P[k, 3]
        if ddx * ddx + ddy * ddy > 9.0 * R * R:
            continue
        if kind == 2.0:
            h += P[k, 2] * math.exp(-(ddx * ddx + ddy * ddy) / (R * R))
            continue
        H = P[k, 2]
        wq = 0.30 * R * fbm2(x / (0.9 * R) + 0.37 * k, y / (0.9 * R), 3.0, 500 + k)
        wr = 0.30 * R * fbm2(x / (0.9 * R) - 1.9, y / (0.9 * R) + 0.29 * k, 3.0, 540 + k)
        dx = ddx + wq
        dy = ddy + wr
        nf = 3 + (k % 3)
        th0 = 1.37 * k + 0.5
        dp = -1e18
        for f in range(nf):
            a = th0 + 2.0 * math.pi * f / nf
            v = dx * math.cos(a) + dy * math.sin(a)
            if v > dp:
                dp = v
        dp /= math.cos(math.pi / nf)
        dc = math.sqrt(dx * dx + dy * dy)
        q = (0.86 * dp + 0.14 * dc) / R
        if q < 1.0:
            rr = ridged2(dx / (0.38 * R) + 3.3 * k, dy / (0.38 * R) - 1.1 * k,
                         _lod(0.38 * R, fp), 560 + k, 2.0, 0.5, 1.8)
            hp = H * (1.0 - q) ** 1.35 * (0.84 + 0.2 * rr) - 0.06 * H * q * (1.0 - rr)
            kk = 0.05 * H + 10.0
            if hp > h - kk:
                if hp > h + kk:
                    h = hp
                else:
                    t = (hp - h + kk) / (2.0 * kk)
                    h = h + (hp - h) * t * t * (3.0 - 2.0 * t)
    return h


@njit(cache=True)
def pads(x, y, h, P):
    """Flatten small pads (P rows with kind 1: x, y, target height, radius) for the cairns, and
    scatter snow-capped boulders around them (so a summit's silhouette reads as rock)."""
    n = P.shape[0]
    for k in range(n):
        if P[k, 4] != 1.0:
            continue
        ddx = x - P[k, 0]
        ddy = y - P[k, 1]
        d = math.sqrt(ddx * ddx + ddy * ddy)
        R = P[k, 3]
        if d > 40.0:
            continue
        if P[k, 5] > 0.0:
            # truncate the summit: an irregular little plateau that only ever cuts the land
            cap = P[k, 2] - 0.035 * max(d - R, 0.0) ** 2 + 0.25 * fbm2(x * 0.3, y * 0.3, 2.0, 745)
            if h > cap:
                h = cap
        elif d < 2.5 * R:
            w = smoothstep(2.5 * R, 0.9 * R, d)
            dome = P[k, 2] - 0.06 * R * (d / R) ** 2
            h = h * (1.0 - w) + dome * w
        # boulders (on the summit and its rim)
        rin = 4.0 if P[k, 5] > 0.0 else 7.5
        rspan = 12.0 if P[k, 5] > 0.0 else 20.0
        for b in range(16):
            hb = hash2(b, k, 733)
            a = ((hb & 0xFFFF) / 65535.0) * 2.0 * math.pi
            rr = rin + rspan * ((hb >> 16) / 65535.0)
            h2 = hash2(b, k, 734)
            br = 1.1 + 2.3 * ((h2 & 0xFFFF) / 65535.0)
            bh = (0.6 + 0.9 * ((h2 >> 16) / 65535.0)) * br
            bx = P[k, 0] + rr * math.cos(a)
            by = P[k, 1] + rr * math.sin(a)
            q = ((x - bx) ** 2 + (y - by) ** 2) / (br * br)
            if q < 1.0:
                # lumpy: a little noise on the boulder
                top = h + bh * math.sqrt(1.0 - q) * (0.85 + 0.3 * fbm2(x * 0.9, y * 0.9, 2.0, 740 + b))
                if top > h:
                    h = top
    return h


@njit(cache=True)
def seg_dist(x, y, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    t = ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy)
    t = min(max(t, 0.0), 1.0)
    dx = x - ax - t * vx
    dy = y - ay - t * vy
    return math.sqrt(dx * dx + dy * dy), t


@njit(cache=True)
def story_shape(x, y, Z):
    """Designed low-frequency land of the story zone: polygonal horn peaks, smooth-maxed.
    Z rows: x, y, H, R, n_faces, orientation (rad), profile power, stretch (along orientation)
    Returns (height, q) with q = the winning horn's normalised polygonal distance."""
    h = -300.0
    qw = 1.0
    for k in range(Z.shape[0]):
        dx0 = x - Z[k, 0]
        dy0 = y - Z[k, 1]
        R = Z[k, 3]
        if dx0 * dx0 + dy0 * dy0 > 9.0 * R * R:
            continue
        # anisotropy: compress the distance along the orientation axis
        ca = math.cos(Z[k, 5])
        sa = math.sin(Z[k, 5])
        u = (dx0 * ca + dy0 * sa) * Z[k, 7]
        v = -dx0 * sa + dy0 * ca
        dx = u * ca - v * sa
        dy = u * sa + v * ca
        nf = int(Z[k, 4])
        dp = -1e18
        for f in range(nf):
            a = Z[k, 5] + 2.0 * math.pi * f / nf
            v = dx * math.cos(a) + dy * math.sin(a)
            if v > dp:
                dp = v
        dp /= math.cos(math.pi / nf)
        dc = math.sqrt(dx * dx + dy * dy)
        q = (0.8 * dp + 0.2 * dc) / R
        hp = -160.0 + (Z[k, 2] + 160.0) * max(1.0 - q, 0.0) ** Z[k, 6]
        kk = 14.0
        if hp > h - kk:
            if hp > h + kk:
                h = hp
                qw = q
            else:
                t2 = (hp - h + kk) / (2.0 * kk)
                h = h + (hp - h) * t2 * t2 * (3.0 - 2.0 * t2) + kk * 0.2 * t2 * (1.0 - t2)
                if t2 > 0.5:
                    qw = q
    return h, qw


@njit(cache=True)
def gullies(x, y, fp, gx, gy, amp):
    """Slope-aligned erosion stripes for a surface with gradient (gx, gy); returns displacement."""
    slope = math.sqrt(gx * gx + gy * gy)
    L0 = 480.0
    f = 1.0 / L0
    a = 1.0
    eh = 0.0
    ex = 0.0
    ey = 0.0
    for o in range(7):
        wl = 1.0 / f
        if wl < 2.5 * fp:
            break
        dgx = gx + ex
        dgy = gy + ey
        dl = math.sqrt(dgx * dgx + dgy * dgy) + 1e-6
        dirx = dgy / dl
        diry = -dgx / dl
        v, vx, vy = _erosion_cell(x * f, y * f, dirx, diry)
        wfade = smoothstep(2.5 * fp, 5.0 * fp, wl)
        eh += a * v * wfade
        ex += a * vx * f * wfade * 2.0 * math.pi * amp
        ey += a * vy * f * wfade * 2.0 * math.pi * amp
        a *= 0.5
        f *= 2.0
    st = smoothstep(0.12, 0.8, slope)
    return amp * eh * st


@njit(cache=True)
def zone_height(x, y, fp, Z, P, SP, LP):
    """Designed envelope + the natural land's own high-frequency relief + face undulation."""
    wx = 45.0 * fbm2(x / 700.0 + 1.3, y / 700.0, 3.0, 201)
    wy = 45.0 * fbm2(x / 700.0 - 4.1, y / 700.0 + 2.2, 3.0, 202)
    e = max(fp, 2.0)
    h0, q = story_shape(x + wx, y + wy, Z)
    hx, _ = story_shape(x + e + wx, y + wy, Z)
    hy, _ = story_shape(x + wx, y + e + wy, Z)
    gx = (hx - h0) / e
    gy = (hy - h0) / e
    sl = math.sqrt(gx * gx + gy * gy)
    steep = smoothstep(0.3, 1.1, sl)
    crest = 0.45 + 0.55 * smoothstep(0.02, 0.18, q)   # relief is gentler on a summit
    # the natural land's fine relief (gullies, aretes, ribs): natural minus its low-pass
    hn = height0(x, y, fp, P, SP, LP)
    hl = height0(x, y, max(fp, 260.0), P, SP, LP)
    det = hn - hl
    det = 45.0 * math.tanh(det / 45.0)
    h = h0 + det * (0.45 + 0.55 * crest)
    # face undulation: buttresses and bays so no face is planar
    o = _lod(150.0, fp)
    h += (26.0 * steep + 6.0) * fbm2(x / 150.0 + 2.2, y / 150.0 - 0.7, o, 207) * (0.3 + 0.7 * crest)
    # ribs on steep ground
    o2 = _lod(55.0, fp)
    r = ridged2(x / 55.0 - 1.3, y / 55.0 + 4.4, o2, 210, 2.1, 0.55, 1.6)
    h += (r - 0.45) * 16.0 * steep * crest
    # small steps / boulders on the gentle snow, blocky outcrops on the steep
    o3 = _lod(16.0, fp)
    h += 0.9 * fbm2(x / 16.0 + 5.5, y / 16.0 - 3.1, o3, 208)
    o4 = _lod(9.0, fp)
    rb = ridged2(x / 9.0 + 1.7, y / 9.0 - 8.2, o4, 211, 2.0, 0.5, 1.2)
    h += (rb - 0.5) * 2.6 * steep
    return h + crest * gullies(x, y, fp, gx, gy, 14.0)


@njit(cache=True)
def height(x, y, fp, P, SP, Z, ZC, LP):
    """Full height with the story zone blended in. ZC = (cx, cy, r_in, r_out, unused)."""
    dz = math.sqrt((x - ZC[0]) ** 2 + (y - ZC[1]) ** 2)
    if dz >= ZC[3] or Z.shape[0] == 0:
        return pads(x, y, height0(x, y, fp, P, SP, LP), P)
    w = smoothstep(ZC[3], ZC[2], dz)
    hz = zone_height(x, y, fp, Z, P, SP, LP)
    if w >= 1.0:
        return pads(x, y, hz, P)
    hn = height0(x, y, fp, P, SP, LP)
    return pads(x, y, hn * (1.0 - w) + hz * w, P)


@njit(cache=True)
def height0(x, y, fp, P, SP, LP):
    """Natural height at (x, y): base + slope-aligned gullies. fp = sample footprint in metres."""
    e = max(fp, 2.0)
    h0 = base_height(x, y, fp, P, SP, LP)
    hx = base_height(x + e, y, fp, P, SP, LP)
    hy = base_height(x, y + e, fp, P, SP, LP)
    gx = (hx - h0) / e
    gy = (hy - h0) / e
    return h0 + LS * gullies(x / LS, y / LS, fp / LS, gx, gy, LP[4])


@njit(parallel=True, cache=True)
def raster(nx, ny, x0, y0, dx, P, SP, Z, ZC, LP):
    g = np.empty((ny, nx), np.float32)
    for j in prange(ny):
        for i in range(nx):
            g[j, i] = height(x0 + i * dx, y0 + j * dx, dx, P, SP, Z, ZC, LP)
    return g


@njit(parallel=True, cache=True)
def eval_points(X, Y, FP, P, SP, Z, ZC, LP):
    n = X.shape[0]
    out = np.empty(n, np.float64)
    for k in prange(n):
        out[k] = height(X[k], Y[k], FP[k], P, SP, Z, ZC, LP)
    return out
