"""Scene geometry for ACCORD: hooded emissaries and standing stones as SDFs,
analytic table / dais / ground, ray tracing + cone-traced soft shadows."""
import math

from numba import njit

from nbcore import FM, clamp, sstep, smin, sd_ellipsoid, sd_capsule, sd_capsule_r

# ------------------------------------------------------------ constants ---
TABLE_R = 2.56
TABLE_Z = 0.80
DAIS_R = 5.20
DAIS_Z = 0.12
BOWL_R = 0.50
LIP_R = 0.64
BOWL_D = 0.17

# figure param layout
F_X, F_Y, F_ANG, F_C, F_S, F_HS, F_WS, F_TYPE, F_SEED = 0, 1, 2, 3, 4, 5, 6, 7, 8
F_R, F_G, F_B = 9, 10, 11
F_SIDE = 12
F_HX, F_HY, F_HZ = 13, 14, 15          # hand (local)
F_TX, F_TY, F_TZ = 16, 17, 18          # torch direction (local, unit)
F_EX, F_EY, F_EZ = 19, 20, 21          # elbow (local)
F_BOW = 22                             # head bow (forward shift of hood, m)
F_BB = 23                              # 23..28 world AABB xmin,xmax,ymin,ymax,zmin,zmax
F_TLIT = 29                            # torch head lit (emissive char)
F_BREATH = 30
F_N = 32

# stone param layout
S_X, S_Y, S_C, S_S, S_HX, S_HY, S_H, S_TAPER, S_LX, S_LY, S_TNX, S_TNY, S_TOP, S_SEED, S_ALB = range(15)
S_BB = 15                               # 15..20
S_N = 22


@njit(**FM)
def sd_fig(px, py, pz, F, i):
    """Returns (distance, material). material: 0 cloth, 1 torch, 2 hands/skin, 3 face shadow, 4 fur/wrap."""
    dx = px - F[i, F_X]
    dy = py - F[i, F_Y]
    c = F[i, F_C]
    s = F[i, F_S]
    x = dx * c + dy * s
    y = -dx * s + dy * c
    z = pz
    hs = F[i, F_HS]
    ws = F[i, F_WS] * (1.0 + F[i, F_BREATH])
    seed = F[i, F_SEED]
    H0 = 1.38 * hs
    t = clamp(z / H0, 0.0, 1.0)
    tt = t ** 0.8
    a = (0.30 + (0.155 - 0.30) * tt) * ws
    b = (0.37 + (0.215 - 0.37) * tt) * ws
    if x < 0.0:
        a *= 1.0 + 0.35 * (1.0 - t) ** 3
    phi = math.atan2(y / b, x / a)
    om = 1.0 - t
    fold = 1.0 + (0.055 * math.sin(7.0 * phi + seed) + 0.03 * math.sin(13.0 * phi + 2.1 * seed)) * om ** 1.3
    q = math.sqrt((x / a) ** 2 + (y / b) ** 2) / fold
    d_body = (q - 1.0) * min(a, b) * fold * 0.85
    d_body = max(d_body, max(-z, z - H0))
    d_sh = sd_ellipsoid(x + 0.01, y, z - 1.36 * hs, 0.155 * ws, 0.245 * ws, 0.11 * hs)
    d = smin(d_body, d_sh, 0.07)
    # head / hood
    typ = int(F[i, F_TYPE])
    bow = F[i, F_BOW]
    hx = x + 0.025 - bow
    hz = z - 1.55 * hs
    mat_head = 0
    if typ == 0:      # pointed hood
        d_h = sd_ellipsoid(hx, y, hz, 0.155, 0.135, 0.17)
        d_p = sd_ellipsoid(hx + 0.10, y, hz - 0.08, 0.10, 0.08, 0.105)
        d_h = smin(d_h, d_p, 0.06)
        d = smin(d, d_h, 0.05)
    elif typ == 1:    # round deep hood
        d_h = sd_ellipsoid(hx + 0.01, y, hz + 0.01, 0.17, 0.145, 0.165)
        d = smin(d, d_h, 0.06)
    elif typ == 2:    # head-wrap
        d_h = sd_ellipsoid(hx, y, hz + 0.01, 0.11, 0.095, 0.13)
        d_w = sd_ellipsoid(hx + 0.01, y, hz - 0.07, 0.14, 0.145, 0.085)
        d_h = smin(d_h, d_w, 0.03)
        if d_w < d_h + 0.01:
            mat_head = 4
        d = smin(d, d_h, 0.04)
    elif typ == 3:    # broad shawl over head
        d_h = sd_ellipsoid(hx + 0.01, y, hz + 0.02, 0.175, 0.17, 0.15)
        d = smin(d, d_h, 0.10)
    else:             # bare head, heavy fur collar
        d_h = sd_ellipsoid(hx - 0.01, y, hz - 0.02, 0.10, 0.085, 0.125)
        # collar torus at neck
        rq = math.sqrt((x + 0.01) ** 2 / 1.0 + (y * 0.85) ** 2) - 0.15
        d_c = math.sqrt(rq * rq + (z - 1.43 * hs) ** 2) - 0.075
        d = smin(d, d_c, 0.03)
        if d_h < d:
            mat_head = 5
        d = min(d, d_h)
    # arm (sleeve) : shoulder -> elbow -> hand
    side = F[i, F_SIDE]
    sx, sy, sz = 0.0, side * 0.215 * ws, 1.33 * hs
    ex, ey, ez = F[i, F_EX], F[i, F_EY], F[i, F_EZ]
    hx2, hy2, hz2 = F[i, F_HX], F[i, F_HY], F[i, F_HZ]
    d_u = sd_capsule_r(x, y, z, sx, sy, sz, ex, ey, ez, 0.078, 0.066)
    d_f = sd_capsule_r(x, y, z, ex, ey, ez, hx2, hy2, hz2, 0.062, 0.080)
    d_arm = min(d_u, d_f)
    d = smin(d, d_arm, 0.045)
    mat = 0
    if mat_head != 0 and d_h < d + 0.004:
        mat = mat_head
    # hand
    d_hand = math.sqrt((x - hx2) ** 2 + (y - hy2) ** 2 + (z - hz2) ** 2) - 0.042
    if d_hand < d:
        d = d_hand
        mat = 2
    # torch
    tx, ty, tz = F[i, F_TX], F[i, F_TY], F[i, F_TZ]
    d_t = sd_capsule(x, y, z, hx2 - 0.10 * tx, hy2 - 0.10 * ty, hz2 - 0.10 * tz,
                     hx2 + 0.40 * tx, hy2 + 0.40 * ty, hz2 + 0.40 * tz, 0.021)
    d_th = sd_capsule(x, y, z, hx2 + 0.34 * tx, hy2 + 0.34 * ty, hz2 + 0.34 * tz,
                      hx2 + 0.48 * tx, hy2 + 0.48 * ty, hz2 + 0.48 * tz, 0.036)
    d_t = min(d_t, d_th)
    if d_t < d:
        d = d_t
        mat = 1
    # face shadow region (front of hood)
    if mat == 0 and typ != 4 and hx > 0.05 and hz > -0.12 and hz < 0.08 and abs(y) < 0.11:
        mat = 3
    return d, mat


@njit(**FM)
def sd_stone(px, py, pz, S, j):
    dx = px - S[j, S_X]
    dy = py - S[j, S_Y]
    c = S[j, S_C]
    s = S[j, S_S]
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    z = pz
    H = S[j, S_H]
    lx -= S[j, S_LX] * z
    ly -= S[j, S_LY] * z
    sc = 1.0 - S[j, S_TAPER] * clamp(z / H, 0.0, 1.0)
    qx = abs(lx) / sc - S[j, S_HX]
    qy = abs(ly) / sc - S[j, S_HY]
    qz = abs(z - 0.5 * H) - 0.5 * H
    rr = 0.10
    mx = max(qx + rr, 0.0)
    my = max(qy + rr, 0.0)
    mz = max(qz + rr, 0.0)
    d = math.sqrt(mx * mx + my * my + mz * mz) - rr + min(max(qx, max(qy, qz)), 0.0)
    d *= sc
    d = max(d, lx * S[j, S_TNX] + ly * S[j, S_TNY] + z - S[j, S_TOP])
    seed = S[j, S_SEED]
    d += 0.03 * (math.sin(3.1 * lx + seed) * math.sin(2.3 * z + 1.7 * seed)
                 + math.sin(4.7 * ly - 1.3 * seed) * math.sin(3.7 * z + seed))
    return d


@njit(inline='always', **FM)
def ray_aabb(ox, oy, oz, idx, idy, idz, x0, x1, y0, y1, z0, z1):
    tx0 = (x0 - ox) * idx
    tx1 = (x1 - ox) * idx
    if tx0 > tx1:
        tx0, tx1 = tx1, tx0
    ty0 = (y0 - oy) * idy
    ty1 = (y1 - oy) * idy
    if ty0 > ty1:
        ty0, ty1 = ty1, ty0
    tz0 = (z0 - oz) * idz
    tz1 = (z1 - oz) * idz
    if tz0 > tz1:
        tz0, tz1 = tz1, tz0
    tn = max(tx0, max(ty0, tz0))
    tf = min(tx1, min(ty1, tz1))
    return tn, tf


@njit(**FM)
def trace_fig(ox, oy, oz, dx, dy, dz, t0, t1, F, i, pix):
    t = max(t0, 0.0)
    for it in range(90):
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        d, m = sd_fig(px, py, pz, F, i)
        eps = max(0.0008, t * pix * 0.35)
        if d < eps:
            return t, m
        t += d * 0.75
        if t > t1:
            break
    return -1.0, 0


@njit(**FM)
def trace_stone(ox, oy, oz, dx, dy, dz, t0, t1, S, j, pix):
    t = max(t0, 0.0)
    for it in range(80):
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        d = sd_stone(px, py, pz, S, j)
        eps = max(0.001, t * pix * 0.35)
        if d < eps:
            return t
        t += d * 0.8
        if t > t1:
            break
    return -1.0


@njit(**FM)
def normal_fig(px, py, pz, F, i, h):
    k1, _ = sd_fig(px + h, py - h, pz - h, F, i)
    k2, _ = sd_fig(px - h, py - h, pz + h, F, i)
    k3, _ = sd_fig(px - h, py + h, pz - h, F, i)
    k4, _ = sd_fig(px + h, py + h, pz + h, F, i)
    nx = k1 - k2 - k3 + k4
    ny = -k1 - k2 + k3 + k4
    nz = -k1 + k2 - k3 + k4
    l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / l, ny / l, nz / l


@njit(**FM)
def normal_stone(px, py, pz, S, j, h):
    k1 = sd_stone(px + h, py - h, pz - h, S, j)
    k2 = sd_stone(px - h, py - h, pz + h, S, j)
    k3 = sd_stone(px - h, py + h, pz - h, S, j)
    k4 = sd_stone(px + h, py + h, pz + h, S, j)
    nx = k1 - k2 - k3 + k4
    ny = -k1 - k2 + k3 + k4
    nz = -k1 + k2 - k3 + k4
    l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / l, ny / l, nz / l


@njit(**FM)
def bowl_height(r):
    """Table-top relief near the hearth: bowl (r<BOWL_R) and raised lip ring."""
    h = 0.0
    if r < BOWL_R:
        u = r / BOWL_R
        h -= BOWL_D * (1.0 - u * u) ** 1.2
    # lip: smooth bump centred at 0.57
    lip = math.exp(-((r - 0.565) / 0.045) ** 2)
    h += 0.05 * lip
    return h


@njit(**FM)
def trace_bowl(ox, oy, oz, dx, dy, dz):
    """Height-field march in the hearth region. Returns t or -1."""
    # search between planes z = TABLE_Z+0.06 and z = TABLE_Z - BOWL_D - 0.01
    if dz >= -1e-6:
        return -1.0
    ta = (TABLE_Z + 0.06 - oz) / dz
    tb = (TABLE_Z - BOWL_D - 0.02 - oz) / dz
    n = 48
    prev_t = ta
    prev_f = 1.0
    for k in range(n + 1):
        t = ta + (tb - ta) * k / n
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        r = math.sqrt(px * px + py * py)
        if r > LIP_R + 0.05:
            f = pz - TABLE_Z
        else:
            f = pz - (TABLE_Z + bowl_height(r))
        if f < 0.0:
            # refine
            lo = prev_t
            hi = t
            for it in range(8):
                mid = 0.5 * (lo + hi)
                px = ox + dx * mid
                py = oy + dy * mid
                pz = oz + dz * mid
                r = math.sqrt(px * px + py * py)
                fm = pz - (TABLE_Z + bowl_height(r))
                if fm < 0.0:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        prev_t = t
        prev_f = f
    return -1.0


@njit(**FM)
def shadow_occluders(px, py, pz, lx, ly, lz, rl, F, nf, S, ns, skip_fig):
    """Soft visibility of spherical light (centre l, radius rl) from p against the table,
    figures and stones. Cone-traced SDF soft shadows."""
    vx = lx - px
    vy = ly - py
    vz = lz - pz
    D = math.sqrt(vx * vx + vy * vy + vz * vz)
    if D < 1e-6:
        return 1.0
    dx = vx / D
    dy = vy / D
    dz = vz / D
    vis = 1.0
    # table edge (analytic) for points outside the table and below its top
    rp = math.sqrt(px * px + py * py)
    if rp > TABLE_R and pz < TABLE_Z:
        # parameter where horizontal radius hits TABLE_R (light near axis)
        rl2 = math.sqrt(lx * lx + ly * ly)
        sc = (rp - TABLE_R) / max(rp - rl2, 1e-6)
        if sc > 0.0 and sc < 1.0:
            zc = pz + sc * vz
            w = rl * sc + 0.002
            vis = min(vis, sstep(-w, w, zc - TABLE_Z))
            if vis <= 0.001:
                return 0.0
    idx = 1.0 / dx if abs(dx) > 1e-9 else 1e9
    idy = 1.0 / dy if abs(dy) > 1e-9 else 1e9
    idz = 1.0 / dz if abs(dz) > 1e-9 else 1e9
    for i in range(nf):
        if i == skip_fig:
            continue
        # expand AABB by cone radius at far end (conservative)
        e = rl * 0.6
        t0, t1 = ray_aabb(px, py, pz, idx, idy, idz,
                          F[i, F_BB] - e, F[i, F_BB + 1] + e, F[i, F_BB + 2] - e, F[i, F_BB + 3] + e,
                          F[i, F_BB + 4] - e, F[i, F_BB + 5] + e)
        if t1 < max(t0, 0.0) or t0 > D:
            continue
        s = max(t0, 0.01)
        send = min(t1, D - 0.05)
        for it in range(40):
            qx = px + dx * s
            qy = py + dy * s
            qz = pz + dz * s
            d, m = sd_fig(qx, qy, qz, F, i)
            cr = rl * s / D + 0.004
            v = sstep(-1.0, 1.0, d / cr)
            if v < vis:
                vis = v
            if vis < 0.003:
                return 0.0
            s += max(abs(d) * 0.7, 0.015)
            if s > send:
                break
    for j in range(ns):
        e = rl * 0.6
        t0, t1 = ray_aabb(px, py, pz, idx, idy, idz,
                          S[j, S_BB] - e, S[j, S_BB + 1] + e, S[j, S_BB + 2] - e, S[j, S_BB + 3] + e,
                          S[j, S_BB + 4] - e, S[j, S_BB + 5] + e)
        if t1 < max(t0, 0.0) or t0 > D:
            continue
        s = max(t0, 0.01)
        send = min(t1, D - 0.05)
        for it in range(40):
            qx = px + dx * s
            qy = py + dy * s
            qz = pz + dz * s
            d = sd_stone(qx, qy, qz, S, j)
            cr = rl * s / D + 0.004
            v = sstep(-1.0, 1.0, d / cr)
            if v < vis:
                vis = v
            if vis < 0.003:
                return 0.0
            s += max(abs(d) * 0.75, 0.02)
            if s > send:
                break
    return vis


@njit(**FM)
def shadow_dir(px, py, pz, dx, dy, dz, soft, S, ns, F, nf):
    """Directional (moon) soft shadow against stones + figures (soft = tan of penumbra angle)."""
    vis = 1.0
    idx = 1.0 / dx if abs(dx) > 1e-9 else 1e9
    idy = 1.0 / dy if abs(dy) > 1e-9 else 1e9
    idz = 1.0 / dz if abs(dz) > 1e-9 else 1e9
    for j in range(ns):
        t0, t1 = ray_aabb(px, py, pz, idx, idy, idz,
                          S[j, S_BB], S[j, S_BB + 1], S[j, S_BB + 2], S[j, S_BB + 3],
                          S[j, S_BB + 4], S[j, S_BB + 5])
        if t1 < max(t0, 0.0):
            continue
        s = max(t0, 0.01)
        for it in range(30):
            d = sd_stone(px + dx * s, py + dy * s, pz + dz * s, S, j)
            cr = soft * s + 0.01
            v = sstep(-1.0, 1.0, d / cr)
            if v < vis:
                vis = v
            if vis < 0.01:
                return 0.0
            s += max(abs(d) * 0.8, 0.03)
            if s > t1:
                break
    for i in range(nf):
        t0, t1 = ray_aabb(px, py, pz, idx, idy, idz,
                          F[i, F_BB], F[i, F_BB + 1], F[i, F_BB + 2], F[i, F_BB + 3],
                          F[i, F_BB + 4], F[i, F_BB + 5])
        if t1 < max(t0, 0.0):
            continue
        s = max(t0, 0.01)
        for it in range(30):
            d, m = sd_fig(px + dx * s, py + dy * s, pz + dz * s, F, i)
            cr = soft * s + 0.01
            v = sstep(-1.0, 1.0, d / cr)
            if v < vis:
                vis = v
            if vis < 0.01:
                return 0.0
            s += max(abs(d) * 0.75, 0.02)
            if s > t1:
                break
    return vis
