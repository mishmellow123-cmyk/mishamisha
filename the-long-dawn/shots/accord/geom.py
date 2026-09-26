"""Scene geometry for ACCORD: hooded emissaries and standing stones as SDFs,
analytic table / dais / ground, ray tracing + cone-traced soft shadows."""
import math

from numba import njit

from nbcore import FM, clamp, sstep, smin, sd_ellipsoid, sd_capsule, sd_capsule_r

# ------------------------------------------------------------ constants ---
TABLE_R = 1.97
TABLE_Z = 0.62
DAIS_R = 6.30
DAIS_Z = 0.10
BOWL_R = 0.50
LIP_R = 0.64
BOWL_D = 0.17

# figure param layout
F_X, F_Y, F_ANG, F_C, F_S, F_HS, F_WS, F_TYPE, F_SEED = 0, 1, 2, 3, 4, 5, 6, 7, 8
F_R, F_G, F_B = 9, 10, 11              # main cloth albedo (linear)
F_SIDE = 12
F_HX, F_HY, F_HZ = 13, 14, 15          # hand (local)
F_TX, F_TY, F_TZ = 16, 17, 18          # torch direction (local, unit)
F_EX, F_EY, F_EZ = 19, 20, 21          # elbow (local)
F_BOW = 22                             # head bow (radians, forward)
F_BB = 23                              # 23..28 world AABB xmin,xmax,ymin,ymax,zmin,zmax
F_TLIT = 29                            # torch head lit (emissive char)
F_BREATH = 30
F_R2, F_G2, F_B2 = 31, 32, 33          # secondary cloth (hood / wrap / veil / cape / fur)
F_CAPE = 34                            # shoulder-cape drop below the shoulder line (m); 0 = none
F_TRAIN = 35                           # extra length of the cloak's back at the floor (m)
F_SKIN = 36                            # skin tone 0..1 (dark..light)
F_WEAVE = 37                           # weave: scale of the cloth's slub/twill (varies per emissary)
F_SHEENK = 38                          # sheen: 0.5 matte wool .. 1.6 velvet
F_HAND = 39                            # 0..1 hand drawn back into the sleeve (after the merge)
F_N = 40

# figure materials
M_CLOTH, M_TORCH, M_SKIN, M_SHADOW, M_CLOTH2, M_HAIR, M_FUR = 0, 1, 2, 3, 4, 5, 6

# stone param layout
S_X, S_Y, S_C, S_S, S_HX, S_HY, S_H, S_TAPER, S_LX, S_LY, S_TNX, S_TNY, S_TOP, S_SEED, S_ALB = range(15)
S_BB = 15                               # 15..20
S_N = 22


@njit(inline='always', **FM)
def _crease(w):
    """Cloth fold profile over phase w: rounded ridges, sharp creases (in [-0.5, 0.5])."""
    return abs(math.sin(0.5 * w)) - 0.5


@njit(**FM)
def _hood(hx, y, hz, typ, seed):
    """Hood / head in the (bowed) head frame. Returns (distance, material, in_cavity)."""
    if typ == 2:      # head-wrap: head + wound crown of cloth (ridged), face below it
        d_h = sd_ellipsoid(hx, y, hz - 0.005, 0.104, 0.090, 0.116)
        d_w = sd_ellipsoid(hx - 0.014, y, hz - 0.060, 0.128, 0.121, 0.084)
        d_w += 0.0045 * math.sin(52.0 * hz + 2.0 * math.atan2(y, hx - 0.014) + seed)
        if d_w < d_h + 0.004:
            return smin(d_h, d_w, 0.02), M_CLOTH2, False
        return smin(d_h, d_w, 0.02), M_SKIN, False
    if typ == 3:      # bare head, dark hair
        d_h = sd_ellipsoid(hx - 0.008, y, hz - 0.012, 0.098, 0.086, 0.114)
        if hx > 0.05 and hz < 0.03:
            return d_h, M_SKIN, False
        return d_h, M_HAIR, False
    # hood shells: outer - cavity, opened at the front; the head inside, in shadow
    if typ == 1:
        ao_, bo_, co_, xo = 0.172, 0.150, 0.172, 0.008
        ai_, bi_, ci_ = 0.136, 0.114, 0.138
        xf, oy, oz = 0.040, 0.092, 0.118
    elif typ == 4:    # veil: close to the head, small opening
        ao_, bo_, co_, xo = 0.134, 0.117, 0.146, 0.004
        ai_, bi_, ci_ = 0.108, 0.092, 0.120
        xf, oy, oz = 0.062, 0.070, 0.094
    else:             # 0 cowl-with-liripipe, 5 capuchin
        ao_, bo_, co_, xo = 0.158, 0.138, 0.166, 0.012
        ai_, bi_, ci_ = 0.124, 0.104, 0.134
        xf, oy, oz = 0.046, 0.086, 0.116
    d_out = sd_ellipsoid(hx + xo, y, hz - 0.004, ao_, bo_, co_)
    # the brim: the hood's upper front overhangs the face
    d_b = sd_ellipsoid(hx - 0.045, y, hz - 0.035, 0.150, bo_ * 0.92, 0.105)
    d_out = smin(d_out, d_b, 0.03)
    if typ == 1:      # even the round hoods draw to a soft point behind
        d_p = sd_capsule_r(hx, y, hz, -0.06, 0.0, 0.05, -0.16, 0.0, 0.02, 0.080, 0.035)
        d_out = smin(d_out, d_p, 0.05)
    if typ == 0:      # drawn back into a point that falls down the back
        d_p = sd_capsule_r(hx, y, hz, -0.04, 0.0, 0.07, -0.18, 0.0, -0.005, 0.085, 0.030)
        d_out = smin(d_out, d_p, 0.045)
    elif typ == 5:    # a tall point, up and back
        d_p = sd_capsule_r(hx, y, hz, -0.03, 0.0, 0.08, -0.11, 0.0, 0.27, 0.070, 0.008)
        d_out = smin(d_out, d_p, 0.035)
    # soft drape folds on the hood: creases running back from the opening
    ph = math.atan2(hz - 0.02, y) * 5.0 + seed
    d_out += 0.0075 * _crease(ph) * sstep(0.10, -0.12, hx)
    d_in = sd_ellipsoid(hx - 0.006, y, hz - 0.012, ai_, bi_, ci_)
    d_shell = max(d_out, -d_in)
    e = math.sqrt((y / oy) ** 2 + ((hz + 0.030) / oz) ** 2)
    d_cut = max(xf - hx, (e - 1.0) * min(oy, oz))
    d_shell = max(d_shell, -d_cut)
    d_head = sd_ellipsoid(hx - 0.004, y, hz + 0.004, 0.094, 0.080, 0.110)
    if d_head < d_shell:
        return d_head, M_SHADOW, True
    # the inside of the shell (the cavity wall) is in shadow too
    cav = -d_in > d_out and -d_in > -d_cut
    return d_shell, (M_SHADOW if cav else M_CLOTH), cav


@njit(**FM)
def sd_fig(px, py, pz, F, i):
    """Hooded emissary. Returns (distance, material) - see M_* ids.
    Local frame: x toward the hearth, y lateral, z up from the dais floor."""
    dx = px - F[i, F_X]
    dy = py - F[i, F_Y]
    c = F[i, F_C]
    s = F[i, F_S]
    x = dx * c + dy * s
    y = -dx * s + dy * c
    z = pz - DAIS_Z
    hs = F[i, F_HS]
    ws = F[i, F_WS] * (1.0 + F[i, F_BREATH])
    seed = F[i, F_SEED]
    typ = int(F[i, F_TYPE])
    zsh = 1.33 * hs                                   # shoulder line
    nfold = 5.0 + (seed * 3.7) % 2.5                  # 5..7.5 deep folds round the body
    # ---------------------------------------------------------------- cloak body
    zc = min(max(z, 0.0), zsh)
    t = zc / zsh
    fl = (1.0 - t) ** 2.2
    ab = (0.148 + 0.036 * fl) * ws                    # front half-depth
    if x < 0.0:                                       # the back falls further and trails on the floor
        ab = (0.160 + 0.025 * fl + F[i, F_TRAIN] * fl ** 1.6) * ws
    bb = (0.212 + 0.050 * fl) * ws
    phi = math.atan2(y / bb, x / ab)
    w = nfold * phi + seed + 1.3 * math.sin(2.3 * zc + seed) + 0.6 * math.sin(3.0 * phi + 1.7 * seed)
    famp = 0.022 + 0.070 * (1.0 - t) ** 1.3
    fold = 1.0 + famp * (_crease(w) + 0.40 * _crease(2.37 * w + 1.3 * seed + 1.7 * zc))
    q = math.sqrt((x / ab) ** 2 + (y / bb) ** 2) / fold
    if z < zsh:
        d_body = max((q - 1.0) * min(ab, bb) * 0.9, -z)
    else:
        # the shoulders rise to the neck; folds fan out from the neck across them
        rr = math.sqrt((x / ab) ** 2 + (y / bb) ** 2)
        fd = 1.0 + 0.030 * sstep(0.35, 1.0, rr) * (_crease(w) + 0.45 * _crease(1.9 * w + 1.3 * seed))
        d_body = sd_ellipsoid(x, y, z - zsh, ab * fd, bb * fd, 0.108 * hs)
    d = d_body
    mat = M_CLOTH
    # ------------------------------------------------------------ shoulder cape
    drop = F[i, F_CAPE]
    if drop > 0.0:
        zt = zsh - drop
        u = clamp((zsh - z) / drop, 0.0, 1.0)
        ac = (0.166 + 0.050 * u) * ws
        if x < 0.0:
            ac = (0.182 + 0.060 * u) * ws
        bc = (0.236 + 0.095 * u) * ws
        phc = math.atan2(y / bc, x / ac)
        wc = (nfold + 2.0) * phc + 0.7 * seed
        fc = 1.0 + (0.020 + 0.085 * u) * (_crease(wc) + 0.45 * _crease(2.3 * wc + seed))
        qc = math.sqrt((x / ac) ** 2 + (y / bc) ** 2) / fc
        # the hem hangs lower where the mantle drapes over the torch arm
        zt = zsh - drop * (1.0 + 0.25 * sstep(0.0, 0.2, y * F[i, F_SIDE]))
        if z < zsh:
            d_cape = max((qc - 1.0) * min(ac, bc) * 0.9, zt - z)
        else:
            rr = math.sqrt((x / ac) ** 2 + (y / bc) ** 2)
            fd = 1.0 + 0.035 * sstep(0.35, 1.0, rr) * _crease(wc)
            d_cape = sd_ellipsoid(x, y, z - zsh, ac * fd, bc * fd, 0.120 * hs)
        if d_cape < d:
            d = d_cape
            mat = M_CLOTH2
    # --------------------------------------------------------------- head/hood
    bow = F[i, F_BOW]
    cb = math.cos(bow)
    sb = math.sin(bow)
    xr = x - 0.012
    zr = z - (zsh + 0.07 * hs)
    xh = cb * xr - sb * zr                            # head frame (un-bowed)
    zh = sb * xr + cb * zr
    hx = xh - 0.015
    hz = zh - 0.12 * hs
    # the head's bounding sphere: far from it, a cheap lower bound stands in for the hood math
    hb = math.sqrt(hx * hx + y * y + (hz - 0.06) ** 2)
    far = hb > 0.52
    if far:
        d_h, hm, cav = hb - 0.44, M_CLOTH, False
    else:
        d_h, hm, cav = _hood(hx, y, hz, typ, seed)
    # the cowl: the hood's fall gathers round the neck onto the shoulders
    if typ != 2 and typ != 3 and not far:
        ang = math.atan2(y, x + 0.01)
        rq = (math.sqrt(((x + 0.01) / 0.130) ** 2 + (y / 0.158) ** 2) - 1.0) * 0.14
        zq = z - (zsh + 0.062 * hs)
        d_cw = math.sqrt(rq * rq + (zq * 1.25) ** 2) - 0.050 + 0.006 * _crease(11.0 * ang + seed)
        if typ == 4:
            d_cw -= 0.012
        d_h = smin(d_h, d_cw, 0.035)
        if d_cw < d_h + 0.004 and not cav:
            hm = M_CLOTH2 if typ == 4 else M_CLOTH
    if typ == 0:      # the liripipe down the back
        d_l = sd_capsule_r(x, y, z, -0.160, 0.012, zsh + 0.13 * hs, -0.240, 0.03, zsh - 0.30 * hs, 0.032, 0.018)
        d_h = smin(d_h, d_l, 0.03)
    elif typ == 2:    # the wrap's tail over the shoulder
        d_t = sd_capsule_r(x, y, z, -0.085, 0.05, zsh + 0.20 * hs, -0.17, 0.13, zsh - 0.24 * hs, 0.034, 0.022)
        if d_t < d_h:
            hm = M_CLOTH2
        d_h = smin(d_h, d_t, 0.03)
    elif typ == 3:    # the hood thrown back: a heavy fold of cloth lying behind the neck
        ang = math.atan2(y, x)
        wb = sstep(0.8, 2.5, abs(ang))                    # 0 at the front, 1 behind
        rq = (math.sqrt((x / (0.150 + 0.03 * wb)) ** 2 + (y / 0.190) ** 2) - 1.0) * 0.17 * ws
        zq = z - (zsh + (0.035 - 0.045 * wb) * hs)
        tw = 0.007 * _crease(7.0 * ang + 12.0 * zq + seed)
        d_c = math.sqrt(rq * rq + (zq * 1.1) ** 2) - (0.026 + 0.072 * wb) * ws + tw
        d_prev = d
        d = smin(d, d_c, 0.03)
        if d_c < d_prev:
            mat = M_CLOTH2
    elif typ == 4:    # the veil falls over the shoulders and upper back
        u = clamp((zsh + 0.05 - z) / 0.36, 0.0, 1.0)
        av = (0.150 + 0.030 * u) * ws
        if x < 0.0:
            av = (0.168 + 0.045 * u) * ws
        bv = (0.185 + 0.070 * u) * ws
        phv = math.atan2(y / bv, x / av)
        fv = 1.0 + (0.012 + 0.035 * u) * _crease((nfold + 2.0) * phv + seed)
        qv = math.sqrt((x / av) ** 2 + (y / bv) ** 2) / fv
        d_v = max((qv - 1.0) * min(av, bv) * 0.9, (zsh - 0.31 * hs) - z)
        d_v = max(d_v, z - (zsh + 0.16 * hs))
        d_h = smin(d_h, d_v, 0.06)
        if d_v < d_h + 0.004 and not cav:
            hm = M_CLOTH2
        if not cav and hm == M_CLOTH:
            hm = M_CLOTH2
    kb = 0.045
    if typ == 2 or typ == 3:
        kb = 0.03
    d_prev = d
    d = smin(d, d_h, kb)
    if d_h < d_prev + 0.006:
        mat = hm
    # ------------------------------------------------------------ sleeve + hand
    side = F[i, F_SIDE]
    sx, sy, sz = 0.0, side * 0.200 * ws, zsh - 0.035 * hs
    ex, ey, ez = F[i, F_EX], F[i, F_EY], F[i, F_EZ]
    hx2, hy2, hz2 = F[i, F_HX], F[i, F_HY], F[i, F_HZ]
    d_u = sd_capsule_r(x, y, z, sx, sy, sz, ex, ey, ez, 0.074, 0.070)
    d_f = sd_capsule_r(x, y, z, ex, ey, ez, hx2, hy2, hz2, 0.068, 0.088)
    # the sleeve's own folds
    d_arm = min(d_u, d_f) + 0.004 * _crease(40.0 * (x + y + z) + seed)
    d_prev = d
    d = smin(d, d_arm, 0.04)
    if d_arm < d_prev and mat != M_CLOTH2 and mat != M_SHADOW:
        mat = M_CLOTH
    tx, ty, tz = F[i, F_TX], F[i, F_TY], F[i, F_TZ]
    kh = 0.03 - 0.06 * F[i, F_HAND]
    d_hand = math.sqrt((x - hx2 - kh * tx) ** 2 + (y - hy2 - kh * ty) ** 2 + (z - hz2 - kh * tz) ** 2) \
        - 0.034 * (1.0 - 0.35 * F[i, F_HAND])
    if d_hand < d:
        d = d_hand
        mat = M_SKIN
    # torch
    d_t = sd_capsule(x, y, z, hx2 - 0.22 * tx, hy2 - 0.22 * ty, hz2 - 0.22 * tz,
                     hx2 + 0.40 * tx, hy2 + 0.40 * ty, hz2 + 0.40 * tz, 0.020)
    d_th = sd_capsule(x, y, z, hx2 + 0.30 * tx, hy2 + 0.30 * ty, hz2 + 0.30 * tz,
                      hx2 + 0.47 * tx, hy2 + 0.47 * ty, hz2 + 0.47 * tz, 0.027)
    d_t = min(d_t, d_th)
    if d_t < d:
        d = d_t
        mat = M_TORCH
    return d, mat


@njit(**FM)
def sd_fig_proxy(px, py, pz, F, i):
    """Cheap silhouette of an emissary for soft shadows: the cloak's mass (no folds), the hood's
    outer shell, the arm and the torch."""
    dx = px - F[i, F_X]
    dy = py - F[i, F_Y]
    c = F[i, F_C]
    s = F[i, F_S]
    x = dx * c + dy * s
    y = -dx * s + dy * c
    z = pz - DAIS_Z
    hs = F[i, F_HS]
    ws = F[i, F_WS]
    zsh = 1.33 * hs
    zc = min(max(z, 0.0), zsh)
    fl = (1.0 - zc / zsh) ** 2.2
    ab = (0.148 + 0.036 * fl) * ws
    if x < 0.0:
        ab = (0.160 + 0.025 * fl + F[i, F_TRAIN] * fl ** 1.6) * ws
    bb = (0.212 + 0.050 * fl) * ws
    drop = F[i, F_CAPE]
    if drop > 0.0 and z > zsh - 1.25 * drop:
        ab += 0.045 * ws
        bb += 0.075 * ws
    if z < zsh:
        d = max((math.sqrt((x / ab) ** 2 + (y / bb) ** 2) - 1.0) * min(ab, bb) * 0.9, -z)
    else:
        d = sd_ellipsoid(x, y, z - zsh, ab, bb, 0.108 * hs)
    bow = F[i, F_BOW]
    xr = x - 0.012
    zr = z - (zsh + 0.07 * hs)
    xh = math.cos(bow) * xr - math.sin(bow) * zr - 0.015
    zh = math.sin(bow) * xr + math.cos(bow) * zr - 0.12 * hs
    d = smin(d, sd_ellipsoid(xh + 0.01, y, zh, 0.16, 0.14, 0.17), 0.04)
    side = F[i, F_SIDE]
    d_u = sd_capsule(x, y, z, 0.0, side * 0.2 * ws, zsh - 0.035 * hs, F[i, F_EX], F[i, F_EY], F[i, F_EZ], 0.072)
    d_f = sd_capsule(x, y, z, F[i, F_EX], F[i, F_EY], F[i, F_EZ], F[i, F_HX], F[i, F_HY], F[i, F_HZ], 0.075)
    d = min(d, min(d_u, d_f))
    tx, ty, tz = F[i, F_TX], F[i, F_TY], F[i, F_TZ]
    hx2, hy2, hz2 = F[i, F_HX], F[i, F_HY], F[i, F_HZ]
    d_t = sd_capsule(x, y, z, hx2 - 0.22 * tx, hy2 - 0.22 * ty, hz2 - 0.22 * tz,
                     hx2 + 0.47 * tx, hy2 + 0.47 * ty, hz2 + 0.47 * tz, 0.024)
    return min(d, d_t)


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
        for it in range(32):
            qx = px + dx * s
            qy = py + dy * s
            qz = pz + dz * s
            d = sd_fig_proxy(qx, qy, qz, F, i)
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
        for it in range(24):
            d = sd_fig_proxy(px + dx * s, py + dy * s, pz + dz * s, F, i)
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
