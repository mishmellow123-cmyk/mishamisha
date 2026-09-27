"""ACCORD v3 geometry (cut C, the council): the flat stone, the cold hearth, the emissaries and her,
gloved hands, torches, the real-size Ring, and analytic soft-shadow occluders.

World: metres, z up, origin = the centre of the flat stone on the ground. Figures use a local frame:
x toward where they face, y to their left, z up from the ground.
"""
import math

from numba import njit

from nbcore import FM, clamp, sstep, smin, sd_ellipsoid, sd_capsule, sd_capsule_r, vnoise2, fbm2
from geom import _crease, _hood, sd_stone, ray_aabb  # noqa: F401  (v2 hood shells and standing stones)

# ------------------------------------------------------------- the centre ---
STONE_R = 0.37          # the flat stone: mean radius
STONE_TOP = 0.36        # its top (z)
STONE_BOT = -0.12
ASH_R0, ASH_R1 = 0.40, 0.84   # the ash bed of the hearth (annulus round the stone)
KERB_R = 0.92           # the ring of blackened kerb stones
NKERB = 19
HEARTH_ZMAX = 0.20

# ------------------------------------------------------------ figure rows ---
F_X, F_Y, F_ANG, F_C, F_S, F_HS, F_WS, F_TYPE, F_SEED = 0, 1, 2, 3, 4, 5, 6, 7, 8
F_R, F_G, F_B = 9, 10, 11
F_SIDE = 12
F_HX, F_HY, F_HZ = 13, 14, 15           # torch hand (local)
F_TX, F_TY, F_TZ = 16, 17, 18           # torch axis (local unit, hand -> head)
F_EX, F_EY, F_EZ = 19, 20, 21           # torch elbow
F_BOW = 22
F_BB = 23                               # 23..28 world AABB
F_TLIT = 29                             # embers in the torch head (0..1)
F_BREATH = 30
F_R2, F_G2, F_B2 = 31, 32, 33           # second cloth (hood, mantle, shawl)
F_CAPE = 34
F_TRAIN = 35
F_GILT = 36                             # the torch hand is crusted with gold (0..1)
F_WEAVE = 37
F_SHEENK = 38
F_TORCH = 39                            # 1 = carries a torch
F_KNEEL = 40                            # 0..1
F_LEAN = 41                             # forward pitch of the upper body (rad)
F_ARM2 = 42                             # 1 = the second arm is out of the cloak
F_E2X, F_E2Y, F_E2Z = 43, 44, 45
F_H2X, F_H2Y, F_H2Z = 46, 47, 48
F_GRIP = 49                             # unused (kept for layout)
F_GRIP2 = 50                            # second hand closed (0 open .. 1 fist)
F_SHAWL = 51                            # 1 = the red woven shawl
F_SWAY = 52                             # walking: lateral sway of the upper body (m)
F_HEM = 53                              # walking: hem swing (m)
F_PHASE = 54                            # walking phase (rad)
F_SHX, F_SHY, F_SHZ = 55, 56, 57        # torch shoulder (local)
F_S2X, F_S2Y, F_S2Z = 58, 59, 60        # second shoulder (local)
F_D2X, F_D2Y, F_D2Z = 61, 62, 63        # second hand: finger direction (local unit)
F_HEROHAND = 64                         # 1 = the torch hand is drawn by the detailed hand (sd_hand)
F_N = 66

# materials
M_CLOTH, M_TORCH, M_GLOVE, M_SHADOW, M_CLOTH2, M_HAIR, M_FUR, M_SHAWL, M_WOOD, M_GILT = range(10)

# hearth object ids
H_FLAT, H_KERB, H_LOG, H_CHAR = 1, 2, 3, 4

# standing stones (v2 layout)
S_X, S_Y, S_C, S_S, S_HX, S_HY, S_H, S_TAPER, S_LX, S_LY, S_TNX, S_TNY, S_TOP, S_SEED, S_ALB = range(15)
S_BB = 15
S_N = 22

# the detailed hand: H[0:3] wrist, H[3:6] a (wrist->knuckles), H[6:9] b (thumb side), H[9:12] c (back of hand),
# H[12] scale, H[13] forearm length, H[14] gilt, H[15] sleeve (0/1); joints J (5 fingers x 4 points x 3) local (u,v,w)
HD_N = 16
# the Ring: RP[0] on, RP[1:4] centre, RP[4:7] axis, RP[7] glow, RP[8] hide-inside-fist
R_IN, R_THICK, R_WIDTH, R_SQ = 0.0094, 0.0023, 0.0052, 2.8
R_MID = R_IN + 0.5 * R_THICK
RP_N = 10


# =============================================================== the centre ===

@njit(**FM)
def flat_radius(th):
    return STONE_R * (1.0 + 0.055 * math.sin(3.0 * th + 1.1) + 0.035 * math.sin(5.0 * th + 2.3)
                      + 0.018 * math.sin(8.0 * th + 0.4))


@njit(**FM)
def flat_top(r, th):
    """The worn top of the flat stone: nearly flat, the rim weathered down."""
    R = flat_radius(th)
    u = r / R
    return STONE_TOP - 0.018 * max(u - 0.55, 0.0) ** 2 * 4.0 + 0.004 * math.sin(2.0 * th + 0.7) * u


@njit(**FM)
def sd_flat(px, py, pz):
    r = math.sqrt(px * px + py * py)
    th = math.atan2(py, px)
    R = flat_radius(th)
    zc = 0.5 * (STONE_TOP + STONE_BOT)
    hh = 0.5 * (STONE_TOP - STONE_BOT)
    # the sides bulge a little at mid-height
    zz = (pz - zc) / hh
    Rb = R * (1.0 + 0.035 * max(1.0 - zz * zz, 0.0))
    e = 0.055
    top = flat_top(min(r, R), th)
    qx = r - (Rb - e)
    qz = max(pz - (top - e), (STONE_BOT + e) - pz)
    mx = max(qx, 0.0)
    mz = max(qz, 0.0)
    d = math.sqrt(mx * mx + mz * mz) + min(max(qx, qz), 0.0) - e
    # weathering
    d += 0.0035 * (vnoise2(th * 9.0, pz * 14.0, 301) - 0.5) + 0.002 * (vnoise2(px * 40.0, py * 40.0 + pz * 30.0, 302) - 0.5)
    return d


@njit(**FM)
def sd_hearth(px, py, pz, KB, LG, nlog, CH, nch):
    """The cold hearth round the flat stone: kerb stones (KB: x,y,z,ax,ay,az,yaw_c,yaw_s,seed), laid logs and
    sticks (LG: capsules ax,ay,az,bx,by,bz,ra,rb,seed,kind), charcoal lumps (CH: x,y,z,rx,ry,rz,seed).
    Returns (distance, part id)."""
    d = 1e9
    part = 0
    r = math.sqrt(px * px + py * py)
    if r > KERB_R - 0.16 and r < KERB_R + 0.16:
        th = math.atan2(py, px)
        k0 = int(math.floor(th / (2.0 * math.pi / NKERB) + 0.5))
        for dk in range(-1, 2):
            k = (k0 + dk) % NKERB
            lx = px - KB[k, 0]
            ly = py - KB[k, 1]
            lz = pz - KB[k, 2]
            c = KB[k, 6]
            s = KB[k, 7]
            x = lx * c + ly * s
            y = -lx * s + ly * c
            dd = sd_ellipsoid(x, y, lz, KB[k, 3], KB[k, 4], KB[k, 5])
            dd += 0.006 * (vnoise2(x * 40.0 + KB[k, 8], y * 40.0 + lz * 30.0, 311) - 0.5)
            if dd < d:
                d = dd
                part = H_KERB
    if r < ASH_R1 + 0.06 and pz < HEARTH_ZMAX:
        for k in range(nlog):
            dd = sd_capsule_r(px, py, pz, LG[k, 0], LG[k, 1], LG[k, 2], LG[k, 3], LG[k, 4], LG[k, 5],
                              LG[k, 6], LG[k, 7])
            if dd < d + 0.02:
                # bark and splits
                ang = math.atan2(pz - 0.5 * (LG[k, 2] + LG[k, 5]), (px - LG[k, 0]) * 0.7 + (py - LG[k, 1]) * 0.7)
                dd += 0.0025 * math.sin(9.0 * ang + LG[k, 8]) + 0.002 * (vnoise2(px * 60.0, py * 60.0, 312) - 0.5)
            if dd < d:
                d = dd
                part = H_LOG
        for k in range(nch):
            dd = sd_ellipsoid(px - CH[k, 0], py - CH[k, 1], pz - CH[k, 2], CH[k, 3], CH[k, 4], CH[k, 5])
            dd += 0.004 * (vnoise2(px * 70.0 + CH[k, 6], py * 70.0, 313) - 0.5)
            if dd < d:
                d = dd
                part = H_CHAR
    return d, part


# ================================================================== figures ===

@njit(inline='always', **FM)
def _local(px, py, pz, F, i):
    dx = px - F[i, F_X]
    dy = py - F[i, F_Y]
    c = F[i, F_C]
    s = F[i, F_S]
    return dx * c + dy * s, -dx * s + dy * c, pz


@njit(**FM)
def fig_zsh(F, i):
    """Shoulder line of figure i (kneeling lowers it)."""
    return 1.33 * F[i, F_HS] - 0.43 * F[i, F_HS] * F[i, F_KNEEL]


@njit(**FM)
def _fist(x, y, z, hx, hy, hz, tx, ty, tz, bx, by, bz, gilt):
    """A gloved fist round a shaft: a knuckle block along the shaft with finger grooves and a thumb.
    (hx,hy,hz) = grip centre on the shaft axis, t = shaft axis, b = back-of-hand direction (unit, _|_ t)."""
    cx = ty * bz - tz * by
    cy = tz * bx - tx * bz
    cz = tx * by - ty * bx
    qx = x - hx
    qy = y - hy
    qz = z - hz
    a = qx * tx + qy * ty + qz * tz          # along the shaft
    b = qx * bx + qy * by + qz * bz          # toward the back of the hand
    c = qx * cx + qy * cy + qz * cz          # across
    # the wrapped fingers: a rounded block round the shaft, offset toward the back of the hand
    ea = abs(a) - 0.040
    eb = abs(b - 0.010) - 0.030
    ec = abs(c) - 0.026
    rr = 0.018
    mx = max(ea + rr, 0.0)
    my = max(eb + rr, 0.0)
    mz = max(ec + rr, 0.0)
    d = math.sqrt(mx * mx + my * my + mz * mz) - rr + min(max(ea, max(eb, ec)), 0.0)
    # finger grooves across the front (c > 0 side, the fingertips' side) and knuckle bumps on the back
    g = math.cos(a / 0.0205 * math.pi)
    d += 0.0022 * g * sstep(0.0, 0.02, c)
    d -= 0.0028 * max(g, 0.0) * sstep(0.02, 0.035, b)
    # the thumb, wrapped round the shaft on the other side
    dt = sd_capsule_r(a, b, c, 0.038, -0.004, -0.024, 0.030, -0.018, 0.020, 0.0105, 0.0095)
    d = smin(d, dt, 0.008)
    # the cuff
    dc = sd_capsule_r(a, b, c, -0.046, 0.012, 0.0, -0.075, 0.020, 0.0, 0.030, 0.034)
    d = smin(d, dc, 0.010)
    return d


@njit(**FM)
def _open_hand(x, y, z, hx, hy, hz, fx, fy, fz, grip):
    """Her second gloved hand (seen small): palm facing down, fingers along f; grip closes it."""
    # palm frame: a = f (fingers), w = up-ish (back of hand), v = a x w
    wx, wy, wz = 0.0, 0.0, 1.0
    dot = wx * fx + wy * fy + wz * fz
    wx -= dot * fx
    wy -= dot * fy
    wz -= dot * fz
    wl = math.sqrt(wx * wx + wy * wy + wz * wz) + 1e-9
    wx /= wl
    wy /= wl
    wz /= wl
    vx = fy * wz - fz * wy
    vy = fz * wx - fx * wz
    vz = fx * wy - fy * wx
    qx = x - hx
    qy = y - hy
    qz = z - hz
    a = qx * fx + qy * fy + qz * fz
    v = qx * vx + qy * vy + qz * vz
    w = qx * wx + qy * wy + qz * wz
    # palm
    d = sd_ellipsoid(a - 0.02, v, w, 0.050, 0.040, 0.016)
    # fingers as one mitten-like fan that curls under (grip)
    ang = 1.25 * grip
    fl = 0.075
    ex = 0.065 + fl * math.cos(ang)
    ew = -fl * math.sin(ang)
    for k in range(4):
        vo = (k - 1.5) * 0.0185
        df = sd_capsule_r(a, v, w, 0.060, vo, 0.0, 0.060 + (ex - 0.060) * 0.55, vo * 1.1, ew * 0.55, 0.0095, 0.0088)
        df = min(df, sd_capsule_r(a, v, w, 0.060 + (ex - 0.060) * 0.55, vo * 1.1, ew * 0.55, ex - 0.004 * abs(k - 1.5),
                                  vo * 1.15, ew - 0.012 * grip, 0.0088, 0.0078))
        d = smin(d, df, 0.006)
    dt = sd_capsule_r(a, v, w, 0.005, 0.038, -0.006, 0.045 + 0.02 * (1 - grip), 0.050 - 0.02 * grip,
                      -0.014 - 0.01 * grip, 0.0105, 0.0090)
    d = smin(d, dt, 0.008)
    return d


@njit(**FM)
def sd_fig(px, py, pz, F, i):
    """A hooded, cloaked figure. Returns (distance, material). Local: x forward, y left, z up."""
    x0, y0, z = _local(px, py, pz, F, i)
    hs = F[i, F_HS]
    ws = F[i, F_WS] * (1.0 + F[i, F_BREATH])
    seed = F[i, F_SEED]
    typ = int(F[i, F_TYPE])
    kneel = F[i, F_KNEEL]
    zsh = fig_zsh(F, i)
    # ---- the pose warp: the upper body leans forward about the hip and sways when walking
    zhip = 0.62 * zsh
    lean = F[i, F_LEAN] * sstep(zhip - 0.20, zhip + 0.25, z)
    x = x0
    y = y0 - F[i, F_SWAY] * sstep(0.35 * zsh, zsh, z)
    if lean != 0.0:
        cl = math.cos(lean)
        sl = math.sin(lean)
        rx = x
        rz = z - zhip
        x = cl * rx - sl * rz
        z = sl * rx + cl * rz + zhip
    nfold = 5.0 + (seed * 3.7) % 2.5
    # ---- the cloak's body (kneeling: the skirt pools round the knees and heels)
    zc = min(max(z, 0.0), zsh)
    t = zc / zsh
    fl = (1.0 - t) ** 2.2
    pool = kneel * (1.0 - t) ** 3.0
    ab = (0.148 + 0.036 * fl + 0.10 * pool) * ws
    hem = F[i, F_HEM] * math.sin(F[i, F_PHASE]) * fl
    if x < 0.0:
        ab = (0.160 + 0.025 * fl + F[i, F_TRAIN] * fl ** 1.6 + 0.16 * pool) * ws - hem
    else:
        ab += hem
    bb = (0.212 + 0.050 * fl + 0.06 * pool) * ws
    phi = math.atan2(y / bb, x / ab)
    w = nfold * phi + seed + 1.3 * math.sin(2.3 * zc + seed) + 0.6 * math.sin(3.0 * phi + 1.7 * seed)
    famp = 0.022 + 0.070 * (1.0 - t) ** 1.3
    fold = 1.0 + famp * (_crease(w) + 0.40 * _crease(2.37 * w + 1.3 * seed + 1.7 * zc))
    q = math.sqrt((x / ab) ** 2 + (y / bb) ** 2) / fold
    if z < zsh:
        d_body = max((q - 1.0) * min(ab, bb) * 0.9, -z)
    else:
        rr = math.sqrt((x / ab) ** 2 + (y / bb) ** 2)
        fd = 1.0 + 0.030 * sstep(0.35, 1.0, rr) * (_crease(w) + 0.45 * _crease(1.9 * w + 1.3 * seed))
        d_body = sd_ellipsoid(x, y, z - zsh, ab * fd, bb * fd, 0.108 * hs)
    d = d_body
    mat = M_CLOTH
    # ---- mantle or shawl over the shoulders
    drop = F[i, F_CAPE]
    shawl = F[i, F_SHAWL] > 0.5
    if drop > 0.0:
        u = clamp((zsh - z) / drop, 0.0, 1.0)
        ac = (0.166 + 0.050 * u) * ws
        if x < 0.0:
            ac = (0.182 + 0.060 * u) * ws
        bc = (0.236 + 0.095 * u) * ws
        phc = math.atan2(y / bc, x / ac)
        wc = (nfold + 2.0) * phc + 0.7 * seed
        fc = 1.0 + (0.020 + 0.085 * u) * (_crease(wc) + 0.45 * _crease(2.3 * wc + seed))
        qc = math.sqrt((x / ac) ** 2 + (y / bc) ** 2) / fc
        if shawl:
            # a woven shawl: crossed at the front, its ends hanging lower in front, shorter behind
            zt = zsh - drop * (1.0 + 0.55 * sstep(-0.02, 0.12, x) - 0.25 * sstep(0.0, -0.15, x))
        else:
            zt = zsh - drop * (1.0 + 0.25 * sstep(0.0, 0.2, y * F[i, F_SIDE]))
        if z < zsh:
            d_cape = max((qc - 1.0) * min(ac, bc) * 0.9, zt - z)
        else:
            rr = math.sqrt((x / ac) ** 2 + (y / bc) ** 2)
            fd = 1.0 + 0.035 * sstep(0.35, 1.0, rr) * _crease(wc)
            d_cape = sd_ellipsoid(x, y, z - zsh, ac * fd, bc * fd, 0.120 * hs)
        if d_cape < d:
            d = d_cape
            mat = M_SHAWL if shawl else M_CLOTH2
    # ---- head and hood
    bow = F[i, F_BOW]
    cb = math.cos(bow)
    sb = math.sin(bow)
    xr = x - 0.012
    zr = z - (zsh + 0.07 * hs)
    xh = cb * xr - sb * zr
    zh = sb * xr + cb * zr
    hx = xh - 0.015
    hz = zh - 0.12 * hs
    hb = math.sqrt(hx * hx + y * y + (hz - 0.06) ** 2)
    far = hb > 0.52
    if far:
        d_h, hm, cav = hb - 0.44, M_CLOTH, False
    else:
        d_h, hm, cav = _hood(hx, y, hz, typ, seed)
        if hm == 2:           # v2's skin (a face): never lit in v3 - it stays in the hood's shadow
            hm = M_SHADOW
    if typ != 3 and not far:
        ang = math.atan2(y, x + 0.01)
        rq = (math.sqrt(((x + 0.01) / 0.130) ** 2 + (y / 0.158) ** 2) - 1.0) * 0.14
        zq = z - (zsh + 0.062 * hs)
        d_cw = math.sqrt(rq * rq + (zq * 1.25) ** 2) - 0.050 + 0.006 * _crease(11.0 * ang + seed)
        d_h = smin(d_h, d_cw, 0.035)
        if d_cw < d_h + 0.004 and not cav:
            hm = M_CLOTH
    if typ == 0:
        d_l = sd_capsule_r(x, y, z, -0.160, 0.012, zsh + 0.13 * hs, -0.240, 0.03, zsh - 0.30 * hs, 0.032, 0.018)
        d_h = smin(d_h, d_l, 0.03)
    elif typ == 3:
        ang = math.atan2(y, x)
        wb = sstep(0.8, 2.5, abs(ang))
        rq = (math.sqrt((x / (0.150 + 0.03 * wb)) ** 2 + (y / 0.190) ** 2) - 1.0) * 0.17 * ws
        zq = z - (zsh + (0.035 - 0.045 * wb) * hs)
        tw = 0.007 * _crease(7.0 * ang + 12.0 * zq + seed)
        d_c = math.sqrt(rq * rq + (zq * 1.1) ** 2) - (0.026 + 0.072 * wb) * ws + tw
        d_prev = d
        d = smin(d, d_c, 0.03)
        if d_c < d_prev:
            mat = M_CLOTH2
    kb = 0.045
    if typ == 3:
        kb = 0.03
    d_prev = d
    d = smin(d, d_h, kb)
    if d_h < d_prev + 0.006:
        mat = hm
    # ---- arms (unwarped local coordinates: scene3 places the joints in the leaned pose)
    if F[i, F_TORCH] > 0.5:
        sx, sy, sz = F[i, F_SHX], F[i, F_SHY], F[i, F_SHZ]
        ex, ey, ez = F[i, F_EX], F[i, F_EY], F[i, F_EZ]
        hx2, hy2, hz2 = F[i, F_HX], F[i, F_HY], F[i, F_HZ]
        tx, ty, tz = F[i, F_TX], F[i, F_TY], F[i, F_TZ]
        d_u = sd_capsule_r(x0, y0, pz, sx, sy, sz, ex, ey, ez, 0.074, 0.068)
        # the sleeve ends a hand's length short of the grip
        kx = hx2 - ex
        ky = hy2 - ey
        kz = hz2 - ez
        kl = math.sqrt(kx * kx + ky * ky + kz * kz) + 1e-9
        cuff = max(kl - 0.070, 0.02) / kl
        d_f = sd_capsule_r(x0, y0, pz, ex, ey, ez, ex + kx * cuff, ey + ky * cuff, ez + kz * cuff, 0.066, 0.078)
        d_arm = min(d_u, d_f) + 0.004 * _crease(40.0 * (x0 + y0 + pz) + seed)
        d_prev = d
        d = smin(d, d_arm, 0.04)
        if d_arm < d_prev and mat != M_CLOTH2 and mat != M_SHADOW and mat != M_SHAWL:
            mat = M_CLOTH
        # the gloved fist: back of the hand faces away from the shoulder
        if F[i, F_HEROHAND] < 0.5:
            bx = hx2 - sx
            by = hy2 - sy
            bz = 0.0
            dt_ = bx * tx + by * ty + bz * tz
            bx -= dt_ * tx
            by -= dt_ * ty
            bz -= dt_ * tz
            bl = math.sqrt(bx * bx + by * by + bz * bz) + 1e-9
            gh = math.sqrt((x0 - hx2) ** 2 + (y0 - hy2) ** 2 + (pz - hz2) ** 2)
            if gh < 0.16:
                d_g = _fist(x0, y0, pz, hx2, hy2, hz2, tx, ty, tz, bx / bl, by / bl, bz / bl, F[i, F_GILT])
            else:
                d_g = gh - 0.10
            if d_g < d:
                d = d_g
                mat = M_GILT if F[i, F_GILT] > 0.0 else M_GLOVE
        # the torch: a shaft, and a head of wound, pitch-soaked cloth
        d_t = sd_capsule(x0, y0, pz, hx2 - 0.24 * tx, hy2 - 0.24 * ty, hz2 - 0.24 * tz,
                         hx2 + 0.36 * tx, hy2 + 0.36 * ty, hz2 + 0.36 * tz, 0.0165)
        if d_t < d:
            d = d_t
            mat = M_WOOD
        ax_ = x0 - hx2
        ay_ = y0 - hy2
        az_ = pz - hz2
        along = ax_ * tx + ay_ * ty + az_ * tz
        d_th = sd_capsule_r(x0, y0, pz, hx2 + 0.31 * tx, hy2 + 0.31 * ty, hz2 + 0.31 * tz,
                            hx2 + 0.47 * tx, hy2 + 0.47 * ty, hz2 + 0.47 * tz, 0.031, 0.025)
        d_th += 0.0035 * math.sin(along * 190.0) - 0.002 * (vnoise2(along * 80.0, (ax_ + ay_) * 80.0, 331) - 0.5)
        if d_th < d:
            d = d_th
            mat = M_TORCH
    if F[i, F_ARM2] > 0.5:
        sx, sy, sz = F[i, F_S2X], F[i, F_S2Y], F[i, F_S2Z]
        ex, ey, ez = F[i, F_E2X], F[i, F_E2Y], F[i, F_E2Z]
        hx2, hy2, hz2 = F[i, F_H2X], F[i, F_H2Y], F[i, F_H2Z]
        fx, fy, fz = F[i, F_D2X], F[i, F_D2Y], F[i, F_D2Z]
        d_u = sd_capsule_r(x0, y0, pz, sx, sy, sz, ex, ey, ez, 0.064, 0.058)
        cx_ = hx2 - 0.045 * fx
        cy_ = hy2 - 0.045 * fy
        cz_ = hz2 - 0.045 * fz
        d_f = sd_capsule_r(x0, y0, pz, ex, ey, ez, cx_, cy_, cz_, 0.056, 0.060)
        d_arm = min(d_u, d_f) + 0.003 * _crease(44.0 * (x0 - y0 + pz) + seed)
        d_prev = d
        d = smin(d, d_arm, 0.035)
        if d_arm < d_prev and mat != M_SHADOW and mat != M_SHAWL:
            mat = M_CLOTH
        gh = math.sqrt((x0 - hx2) ** 2 + (y0 - hy2) ** 2 + (pz - hz2) ** 2)
        if gh < 0.16:
            d_g = _open_hand(x0, y0, pz, hx2 - 0.02 * fx, hy2 - 0.02 * fy, hz2 - 0.02 * fz, fx, fy, fz, F[i, F_GRIP2])
        else:
            d_g = gh - 0.11
        if d_g < d:
            d = d_g
            mat = M_GLOVE
    return d, mat


@njit(**FM)
def trace_fig(ox, oy, oz, dx, dy, dz, t0, t1, F, i, pix):
    t = max(t0, 0.0)
    for it in range(110):
        d, m = sd_fig(ox + dx * t, oy + dy * t, oz + dz * t, F, i)
        eps = max(0.0006, t * pix * 0.35)
        if d < eps:
            return t, m
        t += d * 0.72
        if t > t1:
            break
    return -1.0, 0


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


# ============================================================ the detailed hand ===

@njit(**FM)
def sd_hand(px, py, pz, HD, J):
    """A gloved hand (thin leather) with articulated fingers. J[f, k] = joint k (0..3) of finger f (0 thumb,
    1 index .. 4 little) in hand-local (u,v,w): u wrist->knuckles, v toward the thumb, w the back of the hand.
    Returns (distance, material, glove seam coordinate)."""
    qx = px - HD[0]
    qy = py - HD[1]
    qz = pz - HD[2]
    s = HD[12]
    u = (qx * HD[3] + qy * HD[4] + qz * HD[5]) / s
    v = (qx * HD[6] + qy * HD[7] + qz * HD[8]) / s
    w = (qx * HD[9] + qy * HD[10] + qz * HD[11]) / s
    # palm: a rounded slab, arched across the back
    ea = abs(u - 0.050) - 0.044
    eb = abs(v + 0.002) - 0.036
    arch = 0.005 * (1.0 - (v / 0.04) ** 2)
    ec = abs(w - 0.001 - arch) - 0.011
    rr = 0.011
    mx = max(ea + rr, 0.0)
    my = max(eb + rr, 0.0)
    mz = max(ec + rr, 0.0)
    d = math.sqrt(mx * mx + my * my + mz * mz) - rr + min(max(ea, max(eb, ec)), 0.0)
    # the heel of the hand and the thumb's mound
    d = smin(d, sd_ellipsoid(u - 0.028, v - 0.024, w + 0.010, 0.030, 0.020, 0.014), 0.012)
    mat = M_GLOVE
    # fingers
    rads = (0.0112, 0.0092, 0.0094, 0.0090, 0.0080)
    for f in range(5):
        r0 = rads[f]
        for k in range(3):
            ra = r0 * (1.0 - 0.07 * k)
            rb = r0 * (1.0 - 0.07 * (k + 1))
            dd = sd_capsule_r(u, v, w, J[f, k, 0], J[f, k, 1], J[f, k, 2], J[f, k + 1, 0], J[f, k + 1, 1],
                              J[f, k + 1, 2], ra, rb)
            d = smin(d, dd, 0.0045 if k == 0 else 0.0025)
    # knuckles on the back
    for f in range(1, 5):
        d = smin(d, math.sqrt((u - J[f, 0, 0]) ** 2 + (v - J[f, 0, 1]) ** 2 + (w - J[f, 0, 2] - 0.004) ** 2) - 0.0095,
                 0.006)
    # wrist, cuff and sleeve
    L = HD[13]
    dw = sd_capsule_r(u, v, w, 0.004, 0.0, 0.0, -L, 0.0, 0.0, 0.026, 0.034)
    d = smin(d, dw, 0.010)
    dcf = sd_capsule_r(u, v, w, -0.012, 0.0, 0.0, -0.030, 0.0, 0.0, 0.0285, 0.0300)
    if dcf < d:
        d = dcf
    if HD[15] > 0.5:
        dsl = sd_capsule_r(u, v, w, -0.040, 0.0, 0.0, -L - 0.2, 0.0, 0.0, 0.040, 0.052)
        dsl += 0.003 * _crease(60.0 * u + 3.0 * math.atan2(w, v))
        if dsl < d:
            d = dsl
            mat = M_CLOTH
    if HD[14] > 0.0 and mat == M_GLOVE and w > -0.004:
        mat = M_GILT
    return d * s, mat


@njit(**FM)
def trace_hand(ox, oy, oz, dx, dy, dz, t0, t1, HD, J, pix):
    t = max(t0, 0.0)
    for it in range(120):
        d, m = sd_hand(ox + dx * t, oy + dy * t, oz + dz * t, HD, J)
        eps = max(0.00015, t * pix * 0.3)
        if d < eps:
            return t, m
        t += d * 0.8
        if t > t1:
            break
    return -1.0, 0


@njit(**FM)
def normal_hand(px, py, pz, HD, J, h):
    k1, _ = sd_hand(px + h, py - h, pz - h, HD, J)
    k2, _ = sd_hand(px - h, py - h, pz + h, HD, J)
    k3, _ = sd_hand(px - h, py + h, pz - h, HD, J)
    k4, _ = sd_hand(px + h, py + h, pz + h, HD, J)
    nx = k1 - k2 - k3 + k4
    ny = -k1 - k2 + k3 + k4
    nz = -k1 + k2 - k3 + k4
    l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / l, ny / l, nz / l


# ================================================================== the Ring ===

@njit(**FM)
def ring_local(px, py, pz, RP):
    """Ring-local cylindrical coords: (rho, z along the axis, x, y in the ring plane)."""
    qx = px - RP[1]
    qy = py - RP[2]
    qz = pz - RP[3]
    ax, ay, az = RP[4], RP[5], RP[6]
    zz = qx * ax + qy * ay + qz * az
    rx = qx - zz * ax
    ry = qy - zz * ay
    rz = qz - zz * az
    return math.sqrt(rx * rx + ry * ry + rz * rz), zz, rx, ry, rz


@njit(**FM)
def sd_ring(px, py, pz, RP):
    rho, zz, _, _, _ = ring_local(px, py, pz, RP)
    a = 0.5 * R_THICK
    b = 0.5 * R_WIDTH
    x = abs(rho - R_MID) / a
    y = abs(zz) / b
    n = R_SQ
    k = (x ** n + y ** n) ** (1.0 / n)
    return (k - 1.0) * min(a, b) * 0.9


@njit(**FM)
def trace_ring(ox, oy, oz, dx, dy, dz, RP, tmax):
    cx = ox - RP[1]
    cy = oy - RP[2]
    cz = oz - RP[3]
    rb = R_IN + R_THICK + 0.002
    b = cx * dx + cy * dy + cz * dz
    c = cx * cx + cy * cy + cz * cz - rb * rb
    disc = b * b - c
    if disc <= 0.0:
        return -1.0
    sq = math.sqrt(disc)
    t = max(-b - sq, 0.0)
    t1 = min(-b + sq, tmax)
    for it in range(90):
        if t > t1:
            break
        d = sd_ring(ox + dx * t, oy + dy * t, oz + dz * t, RP)
        if d < 0.00006:
            return t
        t += max(d * 0.9, 0.00003)
    return -1.0


# ======================================================= analytic occluders ===
# OC rows: ax,ay,az, bx,by,bz, radius, owner (figure index, or -1)
OC_N = 8


@njit(inline='always', **FM)
def _seg_seg(p1x, p1y, p1z, q1x, q1y, q1z, p2x, p2y, p2z, q2x, q2y, q2z):
    """Closest points between segments p1q1 and p2q2: returns (s, t, dist)."""
    d1x = q1x - p1x
    d1y = q1y - p1y
    d1z = q1z - p1z
    d2x = q2x - p2x
    d2y = q2y - p2y
    d2z = q2z - p2z
    rx = p1x - p2x
    ry = p1y - p2y
    rz = p1z - p2z
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    c = d1x * rx + d1y * ry + d1z * rz
    b = d1x * d2x + d1y * d2y + d1z * d2z
    den = a * e - b * b
    if den > 1e-12:
        s = clamp((b * f - c * e) / den, 0.0, 1.0)
    else:
        s = 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0.0:
        t = 0.0
        s = clamp(-c / a, 0.0, 1.0) if a > 1e-12 else 0.0
    elif t > 1.0:
        t = 1.0
        s = clamp((b - c) / a, 0.0, 1.0) if a > 1e-12 else 0.0
    cx = p1x + d1x * s - (p2x + d2x * t)
    cy = p1y + d1y * s - (p2y + d2y * t)
    cz = p1z + d1z * s - (p2z + d2z * t)
    return s, t, math.sqrt(cx * cx + cy * cy + cz * cz)


@njit(**FM)
def soft_vis(px, py, pz, lx, ly, lz, rl, OC, noc, skip):
    """Soft visibility of a spherical light (centre l, radius rl) from p against capsule occluders."""
    vx = lx - px
    vy = ly - py
    vz = lz - pz
    D = math.sqrt(vx * vx + vy * vy + vz * vz)
    if D < 1e-5:
        return 1.0
    vis = 1.0
    for k in range(noc):
        if OC[k, 7] == skip and skip >= 0.0:
            continue
        # quick reject: the capsule's bounding sphere against the segment
        mx = 0.5 * (OC[k, 0] + OC[k, 3])
        my = 0.5 * (OC[k, 1] + OC[k, 4])
        mz = 0.5 * (OC[k, 2] + OC[k, 5])
        hx = OC[k, 3] - OC[k, 0]
        hy = OC[k, 4] - OC[k, 1]
        hz = OC[k, 5] - OC[k, 2]
        rbound = 0.5 * math.sqrt(hx * hx + hy * hy + hz * hz) + OC[k, 6] + rl
        wx = mx - px
        wy = my - py
        wz = mz - pz
        tp = clamp((wx * vx + wy * vy + wz * vz) / (D * D), 0.0, 1.0)
        ex = wx - vx * tp
        ey = wy - vy * tp
        ez = wz - vz * tp
        if ex * ex + ey * ey + ez * ez > rbound * rbound:
            continue
        s, t, dist = _seg_seg(px, py, pz, lx, ly, lz, OC[k, 0], OC[k, 1], OC[k, 2], OC[k, 3], OC[k, 4], OC[k, 5])
        if s < 0.002:
            continue
        h = dist - OC[k, 6]
        w = rl * s + 0.012
        v = sstep(-w, w, h)
        if v < vis:
            vis = v
            if vis < 0.002:
                return 0.0
    return vis


@njit(**FM)
def soft_vis_dir(px, py, pz, dx, dy, dz, soft, OC, noc, skip):
    """Directional light (unit dir toward the light) soft shadow against the capsules."""
    L = 40.0
    return soft_vis(px, py, pz, px + dx * L, py + dy * L, pz + dz * L, soft * L, OC, noc, skip)
