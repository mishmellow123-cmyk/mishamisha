"""ACCORD v3 geometry (cut C, the council): the flat stone, the cold hearth, the emissaries and her,
gloved hands, torches, the real-size Ring, and analytic soft-shadow occluders.

World: metres, z up, origin = the centre of the flat stone on the ground. Figures use a local frame:
x toward where they face, y to their left, z up from the ground.
"""
import math

from numba import njit

import numpy as np

from nbcore import FM, clamp, sstep, smin, sd_ellipsoid, sd_capsule, sd_capsule_r, vnoise2, fbm2, hash2i
from geom import _crease, _hood, sd_stone, ray_aabb  # noqa: F401  (v2 hood shells and standing stones)

# ------------------------------------------------------------- the centre ---
STONE_R = 0.37          # the flat stone: mean radius
STONE_TOP = 0.30        # its top (z)
STONE_BOT = -0.12
ASH_R0, ASH_R1 = 0.40, 0.84   # the ash bed of the hearth (annulus round the stone)
KERB_R = 0.92           # the ring of blackened kerb stones
NKERB = 17
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
HD_N = 24          # + [16:19] elbow, [19:22] shoulder (hand-local, hand units), [22] spare, [23] arm mode
# the Ring: RP[0] on, RP[1:4] centre, RP[4:7] axis, RP[7] glow, RP[8] hide-inside-fist
R_IN, R_THICK, R_WIDTH, R_SQ = 0.0094, 0.0023, 0.0052, 2.8
R_MID = R_IN + 0.5 * R_THICK
RP_N = 10


# =============================================================== the centre ===

# the flat stone's outline: an irregular heptagon of split faces (angle, distance), corners rounded a little
FLAT_A = np.array([0.20, 1.02, 1.63, 2.61, 3.49, 4.31, 5.33])
FLAT_D = np.array([0.338, 0.300, 0.352, 0.322, 0.346, 0.296, 0.334])
NFLAT = 7
FLAT_RC = 0.070          # corner rounding (the outline is the heptagon shrunk by this, then grown back round)
# chips broken out of the rim: (angle, depth into the stone, radius)
CHIP = np.array([[0.72, 0.012, 0.050], [2.45, 0.006, 0.034], [3.30, 0.016, 0.060], [4.95, 0.008, 0.042],
                 [5.85, 0.004, 0.030], [1.62, 0.002, 0.028]])


@njit(**FM)
def flat_radius(th):
    R = 1e9
    for k in range(NFLAT):
        c = math.cos(th - FLAT_A[k])
        if c > 1e-3:
            R = min(R, FLAT_D[k] / c)
    return R


@njit(**FM)
def _flat2d(px, py):
    """Signed distance to the slab's outline in plan (negative inside): an irregular polygon of split faces,
    its corners weathered round, its faces a little wavy."""
    d = -1e9
    for k in range(NFLAT):
        d = max(d, px * math.cos(FLAT_A[k]) + py * math.sin(FLAT_A[k]) - (FLAT_D[k] - FLAT_RC))
    th = math.atan2(py, px)
    return (d - FLAT_RC + 0.008 * math.sin(5.0 * th + 1.3) + 0.005 * math.sin(11.0 * th + 0.4)
            + 0.006 * (vnoise2(th * 4.0, 0.3, 304) - 0.5))


@njit(**FM)
def flat_top(r, th):
    """The top of the slab: nearly flat, gently uneven, bevelled down at the rim by weather."""
    px = r * math.cos(th)
    py = r * math.sin(th)
    d2 = _flat2d(px, py)
    # an old frost crack across the top, a shallow groove
    cr = abs(0.62 * px + 0.78 * py - 0.05 + 0.03 * math.sin(9.0 * px - 4.0 * py))
    return (STONE_TOP - 0.024 * sstep(-0.07, 0.0, d2) + 0.012 * px - 0.006 * py
            + 0.010 * math.sin(2.3 * px + 1.1) * math.sin(3.1 * py) + 0.007 * (vnoise2(px * 6.0, py * 6.0, 303) - 0.5)
            - 0.004 * sstep(0.006, 0.0, cr))


@njit(**FM)
def sd_flat(px, py, pz):
    r = math.sqrt(px * px + py * py)
    th = math.atan2(py, px)
    zc = 0.5 * (STONE_TOP + STONE_BOT)
    hh = 0.5 * (STONE_TOP - STONE_BOT)
    zz = (pz - zc) / hh
    # the split faces lean out a little toward the ground, and bulge at mid-height
    d2 = _flat2d(px, py) - 0.012 * max(1.0 - zz * zz, 0.0) + 0.010 * (zz - 1.0) * 0.5
    top = flat_top(min(r, 0.45), th)
    e = 0.022
    qx = d2 + e
    qz = max(pz - (top - e), (STONE_BOT + e) - pz)
    mx = max(qx, 0.0)
    mz = max(qz, 0.0)
    d = math.sqrt(mx * mx + mz * mz) + min(max(qx, qz), 0.0) - e
    # chips broken out of the rim
    for k in range(CHIP.shape[0]):
        a = CHIP[k, 0]
        Rr = flat_radius(a)
        cx = (Rr - CHIP[k, 1] + CHIP[k, 2] * 0.55) * math.cos(a)
        cy = (Rr - CHIP[k, 1] + CHIP[k, 2] * 0.55) * math.sin(a)
        cz = STONE_TOP + CHIP[k, 2] * 0.35
        dc = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2) - CHIP[k, 2]
        d = max(d, -dc)
    # weathering and grain
    # weathering on the split faces only (never keyed on the angle over the top, which would crease it radially)
    d += 0.0045 * (vnoise2(px * 11.0 + py * 7.0, pz * 14.0, 301) - 0.5) * sstep(-0.05, 0.0, _flat2d(px, py))
    return d


@njit(**FM)
def sd_charcoal(px, py, pz):
    """Charcoal broken over the ash: one lump or none per 4.5 cm cell, bigger and denser toward the stone."""
    cs = 0.045
    ix = int(math.floor(px / cs))
    iy = int(math.floor(py / cs))
    d = 1e9
    for ddx in range(-1, 2):
        for ddy in range(-1, 2):
            cx = ix + ddx
            cy = iy + ddy
            ox = (cx + 0.15 + 0.7 * hash2i(cx, cy, 502)) * cs
            oy = (cy + 0.15 + 0.7 * hash2i(cx, cy, 503)) * cs
            orr = math.sqrt(ox * ox + oy * oy)
            if orr < ASH_R0 + 0.015 or orr > ASH_R1 - 0.01:
                continue
            pd = 0.10 + 0.22 * sstep(ASH_R1, ASH_R0 + 0.05, orr)
            if hash2i(cx, cy, 501) > pd:
                continue
            h4 = hash2i(cx, cy, 504)
            rx = 0.004 + 0.013 * h4 * h4
            ry = rx * (0.45 + 0.40 * hash2i(cx, cy, 505))
            rz = rx * (0.22 + 0.22 * hash2i(cx, cy, 506))
            yaw = 6.2832 * hash2i(cx, cy, 507)
            c = math.cos(yaw)
            sn = math.sin(yaw)
            lx = px - ox
            ly = py - oy
            x = lx * c + ly * sn
            y = -lx * sn + ly * c
            z = pz - rz * 0.30
            if x * x + y * y > (rx + 0.01) ** 2 * 4.0:
                continue
            dd = sd_ellipsoid(x, y, z, rx, ry, rz)
            # broken, faceted
            dd += 0.30 * rx * (vnoise2(x / rx * 2.6 + 3.0 * h4, y / rx * 2.6 + z / rx * 1.5, 508) - 0.5)
            if dd < d:
                d = dd
    return d


@njit(**FM)
def sd_hearth(px, py, pz, KB, LG, nlog, CH, nch):
    """The cold hearth round the flat stone: kerb stones (KB: x,y,z,ax,ay,az,yaw_c,yaw_s,seed), laid logs and
    sticks (LG: capsules ax,ay,az,bx,by,bz,ra,rb,seed,kind), charcoal (cells) and a few big chunks (CH).
    Returns (distance, part id)."""
    r = math.sqrt(px * px + py * py)
    # a safe bound for everything in the hearth: all of it lies below 0.18 m, between r 0.36 and KERB_R + 0.22
    # (never return "infinity" where no part is evaluated: the march would leap out of the hearth)
    zb = pz - 0.18
    rb = max(0.36 - r, r - (KERB_R + 0.22))
    if zb > 0.012 or rb > 0.012:
        # >= 1.2 cm: never inside a far camera's hit tolerance (a false hit on the bound shades black)
        return max(zb, rb), 0
    d = 1e9
    part = 0
    nk = KB.shape[0]
    if r > KERB_R - 0.22:
        for k in range(nk):
            lx = px - KB[k, 0]
            ly = py - KB[k, 1]
            if lx * lx + ly * ly > (KB[k, 3] + 0.15) ** 2:
                dd = math.sqrt(lx * lx + ly * ly) - KB[k, 3] - 0.03
            else:
                lz = pz - KB[k, 2]
                c = KB[k, 6]
                sn = KB[k, 7]
                x = lx * c + ly * sn
                y = -lx * sn + ly * c
                dd = sd_ellipsoid(x, y, lz, KB[k, 3], KB[k, 4], KB[k, 5])
                # rough field stones, a flat-ish top
                dd += 0.022 * (vnoise2(x * 9.0 + KB[k, 8], y * 9.0 + lz * 7.0, 311) - 0.5) \
                    + 0.008 * (vnoise2(x * 26.0, y * 26.0 + lz * 20.0 + KB[k, 8], 316) - 0.5)
            if dd < d:
                d = dd
                part = H_KERB
    if r < ASH_R1 + 0.10 and pz < HEARTH_ZMAX:
        for k in range(nlog):
            dd = sd_capsule_r(px, py, pz, LG[k, 0], LG[k, 1], LG[k, 2], LG[k, 3], LG[k, 4], LG[k, 5],
                              LG[k, 6], LG[k, 7])
            if dd < d + 0.02:
                # bark ridges along the log, a split face on the big ones
                ax = LG[k, 3] - LG[k, 0]
                ay = LG[k, 4] - LG[k, 1]
                al = math.sqrt(ax * ax + ay * ay) + 1e-9
                along = ((px - LG[k, 0]) * ax + (py - LG[k, 1]) * ay) / al
                side = (-(px - LG[k, 0]) * ay + (py - LG[k, 1]) * ax) / al
                ang = math.atan2(pz - 0.5 * (LG[k, 2] + LG[k, 5]), side)
                dd += 0.0030 * (vnoise2(along * 12.0 + LG[k, 8], ang * 2.0, 312) - 0.5)
            if dd < d:
                d = dd
                part = H_LOG
        if pz >= 0.06:
            # the charcoal lies below 0.03 m: bound the step so it can never be leapt through
            if pz - 0.03 < d:
                d = pz - 0.03
                part = 0
        elif r > ASH_R0 - 0.02:
            dd = sd_charcoal(px, py, pz)
            for k in range(nch):
                dc = sd_ellipsoid(px - CH[k, 0], py - CH[k, 1], pz - CH[k, 2], CH[k, 3], CH[k, 4], CH[k, 5])
                dc += 0.005 * (vnoise2(px * 70.0 + CH[k, 6], py * 70.0, 313) - 0.5)
                dd = min(dd, dc)
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
    # the wrapped fingers: a rounded roll round the shaft, offset toward the back of the hand
    ea = abs(a) - 0.034
    eb = abs(b - 0.008) - 0.024
    ec = abs(c) - 0.021
    rr = 0.019
    mx = max(ea + rr, 0.0)
    my = max(eb + rr, 0.0)
    mz = max(ec + rr, 0.0)
    d = math.sqrt(mx * mx + my * my + mz * mz) - rr + min(max(ea, max(eb, ec)), 0.0)
    # four finger rolls across the front and knuckles on the back
    g = math.cos(a / 0.0180 * math.pi)
    d += 0.0020 * g * sstep(0.0, 0.02, c)
    d -= 0.0022 * max(g, 0.0) * sstep(0.015, 0.03, b)
    # the thumb, wrapped round the shaft on the other side
    dt = sd_capsule_r(a, b, c, 0.034, -0.004, -0.021, 0.026, -0.016, 0.018, 0.0100, 0.0090)
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
def _hood3(hx, y, hz, typ, seed):
    """A wool hood seen from above, in the (bowed) head frame (hx forward, y left, hz up from the head's centre):
    close over the skull (about half the shoulders' width), a centre seam from brow to nape, drawn to a peak
    behind (a soft point, a liripipe, or a tall capuchin point by type); the face opening and the head inside it
    stay in shadow. Returns (distance, material, in_cavity)."""
    if typ == 3:      # the hood thrown back: a bare dark head of hair, the hood's folds lying on the shoulders
        d_h = sd_ellipsoid(hx - 0.008, y, hz - 0.012, 0.098, 0.084, 0.112)
        d_h += 0.004 * _crease(math.atan2(y, hx) * 9.0 + hz * 30.0 + seed)
        if hx > 0.05 and hz < 0.03:
            return d_h, M_SHADOW, False
        return d_h, M_HAIR, False
    ao_, bo_, co_ = 0.130, 0.110, 0.138
    if typ == 5:
        ao_, bo_ = 0.126, 0.106
    d_out = sd_ellipsoid(hx + 0.010, y, hz - 0.004, ao_, bo_, co_)
    # the brow of the hood over the face: a soft roll, not a visor
    d_b = sd_ellipsoid(hx - 0.032, y, hz - 0.028, 0.118, bo_ * 0.93, 0.092)
    d_out = smin(d_out, d_b, 0.025)
    # the peak behind
    if typ == 1:
        d_p = sd_capsule_r(hx, y, hz, -0.050, 0.0, 0.050, -0.128, 0.0, 0.020, 0.062, 0.022)
        d_out = smin(d_out, d_p, 0.045)
    elif typ == 0:    # a liripipe: the point drawn out into a tail that falls down the back
        d_p = sd_capsule_r(hx, y, hz, -0.050, 0.0, 0.055, -0.150, 0.004, -0.010, 0.060, 0.026)
        d_p = min(d_p, sd_capsule_r(hx, y, hz, -0.150, 0.004, -0.010, -0.190, 0.012, -0.300, 0.026, 0.018))
        d_out = smin(d_out, d_p, 0.040)
    elif typ == 5:    # a capuchin: a tall stiff point, up and back
        d_p = sd_capsule_r(hx, y, hz, -0.035, 0.0, 0.070, -0.110, 0.0, 0.215, 0.060, 0.007)
        d_out = smin(d_out, d_p, 0.030)
    # the centre seam: a low ridge from the brow over the crown to the nape
    d_out -= 0.0045 * sstep(0.022, 0.0, abs(y)) * sstep(-0.08, 0.03, hz)
    # soft creases running back from the opening, and a few falling from the crown
    ph = math.atan2(hz - 0.02, y) * 5.0 + seed
    d_out += 0.0055 * _crease(ph) * sstep(0.08, -0.10, hx)
    d_out += 0.0035 * _crease(math.atan2(y, hx + 0.03) * 7.0 + 1.7 * seed) * sstep(0.02, -0.10, hz)
    d_in = sd_ellipsoid(hx - 0.004, y, hz - 0.010, ao_ - 0.024, bo_ - 0.022, co_ - 0.022)
    d_shell = max(d_out, -d_in)
    oy = 0.070
    oz = 0.095
    e = math.sqrt((y / oy) ** 2 + ((hz + 0.030) / oz) ** 2)
    d_cut = max(0.048 - hx, (e - 1.0) * min(oy, oz))
    d_shell = max(d_shell, -d_cut)
    d_head = sd_ellipsoid(hx - 0.006, y, hz + 0.004, 0.092, 0.077, 0.106)
    if d_head < d_shell:
        return d_head, M_SHADOW, True
    cav = -d_in > d_out and -d_in > -d_cut
    return d_shell, (M_SHADOW if cav else M_CLOTH2), cav


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
            # a woven wool shawl: it clings to the shoulders and upper arms, its point hangs down her back,
            # its two ends cross low at her breast; a few soft, irregular folds (never a ruff)
            ac = (0.160 + 0.020 * u) * ws
            if x < 0.0:
                ac = (0.172 + 0.026 * u) * ws
            bc = (0.224 + 0.030 * u) * ws
            phc = math.atan2(y / bc, x / ac)
            fc = 1.0 + (0.008 + 0.030 * u) * (0.8 * math.sin(3.0 * phc + seed) + 0.5 * math.sin(5.3 * phc + 2.1 * seed + 5.0 * u)
                                               + 0.35 * _crease(8.0 * phc + seed))
            qc = math.sqrt((x / ac) ** 2 + (y / bc) ** 2) / fc
            back = sstep(0.0, -0.10, x) * sstep(0.55, 0.0, abs(y) / bc)
            front = sstep(0.0, 0.10, x) * sstep(0.15, 0.55, abs(y) / bc)
            zt = zsh - drop * (0.75 + 0.85 * back + 0.45 * front)
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
    far = hb > 0.50
    if far:
        d_h, hm, cav = hb - 0.42, M_CLOTH2, False
    else:
        d_h, hm, cav = _hood3(hx, y, hz, typ, seed)
    if typ != 3 and not far:
        # the hood's cape: its cloth falls from round the face onto the shoulders as a sloping cone with folds
        zb = zsh - 0.10 * hs
        zt_ = zsh + 0.15 * hs
        v = clamp((z - zb) / (zt_ - zb), 0.0, 1.0)
        rc = 0.200 * ws * (1.0 - v) + 0.100 * v
        ang = math.atan2(y, x + 0.01)
        rq = math.sqrt(((x + 0.012) / 0.86) ** 2 + y * y)
        fcp = 1.0 + (0.018 + 0.060 * (1.0 - v)) * _crease(7.0 * ang + 1.3 * seed + 2.0 * v)
        d_cw = max((rq - rc * fcp) * 0.8, max(zb - z, z - zt_))
        d_h = smin(d_h, d_cw, 0.05)
        if d_cw < d_h + 0.006 and not cav:
            hm = M_CLOTH2
    if F[i, F_SHAWL] > 0.5 and hm == M_CLOTH2:
        hm = M_CLOTH          # her hood is her dark wool; only the shawl is red
        if typ != 3 and not far and d_cw < d_h + 0.004 and not cav and z < zsh + 0.10 * hs:
            hm = M_SHAWL      # the shawl is drawn up round her neck, under the hood
    if typ == 3:
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
        if F[i, F_HEROHAND] > 0.5:
            # the detailed glove carries its own cuff and sleeve: the cloak's sleeve stops well short of the grip
            cuff = max(kl - 0.170, 0.02) / kl
            d_f = sd_capsule_r(x0, y0, pz, ex, ey, ez, ex + kx * cuff, ey + ky * cuff, ez + kz * cuff, 0.064, 0.060)
        else:
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
        d_th = sd_capsule_r(x0, y0, pz, hx2 + 0.33 * tx, hy2 + 0.33 * ty, hz2 + 0.33 * tz,
                            hx2 + 0.45 * tx, hy2 + 0.45 * ty, hz2 + 0.45 * tz, 0.034, 0.029)
        # rag wound round it: shallow, helical, irregular
        rx_ = ax_ - along * tx
        ry_ = ay_ - along * ty
        wang = math.atan2(ry_, rx_)
        d_th += 0.0016 * math.sin(along * 130.0 + 2.0 * wang + 3.0 * vnoise2(along * 20.0, wang * 2.0, 332)) \
            - 0.003 * (vnoise2(along * 70.0, wang * 6.0, 331) - 0.5)
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

@njit(inline='always', **FM)
def _seg_dist(u, v, w, ax, ay, az, bx, by, bz):
    """Distance from (u,v,w) to segment a-b and the parameter along it (0..1)."""
    ex = bx - ax
    ey = by - ay
    ez = bz - az
    L2 = ex * ex + ey * ey + ez * ez + 1e-12
    h = ((u - ax) * ex + (v - ay) * ey + (w - az) * ez) / L2
    h = min(max(h, 0.0), 1.0)
    dx = u - ax - ex * h
    dy = v - ay - ey * h
    dz = w - az - ez * h
    return math.sqrt(dx * dx + dy * dy + dz * dz), h


@njit(**FM)
def gilt_crust(u, v, w, J):
    """The gold that ran over the back of the grasping hand and set (0..1, the crust's thickness), in hand
    units: a lobed sheet poured over the back of the hand, run forward over the knuckles in tongues along the
    backs of the first three fingers; the leather shows round it and between the tongues."""
    if w < -0.030:
        return 0.0
    du = (u - 0.060) / 0.034
    dv = (v + 0.006) / 0.023
    rr = math.sqrt(du * du + dv * dv)
    lob = 0.55 * (vnoise2(u * 40.0 + 3.1, v * 40.0, 601) - 0.5) + 0.20 * (vnoise2(u * 110.0, v * 110.0 + 1.7, 602) - 0.5)
    sheet = sstep(0.0, 0.10, 1.0 - rr + lob) * sstep(-0.012, 0.004, w)
    # tongues over the knuckles, along the backs of the proximal phalanges (index long, middle, ring short)
    tg = 0.0
    for f in range(1, 3):
        ax = J[f, 0, 0]
        ay = J[f, 0, 1]
        az = J[f, 0, 2]
        ex = J[f, 1, 0] - ax
        ey = J[f, 1, 1] - ay
        ez = J[f, 1, 2] - az
        el = math.sqrt(ex * ex + ey * ey + ez * ez) + 1e-9
        ex /= el
        ey /= el
        ez /= el
        # the dorsal side of the segment (the curl turns it with the finger)
        nx = -ez
        nz = ex
        ln = (0.85, 0.45, 0.0)[f - 1] * el
        ox = ax + nx * 0.0055 - ex * 0.012
        oz = az + nz * 0.0055 - ez * 0.012
        dd, hh = _seg_dist(u, v, w, ox, ay - ey * 0.012, oz, ax + ex * ln + nx * 0.0055, ay + ey * ln, az + ez * ln + nz * 0.0055)
        rad = 0.0078 * (1.0 - 0.45 * hh) + 0.0015 * (vnoise2(hh * 9.0 + f, 0.5, 603) - 0.5)
        tg = max(tg, sstep(0.0012, -0.0006, dd - rad))
    return max(sheet, tg)


@njit(**FM)
def sd_hand(px, py, pz, HD, J):
    """A gloved hand (thin leather) with articulated fingers. J[f, k] = joint k (0..3) of finger f (0 thumb,
    1 index .. 4 little) in hand-local (u,v,w): u wrist->knuckles, v toward the thumb, w the back of the hand.
    HD[23] = 1: a real arm in a wool sleeve (wrist -> elbow HD[16:19] -> shoulder HD[19:22]); else a short
    straight sleeve along -u. HD[14] > 0: the gilded glove (a raised gold crust, material M_GILT).
    Returns (distance, material)."""
    qx = px - HD[0]
    qy = py - HD[1]
    qz = pz - HD[2]
    s = HD[12]
    u = (qx * HD[3] + qy * HD[4] + qz * HD[5]) / s
    v = (qx * HD[6] + qy * HD[7] + qz * HD[8]) / s
    w = (qx * HD[9] + qy * HD[10] + qz * HD[11]) / s
    arm = HD[23] > 0.5
    # the sleeve first: far from the hand only the sleeve matters (cheap)
    d_sl = 1e9
    if HD[15] > 0.5:
        if arm:
            ex = HD[16]
            ey = HD[17]
            ez = HD[18]
            el = math.sqrt(ex * ex + ey * ey + ez * ez) + 1e-9
            fx = ex / el
            fy = ey / el
            fz = ez / el
            # the forearm in a wool sleeve that opens a little over the glove's cuff and widens to the elbow
            ax_ = fx * 0.045
            ay_ = fy * 0.045
            az_ = fz * 0.045
            d1, h1 = _seg_dist(u, v, w, ax_, ay_, az_, ex, ey, ez)
            r1 = 0.047 + 0.020 * h1
            # folds: rings of soft creases pushed up the forearm, and a few long ones
            lx = u - ax_ - (ex - ax_) * h1
            ly = v - ay_ - (ey - ay_) * h1
            lz = w - az_ - (ez - az_) * h1
            ph = math.atan2(ly * fz - lz * fy, lx)
            fo = 0.0045 * _crease(h1 * 34.0 + 1.6 * math.sin(ph * 2.0 + h1 * 5.0)) + 0.0030 * _crease(ph * 5.0 + h1 * 3.0)
            d_sl = d1 - r1 - fo
            # the upper arm up into her cloak at the shoulder
            d2, h2 = _seg_dist(u, v, w, ex, ey, ez, HD[19], HD[20], HD[21])
            d_sl = smin(d_sl, d2 - (0.064 + 0.012 * h2) - 0.004 * _crease(h2 * 22.0 + ph * 3.0), 0.03)
        else:
            L = HD[13]
            d_sl = sd_capsule_r(u, v, w, -0.040, 0.0, 0.0, -L - 0.2, 0.0, 0.0, 0.040, 0.052)
            d_sl += 0.003 * _crease(60.0 * u + 3.0 * math.atan2(w, v))
    # bound: the hand itself (fingers straight: <= 0.19) and the cuff lie within 0.21 of the wrist
    rh = math.sqrt(u * u + v * v + w * w)
    if rh > 0.235:
        return min(d_sl, rh - 0.215) * s, M_CLOTH
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
    # knuckles on the back, and the tendons fanning to them under the thin leather
    for f in range(1, 5):
        d = smin(d, math.sqrt((u - J[f, 0, 0]) ** 2 + (v - J[f, 0, 1]) ** 2 + (w - J[f, 0, 2] - 0.004) ** 2) - 0.0095,
                 0.006)
    # wrist and the glove's cuff
    if arm:
        el = math.sqrt(HD[16] ** 2 + HD[17] ** 2 + HD[18] ** 2) + 1e-9
        fx = HD[16] / el
        fy = HD[17] / el
        fz = HD[18] / el
        dw = sd_capsule_r(u, v, w, 0.004, 0.0, 0.0, fx * 0.075, fy * 0.075, fz * 0.075, 0.026, 0.031)
    else:
        L = HD[13]
        dw = sd_capsule_r(u, v, w, 0.004, 0.0, 0.0, -L, 0.0, 0.0, 0.026, 0.034)
    d = smin(d, dw, 0.010)
    if not arm:
        dcf = sd_capsule_r(u, v, w, -0.012, 0.0, 0.0, -0.030, 0.0, 0.0, 0.0285, 0.0300)
        if dcf < d:
            d = dcf
    # the gilded glove: a raised crust of set gold, lumpy, with a bead at its edge where it stopped running
    if HD[14] > 0.0 and d < 0.006:
        c = gilt_crust(u, v, w, J) * HD[14]
        if c > 0.0:
            bump = 0.35 + 0.65 * vnoise2(u * 230.0, v * 230.0 + w * 170.0, 604) + 0.35 * (vnoise2(u * 700.0, v * 700.0, 605) - 0.5)
            lip = sstep(0.15, 0.45, c) * sstep(0.95, 0.55, c)
            d -= 0.0016 * c * bump + 0.0007 * lip
            if c > 0.35:
                mat = M_GILT
    if d_sl < d:
        return d_sl * s, M_CLOTH
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
