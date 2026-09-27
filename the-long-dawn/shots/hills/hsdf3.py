"""HEROINE v3 (BIBLE_V3 H2, H3, H5): the close-up renderer for her hands and the things they hold.

A fork of heroine_sdf.py's sphere tracer. FIRST BEACON (H1) never imports this file, so the accepted v2b path is
untouched. The rows, groups and Builder are heroine.py's; the primitives 0-5 are evaluated by heroine_sdf.prim_sd
itself. Added here:

* T_BAND: the Ring's band, a superellipse-section torus with tolkien.py's section (exponent 3.2, TB:HB 0.4:0.95)
  so it is the same object the EMBERS-C lane forges;
* materials 11+: GOLD (a polished metal lit by the point lights AND by an environment map, which is what the band
  reflects: the night, the snow, the fire, the vision), CLAY, EMBER and COAL (emissive cracks under a skin of ash
  that spreads as their `life` falls), SNOW (wrapped, with glints), ICE (a dark glossy glaze), IRON, ASH;
* the Ring's inscription: embers/inscription.py strips (seeds 7 and 23, the E5 Ring's own lines, outside and in),
  engraved in the gold and, when awake, burning in it (XP[17]);
* any material may reflect the environment map (material column 17).

XP (float64[64]) carries the per-frame extras: 0 time; 1 ring on; 2-4 ring centre; 5-13 world->ring rows (row 1 =
the ring's axis); 14 R, 15 TB, 16 HB, 17 letters' glow, 18 engraving depth; 19 ember life, 20 ember gain,
24 ember crack scale (1/m), 25 env map on, 26 env gain, 28 coal gain, 29 coal crack scale, 30 coal life,
40-51 the inscription strip layout (see ring_xp), 52 inscription levels.
"""
import math

import numba as nb
import numpy as np

import heroine_sdf as hs
from core import cam_ray, perlin3

T_BAND = 6
PEXP = 3.2

# materials 0-10 are heroine_sdf's; the v3 props follow
(M_SKIN, M_EYE, M_HAIR, M_COAT, M_SCARF, M_CLOTH, M_BOOT, M_STEEL, M_FLINT, M_NAIL, M_KNIT) = range(11)
M_GOLD, M_CLAY, M_EMBER, M_SNOW, M_ICE, M_COAL, M_IRON, M_ASH, M_GLOVE = range(11, 20)
NMAT3 = 20
NCOL = 24
# material columns: 0-15 as heroine_sdf (albedo, rough, F0, wrap, sheen, trans, rim, amb, metal, tex type,
# tex scale, tex amp, bump, spare); 16 emission type (1 = ember/coal cracks); 17 env reflection gain;
# 18 glint density; 19 glint gain


def material_table3():
    import heroine as hero
    M = np.zeros((NMAT3, NCOL))
    M[:11, :16] = hero.material_table()
    #          albedo                 rough  F0     wrap  sheen trans rim  amb  metal tex  sc     amp   bump
    M[M_GOLD, :15] = [1.00, 0.74, 0.32, 0.16, 0.90, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 0, 0.0, 0.0, 0.0]
    M[M_CLAY, :15] = [0.23, 0.105, 0.060, 0.78, 0.030, 0.22, 0.15, 0.0, 0.8, 1.0, 0.0, 1, 55.0, 0.30, 0.10]
    M[M_EMBER, :15] = [0.035, 0.031, 0.029, 0.90, 0.020, 0.10, 0.0, 0.0, 0.5, 1.0, 0.0, 0, 0.0, 0.0, 0.20]
    M[M_SNOW, :15] = [0.80, 0.84, 0.90, 0.62, 0.020, 0.55, 0.0, 0.10, 0.6, 1.0, 0.0, 1, 90.0, 0.06, 0.05]
    M[M_ICE, :15] = [0.030, 0.038, 0.048, 0.07, 0.022, 0.10, 0.0, 0.0, 0.8, 1.0, 0.0, 0, 0.0, 0.0, 0.0]
    M[M_COAL, :15] = [0.030, 0.027, 0.025, 0.85, 0.030, 0.10, 0.0, 0.0, 0.4, 1.0, 0.0, 0, 0.0, 0.0, 0.25]
    M[M_IRON, :15] = [0.045, 0.040, 0.036, 0.55, 0.120, 0.0, 0.0, 0.0, 0.8, 1.0, 0.6, 1, 120.0, 0.35, 0.15]
    M[M_ASH, :15] = [0.21, 0.20, 0.19, 0.98, 0.020, 0.40, 0.25, 0.0, 0.5, 1.0, 0.0, 1, 300.0, 0.25, 0.20]
    M[M_GLOVE, :15] = [0.105, 0.056, 0.036, 0.40, 0.040, 0.10, 0.22, 0.0, 0.7, 1.0, 0.0, 1, 260.0, 0.14, 0.16]
    M[M_EMBER, 16] = 1.0
    M[M_COAL, 16] = 1.0
    M[M_GOLD, 17] = 1.0
    M[M_ICE, 17] = 1.0
    M[M_STEEL, 17] = 0.35
    M[M_IRON, 17] = 0.25
    M[M_SNOW, 18:20] = [0.05, 3.0]
    return M


# ------------------------------------------------------------- the band ---

def band(B, c, Rw, R, tb, hb, rnd=0.0):
    """Append the Ring's band to the current group. Rw rows = world->local (row 1 = the ring's axis). rnd > 0: a
    rounded-rectangle section (ring.py's band); rnd = 0: tolkien.py's superellipse section."""
    row = B._row(T_BAND, 0.0, 0)
    row[5:8] = c
    row[11:20] = np.asarray(Rw, np.float64).reshape(-1)
    row[20] = tb
    row[21] = hb
    row[22] = R
    row[23] = rnd
    B.cur['prims'].append((row, np.r_[np.asarray(c, np.float64), R + tb + hb * 0.2 + 0.002]))
    return B


@nb.njit(cache=True, fastmath=True)
def _sd_band(P, i, px, py, pz):
    lx = px - P[i, 5]
    ly = py - P[i, 6]
    lz = pz - P[i, 7]
    x = P[i, 11] * lx + P[i, 12] * ly + P[i, 13] * lz
    y = P[i, 14] * lx + P[i, 15] * ly + P[i, 16] * lz
    z = P[i, 17] * lx + P[i, 18] * ly + P[i, 19] * lz
    q = abs(math.sqrt(x * x + z * z) - P[i, 22])
    tb = P[i, 20]
    hb = P[i, 21]
    a = abs(y)
    rnd = P[i, 23]
    if rnd > 0.0:
        # ring.py's band: a rounded rectangle revolved round the axis (exact)
        qx = q - tb + rnd
        qy = a - hb + rnd
        ox = max(qx, 0.0)
        oy = max(qy, 0.0)
        return math.sqrt(ox * ox + oy * oy) + min(max(qx, qy), 0.0) - rnd
    # superellipse |q/tb|^p + |y/hb|^p = 1 (approximate distance, scaled by the short semi-axis: safe)
    f = ((q / tb) ** PEXP + (a / hb) ** PEXP) ** (1.0 / PEXP) - 1.0
    if f > 0.6:
        # far away: a rounded box bound is tighter and exact enough
        ox = max(q - tb, 0.0)
        oy = max(a - hb, 0.0)
        return math.sqrt(ox * ox + oy * oy) + min(max(q - tb, a - hb), 0.0)
    return f * tb * 0.92


@nb.njit(cache=True, fastmath=True)
def prim_sd3(P, i, px, py, pz):
    if int(P[i, 0]) == T_BAND:
        return _sd_band(P, i, px, py, pz)
    return hs.prim_sd(P, i, px, py, pz)


@nb.njit(cache=True, fastmath=True)
def sdf3(P, G, cand, nc, px, py, pz):
    best = 1e9
    bg = -1
    cur = -1
    gd = 1e9
    for ii in range(nc):
        i = cand[ii]
        g = int(P[i, 1])
        if g != cur:
            if cur >= 0 and gd < 1e8:
                if gd < G[cur, 6]:
                    gd += hs.group_disp(G, cur, px, py, pz)
                if gd < best:
                    best = gd
                    bg = cur
            cur = g
            gd = 1e9
        d = prim_sd3(P, i, px, py, pz)
        op = int(P[i, 3])
        if op == 0:
            gd = hs._smin(gd, d, P[i, 2])
        elif op == 1:
            gd = hs._smax(gd, -d, P[i, 2])
        else:
            gd = hs._smax(gd, d, P[i, 2])
    if cur >= 0 and gd < 1e8:
        if gd < G[cur, 6]:
            gd += hs.group_disp(G, cur, px, py, pz)
        if gd < best:
            best = gd
            bg = cur
    return best, bg


@nb.njit(cache=True, fastmath=True)
def calc_normal3(P, G, cand, nc, px, py, pz, e):
    k1, _ = sdf3(P, G, cand, nc, px + e, py - e, pz - e)
    k2, _ = sdf3(P, G, cand, nc, px - e, py - e, pz + e)
    k3, _ = sdf3(P, G, cand, nc, px - e, py + e, pz - e)
    k4, _ = sdf3(P, G, cand, nc, px + e, py + e, pz + e)
    nx = k1 - k2 - k3 + k4
    ny = -k1 - k2 + k3 + k4
    nz = -k1 + k2 - k3 + k4
    nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / nn, ny / nn, nz / nn


@nb.njit(cache=True, fastmath=True)
def soft_shadow3(P, G, BS, allidx, nall, px, py, pz, lx, ly, lz, dist, k, buf):
    skip = k < 0.0
    if skip:
        k = -k
    nc, t0, t1 = hs.ray_cands(BS, allidx, nall, px, py, pz, lx, ly, lz, dist, buf, skip, P, G)
    if nc == 0:
        return 1.0
    res = 1.0
    t = 0.0015
    if t0 > t:
        t = t0
    tend = min(dist, t1)
    ph = 1e10
    for it in range(96):
        if t >= tend:
            break
        h, _ = sdf3(P, G, buf, nc, px + lx * t, py + ly * t, pz + lz * t)
        if h < 0.00015:
            return 0.0
        y = h * h / (2.0 * ph)
        d = math.sqrt(max(h * h - y * y, 0.0))
        r = k * d / max(1e-4, t - y)
        if r < res:
            res = r
        ph = h
        t += min(max(h * 0.9, 0.0004), 0.04)
    res = min(max(res, 0.0), 1.0)
    return res * res * (3.0 - 2.0 * res)


@nb.njit(cache=True, fastmath=True)
def calc_ao3(P, G, cand, nc, px, py, pz, nx, ny, nz, scale):
    occ = 0.0
    sca = 1.0
    for q in range(5):
        h = scale * (0.12 + 0.22 * q)
        d, _ = sdf3(P, G, cand, nc, px + nx * h, py + ny * h, pz + nz * h)
        occ += (h - d) * sca
        sca *= 0.72
    v = 1.0 - 1.6 * occ / scale
    return min(max(v, 0.0), 1.0)


# ------------------------------------------------------- textures, env ---

@nb.njit(cache=True, fastmath=True)
def _bilin(T, lev, u, v):
    """Bilinear sample of level `lev` of a (L, h, w) stack (u, v in texels of that level; clamped)."""
    h = T.shape[1]
    w = T.shape[2]
    x = u - 0.5
    y = v - 0.5
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    x0 = min(max(ix, 0), w - 1)
    x1 = min(max(ix + 1, 0), w - 1)
    y0 = min(max(iy, 0), h - 1)
    y1 = min(max(iy + 1, 0), h - 1)
    return (T[lev, y0, x0] * (1 - fx) * (1 - fy) + T[lev, y0, x1] * fx * (1 - fy) +
            T[lev, y1, x0] * (1 - fx) * fy + T[lev, y1, x1] * fx * fy)


@nb.njit(cache=True, fastmath=True)
def ins_sample(INS, XP, x, y, z, fp):
    """Coverage 0..1 of the inscription at a ring-local point (axis = local y). fp = pixel footprint (m).
    Line 1 on the outer face, read from outside; line 2 on the inner face, mirrored (tolkien.py's layout)."""
    R = XP[14]
    if XP[53] > 0.5:
        # ring.py's layout: one strip once round the band (u), across its width (v); the inner face mirrored
        tb = XP[15]
        rr = math.sqrt(x * x + z * z)
        qq = rr - R
        fade = min(1.0, max(0.0, (abs(qq) / tb - 0.35) / 0.40))
        if fade <= 0.0:
            return 0.0
        u = math.atan2(z, x) / 6.2831853
        if qq < 0.0:
            u = 0.37 - u
        u = u - math.floor(u)
        v = 0.5 - y / (2.0 * XP[16])
        if v < 0.0 or v > 1.0:
            return 0.0
        lf = math.log(max(1.0, XP[48] * fp * 1.3)) / 0.6931472
        nlev = int(XP[52])
        l0 = min(int(lf), nlev - 1)
        l1 = min(l0 + 1, nlev - 1)
        fr = min(max(lf - l0, 0.0), 1.0)
        s0 = 2.0 ** l0
        s1 = 2.0 ** l1
        c0 = _bilin(INS, l0, u * XP[41] / s0, v * XP[40] / s0)
        c1 = _bilin(INS, l1, u * XP[41] / s1, v * XP[40] / s1)
        return (c0 * (1 - fr) + c1 * fr) * fade
    em = XP[48]
    if em <= 0.0:
        return 0.0
    th = math.atan2(z, x)
    if th < 0.0:
        th += 6.2831853
    outer = math.sqrt(x * x + z * z) > R
    if outer:
        circ = XP[50]
        e = XP[42]
        base = XP[43]
        row0 = 0.0
        wline = XP[41]
        hline = XP[40]
    else:
        th = 6.2831853 - th
        circ = XP[51]
        e = XP[46]
        base = XP[47]
        row0 = XP[40]
        wline = XP[45]
        hline = XP[44]
    ucol = th / 6.2831853 * circ / em * e
    vrow = base - (y / em + XP[49]) * e
    if vrow < 0.0 or vrow > hline or ucol > wline:
        return 0.0
    # mip level from the footprint: texels per metre x metres per pixel
    tpm = e / em
    lf = math.log(max(1.0, tpm * fp * 1.3)) / 0.6931472
    nlev = int(XP[52])
    l0 = min(int(lf), nlev - 1)
    l1 = min(l0 + 1, nlev - 1)
    fr = min(max(lf - l0, 0.0), 1.0)
    s0 = 2.0 ** l0
    s1 = 2.0 ** l1
    c0 = _bilin(INS, l0, ucol / s0, (vrow + row0) / s0)
    c1 = _bilin(INS, l1, ucol / s1, (vrow + row0) / s1)
    return c0 * (1 - fr) + c1 * fr


@nb.njit(cache=True, fastmath=True)
def env_sample(ENV, dx, dy, dz, lev):
    """Equirect env stack (L, h, w, 3): u = atan2(x, z) (0 = +z), v = polar angle from +y."""
    L = ENV.shape[0]
    h = ENV.shape[1]
    w = ENV.shape[2]
    u = (math.atan2(dx, dz) / 6.2831853 + 0.5) * w
    v = math.acos(max(-1.0, min(1.0, dy))) / 3.14159265 * h
    lev = min(max(lev, 0.0), L - 1.0)
    l0 = int(lev)
    l1 = min(l0 + 1, L - 1)
    fr = lev - l0
    x = u - 0.5
    y = v - 0.5
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    x0 = ix % w
    x1 = (ix + 1) % w
    y0 = min(max(iy, 0), h - 1)
    y1 = min(max(iy + 1, 0), h - 1)
    r = 0.0
    g = 0.0
    b = 0.0
    for (ll, wl) in ((l0, 1.0 - fr), (l1, fr)):
        if wl <= 0.0:
            continue
        for c in range(3):
            val = (ENV[ll, y0, x0, c] * (1 - fx) * (1 - fy) + ENV[ll, y0, x1, c] * fx * (1 - fy) +
                   ENV[ll, y1, x0, c] * (1 - fx) * fy + ENV[ll, y1, x1, c] * fx * fy) * wl
            if c == 0:
                r += val
            elif c == 1:
                g += val
            else:
                b += val
    return r, g, b


@nb.njit(cache=True, fastmath=True)
def _hash3(ix, iy, iz):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@nb.njit(cache=True, fastmath=True)
def ember_emit(px, py, pz, sc, life, seed):
    """(glow 0..~1.5, ash cover 0..1) for a burning / dying lump. Cracks are the zero-set of a noise; the hot
    regions are the high parts of a broad noise; both shrink as life falls and the ash skin closes over them."""
    n1 = perlin3(px * sc + seed, py * sc, pz * sc)
    n2 = perlin3(px * sc * 2.7, py * sc * 2.7 + seed, pz * sc * 2.7)
    ridge = 1.0 - abs(n1 + 0.35 * n2)
    wd = 0.03 + 0.20 * life
    crack = min(1.0, max(0.0, (ridge - (1.0 - wd)) / wd))
    crack = crack * crack * (3.0 - 2.0 * crack)
    blob = perlin3(px * sc * 0.45 - seed, py * sc * 0.45, pz * sc * 0.45 + 2.1) + 0.3 * n2
    th = 0.55 - 1.05 * life
    hot = min(1.0, max(0.0, (blob - th) / 0.30))
    hot = hot * hot * (3.0 - 2.0 * hot)
    glow = crack * (0.25 + 0.75 * life) * (0.35 + 0.65 * hot) + 0.9 * hot * life * life
    mott = perlin3(px * sc * 5.0 + 3.3, py * sc * 5.0, pz * sc * 5.0 - seed)
    ash = min(1.0, max(0.0, (1.0 - life) * 1.25 + 0.35 * mott - 0.9 * hot * life))
    return glow, ash


# -------------------------------------------------------------- shading ---

@nb.njit(cache=True, fastmath=True)
def shade3(P, G, BS, allidx, nall, M, L, nl, env, H, cand, nc, buf, XP, ENV, INS,
           px, py, pz, nx, ny, nz, vx, vy, vz, g, fp, out):
    """heroine_sdf.shade + the v3 materials. out[0:3] <- linear radiance."""
    m = int(G[g, 7])
    ar = M[m, 0]
    ag = M[m, 1]
    ab = M[m, 2]
    tm = hs.tex_mod(M, m, px, py, pz)
    ar *= tm
    ag *= tm
    ab *= tm
    if M[m, 14] > 0.0:
        sc = M[m, 12] * 1.9
        if sc <= 0.0:
            sc = 400.0
        e = 0.3 / sc
        n0 = perlin3(px * sc, py * sc, pz * sc)
        gx = (perlin3((px + e) * sc, py * sc, pz * sc) - n0) / (e * sc)
        gy = (perlin3(px * sc, (py + e) * sc, pz * sc) - n0) / (e * sc)
        gz = (perlin3(px * sc, py * sc, (pz + e) * sc) - n0) / (e * sc)
        sc2 = sc * 3.3
        n1 = perlin3(px * sc2 + 5.1, py * sc2, pz * sc2)
        gx += 0.5 * (perlin3((px + e) * sc2 + 5.1, py * sc2, pz * sc2) - n1) / (e * sc)
        gy += 0.5 * (perlin3(px * sc2 + 5.1, (py + e) * sc2, pz * sc2) - n1) / (e * sc)
        gz += 0.5 * (perlin3(px * sc2 + 5.1, py * sc2, (pz + e) * sc2) - n1) / (e * sc)
        dn = gx * nx + gy * ny + gz * nz
        bs = M[m, 14]
        nx -= bs * (gx - dn * nx)
        ny -= bs * (gy - dn * ny)
        nz -= bs * (gz - dn * nz)
        nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
        nx /= nn
        ny /= nn
        nz /= nn
    rough = M[m, 3]
    f0 = M[m, 4]
    wrap = M[m, 5]
    sheen = M[m, 6]
    trans = M[m, 7]
    metal = M[m, 10]
    emr = 0.0
    emg = 0.0
    emb = 0.0
    refl = M[m, 17]
    # ---- the Ring: engraved letters (darker, rougher) that burn when awake
    if m == M_GOLD and XP[1] > 0.5:
        lx = px - XP[2]
        ly = py - XP[3]
        lz = pz - XP[4]
        x = XP[5] * lx + XP[6] * ly + XP[7] * lz
        y = XP[8] * lx + XP[9] * ly + XP[10] * lz
        z = XP[11] * lx + XP[12] * ly + XP[13] * lz
        ins = ins_sample(INS, XP, x, y, z, fp)
        if ins > 0.0:
            k = XP[18] * ins
            ar *= 1.0 - 0.55 * k
            ag *= 1.0 - 0.62 * k
            ab *= 1.0 - 0.70 * k
            rough = rough + 0.30 * k
            refl *= 1.0 - 0.45 * k
            gl = XP[17] * ins * (1.0 + 0.10 * math.sin(XP[0] * 7.3 + x * 900.0))
            emr += gl * 1.00
            emg += gl * 0.36
            emb += gl * 0.07
    # ---- burning lumps: cracks and hot patches under an ash skin
    if M[m, 16] > 0.5:
        if m == M_COAL:
            glow, ash = ember_emit(px, py, pz, XP[29], XP[30], 3.7)
            gain = XP[28]
        else:
            glow, ash = ember_emit(px, py, pz, XP[24], XP[19], 1.3)
            gain = XP[20]
        ar = ar * (1 - ash) + 0.42 * ash
        ag = ag * (1 - ash) + 0.40 * ash
        ab = ab * (1 - ash) + 0.38 * ash
        if glow > 0.0 and gain > 0.0:
            tq = min(1.5, glow)
            emr += gain * tq * 1.0
            emg += gain * tq * (0.06 + 0.22 * tq)
            emb += gain * tq * (0.008 + 0.03 * tq * tq)
    gloss_boost = 0.0
    if m == M_SKIN and H[0] > 0.5:
        lx = px - H[1]
        ly = py - H[2]
        lz = pz - H[3]
        u = lx * H[4] + ly * H[5] + lz * H[6]
        v = lx * H[7] + ly * H[8] + lz * H[9]
        w = lx * H[10] + ly * H[11] + lz * H[12]
        dv = (v - H[13]) / H[14]
        lipw = math.exp(-dv * dv * dv * dv) * math.exp(-(w / H[15]) ** 4) * (1.0 if u > H[16] else math.exp(-((u - H[16]) / 0.004) ** 2))
        ar = ar * (1.0 - lipw) + H[17] * lipw
        ag = ag * (1.0 - lipw) + H[18] * lipw
        ab = ab * (1.0 - lipw) + H[19] * lipw
        gloss_boost = 0.5 * lipw
    if m == M_SKIN and H[40] > 0.5:
        wsum = 0.0
        for q in range(int(H[40])):
            dx = px - H[41 + 3 * q]
            dy = py - H[42 + 3 * q]
            dz = pz - H[43 + 3 * q]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < 0.0004:
                wsum += math.exp(-d2 / (2.0 * 0.0075 * 0.0075))
        wsum = min(1.0, wsum)
        if wsum > 0.0:
            ar *= 1.0 + 0.20 * wsum
            ag *= 1.0 - 0.16 * wsum
            ab *= 1.0 - 0.12 * wsum
    if m == M_SKIN:
        # fine skin relief: pores and the lines of the knuckles and palm (bump only, never geometry)
        sc = 2600.0
        e = 0.25 / sc
        n0 = perlin3(px * sc, py * sc, pz * sc)
        gx = (perlin3((px + e) * sc, py * sc, pz * sc) - n0) / (e * sc)
        gy = (perlin3(px * sc, (py + e) * sc, pz * sc) - n0) / (e * sc)
        gz = (perlin3(px * sc, py * sc, (pz + e) * sc) - n0) / (e * sc)
        dn = gx * nx + gy * ny + gz * nz
        bs = 0.035
        nx -= bs * (gx - dn * nx)
        ny -= bs * (gy - dn * ny)
        nz -= bs * (gz - dn * nz)
        nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
        nx /= nn
        ny /= nn
        nz /= nn
    ndv = nx * vx + ny * vy + nz * vz
    if ndv < 0.0:
        ndv = 0.0
    r = 0.0
    gg = 0.0
    b = 0.0
    ao = 1.0
    if env[12] > 0.0:
        ao = calc_ao3(P, G, cand, nc, px + nx * 0.0005, py + ny * 0.0005, pz + nz * 0.0005, nx, ny, nz, env[12])
    metal_col = metal > 0.0 and m != M_STEEL
    # rim-only groups (G col 9 = strength): the warm near sources (soft_k != 0) reach them only at grazing angles
    sil = G[g, 9] if G.shape[1] > 9 else 0.0
    rimw = 1.0
    if sil > 0.0:
        rimw = (1.0 - sil) + sil * (1.0 - ndv) ** 4
    for q in range(nl):
        lx = L[q, 0] - px
        ly = L[q, 1] - py
        lz = L[q, 2] - pz
        d2 = lx * lx + ly * ly + lz * lz
        dist = math.sqrt(d2) + 1e-9
        lx /= dist
        ly /= dist
        lz /= dist
        E = 1.0 / (d2 + L[q, 6] * L[q, 6])
        ndl = nx * lx + ny * ly + nz * lz
        wr = wrap
        wg = wrap * 0.55
        wb = wrap * 0.35
        dr = max(0.0, (ndl + wr) / (1.0 + wr))
        dg = max(0.0, (ndl + wg) / (1.0 + wg))
        db = max(0.0, (ndl + wb) / (1.0 + wb))
        if dr <= 0.0 and trans <= 0.0 and sheen <= 0.0 and ndl <= 0.0:
            continue
        sh = 1.0
        if env[13] > 0.0 and L[q, 7] != 0.0 and (dr > 0.0 or trans > 0.0):
            sh = soft_shadow3(P, G, BS, allidx, nall, px + nx * 0.0012 + lx * 0.001, py + ny * 0.0012 + ly * 0.001,
                              pz + nz * 0.0012 + lz * 0.001, lx, ly, lz, dist - L[q, 6] * 0.5, L[q, 7], buf)
        shr = sh
        shg = sh
        shb = sh
        if m == M_SKIN and sh < 1.0:
            shr = sh + (1.0 - sh) * 0.10
            shg = sh + (1.0 - sh) * 0.02
        if sil > 0.0 and L[q, 7] != 0.0:
            E *= rimw
        Er = L[q, 3] * E
        Eg = L[q, 4] * E
        Eb = L[q, 5] * E
        cr = ar * dr * shr * (1.0 - metal)
        cgg = ag * dg * shg * (1.0 - metal)
        cb = ab * db * shb * (1.0 - metal)
        hx = lx + vx
        hy = ly + vy
        hz = lz + vz
        hn = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
        ndh = max(0.0, (nx * hx + ny * hy + nz * hz) / hn)
        vdh = max(0.0, (vx * hx + vy * hy + vz * hz) / hn)
        rgh = rough * (1.0 - 0.5 * gloss_boost)
        # an extended source widens its highlight (radius / distance)
        rgh = math.sqrt(rgh * rgh + min(0.25, (L[q, 6] / dist) ** 2) * 0.5)
        a = max(0.02, rgh * rgh)
        sr = 0.0
        sg = 0.0
        sb = 0.0
        if m == M_HAIR:
            pass
        elif ndl > 0.0:
            D = hs._ggx(ndh, a) / (4.0 * max(ndv, 0.08) * max(ndl, 0.08)) * ndl * sh
            if metal_col:
                fw = (1.0 - vdh) ** 5
                sr = D * (ar + (1.0 - ar) * fw)
                sg = D * (ag + (1.0 - ag) * fw)
                sb = D * (ab + (1.0 - ab) * fw)
            else:
                F = f0 + (1.0 - f0) * (1.0 - vdh) ** 5
                spec = D * F * (1.0 + 1.5 * gloss_boost)
                sr = spec
                sg = spec
                sb = spec
                if metal > 0.0:
                    sr *= ar * 8.0
                    sg *= ag * 8.0
                    sb *= ab * 8.0
        fz = 0.0
        if sheen > 0.0:
            fz = sheen * (1.0 - ndv) ** 3 * max(0.0, ndl + 0.35) * (0.4 + 0.6 * sh)
        tr = 0.0
        if trans > 0.0 and ndl < 0.2:
            back = max(0.0, -(vx * lx + vy * ly + vz * lz))
            tr = trans * (0.25 + back * back) * max(0.0, 0.2 - ndl) * sh
        # snow glints: a few crystal facets per mm^2 flash when they mirror the light into the eye
        gl = 0.0
        if M[m, 18] > 0.0 and ndl > 0.0:
            cs = 1100.0
            ix = int(math.floor(px * cs))
            iy = int(math.floor(py * cs))
            iz = int(math.floor(pz * cs))
            hh = _hash3(ix, iy, iz)
            if hh < M[m, 18]:
                fx = _hash3(iy, iz, ix) * 2.0 - 1.0
                fy = _hash3(iz, ix, iy) * 2.0 - 1.0
                fzz = _hash3(ix + 7, iy - 3, iz + 11) * 2.0 - 1.0
                mx = nx + 0.9 * fx
                my = ny + 0.9 * fy
                mz = nz + 0.9 * fzz
                mn = math.sqrt(mx * mx + my * my + mz * mz) + 1e-9
                c = (mx * hx + my * hy + mz * hz) / (mn * hn)
                if c > 0.995:
                    gl = M[m, 19] * ((c - 0.995) / 0.005) ** 4 * sh
        r += Er * (cr + sr + fz * ar * 2.0 + tr * ar * 3.0 + gl)
        gg += Eg * (cgg + sg + fz * ag * 2.0 + tr * ag * 1.2 + gl)
        b += Eb * (cb + sb + fz * ab * 2.0 + tr * ab * 0.8 + gl)
    ndr = nx * env[0] + ny * env[1] + nz * env[2]
    if ndr > 0.0:
        rim = M[m, 8] * ndr * (1.0 - ndv) ** 5
        if m == M_HAIR or m == M_COAT or m == M_SCARF or m == M_KNIT:
            rim *= 1.3
        r += env[3] * rim * (0.35 + ar * 2.0)
        gg += env[4] * rim * (0.35 + ag * 2.0)
        b += env[5] * rim * (0.35 + ab * 2.0)
    up = 0.5 + 0.5 * ny
    amb = M[m, 9] * ao
    bw = (1.0 - up) * (1.0 - up) * rimw
    r += amb * ar * (env[6] * up + env[9] * bw) * (1.0 - metal)
    gg += amb * ag * (env[7] * up + env[10] * bw) * (1.0 - metal)
    b += amb * ab * (env[8] * up + env[11] * bw) * (1.0 - metal)
    # ---- what the surface mirrors: the environment map (night, snow, fire, the vision)
    if refl > 0.0 and XP[25] > 0.5:
        rx = 2.0 * ndv * nx - vx
        ry = 2.0 * ndv * ny - vy
        rz = 2.0 * ndv * nz - vz
        lev = rough * 9.0
        er, eg, eb = env_sample(ENV, rx, ry, rz, lev)
        fw = (1.0 - ndv) ** 5
        occ = ao * ao
        if metal_col:
            r += refl * XP[26] * er * (ar + (1.0 - ar) * fw) * occ
            gg += refl * XP[26] * eg * (ag + (1.0 - ag) * fw) * occ
            b += refl * XP[26] * eb * (ab + (1.0 - ab) * fw) * occ
        else:
            F = f0 + (1.0 - f0) * fw
            if metal > 0.0:
                r += refl * XP[26] * er * ar * 2.5 * occ
                gg += refl * XP[26] * eg * ag * 2.5 * occ
                b += refl * XP[26] * eb * ab * 2.5 * occ
            else:
                r += refl * XP[26] * er * F * occ
                gg += refl * XP[26] * eg * F * occ
                b += refl * XP[26] * eb * F * occ
    out[0] = r + emr
    out[1] = gg + emg
    out[2] = b + emb


# ------------------------------------------------------------- renderer ---

@nb.njit(cache=True, fastmath=True)
def trace_sample3(cam, P, G, BS, M, L, nl, env, H, XP, ENV, INS, lst, nlst, allidx, nall, cand, buf,
                  sx, sy, res):
    ox = cam[0]
    oy = cam[1]
    oz = cam[2]
    dx, dy, dz = cam_ray(cam, sx, sy)
    res[0] = 0.0
    res[1] = 0.0
    res[2] = 0.0
    res[3] = 0.0
    res[4] = 1e9
    res[5] = -1.0
    nc, t0, t1 = hs.ray_cands(BS, lst, nlst, ox, oy, oz, dx, dy, dz, 1e9, cand, False, P, G)
    if nc == 0 or t1 < 0.0:
        return
    pixang = 1.0 / cam[12]
    t = max(t0, 0.02)
    hit = False
    qmin = 1e9
    tq = t
    for it in range(220):
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        d, g = sdf3(P, G, cand, nc, px, py, pz)
        q = d / (t * pixang)
        if q < qmin:
            qmin = q
            tq = t
        if d < 0.2 * t * pixang or d < 1e-5:
            hit = True
            break
        t += d * 0.85
        if t > t1:
            break
    if hit:
        cov = 1.0
        tq = t
    else:
        cov = 0.5 - qmin
        if cov <= 0.0:
            return
        if cov > 1.0:
            cov = 1.0
    px = ox + dx * tq
    py = oy + dy * tq
    pz = oz + dz * tq
    d, g = sdf3(P, G, cand, nc, px, py, pz)
    if g < 0:
        return
    e = max(5e-5, 0.5 * tq * pixang)
    nx, ny, nz = calc_normal3(P, G, cand, nc, px, py, pz, e)
    col = np.zeros(3)
    shade3(P, G, BS, allidx, nall, M, L, nl, env, H, cand, nc, buf, XP, ENV, INS,
           px, py, pz, nx, ny, nz, -dx, -dy, -dz, g, tq * pixang, col)
    res[0] = col[0] * cov
    res[1] = col[1] * cov
    res[2] = col[2] * cov
    res[3] = cov
    res[4] = tq
    res[5] = g


@nb.njit(cache=True, fastmath=True)
def render_region3(cam, P, G, BS, M, L, nl, env, H, XP, ENV, INS, tile_off, tile_idx, allidx, X0, Y0, w, h, TS,
                   ss, rgb, alpha, depth, gbuf):
    ntx = (w + TS - 1) // TS
    nty = (h + TS - 1) // TS
    cand = np.zeros(hs.MAXC, np.int64)
    buf = np.zeros(hs.MAXC, np.int64)
    res = np.zeros(6)
    nall = allidx.shape[0]
    for ty in range(nty):
        for tx in range(ntx):
            ti = ty * ntx + tx
            a0 = tile_off[ti]
            a1 = tile_off[ti + 1]
            if a1 <= a0:
                continue
            lst = tile_idx[a0:a1]
            for j in range(ty * TS, min(h, ty * TS + TS)):
                for i in range(tx * TS, min(w, tx * TS + TS)):
                    trace_sample3(cam, P, G, BS, M, L, nl, env, H, XP, ENV, INS, lst, a1 - a0, allidx, nall,
                                  cand, buf, X0 + i + 0.5, Y0 + j + 0.5, res)
                    rgb[j, i, 0] = res[0]
                    rgb[j, i, 1] = res[1]
                    rgb[j, i, 2] = res[2]
                    alpha[j, i] = res[3]
                    depth[j, i] = res[4]
                    gbuf[j, i] = res[5] if res[3] >= 0.999 else -2.0
    if ss <= 1:
        return
    edge = np.zeros((h, w), np.uint8)
    for j in range(h):
        for i in range(w):
            a = alpha[j, i]
            if a > 0.0 and a < 0.999:
                edge[j, i] = 1
                continue
            g0 = gbuf[j, i]
            l0 = rgb[j, i, 0] + rgb[j, i, 1] + rgb[j, i, 2]
            for (oj, oi) in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                jj = j + oj
                ii = i + oi
                if jj < 0 or ii < 0 or jj >= h or ii >= w:
                    continue
                if gbuf[jj, ii] != g0:
                    edge[j, i] = 1
                    break
                if g0 >= 0.0:
                    if abs(depth[jj, ii] - depth[j, i]) > 0.004:
                        edge[j, i] = 1
                        break
                    l1 = rgb[jj, ii, 0] + rgb[jj, ii, 1] + rgb[jj, ii, 2]
                    if abs(l1 - l0) > 0.25 * (l0 + l1) + 0.004:
                        edge[j, i] = 1
                        break
    for ty in range(nty):
        for tx in range(ntx):
            ti = ty * ntx + tx
            a0 = tile_off[ti]
            a1 = tile_off[ti + 1]
            if a1 <= a0:
                continue
            lst = tile_idx[a0:a1]
            for j in range(ty * TS, min(h, ty * TS + TS)):
                for i in range(tx * TS, min(w, tx * TS + TS)):
                    if edge[j, i] == 0:
                        continue
                    sr = 0.0
                    sg = 0.0
                    sb = 0.0
                    sa = 0.0
                    dmin = 1e9
                    for sj in range(ss):
                        for si in range(ss):
                            ox = (si + 0.5) / ss
                            oy = (sj + 0.5) / ss
                            trace_sample3(cam, P, G, BS, M, L, nl, env, H, XP, ENV, INS, lst, a1 - a0, allidx,
                                          nall, cand, buf, X0 + i + ox, Y0 + j + oy, res)
                            sr += res[0]
                            sg += res[1]
                            sb += res[2]
                            sa += res[3]
                            if res[4] < dmin:
                                dmin = res[4]
                    n = ss * ss
                    rgb[j, i, 0] = sr / n
                    rgb[j, i, 1] = sg / n
                    rgb[j, i, 2] = sb / n
                    alpha[j, i] = sa / n
                    depth[j, i] = dmin


def render(cam, B, H, lights, env, XP=None, ENV=None, INS=None, M=None, ss=3, pad=6, region=None, sil=None):
    """Sphere-trace a Builder. Returns (Y0, X0, rgb_premul, alpha, depth, group) or None. region = (x0, y0, x1, y1)
    clips the traced rectangle (render px). sil = {group name: 0..1}: rim-only strength under the warm sources."""
    P, G, BS = B.arrays()
    if sil:
        col = np.zeros((len(G), 1))
        for gi, g in enumerate(B.groups):
            col[gi, 0] = sil.get(g['name'], 0.0)
        G = np.c_[G, col]
    if XP is None:
        XP = np.zeros(64)
    if ENV is None:
        ENV = np.zeros((1, 2, 4, 3), np.float32)
    if INS is None:
        INS = np.zeros((1, 2, 2), np.float32)
    if M is None:
        M = material_table3()
    sx, sy, z = cam.project(BS[:, :3])
    ok = z > 0.02
    if not np.any(ok):
        return None
    r = cam.f * BS[:, 3] / np.maximum(z - BS[:, 3], 0.02)
    X0 = int(max(0, np.floor((sx[ok] - r[ok]).min()) - pad))
    X1 = int(min(cam.W, np.ceil((sx[ok] + r[ok]).max()) + pad))
    Y0 = int(max(0, np.floor((sy[ok] - r[ok]).min()) - pad))
    Y1 = int(min(cam.H, np.ceil((sy[ok] + r[ok]).max()) + pad))
    if region is not None:
        X0, Y0 = max(X0, region[0]), max(Y0, region[1])
        X1, Y1 = min(X1, region[2]), min(Y1, region[3])
    if X1 <= X0 or Y1 <= Y0:
        return None
    w, h = X1 - X0, Y1 - Y0
    TS = 16
    off, idx = hs.build_tiles(cam, BS, X0, Y0, w, h, TS, P)
    allidx = np.arange(len(P), dtype=np.int64)
    rgb = np.zeros((h, w, 3))
    a = np.zeros((h, w))
    d = np.full((h, w), 1e9)
    gb = np.full((h, w), -1.0)
    Lg = np.asarray(lights, np.float64).reshape(-1, 8)
    render_region3(cam.params(), P, G, BS, M, Lg, len(Lg), np.asarray(env, np.float64), H,
                   np.asarray(XP, np.float64), ENV, INS, off, idx, allidx, X0, Y0, w, h, TS, ss, rgb, a, d, gb)
    return Y0, X0, rgb.astype(np.float32), a.astype(np.float32), d.astype(np.float32), gb


def gloves(B, H=None, inflate=0.0009):
    """Her hands as leather gloves: the hand groups get M_GLOVE and grow by `inflate` (m), the nails go, and the
    cold flush on knuckles and fingertips is switched off (H[40] = 0)."""
    for g in B.groups:
        if g['name'] in ('hand_n', 'hand_f'):
            g['mat'] = M_GLOVE
            g['band'] = 0.003
            for row, bs in g['prims']:
                typ = int(row[0])
                if typ == 0 or typ == 2:
                    row[8:11] += inflate
                    if typ == 2:
                        row[20] += inflate
                elif typ == 1:
                    row[20] += inflate
                    row[21] += inflate
                bs[3] += inflate
        elif g['name'] in ('hand_n_nails', 'hand_f_nails'):
            g['prims'] = []
    if H is not None:
        H[40] = 0.0
    return B


def group_index(B, name):
    for gi, g in enumerate(B.groups):
        if g['name'] == name:
            return gi
    return -1


# ------------------------------------------------------ the Ring's data ---

def ring_frame(axis, ref=(1.0, 0.0, 0.0)):
    """World->local rows for a ring whose axis is `axis` (row 1)."""
    ay = np.asarray(axis, np.float64)
    ay = ay / np.linalg.norm(ay)
    ax = np.asarray(ref, np.float64) - np.dot(ref, ay) * ay
    if np.linalg.norm(ax) < 1e-6:
        ax = np.array([0.0, 0.0, 1.0]) - ay[2] * ay
    ax /= np.linalg.norm(ax)
    az = np.cross(ax, ay)
    return np.stack([ax, ay, az])


_INS = {}


def inscription_stack(levels=6):
    """(INS (L, h, w) float32, layout dict): the E5 Ring's two lines (embers/inscription.py, seeds 7 and 23, em
    120 px, laid out for tolkien.py's R_RACE=5 / TB=0.4 ring) stacked outer-over-inner, with a 2x mip chain."""
    if 'ins' in _INS:
        return _INS['ins']
    import importlib.util
    import os
    import cv2
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, 'cache', 'ring_inscription_v3.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        m1, m2 = z['m1'], z['m2']
        e1, b1, e2, b2 = float(z['e1']), float(z['b1']), float(z['e2']), float(z['b2'])
    else:
        spec = importlib.util.spec_from_file_location('ins_embers', os.path.join(here, '..', 'embers', 'inscription.py'))
        INSm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(INSm)
        em_u = 1.28 / (INSm.ASC - INSm.DSC)
        m1, e1, b1 = INSm.render_strip(seed=7, length=2 * math.pi * 5.4 / em_u, em=120)
        m2, e2, b2 = INSm.render_strip(seed=23, length=2 * math.pi * 4.6 / em_u, em=120)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, m1=m1, m2=m2, e1=e1, b1=b1, e2=e2, b2=b2)
    h1, w1 = m1.shape
    h2, w2 = m2.shape
    W = max(w1, w2)
    base = np.zeros((h1 + h2, W), np.float32)
    base[:h1, :w1] = m1
    base[h1:, :w2] = m2
    lv = [base]
    for _ in range(levels - 1):
        a = lv[-1]
        lv.append(cv2.resize(a, (max(1, a.shape[1] // 2), max(1, a.shape[0] // 2)), interpolation=cv2.INTER_AREA))
    stack = np.zeros((levels, base.shape[0], base.shape[1]), np.float32)
    for k, a in enumerate(lv):
        stack[k, :a.shape[0], :a.shape[1]] = a
    lay = dict(h1=h1, w1=w1, e1=e1, b1=b1, h2=h2, w2=w2, e2=e2, b2=b2, levels=levels)
    _INS['ins'] = (stack, lay)
    return stack, lay


def ring_xp(XP, centre, rows, R, tb, hb, glow=0.0, engrave=1.0):
    """Fill the Ring's slots of XP. The inscription is laid out exactly as tolkien.py lays it on its unit ring
    (R 5, TB 0.4, HB 0.95), scaled by R / 5."""
    _, lay = inscription_stack()
    s = R / 5.0
    em = 1.28 / (2.3 + 1.15) * s
    XP[1] = 1.0
    XP[2:5] = centre
    XP[5:14] = np.asarray(rows, np.float64).reshape(-1)
    XP[14], XP[15], XP[16] = R, tb, hb
    XP[17] = glow
    XP[18] = engrave
    XP[40], XP[41], XP[42], XP[43] = lay['h1'], lay['w1'], lay['e1'], lay['b1']
    XP[44], XP[45], XP[46], XP[47] = lay['h2'], lay['w2'], lay['e2'], lay['b2']
    XP[48] = em
    XP[49] = 0.5 * (2.3 - 1.15)
    XP[50] = 2 * math.pi * (R + tb)
    XP[51] = 2 * math.pi * (R - tb)
    XP[52] = lay['levels']
    return XP


_INSA = {}


def inscription_accord(levels=6):
    """(INS (L, h, w), (h, w)): accord/ring.py's inscription strip (its own invented script, seed 1914) with a 2x mip
    chain; cached in shots/hills/cache."""
    if 'a' in _INSA:
        return _INSA['a']
    import importlib.util
    import os
    import cv2
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, 'cache', 'ring_inscription_accord_v3.npz')
    if os.path.exists(cache):
        cov = np.load(cache)['cov']
    else:
        spec = importlib.util.spec_from_file_location('ring_accord_ins', os.path.join(here, '..', 'accord', 'ring.py'))
        RG = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(RG)
        cov = RG.inscription(RG.STRIP_W, RG.STRIP_H).astype(np.float32)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, cov=cov)
    lv = [cov]
    for _ in range(levels - 1):
        a = lv[-1]
        lv.append(cv2.resize(a, (max(1, a.shape[1] // 2), max(1, a.shape[0] // 2)), interpolation=cv2.INTER_AREA))
    stack = np.zeros((levels,) + cov.shape, np.float32)
    for k, a in enumerate(lv):
        stack[k, :a.shape[0], :a.shape[1]] = a
    _INSA['a'] = (stack, cov.shape)
    return _INSA['a']


def ring_xp_accord(XP, centre, rows, R, tb, hb, glow=0.0, engrave=1.0):
    """The Ring's XP slots with ring.py's inscription layout (XP[53] = 1)."""
    stack, (hh, ww) = inscription_accord()
    XP[1] = 1.0
    XP[2:5] = centre
    XP[5:14] = np.asarray(rows, np.float64).reshape(-1)
    XP[14], XP[15], XP[16] = R, tb, hb
    XP[17] = glow
    XP[18] = engrave
    XP[40], XP[41] = hh, ww
    XP[48] = ww / (2 * math.pi * (R + tb))          # texels per metre round the outer face
    XP[52] = stack.shape[0]
    XP[53] = 1.0
    return XP
