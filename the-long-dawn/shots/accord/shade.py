"""Per-pixel surface render for ACCORD (numba): primary visibility + shading + emission."""
import math

import numpy as np
from numba import njit, prange

from nbcore import FM, clamp, sstep, mix, vnoise2, fbm2, hash2i, tex_sample, grid_sample
from geom import (TABLE_R, TABLE_Z, DAIS_R, DAIS_Z, BOWL_R, LIP_R, sd_fig, sd_stone, ray_aabb,
                  trace_fig, trace_stone, normal_fig, normal_stone, trace_bowl, bowl_height,
                  shadow_occluders, shadow_dir,
                  F_BB, F_R, F_G, F_B, F_TLIT, F_X, F_Y, S_BB, S_ALB, S_X, S_Y, S_SEED)
from textmaps import (OATH_R_IN, OATH_R_OUT, OATH_CAP, MAX_DEPTH, TOG_R, TOG_BORDER, BAND_IN, BAND_OUT,
                      RAY_R, FLOOR_BAND)

# ------------------------------------------------------------- param idx ---
P_T = 0
P_LX, P_LY = 1, 2
P_L1Z, P_L1RAD, P_L1I = 3, 4, 5            # low hearth light (rakes the table)
P_L2Z, P_L2RAD, P_L2I = 6, 7, 8            # high hearth light (long shadows)
P_LCR, P_LCG, P_LCB = 9, 10, 11
P_MDX, P_MDY, P_MDZ, P_MI, P_MCR, P_MCG, P_MCB = 12, 13, 14, 15, 16, 17, 18
P_SKI, P_SKR, P_SKG, P_SKB = 19, 20, 21, 22
P_WI, P_WCR, P_WCG, P_WCB = 23, 24, 25, 26
P_OP = 27                      # 27..30 oath writing progress (<0 not started)
P_TG, P_TGPHI = 31, 32         # together sweep progress 0..1, start angle (world)
P_ORNB, P_ORNR = 33, 34        # border ring light, hearth rays light
P_FL = 35                      # flare 0..1
P_COAL = 36
P_FALLP, P_FALLD0 = 37, 38
P_EMO, P_EMT, P_EMHOT = 39, 40, 41
P_NT = 42                      # number of torch lights
P_TEXR = 43
P_PGC_X0, P_PGC_CELL, P_PGF_X0, P_PGF_CELL = 44, 45, 46, 47
P_IGC_X0, P_IGC_CELL, P_IGF_X0, P_IGF_CELL = 48, 49, 50, 51
P_GNDI, P_GNDK = 52, 53
P_SHIM = 54
P_SHEEN = 55
P_NPARAM = 64

R_SPLIT = 0.5 * (OATH_R_IN + OATH_CAP + OATH_R_OUT)
BAND_LO = BAND_IN - 0.06
BAND_HI = BAND_OUT + 0.06

GOLD = (1.0, 0.644, 0.195)
PALE = (1.0, 0.879, 0.584)


@njit(inline='always', **FM)
def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@njit(**FM)
def oath_emission(r, th, depth_n, PR, OANG):
    """(lit, hot) for oath letters at polar (r, th)."""
    lit_sum = 0.0
    hot_sum = 0.0
    for k in range(4):
        p = PR[P_OP + k]
        if p <= 0.0:
            continue
        line2 = r < R_SPLIT
        if line2:
            ts = OANG[k, 2]
            te = OANG[k, 3]
        else:
            ts = OANG[k, 0]
            te = OANG[k, 1]
        mid = 0.5 * (ts + te)
        half = 0.5 * (ts - te)
        d = wrap_pi(th - mid)
        if abs(d) > half + 0.05:
            continue
        s = clamp((half - d) / (2.0 * half), 0.0, 1.0)
        s = 0.5 * s + (0.5 if line2 else 0.0)
        se = s + 0.03 * (1.0 - depth_n)
        lit = sstep(se - 0.01, se + 0.01, p)
        hot = math.exp(-((p - se) / 0.028) ** 2)
        # after-glow: freshly written letters are hotter, cooling over ~0.4 of the write time
        cool = math.exp(-max(p - se, 0.0) / 0.25)
        if p > 1.6:
            hot = 0.0
        lit_sum += lit * (1.0 + 0.9 * cool)
        hot_sum += hot
    return lit_sum, hot_sum


@njit(**FM)
def oath_lit_s(r, th, PR, OANG):
    tot = 0.0
    for k in range(4):
        p = PR[P_OP + k]
        if p <= 0.0:
            continue
        for ln in range(2):
            ts = OANG[k, 2 * ln]
            te = OANG[k, 2 * ln + 1]
            mid = 0.5 * (ts + te)
            half = 0.5 * (ts - te)
            d = wrap_pi(th - mid)
            if abs(d) > half + 0.08:
                continue
            s = clamp((half - d) / (2.0 * half), 0.0, 1.0)
            s = 0.5 * s + 0.5 * ln
            tot = max(tot, sstep(s - 0.03, s + 0.03, p))
    return tot


@njit(**FM)
def hearth_term(px, py, pz, nx, ny, nz, vx, vy, vz, lz, rad, I, is_ground, sheen, skip_fig,
                PR, F, nf, S, ns):
    """Irradiance (scalar) from one hearth light + sheen term."""
    lx = PR[P_LX] - px
    ly = PR[P_LY] - py
    lzz = lz - pz
    d2 = lx * lx + ly * ly + lzz * lzz
    d = math.sqrt(d2)
    lx /= d
    ly /= d
    lzz /= d
    ndl = nx * lx + ny * ly + nz * lzz
    if ndl <= 0.0 and sheen <= 0.0:
        return 0.0
    fall = (d2 + PR[P_FALLD0] ** 2) ** (-0.5 * PR[P_FALLP])
    if is_ground:
        if ndl > 0.0:
            ndl = ndl ** PR[P_GNDI]
        fall *= PR[P_GNDK]
    vis = shadow_occluders(px, py, pz, PR[P_LX], PR[P_LY], lz, rad, F, nf, S, ns, skip_fig)
    e = I * max(ndl, 0.0) * fall * vis
    if sheen > 0.0 and vis > 0.0:
        nv = abs(nx * vx + ny * vy + nz * vz)
        wrapl = max(ndl + 0.35, 0.0) / 1.35
        e += sheen * (1.0 - nv) ** 2.5 * I * fall * vis * wrapl
    return e


@njit(**FM)
def light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, sheen, is_ground, skip_fig,
             PR, TL, F, nf, S, ns, igc, igf, ao):
    """Diffuse lighting of a surface point. Returns rgb radiance (excluding emission)."""
    er = 0.0
    eg = 0.0
    eb = 0.0
    e = 0.0
    if PR[P_L1I] > 0.0:
        e += hearth_term(px, py, pz, nx, ny, nz, vx, vy, vz, PR[P_L1Z], PR[P_L1RAD], PR[P_L1I],
                         is_ground, sheen, skip_fig, PR, F, nf, S, ns)
    if PR[P_L2I] > 0.0:
        e += hearth_term(px, py, pz, nx, ny, nz, vx, vy, vz, PR[P_L2Z], PR[P_L2RAD], PR[P_L2I],
                         is_ground, sheen, skip_fig, PR, F, nf, S, ns)
    er += e * PR[P_LCR]
    eg += e * PR[P_LCG]
    eb += e * PR[P_LCB]
    # torches / ribbons (unshadowed point lights)
    nt = int(PR[P_NT])
    for k in range(nt):
        lx = TL[k, 0] - px
        ly = TL[k, 1] - py
        lz = TL[k, 2] - pz
        d2 = lx * lx + ly * ly + lz * lz
        d = math.sqrt(d2)
        ndl = (nx * lx + ny * ly + nz * lz) / d
        f = 0.0
        if ndl > 0.0:
            if is_ground:
                ndl = ndl ** 0.6
            f = ndl / (d2 + 0.12)
        if sheen > 0.0:
            nv = abs(nx * vx + ny * vy + nz * vz)
            f += sheen * 0.6 * (1.0 - nv) ** 2.5 * max(ndl + 0.3, 0.0) / (d2 + 0.12)
        er += f * TL[k, 3]
        eg += f * TL[k, 4]
        eb += f * TL[k, 5]
    # moon
    mdx = PR[P_MDX]
    mdy = PR[P_MDY]
    mdz = PR[P_MDZ]
    ndm = nx * mdx + ny * mdy + nz * mdz
    if ndm > 0.0:
        mv = 1.0
        rr = math.sqrt(px * px + py * py)
        if rr > 4.5 and rr < 14.0 and pz < 0.5:
            mv = shadow_dir(px, py, pz, mdx, mdy, mdz, 0.02, S, ns, F, 0)
        e = PR[P_MI] * ndm * mv
        er += e * PR[P_MCR]
        eg += e * PR[P_MCG]
        eb += e * PR[P_MCB]
    # sky ambient
    sk = (0.55 + 0.45 * nz) * ao
    er += sk * PR[P_SKI] * PR[P_SKR]
    eg += sk * PR[P_SKI] * PR[P_SKG]
    eb += sk * PR[P_SKI] * PR[P_SKB]
    # walker torchlight (ground-ish points)
    if pz < 1.0 and PR[P_WI] > 0.0:
        w = grid_sample(igf, PR[P_IGF_X0], PR[P_IGF_X0], PR[P_IGF_CELL], px, py)
        if w == 0.0:
            w = grid_sample(igc, PR[P_IGC_X0], PR[P_IGC_X0], PR[P_IGC_CELL], px, py)
        w *= PR[P_WI] * max(nz, 0.2)
        er += w * PR[P_WCR]
        eg += w * PR[P_WCG]
        eb += w * PR[P_WCB]
    return ar * er * ao, ag * eg * ao, ab * eb * ao


@njit(**FM)
def contact_ao(px, py, pz, F, nf, S, ns):
    ao = 1.0
    for i in range(nf):
        dx = px - F[i, F_X]
        dy = py - F[i, F_Y]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1.0:
            ao *= 0.45 + 0.55 * sstep(0.22, 0.85, d)
    for j in range(ns):
        dx = px - S[j, S_X]
        dy = py - S[j, S_Y]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 2.2:
            ao *= 0.45 + 0.55 * sstep(0.45, 1.9, d)
    return ao


@njit(**FM)
def ground_material(x, y, fp, PR, pgc, pgf):
    """Albedo + terrain normal of the plain."""
    r = math.sqrt(x * x + y * y)
    n1 = fbm2(x * 0.018, y * 0.018, 11, 5, 2.1, 0.55, fp * 0.018)
    n2 = fbm2(x * 0.33, y * 0.33, 12, 5, 2.2, 0.5, fp * 0.33)
    n3 = fbm2(x * 2.7, y * 2.7, 13, 4, 2.3, 0.5, fp * 2.7)
    g = 1.0 + 1.3 * n1 + 0.9 * n2 + 0.6 * n3
    ar = 0.048 * g
    ag = 0.058 * g
    ab = 0.046 * g
    dark = sstep(0.04, 0.22, n1 + 0.35 * n2)
    ar = mix(ar, 0.024, dark * 0.8)
    ag = mix(ag, 0.028, dark * 0.8)
    ab = mix(ab, 0.027, dark * 0.8)
    # trampled earth around the stone ring
    tr = sstep(20.0, 8.0, r) * (0.6 + 0.8 * (n2 + 0.5))
    tr = clamp(tr, 0.0, 1.0)
    ar = mix(ar, 0.06 * (1 + 0.6 * n3), tr * 0.7)
    ag = mix(ag, 0.053 * (1 + 0.6 * n3), tr * 0.7)
    ab = mix(ab, 0.046 * (1 + 0.6 * n3), tr * 0.7)
    # paths
    pm = grid_sample(pgf, PR[P_PGF_X0], PR[P_PGF_X0], PR[P_PGF_CELL], x, y)
    if abs(x) > -PR[P_PGF_X0] - 1.0 or abs(y) > -PR[P_PGF_X0] - 1.0:
        pm = grid_sample(pgc, PR[P_PGC_X0], PR[P_PGC_X0], PR[P_PGC_CELL], x, y)
    pm = clamp(pm * (0.85 + 0.5 * n3), 0.0, 1.0)
    ar = mix(ar, 0.10 * (1 + 0.5 * n3), pm)
    ag = mix(ag, 0.090 * (1 + 0.5 * n3), pm)
    ab = mix(ab, 0.076 * (1 + 0.5 * n3), pm)
    # terrain normal: gentle hills, flattened near the circle
    relief = sstep(25.0, 80.0, r)
    e = 3.0
    h0 = fbm2(x / 260.0, y / 260.0, 21, 3, 2.0, 0.5, 0.0)
    hx = fbm2((x + e) / 260.0, y / 260.0, 21, 3, 2.0, 0.5, 0.0)
    hy = fbm2(x / 260.0, (y + e) / 260.0, 21, 3, 2.0, 0.5, 0.0)
    amp = 55.0 * relief
    gx = (hx - h0) / e * amp
    gy = (hy - h0) / e * amp
    nx = -gx - 0.07 * fbm2(x * 2.7 + 5.1, y * 2.7, 14, 3, 2.2, 0.5, fp * 2.7)
    ny = -gy - 0.07 * fbm2(x * 2.7, y * 2.7 + 3.3, 15, 3, 2.2, 0.5, fp * 2.7)
    l = math.sqrt(nx * nx + ny * ny + 1.0)
    return ar, ag, ab, nx / l, ny / l, 1.0 / l


@njit(**FM)
def dais_material(x, y, fp):
    r = math.sqrt(x * x + y * y)
    th = math.atan2(y, x)
    # rings of slabs outside the floor band
    if r < FLOOR_BAND[0]:
        r0 = TABLE_R
        r1 = FLOOR_BAND[0]
        k = 0
    else:
        k = 1 + int((r - FLOOR_BAND[1]) / 0.72)
        r0 = FLOOR_BAND[1] + (k - 1) * 0.72
        r1 = r0 + 0.72
    nslab = int(2.0 * math.pi * 0.5 * (r0 + r1) / 0.9)
    off = hash2i(k, 7, 3)
    u = (th / (2.0 * math.pi) + 0.5) * nslab + off * 7.0 + 0.25 * (vnoise2(r * 3.0, k * 5.0, 9) - 0.5)
    si = math.floor(u)
    fu = u - si
    circ = 2.0 * math.pi * r / nslab
    dj = min(min(fu, 1.0 - fu) * circ, min(r - r0, r1 - r))
    h = hash2i(int(si), k, 5)
    joint = (1.0 - sstep(0.002 + fp * 0.5, 0.009 + fp, dj)) * 0.8
    n2 = fbm2(x * 4.0, y * 4.0, 31, 4, 2.2, 0.5, fp * 4.0)
    tint = 0.75 + 0.45 * h + 0.4 * n2
    ar = 0.080 * tint
    ag = 0.075 * tint
    ab = 0.067 * tint
    ar = mix(ar, 0.018, joint)
    ag = mix(ag, 0.022, joint)
    ab = mix(ab, 0.016, joint)
    tnx = (hash2i(int(si), k, 11) - 0.5) * 0.05 + 0.07 * fbm2(x * 9.0, y * 9.0, 32, 3, 2.1, 0.5, fp * 9.0)
    tny = (hash2i(int(si), k, 12) - 0.5) * 0.05 + 0.07 * fbm2(x * 9.0 + 3.0, y * 9.0, 33, 3, 2.1, 0.5, fp * 9.0)
    return ar, ag, ab, tnx, tny, joint


@njit(**FM)
def carve(px, py, fp, PR, tflat, toffs, tsizes, buf, tmp, res):
    """Sample inscription map: res = [depth_n, cov_oath, cov_tog, cov_orn, gx, gy]."""
    texel0 = 2.0 * PR[P_TEXR] / tsizes[0]
    lod = math.log2(max(fp / texel0, 1.0))
    tex_sample(tflat, toffs, tsizes, PR[P_TEXR], px, py, lod, buf, tmp)
    dep = buf[0]
    res[0] = dep / MAX_DEPTH
    res[1] = buf[1]
    res[2] = buf[2]
    res[3] = buf[3]
    res[4] = 0.0
    res[5] = 0.0
    if dep > 1e-6 or lod < 2.5:
        texel = texel0 * (2.0 ** max(lod, 0.0))
        e = max(texel, 0.0015)
        tex_sample(tflat, toffs, tsizes, PR[P_TEXR], px + e, py, lod, tmp, buf)
        a = tmp[0]
        tex_sample(tflat, toffs, tsizes, PR[P_TEXR], px - e, py, lod, tmp, buf)
        res[4] = (a - tmp[0]) / (2 * e)
        tex_sample(tflat, toffs, tsizes, PR[P_TEXR], px, py + e, lod, tmp, buf)
        a = tmp[0]
        tex_sample(tflat, toffs, tsizes, PR[P_TEXR], px, py - e, lod, tmp, buf)
        res[5] = (a - tmp[0]) / (2 * e)


@njit(**FM)
def tog_emission(th, cov_t, depth_n, PR):
    if cov_t <= 0.001 or PR[P_TG] <= 0.0:
        return 0.0, 0.0
    sweep = PR[P_TG] * 2.0 * math.pi
    a = (PR[P_TGPHI] - th) % (2.0 * math.pi)
    lit = sstep(a - 0.03, a + 0.03, sweep)
    hot = math.exp(-((sweep - a) / 0.10) ** 2) if PR[P_TG] < 1.0 else 0.0
    cool = math.exp(-max(sweep - a, 0.0) / 0.8) if lit > 0.0 else 0.0
    pool = 0.55 + 0.45 * clamp(depth_n * 1.6, 0.0, 1.0)
    return cov_t * lit * pool * (1.0 + 0.8 * cool), cov_t * hot


@njit(**FM)
def shade_sample(ox, oy, oz, dx, dy, dz, pix, PR, TL, F, nf, S, ns, tflat, toffs, tsizes,
                 pgc, pgf, igc, igf, OANG, buf, tmp, res):
    """Trace + shade one ray. Returns (r,g,b, t, id)."""
    tbest = 1e30
    idb = -1
    mat = 0
    if dz < 0.0:
        t = (TABLE_Z - oz) / dz
        hx = ox + dx * t
        hy = oy + dy * t
        rr = hx * hx + hy * hy
        if rr <= TABLE_R * TABLE_R:
            if rr < (LIP_R + 0.08) ** 2:
                tb = trace_bowl(ox, oy, oz, dx, dy, dz)
                if tb > 0.0:
                    tbest = tb
                    idb = 2
            else:
                tbest = t
                idb = 2
        if idb < 0:
            t = (DAIS_Z - oz) / dz
            hx = ox + dx * t
            hy = oy + dy * t
            rr = hx * hx + hy * hy
            if rr <= DAIS_R * DAIS_R:
                tbest = t
                idb = 1
            else:
                t = -oz / dz
                tbest = t
                idb = 0
    idx = 1.0 / dx if abs(dx) > 1e-9 else 1e9
    idy = 1.0 / dy if abs(dy) > 1e-9 else 1e9
    idz = 1.0 / dz if abs(dz) > 1e-9 else 1e9
    for i in range(nf):
        t0, t1 = ray_aabb(ox, oy, oz, idx, idy, idz, F[i, F_BB], F[i, F_BB + 1], F[i, F_BB + 2],
                          F[i, F_BB + 3], F[i, F_BB + 4], F[i, F_BB + 5])
        if t1 < max(t0, 0.0) or t0 > tbest:
            continue
        th, m = trace_fig(ox, oy, oz, dx, dy, dz, t0, min(t1, tbest), F, i, pix)
        if th > 0.0 and th < tbest:
            tbest = th
            idb = 10 + i
            mat = m
    for j in range(ns):
        t0, t1 = ray_aabb(ox, oy, oz, idx, idy, idz, S[j, S_BB], S[j, S_BB + 1], S[j, S_BB + 2],
                          S[j, S_BB + 3], S[j, S_BB + 4], S[j, S_BB + 5])
        if t1 < max(t0, 0.0) or t0 > tbest:
            continue
        th = trace_stone(ox, oy, oz, dx, dy, dz, t0, min(t1, tbest), S, j, pix)
        if th > 0.0 and th < tbest:
            tbest = th
            idb = 100 + j
    if idb < 0:
        return 0.0, 0.0, 0.0, 1e6, -1

    px = ox + dx * tbest
    py = oy + dy * tbest
    pz = oz + dz * tbest
    fp = tbest * pix
    vx = -dx
    vy = -dy
    vz = -dz
    er = 0.0
    eg = 0.0
    eb = 0.0
    T = PR[P_T]
    if idb == 0:
        ar, ag, ab, nx, ny, nz = ground_material(px, py, fp, PR, pgc, pgf)
        ao = contact_ao(px, py, pz, F, nf, S, ns)
        cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, 0.0, True, -1,
                              PR, TL, F, nf, S, ns, igc, igf, ao)
    elif idb == 1:
        r = math.sqrt(px * px + py * py)
        th = math.atan2(py, px)
        inband = r > FLOOR_BAND[0] and r < FLOOR_BAND[1]
        depth_n = 0.0
        cov_t = 0.0
        cov_n = 0.0
        if inband:
            n1 = fbm2(px * 6.0, py * 6.0, 61, 4, 2.1, 0.5, fp * 6.0)
            sp = fbm2(px * 80.0, py * 80.0, 62, 3, 2.3, 0.6, fp * 80.0)
            g = 1.0 + 0.35 * n1 + 0.4 * sp
            ar = 0.066 * g
            ag = 0.063 * g
            ab = 0.058 * g
            nx = 0.03 * fbm2(px * 30.0, py * 30.0, 63, 3, 2.2, 0.5, fp * 30.0)
            ny = 0.03 * fbm2(px * 30.0 + 4.0, py * 30.0, 64, 3, 2.2, 0.5, fp * 30.0)
            joint = 0.0
            # edge bevels of the band
            eb_ = min(r - FLOOR_BAND[0], FLOOR_BAND[1] - r)
            if eb_ < 0.025:
                sgn = 1.0 if r - FLOOR_BAND[0] < FLOOR_BAND[1] - r else -1.0
                nx += -sgn * 0.6 * (1 - eb_ / 0.025) * px / r
                ny += -sgn * 0.6 * (1 - eb_ / 0.025) * py / r
            carve(px, py, fp, PR, tflat, toffs, tsizes, buf, tmp, res)
            depth_n = res[0]
            cov_t = res[2]
            cov_n = res[3]
            nx += res[4]
            ny += res[5]
            pale = 1.0 + 0.4 * clamp(depth_n * 1.5, 0.0, 1.0)
            ar *= pale
            ag *= pale
            ab *= pale
        else:
            ar, ag, ab, nx, ny, joint = dais_material(px, py, fp)
        nz = 1.0
        l = math.sqrt(nx * nx + ny * ny + 1.0)
        nx /= l
        ny /= l
        nz /= l
        ao = contact_ao(px, py, pz, F, nf, S, ns) * (1.0 - 0.3 * joint) * (1.0 - 0.3 * depth_n)
        ao *= 0.5 + 0.5 * sstep(TABLE_R, TABLE_R + 0.3, r)
        cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, 0.0, True, -1,
                              PR, TL, F, nf, S, ns, igc, igf, ao)
        if inband:
            lit, hot = tog_emission(th, cov_t, depth_n, PR)
            em = lit * PR[P_EMT]
            ehh = hot * PR[P_EMHOT] * 0.7
            er += em * GOLD[0] + ehh * PALE[0]
            eg += em * GOLD[1] + ehh * PALE[1]
            eb += em * GOLD[2] + ehh * PALE[2]
            if cov_n > 0.001 and PR[P_TG] > 0.0:
                sweep = PR[P_TG] * 2.0 * math.pi
                a = (PR[P_TGPHI] - th) % (2.0 * math.pi)
                e = sstep(a - 0.05, a + 0.05, sweep) * 0.6 * cov_n * PR[P_EMT]
                er += e * GOLD[0]
                eg += e * GOLD[1]
                eb += e * GOLD[2]
    elif idb == 2:
        r = math.sqrt(px * px + py * py)
        th = math.atan2(py, px)
        n1 = fbm2(px * 3.2, py * 3.2, 41, 4, 2.1, 0.5, fp * 3.2)
        sp = fbm2(px * 70.0, py * 70.0, 42, 3, 2.3, 0.6, fp * 70.0)
        g = 1.0 + 0.55 * n1 + 0.5 * sp
        ar = 0.105 * g
        ag = 0.094 * g
        ab = 0.080 * g
        lich = sstep(1.70, 1.93, r) * sstep(0.02, 0.2, fbm2(px * 3.0, py * 3.0, 43, 4, 2.0, 0.5, fp * 3.0))
        ar = mix(ar, 0.085, lich * 0.6)
        ag = mix(ag, 0.098, lich * 0.6)
        ab = mix(ab, 0.07, lich * 0.6)
        soot = sstep(1.0, 0.62, r)
        ar = mix(ar, 0.030, soot * 0.9)
        ag = mix(ag, 0.026, soot * 0.9)
        ab = mix(ab, 0.024, soot * 0.9)
        nx = 0.0
        ny = 0.0
        nz = 1.0
        depth_n = 0.0
        cov_o = 0.0
        cov_n = 0.0
        if r < LIP_R + 0.08:
            e = 0.004
            h0 = bowl_height(r)
            h1 = bowl_height(r + e)
            gr = (h1 - h0) / e
            nx = -gr * px / (r + 1e-6)
            ny = -gr * py / (r + 1e-6)
            if r < BOWL_R:
                ash = 0.5 + 0.8 * fbm2(px * 30.0, py * 30.0, 44, 3, 2.1, 0.5, fp * 30.0)
                ar = 0.035 * ash
                ag = 0.032 * ash
                ab = 0.030 * ash
                coal = PR[P_COAL]
                if coal > 0.0:
                    cn = 0.5 + fbm2(px * 14.0 + 3.0, py * 14.0 - T * 0.01, 45, 4, 2.1, 0.55, fp * 14.0)
                    cn2 = 0.5 + fbm2(px * 5.0 - T * 0.03, py * 5.0, 46, 3, 2.0, 0.5, fp * 5.0)
                    glow = sstep(0.35, 0.85, cn * 0.55 + cn2 * 0.55) * (1.0 - (r / BOWL_R) ** 2)
                    flick = 0.75 + 0.25 * math.sin(T * 0.9 + cn * 20.0)
                    em = coal * glow * flick * 5.0
                    er += em * 1.0
                    eg += em * 0.45
                    eb += em * 0.09
        else:
            carve(px, py, fp, PR, tflat, toffs, tsizes, buf, tmp, res)
            depth_n = res[0]
            cov_o = res[1]
            cov_n = res[3]
            nx = res[4]
            ny = res[5]
            pale = 1.0 + 0.45 * clamp(depth_n * 1.5, 0.0, 1.0)
            ar *= pale
            ag *= pale
            ab *= pale
            nx += 0.05 * fbm2(px * 40.0, py * 40.0, 47, 3, 2.2, 0.5, fp * 40.0)
            ny += 0.05 * fbm2(px * 40.0 + 7.7, py * 40.0, 48, 3, 2.2, 0.5, fp * 40.0)
            if r > TABLE_R - 0.06:
                b = (r - (TABLE_R - 0.06)) / 0.06
                nx += 1.4 * b * b * px / r
                ny += 1.4 * b * b * py / r
        l = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= l
        ny /= l
        nz /= l
        spill = 0.0
        if r > BAND_LO and r < BAND_HI:
            ls = oath_lit_s(r, th, PR, OANG)
            if ls > 0.0:
                tex_sample(tflat, toffs, tsizes, PR[P_TEXR], px, py, 4.6, tmp, buf)
                spill = tmp[1] * ls
        ao = 1.0 - 0.4 * depth_n
        cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, 0.0, False, -1,
                              PR, TL, F, nf, S, ns, igc, igf, ao)
        if spill > 0.0:
            s = spill * PR[P_EMO] * 1.3
            cr += ar * s * GOLD[0]
            cg += ag * s * GOLD[1]
            cb += ab * s * GOLD[2]
        if cov_o > 0.001:
            lit, hot = oath_emission(r, th, depth_n, PR, OANG)
            if lit > 0.0 or hot > 0.0:
                pool = 0.45 + 0.55 * clamp(depth_n * 1.4, 0.0, 1.0)
                shim = 1.0 + PR[P_SHIM] * (vnoise2(th * 60.0 + T * 0.13, r * 40.0, 49) - 0.5)
                em = cov_o * lit * pool * PR[P_EMO] * shim
                ehh = cov_o * hot * PR[P_EMHOT]
                er += em * GOLD[0] + ehh * PALE[0]
                eg += em * GOLD[1] + ehh * PALE[1]
                eb += em * GOLD[2] + ehh * PALE[2]
        if cov_n > 0.001:
            e = 0.0
            if r < RAY_R[1] + 0.05:
                e = PR[P_ORNR]
            elif r < BAND_HI:
                a = (math.pi / 2 - th) % (2.0 * math.pi)
                e = sstep(a - 0.2, a + 0.2, PR[P_ORNB] * 2.0 * math.pi) * 0.7
            if e > 0.0:
                em = cov_n * e * PR[P_EMO] * (0.5 + 0.5 * clamp(depth_n * 1.5, 0.0, 1.0))
                er += em * GOLD[0]
                eg += em * GOLD[1]
                eb += em * GOLD[2]
    elif idb >= 100:
        j = idb - 100
        h = max(0.002, fp * 0.5)
        nx, ny, nz = normal_stone(px, py, pz, S, j, h)
        seed = S[j, S_SEED]
        a = S[j, S_ALB]
        ln = fbm2(px * 3.0 + seed, py * 3.0 + pz * 2.0, 51, 4, 2.1, 0.55, fp * 3.0)
        g = 1.0 + 0.6 * ln + 0.25 * fbm2(px * 25.0 + pz * 11.0, py * 25.0, 52, 3, 2.2, 0.5, fp * 25.0)
        ar = a * g
        ag = a * 0.97 * g
        ab = a * 0.92 * g
        lich = sstep(0.1, 0.3, ln)
        ar = mix(ar, a * 1.2, lich * 0.5)
        ag = mix(ag, a * 1.2, lich * 0.5)
        ab = mix(ab, a * 0.8, lich * 0.5)
        moss = sstep(0.5, 0.0, pz)
        ar = mix(ar, 0.02, moss * 0.7)
        ag = mix(ag, 0.03, moss * 0.7)
        ab = mix(ab, 0.018, moss * 0.7)
        cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, 0.0, False, -1,
                              PR, TL, F, nf, S, ns, igc, igf, 1.0)
    else:
        i = idb - 10
        h = max(0.0015, fp * 0.5)
        nx, ny, nz = normal_fig(px, py, pz, F, i, h)
        ar = F[i, F_R]
        ag = F[i, F_G]
        ab = F[i, F_B]
        sheen = PR[P_SHEEN]
        if mat == 1:
            ar = 0.03
            ag = 0.024
            ab = 0.02
            sheen = 0.0
            if F[i, F_TLIT] > 0.0:
                c = F[i, F_TLIT] * sstep(0.3, 0.9, vnoise2(px * 80.0, py * 80.0 + pz * 50.0, 53))
                er += 2.0 * c
                eg += 0.5 * c
                eb += 0.06 * c
        elif mat == 2:
            ar = 0.10
            ag = 0.066
            ab = 0.046
            sheen *= 0.3
        elif mat == 3:
            ar *= 0.15
            ag *= 0.15
            ab *= 0.15
            sheen *= 0.2
        elif mat == 4:
            ar = 0.16
            ag = 0.125
            ab = 0.08
        elif mat == 5:
            ar = 0.02
            ag = 0.017
            ab = 0.014
        cn = 1.0 + 0.25 * fbm2(px * 30.0 + pz * 7.0, py * 30.0 - pz * 5.0, 55, 3, 2.2, 0.5, fp * 30.0)
        ar *= cn
        ag *= cn
        ab *= cn
        cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, sheen, False, i,
                              PR, TL, F, nf, S, ns, igc, igf, 1.0)
    return cr + er, cg + eg, cb + eb, tbest, idb


@njit(parallel=True, **FM)
def render_surfaces(Wd, Hd, cam, PR, TL, F, nf, S, ns, tflat, toffs, tsizes, pgc, pgf, igc, igf, OANG,
                    rgb, depth, oid, aa_pass, mask, nsub):
    Cx, Cy, Cz = cam[0], cam[1], cam[2]
    Rx, Ry, Rz = cam[3], cam[4], cam[5]
    Ux, Uy, Uz = cam[6], cam[7], cam[8]
    Fx, Fy, Fz = cam[9], cam[10], cam[11]
    f = cam[12]
    cx = cam[13]
    cy = cam[14]
    pix = 1.0 / f
    for y in prange(Hd):
        buf = np.zeros(4)
        tmp = np.zeros(4)
        res = np.zeros(6)
        for x in range(Wd):
            if aa_pass and not mask[y, x]:
                continue
            ns_ = 1 if not aa_pass else nsub
            ar = 0.0
            ag = 0.0
            ab = 0.0
            for s in range(ns_):
                if not aa_pass:
                    jx = 0.0
                    jy = 0.0
                else:
                    jx = ((s + 1) * 0.7548776662 + 0.25) % 1.0 - 0.5
                    jy = ((s + 1) * 0.5698402910 + 0.61) % 1.0 - 0.5
                sx = (x + 0.5 + jx - cx) / f
                sy = -(y + 0.5 + jy - cy) / f
                dx = Fx + sx * Rx + sy * Ux
                dy = Fy + sx * Ry + sy * Uy
                dz = Fz + sx * Rz + sy * Uz
                l = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= l
                dy /= l
                dz /= l
                r, g, b, t, i = shade_sample(Cx, Cy, Cz, dx, dy, dz, pix, PR, TL, F, nf, S, ns,
                                             tflat, toffs, tsizes, pgc, pgf, igc, igf, OANG, buf, tmp, res)
                ar += r
                ag += g
                ab += b
                if not aa_pass:
                    depth[y, x] = t
                    oid[y, x] = i
            if not aa_pass:
                rgb[y, x, 0] = ar
                rgb[y, x, 1] = ag
                rgb[y, x, 2] = ab
            else:
                w0 = 1.0 / (ns_ + 1)
                rgb[y, x, 0] = (rgb[y, x, 0] + ar) * w0
                rgb[y, x, 1] = (rgb[y, x, 1] + ag) * w0
                rgb[y, x, 2] = (rgb[y, x, 2] + ab) * w0
