"""ACCORD v3 surfaces (numba): primary visibility + shading of the council at night.

Lights LT (n, 8): x, y, z, radius, r, g, b, owner (figure index whose torch it is, or -1).
Occluders OC (n, 8): capsules for soft shadows (geom3.soft_vis).
"""
import math

import numpy as np
from numba import njit, prange

from nbcore import FM, clamp, sstep, mix, vnoise2, fbm2, grid_sample, sd_capsule_r
from geom import trace_stone, normal_stone
import geom3 as G
from geom3 import (sd_flat, sd_hearth, sd_fig, trace_fig, normal_fig, sd_hand, trace_hand, normal_hand,
                   sd_ring, trace_ring, ring_local, soft_vis, ray_aabb,
                   STONE_TOP, STONE_BOT, ASH_R0, ASH_R1, KERB_R, HEARTH_ZMAX,
                   F_BB, F_R, F_G, F_B, F_R2, F_G2, F_B2, F_TLIT, F_SEED, F_WEAVE, F_SHEENK, F_X, F_Y,
                   M_CLOTH, M_TORCH, M_GLOVE, M_SHADOW, M_CLOTH2, M_HAIR, M_SHAWL, M_WOOD, M_GILT,
                   S_BB, S_ALB, S_SEED, H_KERB, H_LOG, H_CHAR, R_MID, R_WIDTH)

# ------------------------------------------------------------- param idx ---
P_T = 0
P_MDX, P_MDY, P_MDZ, P_MI, P_MCR, P_MCG, P_MCB = 1, 2, 3, 4, 5, 6, 7
P_SKI, P_SKR, P_SKG, P_SKB = 8, 9, 10, 11
P_NL = 12                      # number of lights in LT
P_OWNK = 13                    # a bearer's own torch on its bearer (fraction)
P_EMB = 14                     # embers in the laid fuel and the ash (0..1)
P_EMBW = 15                    # white-hot of the bed
P_CROWD = 16                   # crowd torches beyond the stones: warm back-rim on the figures
P_WI = 17                      # walker torchlight on the ground
P_IGC_X0, P_IGC_CELL, P_IGF_X0, P_IGF_CELL = 18, 19, 20, 21
P_SHEEN = 22
P_FIGRIM = 23                  # firelit rim on the figures from the hearth fire
P_FIRE_Z = 24                  # the hearth fire's light centre height
P_NHAND = 25
P_RGLOW = 26                   # the Ring's letters
P_NOC = 27
P_FIRE_I = 28                  # the hearth fire's strength (for the Ring's environment)
P_ASHG = 29                    # ash glow radius progress
P_COAL = 30                    # bar 70: the bed of coals on the stone's top under the fire that remains
P_NPARAM = 32

GOLD = (1.0, 0.72, 0.30)


@njit(**FM)
def light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, sheen, is_ground, skip,
             PR, LT, OC, igc, igf, ao):
    """Diffuse (+ cloth sheen) lighting of a surface point. Returns rgb radiance."""
    er = 0.0
    eg = 0.0
    eb = 0.0
    nl = int(PR[P_NL])
    noc = int(PR[P_NOC])
    for k in range(nl):
        lx = LT[k, 0] - px
        ly = LT[k, 1] - py
        lz = LT[k, 2] - pz
        d2 = lx * lx + ly * ly + lz * lz
        d = math.sqrt(d2)
        ndl = (nx * lx + ny * ly + nz * lz) / d
        rl = LT[k, 3]
        fall = 1.0 / (d2 + rl * rl + 0.02)
        I = max(LT[k, 4], max(LT[k, 5], LT[k, 6])) * fall
        if I < 0.0015:
            continue
        f = 0.0
        if ndl > 0.0:
            if is_ground:
                ndl = ndl ** 0.7
            f = ndl
        if sheen > 0.0:
            nv = abs(nx * vx + ny * vy + nz * vz)
            f += sheen * 0.6 * (1.0 - nv) ** 2.5 * max(ndl + 0.3, 0.0)
        if f <= 0.0:
            continue
        own = LT[k, 7]
        if own >= 0.0 and own == skip:
            f *= PR[P_OWNK]
        vis = soft_vis(px, py, pz, LT[k, 0], LT[k, 1], LT[k, 2], rl, OC, noc, skip)
        f *= fall * vis
        er += f * LT[k, 4]
        eg += f * LT[k, 5]
        eb += f * LT[k, 6]
    # the moon
    mdx = PR[P_MDX]
    mdy = PR[P_MDY]
    mdz = PR[P_MDZ]
    ndm = nx * mdx + ny * mdy + nz * mdz
    if ndm > 0.0 and PR[P_MI] > 0.0:
        mv = soft_vis(px, py, pz, px + mdx * 40.0, py + mdy * 40.0, pz + mdz * 40.0, 0.8, OC, noc, skip)
        e = PR[P_MI] * ndm * mv
        er += e * PR[P_MCR]
        eg += e * PR[P_MCG]
        eb += e * PR[P_MCB]
    # the night sky
    sk = (0.55 + 0.45 * nz) * ao
    er += sk * PR[P_SKI] * PR[P_SKR]
    eg += sk * PR[P_SKI] * PR[P_SKG]
    eb += sk * PR[P_SKI] * PR[P_SKB]
    # the crowd's torches on the plain
    if pz < 1.0 and PR[P_WI] > 0.0:
        w = grid_sample(igf, PR[P_IGF_X0], PR[P_IGF_X0], PR[P_IGF_CELL], px, py)
        if w == 0.0:
            w = grid_sample(igc, PR[P_IGC_X0], PR[P_IGC_X0], PR[P_IGC_CELL], px, py)
        w *= PR[P_WI] * max(nz, 0.2)
        er += w * 1.0
        eg += w * 0.56
        eb += w * 0.16
    return ar * er * ao, ag * eg * ao, ab * eb * ao


@njit(**FM)
def spec_at(px, py, pz, nx, ny, nz, vx, vy, vz, expo, PR, LT, OC, skip):
    """Blinn-Phong highlights from every light (shadowed), for metals and leather."""
    er = 0.0
    eg = 0.0
    eb = 0.0
    nl = int(PR[P_NL])
    noc = int(PR[P_NOC])
    norm = (expo + 8.0) / 25.0
    for k in range(nl):
        lx = LT[k, 0] - px
        ly = LT[k, 1] - py
        lz = LT[k, 2] - pz
        d2 = lx * lx + ly * ly + lz * lz
        d = math.sqrt(d2)
        lx /= d
        ly /= d
        lz /= d
        if nx * lx + ny * ly + nz * lz <= 0.0:
            continue
        hx = lx + vx
        hy = ly + vy
        hz = lz + vz
        hl = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
        ndh = max((nx * hx + ny * hy + nz * hz) / hl, 0.0)
        s = norm * ndh ** expo / (d2 + LT[k, 3] ** 2 + 0.02)
        if s < 1e-4:
            continue
        vis = soft_vis(px, py, pz, LT[k, 0], LT[k, 1], LT[k, 2], LT[k, 3], OC, int(PR[P_NOC]), skip)
        er += s * vis * LT[k, 4]
        eg += s * vis * LT[k, 5]
        eb += s * vis * LT[k, 6]
    return er, eg, eb


@njit(**FM)
def ground_albedo(x, y, fp, PR):
    """The council ground: the ash bed, the scorched kerb ring, trampled turf inside the stones, heath beyond."""
    r = math.sqrt(x * x + y * y)
    T = PR[P_T]
    n2 = fbm2(x * 0.33, y * 0.33, 12, 5, 2.2, 0.5, fp * 0.33)
    n3 = fbm2(x * 2.7, y * 2.7, 13, 4, 2.3, 0.5, fp * 2.7)
    n4 = fbm2(x * 13.0, y * 13.0, 14, 3, 2.2, 0.55, fp * 13.0)
    # heath and grass
    n1 = fbm2(x * 0.018, y * 0.018, 11, 5, 2.1, 0.55, fp * 0.018)
    g = 1.0 + 1.3 * n1 + 0.9 * n2 + 0.6 * n3
    ar = 0.040 * g
    ag = 0.046 * g
    ab = 0.036 * g
    # inside the stones: trampled earth and worn turf
    tr = sstep(10.5, 7.0, r) * clamp(0.55 + 0.9 * (n2 + 0.4) + 0.5 * n3, 0.0, 1.0)
    ar = mix(ar, 0.052 * (1 + 0.6 * n3 + 0.4 * n4), tr)
    ag = mix(ag, 0.045 * (1 + 0.6 * n3 + 0.4 * n4), tr)
    ab = mix(ab, 0.037 * (1 + 0.6 * n3 + 0.4 * n4), tr)
    # worn turf in patches between the bare, trodden earth (most worn where the council stands)
    wear = sstep(2.6, 1.9, r) * sstep(0.9, 1.3, r)
    turf = sstep(-0.04, 0.14, fbm2(x * 1.6 + 7.0, y * 1.6, 23, 3, 2.1, 0.5, fp * 1.6) + 0.10 * n4) * (1.0 - 0.8 * wear)
    tf = 1.0 + 0.45 * fbm2(x * 40.0, y * 40.0, 24, 2, 2.1, 0.5, fp * 40.0)
    ar = mix(ar, 0.030 * tf, turf * 0.8)
    ag = mix(ag, 0.033 * tf, turf * 0.8)
    ab = mix(ab, 0.022 * tf, turf * 0.8)
    # small pebbles in the trodden earth
    pb = sstep(0.13, 0.19, fbm2(x * 28.0 + 3.0, y * 28.0, 25, 2, 2.0, 0.5, fp * 28.0)) * (1.0 - turf)
    ar = mix(ar, 0.080, pb * 0.6)
    ag = mix(ag, 0.075, pb * 0.6)
    ab = mix(ab, 0.068, pb * 0.6)
    # scorched round the hearth
    sc = sstep(KERB_R + 0.40, KERB_R, r) * (0.6 + 0.4 * n3)
    ar = mix(ar, 0.036, sc * 0.7)
    ag = mix(ag, 0.032, sc * 0.7)
    ab = mix(ab, 0.029, sc * 0.7)
    ash = 0.0
    if r < ASH_R1 + 0.12:
        # fine grey wood ash with a ragged edge: palest at the rim where it is oldest, sootier toward the stone
        ash = sstep(ASH_R1 + 0.09, ASH_R1 - 0.04, r + 0.06 * n3)
        a = 0.036 * (1.0 + 0.35 * n3 + 0.22 * n4) * (0.72 + 0.40 * sstep(ASH_R0, ASH_R1, r))
        # charcoal dust worked into it in soft drifts (never blots)
        dust = sstep(-0.10, 0.20, fbm2(x * 5.0 + 3.1, y * 5.0, 15, 3, 2.1, 0.5, fp * 5.0))
        a *= 1.0 - 0.45 * dust
        # fine specks of char
        spk = fbm2(x * 55.0, y * 55.0 + 1.3, 20, 2, 2.1, 0.5, fp * 55.0)
        a *= 1.0 - 0.30 * sstep(0.08, 0.20, spk)
        a *= 1.0 - 0.55 * PR[P_COAL]
        ar = mix(ar, a * 1.00, ash)
        ag = mix(ag, a * 0.98, ash)
        ab = mix(ab, a * 0.95, ash)
    return ar, ag, ab, ash


@njit(**FM)
def bed_embers(x, y, fp, PR):
    """Glow of the bed under the fire (P2 after the catch, P3): mottled, crawling, hottest near the stone."""
    e = PR[P_EMB]
    if e <= 0.0:
        return 0.0, 0.0, 0.0
    r = math.sqrt(x * x + y * y)
    if r > ASH_R1 + 0.08:
        return 0.0, 0.0, 0.0
    T = PR[P_T]
    ring = sstep(ASH_R1 + 0.06, ASH_R1 - 0.08, r) * sstep(ASH_R0 - 0.05, ASH_R0 + 0.05, r)
    ring *= sstep(0.0, 0.2, PR[P_ASHG] * 1.2 - (r - ASH_R0) / (ASH_R1 - ASH_R0) * 0.2)
    c1 = vnoise2(x * 11.0 + T * 0.012, y * 11.0 - T * 0.009, 401)
    c2 = vnoise2(x * 4.0 - T * 0.006, y * 4.0, 402)
    k = sstep(0.35, 0.85, 0.55 * c1 + 0.6 * c2) * ring * e
    wh = PR[P_EMBW]
    if PR[P_COAL] > 0.0:
        # bar 70: the old bed has burned down: soft deep-red patches, strongest toward the stone
        k = sstep(0.55, 0.95, 0.35 * c1 + 0.8 * c2) * ring * e * (0.25 + 0.75 * sstep(ASH_R1, ASH_R0, r))
        return 0.9 * k, 0.9 * k * 0.18, 0.9 * k * 0.02
    return 2.2 * k * (1.0 + 2.0 * wh), 2.2 * k * (0.24 + 0.45 * wh), 2.2 * k * (0.03 + 0.25 * wh)


@njit(**FM)
def shade_ring(px, py, pz, vx, vy, vz, fp, PR, LT, OC, RP, stex, ssz):
    """The canonical Ring: polished gold catching the torches; its letters only in fire."""
    h = 0.00012
    nx = sd_ring(px + h, py, pz, RP) - sd_ring(px - h, py, pz, RP)
    ny = sd_ring(px, py + h, pz, RP) - sd_ring(px, py - h, pz, RP)
    nz = sd_ring(px, py, pz + h, RP) - sd_ring(px, py, pz - h, RP)
    l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    nx /= l
    ny /= l
    nz /= l
    f0r, f0g, f0b = 1.0, 0.766, 0.336
    ndv = max(nx * vx + ny * vy + nz * vz, 0.0)
    fres = (1.0 - ndv) ** 5
    fr = f0r + (1.0 - f0r) * fres
    fg = f0g + (1.0 - f0g) * fres
    fb = f0b + (1.0 - f0b) * fres
    rx = 2.0 * ndv * nx - vx
    ry = 2.0 * ndv * ny - vy
    rz = 2.0 * ndv * nz - vz
    # the environment: the night sky above, the lit ash and stone below, the fire when it burns
    up = sstep(-0.1, 0.6, rz)
    fi = PR[P_FIRE_I]
    env_r = 0.004 * up + (1.0 - up) * (0.020 + 0.9 * fi)
    env_g = 0.005 * up + (1.0 - up) * (0.013 + 0.55 * fi)
    env_b = 0.009 * up + (1.0 - up) * (0.008 + 0.18 * fi)
    env_r += up * 2.2 * fi
    env_g += up * 1.3 * fi
    env_b += up * 0.35 * fi
    sr, sg, sb = spec_at(px, py, pz, nx, ny, nz, vx, vy, vz, 140.0, PR, LT, OC, -1.0)
    cr = (env_r + sr * 9.0) * fr
    cg = (env_g + sg * 9.0) * fg
    cb = (env_b + sb * 9.0) * fb
    g = PR[P_RGLOW]
    if g > 0.0:
        rho, zz, _, _, _ = ring_local(px, py, pz, RP)
        if rho > R_MID:
            qx = px - RP[1]
            qy = py - RP[2]
            ang = math.atan2(qy, qx) / (2.0 * math.pi)
            u = ang % 1.0
            v = 0.5 - zz / R_WIDTH
            lvl = 0
            texel = 2.0 * math.pi * rho / ssz[0, 1]
            while lvl < ssz.shape[0] - 1 and texel * 2.0 < fp:
                texel *= 2.0
                lvl += 1
            c = strip_sample(stex, ssz, u, v, lvl)
            e = c * g * 18.0
            cr += e * 1.0
            cg += e * 0.30
            cb += e * 0.05
    return cr, cg, cb


@njit(**FM)
def strip_sample(stex, ssz, u, v, lvl):
    off = 0
    for i in range(lvl):
        off += ssz[i, 0] * ssz[i, 1]
    h = ssz[lvl, 0]
    w = ssz[lvl, 1]
    x = (u % 1.0) * w - 0.5
    y = v * h - 0.5
    if y < 0.0 or y > h - 1.0:
        return 0.0
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    x0 = ix % w
    x1 = (ix + 1) % w
    y1 = min(iy + 1, h - 1)
    a = stex[off + iy * w + x0] * (1 - fx) + stex[off + iy * w + x1] * fx
    b = stex[off + y1 * w + x0] * (1 - fx) + stex[off + y1 * w + x1] * fx
    return a * (1 - fy) + b * fy


@njit(**FM)
def cloth_ao(px, py, pz, nx, ny, nz, F, i):
    occ = 0.0
    for k in range(3):
        dd = 0.012 + 0.03 * k
        sd_, m_ = sd_fig(px + nx * dd, py + ny * dd, pz + nz * dd, F, i)
        occ += (dd - sd_) / dd * (0.5 ** k)
    return clamp(1.0 - 0.55 * occ, 0.25, 1.0)


@njit(**FM)
def shade_leather(px, py, pz, nx, ny, nz, vx, vy, vz, u, v, w, gilt, fp, PR, LT, OC, igc, igf, skip, ao):
    """Thin dark leather: a soft sheen that rakes along the fingers, the grain, the three stitched points on the
    back of the hand. Or (gilt) the crust: set gold is a metal, dark but for what it reflects (the torch over
    it, the fire), lumpy and cracked, a few cracks still holding an ember's glow."""
    if gilt > 0.0:
        hr, hg, hb = spec_at(px, py, pz, nx, ny, nz, vx, vy, vz, 90.0, PR, LT, OC, skip)
        br, bg, bb = spec_at(px, py, pz, nx, ny, nz, vx, vy, vz, 12.0, PR, LT, OC, skip)
        n2 = vnoise2(u * 260.0, v * 190.0 + w * 150.0, 94)
        g = 2.4 * (0.75 + 0.5 * n2)
        # schlick toward white at grazing, the warm dark it mirrors below
        ndv = max(nx * vx + ny * vy + nz * vz, 0.0)
        fr = 0.25 * (1.0 - ndv) ** 5
        cr = (hr * g + br * 0.55) * (1.00 * (1.0 - fr) + fr) + 0.012 * ao
        cg = (hg * g + bg * 0.55) * (0.74 * (1.0 - fr) + fr) + 0.0085 * ao
        cb = (hb * g + bb * 0.55) * (0.32 * (1.0 - fr) + fr) + 0.0030 * ao
        cn = vnoise2(u * 170.0 + 7.0, v * 170.0 + w * 90.0, 95)
        cracks = sstep(0.030, 0.0, abs(cn - 0.5))
        live = sstep(0.52, 0.78, vnoise2(u * 40.0, v * 40.0 + 2.0, 96))
        cr *= 1.0 - 0.85 * cracks
        cg *= 1.0 - 0.85 * cracks
        cb *= 1.0 - 0.85 * cracks
        pul = 0.8 + 0.2 * math.sin(PR[P_T] * 0.21 + u * 60.0)
        e = cracks * live * 1.5 * pul
        return cr + e * 1.0, cg + e * 0.24, cb + e * 0.035
    ar = 0.0115
    ag = 0.0092
    ab = 0.0082
    wr = fbm2(u * 900.0 + w * 300.0, v * 900.0, 91, 3, 2.1, 0.5, fp * 900.0)
    gk = 1.0 + 0.30 * wr
    # the three points: stitched seams between the knuckles and the wrist on the back
    if w > 0.004 and u > 0.030 and u < 0.088:
        for kk in range(3):
            sv = -0.016 + 0.016 * kk
            q = abs(v - sv) / 0.0011
            if q < 2.5:
                gk *= 1.0 - 0.45 * math.exp(-q * q)
    ar *= gk
    ag *= gk
    ab *= gk
    cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, 0.5, False, skip, PR, LT, OC,
                          igc, igf, ao)
    sr, sg, sb = spec_at(px, py, pz, nx, ny, nz, vx, vy, vz, 16.0, PR, LT, OC, skip)
    ndv = max(nx * vx + ny * vy + nz * vz, 0.0)
    k = (0.07 + 0.20 * (1.0 - ndv) ** 3) * ao
    return cr + sr * k, cg + sg * k * 0.92, cb + sb * k * 0.85


@njit(**FM)
def shade_sample(ox, oy, oz, dx, dy, dz, pix, PR, LT, OC, F, nf, S, ns, KB, LG, nlog, CH, nch,
                 HDs, Js, HBB, RP, stex, ssz, igc, igf):
    """Trace + shade one ray. Returns (r, g, b, t, id)."""
    tbest = 1e30
    idb = -1
    mat = 0
    part = 0
    if dz < 0.0:
        tbest = -oz / dz
        idb = 0
    idx = 1.0 / dx if abs(dx) > 1e-9 else 1e9
    idy = 1.0 / dy if abs(dy) > 1e-9 else 1e9
    idz = 1.0 / dz if abs(dz) > 1e-9 else 1e9
    # the flat stone
    t0, t1 = ray_aabb(ox, oy, oz, idx, idy, idz, -0.46, 0.46, -0.46, 0.46, STONE_BOT, STONE_TOP + 0.03)
    if t1 >= max(t0, 0.0) and t0 < tbest:
        t = max(t0, 0.0)
        for it in range(80):
            d = sd_flat(ox + dx * t, oy + dy * t, oz + dz * t)
            if d < max(0.0005, t * pix * 0.3):
                if t < tbest:
                    tbest = t
                    idb = 1
                break
            t += d * 0.85
            if t > min(t1, tbest):
                break
    # the cold hearth
    t0, t1 = ray_aabb(ox, oy, oz, idx, idy, idz, -KERB_R - 0.22, KERB_R + 0.22, -KERB_R - 0.22, KERB_R + 0.22,
                      -0.02, HEARTH_ZMAX)
    if t1 >= max(t0, 0.0) and t0 < tbest:
        t = max(t0, 0.0)
        for it in range(140):
            d, pt = sd_hearth(ox + dx * t, oy + dy * t, oz + dz * t, KB, LG, nlog, CH, nch)
            if d < max(0.0004, t * pix * 0.25):
                if t < tbest:
                    tbest = t
                    idb = 2
                    part = pt
                break
            t += d * 0.62
            if t > min(t1, tbest):
                break
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
    nh = int(PR[P_NHAND])
    for k in range(nh):
        t0, t1 = ray_aabb(ox, oy, oz, idx, idy, idz, HBB[k, 0], HBB[k, 1], HBB[k, 2], HBB[k, 3], HBB[k, 4],
                          HBB[k, 5])
        if t1 < max(t0, 0.0) or t0 > tbest:
            continue
        th, m = trace_hand(ox, oy, oz, dx, dy, dz, t0, min(t1, tbest), HDs[k], Js[k], pix)
        if th > 0.0 and th < tbest:
            tbest = th
            idb = 200 + k
            mat = m
    if RP[0] > 0.0:
        tr = trace_ring(ox, oy, oz, dx, dy, dz, RP, tbest)
        if tr > 0.0 and tr < tbest:
            tbest = tr
            idb = 5
    if idb < 0:
        return 0.0, 0.0, 0.0, 1e6, -1

    px = ox + dx * tbest
    py = oy + dy * tbest
    pz = oz + dz * tbest
    fp = tbest * pix
    vx = -dx
    vy = -dy
    vz = -dz
    T = PR[P_T]
    if idb == 0:
        ar, ag, ab, ash = ground_albedo(px, py, fp, PR)
        nx = -0.08 * fbm2(px * 2.7 + 5.1, py * 2.7, 16, 3, 2.2, 0.5, fp * 2.7)
        ny = -0.08 * fbm2(px * 2.7, py * 2.7 + 3.3, 17, 3, 2.2, 0.5, fp * 2.7)
        # the relief of trampled turf and earth: tussocks and pebbles, stronger away from the hearth
        rel = (1.0 - ash) * (0.55 + 0.45 * sstep(1.2, 2.2, math.sqrt(px * px + py * py)))
        nx -= 0.55 * rel * fbm2(px * 22.0 + 1.7, py * 22.0, 21, 3, 2.1, 0.55, fp * 22.0)
        ny -= 0.55 * rel * fbm2(px * 22.0, py * 22.0 + 6.1, 22, 3, 2.1, 0.55, fp * 22.0)
        nx -= 0.12 * ash * fbm2(px * 30.0, py * 30.0, 18, 3, 2.1, 0.5, fp * 30.0)
        ny -= 0.12 * ash * fbm2(px * 30.0 + 4.0, py * 30.0, 19, 3, 2.1, 0.5, fp * 30.0)
        l = math.sqrt(nx * nx + ny * ny + 1.0)
        nx /= l
        ny /= l
        nz = 1.0 / l
        # contact darkening at the feet of figures, stones and round the flat stone
        ao = 1.0
        for i in range(nf):
            ddx = px - F[i, F_X]
            ddy = py - F[i, F_Y]
            dd = math.sqrt(ddx * ddx + ddy * ddy)
            if dd < 0.9:
                ao *= 0.5 + 0.5 * sstep(0.18, 0.75, dd)
        rr = math.sqrt(px * px + py * py)
        ao *= 0.55 + 0.45 * sstep(G.STONE_R * 0.90, G.STONE_R + 0.14, rr)
        if rr < KERB_R + 0.26:
            occ = 0.0
            for kk in range(3):
                hh_ = 0.008 + 0.018 * kk
                sd_, pt_ = sd_hearth(px, py, hh_, KB, LG, nlog, CH, nch)
                if pt_ == 0:
                    continue
                occ += clamp((hh_ - sd_) / hh_, 0.0, 1.0) * (0.55 ** kk)
            ao *= clamp(1.0 - 0.55 * occ, 0.25, 1.0)
        cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, 0.0, True, -1.0,
                              PR, LT, OC, igc, igf, ao)
        er, eg, eb = bed_embers(px, py, fp, PR)
        return cr + er, cg + eg, cb + eb, tbest, idb
    if idb == 1:
        h = max(0.0015, fp * 0.5)
        k1 = sd_flat(px + h, py - h, pz - h)
        k2 = sd_flat(px - h, py - h, pz + h)
        k3 = sd_flat(px - h, py + h, pz - h)
        k4 = sd_flat(px + h, py + h, pz + h)
        nx = k1 - k2 - k3 + k4
        ny = -k1 - k2 + k3 + k4
        nz = -k1 + k2 - k3 + k4
        l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= l
        ny /= l
        nz /= l
        # a dark granite: pale feldspar specks and dark mica in a grey ground, weather pits on the top,
        # a pale lichen crust on the rim, soot low on its sides; the top worn smooth where hands have rested
        ln = fbm2(px * 7.0, py * 7.0 + pz * 5.0, 81, 4, 2.1, 0.55, fp * 7.0)
        gr = fbm2(px * 130.0, py * 130.0 + pz * 95.0, 82, 2, 2.3, 0.6, fp * 130.0)
        rw_ = math.sqrt(px * px + py * py)
        a = 0.032 * (1.0 + 0.55 * ln) * (1.0 + 0.45 * sstep(0.30, 0.05, rw_) * sstep(STONE_TOP - 0.02, STONE_TOP, pz))
        crk = sstep(0.008, 0.0, abs(0.62 * px + 0.78 * py - 0.05 + 0.03 * math.sin(9.0 * px - 4.0 * py))) * sstep(STONE_TOP - 0.04, STONE_TOP - 0.01, pz)
        a *= 1.0 - 0.45 * crk
        a *= 1.0 + 0.10 * sstep(0.05, 0.16, gr) - 0.06 * sstep(-0.05, -0.16, gr)
        a *= 0.62 + 0.38 * sstep(STONE_TOP - 0.05, STONE_TOP - 0.015, pz)
        pit = 0.0
        ar = a * 1.03
        ag = a * 1.00
        ab = a * 0.95
        rim = sstep(-0.05, -0.005, G._flat2d(px, py)) * sstep(STONE_TOP - 0.10, STONE_TOP - 0.02, pz)
        lich = sstep(0.10, 0.24, fbm2(px * 11.0 + 2.0, py * 11.0, 83, 4, 2.0, 0.5, fp * 11.0)) * rim
        ar = mix(ar, 0.040, lich * 0.25)
        ag = mix(ag, 0.040, lich * 0.25)
        ab = mix(ab, 0.036, lich * 0.25)
        soot = sstep(0.20, 0.02, pz) * (1.0 - 0.3 * nz)
        # after the fire has burned on it (P2 from the catch, bar 70) its top is fire-blackened
        soot = max(soot, 0.8 * PR[P_EMB] * sstep(0.35, 0.65, vnoise2(px * 9.0, py * 9.0, 98) + 0.3))
        ar = mix(ar, 0.016, soot * 0.9)
        ag = mix(ag, 0.014, soot * 0.9)
        ab = mix(ab, 0.013, soot * 0.9)
        # bar 70: the coals heaped on its top where the Ring was, crusted black, glowing through in the seams
        ce_r = 0.0
        ce_g = 0.0
        ce_b = 0.0
        co = PR[P_COAL]
        if co > 0.0:
            rr_ = math.sqrt(px * px + py * py)
            bed = sstep(0.23, 0.13, rr_ + 0.05 * fbm2(px * 9.0, py * 9.0, 88, 3, 2.1, 0.5, fp * 9.0)) \
                * sstep(STONE_TOP - 0.06, STONE_TOP - 0.025, pz)
            if bed > 0.0:
                c1 = fbm2(px * 26.0 + T * 0.003, py * 26.0, 89, 4, 2.2, 0.55, fp * 26.0)
                c2 = fbm2(px * 9.0 - T * 0.004, py * 9.0 + 3.0, 90, 3, 2.1, 0.5, fp * 9.0)
                heat = (0.45 + 0.55 * sstep(0.20, 0.0, rr_)) * (0.70 + 0.60 * c2)
                # the coals glow through where the black crust has broken (soft patches)
                open_ = sstep(-0.02, 0.22, c1 + 0.4 * c2)
                glow = bed * co * heat * (0.06 + 0.80 * open_)
                ar = mix(ar, 0.004, bed)
                ag = mix(ag, 0.0036, bed)
                ab = mix(ab, 0.0035, bed)
                ce_r = 2.4 * glow
                ce_g = 2.4 * glow * 0.24
                ce_b = 2.4 * glow * 0.03
        cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, 0.0, False, -1.0,
                              PR, LT, OC, igc, igf, 1.0)
        sr, sg, sb = spec_at(px, py, pz, nx, ny, nz, vx, vy, vz, 22.0, PR, LT, OC, -1.0)
        wk = 0.016 * sstep(0.7, 0.95, nz) * (1.0 - lich)
        return cr + sr * wk + ce_r, cg + sg * wk + ce_g, cb + sb * wk + ce_b, tbest, idb
    if idb == 2:
        h = max(0.0012, fp * 0.5)
        k1, _ = sd_hearth(px + h, py - h, pz - h, KB, LG, nlog, CH, nch)
        k2, _ = sd_hearth(px - h, py - h, pz + h, KB, LG, nlog, CH, nch)
        k3, _ = sd_hearth(px - h, py + h, pz - h, KB, LG, nlog, CH, nch)
        k4, _ = sd_hearth(px + h, py + h, pz + h, KB, LG, nlog, CH, nch)
        nx = k1 - k2 - k3 + k4
        ny = -k1 - k2 + k3 + k4
        nz = -k1 + k2 - k3 + k4
        l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= l
        ny /= l
        nz /= l
        rr = math.sqrt(px * px + py * py)
        er = 0.0
        eg = 0.0
        eb = 0.0
        if part == H_KERB:
            # field stones, grey and lichened outside, sooted black on the side that faced the fire
            sp = fbm2(px * 25.0, py * 25.0 + pz * 20.0, 84, 3, 2.2, 0.55, fp * 25.0)
            a = 0.060 * (1.0 + 0.30 * sp) * (0.75 + 0.5 * vnoise2(px * 6.0 + KB[0, 8], py * 6.0, 90))
            inner = -(nx * px + ny * py) / (rr + 1e-6)
            soot = clamp(0.25 + 0.75 * sstep(-0.2, 0.5, inner) * sstep(KERB_R + 0.10, KERB_R - 0.08, rr), 0.0, 1.0)
            ar = mix(a, 0.015, soot * 0.85)
            ag = mix(a * 0.98, 0.013, soot * 0.85)
            ab = mix(a * 0.93, 0.012, soot * 0.85)
        elif part == H_LOG:
            # which log: its axis gives the bark's furrows (along it) and the char's checks (across it)
            kb_ = 0
            db_ = 1e9
            for kk in range(nlog):
                dd_ = sd_capsule_r(px, py, pz, LG[kk, 0], LG[kk, 1], LG[kk, 2], LG[kk, 3], LG[kk, 4], LG[kk, 5],
                                     LG[kk, 6], LG[kk, 7])
                if dd_ < db_:
                    db_ = dd_
                    kb_ = kk
            axx = LG[kb_, 3] - LG[kb_, 0]
            axy = LG[kb_, 4] - LG[kb_, 1]
            axz = LG[kb_, 5] - LG[kb_, 2]
            al_ = math.sqrt(axx * axx + axy * axy + axz * axz) + 1e-9
            along = ((px - LG[kb_, 0]) * axx + (py - LG[kb_, 1]) * axy + (pz - LG[kb_, 2]) * axz) / al_
            sd2 = (-(px - LG[kb_, 0]) * axy + (py - LG[kb_, 1]) * axx) / al_
            ang = math.atan2(pz - LG[kb_, 2] - along / al_ * axz, sd2)
            # grey-brown bark in long furrows; charred black toward the stone, checked into blocks
            fur = sstep(0.25, 0.75, 0.5 + 0.5 * math.sin(ang * 9.0 + 2.5 * vnoise2(along * 8.0, ang, 85) + LG[kb_, 8]))
            a = 0.030 * (0.55 + 0.45 * fur) * (0.8 + 0.4 * vnoise2(along * 20.0, ang * 2.0, 91))
            ch = sstep(0.76, 0.56, rr + 0.04 * (vnoise2(along * 9.0, ang, 94) - 0.5))
            chk = sstep(0.035, 0.0, abs(vnoise2(along * 38.0 + LG[kb_, 8], ang * 2.2, 95) - 0.5)) \
                + 0.8 * sstep(0.03, 0.0, abs(vnoise2(along * 14.0, ang * 5.0 + LG[kb_, 8], 96) - 0.5))
            cha = 0.016 * (1.0 - 0.75 * min(chk, 1.0))
            crack = min(chk, 1.0) * ch
            ar = mix(a * 1.00, cha, ch)
            ag = mix(a * 0.90, cha * 0.97, ch)
            ab = mix(a * 0.80, cha * 0.95, ch)
            wash = sstep(0.7, 0.95, nz) * sstep(0.62, 0.80, vnoise2(px * 60.0, py * 60.0, 92)) * ch
            ar = mix(ar, 0.030, wash * 0.25)
            ag = mix(ag, 0.029, wash * 0.25)
            ab = mix(ab, 0.028, wash * 0.25)
            eb_ = PR[P_EMB]
            if eb_ > 0.0:
                cr_ = vnoise2(px * 120.0 + T * 0.03, py * 120.0 + pz * 60.0, 86)
                glow = (sstep(0.45, 0.8, cr_) + 0.8 * crack) * eb_ * (0.5 + 0.5 * ch) * sstep(0.0, 0.3, PR[P_ASHG] * 1.2 - (rr - ASH_R0) / (ASH_R1 - ASH_R0) * 0.2)
                pat = sstep(0.40, 0.70, vnoise2(along * 9.0 + LG[kb_, 8], ang * 1.3, 88))
                glow = (0.9 * crack * (0.4 + 0.6 * pat) + 0.25 * pat) * eb_ * ch * sstep(0.80, 0.45, rr)
                wh = PR[P_EMBW]
                er += 2.6 * glow * (1.0 + 1.5 * wh)
                eg += 2.6 * glow * (0.24 + 0.5 * wh)
                eb += 2.6 * glow * (0.03 + 0.3 * wh)
        else:
            # charcoal: black, a faint silvery sheen on its broken faces
            a = 0.011 * (1.0 + 0.5 * vnoise2(px * 200.0, py * 200.0, 93))
            ar = a
            ag = a * 0.98
            ab = a * 0.97
            sr_, sg_, sb_ = spec_at(px, py, pz, nx, ny, nz, vx, vy, vz, 30.0, PR, LT, OC, -1.0)
            er += 0.02 * sr_
            eg += 0.02 * sg_
            eb += 0.02 * sb_
            eb_ = PR[P_EMB]
            if eb_ > 0.0:
                cr_ = vnoise2(px * 60.0 - T * 0.01, py * 60.0 + pz * 40.0, 87)
                lump = vnoise2(px * 18.0, py * 18.0, 97)
                glow = sstep(0.55, 0.85, cr_) * sstep(0.45, 0.7, lump) * eb_ * sstep(0.95, 0.5, rr)
                glow *= 1.0 - 0.75 * PR[P_COAL]
                er += 2.4 * glow
                eg += 2.4 * glow * 0.26
                eb += 2.4 * glow * 0.035
        cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, 0.0, False, -1.0,
                              PR, LT, OC, igc, igf, 1.0)
        return cr + er, cg + eg, cb + eb, tbest, idb
    if idb == 5:
        cr, cg, cb = shade_ring(px, py, pz, vx, vy, vz, fp, PR, LT, OC, RP, stex, ssz)
        return cr, cg, cb, tbest, idb
    if idb >= 200:
        k = idb - 200
        hh = max(0.00035, fp * 0.5)
        nx, ny, nz = normal_hand(px, py, pz, HDs[k], Js[k], hh)
        qx = px - HDs[k, 0]
        qy = py - HDs[k, 1]
        qz = pz - HDs[k, 2]
        s = HDs[k, 12]
        u = (qx * HDs[k, 3] + qy * HDs[k, 4] + qz * HDs[k, 5]) / s
        v = (qx * HDs[k, 6] + qy * HDs[k, 7] + qz * HDs[k, 8]) / s
        w = (qx * HDs[k, 9] + qy * HDs[k, 10] + qz * HDs[k, 11]) / s
        # occlusion between the fingers
        occ = 0.0
        for kk in range(3):
            dd = 0.004 + 0.008 * kk
            sd_, m_ = sd_hand(px + nx * dd, py + ny * dd, pz + nz * dd, HDs[k], Js[k])
            occ += (dd - sd_) / dd * (0.5 ** kk)
        ao = clamp(1.0 - 0.6 * occ, 0.2, 1.0)
        if mat == M_CLOTH:
            wv = 1.0 + 0.2 * fbm2(u * 300.0, v * 300.0 + w * 200.0, 97, 3, 2.2, 0.5, fp * 300.0)
            ar = 0.0085 * wv
            ag = 0.0074 * wv
            ab = 0.0070 * wv
            cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, PR[P_SHEEN], False, -1.0,
                                  PR, LT, OC, igc, igf, ao)
        else:
            gilt = 1.0 if mat == M_GILT else 0.0
            cr, cg, cb = shade_leather(px, py, pz, nx, ny, nz, vx, vy, vz, u, v, w, gilt, fp, PR, LT, OC,
                                       igc, igf, -1.0, ao)
        # the hearth's fire wraps the hand in its own light (rim)
        if PR[P_FIGRIM] > 0.0:
            lx = -px
            ly = -py
            lz = PR[P_FIRE_Z] - pz
            ll = math.sqrt(lx * lx + ly * ly + lz * lz)
            ndl = (nx * lx + ny * ly + nz * lz) / ll
            nv = abs(nx * vx + ny * vy + nz * vz)
            rim = sstep(-0.2, 0.6, ndl) * (1.0 - nv) ** 2.0 * ao
            e = PR[P_FIGRIM] * rim * 0.6
            cr += e * 1.0
            cg += e * 0.55
            cb += e * 0.18
        return cr, cg, cb, tbest, idb
    if idb >= 100:
        j = idb - 100
        h = max(0.002, fp * 0.5)
        nx, ny, nz = normal_stone(px, py, pz, S, j, h)
        seed = S[j, S_SEED]
        a = S[j, S_ALB] * 0.34
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
        cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, 0.0, False, -1.0,
                              PR, LT, OC, igc, igf, 1.0)
        return cr, cg, cb, tbest, idb
    # ---- the figures
    i = idb - 10
    h = max(0.0012, fp * 0.5)
    nx, ny, nz = normal_fig(px, py, pz, F, i, h)
    sheen = PR[P_SHEEN]
    seed = F[i, F_SEED]
    er = 0.0
    eg = 0.0
    eb = 0.0
    ao = cloth_ao(px, py, pz, nx, ny, nz, F, i)
    fi = float(i)
    if mat == M_GLOVE or mat == M_GILT:
        cr, cg, cb = shade_leather(px, py, pz, nx, ny, nz, vx, vy, vz, px, py, pz, 1.0 if mat == M_GILT else 0.0,
                                   fp, PR, LT, OC, igc, igf, fi, ao)
        return cr, cg, cb, tbest, idb
    if mat == M_CLOTH or mat == M_CLOTH2 or mat == M_SHAWL:
        if mat == M_CLOTH:
            ar = F[i, F_R]
            ag = F[i, F_G]
            ab = F[i, F_B]
        else:
            ar = F[i, F_R2]
            ag = F[i, F_G2]
            ab = F[i, F_B2]
        wf = 38.0 * F[i, F_WEAVE]
        wv = 1.0 + (0.14 + 0.10 * F[i, F_WEAVE]) * fbm2(px * wf + pz * 9.0 + seed, py * wf - pz * 7.0, 55, 3,
                                                         2.2, 0.5, fp * wf)
        sheen *= F[i, F_SHEENK]
        if mat == M_SHAWL:
            # woven wool: a soft twill, slubs in the yarn, a few darker threads, and a fuzz that catches light
            tw = 0.5 + 0.5 * math.sin((px + py) * 900.0 + pz * 700.0)
            tw = 1.0 + 0.10 * (tw - 0.5) * sstep(3.0 * fp, 1.0 * fp, 0.002)
            sl = 1.0 + 0.30 * fbm2(px * 160.0 + seed, py * 160.0 - pz * 90.0, 58, 3, 2.2, 0.5, fp * 160.0)
            th_ = 1.0 - 0.25 * sstep(0.25, 0.45, fbm2(px * 45.0, py * 45.0 + pz * 30.0, 59, 2, 2.2, 0.5, fp * 45.0))
            wv = tw * sl * th_
            sheen *= 1.6
        dust = sstep(0.30, 0.03, pz)
        ar = ar * wv * (1.0 + 0.9 * dust) + 0.004 * dust
        ag = ag * wv * (1.0 + 0.8 * dust) + 0.0035 * dust
        ab = ab * wv * (1.0 + 0.6 * dust) + 0.003 * dust
    elif mat == M_TORCH:
        ar = 0.0042
        ag = 0.0036
        ab = 0.0034
        sheen = 0.0
        tl = F[i, F_TLIT]
        if tl > 0.0:
            g1 = vnoise2(px * 55.0 + T * 0.09, py * 55.0 + pz * 40.0, 53)
            g2 = vnoise2(px * 160.0 - T * 0.13, py * 160.0 + pz * 120.0, 56)
            c = tl * (0.55 + 0.9 * g1 * g1 + 0.35 * g2)
            er += 1.6 * c
            eg += 0.55 * c
            eb += 0.08 * c
    elif mat == M_WOOD:
        gr = vnoise2(px * 40.0 + pz * 300.0, py * 40.0, 54)
        ar = 0.015 * (0.75 + 0.5 * gr)
        ag = 0.0105 * (0.75 + 0.5 * gr)
        ab = 0.0075 * (0.75 + 0.5 * gr)
        sheen = 0.0
    elif mat == M_SHADOW:
        ar = 0.004
        ag = 0.0035
        ab = 0.0035
        sheen = 0.0
    else:   # hair
        ar = 0.012
        ag = 0.010
        ab = 0.009
        sheen *= 0.7
    cr, cg, cb = light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, sheen, False, fi,
                          PR, LT, OC, igc, igf, ao)
    # the crowd's torches beyond the stones rim the figures from behind
    if PR[P_CROWD] > 0.0 and mat != M_SHADOW:
        rq = math.sqrt(px * px + py * py) + 1e-9
        ox_ = px / rq
        oy_ = py / rq
        ol = math.sqrt(1.0 + 0.18 * 0.18)
        ndo = (nx * ox_ + ny * oy_ + nz * 0.18) / ol
        if ndo > -0.25:
            nv = abs(nx * vx + ny * vy + nz * vz)
            k = PR[P_CROWD] * (max(ndo, 0.0) * 0.30 + 1.1 * (1.0 - nv) ** 2 * sstep(-0.25, 0.5, ndo)) * ao
            cr += ar * k * 1.00
            cg += ag * k * 0.56
            cb += ab * k * 0.16
    # the hearth fire's rim on the side facing it
    if PR[P_FIGRIM] > 0.0 and mat != M_SHADOW:
        lx = -px
        ly = -py
        lz = PR[P_FIRE_Z] - pz
        ll = math.sqrt(lx * lx + ly * ly + lz * lz)
        ndl = (nx * lx + ny * ly + nz * lz) / ll
        nv = abs(nx * vx + ny * vy + nz * vz)
        rim = sstep(0.0, 0.6, ndl) * (1.0 - nv) ** 3.0 * ao
        e = PR[P_FIGRIM] * rim * 18.0
        er += e * ar * 1.0
        eg += e * ag * 0.62
        eb += e * ab * 0.22
    return cr + er, cg + eg, cb + eb, tbest, idb


@njit(parallel=True, **FM)
def render_surfaces(Wd, Hd, cam, PR, LT, OC, F, nf, S, ns, KB, LG, nlog, CH, nch, HDs, Js, HBB, RP, stex, ssz,
                    igc, igf, rgb, depth, oid, aa_pass, mask, nsub):
    Cx, Cy, Cz = cam[0], cam[1], cam[2]
    Rx, Ry, Rz = cam[3], cam[4], cam[5]
    Ux, Uy, Uz = cam[6], cam[7], cam[8]
    Fx, Fy, Fz = cam[9], cam[10], cam[11]
    f = cam[12]
    cx = cam[13]
    cy = cam[14]
    pix = 1.0 / f
    for y in prange(Hd):
        for x in range(Wd):
            if aa_pass and not mask[y, x]:
                continue
            ns_ = 1 if not aa_pass else nsub
            ar = 0.0
            ag = 0.0
            ab = 0.0
            tmin = 1e30
            idm = -1
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
                r, g, b, t, idv = shade_sample(Cx, Cy, Cz, dx, dy, dz, pix, PR, LT, OC, F, nf, S, ns, KB, LG, nlog,
                                               CH, nch, HDs, Js, HBB, RP, stex, ssz, igc, igf)
                ar += r
                ag += g
                ab += b
                if t < tmin:
                    tmin = t
                    idm = idv
            if aa_pass:
                # blend the 1-spp result with the extra samples
                rgb[y, x, 0] = (rgb[y, x, 0] + ar) / (ns_ + 1)
                rgb[y, x, 1] = (rgb[y, x, 1] + ag) / (ns_ + 1)
                rgb[y, x, 2] = (rgb[y, x, 2] + ab) / (ns_ + 1)
                if tmin < depth[y, x]:
                    depth[y, x] = tmin
            else:
                rgb[y, x, 0] = ar
                rgb[y, x, 1] = ag
                rgb[y, x, 2] = ab
                depth[y, x] = tmin
                oid[y, x] = idm
