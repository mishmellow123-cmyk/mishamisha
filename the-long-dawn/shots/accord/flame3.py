"""ACCORD v3 fire (numba): real torch flames (ray-marched volumes, one per torch), the hearth fire that the
torches light together, and sparks.

FL rows (torch flames): bx, by, bz (base = the top of the torch head), Hf, Rf, lean_x, lean_y, I, seed, lit
HP (hearth): see HP_* below; ANG = the azimuths where the torches meet the fuel.
"""
import math

import numpy as np
from numba import njit, prange

from nbcore import FM, clamp, sstep, mix, tex3

FL_N = 10
HP_T, HP_ON, HP_H, HP_WHITE, HP_SPREAD, HP_HOLLOW, HP_CONV, HP_CALM, HP_FX, HP_FY, HP_FZ, HP_I = range(12)
HP_P3 = 12          # 1 = bar 70: the fire that remains (calm_density over the CF flames)
HP_NMAIN = 13       # CF rows [0, nmain) stand on the stone; the rest lick along the logs
HP_N = 14
CF_N = 8            # calm-fire flames: x, y, z0, Hf, Rf, sx, sy, seed (Hf and the sway are per frame)

# a real flame's colour: deep orange at the cool edges -> orange -> yellow -> near-white in the core
RAMP = np.array([[0.00, 0.45, 0.060, 0.008],
                 [0.25, 0.85, 0.190, 0.022],
                 [0.50, 1.00, 0.400, 0.060],
                 [0.75, 1.00, 0.640, 0.190],
                 [1.00, 1.00, 0.860, 0.520]])


@njit(inline='always', **FM)
def ramp(t):
    if t <= 0.0:
        return RAMP[0, 1], RAMP[0, 2], RAMP[0, 3]
    if t >= 1.0:
        return RAMP[4, 1], RAMP[4, 2], RAMP[4, 3]
    k = 0
    while k < 3 and t > RAMP[k + 1, 0]:
        k += 1
    w = (t - RAMP[k, 0]) / (RAMP[k + 1, 0] - RAMP[k, 0])
    return (RAMP[k, 1] + (RAMP[k + 1, 1] - RAMP[k, 1]) * w,
            RAMP[k, 2] + (RAMP[k + 1, 2] - RAMP[k, 2]) * w,
            RAMP[k, 3] + (RAMP[k + 1, 3] - RAMP[k, 3]) * w)


@njit(inline='always', **FM)
def _fbm3(n3, x, y, z, oct_):
    n = 0.0
    amp = 0.5
    tot = 0.0
    for o in range(oct_):
        n += amp * tex3(n3, x + 13.1 * o, y - 7.7 * o, z + 3.3 * o)
        tot += amp
        amp *= 0.5
        x *= 2.07
        y *= 2.07
        z *= 1.93
    return n / tot


@njit(inline='always', **FM)
def _turb3(n3, x, y, z, oct_, gain):
    """Turbulence: sum of |noise - 0.5| octaves, sharp creases where the flame sheet folds (0..~1)."""
    n = 0.0
    amp = 1.0
    tot = 0.0
    for o in range(oct_):
        n += amp * abs(tex3(n3, x + 13.1 * o, y - 7.7 * o, z + 3.3 * o) - 0.5)
        tot += amp
        amp *= gain
        x *= 2.13
        y *= 2.13
        z *= 1.97
    return 2.0 * n / tot


@njit(**FM)
def torch_flicker(seed, T, n3):
    """The whole flame pumps (5-6 Hz) and breathes (~1 Hz): the height factor (also drives its light)."""
    Ts = T / 24.0
    f1 = tex3(n3, seed * 3.7, 5.0 + seed, Ts * 5.5)
    f2 = tex3(n3, 9.0 + seed, seed * 2.1, Ts * 1.1)
    return 0.80 + 0.28 * f1 + 0.14 * f2


@njit(**FM)
def torch_density(qx, qy, qz, FL, k, T, n3):
    """A torch flame, q relative to its base (world axes: a flame rises straight up and bends downwind).
    Returns (density, temperature): crisp torn tongues that rise faster than the flame and twist, wrapped
    round the head below, a hot core just above the head, detached flamelets burning out at the top."""
    lit = FL[k, 9]
    seed = FL[k, 8]
    Ts = T / 24.0
    Hf = FL[k, 3] * (0.35 + 0.65 * lit) * torch_flicker(seed, T, n3)
    Rf = FL[k, 4] * (0.55 + 0.45 * lit)
    u = qz / Hf
    if u < -0.34 or u > 1.3:
        return 0.0, 0.0
    uc = max(u, 0.0)
    # it bends downwind with height and sways slowly
    sx = FL[k, 5] + 0.40 * (tex3(n3, seed * 5.0, 17.0, Ts * 1.6) - 0.5)
    sy = FL[k, 6] + 0.40 * (tex3(n3, 31.0, seed * 5.0, Ts * 1.6) - 0.5)
    bend = Hf * uc ** 1.6
    x = qx - sx * bend
    y = qy - sy * bend
    rho = math.sqrt(x * x + y * y)
    # the envelope: wrapped round the head below, widest just above it, drawn up to a point
    if u < 0.18:
        w = (u - 0.18) / 0.52
        env = math.sqrt(max(1.0 - w * w, 0.0))
    else:
        w = (u - 0.18) / 0.90
        env = max(1.0 - w, 0.0) ** 0.75
    R = Rf * max(env, 0.16)
    if rho > R * 2.8 + 0.015:
        return 0.0, 0.0
    # twist the field, then tear it into a few big tongues (rising faster than the flame) with crinkled edges
    ang = 2.0 * uc + seed + 0.6 * Ts
    ca = math.cos(ang)
    sa = math.sin(ang)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    rise = 1.7 * Ts
    scl = 1.0 / 0.065
    wx = tex3(n3, xr * scl * 0.5 + 3.0 * seed, yr * scl * 0.5, (qz - rise * 0.8) * scl * 0.25) - 0.5
    wy = tex3(n3, xr * scl * 0.5, yr * scl * 0.5 + 3.0 * seed, (qz - rise * 0.8) * scl * 0.25 + 7.0) - 0.5
    nl = _fbm3(n3, (xr + 0.08 * wx) * scl + seed * 7.0, (yr + 0.08 * wy) * scl, (qz - rise) * scl * 0.36, 3)
    scf = 1.0 / 0.022
    nf = _turb3(n3, (xr + 0.05 * wx) * scf + 40.0, (yr + 0.05 * wy) * scf, (qz - 1.25 * rise) * scf * 0.5, 2, 0.55)
    AL = 0.55 + 1.35 * sstep(0.0, 1.0, u)
    AF = 0.30 + 0.45 * uc
    e = 1.0 - rho / R + AL * (2.0 * nl - 1.0) * 1.5 - AF * (nf - 0.40) - 2.6 * max(u - 0.82, 0.0)
    d = sstep(0.0, 0.10, e)
    if d <= 0.0:
        return 0.0, 0.0
    # hottest in the middle third (the luminous heart), cooler at the roots round the head, orange-red tips
    tax = sstep(-0.30, 0.22, u) * (1.0 - sstep(0.42, 1.05, u))
    temp = clamp(0.15 + 0.95 * tax * (1.0 - 0.45 * min(rho / R, 1.0)) + 0.55 * (nl - 0.5)
                 + 0.14 * sstep(0.0, 0.6, e), 0.0, 1.0)
    # the luminous sheet: emission peaks where the flame surface folds (limb-bright tongues, crisp edges)
    sh = (e - 0.14) / 0.16
    d = d * (0.30 + 0.70 * math.exp(-sh * sh))
    return d, temp


@njit(inline='always', **FM)
def _proj(cam, x, y, z):
    vx = x - cam[0]
    vy = y - cam[1]
    vz = z - cam[2]
    zc = vx * cam[9] + vy * cam[10] + vz * cam[11]
    if zc <= 1e-4:
        return 0.0, 0.0, -1.0
    xs = cam[13] + cam[12] * (vx * cam[3] + vy * cam[4] + vz * cam[5]) / zc
    ys = cam[14] - cam[12] * (vx * cam[6] + vy * cam[7] + vz * cam[8]) / zc
    return xs, ys, zc


@njit(parallel=True, **FM)
def torch_flames(img, depth, cam, FL, nfl, T, n3, alpha):
    """Ray-march every torch flame through its own world box (tight for the top-down camera) and add it to img
    (with a little soot absorption). ~7 mm steps along each ray."""
    Hd = img.shape[0]
    Wd = img.shape[1]
    f = cam[12]
    for k in range(nfl):
        if FL[k, 9] <= 0.002 or FL[k, 7] <= 0.0:
            continue
        Hmax = FL[k, 3] * 1.25
        Rm = FL[k, 4] * 1.9 + 0.02
        lx = FL[k, 5]
        ly = FL[k, 6]
        bx0 = FL[k, 0] + min(0.0, (lx - 0.22) * Hmax) - Rm
        bx1 = FL[k, 0] + max(0.0, (lx + 0.22) * Hmax) + Rm
        by0 = FL[k, 1] + min(0.0, (ly - 0.22) * Hmax) - Rm
        by1 = FL[k, 1] + max(0.0, (ly + 0.22) * Hmax) + Rm
        bz0 = FL[k, 2] - 0.36 * Hmax
        bz1 = FL[k, 2] + 1.3 * Hmax
        # the screen box of the world box
        xa = 1e9
        xb = -1e9
        ya = 1e9
        yb = -1e9
        ok = True
        for c in range(8):
            px = bx1 if (c & 1) else bx0
            py = by1 if (c & 2) else by0
            pz = bz1 if (c & 4) else bz0
            xs, ys, zc = _proj(cam, px, py, pz)
            if zc <= 0.0:
                ok = False
                break
            xa = min(xa, xs)
            xb = max(xb, xs)
            ya = min(ya, ys)
            yb = max(yb, ys)
        if not ok:
            continue
        x0 = max(int(xa) - 1, 0)
        x1 = min(int(xb) + 2, Wd)
        y0 = max(int(ya) - 1, 0)
        y1 = min(int(yb) + 2, Hd)
        if x1 <= x0 or y1 <= y0:
            continue
        I = FL[k, 7]
        for y in prange(y0, y1):
            for x in range(x0, x1):
                sx = (x + 0.5 - cam[13]) / f
                sy = -(y + 0.5 - cam[14]) / f
                dx = cam[9] + sx * cam[3] + sy * cam[6]
                dy = cam[10] + sx * cam[4] + sy * cam[7]
                dz = cam[11] + sx * cam[5] + sy * cam[8]
                l = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= l
                dy /= l
                dz /= l
                # ray / box slab test
                ta = 0.01
                tb = depth[y, x]
                if abs(dx) > 1e-9:
                    t0_ = (bx0 - cam[0]) / dx
                    t1_ = (bx1 - cam[0]) / dx
                    ta = max(ta, min(t0_, t1_))
                    tb = min(tb, max(t0_, t1_))
                elif cam[0] < bx0 or cam[0] > bx1:
                    continue
                if abs(dy) > 1e-9:
                    t0_ = (by0 - cam[1]) / dy
                    t1_ = (by1 - cam[1]) / dy
                    ta = max(ta, min(t0_, t1_))
                    tb = min(tb, max(t0_, t1_))
                elif cam[1] < by0 or cam[1] > by1:
                    continue
                if abs(dz) > 1e-9:
                    t0_ = (bz0 - cam[2]) / dz
                    t1_ = (bz1 - cam[2]) / dz
                    ta = max(ta, min(t0_, t1_))
                    tb = min(tb, max(t0_, t1_))
                elif cam[2] < bz0 or cam[2] > bz1:
                    continue
                if tb <= ta:
                    continue
                nsteps = int(min(max((tb - ta) / 0.007, 16.0), 110.0))
                ds = (tb - ta) / nsteps
                jit = (52.9829189 * ((0.06711056 * x + 0.00583715 * y + 0.1731 * (T % 7.0)) % 1.0)) % 1.0
                er = 0.0
                eg = 0.0
                eb = 0.0
                tr = 1.0
                for s in range(nsteps):
                    tt = ta + (s + jit) * ds
                    qx = cam[0] + dx * tt - FL[k, 0]
                    qy = cam[1] + dy * tt - FL[k, 1]
                    qz = cam[2] + dz * tt - FL[k, 2]
                    d, temp = torch_density(qx, qy, qz, FL, k, T, n3)
                    if d > 0.0:
                        cr, cg, cb = ramp(temp)
                        e = I * d * (0.05 + temp ** 2.6) * ds * tr
                        er += e * cr
                        eg += e * cg
                        eb += e * cb
                        tr *= math.exp(-d * ds * 7.0)
                        if tr < 0.01:
                            break
                img[y, x, 0] = img[y, x, 0] * (1.0 - alpha * (1.0 - tr)) + er
                img[y, x, 1] = img[y, x, 1] * (1.0 - alpha * (1.0 - tr)) + eg
                img[y, x, 2] = img[y, x, 2] * (1.0 - alpha * (1.0 - tr)) + eb


@njit(parallel=True, **FM)
def airlight(img, depth, cam, LT, nl, sigma, hmin, lam):
    """Firelight scattered by the smoke round each flame: for each light (isotropic point source) the in-scatter
    integral along the ray, 1/(h^2 + (t - t0)^2) from 0 to the surface depth, weighted by the smoke's falloff
    away from its flame (1 / (1 + (h/lam)^2)), so each torch wears its own soft halo."""
    Hd = img.shape[0]
    Wd = img.shape[1]
    f = cam[12]
    for y in prange(Hd):
        for x in range(Wd):
            sx = (x + 0.5 - cam[13]) / f
            sy = -(y + 0.5 - cam[14]) / f
            dx = cam[9] + sx * cam[3] + sy * cam[6]
            dy = cam[10] + sx * cam[4] + sy * cam[7]
            dz = cam[11] + sx * cam[5] + sy * cam[8]
            l = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= l
            dy /= l
            dz /= l
            tm = min(depth[y, x], 60.0)
            er = 0.0
            eg = 0.0
            eb = 0.0
            for k in range(nl):
                lx = LT[k, 0] - cam[0]
                ly = LT[k, 1] - cam[1]
                lz = LT[k, 2] - cam[2]
                t0 = lx * dx + ly * dy + lz * dz
                hx = lx - t0 * dx
                hy = ly - t0 * dy
                hz = lz - t0 * dz
                h = math.sqrt(hx * hx + hy * hy + hz * hz + hmin * hmin)
                v = (math.atan((tm - t0) / h) - math.atan(-t0 / h)) / h / (1.0 + (h / lam) ** 2)
                er += v * LT[k, 4]
                eg += v * LT[k, 5]
                eb += v * LT[k, 6]
            img[y, x, 0] += sigma * er
            img[y, x, 1] += sigma * eg
            img[y, x, 2] += sigma * eb


# ================================================================= the hearth ===

@njit(**FM)
def hearth_density(x, y, z, HP, ANG, nang, n3):
    """The fire everyone lit. From the ring of laid fuel round the stone the flames catch where the torches
    touched it and run together; they rise in torn tongues and lean in over the stone, hollow round her fist
    until the heart goes white; calm (bar 70): the burning logs' flames lean in and meet over the stone as one
    steady, warm fire. Returns (density, temperature)."""
    H = HP[HP_H]
    if H <= 0.0 or HP[HP_ON] <= 0.0:
        return 0.0, 0.0
    u = z / H
    if u <= -0.02 or u >= 1.15:
        return 0.0, 0.0
    uc = max(u, 0.0)
    T = HP[HP_T]
    Ts = T / 24.0
    r = math.sqrt(x * x + y * y)
    th = math.atan2(y, x)
    calm = HP[HP_CALM]
    # where the fuel burns: the ignited arcs of the ring
    ign = 0.0
    gap = math.pi / nang
    for k in range(nang):
        dd = abs((th - ANG[k] + math.pi) % (2.0 * math.pi) - math.pi)
        g = sstep(gap * (HP[HP_SPREAD] * 1.15 + 0.12) + 0.05, gap * (HP[HP_SPREAD] * 1.15 + 0.12) - 0.05, dd)
        if g > ign:
            ign = g
    ign = max(ign, calm)
    if ign <= 0.0:
        return 0.0, 0.0
    # the sheet of flame rises from the laid fuel (r ~ 0.6) and leans in over the stone as it climbs;
    # calm: the flames lean in and meet over the stone
    conv = HP[HP_CONV] * (1.0 - calm) + 0.92 * calm
    r_src = 0.62 * (1.0 - conv * uc ** 0.75) * (1.0 - calm) + 0.56 * (1.0 - conv * uc ** 0.6) * calm
    wdt = (0.17 + 0.10 * (1.0 - uc)) * (1.0 - calm) + (0.15 + 0.12 * (1.0 - uc)) * calm
    rr = abs(r - r_src) / max(wdt, 1e-3)
    if rr > 3.2:
        return 0.0, 0.0
    # twist slowly, then tear the sheet into big tongues (rising faster than the fire) with crinkled edges
    ang = 0.45 * uc + 0.10 * Ts
    ca = math.cos(ang)
    sa = math.sin(ang)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    rise = 2.1 * Ts
    scl = 1.0 / 0.11
    wx = tex3(n3, xr * scl * 0.5 + 5.0, yr * scl * 0.5, (z - rise * 0.7) * scl * 0.22) - 0.5
    wy = tex3(n3, xr * scl * 0.5, yr * scl * 0.5 + 9.0, (z - rise * 0.7) * scl * 0.22 + 4.0) - 0.5
    nl = _fbm3(n3, (xr + 0.12 * wx) * scl, (yr + 0.12 * wy) * scl, (z - rise) * scl * 0.34, 3)
    scf = 1.0 / 0.035
    nf = _turb3(n3, (xr + 0.08 * wx) * scf + 40.0, (yr + 0.08 * wy) * scf, (z - 1.3 * rise) * scf * 0.5, 2, 0.55)
    AL = 0.45 + 1.25 * sstep(0.0, 1.0, u)
    AF = 0.25 + 0.40 * uc
    e = 1.0 - rr + AL * (2.0 * nl - 1.0) * 1.5 - AF * (nf - 0.40) - 2.4 * max(u - 0.80, 0.0)
    # hollow round her fist until the white
    fx = x - HP[HP_FX]
    fy = y - HP[HP_FY]
    fz = z - HP[HP_FZ]
    hol = HP[HP_HOLLOW]
    if hol > 0.0:
        dh = math.sqrt(fx * fx + fy * fy + 0.5 * fz * fz)
        e -= hol * 0.8 * sstep(0.20, 0.07, dh)
    d = sstep(0.0, 0.10, e) * ign * sstep(-0.02, 0.04, u)
    if d <= 0.0:
        return 0.0, 0.0
    # hot at the roots in the fuel, orange in the body, red at the torn tips; calm is a little cooler
    troot = 1.0 - sstep(0.05, 0.75, u)
    temp = clamp((0.30 + 0.72 * troot * (1.0 - 0.45 * min(rr, 1.0)) + 0.50 * (nl - 0.5)
                  + 0.14 * sstep(0.0, 0.6, e)) * (1.0 - 0.12 * calm), 0.0, 1.0)
    # the luminous sheet: emission peaks where the flame surface folds
    sh = (e - 0.14) / 0.16
    d = d * (0.30 + 0.70 * math.exp(-sh * sh))
    return d, temp


@njit(**FM)
def calm_density(x, y, z, HP, CF, ncf, n3):
    """Bar 70, the fire that remains: one fire standing on the stone where the Ring was. Each CF row is a flame
    (an envelope wrapped round its root, drawn up to a point, bending downwind); ONE shared field of tongues and
    crinkle tears their union, so the flames lick together as one fire instead of a clump of torches.
    Hot yellow at the roots over the coals, orange in the body, dark red at the torn tips. (density, temp)"""
    T = HP[HP_T]
    Ts = T / 24.0
    nmain = int(HP[HP_NMAIN])
    best = -1e9
    ub = 0.0
    rb = 1.0
    r0 = 0.1
    r2 = x * x + y * y
    k0 = 0
    k1 = ncf
    if z > 0.62:
        k1 = nmain                  # the log flames stand below ~0.6 m
    if r2 > 0.30:
        k0 = nmain                  # the flames on the stone stand within ~0.55 m of its centre
    for k in range(k0, k1):
        Hf = CF[k, 3]
        qz = z - CF[k, 2]
        u = qz / Hf
        if u < -0.30 or u > 1.25:
            continue
        uc = max(u, 0.0)
        bend = Hf * uc ** 1.5
        qx = x - CF[k, 0] - CF[k, 5] * bend
        qy = y - CF[k, 1] - CF[k, 6] * bend
        rho = math.sqrt(qx * qx + qy * qy)
        if rho > CF[k, 4] * 2.4 + 0.02:
            continue
        if u < 0.15:
            w = (u - 0.15) / 0.42
            env = math.sqrt(max(1.0 - w * w, 0.0))
        else:
            w = (u - 0.15) / 0.92
            env = max(1.0 - w, 0.0) ** 0.8
        R = CF[k, 4] * max(env, 0.14)
        ek = 1.0 - rho / R - 2.4 * max(u - 0.80, 0.0)
        if ek > best:
            best = ek
            ub = u
            rb = rho / R
            r0 = CF[k, 4]
    if best < -1.4:
        return 0.0, 0.0
    # the shared field: a slow twist as it climbs, big tongues rising faster than the fire, crinkled edges
    zz = z - 0.28
    ang = 0.8 * max(zz, 0.0) + 0.25 * Ts
    ca = math.cos(ang)
    sa = math.sin(ang)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    rise = 1.25 * Ts
    # the tongues' size follows the flame that owns the point: the heart tears in big tongues, the log flames fine
    lt = 0.40 * r0 + 0.014
    scl = 1.0 / lt
    wx = tex3(n3, xr * scl * 0.5 + 5.0, yr * scl * 0.5, (zz - rise * 0.8) * scl * 0.25) - 0.5
    wy = tex3(n3, xr * scl * 0.5, yr * scl * 0.5 + 9.0, (zz - rise * 0.8) * scl * 0.25 + 4.0) - 0.5
    nl = _fbm3(n3, (xr + 1.25 * lt * wx) * scl, (yr + 1.25 * lt * wy) * scl, (zz - rise) * scl * 0.36, 3)
    scf = 1.0 / (0.33 * lt)
    nf = _turb3(n3, (xr + 0.75 * lt * wx) * scf + 40.0, (yr + 0.75 * lt * wy) * scf, (zz - 1.25 * rise) * scf * 0.5, 2, 0.55)
    uu = max(ub, 0.0)
    AL = 0.62 + 1.45 * sstep(0.0, 1.0, ub)
    AF = 0.30 + 0.42 * uu
    e = best + AL * (2.0 * nl - 1.0) * 1.5 - AF * (nf - 0.40)
    d = sstep(0.0, 0.10, e)
    if d <= 0.0:
        return 0.0, 0.0
    # hot over the coals (the roots), cooling through the body, the torn tips dull red
    troot = 1.0 - sstep(0.02, 0.85, uu)
    temp = clamp(0.20 + 0.78 * troot * (1.0 - 0.40 * min(rb, 1.0)) + 0.50 * (nl - 0.5)
                 + 0.14 * sstep(0.0, 0.6, e), 0.0, 1.0)
    # the luminous sheet: emission peaks where the flame surface folds (limb-bright tongues, crisp edges)
    sh = (e - 0.14) / 0.16
    d = d * (0.30 + 0.70 * math.exp(-sh * sh))
    return d, temp


@njit(inline='always', **FM)
def _ray_cyl(Cx, Cy, dx, dy, R):
    """Ray (horizontal part) vs an infinite vertical cylinder of radius R about the z axis: (ta, tb) or (1, 0)."""
    a = dx * dx + dy * dy
    b = 2.0 * (Cx * dx + Cy * dy)
    c = Cx * Cx + Cy * Cy - R * R
    if a < 1e-12:
        if c > 0.0:
            return 1.0, 0.0
        return -1e9, 1e9
    disc = b * b - 4 * a * c
    if disc <= 0.0:
        return 1.0, 0.0
    sq = math.sqrt(disc)
    return (-b - sq) / (2 * a), (-b + sq) / (2 * a)


@njit(inline='always', **FM)
def _ray_slab(Cz, dz, z0, z1):
    if abs(dz) < 1e-9:
        if Cz < z0 or Cz > z1:
            return 1.0, 0.0
        return -1e9, 1e9
    ta = (z0 - Cz) / dz
    tb = (z1 - Cz) / dz
    if ta > tb:
        ta, tb = tb, ta
    return ta, tb


@njit(parallel=True, **FM)
def hearth_volume(Wd, Hd, cam, HP, ANG, nang, n3, depth, out, nsteps, CF, ncf):
    Cx, Cy, Cz = cam[0], cam[1], cam[2]
    f = cam[12]
    H = HP[HP_H]
    if H <= 0.0 or HP[HP_ON] <= 0.0:
        return
    p3 = HP[HP_P3] > 0.5
    Rb = 1.02
    z0 = -0.02
    z1 = H
    # bar 70: the fire on the stone (a narrow tall column) and the low flames on the logs (a wide flat disc)
    zc1 = 0.0
    zl1 = 0.0
    if p3:
        for k in range(ncf):
            top = CF[k, 2] + CF[k, 3] * 1.25
            if k < int(HP[HP_NMAIN]):
                zc1 = max(zc1, top)
            else:
                zl1 = max(zl1, top)
    I = HP[HP_I]
    wh = HP[HP_WHITE]
    for y in prange(Hd):
        for x in range(Wd):
            sx = (x + 0.5 - cam[13]) / f
            sy = -(y + 0.5 - cam[14]) / f
            dx = cam[9] + sx * cam[3] + sy * cam[6]
            dy = cam[10] + sx * cam[4] + sy * cam[7]
            dz = cam[11] + sx * cam[5] + sy * cam[8]
            l = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= l
            dy /= l
            dz /= l
            if p3:
                # two tight intervals: the column over the stone, the disc over the logs (march their union)
                ca, cb = _ray_cyl(Cx, Cy, dx, dy, 0.62)
                za, zb = _ray_slab(Cz, dz, 0.20, zc1)
                ta1 = max(max(ca, za), 0.02)
                tb1 = min(min(cb, zb), depth[y, x])
                ca, cb = _ray_cyl(Cx, Cy, dx, dy, Rb)
                za, zb = _ray_slab(Cz, dz, -0.02, zl1)
                ta2 = max(max(ca, za), 0.02)
                tb2 = min(min(cb, zb), depth[y, x])
                if tb1 <= ta1 and tb2 <= ta2:
                    continue
                if tb1 <= ta1:
                    ta = ta2
                    tb = tb2
                elif tb2 <= ta2:
                    ta = ta1
                    tb = tb1
                else:
                    ta = min(ta1, ta2)
                    tb = max(tb1, tb2)
                ns = int(min(max((tb - ta) / 0.009, 24.0), 200.0))
            else:
                ca, cb = _ray_cyl(Cx, Cy, dx, dy, Rb)
                if cb <= ca:
                    continue
                za, zb = _ray_slab(Cz, dz, z0, z1)
                ta = max(max(ca, za), 0.02)
                tb = min(min(cb, zb), depth[y, x])
                ns = nsteps
            if tb <= ta:
                continue
            ds = (tb - ta) / ns
            jit = (52.9829189 * ((0.06711056 * x + 0.00583715 * y + 0.1731 * (HP[HP_T] % 7.0)) % 1.0)) % 1.0
            er = 0.0
            eg = 0.0
            eb = 0.0
            tr = 1.0
            for k in range(ns):
                tt = ta + (k + jit) * ds
                px = Cx + dx * tt
                py = Cy + dy * tt
                pz = Cz + dz * tt
                if p3:
                    d, temp = calm_density(px, py, pz, HP, CF, ncf, n3)
                else:
                    d, temp = hearth_density(px, py, pz, HP, ANG, nang, n3)
                if d > 0.0:
                    t2 = min(temp + wh * 0.7, 1.0)
                    cr, cg, cb = ramp(t2)
                    if wh > 0.0:
                        cr = mix(cr, 1.0, wh * 0.75)
                        cg = mix(cg, 0.96, wh * 0.75)
                        cb = mix(cb, 0.88, wh * 0.75)
                    e = I * d * (0.05 + t2 ** 2.6) * ds * tr * (1.0 + 4.0 * wh)
                    er += e * cr
                    eg += e * cg
                    eb += e * cb
                    tr *= math.exp(-d * ds * (1.6 if p3 else 5.0) * (1.0 - wh))
                    if tr < 0.01:
                        break
            out[y, x, 0] = out[y, x, 0] * tr + er
            out[y, x, 1] = out[y, x, 1] * tr + eg
            out[y, x, 2] = out[y, x, 2] * tr + eb


# ================================================================= sparks ===

def sparks(t, fire, n=260, seed=5):
    """Sparks rising off the fire (closed-form, deterministic): rows for fire.splat_streaks."""
    if fire['on'] <= 0.0:
        return np.zeros((0, 11))
    rng = np.random.default_rng(seed)
    birth = rng.uniform(5190.0, 5700.0, n)
    life = rng.uniform(18.0, 46.0, n)
    a0 = rng.uniform(0, 2 * np.pi, n)
    r0 = rng.uniform(0.2, 0.8, n)
    vz = rng.uniform(0.8, 2.2, n)
    sw = rng.uniform(-1.5, 1.5, n)
    drift = rng.uniform(0.05, 0.35, n)
    out = []
    for k in range(n):
        age = t - birth[k]
        if age < 0 or age > life[k]:
            continue
        # fewer sparks when the fire is calm, a burst at the catch
        dens = 0.55 + 0.45 * (1.0 - fire['calm'])
        if (k % 100) / 100.0 > dens:
            continue
        rows = []
        for dt in (-0.35, 0.0):
            s = max(age + dt, 0.0) / 24.0
            ang = a0[k] + sw[k] * s
            r = r0[k] * (1.0 - 0.3 * min(s * 2.0, 1.0)) + drift[k] * s
            z = 0.25 + vz[k] * s - 0.9 * s * s
            rows.append((r * math.cos(ang), r * math.sin(ang), z))
        fade = (1.0 - age / life[k]) ** 1.5 * min(age / 3.0, 1.0)
        hot = 1.0 + 1.5 * fire['white']
        c = np.array([1.0, 0.42, 0.08]) * 8.0 * fade * hot
        out.append([*rows[0], *rows[1], 0.0022, c[0], c[1], c[2], 1.0])
    return np.array(out, np.float64) if out else np.zeros((0, 11))
