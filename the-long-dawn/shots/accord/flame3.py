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
HP_N = 14

# a real flame's colour: deep orange at the cool edges -> orange -> yellow -> near-white in the core
RAMP = np.array([[0.00, 0.50, 0.090, 0.012],
                 [0.25, 0.90, 0.250, 0.035],
                 [0.50, 1.00, 0.470, 0.090],
                 [0.75, 1.00, 0.700, 0.260],
                 [1.00, 1.00, 0.900, 0.620]])


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


@njit(**FM)
def torch_density(qx, qy, qz, FL, k, T, n3):
    """A torch flame in its own frame (q relative to the base). Returns (density, temperature)."""
    lit = FL[k, 9]
    Hf = FL[k, 3] * (0.35 + 0.65 * lit)
    Rf = FL[k, 4] * (0.55 + 0.45 * lit)
    seed = FL[k, 8]
    Ts = T / 24.0
    # the flame's height breathes and flickers
    fl = 0.86 + 0.14 * math.sin(Ts * 7.3 + seed) * math.sin(Ts * 3.1 + 2.0 * seed) + 0.08 * math.sin(Ts * 13.7 + seed)
    Hf *= fl
    u = qz / Hf
    if u < -0.25 or u > 1.05:
        return 0.0, 0.0
    uc = max(u, 0.0)
    # the tongue bends downwind with height
    cx = FL[k, 5] * Hf * uc ** 1.5
    cy = FL[k, 6] * Hf * uc ** 1.5
    x = qx - cx
    y = qy - cy
    rho = math.sqrt(x * x + y * y)
    # teardrop envelope, wide round the head, a point at the top
    ub = (u + 0.25) / 1.25
    R = Rf * 1.45 * ub ** 0.45 * (1.0 - ub) ** 0.85
    if R <= 1e-5 or rho > R * 1.9:
        return 0.0, 0.0
    sc = 1.0 / (0.045 + 0.02 * uc)
    rise = 2.6 * Ts
    wx = tex3(n3, x * sc * 0.5 + seed, y * sc * 0.5, (qz - rise * 0.5) * sc * 0.3) - 0.5
    wy = tex3(n3, x * sc * 0.5, y * sc * 0.5 + seed, (qz - rise * 0.5) * sc * 0.3 + 4.0) - 0.5
    n = _fbm3(n3, (x + wx * 0.03) * sc + seed * 3.0, (y + wy * 0.03) * sc, (qz - rise * 0.35) * sc * 0.55, 3)
    e = 1.0 - rho / R + 0.9 * (n - 0.5) - 0.55 * uc * uc
    d = sstep(0.0, 0.35, e)
    if d <= 0.0:
        return 0.0, 0.0
    temp = clamp(0.95 * (1.0 - uc) ** 1.2 * (1.0 - 0.6 * rho / R) + 0.35 * (n - 0.5) + 0.1, 0.0, 1.0)
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
    """Ray-march every torch flame inside its own screen box and add it to img (with a little soot)."""
    Hd = img.shape[0]
    Wd = img.shape[1]
    f = cam[12]
    for k in range(nfl):
        if FL[k, 9] <= 0.002 or FL[k, 7] <= 0.0:
            continue
        Hf = FL[k, 3] * 1.15
        mx = FL[k, 0] + FL[k, 5] * Hf * 0.5
        my = FL[k, 1] + FL[k, 6] * Hf * 0.5
        mz = FL[k, 2] + Hf * 0.40
        Rb = Hf * 0.62 + FL[k, 4] * 1.5 + math.sqrt(FL[k, 5] ** 2 + FL[k, 6] ** 2) * Hf * 0.5
        xs, ys, zc = _proj(cam, mx, my, mz)
        if zc <= 0.0:
            continue
        rp = f * Rb / max(zc - Rb, 0.05)
        x0 = max(int(xs - rp) - 1, 0)
        x1 = min(int(xs + rp) + 2, Wd)
        y0 = max(int(ys - rp) - 1, 0)
        y1 = min(int(ys + rp) + 2, Hd)
        if x1 <= x0 or y1 <= y0:
            continue
        nsteps = int(min(max(rp * 0.8, 10.0), 40.0))
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
                ocx = cam[0] - mx
                ocy = cam[1] - my
                ocz = cam[2] - mz
                b = ocx * dx + ocy * dy + ocz * dz
                c = ocx * ocx + ocy * ocy + ocz * ocz - Rb * Rb
                disc = b * b - c
                if disc <= 0.0:
                    continue
                sq = math.sqrt(disc)
                ta = max(-b - sq, 0.01)
                tb = min(-b + sq, depth[y, x])
                if tb <= ta:
                    continue
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
                        e = I * d * (0.12 + temp ** 2.2) * ds * tr
                        er += e * cr
                        eg += e * cg
                        eb += e * cb
                        tr *= math.exp(-d * ds * 9.0 * (0.3 + qz / FL[k, 3]))
                img[y, x, 0] = img[y, x, 0] * (1.0 - alpha * (1.0 - tr)) + er
                img[y, x, 1] = img[y, x, 1] * (1.0 - alpha * (1.0 - tr)) + eg
                img[y, x, 2] = img[y, x, 2] * (1.0 - alpha * (1.0 - tr)) + eb


# ================================================================= the hearth ===

@njit(**FM)
def hearth_density(x, y, z, HP, ANG, nang, n3):
    """The fire everyone lit. From the ring of laid fuel round the stone the flames catch where the torches
    touched it and run together; they rise and lean in over the stone, hollow round her fist until the
    heart goes white; out of the white (calm) it settles to one warm, steady fire on the stone."""
    H = HP[HP_H]
    if H <= 0.0 or HP[HP_ON] <= 0.0:
        return 0.0, 0.0
    u = z / H
    if u <= -0.02 or u >= 1.0:
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
    # the sheet of flame: rising from r ~ 0.62 and leaning in over the stone as it climbs
    conv = HP[HP_CONV]
    r_src = 0.62 * (1.0 - conv * uc ** 0.75) * (1.0 - calm) + 0.18 * calm * (1.0 - 0.8 * uc)
    wdt = (0.20 + 0.10 * (1.0 - uc)) * (1.0 - calm) + (0.34 * (1.0 - 0.55 * uc)) * calm
    sw = HP[HP_SPREAD]
    # swirl, and the noise that tears the sheet into tongues
    ang = 0.55 * uc + 0.12 * Ts
    ca = math.cos(ang)
    sa = math.sin(ang)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    sc = 7.5
    rise = 3.0 * Ts
    wx = tex3(n3, xr * sc * 0.4 + 5.0, yr * sc * 0.4, (z - rise * 0.6) * sc * 0.18) - 0.5
    wy = tex3(n3, xr * sc * 0.4, yr * sc * 0.4 + 9.0, (z - rise * 0.6) * sc * 0.18 + 4.0) - 0.5
    n = _fbm3(n3, (xr + wx * 0.16) * sc, (yr + wy * 0.16) * sc, (z - rise) * sc * 0.42, 4)
    rr = abs(r - r_src) / max(wdt, 1e-3)
    e = 1.0 - rr + 1.1 * (n - 0.5) - 0.9 * uc ** 1.4
    # hollow round her fist until the white
    fx = x - HP[HP_FX]
    fy = y - HP[HP_FY]
    fz = z - HP[HP_FZ]
    hol = HP[HP_HOLLOW]
    if hol > 0.0:
        dh = math.sqrt(fx * fx + fy * fy + 0.5 * fz * fz)
        e -= hol * 0.9 * sstep(0.26, 0.08, dh)
    d = sstep(0.0, 0.3, e) * ign * sstep(-0.02, 0.05, u)
    # the base glows as one body where the fuel burns
    base = sstep(0.12, 0.0, z) * ign * sstep(0.30, 0.0, abs(r - (0.62 * (1.0 - calm) + 0.15 * calm)) - 0.1)
    d = max(d, 0.6 * base)
    if d <= 0.0:
        return 0.0, 0.0
    temp = clamp(0.85 * (1.0 - uc) ** 1.3 + 0.45 * (n - 0.5) + 0.25 * base + 0.12, 0.0, 1.0)
    return d, temp


@njit(parallel=True, **FM)
def hearth_volume(Wd, Hd, cam, HP, ANG, nang, n3, depth, out, nsteps):
    Cx, Cy, Cz = cam[0], cam[1], cam[2]
    f = cam[12]
    H = HP[HP_H]
    if H <= 0.0 or HP[HP_ON] <= 0.0:
        return
    Rb = 1.02
    z0 = -0.02
    z1 = H
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
            a = dx * dx + dy * dy
            b = 2.0 * (Cx * dx + Cy * dy)
            c = Cx * Cx + Cy * Cy - Rb * Rb
            if a < 1e-12:
                if c > 0.0:
                    continue
                ta = -1e9
                tb = 1e9
            else:
                disc = b * b - 4 * a * c
                if disc <= 0.0:
                    continue
                sq = math.sqrt(disc)
                ta = (-b - sq) / (2 * a)
                tb = (-b + sq) / (2 * a)
            if abs(dz) > 1e-9:
                tz0 = (z1 - Cz) / dz
                tz1 = (z0 - Cz) / dz
                if tz0 > tz1:
                    tz0, tz1 = tz1, tz0
                ta = max(ta, tz0)
                tb = min(tb, tz1)
            ta = max(ta, 0.02)
            tb = min(tb, depth[y, x])
            if tb <= ta:
                continue
            ds = (tb - ta) / nsteps
            jit = (52.9829189 * ((0.06711056 * x + 0.00583715 * y + 0.1731 * (HP[HP_T] % 7.0)) % 1.0)) % 1.0
            er = 0.0
            eg = 0.0
            eb = 0.0
            tr = 1.0
            for k in range(nsteps):
                tt = ta + (k + jit) * ds
                px = Cx + dx * tt
                py = Cy + dy * tt
                pz = Cz + dz * tt
                d, temp = hearth_density(px, py, pz, HP, ANG, nang, n3)
                if d > 0.0:
                    t2 = min(temp + wh * 0.7, 1.0)
                    cr, cg, cb = ramp(t2)
                    if wh > 0.0:
                        cr = mix(cr, 1.0, wh * 0.75)
                        cg = mix(cg, 0.96, wh * 0.75)
                        cb = mix(cb, 0.88, wh * 0.75)
                    e = I * d * (0.08 + temp ** 2.6) * ds * tr * (1.0 + 4.0 * wh)
                    er += e * cr
                    eg += e * cg
                    eb += e * cb
                    zr = pz / H
                    tr *= math.exp(-d * ds * 2.2 * zr * (1.0 - wh))
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
