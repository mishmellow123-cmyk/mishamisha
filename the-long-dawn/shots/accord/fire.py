"""Fire, particles and post kernels for ACCORD (numba).

* fire_volume   – ray-marched emissive golden fire plume above the hearth
* splat_blobs   – additive gaussian blobs (torch flames, ribbons, merge core)
* splat_streaks – motion-blurred, DOF-sized particle streaks (embers, walker torches)
* dark_sprites  – alpha-darkening ellipses (walker bodies seen from above)
* mist_layers   – moonlit mist planes for the descent
* motion_blur   – camera motion blur from the depth buffer
"""
import math

import numpy as np
from numba import njit, prange

from nbcore import FM, clamp, sstep, mix, tex3, vnoise2, fbm2, grid_sample

# fire params
FP_T, FP_I, FP_SCALE, FP_H, FP_SWIRL, FP_Z0, FP_WHITE, FP_R0, FP_RISE, FP_SPREAD = range(10)
FP_N = 12


@njit(inline='always', **FM)
def gold_ramp(t):
    """Golden fire colour ramp (linear RGB), t in [0,1]."""
    if t < 0.35:
        u = t / 0.35
        return 0.30 + (0.807 - 0.30) * u, 0.07 + (0.366 - 0.07) * u, 0.004 + (0.047 - 0.004) * u
    elif t < 0.62:
        u = (t - 0.35) / 0.27
        return 0.807 + (1.0 - 0.807) * u, 0.366 + (0.644 - 0.366) * u, 0.047 + (0.195 - 0.047) * u
    elif t < 0.85:
        u = (t - 0.62) / 0.23
        return 1.0, 0.644 + (0.879 - 0.644) * u, 0.195 + (0.584 - 0.195) * u
    else:
        u = min((t - 0.85) / 0.15, 1.0)
        return 1.0, 0.879 + (0.985 - 0.879) * u, 0.584 + (0.93 - 0.584) * u


@njit(**FM)
def fire_density(x, y, z, FP, n3):
    """Flame tongues seen from above: vertically stretched, domain-warped turbulence thresholded
    into thin sheets that fan outward and swirl as they rise (petals), with a compact hot base."""
    s = FP[FP_SCALE]
    H = FP[FP_H] * s
    z0 = FP[FP_Z0]
    zr = (z - z0) / H
    if zr <= 0.0 or zr >= 1.0:
        return 0.0, 0.0
    k = 1.0 + FP[FP_SPREAD] * zr ** 1.15
    r = math.sqrt(x * x + y * y)
    rb = r / k
    w = FP[FP_R0] * s + 0.02
    shape = math.exp(-(rb / w) ** 2)
    if shape < 0.008:
        return 0.0, 0.0
    Ts = FP[FP_T] / 24.0
    ang = FP[FP_SWIRL] * zr ** 1.2 + 0.40 * Ts
    ca = math.cos(ang)
    sa = math.sin(ang)
    xb = (x * ca - y * sa) / k
    yb = (x * sa + y * ca) / k
    sc = 16.0 / max(s, 0.3) ** 0.5
    rise = FP[FP_RISE] * Ts
    wx = tex3(n3, xb * sc * 0.35 + 5.0, yb * sc * 0.35, (z - rise * 0.6) * sc * 0.16) - 0.5
    wy = tex3(n3, xb * sc * 0.35, yb * sc * 0.35 + 9.0, (z - rise * 0.6) * sc * 0.16 + 4.0) - 0.5
    nx = (xb + wx * 0.20) * sc
    ny = (yb + wy * 0.20) * sc
    nz = (z - rise) * sc * 0.28
    n = 0.0
    amp = 0.5
    tot = 0.0
    for o in range(4):
        n += amp * tex3(n3, nx + 13.1 * o, ny - 7.7 * o, nz + 3.3 * o)
        tot += amp
        amp *= 0.52
        nx *= 2.11
        ny *= 2.11
        nz *= 1.9
    n /= tot
    th = 0.46 + 0.14 * zr + 0.30 * (1.0 - shape)
    d = sstep(th - 0.012, th + 0.05, n) * sstep(0.0, 0.04, zr) * (1.0 - zr) ** 0.9
    base = math.exp(-(r / (0.34 * s + 0.02)) ** 2) * math.exp(-zr / 0.07)
    d = max(d, base * 0.9)
    if d <= 0.0:
        return 0.0, 0.0
    temp = clamp((n - th) * 4.0 + 0.72 * (1.0 - zr) ** 2.2 + 0.18 * shape - 0.22 + base * 0.7, 0.0, 1.0)
    return d, temp


@njit(parallel=True, **FM)
def fire_volume(Wd, Hd, cam, FP, n3, depth, out, nsteps):
    Cx, Cy, Cz = cam[0], cam[1], cam[2]
    Rx, Ry, Rz = cam[3], cam[4], cam[5]
    Ux, Uy, Uz = cam[6], cam[7], cam[8]
    Fx, Fy, Fz = cam[9], cam[10], cam[11]
    f = cam[12]
    cx = cam[13]
    cy = cam[14]
    s = FP[FP_SCALE]
    if s <= 0.0 or FP[FP_I] <= 0.0:
        return
    Rb = (FP[FP_R0] * s * 1.25 + 0.1) * (1.0 + FP[FP_SPREAD] * 0.8)
    z0 = FP[FP_Z0]
    z1 = z0 + FP[FP_H] * s
    I = FP[FP_I]
    wh = FP[FP_WHITE]
    for y in prange(Hd):
        for x in range(Wd):
            sx = (x + 0.5 - cx) / f
            sy = -(y + 0.5 - cy) / f
            dx = Fx + sx * Rx + sy * Ux
            dy = Fy + sx * Ry + sy * Uy
            dz = Fz + sx * Rz + sy * Uz
            l = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= l
            dy /= l
            dz /= l
            # cylinder r<Rb, z in [z0,z1]
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
            n = nsteps
            ds = (tb - ta) / n
            jit = (52.9829189 * ((0.06711056 * x + 0.00583715 * y + 0.1731 * (FP[FP_T] % 7.0)) % 1.0)) % 1.0
            er = 0.0
            eg = 0.0
            eb = 0.0
            trans = 1.0
            for k in range(n):
                tt = ta + (k + jit) * ds
                px = Cx + dx * tt
                py = Cy + dy * tt
                pz = Cz + dz * tt
                d, temp = fire_density(px, py, pz, FP, n3)
                if d > 0.0:
                    t2 = min(temp + wh * 0.6, 1.0)
                    cr, cg, cb = gold_ramp(t2)
                    if wh > 0.0:
                        cr = mix(cr, 1.0, wh * 0.7)
                        cg = mix(cg, 0.97, wh * 0.7)
                        cb = mix(cb, 0.92, wh * 0.7)
                    e = I * d * (0.05 + temp ** 3.5) * ds * trans
                    er += e * cr
                    eg += e * cg
                    eb += e * cb
                    # faint self-absorption in the upper, cooler parts
                    zr = (pz - z0) / (z1 - z0)
                    trans *= math.exp(-d * ds * 1.2 * zr)
            out[y, x, 0] += er
            out[y, x, 1] += eg
            out[y, x, 2] += eb


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


@njit(**FM)
def splat_blobs(img, depth, cam, B, nb, soft_depth, dimband):
    """B[k] = x,y,z, radius_m, r,g,b (peak radiance), depthtest(0/1).
    dimband = (y0,y1,amount) screen band attenuation (text-safe band)."""
    Hd = img.shape[0]
    Wd = img.shape[1]
    for k in range(nb):
        xs, ys, zc = _proj(cam, B[k, 0], B[k, 1], B[k, 2])
        if zc <= 0.0:
            continue
        rp = cam[12] * B[k, 3] / zc
        sig = max(rp, 0.55)
        amp = (rp * rp) / (sig * sig) if rp < 0.55 else 1.0
        ext = int(3.2 * sig) + 1
        x0 = int(xs) - ext
        x1 = int(xs) + ext + 1
        y0 = int(ys) - ext
        y1 = int(ys) + ext + 1
        if x1 < 0 or y1 < 0 or x0 >= Wd or y0 >= Hd:
            continue
        att = 1.0
        if dimband[2] > 0.0:
            att = 1.0 - dimband[2] * sstep(dimband[0] - 30, dimband[0] + 10, ys) * sstep(dimband[1] + 30, dimband[1] - 10, ys)
        inv = 1.0 / (2.0 * sig * sig)
        for yy in range(max(y0, 0), min(y1, Hd)):
            for xx in range(max(x0, 0), min(x1, Wd)):
                ddx = xx + 0.5 - xs
                ddy = yy + 0.5 - ys
                g = math.exp(-(ddx * ddx + ddy * ddy) * inv) * amp * att
                if g < 1e-4:
                    continue
                if B[k, 7] > 0.0 and depth[yy, xx] < zc - soft_depth:
                    # occluded (soft)
                    g *= 0.0
                img[yy, xx, 0] += g * B[k, 4]
                img[yy, xx, 1] += g * B[k, 5]
                img[yy, xx, 2] += g * B[k, 6]


@njit(**FM)
def splat_streaks(img, depth, cam, P, n, aperture, zfocus, dimband, maxcoc):
    """P[k] = x0,y0,z0, x1,y1,z1, radius_m, r,g,b (radiance), depthtest.
    Energy-conserving motion-blurred + DOF gaussian streak."""
    Hd = img.shape[0]
    Wd = img.shape[1]
    f = cam[12]
    for k in range(n):
        xa, ya, za = _proj(cam, P[k, 0], P[k, 1], P[k, 2])
        xb, yb, zb = _proj(cam, P[k, 3], P[k, 4], P[k, 5])
        if za <= 0.0 or zb <= 0.0:
            continue
        zm = 0.5 * (za + zb)
        rp = f * P[k, 6] / zm
        coc = aperture * f * abs(1.0 / zm - 1.0 / zfocus)
        coc = min(coc, maxcoc)
        sig = math.sqrt(max(rp, 0.5) ** 2 + (0.5 * coc) ** 2)
        flux = math.pi * max(rp, 0.35) ** 2
        lx = xb - xa
        ly = yb - ya
        L = math.sqrt(lx * lx + ly * ly)
        norm = flux / (2.0 * math.pi * sig * sig + 2.5066 * sig * L)
        ext = int(3.0 * sig) + 1
        bx0 = int(min(xa, xb)) - ext
        bx1 = int(max(xa, xb)) + ext + 1
        by0 = int(min(ya, yb)) - ext
        by1 = int(max(ya, yb)) + ext + 1
        if bx1 < 0 or by1 < 0 or bx0 >= Wd or by0 >= Hd:
            continue
        if (bx1 - bx0) * (by1 - by0) > 400000:
            continue
        att = 1.0
        if dimband[2] > 0.0:
            ym = 0.5 * (ya + yb)
            att = 1.0 - dimband[2] * sstep(dimband[0] - 30, dimband[0] + 10, ym) * sstep(dimband[1] + 30, dimband[1] - 10, ym)
        inv = 1.0 / (2.0 * sig * sig)
        L2 = L * L
        for yy in range(max(by0, 0), min(by1, Hd)):
            for xx in range(max(bx0, 0), min(bx1, Wd)):
                qx = xx + 0.5 - xa
                qy = yy + 0.5 - ya
                h = 0.0
                if L2 > 1e-6:
                    h = clamp((qx * lx + qy * ly) / L2, 0.0, 1.0)
                ddx = qx - lx * h
                ddy = qy - ly * h
                g = math.exp(-(ddx * ddx + ddy * ddy) * inv) * norm * att
                if g < 1e-5:
                    continue
                if P[k, 10] > 0.0 and depth[yy, xx] < zm - 0.05:
                    continue
                img[yy, xx, 0] += g * P[k, 7]
                img[yy, xx, 1] += g * P[k, 8]
                img[yy, xx, 2] += g * P[k, 9]


@njit(**FM)
def dark_sprites(img, depth, cam, Q, n):
    """Q[k] = x,y,z, heading(rad), length_m, width_m, alpha, lit(0..1 warm rim), torch side (+-1)."""
    Hd = img.shape[0]
    Wd = img.shape[1]
    f = cam[12]
    for k in range(n):
        xs, ys, zc = _proj(cam, Q[k, 0], Q[k, 1], Q[k, 2])
        if zc <= 0.0:
            continue
        # heading direction in screen space
        hx = math.cos(Q[k, 3])
        hy = math.sin(Q[k, 3])
        xs2, ys2, zc2 = _proj(cam, Q[k, 0] + hx, Q[k, 1] + hy, Q[k, 2])
        ux = xs2 - xs
        uy = ys2 - ys
        ul = math.sqrt(ux * ux + uy * uy) + 1e-9
        ux /= ul
        uy /= ul
        sc = f / zc
        a = 0.5 * Q[k, 4] * sc
        b = 0.5 * Q[k, 5] * sc
        if b < 0.35:
            continue
        ext = int(max(a, b) * 1.6) + 2
        alpha = Q[k, 6]
        for yy in range(max(int(ys) - ext, 0), min(int(ys) + ext + 1, Hd)):
            for xx in range(max(int(xs) - ext, 0), min(int(xs) + ext + 1, Wd)):
                qx = xx + 0.5 - xs
                qy = yy + 0.5 - ys
                along = qx * ux + qy * uy
                side = -qx * uy + qy * ux
                # body: ellipse (shoulders) + head bump forward
                e1 = (along / a) ** 2 + (side / b) ** 2
                e2 = ((along - 0.25 * a) / (0.55 * b)) ** 2 + (side / (0.55 * b)) ** 2
                e = min(e1, e2 * 1.0)
                cov = clamp((1.0 - e) * b * 0.9 + 0.5, 0.0, 1.0)
                if cov <= 0.0:
                    continue
                cov *= alpha
                shade = 0.02
                rim = Q[k, 7] * sstep(0.2, 1.0, (Q[k, 8] * side / b)) * sstep(0.0, 0.6, e)
                img[yy, xx, 0] = img[yy, xx, 0] * (1 - cov) + cov * (shade * 0.5 + rim * 0.9)
                img[yy, xx, 1] = img[yy, xx, 1] * (1 - cov) + cov * (shade * 0.5 + rim * 0.42)
                img[yy, xx, 2] = img[yy, xx, 2] * (1 - cov) + cov * (shade * 0.6 + rim * 0.12)


@njit(parallel=True, **FM)
def mist_layers(img, cam, layers, nl, T, irr_c, ig_x0, ig_cell, warm):
    """layers[k] = z, opacity, scale(m), drift_x, drift_y, seed. Composited far-to-near."""
    Hd = img.shape[0]
    Wd = img.shape[1]
    Cx, Cy, Cz = cam[0], cam[1], cam[2]
    f = cam[12]
    for y in prange(Hd):
        for x in range(Wd):
            sx = (x + 0.5 - cam[13]) / f
            sy = -(y + 0.5 - cam[14]) / f
            dx = cam[9] + sx * cam[3] + sy * cam[6]
            dy = cam[10] + sx * cam[4] + sy * cam[7]
            dz = cam[11] + sx * cam[5] + sy * cam[8]
            for k in range(nl):
                zl = layers[k, 0]
                if Cz <= zl + 1.0 or dz >= 0.0:
                    continue
                t = (zl - Cz) / dz
                px = Cx + dx * t
                py = Cy + dy * t
                s = layers[k, 2]
                fp = t / f / s
                qx = px + layers[k, 3] * (T - 1912.0)
                qy = py + layers[k, 4] * (T - 1912.0)
                n = 0.5 + fbm2(qx / s, qy / s, int(layers[k, 5]), 6, 2.05, 0.55, fp)
                d = sstep(0.42, 0.85, n)
                near = sstep(2.0, 45.0, Cz - zl)
                a = d * layers[k, 1] * near
                if a <= 0.0:
                    continue
                # colour: moonlit + warm glow of torch rivers below
                w = grid_sample(irr_c, ig_x0, ig_x0, ig_cell, px, py)
                mr = 0.020 + warm * w * 1.0
                mg = 0.026 + warm * w * 0.45
                mb = 0.040 + warm * w * 0.12
                img[y, x, 0] = img[y, x, 0] * (1 - a) + a * mr
                img[y, x, 1] = img[y, x, 1] * (1 - a) + a * mg
                img[y, x, 2] = img[y, x, 2] * (1 - a) + a * mb


@njit(parallel=True, **FM)
def motion_blur(src, dst, depth, cam, camA, camB, nsamp, maxlen):
    """Per-pixel blur along the screen path of the surface point between camA and camB."""
    Hd = src.shape[0]
    Wd = src.shape[1]
    f = cam[12]
    for y in prange(Hd):
        for x in range(Wd):
            sx = (x + 0.5 - cam[13]) / f
            sy = -(y + 0.5 - cam[14]) / f
            dx = cam[9] + sx * cam[3] + sy * cam[6]
            dy = cam[10] + sx * cam[4] + sy * cam[7]
            dz = cam[11] + sx * cam[5] + sy * cam[8]
            l = math.sqrt(dx * dx + dy * dy + dz * dz)
            t = depth[y, x]
            px = cam[0] + dx / l * t
            py = cam[1] + dy / l * t
            pz = cam[2] + dz / l * t
            xa, ya, za = _proj(camA, px, py, pz)
            xb, yb, zb = _proj(camB, px, py, pz)
            if za <= 0 or zb <= 0:
                dst[y, x, 0] = src[y, x, 0]
                dst[y, x, 1] = src[y, x, 1]
                dst[y, x, 2] = src[y, x, 2]
                continue
            # offsets relative to this pixel's current position
            ox = xa - (x + 0.5)
            oy = ya - (y + 0.5)
            qx = xb - (x + 0.5)
            qy = yb - (y + 0.5)
            L = math.sqrt((qx - ox) ** 2 + (qy - oy) ** 2)
            if L < 0.75:
                dst[y, x, 0] = src[y, x, 0]
                dst[y, x, 1] = src[y, x, 1]
                dst[y, x, 2] = src[y, x, 2]
                continue
            if L > maxlen:
                sc = maxlen / L
                ox *= sc
                oy *= sc
                qx *= sc
                qy *= sc
            n = min(nsamp, int(L) + 2)
            ar = 0.0
            ag = 0.0
            ab = 0.0
            wsum = 0.0
            for k in range(n):
                u = (k + 0.5) / n
                # sample backwards along the motion (gather)
                sxp = x - (ox + (qx - ox) * u)
                syp = y - (oy + (qy - oy) * u)
                ix = int(sxp + 0.5)
                iy = int(syp + 0.5)
                if ix < 0 or iy < 0 or ix >= Wd or iy >= Hd:
                    continue
                ar += src[iy, ix, 0]
                ag += src[iy, ix, 1]
                ab += src[iy, ix, 2]
                wsum += 1.0
            if wsum > 0:
                dst[y, x, 0] = ar / wsum
                dst[y, x, 1] = ag / wsum
                dst[y, x, 2] = ab / wsum
            else:
                dst[y, x, 0] = src[y, x, 0]
                dst[y, x, 1] = src[y, x, 1]
                dst[y, x, 2] = src[y, x, 2]
