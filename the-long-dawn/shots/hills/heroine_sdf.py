"""FIRST BEACON heroine (v2): a true 3-D SDF figure, sphere-traced per pixel.

Replaces the 2-D pillow-shaded card puppet for the Young Woman in FIRST BEACON only (the Elder,
the Child and every other shot keep puppet.py). The figure is a list of primitives (ellipsoid,
round cone, round box, elliptical torus) grouped into smooth-union groups (skin, coat, scarf,
hair, hands ...); groups are hard-unioned. Rendering:

* tiles of 16x16 px get the primitives whose projected bounding spheres touch them; each ray
  keeps only the ones its own path crosses, so a sample evaluates a handful of primitives;
* pass 1 = one sample per pixel; pass 2 re-renders silhouette / material / depth edges with
  a 3x3 grid (anti-aliasing without paying 9x everywhere);
* lighting is physical-ish: point lights (strike flash, ember, flame, the roaring fire) with
  inverse-square falloff and SDF soft shadows, wrap/sub-surface skin, GGX specular, wool
  sheen, a thin cold rim from behind (sky/moon), faint sky ambient and warm snow bounce, AO.

Everything is linear-light HDR, premultiplied RGBA out, plus a depth buffer (for strands).
"""
import math

import numba as nb
import numpy as np

from core import cam_ray, perlin3

T_ELL, T_CONE, T_BOX, T_TORUS, T_PLANE, T_LID = 0, 1, 2, 3, 4, 5
NPR = 32            # floats per primitive row
MAXC = 160          # max candidates per ray

# material ids
M_SKIN, M_EYE, M_HAIR, M_COAT, M_SCARF, M_CLOTH, M_BOOT, M_STEEL, M_FLINT, M_NAIL, M_KNIT = range(11)
NMAT = 11
# material table columns
# 0-2 albedo, 3 rough, 4 F0, 5 wrap, 6 sheen, 7 translucency, 8 rim gain, 9 ambient gain,
# 10 metal, 11 texture type, 12 tex scale (1/m), 13 tex albedo amp, 14 spare, 15 spare


# ------------------------------------------------------------------ SDF ---

@nb.njit(cache=True, fastmath=True, inline='always')
def _smin(a, b, k):
    if k <= 0.0:
        return a if a < b else b
    h = k - abs(a - b)
    if h < 0.0:
        h = 0.0
    h /= k
    m = a if a < b else b
    return m - h * h * k * 0.25


@nb.njit(cache=True, fastmath=True, inline='always')
def _smax(a, b, k):
    return -_smin(-a, -b, k)


@nb.njit(cache=True, fastmath=True, inline='always')
def _sgn(x):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@nb.njit(cache=True, fastmath=True, inline='always')
def _sd_ell(x, y, z, a, b, c):
    k0 = math.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2)
    if k0 < 1.0:
        # inside: IQ's estimate is poor for flat ellipsoids; use a conservative bound
        return (k0 - 1.0) * min(a, min(b, c))
    k1 = math.sqrt((x / (a * a)) ** 2 + (y / (b * b)) ** 2 + (z / (c * c)) ** 2)
    if k1 < 1e-12:
        return -min(a, min(b, c))
    return k0 * (k0 - 1.0) / k1


@nb.njit(cache=True, fastmath=True, inline='always')
def _sd_ell2(x, y, a, b):
    k0 = math.sqrt((x / a) ** 2 + (y / b) ** 2)
    k1 = math.sqrt((x / (a * a)) ** 2 + (y / (b * b)) ** 2)
    if k1 < 1e-12:
        return -min(a, b)
    return k0 * (k0 - 1.0) / k1


@nb.njit(cache=True, fastmath=True)
def _sd_rcone(px, py, pz, ax, ay, az, bx, by, bz, r1, r2):
    """IQ's exact round cone between spheres (a, r1) and (b, r2)."""
    bax = bx - ax
    bay = by - ay
    baz = bz - az
    l2 = bax * bax + bay * bay + baz * baz
    pax = px - ax
    pay = py - ay
    paz = pz - az
    if l2 < 1e-12:
        return math.sqrt(pax * pax + pay * pay + paz * paz) - max(r1, r2)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    y = pax * bax + pay * bay + paz * baz
    z = y - l2
    qx = pax * l2 - bax * y
    qy = pay * l2 - bay * y
    qz = paz * l2 - baz * y
    x2 = qx * qx + qy * qy + qz * qz
    y2 = y * y * l2
    z2 = z * z * l2
    k = _sgn(rr) * rr * rr * x2
    if _sgn(z) * a2 * z2 > k:
        return math.sqrt(x2 + z2) * il2 - r2
    if _sgn(y) * a2 * y2 < k:
        return math.sqrt(x2 + y2) * il2 - r1
    if a2 < 1e-12:
        return math.sqrt(pax * pax + pay * pay + paz * paz) - max(r1, r2)
    return (math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1


@nb.njit(cache=True, fastmath=True)
def prim_sd(P, i, px, py, pz):
    typ = int(P[i, 0])
    if typ == 1:
        d = _sd_rcone(px, py, pz, P[i, 5], P[i, 6], P[i, 7], P[i, 8], P[i, 9], P[i, 10], P[i, 20], P[i, 21])
        amp = P[i, 23]
        if amp != 0.0 and d < 0.03:
            # folds: rings along the axis, wobbling with the angle around it, inside an envelope
            ax = P[i, 8] - P[i, 5]
            ay = P[i, 9] - P[i, 6]
            az = P[i, 10] - P[i, 7]
            L = math.sqrt(ax * ax + ay * ay + az * az) + 1e-9
            wx = px - P[i, 5]
            wy = py - P[i, 6]
            wz = pz - P[i, 7]
            s = (wx * ax + wy * ay + wz * az) / (L * L)
            e1 = wx * P[i, 11] + wy * P[i, 12] + wz * P[i, 13]
            e2 = wx * P[i, 14] + wy * P[i, 15] + wz * P[i, 16]
            ang = math.atan2(e2, e1)
            env = math.exp(-((s - P[i, 26]) / P[i, 27]) ** 2)
            ph = s * L / P[i, 24] * 6.2831853 + P[i, 25] + P[i, 28] * math.sin(ang + P[i, 29]) \
                + 0.8 * perlin3(s * 3.0 + P[i, 25], ang * 0.8, P[i, 29])
            lip = 1.0 + amp * 6.2831853 / P[i, 24] * 0.9
            d = (d - amp * env * (0.5 + 0.5 * math.sin(ph)) ** 1.5) / lip
        return d
    lx = px - P[i, 5]
    ly = py - P[i, 6]
    lz = pz - P[i, 7]
    x = P[i, 11] * lx + P[i, 12] * ly + P[i, 13] * lz
    y = P[i, 14] * lx + P[i, 15] * ly + P[i, 16] * lz
    z = P[i, 17] * lx + P[i, 18] * ly + P[i, 19] * lz
    if typ == 0:
        return _sd_ell(x, y, z, P[i, 8], P[i, 9], P[i, 10])
    if typ == 2:
        r = P[i, 20]
        qx = abs(x) - P[i, 8] + r
        qy = abs(y) - P[i, 9] + r
        qz = abs(z) - P[i, 10] + r
        ox = qx if qx > 0.0 else 0.0
        oy = qy if qy > 0.0 else 0.0
        oz = qz if qz > 0.0 else 0.0
        return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0) - r
    if typ == 4:
        # half-space: local +x is outside (use with op=intersect to keep x<0); a gentle
        # quadratic bend along local y and z lets a hairline curve round the head
        gy = 2.0 * P[i, 20] * y
        gz = 2.0 * P[i, 21] * z
        return (x + P[i, 20] * y * y + P[i, 21] * z * z) / math.sqrt(1.0 + gy * gy + gz * gz)
    if typ == 5:
        # eyelids: a thin shell (R_in..R_out) round the eyeball, minus the palpebral fissure
        # cut in angular coords (x = look direction, y = up, z = side). The fissure is open
        # where lo(az) < elevation < up(az), |az| < azmax; margins follow cos-shaped curves.
        rr = math.sqrt(x * x + y * y + z * z) + 1e-9
        shell = max(rr - P[i, 21], P[i, 20] - rr)
        az = math.atan2(z, x)
        el = math.asin(max(-1.0, min(1.0, y / rr)))
        am = P[i, 24] if az >= 0.0 else P[i, 25]
        t = az / am
        if abs(t) < 1.0:
            # almond: both margins converge on the canthi (no cut at the ends)
            c = math.cos(t * 1.5707963)
            cu = c ** 0.55
            cl = c ** 0.75
            up = P[i, 22] * cu + P[i, 26] * t
            lo = P[i, 23] * cl + P[i, 26] * t
            fis = max(el - up, lo - el)
        else:
            tl = P[i, 26] * (1.0 if t > 0.0 else -1.0)
            fis = max(abs(el - tl), (abs(az) - am) * math.cos(el))
        if P[i, 27] > 0.5:
            # fissure-carve mode: the opening itself, through the radial band (use op=1)
            return max(fis * rr, shell)
        return max(shell, -fis * rr)
    # elliptical torus around local y; optional angular cut (P[22] = major R, 24/25 = arc)
    q = math.sqrt(x * x + z * z) - P[i, 22]
    if P[i, 28] != 0.0:
        # tilt the section (radial, axial) by P[28] radians
        ct = math.cos(P[i, 28])
        st = math.sin(P[i, 28])
        q, y = ct * q + st * y, -st * q + ct * y
    if P[i, 25] < 3.1415:
        # an arc |angle - centre| < half (angle in local x-z); P[26] > 0 tapers the section
        # to nothing at the arc ends (lips round the dental arch, scarf loops)
        a = math.atan2(z, x) - P[i, 24]
        while a > 3.14159265:
            a -= 6.2831853
        while a < -3.14159265:
            a += 6.2831853
        cut = (abs(a) - P[i, 25]) * max(P[i, 22], 1e-3)
        if P[i, 26] > 0.0:
            ra = a / P[i, 25]
            f = 1.0 - ra * ra
            if f < 0.0025:
                f = 0.0025
            f = math.sqrt(f) ** P[i, 26]
            d = _sd_ell2(q, y - P[i, 27] * ra * ra, P[i, 20] * f, P[i, 21] * f)
            return max(d, cut)
        d = _sd_ell2(q, y, P[i, 20], P[i, 21])
        return max(d, cut)
    return _sd_ell2(q, y, P[i, 20], P[i, 21])


@nb.njit(cache=True, fastmath=True)
def group_disp(G, g, px, py, pz):
    """Small surface relief per group (wool felt, knit, hair grooves). Returns metres."""
    typ = int(G[g, 0])
    if typ == 0:
        return 0.0
    amp = G[g, 1]
    sc = G[g, 2]
    if typ == 1:      # felted wool: two octaves of isotropic noise
        return amp * (perlin3(px * sc, py * sc, pz * sc) + 0.5 * perlin3(px * sc * 2.3 + 7.1, py * sc * 2.3, pz * sc * 2.3 - 3.3))
    if typ == 2:      # knit: ribs across a direction + noise
        dx = G[g, 3]
        dy = G[g, 4]
        dz = G[g, 5]
        u = (px * dx + py * dy + pz * dz) * sc
        rib = abs(math.sin(u * 3.14159265))
        return amp * (0.7 * rib + 0.5 * perlin3(px * sc * 0.6, py * sc * 0.6, pz * sc * 0.6))
    # hair grooves: fine stripes across the sweep direction (noise stretched along it)
    dx = G[g, 3]
    dy = G[g, 4]
    dz = G[g, 5]
    a = px * dx + py * dy + pz * dz
    ex = px - a * dx
    ey = py - a * dy
    ez = pz - a * dz
    return amp * perlin3(ex * sc, ey * sc + a * sc * 0.06, ez * sc)


@nb.njit(cache=True, fastmath=True)
def sdf(P, G, cand, nc, px, py, pz):
    """Distance + group id at a point, over a candidate list (sorted by prim index; prims are
    stored group-contiguous, unions before subtractions/intersections)."""
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
                    gd += group_disp(G, cur, px, py, pz)
                if gd < best:
                    best = gd
                    bg = cur
            cur = g
            gd = 1e9
        d = prim_sd(P, i, px, py, pz)
        op = int(P[i, 3])
        if op == 0:
            gd = _smin(gd, d, P[i, 2])
        elif op == 1:
            gd = _smax(gd, -d, P[i, 2])
        else:
            gd = _smax(gd, d, P[i, 2])
    if cur >= 0 and gd < 1e8:
        if gd < G[cur, 6]:
            gd += group_disp(G, cur, px, py, pz)
        if gd < best:
            best = gd
            bg = cur
    return best, bg


@nb.njit(cache=True, fastmath=True)
def ray_cands(BS, lst, nl, ox, oy, oz, dx, dy, dz, tmax, out, skip_near, P, G):
    """Candidates whose bounding sphere the ray (segment 0..tmax) crosses. Returns (n, t0, t1)."""
    n = 0
    t0 = 1e9
    t1 = -1.0
    for q in range(nl):
        i = lst[q]
        if skip_near and G[int(P[i, 1]), 8] > 0.5:
            continue
        cx = BS[i, 0] - ox
        cy = BS[i, 1] - oy
        cz = BS[i, 2] - oz
        R = BS[i, 3]
        tc = cx * dx + cy * dy + cz * dz
        d2 = cx * cx + cy * cy + cz * cz - tc * tc
        if d2 > R * R:
            continue
        h = math.sqrt(R * R - d2)
        te = tc - h
        tx = tc + h
        if tx < 0.0 or te > tmax:
            continue
        if n < MAXC:
            out[n] = i
            n += 1
        if int(P[i, 3]) == 0:
            if te < t0:
                t0 = te
            if tx > t1:
                t1 = tx
    return n, t0, t1


@nb.njit(cache=True, fastmath=True)
def calc_normal(P, G, cand, nc, px, py, pz, e):
    k1, _ = sdf(P, G, cand, nc, px + e, py - e, pz - e)
    k2, _ = sdf(P, G, cand, nc, px - e, py - e, pz + e)
    k3, _ = sdf(P, G, cand, nc, px - e, py + e, pz - e)
    k4, _ = sdf(P, G, cand, nc, px + e, py + e, pz + e)
    nx = k1 - k2 - k3 + k4
    ny = -k1 - k2 + k3 + k4
    nz = -k1 + k2 - k3 + k4
    nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / nn, ny / nn, nz / nn


@nb.njit(cache=True, fastmath=True)
def soft_shadow(P, G, BS, allidx, nall, px, py, pz, lx, ly, lz, dist, k, buf):
    # k < 0: a source held in her hands (strike, ember, tinder flame): hands and tools cast no shadow
    skip = k < 0.0
    if skip:
        k = -k
    nc, t0, t1 = ray_cands(BS, allidx, nall, px, py, pz, lx, ly, lz, dist, buf, skip, P, G)
    if nc == 0:
        return 1.0
    res = 1.0
    t = 0.002
    if t0 > t:
        t = t0
    tend = min(dist, t1)
    ph = 1e10
    for it in range(72):
        if t >= tend:
            break
        h, _ = sdf(P, G, buf, nc, px + lx * t, py + ly * t, pz + lz * t)
        if h < 0.0002:
            return 0.0
        y = h * h / (2.0 * ph)
        d = math.sqrt(max(h * h - y * y, 0.0))
        r = k * d / max(1e-4, t - y)
        if r < res:
            res = r
        ph = h
        t += min(max(h * 0.9, 0.0006), 0.05)
    res = min(max(res, 0.0), 1.0)
    return res * res * (3.0 - 2.0 * res)


@nb.njit(cache=True, fastmath=True)
def calc_ao(P, G, cand, nc, px, py, pz, nx, ny, nz, scale):
    occ = 0.0
    sca = 1.0
    for q in range(5):
        h = scale * (0.12 + 0.22 * q)
        d, _ = sdf(P, G, cand, nc, px + nx * h, py + ny * h, pz + nz * h)
        occ += (h - d) * sca
        sca *= 0.72
    v = 1.0 - 1.6 * occ / scale
    return min(max(v, 0.0), 1.0)


# -------------------------------------------------------------- shading ---

@nb.njit(cache=True, fastmath=True, inline='always')
def _ggx(ndh, a):
    a2 = a * a
    d = ndh * ndh * (a2 - 1.0) + 1.0
    return a2 / (3.14159265 * d * d + 1e-9)


@nb.njit(cache=True, fastmath=True)
def tex_mod(M, m, px, py, pz):
    """Albedo multiplier from the material's texture."""
    tt = int(M[m, 11])
    if tt == 0:
        return 1.0
    sc = M[m, 12]
    a = M[m, 13]
    if tt == 1:     # wool / cloth mottling
        n = perlin3(px * sc, py * sc, pz * sc) * 0.6 + perlin3(px * sc * 3.1, py * sc * 3.1, pz * sc * 3.1) * 0.4
        return max(0.2, 1.0 + a * n)
    if tt == 2:     # skin: faint blotch
        n = perlin3(px * sc, py * sc, pz * sc)
        return 1.0 + a * n
    return 1.0


@nb.njit(cache=True, fastmath=True)
def shade(P, G, BS, allidx, nall, M, L, nl, env, H, cand, nc, buf,
          px, py, pz, nx, ny, nz, vx, vy, vz, g, out):
    """out[0:3] <- linear radiance. L rows: x,y,z, r,g,b (intensity at 1 m), radius, soft_k.
    env: 0-2 rim dir, 3-5 rim rgb, 6-8 sky amb rgb, 9-11 bounce rgb, 12 ao scale,
    13 shadows on, 14 exposure-free spare. H: head params (see tints)."""
    m = int(G[g, 7])
    ar = M[m, 0]
    ag = M[m, 1]
    ab = M[m, 2]
    tm = tex_mod(M, m, px, py, pz)
    ar *= tm
    ag *= tm
    ab *= tm
    if M[m, 14] > 0.0:
        # fabric / stone relief as a bump map (no displacement: no acne, no march cost)
        sc = M[m, 12] * 1.9
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
    # ---- region tints on skin (lips, brows, cheek) & the eye (iris/pupil) from head space
    gloss_boost = 0.0
    if m == M_SKIN and H[0] > 0.5:
        # head-local coords: u fwd, v up, w lateral
        lx = px - H[1]
        ly = py - H[2]
        lz = pz - H[3]
        u = lx * H[4] + ly * H[5] + lz * H[6]
        v = lx * H[7] + ly * H[8] + lz * H[9]
        w = lx * H[10] + ly * H[11] + lz * H[12]
        # lips: vermilion band around the mouth line, fading laterally
        dv = (v - H[13]) / H[14]
        lipw = math.exp(-dv * dv * dv * dv) * math.exp(-(w / H[15]) ** 4) * (1.0 if u > H[16] else math.exp(-((u - H[16]) / 0.004) ** 2))
        ar = ar * (1.0 - lipw) + H[17] * lipw
        ag = ag * (1.0 - lipw) + H[18] * lipw
        ab = ab * (1.0 - lipw) + H[19] * lipw
        gloss_boost = 0.5 * lipw
        # brows: a dark, hairy band over each eye
        bw = abs(w) - H[21]
        bv = v - H[20] + 0.10 * bw * bw / 0.03
        brow = math.exp(-(bv / 0.0042) ** 2) * math.exp(-(bw / 0.018) ** 4) * (1.0 if u > H[22] else 0.0)
        brow *= 0.75 + 0.25 * perlin3(u * 900.0, v * 250.0, w * 900.0)
        ar *= 1.0 - 0.80 * brow
        ag *= 1.0 - 0.84 * brow
        ab *= 1.0 - 0.86 * brow
        # soft warm flush on the cheek and nose tip (life in the cold)
        cw = math.exp(-((u - H[23]) / 0.02) ** 2 - ((v - H[24]) / 0.018) ** 2 - ((abs(w) - H[25]) / 0.02) ** 2)
        nw = math.exp(-((u - 0.112) / 0.012) ** 2 - ((v + 0.037) / 0.012) ** 2 - (w / 0.014) ** 2)
        cw = min(1.0, 1.2 * cw + 1.1 * nw)
        ar *= 1.0 + 0.16 * cw
        ag *= 1.0 - 0.14 * cw
        ab *= 1.0 - 0.10 * cw
    if m == M_SKIN and H[40] > 0.5:
        # cold hands: red round the knuckles and fingertips
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
    if m == M_EYE and H[0] > 0.5:
        # which eye: nearest centre
        best = 1e9
        ex = 0.0
        ey = 0.0
        ez = 0.0
        for q in range(2):
            cx = H[26 + 3 * q]
            cy = H[27 + 3 * q]
            cz = H[28 + 3 * q]
            dd = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
            if dd < best:
                best = dd
                ex = px - cx
                ey = py - cy
                ez = pz - cz
        ee = math.sqrt(ex * ex + ey * ey + ez * ez) + 1e-9
        cg = (ex * H[32] + ey * H[33] + ez * H[34]) / ee     # cos to gaze
        iris = 1.0 / (1.0 + math.exp(-(cg - 0.865) * 70.0))
        pupil = 1.0 / (1.0 + math.exp(-(cg - 0.960) * 110.0))
        ar = ar * (1.0 - iris) + 0.050 * iris
        ag = ag * (1.0 - iris) + 0.026 * iris
        ab = ab * (1.0 - iris) + 0.013 * iris
        ar *= 1.0 - 0.85 * pupil
        ag *= 1.0 - 0.85 * pupil
        ab *= 1.0 - 0.85 * pupil
    # hair: strand tangent (group sweep direction projected on the surface) + strand stripes
    hair = m == M_HAIR
    htx = 0.0
    hty = 0.0
    htz = 0.0
    if hair:
        htx = G[g, 3]
        hty = G[g, 4]
        htz = G[g, 5]
        dn = htx * nx + hty * ny + htz * nz
        htx -= dn * nx
        hty -= dn * ny
        htz -= dn * nz
        tn = math.sqrt(htx * htx + hty * hty + htz * htz) + 1e-9
        htx /= tn
        hty /= tn
        htz /= tn
        al = px * htx + py * hty + pz * htz
        qx = px - al * htx
        qy = py - al * hty
        qz = pz - al * htz
        st = perlin3(qx * 700.0, qy * 700.0 + al * 18.0, qz * 700.0) + 0.5 * perlin3(qx * 1900.0, qy * 1900.0 + al * 40.0, qz * 1900.0)
        sm = 0.45 + 1.3 * max(-0.5, min(0.5, st))
        ar *= sm
        ag *= sm
        ab *= sm
    ndv = nx * vx + ny * vy + nz * vz
    if ndv < 0.0:
        ndv = 0.0
    r = 0.0
    gg = 0.0
    b = 0.0
    ao = 1.0
    if env[12] > 0.0:
        ao = calc_ao(P, G, cand, nc, px + nx * 0.0005, py + ny * 0.0005, pz + nz * 0.0005, nx, ny, nz, env[12])
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
        # wrapped diffuse per channel (red wraps furthest: light bleeding under skin)
        wr = wrap
        wg = wrap * 0.55
        wb = wrap * 0.35
        dr = max(0.0, (ndl + wr) / (1.0 + wr))
        dg = max(0.0, (ndl + wg) / (1.0 + wg))
        db = max(0.0, (ndl + wb) / (1.0 + wb))
        if dr <= 0.0 and trans <= 0.0 and sheen <= 0.0:
            continue
        sh = 1.0
        if env[13] > 0.0 and L[q, 7] != 0.0 and (dr > 0.0 or trans > 0.0):
            sh = soft_shadow(P, G, BS, allidx, nall, px + nx * 0.0025 + lx * 0.002, py + ny * 0.0025 + ly * 0.002,
                             pz + nz * 0.0025 + lz * 0.002,
                             lx, ly, lz, dist - L[q, 6] * 0.5, L[q, 7], buf)
            # light leaking through thin parts still reaches the far side a little
        shr = sh
        shg = sh
        shb = sh
        if m == M_SKIN and sh < 1.0:
            # sub-surface: shadow terminators go red, not black
            shr = sh + (1.0 - sh) * 0.10
            shg = sh + (1.0 - sh) * 0.02
        Er = L[q, 3] * E
        Eg = L[q, 4] * E
        Eb = L[q, 5] * E
        cr = ar * dr * shr * (1.0 - metal)
        cgg = ag * dg * shg * (1.0 - metal)
        cb = ab * db * shb * (1.0 - metal)
        # specular (GGX, Schlick)
        hx = lx + vx
        hy = ly + vy
        hz = lz + vz
        hn = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
        ndh = max(0.0, (nx * hx + ny * hy + nz * hz) / hn)
        vdh = max(0.0, (vx * hx + vy * hy + vz * hz) / hn)
        rgh = rough * (1.0 - 0.5 * gloss_boost)
        a = max(0.02, rgh * rgh)
        F = f0 + (1.0 - f0) * (1.0 - vdh) ** 5
        spec = 0.0
        if hair:
            # Kajiya-Kay: highlight where the strand is perpendicular to the half vector
            tdh = (htx * hx + hty * hy + htz * hz) / hn
            sth = math.sqrt(max(0.0, 1.0 - tdh * tdh))
            spec = 0.10 * sth ** 50.0 + 0.05 * sth ** 12.0
            spec *= max(0.0, ndl + 0.2) * sh
        elif ndl > 0.0:
            spec = _ggx(ndh, a) * F * ndl / (4.0 * max(ndv, 0.08) * max(ndl, 0.08)) * sh
            spec *= 1.0 + 1.5 * gloss_boost
        sr = spec
        sg = spec
        sb = spec
        if metal > 0.0:
            sr *= ar * 8.0
            sg *= ag * 8.0
            sb *= ab * 8.0
        # sheen / fuzz: fibres catching grazing light (wool, hair)
        fz = 0.0
        if sheen > 0.0:
            fz = sheen * (1.0 - ndv) ** 3 * max(0.0, ndl + 0.35) * (0.4 + 0.6 * sh)
        # translucency: thin parts glow when the light is behind them
        tr = 0.0
        if trans > 0.0 and ndl < 0.2:
            back = max(0.0, -(vx * lx + vy * ly + vz * lz))
            tr = trans * (0.25 + back * back) * max(0.0, 0.2 - ndl) * sh
        r += Er * (cr + sr + fz * ar * 2.0 + tr * ar * 3.0)
        gg += Eg * (cgg + sg + fz * ag * 2.0 + tr * ag * 1.2)
        b += Eb * (cb + sb + fz * ab * 2.0 + tr * ab * 0.8)
    # cold rim from behind (sky/moon): thin, only on edges turned toward it
    ndr = nx * env[0] + ny * env[1] + nz * env[2]
    if ndr > 0.0:
        rim = M[m, 8] * ndr * (1.0 - ndv) ** 5
        if m == M_HAIR or m == M_COAT or m == M_SCARF or m == M_KNIT:
            rim *= 1.3
        r += env[3] * rim * (0.35 + ar * 2.0)
        gg += env[4] * rim * (0.35 + ag * 2.0)
        b += env[5] * rim * (0.35 + ab * 2.0)
    # sky ambient (from above) + warm snow bounce (from below), both occluded
    up = 0.5 + 0.5 * ny
    amb = M[m, 9] * ao
    r += amb * ar * (env[6] * up + env[9] * (1.0 - up) * (1.0 - up))
    gg += amb * ag * (env[7] * up + env[10] * (1.0 - up) * (1.0 - up))
    b += amb * ab * (env[8] * up + env[11] * (1.0 - up) * (1.0 - up))
    out[0] = r
    out[1] = gg
    out[2] = b


# ------------------------------------------------------------- renderer ---

@nb.njit(cache=True, fastmath=True)
def trace_sample(cam, P, G, BS, M, L, nl, env, H, lst, nlst, allidx, nall, cand, buf,
                 sx, sy, res):
    """One camera sample at render-pixel coords (sx, sy). res <- r,g,b,alpha,depth,group."""
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
    nc, t0, t1 = ray_cands(BS, lst, nlst, ox, oy, oz, dx, dy, dz, 1e9, cand, False, P, G)
    if nc == 0 or t1 < 0.0:
        return
    pixang = 1.0 / cam[12]
    t = max(t0, 0.02)
    hit = False
    qmin = 1e9
    tq = t
    for it in range(160):
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        d, g = sdf(P, G, cand, nc, px, py, pz)
        q = d / (t * pixang)
        if q < qmin:
            qmin = q
            tq = t
        if d < 0.2 * t * pixang or d < 2e-5:
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
    d, g = sdf(P, G, cand, nc, px, py, pz)
    if g < 0:
        return
    e = max(1e-4, 0.5 * tq * pixang)
    nx, ny, nz = calc_normal(P, G, cand, nc, px, py, pz, e)
    col = np.zeros(3)
    shade(P, G, BS, allidx, nall, M, L, nl, env, H, cand, nc, buf,
          px, py, pz, nx, ny, nz, -dx, -dy, -dz, g, col)
    res[0] = col[0] * cov
    res[1] = col[1] * cov
    res[2] = col[2] * cov
    res[3] = cov
    res[4] = tq
    res[5] = g


@nb.njit(cache=True, fastmath=True)
def render_region(cam, P, G, BS, M, L, nl, env, H, tile_off, tile_idx, allidx, X0, Y0, w, h, TS,
                  ss, rgb, alpha, depth):
    """Render the (w x h) region at render-pixel offset (X0, Y0). rgb premultiplied."""
    ntx = (w + TS - 1) // TS
    nty = (h + TS - 1) // TS
    cand = np.zeros(MAXC, np.int64)
    buf = np.zeros(MAXC, np.int64)
    res = np.zeros(6)
    grp = np.full((h, w), -1.0)
    nall = allidx.shape[0]
    # pass 1: one sample per pixel
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
                    trace_sample(cam, P, G, BS, M, L, nl, env, H, lst, a1 - a0, allidx, nall, cand, buf,
                                 X0 + i + 0.5, Y0 + j + 0.5, res)
                    rgb[j, i, 0] = res[0]
                    rgb[j, i, 1] = res[1]
                    rgb[j, i, 2] = res[2]
                    alpha[j, i] = res[3]
                    depth[j, i] = res[4]
                    grp[j, i] = res[5] if res[3] >= 0.999 else -2.0
    if ss <= 1:
        return
    # pass 2: supersample edges (coverage, group, depth or strong luminance steps)
    edge = np.zeros((h, w), np.uint8)
    for j in range(h):
        for i in range(w):
            a = alpha[j, i]
            if a > 0.0 and a < 0.999:
                edge[j, i] = 1
                continue
            g0 = grp[j, i]
            l0 = rgb[j, i, 0] + rgb[j, i, 1] + rgb[j, i, 2]
            for (oj, oi) in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                jj = j + oj
                ii = i + oi
                if jj < 0 or ii < 0 or jj >= h or ii >= w:
                    continue
                if grp[jj, ii] != g0:
                    edge[j, i] = 1
                    break
                if g0 >= 0.0:
                    if abs(depth[jj, ii] - depth[j, i]) > 0.012:
                        edge[j, i] = 1
                        break
                    l1 = rgb[jj, ii, 0] + rgb[jj, ii, 1] + rgb[jj, ii, 2]
                    if abs(l1 - l0) > 0.25 * (l0 + l1) + 0.02:
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
                            trace_sample(cam, P, G, BS, M, L, nl, env, H, lst, a1 - a0, allidx, nall, cand, buf,
                                         X0 + i + ox, Y0 + j + oy, res)
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


def build_tiles(cam, BS, X0, Y0, w, h, TS, P):
    """Per-tile candidate lists (primitive indices, ascending) from projected bounding spheres."""
    ntx = (w + TS - 1) // TS
    nty = (h + TS - 1) // TS
    sx, sy, z = cam.project(BS[:, :3])
    z = np.maximum(z, 1e-3)
    rpx = cam.f * BS[:, 3] / np.maximum(z - BS[:, 3], 0.02) + 2.0
    lists = [[] for _ in range(ntx * nty)]
    for i in range(len(BS)):
        x0 = int(np.floor((sx[i] - rpx[i] - X0) / TS))
        x1 = int(np.floor((sx[i] + rpx[i] - X0) / TS))
        y0 = int(np.floor((sy[i] - rpx[i] - Y0) / TS))
        y1 = int(np.floor((sy[i] + rpx[i] - Y0) / TS))
        for ty in range(max(0, y0), min(nty - 1, y1) + 1):
            for tx in range(max(0, x0), min(ntx - 1, x1) + 1):
                lists[ty * ntx + tx].append(i)
    off = np.zeros(ntx * nty + 1, np.int64)
    for k, l in enumerate(lists):
        off[k + 1] = off[k] + len(l)
    idx = np.zeros(max(1, off[-1]), np.int64)
    for k, l in enumerate(lists):
        idx[off[k]:off[k + 1]] = l
    return off, idx


# ---------------------------------------------------------------- tools ---

@nb.njit(cache=True, fastmath=True)
def sdf_points(P, G, pts, out_d, out_g, out_p):
    """Evaluate the full figure SDF at points (debug / profile extraction). out_p = nearest
    primitive index (by raw distance) inside the winning group."""
    n = P.shape[0]
    allc = np.arange(n)
    for k in range(pts.shape[0]):
        d, g = sdf(P, G, allc, n, pts[k, 0], pts[k, 1], pts[k, 2])
        out_d[k] = d
        out_g[k] = g
        best = 1e9
        bi = -1
        for i in range(n):
            if int(P[i, 1]) != g or int(P[i, 3]) != 0:
                continue
            di = abs(prim_sd(P, i, pts[k, 0], pts[k, 1], pts[k, 2]))
            if di < best:
                best = di
                bi = i
        out_p[k] = bi


@nb.njit(cache=True, fastmath=True)
def draw_strands(rgb, alpha, depth, X0, Y0, P2, Z, C, A, Wd):
    """Anti-aliased polyline strands composited OVER the figure layer (premultiplied), depth-tested
    against the figure's ray distances. P2 (ns, m, 2) absolute render px; Z (ns, m) ray distance;
    C (ns, m, 3) radiance; A (ns, m) opacity; Wd (ns, m) width in px. Caller sorts far -> near."""
    h = alpha.shape[0]
    w = alpha.shape[1]
    sig = 0.6
    inv = 1.0 / (2.0 * sig * sig)
    norm = 1.0 / (6.2831853 * sig * sig)
    for s in range(P2.shape[0]):
        for k in range(P2.shape[1] - 1):
            x0 = P2[s, k, 0] - X0
            y0 = P2[s, k, 1] - Y0
            x1 = P2[s, k + 1, 0] - X0
            y1 = P2[s, k + 1, 1] - Y0
            L = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            n = int(L / 0.45) + 1
            dl = L / n
            for q in range(n):
                u = (q + 0.5) / n
                x = x0 + (x1 - x0) * u
                y = y0 + (y1 - y0) * u
                z = Z[s, k] * (1 - u) + Z[s, k + 1] * u
                a_ = A[s, k] * (1 - u) + A[s, k + 1] * u
                wd = Wd[s, k] * (1 - u) + Wd[s, k + 1] * u
                cr = C[s, k, 0] * (1 - u) + C[s, k + 1, 0] * u
                cg = C[s, k, 1] * (1 - u) + C[s, k + 1, 1] * u
                cb = C[s, k, 2] * (1 - u) + C[s, k + 1, 2] * u
                ix = int(math.floor(x))
                iy = int(math.floor(y))
                for jj in range(iy - 1, iy + 2):
                    if jj < 0 or jj >= h:
                        continue
                    for ii in range(ix - 1, ix + 2):
                        if ii < 0 or ii >= w:
                            continue
                        if depth[jj, ii] < z - 0.008:
                            continue
                        dx = ii + 0.5 - x
                        dy = jj + 0.5 - y
                        wg = math.exp(-(dx * dx + dy * dy) * inv) * norm * wd * dl * a_
                        if wg > 0.95:
                            wg = 0.95
                        rgb[jj, ii, 0] = rgb[jj, ii, 0] * (1 - wg) + cr * wg
                        rgb[jj, ii, 1] = rgb[jj, ii, 1] * (1 - wg) + cg * wg
                        rgb[jj, ii, 2] = rgb[jj, ii, 2] * (1 - wg) + cb * wg
                        alpha[jj, ii] = alpha[jj, ii] * (1 - wg) + wg


@nb.njit(cache=True, fastmath=True)
def splat_fog(img, cx, cy, sig, amp, r, g, b, seed, t, fscale):
    """Thin lit fog puff: additive, noise-modulated gaussian. amp = peak opacity x scattered
    radiance; fscale = noise feature size in px."""
    H = img.shape[0]
    W = img.shape[1]
    rad = int(math.ceil(sig * 2.8))
    ix = int(cx)
    iy = int(cy)
    inv = 1.0 / (2.0 * sig * sig)
    k = 1.0 / max(fscale, 0.5)
    for j in range(max(0, iy - rad), min(H, iy + rad + 1)):
        dy = j + 0.5 - cy
        for i in range(max(0, ix - rad), min(W, ix + rad + 1)):
            dx = i + 0.5 - cx
            e = math.exp(-(dx * dx + dy * dy) * inv)
            if e < 0.01:
                continue
            n = perlin3(i * k + seed, j * k - seed * 0.7, t) + 0.5 * perlin3(i * k * 2.3 - seed, j * k * 2.3, t * 1.7)
            d = e * max(0.0, 0.62 + 1.0 * n)
            img[j, i, 0] += r * d * amp
            img[j, i, 1] += g * d * amp
            img[j, i, 2] += b * d * amp
