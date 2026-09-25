"""Night sky: palette gradient, moon + halo, Milky Way band, splatted star catalogue, aurora."""
import math

import numpy as np
from numba import njit, prange

from .noise import fbm2, fbm3, gnoise2, gnoise3, smoothstep, h01


def lin(hexcol):
    h = hexcol.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def sky_params(zenith='#070B1C', horizon='#27335E', moon_dir=(0.3, 0.4, 1.0), moon_col='#9DB4D9',
               moon_radius_deg=0.9, moon_I=6.0, halo_I=0.05, halo_w=0.25, halo2_I=0.02, halo2_w=0.9,
               horizon_glow=0.0, glow_col='#27335E', gain=1.0, below_col=None):
    """Pack sky parameters into a float64 array for numba."""
    z = lin(zenith) * gain
    h = lin(horizon) * gain
    b = lin(below_col) * gain if below_col else h * 0.6
    md = np.asarray(moon_dir, np.float64)
    md = md / np.linalg.norm(md)
    mc = lin(moon_col)
    g = lin(glow_col) * gain
    S = np.zeros(32)
    S[0:3] = z
    S[3:6] = h
    S[6:9] = md
    S[9:12] = mc
    S[12] = math.radians(moon_radius_deg)
    S[13] = moon_I
    S[14] = halo_I
    S[15] = halo_w
    S[16] = halo2_I
    S[17] = halo2_w
    S[18] = horizon_glow
    S[19:22] = g
    S[22:25] = b
    return S


@njit(inline='always', fastmath=True)
def sky_base(dx, dy, dz, S):
    """Sky radiance without moon disk/stars (used for reflections too)."""
    if dy < 0.0:
        u = min(-dy * 4.0, 1.0)
        r = S[3] * (1 - u) + S[22] * u
        g = S[4] * (1 - u) + S[23] * u
        b = S[5] * (1 - u) + S[24] * u
    else:
        u = smoothstep(0.0, 0.75, dy) ** 0.6
        r = S[3] * (1 - u) + S[0] * u
        g = S[4] * (1 - u) + S[1] * u
        b = S[5] * (1 - u) + S[2] * u
        gl = S[18] * math.exp(-dy * 18.0)
        r += S[19] * gl
        g += S[20] * gl
        b += S[21] * gl
    # moon halo (forward scattering in the atmosphere)
    cosm = dx * S[6] + dy * S[7] + dz * S[8]
    ang = math.acos(min(max(cosm, -1.0), 1.0))
    halo = S[14] * math.exp(-ang / S[15]) + S[16] * math.exp(-(ang / S[17]) ** 2)
    r += S[9] * halo
    g += S[10] * halo
    b += S[11] * halo
    return r, g, b


@njit(inline='always', fastmath=True)
def moon_disk(dx, dy, dz, S, pix_ang):
    """Moon disk radiance (with limb darkening + maria), anti-aliased by pix_ang."""
    cosm = dx * S[6] + dy * S[7] + dz * S[8]
    ang = math.acos(min(max(cosm, -1.0), 1.0))
    R = S[12]
    if ang > R + 2 * pix_ang:
        return 0.0, 0.0, 0.0
    cov = min(max((R - ang) / pix_ang + 0.5, 0.0), 1.0)
    # local disk coords for maria texture
    mx, my, mz = S[6], S[7], S[8]
    # build tangent frame
    tx, ty, tz = mz, 0.0, -mx
    tn = math.sqrt(tx * tx + tz * tz) + 1e-9
    tx /= tn
    tz /= tn
    bx = my * tz - mz * ty
    by = mz * tx - mx * tz
    bz = mx * ty - my * tx
    u = (dx * tx + dy * ty + dz * tz) / R
    v = (dx * bx + dy * by + dz * bz) / R
    rr = min(u * u + v * v, 1.0)
    limb = 0.55 + 0.45 * math.sqrt(1.0 - rr)
    m = fbm2(u * 1.7 + 3.0, v * 1.7 - 1.0, 4.0, 91)
    maria = 1.0 - 0.28 * smoothstep(-0.05, 0.25, m)
    I = S[13] * cov * limb * maria
    return S[9] * I * 1.25, S[10] * I * 1.08, S[11] * I


@njit(inline='always', fastmath=True)
def milky_way(dx, dy, dz, G, I):
    """G: [gx,gy,gz (plane normal), cx,cy,cz (galactic centre dir), width, seed]."""
    gx, gy, gz = G[0], G[1], G[2]
    sb = dx * gx + dy * gy + dz * gz
    b = math.asin(min(max(sb, -1.0), 1.0))
    # longitude
    cx, cy, cz = G[3], G[4], G[5]
    # e1 = centre dir, e2 = g x e1
    e2x = gy * cz - gz * cy
    e2y = gz * cx - gx * cz
    e2z = gx * cy - gy * cx
    l = math.atan2(dx * e2x + dy * e2y + dz * e2z, dx * cx + dy * cy + dz * cz)
    w = G[6] * (1.0 + 0.9 * math.exp(-(l / 0.5) ** 2))
    band = math.exp(-(b / w) ** 2)
    if band < 0.01:
        return 0.0, 0.0, 0.0
    bulge = 1.0 + 1.6 * math.exp(-(l / 0.35) ** 2) * math.exp(-(b / (w * 0.8)) ** 2)
    cl = fbm2(l * 5.0, b * 9.0, 6.0, int(G[7]))
    clouds = 0.55 + 0.9 * max(cl, -0.4)
    fine = fbm2(l * 22.0, b * 30.0, 4.0, int(G[7]) + 7)
    clouds *= 0.85 + 0.35 * fine
    # dust lanes near the plane (Great Rift)
    dn = fbm2(l * 7.0 + 11.0, b * 16.0, 5.0, int(G[7]) + 3)
    lane = smoothstep(0.02, 0.28, dn) * math.exp(-(b / (w * 0.45)) ** 2)
    dust = 1.0 - 0.75 * lane
    v = I * band * bulge * clouds * dust
    warm = math.exp(-(l / 0.6) ** 2)
    return v * (0.92 + 0.18 * warm), v * (0.95 + 0.05 * warm), v * (1.08 - 0.15 * warm)


def make_stars(n, seed, lum_scale=1.0, band=None, band_frac=0.0):
    """Star catalogue: unit directions (n,3), luminance (n,), colour (n,3).
    band: (normal vector, width_rad) to concentrate band_frac of stars near a great circle."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    if band is not None and band_frac > 0:
        nb = int(n * band_frac)
        g = np.asarray(band[0], np.float64)
        g /= np.linalg.norm(g)
        vv = rng.normal(size=(nb, 3))
        vv -= (vv @ g)[:, None] * g[None, :]
        vv /= np.linalg.norm(vv, axis=1, keepdims=True)
        lat = rng.normal(size=nb) * band[1]
        vv = vv * np.cos(lat)[:, None] + g[None, :] * np.sin(lat)[:, None]
        v[:nb] = vv
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # magnitude distribution: many faint, few bright
    u = rng.random(n)
    lum = 0.0025 * np.exp(-np.log(u + 1e-9) * 1.0) ** 1.35
    lum = np.minimum(lum, 1.2) * lum_scale
    temp = rng.random(n)
    col = np.stack([0.85 + 0.3 * temp, 0.92 + 0.06 * np.sin(temp * 3), 1.15 - 0.4 * temp], 1)
    col /= col.mean(1, keepdims=True)
    tw = rng.random(n) * 6.283
    return dict(dir=v, lum=lum, col=col.astype(np.float64), phase=tw)


@njit(fastmath=True)
def _splat(img, sx, sy, lum, col, sig, mask):
    H, W = img.shape[0], img.shape[1]
    n = sx.shape[0]
    for k in range(n):
        x = sx[k]
        y = sy[k]
        if x < -3 or y < -3 or x > W + 3 or y > H + 3:
            continue
        s = sig[k]
        rad = int(math.ceil(3.0 * s))
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        norm = lum[k] / (6.2832 * s * s)
        for jj in range(iy - rad, iy + rad + 1):
            if jj < 0 or jj >= H:
                continue
            for ii in range(ix - rad, ix + rad + 1):
                if ii < 0 or ii >= W:
                    continue
                dx = ii + 0.5 - x
                dy = jj + 0.5 - y
                w = math.exp(-(dx * dx + dy * dy) / (2 * s * s)) * norm * mask[jj, ii]
                img[jj, ii, 0] += w * col[k, 0]
                img[jj, ii, 1] += w * col[k, 1]
                img[jj, ii, 2] += w * col[k, 2]


def splat_stars(img, cam, stars, mask, t=0.0, gain=1.0, min_elev=-0.05, twinkle=0.25, scale=1.0,
                extinction=0.12):
    """Project star catalogue through cam and splat into img (H,W,3) where mask>0 (sky)."""
    d = stars['dir']
    x = d[:, 0] * cam.right[0] + d[:, 2] * cam.right[2]
    z = d[:, 0] * cam.fwd[0] + d[:, 2] * cam.fwd[2]
    y = d[:, 1]
    ok = (z > 0.05) & (y > min_elev)
    x, y, z = x[ok], y[ok], z[ok]
    sx = cam.cx + cam.f * x / z
    sy = cam.cy + cam.shift - cam.f * y / z
    lum = stars['lum'][ok] * gain
    lum = lum * (1.0 + twinkle * np.sin(t * 9.0 + stars['phase'][ok]) * np.minimum(lum / 0.05, 1.0))
    # atmospheric extinction toward the horizon
    lum = lum * np.clip(y / extinction, 0, 1) ** 1.5
    col = stars['col'][ok]
    sig = np.maximum(0.55 * scale, 0.45) * (1.0 + 0.8 * np.clip(lum / (0.3 * gain + 1e-9), 0, 1))
    _splat(img, sx.astype(np.float64), sy.astype(np.float64), lum.astype(np.float64), col, sig.astype(np.float64),
           mask.astype(np.float32))


# --------------------------------------------------------------- aurora ---

@njit(inline='always', fastmath=True)
def aurora_ray(ox, oy, oz, dx, dy, dz, A, t):
    """Ray-march a slab of auroral curtains.  A = [y0, y1, scale, seed, intensity, speed,
    offset_x, offset_z, nsteps, curtain_sharp].  Returns rgb."""
    y0 = A[0]
    y1 = A[1]
    if dy <= 0.01:
        return 0.0, 0.0, 0.0
    t0 = (y0 - oy) / dy
    t1 = (y1 - oy) / dy
    n = int(A[8])
    sc = A[2]
    seed = int(A[3])
    sharp = A[9]
    r = 0.0
    g = 0.0
    b = 0.0
    dt = (t1 - t0) / n
    # jitter start to hide banding
    jit = 0.5 * dt
    for k in range(n):
        tt = t0 + jit + dt * k
        px = ox + dx * tt + A[6]
        pz = oz + dz * tt + A[7]
        hy = (oy + dy * tt - y0) / (y1 - y0)   # 0 at the curtain foot, 1 at the top
        # curtain lines: warped 1D field in xz, thin where |sin| is small
        wx = px * sc
        wz = pz * sc
        warp = fbm2(wx * 0.35 + t * 0.02 * A[5], wz * 0.35, 3.0, seed) * 3.0
        field = wz * 1.2 + warp + 0.25 * math.sin(wx * 0.9 + t * 0.15 * A[5])
        c = abs(math.sin(field))
        curtain = math.exp(-c * sharp)
        # vertical rays (fine striations moving along the curtain)
        rays = 0.55 + 0.45 * gnoise2(wx * 9.0 + t * 0.35 * A[5], field * 2.0, seed + 5)
        rays = rays * rays
        # brightness along the curtain (patchy)
        patch = 0.5 + 0.5 * fbm2(wx * 0.6 - t * 0.05 * A[5], wz * 0.6, 3.0, seed + 9)
        # vertical profile: sharp lower edge, long fade upward
        prof = smoothstep(0.0, 0.06, hy) * math.exp(-hy * 2.6)
        dens = curtain * rays * patch * prof
        # colour: green low, violet/magenta high
        cg = math.exp(-hy * 3.2)
        cv = smoothstep(0.25, 0.9, hy)
        r += dens * (0.05 * cg + 0.55 * cv)
        g += dens * (1.0 * cg + 0.10 * cv)
        b += dens * (0.35 * cg + 0.85 * cv)
    s = A[4] * abs(dt) / (y1 - y0) * 4.0
    return r * s, g * s, b * s
