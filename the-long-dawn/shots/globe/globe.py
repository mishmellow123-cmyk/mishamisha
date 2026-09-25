"""GLOBE core renderer: Earth, atmosphere, clouds, night lights, stars, sun.

Units: Earth radius = 1. Earth-fixed frame: +z north pole, +x (lat 0, lon 0), +y (lat 0, lon 90E).
Everything is a pure function of its inputs (stateless per frame).
"""
import math
import os
import sys

import cv2
import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import look  # noqa: E402

CACHE = os.path.join(ROOT, 'renders', 'globe', 'cache')
R_KM = 6371.0


# ----------------------------------------------------------------- utils ---

def vec_ll(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def normalize(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def enu(lat, lon):
    """East, North, Up unit vectors at lat/lon."""
    up = vec_ll(lat, lon)
    east = normalize(np.cross([0.0, 0.0, 1.0], up))
    north = np.cross(up, east)
    return east, north, up


class Camera:
    """Pinhole camera. pos in Earth radii; basis rows = right, up, forward."""

    def __init__(self, pos, fwd, up_hint, hfov_deg, W, H):
        self.pos = np.asarray(pos, np.float64)
        f = normalize(fwd)
        r = normalize(np.cross(f, up_hint))
        u = np.cross(r, f)
        self.R = np.stack([r, u, f]).astype(np.float64)
        self.W, self.H = W, H
        self.f = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.cx, self.cy = W / 2.0, H / 2.0

    @staticmethod
    def orbit(lat, lon, alt_km, heading_deg, pitch_deg, hfov_deg, W, H, roll_deg=0.0):
        """Camera above (lat, lon) at altitude, looking at compass heading, pitched down by pitch_deg."""
        e, n, u = enu(lat, lon)
        pos = u * (1.0 + alt_km / R_KM)
        hd, pt = math.radians(heading_deg), math.radians(pitch_deg)
        horiz = math.sin(hd) * e + math.cos(hd) * n
        fwd = math.cos(pt) * horiz - math.sin(pt) * u
        right = normalize(np.cross(fwd, u))
        upv = np.cross(right, fwd)
        rl = math.radians(roll_deg)
        up_hint = math.cos(rl) * upv + math.sin(rl) * right
        return Camera(pos, fwd, up_hint, hfov_deg, W, H)

    def project(self, X):
        """World points (N,3) -> pixel coords (N,2), depth (N,)."""
        d = (X - self.pos) @ self.R.T
        z = d[:, 2]
        zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
        px = self.cx + self.f * d[:, 0] / zz
        py = self.cy - self.f * d[:, 1] / zz
        return np.stack([px, py], 1), z


# ------------------------------------------------------------ atmosphere ---

class Atmosphere:
    """Single-scattering Rayleigh/Mie/ozone shell, thickness exaggerated by X (optical depth kept)."""

    def __init__(self, X=2.5, mie_scale=3.0, g=0.80, airglow=0.0, soft_deg=0.45, rim=1.0,
                 airglow_h_km=120.0, airglow_w_km=2.2, airglow_warm=0.25):
        self.X = X
        k = 6.371e6 / X                                        # per metre -> per (R/X)
        self.params = np.zeros(28, np.float64)
        p = self.params
        p[0] = 1.0 + 100.0 * X / R_KM                            # top radius
        p[1] = 8.0 * X / R_KM                                    # Rayleigh scale height
        p[2] = 1.2 * X / R_KM                                    # Mie scale height
        p[3:6] = np.array([5.802e-6, 13.558e-6, 33.1e-6]) * k    # Rayleigh scattering (= extinction)
        p[6] = 3.996e-6 * k * mie_scale                          # Mie scattering
        p[7] = 4.440e-6 * k * mie_scale                          # Mie extinction
        p[8:11] = np.array([0.650e-6, 1.881e-6, 0.085e-6]) * k   # ozone absorption
        p[11] = 25.0 * X / R_KM                                  # ozone peak
        p[12] = 15.0 * X / R_KM                                  # ozone half width
        p[13] = g
        p[14] = math.radians(soft_deg)                           # soft terminator (sun disk + refraction)
        p[15] = airglow_h_km / R_KM                              # airglow: a hairline above the haze
        p[16] = airglow_w_km / R_KM
        p[17:20] = np.array([0.45, 1.0, 0.55]) * airglow         # pale green emission
        p[20] = rim                                              # extra gain on moonlit scattering (rim)
        p[21:24] = np.array([1.0, 0.45, 0.12]) * airglow * airglow_warm   # faint warm layer below
        p[24] = (airglow_h_km - 14.0) / R_KM
        self.lut = build_transmittance(p, 64, 256)

    @property
    def top(self):
        return self.params[0]


@njit(cache=True, fastmath=True)
def _extinction(h, p, out):
    rr = math.exp(-h / p[1])
    rm = math.exp(-h / p[2])
    ro = max(0.0, 1.0 - abs(h - p[11]) / p[12])
    for c in range(3):
        out[c] = p[3 + c] * rr + p[7] * rm + p[8 + c] * ro


@njit(cache=True)
def build_transmittance(p, NR, NMU):
    rt = p[0]
    Hh = math.sqrt(rt * rt - 1.0)
    lut = np.zeros((NR, NMU, 3), np.float32)
    ext = np.zeros(3)
    for i in range(NR):
        xr = i / (NR - 1)
        rho = Hh * xr
        r = math.sqrt(rho * rho + 1.0)
        for j in range(NMU):
            xm = j / (NMU - 1)
            dmin = rt - r
            dmax = rho + Hh
            d = dmin + xm * (dmax - dmin)
            if d <= 1e-12:
                mu = 1.0
            else:
                mu = (Hh * Hh - rho * rho - d * d) / (2.0 * r * d)
                mu = min(1.0, max(-1.0, mu))
            n = 400
            dt = d / n
            tau0 = 0.0
            tau1 = 0.0
            tau2 = 0.0
            for s in range(n):
                t = (s + 0.5) * dt
                rr = math.sqrt(r * r + t * t + 2.0 * r * mu * t)
                _extinction(rr - 1.0, p, ext)
                tau0 += ext[0] * dt
                tau1 += ext[1] * dt
                tau2 += ext[2] * dt
            lut[i, j, 0] = math.exp(-tau0)
            lut[i, j, 1] = math.exp(-tau1)
            lut[i, j, 2] = math.exp(-tau2)
    return lut


@njit(cache=True, fastmath=True, inline='always')
def trans_lookup(lut, p, r, mu):
    """Transmittance (rgb tuple) from radius r along a direction with cosine-zenith mu to the
    top of the atmosphere, times a soft Earth-shadow factor (0 when the ray hits the ground)."""
    rt = p[0]
    NR = lut.shape[0]
    NMU = lut.shape[1]
    if r > rt:
        r = rt
    if r < 1.0:
        r = 1.0
    Hh = math.sqrt(rt * rt - 1.0)
    rho = math.sqrt(max(r * r - 1.0, 0.0))
    muh = -rho / r
    soft = p[14]
    el = math.asin(max(-1.0, min(1.0, mu))) - math.asin(max(-1.0, min(1.0, muh)))
    vis = (el + soft) / (2.0 * soft)
    if vis <= 0.0:
        return 0.0, 0.0, 0.0
    if vis > 1.0:
        vis = 1.0
    vis = vis * vis * (3.0 - 2.0 * vis)
    mu2 = max(mu, muh + 1e-6)
    disc = r * r * (mu2 * mu2 - 1.0) + rt * rt
    d = -r * mu2 + math.sqrt(max(disc, 0.0))
    dmin = rt - r
    dmax = rho + Hh
    xm = (d - dmin) / max(dmax - dmin, 1e-12)
    xr = rho / Hh
    fi = min(max(xr, 0.0), 1.0) * (NR - 1)
    fj = min(max(xm, 0.0), 1.0) * (NMU - 1)
    i0 = min(int(fi), NR - 2)
    j0 = min(int(fj), NMU - 2)
    a = fi - i0
    b = fj - j0
    w00 = (1 - a) * (1 - b)
    w10 = a * (1 - b)
    w01 = (1 - a) * b
    w11 = a * b
    t0 = lut[i0, j0, 0] * w00 + lut[i0 + 1, j0, 0] * w10 + lut[i0, j0 + 1, 0] * w01 + lut[i0 + 1, j0 + 1, 0] * w11
    t1 = lut[i0, j0, 1] * w00 + lut[i0 + 1, j0, 1] * w10 + lut[i0, j0 + 1, 1] * w01 + lut[i0 + 1, j0 + 1, 1] * w11
    t2 = lut[i0, j0, 2] * w00 + lut[i0 + 1, j0, 2] * w10 + lut[i0, j0 + 1, 2] * w01 + lut[i0 + 1, j0 + 1, 2] * w11
    return t0 * vis, t1 * vis, t2 * vis


# ------------------------------------------------------------- textures ---

@njit(cache=True, fastmath=True, inline='always')
def ll_of(x, y, z):
    lon = math.atan2(y, x)
    lat = math.asin(max(-1.0, min(1.0, z)))
    return lat, lon


@njit(cache=True, fastmath=True, inline='always')
def tex_uv(lat, lon, W, H):
    u = (lon + math.pi) / (2.0 * math.pi) * W - 0.5
    v = (0.5 * math.pi - lat) / math.pi * H - 0.5
    return u, v


@njit(cache=True, fastmath=True, inline='always')
def _taps(u, v, W, H):
    x0 = int(math.floor(u))
    y0 = int(math.floor(v))
    a = u - x0
    b = v - y0
    x0 = x0 % W
    x1 = (x0 + 1) % W
    y1 = min(max(y0 + 1, 0), H - 1)
    y0 = min(max(y0, 0), H - 1)
    return x0, x1, y0, y1, a, b


@njit(cache=True, fastmath=True, inline='always')
def bilin1(tex, u, v):
    x0, x1, y0, y1, a, b = _taps(u, v, tex.shape[1], tex.shape[0])
    return ((tex[y0, x0] * (1 - a) + tex[y0, x1] * a) * (1 - b)
            + (tex[y1, x0] * (1 - a) + tex[y1, x1] * a) * b)


@njit(cache=True, fastmath=True, inline='always')
def bilin3(tex, u, v, lut):
    """uint8 sRGB texture -> linear rgb tuple via lut."""
    x0, x1, y0, y1, a, b = _taps(u, v, tex.shape[1], tex.shape[0])
    w00 = (1 - a) * (1 - b)
    w10 = a * (1 - b)
    w01 = (1 - a) * b
    w11 = a * b
    r = lut[tex[y0, x0, 0]] * w00 + lut[tex[y0, x1, 0]] * w10 + lut[tex[y1, x0, 0]] * w01 + lut[tex[y1, x1, 0]] * w11
    g = lut[tex[y0, x0, 1]] * w00 + lut[tex[y0, x1, 1]] * w10 + lut[tex[y1, x0, 1]] * w01 + lut[tex[y1, x1, 1]] * w11
    bb = lut[tex[y0, x0, 2]] * w00 + lut[tex[y0, x1, 2]] * w10 + lut[tex[y1, x0, 2]] * w01 + lut[tex[y1, x1, 2]] * w11
    return r, g, bb


@njit(cache=True, fastmath=True, inline='always')
def bilin3f(tex, u, v):
    x0, x1, y0, y1, a, b = _taps(u, v, tex.shape[1], tex.shape[0])
    w00 = (1 - a) * (1 - b)
    w10 = a * (1 - b)
    w01 = (1 - a) * b
    w11 = a * b
    r = tex[y0, x0, 0] * w00 + tex[y0, x1, 0] * w10 + tex[y1, x0, 0] * w01 + tex[y1, x1, 0] * w11
    g = tex[y0, x0, 1] * w00 + tex[y0, x1, 1] * w10 + tex[y1, x0, 1] * w01 + tex[y1, x1, 1] * w11
    bb = tex[y0, x0, 2] * w00 + tex[y0, x1, 2] * w10 + tex[y1, x0, 2] * w01 + tex[y1, x1, 2] * w11
    return r, g, bb


# ---------------------------------------------------------------- noise ---

@njit(cache=True, fastmath=True, inline='always')
def hash3(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 1274126177) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777215.0


@njit(cache=True, fastmath=True, inline='always')
def vnoise(x, y, z):
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    iz = int(math.floor(z))
    fx = x - ix
    fy = y - iy
    fz = z - iz
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10)
    a = hash3(ix, iy, iz)
    b = hash3(ix + 1, iy, iz)
    c = hash3(ix, iy + 1, iz)
    d = hash3(ix + 1, iy + 1, iz)
    e = hash3(ix, iy, iz + 1)
    f = hash3(ix + 1, iy, iz + 1)
    g = hash3(ix, iy + 1, iz + 1)
    h = hash3(ix + 1, iy + 1, iz + 1)
    k0 = a + (b - a) * ux
    k1 = c + (d - c) * ux
    k2 = e + (f - e) * ux
    k3 = g + (h - g) * ux
    l0 = k0 + (k1 - k0) * uy
    l1 = k2 + (k3 - k2) * uy
    return l0 + (l1 - l0) * uz


@njit(cache=True, fastmath=True)
def fbm(x, y, z, f0, octaves, lod):
    """fbm in [0,1]; octaves whose wavelength is below ~3x lod (Earth radii) fade out."""
    s = 0.0
    amp = 0.5
    norm = 0.0
    f = f0
    for o in range(octaves):
        wl = 1.0 / f
        w = min(1.0, max(0.0, (wl / max(lod, 1e-9) - 2.0) / 2.0))
        if w <= 0.0:
            break
        s += amp * w * vnoise(x * f + 17.3 * o, y * f - 9.1 * o, z * f + 3.7 * o)
        norm += amp * w
        amp *= 0.5
        f *= 2.03
    # missing octaves contribute their mean
    s += 0.5 * (1.0 - norm) if norm < 1.0 else 0.0
    return s if norm > 0 else 0.5


@njit(cache=True, fastmath=True)
def cloud_density(x, y, z, clouds, cp, lod):
    """Cloud opacity at a unit-sphere point: low-frequency coverage map + procedural detail."""
    lat, lon = ll_of(x, y, z)
    u, v = tex_uv(lat, lon, clouds.shape[1], clouds.shape[0])
    base = bilin1(clouds, u, v)
    if base * cp[1] + 0.5 * cp[2] - cp[3] <= 0.0:
        return 0.0
    wx = fbm(x, y, z, cp[4], 3, lod) - 0.5
    wy = fbm(y + 5.2, z, x, cp[4], 3, lod) - 0.5
    ws = cp[5]
    det = fbm(x + ws * wx, y + ws * wy, z - ws * wx, cp[0], 6, lod)
    c = base * cp[1] + (det - 0.5) * cp[2] - cp[3]
    c = min(1.0, max(0.0, c))
    return c * c * (3.0 - 2.0 * c) * cp[6]


# --------------------------------------------------------------- kernel ---

@njit(cache=True, fastmath=True, inline='always')
def ray_sphere(ox, oy, oz, dx, dy, dz, r):
    b = ox * dx + oy * dy + oz * dz
    c = ox * ox + oy * oy + oz * oz - r * r
    disc = b * b - c
    if disc < 0:
        return -1.0, -1.0
    s = math.sqrt(disc)
    return -b - s, -b + s


@njit(cache=True, fastmath=True, inline='always')
def phase_r(nu):
    return 3.0 / (16.0 * math.pi) * (1.0 + nu * nu)


@njit(cache=True, fastmath=True, inline='always')
def phase_m(nu, g):
    g2 = g * g
    return 3.0 / (8.0 * math.pi) * ((1 - g2) * (1 + nu * nu)) / ((2 + g2) * math.pow(max(1 + g2 - 2 * g * nu, 1e-6), 1.5))


@njit(cache=True, fastmath=True)
def airglow(ox, oy, oz, dx, dy, dz, t_max, p):
    """Emission of the thin airglow shells, integrated over the exact ray/shell crossings."""
    rc = 1.0 + 0.5 * (p[15] + p[24])
    half = 0.5 * (p[15] - p[24]) + 4.5 * p[16]
    r_out = rc + half
    r_in = rc - half
    to0, to1 = ray_sphere(ox, oy, oz, dx, dy, dz, r_out)
    if to1 <= 0.0:
        return 0.0, 0.0, 0.0
    ti0, ti1 = ray_sphere(ox, oy, oz, dx, dy, dz, r_in)
    g0 = 0.0
    g1 = 0.0
    g2 = 0.0
    for seg in range(2):
        if ti1 > 0.0 and ti0 > -1.0 and (ti0 != -1.0 or ti1 != -1.0):
            if seg == 0:
                a = max(to0, 0.0)
                b = ti0
            else:
                a = max(ti1, 0.0)
                b = to1
        else:
            if seg == 1:
                break
            a = max(to0, 0.0)
            b = to1
        b = min(b, t_max)
        if b <= a:
            continue
        n = 24
        dt = (b - a) / n
        for i in range(n):
            t = a + (i + 0.5) * dt
            x = ox + dx * t
            y = oy + dy * t
            z = oz + dz * t
            h = math.sqrt(x * x + y * y + z * z) - 1.0
            q = (h - p[15]) / p[16]
            q2 = (h - p[24]) / (2.5 * p[16])
            eg = math.exp(-q * q) * dt
            ew = math.exp(-q2 * q2) * dt
            g0 += eg * p[17] + ew * p[21]
            g1 += eg * p[18] + ew * p[22]
            g2 += eg * p[19] + ew * p[23]
    return g0, g1, g2


@njit(cache=True, fastmath=True)
def shade_ray(C, dx, dy, dz, pix_ang, S, E_sun, Sd, sun_rad, sun_ang, M, E_moon, p, lut,
              albedo, srgb_lut, mask, clouds, normal, cp, sp):
    """Radiance along one camera ray (excluding splatted lights/stars/web).
    Returns r, g, b, earth coverage (0/1), view transmittance (green)."""
    ox, oy, oz = C[0], C[1], C[2]
    rt = p[0]
    ta0, ta1 = ray_sphere(ox, oy, oz, dx, dy, dz, rt)
    te0, te1 = ray_sphere(ox, oy, oz, dx, dy, dz, 1.0)
    hit = te0 > 0.0
    L0 = 0.0
    L1 = 0.0
    L2 = 0.0
    tau0 = 0.0
    tau1 = 0.0
    tau2 = 0.0
    if ta1 > 0.0:
        t_start = max(ta0, 0.0)
        t_end = te0 if hit else ta1
        bb = ox * dx + oy * dy + oz * dz
        if hit:
            t_low = t_end
        else:
            t_low = min(max(-bb, t_start), t_end)
        nseg = int(sp[0]) if hit else int(sp[1])
        pw = 2.0 if hit else 1.5
        nuS = dx * S[0] + dy * S[1] + dz * S[2]
        nuM = dx * M[0] + dy * M[1] + dz * M[2]
        g = p[13]
        prS = phase_r(nuS)
        pmS = phase_m(nuS, g) * p[6]
        prM = phase_r(nuM)
        pmM = phase_m(nuM, g) * p[6]
        moon_on = E_moon[0] + E_moon[1] + E_moon[2] > 0.0
        # rim gain: full for rays that miss the ground, ramping in for grazing ground rays
        wr = 1.0
        if hit:
            hx_ = ox + dx * te0
            hy_ = oy + dy * te0
            hz_ = oz + dz * te0
            cv_ = -(hx_ * dx + hy_ * dy + hz_ * dz)
            wr = min(1.0, max(0.0, 1.0 - cv_ / 0.4))
            wr = wr * wr * wr
        rim = 1.0 + (p[20] - 1.0) * wr
        for half in range(2):
            if half == 0:
                span = t_low - t_start
            else:
                span = t_end - t_low
            if span <= 1e-12:
                continue
            n = nseg
            for i in range(n):
                u0 = i / n
                u1 = (i + 1.0) / n
                um = (i + 0.5) / n
                if half == 0:
                    # dense toward t_low (the end of this half)
                    s0 = t_start + span * (1.0 - math.pow(1.0 - u0, pw))
                    s1 = t_start + span * (1.0 - math.pow(1.0 - u1, pw))
                    t = t_start + span * (1.0 - math.pow(1.0 - um, pw))
                else:
                    s0 = t_low + span * math.pow(u0, pw)
                    s1 = t_low + span * math.pow(u1, pw)
                    t = t_low + span * math.pow(um, pw)
                ds = s1 - s0
                x = ox + dx * t
                y = oy + dy * t
                z = oz + dz * t
                r = math.sqrt(x * x + y * y + z * z)
                h = r - 1.0
                if h < 0.0:
                    h = 0.0
                rr = math.exp(-h / p[1])
                rm = math.exp(-h / p[2])
                ro = max(0.0, 1.0 - abs(h - p[11]) / p[12])
                e0 = p[3] * rr + p[7] * rm + p[8] * ro
                e1 = p[4] * rr + p[7] * rm + p[9] * ro
                e2 = p[5] * rr + p[7] * rm + p[10] * ro
                tv0 = math.exp(-(tau0 + 0.5 * e0 * ds))
                tv1 = math.exp(-(tau1 + 0.5 * e1 * ds))
                tv2 = math.exp(-(tau2 + 0.5 * e2 * ds))
                muS = (x * S[0] + y * S[1] + z * S[2]) / r
                T0, T1, T2 = trans_lookup(lut, p, r, muS)
                sr = rr * prS
                sm_ = rm * pmS
                L0 += tv0 * T0 * (p[3] * sr + sm_) * ds * E_sun[0]
                L1 += tv1 * T1 * (p[4] * sr + sm_) * ds * E_sun[1]
                L2 += tv2 * T2 * (p[5] * sr + sm_) * ds * E_sun[2]
                if moon_on:
                    muM = (x * M[0] + y * M[1] + z * M[2]) / r
                    Q0, Q1, Q2 = trans_lookup(lut, p, r, muM)
                    sr2 = rr * prM * rim
                    sm2 = rm * pmM * rim
                    L0 += tv0 * Q0 * (p[3] * sr2 + sm2) * ds * E_moon[0]
                    L1 += tv1 * Q1 * (p[4] * sr2 + sm2) * ds * E_moon[1]
                    L2 += tv2 * Q2 * (p[5] * sr2 + sm2) * ds * E_moon[2]
                tau0 += e0 * ds
                tau1 += e1 * ds
                tau2 += e2 * ds
    tr0 = math.exp(-tau0)
    tr1 = math.exp(-tau1)
    tr2 = math.exp(-tau2)
    if p[17] + p[18] + p[19] + p[21] > 0.0:
        a0_, a1_, a2_ = airglow(ox, oy, oz, dx, dy, dz, te0 if hit else 1e30, p)
        L0 += a0_
        L1 += a1_
        L2 += a2_
    if hit:
        x = ox + dx * te0
        y = oy + dy * te0
        z = oz + dz * te0
        rn = math.sqrt(x * x + y * y + z * z)
        x /= rn
        y /= rn
        z /= rn
        cosv = -(x * dx + y * dy + z * dz)
        lod = te0 * pix_ang / max(cosv, 0.08)
        g0, g1, g2 = shade_surface(x, y, z, dx, dy, dz, lod, S, E_sun, M, E_moon, p, lut, albedo,
                                   srgb_lut, mask, clouds, normal, cp)
        return L0 + tr0 * g0, L1 + tr1 * g1, L2 + tr2 * g2, 1.0, tr1
    # sun disk
    cs = dx * Sd[0] + dy * Sd[1] + dz * Sd[2]
    if cs > math.cos(sun_ang * 1.02):
        ang = math.acos(min(1.0, cs))
        rr_ = ang / sun_ang
        if rr_ < 1.0:
            ld = 1.0 - 0.6 * (1.0 - math.sqrt(max(0.0, 1.0 - rr_ * rr_)))
            L0 += sun_rad * ld * tr0
            L1 += sun_rad * ld * tr1
            L2 += sun_rad * ld * tr2
    return L0, L1, L2, 0.0, tr1


@njit(cache=True, fastmath=True)
def shade_surface(x, y, z, dx, dy, dz, lod, S, E_sun, M, E_moon, p, lut, albedo, srgb_lut,
                  mask, clouds, normal, cp):
    lat, lon = ll_of(x, y, z)
    u, v = tex_uv(lat, lon, albedo.shape[1], albedo.shape[0])
    a0, a1, a2 = bilin3(albedo, u, v, srgb_lut)
    land = bilin1(mask, u, v) / 255.0
    # relief normal (tangent space, +x east, +y north)
    ex = -y
    ey = x
    el = math.sqrt(ex * ex + ey * ey)
    if el < 1e-9:
        ex, ey, el = 1.0, 0.0, 1.0
    ex /= el
    ey /= el
    nx_ = -z * ey
    ny_ = z * ex
    nz_ = x * ey - y * ex
    un, vn = tex_uv(lat, lon, normal.shape[1], normal.shape[0])
    m0, m1, m2 = bilin3f(normal, un, vn)
    ks = cp[10] * land
    Nx = x + ks * (m0 * ex + m1 * nx_)
    Ny = y + ks * (m0 * ey + m1 * ny_)
    Nz = z + ks * (m1 * nz_)
    nl = math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
    Nx /= nl
    Ny /= nl
    Nz /= nl
    # procedural albedo detail on land
    if land > 0.01:
        dn = fbm(x, y, z, 700.0, 4, lod)
        k = 1.0 + (dn - 0.5) * 0.6 * land
        a0 *= k
        a1 *= k
        a2 *= k
    muS = x * S[0] + y * S[1] + z * S[2]
    muM = x * M[0] + y * M[1] + z * M[2]
    Tg0, Tg1, Tg2 = trans_lookup(lut, p, 1.0, muS)
    c = cloud_density(x, y, z, clouds, cp, lod)
    csh = 0.0
    if muS > -0.02:
        off = cp[7] / max(muS, 0.06)
        sx = x + S[0] * off
        sy = y + S[1] * off
        sz = z + S[2] * off
        sl = math.sqrt(sx * sx + sy * sy + sz * sz)
        csh = cloud_density(sx / sl, sy / sl, sz / sl, clouds, cp, lod * 2.0)
    shadow = 1.0 - cp[8] * csh
    ndl = max(0.0, Nx * S[0] + Ny * S[1] + Nz * S[2])
    ndl = ndl * min(1.0, max(0.0, (muS + 0.03) / 0.06))
    # twilight skylight (multiple-scattering stand-in): warm at the terminator, then blue
    el_s = math.asin(max(-1.0, min(1.0, muS)))
    tw = 0.0
    tr = 0.0
    tg = 0.0
    tb = 0.0
    if el_s > -0.14:
        q = math.exp(el_s / 0.035) if el_s < 0 else 1.0
        tw = q * cp[9]
        w = min(1.0, max(0.0, -el_s / 0.06))
        tr = (1.0 - w) * 1.0 + w * 0.28
        tg = (1.0 - w) * 0.62 + w * 0.36
        tb = (1.0 - w) * 0.50 + w * 0.80
    day_amb = cp[9] * 0.6 * min(1.0, max(0.0, muS * 4.0))
    inv_pi = 1.0 / math.pi
    dl = inv_pi * ndl * shadow
    g0 = a0 * E_sun[0] * (Tg0 * dl + tw * tr + day_amb * 0.55)
    g1 = a1 * E_sun[1] * (Tg1 * dl + tw * tg + day_amb * 0.70)
    g2 = a2 * E_sun[2] * (Tg2 * dl + tw * tb + day_amb * 1.00)
    # ocean glint (GGX)
    water = 1.0 - land
    if water > 0.01 and muS > -0.02:
        vx = -dx
        vy = -dy
        vz = -dz
        hx = vx + S[0]
        hy = vy + S[1]
        hz = vz + S[2]
        hl = math.sqrt(hx * hx + hy * hy + hz * hz)
        hx /= hl
        hy /= hl
        hz /= hl
        ndh = max(0.0, x * hx + y * hy + z * hz)
        ndv = max(1e-3, x * vx + y * vy + z * vz)
        ndl2 = max(0.0, muS)
        a = cp[11]
        a2_ = a * a
        dd = ndh * ndh * (a2_ - 1.0) + 1.0
        D = a2_ / (math.pi * dd * dd)
        vdh = max(0.0, vx * hx + vy * hy + vz * hz)
        F = 0.02 + 0.98 * math.pow(1.0 - vdh, 5.0)
        k_ = a * 0.5
        G = (ndl2 / (ndl2 * (1 - k_) + k_)) * (ndv / (ndv * (1 - k_) + k_))
        spec = D * F * G / (4.0 * ndv + 1e-4) * water * shadow
        g0 += spec * E_sun[0] * Tg0
        g1 += spec * E_sun[1] * Tg1
        g2 += spec * E_sun[2] * Tg2
    # moonlight
    Tm0 = 0.0
    Tm1 = 0.0
    Tm2 = 0.0
    if muM > -0.1:
        Tm0, Tm1, Tm2 = trans_lookup(lut, p, 1.0, muM)
        ndm = max(0.0, Nx * M[0] + Ny * M[1] + Nz * M[2]) * inv_pi * cp[12]
        g0 += a0 * E_moon[0] * Tm0 * ndm
        g1 += a1 * E_moon[1] * Tm1 * ndm
        g2 += a2 * E_moon[2] * Tm2 * ndm
        if water > 0.01 and cp[14] > 0.0:
            vx = -dx
            vy = -dy
            vz = -dz
            hx = vx + M[0]
            hy = vy + M[1]
            hz = vz + M[2]
            hl = math.sqrt(hx * hx + hy * hy + hz * hz)
            ndh = max(0.0, (x * hx + y * hy + z * hz) / hl)
            ndv = max(1e-3, x * vx + y * vy + z * vz)
            a = 0.35
            a2_ = a * a
            dd = ndh * ndh * (a2_ - 1.0) + 1.0
            D = a2_ / (math.pi * dd * dd)
            vdh = max(0.0, (vx * hx + vy * hy + vz * hz) / hl)
            F = 0.02 + 0.98 * math.pow(1.0 - vdh, 5.0)
            spec = D * F * max(0.0, muM) / (4.0 * ndv + 1e-3) * water * cp[14]
            g0 += spec * E_moon[0] * Tm0
            g1 += spec * E_moon[1] * Tm1
            g2 += spec * E_moon[2] * Tm2
    # cloud layer, lit at cloud-top height (keeps the light past the ground terminator)
    if c > 0.0:
        rc = 1.0 + cp[7]
        Tc0, Tc1, Tc2 = trans_lookup(lut, p, rc, muS)
        wrap = max(0.0, (muS + 0.18) / 1.18)
        vs = dx * S[0] + dy * S[1] + dz * S[2]
        fwd = 1.0 + 0.8 * max(0.0, vs) ** 8
        calb = 0.85
        dl = inv_pi * wrap * fwd
        mm = inv_pi * max(0.0, (muM + 0.1) / 1.1) * cp[13]
        c0 = calb * (E_sun[0] * (Tc0 * dl + tw * tr * 1.3 + day_amb * 0.55) + E_moon[0] * Tm0 * mm)
        c1 = calb * (E_sun[1] * (Tc1 * dl + tw * tg * 1.3 + day_amb * 0.70) + E_moon[1] * Tm1 * mm)
        c2 = calb * (E_sun[2] * (Tc2 * dl + tw * tb * 1.3 + day_amb * 1.00) + E_moon[2] * Tm2 * mm)
        g0 = g0 * (1.0 - c) + c0 * c
        g1 = g1 * (1.0 - c) + c1 * c
        g2 = g2 * (1.0 - c) + c2 * c
    return g0, g1, g2


@njit(parallel=True, cache=True, fastmath=True)
def render_kernel(img, cov, tview, C, Rm, f, cx, cy, S, E_sun, Sd, sun_rad, sun_ang, M, E_moon,
                  p, lut, albedo, srgb_lut, mask, clouds, normal, cp, sp):
    H = img.shape[0]
    W = img.shape[1]
    pix_ang = 1.0 / f
    c0 = C[0] * C[0] + C[1] * C[1] + C[2] * C[2]
    for py in prange(H):
        for px in range(W):
            vx = (px + 0.5 - cx) / f
            vy = -(py + 0.5 - cy) / f
            dx = Rm[0, 0] * vx + Rm[1, 0] * vy + Rm[2, 0]
            dy = Rm[0, 1] * vx + Rm[1, 1] * vy + Rm[2, 1]
            dz = Rm[0, 2] * vx + Rm[1, 2] * vy + Rm[2, 2]
            dl = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= dl
            dy /= dl
            dz /= dl
            b = C[0] * dx + C[1] * dy + C[2] * dz
            bd = math.sqrt(max(0.0, c0 - b * b))
            edge = (-b > 0) and abs(bd - 1.0) < 1.5 * (-b) * pix_ang
            cs = dx * Sd[0] + dy * Sd[1] + dz * Sd[2]
            near_sun = cs > math.cos(sun_ang + 2.5 * pix_ang)
            n = 4 if (edge or near_sun) else 1
            r0 = 0.0
            r1 = 0.0
            r2 = 0.0
            cv = 0.0
            tv = 0.0
            for sy in range(n):
                for sx in range(n):
                    ex_ = (sx + 0.5) / n
                    ey_ = (sy + 0.5) / n
                    vx = (px + ex_ - cx) / f
                    vy = -(py + ey_ - cy) / f
                    dx = Rm[0, 0] * vx + Rm[1, 0] * vy + Rm[2, 0]
                    dy = Rm[0, 1] * vx + Rm[1, 1] * vy + Rm[2, 1]
                    dz = Rm[0, 2] * vx + Rm[1, 2] * vy + Rm[2, 2]
                    dl = math.sqrt(dx * dx + dy * dy + dz * dz)
                    o0, o1, o2, o3, o4 = shade_ray(C, dx / dl, dy / dl, dz / dl, pix_ang, S, E_sun, Sd,
                                                   sun_rad, sun_ang, M, E_moon, p, lut, albedo, srgb_lut,
                                                   mask, clouds, normal, cp, sp)
                    r0 += o0
                    r1 += o1
                    r2 += o2
                    cv += o3
                    tv += o4
            inv = 1.0 / (n * n)
            img[py, px, 0] = r0 * inv
            img[py, px, 1] = r1 * inv
            img[py, px, 2] = r2 * inv
            cov[py, px] = cv * inv
            tview[py, px] = tv * inv


# ------------------------------------------------------------ splatting ---

@njit(cache=True, fastmath=True)
def splat_points(img, px, py, rgb, sigma):
    """Additive gaussian splats (sigma in px, per point). rgb = total energy per point."""
    H = img.shape[0]
    W = img.shape[1]
    for i in range(px.shape[0]):
        x = px[i]
        y = py[i]
        s = sigma[i]
        rad = int(math.ceil(3.0 * s))
        if x < -rad or y < -rad or x > W + rad or y > H + rad:
            continue
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        inv = 1.0 / (2.0 * s * s)
        # normalise discretely
        tot = 0.0
        for yy in range(iy - rad, iy + rad + 1):
            for xx in range(ix - rad, ix + rad + 1):
                ddx = xx + 0.5 - x
                ddy = yy + 0.5 - y
                tot += math.exp(-(ddx * ddx + ddy * ddy) * inv)
        if tot <= 0:
            continue
        k = 1.0 / tot
        for yy in range(max(iy - rad, 0), min(iy + rad + 1, H)):
            for xx in range(max(ix - rad, 0), min(ix + rad + 1, W)):
                ddx = xx + 0.5 - x
                ddy = yy + 0.5 - y
                w = math.exp(-(ddx * ddx + ddy * ddy) * inv) * k
                img[yy, xx, 0] += rgb[i, 0] * w
                img[yy, xx, 1] += rgb[i, 1] * w
                img[yy, xx, 2] += rgb[i, 2] * w


@njit(cache=True, fastmath=True)
def splat_segments(img, x0, y0, x1, y1, c0, c1, w0, w1):
    """Anti-aliased additive line segments with gaussian cross-section.
    c0/c1: rgb radiance at the ends (peak value on the line), w0/w1: sigma in px."""
    H = img.shape[0]
    W = img.shape[1]
    for i in range(x0.shape[0]):
        ax = x0[i]
        ay = y0[i]
        bx = x1[i]
        by = y1[i]
        sw = max(w0[i], w1[i])
        rad = 3.0 * sw + 1.0
        minx = int(math.floor(min(ax, bx) - rad))
        maxx = int(math.ceil(max(ax, bx) + rad))
        miny = int(math.floor(min(ay, by) - rad))
        maxy = int(math.ceil(max(ay, by) + rad))
        if maxx < 0 or maxy < 0 or minx >= W or miny >= H:
            continue
        if maxx - minx > 4000 or maxy - miny > 4000:
            continue
        ex = bx - ax
        ey = by - ay
        l2 = ex * ex + ey * ey
        for yy in range(max(miny, 0), min(maxy + 1, H)):
            for xx in range(max(minx, 0), min(maxx + 1, W)):
                qx = xx + 0.5 - ax
                qy = yy + 0.5 - ay
                if l2 > 1e-9:
                    t = (qx * ex + qy * ey) / l2
                else:
                    t = 0.0
                t = min(1.0, max(0.0, t))
                dx = qx - t * ex
                dy = qy - t * ey
                s = w0[i] + (w1[i] - w0[i]) * t
                d2 = (dx * dx + dy * dy) / (2.0 * s * s)
                if d2 > 9.0:
                    continue
                w = math.exp(-d2)
                # avoid double-counting at shared joints: weight the ends by coverage along the axis
                img[yy, xx, 0] += (c0[i, 0] + (c1[i, 0] - c0[i, 0]) * t) * w
                img[yy, xx, 1] += (c0[i, 1] + (c1[i, 1] - c0[i, 1]) * t) * w
                img[yy, xx, 2] += (c0[i, 2] + (c1[i, 2] - c0[i, 2]) * t) * w


@njit(cache=True, fastmath=True)
def splat_polyline(img, xs, ys, cols, sig, starts, ok):
    """Draw polylines with per-vertex colour (peak radiance) and gaussian sigma (px).
    A segment is drawn only if both its vertices have ok != 0. Joints are owned half-open so
    they are not double counted."""
    H = img.shape[0]
    W = img.shape[1]
    for k in range(starts.shape[0] - 1):
        a = starts[k]
        b = starts[k + 1]
        for i in range(a, b - 1):
            if ok[i] == 0 or ok[i + 1] == 0:
                continue
            ax = xs[i]
            ay = ys[i]
            bx = xs[i + 1]
            by = ys[i + 1]
            s0 = sig[i]
            s1 = sig[i + 1]
            sw = max(s0, s1)
            rad = 3.0 * sw + 1.0
            if abs(ax) > 1e5 or abs(bx) > 1e5 or abs(ay) > 1e5 or abs(by) > 1e5:
                continue
            minx = int(math.floor(min(ax, bx) - rad))
            maxx = int(math.ceil(max(ax, bx) + rad))
            miny = int(math.floor(min(ay, by) - rad))
            maxy = int(math.ceil(max(ay, by) + rad))
            if maxx < 0 or maxy < 0 or minx >= W or miny >= H:
                continue
            if maxx - minx > 3000 or maxy - miny > 3000:
                continue
            ex = bx - ax
            ey = by - ay
            l2 = ex * ex + ey * ey
            first = (i == a) or ok[i - 1] == 0
            last = (i == b - 2) or ok[i + 2] == 0
            for yy in range(max(miny, 0), min(maxy + 1, H)):
                for xx in range(max(minx, 0), min(maxx + 1, W)):
                    qx = xx + 0.5 - ax
                    qy = yy + 0.5 - ay
                    if l2 > 1e-9:
                        t = (qx * ex + qy * ey) / l2
                    else:
                        t = 0.0
                    if t < 0.0 and not first:
                        continue
                    if t >= 1.0 and not last:
                        continue
                    tc = min(1.0, max(0.0, t))
                    dx = qx - tc * ex
                    dy = qy - tc * ey
                    s = s0 + (s1 - s0) * tc
                    d2 = (dx * dx + dy * dy) / (2.0 * s * s)
                    if d2 > 9.0:
                        continue
                    w = math.exp(-d2)
                    img[yy, xx, 0] += (cols[i, 0] + (cols[i + 1, 0] - cols[i, 0]) * tc) * w
                    img[yy, xx, 1] += (cols[i, 1] + (cols[i + 1, 1] - cols[i, 1]) * tc) * w
                    img[yy, xx, 2] += (cols[i, 2] + (cols[i + 1, 2] - cols[i, 2]) * tc) * w


# ------------------------------------------------------------ occlusion ---

def visible(cam_pos, X, eps=2e-4):
    """True where points X (N,3) are not hidden behind the Earth (radius 1) from cam_pos."""
    D = X - cam_pos[None, :]
    dist = np.linalg.norm(D, axis=1)
    d = D / np.maximum(dist, 1e-12)[:, None]
    b = d @ cam_pos
    c = cam_pos @ cam_pos - 1.0
    disc = b * b - c
    t0 = -b - np.sqrt(np.maximum(disc, 0.0))
    hidden = (disc > 0) & (t0 > 0) & (t0 < dist - eps)
    return ~hidden


# ------------------------------------------------------------ resources ---

class World:
    """Loads the cached maps once."""

    def __init__(self):
        a = cv2.imread(os.path.join(CACHE, 'albedo_8k.png'))
        self.albedo = np.ascontiguousarray(a[..., ::-1])
        self.mask = cv2.imread(os.path.join(CACHE, 'land_mask_8k.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m = np.load(os.path.join(CACHE, 'maps.npz'))
        self.clouds = np.ascontiguousarray(m['clouds'].astype(np.float32))
        self.normal = np.ascontiguousarray(m['normal'].astype(np.float32))
        self.srgb_lut = look.srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0).astype(np.float64)
        lp = np.load(os.path.join(CACHE, 'lights_points.npz'))
        self.lP = lp['P'].astype(np.float64)
        self.le = lp['e'].astype(np.float64)
        self.ltint = lp['tint'].astype(np.float64)
        self._stars = None

    # cloud params: [f_detail, base_gain, detail_gain, bias, warp_f, warp_strength, opacity,
    #                cloud_height, shadow_strength, twilight_gain, relief, ocean_roughness]
    #                night_land, night_cloud, moon_glint]
    @staticmethod
    def default_cp(X=2.5):
        return np.array([95.0, 1.35, 0.95, 0.12, 22.0, 0.035, 0.95,
                         9.0 * X / R_KM, 0.55, 0.035, 0.9, 0.16,
                         0.30, 0.55, 1.0], np.float64)

    def stars(self):
        if self._stars is None:
            rng = np.random.default_rng(11)
            n = 14000
            v = rng.normal(size=(n, 3))
            v /= np.linalg.norm(v, axis=1)[:, None]
            # a faint galactic band: extra stars concentrated near a great circle
            gpole = normalize([0.25, -0.55, 0.8])
            m = 9000
            w = rng.normal(size=(m, 3))
            w -= (w @ gpole)[:, None] * gpole[None, :]
            w /= np.linalg.norm(w, axis=1)[:, None]
            w += gpole[None, :] * rng.normal(0, 0.08, m)[:, None]
            w /= np.linalg.norm(w, axis=1)[:, None]
            v = np.concatenate([v, w])
            mag = np.concatenate([rng.power(3.2, n) * 7.0, 4.5 + rng.power(2.0, m) * 3.0])
            mag = 7.0 - mag   # small = bright
            flux = 10 ** (-0.4 * mag)
            temp = rng.random(len(v))
            col = np.stack([0.85 + 0.3 * temp, 0.9 + 0.05 * temp, 1.15 - 0.4 * temp], 1)
            self._stars = (v, flux, col)
        return self._stars


def render_planet(world, cam, atmo, S, E_sun, Sd, sun_rad, M, E_moon, cp=None, sp=(10, 36),
                  sun_ang_deg=0.2665):
    H, W = cam.H, cam.W
    img = np.zeros((H, W, 3), np.float32)
    cov = np.zeros((H, W), np.float32)
    tv = np.zeros((H, W), np.float32)
    if cp is None:
        cp = World.default_cp(atmo.X)
    render_kernel(img, cov, tv, cam.pos, cam.R, cam.f, cam.cx, cam.cy,
                  np.asarray(S, np.float64), np.asarray(E_sun, np.float64),
                  np.asarray(Sd, np.float64), float(sun_rad), math.radians(sun_ang_deg),
                  np.asarray(M, np.float64), np.asarray(E_moon, np.float64),
                  atmo.params, atmo.lut, world.albedo, world.srgb_lut, world.mask, world.clouds,
                  world.normal, cp, np.asarray(sp, np.float64))
    return img, cov, tv


# ------------------------------------------------------- lights & stars ---

@njit(parallel=True, cache=True, fastmath=True)
def _lights_eval(P, e, tint, C, S, p, lut, clouds, cp, gain, out_rgb, out_ok, lod_ang):
    n = P.shape[0]
    for i in prange(n):
        x = P[i, 0]
        y = P[i, 1]
        z = P[i, 2]
        vx = C[0] - x
        vy = C[1] - y
        vz = C[2] - z
        dist = math.sqrt(vx * vx + vy * vy + vz * vz)
        cosv = (x * vx + y * vy + z * vz) / dist
        if cosv <= 0.02:
            out_ok[i] = 0
            continue
        muS = x * S[0] + y * S[1] + z * S[2]
        # lights switch on through dusk (sun elevation +2deg .. -8deg)
        el = math.asin(max(-1.0, min(1.0, muS)))
        night = min(1.0, max(0.0, (0.035 - el) / 0.17))
        night = night * night * (3 - 2 * night)
        if night <= 0.0:
            out_ok[i] = 0
            continue
        T0, T1, T2 = trans_lookup(lut, p, 1.0, cosv)
        lod = dist * lod_ang / max(cosv, 0.1)
        c = cloud_density(x, y, z, clouds, cp, lod)
        k = e[i] * gain * night * (1.0 - 0.8 * c) * cosv / (dist * dist)
        tt = tint[i]
        out_rgb[i, 0] = k * T0
        out_rgb[i, 1] = k * T1 * (0.62 + 0.2 * tt)
        out_rgb[i, 2] = k * T2 * (0.30 + 0.35 * tt)
        out_ok[i] = 1


def render_lights(world, cam, atmo, S, gain, cp=None, sigma=0.65):
    if cp is None:
        cp = World.default_cp(atmo.X)
    n = len(world.le)
    rgb = np.zeros((n, 3), np.float64)
    ok = np.zeros(n, np.uint8)
    _lights_eval(world.lP, world.le, world.ltint, cam.pos, np.asarray(S, np.float64), atmo.params,
                 atmo.lut, world.clouds, cp, gain * cam.f * cam.f, rgb, ok, 1.0 / cam.f)
    sel = ok.astype(bool)
    uv, z = cam.project(world.lP[sel])
    m = (z > 0) & (uv[:, 0] > -3) & (uv[:, 0] < cam.W + 3) & (uv[:, 1] > -3) & (uv[:, 1] < cam.H + 3)
    img = np.zeros((cam.H, cam.W, 3), np.float32)
    uv = uv[m]
    col = rgb[sel][m]
    splat_points(img, uv[:, 0].copy(), uv[:, 1].copy(), col, np.full(len(uv), sigma))
    return img


def render_stars(world, cam, atmo, gain, sigma=0.55):
    v, flux, col = world.stars()
    d = v @ cam.R.T
    m = d[:, 2] > 0.05
    v, flux, col, d = v[m], flux[m], col[m], d[m]
    px = cam.cx + cam.f * d[:, 0] / d[:, 2]
    py = cam.cy - cam.f * d[:, 1] / d[:, 2]
    m = (px > -2) & (px < cam.W + 2) & (py > -2) & (py < cam.H + 2)
    v, flux, col, px, py = v[m], flux[m], col[m], px[m], py[m]
    # hidden by the Earth / dimmed through the atmosphere
    C = cam.pos
    b = v @ C
    c = C @ C
    disc_e = b * b - (c - 1.0)
    hit = (disc_e > 0) & (-b - np.sqrt(np.maximum(disc_e, 0)) > 0)
    bd = np.sqrt(np.maximum(c - b * b, 0.0))
    h = np.maximum(bd - 1.0, 0.0)
    tau = 0.25 * np.exp(-h / atmo.params[1]) * np.sqrt(2 * math.pi * 1.0 / atmo.params[1]) * atmo.params[1] * 40
    T = np.exp(-np.minimum(tau, 50))
    k = (~hit) * T * gain
    rgb = col * (flux * k)[:, None]
    img = np.zeros((cam.H, cam.W, 3), np.float32)
    splat_points(img, px.copy(), py.copy(), rgb.astype(np.float64), np.full(len(px), sigma))
    return img


# ------------------------------------------------------------------ post ---

def sun_glare(W, H, sx, sy, core, spikes_amt, flash=0.0, rot_deg=11.0, tint=(1.0, 0.88, 0.70),
              n_spikes=16, spike_len=1.0, ghosts=0.0, spike_mask_y=None):
    """Lens response to the sun (HDR, additive), sized for a 1920-wide frame and scaled to W.
    core: brightness of the halo; spikes_amt: diffraction spike brightness (0 while the disk is
    hidden); flash: extra wide bloom for the burst (decays over a few frames)."""
    img = np.zeros((H, W, 3), np.float32)
    k = W / 1920.0
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (x + 0.5 - sx) / k
    dy = (y + 0.5 - sy) / k
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    tint = np.asarray(tint, np.float32)
    hot = np.asarray((1.0, 0.96, 0.9), np.float32)
    if core > 0:
        g_hot = core * (1.4 * np.exp(-r / 6.0) + 0.55 * np.exp(-r / 22.0))
        g_warm = core * (0.16 / (1.0 + (r / 60.0) ** 2) + 0.035 / (1.0 + (r / 220.0) ** 2) ** 1.5)
        img += g_hot[..., None] * hot + g_warm[..., None] * tint
    if flash > 0:
        g = flash * (2.4 * np.exp(-r / 26.0) + 0.5 / (1.0 + (r / 90.0) ** 2) + 0.03 / (1.0 + (r / 300.0) ** 2))
        img += g[..., None] * np.asarray((1.0, 0.93, 0.82), np.float32)
    if spikes_amt > 0:
        th = np.arctan2(dy, dx)
        acc = np.zeros((H, W), np.float32)
        rng = np.random.default_rng(3)
        for i in range(n_spikes):
            a = math.radians(rot_deg) + i * 2 * math.pi / n_spikes
            main = i % 2 == 0
            L = (300.0 if main else 120.0) * spike_len * rng.uniform(0.8, 1.2)
            amp = (1.0 if main else 0.45) * rng.uniform(0.75, 1.1)
            dth = np.angle(np.exp(1j * (th - a)))
            wpx = 0.9 + r * 0.004                       # spike half-width in px, widening slowly
            lobe = np.exp(-0.5 * (dth * r / wpx) ** 2)
            fall = np.exp(-r / L) / (1.0 + r / 25.0)
            acc += amp * lobe * fall
        if spike_mask_y is not None:
            y0, y1, amt = spike_mask_y
            yy = y / k
            m = np.clip((yy - y0) / max(y1 - y0, 1.0), 0, 1)
            acc *= 1.0 - amt * m * m * (3 - 2 * m)
        sp = (acc * spikes_amt)[..., None]
        fr = np.clip(r / 260.0, 0, 1)[..., None]
        chrom = np.concatenate([1.0 + 0.15 * fr, 1.0 - 0.02 * fr, 1.0 - 0.35 * fr], -1)
        img += sp * tint * chrom
    if ghosts > 0:
        cx, cy = W / 2.0, H / 2.0
        for (t_, rad, col) in [(-0.35, 14.0, (0.55, 0.75, 1.0)), (-0.7, 30.0, (1.0, 0.75, 0.45)),
                               (-1.1, 55.0, (0.6, 1.0, 0.75)), (0.3, 9.0, (1.0, 0.85, 0.6))]:
            gx = cx + (sx - cx) * t_
            gy = cy + (sy - cy) * t_
            rr = np.sqrt((x - gx) ** 2 + (y - gy) ** 2) / k
            d = np.clip((rad - rr) / 3.0, 0, 1) * (0.35 + 0.65 * np.clip(rr / rad, 0, 1) ** 3)
            img += (d * ghosts)[..., None] * np.asarray(col, np.float32)
    return img


def finish_frame(hdr, sun_layer=None, exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8,
                 streak_strength=0.0, streak_threshold=2.0, streak_length=0.35, streak_tint=(1.0, 0.75, 0.55),
                 vignette_amount=0.2, lift=0.004):
    """look.finish() with the same functions in the same order, except that the sun's lens layer
    (already a glare) is added after the bloom so it is not bloomed twice, and the anamorphic
    streak is driven by the sun layer only (so the beacons don't each smear sideways)."""
    x = hdr.astype(np.float32)
    if bloom_strength > 0:
        x = look.bloom(x, bloom_strength, bloom_threshold)
    if sun_layer is not None:
        x = x + sun_layer
        if streak_strength > 0:
            s = look.streak(sun_layer, streak_strength, streak_threshold, streak_length, streak_tint) - sun_layer
            x = x + s
    if vignette_amount > 0:
        x = look.vignette(x, vignette_amount)
    x = look.tonemap(x, exposure)
    x = x + lift * (1 - x)
    return look.linear_to_srgb(x)
