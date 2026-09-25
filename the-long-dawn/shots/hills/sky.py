"""Sky: twilight gradient, Milky Way, stars, Moon (earthshine + city lights), orbital ring."""
import math

import numba as nb
import numpy as np

from core import (C, cam_ray, fbm2, fbm3, perlin3, splat_gauss, splat_streak, hash11)

DEG = math.pi / 180.0


def dir_from_az_el(daz_deg, el_deg):
    """World direction; daz relative to the +z view axis (positive = right)."""
    a, e = daz_deg * DEG, el_deg * DEG
    return np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])


def az_el_from_dir(d):
    d = np.asarray(d, np.float64)
    return math.degrees(math.atan2(d[0], d[2])), math.degrees(math.asin(np.clip(d[1], -1, 1)))


# ------------------------------------------------------------ parameters ---

class Sky:
    """Container for all sky settings of a scene. Everything in world directions."""

    def __init__(self, **kw):
        self.sun_az, self.sun_el = kw.get('sun', (-14.0, -6.5))
        self.zenith = kw.get('zenith', C('night_zenith', 1.0))
        self.horizon = kw.get('horizon', C('night_horizon', 1.0))
        self.amber = kw.get('amber', C('dusk_amber', 1.0))
        self.rose = kw.get('rose', C('dusk_rose', 1.0))
        self.violet = kw.get('violet', C('dusk_violet', 1.0))
        self.base_fall = kw.get('base_fall', 0.22)
        self.amber_fall = kw.get('amber_fall', 0.035)
        self.rose_fall = kw.get('rose_fall', 0.10)
        self.violet_fall = kw.get('violet_fall', 0.30)
        self.az_pow = kw.get('az_pow', 2.0)
        self.gain = kw.get('gain', 1.0)
        # moon
        self.moon = kw.get('moon', None)       # (daz, el) or None
        self.moon_radius_deg = kw.get('moon_radius_deg', 0.8)
        self.moon_gain = kw.get('moon_gain', 1.0)
        self.moon_aureole = kw.get('moon_aureole', 0.02)
        self.earthshine = kw.get('earthshine', 0.02)
        self.city_gain = kw.get('city_gain', 1.0)
        self.moon_phase = kw.get('moon_phase', 140.0)   # phase angle (deg): 180 = new, 0 = full
        self.moon_pa = kw.get('moon_pa', -35.0)         # screen angle of bright limb (deg, 0=right, 90=up)
        # milky way
        self.mw = kw.get('mw', 0.0)
        self.mw_pole = kw.get('mw_pole', dir_from_az_el(-60.0, 30.0))
        self.mw_center = kw.get('mw_center', None)
        # stars
        self.star_gain = kw.get('star_gain', 1.0)
        self.star_thresh = kw.get('star_thresh', 1.0)
        self.n_stars = kw.get('n_stars', 14000)
        # ring
        self.ring = kw.get('ring', None)       # dict or None
        self.seed = kw.get('seed', 1)

        self.sun_dir = dir_from_az_el(self.sun_az, self.sun_el)
        self.moon_dir = dir_from_az_el(*self.moon) if self.moon is not None else None
        if self.mw_center is None:
            p = self.mw_pole
            t = np.cross(p, [0, 1, 0])
            self.mw_center = t / np.linalg.norm(t)
        self._stars = None
        self._ring = None
        self._cities = None

    def packed(self):
        m = self.moon_dir if self.moon_dir is not None else np.array([0, -1.0, 0])
        g = self.mw_pole / np.linalg.norm(self.mw_pole)
        c = self.mw_center - np.dot(self.mw_center, g) * g
        c /= np.linalg.norm(c)
        return np.concatenate([
            self.sun_dir, self.zenith, self.horizon, self.amber, self.rose, self.violet,
            [self.base_fall, self.amber_fall, self.rose_fall, self.violet_fall, self.az_pow],
            m, [self.moon_aureole if self.moon_dir is not None else 0.0, 0.0],
            C('moonlight'),
            [self.mw], g, c, [self.gain],
        ]).astype(np.float64)

    # ---- star catalogue
    def stars(self):
        if self._stars is None:
            rng = np.random.default_rng(1000 + self.seed)
            n = self.n_stars
            v = rng.normal(size=(n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            # magnitudes: N(<m) ~ 10^(0.45 m), m in [-1.2, 7]
            u = rng.random(n)
            mag = 7.0 + np.log10(np.maximum(u, 1e-9)) / 0.45
            mag = np.maximum(mag, -1.3)
            flux = 10 ** (-0.4 * mag)
            # concentrate a share of faint stars toward the milky way plane
            if self.mw > 0:
                g = self.mw_pole / np.linalg.norm(self.mw_pole)
                k = int(n * 0.35)
                w = rng.normal(size=(k, 3))
                w -= (w @ g)[:, None] * g[None, :] * (1 - rng.normal(0, 0.12, k)[:, None] ** 2)
                w += g[None, :] * rng.normal(0, 0.10, (k, 1))
                w /= np.linalg.norm(w, axis=1, keepdims=True)
                v[:k] = w
                flux[:k] *= 0.5
            temp = rng.normal(0.55, 0.22, n).clip(0, 1)   # 0 = warm (K/M), 1 = blue (B/A)
            col = np.stack([1.0 - 0.35 * temp, 0.86 + 0.06 * temp - 0.05 * (1 - temp), 0.62 + 0.45 * temp], 1)
            col /= col.mean(axis=1, keepdims=True)
            phase = rng.random(n) * 1000
            self._stars = (v.astype(np.float64), flux.astype(np.float64), col.astype(np.float64),
                           phase.astype(np.float64))
        return self._stars

    # ---- moon city lights (unit vectors in a moon-fixed frame)
    def cities(self):
        if self._cities is None:
            rng = np.random.default_rng(77 + self.seed)
            pts = []
            # clusters along "coasts" of maria + scattered settlements
            # two bright clusters + a few strings of settlements + scattered pinpricks,
            # all on the part of the disc that stays in earthshine (away from the lit limb)
            centers = [(-0.35, 0.30, 11, 0.06), (-0.05, -0.30, 8, 0.05), (-0.55, -0.15, 5, 0.04),
                       (0.10, 0.45, 5, 0.035), (-0.30, -0.55, 4, 0.03)]
            for cx_, cy_, k, spread in centers:
                for j in range(k):
                    x = cx_ + rng.normal(0, spread)
                    y = cy_ + rng.normal(0, spread * 0.8)
                    r2 = x * x + y * y
                    if r2 < 0.92:
                        pts.append([x, y, -math.sqrt(1 - r2)])
            for j in range(18):
                x, y = rng.uniform(-0.85, 0.35), rng.uniform(-0.8, 0.8)
                r2 = x * x + y * y
                if r2 < 0.9:
                    pts.append([x, y, -math.sqrt(1 - r2)])
            pts = np.array(pts)
            b = 0.35 + rng.random(len(pts)) ** 2 * 1.3
            self._cities = (pts, b)
        return self._cities

    # ---- ring
    def ring_path(self):
        """World directions along the orbital ring + lit factor. Physically placed
        equatorial ring (radius in Earth radii) seen from latitude lat; the view axis +z
        points to azimuth view_az. The Earth-shadow boundary is placed at `shadow_theta`."""
        if self.ring is None:
            return None
        if self._ring is None:
            r = self.ring
            lat = r.get('lat', 57.0) * DEG
            Rr = r.get('radius', 3.2)
            view_az = r.get('view_az', 235.0) * DEG
            n = 20000
            th = np.linspace(-math.pi, math.pi, n, endpoint=False)
            P = np.stack([Rr * np.cos(th), Rr * np.sin(th), np.zeros(n)], 1)
            O = np.array([math.cos(lat), 0, math.sin(lat)])
            up = O.copy()
            east = np.array([0.0, 1.0, 0.0])
            north = np.array([-math.sin(lat), 0, math.cos(lat)])
            v = P - O
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            e, u, no = v @ east, v @ up, v @ north
            A = np.arctan2(e, no)            # azimuth from north toward east
            el = np.arcsin(np.clip(u, -1, 1))
            dA = (A - view_az + math.pi) % (2 * math.pi) - math.pi
            dirs = np.stack([np.sin(dA) * np.cos(el), np.sin(el), np.cos(dA) * np.cos(el)], 1)
            # lit factor: smooth fade into Earth's shadow around theta_shadow
            ts = r.get('shadow_theta', 60.0) * DEG
            tw = r.get('shadow_width', 18.0) * DEG
            dth = np.abs((th - ts + math.pi) % (2 * math.pi) - math.pi)
            lit = np.clip((dth - tw) / (0.6 * tw), 0, 1)
            lit = lit * lit * (3 - 2 * lit)
            lit = r.get('shadow_floor', 0.06) + (1 - r.get('shadow_floor', 0.06)) * lit
            # station nodes
            node = np.zeros(n)
            for k in range(r.get('nodes', 24)):
                c = -math.pi + (k + 0.37) * 2 * math.pi / r.get('nodes', 24)
                node += np.exp(-(((th - c + math.pi) % (2 * math.pi) - math.pi) / 0.0025) ** 2)
            self._ring = (dirs, lit, node, th)
        return self._ring


# ---------------------------------------------------------------- kernels ---

@nb.njit(cache=True, fastmath=True, inline='always')
def sky_rad(dx, dy, dz, sp):
    """Sky radiance (linear) in direction d. `sp` = Sky.packed()."""
    el = math.asin(max(-1.0, min(1.0, dy)))
    h = max(el, 0.0)
    # horizontal cosine to the sun azimuth
    hx = dx
    hz = dz
    hn = math.sqrt(hx * hx + hz * hz) + 1e-9
    sx = sp[0]
    sz = sp[2]
    sn = math.sqrt(sx * sx + sz * sz) + 1e-9
    caz = (hx * sx + hz * sz) / (hn * sn)
    w = (0.5 + 0.5 * caz) ** sp[22]
    eb = math.exp(-h / sp[18])
    ea = math.exp(-h / sp[19]) * w * w
    er = math.exp(-h / sp[20]) * w
    ev = math.exp(-h / sp[21]) * (0.35 + 0.65 * w)
    r = sp[3] + (sp[6] - sp[3]) * eb + sp[9] * ea + sp[12] * er + sp[15] * ev
    g = sp[4] + (sp[7] - sp[4]) * eb + sp[10] * ea + sp[13] * er + sp[16] * ev
    b = sp[5] + (sp[8] - sp[5]) * eb + sp[11] * ea + sp[14] * er + sp[17] * ev
    # moon aureole
    if sp[26] > 0:
        cm = dx * sp[23] + dy * sp[24] + dz * sp[25]
        ang = math.acos(max(-1.0, min(1.0, cm)))
        a = sp[26] * (math.exp(-ang / 0.035) * 1.5 + math.exp(-ang / 0.25))
        r += a * sp[28]
        g += a * sp[29]
        b += a * sp[30]
    gg = sp[38]
    return r * gg, g * gg, b * gg


@nb.njit(cache=True, fastmath=True)
def mw_rad(dx, dy, dz, sp):
    """Milky Way band intensity (scalar) in direction d."""
    if sp[31] <= 0:
        return 0.0
    gx, gy, gz = sp[32], sp[33], sp[34]
    sb = dx * gx + dy * gy + dz * gz
    b = math.asin(max(-1.0, min(1.0, sb)))
    if abs(b) > 0.6:
        return 0.0
    cx, cy, cz = sp[35], sp[36], sp[37]
    # second in-plane axis
    tx = gy * cz - gz * cy
    ty = gz * cx - gx * cz
    tz = gx * cy - gy * cx
    l = math.atan2(dx * tx + dy * ty + dz * tz, dx * cx + dy * cy + dz * cz)
    width = 0.16 + 0.05 * math.exp(-l * l / 0.4)
    band = math.exp(-(b / width) ** 2)
    bulge = 1.0 + 1.6 * math.exp(-(l * l / 0.25 + b * b / 0.04))
    n1 = fbm3(dx * 5.0, dy * 5.0, dz * 5.0, 5, 2.1, 0.55)
    n2 = fbm3(dx * 18.0 + 3.1, dy * 18.0, dz * 18.0 - 1.7, 4, 2.2, 0.5)
    clump = 0.55 + 0.6 * n1 + 0.25 * n2
    # dark rift near the plane
    rift = math.exp(-((b - 0.02 + 0.04 * n1) / 0.035) ** 2) * (0.5 + 0.5 * math.tanh(3.0 * (0.4 - abs(l))))
    dust = 1.0 - 0.75 * rift * (0.6 + 0.8 * max(0.0, n2 + 0.2))
    v = band * bulge * max(0.0, clump) * max(0.0, dust)
    return sp[31] * v


@nb.njit(cache=True, fastmath=True, parallel=False)
def render_sky(out, cam, sp, mw_col):
    H = out.shape[0]
    W = out.shape[1]
    for j in range(H):
        for i in range(W):
            dx, dy, dz = cam_ray(cam, i + 0.5, j + 0.5)
            r, g, b = sky_rad(dx, dy, dz, sp)
            m = mw_rad(dx, dy, dz, sp)
            if m > 0:
                # visibility against sky brightness
                lum = 0.3 * r + 0.6 * g + 0.1 * b
                vis = max(0.0, 1.0 - lum / 0.05)
                r += m * mw_col[0] * vis
                g += m * mw_col[1] * vis
                b += m * mw_col[2] * vis
            out[j, i, 0] = r
            out[j, i, 1] = g
            out[j, i, 2] = b


@nb.njit(cache=True, fastmath=True)
def render_stars(out, cam, sp, dirs, flux, cols, phase, t, gain, thresh, sig, escale, twinkle):
    H = out.shape[0]
    W = out.shape[1]
    f = cam[12]
    for k in range(dirs.shape[0]):
        dx, dy, dz = dirs[k, 0], dirs[k, 1], dirs[k, 2]
        if dy < -0.02:
            continue
        # into camera space
        qx = cam[3] * dx + cam[6] * dy + cam[9] * dz
        qy = cam[4] * dx + cam[7] * dy + cam[10] * dz
        qz = cam[5] * dx + cam[8] * dy + cam[11] * dz
        if qz <= 0.05:
            continue
        px = cam[13] + f * qx / qz
        py = cam[14] - f * qy / qz
        if px < -3 or py < -3 or px > W + 3 or py > H + 3:
            continue
        r, g, b = sky_rad(dx, dy, dz, sp)
        lum = 0.3 * r + 0.6 * g + 0.1 * b
        F = flux[k] * gain
        # extinction toward the horizon
        el = math.asin(max(-1.0, min(1.0, dy)))
        ext = math.exp(-0.12 / max(0.03, math.sin(max(el, 0.0)) + 0.03))
        F *= ext
        # contrast threshold against sky
        vis = F - thresh * lum
        if vis <= 0:
            continue
        # twinkle (stronger near horizon)
        tw = twinkle * (0.5 + 1.5 * math.exp(-el / 0.25))
        n = perlin3(phase[k], t * 2.7, 0.5) + 0.5 * perlin3(phase[k] + 9.1, t * 7.3, 1.5)
        vis *= max(0.0, 1.0 + tw * n)
        e = vis * escale
        splat_gauss(out, px, py, sig, e * cols[k, 0], e * cols[k, 1], e * cols[k, 2])


@nb.njit(cache=True, fastmath=True)
def render_moon(out, cov, cam, mdir, rad, sdir, gain, earthshine, e1, e2, seed):
    """Moon disc (additive radiance) + coverage mask (for hiding stars behind it)."""
    H = out.shape[0]
    W = out.shape[1]
    f = cam[12]
    qx = cam[3] * mdir[0] + cam[6] * mdir[1] + cam[9] * mdir[2]
    qy = cam[4] * mdir[0] + cam[7] * mdir[1] + cam[10] * mdir[2]
    qz = cam[5] * mdir[0] + cam[8] * mdir[1] + cam[11] * mdir[2]
    if qz <= 0.05:
        return
    px = cam[13] + f * qx / qz
    py = cam[14] - f * qy / qz
    rpx = f * math.tan(rad) / qz
    R = int(rpx + 4)
    for j in range(max(0, int(py) - R), min(H, int(py) + R + 1)):
        for i in range(max(0, int(px) - R), min(W, int(px) + R + 1)):
            dx, dy, dz = cam_ray(cam, i + 0.5, j + 0.5)
            a1 = (dx * e1[0] + dy * e1[1] + dz * e1[2]) / math.tan(rad)
            a2 = (dx * e2[0] + dy * e2[1] + dz * e2[2]) / math.tan(rad)
            r2 = a1 * a1 + a2 * a2
            rr = math.sqrt(r2)
            c = (1.0 - rr) * rpx + 0.5
            if c <= 0:
                continue
            if c > 1:
                c = 1.0
            z = math.sqrt(max(0.0, 1.0 - min(r2, 1.0)))
            # surface normal (world): a1 e1 + a2 e2 - z m
            nx = a1 * e1[0] + a2 * e2[0] - z * mdir[0]
            ny = a1 * e1[1] + a2 * e2[1] - z * mdir[1]
            nz = a1 * e1[2] + a2 * e2[2] - z * mdir[2]
            mu = nx * sdir[0] + ny * sdir[1] + nz * sdir[2]
            # lambert with a slightly softened terminator
            lit = 0.0
            if mu > -0.03:
                q = min(1.0, (mu + 0.03) / 0.09)
                lit = (max(mu, 0.0) ** 0.75) * q * q * (3 - 2 * q) + 0.02 * q
            # albedo: maria
            n = fbm3(a1 * 1.7 + seed, a2 * 1.7, z * 1.7, 5, 2.0, 0.55)
            n2 = fbm3(a1 * 6.0 - seed, a2 * 6.0, z * 6.0 + 2.0, 3, 2.0, 0.5)
            alb = 0.62 + 0.30 * math.tanh(3.0 * (n + 0.05)) + 0.10 * n2
            es = earthshine * (0.55 + 0.45 * z) * alb
            v = (lit * alb * gain + es) * c
            out[j, i, 0] += v * 1.0
            out[j, i, 1] += v * 0.97
            out[j, i, 2] += v * 0.9
            cov[j, i] = max(cov[j, i], c)


def draw_moon(img, cam, sky, t=0.0, escale=1.0, star_mask=None):
    """Moon + city lights. Returns moon coverage (HxW) for star masking."""
    H, W = img.shape[:2]
    cov = np.zeros((H, W), np.float32)
    if sky.moon_dir is None:
        return cov
    m = sky.moon_dir
    # disc axes: e1 horizontal-ish right, e2 up
    e1 = np.cross([0, 1, 0], m)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(m, e1)
    rad = sky.moon_radius_deg * DEG
    al = sky.moon_phase * DEG
    pa = sky.moon_pa * DEG
    sm = math.cos(al) * (-m) + math.sin(al) * (math.cos(pa) * e1 + math.sin(pa) * e2)
    sm /= np.linalg.norm(sm)
    render_moon(img, cov, cam.params(), m.astype(np.float64), rad, sm.astype(np.float64),
                float(sky.moon_gain), float(sky.earthshine), e1, e2, float(sky.seed) * 3.3)
    # city lights on the night side
    pts, b = sky.cities()
    # moon-fixed frame: x=e1, y=e2, z=m (so -z faces the observer)
    P = pts[:, 0:1] * e1 + pts[:, 1:2] * e2 + pts[:, 2:3] * m
    vis_side = -(P @ m)              # >0 on the visible hemisphere
    mu = P @ sm
    camp = cam.params()
    sx, sy, _ = None, None, None
    for k in range(len(pts)):
        if vis_side[k] < 0.08 or mu[k] > -0.04:
            continue
        # sky direction of that surface point
        d = m + math.tan(rad) * (pts[k, 0] * e1 + pts[k, 1] * e2)
        d /= np.linalg.norm(d)
        px, py, z = cam.project(cam.pos + d * 1e6)
        if z <= 0:
            continue
        night = min(1.0, (-mu[k] - 0.04) / 0.12)
        e = sky.city_gain * b[k] * night * (0.4 + 0.6 * vis_side[k]) * escale
        tw = 1.0 + 0.25 * math.sin(t * 0.9 + k * 1.7)
        splat_gauss(img, float(px), float(py), 0.45 * cam.scale + 0.2, e * 1.0 * tw, e * 0.62 * tw, e * 0.30 * tw)
    return cov


def draw_ring(img, cam, sky, gain=1.0, width=0.75, t=0.0):
    """Thin luminous ring polyline (energy per pixel length ~ gain)."""
    rp = sky.ring_path()
    if rp is None:
        return
    dirs, lit, node, th = rp
    sx, sy, z = cam.project(cam.pos + dirs * 1e7)
    ok = (z > 0) & (dirs[:, 1] > -0.03)
    ok &= (sx > -50) & (sx < cam.W + 50) & (sy > -50) & (sy < cam.H + 50)
    idx = np.nonzero(ok)[0]
    if len(idx) < 2:
        return
    col = sky.ring.get('color', np.array([1.0, 0.93, 0.82]))
    sig = max(0.35, width * cam.scale)
    _ring_kernel(img, sx.astype(np.float64), sy.astype(np.float64), ok.astype(np.uint8),
                 lit.astype(np.float64), node.astype(np.float64), dirs[:, 1].astype(np.float64),
                 np.asarray(col, np.float64), float(gain * cam.scale), sig,
                 float(sky.ring.get('node_gain', 2.5)), float(t))


@nb.njit(cache=True, fastmath=True)
def _ring_kernel(img, sx, sy, ok, lit, node, ely, col, gain, sig, node_gain, t):
    n = sx.shape[0]
    for k in range(n):
        k2 = (k + 1) % n
        if ok[k] == 0 or ok[k2] == 0:
            continue
        L = math.sqrt((sx[k2] - sx[k]) ** 2 + (sy[k2] - sy[k]) ** 2)
        if L > 60:
            continue
        # horizon extinction
        el = math.asin(max(-1.0, min(1.0, ely[k])))
        ext = math.exp(-0.08 / max(0.02, math.sin(max(el, 0.0)) + 0.02))
        # faint travelling shimmer
        sh = 1.0 + 0.08 * math.sin(k * 0.013 + t * 0.4)
        e = gain * L * lit[k] * ext * sh * (1.0 + node_gain * node[k])
        splat_streak(img, sx[k], sy[k], sx[k2], sy[k2], sig, e * col[0], e * col[1], e * col[2])


def render_full_sky(cam, sky, t, star_sig=0.6, mw_col=None, ring_gain=1.0):
    """Complete sky layer (HxWx3 float32 linear)."""
    img = np.zeros((cam.H, cam.W, 3), np.float32)
    sp = sky.packed()
    camp = cam.params()
    if mw_col is None:
        mw_col = np.array([0.80, 0.86, 1.0]) * 0.02
    render_sky(img, camp, sp, np.asarray(mw_col, np.float64))
    stars = np.zeros_like(img)
    v, flux, col, ph = sky.stars()
    render_stars(stars, camp, sp, v, flux, col, ph, float(t), float(sky.star_gain),
                 float(sky.star_thresh), max(0.3, star_sig * cam.scale), cam.scale ** 2, 0.35)
    moon = np.zeros_like(img)
    cov = draw_moon(moon, cam, sky, t=t, escale=cam.scale ** 2)
    img += stars * (1.0 - cov[..., None]) + moon
    draw_ring(img, cam, sky, gain=ring_gain, t=t)
    return img
