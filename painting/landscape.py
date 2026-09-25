"""
The world outside the loggia.

A real terrain -- valley, lake, river, a hill town, crags and a far range of
Leonardo's mountains -- rendered column by column from the viewpoint of the
loggia (a "voxel space" renderer, marching outward along every azimuth at
once), then coloured the way the Florentines coloured distance: every mile of
air adds a veil of blue, gold toward the sun. Following Leonardo's own note
that the tops of mountains look clearer than their feet, the veil thins with
altitude.

Everything is a function of direction (azimuth, elevation in degrees), so the
same world renders sharply where we see it directly and coarsely, but
identically, for the wide, inverted view seen through the glass.
"""
import math

import numpy as np

SUN_AZ, SUN_EL = -38.0, 44.0
CAM_H = 640.0                       # metres above the valley floor
SUN = np.array([math.sin(math.radians(SUN_AZ)) * math.cos(math.radians(SUN_EL)),
                math.sin(math.radians(SUN_EL)),
                math.cos(math.radians(SUN_AZ)) * math.cos(math.radians(SUN_EL))])


# ---------------------------------------------------------------------------
# resolution-independent noise
# ---------------------------------------------------------------------------
def _hash(ix, iy, seed):
    h = (ix * np.int64(73856093)) ^ (iy * np.int64(19349663)) ^ np.int64(seed * 83492791 + 17)
    h = (h ^ (h >> 13)) * np.int64(1274126177)
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float32) / np.float32(0xFFFFFF)


def vnoise(x, y, seed=0):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = (x - ix).astype(np.float32)
    fy = (y - iy).astype(np.float32)
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    a = _hash(ix, iy, seed)
    b = _hash(ix + 1, iy, seed)
    c = _hash(ix, iy + 1, seed)
    d = _hash(ix + 1, iy + 1, seed)
    ab = a + (b - a) * ux
    return ab + ((c + (d - c) * ux) - ab) * uy


def fbm(x, y, octaves=4, seed=0, gain=0.5, lac=2.0):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    s = np.zeros(np.broadcast(x, y).shape, np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        s += amp * (vnoise(x, y, seed + 31 * o) - 0.5)
        norm += amp * 0.5
        x = x * lac + 5.3
        y = y * lac + 1.7
        amp *= gain
    return s / norm


def ridged2(x, y, octaves=6, seed=0, sharp=2.0, gain=0.5):
    """Ridged multifractal: knife-edge crests and deep gullies."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    s = np.zeros(np.broadcast(x, y).shape, np.float32)
    amp, norm, weight = 1.0, 0.0, 1.0
    for o in range(octaves):
        n = 1 - np.abs(2 * vnoise(x, y, seed + 17 * o) - 1)
        n = n ** sharp
        s += amp * n * weight
        weight = np.clip(n * 1.5, 0, 1)
        norm += amp
        x = x * 2.1 + 3.7
        y = y * 2.1 + 1.3
        amp *= gain
    return s / norm


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Air and sky
# ---------------------------------------------------------------------------
def sun_angle(az, el):
    a1, e1 = np.radians(az), np.radians(el)
    a2, e2 = math.radians(SUN_AZ), math.radians(SUN_EL)
    c = np.sin(e1) * math.sin(e2) + np.cos(e1) * math.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def haze_color(az):
    """The colour of the air near the horizon: gold toward the sun, blue away."""
    warm = 0.85 * np.exp(-np.abs(np.asarray(az, np.float64) - SUN_AZ) / 24.0)[..., None]
    cool = np.array([0.50, 0.60, 0.80], np.float32)
    gold = np.array([0.98, 0.80, 0.54], np.float32)
    return (cool * (1 - warm) + gold * warm).astype(np.float32)


def sky(az, el):
    az = np.asarray(az, np.float64)
    el = np.asarray(el, np.float64)
    e = np.maximum(el, 0.0)
    t = smoothstep(0.0, 1.0, (e / 48.0) ** 0.62)[..., None]
    horizon = haze_color(az) * 1.08
    zenith = np.array([0.05, 0.12, 0.40], np.float32)
    col = horizon * (1 - t) + zenith * t
    g = sun_angle(az, el)[..., None]
    col = col + np.array([1.0, 0.80, 0.52], np.float32) * (0.9 * np.exp(-g / 7.0) + 0.25 * np.exp(-g / 26.0))
    # clouds: soft banks, gold on the sun side, blue-grey beneath
    cx, cy = az / 16.0, el / 4.2
    dens = fbm(cx, cy, 5, seed=101) + 0.45 * fbm(cx * 2.7, cy * 2.0, 3, seed=202)
    band = smoothstep(5.0, 10.0, e) * (1 - smoothstep(20.0, 34.0, e))
    d = np.clip((dens - 0.22) * 2.6, 0, 1) * band
    grad = fbm(cx - 0.12, cy + 0.10, 5, seed=101) - fbm(cx, cy, 5, seed=101)
    lit = np.clip(0.45 + 2.2 * grad, 0, 1)[..., None]
    c_lit = np.array([1.10, 0.95, 0.78], np.float32)
    c_shd = np.array([0.52, 0.56, 0.70], np.float32)
    ccol = c_shd * (1 - lit) + c_lit * lit
    d = d[..., None]
    return (col * (1 - 0.9 * d) + ccol * (0.9 * d)).astype(np.float32)


# ---------------------------------------------------------------------------
# The land
# ---------------------------------------------------------------------------
def river_x(Z):
    """Centre line of the river (km), as it winds from the lake toward us."""
    Z = np.asarray(Z, np.float64)
    return 0.30 + 0.75 * np.sin(Z * 0.95 + 0.4) * np.clip(Z / 6.0, 0.25, 1.0) - 0.08 * Z


LAKE = (-1.6, 13.5, 6.5, 3.8)          # centre x, z, radius x, z (km)
TOWN = (-1.10, 4.10)                    # km
BRIDGE_Z = 3.0


def terrain(X, Z):
    """Height in metres at (X, Z) km. Water level is 0."""
    X = np.asarray(X, np.float64)
    Z = np.asarray(Z, np.float64)
    # rolling country
    hills = 170.0 * (fbm(X / 1.7, Z / 1.7, 6, seed=3) + 0.42)
    hills = np.maximum(hills, 0.0) + 22.0 * (fbm(X * 1.3, Z * 1.3, 3, seed=4) + 0.5) + 6.0 * fbm(X * 9, Z * 9, 3, seed=5)
    # the river's valley, and the river in it
    dv = np.abs(X - river_x(Z))
    valley = smoothstep(0.08, 1.3 + 0.05 * Z, dv)
    wid = 0.055 + 0.002 * Z
    h = hills * (0.03 + 0.97 * valley) - 9.0 * (1 - smoothstep(wid * 0.6, wid, dv))
    # the town's hill
    tx, tz = TOWN
    h += 235.0 * np.exp(-(((X - tx) / 0.55) ** 2 + ((Z - tz) / 0.45) ** 2) ** 1.4)
    # shoulders of land near us, left and right (the painter's repoussoir)
    h += 300.0 * np.exp(-(((X + 1.5) / 0.9) ** 2 + ((Z - 1.9) / 1.0) ** 2))
    h += 340.0 * np.exp(-(((X - 1.7) / 1.0) ** 2 + ((Z - 2.4) / 1.1) ** 2))
    # the lake
    lx, lz, rx, rz = LAKE
    dl = ((X - lx) / rx) ** 2 + ((Z - lz) / rz) ** 2
    h = np.where(dl < 1.4, h * smoothstep(0.75, 1.35, dl) - 12.0 * (1 - smoothstep(0.7, 1.0, dl)), h)
    # crags in the middle distance, a little fantastical
    cr = (np.exp(-(((X - 3.8) / 1.6) ** 2 + ((Z - 13.0) / 1.4) ** 2))
          + 0.8 * np.exp(-(((X + 8.2) / 1.4) ** 2 + ((Z - 12.0) / 1.2) ** 2)))
    h += 900.0 * cr * ridged2(X / 0.9, Z / 0.9, 6, seed=21, sharp=2.4)
    # the far range
    az = np.degrees(np.arctan2(X, np.maximum(Z, 1e-3)))
    massif = (0.55 + np.exp(-((az + 17) / 11) ** 2) + 0.7 * np.exp(-((az - 28) / 10) ** 2)
              + 0.5 * np.exp(-((az + 52) / 14) ** 2))
    far = smoothstep(21.0, 30.0, Z) * massif
    h += 2300.0 * far * (0.25 + 0.75 * ridged2(X / 5.0, Z / 5.0, 7, seed=31, sharp=1.8)) ** 1.4
    return h


def terrain_grad(X, Z, e=0.004):
    """Height and its gradient (metres per metre)."""
    h = terrain(X, Z)
    hx = (terrain(X + e, Z) - terrain(X - e, Z)) / (2 * e * 1000.0)
    hz = (terrain(X, Z + e) - terrain(X, Z - e)) / (2 * e * 1000.0)
    return h, hx, hz


def march(az_deg, d_min=0.25, d_max=60.0, rate=0.0025):
    """Distances (km) and, for every column, the ground positions along its ray."""
    n = int(math.log(d_max / d_min) / math.log(1 + rate)) + 1
    d = d_min * (1 + rate) ** np.arange(n)
    a = np.radians(az_deg)
    X = d[:, None] * np.sin(a)[None, :]
    Z = d[:, None] * np.cos(a)[None, :]
    return d, X, Z


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def paint_env(az0, az1, el0, el1, ppd):
    """Render the world for az in [az0, az1], el in [el0, el1] at `ppd`
    pixels per degree. Returns linear RGB float32 (h, w, 3) and the distance
    (km) to whatever each pixel sees (inf for sky)."""
    w = int(round((az1 - az0) * ppd))
    hgt = int(round((el1 - el0) * ppd))
    az = az0 + (np.arange(w, dtype=np.float64) + 0.5) / ppd
    el = el1 - (np.arange(hgt, dtype=np.float64) + 0.5) / ppd

    d, X, Z = march(az)
    Hh = terrain(X, Z)
    H_eff = Hh - (d[:, None] * 1000.0) ** 2 / (2 * 7.4e6)      # the earth curves away (with refraction)
    elev = np.degrees(np.arctan2(np.maximum(H_eff, 0.0) - CAM_H, d[:, None] * 1000.0))
    M = np.maximum.accumulate(elev, axis=0)                     # silhouette so far, near to far
    k = np.empty((hgt, w), np.int64)
    targets = el[::-1]                                          # ascending
    for c in range(w):
        k[:, c] = np.searchsorted(M[:, c], targets, side="left")[::-1]
    sky_mask = k >= len(d)
    kk = np.minimum(k, len(d) - 1)
    cols = np.broadcast_to(np.arange(w), (hgt, w))
    D = d[kk]
    Xp, Zp = X[kk, cols], Z[kk, cols]
    AZ = np.broadcast_to(az[None, :], (hgt, w))
    EL = np.broadcast_to(el[:, None], (hgt, w))

    img = np.zeros((hgt, w, 3), np.float32)
    img[sky_mask] = sky(AZ[sky_mask], EL[sky_mask])
    land = ~sky_mask
    Dl, Xl, Zl = D[land], Xp[land], Zp[land]
    h, hx, hz = terrain_grad(Xl, Zl)
    water = h <= 0.0
    nrm = np.stack([-hx, np.ones_like(hx), -hz], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    slope = np.hypot(hx, hz)

    # materials
    meadow = np.array([0.105, 0.125, 0.045], np.float32)
    field = np.array([0.24, 0.19, 0.085], np.float32)
    wood = np.array([0.030, 0.048, 0.024], np.float32)
    rock = np.array([0.22, 0.20, 0.17], np.float32)
    snow = np.array([0.80, 0.82, 0.86], np.float32)
    f_field = (smoothstep(0.1, 0.3, fbm(Xl * 2.2, Zl * 2.2, 4, seed=51)) * smoothstep(0.35, 0.1, slope))[:, None]
    f_wood = smoothstep(0.05, 0.25, fbm(Xl * 1.4 + 3, Zl * 1.4, 4, seed=52))[:, None]
    f_rock = smoothstep(0.45, 0.9, slope)[:, None]
    f_snow = (smoothstep(1500.0, 2100.0, h) * smoothstep(1.2, 0.5, slope))[:, None]
    alb = meadow * (1 - f_field) + field * f_field
    alb = alb * (1 - 0.8 * f_wood) + wood * (0.8 * f_wood)
    alb = alb * (1 - f_rock) + rock * f_rock
    alb = alb * (1 - f_snow) + snow * f_snow
    alb *= (0.85 + 0.3 * fbm(Xl * 9, Zl * 9, 3, seed=53))[:, None]

    # light: the sun, and the sky's blue fill
    ndl = np.clip(nrm @ SUN, 0, 1)[:, None]
    sky_amb = np.array([0.34, 0.42, 0.60], np.float32) * (0.55 + 0.45 * nrm[:, 1:2])
    col = alb * (1.35 * np.array([1.0, 0.90, 0.72], np.float32) * ndl + 0.55 * sky_amb)

    # water: the sky, mirrored
    if water.any():
        el_w = EL[land][water]
        refl = sky(AZ[land][water], -el_w + 0.3)
        rip = fbm(Xl[water] * 30, Zl[water] * 6, 3, seed=61)
        col[water] = refl * (0.62 + 0.18 * rip)[:, None]

    # air: Leonardo's blue veil, thinner on the heights
    dens = 1.0 / (1.0 + np.maximum(h, 0.0) / 900.0)
    T = np.exp(-Dl * dens / 21.0)[:, None]
    col = col * T + haze_color(AZ[land]) * (1 - T)
    img[land] = col

    dist = np.where(sky_mask, np.inf, D).astype(np.float32)
    add_sprites(img, dist, az0, az1, el0, el1, ppd)
    return img, dist


# ---------------------------------------------------------------------------
# Trees, the town, the bridge: small painted things, placed in the world
# ---------------------------------------------------------------------------
def project(X, Z, y):
    """World point (km, km, metres above the valley) to (az, el, distance km)."""
    d = np.hypot(X, Z)
    az = np.degrees(np.arctan2(X, Z))
    el = np.degrees(np.arctan2(y - (d * 1000.0) ** 2 / (2 * 7.4e6) - CAM_H, d * 1000.0))
    return az, el, d


def air(col, d, h, az):
    dens = 1.0 / (1.0 + max(h, 0.0) / 900.0)
    T = math.exp(-d * dens / 17.0)
    return np.asarray(col, np.float32) * T + haze_color(np.array([az]))[0] * (1 - T)


def composite(img, dist, x0, y0, rgb, alpha, d):
    """Lay a painted patch into the world, behind whatever is nearer."""
    H, W = img.shape[:2]
    h, w = alpha.shape
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    a = alpha[ya - y0:yb - y0, xa - x0:xb - x0]
    c = rgb[ya - y0:yb - y0, xa - x0:xb - x0]
    vis = dist[ya:yb, xa:xb] > d
    a = a * vis
    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a[..., None]) + c * a[..., None]
    dist[ya:yb, xa:xb] = np.where(a > 0.5, d, dist[ya:yb, xa:xb])


def blob(xx, yy, cx, cy, rx, ry):
    q = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    return np.clip((1 - q) * min(rx, ry) + 0.5, 0, 1)


def paint_tree(kind, hpx, rng, dark, mid, lit, trunk):
    """A tree in a small patch, lit from the upper left. hpx = height in pixels."""
    wpx = max(3, int(hpx * (0.30 if kind == 2 else 0.75)) + 3)
    hh = max(4, int(hpx) + 3)
    yy, xx = np.mgrid[0:hh, 0:wpx].astype(np.float32)
    cx0, base = wpx / 2.0, hh - 1.0
    alpha = np.zeros((hh, wpx), np.float32)
    rgb = np.zeros((hh, wpx, 3), np.float32)

    def put(m, shade):
        col = dark[None, None] * (1 - shade[..., None]) + mid[None, None] * shade[..., None]
        col = col * (1 - np.clip(shade - 0.6, 0, 1)[..., None] * 2) + lit[None, None] * (np.clip(shade - 0.6, 0, 1)[..., None] * 2)
        rgb[:] = rgb * (1 - m[..., None]) + col * m[..., None]
        alpha[:] = alpha + m * (1 - alpha)

    tw = max(0.6, hpx * 0.035)
    tm = np.clip(tw - np.abs(xx - cx0), 0, 1) * (yy > base - hpx * (0.5 if kind == 1 else 0.9))
    rgb[:] = trunk[None, None] * tm[..., None]
    alpha[:] = tm
    if kind == 0:                                   # Umbrian: slender, feathery
        n = int(rng.integers(6, 10))
        for i in range(n):
            f = 0.30 + 0.70 * i / (n - 1)
            r = hpx * (0.12 - 0.06 * f) * rng.uniform(0.8, 1.2) + 0.6
            cx = cx0 + (-1) ** i * rng.uniform(0.05, 0.14) * hpx * (1.1 - f)
            cy = base - f * hpx * 0.95
            m = blob(xx, yy, cx, cy, r * 1.15, r * 0.8)
            shade = np.clip(0.55 - 0.55 * ((xx - cx) * 0.7 + (yy - cy) * 0.7) / r, 0, 1)
            put(m, shade)
    elif kind == 1:                                 # round crown
        R = hpx * 0.33
        for i in range(10):
            ang = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(0, 0.55) * R
            cx = cx0 + rr * math.cos(ang)
            cy = base - hpx * 0.62 + rr * math.sin(ang) * 0.8
            r = R * rng.uniform(0.45, 0.6) + 0.6
            m = blob(xx, yy, cx, cy, r, r * 0.85)
            gx, gy = (xx - (cx0 - R * 0.3)) / R, (yy - (base - hpx * 0.8)) / R
            shade = np.clip(0.6 - 0.45 * (gx * 0.7 + gy * 0.7) - 0.25 * ((xx - cx) * 0.7 + (yy - cy) * 0.7) / r, 0, 1)
            put(m, shade)
    else:                                           # cypress: a dark flame
        prof = np.clip((base - yy) / hpx, 0, 1)
        half = hpx * 0.13 * np.sin(np.pi * np.clip(prof, 0, 1) ** 0.8) * (1 - 0.3 * prof) + 0.5
        m = np.clip(half - np.abs(xx - cx0) + 0.5, 0, 1) * (prof > 0.02)
        shade = np.clip(0.35 - 0.8 * (xx - cx0) / (half + 1e-3) * 0.5 + 0.1 * rng.normal(size=xx.shape), 0, 1)
        put(m, shade * 0.8)
    return rgb, alpha


def add_sprites(img, dist, az0, az1, el0, el1, ppd):
    rng = np.random.default_rng(77)
    sun_c = 1.35 * np.array([1.0, 0.90, 0.72], np.float32)
    amb = 0.55 * np.array([0.34, 0.42, 0.60], np.float32)
    # --- trees ---------------------------------------------------------
    n_cand = int(9000 * (az1 - az0) / 50.0)
    Zc = 0.35 + (rng.uniform(0, 1, n_cand) ** 1.6) * 9.0
    azc = rng.uniform(az0 - 2, az1 + 2, n_cand)
    Xc = Zc * np.tan(np.radians(azc))
    h, hx, hz = terrain_grad(Xc, Zc)
    wood = smoothstep(0.0, 0.25, fbm(Xc * 1.4 + 3, Zc * 1.4, 4, seed=52))
    keep = (h > 3.0) & (np.hypot(hx, hz) < 0.45) & (rng.uniform(size=n_cand) < 0.12 + 0.88 * wood)
    town_d = np.hypot(Xc - TOWN[0], Zc - TOWN[1])
    keep &= town_d > 0.35
    Xc, Zc, hc = Xc[keep], Zc[keep], h[keep]
    order = np.argsort(-np.hypot(Xc, Zc))           # far to near
    for i in order:
        X, Z, y = Xc[i], Zc[i], hc[i]
        tall = rng.uniform(9, 22)
        kind = int(rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20]))
        if kind == 2:
            tall *= 1.35
        az_b, el_b, d = project(X, Z, y)
        _, el_t, _ = project(X, Z, y + tall)
        hpx = (el_t - el_b) * ppd
        if hpx < 1.5:
            continue
        x = (az_b - az0) * ppd
        yb = (el1 - el_b) * ppd
        light = sun_c * 0.55 + amb
        dark = air(np.array([0.020, 0.030, 0.016]) * light, d, y, az_b)
        mid = air(np.array([0.045, 0.058, 0.024]) * light, d, y, az_b)
        lit = air(np.array([0.11, 0.115, 0.045]) * light, d, y, az_b)
        trunk = air(np.array([0.03, 0.025, 0.02]) * light, d, y, az_b)
        rgb, alpha = paint_tree(kind, hpx, rng, dark, mid, lit, trunk)
        composite(img, dist, int(x - alpha.shape[1] / 2), int(yb - alpha.shape[0] + 1), rgb, alpha, d - 0.005)
    # --- the town --------------------------------------------------------
    trng = np.random.default_rng(9)
    bld = []
    for _ in range(46):
        bx = TOWN[0] + trng.normal(0, 0.12)
        bz = TOWN[1] + trng.normal(0, 0.10)
        bld.append((bx, bz, trng.uniform(8, 16), trng.uniform(10, 22), 0))
    bld.append((TOWN[0] - 0.02, TOWN[1] + 0.02, 55.0, 9.0, 1))     # campanile
    bld.append((TOWN[0] + 0.07, TOWN[1] + 0.06, 30.0, 26.0, 2))    # the dome
    bld.sort(key=lambda b: -math.hypot(b[0], b[1]))
    for bx, bz, bh, bw, kind in bld:
        y = float(terrain(np.array([bx]), np.array([bz]))[0]) - 2
        az_b, el_b, d = project(bx, bz, y)
        _, el_t, _ = project(bx, bz, y + bh)
        hpx = (el_t - el_b) * ppd
        wpx = math.degrees(bw / (d * 1000)) * ppd
        if hpx < 1:
            continue
        x = (az_b - az0) * ppd
        yb = (el1 - el_b) * ppd
        W_, H_ = int(wpx * 1.8) + 4, int(hpx * 1.6) + 4
        yy, xx = np.mgrid[0:H_, 0:W_].astype(np.float32)
        base, left = H_ - 2.0, 2.0
        rgb = np.zeros((H_, W_, 3), np.float32)
        alpha = np.zeros((H_, W_), np.float32)
        wall_l = air(np.array([0.62, 0.55, 0.44]) * (sun_c * 0.5 + amb), d, y, az_b)
        wall_s = air(np.array([0.62, 0.55, 0.44]) * amb * 1.1, d, y, az_b)
        roof = air(np.array([0.36, 0.14, 0.07]) * (sun_c * 0.45 + amb), d, y, az_b)

        def rect(x0_, x1_, y0_, y1_, col):
            m = (np.clip(xx - x0_ + 0.5, 0, 1) * np.clip(x1_ - xx + 0.5, 0, 1) *
                 np.clip(yy - y0_ + 0.5, 0, 1) * np.clip(y1_ - yy + 0.5, 0, 1))
            rgb[:] = rgb * (1 - m[..., None]) + col * m[..., None]
            alpha[:] = alpha + m * (1 - alpha)
        if kind == 2:                                   # a dome on a drum, with its lantern
            rect(left, left + wpx, base - hpx * 0.45, base, wall_l)
            m = blob(xx, yy, left + wpx / 2, base - hpx * 0.45, wpx * 0.46, hpx * 0.45) * (yy < base - hpx * 0.45)
            shade = np.clip(0.6 - (xx - left - wpx / 2) / (wpx * 0.5), 0, 1)[..., None]
            col = roof * (0.5 + 0.7 * shade)
            rgb[:] = rgb * (1 - m[..., None]) + col * m[..., None]
            alpha[:] = alpha + m * (1 - alpha)
            rect(left + wpx * 0.45, left + wpx * 0.55, base - hpx * 1.0, base - hpx * 0.86, wall_l)
        else:
            rect(left, left + wpx, base - hpx, base, wall_l)
            rect(left + wpx, left + wpx * 1.45, base - hpx + 1, base, wall_s)
            rh = hpx * (0.5 if kind == 1 else 0.28)
            m = np.clip(1 - np.abs(xx - (left + wpx * 0.7)) / (wpx * 0.78) - (base - hpx - yy) / rh, 0, 1)
            m = np.clip(m * 4, 0, 1) * (yy < base - hpx + 1)
            rgb[:] = rgb * (1 - m[..., None]) + roof * m[..., None]
            alpha[:] = alpha + m * (1 - alpha)
            if kind == 1:
                rect(left + wpx * 0.25, left + wpx * 0.75, base - hpx * 0.8, base - hpx * 0.68,
                     air(np.array([0.02, 0.02, 0.02]), d, y, az_b))
        composite(img, dist, int(x - left), int(yb - base), rgb, alpha, d - 0.01)
    # --- the bridge ------------------------------------------------------
    bz = BRIDGE_Z
    bx = float(river_x(np.array([bz]))[0])
    az_b, el_b, d = project(bx, bz, 2.0)
    _, el_t, _ = project(bx, bz, 26.0)
    hpx = (el_t - el_b) * ppd
    wpx = math.degrees(0.16 / d) * ppd
    if hpx >= 1:
        W_, H_ = int(wpx) + 4, int(hpx) + 4
        yy, xx = np.mgrid[0:H_, 0:W_].astype(np.float32)
        stone = air(np.array([0.50, 0.44, 0.36]) * (sun_c * 0.45 + amb), d, 20, az_b)
        dark = air(np.array([0.02, 0.02, 0.025]), d, 20, az_b)
        alpha = np.clip(xx - 1.5, 0, 1) * np.clip(W_ - 2.5 - xx, 0, 1) * np.clip(yy - 1.5, 0, 1)
        rgb = np.broadcast_to(stone, (H_, W_, 3)).copy()
        n = 4
        for k in range(n):
            cx = 2 + wpx * (k + 0.5) / n
            m = blob(xx, yy, cx, H_ - 1.0, wpx / n * 0.36, hpx * 0.6)
            rgb = rgb * (1 - m[..., None]) + dark * m[..., None]
        composite(img, dist, int((az_b - az0) * ppd - W_ / 2), int((el1 - el_b) * ppd - H_ + 2), rgb, alpha, d - 0.01)
