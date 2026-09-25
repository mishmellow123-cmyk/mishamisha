"""Shot 1 (1440-1519) FAR PEAK.

One idea: a watcher sees the first light flare on the far horizon, and answers.
A shepherd stands with his back to us on a snowy summit above a moonlit sea of cloud and peaks,
a lit torch held low in front of him haloing his silhouette. Far on the horizon a pin-prick of
fire flares (the first beacon). He turns, raises the torch and thrusts it into his own beacon:
ignition on the hit at 1480, firelight floods the foreground, sparks and smoke stream downwind.
Camera: slow push toward the horizon light; kick + tilt up with the flames.
"""
import math

import numpy as np
from numba import njit, prange

import common as CM
from mt import terrain as T, sky as SK, fire as F, figure as FG, props as PR, people as PP
from mt.cam import Cam, keys, smooth, smoother, ease_out, kick, ramp
from mt.noise import fbm2, ridged2, gnoise2, smoothstep

START, END, IGN = 1440, 1519, 1480
FIRST = 1447                       # the far light appears
R_EARTH = 6.371e6

# first-beacon massif: placed so its summit light lands at screen (LIGHT_SX, ~LIGHT_SY) of the
# opening camera (full-res pixels), BEACON_D metres away
LIGHT_SX, LIGHT_SY, BEACON_D = 430.0, 297.0, 32000.0
HFOV = 36.0
F_FULL = 960.0 / math.tan(math.radians(HFOV / 2))
HORIZON_Y0 = 300.0
CAM0 = np.array([0.35, 2.50, 0.0])
_bx = (LIGHT_SX - 960.0) / F_FULL
XB = CAM0[0] + BEACON_D * _bx / math.sqrt(1 + _bx * _bx)
ZB = CAM0[2] + BEACON_D / math.sqrt(1 + _bx * _bx)
HB = CAM0[1] + BEACON_D * (HORIZON_Y0 - LIGHT_SY) / F_FULL + BEACON_D ** 2 / (2 * 6.371e6)

FEET = np.array([-1.1, 0.0, 16.0])
CAIRN = np.array([0.35, 0.0, 16.6])
WIND = 1.0                         # wind blows toward +x

MOON_DIR = np.array([-0.80, 0.36, 0.48])
MOON_DIR = MOON_DIR / np.linalg.norm(MOON_DIR)
CLOUD_Y = -650.0


@njit(inline='always', fastmath=True)
def _lod(scale, fp, lo, hi):
    o = math.log2(max(scale / max(fp * 2.0, 1e-4), 1.0))
    return min(max(o, lo), hi)


@njit(fastmath=True)
def h_near(x, z, fp):
    """Summit: a rocky, snow-dusted shoulder ending in a lip ~2.5 m beyond the figure."""
    lip = 18.7 + 1.3 * gnoise2(x * 0.09, 0.37, 11) + 0.02 * x * x * 0.1
    dz = z - lip
    h = 0.0
    if dz > 0.0:
        h -= 0.6 * dz * dz + 2.2 * dz
    ax = abs(x + 0.4)
    if ax > 6.0:
        h -= 0.06 * (ax - 6.0) ** 2
    ddx = x + 0.4
    ddz = z - 16.3
    dstand = math.sqrt(ddx * ddx * 0.45 + ddz * ddz)
    amp = 0.10 + 0.8 * smoothstep(2.0, 6.0, dstand)
    o = _lod(5.0, fp, 1.0, 7.0)
    r = ridged2(x * 0.25 + 3.1, z * 0.25 - 1.7, o, 5)
    h += (r - 0.45) * amp * 1.3
    o2 = _lod(1.2, fp, 1.0, 6.0)
    h += 0.05 * fbm2(x * 0.6, z * 1.0, o2, 9)
    h += 0.2 * math.exp(-((z - lip + 1.0) / 1.3) ** 2)
    for bx, bz, br, bh in ((-4.6, 15.2, 1.5, 1.0), (5.0, 16.4, 1.3, 0.9), (-2.9, 18.4, 0.8, 0.45),
                           (3.0, 18.9, 0.7, 0.35), (-7.0, 17.0, 2.0, 1.4)):
        qx = x - bx
        qz = z - bz
        q = (qx * qx + qz * qz) / (br * br)
        if q < 1.0:
            hb = bh * math.sqrt(1.0 - q) * (1.0 + 0.3 * gnoise2(x * 1.7, z * 1.7, 13)) + h * 0.3
            if hb > h:
                h = hb
    return h


@njit(fastmath=True)
def h_far(x, z, fp):
    s = 3400.0
    o = _lod(s, fp, 1.0, 11.0)
    # domain warp so ridges wander, and a massif envelope so peaks cluster with open cloud between
    wx = fbm2(x / 7000.0 + 5.1, z / 7000.0, 3.0, 61)
    wz = fbm2(x / 7000.0 - 2.3, z / 7000.0 + 1.9, 3.0, 62)
    r = ridged2(x / s + 0.31 + 0.55 * wx, z / s - 1.73 + 0.55 * wz, o, 21)
    m = fbm2(x / 11000.0 + 0.7, z / 11000.0 - 0.4, 3.0, 63)
    env = smoothstep(-0.35, 0.35, m)
    az = math.atan2(x - CAM0[0], z - CAM0[2])
    dd = math.sqrt((x - CAM0[0]) ** 2 + (z - CAM0[2]) ** 2)
    corridor = math.exp(-((az + 0.045) / 0.075) ** 2) * (1.0 - smoothstep(16000.0, 26000.0, dd))
    prom = (1.0 - 0.55 * corridor) * (0.55 + 0.45 * env)
    h = -1800.0 + 2100.0 * prom * r ** 1.45
    dxb = x - XB
    dzb = z - ZB
    h += (HB + 450.0) * math.exp(-(dxb * dxb + dzb * dzb) / (2.0 * 2600.0 * 2600.0))
    return h


@njit(fastmath=True)
def h_cloud(x, z, fp, t):
    o = _lod(2200.0, fp, 1.0, 6.0)
    n = fbm2(x / 2400.0 + t * 0.003, z / 2400.0, o, 33)
    o2 = _lod(500.0, fp, 1.0, 5.0)
    m = fbm2(x / 520.0 - t * 0.006, z / 520.0, o2, 34)
    o3 = _lod(120.0, fp, 0.0, 4.0)
    q = fbm2(x / 120.0, z / 120.0 + t * 0.01, o3, 35)
    return CLOUD_Y + 150.0 * n + 55.0 * (1.0 - abs(m)) + 10.0 * q


@njit(fastmath=True)
def hfun(x, z, fp, P):
    dx = x - P[0]
    dz = z - P[1]
    d2 = dx * dx + dz * dz
    curv = d2 / (2.0 * R_EARTH)
    h = h_near(x, z, fp) if d2 < 250.0 * 250.0 else -1e5
    if d2 > 40.0 * 40.0:
        hf = h_far(x, z, fp)
        if hf > h:
            h = hf
        hc = h_cloud(x, z, fp, P[2])
        if hc > h:
            h = hc
    return h - curv


@njit(fastmath=True)
def hshadow(x, z, fp, P):
    return hfun(x, z, fp, P)


@njit(parallel=True, fastmath=True)
def shade(C, D, P, S, LT, Lm, Im, amb, fogp, out, dep):
    H, W = D.shape
    mx, my, mz = Lm[0], Lm[1], Lm[2]
    for j in prange(H):
        for i in range(W):
            dx, dy, dz, sl, dxh, dzh = T.ray_of(C, i, j)
            d = D[j, i]
            if d >= 1e29:
                r, g, b = SK.sky_base(dx, dy, dz, S)
                out[j, i, 0] = r
                out[j, i, 1] = g
                out[j, i, 2] = b
                dep[j, i] = 1e9
                continue
            x = C[0] + dxh * d
            z = C[2] + dzh * d
            y = C[1] + sl * d
            dist = d * math.sqrt(1.0 + sl * sl)
            fp = dist / C[7]
            e = max(fp * 0.8, 0.004)
            ddx = x - P[0]
            ddz = z - P[1]
            curv = (ddx * ddx + ddz * ddz) / (2.0 * R_EARTH)
            hn = h_near(x, z, fp) if dist < 260.0 else -1e5
            hf = h_far(x, z, fp) if dist > 35.0 else -1e5
            hc = h_cloud(x, z, fp, P[2]) if dist > 35.0 else -1e5
            surf = 0
            if hf > hn:
                surf = 1
            if hc > max(hn, hf):
                surf = 2
            # normal from the winning surface
            if surf == 0:
                h0 = hn
                hx = h_near(x + e, z, fp)
                hz = h_near(x, z + e, fp)
            elif surf == 1:
                h0 = hf
                hx = h_far(x + e, z, fp)
                hz = h_far(x, z + e, fp)
            else:
                h0 = hc
                hx = h_cloud(x + e, z, fp, P[2])
                hz = h_cloud(x, z + e, fp, P[2])
            nx = -(hx - h0)
            ny = e
            nz = -(hz - h0)
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= inv
            ny *= inv
            nz *= inv
            ndl = nx * mx + ny * my + nz * mz
            if surf == 2:
                # cloud sea: soft wrap lighting + forward scatter toward the moon
                cosv = -(dx * mx + dy * my + dz * mz)
                fwd = 1.0 + 1.6 * max(-cosv, 0.0) ** 4
                wrap = max((ndl + 0.8) / 1.8, 0.0)
                cr = 0.9 * (Im * Lm[3] * wrap * fwd + amb[0] * 1.5)
                cg = 0.9 * (Im * Lm[4] * wrap * fwd + amb[1] * 1.5)
                cb = 0.9 * (Im * Lm[5] * wrap * fwd + amb[2] * 1.5)
                # peaks cast long moon shadows across the cloud sea
                shc = T.soft_shadow(hshadow, P, x, h0 - curv + 5.0, z, mx, my, mz, 40.0, 12000.0, 16, 6.0, fp)
                shc = 0.35 + 0.65 * shc
                cr *= shc
                cg *= shc
                cb *= shc
                # darker in the troughs, brighter billow tops
                trough = smoothstep(CLOUD_Y - 120.0, CLOUD_Y + 110.0, h0)
                cr *= 0.45 + 0.55 * trough
                cg *= 0.45 + 0.55 * trough
                cb *= 0.45 + 0.55 * trough
            else:
                # snow / rock
                if surf == 0:
                    sn = gnoise2(x * 0.8, z * 0.8, 71) * 0.12 + gnoise2(x * 3.1, z * 3.1, 72) * 0.05
                    snow = smoothstep(0.45, 0.70, ny + sn)
                else:
                    es = max(fp * 6.0, 45.0)
                    hsx = h_far(x + es, z, fp * 4.0)
                    hsz = h_far(x, z + es, fp * 4.0)
                    nsx = -(hsx - h0)
                    nsz = -(hsz - h0)
                    nsy = 1.0 / math.sqrt(nsx * nsx / (es * es) + nsz * nsz / (es * es) + 1.0)
                    sn = gnoise2(x / 380.0, z / 380.0, 73) * 0.18 + gnoise2(x / 95.0, z / 95.0, 74) * 0.06
                    snow = smoothstep(0.38, 0.58, nsy + sn)
                ar = 0.055 + (0.80 - 0.055) * snow
                ag = 0.056 + (0.86 - 0.056) * snow
                ab = 0.062 + (0.98 - 0.062) * snow
                shd = 1.0
                if ndl > 0.0:
                    if surf == 0:
                        shd = T.soft_shadow(hshadow, P, x, h0 - curv + 0.03, z, mx, my, mz, 0.08, 60.0, 18, 10.0, fp)
                    else:
                        shd = T.soft_shadow(hshadow, P, x, h0 - curv + 2.0, z, mx, my, mz, 25.0, 9000.0, 16, 12.0, fp)
                dif = max(ndl, 0.0) * shd
                skyl = 0.55 + 0.45 * ny
                cr = ar * (Im * Lm[3] * dif + amb[0] * skyl)
                cg = ag * (Im * Lm[4] * dif + amb[1] * skyl)
                cb = ab * (Im * Lm[5] * dif + amb[2] * skyl)
                # snow sheen toward the moon (backlit crest glints)
                if snow > 0.0:
                    hx_ = mx - dx
                    hy_ = my - dy
                    hz_ = mz - dz
                    hl = math.sqrt(hx_ * hx_ + hy_ * hy_ + hz_ * hz_) + 1e-9
                    nh = max((nx * hx_ + ny * hy_ + nz * hz_) / hl, 0.0)
                    sp = snow * 0.25 * nh ** 18 * shd
                    cr += sp * Im * Lm[3]
                    cg += sp * Im * Lm[4]
                    cb += sp * Im * Lm[5]
                # fire light (point lights, inverse square, no shadows)
                for li in range(LT.shape[0]):
                    lx = LT[li, 0] - x
                    ly = LT[li, 1] - (h0 - curv)
                    lz = LT[li, 2] - z
                    l2 = lx * lx + ly * ly + lz * lz
                    ll = math.sqrt(l2) + 1e-9
                    ndf = (nx * lx + ny * ly + nz * lz) / ll
                    if ndf > 0.0:
                        E = LT[li, 6] * ndf / (l2 + LT[li, 7] * LT[li, 7])
                        cr += ar * E * LT[li, 3]
                        cg += ag * E * LT[li, 4]
                        cb += ab * E * LT[li, 5]
            # atmosphere: aerial perspective + low mist over the cloud sea
            yc = C[1]
            yh = h0 - curv
            tau = T.height_fog_tau(dist, yc, yh, fogp[0], fogp[1])
            tau += T.height_fog_tau(dist, yc - CLOUD_Y, yh - CLOUD_Y, fogp[2], fogp[3])
            tr = math.exp(-tau)
            cosv = dx * mx + dy * my + dz * mz
            ph = 1.0 + fogp[4] * max(cosv, 0.0) ** 4
            fr = fogp[5] * ph
            fg = fogp[6] * ph
            fb = fogp[7] * ph
            out[j, i, 0] = cr * tr + fr * (1.0 - tr)
            out[j, i, 1] = cg * tr + fg * (1.0 - tr)
            out[j, i, 2] = cb * tr + fb * (1.0 - tr)
            dep[j, i] = dist


def _beacon_summit():
    # summit of the first-beacon massif (search a small neighbourhood for the max)
    best = (-1e9, XB, ZB)
    for dx in np.linspace(-1500, 1500, 31):
        for dz in np.linspace(-1500, 1500, 31):
            h = h_far(XB + dx, ZB + dz, 5.0)
            if h > best[0]:
                best = (h, XB + dx, ZB + dz)
    return np.array([best[1], best[0] + 4.0, best[2]])


_SUMMIT = None


def summit():
    global _SUMMIT
    if _SUMMIT is None:
        _SUMMIT = _beacon_summit()
    return _SUMMIT


# ------------------------------------------------------------------ timeline ---

def camera(frame, W=1920, H=804):
    t = CM.ftime(frame)
    ti = CM.ftime(IGN)
    tilt0 = math.degrees(math.atan((HORIZON_Y0 - 402.0) / F_FULL))
    push = keys(frame, [(START, 0.0), (IGN - 2, 1.15), (END, 0.9)], ease=smoother)
    tilt = keys(frame, [(START, tilt0), (IGN, tilt0), (IGN + 30, tilt0 + 2.2), (END, tilt0 + 2.5)], ease=smoother)
    k = kick(t, ti, amp=1.0, freq=6.0, decay=6.0)
    yaw = 0.06 * k
    pos = (CAM0[0], CAM0[1] + 0.012 * k, CAM0[2] + push)
    return Cam(pos, yaw, tilt + 0.10 * k, HFOV, W, H)


def pose(frame):
    f = frame
    turn = keys(f, [(START, 0.0), (1459, 0.0), (1471, 1.0)], ease=smoother)
    arm = keys(f, [(START, 0.0), (1460, 0.0), (1471, 0.5), (1478.5, 1.0), (1484, 0.78), (1500, 0.35),
                   (END, 0.3)], ease=smoother)
    lean = keys(f, [(START, 0.0), (1452, 0.0), (1458, 0.02), (1471, 0.05), (1478.5, 0.20), (1484, -0.05),
                    (1495, 0.02), (END, 0.02)], ease=smoother)
    head_up = keys(f, [(START, 0.0), (1450, 0.0), (1456, 1.0), (1466, 0.6), (END, 0.3)], ease=smoother)
    step = keys(f, [(START, 0.0), (1470, 0.0), (1478, 1.0), (1486, 0.3), (END, 0.3)], ease=smoother)
    recoil = keys(f, [(START, 0.0), (1480, 0.0), (1483, 1.0), (1492, 0.2), (END, 0.0)], ease=smoother)
    return dict(turn=turn, arm=arm, lean=lean, head_up=head_up, step=step, recoil=recoil)


def wind_fn(t, y):
    g = 0.6 + 0.4 * math.sin(t * 1.3) * math.sin(t * 0.7 + 1.0)
    return (WIND * 3.2 * g + 0.8 * math.sin(t * 5.1 + y * 3.0), 0.4 * math.sin(t * 4.3 + y * 2.0))


def cloth(frame):
    """Coat-hem and shawl-fringe chains simulated from the shot start (figure-local)."""
    t0 = CM.ftime(START - 24)
    t1 = CM.ftime(frame)

    def hem_anchor(tt):
        p = pose(tt * CM.FPS)
        T_ = p['turn']
        return (-0.26 * (1 - T_) - 0.18 * T_, 0.52)

    def shawl_anchor(tt):
        p = pose(tt * CM.FPS)
        T_ = p['turn']
        return (-0.28 * (1 - T_) - 0.12 * T_, 1.18)

    hem = FG.verlet_chain(hem_anchor, 5, 0.07, t0, t1, wind_fn, init_dir=(-0.3, -1.0))
    sh = FG.verlet_chain(shawl_anchor, 4, 0.06, t0, t1, wind_fn, init_dir=(-0.2, -1.0))
    return hem, sh


def render(frame, scale=0.5, ss=1.5):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    t = CM.ftime(frame)
    cam = camera(frame, W, H)
    cami = cam.scaled(ss)
    C = cami.params()
    P = np.array([cam.pos[0], cam.pos[2], t, 0.0])
    D = np.zeros((cami.H, cami.W))
    T.march(hfun, P, C, 0.3, 90000.0, 0.0035, 0.35, 700.0, 9, D)

    # --- fire state
    ti = CM.ftime(IGN)
    size, inten, light = F.ignite_env(t, ti)
    fl = F.flicker(t, 3)
    back, front, fb, rb = PR.cairn(seed=7)
    fire_base = CAIRN + np.array([0.0, fb, 0.0])
    fire_I = 7.5 * light * fl
    p = pose(frame)
    hem, sh = cloth(frame)
    fig, pts = PP.shepherd(coat_pts=hem, shawl_pts=sh, **p)
    zt = -0.28 * (1 - p['turn']) - 0.05
    torch_w = PR.local_to_world(cam, FEET, pts['torch_head'][0], pts['torch_head'][1], zt)
    torch_I = 0.55 * F.flicker(t, 7) * (1.0 - 0.6 * min(1.0, light))

    LT = []
    LT.append([torch_w[0], torch_w[1] + 0.15, torch_w[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2],
               torch_I, 0.25])
    if fire_I > 0:
        LT.append([fire_base[0], fire_base[1] + 0.7, fire_base[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1],
                   F.FIRE_LIGHT[2], fire_I, 0.5])
    LT = np.array(LT, np.float64)

    moon_col = CM.lin('#9DB4D9')
    Im = 0.55
    Lm = np.r_[MOON_DIR, moon_col]
    amb = CM.lin('#27335E') * 0.35
    S = SK.sky_params(zenith='#070B1C', horizon='#2A3866', moon_dir=MOON_DIR, moon_radius_deg=0.8,
                      halo_I=0.025, halo_w=0.22, halo2_I=0.012, halo2_w=0.7, horizon_glow=0.25, gain=1.0)
    fogc = CM.lin('#2E3D66') * 0.95
    fogp = np.array([5.0e-5, 1 / 1500.0, 2.2e-4, 1 / 140.0, 1.5, fogc[0], fogc[1], fogc[2]])
    out = np.zeros((cami.H, cami.W, 3), np.float32)
    dep = np.zeros((cami.H, cami.W), np.float32)
    shade(C, D, P, S, LT, Lm, Im, amb, fogp, out, dep)
    img = CM.downsample(out, W, H)
    depth = CM.depth_down(dep, W, H)
    skymask = (CM.downsample((dep > 1e8).astype(np.float32), W, H))

    # --- stars
    stars = _stars()
    SK.splat_stars(img, cam, stars, skymask, t=t, gain=1.0, scale=scale)

    # --- the first beacon on the horizon
    if frame >= FIRST - 1:
        u = frame - FIRST
        flare = 1.0 - math.exp(-max(u + 1, 0) / 1.2)
        over = 1.8 * math.exp(-((u - 2.0) / 2.0) ** 2)
        e = (flare + over) * F.flicker(t, 11, 1.4)
        sx, sy, z = cam.project(summit())
        s1 = scale
        F.glow(img, depth, sx, sy, 0.9 * s1 + 0.45, 14.0 * e * s1 * s1, z=z, zbias=500.0)
        F.halo(img, depth, sx, sy, 14 * s1, 0.03 * e, z=z, zbias=500.0)
        F.halo(img, depth, sx, sy, 60 * s1, 0.006 * e, z=z, zbias=500.0)

    # --- torch glow in the air (behind the figure; the body occludes its centre)
    sx, sy, z = cam.project(torch_w)
    F.halo(img, depth, sx, sy, 60 * scale, 0.05 * torch_I, z=z, zbias=0.5)

    moon_light = dict(dir=MOON_DIR, col=moon_col, I=Im * 0.7)
    lights = [moon_light, dict(pos=torch_w, col=F.FIRE_LIGHT, I=torch_I * 0.8, r0=0.12)]
    if fire_I > 0:
        lights.append(dict(pos=fire_base + np.array([0, 0.6, 0]), col=F.FIRE_LIGHT, I=fire_I * 0.8, r0=0.3))
    famb = amb * 1.2

    # beacon: stones + logs, flame, basket bars
    FG.render(img, depth, cam, back, CAIRN, lights, amb=famb, t=t, emissive_gain=min(light, 1.5) * 1.2,
              write_depth=True)
    # figure
    FG.render(img, depth, cam, fig, FEET, lights, amb=famb, t=t, seed=5)
    # torch flame (occluded by the body while he faces away)
    F.flame(img, depth, cam, torch_w + np.array([0.0, 0.01, 0.0]), 0.34, 0.055, t, seed=17, I=16.0, tongues=3,
            lean=0.08 * WIND, zbias=0.05)
    if size > 0:
        # smoke, heat shimmer, flame
        sm = _smoke()
        sm.render(img, depth, cam, t, fire_base + np.array([0, 0.9, 0]), 0.55 * light * fl, albedo=0.22,
                  amb=(0.004, 0.005, 0.009))
        F.shimmer(img, cam, fire_base, 2.2 * size, rb, t, amp_px=1.4 * scale)
        F.flame(img, depth, cam, fire_base, 2.25 * size, rb, t, seed=2, I=26.0 * inten, lean=0.35 * WIND * size,
                zbias=0.4)
        sx, sy, z = cam.project(fire_base + np.array([0, 0.9, 0]))
        F.halo(img, depth, sx, sy, 260 * scale * size, 0.010 * light * fl, z=z, zbias=3.0)
    FG.render(img, depth, cam, front, CAIRN, lights, amb=famb, t=t, write_depth=False)
    if size > 0:
        _sparks().render(img, depth, cam, t)
    return img


_ST = None
_SM = None
_SP = None


def _stars():
    global _ST
    if _ST is None:
        _ST = SK.make_stars(14000, 101, lum_scale=7.0)
    return _ST


def _smoke():
    global _SM
    if _SM is None:
        _SM = F.Smoke(41, CAIRN + np.array([0, 2.3, 0]), CM.ftime(IGN) + 0.1, CM.ftime(END) + 0.1, rate=5.5,
                      wind=(1.6 * WIND, 0, 0.2), rise=1.4, life=5.0, r0=0.35, growth=0.5, dens=0.9)
    return _SM


def _sparks():
    global _SP
    if _SP is None:
        _SP = F.Sparks(43, CAIRN + np.array([0, 1.5, 0]), CM.ftime(IGN), CM.ftime(END) + 0.1, burst=320, rate=60,
                       ember_rate=14, wind=(2.2 * WIND, 0.0, 0.3), I=30.0)
    return _SP


FINISH = dict(exposure=1.0, bloom_strength=0.07, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)
