"""DAWN in the east (cut C, v2 frames 2400-2655).

The same world the morning after, from where THE BEACON RUN ended: the camera floats over the shepherd's
foothills looking out across the islands of the cloud sea toward the eastern ranges. At 2400 the sun breaks
over the far skyline (upper left) -- a clean white-gold disc, no starburst -- and as it climbs ~1.5 deg the
long shadows of the ranges sweep back across the cloud sea toward us: the light floods in. Low raking light
on the spires (gold faces, blue shadows), thin smoke still rising from every summit that held a beacon, and
late in the shot a few great eagles glide away into the light. The camera rises slowly; the lower third
stays calm for cut C's line (2512-2600); it ends still for the dissolve (2624-2655).
"""
import math
import os
import sys

import cv2
import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402

from mt.noise import fbm2, fbm3, gnoise2, smoothstep   # noqa: E402

START, END = 2400, 2655
FPS = 24.0
CLOUD_Y = WD.CLOUD_Y
R_EARTH = WD.R_EARTH

# ------------------------------------------------------------------ camera ---
HFOV = 46.0
YAW = -19.0                                  # looking where the Run's chain raced to
CAM0 = np.array([60.0, 150.0, 480.0])        # 2400
RISE = 20.0                                  # metres over the shot
DRIFT = 22.0                                 # metres forward over the shot
PITCH = [(2400, -2.6), (2655, -3.4)]


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def camera(frame, W=1920, H=804):
    u = (frame - START) / (END - START)
    # a slow crane: steady rise, easing out into stillness at the very end
    r = 0.85 * u + 0.15 * (1 - (1 - u) ** 2)
    yaw = math.radians(YAW)
    fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    pos = CAM0 + np.array([0.0, RISE * r, 0.0]) + fwd * DRIFT * r
    pitch = PITCH[0][1] + (PITCH[1][1] - PITCH[0][1]) * _ease(u)
    return RC.RCam(pos, YAW, pitch, 0.0, HFOV, W, H)


# ------------------------------------------------------------------ sun ---
SUN_AZ = -30.0                               # deg (world yaw of the sun)
SUN_EL0 = None                               # skyline elevation at the sun's azimuth (computed)


def skyline_el(az_deg, cam_pos):
    """Elevation (deg) of the terrain skyline seen from cam_pos toward azimuth az (incl. curvature)."""
    a = math.radians(az_deg)
    d = np.array([math.sin(a), math.cos(a)])
    best = -10.0
    CR = _CR()
    for r in np.geomspace(200.0, 90000.0, 900):
        x, z = cam_pos[0] + d[0] * r, cam_pos[2] + d[1] * r
        h = max(WD.ground(x, z, CR, r / 2000.0), WD.h_cloud(x, z, r / 2000.0, 100.0)) - r * r / (2 * R_EARTH)
        best = max(best, math.degrees(math.atan2(h - cam_pos[1], r)))
    return best


def sun_elev(frame):
    """Top limb touches the skyline at 2399.5; the disc clears it by ~2432; then a slow climb."""
    global SUN_EL0
    if SUN_EL0 is None:
        SUN_EL0 = skyline_el(SUN_AZ, camera(START).pos)
    rad = 0.27
    u = (frame - 2399.5) / FPS                   # seconds after first light
    # fast at first (the break), easing to a slow climb
    e = SUN_EL0 - rad + 0.62 * (1 - math.exp(-max(u, -2.0) / 1.3)) * 1.0 + 0.085 * u
    return e


def sun_dir(frame):
    e = math.radians(sun_elev(frame))
    a = math.radians(SUN_AZ)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


def light_dir(frame):
    """Direction the terrain is lit from: the disc's direction, lifted ~2.3 deg over 2400-2470 so the long
    shadows of the eastern ranges visibly sweep back across the cloud sea (the flood) while the disc itself
    climbs slowly (no time-lapse look)."""
    u = min(max((frame - 2398.0) / 76.0, 0.0), 1.0)
    lift = 2.3 * (u * u * (3 - 2 * u))
    e = math.radians(sun_elev(frame) + lift)
    a = math.radians(SUN_AZ)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


def light_gain(frame):
    """The flood: sunlight on the land comes up as the disc clears the skyline."""
    u = min(max((frame - 2397.0) / 28.0, 0.0), 1.0)
    return 0.30 + 0.70 * (u * u * (3 - 2 * u))


def sun_col(frame):
    """Low sun: orange-gold at first light, warming toward gold-white as it clears the haze."""
    e = max(sun_elev(frame) - (SUN_EL0 or 0.0), 0.0)
    w = min(e / 1.4, 1.0)
    c0 = np.array([1.0, 0.44, 0.19])
    c1 = np.array([1.0, 0.62, 0.34])
    return c0 * (1 - w) + c1 * w


# ------------------------------------------------------------------ the set ---
_CRC = None


def _CR():
    global _CRC
    if _CRC is None:
        import run as RN                 # same world as the Run (foothills + the pyre tower)
        _CRC = RN.CR
    return _CRC


# ------------------------------------------------------------------ sky ---
# SD: 0-2 sun dir | 3-5 sun col | 6 sun radiance | 7 disc radius (rad) | 8-10 zenith | 11-13 horizon warm
# 14-16 horizon cool | 17 aureole A | 18 aureole B | 19 horizon band width | 20 warm band azimuth width


@njit(inline='always', fastmath=True)
def dawn_sky(dx, dy, dz, SD, pix_ang):
    sx, sy, sz = SD[0], SD[1], SD[2]
    el = math.asin(min(max(dy, -1.0), 1.0))
    hl = math.sqrt(dx * dx + dz * dz) + 1e-9
    shl = math.sqrt(sx * sx + sz * sz) + 1e-9
    caz = (dx * sx + dz * sz) / (hl * shl)
    daz = math.acos(min(max(caz, -1.0), 1.0))
    warm = math.exp(-(daz / SD[20]) ** 2)
    # horizon colour: gold toward the sun, pale peach -> lilac-blue away from it
    hr = SD[14] + (SD[11] - SD[14]) * warm
    hg = SD[15] + (SD[12] - SD[15]) * warm
    hb = SD[16] + (SD[13] - SD[16]) * warm
    # clear air: blue within a few degrees of the horizon (a little slower near the sun)
    u = smoothstep(0.0, 0.10 + 0.08 * warm, max(el, 0.0)) ** 0.75
    r = hr + (SD[8] - hr) * u
    g = hg + (SD[9] - hg) * u
    b = hb + (SD[10] - hb) * u
    # a bright gold band hugging the skyline toward the sun
    band = math.exp(-max(el, 0.0) / SD[19]) * warm
    r += 0.9 * band * SD[11]
    g += 0.9 * band * SD[12]
    b += 0.9 * band * SD[13]
    # aureole (forward scattering) around the sun
    cosg = dx * sx + dy * sy + dz * sz
    ang = math.acos(min(max(cosg, -1.0), 1.0))
    au = SD[17] * math.exp(-ang / 0.045) + SD[18] * math.exp(-ang / 0.28)
    r += au * SD[3]
    g += au * SD[4]
    b += au * SD[5]
    # the disc (limb darkened, anti-aliased)
    R = SD[7]
    if ang < R + 2.0 * pix_ang:
        cov = min(max((R - ang) / pix_ang + 0.5, 0.0), 1.0)
        rr = min((ang / R) ** 2, 1.0)
        limb = 0.62 + 0.38 * math.sqrt(1.0 - rr)
        I = SD[6] * cov * limb
        r += I * 1.0
        g += I * 0.93
        b += I * 0.80
    return r, g, b


# ------------------------------------------------------------------ shading ---
# QD: 0 sun intensity | 1 cloud fwd gain | 2 cloud fwd power | 3 cloud shadow floor | 4 cloud albedo
#     5 cloud ambient gain | 6 far shadow k | 7 far shadow tmax | 8 sheen | 9 bounce | 10 trough dark
#     11-13 warm fog colour | 14 warm fog power | 15 near shadow k


@njit(parallel=True, fastmath=True, cache=True)
def dawn_shade(C, D, P, CR, SD, Lk, QD, amb, fogp, out, zbuf, dist_out):
    H, W = D.shape
    mx, my, mz = Lk[0], Lk[1], Lk[2]
    Ik = QD[0]
    pix_ang = 1.0 / C[7]
    for j in prange(H):
        for i in range(W):
            fxx, fzz, rxx, rzz = C[3], C[4], C[5], C[6]
            f, cx, cyy = C[7], C[8], C[9]
            xo = i + 0.5 - cx
            dxh = fxx * f + rxx * xo
            dzh = fzz * f + rzz * xo
            hl = math.sqrt(dxh * dxh + dzh * dzh)
            vy = cyy - (j + 0.5)
            nn = math.sqrt(hl * hl + vy * vy)
            dx = dxh / nn
            dy = vy / nn
            dz = dzh / nn
            sl = vy / hl
            dxh /= hl
            dzh /= hl
            d = D[j, i]
            if d >= 1e29:
                r, g, b = dawn_sky(dx, dy, dz, SD, pix_ang)
                out[j, i, 0] = r
                out[j, i, 1] = g
                out[j, i, 2] = b
                zbuf[j, i] = 1e9
                dist_out[j, i] = 1e9
                continue
            x = C[0] + dxh * d
            z = C[2] + dzh * d
            dist = d * math.sqrt(1.0 + sl * sl)
            fp = dist / C[7]
            e = max(fp * 0.8, 0.003)
            ddx = x - P[0]
            ddz = z - P[1]
            curv = (ddx * ddx + ddz * ddz) / (2.0 * R_EARTH)
            hn = WD.h_near(x, z, fp)
            hf = WD.h_rock(x, z, fp, CR)
            hc = WD.h_cloud(x, z, fp, P[2])
            surf = 0
            if hf > hn:
                surf = 1
            if hc > max(hn, hf):
                surf = 2
            if surf == 0:
                h0 = hn
                hx = WD.h_near(x + e, z, fp)
                hz = WD.h_near(x, z + e, fp)
            elif surf == 1:
                h0 = hf
                hx = WD.h_rock(x + e, z, fp, CR)
                hz = WD.h_rock(x, z + e, fp, CR)
            else:
                h0 = hc
                hx = WD.h_cloud(x + e, z, fp, P[2])
                hz = WD.h_cloud(x, z + e, fp, P[2])
            nx = -(hx - h0)
            ny = e
            nz = -(hz - h0)
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= inv
            ny *= inv
            nz *= inv
            ndl = nx * mx + ny * my + nz * mz
            yw = h0 - curv
            if surf == 2:
                cosk = dx * mx + dy * my + dz * mz
                fwd = 1.0 + QD[1] * max(cosk, 0.0) ** QD[2]
                wrap = max((ndl + 0.8) / 1.8, 0.0)
                shc = WD.soft_shadow(P, CR, x, yw + 4.0, z, mx, my, mz, 30.0, QD[7], 22, QD[16], fp)
                lit = QD[3] + (1.0 - QD[3]) * shc
                ca = QD[4]
                cr = ca * (Ik * Lk[3] * wrap * fwd * lit + amb[0] * QD[5])
                cg = ca * (Ik * Lk[4] * wrap * fwd * lit + amb[1] * QD[5])
                cb = ca * (Ik * Lk[5] * wrap * fwd * lit + amb[2] * QD[5])
                trough = smoothstep(CLOUD_Y - 120.0, CLOUD_Y + 110.0, h0)
                tk = (1.0 - QD[10]) + QD[10] * trough
                cr *= tk
                cg *= tk
                cb *= tk
            else:
                es = max(fp * 6.0, 45.0)
                if surf == 0:
                    hsx = WD.h_near(x + es, z, fp * 4.0)
                    hsz = WD.h_near(x, z + es, fp * 4.0)
                else:
                    hsx = WD.h_rock(x + es, z, fp * 4.0, CR)
                    hsz = WD.h_rock(x, z + es, fp * 4.0, CR)
                nsx = -(hsx - h0) / es
                nsz = -(hsz - h0) / es
                nsy = 1.0 / math.sqrt(nsx * nsx + nsz * nsz + 1.0)
                if surf == 0:
                    sn = gnoise2(x * 0.8, z * 0.8, 71) * 0.12 + gnoise2(x * 3.1, z * 3.1, 72) * 0.05
                    snow = smoothstep(0.45, 0.70, ny + sn)
                else:
                    sn = gnoise2(x / 380.0, z / 380.0, 73) * 0.18 + gnoise2(x / 95.0, z / 95.0, 74) * 0.06
                    snow = smoothstep(0.38, 0.58, nsy + sn)
                    wf = smoothstep(3.0, 0.4, fp)
                    if wf > 0.0:
                        em = max(fp * 6.0, 3.0)
                        hmx = WD.h_rock(x + em, z, fp * 2.0, CR)
                        hmz = WD.h_rock(x, z + em, fp * 2.0, CR)
                        nmx = -(hmx - h0) / em
                        nmz = -(hmz - h0) / em
                        nmy = 1.0 / math.sqrt(nmx * nmx + nmz * nmz + 1.0)
                        sf = gnoise2(x / 9.0, z / 9.0, 75) * 0.06 + gnoise2(x * 0.9, z * 0.9, 76) * 0.035
                        sv = 0.42 * ny + 0.33 * nmy + 0.25 * nsy + sn * 0.6 + sf
                        snow = snow + (smoothstep(0.40, 0.60, sv) - snow) * wf
                rv = 1.0
                if fp < 20.0:
                    rv = 0.75 + 0.5 * (0.5 + 0.5 * gnoise2(x / 7.0 + h0 / 3.0, z / 7.0, 81))
                    rv *= 0.85 + 0.3 * (0.5 + 0.5 * gnoise2(h0 * 0.35, x / 40.0 + z / 40.0, 82))
                    rv = 1.0 + (rv - 1.0) * smoothstep(20.0, 4.0, fp)
                # dawn rock is warmer than the night's (granite catching gold light)
                ar = 0.070 * rv + (0.82 - 0.070 * rv) * snow
                ag = 0.062 * rv + (0.86 - 0.062 * rv) * snow
                ab = 0.058 * rv + (0.95 - 0.058 * rv) * snow
                shd = 0.0
                if ndl > 0.0:
                    t0 = max(fp * 1.5, 0.08)
                    shd = WD.soft_shadow(P, CR, x, yw + max(fp * 1.2, 0.03), z, mx, my, mz, t0, QD[7], 26,
                                         QD[15], fp)
                dif = max(ndl, 0.0) * shd
                skyl = 0.55 + 0.45 * ny
                bnc = QD[9] * Ik * (0.5 - 0.5 * ny) * (0.6 + 0.4 * max(-(nx * mx + nz * mz), 0.0))
                cr = ar * (Ik * Lk[3] * dif + amb[0] * skyl + Lk[3] * bnc)
                cg = ag * (Ik * Lk[4] * dif + amb[1] * skyl + Lk[4] * bnc)
                cb = ab * (Ik * Lk[5] * dif + amb[2] * skyl + Lk[5] * bnc)
                if snow > 0.0:
                    hx_ = mx - dx
                    hy_ = my - dy
                    hz_ = mz - dz
                    hl_ = math.sqrt(hx_ * hx_ + hy_ * hy_ + hz_ * hz_) + 1e-9
                    nh = max((nx * hx_ + ny * hy_ + nz * hz_) / hl_, 0.0)
                    sp = snow * QD[8] * nh ** 24 * shd
                    cr += sp * Ik * Lk[3]
                    cg += sp * Ik * Lk[4]
                    cb += sp * Ik * Lk[5]
            # atmosphere: warm toward the sun, cool elsewhere
            yc = C[1]
            tau = WD.height_fog_tau(dist, yc, yw, fogp[0], fogp[1])
            tau += WD.height_fog_tau(dist, yc - CLOUD_Y, yw - CLOUD_Y, fogp[2], fogp[3])
            tr = math.exp(-tau)
            cosv = dx * mx + dy * my + dz * mz
            wv = max(cosv, 0.0) ** QD[14]
            ph = 1.0 + fogp[4] * wv
            fr = (fogp[5] * (1.0 - wv) + QD[11] * wv) * ph
            fg = (fogp[6] * (1.0 - wv) + QD[12] * wv) * ph
            fb = (fogp[7] * (1.0 - wv) + QD[13] * wv) * ph
            out[j, i, 0] = cr * tr + fr * (1.0 - tr)
            out[j, i, 1] = cg * tr + fg * (1.0 - tr)
            out[j, i, 2] = cb * tr + fb * (1.0 - tr)
            zbuf[j, i] = d * (dxh * C[3] + dzh * C[4])
            dist_out[j, i] = dist


def dawn_light(frame):
    sd = sun_dir(frame)
    sc = sun_col(frame)
    lin = PI.CM.lin
    # key: the low sun, lit from light_dir (the flood), sky/sun disc at sun_dir
    Lk = np.r_[light_dir(frame), sc]
    amb = lin('#4F74C0') * 0.20                       # blue sky fill
    SD = np.zeros(24)
    SD[0:3] = sd
    SD[3:6] = sc
    SD[6] = 900.0
    SD[7] = math.radians(0.27)
    SD[8:11] = lin('#2F62B0') * 0.60                  # zenith: clear dawn blue
    SD[11:14] = lin('#FFC27E') * 0.95                 # horizon toward the sun: gold
    SD[14:17] = lin('#B9B3D6') * 0.40                 # horizon away: pale lilac-blue
    SD[17] = 1.1                                      # tight aureole
    SD[18] = 0.035                                    # broad aureole (kept faint: clear air)
    SD[19] = 0.030                                    # horizon band height (rad)
    SD[20] = 0.42                                     # warm band azimuth width (rad)
    fogc = lin('#7288BE') * 0.33                      # cool blue-violet air
    fogp = np.array([5.0e-5, 1 / 1500.0, 2.0e-4, 1 / 150.0, 1.2, fogc[0], fogc[1], fogc[2]])
    QD = np.zeros(20)
    QD[0] = 2.1 * light_gain(frame)
    QD[1] = 2.2
    QD[2] = 8.0
    QD[3] = 0.30
    QD[4] = 0.95
    QD[5] = 1.5
    QD[6] = 90.0
    QD[7] = 45000.0
    QD[8] = 0.35
    QD[9] = 0.10
    QD[10] = 0.45
    wf = lin('#FFC58A') * 0.75
    QD[11:14] = wf
    QD[14] = 40.0                                     # warm haze only close to the sun
    QD[15] = 40.0
    QD[16] = 22.0                                     # cloud shadow softness (k)
    return Lk, amb, SD, fogp, QD


def render_terrain(fr, frame):
    scam = fr.src
    C = scam.params()
    CR = _CR()
    P = np.array([scam.pos[0], scam.pos[2], frame / FPS, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(P, CR, C, 0.2, 90000.0, 0.0035, 0.35, 900.0, 9, D)
    Lk, amb, SD, fogp, QD = dawn_light(frame)
    out = np.zeros((scam.H, scam.W, 3), np.float32)
    zb = np.zeros((scam.H, scam.W), np.float32)
    di = np.zeros((scam.H, scam.W), np.float32)
    dawn_shade(C, D, P, CR, SD, Lk, QD, amb, fogp, out, zb, di)
    fr.img, fr.zb, fr.dist = out, zb, di
    return fr


# ------------------------------------------------------------------ frame ---
FINISH = dict(exposure=0.70, bloom_strength=0.07, bloom_threshold=1.5, streak_strength=0.004, vignette_amount=0.22)


def render(frame, scale=1.0, ss=1.5, extras=True):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = camera(frame, W, H)
    fr = PI.Frame(tcam, ss)
    render_terrain(fr, frame)
    if extras:
        draw_smoke(fr, frame)
    img, zb, di = PI.to_target(fr)
    if extras:
        draw_eagles(img, zb, tcam, frame)
    return img


# ------------------------------------------------------------------ smoke ---
# Thin smoke still rising from every summit that held a beacon: columns of soft puffs scrolling upward,
# leaning a little in a light breeze and flattening under an inversion; lit by the low sun (strong forward
# scattering when seen toward it) and the blue sky.
WIND = np.array([2.2, 0.0, -0.8])            # m/s, a light morning breeze


def _plume_sources():
    B = np.load(os.path.join(HERE, 'beacons.npy'))
    import run as RN
    src = [(b[0], b[1], b[2], b[4], int(b[5])) for b in B]
    src.append((RN.PYRE_XZ[0], RN.PYRE_XZ[1], RN.PYRE_XZ[2], 1, 90))
    return src


_PL = None


def _plumes():
    global _PL
    if _PL is None:
        rng = np.random.default_rng(31)
        rows = []
        for x, y, z, kind, k in _plume_sources():
            big = kind == 1
            Hp = (230.0 if big else 150.0) * (0.8 + 0.4 * rng.random())
            n = 70 if big else 48
            for i in range(n):
                s0 = (i + rng.random()) / n
                rows.append([x, y, z, Hp, s0, rng.normal() * 0.35, rng.normal() * 0.35, 1.0 if big else 0.7,
                             rng.integers(0, 999), rng.random()])
        _PL = np.array(rows)
    return _PL


def _puff_world(PLR, t):
    """Puff centres (N,3), radii (N,), densities (N,) at time t (s)."""
    x, y, z, Hp, s0, ox, oz, g, seed, ph = [PLR[:, k] for k in range(10)]
    s = np.mod(s0 + t * 1.6 / Hp, 1.0)                  # rising ~1.6 m/s, loops over the column height
    hgt = s * Hp
    top = np.clip((s - 0.72) / 0.28, 0.0, 1.0)          # the inversion: spreads and drifts at the top
    drift = hgt * 0.25 + top * Hp * 0.9
    wn = WIND / np.linalg.norm(WIND)
    cx = x + wn[0] * drift + ox * (2.0 + 0.10 * hgt)
    cz = z + wn[2] * drift + oz * (2.0 + 0.10 * hgt)
    cy = y + 2.0 + hgt - top * Hp * 0.12
    r = (1.5 + 0.09 * hgt + top * 14.0) * g
    dens = 0.16 * g * np.clip(s / 0.06, 0, 1) * (1.0 - s) ** 0.8 * (1.0 + 0.6 * top)
    return np.stack([cx, cy, cz], 1), r, dens


@njit(fastmath=True, cache=True)
def _splat_puffs(img, depth, sx, sy, z, rpx, a0, col, seeds, zbias, ang):
    H, W = img.shape[0], img.shape[1]
    for k in range(sx.shape[0]):
        R = rpx[k] * 1.6
        x0 = int(max(0, sx[k] - R))
        x1 = int(min(W, sx[k] + R + 1))
        y0 = int(max(0, sy[k] - R))
        y1 = int(min(H, sy[k] + R + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        sd = seeds[k]
        for py in range(y0, y1):
            for px in range(x0, x1):
                if depth[py, px] < z[k] - zbias[k]:
                    continue
                u = (px + 0.5 - sx[k]) / rpx[k]
                v = (py + 0.5 - sy[k]) / rpx[k]
                q = math.sqrt(u * u + v * v)
                if q > 1.6:
                    continue
                nb = fbm3(u * 1.4 + sd, v * 1.4, ang[k], 3.0, 17)
                a = a0[k] * smoothstep(1.05 + 0.35 * nb, 0.15, q)
                if a <= 1e-4:
                    continue
                img[py, px, 0] = img[py, px, 0] * (1.0 - a) + col[k, 0] * a
                img[py, px, 1] = img[py, px, 1] * (1.0 - a) + col[k, 1] * a
                img[py, px, 2] = img[py, px, 2] * (1.0 - a) + col[k, 2] * a


def draw_smoke(fr, frame):
    scam = fr.src
    t = frame / FPS
    P, r, dens = _puff_world(_plumes(), t)
    sx, sy, z = scam.project(P)
    ok = (z > 20.0) & (sx > -200) & (sx < scam.W + 200) & (sy > -200) & (sy < scam.H + 200)
    if not ok.any():
        return
    P, r, dens, sx, sy, z = P[ok], r[ok], dens[ok], sx[ok], sy[ok], z[ok]
    rpx = np.maximum(r * scam.f / z, 0.6)
    # thin: sub-pixel puffs fade instead of shrinking below a pixel
    a0 = dens * np.clip(r * scam.f / z / 0.6, 0.25, 1.0)
    Lk, amb, SD, fogp, QD = dawn_light(frame)
    sd = sun_dir(frame)
    sc = Lk[3:6]
    d = P - scam.pos
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    cosg = d @ sd
    phase = 0.35 + 3.2 * np.clip(cosg, 0, 1) ** 10 + 0.6 * np.clip(cosg, 0, 1) ** 2
    lit = np.clip((P[:, 1] - (CLOUD_Y + 40.0)) / 60.0, 0.15, 1.0)
    alb = 0.55
    col = alb * (QD[0] * sc[None, :] * (phase * lit)[:, None] * 0.55 + amb[None, :] * 1.3)
    # aerial perspective on the smoke itself
    dist = np.linalg.norm(P - scam.pos, axis=1)
    T = np.exp(-6.0e-5 * dist)[:, None]
    fogc = fogp[5:8]
    col = col * T + fogc[None, :] * (1 - T)
    a0 = a0 * (0.35 + 0.65 * T[:, 0])
    order = np.argsort(-z)
    zb = np.maximum(0.004 * z, 3.0)
    _splat_puffs(fr.img, fr.zb, sx[order].astype(np.float64), sy[order].astype(np.float64),
                 z[order].astype(np.float64), rpx[order].astype(np.float64), a0[order].astype(np.float64),
                 col[order].astype(np.float64), (_plumes()[ok][:, 8][order] * 0.37).astype(np.float64),
                 zb[order].astype(np.float64), (t * 0.25 + _plumes()[ok][:, 9][order] * 7.0).astype(np.float64))


# ------------------------------------------------------------------ eagles ---
# One loose, irregular pass of great eagles, late (from ~2474), small (wingspan <= ~4% of frame width), gliding
# away from the camera into the light, never across the sun disc (their vanishing point sits right of and
# above the sun). Mostly gliding; two of them give a couple of slow wingbeats as they settle.
_WING = np.array([  # right wing + half body/tail outline, bird-local metres: x right, y up, z forward
    [0.00, 0.02, 0.47], [0.04, 0.02, 0.40], [0.07, 0.01, 0.30], [0.10, 0.00, 0.16],
    [0.30, 0.02, 0.21], [0.50, 0.035, 0.225], [0.64, 0.05, 0.19], [0.74, 0.055, 0.13],
    [0.86, 0.065, 0.10], [1.10, 0.080, 0.04], [0.90, 0.066, 0.045],
    [1.08, 0.078, -0.03], [0.90, 0.066, -0.015],
    [1.03, 0.072, -0.10], [0.87, 0.062, -0.075],
    [0.96, 0.066, -0.16], [0.83, 0.058, -0.13],
    [0.88, 0.060, -0.21], [0.76, 0.052, -0.20],
    [0.56, 0.035, -0.255], [0.34, 0.015, -0.25], [0.14, 0.00, -0.20],
    [0.08, 0.00, -0.30], [0.17, 0.00, -0.50], [0.06, 0.00, -0.57], [0.00, 0.00, -0.58]])


def _outline(dihedral, span=2.1):
    R = _WING.copy()
    R[:, 0] *= span / 2.2
    L = R[::-1].copy()
    L[:, 0] *= -1
    pts = np.vstack([R, L[1:-1]])
    out = pts.copy()
    wing = np.abs(pts[:, 0]) > 0.09
    th = np.where(pts[:, 0] > 0, dihedral, -dihedral)
    c, s_ = np.cos(th), np.sin(th)
    x, y = pts[:, 0], pts[:, 1]
    out[wing, 0] = (x * c - y * s_)[wing]
    out[wing, 1] = (x * s_ + y * c)[wing]
    return out


# (start frame, bearing from view centre deg (+right), elevation deg, distance m, speed m/s, flaps, roll phase,
#  span m) -- each bird is placed where it enters the frame (just inside the right edge) at its start frame
BIRDS = [(2476, 21.5, 2.6, 76.0, 17.0, (2478, 2), 0.3, 2.15),
         (2484, 22.5, 4.2, 88.0, 16.2, None, 1.9, 2.05),
         (2492, 20.5, 1.6, 81.0, 17.6, None, 3.1, 2.20),
         (2503, 22.8, 3.3, 94.0, 16.6, (2509, 2), 4.4, 2.00),
         (2512, 21.2, 5.0, 84.0, 17.2, None, 5.2, 2.10),
         (2523, 22.6, 2.2, 90.0, 16.0, None, 0.8, 2.12)]
BIRD_HEAD = 7.5          # deg to the right of the sun's azimuth
BIRD_CLIMB = 5.0         # deg


def _bird_state(b, frame):
    f0, baz, bel, dist, v, flaps, ph, span = b
    if frame < f0 - 0.5:
        return None
    t = (frame - f0) / FPS
    c0 = camera(f0)
    az = math.radians(YAW + baz)
    el = math.radians(bel)
    start = c0.pos + dist * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    hd = math.radians(SUN_AZ + BIRD_HEAD + 1.2 * math.sin(0.35 * t + ph))
    cl = math.radians(BIRD_CLIMB)
    dirv = np.array([math.cos(cl) * math.sin(hd), math.sin(cl), math.cos(cl) * math.cos(hd)])
    pos = start + dirv * v * t + np.array([0.0, 0.6 * math.sin(0.9 * t + ph), 0.0])
    roll = math.radians(7.0 * math.sin(0.45 * t + ph))
    dihed = math.radians(7.0)
    if flaps is not None:
        fs, nb = flaps
        u = (frame - fs) / FPS / 1.05
        if 0.0 <= u < nb:
            dihed += math.radians(24.0) * math.sin(2 * math.pi * u) * math.sin(math.pi * u / nb)
    return pos, hd, cl, roll, dihed, span


def draw_eagles(img, zb, tcam, frame, sun_px_clear=26.0):
    H, W = img.shape[:2]
    Lk, amb, SD, fogp, QD = dawn_light(frame)
    sun = tcam.project_dir(sun_dir(frame))
    for b in BIRDS:
        st = _bird_state(b, frame)
        if st is None:
            continue
        pos, hd, cl, roll, dihed, span = st
        loc = _outline(dihed, span)
        # bird frame -> world (yaw hd, pitch cl, roll)
        fwd = np.array([math.cos(cl) * math.sin(hd), math.sin(cl), math.cos(cl) * math.cos(hd)])
        rgt = np.array([math.cos(hd), 0.0, -math.sin(hd)])
        up = np.cross(fwd, rgt)
        up = -up if up[1] < 0 else up
        cr, sr = math.cos(roll), math.sin(roll)
        rgt2 = rgt * cr - up * sr
        up2 = up * cr + rgt * sr
        P = pos[None, :] + loc[:, 0:1] * rgt2[None, :] + loc[:, 1:2] * up2[None, :] + loc[:, 2:3] * fwd[None, :]
        sx, sy, z = tcam.project(P)
        if (z <= 1.0).any():
            continue
        # never across the sun disc
        cx, cy = sx.mean(), sy.mean()
        if math.hypot(cx - sun[0], cy - sun[1]) < sun_px_clear * W / 1920:
            continue
        x0, x1 = int(sx.min()) - 2, int(sx.max()) + 3
        y0, y1 = int(sy.min()) - 2, int(sy.max()) + 3
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            continue
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        k = 6
        mask = np.zeros(((y1 - y0) * k, (x1 - x0) * k), np.uint8)
        poly = np.stack([(sx - x0) * k, (sy - y0) * k], 1).round().astype(np.int32)
        cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_8)
        a = cv2.resize(mask.astype(np.float32) / 255.0, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        a = a[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
        # silhouette with a little aerial haze
        d = float(np.linalg.norm(pos - tcam.pos))
        T = math.exp(-d / 2500.0)
        col = np.array([0.010, 0.009, 0.011]) * T + fogp[5:8] * 1.6 * (1 - T)
        reg = img[y0c:y1c, x0c:x1c]
        img[y0c:y1c, x0c:x1c] = reg * (1 - a[..., None]) + col[None, None, :] * a[..., None]


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    look = PI.look
    base = os.path.join(PI.CM.ROOT, 'renders', 'dawn_C')
    out = base if a.out is None else os.path.join(PI.CM.ROOT, 'renders', 'run_v2', 'tests', a.out)
    os.makedirs(out, exist_ok=True)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        s, e = (a.range or f'{START}-{END}').split('-')
        frames = list(range(int(s), int(e) + 1, a.step))
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.frame_path(out, f))]
    if a.procs <= 1:
        _work((frames, a.scale, a.ss, out))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(c, a.scale, a.ss, out) for c in chunks])


def _work(args):
    import time
    frames, scale, ss, out = args
    look = PI.look
    for f in frames:
        t0 = time.time()
        img = look.finish(render(f, scale=scale, ss=ss), **FINISH)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
