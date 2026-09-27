"""B1 . DUSK (R8): the range at sunset from high above the cloud sea.

The sun sets behind the camera; we look east over the range toward her summit. Each point is lit by the sun
only while the grazing ray toward the sun clears the cloud sea and the ranges (a long clearance march with the
Earth's curvature), and the light it gets is the sun seen through that grazing path (Kasten-Young air mass,
Rayleigh + a little aerosol): gold where the sun is still well up for it, deep orange-red as it grazes, gone
after. So the shadow line climbs every face as the sun sinks, the low peaks go first and her summit (the highest
in frame) keeps the last red point. The sky opposite the sun carries the Earth's shadow (a blue-grey band rising
from the horizon) under the Belt of Venus (rose). Same world and set as the lifetime (keeper.CR_B).

  python shots/run/dusk.py --stills          # three states (early / mid / last point) at quarter scale
  python shots/run/dusk.py --still 0.5 --scale 0.5
"""
import math
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import keeper as KP         # noqa: E402
from mt.noise import gnoise2, smoothstep   # noqa: E402

CM = PI.CM
look = PI.look
CLOUD_Y = WD.CLOUD_Y
R_EARTH = WD.R_EARTH
CR = KP.CR_B

# ------------------------------------------------------------------ camera ---
CAM_AZ_FROM_TOP = 212.0      # the camera sits SSW of her summit, looking NNE (B's west is SSW)
CAM_DIST = 5200.0
CAM_Y = 430.0
HFOV = 36.0
SUN_AZ = 205.0               # sets behind the camera, a little to its left
PITCH = -0.2


def camera(W, H, dist=CAM_DIST, y=CAM_Y, az=CAM_AZ_FROM_TOP, look_off=(-6.5, 0.0)):
    top = KP.TOP
    a = math.radians(az)
    pos = np.array([top[0] + dist * math.sin(a), y, top[2] + dist * math.cos(a)])
    d = top - pos
    yaw = math.degrees(math.atan2(d[0], d[2])) + look_off[0]
    curv = (d[0] ** 2 + d[2] ** 2) / (2 * R_EARTH)
    pitch = PITCH + look_off[1]
    return RC.RCam(pos, yaw, pitch, 0.0, HFOV, W, H)


def glow_dir(el_deg):
    """Where the western twilight glow comes from: the sun's azimuth, ~6 deg up."""
    return sun_dir(6.0)


def sun_dir(el_deg, az=SUN_AZ):
    e, a = math.radians(el_deg), math.radians(az)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


# ------------------------------------------------------------------ light ---

@njit(inline='always', fastmath=True)
def airmass(c):
    """Kasten-Young air mass at apparent elevation c (radians, clamped at 0)."""
    cd = max(c * 57.29578, 0.0)
    return 1.0 / (math.sin(math.radians(cd)) + 0.50572 * (cd + 6.07995) ** -1.6364)


@njit(inline='always', fastmath=True)
def clearance(P, CR, x, y, z, lx, ly, lz, fp, nsteps, tmax):
    """Smallest angular clearance (radians) of the ray toward the sun over the terrain + cloud sea."""
    t = max(fp * 2.0, 0.5)
    grow = (tmax / t) ** (1.0 / nsteps)
    c = 1.0
    for _s in range(nsteps):
        h = WD.hfun(x + lx * t, z + lz * t, fp + t * 0.002, P, CR)
        a = (y + ly * t - h) / t
        if a < c:
            c = a
        t *= grow
    return c


# SK: 0-2 anti-solar horizontal dir | 3 shadow top el (rad) | 4-6 zenith | 7-9 horizon | 10-12 shadow col
#     13-15 belt col | 16 belt width (rad) | 17 twilight gain
@njit(inline='always', fastmath=True)
def dusk_sky(dx, dy, dz, SK):
    el = math.asin(min(max(dy, -1.0), 1.0))
    hl = math.sqrt(dx * dx + dz * dz) + 1e-9
    ca = (dx * SK[0] + dz * SK[2]) / hl
    wa = (0.5 + 0.5 * ca) ** 1.5                   # toward the anti-solar point
    u = smoothstep(0.0, 0.55, max(el, 0.0)) ** 0.5
    r = SK[7] + (SK[4] - SK[7]) * u
    g = SK[8] + (SK[5] - SK[8]) * u
    b = SK[9] + (SK[6] - SK[9]) * u
    es = SK[3]
    # the Earth's shadow (blue-grey, soft-edged) and the Belt of Venus (rose) just above it
    sh = (1.0 - smoothstep(es - 0.012, es + 0.014, el)) * wa
    belt = math.exp(-((el - es - 0.035) / SK[16]) ** 2) * wa * (1.0 - sh)
    r = r * (1.0 - sh) + SK[10] * sh + SK[13] * belt
    g = g * (1.0 - sh) + SK[11] * sh + SK[14] * belt
    b = b * (1.0 - sh) + SK[12] * sh + SK[15] * belt
    return r * SK[17], g * SK[17], b * SK[17]


@njit(parallel=True, fastmath=True, cache=True)
def dusk_shade(C, D, P, CR, SK, L, QD, amb, fogp, out, dist_out, GW):
    """QD: 0 sun I | 1-3 Rayleigh tau rgb | 4 aerosol tau | 5 cloud albedo | 6 cloud amb gain | 7 clearance steps
    | 8 clearance tmax | 9 belt fill on west faces | 10 sheen."""
    H, W = D.shape
    lx, ly, lz = L[0], L[1], L[2]
    rsun = QD[11]                                   # effective disc radius: art-directed small, so the
                                                    # shadow line reads as it climbs (true penumbra ~1 km)
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
                r, g, b = dusk_sky(dx, dy, dz, SK)
                out[j, i, 0] = r
                out[j, i, 1] = g
                out[j, i, 2] = b
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
            hf = WD.h_rock(x, z, fp, CR)
            hc = WD.h_cloud(x, z, fp, P[2])
            cloud = hc > hf
            if cloud:
                h0 = hc
                hx = WD.h_cloud(x + e, z, fp, P[2])
                hz = WD.h_cloud(x, z + e, fp, P[2])
            else:
                h0 = hf
                hx = WD.h_rock(x + e, z, fp, CR)
                hz = WD.h_rock(x, z + e, fp, CR)
            nx = -(hx - h0)
            ny = e
            nz = -(hz - h0)
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= inv
            ny *= inv
            nz *= inv
            yw = h0 - curv
            ndl = nx * lx + ny * ly + nz * lz
            # the sun through the grazing path
            sr = 0.0
            sg = 0.0
            sb = 0.0
            if ndl > -0.05:
                c = clearance(P, CR, x, yw + max(fp * 1.5, 0.3), z, lx, ly, lz, fp, int(QD[7]), QD[8])
                vis = smoothstep(-rsun, rsun, c)
                if vis > 0.0:
                    m = airmass(c + ly * 0.0)
                    sr = vis * math.exp(-(QD[1] + QD[4]) * m)
                    sg = vis * math.exp(-(QD[2] + QD[4]) * m)
                    sb = vis * math.exp(-(QD[3] + QD[4]) * m)
            # the western twilight glow (behind the camera, low, warm): a broad soft source on every face
            gw = max((nx * GW[0] + ny * GW[1] + nz * GW[2]) * 0.5 + 0.5, 0.0) ** 1.5
            if cloud:
                wrap = max((ndl + 0.6) / 1.6, 0.0)
                ca = QD[5]
                sky = 0.55 + 0.45 * ny
                cr = ca * (QD[0] * sr * wrap + amb[0] * QD[6] * sky + GW[3] * gw)
                cg = ca * (QD[0] * sg * wrap + amb[1] * QD[6] * sky + GW[4] * gw)
                cb = ca * (QD[0] * sb * wrap + amb[2] * QD[6] * sky + GW[5] * gw)
                trough = smoothstep(CLOUD_Y - 120.0, CLOUD_Y + 110.0, h0)
                tk = 0.55 + 0.45 * trough
                cr *= tk
                cg *= tk
                cb *= tk
            else:
                es = max(fp * 6.0, 45.0)
                hsx = WD.h_rock(x + es, z, fp * 4.0, CR)
                hsz = WD.h_rock(x, z + es, fp * 4.0, CR)
                nsx = -(hsx - h0) / es
                nsz = -(hsz - h0) / es
                nsy = 1.0 / math.sqrt(nsx * nsx + nsz * nsz + 1.0)
                sn = gnoise2(x / 380.0, z / 380.0, 73) * 0.18 + gnoise2(x / 95.0, z / 95.0, 74) * 0.06
                snow = smoothstep(0.30, 0.52, nsy + sn + 0.05 * gnoise2(x / 30.0, z / 30.0, 75))
                # couloirs and ribs down the fall line: noise stretched along the slope, so the big faces read as
                # gullied rock and snow, not smooth facets
                gl = math.sqrt(nsx * nsx + nsz * nsz) + 1e-6
                ux = -nsz / gl
                uz = nsx / gl
                ac = (x * ux + z * uz)                  # across the slope
                al = (x * nsx + z * nsz) / gl           # down the slope
                gul = gnoise2(ac / 9.0, al / 70.0, 77) * 0.6 + gnoise2(ac / 3.5, al / 25.0, 78) * 0.4
                steep = smoothstep(0.35, 0.8, gl)
                snow = min(max(snow + steep * (0.55 * gul + 0.05), 0.0), 1.0)
                ar = 0.070 + (0.82 - 0.070) * snow
                ag = 0.062 + (0.86 - 0.062) * snow
                ab = 0.058 + (0.95 - 0.058) * snow
                dif = max(ndl, 0.0)
                skyl = 0.55 + 0.45 * ny
                # the belt's rose light on faces turned toward the sunset (after the sun has gone from them)
                west = max(nx * lx + nz * lz, 0.0) / (math.sqrt(lx * lx + lz * lz) + 1e-9)
                belt = QD[9] * west * (1.0 - min(sr * 4.0, 1.0))
                cr = ar * (QD[0] * sr * dif + amb[0] * skyl + belt * 1.0 + GW[3] * gw)
                cg = ag * (QD[0] * sg * dif + amb[1] * skyl + belt * 0.55 + GW[4] * gw)
                cb = ab * (QD[0] * sb * dif + amb[2] * skyl + belt * 0.62 + GW[5] * gw)
                if snow > 0.0 and dif > 0.0:
                    hx_ = lx - dx
                    hy_ = ly - dy
                    hz_ = lz - dz
                    hl_ = math.sqrt(hx_ * hx_ + hy_ * hy_ + hz_ * hz_) + 1e-9
                    nh = max((nx * hx_ + ny * hy_ + nz * hz_) / hl_, 0.0)
                    sp = snow * QD[10] * nh ** 20
                    cr += sp * QD[0] * sr
                    cg += sp * QD[0] * sg
                    cb += sp * QD[0] * sb
            # aerial perspective: blue dusk air
            yc = C[1]
            tau = WD.height_fog_tau(dist, yc, yw, fogp[0], fogp[1])
            tau += WD.height_fog_tau(dist, yc - CLOUD_Y, yw - CLOUD_Y, fogp[2], fogp[3])
            tr = math.exp(-tau)
            out[j, i, 0] = cr * tr + fogp[5] * (1.0 - tr)
            out[j, i, 1] = cg * tr + fogp[6] * (1.0 - tr)
            out[j, i, 2] = cb * tr + fogp[7] * (1.0 - tr)
            dist_out[j, i] = dist


def dusk_params(sun_el):
    """Light and sky for a sun elevation (deg, camera frame). The shadow top rises as the sun sinks."""
    lin = CM.lin
    L = sun_dir(sun_el)
    anti = -L.copy()
    anti[1] = 0.0
    anti /= np.linalg.norm(anti)
    dep = max(-sun_el, 0.0)
    SKP = np.zeros(20)
    SKP[0:3] = anti
    SKP[3] = math.radians(-1.25 + 1.3 * dep + 0.35)       # the Earth's shadow top (from the cloud horizon up)
    SKP[4:7] = lin('#1E2F66') * 0.50
    SKP[7:10] = lin('#7F8BB8') * 0.44
    SKP[10:13] = lin('#4F5C8A') * 0.36
    SKP[13:16] = lin('#EFA8B0') * 0.30 * (1.0 - smoothstep(2.0, 6.0, dep))
    SKP[16] = math.radians(3.3)
    SKP[17] = 1.0
    amb = lin('#4A5F98') * 0.16
    fogc = lin('#6E6E9C') * 0.40                     # the far air: the Earth's shadow, a little lavender
    fogp = np.array([7.2e-5, 1 / 2600.0, 4.0e-5, 1 / 160.0, 0.0, fogc[0], fogc[1], fogc[2]])
    QD = np.zeros(12)
    QD[0] = 5.5
    QD[1:4] = [0.030, 0.068, 0.150]
    QD[4] = 0.022
    QD[5] = 0.92
    QD[6] = 2.6
    QD[7] = 56
    QD[8] = 160000.0
    QD[9] = 0.020
    QD[10] = 0.3
    QD[11] = 0.0011
    return L, SKP, amb, fogp, QD


def render(sun_el, scale=0.25, ss=1.5, cam=None):
    """The one-off still path (a full march per call). The motion path is `DuskShot` below."""
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tc = cam or camera(W, H)
    fr = PI.Frame(tc, ss)
    scam = fr.src
    C = scam.params()
    P = np.array([scam.pos[0], scam.pos[2], 0.0, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(P, CR, C, 0.2, 160000.0, 0.0035, 0.35, 2500.0, 9, D)
    L, SKP, amb, fogp, QD = dusk_params(sun_el)
    out = np.zeros((scam.H, scam.W, 3), np.float32)
    di = np.zeros((scam.H, scam.W), np.float32)
    GW = np.r_[glow_dir(sun_el), CM.lin('#E9A98E') * 0.030]
    dusk_shade(C, D, P, CR, SKP, L, QD, amb, fogp, out, di, GW)
    fr.img = out
    fr.zb = di.copy()
    fr.dist = di
    img, zb, dd = PI.to_target(fr)
    return img


# ------------------------------------------------------------------ motion (R8, frames 0-639) ---
# The locked frame is marched once (relight.build: G-buffer + each pixel's clearance toward SUN_AZ); every frame is
# then a relight at the sun elevation e(f). e(f) is solved from the bar map: six peaks lose their last light on the
# downbeats of bars 2-7 (f 80..480, lowest first) and her summit's last red point goes out on bar 8 b3 (f 600).
import relight as RL        # noqa: E402

F0, F1 = 0, 640
SYNC_PEAKS = [80, 160, 240, 320, 400, 480]
SYNC_LAST = 600
SNOW_BIAS = 0.10             # B's peaks carry more snow: the alpenglow is rose on snow, not orange rock


def motion_params(e):
    """LP / SKY / amb / fog for the relight at sun elevation e (deg)."""
    lin = CM.lin
    L, SKP, amb0, fogp0, QD = dusk_params(e)
    LP = np.zeros(RL.NLP)
    LP[0:3] = L
    LP[3] = 5.0
    LP[4:7] = [0.024, 0.062, 0.112]            # a little less blue lost than the still: rose, not orange
    LP[7] = 0.020
    LP[8] = QD[11]
    LP[9] = 0.92
    LP[10] = 2.4
    LP[11] = 0.050                             # the Belt of Venus' rose on the faces toward the sunset
    LP[12] = 0.35
    LP[13:16] = glow_dir(e)
    LP[16:19] = lin('#E9A98E') * 0.030
    LP[19] = 0.0
    LP[21] = 0.0
    LP[30] = 0.30                               # her last point stays a vivid red to the end...
    LP[31] = 0.00035                            # ...and goes out crisply (on bar 8 b3)
    amb = lin('#5A6CA8') * 0.30
    fogp = np.zeros(12)
    fogp[:8] = fogp0
    fogp[8] = 8.0
    fogp[9:12] = fogp0[5:8]
    return LP, SKP, amb, fogp


class DuskShot:
    def __init__(self, scale=0.25, ss=1.5, cache=True):
        self.scale, self.ss = scale, ss
        W, H = int(round(1920 * scale)), int(round(804 * scale))
        self.tc = camera(W, H)
        self.fr = PI.Frame(self.tc, ss)
        scam = self.fr.src
        self.P = np.array([scam.pos[0], scam.pos[2], 0.0, 0.0])
        p = os.path.join(KP.CACHE, f'dusk_G_{scale:.3f}_{ss:.2f}.npy')
        if cache and os.path.exists(p):
            self.G = np.load(p)
        else:
            self.G = RL.build(scam, self.P, CR, SUN_AZ, nsteps=80, snow_bias=SNOW_BIAS)
            if cache:
                os.makedirs(KP.CACHE, exist_ok=True)
                np.save(p, self.G)
        self.clamp_to_her()
        self.schedule()

    def clamp_to_her(self, margin=0.05):
        """Art direction: her summit holds the last red point. Every lit pixel off her summit (more than 140 m from
        her top, or 70 m below it) loses its light at least `margin` deg of sun elevation before her tip does."""
        G = self.G
        rock = G[..., RL.G_FLAG] == 1.0
        dxz = np.hypot(G[..., RL.G_X] - KP.TOP[0], G[..., RL.G_Z] - KP.TOP[2])
        hers = rock & (dxz < 140.0) & (G[..., RL.G_Y] > KP.TOP_Y - 70.0 - (dxz ** 2) / (2 * WD.R_EARTH) * 0)
        c0 = G[..., RL.G_C0]
        c_her = float(c0[hers].max()) if hers.any() else float(c0[rock].max())
        e_her = math.degrees(math.asin(np.clip(-c_her - 0.0011, -0.2, 0.2)))
        cap = -math.sin(math.radians(e_her + margin)) - 0.0011
        G[..., RL.G_C0] = np.where(rock & ~hers, np.minimum(c0, cap), c0)
        G[..., RL.G_TROUGH] = np.where(hers, 1.0, np.where(rock, 0.0, G[..., RL.G_TROUGH]))   # tag her summit
        self.her_mask = hers

    def dark_el(self):
        """Per pixel: the sun elevation (deg) at which the pixel's last direct light goes (rock only)."""
        c0 = self.G[..., RL.G_C0]
        rock = self.G[..., RL.G_FLAG] == 1.0
        pen = np.where(self.her_mask, 0.00035, 0.0011)
        s = np.clip(-c0 - pen, -0.2, 0.2)
        e = np.degrees(np.arcsin(s))
        return np.where(rock, e, -9.0)

    def schedule(self):
        """Pick the six bar-map peaks and solve e(f)."""
        from scipy.ndimage import minimum_filter, label
        from scipy.interpolate import PchipInterpolator
        E = self.dark_el()
        scam = self.fr.src
        # her summit's last point: the lowest dark-elevation near the projection of her top
        hx, hy, hz = scam.project(KP.TOP + np.array([0.0, 2.0, 0.0]))
        y0, x0 = int(hy), int(hx)
        self.e_her = float(E[self.her_mask].min()) if self.her_mask.any() else float(E[E > -8].min())
        # candidate peak tips: local minima of the dark elevation (each peak's last lit point)
        k = max(int(round(18 * self.scale * self.ss)), 3)
        mn = minimum_filter(np.where(E > -8, E, 9.0), size=k)
        tips = np.argwhere((E == mn) & (E > -8) & (E < 5.0))
        cand = []
        for j, i in tips:
            if abs(i - x0) < 3 * k and abs(j - y0) < 3 * k:
                continue                       # not her summit
            # prominent: the region lit a bar before the tip goes dark is at least a few pixels
            area = int(((E[max(j - k, 0):j + k, max(i - k, 0):i + k] < E[j, i] + 0.12)
                        & (E[max(j - k, 0):j + k, max(i - k, 0):i + k] > -8)).sum())
            cand.append((float(E[j, i]), int(i), int(j), area))
        cand = [c for c in cand if c[0] > self.e_her + 0.08 and c[3] >= 4]
        cand.sort(key=lambda c: -c[0])
        # six, spread evenly in darkening elevation between the first and her summit, biggest at each step
        e_hi = cand[0][0] if cand else self.e_her + 0.8
        targets = np.linspace(min(e_hi, self.e_her + 0.95), self.e_her + 0.10, 6)
        chosen = []
        for tg in targets:
            best = None
            for c in cand:
                if any(abs(c[1] - q[1]) < 2 * k and abs(c[2] - q[2]) < 2 * k for q in chosen):
                    continue
                if chosen and c[0] >= chosen[-1][0] - 0.02:
                    continue
                score = abs(c[0] - tg) * 8.0 - min(c[3], 400) / 400.0
                if best is None or score < best[0]:
                    best = (score, c)
            if best is not None:
                chosen.append(best[1])
        self.chosen = chosen
        fs = [F0] + SYNC_PEAKS[:len(chosen)] + [SYNC_LAST, F1 - 1]
        es = [chosen[0][0] + 0.25] + [c[0] for c in chosen] + [self.e_her, self.e_her - 0.06]
        self.e_of = PchipInterpolator(np.array(fs, float), np.array(es, float))

    def render(self, frame):
        e = float(self.e_of(frame))
        LP, SKP, amb, fogp = motion_params(e)
        scam = self.fr.src
        out = np.zeros((scam.H, scam.W, 3), np.float32)
        RL.shade(self.G, LP, SKP, amb, fogp, float(scam.pos[1]), out)
        fr = self.fr
        fr.img = out
        dist = self.G[..., RL.G_DIST]
        fr.zb = dist
        fr.dist = dist
        img, _, _ = PI.to_target(fr)
        return img


FINISH = dict(exposure=1.05, bloom_strength=0.06, bloom_threshold=1.2, streak_strength=0.0, vignette_amount=0.22)


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--still', type=float, default=None, help='sun elevation (deg)')
    ap.add_argument('--range', default=None, help='motion frames a-b (inclusive), 0-639')
    ap.add_argument('--frames', default=None, help='motion frames, comma list')
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default='dusk')
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    out = os.path.join(CM.ROOT, a.out) if a.out.startswith('renders/') else os.path.join(KP.OUTDIR, a.out)
    os.makedirs(out, exist_ok=True)
    if a.range or a.frames:
        if a.range:
            s0, s1 = a.range.split('-')
            frames = list(range(int(s0), int(s1) + 1))
        else:
            frames = [int(x) for x in a.frames.split(',')]
        if a.skip:
            frames = [f for f in frames if not os.path.exists(look.frame_path(out, f))]
        t0 = time.time()
        shot = DuskShot(a.scale, a.ss)
        print(f'G-buffer {time.time() - t0:.1f}s; her last point at e={shot.e_her:.3f}; peaks '
              + ', '.join(f'{c[0]:.3f}@({c[1]},{c[2]})' for c in shot.chosen), flush=True)
        for f in frames:
            t1 = time.time()
            img = look.finish(shot.render(f), **FINISH)
            look.save_png(look.frame_path(out, f), img)
            print(f'frame {f} e={float(shot.e_of(f)):+.3f} {time.time() - t1:.2f}s', flush=True)
        return
    els = [a.still] if a.still is not None else [0.1, -0.55, -0.80, -0.90]
    for el in els:
        t0 = time.time()
        img = look.finish(render(el, a.scale, a.ss), **FINISH)
        p = os.path.join(out, f'dusk_{el:+.2f}_{a.scale:.2f}.png')
        look.save_png(p, img)
        print(p, f'{time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
