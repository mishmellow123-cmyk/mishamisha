"""Shot 3 (1580-1639) ICE.

One idea: warm fire under a green sky.
Low on a snowy shore; across a still black fjord a tidewater glacier and a moonlit massif (right),
a low far shore in the centre; a few folded aurora curtains hang over it all and are mirrored in
the fjord. On a snowy promontory (left third) a figure in a fur-ruffed parka, silhouetted against
the aurora, lowers a torch into the beacon on the hit at 1600. Camera: slow push, small kick and
tilt up after the ignition.
Lower third (text 1580-1600, 1620-1639) = smooth snow and calm dim water.

Aurora (v2): explicit curtains, not a volume. Each curtain is a vertical emitting sheet hanging from
~100 km along a folded curve in the horizontal plane (Earth curvature included). The camera has no
roll and tilt is a lens shift, so every screen column is a vertical plane = one azimuth: per column
we box-filter the curtain segments that cross it (exact edge-on brightening at folds, alias-free
caustics), then lay down the analytic vertical profile (sharp lower edge, rays of varying height,
green -> faint red/purple aloft). The fjord reflection is a true mirror camera (terrain + sky +
aurora), remapped through gentle, footprint-filtered ripples.
"""
import math

import cv2
import numpy as np
from numba import njit, prange
from scipy.interpolate import PchipInterpolator

import common as CM
from mt import terrain as T, sky as SK, fire as F, figure as FG, props as PR, people as PP
from mt.cam import Cam, keys, smoother, kick
from mt.noise import fbm2, ridged2, gnoise2, smoothstep

START, END, IGN = 1580, 1639, 1600
HFOV = 52.0
F_FULL = 960.0 / math.tan(math.radians(HFOV / 2))
HORIZON_Y0 = 575.0
WATER = -0.5
MOON_DIR = np.array([0.35, 0.28, -0.89])
MOON_DIR = MOON_DIR / np.linalg.norm(MOON_DIR)
FEET = np.array([-7.2, 0.0, 24.0])
CAIRN = np.array([-5.7, 0.0, 24.4])
WIND = 1.0
CAM_Y = 1.9
R_KM = 6371.0


def _mass(az_deg, dist, height, radius):
    a = math.radians(az_deg)
    return [dist * math.sin(a), dist * math.cos(a), height, radius]


# mountain masses (world x, z, height, radius) in metres; view centre is azimuth -3 deg,
# frame spans about -29 .. +23 deg. Low behind the figure, a gap in the centre, massif right.
_MS = np.array([
    _mass(13.0, 13000.0, 860.0, 2600.0),
    _mass(20.5, 11500.0, 560.0, 2300.0),
    _mass(29.0, 9000.0, 520.0, 2600.0),
    _mass(7.0, 19000.0, 380.0, 3000.0),
    _mass(-21.0, 30000.0, 300.0, 5200.0),
    _mass(-40.0, 17000.0, 420.0, 4000.0),
])


@njit(inline='always', fastmath=True)
def _lod(scale, fp, lo, hi):
    o = math.log2(max(scale / max(fp * 2.0, 1e-4), 1.0))
    return min(max(o, lo), hi)


@njit(fastmath=True)
def h_far(x, z, fp):
    """Far shore, tidewater glacier (right) and mountain massifs. Returns (h, material)."""
    D = math.sqrt(x * x + z * z)
    az = math.atan2(x, z)
    # low far shore: snowy strand + rolling tundra
    ds = 4200.0 + 800.0 * fbm2(az * 4.0 + 1.0, 1.3, 3.0, 51)
    o = _lod(500.0, fp, 1.0, 6.0)
    hb = 4.0 + 22.0 * smoothstep(ds, ds + 9000.0, D) + 7.0 * fbm2(x / 600.0, z / 600.0, o, 52)
    h = -8.0 + (hb + 8.0) * smoothstep(ds - 30.0, ds + 160.0, D)
    mat = 3
    # mountains: art-directed masses x ridged multifractal detail
    E = 0.0
    for k in range(_MS.shape[0]):
        dx = x - _MS[k, 0]
        dz = z - _MS[k, 1]
        q = (dx * dx + dz * dz) / (_MS[k, 3] * _MS[k, 3])
        if q < 12.0:
            E += _MS[k, 2] * math.exp(-q ** 0.8)
    if E > 4.0:
        o2 = _lod(2200.0, fp, 1.0, 9.0)
        r = ridged2(x / 2200.0 + 0.4, z / 2200.0 - 2.2, o2, 21)
        hm = E * (0.45 + 0.75 * r) - 30.0
        if hm > h:
            h = hm
            mat = 2
    # tidewater glacier: ice cliffs across the right of the fjord, surface rising to the massif
    zf = 1250.0 + 0.35 * x + 60.0 * fbm2(x / 300.0, 0.7, 3.0, 57)
    if x > 90.0 and z > zf - 8.0:
        o3 = _lod(30.0, fp, 1.0, 6.0)
        side = smoothstep(90.0, 300.0, x)
        seracs = 5.0 * fbm2(x / 16.0, z / 40.0, o3, 58) + 3.0 * abs(fbm2(x / 55.0, z / 25.0, o3, 59))
        top = (26.0 + min(0.012 * (z - zf), 40.0) + seracs) * side
        wall = smoothstep(zf - 6.0, zf + 1.5, z)
        hg = -30.0 + (top + 30.0) * wall
        if hg > h:
            h = hg
            mat = 1
    return h, mat


@njit(fastmath=True)
def h_land_m(x, z, fp):
    # shore: slopes from the camera down to the waterline ~z=43
    h = 0.35 - 0.00045 * z * z - 0.01 * max(z - 42.0, 0.0) ** 2
    mat = 0
    o = _lod(3.0, fp, 1.0, 5.0)
    h += 0.12 * fbm2(x * 0.3, z * 0.3, o, 3)
    # promontory on the left carrying the figure + beacon
    px = (x + 7.0) / 5.5
    pz = (z - 25.0) / 6.0
    q = px * px + pz * pz
    if q < 1.6:
        hp = 3.3 * math.exp(-q * q * 1.6)
        o2 = _lod(4.0, fp, 1.0, 5.0)
        hp += 0.35 * fbm2(x * 0.35, z * 0.35, o2, 5) * smoothstep(0.1, 0.8, q)
        if hp > h:
            h = hp
    if z > 1000.0:
        hf, mf = h_far(x, z, fp)
        if hf > h:
            h = hf
            mat = mf
    # a few ice floes on the fjord
    if z > 70.0 and z < 900.0:
        fl = fbm2(x / 11.0, z / 14.0, 3.0, 11)
        if fl > 0.52:
            hf = WATER + 0.22 * smoothstep(0.52, 0.58, fl)
            if hf > h:
                h = hf
                mat = 4
    return h, mat


@njit(fastmath=True)
def h_land(x, z, fp):
    h, m = h_land_m(x, z, fp)
    return h


@njit(fastmath=True)
def hfun(x, z, fp, P):
    h = h_land(x, z, fp)
    if h < WATER:
        h = WATER
    dx = x - P[0]
    dz = z - P[1]
    return h - (dx * dx + dz * dz) / (2.0 * 6.371e6)


@njit(fastmath=True)
def hfun_m(x, z, fp, P):
    """Height field seen by the mirror camera: only what stands above the water beyond the shore."""
    if z < 60.0:
        return -80.0
    h = h_land(x, z, fp)
    if h <= WATER + 0.01:
        return -80.0
    dx = x - P[0]
    dz = z - P[1]
    return h - (dx * dx + dz * dz) / (2.0 * 6.371e6)


# ------------------------------------------------------------------ aurora ---

@njit(fastmath=True)
def build_curtain(phib, rhob, cp, t, seg):
    """Fold a base curve (world azimuth phib [rad], distance rhob [km]) into a curtain and write its
    segments: seg[k] = [az0, az1, rho, length, emission, h0, Hg, purple]."""
    N = phib.shape[0]
    seed = int(cp[0])
    I = cp[1]
    h00 = cp[2]
    Hg0 = cp[3]
    famp = cp[4]
    flen = cp[5]
    fspd = cp[6]
    rspd = cp[7]
    pur = cp[8]
    taper = cp[9]
    pbias = cp[10]
    px = np.empty(N)
    pz = np.empty(N)
    jw = np.empty(N)
    h0 = np.empty(N)
    hg = np.empty(N)
    L = 0.0
    Ltot = 0.0
    for k in range(1, N):
        ax = rhob[k] * math.sin(phib[k]) - rhob[k - 1] * math.sin(phib[k - 1])
        az = rhob[k] * math.cos(phib[k]) - rhob[k - 1] * math.cos(phib[k - 1])
        Ltot += math.sqrt(ax * ax + az * az)
    for k in range(N):
        if k > 0:
            ax = rhob[k] * math.sin(phib[k]) - rhob[k - 1] * math.sin(phib[k - 1])
            az = rhob[k] * math.cos(phib[k]) - rhob[k - 1] * math.cos(phib[k - 1])
            L += math.sqrt(ax * ax + az * az)
        # folds: a travelling kink, tangential + radial, amplitude wandering along the curtain
        env = famp * (0.35 + 0.65 * smoothstep(-0.35, 0.35, fbm2(L / 420.0 + 3.1, 0.37, 2.0, seed + 1)))
        ph = 6.2832 * L / flen - fspd * t + 2.2 * fbm2(L / 600.0 - 0.8, 1.7, 2.0, seed + 2)
        dtan = env * math.sin(ph)
        drad = 0.9 * env * math.cos(ph)
        a = phib[k] + dtan / rhob[k]
        r = rhob[k] + drad
        px[k] = r * math.sin(a)
        pz[k] = r * math.cos(a)
        # emission: bright sections and gaps x drifting rays
        patch = smoothstep(-0.12, 0.45, fbm2(L / 210.0 - 0.02 * t, 5.3, 3.0, seed + 3) + pbias)
        f1 = 0.5 + 0.5 * gnoise2(L / 1.9 - rspd * t, 0.37, seed + 4)
        f2 = 0.5 + 0.5 * gnoise2(L / 6.5 - 0.55 * rspd * t, 0.71, seed + 5)
        f3 = 0.5 + 0.5 * gnoise2(L / 23.0 - 0.25 * rspd * t, 0.13, seed + 6)
        ray = 0.12 + 1.9 * (0.22 * f1 * f1 + 0.45 * f2 * f2 * f2 + 0.33 * f3 * f3 * f3)
        ends = smoothstep(0.0, taper, L) * smoothstep(0.0, taper, Ltot - L)
        jw[k] = I * patch * ray * ends
        h0[k] = h00 + 4.0 * gnoise2(L / 55.0 - 0.35 * t, 2.2, seed + 7) + 1.8 * gnoise2(L / 14.0 + 0.9 * t, 3.3,
                                                                                    seed + 8)
        hg[k] = Hg0 * min(0.25 + 2.2 * f2 * f2 * f3 * f3 + 0.2 * f1, 1.8)
    for k in range(N - 1):
        seg[k, 0] = math.atan2(px[k], pz[k])
        seg[k, 1] = math.atan2(px[k + 1], pz[k + 1])
        r0 = math.sqrt(px[k] * px[k] + pz[k] * pz[k])
        r1 = math.sqrt(px[k + 1] * px[k + 1] + pz[k + 1] * pz[k + 1])
        seg[k, 2] = 0.5 * (r0 + r1)
        ddx = px[k + 1] - px[k]
        ddz = pz[k + 1] - pz[k]
        seg[k, 3] = math.sqrt(ddx * ddx + ddz * ddz)
        seg[k, 4] = 0.5 * (jw[k] + jw[k + 1])
        seg[k, 5] = 0.5 * (h0[k] + h0[k + 1])
        seg[k, 6] = 0.5 * (hg[k] + hg[k + 1])
        seg[k, 7] = pur


@njit(parallel=True, fastmath=True)
def aurora_img(C, SEG, NSEG, tau, thick, out):
    """Render the curtains for camera C into out (H,W,3), column by column (see module doc)."""
    H, W = out.shape[0], out.shape[1]
    fx, fz, rx, rz = C[3], C[4], C[5], C[6]
    f, cx, cyy = C[7], C[8], C[9]
    R2 = 2.0 * 6371.0
    sig = 1.3
    for i in prange(W):
        xo = i + 0.5 - cx
        a_lo = math.atan2(fx * f + rx * (xo - 0.5), fz * f + rz * (xo - 0.5))
        a_hi = math.atan2(fx * f + rx * (xo + 0.5), fz * f + rz * (xo + 0.5))
        dcol = a_hi - a_lo
        hl = math.sqrt(f * f + xo * xo)
        for c in range(SEG.shape[0]):
            for k in range(NSEG[c]):
                s0 = SEG[c, k, 0]
                s1 = SEG[c, k, 1]
                rho = SEG[c, k, 2]
                hw = 0.5 * thick / rho
                lo = min(s0, s1) - hw
                hi = max(s0, s1) + hw
                ov = min(hi, a_hi) - max(lo, a_lo)
                if ov <= 0.0:
                    continue
                span = hi - lo
                frac = ov / span
                wgt = SEG[c, k, 4] * SEG[c, k, 3] / (rho * dcol) * frac
                if wgt < 1e-6:
                    continue
                h0 = SEG[c, k, 5]
                Hg = SEG[c, k, 6]
                pur = SEG[c, k, 7]
                drop = rho * rho / R2
                smin = (h0 - 4.0 * sig - drop) / rho
                smax = (h0 + 7.0 * Hg + 25.0 - drop) / rho
                jmax = int(cyy - 0.5 - smin * hl) + 1
                jmin = int(cyy - 0.5 - smax * hl) - 1
                if jmax > H - 1:
                    jmax = H - 1
                if jmin < 0:
                    jmin = 0
                for j in range(jmin, jmax + 1):
                    s = (cyy - (j + 0.5)) / hl
                    if s <= 0.0:
                        continue
                    a = rho * s + drop - h0
                    if a < 0.0:
                        q = a / sig
                        g = math.exp(-q * q)
                        p = 0.0
                        fr = g * math.exp(-((a + 1.5) / 1.2) ** 2)
                    else:
                        g = 0.55 * math.exp(-a / 5.5) + 0.45 * math.exp(-a / Hg)
                        p = pur * smoothstep(0.35 * Hg, 1.6 * Hg, a) * math.exp(-a / (1.5 * Hg))
                        fr = math.exp(-((a + 1.5) / 1.2) ** 2)
                    out[j, i, 0] += wgt * (0.10 * g + 0.60 * p + 0.35 * fr)
                    out[j, i, 1] += wgt * (1.00 * g + 0.05 * p + 0.02 * fr)
                    out[j, i, 2] += wgt * (0.40 * g + 0.34 * p + 0.25 * fr)
        # atmospheric extinction toward the horizon (Kasten-Young air mass)
        for j in range(H):
            s = (cyy - (j + 0.5)) / hl
            if s <= 0.0:
                out[j, i, 0] = 0.0
                out[j, i, 1] = 0.0
                out[j, i, 2] = 0.0
                continue
            e = math.degrees(math.atan(s))
            am = 1.0 / (math.sin(math.radians(e)) + 0.50572 * (e + 6.07995) ** -1.6364)
            ex = math.exp(-tau * am)
            out[j, i, 0] *= ex
            out[j, i, 1] *= ex
            out[j, i, 2] *= ex


def _rho_of_elev(e_deg, h0=100.0):
    te = np.tan(np.radians(e_deg))
    return R_KM * (-te + np.sqrt(te * te + 2.0 * h0 / R_KM))


# curtains: lower-edge elevation (deg) at world-azimuth control points (deg) + fold/ray params
# cp = [seed, I, h0, Hg, fold_amp km, fold_len km, fold_speed rad/s, ray_speed km/s, purple, taper km]
CURTAINS = [
    # hero: sweeps from low behind the massif (right) up over the figure (upper left)
    dict(ctrl=[(-62, 13.5), (-40, 12.2), (-25, 11.0), (-10, 9.2), (3, 7.2), (14, 5.2), (26, 3.6), (44, 2.4)],
         cp=[101, 0.66, 102.0, 40.0, 62.0, 270.0, 0.22, 3.0, 0.8, 60.0, 0.0], n=4200),
    # far arc low along the horizon (centre), mirrored in the fjord
    dict(ctrl=[(-48, 3.4), (-30, 2.6), (-15, 2.0), (-3, 1.7), (8, 1.8), (20, 2.2), (36, 2.8)],
         cp=[202, 0.95, 104.0, 55.0, 40.0, 200.0, 0.18, 2.2, 0.5, 80.0, 0.3], n=3200),
    # faint high ribbon, top right, ends inside the frame
    dict(ctrl=[(-14, 17.5), (-2, 15.8), (10, 14.4), (22, 13.8), (34, 13.6)],
         cp=[303, 0.14, 106.0, 26.0, 22.0, 140.0, 0.30, 4.0, 0.3, 90.0, 0.0], n=1800),
]


def _curtain_segs(t):
    nmax = max(c['n'] for c in CURTAINS)
    SEG = np.zeros((len(CURTAINS), nmax - 1, 8))
    NSEG = np.zeros(len(CURTAINS), np.int64)
    for ci, c in enumerate(CURTAINS):
        ctrl = np.array(c['ctrl'], np.float64)
        phi = np.linspace(ctrl[0, 0], ctrl[-1, 0], c['n'])
        # smooth (monotone cubic) interpolation of the lower-edge elevation
        e = PchipInterpolator(ctrl[:, 0], ctrl[:, 1])(phi)
        rho = _rho_of_elev(e, c['cp'][2])
        cp = np.array(c['cp'], np.float64)
        seg = np.zeros((c['n'] - 1, 8))
        build_curtain(np.radians(phi), rho, cp, t, seg)
        SEG[ci, :c['n'] - 1] = seg
        NSEG[ci] = c['n'] - 1
    return SEG, NSEG


def aurora(C, H, W, t, tau=0.035):
    SEG, NSEG = _curtain_segs(t)
    out = np.zeros((H, W, 3), np.float32)
    aurora_img(C, SEG, NSEG, tau, 2.5, out)
    return out


# ------------------------------------------------------------------ shading ---

@njit(parallel=True, fastmath=True)
def shade(C, D, P, S, t, LT, Lm, Im, amb, aur_amb, La, fogp, out, dep, wf, mapx, mapy):
    H, W = D.shape
    mx, my, mz = Lm[0], Lm[1], Lm[2]
    for j in prange(H):
        for i in range(W):
            dx, dy, dz, sl, dxh, dzh = T.ray_of(C, i, j)
            d = D[j, i]
            wf[j, i] = 0.0
            if d >= 1e29:
                r, g, b = SK.sky_base(dx, dy, dz, S)
                wf[j, i] = -1.0
                out[j, i, 0] = r
                out[j, i, 1] = g
                out[j, i, 2] = b
                dep[j, i] = 1e9
                continue
            x = C[0] + dxh * d
            z = C[2] + dzh * d
            dist = d * math.sqrt(1.0 + sl * sl)
            fp = dist / C[7]
            e = max(fp * 0.7, 0.01)
            hl, mat = h_land_m(x, z, fp)
            if hl <= WATER + 0.002:
                # still water: gentle swell + fine ripples, filtered by the (grazing) footprint
                fpz = fp / max(-dy, 0.002)
                w1 = 1.0 - smoothstep(2.0, 9.0, fpz)
                w2 = 1.0 - smoothstep(0.4, 1.6, fpz)
                n1 = gnoise2(x * 0.11 + t * 0.03, z * 0.07 - t * 0.05, 31)
                n2 = gnoise2(x * 0.11 + 7.0, z * 0.07 + t * 0.04, 33)
                n3 = gnoise2(x * 0.7 + t * 0.2, z * 0.45, 32)
                n4 = gnoise2(x * 0.7 + 5.0, z * 0.45 - t * 0.25, 34)
                nx = 0.0035 * w1 * n1 + 0.0015 * w2 * n3
                nz = 0.0050 * w1 * n2 + 0.0020 * w2 * n4
                ny = 1.0
                inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
                nx *= inv
                ny *= inv
                nz *= inv
                dn = dx * nx + dy * ny + dz * nz
                rx = dx - 2.0 * dn * nx
                ry = abs(dy - 2.0 * dn * ny)
                rz = dz - 2.0 * dn * nz
                fres = (0.03 + 0.97 * (1.0 - max(-dn, 0.0)) ** 5) * 0.9
                # where the reflected ray lands in the (mirror) camera image
                fx_, fz_, rx_, rz_ = C[3], C[4], C[5], C[6]
                zc = rx * fx_ + rz * fz_
                xc = rx * rx_ + rz * rz_
                zc = max(zc, 0.05)
                mapx[j, i] = C[8] + C[7] * xc / zc
                mapy[j, i] = C[9] - C[7] * ry / zc
                wf[j, i] = fres
                cr = 0.0015
                cg = 0.0022
                cb = 0.0040
                for li in range(LT.shape[0]):
                    lx = LT[li, 0] - x
                    ly = LT[li, 1] - WATER
                    lz = LT[li, 2] - z
                    ll = math.sqrt(lx * lx + ly * ly + lz * lz) + 1e-9
                    hx = lx / ll - dx
                    hy = ly / ll - dy
                    hz = lz / ll - dz
                    hn = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
                    nh = max((nx * hx + ny * hy + nz * hz) / hn, 0.0)
                    sp = nh ** 900.0 * 60.0 + nh ** 120.0 * 1.5
                    E = LT[li, 6] * sp * fres / (1.0 + ll * 0.02)
                    cr += E * LT[li, 3]
                    cg += E * LT[li, 4]
                    cb += E * LT[li, 5]
            else:
                nx, ny, nz, h0 = T.normal_at(hfun, P, x, z, e)
                steep = 1.0 - smoothstep(0.35, 0.6, ny)
                ar = 0.80 - 0.35 * steep
                ag = 0.86 - 0.12 * steep
                ab = 0.98
                if mat == 1:
                    # glacier: blue ice cliffs with vertical fluting, snow on top
                    fl = 0.75 + 0.25 * gnoise2(x * 0.09, h0 * 0.03, 41) + 0.12 * gnoise2(x * 0.45, h0 * 0.08, 42)
                    ice = smoothstep(0.35, 0.75, 1.0 - ny)
                    ar = (0.84 * (1.0 - ice) + 0.30 * ice * fl)
                    ag = (0.88 * (1.0 - ice) + 0.58 * ice * fl)
                    ab = (0.98 * (1.0 - ice) + 0.86 * ice * fl)
                elif mat == 2 or mat == 3:
                    # mountains / far shore: snow fields, dark rock on steep faces
                    th = 0.50 + 0.10 * gnoise2(x / 90.0, z / 90.0, 43) + 0.05 * gnoise2(x / 23.0, z / 23.0, 44)
                    rock = 1.0 - smoothstep(th - 0.08, th + 0.06, ny)
                    ar = 0.84 * (1.0 - rock) + 0.055 * rock
                    ag = 0.88 * (1.0 - rock) + 0.060 * rock
                    ab = 0.98 * (1.0 - rock) + 0.070 * rock
                elif mat == 4:
                    ar = 0.70
                    ag = 0.76
                    ab = 0.86
                ndl = nx * mx + ny * my + nz * mz
                shd = 1.0
                if ndl > 0.0:
                    if dist < 1500.0:
                        shd = T.soft_shadow(hfun, P, x, h0 + 0.05, z, mx, my, mz, 0.3, 900.0, 14, 10.0, fp)
                    else:
                        shd = T.soft_shadow(hfun, P, x, h0 + 2.0, z, mx, my, mz, 8.0, 5000.0, 16, 12.0, fp)
                dif = max(ndl, 0.0) * shd
                up = 0.5 + 0.5 * ny
                # aurora light: broad source low in the north (ahead of camera)
                adl = max(nx * La[0] + ny * La[1] + nz * La[2], 0.0)
                aw = 0.35 * up + 0.9 * adl
                cr = ar * (Im * Lm[3] * dif + amb[0] * up + aur_amb[0] * aw)
                cg = ag * (Im * Lm[4] * dif + amb[1] * up + aur_amb[1] * aw)
                cb = ab * (Im * Lm[5] * dif + amb[2] * up + aur_amb[2] * aw)
                for li in range(LT.shape[0]):
                    lx = LT[li, 0] - x
                    ly = LT[li, 1] - h0
                    lz = LT[li, 2] - z
                    l2 = lx * lx + ly * ly + lz * lz
                    ll = math.sqrt(l2) + 1e-9
                    ndf = (nx * lx + ny * ly + nz * lz) / ll
                    if ndf > 0.0:
                        E = LT[li, 6] * ndf / (l2 + LT[li, 7] * LT[li, 7])
                        cr += ar * E * LT[li, 3]
                        cg += ag * E * LT[li, 4]
                        cb += ab * E * LT[li, 5]
            tau = dist * fogp[0]
            tr = math.exp(-tau)
            out[j, i, 0] = cr * tr + fogp[1] * (1.0 - tr)
            out[j, i, 1] = cg * tr + fogp[2] * (1.0 - tr)
            out[j, i, 2] = cb * tr + fogp[3] * (1.0 - tr)
            dep[j, i] = dist


def camera(frame, W=1920, H=804):
    t = CM.ftime(frame)
    tilt0 = math.degrees(math.atan((HORIZON_Y0 - 402.0) / F_FULL))
    push = keys(frame, [(START, 0.0), (END, 2.2)], ease=lambda u: u * 0.7 + 0.3 * u * u * (3 - 2 * u))
    tilt = keys(frame, [(START, tilt0), (IGN, tilt0), (IGN + 24, tilt0 + 1.6), (END, tilt0 + 2.0)], ease=smoother)
    k = kick(t, CM.ftime(IGN), amp=1.0, freq=5.5, decay=5.5)
    return Cam((0.0, CAM_Y + 0.01 * k, push), -3.0 + 0.05 * k, tilt + 0.08 * k, HFOV, W, H)


def pose(frame):
    arm = keys(frame, [(START, 0.0), (1590, 0.0), (1599.5, 1.0), (1606, 0.7), (1620, 0.25), (END, 0.2)],
               ease=smoother)
    lean = keys(frame, [(START, 0.0), (1590, 0.0), (1599.5, 0.16), (1605, -0.05), (1618, 0.0), (END, 0.0)],
                ease=smoother)
    return dict(arm=arm, lean=lean)


def _ground(x, z):
    return h_land(x, z, 0.01)


def _elev_row(C, elev_deg):
    """Row (in C's image) of a given elevation at the image centre column."""
    return C[9] - C[7] * math.tan(math.radians(elev_deg))


def render(frame, scale=0.5, ss=1.5):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    t = CM.ftime(frame)
    cam = camera(frame, W, H)
    cami = cam.scaled(ss)
    C = cami.params()
    P = np.array([cam.pos[0], cam.pos[2], t, 0.0])
    D = np.zeros((cami.H, cami.W))
    T.march(hfun, P, C, 0.3, 45000.0, 0.004, 0.4, 2000.0, 9, D)

    feet = FEET.copy()
    feet[1] = _ground(feet[0], feet[2])
    cairn = CAIRN.copy()
    cairn[1] = _ground(cairn[0], cairn[2])
    ti = CM.ftime(IGN)
    size, inten, light = F.ignite_env(t, ti)
    fl = F.flicker(t, 13)
    back, front, fb, rb = PR.cairn(seed=31, height=0.95)
    fire_base = cairn + np.array([0.0, fb, 0.0])
    fire_I = 6.5 * light * fl
    p = pose(frame)
    fig, pts = PP.torch_person('parka', **p)
    th = pts['torch_head']
    torch_w = PR.local_to_world(cam, feet, th[0], th[1], 0.0)
    torch_I = 0.5 * F.flicker(t, 3)
    LT = [[torch_w[0], torch_w[1] + 0.12, torch_w[2], *F.FIRE_LIGHT, torch_I, 0.2]]
    if fire_I > 0:
        LT.append([fire_base[0], fire_base[1] + 0.7, fire_base[2], *F.FIRE_LIGHT, fire_I * 1.6, 0.5])
    LT = np.array(LT, np.float64)

    moon_col = CM.lin('#9DB4D9')
    Im = 0.22
    Lm = np.r_[MOON_DIR, moon_col]
    amb = CM.lin('#27335E') * 0.25
    aur_amb = np.array([0.004, 0.034, 0.016])
    La = np.array([0.0, 0.45, 1.0])
    La = La / np.linalg.norm(La)
    S = SK.sky_params(zenith='#060A1A', horizon='#1D2A50', moon_dir=MOON_DIR, halo_I=0.0, halo2_I=0.0,
                      horizon_glow=0.25)
    fogc = CM.lin('#1D2A50') * 0.8
    fogp = np.array([1.0 / 42000.0, fogc[0], fogc[1], fogc[2]])
    out = np.zeros((cami.H, cami.W, 3), np.float32)
    dep = np.zeros((cami.H, cami.W), np.float32)
    wf = np.zeros((cami.H, cami.W), np.float32)
    mapx = np.zeros((cami.H, cami.W), np.float32)
    mapy = np.zeros((cami.H, cami.W), np.float32)
    shade(C, D, P, S, t, LT, Lm, Im, amb, aur_amb, La, fogp, out, dep, wf, mapx, mapy)

    # aurora in the sky (same supersampled grid as the terrain)
    aur = aurora(C, cami.H, cami.W, t)
    # faint atmospheric glow around the curtains
    small = cv2.resize(aur, (max(cami.W // 8, 8), max(cami.H // 8, 8)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 5.0 * scale / 0.5 / 1.0)
    glow = cv2.resize(small, (cami.W, cami.H), interpolation=cv2.INTER_LINEAR)
    sky = (wf < -0.5).astype(np.float32)[..., None]
    out += (aur + 0.10 * glow) * sky

    # mirror camera for the fjord: a band of rows from just below the horizon to ~+5 deg
    j_top = int(max(math.floor(_elev_row(C, 5.0)), 0))
    j_bot = int(min(math.ceil(_elev_row(C, -0.6)), cami.H))
    if j_bot - j_top > 4 and (wf > 0).any():
        Hb = j_bot - j_top
        Cb = C.copy()
        Cb[1] = 2.0 * WATER - C[1]
        Cb[9] = C[9] - j_top
        Db = np.zeros((Hb, cami.W))
        T.march(hfun_m, P, Cb, 60.0, 45000.0, 0.004, 0.4, 2000.0, 9, Db)
        outb = np.zeros((Hb, cami.W, 3), np.float32)
        depb = np.zeros((Hb, cami.W), np.float32)
        wfb = np.zeros((Hb, cami.W), np.float32)
        mxb = np.zeros((Hb, cami.W), np.float32)
        myb = np.zeros((Hb, cami.W), np.float32)
        noLT = LT[:0].copy()
        shade(Cb, Db, P, S, t, noLT, Lm, Im, amb, aur_amb, La, fogp, outb, depb, wfb, mxb, myb)
        aurb = aurora(Cb, Hb, cami.W, t)
        glowb = glow[j_top:j_bot]
        skyb = (wfb < -0.5).astype(np.float32)[..., None]
        outb += (aurb + 0.10 * glowb) * skyb
        refl = cv2.remap(outb, mapx, mapy - j_top, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        refl = cv2.GaussianBlur(refl, (0, 0), sigmaX=0.5 * ss * scale * 2.0, sigmaY=1.2 * ss * scale * 2.0)
        out += refl * np.clip(wf, 0, None)[..., None]

    img = CM.downsample(out, W, H)
    depth = CM.depth_down(dep, W, H)
    skymask = CM.downsample((dep > 1e8).astype(np.float32), W, H)
    SK.splat_stars(img, cam, _stars(), skymask, t=t, gain=1.0, scale=scale)

    lights = [dict(dir=MOON_DIR, col=moon_col, I=Im * 0.8), dict(pos=torch_w, col=F.FIRE_LIGHT, I=torch_I, r0=0.12),
              dict(dir=(0.0, 1.0, 0.3), col=(0.2, 1.0, 0.45), I=0.02)]
    if fire_I > 0:
        lights.append(dict(pos=fire_base + np.array([0, 0.6, 0]), col=F.FIRE_LIGHT, I=fire_I * 0.8, r0=0.3))
    famb = amb * 1.2 + aur_amb
    FG.render(img, depth, cam, back, cairn, lights, amb=famb, t=t, emissive_gain=min(light, 1.5) * 1.2)
    FG.render(img, depth, cam, fig, feet, lights, amb=famb, t=t, seed=11)
    sx, sy, z = cam.project(torch_w)
    F.halo(img, depth, sx, sy, 60 * scale, 0.10 * torch_I, z=z, zbias=0.5)
    F.flame(img, depth, cam, torch_w + np.array([0.0, 0.01, 0.0]), 0.32, 0.05, t, seed=19, I=16.0, tongues=3,
            lean=0.10 * WIND, zbias=0.1)
    if size > 0:
        _smoke().render(img, depth, cam, t, fire_base + np.array([0, 0.9, 0]), 0.5 * light * fl, albedo=0.22,
                        amb=(0.003, 0.006, 0.006))
        F.shimmer(img, cam, fire_base, 2.1 * size, rb, t, amp_px=1.2 * scale)
        F.flame(img, depth, cam, fire_base, 2.2 * size, rb, t, seed=6, I=26.0 * inten, lean=0.3 * WIND * size,
                zbias=0.4)
        sx, sy, z = cam.project(fire_base + np.array([0, 0.9, 0]))
        F.halo(img, depth, sx, sy, 220 * scale * size, 0.010 * light * fl, z=z, zbias=3.0)
    FG.render(img, depth, cam, front, cairn, lights, amb=famb, t=t, write_depth=False)
    if size > 0:
        _sparks().render(img, depth, cam, t)
    return img


_ST = None
_SM = None
_SP = None


def _stars():
    global _ST
    if _ST is None:
        _ST = SK.make_stars(16000, 303, lum_scale=6.0)
    return _ST


def _smoke():
    global _SM
    if _SM is None:
        c = CAIRN.copy()
        c[1] = _ground(c[0], c[2])
        _SM = F.Smoke(61, c + np.array([0, 2.3, 0]), CM.ftime(IGN) + 0.15, CM.ftime(END) + 0.1, rate=4.0,
                      wind=(1.4 * WIND, 0, 0.2), rise=1.4, life=4.0, r0=0.3, growth=0.6, dens=0.35)
    return _SM


def _sparks():
    global _SP
    if _SP is None:
        c = CAIRN.copy()
        c[1] = _ground(c[0], c[2])
        _SP = F.Sparks(63, c + np.array([0, 1.4, 0]), CM.ftime(IGN), CM.ftime(END) + 0.1, burst=300, rate=55,
                       ember_rate=12, wind=(1.8 * WIND, 0.0, 0.2), I=30.0)
    return _SP


FINISH = dict(exposure=1.0, bloom_strength=0.07, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)
