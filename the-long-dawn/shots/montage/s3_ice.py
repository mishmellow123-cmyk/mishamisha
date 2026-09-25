"""Shot 3 (1580-1639) ICE.

One idea: warm fire under a green sky.
Low on a snowy shore; across a black fjord a glacier front and white peaks; green-violet aurora
curtains ripple overhead and in the water. On a snowy promontory (left third) a figure in a
fur-ruffed parka, silhouetted against the aurora, lowers a torch into the beacon on the hit at
1600. Camera: slow push, small kick and tilt up after the ignition.
Lower third (text 1580-1600, 1620-1639) = smooth snow and dim water.
"""
import math

import numpy as np
from numba import njit, prange

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


@njit(inline='always', fastmath=True)
def _lod(scale, fp, lo, hi):
    o = math.log2(max(scale / max(fp * 2.0, 1e-4), 1.0))
    return min(max(o, lo), hi)


@njit(fastmath=True)
def h_land(x, z, fp):
    # shore: slopes from the camera down to the waterline ~z=45
    h = 0.35 - 0.00045 * z * z - 0.01 * max(z - 42.0, 0.0) ** 2
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
    # glacier front across the fjord + glacier surface rising behind
    zf = 640.0 + 55.0 * fbm2(x / 260.0, 0.3, 3.0, 7)
    if z > zf - 5.0:
        o3 = _lod(30.0, fp, 1.0, 6.0)
        wall = 46.0 + 7.0 * fbm2(x / 22.0, z / 60.0, o3, 8) + 0.012 * (z - zf)
        t = smoothstep(zf - 4.0, zf + 1.5, z)
        hg = -30.0 + (wall + 30.0) * t
        if hg > h:
            h = hg
    # mountains behind
    if z > 3000.0:
        o4 = _lod(1600.0, fp, 1.0, 10.0)
        r = ridged2(x / 1900.0 + 0.2, z / 1900.0 + 3.1, o4, 21)
        hm = -250.0 + 780.0 * r ** 1.7 * smoothstep(3000.0, 6000.0, z)
        if hm > h:
            h = hm
    # ice floes on the fjord
    if z > 70.0 and z < 600.0:
        fl = fbm2(x / 9.0, z / 9.0, 3.0, 11)
        if fl > 0.38:
            hf = WATER + 0.25 * smoothstep(0.38, 0.45, fl)
            if hf > h:
                h = hf
    return h


@njit(fastmath=True)
def hfun(x, z, fp, P):
    h = h_land(x, z, fp)
    if h < WATER:
        h = WATER
    dx = x - P[0]
    dz = z - P[1]
    return h - (dx * dx + dz * dz) / (2.0 * 6.371e6)


@njit(inline='always', fastmath=True)
def sky_all(ox, oy, oz, dx, dy, dz, S, A, t):
    r, g, b = SK.sky_base(dx, dy, dz, S)
    ar, ag, ab = SK.aurora_ray(ox, oy, oz, dx, dy, dz, A, t)
    return r + ar, g + ag, b + ab


@njit(parallel=True, fastmath=True)
def shade(C, D, P, S, A, t, LT, Lm, Im, amb, aur_amb, fogp, out, dep):
    H, W = D.shape
    mx, my, mz = Lm[0], Lm[1], Lm[2]
    for j in prange(H):
        for i in range(W):
            dx, dy, dz, sl, dxh, dzh = T.ray_of(C, i, j)
            d = D[j, i]
            if d >= 1e29:
                r, g, b = sky_all(C[0], C[1], C[2], dx, dy, dz, S, A, t)
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
            hl = h_land(x, z, fp)
            if hl <= WATER + 0.002:
                # water: rippled mirror of the sky (aurora), plus fire glints
                n1 = gnoise2(x * 0.9 + t * 0.6, z * 2.2, 31)
                n2 = gnoise2(x * 0.9 + 5.0, z * 2.2 - t * 0.5, 32)
                amp = 0.035 * (1.0 - smoothstep(0.005, 0.05, fp)) + 0.006
                nx = amp * n1
                nz = amp * n2 - 0.01
                ny = 1.0
                inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
                nx *= inv
                ny *= inv
                nz *= inv
                dn = dx * nx + dy * ny + dz * nz
                rx = dx - 2.0 * dn * nx
                ry = abs(dy - 2.0 * dn * ny)
                rz = dz - 2.0 * dn * nz
                r, g, b = sky_all(x, WATER, z, rx, ry, rz, S, A, t)
                fres = (0.03 + 0.97 * (1.0 - max(-dn, 0.0)) ** 5) * 0.6
                cr = r * fres + 0.002
                cg = g * fres + 0.003
                cb = b * fres + 0.005
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
                h0 = WATER
            else:
                nx, ny, nz, h0 = T.normal_at(hfun, P, x, z, e)
                steep = 1.0 - smoothstep(0.35, 0.6, ny)
                # snow; ice-blue on the steep glacier face
                ar = 0.80 - 0.35 * steep
                ag = 0.86 - 0.12 * steep
                ab = 0.98
                if z > 600.0 and steep > 0.5:
                    fl = 0.8 + 0.2 * gnoise2(x * 0.07, h0 * 0.04, 41) + 0.08 * gnoise2(x * 0.3, h0 * 0.1, 42)
                    ar *= fl
                    ag *= fl
                ndl = nx * mx + ny * my + nz * mz
                shd = 1.0
                if ndl > 0.0 and dist < 4000.0:
                    shd = T.soft_shadow(hfun, P, x, h0 + 0.05, z, mx, my, mz, 0.3, 900.0, 14, 10.0, fp)
                dif = max(ndl, 0.0) * shd
                up = 0.5 + 0.5 * ny
                cr = ar * (Im * Lm[3] * dif + amb[0] * up + aur_amb[0] * up)
                cg = ag * (Im * Lm[4] * dif + amb[1] * up + aur_amb[1] * up)
                cb = ab * (Im * Lm[5] * dif + amb[2] * up + aur_amb[2] * up)
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
    return Cam((0.0, 1.9 + 0.01 * k, push), -3.0 + 0.05 * k, tilt + 0.08 * k, HFOV, W, H)


def pose(frame):
    arm = keys(frame, [(START, 0.0), (1590, 0.0), (1599.5, 1.0), (1606, 0.7), (1620, 0.25), (END, 0.2)],
               ease=smoother)
    lean = keys(frame, [(START, 0.0), (1590, 0.0), (1599.5, 0.16), (1605, -0.05), (1618, 0.0), (END, 0.0)],
                ease=smoother)
    return dict(arm=arm, lean=lean)


def _ground(x, z):
    return h_land(x, z, 0.01)


def render(frame, scale=0.5, ss=1.5):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    t = CM.ftime(frame)
    cam = camera(frame, W, H)
    cami = cam.scaled(ss)
    C = cami.params()
    P = np.array([cam.pos[0], cam.pos[2], t, 0.0])
    D = np.zeros((cami.H, cami.W))
    T.march(hfun, P, C, 0.3, 30000.0, 0.004, 0.4, 1800.0, 9, D)

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
    aur_amb = np.array([0.006, 0.030, 0.014])
    S = SK.sky_params(zenith='#060A1A', horizon='#1D2A50', moon_dir=MOON_DIR, halo_I=0.0, halo2_I=0.0,
                      horizon_glow=0.25)
    A = np.array([1200.0, 5200.0, 1.0 / 2600.0, 7.0, 1.8, 1.0, 0.0, -1800.0, 56.0, 6.0])
    fogc = CM.lin('#1D2A50') * 0.9
    fogp = np.array([1.0 / 9000.0, fogc[0], fogc[1], fogc[2]])
    out = np.zeros((cami.H, cami.W, 3), np.float32)
    dep = np.zeros((cami.H, cami.W), np.float32)
    shade(C, D, P, S, A, t, LT, Lm, Im, amb, aur_amb, fogp, out, dep)
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
