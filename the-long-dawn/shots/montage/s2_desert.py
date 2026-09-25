"""Shot 2 (1520-1579) DESERT.

One idea: a lone robe on a knife-edge crest under the whole galaxy.
Moonlit transverse dunes; raking moon from the left splits every crest into a lit face and a
shadowed slip face; S-curved crest lines lead to a robed figure on a crest who lowers a torch
into a stone beacon on the hit at 1540. Huge sky, Milky Way core low on the horizon.
Camera: slow truck right (dune parallax). Lower third (text 1530-1600) = smooth lit sand.
"""
import math

import numpy as np
from numba import njit, prange

import common as CM
from mt import terrain as T, sky as SK, fire as F, figure as FG, props as PR, people as PP
from mt.cam import Cam, keys, smoother, kick
from mt.noise import fbm2, gnoise2, smoothstep, h01

START, END, IGN = 1520, 1579, 1540
LAM = 78.0
HFOV = 46.0
F_FULL = 960.0 / math.tan(math.radians(HFOV / 2))
HORIZON_Y0 = 548.0
MOON_DIR = np.array([0.86, 0.33, -0.40])
MOON_DIR = MOON_DIR / np.linalg.norm(MOON_DIR)
WIND = -1.0    # toward -x (over the crest, into the slip face)
YAW = 0.0


@njit(inline='always', fastmath=True)
def _lod(scale, fp, lo, hi):
    o = math.log2(max(scale / max(fp * 2.0, 1e-4), 1.0))
    return min(max(o, lo), hi)


# hero crest, designed in image space: control points (z, x, H) chosen so the crest enters lower-left,
# climbs diagonally to the summit (figure) a third in from the right, then falls away to the right.
# Camera eye is at y = EYE; crest heights below the eye near the camera (we see the slip face), above
# it at the summit (the figure stands against the sky).
EYE = 7.0
_CZ = np.array([-60.0, -20.0, 0.0, 10.0, 20.0, 40.0, 60.0, 75.0, 90.0, 140.0, 250.0, 500.0])
_CX = np.array([2.0, -0.4, -1.4, -3.3, -4.3, -1.5, 9.5, 15.0, 21.0, 30.0, 40.0, 60.0])
_CH = np.array([5.0, 5.2, 5.3, 5.0, 5.5, 8.6, 14.2, 14.4, 12.0, 7.5, 5.0, 4.0])


def _table():
    from scipy.interpolate import PchipInterpolator
    zz = np.arange(-60.0, 500.01, 0.25)
    fx = PchipInterpolator(_CZ, _CX)
    fh = PchipInterpolator(_CZ, _CH)
    return np.stack([fx(zz), fx.derivative()(zz), fh(zz)], 1).astype(np.float64)


CREST = _table()


@njit(inline='always', fastmath=True)
def crest_at(z):
    u = (z + 60.0) * 4.0
    if u < 0.0:
        u = 0.0
    n = CREST.shape[0] - 1
    if u > n - 1e-6:
        u = n - 1e-6
    k = int(u)
    f = u - k
    x = CREST[k, 0] * (1 - f) + CREST[k + 1, 0] * f
    dx = CREST[k, 1] * (1 - f) + CREST[k + 1, 1] * f
    h = CREST[k, 2] * (1 - f) + CREST[k + 1, 2] * f
    return x, dx, h


def crest_x(z):
    return crest_at(z)[0]


@njit(fastmath=True)
def h_field(x, z, fp):
    """Background transverse dune sea (crests roughly parallel to the hero crest)."""
    ca = 0.94
    sa = 0.33
    xr = ca * x - sa * z
    zr = sa * x + ca * z
    o = _lod(200.0, fp, 1.0, 5.0)
    wv = 26.0 * fbm2(zr / 190.0 + 0.3, xr / 900.0, o, 3) + 7.0 * fbm2(zr / 55.0, xr / 300.0, o - 1.0, 4)
    u = (xr + wv) / LAM
    cell = math.floor(u)
    s = u - cell
    ci = int(cell)
    amp = 9.0 * (0.72 + 0.35 * h01(ci, 7, 5) + 0.3 * fbm2(zr / 260.0, cell * 1.7, 3.0, 6))
    c = 0.74 + 0.06 * fbm2(zr / 120.0, cell * 3.1, 2.0, 8)
    if s < c:
        q = s / c
        p = 1.0 - (1.0 - q) ** 1.55
    else:
        q = (s - c) / (1.0 - c)
        p = (1.0 - q) ** 1.08
    return amp * p - 3.0


@njit(fastmath=True)
def h_dune(x, z, fp):
    # hero dune: sharp sinuous crest; windward (lit) face toward +x, slip face toward -x
    xc, dxc, H = crest_at(z)
    cosa = 1.0 / math.sqrt(1.0 + dxc * dxc)
    w = (x - xc) * cosa
    if w >= 0.0:
        u = w / (3.4 * H)
        hh = H * max(1.0 - u, 0.0) ** 1.7
    else:
        hh = H + w * 0.64
        # soft foot of the slip face
        if hh < 0.8:
            hh = 0.8 * math.exp((hh - 0.8) / 0.8)
    o = _lod(18.0, fp, 0.0, 3.0)
    hh += 0.25 * fbm2(x / 16.0, z / 12.0, o, 10) * min(1.0, abs(w) / 3.0)
    hb = h_field(x, z, fp)
    h = max(hh, hb)
    o2 = _lod(800.0, fp, 1.0, 4.0)
    h += 12.0 * fbm2(x / 820.0, z / 820.0, o2, 9) * smoothstep(60.0, 300.0, math.sqrt(x * x + z * z))
    return h


@njit(fastmath=True)
def hfun(x, z, fp, P):
    dx = x - P[0]
    dz = z - P[1]
    return h_dune(x, z, fp) - (dx * dx + dz * dz) / (2.0 * 6.371e6)


@njit(parallel=True, fastmath=True)
def shade(C, D, P, S, G, LT, Lm, Im, amb, fogp, mw_I, out, dep):
    H, W = D.shape
    mx, my, mz = Lm[0], Lm[1], Lm[2]
    for j in prange(H):
        for i in range(W):
            dx, dy, dz, sl, dxh, dzh = T.ray_of(C, i, j)
            d = D[j, i]
            if d >= 1e29:
                r, g, b = SK.sky_base(dx, dy, dz, S)
                mr, mg, mb = SK.milky_way(dx, dy, dz, G, mw_I)
                ext = smoothstep(0.0, 0.18, dy)
                out[j, i, 0] = r + mr * ext
                out[j, i, 1] = g + mg * ext
                out[j, i, 2] = b + mb * ext
                dep[j, i] = 1e9
                continue
            x = C[0] + dxh * d
            z = C[2] + dzh * d
            dist = d * math.sqrt(1.0 + sl * sl)
            fp = dist / C[7]
            e = max(fp * 0.7, 0.01)
            nx, ny, nz, h0 = T.normal_at(hfun, P, x, z, e)
            # wind ripples near camera (normal perturbation, faded by footprint)
            rip = 1.0 - smoothstep(0.008, 0.03, fp)
            if rip > 0.0:
                ph = (x * 0.94 + z * 0.34 + 0.8 * math.sin(z * 0.37 + x * 0.05)) / 0.42
                sk = math.cos(ph * 6.2832)
                amp = 0.16 * rip * (0.5 + 0.5 * gnoise2(x * 0.2, z * 0.2, 12))
                nx += amp * sk * 0.94
                nz += amp * sk * 0.34
                inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
                nx *= inv
                ny *= inv
                nz *= inv
            ar, ag, ab = 0.46, 0.43, 0.40
            ndl = nx * mx + ny * my + nz * mz
            shd = 1.0
            if ndl > 0.0:
                shd = T.soft_shadow(hfun, P, x, h0 + 0.05, z, mx, my, mz, 0.3, 700.0, 18, 14.0, fp)
            dif = max(ndl, 0.0) * shd
            skyl = 0.5 + 0.5 * ny
            cr = ar * (Im * Lm[3] * dif + amb[0] * skyl)
            cg = ag * (Im * Lm[4] * dif + amb[1] * skyl)
            cb = ab * (Im * Lm[5] * dif + amb[2] * skyl)
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
            cosv = dx * mx + dy * my + dz * mz
            ph = 1.0 + fogp[1] * max(cosv, 0.0) ** 3
            out[j, i, 0] = cr * tr + fogp[2] * ph * (1.0 - tr)
            out[j, i, 1] = cg * tr + fogp[3] * ph * (1.0 - tr)
            out[j, i, 2] = cb * tr + fogp[4] * ph * (1.0 - tr)
            dep[j, i] = dist


# ------------------------------------------------------------------ placement ---

CAM_Z = 0.0
FIG_Z = 60.0


_PL = None


def placement():
    global _PL
    if _PL is None:
        fx = crest_x(FIG_Z)
        # summit sits right on the crest; the beacon just before it, a touch lower
        bz = FIG_Z - 2.2
        bx = crest_x(bz)
        cx0 = crest_x(CAM_Z) + 1.4
        _PL = dict(feet=np.array([fx + 0.15, h_dune(fx + 0.15, FIG_Z, 0.01), FIG_Z]),
                   cairn=np.array([bx + 0.1, h_dune(bx + 0.1, bz, 0.01), bz]),
                   cam=np.array([cx0, EYE, CAM_Z]))
    return _PL


def camera(frame, W=1920, H=804):
    pl = placement()
    t = CM.ftime(frame)
    tilt0 = math.degrees(math.atan((HORIZON_Y0 - 402.0) / F_FULL))
    tx = keys(frame, [(START, -0.5), (END, 0.5)], ease=lambda u: u * u * (3 - 2 * u) * 0.35 + u * 0.65)
    k = kick(t, CM.ftime(IGN), amp=1.0, freq=5.0, decay=5.0)
    pos = pl['cam'] + np.array([tx, 0.004 * k, 0.0])
    yaw = YAW + 0.05 * k
    return Cam(pos, yaw, tilt0 + 0.05 * k, HFOV, W, H)


def pose(frame):
    arm = keys(frame, [(START, 0.0), (1531, 0.0), (1539.5, 1.0), (1546, 0.75), (1560, 0.35), (END, 0.3)],
               ease=smoother)
    lean = keys(frame, [(START, 0.0), (1531, 0.0), (1539.5, 0.14), (1545, -0.04), (1558, 0.0), (END, 0.0)],
                ease=smoother)
    return dict(arm=arm, lean=lean)


def wind_fn(t, y):
    g = 0.7 + 0.3 * math.sin(t * 1.1) * math.sin(t * 0.6 + 2.0)
    # figure is mirrored (faces left), so world +x wind is local -x
    return (-WIND * 4.0 * g - 0.8 * math.sin(t * 6.1 + y * 5.0), 0.6 * math.sin(t * 5.3 + y * 3.0))


def cloth(frame):
    t0 = CM.ftime(START - 30)
    t1 = CM.ftime(frame)

    def tail_anchor(tt):
        p = pose(tt * CM.FPS)
        _, pts = PP.robed(**p)
        return pts['tail_anchor']

    def hem_anchor(tt):
        return (-0.28, 0.12)

    tail = FG.verlet_chain(tail_anchor, 6, 0.09, t0, t1, wind_fn, init_dir=(-1.0, -0.6), drag=3.0)
    hem = FG.verlet_chain(hem_anchor, 5, 0.08, t0, t1, wind_fn, init_dir=(-0.6, -0.4), drag=2.5)
    return tail, hem


def render(frame, scale=0.5, ss=1.5):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    t = CM.ftime(frame)
    pl = placement()
    cam = camera(frame, W, H)
    cami = cam.scaled(ss)
    C = cami.params()
    P = np.array([cam.pos[0], cam.pos[2], t, 0.0])
    D = np.zeros((cami.H, cami.W))
    T.march(hfun, P, C, 0.5, 30000.0, 0.004, 0.5, 90.0, 9, D)

    ti = CM.ftime(IGN)
    size, inten, light = F.ignite_env(t, ti)
    fl = F.flicker(t, 5)
    back, front, fb, rb = PR.cairn(seed=21, height=0.95)
    cairn = pl['cairn']
    fire_base = cairn + np.array([0.0, fb, 0.0])
    fire_I = 6.5 * light * fl
    p = pose(frame)
    tail, hem = cloth(frame)
    fig, pts = PP.robed(tail_pts=tail, hem_pts=hem, **p)
    feet = pl['feet']
    th = pts['torch_head']
    torch_w = PR.local_to_world(cam, feet, -th[0], th[1], 0.0)
    torch_I = 0.5 * F.flicker(t, 9)

    LT = [[torch_w[0], torch_w[1] + 0.12, torch_w[2], *F.FIRE_LIGHT, torch_I, 0.2]]
    if fire_I > 0:
        LT.append([fire_base[0], fire_base[1] + 0.7, fire_base[2], *F.FIRE_LIGHT, fire_I * 2.2, 0.5])
    LT = np.array(LT, np.float64)

    moon_col = CM.lin('#9DB4D9')
    Im = 0.75
    Lm = np.r_[MOON_DIR, moon_col]
    amb = CM.lin('#27335E') * 0.30
    S = SK.sky_params(zenith='#070B1C', horizon='#27335E', moon_dir=MOON_DIR, halo_I=0.02, halo_w=0.35,
                      halo2_I=0.012, halo2_w=1.1, horizon_glow=0.35)
    G = _galaxy()
    fogc = CM.lin('#27335E') * 1.05
    fogp = np.array([1.0 / 14000.0, 0.8, fogc[0], fogc[1], fogc[2]])
    out = np.zeros((cami.H, cami.W, 3), np.float32)
    dep = np.zeros((cami.H, cami.W), np.float32)
    shade(C, D, P, S, G, LT, Lm, Im, amb, fogp, 0.05, out, dep)
    img = CM.downsample(out, W, H)
    depth = CM.depth_down(dep, W, H)
    skymask = CM.downsample((dep > 1e8).astype(np.float32), W, H)
    SK.splat_stars(img, cam, _stars(), skymask, t=t, gain=1.0, scale=scale)

    moon_light = dict(dir=MOON_DIR, col=moon_col, I=Im * 0.8)
    lights = [moon_light, dict(pos=torch_w, col=F.FIRE_LIGHT, I=torch_I, r0=0.12)]
    if fire_I > 0:
        lights.append(dict(pos=fire_base + np.array([0, 0.6, 0]), col=F.FIRE_LIGHT, I=fire_I * 0.8, r0=0.3))
    famb = amb * 1.2
    FG.render(img, depth, cam, back, cairn, lights, amb=famb, t=t, emissive_gain=min(light, 1.5) * 1.2)
    FG.render(img, depth, cam, fig, feet, lights, amb=famb, t=t, seed=9, flipx=True)
    sx, sy, z = cam.project(torch_w)
    F.halo(img, depth, sx, sy, 50 * scale, 0.08 * torch_I, z=z, zbias=0.5)
    F.flame(img, depth, cam, torch_w + np.array([0.0, 0.01, 0.0]), 0.32, 0.05, t, seed=13, I=16.0, tongues=3,
            lean=0.10 * WIND, zbias=0.1)
    if size > 0:
        _smoke().render(img, depth, cam, t, fire_base + np.array([0, 0.9, 0]), 0.5 * light * fl, albedo=0.22,
                        amb=(0.004, 0.005, 0.009))
        F.shimmer(img, cam, fire_base, 2.1 * size, rb, t, amp_px=1.2 * scale)
        F.flame(img, depth, cam, fire_base, 2.1 * size, rb, t, seed=4, I=26.0 * inten, lean=0.45 * WIND * size,
                zbias=0.4)
        sx, sy, z = cam.project(fire_base + np.array([0, 0.9, 0]))
        F.halo(img, depth, sx, sy, 150 * scale * size, 0.012 * light * fl, z=z, zbias=3.0)
    FG.render(img, depth, cam, front, cairn, lights, amb=famb, t=t, write_depth=False)
    if size > 0:
        _sparks().render(img, depth, cam, t)
    return img


_ST = None
_SM = None
_SP = None
_G = None


def _galaxy():
    global _G
    if _G is None:
        yaw = math.radians(YAW)

        def d(az, el):
            az = math.radians(az) + yaw
            el = math.radians(el)
            return np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        A = d(16.0, 0.0)
        B = d(-30.0, 64.0)
        g = np.cross(A, B)
        g /= np.linalg.norm(g)
        _G = np.r_[g, A, 0.16, 5.0]
    return _G


def _stars():
    global _ST
    if _ST is None:
        G = _galaxy()
        _ST = SK.make_stars(26000, 202, lum_scale=6.0, band=(G[:3], 0.14), band_frac=0.4)
    return _ST


def _smoke():
    global _SM
    if _SM is None:
        pl = placement()
        _SM = F.Smoke(51, pl['cairn'] + np.array([0, 2.3, 0]), CM.ftime(IGN) + 0.15, CM.ftime(END) + 0.1, rate=4.0,
                      wind=(2.0 * WIND, 0, 0.3), rise=1.3, life=4.0, r0=0.3, growth=0.6, dens=0.35)
    return _SM


def _sparks():
    global _SP
    if _SP is None:
        pl = placement()
        _SP = F.Sparks(53, pl['cairn'] + np.array([0, 1.4, 0]), CM.ftime(IGN), CM.ftime(END) + 0.1, burst=300,
                       rate=55, ember_rate=12, wind=(3.0 * WIND, 0.0, 0.3), I=30.0)
    return _SP


FINISH = dict(exposure=1.0, bloom_strength=0.07, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)
