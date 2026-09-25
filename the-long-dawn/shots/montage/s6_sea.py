"""Shot 6 (1720-1767) SEA.

One idea: a boat in the moon-path raises fire, and the whole coast answers.
Dark swells, the moon low over the sea laying a glitter path; a small boat sits in the path, a
sailor in oilskins at the bow strikes and raises a burning flare on the hit at 1730; then along
the far coast a chain of beacons ignites one after another, each laying its own streak on the
water (running into the 1760-1767 handle and on into the globe). Camera floats on the swell.
"""
import math

import numpy as np
from numba import njit, prange

import common as CM
from mt import sky as SK, fire as F, figure as FG, props as PR, people as PP
from mt.figure import Drawing
from mt.cam import Cam, keys, smoother, kick
from mt.noise import fbm2, gnoise2, smoothstep

START, END, IGN = 1720, 1767, 1730
HFOV = 45.0
F_FULL = 960.0 / math.tan(math.radians(HFOV / 2))
HORIZON_Y0 = 352.0
MOON_AZ, MOON_EL = 8.5, 6.2
MOON_DIR = np.array([math.sin(math.radians(MOON_AZ)) * math.cos(math.radians(MOON_EL)), math.sin(math.radians(MOON_EL)),
                     math.cos(math.radians(MOON_AZ)) * math.cos(math.radians(MOON_EL))])
BOAT = np.array([6.3, 0.0, 72.0])
COAST_D = 7000.0

# wave spectrum: wavelength, amplitude, direction (rad), phase
_rng = np.random.default_rng(606)
_L = np.array([46.0, 27.0, 15.0, 8.7, 5.1, 3.0, 1.8, 1.1, 0.7])
WAVES = np.stack([_L, 0.0075 * _L ** 1.05, 0.25 + _rng.normal(size=len(_L)) * 0.5, _rng.random(len(_L)) * 6.28], 1)


@njit(inline='always', fastmath=True)
def wave_slope(x, z, t, fp, Wv):
    sx = 0.0
    sz = 0.0
    var = 0.0
    for i in range(Wv.shape[0]):
        lam = Wv[i, 0]
        k = 6.2832 / lam
        a = Wv[i, 1]
        ca = math.cos(Wv[i, 2])
        sa = math.sin(Wv[i, 2])
        om = math.sqrt(9.81 * k)
        w = 1.0 - smoothstep(0.12, 0.5, fp / lam)
        ph = k * (ca * x + sa * z) - om * t + Wv[i, 3]
        c = math.cos(ph) * a * k
        sx += w * c * ca
        sz += w * c * sa
        var += (1.0 - w) * 0.5 * (a * k) ** 2
    return sx, sz, var


@njit(fastmath=True)
def swell_h(x, z, t, Wv):
    h = 0.0
    for i in range(3):
        lam = Wv[i, 0]
        k = 6.2832 / lam
        om = math.sqrt(9.81 * k)
        h += Wv[i, 1] * math.sin(k * (math.cos(Wv[i, 2]) * x + math.sin(Wv[i, 2]) * z) - om * t + Wv[i, 3])
    return h


@njit(inline='always', fastmath=True)
def coast_elev(az):
    """Angular height (rad) of the far coast above the sea horizon at azimuth az (rad)."""
    if az > 0.075:
        return -1.0
    base = 0.010 + 0.016 * max(fbm2(az * 7.0 + 3.0, 0.5, 5.0, 91), -0.3) + 0.006 * math.sin(az * 23.0)
    fade = smoothstep(0.075, 0.035, az)
    return max(base, 0.002) * fade


@njit(parallel=True, fastmath=True)
def shade(C, S, Wv, t, LT, out, dep, skym):
    H, W = out.shape[0], out.shape[1]
    fx, fz, rx, rz = C[3], C[4], C[5], C[6]
    f, cx, cyy = C[7], C[8], C[9]
    ox, oy, oz = C[0], C[1], C[2]
    pix = 1.0 / f
    for j in prange(H):
        for i in range(W):
            vx = (i + 0.5 - cx)
            vy = (cyy - (j + 0.5))
            dx = fx * f + rx * vx
            dz = fz * f + rz * vx
            dy = vy
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n
            dy /= n
            dz /= n
            az = math.atan2(dx, dz)
            # sea horizon dips slightly below 0 because the camera is above the water
            dip = -math.sqrt(2.0 * max(oy, 0.1) / 6.371e6)
            if dy > dip:
                ce = coast_elev(az)
                if ce > 0.0 and dy < dip + ce:
                    # far coast: hazy dark land, faint moonlit rim on the ridge
                    rimv = math.exp(-((dip + ce - dy) / (2.0 * pix)))
                    out[j, i, 0] = 0.0085 + 0.004 * rimv
                    out[j, i, 1] = 0.0115 + 0.005 * rimv
                    out[j, i, 2] = 0.022 + 0.008 * rimv
                    dep[j, i] = 7000.0
                    skym[j, i] = 0.0
                    continue
                r, g, b = SK.sky_base(dx, dy, dz, S)
                mr, mg, mb = SK.moon_disk(dx, dy, dz, S, pix)
                out[j, i, 0] = r + mr
                out[j, i, 1] = g + mg
                out[j, i, 2] = b + mb
                dep[j, i] = 1e9
                skym[j, i] = 1.0
                continue
            skym[j, i] = 0.0
            tt = -oy / dy
            px = ox + dx * tt
            pz = oz + dz * tt
            fp = tt * pix / max(-dy, 0.02)
            sx, sz, var = wave_slope(px, pz, t, fp, Wv)
            nx = -sx
            ny = 1.0
            nz = -sz
            nn = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= nn
            ny /= nn
            nz /= nn
            dn = dx * nx + dy * ny + dz * nz
            rx_ = dx - 2.0 * dn * nx
            ry_ = max(dy - 2.0 * dn * ny, 0.002)
            rz_ = dz - 2.0 * dn * nz
            rn = math.sqrt(rx_ * rx_ + ry_ * ry_ + rz_ * rz_)
            rx_ /= rn
            ry_ /= rn
            rz_ /= rn
            fres = 0.02 + 0.98 * (1.0 - max(-dn, 0.0)) ** 5
            r, g, b = SK.sky_base(rx_, ry_, rz_, S)
            # moon glitter: disk lobe widened by the unresolved slope variance
            cm = rx_ * S[6] + ry_ * S[7] + rz_ * S[8]
            th = math.acos(min(max(cm, -1.0), 1.0))
            tw2 = S[12] * S[12] + 4.0 * var + 1e-6
            gl = S[13] * 7.0 * (S[12] * S[12] / tw2) * math.exp(-th * th / tw2)
            r += gl * S[9]
            g += gl * S[10]
            b += gl * S[11]
            cr = r * fres + 0.0012
            cg = g * fres + 0.0022
            cb = b * fres + 0.0032
            # fires: specular streaks
            for li in range(LT.shape[0]):
                lx = LT[li, 0] - px
                ly = LT[li, 1]
                lz = LT[li, 2] - pz
                ll = math.sqrt(lx * lx + ly * ly + lz * lz)
                lx /= ll
                ly /= ll
                lz /= ll
                c2 = rx_ * lx + ry_ * ly + rz_ * lz
                th2 = math.acos(min(max(c2, -1.0), 1.0))
                w2 = LT[li, 7] * LT[li, 7] + 4.0 * var + 1e-6
                E = LT[li, 6] * (LT[li, 7] * LT[li, 7] / w2) * math.exp(-th2 * th2 / w2) * fres * 25.0
                cr += E * LT[li, 3]
                cg += E * LT[li, 4]
                cb += E * LT[li, 5]
            dist = tt
            tr = math.exp(-dist / 9000.0)
            out[j, i, 0] = cr * tr + 0.012 * (1 - tr)
            out[j, i, 1] = cg * tr + 0.018 * (1 - tr)
            out[j, i, 2] = cb * tr + 0.040 * (1 - tr)
            dep[j, i] = dist


def camera(frame, W=1920, H=804):
    t = CM.ftime(frame)
    tilt0 = math.degrees(math.atan((HORIZON_Y0 - 402.0) / F_FULL))
    push = keys(frame, [(START, 0.0), (END, 2.5)], ease=lambda u: u)
    bob = 0.20 * math.sin(2 * math.pi * t / 5.3) + 0.06 * math.sin(2 * math.pi * t / 2.1 + 1.0)
    tl = 0.30 * math.sin(2 * math.pi * t / 4.4 + 1.0)
    k = kick(t, CM.ftime(IGN), amp=1.0, freq=5.0, decay=5.0)
    return Cam((0.0, 2.3 + bob, push), -0.5 + 0.05 * k, tilt0 + tl + 0.06 * k, HFOV, W, H)


def boat_drawing():
    d = Drawing()
    d.new_group()
    # hull: sheer line rising to the bow (left); sits ~1.1 m above the water
    d.trap((0.3, -0.15), (0.3, 1.05), 3.6, 4.2, rnd=0.05, mat=0)
    d.tri((-3.9, 1.05), (-4.9, 1.6), (-3.2, 0.2), rnd=0.02, k=0.1, mat=0)
    d.capsule((-4.8, 1.55), (3.9, 1.15), 0.05, 0.05, k=0.03, mat=5)          # gunwale
    d.new_group()
    d.trap((1.8, 1.0), (1.8, 2.9), 0.95, 0.85, rnd=0.04, mat=0)               # wheelhouse
    d.trap((1.8, 2.85), (1.8, 3.05), 1.1, 1.05, rnd=0.02, mat=0)
    d.new_group()
    d.capsule((-0.6, 1.0), (-0.6, 6.4), 0.06, 0.04, mat=4)                    # mast
    d.capsule((-0.6, 6.3), (-4.7, 1.6), 0.008, 0.008, mat=5)                  # forestay
    d.capsule((-0.6, 6.3), (3.8, 1.3), 0.008, 0.008, mat=5)                   # backstay
    d.capsule((-0.6, 2.2), (2.6, 2.4), 0.04, 0.035, mat=4)                    # boom
    return d


COAST_FIRES = [(-0.44, 1735), (-0.36, 1738), (-0.285, 1741), (-0.21, 1744), (-0.15, 1747), (-0.095, 1750),
               (-0.045, 1753), (0.0, 1756), (0.035, 1759), (-0.40, 1762), (-0.25, 1764), (-0.12, 1766)]


def render(frame, scale=0.5, ss=1.25):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    t = CM.ftime(frame)
    cam = camera(frame, W, H)
    cami = cam.scaled(ss)
    ti = CM.ftime(IGN)
    size, inten, light = F.ignite_env(t, ti)
    fl = F.flicker(t, 41)
    # boat on the swell
    bh = swell_h(BOAT[0], BOAT[2], t, WAVES) * 0.9
    roll = 0.05 * math.sin(2 * math.pi * t / 4.1 + 0.6)
    bpos = BOAT + np.array([0.0, bh - 0.15, 0.0])
    # sailor at the bow (mirrored: faces left, toward the coast)
    arm = keys(frame, [(START, 1.0), (1729, 1.0), (1731, 0.8), (1738, -0.25), (END, -0.3)], ease=smoother)
    lean = keys(frame, [(START, 0.1), (1729, 0.12), (1738, -0.05), (END, -0.05)], ease=smoother)
    sail, pts = PP.torch_person('sailor', arm=arm, lean=lean)
    off = np.array([-3.0, 1.1])           # feet relative to boat centre (screen frame)
    c, s_ = math.cos(roll), math.sin(roll)
    offr = np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1]])
    sail.rotate(-roll, pivot=(off[0], -off[1]))
    feet = PR.local_to_world(cam, bpos, offr[0], offr[1], 0.0)
    th = pts['torch_head']
    thr = np.array([c * (-th[0]) - s_ * th[1], s_ * (-th[0]) + c * th[1]])
    flare_w = PR.local_to_world(cam, feet, thr[0], thr[1], 0.0)
    flare_I = 2.8 * light * fl
    # coastal beacons (directions -> far points)
    cf = []
    for az, fr in COAST_FIRES:
        if frame >= fr - 1:
            s2, i2, l2 = F.ignite_env(t, CM.ftime(fr))
            el = coast_elev(az) - math.sqrt(2.0 * 2.3 / 6.371e6) + 0.0008
            d = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
            P = cam.pos * np.array([1, 0, 1]) + d * COAST_D + np.array([0, 2.3, 0])
            cf.append((P, i2 * F.flicker(t, fr) * min(s2 / 0.5, 1.0)))
    LT = []
    if flare_I > 0:
        LT.append([flare_w[0], flare_w[1] + 0.2, flare_w[2], *F.FIRE_LIGHT, flare_I * 0.5, 0.012, 0])
    for P, e in cf:
        LT.append([P[0], P[1], P[2], *F.FIRE_LIGHT, 0.35 * e, 0.0015, 0])
    LT = np.array(LT, np.float64).reshape(-1, 9) if LT else np.zeros((0, 9))
    S = SK.sky_params(zenith='#070B1C', horizon='#2A3866', moon_dir=MOON_DIR, moon_radius_deg=0.55, moon_I=9.0,
                      halo_I=0.06, halo_w=0.10, halo2_I=0.03, halo2_w=0.6, horizon_glow=0.3)
    out = np.zeros((cami.H, cami.W, 3), np.float32)
    dep = np.zeros((cami.H, cami.W), np.float32)
    skm = np.zeros((cami.H, cami.W), np.float32)
    shade(cami.params(), S, WAVES, t, LT, out, dep, skm)
    img = CM.downsample(out, W, H)
    depth = CM.depth_down(dep, W, H)
    skymask = CM.downsample(skm, W, H)
    SK.splat_stars(img, cam, _stars(), skymask, t=t, gain=0.8, scale=scale)
    # coastal fires on the ridge line
    for P, e in cf:
        sx, sy, z = cam.project(P)
        F.glow(img, depth, sx, sy, 0.8 * scale + 0.35, 16.0 * e * scale * scale, z=z - 10, zbias=0)
        F.halo(img, depth, sx, sy, 14 * scale, 0.02 * e, z=z - 10, zbias=0)
    moon_col = CM.lin('#9DB4D9')
    lights = [dict(dir=-MOON_DIR * np.array([1, -1, 1]), col=moon_col, I=0.12)]
    if flare_I > 0:
        lights.append(dict(pos=flare_w + np.array([0, 0.1, 0]), col=F.FIRE_LIGHT, I=flare_I, r0=0.2))
    famb = np.array([0.008, 0.010, 0.018])
    boat = boat_drawing().rotate(roll)
    FG.render(img, depth, cam, boat, bpos, lights, amb=famb, t=t)
    FG.render(img, depth, cam, sail, feet, lights, amb=famb, t=t, seed=31, flipx=True)
    if size > 0:
        _smoke().render(img, depth, cam, t, flare_w, 0.8 * light * fl, albedo=0.25, amb=(0.004, 0.005, 0.009))
        F.flame(img, depth, cam, flare_w, 0.55 * size, 0.07, t, seed=14, I=28.0 * inten, lean=0.18, zbias=0.3,
                tongues=3)
        sx, sy, z = cam.project(flare_w + np.array([0, 0.25, 0]))
        F.halo(img, depth, sx, sy, 150 * scale * size, 0.02 * light * fl, z=z, zbias=3.0)
        _sparks().render(img, depth, cam, t)
    return img


_ST = None
_SM = None
_SP = None


def _stars():
    global _ST
    if _ST is None:
        _ST = SK.make_stars(12000, 606, lum_scale=5.0)
    return _ST


class _Follow:
    """Smoke/sparks that follow the flare (origin re-anchored every frame)."""


def _smoke():
    global _SM
    if _SM is None:
        _SM = F.Smoke(91, BOAT + np.array([-3.4, 3.4, 0]), CM.ftime(IGN) + 0.1, CM.ftime(END) + 0.1, rate=7.0,
                      wind=(1.6, 0, 0.2), rise=0.9, life=3.5, r0=0.18, growth=0.55, dens=0.5)
    return _SM


def _sparks():
    global _SP
    if _SP is None:
        _SP = F.Sparks(93, BOAT + np.array([-3.4, 3.2, 0]), CM.ftime(IGN), CM.ftime(END) + 0.1, burst=160, rate=45,
                       ember_rate=6, wind=(1.8, 0.0, 0.2), radius=0.1, burst_speed=(3.0, 8.0), I=30.0)
    return _SP


FINISH = dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8, streak_strength=0.02, vignette_amount=0.25)
