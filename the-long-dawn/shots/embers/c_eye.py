"""EMBERS-C2 . E12 THE EYE ONTO NOTHING (cut C, C9, C frames 1920-2080, bars 25-26).

The book's Eye, not the films': no tower under it, no horns, no fireball. "The Eye was rimmed with fire, but was
itself glazed, yellow as a cat's, watchful and intent, and the black slit of its pupil opened on a pit, a window into
nothing." Here it is the race's own will: the storm over every forge resolves into it, and there is no one behind it.

    1920-1991  MAP's page burns through from the Deep's red glow (book_C over these frames, matte); under it the storm:
               dark billows churning round a red heart, the forges below
    1936-1990  the storm draws back from the heart in a ragged ring; the light gathers into a glazed yellow iris of
               streaming fibres, rimmed with fire; a closed seam down its middle; every tower bends toward it
    2000       (bar 26 b1, C's loudest moment) the slit opens for the first time: onto empty black. No highlight, no
               storm, no fire behind it. The rim flares, a pressure wave pushes the billows out, the towers lurch
    2000-2080  a slow push toward the slit; embers near it are drawn in and go out as they cross its edge

Built as: C's forge-towers (scene_b.Towers laid out by c3.layout_towers, bent toward the Eye by `tower_post`) and
the engine's smoke / tower embers / dust; the storm as three billboard layers (c2.storm_layer, behind the Eye and in
front of it: they parallax); the Eye as a disc in its own plane (`_eye`, per pixel: iris, fibres, crypts, collarette,
limbus, the slit, the glaze = the burning forges reflected in its lower curve); the rim of fire and the drawn-in
embers as particles in their own frame. Composited in post, back to front; the towers' occluder masks it all.
"""
import math

import cv2
import numpy as np
from numba import njit, prange

import look
from core import Camera, Frame, smoothstep, smootherstep, lerp, catmull, rng, vnoise, perlin3, ease_out
import scene_b as B
import c3
import c2
from c2 import T_EYE, T_SLIT, T_EYE_END

G = B.GROUND
EYE_C = np.array([0.0, G + 57.0, 0.0])
EYE_R = 15.5
CAM_AZ = c3.AZ0 + 0.1
# (frame, distance from the axis, height above ground, target height above ground, hfov)
CAM_KEYS = [(1880, 161.0, 22.0, 49.4, 50.0), (1920, 158.0, 22.0, 49.8, 50.0), (1960, 154.0, 22.1, 50.3, 50.0),
            (2000, 149.0, 22.2, 51.0, 50.0), (2040, 138.5, 21.8, 53.0, 49.2), (2080, 127.0, 21.4, 55.0, 48.4),
            (2120, 120.0, 21.2, 56.0, 48.0)]
LAYERS = [  # (depth toward the camera, units of EYE_R; seed; opacity sigma; which)
    (-2.2, 11.0, 2.4, 'back'),
    (0.8, 23.0, 2.2, 'front'),
    (2.5, 37.0, 1.6, 'near'),
]


# ============================================================== timing ===

def form(t):
    """0 storm .. 1 the Eye formed"""
    return float(smootherstep(T_EYE + 16, T_EYE + 70, t))


def opening(t):
    """the slit: 0 closed .. 1 open (fast, decisive; then it keeps widening a little)"""
    if t < T_SLIT:
        return 0.0
    return float(ease_out((t - T_SLIT) / 7.0, 3.0))


def flare(t):
    """the opening's shock (decays over ~10 frames); a pre-swell into it"""
    pre = 0.35 * float(smoothstep(T_SLIT - 24, T_SLIT, t)) * (1.0 if t < T_SLIT else 0.0)
    post = math.exp(-(t - T_SLIT) / 9.0) if t >= T_SLIT else 0.0
    return pre + post


def swell(t):
    """the tam-tam swell into the opening"""
    return float(smoothstep(T_SLIT - 40, T_SLIT, t)) * (1.0 if t < T_SLIT else math.exp(-(t - T_SLIT) / 30.0))


def lean(t):
    """every tower bends toward the Eye as it forms; a lurch at the opening"""
    x = 0.06 + 0.12 * float(smootherstep(T_EYE + 20, T_SLIT - 4, t))
    if t >= T_SLIT:
        x += 0.05 * float(ease_out((t - T_SLIT) / 12.0, 2.0)) + 0.012 * (t - T_SLIT) / 80.0
    return x


def slit(t):
    """(half-width at the widest, half-height), units of EYE_R"""
    o = opening(t)
    w = lerp(0.006, 0.19, o)
    if t >= T_SLIT:
        w += 0.03 * (t - T_SLIT) / 80.0
    return w, 0.86 + 0.04 * o


def brightness(t):
    return 0.7 + 0.3 * form(t) + 0.22 * swell(t) + 0.3 * flare(t)


# ============================================================ schedule ===

class EyeSched(c3.C3Sched):
    """C's forges as the race left them (c3's schedule), held past its end; bent toward the Eye"""
    end = float(T_EYE_END + 1)
    pulses = []

    def beat_pulse(self, t):
        return 0.35 * swell(t) + 0.9 * (flare(t) if t >= T_SLIT else 0.0)

    def tower_lean(self, t):
        return 0.0

    def tower_post(self, towers, i, P, t):
        b = towers.base(i)
        h = max(towers.height(i, t), 1.0)
        d = -np.array([b[0], b[2]])
        d = d / max(np.linalg.norm(d), 1e-6)
        s = np.clip((P[:, 1] - b[1]) / h, 0.0, 1.3)
        off = (lean(t) * h) * s * s
        Q = P.copy()
        Q[:, 0] += (d[0] * off).astype(Q.dtype)
        Q[:, 2] += (d[1] * off).astype(Q.dtype)
        Q[:, 1] -= (0.5 * lean(t) * off * s).astype(Q.dtype)     # a bend, not a shear: the crown comes down a little
        return Q


SCHED = EyeSched()


def use():
    B.SCHED = SCHED
    B.IGN = SCHED.ign
    B.BEATS = list(SCHED.beats)


# ============================================================== camera ===

def cam_state(t):
    k = [(f, np.array([r, y, ty, hf])) for f, r, y, ty, hf in CAM_KEYS]
    r, y, ty, hf = catmull(t, k)
    pos = np.array([r * math.cos(CAM_AZ), G + y, r * math.sin(CAM_AZ)])
    tgt = np.array([0.0, G + ty, 0.0])
    if t >= T_SLIT:                                   # one shudder as it opens (a few pixels, dying fast)
        x = t - T_SLIT
        g = 0.22 * math.exp(-x / 4.0)
        tgt = tgt + g * np.array([math.sin(2.3 * x), math.sin(3.1 * x + 1.0), 0.0])
    return pos, tgt, float(hf)


def eye_basis():
    """the Eye's plane faces the lens at the moment it opens"""
    pos, tgt, hf = cam_state(float(T_SLIT))
    n = pos - EYE_C
    n = n / np.linalg.norm(n)
    ex = np.cross(np.array([0.0, 1.0, 0.0]), n)
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    return n, ex, ey


N_EYE, EX, EY = eye_basis()


def eye_world(x, y, z=0.0):
    """eye-plane coordinates (units of EYE_R; z toward the lens) -> world"""
    x, y, z = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(z, np.float64))
    return EYE_C + EYE_R * (x[..., None] * EX + y[..., None] * EY + z[..., None] * N_EYE)


# ============================================================= the Eye ===

@njit(fastmath=True, cache=True, inline='always')
def _ss(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(parallel=True, fastmath=True, cache=True)
def _eye(X, Y, ok, aa_, t, pr, HL, out_rgb, out_a, out_void, out_bev):
    """the Eye, per pixel, from its plane coordinates X, Y (units of EYE_R).
    pr: 0 form, 1 slit half-width, 2 slit half-height, 3 brightness, 4 glaze, 5 lip, 6 seam, 7 stream, 8 flare
    HL: (K, 4) glaze highlights (x, y, sigma, intensity)"""
    H, W = X.shape
    F = pr[0]
    for j in prange(H):
        for i in range(W):
            out_rgb[j, i, 0] = 0.0
            out_rgb[j, i, 1] = 0.0
            out_rgb[j, i, 2] = 0.0
            out_bev[j, i, 0] = 0.0
            out_bev[j, i, 1] = 0.0
            out_bev[j, i, 2] = 0.0
            out_a[j, i] = 0.0
            out_void[j, i] = 0.0
            if not ok[j, i]:
                continue
            ex = X[j, i]
            ey = Y[j, i]
            rho = math.sqrt(ex * ex + ey * ey)
            if rho > 1.75:
                continue
            aa = aa_[j, i]
            cph = ex / (rho + 1e-9)
            sph = ey / (rho + 1e-9)
            # before it forms: the storm's heart, an undifferentiated glow (light only, no surface)
            glow = 1.5 * math.exp(-rho * rho / 0.55) * (1.0 - F) ** 1.5
            gr = glow * 1.0
            gg = glow * 0.36
            gb = glow * 0.1
            # the limbus, ragged
            lim = 1.0 + 0.03 * perlin3(cph * 3.0, sph * 3.0, t * 0.021 + 1.0) \
                + 0.014 * perlin3(cph * 9.0, sph * 9.0, t * 0.05 + 4.0)
            disc = 1.0 - _ss(lim - aa, lim + aa, rho)
            # ---- the rim of fire: tongues licking outward from the limbus into the storm
            s_ = (rho - lim * 0.95) / 0.78
            fire = 0.0
            if s_ > -0.2 and s_ < 1.2:
                zf = t * 0.075
                n1 = perlin3(cph * 5.0, sph * 5.0, s_ * 2.2 - zf) * 0.62 \
                    + perlin3(cph * 11.0, sph * 11.0, s_ * 4.6 - zf * 1.9 + 3.0) * 0.28 \
                    + perlin3(cph * 23.0, sph * 23.0, s_ * 9.0 - zf * 3.1 + 7.0) * 0.1
                big = perlin3(cph * 1.6, sph * 1.6, t * 0.018 + 2.0)
                tongue = n1 * 2.1 + 0.36 + 0.55 * big - s_ * (1.1 - 0.35 * pr[8])
                if tongue > 0.0:
                    fire = tongue ** 1.6 * _ss(-0.2, 0.05, s_) * (1.0 + 1.2 * pr[8])
            fk = fire * 2.3 * F
            gr += fk * 1.0
            gg += fk * (0.46 * _ss(0.65, 0.0, s_) + 0.13)
            gb += fk * (0.1 * _ss(0.5, 0.0, s_) + 0.015)
            if disc <= 0.0:
                out_rgb[j, i, 0] = gr
                out_rgb[j, i, 1] = gg
                out_rgb[j, i, 2] = gb
                continue
            u = rho / lim
            # ---- the iris: fibres streaming inward, crypts, the collarette, contraction furrows, a dark limbus
            zs = t * pr[7]
            # the fibres wander (a bend that changes along the radius) and gather in bundles
            wob = 0.09 * perlin3(cph * 3.0, sph * 3.0, u * 4.0 + 1.3) + 0.04 * perlin3(cph * 8.0, sph * 8.0, u * 9.0)
            cw = math.cos(wob)
            sw = math.sin(wob)
            qx = cph * cw - sph * sw
            qy = cph * sw + sph * cw
            f1 = perlin3(qx * 24.0, qy * 24.0, u * 2.6 + zs)
            f2 = perlin3(qx * 52.0, qy * 52.0, u * 5.4 + zs * 1.7 + 3.3)
            bund = 0.55 + 0.45 * perlin3(qx * 7.0, qy * 7.0, u * 2.0 + 6.1)
            fib = 0.5 + 0.5 * (0.66 * f1 + 0.34 * f2) / 0.66
            if fib < 0.0:
                fib = 0.0
            fc = _ss(0.06, 0.3, u)                                      # no fibre moire at the very centre
            Ef = (1.0 - fc) + fc * (0.66 + 0.55 * fib ** 1.4) * (0.8 + 0.4 * bund)
            cr = perlin3(qx * 11.0, qy * 11.0, u * 2.4 + 2.2 + zs * 0.5)
            crypt = _ss(0.3, 0.55, cr) * _ss(0.42, 0.52, u) * (1.0 - _ss(0.8, 0.9, u))
            uc = 0.37 + 0.035 * perlin3(cph * 4.0, sph * 4.0, 5.5)
            coll = math.exp(-((u - uc) / 0.022) ** 2)
            fur = math.exp(-((u - 0.63 - 0.02 * perlin3(cph * 5.0, sph * 5.0, 8.1)) / 0.012) ** 2) \
                + 0.7 * math.exp(-((u - 0.76 - 0.02 * perlin3(cph * 5.0, sph * 5.0, 9.4)) / 0.011) ** 2)
            # brightness and colour by zone: the pupillary zone yellow-gold, amber, a burnt-orange ring, the limbus dark
            E = 0.95 * math.exp(-u * u / 0.45) + 0.22
            E *= 1.0 - 0.85 * _ss(0.78, 0.99, u)
            E *= Ef * (1.0 - 0.38 * crypt) * (1.0 + 0.45 * coll) * (1.0 - 0.22 * fur)
            if u < uc:
                E *= 1.12
            E *= 1.0 + 0.6 * math.exp(-((u - uc) / 0.1) ** 2)     # EMBERS-C4: the fire round the collarette
            k1 = _ss(0.3, 0.62, u)
            k2 = _ss(0.62, 0.92, u)
            cr_ = 1.0
            cg_ = 0.78 * (1.0 - k1) + 0.42 * k1               # EMBERS-C4: more chroma (ACES was greying it to beige)
            cb_ = 0.16 * (1.0 - k1) + 0.04 * k1
            cg_ = cg_ * (1.0 - k2) + 0.3 * k2
            cb_ = cb_ * (1.0 - k2) + 0.05 * k2
            # ---- the slit (a closed seam until it opens)
            hh = pr[2]
            yy = ey / hh
            half = 0.0
            if abs(yy) < 1.0:
                half = pr[1] * (1.0 - yy * yy) ** 0.85 * (1.0 + 0.07 * perlin3(ey * 11.0, t * 0.04, 7.7))
            ds = abs(ex) - half                                          # < 0 inside the slit
            wv = 0.03 + 0.45 * pr[1]
            dk = _ss(0.0, wv, ds)
            seam = pr[6]
            E *= (1.0 - seam) + seam * (0.05 + 0.95 * dk ** 1.4)
            # the pit's rim: a dull red edge (never a bright line)
            lip = pr[5] * 0.25 * math.exp(-((ds - 0.035) / 0.03) ** 2) * _ss(0.01, 0.05, pr[1])
            void = (1.0 - _ss(-aa, aa, ds)) * _ss(0.012, 0.03, pr[1])
            if abs(yy) >= 1.0:
                void = 0.0
            # ---- the glaze: the forges' fires reflected in its lower curve, a broad sheen; never over the void
            sp = 0.0
            for k in range(HL.shape[0]):
                dx = ex - HL[k, 0]
                dy = ey - HL[k, 1]
                sp += HL[k, 3] * math.exp(-(dx * dx + dy * dy) / (HL[k, 2] * HL[k, 2]))
            # the rim of fire reflected in the glaze: a broken arc high on the left, and a softer sheen
            aph = math.atan2(ey, ex)
            arc = math.exp(-((rho - 0.8) / 0.035) ** 2) * math.exp(-((aph - 2.2) / 0.42) ** 2) \
                * (0.6 + 0.4 * perlin3(cph * 9.0, sph * 9.0, t * 0.05))
            sh = 0.55 * arc + 0.05 * math.exp(-((ex + 0.3) ** 2 + (ey - 0.4) ** 2) / 0.12)
            gz = pr[4] * (sp + sh) * (1.0 - _ss(0.9, 1.02, u))
            b = pr[3] * (1.0 + 0.25 * pr[8])
            Fi = F ** 1.3
            ir = (E * cr_ * b + lip * 0.3) * Fi + gz
            ig = (E * cg_ * b + lip * 0.04) * Fi + gz * 0.86
            ib = (E * cb_ * b + lip * 0.01) * Fi + gz * 0.62
            a = disc * F
            out_rgb[j, i, 0] = (gr * (1.0 - a) + ir * disc) * (1.0 - void)
            out_rgb[j, i, 1] = (gg * (1.0 - a) + ig * disc) * (1.0 - void)
            out_rgb[j, i, 2] = (gb * (1.0 - a) + ib * disc) * (1.0 - void)
            out_a[j, i] = a
            out_void[j, i] = void * F
            # the pit's far wall: a thin dark-red bevel inside one edge, then nothing
            if ds < 0.0 and void > 0.0:
                bv = 0.2 * math.exp(ds / 0.016) * (0.6 + 0.4 * _ss(-0.2, 0.2, ex)) * b * F
                out_bev[j, i, 0] = bv
                out_bev[j, i, 1] = bv * 0.22
                out_bev[j, i, 2] = bv * 0.05


# ============================================================ particles ===

class RimFire:
    """the rim of fire: flame particles born on the limbus, licking outward into the storm, in tongues"""

    def __init__(self, seed=4401, n=40000):
        r = rng(seed)
        self.n = n
        self.tb = r.uniform(T_EYE - 30, T_EYE_END + 2, n)
        self.life = r.uniform(7.0, 20.0, n)
        self.phi = r.uniform(0, 2 * np.pi, n)
        self.r0 = 0.99 + 0.025 * r.standard_normal(n)
        self.vr = r.uniform(0.006, 0.02, n)
        self.vt = r.normal(0, 0.0025, n)
        self.dz = r.normal(0, 0.04, n)
        self.E = r.lognormal(0, 0.5, n)
        self.sz = r.uniform(0.006, 0.014, n)

    def pos(self, idx, tq):
        a = np.maximum(tq - self.tb[idx], 0.0)
        fl = flare(float(tq))
        rr = self.r0[idx] + self.vr[idx] * a * (1.0 + 2.2 * fl) + 0.0007 * a * a
        ph = self.phi[idx] + self.vt[idx] * a
        Q = np.stack([np.cos(ph) * 3.0, np.sin(ph) * 3.0, np.full(len(idx), 0.05 * tq)], 1)
        w = vnoise(Q, 1.0, (1.3, 2.1, 0.7), 2)
        rr = rr + 0.02 * w[:, 0] * a / 10.0
        ph = ph + 0.015 * w[:, 1] * a / 10.0
        return eye_world(rr * np.cos(ph), rr * np.sin(ph), self.dz[idx]), rr

    def emit(self, ctx, fr):
        t = ctx.t
        F = form(t)
        if F <= 0.01:
            return
        age = t - self.tb
        m = (age >= 0) & (age < self.life)
        idx = np.nonzero(m)[0]
        if len(idx) == 0:
            return
        P0, _ = self.pos(idx, ctx.t0)
        P1, rr = self.pos(idx, ctx.t1)
        u = age[idx] / self.life[idx]
        ph = self.phi[idx]
        Q = np.stack([np.cos(ph) * 2.6, np.sin(ph) * 2.6, np.full(len(idx), 0.045 * t)], 1)
        from core import snoise
        tongue = np.clip(0.5 + 0.9 * snoise(Q, 1.0, (4.4, 0.3, 1.9), 2), 0.05, 1.4) ** 2
        cam = ctx.cam
        z = np.maximum((P1 - cam.pos) @ cam.R[2], 1.0)
        rpx = cam.f_px(1920) * EYE_R / float((EYE_C - cam.pos) @ cam.R[2])
        k = (rpx / 150.0) ** 2
        e = self.E[idx] * (1.0 - u) ** 1.4 * tongue * F * brightness(t) * (1.0 + 1.5 * flare(t)) * 2.2 * k
        col = look.blackbody(np.clip(0.97 - 0.5 * u - 0.08 * (rr - 1.0) * 10.0, 0.35, 1.0))
        rw = EYE_R * self.sz[idx] * (1.0 + 1.2 * u)
        fr.splat(P0, P1, rw, e, col, ctx.cam0, ctx.cam1, profile=1)


class EyeEmbers:
    """embers in the storm round the Eye; once it opens, the near ones are drawn into the slit and go out"""

    def __init__(self, seed=4413, n=9000):
        r = rng(seed)
        self.n = n
        self.r0 = 1.12 + 2.4 * r.random(n) ** 1.3
        self.ph0 = r.uniform(0, 2 * np.pi, n)
        self.z0 = r.uniform(-0.8, 0.6, n)
        self.w = 0.004 / self.r0 ** 1.2 * r.uniform(0.6, 1.4, n)
        self.E = r.lognormal(0, 0.7, n)
        self.T = r.uniform(0.45, 0.85, n)
        self.fq = r.uniform(0.1, 0.4, n)
        self.fp = r.uniform(0, 6.28, n)
        # drawn in: the nearest go first
        self.ts = T_SLIT + 2.0 + 55.0 * np.clip((self.r0 - 1.12) / 1.6, 0, 1) ** 1.2 + r.uniform(0, 8, n)
        self.ty = r.uniform(-0.7, 0.7, n)

    def xyz(self, tq):
        ph = self.ph0 + self.w * (tq - T_EYE)
        x = self.r0 * np.cos(ph)
        y = self.r0 * np.sin(ph)
        z = self.z0.copy()
        a = np.clip((tq - self.ts) / 26.0, 0.0, 1.0) ** 2.2 * (self.r0 < 2.9)
        # toward a point inside the slit, sliding into it
        x = x * (1 - a)
        y = y * (1 - a) + self.ty * slit(tq)[1] * a
        z = z * (1 - a)
        return x, y, z, a

    def emit(self, ctx, fr):
        t = ctx.t
        x0, y0, z0, _ = self.xyz(ctx.t0)
        x1, y1, z1, a = self.xyz(ctx.t1)
        P0 = eye_world(x0, y0, z0)
        P1 = eye_world(x1, y1, z1)
        fl = 0.6 + 0.4 * np.sin(self.fq * t + self.fp)
        e = self.E * fl * 0.9 * (0.3 + 0.7 * form(t)) * (1.0 + 1.2 * a)
        col = look.blackbody(self.T + 0.15 * a)
        fr.splat(P0, P1, 0.05, e, col, ctx.cam0, ctx.cam1, profile=1)


# =============================================================== shot ===

class EyeShot:
    def __init__(self):
        use()
        self._cache = {}
        self.rim = RimFire()
        self.embers = EyeEmbers()

    def _get(self, name, make):
        if name not in self._cache:
            use()
            self._cache[name] = make()
        return self._cache[name]

    @property
    def towers(self):
        def mk():
            use()
            return c3.layout_towers(B.Towers())
        return self._get('towers', mk)

    def camera(self, t):
        use()
        pos, tgt, hf = cam_state(t)
        focus = float(np.linalg.norm(EYE_C - pos))
        return Camera(pos, tgt, hfov=hf, focus=focus, aperture=0.02)

    def render_opts(self, f):
        return dict(bokeh_pow=0.25, bokeh_cap=1.6, fog_start=90.0, fog_len=120.0, near=0.3)

    def light(self, t):
        return EYE_C.copy(), np.array([1.0, 0.62, 0.26]), 95.0 * brightness(t)

    def emit(self, ctx):
        use()
        t = ctx.t
        lp, lc, lpw = self.light(t)
        tw = self.towers
        tw.prepare(ctx)
        dust = self._get('dust', B.Dust)
        tsm = self._get('tsmoke', lambda: B.TowerSmoke(tw))
        tem = self._get('tembers', lambda: B.TowerEmbers(tw))
        dust.emit(ctx)
        tw.emit(ctx, lp, lc, lpw)
        tem.emit(ctx)
        tsm.emit(ctx, lp, lc, lpw)
        # the Eye's own particles go to their own frame (composited inside the storm, under its front layers)
        ctx.fr_eye = Frame(ctx.scale)
        ctx.fr_eye.prm[:] = ctx.fr.prm
        ctx.fr_eye.prm[6] = 1e9
        self.rim.emit(ctx, ctx.fr_eye)
        self.embers.emit(ctx, ctx.fr_eye)

    def highlights(self, ctx):
        """the burning forges reflected in the Eye's glaze (a convex cornea of radius 1.7 R behind the iris)"""
        t = ctx.t
        tw = self.towers
        cam = ctx.cam
        v = cam.pos - EYE_C
        v /= np.linalg.norm(v)
        out = []
        for i in range(tw.k_all):
            h = tw.height(i, t)
            if h < 3.0:
                continue
            top = tw.base(i) + np.array([0.0, h, 0.0])
            d = top - EYE_C
            d /= np.linalg.norm(d)
            n = v + d
            n /= np.linalg.norm(n)
            x, y = float(n @ EX) * 1.7, float(n @ EY) * 1.7
            if x * x + y * y > 0.92:
                continue
            fl = 1.0 + 0.6 * math.sin(0.37 * t + 1.7 * i) * math.sin(0.11 * t + i)
            out.append((x, y, 0.016 + 0.006 * (i % 3), 0.5 * fl * (1.0 + 1.5 * flare(t))))
        if not out:
            out = [(9.0, 9.0, 0.01, 0.0)]
        return np.array(out, np.float64)

    def post(self, ctx, hdr):
        t = ctx.t
        W, H = ctx.fr.W, ctx.fr.H
        cam = ctx.cam
        F = form(t)
        fl = flare(t)
        br = brightness(t)
        f_px = cam.f_px(W)
        # ---- the storm behind the Eye
        sky = np.zeros((H, W, 3), np.float32)
        comp_front = []
        for dz, seed, sigma, which in LAYERS:
            C = EYE_C + N_EYE * (dz * EYE_R)
            X, Y, ok = c2.plane_coords(cam, W, H, C, EX, EY, N_EYE, EYE_R)
            pr = np.array([seed, 0.42 if which != 'near' else 0.36, math.exp(0.0042 * (t - T_EYE)),
                           0.55, F, (1.02 + 0.08 * F) if which == 'back' else 0.97, 0.16,
                           1.25 if which == 'back' else 1.18, 3.6, 0.02,
                           0.012, fl if t >= T_SLIT else 0.0, 1.0 if which != 'near' else 0.5], np.float64)
            if which == 'back':
                pr[9] = 0.08
                lp = np.array([2.2, 1.35, 1.6, 0.25, sigma, 0.1, -dz, br, 0.06, 0.07, fl], np.float64)
            elif which == 'front':
                pr[9] = -0.02
                lp = np.array([1.45, 1.35, 1.0, 0.45, sigma, 0.08, -dz, br, 0.06, 0.04, fl], np.float64)
            else:
                pr[9] = -0.16
                lp = np.array([1.1, 1.6, 0.5, 0.45, sigma, 0.08, -dz, br, 0.07, 0.03, fl], np.float64)
            rgb = np.empty((H, W, 3), np.float32)
            a = np.empty((H, W), np.float32)
            c2.storm_layer(X, Y, ok, float(t), pr, lp, rgb, a)
            if which == 'back':
                sky = rgb
            else:
                comp_front.append((rgb, a))
        # ---- the Eye in its plane
        X, Y, ok = c2.plane_coords(cam, W, H, EYE_C, EX, EY, N_EYE, EYE_R)
        zc = float((EYE_C - cam.pos) @ cam.R[2])
        aa = np.full((H, W), 1.3 * zc / (f_px * EYE_R), np.float32)
        w, hh = slit(t)
        seam = float(smoothstep(0.55, 0.9, F))
        pr = np.array([F, w, hh, br, float(smoothstep(0.7, 1.0, F)), 1.4 + 2.0 * fl, seam, 0.012, fl], np.float64)
        erg = np.empty((H, W, 3), np.float32)
        ea = np.empty((H, W), np.float32)
        ev = np.empty((H, W), np.float32)
        eb = np.empty((H, W, 3), np.float32)
        _eye(X, Y, ok, aa, float(t), pr, self.highlights(ctx), erg, ea, ev, eb)
        sky = sky * (1.0 - ea[..., None]) + erg
        sky = sky + ctx.fr_eye.resolve()
        sky = sky * (1.0 - ev[..., None]) + eb                 # the window into nothing: nothing behind it shows
        for rgb, a in comp_front:
            sky = sky * (1.0 - a[..., None]) + rgb
        # ---- the towers stand in front of all of it
        vis = np.ones((H, W), np.float32)
        if ctx.fr.occ is not None:
            A = ctx.fr.occ[1][0]
            Af = cv2.resize(A, (A.shape[1] * 2, A.shape[0] * 2), interpolation=cv2.INTER_LINEAR)[:H, :W]
            vis = 1.0 - np.clip(Af, 0.0, 1.0)
        return hdr + sky * vis[..., None]

    def finish_opts(self, f):
        fl = flare(float(f))
        return dict(exposure=1.0, bloom_strength=0.09 + 0.05 * fl, bloom_threshold=1.1, streak_strength=0.0,
                    vignette_amount=0.28)
