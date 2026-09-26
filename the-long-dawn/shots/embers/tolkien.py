"""Cut C (the Tolkien cut): the Ring, the Eye, and the Ring in the grasp. Our own designs throughout.

THE RING (562-880). The thinking fire does not open into a crown of flame: it is forged into a plain, heavy gold
band. A white-hot front runs once round the circle (596-622) leaving gold behind it as it cools; then fine lines of
fire burn up out of the metal (616-640) and settle into an inscription in an invented flowing script
(inscription.py; outside and in). The band is ember-light, not chrome: a warm self-glow, a polished highlight
where it would catch the light from above, brighter edges. It floats above the towers through the race, turning
slowly so the inscription travels, and stays gold while the world bleeds crimson.

THE EYE (836-880). As the storm grows the Ring turns to face the lens (scene_b.crown_tilt) and burns from gold to
fire; within it the Eye resolves: an iris of flame whose fibres are drawn slowly inward to a vertical slit of real
darkness (a post-process mask removes the light behind it), rimmed with white heat. It does not blink or blaze; it
narrows a little, and holds.

THE GRASP (960-1039). The Ring, small and bright, hangs in the dark eye of the storm; the hand closes on it
(the hand's own occlusion hides it as the fingers shut).
"""
import math
import os

import cv2
import numpy as np

import look
from core import smoothstep, smootherstep, lerp, vnoise, snoise, rng, Frame
import scene_b as B
import inscription as INS

C_GOLD = look.hexrgb('#F7B23E')
C_PALE = look.hexrgb(look.PALETTE['accord_pale'])
C_AMBER = look.hexrgb(look.PALETTE['accord_amber'])
C_RED = look.hexrgb(look.PALETTE['race_red'])

R_RACE = 5.0            # centreline radius of the band (world units) in the race
HB = 0.95               # half-height of the band (along its axis)
TB = 0.4                # half-thickness (radial)
PEXP = 3.2              # superellipse exponent of the section (a heavy, softly squared band)
EYE_R = 6.6             # the band's radius once it has become the Eye's iris
GRASP_R = 2.6           # in the storm's eye, for the grasp

T_FORGE0, T_FORGE1 = 596.0, 622.0     # the white-hot front runs round the circle
T_WRITE0, T_WRITE1 = 616.0, 642.0     # the inscription burns up out of the metal
T_EYE0, T_EYE1 = 836.0, 858.0         # the Eye resolves


def _section(psi):
    """superellipse section: returns r (radial offset), y (axial), and the outward normal (nr, ny)"""
    c, s = np.cos(psi), np.sin(psi)
    e = 2.0 / PEXP
    r = TB * np.sign(c) * np.abs(c) ** e
    y = HB * np.sign(s) * np.abs(s) ** e
    nr = np.sign(r) * np.abs(r / TB) ** (PEXP - 1) / TB
    ny = np.sign(y) * np.abs(y / HB) ** (PEXP - 1) / HB
    ln = np.sqrt(nr * nr + ny * ny) + 1e-12
    return r, y, nr / ln, ny / ln


def _outer_r(y):
    return TB * np.clip(1.0 - np.abs(y / HB) ** PEXP, 0, 1) ** (1.0 / PEXP)


class RingModel:
    """Band + inscription as points in ring-local coordinates (axis = +Y), for a unit scale of R_RACE."""

    def __init__(self, seed=404, n_band=150000, n_out=90000, n_in=50000):
        r = rng(seed)
        # --- band surface, area-uniform
        psi = np.linspace(0, 2 * np.pi, 4001)
        rr, yy, _, _ = _section(psi)
        ds = np.hypot(np.gradient(rr), np.gradient(yy))
        w = ds * (R_RACE + rr)
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        ps = np.interp(r.random(n_band), cdf, psi)
        th = r.uniform(0, 2 * np.pi, n_band)
        self.b_psi, self.b_th = ps, th
        self.b_r, self.b_y, self.b_nr, self.b_ny = _section(ps)
        self.b_rnd = r.random(n_band)
        area = 2 * np.pi * float(w.sum())            # ds is per sample step: sum, don't integrate over psi
        self.b_area = area / n_band
        # --- inscription: outside (read from without) and inside (a second line, read from within)
        em = 1.28 / (INS.ASC - INS.DSC)
        mid = 0.5 * (INS.ASC + INS.DSC)
        self.em = em
        circ_out = 2 * np.pi * (R_RACE + TB)
        circ_in = 2 * np.pi * (R_RACE - TB)
        cache = os.path.join(os.path.dirname(__file__), '..', '..', 'renders', 'embers_C', 'cache')
        os.makedirs(cache, exist_ok=True)
        f = os.path.join(cache, 'inscription_v1.npz')
        if os.path.exists(f):
            d = np.load(f)
            m1, m2, e1, e2, b1, b2 = d['m1'], d['m2'], float(d['e1']), float(d['e2']), int(d['b1']), int(d['b2'])
        else:
            m1, e1, b1 = INS.render_strip(seed=7, length=circ_out / em, em=120)
            m2, e2, b2 = INS.render_strip(seed=23, length=circ_in / em, em=120)
            np.savez_compressed(f, m1=m1, m2=m2, e1=e1, e2=e2, b1=b1, b2=b2)
        u1, v1 = INS.strip_points(m1, n_out, r, e1, b1)
        u2, v2 = INS.strip_points(m2, n_in, r, e2, b2)
        # outer line: u -> angle, v -> height on the domed outer face
        y1 = (v1 - mid) * em
        self.o_th = 2 * np.pi * u1 * (m1.shape[1] / e1 * em) / circ_out
        self.o_y = y1
        self.o_r = _outer_r(y1) + 0.012
        # inner line (mirrored so that it runs the same way when seen through the ring)
        y2 = (v2 - mid) * em
        self.i_th = -2 * np.pi * u2 * (m2.shape[1] / e2 * em) / circ_in
        self.i_y = y2
        self.i_r = -(_outer_r(y2) + 0.012)
        self.o_rnd = r.random(n_out)
        self.i_rnd = r.random(n_in)
        # each letter point stands for an equal share of the inked area (world units^2)
        self.o_area = float(m1.sum()) / (e1 * e1) * em * em / n_out
        self.i_area = float(m2.sum()) / (e2 * e2) * em * em / n_in
        print('ring: band', n_band, 'inscription', n_out, n_in)

    def band(self, R, spin):
        th = self.b_th + spin
        c, s = np.cos(th), np.sin(th)
        rad = R + self.b_r
        P = np.stack([rad * c, self.b_y, rad * s], 1)
        N = np.stack([self.b_nr * c, self.b_ny, self.b_nr * s], 1)
        return P, N, self.b_th

    def letters(self, R, spin, which):
        if which == 0:
            th, y, rr = self.o_th + spin, self.o_y, self.o_r
            nsgn = 1.0
        else:
            th, y, rr = self.i_th + spin, self.i_y, self.i_r
            nsgn = -1.0
        c, s = np.cos(th), np.sin(th)
        rad = R + rr
        P = np.stack([rad * c, y, rad * s], 1)
        # normal of the face they are written on (outward for the outer line, inward for the inner)
        N = np.stack([nsgn * c, np.zeros_like(c), nsgn * s], 1)
        return P, N, (self.o_th if which == 0 else self.i_th)


def _sheen(N, V, up_light):
    """polished-metal cue for an emissive band: a highlight where the reflected view ray points at the light
    above, and a Fresnel-ish brightening toward grazing"""
    ndv = np.clip((N * V).sum(1), 0, 1)
    Rv = 2 * ndv[:, None] * N - V
    hl = np.clip(Rv @ up_light, 0, 1) ** 24
    fres = (1 - ndv) ** 3
    return ndv, hl, fres


class Ring:
    def __init__(self):
        self.M = RingModel()

    # ------------------------------------------------------------- state over time
    @staticmethod
    def radius(t):
        if t >= 960:
            return GRASP_R
        return R_RACE + (EYE_R - R_RACE) * float(smootherstep(T_EYE0, T_EYE1 + 6, t))

    @staticmethod
    def spin(t):
        if t >= 960:
            return 0.9 + 0.006 * (t - 960)
        return 0.011 * (t - 562.0)

    @staticmethod
    def frame(t):
        """local->world rotation and centre"""
        if t >= 960:
            C = B.crown_centre(960.0)
            import scene_c as SC
            from core import Camera
            # it hangs in the storm's eye while the hand rises; hidden behind the reaching fingers it is drawn into
            # the hollow of the palm, where it shows in the opening of the grip as fingers and thumb close over it
            pos, tgt = SC.cam_grasp(t)
            cr = Camera(pos, tgt).R
            w = float(smootherstep(1004.0, 1018.0, t))
            C = C + w * (5.0 * cr[0] - 5.0 * cr[1])
            # face partly toward the grasp camera, so it reads as a ring (not a line)
            d = SC.grasp_dir()
            a = math.atan2(-d[2], -d[0])
            Rm = B._rot_about(a, math.radians(85.0))
            return Rm, C
        return B.crown_tilt(t), B.crown_centre(t)

    def emit(self, ctx):
        t = ctx.t
        if not ((560 <= t < 880) or (960 <= t < 1040)):
            return
        grasp = t >= 960
        s = 1.0 if grasp else float(smoothstep(T_FORGE0 - 4, T_FORGE0 + 2, t))
        if s <= 0:
            return
        cam = ctx.cam
        fpx = cam.f_px(1920)
        M = self.M
        R0, R1, Rm = self.radius(ctx.t0), self.radius(ctx.t1), self.radius(t)
        sc0, sc1, scm = R0 / R_RACE, R1 / R_RACE, Rm / R_RACE
        if grasp:
            sc0 = sc1 = scm = GRASP_R / R_RACE
        Rot0, C0 = self.frame(ctx.t0)
        Rot1, C1 = self.frame(ctx.t1)
        Rotm, Cm = self.frame(t)
        up_light = np.array([0.0, 1.0, 0.0])
        red = B.redness(t) if not grasp else 1.0
        eye = 0.0 if grasp else float(smootherstep(T_EYE0 - 4, T_EYE1, t))     # gold -> fire (the iris)
        glow = 1.0 + 0.15 * math.sin(2 * math.pi * (t - 480) / 80.0)            # it breathes with the fire

        def place(P, Rot, C, sc):
            return C + (P * sc) @ Rot.T

        # ---------------- the band
        P, N, th = M.band(R_RACE, self.spin(t))
        P0 = place(M.band(R_RACE, self.spin(ctx.t0))[0], Rot0, C0, sc0)
        P1 = place(M.band(R_RACE, self.spin(ctx.t1))[0], Rot1, C1, sc1)
        Pw = place(P, Rotm, Cm, scm)
        Nw = N @ Rotm.T
        V = cam.pos[None, :] - Pw
        dist = np.linalg.norm(V, axis=1)
        V /= dist[:, None]
        ndv, hl, fres = _sheen(Nw, V, up_light)
        face = smoothstep(-0.02, 0.08, (Nw * V).sum(1))
        z = np.maximum((Pw - cam.pos) @ cam.R[2], 0.5)
        # forging: a white-hot front runs round the circle; ahead of it there is only fire
        if grasp:
            age = np.full(len(th), 99.0)
        else:
            frac = np.mod(th / (2 * np.pi) + 0.15, 1.0)
            t_s = T_FORGE0 + (T_FORGE1 - T_FORGE0) * frac
            age = t - t_s
        born = smoothstep(-1.0, 0.6, age)
        hot = np.exp(-np.maximum(age, 0) / 7.0) * born            # still glowing from the forge
        front = np.exp(-np.maximum(age, 0) / 1.8) * born
        shimmer = 1 + 0.12 * np.sin(0.9 * t + 17 * M.b_rnd) * np.sin(0.31 * t + 29 * M.b_rnd)
        L = (0.55 + 5.0 * hl + 1.6 * fres) * (0.85 + 0.3 * M.b_rnd) * shimmer * glow
        col = C_GOLD[None, :] * (1 - 0.35 * hl[:, None]) + C_PALE[None, :] * (0.35 * hl[:, None])
        colE = col * L[:, None] * (1.3 if not grasp else 6.0)     # small and bright in the storm's eye
        # the forging front: white heat, cooling through orange into gold
        hc = look.blackbody(0.5 + 0.47 * front) * (1.6 + 3.2 * front + 1.2 * hot)[:, None]
        colE = colE * (1 - hot[:, None]) + hc * hot[:, None]
        # the Eye: the gold burns into a ring of fire
        if eye > 0:
            flick = 0.75 + 0.5 * np.sin(1.3 * t + 37 * M.b_rnd) * np.sin(0.41 * t + 11 * M.b_rnd)
            fire = look.blackbody(0.53 + 0.1 * hl + 0.1 * flick)
            colE = colE * (1 - 0.9 * eye) + fire * ((2.3 + 1.8 * hl) * flick * eye)[:, None]
        colE = colE * (born * face * s)[:, None]
        pa = M.b_area * scm * scm * np.clip(ndv, 0.08, 1) * (fpx / z) ** 2
        E = colE.max(1) * pa
        m = E > 1e-7
        rw = math.sqrt(M.b_area / math.pi) * scm * 1.4
        ctx.fr.splat(P0[m], P1[m], rw, E[m], colE[m] / (colE[m].max(1)[:, None] + 1e-12), ctx.cam0, ctx.cam1,
                     zref=0.0, profile=1)
        # ---------------- the inscription: fine lines of fire, outside and in
        for which in (0, 1):
            P, N, th0 = M.letters(R_RACE, self.spin(t), which)
            P0 = place(M.letters(R_RACE, self.spin(ctx.t0), which)[0], Rot0, C0, sc0)
            P1 = place(M.letters(R_RACE, self.spin(ctx.t1), which)[0], Rot1, C1, sc1)
            Pw = place(P, Rotm, Cm, scm)
            Nw = N @ Rotm.T
            V = cam.pos[None, :] - Pw
            dist = np.linalg.norm(V, axis=1)
            V /= dist[:, None]
            ndv = (Nw * V).sum(1)
            face = smoothstep(0.0, 0.15, ndv)
            z = np.maximum((Pw - cam.pos) @ cam.R[2], 0.5)
            rnd = M.o_rnd if which == 0 else M.i_rnd
            if grasp:
                write = np.ones(len(th0))
                flare = np.zeros(len(th0))
            else:
                # the letters burn up out of the metal along the circle, then settle
                frac = np.mod(np.abs(th0) / (2 * np.pi) + 0.4, 1.0)
                tw = T_WRITE0 + (T_WRITE1 - T_WRITE0) * frac
                a = t - tw
                write = smoothstep(-1.0, 1.5, a)
                flare = np.exp(-np.maximum(a, 0) / 7.0) * write
            # a slow wave of heat travels along the words
            wave = 0.6 + 0.4 * np.sin(np.abs(th0) * 5.0 - 0.09 * t) * np.sin(np.abs(th0) * 2.0 + 0.05 * t + 1.3)
            fl = 1 + 0.25 * np.sin(1.7 * t + 23 * rnd)
            Lt = (4.5 * wave * fl + 12.0 * flare) * write * (1.0 + 1.2 * eye) * glow
            T = np.clip(0.5 + 0.1 * wave + 0.35 * flare + 0.08 * eye, 0, 1)
            col = look.blackbody(T)
            colE = col * (Lt * face * s)[:, None]
            pa = (M.o_area if which == 0 else M.i_area) * scm * scm * np.clip(ndv, 0.1, 1) * (fpx / z) ** 2
            # the letters' own area: they are fine lines, so each point stands for a small patch
            E = colE.max(1) * pa
            m = E > 1e-7
            ctx.fr.splat(P0[m], P1[m], 0.012 * scm, E[m], colE[m] / (colE[m].max(1)[:, None] + 1e-12),
                         ctx.cam0, ctx.cam1, zref=0.0)
        # ---------------- a soft light of its own (small in the race, a clear beacon in the storm's eye)
        H = np.array([Cm, Cm])
        if grasp:
            dz = float(np.linalg.norm(Cm - cam.pos))
            eh = np.array([900.0, 380.0]) * (1 + 0.1 * math.sin(0.4 * t)) * (dz / 10.0) ** 2 * 0.036
            rr = np.array([GRASP_R * 1.6, GRASP_R * 4.0])
            hc = np.array([C_GOLD * 0.7 + C_PALE * 0.3, C_GOLD])
        else:
            eh = np.array([160.0, 90.0]) * s * (1 - 0.7 * eye)
            rr = np.array([R_RACE * 1.2, R_RACE * 2.6]) * scm
            hc = np.array([C_GOLD, C_AMBER])
        ctx.fr.splat(H, H, rr, eh, hc, ctx.cam0, ctx.cam1, profile=1)


# ===================================================================== THE EYE ===

class Eye:
    """Iris fibres of flame drawn inward to a vertical slit of darkness (resolves 836-858, holds to the cut)."""

    def __init__(self, seed=777, n_fib=110, per=460):
        r = rng(seed)
        n = n_fib * per
        self.n = n
        fib = np.repeat(np.arange(n_fib), per)
        self.phi = (fib + r.normal(0, 0.12, n)) * 2 * np.pi / n_fib
        self.fheat = (0.25 + 0.75 * r.random(n_fib) ** 1.5)[fib]      # some fibres burn, some smoulder
        self.ph = r.random(n)
        self.v = r.uniform(0.018, 0.032, n)            # inward speed (fraction of the iris per frame)
        self.E = r.lognormal(0, 0.5, n)
        self.rnd = r.random(n)
        self.wig = r.normal(0, 1, n)
        # flames licking off the burning ring
        nf = 16000
        self.fl_a = r.uniform(0, 2 * np.pi, nf)
        self.fl_ph = r.random(nf)
        self.fl_life = r.uniform(7.0, 16.0, nf)
        self.fl_v = r.uniform(0.08, 0.2, nf)
        self.fl_E = r.lognormal(0, 0.6, nf)
        # slit rim
        m = 9000
        self.rim_s = r.uniform(-1, 1, m)
        self.rim_side = np.where(r.random(m) < 0.5, -1.0, 1.0)
        self.rim_rnd = r.random(m)

    @staticmethod
    def resolve(t):
        return float(smootherstep(T_EYE0, T_EYE1, t))

    @staticmethod
    def slit(t):
        """half-height and half-width of the slit, as fractions of the iris radius"""
        k = float(smootherstep(T_EYE0 + 4, T_EYE1 + 4, t))
        narrow = float(smoothstep(854, 878, t))
        return 0.86 * (0.25 + 0.75 * k), (0.02 + 0.235 * k) * (1 - 0.32 * narrow)

    @staticmethod
    def basis(t, cam):
        Rot = B.crown_tilt(t)
        C = B.crown_centre(t)
        n = Rot[:, 1]
        up = cam.R[1]
        e_up = up - (up @ n) * n
        e_up /= np.linalg.norm(e_up)
        e_side = np.cross(n, e_up)
        return C, n, e_up, e_side

    def iris_radius(self, t):
        return (Ring.radius(t) - TB) * 0.97

    def emit(self, ctx):
        t = ctx.t
        k = self.resolve(t)
        ctx.eye_k = k
        if k <= 0 or t >= 880:
            return
        cam = ctx.cam
        fr = Frame(ctx.scale)
        fr.prm[:] = ctx.fr.prm
        ctx.fr_eye = fr
        Ri = self.iris_radius(t)
        hh, hw = self.slit(t)

        def pts(tq):
            C, n, eu, es = self.basis(tq, cam)
            u = ((tq - 700.0) * self.v + self.ph) % 1.0             # 0 at the rim -> 1 at the pupil
            rho = Ri * (1.0 - 0.9 * u ** 0.9)
            ph = self.phi + 0.35 * u + 0.06 * self.wig * u          # a slight spiral, the storm's turn
            x = rho * np.cos(ph)
            y = rho * np.sin(ph)
            return C + x[:, None] * es[None, :] + y[:, None] * eu[None, :], x, y, u
        P0, _, _, u0 = pts(ctx.t0)
        P1, x, y, u = pts(ctx.t1)
        wrap = u >= u0                                              # no streak from the pupil back to the rim
        # fibres die at the edge of the slit
        s_ = np.clip(y / (hh * Ri), -1, 1)
        edge = hw * Ri * np.clip(1 - s_ * s_, 0, 1) ** 0.75
        out = np.abs(x) - edge
        alive = (out > 0.0) | (np.abs(y) > hh * Ri)
        near = np.exp(-np.maximum(out, 0) / (0.12 * Ri))            # hottest where they fall into the dark
        T = np.clip(0.36 + 0.36 * u ** 1.4 * self.fheat + 0.1 * near, 0, 0.84)
        col = look.blackbody(T)
        fl = 1 + 0.3 * np.sin(1.1 * t + 31 * self.rnd)
        e = self.E * self.fheat * (0.3 + 1.3 * u ** 2 + 0.9 * near) * fl * smoothstep(0.0, 0.08, u) * alive * wrap \
            * k * 2.6
        fr.splat(P0, P1, 0.02, e, col, ctx.cam0, ctx.cam1, zref=60.0)
        # the ring burns: short tongues of flame lick outward from it, into the storm
        Rr = Ring.radius(t) + TB

        def flames(tq):
            C, n, eu, es = self.basis(tq, cam)
            age = ((tq - 700.0) / self.fl_life + self.fl_ph) % 1.0 * self.fl_life
            d = Rr + self.fl_v * age
            a = self.fl_a + 0.004 * age
            P = C + d[:, None] * (np.cos(a)[:, None] * es[None, :] + np.sin(a)[:, None] * eu[None, :])
            P = P + n[None, :] * (0.05 * age)[:, None]
            return P, age
        F0, a0 = flames(ctx.t0)
        F1, a1 = flames(ctx.t1)
        w = vnoise(F1 * 0.4 + np.array([0.0, 0.0, 0.05 * t]), 0.6, (0, 0, 0), 1)
        F0 = F0 + w * (0.06 * a0)[:, None]
        F1 = F1 + w * (0.06 * a1)[:, None]
        uu = a1 / self.fl_life
        ef = self.fl_E * (1 - uu) ** 1.3 * smoothstep(0.0, 1.5, a1) * (a1 >= a0) * 4.0 * k
        fr.splat(F0, F1, 0.03, ef, look.blackbody(np.clip(0.66 - 0.3 * uu, 0.2, 1)), ctx.cam0, ctx.cam1, zref=60.0)
        # the rim of the slit: a thin line of white heat
        C, n, eu, es = self.basis(t, cam)
        sy = self.rim_s * hh * Ri
        sx = self.rim_side * hw * Ri * np.clip(1 - self.rim_s ** 2, 0, 1) ** 0.75
        Pr = C + sx[:, None] * es[None, :] + sy[:, None] * eu[None, :]
        er = (0.6 + 0.4 * np.sin(0.7 * t + 40 * self.rim_rnd)) * (1 - np.abs(self.rim_s) ** 6) * 1.6 * k
        fr.splat(Pr, Pr, 0.015, er, look.blackbody(0.86), ctx.cam0, ctx.cam1, zref=60.0)
        # remember the slit for the darkness mask
        ctx.eye_geo = (C, eu, es, Ri, hh, hw)

    @staticmethod
    def darkness(ctx, hdr):
        """remove the light behind the slit (true dark) and calm the storm's white core behind the iris"""
        if not hasattr(ctx, 'eye_geo'):
            return hdr
        C, eu, es, Ri, hh, hw = ctx.eye_geo
        k = ctx.eye_k
        H, W = hdr.shape[:2]
        ss = 4
        # slit polygon
        s = np.linspace(-1, 1, 97)
        wx = hw * Ri * np.clip(1 - s * s, 0, 1) ** 0.75
        right = C[None, :] + wx[:, None] * es[None, :] + (s * hh * Ri)[:, None] * eu[None, :]
        left = C[None, :] - wx[::-1, None] * es[None, :] + (s[::-1] * hh * Ri)[:, None] * eu[None, :]
        poly = np.vstack([right, left])
        u, v, _ = ctx.cam.project(poly, W, H)
        m = np.zeros((H, W), np.uint8)
        pts = np.round(np.stack([u, v], 1) * 16).astype(np.int32)
        cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA, shift=4)
        a_slit = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 0.8 * ctx.scale + 0.4)
        # iris disc (the Eye is seen through the storm's glare: dim what lies behind it)
        ang = np.linspace(0, 2 * np.pi, 97)
        disc = C[None, :] + 0.95 * Ri * (np.cos(ang)[:, None] * es[None, :] + np.sin(ang)[:, None] * eu[None, :])
        u, v, _ = ctx.cam.project(disc, W, H)
        md = np.zeros((H, W), np.uint8)
        cv2.fillPoly(md, [np.round(np.stack([u, v], 1) * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA,
                     shift=4)
        a_iris = cv2.GaussianBlur(md.astype(np.float32) / 255.0, (0, 0), 1.5 * ctx.scale + 0.5)
        alpha = np.clip(0.985 * a_slit * k + 0.88 * a_iris * k, 0, 0.99)
        return hdr * (1.0 - alpha[..., None])
