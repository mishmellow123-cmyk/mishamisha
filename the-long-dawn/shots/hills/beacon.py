"""FIRST BEACON 1200-1439: the Young Woman on a Himalayan summit, ~60 years earlier.

One continuous take. 1200-1235 black (a breath of blowing snow). Flint strikes at 1236,
1262, 1290: spark bursts light her hands, her profile, the red scarf. A spark lives in
the tinder; she blows; 1318 the kindling catches (face lit from below, breath vapour).
1360 the beacon ROARS; she rises and steps back; the camera pulls back and up (ease-out)
to reveal the snowy summit, spindrift, a sea of moonlit peaks under a vast starfield.
The moonlit world fades up from black as it is revealed (eye adaptation).
"""
import math

import cv2
import numpy as np

import characters as ch
import fire
from core import Camera, fbm1_np, fnoise1, look, over, over_region, smoothstep, track, splat_gauss
from hillscene import make_wind_fn, wind_at
from land import Ridge, make_profile, pack_ridges, render_ridges
from puppet import Chain, Figure
from sky import Sky, dir_from_az_el, render_full_sky

FPS = 24.0
F0, F1 = 1200, 1439
STRIKES = (1236, 1262, 1290)
CATCH = 1318
ROAR = 1360
R_EARTH = 6371000.0
YW_X = 0.96                    # her hip x while kneeling
CAIRN = ch.Cairn(seed=11)
FIRE_BASE = np.array([0.0, CAIRN.bk_bot + 0.10, 0.0])
TINDER = np.array([0.20, CAIRN.bk_top - 0.03, -0.02])

MOON = np.array([0.55, 0.70, 0.95]) * 0.55          # moonlight on snow (linear)
FOG = np.array([1.0 / 42000.0, 1.0 / 2600.0, -2250.0, 120.0, 0.9, math.radians(3.0), 0.9,
                30.0, 1.0 / 60.0, MOON[0], MOON[1], MOON[2], 1.0], np.float64)


# ------------------------------------------------------------------ world ---

def summit_profile(X):
    X = np.asarray(X, np.float64)
    a = np.abs(X)
    h = -0.025 * X ** 2
    h -= np.where(a > 2.5, 0.55 * (a - 2.5) ** 1.35, 0.0)
    h -= np.where(X > 0, 0.004 * X ** 3, 0.0)
    h += 0.05 * fbm1_np(X / 0.9 + 3.0, 4, 2.0, 0.5, 71)
    return h


def build_world():
    ridges = []
    # (z, top angle deg, amp deg, scale, ridged, albedo, mist)
    L = [(260.0, -16.0, 5.0, 160.0, 0.9, 0.9, 0.2),
         (700.0, -11.0, 4.0, 380.0, 0.9, 0.8, 0.5),
         (1600.0, -8.0, 3.2, 700.0, 0.92, 0.7, 0.8),
         (3600.0, -5.6, 2.4, 1300.0, 0.94, 0.6, 1.0),
         (8000.0, -4.0, 1.7, 2600.0, 0.95, 0.5, 1.0),
         (17000.0, -3.1, 1.1, 5000.0, 0.95, 0.45, 0.9),
         (38000.0, -2.6, 0.75, 9000.0, 0.95, 0.4, 0.6),
         (80000.0, -2.7, 0.55, 16000.0, 0.95, 0.35, 0.4)]
    for i, (z, top, amp, sc, rd, al, mist) in enumerate(L):
        x0, x1 = -1.2 * z - 200, 1.2 * z + 200
        n = int(min(20000, max(3000, (x1 - x0) / (z / 3000.0))))
        drop = z * z / (2 * R_EARTH)
        base = z * math.tan(math.radians(top))
        ampm = z * math.tan(math.radians(amp))
        h = make_profile(x0, x1, n, base - drop - 0.3 * ampm, ampm, sc, 500 + i * 13, octaves=8, ridged=rd)
        ridges.append(Ridge(z, x0, x1, h, np.array([0.004, 0.006, 0.012]) * al, fog_mul=1.0, rim=0.05,
                            mist=mist, tex=0.0, name=f'M{i}', fog_el=math.radians(2.0),
                            mist_scale=1.0 / (0.6 * sc), seed=3.1 * i, snow=1.0))
    X = np.linspace(-60, 60, 12000)
    summit = Ridge(0.02, -60, 60, summit_profile(X), np.array([0.004, 0.006, 0.012]), fog_mul=0.0, rim=0.0,
                   mist=0.0, tex=0.0, name='summit', fog_el=math.radians(2.0), snow=1.25, seed=9.0)
    return pack_ridges(ridges), pack_ridges([summit])


def sky():
    return Sky(sun=(0.0, -40.0), zenith=np.array([0.0005, 0.0010, 0.0065]),
               horizon=np.array([0.010, 0.018, 0.050]), amber=np.zeros(3), rose=np.zeros(3),
               violet=np.array([0.004, 0.006, 0.018]), base_fall=0.30, amber_fall=0.03, rose_fall=0.05,
               violet_fall=0.3, az_pow=40.0, moon=None, mw=1.6, mw_pole=dir_from_az_el(35.0, 38.0),
               star_gain=48.0, star_thresh=0.6, n_stars=26000, ring=None, seed=7)


# --------------------------------------------------------------- animation ---

def strike_env(f):
    """Spark-light envelope: sum of the three strikes (peak ~1 on the frame, fast decay)."""
    e = 0.0
    for k, s in enumerate(STRIKES):
        d = f - s
        if d >= 0:
            e += (0.55 + 0.45 * k) * math.exp(-d / (2.2 + 0.6 * k))
    return e


def ember_env(f):
    """The spark that lives in the tinder after strike 3; pulses as she blows."""
    if f < STRIKES[2] + 1 or f >= CATCH + 4:
        return 0.0
    tr = f - STRIKES[2]
    base = 0.12 + 0.10 * smoothstep(0, 25, tr)
    puff = 0.5 + 0.5 * math.sin((f - 1296) / 7.5 * 2 * math.pi) if f > 1296 else 0.0
    return base * (1.0 + 1.3 * puff)


def flame_level(f):
    """0 before the catch; small flame 1318-1359; roar after 1360."""
    if f < CATCH:
        return 0.0
    small = smoothstep(CATCH, CATCH + 14, f) * (0.35 + 0.65 * smoothstep(CATCH + 10, ROAR, f))
    if f < ROAR:
        return small
    tr = (f - ROAR) / FPS
    return 1.0 + 3.0 * fire.ignition(tr, overshoot=0.55, rise=0.30, settle=0.9)


def yw_pose(f):
    t = f / FPS
    br = math.sin(2 * math.pi * 0.28 * t)
    # strike cycle for the near (steel) hand: raise then snap down past the flint
    hx, hy = 0.34, 1.13
    for s in STRIKES:
        d = f - s
        if -10 <= d <= 6:
            if d < 0:
                a = smoothstep(-10, -2, d)
                hy += 0.07 * a
                hx += 0.02 * a
            else:
                a = 1 - smoothstep(0, 6, d)
                hy -= 0.035 * a
                hx -= 0.015 * a
            if -2 <= d < 0:
                hy -= 0.10 * smoothstep(-2, 0, d)
    # blowing on the tinder 1294-1320: lean in, head low
    lean = smoothstep(1292, 1302, f) * (1 - smoothstep(1322, 1336, f))
    rise = smoothstep(ROAR + 2, ROAR + 30, f)
    step = smoothstep(ROAR + 18, ROAR + 44, f)
    x = YW_X + 0.30 * step
    kneel = dict(hip_y=0.50, lumbar=18 + 8 * lean, thorax=30 + 14 * lean, neck=32 + 10 * lean,
                 head=22 + 12 * lean - 10 * smoothstep(CATCH + 2, CATCH + 16, f),
                 n_th=95.0, n_sh=182.0, f_th=178.0, f_sh=-88.0)
    stand = dict(hip_y=0.92, lumbar=-4.0, thorax=-6.0, neck=4.0, head=-6.0,
                 n_th=176.0, n_sh=182.0, f_th=186.0, f_sh=178.0)
    p = {}
    for k in kneel:
        p[k] = kneel[k] + (stand[k] - kneel[k]) * rise
    p['x'] = x
    p['breath'] = br
    p['flint'] = rise < 0.5
    p['hem_wind'] = 0.10
    if rise < 0.5:
        p['f_tgt'] = (0.30 + 0.01 * br, 1.08)
        p['f_bend'] = 1.0
        p['f_h'] = 70.0
        p['n_tgt'] = (hx, hy)
        p['n_bend'] = 1.0
        p['n_h'] = 110.0
    else:
        # arm raised against the heat, then lowered
        u = smoothstep(ROAR + 12, ROAR + 40, f)
        p['n_ua'] = 150.0 + 20.0 * u
        p['n_fa'] = 40.0 + 120.0 * u
        p['n_h'] = 40.0 + 120.0 * u
        p['f_ua'] = 178.0
        p['f_fa'] = 175.0
    return p


def camera(f, scale):
    t = f / FPS
    hand = 0.004 * fnoise1(t * 0.9, 1.0) , 0.003 * fnoise1(t * 0.8, 2.0)
    c_pos = np.array([0.50 + hand[0], 0.98 + hand[1], -1.30])
    c_tgt = np.array([0.46, 1.06, 0.0])
    w_pos = np.array([1.10, 4.3, -21.0])
    w_tgt = np.array([0.35, -1.3, 40.0])
    u = smoothstep(ROAR, F1 + 6, f)
    u = 1 - (1 - u) ** 3            # explosive start, long deceleration
    pos = c_pos + (w_pos - c_pos) * u
    tgt = c_tgt + (w_tgt - c_tgt) * (u ** 0.8)
    hfov = 42.0 + 16.0 * u
    cam = Camera(pos, hfov=hfov, scale=scale)
    yaw, pitch = cam.look_at(tgt)
    focus = float(np.linalg.norm(np.array([0.4, 1.0, 0.0]) - pos))
    return Camera(pos, yaw=yaw, pitch=pitch, hfov=hfov, scale=scale), focus, u


# ------------------------------------------------------------------ shot ---

class FirstBeacon:
    def __init__(self):
        self.far, self.summit = build_world()
        self.sky = sky()
        self._sim = False

    def simulate(self):
        if self._sim:
            return
        rng = np.random.default_rng(21)

        def wbase(ff):
            return 7.0 + 2.5 * fnoise1(ff / FPS * 0.4, 3.0, 2)

        def anchor(ff):
            _, an = ch.young_woman(yw_pose(ff), ff / FPS, anchors_only=True)
            return an['scarf_anchor']
        self.scarf = Chain(14, 0.078, anchor, dir0=(1.0, 0.0), drag=6.0, iters=6, damp=0.99)
        self.scarf.simulate(F0, F1, make_wind_fn(1.0, 0.35, 5.0, lift=0.8, flutter=2.6,
                                                 extra=lambda ff: wbase(ff) - 1.0))
        self.hair = []
        for k in range(9):
            def ha(ff, k=k):
                _, an = ch.young_woman(yw_pose(ff), ff / FPS, anchors_only=True)
                return an['hair_root'] + np.array([-0.01 + 0.004 * k, 0.03 - 0.012 * k])
            c = Chain(8, 0.045 + 0.006 * (k % 3), ha, dir0=(1.0, -0.1), drag=7.0, iters=4, damp=0.98, grav=4.0)
            c.simulate(F0, F1, make_wind_fn(1.0, 0.5, 20.0 + k, lift=0.5, flutter=2.2,
                                            extra=lambda ff: wbase(ff) - 1.0))
            self.hair.append(c)
        # strike sparks (hot, fast, gravity) + beacon sparks/embers (buoyant)
        self.sparks = fire.ParticleSim(seed=31, cap=3000)

        def emit(sim, ff, dt):
            for k, s in enumerate(STRIKES):
                if s - 0.5 <= ff < s + 0.5:
                    n = int((35, 60, 110)[k] * dt * FPS * 1.0)
                    p = ch.young_woman(yw_pose(s), s / FPS)[1]['flint']
                    pos = np.array([p[0] - 0.01, p[1], -0.01]) + rng.normal(0, 0.004, (n, 3))
                    ang = rng.normal(-2.2, 0.55, n)          # mostly down-left into the basket
                    sp = rng.gamma(3.0, 0.9, n) + 0.6
                    vel = np.stack([np.cos(ang) * sp, np.sin(ang) * sp + 0.4, rng.normal(0, 0.5, n)], 1)
                    sim.spawn(pos, vel, rng.uniform(0.12, 0.55, n), rng.random(n) ** 2.2 * 2.0 + 0.3,
                              np.zeros(n, np.int64), 1.0)
            lv = flame_level(ff)
            if lv > 0:
                rate = 6.0 + 26.0 * max(0.0, lv - 1.0) + 90.0 * smoothstep(ROAR, ROAR + 4, ff) * (1 - smoothstep(ROAR + 10, ROAR + 30, ff))
                n = rng.poisson(rate * dt)
                if n > 0:
                    base = FIRE_BASE if ff >= ROAR else TINDER
                    spread = 0.20 if ff >= ROAR else 0.03
                    pos = base + np.stack([rng.normal(0, spread, n), rng.uniform(0.1, 0.6 if ff >= ROAR else 0.1, n),
                                           rng.normal(0, spread * 0.5, n)], 1)
                    vel = np.stack([rng.normal(0.5, 0.5, n), rng.uniform(1.0, 3.5, n), rng.normal(0, 0.3, n)], 1)
                    kind = (rng.random(n) > 0.3).astype(np.int64)
                    sim.spawn(pos, vel, rng.gamma(2.0, 0.6, n) + 0.3, rng.random(n) ** 2.5 * 1.8 + 0.2, kind,
                              np.where(kind == 0, 1.0, rng.uniform(0.6, 0.85, n)))

        def params(ff):
            return dict(wind=(wbase(ff) * 0.5, 0.0, 0.0), buoy=2.8, drag=1.2, turb=0.9, turb_sc=1.6,
                        grav=2.5, cool=1.0)

        self.sparks.run(F0, F1, emit, params, substeps=4)
        # spindrift snow grains near the camera path
        self.snow = rng.uniform([-6, -1, -8], [8, 7, 6], (2600, 3))
        self.snow_ph = rng.random(2600)
        self._sim = True

    def render(self, f, scale=0.5):
        self.simulate()
        t = f / FPS
        cam, focus, u = camera(f, scale)
        reveal = 0.03 + 0.97 * smoothstep(ROAR + 4, ROAR + 58, f)
        lv = flame_level(f)
        flick = fire.flicker(t, 3.3, 1.3)
        st_e = strike_env(f)
        em_e = ember_env(f)
        # --- lights
        lights = []
        flint = ch.young_woman(yw_pose(f), t)[1]['flint']
        if st_e > 0.01:
            lights.append([flint[0] - 0.05, flint[1] - 0.05, -0.08, 3.2 * st_e, 2.6 * st_e, 1.9 * st_e, 0.30, 0.0])
        if em_e > 0:
            lights.append([TINDER[0], TINDER[1], -0.05, 1.4 * em_e, 0.45 * em_e, 0.10 * em_e, 0.18, 0.0])
        if lv > 0:
            if f < ROAR:
                I = lv * flick
                lights.append([TINDER[0], TINDER[1] + 0.06, -0.06, 2.6 * I, 1.25 * I, 0.40 * I, 0.35, 0.0])
            else:
                I = min(lv, 2.5) * flick
                lights.append([FIRE_BASE[0], FIRE_BASE[1] + 0.6, -0.1, 6.0 * I, 2.8 * I, 0.9 * I, 1.1, 0.0])
        lights = np.array(lights, np.float64).reshape(-1, 8)
        # --- sky + range (moonlit world scaled by the reveal)
        sk = self.sky
        img = render_full_sky(cam, sk, t) * (0.25 + 0.75 * reveal)
        fog = FOG.copy()
        rs, meta, hs = self.far
        rgbp = np.zeros_like(img)
        a = np.zeros(img.shape[:2], np.float32)
        dep = np.full(img.shape[:2], 1e9, np.float32)
        render_ridges(rgbp, a, dep, cam.params(), meta, hs, sk.packed(), fog, np.zeros((0, 8)), np.zeros(1), 1.0, t)
        over(img, rgbp * reveal, a)
        # summit snow (+ warm pool from the fire)
        pool = []
        if lv > 0 and f >= ROAR:
            I = min(lv, 2.5) * flick
            pool.append([FIRE_BASE[0], 0.0, 0.02, 3.2, 0.10 * I, 0.045 * I, 0.014 * I, 0])
        elif lv > 0:
            pool.append([TINDER[0], 0.0, 0.02, 0.8, 0.02 * lv, 0.009 * lv, 0.003 * lv, 0])
        rs2, meta2, hs2 = self.summit
        srgb = np.zeros_like(img)
        sa = np.zeros(img.shape[:2], np.float32)
        sd = np.full(img.shape[:2], 1e9, np.float32)
        render_ridges(srgb, sa, sd, cam.params(), meta2, hs2, sk.packed(), fog,
                      np.array(pool, np.float64).reshape(-1, 8), np.zeros(1), 1.0, t)
        # the summit keeps a little moonlight even in the close-up (dim), full after reveal
        over(img, srgb * (0.15 + 0.85 * reveal), sa)
        # DOF on background in the close-up
        coc = 0.012 * cam.f / max(focus, 0.3) * (1 - u) ** 2
        if coc * 0.42 > 0.8:
            img = cv2.GaussianBlur(img, (0, 0), coc * 0.42)
        # --- beacon smoke (behind her)
        if f >= ROAR:
            sm_rgb = np.zeros_like(img)
            sm_a = np.zeros(img.shape[:2], np.float32)
            b = FIRE_BASE + np.array([0.0, 0.9, 0.05])
            pts = np.array([b + np.array([-1.0, -0.2, 0]), b + np.array([9.0, 8.0, 0])])
            sx, sy, z = cam.project(pts)
            if np.all(z > 0.05):
                I = min(lv, 2.5) * flick
                fire.render_smoke(sm_rgb, sm_a, cam.params(), float(b[0]), float(b[1]), float(b[2]), t, 1.7,
                                  7.0, 0.25, 0.30, 0.9, 0.9, 0.55 * smoothstep(ROAR, ROAR + 12, f),
                                  0.30 * I, 0.11 * I, 0.03 * I, 0.9, 0.004, 0.005, 0.009,
                                  int(min(sx)) - 20, int(min(sy)) - 20, int(max(sx)) + 20, int(max(sy)) + 20)
                over(img, sm_rgb, sm_a)
        # --- cairn + woman
        amb_top = np.array([0.010, 0.016, 0.040]) * reveal
        back = np.array([0.03, 0.05, 0.12]) * (0.3 + 0.7 * reveal)
        yp = yw_pose(f)
        hair = [c.at(f) for c in self.hair]
        yg, ya = ch.young_woman(yp, t, scarf_pts=self.scarf.at(f), hair_pts=hair)
        items = [Figure([0.0, 0.0, 0.0], CAIRN.groups(x=0.0)), Figure([0.0, 0.0, -0.01], yg)]
        blur = 0.0
        for it in items:
            r = it.render(cam, lights, amb_top, np.zeros(3), back=back, blur_px=blur)
            if r is not None:
                y0, x0, rgb, al = r
                over_region(img, y0, x0, rgb, al)
        # --- fire
        fa = np.zeros(img.shape[:2], np.float32)
        if lv > 0:
            if f < ROAR:
                h = 0.05 + 0.20 * lv
                fire.draw_flame(img, fa, cam, TINDER + np.array([0.0, -0.01, 0.0]), h, 0.028 + 0.05 * lv, 0.45,
                                t, 4.4, 6.5 * flick * (0.6 + 0.4 * lv), fire.TORCH_STYLE)
            else:
                grow = min(lv, 2.2)
                fire.draw_flame(img, fa, cam, FIRE_BASE, 0.55 + 0.75 * grow, 0.31, 0.55, t, 2.9,
                                5.0 + 5.0 * min(1.0, lv - 1.0 + 0.3), fire.BONFIRE_STYLE)
            fire.add_glow(img, cam, (TINDER if f < ROAR else FIRE_BASE + np.array([0, 0.7, 0])),
                          0.12 if f < ROAR else 0.9, (0.05 * lv if f < ROAR else 0.10 * min(lv, 2.5)) * flick)
        if em_e > 0:
            sx, sy, z = cam.project(TINDER)
            e = 6.0 * em_e * cam.scale ** 2
            splat_gauss(img, float(sx), float(sy), max(0.5, 2.0 * cam.scale), e, e * 0.35, e * 0.06)
            fire.add_glow(img, cam, TINDER, 0.05, 0.06 * em_e)
        # --- breath vapour after the catch (lit by the small flame)
        if CATCH <= f < ROAR + 10:
            self._breath(img, cam, f, ya['mouth'], lv)
        # --- sparks & embers
        P, V, T, S, K = self.sparks.snap[f]
        ap = 0.010 * cam.f * (1 - u)
        fire.render_sparks(img, cam.params(), P, V, T, S, K, 0.5 / FPS, 1.8 * cam.scale ** 2 * 6.0,
                           cam.scale * 1.4, focus, ap, 1.0, 1.4, 0.05)
        # --- spindrift grains
        self._spindrift(img, cam, f, reveal, lv)
        expo = 1.05
        out = look.finish(img, exposure=expo, bloom_strength=0.09, bloom_threshold=0.9, vignette_amount=0.25)
        return out

    def _breath(self, img, cam, f, mouth, lv):
        rng = np.random.default_rng(3)
        for k in range(4):
            f0 = CATCH + 4 + k * 11
            age = (f - f0) / FPS
            if age < 0 or age > 1.6:
                continue
            for q in range(6):
                off = rng.normal(0, 1, 2)
                p = np.array([mouth[0] + 0.55 * age + 0.02 * off[0] + 0.05 * q * age,
                              mouth[1] + 0.05 * age + 0.015 * off[1], -0.02])
                sx, sy, z = cam.project(p)
                if z <= 0:
                    continue
                rad = cam.f * (0.012 + 0.05 * age) / z
                fade = math.exp(-age / 0.6) * (1 - math.exp(-age / 0.08))
                e = 0.035 * fade * (0.4 + lv) * rad * rad
                splat_gauss(img, float(sx), float(sy), max(0.6, rad * 0.6), e * 1.0, e * 0.62, e * 0.40)

    def _spindrift(self, img, cam, f, reveal, lv):
        t = f / FPS
        P = self.snow.copy()
        speed = 7.5
        P[:, 0] = (P[:, 0] + speed * t + 3.0 * self.snow_ph) % 14.0 - 6.0
        P[:, 1] += 0.25 * np.sin(t * 2.0 + self.snow_ph * 20)
        V = np.zeros_like(P)
        V[:, 0] = speed
        sx, sy, z = cam.project(P)
        tx, ty, _ = cam.project(P - V * 0.5 / FPS)
        m = (z > 0.2) & (sx > -20) & (sx < cam.W + 20) & (sy > -20) & (sy < cam.H + 20)
        if not np.any(m):
            return
        d_fire = np.linalg.norm(P - (FIRE_BASE if f >= ROAR else TINDER), axis=1)
        warm = (0.08 + 1.2 * min(lv, 2.0)) / (1.0 + (d_fire / (0.5 + 0.6 * (f >= ROAR))) ** 2)
        cool = 0.03 * reveal
        e = (warm + cool)[m] * cam.scale ** 2 * 1.2 / np.maximum(z[m], 0.5)
        cols = np.stack([e * (0.25 + 0.75 * (warm[m] > cool)) + e * 0.2, e * 0.75, e * 0.7], 1)
        import fire as _f
        from core import splat_particles
        sig = np.full(m.sum(), max(0.35, 0.45 * cam.scale))
        splat_particles(img, tx[m].astype(np.float64), ty[m].astype(np.float64), sx[m].astype(np.float64),
                        sy[m].astype(np.float64), sig, cols.astype(np.float64))
