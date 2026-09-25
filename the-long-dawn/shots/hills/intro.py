"""INTRO 0-359: the festival hill at blue hour.

Shot A  0-159  WIDE: landscape, beacons igniting one by one, the Child points, turns
               and looks up at the Elder (T1 120-200).
Shot B 160-359 TWO-SHOT: the Elder looks down at the Child, then out to the fires (T2
               215-310); from ~225 the camera pushes into the torch flame until rising
               embers fill the frame (crossfade to EMBERS 300-340).
"""
import math

import cv2
import numpy as np

import characters as ch
import fire
import hillworld as hw
from core import Camera, fnoise1, look, over, over_region, smoothstep, track
from hillscene import (FPS, HillScene, ground_y, make_wind_fn, place_beacons, wind_at,
                       ridges_split, render_far, render_crest)
from puppet import Chain, Figure
from sky import render_full_sky

F0, F1 = 0, 359
CUT = 160
Z = hw.CREST_Z
X_ELDER = 3.62
X_CHILD = 3.12
X_CAIRN = 2.20
GE = ground_y(X_ELDER)
GC = ground_y(X_CHILD)
GK = ground_y(X_CAIRN)
HOLD = np.array([3.385, GC + 0.60])          # world (x, y) of the held hands


def wind_base(f):
    # steady hill wind; a lull during the push-in so the embers rise
    return 2.3 - 2.0 * smoothstep(232, 296, f)


# --------------------------------------------------------------- animation ---

def elder_pose(f):
    t = f / FPS
    br = math.sin(2 * math.pi * 0.21 * t)
    head = track(f, [(0, -3.0), (150, -1.0), (172, 0.0), (188, 22.0, 'io'), (206, 24.0), (226, -4.0, 'io'),
                     (300, -6.0), (359, -5.0)])
    head += 1.2 * fnoise1(t * 0.6, 3.0)
    thor = 30.0 + 0.8 * fnoise1(t * 0.3, 5.0)
    lift = track(f, [(0, 0.0), (214, 0.0), (240, 0.03, 'io'), (359, 0.04)])     # raises torch a bit on the answer
    tx = -0.30 + 0.006 * fnoise1(t * 0.5, 7.0)
    ty = 1.20 + 0.004 * br + lift
    pose = dict(x=X_ELDER, breath=br, head=head, thorax=thor, neck=40.0 + 0.5 * head * 0.2,
                f_tgt=(X_ELDER + tx, ty), f_bend=1.0, f_h=35.0, torch_ang=5.0 + 2.0 * fnoise1(t * 0.8, 9.0),
                n_tgt=(HOLD[0], HOLD[1] - GE), n_bend=1.0, n_h=150.0,
                hem_wind=0.02 * wind_at(t, wind_base(f), 0.5, 1.0))
    return pose


def child_pose(f):
    t = f / FPS
    br = math.sin(2 * math.pi * 0.33 * t + 1.0)
    facing = -1 if f < 104 else 1
    # head: watches the hills, follows the new fire, then turns up to the Elder
    if f < 104:
        yaw = track(f, [(0, 90.0), (96, 90.0), (104, 5.0, 'in')])
        head = track(f, [(0, -4.0), (70, -6.0), (82, -12.0, 'io'), (96, -8.0), (104, -20.0)])
    else:
        yaw = track(f, [(104, 5.0), (113, 90.0, 'out')])
        head = track(f, [(104, -20.0), (114, -36.0, 'out'), (200, -38.0), (230, -34.0), (359, -36.0)])
    head += 1.5 * fnoise1(t * 0.9, 13.0)
    pose = dict(x=X_CHILD, breath=br, head=head, head_yaw=yaw, lean=-2.0 if facing == 1 else 2.0,
                n_tgt=(HOLD[0] + 0.02, HOLD[1] - GC), n_bend=-1.0 if facing == -1 else 1.0,
                hat_pom=(0.012 * math.sin(t * 5.5) * (0.5 + 0.25 * wind_base(f)) + 0.012,
                         0.004 * math.cos(t * 4.3)))
    # pointing at the fire that ignites at 80 (free arm), 76-100
    pt = smoothstep(74, 84, f) * (1 - smoothstep(96, 104, f))
    if pt > 0:
        ua = 180.0 + (58.0 - 180.0) * pt
        fa = 180.0 + (52.0 - 180.0) * pt
        pose.update(f_ua=ua, f_fa=fa)
    return pose, facing


# ------------------------------------------------------------------ camera ---

def camera(f, scale):
    if f < CUT:
        u = smoothstep(0, CUT, f)
        pos = [0.0 - 0.55 * u, 0.0, 0.0 + 1.1 * u]
        return Camera(pos, yaw=math.radians(0.9 * u), pitch=math.radians(1.55), hfov=50, scale=scale), None
    # shot B: medium two-shot (figures ~55% of frame height), then the push into the
    # flame that tilts up the ember column (matching EMBERS' "low, tilted up" start)
    tf = torch_world(f)
    start_pos = np.array([3.36, GC + 1.00, Z - 6.4])
    start_tgt = np.array([3.47, GC + 0.98, Z])
    mid_pos = np.array([3.37, GC + 1.01, Z - 5.95])
    end_pos = np.array([tf[0] - 0.01, tf[1] + 0.02, Z - 0.62])
    end_tgt = np.array([tf[0] + 0.02, tf[1] + 0.60, Z + 0.10])
    a = smoothstep(CUT, 230, f)
    pos = start_pos + (mid_pos - start_pos) * a
    u = smoothstep(228, 308, f)
    u = u * u * (1.7 - 0.7 * u)
    pos = pos + (end_pos - pos) * u
    w = smoothstep(262, 312, f)                 # the tilt-up happens late in the move
    tgt = start_tgt + (tf + np.array([0.0, 0.12, 0.0]) - start_tgt) * min(1.0, u * 1.15)
    tgt = tgt + (end_tgt - tgt) * w
    drift = smoothstep(305, 359, f)
    pos = pos + np.array([0.0, 0.10, 0.04]) * drift
    tgt = tgt + np.array([0.0, 0.30, 0.0]) * drift
    hfov = 50.0 - 6.0 * u
    cam = Camera(pos, hfov=hfov, scale=scale)
    yaw, pitch = cam.look_at(tgt)
    focus = float(np.linalg.norm(tf + np.array([0, 0.15, 0]) - pos))
    return Camera(pos, yaw=yaw, pitch=pitch, hfov=hfov, scale=scale), focus


def torch_world(f):
    """World position of the torch flame base for frame f."""
    _, an = ch.elder(elder_pose(f), f / FPS)
    tt = an['torch_top']
    return np.array([tt[0], tt[1] + GE, Z])


# ------------------------------------------------------------------ render ---

class Intro:
    def __init__(self):
        self.scene = HillScene()
        self.sky = hw.sky_intro()
        rl = hw.build_ridges()
        specs = [(4, -2700.0, 22, 4.0, 1.1), (3, -1000.0, 80, 3.8, 2.3), (5, -5400.0, 112, 5.0, 3.7),
                 (3, -300.0, 138, 3.5, 4.1), (4, 1300.0, 196, 4.0, 5.3), (5, 4200.0, 238, 5.0, 6.9),
                 (2, 700.0, 262, 3.0, 7.7)]
        self.beacons = place_beacons(specs, rl)
        self._sim_done = False

    # --- simulations (deterministic, run once per process)
    def simulate(self):
        if self._sim_done:
            return
        # scarf tail: local coords of the Elder card (origin y = GE)
        def anchor(ff):
            _, an = ch.elder(elder_pose(ff), ff / FPS, anchors_only=True)
            return an['scarf_anchor']
        self.scarf = Chain(14, 0.078, anchor, dir0=(1.0, -0.3), drag=8.5, iters=6, damp=0.99)
        self.scarf.simulate(F0, F1, make_wind_fn(1.0, 0.45, 2.0, lift=0.55, flutter=1.4,
                                                  extra=lambda ff: wind_base(ff) - 1.0))
        # grey wisps from the bun
        self.wisps = []
        for k in range(3):
            def a2(ff, k=k):
                _, an = ch.elder(elder_pose(ff), ff / FPS, anchors_only=True)
                return an['bun'] + np.array([0.01 * k, -0.012 * k])
            c = Chain(5, 0.022 + 0.004 * k, a2, dir0=(1.0, 0.2), drag=8.0, iters=4, damp=0.97, grav=3.0)
            c.simulate(F0, F1, make_wind_fn(1.0, 0.6, 10.0 + k, lift=0.3, flutter=1.4,
                                            extra=lambda ff: wind_base(ff) - 1.0))
            self.wisps.append(c)
        # embers from the torch
        self.embers = fire.ParticleSim(seed=11, cap=4000)
        rng = np.random.default_rng(5)
        tor = {}

        def tw(ff):
            k = int(ff)
            if k not in tor:
                tor[k] = torch_world(k)
            return tor[k]

        def emitter(sim, ff, dt):
            rate = 10.0 + 70.0 * smoothstep(236, 285, ff) + 200.0 * smoothstep(262, 305, ff)
            n = rng.poisson(rate * dt)
            if n <= 0:
                return
            base = tw(ff)
            spread = 1.0 + 1.5 * smoothstep(260, 320, ff)
            pos = base + np.stack([rng.normal(0, 0.020 * spread, n), rng.uniform(0.06, 0.26, n),
                                   rng.normal(0, 0.018 * spread, n)], 1)
            vel = np.stack([rng.normal(0.10, 0.22, n), rng.uniform(0.45, 1.3, n), rng.normal(0, 0.14, n)], 1)
            life = rng.gamma(2.0, 0.8, n) + 0.3
            size = rng.random(n) ** 2.5 * 1.8 + 0.18
            kind = (rng.random(n) >= 0.12).astype(np.int64)          # 12% hot sparks, rest embers
            T0 = np.where(kind == 0, 1.0, rng.uniform(0.62, 0.86, n))
            sim.spawn(pos, vel, life, size, kind, T0)
            # the swirl that becomes the legend: many embers lifting off around the flame top
            m = rng.poisson(900.0 * smoothstep(268, 300, ff) * dt)
            if m > 0:
                p2 = base + np.stack([rng.normal(0.0, 0.07, m), rng.uniform(0.15, 0.45, m), rng.normal(0.0, 0.07, m)], 1)
                v2 = np.stack([rng.normal(0.0, 0.10, m), rng.uniform(0.35, 0.9, m), rng.normal(0.0, 0.10, m)], 1)
                sim.spawn(p2, v2, rng.gamma(2.2, 0.7, m) + 0.5, rng.random(m) ** 2.5 * 1.5 + 0.2,
                          np.ones(m, np.int64), rng.uniform(0.55, 0.80, m))

        def params(ff):
            w = wind_at(ff / FPS, wind_base(ff), 0.5, 1.0)
            return dict(wind=(w * 0.8, 0.25, 0.0), buoy=2.2, drag=1.3, turb=0.55, turb_sc=2.4, grav=0.0, cool=1.0)

        self.embers.run(F0, F1, emitter, params, substeps=3)
        self._sim_done = True

    def render(self, f, scale=0.5):
        self.simulate()
        t = f / FPS
        cam, focus = camera(f, scale)
        wind = wind_at(t, wind_base(f), 0.5, 1.0)
        ep = elder_pose(f)
        cp, cf = child_pose(f)
        tfl = torch_world(f)
        flick = fire.flicker(t, 1.7, 1.0)
        torch_I = np.array([3.4, 1.65, 0.55]) * flick
        # background: sky + far ridges + beacons (+ torch glow on the crest ground)
        img, dep = self.scene.background(cam, self.sky, f, self.beacons, crest=False)
        # DOF on the far background during the two-shot/push
        if focus is not None:
            A = 0.016 * cam.f          # aperture (px*m)
            coc = A / max(focus, 0.3)
            sig = coc * 0.42
            if sig > 0.6:
                img = blur_big(img, sig)
        # crest (ground under the figures) with the torch pool
        pool = [(tfl[0], ground_y(tfl[0]), Z, 1.1, 0.07 * flick, 0.032 * flick, 0.010 * flick, 'crest')]
        cr = render_crest(cam, self.sky, f, pool)
        over(img, cr[0], cr[1])
        # grass along the crest
        lights = np.array([[tfl[0], tfl[1] + 0.12, Z - 0.05, torch_I[0], torch_I[1], torch_I[2], 0.30, 0.0]])
        amb_top = np.array([0.020, 0.030, 0.070])
        amb_bot = np.array([0.003, 0.003, 0.005])
        back = np.array([0.34, 0.17, 0.15]) if f < CUT else np.array([0.30, 0.15, 0.14])
        items = []
        gy0 = 0.0
        items.append(Figure([0.0, 0.0, Z + 0.02], [self.scene.grass.group(t, wind)]))
        items.append(Figure([0.0, GK, Z + 0.01], self.scene.cairn.groups(x=X_CAIRN)))
        # Elder
        ep_s = dict(ep)
        scarf = self.scarf.at(f)
        wisps = [w.at(f) for w in self.wisps]
        eg, ea = ch.elder(ep_s, t, scarf_pts=scarf, wisps=wisps)
        items.append(Figure([0.0, GE, Z], eg))
        cg, ca = ch.child(cp, t, facing=cf)
        items.append(Figure([0.0, GC, Z - 0.01], cg))
        blur_fig = 0.0
        if focus is not None:
            A = 0.016 * cam.f
            dz = abs(1.0 / max(Z - cam.pos[2], 0.2) - 1.0 / max(focus, 0.2))
            blur_fig = A * dz * 0.42
        for it in items:
            loc = lights.copy()
            r = it.render(cam, loc, amb_top, amb_bot, back=back, blur_px=blur_fig if blur_fig > 0.6 else 0.0)
            if r is not None:
                y0, x0, rgb, a = r
                over_region(img, y0, x0, rgb, a)
        # smoke from the torch (thin, dark, lit near the flame)
        smoke_rgb = np.zeros_like(img)
        smoke_a = np.zeros(img.shape[:2], np.float32)
        draw_smoke(smoke_rgb, smoke_a, cam, tfl + np.array([0, 0.22, 0]), t, wind, flick)
        over(img, smoke_rgb, smoke_a)
        # torch flame
        fa = np.zeros(img.shape[:2], np.float32)
        lean = 0.10 + 0.16 * wind
        fire.draw_flame(img, fa, cam, tfl + np.array([0.0, -0.035, 0.0]), 0.30 * (0.95 + 0.08 * flick),
                        0.050, lean, t, 1.7, 7.5 * flick, fire.TORCH_STYLE)
        # embers
        P, V, T, S, K = self.embers.snap[f]
        ap = 0.0 if focus is None else 0.016 * cam.f
        fire.render_sparks(img, cam.params(), P, V, T, S, K, 0.35 / FPS, 2.2 * cam.scale ** 2 * 6.0,
                           cam.scale * 1.6, focus if focus else 14.0, ap, 1.0, 1.6, 0.05)
        # exposure: pull down as the flame fills the frame
        expo = 1.0 - 0.42 * smoothstep(250, 320, f)
        out = look.finish(img, exposure=expo, bloom_strength=0.085, bloom_threshold=0.9,
                          vignette_amount=0.22)
        return out


def blur_big(img, sig):
    """Gaussian blur for large sigmas via downsampling."""
    if sig < 3:
        return cv2.GaussianBlur(img, (0, 0), sig)
    k = 2 ** int(min(3, max(0, math.log2(sig / 2.5))))
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, w // k), max(8, h // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sig / k)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def draw_smoke(rgb, a, cam, base, t, wind, flick):
    # screen bbox
    pts = np.array([base + np.array([-0.3, -0.05, 0]), base + np.array([1.4, 1.3, 0])])
    sx, sy, z = cam.project(pts)
    if np.any(z <= 0.05):
        return
    x0, x1 = int(min(sx)) - 10, int(max(sx)) + 10
    y0, y1 = int(min(sy)) - 10, int(max(sy)) + 10
    fire.render_smoke(rgb, a, cam.params(), float(base[0]), float(base[1]), float(base[2]), float(t), 3.3,
                      1.2, 0.035, 0.22, 0.35 * wind, 0.55, 0.30,
                      0.20 * flick, 0.07 * flick, 0.02 * flick, 0.20,
                      0.006, 0.007, 0.012, x0, y0, x1, y1)
