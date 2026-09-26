"""CODA 2460-2807: back on the festival hill, deeper night.

C1 2460-2627 MEDIUM: the Child looks up (T15 2500-2570). The Elder meets her eyes, then
   looks away and says nothing; a gust lifts the red scarf; 2580-2620 she lowers the torch
   into the Child's raised hands and rests her hand on the Child's shoulder.
C2 2628-2807 WIDE: the Child lifts the torch into the basket - it catches at 2640 - and
   answering fires ignite across every hill to the horizon in a rolling wave (2645-2720).
   The camera cranes up into the sky (stars, the ring, the lit Moon, slow ship lights);
   quiet and dark for the title (2668-2780).
"""
import json
import math
import os

import numpy as np

import cairn2
import characters as ch
import figures2 as f2
import fire
import fires2
import silhouette as sil
import hillworld as hw
from core import Camera, fnoise1, look, over, over_region, smoothstep, track, splat_gauss, RENDER_DIR
from hillscene import (FPS, HillScene, ground_y, make_wind_fn, place_beacons, wind_at, render_crest, Beacon)
from intro import blur_big, draw_smoke
from puppet import Chain, Figure
from sky import dir_from_az_el, render_full_sky

F0, F1 = 2460, 2807
# v2 (Sep 2026, CODA dept): merged-silhouette Elder + Child (figures2.py, silhouette.py), cloth
# scarf, dry-stone cairn, distinct answering fires. False = the v1 puppets (for comparison).
V2 = True
CUT = 2628
CATCH = 2640
Z = hw.CREST_Z
X_CAIRN = 2.20
GK = ground_y(X_CAIRN)
TORCH_LEN = 0.46


def xs(f):
    """Positions: in C2 they have stepped up to the cairn."""
    if f < CUT:
        return 3.62, 3.12
    return 3.22, 2.74


def wind_base(f):
    g = 1.6 * smoothstep(2548, 2566, f) * (1 - smoothstep(2590, 2618, f))     # the gust
    return 2.0 + g


# --------------------------------------------------------------- animation ---

def handover(f):
    """0 -> 1 over the torch handover."""
    return smoothstep(2580, 2606, f)


def elder_pose(f):
    t = f / FPS
    XE, XC = xs(f)
    GE = ground_y(XE)
    GC = ground_y(XC)
    br = math.sin(2 * math.pi * 0.21 * t)
    head = track(f, [(2460, -3.0), (2500, 2.0), (2512, 22.0, 'io'), (2534, 24.0), (2552, -2.0, 'io'),
                     (2582, -2.0), (2598, 20.0, 'io'), (2627, 22.0)])
    if f >= CUT:
        head = track(f, [(CUT, 14.0), (2645, 10.0), (2665, -3.0, 'io'), (2807, -4.0)])
    head += 1.0 * fnoise1(t * 0.6, 3.0)
    p = dict(x=XE, breath=br, head=head, thorax=30.0 + 0.8 * fnoise1(t * 0.3, 5.0), neck=40.0,
             hem_wind=0.02 * wind_at(t, wind_base(f), 0.5, 1.0))
    h = handover(f)
    if f < CUT:
        # torch hand: from raised to lowered in front of the child's chest
        tx0, ty0 = XE - 0.30, 1.20
        tx1, ty1 = XC + 0.14, 0.86 + (GC - GE)
        tx = tx0 + (tx1 - tx0) * h
        ty = ty0 + (ty1 - ty0) * h + 0.004 * br
        rel = smoothstep(2610, 2618, f)            # she lets go
        if rel <= 0:
            p.update(f_tgt=(tx, ty), f_bend=1.0, f_h=35.0 - 20.0 * h, torch_ang=5.0 - 8.0 * h, torch=True)
        else:
            # hand moves to the child's shoulder
            sx_, sy_ = XC + 0.02, 0.80 + (GC - GE)
            p.update(f_tgt=(tx + (sx_ - tx) * smoothstep(2612, 2627, f), ty + (sy_ - ty) * smoothstep(2612, 2627, f)),
                     f_bend=1.0, f_h=140.0, torch=False)
        # near hand: holds the child's hand until the handover starts, then lets it go
        hold = np.array([0.5 * (XE + XC) - 0.03, 0.60 + (GC - GE)])
        p.update(n_tgt=(hold[0], hold[1] - 0.10 * h), n_bend=1.0, n_h=150.0)
    else:
        # C2: near hand on the child's shoulder, far arm hanging
        p.update(torch=False, n_tgt=(XC + 0.06, 0.80 + (GC - GE)), n_bend=1.0, n_h=150.0,
                 f_ua=176.0, f_fa=170.0, f_h=175.0)
    return p


def child_torch_grip(f):
    """(grip point local to the child's card, torch direction) once the child has it."""
    XE, XC = xs(f)
    if f < CUT:
        # just below where the Elder's hand was at the release
        g = np.array([XC + 0.14, 0.72])
        return g, ch.dirv(-3.0)
    # C2: raise the torch into the basket (2628-2642), then lower (2650-2672)
    up = smoothstep(CUT, 2638, f) * (1 - smoothstep(2650, 2672, f))
    g0 = np.array([XC - 0.14, 0.70])
    g1 = np.array([XC - 0.30, 0.92])
    g = g0 + (g1 - g0) * up
    ang = 5.0 + 42.0 * up
    return g, ch.dirv(ang)


def child_pose(f):
    t = f / FPS
    XE, XC = xs(f)
    br = math.sin(2 * math.pi * 0.33 * t + 1.0)
    h = handover(f)
    if f < CUT:
        head = track(f, [(2460, -8.0), (2476, -30.0, 'io'), (2500, -37.0), (2570, -36.0), (2588, -30.0),
                         (2612, -12.0, 'io'), (2627, -10.0)])
        head += 1.2 * fnoise1(t * 0.9, 13.0)
        p = dict(x=XC, breath=br, head=head, head_yaw=90.0, lean=-2.0,
                 hat_pom=(0.010 * math.sin(t * 5.5) + 0.010, 0.004 * math.cos(t * 4.3)))
        GE, GC = ground_y(XE), ground_y(XC)
        hold = np.array([0.5 * (XE + XC) - 0.03 + 0.02, 0.60 + 0.0])
        g, d = child_torch_grip(f)
        reach = smoothstep(2584, 2604, f)
        # both hands rise to the handle
        n_t = hold * (1 - reach) + (g + np.array([0.0, 0.03])) * reach
        f_t = np.array([XC + 0.05, 0.42]) * (1 - reach) + (g - np.array([0.0, 0.04])) * reach
        p.update(n_tgt=tuple(n_t), n_bend=1.0, f_tgt=tuple(f_t), f_bend=1.0)
        return p, 1
    # C2: facing the cairn (left)
    head = track(f, [(CUT, -18.0), (2640, -22.0), (2652, -10.0, 'io'), (2700, -6.0), (2807, -8.0)])
    head += 1.0 * fnoise1(t * 0.9, 13.0)
    g, d = child_torch_grip(f)
    p = dict(x=XC, breath=br, head=head, head_yaw=90.0, lean=4.0,
             hat_pom=(0.010 * math.sin(t * 5.5) + 0.010, 0.004 * math.cos(t * 4.3)),
             n_tgt=tuple(g + d * 0.05), n_bend=-1.0, f_tgt=tuple(g - d * 0.05), f_bend=-1.0)
    return p, -1


def torch_world(f):
    """(flame base world pos, holder) ."""
    XE, XC = xs(f)
    if f < CUT and smoothstep(2610, 2618, f) <= 0:
        _, an = (f2.elder2 if V2 else ch.elder)(elder_pose(f), f / FPS)
        tt = an['torch_top']
        return np.array([tt[0], tt[1] + ground_y(XE), Z]), 'elder'
    g, d = child_torch_grip(f)
    top = g + d * 0.41
    return np.array([top[0], top[1] + ground_y(XC), Z]), 'child'


# ------------------------------------------------------------------ camera ---

def camera(f, scale):
    if f < CUT:
        XE, XC = xs(f)
        GC = ground_y(XC)
        u = smoothstep(F0, CUT, f)
        pos = np.array([3.30 - 0.10 * u, GC + 0.86, Z - 5.6 + 0.5 * u])
        tgt = np.array([3.44, GC + 0.95, Z])
        cam = Camera(pos, hfov=46, scale=scale)
        yaw, pitch = cam.look_at(tgt)
        return Camera(pos, yaw=yaw, pitch=pitch, hfov=46, scale=scale), float(np.linalg.norm(tgt - pos))
    c = smoothstep(2662, 2800, f)
    c = c * c * (3 - 2 * c)
    pos = np.array([1.25, 0.05 + 13.0 * c, 5.2 - 2.0 * c])
    pitch = math.radians(1.55 + 12.5 * c)
    hfov = 50.0 + 6.0 * c
    return Camera(pos, yaw=math.radians(2.0), pitch=pitch, hfov=hfov, scale=scale), None


# ------------------------------------------------------------ answering fires ---

FESTIVAL = [(4, -2700.0, 22, 4.0, 1.1), (3, -1000.0, 80, 3.8, 2.3), (5, -5400.0, 112, 5.0, 3.7),
            (3, -300.0, 138, 3.5, 4.1), (4, 1300.0, 196, 4.0, 5.3), (5, 4200.0, 238, 5.0, 6.9),
            (2, 700.0, 262, 3.0, 7.7)]          # the festival fires lit in the INTRO (layer, x, frame, size, seed)


def answering_fires(rl):
    """(fires, info) of the answering-fire wave. v2: distinct fires on visible ridgelines, one
    per SFX whump (fires2.PLAN); v1: ~50 splats."""
    if V2:
        fest = fires2.festival_fires(FESTIVAL, rl)
        return fires2.answering_fires_v2(rl, lambda ff: camera(ff, 1.0)[0], avoid=fest)
    return answering_fires_v1(rl)


def answering_fires_v1(rl):
    rng = np.random.default_rng(2645)
    per = [(0, 3), (1, 4), (2, 6), (3, 7), (4, 8), (5, 9), (6, 9), (7, 6)]
    cands = []
    for li, k in per:
        r = rl[li]
        span = r.z * math.tan(math.radians(27.0))
        lo, hi = -span + 4.0 * 0, span * 0.85
        peaks = r.peaks(-span, span * 0.95, min_sep=0.05 * r.z)
        if len(peaks) == 0:
            continue
        idx = rng.choice(len(peaks), size=min(k, len(peaks)), replace=False)
        for i in idx:
            x, h = peaks[i]
            cands.append((li, x, h, r.z))
    # rolling wave outward from our hill (the camera side), left-to-right bias
    d = np.array([math.hypot(c[3], (c[1] - 900.0) * 0.8) for c in cands])
    order = np.argsort(d)
    dn = (d - d.min()) / (d.max() - d.min() + 1e-9)
    beacons = []
    info = []
    for rank, i in enumerate(order):
        li, x, h, z = cands[i]
        fi = 2645.0 + 75.0 * dn[i] ** 0.65 + rng.uniform(-1.5, 1.5)
        size = 3.2 + 1.4 * rng.random() + 0.00005 * z
        b = Beacon(li, x, h, z, fi, size, seed=17.0 + rank * 1.37)
        beacons.append(b)
        info.append(dict(frame=round(fi, 1), layer=li, distance_m=round(float(math.hypot(z, x))), x_m=round(float(x)),
                         voiced=rank < 18))
    return beacons, info


# ------------------------------------------------------------------ render ---

class Coda:
    def __init__(self):
        self.scene = HillScene()
        self.sky = hw.sky_coda()
        rl = hw.build_ridges()
        specs = [(4, -2700.0, 22, 4.0, 1.1), (3, -1000.0, 80, 3.8, 2.3), (5, -5400.0, 112, 5.0, 3.7),
                 (3, -300.0, 138, 3.5, 4.1), (4, 1300.0, 196, 4.0, 5.3), (5, 4200.0, 238, 5.0, 6.9),
                 (2, 700.0, 262, 3.0, 7.7)]
        self.old = place_beacons(specs, rl)          # the festival fires lit during the INTRO
        if V2:
            self.old = fires2.festival_fires(specs, rl)
        self.wave, self.wave_info = answering_fires(rl)
        self.cairn2 = cairn2.DryStoneCairn(seed=7) if V2 else None
        self._sim = False

    def export_fires(self, out_dir=None):
        if V2:
            out_dir = out_dir or os.path.join(os.path.dirname(RENDER_DIR), 'hills_v2')
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, 'coda_fires.json')
            fb = np.array([X_CAIRN, GK + self.scene.cairn.bk_bot + 0.10, Z])
            fest = [dict(frame=int(b.frame), layer=int(b.layer), x_m=round(b.x, 1), y_m=round(b.y, 1), z_m=round(b.z),
                         flame_height_m=round(b.height, 2)) for b in self.old]
            with open(path, 'w') as fh:
                json.dump(dict(
                    note=('CODA v2 answering fires. frame = SRC ignition (the whoomp peaks 2-6 frames later); '
                          'frame_v2 = frame + 160 = the SFX answering_fire_NN whump frames (sfx_v2.py). '
                          'World metres (x right, y up, z forward; camera = coda.camera). pan = screen x at '
                          'ignition, -1 left .. +1 right (the SFX pans alternate differently; re-pan to match '
                          'if wanted). festival = the INTRO fires, still burning.'),
                    beacon_catch=CATCH, beacon_catch_v2=CATCH + 160,
                    beacon_fire_base_m=[round(float(v), 3) for v in fb],
                    fires=sorted(self.wave_info, key=lambda d: d['frame']), festival=fest), fh, indent=1)
            return path
        path = os.path.join(RENDER_DIR, 'coda_fires.json')
        with open(path, 'w') as fh:
            json.dump(dict(note='CODA answering fires (global frames). voiced = the first 18 in order, '
                                'suggested one bell each; x_m negative = screen left.',
                           beacon_catch=CATCH, fires=sorted(self.wave_info, key=lambda d: d['frame'])), fh, indent=1)
        return path

    def simulate(self):
        if self._sim:
            return
        elder_fn = f2.elder2 if V2 else ch.elder

        def anchor(ff):
            _, an = elder_fn(elder_pose(ff), ff / FPS, anchors_only=True)
            return an['scarf_anchor']
        if V2:
            # a longer, finer chain: the tail is cloth (tapering, twisting, rippling), ~1.05 m
            self.scarf = Chain(20, 0.053, anchor, dir0=(1.0, -0.3), drag=8.0, iters=8, damp=0.99)
        else:
            self.scarf = Chain(14, 0.078, anchor, dir0=(1.0, -0.3), drag=8.5, iters=6, damp=0.99)
        self.scarf.simulate(F0, F1, make_wind_fn(1.0, 0.45, 7.0, lift=0.55, flutter=1.5,
                                                 extra=lambda ff: wind_base(ff) - 1.0 + 0.0))
        self.wisps = []
        for k in range(3):
            def a2(ff, k=k):
                _, an = elder_fn(elder_pose(ff), ff / FPS, anchors_only=True)
                return an['bun'] + np.array([0.01 * k, -0.012 * k])
            c = Chain(5, 0.022 + 0.004 * k, a2, dir0=(1.0, 0.2), drag=8.0, iters=4, damp=0.97, grav=3.0)
            c.simulate(F0, F1, make_wind_fn(1.0, 0.6, 30.0 + k, lift=0.3, flutter=1.4,
                                            extra=lambda ff: wind_base(ff) - 1.0))
            self.wisps.append(c)
        rng = np.random.default_rng(9)
        self.embers = fire.ParticleSim(seed=13, cap=5000)
        tcache = {}

        def tw(ff):
            k = int(ff)
            if k not in tcache:
                tcache[k] = torch_world(k)[0]
            return tcache[k]

        base_fire = np.array([X_CAIRN, GK + ch.Cairn().bk_bot + 0.1, Z])

        def emitter(sim, ff, dt):
            n = rng.poisson(8.0 * dt)
            if n > 0:
                b = tw(ff)
                pos = b + np.stack([rng.normal(0, 0.018, n), rng.uniform(0.05, 0.22, n), rng.normal(0, 0.015, n)], 1)
                vel = np.stack([rng.normal(0.15, 0.25, n), rng.uniform(0.5, 1.3, n), rng.normal(0, 0.14, n)], 1)
                kind = (rng.random(n) >= 0.15).astype(np.int64)
                sim.spawn(pos, vel, rng.gamma(2.0, 0.7, n) + 0.3, rng.random(n) ** 2.5 * 1.6 + 0.18, kind,
                          np.where(kind == 0, 1.0, rng.uniform(0.6, 0.85, n)))
            if ff >= CATCH:
                lv = self.cairn_level(ff)
                if V2:       # the burst rides the whoosh ON the catch frame
                    rate = 20.0 + 60.0 * min(1.5, lv) + 220.0 * smoothstep(CATCH - 0.5, CATCH + 1.5, ff) * (1 - smoothstep(CATCH + 6, CATCH + 24, ff))
                else:
                    rate = 20.0 + 60.0 * min(1.5, lv) + 200.0 * smoothstep(CATCH + 4, CATCH + 8, ff) * (1 - smoothstep(CATCH + 12, CATCH + 30, ff))
                m = rng.poisson(rate * dt)
                if m > 0:
                    pos = base_fire + np.stack([rng.normal(0, 0.14, m), rng.uniform(0.1, 0.8, m), rng.normal(0, 0.08, m)], 1)
                    vel = np.stack([rng.normal(0.4, 0.5, m), rng.uniform(1.0, 3.2, m), rng.normal(0, 0.3, m)], 1)
                    kind = (rng.random(m) >= 0.25).astype(np.int64)
                    sim.spawn(pos, vel, rng.gamma(2.0, 0.7, m) + 0.3, rng.random(m) ** 2.5 * 1.8 + 0.2, kind,
                              np.where(kind == 0, 1.0, rng.uniform(0.6, 0.85, m)))

        def params(ff):
            w = wind_at(ff / FPS, wind_base(ff), 0.5, 1.0)
            return dict(wind=(w * 0.8, 0.2, 0.0), buoy=2.4, drag=1.3, turb=0.6, turb_sc=2.0, grav=0.0, cool=1.0)

        self.embers.run(F0, F1, emitter, params, substeps=3)
        self._sim = True

    def cairn_level(self, f):
        if V2:
            # the kindling catches one frame before the hit (the torch has been in the basket
            # since 2638), and the flame WHOOSHES up with overshoot ON 2640 (the music's hit)
            if f < CATCH - 1:
                return 0.0
            if f < CATCH:
                return 0.10 + 0.05 * (f - (CATCH - 1))
            tr = (f - CATCH) / FPS + 0.025
            return 0.12 + 1.15 * fire.ignition(tr, overshoot=0.75, rise=0.22, settle=0.8)
        if f < CATCH:
            return 0.0
        tr = (f - CATCH) / FPS
        grow = smoothstep(0.0, 0.7, tr)
        return grow * (0.25 + 0.75 * smoothstep(0.1, 0.8, tr)) + 0.8 * fire.ignition(tr - 0.25, 0.6, 0.35, 0.9) * (tr > 0.25)

    def render(self, f, scale=0.5):
        self.simulate()
        t = f / FPS
        cam, focus = camera(f, scale)
        wind = wind_at(t, wind_base(f), 0.5, 1.0)
        XE, XC = xs(f)
        GE, GC = ground_y(XE), ground_y(XC)
        ep = elder_pose(f)
        cp, cf = child_pose(f)
        tfl, holder = torch_world(f)
        flick = fire.flicker(t, 1.7, 1.0)
        tI = np.array([3.4, 1.65, 0.55]) * flick
        cl = self.cairn_level(f)
        cflick = fire.flicker(t, 4.1, 1.3)
        cairn = self.scene.cairn
        fire_base = np.array([X_CAIRN, GK + cairn.bk_bot + 0.10, Z])
        beacons = self.old + self.wave
        if V2:
            img, dep = fires2.background_v2(self.scene, cam, self.sky, f, beacons, render_full_sky)
            fires2.draw_fires(img, dep, cam, beacons, f, wind=wind_at(t, wind_base(f), 0.5, 1.0))
        else:
            img, dep = self.scene.background(cam, self.sky, f, beacons, crest=False)
        # ship lights (slow), after the crane starts
        self._ships(img, cam, f)
        if focus is not None:
            coc = 0.014 * cam.f / max(focus, 0.3)
            if coc * 0.42 > 0.6:
                img = blur_big(img, coc * 0.42)
        pool = [(tfl[0], ground_y(tfl[0]), Z, 1.1, 0.07 * flick, 0.032 * flick, 0.010 * flick, 'crest')]
        if cl > 0:
            I = min(cl, 1.6) * cflick
            pool.append((X_CAIRN, GK, Z, 2.4, 0.14 * I, 0.06 * I, 0.018 * I, 'crest'))
        cr = render_crest(cam, self.sky, f, pool)
        if V2:
            fires2.crest_texture(cr[0], cr[1], cam.params(), float(Z))
        over(img, cr[0], cr[1])
        lights = [[tfl[0], tfl[1] + 0.12, Z - 0.05, tI[0], tI[1], tI[2], 0.30, 0.0]]
        if cl > 0:
            I = min(cl, 1.6) * cflick
            lights.append([fire_base[0], fire_base[1] + 0.45, Z - 0.1, 7.0 * I, 3.2 * I, 1.0 * I, 0.55, 0.0])
        lights = np.array(lights, np.float64)
        amb_top = np.array([0.012, 0.018, 0.045])
        back = np.array([0.16, 0.10, 0.14]) if f < CUT else np.array([0.12, 0.08, 0.12])
        items = [Figure([0.0, 0.0, Z + 0.02], [self.scene.grass.group(t, wind)]),
                 Figure([0.0, GK, Z + 0.01], cairn.groups(x=X_CAIRN))]
        if V2:
            r = items[0].render(cam, lights, amb_top, np.zeros(3), back=back)
            if r is not None:
                over_region(img, *r)
            r = self.cairn2.render(cam, [X_CAIRN, GK, Z + 0.01], lights, amb_top, bg=img)
            if r is not None:
                over_region(img, *r)
            r = sil.Silhouette([0.0, GK, Z + 0.01], self.cairn2.basket_groups(x=X_CAIRN)).render(
                cam, lights, amb_top, np.zeros(3), bg=img, t=t)
            if r is not None:
                over_region(img, *r)
            self._figures_v2(img, cam, f, t, ep, cp, cf, holder, lights, amb_top, GE, GC)
            items = []
        else:
            eg, ea = ch.elder(ep, t, scarf_pts=self.scarf.at(f), wisps=[w.at(f) for w in self.wisps])
            items.append(Figure([0.0, GE, Z], eg))
            cg, ca = ch.child(cp, t, facing=cf)
            if holder == 'child':
                g, d = child_torch_grip(f)
                tg, _ = ch.torch_alone(g - d * 0.16, d, length=TORCH_LEN, t=t)
                cg = [tg] + cg if cf == -1 else cg + [tg]
            items.append(Figure([0.0, GC, Z - 0.01], cg))
        blur_fig = 0.0
        for it in items:
            r = it.render(cam, lights, amb_top, np.zeros(3), back=back, blur_px=blur_fig)
            if r is not None:
                y0, x0, rgb, a = r
                over_region(img, y0, x0, rgb, a)
        # beacon smoke + flame
        if cl > 0:
            srgb = np.zeros_like(img)
            sa = np.zeros(img.shape[:2], np.float32)
            b = fire_base + np.array([0.0, 0.9, 0.02])
            pts = np.array([b + np.array([-1.0, -0.2, 0]), b + np.array([7.0, 6.0, 0])])
            sx, sy, z = cam.project(pts)
            if np.all(z > 0.05):
                I = min(cl, 1.6) * cflick
                if V2:
                    fires2.smoke_wisp(srgb, sa, np.full(img.shape[:2], 1e9, np.float32), cam.params(),
                                      float(b[0]), float(b[1]), float(b[2]), t, 2.3, 5.5,
                                      0.22, 0.26, 0.42 * wind, 0.9, 0.42 * smoothstep(CATCH, CATCH + 20, f),
                                      0.25 * I, 0.09 * I, 0.025 * I, 0.8, 0.004, 0.005, 0.010,
                                      int(min(sx)) - 20, int(min(sy)) - 20, int(max(sx)) + 20, int(max(sy)) + 20)
                else:
                    fire.render_smoke(srgb, sa, cam.params(), float(b[0]), float(b[1]), float(b[2]), t, 2.3, 5.0,
                                      0.2, 0.28, 0.5 * wind, 0.8, 0.5 * smoothstep(CATCH, CATCH + 20, f),
                                      0.25 * I, 0.09 * I, 0.025 * I, 0.8, 0.004, 0.005, 0.010,
                                      int(min(sx)) - 20, int(min(sy)) - 20, int(max(sx)) + 20, int(max(sy)) + 20)
                over(img, srgb, sa)
        fa = np.zeros(img.shape[:2], np.float32)
        if cl > 0:
            g2 = min(cl, 1.8)
            fire.draw_flame(img, fa, cam, fire_base, 0.35 + 0.80 * g2, 0.30, 0.12 + 0.15 * wind, t, 2.9,
                            4.5 + 4.0 * min(1.0, cl), fire.BONFIRE_STYLE)
            fire.add_glow(img, cam, fire_base + np.array([0, 0.6, 0]), 0.8, 0.06 * min(cl, 1.6) * cflick)
            if V2:
                # the iron cage stands dark against its own fire
                r = sil.Silhouette([0.0, GK, Z + 0.01], self.cairn2.basket_groups(x=X_CAIRN, only_iron=True)).render(
                    cam, lights, amb_top, np.zeros(3), bg=img, t=t)
                if r is not None:
                    y0, x0, rgb, a = r
                    over_region(img, y0, x0, rgb * 0.72, a * 0.72)
        # torch flame
        lean = 0.10 + 0.16 * wind
        fire.draw_flame(img, fa, cam, tfl + np.array([0.0, -0.035, 0.0]), 0.30 * (0.95 + 0.08 * flick),
                        0.050, lean, t, 1.7, 7.5 * flick, fire.TORCH_STYLE)
        fire.add_glow(img, cam, tfl + np.array([0, 0.12, 0]), 0.10, 0.10 * flick)
        P, V, T, S, K = self.embers.snap[f]
        fire.render_sparks(img, cam.params(), P, V, T, S, K, 0.5 / FPS, 2.0 * cam.scale ** 2 * 6.0,
                           cam.scale * 1.5, focus if focus else 10.0, 0.0, 1.0, 1.5, 0.05)
        out = look.finish(img, exposure=1.0, bloom_strength=0.085, bloom_threshold=0.9, vignette_amount=0.24)
        return out

    def _figures_v2(self, img, cam, f, t, ep, cp, cf, holder, lights, amb_top, GE, GC):
        """The Elder and the Child as ONE merged silhouette (rims only on the outer contour)."""
        eg, ea = f2.elder2(ep, t, scarf_pts=self.scarf.at(f), wisps=[w.at(f) for w in self.wisps])
        cg, ca = f2.child2(cp, t, facing=cf)
        if holder == 'child':
            g, d = child_torch_grip(f)
            tgs, _ = f2.torch_alone2(g - d * 0.16, d, length=TORCH_LEN, t=t)
            cg = tgs + cg if cf == -1 else cg + tgs
        f2.translate(cg, 0.0, GC - GE)
        fig = sil.Silhouette([0.0, GE, Z], eg + cg)
        r = fig.render(cam, lights, amb_top, np.zeros(3), bg=img, t=t)
        if r is not None:
            over_region(img, *r)

    def _ships(self, img, cam, f):
        # a few slow lights crossing the sky (future traffic), steady with a soft blink on one
        ships = [(-30.0, 11.0, 0.10, 0.03, 1.0), (8.0, 17.0, -0.06, 0.02, 0.7), (18.0, 7.0, -0.05, 0.035, 0.8),
                 (-6.0, 23.0, 0.04, -0.01, 0.6)]
        t = (f - 2600) / FPS
        for k, (a0, e0, da, de, br) in enumerate(ships):
            d = dir_from_az_el(a0 + da * t * 3.0, e0 + de * t * 3.0)
            sx, sy, z = cam.project(cam.pos + d * 1e6)
            if z <= 0:
                continue
            blink = 1.0 if k != 1 else (0.55 + 0.45 * (math.sin(t * 3.0) > 0.6))
            e = 3.0 * br * blink * cam.scale ** 2
            splat_gauss(img, float(sx), float(sy), max(0.35, 0.6 * cam.scale), e * 1.0, e * 0.93, e * 0.82)
