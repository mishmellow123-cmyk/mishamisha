"""Scene renderer for the festival hill (INTRO + CODA).

State for one frame is a dict built by the shot scripts (intro.py / coda.py):
  cam, sky, t (seconds, global), elder pose, child pose, torch owner, beacon fires,
  cairn fire level, ember emitters, dof settings, exposure ...
"""
import math

import cv2
import numpy as np

import characters as ch
import fire
import hillworld as hw
from core import (Camera, fnoise1, look, over, over_region, smoothstep, splat_gauss)
from land import pack_ridges, render_ridges
from puppet import Figure, Group, Chain
from sky import render_full_sky

FPS = 24.0
CREST_Z = hw.CREST_Z

_RIDGES = None


def ridges():
    global _RIDGES
    if _RIDGES is None:
        rs = hw.build_ridges() + [hw.crest_ridge()]
        _RIDGES = pack_ridges(rs)
    return _RIDGES


def ground_y(x):
    return hw.crest_height(x)


# ------------------------------------------------------------------ wind ---

def wind_at(t, base=2.4, gust=0.5, seed=0.0):
    """Horizontal wind speed (m/s, +x = to the right) with gusts."""
    g = fnoise1(t * 0.35, seed + 1.0, 3)
    g2 = fnoise1(t * 1.3, seed + 4.0, 2)
    return base * (1.0 + gust * (0.8 * g + 0.35 * g2))


def make_wind_fn(base, gust, seed, lift=0.0, flutter=1.0, extra=None):
    """Air velocity at chain nodes: gusting horizontal + travelling flutter wave."""
    def fn(f, P):
        t = f / FPS
        w = wind_at(t, base, gust, seed)
        if extra is not None:
            w += extra(f)
        n = len(P)
        s = np.arange(n) / max(1, n - 1)
        wx = np.full(n, w)
        wy = lift + flutter * (0.9 * np.sin(s * 9.0 - t * 11.0 + seed) + 0.5 * np.sin(s * 17.0 - t * 19.0))
        wy *= (0.25 + 0.75 * s) * (0.5 + 0.25 * abs(w))
        wx += flutter * 0.3 * np.cos(s * 7.0 - t * 9.0)
        return np.stack([wx, wy], 1)
    return fn


# ----------------------------------------------------------------- grass ---

class Grass:
    """Tufts along the crest line near the figures (z = CREST_Z)."""

    def __init__(self, x0, x1, seed=3, density=9.0):
        rng = np.random.default_rng(seed)
        n = int((x1 - x0) * density)
        self.tufts = []
        xs = np.sort(rng.uniform(x0, x1, n))
        for x in xs:
            k = rng.integers(3, 8)
            h = rng.uniform(0.07, 0.24) * (0.5 + 0.5 * rng.random())
            blades = [(rng.uniform(-0.03, 0.03), h * rng.uniform(0.55, 1.1), rng.uniform(-0.35, 0.35),
                       rng.uniform(0.0025, 0.0045), rng.uniform(0, 6.28)) for _ in range(k)]
            self.tufts.append((x, blades))

    def group(self, t, wind, xmin=-1e9, xmax=1e9, lean_bias=0.0):
        g = ch.G('grass', 'coat', k=0.0, sheen=1.4, soft=0.05, sky_rim=0.5)
        for (x, blades) in self.tufts:
            if x < xmin or x > xmax:
                continue
            gy = ground_y(x) - 0.01
            for (dx, h, ang, w, ph) in blades:
                bx = x + dx
                sway = 0.22 * (wind / 2.4) + 0.10 * math.sin(t * 2.3 + ph + bx * 3.0) * (wind / 2.4) + lean_bias
                a = ang * 0.5 + sway
                p0 = np.array([bx, gy])
                p1 = p0 + np.array([math.sin(a * 0.4) * h * 0.45, math.cos(a * 0.4) * h * 0.45])
                p2 = p1 + np.array([math.sin(a * 1.0) * h * 0.35, math.cos(a * 1.0) * h * 0.35])
                p3 = p2 + np.array([math.sin(a * 1.6) * h * 0.25, math.cos(a * 1.6) * h * 0.25])
                g.chain([p0, p1, p2, p3], [w, w * 0.8, w * 0.55, w * 0.2], k=0.0)
        return g


# --------------------------------------------------------------- beacons ---

class Beacon:
    def __init__(self, layer, x, y, z, f_ign, size=3.0, seed=0.0, bell=False):
        self.layer, self.x, self.y, self.z = layer, x, y, z
        self.f_ign = f_ign
        self.size = size
        self.seed = seed
        self.bell = bell

    def level(self, f):
        tr = (f - self.f_ign) / FPS
        if tr <= 0:
            return 0.0
        return fire.ignition(tr, overshoot=0.7, rise=0.35, settle=0.8) * fire.flicker(f / FPS, self.seed, 1.4)


def place_beacons(specs, ridge_list):
    """specs: [(layer_index, screen_frac_x_hint in metres X, f_ign, size)] -> Beacons on ridge peaks."""
    out = []
    for (li, X, f_ign, size, seed) in specs:
        r = ridge_list[li]
        xs = np.linspace(r.x0, r.x1, len(r.h))
        # snap to the local maximum within +-4% of depth
        w = 0.04 * r.z
        m = (xs > X - w) & (xs < X + w)
        idx = np.nonzero(m)[0]
        i = idx[np.argmax(r.h[idx])]
        out.append(Beacon(li, xs[i], r.h[i], r.z, f_ign, size, seed))
    return out


def beacon_lights(beacons, f, layer_of):
    """Warm glow on the hills around each lit beacon (for render_ridges)."""
    L = []
    for b in beacons:
        lv = b.level(f)
        if lv <= 0:
            continue
        li = layer_of[b.layer]
        rad = 18.0 * b.size / 3.0 + 0.004 * b.z
        e = 0.030 * lv * (b.size / 3.0)
        L.append([b.x, b.y, b.z, rad, e * 1.0, e * 0.42, e * 0.12, li])
    return np.array(L, np.float64).reshape(-1, 8)


def draw_beacons(img, depth, cam, beacons, f):
    """Distant beacons: small flames/points with a halo; nearer ones get a real flame."""
    alpha_dummy = np.zeros(img.shape[:2], np.float32)
    for b in beacons:
        lv = b.level(f)
        if lv <= 0:
            continue
        sx, sy, z = cam.project(np.array([b.x, b.y, b.z]))
        if z <= 0 or sx < -80 or sx > cam.W + 80 or sy < -80 or sy > cam.H + 80:
            continue
        hpx = cam.f * b.size / z * (0.85 + 0.25 * lv)
        if hpx > 9:
            # real little flame
            fire.draw_flame(img, alpha_dummy, cam, np.array([b.x, b.y, b.z]), b.size * lv ** 0.5,
                            b.size * 0.35, 0.35, f / FPS + b.seed * 7, b.seed, 5.0 * lv, fire.BONFIRE_STYLE)
            core_e = 0.0
        else:
            core_e = 1.0
        s = cam.scale
        e = (14.0 + 5.0 * b.size) * lv * s * s * core_e
        halo = max(2.0, 3.2 * hpx)
        he = (2.0 + 0.8 * b.size) * lv * s * s
        fire.splat_beacon(img, depth, float(sx), float(sy), float(z), max(1.2 * s, hpx), e,
                          halo, he, 1.0, 0.55, 0.20)


# ------------------------------------------------------------ the scene ---

class HillScene:
    def __init__(self, seed=5):
        self.grass = Grass(-3.0, 9.0, seed=seed)
        self.cairn = ch.Cairn()

    def background(self, cam, sky, f, beacons, extra_lights=()):
        """sky + ridges + distant beacons. Returns (img, depth)."""
        t = f / FPS
        img = render_full_sky(cam, sky, t)
        rs, meta, hs = ridges()
        layer_of = {i: len(rs) - 1 - (len(rs) - 1 - i) for i in range(len(rs))}
        # rs sorted far->near; build map from hw layer index (0 near .. 7 far) to packed index
        name_to_idx = {r.name: k for k, r in enumerate(rs)}
        lmap = {i: name_to_idx[f'L{i}'] for i in range(len(hw.LAYERS))}
        lmap['crest'] = name_to_idx['crest']
        lights = beacon_lights(beacons, f, lmap)
        if len(extra_lights):
            ex = np.array([[l[0], l[1], l[2], l[3], l[4], l[5], l[6], lmap[l[7]]] for l in extra_lights], np.float64)
            lights = np.concatenate([lights, ex], 0)
        rgbp = np.zeros_like(img)
        a = np.zeros(img.shape[:2], np.float32)
        dep = np.full(img.shape[:2], 1e9, np.float32)
        render_ridges(rgbp, a, dep, cam.params(), meta, hs, sky.packed(), hw.FOG, lights,
                      np.zeros(1), 1.0, t)
        over(img, rgbp, a)
        draw_beacons(img, dep, cam, beacons, f)
        return img, dep
