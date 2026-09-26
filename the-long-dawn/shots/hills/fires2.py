"""Distant fires, v2 (CODA dept, Sep 2026): the answering fires and the festival fires.

v1 drew every distant fire as a splat (two-lobe core + a round halo), ~50 of them: across the
valley they read as a field of golden bokeh. Here:

* fewer, distinct fires, each on a ridgeline that is really visible from the camera, igniting
  outward in a wave - one fire per answering-fire whump in the SFX (music/src/sfx_v2.py
  answering_fire_01..12, v2 2805..2876 = src 2645..2716), a couple of far, unvoiced ones between;
* each fire is a small real flame (supersampled teardrop body, 2-3 licking tongues, hot core
  low, broken tips, flicker, lean in the wind) that WHOOMPS up with overshoot on its frame;
* a restrained glow (air scatter close round the flame, no round bokeh halo), the hillside
  lit round it (land.render_ridges lights) and a wisp of smoke lit orange from below, drifting
  downwind, occluded by nearer ridges.

`AnswerFire` keeps the old hillscene.Beacon attributes (layer, x, y, z, f_ign, size, seed,
level()), so edit/ember_title_fires.py can re-extract positions unchanged.
"""
import math

import numba as nb
import numpy as np

import fire
from core import FULL_W, fbm2, fbm3, perlin3, cam_ray, smoothstep
from hillscene import FPS, beacon_lights, ridges_split, render_far
from land import render_ridges
import hillworld as hw

# the SFX answering-fire whumps (src frames; v2 = src + 160) and their stereo pans
WHUMPS = [2645, 2651, 2657, 2663, 2669, 2675, 2682, 2689, 2696, 2703, 2710, 2716]

# The wave: (src ignition frame, ridge layer, screen-x target at full res, flame height m, voiced).
# Outward: the near ridges first, alternating sides, successive ridgelines later (the crane-up
# keeps the far ones in frame as the camera rises).
PLAN = [
    (2645, 0, 601, 6.2, True),      # the next hill answers first (left of the cairn)
    (2651, 2, 1673, 8.6, True),     # right: the near ridges there are hidden by our hilltop
    (2657, 0, 309, 6.0, True),
    (2663, 1, 843, 7.8, True),
    (2669, 2, 1416, 8.8, True),
    (2675, 1, 172, 7.6, True),
    (2682, 3, 1798, 10.5, True),
    (2686, 3, 641, 9.5, False),
    (2689, 2, 506, 8.6, True),
    (2696, 3, 1338, 10.2, True),
    (2699, 4, 325, 11.5, False),
    (2703, 3, 76, 10.0, True),
    (2710, 4, 1656, 12.0, True),
    (2716, 4, 829, 12.5, True),
]


class AnswerFire:
    """One distant fire. level(f): 0 before ignition, whoomp with overshoot, then flicker."""

    def __init__(self, layer, x, y, z, f_ign, height, seed, voiced=False, pre=0.6):
        self.layer, self.x, self.y, self.z = layer, x, y, z
        self.f_ign = f_ign - pre              # so the flame is already leaping ON its frame
        self.frame = f_ign                    # the sync frame (whump)
        self.height = height                  # flame height, metres, at level 1
        self.size = height / 1.35             # old Beacon 'size' units (glow on the hill)
        self.seed = seed
        self.voiced = voiced
        self.bell = voiced

    def level(self, f):
        tr = (f - self.f_ign) / FPS
        if tr <= 0:
            return 0.0
        return fire.ignition(tr, overshoot=0.85, rise=0.22, settle=0.75) * fire.flicker(f / FPS, self.seed, 1.4)


_BARE = {}


def bare_ridge(li, seed=3):
    """Ridge layer li WITHOUT its trees (same profile as hillworld.build_ridges): fires sit on
    the ground of a hilltop, not on a treetop."""
    if li in _BARE:
        return _BARE[li]
    from land import make_profile, Ridge
    L = hw.LAYERS[li]
    z = L['z']
    x0, x1 = -1.05 * z - 300, 1.05 * z + 300
    n = int(min(20000, max(3000, (x1 - x0) / (z / 3600.0))))
    drop = z * z / (2 * hw.R_EARTH)
    base = z * math.tan(math.radians(L['top'])) + drop
    amp = z * math.tan(math.radians(L['amp']))
    h = make_profile(x0, x1, n, base - drop - 0.35 * amp, amp, L['scale'], seed * 31 + li * 7,
                     octaves=8, ridged=L['ridged'], trees=None)
    r = Ridge(z, x0, x1, h, np.zeros(3), name=f'bare{li}')
    _BARE[li] = r
    return r


def festival_fires(specs, ridge_list):
    """The INTRO's festival fires (hillscene.place_beacons: the local top within +-4% of depth
    round X) as AnswerFires, on the bare ground."""
    out = []
    for (li, X, f_ign, size, seed) in specs:
        r = bare_ridge(li)
        xs = np.linspace(r.x0, r.x1, len(r.h))
        w = 0.04 * r.z
        idx = np.nonzero((xs > X - w) & (xs < X + w))[0]
        i = idx[np.argmax(r.h[idx])]
        out.append(AnswerFire(li, xs[i], r.h[i], r.z, f_ign, size * 1.7, seed, voiced=False, pre=0.0))
    return out


def _visible(cam, rl, crest, li, x, h):
    """Is the ridge point (x,h) on layer li unoccluded (by nearer layers and our crest)?"""
    r = rl[li]
    for lj in range(li):
        q = rl[lj]
        tq = (q.z - cam.pos[2]) / (r.z - cam.pos[2])
        X = cam.pos[0] + (x - cam.pos[0]) * tq
        Y = cam.pos[1] + (h - cam.pos[1]) * tq
        if q.height(X) > Y - 0.006 * q.z:
            return False
    tq = (crest.z - cam.pos[2]) / (r.z - cam.pos[2])
    X = cam.pos[0] + (x - cam.pos[0]) * tq
    Y = cam.pos[1] + (h - cam.pos[1]) * tq
    return crest.height(X) < Y - 0.03


def answering_fires_v2(rl, camera_fn, avoid=()):
    """Place PLAN on real, visible ridge peaks. camera_fn(f) -> Camera (full res); avoid = other
    fires (the festival ones) to keep clear of on screen."""
    crest = hw.crest_ridge()
    rng = np.random.default_rng(2645)
    fires = []
    info = []
    used = []
    cam0 = camera_fn(2650)
    for o in avoid:
        sx, sy, _ = cam0.project(np.array([o.x, o.y, o.z]))
        used.append((float(sx), float(sy)))
    for k, (fi, li, tx, height, voiced) in enumerate(PLAN):
        r = bare_ridge(li)
        cam = camera_fn(fi + 4)
        span = r.z * math.tan(math.radians(31.0))
        peaks = r.peaks(-span, span, min_sep=0.03 * r.z)
        best = None
        for (x, h) in peaks:
            sx, sy, z = cam.project(np.array([x, h, r.z]))
            if z <= 0 or not (30 < sx < FULL_W - 30):
                continue
            if not _visible(cam, rl, crest, li, x, h):
                continue
            # keep clear of other fires on screen
            if any(abs(sx - u[0]) < 60 and abs(sy - u[1]) < 40 for u in used):
                continue
            cost = abs(sx - tx)
            if best is None or cost < best[0]:
                best = (cost, x, h, sx, sy)
        if best is None:
            continue
        _, x, h, sx, sy = best
        used.append((sx, sy))
        seed = 17.0 + k * 1.37
        f = AnswerFire(li, float(x), float(h), float(r.z), float(fi), height * (0.92 + 0.16 * rng.random()), seed,
                       voiced=voiced)
        fires.append(f)
        pan = float(np.clip((sx - FULL_W / 2) / (FULL_W / 2), -1, 1))
        info.append(dict(frame=int(fi), frame_v2=int(fi) + 160, layer=int(li), distance_m=round(float(math.hypot(r.z, x))),
                         x_m=round(float(x), 1), y_m=round(float(h), 1), z_m=round(float(r.z)),
                         flame_height_m=round(f.height, 2), screen_x_at_ignition=round(float(sx)),
                         screen_y_at_ignition=round(float(sy)), pan=round(pan, 2), voiced=bool(voiced)))
    return fires, info


# ------------------------------------------------------------------ render ---

@nb.njit(cache=True, fastmath=True)
def small_flame(img, depth, cx, cy, z, hpx, wpx, lean, t, seed, inten, glow_e, glow_s):
    """A small HDR flame in screen space, base centre (cx, cy), height hpx, half-width wpx
    (pixels), 4x4 supersampled; occluded by `depth` (nearer than z*0.985). Adds a tight glow."""
    H = img.shape[0]
    W = img.shape[1]
    x0 = int(math.floor(cx - 2.2 * wpx - 2 - abs(lean) * hpx * 0.2))
    x1 = int(math.ceil(cx + 2.2 * wpx + 2 + abs(lean) * hpx * 1.2))
    y0 = int(math.floor(cy - 1.5 * hpx - 2))
    y1 = int(math.ceil(cy + 0.25 * hpx + 2))
    ss = 4
    for j in range(max(0, y0), min(H, y1)):
        for i in range(max(0, x0), min(W, x1)):
            d = depth[j, i]
            if d < z * 0.985:
                continue
            er = 0.0
            eg = 0.0
            eb = 0.0
            for sj in range(ss):
                for si in range(ss):
                    px = i + (si + 0.5) / ss
                    py = j + (sj + 0.5) / ss
                    v = (cy - py) / hpx
                    if v < -0.18 or v > 1.55:
                        continue
                    vc = v if v > 0.0 else 0.0
                    u = (px - cx - lean * hpx * vc ** 1.5) / wpx
                    # rising turbulence (in flame-normalised coords)
                    w1 = fbm3(u * 0.7 + seed, v * 1.2 - t * 2.1, t * 0.4 + seed, 2, 2.0, 0.5)
                    uw = u + 0.55 * w1 * (0.2 + v)
                    vv = v if v > 0.0 else 0.0
                    one = 1.0 - min(vv, 1.0)
                    # bonfire envelope: as wide as the pile at the base, licking up to the tips
                    hw_ = one ** 0.72 * (1.0 + 0.14 * w1) * (1.0 + 0.22 * fire.sstep(0.22, 0.0, vv))
                    if hw_ < 1e-3:
                        hw_ = 1e-3
                    edge = 1.0 - abs(uw) / hw_
                    tn = fbm3(uw * 3.2 + seed * 3.1, v * 1.3 - t * 3.0, t * 0.5 + seed, 3, 2.0, 0.5)
                    vtop = 0.62 + 0.62 * tn
                    tong = fire.sstep(vtop + 0.06, vtop - 0.24, v)
                    fr = fbm3(uw * 3.5 + 9.0, v * 2.6 - t * 4.2, t * 0.9 + seed, 2, 2.0, 0.5)
                    frag = fire.sstep(0.18, 0.36, fr) * fire.sstep(vtop - 0.05, vtop + 0.2, v) * fire.sstep(1.5, 1.0, v)
                    dens = fire.sstep(-0.02, 0.30, edge) * max(tong, 0.7 * frag)
                    if v < 0.0:
                        dens *= fire.sstep(-0.16, 0.0, v)
                    if dens <= 0.003:
                        continue
                    core = math.exp(-(uw / 0.40) ** 2) * math.exp(-((v - 0.10) / 0.22) ** 2) * 0.55
                    T = dens * (0.74 - 0.50 * vv) + core * dens
                    T -= 0.26 * fire.sstep(0.45, 1.1, v)
                    if T <= 0.0:
                        continue
                    if T > 1.2:
                        T = 1.2
                    cr, cg, cb = fire.bb(0.15 + 0.85 * min(T, 1.0))
                    e = inten * (T * T * T + 0.02 * T)
                    er += cr * e
                    eg += cg * e
                    eb += cb * e
            k = 1.0 / (ss * ss)
            img[j, i, 0] += er * k
            img[j, i, 1] += eg * k
            img[j, i, 2] += eb * k
    # the burning pile: a low, wide, deep-orange glow at the base
    pr = int(math.ceil(3.0 * max(wpx * 1.1, 1.0)))
    pw = max(0.6, wpx * 1.05)
    ph = max(0.45, hpx * 0.09)
    pe = inten * 0.030 * (wpx * 2.0 + 1.0) * (ph * 2.0 + 1.0)
    for j in range(max(0, int(cy) - pr), min(H, int(cy) + pr + 1)):
        for i in range(max(0, int(cx) - pr), min(W, int(cx) + pr + 1)):
            if depth[j, i] < z * 0.985 and j > cy + 1:
                continue
            dx = (i + 0.5 - cx) / pw
            dy = (j + 0.5 - cy) / ph
            w = math.exp(-0.5 * (dx * dx + dy * dy)) / (2.0 * math.pi * pw * ph) * pe
            img[j, i, 0] += w * 1.0
            img[j, i, 1] += w * 0.30
            img[j, i, 2] += w * 0.06
    # tight air-scatter glow round the flame body (not a bokeh disc): vertical ellipse
    if glow_e > 0.0:
        gy = cy - 0.45 * hpx
        sx_ = glow_s * (0.6 + 0.5 * wpx / max(hpx, 1e-3))
        sy_ = glow_s
        r = int(math.ceil(3.0 * max(sx_, sy_)))
        tot = 2.0 * math.pi * sx_ * sy_
        for j in range(max(0, int(gy) - r), min(H, int(gy) + r + 1)):
            for i in range(max(0, int(cx) - r), min(W, int(cx) + r + 1)):
                if depth[j, i] < z * 0.985 and j > cy:
                    continue
                dx = (i + 0.5 - cx) / sx_
                dy = (j + 0.5 - gy) / sy_
                w = math.exp(-0.5 * (dx * dx + dy * dy)) / tot * glow_e
                img[j, i, 0] += w * 1.0
                img[j, i, 1] += w * 0.46
                img[j, i, 2] += w * 0.14


@nb.njit(cache=True, fastmath=True)
def smoke_wisp(rgb, alpha, depth, cam, bx, by, bz, t, seed, h_max, w0, grow, drift, rise, opacity,
               lit_r, lit_g, lit_b, lit_len, amb_r, amb_g, amb_b, x0, y0, x1, y1):
    """A smoke plume like fire.render_smoke, but (a) occluded by `depth` and (b) with its billow
    noise advected in plume-local coordinates of CONSTANT scale. (fire.render_smoke divides the
    advected height Y - t*rise by the height-dependent width, so late in the film - t ~ 110 s -
    the vertical noise frequency explodes into horizontal streaks.)"""
    H = rgb.shape[0]
    W = rgb.shape[1]
    for j in range(max(0, y0), min(H, y1)):
        for i in range(max(0, x0), min(W, x1)):
            if depth[j, i] < bz * 0.985:
                continue
            dx, dy, dz = cam_ray(cam, i + 0.5, j + 0.5)
            if dz <= 1e-6:
                continue
            th = (bz - cam[2]) / dz
            X = cam[0] + th * dx - bx
            Y = cam[1] + th * dy - by
            if Y < 0 or Y > h_max:
                continue
            xc = drift * Y ** 1.15 + 0.25 * w0 * perlin3(Y * 0.6 / w0 - t * rise * 0.3, seed, t * 0.1)
            wd = w0 + grow * Y
            q = (X - xc) / wd
            if abs(q) > 2.2:
                continue
            env = math.exp(-q * q * 1.6) * fire.sstep(0.0, 0.10 * h_max, Y) * fire.sstep(h_max, 0.45 * h_max, Y)
            ws = w0 + grow * 0.5 * h_max
            tl = t - math.floor(t / 97.0) * 97.0
            n = fbm3(q * 1.6 + seed, (Y - tl * rise) * 1.1 / ws, tl * 0.12 + seed, 3, 2.0, 0.55)
            dens = env * fire.sstep(-0.05, 0.35, n + 0.12 * env)
            if dens <= 0.003:
                continue
            a = opacity * dens
            lit = math.exp(-Y / lit_len)
            cr = amb_r + lit_r * lit
            cg = amb_g + lit_g * lit
            cb = amb_b + lit_b * lit
            ao = 1.0 - a
            rgb[j, i, 0] = rgb[j, i, 0] * ao + cr * a
            rgb[j, i, 1] = rgb[j, i, 1] * ao + cg * a
            rgb[j, i, 2] = rgb[j, i, 2] * ao + cb * a
            alpha[j, i] = alpha[j, i] * ao + a


def draw_fires(img, depth, cam, fires, f, wind=2.0, sky_amb=(0.010, 0.012, 0.030), smoke=True):
    """All distant fires for frame f: smoke wisps (behind), then flames + glow."""
    t = f / FPS
    lit = []
    for b in fires:
        lv = b.level(f)
        if lv <= 0:
            continue
        sx, sy, z = cam.project(np.array([b.x, b.y, b.z]))
        if z <= 0 or sx < -120 or sx > cam.W + 120 or sy < -200 or sy > cam.H + 60:
            continue
        lit.append((b, lv, float(sx), float(sy), float(z)))
    if not lit:
        return
    s = cam.scale
    if smoke:
        srgb = np.zeros_like(img)
        sa = np.zeros(img.shape[:2], np.float32)
        camp = cam.params()
        for (b, lv, sx, sy, z) in lit:
            hm = b.height * 7.0
            grow_in = min(1.0, (f - b.f_ign) / (FPS * 2.5))      # the plume builds after ignition
            if grow_in <= 0.02:
                continue
            hmax = hm * (0.35 + 0.65 * grow_in)
            pts = np.array([[b.x - hm * 0.5, b.y, b.z], [b.x + hm * 1.6, b.y + hmax, b.z]])
            px, py, pz = cam.project(pts)
            x0, x1 = int(min(px)) - 4, int(max(px)) + 4
            y0, y1 = int(min(py)) - 4, int(max(py)) + 4
            if x1 < 0 or y1 < 0 or x0 > cam.W or y0 > cam.H:
                continue
            L = min(1.4, lv)
            smoke_wisp(srgb, sa, depth, camp, b.x, b.y + b.height * 0.45, b.z, t, b.seed * 1.7, hmax,
                       b.height * 0.30, 0.17, 0.10 * wind, b.height * 0.9, 0.55 * min(1.0, grow_in * 1.5),
                       0.15 * L, 0.052 * L, 0.012 * L, b.height * 1.6,
                       sky_amb[0] * 1.5, sky_amb[1] * 1.5, sky_amb[2] * 1.4, x0, y0, x1, y1)
        a3 = sa[..., None]
        img *= (1.0 - a3)
        img += srgb
    for (b, lv, sx, sy, z) in lit:
        hpx = cam.f * b.height / z * (0.62 + 0.38 * min(lv, 1.8))
        wpx = hpx * 0.42
        # below ~3 px a flame is a point: keep it a small bright upright spot
        if hpx < 3.0 * s:
            hpx = max(hpx, 1.6 * s)
            wpx = max(0.55 * s, hpx * 0.40)
        inten = 4.2 * min(1.6, lv) * (1.0 + max(0.0, 3.0 * s - hpx) / max(3.0 * s, 1e-3))
        glow_s = max(1.2 * s, 0.85 * hpx)
        glow_e = 0.50 * min(1.6, lv) * (wpx + 1.0) * (hpx + 2.0 * s) * 0.9
        small_flame(img, depth, sx, sy + 0.08 * hpx, z, hpx, wpx, 0.22 + 0.05 * wind, t + b.seed * 7.0,
                    b.seed, inten, glow_e, glow_s)


def fire_lights(fires, f, layer_of):
    """Hillside glow round each fire, in land.render_ridges light format (old Beacon scale)."""
    return beacon_lights(fires, f, layer_of)


def background_v2(scene, cam, sky, f, fires, render_full_sky):
    """Sky + far ridges lit by the fires (no v1 splats); returns (img, depth). Fires are drawn
    separately (draw_fires) so the caller can keep them behind the crest."""
    t = f / FPS
    img = render_full_sky(cam, sky, t)
    rs = ridges_split()[0][0]
    name_to_idx = {r.name: k for k, r in enumerate(rs)}
    lmap = {i: name_to_idx[f'L{i}'] for i in range(len(hw.LAYERS))}
    lights = fire_lights(fires, f, lmap)
    rgbp, a, dep = render_far(cam, sky, f, lights)
    img *= (1.0 - a[..., None])
    img += rgbp
    return img, dep


@nb.njit(cache=True, fastmath=True)
def crest_texture(rgb, a, cam, zc):
    """Break up the firelit hilltop: soil/turf mottling and fine vertical grass streaks on the
    crest card (in place on its premultiplied colour)."""
    H = rgb.shape[0]
    W = rgb.shape[1]
    for j in range(H):
        for i in range(W):
            if a[j, i] <= 0.0:
                continue
            dx, dy, dz = cam_ray(cam, i + 0.5, j + 0.5)
            if dz <= 1e-6:
                continue
            th = (zc - cam[2]) / dz
            X = cam[0] + th * dx
            Y = cam[1] + th * dy
            n1 = fbm2(X * 3.1 + 1.7, Y * 7.0, 4, 2.0, 0.55)
            n2 = fbm2(X * 48.0, Y * 5.0 + 3.3, 3, 2.0, 0.5)
            n3 = fbm2(X * 11.0 - 4.0, Y * 30.0, 2, 2.0, 0.5)
            m = 0.70 + 0.42 * n1 + 0.26 * n2 + 0.12 * n3
            if m < 0.28:
                m = 0.28
            if m > 1.45:
                m = 1.45
            rgb[j, i, 0] *= m
            rgb[j, i, 1] *= m
            rgb[j, i, 2] *= m
