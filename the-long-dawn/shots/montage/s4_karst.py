"""Shot 4 (1640-1679) KARST.

One idea: one spark in a sea of mist.
Layer upon layer of limestone towers (Guilin / Zhangjiajie) rising from a moonlit mist sea and
dissolving into haze with distance; on the crown of the nearest tower a beacon flares on the hit
at 1650 and its glow blooms in the mist. Built as true 2.5D: silhouette layers at real depths,
so the slow lateral camera drift gives real parallax. Lower third (text 1620-1700) = mist sea.
"""
import math

import numpy as np

import common as CM
from mt import sky as SK, fire as F, figure as FG, props as PR, people as PP
from mt.cam import Cam, keys, smoother, kick
from mt.noise import fbm2, gnoise2

START, END, IGN = 1640, 1679, 1650
HFOV = 44.0
F_FULL = 960.0 / math.tan(math.radians(HFOV / 2))
HORIZON_Y0 = 440.0
MIST_Y = -24.0
MOON_DIR = np.array([-0.42, 0.34, 0.84])
MOON_DIR = MOON_DIR / np.linalg.norm(MOON_DIR)
DEPTHS = [2600.0, 1700.0, 1100.0, 720.0, 470.0, 300.0, 190.0, 95.0]
HERO_Z = 95.0
HERO_X = 13.6
HERO_TOP = 4.0


def _layer_towers(k, z, rng):
    """Towers of layer k: arrays of (x, half-width, top, bulge-phase)."""
    span = z * 1.4 + 200
    spacing = max(z * 0.11, 9.0)
    xs = np.arange(-span, span, spacing) + rng.random(int(np.ceil(2 * span / spacing)) + 1)[:len(np.arange(-span, span, spacing))] * spacing * 0.8
    n = len(xs)
    w = spacing * (0.15 + 0.17 * rng.random(n))
    top = MIST_Y + (22 + 70 * rng.random(n) ** 0.8) * (0.7 + 0.5 * min(z / 1200.0, 1.0))
    if z == HERO_Z:
        # the hero tower (beacon) + two companions; keep the rest of this layer at the frame edges
        keep = (np.abs(xs - HERO_X) > 45)
        xs, w, top = xs[keep], w[keep], top[keep] + 15
        xs = np.r_[xs, HERO_X]
        w = np.r_[w, 7.0]
        top = np.r_[top, HERO_TOP]
    ph = rng.random(len(xs)) * 6.28
    return np.stack([xs, w, top, ph], 1)


_TOWERS = None


def towers():
    global _TOWERS
    if _TOWERS is None:
        rng = np.random.default_rng(404)
        _TOWERS = [_layer_towers(k, z, rng) for k, z in enumerate(DEPTHS)]
    return _TOWERS


def camera(frame, W=1920, H=804):
    t = CM.ftime(frame)
    tilt0 = math.degrees(math.atan((HORIZON_Y0 - 402.0) / F_FULL))
    x = keys(frame, [(START, 1.6), (END, -2.2)], ease=lambda u: u * 0.75 + 0.25 * u * u * (3 - 2 * u))
    k = kick(t, CM.ftime(IGN), amp=1.0, freq=5.0, decay=5.0)
    return Cam((x, 0.0 + 0.02 * k, 0.0 + 0.8 * (frame - START) / 40.0), 0.0, tilt0 + 0.06 * k, HFOV, W, H)


def _sky(cam, t, S):
    H, W = cam.H, cam.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    vx = (xx + 0.5 - cam.cx) / cam.f
    vy = (cam.cy + cam.shift - (yy + 0.5)) / cam.f
    n = np.sqrt(vx * vx + vy * vy + 1.0)
    dx = (cam.right[0] * vx + cam.fwd[0]) / n
    dy = vy / n
    dz = (cam.right[2] * vx + cam.fwd[2]) / n
    zc = CM.lin('#070B1C')
    hc = CM.lin('#2B3A6A')
    u = np.clip(dy / 0.6, 0, 1) ** 0.55
    img = hc[None, None, :] * (1 - u[..., None]) + zc[None, None, :] * u[..., None]
    cosm = np.clip(dx * MOON_DIR[0] + dy * MOON_DIR[1] + dz * MOON_DIR[2], -1, 1)
    ang = np.arccos(cosm)
    mc = CM.lin('#9DB4D9')
    halo = 0.06 * np.exp(-ang / 0.12) + 0.03 * np.exp(-(ang / 0.55) ** 2)
    img += mc[None, None, :] * halo[..., None]
    R = math.radians(1.1)
    pix = 1.0 / cam.f
    disk = np.clip((R - ang) / pix + 0.5, 0, 1)
    img += (np.array([1.0, 0.97, 0.92]) * 5.0)[None, None, :] * disk[..., None]
    return img.astype(np.float32), dy


def render(frame, scale=0.5):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    t = CM.ftime(frame)
    cam = camera(frame, W, H)
    S = None
    img, dy = _sky(cam, t, S)
    depth = np.full((H, W), 1e9, np.float32)
    SK.splat_stars(img, cam, _stars(), np.ones((H, W), np.float32) * np.clip(1 - np.exp(-((dy - 0.02) / 0.1)), 0, 1),
                   t=t, gain=0.6, scale=scale)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ti = CM.ftime(IGN)
    size, inten, light = F.ignite_env(t, ti)
    fl = F.flicker(t, 21)
    hero_top_world = np.array([HERO_X, HERO_TOP, HERO_Z])
    fire_base = np.array([HERO_X - 1.6, HERO_TOP + 0.95 + 0.07, HERO_Z])
    fire_I = 7.0 * light * fl
    mist_col = CM.lin('#8FA3C4') * 0.16
    haze_col = CM.lin('#3B4F78') * 0.55
    moon_col = CM.lin('#9DB4D9')
    for k, z in enumerate(DEPTHS):
        TW = towers()[k]
        zc = z - cam.pos[2]
        # world coords of each pixel on this layer's plane
        X = cam.pos[0] + (xx + 0.5 - cam.cx) * zc / cam.f
        Y = cam.pos[1] + (cam.cy + cam.shift - (yy + 0.5)) * zc / cam.f
        ppm = cam.f / zc
        inside = np.full((H, W), -1e9)
        rim = np.zeros((H, W))
        xrow = X[0]
        for (tx, tw, ttop, tph) in TW:
            sx0 = cam.cx + (tx - tw * 1.6 - cam.pos[0]) * cam.f / zc
            sx1 = cam.cx + (tx + tw * 1.6 - cam.pos[0]) * cam.f / zc
            if sx1 < 0 or sx0 > W:
                continue
            c0 = max(0, int(sx0) - 1)
            c1 = min(W, int(sx1) + 2)
            Xs = X[:, c0:c1]
            Ys = Y[:, c0:c1]
            # column half-width bulges with height (karst towers are barrel-like), rough edges
            hy = np.clip((Ys - MIST_Y) / max(ttop - MIST_Y, 1.0), 0, 1.2)
            # slightly tapered column with irregular fissured edges (noise in height)
            nY = np.sin(Ys * 0.23 + tph) * 0.5 + np.sin(Ys * 0.61 + 2.3 * tph) * 0.3 + np.sin(Ys * 1.7 + tph) * 0.2
            wy = tw * (1.05 - 0.2 * hy + 0.10 * nY)
            dxw = np.abs(Xs - tx)
            # rounded crown with tufts of trees
            dome = np.sqrt(np.clip(1 - (dxw / (tw * 0.95)) ** 2, 0, 1))
            tf = np.abs(np.sin(Xs * 1.9 + tph)) * 0.6 + np.abs(np.sin(Xs * 0.77 + 2 * tph)) * 0.4
            tufts = (0.9 + 0.6 * tw / 8.0) * tf * dome
            crown = ttop + (dome - 1.0) * tw * 0.45 + tufts
            ins = np.minimum(wy - dxw, crown - Ys)
            better = ins > inside[:, c0:c1]
            inside[:, c0:c1] = np.where(better, ins, inside[:, c0:c1])
            # moon-side rim: left edges catch moonlight (moon is up-left)
            side = np.clip(-(Xs - tx) / (tw + 1e-6), 0, 1)
            r_here = side * np.exp(-np.clip(wy - dxw, 0, None) / (tw * 0.25)) + 0.6 * np.exp(-np.clip(crown - Ys, 0, None) / (tw * 0.18))
            rim[:, c0:c1] = np.where(better, r_here, rim[:, c0:c1])
        cov = np.clip(inside * ppm + 0.5, 0, 1)
        if cov.max() <= 0:
            continue
        # rock colour: dark green-grey, moonlit rims, vertical limestone streaks
        streak = 0.8 + 0.2 * np.sin(X * 2.1 + np.sin(Y * 0.07) * 3.0)
        base = np.array([0.012, 0.016, 0.018]) * 1.0
        lit = moon_col * 0.06
        col = base[None, None, :] * streak[..., None] + lit[None, None, :] * rim[..., None] * streak[..., None]
        # fire light on the hero crown
        if z == HERO_Z and fire_I > 0:
            d2 = (X - fire_base[0]) ** 2 + (Y - fire_base[1]) ** 2 + 1.0
            E = fire_I * 1.6 / d2 * np.clip((Y - (HERO_TOP - 14)) / 14, 0, 1)
            col += (F.FIRE_LIGHT * 0.06)[None, None, :] * E[..., None]
        # aerial perspective + mist swallowing the tower feet
        Ta = math.exp(-z / 620.0)
        mist = np.clip((MIST_Y + 10.0 - Y) / 26.0, 0, 1) ** 1.1
        mist_n = 0.75 + 0.25 * np.sin(X * 0.02 + t * 0.1 + k) * np.sin(X * 0.007 - t * 0.05)
        col = col * Ta + haze_col[None, None, :] * (1 - Ta)
        col = col * (1 - mist[..., None] * mist_n[..., None]) + mist_col[None, None, :] * (mist * mist_n)[..., None]
        img = img * (1 - cov[..., None]) + col.astype(np.float32) * cov[..., None]
        depth = np.where(cov > 0.5, zc, depth).astype(np.float32)
        # a drifting mist sheet in front of each layer (below the crowns)
        sheet_y = MIST_Y + 4.0 + 5.0 * np.sin(k * 1.7)
        dens = np.clip((sheet_y + 10.0 - Y) / 22.0, 0, 1) * (0.55 + 0.45 * np.sin(X * 0.011 + t * 0.15 + k * 2.1)) * 0.4
        img = img * (1 - dens[..., None]) + (mist_col * 1.1)[None, None, :] * dens[..., None]
    # beacon, figure, fire
    feet = np.array([HERO_X + 1.3, HERO_TOP + 0.05, HERO_Z])
    cairn = np.array([HERO_X - 1.6, HERO_TOP, HERO_Z])
    back, front, fb, rb = PR.cairn(seed=41, height=0.95)
    arm = keys(frame, [(START, 0.55), (1646, 0.55), (1649.5, 1.0), (1655, 0.7), (END, 0.4)], ease=smoother)
    lean = keys(frame, [(START, 0.05), (1649.5, 0.15), (1656, 0.0), (END, 0.0)], ease=smoother)
    fig, pts = PP.torch_person('parka', arm=arm, lean=lean)
    th = pts['torch_head']
    lights = [dict(dir=MOON_DIR, col=moon_col, I=0.25)]
    tflip = True
    torch_w = PR.local_to_world(cam, feet, -th[0], th[1], 0.0)
    lights.append(dict(pos=torch_w, col=F.FIRE_LIGHT, I=0.4, r0=0.12))
    if fire_I > 0:
        lights.append(dict(pos=fire_base + np.array([0, 0.6, 0]), col=F.FIRE_LIGHT, I=fire_I * 0.8, r0=0.3))
    famb = CM.lin('#27335E') * 0.3
    # glow of the fire in the mist (big, soft) -- drawn before the props so they silhouette against it
    if fire_I > 0:
        sx, sy, zf = cam.project(fire_base + np.array([0, 0.8, 0]))
        F.halo(img, depth, sx, sy, 330 * scale * min(size, 1.2), 0.05 * light * fl, z=1e9)
        F.halo(img, depth, sx, sy, 90 * scale * min(size, 1.2), 0.10 * light * fl, z=1e9)
    FG.render(img, depth, cam, back, cairn, lights, amb=famb, t=t, emissive_gain=min(light, 1.5) * 1.2)
    FG.render(img, depth, cam, fig, feet, lights, amb=famb, t=t, seed=17, flipx=tflip)
    F.flame(img, depth, cam, torch_w, 0.3, 0.05, t, seed=23, I=16.0, tongues=3, lean=0.05, zbias=0.1)
    if size > 0:
        _smoke().render(img, depth, cam, t, fire_base + np.array([0, 0.9, 0]), 0.5 * light * fl, albedo=0.2,
                        amb=(0.004, 0.005, 0.008))
        F.flame(img, depth, cam, fire_base, 2.2 * size, rb, t, seed=8, I=26.0 * inten, lean=0.2 * size, zbias=0.4)
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
        _ST = SK.make_stars(9000, 404, lum_scale=4.0)
    return _ST


def _smoke():
    global _SM
    if _SM is None:
        _SM = F.Smoke(71, np.array([HERO_X - 1.6, HERO_TOP + 3.2, HERO_Z]), CM.ftime(IGN) + 0.15, CM.ftime(END) + 0.1,
                      rate=4.0, wind=(0.8, 0, 0.2), rise=1.4, life=4.0, r0=0.3, growth=0.6, dens=0.35)
    return _SM


def _sparks():
    global _SP
    if _SP is None:
        _SP = F.Sparks(73, np.array([HERO_X - 1.6, HERO_TOP + 1.4, HERO_Z]), CM.ftime(IGN), CM.ftime(END) + 0.1,
                       burst=300, rate=55, ember_rate=12, wind=(1.0, 0.0, 0.2), I=30.0)
    return _SP


FINISH = dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)
