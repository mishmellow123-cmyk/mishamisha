"""Shot 5 (1680-1719) CITY.

One idea: this is our world.
A rooftop at night: water tank on stilts, antenna, parapet; beyond, a real skyline of window
grids fading into haze. A young person in a hoodie lowers a torch into an oil-drum fire on the
hit at 1690; across the city, rooftop fires answer one after another. Built 2.5D (skyline layers
at real depths + ray-cast roof deck), camera: slow push with a slight pan toward the answers.
Lower third (text until 1700) = dark parapet and roof deck.
"""
import math

import cv2
import numpy as np

import common as CM
from mt import sky as SK, fire as F, figure as FG, props as PR, people as PP
from mt.figure import Drawing
from mt.cam import Cam, keys, smoother, kick

START, END, IGN = 1680, 1719, 1690
HFOV = 50.0
F_FULL = 960.0 / math.tan(math.radians(HFOV / 2))
HORIZON_Y0 = 430.0
ROOF_Y = -1.62            # roof deck (camera eye at 0)
STREET_Y = -42.0
PARAPET_Z = 17.0
DRUM = np.array([1.45, ROOF_Y, 10.4])
FEET = np.array([2.35, ROOF_Y, 10.0])
LAYERS = [2400.0, 1400.0, 820.0, 480.0, 290.0, 185.0]


def _buildings():
    rng = np.random.default_rng(505)
    out = []
    for k, z in enumerate(LAYERS):
        span = z * 1.25 + 60
        x = -span
        bl = []
        while x < span:
            w = (14 + 30 * rng.random()) * (0.8 + 0.6 * min(z / 1500.0, 1.0))
            gap = 2 + 6 * rng.random() if k >= 4 else 1 + 4 * rng.random()
            downtown = math.exp(-((x - 150.0 * (z / 1000.0)) / (0.55 * span)) ** 2)
            if k <= 1:
                h = 60 + 260 * rng.random() ** 1.4 * (0.4 + 0.6 * downtown)
            elif k <= 3:
                h = 35 + 140 * rng.random() ** 1.3 * (0.5 + 0.5 * downtown)
            else:
                h = 28 + 55 * rng.random() ** 1.2
            kind = rng.random()
            lit_p = 0.18 + 0.4 * rng.random()
            tone = rng.random()
            bl.append([x + w / 2, w / 2, STREET_Y + h, kind, lit_p, tone, rng.integers(0, 1 << 30)])
            x += w + gap
        out.append(np.array(bl))
    return out


_B = None


def buildings():
    global _B
    if _B is None:
        _B = _buildings()
    return _B


def camera(frame, W=1920, H=804):
    t = CM.ftime(frame)
    tilt0 = math.degrees(math.atan((HORIZON_Y0 - 402.0) / F_FULL))
    push = keys(frame, [(START, 0.0), (END, 1.3)], ease=lambda u: u * 0.7 + 0.3 * u * u * (3 - 2 * u))
    yaw = keys(frame, [(START, -1.0), (IGN, -1.0), (END, 1.2)], ease=smoother)
    k = kick(t, CM.ftime(IGN), amp=1.0, freq=5.5, decay=5.5)
    return Cam((0.0, 0.01 * k, push), yaw + 0.05 * k, tilt0 + 0.07 * k, HFOV, W, H)


def _rays(cam):
    H, W = cam.H, cam.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    vx = (xx + 0.5 - cam.cx) / cam.f
    vy = (cam.cy + cam.shift - (yy + 0.5)) / cam.f
    dx = cam.right[0] * vx + cam.fwd[0]
    dz = cam.right[2] * vx + cam.fwd[2]
    return dx, vy, dz, xx, yy      # un-normalised (dz ~ 1 along the view axis)


def _sky(cam, dx, dy, dz, t):
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    ey = dy / n
    zc = CM.lin('#070B1C')
    hc = CM.lin('#2A3558')
    u = np.clip(ey / 0.55, 0, 1) ** 0.5
    img = hc * (1 - u[..., None]) + zc * u[..., None]
    # city glow near the horizon (cool, desaturated -- fire stays the only warm thing)
    glow = np.exp(-np.clip(ey, 0, None) / 0.05) * 0.035
    img = img + np.array([0.55, 0.6, 0.75]) * glow[..., None]
    return img.astype(np.float32)


def _skyline(img, depth, cam, t, ssf=2):
    """Composite the building layers (supersampled by ssf for window anti-aliasing)."""
    H, W = img.shape[:2]
    Hs, Ws = H * ssf, W * ssf
    cs = cam.scaled(ssf)
    yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float64)
    haze = CM.lin('#2E3A5E') * 0.9
    lowglow = np.array([0.35, 0.38, 0.48])
    acc = np.zeros((Hs, Ws, 3), np.float32)
    alpha = np.zeros((Hs, Ws), np.float32)
    dep = np.full((Hs, Ws), 1e9, np.float32)
    B = buildings()
    for k in range(len(LAYERS) - 1, -1, -1):   # near to far (front-to-back compositing)
        pass
    # paint far -> near with over-compositing
    out = np.zeros((Hs, Ws, 3), np.float32)
    cov_tot = np.zeros((Hs, Ws), np.float32)
    for k, z in enumerate(LAYERS):
        zc = z - cam.pos[2]
        f = cs.f
        X = cam.pos[0] + (xx[0] + 0.5 - cs.cx) * zc / f                      # per column
        Y = cam.pos[1] + (cs.cy + cs.shift - (yy[:, 0] + 0.5)) * zc / f      # per row
        ppm = f / zc
        top = np.full(Ws, -1e9)
        bid = np.full(Ws, -1, np.int64)
        for bi, (bx, bw, btop, kind, litp, tone, seed) in enumerate(B[k]):
            m = np.abs(X - bx) <= bw
            # setbacks / spires / antennas / tanks on the roofs
            tt = btop
            if kind > 0.8 and k <= 2:
                inner = np.abs(X - bx) <= bw * 0.55
                tt = np.where(inner, btop + bw * 0.5, btop)
            elif kind > 0.6:
                tt = np.where(np.abs(X - bx) <= bw * 0.7, btop + 4.0, btop)
            if kind < 0.12 and k <= 3:
                tt = np.where(np.abs(X - bx) <= 0.6, btop + 35.0, tt)             # antenna
            if 0.3 < kind < 0.45 and k >= 3:
                tank = np.abs(X - (bx - bw * 0.3)) <= 2.2
                tt = np.where(tank, btop + 6.5, tt)                              # water tank
            tt = np.where(m, tt, -1e9)
            better = tt > top
            top = np.where(better, tt, top)
            bid = np.where(better & m, bi, bid)
        cov_row = np.clip((top[None, :] - Y[:, None]) * ppm + 0.5, 0, 1)
        # windows
        fl = 3.6
        cw = 2.3
        col = np.zeros((Hs, Ws, 3), np.float32)
        valid = bid >= 0
        bsel = np.where(valid, bid, 0)
        Bk = B[k]
        bx = Bk[bsel, 0]
        bw = Bk[bsel, 1]
        litp = Bk[bsel, 4]
        tone = Bk[bsel, 5]
        seed = Bk[bsel, 6].astype(np.int64)
        u = (X - (bx - bw)) / cw                       # window column index (float)
        v = (Y[:, None] - STREET_Y) / fl               # floor index (float)
        iu = np.floor(u).astype(np.int64)
        iv = np.floor(v).astype(np.int64)
        fu = u - iu
        fv = v - iv
        hsh = (iu[None, :] * 73856093 ^ iv * 19349663 ^ seed[None, :] * 83492791) & 0xFFFFFF
        r01 = hsh / float(0xFFFFFF)
        # floors lit in bands (offices) + random rooms
        band = ((iv * 2654435761 ^ seed[None, :]) & 0xFF) / 255.0
        lit = (r01 < litp[None, :] * (0.55 + 0.9 * band)).astype(np.float32)
        # window aperture (anti-aliased by supersampling + a soft edge)
        win_px = cw * ppm
        soft = max(0.5 / max(win_px, 1e-3), 0.02)
        ax = np.clip((0.36 - np.abs(fu[None, :] - 0.5)) / soft, 0, 1) * np.clip((0.3 - np.abs(fv - 0.55)) / (soft * cw / fl), 0, 1)
        if win_px < 1.2:
            ax = ax * 0 + 0.36 * 0.28 * 4   # unresolved: average aperture
        wc_cool = np.array([0.75, 0.88, 1.0])
        wc_warm = np.array([1.0, 0.86, 0.66])
        mixw = (tone[None, :] > 0.72).astype(np.float32)[..., None]
        wcol = wc_cool * (1 - mixw) + wc_warm * mixw
        br = (0.35 + 0.65 * ((hsh >> 7) & 255) / 255.0) * lit * ax
        facade = np.array([0.010, 0.012, 0.018])
        col = facade + wcol * (br * 0.55)[..., None]
        # haze: farther = lighter, glow rising from the streets
        Ta = math.exp(-z / 1800.0)
        lowf = np.clip((STREET_Y + 60 - Y) / 60.0, 0, 1)[:, None, None]
        hz = haze * (1 - lowf) + (haze + lowglow * 0.04) * lowf
        col = col * Ta + hz * (1 - Ta)
        col = np.where(valid[None, :, None], col, 0)
        c = (cov_row * valid[None, :]).astype(np.float32)
        out = out * (1 - c[..., None]) + col.astype(np.float32) * c[..., None]
        cov_tot = cov_tot * (1 - c) + c
        dep = np.where(c > 0.5, zc, dep)
    small = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    cov = cv2.resize(cov_tot, (W, H), interpolation=cv2.INTER_AREA)
    img[:] = img * (1 - cov[..., None]) + small
    d2 = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
    depth[:] = np.minimum(depth, d2)


def _roof(img, depth, cam, fire_pos, fire_I, t):
    """Ray-cast roof deck (y = ROOF_Y, z < PARAPET_Z) and the parapet wall face."""
    dx, dy, dz, xx, yy = _rays(cam)
    oy = cam.pos[1]
    # parapet: wall at z = PARAPET_Z, from ROOF_Y to ROOF_Y + 1.05
    tw = (PARAPET_Z - cam.pos[2]) / dz
    wy = oy + dy * tw
    wx = cam.pos[0] + dx * tw
    on_wall = (wy <= ROOF_Y + 1.05) & (wy >= ROOF_Y - 0.2) & (dz > 0)
    # deck
    td = np.where(dy < -1e-6, (ROOF_Y - oy) / np.where(dy < -1e-6, dy, -1), 1e9)
    px = cam.pos[0] + dx * td
    pz = cam.pos[2] + dz * td
    on_deck = (dy < 0) & (pz < PARAPET_Z) & (pz > cam.pos[2] + 0.5)
    alb = 0.05
    # deck shading: city ambient + fire pool
    Lx = fire_pos[0] - px
    Ly = fire_pos[1] - ROOF_Y
    Lz = fire_pos[2] - pz
    d2 = Lx * Lx + Ly * Ly + Lz * Lz
    E = fire_I * (Ly / np.sqrt(d2)) / (d2 + 0.3)
    grain = 0.85 + 0.15 * np.sin(px * 7.3 + np.sin(pz * 5.1) * 2.0) * np.sin(pz * 6.1)
    amb = np.array([0.010, 0.012, 0.018])
    deck = alb * grain[..., None] * (amb * 1.5 + F.FIRE_LIGHT * E[..., None])
    # parapet face (faces the camera) + a pale coping line on its top edge
    Lzw = fire_pos[2] - PARAPET_Z
    d2w = (fire_pos[0] - wx) ** 2 + (fire_pos[1] - wy) ** 2 + Lzw ** 2
    Ew = fire_I * np.clip(-Lzw / np.sqrt(d2w), 0, 1) / (d2w + 0.3)
    wall = 0.06 * (amb * 1.2 + F.FIRE_LIGHT * Ew[..., None])
    coping = np.clip(1 - np.abs(wy - (ROOF_Y + 1.0)) / 0.05, 0, 1)
    wall = wall + (np.array([0.02, 0.022, 0.03]) + F.FIRE_LIGHT * 0.1 * Ew[..., None]) * coping[..., None]
    wall_cov = np.clip(((ROOF_Y + 1.05) - wy) * cam.f / np.maximum(tw, 1e-3) + 0.5, 0, 1) * on_wall
    img[:] = np.where(on_deck[..., None], deck, img)
    depth[:] = np.where(on_deck, td * np.sqrt(dx * dx + dy * dy + dz * dz), depth)
    img[:] = img * (1 - wall_cov[..., None]) + wall.astype(np.float32) * wall_cov[..., None]
    depth[:] = np.where(wall_cov > 0.5, tw, depth)


def _props():
    """Water tank on stilts (left) and an antenna mast (right) as SDF drawings."""
    tank = Drawing()
    tank.new_group()
    for lx in (-1.15, -0.4, 0.4, 1.15):
        tank.capsule((lx, 0.0), (lx * 0.92, 3.3), 0.06, 0.05, mat=11)
    tank.capsule((-1.15, 1.2), (1.15, 2.6), 0.025, 0.025, mat=11)
    tank.capsule((1.15, 1.2), (-1.15, 2.6), 0.025, 0.025, mat=11)
    tank.trap((0, 3.3), (0, 6.3), 1.35, 1.3, rnd=0.03, mat=4)
    for yb in (3.8, 4.6, 5.4, 6.1):
        tank.capsule((-1.38, yb), (1.38, yb), 0.03, 0.03, mat=5)
    tank.tri((-1.5, 6.25), (1.5, 6.25), (0.0, 7.35), rnd=0.02, mat=4)
    mast = Drawing()
    mast.new_group()
    mast.capsule((0, 0), (0, 6.5), 0.05, 0.035, mat=5)
    mast.capsule((-0.7, 5.4), (0.7, 5.4), 0.02, 0.02, mat=5)
    mast.capsule((-0.45, 6.1), (0.45, 6.1), 0.018, 0.018, mat=5)
    mast.capsule((0, 6.5), (0, 7.4), 0.012, 0.008, mat=5)
    box = Drawing()
    box.new_group()
    box.trap((0, 0), (0, 0.9), 0.75, 0.75, rnd=0.02, mat=11)
    return tank, mast, box


ANSWERS = [(1694, 4, -0.18), (1697, 3, 0.22), (1700, 2, -0.35), (1702, 4, 0.40), (1705, 1, 0.05),
           (1708, 3, -0.05), (1710, 2, 0.55), (1713, 1, -0.45), (1716, 0, 0.30)]


def _answer_sites(cam0):
    """Pick a roof for each answering fire: building in layer k nearest to screen fraction u."""
    B = buildings()
    sites = []
    for fr, k, u in ANSWERS:
        z = LAYERS[k]
        xs_target = cam0.cx + u * cam0.W * 0.5
        xw = (xs_target - cam0.cx) * z / cam0.f
        best = None
        for bx, bw, btop, kind, litp, tone, seed in B[k]:
            s = abs(bx - xw)
            yscr = cam0.cy + cam0.shift - cam0.f * btop / z
            if yscr > 470:          # hidden behind the parapet / too low
                continue
            if best is None or s < best[0]:
                best = (s, bx, btop)
        if best:
            sites.append((fr, np.array([best[1], best[2], z])))
    return sites


_SITES = None


def render(frame, scale=0.5):
    global _SITES
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    t = CM.ftime(frame)
    cam = camera(frame, W, H)
    if _SITES is None:
        _SITES = _answer_sites(camera(START, 1920, 804))
    dx, dy, dz, xx, yy = _rays(cam)
    img = _sky(cam, dx, dy, dz, t)
    depth = np.full((H, W), 1e9, np.float32)
    ey = dy / np.sqrt(dx * dx + dy * dy + dz * dz)
    SK.splat_stars(img, cam, _stars(), np.clip(ey / 0.15, 0, 1).astype(np.float32), t=t, gain=0.5, scale=scale)
    _skyline(img, depth, cam, t)
    ti = CM.ftime(IGN)
    size, inten, light = F.ignite_env(t, ti)
    fl = F.flicker(t, 31)
    dback, dfront, fb, rb = PR.drum()
    fire_base = DRUM + np.array([0.0, fb, 0.0])
    fire_I = 5.0 * light * fl
    arm = keys(frame, [(START, 0.1), (1684, 0.1), (1689.5, 1.0), (1695, 0.6), (1705, 0.2), (END, 0.15)], ease=smoother)
    lean = keys(frame, [(START, 0.0), (1684, 0.02), (1689.5, 0.2), (1695, -0.04), (1705, 0.0), (END, 0.0)],
                ease=smoother)
    fig, pts = PP.torch_person('hoodie', arm=arm, lean=lean)
    th = pts['torch_head']
    torch_w = PR.local_to_world(cam, FEET, -th[0], th[1], 0.0)
    torch_I = 0.45 * F.flicker(t, 5)
    fpos = fire_base + np.array([0, 0.6, 0])
    _roof(img, depth, cam, fpos if fire_I > 0 else torch_w, fire_I if fire_I > 0 else torch_I, t)
    # answering rooftop fires across the city
    for fr, P in _SITES:
        if frame >= fr - 1:
            s_, i_, l_ = F.ignite_env(t, CM.ftime(fr))
            sx, sy, z = cam.project(P + np.array([0, 0.5, 0]))
            e = (1.0 + 0.8 * math.exp(-(frame - fr) / 3.0)) * F.flicker(t, fr) * min(s_ / 0.5, 1.0)
            k = 1.0 + 0.6 * (LAYERS[-1] / P[2]) ** 0.5
            F.glow(img, depth, sx, sy, (0.9 + 0.6 * k) * scale + 0.3, 30.0 * e * k * scale * scale, z=z - 2, zbias=0)
            F.halo(img, depth, sx, sy, 22 * scale * k, 0.02 * e, z=z - 2, zbias=0)
    moon_col = CM.lin('#9DB4D9')
    lights = [dict(dir=(-0.3, 0.6, 0.5), col=moon_col, I=0.05), dict(pos=torch_w, col=F.FIRE_LIGHT, I=torch_I, r0=0.1)]
    if fire_I > 0:
        lights.append(dict(pos=fpos, col=F.FIRE_LIGHT, I=fire_I * 0.9, r0=0.3))
    famb = np.array([0.012, 0.014, 0.022])
    tank, mast, box = _props()
    FG.render(img, depth, cam, tank, np.array([-5.6, ROOF_Y, 12.5]), lights, amb=famb, t=t)
    FG.render(img, depth, cam, mast, np.array([6.4, ROOF_Y, 15.5]), lights, amb=famb, t=t)
    FG.render(img, depth, cam, box, np.array([-2.2, ROOF_Y, 14.0]), lights, amb=famb, t=t)
    FG.render(img, depth, cam, dback, DRUM, lights, amb=famb, t=t, emissive_gain=min(light, 1.5))
    FG.render(img, depth, cam, fig, FEET, lights, amb=famb, t=t, seed=21, flipx=True)
    sx, sy, z = cam.project(torch_w)
    F.halo(img, depth, sx, sy, 50 * scale, 0.08 * torch_I, z=z, zbias=0.5)
    F.flame(img, depth, cam, torch_w, 0.3, 0.05, t, seed=29, I=16.0, tongues=3, lean=0.03, zbias=0.1)
    if size > 0:
        _smoke().render(img, depth, cam, t, fire_base + np.array([0, 0.9, 0]), 0.5 * light * fl, albedo=0.2,
                        amb=(0.006, 0.007, 0.011))
        F.shimmer(img, cam, fire_base, 1.6 * size, rb, t, amp_px=1.2 * scale)
        F.flame(img, depth, cam, fire_base, 1.6 * size, rb, t, seed=12, I=24.0 * inten, lean=0.15 * size, zbias=0.3,
                tongues=4)
        sx, sy, z = cam.project(fire_base + np.array([0, 0.7, 0]))
        F.halo(img, depth, sx, sy, 200 * scale * size, 0.012 * light * fl, z=z, zbias=3.0)
    FG.render(img, depth, cam, dfront, DRUM, lights, amb=famb, t=t, write_depth=False)
    if size > 0:
        _sparks().render(img, depth, cam, t)
    return img


_ST = None
_SM = None
_SP = None


def _stars():
    global _ST
    if _ST is None:
        _ST = SK.make_stars(3000, 505, lum_scale=3.0)
    return _ST


def _smoke():
    global _SM
    if _SM is None:
        _SM = F.Smoke(81, DRUM + np.array([0, 2.2, 0]), CM.ftime(IGN) + 0.15, CM.ftime(END) + 0.1, rate=4.0,
                      wind=(0.7, 0, 0.3), rise=1.4, life=4.0, r0=0.25, growth=0.55, dens=0.35)
    return _SM


def _sparks():
    global _SP
    if _SP is None:
        _SP = F.Sparks(83, DRUM + np.array([0, 1.0, 0]), CM.ftime(IGN), CM.ftime(END) + 0.1, burst=260, rate=50,
                       ember_rate=10, wind=(0.9, 0.0, 0.3), radius=0.25, I=30.0)
    return _SP


FINISH = dict(exposure=1.0, bloom_strength=0.07, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)
