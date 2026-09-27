"""C13 THE RING FALLS, part b (R13b; RUN-C): cut-C frames 2840-2959 (bar 36 b3 -> the end of bar 37), in the ink world.

Out of EMBERS-C2's E13a (the band sinking through a moonlit deck, a small warm spark falling down-right) the cut
opens on her summit at night, drawn in ink under a cool night wash, the moon out of frame upper left. The Ring
streaks down-right over the range like a falling star (a drawn gold head with a tapering, broken tail between two
fine lines, as the old books draw a comet) and on bar 37 b2 (C 2900) strikes the snow beside the old cairn: a brief
gold bloom on the snow and the cairn's stones, a few short sparks, then a plume of steam rising and leaning downwind.
Her beacon is dark (it is lit in C15). MONTAGE-3D's C14 opens on the band in a melted hollow by her knee.

  python3 shots/run/ringfall_ink.py --range 2840-2959 --procs 2 --threads 2 --skip     # -> renders/runC_ringfall/
  python3 shots/run/ringfall_ink.py --frames 2860,2900,2930 --scale 0.5 --out t1       # a test
Same pipeline as ink_final.py (the Run world's AOVs at 2x -> the ink pass -> 1920x804); nothing here changes the
other C shots' files.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

C0 = 2840                   # cut-C frame of the shot's first frame (bar 36 b3)
N = 120                     # 2840-2959
STRIKE = 2900               # bar 37 b2
FPS = 24.0

# the camera: a wide view of her summit, level with the shelf, the moon 40 deg to the left of the lens and 21 deg up
# (out of frame upper left, as E13a lights it); the cairn right of centre and below it; a slow push over the shot
CAM = dict(yaw=-19.0, dist=(118.0, 104.0), up=(9.0, 7.0), aim_dx=8.5, aim_dy=5.5, hfov=48.0)
FALL = dict(sx=0.13, sy=0.04, z=200.0, ease=0.8, tail=10.0)   # the start: page position (0-1) and depth (m) at C0


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def strike_point():
    import keeper as KP
    c = np.asarray(KP.CAIRN, np.float64)
    b = np.asarray(KP.BEACON, np.float64)
    d = (b - c) / max(np.linalg.norm(b - c), 1e-6)
    return c + d * 1.3                          # beside the cairn, on the beacon's side (by where she kneels)


def camera(frame, W=1920, H=804):
    import rcam as RC
    u = _smooth((frame - C0) / (N - 1))
    a = math.radians(CAM['yaw'])
    fwd = np.array([math.sin(a), 0.0, math.cos(a)])
    S = strike_point()
    dist = CAM['dist'][0] + (CAM['dist'][1] - CAM['dist'][0]) * u
    up = CAM['up'][0] + (CAM['up'][1] - CAM['up'][0]) * u
    pos = S - fwd * dist + np.array([0.0, up, 0.0])
    # aim so the strike point sits right of centre and below it
    yaw = CAM['yaw'] - CAM['aim_dx']
    to = S - pos
    pitch = math.degrees(math.atan2(to[1], math.hypot(to[0], to[2]))) + CAM['aim_dy']
    return RC.RCam(pos, yaw, pitch, 0.0, CAM['hfov'], W, H)


class RingFallShot:
    """The shot as render_ink.render_aov and inkpass.compose expect it (cam, CR, B, light, lights_steady)."""

    def __init__(self):
        import keeper as KP
        self.name = 'ringfall'
        self.CR = np.ascontiguousarray(KP.CR_B)
        self.cam = camera
        self.B = np.zeros((0, 7))                # her beacon is dark: no drawn fire

    def light(self, frame):
        import world as WD
        Lk, amb, S, fogp, Q = WD.night_light()
        return Lk, Q, fogp

    def lights_steady(self, frame):
        return np.zeros((0, 8), np.float64)


# ------------------------------------------------------------------ the fall, in the world

def fall_pos(fr):
    """The Ring's world position at (fractional) frame fr: from high over the range, left of the lens, to the snow
    by the cairn at STRIKE, accelerating (a falling star, not a float)."""
    S = strike_point()
    c = camera(C0)                                            # the start sits on the first frame's page (in frame)
    ray = c.ray(np.array(FALL['sx'] * c.W), np.array(FALL['sy'] * c.H))
    ray = np.asarray(ray, np.float64).reshape(3)
    ray = ray / np.linalg.norm(ray)
    P0 = c.pos + ray * (FALL['z'] / max(float(ray @ c.fwd), 1e-3))
    u = min(max((fr - C0) / (STRIKE - C0), 0.0), 1.0)
    w = u ** FALL['ease']
    return P0 + (S - P0) * w


def project(c, X):
    d = np.asarray(X, np.float64) - c['pos']
    z = float(d @ c['fwd'])
    if z <= 0.5:
        return None
    return (c['cx'] + c['f'] * float(d @ c['right']) / z, c['cy'] - c['f'] * float(d @ c['up']) / z, z)


# ------------------------------------------------------------------ the drawing (at the ink pass's resolution)

def _seg_d(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    L2 = max(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0, 1)
    return np.hypot(px - x0 - t * dx, py - y0 - t * dy), t


def draw_cairn(img_lin, c, A, kpx, IP):
    """The old cairn: a low heap of field stones, drawn (outlined, hatched on the side away from the moon), seated
    on the snow at keeper.CAIRN. Returns the stones' mask (for the flash's warm light)."""
    import keeper as KP
    H, W = img_lin.shape[:2]
    base = np.asarray(KP.CAIRN, np.float64)
    rng = np.random.default_rng(37)
    stones = []
    for layer, (n, r, y) in enumerate(((5, 0.34, 0.16), (4, 0.30, 0.46), (3, 0.26, 0.74), (1, 0.22, 0.98))):
        for k in range(n):
            ang = (k + 0.5 * layer) / max(n, 1) * 2 * math.pi + rng.uniform(-0.3, 0.3)
            rad = (0.62 - 0.14 * layer) * (0.8 + 0.3 * rng.uniform())
            p = base + np.array([rad * math.cos(ang), y, rad * math.sin(ang)])
            stones.append((p, r * (0.8 + 0.4 * rng.uniform()), rng.uniform(0.6, 1.0), rng.uniform(0, 6.283)))
    # far to near
    stones.sort(key=lambda s: -float(np.linalg.norm(s[0] - c['pos'])))
    mask = np.zeros((H, W), np.float32)
    ink = np.zeros((H, W), np.float32)
    shade = np.zeros((H, W), np.float32)
    for p, r, sq, rot in stones:
        q = project(c, p)
        if q is None:
            continue
        sx, sy, z = q
        R = r * c['f'] / z
        if R < 0.8:
            continue
        x0, x1 = int(max(sx - 2 * R - 3, 0)), int(min(sx + 2 * R + 4, W))
        y0, y1 = int(max(sy - 2 * R - 3, 0)), int(min(sy + 2 * R + 4, H))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        dx, dy = (xx + 0.5 - sx) / R, (yy + 0.5 - sy) / (R * sq)
        ang = np.arctan2(dy, dx)
        rr = np.hypot(dx, dy) / (1.0 + 0.10 * np.sin(3 * ang + rot) + 0.06 * np.sin(5 * ang + 2 * rot))
        sd = (rr - 1.0) * R                                   # page px, <0 inside
        inside = np.clip(0.5 - sd / kpx, 0, 1)
        # this stone covers what is behind it
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], inside)
        ink[y0:y1, x0:x1] = ink[y0:y1, x0:x1] * (1 - inside)
        shade[y0:y1, x0:x1] = shade[y0:y1, x0:x1] * (1 - inside)
        # the side away from the moon (lower right) hatched
        away = np.clip((dx * 0.8 + dy * 0.6) * 1.2, 0, 1)
        hat = (np.abs(((xx + yy) / (2.6 * kpx)) % 1.0 - 0.5) < 0.18).astype(np.float64)
        shade[y0:y1, x0:x1] = np.maximum(shade[y0:y1, x0:x1], inside * away * 0.55)
        ink[y0:y1, x0:x1] = np.maximum(ink[y0:y1, x0:x1], inside * away * hat * 0.7)
        ink[y0:y1, x0:x1] = np.maximum(ink[y0:y1, x0:x1], np.clip(0.5 * 1.0 * kpx + 0.5 - np.abs(sd), 0, 1) * 0.9)
    # land in front of the cairn hides it
    d_c = float(np.linalg.norm(base - c['pos']))
    vis = (A[..., 1] > d_c - 3.0).astype(np.float32)
    snow = IP.s2l(np.array([214, 214, 222]))
    rock = IP.s2l(IP.WASH)
    col = rock * 0.8 + snow * 0.2
    m = (mask * vis)[..., None]
    img_lin = img_lin * (1 - m) + (col[None, None] * (1 - 0.35 * shade[..., None])) * m
    ia = (ink * vis)[..., None]
    img_lin = img_lin * (1 - 0.9 * ia) + IP.s2l(IP.INK)[None, None] * (0.9 * ia)
    return img_lin, mask * vis


def night_wash(img_lin, A, kpx, H, W):
    """The drawing under a cool night wash (multiplied, so the paper's grain and the line work stay): deep
    blue-grey, paler where the moon lights the snow; the sky a deeper ultramarine toward the top."""
    import cv2
    surf = A[..., 0]
    sky = (surf >= 2.5).astype(np.float32)
    lit = np.clip(A[..., 10], 0, 1) * np.clip(A[..., 11], 0, 1)
    snow = np.clip(A[..., 9], 0, 1)
    lit = cv2.GaussianBlur((lit * (0.35 + 0.65 * snow)).astype(np.float32), (0, 0), 2.0 * kpx)
    night = np.array([0.30, 0.34, 0.48], np.float32)
    moon = np.array([0.66, 0.72, 0.86], np.float32)
    m = night[None, None] + (moon - night)[None, None] * np.clip(lit * 1.3, 0, 1)[..., None]
    yy = (np.arange(H, dtype=np.float32) / H)[:, None, None]
    msky = night[None, None] * (0.72 + 0.34 * yy)
    m = m * (1 - sky[..., None]) + msky * sky[..., None]
    return img_lin * m


def draw_stars(img_lin, c, A, kpx, H, W):
    """A few fixed stars (white gouache pricks), fixed to the sky (directions), only in the sky."""
    rng = np.random.default_rng(1301)
    n = 900
    v = rng.standard_normal((n, 3))
    v[:, 1] = np.abs(v[:, 1]) + 0.05
    v /= np.linalg.norm(v, axis=1)[:, None]
    br = rng.uniform(0.15, 1.0, n) ** 2.2
    sky = A[..., 0] >= 2.5
    out = img_lin.copy()
    fwd, right, up = c['fwd'], c['right'], c['up']
    for k in range(n):
        z = float(v[k] @ fwd)
        if z <= 0.05:
            continue
        sx = c['cx'] + c['f'] * float(v[k] @ right) / z
        sy = c['cy'] - c['f'] * float(v[k] @ up) / z
        if not (2 <= sx < W - 2 and 2 <= sy < H - 2):
            continue
        ix, iy = int(sx), int(sy)
        if not sky[iy, ix]:
            continue
        r = (0.55 + 0.9 * br[k]) * kpx
        a0 = 0.30 + 0.55 * br[k]
        x0, x1, y0, y1 = ix - 3, ix + 4, iy - 3, iy + 4
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        cov = np.clip(r + 0.5 - np.hypot(xx + 0.5 - sx, yy + 0.5 - sy), 0, 1) * a0
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - cov[..., None]) + np.array([0.82, 0.84, 0.88])[None, None] * cov[..., None]
    return out


def draw_fall(img_lin, c, frame, kpx, A, H, W, IP):
    """The Ring: a gold head and a tapering tail between two fine lines, broken toward its end; drawn from the
    world path over the last FALL['tail'] frames (the shutter's smear, as a pen would draw it)."""
    if frame >= STRIKE:
        return img_lin
    k = 28
    frs = np.linspace(frame - FALL['tail'], frame, k)
    pts = [project(c, fall_pos(f)) for f in frs]
    if any(p is None for p in pts):
        return img_lin
    P = np.array([[p[0], p[1]] for p in pts])
    x0, x1 = int(max(P[:, 0].min() - 14 * kpx, 0)), int(min(P[:, 0].max() + 14 * kpx, W))
    y0, y1 = int(max(P[:, 1].min() - 14 * kpx, 0)), int(min(P[:, 1].max() + 14 * kpx, H))
    if x1 <= x0 or y1 <= y0:
        return img_lin
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    px, py = xx + 0.5, yy + 0.5
    best = np.full(px.shape, 1e9)
    tpar = np.zeros(px.shape)
    side = np.zeros(px.shape)
    for i in range(k - 1):
        d, t = _seg_d(px, py, P[i, 0], P[i, 1], P[i + 1, 0], P[i + 1, 1])
        m = d < best
        best = np.where(m, d, best)
        tpar = np.where(m, (i + t) / (k - 1), tpar)
        nx, ny = -(P[i + 1, 1] - P[i, 1]), P[i + 1, 0] - P[i, 0]
        nn = max(math.hypot(nx, ny), 1e-9)
        side = np.where(m, ((px - P[i, 0]) * nx + (py - P[i, 1]) * ny) / nn, side)
    u = tpar                                                  # 0 at the tail's end, 1 at the head
    wmax = 3.2 * kpx
    wid = 0.35 * kpx + (wmax - 0.35 * kpx) * u ** 1.6          # half-width of the gold wash
    brk = (np.sin(u * 23.0 - frame * 0.9) > -0.35) | (u > 0.55)  # the tail breaks into dashes toward its end
    wash = np.clip(wid + 0.5 - best, 0, 1) * (0.25 + 0.75 * u ** 1.2) * brk
    # the two fine lines along the edges, lighter toward the tail
    edge = np.clip(0.45 * kpx + 0.5 - np.abs(np.abs(side) - wid), 0, 1) * u ** 0.8 * brk * (u > 0.12)
    # the head: a small pale-gold core, gilded rim
    hx, hy = P[-1]
    rh = np.hypot(px - hx, py - hy)
    core = np.clip(1.6 * kpx + 0.5 - rh, 0, 1)
    glow = np.exp(-(rh / (5.5 * kpx)) ** 2) * 0.35
    g_hi, g_mid, g_lo = IP.s2l(IP.GOLD_HI), IP.s2l(IP.GOLD), IP.s2l(IP.GOLD_LO)
    sub = img_lin[y0:y1, x0:x1]
    a = np.clip(wash * 0.85 + glow, 0, 1)[..., None]
    col = g_mid * (1 - u[..., None]) + g_hi * u[..., None]
    sub = sub * (1 - a) + col * a
    sub = sub * (1 - 0.8 * edge[..., None]) + g_lo[None, None] * (0.8 * edge[..., None])
    sub = sub * (1 - core[..., None]) + np.array([1.0, 0.95, 0.80])[None, None] * core[..., None]
    img_lin[y0:y1, x0:x1] = sub
    return img_lin


def draw_strike(img_lin, c, frame, kpx, A, cairn_mask, H, W, IP):
    """From STRIKE: a gold bloom laid on the snow and the stones (brief), a few short sparks (one or two frames),
    then the steam: pale strands winding up and leaning downwind, growing and thinning."""
    if frame < STRIKE:
        return img_lin
    import cv2
    t = (frame - STRIKE) / FPS
    S = strike_point()
    q = project(c, S)
    if q is None:
        return img_lin
    sx, sy, z = q
    mpp = z / c['f']                                          # metres per page px at the strike
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # the bloom: gold wash on the land (and the cairn), a world-scale disc growing to ~5 m, gone in ~0.5 s
    fl = math.exp(-t / 0.18) * min(1.0, 0.4 + (frame - STRIKE + 1) * 0.6)
    if fl > 0.01:
        Rb = (1.5 + 3.5 * (1 - math.exp(-t / 0.10))) / mpp
        rr = np.hypot(xx + 0.5 - sx, (yy + 0.5 - sy) * 1.8)
        land = (A[..., 0] < 1.5).astype(np.float32)
        b = np.clip(1.0 - rr / Rb, 0, 1) ** 0.8 * fl * np.maximum(land, cairn_mask)
        b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1.2 * kpx)
        g = IP.s2l(IP.GOLD_HI) * 0.7 + IP.s2l(IP.GOLD) * 0.3
        img_lin = img_lin * (1 - 0.85 * b[..., None]) + g[None, None] * (0.85 * b[..., None])
    # sparks: 7 short strokes flung up and out, the first two frames only
    if frame - STRIKE <= 1:
        rng = np.random.default_rng(29)
        sp = np.zeros((H, W), np.float32)
        for k in range(7):
            ang = math.radians(rng.uniform(25, 155))
            L0 = (0.6 + 0.4 * (frame - STRIKE)) / mpp
            L1 = L0 + rng.uniform(0.8, 1.6) / mpp
            xa, ya = sx + math.cos(ang) * L0, sy - math.sin(ang) * L0
            xb, yb = sx + math.cos(ang) * L1, sy - math.sin(ang) * L1
            x0, x1 = int(max(min(xa, xb) - 4, 0)), int(min(max(xa, xb) + 5, W))
            y0, y1 = int(max(min(ya, yb) - 4, 0)), int(min(max(ya, yb) + 5, H))
            if x1 <= x0 or y1 <= y0:
                continue
            d, tt = _seg_d(xx[y0:y1, x0:x1] + 0.5, yy[y0:y1, x0:x1] + 0.5, xa, ya, xb, yb)
            sp[y0:y1, x0:x1] = np.maximum(sp[y0:y1, x0:x1], np.clip(0.6 * kpx * (1 - tt) + 0.5 - d, 0, 1))
        img_lin = img_lin * (1 - sp[..., None]) + (IP.s2l(IP.GOLD_HI))[None, None] * sp[..., None]
    # the steam: two pale strands and a thin pale wash, rising from the hollow and leaning downwind
    if t > 0.04:
        a = math.radians(CAM['yaw'])
        right = np.array([math.cos(a), 0.0, -math.sin(a)])
        hmax = 16.0 * (1 - math.exp(-t / 0.9))                   # metres
        s = np.linspace(0.0, 1.0, 40)
        Ppl = np.array([S + np.array([0.0, 0.3 + hmax * v, 0.0]) + right * (3.0 * v * v * min(t / 1.5, 1.0))
                        for v in s])
        pr = [project(c, p) for p in Ppl]
        if all(p is not None for p in pr):
            X = np.array([p[0] for p in pr])
            Y = np.array([p[1] for p in pr])
            rad = (0.35 + 1.4 * s) / mpp
            fade = np.clip((5.0 - t) / 2.0, 0, 1) * np.clip(t / 0.25, 0, 1)
            x0, x1 = int(max(X.min() - rad.max() - 6, 0)), int(min(X.max() + rad.max() + 6, W))
            y0, y1 = int(max(Y.min() - rad.max() - 6, 0)), int(min(Y.max() + 6, H))
            if x1 > x0 and y1 > y0:
                qx = (xx[y0:y1, x0:x1] + 0.5).ravel()
                qy = (yy[y0:y1, x0:x1] + 0.5).ravel()
                import inkpass as IPm
                dc, uc = IPm._seg_dist(qx, qy, X, Y)
                rw = np.interp(uc, s, rad)
                wash = (np.clip(1.0 - dc / np.maximum(rw, 0.8 * kpx), 0, 1) ** 1.5 * (1 - uc) ** 0.8 * 0.45 * fade)
                wash = wash.reshape((y1 - y0, x1 - x0))
                lines = np.zeros_like(wash)
                nx_, ny_ = np.gradient(Y), -np.gradient(X)
                nn = np.maximum(np.hypot(nx_, ny_), 1e-6)
                nx_, ny_ = nx_ / nn, ny_ / nn
                for sd_ in (-1.0, 1.0):
                    wob = sd_ * 0.5 + 0.45 * np.sin(6.283 * (1.7 * s - 0.35 * t) + (0 if sd_ > 0 else 2.1))
                    ux, uy = X + nx_ * rad * wob, Y + ny_ * rad * wob
                    dl, ul = IPm._seg_dist(qx, qy, ux, uy)
                    ln = np.clip(0.4 * kpx + 0.5 - dl, 0, 1) * np.clip(1.0 - ul, 0, 1) ** 0.9 * fade * 0.6
                    lines = np.maximum(lines, ln.reshape(wash.shape))
                sub = img_lin[y0:y1, x0:x1]
                pale = np.array([0.78, 0.80, 0.86])
                sub = sub * (1 - wash[..., None]) + pale[None, None] * wash[..., None]
                sub = sub * (1 - lines[..., None]) + (IP.s2l(IP.INK_LIGHT) * 0.9)[None, None] * lines[..., None]
                img_lin[y0:y1, x0:x1] = sub
    return img_lin


def render_frame(frame, scale=1.0, ss=2.0, threads=1):
    import cv2
    import render_ink as RI
    import inkpass as IP
    look = IP.look
    shot = RingFallShot()
    R = RI.render_aov(shot, frame, scale, ss)
    kpx = scale * ss
    R['kpx'] = kpx
    img, _ = IP.compose(R, B=shot.B, plate=None, CR=shot.CR)
    A = R['A']
    H, W = A.shape[:2]
    c = R['cam']
    lin = look.srgb_to_linear(np.clip(img, 0, 1)).astype(np.float64)
    lin, cm = draw_cairn(lin, c, A, kpx, IP)
    lin = night_wash(lin, A, kpx, H, W)
    lin = draw_stars(lin, c, A, kpx, H, W)
    lin = draw_fall(lin, c, frame, kpx, A, H, W, IP)
    lin = draw_strike(lin, c, frame, kpx, A, cm, H, W, IP)
    out = look.linear_to_srgb(np.clip(lin, 0, 1)).astype(np.float32)
    Wt, Ht = int(round(1920 * scale)), int(round(804 * scale))
    if out.shape[:2] != (Ht, Wt):
        out = cv2.resize(out, (Wt, Ht), interpolation=cv2.INTER_AREA)
    return out


def work(args):
    frames, scale, ss, out, threads = args
    import numba
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    import cv2
    cv2.setNumThreads(1)
    import inkpass as IP
    look = IP.look
    for f in frames:
        t0 = time.time()
        img = render_frame(f, scale, ss, threads)
        look.save_png(look.frame_path(out, f), img)
        print(f'ringfall {f} {time.time() - t0:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=2.0)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default=None, help='sub-folder of renders/runC_ringfall/ (tests)')
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    os.environ['NUMBA_NUM_THREADS'] = str(max(1, a.threads))
    import pipe as PI
    look = PI.look
    base = os.path.join(PI.CM.ROOT, 'renders', 'runC_ringfall')
    out = base if a.out is None else os.path.join(base, a.out)
    os.makedirs(out, exist_ok=True)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    else:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1))
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.frame_path(out, f))]
    t0 = time.time()
    if a.procs <= 1:
        work((frames, a.scale, a.ss, out, a.threads))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(work, [(c, a.scale, a.ss, out, a.threads) for c in chunks])
    print(f'total {time.time() - t0:.1f}s for {len(frames)} frames')


if __name__ == '__main__':
    main()
