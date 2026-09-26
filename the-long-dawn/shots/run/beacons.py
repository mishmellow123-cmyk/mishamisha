"""The beacon chain of THE BEACON RUN: placement (from the summit catalogue, matched to the SFX beats and
pans), ignition envelopes, and drawing (lights for the terrain shader, glows / flames / smoke / cairns in
source space, sparks as camera-motion streaks in target space)."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world as WD          # noqa: E402
import run as RN            # noqa: E402

from mt import fire as F, props as PR, figure as FG   # noqa: E402
import fire2 as F2          # noqa: E402

FPS = 24.0
FIRE_COL = np.array([1.0, 0.60, 0.24])

# (ignition frame, target screen x (0..1), distance range m, kind)  kind: 0 cairn, 1 great pyre
# (ignition frame, target screen x at that frame (SFX pan), distance band m, kind: 1 great pyre, 0 cairn)
BEATS = [(1540, ('screen', 0.38, 0.71), (700.0, 1500.0), 1), (1560, 0.70, (300.0, 3000.0), 1),
         (1600, ('xz', 1408.0, 7536.0), (0.0, 1e9), 1),
         (1620, 0.40, (2500.0, 7000.0), 1), (1640, 0.60, (4500.0, 11000.0), 1), (1660, 0.47, (8000.0, 18000.0), 1)]
FILLERS = [(1550, 0.86, (2500.0, 9000.0), 0), (1570, 0.14, (2500.0, 9000.0), 0), (1590, 0.82, (3000.0, 12000.0), 0),
           (1610, 0.14, (4000.0, 14000.0), 0), (1630, 0.84, (5000.0, 16000.0), 0), (1650, 0.18, (7000.0, 20000.0), 0),
           (1670, 0.78, (10000.0, 26000.0), 0)]


def _visible(cam, P, CR, t):
    """True if world point P is not hidden by terrain/cloud from cam (sampled segment test)."""
    o = cam.pos
    d = P - o
    L = float(np.linalg.norm(d))
    Pp = np.array([o[0], o[2], t, 0.0])
    for u in np.linspace(0.02, 0.985, 90) ** 1.5:
        q = o + d * u
        fp = max(u * L / 1500.0, 0.2)
        if WD.hfun(q[0], q[2], fp, Pp, CR) > q[1]:
            return False
    return True


def _pick(cat, used, f, sx_t, drange, W=1920, H=804, sy_max=0.80):
    cam = RN.camera(f, W, H)
    best = None
    for k in range(len(cat)):
        if k in used:
            continue
        x, y, z, prom = cat[k]
        P = np.array([x, y + 1.0, z])
        dist = float(np.linalg.norm(P - cam.pos))
        if dist < drange[0] or dist > drange[1]:
            continue
        sx, sy, zc = cam.project(P)
        if zc <= 0 or sx < 0.04 * W or sx > 0.96 * W or sy < 0.06 * H or sy > sy_max * H:
            continue
        score = -abs(sx / W - sx_t) * 6.0 + min(prom, 400.0) / 400.0 - 0.3 * abs(math.log(dist / math.sqrt(
            drange[0] * drange[1])))
        if best is None or score > best[0]:
            best = (score, k, P, dist)
    return best


def _raycast(cam, sx, sy, CR, t, dmax=60000.0):
    d = cam.ray(np.array([sx]), np.array([sy]))[0]
    d /= np.linalg.norm(d)
    Pp = np.array([cam.pos[0], cam.pos[2], t, 0.0])
    tt = 2.0
    prev = tt
    while tt < dmax:
        q = cam.pos + d * tt
        if WD.hfun(q[0], q[2], max(tt / 1500.0, 0.2), Pp, CR) > q[1]:
            lo, hi = prev, tt
            for _ in range(20):
                m = 0.5 * (lo + hi)
                qm = cam.pos + d * m
                if WD.hfun(qm[0], qm[2], max(m / 1500.0, 0.2), Pp, CR) > qm[1]:
                    hi = m
                else:
                    lo = m
            return cam.pos + d * hi, hi
        prev = tt
        tt += max(1.0, 0.01 * tt)
    return None, None


def _prominence(P, CR, r=None):
    d = float(np.linalg.norm(P))
    r = r or 30.0
    ring = [WD.ground(P[0] + r * math.cos(a), P[2] + r * math.sin(a), CR) for a in np.linspace(0, 6.283, 9)[:-1]]
    return P[1] - float(np.mean(ring))


def _target(f, sxt, drange, t, used_xz):
    """Scan the frame around the pan target (or an explicit (sx, sy) target), raycast, climb to the summit,
    score."""
    cam = RN.camera(f)
    best = None
    syt = None
    if isinstance(sxt, tuple):
        sxt, syt = sxt
    sy_grid = np.linspace(0.30, 0.78, 13) if syt is None else np.linspace(syt - 0.06, syt + 0.1, 9)
    for sxf in np.linspace(sxt - 0.13, sxt + 0.13, 9):
        for syf in sy_grid:
            hit, dist = _raycast(cam, sxf * 1920, syf * 804, RN.CR, t)
            if hit is None or hit[1] < WD.CLOUD_Y + 60.0 or dist < 0.5 * drange[0] or dist > 1.6 * drange[1]:
                continue
            sm = WD.summit_near(hit[0], hit[2], RN.CR, rad=max(30.0, 0.015 * dist), n=13)
            if any(math.hypot(sm[0] - q[0], sm[2] - q[1]) < 150.0 + 0.02 * dist for q in used_xz):
                continue
            P = sm + np.array([0.0, 0.8, 0.0])
            sx, sy, z = cam.project(P)
            D = float(np.linalg.norm(P - cam.pos))
            if z <= 0 or sy < 0.08 * 804 or sy > 0.82 * 804 or sx < 0.04 * 1920 or sx > 0.96 * 1920:
                continue
            prom = _prominence(sm, RN.CR, r=max(30.0, 0.02 * D))
            lo, hi = drange
            dpen = 0.0 if lo <= D <= hi else abs(math.log(D / (lo if D < lo else hi)))
            score = -5.0 * abs(sx / 1920 - sxt) - 2.0 * dpen + 0.8 * min(prom / (0.04 * D + 20.0), 1.5)
            if syt is not None:
                score -= 5.0 * abs(sy / 804 - syt)
            if best is not None and score <= best[0]:
                continue
            if not _visible(cam, P + np.array([0, 2.5, 0]), RN.CR, t):
                continue
            best = (score, P, D)
    return best


def place(seed=5):
    """Deterministic chain: beats, fillers, and the cascade to the horizon. Cached to beacons.npy."""
    path = os.path.join(HERE, 'beacons.npy')
    if os.path.exists(path):
        return np.load(path)
    cat = np.load(os.path.join(HERE, 'summits.npy'))
    t = 63.0
    rows = []
    used = set()

    def add(P, f, kind, dist):
        rows.append([P[0], P[1], P[2], f, kind, len(rows), dist])

    hs = WD.S1.summit()
    add(np.array([hs[0], hs[1] - 3.0, hs[2]]), 1360.0, 1, float(np.linalg.norm(hs)))
    used_xz = [(RN.PYRE_XZ[0], RN.PYRE_XZ[2]), (hs[0], hs[2])]
    for f, sxt, dr, kind in BEATS + FILLERS:
        if isinstance(sxt, tuple) and sxt[0] == 'xz':
            k = int(np.argmin(np.hypot(cat[:, 0] - sxt[1], cat[:, 2] - sxt[2])))
            P = cat[k, :3] + np.array([0.0, 1.0, 0.0])
            b = (0.0, P, float(np.linalg.norm(P - RN.camera(f).pos)))
        elif isinstance(sxt, tuple) and sxt[0] == 'screen':
            cam = RN.camera(f)
            best = None
            for x, y, z, prom in cat:
                P = np.array([x, y + 1.0, z])
                a, bq, zc = cam.project(P)
                D = float(np.linalg.norm(P - cam.pos))
                if zc <= 0 or D < dr[0] or D > dr[1]:
                    continue
                sc = -abs(a / 1920 - sxt[1]) - abs(bq / 804 - sxt[2])
                if (best is None or sc > best[0]) and _visible(cam, P + np.array([0, 3.0, 0]), RN.CR, t):
                    best = (sc, P, D)
            b = best
        else:
            b = _target(f, sxt, dr, t, used_xz)
        if b is None:
            print('no beacon for', f)
            continue
        used_xz.append((b[1][0], b[1][2]))
        add(b[1], f, kind, b[2])
    # the cascade: from the last beats the signal races on, summit to summit, toward the horizon (chains,
    # not a scatter); each link a little sooner than the last
    cam = RN.camera(1679)
    fwd = np.array([math.sin(math.radians(cam.yaw_d)), math.cos(math.radians(cam.yaw_d))])
    rng = np.random.default_rng(seed)
    pts = cat[:, [0, 2]]

    def bearing(xz):
        v = xz - cam.pos[[0, 2]]
        return math.degrees(math.atan2(v[0], v[1]))

    heads = {int(r[3]): np.array(r[:3]) for r in rows}
    starts = [(1660, 0.0, 1660.0), (1640, 12.0, 1640.0), (1620, -14.0, 1620.0)]
    for f0, drift, fstart in starts:
        if f0 not in heads:
            continue
        cur = heads[f0]
        fcur = fstart
        dt = 4.0
        b0 = bearing(cur[[0, 2]]) + drift
        for link in range(14):
            dcur = float(np.linalg.norm(cur - cam.pos))
            best = None
            for k in range(len(cat)):
                x, y, z, prom = cat[k]
                if prom < 40.0:
                    continue
                step = math.hypot(x - cur[0], z - cur[2])
                D = math.hypot(x - cam.pos[0], z - cam.pos[2])
                if step < 2500.0 or step > 9000.0 or D < dcur + 1500.0 or D > 75000.0:
                    continue
                bb = bearing(np.array([x, z]))
                if abs(bb - b0) > 9.0 + 3.0 * rng.random():
                    continue
                P = np.array([x, y + 1.0, z])
                sx, sy, zc = cam.project(P)
                if zc <= 0 or sx < 0.03 * 1920 or sx > 0.97 * 1920 or sy > 0.75 * 804:
                    continue
                if any(math.hypot(x - r[0], z - r[2]) < 1200.0 for r in rows):
                    continue
                score = min(prom, 500.0) / 500.0 - abs(bb - b0) / 12.0 - abs(step - 5000.0) / 6000.0
                if best is None or score > best[0]:
                    best = (score, P, D)
            if best is None:
                break
            if not _visible(cam, best[1] + np.array([0, 3.0, 0]), RN.CR, t):
                # hidden from the final view: still lights (its glow shows over the ridge), keep the chain going
                pass
            fcur += dt
            dt = max(0.9, dt * 0.78)
            if fcur > 1679.5:
                break
            add(best[1], fcur, 0, best[2])
            cur = best[1]
            b0 = 0.7 * b0 + 0.3 * bearing(cur[[0, 2]])
    arr = np.array(rows)
    np.save(path, arr)
    return arr


# ------------------------------------------------------------------ firing ---
# The critic's rules for every fire of the Run: a beat beacon is a real bonfire at least ~25 px tall at 1920
# (sized from its distance and the lens at its beat, so the far great beacons are as big on their peaks as
# the near ones on their knolls), with a 2-3 frame flash as it catches, a smoke column lit from below and a
# warm pool on the snow; far fillers are small flames (never round dots) that vary ~3:1 in size and
# brightness and flicker each at its own rate; everything sits in the same haze as the terrain.

# code-side additions to the chain (beacons.npy is never regenerated): the 1550 filler on the right
# (SFX run_far_1, pan +0.8) that the catalogue search had missed
EXTRA = np.array([[4720.0, -107.0, 7460.0, 1550.0, 0.0, 25.0, 8749.0]])
BEAT_PX = 26.0                 # settled VISIBLE flame height (px at 1920) of a beat beacon at its beat
FILL_PX = 11.0                 # ... of a filler at 12 km (x its own 0.62-1.72 size factor; ~sqrt(1/d) beyond)
VIS = 0.56                     # visible flame / nominal Hf (tongue tips are sparse and dim)
FLASH_TAU = 1.5                # frames: the catch flash (u = 0, 1, 2 -> 1, .51, .26)
FIRE_BLUR = 0.5                # fires are blurred over half the true shutter path: the eye tracks them
                               # through the pans, and a bright small flame must keep its shape on a still

_CHAIN = None


def _seat(x, z, r, rad, n=15):
    """The summit the renderer actually draws near (x, z): the march steps 0.35% of the distance, so a
    needle thinner than ~a step is skipped and the drawn peak is blunter than the geometry. Maximise the
    terrain eroded by a disk of radius r (min over the centre and a ring of 10) around the true summit."""
    ang = np.linspace(0.0, 2.0 * math.pi, 11)[:-1]
    best = (-1e9, x, z)
    cx, cz = x, z
    for it in range(3):
        for xx in np.linspace(cx - rad, cx + rad, n):
            for zz in np.linspace(cz - rad, cz + rad, n):
                h = WD.ground(xx, zz, RN.CR)
                if h <= best[0]:
                    continue
                for a in ang:
                    h = min(h, WD.ground(xx + r * math.cos(a), zz + r * math.sin(a), RN.CR))
                    if h <= best[0]:
                        break
                if h > best[0]:
                    best = (h, xx, zz)
        cx, cz = best[1], best[2]
        rad *= 0.3
    return np.array([best[1], best[0], best[2]])


def chain():
    """beacons.npy + EXTRA, with every render position (except the heroine's, which s1 placed) snapped to the
    true local summit (the catalogue's 60 m grid left some fires floating beside their peaks or sunk in
    them), then seated on the summit as drawn at the distance it is seen from."""
    global _CHAIN
    if _CHAIN is None:
        B = np.vstack([place(), EXTRA])
        out = B.copy()
        for i, b in enumerate(B):
            if b[3] < 1500:
                continue
            r = 12.0 if b[6] < 3000 else 45.0
            sm = WD.summit_near(b[0], b[2], RN.CR, rad=r, n=13)
            cam = RN.camera(min(max(b[3] + 6.0, RN.START), RN.END))
            d = float(np.linalg.norm(sm - cam.pos))
            rs = max(0.5, 0.5 * 0.0035 * d)
            out[i, :3] = _seat(sm[0], sm[2], rs, max(3.0 * rs, 6.0)) if rs > 1.0 else sm
        _CHAIN = out
    return _CHAIN


def env(frame, f0):
    t = frame / FPS
    return F.ignite_env(t, f0 / FPS)


def flash(frame, f0):
    u = frame - f0
    if u < -0.5:
        return 0.0
    return math.exp(-max(u, 0.0) / FLASH_TAU)


_SPEC = {}


def spec(b):
    """Per-fire constants: flame height Hf (m), base half-width Rb, brightness, flicker rate/seed, tongues."""
    k = int(b[5])
    if k in _SPEC:
        return _SPEC[k]
    rng = np.random.default_rng(700 + k)
    sv = math.exp(rng.uniform(math.log(0.62), math.log(1.72)))       # size, ~2.8 : 1
    bv = math.exp(rng.uniform(math.log(0.58), math.log(1.62)))       # brightness, ~2.8 : 1
    rate = rng.uniform(0.72, 1.38)
    P = np.array(b[:3])
    if b[3] < 1500:
        # the heroine's beacon (lit at 1360), 32 km out on the horizon where s1 showed it
        Hf, beat, bv, sv = 55.0, False, 1.0, 1.0
    else:
        fr = min(max(b[3] + 6.0, RN.START), RN.END)
        cam = RN.camera(fr)
        d = float(np.linalg.norm(P - cam.pos))
        beat = b[4] == 1
        if beat:
            Hf = max(8.0, BEAT_PX / VIS * d / cam.f)
            sv, bv = 1.0, 1.0
        else:
            px = FILL_PX * min(1.0, math.sqrt(12000.0 / d)) * sv
            Hf = max(px, 3.2) / VIS * d / cam.f
    n = 6 if beat else 4
    s = dict(Hf=Hf, Rb=(0.24 if beat else 0.22) * Hf, beat=beat, bv=bv, sv=sv, rate=rate, seed=k * 7 + 3,
             TG=F2.tongues_for(n, k * 7 + 3, Hf, spread=0.8 if beat else 0.7, tall=(1, 4) if beat else (1,)))
    _SPEC[k] = s
    return s


def platforms(B, extra=()):
    """Cleared rock platforms under the near beacons (x, y, z, radius)."""
    out = [[b[0], b[1], b[2], max(2.8, 1.6 * spec(b)['Rb'])] for b in B if b[6] < 3000.0 and b[3] >= 1500]
    out += list(extra)
    return np.array(out, np.float64).reshape(-1, 4)


def _fire_state(frame, b):
    s = spec(b)
    t = frame / FPS
    size, inten, light = env(frame, b[3])
    if b[3] < 1500:
        size, inten, light = 1.0, 1.0, 1.0
    fl = F.flicker(t * s['rate'], int(b[5]) + 11, 1.0 if s['beat'] else 1.7)
    return s, size, inten, light, fl, flash(frame, b[3])


def lights(frame, B, extra=(), cam_pos=None):
    """LT rows for the terrain shader: the pool of light each fire throws on its summit (flaring as it
    catches). cam_pos: the render camera (the shader's surfaces are dropped by the Earth's curvature)."""
    LT = []
    for b in B:
        s, size, inten, light, fl, fla = _fire_state(frame, b)
        if light <= 0:
            continue
        Hv = VIS * s['Hf']                   # the light comes from the visible fire
        # a warm pool out to ~2 fire-heights at about moonlight level, ~3x that at the fire's foot
        I0 = 16.0 * (Hv / 4.0) ** 2 * (1.0 if s['beat'] else s['bv'])
        p = _base(b, s, cam_pos)
        LT.append([p[0], p[1] + 0.35 * Hv * min(size, 1.0), p[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2],
                   I0 * light * fl * (1.0 + 1.5 * fla), 0.6 * Hv])
    for e in extra:
        LT.append(e)
    return np.array(LT, np.float64).reshape(-1, 8)


_PLUME = {}


def _plume(b):
    k = int(b[5])
    if k not in _PLUME:
        s = spec(b)
        t0 = b[3] / FPS
        if b[3] < 1500:
            t0 = 1500 / FPS - 5.0                     # long established
        _PLUME[k] = F2.Plume(300 + k, _base(b, s, RN.path(min(max(b[3], RN.START), RN.END))), VIS * s['Hf'], t0,
                             rate=16.0 if s['beat'] else 8.0,
                             life=4.2 if s['beat'] else 3.0, dens=0.42 if s['beat'] else 0.28)
    return _PLUME[k]


def fog_tr(cam_pos, P, fogp, Q=None):
    """Transmittance of the Run's air (all fog layers) between the camera and a world point."""
    d = float(np.linalg.norm(np.asarray(P) - cam_pos))
    tau = WD.height_fog_tau(d, cam_pos[1], P[1], fogp[0], fogp[1])
    tau += WD.height_fog_tau(d, cam_pos[1] - WD.CLOUD_Y, P[1] - WD.CLOUD_Y, fogp[2], fogp[3])
    if Q is not None and Q[18] > 0:
        tau += WD.height_fog_tau(d, cam_pos[1] - WD.CLOUD_Y, P[1] - WD.CLOUD_Y, Q[18], Q[19])
    return math.exp(-tau)


def _base(b, s, cam_pos=None):
    """The fire's base in world space; with cam_pos, dropped by the Earth's curvature as the terrain is
    rendered (d^2 / 2R from the camera: 5 m at 8 km, 110 m at 37 km)."""
    y = b[1] + 0.03 * s['Hf']
    if cam_pos is not None:
        y -= ((b[0] - cam_pos[0]) ** 2 + (b[2] - cam_pos[2]) ** 2) / (2.0 * WD.R_EARTH)
    return np.array([b[0], y, b[2]])


def draw_source(img, zb, scam, frame, B, pxs, fogp, moon_dir, amb, Q=None):
    """Source-space part: only the heroine's far beacon (s1's far-light recipe, so its glow at the cut
    matches s1's last frame). Everything else is drawn after the motion blur (draw_fx_target)."""
    for b in B:
        if b[3] >= 1500:
            continue
        s, size, inten, light, fl, fla = _fire_state(frame, b)
        base = _base(b, s, scam.pos)
        sx, sy, z = scam.project(base)
        if z <= 1.0:
            continue
        hv = VIS * s['Hf'] * scam.f / z
        e = light * fl
        # verbatim s1_peak's far light (glow + two halos, same pixel sizes and energy): across the cut from
        # s1's last frame it is the same light, not a smaller one
        F.glow(img, zb, sx, sy - 0.3 * hv, 0.9 * pxs + 0.45, 14.0 * e * pxs * pxs, z=z, zbias=500.0)
        F2.halo(img, zb, sx, sy - 0.3 * hv, 14.0 * pxs, 0.03 * e, z=z, zbias=500.0)
        F2.halo(img, zb, sx, sy - 0.3 * hv, 60.0 * pxs, 0.006 * e, z=z, zbias=500.0)


def _halo_blur(img, zt, cx, cy, offs, sig, peak, z, zbias, col=None, squash=1.0):
    K = len(offs)
    for o in offs:
        F2.halo(img, zt, cx + o[0], cy + o[1], sig, peak / K, z=z, zbias=zbias, col=col, squash=squash)


def draw_fx_target(img, zt, frame, W, H, B, fogp, moon_dir, amb, Q=None, shutter=0.35, cam_fn=None):
    """Every beacon after the frame's motion blur, in TARGET space, far to near: its smoke column (lit from
    below, moonlit above, in the haze), its air glow and the 2-3 frame catch flare, then the flame itself.
    Glows and flames are laid down along their own screen path through the open shutter (an exact blur of a
    rigid sprite), so a small fire keeps its shape through the zoom and the pans."""
    cam_fn = cam_fn or (lambda f: RN.camera(f, W, H))
    t = frame / FPS
    c0 = cam_fn(frame)
    ca = cam_fn(frame - 0.5 * shutter)
    cb = cam_fn(frame + 0.5 * shutter)
    fogc = fogp[5:8]
    pxs = W / 1920.0
    order = np.argsort([-np.linalg.norm(b[:3] - c0.pos) for b in B])
    for i in order:
        b = B[i]
        s, size, inten, light, fl, fla = _fire_state(frame, b)
        if size <= 0:
            continue
        base = _base(b, s, c0.pos)
        sx, sy, z = c0.project(base)
        if z <= 1.0:
            continue
        ppm = c0.f / z
        Hf = s['Hf'] * size * (1.0 + 0.06 * (fl - 1.0))
        hpx = Hf * ppm
        m = max(4.0 * s['Hf'] * ppm, 60.0)
        if sx < -m or sx > W + m or sy < -m or sy > H + 2 * m:
            continue
        tr = 1.0 if b[3] < 1500 else fog_tr(c0.pos, base, fogp, Q)
        zbias = max(3.0, 0.004 * z, 0.6 * s['Hf'])
        e = light * fl * s['bv']
        pa = np.array(ca.project(base)[:2])
        pb = np.array(cb.project(base)[:2])
        d = (pb - pa) * FIRE_BLUR
        K = int(min(max(math.ceil(np.linalg.norm(d) / 1.0), 1), 16))
        u = (np.arange(K) + 0.5) / K - 0.5
        offs = (u[:, None] * d[None, :]).astype(np.float64)
        ang = c0.up_screen(base, h=max(s['Hf'], 1.0))
        ux, uy = math.sin(ang), -math.cos(ang)
        hv = VIS * hpx
        # smoke (soft: drawn at the shutter centre)
        if b[3] >= 1500 and s['Hf'] * ppm > 2.0:
            mp = c0.project(base + moon_dir * 1000.0)
            m2 = np.array([mp[0] - sx, mp[1] - sy])
            m2 /= max(np.linalg.norm(m2), 1e-6)
            _plume(b).render(img, zt, c0, t, 22.0 * e, m2, 0.55 * 0.5, amb, fogc, tr, zbias)
        if b[3] >= 1500:
            # air glow: tight and warm, stronger in the flash (the catch lights the air around it)
            cx, cy = sx + ux * 0.4 * hv, sy + uy * 0.4 * hv
            g = e * tr ** 0.35
            _halo_blur(img, zt, cx, cy, offs, max(0.6 * hv, 1.6 * pxs),
                       (0.020 if s['beat'] else 0.014) * g * (1.0 + 4.0 * fla), z, zbias)
            F2.halo(img, zt, cx, cy, max(2.2 * hv, 5.0 * pxs), (0.0024 if s['beat'] else 0.0014) * g * (1.0 + 3.0 * fla),
                    z=z, zbias=zbias, col=F2.AURA_COL)
            if fla > 0.02:
                # the catch: a 2-3 frame gold flare that shoots up the flame (tall, not round)
                hf = max(VIS * s['Hf'] * ppm, 3.0 * pxs)
                _halo_blur(img, zt, sx + ux * 0.6 * hf, sy + uy * 0.6 * hf, offs, 0.32 * hf,
                           (1.6 if s['beat'] else 0.9) * fla * tr ** 0.35, z, zbias, col=np.array([1.0, 0.70, 0.36]),
                           squash=2.4)
                F2.halo(img, zt, sx + ux * 0.45 * hf, sy + uy * 0.45 * hf, 1.2 * hf,
                        (0.10 if s['beat'] else 0.05) * fla * tr ** 0.35, z=z, zbias=zbias, squash=1.3,
                        col=F2.AURA_COL)
        # the flame
        I = (26.0 if s['beat'] else 22.0) * inten * s['bv'] * tr ** 0.6 * (1.0 + 0.6 * fla) * fl
        E, A, x0, y0 = F2.flame_sprite(sx, sy, ppm, Hf, s['Rb'] * (0.85 + 0.15 * size), t, seed=s['seed'], I=I,
                                        lean=0.05 * s['Hf'] * size, TG=s['TG'], warp=1.2, absorb=0.6, ang=ang)
        F2.composite_blur(img, zt, E, A, x0, y0, offs, float(z), float(zbias))
