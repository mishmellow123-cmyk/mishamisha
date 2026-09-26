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

def env(frame, f0):
    t = frame / FPS
    return F.ignite_env(t, f0 / FPS)


def platforms(B, extra=()):
    """Cleared rock platforms under every beacon (x, y, z, radius)."""
    out = [[b[0], b[1], b[2], 4.5 if b[4] == 1 else 2.8] for b in B if b[6] < 6000.0]
    out += list(extra)
    return np.array(out, np.float64).reshape(-1, 4)


def lights(frame, B, extra=()):
    """LT rows for the terrain shader."""
    LT = []
    for b in B:
        size, inten, light = env(frame, b[3])
        if light <= 0:
            continue
        I0 = 60.0 if b[4] == 1 else 14.0
        fl = F.flicker(frame / FPS, int(b[5]) + 3)
        LT.append([b[0], b[1] + (2.2 if b[4] == 1 else 1.3), b[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2],
                   I0 * light * fl, 0.8 if b[4] == 1 else 0.5])
    for e in extra:
        LT.append(e)
    return np.array(LT, np.float64).reshape(-1, 8)


_SMOKE = {}


def _smoke(b):
    k = int(b[5])
    if k not in _SMOKE:
        big = b[4] == 1
        _SMOKE[k] = F.Smoke(100 + k, np.array([b[0], b[1] + (4.0 if big else 2.2), b[2]]), b[3] / FPS + 0.2,
                            1690 / FPS, rate=3.0 if big else 2.5, wind=(1.6, 0.0, 0.4), rise=2.2 if big else 1.5,
                            life=6.0, r0=0.9 if big else 0.4, growth=1.1 if big else 0.6, dens=0.45)
    return _SMOKE[k]


def draw_source(img, zb, scam, frame, B, pxs):
    """Glows, halos, flames and smoke of every lit beacon, in source space. pxs = pixel scale vs 1920."""
    t = frame / FPS
    order = np.argsort([-np.linalg.norm(b[:3] - scam.pos) for b in B])
    for i in order:
        b = B[i]
        size, inten, light = env(frame, b[3])
        if size <= 0:
            continue
        big = b[4] == 1
        base = np.array([b[0], b[1] + (1.6 if big else 1.1), b[2]])
        sx, sy, z = scam.project(base)
        if z <= 1.0:
            continue
        W, H = img.shape[1], img.shape[0]
        if sx < -300 or sx > W + 300 or sy < -300 or sy > H + 300:
            continue
        ppm = scam.f / z
        fl = F.flicker(t, int(b[5]) + 11, 1.3)
        dist = z
        zbias = max(3.0, 0.004 * dist)
        Hf = (6.5 if big else 2.4) * size
        Rb = (1.0 if big else 0.42)
        # smoke plume (near-ish only)
        if dist < 4000.0:
            _smoke(b).render(img, zb, scam, t, base + np.array([0, 0.8 * Hf, 0]), (2.5 if big else 0.8) * light * fl,
                             albedo=0.25, amb=(0.004, 0.005, 0.009), zbias=zbias)
        # air glow around the fire (world-sized, clamped in px)
        e = light * fl
        sig1 = max(7.0 * ppm, 2.2 * pxs)
        sig2 = max(30.0 * ppm, 9.0 * pxs)
        pk = 1.0 if big else 0.45
        F2.halo(img, zb, sx, sy - 0.4 * Hf * ppm, sig1, 0.018 * pk * e, z=z, zbias=zbias)
        F2.halo(img, zb, sx, sy - 0.4 * Hf * ppm, sig2, 0.0016 * pk * e, z=z, zbias=zbias)
        # the flame (bonfire when resolved; a hot point otherwise)
        F2.flame(img, zb, scam, base, Hf, Rb, t, seed=int(b[5]) * 7 + 3, I=(30.0 if big else 24.0) * inten,
                 lean=0.25 * size, zbias=zbias, tongues=7 if big else 5)
        # unresolved fires still need to read as fires at distance: a minimum hot point
        if Hf * ppm < 3.0 * pxs:
            # a distant beacon: a warm hot point with a soft orange aura (a fire, not a lamp)
            F2.glow(img, zb, sx, sy - 0.3 * Hf * ppm, 0.95 * pxs, (6.5 if big else 3.5) * e * pxs * pxs, z=z,
                    zbias=zbias, col=np.array([1.0, 0.62, 0.26]))
            F2.halo(img, zb, sx, sy - 0.3 * Hf * ppm, 3.2 * pxs, (0.10 if big else 0.06) * e, z=z, zbias=zbias,
                    col=F2.AURA_COL)
            F2.halo(img, zb, sx, sy - 0.3 * Hf * ppm, 11.0 * pxs, (0.012 if big else 0.007) * e, z=z, zbias=zbias,
                    col=F2.AURA_COL)
