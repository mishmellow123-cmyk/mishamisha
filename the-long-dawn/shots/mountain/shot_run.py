"""THE BEACON RUN (v2 frames 1520-1679): camera path + beacons (venv python).

python shot_run.py          -> cache/run_cam.json, cache/design_peaks.json, cache/run_beacons.json
python shot_run.py check    -> + framing report (clearance, rotation rates, beacon screen positions)

Order of construction (no circularity):
  1. the camera path (designed by hand; independent of the land except the story zone)
  2. B1/B2: designed horn peaks placed by inverse projection from their ignition frames
  3. the land grids (with those horns), then the chain: real summits chosen so the fires line
     up for the final composition (converging from lower-left to the horizon) and are visible
     both when they ignite and at the end.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mtn import campath, land, bake
import world

CACHE = os.path.join(HERE, 'cache')
F0, F1 = 1520, 1679
HANDLE = 3                                  # extra frames each side (motion blur, sims)
FRAMES = np.arange(F0 - HANDLE, F1 + HANDLE + 1)
W, H = 1920, 804

# ------------------------------------------------------------------ camera ---
HORIZ = [(-40.0, -135.0), (-62.0, -55.0), (-70.0, 30.0), (-56.0, 115.0), (-24.0, 215.0),
         (6.0, 320.0), (32.0, 405.0), (58.0, 505.0), (82.0, 600.0), (108.0, 700.0),
         (132.0, 800.0), (150.0, 900.0), (165.0, 1000.0), (175.0, 1100.0), (182.0, 1200.0),
         (186.0, 1300.0), (188.0, 1400.0)]
SPEED = [(1508, 118.0), (1520, 128.0), (1530, 172.0), (1542, 215.0), (1555, 222.0),
         (1568, 205.0), (1580, 185.0), (1590, 176.0), (1600, 150.0), (1612, 118.0),
         (1625, 92.0), (1640, 74.0), (1660, 60.0), (1679, 52.0), (1692, 48.0)]
ALT = [(1508, 352.0), (1520, 348.0), (1530, 336.0), (1540, 298.0), (1550, 240.0),
       (1560, 190.0), (1568, 186.0), (1576, 198.0), (1582, 207.0), (1588, 213.0),
       (1593, 222.0), (1598, 250.0), (1606, 330.0), (1614, 420.0), (1622, 510.0), (1632, 610.0),
       (1642, 700.0), (1652, 780.0), (1662, 850.0), (1672, 905.0), (1679, 935.0),
       (1692, 975.0)]
CROSS_FRAME = 1588          # closest approach to the great pyre
FINAL_YAW = 13.0            # bearing the camera settles on for the reveal (deg)


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def pads():
    P = world.land_args()[0]
    pd = P[P[:, 4] == 1]
    return ((float(pd[0, 0]), float(pd[0, 1]), float(pd[0, 2])),
            (float(pd[1, 0]), float(pd[1, 1]), float(pd[1, 2])))


def make_path():
    s0, b3 = pads()
    s0 = np.array(s0) + np.array([0, 0, 3.0])
    b3 = np.array(b3) + np.array([0, 0, 4.0])
    spd = [list(k) for k in SPEED]
    for it in range(6):
        wp = campath.explicit_track(HORIZ, spd, ALT, 1520, 0.0, 1508, 1692)
        Wp = np.array(wp)
        d = np.linalg.norm(Wp[:, 1:3] - np.array(b3[:2]), axis=1)
        fc = Wp[np.argmin(d), 0]
        k = 1.0 + (fc - CROSS_FRAME) / (CROSS_FRAME - 1520) * 0.9
        for row in spd:
            if row[0] <= CROSS_FRAME:
                row[1] *= k
        if abs(fc - CROSS_FRAME) < 0.3:
            break
    wp = campath.explicit_track(HORIZ, spd, ALT, 1520, 0.0, 1508, 1692)
    far = np.array([0.0, 0.0, 0.0])

    def look(f):
        if f < 1540:
            return s0, 0.5 * (1 - smoothstep(1524, 1539, f))
        if 1560 < f < 1590:
            w = 0.22 * smoothstep(1560, 1574, f) * (1 - smoothstep(1580, 1589, f))
            return b3, w, 0.6 * w
        if f >= 1596:
            # settle onto a fixed bearing for the reveal (a point far along FINAL_YAW)
            p = wp[-1]
            tgt = np.array([p[1] + 1e6 * math.sin(math.radians(FINAL_YAW)),
                            p[2] + 1e6 * math.cos(math.radians(FINAL_YAW)), p[3]])
            return tgt, 0.85 * smoothstep(1596, 1650, f), 0.0
        return None, 0.0

    def pitch_off(f):
        return (2.0 - 3.5 * smoothstep(1562, 1578, f) - 8.0 * smoothstep(1588, 1608, f)
                - 3.5 * smoothstep(1626, 1679, f))

    rng_t = campath._smooth_noise

    def turb(f):
        t = f / 24.0
        base = 0.35
        kick = 0.0
        if f >= 1580:
            u = (f - 1580) / 24.0
            kick = 1.4 * math.exp(-u * 3.0) * math.sin(2 * math.pi * 2.2 * u)
        return (base * rng_t(np.array([t]), 3, 0.35)[0] + kick,
                0.5 * base * rng_t(np.array([t]), 4, 0.3)[0] + 0.4 * kick,
                0.4 * base * rng_t(np.array([t]), 5, 0.25)[0])

    def hfov(f):
        return 62.0 - 6.0 * smoothstep(1596, 1679, f)

    path = campath.Path(wp, FRAMES, look=look, pitch_k=0.18, pitch_off=pitch_off,
                        bank_gain=0.3, bank_max=9.0, turb=turb, hfov=hfov)
    return path


def project(cam, P):
    c = np.array(cam['loc'])
    f = np.array(cam['fwd'])
    r = np.array(cam['right'])
    u = np.array(cam['up'])
    d = np.asarray(P, float) - c
    z = d @ f
    if z <= 0:
        return None
    th = math.tan(math.radians(cam['hfov'] / 2))
    sx = W / 2 + (d @ r) / z / th * W / 2
    sy = H / 2 - (d @ u) / z / th * W / 2
    return sx, sy, z


def unproject(cam, sx, sy, hdist):
    """World point on the pixel ray (sx, sy) at horizontal distance hdist."""
    c = np.array(cam['loc'])
    f = np.array(cam['fwd'])
    r = np.array(cam['right'])
    u = np.array(cam['up'])
    th = math.tan(math.radians(cam['hfov'] / 2))
    x = (sx - W / 2) / (W / 2) * th
    y = -(sy - H / 2) / (W / 2) * th
    d = f + x * r + y * u
    d /= np.linalg.norm(d)
    t = hdist / math.hypot(d[0], d[1])
    return c + d * t


# ------------------------------------------------------------ story peaks ---
# (name, ignition frame, screen x, screen y, horizontal distance, radius) -> designed horns
STORY_HORNS = [('B1', 1540, 1230.0, 262.0, 2700.0, 760.0),
               ('B2', 1560, 610.0, 238.0, 3400.0, 820.0)]


def design_story_peaks(cams, ref, clearings=None):
    byf = {int(c['frame']): c for c in cams}
    rows = []
    for name, t, sx, sy, hd, R in STORY_HORNS:
        p = unproject(byf[t], sx, sy, hd)
        rows.append(dict(name=name, x=float(p[0]), y=float(p[1]), H=float(p[2]) - 3.0, R=R))
    rows += chain_designs(byf[F1], ref)
    rows += clearings or []
    json.dump(rows, open(os.path.join(CACHE, 'design_peaks.json'), 'w'), indent=1)
    world.reset_args()
    return rows


def first_block(G, c, p, ref, clear=6.0):
    """(x, y, excess) of the worst obstruction on the segment c -> p, or None."""
    (G0, M0), (G1, M1), (G2, M2) = G
    d = p - c
    L = math.hypot(d[0], d[1])
    n = int(max(L / 10.0, 50))
    worst = None
    for k in range(1, n):
        t = k / n
        if (1 - t) * L < 300.0:
            break
        x, y, z = c + d * t
        h = bake.hgrid(G0, M0, G1, M1, G2, M2, x, y) - ((x - ref[0]) ** 2 + (y - ref[1]) ** 2) / (2 * 6.371e6)
        ex = h - (z - clear)
        if ex > 0 and (worst is None or ex > worst[2]):
            worst = (x, y, ex)
    return worst


# ------------------------------------------------------------ the chain ---
# The chain recedes along a line converging to a vanishing point on the final frame's horizon:
# screen x(d) = VP_X + (NEAR_X - VP_X) * NEAR_D / d; summit height z(d) = LINE_Z0 + LINE_DZ * d
# (above the curved datum). (name, ignition frame, nominal distance at 1679, scale)
VP_X, NEAR_X, NEAR_D = 1080.0, 470.0, 6000.0
LINE_Z0, LINE_DZ = 430.0, 0.0085
CHAIN = [
    ('B4', 1600, 6000.0, 1.6),
    ('F1', 1611, 8600.0, 1.4),
    ('B5', 1620, 11800.0, 1.7),
    ('F2', 1631, 15800.0, 1.6),
    ('B6', 1640, 21000.0, 1.9),
    ('F3', 1651, 27500.0, 2.0),
    ('B7', 1660, 36000.0, 2.3),
    ('F4', 1665, 47000.0, 2.5),
    ('F5', 1669, 61000.0, 2.9),
    ('F6', 1673, 79000.0, 3.3),
    ('F7', 1676, 101000.0, 3.7),
    ('F8', 1678, 128000.0, 4.1),
]
# side branches: the answer spreading across the other ranges (final-frame x targets)
SIDE = [
    ('S1', 1606, 1560.0, (3000, 9000), 1.4),
    ('S2', 1618, 250.0, (4000, 12000), 1.4),
    ('S3', 1628, 1720.0, (7000, 16000), 1.6),
    ('S4', 1636, 120.0, (9000, 20000), 1.7),
    ('S5', 1646, 1400.0, (12000, 26000), 1.9),
    ('S6', 1654, 520.0, (15000, 32000), 2.1),
    ('S7', 1662, 1300.0, (22000, 45000), 2.4),
    ('S8', 1667, 700.0, (28000, 60000), 2.6),
]


def chain_designs(fin, ref):
    """Designed great peaks for the chain, placed on the line in the final frame."""
    rows = []
    for name, t, d, sc in CHAIN:
        tx = VP_X + (NEAR_X - VP_X) * NEAR_D / d
        # point on the column of screen x = tx at horizontal distance d, at the line height
        c = np.array(fin['loc'])
        p0 = unproject(fin, tx, H / 2, d)
        zline = LINE_Z0 + LINE_DZ * d
        cz = ((p0[0] - ref[0]) ** 2 + (p0[1] - ref[1]) ** 2) / (2 * 6.371e6)
        R = 520.0 + 0.028 * d
        rows.append(dict(name=name, x=float(p0[0]), y=float(p0[1]), H=float(zline + cz), R=R,
                         kind=0))
    return rows


def summit_candidates(G):
    (G0, M0), (G1, M1), (G2, M2) = G
    cand = []
    for g, m, r, hmin in ((G1, M1, 7, 120.0), (G2, M2, 3, 250.0)):
        idx = bake.local_maxima(g.astype(np.float64), r, hmin)
        for (j, i) in idx:
            x = m[0] + i * m[2]
            y = m[1] + j * m[2]
            if g is G2 and M1[0] < x < M1[0] + G1.shape[1] * M1[2] and \
                    M1[1] < y < M1[1] + G1.shape[0] * M1[2]:
                continue
            cand.append((x, y, float(g[j, i])))
    return np.array(cand)


def refine_summit(x, y, rad, step):
    args = world.land_args()
    for r, s in ((rad, step), (step * 1.5, 1.0)):
        xs = np.arange(x - r, x + r + 1e-6, s)
        X, Y = np.meshgrid(xs, np.arange(y - r, y + r + 1e-6, s))
        Hh = land.eval_points(X.ravel(), Y.ravel(), np.full(X.size, 2.0), *args)
        i = int(np.argmax(Hh))
        x, y = float(X.ravel()[i]), float(Y.ravel()[i])
    return x, y, float(Hh[i])


def curv(p, ref):
    return ((p[0] - ref[0]) ** 2 + (p[1] - ref[1]) ** 2) / (2 * 6.371e6)


def place_chain(cams, G, ref):
    (G0, M0), (G1, M1), (G2, M2) = G
    byf = {int(c['frame']): c for c in cams}
    fin = byf[F1]
    cl = np.array(fin['loc'])
    C = summit_candidates(G)
    used = []
    out = []
    jobs = list(SIDE)
    for r in json.load(open(os.path.join(CACHE, 'design_peaks.json'))):
        if r.get('kind', 0) == 0:
            used.append((r['x'], r['y']))
    for name, t, sx_t, (dmin, dmax), sc in jobs:
        best = None
        for (x, y, z) in C:
            hd = math.hypot(x - cl[0], y - cl[1])
            if not (dmin <= hd <= dmax):
                continue
            if any(math.hypot(x - u[0], y - u[1]) < 700.0 for u in used):
                continue
            zc = z - curv((x, y), ref) + 3.0
            pr = project(fin, (x, y, zc))
            if pr is None or not (30 < pr[0] < W - 30 and 30 < pr[1] < H - 30):
                continue
            if sx_t is None:
                tx = VP_X + (NEAR_X - VP_X) * NEAR_D / hd
                tz = LINE_Z0 + LINE_DZ * hd - curv((x, y), ref)
                # the target point: on the ray of screen x = tx at this distance, height tz
                ty = project(fin, (x, y, tz))[1]
                err = math.hypot(pr[0] - tx, 2.0 * (pr[1] - ty))
            else:
                err = abs(pr[0] - sx_t) - 0.02 * z
            if best is not None and err >= best[0]:
                continue
            ok = bake.visible(G0, M0, G1, M1, G2, M2, cl[0], cl[1], cl[2], x, y, zc + 2.0,
                              ref[0], ref[1])
            if not ok:
                continue
            ci = byf[t]
            pi = project(ci, (x, y, zc))
            if pi is None or not (40 < pi[0] < W - 40 and 30 < pi[1] < H - 30):
                continue
            li = np.array(ci['loc'])
            if not bake.visible(G0, M0, G1, M1, G2, M2, li[0], li[1], li[2], x, y, zc + 2.0,
                                ref[0], ref[1]):
                continue
            best = (err, x, y, z)
        if best is None:
            print('  (no summit for %s)' % name)
            continue
        _, x, y, z = best
        if math.hypot(x - 1000, y - 12000) < 21000:
            x, y, z = refine_summit(x, y, 30.0, 5.0)
        used.append((x, y))
        out.append(dict(name=name, pos=(x, y, z), t=t, scale=sc))
    return out


def beacons(cams, ref, G):
    s0, b3 = pads()
    out = [dict(name='S0', pos=s0, t=1480, scale=1.0),
           dict(name='B3', pos=b3, t=1580, scale=1.7)]
    tmap = {s[0]: s[1] for s in STORY_HORNS}
    tmap.update({c[0]: c[1] for c in CHAIN})
    smap = {s[0]: 1.4 for s in STORY_HORNS}
    smap.update({c[0]: c[3] for c in CHAIN})
    for r in json.load(open(os.path.join(CACHE, 'design_peaks.json'))):
        if r.get('kind', 0) != 0:
            continue
        near = math.hypot(r['x'] - 1000, r['y'] - 12000) < 21000
        if near:
            x, y, z = refine_summit(r['x'], r['y'], 60.0 if r['R'] < 1500 else 150.0, 5.0)
        else:
            x, y = r['x'], r['y']
            z = float(land.eval_points(np.array([x]), np.array([y]), np.array([20.0]),
                                       *world.land_args())[0])
        out.append(dict(name=r['name'], pos=(x, y, z), t=tmap[r['name']], scale=smap[r['name']]))
    out += place_chain(cams, G, ref)
    return out


def solve_designs(cams, ref, iters=4):
    """Designed peaks + clearing depressions until every chain fire is seen when it ignites
    and at the end."""
    import build_mesh as BM
    byf = {int(c['frame']): c for c in cams}
    clearings = []
    for it in range(iters):
        design_story_peaks(cams, ref, clearings)
        G = BM.grids(force=True, quick=(it < iters - 1), tag='%s_it' % world.LAND_VERSION)
        added = 0
        for r in json.load(open(os.path.join(CACHE, 'design_peaks.json'))):
            if r.get('kind', 0) != 0:
                continue
            t = dict((c[0], c[1]) for c in CHAIN + [(s[0], s[1]) for s in STORY_HORNS])[r['name']]
            zc = r['H'] - ((r['x'] - ref[0]) ** 2 + (r['y'] - ref[1]) ** 2) / (2 * 6.371e6) + 4.0
            p = np.array([r['x'], r['y'], zc])
            for f in (t, F1):
                c = np.array(byf[f]['loc'])
                blk = first_block(G, c, p, ref)
                if blk is not None:
                    x, y, ex = blk
                    dist = math.hypot(x - c[0], y - c[1])
                    clearings.append(dict(name='clr', x=x, y=y, H=-(ex + 40.0),
                                          R=max(350.0, 0.03 * dist), kind=2))
                    added += 1
        print('design iteration', it, 'clearings added', added, flush=True)
        if added == 0:
            break
    return BM.grids(force=True, quick=False)


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    path = make_path()
    cams = path.evaluate(sig_roll=5.0, sig_yp=2.5)
    campath.save(os.path.join(CACHE, 'run_cam.json'), cams, dict(F0=F0, F1=F1))
    ref = np.array([c['loc'] for c in cams]).mean(0)
    G = solve_designs(cams, ref)
    B = beacons(cams, ref, G)
    json.dump(B, open(os.path.join(CACHE, 'run_beacons.json'), 'w'), indent=1)
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        args = world.land_args()
        print('frame  pos                       speed  clear   yaw  pitch  roll')
        for c in cams[::6]:
            x, y, z = c['loc']
            g = land.eval_points(np.array([x]), np.array([y]), np.array([2.0]), *args)[0]
            print('%5d  (%6.0f %6.0f %5.0f)  %5.0f  %5.0f  %5.1f  %5.1f  %5.1f' % (
                c['frame'], x, y, z, c['speed'], z - max(g, 0.0), c['yaw'], c['pitch'], c['roll']))
        byf = {int(c['frame']): c for c in cams}
        prev = None
        worst = []
        for c in cams:
            if prev is not None:
                a = math.degrees(math.acos(max(-1, min(1, np.dot(c['fwd'], prev['fwd'])))))
                worst.append((a, int(c['frame']), c['roll'] - prev['roll']))
            prev = c
        worst.sort(reverse=True)
        print('rotation worst (frame: view deg / roll deg):',
              ['%d:%.2f/%.2f' % (f, a, r) for a, f, r in worst[:8]])
        print('\nbeacon  t     at ignition (x, y, dist)          at 1679')
        for b in B:
            p = np.array(b['pos'], float)
            p[2] -= curv(p, ref)
            pi = project(byf[b['t']], p) if b['t'] in byf else None
            pf = project(byf[F1], p)
            fmt = lambda q: 'behind' if q is None else '(%5.0f %5.0f %7.0f)' % q
            print('%-5s %5d  %-30s %s' % (b['name'], b['t'], fmt(pi), fmt(pf)))
