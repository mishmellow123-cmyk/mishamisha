"""ACCORD-CROWD (cut C, C frames 4464-5679): the emissaries of every people, their roads, the rivers of torches, the
crowd's ring round the stones, and bar 70's walk-out. It plugs into accord3 (see NOTES.md, "ACCORD-CROWD").

    state(t)                          the crowd at t: figure rows (geom3's layout), torch flames, torch lights
    ground_light(cs, st, cam, W, H)   the crowd's torchlight on the ground (shade3's igf/igc), each bearer's own shadow
    composite(rgb, depth, oid, cam, st, scale)   the land beyond the stones, then the crowd's figures over the frame
    flames(rgb, depth, cam, st, scale)           the crowd's torch flames (flame3's volumes) and their airlight
    occluders(cs)                     capsules for accord3's soft shadows (P3, near the fire)
    lit_fraction(cs)                  scales shade3's P_CROWD rim on the council
    her_walkin(t)                     her walk in along her road to her place (t < HER_ARRIVE)

Beats (C numbering; bar n starts at (n-1)*80):
  4464-4760  the rivers converge on the ring of stones; arrivals settle in ranks round the stones (P1 layout)
  4760-5119  everyone stands, torch upright before them; nobody moves toward the centre (AC4)
  5120-5200  the crowd's torches come down with the council's and go out (AC2; off screen: light only)
  5520-5679  after the white the crowd has closed in round the fire, torches spent (P3 layout); the flame passes
             torch to torch outward from the council's dip, and each bearer turns and walks out: the lights stream
             OUTWARD along the roads.
"""
import glob
import hashlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.path.dirname(HERE)
if ACC not in sys.path:
    sys.path.insert(0, ACC)


def _stamp():
    """Own numba cache stamp over accord/*.py + crowd/*.py (never touches accord's own cache)."""
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(ACC, '*.py')) + glob.glob(os.path.join(HERE, '*.py'))):
        with open(p, 'rb') as fh:
            h.update(fh.read())
    d = os.path.join(HERE, '__pycache__')
    os.makedirs(d, exist_ok=True)
    stp = os.path.join(d, 'src.sha1')
    old = open(stp).read() if os.path.exists(stp) else ''
    if old != h.hexdigest():
        for f in glob.glob(os.path.join(d, '*.nb[ic]')):
            try:
                os.remove(f)
            except OSError:
                pass
        with open(stp, 'w') as fh:
            fh.write(h.hexdigest())


_stamp()

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from numba import njit, prange  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import scene3 as SC  # noqa: E402
import geom3 as G  # noqa: E402
import shade3 as SH  # noqa: E402
import flame3 as FL3  # noqa: E402
import nbcore  # noqa: E402
from nbcore import FM, clamp, sstep, mix, vnoise2, fbm2, grid_sample, tex3  # noqa: E402

# ================================================================ constants ===
T_FREEZE = 4760.0          # the rivers have arrived, or stop where they stand (the crowd is off screen after ~4690)
T_SCHED0 = 3380.0          # the first arrivals at the mouths (the gathering has been going on a long while)
R_FRONT = 8.45             # P1: the crowd's front rank, just outside the stones
RANK_DR = 1.12
N_RANKS = 5
SPACING = 1.16             # along a rank
R_WALK = 14.35             # the walkway round the back of the crowd
R_MOUTH = 15.2             # where the roads end
R_GAP = 8.0                # the roads' worn tracks start in the gaps between the stones
R_FAR = 215.0              # the network's far ends (beyond the widest frame, 4480)
TRUNK_DEG = np.array([357.5, 67.5, 123.0, 162.0, 195.5, 254.5, 321.5])
TRUNK_AZ = np.radians(TRUNK_DEG)
HER_TRUNK = 4              # her road comes in through the gap at 195.5 deg, beside her place (200 deg)
AISLE_HALF = 0.85          # her aisle through the crowd
HER_V = 0.95               # m/s: she is small and has come far
HER_ARRIVE = 4704.0        # she takes her place in the council circle
ZTOP = 2.45                # nobody's torch flame base is higher than this (figure slab for the ray tests)
BIN = 2.0                  # figure bins (m)
# P3 (bar 70)
P3_R0 = 2.62               # the front rank behind the council (their P3 circle is at 1.32 m)
P3_DR = 1.08
P3_RMAX = 12.3
P3_SP = 1.05
T_W0 = 5563.0              # the front rank catches from the council's newly lit torches
WAVE_V = 0.14              # m per frame: the flame passes torch to torch outward
Q_RMAX = 50.0              # the queues on the roads (P3) reach this far out
# light
TORCH_I = 1.25             # = accord3.flames_and_lights (a torch's light, before its flicker)
FLAME_I = 16.0             # accord3's council torches use 22; the crowd's are a touch lower so cores stay yellow
FIRE_HOT = SC.FIRE_HOT
CROWD_SEED = 20260927
MAXF_RENDER = 6000

_CLOTH = np.array([
    (0.0120, 0.0116, 0.0112),   # charcoal
    (0.0165, 0.0118, 0.0085),   # umber
    (0.0190, 0.0166, 0.0132),   # undyed dark wool
    (0.0138, 0.0136, 0.0133),   # grey
    (0.0152, 0.0113, 0.0087),   # brown
    (0.0092, 0.0090, 0.0088),   # black
    (0.0100, 0.0112, 0.0146),   # blue-grey (rare)
    (0.0172, 0.0138, 0.0104),   # dun
    (0.0124, 0.0097, 0.0079),   # peat
    (0.0115, 0.0118, 0.0124),   # slate
])
_CLOTH_P = np.array([0.16, 0.15, 0.10, 0.10, 0.13, 0.12, 0.04, 0.07, 0.09, 0.04])


def _sm(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smr(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _ramp(t, a, b):
    return np.clip((t - a) / (b - a), 0.0, 1.0)


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


_A3 = {}


def _accord3():
    return _A3.get('m') or sys.modules.get('accord3')


def wind():
    A3 = _accord3()
    return np.asarray(A3._WIND, np.float64) if A3 is not None and hasattr(A3, '_WIND') else np.array([0.40, -0.16])


def haze():
    """The smoke-haze density accord3 used for this frame's airlight (recorded by the FL3 proxy), else its HAZE."""
    if 'hz' in _A3:
        return float(_A3['hz'])
    A3 = _accord3()
    return float(A3.HAZE) if A3 is not None and hasattr(A3, 'HAZE') else 0.0025


# ==================================================================== roads ===

def _meander(rng, n=3, lam=(38.0, 150.0)):
    ks = rng.uniform(1.0 / lam[1], 1.0 / lam[0], n) * 2 * np.pi
    ph = rng.uniform(0, 2 * np.pi, n)
    am = rng.uniform(0.5, 1.0, n)
    am /= am.sum()
    return lambda s: float(np.sum(am * np.sin(ks * s + ph)))


def _grow(rng, p0, heading, length, avoid, amp, rest_k, radial_until=0.0):
    """Walk a path outward in 1 m steps: a meandering heading, pulled back toward the outward radial, pushed away
    from the paths already laid (avoid: KD-tree or None)."""
    mean = _meander(rng)
    pts = [np.asarray(p0, np.float64)]
    h = heading
    p = pts[0].copy()
    for s in np.arange(1.0, length + 1.0):
        r = float(np.hypot(p[0], p[1]))
        rad = math.atan2(p[1], p[0])
        if r < radial_until:
            h = rad + 0.6 * _wrap(h - rad)
        else:
            k = amp * mean(s) * float(np.clip((r - radial_until) / 25.0, 0.0, 1.0))
            h += k + rest_k * _wrap(rad - h)
            if avoid is not None:
                d, j = avoid.query(p)
                if d < 26.0:
                    q = avoid.data[j]
                    away = math.atan2(p[1] - q[1], p[0] - q[0])
                    h += 0.06 * (1.0 - d / 26.0) * _wrap(away - h)
        p = p + np.array([math.cos(h), math.sin(h)])
        pts.append(p.copy())
    return np.array(pts)


def _arclen(P):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])


def _build_roads():
    """Seven trunks from gaps between the stones, each forking once or twice into tributaries. Routes run
    OUTWARD from the gap (s = 0 at R_GAP); a walker's route is trunk + one branch."""
    rng = np.random.default_rng(CROWD_SEED)
    laid = []
    trunks = []
    for j, a in enumerate(TRUNK_AZ):
        L = rng.uniform(48.0, 105.0)
        p0 = np.array([R_GAP * math.cos(a), R_GAP * math.sin(a)])
        tree = cKDTree(np.concatenate(laid)) if laid else None
        P = _grow(rng, p0, a, L, tree, amp=0.030, rest_k=0.035, radial_until=R_MOUTH + 3.0)
        trunks.append(P)
        laid.append(P[::2])
    routes = []
    for j, P in enumerate(trunks):
        h_end = math.atan2(*(P[-1] - P[-3])[::-1])
        nb = 2 if rng.random() < 0.75 else 3
        spread = rng.uniform(0.28, 0.52)
        for b in range(nb):
            dh = (b - (nb - 1) / 2.0) * spread * (2.0 / max(nb - 1, 1)) + rng.normal(0, 0.05)
            r_end = float(np.hypot(*P[-1]))
            L = max(R_FAR - r_end, 40.0) * rng.uniform(1.05, 1.25)
            tree = cKDTree(np.concatenate(laid))
            B = _grow(rng, P[-1], h_end + dh, L, tree, amp=0.045, rest_k=0.020)
            laid.append(B[::2])
            R = np.concatenate([P, B[1:]])
            routes.append(dict(trunk=j, pts=R, cum=_arclen(R), n_trunk=len(P), s_fork=_arclen(P)[-1]))
    # where each trunk passes the mouth (r = R_MOUTH) and the walkway (r = R_WALK)
    for rt in routes:
        r = np.hypot(rt['pts'][:, 0], rt['pts'][:, 1])
        rt['s_mouth'] = float(np.interp(R_MOUTH, r[:40], rt['cum'][:40]))
    return dict(trunks=trunks, routes=routes)


def route_at(rt, s):
    """Position (n,2) and unit tangent (outward) on route rt at arc lengths s (array)."""
    s = np.clip(np.asarray(s, np.float64), 0.0, rt['cum'][-1] - 1e-6)
    x = np.interp(s, rt['cum'], rt['pts'][:, 0])
    y = np.interp(s, rt['cum'], rt['pts'][:, 1])
    ds = 0.8
    x2 = np.interp(s + ds, rt['cum'], rt['pts'][:, 0])
    y2 = np.interp(s + ds, rt['cum'], rt['pts'][:, 1])
    x1 = np.interp(s - ds, rt['cum'], rt['pts'][:, 0])
    y1 = np.interp(s - ds, rt['cum'], rt['pts'][:, 1])
    tx, ty = x2 - x1, y2 - y1
    n = np.hypot(tx, ty) + 1e-9
    return np.stack([x, y], -1), np.stack([tx / n, ty / n], -1)


# ============================================================ the appearance ===

def _looks(n, rng):
    """Hoods and cloaks of many cuts and heights, in the undyed, umber and charcoal wools of every people:
    no costume, no regional dress, and no red (the one red is hers)."""
    L = {}
    hs = np.clip(rng.normal(1.01, 0.062, n), 0.88, 1.16)
    small = rng.random(n) < 0.035
    hs[small] = rng.uniform(0.86, 0.92, small.sum())
    L['hs'] = hs
    L['ws'] = np.clip(rng.normal(1.0, 0.055, n) * (0.55 + 0.45 * hs), 0.84, 1.20)
    L['typ'] = rng.choice([0, 1, 2, 3, 5], n, p=[0.22, 0.30, 0.22, 0.09, 0.17]).astype(np.float64)
    L['cape'] = np.where(rng.random(n) < 0.72, rng.uniform(0.26, 0.46, n), 0.0)
    L['train'] = rng.uniform(0.12, 0.30, n)
    bow = rng.uniform(2.0, 10.0, n)
    elder = rng.random(n) < 0.05
    bow[elder] = rng.uniform(13.0, 20.0, elder.sum())
    L['bow'] = np.radians(bow)
    L['side'] = rng.choice([-1.0, 1.0], n)
    L['weave'] = rng.uniform(0.7, 2.0, n)
    L['sheenk'] = rng.uniform(0.5, 1.6, n)
    c1 = rng.choice(len(_CLOTH), n, p=_CLOTH_P)
    c2 = np.where(rng.random(n) < 0.6, c1, rng.choice(len(_CLOTH), n, p=_CLOTH_P))
    tint = rng.uniform(0.80, 1.10, (n, 1))
    L['c1'] = _CLOTH[c1] * tint
    L['c2'] = _CLOTH[c2] * rng.uniform(0.80, 1.10, (n, 1))
    L['seed'] = rng.uniform(0.0, 60.0, n)
    L['bph'] = rng.uniform(0.0, 2 * np.pi, n)
    L['hf'] = 0.44 * rng.uniform(0.86, 1.14, n)
    L['fseed'] = rng.uniform(0.0, 200.0, n)
    L['ember'] = rng.uniform(0.05, 0.12, n)
    L['reach'] = rng.uniform(-0.03, 0.07, n)
    L['wander'] = rng.uniform(0.05, 0.14, n)
    L['wph'] = rng.uniform(0, 2 * np.pi, n)
    L['wper'] = rng.uniform(140.0, 260.0, n)
    return L


# ============================================================ P1: the rivers ===

def _stone_clear(x, y, S, margin):
    ok = np.ones(np.shape(x), bool)
    for j in range(S.shape[0]):
        d = np.hypot(x - S[j, G.S_X], y - S[j, G.S_Y])
        ok &= d > max(S[j, G.S_HX], S[j, G.S_HY]) + margin
    return ok


def _p1_slots(rng, S):
    sl = []
    for k in range(N_RANKS):
        r = R_FRONT + k * RANK_DR
        n = int(2 * np.pi * r / SPACING)
        az0 = rng.uniform(0, 2 * np.pi)
        az = az0 + 2 * np.pi * np.arange(n) / n + rng.normal(0, 0.14 * SPACING / r, n)
        rr = r + rng.normal(0, 0.13, n)
        x, y = rr * np.cos(az), rr * np.sin(az)
        keep = np.abs(_wrap(az - TRUNK_AZ[HER_TRUNK])) * rr > AISLE_HALF + 0.25
        keep &= _stone_clear(x, y, S, 0.48)
        for q in np.nonzero(keep)[0]:
            sl.append((x[q], y[q], rr[q], az[q] % (2 * np.pi), k))
    return np.array(sl)


def _build_p1(roads):
    """The walkers: groups (delegations) scheduled per trunk, each on one of its trunk's routes; the ones that reach
    their mouth before T_FREEZE take a place in the ring (front ranks first, near their mouth)."""
    rng = np.random.default_rng(CROWD_SEED + 1)
    S = SC.stones()
    routes = roads['routes']
    by_trunk = [[k for k, rt in enumerate(routes) if rt['trunk'] == j] for j in range(len(TRUNK_AZ))]
    rows = []      # route, t_mouth (at the mouth), v, lane offset
    for j in range(len(TRUNK_AZ)):
        v = rng.uniform(1.08, 1.22)
        t = T_SCHED0 + rng.uniform(0, 90)
        s_max = max(routes[k]['cum'][-1] for k in by_trunk[j])
        t_end = 4480.0 + (s_max - routes[by_trunk[j][0]]['s_mouth']) / v * 24.0 + 24.0
        while t < t_end:
            u = rng.random()
            size = int(rng.integers(1, 4)) if u < 0.14 else (int(rng.integers(4, 10)) if u < 0.58 else int(rng.integers(10, 25)))
            ab = 1 if size <= 2 else int(rng.choice([1, 2, 2, 2, 3], p=[0.15, 0.25, 0.25, 0.2, 0.15]))
            nrow = int(math.ceil(size / ab))
            drow = rng.uniform(1.15, 1.55)
            k = int(rng.choice(by_trunk[j]))
            vg = v * rng.uniform(0.985, 1.015)
            for m in range(size):
                rw = m // ab
                n_in_row = ab if rw < nrow - 1 else size - ab * (nrow - 1)
                ln = (m % ab) - (n_in_row - 1) / 2.0
                lane = ln * rng.uniform(0.80, 1.02) + rng.normal(0, 0.12)
                tm = t + rw * drow / vg * 24.0 + rng.normal(0, 4.0)
                rows.append((k, tm, vg * rng.uniform(0.99, 1.01), lane, j))
            t += ((nrow - 1) * drow + rng.exponential(13.0) + 2.5) / v * 24.0
    rows = np.array(rows)
    order = np.argsort(rows[:, 1])
    rows = rows[order]
    n = rows.shape[0]
    P = dict(route=rows[:, 0].astype(np.int64), tm=rows[:, 1], v=rows[:, 2], lane=rows[:, 3],
             trunk=rows[:, 4].astype(np.int64))
    P.update(_looks(n, rng))
    # places in the ring round the stones, in order of arrival
    slots = _p1_slots(rng, S)
    arrive = np.nonzero(P['tm'] < T_FREEZE)[0]
    free = np.ones(len(slots), bool)
    slot_of = -np.ones(n, np.int64)
    for i in arrive:
        if not free.any():
            break
        am = TRUNK_AZ[P['trunk'][i]]
        dth = np.abs(_wrap(slots[:, 3] - am))
        cost = 7.5 * slots[:, 4] + dth * R_WALK + rng.uniform(0, 0.6, len(slots))
        cost[~free] = 1e9
        q = int(np.argmin(cost))
        slot_of[i] = q
        free[q] = False
    P['slot'] = slot_of
    P['slots'] = slots
    # the settling walk: in along the road to the walkway, round it, then in to the place
    P['s_walk_mouth'] = np.array([routes[k]['s_mouth'] for k in P['route']])
    return P


def _p1_positions(P, roads, t):
    """Positions, headings, walking flags and gait phases of every P1 walker at t (vectorised)."""
    routes = roads['routes']
    n = len(P['tm'])
    x = np.zeros(n)
    y = np.zeros(n)
    hx = np.zeros(n)
    hy = np.zeros(n)
    walk = np.zeros(n)
    dist = np.zeros(n)
    tt = min(t, T_FREEZE)
    # --- still on the road: s from the gap outward (the mouth is at s_mouth)
    sm = P['s_walk_mouth']
    s = sm + P['v'] * (P['tm'] - tt) / 24.0
    on_road = s > sm
    for k in range(len(routes)):
        m = on_road & (P['route'] == k)
        if not m.any():
            continue
        p, tg = route_at(routes[k], s[m])
        nx, ny = -tg[:, 1], tg[:, 0]
        wl = P['wander'][m] * np.sin(2 * np.pi * tt / P['wper'][m] + P['wph'][m])
        x[m] = p[:, 0] + nx * (P['lane'][m] + wl)
        y[m] = p[:, 1] + ny * (P['lane'][m] + wl)
        hx[m], hy[m] = -tg[:, 0], -tg[:, 1]          # walking inward
        walk[m] = 1.0
        dist[m] = -s[m]
    # past the freeze: the ones still out on the road stop where they are (and stand facing in along it)
    if t > T_FREEZE:
        stop = on_road
        walk[stop] = 1.0 - _sm((t - T_FREEZE) / 20.0)
    # --- the settling walk to the place: u = distance walked past the mouth
    st = ~on_road
    has = st & (P['slot'] >= 0)
    if has.any():
        idx = np.nonzero(has)[0]
        u = P['v'][idx] * (t - P['tm'][idx]) / 24.0
        sl = P['slots'][P['slot'][idx]]
        am = TRUNK_AZ[P['trunk'][idx]]
        sa = sl[:, 3]
        dth = _wrap(sa - am)
        lane = P['lane'][idx]
        rw = R_WALK + lane * 0.5
        arc = np.abs(dth) * rw
        inn = np.maximum(rw - sl[:, 2], 0.0)
        Lm = R_MOUTH - rw                                   # from the mouth in to the walkway
        Ltot = Lm + arc + inn
        uu = np.clip(u, 0.0, Ltot)
        # smooth blend: first in from the mouth, then round, then in; the corners rounded by the smoothsteps
        a_arc = _sm((uu - Lm * 0.3) / (Lm + arc + 0.6 - Lm * 0.3 + 1e-6))
        az = am + dth * a_arc
        r_in = R_MOUTH - (R_MOUTH - rw) * _sm(uu / (Lm + 0.8))
        r = r_in - inn * _sm((uu - (Lm + arc - 0.8)) / (inn + 0.8 + 1e-6))
        xx = r * np.cos(az)
        yy = r * np.sin(az)
        # the last metre eases into the place
        done = u >= Ltot
        xx[done] = sl[done, 0]
        yy[done] = sl[done, 1]
        # blend the eased path end onto the exact slot
        e = _sm((uu - (Ltot - 1.0)) / 1.0)
        xx = xx * (1 - e) + sl[:, 0] * e
        yy = yy * (1 - e) + sl[:, 1] * e
        x[idx] = xx
        y[idx] = yy
        # heading: along the path while walking, then turning to face the centre
        u2 = np.clip(uu + 0.25, 0.0, Ltot)
        a2 = am + dth * _sm((u2 - Lm * 0.3) / (Lm + arc + 0.6 - Lm * 0.3 + 1e-6))
        r2 = (R_MOUTH - (R_MOUTH - rw) * _sm(u2 / (Lm + 0.8))) - inn * _sm((u2 - (Lm + arc - 0.8)) / (inn + 0.8 + 1e-6))
        vx_, vy_ = r2 * np.cos(a2) - xx, r2 * np.sin(a2) - yy
        vn = np.hypot(vx_, vy_) + 1e-9
        cx_, cy_ = -sl[:, 0], -sl[:, 1]
        cn = np.hypot(cx_, cy_) + 1e-9
        tl = (t - P['tm'][idx]) - Ltot / P['v'][idx] * 24.0      # frames since arriving at the place
        k = _sm((tl + 10.0) / 34.0)
        fx = (vx_ / vn) * (1 - k) + (cx_ / cn) * k
        fy = (vy_ / vn) * (1 - k) + (cy_ / cn) * k
        fn = np.hypot(fx, fy) + 1e-9
        hx[idx], hy[idx] = fx / fn, fy / fn
        walk[idx] = 1.0 - _sm((tl + 14.0) / 16.0)
        dist[idx] = uu
    # past the mouth with no place (none: every arrival before the freeze has one); keep them at the mouth
    nos = st & (P['slot'] < 0)
    if nos.any():
        for k in range(len(routes)):
            m = nos & (P['route'] == k)
            if m.any():
                p, tg = route_at(routes[k], np.full(m.sum(), sm[m][0]))
                x[m], y[m] = p[:, 0], p[:, 1]
                hx[m], hy[m] = -tg[:, 0], -tg[:, 1]
    phase = dist / 0.36 * math.pi
    return x, y, np.arctan2(hy, hx), walk, phase


# =============================================================== P3: bar 70 ===

def _build_p3(roads):
    """After the white: the gathering has closed in round the fire (ranks from P3_R0 out past the stones), and the
    rest stand in queues along the roads. Everyone's torch is spent."""
    rng = np.random.default_rng(CROWD_SEED + 3)
    S = SC.stones()
    pts = []
    r = P3_R0
    k = 0
    while r < P3_RMAX:
        n = int(2 * np.pi * r / P3_SP)
        az0 = rng.uniform(0, 2 * np.pi)
        az = az0 + 2 * np.pi * np.arange(n) / n + rng.normal(0, 0.10 / r, n)
        rr = r + rng.normal(0, 0.07, n)
        x, y = rr * np.cos(az), rr * np.sin(az)
        keep = _stone_clear(x, y, S, 0.45)
        for q in np.nonzero(keep)[0]:
            pts.append((x[q], y[q], rr[q], az[q] % (2 * np.pi), -1, 0.0, 0.0))
        r += P3_DR + rng.uniform(-0.05, 0.05)
        k += 1
    # the queues along the roads (two files) from just outside the ring to Q_RMAX
    for rt_i, rt in enumerate(roads['routes']):
        if rt_i != min(i for i, q in enumerate(roads['routes']) if q['trunk'] == rt['trunk']):
            continue                    # one queue per trunk
        r_all = np.hypot(rt['pts'][:, 0], rt['pts'][:, 1])
        s0 = float(np.interp(P3_RMAX + 0.9, r_all[:60], rt['cum'][:60]))
        s1 = float(np.interp(Q_RMAX, r_all, rt['cum']))
        s = s0
        while s < s1:
            for ln in (-0.42, 0.42):
                ss = s + rng.normal(0, 0.12)
                p, tg = route_at(rt, np.array([ss]))
                lane = ln + rng.normal(0, 0.07)
                x = p[0, 0] - tg[0, 1] * lane
                y = p[0, 1] + tg[0, 0] * lane
                pts.append((x, y, float(np.hypot(x, y)), math.atan2(y, x) % (2 * np.pi), rt_i, ss, lane))
            s += rng.uniform(1.15, 1.45)
    pts = np.array(pts)
    n = len(pts)
    P = dict(x=pts[:, 0], y=pts[:, 1], r=pts[:, 2], az=pts[:, 3], route=pts[:, 4].astype(np.int64),
             s=pts[:, 5], lane=pts[:, 6])
    P.update(_looks(n, rng))
    # the wave: each catches from the lit torch in front of it
    ring = P['route'] < 0
    tw = np.empty(n)
    tw[ring] = T_W0 + (P['r'][ring] - P3_R0) / WAVE_V
    rq = ~ring
    s_edge = np.array([float(np.interp(P3_RMAX, np.hypot(roads['routes'][k]['pts'][:60, 0], roads['routes'][k]['pts'][:60, 1]),
                                       roads['routes'][k]['cum'][:60])) for k in P['route'][rq]])
    tw[rq] = T_W0 + (P3_RMAX - P3_R0) / WAVE_V + (P['s'][rq] - s_edge) / WAVE_V
    P['tw'] = tw + rng.normal(0, 2.2, n)
    P['v'] = rng.uniform(1.05, 1.32, n)
    # clear the council's lanes: in the first ranks, whoever stands in a council bearer's line walks no slower
    for i_ in range(SC.NEM):
        psi = SC.FIG_PSI[i_]
        near = ring & (P['r'] < 6.5) & (np.abs(_wrap(P['az'] - psi)) * P['r'] < 0.7)
        P['v'][near] = np.maximum(P['v'][near], SC.FIG_WALK_V[i_] + 0.06)
    # the walk out: round the stones, then drifting toward the nearest road
    P['az_road'] = np.array([TRUNK_AZ[int(np.argmin(np.abs(_wrap(TRUNK_AZ - a))))] for a in P['az']])
    P['dstone'] = np.zeros(n)
    Sx, Sy = S[:, G.S_X], S[:, G.S_Y]
    Saz = np.arctan2(Sy, Sx)
    Sw = np.maximum(S[:, G.S_HX], S[:, G.S_HY]) + 0.55
    for i_ in np.nonzero(ring & (P['r'] < 7.9))[0]:
        dz = _wrap(P['az'][i_] - Saz)
        j = int(np.argmin(np.abs(dz)))
        rs = np.hypot(Sx[j], Sy[j])
        clear = Sw[j] / rs
        if abs(dz[j]) < clear:
            P['dstone'][i_] = (clear - abs(dz[j])) * (1.0 if dz[j] >= 0 else -1.0)
    return P


def _walk_dist(tau, v):
    """Distance walked tau frames after setting off (eased in over 12 frames)."""
    tau = np.maximum(tau, 0.0)
    return v / 24.0 * np.where(tau < 12.0, tau * tau / 24.0, tau - 6.0)


def _p3_positions(P, roads, t):
    tau = t - P['tw']
    d = _walk_dist(tau - 20.0, P['v'])
    n = len(P['x'])
    x = P['x'].copy()
    y = P['y'].copy()
    ring = P['route'] < 0
    # ring: radially out, round the stones, drifting toward the road's azimuth beyond the stones
    r = P['r'][ring] + d[ring]
    az = P['az'][ring]
    rs = P['r'][ring]
    az = az + P['dstone'][ring] * _sm((r - 5.8) / 1.2) * (1.0 - _sm((r - 8.6) / 1.4))
    pull = _sm((r - 9.0) / 9.0) * 0.55
    az = az + _wrap(P['az_road'][ring] - az) * pull * np.clip((r - rs) / 10.0, 0.0, 1.0)
    x[ring] = r * np.cos(az)
    y[ring] = r * np.sin(az)
    # facing: in toward the fire; they turn (the torch side leading) and walk out
    face_in = np.arctan2(-P['y'], -P['x'])
    heading = face_in.copy()
    rq = ~ring
    for k in np.unique(P['route'][rq]):
        m = rq & (P['route'] == k)
        s = P['s'][m] + d[m]
        p, tg = route_at(roads['routes'][int(k)], s)
        ln = P['lane'][m]
        x[m] = p[:, 0] - tg[:, 1] * ln
        y[m] = p[:, 1] + tg[:, 0] * ln
        heading[m] = np.arctan2(-tg[:, 1], -tg[:, 0])       # standing: facing in along the road
    turn = _smr((tau - 10.0) / 18.0)
    ang = heading + P['side'] * np.pi * turn
    walk = _sm((tau - 18.0) / 12.0)
    phase = d / 0.36 * math.pi
    return x, y, ang, walk, phase, tau


# ================================================================ the poses ===

def _shoulder(hs, ws, side, kneel, lean):
    """= scene3.shoulder_local, vectorised."""
    zsh = 1.33 * hs - 0.43 * hs * kneel
    zhip = 0.62 * zsh
    z = zsh - 0.035 * hs
    lz = lean * _sm((z - (zhip - 0.20)) / 0.45)
    rz = z - zhip
    xw = np.sin(lz) * rz
    zw = np.cos(lz) * rz + zhip
    return np.stack([xw, side * 0.200 * ws, zw], -1)


def _two_bone(S, Hd, L1, L2, pole):
    d = Hd - S
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    dc = np.clip(dist, 0.08, L1 + L2 - 1e-3)
    u = d / np.maximum(dist, 1e-6)
    Hh = S + u * dc
    a = (L1 * L1 - L2 * L2 + dc * dc) / (2 * dc)
    hgt = np.sqrt(np.maximum(L1 * L1 - a * a, 0.0))
    p = pole - u * np.sum(pole * u, axis=-1, keepdims=True)
    p = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-9)
    return S + u * a + p * hgt, Hh


def _rot(ang, v):
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c, v[:, 2]], -1)


def _torch_pose_p1(L, t, walk, tdown):
    """Torch hand (local) and axis for P1/P2: carried high while walking, held upright before them standing; in P2
    lowered to the earth and put out, then held up spent."""
    n = len(walk)
    hs, side = L['hs'], L['side']
    b = 0.004 * np.sin(2 * np.pi * t / 96.0 + L['bph'])
    stand_h = np.stack([np.full(n, 0.38) + L['reach'], side * (0.21 + 0.5 * L['reach']), 1.10 * hs + 2 * b], -1)
    stand_a = _unit(np.stack([np.full(n, 0.16), side * 0.05, np.ones(n)], -1))
    walk_h = np.stack([np.full(n, 0.40) + L['reach'], side * (0.23 + 0.5 * L['reach']), 1.19 * hs], -1)
    walk_a = _unit(np.stack([np.full(n, 0.24), side * 0.06, np.ones(n)], -1))
    w = walk[:, None]
    H = stand_h * (1 - w) + walk_h * w
    A = _unit(stand_a * (1 - w) + walk_a * w)
    lit = np.ones(n)
    if tdown is not None:
        u = _smr(_ramp(t, tdown, tdown + 22.0))[:, None]
        back = _smr(_ramp(t, tdown + 40.0, tdown + 64.0))[:, None]
        dn_h = np.stack([np.full(n, 0.42), side * 0.11, 0.40 * hs], -1)
        dn_a = _unit(np.stack([np.full(n, 0.86), -side * 0.04, np.full(n, -0.52)], -1))
        bk_h = np.stack([np.full(n, 0.26), side * 0.10, 0.92 * hs], -1)
        bk_a = _unit(np.stack([np.full(n, 0.08), -side * 0.03, np.ones(n)], -1))
        H = H * (1 - u) + dn_h * u
        A = _unit(A * (1 - u) + dn_a * u)
        H = H * (1 - back) + bk_h * back
        A = _unit(A * (1 - back) + bk_a * back)
        lit = 1.0 - _sm(_ramp(t, tdown + 16.0, tdown + 30.0))
    return H, A, lit


def _torch_pose_p3(L, tau):
    n = len(tau)
    hs, side = L['hs'], L['side']
    sp_h = np.stack([np.full(n, 0.26), side * 0.10, 0.92 * hs], -1)
    sp_a = _unit(np.stack([np.full(n, 0.08), -side * 0.03, np.ones(n)], -1))
    dip_h = np.stack([np.full(n, 0.50), side * 0.07, 1.10 * hs], -1)
    dip_a = _unit(np.stack([np.full(n, 0.86), -side * 0.03, np.full(n, 0.40)], -1))
    up_h = np.stack([np.full(n, 0.40) + L['reach'], side * (0.23 + 0.5 * L['reach']), 1.20 * hs], -1)
    up_a = _unit(np.stack([np.full(n, 0.20), side * 0.05, np.ones(n)], -1))
    u = _smr((tau + 12.0) / 10.0)
    v = _smr((tau - 4.0) / 10.0)
    w = (u * (1 - v))[:, None]
    H = sp_h * (1 - w) + dip_h * w
    A = _unit(sp_a * (1 - w) + dip_a * w)
    up = (_sm((tau - 6.0) / 12.0))[:, None]
    H = H * (1 - up) + up_h * up
    A = _unit(A * (1 - up) + up_a * up)
    lean = np.radians(15.0) * w[:, 0]
    lit = _sm((tau + 3.0) / 8.0)
    return H, A, lit, lean


def _rows(L, x, y, ang, H, A, lit, walk, phase, lean, t):
    """Figure rows in geom3's layout (F_N columns), vectorised."""
    n = len(x)
    F = np.zeros((n, G.F_N))
    hs, ws, side = L['hs'], L['ws'], L['side']
    kneel = np.zeros(n)
    F[:, G.F_X], F[:, G.F_Y], F[:, G.F_ANG] = x, y, ang
    F[:, G.F_C], F[:, G.F_S] = np.cos(ang), np.sin(ang)
    F[:, G.F_HS], F[:, G.F_WS] = hs, ws
    F[:, G.F_TYPE] = L['typ']
    F[:, G.F_SEED] = L['seed']
    F[:, G.F_R:G.F_B + 1] = L['c1']
    F[:, G.F_R2:G.F_B2 + 1] = L['c2']
    F[:, G.F_SIDE] = side
    Sh = _shoulder(hs, ws, side, kneel, lean)
    pole = np.stack([np.full(n, -0.35), side * 1.0, np.full(n, -0.9)], -1)
    E, Hh = _two_bone(Sh, H, 0.31, 0.30, pole)
    F[:, G.F_SHX:G.F_SHZ + 1] = Sh
    F[:, G.F_EX:G.F_EZ + 1] = E
    F[:, G.F_HX:G.F_HZ + 1] = Hh
    F[:, G.F_TX:G.F_TZ + 1] = A
    F[:, G.F_BOW] = L['bow'] + np.radians(1.2) * np.sin(2 * np.pi * t / 131.0 + L['bph'])
    F[:, G.F_TLIT] = np.maximum(lit, L['ember'])
    F[:, G.F_BREATH] = 0.006 * np.sin(2 * np.pi * t / 96.0 + L['bph'])
    F[:, G.F_CAPE] = L['cape'] * hs
    F[:, G.F_TRAIN] = L['train']
    F[:, G.F_WEAVE] = L['weave']
    F[:, G.F_SHEENK] = L['sheenk']
    F[:, G.F_TORCH] = 1.0
    F[:, G.F_KNEEL] = kneel
    F[:, G.F_LEAN] = lean
    F[:, G.F_SWAY] = 0.018 * walk * np.sin(phase)
    F[:, G.F_HEM] = 0.05 * walk
    F[:, G.F_PHASE] = phase
    # AABB (local corners + the arm and torch), world
    xa = -(0.30 + L['train']) * ws
    xb = 0.34 * ws
    ya = 0.40 * ws
    zb = 1.97 * hs
    loc = np.stack([np.stack([xa, -ya, np.zeros(n)], -1), np.stack([xb, ya, zb], -1),
                    np.stack([xa, ya, zb], -1), np.stack([xb, -ya, np.zeros(n)], -1),
                    Hh + 0.60 * A, Hh - 0.30 * A, E, Sh,
                    np.stack([0.9 * np.sin(lean) + 0.25, np.zeros(n), 1.7 * hs], -1)], 1)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    wx = x[:, None] + loc[..., 0] * c - loc[..., 1] * s
    wy = y[:, None] + loc[..., 0] * s + loc[..., 1] * c
    wz = loc[..., 2]
    m = 0.10
    F[:, G.F_BB] = wx.min(1) - m
    F[:, G.F_BB + 1] = wx.max(1) + m
    F[:, G.F_BB + 2] = wy.min(1) - m
    F[:, G.F_BB + 3] = wy.max(1) + m
    F[:, G.F_BB + 4] = -0.02
    F[:, G.F_BB + 5] = wz.max(1) + m
    # torch tops (world) and axes
    top = np.stack([x + (Hh[:, 0] + 0.44 * A[:, 0]) * c[:, 0] - (Hh[:, 1] + 0.44 * A[:, 1]) * s[:, 0],
                    y + (Hh[:, 0] + 0.44 * A[:, 0]) * s[:, 0] + (Hh[:, 1] + 0.44 * A[:, 1]) * c[:, 0],
                    Hh[:, 2] + 0.44 * A[:, 2]], -1)
    axw = _rot(ang, A)
    return F, top, axw


# ================================================================== world ===

_W = {}


def world():
    if not _W:
        roads = _build_roads()
        _W['roads'] = roads
        _W['p1'] = _build_p1(roads)
        _W['p3'] = _build_p3(roads)
        _W['n3'] = nbcore.make_noise3(64, 7)
        _W['S'] = SC.stones()
    return _W


def plate_of(t):
    if t < SC.P2[0]:
        return 1
    if t < SC.P3[0] - 40:
        return 2
    return 3


def _people(t):
    """(L, x, y, ang, H, A, lit, walk, phase, lean) of everyone in play at t."""
    W = world()
    pl = plate_of(t)
    if pl in (1, 2):
        P = W['p1']
        x, y, ang, walk, phase = _p1_positions(P, W['roads'], t)
        tdown = None
        if t >= SC.TORCH_DOWN - 6:
            r = np.hypot(x, y)
            rng = np.random.default_rng(CROWD_SEED + 7)
            tdown = SC.TORCH_DOWN + 2.0 + 0.9 * np.clip(r - R_FRONT, 0, 40) + rng.uniform(-3, 3, len(x))
        H, A, lit = _torch_pose_p1(P, t, walk, tdown)
        lean = np.zeros(len(x))
        return P, x, y, ang, H, A, lit, walk, phase, lean
    P = W['p3']
    x, y, ang, walk, phase, tau = _p3_positions(P, W['roads'], t)
    H, A, lit, lean = _torch_pose_p3(P, tau)
    return P, x, y, ang, H, A, lit, walk, phase, lean


def state(t, prev=True):
    """The crowd at t: CF (figure rows), FL (torch flames, flame3 layout), TL (torch lights, shade3 LT layout),
    plus the torch tops for the lights' neighbour lists."""
    t = float(t)
    L, x, y, ang, H, A, lit, walk, phase, lean = _people(t)
    F, top, axw = _rows(L, x, y, ang, H, A, lit, walk, phase, lean, t)
    # velocity of the torch tops (for the flames' trail)
    if prev:
        L0, x0, y0, a0, H0, A0, l0, w0, p0, le0 = _people(t - 1.0)
        _, top0, _ = _rows(L0, x0, y0, a0, H0, A0, l0, w0, p0, le0, t - 1.0)
        vel = (top - top0) * 24.0
    else:
        vel = np.zeros_like(top)
    W = world()
    n3 = W['n3']
    wd = wind()
    base = top - 0.03 * axw / (np.linalg.norm(axw, axis=1, keepdims=True) + 1e-9)
    lean_xy = wd[None, :] - 0.24 * vel[:, :2]
    ln = np.linalg.norm(lean_xy, axis=1, keepdims=True)
    lean_xy = np.where(ln > 1.6, lean_xy * 1.6 / np.maximum(ln, 1e-9), lean_xy)
    FLr = np.zeros((len(x), FL3.FL_N))
    FLr[:, 0:3] = base
    FLr[:, 3] = L['hf']
    FLr[:, 4] = 0.088
    FLr[:, 5:7] = lean_xy
    FLr[:, 7] = FLAME_I
    FLr[:, 8] = L['fseed']
    FLr[:, 9] = lit
    fl = np.array([FL3.torch_flicker(float(L['fseed'][i]), t, n3) for i in range(len(x))]) / 0.98 \
        if len(x) else np.zeros(0)
    I = TORCH_I * lit * (0.55 + 0.45 * fl)
    TL = np.zeros((len(x), 8))
    TL[:, 0] = base[:, 0] + lean_xy[:, 0] * 0.08
    TL[:, 1] = base[:, 1] + lean_xy[:, 1] * 0.08
    TL[:, 2] = base[:, 2] + 0.22
    TL[:, 3] = 0.09
    TL[:, 4:7] = I[:, None] * FIRE_HOT[None, :]
    TL[:, 7] = -5.0                                     # owner: not one of accord3's figures
    return dict(t=t, plate=plate_of(t), CF=F, FL=FLr, TL=TL, lit=lit, x=x, y=y, top=top, seed=L['fseed'])


def lit_fraction(cs):
    """How much of the crowd's torchlight is burning (scales shade3's P_CROWD rim on the council)."""
    if cs is None or len(cs['lit']) == 0:
        return 0.0
    near = np.hypot(cs['x'], cs['y']) < 20.0
    return float(np.mean(cs['lit'][near])) if near.any() else 0.0


# ================================================================== her walk ===

def her_path():
    """Her walk in: up her road's aisle through the crowd, between the stones, across the open floor to her place."""
    a = TRUNK_AZ[HER_TRUNK]
    psi = SC.PSI_HER
    r_end = SC.FIG_R[SC.HER]
    pts = [(r * math.cos(a), r * math.sin(a)) for r in np.linspace(17.5, 7.2, 42)]
    for u in np.linspace(0.0, 1.0, 40)[1:]:
        r = 7.2 + (r_end - 7.2) * u
        az = a + _wrap(psi - a) * (u ** 1.6)
        pts.append((r * math.cos(az), r * math.sin(az)))
    P = np.array(pts)
    return P, _arclen(P)


def her_walkin(t):
    """For t < HER_ARRIVE + 26: her position, facing, walk (0..1) and gait phase; None after (she is in her place)."""
    if t >= HER_ARRIVE + 26.0:
        return None
    P, cum = her_path()
    Ltot = cum[-1]
    u = Ltot - HER_V * (HER_ARRIVE - t) / 24.0
    ue = float(np.clip(u, 0.0, Ltot))
    x = float(np.interp(ue, cum, P[:, 0]))
    y = float(np.interp(ue, cum, P[:, 1]))
    x2 = float(np.interp(min(ue + 0.3, Ltot), cum, P[:, 0]))
    y2 = float(np.interp(min(ue + 0.3, Ltot), cum, P[:, 1]))
    ang_w = math.atan2(y2 - y, x2 - x) if ue < Ltot - 0.05 else SC.PSI_HER + math.pi
    k = float(_sm((t - (HER_ARRIVE - 12.0)) / 34.0))
    ang = ang_w + _wrap(SC.PSI_HER + math.pi - ang_w) * k
    walk = float(1.0 - _sm((t - (HER_ARRIVE - 16.0)) / 16.0))
    return dict(pos=np.array([x, y]), ang=ang, walk=walk, phase=ue / 0.30 * math.pi)


# ============================================================ the land's roads ===
MOON_MOOR = 4.0                      # the moon on the open moor, x shade3's (the far dark must read as land)
SKY_MOOR = 0.6
PMC_R, PMC_CELL = 236.0, 0.5         # coarse worn-road mask
PMF_R, PMF_CELL = 30.0, 0.04         # fine worn-road mask near the ring


@njit(parallel=True, **FM)
def _raster_roads(M, x0, cell, SEG, nseg):
    """Worn tracks: a trodden band with two or three foot-lanes in it and a ragged edge (max over segments).
    SEG rows: ax, ay, bx, by, half-width, strength."""
    n = M.shape[0]
    for row in prange(n):
        yc = x0 + (row + 0.5) * cell
        for k in range(nseg):
            ax, ay, bx, by, hw, st = SEG[k, 0], SEG[k, 1], SEG[k, 2], SEG[k, 3], SEG[k, 4], SEG[k, 5]
            ext = hw * 1.6 + 0.2
            if yc < min(ay, by) - ext or yc > max(ay, by) + ext:
                continue
            c0 = max(int((min(ax, bx) - ext - x0) / cell), 0)
            c1 = min(int((max(ax, bx) + ext - x0) / cell) + 1, n)
            ux = bx - ax
            uy = by - ay
            L2 = ux * ux + uy * uy + 1e-12
            for col in range(c0, c1):
                xc = x0 + (col + 0.5) * cell
                tq = clamp(((xc - ax) * ux + (yc - ay) * uy) / L2, 0.0, 1.0)
                qx = xc - (ax + ux * tq)
                qy = yc - (ay + uy * tq)
                # signed offset across the track (for the foot-lanes)
                off = (qx * (-uy) + qy * ux) / math.sqrt(L2)
                d = math.sqrt(qx * qx + qy * qy)
                edge = hw * (1.0 + 0.35 * (vnoise2(xc * 0.9, yc * 0.9, 311) - 0.5) + 0.15 * (vnoise2(xc * 4.0, yc * 4.0, 312) - 0.5))
                band = sstep(edge, edge * 0.55, d)
                lanes = 0.5 + 0.5 * math.cos(off / (hw * 0.62) * math.pi * 1.5)
                v = st * band * (0.62 + 0.38 * lanes * sstep(hw * 0.95, hw * 0.3, d))
                if v > M[row, col]:
                    M[row, col] = v


def _road_masks(roads):
    segs = []
    for j, P in enumerate(roads['trunks']):
        for i in range(len(P) - 1):
            r = float(np.hypot(*P[i]))
            segs.append((P[i, 0], P[i, 1], P[i + 1, 0], P[i + 1, 1], 1.55 * (0.8 + 0.2 * min(r / 30.0, 1.0)), 1.0))
    for rt in roads['routes']:
        B = rt['pts'][rt['n_trunk'] - 1:]
        for i in range(len(B) - 1):
            segs.append((B[i, 0], B[i, 1], B[i + 1, 0], B[i + 1, 1], 1.05, 0.8))
    SEG = np.array(segs, np.float64)
    nc = int(2 * PMC_R / PMC_CELL)
    MC = np.zeros((nc, nc), np.float32)
    _raster_roads(MC, -PMC_R, PMC_CELL, SEG, SEG.shape[0])
    near = (np.hypot(SEG[:, 0], SEG[:, 1]) < PMF_R + 3.0)
    SF = np.ascontiguousarray(SEG[near])
    nf = int(2 * PMF_R / PMF_CELL)
    MF = np.zeros((nf, nf), np.float32)
    _raster_roads(MF, -PMF_R, PMF_CELL, SF, SF.shape[0])
    return MC, MF


def masks():
    W = world()
    if 'pm' not in W:
        W['pm'] = _road_masks(W['roads'])
    return W['pm']


# ============================================================ the torchlight ===
R_NEAR0, R_NEAR1 = 3.2, 4.2          # near field (fine, with every bearer's shadow) fades out between these
R_FAR = 16.0                         # the far field's reach (coarse, the stones' and council's shadows)


@njit(parallel=True, **FM)
def _splat(Gd, x0, cell, TL, order, tys, OCA, OFF, rmax, r0, r1, near, ch, DN, dx0, dcell, sig):
    """Torchlight on the ground (irradiance on the level ground, ndl^0.7 as shade3's ground), rows in parallel.
    near=True: only within r0..r1 of each torch (fading out), with the soft shadows of that torch's own
    occluders OCA[OFF[k]:OFF[k+1]]; near=False: only beyond (fading in), same occluder lists."""
    n = Gd.shape[0]
    m = Gd.shape[1]
    for row in prange(n):
        yc = x0 + (row + 0.5) * cell
        lo = np.searchsorted(tys, yc - rmax)
        hi = np.searchsorted(tys, yc + rmax)
        for q in range(lo, hi):
            k = order[q]
            I = TL[k, ch]
            if I <= 1e-5:
                continue
            lx = TL[k, 0]
            ly = TL[k, 1]
            lz = max(TL[k, 2], 0.05)
            rl = TL[k, 3]
            dy = yc - ly
            sp2 = rmax * rmax - dy * dy
            if sp2 <= 0.0:
                continue
            sp = math.sqrt(sp2)
            c0 = max(int((lx - sp - x0) / cell), 0)
            c1 = min(int((lx + sp - x0) / cell) + 1, m)
            o0 = OFF[k]
            o1 = OFF[k + 1]
            for col in range(c0, c1):
                xc = x0 + (col + 0.5) * cell
                dx = xc - lx
                dxy2 = dx * dx + dy * dy
                dxy = math.sqrt(dxy2)
                wn = sstep(r1, r0, dxy)
                w = wn if near else 1.0 - wn
                if w <= 0.0:
                    continue
                d2 = dxy2 + lz * lz
                ndl = lz / math.sqrt(d2)
                f = ndl ** 0.7 / (d2 + rl * rl + 0.02) * w
                if o1 > o0:
                    f *= G.soft_vis(xc, yc, 0.0, lx, ly, lz, rl, OCA[o0:o1], o1 - o0, -1.0)
                if sig > 0.0 and dxy > 0.6:
                    # bodies in the way: the crowd's density along the path (three samples), its bodies ~0.45 m wide
                    dn = (grid_sample(DN, dx0, dx0, dcell, lx + dx * 0.25, ly + dy * 0.25)
                          + grid_sample(DN, dx0, dx0, dcell, lx + dx * 0.5, ly + dy * 0.5)
                          + grid_sample(DN, dx0, dx0, dcell, lx + dx * 0.75, ly + dy * 0.75)) / 3.0
                    f *= math.exp(-sig * dn * (dxy - 0.6))
                Gd[row, col] += I * f


@njit(parallel=True, **FM)
def _upsample_add(Gf, x0f, cf, Gc, x0c, cc):
    n = Gf.shape[0]
    for row in prange(n):
        yc = x0f + (row + 0.5) * cf
        for col in range(Gf.shape[1]):
            xc = x0f + (col + 0.5) * cf
            Gf[row, col] += grid_sample(Gc, x0c, x0c, cc, xc, yc)


@njit(parallel=True, **FM)
def _contact(CA, MS, x0, cell, FB, order, fys, mdx, mdy, mdz):
    """Contact darkening at the crowd's feet (CA, multiplies) and the moon shadows of their bodies (MS, 0..1).
    FB rows: x, y, body radius, top z."""
    n = CA.shape[0]
    tl = mdz / max(math.sqrt(mdx * mdx + mdy * mdy), 1e-6)
    mx = mdx / max(math.sqrt(mdx * mdx + mdy * mdy), 1e-6)
    my = mdy / max(math.sqrt(mdx * mdx + mdy * mdy), 1e-6)
    for row in prange(n):
        yc = x0 + (row + 0.5) * cell
        lo = np.searchsorted(fys, yc - 3.2)
        hi = np.searchsorted(fys, yc + 3.2)
        for q in range(lo, hi):
            k = order[q]
            bx = FB[k, 0]
            by = FB[k, 1]
            rb = FB[k, 2]
            L = FB[k, 3] / tl
            # the box that holds the feet and the moon shadow (cast away from the moon)
            xa = min(bx, bx - mx * L) - rb - 0.8
            xb = max(bx, bx - mx * L) + rb + 0.8
            ya = min(by, by - my * L) - rb - 0.8
            yb = max(by, by - my * L) + rb + 0.8
            if yc < ya or yc > yb:
                continue
            c0 = max(int((xa - x0) / cell), 0)
            c1 = min(int((xb - x0) / cell) + 1, CA.shape[1])
            for col in range(c0, c1):
                xc = x0 + (col + 0.5) * cell
                dx = xc - bx
                dy = yc - by
                d = math.sqrt(dx * dx + dy * dy)
                if d < 0.9:
                    CA[row, col] *= 0.50 + 0.50 * sstep(0.55 * rb, 2.6 * rb, d)
                # moon shadow: how far the body's axis passes from this point's ray to the moon
                s = -(dx * mx + dy * my)
                if s > 0.0 and s < L + rb:
                    px_ = dx + mx * s
                    py_ = dy + my * s
                    off = math.sqrt(px_ * px_ + py_ * py_)
                    h = s * tl
                    taper = 1.0 - 0.45 * sstep(0.55 * FB[k, 3], FB[k, 3], h)
                    sh = sstep(rb * taper + 0.05 + 0.02 * s, rb * taper - 0.03, off)
                    if sh > MS[row, col]:
                        MS[row, col] = sh


def _visible_box(cam, Wd, Hd, margin):
    C, R, U, Fw = cam[0:3], cam[3:6], cam[6:9], cam[9:12]
    f, cx, cy = cam[12], cam[13], cam[14]
    pts = []
    for (x, y) in [(0, 0), (Wd, 0), (0, Hd), (Wd, Hd), (Wd / 2, 0), (Wd / 2, Hd), (0, Hd / 2), (Wd, Hd / 2)]:
        d = Fw + (x - cx) / f * R - (y - cy) / f * U
        d = d / np.linalg.norm(d)
        tt = -C[2] / d[2] if d[2] < -1e-6 else 400.0
        q = C + d * min(tt, 400.0)
        pts.append(q[:2])
    pts = np.array(pts)
    return pts[:, 0].min() - margin, pts[:, 0].max() + margin, pts[:, 1].min() - margin, pts[:, 1].max() + margin


def _occ_rows(cs, idx):
    """Body capsules (shade3/geom3 OC layout) of crowd figures idx."""
    CF = cs['CF']
    x, y = CF[idx, G.F_X], CF[idx, G.F_Y]
    hs, ws = CF[idx, G.F_HS], CF[idx, G.F_WS]
    O = np.zeros((len(idx), 8))
    O[:, 0], O[:, 1], O[:, 2] = x, y, 0.15
    O[:, 3], O[:, 4], O[:, 5] = x, y, 1.40 * hs
    O[:, 6] = 0.20 * ws
    O[:, 7] = -6.0
    return O


def ground_light(cs, cam, Wd, Hd, OC, PR):
    """The crowd's torchlight on the ground as shade3's igf (fine: near field with every bearer's soft shadow, plus
    the upsampled far field) and igc (coarse: the far field alone), the crowd's own ambient (EF), the contact
    darkening (CA) and moon shadows (MS) of their bodies. Fills the grid geometry into PR."""
    x0b, x1b, y0b, y1b = _visible_box(cam, Wd, Hd, 1.0)
    Hs = max(abs(x0b), abs(x1b), abs(y0b), abs(y1b)) + R_NEAR1 + 0.5
    Hs = min(Hs, 420.0)
    fpx = float(np.linalg.norm(cam[0:3] - np.array([0, 0, 0.3]))) / cam[12]
    cell = float(np.clip(fpx * 1.25, 0.012, 0.40))
    n = int(math.ceil(2 * Hs / cell))
    if n > 2400:
        cell = 2 * Hs / 2400
        n = 2400
    x0 = -Hs
    TL = cs['TL']
    CF = cs['CF']
    lit = TL[:, 4] > 1e-4
    # torches whose light can reach the visible ground
    tx, ty = TL[:, 0], TL[:, 1]
    inb = lit & (tx > x0b - R_FAR) & (tx < x1b + R_FAR) & (ty > y0b - R_FAR) & (ty < y1b + R_FAR)
    inb_near = inb & (tx > x0b - R_NEAR1) & (tx < x1b + R_NEAR1) & (ty > y0b - R_NEAR1) & (ty < y1b + R_NEAR1)
    # occluders per torch: its bearer and the bodies within 2.6 m (near); the stones and council (both)
    OCs = np.ascontiguousarray(OC[OC[:, 7] > -3.0]) if OC is not None and len(OC) else np.zeros((0, 8))
    body_i = np.nonzero(np.hypot(CF[:, G.F_X] - 0, CF[:, G.F_Y] - 0) >= 0)[0]
    BO = _occ_rows(cs, body_i)
    tree = cKDTree(np.stack([CF[:, G.F_X], CF[:, G.F_Y]], -1)) if len(CF) else None
    rows, off = [], [0]
    idx_n = np.nonzero(inb_near)[0]
    stat_xy = np.stack([OCs[:, 0], OCs[:, 1]], -1) if len(OCs) else np.zeros((0, 2))
    for k in range(len(TL)):
        if not inb_near[k]:
            off.append(off[-1])
            continue
        nb = tree.query_ball_point([tx[k], ty[k]], 2.6) if tree is not None else []
        r_ = [BO[j] for j in nb]
        if len(stat_xy):
            d = np.hypot(stat_xy[:, 0] - tx[k], stat_xy[:, 1] - ty[k])
            r_ += [OCs[j] for j in np.nonzero(d < R_NEAR1 + 2.5)[0]]
        rows += r_
        off.append(off[-1] + len(r_))
    OCA = np.array(rows, np.float64) if rows else np.zeros((1, 8))
    OFF = np.array(off, np.int64)
    order = np.argsort(ty).astype(np.int64)
    tys = np.ascontiguousarray(ty[order])
    TLn = np.ascontiguousarray(np.where(inb_near[:, None], TL, 0.0))
    # the crowd's density (people per m2, 1 m cells, softened) for the far field's transmittance
    Hc = min(Hs + R_FAR, 440.0)
    DN = np.zeros((int(math.ceil(2 * Hc)), int(math.ceil(2 * Hc))), np.float32)
    ix = np.floor(CF[:, G.F_X] + Hc).astype(np.int64)
    iy = np.floor(CF[:, G.F_Y] + Hc).astype(np.int64)
    ok = (ix >= 0) & (ix < DN.shape[1]) & (iy >= 0) & (iy < DN.shape[0])
    np.add.at(DN, (iy[ok], ix[ok]), 1.0)
    DN = cv2.GaussianBlur(DN, (0, 0), 1.1)
    Gf = np.zeros((n, n), np.float32)
    _splat(Gf, x0, cell, TLn, order, tys, OCA, OFF, R_NEAR1, R_NEAR0, R_NEAR1, True, 4, DN, -Hc, 1.0, 0.0)
    # far field (coarse): every torch in reach, the stones' and council's shadows only
    cc = float(np.clip(cell * 4.0, 0.25, 1.0))
    ncg = int(math.ceil(2 * Hc / cc))
    Gc = np.zeros((ncg, ncg), np.float32)
    rows, off = [], [0]
    for k in range(len(TL)):
        if not inb[k] or not len(stat_xy):
            off.append(off[-1])
            continue
        d = np.hypot(stat_xy[:, 0] - tx[k], stat_xy[:, 1] - ty[k])
        sel = np.nonzero(d < R_FAR + 3.0)[0]
        rows += [OCs[j] for j in sel]
        off.append(off[-1] + len(sel))
    OCF = np.array(rows, np.float64) if rows else np.zeros((1, 8))
    OFFF = np.array(off, np.int64)
    TLf = np.ascontiguousarray(np.where(inb[:, None], TL, 0.0))
    _splat(Gc, -Hc, cc, TLf, order, tys, OCF, OFFF, R_FAR, R_NEAR0, R_NEAR1, False, 4, DN, -Hc, 1.0, 0.45)
    EF = Gc.copy()
    _upsample_add(Gf, x0, cell, Gc, -Hc, cc)
    Gf += 1e-7                      # shade3 falls back to igc only where igf is exactly 0
    # contact darkening + moon shadows (half the fine resolution)
    cell2 = cell * 2.0
    n2 = int(math.ceil(2 * Hs / cell2))
    CA = np.ones((n2, n2), np.float32)
    MS = np.zeros((n2, n2), np.float32)
    fx, fy = CF[:, G.F_X], CF[:, G.F_Y]
    vis = (fx > x0b - 3) & (fx < x1b + 3) & (fy > y0b - 3) & (fy < y1b + 3)
    vi = np.nonzero(vis)[0]
    if len(vi):
        FB = np.stack([fx[vi], fy[vi], 0.21 * CF[vi, G.F_WS], 1.62 * CF[vi, G.F_HS]], -1)
        o2 = np.argsort(FB[:, 1]).astype(np.int64)
        _contact(CA, MS, x0, cell2, FB, o2, np.ascontiguousarray(FB[o2, 1]), PR[SH.P_MDX], PR[SH.P_MDY], PR[SH.P_MDZ])
    return dict(igf=Gf, igc=Gc, x0f=x0, cf=cell, x0c=-Hc, cc=cc, EF=EF, CA=CA, MS=MS, cell2=cell2)


# ================================================================== the land ===

@njit(**FM)
def _relief(x, y):
    """Gentle folds of the dark land (m): long swells and shallow hollows; level round the ring."""
    r = math.sqrt(x * x + y * y)
    h = 2.4 * fbm2(x * 0.0060 + 3.1, y * 0.0060 - 1.7, 71, 3, 2.1, 0.5, 0.0) \
        + 0.8 * fbm2(x * 0.021, y * 0.021 + 5.0, 72, 3, 2.2, 0.5, 0.0)
    return h * sstep(10.0, 34.0, r)


@njit(parallel=True, **FM)
def _land_pass(rgb, depth, oid, cam, PR, CA, MS, x0, cell2, PMC, PMF):
    """Ground pixels only: the crowd's contact darkening and moon shadows everywhere; beyond the stones, the land's
    roads, heather and folds, applied as a ratio to shade3's ground (so there is no seam with it)."""
    Hd = rgb.shape[0]
    Wd = rgb.shape[1]
    f = cam[12]
    mdx = PR[SH.P_MDX]
    mdy = PR[SH.P_MDY]
    mdz = PR[SH.P_MDZ]
    mi = PR[SH.P_MI]
    for y in prange(Hd):
        for x in range(Wd):
            if oid[y, x] != 0:
                continue
            tq = depth[y, x]
            if tq > 1e5:
                continue
            sx = (x + 0.5 - cam[13]) / f
            sy = -(y + 0.5 - cam[14]) / f
            dx = cam[9] + sx * cam[3] + sy * cam[6]
            dy = cam[10] + sx * cam[4] + sy * cam[7]
            dz = cam[11] + sx * cam[5] + sy * cam[8]
            l = math.sqrt(dx * dx + dy * dy + dz * dz)
            px = cam[0] + dx / l * tq
            py = cam[1] + dy / l * tq
            fp = tq / f
            r = math.sqrt(px * px + py * py)
            ca = grid_sample(CA, x0, x0, cell2, px, py)
            if ca == 0.0:
                ca = 1.0
            ms = grid_sample(MS, x0, x0, cell2, px, py)
            w = sstep(9.0, 10.8, r)
            if w <= 0.0 and ca >= 0.999 and ms <= 0.001:
                continue
            ar, ag, ab, ash = SH.ground_albedo(px, py, fp, PR)
            nr = ar
            ng = ag
            nb = ab
            if w > 0.0:
                pm = grid_sample(PMF, -PMF_R, -PMF_R, PMF_CELL, px, py) if r < PMF_R - 1.0 else \
                    grid_sample(PMC, -PMC_R, -PMC_R, PMC_CELL, px, py)
                n1 = fbm2(px * 0.045 + 1.3, py * 0.045, 331, 4, 2.1, 0.55, fp * 0.045)
                n2 = fbm2(px * 0.6, py * 0.6 + 2.0, 332, 3, 2.2, 0.5, fp * 0.6)
                n3 = fbm2(px * 0.011 - 4.0, py * 0.011 + 2.2, 333, 4, 2.1, 0.5, fp * 0.011)
                # the open moor: dry grass and old heath, paler than the trodden council ground under the moon
                g = 1.0 + 0.9 * n3 + 0.45 * n1 + 0.30 * n2
                nr = mix(nr, 0.068 * g, w)
                ng = mix(ng, 0.063 * g, w)
                nb = mix(nb, 0.050 * g, w)
                # heather and scrub in dark blotches, bare earth where the gathering has trodden the heath flat
                heath = sstep(0.02, 0.16, n1) * (1.0 - pm)
                nr = mix(nr, 0.030 * (1.0 + 0.5 * n2), 0.60 * heath)
                ng = mix(ng, 0.026 * (1.0 + 0.5 * n2), 0.60 * heath)
                nb = mix(nb, 0.024 * (1.0 + 0.5 * n2), 0.60 * heath)
                trod = sstep(19.0, 12.5, r) * (0.55 + 0.45 * sstep(-0.1, 0.2, n2))
                nr = mix(nr, 0.050 * (1.0 + 0.4 * n2), 0.7 * trod)
                ng = mix(ng, 0.045 * (1.0 + 0.4 * n2), 0.7 * trod)
                nb = mix(nb, 0.037 * (1.0 + 0.4 * n2), 0.7 * trod)
                # the worn roads: trodden earth, darker and browner than the dry grass either side
                ea = 0.046 * (1.0 + 0.35 * n2)
                nr = mix(nr, ea * 1.00, 0.80 * pm)
                ng = mix(ng, ea * 0.86, 0.80 * pm)
                nb = mix(nb, ea * 0.68, 0.80 * pm)
                nr = mix(ar, nr, w)
                ng = mix(ag, ng, w)
                nb = mix(ab, nb, w)
            c0 = rgb[y, x, 0]
            c1 = rgb[y, x, 1]
            c2 = rgb[y, x, 2]
            c0 = c0 * nr / max(ar, 1e-5)
            c1 = c1 * ng / max(ag, 1e-5)
            c2 = c2 * nb / max(ab, 1e-5)
            # the far dark: the moon on the open moor (brighter than on the council ground, where the torches rule),
            # shaped by the folds of the land
            wr = sstep(12.0, 30.0, r)
            wm = sstep(11.0, 22.0, r)
            if (wr > 0.0 or wm > 0.0) and mi > 0.0:
                e = 2.0
                hx = (_relief(px + e, py) - _relief(px - e, py)) / (2 * e)
                hy = (_relief(px, py + e) - _relief(px, py - e)) / (2 * e)
                hx *= wr
                hy *= wr
                nl = math.sqrt(hx * hx + hy * hy + 1.0)
                ndm = (-hx * mdx - hy * mdy + mdz) / nl
                boost = 1.0 + (MOON_MOOR - 1.0) * wm
                d_m = (max(ndm, 0.0) * boost - mdz) * mi
                sk = (boost * (0.55 + 0.45 / nl) - 1.0) * PR[SH.P_SKI] * SKY_MOOR
                c0 += nr * (d_m * PR[SH.P_MCR] + sk * PR[SH.P_SKR])
                c1 += ng * (d_m * PR[SH.P_MCG] + sk * PR[SH.P_SKG])
                c2 += nb * (d_m * PR[SH.P_MCB] + sk * PR[SH.P_SKB])
            # the crowd's moon shadows (the moor's brighter moon too), then their contact darkening
            if ms > 0.0 and mi > 0.0:
                e_m = mi * mdz * ms * (1.0 + (MOON_MOOR - 1.0) * sstep(11.0, 22.0, r))
                c0 = max(c0 - nr * e_m * PR[SH.P_MCR], c0 * 0.35)
                c1 = max(c1 - ng * e_m * PR[SH.P_MCG], c1 * 0.35)
                c2 = max(c2 - nb * e_m * PR[SH.P_MCB], c2 * 0.35)
            rgb[y, x, 0] = max(c0 * ca, 0.0)
            rgb[y, x, 1] = max(c1 * ca, 0.0)
            rgb[y, x, 2] = max(c2 * ca, 0.0)


# ========================================================== the crowd's figures ===

@njit(**FM)
def _trace_crowd(ox, oy, oz, dx, dy, dz, tmax, CF, BST, BID, bx0, by0, bcell, nbx, nby, pix):
    """Nearest crowd figure along the ray before tmax: (t, i, mat), i = -1 for none. The ray's stretch through the
    figures' slab (0 < z < ZTOP) is walked across the 2-D bins (DDA)."""
    if abs(dz) < 1e-9:
        if oz < 0.0 or oz > ZTOP:
            return -1.0, -1, 0
        ta = 0.0
        tb = tmax
    else:
        t0 = (0.0 - oz) / dz
        t1 = (ZTOP - oz) / dz
        ta = max(min(t0, t1), 0.0)
        tb = min(max(t0, t1), tmax)
    tb = min(tb, ta + 80.0)
    if tb <= ta:
        return -1.0, -1, 0
    idx = 1.0 / dx if abs(dx) > 1e-12 else 1e12
    idy = 1.0 / dy if abs(dy) > 1e-12 else 1e12
    idz = 1.0 / dz if abs(dz) > 1e-12 else 1e12
    gx = (ox + dx * ta - bx0) / bcell
    gy = (oy + dy * ta - by0) / bcell
    ix = int(math.floor(gx))
    iy = int(math.floor(gy))
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    # ray parameter per unit bin step in x, y
    tdx = abs(bcell / dx) if abs(dx) > 1e-12 else 1e30
    tdy = abs(bcell / dy) if abs(dy) > 1e-12 else 1e30
    if dx > 0:
        tmx = ta + ((ix + 1) - gx) * tdx
    elif dx < 0:
        tmx = ta + (gx - ix) * tdx
    else:
        tmx = 1e30
    if dy > 0:
        tmy = ta + ((iy + 1) - gy) * tdy
    elif dy < 0:
        tmy = ta + (gy - iy) * tdy
    else:
        tmy = 1e30
    best = tmax
    bi = -1
    bm = 0
    tcell = ta
    for it in range(400):
        if 0 <= ix < nbx and 0 <= iy < nby:
            b = iy * nbx + ix
            for q in range(BST[b], BST[b + 1]):
                i = BID[q]
                t0_, t1_ = G.ray_aabb(ox, oy, oz, idx, idy, idz, CF[i, G.F_BB], CF[i, G.F_BB + 1], CF[i, G.F_BB + 2],
                                      CF[i, G.F_BB + 3], CF[i, G.F_BB + 4], CF[i, G.F_BB + 5])
                if t1_ < max(t0_, 0.0) or t0_ > best:
                    continue
                th, m = G.trace_fig(ox, oy, oz, dx, dy, dz, t0_, min(t1_, best), CF, i, pix)
                if th > 0.0 and th < best:
                    best = th
                    bi = i
                    bm = m
        tnext = min(tmx, tmy)
        if best <= tnext or tnext > tb:
            break
        tcell = tnext
        if tmx < tmy:
            ix += sx
            tmx += tdx
        else:
            iy += sy
            tmy += tdy
    if bi < 0:
        return -1.0, -1, 0
    return best, bi, bm


@njit(**FM)
def _self_vis(px, py, pz, lx, ly, lz, CF, i):
    """Soft shadow of a figure on itself from its own torch (a short march through its own SDF)."""
    vx = lx - px
    vy = ly - py
    vz = lz - pz
    D = math.sqrt(vx * vx + vy * vy + vz * vz)
    if D < 1e-4:
        return 1.0
    vx /= D
    vy /= D
    vz /= D
    t = 0.014
    res = 1.0
    for it in range(14):
        if t > D - 0.12:
            break
        d, m = G.sd_fig(px + vx * t, py + vy * t, pz + vz * t, CF, i)
        if d < 0.0008:
            return 0.0
        res = min(res, 9.0 * d / t)
        t += max(d, 0.012)
    return sstep(0.0, 1.0, res)


@njit(**FM)
def _shade_fig(px, py, pz, vx, vy, vz, mat, CF, i, fp, PR, LT, OC, igc, igf, TL, NBR, EF, efx0, efc):
    T = PR[SH.P_T]
    h = max(0.0012, fp * 0.5)
    nx, ny, nz = G.normal_fig(px, py, pz, CF, i, h)
    ao = SH.cloth_ao(px, py, pz, nx, ny, nz, CF, i)
    sheen = PR[SH.P_SHEEN] * CF[i, G.F_SHEENK]
    er = 0.0
    eg = 0.0
    eb = 0.0
    seed = CF[i, G.F_SEED]
    if mat == G.M_CLOTH or mat == G.M_CLOTH2 or mat == G.M_SHAWL:
        if mat == G.M_CLOTH:
            ar = CF[i, G.F_R]
            ag = CF[i, G.F_G]
            ab = CF[i, G.F_B]
        else:
            ar = CF[i, G.F_R2]
            ag = CF[i, G.F_G2]
            ab = CF[i, G.F_B2]
        wf = 38.0 * CF[i, G.F_WEAVE]
        wv = 1.0 + (0.14 + 0.10 * CF[i, G.F_WEAVE]) * fbm2(px * wf + pz * 9.0 + seed, py * wf - pz * 7.0, 55, 3,
                                                            2.2, 0.5, fp * wf)
        dust = sstep(0.30, 0.03, pz)
        ar = ar * wv * (1.0 + 0.9 * dust) + 0.004 * dust
        ag = ag * wv * (1.0 + 0.8 * dust) + 0.0035 * dust
        ab = ab * wv * (1.0 + 0.6 * dust) + 0.003 * dust
    elif mat == G.M_TORCH:
        ar = 0.0070
        ag = 0.0058
        ab = 0.0052
        sheen = 0.0
        tl = CF[i, G.F_TLIT]
        if tl > 0.0:
            g1 = vnoise2(px * 55.0 + T * 0.09, py * 55.0 + pz * 40.0, 53)
            g2 = vnoise2(px * 160.0 - T * 0.13, py * 160.0 + pz * 120.0, 56)
            c = tl * (0.55 + 0.9 * g1 * g1 + 0.35 * g2)
            if tl < 0.2:
                c *= sstep(0.55, 0.85, g1)       # a spent head: a few embers in the char
            er += 1.6 * c
            eg += 0.55 * c
            eb += 0.08 * c
    elif mat == G.M_WOOD:
        gr = vnoise2(px * 40.0 + pz * 300.0, py * 40.0, 54)
        ar = 0.015 * (0.75 + 0.5 * gr)
        ag = 0.0105 * (0.75 + 0.5 * gr)
        ab = 0.0075 * (0.75 + 0.5 * gr)
        sheen = 0.0
    elif mat == G.M_GLOVE or mat == G.M_GILT:
        ar = 0.0105
        ag = 0.0080
        ab = 0.0066
        sheen = 0.7
    elif mat == G.M_SHADOW:
        ar = 0.004
        ag = 0.0035
        ab = 0.0035
        sheen = 0.0
    else:
        ar = 0.012
        ag = 0.010
        ab = 0.009
        sheen *= 0.7
    cr, cg, cb = SH.light_at(px, py, pz, nx, ny, nz, vx, vy, vz, ar, ag, ab, sheen, False, -1.0,
                             PR, LT, OC, igc, igf, ao)
    # its own torch and its neighbours' (the crowd's lights are not in accord3's LT)
    nn = NBR.shape[1]
    for q in range(-1, nn):
        k = i if q < 0 else NBR[i, q]
        if k < 0:
            continue
        I = TL[k, 4]
        if I <= 1e-5:
            continue
        lx = TL[k, 0] - px
        ly = TL[k, 1] - py
        lz = TL[k, 2] - pz
        d2 = lx * lx + ly * ly + lz * lz
        d = math.sqrt(d2)
        rl = TL[k, 3]
        fall = 1.0 / (d2 + rl * rl + 0.02)
        if I * fall < 0.0008:
            continue
        ndl = (nx * lx + ny * ly + nz * lz) / d
        f = max(ndl, 0.0)
        if sheen > 0.0:
            nv = abs(nx * vx + ny * vy + nz * vz)
            f += sheen * 0.6 * (1.0 - nv) ** 2.5 * max(ndl + 0.3, 0.0)
        if f <= 0.0:
            continue
        if q < 0:
            # a bearer's own torch on its bearer: accord3's convention (P_OWNK), with its own body's soft shadow
            f *= PR[SH.P_OWNK] * _self_vis(px + nx * 0.006, py + ny * 0.006, pz + nz * 0.006, TL[k, 0], TL[k, 1],
                                           TL[k, 2], CF, i)
        f *= fall * ao
        cr += ar * f * TL[k, 4]
        cg += ag * f * TL[k, 5]
        cb += ab * f * TL[k, 6]
    # the far crowd's glow from all round
    ef = grid_sample(EF, efx0, efx0, efc, px, py)
    ka = ef * (0.12 + 0.28 * max(nz, 0.0)) * ao
    cr += ar * ka * 1.0
    cg += ag * ka * 0.56
    cb += ab * ka * 0.16
    return cr + er, cg + eg, cb + eb


@njit(parallel=True, **FM)
def _crowd_pass(Wd, Hd, cam, CF, BST, BID, bx0, by0, bcell, nbx, nby, PR, LT, OC, igc, igf, TL, NBR, EF, efx0, efc,
                bg, dbg, out, dout, cid, aa, mask, nsub):
    Cx, Cy, Cz = cam[0], cam[1], cam[2]
    f = cam[12]
    pix = 1.0 / f
    for y in prange(Hd):
        for x in range(Wd):
            if aa and not mask[y, x]:
                continue
            ns = 1 if not aa else nsub
            ar = 0.0
            ag = 0.0
            ab = 0.0
            tmin = dbg[y, x]
            idm = -1
            for s in range(ns):
                if not aa:
                    jx = 0.0
                    jy = 0.0
                else:
                    jx = ((s + 1) * 0.7548776662 + 0.13) % 1.0 - 0.5
                    jy = ((s + 1) * 0.5698402910 + 0.47) % 1.0 - 0.5
                sx = (x + 0.5 + jx - cam[13]) / f
                sy = -(y + 0.5 + jy - cam[14]) / f
                dx = cam[9] + sx * cam[3] + sy * cam[6]
                dy = cam[10] + sx * cam[4] + sy * cam[7]
                dz = cam[11] + sx * cam[5] + sy * cam[8]
                l = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= l
                dy /= l
                dz /= l
                th, i, m = _trace_crowd(Cx, Cy, Cz, dx, dy, dz, dbg[y, x], CF, BST, BID, bx0, by0, bcell, nbx, nby,
                                        pix)
                if i >= 0:
                    r, g, b = _shade_fig(Cx + dx * th, Cy + dy * th, Cz + dz * th, -dx, -dy, -dz, m, CF, i, th * pix,
                                         PR, LT, OC, igc, igf, TL, NBR, EF, efx0, efc)
                    ar += r
                    ag += g
                    ab += b
                    if th < tmin:
                        tmin = th
                        idm = i
                else:
                    ar += bg[y, x, 0]
                    ag += bg[y, x, 1]
                    ab += bg[y, x, 2]
            if aa:
                out[y, x, 0] = (out[y, x, 0] + ar) / (ns + 1)
                out[y, x, 1] = (out[y, x, 1] + ag) / (ns + 1)
                out[y, x, 2] = (out[y, x, 2] + ab) / (ns + 1)
                if tmin < dout[y, x]:
                    dout[y, x] = tmin
            else:
                out[y, x, 0] = ar
                out[y, x, 1] = ag
                out[y, x, 2] = ab
                dout[y, x] = tmin
                cid[y, x] = idm


def _bins(CF, idx):
    """2-D bins (BIN m) of the figures idx over their AABBs: (BST, BID, x0, y0, nbx, nby)."""
    if len(idx) == 0:
        return np.zeros(2, np.int64), np.zeros(1, np.int64), 0.0, 0.0, 1, 1
    xa, xb = CF[idx, G.F_BB], CF[idx, G.F_BB + 1]
    ya, yb = CF[idx, G.F_BB + 2], CF[idx, G.F_BB + 3]
    x0 = math.floor(xa.min() / BIN) * BIN
    y0 = math.floor(ya.min() / BIN) * BIN
    nbx = int((xb.max() - x0) / BIN) + 1
    nby = int((yb.max() - y0) / BIN) + 1
    lists = [[] for _ in range(nbx * nby)]
    for q, i in enumerate(idx):
        for by in range(int((ya[q] - y0) / BIN), int((yb[q] - y0) / BIN) + 1):
            for bx in range(int((xa[q] - x0) / BIN), int((xb[q] - x0) / BIN) + 1):
                lists[by * nbx + bx].append(i)
    BST = np.zeros(nbx * nby + 1, np.int64)
    BST[1:] = np.cumsum([len(l) for l in lists])
    BID = np.array([i for l in lists for i in l] or [0], np.int64)
    return BST, BID, x0, y0, nbx, nby


def _in_view(CF, cam, Wd, Hd, margin_px=6.0):
    """Figures whose AABB can touch the frame."""
    if len(CF) == 0:
        return np.zeros(0, np.int64)
    c = np.stack([(CF[:, G.F_BB] + CF[:, G.F_BB + 1]) / 2, (CF[:, G.F_BB + 2] + CF[:, G.F_BB + 3]) / 2,
                  (CF[:, G.F_BB + 4] + CF[:, G.F_BB + 5]) / 2], -1)
    rad = 0.5 * np.linalg.norm(np.stack([CF[:, G.F_BB + 1] - CF[:, G.F_BB], CF[:, G.F_BB + 3] - CF[:, G.F_BB + 2],
                                         CF[:, G.F_BB + 5] - CF[:, G.F_BB + 4]], -1), axis=1)
    v = c - cam[0:3]
    zc = v @ cam[9:12]
    ok = zc > 0.05
    f = cam[12]
    xs = cam[13] + f * (v @ cam[3:6]) / np.maximum(zc, 1e-3)
    ys = cam[14] - f * (v @ cam[6:9]) / np.maximum(zc, 1e-3)
    rp = f * rad / np.maximum(zc, 1e-3) + margin_px
    ok &= (xs + rp > 0) & (xs - rp < Wd) & (ys + rp > 0) & (ys - rp < Hd)
    return np.nonzero(ok)[0].astype(np.int64)


def _neighbours(cs, K=6, R=5.0):
    TL = cs['TL']
    n = len(TL)
    NBR = -np.ones((max(n, 1), K), np.int64)
    lit = np.nonzero(TL[:, 4] > 1e-4)[0]
    if len(lit) == 0 or n == 0:
        return NBR
    tree = cKDTree(TL[lit, 0:2])
    d, j = tree.query(np.stack([cs['CF'][:, G.F_X], cs['CF'][:, G.F_Y]], -1), k=min(K + 1, len(lit)),
                      distance_upper_bound=R)
    if d.ndim == 1:
        d, j = d[:, None], j[:, None]
    for i in range(n):
        q = 0
        for c in range(j.shape[1]):
            if not np.isfinite(d[i, c]) or j[i, c] >= len(lit):
                continue
            k = lit[j[i, c]]
            if k == i:
                continue
            if q < K:
                NBR[i, q] = k
                q += 1
    return NBR


def composite(rgb, depth, oid, cam, PR, LT, OC, g, cs):
    """The land beyond the stones and the crowd's contact shadows (ground pixels), then the crowd's figures,
    z-tested against accord3's frame and anti-aliased over it. rgb, depth are updated in place."""
    Hd, Wd = depth.shape
    MC, MF = masks()
    _land_pass(rgb, depth, oid, cam, PR, g['CA'], g['MS'], g['x0f'], g['cell2'], MC, MF)
    CF = cs['CF']
    idx = _in_view(CF, cam, Wd, Hd)
    if len(idx) == 0:
        return
    if len(idx) > MAXF_RENDER:
        idx = idx[:MAXF_RENDER]
    BST, BID, bx0, by0, nbx, nby = _bins(CF, idx)
    NBR = _neighbours(cs)
    PRc = PR.copy()
    PRc[SH.P_WI] = 0.0
    TL = cs['TL']
    bg = rgb.copy()
    dbg = depth.copy()
    out = rgb
    cid = np.full((Hd, Wd), -1, np.int32)
    dummy = np.zeros((1, 1), np.bool_)
    _crowd_pass(Wd, Hd, cam, CF, BST, BID, float(bx0), float(by0), BIN, nbx, nby, PRc, LT, OC, g['igc'], g['igf'],
                TL, NBR, g['EF'], g['x0c'], g['cc'], bg, dbg, out, depth, cid, False, dummy, 1)
    # anti-alias the crowd's silhouettes and its high-contrast edges
    m = np.zeros((Hd, Wd), bool)
    d = cid[:, 1:] != cid[:, :-1]
    m[:, 1:] |= d
    m[:, :-1] |= d
    d = cid[1:, :] != cid[:-1, :]
    m[1:, :] |= d
    m[:-1, :] |= d
    Lg = np.log(out.max(axis=2) + 1e-3)
    c = np.zeros((Hd, Wd), bool)
    dx = np.abs(Lg[:, 1:] - Lg[:, :-1]) > 0.3
    c[:, 1:] |= dx
    c[:, :-1] |= dx
    dy = np.abs(Lg[1:, :] - Lg[:-1, :]) > 0.3
    c[1:, :] |= dy
    c[:-1, :] |= dy
    m |= c & (cid >= 0)
    if m.any():
        _crowd_pass(Wd, Hd, cam, CF, BST, BID, float(bx0), float(by0), BIN, nbx, nby, PRc, LT, OC, g['igc'],
                    g['igf'], TL, NBR, g['EF'], g['x0c'], g['cc'], bg, dbg, out, depth, cid, True, m, 6)
    oid[cid >= 0] = 400


# =================================================================== the flames ===

@njit(parallel=True, **FM)
def _flames(img, depth, cam, FL, nfl, T, n3, alpha):
    """flame3's torch flames (the same density and colour), each marched through its own world box; a flame
    smaller than a few pixels is supersampled so it keeps its light and does not shimmer."""
    Hd = img.shape[0]
    Wd = img.shape[1]
    f = cam[12]
    for k in range(nfl):
        if FL[k, 9] <= 0.002 or FL[k, 7] <= 0.0:
            continue
        Hmax = FL[k, 3] * 1.25
        Rm = FL[k, 4] * 1.9 + 0.02
        lx = FL[k, 5]
        ly = FL[k, 6]
        bx0 = FL[k, 0] + min(0.0, (lx - 0.22) * Hmax) - Rm
        bx1 = FL[k, 0] + max(0.0, (lx + 0.22) * Hmax) + Rm
        by0 = FL[k, 1] + min(0.0, (ly - 0.22) * Hmax) - Rm
        by1 = FL[k, 1] + max(0.0, (ly + 0.22) * Hmax) + Rm
        bz0 = FL[k, 2] - 0.36 * Hmax
        bz1 = FL[k, 2] + 1.3 * Hmax
        xa = 1e9
        xb = -1e9
        ya = 1e9
        yb = -1e9
        ok = True
        for c in range(8):
            px = bx1 if (c & 1) else bx0
            py = by1 if (c & 2) else by0
            pz = bz1 if (c & 4) else bz0
            vx = px - cam[0]
            vy = py - cam[1]
            vz = pz - cam[2]
            zc = vx * cam[9] + vy * cam[10] + vz * cam[11]
            if zc <= 1e-3:
                ok = False
                break
            xs = cam[13] + f * (vx * cam[3] + vy * cam[4] + vz * cam[5]) / zc
            ys = cam[14] - f * (vx * cam[6] + vy * cam[7] + vz * cam[8]) / zc
            xa = min(xa, xs)
            xb = max(xb, xs)
            ya = min(ya, ys)
            yb = max(yb, ys)
        if not ok:
            continue
        x0 = max(int(xa) - 1, 0)
        x1 = min(int(xb) + 2, Wd)
        y0 = max(int(ya) - 1, 0)
        y1 = min(int(yb) + 2, Hd)
        if x1 <= x0 or y1 <= y0:
            continue
        size = max(xb - xa, yb - ya)
        ss = 1
        if size < 14.0:
            ss = 2
        if size < 7.0:
            ss = 3
        if size < 3.5:
            ss = 4
        I = FL[k, 7]
        for y in prange(y0, y1):
            for x in range(x0, x1):
                er = 0.0
                eg = 0.0
                eb = 0.0
                trs = 0.0
                for sj in range(ss * ss):
                    ox_ = ((sj % ss) + 0.5) / ss - 0.5 if ss > 1 else 0.0
                    oy_ = ((sj // ss) + 0.5) / ss - 0.5 if ss > 1 else 0.0
                    sx = (x + 0.5 + ox_ - cam[13]) / f
                    sy = -(y + 0.5 + oy_ - cam[14]) / f
                    dx = cam[9] + sx * cam[3] + sy * cam[6]
                    dy = cam[10] + sx * cam[4] + sy * cam[7]
                    dz = cam[11] + sx * cam[5] + sy * cam[8]
                    l = math.sqrt(dx * dx + dy * dy + dz * dz)
                    dx /= l
                    dy /= l
                    dz /= l
                    ta = 0.01
                    tb = depth[y, x]
                    if abs(dx) > 1e-9:
                        t0_ = (bx0 - cam[0]) / dx
                        t1_ = (bx1 - cam[0]) / dx
                        ta = max(ta, min(t0_, t1_))
                        tb = min(tb, max(t0_, t1_))
                    if abs(dy) > 1e-9:
                        t0_ = (by0 - cam[1]) / dy
                        t1_ = (by1 - cam[1]) / dy
                        ta = max(ta, min(t0_, t1_))
                        tb = min(tb, max(t0_, t1_))
                    if abs(dz) > 1e-9:
                        t0_ = (bz0 - cam[2]) / dz
                        t1_ = (bz1 - cam[2]) / dz
                        ta = max(ta, min(t0_, t1_))
                        tb = min(tb, max(t0_, t1_))
                    tr = 1.0
                    if tb > ta:
                        nsteps = int(min(max((tb - ta) / 0.007, 12.0), 110.0 if ss == 1 else 48.0))
                        ds = (tb - ta) / nsteps
                        jit = (52.9829189 * ((0.06711056 * (x + ox_) + 0.00583715 * (y + oy_) + 0.1731 * (T % 7.0)) % 1.0)) % 1.0
                        for s in range(nsteps):
                            tt = ta + (s + jit) * ds
                            qx = cam[0] + dx * tt - FL[k, 0]
                            qy = cam[1] + dy * tt - FL[k, 1]
                            qz = cam[2] + dz * tt - FL[k, 2]
                            d, temp = FL3.torch_density(qx, qy, qz, FL, k, T, n3)
                            if d > 0.0:
                                cr, cg, cb = FL3.ramp(temp)
                                e = I * d * (0.05 + temp ** 2.6) * ds * tr
                                er += e * cr
                                eg += e * cg
                                eb += e * cb
                                tr *= math.exp(-d * ds * 7.0)
                                if tr < 0.01:
                                    break
                    trs += tr
                nq = ss * ss
                er /= nq
                eg /= nq
                eb /= nq
                tr = trs / nq
                img[y, x, 0] = img[y, x, 0] * (1.0 - alpha * (1.0 - tr)) + er
                img[y, x, 1] = img[y, x, 1] * (1.0 - alpha * (1.0 - tr)) + eg
                img[y, x, 2] = img[y, x, 2] * (1.0 - alpha * (1.0 - tr)) + eb


@njit(parallel=True, **FM)
def _airlight(img, depth, cam, TL, nt, sigma, hmin, lam, reach):
    """flame3.airlight's in-scatter round each torch, but each torch only over the pixels its halo can reach."""
    Hd = img.shape[0]
    Wd = img.shape[1]
    f = cam[12]
    for k in range(nt):
        I = max(TL[k, 4], max(TL[k, 5], TL[k, 6]))
        if I <= 1e-5:
            continue
        vx = TL[k, 0] - cam[0]
        vy = TL[k, 1] - cam[1]
        vz = TL[k, 2] - cam[2]
        zc = vx * cam[9] + vy * cam[10] + vz * cam[11]
        if zc <= 0.05:
            continue
        xs = cam[13] + f * (vx * cam[3] + vy * cam[4] + vz * cam[5]) / zc
        ys = cam[14] - f * (vx * cam[6] + vy * cam[7] + vz * cam[8]) / zc
        rp = f * reach / zc + 2.0
        x0 = max(int(xs - rp), 0)
        x1 = min(int(xs + rp) + 1, Wd)
        y0 = max(int(ys - rp), 0)
        y1 = min(int(ys + rp) + 1, Hd)
        if x1 <= x0 or y1 <= y0:
            continue
        for y in prange(y0, y1):
            for x in range(x0, x1):
                sx = (x + 0.5 - cam[13]) / f
                sy = -(y + 0.5 - cam[14]) / f
                dx = cam[9] + sx * cam[3] + sy * cam[6]
                dy = cam[10] + sx * cam[4] + sy * cam[7]
                dz = cam[11] + sx * cam[5] + sy * cam[8]
                l = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= l
                dy /= l
                dz /= l
                tm = min(depth[y, x], zc + 60.0)
                t0 = vx * dx + vy * dy + vz * dz
                hx = vx - t0 * dx
                hy = vy - t0 * dy
                hz = vz - t0 * dz
                h = math.sqrt(hx * hx + hy * hy + hz * hz + hmin * hmin)
                v = (math.atan((tm - t0) / h) - math.atan(-t0 / h)) / h / (1.0 + (h / lam) ** 2)
                v *= sstep(reach, reach * 0.6, h)
                img[y, x, 0] += sigma * v * TL[k, 4]
                img[y, x, 1] += sigma * v * TL[k, 5]
                img[y, x, 2] += sigma * v * TL[k, 6]


SMOKE_SIG = 0.010          # torch smoke: its in-scatter of its own flame (radiance per unit of torch intensity)
SMOKE_K = 0.25             # the smoke is splatted at this fraction of the frame's resolution


@njit(parallel=True, **FM)
def _plumes(buf, dmin, cam, k, TL, nt, SEED, wx, wy, T, n3, sig):
    """Torch smoke: from each lit torch a soft plume drifts downwind and rises, wavering in the eddies, lit by its
    own flame (brightest just above it). Gaussian puffs at the buffer's resolution, hidden behind nearer surfaces."""
    Hb = buf.shape[0]
    Wb = buf.shape[1]
    f = cam[12] * k
    cx = cam[13] * k
    cy = cam[14] * k
    NB = 7
    for i in range(nt):
        I = TL[i, 4]
        if I <= 1e-4:
            continue
        sd = SEED[i]
        for b in range(NB):
            sp = (b + 0.5) / NB
            Lp = 3.4
            wob = 0.45 * sp
            ox = TL[i, 0] + wx * Lp * sp + wob * (tex3(n3, sd * 3.1, sp * 5.0 - T * 0.05, 1.3) - 0.5) * 2.0
            oy = TL[i, 1] + wy * Lp * sp + wob * (tex3(n3, 7.7, sd * 3.1 + sp * 5.0 - T * 0.05, 2.1) - 0.5) * 2.0
            oz = TL[i, 2] + 0.22 + 1.15 * sp
            rad = 0.14 + 0.50 * sp
            e = I * sig / (1.0 + (Lp * sp / 0.55) ** 2) * (1.0 - 0.45 * sp)
            vx = ox - cam[0]
            vy = oy - cam[1]
            vz = oz - cam[2]
            zc = vx * cam[9] + vy * cam[10] + vz * cam[11]
            if zc <= 0.2:
                continue
            xs = cx + f * (vx * cam[3] + vy * cam[4] + vz * cam[5]) / zc
            ys = cy - f * (vx * cam[6] + vy * cam[7] + vz * cam[8]) / zc
            R = max(f * rad / zc, 0.35)
            rb = 2.6 * R + 1.0
            x0 = max(int(xs - rb), 0)
            x1 = min(int(xs + rb) + 1, Wb)
            y0 = max(int(ys - rb), 0)
            y1 = min(int(ys + rb) + 1, Hb)
            if x1 <= x0 or y1 <= y0:
                continue
            dist = math.sqrt(vx * vx + vy * vy + vz * vz)
            # a puff smaller than a pixel keeps its light (area-normalised below one pixel)
            amp = e * min(1.0, (R / 0.6) ** 2) if R < 0.6 else e
            for y in prange(y0, y1):
                for x in range(x0, x1):
                    if dmin[y, x] < dist - rad:
                        continue
                    dd = ((x + 0.5 - xs) ** 2 + (y + 0.5 - ys) ** 2) / (2.0 * R * R)
                    if dd > 6.0:
                        continue
                    g = amp * math.exp(-dd)
                    buf[y, x, 0] += g * 1.00
                    buf[y, x, 1] += g * 0.60
                    buf[y, x, 2] += g * 0.30


def smoke(rgb, depth, cam, cs, t, idx):
    """The torches' smoke over the frame (a quarter-resolution splat, softened and added)."""
    TL = cs['TL']
    if len(idx) == 0 or SMOKE_SIG <= 0.0:
        return
    Hd, Wd = depth.shape
    hb, wb = max(int(Hd * SMOKE_K), 1), max(int(Wd * SMOKE_K), 1)
    k = wb / float(Wd)
    dmin = cv2.resize(depth, (wb, hb), interpolation=cv2.INTER_NEAREST)
    dmin = cv2.erode(dmin, np.ones((3, 3), np.uint8))
    buf = np.zeros((hb, wb, 3), np.float32)
    wd = wind()
    wn = float(np.linalg.norm(wd)) + 1e-9
    seeds = np.ascontiguousarray(cs['seed'][idx])
    _plumes(buf, dmin, cam, k, np.ascontiguousarray(TL[idx]), len(idx), seeds, wd[0] / wn, wd[1] / wn, float(t),
            world()['n3'], SMOKE_SIG)
    buf = cv2.GaussianBlur(buf, (0, 0), 0.8)
    rgb += cv2.resize(buf, (Wd, Hd), interpolation=cv2.INTER_LINEAR)


def flames(rgb, depth, cam, cs, t):
    """The crowd's torch flames and the firelight in their smoke."""
    W = world()
    TL = cs['TL']
    FL = cs['FL']
    Hd, Wd = depth.shape
    # only flames that can be on screen
    idx = _in_view(cs['CF'], cam, Wd, Hd, 12.0)
    if len(idx) == 0:
        return
    FLv = np.ascontiguousarray(FL[idx])
    TLv = np.ascontiguousarray(TL[idx])
    _airlight(rgb, depth, cam, TLv, TLv.shape[0], haze(), 0.06, 0.35, 1.6)
    if os.environ.get('CROWD_SMOKE', '1') != '0':
        smoke(rgb, depth, cam, cs, t, idx)
    _flames(rgb, depth, cam, FLv, FLv.shape[0], float(t), W['n3'], 0.85)


# ================================================================ the plug-in ===
_C = {}
TIMING = os.environ.get('CROWD_TIMING') == '1'


def _tick(label, t0):
    import time as _t
    if TIMING:
        print(f'  crowd3 {label}: {_t.time() - t0:.2f}s', flush=True)
    return _t.time()


def _frame(t):
    c = _C.get('cs')
    if c is None or c['t'] != t:
        c = state(t)
        _C['cs'] = c
        _C.pop('g', None)
    return c


def occluders(cs, OC):
    """accord3's occluders plus the bodies of the crowd within ~6.5 m of the fire (P3: they shadow its light)."""
    if cs is None or cs['plate'] != 3:
        return OC
    CF = cs['CF']
    near = np.nonzero(np.hypot(CF[:, G.F_X], CF[:, G.F_Y]) < 6.5)[0]
    if len(near) == 0:
        return OC
    return np.ascontiguousarray(np.concatenate([OC, _occ_rows(cs, near)]))


def _failed(what):
    """A crowd failure never takes accord3's frame down with it, unless CROWD_REQUIRED=1 (finals)."""
    import traceback
    if os.environ.get('CROWD_REQUIRED') == '1':
        raise
    print(f'crowd3: {what} FAILED (frame renders without it):', flush=True)
    traceback.print_exc()


def install(A3=None):
    """Plug the crowd into accord3 without touching its source: accord3's SH and FL3 names are pointed at thin
    proxies whose render_surfaces / torch_flames run the original, and add the crowd (its light on the ground,
    the land, its figures after the AA pass, its flames). Idempotent."""
    import types
    A3 = A3 or sys.modules.get('accord3')
    if A3 is None:
        raise RuntimeError('install(): import accord3 first')
    if os.environ.get('CROWD') == '0':
        return A3
    if getattr(A3, '_CROWD_INSTALLED', False):
        return A3
    rs0 = SH.render_surfaces
    tf0 = FL3.torch_flames
    al0 = FL3.airlight
    _A3['m'] = A3

    def render_surfaces(Wd, Hd, cam, PR, LT, OC, F, nf, S, ns, KB, LG, nlog, CH, nch, HDs, Js, HBB, RP, stex, ssz,
                        igc, igf, rgb, depth, oid, aa_pass, mask, nsub):
        g = None
        try:
            t = float(PR[SH.P_T])
            cs = _frame(t)
            key = (t, cam.tobytes(), int(Wd), int(Hd))
            g = _C.get('g')
            if g is None or g['key'] != key:
                import time as _t
                t0_ = _t.time()
                OCx = occluders(cs, OC)
                g = ground_light(cs, cam, Wd, Hd, OCx, PR)
                _tick(f'ground light {g["igf"].shape[0]}^2 cell {g["cf"]:.3f}', t0_)
                g['key'] = key
                g['OC'] = OCx
                g['pcrowd0'] = float(PR[SH.P_CROWD])
                _C['g'] = g
            OCx = g['OC']
            PR[SH.P_NOC] = OCx.shape[0]
            PR[SH.P_WI] = 1.0
            PR[SH.P_IGF_X0], PR[SH.P_IGF_CELL] = g['x0f'], g['cf']
            PR[SH.P_IGC_X0], PR[SH.P_IGC_CELL] = g['x0c'], g['cc']
            PR[SH.P_CROWD] = g['pcrowd0'] * lit_fraction(cs)
        except Exception:
            _failed('crowd light')
            g = None
        if g is None:
            rs0(Wd, Hd, cam, PR, LT, OC, F, nf, S, ns, KB, LG, nlog, CH, nch, HDs, Js, HBB, RP, stex, ssz,
                igc, igf, rgb, depth, oid, aa_pass, mask, nsub)
            return
        rs0(Wd, Hd, cam, PR, LT, OCx, F, nf, S, ns, KB, LG, nlog, CH, nch, HDs, Js, HBB, RP, stex, ssz,
            g['igc'], g['igf'], rgb, depth, oid, aa_pass, mask, nsub)
        if aa_pass:
            try:
                import time as _t
                t0_ = _t.time()
                composite(rgb, depth, oid, cam, PR, LT, OCx, g, _frame(float(PR[SH.P_T])))
                _tick('land + figures', t0_)
            except Exception:
                _failed('crowd figures')

    def torch_flames(img, depth, cam, FL, nfl, T, n3, alpha):
        tf0(img, depth, cam, FL, nfl, T, n3, alpha)
        try:
            import time as _t
            t0_ = _t.time()
            flames(img, depth, cam, _frame(float(T)), float(T))
            _tick('flames', t0_)
        except Exception:
            _failed('crowd flames')

    def airlight(img, depth, cam, LT, nl, sigma, hmin, lam):
        _A3['hz'] = float(sigma)
        al0(img, depth, cam, LT, nl, sigma, hmin, lam)

    shp = types.ModuleType('shade3_crowd')
    shp.__dict__.update(SH.__dict__)
    shp.render_surfaces = render_surfaces
    flp = types.ModuleType('flame3_crowd')
    flp.__dict__.update(FL3.__dict__)
    flp.torch_flames = torch_flames
    flp.airlight = airlight
    A3.SH = shp
    A3.FL3 = flp
    A3._CROWD_INSTALLED = True
    return A3
