"""ACCORD v3 (cut C): the rivers of torches and the crowd round the stones.

The paths are v2's worn roads (plain.py's generator, same seeds, rebuilt here so a cloud box needs no cache):
10 main roads from 470-560 m out and 16 branches, all running in to R_END just outside the ring of stones.
Walkers carry torches along them toward the stones, a time-lapse at altitude easing to real time near the
ground; at the road's end each walks on to a place in the crowd and stands, facing the council. A crowd has
already gathered when the film arrives (a ring two or three deep outside the stones, thickest at the road
mouths). Bar 70's end (P3, 5590+): the crowd turns and carries the flame back out along the roads.

People near the camera (and big enough to read) are real figures: geom3.sd_fig rows appended after the council
in Fa (F_CROWD = 1: lit by their own torch in shade3, never by the council's crowd rim), each with a real torch
flame (flame3.torch_flames rows). The rest are drawn as dark sprites with torch streaks and a soft halo (v2).
Their torchlight on the ground comes from two splatted irradiance grids (shade3 P_WI)."""
import math

import numpy as np
from numba import njit

import scene3 as SC
from scene3 import smooth, smoother, ramp, FIRE_HOT, FIRE_CORE, FIRE_MID
import geom3 as G
import flame3 as FL3
from nbcore import FM

R_END = 9.6                         # the roads end just outside the stones (r 7.5)
IGC_R, IGC_CELL = 560.0, 2.0        # coarse torchlight on the plain
IGF_R, IGF_CELL = 30.0, 0.10        # fine torchlight near the stones
F_CROWD = 65                        # Fa column: 1 = a crowd figure (see module doc)
N_STAND = 78                        # the crowd already gathered at 4480

_P = {}


# ------------------------------------------------------------------ roads ---

def _main_paths(rng):
    paths = []
    n = 10
    for k in range(n):
        a = 2 * np.pi * k / n + rng.uniform(-0.22, 0.22)
        R0 = rng.uniform(470, 560)
        r = np.geomspace(R0, R_END, 900)
        lr = np.log(r)
        th = np.full_like(r, a)
        for _ in range(3):
            A = rng.uniform(0.05, 0.16)
            fq = rng.uniform(1.2, 3.4)
            ph = rng.uniform(0, 2 * np.pi)
            th += A * np.sin(fq * lr + ph) * np.clip((r - R_END) / 60.0, 0, 1)
        paths.append(np.stack([r * np.cos(th), r * np.sin(th)], -1))
    return paths


def _branches(rng, mains):
    out = []
    for k in range(16):
        m = mains[rng.integers(len(mains))]
        rj = rng.uniform(60, 260)
        r_m = np.hypot(m[:, 0], m[:, 1])
        j = int(np.argmin(np.abs(r_m - rj)))
        J = m[j]
        aj = math.atan2(J[1], J[0])
        a0 = aj + rng.choice([-1, 1]) * rng.uniform(0.25, 0.9)
        R0 = rng.uniform(rj + 150, 560)
        P0 = np.array([R0 * math.cos(a0), R0 * math.sin(a0)])
        s = np.linspace(0, 1, 500)[:, None]
        mid = 0.5 * (P0 + J) + rng.normal(0, 25, 2)
        pts = (1 - s) ** 2 * P0 + 2 * (1 - s) * s * mid + s * s * J
        pts = np.concatenate([pts, m[j + 1:]], 0)
        out.append(pts)
    return out


def _arclen(p):
    d = np.hypot(*np.diff(p, axis=0).T)
    return np.concatenate([[0], np.cumsum(d)])


# -------------------------------------------------------------- the clock ---

def _speed(t):
    """Walking speed on screen (m/s): a time-lapse at altitude, real time once the camera is among them;
    bar 70's end: the flame carried out, easing back into a time-lapse as the camera rises."""
    s = 1.30 + 10.5 * (1.0 - smooth(ramp(t, 4480, 4612)))
    if t > 5560:
        s = 1.30 + 7.0 * smooth(ramp(t, 5610, 5679))
    return s


_TG = np.arange(4400.0, 5720.0, 0.25)
_D = np.concatenate([[0.0], np.cumsum(np.array([_speed(u) for u in _TG[:-1]]) * 0.25 / 24.0)])


def _dist(t):
    return float(np.interp(t, _TG, _D) - np.interp(4480.0, _TG, _D))


# ---------------------------------------------------------------- people ---

_CLOTH = [SC._CL[k] for k in ('black', 'umber', 'charcoal', 'wool', 'brown', 'grey', 'bluegrey', 'charcoal',
                             'umber', 'black', 'brown', 'charcoal')]


def load():
    if 'paths' in _P:
        return _P
    rng = np.random.default_rng(1234)
    mains = _main_paths(rng)
    paths = mains + _branches(rng, mains)
    Ls = [_arclen(p) for p in paths]
    _P['paths'] = paths
    _P['L'] = Ls
    rng = np.random.default_rng(99)
    W = []
    for pi, p in enumerate(paths):
        L = Ls[pi]
        # walkers spaced along the road in clumps; a few beyond its far end arrive later
        s = rng.uniform(-40.0, 30.0)
        dens = 6.5 if pi < 10 else 10.0
        while s < L[-1] + 40:
            W.append((pi, L[-1] - s, rng.uniform(0.85, 1.15), rng.normal(0, 0.45), rng.choice([-1.0, 1.0]),
                      rng.uniform(0, 6.28)))
            s += rng.exponential(dens) + 0.8
    W = np.array(W, np.float64)            # (path, arc position at 4480, speed k, lateral, torch side, phase)
    # a walker arriving walks on to a place in the crowd near its road's mouth
    n = W.shape[0]
    ends = np.array([paths[int(i)][-1] for i in W[:, 0]])
    ea = np.arctan2(ends[:, 1], ends[:, 0])
    cr = 10.6 + np.clip(rng.exponential(2.2, n), 0, 7)
    ca = ea + rng.normal(0, 1, n) * (0.10 + 1.2 / cr)
    _P['W'] = W
    _P['spot'] = np.stack([cr * np.cos(ca), cr * np.sin(ca)], -1)
    # the crowd already gathered: two or three deep outside the stones, thickest at the road mouths
    mouths = np.array([math.atan2(p[-1, 1], p[-1, 0]) for p in paths[:10]])
    a = np.concatenate([mouths[rng.integers(0, 10, N_STAND // 2)] + rng.normal(0, 0.16, N_STAND // 2),
                        rng.uniform(0, 2 * np.pi, N_STAND - N_STAND // 2)])
    r = 9.25 + np.clip(rng.exponential(1.1, N_STAND), 0, 4.0)
    # keep them from standing inside one another (a little relaxation)
    P = np.stack([r * np.cos(a), r * np.sin(a)], -1)
    for _ in range(40):
        d = P[:, None, :] - P[None, :, :]
        dd = np.hypot(d[..., 0], d[..., 1]) + np.eye(N_STAND) * 9
        push = np.clip(0.62 - dd, 0, None)[..., None] * d / dd[..., None]
        P += 0.5 * push.sum(1)
        rr = np.hypot(P[:, 0], P[:, 1])
        P *= (np.maximum(rr, 9.15) / rr)[:, None]
    _P['stand'] = P
    # everyone's build and cloth: [walkers..., standing...]
    m = n + N_STAND
    _P['hs'] = rng.uniform(0.90, 1.12, m)
    _P['ws'] = rng.uniform(0.92, 1.12, m)
    _P['typ'] = rng.choice([0, 1, 1, 3, 5, 1, 0], m)
    _P['col'] = rng.integers(0, len(_CLOTH), m)
    _P['col2'] = rng.integers(0, len(_CLOTH), m)
    _P['side'] = np.concatenate([W[:, 4], rng.choice([-1.0, 1.0], N_STAND)])
    _P['ph'] = np.concatenate([W[:, 5], rng.uniform(0, 6.28, N_STAND)])
    return _P


def people(t):
    """Everyone at t: pos (n,2), heading (n,), walk 0..1 (n,), gait phase (n,), visible (n,)."""
    P = load()
    W = P['W']
    paths = P['paths']
    Ls = P['L']
    D = _dist(t)
    n = W.shape[0]
    pos = np.zeros((n, 2))
    head = np.zeros(n)
    walk = np.zeros(n)
    vis = np.ones(n, bool)
    out = t > 5590.0
    for k in range(n):
        pi = int(W[k, 0])
        p = paths[pi]
        L = Ls[pi]
        s = W[k, 1] + W[k, 2] * D
        if s < 0.0:
            vis[k] = False
            continue
        if s <= L[-1]:
            x = np.interp(s, L, p[:, 0])
            y = np.interp(s, L, p[:, 1])
            s2 = min(s + 1.0, L[-1])
            x2 = np.interp(s2, L, p[:, 0])
            y2 = np.interp(s2, L, p[:, 1])
            hd = math.atan2(y2 - y, x2 - x)
            nx, ny = -(y2 - y), (x2 - x)
            nl = math.hypot(nx, ny) + 1e-9
            pos[k] = (x + W[k, 3] * nx / nl, y + W[k, 3] * ny / nl)
            head[k] = hd
            walk[k] = 1.0
        else:
            # arrived: on to a place in the crowd, then stand facing the council
            over = s - L[-1]
            e = p[-1]
            c = P['spot'][k]
            dl = math.hypot(*(c - e)) + 1e-6
            u = min(over / dl, 1.0)
            pos[k] = e + (c - e) * u
            head[k] = math.atan2(c[1] - e[1], c[0] - e[0]) if u < 1.0 else math.atan2(-pos[k, 1], -pos[k, 0])
            walk[k] = 1.0 - smooth(ramp(over - dl, -0.6, 0.4))
    st = P['stand']
    sw = 0.03 * np.stack([np.sin(t * 0.05 + P['ph'][n:]), np.cos(t * 0.04 + P['ph'][n:])], -1)
    pos = np.concatenate([pos, st + sw], 0)
    head = np.concatenate([head, np.arctan2(-st[:, 1], -st[:, 0])])
    walk = np.concatenate([walk, np.zeros(N_STAND)])
    vis = np.concatenate([vis, np.ones(N_STAND, bool)])
    if out:
        # bar 70's end: everyone turns and carries the flame back out along the roads (radially, then the roads)
        u = smooth(ramp(t, 5590, 5606))
        rr = np.hypot(pos[:, 0], pos[:, 1]) + 1e-9
        ho = np.arctan2(pos[:, 1], pos[:, 0])
        dh = (ho - head + np.pi) % (2 * np.pi) - np.pi
        head = head + dh * u
        dd = max(_dist(t) - _dist(5598.0), 0.0) * smooth(ramp(t, 5598, 5612))
        pos = pos + np.stack([np.cos(ho), np.sin(ho)], -1) * (dd * (0.85 + 0.3 * np.cos(P['ph'] * 3.0)))[:, None]
        walk = np.maximum(walk, smooth(ramp(t, 5598, 5610)))
    phase = P['ph'] + _dist(t) / 0.36 * math.pi * walk
    return pos, head, walk, phase, vis


# ------------------------------------------------------- figures (near) ---

def _two_bone(S, Hd, L1, L2, pole):
    return SC._two_bone(S, Hd, L1, L2, pole)


def fig_rows(t, cam, scale, n_max=190):
    """Real figures for the people near the camera and in frame: Fa rows (after the council) and torch FL rows.
    Returns (rows, FL rows, mask of who they are)."""
    P = load()
    pos, head, walk, phase, vis = people(t)
    h = cam[2]
    if h > 150.0:
        return np.zeros((0, G.F_N)), np.zeros((0, FL3.FL_N)), np.zeros(pos.shape[0], bool)
    # in frame (with a margin), nearest first
    xy, zc = SC.project(cam, np.concatenate([pos, np.full((pos.shape[0], 1), 1.0)], 1))
    W_, H_ = SC.W * scale, SC.H * scale
    mg = 80 * scale
    ok = vis & (zc > 0.5) & (xy[:, 0] > -mg) & (xy[:, 0] < W_ + mg) & (xy[:, 1] > -mg) & (xy[:, 1] < H_ + mg)
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(zc[idx])][:n_max]
    rows = np.zeros((len(idx), G.F_N))
    FL = np.zeros((len(idx), FL3.FL_N))
    sel = np.zeros(pos.shape[0], bool)
    sel[idx] = True
    for r_, k in enumerate(idx):
        hs, ws = P['hs'][k], P['ws'][k]
        side = P['side'][k]
        a = head[k]
        wk = walk[k]
        ph = phase[k]
        R = rows[r_]
        R[G.F_X], R[G.F_Y] = pos[k]
        R[G.F_ANG] = a
        R[G.F_C], R[G.F_S] = math.cos(a), math.sin(a)
        R[G.F_HS], R[G.F_WS] = hs, ws
        R[G.F_TYPE] = P['typ'][k]
        R[G.F_SEED] = 3.3 * k + 0.7
        R[G.F_R:G.F_B + 1] = _CLOTH[P['col'][k]]
        R[G.F_R2:G.F_B2 + 1] = _CLOTH[P['col2'][k]]
        R[G.F_CAPE] = (0.28 + 0.14 * ((k * 7) % 5) / 4.0) * hs * (0.0 if (k % 4) == 1 else 1.0)
        R[G.F_TRAIN] = 0.14 + 0.12 * ((k * 3) % 5) / 4.0
        R[G.F_WEAVE] = 0.7 + 1.2 * ((k * 11) % 7) / 6.0
        R[G.F_SHEENK] = 0.6 + 0.9 * ((k * 5) % 7) / 6.0
        R[G.F_SIDE] = side
        R[G.F_LEAN] = math.radians(4.0) * wk
        R[G.F_BREATH] = 0.006 * math.sin(2 * math.pi * t / 96.0 + ph)
        R[G.F_BOW] = math.radians(4.0 + 3.0 * ((k * 13) % 5) / 4.0)
        R[G.F_SWAY] = 0.018 * wk * math.sin(ph)
        R[G.F_HEM] = 0.05 * wk
        R[G.F_PHASE] = ph
        R[G.F_TORCH] = 1.0
        R[G.F_TLIT] = 1.0
        R[F_CROWD] = 1.0
        # the torch held up before them (bobbing a little with the walk)
        zsh = 1.33 * hs
        sh = np.array([0.0, side * 0.200 * ws, zsh - 0.035 * hs])
        hand = np.array([0.30, side * 0.13, 1.08 * hs + 0.012 * wk * math.sin(2 * ph)])
        tor = SC._unit([0.20, -side * 0.05, 1.0])
        E, Hh = _two_bone(sh, hand, 0.31, 0.30, np.array([-0.35, side * 1.0, -0.9]))
        R[G.F_SHX:G.F_SHZ + 1] = sh
        R[G.F_EX:G.F_EZ + 1] = E
        R[G.F_HX:G.F_HZ + 1] = Hh
        R[G.F_TX:G.F_TZ + 1] = tor
        ca, sa = math.cos(a), math.sin(a)
        pts = np.array([[sx, sy, sz] for sx in (-(0.30 + R[G.F_TRAIN]) * ws, 0.34 * ws)
                        for sy in (-0.40 * ws, 0.40 * ws) for sz in (0.0, 1.97 * hs)]
                       + [Hh + 0.60 * tor, Hh - 0.30 * tor, E])
        wx = pos[k, 0] + pts[:, 0] * ca - pts[:, 1] * sa
        wy = pos[k, 1] + pts[:, 0] * sa + pts[:, 1] * ca
        m = 0.10
        R[G.F_BB:G.F_BB + 6] = (wx.min() - m, wx.max() + m, wy.min() - m, wy.max() + m, -0.02, pts[:, 2].max() + m)
        # its flame (world): the head of the torch
        top = Hh + 0.44 * tor
        bx = pos[k, 0] + top[0] * ca - top[1] * sa
        by = pos[k, 1] + top[0] * sa + top[1] * ca
        ax_ = tor[0] * ca - tor[1] * sa
        ay_ = tor[0] * sa + tor[1] * ca
        vel = 1.3 * wk * np.array([math.cos(a), math.sin(a)])
        lean = SC_WIND - 0.35 * vel
        ln = np.linalg.norm(lean)
        if ln > 1.6:
            lean *= 1.6 / ln
        FL[r_] = (bx - 0.03 * ax_, by - 0.03 * ay_, top[2] - 0.03 * tor[2], 0.40 * (0.9 + 0.2 * ((k * 17) % 7) / 6.0),
                  0.082, lean[0], lean[1], 20.0, 5.1 * k + 0.3, 1.0)
    return rows, FL, sel


SC_WIND = np.array([0.40, -0.16])     # the same breeze as the council's flames (accord3._WIND)


# ------------------------------------------------------ sprites (far) ---

def sprites(t, cam, sel, dt_trail):
    """Dark sprites (FI.dark_sprites rows), torch streaks (FI.splat_streaks rows) and halos (FI.splat_blobs)
    for everyone not drawn as a real figure. dt_trail: the torch trail's length in frames (time-lapse smear)."""
    pos, head, walk, phase, vis = people(t)
    posp, _, _, _, _ = people(t - dt_trail)
    P = load()
    k = np.nonzero(vis & ~sel)[0]
    n = len(k)
    Q = np.zeros((n, 9))
    Q[:, 0:2] = pos[k]
    Q[:, 2] = 1.3
    Q[:, 3] = head[k]
    Q[:, 4] = 0.50 * P['ws'][k]
    Q[:, 5] = 0.60 * P['ws'][k]
    Q[:, 6] = 0.92
    Q[:, 7] = 0.35
    Q[:, 8] = P['side'][k]
    side = P['side'][k]
    hk = head[k]
    off = 0.26
    tx = pos[k, 0] - side * off * np.sin(hk) + 0.18 * np.cos(hk)
    ty = pos[k, 1] + side * off * np.cos(hk) + 0.18 * np.sin(hk)
    tpx = posp[k, 0] - side * off * np.sin(hk) + 0.18 * np.cos(hk)
    tpy = posp[k, 1] + side * off * np.cos(hk) + 0.18 * np.sin(hk)
    fl = 0.85 + 0.15 * np.sin(t * 2.3 + k * 1.7)
    col = (0.55 * FIRE_HOT + 0.45 * FIRE_CORE)[None, :] * (20.0 * fl)[:, None]
    S = np.zeros((n, 11))
    S[:, 0] = tpx
    S[:, 1] = tpy
    S[:, 2] = 1.95
    S[:, 3] = tx
    S[:, 4] = ty
    S[:, 5] = 1.95
    S[:, 6] = 0.07
    S[:, 7:10] = col
    S[:, 10] = 1.0
    Hs = np.zeros((n, 8))
    Hs[:, 0] = tx
    Hs[:, 1] = ty
    Hs[:, 2] = 1.95
    Hs[:, 3] = 0.9
    Hs[:, 4:7] = FIRE_MID[None, :] * 0.035
    Hs[:, 7] = 1.0
    return Q, S, Hs


# ---------------------------------------------------------- irradiance ---

@njit(**FM)
def _splat_irr(g, x0, cell, P, n, I, h, rmax):
    N = g.shape[0]
    for k in range(n):
        u = (P[k, 0] - x0) / cell
        v = (P[k, 1] - x0) / cell
        rc = int(rmax / cell) + 1
        iu = int(u)
        iv = int(v)
        if iu + rc < 0 or iv + rc < 0 or iu - rc >= N or iv - rc >= N:
            continue
        for yy in range(max(iv - rc, 0), min(iv + rc + 1, N)):
            for xx in range(max(iu - rc, 0), min(iu + rc + 1, N)):
                dx = (xx + 0.5 - u) * cell
                dy = (yy + 0.5 - v) * cell
                d2 = dx * dx + dy * dy
                if d2 > rmax * rmax:
                    continue
                fall = 1.0 - d2 / (rmax * rmax)
                g[yy, xx] += I * h / (d2 + h * h) ** 1.5 * fall * fall


def irradiance(t):
    """Torchlight on the plain: a coarse grid (the rivers from altitude) and a fine one near the stones."""
    pos, head, walk, phase, vis = people(t)
    P = load()
    side = P['side']
    tx = pos[:, 0] - side * 0.26 * np.sin(head) + 0.18 * np.cos(head)
    ty = pos[:, 1] + side * 0.26 * np.cos(head) + 0.18 * np.sin(head)
    T = np.stack([tx, ty], -1)[vis]
    Nc = int(2 * IGC_R / IGC_CELL)
    Nf = int(2 * IGF_R / IGF_CELL)
    igc = np.zeros((Nc, Nc), np.float32)
    igf = np.zeros((Nf, Nf), np.float32)
    I = 0.30
    _splat_irr(igc, -IGC_R, IGC_CELL, T, T.shape[0], I, 1.95, 7.0)
    near = np.nonzero((np.abs(T[:, 0]) < IGF_R + 6) & (np.abs(T[:, 1]) < IGF_R + 6))[0]
    Tn = np.ascontiguousarray(T[near])
    _splat_irr(igf, -IGF_R, IGF_CELL, Tn, Tn.shape[0], I, 1.95, 6.0)
    return igc, igf


MIST = np.array([
    # z, opacity, scale, drift_x, drift_y, seed
    [255.0, 0.40, 150.0, 0.05, 0.02, 71],
    [140.0, 0.32, 95.0, -0.04, 0.03, 72],
    [66.0, 0.24, 60.0, 0.03, -0.02, 73],
], np.float64)
