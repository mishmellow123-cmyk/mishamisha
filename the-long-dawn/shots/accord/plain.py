"""The night plain for ACCORD's descent: worn paths, rivers of torch-bearers converging on the
stone ring (time-lapse at altitude, real time near the ground), the crowd, their light, mist."""
import math
import os

import cv2
import numpy as np
from numba import njit

import scene as SC
from scene import smooth, ramp, FIRE_HOT, FIRE_CORE, FIRE_MID
import fire as FI
from nbcore import FM

CACHE = os.path.join(SC.ROOT, 'renders', 'accord_cache')
PGC_R, PGC_CELL = 560.0, 0.5        # coarse path mask: +-560 m @ 0.5 m
PGF_R, PGF_CELL = 40.0, 0.04        # fine path mask: +-40 m @ 4 cm
IGC_R, IGC_CELL = 560.0, 2.0        # coarse walker irradiance
IGF_R, IGF_CELL = 30.0, 0.10        # fine walker irradiance
R_END = 9.6                         # paths end just outside the stones (r=7.6)

_P = {}


# ------------------------------------------------------------------ paths ---

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
        # then follow the main path inward
        pts = np.concatenate([pts, m[j + 1:]], 0)
        out.append(pts)
    return out


def _arclen(p):
    d = np.hypot(*np.diff(p, axis=0).T)
    return np.concatenate([[0], np.cumsum(d)])


def build():
    rng = np.random.default_rng(1234)
    mains = _main_paths(rng)
    branches = _branches(rng, mains)
    paths = mains + branches
    widths = [2.6] * len(mains) + [1.5] * len(branches)
    # masks
    Nc = int(2 * PGC_R / PGC_CELL)
    pgc = np.zeros((Nc, Nc), np.float32)
    for p, w in zip(paths, widths):
        q = ((p + PGC_R) / PGC_CELL * 8).astype(np.int32)
        cv2.polylines(pgc, [q.reshape(-1, 1, 2)], False, 1.0, max(1, int(w / PGC_CELL)), cv2.LINE_AA, shift=3)
    pgc = cv2.GaussianBlur(pgc, (0, 0), 1.0)
    Nf = int(2 * PGF_R / PGF_CELL)
    pgf = np.zeros((Nf, Nf), np.float32)
    for p, w in zip(paths, widths):
        rr = np.hypot(p[:, 0], p[:, 1])
        sel = rr < PGF_R * 1.5
        if sel.sum() < 2:
            continue
        pp = p[sel]
        # widen near the stones (trampled approaches)
        for i0 in range(0, len(pp) - 1):
            r = math.hypot(*pp[i0])
            ww = w * (1.0 + 1.2 * max(0.0, 1.0 - (r - R_END) / 20.0))
            a = ((pp[i0] + PGF_R) / PGF_CELL * 8).astype(np.int32)
            b = ((pp[i0 + 1] + PGF_R) / PGF_CELL * 8).astype(np.int32)
            cv2.line(pgf, tuple(a), tuple(b), 1.0, max(1, int(ww / PGF_CELL)), cv2.LINE_AA, shift=3)
    pgf = cv2.GaussianBlur(pgf, (0, 0), 6.0)
    pgf = np.clip(pgf * 1.3, 0, 1)
    return paths, pgc, pgf


def load():
    if 'pgc' in _P:
        return _P['plain']
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, 'plain_v2.npz')
    paths, _, _ = None, None, None
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=True)
        pgc, pgf = z['pgc'], z['pgf']
        paths = list(z['paths'])
    else:
        paths, pgc, pgf = build()
        np.savez(fn, pgc=pgc, pgf=pgf, paths=np.array(paths, dtype=object))
    _P['paths'] = paths
    _P['pgc'] = pgc
    _P['plain'] = dict(pgc=pgc, pgf=pgf, pgc_x0=-PGC_R, pgc_cell=PGC_CELL, pgf_x0=-PGF_R, pgf_cell=PGF_CELL)
    _build_walkers()
    return _P['plain']


# ---------------------------------------------------------------- walkers ---

def _speed(t):
    """Time-lapse walking speed (m/s): fast at altitude, real time near the ground."""
    return 1.35 + 11.0 * (1.0 - smooth(ramp(t, 1912, 1980)))


_TGRID = np.arange(1880.0, 2300.0, 0.25)
_DIST = np.concatenate([[0], np.cumsum(np.array([_speed(t) for t in _TGRID[:-1]]) * 0.25 / 24.0)])
_D0 = float(np.interp(1912.0, _TGRID, _DIST))


def _dist(t):
    return np.interp(t, _TGRID, _DIST) - _D0


def _build_walkers():
    rng = np.random.default_rng(99)
    paths = _P['paths']
    W = []
    for pi, p in enumerate(paths):
        L = _arclen(p)
        _P.setdefault('L', []).append(L)
        # walkers spaced along the path (clumps), some start beyond the far end (arrive later)
        s = -rng.uniform(0, 120)
        dens = 7.0 if pi < 10 else 11.0
        while s < L[-1] + 40:
            W.append((pi, s, rng.uniform(0.85, 1.15), rng.normal(0, 0.5), rng.choice([-1, 1]), rng.uniform(0, 6.28)))
            s += rng.exponential(dens)
    W = np.array(W, np.float64)
    _P['W'] = W
    # crowd spots for arrivals and the standing crowd
    n = W.shape[0]
    ends = np.array([paths[int(i)][-1] for i in W[:, 0]])
    ea = np.arctan2(ends[:, 1], ends[:, 0])
    cr = 9.0 + np.clip(rng.exponential(2.6, n), 0, 9)
    ca = ea + rng.normal(0, 1, n) * (0.10 + 1.4 / cr)
    _P['crowd'] = np.stack([cr * np.cos(ca), cr * np.sin(ca)], -1)
    m = 300
    r2 = 9.0 + np.clip(rng.exponential(2.2, m), 0, 8)
    a2 = rng.uniform(0, 2 * np.pi, m)
    _P['stand'] = np.stack([r2 * np.cos(a2), r2 * np.sin(a2)], -1)
    _P['stand_ph'] = rng.uniform(0, 6.28, m)


def walker_state(t):
    """Returns positions (n,2), headings (n,), torch positions (n,3), trail start (n,3)."""
    W = _P['W']
    paths = _P['paths']
    Ls = _P['L']
    D = _dist(t)
    trail_dt = 7.0 + 5.0 * (1 - smooth(ramp(t, 1930, 1980)))
    Dp = _dist(t - trail_dt)
    pos = np.zeros((W.shape[0], 2))
    posp = np.zeros((W.shape[0], 2))
    head = np.zeros(W.shape[0])
    hidden = np.zeros(W.shape[0], bool)
    for k in range(W.shape[0]):
        pi = int(W[k, 0])
        p = paths[pi]
        L = Ls[pi]
        for (dd, out) in ((D, 0), (Dp, 1)):
            s = W[k, 1] + W[k, 2] * dd
            if s < 0:
                if out == 0:
                    hidden[k] = True
                s = 0.0
            if s <= L[-1]:
                x = np.interp(s, L, p[:, 0])
                y = np.interp(s, L, p[:, 1])
                if out == 0:
                    s2 = min(s + 1.0, L[-1])
                    x2 = np.interp(s2, L, p[:, 0])
                    y2 = np.interp(s2, L, p[:, 1])
                    head[k] = math.atan2(y2 - y, x2 - x)
                    # lateral offset
                    nx, ny = -(y2 - y), (x2 - x)
                    nl = math.hypot(nx, ny) + 1e-9
                    x += W[k, 3] * nx / nl
                    y += W[k, 3] * ny / nl
            else:
                # arrived: walk to crowd spot over ~6 m then stand
                over = s - L[-1]
                e = p[-1]
                c = _P['crowd'][k]
                u = min(over / (np.hypot(*(c - e)) + 1e-6), 1.0)
                x, y = e + (c - e) * u
                if out == 0:
                    head[k] = math.atan2(-y, -x) if u >= 1 else math.atan2(c[1] - e[1], c[0] - e[0])
            if out == 0:
                pos[k] = (x, y)
            else:
                posp[k] = (x, y)
    pos[hidden] = 1e5
    posp[hidden] = 1e5
    # standing crowd
    st = _P['stand']
    sw = 0.03 * np.stack([np.sin(t * 0.05 + _P['stand_ph']), np.cos(t * 0.04 + _P['stand_ph'])], -1)
    pos = np.concatenate([pos, st + sw], 0)
    posp = np.concatenate([posp, st + sw], 0)
    head = np.concatenate([head, np.arctan2(-st[:, 1], -st[:, 0])])
    side = np.concatenate([W[:, 4], np.where(_P['stand_ph'] > 3.14, 1.0, -1.0)])
    off = 0.28
    tx = pos[:, 0] + side * off * -np.sin(head) * -1 + 0.15 * np.cos(head)
    ty = pos[:, 1] + side * off * np.cos(head) * -1 + 0.15 * np.sin(head)
    tpx = posp[:, 0] + side * off * np.sin(head) + 0.15 * np.cos(head)
    tpy = posp[:, 1] - side * off * np.cos(head) + 0.15 * np.sin(head)
    torch = np.stack([tx, ty, np.full_like(tx, 1.95)], -1)
    torchp = np.stack([tpx, tpy, np.full_like(tx, 1.95)], -1)
    return pos, head, torch, torchp, side


# ------------------------------------------------------------ irradiance ---

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


_IRR = {}


def irradiance(t):
    load()
    key = round(t * 100)
    if key in _IRR:
        return _IRR[key]
    pos, head, torch, torchp, side = walker_state(t)
    Nc = int(2 * IGC_R / IGC_CELL)
    Nf = int(2 * IGF_R / IGF_CELL)
    igc = np.zeros((Nc, Nc), np.float32)
    igf = np.zeros((Nf, Nf), np.float32)
    fl = 1.0 + 0.1 * np.sin(t * 1.7 + np.arange(torch.shape[0]))
    P = torch[:, :2].copy()
    I = 0.30
    _splat_irr(igc, -IGC_R, IGC_CELL, P, P.shape[0], I, 1.95, 7.0)
    near = np.nonzero((np.abs(P[:, 0]) < IGF_R + 6) & (np.abs(P[:, 1]) < IGF_R + 6))[0]
    Pn = P[near].copy()
    _splat_irr(igf, -IGF_R, IGF_CELL, Pn, Pn.shape[0], I, 1.95, 6.0)
    _IRR.clear()
    _IRR[key] = (igc, igf, (-IGC_R, IGC_CELL, -IGF_R, IGF_CELL))
    return _IRR[key]


# ------------------------------------------------------------------- draw ---

MIST = np.array([
    # z, opacity, scale, drift_x, drift_y, seed
    [270.0, 0.46, 150.0, 0.05, 0.02, 71],
    [150.0, 0.38, 95.0, -0.04, 0.03, 72],
    [70.0, 0.30, 60.0, 0.03, -0.02, 73],
], np.float64)


def draw(rgb, depth, cam, t, scale, band):
    load()
    if cam[2] > 26.0 or t < 2000:
        pos, head, torch, torchp, side = walker_state(t)
        n = pos.shape[0]
        # bodies (seen from above) - only when they are a few px big
        Q = np.zeros((n, 9))
        Q[:, 0:2] = pos
        Q[:, 2] = 1.3
        Q[:, 3] = head
        Q[:, 4] = 0.5
        Q[:, 5] = 0.6
        Q[:, 6] = 0.92
        Q[:, 7] = 0.35
        Q[:, 8] = side
        FI.dark_sprites(rgb, depth, cam, Q, n)
        # torch flames with time-lapse trails
        fl = 0.85 + 0.15 * np.sin(t * 2.3 + np.arange(n) * 1.7)
        col = (0.55 * FIRE_HOT + 0.45 * FIRE_CORE)[None, :] * (20.0 * fl)[:, None]
        S = np.zeros((n, 11))
        S[:, 0:3] = torchp
        S[:, 3:6] = torch
        S[:, 6] = 0.07
        S[:, 7:10] = col
        S[:, 10] = 0.0
        FI.splat_streaks(rgb, depth, cam, S, n, 0.0, 10.0, band, 1.0)
        # a soft halo so rivers read from altitude
        Hs = np.zeros((n, 8))
        Hs[:, 0:3] = torch
        Hs[:, 3] = 0.9
        Hs[:, 4:7] = FIRE_MID[None, :] * 0.035
        FI.splat_blobs(rgb, depth, cam, Hs, n, 0.3, band)
    if cam[2] > 40.0:
        igc = irradiance(t)[0]
        FI.mist_layers(rgb, cam, MIST, MIST.shape[0], t, igc, -IGC_R, IGC_CELL, 0.25)
