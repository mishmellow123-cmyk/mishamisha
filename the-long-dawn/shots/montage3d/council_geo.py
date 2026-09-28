"""THE COUNCIL in Cycles (lane COUNCIL-C): numpy-only geometry, used by councilc.py inside Blender.

Every builder returns (V, faces, mats) or a dict of them: V (n, 3) float64, faces = list of int arrays (quads or
tris), mats = per-face material slot. No bpy here, so the shapes can be checked in the venv too.

Figures are built from ACCORD's figure rows (geom3's F layout, from scene3.figures / crowd3.state), so the staging,
poses, torch hands and walk-out are ACCORD's own. The look is new: heavy wool cloaks that hang from the shoulders in
deep, rounded folds and break on the ground with a thick turned hem; a deep hood with a rolled rim and a void where
the face would be (never a lit face); a mantle, or her red woven shawl; the torch arm in a loose wool sleeve that
droops and bunches at the elbow. Local frame of a figure: x forward, y left, z up (geom3).
Material slots: 0 wool (colour 1), 1 wool (colour 2: mantle, hood), 2 void (the dark inside), 3 her shawl.
"""
import math

import numpy as np

# geom3 figure-row layout (duplicated: this module needs neither numba nor the accord lane)
F_X, F_Y, F_ANG, F_C, F_S, F_HS, F_WS, F_TYPE, F_SEED = 0, 1, 2, 3, 4, 5, 6, 7, 8
F_SIDE = 12
F_HX, F_TX, F_EX = 13, 16, 19
F_BOW = 22
F_TLIT = 29
F_BREATH = 30
F_CAPE, F_TRAIN, F_GILT = 34, 35, 36
F_TORCH, F_KNEEL, F_LEAN, F_ARM2 = 39, 40, 41, 42
F_E2X, F_H2X = 43, 46
F_SHAWL, F_SWAY, F_HEM, F_PHASE = 51, 52, 53, 54
F_SHX, F_S2X = 55, 58
F_HEROHAND = 64

M_WOOL1, M_WOOL2, M_VOID, M_SHAWL, M_LEATHER = 0, 1, 2, 3, 4


def sst(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Mesh:
    """Accumulates vertices and faces (quads/tris) with a material slot per face and per-vertex float attributes."""

    def __init__(self):
        self.V = []
        self.F = []
        self.M = []
        self.n = 0
        self.attr = {}

    def add_grid(self, P, mat, wrap=True, attrs=None, flip=False):
        """P (nv, nu, 3): rows of a lofted surface; quads between rows (u wraps if wrap)."""
        nv, nu = P.shape[0], P.shape[1]
        base = self.n
        self.V.append(P.reshape(-1, 3))
        self._attrs(attrs, nv * nu)
        cols = nu if wrap else nu - 1
        j, i = np.meshgrid(np.arange(nv - 1), np.arange(cols), indexing='ij')
        i2 = (i + 1) % nu
        a = base + j * nu + i
        b = base + j * nu + i2
        c = base + (j + 1) * nu + i2
        d = base + (j + 1) * nu + i
        q = np.stack([a, b, c, d], -1).reshape(-1, 4)
        if flip:
            q = q[:, ::-1]
        mm = np.full(len(q), mat, np.int32) if np.isscalar(mat) else np.asarray(mat, np.int32).reshape(-1)
        self.F.append(q)
        self.M.append(mm)
        self.n += nv * nu
        return base

    def add_fan(self, ring_start, nu, center, mat, flip=False, attrs=None):
        """Close a ring of nu vertices (starting at index ring_start) to a new centre vertex."""
        c = self.n
        self.V.append(np.asarray(center, np.float64).reshape(1, 3))
        self._attrs(attrs, 1)
        self.n += 1
        i = np.arange(nu)
        t = np.stack([ring_start + i, ring_start + (i + 1) % nu, np.full(nu, c)], -1)
        if flip:
            t = t[:, ::-1]
        self.F.append(t)
        self.M.append(np.full(nu, mat, np.int32))

    def _attrs(self, attrs, n):
        attrs = attrs or {}
        for k in set(self.attr) | set(attrs):
            if k not in self.attr:
                self.attr[k] = [np.zeros(self.n)]
            v = attrs.get(k)
            self.attr[k].append(np.zeros(n) if v is None else np.asarray(v, np.float64).reshape(-1))

    def arrays(self):
        V = np.concatenate(self.V, 0) if self.V else np.zeros((0, 3))
        attrs = {k: np.concatenate(v) for k, v in self.attr.items()}
        return V, self.F, np.concatenate(self.M) if self.M else np.zeros(0, np.int32), attrs


def faces_flat(F):
    """(loop_start, loop_total, vertex indices) for bpy's foreach_set from a list of (m, k) arrays."""
    tot = np.concatenate([np.full(len(f), f.shape[1], np.int32) for f in F])
    verts = np.concatenate([f.reshape(-1) for f in F]).astype(np.int32)
    start = np.concatenate([[0], np.cumsum(tot)[:-1]]).astype(np.int32)
    return start, tot, verts


# ================================================================ figures ===

def fig_zsh(F):
    return 1.33 * F[F_HS] - 0.43 * F[F_HS] * F[F_KNEEL]


def _lean_warp(P, F):
    """geom3's pose warp, body -> world: the upper body pitched forward about the hip, the walking sway."""
    zsh = fig_zsh(F)
    zhip = 0.62 * zsh
    lean = F[F_LEAN]
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    y = y - F[F_SWAY] * sst(0.35 * zsh, zsh, z)
    if abs(lean) > 1e-6:
        lz = lean * sst(zhip - 0.20, zhip + 0.25, z)
        rz = z - zhip
        xw = np.cos(lz) * x + np.sin(lz) * rz
        zw = -np.sin(lz) * x + np.cos(lz) * rz + zhip
        x, z = xw, zw
    return np.stack([x, y, z], -1)


def _radii(F, z, c):
    """The cloak's hanging section at height z: front/back semi-axis (by the sign of cos(theta)) and side."""
    hs = F[F_HS]
    ws = F[F_WS] * (1.0 + F[F_BREATH])
    zsh = fig_zsh(F)
    t = np.clip(z / zsh, 0.0, 1.0)
    fl = (1.0 - t) ** 1.6
    pool = F[F_KNEEL] * (1.0 - t) ** 3.0
    a_f = (0.150 + 0.120 * fl + 0.12 * pool) * ws
    a_b = (0.165 + 0.130 * fl + F[F_TRAIN] * fl ** 1.6 + 0.16 * pool) * ws
    b = (0.212 + 0.150 * fl + 0.07 * pool) * ws
    a = a_f * (0.5 + 0.5 * np.tanh(6.0 * c)) + a_b * (0.5 - 0.5 * np.tanh(6.0 * c))
    return a, b, t, hs, ws


def _folds(theta, z, t, seed, nf, amp_top=0.004, amp_bot=0.062, ws=1.0):
    """Heavy wool: broad rounded ridges, tighter valleys, deepening toward the hem, wandering a little with height."""
    w = nf * theta + seed + 1.3 * np.sin(2.3 * z + seed) + 0.6 * np.sin(3.0 * theta + 1.7 * seed)
    f = (np.sin(w) + 0.42 * np.sin(2.37 * w + 1.3 * seed + 1.7 * z) + 0.22 * np.sin(0.53 * w + 2.9 * seed)) / 1.45
    g = np.where(f > 0, np.abs(f) ** 0.75, -np.abs(f) ** 1.35)
    famp = (amp_top + amp_bot * (1.0 - t) ** 1.25) * ws
    return famp * g, g


def figure(F, lod=1.0, torch=True, fist=False):
    """One cloaked figure from its F row -> Mesh in LOCAL coordinates (apply fig_to_world)."""
    m = Mesh()
    hs = F[F_HS]
    ws = F[F_WS] * (1.0 + F[F_BREATH])
    seed = float(F[F_SEED])
    typ = int(F[F_TYPE])
    zsh = fig_zsh(F)
    side = F[F_SIDE] if F[F_SIDE] != 0 else 1.0
    nf = 5.0 + (seed * 3.7) % 2.5
    nu = max(24, int(round(72 * lod / 4.0)) * 4)
    nv = max(9, int(round(30 * lod)))
    th = np.linspace(0.0, 2 * np.pi, nu, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    # ---- the hem: resting on the ground, lifted a little in places; the train lies flat behind
    zh = 0.010 + 0.024 * (0.5 + 0.5 * np.sin(3.0 * th + seed)) * (1.0 - sst(-0.2, -0.6, c)) + 0.006 * np.sin(7 * th + 2 * seed)
    zh = np.maximum(zh, 0.004)
    sv = (np.arange(nv) / (nv - 1.0)) ** 1.3
    Z = zh[None, :] + (zsh - zh[None, :]) * sv[:, None]
    a, b, t, _, _ = _radii(F, Z, c[None, :])
    nx, ny = c[None, :] / a, s[None, :] / b
    nn = np.sqrt(nx * nx + ny * ny)
    nx, ny = nx / nn, ny / nn
    d, g = _folds(th[None, :], Z, t, seed, nf, ws=ws)
    X = a * c[None, :] + nx * d
    Y = b * s[None, :] + ny * d
    # the walk swings the hem
    hem = F[F_HEM] * math.sin(F[F_PHASE]) * (1.0 - t) ** 2.2
    X = X + hem * (0.6 + 0.4 * c[None, :])
    # cloth breaking on the ground flares out a touch in the lowest rows
    brk = (1.0 - sst(0.0, 0.08, Z - zh[None, :]))
    X = X + brk * 0.018 * c[None, :]
    Y = Y + brk * 0.018 * s[None, :]
    body = np.stack([X, Y, Z], -1)
    # ---- the turned hem (thickness) and the dark inside
    r0 = np.stack([X[0], Y[0]], -1)
    rn = np.stack([nx[0], ny[0]], -1)
    h1 = np.concatenate([r0 - rn * 0.013, zh[:, None]], -1)
    h2 = np.concatenate([r0 * 0.93 - rn * 0.03, (zh + 0.12)[:, None]], -1)
    # ---- the shoulder cap (hidden under hood and mantle), closing to the neck
    ncap = 4
    a_sh, b_sh = a[-1], b[-1]
    caps = []
    for k in range(1, ncap + 1):
        u = 0.5 * np.pi * k / ncap
        rx = 0.062 + (a_sh - 0.062) * np.cos(u) ** 0.8
        ry = 0.062 + (b_sh - 0.062) * np.cos(u) ** 0.8
        caps.append(np.stack([rx * c, ry * s, np.full(nu, zsh + 0.105 * hs * np.sin(u))], -1))
    grid = np.concatenate([h2[None], h1[None], body, np.stack(caps, 0)], 0)
    hemf = np.concatenate([np.zeros((2, nu)), np.clip((Z - zh[None, :]) / 0.25, 0, 1), np.ones((ncap, nu))], 0)
    ridge = np.concatenate([np.zeros((2, nu)), np.clip(g, -1, 1), np.zeros((ncap, nu))], 0)
    mats = np.full((grid.shape[0] - 1, nu), M_WOOL1, np.int32)
    base = m.add_grid(grid, mats, attrs={'hem': hemf, 'ridge': ridge})
    m.add_fan(base, nu, (0.0, 0.0, 0.10), M_VOID, flip=True)                  # the dark inside (under the hem)
    m.add_fan(base + (grid.shape[0] - 1) * nu, nu, (0.0, 0.0, zsh + 0.11 * hs), M_WOOL1,
              attrs={'hem': [1.0]})
    # ---- mantle / shawl
    shawl = F[F_SHAWL] > 0.5
    drop = F[F_CAPE]
    if drop > 0.0 or shawl:
        _mantle(m, F, th, c, s, zsh, hs, ws, seed, nf, drop, shawl, side, lod)
    # ---- hood (+ its collar over the shoulders)
    _hood(m, F, zsh, hs, ws, seed, typ, lod)
    # ---- the torch arm, the second arm
    if torch and F[F_TORCH] > 0.5:
        S, E, Hh = F[F_SHX:F_SHX + 3], F[F_EX:F_EX + 3], F[F_HX:F_HX + 3]
        short = 0.11 if F[F_HEROHAND] > 0.5 else 0.085
        _sleeve(m, S, E, Hh, seed, lod, short=short)
        if fist:
            Q = fist_blob(F)
            base = m.add_grid(Q[1:-1], M_LEATHER, attrs={'hem': np.ones(Q[1:-1].shape[:2])})
            m.add_fan(base, Q.shape[1], Q[0, 0], M_LEATHER, flip=True, attrs={'hem': [1.0]})
            m.add_fan(base + (Q.shape[0] - 3) * Q.shape[1], Q.shape[1], Q[-1, 0], M_LEATHER, attrs={'hem': [1.0]})
    if F[F_ARM2] > 0.5:
        S, E, Hh = F[F_S2X:F_S2X + 3], F[F_E2X:F_E2X + 3], F[F_H2X:F_H2X + 3]
        _sleeve(m, S, E, Hh, seed + 3.0, lod, short=0.075)
    V, Fc, Mt, attrs = m.arrays()
    # the pose warp applies to the cloak and hood (the arms were placed in the leaned pose by scene3)
    nwarp = getattr(m, 'nwarp', len(V))
    V[:nwarp] = _lean_warp(V[:nwarp], F)
    return V, Fc, Mt, attrs


def _mantle(m, F, th, c, s, zsh, hs, ws, seed, nf, drop, shawl, side, lod):
    nu = len(th)
    nvm = max(6, int(round(14 * lod)))
    if shawl:
        drop = max(drop, 0.26 * hs)
        back = sst(0.0, -0.55, c) * sst(0.55, 0.0, np.abs(s))
        front = sst(0.0, 0.5, c) * sst(0.15, 0.55, np.abs(s))
        dth = drop * (0.62 + 0.95 * back + 0.40 * front)
    else:
        dth = drop * (1.0 + 0.25 * sst(0.0, 0.2, s * side)) * (1.0 + 0.10 * np.sin(2 * th + seed))
    rows = []
    for k in range(nvm):
        v = k / (nvm - 1.0)
        if v < 0.28:
            u = v / 0.28                                        # over the shoulder cap: neck -> shoulder
            ang = 0.5 * np.pi * (1.0 - u)
            a_sh, b_sh, _, _, _ = _radii(F, np.full(nu, zsh), c)
            rx = 0.075 + (a_sh + 0.022 - 0.075) * np.cos(ang) ** 0.8
            ry = 0.075 + (b_sh + 0.022 - 0.075) * np.cos(ang) ** 0.8
            z = zsh + 0.115 * hs * np.sin(ang)
            X, Y = rx * c, ry * s
            rows.append(np.stack([X, Y, np.broadcast_to(z, X.shape)], -1))
        else:
            u = (v - 0.28) / 0.72
            z = zsh - dth * u
            a, b, t, _, _ = _radii(F, z, c)
            a = a + 0.024 + 0.035 * u
            b = b + 0.024 + 0.030 * u
            nx, ny = c / a, s / b
            nn = np.sqrt(nx * nx + ny * ny)
            nx, ny = nx / nn, ny / nn
            w = (nf + 2.0) * th + 0.7 * seed + 0.9 * np.sin(4.0 * z + seed)
            f = np.sin(w) + 0.45 * np.sin(2.3 * w + seed)
            gg = np.where(f > 0, np.abs(f / 1.45) ** 0.7, -np.abs(f / 1.45) ** 1.3)
            amp = (0.004 + (0.030 if not shawl else 0.012) * u ** 1.1) * ws
            X = a * c + nx * amp * gg
            Y = b * s + ny * amp * gg
            rows.append(np.stack([X, Y, z], -1))
    P = np.stack(rows, 0)
    last = P[-1]
    nrm = _unit(np.stack([last[:, 0], last[:, 1], np.zeros(nu)], -1))
    inner = last - nrm * 0.010 + np.array([0.0, 0.0, 0.03])
    P = np.concatenate([P, inner[None]], 0)
    mat = M_SHAWL if shawl else M_WOOL2
    m.add_grid(P, mat, attrs={'hem': np.ones(P.shape[:2]), 'ridge': np.zeros(P.shape[:2])})


def _hood(m, F, zsh, hs, ws, seed, typ, lod):
    """A deep hood: a shell round the head about the forward axis with the face opening cut out, its upper rim
    drawn forward over the face, the rim rolled in (thick cloth), the inside a void; a peak at the back; its collar
    of cloth falls onto the shoulders."""
    na = max(8, int(round(18 * lod)))
    nb = max(20, int(round(44 * lod / 4.0)) * 4)
    kk = {0: 1.00, 1: 1.02, 3: 1.10, 5: 1.08}.get(typ, 1.0) * (0.94 + 0.06 * hs)
    pk = {0: 0.030, 1: 0.070, 3: 0.012, 5: 0.050}.get(typ, 0.04)
    brim = {0: 0.050, 1: 0.060, 3: 0.035, 5: 0.075}.get(typ, 0.05)
    rx, ry, rz = 0.140 * kk, 0.128 * kk, 0.152 * kk
    a0 = 0.92 + 0.08 * math.sin(seed)
    al = a0 + (np.pi - 0.06 - a0) * (np.arange(na) / (na - 1.0)) ** 1.1
    be = np.linspace(0.0, 2 * np.pi, nb, endpoint=False)
    A, B = np.meshgrid(al, be, indexing='ij')
    dx, dy, dz = np.cos(A), np.sin(A) * np.cos(B), np.sin(A) * np.sin(B)
    r = 1.0 / np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2 + (dz / rz) ** 2)
    r = r + pk * np.exp(-((A - 2.45) / 0.38) ** 2) * np.clip(np.sin(B), 0, 1) ** 3
    near = sst(a0 + 0.75, a0, A)                                   # 1 at the rim
    r = r + 0.0055 * np.sin(8.0 * B + seed) * near + 0.004 * np.sin(5.0 * B + 2.0 * A + seed)
    r = r * (1.0 - 0.16 * near * np.clip(-np.sin(B), 0, 1))          # the lower rim closes round the throat
    X, Y, Zh = r * dx, r * dy, r * dz
    up = np.clip(np.sin(B), 0, 1) ** 1.4
    X = X + brim * near * up                                       # the upper rim drawn out over the face
    Zh = Zh - 0.022 * near * up
    Zh = Zh - 0.035 * sst(1.6, 2.9, A) * np.clip(-np.sin(B) + 0.3, 0, 1)   # the back hangs toward the nape
    shell = np.stack([X, Y, Zh], -1)
    rim = shell[0]
    ctr = np.array([0.02, 0.0, -0.01])
    inward = _unit(ctr - rim)
    in1 = rim + inward * 0.016 + np.array([-0.012, 0.0, 0.0])
    in2 = rim + inward * 0.045 + np.array([-0.050, 0.0, 0.0])
    P = np.concatenate([in2[None], in1[None], shell], 0)
    mats = np.full((P.shape[0] - 1, nb), M_WOOL2, np.int32)
    mats[0] = M_VOID
    hood_col = M_WOOL1 if F[F_SHAWL] > 0.5 else M_WOOL2
    mats[1:] = hood_col
    # head frame -> body: the head centre sits 0.12 hs above the pivot, bowed about the pivot
    Pv = np.array([0.012, 0.0, zsh + 0.07 * hs])
    Hc = np.array([0.015, 0.0, 0.12 * hs])
    bow = F[F_BOW]
    cb, sb = math.cos(bow), math.sin(bow)

    def to_body(Q):
        q = Q + Hc
        xr = cb * q[..., 0] + sb * q[..., 2]
        zr = -sb * q[..., 0] + cb * q[..., 2]
        return np.stack([xr + Pv[0], q[..., 1], zr + Pv[2]], -1)

    base = m.add_grid(to_body(P), mats, attrs={'hem': np.ones(P.shape[:2]), 'ridge': np.zeros(P.shape[:2])})
    m.add_fan(base + (P.shape[0] - 1) * nb, nb, to_body(np.array([-(rx + 0.01), 0.0, -0.02])), hood_col,
              attrs={'hem': [1.0]})
    # the void: a dark head inside, so nothing is ever seen through the opening
    hv = _ellipsoid(np.array([0.0, 0.0, -0.005]), (0.085, 0.078, 0.10), 12, 16)
    base = m.add_grid(to_body(hv[1:-1]), M_VOID)
    m.add_fan(base, 16, to_body(hv[0, 0]), M_VOID, flip=True)
    m.add_fan(base + (hv.shape[0] - 3) * 16, 16, to_body(hv[-1, 0]), M_VOID)
    # the collar: the hood's cloth falls from round the face onto the shoulders
    if typ != 3:
        nc = max(5, int(round(9 * lod)))
        rows = []
        for k in range(nc):
            v = k / (nc - 1.0)                                     # 0 = top (neck), 1 = on the shoulders
            z = zsh + 0.16 * hs - v * 0.27 * hs
            rc = 0.095 + (0.215 * ws - 0.095) * v ** 0.8
            ang = be
            fcp = (0.010 + 0.034 * v) * np.sin(7.0 * ang + 1.3 * seed + 2.0 * v) \
                + 0.012 * v * np.sin(3.0 * ang + 2.2 * seed)
            rr = rc + fcp
            X = (rr * np.cos(ang)) * 0.90 - 0.012
            Y = rr * np.sin(ang)
            rows.append(np.stack([X, Y, np.full(nb, z)], -1))
        P = np.stack(rows, 0)
        last = P[-1]
        nrm = _unit(np.stack([last[:, 0], last[:, 1], np.zeros(nb)], -1))
        P = np.concatenate([P, (last - nrm * 0.010 + np.array([0, 0, 0.025]))[None]], 0)
        m.add_grid(P, hood_col, attrs={'hem': np.ones(P.shape[:2]), 'ridge': np.zeros(P.shape[:2])})
    m.nwarp = m.n                                                   # everything so far takes the pose warp


def _ellipsoid(c, r, nv, nu):
    ph = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nv)
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    P, T = np.meshgrid(ph, th, indexing='ij')
    return np.stack([c[0] + r[0] * np.cos(P) * np.cos(T), c[1] + r[1] * np.cos(P) * np.sin(T),
                     c[2] + r[2] * np.sin(P)], -1)


def _frames_along(C):
    """Parallel-transport frames along a polyline C (n, 3): tangent, normal, binormal."""
    T = np.gradient(C, axis=0)
    T = _unit(T)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N = [_unit(np.cross(ref, T[0]))]
    for k in range(1, len(C)):
        n = N[-1] - T[k] * (N[-1] @ T[k])
        N.append(_unit(n))
    N = np.array(N)
    Bn = np.cross(T, N)
    return T, N, Bn


def _sleeve(m, S, E, Hh, seed, lod, short=0.085):
    """A loose wool sleeve from inside the shoulder over the elbow to a wide cuff a hand's length short of the
    grip; it bunches at the elbow and its underside hangs."""
    S, E, Hh = (np.asarray(v, np.float64) for v in (S, E, Hh))
    fa = Hh - E
    L2 = np.linalg.norm(fa)
    cuff = E + fa * max(0.0, (L2 - short) / (L2 + 1e-9))
    S0 = S + _unit(S - E) * 0.04
    n1, n2 = max(6, int(10 * lod)), max(6, int(10 * lod))
    u1 = np.linspace(0, 1, n1, endpoint=False)
    u2 = np.linspace(0, 1, n2)
    # a smooth path through the elbow (quadratic blend)
    C = np.concatenate([S0[None] + (E - S0)[None] * u1[:, None], E[None] + (cuff - E)[None] * u2[:, None]], 0)
    for _ in range(3):
        C[1:-1] = 0.25 * C[:-2] + 0.5 * C[1:-1] + 0.25 * C[2:]
    n = len(C)
    sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    s = sl / (sl[-1] + 1e-9)
    T, N, Bn = _frames_along(C)
    nu = max(10, int(round(20 * lod / 2.0)) * 2)
    be = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    rows = []
    for k in range(n):
        r = 0.082 - 0.016 * s[k] + 0.012 * sst(0.82, 1.0, s[k])     # wide at the cuff
        elb = math.exp(-((s[k] - 0.52) / 0.12) ** 2)
        rr = r + (0.004 + 0.008 * elb) * np.sin(3.0 * be + seed + 9.0 * s[k]) + 0.006 * elb * np.sin(6.0 * be + 2 * seed)
        ring = C[k][None] + rr[:, None] * (np.cos(be)[:, None] * N[k][None] + np.sin(be)[:, None] * Bn[k][None])
        # the underside hangs (gravity), more toward the cuff
        dn = ring[:, 2] - C[k][2]
        hang = (0.010 + 0.022 * s[k]) * np.clip(-dn / (rr + 1e-9), 0, 1) ** 2
        ring[:, 2] -= hang
        rows.append(ring)
    P = np.stack(rows, 0)
    last = P[-1]
    inner = C[-1] + (last - C[-1]) * 0.80 - T[-1] * 0.05
    P = np.concatenate([P, inner[None]], 0)
    mats = np.full((P.shape[0] - 1, nu), M_WOOL1, np.int32)
    mats[-1] = M_VOID
    m.add_grid(P, mats, attrs={'hem': np.ones(P.shape[:2]), 'ridge': np.zeros(P.shape[:2])})


def fig_to_world(V, F):
    c, s = math.cos(F[F_ANG]), math.sin(F[F_ANG])
    out = np.empty_like(V)
    out[:, 0] = F[F_X] + V[:, 0] * c - V[:, 1] * s
    out[:, 1] = F[F_Y] + V[:, 0] * s + V[:, 1] * c
    out[:, 2] = V[:, 2]
    return out


def local_to_world(P, F):
    return fig_to_world(np.atleast_2d(np.asarray(P, np.float64)), F)


def fist_blob(F):
    """A gloved fist round the torch shaft for far figures (local): a rounded lump at the grip."""
    Hh = F[F_HX:F_HX + 3]
    T = _unit(F[F_TX:F_TX + 3])
    E = F[F_EX:F_EX + 3]
    a = _unit((Hh - E) - T * ((Hh - E) @ T))
    b = np.cross(T, a)
    P = _ellipsoid(np.zeros(3), (0.045, 0.040, 0.050), 8, 12)
    Q = Hh + P[..., 0:1] * a + P[..., 1:2] * b + P[..., 2:3] * T
    return Q


# ================================================================ hearth ===

STONE_TOP = 0.30


def slab_outline(th):
    """The flat stone's edge radius: a weathered natural slab, irregular, with a few old chips (not a polygon)."""
    r = (0.372 + 0.040 * np.sin(2 * th + 0.9) + 0.026 * np.sin(3 * th + 2.1) + 0.015 * np.sin(5 * th + 0.3)
         + 0.008 * np.sin(9 * th + 1.7) + 0.004 * np.sin(17 * th + 0.4))
    for a, w, dd in ((0.72, 0.10, 0.030), (2.45, 0.07, 0.020), (3.30, 0.13, 0.045), (4.95, 0.09, 0.026),
                     (5.80, 0.05, 0.012)):
        dth = np.angle(np.exp(1j * (th - a)))
        r = r - dd * np.exp(-(dth / w) ** 2)
    return r


def slab_top(x, y):
    """Height of the slab's top: a natural cleft face, nearly flat, with a gentle tilt and shallow undulation."""
    return (STONE_TOP + 0.010 * x - 0.005 * y + 0.006 * np.sin(2.3 * x + 1.1) * np.sin(3.1 * y)
            + 0.003 * np.sin(7.0 * x + 2.0 * y + 0.5) + 0.002 * np.sin(11.0 * y - 5.0 * x))


def slab(n_r=26, n_t=160):
    """The flat stone: top face (polar grid), a rounded weathered edge, the side falling into the ground."""
    th = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    R = slab_outline(th)
    rows = []
    rr = np.linspace(0.0, 1.0, n_r) ** 0.8
    for k in range(1, n_r):
        f = rr[k] * (1.0 - 0.10 * rr[k] ** 6)                           # stop short of the edge (the bevel)
        x, y = f * R * np.cos(th), f * R * np.sin(th)
        rows.append(np.stack([x, y, slab_top(x, y)], -1))
    # the bevel and side: rounded over, weathered, falling to below the ground
    edge = []
    for k, (fr, dz) in enumerate(((0.935, -0.006), (0.965, -0.018), (0.985, -0.040), (1.0, -0.075),
                                  (1.01, -0.14), (1.02, -0.22), (1.03, -0.34), (1.00, -0.44))):
        x, y = fr * R * np.cos(th), fr * R * np.sin(th)
        zt = slab_top(x, y)
        wob = 0.010 * np.sin(6 * th + k) * (k >= 3) + 0.006 * np.sin(13 * th + 2 * k) * (k >= 2)
        x = x + wob * np.cos(th)
        y = y + wob * np.sin(th)
        edge.append(np.stack([x, y, zt + dz], -1))
    P = np.stack(rows + edge, 0)
    m = Mesh()
    base = m.add_grid(P, 0)
    m.add_fan(base, n_t, (0.0, 0.0, slab_top(0.0, 0.0)), 0, flip=True)
    return m.arrays()


def rock(center, radii, seed, nv=14, nu=22, rough=0.18, flat_bottom=None):
    """An irregular field stone: a lumpy ellipsoid, a few flattened faces (weathered, not a pebble)."""
    rng = np.random.default_rng(int(seed * 1000) % (2 ** 31))
    P = _ellipsoid(np.zeros(3), radii, nv, nu)
    d = _unit(P)
    k = np.zeros(P.shape[:2])
    for _ in range(4):                                  # flattened facets
        n = _unit(rng.normal(size=3))
        k = k + 0.10 * np.clip((d @ n) - 0.55, 0, 1)
    ph = rng.uniform(0, 6.28, 6)
    lump = (0.06 * np.sin(3 * d[..., 0] + ph[0]) * np.sin(2 * d[..., 1] + ph[1])
            + 0.04 * np.sin(5 * d[..., 2] + ph[2]) + 0.03 * np.sin(7 * d[..., 0] + 4 * d[..., 1] + ph[3]))
    P = P * (1.0 + rough * lump[..., None] - k[..., None])
    if flat_bottom is not None:
        P[..., 2] = np.maximum(P[..., 2], flat_bottom)
    P = P + np.asarray(center)
    m = Mesh()
    base = m.add_grid(P[1:-1], 0)
    m.add_fan(base, nu, P[0, 0], 0, flip=True)
    m.add_fan(base + (nv - 3) * nu, nu, P[-1, 0], 0)
    return m.arrays()


def log(A, B, ra, seed, burn_a=0.35, n_l=36, n_a=18):
    """A split log from A (the inner end, burnt down) to B: a wedge section (the split face, the bark side),
    knots, a crooked axis; the inner end tapers where it has burnt away."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    ax = B - A
    L = np.linalg.norm(ax)
    T = ax / L
    up = np.array([0.0, 0.0, 1.0])
    N = _unit(np.cross(up, T))
    Bn = np.cross(T, N)
    rng = np.random.default_rng(int(seed * 997) % (2 ** 31))
    roll = rng.uniform(0, 6.28)
    ph = rng.uniform(0, 6.28, 5)
    u = np.linspace(0, 1, n_l)
    be = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
    U, Bt = np.meshgrid(u, be, indexing='ij')
    # wedge: a rounded triangle (split faces) with the bark arc; burnt ends round everything
    wedge = 1.0 - 0.22 * np.abs(np.cos(1.5 * (Bt + roll))) ** 1.5
    taper = sst(0.0, burn_a, U) ** 0.55 * 0.75 + 0.25                  # the inner end burnt down
    taper = taper * (1.0 - 0.35 * sst(0.93, 1.0, U))                    # the sawn/broken outer end rounds off
    knot = 0.10 * np.exp(-((U - rng.uniform(0.3, 0.8)) / 0.05) ** 2) * np.clip(np.cos(Bt - ph[0]), 0, 1) ** 4
    lumpy = 0.05 * np.sin(9 * U + 3 * Bt + ph[1]) + 0.03 * np.sin(23 * U - 2 * Bt + ph[2])
    R = ra * wedge * taper * (1.0 + knot + lumpy)
    bend = 0.25 * ra * np.sin(np.pi * U * 1.3 + ph[3])
    C = A[None, None] + (U * L)[..., None] * T + bend[..., None] * N
    P = C + R[..., None] * (np.cos(Bt)[..., None] * N + np.sin(Bt)[..., None] * Bn)
    m = Mesh()
    ua = np.broadcast_to(U, P.shape[:2])
    base = m.add_grid(P, 0, attrs={'u': ua})
    m.add_fan(base, n_a, A - T * 0.002, 0, flip=True, attrs={'u': [0.0]})
    m.add_fan(base + (n_l - 1) * n_a, n_a, B + T * 0.002, 0, attrs={'u': [1.0]})
    return m.arrays()


def stick(A, B, r, seed, n_l=14, n_a=8):
    """A thin crooked stick of kindling with broken ends."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    ax = B - A
    L = np.linalg.norm(ax) + 1e-9
    T = ax / L
    N = _unit(np.cross(np.array([0.0, 0.0, 1.0]) + 1e-3, T))
    Bn = np.cross(T, N)
    rng = np.random.default_rng(int(seed * 991) % (2 ** 31))
    ph = rng.uniform(0, 6.28, 4)
    u = np.linspace(0, 1, n_l)
    be = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
    U, Bt = np.meshgrid(u, be, indexing='ij')
    bend = 0.6 * r * np.sin(np.pi * U * 2.1 + ph[0]) + 0.3 * r * np.sin(7 * U + ph[1])
    R = r * (1.0 - 0.25 * U) * (1.0 + 0.08 * np.sin(11 * U + 2 * Bt + ph[2]))
    C = A[None, None] + (U * L)[..., None] * T + bend[..., None] * N
    P = C + R[..., None] * (np.cos(Bt)[..., None] * N + np.sin(Bt)[..., None] * Bn)
    m = Mesh()
    base = m.add_grid(P, 0, attrs={'u': np.broadcast_to(U, P.shape[:2])})
    m.add_fan(base, n_a, A, 0, flip=True, attrs={'u': [0.0]})
    m.add_fan(base + (n_l - 1) * n_a, n_a, B, 0, attrs={'u': [1.0]})
    return m.arrays()


def chunk(center, size, seed):
    """A lump of charcoal: angular, cracked faces (never a pill)."""
    rng = np.random.default_rng(int(seed * 983) % (2 ** 31))
    P = _ellipsoid(np.zeros(3), (1.0, 1.0, 1.0), 9, 14)
    d = _unit(P)
    # a box-ish lump: pull toward a random cuboid with chipped facets
    Rm = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    q = d @ Rm
    box = 1.0 / np.max(np.abs(q) / np.array([1.0, 0.8, 0.55]), axis=-1)
    box = np.minimum(box, 1.25)
    for _ in range(5):
        n = _unit(rng.normal(size=3))
        box = box - 0.12 * np.clip(d @ n - 0.6, 0, 1)
    P = d * box[..., None] * np.asarray(size)
    P[..., 2] = np.maximum(P[..., 2], -0.35 * size[2])
    P = P + np.asarray(center)
    m = Mesh()
    base = m.add_grid(P[1:-1], 0)
    m.add_fan(base, 14, P[0, 0], 0, flip=True)
    m.add_fan(base + 6 * 14, 14, P[-1, 0], 0)
    return m.arrays()


# ============================================================ torch ===

def torch_mesh(n_a=14):
    """A torch in its own frame (the grip at the origin, +z along the shaft toward the head): a hand-cut stave and a
    head of cloth strips wound round it and soaked in pitch (lumpy, bulging, with the strip edges standing)."""
    m = Mesh()
    be = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
    zs = np.linspace(-0.30, 0.31, 16)
    rows = []
    for z in zs:
        r = 0.0155 + 0.0012 * math.sin(9 * z) + 0.0008 * np.sin(3 * be + 20 * z)
        rows.append(np.stack([r * np.cos(be), r * np.sin(be), np.full(n_a, z)], -1))
    base = m.add_grid(np.stack(rows, 0), 0, attrs={'head': np.zeros((len(zs), n_a))})
    m.add_fan(base, n_a, (0, 0, -0.303), 0, flip=True, attrs={'head': [0.0]})
    # the head: 0.29 .. 0.45
    nb = 28
    be2 = np.linspace(0, 2 * np.pi, nb, endpoint=False)
    zh = np.linspace(0.285, 0.452, 22)
    rows = []
    hv = []
    for z in zh:
        u = (z - 0.285) / 0.167
        prof = 0.018 + 0.024 * np.sin(np.pi * np.clip(u * 1.05, 0, 1)) ** 0.6
        wind = 0.0030 * np.cos(2 * np.pi * (u * 5.5) - be2)            # the strips spiral round
        lump = 0.0024 * np.sin(5 * be2 + 31 * z) + 0.0018 * np.sin(11 * be2 - 17 * z + 1.0)
        r = prof + wind + lump
        if u > 0.93:
            r = r * (1.0 - (u - 0.93) / 0.07 * 0.55)
        rows.append(np.stack([r * np.cos(be2), r * np.sin(be2), np.full(nb, z)], -1))
        hv.append(np.full(nb, 1.0))
    base = m.add_grid(np.stack(rows, 0), 1, attrs={'head': np.stack(hv, 0)})
    m.add_fan(base, nb, (0, 0, 0.283), 1, flip=True, attrs={'head': [1.0]})
    m.add_fan(base + 21 * nb, nb, (0, 0, 0.456), 1, attrs={'head': [1.0]})
    return m.arrays()


# ============================================================ standing stones ===

def standing_stone(S):
    """A standing stone from scene3.stones()'s row (geom3 S layout): unhewn, tapered, leaning, a tilted broken top,
    weathered lumps. World coordinates."""
    x, y, cy, sy, hx, hy, H = S[0], S[1], S[2], S[3], S[4], S[5], S[6]
    taper, lx, ly, tnx, tny, top, seed = S[7], S[8], S[9], S[10], S[11], S[12], S[13]
    rng = np.random.default_rng(int(seed * 1000) % (2 ** 31))
    ph = rng.uniform(0, 6.28, 8)
    nz, na = 30, 40
    be = np.linspace(0, 2 * np.pi, na, endpoint=False)
    rows = []
    for k in range(nz):
        u = k / (nz - 1.0)
        z = -0.25 + (top + 0.25) * u
        sc = 1.0 - taper * u ** 1.3
        ex, ey = hx * sc, hy * sc
        # a superellipse section (flattish faces, rounded arrises), lumpy
        c, s = np.cos(be), np.sin(be)
        px = ex * np.sign(c) * np.abs(c) ** 0.7
        py = ey * np.sign(s) * np.abs(s) ** 0.7
        lump = (0.05 * np.sin(3 * be + ph[0] + 2 * u) + 0.035 * np.sin(5 * be + ph[1] + 5 * u)
                + 0.02 * np.sin(9 * be + ph[2] + 9 * u) + 0.03 * np.sin(7 * u + ph[3]))
        px = px * (1 + lump)
        py = py * (1 + lump)
        # the top: tilted and broken
        zt = z + (tnx * px + tny * py) * sst(0.8, 1.0, u)
        # lean
        X = px + lx * z
        Y = py + ly * z
        rows.append(np.stack([X, Y, zt], -1))
    P = np.stack(rows, 0)
    Xw = x + P[..., 0] * cy - P[..., 1] * sy
    Yw = y + P[..., 0] * sy + P[..., 1] * cy
    P = np.stack([Xw, Yw, P[..., 2]], -1)
    m = Mesh()
    base = m.add_grid(P, 0)
    topc = P[-1].mean(0) + np.array([0, 0, 0.03])
    m.add_fan(base + (nz - 1) * na, na, topc, 0)
    return m.arrays()


# ============================================================ heather ===

def heather_tuft(seed, height=0.22, n_stems=18, flat=0.0):
    """A heather tuft: woody stems fanning up and out from a root crown, each clothed toward its tip in tiny scale
    leaves (thin tubes of foliage), a few old flower spikes. flat > 0: trampled (bent over, lying). Returns a Mesh
    with slots 0 = woody stem, 1 = foliage."""
    rng = np.random.default_rng(seed)
    m = Mesh()
    for k in range(n_stems):
        az = rng.uniform(0, 2 * np.pi)
        tilt = rng.uniform(0.15, 0.85) + flat * rng.uniform(0.6, 1.1)
        tilt = min(tilt, 1.45)
        L = height * rng.uniform(0.55, 1.15) / max(0.4, math.cos(min(tilt, 1.2)) ** 0.3)
        n = 7
        u = np.linspace(0, 1, n)
        curve = tilt * (0.6 + 0.6 * u)                        # stems arch outward
        droop = flat * 0.5 * u ** 2
        dirs = np.stack([np.sin(curve) * math.cos(az), np.sin(curve) * math.sin(az), np.cos(curve) - droop], -1)
        dirs = _unit(dirs)
        C = np.cumsum(dirs * (L / (n - 1)), 0) - dirs[0] * (L / (n - 1))
        C[:, 2] = np.maximum(C[:, 2], 0.004)
        C = C + np.array([rng.normal(0, 0.012), rng.normal(0, 0.012), 0.0])
        # woody stem
        _tube(m, C[:4], 0.0022, 0.0012, 5, 0)
        # foliage: a fuzzy, lumpy sleeve of tiny leaves along the upper stem
        r = 0.0055 * (1.0 - 0.5 * u[2:]) * rng.uniform(0.8, 1.2)
        _tube(m, C[2:], r, None, 6, 1, lumpy=seed + k)
    return m


def _tube(m, C, r0, r1, na, mat, lumpy=None):
    n = len(C)
    T, N, Bn = _frames_along(C)
    be = np.linspace(0, 2 * np.pi, na, endpoint=False)
    rows = []
    for k in range(n):
        if np.ndim(r0):
            r = r0[k]
        else:
            r = r0 + ((r1 if r1 is not None else r0) - r0) * k / max(1, n - 1)
        rr = r * np.ones(na)
        if lumpy is not None:
            rr = rr * (1.0 + 0.35 * np.sin(3 * be + lumpy + 2.7 * k) + 0.2 * np.sin(5 * be - 1.3 * k + 2 * lumpy))
        rows.append(C[k][None] + rr[:, None] * (np.cos(be)[:, None] * N[k][None] + np.sin(be)[:, None] * Bn[k][None]))
    P = np.stack(rows, 0)
    base = m.add_grid(P, mat)
    m.add_fan(base + (n - 1) * na, na, C[-1] + T[-1] * (r0 if not np.ndim(r0) else r0[-1]) * 0.8, mat)
