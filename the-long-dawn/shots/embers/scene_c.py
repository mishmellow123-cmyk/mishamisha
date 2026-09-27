"""EMBERS part C: the world as a cracking ember globe (880-960), the GRASP (960-1040),
and the SILENCE (1040-1199)."""
import json
import math
import os

import cv2
import numpy as np

import look
from numba import njit, prange

from core import (clamp01, smoothstep, smootherstep, ease_out, ease_in, ease_in_out, lerp, vnoise, snoise,
                  rng, rand_dirs, catmull, Camera)
import scene_b as B

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
C_CORE = look.hexrgb(look.PALETTE['mind_core'])
C_ICE = look.hexrgb(look.PALETTE['mind_ice'])
C_GOLD = look.hexrgb(look.PALETTE['mind_gold'])
C_RED = look.hexrgb(look.PALETTE['race_red'])
C_CRIMSON = look.hexrgb(look.PALETTE['race_crimson'])


# =================================================================== GLOBE ===

def land_mask(W=2048, H=1024):
    d = json.load(open(os.path.join(ROOT, 'assets', 'data', 'ne_110m_land.geojson')))
    m = np.zeros((H, W), np.uint8)
    for f in d['features']:
        for ring in f['geometry']['coordinates'][:1]:
            pts = np.array([[(lon + 180) / 360 * W, (90 - lat) / 180 * H] for lon, lat in ring], np.float64)
            cv2.fillPoly(m, [np.round(pts * 8).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=3)
        for ring in f['geometry']['coordinates'][1:]:
            pts = np.array([[(lon + 180) / 360 * W, (90 - lat) / 180 * H] for lon, lat in ring], np.float64)
            cv2.fillPoly(m, [np.round(pts * 8).astype(np.int32)], 0, lineType=cv2.LINE_8, shift=3)
    return m


def sph_to_ll(p):
    lat = np.degrees(np.arcsin(np.clip(p[:, 1], -1, 1)))
    lon = np.degrees(np.arctan2(-p[:, 2], p[:, 0]))
    return lat, lon


def lookup(mask, p):
    H, W = mask.shape
    lat, lon = sph_to_ll(p)
    x = np.clip(((lon + 180) / 360 * W).astype(int), 0, W - 1)
    y = np.clip(((90 - lat) / 180 * H).astype(int), 0, H - 1)
    return mask[y, x]


# v2 (framing review): v1 framed East Asia and parted its plates along a seam past Japan, Taiwan and the Philippines.
# Now the world is seen from above the Arctic, every continent on the rim together, turning; the fire starts in the
# high Arctic (no one's) and reaches every continent at once; the plates are laid out (globe_plates.py) so that no
# seam runs along or through any real flashpoint, strait or border, or through a capital or AI hub.
GLOBE_SEEDS = [(43.92, 115.42), (42.91, 84.42), (47.85, 36.19), (19.26, -15.71), (68.84, -67.97), (9.34, -67.68), (7.62, 1.05), (11.09, -43.35), (48.23, -56.11), (8.67, -136.92), (62.98, 105.24), (6.04, -171.20), (58.62, -156.96), (29.98, -167.58), (32.62, -125.47), (5.19, 21.64), (11.48, 54.70), (3.22, 74.37), (8.01, -102.37), (-30.00, -160.00), (-51.63, -120.00), (-38.26, -80.00), (-59.89, -40.00), (-46.52, 0.00), (-33.15, 40.00), (-54.78, 80.00), (-41.41, 120.00), (-63.04, 160.00)]
# the cells merged into plates (only seams between plates crack); see globe_plates.py
GLOBE_GROUPS = [0, 0, 1, 1, 1, 2, 3, 4, 2, 5, 6, 7, 8, 7, 5, 3, 3, 9, 2, 10, 5, 2, 5, 11, 3, 12, 9, 10]
POLE_TILT = math.radians(8.0)        # we look down from about 82 N
LON_NEAR = math.radians(-20.0)       # meridian nearest the lens at mid-shot (it turns eastward through the shot)
GLOBE_SPIN = 0.006                   # rad / frame (~27 degrees over the shot)


def ll2v(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), -np.cos(lat) * np.sin(lon)], -1)


def globe_basis():
    pos, _ = cam_globe(920.0)
    F = pos / np.linalg.norm(pos)
    Y = np.array([0.0, 1.0, 0.0])
    U = Y - (Y @ F) * F
    U /= np.linalg.norm(U)
    n_w = math.cos(POLE_TILT) * F + math.sin(POLE_TILT) * U
    e_b = F - (F @ n_w) * n_w
    e_b /= np.linalg.norm(e_b)
    a1 = np.array([math.cos(LON_NEAR), 0.0, -math.sin(LON_NEAR)])
    a2 = np.array([0.0, 1.0, 0.0])
    W = np.stack([e_b, n_w, np.cross(e_b, n_w)], 1)
    Aa = np.stack([a1, a2, np.cross(a1, a2)], 1)
    return W @ Aa.T


class Globe:
    def __init__(self, seed=202, groups=None):
        r = rng(seed)
        mask = land_mask()
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        P = rand_dirs(r, 900000)
        onland = lookup(mask, P) > 0
        self.land = P[onland][:260000]
        Q = rand_dirs(r, 1400000)
        oncoast = lookup(edge, Q) > 0
        self.coast = Q[oncoast][:60000]
        self.ocean = P[~onland][:26000]
        # plates (v2, third pass: globe_layout.py): irregular shields holding every flashpoint region whole, shards
        # toward the rim, seams that wander and branch; the v1 draws are kept for the random stream
        import globe_layout as GL
        _ = rand_dirs(r, 16)
        C = rand_dirs(r, 1600000)
        lab, lab2, gap, fis = GL.evaluate(C, chunk=100000)
        # every seam its own character: hairline to wide open, faint to roaring (v1: all alike)
        pr_all = GL.seam_pair(lab, lab2)
        up0, inv0 = np.unique(pr_all, return_inverse=True)
        rw = np.random.default_rng(405)
        width = rw.uniform(0.45, 1.35, len(up0))
        self._pair_hot = dict(zip(up0.tolist(), rw.uniform(0.35, 1.3, len(up0)).tolist()))
        main = gap < 2 * GL.SEAM_W * width[inv0]
        crack = main | fis
        self.crack = C[crack]
        self.c_main = main[crack]
        self.c_gap = gap[crack]
        pair = GL.seam_pair(lab, lab2)[crack]
        # each plate moves out along its own centroid; shards fly further than the shields, and all at their own time
        npl = GL.NP
        cnt = np.bincount(lab, minlength=npl).astype(float)
        cen = np.zeros((npl, 3))
        np.add.at(cen, lab, C)
        cen /= np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-9)
        self.gdir = cen
        rr_ = np.random.default_rng(404)
        area = cnt / cnt.sum()
        big = np.clip(area / 0.03, 0, 1)                                   # 1 for the shields
        self.sp_t0 = rr_.uniform(916.0, 934.0, npl)
        self.sp_mag = (0.11 - 0.06 * big) * rr_.uniform(0.85, 1.15, npl)
        # crack activation: the fire starts in the high Arctic and runs outward, reaching every continent at about
        # the same time; each seam segment has its own delay, and the fissures grow out of their seam after it
        _ = rand_dirs(r, 5)
        origins = ll2v(np.array([84.0, 84.0, 84.0]), np.array([0.0, 120.0, 240.0]))
        dg = np.degrees(np.arccos(np.clip(self.crack @ origins.T, -1, 1)).min(axis=1))
        up, inv = np.unique(pair, return_inverse=True)
        delay = rr_.uniform(-4.0, 18.0, len(up))[inv]
        self.c_hot = np.array([self._pair_hot.get(int(p_), 1.0) for p_ in up])[inv]
        grow = np.where(self.c_main, 0.0, 3.0 + 1.6 * (self.c_gap - 2 * GL.SEAM_W))
        self.t_act = (881.0 + 0.5 * dg + delay + grow + 3.0 * snoise(self.crack, 2.5, (1.0, 2.0, 3.0), 2)
                      + r.uniform(-2, 2, len(self.crack)))
        self.cell = {'crack': lab[crack]}
        self.c_lab2 = lab2[crack]
        for name in ('land', 'coast', 'ocean'):
            self.cell[name] = GL.evaluate(getattr(self, name), chunk=100000)[0]
        # fire spilling out of the cracks
        n = 60000
        self.sp_i = r.integers(0, len(self.crack), n)
        self.sp_ph = r.random(n)
        self.sp_v = r.uniform(0.004, 0.012, n)
        self.sp_E = r.lognormal(0, 0.6, n)
        self.rnd = {k: r.random(len(getattr(self, k))) for k in ('land', 'coast', 'ocean', 'crack')}
        print('globe: land', len(self.land), 'coast', len(self.coast), 'crack', len(self.crack))

    def _groups(self, n, target=13, seed=5):
        """agglomerate neighbouring Voronoi cells into irregular plates of 1-4 cells"""
        r = np.random.default_rng(seed)
        P = rand_dirs(r, 60000)
        dd = P @ self.seeds.T
        i2 = np.argsort(-dd, axis=1)[:, :2]
        adj = set(map(tuple, np.sort(i2, axis=1)))
        g = np.arange(n)
        size = np.ones(n, int)
        pairs = sorted(adj)
        while len(set(g)) > target:
            a, b = pairs[r.integers(0, len(pairs))]
            ga, gb = g[a], g[b]
            if ga == gb or size[ga] + size[gb] > 4:
                continue
            g[g == gb] = ga
            size[ga] += size[gb]
        _, g = np.unique(g, return_inverse=True)
        return g

    _M = None

    def rot(self, t):
        """v2: seen from above the Arctic (every continent on the rim), turning eastward"""
        if Globe._M is None:
            Globe._M = globe_basis()
        a = GLOBE_SPIN * (t - 920.0)
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        return Globe._M @ Ry

    def split(self, t):
        """per plate: each opens on its own time (v1: all at once)"""
        return self.sp_mag * ease_in_out((t - self.sp_t0) / 26.0)

    def world(self, name, t, radius=1.0):
        pts = getattr(self, name)
        c = self.cell[name]
        sp = self.split(t)
        off = self.gdir[c] * sp[c][:, None]
        if name == 'crack':
            # a seam's fire stays in the middle of the opening rift (riding with one plate it would outline every
            # plate twice, like the stitched panels of a ball); fissures ride their own plate
            c2 = self.c_lab2
            mid = 0.5 * (off + self.gdir[c2] * sp[c2][:, None])
            off = np.where(self.c_main[:, None], mid, off)
        return (pts * radius + off) @ self.rot(t).T

    def emit(self, ctx):
        t = ctx.t
        cam = ctx.cam
        vdir = cam.pos / np.linalg.norm(cam.pos)       # globe at origin
        heat = float(smoothstep(935, 959, t))
        flare = float(smoothstep(950, 959, t))

        def facing(P):
            n = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
            c = n @ vdir
            return smoothstep(-0.05, 0.25, c)
        # land embers
        P0 = self.world('land', ctx.t0)
        P1 = self.world('land', ctx.t1)
        f = facing(P1)
        rnd = self.rnd['land']
        e = (0.55 + 0.45 * np.sin(0.9 * t + 40 * rnd)) * (0.6 + 0.8 * rnd) * 1.1 * f * (1 + 2 * heat)
        col = look.blackbody(0.3 + 0.18 * rnd)
        ctx.fr.splat(P0, P1, 0.0015, e, col, ctx.cam0, ctx.cam1)
        # coastlines: brighter, gold
        P0 = self.world('coast', ctx.t0)
        P1 = self.world('coast', ctx.t1)
        f = facing(P1)
        e = 2.2 * f * (1 + heat)
        ctx.fr.splat(P0, P1, 0.0015, e, look.blackbody(0.55), ctx.cam0, ctx.cam1)
        # ocean: sparse deep-red embers
        P0 = self.world('ocean', ctx.t0)
        P1 = self.world('ocean', ctx.t1)
        f = facing(P1)
        ctx.fr.splat(P0, P1, 0.0015, 0.5 * f, C_CRIMSON, ctx.cam0, ctx.cam1)
        # cracks: hot front, then glowing seams (and the molten core showing through as plates part)
        act = smoothstep(self.t_act, self.t_act + 3.0, t)
        front = np.exp(-np.maximum(t - self.t_act, 0) / 5.0) * act
        P0 = self.world('crack', ctx.t0)
        P1 = self.world('crack', ctx.t1)
        f = facing(P1)
        # the fissures branching off the seams are finer and dimmer, and taper toward their dead ends
        fis_k = np.where(self.c_main, 1.0, 0.5 * np.clip(1.15 - (self.c_gap - 1.0) / 9.0, 0.25, 1.0)) * self.c_hot
        e = (2.6 * act + 9.0 * front) * f * (1 + 2.5 * flare) * fis_k
        col = look.blackbody(np.clip(0.62 + 0.35 * front + 0.2 * heat, 0, 1))
        ctx.fr.splat(P0, P1, 0.0015, e, col, ctx.cam0, ctx.cam1)
        spl = self.split(t)[self.cell['crack']]
        on = (spl > 0.002) & self.c_main
        if on.any():
            # the molten core showing through where each plate has begun to part
            Pc = (self.crack[on] * 0.965) @ self.rot(t).T
            fc = facing(Pc)
            ec = act[on] * fc * 6.0 * (spl[on] / 0.075) * (1 + 3 * flare)
            ctx.fr.splat(Pc, Pc, 0.002, ec, look.blackbody(0.9), ctx.cam0, ctx.cam1)
        # fire spilling out of the cracks, rising off the surface
        i = self.sp_i
        base = self.crack[i]
        ok = t > self.t_act[i] + 2

        def spill(tq):
            k = (self.sp_ph + self.sp_v * (tq - 880)) % 1.0
            p = base * (1.0 + 0.2 * k[:, None] ** 1.2)
            w = vnoise(p * 4.0 + np.array([0, 0.02 * tq, 0]), 1.0, (0, 0, 0), 1)
            p = p + w * (0.02 + 0.05 * k)[:, None]
            c_ = self.cell['crack'][i]
            off = self.gdir[c_] * self.split(tq)[c_][:, None]
            return (p + off) @ self.rot(tq).T, k
        S0, k0 = spill(ctx.t0)
        S1, k1 = spill(ctx.t1)
        f = smoothstep(-0.3, 0.2, (S1 / np.linalg.norm(S1, axis=1, keepdims=True)) @ vdir)
        e = self.sp_E * (1 - k1) ** 1.5 * 3.0 * ok * f * (k1 >= k0) * (1 + 2 * heat)
        col = look.blackbody(np.clip(0.85 - 0.5 * k1, 0.2, 1))
        ctx.fr.splat(S0, S1, 0.002, e, col, ctx.cam0, ctx.cam1)
        # halo of the burning world
        H = np.zeros((1, 3))
        ctx.fr.splat(H, H, np.array([1.9]), np.array([2600.0 * (0.4 + heat + 3 * flare)]),
                     np.array([C_RED * 0.7 + look.blackbody(0.6) * 0.3]), ctx.cam0, ctx.cam1, profile=1)


def cam_globe(t):
    u = (t - 880) / 80.0
    d = lerp(10.2, 8.6, float(ease_in_out(u)))
    a = 0.25 - 0.12 * u
    pos = np.array([d * math.sin(a), 1.2 + 0.2 * u, d * math.cos(a)])
    tgt = np.array([0.0, -0.18, 0.0])
    return pos, tgt


# ==================================================================== HAND ===
#
# One closed organic surface, not a kit of capsules. The hand is a signed distance field: a smooth union of
# tapered round-cone "bones" posed by forward kinematics (carpus, 4 metacarpals fanning out to the knuckle line,
# 4 fingers x 3 phalanges with knuckle bulges, a thumb whose metacarpal melts into the palm as the thenar mass,
# radius + ulna + a muscle belly for the forearm). Surface points are sampled once per bone, ride their bone,
# and are Newton-projected onto the union surface every frame. A soft ownership weight (a partition of unity
# over the bones) removes the doubled layers where bones meet, so no joint draws a ring. Only camera-facing
# points emit, cosine-weighted so that a glowing surface has uniform radiance (no limb brightening): the body
# is a near-black crust, and the light lives in a static Worley crack network on the skin (a coal bed), in
# burning rims and knuckles, and in embers shed off the edges. Coverage of the same surface builds the alpha
# that hides the world behind the hand.

# local frame (units of hand length L = wrist crease -> middle fingertip):
# wrist at origin, fingers +Y, palm faces +Z, back of hand -Z, thumb on -X
FINGERS = {  # name: (mcp x, mcp y, phalanx lengths, joint radii at mcp/pip/dip/tip)
    'index': (-0.128, 0.455, (0.205, 0.122, 0.090), (0.046, 0.040, 0.035, 0.029)),
    'middle': (-0.042, 0.468, (0.222, 0.138, 0.095), (0.048, 0.042, 0.036, 0.030)),
    'ring': (0.044, 0.457, (0.208, 0.130, 0.090), (0.045, 0.039, 0.034, 0.028)),
    'little': (0.125, 0.428, (0.166, 0.100, 0.084), (0.040, 0.034, 0.030, 0.025)),
}
FNAMES = ['index', 'middle', 'ring', 'little']
SPREAD_OPEN = {'index': -0.24, 'middle': -0.06, 'ring': 0.1, 'little': 0.28}
SPREAD_REST = {'index': -0.07, 'middle': -0.02, 'ring': 0.03, 'little': 0.09}
FLEN = 1.07                    # a touch longer and gaunter than life: menace
THUMB_L = (0.20, 0.15, 0.12)
THUMB_R = (0.064, 0.052, 0.044, 0.030)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _xf(p, R):
    """p @ R.T without BLAS (elementwise; much faster for (N,3) x (3,3))"""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([R[0, 0] * x + R[0, 1] * y + R[0, 2] * z,
                     R[1, 0] * x + R[1, 1] * y + R[1, 2] * z,
                     R[2, 0] * x + R[2, 1] * y + R[2, 2] * z], 1)


def _frame_y(d):
    """orthonormal frame (columns x, y, z) whose y column is along d"""
    d = np.asarray(d, np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.eye(3)
    y = d / n
    ref = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(y, ref)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z], 1)


# ------------------------------------------------------------ SDF kernels ---

@njit(fastmath=True, cache=True, inline='always')
def _cone_d(px, py, pz, j, A, BA, IL2, RA, DR):
    """distance to a tapered capsule (sphere-swept segment, radius RA..RA+DR) and its outward normal"""
    qx = px - A[j, 0]
    qy = py - A[j, 1]
    qz = pz - A[j, 2]
    h = (qx * BA[j, 0] + qy * BA[j, 1] + qz * BA[j, 2]) * IL2[j]
    if h < 0.0:
        h = 0.0
    elif h > 1.0:
        h = 1.0
    cx = qx - BA[j, 0] * h
    cy = qy - BA[j, 1] * h
    cz = qz - BA[j, 2] * h
    l = math.sqrt(cx * cx + cy * cy + cz * cz) + 1e-12
    return l - (RA[j] + DR[j] * h), cx / l, cy / l, cz / l


@njit(fastmath=True, cache=True)
def _sdf(px, py, pz, A, BA, IL2, RA, DR, GS, GE, KG, KJ):
    """smooth union within each group (blend KG[g]), then across groups (blend KJ); returns d, unit gradient"""
    D = 1e9
    GX = 0.0
    GY = 0.0
    GZ = 1.0
    for g in range(GS.shape[0]):
        k = KG[g]
        d = 1e9
        gx = 0.0
        gy = 0.0
        gz = 1.0
        for j in range(GS[g], GE[g]):
            dj, nx, ny, nz = _cone_d(px, py, pz, j, A, BA, IL2, RA, DR)
            if j == GS[g]:
                d = dj
                gx = nx
                gy = ny
                gz = nz
            else:
                hh = 0.5 + 0.5 * (dj - d) / k
                if hh < 0.0:
                    hh = 0.0
                elif hh > 1.0:
                    hh = 1.0
                d = dj + (d - dj) * hh - k * hh * (1.0 - hh)
                gx = nx + (gx - nx) * hh
                gy = ny + (gy - ny) * hh
                gz = nz + (gz - nz) * hh
        if g == 0:
            D = d
            GX = gx
            GY = gy
            GZ = gz
        else:
            hh = 0.5 + 0.5 * (d - D) / KJ
            if hh < 0.0:
                hh = 0.0
            elif hh > 1.0:
                hh = 1.0
            D = d + (D - d) * hh - KJ * hh * (1.0 - hh)
            GX = gx + (GX - gx) * hh
            GY = gy + (GY - gy) * hh
            GZ = gz + (GZ - gz) * hh
    l = math.sqrt(GX * GX + GY * GY + GZ * GZ) + 1e-12
    return D, GX / l, GY / l, GZ / l


@njit(parallel=True, fastmath=True, cache=True)
def _project(P, OWN, A, BA, IL2, RA, DR, GS, GE, KG, KJ, iters, tau, NRM, D0, WOWN):
    """Newton-project points onto the union surface (in place); normal at the result; the soft ownership weight
    of each point's own bone (softmin over all bones' raw distances -> the weights of all bones sum to 1)."""
    M = A.shape[0]
    for i in prange(P.shape[0]):
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
        d, gx, gy, gz = _sdf(px, py, pz, A, BA, IL2, RA, DR, GS, GE, KG, KJ)
        D0[i] = d
        for it in range(iters):
            px -= d * gx
            py -= d * gy
            pz -= d * gz
            d, gx, gy, gz = _sdf(px, py, pz, A, BA, IL2, RA, DR, GS, GE, KG, KJ)
        P[i, 0] = px
        P[i, 1] = py
        P[i, 2] = pz
        NRM[i, 0] = gx
        NRM[i, 1] = gy
        NRM[i, 2] = gz
        j0 = OWN[i]
        do, _a, _b, _c = _cone_d(px, py, pz, j0, A, BA, IL2, RA, DR)
        s = 0.0
        for j in range(M):
            dj, _a, _b, _c = _cone_d(px, py, pz, j, A, BA, IL2, RA, DR)
            x = (do - dj) / tau
            if x > 40.0:
                x = 40.0
            if x > -30.0:
                s += math.exp(x)
        WOWN[i] = 1.0 / s if s > 0.0 else 0.0


@njit(parallel=True, fastmath=True, cache=True)
def _attach(OWN, Q, A, R, out):
    for i in prange(OWN.shape[0]):
        j = OWN[i]
        qx = Q[i, 0]
        qy = Q[i, 1]
        qz = Q[i, 2]
        out[i, 0] = A[j, 0] + R[j, 0, 0] * qx + R[j, 0, 1] * qy + R[j, 0, 2] * qz
        out[i, 1] = A[j, 1] + R[j, 1, 0] * qx + R[j, 1, 1] * qy + R[j, 1, 2] * qz
        out[i, 2] = A[j, 2] + R[j, 2, 0] * qx + R[j, 2, 1] * qy + R[j, 2, 2] * qz


@njit(fastmath=True, cache=True, inline='always')
def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = h & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x7FEB352D) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x846CA68B) & 0xFFFFFFFF
    h ^= h >> 16
    return h


@njit(parallel=True, fastmath=True, cache=True)
def _worley(P, freq, seed, F1, F2, CID):
    """3D cellular noise: distance to the nearest and second-nearest feature point (cell units), nearest cell id"""
    for i in prange(P.shape[0]):
        x = P[i, 0] * freq
        y = P[i, 1] * freq
        z = P[i, 2] * freq
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        iz = int(math.floor(z))
        f1 = 1e9
        f2 = 1e9
        c1 = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cx = ix + dx
                    cy = iy + dy
                    cz = iz + dz
                    h1 = _hash3(cx, cy, cz, seed)
                    h2 = _hash3(cx, cy, cz, seed + 17)
                    h3 = _hash3(cx, cy, cz, seed + 31)
                    fx = cx + (h1 & 1023) / 1024.0
                    fy = cy + (h2 & 1023) / 1024.0
                    fz = cz + (h3 & 1023) / 1024.0
                    d2 = (x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2
                    if d2 < f1:
                        f2 = f1
                        f1 = d2
                        c1 = h1
                    elif d2 < f2:
                        f2 = d2
        F1[i] = math.sqrt(f1)
        F2[i] = math.sqrt(f2)
        CID[i] = c1 & 0xFFFF


# ------------------------------------------------------------- the rig ---

class HandSkel:
    """Bone list (tapered capsules) grouped for the smooth union. pose() -> per-bone origin A and frame R
    (columns x, y, z; y along the bone); static per-bone length, radii and group."""
    KG = np.array([0.045, 0.014, 0.014, 0.014, 0.014, 0.014])   # palm melts; fingers keep their creases
    KJ = 0.022

    def __init__(self):
        spec = []          # (group, name, ra, rb, length) in bone order (grouped, ascending)
        self.static = {}   # palm bones: fixed (a, b)
        pal = [
            ('carpus', (-0.075, 0.045, 0.0), (0.075, 0.05, 0.0), 0.058, 0.055),
            ('thenar', (-0.07, 0.085, 0.034), (-0.118, 0.205, 0.034), 0.058, 0.044),
            ('hypothenar', (0.105, 0.1, 0.022), (0.13, 0.34, 0.012), 0.048, 0.04),
            ('pad0', (-0.06, 0.2, 0.024), (0.08, 0.2, 0.024), 0.058, 0.056),
            ('pad1', (-0.11, 0.385, 0.02), (0.11, 0.37, 0.02), 0.05, 0.045),
            ('radius', (-0.05, 0.02, 0.004), (-0.075, -1.25, 0.02), 0.07, 0.1),
            ('ulna', (0.05, 0.02, 0.0), (0.065, -1.25, 0.02), 0.064, 0.095),
            ('belly', (0.0, -0.3, -0.02), (0.0, -1.25, -0.03), 0.075, 0.115),
        ]
        for f in FNAMES:
            mx, my = FINGERS[f][0], FINGERS[f][1]
            pal.append(('meta_' + f, (mx * 0.42, 0.075, -0.004), (mx, my - 0.008, -0.006), 0.041, 0.049))
        for name, a, b, ra, rb in pal:
            a, b = np.array(a, np.float64), np.array(b, np.float64)
            self.static[name] = (a, b)
            spec.append((0, name, ra, rb, float(np.linalg.norm(b - a))))
        spec.append((0, 'thumb0', THUMB_R[0], THUMB_R[1], THUMB_L[0]))
        for fi, f in enumerate(FNAMES):
            lens = [FLEN * v for v in FINGERS[f][2]]
            rad = FINGERS[f][3]
            g = 1 + fi
            spec.append((g, f + '_mcp', rad[0] * 1.04, rad[0] * 1.04, 0.0))
            spec.append((g, f + '_p0', rad[0], rad[1], lens[0]))
            spec.append((g, f + '_pip', rad[1] * 1.1, rad[1] * 1.1, 0.0))
            spec.append((g, f + '_p1', rad[1], rad[2], lens[1]))
            spec.append((g, f + '_dip', rad[2] * 1.07, rad[2] * 1.07, 0.0))
            spec.append((g, f + '_p2', rad[2], rad[3] * 0.72, lens[2]))
        spec.append((5, 'thumb1', THUMB_R[1], THUMB_R[2], THUMB_L[1]))
        spec.append((5, 'thumb_ip', THUMB_R[2] * 1.08, THUMB_R[2] * 1.08, 0.0))
        spec.append((5, 'thumb2', THUMB_R[2], THUMB_R[3] * 0.8, THUMB_L[2]))
        self.names = [s[1] for s in spec]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.grp = np.array([s[0] for s in spec], np.int64)
        self.RA = np.array([s[2] for s in spec], np.float64)
        self.RB = np.array([s[3] for s in spec], np.float64)
        self.LEN = np.array([s[4] for s in spec], np.float64)
        self.DR = self.RB - self.RA
        ng = int(self.grp.max()) + 1
        self.GS = np.array([int(np.nonzero(self.grp == g)[0][0]) for g in range(ng)], np.int64)
        self.GE = np.array([int(np.nonzero(self.grp == g)[0][-1]) + 1 for g in range(ng)], np.int64)
        self.M = len(spec)
        self.forearm = np.array([n in ('radius', 'ulna', 'belly') for n in self.names])

    def pose(self, flex, spread, thumb_close):
        M = self.M
        A = np.zeros((M, 3))
        R = np.zeros((M, 3, 3))
        ix = self.idx
        for name, (a, b) in self.static.items():
            j = ix[name]
            A[j] = a
            R[j] = _frame_y(b - a)
        for fi, f in enumerate(FNAMES):
            mx, my = FINGERS[f][0], FINGERS[f][1]
            lens = [FLEN * v for v in FINGERS[f][2]]
            j = ix[f + '_mcp']
            A[j] = (mx, my - 0.006, -0.012)
            R[j] = np.eye(3)
            Rf = _rz(-spread[f])
            p0 = np.array([mx, my, -0.002])
            for k, (bone, node) in enumerate((('_p0', '_pip'), ('_p1', '_dip'), ('_p2', None))):
                Rf = Rf @ _rx(flex[f][k])
                j = ix[f + bone]
                A[j] = p0
                R[j] = Rf
                p0 = p0 + Rf[:, 1] * lens[k]
                if node is not None:
                    jn = ix[f + node]
                    A[jn] = p0
                    R[jn] = Rf
        # thumb: CMC on the radial side near the wrist; swings from abducted to opposed
        cmc = np.array([-0.1, 0.075, 0.02])
        d_open = np.array([-0.66, 0.62, 0.42])
        d_closed = np.array([0.0, 0.75, 0.66])
        d = lerp(d_open, d_closed, thumb_close)
        d /= np.linalg.norm(d)
        y = d
        z = np.array([1.0, 0.0, -0.2])               # flexion carries the tip across the curled fingers
        z = z - (z @ y) * y
        z /= np.linalg.norm(z)
        x = np.cross(y, z)
        Rt = np.stack([x, y, z], 1)
        bend = 0.15 + 0.35 * thumb_close
        p0 = cmc
        for k, name in enumerate(('thumb0', 'thumb1', 'thumb2')):
            if k > 0:
                Rt = Rt @ _rx(bend)
            j = ix[name]
            A[j] = p0
            R[j] = Rt
            p0 = p0 + Rt[:, 1] * THUMB_L[k]
            if k == 1:
                A[ix['thumb_ip']] = p0
                R[ix['thumb_ip']] = Rt
        return A, R

    def kernel_args(self, A, R):
        BA = R[:, :, 1] * self.LEN[:, None]
        L2 = self.LEN ** 2
        IL2 = np.where(L2 > 1e-12, 1.0 / np.maximum(L2, 1e-12), 0.0)
        return (np.ascontiguousarray(A), np.ascontiguousarray(BA), IL2, self.RA, self.DR,
                self.GS, self.GE, self.KG, self.KJ)

    def sample(self, r, dens, dens_fore):
        """points on each bone's own surface (bone-local offsets), ~uniform per area"""
        OWN, Q, AREA = [], [], []
        for j in range(self.M):
            ra, rb, L = self.RA[j], self.RB[j], self.LEN[j]
            dn = dens_fore if self.forearm[j] else dens
            tube = math.pi * (ra + rb) * L
            caps = 2 * math.pi * ra * ra + 2 * math.pi * rb * rb
            n = max(int((tube + caps) * dn), 60)
            area = (tube + caps) / n
            nt = int(round(n * tube / (tube + caps)))
            nc = n - nt
            u = r.random(nt)
            if abs(rb - ra) > 1e-6:            # lateral area density grows with the radius
                h = (np.sqrt(u * (rb * rb - ra * ra) + ra * ra) - ra) / (rb - ra)
            else:
                h = u
            rr = ra + (rb - ra) * h
            th = r.uniform(0, 2 * np.pi, nt)
            qt = np.stack([rr * np.cos(th), h * L, rr * np.sin(th)], 1)
            dd = rand_dirs(r, nc)
            end = r.random(nc) < rb * rb / (ra * ra + rb * rb)
            dd[:, 1] = np.where(end, np.abs(dd[:, 1]), -np.abs(dd[:, 1]))
            qc = np.where(end[:, None], dd * rb + np.array([0.0, L, 0.0]), dd * ra)
            OWN.append(np.full(n, j, np.int64))
            Q.append(np.concatenate([qt, qc]))
            AREA.append(np.full(n, area))
        return np.concatenate(OWN), np.concatenate(Q), np.concatenate(AREA)


REST_POSE = (
    {f: (0.12, 0.12, 0.08) for f in FNAMES},
    dict(SPREAD_REST),
    0.35,
)


def hand_pose_at(t):
    """rise relaxed (slight curl) -> reach open and spread (~995-1012) -> close 1015-1036 (fist on the crown)."""
    reach = float(smoothstep(986, 1010, t))
    # slow at first (1015-1025), decisive 1025-1036
    close = 0.22 * float(smoothstep(1015, 1025, t)) + 0.78 * float(ease_in((t - 1025) / 11.0, 1.6))
    flex, spread = {}, {}
    rest = (0.3, 0.38, 0.22)
    openp = (0.2, 0.26, 0.18)
    closed = {'index': (1.45, 1.7, 1.25), 'middle': (1.5, 1.72, 1.3), 'ring': (1.55, 1.72, 1.3),
              'little': (1.6, 1.7, 1.25)}
    for fi, f in enumerate(FNAMES):
        # the little finger leads the close slightly, like a real grip
        cf = float(np.clip(close * (1.0 + 0.06 * (fi - 1.5)), 0, 1))
        a = [lerp(rest[k], openp[k], reach) for k in range(3)]
        lag = (0.0, 0.08, 0.16)                      # distal joints lag the proximal
        a = [lerp(a[k], closed[f][k], float(np.clip((cf - lag[k]) / (1 - lag[k]), 0, 1))) for k in range(3)]
        flex[f] = tuple(a)
        s = lerp(SPREAD_REST[f], SPREAD_OPEN[f], reach)
        spread[f] = lerp(s, SPREAD_REST[f] * 0.2, cf)
    thumb = lerp(lerp(0.35, 0.02, reach), 0.9, close)
    return flex, spread, thumb


HAND_L = 60.0
GRASP_AZ = B.ALPHA_C - 0.24


def grasp_dir():
    """horizontal unit vector from the grasp camera toward the crown"""
    a = GRASP_AZ
    return -np.array([math.cos(a), 0.0, math.sin(a)])


def hand_yaw(t):
    """turn about the arm's axis: the back of the hand while it reaches, the thumb side as it grips"""
    return -0.12 - 0.62 * float(ease_in_out((t - 1004) / 32.0))


def hand_transform(t):
    """World placement of the hand (rotation local->world, wrist position).
    Back of the hand faces down/back toward the low camera; the palm opens toward the crown."""
    C = B.crown_centre(960.0)
    dz = grasp_dir()
    up = np.array([0.0, 1.0, 0.0])
    dx = np.cross(up, dz)
    dx /= np.linalg.norm(dx)
    R0 = np.stack([dx, up, dz], 1)
    rise = float(ease_out((t - 960) / 50.0, 2.0))
    approach = float(ease_in_out((t - 1002) / 34.0))
    lean = lerp(-0.3, 0.05, rise) + 0.3 * approach
    R = R0 @ _rx(lean) @ _rz(lerp(0.16, 0.03, rise)) @ _ry(hand_yaw(t))
    # final: the closed fist's hollow (local ~(0, 0.52, 0.11)) sits on the crown
    Rf = R0 @ _rx(0.35) @ _rz(0.03) @ _ry(hand_yaw(1036.0))
    Wf = C - Rf @ (np.array([0.0, 0.52, 0.11]) * HAND_L)
    W0 = Wf + np.array([0.0, -85.0, 0.0]) - dz * 12.0 + dx * 8.0
    Wm = Wf + np.array([0.0, -24.0, 0.0]) - dz * 9.0 + dx * 2.0
    W = lerp(lerp(W0, Wm, rise), Wf, approach)
    return R, W


# ======================================================= v3: THE GRASP THAT CANNOT HOLD (cut C only) ===
# BIBLE_V3 rev. 1 (B9, and the picture red team's "raised fist"): a claw of embers DESCENDS from the upper left,
# palm down, knuckles to the lens, the camera above it looking down into the storm's eye where the Ring hangs. It
# closes on the Ring and cannot hold it: the crust glows from inside and cracks along the hand's own lines, gold
# leaks between the fingers, and the band slips through them and falls away into the eye (the Ring falls).
# No frame shows a forearm rising from below or a fist raised on top of it. (A has no grasp any more.)

C_LEAK = look.hexrgb('#F7B23E') * 0.8 + look.hexrgb(look.PALETTE['mind_gold']) * 0.2   # the Ring's gold
GR_EL0, GR_EL1 = 27.0, 20.0        # camera elevation above the grip (degrees): above the hand, never below
GR_D0, GR_D1 = 74.0, 63.0          # camera distance to the grip (a slow push in)
GR_YAW = math.radians(-32.0)       # the arm comes in from the upper left, not straight down the lens axis
GR_BETA = math.radians(40.0)       # the fingers point down and toward the lens
HOLLOW = np.array([0.0, 0.50, 0.10])   # where the closing fist holds the Ring (hand-local, units of L)
# v3 timing (re-cut for C's locked sheet: c3.py plays src 960-1040 at half speed over C 2320-2480):
T_REACH0, T_REACH1 = 961.0, 979.0      # the descent (C 2322-2358)
T_CLOSE0, T_CLOSE1 = 971.0, 980.0      # the fingers close on the Ring, slow then decisive (C 2360, bar 30 b3)
T_CRACK0 = 1000.0                      # the crust glows and cracks from inside (C 2400, bar 31 b1)
T_SLIP = 1020.0                        # the band slips out between the fingers (C 2440, bar 31 b3)
RING_G = 0.1                           # the fall's gravity (units / frame^2)


GRIP_OVERRIDE = None                   # EMBERS-C (grasp3.py): the grip point in C's own grasp world


def grip_point():
    if GRIP_OVERRIDE is not None:
        return np.asarray(GRIP_OVERRIDE, np.float64).copy()
    return B.crown_centre(960.0)


def _gr_h():
    """horizontal unit vector from the grip toward the grasp camera"""
    a = GRASP_AZ
    return np.array([math.cos(a), 0.0, math.sin(a)])


def cam_grasp_v3(t):
    """Above the hand, looking down past it into the storm's eye (level with it or above, never below)."""
    G = grip_point()
    u = float(np.clip((t - 960) / 80.0, 0, 1))
    push = float(ease_in_out(u))
    el = math.radians(lerp(GR_EL0, GR_EL1, push))
    d = lerp(GR_D0, GR_D1, push)
    h = _gr_h()
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(up, h)
    pos = G + d * (h * math.cos(el) + up * math.sin(el))
    tgt = G + np.array([0.0, lerp(-3.0, -6.5, push), 0.0]) - side * 2.0
    if t > T_CRACK0 + 6:
        k = min((t - (T_CRACK0 + 6)) / 10.0, 1.0)
        pos = pos + 0.25 * k * np.array([math.sin(3.3 * t), math.sin(4.1 * t) * 0.6, math.cos(2.7 * t)])
    return pos, tgt


def _hand_basis():
    """hand-local -> world rotation for the claw: fingers (+Y) down and toward the lens, palm (+Z) down,
    turned so the arm enters from the upper left"""
    h = _gr_h()
    up = np.array([0.0, 1.0, 0.0])
    f = h * math.cos(GR_BETA) - up * math.sin(GR_BETA)
    pn = -up * math.cos(GR_BETA) - h * math.sin(GR_BETA)
    x = np.cross(f, pn)
    R0 = np.stack([x, f, pn], 1)
    c, s_ = math.cos(GR_YAW), math.sin(GR_YAW)
    Ry = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
    return Ry @ R0


def hand_pose_v3(t):
    """a spread claw while it descends; the fingers hook as it arrives; they close on the Ring (slow, then
    decisive), clench, then spasm a little open as the band slips out"""
    reach = float(smoothstep(T_REACH0 + 2, T_REACH1 - 4, t))
    close = 0.3 * float(smoothstep(T_CLOSE0, T_CLOSE0 + 5, t)) + 0.7 * float(ease_in((t - (T_CLOSE0 + 4))
                                                                                   / (T_CLOSE1 - T_CLOSE0 - 4), 1.6))
    close = float(np.clip(close, 0, 1))
    fail = float(smoothstep(T_SLIP - 1.0, T_SLIP + 4.0, t))           # the grip gives as the band slips
    flex, spread = {}, {}
    rest = (0.35, 0.45, 0.3)
    claw = (0.52, 0.97, 0.75)                    # hooked, spread: a claw, not a flat reaching hand
    closed = {'index': (1.4, 1.62, 1.2), 'middle': (1.45, 1.66, 1.25), 'ring': (1.5, 1.66, 1.25),
              'little': (1.55, 1.62, 1.2)}
    for fi, f in enumerate(FNAMES):
        cf = float(np.clip(close * (1.0 + 0.06 * (fi - 1.5)), 0, 1))
        a = [lerp(rest[k], claw[k], reach) for k in range(3)]
        lag = (0.0, 0.08, 0.16)
        a = [lerp(a[k], closed[f][k], float(np.clip((cf - lag[k]) / (1 - lag[k]), 0, 1))) for k in range(3)]
        # the failing grip: the middle and ring fingers part a little, the others shudder
        give = fail * (0.32 if f in ('middle', 'ring') else 0.12)
        a = [a[0] - 0.6 * give, a[1] - give, a[2] - 0.6 * give]
        flex[f] = tuple(a)
        s0 = lerp(SPREAD_REST[f], SPREAD_OPEN[f] * 1.1, reach)
        s1 = lerp(s0, SPREAD_REST[f] * 0.2, cf)
        spread[f] = s1 + fail * {'index': -0.02, 'middle': -0.05, 'ring': 0.05, 'little': 0.02}[f]
    thumb = lerp(lerp(0.35, 0.05, reach), 0.9, close) - 0.15 * fail
    return flex, spread, thumb


def hand_transform_v3(t):
    """World placement of the claw (rotation local->world, wrist position): it descends on a diagonal from the
    upper left and settles so the closing fist's hollow lands on the Ring; as the band slips it jerks back a little."""
    G = grip_point()
    R = _hand_basis()
    Wf = G - R @ (HOLLOW * HAND_L)
    h = _gr_h()
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(up, h)                       # screen right, roughly
    fwd = R[:, 1]
    W0 = Wf - fwd * 34.0 + up * 22.0 - side * 26.0
    arr = float(ease_out((t - T_REACH0) / (T_REACH1 - T_REACH0), 2.2))
    W = W0 + (Wf - W0) * arr
    # a slow drop onto it as the fingers close (weight), then a jerk back as it fails
    W = W - up * (1.2 * float(smoothstep(T_REACH1 - 4, T_CLOSE1 + 2, t)))
    jerk = float(smoothstep(T_SLIP, T_SLIP + 5.0, t))
    W = W - fwd * (2.2 * jerk) + up * (1.6 * jerk)
    tilt = 0.06 * float(smoothstep(T_SLIP, T_SLIP + 6.0, t))
    R = R @ _rx(-tilt)
    return R, W


def ring_state_v3(t):
    """the Ring in the grasp: (centre, rotation) -- it hangs in the storm's eye, is closed on, and slips out
    between the middle and ring fingers, tumbling as it falls away into the eye"""
    G = grip_point()
    h = _gr_h()
    up = np.array([0.0, 1.0, 0.0])
    a = math.atan2(h[2], h[0])
    tilt = math.radians(62.0)                      # its face turned up toward the lens: it reads as a ring
    Rm = B._rot_about(a, tilt)
    C = G.copy()
    if t > T_SLIP:
        x = t - T_SLIP
        slide = 1.6 * float(smoothstep(0.0, 3.5, x)) * min(x, 3.5) / 3.5
        fall = 0.5 * RING_G * max(x - 2.5, 0.0) ** 2
        R_h = _hand_basis()
        C = C - up * (slide + fall) + R_h[:, 0] * (0.9 * float(smoothstep(0.0, 4.0, x)))
        tum = 0.16 * x + 0.02 * x * x
        c, s_ = math.cos(tum), math.sin(tum)
        ax = np.cross(up, h)
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rt = np.eye(3) + s_ * K + (1 - c) * (K @ K)
        Rm = Rt @ Rm
    return C, Rm


def _seg_dist2(P, a, b):
    """distance (x, y only) from points to the segment a-b"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    u = np.clip(((P[:, 0] - a[0]) * d[0] + (P[:, 1] - a[1]) * d[1]) / max(d @ d, 1e-12), 0, 1)
    return np.hypot(P[:, 0] - (a[0] + u * d[0]), P[:, 1] - (a[1] + u * d[1]))


def _poly_dist(P, pts):
    return np.min([_seg_dist2(P, pts[i], pts[i + 1]) for i in range(len(pts) - 1)], axis=0)


PALM_LINES = [  # hand-local (x, y), units of L: heart, head and life lines
    [(0.135, 0.335), (0.06, 0.35), (-0.02, 0.372), (-0.1, 0.405)],
    [(-0.125, 0.305), (-0.04, 0.29), (0.04, 0.265), (0.11, 0.225)],
    [(-0.105, 0.315), (-0.085, 0.24), (-0.06, 0.16), (-0.035, 0.07)],
]


def anat_cracks(P, own, sk, A_rest, R_rest):
    """v3 (picture red team): cracks that follow the hand's anatomy, in the rest pose (hand-local, units of L).
    Large plates on the back of the hand and the forearm, long creases (the tendons to each knuckle, the knuckle
    wrinkles, the palm's three lines, the flexion creases, the wrist), small cells at the joints; bark along the
    fingers. Returns (sharp 0..1: the crack itself, soft 0..1: the crust near a crack, where heat shows through,
    plate id 0..1, crease 0..1: the hottest, widest lines)."""
    n = len(P)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    names = np.array(sk.names)[own]
    is_f = np.array([('_p' in nm or '_mcp' in nm or '_pip' in nm or '_dip' in nm) and not nm.startswith('thumb')
                     for nm in names])
    is_t = np.array([nm.startswith('thumb') for nm in names])
    is_arm = sk.forearm[own]
    dorsal = z < 0.012
    # distance to the nearest knuckle (MCP, PIP, DIP, thumb IP) and to the wrist
    kn = []
    for f in FNAMES:
        for node in ('_mcp', '_pip', '_dip'):
            kn.append(A_rest[sk.idx[f + node]])
    kn.append(A_rest[sk.idx['thumb_ip']])
    kn = np.array(kn)
    dk = np.sqrt(((P[:, None, :] - kn[None, :, :]) ** 2).sum(-1)).min(1)
    near_j = np.exp(-(dk / 0.03) ** 2) + np.exp(-((y - 0.01) / 0.03) ** 2) * (~is_f)
    near_j = np.clip(near_j, 0, 1)

    def wor(Q, freq, seed, width):
        F1, F2, cid = np.empty(n), np.empty(n), np.empty(n, np.int64)
        _worley(np.ascontiguousarray(Q), freq, seed, F1, F2, cid)
        e = F2 - F1
        return 1.0 - smoothstep(0.0, width, e), 1.0 - smoothstep(0.0, width * 2.5, e), (cid % 997) / 997.0
    Qb = P * np.array([1.0, 0.55, 1.0])                        # plates stretched along the arm and hand
    big, big_s, plate = wor(Qb, 6.5, 11, 0.05)
    Qf = P.copy()
    Qf[:, 1] = np.where(y > 0.42, 0.42 + (y - 0.42) * 0.42, y)  # bark running along the fingers
    mid, mid_s, plate_f = wor(Qf, 15.0, 17, 0.06)
    small, small_s, _ = wor(P, 36.0, 29, 0.07)
    w_big = ((~is_f) & (~is_t) & (dorsal | is_arm)).astype(float) * (1 - near_j)
    w_mid = np.clip(1.0 - w_big - near_j, 0, 1)
    sharp = np.maximum.reduce([w_big * big, w_mid * mid * 0.85, near_j * small * 0.75])
    soft = np.maximum.reduce([w_big * big_s, w_mid * mid_s * 0.8, near_j * small_s * 0.7])
    plate = np.where(w_big > 0.5, plate, plate_f)
    # ---- creases
    cre = np.zeros(n)
    # the tendons: wrist -> each knuckle, on the back of the hand
    for f in FNAMES:
        mx, my = FINGERS[f][0], FINGERS[f][1]
        d = _seg_dist2(P, (mx * 0.3, 0.06), (mx, my - 0.03))
        cre = np.maximum(cre, np.exp(-(d / 0.0075) ** 2) * (dorsal & ~is_f & ~is_arm))
    # knuckle wrinkles (back) and flexion creases (front) at every finger joint: short transverse lines
    for f in FNAMES:
        for node, amp in (('_mcp', 0.8), ('_pip', 1.0), ('_dip', 0.9)):
            j = sk.idx[f + node]
            c = A_rest[j]
            ax = R_rest[sk.idx[f + ('_p1' if node == '_pip' else '_p2' if node == '_dip' else '_p0')]][:, 1]
            along = (P - c) @ ax
            lat = np.linalg.norm((P - c) - along[:, None] * ax[None, :], axis=1)
            band = np.exp(-(lat / 0.05) ** 2)
            # EMBERS-C: only across the back (knuckle wrinkles) and the front (flexion creases), never round the
            # sides (a full band round a finger reads as a ring worn on it)
            sx = R_rest[j][:, 0]
            side = np.abs((P - c) @ sx) / np.maximum(lat, 1e-6)
            band = band * np.exp(-(side / 0.55) ** 2)
            for off in ((-0.012, 0.012) if node != '_mcp' else (0.0,)):
                cre = np.maximum(cre, amp * np.exp(-((along - off) / 0.0042) ** 2) * band * (is_f | ~is_arm))
    # the palm's three lines, and the wrist
    pal = (~is_f) & (~is_t) & (~is_arm) & (z > 0.0)
    for L_ in PALM_LINES:
        cre = np.maximum(cre, np.exp(-(_poly_dist(P, L_) / 0.006) ** 2) * pal)
    for yw in (0.015, -0.02):     # (EMBERS-C: the wrist creases on the palm side only: round it they read as a cuff)
        cre = np.maximum(cre, 0.8 * np.exp(-((y - yw) / 0.005) ** 2) * (~is_f) * (~is_t) * (z > 0.01))
    sharp = np.clip(np.maximum(sharp, cre), 0, 1)
    soft = np.clip(np.maximum(soft, 0.6 * np.sqrt(cre)), 0, 1)
    return sharp, soft, plate, cre


def _splat_col(fr, P0, P1, RW, colE, ctx):
    """splat points whose colour carries the energy; dark points are skipped"""
    E = colE.max(1)
    m = E > 1e-7
    if not m.any():
        return
    RW = np.broadcast_to(np.asarray(RW, np.float64), (len(E),))
    fr.splat(P0[m], P1[m], RW[m], E[m], colE[m] / E[m][:, None], ctx.cam0, ctx.cam1, zref=0.0)


class Hand:
    """The colossal hand of embers: a coal-bed crust over one smooth surface (see the section note)."""

    def __init__(self, dens=110000.0):
        self.sk = sk = HandSkel()
        r = rng(303)
        # uniform population (crust, rims, coverage)
        self.u_own, self.u_q, self.u_area = sk.sample(r, dens, dens * 0.45)
        # crack population: dense candidates, kept with probability = crack intensity (importance sampling)
        c_own, c_q, c_area = sk.sample(r, dens * 5.0, dens * 2.0)
        A, R = sk.pose(*REST_POSE)
        args = sk.kernel_args(A, R)
        rest_u = self._surface(self.u_own, self.u_q, A, R, args)[0]
        rest_c = self._surface(c_own, c_q, A, R, args)[0]
        # v3: cracks follow the anatomy (anat_cracks); plus a fine network that only opens as the claw fails
        cr, soft_c, plate, cre_c = anat_cracks(rest_c, c_own, sk, A, R)
        nfine = len(cr)
        F1, F2, cidf = np.empty(nfine), np.empty(nfine), np.empty(nfine, np.int64)
        _worley(np.ascontiguousarray(rest_c), 19.0, 41, F1, F2, cidf)
        fine = (1.0 - smoothstep(0.0, 0.05, F2 - F1)) * 0.32
        late = (r.random(nfine) < fine) & (r.random(nfine) >= cr)
        keep = (r.random(len(cr)) < cr) | late
        # drop points that stay buried inside other bones in every pose the shot visits
        keep &= self._exposed(c_own, c_q)
        self.c_own, self.c_q, self.c_area = c_own[keep], c_q[keep], c_area[keep]
        self.c_tex = rest_c[keep]
        self.c_plate = plate[keep]
        self.c_cre = cre_c[keep]
        self.c_late = late[keep]
        # when each late crack opens: first where it grips (the palm and fingers round the Ring), last up the arm
        dgrip = np.linalg.norm((self.c_tex - np.array([0.0, 0.5, 0.06])) * np.array([1.0, 0.8, 1.0]), axis=1)
        self.c_open = T_CRACK0 + 16.0 * np.clip(dgrip / 0.9, 0, 1.4) + r.uniform(-2.0, 2.0, len(self.c_tex))
        ku = self._exposed(self.u_own, self.u_q)
        self.u_own, self.u_q, self.u_area = self.u_own[ku], self.u_q[ku], self.u_area[ku]
        self.u_tex = rest_u[ku]
        _, self.u_soft, self.u_plate, _ = anat_cracks(self.u_tex, self.u_own, sk, A, R)
        # knuckles (dorsal points of the MCP, PIP and DIP joints in the rest pose) burn hotter
        kpts = []
        for f in FNAMES:
            for node in ('_mcp', '_pip', '_dip'):
                j = sk.idx[f + node]
                kpts.append(A[j] + np.array([0.0, 0.0, -sk.RA[j]]))
        kpts.append(A[sk.idx['thumb_ip']] + R[sk.idx['thumb_ip']][:, 2] * -sk.RA[sk.idx['thumb_ip']])
        kpts = np.array(kpts)

        def knuck(P):
            d2 = ((P[:, None, :] - kpts[None, :, :]) ** 2).sum(-1).min(1)
            return np.exp(-d2 / 0.034 ** 2)
        self.u_kn = knuck(self.u_tex)
        self.c_kn = knuck(self.c_tex)
        rr = rng(5)
        self.u_rnd = rr.random(len(self.u_own))
        self.c_rnd = rr.random(len(self.c_own))
        self.c_ph = rr.uniform(0, 2 * np.pi, len(self.c_own))
        # slow heat field along the cracks (some seams roar, some smoulder)
        self.c_var = 0.5 + 0.5 * np.clip(snoise(self.c_tex, 7.0, (3.1, 0.7, 5.2), 2), -1, 1)
        self.u_fore = sk.forearm[self.u_own]
        self.c_fore = sk.forearm[self.c_own]
        # embers shed off the hand (sources on the crust; they only show where the source is an edge)
        ns = 14000
        self.ns = ns
        self.s_src = rr.integers(0, len(self.u_own), ns)
        self.s_life = rr.uniform(9.0, 26.0, ns)
        self.s_ph = rr.random(ns)
        self.s_v = rr.normal(0, 1, (ns, 3)) * np.array([0.9, 0.5, 0.9]) + np.array([0.0, 1.6, 0.0])
        self.s_E = rr.lognormal(0, 0.7, ns)
        print('hand: crust', len(self.u_own), 'cracks', len(self.c_own), 'bones', sk.M)

    # -------------------------------------------------------------- geometry
    def _surface(self, own, q, A, R, args, iters=5):
        P = np.empty((len(own), 3))
        _attach(own, q, A, R, P)
        N = np.empty_like(P)
        D0 = np.empty(len(own))
        W = np.empty(len(own))
        _project(P, own, *args, iters, 0.0022, N, D0, W)
        return P, N, W

    def _exposed(self, own, q):
        wmax = np.zeros(len(own))
        for pose in (REST_POSE, hand_pose_v3(T_REACH0 + 4), hand_pose_v3(T_REACH1 - 4), hand_pose_v3(T_CLOSE1),
                     hand_pose_v3(T_SLIP + 2), hand_pose_v3(T_SLIP + 10)):
            A, R = self.sk.pose(*pose)
            wmax = np.maximum(wmax, self._surface(own, q, A, R, self.sk.kernel_args(A, R), iters=2)[2])
        return wmax > 0.02

    def _cracks(self, P):
        n = len(P)
        P = P.copy()
        y = P[:, 1]
        P[:, 1] = np.where(y > 0.4, 0.4 + (y - 0.4) * 0.38, y)     # cells run along the fingers (bark, not rings)
        F1, F2, cid = np.empty(n), np.empty(n), np.empty(n, np.int64)
        _worley(P, 13.0, 11, F1, F2, cid)
        c1 = 1.0 - smoothstep(0.0, 0.065, F2 - F1)
        plate = (cid % 997) / 997.0
        g1, g2, cid2 = np.empty(n), np.empty(n), np.empty(n, np.int64)
        _worley(P, 31.0, 29, g1, g2, cid2)
        c2 = 1.0 - smoothstep(0.0, 0.06, g2 - g1)
        return np.clip(np.maximum(c1, 0.4 * c2 * (0.3 + 0.7 * plate)), 0, 1), plate

    def pose_world(self, t):
        flex, spread, th = hand_pose_v3(t)
        A, R = self.sk.pose(flex, spread, th)
        Rh, W = hand_transform_v3(t)
        return A, R, Rh, W

    def raw_world(self, own, q, t):
        """bone-attached positions (no projection) in world space: cheap, for motion vectors and ember births"""
        A, R, Rh, W = self.pose_world(t)
        P = np.empty((len(own), 3))
        _attach(own, q, A, R, P)
        return _xf(P * HAND_L, Rh) + W

    def surface_world(self, own, q, t):
        A, R, Rh, W = self.pose_world(t)
        args = self.sk.kernel_args(A, R)
        P, N, Wn = self._surface(own, q, A, R, args)
        return _xf(P * HAND_L, Rh) + W, _xf(N, Rh), Wn, (Rh, W)

    # -------------------------------------------------------------- render
    def emit(self, ctx, fr_hand, fr_cov):
        t = ctx.t
        if t < 960 or t >= 1040:
            return
        cam = ctx.cam
        fpx = cam.f_px(1920)
        Cc = B.crown_centre(960.0)
        fade_in = float(smoothstep(960, 963, t))
        close = float(smoothstep(T_CLOSE0, T_CLOSE1, t))
        # heat of the hand: it wakes as it descends; v3: as it closes on the Ring it glows from inside and cracks,
        # hottest as the band slips, then the broken crust cools a little
        heat = 0.75 + 0.25 * float(smoothstep(964, T_CLOSE1, t))
        glowk = float(smoothstep(T_CRACK0, T_SLIP + 1.0, t)) ** 1.4 * (1 - 0.35 * float(smoothstep(T_SLIP + 2, 1040, t)))
        surge = 0.0
        ringC = ring_state_v3(t)[0]

        def view(P, N):
            V = cam.pos - P
            dist = np.linalg.norm(V, axis=1)
            V /= dist[:, None]
            f = (N * V).sum(1)
            z = (P - cam.pos) @ cam.R[2]
            pa = (fpx * HAND_L / np.maximum(z, 1.0)) ** 2
            return f, pa, dist

        # ---------------- crust: body glow, burning rims, the cold rim of the crown, coverage
        P, N, Wn, (Rh, Wr) = self.surface_world(self.u_own, self.u_q, t)
        R0 = self.raw_world(self.u_own, self.u_q, ctx.t0)
        R1 = self.raw_world(self.u_own, self.u_q, ctx.t1)
        Rm = self.raw_world(self.u_own, self.u_q, t)
        P0, P1 = P + (R0 - Rm), P + (R1 - Rm)
        f, pa, _ = view(P, N)
        pa = pa * self.u_area                                  # projected px^2 per point
        front = f > 0.0
        fp = np.clip(f, 0.0, 1.0)
        along = (P - Wr) @ Rh[:, 1] / HAND_L                  # position along the arm (0 = wrist)
        fore_fade = smoothstep(-0.95, 0.0, along)
        w = Wn * front * fade_in
        L = Cc - P
        dL = np.linalg.norm(L, axis=1)
        lam = np.clip((N * L).sum(1) / np.maximum(dL, 1e-6), 0, 1)
        near = smoothstep(80.0, 14.0, dL)
        rimw = 0.5
        rim = np.clip(1.0 - fp / rimw, 0, 1) ** 2.2
        rim_c = np.clip(1.0 - fp / 0.4, 0, 1) ** 2
        fl = 1 + 0.35 * np.sin(1.1 * t + 9 * self.u_rnd) * np.sin(0.37 * t + 23 * self.u_rnd)
        # v3: an opaque, near-black crust; heat shows through only where it is thin (beside the cracks, over the
        # knuckles), and more as the claw fails; gold where the Ring burns inside the grip
        dR = np.linalg.norm(P - ringC[None, :], axis=1)
        leak = np.exp(-(np.maximum(dR - 2.6, 0.0) / 1.7) ** 2) * close        # only where it touches the band
        thin = np.clip(self.u_soft + 0.6 * self.u_kn, 0, 1)
        v_body = (0.011 + 0.02 * self.u_kn) * heat * fl + thin ** 2 * (0.01 + 0.02 * glowk) * heat
        v_rim = 3.2 * heat * (0.6 + 0.4 * fl) * fore_fade
        v_cold = lam ** 1.5 * (0.5 + 2.4 * near) * (1 + 0.8 * close)
        e_scale = pa * fp * w
        c_body = (C_CRIMSON * 0.7 + C_RED * 0.3)[None, :] * (1 - thin[:, None] * glowk) \
            + look.blackbody(0.62)[None, :] * (thin[:, None] * glowk)
        c_rim = look.blackbody(0.42 + 0.2 * rim) * 0.75 + C_RED * 0.25
        c_cold = C_ICE * 0.65 + C_CORE * 0.35
        colB = (c_body * (v_body * fore_fade)[:, None] + C_LEAK[None, :] * (0.45 * leak)[:, None]) * e_scale[:, None]
        # rims go to their own layer: post keeps them at the OUTER silhouette (a backlight cannot reach the
        # edge of a finger that lies in front of the palm), so a finger seen end-on is never a lit ring
        colR = (c_rim * (v_rim * rim)[:, None] + c_cold[None, :] * (v_cold * rim_c)[:, None]) * e_scale[:, None]
        spacing = np.sqrt(self.u_area) * HAND_L
        _splat_col(fr_hand, P0, P1, spacing * 0.3, colB, ctx)
        _splat_col(getattr(ctx, 'fr_rim', fr_hand), P0, P1, spacing * 0.3, colR, ctx)
        cov = pa * fp * Wn * front * 9.0 * fade_in             # v3: denser coverage -- the crust is opaque
        fr_cov.splat(P0, P1, spacing * 0.9, cov, np.ones(3), ctx.cam0, ctx.cam1, zref=0.0)

        # ---------------- cracks: the coal bed (seams of fire in the crust, hottest at the knuckles)
        Pc, Nc, Wc, _ = self.surface_world(self.c_own, self.c_q, t)
        Q0 = self.raw_world(self.c_own, self.c_q, ctx.t0)
        Q1 = self.raw_world(self.c_own, self.c_q, ctx.t1)
        Qm = self.raw_world(self.c_own, self.c_q, t)
        fc, pac, _ = view(Pc, Nc)
        pac = pac * self.c_area
        fcp = np.clip(fc, 0.0, 1.0)
        alongc = (Pc - Wr) @ Rh[:, 1] / HAND_L
        # heat pulses travel up the arm into the fingers
        wave = 0.5 + 0.5 * np.sin(2 * np.pi * (self.c_tex[:, 1] * 1.3 - (t - 960) / 30.0))
        flc = 1 + 0.4 * np.sin(1.7 * t + self.c_ph) * np.sin(0.53 * t + 2.3 * self.c_ph)
        # v3: creases burn hottest; plates' seams medium; the late network opens from the grip outward as it fails
        cls = 0.45 + 1.1 * self.c_cre
        opened = np.where(self.c_late, smoothstep(self.c_open, self.c_open + 3.0, t), 1.0)
        fresh = np.where(self.c_late, np.exp(-np.maximum(t - self.c_open, 0.0) / 5.0) * opened, 0.0)
        hot = cls * (0.25 + 0.75 * self.c_var ** 2) * (0.6 + 0.4 * wave) * flc * (1 + 1.5 * self.c_kn)
        v_c = 1.3 * hot * heat * opened * (1 + glowk * (0.15 + 1.8 * self.c_cre) + 1.1 * fresh) * smoothstep(-1.0, -0.05, alongc)
        dRc = np.linalg.norm(Pc - ringC[None, :], axis=1)
        leakc = np.exp(-(np.maximum(dRc - 2.6, 0.0) / 2.0) ** 2) * close
        temp = np.clip(0.36 + 0.16 * self.c_var + 0.12 * self.c_cre + glowk * (0.06 + 0.3 * self.c_cre) + 0.2 * fresh
                       + 0.15 * leakc, 0, 0.93)
        colc = (look.blackbody(temp) * v_c[:, None] + C_LEAK[None, :] * (2.2 * leakc)[:, None]) \
            * (pac * fcp ** 0.8 * Wc * (fc > 0) * fade_in)[:, None]
        sp_c = np.sqrt(self.c_area) * HAND_L
        _splat_col(fr_hand, Pc + (Q0 - Qm), Pc + (Q1 - Qm), sp_c * 0.3, colc, ctx)

        # ---------------- embers shed off the burning edges, streaming up into the storm
        self.emit_embers(ctx, fr_hand, f, P, N, heat * (1 + 0.5 * surge), fade_in)

    def emit_embers(self, ctx, fr_hand, f_u, P_u, N_u, heat, fade_in):
        t = ctx.t
        ns = self.ns
        life = self.s_life
        age = ((t - 900.0) / life + self.s_ph) % 1.0 * life          # frames since birth
        src = self.s_src
        # birth positions: the hand's surface at the birth time (bucketed by age)
        B0 = np.empty((ns, 3))
        edges = np.arange(0.0, 27.0, 4.5)
        for a0 in edges:
            m = (age >= a0) & (age < a0 + 4.5)
            if m.any():
                B0[m] = self.raw_world(self.u_own[src[m]], self.u_q[src[m]], t - (a0 + 2.25))
        n_src = N_u[src]
        edge = np.exp(-np.abs(f_u[src]) / 0.3)
        front = f_u[src] > -0.05

        def pos(a):
            k = a[:, None]
            drift = self.s_v * k * 0.35 + np.array([0.0, 1.0, 0.0]) * 0.015 * k ** 2
            p = B0 + n_src * (0.4 + 0.25 * k) + drift
            w = vnoise(p * 0.08 + np.array([0.0, -0.05 * t, 0.0]), 1.0, (0, 0, 0), 1)
            return p + w * (0.15 * k)
        S0 = pos(np.maximum(age - 0.25, 0))
        S1 = pos(age + 0.25)
        u = age / life
        e = (self.s_E * (1 - u) ** 2 * smoothstep(0.0, 2.0, age) * (0.15 + 2.4 * edge) * 4.0 * heat * fade_in
             * float(smoothstep(964, 972, t)))
        col = look.blackbody(np.clip(0.72 - 0.42 * u, 0.2, 1))
        fr_hand.splat(S0[front], S1[front], 0.03, e[front], col[front], ctx.cam0, ctx.cam1, zref=60.0)
        ctx.fr.splat(S0[~front], S1[~front], 0.03, e[~front], col[~front], ctx.cam0, ctx.cam1, zref=60.0)


def cam_grasp(t):
    """Below and behind the hand, looking up at it and the storm's eye (slow, huge)."""
    C = B.crown_centre(960.0)
    u = (t - 960) / 80.0
    push = float(ease_in_out(u))
    a = GRASP_AZ
    d = lerp(96.0, 86.0, push)
    pos = np.array([d * math.cos(a), lerp(-18.0, -12.0, push), d * math.sin(a)])
    tgt = C + np.array([0.0, lerp(-30.0, -18.0, push), 0.0])
    if t > 1004:
        k = (t - 1004) / 32.0
        sh = 0.3 * k * k
        pos = pos + sh * np.array([math.sin(3.1 * t), math.sin(4.3 * t), 0])
    return pos, tgt


# ================================================================= SILENCE ===

class LastEmber:
    """One small ember drifting down in black, dimming, dying by ~1150 (lower third stays clear)."""

    def __init__(self):
        r = rng(77)
        self.n_smoke = 260
        self.sm_d = r.normal(0, 1, (self.n_smoke, 3))
        self.sm_ph = r.random(self.n_smoke)
        self.sm_v = r.uniform(0.004, 0.01, self.n_smoke)

    def pos(self, t):
        u = t - 1040
        x = 0.35 + 0.18 * math.sin(u * 0.045) + 0.04 * math.sin(u * 0.13)
        y = 1.05 - 0.0068 * u - 0.000012 * u * u
        z = -6.0 + 0.3 * math.sin(u * 0.03)
        return np.array([x, y, z])

    def life(self, t):
        u = t - 1040
        base = math.exp(-u / 70.0)
        fl = 1 + 0.35 * math.sin(u * 0.9) * math.sin(u * 0.23)
        # a last flare, then out
        flare = 0.8 * math.exp(-((u - 98) / 3.0) ** 2)
        die = 1 - smoothstep(100, 110, u)
        return max(base * fl + flare, 0) * die

    def emit(self, ctx):
        t = ctx.t
        if t < 1040:
            return
        lv = self.life(t)
        if lv > 0.001:
            p0 = self.pos(ctx.t0)[None, :]
            p1 = self.pos(ctx.t1)[None, :]
            temp = 0.45 + 0.35 * lv
            e = np.array([60.0 * lv])
            ctx.fr.splat(p0, p1, 0.006, e, look.blackbody(temp)[None, :], ctx.cam0, ctx.cam1)
            H = np.array([self.pos(t)])
            ctx.fr.splat(H, H, np.array([0.12]), np.array([90.0 * lv]), look.blackbody(temp - 0.1)[None, :],
                         ctx.cam0, ctx.cam1, profile=1)
        # a thin wisp of smoke after it dies (barely there)
        u = t - 1150
        if 0 < u < 45:
            base = self.pos(1150.0)
            k = (self.sm_ph + u * self.sm_v * 6) % 1.0
            P = base + np.array([0, 1.0, 0]) * (k * 0.5)[:, None] + self.sm_d * (0.02 + 0.08 * k)[:, None]
            e = (1 - k) * 0.25 * (1 - u / 45.0) * math.exp(-u / 20.0)
            ctx.fr.splat(P, P, 0.03, e, np.array([0.5, 0.35, 0.3]), ctx.cam0, ctx.cam1, profile=1)


def cam_silence(t):
    return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -10.0])
