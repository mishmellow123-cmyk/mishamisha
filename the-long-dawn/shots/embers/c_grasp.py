"""EMBERS-C2 . E8-C THE GRASP THAT CANNOT HOLD (cut C, C11, C frames 2320-2480, bars 30-31).

    2320  a claw of embers descends out of the storm, palm down, hooked, onto the Ring hanging over the forges; the
          Ring's gold lights it from beneath
    2360  (bar 30 b3) it closes on the Ring: gold leaks along the seams between the fingers
    2400  (bar 31 b1) the crust glows from inside and cracks along the anatomy (tendons, knuckles, creases, plates),
          the heat spreading out from the grip
    2440  (bar 31 b3) the band slips through the fingers and falls away into the black; the claw clenches on nothing,
          and the fire goes out of it

The claw is a SOLID: tapered capsules (scene_c.HandSkel's anatomy, thinner, with talon tips) in a smooth union,
sphere-traced per pixel (`_march`: group bounding spheres, cone-traced edge coverage), shaded as charcoal crust with
ember pores, the anatomical crack field (scene_c.anat_cracks, in the rest pose, so it sticks to the skin), light
from inside, the Ring's gold (a point light, plus the leak along the creases between fingers) and a red rim from the
race below. Not a gauntlet: no plates, no metal, no rivets. The Ring is ringsolid's canonical band (EMBERS-C's),
z-merged with the claw. Behind: C's forge-towers far below (engine), the storm, drifting ash.
"""
import math

import cv2
import numpy as np
from numba import njit, prange

import look
from core import Camera, Frame, smoothstep, smootherstep, lerp, catmull, rng, snoise, ease_out, ease_in
import scene_b as B
import scene_c as SC
import ringsolid as RS
import c3
import c2
from c2 import T_GRASP, T_CLOSE, T_CRACK, T_SLIP, T_GRASP_END

G = B.GROUND
RP = np.array([0.0, 30.0, 0.0])            # where the Ring hangs, over the forges (they crown at ~20-25)
RING_W = 1.3                               # the band's width, as c3 (outer R 2.93)
RING_RO = RS.R_OUT * RING_W
L = 29.0                                   # the claw: wrist crease -> middle fingertip (world units)
CAM_AZ = c3.AZ0 + 2.5
CAM_D, CAM_UP = 56.0, 7.5                  # horizontal distance to the Ring, height above it
FNAMES = SC.FNAMES


# ================================================================ timing ===

def descent(t):
    return float(ease_out((t - T_GRASP) / (T_CLOSE - 4 - T_GRASP), 2.4))


def closing(t):
    """0 open claw .. 1 closed on the Ring (slow, then decisive on bar 30 b3)"""
    a = 0.25 * float(smoothstep(T_CLOSE - 18, T_CLOSE - 8, t))
    b = 0.75 * float(np.clip(ease_in((t - (T_CLOSE - 9)) / 9.0, 1.8), 0, 1))
    return float(np.clip(a + b, 0, 1))


def heat(t):
    """the crust's inner fire: from bar 31 b1, rising to the slip, dying after it"""
    up = float(smoothstep(T_CRACK - 2, T_SLIP - 4, t))
    down = float(smoothstep(T_SLIP + 4, T_GRASP_END - 4, t))
    return up * (1.0 - 0.85 * down)


def slip(t):
    """0 held .. 1 out of the fist (bar 31 b3)"""
    return float(smootherstep(T_SLIP - 3, T_SLIP + 9, t))


def leak(t):
    """the Ring's gold leaking between the fingers"""
    k = float(smoothstep(T_CLOSE - 4, T_CLOSE + 2, t)) * (1.0 - float(smoothstep(T_SLIP - 2, T_SLIP + 6, t)))
    return k * (1.0 + 0.9 * float(smoothstep(T_CRACK, T_SLIP - 4, t)))


# ============================================================== skeleton ===

class ClawSkel(SC.HandSkel):
    """scene_c's anatomy, gaunter: thinner phalanges, bony knuckles, talon tips"""

    def __init__(self):
        super().__init__()
        for j, nm in enumerate(self.names):
            if nm.startswith('thumb') or any(nm.startswith(f + '_') for f in FNAMES):
                node = nm.endswith(('_mcp', '_pip', '_dip', '_ip'))
                k = 0.8 if node else 0.7                  # bony knuckles on gaunt phalanges
                self.RA[j] *= k
                self.RB[j] *= k
                if nm.endswith('_p2') or nm == 'thumb2':
                    self.RB[j] *= 0.5                     # talons
            elif nm in ('radius', 'ulna', 'belly'):
                self.RA[j] *= 0.85                        # a gaunt wrist and forearm
                self.RB[j] *= 0.85
        self.DR = self.RB - self.RA


SK = ClawSkel()
POSE_OPEN = ({f: v for f, v in zip(FNAMES, [(0.18, 0.78, 0.66), (0.14, 0.72, 0.62), (0.16, 0.75, 0.62),
                                           (0.22, 0.8, 0.6)])},
             {f: 1.15 * SC.SPREAD_OPEN[f] for f in FNAMES}, 0.06)
POSE_GRIP = ({f: v for f, v in zip(FNAMES, [(0.76, 1.35, 1.08), (0.66, 1.24, 0.99), (0.67, 1.26, 0.99),
                                           (0.84, 1.4, 1.08)])},      # round the band: a hollow of ~0.09 L
             {f: 0.3 * SC.SPREAD_REST[f] for f in FNAMES}, 0.85)
POSE_FIST = ({f: (1.45, 1.72, 1.3) for f in FNAMES}, {f: 0.15 * SC.SPREAD_REST[f] for f in FNAMES}, 0.95)


def _mix_pose(a, b, k, kf=None):
    fa, sa, ta = a
    fb, sb, tb = b
    flex, spread = {}, {}
    for fi, f in enumerate(FNAMES):
        kk = k if kf is None else kf[fi]
        lag = (0.0, 0.1, 0.2)
        flex[f] = tuple(lerp(fa[f][q], fb[f][q], float(np.clip((kk - lag[q]) / (1 - lag[q]), 0, 1))) for q in range(3))
        spread[f] = lerp(sa[f], sb[f], kk)
    return flex, spread, lerp(ta, tb, k)


def pose(t):
    c = closing(t)
    kf = [float(np.clip(c * (1.0 + 0.07 * (fi - 1.5)), 0, 1)) for fi in range(4)]
    flex, spread, thumb = _mix_pose(POSE_OPEN, POSE_GRIP, c, kf)
    # holding: a tremor; the crust heating: the index and little finger clench
    if t >= T_CLOSE:
        tr = 0.025 * math.sin(1.7 * t) * math.sin(0.43 * t + 1.0)
        cl = 0.12 * float(smoothstep(T_CRACK, T_SLIP - 4, t))
        for fi, f in enumerate(FNAMES):
            e = cl if f in ('index', 'little') else 0.25 * cl
            flex[f] = tuple(v + tr * (1 + 0.3 * q) + e for q, v in enumerate(flex[f]))
    # the slip: the middle and ring fingers part and give; then the claw clenches on nothing
    s = slip(t)
    if s > 0:
        for f in FNAMES:
            give = s * (1.0 - float(smoothstep(T_SLIP + 8, T_SLIP + 18, t)))
            g = 0.28 if f in ('middle', 'ring') else 0.1
            flex[f] = tuple(v - g * give * (0.7 + 0.3 * q) for q, v in enumerate(flex[f]))
            spread[f] = spread[f] + give * {'index': -0.03, 'middle': -0.11, 'ring': 0.11, 'little': 0.03}[f]
        k2 = float(smoothstep(T_SLIP + 7, T_SLIP + 17, t))
        if k2 > 0:
            flex2, spread2, thumb2 = POSE_FIST
            for f in FNAMES:
                flex[f] = tuple(lerp(flex[f][q], flex2[f][q], k2) for q in range(3))
                spread[f] = lerp(spread[f], spread2[f], k2)
            thumb = lerp(thumb, thumb2, k2)
    return flex, spread, thumb


A_REST, R_REST = SK.pose(*_mix_pose(POSE_OPEN, POSE_OPEN, 0.0))


# ======================================================= SDF + sphere trace ===

@njit(fastmath=True, cache=True, inline='always')
def _cone(px, py, pz, j, A, BA, IL2, RA, DR):
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
    return l - (RA[j] + DR[j] * h), h, cx / l, cy / l, cz / l


@njit(fastmath=True, cache=True)
def _map(px, py, pz, A, BA, IL2, RA, DR, GS, GE, KG, KJ, GC, GR, res):
    """smooth union (within a group KG[g], across groups KJ); res: 0 nearest bone, 1 its h, 2 crease (two groups
    meeting: the seams between fingers, finger on palm), 3-5 unit gradient"""
    D = 1e9
    GX = 0.0
    GY = 0.0
    GZ = 1.0
    bj = 0
    bh = 0.0
    bd = 1e9
    d1 = 1e9
    d2 = 1e9
    for g in range(GS.shape[0]):
        sx = px - GC[g, 0]
        sy = py - GC[g, 1]
        sz = pz - GC[g, 2]
        ds = math.sqrt(sx * sx + sy * sy + sz * sz) - GR[g]
        if ds > D + 0.06:
            if ds < d2:
                d2 = ds
            continue
        k = KG[g]
        d = 1e9
        gx = 0.0
        gy = 0.0
        gz = 1.0
        for j in range(GS[g], GE[g]):
            dj, h, nx, ny, nz = _cone(px, py, pz, j, A, BA, IL2, RA, DR)
            if dj < bd:
                bd = dj
                bj = j
                bh = h
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
        if d < d1:
            d2 = d1
            d1 = d
        elif d < d2:
            d2 = d
        if D > 1e8:
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
    res[0] = bj
    res[1] = bh
    res[2] = math.exp(-(abs(d1) + abs(d2)) / 0.008) if d2 < 1e8 else 0.0     # only in a real seam
    res[3] = GX / l
    res[4] = GY / l
    res[5] = GZ / l
    return D


@njit(parallel=True, fastmath=True, cache=True)
def _march(M, O, f, cx, cy, i0, i1, j0, j1, BC, BR, A, BA, IL2, RA, DR, GS, GE, KG, KJ, GC, GR,
           o_cov, o_t, o_p, o_n, o_j, o_h, o_cr):
    """sphere-trace the claw (hand-local units, L = 1) for pixels [j0, j1) x [i0, i1). M: camera-space direction ->
    hand-local; O: the lens in hand-local. Coverage from the cone trace (the closest approach in pixels)."""
    pix = 1.0 / f
    for j in prange(j0, j1):
        res = np.empty(6)
        for i in range(i0, i1):
            x = (i - cx) / f
            y = -(j - cy) / f
            nn = math.sqrt(x * x + y * y + 1.0)
            ux = x / nn
            uy = y / nn
            uz = 1.0 / nn
            dx = M[0, 0] * ux + M[0, 1] * uy + M[0, 2] * uz
            dy = M[1, 0] * ux + M[1, 1] * uy + M[1, 2] * uz
            dz = M[2, 0] * ux + M[2, 1] * uy + M[2, 2] * uz
            ox = O[0] - BC[0]
            oy = O[1] - BC[1]
            oz = O[2] - BC[2]
            b = ox * dx + oy * dy + oz * dz
            c = ox * ox + oy * oy + oz * oz - BR * BR
            disc = b * b - c
            o_cov[j, i] = 0.0
            if disc <= 0.0:
                continue
            sq = math.sqrt(disc)
            t = -b - sq
            if t < 0.0:
                t = 0.0
            t1 = -b + sq
            best = 1e9
            bt = t
            hit = False
            for s in range(220):
                px = O[0] + dx * t
                py = O[1] + dy * t
                pz = O[2] + dz * t
                d = _map(px, py, pz, A, BA, IL2, RA, DR, GS, GE, KG, KJ, GC, GR, res)
                r = d / (pix * t + 1e-9)
                if r < best:
                    best = r
                    bt = t
                if r < 0.2:
                    hit = True
                    break
                t += d * 0.92
                if t > t1:
                    break
            cov = 1.0 if hit else 1.0 - best
            if cov <= 0.0:
                continue
            if cov > 1.0:
                cov = 1.0
            tt = t if hit else bt
            px = O[0] + dx * tt
            py = O[1] + dy * tt
            pz = O[2] + dz * tt
            _map(px, py, pz, A, BA, IL2, RA, DR, GS, GE, KG, KJ, GC, GR, res)
            o_cov[j, i] = cov
            o_t[j, i] = tt
            o_p[j, i, 0] = px
            o_p[j, i, 1] = py
            o_p[j, i, 2] = pz
            o_n[j, i, 0] = res[3]
            o_n[j, i, 1] = res[4]
            o_n[j, i, 2] = res[5]
            o_j[j, i] = int(res[0])
            o_h[j, i] = res[1]
            o_cr[j, i] = res[2]


def kernel_args(A, R):
    """the skeleton's bones as the tracer wants them, plus a bounding sphere per group and for the whole claw"""
    BA = R[:, :, 1] * SK.LEN[:, None]
    L2 = SK.LEN ** 2
    IL2 = np.where(L2 > 1e-12, 1.0 / np.maximum(L2, 1e-12), 0.0)
    ng = len(SK.GS)
    GC = np.zeros((ng, 3))
    GR = np.zeros(ng)
    for g in range(ng):
        js = range(SK.GS[g], SK.GE[g])
        pts = np.concatenate([A[list(js)], (A + BA)[list(js)]])
        c = pts.mean(0)
        r = max(float(np.linalg.norm(A[j] - c)) + SK.RA[j] for j in js)
        r = max(r, max(float(np.linalg.norm(A[j] + BA[j] - c)) + SK.RB[j] for j in js))
        GC[g], GR[g] = c, r + 0.01
    allp = np.concatenate([A, A + BA])
    BC = 0.5 * (allp.min(0) + allp.max(0))
    BR = float(np.max(np.linalg.norm(allp - BC, axis=1)) + 0.15)
    return (np.ascontiguousarray(A), np.ascontiguousarray(BA), IL2, SK.RA, SK.DR, SK.GS, SK.GE, SK.KG, float(SK.KJ),
            GC, GR, BC, BR)


# ============================================================ placement ===

def _cam_frame():
    h = np.array([math.cos(CAM_AZ), 0.0, math.sin(CAM_AZ)])           # from the Ring toward the lens
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(up, h)                                           # screen right
    return h, up, side


H_, UP_, SIDE_ = _cam_frame()


def _basis(f, p):
    f = f / np.linalg.norm(f)
    p = p - (p @ f) * f
    p = p / np.linalg.norm(p)
    x = np.cross(f, p)
    return np.stack([x, f, p], 1)                                    # columns: hand-local x, y (fingers), z (palm)


# at the grip: the fingers point right, toward the lens and down; the palm faces down and a little away
B_GRIP = _basis(SIDE_ * 0.72 + H_ * 0.42 - UP_ * 0.55, -UP_ * 0.9 - H_ * 0.35)
# at the start of the descent: steeper, the fingers reaching down
B_HIGH = _basis(SIDE_ * 0.45 + H_ * 0.3 - UP_ * 0.85, -UP_ * 0.45 - H_ * 0.3 + SIDE_ * 0.2)


def _slerp_basis(Ba, Bb, k):
    """rotation interpolation (via the relative rotation's axis-angle)"""
    Rr = Bb @ Ba.T
    ang = math.acos(float(np.clip((np.trace(Rr) - 1.0) / 2.0, -1.0, 1.0)))
    if ang < 1e-6:
        return Bb.copy()
    ax = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / (2.0 * math.sin(ang))
    a = ang * k
    x, y, z = ax
    c, s = math.cos(a), math.sin(a)
    C = 1 - c
    Rk = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                   [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
    return Rk @ Ba


_HOLLOW = [None]


def hollow():
    """where the closed claw holds the Ring (hand-local): the centre of the largest empty circle ENCLOSED by the grip
    (rays in the fingers' plane hit the claw on nearly every side), in the middle/ring fingers' plane"""
    if _HOLLOW[0] is None:
        A, R = SK.pose(*POSE_GRIP)
        ka = kernel_args(A, R)
        res = np.empty(6)
        best, bp = -1.0, None
        xg = SC.FINGERS['middle'][0]                       # the middle finger's curl plane
        dirs = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        for y in np.linspace(0.3, 0.72, 43):
            for z in np.linspace(-0.02, 0.32, 35):
                d = _map(xg, y, z, *ka[:11], res)
                if d <= best:
                    continue
                hits = 0
                for cy_, cz_ in dirs:
                    tt = 0.0
                    for _ in range(40):
                        dd = _map(xg, y + cy_ * tt, z + cz_ * tt, *ka[:11], res)
                        if dd < 1e-3:
                            hits += 1
                            break
                        tt += dd
                        if tt > 0.45:
                            break
                if hits >= 13:
                    best, bp = d, np.array([0.0, y, z])
        _HOLLOW[0] = (bp, best)
    return _HOLLOW[0]


def hand_xf(t):
    """(basis 3x3: hand-local -> world, wrist position): the claw descends from the upper left, settles so the grip's
    hollow closes on the Ring; trembles; recoils as the band slips"""
    hp, _ = hollow()
    k = descent(t)
    Bm = _slerp_basis(B_HIGH, B_GRIP, float(smoothstep(0.0, 1.0, k)))
    Wg = RP - L * (B_GRIP @ hp)
    W0 = Wg + UP_ * 34.0 - SIDE_ * 20.0 - H_ * 6.0
    # an arc (it swings in, not a straight slide)
    mid = 0.5 * (W0 + Wg) + UP_ * 3.0 - SIDE_ * 5.0
    Wp = (1 - k) ** 2 * W0 + 2 * k * (1 - k) * mid + k * k * Wg
    Wp = Wp - UP_ * 0.9 * float(smoothstep(T_CLOSE - 6, T_CLOSE + 4, t)) * (1.0 - 0.6 * float(smoothstep(T_CLOSE + 4, T_CLOSE + 20, t)))
    if t >= T_CLOSE - 2:
        a = 0.09 + 0.12 * float(smoothstep(T_CRACK, T_SLIP, t))
        Wp = Wp + a * np.array([math.sin(2.3 * t), math.sin(3.1 * t + 1.2), math.sin(1.9 * t + 2.0)])
    j = float(smoothstep(T_SLIP + 2, T_SLIP + 9, t))
    Wp = Wp + (UP_ * 2.4 - H_ * 1.2) * j + UP_ * 0.05 * max(t - T_SLIP - 9, 0.0) * j
    return Bm, Wp


# ============================================================== the Ring ===

def _rot(ax, ang):
    ax = np.asarray(ax, float)
    ax = ax / np.linalg.norm(ax)
    x, y, z = ax
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def ring_hang(t):
    """hanging, turning slowly, its hole to the lens at three-quarters"""
    Rt = _rot(np.cross(UP_, H_), math.radians(47.0)) @ _rot(UP_, CAM_AZ + math.radians(35.0) + 0.004 * (t - T_GRASP))
    C = RP + np.array([0.0, 0.15 * math.sin(0.06 * t), 0.0])
    return Rt, C


def _ring_in_hand(t, Bm, Wp):
    """in the grip: the band's axis across the fist (hand-local x), at the hollow; slipping out through the gap
    between the middle and ring fingers (hand-local +z, the fingers' side)"""
    hp, _ = hollow()
    s = slip(t)
    off = np.array([0.0, 0.03 * s, 0.26 * s])
    Rl = _rot([0.0, 0.0, 1.0], math.pi / 2) @ _rot([1.0, 0.0, 0.0], 0.5 * s)     # ring-local axis (+y) -> hand x
    return Bm @ Rl, Wp + L * (Bm @ (hp + off))


def ring_frame(t):
    """(Rot, C): hanging; pulled into the closing fist; held; slipping; falling away (ballistic, tumbling)"""
    if t < T_CLOSE - 10:
        return ring_hang(t)
    if t < T_SLIP + 9:
        Bm, Wp = hand_xf(t)
        Rh, Ch = _ring_in_hand(t, Bm, Wp)
        k = float(smoothstep(T_CLOSE - 10, T_CLOSE, t))
        Ra, Ca = ring_hang(t)
        return _slerp_basis(Ra, Rh, k), Ca + (Ch - Ca) * k
    t0 = T_SLIP + 9.0
    R0, C0 = ring_frame(t0 - 1e-3)
    R1, C1 = ring_frame(t0 - 1.0)
    v = C0 - C1
    x = t - t0
    C = C0 + v * x + np.array([0.0, -0.5 * 0.055 * x * x, 0.0]) - H_ * 0.06 * x * x * 0.1
    return _rot([0.3, 1.0, 0.5], 0.09 * x) @ R0, C


def ring_state(t):
    st = RS.RingState()
    st.heat = None
    st.write = None
    g = leak(t)
    st.letters = 0.35 + 1.8 * g + 0.25 * float(smoothstep(T_CLOSE - 30, T_CLOSE - 4, t))
    st.glow = 0.03 + 0.25 * g
    st.exposure = 1.0 + 0.5 * g
    st.alpha = 1.0
    return st


def ring_env(t, claw_glow):
    e = RS.Env(above=(0.02, 0.008, 0.003), horizon=(0.07, 0.025, 0.01), below=(0.09, 0.022, 0.008))
    e.lobe([0.0, -1.0, 0.0], [0.35, 0.08, 0.02], 2.0)              # the race's fires far below
    e.lobe([0.0, 1.0, 0.0], [0.12, 0.06, 0.02], 1.5)               # the Eye's storm above
    e.lobe([0.3, 0.6, 0.2], [0.32, 0.2, 0.08], 0.6)                # warm fill: the gold stays gold, never copper
    Bm, Wp = hand_xf(t)
    hp, _ = hollow()
    e.point(Wp + L * (Bm @ np.array([0.0, 0.45, 0.0])), np.array([1.0, 0.35, 0.08]) * (0.4 + 6.0 * claw_glow), 6.0)
    return e


# ================================================================ shading ===

C_GOLD = np.array([1.0, 0.66, 0.24])
C_CRUST = np.array([0.055, 0.045, 0.04])


def shade_claw(o, t, cam, W, H, Bm, Wp, A, R, ring_C):
    """HDR colour of the traced claw's pixels (numpy over the covered ones)"""
    cov, pl, nl, bj, hh, crz = o
    m = cov > 0.001
    iy, ix = np.nonzero(m)
    out = np.zeros((H, W, 3), np.float32)
    if len(iy) == 0:
        return out, np.zeros((H, W), np.float32)
    P = pl[iy, ix].astype(np.float64)
    N = nl[iy, ix].astype(np.float64)
    J = bj[iy, ix]
    Pw = Wp[None, :] + L * P @ Bm.T
    Nw = N @ Bm.T
    V = cam.pos[None, :] - Pw
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    ndv = np.clip((Nw * V).sum(1), 0.0, 1.0)
    # rest-pose coordinates (the crust's pattern sticks to the skin)
    q = np.einsum('nji,nj->ni', R[J], P - A[J])
    Pr = A_REST[J] + np.einsum('nij,nj->ni', R_REST[J], q)
    sharp, soft, plate, cre = SC.anat_cracks(Pr, J, SK, A_REST, R_REST)
    # EMBERS-C4: at this size anat_cracks' Worley bands are wide; narrow them, and break the big plates' seams into
    # runs (closed cell outlines on the back of the hand read as drawn loops, not cracks in a crust)
    brk = np.clip((snoise(Pr, 26.0, (4.1, 0.7, 2.6), 2) + 0.12) * 3.0, 0.0, 1.0)
    sharp = np.maximum(cre, (sharp ** 2.6) * brk)
    soft = soft * (0.45 + 0.55 * brk)
    fine = snoise(Pr, 95.0, (3.1, 7.7, 1.3), 3)
    # the crust's relief: a bump from rest-space noise, turned into the world with the bone
    e_ = 0.004
    g0 = snoise(Pr, 60.0, (1.7, 2.9, 0.4), 3)
    grad = np.stack([(snoise(Pr + np.array([e_, 0, 0]), 60.0, (1.7, 2.9, 0.4), 3) - g0),
                     (snoise(Pr + np.array([0, e_, 0]), 60.0, (1.7, 2.9, 0.4), 3) - g0),
                     (snoise(Pr + np.array([0, 0, e_]), 60.0, (1.7, 2.9, 0.4), 3) - g0)], 1) / e_
    gl = np.einsum('nij,nj->ni', R[J], np.einsum('nji,nj->ni', R_REST[J], grad))
    gw = gl @ Bm.T
    gw = gw - (gw * Nw).sum(1, keepdims=True) * Nw
    Nw = Nw - 0.007 * gw
    Nw /= np.linalg.norm(Nw, axis=1, keepdims=True)
    ndv = np.clip((Nw * V).sum(1), 0.0, 1.0)
    pores = np.clip((snoise(Pr, 210.0, (9.2, 1.1, 4.4), 2) - 0.42) * 4.0, 0, 1)
    flick = 0.6 + 0.4 * snoise(np.c_[Pr[:, :2] * 30.0, np.full(len(Pr), 0.08 * t)], 1.0, (0.0, 0.0, 0.0), 1)
    # ---- light
    Lr = ring_C[None, :] - Pw
    dr = np.linalg.norm(Lr, axis=1)
    Lr /= dr[:, None]
    lam = np.clip((Nw * Lr).sum(1), 0.0, 1.0)
    wrap = np.clip(0.4 + 0.6 * (Nw * Lr).sum(1), 0.0, 1.0) ** 2
    I_ring = (1.6 + 5.0 * leak(t)) * (1.0 - 0.85 * slip(t)) / (1.0 + (dr / 7.0) ** 2)
    top = np.clip(Nw @ np.array([0.2, 1.0, 0.1]) / 1.03, 0.0, 1.0) * 0.05
    rim = (1.0 - ndv) ** 3 * np.clip(0.35 - Nw[:, 1], 0.0, 1.0) * 0.55
    alb = C_CRUST[None, :] * (0.8 + 0.4 * (0.5 + 0.5 * fine))[:, None] * (1.0 - 0.6 * sharp)[:, None]
    col = alb * (top[:, None] * np.array([1.0, 0.6, 0.3]) + (0.6 * lam + 0.4 * wrap)[:, None] * I_ring[:, None] * C_GOLD)
    col = col + rim[:, None] * np.array([1.0, 0.18, 0.05])
    # ---- the ember life of the crust (it is made of embers): pores and the deep glow in its fissures
    hw = heat(t)
    d_ring = np.linalg.norm(Pw - ring_C[None, :], axis=1) / L
    spread = np.clip(1.0 - (d_ring - 0.9 * float(smoothstep(T_CRACK, T_SLIP - 2, t))) / 0.35, 0.0, 1.0)
    hk = hw * spread                                                     # heat reaches out from the grip
    thin = np.clip(1.4 - SK.RA[J] / 0.05, 0.3, 1.2)
    # the creases always hold a little fire (it is a claw of embers); the whole network opens as it fails; the
    # plates between stay charcoal
    crack = np.maximum(cre * (0.3 + 0.7 * hk), sharp * smoothstep(0.08, 0.6, hk))
    e_pore = pores * flick * (0.15 + 0.5 * hk) * 0.35
    e_crack = crack * (0.22 + 2.0 * hk) * flick
    e_soft = soft * hk * 0.22 * thin
    T = np.clip(0.48 + 0.3 * hk + 0.06 * crack, 0.35, 0.74)              # never white: orange at the blaze
    ecol = look.blackbody(T)
    col = col + ecol * (e_pore + e_crack + e_soft)[:, None]
    # ---- the Ring's gold leaking along the seams between the fingers (and the palm's crease)
    hp_w = ring_C
    d_hp = np.linalg.norm(Pw - hp_w[None, :], axis=1) / L
    lk = crz[iy, ix] * np.exp(-(d_hp / 0.13) ** 2) * leak(t) * 2.4
    col = col + lk[:, None] * np.array([1.0, 0.72, 0.32])
    c = cov[iy, ix]
    out[iy, ix] = (col * c[:, None]).astype(np.float32)
    a = np.zeros((H, W), np.float32)
    a[iy, ix] = c
    return out, a


# ============================================================== particles ===

class ClawEmbers:
    """embers shed off the crust: trailing up behind it as it descends; flakes falling off it as it fails"""

    def __init__(self, seed=5117, n=9000):
        r = rng(seed)
        own, Q, area = SK.sample(r, 4200.0, 1500.0)
        pick = r.choice(len(own), n, replace=len(own) < n)
        self.own = own[pick]
        self.Q = Q[pick]
        self.tb = r.uniform(T_GRASP - 20, T_GRASP_END, n)
        self.life = r.uniform(10.0, 30.0, n)
        self.v = r.normal(0, 1, (n, 3)) * 0.05
        self.E = r.lognormal(0, 0.6, n)
        self.T = r.uniform(0.45, 0.75, n)

    def pos(self, idx, tb):
        """where each ember left the skin (world)"""
        out = np.zeros((len(idx), 3))
        for k in np.unique(np.round(tb[idx]).astype(int)):
            m = np.round(tb[idx]).astype(int) == k
            Bm, Wp = hand_xf(float(k))
            A, R = SK.pose(*pose(float(k)))
            j = self.own[idx[m]]
            loc = A[j] + np.einsum('nij,nj->ni', R[j], self.Q[idx[m]])
            out[m] = Wp[None, :] + L * loc @ Bm.T
        return out

    def emit(self, ctx):
        t = ctx.t
        age = t - self.tb
        m = (age >= 0) & (age < self.life)
        idx = np.nonzero(m)[0]
        if len(idx) == 0:
            return
        P = self.pos(idx, self.tb)
        hw = heat(t)
        fall = float(smoothstep(T_CRACK, T_SLIP, t))

        def at(tq):
            a = np.maximum(tq - self.tb[idx], 0.0)
            drift = self.v[idx] * a[:, None] + np.array([0.0, 0.03, 0.0]) * a[:, None] * (1 - fall) \
                - np.array([0.0, 0.004, 0.0]) * (a ** 2)[:, None] * fall
            return P + drift
        P0, P1 = at(ctx.t0), at(ctx.t1)
        u = age[idx] / self.life[idx]
        e = self.E[idx] * (1 - u) ** 1.5 * (0.5 + 2.5 * hw + 0.6 * leak(t)) * 0.8
        col = look.blackbody(np.clip(self.T[idx] + 0.25 * hw - 0.3 * u, 0.3, 0.95))
        ctx.fr.splat(P0, P1, 0.05, e, col, ctx.cam0, ctx.cam1, profile=1)


# ============================================================ schedule ===

class GraspSched(c3.C3Sched):
    """C's forges as the race left them, far below the Ring"""
    end = float(T_GRASP_END + 1)
    pulses = []

    def beat_pulse(self, t):
        return 0.0

    def tower_lean(self, t):
        return 0.0


SCHED = GraspSched()


def use():
    B.SCHED = SCHED
    B.IGN = SCHED.ign
    B.BEATS = list(SCHED.beats)


# =============================================================== shot ===

class GraspShot:
    def __init__(self, lab=False):
        use()
        self.lab = lab
        self._cache = {}
        self.embers = ClawEmbers()
        hollow()

    def _get(self, name, make):
        if name not in self._cache:
            use()
            self._cache[name] = make()
        return self._cache[name]

    @property
    def towers(self):
        return self._get('towers', lambda: c3.layout_towers(B.Towers()))

    def camera(self, t):
        use()
        u = float(smoothstep(T_GRASP, T_GRASP_END, t))
        d = CAM_D - 3.0 * u
        pos = RP + H_ * d + UP_ * (CAM_UP - 0.6 * u)
        tgt = RP + UP_ * (2.2 - 1.6 * float(smoothstep(T_CLOSE - 20, T_CLOSE + 10, t))) - SIDE_ * 0.6
        if t >= T_SLIP + 4:                                   # the eye follows the band a little as it falls
            k = float(smoothstep(T_SLIP + 4, T_GRASP_END, t))
            tgt = tgt - UP_ * 3.5 * k
        return Camera(pos, tgt, hfov=36.0 - 2.0 * u, focus=float(np.linalg.norm(RP - pos)), aperture=0.05)

    def render_opts(self, f):
        return dict(bokeh_pow=0.3, bokeh_cap=2.2, fog_start=60.0, fog_len=90.0, near=0.3)

    def light(self, t):
        return RP.copy(), np.array([1.0, 0.7, 0.35]), 40.0 * (1.0 + leak(t))

    def emit(self, ctx):
        use()
        t = ctx.t
        W, H = ctx.fr.W, ctx.fr.H
        cam = ctx.cam
        # the claw and the Ring first: they join the occluder, so what stands behind them is hidden
        Bm, Wp = hand_xf(t)
        A, R = SK.pose(*pose(t))
        ka = kernel_args(A, R)
        M = Bm.T @ cam.R.T
        O = Bm.T @ (cam.pos - Wp) / L
        f = cam.f_px(W)
        BC, BR = ka[11], ka[12]
        # screen bounds of the claw's bounding sphere
        cw = Wp + L * (Bm @ BC)
        u, v, z = cam.project(cw[None], W, H)
        rpx = f * BR * L / max(float(z[0]), 1.0) * 1.15 + 8
        i0, i1 = int(max(u[0] - rpx, 0)), int(min(u[0] + rpx, W))
        j0, j1 = int(max(v[0] - rpx, 0)), int(min(v[0] + rpx, H))
        o_cov = np.zeros((H, W), np.float32)
        o_t = np.zeros((H, W), np.float32)
        o_p = np.zeros((H, W, 3), np.float32)
        o_n = np.zeros((H, W, 3), np.float32)
        o_j = np.zeros((H, W), np.int32)
        o_h = np.zeros((H, W), np.float32)
        o_cr = np.zeros((H, W), np.float32)
        if i1 > i0 and j1 > j0 and z[0] > 0:
            _march(np.ascontiguousarray(M), O, f, (W - 1) / 2.0, (H - 1) / 2.0, i0, i1, j0, j1, BC, BR, *ka[:11],
                   o_cov, o_t, o_p, o_n, o_j, o_h, o_cr)
        # camera depth of the claw
        xs = (np.arange(W) - (W - 1) / 2.0) / f
        ys = -(np.arange(H) - (H - 1) / 2.0) / f
        nn = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2 + 1.0)
        hd = np.where(o_cov > 0, o_t * L / nn, np.inf).astype(np.float32)
        Rr, Cr = ring_frame(t)
        ctx.claw = (Bm, Wp, A, R, (o_cov, o_p, o_n, o_j, o_h, o_cr), hd, Cr)
        # the Ring (three instants across the shutter)
        acc = None
        for tq in (ctx.t0, 0.5 * (ctx.t0 + ctx.t1), ctx.t1):
            cq = self.camera(tq)
            Rq, Cq = ring_frame(tq)
            rgb, a, d = RS.render(cq, W, H, Rq, Cq, RING_W, ring_state(tq), ring_env(tq, heat(tq)))
            if acc is None:
                acc = [rgb, a, d]
            else:
                acc[0] += rgb
                acc[1] += a
                acc[2] = np.minimum(acc[2], d)
        ctx.ring = (acc[0] / 3.0, acc[1] / 3.0, acc[2])
        if not self.lab:
            tw = self.towers
            tw.prepare(ctx)
        # claw + ring join the occluder
        a_all = np.maximum(o_cov, ctx.ring[1])
        d_all = np.minimum(hd, np.where(ctx.ring[1] > 0.3, ctx.ring[2], np.inf)).astype(np.float32)
        ctx.before = RS.merge_occluder(ctx.fr, a_all, d_all)
        lp, lc, lpw = self.light(t)
        dust = self._get('dust', B.Dust)
        dust.emit(ctx)
        if not self.lab:
            smoke = self._get('smoke', B.Smoke)
            smoke.emit(ctx, lp, lc, lpw)
            tw.emit(ctx, lp, lc, lpw)
            self._get('tembers', lambda: B.TowerEmbers(tw)).emit(ctx)
            self._get('tsmoke', lambda: B.TowerSmoke(tw)).emit(ctx, lp, lc, lpw)
        self.embers.emit(ctx)

    def post(self, ctx, hdr):
        t = ctx.t
        W, H = ctx.fr.W, ctx.fr.H
        cam = ctx.cam
        Bm, Wp, A, R, o, hd, Cr = ctx.claw
        # ---- the storm behind everything (far), lit from below by the race and from above by the Eye
        n = -cam.R[2]
        C = RP - n * 95.0 + UP_ * 26.0
        ex = cam.R[0]
        ey = np.cross(n, ex)
        X, Y, ok = c2.plane_coords(cam, W, H, C, ex, ey, n, 30.0)
        pr = np.array([51.0, 0.5, 0.1, 0.55, 0.0, 0.0, 0.1, 0.0, 9.0, 0.0, 0.01, 0.0, 1.0], np.float64)
        lp = np.array([0.22, 1.6, 0.5, 0.15, 2.2, 0.3, 0.3, 1.0, 0.06, 0.02, 0.0], np.float64)
        srgb = np.empty((H, W, 3), np.float32)
        sa = np.empty((H, W), np.float32)
        c2.storm_layer(X, Y, ok, float(t), pr, lp, srgb, sa)
        dark = 1.0 - float(smoothstep(T_SLIP + 6, T_GRASP_END, t)) * 0.7      # into the black
        bg = srgb * dark
        vis_bg = np.ones((H, W), np.float32)
        if ctx.before is not None:
            A0 = ctx.before[1][0]
            Af = cv2.resize(A0, (A0.shape[1] * 2, A0.shape[0] * 2), interpolation=cv2.INTER_LINEAR)[:H, :W]
            vis_bg = 1.0 - np.clip(Af, 0.0, 1.0)
        hdr = hdr + bg * vis_bg[..., None]
        # ---- the claw and the Ring, z-merged, over the towers (their own depth against the towers' occluder)
        crgb, ca = shade_claw(o, t, cam, W, H, Bm, Wp, A, R, Cr)
        rrgb, ra, rd = ctx.ring
        ring_front = rd < hd
        top_rgb = np.where(ring_front[..., None], rrgb + (1 - ra[..., None]) * crgb, crgb + (1 - ca[..., None]) * rrgb)
        top_a = np.where(ring_front, ra + (1 - ra) * ca, ca + (1 - ca) * ra)
        vis = np.ones((H, W), np.float32)
        if ctx.before is not None:
            vis = RS.visibility(ctx.before, np.minimum(hd, rd), H, W)
        hdr = hdr * (1.0 - top_a[..., None] * vis[..., None]) + top_rgb * vis[..., None]
        # ---- the gold's glow in the smoke round the fist while it holds
        g = leak(t)
        if g > 0.01:
            u, v, z = cam.project(Cr[None], W, H)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            rr = np.sqrt((xx - u[0]) ** 2 + (yy - v[0]) ** 2) / (0.12 * W)
            glow = np.exp(-rr * rr) * 0.09 * g + np.exp(-rr) * 0.03 * g
            hdr = hdr + glow[..., None] * np.array([1.0, 0.6, 0.22], np.float32) * (1.0 - 0.6 * ca[..., None])
        return hdr

    def finish_opts(self, f):
        return dict(exposure=1.0, bloom_strength=0.12, bloom_threshold=1.0, streak_strength=0.0, vignette_amount=0.3)
