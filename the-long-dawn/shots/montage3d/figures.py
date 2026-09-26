"""MONTAGE-3D figures (venv side): people as ONE smooth signed-distance body per frame.

Why: rigid capsules read as mannequins (tube ends at shoulders/hips). Here every part (round cones,
ellipsoids) is smooth-min blended into the body with its own fillet radius, cloth gets fold and wind
displacement, and a numba surface-nets mesher turns the SDF into a watertight mesh per frame.
Blender (kit/figure.py: MeshSeq) swaps each frame's mesh in at render time.

Figure-local space: origin between the feet on the ground, +Y forward, +Z up, +X the figure's right.
"""
import json
import math
import os
import struct
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'shots', 'montage'))
from mt.noise import gnoise3  # noqa: E402

# ------------------------------------------------------------------ algebra ---


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def V(*a):
    return np.array(a, np.float64)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


def interp_pose(frame, keys):
    names = set()
    for _, d in keys:
        names.update(d.keys())
    out = {}
    for nm in names:
        kf = [(f, d[nm]) for f, d in keys if nm in d]
        if frame <= kf[0][0]:
            out[nm] = kf[0][1]
            continue
        if frame >= kf[-1][0]:
            out[nm] = kf[-1][1]
            continue
        for (f0, v0), (f1, v1) in zip(kf[:-1], kf[1:]):
            if f0 <= frame <= f1:
                u = ease((frame - f0) / (f1 - f0)) if f1 > f0 else 1.0
                out[nm] = v0 + (v1 - v0) * u
                break
    return out


# ------------------------------------------------------------------- skeleton ---

DEFAULT = dict(lean=0.0, side_lean=0.0, twist=0.0, head_yaw=0.0, head_pitch=0.0, head_roll=0.0, crouch=0.0,
               r_flex=0.1, r_abd=0.08, r_elbow=0.25, l_flex=0.05, l_abd=0.08, l_elbow=0.2,
               r_hip=0.0, r_knee=0.05, l_hip=0.0, l_knee=0.05, r_toe_out=0.15, l_toe_out=0.15,
               stance=0.09, breath=0.0, hip_shift=0.0, sh_shrug=0.0)
DIMS = dict(hip_h=0.93, spine=0.21, chest=0.2, neck=0.095, head=0.115, sh_w=0.185, sh_h=0.035, upper=0.29,
            fore=0.26, hand=0.085, thigh=0.44, shin=0.43, foot=0.2, hip_w=0.095)


def fk(p, dims=None):
    """Joint positions (figure-local) + frames. Abduction swings an arm OUTWARD for both sides."""
    d = dict(DEFAULT)
    d.update(p)
    p = d
    D = dict(DIMS)
    if dims:
        D.update(dims)
    J = {}
    cr = p['crouch']
    pel = V(p['hip_shift'], 0.02 * cr, D['hip_h'] * (1 - 0.14 * cr))
    Rp = rz(-p['twist'] * 0.4) @ ry(p['side_lean'] * 0.5) @ rx(-p['lean'] * 0.5)
    J['pelvis'] = pel
    J['R_pelvis'] = Rp
    Rs = rz(-p['twist'] * 0.3) @ Rp @ rx(-p['lean'] * 0.5) @ ry(-p['side_lean'] * 1.2)
    J['spine'] = pel + Rp @ V(0, 0, D['spine'])
    J['chest'] = J['spine'] + Rs @ V(0, 0.01 + 0.004 * p['breath'], D['chest'])
    J['R_chest'] = Rs
    J['neck'] = J['chest'] + Rs @ V(0, 0.012, D['neck'])
    Rh = Rs @ rz(-p['head_yaw']) @ rx(p['head_pitch']) @ ry(p['head_roll'])
    J['R_head'] = Rh
    J['head'] = J['neck'] + Rh @ V(0, 0.02, D['head'])
    for side, sg in (('r', 1.0), ('l', -1.0)):
        sh = J['chest'] + Rs @ V(sg * D['sh_w'], -0.012, D['sh_h'] + 0.03 * p['sh_shrug'])
        J[side + '_sh'] = sh
        flex, abd, elb = p[side + '_flex'], p[side + '_abd'], p[side + '_elbow']
        Ru = Rs @ rx(flex) @ ry(-sg * abd)                  # -Z hanging; abd -> outward (+sg X)
        el = sh + Ru @ V(0, 0, -D['upper'])
        Rf = Ru @ rx(elb)
        wr = el + Rf @ V(0, 0, -D['fore'])
        hd = wr + Rf @ V(0, 0, -D['hand'])
        J[side + '_el'], J[side + '_wr'], J[side + '_hand'] = el, wr, hd
        J['R_' + side + '_fore'] = Rf
        hp = pel + Rp @ V(sg * D['hip_w'], 0, -0.02)
        J[side + '_hip'] = hp
        hf = p[side + '_hip'] + 0.35 * cr
        kb = p[side + '_knee'] + 0.7 * cr
        st = p['stance']
        Rt = rx(hf) @ ry(sg * math.atan2(st - D['hip_w'], D['hip_h']))
        kn = hp + Rt @ V(0, 0, -D['thigh'])
        Rsn = Rt @ rx(-kb)
        an = kn + Rsn @ V(0, 0, -D['shin'])
        an[2] = max(an[2], 0.075)
        J[side + '_knee'], J[side + '_ankle'] = kn, an
        to = p[side + '_toe_out']
        J[side + '_toe'] = an + V(sg * math.sin(to) * D['foot'], math.cos(to) * D['foot'], -0.055)
        J[side + '_heel'] = an + V(-sg * math.sin(to) * 0.05, -math.cos(to) * 0.05, -0.06)
    return J


def ik_arm(J, side, target, pole=(0.35, -0.3, -1.0), dims=None):
    D = dict(DIMS)
    if dims:
        D.update(dims)
    L1, L2, Lh = D['upper'], D['fore'], D['hand']
    S = J[side + '_sh']
    tgt = np.asarray(target, np.float64)
    fdir = (tgt - S) / (np.linalg.norm(tgt - S) + 1e-9)
    wt = tgt - fdir * Lh
    dw = wt - S
    dist = min(max(np.linalg.norm(dw), 0.05), L1 + L2 - 1e-3)
    dn = dw / (np.linalg.norm(dw) + 1e-9)
    ca = (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist)
    ca = min(max(ca, -1.0), 1.0)
    sa = math.sqrt(max(0.0, 1 - ca * ca))
    pv = np.asarray(pole, np.float64)
    pv = pv - dn * pv.dot(dn)
    pv /= (np.linalg.norm(pv) + 1e-9)
    el = S + dn * L1 * ca + pv * L1 * sa
    wr = S + dn * dist
    hd = wr + (wr - el) / (np.linalg.norm(wr - el) + 1e-9) * Lh
    J[side + '_el'], J[side + '_wr'], J[side + '_hand'] = el, wr, hd
    return J


# ------------------------------------------------------------------ SDF body ---
# primitive rows: [type, ax, ay, az, bx, by, bz, r1, r2, k, mat, op, R(9)]
#   type 0: round cone a->b radii r1->r2;  type 1: ellipsoid centre a, radii (bx, by, bz), rotation R
#   op 0: smooth union with fillet k; op 1: smooth subtraction with k
NCOL = 21


def cone(a, b, r1, r2, k=0.03, mat=0):
    row = np.zeros(NCOL)
    row[0] = 0
    row[1:4] = a
    row[4:7] = b
    row[7], row[8], row[9], row[10], row[11] = r1, r2, k, mat, 0
    row[12:21] = np.eye(3).ravel()
    return row


def ell(c, radii, R=None, k=0.03, mat=0, op=0):
    row = np.zeros(NCOL)
    row[0] = 1
    row[1:4] = c
    row[4:7] = radii
    row[9], row[10], row[11] = k, mat, op
    row[12:21] = (np.eye(3) if R is None else R).ravel()
    return row


@njit(inline="always", fastmath=True, cache=True)
def _roundcone(px, py, pz, ax, ay, az, bx, by, bz, r1, r2):
    bax, bay, baz = bx - ax, by - ay, bz - az
    l2 = bax * bax + bay * bay + baz * baz
    if l2 < 1e-10:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2) - r1
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pax, pay, paz = px - ax, py - ay, pz - az
    y = pax * bax + pay * bay + paz * baz
    z = y - l2
    qx, qy, qz = pax * l2 - bax * y, pay * l2 - bay * y, paz * l2 - baz * y
    x2 = qx * qx + qy * qy + qz * qz
    y2 = y * y * l2
    z2 = z * z * l2
    sr = 1.0 if rr > 0 else (-1.0 if rr < 0 else 0.0)
    k = sr * rr * rr * x2
    sz = 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)
    sy = 1.0 if y > 0 else (-1.0 if y < 0 else 0.0)
    if sz * a2 * z2 > k:
        return math.sqrt(x2 + z2) * il2 - r2
    if sy * a2 * y2 < k:
        return math.sqrt(x2 + y2) * il2 - r1
    return (math.sqrt(max(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1


@njit(inline="always", fastmath=True, cache=True)
def _ellipsoid(px, py, pz, P):
    dx, dy, dz = px - P[1], py - P[2], pz - P[3]
    # into the ellipsoid frame (R^T d)
    lx = P[12] * dx + P[15] * dy + P[18] * dz
    ly = P[13] * dx + P[16] * dy + P[19] * dz
    lz = P[14] * dx + P[17] * dy + P[20] * dz
    k0 = math.sqrt((lx / P[4]) ** 2 + (ly / P[5]) ** 2 + (lz / P[6]) ** 2)
    k1 = math.sqrt((lx / (P[4] * P[4])) ** 2 + (ly / (P[5] * P[5])) ** 2 + (lz / (P[6] * P[6])) ** 2)
    if k1 < 1e-9:
        return -min(P[4], min(P[5], P[6]))
    return k0 * (k0 - 1.0) / k1


@njit(inline="always", fastmath=True, cache=True)
def _smin(a, b, k):
    if k <= 0.0:
        return min(a, b)
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


@njit(fastmath=True, cache=True)
def sdf_point(px, py, pz, PR, disp):
    d = 1e9
    best = 1e9
    mat = 0.0
    for i in range(PR.shape[0]):
        P = PR[i]
        if P[0] == 0:
            di = _roundcone(px, py, pz, P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8])
        else:
            di = _ellipsoid(px, py, pz, P)
        if P[11] == 0:
            d = _smin(d, di, P[9])
            if di < best:
                best = di
                mat = P[10]
        else:
            d = -_smin(-d, di, P[9])       # smooth subtraction of the primitive
    # cloth folds / wrinkles: low-amplitude gradient noise displacement
    if disp[0] > 0.0:
        n1 = gnoise3(px * disp[1] + disp[3], py * disp[1], pz * disp[1] * 0.6, 11)
        n2 = gnoise3(px * disp[2], py * disp[2] + disp[3], pz * disp[2], 12)
        d += disp[0] * (0.7 * n1 + 0.3 * n2)
    return d, mat


@njit(parallel=True, fastmath=True, cache=True)
def _grid(PR, disp, x0, y0, z0, h, nx, ny, nz, phi, mat):
    for i in prange(nx):
        px = x0 + i * h
        for j in range(ny):
            py = y0 + j * h
            for k in range(nz):
                d, m = sdf_point(px, py, z0 + k * h, PR, disp)
                phi[i, j, k] = d
                mat[i, j, k] = m


@njit(fastmath=True, cache=True)
def _surface_nets(phi, mat, x0, y0, z0, h, V, M, Q):
    nx, ny, nz = phi.shape
    vid = -np.ones((nx, ny, nz), np.int32)
    nv = 0
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                inside = 0
                for c in range(8):
                    if phi[i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)] < 0:
                        inside += 1
                if inside == 0 or inside == 8:
                    continue
                sx = 0.0
                sy = 0.0
                sz = 0.0
                cnt = 0
                # 12 edges
                for e in range(12):
                    if e < 4:
                        a0 = (0, e & 1, (e >> 1) & 1)
                        a1 = (1, e & 1, (e >> 1) & 1)
                    elif e < 8:
                        a0 = (e & 1, 0, ((e - 4) >> 1) & 1)
                        a1 = (e & 1, 1, ((e - 4) >> 1) & 1)
                    else:
                        a0 = (e & 1, ((e - 8) >> 1) & 1, 0)
                        a1 = (e & 1, ((e - 8) >> 1) & 1, 1)
                    v0 = phi[i + a0[0], j + a0[1], k + a0[2]]
                    v1 = phi[i + a1[0], j + a1[1], k + a1[2]]
                    if (v0 < 0) != (v1 < 0):
                        t = v0 / (v0 - v1)
                        sx += a0[0] + t * (a1[0] - a0[0])
                        sy += a0[1] + t * (a1[1] - a0[1])
                        sz += a0[2] + t * (a1[2] - a0[2])
                        cnt += 1
                V[nv, 0] = x0 + (i + sx / cnt) * h
                V[nv, 1] = y0 + (j + sy / cnt) * h
                V[nv, 2] = z0 + (k + sz / cnt) * h
                M[nv] = mat[i, j, k]
                vid[i, j, k] = nv
                nv += 1
    nq = 0
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # x edges
                if j > 0 and k > 0:
                    a = phi[i, j, k] < 0
                    b = phi[i + 1, j, k] < 0
                    if a != b:
                        q0, q1, q2, q3 = vid[i, j - 1, k - 1], vid[i, j, k - 1], vid[i, j, k], vid[i, j - 1, k]
                        if a:
                            Q[nq, 0], Q[nq, 1], Q[nq, 2], Q[nq, 3] = q0, q1, q2, q3
                        else:
                            Q[nq, 0], Q[nq, 1], Q[nq, 2], Q[nq, 3] = q3, q2, q1, q0
                        nq += 1
                if i > 0 and k > 0:
                    a = phi[i, j, k] < 0
                    b = phi[i, j + 1, k] < 0
                    if a != b:
                        q0, q1, q2, q3 = vid[i - 1, j, k - 1], vid[i - 1, j, k], vid[i, j, k], vid[i, j, k - 1]
                        if a:
                            Q[nq, 0], Q[nq, 1], Q[nq, 2], Q[nq, 3] = q0, q1, q2, q3
                        else:
                            Q[nq, 0], Q[nq, 1], Q[nq, 2], Q[nq, 3] = q3, q2, q1, q0
                        nq += 1
                if i > 0 and j > 0:
                    a = phi[i, j, k] < 0
                    b = phi[i, j, k + 1] < 0
                    if a != b:
                        q0, q1, q2, q3 = vid[i - 1, j - 1, k], vid[i, j - 1, k], vid[i, j, k], vid[i - 1, j, k]
                        if a:
                            Q[nq, 0], Q[nq, 1], Q[nq, 2], Q[nq, 3] = q0, q1, q2, q3
                        else:
                            Q[nq, 0], Q[nq, 1], Q[nq, 2], Q[nq, 3] = q3, q2, q1, q0
                        nq += 1
    return nv, nq


def mesh_sdf(PR, h=0.011, disp=(0.0, 0.0, 0.0, 0.0), pad=0.12, bounds=None):
    """Polygonize the smooth body. Returns (verts float32 Nx3, quads int32 Mx4, mat float32 N)."""
    PR = np.ascontiguousarray(np.array(PR, np.float64))
    if bounds is None:
        pts = np.vstack([PR[:, 1:4], np.where(PR[:, [0]] == 0, PR[:, 4:7], PR[:, 1:4])])
        rmax = np.max(np.maximum(PR[:, 7], np.where(PR[:, 0] == 1, PR[:, 4:7].max(axis=1), PR[:, 8])))
        lo = pts.min(axis=0) - rmax - pad
        hi = pts.max(axis=0) + rmax + pad
    else:
        lo, hi = np.array(bounds[0]), np.array(bounds[1])
    n = np.ceil((hi - lo) / h).astype(int) + 1
    phi = np.empty((n[0], n[1], n[2]), np.float32)
    mat = np.empty((n[0], n[1], n[2]), np.float32)
    _grid(PR, np.array(disp, np.float64), lo[0], lo[1], lo[2], h, n[0], n[1], n[2], phi, mat)
    cap = int(np.count_nonzero(np.abs(phi) < 2.0 * h)) + 16
    Vv = np.zeros((cap, 3), np.float64)
    Mm = np.zeros(cap, np.float32)
    Qq = np.zeros((cap * 3, 4), np.int32)
    nv, nq = _surface_nets(phi, mat, lo[0], lo[1], lo[2], h, Vv, Mm, Qq)
    return Vv[:nv].astype(np.float32), Qq[:nq].copy(), Mm[:nv].copy()


def write_mesh(path, V, Q, M):
    with open(path, 'wb') as f:
        f.write(struct.pack('<ii', len(V), len(Q)))
        f.write(np.ascontiguousarray(V, np.float32).tobytes())
        f.write(np.ascontiguousarray(Q, np.int32).tobytes())
        f.write(np.ascontiguousarray(M, np.float32).tobytes())


# ------------------------------------------------------------ body builders ---

def _R_from(z_axis, y_hint):
    z = np.asarray(z_axis, np.float64)
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.asarray(y_hint, np.float64)
    x = np.cross(y, z)
    x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def hoodie_body(J, t=0.0):
    """A slim young person in a loose hoodie (hood up), slim jeans, sneakers. mat: 0 hoodie, 1 jeans,
    2 skin (hands), 3 shoes."""
    P = []
    Rp, Rc, Rh = J['R_pelvis'], J['R_chest'], J['R_head']
    pel, sp, ch, nk, hd = J['pelvis'], J['spine'], J['chest'], J['neck'], J['head']
    # --- hoodie torso: hips / belly / chest / shoulder girdle, generously filleted (loose cloth)
    P.append(ell(pel + Rp @ V(0, 0.0, -0.035), (0.165, 0.118, 0.12), Rp, k=0.0, mat=1))       # jeans hips
    P.append(ell(pel + Rp @ V(0, 0.005, 0.07), (0.17, 0.122, 0.13), Rp, k=0.05, mat=0))       # hoodie hem/belly
    P.append(ell(sp + Rc @ V(0, 0.01, 0.02), (0.162, 0.125, 0.16), Rc, k=0.07, mat=0))
    P.append(ell(ch + Rc @ V(0, 0.012, -0.03), (0.172, 0.128, 0.14), Rc, k=0.07, mat=0))
    P.append(cone(J['l_sh'] + Rc @ V(0.02, 0, 0.0), J['r_sh'] + Rc @ V(-0.02, 0, 0.0), 0.075, 0.075, k=0.08, mat=0))
    # trapezius slope from the collar to the shoulders
    P.append(cone(nk + Rc @ V(0, -0.02, -0.02), J['l_sh'] + Rc @ V(0.03, 0, 0.02), 0.06, 0.05, k=0.06, mat=0))
    P.append(cone(nk + Rc @ V(0, -0.02, -0.02), J['r_sh'] + Rc @ V(-0.03, 0, 0.02), 0.06, 0.05, k=0.06, mat=0))
    # --- arms (baggy sleeves, cuffs, hands)
    for s in ('l', 'r'):
        P.append(cone(J[s + '_sh'], J[s + '_el'], 0.066, 0.058, k=0.05, mat=0))
        P.append(cone(J[s + '_el'], J[s + '_wr'], 0.058, 0.047, k=0.03, mat=0))
        P.append(cone(J[s + '_wr'], J[s + '_wr'] + (J[s + '_hand'] - J[s + '_wr']) * 0.25, 0.046, 0.043, k=0.01, mat=0))
        P.append(cone(J[s + '_wr'] + (J[s + '_hand'] - J[s + '_wr']) * 0.3, J[s + '_hand'], 0.036, 0.032, k=0.015, mat=2))
    # --- legs (slim jeans), sneakers
    for s in ('l', 'r'):
        P.append(cone(J[s + '_hip'] + V(0, 0, 0.03), J[s + '_knee'], 0.085, 0.058, k=0.06, mat=1))
        P.append(cone(J[s + '_knee'], J[s + '_ankle'], 0.057, 0.047, k=0.025, mat=1))
        # calf
        P.append(ell(J[s + '_knee'] * 0.62 + J[s + '_ankle'] * 0.38 + V(0, -0.012, 0), (0.052, 0.056, 0.11), k=0.03, mat=1))
        heel, toe = J[s + '_heel'], J[s + '_toe']
        P.append(cone(heel + V(0, 0, 0.03), toe + V(0, 0, 0.022), 0.045, 0.038, k=0.02, mat=3))
        P.append(cone(heel + V(0, 0, 0.012), toe + V(0, 0, 0.01), 0.04, 0.036, k=0.015, mat=3))
    # --- neck, head, hood (dome + crown peak + the fold falling onto the upper back)
    P.append(cone(ch + Rc @ V(0, 0, 0.02), nk + Rc @ V(0, 0.01, 0.04), 0.06, 0.055, k=0.03, mat=2))
    P.append(ell(hd, (0.076, 0.092, 0.105), Rh, k=0.02, mat=2))
    P.append(ell(hd + Rh @ V(0, -0.015, 0.015), (0.1, 0.116, 0.125), Rh, k=0.02, mat=0))
    P.append(ell(hd + Rh @ V(0, -0.05, 0.075), (0.06, 0.066, 0.075), Rh, k=0.05, mat=0))
    P.append(cone(hd + Rh @ V(0, -0.055, -0.03), nk + Rc @ V(0, -0.075, -0.05), 0.085, 0.09, k=0.06, mat=0))
    # face opening: carve the front of the hood (only ever seen edge-on)
    P.append(ell(hd + Rh @ V(0, 0.105, -0.01), (0.065, 0.05, 0.085), Rh, k=0.02, mat=2, op=1))
    return P
