"""FIRST BEACON heroine (v2): the Young Woman as a sculpted 3-D SDF figure.

Rig + sculpt (this file) and the sphere-tracing renderer (heroine_sdf.py). Used only by
beacon.py when HEROINE_V2 is on; INTRO/CODA (the Elder, the Child) never import it.

Conventions: world metres, x right, y up, z away from camera. She faces -x (left). Local
frames are (U forward, V up, W lateral toward the camera = her near side).
"""
import math

import numpy as np

import heroine_sdf as hs
from heroine_sdf import (T_ELL, T_CONE, T_BOX, T_TORUS, T_PLANE, T_LID, NPR, M_SKIN, M_EYE, M_HAIR,
                         M_COAT, M_SCARF, M_CLOTH, M_BOOT, M_STEEL, M_FLINT, M_NAIL, M_KNIT, NMAT)

# ------------------------------------------------------------ materials ---

def material_table():
    M = np.zeros((NMAT, 16))
    #           albedo                rough  F0    wrap  sheen trans rim  amb  metal tex  sc    amp
    M[M_SKIN] = [0.34, 0.172, 0.118, 0.44, 0.028, 0.20, 0.10, 0.30, 1.0, 1.0, 0.0, 2, 180.0, 0.07, 0, 0]
    M[M_EYE] = [0.58, 0.52, 0.47, 0.05, 0.030, 0.0, 0.0, 0.0, 0.6, 1.0, 0.0, 0, 0.0, 0.0, 0, 0]
    M[M_HAIR] = [0.022, 0.014, 0.010, 0.32, 0.046, 0.0, 0.9, 0.25, 1.6, 1.0, 0.0, 0, 0.0, 0.0, 0, 0]
    M[M_COAT] = [0.050, 0.040, 0.034, 0.92, 0.020, 0.10, 0.9, 0.0, 1.2, 1.0, 0.0, 1, 45.0, 0.22, 0.10, 0]
    M[M_SCARF] = [0.341, 0.0103, 0.0103, 0.85, 0.025, 0.25, 1.4, 0.6, 1.3, 1.0, 0.0, 1, 70.0, 0.18, 0.14, 0]   # scarf_red #9E1B1B
    M[M_CLOTH] = [0.030, 0.028, 0.030, 0.90, 0.020, 0.05, 0.6, 0.0, 1.0, 1.0, 0.0, 1, 60.0, 0.25, 0.10, 0]
    M[M_BOOT] = [0.040, 0.026, 0.017, 0.42, 0.040, 0.0, 0.2, 0.0, 1.0, 1.0, 0.0, 1, 50.0, 0.25, 0.10, 0]
    M[M_STEEL] = [0.30, 0.29, 0.28, 0.30, 0.50, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, 0.0, 0.0, 0, 0]
    M[M_FLINT] = [0.070, 0.062, 0.055, 0.22, 0.045, 0.0, 0.0, 0.05, 1.0, 1.0, 0.0, 1, 70.0, 0.4, 0.25, 0]
    M[M_NAIL] = [0.40, 0.25, 0.20, 0.30, 0.035, 0.2, 0.0, 0.3, 1.0, 1.0, 0.0, 0, 0.0, 0.0, 0, 0]
    M[M_KNIT] = [0.042, 0.037, 0.034, 0.95, 0.020, 0.10, 1.1, 0.0, 1.3, 1.0, 0.0, 1, 110.0, 0.30, 0.16, 0]
    return M


# -------------------------------------------------------------- builder ---

def nrm(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def perp_frame(axis):
    a = nrm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = nrm(np.cross(a, ref))
    e2 = np.cross(a, e1)
    return e1, e2


class Frame:
    """Local frame: origin C, world axes U (fwd), V (up), W (lateral, toward camera)."""

    def __init__(self, C, U, V, W, s=1.0):
        self.C = np.asarray(C, np.float64)
        self.U = nrm(U)
        self.V = nrm(V)
        self.W = nrm(W)
        self.s = s

    def p(self, u, v, w=0.0):
        return self.C + self.s * (u * self.U + v * self.V + w * self.W)

    def R(self, rot_uv=0.0, rot_uw=0.0, rot_vw=0.0):
        """World->local rows for an ellipsoid rotated inside this frame (degrees)."""
        U, V, W = self.U, self.V, self.W
        if rot_uv:
            a = math.radians(rot_uv)
            U, V = math.cos(a) * U + math.sin(a) * V, -math.sin(a) * U + math.cos(a) * V
        if rot_uw:
            a = math.radians(rot_uw)
            U, W = math.cos(a) * U + math.sin(a) * W, -math.sin(a) * U + math.cos(a) * W
        if rot_vw:
            a = math.radians(rot_vw)
            V, W = math.cos(a) * V + math.sin(a) * W, -math.sin(a) * V + math.cos(a) * W
        return np.stack([U, V, W])


class Builder:
    def __init__(self):
        self.groups = []
        self.cur = None

    def group(self, name, mat=None, disp=0, amp=0.0, scale=0.0, ddir=(0.0, 0.0, 0.0), band=0.01):
        for g in self.groups:
            if g['name'] == name:
                self.cur = g
                return self
        g = dict(name=name, mat=mat, disp=disp, amp=amp, scale=scale, ddir=nrm(ddir) if np.any(ddir) else np.zeros(3),
                 band=band, prims=[])
        self.groups.append(g)
        self.cur = g
        return self

    def _row(self, typ, k, op):
        row = np.zeros(NPR)
        row[0] = typ
        row[2] = k
        row[3] = op
        return row

    def ell(self, c, r, R=None, k=0.0, op=0):
        row = self._row(T_ELL, k, op)
        row[5:8] = c
        row[8:11] = r
        R = np.eye(3) if R is None else np.asarray(R)
        row[11:20] = R.reshape(-1)
        self.cur['prims'].append((row, np.r_[c, max(r) + k + 0.002]))
        return self

    def cone(self, a, b, ra, rb, k=0.0, op=0, fold=None):
        row = self._row(T_CONE, k, op)
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        row[5:8] = a
        row[8:11] = b
        row[20] = ra
        row[21] = rb
        pad = 0.0
        if fold is not None:
            e1, e2 = perp_frame(b - a)
            row[11:14] = e1
            row[14:17] = e2
            row[23] = fold.get('amp', 0.004)
            row[24] = fold.get('wl', 0.05)
            row[25] = fold.get('ph', 0.0)
            row[26] = fold.get('s0', 0.5)
            row[27] = fold.get('sw', 0.3)
            row[28] = fold.get('wob', 1.2)
            row[29] = fold.get('wph', 0.0)
            pad = abs(row[23])
        c = 0.5 * (a + b)
        self.cur['prims'].append((row, np.r_[c, 0.5 * np.linalg.norm(b - a) + max(ra, rb) + k + pad + 0.002]))
        return self

    def chain(self, pts, radii, k=0.0, op=0):
        pts = np.asarray(pts, np.float64)
        radii = np.broadcast_to(np.asarray(radii, np.float64), (len(pts),))
        for i in range(len(pts) - 1):
            self.cone(pts[i], pts[i + 1], radii[i], radii[i + 1], k, op)
        return self

    def box(self, c, half, R=None, rnd=0.0, k=0.0, op=0):
        row = self._row(T_BOX, k, op)
        row[5:8] = c
        row[8:11] = half
        row[20] = rnd
        R = np.eye(3) if R is None else np.asarray(R)
        row[11:20] = R.reshape(-1)
        self.cur['prims'].append((row, np.r_[c, np.linalg.norm(half) + k + 0.002]))
        return self

    def torus(self, c, R, Rmaj, a, b, k=0.0, op=0, arc=None, taper=0.0, ybend=0.0, tilt=0.0):
        """Elliptical-section torus around local y (R rows = local x,y,z in world). arc =
        (centre angle, half angle) radians in local x-z to keep only part of the ring; taper>0
        shrinks the section to the arc ends (power), ybend lifts the ends (m)."""
        row = self._row(T_TORUS, k, op)
        row[5:8] = c
        row[11:20] = np.asarray(R).reshape(-1)
        row[20] = a
        row[21] = b
        row[22] = Rmaj
        row[24], row[25] = (0.0, 4.0) if arc is None else arc
        row[26] = taper
        row[27] = ybend
        row[28] = tilt
        self.cur['prims'].append((row, np.r_[c, Rmaj + max(a, b) + k + 0.002]))
        return self

    def lids(self, c, R, r_in, r_out, up, lo, azmax, tilt=0.0, k=0.0, op=0, az_med=None, carve=False):
        """Eyelid shell round an eyeball centre c. R rows = (look dir, up, lateral side).
        up/lo: margin elevations (rad) at the fissure centre; azmax: fissure half-width (rad)."""
        row = self._row(T_LID, k, op)
        row[5:8] = c
        row[11:20] = np.asarray(R).reshape(-1)
        row[20] = r_in
        row[21] = r_out
        row[22] = up
        row[23] = lo
        row[24] = azmax
        row[25] = azmax if az_med is None else az_med
        row[26] = tilt
        row[27] = 1.0 if carve else 0.0
        self.cur['prims'].append((row, np.r_[c, r_out + k + 0.002]))
        return self

    def plane(self, c, R, bend_y=0.0, bend_z=0.0, k=0.0, op=2, reach=0.5):
        row = self._row(T_PLANE, k, op)
        row[5:8] = c
        row[11:20] = np.asarray(R).reshape(-1)
        row[20] = bend_y
        row[21] = bend_z
        self.cur['prims'].append((row, np.r_[c, reach]))
        return self

    def arrays(self):
        rows, bss = [], []
        G = np.zeros((len(self.groups), 9))
        for gi, g in enumerate(self.groups):
            near = 1.0 if (g['name'].startswith('hand') or g['name'] in ('steel', 'flint')) else 0.0
            G[gi] = [g['disp'], g['amp'], g['scale'], g['ddir'][0], g['ddir'][1], g['ddir'][2],
                     g['band'], g['mat'], near]
            # CSG is evaluated in insertion order inside a group (carve, then add back, ...)
            for row, bs in g['prims']:
                row = row.copy()
                row[1] = gi
                bs = bs.copy()
                bs[3] += abs(g['amp']) * 1.6
                rows.append(row)
                bss.append(bs)
        P = np.array(rows, np.float64).reshape(-1, NPR)
        BS = np.array(bss, np.float64).reshape(-1, 4)
        return P, G, BS


# ----------------------------------------------------------------- head ---

def head_frame(neck_top, fwd, up, yaw_deg=0.0, scale=0.94):
    """Head frame from the top of the neck (atlas). fwd/up: world vectors of the face
    direction and the crown (already pitched); yaw turns the face toward the camera."""
    U = nrm(fwd)
    V = nrm(up - np.dot(up, U) * U)
    W = np.cross(U, V)            # right-handed: U x V = W ; for U=-x, V=+y -> W = -z (camera side)
    if yaw_deg:
        a = math.radians(yaw_deg)
        U, W = math.cos(a) * U + math.sin(a) * W, -math.sin(a) * U + math.cos(a) * W
    # the atlas sits ~5 cm below and 1 cm behind the eye-level origin
    C = neck_top - scale * (-0.012 * U - 0.052 * V)
    return Frame(C, U, V, W, scale)


def sculpt_head(B, F, expr=None, hair=True):
    """The head in frame F (u fwd, v up (0 = eye level), w lateral; unscaled units, F.s scales).
    Ordered CSG inside the skin group: masses -> carve sockets -> add lids -> carve openings.
    expr: mouth (0..1 open), purse (0..1 'o' for blowing), squint (0..1), blink (0..1),
    brow (-1 frown .. 1 raised), gaze (u, v, w)."""
    e = dict(mouth=0.0, purse=0.0, squint=0.0, blink=0.0, brow=0.0, gaze=(1.0, -0.12, 0.0))
    if expr:
        e.update(expr)
    s = F.s
    P = F.p
    R = F.R
    S = lambda *r: np.array(r) * s
    B.group('skin', M_SKIN, band=0.004)
    # ---- masses (blend radii kept small: smooth-min inflation accumulates across overlaps)
    B.ell(P(-0.004, 0.030), S(0.090, 0.089, 0.071), R())                         # cranium
    B.ell(P(0.046, 0.036), S(0.043, 0.050, 0.058), R(-6), k=0.018 * s)          # forehead
    B.ell(P(0.077, 0.0145 + 0.002 * e['brow']), S(0.0145, 0.0090, 0.043), R(), k=0.010 * s)   # brow
    B.ell(P(0.050, -0.026), S(0.046, 0.040, 0.054), R(), k=0.014 * s)           # face mask
    for sw in (-1, 1):
        B.ell(P(0.018, -0.052, sw * 0.034), S(0.042, 0.032, 0.022), R(-18), k=0.014 * s)   # cheek / masseter fill
        B.ell(P(0.052, -0.023, sw * 0.047), S(0.020, 0.011, 0.011), R(-10, sw * 12.0), k=0.012 * s)  # cheekbones
    B.ell(P(0.066, -0.060), S(0.031, 0.027, 0.030), R(6), k=0.014 * s)          # muzzle
    for sw in (-1, 1):
        B.cone(P(0.068, -0.095, sw * 0.013), P(0.004, -0.074, sw * 0.044), 0.0100 * s, 0.0115 * s, k=0.012 * s)  # jaw
    B.ell(P(0.0815, -0.0965), S(0.0145, 0.0150, 0.0165), R(12), k=0.010 * s)    # chin
    B.cone(P(0.085, 0.001), P(0.1018, -0.0295), 0.0056 * s, 0.0068 * s, k=0.006 * s)   # nose bridge
    B.ell(P(0.1076, -0.0370), S(0.0088, 0.0080, 0.0080), R(-10), k=0.005 * s)   # tip
    B.cone(P(0.1042, -0.0425), P(0.0960, -0.0450), 0.0032 * s, 0.0030 * s, k=0.003 * s)   # columella
    for sw in (-1, 1):
        B.ell(P(0.0970, -0.0415, sw * 0.0118), S(0.0072, 0.0058, 0.0062), R(10), k=0.008 * s)  # alae
    op = e['mouth']
    pu = e['purse']
    RL = np.stack([F.U, F.V, F.W])
    # lips follow the dental arch (tapered torus arcs, section tilted)
    B.torus(P(0.0640 + 0.004 * pu, -0.0604 + 0.001 * op), RL, 0.030 * s * (1 - 0.3 * pu), 0.0088 * s, 0.0066 * s,
            k=0.004 * s, arc=(0.0, 0.72 * (1 - 0.35 * pu)), taper=0.7, ybend=-0.002 * s, tilt=-0.35)
    B.torus(P(0.0620 + 0.004 * pu, -0.0745 - 0.004 * op), RL, 0.029 * s * (1 - 0.3 * pu), 0.0092 * s, 0.0080 * s,
            k=0.004 * s, arc=(0.0, 0.66 * (1 - 0.35 * pu)), taper=0.7, ybend=0.003 * s, tilt=0.35)
    for sw in (-1, 1):
        B.ell(P(-0.012, -0.014, sw * 0.063), S(0.013, 0.024, 0.005), R(-14), k=0.004 * s)   # ears
    # ---- soft hollows: temples, under the cheekbones
    for sw in (-1, 1):
        B.ell(P(0.042, 0.022, sw * 0.070), S(0.022, 0.020, 0.008), R(), k=0.020 * s, op=1)
    # ---- eye: clear a spherical socket round the eyeball, lay the lid shell back in
    g = nrm(e['gaze'][0] * F.U + e['gaze'][1] * F.V + e['gaze'][2] * F.W)
    gp = math.asin(max(-1.0, min(1.0, float(np.dot(g, F.V)))))          # gaze pitch (rad)
    for sw in (-1, 1):
        B.ell(P(0.0665, -0.001, sw * 0.031), S(0.0140, 0.0140, 0.0142), R(), k=0.005 * s, op=1)
        B.ell(P(0.0905, -0.0040, sw * 0.0315), S(0.0095, 0.0105, 0.0140), R(-5), k=0.006 * s, op=1)
    for sw in (-1, 1):
        # lids look where the eye looks (the upper lid follows the gaze down), opened by expr
        a_ = 0.65 * gp
        Lf = math.cos(a_) * F.U + math.sin(a_) * F.V
        Lu = -math.sin(a_) * F.U + math.cos(a_) * F.V
        Ls = sw * F.W
        up0 = 0.36 * (1 - e['blink']) - 0.30 * e['blink'] - 0.12 * e['squint']
        lo0 = -0.34 * (1 - e['blink']) - 0.30 * e['blink'] + 0.10 * e['squint']
        B.lids(P(0.0665, -0.001, sw * 0.031), np.stack([Lf, Lu, Ls]), 0.0112 * s, 0.0135 * s, up0, lo0, 1.40,
               tilt=0.05, k=0.0050 * s, az_med=0.95)
        # re-open the fissure through anything the blends laid over it
        B.lids(P(0.0665, -0.001, sw * 0.031), np.stack([Lf, Lu, Ls]), 0.0110 * s, 0.0175 * s, up0, lo0, 1.40,
               tilt=0.05, k=0.0012 * s, az_med=0.95, carve=True, op=1)
    # mouth line along the arch (a shallow notch in profile)
    B.torus(P(0.0645 + 0.004 * pu, -0.0672 - 0.0015 * op), RL, 0.036 * s * (1 - 0.3 * pu), 0.0042 * s,
            (0.0008 + 0.0035 * op + 0.002 * pu) * s, k=0.0015 * s, op=1, arc=(0.0, 0.66 * (1 - 0.4 * pu)), taper=0.3)
    B.ell(P(0.0955, -0.0855), S(0.0060, 0.0035, 0.0165), R(), k=0.007 * s, op=1)
    # ---- eyes (own group: crisp edge against the lids)
    B.group('eyes', M_EYE, band=0.001)
    eyes = []
    for sw in (-1, 1):
        c = P(0.0665, -0.001, sw * 0.031)
        B.ell(c, S(0.0122, 0.0122, 0.0122), R())
        B.ell(c + g * 0.0062 * s, S(0.0078, 0.0078, 0.0078), R(), k=0.0015 * s)     # cornea bulge
        eyes.append(c)
    if hair:
        # a knitted wool cap pulled down to the brows, a folded brim; her long hair escapes below it
        n_ = nrm(0.3476 * F.U - F.V)                 # brim plane: forehead (v=.030) to nape (v=-.035)
        t_ = nrm(F.U + 0.3476 * F.V)
        P0 = P(0.092, 0.030)
        B.group('cap', M_KNIT, disp=2, amp=0.0006, scale=260.0, ddir=F.U, band=0.003)
        B.ell(P(-0.006, 0.036), S(0.100, 0.099, 0.081), R())
        B.ell(P(-0.034, 0.072), S(0.074, 0.062, 0.068), R(15), k=0.03 * s)            # soft slouch at the back
        B.plane(P0, np.stack([n_, t_, F.W]), bend_z=-2.3 / s, k=0.004 * s, op=2, reach=0.5)
        B.group('cap_brim', M_KNIT, disp=2, amp=0.0007, scale=230.0, ddir=F.U, band=0.003)
        B.ell(P(-0.006, 0.036), S(0.107, 0.106, 0.088), R())
        B.plane(P0 + n_ * 0.003 * s, np.stack([n_, t_, F.W]), bend_z=-2.3 / s, k=0.004 * s, op=2, reach=0.5)
        B.plane(P0 - n_ * 0.024 * s, np.stack([-n_, -t_, F.W]), bend_z=2.3 / s, k=0.004 * s, op=2, reach=0.5)
        B.group('hair', M_HAIR, disp=0, ddir=-F.U - 0.35 * F.V, band=0.003)
        for sw in (-1, 1):
            B.ell(P(-0.020, -0.034, sw * 0.057), S(0.034, 0.042, 0.020), R(-28), k=0.02 * s)   # over the ears
        B.ell(P(-0.082, -0.038), S(0.042, 0.046, 0.056), R(-30), k=0.03 * s)      # gathered at the nape
    anchors = dict(
        eyes=eyes, gaze=g,
        mouth=P(0.103, -0.0672), nose_tip=P(0.120, -0.039), chin=P(0.095, -0.100),
        brow=P(0.094, 0.014), crown=P(0.0, 0.119), back=P(-0.094, 0.03), nape=P(-0.060, -0.045),
        ear=P(-0.010, -0.012, 0.064), hair_root=P(-0.070, 0.020),
    )
    return anchors


def head_params(F, expr=None):
    """Shading-time head parameters (lip / brow / flush regions, eye centres + gaze)."""
    e = dict(mouth=0.0, purse=0.0, gaze=(1.0, -0.12, 0.0))
    if expr:
        e.update(expr)
    s = F.s
    H = np.zeros(160)
    H[0] = 1.0
    H[1:4] = F.C
    H[4:7] = F.U / s
    H[7:10] = F.V / s
    H[10:13] = F.W / s
    H[13] = -0.0675 - 0.001 * e['mouth']      # lip band centre (v)
    H[14] = 0.0125 + 0.003 * e['mouth']        # half height
    H[15] = 0.021 - 0.005 * e['purse']         # lateral extent
    H[16] = 0.086                              # lips only in front of this u
    H[17:20] = [0.30, 0.105, 0.090]            # lip albedo
    H[20] = 0.0215                             # brow line v
    H[21] = 0.030                              # brow centre |w|
    H[22] = 0.070                              # brows in front of u
    H[23:26] = [0.090, -0.030, 0.040]          # cheek flush centre (u, v, |w|)
    for q, sw in enumerate((-1, 1)):
        H[26 + 3 * q:29 + 3 * q] = F.p(0.0665, -0.001, sw * 0.031)
    g = nrm(e['gaze'][0] * F.U + e['gaze'][1] * F.V + e['gaze'][2] * F.W)
    H[32:35] = g
    return H


# ------------------------------------------------------------- skeleton ---

def ik3(S, T, L1, L2, pole):
    """Two-bone IK in 3-D: returns (elbow/knee, wrist/ankle) reaching for T, bending toward pole."""
    S = np.asarray(S, np.float64)
    T = np.asarray(T, np.float64)
    d = T - S
    dist = float(np.linalg.norm(d))
    dc = min(max(dist, abs(L1 - L2) + 1e-4), L1 + L2 - 1e-4)
    dn = d / max(dist, 1e-9)
    pole = np.asarray(pole, np.float64)
    pp = pole - np.dot(pole, dn) * dn
    pp = nrm(pp) if np.linalg.norm(pp) > 1e-6 else perp_frame(dn)[0]
    ca = (L1 * L1 + dc * dc - L2 * L2) / (2 * L1 * dc)
    a = math.acos(max(-1.0, min(1.0, ca)))
    E = S + L1 * (math.cos(a) * dn + math.sin(a) * pp)
    return E, S + dc * dn


def rot_about(v, axis, deg):
    """Rodrigues rotation of v about a unit axis."""
    a = math.radians(deg)
    k = nrm(axis)
    v = np.asarray(v, np.float64)
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * np.dot(k, v) * (1 - math.cos(a))


LEN = dict(lumbar=0.19, thorax=0.27, neck=0.10, uarm=0.285, farm=0.235, thigh=0.43, shin=0.42,
           sh_w=0.160, hip_w=0.085)


def skeleton(pose):
    """World joints from a pose dict (see POSE_KEYS in beacon.py)."""
    J = {}
    yaw = pose.get('yaw', 0.0)
    Up = rot_about(np.array([-1.0, 0.0, 0.0]), [0, 1, 0], -yaw)    # +yaw turns toward camera (-z)
    Vp = np.array([0.0, 1.0, 0.0])
    Wp = np.cross(Up, Vp)
    Pc = np.asarray(pose['pelvis'], np.float64)
    a1 = math.radians(pose.get('lean', 0.0))
    a2 = a1 + math.radians(pose.get('chest', 0.0))
    d1 = math.sin(a1) * Up + math.cos(a1) * Vp
    d2 = math.sin(a2) * Up + math.cos(a2) * Vp
    L1 = Pc + LEN['lumbar'] * d1
    C7 = L1 + LEN['thorax'] * d2
    # thorax frame (+ twist about the spine)
    Vt = d2
    Ut = nrm(math.cos(a2) * Up - math.sin(a2) * Vp)
    Ut = rot_about(Ut, Vt, -pose.get('twist', 0.0))
    Wt = np.cross(Ut, Vt)
    an = math.radians(pose.get('neck', 0.0))
    dn = math.sin(an) * Ut + math.cos(an) * Vt
    dn = nrm(dn - 0.0 * Wt)
    atlas = C7 + LEN['neck'] * dn
    # head orientation: pitch (chin down +) relative to the torso's sagittal plane, then yaw/roll
    ah = math.radians(pose.get('head', 0.0))
    hf = math.cos(ah) * Ut - math.sin(ah) * Vt
    hu = math.sin(ah) * Ut + math.cos(ah) * Vt
    hy = pose.get('head_yaw', 0.0)
    if hy:
        hf = rot_about(hf, hu, -hy)
    hr = pose.get('head_roll', 0.0)
    if hr:
        hu = rot_about(hu, hf, hr)
    J.update(Pc=Pc, L1=L1, C7=C7, atlas=atlas, Up=Up, Vp=Vp, Wp=Wp, Ut=Ut, Vt=Vt, Wt=Wt, dn=dn, d1=d1,
             d2=d2, hf=hf, hu=hu)
    # shoulders (near = +Wt = toward camera), a shrug lifts them
    shrug = pose.get('shrug', 0.0)
    for side, sg in (('n', 1.0), ('f', -1.0)):
        S = C7 - (0.055 - 0.035 * shrug) * Vt + sg * LEN['sh_w'] * Wt - 0.020 * Ut
        T = np.asarray(pose['hand_' + side], np.float64)
        E, W = ik3(S, T, LEN['uarm'], LEN['farm'], pose.get('elbow_' + side, -Vt + 0.3 * sg * Wt))
        J['S' + side], J['E' + side], J['W' + side] = S, E, W
        H = Pc - 0.02 * Vp + sg * LEN['hip_w'] * Wp
        K, A = ik3(H, pose['foot_' + side], LEN['thigh'], LEN['shin'], pose.get('knee_' + side, Up))
        J['H' + side], J['K' + side], J['A' + side] = H, K, A
    return J


# ----------------------------------------------------------------- hands ---

FINGERS = [  # (lateral offset from the middle, knuckle distance from wrist, lengths, base radius)
    (0.0230, 0.089, (0.042, 0.025, 0.020), 0.0074),    # index
    (0.0076, 0.093, (0.046, 0.028, 0.021), 0.0077),    # middle
    (-0.0078, 0.090, (0.043, 0.027, 0.020), 0.0072),   # ring
    (-0.0225, 0.083, (0.035, 0.021, 0.019), 0.0064),   # little
]


def hand(B, gname, W, fdir, palm, thumb_side, curls, thumb=0.3, spread=0.0, thumb_out=0.0, nails=True):
    """A slender bare hand from the wrist W. fdir: along the metacarpals; palm: palmar normal;
    thumb_side: +1 right hand / -1 left hand (sd = n x a * side points to the thumb).
    curls: 4 flexions 0 (straight) .. 1 (fist) for index..little. thumb: 0 open .. 1 folded
    across the palm; thumb_out: abduction away from the palm. Returns anchor points."""
    a = nrm(fdir)
    n = nrm(np.asarray(palm, np.float64) - np.dot(palm, a) * a)
    sd = np.cross(n, a) * thumb_side
    Rh = np.stack([a, sd, n])
    B.group(gname, M_SKIN, band=0.002)
    # wrist, palm (tapering to the wrist), knuckle heads, thenar and hypothenar pads
    B.ell(W + a * 0.004, np.array([0.020, 0.027, 0.016]), Rh)
    B.box(W + a * 0.056 - n * 0.001, np.array([0.036, 0.034, 0.0105]), Rh, rnd=0.0095, k=0.012)
    B.box(W + a * 0.030 - n * 0.001, np.array([0.022, 0.028, 0.0115]), Rh, rnd=0.0100, k=0.012)
    B.ell(W + a * 0.034 + sd * 0.019 + n * 0.008, np.array([0.024, 0.013, 0.010]), Rh, k=0.010)     # thenar
    B.ell(W + a * 0.046 - sd * 0.024 + n * 0.006, np.array([0.030, 0.010, 0.008]), Rh, k=0.010)     # hypothenar
    tips, mids, segs, knuck = [], [], [], []
    for q, (off, kd, Ls, r0) in enumerate(FINGERS):
        cq = curls[q]
        spr = (q - 1.5) * spread * 7.0
        base = W + a * kd + sd * off - n * 0.0015
        B.ell(base - n * 0.0035, np.array([0.0085, 0.0088, 0.0080]), Rh, k=0.006)                  # knuckle head
        d = rot_about(a, n, spr * thumb_side)
        pts = [base]
        ang = 0.0
        dirs = []
        for j, L in enumerate(Ls):
            ang += (70.0, 95.0, 60.0)[j] * cq
            dj = math.cos(math.radians(ang)) * d + math.sin(math.radians(ang)) * n
            dirs.append((dj, ang))
            pts.append(pts[-1] + dj * L)
        rr = [r0, r0 * 0.86, r0 * 0.78, r0 * 0.62]
        for j in range(3):
            B.cone(pts[j], pts[j + 1], rr[j], rr[j + 1], k=0.0022)
        tips.append(pts[-1])
        mids.append(pts[2])
        segs.append((pts, dirs, rr, d))
        knuck.append(pts[1])
        knuck.append(base)
    # thumb: carpometacarpal base low on the radial side, opposing the fingers
    tb = W + a * 0.010 + sd * 0.019 + n * 0.008
    t_rest = nrm(0.55 * a + 0.70 * sd + 0.35 * n + 0.35 * thumb_out * sd)
    t_fold = nrm(0.80 * a - 0.10 * sd + 0.60 * n)
    td = nrm(t_rest * (1 - thumb) + t_fold * thumb)
    p1 = tb + td * 0.044
    td2 = nrm(td + n * (0.15 + 0.55 * thumb) - sd * 0.25 * thumb + a * 0.1)
    p2 = p1 + td2 * 0.031
    td3 = nrm(td2 + n * (0.15 + 0.45 * thumb) - sd * 0.1 * thumb)
    p3 = p2 + td3 * 0.026
    B.cone(tb, p1, 0.0135, 0.0100, k=0.008)
    B.cone(p1, p2, 0.0100, 0.0092, k=0.0025)
    B.cone(p2, p3, 0.0092, 0.0075, k=0.002)
    if nails:
        B.group(gname + '_nails', M_NAIL, band=0.001)
        for (pts, dirs, rr, d) in segs:
            dj, ang = dirs[2]
            dors = math.sin(math.radians(ang)) * d - math.cos(math.radians(ang)) * n
            side = nrm(np.cross(dj, dors))
            c = pts[2] + dj * 0.62 * np.linalg.norm(pts[3] - pts[2]) + dors * rr[3] * 0.80
            B.ell(c, np.array([0.0058, 0.0014, 0.0050]), np.stack([dj, dors, side]))
        dors_t = nrm(np.cross(td3, sd) * thumb_side - n * 0.3)
        dors_t = nrm(-(n - np.dot(n, td3) * td3))
        side = nrm(np.cross(td3, dors_t))
        B.ell(p2 + td3 * 0.017 + dors_t * 0.0062, np.array([0.0065, 0.0015, 0.0060]), np.stack([td3, dors_t, side]))
    return dict(tips=tips, mids=mids, thumb=p3, thumb_mid=p2, palm=W + a * 0.056, a=a, n=n, sd=sd, W=W,
                knuckles=knuck)


# ------------------------------------------------------------ the figure ---

def build_figure(pose, t, scarf_pts=None, hair_pts=None, detail=1.0):
    """Returns (Builder, head Frame, H params, anchors) for one frame."""
    J = skeleton(pose)
    B = Builder()
    Up, Vp, Wp, Ut, Vt, Wt = J['Up'], J['Vp'], J['Wp'], J['Ut'], J['Vt'], J['Wt']
    Pc, L1, C7, atlas = J['Pc'], J['L1'], J['C7'], J['atlas']
    br = pose.get('breath', 0.0)
    # ---- head
    F = head_frame(atlas, J['hf'], J['hu'])
    expr = dict(pose.get('expr', {}))
    if pose.get('look_at') is not None:
        g = nrm(np.asarray(pose['look_at'], np.float64) - F.p(0.0665, -0.001, 0.0))
        gu, gv, gw = float(g @ F.U), float(g @ F.V), float(g @ F.W)
        # eyes turn at most ~38 deg from the face direction
        ang = math.acos(max(-1.0, min(1.0, gu)))
        lim = math.radians(38.0)
        if ang > lim:
            q = math.hypot(gv, gw) + 1e-9
            gu, gv, gw = math.cos(lim), math.sin(lim) * gv / q, math.sin(lim) * gw / q
        expr['gaze'] = (gu, gv, gw)
    an = sculpt_head(B, F, expr, hair=True)
    # ---- neck + throat (skin)
    B.group('skin', M_SKIN)
    nb = C7 + 0.012 * Ut
    B.cone(nb, F.p(-0.012, -0.050), 0.050, 0.046, k=0.02)
    B.cone(F.p(0.030, -0.100), nb + 0.045 * Ut + 0.02 * Vt, 0.020, 0.030, k=0.02)           # throat
    # ---- coat torso
    B.group('coat', M_COAT, band=0.006)
    Rt = np.stack([Ut, Vt, Wt])
    Rl = np.stack([nrm(np.cross(J['d1'], Wp)), J['d1'], Wp])
    chest = L1 + 0.60 * (C7 - L1) + (0.012 + 0.004 * br) * Ut
    B.ell(chest, np.array([0.108 + 0.003 * br, 0.150, 0.158]), Rt)
    B.ell(Pc + 0.55 * (L1 - Pc) + 0.004 * Ut, np.array([0.098, 0.125, 0.142]), Rl, k=0.05)
    B.ell(Pc - 0.012 * Vp - 0.008 * Up, np.array([0.108, 0.110, 0.168]), np.stack([Up, Vp, Wp]), k=0.05)
    B.ell(C7 - 0.100 * Vt - 0.045 * Ut, np.array([0.066, 0.100, 0.140]), Rt, k=0.04)          # upper back
    for sd_ in ('n', 'f'):
        B.cone(C7 - 0.030 * Vt, J['S' + sd_] - 0.004 * Vt, 0.050, 0.056, k=0.035)          # shoulder line
    # collar standing round the neck, and the lowered hood bunched behind it
    RN = np.stack([Ut, J['dn'], np.cross(Ut, J['dn'])])
    B.torus(C7 + 0.028 * J['dn'] - 0.004 * Ut, np.stack([Ut, J['dn'], np.cross(Ut, J['dn'])]), 0.066, 0.013, 0.030,
            k=0.015)
    B.ell(C7 - 0.072 * Ut - 0.030 * Vt, np.array([0.050, 0.060, 0.100]), Rt, k=0.03)          # lowered hood
    # ---- skirt: the coat follows the thighs and hangs under gravity
    hem = pose.get('hem', 0.0)
    for sd_, sg in (('n', 1.0), ('f', -1.0)):
        H, K = J['H' + sd_], J['K' + sd_]
        B.cone(H + sg * 0.02 * Wp + 0.01 * Vp, K + 0.035 * Vp + 0.015 * sg * Wp, 0.112, 0.090, k=0.05,
               fold=dict(amp=0.006, wl=0.09, ph=1.3 * sg, s0=0.55, sw=0.45, wob=2.0, wph=0.7))
    gy = np.array([hem * 0.35, -1.0, 0.0])
    gy = nrm(gy)
    seat = Pc - 0.09 * Up - 0.03 * Vp
    drop = max(0.12, min(0.62, seat[1] - 0.02))
    B.cone(seat, seat + gy * drop * 0.92 - 0.03 * Up, 0.120, 0.150, k=0.05,
           fold=dict(amp=0.010, wl=0.10, ph=0.4, s0=0.75, sw=0.5, wob=2.5, wph=1.9))
    front = Pc + 0.085 * Up - 0.04 * Vp
    fdrop = max(0.10, min(0.55, front[1] - 0.02))
    B.cone(front, front + gy * fdrop * 0.85 + 0.02 * Up, 0.100, 0.125, k=0.05,
           fold=dict(amp=0.008, wl=0.08, ph=2.1, s0=0.7, sw=0.5, wob=2.0, wph=0.3))
    # ---- sleeves (with elbow folds): own group so hand-held light sources ignore them for shadows
    B.group('hand_sleeves', M_COAT, band=0.006)
    for sd_, sg in (('n', 1.0), ('f', -1.0)):
        S, E, W = J['S' + sd_], J['E' + sd_], J['W' + sd_]
        bend = 1.0 - max(0.0, float(np.dot(nrm(E - S), nrm(W - E))))
        B.cone(S, E, 0.058, 0.050, k=0.03,
               fold=dict(amp=0.004 + 0.006 * bend, wl=0.055, ph=0.5 * sg, s0=0.95, sw=0.35, wob=1.6, wph=0.2))
        B.cone(E, W - nrm(W - E) * 0.005, 0.050, 0.046, k=0.02,
               fold=dict(amp=0.004 + 0.005 * bend, wl=0.05, ph=1.7 * sg, s0=0.1, sw=0.45, wob=1.8, wph=2.2))
        # cuff: a slightly flared, turned-back edge at the wrist
        cd = nrm(W - E)
        cf = perp_frame(cd)
        B.torus(W - cd * 0.012, np.stack([cf[0], cd, cf[1]]), 0.040, 0.009, 0.020, k=0.012)
    # ---- hands
    anchors = dict(an)
    for sd_, sg in (('n', 1.0), ('f', -1.0)):
        W = J['W' + sd_]
        fdir = np.asarray(pose.get('fdir_' + sd_, nrm(W - J['E' + sd_])), np.float64)
        palm = np.asarray(pose.get('palm_' + sd_, -Vt), np.float64)
        curls = pose.get('curl_' + sd_, (0.5, 0.5, 0.5, 0.5))
        ha = hand(B, 'hand_' + sd_, W + nrm(fdir) * 0.004, fdir, palm, -sg, curls,
                  thumb=pose.get('thumb_' + sd_, 0.4), spread=pose.get('spread_' + sd_, 0.0),
                  thumb_out=pose.get('thumbout_' + sd_, 0.0))
        anchors['hand_' + sd_] = ha
    # ---- tools: flint pinched in the near (left) hand, C-steel round the far (right) fingers
    flint_pt = None
    steel_pt = None
    tools = pose.get('tools', 'both')
    if tools in ('both', 'flint'):
        hn = anchors['hand_n']
        B.group('flint', M_FLINT, band=0.003)
        # a stone held in the curled fingers, its sharp top edge standing proud of thumb and index
        fc = hn['W'] + hn['a'] * 0.082 + hn['n'] * 0.021 + hn['sd'] * 0.014
        fu = hn['sd']
        fa = nrm(hn['a'] - np.dot(hn['a'], fu) * fu)
        Rf = np.stack([fa, fu, np.cross(fa, fu)])
        B.box(fc, np.array([0.017, 0.021, 0.012]), Rf, rnd=0.004)
        Rf2 = np.stack([nrm(rot_about(fa, Rf[2], 24)), nrm(rot_about(fu, Rf[2], 24)), Rf[2]])
        B.box(fc + fu * 0.012 + fa * 0.004, np.array([0.014, 0.012, 0.009]), Rf2, rnd=0.0015, k=0.003)
        flint_pt = fc + fu * 0.026 + fa * 0.014
    if tools in ('both', 'steel'):
        hf = anchors['hand_f']
        B.group('steel', M_STEEL, band=0.001)
        loop_c = (hf['mids'][0] + hf['mids'][1] + hf['mids'][2]) / 3.0
        front = nrm(loop_c - hf['palm'])
        ax = nrm(hf['sd'] - np.dot(hf['sd'], front) * front)
        z_ = np.cross(front, ax)
        B.torus(loop_c - front * 0.002, np.stack([front, ax, z_]), 0.019, 0.0030, 0.0080, arc=(0.0, 2.3))
        bar_c = loop_c + front * 0.0185
        B.box(bar_c, np.array([0.0030, 0.036, 0.0080]), np.stack([front, ax, z_]), rnd=0.0014)
        steel_pt = bar_c + front * 0.003
    anchors['flint'] = flint_pt
    anchors['steel'] = steel_pt
    # ---- legs: trousers + boots
    B.group('legs', M_CLOTH, band=0.004)
    for sd_ in ('n', 'f'):
        H, K, A = J['H' + sd_], J['K' + sd_], J['A' + sd_]
        B.cone(H, K, 0.070, 0.054, k=0.02)
        B.cone(K, A + nrm(K - A) * 0.10, 0.052, 0.044, k=0.02)
    B.group('boots', M_BOOT, band=0.003)
    for sd_ in ('n', 'f'):
        K, A = J['K' + sd_], J['A' + sd_]
        B.cone(A + nrm(K - A) * 0.15, A, 0.050, 0.047, k=0.01)
        td = nrm(np.asarray(pose.get('toe_' + sd_, Up), np.float64))
        sole_up = nrm(np.asarray(pose.get('sole_' + sd_, Vp), np.float64))
        side = nrm(np.cross(td, sole_up))
        fc = A + td * 0.075 - sole_up * 0.045
        B.box(fc, np.array([0.130, 0.040, 0.050]), np.stack([td, sole_up, side]), rnd=0.030, k=0.02)
    # ---- the rock she kneels on
    rock = pose.get('rock')
    if rock is not None:
        B.group('rock', M_FLINT, disp=1, amp=0.004, scale=16.0, band=0.02)
        B.box(np.asarray(rock[0]), np.asarray(rock[1]), np.stack([nrm([1, 0.08, 0.1]), nrm([-0.08, 1, 0]), nrm([0.1, 0, -1])]),
              rnd=0.05)
    # ---- scarf: two loops round the neck + a short front end + the long tail
    B.group('scarf', M_SCARF, band=0.006)
    dn = J['dn']
    for (hgt, R_, a_, b_, tilt_u, tilt_w) in ((0.006, 0.070, 0.027, 0.022, -16.0, 6.0), (0.040, 0.061, 0.023, 0.019, 8.0, -4.0)):
        ax = rot_about(rot_about(dn, Wt, tilt_u), Ut, tilt_w)
        x0 = nrm(Ut - np.dot(Ut, ax) * ax)
        B.torus(C7 + hgt * dn + 0.004 * Ut, np.stack([x0, ax, np.cross(x0, ax)]), R_, a_, b_, k=0.016)
    fe0 = C7 + 0.004 * dn + 0.072 * Ut + 0.03 * Wt
    B.cone(fe0, fe0 - 0.16 * Vt + 0.03 * Ut + hem * 0.02 * Up, 0.030, 0.034, k=0.02)
    sa = C7 + 0.035 * dn - 0.070 * Ut
    anchors['scarf_anchor'] = sa
    if scarf_pts is not None:
        scarf_tail(B, scarf_pts, t, z0=sa[2] - 0.02)
    # ---- long hair: thick locks along the simulated chains
    if hair_pts is not None:
        B.group('hair')
        for q, hp in enumerate(hair_pts):
            hp = np.asarray(hp, np.float64)
            m = len(hp)
            zq = F.C[2] + (-0.035 + 0.070 * ((q * 0.618) % 1.0))
            P3 = np.c_[hp, np.full(m, zq)]
            P3[:, 2] += np.linspace(0, 1, m) * 0.03 * math.sin(q * 2.3 + t * 1.7)
            r0 = 0.009 if q % 3 == 0 else 0.006
            rr = r0 * (1 - np.linspace(0, 1, m)) ** 1.2 + 0.0015
            B.chain(P3, rr, k=0.010)
    anchors.update(J=J, head=F, expr=expr, mouth=an['mouth'], hair_root=F.p(-0.088, -0.030))
    H = head_params(F, expr)
    # cold hands: knuckles and fingertips flush red (shading-time tint round these points)
    pts = []
    for sd_ in ('n', 'f'):
        ha = anchors.get('hand_' + sd_)
        if ha is not None:
            pts += list(ha['knuckles']) + list(ha['tips']) + [ha['thumb'], ha['thumb_mid']]
    pts = pts[:38]
    H[40] = len(pts)
    for i, q in enumerate(pts):
        H[41 + 3 * i:44 + 3 * i] = q
    return B, F, H, anchors


def scarf_tail(B, pts, t, z0=0.0, width=0.11, thick=0.008):
    """Long wool tail from the verlet chain: twisting, rippling round-box segments + fringe."""
    from core import fnoise1
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    P3 = np.c_[pts, np.full(n, z0)]
    zhat = np.array([0.0, 0.0, 1.0])
    for i in range(n - 1):
        a, b = P3[i], P3[i + 1]
        s = (i + 0.5) / (n - 1)
        tg = nrm(b - a)
        n2 = nrm(np.cross(zhat, tg))
        tw = 1.35 * fnoise1(s * 3.1 - t * 1.9, 3.0, 2) + 0.30
        wdir = nrm(math.cos(tw) * n2 + math.sin(tw) * zhat)
        ndir = np.cross(tg, wdir)
        L = np.linalg.norm(b - a)
        w = width * (0.80 + 0.20 * (1 - s)) * (0.55 + 0.45 * min(1.0, 3.0 * s + 0.3))
        rip = (0.018 * math.sin(s * 13.0 - t * 13.0) + 0.008 * math.sin(s * 29.0 - t * 23.0)) * (0.3 + s)
        c = 0.5 * (a + b) + n2 * rip + ndir * rip * 0.4
        B.box(c, np.array([0.5 * L + 0.012, 0.5 * w, thick]), np.stack([tg, wdir, ndir]), rnd=thick * 0.95, k=0.012)
    # fringe: twisted yarn tassels off the end
    a, b = P3[-2], P3[-1]
    tg = nrm(b - a)
    n2 = nrm(np.cross(zhat, tg))
    tw = 1.1 * fnoise1(1.0 * 2.6 - t * 1.4, 3.0, 2) + 0.35
    wdir = nrm(math.cos(tw) * n2 + math.sin(tw) * zhat)
    for k in range(9):
        off = (k - 4) / 4.0 * width * 0.42
        root = b + wdir * off
        L = 0.065 + 0.012 * ((k * 7) % 3)
        wig = 0.012 * fnoise1(t * 3.1 + k * 1.37, 7.0, 2)
        mid = root + tg * L * 0.5 + wdir * off * 0.1 + n2 * wig
        tip = root + tg * L + wdir * off * 0.15 + n2 * wig * 2.2
        B.cone(root, mid, 0.0032, 0.0026, k=0.004)
        B.cone(mid, tip, 0.0026, 0.0012, k=0.002)


# ------------------------------------------------------------- render api ---

def render(cam, B, H, lights, env, ss=3, M=None, pad=6):
    """Sphere-trace the figure. Returns (Y0, X0, rgb_premul float32, alpha float32, depth float32) for
    the screen region it covers, or None. lights: (n,8) [x,y,z, r,g,b (W at 1 m), radius, soft_k]."""
    P, G, BS = B.arrays()
    if M is None:
        M = material_table()
    sx, sy, z = cam.project(BS[:, :3])
    ok = z > 0.05
    if not np.any(ok):
        return None
    r = cam.f * BS[:, 3] / np.maximum(z - BS[:, 3], 0.05)
    X0 = int(max(0, np.floor((sx[ok] - r[ok]).min()) - pad))
    X1 = int(min(cam.W, np.ceil((sx[ok] + r[ok]).max()) + pad))
    Y0 = int(max(0, np.floor((sy[ok] - r[ok]).min()) - pad))
    Y1 = int(min(cam.H, np.ceil((sy[ok] + r[ok]).max()) + pad))
    if X1 <= X0 or Y1 <= Y0:
        return None
    w, h = X1 - X0, Y1 - Y0
    TS = 16
    off, idx = hs.build_tiles(cam, BS, X0, Y0, w, h, TS, P)
    allidx = np.arange(len(P), dtype=np.int64)
    rgb = np.zeros((h, w, 3))
    a = np.zeros((h, w))
    d = np.full((h, w), 1e9)
    L = np.asarray(lights, np.float64).reshape(-1, 8)
    hs.render_region(cam.params(), P, G, BS, M, L, len(L), np.asarray(env, np.float64), H, off, idx, allidx,
                     X0, Y0, w, h, TS, ss, rgb, a, d)
    return Y0, X0, rgb.astype(np.float32), a.astype(np.float32), d.astype(np.float32)


def env_vec(rim_dir=(0.55, 0.45, 0.70), rim=(0.10, 0.14, 0.22), amb=(0.004, 0.006, 0.012),
            bounce=(0.0, 0.0, 0.0), ao=0.02, shadows=1.0):
    e = np.zeros(16)
    e[0:3] = nrm(rim_dir)
    e[3:6] = rim
    e[6:9] = amb
    e[9:12] = bounce
    e[12] = ao
    e[13] = shadows
    return e


def hand_frame(pose, J, side):
    """Wrist + (a, n, sd) exactly as build_figure/hand() compute them (for cheap anchors)."""
    sg = 1.0 if side == 'n' else -1.0
    W = J['W' + side]
    fdir = np.asarray(pose.get('fdir_' + side, nrm(W - J['E' + side])), np.float64)
    palm = np.asarray(pose.get('palm_' + side, -J['Vt']), np.float64)
    a = nrm(fdir)
    n = nrm(palm - np.dot(palm, a) * a)
    sd = np.cross(n, a) * (-sg)
    return W + a * 0.004, a, n, sd


def cheap_anchors(pose):
    """Scarf anchor, hair root, head frame, mouth and flint point without building primitives."""
    J = skeleton(pose)
    F = head_frame(J['atlas'], J['hf'], J['hu'])
    Wn, a, n, sd = hand_frame(pose, J, 'n')
    fc = Wn + a * 0.082 + n * 0.021 + sd * 0.014
    fa = nrm(a - np.dot(a, sd) * sd)
    flint = fc + sd * 0.026 + fa * 0.014
    sa = J['C7'] + 0.035 * J['dn'] - 0.070 * J['Ut']
    return dict(J=J, head=F, scarf_anchor=sa, hair_root=F.p(-0.088, -0.030), mouth=F.p(0.103, -0.0672),
                flint=flint, nose=F.p(0.120, -0.039))


# ------------------------------------------------------------ fine hair ---

def _spline(P, m):
    """Catmull-Rom resample of a polyline (n,3) to m points."""
    P = np.asarray(P, np.float64)
    n = len(P)
    ts = np.linspace(0, n - 1, m)
    out = np.zeros((m, P.shape[1]))
    for k, t in enumerate(ts):
        i = min(int(t), n - 2)
        u = t - i
        p0 = P[max(i - 1, 0)]
        p1 = P[i]
        p2 = P[i + 1]
        p3 = P[min(i + 2, n - 1)]
        out[k] = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u +
                        (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
    return out


def strand_light(pts, T, lights, env, head_c, head_r=0.10, alb=(0.022, 0.014, 0.010)):
    """Radiance of hair strands at pts (n,3) with tangents T: Kajiya-Kay diffuse + spec from each
    light (softly shadowed by the head sphere) + the cold backlight rim."""
    alb = np.asarray(alb)
    col = np.zeros((len(pts), 3))
    for L in lights:
        lp = L[:3]
        d = lp - pts
        dist = np.linalg.norm(d, axis=1) + 1e-9
        l = d / dist[:, None]
        if L[6] > 0.0 or L[7] != 0.0:
            E = 1.0 / (dist * dist + L[6] * L[6])
        else:
            E = 1.0 / (dist * dist)
        tl = np.abs(np.sum(T * l, 1))
        diff = np.sqrt(np.clip(1 - tl * tl, 0, 1))
        spec = diff ** 40 * 0.06
        # head occlusion: distance from the head centre to the segment strand->light
        hc = head_c - pts
        tproj = np.clip(np.sum(hc * l, 1), 0, dist)
        closest = np.linalg.norm(hc - l * tproj[:, None], axis=1)
        sh = np.clip((closest - head_r * 0.75) / (head_r * 0.5), 0, 1)
        k = E * sh
        col += k[:, None] * (alb[None] * diff[:, None] * 1.2 + spec[:, None]) * L[3:6][None]
    rd = env[0:3]
    tr = np.abs(T @ rd)
    rim = np.sqrt(np.clip(1 - tr * tr, 0, 1)) * 0.9
    col += rim[:, None] * env[3:6][None] * 0.30
    col += alb[None] * env[6:9][None] * 2.0
    return col


def hair_strands(cam, hair_pts, head, t, lights, env, per=16, seed=5):
    """Fine strands along the simulated hair chains: returns arrays for draw_strands."""
    from core import fnoise1
    rng = np.random.default_rng(seed)
    P2s, Zs, Cs, As, Ws = [], [], [], [], []
    m = 22
    zc = head.C[2]
    for q, hp in enumerate(hair_pts):
        hp = np.asarray(hp, np.float64)
        n = len(hp)
        base3 = np.c_[hp, np.full(n, zc)]
        for k in range(per):
            ph = rng.uniform(0, 6.28)
            spread = rng.uniform(0.006, 0.030)
            zo = rng.uniform(-0.05, 0.05)
            length = rng.uniform(0.65, 1.0)
            P = base3.copy()
            s = np.linspace(0, 1, n)
            # offset grows along the strand: in-plane normal + depth; wiggles in time
            tg = np.gradient(P[:, :2], axis=0)
            tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
            nrm2 = np.stack([-tg[:, 1], tg[:, 0]], 1)
            wig = np.array([fnoise1(t * 2.3 + ph + 3.1 * si, 5.0 + q, 2) for si in s])
            off = (spread * (rng.uniform(-1, 1) + 0.6 * wig)) * s ** 1.3
            P[:, :2] += nrm2 * off[:, None]
            P[:, 2] += zo * (0.3 + 0.7 * s)
            P = _spline(P, m)
            keep = int(max(4, round(m * length)))
            P = P[:keep]
            T = np.gradient(P, axis=0)
            T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
            sx, sy, z = cam.project(P)
            dist = np.linalg.norm(P - cam.pos, axis=1)
            col = strand_light(P, T, lights, env, head.p(0.0, 0.02))
            ss = np.linspace(0, 1, keep)
            a = 0.70 * (1 - ss ** 2.5)
            wpx = np.maximum(0.22, cam.f * 0.00030 / np.maximum(z, 0.05)) * (1.3 - 0.6 * ss)
            P2s.append(np.stack([sx, sy], 1)); Zs.append(dist); Cs.append(col); As.append(a); Ws.append(wpx)
    return P2s, Zs, Cs, As, Ws


def lashes(cam, head, expr, lights, env):
    """Upper lashes of both eyes: short dark curves off the lid margin, curling up and forward."""
    e = dict(blink=0.0, squint=0.0)
    e.update(expr or {})
    P2s, Zs, Cs, As, Ws = [], [], [], [], []
    s = head.s
    up0 = 0.36 * (1 - e['blink']) - 0.30 * e['blink'] - 0.12 * e['squint']
    for sw in (-1, 1):
        c = head.p(0.0665, -0.001, sw * 0.031)
        for k in range(11):
            az = -0.7 + 1.9 * k / 10.0                 # medial .. lateral (rad)
            am = 1.40 if az >= 0 else 0.95
            tt = az / am
            cu = math.cos(tt * 1.5707963) ** 0.55
            el = up0 * cu + 0.05 * tt + 0.02
            d = (math.cos(el) * math.cos(az) * head.U + math.sin(el) * head.V + math.cos(el) * math.sin(az) * sw * head.W)
            root = c + d * 0.0136 * s
            L = (0.0075 + 0.0025 * cu) * s
            upv = nrm(head.V * 0.8 + d * 0.6)
            pts = np.array([root, root + d * L * 0.45 + upv * L * 0.15, root + d * L * 0.75 + upv * L * 0.55,
                            root + d * L * 0.85 + upv * L * 0.95])
            sx, sy, z = cam.project(pts)
            dist = np.linalg.norm(pts - cam.pos, axis=1)
            T = np.gradient(pts, axis=0)
            T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
            col = strand_light(pts, T, lights, env, head.p(-0.02, 0.02), head_r=0.05, alb=(0.012, 0.008, 0.006))
            P2s.append(np.stack([sx, sy], 1)); Zs.append(dist); Cs.append(col)
            As.append(np.array([0.9, 0.9, 0.8, 0.5]))
            Ws.append(np.maximum(0.3, cam.f * 0.00022 / np.maximum(z, 0.05)) * np.array([1.2, 1.0, 0.8, 0.5]))
    return P2s, Zs, Cs, As, Ws


def draw_fine(res, cam, sets):
    """Composite strand sets into a render() result in place (far strands first)."""
    if res is None:
        return res
    Y0, X0, rgb, a, d = res
    items = []
    for (P2s, Zs, Cs, As, Ws) in sets:
        for i in range(len(P2s)):
            items.append((float(np.mean(Zs[i])), P2s[i], Zs[i], Cs[i], As[i], Ws[i]))
    if not items:
        return res
    items.sort(key=lambda x: -x[0])
    # pad to equal length per batch of equal-length strands
    groups = {}
    for it in items:
        groups.setdefault(len(it[1]), []).append(it)
    rgb64 = rgb.astype(np.float64)
    a64 = a.astype(np.float64)
    d64 = d.astype(np.float64)
    for m, its in sorted(groups.items(), key=lambda kv: -np.mean([x[0] for x in kv[1]])):
        P2 = np.array([x[1] for x in its], np.float64)
        Z = np.array([x[2] for x in its], np.float64)
        C = np.array([x[3] for x in its], np.float64)
        A = np.array([x[4] for x in its], np.float64)
        W = np.array([x[5] for x in its], np.float64)
        hs.draw_strands(rgb64, a64, d64, X0, Y0, P2, Z, C, A, W)
    return Y0, X0, rgb64.astype(np.float32), a64.astype(np.float32), d


def cap_strands(cam, head, t, lights, env, n=220, seed=9):
    """Hair combed back by the wind over the skull: strands from the hairline to the nape, lying
    just above the hair cap, shaded with the cap normal (so the cap stops reading as a helmet)."""
    from core import fnoise1
    rng = np.random.default_rng(seed)
    s = head.s
    c = np.array([-0.006, 0.033])
    r = np.array([0.096, 0.094, 0.077]) * 1.02
    P2s, Zs, Cs, As, Ws = [], [], [], [], []
    m = 16
    for k in range(n):
        # start on the hairline: forehead top (a~0) round the temple (a~75deg) to above the ear
        a0 = rng.uniform(-0.2, 1.75)
        b0 = 0.62 - 0.52 * min(1.0, abs(a0) / 1.3) + rng.normal(0, 0.03) - (0.07 if k % 3 == 0 else 0.0)
        a1 = math.pi * rng.uniform(0.80, 0.98) * (1 if a0 >= 0 else -1)
        b1 = rng.uniform(-0.45, -0.05)
        ph = rng.uniform(0, 6.28)
        us = np.linspace(0, 1, m)
        al = a0 + (a1 - a0) * us ** 0.9
        be = b0 + (b1 - b0) * us ** 1.2 + 0.03 * np.sin(us * 5 + ph)
        lift = 0.0012 + 0.003 * us + 0.002 * (0.5 + 0.5 * math.sin(t * 3.0 + ph)) * us
        u_ = c[0] + (r[0] + lift) * np.cos(al) * np.cos(be)
        v_ = c[1] + (r[1] + lift) * np.sin(be)
        w_ = (r[2] + lift) * np.sin(al) * np.cos(be)
        P = np.array([head.p(u_[i], v_[i], w_[i]) for i in range(m)])
        Nl = np.stack([np.cos(al) * np.cos(be) / r[0], np.sin(be) / r[1], np.sin(al) * np.cos(be) / r[2]], 1)
        N = Nl[:, 0:1] * head.U + Nl[:, 1:2] * head.V + Nl[:, 2:3] * head.W
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        sx, sy, z = cam.project(P)
        dist = np.linalg.norm(P - cam.pos, axis=1)
        col = np.zeros((m, 3))
        for L in lights:
            d = L[:3] - P
            dd = np.linalg.norm(d, axis=1) + 1e-9
            l = d / dd[:, None]
            E = 1.0 / (dd * dd + L[6] * L[6])
            ndl = np.clip((np.sum(N * l, 1) + 0.25) / 1.25, 0, 1)
            tl = np.abs(np.sum(T * l, 1))
            kk = np.sqrt(np.clip(1 - tl * tl, 0, 1))
            col += (E * ndl)[:, None] * (np.array([0.034, 0.021, 0.014])[None] * kk[:, None] + 0.09 * kk[:, None] ** 24) * L[3:6][None]
        ndr = np.clip(N @ env[0:3], 0, 1)
        col += (ndr ** 2)[:, None] * env[3:6][None] * 0.5
        a = 0.60 * np.minimum(1.0, us * 5 + 0.15) * (1 - us ** 3)
        wpx = np.maximum(0.2, cam.f * 0.00025 / np.maximum(z, 0.05))
        P2s.append(np.stack([sx, sy], 1)); Zs.append(dist); Cs.append(col); As.append(a); Ws.append(wpx)
    return P2s, Zs, Cs, As, Ws


def flyaways(cam, head, t, lights, env, n=46, seed=13):
    """Loose strands lifting off the crown and temple in the wind, catching the firelight."""
    from core import fnoise1
    rng = np.random.default_rng(seed)
    P2s, Zs, Cs, As, Ws = [], [], [], [], []
    m = 14
    c = np.array([-0.006, 0.033])
    r = np.array([0.096, 0.094, 0.077]) * 1.03
    for k in range(n):
        al = rng.uniform(0.35, 2.9)                    # round the side of the head, temple -> nape
        uu = c[0] + r[0] * math.cos(al)
        vv = 0.030 + 0.3476 * (uu - 0.092) - rng.uniform(0.004, 0.030)   # just under the brim
        ww = r[2] * math.sin(al) * rng.choice([1.0, 1.0, -0.4])
        root = head.p(uu, vv, ww)
        L = rng.uniform(0.06, 0.18)
        ph = rng.uniform(0, 6.28)
        pts = [root]
        d = nrm(-head.U * 0.8 + head.V * rng.uniform(0.1, 0.6) + np.array([0.6, 0.0, 0.0]))
        for i in range(1, m):
            s_ = i / (m - 1)
            wig = fnoise1(t * 4.0 + ph + s_ * 3.0, 17.0 + k, 2)
            d = nrm(d + np.array([0.35, 0.10 * wig, 0.0]) * 0.25 + head.V * 0.08 * wig)
            pts.append(pts[-1] + d * L / (m - 1))
        P = np.array(pts)
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        sx, sy, z = cam.project(P)
        col = strand_light(P, T, lights, env, head.p(0.0, 0.02), head_r=0.085)
        ss = np.linspace(0, 1, m)
        P2s.append(np.stack([sx, sy], 1)); Zs.append(np.linalg.norm(P - cam.pos, axis=1)); Cs.append(col)
        As.append(0.55 * (1 - ss ** 2)); Ws.append(np.maximum(0.2, cam.f * 0.00022 / np.maximum(z, 0.05)) * np.ones(m))
    return P2s, Zs, Cs, As, Ws
