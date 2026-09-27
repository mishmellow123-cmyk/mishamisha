"""Small 3-D SDF objects for the mountain world: roped walkers, the great lantern of iron and glass, poles,
cairns. Each object is a union of round cones / capsules and rounded oriented boxes, sphere-traced per pixel
inside its projected bounding sphere, in the SOURCE (lens-shift) camera, depth-tested against the terrain.

Shading: wrapped Lambert from point lights (lanterns, fires; inverse square) + the moon (directional) + sky
ambient; iron gets a Blinn highlight; glass panes are thin emissive shells the ray passes through (the lantern
reads as lit glass between dark iron bars); cloth takes a per-object tint. No shadows (the figures are 20-40 px
in the wides; in the close-up the light is inside the lantern, so the bars' silhouettes are what matter).

Primitive row (NP floats):
  0 type (0 round cone a->b | 1 rounded box | 2 glass box shell | 3 torus about +y)
  1-3 a | 4-6 b (box: 4 = yaw, 5 = pitch-about-right) | 7 ra (box: rounding) | 8 rb | 9 material
  10 blend k (smooth union with the object so far) | 11-13 box half extents | 14 torus major R
Object row (NO floats): 0 first prim | 1 n prims | 2-4 bound centre | 5 bound radius | 6-8 cloth rgb
  | 9 emission gain (glass) | 10-12 glass rgb | 13 depth bias (m)
Materials: 0 cloth | 1 iron | 2 wood | 3 small-lantern glass (emissive) | 4 rope | 5 stone
"""
import math

import numpy as np
from numba import njit, prange

NP = 16
NO = 16


class Scene:
    def __init__(self):
        self.P = []
        self.O = []

    def begin(self, rgb=(0.05, 0.05, 0.05), emit=0.0, glass=(1.0, 1.0, 1.0), zbias=0.02):
        self._o = [len(self.P), 0, 0.0, 0.0, 0.0, 0.0, rgb[0], rgb[1], rgb[2], emit, glass[0], glass[1], glass[2],
                   zbias, 0.0, 0.0]
        self._pts = []

    def _add(self, row, pts):
        self.P.append(row)
        self._o[1] += 1
        self._pts.extend(pts)

    def cone(self, a, b, ra, rb=None, mat=0, k=0.0):
        rb = ra if rb is None else rb
        r = np.zeros(NP)
        r[0] = 0
        r[1:4] = a
        r[4:7] = b
        r[7] = ra
        r[8] = rb
        r[9] = mat
        r[10] = k
        R = max(ra, rb)
        self._add(r, [np.asarray(a) + R, np.asarray(a) - R, np.asarray(b) + R, np.asarray(b) - R])

    def box(self, c, half, yaw=0.0, pitch=0.0, rnd=0.0, mat=0, k=0.0, glass=False):
        r = np.zeros(NP)
        r[0] = 2 if glass else 1
        r[1:4] = c
        r[4] = yaw
        r[5] = pitch
        r[7] = rnd
        r[9] = mat
        r[10] = k
        r[11:14] = half
        R = float(np.linalg.norm(half)) + rnd
        c = np.asarray(c)
        self._add(r, [c + R, c - R])

    def torus(self, c, R, r_, mat=1, k=0.0):
        r = np.zeros(NP)
        r[0] = 3
        r[1:4] = c
        r[7] = r_
        r[9] = mat
        r[10] = k
        r[14] = R
        c = np.asarray(c)
        self._add(r, [c + R + r_, c - R - r_])

    def end(self):
        if self._o[1] == 0:
            return
        pts = np.array(self._pts)
        lo, hi = pts.min(0), pts.max(0)
        c = 0.5 * (lo + hi)
        self._o[2:5] = c
        self._o[5] = 0.5 * float(np.linalg.norm(hi - lo)) + 0.02
        self.O.append(self._o)

    def arrays(self):
        P = np.array(self.P, np.float64).reshape(-1, NP)
        O = np.array(self.O, np.float64).reshape(-1, NO)
        return P, O


# ----------------------------------------------------------------- SDF ---

@njit(inline='always', fastmath=True)
def _sd_cone(px, py, pz, R):
    ax, ay, az = R[1], R[2], R[3]
    bx, by, bz = R[4], R[5], R[6]
    ra, rb = R[7], R[8]
    bax, bay, baz = bx - ax, by - ay, bz - az
    pax, pay, paz = px - ax, py - ay, pz - az
    l2 = bax * bax + bay * bay + baz * baz + 1e-12
    h = (pax * bax + pay * bay + paz * baz) / l2
    h = min(max(h, 0.0), 1.0)
    qx, qy, qz = pax - bax * h, pay - bay * h, paz - baz * h
    return math.sqrt(qx * qx + qy * qy + qz * qz) - (ra + (rb - ra) * h)


@njit(inline='always', fastmath=True)
def _sd_box(px, py, pz, R):
    dx, dy, dz = px - R[1], py - R[2], pz - R[3]
    cy, sy = math.cos(R[4]), math.sin(R[4])
    # yaw: local x = right, z = forward (fwd = (sin yaw, 0, cos yaw))
    lx = dx * cy - dz * sy
    lz = dx * sy + dz * cy
    ly = dy
    if R[5] != 0.0:
        cp, sp = math.cos(R[5]), math.sin(R[5])
        ly2 = ly * cp - lz * sp
        lz = ly * sp + lz * cp
        ly = ly2
    rr = R[7]
    qx = abs(lx) - R[11] + rr
    qy = abs(ly) - R[12] + rr
    qz = abs(lz) - R[13] + rr
    ox, oy, oz = max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)
    return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0) - rr


@njit(inline='always', fastmath=True)
def _sd_torus(px, py, pz, R):
    dx, dy, dz = px - R[1], py - R[2], pz - R[3]
    q = math.sqrt(dx * dx + dz * dz) - R[14]
    return math.sqrt(q * q + dy * dy) - R[7]


@njit(inline='always', fastmath=True)
def _prim(px, py, pz, R):
    t = R[0]
    if t < 0.5:
        return _sd_cone(px, py, pz, R)
    if t < 2.5:
        return _sd_box(px, py, pz, R)
    return _sd_torus(px, py, pz, R)


@njit(inline='always', fastmath=True)
def _smin(a, b, k):
    if k <= 0.0:
        return min(a, b)
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


@njit(inline='always', fastmath=True)
def _map(px, py, pz, P, i0, n):
    """Opaque SDF of an object (glass excluded) and the material of the closest primitive."""
    d = 1e9
    mat = 0.0
    for q in range(i0, i0 + n):
        if P[q, 0] > 1.5 and P[q, 0] < 2.5:
            continue
        dq = _prim(px, py, pz, P[q])
        if dq < d:
            mat = P[q, 9]
        d = _smin(d, dq, P[q, 10])
    return d, mat


@njit(inline='always', fastmath=True)
def _glass(px, py, pz, P, i0, n):
    d = 1e9
    for q in range(i0, i0 + n):
        if P[q, 0] > 1.5 and P[q, 0] < 2.5:
            dq = abs(_sd_box(px, py, pz, P[q])) - 0.004
            if dq < d:
                d = dq
    return d


@njit(parallel=True, fastmath=True, cache=True)
def render(img, zb, C, P, O, LT, moon, amb, fogp, cam_y):
    """C: SrcCam.params() (pos, fwd xz, right xz, f, cx, cyy, W, H). LT: point lights (N,8) as world.shade.
    moon: [dx, dy, dz, r, g, b, I]. amb: sky ambient rgb. fogp: world fog params (aerial perspective)."""
    H, W = img.shape[0], img.shape[1]
    ox, oy, oz = C[0], C[1], C[2]
    fx, fz, rx, rz = C[3], C[4], C[5], C[6]
    f, cx, cyy = C[7], C[8], C[9]
    for oi in range(O.shape[0]):
        bcx, bcy, bcz, br = O[oi, 2], O[oi, 3], O[oi, 4], O[oi, 5]
        vx, vy, vz = bcx - ox, bcy - oy, bcz - oz
        zc = vx * fx + vz * fz
        if zc < 0.05:
            continue
        xc = vx * rx + vz * rz
        sx = cx + f * xc / zc
        sy = cyy - f * vy / zc
        rp = f * br / max(zc - br, 0.05) + 2.0
        x0 = int(max(0.0, sx - rp))
        x1 = int(min(W * 1.0, sx + rp + 1.0))
        y0 = int(max(0.0, sy - rp))
        y1 = int(min(H * 1.0, sy + rp + 1.0))
        if x1 <= x0 or y1 <= y0:
            continue
        i0 = int(O[oi, 0])
        n = int(O[oi, 1])
        for py in prange(y0, y1):
            for px in range(x0, x1):
                # ray through the source pixel (verticals vertical, lens shift in cyy)
                u = px + 0.5 - cx
                v = cyy - (py + 0.5)
                dx = fx * f + rx * u
                dy = v
                dz = fz * f + rz * u
                inv = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
                dx *= inv
                dy *= inv
                dz *= inv
                # bounding sphere
                lx, ly, lz = ox - bcx, oy - bcy, oz - bcz
                bq = lx * dx + ly * dy + lz * dz
                cq = lx * lx + ly * ly + lz * lz - br * br
                disc = bq * bq - cq
                if disc <= 0.0:
                    continue
                sq = math.sqrt(disc)
                t = max(-bq - sq, 0.02)
                t1 = -bq + sq
                # depth of the terrain behind this pixel (horizontal-forward depth)
                hz = dx * fx + dz * fz
                zlim = zb[py, px]
                hit = False
                mat = 0.0
                gl = 0.0
                ing = False
                for _s in range(96):
                    qx, qy, qz = ox + dx * t, oy + dy * t, oz + dz * t
                    d, mat = _map(qx, qy, qz, P, i0, n)
                    dg = _glass(qx, qy, qz, P, i0, n)
                    eps = max(0.35 * t / f, 0.0006)
                    if dg < eps:
                        if not ing:
                            gl += 1.0
                            ing = True
                    elif dg > 3.0 * eps:
                        ing = False
                    if d < eps:
                        hit = True
                        break
                    st = min(d, max(dg, eps)) if dg < 0.05 else d
                    t += max(st * 0.9, eps * 0.5)
                    if t > t1 or t * hz > zlim:
                        break
                emr = O[oi, 9] * O[oi, 10]
                emg = O[oi, 9] * O[oi, 11]
                emb = O[oi, 9] * O[oi, 12]
                if not hit:
                    if gl > 0.0 and emr + emg + emb > 0.0:
                        img[py, px, 0] += 0.5 * gl * emr
                        img[py, px, 1] += 0.5 * gl * emg
                        img[py, px, 2] += 0.5 * gl * emb
                    continue
                zhit = t * hz
                if zhit > zlim - O[oi, 13] * 0.0:
                    if zhit > zlim:
                        continue
                qx, qy, qz = ox + dx * t, oy + dy * t, oz + dz * t
                e = max(0.5 * t / f, 0.0008)
                d1, _m = _map(qx + e, qy - e, qz - e, P, i0, n)
                d2, _m = _map(qx - e, qy - e, qz + e, P, i0, n)
                d3, _m = _map(qx - e, qy + e, qz - e, P, i0, n)
                d4, _m = _map(qx + e, qy + e, qz + e, P, i0, n)
                nx = d1 - d2 - d3 + d4
                ny = -d1 - d2 + d3 + d4
                nz = -d1 + d2 - d3 + d4
                nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
                nx /= nl
                ny /= nl
                nz /= nl
                if mat < 0.5:
                    ar, ag, ab = O[oi, 6], O[oi, 7], O[oi, 8]
                    wrap = 0.35
                    spec = 0.0
                elif mat < 1.5:
                    ar, ag, ab = 0.035, 0.034, 0.033
                    wrap = 0.1
                    spec = 0.5
                elif mat < 2.5:
                    ar, ag, ab = 0.10, 0.065, 0.040
                    wrap = 0.2
                    spec = 0.05
                elif mat < 3.5:
                    ar, ag, ab = 0.0, 0.0, 0.0
                    wrap = 0.0
                    spec = 0.3
                elif mat < 4.5:
                    ar, ag, ab = 0.09, 0.075, 0.055
                    wrap = 0.4
                    spec = 0.0
                else:
                    ar, ag, ab = 0.075, 0.075, 0.08
                    wrap = 0.2
                    spec = 0.0
                cr = ar * amb[0] * (0.6 + 0.4 * ny)
                cg = ag * amb[1] * (0.6 + 0.4 * ny)
                cb = ab * amb[2] * (0.6 + 0.4 * ny)
                # moon
                ndm = nx * moon[0] + ny * moon[1] + nz * moon[2]
                wm = max((ndm + wrap) / (1.0 + wrap), 0.0) * moon[6]
                cr += ar * moon[3] * wm
                cg += ag * moon[4] * wm
                cb += ab * moon[5] * wm
                if spec > 0.0 and ndm > 0.0:
                    hx, hy, hz2 = moon[0] - dx, moon[1] - dy, moon[2] - dz
                    hl = math.sqrt(hx * hx + hy * hy + hz2 * hz2) + 1e-9
                    s = spec * max((nx * hx + ny * hy + nz * hz2) / hl, 0.0) ** 40 * moon[6] * 0.3
                    cr += s * moon[3]
                    cg += s * moon[4]
                    cb += s * moon[5]
                for li in range(LT.shape[0]):
                    lx2 = LT[li, 0] - qx
                    ly2 = LT[li, 1] - qy
                    lz2 = LT[li, 2] - qz
                    l2 = lx2 * lx2 + ly2 * ly2 + lz2 * lz2
                    if l2 > LT[li, 6] * 2.0e4:
                        continue
                    ll = math.sqrt(l2) + 1e-9
                    ndl = (nx * lx2 + ny * ly2 + nz * lz2) / ll
                    E = LT[li, 6] / (l2 + LT[li, 7] * LT[li, 7])
                    wl = max((ndl + wrap) / (1.0 + wrap), 0.0) * E
                    cr += ar * LT[li, 3] * wl
                    cg += ag * LT[li, 4] * wl
                    cb += ab * LT[li, 5] * wl
                    if spec > 0.0 and ndl > 0.0:
                        hx = lx2 / ll - dx
                        hy = ly2 / ll - dy
                        hz2 = lz2 / ll - dz
                        hl = math.sqrt(hx * hx + hy * hy + hz2 * hz2) + 1e-9
                        s = spec * max((nx * hx + ny * hy + nz * hz2) / hl, 0.0) ** 40 * E
                        cr += s * LT[li, 3]
                        cg += s * LT[li, 4]
                        cb += s * LT[li, 5]
                if mat > 2.5 and mat < 3.5:
                    # small lantern glass: emissive, a little brighter facing the camera
                    k = 0.7 + 0.3 * abs(nx * dx + ny * dy + nz * dz)
                    cr += emr * k
                    cg += emg * k
                    cb += emb * k
                if gl > 0.0:
                    cr += 0.5 * gl * emr
                    cg += 0.5 * gl * emg
                    cb += 0.5 * gl * emb
                # aerial perspective (same law as world.shade)
                dist = t
                tr = math.exp(-fogp[0] * dist * math.exp(-fogp[1] * max(qy, cam_y) * 0.0))
                img[py, px, 0] = cr * tr + fogp[5] * (1.0 - tr)
                img[py, px, 1] = cg * tr + fogp[6] * (1.0 - tr)
                img[py, px, 2] = cb * tr + fogp[7] * (1.0 - tr)
                zb[py, px] = zhit


# ------------------------------------------------------------- figures ---

def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def walker(sc, feet, w, phase, rgb, lantern=None, staff=False, pack=True, height=1.0, lean=0.08, hood=True,
           carry=None, arm_out=0.0, carry_side=1.0):
    """A roped walker in a hooded parka, seen at any angle. feet: ground point under the pelvis; w: walking
    direction (horizontal unit); phase: walk-cycle phase (radians). lantern: None, or 'hand' -> returns the
    lantern's world position (it hangs from the leading hand, swinging with the step). carry: None, or
    (left_shoulder_target, right_shoulder_target) for pole bearers (hands up on the poles).
    Returns dict(waist=..., lantern=...)."""
    up = np.array([0.0, 1.0, 0.0])
    w = _unit([w[0], 0.0, w[2]])
    s = np.cross(up, w)            # left of the walker
    h = height
    sn, cs = math.sin(phase), math.cos(phase)
    bob = 0.018 * h * math.cos(2 * phase)
    pel = feet + up * (0.93 * h + bob)
    fwd_lean = w * math.sin(lean) + up * math.cos(lean)
    chest = pel + fwd_lean * 0.50 * h
    neck = chest + fwd_lean * 0.10 * h
    head = neck + up * 0.11 * h + w * 0.02 * h
    sc.begin(rgb=rgb)
    # legs: hip flexion +-24 deg, knee flexes in swing
    out = {}
    for side, ph in ((1.0, 0.0), (-1.0, math.pi)):
        a = phase + ph
        th = math.radians(22.0) * math.sin(a)
        kn = math.radians(8.0 + 42.0 * max(0.0, math.sin(a + 1.3)) ** 2)
        hip = pel + s * side * 0.085 * h
        thigh = (w * math.sin(th) - up * math.cos(th)) * 0.45 * h
        knee = hip + thigh
        tk = th - kn
        shin = (w * math.sin(tk) - up * math.cos(tk)) * 0.45 * h
        ank = knee + shin
        sc.cone(hip, knee, 0.078 * h, 0.062 * h, 0, 0.03)
        sc.cone(knee, ank, 0.060 * h, 0.048 * h, 0, 0.02)
        sc.cone(ank + up * 0.02, ank + w * 0.13 * h + up * 0.01, 0.045 * h, 0.04 * h, 0, 0.02)
    # parka: torso, skirt
    sc.cone(pel - up * 0.02, chest, 0.15 * h, 0.172 * h, 0, 0.05)
    sc.cone(pel, pel - up * 0.22 * h - w * 0.02, 0.17 * h, 0.175 * h, 0, 0.05)
    sc.cone(chest, neck, 0.15 * h, 0.09 * h, 0, 0.04)
    # hood (a hooded head: no face ever shows) with a little peak at the back
    sc.cone(head - w * 0.01, head + up * 0.02, 0.118 * h, 0.12 * h, 0, 0.04)
    if hood:
        sc.cone(head - w * 0.06 * h + up * 0.04 * h, head - w * 0.12 * h - up * 0.03 * h, 0.085 * h, 0.05 * h, 0,
                0.05)
    if pack:
        sc.box(chest - w * 0.19 * h - up * 0.14 * h, (0.17 * h, 0.23 * h, 0.11 * h),
               yaw=math.atan2(w[0], w[2]), rnd=0.05 * h, mat=0, k=0.04)
    # arms
    shL = chest - up * 0.03 * h + s * 0.2 * h
    shR = chest - up * 0.03 * h - s * 0.2 * h
    if carry is not None:
        # one hand steadies the pole on its shoulder; the other arm hangs free and swings a little
        sh, tgt = (shL, carry[0]) if carry_side > 0 else (shR, carry[1])
        el = sh + 0.5 * (tgt - sh) + up * (-0.20 * h) + w * 0.02
        sc.cone(sh, el, 0.058 * h, 0.05 * h, 0, 0.02)
        sc.cone(el, tgt, 0.05 * h, 0.042 * h, 0, 0.02)
        sh2 = shR if carry_side > 0 else shL
        so = -s if carry_side > 0 else s
        sw = 0.15 * math.sin(phase)
        el2 = sh2 + (w * 0.04 * sw - up * 0.28 + so * 0.05) * h
        hd2 = sh2 + (w * (0.10 + 0.1 * sw) - up * 0.55 + so * 0.07) * h
        sc.cone(sh2, el2, 0.058 * h, 0.05 * h, 0, 0.02)
        sc.cone(el2, hd2, 0.05 * h, 0.044 * h, 0, 0.02)
    else:
        # the lantern hand leads a little; the other hand holds a staff or swings
        sw = 0.20 * math.sin(phase + math.pi)
        hand_l = shR + (w * (0.18 + 0.1 * sw) - up * 0.52 - s * (0.06 + arm_out)) * h
        el_l = shR + (w * 0.05 - up * 0.27 - s * 0.05) * h
        sc.cone(shR, el_l, 0.058 * h, 0.05 * h, 0, 0.02)
        sc.cone(el_l, hand_l, 0.05 * h, 0.044 * h, 0, 0.02)
        if staff:
            hand_s = shL + (w * 0.22 - up * 0.38 + s * 0.08) * h
            el_s = shL + (w * 0.02 - up * 0.26 + s * 0.08) * h
            sc.cone(shL, el_s, 0.058 * h, 0.05 * h, 0, 0.02)
            sc.cone(el_s, hand_s, 0.05 * h, 0.044 * h, 0, 0.02)
            tip = feet + w * (0.55 + 0.15 * math.sin(phase)) * h + s * 0.2 * h
            sc.cone(hand_s + up * 0.25 * h, tip, 0.016, 0.014, 2, 0.0)
        else:
            sw2 = -sw
            hand_s = shL + (w * (0.12 * sw2) - up * 0.55 + s * 0.05) * h
            el_s = shL + (w * 0.03 * sw2 - up * 0.28 + s * 0.04) * h
            sc.cone(shL, el_s, 0.058 * h, 0.05 * h, 0, 0.02)
            sc.cone(el_s, hand_s, 0.05 * h, 0.044 * h, 0, 0.02)
        if lantern == 'hand':
            # hanging lantern: a small pendulum lagging the hand
            sway = 0.10 * math.sin(phase * 2.0 - 0.6)
            out['lantern'] = hand_l - up * 0.20 * h + w * sway * 0.3
            out['hand'] = hand_l
    sc.end()
    out['waist'] = pel - up * 0.02 * h - s * 0.12 * h
    out['head'] = head
    out['chest'] = chest
    out['shoulders'] = (shL, shR)
    return out


def small_lantern(sc, top, rgb_glow, gain, w=(0.0, 0.0, 1.0)):
    """A small hand lantern hanging from `top` (the hand): a glass body with an iron cap and base."""
    up = np.array([0.0, 1.0, 0.0])
    sc.begin(rgb=(0.03, 0.03, 0.03), emit=gain, glass=rgb_glow)
    body_c = top - up * 0.13
    sc.cone(top, top - up * 0.035, 0.012, 0.012, 1, 0.0)        # bail
    sc.cone(top - up * 0.04, top - up * 0.07, 0.015, 0.05, 1, 0.0)  # cap
    sc.cone(body_c + up * 0.055, body_c - up * 0.055, 0.042, 0.042, 3, 0.0)  # glass (emissive)
    sc.cone(body_c - up * 0.06, body_c - up * 0.075, 0.05, 0.05, 1, 0.0)    # base
    sc.end()
    return body_c


def great_lantern(sc, c, yaw, glass_rgb, gain, scale=1.0):
    """The great lantern of iron and glass, centred at c (the heart), hung from a crossbar. Hexagonal: six
    glass panes between iron posts, a pierced pyramidal roof, a finial ring, a heavy base."""
    up = np.array([0.0, 1.0, 0.0])
    k = scale
    sc.begin(rgb=(0.03, 0.03, 0.03), emit=gain, glass=glass_rgb, zbias=0.05)
    R = 0.24 * k                    # radius to the posts
    hb = 0.30 * k                   # half height of the glass body
    for i in range(6):
        a = yaw + i * math.pi / 3.0
        p = c + np.array([math.sin(a), 0.0, math.cos(a)]) * R
        sc.cone(p - up * hb, p + up * hb, 0.016 * k, 0.016 * k, 1, 0.0)
        # glass pane between this post and the next
        a2 = a + math.pi / 6.0
        pc = c + np.array([math.sin(a2), 0.0, math.cos(a2)]) * R * 0.866
        sc.box(pc, (R * 0.5, hb, 0.004), yaw=a2, glass=True, mat=0)
        # a horizontal glazing bar at 1/3 height
        q = c + np.array([math.sin(a + math.pi / 3.0), 0.0, math.cos(a + math.pi / 3.0)]) * R
        sc.cone(p - up * hb * 0.25, q - up * hb * 0.25, 0.007 * k, 0.007 * k, 1, 0.0)
    # rings top and bottom of the glass
    sc.torus(c + up * hb, R * 1.02, 0.02 * k, 1, 0.0)
    sc.torus(c - up * hb, R * 1.02, 0.022 * k, 1, 0.0)
    # roof: a hexagonal pyramid approximated by a round cone, with a chimney cap and a ring
    sc.cone(c + up * (hb + 0.01 * k), c + up * (hb + 0.20 * k), R * 1.12, 0.05 * k, 1, 0.0)
    sc.cone(c + up * (hb + 0.18 * k), c + up * (hb + 0.26 * k), 0.06 * k, 0.045 * k, 1, 0.0)
    sc.torus(c + up * (hb + 0.33 * k), 0.06 * k, 0.012 * k, 1, 0.0)
    # base: a heavy dish and a short foot
    sc.cone(c - up * (hb + 0.005 * k), c - up * (hb + 0.07 * k), R * 1.1, R * 0.7, 1, 0.01 * k)
    sc.cone(c - up * (hb + 0.07 * k), c - up * (hb + 0.13 * k), 0.05 * k, 0.03 * k, 1, 0.0)
    sc.end()
    return c + up * (hb + 0.33 * k)      # the hanging point (the finial ring)
