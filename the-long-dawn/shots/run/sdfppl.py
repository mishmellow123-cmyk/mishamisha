"""Small 3-D SDF objects for the mountain world: roped walkers, the great lantern of iron and glass, poles,
cairns. Each object is a union of round cones / capsules and rounded oriented boxes, sphere-traced per pixel
inside its projected bounding sphere, in the SOURCE (lens-shift) camera, depth-tested against the terrain.

Shading: wrapped Lambert from point lights (lanterns, fires; inverse square) + the moon (directional) + sky
ambient; iron gets a Blinn highlight; glass panes are thin emissive shells the ray passes through (the lantern
reads as lit glass between dark iron bars); cloth takes a per-object tint. No shadows (the figures are 20-40 px
in the wides; in the close-up the light is inside the lantern, so the bars' silhouettes are what matter).

Primitive row (NP floats):
  0 type (0 round cone a->b | 1 rounded box | 2 glass box shell | 3 torus about +y | 4 bell: a hanging cloak)
  1-3 a | 4-6 b (box: 4 = yaw, 5 = pitch-about-right) | 7 ra (box: rounding) | 8 rb | 9 material
  10 blend k (smooth union with the object so far) | 11-13 box half extents (bell: the forward reference)
  14 torus major R (bell: fold depth at the hem) | 15 bell fold count | 16 bell fold phase
Object row (NO floats): 0 first prim | 1 n prims | 2-4 bound centre | 5 bound radius | 6-8 cloth rgb
  | 9 emission gain (glass) | 10-12 glass rgb | 13 depth bias (m)
Materials: 0 cloth | 1 iron | 2 wood | 3 small-lantern glass (emissive) | 4 rope | 5 stone
"""
import math

import numpy as np
from numba import njit, prange

from mt.noise import gnoise3

NP = 20
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

    def bell(self, a, b, ra, rb, fwd, depth=0.03, n=9, phase=0.0, mat=0, k=0.0, ell=0.0, wind=0.0, wind_ang=math.pi):
        """A hanging cloak: a cone from a (shoulders) to b (hem centre) with folds that deepen toward the hem.
        ell: front/back slimmer than the sides (a body is not round: 0.3 = 30% slimmer front to back).
        wind: metres of billow at the hem on the lee side (the windward side presses in, the lee hem lifts and
        flutters); wind_ang: the direction the wind blows TOWARD, radians about the axis from `fwd`, positive toward
        the figure's left (the frame of _sd_bell)."""
        r = np.zeros(NP)
        r[0] = 4
        r[1:4] = a
        r[4:7] = b
        r[7] = ra
        r[8] = rb
        r[9] = mat
        r[10] = k
        r[11:14] = fwd
        r[14] = depth
        r[15] = n
        r[16] = phase
        r[17] = ell
        r[18] = wind
        r[19] = wind_ang
        R = max(ra, rb) + depth + 1.2 * abs(wind)
        self._add(r, [np.asarray(a) + R, np.asarray(a) - R, np.asarray(b) + R, np.asarray(b) - R])

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
def _sd_bell(px, py, pz, R):
    ax, ay, az = R[1], R[2], R[3]
    dx, dy, dz = R[4] - ax, R[5] - ay, R[6] - az
    L = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
    dx /= L
    dy /= L
    dz /= L
    pax, pay, paz = px - ax, py - ay, pz - az
    t = pax * dx + pay * dy + paz * dz
    qx, qy, qz = pax - dx * t, pay - dy * t, paz - dz * t
    rho = math.sqrt(qx * qx + qy * qy + qz * qz) + 1e-9
    # angle about the axis, measured from the figure's forward direction
    fd = R[11] * dx + R[12] * dy + R[13] * dz
    fx, fy, fz = R[11] - dx * fd, R[12] - dy * fd, R[13] - dz * fd
    fl = math.sqrt(fx * fx + fy * fy + fz * fz) + 1e-9
    fx /= fl
    fy /= fl
    fz /= fl
    sx, sy, sz = dy * fz - dz * fy, dz * fx - dx * fz, dx * fy - dy * fx
    th = math.atan2(qx * sx + qy * sy + qz * sz, qx * fx + qy * fy + qz * fz)
    h = min(max(t / L, 0.0), 1.0)
    ph = R[16]
    ell = R[17]
    W = R[18]
    if ell == 0.0 and W == 0.0:
        # v2 bell (kept exactly for any caller that asks for neither)
        fold = R[14] * h * (math.sin(R[15] * th + ph) + 0.45 * math.sin(2.3 * R[15] * th + 1.7 * ph + 1.3))
        r = R[7] + (R[8] - R[7]) * h + fold
        Lh = L * (1.0 + 0.022 * math.sin(3.0 * th + 2.0 * ph) + 0.012 * math.sin(7.0 * th - ph))
        slope = (R[8] - R[7]) / L
        ds = (rho - r) / math.sqrt(1.0 + slope * slope)
        dc = max(-t, t - Lh)
        if ds > 0.0 and dc > 0.0:
            d = math.sqrt(ds * ds + dc * dc)
        else:
            d = max(ds, dc)
        return 0.6 * d          # conservative: the folds make it a non-exact distance
    # v3 CLOTH: weight, creases and wind. Folds hang from the shoulders and open toward the hem; they are unevenly
    # spaced, twist a little down the fall (more in wind), and have soft crests with sharp creases between them
    # (|sin|, softened), not a fluted sine. The windward side presses in, the lee side billows and its hem lifts
    # and flutters (a travelling wave on the phase). A body is flatter front to back than across (ell).
    cw = math.cos(th - R[19])
    lee = 0.5 + 0.5 * cw
    thp = th + 0.32 * math.sin(2.0 * th + 0.6 * ph + 0.4) + (0.20 + 1.6 * abs(W)) * h * math.sin(0.37 * ph + 1.9 * th)
    x = 0.5 * R[15] * thp + 0.5 * ph
    sx2 = math.sin(x)
    g = (math.sqrt(sx2 * sx2 + 0.035) - 0.187) / 0.83           # 0 in a crease, 1 on a crest
    dep = R[14] * (0.55 + 0.45 * math.sin(3.0 * th + 0.23 * ph + 1.1)) * (1.0 + 1.5 * abs(W) * lee)
    fold = dep * h ** 1.15 * (2.0 * g - 1.0)
    hw = h ** 1.4
    billow = W * hw * (1.15 * lee * lee - 0.30 * (1.0 - lee)) \
        + 0.22 * W * hw * lee * math.sin(4.0 * th - 2.3 * ph + 5.0 * h)
    ce = math.cos(th)
    r = (R[7] + (R[8] - R[7]) * h) * (1.0 - ell * ce * ce) + fold + billow
    Lh = L * (1.0 + 0.018 * math.sin(3.0 * th + 2.0 * ph) + 0.010 * math.sin(7.0 * th - ph)) + 0.35 * fold \
        - abs(W) * (0.50 * lee * lee + 0.10 * lee * math.sin(5.0 * th - 2.9 * ph))
    slope = (R[8] - R[7]) / L
    ds = (rho - r) / math.sqrt(1.0 + slope * slope)
    dc = max(-t, t - Lh)
    if ds > 0.0 and dc > 0.0:
        d = math.sqrt(ds * ds + dc * dc)
    else:
        d = max(ds, dc)
    return 0.45 * d         # conservative: creases, billow and flutter make it far from an exact distance


@njit(inline='always', fastmath=True)
def _prim(px, py, pz, R):
    t = R[0]
    if t < 0.5:
        return _sd_cone(px, py, pz, R)
    if t < 2.5:
        return _sd_box(px, py, pz, R)
    if t < 3.5:
        return _sd_torus(px, py, pz, R)
    return _sd_bell(px, py, pz, R)


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
                ao = 1.0
                rimk = 0.0
                if mat < 0.5:
                    # cloth: small hanging creases (noise stretched along the vertical) bend the normal; the albedo
                    # carries long fibre streaks and broad wear (wool), not a blotchy mottle (which reads as clay)
                    fsc = 1.0 / max(e * 6.0, 0.012)
                    fk = min(fsc / 60.0, 1.0)
                    if fk > 0.05:
                        f1 = gnoise3(qx * 16.0, qy * 3.0, qz * 16.0, 11)
                        f2 = gnoise3(qx * 16.0 + 7.1, qy * 3.0, qz * 16.0 - 2.3, 12)
                        nx += 0.35 * fk * f1
                        nz += 0.35 * fk * f2
                        ny += 0.10 * fk * gnoise3(qx * 30.0, qy * 30.0, qz * 30.0, 13)
                        nl2 = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
                        nx /= nl2
                        ny /= nl2
                        nz /= nl2
                    wv = gnoise3(qx * 26.0, qy * 3.2, qz * 26.0, 14)
                    wr = gnoise3(qx * 2.6, qy * 1.8, qz * 2.6, 15)
                    mot = (0.86 + 0.14 * wv) * (0.84 + 0.16 * wr)
                    ar, ag, ab = O[oi, 6] * mot, O[oi, 7] * mot, O[oi, 8] * mot
                    wrap = 0.12                  # little wrap: lights behind a figure rim it, they do not fill it
                    spec = 0.0
                    # the creases hold shadow: occlusion from two probes along the normal (the cloth SDF is scaled
                    # by 0.45, so a flat cloak reads 0.45 x the probe distance, which counts as open)
                    o1, _m = _map(qx + nx * 0.02, qy + ny * 0.02, qz + nz * 0.02, P, i0, n)
                    o2, _m = _map(qx + nx * 0.06, qy + ny * 0.06, qz + nz * 0.06, P, i0, n)
                    oc1 = min(max(1.0 - o1 / (0.45 * 0.02), 0.0), 1.0)
                    oc2 = min(max(1.0 - o2 / (0.45 * 0.06), 0.0), 1.0)
                    ao = min(max(1.0 - 0.8 * (0.45 * oc1 + 0.55 * oc2), 0.3), 1.0)
                    rimk = 0.9                   # wool fibres glow at the edge when a light is behind the figure
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
                cr = ar * amb[0] * (0.6 + 0.4 * ny) * ao
                cg = ag * amb[1] * (0.6 + 0.4 * ny) * ao
                cb = ab * amb[2] * (0.6 + 0.4 * ny) * ao
                aod = 0.45 + 0.55 * ao
                # rim: how edge-on the surface is to the camera (1 at the silhouette)
                edge = 1.0 - abs(nx * dx + ny * dy + nz * dz)
                edge = edge * edge * math.sqrt(edge)
                # moon
                ndm = nx * moon[0] + ny * moon[1] + nz * moon[2]
                wm = max((ndm + wrap) / (1.0 + wrap), 0.0) * moon[6] * aod
                cr += ar * moon[3] * wm
                cg += ag * moon[4] * wm
                cb += ab * moon[5] * wm
                if rimk > 0.0:
                    vm = max(dx * moon[0] + dy * moon[1] + dz * moon[2], 0.0)
                    rm = rimk * edge * (0.25 + vm * vm * vm) * moon[6] * 3.0
                    cr += ar * moon[3] * rm
                    cg += ag * moon[4] * rm
                    cb += ab * moon[5] * rm
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
                    wl = max((ndl + wrap) / (1.0 + wrap), 0.0) * E * aod
                    cr += ar * LT[li, 3] * wl
                    cg += ag * LT[li, 4] * wl
                    cb += ab * LT[li, 5] * wl
                    if rimk > 0.0:
                        vl = max((dx * lx2 + dy * ly2 + dz * lz2) / ll, 0.0)
                        rl = rimk * edge * (0.2 + vl * vl * vl) * E * 2.5
                        cr += ar * LT[li, 3] * rl
                        cg += ag * LT[li, 4] * rl
                        cb += ab * LT[li, 5] * rl
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


def _knee(hip, ank, l1, l2, w):
    """Two-bone IK: the knee for a hip and an ankle, bending forward (toward w)."""
    d = ank - hip
    L = float(np.linalg.norm(d)) + 1e-9
    L2 = min(L, (l1 + l2) * 0.999)
    a = (l1 * l1 - l2 * l2 + L2 * L2) / (2.0 * L2)
    hk = math.sqrt(max(l1 * l1 - a * a, 0.0))
    u = d / L
    n = w - u * float(w @ u)
    nl = float(np.linalg.norm(n))
    n = n / nl if nl > 1e-6 else np.array([0.0, 1.0, 0.0])
    return hip + u * a + n * hk


def _bell_wind_ang(a, b, w, wind_vec):
    """The wind's direction about a bell's axis a->b, in _sd_bell's frame (0 = the figure's forward w, positive toward
    its left), and the wind's horizontal strength (|wind_vec|)."""
    d = _unit(np.asarray(b) - np.asarray(a))
    f = w - d * float(w @ d)
    f = _unit(f)
    s = np.cross(d, f)
    wv = np.asarray(wind_vec, np.float64)
    return math.atan2(float(wv @ s), float(wv @ f)), float(np.linalg.norm(wv[[0, 2]]))


def traveller(sc, pel, w, ank_l, ank_r, rgb, h=1.0, lean=0.10, hem=0.30, cloak=(0.215, 0.34), folds=9,
              fold_depth=0.03, fold_phase=0.0, sway=0.0, pack=False, staff_tip=None, lantern_side=0.0,
              lantern_swing=0.0, carry=None, carry_side=1.0, peak=True, free_swing=0.0, reach=None,
              wind=None, ell=None, flutter=0.0, cloth=True):
    """v3 CLOTH (cloth=True, the default): the cloak is elliptical (a body is flatter front to back), hangs with
    creased folds, and takes the wind: `wind` = a world vector, the direction the wind blows toward, its length the
    hem's billow in metres (0.1-0.3); the hem centre streams to the lee, the lee side billows and its hem lifts and
    flutters (`flutter` = a phase that runs with time), the windward side presses onto the legs. The short cape is
    cloth too, and the hood has a front brim, so a profile reads as a deep cowl, not a ball on a cone.
    cloth=False keeps the v2 figure exactly."""
    if not cloth:
        return _traveller_v2(sc, pel, w, ank_l, ank_r, rgb, h, lean, hem, cloak, folds, fold_depth, fold_phase, sway,
                             pack, staff_tip, lantern_side, lantern_swing, carry, carry_side, peak, free_swing, reach)
    up = np.array([0.0, 1.0, 0.0])
    w = _unit([w[0], 0.0, w[2]])
    s = np.cross(up, w)                              # the walker's RIGHT in this world (sd_bell's +theta is its left)
    fl = w * math.sin(lean) + up * math.cos(lean)
    chest = pel + fl * 0.47 * h
    neck = chest + fl * 0.11 * h
    head = neck + up * 0.10 * h + w * 0.035 * h
    ground_y = min(ank_l[1], ank_r[1]) - 0.07
    wv = np.zeros(3) if wind is None else np.asarray(wind, np.float64)
    wn = float(np.linalg.norm(wv))
    wdir = wv / wn if wn > 1e-9 else np.zeros(3)
    el = 0.30 if ell is None else ell
    out = {}
    sc.begin(rgb=rgb)
    # legs (only the shins and boots show below the hem; the knees press the cloth where the stride opens it)
    for side, ank in ((1.0, ank_l), (-1.0, ank_r)):
        hip = pel + s * side * 0.085 * h
        kn = _knee(hip, ank, 0.46 * h, 0.45 * h, w)
        sc.cone(hip, kn, 0.074 * h, 0.058 * h, 0, 0.03)
        sc.cone(kn, ank, 0.055 * h, 0.044 * h, 0, 0.02)
        sc.box(ank + w * 0.045 * h - up * 0.035 * h, (0.052 * h, 0.048 * h, 0.13 * h),
               yaw=math.atan2(w[0], w[2]), rnd=0.033 * h, mat=0, k=0.02)
    # the cape over sloping shoulders: a short cloth bell, lightly folded, lifting a little in the wind
    cape_a = neck - up * 0.015 * h
    cape_b = chest - up * 0.07 * h + wdir * 0.2 * wn
    ang, _ = _bell_wind_ang(cape_a, cape_b, w, wv)
    sc.bell(cape_a, cape_b, 0.085 * h, 0.215 * h, w, 0.012 * h, 7, fold_phase * 0.7 + 0.9, 0, 0.05 * h,
            ell=0.8 * el, wind=0.25 * wn, wind_ang=ang)
    # the cloak: from the chest to the hem, the hem centre streaming to the lee and swaying with the gait
    hem_c = np.array([pel[0], ground_y + hem * h, pel[2]]) - w * 0.04 * h + s * sway + wdir * 0.55 * wn
    top = chest - up * 0.03 * h
    ang, _ = _bell_wind_ang(top, hem_c, w, wv)
    sc.bell(top, hem_c, cloak[0] * h, cloak[1] * h, w, fold_depth * h * 1.25, folds, fold_phase + flutter, 0,
            0.06 * h, ell=el, wind=wn, wind_ang=ang)
    if pack:
        # a bundle on the back, under the cloak: it makes the hump
        sc.box(chest - w * 0.18 * h - up * 0.10 * h, (0.16 * h, 0.21 * h, 0.11 * h), yaw=math.atan2(w[0], w[2]),
               rnd=0.06 * h, mat=0, k=0.09 * h)
    # the hood: a cowl round the head, a front brim standing proud of the (unseen) face, a soft point at the back,
    # the cloth falling from it into the cape (no neck shows)
    sc.cone(head - w * 0.02 * h, head + up * 0.012 * h, 0.112 * h, 0.116 * h, 0, 0.06 * h)
    sc.cone(head + w * 0.075 * h + up * 0.045 * h, head + w * 0.10 * h - up * 0.06 * h, 0.05 * h, 0.042 * h, 0,
            0.07 * h)
    sc.cone(head - up * 0.05 * h - w * 0.02 * h, chest + up * 0.03 * h - w * 0.03 * h, 0.10 * h, 0.17 * h, 0,
            0.07 * h)
    if peak:
        tip = head - w * 0.16 * h + up * 0.005 * h + wdir * 0.25 * wn
        sc.cone(head - w * 0.05 * h + up * 0.05 * h, tip, 0.07 * h, 0.028 * h, 0, 0.05 * h)
    # arms: loose sleeves, gloved hands
    shL = chest + up * 0.02 * h + s * 0.19 * h
    shR = chest + up * 0.02 * h - s * 0.19 * h

    def arm(sh, hand, out_dir):
        el_ = sh + 0.5 * (hand - sh) + out_dir * 0.05 * h - up * 0.05 * h
        sc.cone(sh, el_, 0.076 * h, 0.070 * h, 0, 0.04 * h)
        sc.cone(el_, hand - (hand - el_) * 0.12, 0.070 * h, 0.086 * h, 0, 0.02 * h)
        sc.cone(hand, hand, 0.045 * h, 0.045 * h, 0, 0.015 * h)
        return hand

    if carry is not None:
        # both hands steady the poles in front of the shoulders (a two-pole litter): no arm hangs free, so
        # nothing below the cloak can read as a dangling limb
        arm(shL, carry[0], s)
        out['hand'] = arm(shR, carry[1], -s)
    else:
        sw = free_swing
        if lantern_side != 0.0:
            sl = -s if lantern_side > 0 else s
            shA, shB = (shR, shL) if lantern_side > 0 else (shL, shR)
            hand_l = shA + (w * (0.16 + 0.5 * sw) - up * 0.50 + sl * 0.07) * h
            arm(shA, hand_l, sl)
            out['lantern'] = hand_l - up * 0.21 * h + w * lantern_swing + wdir * 0.12 * wn
            out['hand'] = hand_l
        else:
            sl = -s
            shA, shB = shR, shL
            hand_a = shA + (w * (0.06 - 0.5 * sw) - up * 0.52 + sl * 0.06) * h
            arm(shA, hand_a, sl)
        if staff_tip is not None:
            hand_s = shB + (w * 0.22 - up * 0.34 - sl * 0.07) * h
            arm(shB, hand_s, -sl)
            sc.cone(hand_s + up * 0.28 * h, staff_tip, 0.017, 0.014, 2, 0.0)
        elif reach is not None:
            d = reach - shB
            d = d / (np.linalg.norm(d) + 1e-9)
            arm(shB, shB + d * 0.62 * h, -sl)
        else:
            hand_s = shB + (w * (0.08 - 0.5 * sw) - up * 0.52 - sl * 0.06) * h
            arm(shB, hand_s, -sl)
    sc.end()
    out['waist'] = pel + up * 0.02 * h - s * 0.2 * h
    out['head'] = head
    out['chest'] = chest
    out['shoulders'] = (shL, shR)
    return out


def _traveller_v2(sc, pel, w, ank_l, ank_r, rgb, h=1.0, lean=0.10, hem=0.30, cloak=(0.215, 0.34), folds=9,
                  fold_depth=0.03, fold_phase=0.0, sway=0.0, pack=False, staff_tip=None, lantern_side=0.0,
                  lantern_swing=0.0, carry=None, carry_side=1.0, peak=True, free_swing=0.0, reach=None):
    """A hooded, cloaked traveller seen at any angle (no capsule limbs showing: a wool cloak from the shoulders to
    `hem` above the ground, folds deepening toward the hem and swinging with the gait, a hood, loose sleeves, gloved
    hands, boots). The caller plants the feet: pel = pelvis, ank_l / ank_r = ankle targets (world); the legs are
    solved by two-bone IK so the stance foot stays planted where it was put.
    lantern_side: +1 right hand / -1 left hand / 0 none (a hand-lantern hangs below it, `lantern_swing` in m).
    carry: (left target, right target) on the poles; carry_side picks the hand that steadies the pole.
    staff_tip: ground point of a staff held in the free hand. reach: a world point the free hand reaches toward
    (feeding a fire). Returns dict(waist, lantern, hand, head, chest, shoulders)."""
    up = np.array([0.0, 1.0, 0.0])
    w = _unit([w[0], 0.0, w[2]])
    s = np.cross(up, w)                              # the walker's left
    fl = w * math.sin(lean) + up * math.cos(lean)
    chest = pel + fl * 0.47 * h
    neck = chest + fl * 0.11 * h
    head = neck + up * 0.10 * h + w * 0.035 * h
    ground_y = min(ank_l[1], ank_r[1]) - 0.07
    out = {}
    sc.begin(rgb=rgb)
    # legs (only the shins and boots show below the hem)
    for side, ank in ((1.0, ank_l), (-1.0, ank_r)):
        hip = pel + s * side * 0.085 * h
        kn = _knee(hip, ank, 0.46 * h, 0.45 * h, w)
        sc.cone(hip, kn, 0.080 * h, 0.062 * h, 0, 0.03)
        sc.cone(kn, ank, 0.060 * h, 0.047 * h, 0, 0.02)
        sc.box(ank + w * 0.045 * h - up * 0.035 * h, (0.055 * h, 0.05 * h, 0.13 * h),
               yaw=math.atan2(w[0], w[2]), rnd=0.035 * h, mat=0, k=0.02)
    # the cloak: sloping shoulders (a short cape), then the bell to the hem, trailing a little and swaying
    cape_top = neck - up * 0.02 * h
    sc.cone(cape_top, chest - up * 0.05 * h, 0.11 * h, 0.205 * h, 0, 0.05 * h)
    hem_c = np.array([pel[0], ground_y + hem * h, pel[2]]) - w * 0.05 * h + s * sway
    sc.bell(chest - up * 0.03 * h, hem_c, cloak[0] * h, cloak[1] * h, w, fold_depth * h, folds, fold_phase, 0,
            0.06 * h)
    if pack:
        # a bundle on the back, under the cloak: it makes the hump
        sc.box(chest - w * 0.20 * h - up * 0.10 * h, (0.17 * h, 0.22 * h, 0.12 * h), yaw=math.atan2(w[0], w[2]),
               rnd=0.06 * h, mat=0, k=0.09 * h)
    # the hood: a head-shaped cowl, a soft point at the back, gathered into the cape
    sc.cone(head - w * 0.015 * h, head + up * 0.015 * h, 0.122 * h, 0.125 * h, 0, 0.06 * h)
    if peak:
        sc.cone(head - w * 0.05 * h + up * 0.05 * h, head - w * 0.15 * h + up * 0.0 * h, 0.075 * h, 0.03 * h, 0,
                0.05 * h)
    # arms: loose sleeves, gloved hands
    shL = chest + up * 0.02 * h + s * 0.19 * h
    shR = chest + up * 0.02 * h - s * 0.19 * h

    def arm(sh, hand, out_dir):
        el = sh + 0.5 * (hand - sh) + out_dir * 0.05 * h - up * 0.05 * h
        sc.cone(sh, el, 0.080 * h, 0.072 * h, 0, 0.04 * h)
        sc.cone(el, hand - (hand - el) * 0.12, 0.072 * h, 0.088 * h, 0, 0.02 * h)
        sc.cone(hand, hand, 0.047 * h, 0.047 * h, 0, 0.015 * h)
        return hand

    if carry is not None:
        # both hands steady the poles in front of the shoulders (a two-pole litter): no arm hangs free, so
        # nothing below the cloak can read as a dangling limb
        arm(shL, carry[0], s)
        out['hand'] = arm(shR, carry[1], -s)
    else:
        sw = free_swing
        if lantern_side != 0.0:
            sl = -s if lantern_side > 0 else s
            shA, shB = (shR, shL) if lantern_side > 0 else (shL, shR)
            hand_l = shA + (w * (0.16 + 0.5 * sw) - up * 0.50 + sl * 0.07) * h
            arm(shA, hand_l, sl)
            out['lantern'] = hand_l - up * 0.21 * h + w * lantern_swing
            out['hand'] = hand_l
        else:
            sl = -s
            shA, shB = shR, shL
            hand_a = shA + (w * (0.06 - 0.5 * sw) - up * 0.52 + sl * 0.06) * h
            arm(shA, hand_a, sl)
        if staff_tip is not None:
            hand_s = shB + (w * 0.22 - up * 0.34 - sl * 0.07) * h
            arm(shB, hand_s, -sl)
            sc.cone(hand_s + up * 0.28 * h, staff_tip, 0.017, 0.014, 2, 0.0)
        elif reach is not None:
            d = reach - shB
            d = d / (np.linalg.norm(d) + 1e-9)
            arm(shB, shB + d * 0.62 * h, -sl)
        else:
            hand_s = shB + (w * (0.08 - 0.5 * sw) - up * 0.52 - sl * 0.06) * h
            arm(shB, hand_s, -sl)
    sc.end()
    out['waist'] = pel + up * 0.02 * h - s * 0.2 * h
    out['head'] = head
    out['chest'] = chest
    out['shoulders'] = (shL, shR)
    return out


SEAT_POSES = ('knees', 'cross', 'back', 'side', 'kneel', 'lie')


def seated(sc, base, w, rgb, h=1.0, pose='knees', lean=None, tilt=0.0, turn=0.0, wind=None, flutter=0.0,
           peak=True, reach=None, breath=0.0, pack=True, ground=None, fold_phase=0.0):
    """A hooded adult sitting on the snow, in cloth (v3): lean proportions (a long back, sloping shoulders, a head a
    seventh of the height), the cloak draped from the shoulders to the snow behind, the arms and elbows breaking
    its outline, so a figure seen from behind is a person resting, not a bell or a plush toy.
    base: the seat point on the snow; w: facing (horizontal); pose in SEAT_POSES:
      knees  knees drawn up, arms round them, elbows out      cross  cross-legged, hands in the lap
      back   leaning back on both hands, legs out            side   legs folded to one side, a hand planted
      kneel  sitting on the heels (reach: a hand toward it)  lie    lying back on the pack, hands behind the head
    tilt: the head toward its left (+) or right (-) (a head on a shoulder); turn: the head turned (rad, + left);
    breath: the phase of a slow breath (the shoulders rise); ground(p) -> snow height for planted hands and feet.
    Returns dict(head, chest, left, right (shoulders), w)."""
    up = np.array([0.0, 1.0, 0.0])
    w = _unit([w[0], 0.0, w[2]])
    s = np.cross(up, w)                               # the figure's RIGHT (this world's convention)
    B = np.asarray(base, np.float64)

    def gnd(p):
        if ground is None:
            return p
        q = np.array(p, np.float64)
        q[1] = ground(q) + (q[1] - B[1])
        return q

    L_ = dict(knees=0.30, cross=0.10, back=-0.38, side=0.05, kneel=0.18, lie=-1.05)
    ln = L_[pose] if lean is None else lean
    pel_h = dict(knees=0.13, cross=0.12, back=0.12, side=0.12, kneel=0.30, lie=0.13)[pose]
    if pack and pose in ('knees', 'cross', 'side'):
        pel_h += 0.10                                  # sitting on the pack
    P = B + up * pel_h * h
    br = 0.006 * h * math.sin(breath)
    tdir = w * math.sin(ln) + up * math.cos(ln)
    C = P + tdir * 0.46 * h + up * br
    N = C + tdir * 0.11 * h
    sL = -s                                            # the figure's left
    hd_dir = up * math.cos(tilt) + sL * math.sin(tilt)
    Hd = N + hd_dir * 0.095 * h + w * 0.03 * h
    if pose == 'lie':
        Hd = N + tdir * 0.09 * h + up * 0.03 * h
    hf = _unit(w * math.cos(turn) + sL * math.sin(turn))           # where the face points
    wv = np.zeros(3) if wind is None else np.asarray(wind, np.float64)
    wn = float(np.linalg.norm(wv))
    wdir = wv / wn if wn > 1e-9 else np.zeros(3)
    out = dict(head=Hd, chest=C, w=w)
    sc.begin(rgb=rgb)
    # ---- legs
    if pose == 'knees':
        K = [B + (w * 0.36 + s * sd * 0.11 + up * 0.40) * h for sd in (1.0, -1.0)]
        A = [B + (w * 0.50 + s * sd * 0.12 + up * 0.06) * h for sd in (1.0, -1.0)]
    elif pose == 'cross':
        K = [B + (w * 0.20 + s * sd * 0.30 + up * 0.10) * h for sd in (1.0, -1.0)]
        A = [B + (w * 0.30 - s * sd * 0.09 + up * 0.05) * h for sd in (1.0, -1.0)]
    elif pose == 'back':
        K = [B + (w * 0.42 + s * sd * 0.12 + up * 0.24) * h for sd in (1.0, -1.0)]
        A = [B + (w * 0.80 + s * sd * 0.14 + up * 0.05) * h for sd in (1.0, -1.0)]
    elif pose == 'side':
        K = [B + (w * 0.30 + s * 0.24 + up * 0.13) * h, B + (w * 0.24 + s * 0.10 + up * 0.17) * h]
        A = [B + (w * 0.02 + s * 0.46 + up * 0.05) * h, B + (-w * 0.02 + s * 0.34 + up * 0.05) * h]
    elif pose == 'kneel':
        K = [B + (w * 0.32 + s * sd * 0.12 + up * 0.06) * h for sd in (1.0, -1.0)]
        A = [B + (-w * 0.06 + s * sd * 0.10 + up * 0.05) * h for sd in (1.0, -1.0)]
    else:                                              # lie
        K = [B + (w * 0.36 + s * sd * 0.12 + up * 0.38) * h for sd in (1.0, -1.0)]
        A = [B + (w * 0.56 + s * sd * 0.13 + up * 0.06) * h for sd in (1.0, -1.0)]
    A = [gnd(a) for a in A]
    for sd, k_, a_ in zip((1.0, -1.0), K, A):
        hip = P + s * sd * 0.085 * h
        sc.cone(hip, k_, 0.074 * h, 0.058 * h, 0, 0.03 * h)
        sc.cone(k_, a_, 0.055 * h, 0.044 * h, 0, 0.02 * h)
        fw = _unit((a_ - k_) * np.array([1.0, 0.0, 1.0]) + w * 0.3)
        sc.box(a_ + fw * 0.05 * h - up * 0.02 * h, (0.050 * h, 0.045 * h, 0.12 * h), yaw=math.atan2(fw[0], fw[2]),
               rnd=0.032 * h, mat=0, k=0.02 * h)
    # ---- the cloak: a short cape over sloping shoulders, then the drape from the shoulders to the snow behind
    cape_a = N - up * 0.015 * h
    cape_b = C - up * 0.07 * h
    ang, _ = _bell_wind_ang(cape_a, cape_b, w, wv)
    sc.bell(cape_a, cape_b, 0.085 * h, 0.21 * h, w, 0.012 * h, 7, fold_phase * 0.7 + 0.9, 0, 0.05 * h,
            ell=0.26, wind=0.2 * wn, wind_ang=ang)
    if pose != 'lie':
        back = dict(knees=0.10, cross=0.14, back=0.30, side=0.12, kneel=0.16)[pose]
        hem_c = B - w * back * h + up * 0.03 * h + wdir * 0.5 * wn
        hem_c = gnd(hem_c)
        top = C - up * 0.03 * h
        ang, _ = _bell_wind_ang(top, hem_c, w, wv)
        rb = dict(knees=0.33, cross=0.36, back=0.30, side=0.34, kneel=0.33)[pose]
        sc.bell(top, hem_c, 0.19 * h, rb * h, w, 0.030 * h, 11, fold_phase + flutter, 0, 0.06 * h,
                ell=0.24, wind=wn, wind_ang=ang)
    else:
        # lying back: the cloak spread under and around the body, the pack under the shoulders
        sc.box(P - w * 0.25 * h + up * 0.02 * h, (0.26 * h, 0.035 * h, 0.40 * h), yaw=math.atan2(w[0], w[2]),
               rnd=0.03 * h, mat=0, k=0.05 * h)
        sc.cone(C - up * 0.02 * h, P, 0.17 * h, 0.16 * h, 0, 0.06 * h)
    # ---- the hood: a cowl round the head, a front brim, a drape to the shoulders (no neck shows)
    sc.cone(Hd - hf * 0.02 * h, Hd + up * 0.012 * h, 0.108 * h, 0.112 * h, 0, 0.06 * h)
    sc.cone(Hd + hf * 0.072 * h + up * 0.045 * h, Hd + hf * 0.095 * h - up * 0.055 * h, 0.048 * h, 0.040 * h, 0,
            0.07 * h)
    sc.cone(Hd - up * 0.05 * h - hf * 0.02 * h, C + up * 0.03 * h - w * 0.03 * h, 0.098 * h, 0.165 * h, 0, 0.07 * h)
    if peak:
        sc.cone(Hd - hf * 0.05 * h + up * 0.05 * h, Hd - hf * 0.15 * h - up * 0.01 * h + wdir * 0.2 * wn,
                0.066 * h, 0.026 * h, 0, 0.05 * h)
    # ---- arms: loose sleeves, gloved hands
    S = [C + up * 0.02 * h + s * sd * 0.18 * h for sd in (1.0, -1.0)]

    def arm(sh, hand, elbow_out):
        el_ = 0.5 * (sh + hand) + elbow_out
        sc.cone(sh, el_, 0.074 * h, 0.068 * h, 0, 0.04 * h)
        sc.cone(el_, hand - (hand - el_) * 0.12, 0.068 * h, 0.082 * h, 0, 0.02 * h)
        sc.cone(hand, hand, 0.043 * h, 0.043 * h, 0, 0.015 * h)

    if pose == 'knees':
        km = 0.5 * (K[0] + K[1])
        for sd, sh in zip((1.0, -1.0), S):
            arm(sh, km + (w * 0.07 - up * 0.07 + s * sd * 0.035) * h, (s * sd * 0.11 - up * 0.03) * h)
    elif pose == 'cross':
        for sd, sh in zip((1.0, -1.0), S):
            arm(sh, P + (w * 0.22 + s * sd * 0.07 + up * 0.10) * h, (s * sd * 0.08 - w * 0.02) * h)
    elif pose == 'back':
        for sd, sh in zip((1.0, -1.0), S):
            arm(sh, gnd(B + (-w * 0.30 + s * sd * 0.22 + up * 0.04) * h), (s * sd * 0.04) * h)
    elif pose == 'side':
        arm(S[1], gnd(B + (-s * 0.30 + w * 0.02 + up * 0.04) * h), (-s * 0.05) * h)
        arm(S[0], K[0] + up * 0.06 * h, (s * 0.06) * h)
    elif pose == 'kneel':
        if reach is not None:
            d = _unit(np.asarray(reach) - S[0])
            arm(S[0], S[0] + d * 0.60 * h, (s * 0.03 - up * 0.04) * h)
        else:
            arm(S[0], K[0] + (up * 0.07 - w * 0.02) * h, (s * 0.06) * h)
        arm(S[1], K[1] + (up * 0.07 - w * 0.02) * h, (-s * 0.06) * h)
    else:                                              # lie: hands behind the head, elbows out
        for sd, sh in zip((1.0, -1.0), S):
            arm(sh, Hd + (-tdir * 0.05 + s * sd * 0.07) * h, (s * sd * 0.20 + up * 0.04) * h)
    sc.end()
    if pack and pose in ('back', 'kneel'):
        # the pack set down beside them
        sc.begin(rgb=tuple(0.8 * c for c in rgb))
        pk = gnd(B + (-w * 0.05 + s * 0.42 + up * 0.12) * h)
        sc.box(pk, (0.15 * h, 0.12 * h, 0.11 * h), yaw=math.atan2(w[0], w[2]) + 0.5, rnd=0.05 * h, mat=0, k=0.0)
        sc.end()
    out['right'] = S[0]
    out['left'] = S[1]
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
