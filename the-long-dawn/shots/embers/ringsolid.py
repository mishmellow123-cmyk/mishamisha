"""EMBERS-C: THE RING as a SOLID polished band (cut C). Our own design, matched to the canonical Ring.

Why: the splat Ring (tolkien.Ring) was a thin additive hoop, R 5 x width 1: it read as a halo round the fire and a
ring-toss hoop over the towers, and its back showed through its front. This one is a surface:

* canonical proportions (shots/montage3d/ringc.py: R_IN 9.4, THICK 2.3, WIDTH 5.2 mm; superellipse section SQ 2.8),
  in units of the band WIDTH; `scale` sets the width in world units;
* the canonical inscription (assets/ring/inscription_{outer,inner}.png, 16-bit coverage; mapping in
  assets/ring/inscription.json: outer u = angle/360 counter-clockwise seen from +axis, inner u = 1 - that, row 0 =
  the +axis edge), as emission only: the metal is plain when cold, the letters exist in fire (deep orange-red core,
  never white);
* rasterised as a parametric mesh with a z-buffer (numba, supersampled), shaded per sample: polished gold (Schlick
  Fresnel, F0 of gold) reflecting an environment of the burning world (spherical-Gaussian lobes: the fire, the tower
  throats, the smoke canopy, a ground glow), plus the heat of the forge (white -> yellow -> gold) and the letters;
* motion-blurred by averaging sub-frames across the shutter;
* it returns premultiplied HDR colour, coverage and depth at the frame's resolution, so the caller can merge it into
  the splat frame's occluder (everything behind it is hidden) and add it after resolve.

Ring-local frame: axis = +Y, centre at the origin; the world placement is C + scale * Rot @ p.
"""
import math
import os

import cv2
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

# canonical band, in units of its width
WIDTH = 1.0
THICK = 2.3 / 5.2
R_IN = 9.4 / 5.2
R_MID = R_IN + 0.5 * THICK
R_OUT = R_IN + THICK
SQ = 2.8
F0_GOLD = np.array([1.0, 0.766, 0.336])          # gold's specular colour (linear)
C_LETTER = np.array([1.0, 0.3, 0.045])            # the letters in fire: deep orange-red, never white


def sgnpow(x, p):
    return np.sign(x) * np.abs(x) ** p


def section(psi, sec=(1.0, 1.0)):
    """superellipse section: radial offset r, axial y, and the unit normal (nr, ny) in the (r, y) plane.
    sec = (radial, axial) scale of the section (the forging beats a thin thread out into the full band)"""
    c, s = np.cos(psi), np.sin(psi)
    a, b = 0.5 * THICK * sec[0], 0.5 * WIDTH * sec[1]
    r = a * sgnpow(c, 2.0 / SQ)
    y = b * sgnpow(s, 2.0 / SQ)
    nr = np.sign(r) * np.abs(r / a) ** (SQ - 1.0) / a
    ny = np.sign(y) * np.abs(y / b) ** (SQ - 1.0) / b
    ln = np.sqrt(nr * nr + ny * ny) + 1e-12
    return r, y, nr / ln, ny / ln


def local_points(theta, psi, sec=(1.0, 1.0)):
    """ring-local positions and normals for parameters (theta round the axis, psi round the section)"""
    r, y, nr, ny = section(psi, sec)
    c, s = np.cos(theta), np.sin(theta)
    rad = R_MID + r
    P = np.stack([rad * c, y, rad * s], -1)
    N = np.stack([nr * c, ny, nr * s], -1)
    return P, N


# =============================================================== inscription ===

class Inscription:
    """the canonical strips as mip pyramids (float32 coverage 0..1)"""

    def __init__(self):
        self.lv = {}
        for face in ('outer', 'inner'):
            im = cv2.imread(os.path.join(ROOT, 'assets', 'ring', f'inscription_{face}.png'), cv2.IMREAD_UNCHANGED)
            m = im.astype(np.float32) / 65535.0
            if m.ndim == 3:
                m = m[..., 0]
            lv = [m]
            while lv[-1].shape[0] > 8:
                p = lv[-1]
                lv.append(cv2.resize(p, (max(p.shape[1] // 2, 1), max(p.shape[0] // 2, 1)),
                                     interpolation=cv2.INTER_AREA))
            self.lv[face] = lv

    def sample(self, face, u, v, texels_per_sample):
        """bilinear lookup at (u, v) in [0,1) (u wraps) from the mip level matching the footprint"""
        lv = self.lv[face]
        out = np.zeros(len(u), np.float32)
        L = np.clip(np.floor(np.log2(np.maximum(texels_per_sample, 1.0))), 0, len(lv) - 1).astype(int)
        for l in np.unique(L):
            m = L == l
            img = lv[l]
            H, W = img.shape
            x = (u[m] % 1.0) * W - 0.5
            y = np.clip(v[m], 0.0, 1.0) * (H - 1)
            x0 = np.floor(x).astype(int)
            y0 = np.clip(np.floor(y).astype(int), 0, H - 2)
            fx = (x - x0).astype(np.float32)
            fy = (y - y0).astype(np.float32)
            x0 %= W
            x1 = (x0 + 1) % W
            a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
            b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x1] * fx
            out[m] = a * (1 - fy) + b * fy
        return out


_INS = [None]


def inscription():
    if _INS[0] is None:
        _INS[0] = Inscription()
    return _INS[0]


# ================================================================ rasteriser ===

@njit(cache=True, fastmath=True)
def _raster(X, Y, Z, TH, PS, ZB, TB, PB):
    """z-buffered, perspective-correct rasterisation of the (nt+1) x (np+1) parametric grid (two triangles per quad).
    X, Y: screen coords in buffer pixels (pixel centres at integers); Z: camera depth; TH, PS: parameters."""
    H = ZB.shape[0]
    W = ZB.shape[1]
    nt = X.shape[0] - 1
    npp = X.shape[1] - 1
    for i in range(nt):
        for j in range(npp):
            for tri in range(2):
                if tri == 0:
                    ia, ja, ib, jb, ic, jc = i, j, i + 1, j, i + 1, j + 1
                else:
                    ia, ja, ib, jb, ic, jc = i, j, i + 1, j + 1, i, j + 1
                za = Z[ia, ja]
                zb = Z[ib, jb]
                zc = Z[ic, jc]
                if za <= 0.0 or zb <= 0.0 or zc <= 0.0:
                    continue
                ax = X[ia, ja]
                ay = Y[ia, ja]
                bx = X[ib, jb]
                by = Y[ib, jb]
                cx = X[ic, jc]
                cy = Y[ic, jc]
                area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                if abs(area) < 1e-12:
                    continue
                x0 = int(math.floor(min(ax, min(bx, cx))))
                x1 = int(math.ceil(max(ax, max(bx, cx))))
                y0 = int(math.floor(min(ay, min(by, cy))))
                y1 = int(math.ceil(max(ay, max(by, cy))))
                if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                    continue
                if x0 < 0:
                    x0 = 0
                if y0 < 0:
                    y0 = 0
                if x1 > W - 1:
                    x1 = W - 1
                if y1 > H - 1:
                    y1 = H - 1
                iza = 1.0 / za
                izb = 1.0 / zb
                izc = 1.0 / zc
                inv = 1.0 / area
                ta = TH[ia, ja]
                tb = TH[ib, jb]
                tc = TH[ic, jc]
                pa = PS[ia, ja]
                pb = PS[ib, jb]
                pc = PS[ic, jc]
                for yy in range(y0, y1 + 1):
                    for xx in range(x0, x1 + 1):
                        w0 = ((bx - xx) * (cy - yy) - (by - yy) * (cx - xx)) * inv
                        w1 = ((cx - xx) * (ay - yy) - (cy - yy) * (ax - xx)) * inv
                        w2 = 1.0 - w0 - w1
                        if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                            continue
                        iz = w0 * iza + w1 * izb + w2 * izc
                        z = 1.0 / iz
                        if z < ZB[yy, xx]:
                            ZB[yy, xx] = z
                            TB[yy, xx] = (w0 * ta * iza + w1 * tb * izb + w2 * tc * izc) * z
                            PB[yy, xx] = (w0 * pa * iza + w1 * pb * izb + w2 * pc * izc) * z


# =============================================================== environment ===

class Env:
    """the burning world as seen from the Ring: a vertical gradient plus spherical-Gaussian lobes
    (directions fixed in the world) and point lights (their direction taken per sample)."""

    def __init__(self, above=(0.004, 0.003, 0.002), horizon=(0.06, 0.028, 0.012), below=(0.03, 0.012, 0.005)):
        self.above = np.asarray(above, np.float64)
        self.horizon = np.asarray(horizon, np.float64)
        self.below = np.asarray(below, np.float64)
        self.lobes = []         # (dir (3,), col (3,), sharpness)
        self.points = []        # (pos (3,), col (3,), sharpness at 1 unit, radius)

    def lobe(self, d, col, sharp):
        d = np.asarray(d, np.float64)
        self.lobes.append((d / max(np.linalg.norm(d), 1e-9), np.asarray(col, np.float64), float(sharp)))
        return self

    def point(self, pos, col, radius):
        """a glowing body at pos of angular size ~ radius / distance (a soft disc in the reflection)"""
        self.points.append((np.asarray(pos, np.float64), np.asarray(col, np.float64), float(radius)))
        return self

    def radiance(self, R, P):
        ry = R[:, 1]
        up = np.clip(ry, 0, 1)[:, None]
        dn = np.clip(-ry, 0, 1)[:, None]
        hz = (1.0 - np.abs(ry))[:, None] ** 3
        L = self.above[None, :] * up + self.below[None, :] * dn + self.horizon[None, :] * hz
        for d, col, sh in self.lobes:
            L = L + col[None, :] * np.exp(sh * (R @ d - 1.0))[:, None]
        for pos, col, rad in self.points:
            D = pos[None, :] - P
            dist = np.linalg.norm(D, axis=1)
            D = D / np.maximum(dist, 1e-9)[:, None]
            ang = np.maximum(rad / np.maximum(dist, 1e-6), 0.02)          # the body's angular radius
            sh = 1.0 / (ang * ang)
            L = L + col[None, :] * np.exp(sh * (np.sum(R * D, 1) - 1.0))[:, None]
        return L


# ==================================================================== render ===

class RingState:
    """what the band is doing: set by the caller per frame (arrays are functions of theta, the angle round the band)"""

    def __init__(self):
        self.heat = None            # fn(theta) -> 0..1 (1 white-hot, 0 cold gold)
        self.write = None           # fn(theta) -> 0..1: the letters are awake (burning up out of the metal)
        self.letters = 0.0          # the letters' brightness
        self.letters_col = C_LETTER
        self.glow = 0.0             # the band's own faint ember-light (so it never goes flat black)
        self.glow_col = np.array([1.0, 0.62, 0.22])
        self.exposure = 1.0         # multiplies the reflection
        self.alpha = 1.0            # overall opacity (fade in/out)
        self.sec = (1.0, 1.0)       # section scale (radial, axial): the forging beats it out to (1, 1)
        self.hammer = 0.0           # hammer marks in the hot metal (0..1)


def _frame_samples(cam, W, H, Rot, C, scale, nt, npp, th_range=None, sec=(1.0, 1.0)):
    th0, th1 = (0.0, 2.0 * np.pi) if th_range is None else th_range
    th = np.linspace(th0, th1, nt + 1)
    ps = np.linspace(-np.pi, np.pi, npp + 1)
    T, Pp = np.meshgrid(th, ps, indexing='ij')
    Pl, _ = local_points(T, Pp, sec)
    Pw = C[None, None, :] + scale * np.einsum('ij,abj->abi', Rot, Pl)
    d = (Pw - cam.pos) @ cam.R.T
    return T, Pp, d


def render(cam, W, H, Rot, C, scale, st, env, ss=None, nt=None, npp=None, pad=4, th_range=None):
    """one instant: -> rgb (H, W, 3) premultiplied HDR, alpha (H, W), depth (H, W) (inf where empty).
    th_range=(th0, th1): only that arc of the band exists (the forging's white-hot front runs round)"""
    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    depth = np.full((H, W), np.inf, np.float32)
    f = cam.f_px(W)
    # screen size of the ring: choose tessellation and supersampling
    dc = (np.asarray(C) - cam.pos) @ cam.R.T
    if dc[2] <= 1e-3:
        return rgb, alpha, depth
    rpx = f * R_OUT * scale / max(dc[2], 1e-3)
    if ss is None:
        ss = 3 if rpx < 250 else 2
    if nt is None:
        nt = int(np.clip(rpx * 2.2, 96, 1440))
    if th_range is not None:
        nt = max(int(nt * (th_range[1] - th_range[0]) / (2 * np.pi)), 4)
    if npp is None:
        npp = int(np.clip(rpx * 0.25, 24, 96))
    T, Pp, d = _frame_samples(cam, W, H, Rot, C, scale, nt, npp, th_range, st.sec)
    z = d[..., 2]
    if (z <= 0.05).all():
        return rgb, alpha, depth
    zc = np.maximum(z, 1e-6)
    u = (W - 1) / 2.0 + f * d[..., 0] / zc
    v = (H - 1) / 2.0 - f * d[..., 1] / zc
    ok = z > 0.05
    ux0 = int(max(math.floor(u[ok].min()) - pad, 0))
    ux1 = int(min(math.ceil(u[ok].max()) + pad, W - 1))
    vy0 = int(max(math.floor(v[ok].min()) - pad, 0))
    vy1 = int(min(math.ceil(v[ok].max()) + pad, H - 1))
    if ux1 <= ux0 or vy1 <= vy0:
        return rgb, alpha, depth
    bw, bh = (ux1 - ux0 + 1) * ss, (vy1 - vy0 + 1) * ss
    X = (u - ux0 + 0.5) * ss - 0.5
    Y = (v - vy0 + 0.5) * ss - 0.5
    ZB = np.full((bh, bw), np.inf, np.float64)
    TB = np.zeros((bh, bw), np.float64)
    PB = np.zeros((bh, bw), np.float64)
    _raster(np.ascontiguousarray(X), np.ascontiguousarray(Y), np.ascontiguousarray(np.where(ok, z, -1.0)),
            np.ascontiguousarray(T), np.ascontiguousarray(Pp), ZB, TB, PB)
    cov = np.isfinite(ZB)
    if not cov.any():
        return rgb, alpha, depth
    buf = np.zeros((bh, bw, 3), np.float32)
    iy, ix = np.nonzero(cov)
    CH = 150000                                   # shade in chunks: this Mac has 8 GB
    for k0 in range(0, len(iy), CH):
        sl = slice(k0, k0 + CH)
        th = TB[iy[sl], ix[sl]]
        ps = PB[iy[sl], ix[sl]]
        buf[iy[sl], ix[sl]] = shade(th, ps, cam, f, ss, Rot, C, scale, st, env)
    cv_ = cov.astype(np.float32)
    zbuf = np.where(cov, ZB, np.inf).astype(np.float32)
    h0, w0 = bh // ss, bw // ss
    buf = buf[:h0 * ss, :w0 * ss].reshape(h0, ss, w0, ss, 3).mean(axis=(1, 3))
    cv_ = cv_[:h0 * ss, :w0 * ss].reshape(h0, ss, w0, ss).mean(axis=(1, 3))
    zb = zbuf[:h0 * ss, :w0 * ss].reshape(h0, ss, w0, ss).min(axis=(1, 3))
    a = st.alpha
    rgb[vy0:vy0 + h0, ux0:ux0 + w0] = buf * a
    alpha[vy0:vy0 + h0, ux0:ux0 + w0] = cv_ * a
    depth[vy0:vy0 + h0, ux0:ux0 + w0] = zb
    return rgb, alpha, depth


def shade(th, ps, cam, f, ss, Rot, C, scale, st, env):
    """HDR colour of band samples (theta, psi): polished gold reflecting env, the forge's heat, the letters"""
    Pl, Nl = local_points(th, ps, st.sec)
    Pw = C[None, :] + scale * Pl @ Rot.T
    Nw = Nl @ Rot.T
    V = cam.pos[None, :] - Pw
    V /= np.maximum(np.linalg.norm(V, axis=1), 1e-9)[:, None]
    ndv = np.clip(np.sum(Nw * V, 1), 0.0, 1.0)
    Rv = 2.0 * ndv[:, None] * Nw - V
    Fr = F0_GOLD[None, :] + (1.0 - F0_GOLD[None, :]) * ((1.0 - ndv) ** 5)[:, None]
    col = Fr * env.radiance(Rv, Pw) * st.exposure
    # its own faint ember-light, stronger toward grazing (a warm skin, never a flat black)
    if st.glow > 0:
        col = col + st.glow_col[None, :] * (st.glow * (0.35 + 0.65 * (1.0 - ndv) ** 2))[:, None]
    # the forge's heat: white -> yellow -> orange -> dull red -> cold gold, emissive; hammer marks in the hot metal
    if st.heat is not None:
        hk = np.clip(st.heat(th), 0.0, 1.0)
        if hk.max() > 0:
            if st.hammer > 0:
                from core import snoise
                q = np.stack([np.cos(th) * 9.0, np.sin(th) * 9.0, ps * 1.3], 1)
                hk = np.clip(hk * (1.0 - st.hammer * 0.28 * (0.5 + 0.5 * snoise(q, 1.0, (3.0, 1.0, 7.0), 2))), 0, 1)
            hot = (np.array([1.0, 0.96, 0.86])[None, :] * _ss(0.72, 1.0, hk)[:, None]
                   + np.array([1.0, 0.78, 0.34])[None, :] * (_ss(0.42, 0.72, hk) * (1 - _ss(0.72, 1.0, hk)))[:, None]
                   + np.array([1.0, 0.42, 0.08])[None, :] * (_ss(0.18, 0.42, hk) * (1 - _ss(0.42, 0.72, hk)))[:, None]
                   + np.array([0.55, 0.1, 0.02])[None, :] * (_ss(0.0, 0.18, hk) * (1 - _ss(0.18, 0.42, hk)))[:, None])
            col = col * (1.0 - 0.8 * _ss(0.15, 0.6, hk))[:, None] + hot * (hk ** 2.2 * 5.5 + 0.15 * hk)[:, None]
    # the letters: emission through the canonical strips
    if st.letters > 0:
        outer = np.cos(ps) >= 0.0
        vv = (0.5 * WIDTH - scale_free_y(ps)) / WIDTH
        uo = (-th / (2 * np.pi)) % 1.0
        ui = (th / (2 * np.pi)) % 1.0
        # footprint: world length of one sample, in texels of the strip
        zs = np.maximum((Pw - cam.pos) @ cam.R[2], 1e-6)
        wlen = zs / (f * ss)
        ins = inscription()
        tex = np.zeros(len(th), np.float32)
        for face, m, uu, Rf in (('outer', outer, uo, R_OUT), ('inner', ~outer, ui, R_IN)):
            if m.any():
                Wt = ins.lv[face][0].shape[1]
                tps = wlen[m] / (2 * np.pi * Rf * scale) * Wt / np.maximum(ndv[m], 0.25)
                tex[m] = ins.sample(face, uu[m], vv[m], tps)
        wr = st.write(th) if st.write is not None else np.ones(len(th))
        col = col + st.letters_col[None, :] * (st.letters * tex * wr)[:, None]
    return col.astype(np.float32)


def scale_free_y(ps):
    """the axial coordinate (units of width) of section parameter psi"""
    return 0.5 * WIDTH * sgnpow(np.sin(ps), 2.0 / SQ)


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def render_blur(cams, W, H, frames, st_fn, env_fn, **kw):
    """motion blur: average instants. cams: list of cameras; frames: list of (Rot, C, scale) at the same instants;
    st_fn(k) / env_fn(k) give the state and environment of instant k."""
    acc = None
    for k, (cam, (Rot, C, sc)) in enumerate(zip(cams, frames)):
        r, a, d = render(cam, W, H, Rot, C, sc, st_fn(k), env_fn(k), **kw)
        if acc is None:
            acc = [r, a, d]
        else:
            acc[0] += r
            acc[1] += a
            acc[2] = np.minimum(acc[2], d)
    n = float(len(cams))
    return acc[0] / n, acc[1] / n, acc[2]


# ============================================================ into the splats ===

def merge_occluder(fr, alpha, depth, thr=0.35):
    """join the Ring to the splat frame's occluder pyramid (fr.occ; created if absent), so the splats behind it are
    hidden. Returns the occluder as it was before (for the Ring's own visibility against the towers)."""
    H1 = -(-fr.H // 2)
    W1 = -(-fr.W // 2)
    a2 = cv2.resize(alpha, (W1, H1), interpolation=cv2.INTER_AREA)
    d = np.where(alpha > thr, depth, np.inf).astype(np.float32)
    hp, wp = H1 * 2, W1 * 2
    dp = np.full((hp, wp), np.inf, np.float32)
    dp[:d.shape[0], :d.shape[1]] = d
    d2 = dp.reshape(H1, 2, W1, 2).min(axis=(1, 3))
    before = None
    if fr.occ is None:
        Zs = [np.full((H1, W1), 1e30, np.float32)]
        As = [np.zeros((H1, W1), np.float32)]
        Is = [np.full((H1, W1), -1, np.int32)]
        for _ in range(2):
            h2, w2 = -(-Zs[-1].shape[0] // 2), -(-Zs[-1].shape[1] // 2)
            Zs.append(np.full((h2, w2), 1e30, np.float32))
            As.append(np.zeros((h2, w2), np.float32))
            Is.append(np.full((h2, w2), -1, np.int32))
        fr.occ = (Zs, As, Is)
    else:
        Zs, As, Is = fr.occ
        before = ([z.copy() for z in Zs], [a.copy() for a in As])
    Zs, As, Is = fr.occ
    z0 = np.where(np.isfinite(d2), d2, 1e30).astype(np.float32)
    near = z0 < Zs[0]
    As[0][...] = np.where(near, np.maximum(As[0], np.clip(a2 * 1.02, 0, 0.995)), As[0])
    Zs[0][...] = np.where(near, z0, Zs[0])
    Is[0][...] = np.where(near, 777, Is[0])
    for l in range(1, 3):
        z, a = Zs[l - 1], As[l - 1]
        h2, w2 = Zs[l].shape
        zp = np.full((h2 * 2, w2 * 2), 1e30, np.float32)
        ap = np.zeros((h2 * 2, w2 * 2), np.float32)
        zp[:z.shape[0], :z.shape[1]] = z
        ap[:a.shape[0], :a.shape[1]] = a
        Zs[l][...] = zp.reshape(h2, 2, w2, 2).min(axis=(1, 3))
        As[l][...] = ap.reshape(h2, 2, w2, 2).mean(axis=(1, 3))
    return before


def visibility(before, depth, H, W):
    """1 where the Ring is in front of the towers' occluder, (1 - their alpha) where it is behind them"""
    if before is None:
        return np.ones((H, W), np.float32)
    Z, A = before[0][0], before[1][0]
    Zf = cv2.resize(Z, (Z.shape[1] * 2, Z.shape[0] * 2), interpolation=cv2.INTER_NEAREST)[:H, :W]
    Af = cv2.resize(A, (A.shape[1] * 2, A.shape[0] * 2), interpolation=cv2.INTER_LINEAR)[:H, :W]
    return np.where(depth <= Zf + 0.3, 1.0, 1.0 - Af).astype(np.float32)
