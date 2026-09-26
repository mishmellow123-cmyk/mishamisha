"""The drawing: every ink mark on the map as vector data in map coordinates.

* coasts, lakes, rivers (polylines, hand-wobbled, pressure-varied)
* glyphs in painter's order (north first, so nearer marks overlap farther ones), each a paper
  fill (occluder) + pen strokes: mountains (profile peaks, shaded east face), hills (hatched
  arcs), trees (round-crowned and conifers), and desert stipple
* the compass rose and the border are built in bake.py (they are frame furniture)

Everything is seeded and cached in renders/map_C/cache/features.npz.
"""
import math
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

import geo
import ink
from noise import wobble1d
from rivers import RIVERS, SIZE as RIVER_SIZE

FPPD = 8.0           # resolution of the map-space placement rasters (px per map degree)


def smoothstep(x, a, b):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------ map fields ---

def map_fields():
    """Placement rasters in map space (FPPD px per degree), from the equirect fields."""
    f = geo.fields()
    W = int(360 * FPPD)
    H = int(np.ceil((geo.MAP_Y1 - geo.MAP_Y0) * FPPD))
    X = geo.MAP_X0 + (np.arange(W) + 0.5) / FPPD
    Y = geo.MAP_Y1 - (np.arange(H) + 0.5) / FPPD
    lon = geo.x2lon(X)
    lat = geo.ilat(Y)
    ex = ((lon + 180.0) / 360.0 * geo.EQ_W - 0.5).astype(np.float32)
    ey = ((90.0 - lat) / 180.0 * geo.EQ_H - 0.5).astype(np.float32)
    mx, my = np.meshgrid(ex, ey)
    out = {}
    for k, v in f.items():
        out[k] = cv2.remap(v, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    out['lat'] = np.repeat(lat[:, None], W, 1).astype(np.float32)
    out['lon'] = np.repeat(lon[None, :], H, 0).astype(np.float32)
    # precise land mask in map space
    m = np.zeros((H, W), np.uint8)
    for ll, hole in geo.land_rings():
        xy = geo.to_map_ring(ll)
        for c in geo.wrap_copies(xy):
            pts = np.stack([(c[:, 0] - geo.MAP_X0) * FPPD, (geo.MAP_Y1 - c[:, 1]) * FPPD], 1)
            cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 0 if hole else 255, cv2.LINE_AA, shift=4)
    out['landm'] = m.astype(np.float32) / 255.0
    return out, W, H


def _greenland(lat, lon):
    return (lat > 59.5) & (lon > -74) & (lon < -10) & ~((lon > -26) & (lat < 67))


# --------------------------------------------------------------- placing ---

def _candidates(field, thresh, rng, n, W, H):
    """Random candidate positions (map XY) where field > thresh, weighted by the field."""
    w = np.clip(field - thresh, 0, None).ravel()
    if w.sum() <= 0:
        return np.zeros((0, 2)), np.zeros(0)
    idx = rng.choice(len(w), size=n, p=w / w.sum())
    yy, xx = np.divmod(idx, W)
    X = geo.MAP_X0 + (xx + rng.random(n)) / FPPD
    Y = geo.MAP_Y1 - (yy + rng.random(n)) / FPPD
    return np.stack([X, Y], 1), field.ravel()[idx]


def _sample(field, XY):
    W = field.shape[1]
    x = ((XY[:, 0] - geo.MAP_X0) * FPPD - 0.5).astype(np.float32)
    y = ((geo.MAP_Y1 - XY[:, 1]) * FPPD - 0.5).astype(np.float32)
    return cv2.remap(field, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).ravel()


class Placer:
    """Greedy anisotropic Poisson placement (rows of glyphs may overlap vertically)."""

    def __init__(self, kx, ky):
        self.kx, self.ky = kx, ky
        self.pts = []
        self.size = []
        self.cell = 2.0
        self.grid = {}

    def ok(self, x, y, s, others=()):
        c = self.cell
        gx, gy = int(math.floor(x / c)), int(math.floor(y / c))
        for ix in range(gx - 2, gx + 3):
            for iy in range(gy - 2, gy + 3):
                for (px, py, ps, kx, ky) in self.grid.get((ix, iy), ()):
                    m = 0.5 * (s + ps)
                    dx = (x - px) / (m * max(kx, self.kx))
                    dy = (y - py) / (m * max(ky, self.ky))
                    if dx * dx + dy * dy < 1.0:
                        return False
        for o in others:
            if not o.ok(x, y, s):
                return False
        return True

    def add(self, x, y, s):
        c = self.cell
        key = (int(math.floor(x / c)), int(math.floor(y / c)))
        self.grid.setdefault(key, []).append((x, y, s, self.kx, self.ky))
        self.pts.append((x, y))
        self.size.append(s)


# ---------------------------------------------------------------- glyphs ---

def _wob(p, seed, amp, wl):
    """Perturb a polyline perpendicular to itself (hand tremor)."""
    if len(p) < 2:
        return p
    s = ink.arclen(p)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None] + 1e-12
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    return p + nrm * wobble1d(s, seed, wl, amp)[:, None]


def _press(n, seed, lo=0.65, hi=1.1, taper=0.25):
    """Pen pressure along a stroke: tapered ends, slow variation."""
    u = np.linspace(0, 1, n)
    rng = np.random.default_rng(seed)
    base = lo + (hi - lo) * (0.5 + 0.5 * np.sin(2 * np.pi * (u * rng.uniform(0.6, 1.4) + rng.random())))
    tp = np.clip(np.minimum(u, 1 - u) / max(taper, 1e-6), 0, 1) ** 0.6
    return base * (0.35 + 0.65 * tp)


def glyph_mountain(X, Y, s, M, rng):
    """A peak in profile (sharp, twin or blunt): outline, a ridge, and fall-line hatching on the
    east (shadow) side."""
    w = s
    h = s * rng.uniform(0.6, 0.8) * (0.9 + 0.35 * M)
    px = rng.uniform(-0.13, 0.13) * w
    lb = np.array([-0.5 * w + rng.uniform(-0.05, 0.05) * w, 0.0])
    rb = np.array([0.5 * w + rng.uniform(-0.05, 0.05) * w, 0.0])
    r = rng.random()
    if r < 0.55:                                   # a single sharp summit
        ctrl = [lb, (px, h), rb]
    elif r < 0.82:                                 # twin summits with a notch
        side = 1 if rng.random() < 0.6 else -1
        p2 = px + side * rng.uniform(0.2, 0.28) * w
        h2 = h * rng.uniform(0.62, 0.8)
        notch = 0.5 * (px + p2)
        ctrl = [lb, (px, h), (notch, h2 * rng.uniform(0.72, 0.85)), (p2, h2), rb] if side > 0 else \
               [lb, (p2, h2), (notch, h2 * rng.uniform(0.72, 0.85)), (px, h), rb]
    else:                                          # a blunt, weathered top
        c = rng.uniform(0.07, 0.12) * w
        ctrl = [lb, (px - c, h * 0.9), (px, h), (px + c * 0.8, h * 0.92), rb]
    ctrl = np.array(ctrl, np.float64)
    pts = [ctrl[0]]
    cav = rng.uniform(0.05, 0.12)
    for k in range(len(ctrl) - 1):
        p0, p1 = ctrl[k], ctrl[k + 1]
        n = max(3, int(round(6 * np.linalg.norm(p1 - p0) / (0.5 * w))))
        t = np.linspace(0, 1, n + 1)[1:]
        seg = p0[None, :] + (p1 - p0)[None, :] * t[:, None]
        if k == 0:
            seg[:, 1] -= cav * h * np.sin(np.pi * t) * (1 - t) * 1.6
        if k == len(ctrl) - 2:
            seg[:, 1] -= cav * h * np.sin(np.pi * t) * t * 1.6
        pts.extend(seg)
    out = np.array(pts)
    out[1:-1] += rng.normal(0, 0.011 * h, (len(out) - 2, 2))
    if r >= 0.82:
        out = ink.chaikin(out, 2)
    else:
        out = ink.chaikin(out, 1)
    out = _wob(out, int(rng.integers(1 << 30)), 0.008 * s, 0.35 * s)
    O = np.array([X, Y])
    fill = np.concatenate([out, [[out[-1, 0], -0.03 * h], [out[0, 0], -0.03 * h]]]) + O
    st = ink.Strokes()
    r0 = 0.013 + 0.009 * s
    st.add(out + O, r0 * _press(len(out), int(rng.integers(1 << 30)), 0.75, 1.1, 0.12), 0.95)
    # the main summit and its ridge down the face
    im = int(np.argmax(out[:, 1]))
    top = out[im]
    rl = rng.uniform(0.35, 0.62)
    rx = top[0] + rng.uniform(0.04, 0.16) * w
    ridge = np.stack([np.linspace(top[0], rx, 5), np.linspace(top[1] - 0.01 * h, top[1] * (1 - rl), 5)], 1)
    ridge[1:-1, 0] += rng.normal(0, 0.02 * w, 3)
    ridge = _wob(ink.chaikin(ridge, 1), int(rng.integers(1 << 30)), 0.006 * s, 0.3 * s)
    st.add(ridge + O, r0 * 0.7 * _press(len(ridge), int(rng.integers(1 << 30)), 0.7, 1.0, 0.4), 0.9)
    # hatching: fall lines from the east-facing (descending) stretches of the skyline
    d = np.diff(out, axis=0)
    desc = (d[:, 1] < -0.15 * np.abs(d[:, 0])) & (d[:, 0] > 0)
    segl = np.where(desc, np.linalg.norm(d, axis=1), 0.0)
    segl[:max(im - 1, 0)] *= 0.35                      # mostly the main east face
    if segl.sum() > 0:
        nh = int(rng.integers(4, 8) + (2 if s > 1.5 else 0))
        cs = np.cumsum(segl)
        picks = np.sort(rng.uniform(0.04, 0.9, nh)) * cs[-1]
        for pv in picks:
            k = int(np.searchsorted(cs, pv))
            k = min(k, len(d) - 1)
            f = (pv - (cs[k] - segl[k])) / max(segl[k], 1e-9)
            a0 = out[k] + d[k] * f + np.array([-0.012 * w, -0.025 * h])
            if a0[1] < 0.12 * h:
                continue
            dv = np.array([-rng.uniform(0.18, 0.42), -1.0])
            ln = a0[1] * rng.uniform(0.5, 0.92)
            b0 = a0 + dv * ln
            seg = np.stack([np.linspace(a0[0], b0[0], 4), np.linspace(a0[1], b0[1], 4)], 1)
            seg = _wob(seg, int(rng.integers(1 << 30)), 0.004 * s, 0.2 * s)
            st.add(seg + O, r0 * 0.7 * _press(4, int(rng.integers(1 << 30)), 0.7, 1.0, 0.45), 0.85)
    # a few short strokes on the lit face near the foot, now and then
    if rng.random() < 0.35:
        a0 = np.array([lb[0] * 0.55, h * 0.18])
        seg = np.stack([np.linspace(a0[0], a0[0] + 0.1 * w, 3), np.linspace(a0[1], a0[1] - 0.12 * h, 3)], 1)
        st.add(seg + O, r0 * 0.5, 0.7)
    foot = np.array([[min(ridge[-1, 0] + 0.22 * w, out[-1, 0] - 0.05 * w), 0.0]])
    shade = np.concatenate([out[im:], foot, ridge[::-1]]) + O
    return fill, st, shade


def glyph_hill(X, Y, s, rng):
    """A hatched arc (the classic hill mark)."""
    w = s
    h = s * rng.uniform(0.3, 0.42)
    n = 11
    th = np.linspace(np.pi, 0, n)
    sk = rng.uniform(-0.12, 0.12)
    arc = np.stack([0.5 * w * np.cos(th) + sk * w * np.sin(th) * 0.5, h * np.sin(th) ** 0.85], 1)
    arc[1:-1] += rng.normal(0, 0.012 * h, (n - 2, 2))
    arc = _wob(ink.chaikin(arc, 1), int(rng.integers(1 << 30)), 0.006 * s, 0.4 * s)
    O = np.array([X, Y])
    fill = np.concatenate([arc, [[0.5 * w, -0.03 * h], [-0.5 * w, -0.03 * h]]]) + O
    st = ink.Strokes()
    r0 = 0.011 + 0.008 * s
    st.add(arc + O, r0 * _press(len(arc), int(rng.integers(1 << 30)), 0.8, 1.1, 0.2), 0.92)
    # 2-4 short hatch strokes under the east shoulder
    nh = int(rng.integers(2, 5))
    for k in range(nh):
        u = 0.1 + 0.32 * (k + rng.uniform(0.2, 0.8)) / nh
        x0 = u * w
        top = h * np.sqrt(max(1 - (2 * x0 / w) ** 2, 0.0)) * 0.85
        ln = top * rng.uniform(0.45, 0.8)
        seg = np.array([[x0, top - 0.06 * h], [x0 - 0.12 * ln, top - 0.06 * h - ln]])
        st.add(seg + O, r0 * 0.65, 0.8)
    k0 = len(arc) // 2
    shade = np.concatenate([arc[k0:], [[arc[-1, 0], 0.0], [arc[k0, 0], 0.0]]]) + O
    return fill, st, shade


def glyph_tree(X, Y, s, rng, conifer=False):
    O = np.array([X, Y])
    st = ink.Strokes()
    r0 = 0.0065 + 0.012 * s
    if conifer:
        w = s * rng.uniform(0.42, 0.55)
        hb = s * 0.18
        tip = np.array([rng.uniform(-0.03, 0.03) * s, s])
        L = np.array([[-0.5 * w, hb], tip])
        R = np.array([tip, [0.5 * w, hb]])
        outl = np.concatenate([ink.resample(L, s * 0.12), ink.resample(R, s * 0.12)[1:]])
        outl = _wob(outl, int(rng.integers(1 << 30)), 0.02 * s, 0.4 * s)
        base = np.array([[0.5 * w, hb], [-0.5 * w, hb]])
        fill = np.concatenate([outl, base]) + O
        st.add(outl + O, r0 * _press(len(outl), int(rng.integers(1 << 30)), 0.8, 1.05, 0.2), 0.9)
        st.add(np.array([[0.0, 0.0], [0.0, hb + 0.02 * s]]) + O, r0 * 0.8, 0.9)
        # shade the east half
        for k in range(int(rng.integers(1, 3))):
            y0 = hb + (0.2 + 0.25 * k) * (s - hb)
            x1 = 0.5 * w * (1 - (y0 - hb) / (s - hb)) * 0.8
            st.add(np.array([[0.04 * s, y0 + 0.05 * s], [x1, y0 - 0.04 * s]]) + O, r0 * 0.6, 0.8)
        return fill, st, None
    r = s * rng.uniform(0.3, 0.36)
    c = np.array([rng.uniform(-0.03, 0.03) * s, s - r])
    nb = int(rng.integers(5, 8))
    ph = rng.random() * 2 * np.pi
    th = np.linspace(-np.pi / 2 + 0.5, 1.5 * np.pi - 0.5, 28)
    rr = r * (1.0 + 0.12 * np.abs(np.sin(nb * 0.5 * th + ph)))
    crown = c[None, :] + np.stack([rr * np.cos(th), rr * np.sin(th)], 1)
    crown = _wob(crown, int(rng.integers(1 << 30)), 0.02 * s, 0.5 * s)
    fill = c[None, :] + np.stack([r * 1.08 * np.cos(np.linspace(0, 2 * np.pi, 20)),
                                  r * 1.08 * np.sin(np.linspace(0, 2 * np.pi, 20))], 1)
    fill = np.concatenate([fill, [[0.05 * s, 0.0], [-0.05 * s, 0.0]]]) + O
    st.add(crown + O, r0 * _press(len(crown), int(rng.integers(1 << 30)), 0.8, 1.05, 0.15), 0.9)
    st.add(np.array([[0.0, 0.0], [0.0, s - 2 * r + 0.03 * s]]) + O, r0 * 0.85, 0.9)
    # shade: two little arcs in the east half of the crown
    for k in range(int(rng.integers(1, 3))):
        a0 = -0.9 + 0.55 * k
        tt = np.linspace(a0, a0 + 0.7, 5)
        rr2 = r * (0.55 - 0.18 * k)
        arcp = c[None, :] + np.stack([rr2 * np.cos(tt), rr2 * np.sin(tt)], 1)
        st.add(arcp + O, r0 * 0.6, 0.8)
    return fill, st, None


# ------------------------------------------------------------- building ---

def build(seed=11):
    path = os.path.join(geo.CACHE, f'features_{seed}.npz')
    if os.path.exists(path):
        return load(path)
    rng = np.random.default_rng(seed)
    F, W, H = map_fields()
    lat, lon = F['lat'], F['lon']
    landm = F['landm']
    inland = cv2.erode((landm > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    rug, elev = F['rug'], F['elev']
    gl = _greenland(lat, lon)
    # ---- relief structure: range crests (Hessian ridge strength) and range fronts (coarse slope)
    P = FPPD

    def ridge(E, sig):
        Es = cv2.GaussianBlur(E, (0, 0), sig * P)
        dxx = cv2.Sobel(Es, cv2.CV_32F, 2, 0, ksize=5)
        dyy = cv2.Sobel(Es, cv2.CV_32F, 0, 2, ksize=5)
        dxy = cv2.Sobel(Es, cv2.CV_32F, 1, 1, ksize=5)
        tr = dxx + dyy
        det = dxx * dyy - dxy * dxy
        l1 = tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - det, 0))
        return np.maximum(-l1, 0) * (sig * P) ** 2

    R2 = ridge(elev, 1.2)
    R1 = ridge(elev, 0.5)
    Eg = cv2.GaussianBlur(elev, (0, 0), 0.6 * P)
    gx = cv2.Sobel(Eg, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Eg, cv2.CV_32F, 0, 1, ksize=3)
    Gr = np.sqrt(gx * gx + gy * gy) * P * 0.6
    onl = landm > 0.5
    pr = lambda a, q: float(np.percentile(a[onl], q))
    crest = smoothstep(R2, pr(R2, 90), pr(R2, 99.2))
    front = smoothstep(Gr, pr(Gr, 91), pr(Gr, 99.5))
    M = np.maximum(crest, front) * smoothstep(elev, 0.7, 2.6) * smoothstep(rug, 0.12, 0.35)
    M *= inland * (~gl) * (lat < 80)
    M = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 0.8)
    glyphs = []           # (Y, kind, X, s, param, seed)
    mount = Placer(0.66, 0.5)
    cand, val = _candidates(M, 0.3, rng, 50000, W, H)
    order = np.argsort(-(val + rng.normal(0, 0.06, len(val))))
    for i in order:
        x, y = cand[i]
        m = float(val[i])
        e = float(_sample(elev, np.array([[x, y]]))[0])
        s = 0.85 + 0.55 * min(m, 1.0) + 0.5 * float(smoothstep(e, 2.0, 8.0))
        if mount.ok(x, y, s):
            # the whole foot must be on land
            if _sample(landm, np.array([[x - 0.45 * s, y], [x + 0.45 * s, y]])).min() < 0.5:
                continue
            mount.add(x, y, s)
            glyphs.append((y, 0, x, s, m, int(rng.integers(1 << 30))))
    # ---- hills: lesser ridges and hill country, sparse, never a carpet
    Hf = np.maximum(smoothstep(R1, pr(R1, 82), pr(R1, 97)), 0.8 * smoothstep(Gr, pr(Gr, 80), pr(Gr, 94)))
    Hf *= smoothstep(rug, 0.1, 0.3) * (1 - smoothstep(M, 0.2, 0.4))
    Hf *= inland * (~gl) * (lat < 78)
    Hf = cv2.GaussianBlur(Hf.astype(np.float32), (0, 0), 1.0)
    hills = Placer(0.95, 0.75)
    cand, val = _candidates(Hf, 0.22, rng, 50000, W, H)
    order = np.argsort(-(val + rng.normal(0, 0.15, len(val))))
    for i in order:
        x, y = cand[i]
        hv = float(val[i])
        if rng.random() > 0.1 + 0.8 * hv:
            continue
        s = 0.5 + 0.3 * min(hv, 1.0) + rng.uniform(-0.05, 0.08)
        if hills.ok(x, y, s, (mount,)):
            if _sample(landm, np.array([[x - 0.45 * s, y], [x + 0.45 * s, y]])).min() < 0.5:
                continue
            hills.add(x, y, s)
            glyphs.append((y, 1, x, s, hv, int(rng.integers(1 << 30))))
    # ---- forests: round crowns where the Blue Marble is forested; conifers in the boreal belt
    boreal = smoothstep(lat, 47.0, 53.0) * (1 - smoothstep(lat, 63.0, 69.0)) * (lon > -170)
    patch = cv2.GaussianBlur(rng.random((H // 8, W // 8)).astype(np.float32), (0, 0), 1.5)
    patch = cv2.resize(patch, (W, H), interpolation=cv2.INTER_CUBIC)
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)
    Ft = np.clip(F['forest'] * 1.25, 0, 1) * (lat < 55)
    Fb = boreal * np.clip(0.55 + 0.35 * patch, 0, 1) * (1 - F['desert'])
    Ff = np.maximum(Ft, Fb * 0.85) * inland * (~gl)
    Ff *= (1 - smoothstep(M, 0.15, 0.3)) * (1 - smoothstep(Hf, 0.45, 0.7))
    Ff = cv2.GaussianBlur(Ff.astype(np.float32), (0, 0), 0.8)
    trees = Placer(0.72, 0.62)
    cand, val = _candidates(Ff, 0.3, rng, 90000, W, H)
    order = rng.permutation(len(cand))
    for i in order:
        x, y = cand[i]
        fv = float(val[i])
        if rng.random() > (fv - 0.3) * 2.2:
            continue
        la = float(geo.ilat(y))
        con = la > 47.0 + rng.uniform(-3, 3) or float(_sample(elev, np.array([[x, y]]))[0]) > 3.0
        s = rng.uniform(0.3, 0.38) if not con else rng.uniform(0.3, 0.4)
        if trees.ok(x, y, s, (mount, hills)):
            trees.add(x, y, s)
            glyphs.append((y, 3 if con else 2, x, s, fv, int(rng.integers(1 << 30))))
    # ---- desert stipple (plain dots, weighted)
    D = F['desert'] * inland * (1 - smoothstep(M, 0.1, 0.3))
    n = 160000
    cand, val = _candidates(D, 0.08, rng, n, W, H)
    keep = rng.random(len(cand)) < np.clip((val - 0.08) * 1.6, 0, 1)
    dots = cand[keep]
    # thin dots that fall on glyphs (they'd be hidden anyway; saves time)
    dots = dots[rng.random(len(dots)) < 0.6]
    glyphs.sort(key=lambda g: -g[0])
    G = np.array(glyphs, np.float64)
    np.savez_compressed(path, glyphs=G, dots=dots)
    return load(path)


def load(path):
    d = np.load(path)
    return dict(glyphs=d['glyphs'], dots=d['dots'])


def make_glyph(g):
    y, kind, x, s, p, sd = g
    rng = np.random.default_rng(int(sd))
    k = int(kind)
    if k == 0:
        return glyph_mountain(x, y, s, p, rng)
    if k == 1:
        return glyph_hill(x, y, s, rng)
    return glyph_tree(x, y, s, rng, conifer=(k == 3))


# ------------------------------------------------------------ polylines ---

def coast_lines(step=0.04, seed=3):
    """Land rings in map coords: smoothed a touch, resampled, hand-wobbled. Cached."""
    path = os.path.join(geo.CACHE, f'coast_{seed}.npz')
    if os.path.exists(path):
        d = np.load(path)
        P, off, hole = d['P'], d['off'], d['hole']
        return [(P[off[i]:off[i + 1]], bool(hole[i])) for i in range(len(off) - 1)]
    out = []
    rng = np.random.default_rng(seed)
    for ll, hole in geo.land_rings():
        xy = geo.to_map_ring(ll)
        if len(xy) < 4:
            continue
        xy = ink.chaikin(xy[:-1] if np.allclose(xy[0], xy[-1]) else xy, 1, closed=True)
        xy = ink.resample(xy, step, closed=True)
        if len(xy) < 4:
            continue
        s = ink.arclen(np.concatenate([xy, xy[:1]]))[:-1]
        t = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
        t /= np.linalg.norm(t, axis=1)[:, None] + 1e-12
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        L = s[-1] + step
        # closed wobble: sines with integer cycles around the ring
        wv = np.zeros(len(xy))
        for k, (wl, a) in enumerate(((0.9, 0.022), (0.33, 0.011), (0.15, 0.005))):
            cyc = max(1, int(round(L / wl)))
            wv += a * np.sin(2 * np.pi * cyc * s / L + rng.random() * 6.283)
        xy = xy + nrm * wv[:, None]
        for c in geo.wrap_copies(xy):
            out.append((c, hole))
    P = np.concatenate([o[0] for o in out])
    off = np.cumsum([0] + [len(o[0]) for o in out])
    np.savez_compressed(path, P=P, off=off, hole=np.array([o[1] for o in out]))
    return coast_lines(step, seed)


def lake_lines(step=0.04, seed=4):
    out = []
    rng = np.random.default_rng(seed)
    for ll in geo.lakes():
        xy = geo.to_map_ring(ll)
        xy = ink.chaikin(xy, 2, closed=True)
        xy = ink.resample(xy, step, closed=True)
        if len(xy) < 5:
            continue
        xy = _wob(np.concatenate([xy, xy[:1]]), int(rng.integers(1 << 30)), 0.01, 0.5)
        for c in geo.wrap_copies(xy):
            out.append(c)
    return out


def river_lines(seed=5):
    """[(polyline map XY, radius array)] from source to mouth."""
    out = []
    rng = np.random.default_rng(seed)
    for name, pts in RIVERS.items():
        ll = np.array(pts, np.float64)
        xy = geo.ll2map(ll[:, 0], ll[:, 1])
        xy = ink.chaikin(xy, 3)
        xy = ink.resample(xy, 0.03)
        L = ink.arclen(xy)
        # meanders: bigger wiggles upstream, a lazier swing downstream
        u = L / max(L[-1], 1e-6)
        mw = wobble1d(L, int(rng.integers(1 << 30)), 0.9, 0.05) + wobble1d(L, int(rng.integers(1 << 30)), 0.25, 0.018)
        t = np.gradient(xy, axis=0)
        t /= np.linalg.norm(t, axis=1)[:, None] + 1e-12
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        env = np.clip(np.minimum(L, L[-1] - L) / 0.4, 0, 1)
        xy = xy + nrm * (mw * env)[:, None]
        sz = RIVER_SIZE.get(name, 0.6)
        rad = (0.010 + 0.02 * sz * u ** 0.6) * (0.9 + 0.2 * np.sin(L * 3.1 + rng.random() * 6))
        out.append((xy, rad, name))
    return out


if __name__ == '__main__':
    import time
    t0 = time.time()
    f = build()
    g = f['glyphs']
    print('glyphs', len(g), 'mount', int((g[:, 1] == 0).sum()), 'hills', int((g[:, 1] == 1).sum()),
          'trees', int((g[:, 1] == 2).sum()), 'conifers', int((g[:, 1] == 3).sum()), 'dots', len(f['dots']),
          round(time.time() - t0, 1), 's')
    t0 = time.time()
    c = coast_lines()
    print('coast rings', len(c), sum(len(p) for p, _ in c), round(time.time() - t0, 1), 's')
