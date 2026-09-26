"""THE WORLD ANSWERS v2: fire, not fibre.

The answering world is drawn as FIRES, not as a graph. Beacon sites sit where people have always
lit signal fires: on the crests of mountain ranges, on coasts and along the great rivers, with a few
lonely fires in between (deserts, steppe, islands). The fire spreads from the first beacon in the
Himalaya as a race along those chains (a seeded Dijkstra over a chain graph whose ridge, coast and
river links are cheaper than cross-country ones). Every link is a TRAVELLING EMBER: a bright head
with a short fading tail, gone about a second after it lands. Nothing persists but the fires: small
flickering flames of uneven size, each with a brief flare when it catches.

Deterministic and stateless per frame: `FireNet.draw(img, cam, t)` renders the fires as they are
at global frame t. All times are global frames.
"""
import heapq
import json
import math
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

import globe as G
import places
from rivers import RIVERS

R_KM = G.R_KM
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(ROOT, 'renders', 'globe', 'cache')
EQ_W, EQ_H = 2048, 1024

KIND_NAMES = ('origin', 'ridge', 'coast', 'river', 'lone', 'island')
K_ORIGIN, K_RIDGE, K_COAST, K_RIVER, K_LONE, K_ISLAND = range(6)
F_HOP, F_SEA, F_LEAP, F_THROW = 0, 1, 2, 3      # flight kinds (THROW: the first beacon's long throws)


def ll2v(lat, lon):
    lat = np.radians(np.asarray(lat, np.float64))
    lon = np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def v2ll(P):
    P = np.asarray(P, np.float64)
    return np.degrees(np.arcsin(np.clip(P[..., 2], -1, 1))), np.degrees(np.arctan2(P[..., 1], P[..., 0]))


def gc_km(a, b):
    return np.arccos(np.clip(np.sum(a * b, -1), -1, 1)) * R_KM


def smoothstep(x, a, b):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def eq_xy(lat, lon, W=EQ_W, H=EQ_H):
    return (np.asarray(lon) + 180.0) / 360.0 * W - 0.5, (90.0 - np.asarray(lat)) / 180.0 * H - 0.5


def sample_eq(field, lat, lon):
    x, y = eq_xy(lat, lon, field.shape[1], field.shape[0])
    return cv2.remap(np.asarray(field, np.float32), np.asarray(x, np.float32).reshape(-1, 1),
                     np.asarray(y, np.float32).reshape(-1, 1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP).reshape(np.shape(lat))


def value_noise_sphere(P, freq, seed):
    """Smooth pseudo-random field on the unit sphere (0..1), for clustering the spacing."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(P))
    norm = 0.0
    amp = 1.0
    f = freq
    for o in range(3):
        d = rng.normal(size=(6, 3))
        ph = rng.random(6) * 2 * np.pi
        s = np.zeros(len(P))
        for k in range(6):
            s += np.sin(P @ d[k] * f + ph[k])
        acc += amp * s / 6.0
        norm += amp
        amp *= 0.5
        f *= 2.1
    return np.clip(0.5 + 0.5 * acc / norm * 1.8, 0, 1)


# ------------------------------------------------------------ geography ---

def geo_fields():
    """Equirect 2048x1024 fields (cached): land (0..1), elev (relative proxy), ridge (crest
    strength), thin (1 on range crest lines, non-maximum suppressed)."""
    path = os.path.join(CACHE, 'fires_geo_v1.npz')
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    W, H = EQ_W, EQ_H
    mask8 = cv2.imread(os.path.join(CACHE, 'land_mask_8k.png'), cv2.IMREAD_GRAYSCALE)
    land = cv2.resize(mask8, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # elevation proxy: Frankot-Chellappa integration of the relief normal map (the MAP
    # department's method, geo.py), referenced to sea level
    nm = cv2.imread(os.path.join(ROOT, 'assets', 'textures', 'earth_normal_2048.jpg')).astype(np.float32)[..., ::-1]
    nm = nm / 255.0 * 2 - 1
    nm = cv2.resize(nm, (W, H), interpolation=cv2.INTER_AREA)
    nz = np.maximum(nm[..., 2], 0.2)
    p = -nm[..., 0] / nz
    q = -nm[..., 1] / nz
    Pp = np.concatenate([p, p[::-1]], 0)
    Qq = np.concatenate([q, -q[::-1]], 0)
    wy = np.fft.fftfreq(2 * H)[:, None] * 2 * np.pi
    wx = np.fft.fftfreq(W)[None, :] * 2 * np.pi
    den = wx ** 2 + wy ** 2
    den[0, 0] = 1
    Z = (-1j * wx * np.fft.fft2(Pp) - 1j * wy * np.fft.fft2(Qq)) / den
    Z[0, 0] = 0
    z = np.real(np.fft.ifft2(Z))[:H].astype(np.float32)
    sea = cv2.erode((land < 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    num = cv2.GaussianBlur(z * sea, (0, 0), 40, borderType=cv2.BORDER_REFLECT)
    dd = cv2.GaussianBlur(sea, (0, 0), 40, borderType=cv2.BORDER_REFLECT)
    elev = ((z - num / np.maximum(dd, 1e-3)) * land).astype(np.float32)
    # crest strength: Hessian ridge measure at two scales (px of this grid; 1 px ~ 19.5 km)
    ridge = np.zeros((H, W), np.float32)
    nx = np.zeros((H, W), np.float32)
    ny = np.zeros((H, W), np.float32)
    for sig, wgt in ((3.0, 1.0), (5.5, 0.9)):
        Es = cv2.GaussianBlur(elev, (0, 0), sig, borderType=cv2.BORDER_WRAP if False else cv2.BORDER_REFLECT)
        dxx = cv2.Sobel(Es, cv2.CV_32F, 2, 0, ksize=5) / 16.0
        dyy = cv2.Sobel(Es, cv2.CV_32F, 0, 2, ksize=5) / 16.0
        dxy = cv2.Sobel(Es, cv2.CV_32F, 1, 1, ksize=5) / 16.0
        tr = dxx + dyy
        disc = np.sqrt(np.maximum(tr * tr / 4 - (dxx * dyy - dxy * dxy), 0))
        l1 = tr / 2 - disc                      # most negative curvature (across the crest)
        r = np.maximum(-l1, 0) * sig * sig * wgt
        # eigenvector of l1 (across-ridge direction)
        ex = dxy
        ey = l1 - dxx
        bad = np.abs(ex) + np.abs(ey) < 1e-9
        ex = np.where(bad, l1 - dyy, ex)
        ey = np.where(bad, dxy, ey)
        nl = np.sqrt(ex * ex + ey * ey) + 1e-12
        better = r > ridge
        ridge = np.where(better, r, ridge)
        nx = np.where(better, ex / nl, nx)
        ny = np.where(better, ey / nl, ny)
    ridge *= (land > 0.5)
    # non-maximum suppression across the crest -> thin crest lines
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = cv2.remap(ridge, xx + nx, yy + ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    b = cv2.remap(ridge, xx - nx, yy - ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    thin = ((ridge >= a) & (ridge >= b) & (ridge > 0)).astype(np.float32) * ridge
    np.savez_compressed(path, land=land, elev=elev, ridge=ridge, thin=thin)
    return geo_fields()


def _hysteresis(thin, lo, hi, min_px):
    weak = (thin > lo).astype(np.uint8)
    strong = thin > hi
    # bridge one-pixel gaps in the crest lines
    wk = cv2.dilate(weak, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(wk, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong & (wk > 0)])] = True
    cnt = np.bincount(lab.ravel(), minlength=n)
    keep &= cnt >= min_px
    keep[0] = False
    return keep[lab] & (weak > 0)


def crest_pixels(F, lo_pct=93.0, hi_pct=98.6, min_px=9):
    """Range crests: thin ridge pixels kept by hysteresis, on land, not on the ice caps.
    Returns lat, lon, strength (0..1)."""
    land = F['land']
    thin = F['thin']
    H, W = thin.shape
    lat_c = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    latg = np.repeat(lat_c[:, None], W, 1)
    lon_c = (np.arange(W) + 0.5) / W * 360.0 - 180.0
    long_ = np.repeat(lon_c[None, :], H, 0)
    ok = (land > 0.6) & (np.abs(latg) < 72) & ~_greenland(latg, long_) & (latg > -56)
    vals = F['ridge'][ok & (F['ridge'] > 0)]
    lo = np.percentile(vals, lo_pct)
    hi = np.percentile(vals, hi_pct)
    m = _hysteresis(np.where(ok, thin, 0), lo, hi, min_px)
    # a relief floor: a crest must stand above its surroundings
    m &= F['elev'] > np.percentile(F['elev'][land > 0.6], 55)
    yy, xx = np.nonzero(m)
    s = np.clip((thin[yy, xx] - lo) / (hi - lo), 0, 1.5)
    return lat_c[yy], lon_c[xx], s


def _greenland(lat, lon):
    return (lat > 59.5) & (lon > -74) & (lon < -10) & ~((lon > -26) & (lat < 67))


def coast_lines(min_len_km=260.0):
    """Natural Earth 50m coastlines as unit-vector polylines, densified to ~12 km, without tiny
    islets, Antarctica or the ice-bound Arctic."""
    d = json.load(open(os.path.join(ROOT, 'assets', 'data', 'ne_50m_coastline.geojson')))
    out = []
    for ft in d['features']:
        g = ft['geometry']
        cs = [g['coordinates']] if g['type'] == 'LineString' else g['coordinates']
        for c in cs:
            a = np.asarray(c, np.float64)
            V = ll2v(a[:, 1], a[:, 0])
            seg = gc_km(V[1:], V[:-1])
            L = seg.sum()
            if L < min_len_km:
                continue
            la = a[:, 1]
            if la.max() < -56 or la.min() > 66.0:
                continue
            out.append(_densify(V, 12.0))
    return out


def _densify(V, step_km):
    pts = [V[0]]
    for a, b in zip(V[:-1], V[1:]):
        L = gc_km(a, b)
        n = int(L // step_km)
        for k in range(1, n + 1):
            u = k / (n + 1)
            p = a * (1 - u) + b * u
            pts.append(p / np.linalg.norm(p))
        pts.append(b)
    return np.asarray(pts)


def river_lines():
    out = []
    for name, pts in RIVERS.items():
        a = np.asarray(pts, np.float64)
        out.append((name, _densify(ll2v(a[:, 0], a[:, 1]), 12.0)))
    return out


# ------------------------------------------------------------- the net ---

class _SphereGrid:
    """Incremental neighbour queries on the unit sphere (3-D hash grid)."""

    def __init__(self, cell_km):
        self.c = cell_km / R_KM
        self.cells = {}
        self.P = []

    def _key(self, p):
        return (int(math.floor(p[0] / self.c)), int(math.floor(p[1] / self.c)), int(math.floor(p[2] / self.c)))

    def near(self, p, r_km):
        """Indices of stored points within r_km (chord approx; r_km <= cell size)."""
        kx, ky, kz = self._key(p)
        out = []
        r = r_km / R_KM
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for i in self.cells.get((kx + dx, ky + dy, kz + dz), ()):
                        q = self.P[i]
                        if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2 <= r * r:
                            out.append(i)
        return out

    def add(self, p):
        self.P.append(p)
        self.cells.setdefault(self._key(p), []).append(len(self.P) - 1)
        return len(self.P) - 1


class FireNet:
    """Beacon sites, their chain graph, the spread from the first beacon, and the drawing."""

    VERSION = 8

    def __init__(self, seed=3, t0=1760.0, cache=True, p=0.35, throw_land=34.0):
        self.seed = seed
        self.t0 = t0
        # the shipped net (tracked in shots/globe/data/) wins, so every machine draws the same fires;
        # otherwise the net is built and cached in renders/globe/cache/
        name = f'fires_net_v{self.VERSION}_{seed}.npz'
        shipped = os.path.join(HERE, 'data', name)
        path = os.path.join(CACHE, name)
        self._stored = {}
        if cache and (os.path.exists(shipped) or os.path.exists(path)):
            d = np.load(shipped if os.path.exists(shipped) else path)
            for k in d.files:
                setattr(self, k, d[k])
                self._stored[k] = d[k]
        else:
            self._build()
            if cache:
                np.savez(path, P=self.P, kind=self.kind, size=self.size, s_ign=self.s_ign,
                         parent=self.parent, fl=self.fl, edges=self.edges)
        self.set_timing(p, throw_land)
        self._prep_draw()

    def ship(self):
        """Write the net plus every random draw the renderer uses to shots/globe/data/ (tracked)."""
        os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
        out = os.path.join(HERE, 'data', f'fires_net_v{self.VERSION}_{self.seed}.npz')
        np.savez_compressed(out, P=self.P, kind=self.kind, size=self.size, s_ign=self.s_ign,
                            parent=self.parent, fl=self.fl, edges=self.edges, sat_par=self.sat_par,
                            sat_P=self.sat_P, sat_dt=self.sat_dt, sat_szf=self.sat_szf,
                            Dtemp=self.Dtemp, fq=self.fq, fph=self.fph)
        return out

    # ------------------------------------------------------------ build ---
    def _build(self):
        rng = np.random.default_rng(self.seed)
        F = geo_fields()
        cands = []          # (P, kind, prio, spacing_km, feature, order)

        def add(P, kind, prio, d, feat=-1, order=None):
            n = len(P)
            order = np.arange(n) if order is None else order
            cands.append((np.asarray(P, np.float64), np.full(n, kind), np.asarray(prio, np.float64) * np.ones(n),
                          np.asarray(d, np.float64) * np.ones(n), np.full(n, feat), np.asarray(order)))

        # the first beacon
        add(ll2v(*places.ORIGIN)[None, :], K_ORIGIN, 99.0, 120.0)
        # range crests (strongest first); spacing clusters and thins with a smooth noise field
        la, lo, s = crest_pixels(F)
        P = ll2v(la, lo)
        clus = value_noise_sphere(P, 9.0, self.seed + 1)
        d = (106.0 - 26.0 * np.clip(s, 0, 1)) * (0.75 + 0.6 * clus)
        add(P, K_RIDGE, 3.0 + s + rng.normal(0, 0.15, len(s)), d)
        # islands and the far places the ocean leaps land on
        isl = [p for p in places.HIGH[-60:]] + [a for ab in places.LEAPS for a in ab]
        isl = np.asarray(isl, np.float64)
        add(ll2v(isl[:, 0], isl[:, 1]), K_ISLAND, 2.9, 110.0)
        # rivers: every other densified point is a candidate
        for fi, (name, V) in enumerate(river_lines()):
            V = V[::2]
            clus = value_noise_sphere(V, 11.0, self.seed + 2)
            add(V, K_RIVER, 2.0 + 0.3 * rng.random(len(V)), 215.0 * (0.7 + 0.8 * clus), feat=1000 + fi)
        # coasts
        for fi, V in enumerate(coast_lines()):
            V = V[::3]
            clus = value_noise_sphere(V, 7.0, self.seed + 3)
            add(V, K_COAST, 1.0 + 0.4 * rng.random(len(V)), 300.0 * (0.7 + 0.85 * clus), feat=5000 + fi)
        # lonely fires across open land (deserts, steppe, forest): people everywhere
        W, H = EQ_W, EQ_H
        lat_c = 90.0 - (np.arange(H) + 0.5) / H * 180.0
        lon_c = (np.arange(W) + 0.5) / W * 360.0 - 180.0
        w = (F['land'] > 0.9) * np.cos(np.radians(lat_c))[:, None]
        latg = np.repeat(lat_c[:, None], W, 1)
        w = w * (np.abs(latg) < 70) * (latg > -56) * ~_greenland(latg, np.repeat(lon_c[None, :], H, 0))
        pw = w.ravel() / w.sum()
        idx = rng.choice(len(pw), size=9000, p=pw)
        yy, xx = np.divmod(idx, W)
        Pl = ll2v(lat_c[yy], lon_c[xx])
        add(Pl, K_LONE, 0.2 + 0.2 * rng.random(len(Pl)), 420.0 * (0.8 + 0.4 * rng.random(len(Pl))))

        P = np.concatenate([c[0] for c in cands])
        kind = np.concatenate([c[1] for c in cands])
        prio = np.concatenate([c[2] for c in cands])
        dsp = np.concatenate([c[3] for c in cands])
        feat = np.concatenate([c[4] for c in cands])
        order = np.concatenate([c[5] for c in cands])
        # greedy Poisson-disk selection in order of priority
        grid = _SphereGrid(700.0)
        acc = []
        acc_d = []
        for i in np.argsort(-prio, kind='stable'):
            p = P[i]
            ok = True
            for j in grid.near(p, 0.5 * (dsp[i] + 500.0)):
                q = grid.P[j]
                dk = math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2) * R_KM
                if dk < 0.47 * (dsp[i] + acc_d[j]):
                    ok = False
                    break
            if ok:
                grid.add(p)
                acc.append(i)
                acc_d.append(dsp[i])
        acc = np.asarray(acc)
        self.P = P[acc]
        self.kind = kind[acc].astype(np.int32)
        self.dsp = dsp[acc]
        feat = feat[acc]
        order = order[acc]
        n = len(acc)
        # nobody builds beacons on a ruler line: nudge crest and river fires off the exact line
        jm = (self.kind == K_RIDGE) | (self.kind == K_RIVER)
        self.P[jm] += rng.normal(0, 1, (int(jm.sum()), 3)) * (8.0 / R_KM)
        self.P /= np.linalg.norm(self.P, axis=1)[:, None]
        print('fires: sites', n, {KIND_NAMES[k]: int((self.kind == k).sum()) for k in range(6)})

        # ------------------------------------------------ the chain graph
        E = {}              # (i, j) -> (multiplier, flight kind)

        def link(i, j, m, fk):
            a, b = (i, j) if i < j else (j, i)
            if a == b:
                return
            if (a, b) not in E or E[(a, b)][0] > m:
                E[(a, b)] = (m, fk)

        crest = np.zeros((EQ_H, EQ_W), np.uint8)
        x, y = eq_xy(la, lo)
        crest[np.clip(np.round(y).astype(int), 0, EQ_H - 1), np.round(x).astype(int) % EQ_W] = 1
        crest = cv2.dilate(crest, np.ones((3, 3), np.uint8)).astype(np.float32)
        land = F['land']
        tree = cKDTree(self.P)

        def along(i, j, field, step_km=9.0):
            L = gc_km(self.P[i], self.P[j])
            m = max(3, int(L / step_km))
            u = np.linspace(0, 1, m)[:, None]
            Q = self.P[i] * (1 - u) + self.P[j] * u
            Q /= np.linalg.norm(Q, axis=1)[:, None]
            lat_, lon_ = v2ll(Q)
            return sample_eq(field, lat_, lon_), L

        # ridge chains: neighbours whose connecting line runs along the crest
        rid = np.where(self.kind <= K_RIDGE)[0]
        for i in rid:
            for j in tree.query_ball_point(self.P[i], 2.1 * self.dsp[i] / R_KM):
                if j <= i or self.kind[j] > K_RIDGE:
                    continue
                sup, L = along(i, j, crest)
                if (sup > 0.5).mean() >= 0.62:
                    link(i, j, 0.72, F_HOP)
        # river and coast chains: consecutive accepted sites along each line
        for f in np.unique(feat[feat >= 1000]):
            ids = np.where(feat == f)[0]
            ids = ids[np.argsort(order[ids])]
            for a, b in zip(ids[:-1], ids[1:]):
                L = gc_km(self.P[a], self.P[b])
                if L < 2.6 * max(self.dsp[a], self.dsp[b]):
                    wat, _ = along(a, b, land)
                    if (wat < 0.5).mean() < 0.35 or L < 260:
                        link(a, b, 0.9 if f < 5000 else 0.95, F_HOP)
        # line of sight between chains (and to the lonely fires)
        for i in range(n):
            r = max(1.55 * self.dsp[i], 280.0)
            if self.kind[i] == K_LONE:
                r = 900.0
            dd, jj = tree.query(self.P[i], k=8, distance_upper_bound=r / R_KM)
            nk = 0
            for dk, j in zip(dd, jj):
                if not np.isfinite(dk) or j == i or (min(i, j), max(i, j)) in E:
                    continue
                wat, L = along(i, j, land)
                wf = (wat < 0.5).mean()
                if wf * L > 140.0:
                    if L < 650.0:
                        link(i, j, 1.6, F_SEA)
                        nk += 1
                else:
                    link(i, j, 1.5 if self.kind[i] == K_LONE or self.kind[j] == K_LONE else 1.3, F_HOP)
                    nk += 1
                if nk >= 2:
                    break
        # designated ocean leaps
        for a, b in places.LEAPS:
            ia = int(tree.query(ll2v(*a))[1])
            ib = int(tree.query(ll2v(*b))[1])
            link(ia, ib, 0.85, F_LEAP)
        self.E = E
        self._spread(rng)

    def _spread(self, rng):
        n = len(self.P)
        adj = [[] for _ in range(n)]

        def cost(a, b, m, fk):
            wind = 260.0 if fk == F_LEAP else (40.0 if fk == F_SEA else 0.0)
            return wind + gc_km(self.P[a], self.P[b]) * m

        for (a, b), (m, fk) in self.E.items():
            c = cost(a, b, m, fk)
            adj[a].append((b, c, fk))
            adj[b].append((a, c, fk))
        catch = rng.uniform(16.0, 70.0, n)       # a watcher sees the fire land, then throws on
        catch[0] = 1.0
        # the first beacon throws far: three embers to heights 300-650 km away in different
        # directions (heavy and slow, the biggest things on screen); the chains then grow back
        # and onward from where they land
        d0 = gc_km(self.P, self.P[0][None, :])
        up0 = self.P[0]
        e1 = np.cross(up0, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up0, e1)
        cand = np.where((d0 > 260.0) & (d0 < 650.0) & (self.kind != K_LONE) & (self.kind != K_ISLAND))[0]
        sc = -np.abs(d0[cand] - 470.0) / 150.0 + np.where(self.kind[cand] == K_RIDGE, 1.0, 0.0)
        order = cand[np.argsort(-(sc + rng.normal(0, 0.3, len(cand))))]
        throws = []
        for k in order:
            v = self.P[k] - up0
            a = math.atan2(float(v @ e2), float(v @ e1))
            if all(abs((a - b + math.pi) % (2 * math.pi) - math.pi) > math.radians(75) for b, _ in throws):
                throws.append((a, int(k)))
            if len(throws) == 3:
                break
        first = []
        for a, k in throws:
            self.E[(0, k)] = (0.45, F_THROW)
            first.append((k, cost(0, k, 0.45, F_THROW), F_THROW))
        budget = np.where(rng.random(n) < 0.6, 2, 3)
        budget[0] = 3
        s_ign = np.full(n, np.inf)
        s_lau = np.zeros(n)
        parent = np.full(n, -1)
        pfk = np.zeros(n, np.int64)
        claimed = np.zeros(n, bool)
        claimed[0] = True
        s_ign[0] = 0.0
        heap = [(0.0, 0, 0, -1, 0, 0.0)]           # (s, event 0=lit 1=late, node, from, fk, launch)
        while True:
            while heap:
                sv, ev, j, i, fk, sl = heapq.heappop(heap)
                if ev == 1:
                    if claimed[j]:
                        continue
                    claimed[j] = True
                    parent[j] = i
                    pfk[j] = fk
                    s_ign[j] = sv
                    s_lau[j] = sl
                    heapq.heappush(heap, (sv, 0, j, i, fk, sl))
                    continue
                # node j is lit: throw on
                if j == 0:
                    for m_, (k, c, f) in enumerate(first):
                        sl2 = sv + catch[0] + m_ * rng.uniform(2.0, 6.0)
                        claimed[k] = True
                        parent[k] = 0
                        pfk[k] = f
                        s_ign[k] = sl2 + c
                        s_lau[k] = sl2
                        heapq.heappush(heap, (sl2 + c, 0, k, 0, f, sl2))
                    continue
                opts = sorted((c, k, f) for (k, c, f) in adj[j] if not claimed[k])
                thrown = 0
                for c, k, f in opts:
                    if f == F_LEAP or thrown < budget[j]:
                        sl2 = sv + catch[j] + thrown * rng.uniform(25.0, 70.0) + rng.uniform(0.0, 10.0)
                        claimed[k] = True
                        parent[k] = j
                        pfk[k] = f
                        s_ign[k] = sl2 + c
                        s_lau[k] = sl2
                        heapq.heappush(heap, (sl2 + c, 0, k, j, f, sl2))
                        if f != F_LEAP:
                            thrown += 1
                    else:
                        sl2 = sv + catch[j] + 60.0 + rng.uniform(0.0, 50.0)
                        heapq.heappush(heap, (sl2 + c * 1.15, 1, k, j, f, sl2))
            un = np.where(~claimed)[0]
            if len(un) == 0:
                break
            # repair: the unreached site nearest to the lit world is answered across the sea
            lit = np.where(claimed)[0]
            lt = cKDTree(self.P[lit])
            dd, ii = lt.query(self.P[un])
            k = int(np.argmin(dd))
            i, j = int(lit[ii[k]]), int(un[k])
            L = dd[k] * R_KM
            fk = F_SEA if L < 900 else F_LEAP
            m = 1.6 if fk == F_SEA else 0.85
            self.E[(min(i, j), max(i, j))] = (m, fk)
            c = cost(i, j, m, fk)
            adj[i].append((j, c, fk))
            adj[j].append((i, c, fk))
            sl2 = s_ign[i] + catch[i] + 30.0
            heapq.heappush(heap, (sl2 + c, 1, j, i, fk, sl2))
        self.s_ign = s_ign
        self.parent = parent
        # flights: parent -> child (the tree), with launch in s-time
        ch = np.where(parent >= 0)[0]
        self.fl = np.stack([parent[ch], ch, s_lau[ch], s_ign[ch], pfk[ch]], 1).astype(np.float64)
        self.edges = np.array([[a, b, m, fk] for (a, b), (m, fk) in self.E.items()], np.float64)
        # fire sizes: uneven; bonfires on the heights, smaller on the plains
        base = np.array([3.0, 1.35, 0.85, 0.8, 0.6, 0.95])[self.kind]
        self.size = base * rng.lognormal(0.0, 0.62, n) * np.where(rng.random(n) < 0.06, 2.2, 1.0)
        self.size[0] = 3.0
        print('fires: flights', len(self.fl), 'kinds', np.bincount(self.fl[:, 4].astype(int), minlength=3),
              's_ign pct', np.percentile(s_ign, [5, 25, 50, 75, 95, 100]).round(0))

    # ----------------------------------------------------------- timing ---
    def set_timing(self, p, throw_land):
        """Travel time s (km-like) -> frames: f = t0 + K s^p (p < 1: heavy start, then faster and
        faster). K puts the first beacon's long throws down (median) `throw_land` frames after t0."""
        self.p = p
        th = self.fl[:, 4] == F_THROW
        s_th = float(np.median(self.fl[th, 3])) if th.any() else 200.0
        self.K = throw_land / s_th ** p
        self.t_ign = self.frame_of(self.s_ign)
        self.t_ign[0] = self.t0 - 400.0             # the first beacon has burned since FIRST BEACON

    def frame_of(self, s):
        return self.t0 + self.K * np.power(np.maximum(s, 0.0), self.p)

    def s_of(self, f):
        return np.power(np.maximum(f - self.t0, 0.0) / self.K, 1.0 / self.p)

    def summary(self):
        ti = self.t_ign[1:]
        return dict(sites=len(self.P), flights=len(self.fl),
                    t_pct=np.percentile(ti, [5, 25, 50, 75, 90, 99, 100]).round(1).tolist())

    # ---------------------------------------------------------- drawing ---
    def _prep_draw(self):
        n = len(self.P)
        if 'sat_par' not in self._stored:
            rng = np.random.default_rng(self.seed + 77)
            # every beacon is a small cluster: the beacon itself plus a few lesser fires lit around
            # it a moment later (a hill-top and its village): no string-of-beads regularity
            lam = np.array([3.0, 1.4, 0.9, 1.1, 0.5, 1.1])[self.kind]
            nsat = rng.poisson(lam)
            par = np.repeat(np.arange(n), nsat)
            m = len(par)
            up = self.P[par]
            e1 = np.cross(up, [0.0, 0.0, 1.0])
            e1 /= np.linalg.norm(e1, axis=1)[:, None] + 1e-12
            e2 = np.cross(up, e1)
            ang = rng.random(m) * 2 * np.pi
            dist = rng.uniform(6.0, 26.0, m) / R_KM
            SP = up + (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2) * dist[:, None]
            SP /= np.linalg.norm(SP, axis=1)[:, None]
            dt = rng.uniform(3.0, 16.0, m)
            org = par == 0
            dt[org] = rng.uniform(4.0, 22.0, int(org.sum()))              # they answer the flare
            szf = rng.uniform(0.15, 0.5, m)
            N = n + m
            self.sat_par, self.sat_P, self.sat_dt, self.sat_szf = par, SP, dt, szf
            self.Dtemp = rng.beta(2.0, 2.0, N)
            # flicker: three incommensurate breaths per fire (1-4 Hz) plus a small fast lick
            self.fq = np.stack([rng.uniform(0.22, 0.45, N), rng.uniform(0.5, 0.9, N), rng.uniform(1.4, 2.6, N)], 1)
            self.fph = rng.random((N, 3)) * 2 * np.pi
        par = self.sat_par
        m = len(par)
        st = self.t_ign[par] + self.sat_dt
        org = par == 0
        st[org] = self.t0 + self.sat_dt[org]
        self.DP = np.concatenate([self.P, self.sat_P])
        self.Dt = np.concatenate([self.t_ign, st])
        self.Dsz = np.concatenate([self.size, self.size[par] * self.sat_szf])
        self.Dkind = np.concatenate([self.kind, self.kind[par]])
        self.Dmain = np.concatenate([np.ones(n, bool), np.zeros(m, bool)])
        self.famp = np.array([0.2, 0.13, 0.07])
        fl = self.fl
        self.fi = fl[:, 0].astype(np.int64)
        self.fj = fl[:, 1].astype(np.int64)
        self.fk = fl[:, 4].astype(np.int64)
        self.f_l = self.frame_of(fl[:, 2])
        self.f_a = self.frame_of(fl[:, 3])
        A = self.P[self.fi]
        B = self.P[self.fj]
        self.omega = np.arccos(np.clip(np.sum(A * B, 1), -1, 1))
        self.len_km = self.omega * R_KM
        lift = np.where(self.fk == F_LEAP, 0.15, np.where(self.fk == F_SEA, 0.08,
                        np.where(self.fk == F_THROW, 0.075, 0.035)))
        self.hmax = np.minimum(self.len_km * lift, 1100.0) / R_KM
        self.nseg = np.clip((self.len_km / 14.0).astype(int), 8, 240)
        # a flight along a chain (both ends on ranges, rivers or coasts) is drawn a little
        # brighter than a side hop to a lonely fire
        kk = self.kind
        chain = (kk[self.fi] <= K_RIVER) & (kk[self.fj] <= K_RIVER)
        self.fgain = np.where(self.fk == F_LEAP, 1.5, np.where(self.fk == F_THROW, 1.7, np.where(chain, 1.0, 0.7)))
        self.tau = np.where(self.fk == F_LEAP, 4.5, np.where(self.fk == F_SEA, 3.5,
                            np.where(self.fk == F_THROW, 3.5, 2.5)))
        self.ltail = np.where(self.fk == F_LEAP, 170.0, np.where(self.fk == F_THROW, 32.0,
                              np.where(self.fk == F_SEA, np.minimum(0.25 * self.len_km, 80.0),
                                       np.minimum(0.22 * self.len_km, 40.0))))
        self.pulses = [(0, self.t0, 1.0)]        # the first beacon flares when the choir enters

    def arc_points(self, k, u):
        A = self.P[self.fi[k]]
        B = self.P[self.fj[k]]
        om = self.omega[k]
        so = math.sin(om)
        if so < 1e-9:
            d = np.outer(1 - u, A) + np.outer(u, B)
        else:
            d = (np.outer(np.sin((1 - u) * om), A) + np.outer(np.sin(u * om), B)) / so
        d /= np.linalg.norm(d, axis=1)[:, None]
        h = self.hmax[k] * np.sin(np.pi * u) + 0.8 / R_KM
        return d * (1.0 + h)[:, None]

    @staticmethod
    def _px(sig, I, scale, lo=0.45):
        """Scale full-res sigmas to the render scale; keep energy when clamping thin lines."""
        s = np.asarray(sig, np.float64) * scale
        k = np.where(s < lo, s / lo, 1.0)
        return np.maximum(s, lo), np.asarray(I, np.float64) * k

    @staticmethod
    def _spot(img, uv, peak, sigma_full, col, scale):
        """Gaussian spots with given PEAK radiance (per point) and full-res sigma (scalar or per point)."""
        if len(uv) == 0:
            return
        s = np.asarray(sigma_full, np.float64) * scale * np.ones(len(uv))
        lo = 0.5
        k = np.where(s >= lo, 1.0, (s / lo) ** 2)
        s = np.maximum(s, lo)
        e = np.asarray(peak, np.float64) * k * 2 * np.pi * s * s
        col = np.asarray(col, np.float64)
        rgb = e[:, None] * (col[None, :] if col.ndim == 1 else col)
        G.splat_points(img, uv[:, 0].copy(), uv[:, 1].copy(), np.ascontiguousarray(rgb), s)

    @staticmethod
    def view_terms(cam, X):
        """cos(view angle at the surface), distance factor, limb fade."""
        rX = np.linalg.norm(X, axis=1)
        V = cam.pos[None, :] - X
        dist = np.linalg.norm(V, axis=1)
        cosv = np.sum(X * V, 1) / (rX * dist)
        near = np.clip(0.9 * np.linalg.norm(cam.pos) / np.maximum(dist, 1e-3), 0.45, 1.6)
        wl = np.clip((cosv + 0.02) / 0.34, 0, 1)
        wl = wl * wl * (3 - 2 * wl)
        return cosv, near, wl

    @staticmethod
    def dayfade(Pn, sun, keep):
        """The sun takes over from the beacons: 1 at night, `keep` in full sun; also returns the
        daylight fraction (0..1) for paling the colour."""
        el = np.degrees(np.arcsin(np.clip(Pn @ np.asarray(sun), -1, 1)))
        d = np.clip((el + 1.5) / 7.0, 0, 1)
        d = d * d * (3 - 2 * d)
        return 1.0 - (1.0 - keep) * d, d

    def draw(self, img, cam, t, gain=1.0, scale=1.0, calm=None, t_anim=None, embers=True, sun=None,
             day_keep=0.05, sparks=True, planet=None, light=1.0):
        """Additively draw the fires (and the embers in flight) as they are at frame t.
        planet: the rendered surface (HDR, before lights) -- if given, the fires also light the
        land around them (warm light on moonlit snow, rock and cloud)."""
        ta = t if t_anim is None else t_anim
        if embers:
            self._draw_embers(img, cam, t, gain, scale, calm, sparks)
        self._draw_fires(img, cam, t, ta, gain, scale, calm, sun, day_keep, sparks and embers,
                         planet, light)

    # ------------------------------------------------------------ fires ---
    def _draw_fires(self, img, cam, t, ta, gain, scale, calm, sun, day_keep, sparks, planet=None, light=1.0):
        lit = np.where(self.Dt <= t)[0]
        if len(lit) == 0:
            return
        X = self.DP[lit] * (1.0 + 0.6 / R_KM)
        uv, z = cam.project(X)
        vis = G.visible(cam.pos, X) & (z > 1e-3)
        vis &= (uv[:, 0] > -40) & (uv[:, 0] < cam.W + 40) & (uv[:, 1] > -40) & (uv[:, 1] < cam.H + 40)
        lit, uv, X = lit[vis], uv[vis], X[vis]
        if len(lit) == 0:
            return
        age = t - self.Dt[lit]
        # catching: a bright flare, then the fire builds to its steady size
        flare = np.exp(-age / 2.6)
        glow = np.exp(-age / 9.0)
        grow = 0.35 + 0.65 * np.clip(age / 10.0, 0, 1) ** 0.7
        fl = 1.0 + (self.famp[None, :] * np.sin(ta * self.fq[lit] + self.fph[lit])).sum(1)
        for (ni, tp, amp) in self.pulses:
            m = lit == ni
            if m.any() and t >= tp:
                flare[m] += amp * 2.2 * math.exp(-(t - tp) / 5.0)
                glow[m] += amp * 1.0 * math.exp(-(t - tp) / 14.0)
        cosv, near, wl = self.view_terms(cam, X)
        sz = self.Dsz[lit]
        temp = self.Dtemp[lit]
        dim = (0.55 + 0.45 * np.clip(near, 0, 1.2)) * (0.1 + 0.9 * wl) * gain
        if calm is not None:
            dim = dim * calm(uv[:, 1])
        pale = np.zeros(len(lit))
        if sun is not None:
            f, pale = self.dayfade(X / np.linalg.norm(X, axis=1)[:, None], sun, day_keep)
            dim = dim * f
        hot = srgb('#FFC56B')
        mid = srgb('#FF8A2A')
        deep = srgb('#C2410C')
        corec = srgb('#FFE9BE')
        paleC = srgb('#FFF3DA')
        # colour: big fires burn hotter; each fire has its own temperature, and it breathes
        # with the flicker (brighter = yellower); in daylight it pales
        heat = np.clip(0.3 + 0.28 * np.log(np.maximum(sz, 0.05)) + 0.45 * (temp - 0.5) + 0.5 * (fl - 1.0), 0, 1)[:, None]
        core_c = mid[None, :] * (1 - heat) + corec[None, :] * heat
        w = np.clip(0.25 + 0.55 * temp + 1.2 * (fl - 1.0), 0, 1)[:, None]
        flame = deep[None, :] * (1 - w) ** 2 + mid[None, :] * 2 * w * (1 - w) + hot[None, :] * w * w
        pw = 0.6 * pale[:, None]
        flame = flame * (1 - pw) + paleC[None, :] * pw
        core_c = core_c * (1 - pw) + paleC[None, :] * pw
        # near the limb the fires are seen through the long air: dimmer and redder
        ext = np.clip(cosv / 0.25, 0.0, 1.0)[:, None]
        tint = np.array([1.0, 0.78, 0.55]) * (1 - ext) + ext
        big = np.clip(sz, 0.1, 4.0)
        core = (3.4 * sz * fl * grow + 24.0 * flare * np.sqrt(sz)) * dim
        core = np.where(self.Dkind[lit] == K_ORIGIN, core * 1.8, core)
        halo = (0.3 * sz * fl * grow + 1.6 * glow * np.sqrt(sz)) * dim
        pool = (0.03 * sz * grow + 0.12 * glow) * dim
        nc = np.clip(near, 0.6, 1.4)
        s_core = 0.55 + 0.07 * np.sqrt(big)
        s_halo = (1.15 + 0.3 * np.sqrt(big)) * nc
        s_pool = (3.5 + 1.2 * np.sqrt(big)) * nc
        self._spot(img, uv, core, s_core, core_c * tint, scale)
        self._spot(img, uv, halo, s_halo, flame * tint, scale)
        self._spot(img, uv, pool, s_pool, deep[None, :] * tint, scale)
        om = np.where((self.Dkind[lit] == K_ORIGIN) & self.Dmain[lit])[0]
        if len(om):
            fb = 0.0
            for (ni, tp, amp) in self.pulses:
                if ni == 0 and t >= tp:
                    fb += amp * math.exp(-(t - tp) / 7.0)
            o = om[:1]
            st_ = 1.0 + 0.25 * (fl[o] - 1.0)
            # steady: the source reads as the biggest fire on the planet; burst: the flare
            self._spot(img, uv[o], dim[o] * (1.1 * st_ + 6.0 * fb), 3.6 * nc[o], hot[None, :] * tint[o], scale)
            self._spot(img, uv[o], dim[o] * (0.03 * st_ + 0.16 * fb), 26.0 * nc[o], mid[None, :] * tint[o], scale)
        if planet is not None and light > 0:
            # firelight on the land: each fire lights ~30 km around it (the first beacon far more,
            # and its flare floods the range); the light is multiplied into the moonlit surface
            # so snow, rock and cloud catch it and the dark sea does not
            dist = np.linalg.norm(cam.pos[None, :] - X, axis=1) * R_KM
            ppk = cam.f / np.maximum(dist, 1.0)
            r_km = 16.0 + 8.0 * np.sqrt(big) + 14.0 * glow * np.sqrt(sz)
            org = (self.Dkind[lit] == K_ORIGIN) & self.Dmain[lit]
            if org.any():
                r_km = np.where(org, 55.0 + 60.0 * np.minimum(glow, 1.5), r_km)
            hi = np.where(self.Dkind[lit] == K_RIDGE, 1.4, 1.0)       # snow and rock on the heights
            I = (1.3 * sz * fl * grow * hi + 2.0 * glow * np.sqrt(sz)) * dim * light
            I = np.where(org, I * 0.8, I)
            if sun is not None:
                I = I * (1.0 - pale) ** 2             # the sun takes over: firelight goes first
            ds = 4
            H, W = img.shape[:2]
            Lb = np.zeros(((H + ds - 1) // ds, (W + ds - 1) // ds, 3), np.float32)
            sg = np.maximum(r_km * ppk / ds, 0.6)
            col = (flame * 0.7 + deep[None, :] * 0.3) * tint
            e = I * 2 * np.pi * sg * sg
            G.splat_points(Lb, uv[:, 0] / ds, uv[:, 1] / ds, np.ascontiguousarray(e[:, None] * col), sg)
            Lf = cv2.resize(Lb, (W, H), interpolation=cv2.INTER_LINEAR)
            # the light falls on moonlit ground; where twilight already lights the land it adds
            # little (a cap keeps a dawn-bright surface from blowing the fire up into a blob)
            img += np.minimum(planet, 0.022) * Lf + Lf * 0.0006
            # the first beacon's flare washes the range around it: snow and cloud tops catch it
            fb = 0.0
            for (ni, tp, amp) in self.pulses:
                if ni == 0 and t >= tp:
                    fb += amp * math.exp(-(t - tp) / 9.0)
            om = np.where(org)[0]
            if fb > 0.01 and len(om):
                o = om[0]
                Lw = np.zeros_like(Lb)
                rw = (55.0 + 110.0 * fb) * ppk[o] / ds
                ew = 12.0 * fb * float(dim[o]) * light * 2 * np.pi * rw * rw
                G.splat_points(Lw, uv[o:o + 1, 0] / ds, uv[o:o + 1, 1] / ds,
                               np.ascontiguousarray((ew * (hot * 0.6 + mid * 0.4))[None, :]), np.array([rw]))
                Lw = cv2.resize(Lw, (W, H), interpolation=cv2.INTER_LINEAR)
                img += np.minimum(planet, 0.04) * Lw
        if sparks:
            mm = self.Dmain[lit]
            self._ignition_sparks(img, cam, t, lit[mm], age[mm], dim[mm], scale)

    def _ignition_sparks(self, img, cam, t, lit, age, dim, scale):
        """A small burst of sparks thrown up when a fire catches (and the first beacon's flare)."""
        sel = np.where((age < 16.0) & (self.size[lit] > 1.1))[0]
        items = [(int(lit[k]), float(age[k]), float(dim[k]), 5) for k in sel]
        for (ni, tp, amp) in self.pulses:
            if 0 <= t - tp < 60 and (lit == ni).any():
                d0 = float(dim[np.where(lit == ni)[0][0]])
                items.append((ni, t - tp, d0 * 0.8, 36))
        if not items:
            return
        X0, X1, E = [], [], []
        for (ni, a, dm, nsp) in items:
            rng = np.random.default_rng(ni * 7919 + nsp)
            up = self.P[ni]
            e1 = np.cross(up, [0.0, 0.0, 1.0])
            e1 /= np.linalg.norm(e1) + 1e-12
            e2 = np.cross(up, e1)
            big_ = nsp > 20
            v_up = rng.uniform(3.0, 7.0, nsp) if big_ else rng.uniform(4.0, 9.0, nsp)       # km per frame
            v_lat = rng.normal(0, 0.6, (nsp, 2)) + (np.array([1.1, 0.4]) if big_ else 0.0)  # a wind
            v_lat = v_lat * (1.0 if big_ else 1.2)
            life = rng.uniform(10.0, 22.0, nsp) if big_ else rng.uniform(8.0, 16.0, nsp)
            delay = rng.uniform(0.0, 38.0, nsp) if big_ else rng.uniform(0.0, 3.0, nsp)
            aa = a - delay
            m = (aa >= 0) & (aa < life)
            if not m.any():
                continue
            g = 0.25 if big_ else 0.6                                               # km per frame^2

            def pos(tt):
                h = v_up[m] * tt - 0.5 * g * tt * tt
                lat = v_lat[m] * tt[:, None]
                return (up[None, :] * (1.0 + (0.6 + np.maximum(h, 0.0)) / R_KM)[:, None]
                        + (lat[:, :1] * e1[None, :] + lat[:, 1:] * e2[None, :]) / R_KM)
            tt = aa[m]
            X0.append(pos(np.maximum(tt - 1.2, 0.0)))
            X1.append(pos(tt))
            E.append(dm * (1.0 - tt / life[m]) ** 1.6 * (1.0 if nsp < 20 else 1.3))
        if not X0:
            return
        X0 = np.concatenate(X0)
        X1 = np.concatenate(X1)
        E = np.concatenate(E)
        uv0, z0 = cam.project(X0)
        uv1, z1 = cam.project(X1)
        vis = G.visible(cam.pos, X1) & (z1 > 1e-3) & (z0 > 1e-3)
        if not vis.any():
            return
        nv = int(vis.sum())
        xs = np.empty(2 * nv)
        ys = np.empty(2 * nv)
        xs[0::2] = uv0[vis, 0]
        xs[1::2] = uv1[vis, 0]
        ys[0::2] = uv0[vis, 1]
        ys[1::2] = uv1[vis, 1]
        I = np.repeat(E[vis] * 5.0, 2)
        I[0::2] *= 0.35
        sg, I = self._px(np.full(len(xs), 0.5), I, scale)
        col = srgb('#FFB050')
        st = np.arange(0, len(xs) + 1, 2, dtype=np.int64)
        G.splat_polyline(img, xs, ys, np.outer(I, col), sg, st, np.ones(len(xs), np.uint8))

    # ----------------------------------------------------------- embers ---
    def _draw_embers(self, img, cam, t, gain, scale, calm, sparks):
        s_now = float(self.s_of(t))
        act = np.where((self.f_l <= t) & (t <= self.f_a + 26.0))[0]
        if len(act) == 0:
            return
        xs, ys, cols, sig, oks, starts = [], [], [], [], [], [0]
        gxs, gys, gcols, gsig, goks, gstarts = [], [], [], [], [], [0]
        heads = []
        spark_items = []
        hot = srgb('#FFC56B')
        mid = srgb('#FF8A2A')
        deep = srgb('#C2410C')
        pale = srgb('#FFF1D6')
        cn = np.linalg.norm(cam.pos)
        for k in act:
            sl, sa = self.fl[k, 2], self.fl[k, 3]
            uh = min(1.0, max(0.0, (s_now - sl) / max(sa - sl, 1e-6)))
            if uh <= 0.0:
                continue
            n = self.nseg[k]
            m = max(3, int(math.ceil(n * uh)) + 2)
            u = np.linspace(0.0, uh, m)
            X = self.arc_points(k, u)
            uv, z = cam.project(X)
            vis = G.visible(cam.pos, X) & (z > 1e-3)
            if not vis.any():
                continue
            t_pass = self.frame_of(sl + u * (sa - sl))
            back = (uh - u) * self.len_km[k] / self.ltail[k]
            # a short comet: the tail falls off steeply behind the head and ends; it never reaches
            # back to the fire it left
            b = (np.exp(-np.maximum(t - t_pass, 0.0) / self.tau[k]) * np.exp(-back ** 1.5)
                 * np.clip(1.0 - back / 3.5, 0.0, 1.0))
            # gone about a second after it lands
            b *= np.clip((self.f_a[k] + 24.0 - t) / 8.0, 0.0, 1.0)
            if b.max() < 0.01:
                continue
            fg = self.fgain[k] * gain
            leap = self.fk[k] in (F_LEAP, F_THROW)
            cosv, near, wl = self.view_terms(cam, X)
            rX = np.linalg.norm(X, axis=1)
            lifted = np.clip(((rX - 1.0) * R_KM - 40.0) / 200.0, 0, 1)
            vf = (0.5 + 0.5 * np.clip(near, 0, 1.2)) * (0.3 + 0.7 * np.maximum(wl, lifted))
            I = b * fg * vf * (7.0 if leap else 5.0)
            if calm is not None:
                I = I * calm(uv[:, 1])
            bb = b[:, None]
            col = deep[None, :] * (1 - bb) ** 2 + mid[None, :] * 2 * bb * (1 - bb) + hot[None, :] * bb * bb
            w = np.clip(0.55 * near, 0.45, 1.0) * (1.25 if leap else 1.0) * (0.55 + 0.45 * b)
            xs.append(uv[:, 0])
            ys.append(uv[:, 1])
            cols.append(I[:, None] * col)
            sig.append(w)
            oks.append(vis.astype(np.uint8))
            starts.append(starts[-1] + len(u))
            if uh < 1.0 and vis[-1]:
                heads.append((uv[-1], fg * vf[-1] * (1.6 if leap else 1.0) * (calm(uv[-1:, 1])[0] if calm is not None else 1.0)))
            if sparks and uh < 1.0:
                spark_items.append(k)
        if xs:
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            cols = np.concatenate(cols)
            sig = np.concatenate(sig)
            oks = np.concatenate(oks)
            st = np.asarray(starts, np.int64)
            sg, _ = self._px(sig * 3.2, np.ones(len(sig)), scale)
            k2 = np.where(sig * 3.2 * scale < 0.45, sig * 3.2 * scale / 0.45, 1.0)[:, None]
            G.splat_polyline(img, xs, ys, cols * 0.07 * k2, sg, st, oks)
            sg, _ = self._px(sig, np.ones(len(sig)), scale)
            k1 = np.where(sig * scale < 0.45, sig * scale / 0.45, 1.0)[:, None]
            G.splat_polyline(img, xs, ys, cols * k1, sg, st, oks)
        if heads:
            uvh = np.array([h[0] for h in heads])
            pk = np.array([h[1] for h in heads])
            self._spot(img, uvh, 9.0 * pk, 0.7, pale, scale)
            self._spot(img, uvh, 1.1 * pk, 2.2, hot, scale)
        if spark_items:
            self._ember_sparks(img, cam, t, spark_items, gain, calm, scale)

    def _ember_sparks(self, img, cam, t, items, gain, calm, scale):
        X0, X1, E = [], [], []
        for k in items:
            L = self.len_km[k]
            nsp = int(min(40, max(3, L / 40.0)))
            rng = np.random.default_rng(int(k) * 7919 + self.seed)
            us = np.sort(rng.random(nsp))
            life = rng.uniform(4.0, 10.0, nsp)
            sl, sa = self.fl[k, 2], self.fl[k, 3]
            tpass = self.frame_of(sl + us * (sa - sl))
            a = t - tpass
            m = (a >= 0) & (a < life)
            if not m.any():
                continue
            us, life, a = us[m], life[m], a[m]
            rs = rng.normal(0, 1, (nsp, 3))[m]
            P0 = self.arc_points(k, us)
            vel = rs * 2.2 / R_KM - P0 * (0.9 / R_KM)
            X1.append(P0 + vel * a[:, None])
            X0.append(P0 + vel * np.maximum(a - 1.2, 0)[:, None])
            E.append((1.0 - a / life) ** 1.5 * 2.2 * self.fgain[k])
        if not X0:
            return
        X0 = np.concatenate(X0)
        X1 = np.concatenate(X1)
        E = np.concatenate(E) * gain
        uv0, z0 = cam.project(X0)
        uv1, z1 = cam.project(X1)
        vis = G.visible(cam.pos, X1) & (z1 > 1e-3) & (z0 > 1e-3)
        if not vis.any():
            return
        if calm is not None:
            E = E * calm(uv1[:, 1])
        nv = int(vis.sum())
        xs = np.empty(2 * nv)
        ys = np.empty(2 * nv)
        xs[0::2] = uv0[vis, 0]
        xs[1::2] = uv1[vis, 0]
        ys[0::2] = uv0[vis, 1]
        ys[1::2] = uv1[vis, 1]
        I = np.repeat(E[vis], 2)
        sg, I = self._px(np.full(len(xs), 0.5), I, scale)
        col = srgb('#FFA040')
        st = np.arange(0, len(xs) + 1, 2, dtype=np.int64)
        G.splat_polyline(img, xs, ys, np.outer(I, col), sg, st, np.ones(len(xs), np.uint8))


_SRGB = {}


def srgb(h):
    if h not in _SRGB:
        import look
        _SRGB[h] = look.hexrgb(h).astype(np.float64)
    return _SRGB[h]
