"""The web of fire on the map: beacon nodes, the threads between them, and when each one burns.

Adapted from the GLOBE department's web.py (seeded spread from the Himalaya over a Delaunay graph,
2-3 children per beacon with angular diversity, ocean leaps, late answers, cross-links), but on
the flat sheet:
  * the graph is planar (map coordinates) and never crosses the map's seam;
  * long sea crossings are designated LEAPS (sparks that fly above the paper); short straits are
    ordinary threads burned across the paper;
  * the spread runs in "travel time" (map units at unit speed) and is then remapped to frames by
    frame = T0 + a * s^p (p < 1), so the first generations are slow and heavy and the rest
    accelerates without seams.

All times are v2 global frames. Deterministic and cached.
"""
import heapq
import json
import math
import os

import numpy as np
from scipy.spatial import Delaunay, cKDTree

import geo
import ink
from noise import wobble1d
import places_globe as places

T0 = 1920.0            # the first beacon flares
T_FIRST = 1946.0       # its first thread lands (sets the remap scale)
T_ALL = 2054.0         # the last beacon is lit by about here
EVEREST = (27.99, 86.93)

# ocean leaps on the flat map (from, to) as (lat, lon); none crosses the seam at 169 W
LEAPS = [
    ((14.70, -17.40), (-8.00, -34.90)),     # Dakar -> Recife
    ((38.78, -9.50), (38.47, -28.40)),      # Cabo da Roca -> Azores
    ((38.47, -28.40), (40.70, -74.00)),     # Azores -> New York
    ((64.10, -21.90), (64.20, -51.70)),     # Iceland -> Greenland
    ((51.90, -8.50), (47.60, -52.70)),      # Ireland -> Newfoundland
    ((28.27, -16.64), (18.40, -66.10)),     # Canaries -> Caribbean
    ((-33.96, 18.40), (-15.97, -5.71)),     # Cape -> St Helena
    ((-15.97, -5.71), (-22.90, -43.20)),    # St Helena -> Rio
    ((57.10, -2.10), (62.00, -6.90)),       # Aberdeen -> Faroes
    ((62.00, -6.90), (64.10, -21.90)),      # Faroes -> Iceland
    ((69.60, 18.90), (78.20, 15.60)),       # Tromso -> Svalbard
    ((-8.11, 112.92), (-12.50, 130.80)),    # Java -> Darwin
    ((-6.30, 145.00), (-19.30, 146.80)),    # New Guinea -> Queensland
    ((-33.90, 151.20), (-36.80, 174.80)),   # Sydney -> Auckland
    ((-27.50, 153.00), (-21.50, 165.50)),   # Brisbane -> New Caledonia
    ((-21.50, 165.50), (-17.70, 178.00)),   # New Caledonia -> Fiji
    ((14.60, 121.00), (13.40, 144.80)),     # Manila -> Guam
    ((37.80, -122.40), (21.30, -157.90)),   # San Francisco -> Hawaii
    ((-33.40, -70.60), (-27.10, -109.40)),  # Santiago -> Easter Island
    ((-27.10, -109.40), (-17.65, -149.43)), # Easter Island -> Tahiti
    ((-0.20, -78.50), (-0.80, -91.10)),     # Quito -> Galapagos
    ((25.80, -80.20), (32.30, -64.80)),     # Miami -> Bermuda
    ((-31.90, 115.90), (-12.20, 96.80)),    # Perth -> Cocos
    ((6.81, 80.50), (4.20, 73.50)),         # Sri Lanka -> Maldives
    ((4.20, 73.50), (-4.60, 55.40)),        # Maldives -> Seychelles
    ((-4.60, 55.40), (-6.80, 39.30)),       # Seychelles -> Zanzibar
    ((-18.90, 47.50), (-20.30, 57.60)),     # Madagascar -> Mauritius
    ((-54.80, -68.30), (-51.70, -59.00)),   # Tierra del Fuego -> Falklands
    ((-51.70, -59.00), (-54.30, -36.50)),   # Falklands -> South Georgia
    ((-15.97, -5.71), (-37.10, -12.30)),    # St Helena -> Tristan da Cunha
]


def ll2xy(lat, lon):
    return np.stack([geo.lon2x(lon), geo.yproj(lat)], -1)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class MapWeb:
    def __init__(self, seed=7, cache=True):
        self.seed = seed
        path = os.path.join(geo.CACHE, f'web_{seed}.npz')
        if cache and os.path.exists(path):
            d = np.load(path)
            self.P, self.kind, self.s_ign, self.arcs = d['P'], d['kind'], d['s_ign'], d['arcs']
        else:
            self.rng = np.random.default_rng(seed)
            self._nodes()
            self._graph()
            self._spread()
            keep = np.isfinite(self.s_ign)
            if not keep.all():
                self.P, self.kind = self.P[keep], self.kind[keep]
                self.rng = np.random.default_rng(seed)
                self._graph()
                self._spread()
            if cache:
                np.savez(path, P=self.P, kind=self.kind, s_ign=self.s_ign, arcs=self.arcs)
        self._timing()
        self._paths()

    # ------------------------------------------------------------ nodes ---
    def _nodes(self):
        rng = self.rng
        F = geo.fields()
        land = F['land']
        V = [ll2xy(*EVEREST)]
        kind = [0]
        lat_ok = lambda la: geo.LAT_BOT + 1.5 < la < geo.LAT_TOP - 2.0

        def far(v, need):
            return np.min(np.hypot(*(np.asarray(V) - v[None, :]).T)) > need

        for la, lo in places.HIGH:
            if not lat_ok(la):
                continue
            v = ll2xy(la, lo)
            if abs(v[0]) > 178.5:
                continue
            if far(v, 2.1):
                V.append(v)
                kind.append(1)
        d = json.load(open(os.path.join(geo.ROOT, 'assets', 'data', 'ne_10m_populated_places_simple.geojson')))
        cities = []
        for ft in d['features']:
            pr = ft['properties']
            pop = pr.get('pop_max') or 0
            if pop >= 200000:
                cities.append((pop, pr['latitude'], pr['longitude']))
        cities.sort(reverse=True)
        for pop, la, lo in cities:
            if not lat_ok(la):
                continue
            v = ll2xy(la, lo)
            if abs(v[0]) > 178.5:
                continue
            need = 2.6 if pop > 3e6 else 3.2
            if far(v, need):
                V.append(v)
                kind.append(2)
        # fill empty land with hill beacons, weighted toward relief
        rel = F['rug'] / (F['rug'].max() + 1e-9)
        w = (land > 0.9).astype(np.float64) * (0.3 + 3.0 * rel)
        lat_c = 90.0 - (np.arange(geo.EQ_H) + 0.5) / geo.EQ_H * 180.0
        w *= np.cos(np.radians(lat_c))[:, None]
        w[(lat_c > geo.LAT_TOP - 3) | (lat_c < geo.LAT_BOT + 2)] = 0.0
        # no fillers on the Greenland ice
        lon_c = (np.arange(geo.EQ_W) + 0.5) / geo.EQ_W * 360.0 - 180.0
        gl = (lat_c[:, None] > 60) & (lon_c[None, :] > -60) & (lon_c[None, :] < -22) & (lat_c[:, None] < 81)
        w[gl] *= 0.0
        p = w.ravel() / w.sum()
        cand = rng.choice(len(p), size=8000, p=p)
        yy, xx = np.divmod(cand, geo.EQ_W)
        la = 90.0 - (yy + rng.random(len(yy))) / geo.EQ_H * 180.0
        lo = (xx + rng.random(len(xx))) / geo.EQ_W * 360.0 - 180.0
        for a, b in zip(la, lo):
            v = ll2xy(a, b)
            if abs(v[0]) > 178.0:
                continue
            if far(v, 4.6):
                V.append(v)
                kind.append(3)
        self.P = np.asarray(V, np.float64)
        self.kind = np.asarray(kind, np.int32)

    # ------------------------------------------------------------ graph ---
    def _landfrac(self, a, b, n=24):
        u = np.linspace(0, 1, n)
        pts = a[None, :] * (1 - u[:, None]) + b[None, :] * u[:, None]
        lat = geo.ilat(pts[:, 1])
        lon = geo.x2lon(pts[:, 0])
        return float(np.mean(geo.sample_eq(geo.fields()['land'], lat, lon) > 0.5))

    def _graph(self, max_len=9.5):
        tri = Delaunay(self.P)
        edges = set()
        for s in tri.simplices:
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
                edges.add((min(a, b), max(a, b)))
        nb = [[] for _ in range(len(self.P))]
        keep = set()
        for a, b in edges:
            L = float(np.hypot(*(self.P[a] - self.P[b])))
            if L > max_len:
                continue
            lf = self._landfrac(self.P[a], self.P[b])
            sea_len = (1 - lf) * L
            if sea_len > 4.2:
                continue
            nb[a].append(b)
            nb[b].append(a)
            keep.add((a, b))
        self.nb = nb
        self.edges = keep

    # ----------------------------------------------------------- spread ---
    def _spread(self):
        """Uniform-speed spread in travel units (1 unit of time = 1 map degree at unit speed)."""
        rng = self.rng
        P = self.P
        n = len(P)
        s_ign = np.full(n, np.inf)
        parent = np.full(n, -1)
        claimed = np.zeros(n, bool)
        gen = np.zeros(n, np.int64)
        arcs = []          # (i, j, s_launch, s_arrive, kind)  kind 0 hop, 1 leap, 2 late, 3 cross
        tree = cKDTree(P)
        leap_from = {}
        for a, b in LEAPS:
            ia = int(tree.query(ll2xy(*a))[1])
            ib = int(tree.query(ll2xy(*b))[1])
            if ia != ib:
                leap_from.setdefault(ia, []).append(ib)
        origin = 0
        s_ign[origin] = 0.0
        claimed[origin] = True
        heap = [(0.0, origin)]
        while heap:
            t, i = heapq.heappop(heap)
            cand = [j for j in self.nb[i] if not claimed[j]]
            if i == origin:
                k, delay = 3, 0.12
            else:
                k = 2 if rng.random() < 0.45 else 3
                delay = rng.uniform(0.5, 1.6)
            if cand:
                fwd = None
                if parent[i] >= 0:
                    fwd = P[i] - P[parent[i]]
                    fwd /= np.linalg.norm(fwd) + 1e-12
                sc = []
                for j in cand:
                    dv = P[j] - P[i]
                    dv /= np.linalg.norm(dv) + 1e-12
                    s = rng.normal(0, 0.35)
                    if fwd is not None:
                        s += 0.8 * float(dv @ fwd)
                    sc.append(s)
                chosen = []
                for o in np.argsort(sc)[::-1]:
                    j = cand[o]
                    ok = True
                    for c in chosen:
                        a = P[j] - P[i]
                        b = P[c] - P[i]
                        if (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12) > 0.8:
                            ok = False
                    if ok:
                        chosen.append(j)
                    if len(chosen) >= k:
                        break
                for m, j in enumerate(chosen):
                    L = float(np.linalg.norm(P[j] - P[i]))
                    if i == origin:
                        tl = t + delay + (0.0, 1.3, 2.9)[min(m, 2)]
                    else:
                        tl = t + delay + m * rng.uniform(0.2, 0.7)
                    ta = tl + L * rng.uniform(0.85, 1.2)
                    claimed[j] = True
                    parent[j] = i
                    gen[j] = gen[i] + 1
                    s_ign[j] = ta
                    arcs.append((i, j, tl, ta, 0))
                    heapq.heappush(heap, (ta, j))
            for j in leap_from.get(i, []):
                if claimed[j]:
                    continue
                L = float(np.linalg.norm(P[j] - P[i]))
                tl = t + delay + rng.uniform(1.0, 2.5)
                ta = tl + 1.5 + L * 0.62
                claimed[j] = True
                parent[j] = i
                gen[j] = gen[i] + 1
                s_ign[j] = ta
                arcs.append((i, j, tl, ta, 1))
                heapq.heappush(heap, (ta, j))
            if not heap:
                lit = np.where(np.isfinite(s_ign))[0]
                unl = np.where(~claimed)[0]
                if len(unl) == 0:
                    break
                lt = cKDTree(P[lit])
                dd, ii = lt.query(P[unl], k=min(4, len(lit)))
                best = None
                for row, j in enumerate(unl):
                    for kk in range(dd.shape[1]):
                        i2 = lit[ii[row, kk]]
                        L = float(np.linalg.norm(P[i2] - P[j]))
                        if L > 16.0:
                            continue
                        tt = s_ign[i2] + 1.0 + L
                        if best is None or tt < best[0]:
                            best = (tt, i2, j, L)
                if best is None:
                    break
                tt, i2, j, L = best
                tl = max(tt - L, s_ign[i2] + 1.0) + rng.uniform(0, 1.5)
                ta = tl + L * rng.uniform(0.85, 1.2)
                claimed[j] = True
                parent[j] = i2
                gen[j] = gen[i2] + 1
                s_ign[j] = ta
                arcs.append((i2, j, tl, ta, 2))
                heapq.heappush(heap, (ta, j))
        # cross links close loops so the tree becomes lace
        conn = set((min(a[0], a[1]), max(a[0], a[1])) for a in arcs)
        for a, b in sorted(self.edges):
            if (a, b) in conn or not (np.isfinite(s_ign[a]) and np.isfinite(s_ign[b])):
                continue
            if rng.random() > 0.2:
                continue
            L = float(np.linalg.norm(P[a] - P[b]))
            i, j = (a, b) if s_ign[a] <= s_ign[b] else (b, a)
            tl = s_ign[j] + rng.uniform(1.0, 8.0)
            ta = tl + L * rng.uniform(0.9, 1.2)
            arcs.append((i, j, tl, ta, 3))
        self.s_ign = s_ign
        self.arcs = np.array(arcs, np.float64)

    # ----------------------------------------------------------- timing ---
    def _timing(self):
        """Travel time -> frames: frame = T0 + a s^p, pinned so the first landing is at T_FIRST and
        (nearly) everything is lit by T_ALL."""
        a = self.arcs
        s_first = float(np.min(a[a[:, 0] == 0, 3]))
        fin = np.isfinite(self.s_ign)
        s_all = float(np.percentile(self.s_ign[fin], 99.5))
        p = math.log((T_ALL - T0) / (T_FIRST - T0)) / math.log(s_all / s_first)
        k = (T_FIRST - T0) / s_first ** p
        self.p, self.k = p, k
        f = lambda s: T0 + k * np.power(np.maximum(s, 0.0), p)
        self.to_frame = f
        self.t_ign = np.where(fin, f(np.where(fin, self.s_ign, 0.0)), np.inf)
        self.t_ign[0] = T0 - 300.0          # the first beacon has burned since the cold mountain
        self.tl = f(a[:, 2])
        self.ta = f(a[:, 3])
        self.ai = a[:, 0].astype(np.int64)
        self.aj = a[:, 1].astype(np.int64)
        self.ak = a[:, 4].astype(np.int64)
        self.sl = a[:, 2]
        self.sa = a[:, 3]
        gen = np.full(len(self.P), 99, np.int64)
        gen[0] = 0
        for kx in np.argsort(self.sa):
            if self.ak[kx] in (0, 1, 2):
                gen[self.aj[kx]] = min(gen[self.aj[kx]], gen[self.ai[kx]] + 1)
        self.gen = gen

    def head_u(self, k, t):
        """Fraction of arc k burned at frame t (eased in travel time, then remapped)."""
        sl, sa = self.sl[k], self.sa[k]
        if t <= self.tl[k]:
            return 0.0
        if t >= self.ta[k]:
            return 1.0
        # invert the remap for the current travel time
        s = ((t - T0) / self.k) ** (1.0 / self.p)
        x = (s - sl) / max(sa - sl, 1e-9)
        x = min(max(x, 0.0), 1.0)
        return 0.55 * x + 0.45 * x * x * (3 - 2 * x)

    # ------------------------------------------------------------ paths ---
    def _paths(self):
        """Polyline for every thread (map XY), a gentle hand-drawn curve; leaps carry a height."""
        rng = np.random.default_rng(self.seed + 17)
        self.paths = []
        self.heights = []
        for k in range(len(self.ai)):
            A = self.P[self.ai[k]]
            B = self.P[self.aj[k]]
            d = B - A
            L = float(np.linalg.norm(d))
            nrm = np.array([-d[1], d[0]]) / max(L, 1e-9)
            leap = self.ak[k] == 1
            bend = rng.uniform(0.05, 0.13) * rng.choice([-1, 1]) * (0.5 if leap else 1.0)
            C = 0.5 * (A + B) + nrm * bend * L
            n = max(8, int(L / 0.06))
            u = np.linspace(0, 1, n)
            pts = ((1 - u) ** 2)[:, None] * A + (2 * u * (1 - u))[:, None] * C + (u ** 2)[:, None] * B
            if not leap:
                w = wobble1d(u * L, int(rng.integers(1 << 30)), 0.9, 0.05) * np.sin(np.pi * u)
                pts = pts + nrm[None, :] * w[:, None]
                h = np.full(n, 0.0)
            else:
                h = np.minimum(0.16 * L, 9.0) * np.sin(np.pi * u)
            self.paths.append(pts)
            self.heights.append(h)
        self.plen = np.array([float(ink.arclen(p)[-1]) for p in self.paths])

    def cut(self, t_cut):
        """No new edges after t_cut and no loop-closing cross-links (fire, not fibre): returns the
        kept-arc mask and node ignition times with the unreached beacons removed (inf)."""
        keep = (self.ak != 3) & (self.tl <= t_cut)
        t = np.full(len(self.P), np.inf)
        t[0] = self.t_ign[0]
        kk = np.where(keep)[0]
        for k in kk[np.argsort(self.ta[kk])]:
            j = self.aj[k]
            if np.isfinite(t[self.ai[k]]) and self.ta[k] < t[j]:
                t[j] = self.ta[k]
        # an arc whose source never lit is dropped too
        keep &= np.isfinite(t[self.ai])
        return keep, t

    def summary(self):
        fin = np.isfinite(self.t_ign)
        return dict(nodes=len(self.P), lit=int(fin.sum()), arcs=len(self.ai),
                    kinds=np.bincount(self.ak).tolist(), p=round(self.p, 3),
                    pct=np.percentile(self.t_ign[fin & (self.t_ign > T0)], [5, 25, 50, 75, 90, 99, 100]).round(1).tolist())


if __name__ == '__main__':
    w = MapWeb(cache=False)
    print(w.summary())
    # landings per 10 frames
    ta = np.sort(w.t_ign[np.isfinite(w.t_ign) & (w.t_ign > T0)])
    print(np.histogram(ta, bins=np.arange(1920, 2100, 10))[0].tolist())
    first = np.where(w.ai == 0)[0]
    print('first arcs', [(round(w.tl[k], 1), round(w.ta[k], 1)) for k in first])
