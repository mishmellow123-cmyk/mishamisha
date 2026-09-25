"""The golden web: beacon nodes + great-circle arcs of fire spreading from the Himalaya.

Deterministic (seeded) and stateless per frame: `Web.draw(img, cam, t)` renders the web as it
is at global frame t. Arc/ignition times are in global frames.
"""
import heapq
import json
import math
import os

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

import globe as G
import places

R_KM = G.R_KM
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def ll2v(lat, lon):
    lat = np.radians(np.asarray(lat, np.float64))
    lon = np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def gc_km(a, b):
    return np.arccos(np.clip(np.sum(a * b, -1), -1, 1)) * R_KM


def srgb_hex(h):
    import look
    return look.hexrgb(h).astype(np.float64)


GOLD = None


def palette():
    global GOLD
    if GOLD is None:
        GOLD = dict(core=srgb_hex('#FFE3A8'), gold=srgb_hex('#FFD27A'), amber=srgb_hex('#E8A33D'),
                    pale=srgb_hex('#FFF1C9'), deep=srgb_hex('#C9731F'), hearth=srgb_hex('#FFB85C'))
    return GOLD


class Web:
    def __init__(self, seed=5, t0=1760.0, speed_km=200.0, cache=True):
        self.seed = seed
        self.t0 = t0
        self.speed = speed_km
        self.rng = np.random.default_rng(seed)
        path = os.path.join(ROOT, 'renders', 'globe', 'cache', f'web_{seed}_{int(t0)}_{int(speed_km)}.npz')
        if cache and os.path.exists(path):
            d = np.load(path)
            self.P = d['P']
            self.kind = d['kind']
            self.t_ign = d['t_ign']
            self.arcs = d['arcs']
        else:
            self._nodes()
            self._graph()
            self._spread()
            if cache:
                np.savez(path, P=self.P, kind=self.kind, t_ign=self.t_ign, arcs=self.arcs)
        self._prep_arcs()

    # ------------------------------------------------------------ nodes ---
    def _nodes(self):
        rng = self.rng
        V = [ll2v(*places.ORIGIN)]
        kind = [0]
        for p in places.HIGH:
            v = ll2v(*p)
            if gc_km(np.asarray(V), v[None, :]).min() > 160.0:
                V.append(v)
                kind.append(1)
        # cities: largest first, thinned by distance
        d = json.load(open(os.path.join(ROOT, 'assets', 'data', 'ne_10m_populated_places_simple.geojson')))
        cities = []
        for ft in d['features']:
            pr = ft['properties']
            pop = pr.get('pop_max') or 0
            if pop >= 150000:
                cities.append((pop, pr['latitude'], pr['longitude']))
        cities.sort(reverse=True)
        tree_pts = list(V)
        for pop, la, lo in cities:
            v = ll2v(la, lo)
            dmin = gc_km(np.asarray(tree_pts), v[None, :]).min()
            need = 270.0 if pop > 3e6 else 330.0
            if dmin > need:
                tree_pts.append(v)
                V.append(v)
                kind.append(2)
        # fill empty land with hill beacons (weighted toward relief)
        import cv2
        mask = cv2.imread(os.path.join(ROOT, 'renders', 'globe', 'cache', 'land_mask_8k.png'), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (2048, 1024), interpolation=cv2.INTER_AREA)
        nm = np.load(os.path.join(ROOT, 'renders', 'globe', 'cache', 'maps.npz'))['normal'].astype(np.float32)
        relief = np.sqrt(nm[..., 0] ** 2 + nm[..., 1] ** 2)
        relief = cv2.GaussianBlur(relief, (0, 0), 3)
        w = (mask > 200).astype(np.float64) * (0.25 + relief / (relief.max() + 1e-9) * 4.0)
        lat_c = 90.0 - (np.arange(1024) + 0.5) / 1024 * 180.0
        w *= np.cos(np.radians(lat_c))[:, None]
        w[np.abs(lat_c) > 72] = 0.0     # no fillers on the ice caps' far edges
        p = w.ravel() / w.sum()
        cand = rng.choice(len(p), size=6000, p=p)
        yy, xx = np.divmod(cand, 2048)
        la = 90.0 - (yy + rng.random(len(yy))) / 1024 * 180.0
        lo = (xx + rng.random(len(xx))) / 2048 * 360.0 - 180.0
        for a, b in zip(la, lo):
            v = ll2v(a, b)
            if gc_km(np.asarray(tree_pts), v[None, :]).min() > 430.0:
                tree_pts.append(v)
                V.append(v)
                kind.append(3)
        self.P = np.asarray(V, np.float64)
        self.kind = np.asarray(kind, np.int32)

    # ------------------------------------------------------------ graph ---
    def _graph(self, max_km=1350.0):
        hull = ConvexHull(self.P)
        edges = set()
        for s in hull.simplices:
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
                edges.add((min(a, b), max(a, b)))
        nb = [[] for _ in range(len(self.P))]
        for a, b in edges:
            dk = gc_km(self.P[a], self.P[b])
            if dk <= max_km:
                nb[a].append(b)
                nb[b].append(a)
        self.nb = nb
        self.edges = edges

    # ----------------------------------------------------------- spread ---
    def _travel(self, dkm, leap=False):
        rng = self.rng
        if leap:
            return 12.0 + dkm / (self.speed * 0.8) * rng.uniform(0.9, 1.15)
        return 2.5 + dkm / self.speed * rng.uniform(0.85, 1.25)

    def _spread(self):
        rng = self.rng
        P = self.P
        n = len(P)
        t_ign = np.full(n, np.inf)
        parent = np.full(n, -1)
        claimed = np.zeros(n, bool)
        arcs = []          # (i, j, t_launch, t_arrive, kind)  kind 0 hop, 1 leap, 2 late, 3 cross
        tree = cKDTree(P)
        leap_from = {}
        for a, b in places.LEAPS:
            ia = tree.query(ll2v(*a))[1]
            ib = tree.query(ll2v(*b))[1]
            leap_from.setdefault(ia, []).append(ib)
        origin = 0
        t_ign[origin] = self.t0
        claimed[origin] = True
        heap = [(self.t0, origin)]
        # the first beacon throws three arcs at once (the choir)
        while heap:
            t, i = heapq.heappop(heap)
            cand = [j for j in self.nb[i] if not claimed[j]]
            if i == origin:
                k = 3
                delay = 4.0
            else:
                k = 2 if rng.random() < 0.45 else 3
                delay = rng.uniform(2.5, 7.0)
            if cand:
                if parent[i] >= 0:
                    fwd = P[i] - P[parent[i]]
                    fwd /= np.linalg.norm(fwd) + 1e-12
                else:
                    fwd = None
                sc = []
                for j in cand:
                    dvec = P[j] - P[i]
                    dvec /= np.linalg.norm(dvec) + 1e-12
                    s = rng.normal(0, 0.35)
                    if fwd is not None:
                        s += 0.8 * float(dvec @ fwd)
                    sc.append(s)
                order = np.argsort(sc)[::-1]
                chosen = []
                for o in order:
                    j = cand[o]
                    # angular diversity among siblings
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
                    tl = t + delay + m * rng.uniform(1.0, 3.0)
                    ta = tl + self._travel(gc_km(P[i], P[j]))
                    claimed[j] = True
                    parent[j] = i
                    t_ign[j] = ta
                    arcs.append((i, j, tl, ta, 0))
                    heapq.heappush(heap, (ta, j))
            for j in leap_from.get(i, []):
                if claimed[j]:
                    continue
                tl = t + delay + rng.uniform(4.0, 10.0)
                ta = tl + self._travel(gc_km(P[i], P[j]), leap=True)
                claimed[j] = True
                parent[j] = i
                t_ign[j] = ta
                arcs.append((i, j, tl, ta, 1))
                heapq.heappush(heap, (ta, j))
            if not heap:
                # late answers: light the unreached node that can be reached soonest
                best = None
                lit = np.where(np.isfinite(t_ign))[0]
                unl = np.where(~claimed)[0]
                if len(unl) == 0:
                    break
                lt = cKDTree(P[lit])
                dd, ii = lt.query(P[unl], k=min(4, len(lit)))
                for row, j in enumerate(unl):
                    for kk in range(dd.shape[1]):
                        i2 = lit[ii[row, kk]]
                        dkm = gc_km(P[i2], P[j])
                        if dkm > 2600.0:
                            continue
                        tt = t_ign[i2] + 6.0 + dkm / self.speed
                        if best is None or tt < best[0]:
                            best = (tt, i2, j, dkm)
                if best is None:
                    break
                tt, i2, j, dkm = best
                tl = max(tt - dkm / self.speed, t_ign[i2] + 6.0) + rng.uniform(0, 8)
                ta = tl + self._travel(dkm)
                claimed[j] = True
                parent[j] = i2
                t_ign[j] = ta
                arcs.append((i2, j, tl, ta, 2))
                heapq.heappush(heap, (ta, j))
        # cross links: close loops so the tree becomes lace
        conn = set((min(a[0], a[1]), max(a[0], a[1])) for a in arcs)
        for a, b in self.edges:
            if (a, b) in conn or not (np.isfinite(t_ign[a]) and np.isfinite(t_ign[b])):
                continue
            dkm = gc_km(P[a], P[b])
            if dkm > 1100.0 or rng.random() > 0.35:
                continue
            i, j = (a, b) if t_ign[a] <= t_ign[b] else (b, a)
            tl = t_ign[j] + rng.uniform(4.0, 30.0)
            ta = tl + self._travel(dkm)
            arcs.append((i, j, tl, ta, 3))
        self.t_ign = t_ign
        self.arcs = np.array(arcs, np.float64)

    # ---------------------------------------------------------- drawing ---
    def _prep_arcs(self):
        a = self.arcs
        self.ai = a[:, 0].astype(np.int64)
        self.aj = a[:, 1].astype(np.int64)
        self.tl = a[:, 2]
        self.ta = a[:, 3]
        self.ak = a[:, 4].astype(np.int64)
        A = self.P[self.ai]
        B = self.P[self.aj]
        self.omega = np.arccos(np.clip(np.sum(A * B, 1), -1, 1))
        self.len_km = self.omega * R_KM
        lift = np.where(self.ak == 1, 0.13, 0.085)
        self.hmax = np.minimum(self.len_km * lift, 1100.0) / R_KM
        self.nseg = np.clip((self.len_km / 22.0).astype(int), 10, 220)
        rng = np.random.default_rng(self.seed + 99)
        self.phase = rng.random(len(a)) * 2 * np.pi
        self.nphase = rng.random(len(self.P)) * 2 * np.pi
        self.nfreq = rng.uniform(0.25, 0.6, len(self.P))

    def arc_points(self, k, u):
        """World positions on arc k at parameters u (array)."""
        A = self.P[self.ai[k]]
        B = self.P[self.aj[k]]
        om = self.omega[k]
        so = math.sin(om)
        if so < 1e-9:
            d = np.outer(1 - u, A) + np.outer(u, B)
        else:
            d = (np.outer(np.sin((1 - u) * om), A) + np.outer(np.sin(u * om), B)) / so
        d /= np.linalg.norm(d, axis=1)[:, None]
        h = self.hmax[k] * np.sin(np.pi * u) + 0.6 / R_KM
        return d * (1.0 + h)[:, None]

    @staticmethod
    def ease(s):
        s = np.clip(s, 0, 1)
        return 0.55 * s + 0.45 * s * s * (3 - 2 * s)

    def draw(self, img, cam, t, gain=1.0, trail=1.0, head=1.0, node=1.0, width=1.0,
             calm=None, sparks=True, dim_far=0.35, t_anim=None):
        """Additively draw the web at state time t into img (HDR, linear). t_anim drives the
        flicker/flow (defaults to t). calm: optional function(py array) -> multiplier."""
        pal = palette()
        ta_ = t if t_anim is None else t_anim
        H, W = img.shape[:2]
        f = cam.f
        active = np.where(self.tl <= t)[0]
        xs, ys, cols, sig, starts = [], [], [], [], [0]
        heads = []
        spark_segs = []
        for k in active:
            s = (t - self.tl[k]) / max(self.ta[k] - self.tl[k], 1e-6)
            uh = float(self.ease(s))
            if uh <= 0.0:
                continue
            n = self.nseg[k]
            m = max(2, int(math.ceil(n * uh)) + 1)
            u = np.linspace(0.0, uh, m)
            X = self.arc_points(k, u)
            uv, z = cam.project(X)
            vis = G.visible(cam.pos, X) & (z > 1e-3)
            L = self.len_km[k]
            age = t - self.ta[k]              # >0 after arrival
            # trail brightness: settles after arrival; flowing shimmer along the thread
            settle = 1.0 if age < 0 else 0.62 + 0.38 * math.exp(-age / 30.0)
            flow = 1.0 + 0.22 * np.sin(u * L / 90.0 - ta_ * 0.35 + self.phase[k])
            it = trail * settle * flow * (1.25 if self.ak[k] == 1 else 1.0)
            # hot head: tapering behind the tip while in flight
            if s < 1.0:
                back = (uh - u) * L
                hot = np.exp(-back / 140.0) * 7.0 + np.exp(-back / 30.0) * 10.0
                heads.append((X[-1], s))
            else:
                hot = 0.0
            I = (it + head * hot) * gain
            depth = np.maximum(z, 1e-3)
            near = np.clip(0.9 * (cam.pos @ cam.pos) ** 0.5 / depth, 0.4, 1.6)
            I = I * (1.0 - dim_far + dim_far * np.clip(near, 0, 1))
            c = np.outer(I, pal['gold'])
            w = np.clip(0.75 * near, 0.55, 1.6) * width
            if calm is not None:
                c *= calm(uv[:, 1])[:, None]
            xx = uv[:, 0].copy()
            yy = uv[:, 1].copy()
            xx[~vis] = np.nan
            xs.append(xx)
            ys.append(yy)
            cols.append(c)
            sig.append(w)
            starts.append(starts[-1] + len(xx))
            # sparks shed by the head (and briefly after arrival)
            if sparks and (s < 1.0 or age < 18):
                spark_segs.append((k, uh, s, age))
        if xs:
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            cols = np.concatenate(cols)
            sig = np.concatenate(sig)
            # wide soft amber glow under the thin core
            glow = cols * (pal['amber'] / pal['gold'])[None, :] * 0.10
            G.splat_polyline(img, xs, ys, glow, sig * 3.2, np.asarray(starts, np.int64))
            G.splat_polyline(img, xs, ys, cols, sig, np.asarray(starts, np.int64))
        # heads: white-hot points
        if heads:
            HP = np.array([h[0] for h in heads])
            uv, z = cam.project(HP)
            vis = G.visible(cam.pos, HP) & (z > 1e-3)
            uv = uv[vis]
            if len(uv):
                sgh = 1.1 * width
                e = np.full(len(uv), 26.0 * head * gain * 2 * np.pi * sgh * sgh)
                rgb = np.outer(e, pal['pale'])
                if calm is not None:
                    rgb *= calm(uv[:, 1])[:, None]
                G.splat_points(img, uv[:, 0].copy(), uv[:, 1].copy(), rgb, np.full(len(uv), sgh))
        if spark_segs:
            self._sparks(img, cam, t, spark_segs, gain, calm)
        self._nodes_draw(img, cam, t, gain * node, width, calm, ta_)

    def _sparks(self, img, cam, t, segs, gain, calm):
        pal = palette()
        X0, X1, E = [], [], []
        for (k, uh, s, age) in segs:
            L = self.len_km[k]
            nsp = int(max(4, L / 35.0))
            rng = np.random.default_rng(int(k) * 7919 + self.seed)
            us = np.sort(rng.random(nsp))
            life = rng.uniform(6.0, 16.0, nsp)
            # the time the head passed u
            dur = self.ta[k] - self.tl[k]
            # invert ease approx by sampling
            ss = np.linspace(0, 1, 64)
            ue = self.ease(ss)
            tpass = self.tl[k] + np.interp(us, ue, ss) * dur
            a = t - tpass
            m = (a >= 0) & (a < life)
            if not m.any():
                continue
            us, life, a = us[m], life[m], a[m]
            P0 = self.arc_points(k, us)
            vel = rng.normal(0, 1, (len(us), 3)) * 0.0009 + P0 * rng.uniform(-0.0012, 0.0003, (len(us), 1))
            p_t = P0 + vel * a[:, None] / 10.0
            p_prev = P0 + vel * np.maximum(a - 1.2, 0)[:, None] / 10.0
            fade = (1.0 - a / life) ** 1.5
            X0.append(p_prev)
            X1.append(p_t)
            E.append(fade * 3.5)
        if not X0:
            return
        X0 = np.concatenate(X0)
        X1 = np.concatenate(X1)
        E = np.concatenate(E) * gain
        uv0, z0 = cam.project(X0)
        uv1, z1 = cam.project(X1)
        vis = G.visible(cam.pos, X1) & (z1 > 1e-3)
        c = np.outer(E, pal['gold'] * np.array([1.0, 0.85, 0.6]))
        if calm is not None:
            c *= calm(uv1[:, 1])[:, None]
        xs = np.empty(2 * vis.sum())
        ys = np.empty(2 * vis.sum())
        xs[0::2] = uv0[vis, 0]
        xs[1::2] = uv1[vis, 0]
        ys[0::2] = uv0[vis, 1]
        ys[1::2] = uv1[vis, 1]
        cc = np.repeat(c[vis], 2, axis=0)
        sg = np.full(len(xs), 0.6)
        st = np.arange(0, len(xs) + 1, 2, dtype=np.int64)
        G.splat_polyline(img, xs, ys, cc, sg, st)

    def _nodes_draw(self, img, cam, t, gain, width, calm, ta_):
        pal = palette()
        lit = np.where(self.t_ign <= t)[0]
        if len(lit) == 0:
            return
        P = self.P[lit] * (1.0 + 0.6 / R_KM)
        uv, z = cam.project(P)
        vis = G.visible(cam.pos, P) & (z > 1e-3)
        lit, uv, z = lit[vis], uv[vis], z[vis]
        age = t - self.t_ign[lit]
        flash = np.exp(-age / 5.0) * 5.0 + np.exp(-age / 22.0) * 1.2
        fl = 1.0 + 0.13 * np.sin(ta_ * self.nfreq[lit] + self.nphase[lit]) + 0.07 * np.sin(ta_ * 1.7 * self.nfreq[lit] + 2 * self.nphase[lit])
        base = np.where(self.kind[lit] == 0, 2.2, 1.0)
        near = np.clip(0.9 * np.linalg.norm(cam.pos) / np.maximum(z, 1e-3), 0.45, 1.5)
        core = (9.0 * base * fl + 30.0 * flash) * gain * (0.65 + 0.35 * np.clip(near, 0, 1))
        halo = (0.55 * base * fl + 2.2 * flash) * gain
        mul = np.ones(len(lit)) if calm is None else calm(uv[:, 1])
        rgb_c = np.outer(core * mul, pal['pale'])
        rgb_h = np.outer(halo * mul, pal['amber'])
        sc = np.clip(near, 0.6, 1.4) * width
        G.splat_points(img, uv[:, 0].copy(), uv[:, 1].copy(), rgb_h * (2 * np.pi * (5.0 * sc) ** 2)[:, None],
                       5.0 * sc)
        G.splat_points(img, uv[:, 0].copy(), uv[:, 1].copy(), rgb_c * (2 * np.pi * (0.9 * sc) ** 2)[:, None],
                       0.9 * sc)

    def summary(self):
        ti = self.t_ign
        fin = np.isfinite(ti)
        return dict(nodes=len(ti), lit=int(fin.sum()), arcs=len(self.arcs),
                    t_pct=np.percentile(ti[fin], [10, 25, 50, 75, 90, 100]).round(1).tolist())
