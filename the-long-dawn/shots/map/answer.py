"""The peoples answer (MAP, cut C, after ~2010).

No new threads are drawn after T_CUT. Instead beacons catch in chains along the inked ranges (a
flame standing on the summit of a drawn peak), along coasts and down the great rivers. Chains
follow a minimum spanning tree that prefers neighbours of the same kind, so fire runs along a
ridge or a shore the way signal fires pass along a line of sight. The answer starts on every
continent at once (seeded near the burnt web where it touches, spontaneously everywhere else),
so no region is visibly last. Deterministic and cached.
"""
import math
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from scipy.spatial import cKDTree

import bake
import features as ft
import geo
import ink

T_CUT = 2010.0          # no new edges after this
T_START = 2011.0        # the first spontaneous answers
T_END = 2057.0          # the last beacon catches by here

KIND_PEAK, KIND_HILL, KIND_COAST, KIND_RIVER = 0, 1, 2, 3


def _poisson(P, prio, spacing, taken=None):
    """Greedy thinning in priority order; `taken` are points that already block."""
    order = np.argsort(-prio)
    keep = []
    pts = [] if taken is None else [tuple(p) for p in taken]
    tree = None
    grid = {}
    c = spacing

    def ok(p):
        gx, gy = int(math.floor(p[0] / c)), int(math.floor(p[1] / c))
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for q in grid.get((ix, iy), ()):
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < spacing * spacing:
                        return False
        return True

    for q in pts:
        grid.setdefault((int(math.floor(q[0] / c)), int(math.floor(q[1] / c))), []).append(q)
    for i in order:
        p = P[i]
        if ok(p):
            keep.append(i)
            grid.setdefault((int(math.floor(p[0] / c)), int(math.floor(p[1] / c))), []).append(tuple(p))
    return np.array(keep, np.int64)


class Answer:
    def __init__(self, web_P, web_t, seed=31):
        path = os.path.join(geo.CACHE, f'answer_{seed}.npz')
        if os.path.exists(path):
            d = np.load(path)
            self.P, self.kind, self.size, self.t_ign = d['P'], d['kind'], d['size'], d['t_ign']
        else:
            self.rng = np.random.default_rng(seed)
            self._sites(web_P, web_t)
            self._chains(web_P, web_t)
            np.savez(path, P=self.P, kind=self.kind, size=self.size, t_ign=self.t_ign)
        self.phase = np.random.default_rng(seed + 1).random(len(self.P)) * 1000.0

    # ------------------------------------------------------------ sites ---
    def _sites(self, web_P, web_t):
        rng = self.rng
        gb = bake.glyph_bank()
        G = ft.build()['glyphs']
        kinds = gb['kind']
        sP, sO, gS = gb['sP'], gb['sO'], gb['gS']

        def summit(gi):
            a, b = sO[gS[gi]], sO[gS[gi] + 1]
            Q = sP[a:b]
            return Q[int(np.argmax(Q[:, 1]))].astype(np.float64)

        lit_web = web_P[np.isfinite(web_t)]
        # peaks: a beacon on the summit of the drawn mountain; bigger peaks first
        mi = np.where(kinds == 0)[0]
        mP = np.array([summit(g) for g in mi])
        ms = G[mi, 3]
        near_web = cKDTree(lit_web).query(mP)[0] < 0.9
        cand = np.where(~near_web)[0]
        keep = cand[_poisson(mP[cand], ms[cand] + rng.uniform(0, 0.25, len(cand)), 1.2)]
        P = [mP[keep]]
        K = [np.full(len(keep), KIND_PEAK)]
        S = [np.clip(1.15 + 0.85 * (ms[keep] - 0.85) / 1.0, 1.1, 2.0) * rng.uniform(0.85, 1.15, len(keep))]
        taken = np.concatenate([lit_web, mP[keep]])
        # hills: a few, on the crown of the arc
        hi = np.where(kinds == 1)[0]
        hP = np.array([summit(g) for g in hi])
        pr = G[hi, 4] + rng.uniform(0, 0.6, len(hi))
        keep = _poisson(hP, pr, 3.0, taken)
        keep = keep[rng.random(len(keep)) < 0.35]
        P.append(hP[keep])
        K.append(np.full(len(keep), KIND_HILL))
        S.append(rng.uniform(0.7, 0.95, len(keep)))
        taken = np.concatenate([taken, hP[keep]])
        # rivers: settlements along the great rivers (not at the springs)
        rv = []
        for R, rad, name in ft.river_lines():
            L = ink.arclen(R)
            if L[-1] < 4.0:
                continue
            n = int(L[-1] / 0.9)
            u = np.linspace(0.15, 1.0, max(n, 2)) * L[-1]
            rv.append(np.stack([np.interp(u, L, R[:, 0]), np.interp(u, L, R[:, 1])], 1))
        rP = np.concatenate(rv)
        keep = _poisson(rP, rng.random(len(rP)), 2.3, taken)
        P.append(rP[keep])
        K.append(np.full(len(keep), KIND_RIVER))
        S.append(rng.uniform(0.65, 0.9, len(keep)))
        taken = np.concatenate([taken, rP[keep]])
        # coasts: headlands and harbours on the larger shores, set a little inland
        F, W, H = ft.map_fields()
        landm = F['landm']
        cP = []
        for Rg, hole in ft.coast_lines():
            if hole or len(Rg) < 30:
                continue
            x, y = Rg[:, 0], Rg[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            if area < 2.0:
                continue
            L = ink.arclen(np.concatenate([Rg, Rg[:1]]))
            n = int(L[-1] / 0.7)
            if n < 3:
                continue
            u = np.linspace(0, L[-1], n, endpoint=False)
            Q = np.stack([np.interp(u, L[:-1], Rg[:, 0]), np.interp(u, L[:-1], Rg[:, 1])], 1)
            tq = np.roll(Q, -1, 0) - np.roll(Q, 1, 0)
            tq /= np.linalg.norm(tq, axis=1)[:, None] + 1e-9
            nq = np.stack([-tq[:, 1], tq[:, 0]], 1)
            a = ft._sample(landm, Q + 0.3 * nq)
            b = ft._sample(landm, Q - 0.3 * nq)
            inward = np.where((a > b)[:, None], nq, -nq)
            cP.append(Q + 0.22 * inward)
        cP = np.concatenate(cP)
        ok = (np.abs(cP[:, 0]) < 178.0) & (cP[:, 1] > geo.MAP_Y0 + 2) & (cP[:, 1] < geo.MAP_Y1 - 3)
        cP = cP[ok]
        onland = ft._sample(landm, cP) > 0.5
        cP = cP[onland]
        keep = _poisson(cP, rng.random(len(cP)), 4.6, taken)
        keep = keep[rng.random(len(keep)) < 0.6]
        P.append(cP[keep])
        K.append(np.full(len(keep), KIND_COAST))
        S.append(rng.uniform(0.65, 0.9, len(keep)))
        P = np.concatenate(P)
        K = np.concatenate(K)
        S = np.concatenate(S)
        # nothing on the Greenland ice or at the map's rim
        lat = geo.ilat(P[:, 1])
        lon = geo.x2lon(P[:, 0])
        gl = (lat > 60) & (lon > -60) & (lon < -22) & (lat < 82) & (K != KIND_COAST)
        ok = ~gl & (np.abs(P[:, 0]) < 178.5) & (lat < geo.LAT_TOP - 2) & (lat > geo.LAT_BOT + 1.5)
        self.P, self.kind, self.size = P[ok], K[ok], S[ok]

    # ----------------------------------------------------------- chains ---
    def _chains(self, web_P, web_t):
        rng = self.rng
        P = self.P
        n = len(P)
        tree = cKDTree(P)
        k = min(8, n - 1)
        dd, ii = tree.query(P, k=k + 1)
        rows, cols, vals = [], [], []
        for i in range(n):
            for m in range(1, k + 1):
                j = ii[i, m]
                d = dd[i, m]
                if d > 4.2:
                    continue
                same = self.kind[i] == self.kind[j]
                wgt = d * (0.7 if same else 1.35) + 1e-3
                rows.append(i)
                cols.append(j)
                vals.append(wgt)
        A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        A = A.maximum(A.T)
        T = minimum_spanning_tree(A)
        T = T.maximum(T.T)
        ncomp, lab = connected_components(T, directed=False)
        # hop time along the tree (frames): a beat for the catch, plus distance
        Th = T.copy().tocsr()
        Th.data = 1.4 + Th.data * 0.8 + rng.uniform(0.0, 0.8, len(Th.data))
        Th = Th.maximum(Th.T)
        lit = np.isfinite(web_t)
        wtree = cKDTree(web_P[lit])
        wt = web_t[lit]
        seeds = []           # (site, time)
        # where the answer touches the burnt web, it catches from the nearest beacon
        dw, iw = wtree.query(P)
        near = dw < 5.0
        for i in np.where(near)[0]:
            seeds.append((i, max(wt[iw[i]], T_CUT - 6.0) + 5.0 + dw[i] * 1.6))
        # elsewhere, on every continent at once: a few spontaneous first fires per chain
        for c in range(ncomp):
            mem = np.where(lab == c)[0]
            if near[mem].any() and len(mem) < 40:
                continue
            ext = np.ptp(P[mem], axis=0).max() if len(mem) > 1 else 0.0
            ns = max(1, int(math.ceil(ext / 26.0)))
            pick = [int(rng.choice(mem))]
            for _ in range(ns - 1):
                dmin = np.min(np.linalg.norm(P[mem][:, None, :] - P[pick][None, :, :], axis=2), axis=1)
                pick.append(int(mem[int(np.argmax(dmin))]))
            for s in pick:
                seeds.append((s, T_START + rng.uniform(0.0, 20.0)))
        si = np.array([s[0] for s in seeds], np.int64)
        st = np.array([s[1] for s in seeds], np.float64)
        D = dijkstra(Th, directed=False, indices=si)
        t = np.min(st[:, None] + D, axis=0)
        # isolated sites (no tree edge) answer on their own
        iso = ~np.isfinite(t)
        t[iso] = T_START + rng.uniform(0, 30, iso.sum())
        # everything has caught by T_END
        hi = t.max()
        if hi > T_END:
            lo = T_START
            t = np.where(t > lo, lo + (t - lo) * (T_END - lo) / (hi - lo), t)
        self.t_ign = t + rng.uniform(0, 0.6, n)

    def summary(self):
        return dict(sites=len(self.P), kinds=np.bincount(self.kind).tolist(),
                    t=np.percentile(self.t_ign, [0, 10, 50, 90, 100]).round(1).tolist())


if __name__ == '__main__':
    import webmap
    w = webmap.MapWeb()
    keep, t = w.cut(T_CUT)
    print('web: arcs kept', int(keep.sum()), 'beacons lit', int(np.isfinite(t).sum()), 'last', float(np.nanmax(np.where(np.isfinite(t), t, np.nan))))
    a = Answer(w.P, t)
    print(a.summary())
