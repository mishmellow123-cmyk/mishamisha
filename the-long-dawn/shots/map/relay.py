"""The relay (MAP, cut C, rev 3): hill by hill, the peoples answer.

The unit of the spread is a flame. From the first beacon on the Himalaya the fire passes from summit
to summit along the inked ranges: every fire lights ONE fire further along its line, a drawn peak (or
the crown of a drawn hill) at an irregular distance, so some peaks are skipped and the gaps vary.
The fire runs as lines of beacons, never as a tree of threads, and nothing is drawn between them.

* The great lines (ROUTES) follow real ranges out of the frame in every direction: west through the
  Karakoram, the Hindu Kush, the Elburz and Anatolia to the Aegean; east through the Hengduan, the
  Qinling and the Taihang to Korea and Japan; south across the plain to the Ghats and Sri Lanka;
  through Arabia to the Horn; down the Levant to Ethiopia; north along the Tian Shan and the Altai...
* A line leaves from an older fire of its parent line only once that fire's own line has passed on
  two more hops, so a tip never forks; no fire but the first beacon passes the fire on more than twice.
* Short side lines (1-3 fires) climb to drawn peaks beside the great lines, later still.
* Nothing flies over the sea: a strait is crossed by a pause and then a fire on the far shore.

Deterministic; cached in renders/map_C/cache/relay_<seed>.npz. Times are v2 global frames.
"""
import heapq
import math
import os

import numpy as np
from scipy.spatial import cKDTree

import bake
import features as ft
import geo

T0 = 1920.0                            # the first beacon flares
EVEREST = (27.99, 86.93)
REGION = (-34.0, 160.0, -22.0, 76.0)   # X0, X1, Y0, Y1 (map degrees): the relay runs this far
T_STOP = 2100.0                        # nothing catches after this

KIND_PEAK, KIND_HILL = 0, 1

# name, parent line, branch point (lat, lon) on the parent line, delay after the parent's fire there
# catches (frames), then (lat, lon) waypoints along the ranges. The first three leave the first beacon
# at the old first landings (1946, 1951, 1954).
ROUTES = [
    ('west', None, None, 1946.0,
     [(29.4, 83.6), (30.4, 79.8), (32.3, 77.3), (34.4, 75.6), (35.9, 74.4), (36.2, 71.4), (35.1, 68.0),
      (34.6, 63.5), (36.2, 59.2), (36.4, 53.0), (38.3, 46.8), (39.6, 42.2), (38.6, 36.8), (37.4, 32.2),
      (38.6, 27.8), (40.1, 22.4), (42.3, 18.0), (45.5, 13.0)]),
    ('east', None, None, 1951.0,
     [(27.8, 89.3), (28.3, 92.5), (29.6, 95.0), (28.6, 99.0), (30.6, 102.8), (33.3, 106.3), (35.2, 110.5),
      (37.4, 113.8), (40.3, 117.2), (40.6, 121.8), (40.4, 125.6), (38.0, 127.6), (35.6, 128.6), (33.8, 130.6),
      (34.5, 133.5), (35.5, 137.5), (37.5, 140.5)]),
    ('south', None, None, 1954.0,
     [(23.6, 84.8), (21.2, 84.4), (18.6, 83.0), (17.0, 81.6), (15.3, 79.3), (13.3, 78.9), (11.4, 76.9),
      (10.0, 77.2), (8.6, 77.4), (7.2, 80.7)]),
    ('india_w', 'south', (23.6, 84.5), 12.0,
     [(24.3, 81.8), (23.9, 78.6), (22.3, 76.3), (20.3, 73.9), (17.6, 73.8), (15.2, 74.3), (12.8, 75.4)]),
    ('aravalli', 'india_w', (23.9, 77.5), 10.0,
     [(24.6, 74.6), (25.8, 73.8), (27.2, 75.6), (28.4, 76.9)]),
    ('southeast', 'east', (28.0, 91.0), 11.0,
     [(25.6, 91.6), (24.6, 93.6), (22.2, 93.6), (19.6, 94.5), (17.0, 96.6), (14.2, 98.5), (10.5, 99.0),
      (6.5, 100.8), (3.0, 102.0)]),
    ('indochina', 'east', (28.6, 99.0), 12.0,
     [(25.8, 100.9), (22.4, 103.2), (19.2, 104.9), (16.4, 107.2), (13.4, 108.3), (10.5, 107.6)]),
    ('china_se', 'east', (33.3, 106.3), 13.0,
     [(31.2, 109.6), (29.2, 112.8), (27.4, 116.0), (26.2, 118.6), (25.2, 121.0)]),
    ('north', 'west', (35.9, 74.4), 10.0,
     [(38.4, 74.8), (40.9, 76.6), (42.3, 80.1), (43.1, 84.0), (43.4, 88.0), (45.8, 90.4), (48.4, 88.6),
      (50.4, 91.4), (51.8, 96.5), (53.5, 101.0)]),
    ('mongolia', 'north', (45.8, 90.4), 12.0,
     [(47.4, 95.5), (47.6, 99.8), (48.2, 104.2), (48.7, 108.6), (49.5, 113.5)]),
    ('southwest', 'west', (34.4, 75.6), 8.0,
     [(32.6, 72.8), (30.9, 69.8), (28.8, 68.5), (27.0, 67.4), (26.2, 63.5), (26.9, 59.5), (27.0, 57.0),
      (25.2, 56.3), (23.1, 57.7), (21.0, 58.8), (18.6, 56.4), (17.2, 54.0), (15.9, 49.5), (15.2, 45.8),
      (13.6, 44.0), (12.5, 43.3), (11.3, 42.6), (10.2, 44.8), (9.9, 47.5), (8.5, 49.2)]),
    ('ethiopia', 'southwest', (12.5, 43.3), 9.0,
     [(11.2, 40.4), (9.4, 39.4), (7.2, 38.2), (4.5, 37.5), (1.5, 36.8)]),
    ('eritrea', 'ethiopia', (11.2, 40.4), 10.0,
     [(13.4, 39.4), (15.4, 38.8), (17.6, 37.8), (20.2, 36.6)]),
    ('zagros', 'southwest', (27.0, 57.0), 6.0,
     [(28.6, 54.0), (30.2, 51.8), (32.0, 49.8), (34.0, 47.8), (36.0, 46.2)]),
    ('caucasus', 'west', (39.6, 42.2), 11.0,
     [(41.4, 44.2), (42.8, 43.6), (43.3, 41.0), (44.2, 39.0), (45.0, 36.0), (46.5, 33.5)]),
    ('levant', 'west', (38.6, 37.5), 9.0,
     [(36.4, 36.4), (34.3, 36.2), (32.2, 35.4), (29.9, 35.5), (28.5, 34.0), (26.4, 33.4), (23.6, 34.8),
      (20.6, 36.4)]),
    ('hejaz', 'levant', (29.9, 35.5), 9.0,
     [(27.6, 36.6), (25.0, 38.0), (22.4, 39.9), (19.8, 41.6), (17.4, 43.2), (15.4, 44.2)]),
]


# per-line pace (hop delay factor): the eastern lines run a little slower and the western a little
# faster, so the fire reaches every edge of the last framing at about the same time
PACE = {'east': 1.4, 'china_se': 1.45, 'indochina': 1.35, 'southeast': 1.25, 'north': 1.1, 'mongolia': 1.2,
        'west': 0.9, 'southwest': 0.85, 'zagros': 0.85, 'caucasus': 0.9, 'levant': 0.9}


def ll2xy(lat, lon):
    return np.array([float(geo.lon2x(lon)), float(geo.yproj(lat))])


def seg_dist(p, Q):
    """Distance from point p to the polyline Q (N,2)."""
    a, b = Q[:-1], Q[1:]
    ab = b - a
    t = np.clip(np.einsum('ij,ij->i', p[None, :] - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12), 0, 1)
    return float(np.min(np.hypot(*(a + ab * t[:, None] - p[None, :]).T)))


def hop_delay(t):
    """Frames between a fire catching and the next one along its line: slow and heavy at first,
    then the relay gathers pace."""
    return 5.2 + 3.4 * math.exp(-max(t - 1946.0, 0.0) / 25.0)


class Relay:
    KEYS = ('P', 'kind', 'size', 'gain', 'phase', 't_ign', 'parent', 'line', 'side')

    def __init__(self, seed=5, cache=True):
        path = os.path.join(geo.CACHE, f'relay_{seed}.npz')
        if cache and os.path.exists(path):
            d = np.load(path)
            for k in self.KEYS:
                setattr(self, k, d[k])
            return
        self.rng = np.random.default_rng(seed)
        self._sites()
        self._spread()
        lit = np.isfinite(self.t_ign)
        remap = np.cumsum(lit) - 1
        for k in self.KEYS:
            setattr(self, k, getattr(self, k)[lit])
        self.parent = np.where(self.parent >= 0, remap[np.maximum(self.parent, 0)], -1)
        if cache:
            np.savez(path, **{k: getattr(self, k) for k in self.KEYS})

    # ------------------------------------------------------------ sites ---
    def _sites(self):
        """Summits of the drawn peaks and crowns of the drawn hills; the first beacon is site 0."""
        rng = self.rng
        gb = bake.glyph_bank()
        G = ft.build()['glyphs']
        kind = gb['kind'].astype(np.int64)
        sP, sO, gS = gb['sP'], gb['sO'], gb['gS']
        idx = np.where((kind == KIND_PEAK) | (kind == KIND_HILL))[0]
        top = []
        for g in idx:
            Q = sP[sO[gS[g]]:sO[gS[g] + 1]]
            top.append(Q[int(np.argmax(Q[:, 1]))])
        top = np.asarray(top, np.float64)
        X0, X1, Y0, Y1 = REGION
        ok = (top[:, 0] > X0) & (top[:, 0] < X1) & (top[:, 1] > Y0) & (top[:, 1] < Y1)
        org = ll2xy(*EVEREST)
        ok &= np.hypot(top[:, 0] - org[0], top[:, 1] - org[1]) > 0.9
        idx, top = idx[ok], top[ok]
        n = len(idx) + 1
        self.P = np.concatenate([org[None, :], top])
        self.kind = np.concatenate([[KIND_PEAK], kind[idx]])
        s = G[idx, 3]
        # a fire's size follows the drawn height: big peaks, lesser peaks, hills
        size = np.where(kind[idx] == KIND_PEAK, 0.9 + 0.35 * np.clip((s - 0.85) / 1.05, 0, 1), 0.68)
        self.size = np.concatenate([[2.0], size * rng.uniform(0.85, 1.15, n - 1)])
        # and its brightness varies fire to fire
        self.gain = np.concatenate([[1.0], rng.uniform(0.7, 1.3, n - 1)])
        self.phase = rng.random(n) * 1000.0
        # how much of a range stands around each site (fire keeps to the ranges)
        tr = cKDTree(self.P)
        nb = tr.query_ball_point(self.P, 3.0)
        cnt = np.array([len(c) + 1.5 * np.sum(self.kind[c] == KIND_PEAK) for c in nb], np.float64)
        self.dens = np.clip(cnt / 14.0, 0.0, 1.0)

    # ----------------------------------------------------------- spread ---
    def _water(self, a, b):
        """Map degrees of open water on the straight line a -> b."""
        L = float(np.hypot(*(b - a)))
        n = max(4, int(L / 0.08))
        u = np.linspace(0, 1, n)
        pts = a[None, :] * (1 - u[:, None]) + b[None, :] * u[:, None]
        return float(np.mean(ft._sample(self.landm, pts) < 0.5)) * L

    def _spread(self):
        rng = self.rng
        P, n = self.P, len(self.P)
        F, _, _ = ft.map_fields()
        self.landm = F['landm']
        tree = cKDTree(P)
        t = np.full(n, np.inf)
        par = np.full(n, -1, np.int64)
        line = np.full(n, -1, np.int64)
        side = np.zeros(n, bool)
        claimed = np.zeros(n, bool)
        blocked = np.zeros(n, bool)
        heading = np.zeros((n, 2))
        taken = [[] for _ in range(n)]         # directions already passed on from each fire
        children = [[] for _ in range(n)]      # the fires each fire has lit, in order
        wp_i = np.zeros(n, np.int64)           # next waypoint of the fire's line
        names = [r[0] for r in ROUTES]
        wps = [np.array([ll2xy(*w) for w in r[4]]) for r in ROUTES]
        started = [False] * len(ROUTES)
        lanes = {}                             # each great line's range polyline (its corridor)
        for k, (name, parent, bp, _, wpts) in enumerate(ROUTES):
            Q = np.array([P[0] if bp is None else ll2xy(*bp)] + [ll2xy(*w) for w in wpts])
            u = np.linspace(0, 1, 6)[:-1]
            lanes[k] = np.concatenate([Q[m][None, :] * (1 - u[:, None]) + Q[m + 1][None, :] * u[:, None]
                                       for m in range(len(Q) - 1)] + [Q[-1:]])
        side_left = {}                         # side line id -> hops left
        nside = [len(ROUTES)]
        heap = []

        def claim(j, tj, i, u, ln, w=0, sd=False):
            children[i].append(j)
            claimed[j] = True
            t[j] = tj
            par[j] = i
            line[j] = ln
            side[j] = sd
            wp_i[j] = w
            heading[j] = u
            taken[i].append(u)
            # no fire stands right beside another: irregular gaps, never a cluster
            for k in tree.query_ball_point(P[j], rng.uniform(0.75, 1.15)):
                if not claimed[k]:
                    blocked[k] = True
            heapq.heappush(heap, (tj, 0, j))

        def pick(i, d, goal=None, rmax=5.2, amin=0.3, jump=False, lane=None, peaks=False):
            dpref = float(np.clip(math.exp(rng.normal(math.log(3.3), 0.3)), 1.6, 5.4))
            excl = taken[i] + ([-heading[i]] if np.any(heading[i]) else [])
            best, bs = None, -1e9
            for j in tree.query_ball_point(P[i], rmax):
                if claimed[j] or blocked[j]:
                    continue
                if peaks and self.kind[j] != KIND_PEAK:
                    continue
                if lane is not None and seg_dist(P[j], lane) > (2.6 if jump else 1.8):
                    continue                                   # a great line keeps to its range
                v = P[j] - P[i]
                L = float(np.hypot(*v))
                if L < 1.25:
                    continue
                u = v / L
                a = float(u @ d)
                # never within ~53 deg of a direction already taken from here, nor ~37 deg of straight back
                if a < amin or any(float(u @ e) > (0.6 if k < len(taken[i]) else 0.8) for k, e in enumerate(excl)):
                    continue
                wl = self._water(P[i], P[j])
                if wl > 2.4:
                    continue
                s = 1.2 * a - 0.8 * abs(L - dpref) / dpref + 0.3 * (self.kind[j] == KIND_PEAK) \
                    + 0.5 * self.dens[j] - 0.3 * (wl > 0.2) + rng.normal(0, 0.3)
                if goal is not None:
                    prog = (np.hypot(*(goal - P[i])) - np.hypot(*(goal - P[j]))) / L
                    s += 0.6 * prog + 1.5 * min(prog, 0.0)          # never away from the waypoint
                    # keep to the line toward the waypoint: a line of beacons, not a zigzag
                    gv = goal - P[i]
                    gl = float(np.hypot(*gv)) + 1e-9
                    s -= 0.9 * abs(float(v[0] * gv[1] - v[1] * gv[0])) / gl / max(L, 1.0)
                if jump:
                    s -= 0.12 * L
                if s > bs:
                    best, bs = (j, u, L, wl), s
            return best

        # how far a fire looks for the next height: close first; across a plateau, a plain or a desert
        # it is seen from further off (a longer hop and a pause), and a great line looks furthest
        TIERS = ((5.4, 0.45, False, 0.0), (8.5, 0.5, True, 1.0), (12.5, 0.55, True, 2.0))

        def hop(i, tnow, d, goal, ln, w, sd):
            got, pause = None, 0.0
            lane = lanes[ln] if not sd else None
            for rmax, amin, jump, pz in (TIERS[:1] if sd else TIERS):
                got = pick(i, d, goal, rmax=rmax, amin=amin, jump=jump, lane=lane, peaks=sd)
                if got is not None:
                    pause = pz * rng.uniform(0.7, 1.4)
                    break
            if got is None:
                return None
            j, u, L, wl = got
            dt = (hop_delay(tnow) * rng.uniform(0.8, 1.3) + 0.25 * L) * (1.0 if sd else PACE.get(names[ln], 1.0)) + pause
            if wl > 0.2:
                dt += rng.uniform(4.0, 8.0)                 # a strait: a pause, then the far shore
            if rng.random() < 0.1:
                dt += rng.uniform(2.0, 5.0)                 # a people slow to answer
            if tnow + dt > T_STOP:
                return None
            claim(j, tnow + dt, i, u, ln, w, sd)
            return j

        def route_step(i, tnow):
            """Pass the fire on toward the line's next waypoint; a waypoint nothing can reach is
            skipped; the line ends at its last one (off the frame)."""
            r = line[i]
            W = wps[r]
            w = wp_i[i]
            while w < len(W):
                # a waypoint is passed when the fire stands near it, or already nearer the one after it
                while w < len(W) and (np.hypot(*(W[w] - P[i])) < 2.4 or (w + 1 < len(W) and np.hypot(
                        *(W[w + 1] - P[i])) < np.hypot(*(W[w + 1] - W[w])) - 0.5)):
                    w += 1
                if w >= len(W) or (w == len(W) - 1 and np.hypot(*(W[w] - P[i])) < 4.0):
                    return None
                goal = W[w]
                g = (goal - P[i]) / (np.hypot(*(goal - P[i])) + 1e-12)
                d = 0.65 * g + 0.35 * heading[i] if np.any(heading[i]) else g
                d = d / (np.linalg.norm(d) + 1e-12)
                j = hop(i, tnow, d, goal, r, w, False)
                if j is not None:
                    return j
                w += 1
            return None

        def moved_on(i, tnow):
            """Has the fire's own line passed on at least two hops by now (so it is not a tip)? A fire
            whose line ended may start a new one; a fire already passing the fire on twice may not."""
            ks = children[i]
            if len(ks) >= 2:
                return False
            if not ks:
                return True
            c = ks[0]
            return t[c] <= tnow and any(t[g] <= tnow for g in children[c])

        # the first beacon has burned since the cold mountain
        t[0] = T0 - 300.0
        claimed[0] = True
        for k in tree.query_ball_point(P[0], 1.2):
            if k:
                blocked[k] = True
        for r, (name, parent, bp, t_first, _) in enumerate(ROUTES):
            if parent is None:
                started[r] = True
                g = wps[r][0] - P[0]
                g /= np.linalg.norm(g)
                got = pick(0, g, wps[r][0], rmax=6.5, amin=0.6, jump=True)
                if got is not None:
                    j, u, L, wl = got
                    claim(j, t_first, 0, u, r, 0)
        while heap:
            tn, ev, i = heapq.heappop(heap)
            if ev == 0 and not side[i]:
                # a fire on a great line catches: it passes the fire on along its line
                route_step(i, tn)
                # lines that part from here start long after
                for r, (name, parent, bp, delay, _) in enumerate(ROUTES):
                    if started[r] or parent is None or names.index(parent) != line[i]:
                        continue
                    if np.hypot(*(ll2xy(*bp) - P[i])) < 3.0:
                        started[r] = True
                        heapq.heappush(heap, (tn + delay * rng.uniform(0.9, 1.2), 1, i * 64 + r))
                # now and then a short side line answers into the ranges beside it, later still
                if i != 0 and rng.random() < 0.12 and np.hypot(*(P[i] - P[0])) > 4.0:
                    heapq.heappush(heap, (tn + rng.uniform(15.0, 45.0), 2, i))
            elif ev == 0:
                ln = line[i]
                if side_left.get(ln, 0) > 0:
                    side_left[ln] -= 1
                    hop(i, tn, heading[i], None, ln, 0, True)
            elif ev == 1:
                i, r = divmod(i, 64)
                goal = wps[r][0]
                # from the fire nearest the branch point that is free to start a line (never a tip,
                # never a third line from one fire); else wait for the parent line to move on
                bp = ll2xy(*ROUTES[r][2])
                near = np.where((line == line[i]) & ~side & (t <= tn))[0]
                near = near[(np.hypot(*(P[near] - bp).T) < 5.5) & np.array([len(children[k]) < 2 for k in near], bool)]
                done = waiting = False
                for k in sorted(near.tolist(), key=lambda k: np.hypot(*(P[k] - bp))):
                    if not moved_on(k, tn):
                        waiting = True
                        continue
                    g = (goal - P[k]) / (np.hypot(*(goal - P[k])) + 1e-12)
                    if hop(k, tn, g, goal, r, 0, False) is not None:
                        done = True
                        break
                if not done:
                    if waiting and tn < T_STOP - 20:
                        heapq.heappush(heap, (tn + 3.0, 1, i * 64 + r))
                    else:
                        started[r] = False                 # the next fire of the parent line re-arms it
            else:
                # a side line: toward the ranges nobody has answered from yet, never back or along
                if not moved_on(i, tn):
                    if tn < T_STOP - 20:
                        heapq.heappush(heap, (tn + 3.0, 2, i))
                    continue
                best = None
                for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    d = np.array([math.cos(ang), math.sin(ang)])
                    if any(float(d @ e) > 0.5 for e in taken[i] + [-heading[i]]):
                        continue
                    ahead = [k for k in tree.query_ball_point(P[i] + d * 5.0, 4.0) if not claimed[k] and not blocked[k]]
                    sc = len(ahead) + rng.normal(0, 1.0)
                    if len(ahead) >= 4 and (best is None or sc > best[0]):
                        best = (sc, d)
                if best is None:
                    continue
                ln = nside[0]
                nside[0] += 1
                side_left[ln] = int(rng.integers(1, 4))
                hop(i, tn, best[1], None, ln, 0, True)
        self.t_ign = t
        self.parent = par
        self.line = line
        self.side = side

    def summary(self):
        tl = self.t_ign[1:]
        return dict(fires=len(self.P), great=int(np.sum(~self.side)), side=int(np.sum(self.side)),
                    peaks=int(np.sum(self.kind == 0)), hills=int(np.sum(self.kind == 1)),
                    t=np.percentile(tl, [0, 10, 25, 50, 75, 90, 100]).round(1).tolist())


if __name__ == '__main__':
    r = Relay(cache=False)
    print(r.summary())
    tt = np.sort(r.t_ign[1:])
    print('catches per 10 frames', np.histogram(tt, bins=np.arange(1940, 2110, 10))[0].tolist())
    for k, (name, *_) in enumerate(ROUTES):
        m = r.line == k
        if m.any():
            print(f'{name:10s} fires {int(m.sum()):3d}  first {r.t_ign[m].min():.0f}  last {r.t_ign[m].max():.0f}')
