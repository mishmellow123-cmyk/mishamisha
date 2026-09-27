"""The relay (MAP, cut C, rev 4 on the INVENTED world): hill by hill, the peoples answer.

The unit of the spread is a flame. The map opens on her range (the ink Run's range) with her beacon burning on its
high knee and the Run's seven fires already lit along its east arm, the seventh still blooming (the burn-through
from C17 opens on it). From there the fire passes from summit to summit: every fire lights ONE fire further along
its line, a drawn peak (or the crown of a drawn hill) at an irregular distance, so some heights are skipped and the
gaps vary. The fire runs as lines of beacons, never as a tree of threads, and nothing is drawn between them.

* The great lines (ROUTES, map XY) run along the invented world's ranges, down its great rivers (beacon hills
  stand along their banks) and along its coasts (on the headlands), out of the frame in every direction.
* A line leaves from an older fire of its parent line only once that fire's own line has passed on two more
  hops, so a tip never forks; no fire but the first beacon passes the fire on more than twice.
* Short side lines (1-3 fires) climb to drawn peaks beside the great lines, later still.
* Nothing flies over the sea: a strait is crossed by a pause and then a fire on the far shore.

Deterministic; cached in renders/map_C/cache_w/relay_<seed>.npz. Times are v2 global frames (C18: 4160 = 1920).
"""
import heapq
import math
import os

import numpy as np
from scipy.spatial import cKDTree

import bake
import features as ft
import geo
import terra

T0 = 1920.0                            # the map opens (C 4160); the answer begins
REGION = (-62.0, 84.0, -34.0, 64.0)   # X0, X1, Y0, Y1 (map degrees): the relay runs this far (well off the frame)
T_STOP = 2100.0                        # nothing catches after this

KIND_PEAK, KIND_HILL = 0, 1

# THE RUN'S SEVEN: lit along her range's east arm before the map opens (C17 catches them at C 3880-4120; on this
# clock that is ~1773-1899), the seventh still blooming as the burn-through opens the map.
CHAIN = [(15.2, 7.6), (17.4, 7.1), (19.5, 6.6), (21.6, 6.0), (23.8, 5.5), (26.0, 5.2), (28.2, 5.0)]
CHAIN_T = [1800.0, 1818.0, 1836.0, 1854.0, 1872.0, 1890.0, 1918.0]

# name, parent line, branch point (x, y) on the parent line, delay after the parent's fire there catches (frames),
# then (x, y) waypoints. A line with no parent leaves from its start fire: 'hers' (her beacon) or 'seventh'.
ROUTES = []            # filled by routes() from the world (see below)
PACE = {}


def ll2xy(x, y):
    """(compatibility) waypoints are map XY already."""
    return np.array([float(x), float(y)])


def origin(top, kind, size):
    """Her beacon: the summit of the biggest drawn peak on her range's knee (terra.BEACON)."""
    b = np.array(terra.BEACON)
    d = np.hypot(top[:, 0] - b[0], top[:, 1] - b[1])
    m = (kind == KIND_PEAK) & (d < 1.8)
    if not m.any():
        return b.astype(np.float64)
    i = np.where(m)[0][int(np.argmax(size[m] - 0.4 * d[m]))]
    return top[i].astype(np.float64)


def routes():
    """The great lines on the invented world (terra's design): (name, parent, branch point or start fire, delay or
    first catch, waypoints). Every line runs off the widest frame. The High Moor stays dark: it is no one's."""
    global ROUTES, PACE
    if ROUTES:
        return ROUTES
    ROUTES = [
        # the Run's line runs on from the seventh along her range's east arm, over the pass, off the east edge
        ('east', None, 'hers', 0.0,
         [(30.0, 4.9), (33.0, 5.2), (37.5, 6.0), (43.0, 7.8), (50.0, 10.4), (57.0, 13.6), (64.0, 18.0), (70.0, 24.0)]),
        # down the west arm to the south-west lowlands and off the bottom edge
        ('west', None, 'hers', 1944.0,
         [(9.0, 6.2), (5.0, 3.6), (0.5, 0.4), (-4.0, -3.4), (-8.0, -7.0), (-4.0, -12.0), (-9.0, -16.0)]),
        # north across the valley by the lake's hills, to the northern range, and along it east off the top
        ('north', None, 'hers', 1950.0,
         [(16.5, 15.5), (15.5, 21.0), (15.0, 26.0), (13.0, 31.5), (13.0, 34.8), (19.0, 36.8), (25.0, 39.6), (31.0, 43.0),
          (38.0, 49.0)]),
        # along the northern range west, to the hook and its tip
        ('north_w', 'north', (13.0, 34.8), 9.0,
         [(7.5, 33.6), (2.5, 33.2), (-2.5, 33.6), (-8.0, 33.2), (-13.0, 33.5), (-20.0, 38.0), (-26.0, 36.8), (-30.5, 34.4),
          (-32.5, 30.0)]),
        # down the west river to the western bay
        ('river', 'west', (5.0, 3.6), 10.0,
         [(4.5, 8.5), (0.0, 11.2), (-4.0, 12.4), (-8.0, 13.8), (-12.0, 15.4), (-16.0, 16.4)]),
        # along the coast south, down the long cape to its tip
        ('coast_s', 'river', (-12.0, 15.4), 8.0,
         [(-18.0, 12.2), (-22.0, 10.0), (-26.0, 8.4), (-31.0, 5.0), (-36.0, 0.6)]),
        # along the coast north, round the bulge into the hook's bay
        ('coast_n', 'river', (-12.0, 15.4), 12.0,
         [(-18.5, 19.5), (-22.0, 21.5), (-23.0, 25.0), (-21.0, 29.0), (-24.0, 31.0), (-17.0, 41.0)]),
        # east from the lake's hills across the plains into the desert, off the east edge
        ('desert', 'north', (15.5, 21.0), 12.0,
         [(20.0, 22.5), (25.0, 24.5), (30.0, 27.5), (35.0, 29.0), (41.0, 30.0), (48.0, 33.0), (56.0, 37.0)]),
        # south from the west arm across the lowlands to the south range, and along it east
        ('south', 'west', (0.5, 0.4), 10.0,
         [(3.0, -5.0), (7.0, -10.0), (14.0, -13.0), (22.0, -15.0), (30.0, -17.0), (38.0, -22.0)]),
    ]
    # the lines that have furthest to go run a little faster, so the fire reaches every edge of the widest frame
    # at about the same time
    PACE = {'east': 0.95, 'north': 0.95, 'north_w': 0.9, 'desert': 0.95, 'coast_n': 1.0, 'coast_s': 1.0,
            'river': 1.0, 'south': 1.05, 'west': 1.0}
    return ROUTES


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
    KEYS = ('P', 'kind', 'size', 'gain', 'phase', 't_ign', 'parent', 'line', 'side', 'chain')

    def __init__(self, seed=5, cache=True):
        path = os.path.join(geo.CACHE, f'relay_{seed}.npz')
        if cache and os.path.exists(path):
            d = np.load(path)
            for k in self.KEYS:
                setattr(self, k, d[k])
            return
        self.rng = np.random.default_rng(seed)
        routes()
        self._sites()
        self._spread()
        lit = np.isfinite(self.t_ign)
        remap = np.cumsum(lit) - 1
        for k in self.KEYS:
            if k != 'chain':
                setattr(self, k, getattr(self, k)[lit])
        self.parent = np.where(self.parent >= 0, remap[np.maximum(self.parent, 0)], -1)
        self.chain = remap[self.chain]
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
        org = origin(top, kind[idx], G[idx, 3])
        ok &= np.hypot(top[:, 0] - org[0], top[:, 1] - org[1]) > 0.9
        mc, mr = terra.MOOR['c'], terra.MOOR['r']
        ok &= np.hypot(top[:, 0] - mc[0], (top[:, 1] - mc[1]) * 1.12) > 1.05 * mr      # the moor is no one's: dark
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
            if bp is None or bp == 'hers':
                q0 = P[0]
            elif isinstance(bp, str):
                q0 = np.array(CHAIN[int(bp[1:]) - 1])
            else:
                q0 = ll2xy(*bp)
            Q = np.array([q0] + [ll2xy(*w) for w in wpts])
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

        # her beacon has burned since the cold mountain
        t[0] = T0 - 300.0
        claimed[0] = True
        for k in tree.query_ball_point(P[0], 1.2):
            if k:
                blocked[k] = True
        # the Run's seven, already lit along her range's east arm; the seventh still blooming
        run = names.index('east')
        chain = [0]
        prev = 0
        for (cx, cy), tc in zip(CHAIN, CHAIN_T):
            c = np.array([cx, cy])
            best, bd = None, 1e9
            for j in tree.query_ball_point(c, 1.6):
                if claimed[j] or self.kind[j] != KIND_PEAK:
                    continue
                dd = float(np.hypot(*(P[j] - c))) - 0.15 * self.size[j]
                if dd < bd:
                    best, bd = j, dd
            if best is None:
                continue
            v = P[best] - P[prev]
            claim(best, tc, prev, v / (np.linalg.norm(v) + 1e-12), run, 0)
            heap.pop()                      # the chain passed the fire on in C17, not here
            chain.append(best)
            prev = best
        heapq.heapify(heap)
        self.chain = np.array(chain, np.int64)
        seventh = chain[-1]
        heapq.heappush(heap, (t[seventh], 0, seventh))       # the east line runs on from the seventh
        started[run] = True
        for r, (name, parent, bp, t_first, _) in enumerate(ROUTES):
            if parent is None and r != run:
                started[r] = True
                src = 0 if bp in (None, 'hers') else chain[int(bp[1:])]
                g = wps[r][0] - P[src]
                g /= np.linalg.norm(g)
                got = pick(src, g, wps[r][0], rmax=6.5, amin=0.6, jump=True)
                if got is not None:
                    j, u, L, wl = got
                    claim(j, t_first, src, u, r, 0)
        while heap:
            tn, ev, i = heapq.heappop(heap)
            if ev == 0 and not side[i]:
                # a fire on a great line catches: it passes the fire on along its line
                route_step(i, tn)
                # lines that part from here start long after
                for r, (name, parent, bp, delay, _) in enumerate(ROUTES):
                    if started[r] or parent is None or names.index(parent) != line[i]:
                        continue
                    if isinstance(bp, str) or np.hypot(*(ll2xy(*bp) - P[i])) < 5.0:
                        started[r] = True
                        heapq.heappush(heap, (tn + delay * rng.uniform(0.9, 1.2), 1, i * 64 + r))
                # now and then a short side line answers into the ranges beside it, later still
                if i != 0 and rng.random() < 0.3 and np.hypot(*(P[i] - P[0])) > 4.0:
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
                near = near[(np.hypot(*(P[near] - bp).T) < 7.0) & np.array([len(children[k]) < 2 for k in near], bool)]
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
