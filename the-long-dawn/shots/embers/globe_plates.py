"""Design tool for the ember globe's plates (v2 framing review).

The globe is seen from above the Arctic, so every power sits on the rim together; its plates must not part along
any real line. This tool places the plate seeds (Voronoi on the sphere) so that the seams keep clear of real
flashpoints (hard margin), of capitals / AI hubs / big cities (soft margin), and prefer not to run along land borders
or through dense population. It prints the seed list that scene_c.Globe uses and writes a map for review.

    python globe_plates.py [out.png]

The seeds it found are scene_c.GLOBE_SEEDS. The cells are then merged into 13 irregular plates
(scene_c.GLOBE_GROUPS; only seams between plates crack), chosen so the fracture crosses every continent -- a grouping
that left North America whole while everything else cracked would itself read as a statement.
"""
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, 'assets', 'data')

# (name, lat, lon, margin deg) -- places a seam must never trace or cut
HARD = [
    # the first island chain, the Taiwan Strait, Korea, Japan, the South China Sea (the original complaint)
    ('Kyushu', 31.5, 130.5, 7), ('Okinawa', 26.3, 127.8, 7), ('Miyako', 24.8, 125.3, 7), ('Yonaguni', 24.4, 123.0, 7),
    ('Taipei', 25.0, 121.5, 8), ('Taiwan S', 22.0, 120.8, 7), ('Taiwan Strait', 24.5, 119.5, 8),
    ('Luzon Strait', 20.5, 121.9, 6), ('Manila', 14.6, 121.0, 4), ('Paracels', 16.5, 112.0, 4),
    ('Scarborough', 15.2, 117.8, 4), ('Senkaku', 25.7, 123.5, 7),
    ('DMZ', 38.0, 127.0, 7), ('Seoul', 37.6, 127.0, 6), ('Pyongyang', 39.0, 125.7, 6), ('Tokyo', 35.7, 139.7, 5),
    # India / Pakistan / China
    ('Kashmir LoC', 34.5, 74.5, 6), ('Islamabad', 33.7, 73.0, 5), ('Galwan', 34.7, 78.2, 6), ('Doklam', 27.3, 89.0, 5),
    ('Arunachal', 28.0, 94.0, 5), ('New Delhi', 28.6, 77.2, 5), ('Wagah', 31.6, 74.6, 5),
    # Ukraine and the Black Sea; the Baltic
    ('Kyiv', 50.45, 30.5, 5), ('Donbas', 48.0, 38.0, 5), ('Crimea', 45.0, 34.0, 6), ('Kharkiv', 50.0, 36.2, 5),
    ('Odesa', 46.5, 30.7, 5), ('Moscow', 55.75, 37.6, 4), ('Minsk', 53.9, 27.6, 4), ('Kaliningrad', 54.7, 20.5, 4),
    ('Suwalki', 54.1, 23.0, 4), ('Narva', 59.4, 28.2, 3), ('Riga', 56.95, 24.1, 3),
    # the Levant, Suez, the Gulf, the Caucasus
    ('Gaza', 31.4, 34.4, 6), ('Jerusalem', 31.8, 35.2, 6), ('Beirut', 33.9, 35.5, 5), ('Damascus', 33.5, 36.3, 5),
    ('Suez', 30.0, 32.5, 4), ('Tehran', 35.7, 51.4, 4), ('Hormuz', 26.6, 56.3, 5), ('Gulf', 27.0, 51.0, 4),
    ('Tbilisi', 41.7, 44.8, 3), ('Karabakh', 39.8, 46.7, 3),
    # the Bering Strait
    ('Bering', 65.8, -168.8, 6),
]
# (name, lat, lon, margin) -- soft: capitals, AI hubs, contested edges nearer the rim
SOFT = [
    ('Washington', 38.9, -77.0, 5), ('New York', 40.7, -74.0, 5), ('SF Bay', 37.5, -122.2, 5), ('Seattle', 47.6, -122.3, 4),
    ('Toronto', 43.7, -79.4, 4), ('Montreal', 45.5, -73.6, 3), ('Ottawa', 45.4, -75.7, 3), ('Mexico City', 19.4, -99.1, 3),
    ('US-Mex W', 32.5, -117.0, 4), ('US-Mex C', 31.8, -106.4, 4), ('US-Mex E', 26.0, -97.5, 4),
    ('London', 51.5, -0.1, 5), ('Paris', 48.85, 2.35, 5), ('Berlin', 52.5, 13.4, 5), ('Brussels', 50.85, 4.35, 4),
    ('Zurich', 47.4, 8.5, 3), ('Rome', 41.9, 12.5, 3), ('Madrid', 40.4, -3.7, 3), ('Warsaw', 52.2, 21.0, 4),
    ('Stockholm', 59.3, 18.1, 3), ('Helsinki', 60.2, 24.9, 4), ('Finland border', 64.0, 29.8, 4), ('Kola', 69.7, 30.8, 4),
    ('St Petersburg', 59.9, 30.3, 4), ('Istanbul', 41.0, 29.0, 4), ('Ankara', 39.9, 32.9, 3), ('Baghdad', 33.3, 44.4, 4),
    ('Riyadh', 24.7, 46.7, 4), ('Abu Dhabi', 24.5, 54.4, 4), ('Tel Aviv', 32.1, 34.8, 5), ('Cairo', 30.0, 31.2, 4),
    ('Beijing', 39.9, 116.4, 6), ('Shanghai', 31.2, 121.5, 6), ('Hangzhou', 30.3, 120.2, 5), ('Shenzhen', 22.5, 114.1, 6),
    ('Hong Kong', 22.3, 114.2, 6), ('Chengdu', 30.7, 104.1, 3), ('Urumqi', 43.8, 87.6, 4), ('Vladivostok', 43.1, 131.9, 4),
    ('Kuril', 45.0, 148.0, 4), ('Mumbai', 19.1, 72.9, 3), ('Bangalore', 12.97, 77.6, 3), ('Karachi', 24.9, 67.0, 3),
    ('Kabul', 34.5, 69.2, 3), ('Hanoi', 21.0, 105.8, 3), ('Bangkok', 13.75, 100.5, 3), ('Singapore', 1.3, 103.8, 2),
    ('Nuuk', 64.2, -51.7, 4), ('Pituffik', 76.5, -68.7, 4), ('Nares', 80.0, -68.0, 4), ('Svalbard', 78.2, 15.6, 3),
    ('Anchorage', 61.2, -149.9, 3), ('Havana', 23.1, -82.4, 3), ('Caracas', 10.5, -66.9, 2), ('Panama', 9.0, -79.5, 2),
    ('Bab el-Mandeb', 12.6, 43.3, 3), ('Khartoum', 15.6, 32.5, 2), ('Tripoli', 32.9, 13.2, 2),
]

# initial seeds (lat, lon): the flashpoint regions each sit inside one plate; fillers elsewhere
INIT = [
    (27, 125),    # East Asia & the western Pacific rim, whole
    (29, 82),     # South Asia & the Himalaya, whole
    (34, 44),     # the Levant, the Caucasus, the Gulf
    (51, 27),     # Ukraine, the Black Sea's north shore, the Baltic, Belarus
    (44, -4),     # western Europe & the Mediterranean
    (40, -95),    # North America
    (36, -45),    # North Atlantic
    (63, -100),   # Canada
    (74, -40),    # Greenland
    (86, 100),    # the Arctic
    (68, 80),     # western Siberia
    (64, 135),    # eastern Siberia
    (62, -172),   # Bering
    (30, -178),   # North Pacific
    (18, -135),   # eastern Pacific
    (15, 10),     # Sahara
    (8, 55),      # Arabian Sea
    (5, 105),     # South East Asia
    (12, -60),    # Caribbean
]
N_SOUTH = 9       # plates of the (unseen) southern hemisphere


def ll2v(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    # scene_c.sph_to_ll: lat = asin(y), lon = atan2(-z, x)
    return np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), -np.cos(lat) * np.sin(lon)], -1)


def v2ll(v):
    return np.degrees(np.arcsin(np.clip(v[..., 1], -1, 1))), np.degrees(np.arctan2(-v[..., 2], v[..., 0]))


def seam_dist(X, S):
    """angular distance (deg) from points X to the nearest plate boundary of seeds S (exact for convex cells)"""
    d = X @ S.T
    a = np.argmax(d, 1)
    A = S[a]
    diff = A[:, None, :] - S[None, :, :]                       # (n, K, 3)
    nrm = np.maximum(np.linalg.norm(diff, axis=2), 1e-12)
    ratio = np.einsum('nkj,nj->nk', diff, X) / nrm
    ratio[np.arange(len(X)), a] = 1.0                            # the own seed is no boundary
    dist = np.arcsin(np.clip(ratio, 0, 1))
    return np.degrees(dist.min(1)), a


def grids():
    # country id raster -> land-border distance (deg); population density
    W, H = 1440, 720
    cid = np.zeros((H, W), np.int32)
    d = json.load(open(os.path.join(DATA, 'ne_50m_admin_0_countries.geojson')))
    for i, f in enumerate(d['features']):
        g = f['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        for poly in polys:
            ring = np.array(poly[0])
            pts = np.stack([(ring[:, 0] + 180) / 360 * W, (90 - ring[:, 1]) / 180 * H], 1)
            cv2.fillPoly(cid, [np.round(pts * 8).astype(np.int32)], i + 1, shift=3)
    land = cid > 0
    b = np.zeros_like(land)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        s = np.roll(np.roll(cid, dy, 0), dx, 1)
        b |= (cid > 0) & (s > 0) & (s != cid)
    bdist = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5) * (180.0 / H)
    pop = np.zeros((H, W), np.float32)
    pp = json.load(open(os.path.join(DATA, 'ne_10m_populated_places_simple.geojson')))
    for f in pp['features']:
        pr = f['properties']
        pm = pr.get('pop_max') or 0
        if pm < 300000:
            continue
        lon, lat = f['geometry']['coordinates'][:2]
        x = int((lon + 180) / 360 * W) % W
        y = min(int((90 - lat) / 180 * H), H - 1)
        pop[y, x] += pm / 1e6
    pop = cv2.GaussianBlur(pop, (0, 0), 5.0)
    return land, b, bdist, pop


def sample(grid, lat, lon):
    H, W = grid.shape
    x = ((lon + 180) / 360 * W).astype(int) % W
    y = np.clip(((90 - lat) / 180 * H).astype(int), 0, H - 1)
    return grid[y, x]


def seam_fast(X, S):
    """distance (deg) to the bisector of the two nearest seeds: cheap, exact on the seam itself"""
    d = X @ S.T
    i2 = np.argpartition(-d, 1, axis=1)[:, :2]
    a, b = S[i2[:, 0]], S[i2[:, 1]]
    diff = a - b
    return np.degrees(np.arcsin(np.clip(np.abs((X * diff).sum(1)) / np.linalg.norm(diff, axis=1), 0, 1)))


def objective(S, X, Xw, xland, xbd, xpop, Hv, Hm, Sv, Sm):
    ds = seam_fast(X, S)
    seam = ds < 0.6
    ns = max(seam.sum(), 1)
    P_b = (Xw[seam] * (xbd[seam] < 1.2)).sum() / ns                # along/over land borders
    P_p = (Xw[seam] * xpop[seam]).sum() / ns                         # through dense population
    P_l = (Xw[seam] * xland[seam]).sum() / ns                        # over land at all
    dh, _ = seam_dist(Hv, S)
    P_h = (np.maximum(0, Hm - dh) ** 2).sum()
    dso, _ = seam_dist(Sv, S)
    P_s = (np.maximum(0, Sm - dso) ** 2).sum()
    # regular plates: no two seeds closer than 20 deg
    c = np.clip(S @ S.T, -1, 1)
    np.fill_diagonal(c, -1)
    ang = np.degrees(np.arccos(c.max(1)))
    P_r = (np.maximum(0, 20.0 - ang) ** 2).sum()
    J = 60.0 * P_h + 6.0 * P_s + 40.0 * P_b + 6.0 * P_p + 4.0 * P_l + 2.0 * P_r
    return J, dict(hard=P_h, soft=P_s, border=P_b, pop=P_p, land=P_l, reg=P_r, min_hard=float((dh - Hm).min()))


def optimise(seed=3, iters=9000, restarts=4):
    land, b, bdist, pop = grids()
    n = 40000
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    V = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    V = V[V[:, 1] > 0.03]
    lat, lon = v2ll(V)
    Xw = np.clip((lat - 2.0) / 18.0, 0, 1) ** 0.7
    xland = sample(land.astype(np.float32), lat, lon)
    xbd = sample(bdist, lat, lon)
    xpop = sample(pop, lat, lon)
    xpop = xpop / (xpop.mean() + 1e-9)
    xpop = np.minimum(xpop, 20.0) * xland
    Hv = ll2v(np.array([h[1] for h in HARD]), np.array([h[2] for h in HARD]))
    Hm = np.array([h[3] for h in HARD], float)
    Sv = ll2v(np.array([s_[1] for s_ in SOFT]), np.array([s_[2] for s_ in SOFT]))
    Sm = np.array([s_[3] for s_ in SOFT], float)
    north0 = ll2v(np.array([p[0] for p in INIT], float), np.array([p[1] for p in INIT], float))
    southv = []
    for k in range(N_SOUTH):
        la = -30.0 - 35.0 * ((k * 0.618) % 1.0)
        lo = -180 + 360.0 * (k + 0.5) / N_SOUTH
        southv.append(ll2v(np.array(la), np.array(lo)))
    southv = np.array(southv)
    nn = len(north0)
    best_all = None
    for rs in range(restarts):
        r = np.random.default_rng(seed + 101 * rs)
        north = north0.copy()
        if rs > 0:                                       # perturbed starts
            for k in range(nn):
                ax = np.cross(north[k], r.normal(size=3))
                ax /= np.linalg.norm(ax)
                ang = math.radians(r.uniform(0, 10))
                v = north[k]
                v = v * math.cos(ang) + np.cross(ax, v) * math.sin(ang) + ax * (ax @ v) * (1 - math.cos(ang))
                north[k] = v / np.linalg.norm(v)
        S = np.vstack([north, southv])
        J, parts = objective(S, V, Xw, xland, xbd, xpop, Hv, Hm, Sv, Sm)
        best = (J, S.copy(), parts)
        T0, T1 = max(J * 0.02, 5.0), 0.05
        for it in range(iters):
            f = it / iters
            T = T0 * (T1 / T0) ** f
            k = r.integers(0, nn)
            step = math.radians(r.uniform(0.3, 1.0) * (14.0 * (1 - f) + 0.7))
            ax = np.cross(S[k], r.normal(size=3))
            ax /= np.linalg.norm(ax)
            v = S[k]
            vn = v * math.cos(step) + np.cross(ax, v) * math.sin(step) + ax * (ax @ v) * (1 - math.cos(step))
            S2 = S.copy()
            S2[k] = vn / np.linalg.norm(vn)
            if S2[k][1] < 0.02:
                continue
            J2, p2 = objective(S2, V, Xw, xland, xbd, xpop, Hv, Hm, Sv, Sm)
            if J2 < J or r.random() < math.exp(-(J2 - J) / T):
                S, J, parts = S2, J2, p2
                if J < best[0]:
                    best = (J, S.copy(), parts)
            if it % 1500 == 0:
                print('restart', rs, it, 'T %.2f J %.2f best %.2f' % (T, J, best[0]),
                      {k_: round(v_, 3) for k_, v_ in best[2].items()}, flush=True)
        print('restart', rs, 'best %.3f' % best[0], {k_: round(v_, 3) for k_, v_ in best[2].items()}, flush=True)
        if best_all is None or best[0] < best_all[0]:
            best_all = best
    return best_all, (land, b, V, Hv, Hm, Sv, Sm)


def draw(S, land, b, out, Hm=None):
    H, W = land.shape
    img = np.zeros((H, W, 3), np.uint8)
    img[land] = (60, 60, 60)
    img[b] = (110, 110, 110)
    yy, xx = np.mgrid[0:H, 0:W]
    lat = 90 - (yy + 0.5) / H * 180
    lon = (xx + 0.5) / W * 360 - 180
    X = ll2v(lat.ravel(), lon.ravel())
    ds, a = seam_dist(X, S)
    seam = (ds < 0.35).reshape(H, W)
    img[seam] = (40, 60, 255)
    for nm, la, lo, m in HARD:
        x, y = int((lo + 180) / 360 * W), int((90 - la) / 180 * H)
        cv2.circle(img, (x, y), 3, (0, 230, 255), -1)
        cv2.circle(img, (x, y), int(m * H / 180), (0, 160, 200), 1)
    for nm, la, lo, m in SOFT:
        x, y = int((lo + 180) / 360 * W), int((90 - la) / 180 * H)
        cv2.circle(img, (x, y), 2, (255, 220, 0), -1)
    slat, slon = v2ll(S)
    for la, lo in zip(slat, slon):
        cv2.drawMarker(img, (int((lo + 180) / 360 * W), int((90 - la) / 180 * H)), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
    cv2.imwrite(out, img)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'plates.png'
    (J, S, parts), (land, b, V, Hv, Hm, Sv, Sm) = optimise()
    print('best J %.3f' % J, parts)
    dh, _ = seam_dist(Hv, S)
    for (nm, la, lo, m), d in sorted(zip(HARD, dh), key=lambda z: z[1] - z[0][3])[:8]:
        print('  hard %-14s seam at %.1f deg (margin %d)' % (nm, d, m))
    dso, _ = seam_dist(Sv, S)
    for (nm, la, lo, m), d in sorted(zip(SOFT, dso), key=lambda z: z[1] - z[0][3])[:10]:
        print('  soft %-14s seam at %.1f deg (margin %d)' % (nm, d, m))
    lat, lon = v2ll(S)
    print('SEEDS = [' + ', '.join('(%.2f, %.2f)' % (a, o) for a, o in zip(lat, lon)) + ']')
    draw(S, land, b, out)
