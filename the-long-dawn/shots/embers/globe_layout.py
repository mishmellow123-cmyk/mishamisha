"""The ember globe's plates, third pass: irregular, not a football.

A few large continental shields -- each holds a region of real flashpoints WHOLE, so no seam crosses or traces one --
and many small shards toward the rim (oceans, deserts). Plates are an additively weighted Voronoi on the sphere
(angle - w: the big shields have large w, so boundaries between unequal plates curve), computed on domain-warped
positions so the seams wander; a finer network of fissures branches off the main seams into the plates and dies
out. Every seam and every plate opens on its own delay. Checked against globe_plates.HARD / SOFT by
`python globe_layout.py` (map + numbers).
"""
import math

import numpy as np

from core import vnoise

# (lat, lon, weight deg, plate): a plate may own several seeds (a union of cells: irregular, non-convex shields)
SHIELDS = [
    (50.0, 15.0, 11.0, 'europe'), (40.0, 35.0, 11.0, 'europe'), (31.0, 49.0, 7.0, 'europe'), (39.0, 60.0, 1.5, 'europe'),   # Europe to the Gulf
    (29.0, 79.0, 7.0, 'south asia'), (28.0, 92.0, 3.0, 'south asia'),
    (31.0, 124.0, 12.0, 'east asia'), (45.0, 132.0, 4.0, 'east asia'), (16.0, 118.0, 3.0, 'east asia'),
    (46.0, -98.0, 12.0, 'north america'), (40.0, -76.0, 4.0, 'north america'),
    (64.0, 100.0, 9.0, 'siberia'),
    (84.0, -30.0, 6.0, 'siberia'),                                     # one plate spans the pole
    (64.0, -171.0, 3.0, 'bering'),
]
SHARDS = [
    (52.0, -145.0, -2.5),
    (38.0, -155.0, -1.5), (25.0, -140.0, -2.0), (30.0, 170.0, -1.5), (15.0, 160.0, -2.5), (8.0, 140.0, -3.0),
    (18.0, -175.0, -3.0), (10.0, -150.0, -3.0), (12.0, -120.0, -2.5), (14.0, -112.0, -3.5),
    (40.0, -45.0, -1.5), (31.0, -62.0, -2.0), (36.0, -30.0, -3.0),
    (6.0, -100.0, -3.5), (18.0, -36.0, -2.0),
    (19.0, 12.0, -2.0), (19.0, 39.0, -3.5), (12.0, 55.0, -3.0), (14.0, 66.0, -3.5),
    (12.0, 88.0, -3.0), (-1.0, 78.0, -3.5), (5.0, 108.0, -3.0), (4.0, 118.0, -3.0),
    # the southern hemisphere (unseen; bounds the rim plates)
    (-30.0, -160.0, 0.0), (-45.0, -110.0, 0.0), (-35.0, -60.0, 0.0), (-45.0, -15.0, 0.0), (-30.0, 30.0, 0.0),
    (-40.0, 80.0, 0.0), (-30.0, 125.0, 0.0), (-60.0, 170.0, 0.0), (-70.0, 40.0, 0.0), (-8.0, -20.0, 0.0),
    (-8.0, 60.0, 0.0), (-10.0, 150.0, 0.0), (-12.0, -100.0, 0.0), (-10.0, 105.0, 0.0),
]
_names = []
PLATES = []
for la, lo, w, nm in SHIELDS:
    if nm not in _names:
        _names.append(nm)
    PLATES.append((la, lo, w, _names.index(nm)))
NSHIELD = len(_names)
for k, (la, lo, w) in enumerate(SHARDS):
    PLATES.append((la, lo, w, NSHIELD + k))
WARP = ((1.9, 0.10), (4.3, 0.075), (9.5, 0.03))       # (frequency, amplitude rad): the seams wander, and are jagged close up
SUB_N = 260                              # the finer fissure network
BRANCH_REACH = 5.0                       # deg: fissures reach this far off a main seam
SEAM_W = 0.5                             # deg: half-gap in the weighted metric that counts as a seam (v1: 0.52)


def ll2v(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), -np.cos(lat) * np.sin(lon)], -1)


def v2ll(v):
    return np.degrees(np.arcsin(np.clip(v[..., 1], -1, 1))), np.degrees(np.arctan2(-v[..., 2], v[..., 0]))


S = ll2v(np.array([p[0] for p in PLATES]), np.array([p[1] for p in PLATES]))
W = np.radians(np.array([p[2] for p in PLATES]))
PID = np.array([p[3] for p in PLATES], np.int64)
NP = int(PID.max()) + 1


def _per_plate(d):
    """(n, seeds) weighted distances -> (n, plates): each plate's nearest seed"""
    dp = np.full((d.shape[0], NP), np.inf)
    for k in range(d.shape[1]):
        dp[:, PID[k]] = np.minimum(dp[:, PID[k]], d[:, k])
    return dp


def _sub_seeds():
    """an even (jittered Fibonacci) set: random seeds can land together and make blobs of 'fissure'"""
    r = np.random.default_rng(71)
    i = np.arange(SUB_N) + 0.5
    phi = np.arccos(1 - 2 * i / SUB_N)
    th = np.pi * (1 + 5 ** 0.5) * i
    v = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1) + r.normal(0, 0.05, (SUB_N, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


SUB = _sub_seeds()
SUB_KEEP = np.random.default_rng(72).random(SUB_N)


def _sites():
    import globe_plates as GP
    rows = [(la, lo, m) for _, la, lo, m in GP.HARD] + [(la, lo, m) for _, la, lo, m in GP.SOFT]
    v = ll2v(np.array([r_[0] for r_ in rows]), np.array([r_[1] for r_ in rows]))
    m = np.radians(np.array([r_[2] for r_ in rows], float) + 1.0)      # + the seam's own half-width and a margin
    return v, m


SITE_V, SITE_M = _sites()
SITE_L = None


def warp(P):
    Q = np.asarray(P, np.float64)
    for f, a in WARP:
        Q = Q + vnoise(Q, f, (11.3 * f, 4.1, 7.7), 2) * a
    return Q / np.linalg.norm(Q, axis=1, keepdims=True)


def evaluate(P, chunk=200000):
    """per point: plate label, second plate, main gap (deg; seam where < 2*SEAM_W), fissure flag, fissure depth.
    gap is (d2 - w2) - (d1 - w1): about twice the distance to the nearest main seam."""
    n = len(P)
    lab = np.empty(n, np.int64)
    lab2 = np.empty(n, np.int64)
    gap = np.empty(n)
    fis = np.zeros(n, bool)
    global SITE_L
    if SITE_L is None:
        Qs = warp(SITE_V)
        SITE_L = np.argmin(_per_plate(np.arccos(np.clip(Qs @ S.T, -1, 1)) - W[None, :]), axis=1)
    for c0 in range(0, n, chunk):
        Pc = P[c0:c0 + chunk]
        Q = warp(Pc)
        d = _per_plate(np.arccos(np.clip(Q @ S.T, -1, 1)) - W[None, :])
        # every flashpoint / capital / hub pulls the plate that holds it: a smooth bulge that keeps any seam at
        # least its margin away (strength 4.5 m over 3 m: enough even where the warp compresses the metric)
        ds_ = np.arccos(np.clip(Pc @ SITE_V.T, -1, 1))
        bump = 4.5 * SITE_M[None, :] * np.clip(1.0 - ds_ / (3.0 * SITE_M[None, :]), 0, None)
        for L in np.unique(SITE_L):
            d[:, L] -= bump[:, SITE_L == L].max(axis=1)
        i2 = np.argpartition(d, 1, axis=1)[:, :2]
        dd = np.take_along_axis(d, i2, 1)
        o = np.argsort(dd, 1)
        i2 = np.take_along_axis(i2, o, 1)
        dd = np.take_along_axis(dd, o, 1)
        lab[c0:c0 + chunk] = i2[:, 0]
        lab2[c0:c0 + chunk] = i2[:, 1]
        g = np.degrees(dd[:, 1] - dd[:, 0])
        gap[c0:c0 + chunk] = g
        # fissures: edges of the finer network, near a main seam, on a random half of its cells
        ds = Q @ SUB.T
        j2 = np.argpartition(-ds, 1, axis=1)[:, :2]
        a_ = np.take_along_axis(ds, j2, 1)
        sg = np.degrees(np.abs(np.arccos(np.clip(a_[:, 0], -1, 1)) - np.arccos(np.clip(a_[:, 1], -1, 1))))
        keep = (SUB_KEEP[j2[:, 0]] < 0.42) | (SUB_KEEP[j2[:, 1]] < 0.18)
        clear = (ds_ > SITE_M[None, :]).all(axis=1)                    # fissures die out before any site
        fis[c0:c0 + chunk] = (sg < 0.7) & keep & (g < 2 * BRANCH_REACH) & (g >= 2 * SEAM_W) & clear
    return lab, lab2, gap, fis


def seam_pair(lab, lab2):
    a = np.minimum(lab, lab2)
    b = np.maximum(lab, lab2)
    return a * 1000 + b


if __name__ == '__main__':
    import os
    import sys
    import cv2
    sys.path.insert(0, os.path.dirname(__file__))
    import globe_plates as GP
    out = sys.argv[1] if len(sys.argv) > 1 else 'layout.png'
    n = 700000
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    V = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    V = V[V[:, 1] > -0.05]
    lab, lab2, gap, fis = evaluate(V)
    crack = (gap < 2 * SEAM_W) | fis
    C = V[crack]
    lat, lon = v2ll(V)

    def check(sites, tag):
        bad = []
        for nm, la, lo, m in sites:
            x = ll2v(np.array(la), np.array(lo))
            d = np.degrees(np.arccos(np.clip(C @ x, -1, 1))).min()
            if d < m:
                bad.append((nm, round(float(d), 1), m))
        print(tag, 'violations:', bad)
    check(GP.HARD, 'HARD')
    check(GP.SOFT, 'SOFT')
    # seams that trace land borders: per plate pair, the share of its (visible, on-land) seam within 1 deg of a border
    land0, b0, bdist0, pop0 = GP.grids()
    main = (gap < 2 * SEAM_W) & (lat > 8)
    xb = GP.sample(bdist0, lat[main], lon[main])
    xl = GP.sample(land0.astype(np.float32), lat[main], lon[main]) > 0
    pr = seam_pair(lab[main], lab2[main])
    flagged = []
    for p_ in np.unique(pr):
        m_ = (pr == p_) & xl
        if m_.sum() < 40:
            continue
        f_ = float((xb[m_] < 1.0).mean())
        if f_ > 0.3:
            flagged.append((int(p_), round(f_, 2), int(m_.sum()), round(float(lat[main][m_].mean()), 1),
                            round(float(lon[main][m_].mean()), 1)))
    print('border-tracing seams (pair, share, n, lat, lon):', flagged)
    # plate sizes (visible hemisphere)
    vis = lat > 8
    cnt = np.bincount(lab[vis], minlength=NP).astype(float)
    rad = np.degrees(np.sqrt(cnt / cnt.sum() * 2 * np.pi * (1 - math.sin(math.radians(8))) / np.pi))
    rv = rad[cnt > 50]
    print('visible plates %d, equivalent radius min %.1f median %.1f max %.1f deg' % (len(rv), rv.min(), np.median(rv),
                                                                                       rv.max()))
    land, b, bdist, pop = GP.grids()
    H, Wd = land.shape
    img = np.zeros((H, Wd, 3), np.uint8)
    img[land] = (60, 60, 60)
    img[b] = (100, 100, 100)
    x = ((lon + 180) / 360 * Wd).astype(int) % Wd
    y = np.clip(((90 - lat) / 180 * H).astype(int), 0, H - 1)
    img[y[gap < 2 * SEAM_W], x[gap < 2 * SEAM_W]] = (40, 60, 255)
    img[y[fis], x[fis]] = (40, 140, 255)
    for nm, la, lo, m in GP.HARD:
        cv2.circle(img, (int((lo + 180) / 360 * Wd), int((90 - la) / 180 * H)), int(m * H / 180), (0, 200, 255), 1)
    for nm, la, lo, m in GP.SOFT:
        cv2.circle(img, (int((lo + 180) / 360 * Wd), int((90 - la) / 180 * H)), 2, (255, 220, 0), -1)
    cv2.imwrite(out, cv2.resize(img[0:400], (1440, 400), interpolation=cv2.INTER_AREA))
    # a polar preview (orthographic from above the pole)
    Pn = V[V[:, 1] > 0.05]
    labn = lab[V[:, 1] > 0.05]
    crn = crack[V[:, 1] > 0.05]
    pv = np.zeros((500, 500, 3), np.uint8)
    u = (250 + 240 * Pn[:, 0]).astype(int)
    v = (250 + 240 * Pn[:, 2]).astype(int)
    col = np.random.default_rng(3).integers(60, 200, (NP, 3)).astype(np.uint8)
    pv[v, u] = col[labn]
    pv[v[crn], u[crn]] = (255, 255, 255)
    cv2.imwrite(out.replace('.png', '_polar.png'), pv)
