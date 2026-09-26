"""The parchment itself and the frame furniture (border, compass rose): everything on the sheet that
is not geography. All functions of map position (map degrees)."""
import math

import numpy as np
from numba import njit, prange

import geo
import ink
from noise import fbm, gnoise, wobble1d

# folds: the sheet was folded in six (two vertical creases, one horizontal)
FOLD_X = (-60.0, 60.0)
FOLD_Y = (0.5 * (geo.MAP_Y0 + geo.MAP_Y1),)


# ------------------------------------------------------------ parchment ---

@njit(cache=True)
def _edge_dist(x, y, X0, X1, Y0, Y1, rc):
    """Signed distance inside a rounded rectangle (positive inside)."""
    cx = 0.5 * (X0 + X1)
    cy = 0.5 * (Y0 + Y1)
    hx = 0.5 * (X1 - X0) - rc
    hy = 0.5 * (Y1 - Y0) - rc
    qx = abs(x - cx) - hx
    qy = abs(y - cy) - hy
    ox = max(qx, 0.0)
    oy = max(qy, 0.0)
    out = math.sqrt(ox * ox + oy * oy) + min(max(qx, qy), 0.0) - rc
    return -out


@njit(cache=True)
def edge_d(x, y):
    """Distance (map deg) inside the parchment's worn, irregular edge."""
    d = _edge_dist(x, y, -187.5, 187.5, -72.8, 107.5, 2.5)
    d += 0.45 * fbm(x * 0.25, y * 0.25, 911, 3, 2.0, 0.5) + 0.12 * fbm(x * 2.2, y * 2.2, 912, 2, 2.0, 0.5)
    # a few bites out of the edge (wear, small tears)
    d -= 0.9 * max(0.0, gnoise(x * 0.09, y * 0.09, 913) - 0.35) * 4.0
    return d


@njit(cache=True, parallel=True)
def parchment(X0, Y1, H, W, ppd, stains, spots):
    """Returns albedo tone channels (H,W,4): [t_base, t_stain, edge_d, fold_wear] and relief h."""
    out = np.zeros((H, W, 5), np.float32)
    hgt = np.zeros((H, W), np.float32)
    dx = 1.0 / ppd
    fine_on = ppd >= 8.0
    grain_on = ppd >= 18.0
    for i in prange(H):
        y = Y1 - (i + 0.5) * dx
        for j in range(W):
            x = X0 + (j + 0.5) * dx
            # large warped tone + blotches
            qx = fbm(x * 0.02, y * 0.02, 1177, 3, 2.0, 0.5)
            qy = fbm(x * 0.02 + 5.2, y * 0.02 + 1.3, 1191, 3, 2.0, 0.5)
            big = fbm((x + 18.0 * qx) * 0.035, (y + 18.0 * qy) * 0.035, 101, 5, 2.0, 0.5)
            mid = fbm(x * 0.22, y * 0.22, 202, 4, 2.1, 0.55)
            t = 0.5 + 0.95 * big + 0.35 * mid
            g = 0.0
            if fine_on:
                g += 0.9 * fbm(x * 2.2, y * 2.2, 303, 4, 2.2, 0.55)
            if grain_on:
                # fibres whose direction wanders slowly, and the tooth of the sheet
                a = 3.0 * gnoise(x * 0.05, y * 0.05, 409)
                ca = math.cos(a)
                sa = math.sin(a)
                u = (ca * x + sa * y) * 14.0
                v = (-sa * x + ca * y) * 1.6
                g += 0.55 * gnoise(u, v, 404) + 0.35 * gnoise(u * 2.3, v * 2.1, 406) + 0.45 * gnoise(x * 30.0, y * 30.0, 405)
            # tide-mark stains
            st = 0.0
            for k in range(stains.shape[0]):
                sx, sy, R, amp, sd = stains[k, 0], stains[k, 1], stains[k, 2], stains[k, 3], int(stains[k, 4])
                ddx = x - sx
                ddy = y - sy
                r = math.sqrt(ddx * ddx + ddy * ddy)
                if r > R * 1.6:
                    continue
                ang = math.atan2(ddy, ddx)
                Rr = R * (1.0 + 0.22 * gnoise(math.cos(ang) * 1.3 + sd, math.sin(ang) * 1.3, sd)
                          + 0.06 * gnoise(math.cos(ang) * 5.0, math.sin(ang) * 5.0 + sd, sd + 1))
                inside = 1.0 / (1.0 + math.exp((r - Rr) / (0.08 * R)))
                rim = math.exp(-((r - Rr) / (0.035 * R + 0.12)) ** 2) * (1.0 if r < Rr else 0.6)
                st += amp * (0.35 * inside + 1.0 * rim)
            # foxing spots
            for k in range(spots.shape[0]):
                sx, sy, R, amp = spots[k, 0], spots[k, 1], spots[k, 2], spots[k, 3]
                ddx = x - sx
                ddy = y - sy
                if abs(ddx) > 3 * R or abs(ddy) > 3 * R:
                    continue
                r = math.sqrt(ddx * ddx + ddy * ddy) / R
                rr = r * (1.0 + 0.25 * gnoise(ddx / R * 1.7, ddy / R * 1.7, int(sx * 13.0) & 1023))
                st += amp * (0.7 * math.exp(-rr * rr * 1.5) + 0.45 * math.exp(-((rr - 1.0) / 0.22) ** 2))
            ed = edge_d(x, y)
            # fold wear
            fw = 0.0
            hh = 0.0
            for fx in (-60.0, 60.0):
                dd = abs(x - fx)
                fw = max(fw, math.exp(-(dd / 0.35) ** 2))
                hh += 0.07 * max(0.0, 1.0 - dd / 1.4) ** 2
            dd = abs(y - (0.5 * (-65.26378901391432 + 99.965853834121)))
            fw = max(fw, math.exp(-(dd / 0.35) ** 2))
            hh -= 0.06 * max(0.0, 1.0 - dd / 1.4) ** 2
            out[i, j, 0] = t
            out[i, j, 1] = st
            out[i, j, 2] = ed
            out[i, j, 3] = fw
            out[i, j, 4] = g
            # relief: cockle + folds + edge curl
            h = 0.12 * fbm(x * 0.05, y * 0.05, 707, 3, 2.0, 0.5) + hh
            if ed < 3.0:
                h += 0.25 * (1.0 - max(ed, 0.0) / 3.0) ** 2
            hgt[i, j] = h
    return out, hgt


def stain_list(seed=21):
    rng = np.random.default_rng(seed)
    # a few big tide marks, placed to look accidental (one near a corner, one crossing a fold)
    S = [(-150.0, 72.0, 11.0, 0.10), (95.0, -52.0, 8.5, 0.09), (35.0, 88.0, 6.0, 0.07),
         (-62.0, -40.0, 5.0, 0.06), (160.0, 60.0, 4.0, 0.07), (-20.0, 30.0, 13.0, 0.035)]
    return np.array([(x, y, r, a, int(rng.integers(1, 900))) for x, y, r, a in S], np.float64)


def spot_list(seed=22):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(45):
        cx = rng.uniform(-186, 186)
        cy = rng.uniform(-72, 106)
        sig = rng.uniform(2.0, 7.0)
        n = int(rng.integers(4, 20))
        for _ in range(n):
            r = math.exp(rng.uniform(math.log(0.04), math.log(0.35)))
            pts.append((cx + rng.normal(0, sig), cy + rng.normal(0, sig), r, rng.uniform(0.15, 0.55)))
    # heavier foxing toward the edges
    for _ in range(260):
        side = rng.integers(4)
        if side < 2:
            x = rng.uniform(-186, 186)
            y = (106.5 - abs(rng.normal(0, 4))) if side == 0 else (-72 + abs(rng.normal(0, 4)))
        else:
            y = rng.uniform(-72, 106)
            x = (186.5 - abs(rng.normal(0, 4))) if side == 2 else (-186.5 + abs(rng.normal(0, 4)))
        r = math.exp(rng.uniform(math.log(0.04), math.log(0.3)))
        pts.append((x, y, r, rng.uniform(0.2, 0.6)))
    return np.array(pts, np.float64)


# ------------------------------------------------------------- furniture ---

def _poly_strokes(st, P, r, d=0.95, seed=0, wob=0.0):
    P = np.asarray(P, np.float64)
    if wob > 0:
        s = ink.arclen(P)
        t = np.gradient(P, axis=0)
        t /= np.linalg.norm(t, axis=1)[:, None] + 1e-12
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        P = P + nrm * wobble1d(s, seed, 3.0, wob)[:, None]
    st.add(P, r, d)


def border():
    """Frame furniture as (strokes, solid fills [polys]) - no lettering. A double outer rule, a
    zebra band marking every ten degrees of latitude and longitude, the neatline, corner blocks."""
    st = ink.Strokes()
    fills = []
    X0, X1, Y0, Y1 = geo.MAP_X0, geo.MAP_X1, geo.MAP_Y0, geo.MAP_Y1

    def rect(o, r, d=0.95, seed=0):
        pts = np.array([[X0 - o, Y0 - o], [X1 + o, Y0 - o], [X1 + o, Y1 + o], [X0 - o, Y1 + o], [X0 - o, Y0 - o]])
        dense = np.concatenate([ink.resample(pts[k:k + 2], 0.5) for k in range(4)])
        _poly_strokes(st, dense, r, d, seed, wob=0.02)

    rect(0.0, 0.045, 0.95, 1)
    rect(0.9, 0.035, 0.9, 2)
    rect(1.75, 0.035, 0.9, 3)
    rect(2.35, 0.13, 0.97, 4)
    # zebra: alternate filled segments between the neatline and the 0.9 rule
    lons = np.arange(-180, 181, 5)
    xs = geo.lon2x(lons.astype(float))
    xs = np.sort(np.unique(np.round(np.concatenate([xs, [X0, X1]]), 6)))
    for k in range(len(xs) - 1):
        if k % 2 == 0:
            a, b = xs[k], xs[k + 1]
            fills.append(np.array([[a, Y1], [b, Y1], [b, Y1 + 0.9], [a, Y1 + 0.9]]))
            fills.append(np.array([[a, Y0 - 0.9], [b, Y0 - 0.9], [b, Y0], [a, Y0]]))
        st.add(np.array([[xs[k], Y1], [xs[k], Y1 + 0.9]]), 0.03, 0.9)
        st.add(np.array([[xs[k], Y0 - 0.9], [xs[k], Y0]]), 0.03, 0.9)
    lats = np.arange(-60, 86, 5)
    ys = np.concatenate([[Y0], geo.yproj(lats.astype(float)), [Y1]])
    ys = np.sort(np.unique(np.round(ys, 6)))
    for k in range(len(ys) - 1):
        if k % 2 == 0:
            a, b = ys[k], ys[k + 1]
            fills.append(np.array([[X0 - 0.9, a], [X0, a], [X0, b], [X0 - 0.9, b]]))
            fills.append(np.array([[X1, a], [X1 + 0.9, a], [X1 + 0.9, b], [X1, b]]))
        st.add(np.array([[X0 - 0.9, ys[k]], [X0, ys[k]]]), 0.03, 0.9)
        st.add(np.array([[X1, ys[k]], [X1 + 0.9, ys[k]]]), 0.03, 0.9)
    # corner blocks: a square with a small quatrefoil-like cross
    for cx, cy in ((X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)):
        sx = -1 if cx == X0 else 1
        sy = -1 if cy == Y0 else 1
        c = np.array([cx + sx * 1.18, cy + sy * 1.18])
        sq = np.array([[-1.17, -1.17], [1.17, -1.17], [1.17, 1.17], [-1.17, 1.17], [-1.17, -1.17]]) + c
        st.add(sq, 0.035, 0.9)
        th = np.linspace(0, 2 * np.pi, 40)
        st.add(c + np.stack([0.55 * np.cos(th), 0.55 * np.sin(th)], 1), 0.03, 0.9)
        for a in (0, np.pi / 2):
            v = np.array([math.cos(a + np.pi / 4), math.sin(a + np.pi / 4)]) * 1.05
            st.add(np.array([c - v, c + v]), 0.03, 0.9)
    return st, fills


ROSE_C = (float(geo.lon2x(-127.0)), float(geo.yproj(-43.0)))
ROSE_R = 8.0


def rose():
    """A sixteen-point compass rose (no letters): list of (paper_fill, ink_fills, strokes) in
    painter's order."""
    cx, cy = ROSE_C
    R = ROSE_R
    items = []
    th = np.linspace(0, 2 * np.pi, 180)
    ring = ink.Strokes()
    ring.add(np.stack([cx + R * np.cos(th), cy + R * np.sin(th)], 1), 0.06, 0.95)
    ring.add(np.stack([cx + 0.9 * R * np.cos(th), cy + 0.9 * R * np.sin(th)], 1), 0.03, 0.9)
    ring.add(np.stack([cx + 0.42 * R * np.cos(th), cy + 0.42 * R * np.sin(th)], 1), 0.025, 0.8)
    for k in range(64):
        a = k / 64 * 2 * np.pi
        r0 = 0.9 * R if k % 4 else 0.86 * R
        ring.add(np.array([[cx + r0 * math.sin(a), cy + r0 * math.cos(a)],
                           [cx + R * math.sin(a), cy + R * math.cos(a)]]), 0.022, 0.85)
    disc = np.stack([cx + 1.02 * R * np.cos(th), cy + 1.02 * R * np.sin(th)], 1)
    items.append((disc, [], ring))
    # points: minor first, then intercardinal, then cardinal (on top)
    for tier in (1, 2, 0):
        for k in range(16):
            if (k % 4 == 0) != (tier == 0):
                continue
            if tier == 2 and k % 4 != 2:
                continue
            if tier == 1 and k % 2 == 0:
                continue
            a = k * np.pi / 8
            if tier == 0:
                L = 1.28 * R if k else 1.5 * R
                bw = 0.13 * R
            elif tier == 2:
                L, bw = 0.9 * R, 0.1 * R
            else:
                L, bw = 0.6 * R, 0.075 * R
            u = np.array([math.sin(a), math.cos(a)])
            v = np.array([u[1], -u[0]])
            C = np.array([cx, cy])
            tip = C + u * L
            bl = C + u * bw * 0.6 - v * bw
            br = C + u * bw * 0.6 + v * bw
            half_l = np.array([C, bl, tip])
            half_r = np.array([C, tip, br])
            st = ink.Strokes()
            st.add(np.array([bl, tip, br]), 0.03, 0.95)
            st.add(np.array([C, tip]), 0.022, 0.9)
            st.add(np.array([C, bl]), 0.02, 0.85)
            st.add(np.array([C, br]), 0.02, 0.85)
            solid = [half_r] if k % 2 == 0 else [half_l]
            items.append((np.array([C, bl, tip, br]), solid, st))
    # north: a small lozenge beyond the tip
    C = np.array([cx, cy + 1.5 * R + 0.55])
    lz = np.array([[0, 0.5], [0.28, 0], [0, -0.5], [-0.28, 0]]) + C
    st = ink.Strokes()
    st.add(np.concatenate([lz, lz[:1]]), 0.03, 0.95)
    items.append((lz, [lz], st))
    # hub
    hub = np.stack([cx + 0.09 * R * np.cos(th), cy + 0.09 * R * np.sin(th)], 1)
    st = ink.Strokes()
    st.add(hub, 0.03, 0.95)
    items.append((hub, [], st))
    return items
