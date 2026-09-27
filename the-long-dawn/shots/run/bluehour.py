"""THE BLUE HOUR (cut A, R7 as changed by REVISION 1 and the H5 calls) and A20's title sky.
A19: bars 74-78, A cut frames 5840-6239; A20: bars 79-81, 6240-6479 (EDIT composites the title over these).

The line has come off the arete onto a broad snow shoulder among the watch-fires. The great lantern is set down on the
snow (not on a cairn, not raised) and about forty people sit round it, unroped, all turned to the east (so no face is
ever lit). The fires are fed. From bar 75 b1 the east pales to rose where the false dawn once glowed; the sun stays
below the horizon (-6 to -2.4 deg): no disc, no paling rule, no alignment. From bar 76 b1 the camera drifts toward
the edge, where the cloud sea thins over a long valley: through the thinning cloud, fields and woods, a river taking
the sky, and the smoke of many hearths rising in thin grey-blue columns that flatten under the inversion. T14 (bars
77-78) plays over the people together, who stay in the foreground. In A20 the camera lifts to the rose sky over the
valley for the title, and the sky keeps paling.

Render: pass 1 is the world with its cloud sea; pass 2 (only inside the valley's screen box) is the same world with
the cloud removed (a leading hole row: world.py's ADDITIVE -5 rows), with a valley floor below the snow line
(Q[19]) and the low mist capped (fogp[8]). The smoke, the river and the hearth glows go into pass 2; then the cloud
of pass 1 is laid over it with an opacity that thins over the valley (tattered banks and veils, thinning as the
light comes). Then the sky, the stars, the people (sdfppl), the fires and the lantern.

  python bluehour.py --frames 5840,6080,6400 --scale 0.5 --out t1 --procs 3    # stills -> renders/bluehour_A/t1/
  python bluehour.py --range 5840-6479 --procs 8 --skip                         # finals (A cut frames)
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault('NUMBA_NUM_THREADS', '1')   # the pool size: set it in the environment for --threads > 1

import numpy as np                                  # noqa: E402
from numba import njit, prange                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import fire2 as F2          # noqa: E402
import sdfppl as SP         # noqa: E402
import crossing as X        # noqa: E402
from mt import sky as SK, fire as F   # noqa: E402
from mt.noise import fbm2   # noqa: E402

FPS = 24.0
CUT0 = 5840                                      # A19 starts on A's cut frame 5840 (bar 74)
N19 = 400                                        # A19: shot frames 0-399; A20 (title sky): 400-639
NFR = 640
lin = PI.CM.lin
UP = X.UP
E3, S3 = X.E3, X.S3
SUN_AZ = -30.0                                   # the world's sunrise azimuth: where the false dawn glowed


def _ss(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def bar_t(b):
    """Shot time (s) of bar b's downbeat (bar 74 = 0)."""
    return (b - 74) * 80.0 / FPS


def dawn_k(t):
    """0 through bar 74 (the set-down: blue, the east only a cold pallor); from bar 75 b1 the rose grows."""
    return _ss((t - bar_t(75)) / (NFR / FPS - bar_t(75)) * 1.25) if t > bar_t(75) else 0.0


def sun_el(t):
    """-6.3 deg through bar 74, climbing from bar 75 b1 to about -2.4 by the end of A20: never a disc."""
    return -6.3 + 3.9 * dawn_k(t)


def sun_dir(t):
    e, a = math.radians(sun_el(t)), math.radians(SUN_AZ)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


# ------------------------------------------------------------------ the set: a broad shoulder past the east summit ---
SH_U, SH_V, SH_TOP = 330.0, 14.0, 192.0


# ------------------------------------------------------------------ the valley under the cloud sea ---
# bh_probe.py's map (renders/_farmtest/bluehour_a_probe): under the cloud sea east of the shoulder the old ranges'
# floor (< -1500 m) lies in a chain of basins along the dawn, walled off from the shoulder by ridges. A glacial
# trough (world.valley_row, ADDITIVE) joins them into one long valley toward the rose: a broad floor of fields and
# woods, walls steepening into the island ranges. Knots: (metres east of the shoulder along VY, metres to the
# right (south), floor height).
VY = -30.0                                       # the east: where the false dawn glowed, where the rose comes
VAL_KNOTS = [(5600.0, -350.0, -1650.0), (11000.0, 180.0, -1685.0), (17500.0, -260.0, -1715.0),
             (25000.0, 300.0, -1750.0), (34000.0, -150.0, -1785.0)]
VAL_HW = 780.0                                   # half-width of the flat floor
SNOW_LINE = -1250.0                              # Q[19]: fields and woods below it


def _vbasis():
    a = math.radians(VY)
    f = np.array([math.sin(a), 0.0, math.cos(a)])
    r = np.array([math.cos(a), 0.0, -math.sin(a)])       # right of f (this world's convention)
    return f, r


def v_point(fw, rt):
    f, r = _vbasis()
    return X.uv(SH_U, SH_V) + f * fw + r * rt


def valley_rows():
    rows = []
    for k in range(len(VAL_KNOTS) - 1):
        (f0, r0, y0), (f1, r1, y1) = VAL_KNOTS[k], VAL_KNOTS[k + 1]
        rows.append(WD.valley_row(v_point(f0, r0), v_point(f1, r1), y0, y1, hw=VAL_HW * (1.0 + 0.12 * k),
                                  slope=0.35, curv=3.0e-4, k=90.0, noise=14.0, seed=31 + k))
    return np.array(rows)


def build_set():
    X.wfs()                                       # the crossing's final set (arete, summits, fire rocks, ranges)
    p = X.uv(SH_U, SH_V)
    r = WD.crag_row(p[0], p[2], SH_TOP, L=34.0, s_hi=0.30, s_lo=1.35, aniso=1.5, ang=math.radians(X.YAW_E + 10.0),
                    seed=77, k=14.0, detail=0.08, shelf=17.0, nf=5)
    r[14] = 0.95
    r[13] *= 2.5
    r[15] = 0.85                                  # wind-packed snow over the summit boulders (world.crag flag)
    return np.vstack([valley_rows(), X.CR, r[None, :]])       # the carve rows LEAD the table (world.carve)


CR = build_set()


def ground(P, fp=0.02):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    out = np.zeros(len(P))
    WD.heights(P[:, 0].copy(), P[:, 2].copy(), fp, CR, out)
    return out


def ground1(p):
    return float(ground(p)[0])


_VAL = None


def axis_rt(fw):
    """The valley's axis: lateral offset (m, right) at distance fw along VY."""
    return float(np.interp(fw, [k[0] for k in VAL_KNOTS], [k[1] for k in VAL_KNOTS]))


def valley():
    """The river (its lowest line), the hamlets (clusters of hearths along it) and their smoke columns."""
    global _VAL
    if _VAL is None:
        rng = np.random.default_rng(1974)
        f0, f1 = VAL_KNOTS[0][0] - 400.0, VAL_KNOTS[-1][0]
        fine = np.arange(f0, f1, 40.0)
        # the river meanders across the floor about the axis
        rl = np.array([axis_rt(a) for a in fine]) + 0.42 * VAL_HW * np.sin(fine / 1300.0 + 1.3) \
            + 0.12 * VAL_HW * np.sin(fine / 410.0 + 0.4)
        river = np.stack([v_point(a, b) for a, b in zip(fine, rl)])
        river[:, 1] = ground(river, fp=2.0) + 0.6
        # hamlets: clusters of hearths near the river on the floor (below the snow line), none near the valley head
        hearths = []
        for k in range(12):
            fw = VAL_KNOTS[0][0] + 1500.0 + (f1 - VAL_KNOTS[0][0] - 3000.0) * (k + rng.uniform(0.15, 0.85)) / 12.0
            side = 1.0 if rng.random() < 0.5 else -1.0
            rc = float(np.interp(fw, fine, rl)) + side * rng.uniform(150.0, 0.8 * VAL_HW)
            n = int(rng.integers(3, 8))
            for m in range(n):
                q = v_point(fw + rng.normal() * 150.0, rc + rng.normal() * 110.0)
                q[1] = ground1(q)
                if q[1] < SNOW_LINE - 150.0:
                    hearths.append([q[0], q[1], q[2], rng.uniform(170.0, 420.0), rng.uniform(0.0, 6.28),
                                    rng.uniform(0.7, 1.0), rng.uniform(5.0, 9.0)])
        _VAL = dict(river=river, hearths=np.array(hearths, np.float64).reshape(-1, 7))
    return _VAL


@njit(inline='always', fastmath=True)
def _hole_mask(x, z, VP, KN):
    """0 outside the thinning, 1 over the valley floor. KN: the axis polyline (x, z) knots; VP: 7 half-width |
    9 grow (0-1, with the light) | 10 t."""
    best = 1e18
    for k in range(KN.shape[0] - 1):
        ax, az = KN[k, 0], KN[k, 1]
        dx = KN[k + 1, 0] - ax
        dz = KN[k + 1, 1] - az
        L2 = dx * dx + dz * dz + 1e-9
        tc = min(max(((x - ax) * dx + (z - az) * dz) / L2, 0.0), 1.0)
        ex = x - (ax + tc * dx)
        ez = z - (az + tc * dz)
        d2 = ex * ex + ez * ez
        if d2 < best:
            best = d2
    r = math.sqrt(best) / (VP[7] * 1.55)
    m = min(max((1.2 - r) / 0.6, 0.0), 1.0)
    return m * m * (3.0 - 2.0 * m)


@njit(parallel=True, fastmath=True, cache=True)
def composite_valley(img1, D1, img2, D2, x0, y0, C, VP, KN, out_alpha):
    """Lay pass 1's cloud over pass 2 (the valley) inside the sub-window at (x0, y0): where pass 1 hit the cloud
    sea (pass 2 went further), the cloud's opacity thins with the hole mask, torn into banks and veils."""
    h2, w2 = D2.shape
    f, cx, cyy = C[7], C[8], C[9]
    for j in prange(h2):
        for i in range(w2):
            J = j + y0
            I = i + x0
            d1 = D1[J, I]
            if d1 >= 1e29:
                continue
            if D2[j, i] < d1 * 1.002 + 2.0:
                continue                           # pass 1 hit rock: nothing to see through
            xo = I + 0.5 - cx
            dxh = C[3] * f + C[5] * xo
            dzh = C[4] * f + C[6] * xo
            hl = math.sqrt(dxh * dxh + dzh * dzh)
            x = C[0] + dxh / hl * d1
            z = C[2] + dzh / hl * d1
            m = _hole_mask(x, z, VP, KN) * VP[9]
            if m <= 0.0:
                continue
            # torn, partial cloud: banks and gaps at two scales, never one clean hole (it read as an ink blot)
            n1 = fbm2(x / 700.0 + 3.1, z / 700.0, 4.0, 211)
            n2 = fbm2(x / 180.0 - VP[10] * 0.004, z / 180.0, 3.0, 212)
            a = 1.0 - m * (0.62 + 1.05 * n1 + 0.45 * n2)
            a = min(max(a, 0.0), 1.0)
            a = a * a * (3.0 - 2.0 * a)
            out_alpha[J, I] = a
            img1[J, I, 0] = img1[J, I, 0] * a + img2[j, i, 0] * (1.0 - a)
            img1[J, I, 1] = img1[J, I, 1] * a + img2[j, i, 1] * (1.0 - a)
            img1[J, I, 2] = img1[J, I, 2] * a + img2[j, i, 2] * (1.0 - a)


@njit(fastmath=True, cache=True)
def _splat_over(img, zb, sx, sy, z, rpx, a0, col, zbias):
    """Soft discs composited OVER the image (not added), depth-tested (zbias: one per disc): thin smoke."""
    H, W = img.shape[0], img.shape[1]
    for k in range(sx.shape[0]):
        R = rpx[k] * 2.2 + 1.0
        xa = int(max(0, sx[k] - R))
        xb = int(min(W, sx[k] + R + 1))
        ya = int(max(0, sy[k] - R))
        yb = int(min(H, sy[k] + R + 1))
        for py in range(ya, yb):
            for px in range(xa, xb):
                if zb[py, px] < z[k] - zbias[k]:
                    continue
                u = (px + 0.5 - sx[k]) / rpx[k]
                v = (py + 0.5 - sy[k]) / rpx[k]
                q = u * u + v * v
                if q > 4.8:
                    continue
                a = a0[k] * math.exp(-q * 0.9)
                img[py, px, 0] = img[py, px, 0] * (1.0 - a) + col[k, 0] * a
                img[py, px, 1] = img[py, px, 1] * (1.0 - a) + col[k, 1] * a
                img[py, px, 2] = img[py, px, 2] * (1.0 - a) + col[k, 2] * a


def draw_valley(img, zb, scam, t, SPa, fogp):
    """Into pass 2: the river taking the sky, dim hearth glows, and the hearth smoke (thin grey-blue columns that
    rise straight in the still air, lean a little in a breeze aloft and flatten under the inversion)."""
    V = valley()
    k = dawn_k(t)
    cam = scam.pos

    def veil(P, col):
        d = np.linalg.norm(P - cam[None, :], axis=1)
        tau = np.array([WD.height_fog_tau(float(dd), float(cam[1]), float(p[1]), fogp[0], fogp[1]) for dd, p in zip(d, P)])
        tr = np.exp(-tau)[:, None]
        return col * tr + fogp[None, 5:8] * (1.0 - tr), d

    # the river: the sky's own colours laid along the floor, broken where banks and trees hide it
    R = V['river']
    sx, sy, z = scam.project(R)
    sky_c = lin('#7D86B6') * (0.30 + 0.20 * k) * 0.7 + SPa[9:12] * 0.55 * (0.2 + 0.8 * k)
    ok = z > 50.0
    if ok.sum() > 2:
        brk = 0.55 + 0.45 * np.sin(np.arange(len(R)) * 0.37) * np.sin(np.arange(len(R)) * 0.113 + 1.0)
        cols, d = veil(R, np.tile(sky_c, (len(R), 1)))
        wpx = np.maximum(26.0 * scam.f / np.maximum(z, 1.0), 0.45)
        a0 = np.clip(0.75 * brk * np.minimum(wpx / 0.8, 1.0), 0.0, 0.9)
        # dense samples along the polyline
        n = len(R)
        ss_ = []
        for m in range(n - 1):
            if not (ok[m] and ok[m + 1]):
                continue
            L = math.hypot(sx[m + 1] - sx[m], sy[m + 1] - sy[m])
            q = max(int(L / max(0.5 * wpx[m], 0.5)), 1)
            for i in range(q):
                u = i / q
                ss_.append((sx[m] + (sx[m + 1] - sx[m]) * u, sy[m] + (sy[m + 1] - sy[m]) * u,
                            z[m], wpx[m] * 0.5, a0[m] * 0.6, cols[m]))
        if ss_:
            A = np.array([s[:5] for s in ss_], np.float64)
            Cc = np.array([s[5] for s in ss_], np.float64)
            _splat_over(img, zb, A[:, 0], A[:, 1], A[:, 2], np.maximum(A[:, 3], 0.45), A[:, 4], Cc,
                        np.maximum(A[:, 2] * 0.01, 20.0))
    # the smoke columns
    Hh = V['hearths']
    if len(Hh) == 0:
        return
    wind_hi = np.array(X.E3) * -0.35 + np.array(X.S3) * 0.25       # a light air aloft, drifting west-south-west
    pts, rad, alp, col = [], [], [], []
    top_c = lin('#9AA3C8') * (0.34 + 0.22 * k) + SPa[9:12] * 0.10 * k
    low_c = lin('#5E6690') * (0.26 + 0.14 * k)
    for (x, y, z_, Hc, ph, g, w0) in Hh:
        n = 64
        for m in range(n):
            u = (m + 0.5) / n                      # 0..1 the rise, then the spread under the inversion
            rise = Hc * min(u / 0.8, 1.0)
            spread = max(u - 0.8, 0.0) / 0.2
            wob = 6.0 * u * math.sin(2 * math.pi * (rise / 140.0 - t * 1.4 / 140.0) + ph)
            drift = wind_hi * (rise ** 2 / Hc * 0.35 + spread * Hc * 0.55)
            p = np.array([x, y + 2.0 + rise + 12.0 * spread, z_]) + drift + np.array(X.S3) * wob
            pts.append(p)
            rad.append((w0 + 16.0 * u ** 1.3 + 26.0 * spread) * g)
            alp.append(0.34 * g * min(u / 0.04, 1.0) * (1.0 - 0.55 * u) * (1.0 - 0.7 * spread))
            col.append(low_c + (top_c - low_c) * min(u * 1.2, 1.0))
    P = np.array(pts)
    cols, d = veil(P, np.array(col))
    sx, sy, z = scam.project(P)
    ok = (z > 50.0) & (sx > -100) & (sx < scam.W + 100) & (sy > -100) & (sy < scam.H + 100)
    if not ok.any():
        return
    rpx = np.maximum(np.array(rad)[ok] * scam.f / z[ok], 0.5)
    a0 = np.array(alp)[ok] * np.clip(np.array(rad)[ok] * scam.f / z[ok] / 0.8, 0.3, 1.0)
    order = np.argsort(-z[ok])
    _splat_over(img, zb, sx[ok][order].astype(np.float64), sy[ok][order].astype(np.float64),
                z[ok][order].astype(np.float64), rpx[order].astype(np.float64), a0[order].astype(np.float64),
                cols[ok][order].astype(np.float64), np.maximum(z[ok][order] * 0.004, 15.0).astype(np.float64))
    # hearth glows: tiny, dim, warm (a window, a door left open), veiled like everything down there
    gl = []
    for (x, y, z_, Hc, ph, g, w0) in Hh:
        gl.append([x, y + 3.0, z_])
    G = np.array(gl)
    gc, d = veil(G, np.tile(np.array([1.0, 0.55, 0.22]) * 0.9, (len(G), 1)))
    gx, gy, gz = scam.project(G)
    for m in range(len(G)):
        if gz[m] < 50.0:
            continue
        I = 0.010 * (1.0 - 0.5 * k) * (4000.0 / max(d[m], 2000.0)) * F.flicker(t, 700 + m, 0.4)
        F2.glow(img, zb, gx[m], gy[m], 0.7, I, z=gz[m], zbias=40.0)


# ------------------------------------------------------------------ the sky ---
@njit(parallel=True, fastmath=True, cache=True)
def sky_pass(img, dist, C, SP_):
    """SP_: 0-2 sun dir | 3-5 zenith | 6-8 horizon (away) | 9-11 rose (toward the sun) | 12 band width (rad)
    | 13 warm azimuth width | 14 aureole gain | 15 pallor (the cold paling before the rose)."""
    H, W = img.shape[0], img.shape[1]
    f, cx, cyy = C[7], C[8], C[9]
    for j in prange(H):
        for i in range(W):
            if dist[j, i] < 1e8:
                continue
            xo = i + 0.5 - cx
            dxh = C[3] * f + C[5] * xo
            dzh = C[4] * f + C[6] * xo
            vy = cyy - (j + 0.5)
            nn = math.sqrt(dxh * dxh + dzh * dzh + vy * vy)
            dx, dy, dz = dxh / nn, vy / nn, dzh / nn
            el = math.asin(min(max(dy, -1.0), 1.0))
            hl = math.sqrt(dx * dx + dz * dz) + 1e-9
            shl = math.sqrt(SP_[0] * SP_[0] + SP_[2] * SP_[2]) + 1e-9
            caz = (dx * SP_[0] + dz * SP_[2]) / (hl * shl)
            daz = math.acos(min(max(caz, -1.0), 1.0))
            warm = math.exp(-(daz / SP_[13]) ** 2)
            hr = SP_[6] + (SP_[9] - SP_[6]) * warm
            hg = SP_[7] + (SP_[10] - SP_[7]) * warm
            hb = SP_[8] + (SP_[11] - SP_[8]) * warm
            u = min(max(el, 0.0) / (0.05 + 0.09 * warm), 1.0) ** 0.75
            r = hr + (SP_[3] - hr) * u
            g = hg + (SP_[4] - hg) * u
            b = hb + (SP_[5] - hb) * u
            band = math.exp(-max(el, 0.0) / SP_[12]) * warm
            r += 0.55 * band * SP_[9]
            g += 0.55 * band * SP_[10]
            b += 0.55 * band * SP_[11]
            pal = SP_[15] * math.exp(-max(el, 0.0) / 0.10) * (0.35 + 0.65 * warm)
            r += pal * 0.62
            g += pal * 0.70
            b += pal * 0.86
            cosg = dx * SP_[0] + dy * SP_[1] + dz * SP_[2]
            ang = math.acos(min(max(cosg, -1.0), 1.0))
            au = SP_[14] * math.exp(-ang / 0.25)
            r += au * SP_[9]
            g += au * SP_[10]
            b += au * SP_[11]
            img[j, i, 0] = r
            img[j, i, 1] = g
            img[j, i, 2] = b


def sky_params(t):
    k = dawn_k(t)
    SPa = np.zeros(16)
    SPa[0:3] = sun_dir(t)
    SPa[3:6] = lin('#17285F') * (0.58 + 0.30 * k)          # deep blue overhead (bar 74 is the blue hour, not night)
    SPa[6:9] = lin('#4E5B8E') * (0.40 + 0.26 * k)          # lilac-blue horizon away from the east
    SPa[9:12] = lin('#E39A86') * (0.03 + 0.92 * k)         # rose, low in the east, only from bar 75 b1
    SPa[12] = 0.035 + 0.02 * k
    SPa[13] = 0.55
    SPa[14] = 0.02 + 0.08 * k
    SPa[15] = 0.055 * (1.0 - 0.5 * k)                      # the cold pallor that comes before the rose
    return SPa


def light(t):
    """The land in the blue hour: blue skylight, a faint rose key from the paling east, no moon, no sun."""
    Lk, amb, S, fogp, Q = WD.night_light()
    k = dawn_k(t)
    # the rose east is a broad area of sky, not a sun: a weak, high, very soft key; the blue skylight does most
    e, a = math.radians(14.0), math.radians(SUN_AZ)
    Lk = np.r_[np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)]),
               lin('#F0B4A0') * (0.55 + 0.45 * k) + lin('#8C9AD0') * 0.45 * (1.0 - k)]
    amb = lin('#5B6FAA') * (0.26 + 0.20 * k)
    fogc = lin('#6C78A8') * (0.19 + 0.11 * k)
    fogp = np.array([5.0e-5, 1 / 1500.0, 2.2e-4, 1 / 140.0, 2.5, fogc[0], fogc[1], fogc[2]])
    Q = Q.copy()
    Q[0] = 0.04 + 0.13 * k
    Q[11] = 1.2
    Q[7] = 0.04                                  # no sparkle: blue-hour snow is matte
    Q[18] = 1.0
    Q[3] = 0.30
    Q[4] = 0.54
    Q[13] = 1.8
    return Lk, amb, S, fogp, Q


def light_valley(t):
    """Pass 2: the floor under the cloud: fields and woods below the snow line, hazy, not opaque."""
    Lk, amb, S, fogp, Q = light(t)
    k = dawn_k(t)
    Q = Q.copy()
    Q[19] = SNOW_LINE
    Q[0] *= 0.55                                 # the key is weaker down there, under the remaining cloud
    amb = amb * 0.80
    # the valley air is the cloud's own lilac-grey (a darker fog read as a black hole in the cloud sea)
    fogc = lin('#7F88B8') * (0.52 + 0.25 * k)
    fogp2 = np.array([6.0e-5, 1 / 1500.0, 1.6e-5, 1 / 140.0, 1.2, fogc[0], fogc[1], fogc[2], 120.0])
    return Lk, amb, S, fogp2, Q


# ------------------------------------------------------------------ the great lantern, v3 housing ---
# The director's call (27 Sep): plain, timeless iron and horn: straight corner posts, flat horn panes, a stepped
# pyramid roof with a vent and a bail, a flat base plate on four feet; no onion base, no arched panels. The core
# stays the heart of light (the A17 -> A18 match cut), with a flame's flicker and a faint warm edge (HEART CONTRACT
# v2 in the RUN-A NOTES). Built here from Scene primitives only (the crossing, frozen while it renders, still uses
# sdfppl.great_lantern; both switch to one shared version after it lands).
def lantern_v3(sc, c, yaw, pane_rgb, gain, k=1.0):
    """Centred at c (the heart). Returns the base's height below c (so a caller can stand it on the snow)."""
    up = np.array([0.0, 1.0, 0.0])
    fw = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    rt = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    R = 0.19 * k
    hb = 0.30 * k
    b = 0.014 * k

    def P(x, y, z):
        return c + rt * x + up * y + fw * z

    sc.begin(rgb=(0.03, 0.03, 0.03), emit=gain, glass=pane_rgb, zbias=0.05)
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            sc.box(P(sx * R, 0.0, sz * R), (0.017 * k, hb + 0.02 * k, 0.017 * k), yaw=yaw, rnd=0.004 * k, mat=1)
    for y, t_ in ((hb, b), (-hb, b), (-0.12 * hb, 0.6 * b)):
        for sz in (-1.0, 1.0):
            sc.box(P(0.0, y, sz * R), (R, t_, t_), yaw=yaw, rnd=0.003 * k, mat=1)
        for sx in (-1.0, 1.0):
            sc.box(P(sx * R, y, 0.0), (t_, t_, R), yaw=yaw, rnd=0.003 * k, mat=1)
    # the horn panes (flat, straight-sided)
    for sz in (-1.0, 1.0):
        sc.box(P(0.0, 0.0, sz * R * 0.985), (R, hb, 0.004), yaw=yaw, glass=True, mat=0)
    for sx in (-1.0, 1.0):
        sc.box(P(sx * R * 0.985, 0.0, 0.0), (0.004, hb, R), yaw=yaw, glass=True, mat=0)
    # the roof: a stepped pyramid, a vent, a bail
    sc.box(P(0.0, hb + 0.03 * k, 0.0), (1.16 * R, 0.018 * k, 1.16 * R), yaw=yaw, rnd=0.004 * k, mat=1)
    sc.box(P(0.0, hb + 0.085 * k, 0.0), (0.80 * R, 0.042 * k, 0.80 * R), yaw=yaw, rnd=0.01 * k, mat=1)
    sc.box(P(0.0, hb + 0.150 * k, 0.0), (0.46 * R, 0.030 * k, 0.46 * R), yaw=yaw, rnd=0.008 * k, mat=1)
    sc.box(P(0.0, hb + 0.200 * k, 0.0), (0.040 * k, 0.026 * k, 0.040 * k), yaw=yaw, rnd=0.004 * k, mat=1)
    sc.box(P(0.0, hb + 0.232 * k, 0.0), (0.060 * k, 0.007 * k, 0.060 * k), yaw=yaw, rnd=0.003 * k, mat=1)
    a0, a1 = P(-0.05 * k, hb + 0.24 * k, 0.0), P(-0.035 * k, hb + 0.31 * k, 0.0)
    a2, a3 = P(0.035 * k, hb + 0.31 * k, 0.0), P(0.05 * k, hb + 0.24 * k, 0.0)
    for u, v in ((a0, a1), (a1, a2), (a2, a3)):
        sc.cone(u, v, 0.008 * k, 0.008 * k, 1, 0.0)
    # the base: a flat plate on four short feet
    sc.box(P(0.0, -hb - 0.025 * k, 0.0), (1.10 * R, 0.020 * k, 1.10 * R), yaw=yaw, rnd=0.005 * k, mat=1)
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            sc.box(P(sx * 0.9 * R, -hb - 0.060 * k, sz * 0.9 * R), (0.024 * k, 0.016 * k, 0.024 * k), yaw=yaw,
                   rnd=0.006 * k, mat=1)
    sc.end()
    return hb + 0.076 * k


def heart_flicker(f_cut):
    """HEART CONTRACT v2: the core's flame flicker, a function of A's cut frame only."""
    return 1.0 + 0.06 * math.sin(2 * math.pi * f_cut * 7.3 / 24.0) \
        + 0.04 * math.sin(2 * math.pi * f_cut * 11.9 / 24.0 + 1.3) + 0.03 * math.sin(2 * math.pi * f_cut * 3.1 / 24.0 + 2.2)


# ------------------------------------------------------------------ the people ---
# Local frame of the gathering: origin at the lantern; F = the way they face (the valley and the dawn), R = its
# right. Everything is placed in (f, r) metres and dropped onto the snow. The lantern stands LANT_BACK metres short
# of the shoulder's edge (measured, edge_f), so the camera's drift toward the edge opens the valley below.
LANT_BACK, LANT_R = 5.5, -1.0
FACE_YAW = VY                                  # they face the dawn and the valley


def _fr_basis():
    return _vbasis()


_EDGE = None


def edge_f():
    """How far along VY from the shoulder's centre the snow top ends (where the fall steepens past ~27 deg)."""
    global _EDGE
    if _EDGE is None:
        f, r = _vbasis()
        c = X.uv(SH_U, SH_V)
        ss_ = np.arange(0.0, 160.0, 0.5)
        P = np.stack([c + f * a + r * LANT_R for a in ss_])
        h = ground(P)
        sl = (h[:-6] - h[6:]) / 3.0            # drop over the next 3 m
        k = int(np.argmax(sl > 0.5)) if (sl > 0.5).any() else len(ss_) - 7
        _EDGE = float(ss_[k])
    return _EDGE


def lp(fw, rt, dy=0.0):
    """A point on the snow at (fw, rt) from the lantern."""
    f, r = _fr_basis()
    c = X.uv(SH_U, SH_V) + f * (edge_f() - LANT_BACK + fw) + r * (LANT_R + rt)
    c[1] = ground1(c) + dy
    return c


PAL = [(0.030, 0.030, 0.034), (0.022, 0.026, 0.046), (0.028, 0.034, 0.026), (0.046, 0.025, 0.022),
       (0.040, 0.034, 0.030), (0.032, 0.036, 0.043), (0.050, 0.044, 0.038), (0.026, 0.022, 0.030)]
_PEOPLE = None


def people():
    """About forty: the two bearers and the rest of the line, sitting close round the lantern in small groups,
    pairs leaning together, a few lying back, the keepers at the fires. All face the dawn (+-35 deg)."""
    global _PEOPLE
    if _PEOPLE is not None:
        return _PEOPLE
    rng = np.random.default_rng(7474)
    out = []
    # the two bearers, nearest the lantern, on its camera side: one's arm round the other, a head on a shoulder
    out.append(dict(f=-1.05, r=0.55, yaw=-4.0, pose='knees', h=1.02, pal=0, tilt=0.0, pair=1))
    out.append(dict(f=-1.00, r=1.12, yaw=4.0, pose='cross', h=0.97, pal=5, tilt=0.32, pair=-1))
    taken = [(-1.05, 0.55), (-1.00, 1.12)]
    # small groups round the lantern (never a closed ring; seen at eye level, never from above)
    # (the camera's corridor, f < -3 and r in 0.5..3.5, stays clear; nobody sits within 1 m of the edge at f = 5.5)
    centres = [(-3.6, -2.0), (-2.2, -4.6), (0.8, -3.0), (2.6, -0.8), (3.4, 2.2), (1.2, 3.6), (-1.0, 5.2),
               (-4.8, -5.2), (3.8, -4.4), (-0.8, -6.8), (3.0, 5.4), (-6.2, -1.6)]
    poses = ['knees'] * 9 + ['cross'] * 5 + ['back'] * 4 + ['side'] * 4 + ['lie'] * 2
    for (cf, cr) in centres:
        n = int(rng.integers(3, 6))
        for m in range(n):
            for _try in range(30):
                f_ = cf + rng.normal() * 0.8
                r_ = cr + rng.normal() * 0.8
                if f_ * f_ + r_ * r_ < 1.6 ** 2:
                    continue                   # the lantern's pool of light stays open
                if f_ > LANT_BACK - 1.0 or (f_ < -2.6 and 0.3 < r_ < 3.8):
                    continue                   # not over the edge; not in the camera's way
                if all((f_ - a) ** 2 + (r_ - b) ** 2 > 0.78 ** 2 for a, b in taken):
                    break
            else:
                continue
            taken.append((f_, r_))
            pose = poses[int(rng.integers(0, len(poses)))]
            out.append(dict(f=f_, r=r_, yaw=float(rng.normal() * 14.0 - 8.0 * np.sign(r_) * 0.0), pose=pose,
                            h=float(0.93 + 0.12 * rng.random()), pal=int(rng.integers(0, len(PAL))),
                            tilt=float(rng.normal() * 0.10), pair=0, turn=float(rng.normal() * 0.25)))
    # pairs: a few neighbours lean together
    for q in range(2, len(out) - 1, 5):
        a, b = out[q], out[q + 1]
        if (a['f'] - b['f']) ** 2 + (a['r'] - b['r']) ** 2 < 1.3 ** 2 and a['pose'] != 'lie' and b['pose'] != 'lie':
            b['tilt'] = 0.30 if b['r'] > a['r'] else -0.30
    _PEOPLE = out[:42]
    return _PEOPLE


# camera keys (shot bar, f, r, eye height, yaw offset from VY, pitch, hfov): from bar 76 b1 it drifts toward the
# edge and rises a little (a bearer stepping up onto the rise), which opens the valley past the edge; A20 lifts to
# the rose sky over the valley for the title
# (a first pass drifted into the gathering and rose to 2.6 m: by T14's last frame the people had left the frame)
# (from 12 m back the set-down lantern was a speck behind the people: now 8.6 m, on the clear corridor to it)
CAM_KEYS = [(74.0, -8.6, 1.3, 1.58, -3.0, -8.0, 54.0), (75.95, -8.4, 1.4, 1.60, -2.6, -7.6, 54.0),
            (79.0, -7.6, 2.4, 1.90, 3.0, -5.5, 52.0), (80.4, -7.4, 2.5, 2.00, 3.5, -1.8, 52.0),
            (82.0, -7.3, 2.55, 2.02, 3.6, -1.4, 52.0)]


FIRES = [(-3.8, -8.4), (1.4, 7.8)]              # two watch-fires beside the gathering (f, r from the lantern)


def build_scene(t):
    sc = SP.Scene()
    lights = []
    f_, r_ = _fr_basis()
    k = dawn_k(t)
    # the great lantern, set down on the snow: it touches down on bar 74 b1 and rocks once as it settles
    lb = lp(0.0, 0.0)
    rock = 0.05 * math.exp(-t / 0.35) * math.sin(t * 17.0)
    heart = lb + UP * ((0.30 + 0.076) * X.LANT_K) + f_ * rock * 0.4
    hc = np.array([1.00, 0.74, 0.40])
    br = X.breath(t) * heart_flicker(CUT0 + t * FPS)
    # the horn panes glow warmer than the heart they hold
    lantern_v3(sc, heart, math.radians(X.YAW_E) + 0.3, np.array([1.0, 0.66, 0.34]), 0.50 * br, k=X.LANT_K)
    lights.append([heart[0], heart[1], heart[2], hc[0], hc[1], hc[2], 4.2 * br, 0.35])
    # the poles laid down, the rope coiled beside it
    sc.begin(rgb=(0.07, 0.05, 0.035))
    for o in (-0.22, 0.22):
        a_ = lp(-1.9, -0.7 + o, 0.03)
        b_ = lp(1.6, -1.1 + o, 0.03)
        sc.cone(a_, b_, 0.022, 0.022, 2, 0.0)
    c0 = lp(-0.2, 1.0, 0.03)
    n = 70
    prev = None
    for m in range(n + 1):
        ang = m / n * 2 * math.pi * 3.2
        rr = 0.26 + 0.05 * math.sin(m * 0.7)
        q = c0 + np.array([math.cos(ang) * rr, 0.012 * m / n, math.sin(ang) * rr])
        if prev is not None:
            sc.cone(prev, q, 0.012, 0.012, 4, 0.0)
        prev = q
    sc.end()
    # the people
    wind = (-np.array(X.E3) * 0.6 + np.array(X.S3) * 0.4) * 0.05       # the air is almost still at dawn
    heads = {}
    for i, p in enumerate(people()):
        base = lp(p['f'], p['r'])
        a = math.radians(FACE_YAW + p['yaw'])
        w = np.array([math.sin(a), 0.0, math.cos(a)])
        o = SP.seated(sc, base, w, PAL[p['pal']], h=p['h'], pose=p['pose'], tilt=p['tilt'], turn=p.get('turn', 0.0),
                      wind=wind, flutter=0.9 * t + 1.7 * i, peak=(i % 4 == 0),
                      breath=2 * math.pi * t / (3.6 + 0.9 * ((i * 7) % 5) / 5.0) + i, ground=ground1,
                      fold_phase=1.3 * i)
        heads[i] = o
    # the first bearer's arm round the second's back (they carried it together)
    A, B = heads[0], heads[1]
    hand = B['right'] + UP * 0.05 - B['w'] * 0.03
    mid = 0.5 * (A['right'] + hand) - A['w'] * 0.16 + UP * 0.03
    sc.begin(rgb=PAL[0])
    sc.cone(A['right'], mid, 0.072, 0.066, 0, 0.04)
    sc.cone(mid, hand, 0.066, 0.058, 0, 0.04)
    sc.cone(hand, hand, 0.045, 0.045, 0, 0.015)
    sc.end()
    # the watch-fires, fed: a ring of stones, a keeper kneeling on the far side in profile (never front-lit)
    for kf, (ff, fr) in enumerate(FIRES):
        p = lp(ff, fr)
        sc.begin(rgb=(0.07, 0.07, 0.075))
        rng = np.random.default_rng(80 + kf)
        for m in range(12):
            a_ = 2 * math.pi * m / 12 + rng.uniform(-0.15, 0.15)
            q = p + np.array([math.cos(a_), 0.0, math.sin(a_)]) * (0.48 + rng.uniform(-0.04, 0.05))
            hs = rng.uniform(0.6, 1.0)
            sc.box(q + UP * 0.05 * hs, (0.09 * hs, 0.07 * hs, 0.08 * hs), yaw=a_, pitch=rng.uniform(-0.3, 0.3),
                   rnd=0.025, mat=5, k=0.0)
        sc.end()
        # the keeper kneels beside the fire, across the camera's line (in profile): the fire lights a hood's edge
        kp = lp(ff + 0.1, fr + (-0.95 if kf == 0 else 0.95))
        wk = p - kp
        wk[1] = 0.0
        wk /= np.linalg.norm(wk)
        feed = _ss((math.sin(2 * math.pi * t / 7.0 + 2.0 * kf) - 0.55) / 0.3)
        SP.seated(sc, kp, wk, (0.028, 0.024, 0.022), h=0.98, pose='kneel', lean=0.25 + 0.25 * feed,
                  reach=(p + UP * 0.25) if feed > 0.1 else None, ground=ground1, fold_phase=2.0 + kf, wind=wind)
        fl = F.flicker(t, 60 + kf)
        lights.append([p[0], p[1] + 0.8, p[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2],
                       6.5 * fl * (1.0 + 0.4 * feed), 0.8])
    return sc, lights, heart, hc, br


# ------------------------------------------------------------------ camera ---
def cam_state(t):
    """Interpolate CAM_KEYS (eased between keys) at shot time t."""
    b = 74.0 + t * FPS / 80.0
    K = CAM_KEYS
    if b <= K[0][0]:
        return K[0][1:]
    for k in range(len(K) - 1):
        if b <= K[k + 1][0]:
            u = _ss((b - K[k][0]) / (K[k + 1][0] - K[k][0]))
            return tuple(a + (c - a) * u for a, c in zip(K[k][1:], K[k + 1][1:]))
    return K[-1][1:]


def camera(frame, W=1920, H=804):
    """A standing bearer's eye at the back of the gathering. Bars 74-75: nearly still, looking over the seated backs
    to the lantern and the pale east. From bar 76 b1 it drifts forward and right toward the edge, rising a little,
    so the valley opens below; the people stay in the lower frame through T14. A20: it lifts to the rose sky over
    the valley (the title's sky), the valley still below."""
    t = frame / FPS
    f_, r_ = _fr_basis()
    fw, rt, eye, dyaw, pitch, hfov = cam_state(t)
    base = X.uv(SH_U, SH_V) + f_ * (edge_f() - LANT_BACK + fw) + r_ * (LANT_R + rt)
    base[1] = ground1(base) + eye
    return RC.RCam(base, VY + dyaw, pitch, 0.0, hfov, W, H)


# ------------------------------------------------------------------ frame ---
FINISH = dict(exposure=1.05, bloom_strength=0.07, bloom_threshold=1.0, streak_strength=0.0, vignette_amount=0.25)
_STARS = None


def _sub_window(scam, x0, x1, y0, y1):
    a0 = -scam.cx
    b1 = scam.cyy
    return RC.SrcCam(scam.pos, scam.yaw, scam.f, a0 + x0, a0 + x1, b1 - y1, b1 - y0)


def valley_box(scam, t):
    """Source-pixel box of the thinning (projected outline at the floor and at the cloud top), or None."""
    pts = []
    for fw in np.linspace(VAL_KNOTS[0][0] - 2000.0, VAL_KNOTS[-1][0] + 1500.0, 16):
        for rt in (-2.2 * VAL_HW, 0.0, 2.2 * VAL_HW):
            for y in (-1850.0, WD.CLOUD_Y + 200.0):
                q = v_point(fw, rt + axis_rt(fw))
                q[1] = y
                pts.append(q)
    sx, sy, z = scam.project(np.array(pts))
    if not (z > 100.0).any():
        return None
    sx, sy = sx[z > 100.0], sy[z > 100.0]
    x0, x1 = int(max(0, math.floor(sx.min()) - 4)), int(min(scam.W, math.ceil(sx.max()) + 4))
    y0, y1 = int(max(0, math.floor(sy.min()) - 4)), int(min(scam.H, math.ceil(sy.max()) + 4))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, x1, y0, y1


def render(frame, scale=1.0, ss=1.5):
    global _STARS
    t = frame / FPS
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = camera(frame, W, H)
    fr = PI.Frame(tcam, ss)
    Lk, amb, S, fogp, Q = light(t)
    sc, lights, heart, hc, br = build_scene(t)
    scam = fr.src
    C = scam.params()
    Pp = np.array([scam.pos[0], scam.pos[2], t, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(Pp, CR, C, 0.2, 90000.0, 0.0035, 0.35, 700.0, 9, D)
    fr.img = np.zeros((scam.H, scam.W, 3), np.float32)
    fr.zb = np.zeros((scam.H, scam.W), np.float32)
    fr.dist = np.zeros((scam.H, scam.W), np.float32)
    PLs = [[p[0], p[1], p[2], 1.1] for p in (lp(ff, rr) for ff, rr in FIRES)]
    LT = np.array(lights, np.float64).reshape(-1, 8)
    WD.shade(C, D, Pp, CR, S, LT, Lk, Q, amb, fogp, fr.img, fr.zb, fr.dist, np.array(PLs, np.float64).reshape(-1, 4))
    SPa = sky_params(t)
    # pass 2: the valley under the cloud (inside its screen box only), then the cloud laid over it, thinning
    box = valley_box(scam, t)
    if box is not None:
        x0, x1, y0, y1 = box
        sub = _sub_window(scam, x0, x1, y0, y1)
        C2 = sub.params()
        CR2 = np.vstack([WD.hole_row(scam.pos[0], scam.pos[2], 90000.0, 6000.0, edge=0.01)[None, :], CR])
        D2 = np.zeros((sub.H, sub.W))
        WD.march(Pp, CR2, C2, 0.2, 90000.0, 0.0035, 0.35, 700.0, 9, D2)
        Lk2, amb2, S2, fogp2, Q2 = light_valley(t)
        img2 = np.zeros((sub.H, sub.W, 3), np.float32)
        zb2 = np.zeros((sub.H, sub.W), np.float32)
        di2 = np.zeros((sub.H, sub.W), np.float32)
        WD.shade(C2, D2, Pp, CR2, S2, np.zeros((0, 8)), Lk2, Q2, amb2, fogp2, img2, zb2, di2, np.zeros((0, 4)))
        draw_valley(img2, zb2, sub, t, SPa, fogp2)
        grow = 0.75 + 0.25 * _ss(t / bar_t(77))
        VP = np.zeros(11)
        VP[7] = VAL_HW
        VP[9] = grow
        VP[10] = t
        KN = np.array([[p[0], p[2]] for p in (v_point(f0, r0) for f0, r0, _y in VAL_KNOTS)], np.float64)
        alpha = np.ones((scam.H, scam.W), np.float32)
        D2c = np.where(D2 >= 1e29, 1e30, D2)
        composite_valley(fr.img, D, img2, D2c, x0, y0, C, VP, KN, alpha)
    sky_pass(fr.img, fr.dist, C, SPa)
    if _STARS is None:
        _STARS = SK.make_stars(9000, 101, lum_scale=5.0)
    mask = (fr.dist > 1e8).astype(np.float32)
    k = dawn_k(t)
    SK.splat_stars(fr.img, scam, _STARS, mask, t=t, gain=fr.ss * fr.ss * 0.30 * (1.0 - 0.85 * k),
                   scale=PI.src_scale(fr))
    SP.render(fr.img, fr.zb, C, *sc.arrays(), LT, np.array([Lk[0], Lk[1], Lk[2], Lk[3], Lk[4], Lk[5], Q[0]]),
              amb * 1.2, fogp, scam.pos[1])
    for kf, (ff, rr) in enumerate(FIRES):
        p = lp(ff, rr)
        F2.flame(fr.img, fr.zb, scam, p + UP * 0.1, 1.15, 0.36, t, seed=31 + kf, I=10.0, lean=-0.08, zbias=0.5)
    X.draw_heart(fr.img, fr.zb, scam, heart, t, hc, 0.6 * br, PI.src_scale(fr))
    # HEART CONTRACT v2: a faint warm edge round the core
    hx, hy, hz = scam.project(heart)
    if hz > 0.2:
        F.halo(fr.img, fr.zb, hx, hy, max(0.07 * scam.f / hz, 1.0), 0.10 * br, z=hz, zbias=0.4,
               col=np.array([1.00, 0.72, 0.40]))
    img, zb, di = PI.to_target(fr)
    return img


def _work(args):
    frames, scale, ss, out, threads = args
    import numba
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    for f in frames:
        t0 = time.time()
        try:
            img = PI.look.finish(render(f - CUT0, scale, ss), **FINISH)
        except Exception:
            import traceback
            print(f'frame {f} FAILED\n' + traceback.format_exc(), flush=True)
            raise
        PI.look.save_png(PI.look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    a = ap.parse_args()
    base = os.path.join(PI.CM.ROOT, 'renders', 'bluehour_A')
    out = base if a.out is None else os.path.join(base, a.out)
    os.makedirs(out, exist_ok=True)
    if a.range:
        s_, e_ = a.range.split('-')
        frames = list(range(int(s_), int(e_) + 1, a.step))
    else:
        frames = [int(x) for x in (a.frames or str(CUT0 + 240)).split(',')]
    bad = [f for f in frames if not CUT0 <= f < CUT0 + NFR]
    if bad:
        raise SystemExit(f'frames outside A19-A20 (cut {CUT0}-{CUT0 + NFR - 1}): {bad[:5]}')
    if a.skip:
        frames = [f for f in frames if not os.path.exists(PI.look.frame_path(out, f))]
    valley()
    people()
    if a.procs <= 1:
        _work((frames, a.scale, a.ss, out, a.threads))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(c, a.scale, a.ss, out, 1) for c in chunks])


if __name__ == '__main__':
    main()
