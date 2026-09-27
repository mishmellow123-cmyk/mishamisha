"""THE BLUE HOUR (cut A, R7 as changed by REVISION 1 and the H5 calls). A19: bars 74-78, A cut frames 5840-6239;
the title (A20, bars 79-81) plays over the same sky.

The line comes off the arete onto a broad snow shoulder among the watch-fires. The great lantern is set down on the
snow (not on a cairn, not raised); they sit and unrope; the fires are fed. From bar 75 b1 the east pales to rose
where the false dawn once glowed; the sun stays below the horizon (-6 to -2 deg): no disc, no paling rule, no
alignment. From bar 76 b1 the camera drifts toward the edge, where the cloud sea thins over the valley and the smoke
of many hearths rises in grey-blue light. T14 (bars 77-78) plays over people together: the seated, unroped bearers
and the set-down lantern stay in the foreground as the camera drifts to the smoke.

  NUMBA_NUM_THREADS=4 python bluehour.py --frames 6080 --scale 0.5 --threads 4 --out still   # a design still (T14)
  python bluehour.py --range 5840-6239 --procs 4 --skip            # finals, A cut frames -> renders/bluehour_A/
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
import dawn as DW           # noqa: E402
from mt import sky as SK, fire as F   # noqa: E402

FPS = 24.0
CUT0 = 5840                                      # A19 starts on A's cut frame 5840 (bar 74)
NFR = 400
lin = PI.CM.lin
UP = X.UP
E3, S3 = X.E3, X.S3
SUN_AZ = -30.0                                   # the world's sunrise azimuth: where the false dawn glowed


def sun_el(t):
    """Degrees: -6 at bar 74, climbing to about -3 by bar 78 (the east pales from bar 75 b1); never a disc."""
    u = min(max((t - 80 / FPS) / ((NFR - 80) / FPS), 0.0), 1.0)
    return -6.2 + 3.4 * (u * u * (3 - 2 * u))


def sun_dir(t):
    e, a = math.radians(sun_el(t)), math.radians(SUN_AZ)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


# ------------------------------------------------------------------ the set: a broad shoulder past the east summit ---
SH_U, SH_V, SH_TOP = 330.0, 14.0, 192.0


def build_set():
    X.wfs()                                       # the crossing's final set (arete, summits, watch-fire rocks)
    p = X.uv(SH_U, SH_V)
    r = WD.crag_row(p[0], p[2], SH_TOP, L=34.0, s_hi=0.30, s_lo=1.35, aniso=1.5, ang=math.radians(X.YAW_E + 10.0),
                    seed=77, k=14.0, detail=0.08, shelf=17.0, nf=5)
    r[14] = 0.95
    r[13] *= 2.5
    r[15] = 0.85                                  # wind-packed snow over the summit boulders (world.crag flag)
    return np.vstack([X.CR, r[None, :]])


CR = build_set()


def ground(P, fp=0.02):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    out = np.zeros(len(P))
    WD.heights(P[:, 0].copy(), P[:, 2].copy(), fp, CR, out)
    return out


def on_ground(u, v, dy=0.0):
    p = X.uv(u, v)
    p[1] = ground(p)[0] + dy
    return p


# ------------------------------------------------------------------ camera ---
def camera(frame, W=1920, H=804):
    """A standing bearer's eye, a few paces behind the seated pair, looking a little down over them to the cloud sea;
    from bar 76 b1 it drifts right and a little forward, turning toward the hearth smoke, and the pair and the lantern
    slide into the lower-left third (they never leave the frame)."""
    t = frame / FPS
    d = min(max((t - 160 / FPS) / ((NFR - 160) / FPS), 0.0), 1.0)
    d = d * d * (3 - 2 * d)
    base = on_ground(SH_U + 2.6 + 0.4 * d, SH_V - 0.9 + 0.9 * d, 1.45)
    yaw = X.YAW_E - 8.0 + 2.0 * d
    return RC.RCam(base, yaw, -5.0 - 1.0 * d, 0.0, 56.0, W, H)


# ------------------------------------------------------------------ the blue-hour sky ---
@njit(parallel=True, fastmath=True, cache=True)
def sky_pass(img, dist, C, SP_):
    """SP_: 0-2 sun dir | 3-5 zenith | 6-8 horizon (away) | 9-11 horizon toward the sun (rose) | 12 band width (rad)
    | 13 warm azimuth width | 14 aureole gain."""
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
            u = min(max(el, 0.0) / (0.045 + 0.075 * warm), 1.0) ** 0.8
            r = hr + (SP_[3] - hr) * u
            g = hg + (SP_[4] - hg) * u
            b = hb + (SP_[5] - hb) * u
            band = math.exp(-max(el, 0.0) / SP_[12]) * warm
            r += 0.6 * band * SP_[9]
            g += 0.6 * band * SP_[10]
            b += 0.6 * band * SP_[11]
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
    k = min(max((t - 80 / FPS) / ((NFR - 80) / FPS), 0.0), 1.0)
    SPa = np.zeros(16)
    SPa[0:3] = sun_dir(t)
    SPa[3:6] = lin('#1A2D6E') * (0.40 + 0.30 * k)          # deep blue overhead
    SPa[6:9] = lin('#56649A') * (0.28 + 0.26 * k)          # lilac-blue horizon away from the east
    SPa[9:12] = lin('#E39A86') * (0.22 + 0.70 * k)         # rose, only low in the east
    SPa[12] = 0.035
    SPa[13] = 0.55
    SPa[14] = 0.03 + 0.08 * k
    return SPa


def light(t):
    """The land in the blue hour: blue skylight, a faint rose key from the paling east, no moon, no sun."""
    Lk, amb, S, fogp, Q = WD.night_light()
    k = min(max((t - 80 / FPS) / ((NFR - 80) / FPS), 0.0), 1.0)
    # the rose east is a broad area of sky, not a sun: a weak, high, very soft key (no hard shadows from the snow's
    # bumps); the blue skylight does most of the work
    e, a = math.radians(14.0), math.radians(SUN_AZ)
    Lk = np.r_[np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)]), lin('#F0B4A0')]
    amb = lin('#5B6FAA') * (0.30 + 0.18 * k)
    fogc = lin('#707CAA') * (0.20 + 0.10 * k)
    fogp = np.array([5.0e-5, 1 / 1500.0, 2.2e-4, 1 / 140.0, 2.5, fogc[0], fogc[1], fogc[2]])
    Q = Q.copy()
    Q[0] = 0.05 + 0.12 * k
    Q[11] = 1.2
    Q[7] = 0.04                                  # no sparkle: blue-hour snow is matte
    Q[18] = 1.0
    Q[3] = 0.30
    Q[4] = 0.54
    Q[13] = 1.8
    return Lk, amb, S, fogp, Q


# ------------------------------------------------------------------ hearth smoke over the valley ---
_PL = None


def _plumes():
    """Thin smoke columns rising through the thinning cloud sea from hearths in the valley below (east/south-east
    of the shoulder, 2.5-14 km): many, irregular, leaning in a light breeze and flattening under an inversion."""
    global _PL
    if _PL is None:
        rng = np.random.default_rng(574)
        rows = []
        c = X.uv(SH_U, SH_V)
        for k in range(56):
            a = math.radians(SUN_AZ - 4.0 + 22.0 * rng.normal())
            d = 3000.0 * (4.5 ** rng.random())
            x, z = c[0] + math.sin(a) * d, c[2] + math.cos(a) * d
            y = WD.h_cloud(x, z, 8.0, 100.0) - 30.0
            Hp = rng.uniform(260.0, 620.0)
            n = 46
            for i in range(n):
                rows.append([x, y, z, Hp, (i + rng.random()) / n, rng.normal() * 0.35, rng.normal() * 0.35,
                             rng.uniform(0.6, 1.0), rng.integers(0, 999), rng.random()])
        _PL = np.array(rows)
    return _PL


def draw_smoke(fr, t, SPa):
    scam = fr.src
    PL = _plumes()
    P, r, dens = DW._puff_world(PL, t)
    r = r * 3.2
    sx, sy, z = scam.project(P)
    ok = (z > 20.0) & (sx > -200) & (sx < scam.W + 200) & (sy > -200) & (sy < scam.H + 200)
    if not ok.any():
        return
    P, r, dens, sx, sy, z = P[ok], r[ok], dens[ok], sx[ok], sy[ok], z[ok]
    rpx = np.maximum(r * scam.f / z, 0.6)
    a0 = np.minimum(dens * 2.4, 0.5) * np.clip(r * scam.f / z / 0.6, 0.25, 1.0)
    # lit by the blue sky; the higher puffs catch a little of the rose east
    hgt = np.clip((P[:, 1] - (WD.CLOUD_Y - 30.0)) / 450.0, 0.0, 1.0)
    sky = lin('#7C89B8') * 0.42
    rose = SPa[9:12] * 0.35
    col = sky[None, :] * (0.85 + 0.35 * hgt[:, None]) + rose[None, :] * (hgt[:, None] ** 2)
    order = np.argsort(-z)
    zb = np.maximum(0.004 * z, 3.0)
    DW._splat_puffs(fr.img, fr.zb, sx[order].astype(np.float64), sy[order].astype(np.float64),
                    z[order].astype(np.float64), rpx[order].astype(np.float64), a0[order].astype(np.float64),
                    col[order].astype(np.float64), (PL[ok][:, 8][order] * 0.37).astype(np.float64),
                    zb[order].astype(np.float64), (t * 0.25 + PL[ok][:, 9][order] * 7.0).astype(np.float64))


# ------------------------------------------------------------------ the people, the lantern, the fires ---
FIRES = [(SH_U + 14.2, SH_V + 4.4), (SH_U + 6.0, SH_V - 9.5)]


def sitter(sc, u, v, face_yaw, rgb, h=1.0, lean=0.35, tilt=0.0, fold_phase=0.0, peak=True):
    """A hooded figure sitting on its pack on the snow, knees drawn up under the cloak, seen from behind: sloping
    shoulders, a rounded back, the cloak pooling on the snow. tilt (rad) leans the head toward its left (+) or right
    (-): a head on a shoulder."""
    base = on_ground(u, v)
    a = math.radians(face_yaw)
    w = np.array([math.sin(a), 0.0, math.cos(a)])
    s = np.cross(w, UP)                                      # its left (cross(UP, w) is its right in this world)
    fl = w * math.sin(lean) + UP * math.cos(lean)
    pel = base + UP * 0.22 * h
    chest = pel + fl * 0.44 * h
    neck = chest + fl * 0.09 * h
    head = neck + (UP * math.cos(tilt) + s * math.sin(tilt)) * 0.11 * h + w * 0.05 * h
    sc.begin(rgb=rgb)
    # the pack under it, a corner showing below the hem
    sc.box(base + UP * 0.08 * h - w * 0.12 * h, (0.19 * h, 0.08 * h, 0.14 * h), yaw=a + 0.3, rnd=0.04 * h, mat=0,
           k=0.02)
    # knees drawn up in front, under the cloak
    for sd in (1.0, -1.0):
        kn = base + w * 0.34 * h + s * sd * 0.12 * h + UP * 0.40 * h
        sc.cone(pel + s * sd * 0.09 * h, kn, 0.085 * h, 0.07 * h, 0, 0.03)
        sc.cone(kn, base + w * 0.46 * h + s * sd * 0.13 * h + UP * 0.06, 0.065 * h, 0.05 * h, 0, 0.02)
    # the cloak: a short cape over sloping shoulders, then a bell over the back and knees, settling on the snow
    sc.cone(neck - UP * 0.02 * h, chest - UP * 0.04 * h, 0.12 * h, 0.225 * h, 0, 0.05 * h)
    sc.bell(chest - UP * 0.03 * h, base + UP * 0.02 * h + w * 0.12 * h, 0.23 * h, 0.33 * h, w, 0.035 * h, 10,
            fold_phase, 0, 0.07 * h)
    # the hood: a cowl over the head that falls in a drape to the shoulders (no neck: a pawn has a neck), its soft
    # point falling back
    sc.cone(head - w * 0.015 * h, head + UP * 0.015 * h, 0.115 * h, 0.118 * h, 0, 0.10 * h)
    sc.cone(head - UP * 0.02 * h - w * 0.02 * h, chest + UP * 0.04 * h - w * 0.03 * h, 0.11 * h, 0.19 * h, 0,
            0.08 * h)
    if peak:
        sc.cone(head - w * 0.05 * h + UP * 0.04 * h, head - w * 0.15 * h - UP * 0.01 * h, 0.07 * h, 0.03 * h, 0,
                0.05 * h)
    sc.end()
    sh = chest + UP * 0.01 * h
    return dict(head=head, chest=chest, left=sh + s * 0.19 * h, right=sh - s * 0.19 * h, w=w)


def arm_round(sc, frm, to, back, rgb, h=1.0):
    """A sleeved arm from one sitter's shoulder across the other's back to its far shoulder, a gloved hand there."""
    # back = toward the camera (behind them): the sleeve lies across the other's shoulder blades
    hand = to + back * 0.05 * h + UP * 0.06 * h
    mid = 0.5 * (frm + hand) + back * 0.20 * h + UP * 0.02 * h
    sc.begin(rgb=rgb)
    sc.cone(frm + back * 0.06 * h, mid, 0.075 * h, 0.068 * h, 0, 0.04 * h)
    sc.cone(mid, hand, 0.068 * h, 0.058 * h, 0, 0.04 * h)
    sc.cone(hand, hand, 0.05 * h, 0.05 * h, 0, 0.015 * h)
    sc.end()


def seated(sc, u, v, face_yaw, rgb, h=1.0, lean=0.30, hands_to=None, peak=True, fold_phase=0.0):
    """A hooded figure sitting on the snow, cloak drawn around the knees (feet forward, knees up)."""
    base = on_ground(u, v)
    a = math.radians(face_yaw)
    w = np.array([math.sin(a), 0.0, math.cos(a)])
    side = np.cross(UP, w)
    # sitting on a pack on the snow, hunched forward over drawn-up knees: from behind a rounded, sloping shape
    pel = base + UP * 0.26 * h - w * 0.08
    aL = base + w * 0.42 * h + side * 0.15 * h + UP * 0.07
    aR = base + w * 0.40 * h - side * 0.14 * h + UP * 0.07
    aL[1] = ground(aL)[0] + 0.07
    aR[1] = ground(aR)[0] + 0.07
    return cowl(sc, SP.traveller(sc, pel, w, aL, aR, rgb, h=h, lean=lean, hem=0.02, cloak=(0.23, 0.47), folds=10,
                                 fold_depth=0.035, fold_phase=fold_phase, peak=peak,
                                 reach=hands_to if hands_to is not None else base + w * 0.45 * h + UP * 0.35),
                rgb, h)


def cowl(sc, fig, rgb, h=1.0):
    """Fill a traveller's neck: its hood falls in a drape to the shoulders (a hood on a neck reads as a pawn)."""
    sc.begin(rgb=rgb)
    sc.cone(fig['head'] - UP * 0.02 * h, fig['chest'] + UP * 0.03 * h, 0.108 * h, 0.19 * h, 0, 0.0)
    sc.end()
    return fig


def standing(sc, u, v, face_yaw, rgb, h=1.0, lean=0.05, peak=True, fold_phase=0.0, hem=0.24):
    base = on_ground(u, v)
    a = math.radians(face_yaw)
    w = np.array([math.sin(a), 0.0, math.cos(a)])
    side = np.cross(UP, w)
    aL = base + side * 0.12 + UP * 0.07
    aR = base - side * 0.12 + w * 0.10 + UP * 0.07
    aL[1] = ground(aL)[0] + 0.07
    aR[1] = ground(aR)[0] + 0.07
    pel = base + UP * 0.93 * h
    return cowl(sc, SP.traveller(sc, pel, w, aL, aR, rgb, h=h, lean=lean, hem=hem, cloak=(0.21, 0.34), folds=9,
                                 fold_depth=0.03, fold_phase=fold_phase, peak=peak), rgb, h)


def build_scene(t):
    sc = SP.Scene()
    lights = []
    face = X.YAW_E - 6.0
    # the two bearers, unroped, sitting close near the edge with their backs to us: the left one's arm round the
    # other, whose head leans on its shoulder; the poles laid down beside them, the rope coiled, the lantern set
    # on the snow at their side
    A = sitter(sc, SH_U + 8.6, SH_V - 2.77, face - 2.0, (0.050, 0.040, 0.034), h=1.02, lean=0.30, fold_phase=0.7)
    B = sitter(sc, SH_U + 8.65, SH_V - 2.17, face + 3.0, (0.046, 0.040, 0.036), h=0.97, lean=0.42, tilt=0.30,
               peak=False, fold_phase=2.1)
    arm_round(sc, A['right'], B['right'] + B['w'] * 0.02, -A['w'], (0.050, 0.040, 0.034), h=1.0)
    lb = on_ground(SH_U + 8.25, SH_V - 1.50)
    heart = lb + UP * ((0.30 + 0.13) * X.LANT_K)
    hc = np.array([1.00, 0.72, 0.38])
    br = X.breath(t)
    SP.great_lantern(sc, heart, math.radians(X.YAW_E) + 0.3, hc, 0.55 * br, scale=X.LANT_K)
    lights.append([heart[0], heart[1], heart[2], hc[0], hc[1], hc[2], 2.3 * br, 0.35])
    sc.begin(rgb=(0.07, 0.05, 0.035))
    for o in (-0.22, 0.22):
        a_ = on_ground(SH_U + 7.0, SH_V - 3.80 + o, 0.03)
        b_ = on_ground(SH_U + 10.6, SH_V - 3.30 + o, 0.03)
        sc.cone(a_, b_, 0.022, 0.022, 2, 0.0)                        # the poles, laid down
    # the rope, coiled on the snow: a few loose loops
    c0 = on_ground(SH_U + 7.9, SH_V - 0.85, 0.03)
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
    # others of the line: sitting and standing in small groups along the shoulder, facing the east
    pal = [(0.06, 0.035, 0.028), (0.04, 0.045, 0.06), (0.06, 0.05, 0.03), (0.035, 0.045, 0.035), (0.05, 0.05, 0.05)]
    grp = [(13.4, 2.6, 's'), (14.1, 3.7, 's'), (12.6, 6.2, 'u'), (15.0, 7.4, 's'), (15.6, 8.5, 's'),
           (13.8, 11.0, 'u'), (16.3, 5.3, 'u'), (10.5, 13.0, 's')]
    for k, (du, dv, kind) in enumerate(grp):
        rgb = pal[k % len(pal)]
        if kind == 's':
            sitter(sc, SH_U + du, SH_V + dv, face + 10.0 * math.sin(k * 1.7), rgb, h=0.94 + 0.1 * (k % 3) / 2,
                   lean=0.30 + 0.08 * (k % 3), tilt=0.12 * math.sin(k * 2.9), peak=(k % 2 == 0), fold_phase=k * 0.9)
        else:
            standing(sc, SH_U + du, SH_V + dv, face + 15.0 * math.sin(k * 2.3), rgb, h=0.95 + 0.08 * (k % 2),
                     peak=(k % 3 != 1), fold_phase=k * 1.3, hem=0.45)
    # the watch-fires on the shoulder, fed; a keeper at each, beside it
    for k, (fu, fv) in enumerate(FIRES):
        p = on_ground(fu, fv)
        sc.begin(rgb=(0.07, 0.07, 0.075))
        rng = np.random.default_rng(80 + k)
        for m in range(12):
            a_ = 2 * math.pi * m / 12 + rng.uniform(-0.15, 0.15)
            q = p + np.array([math.cos(a_), 0.0, math.sin(a_)]) * (0.48 + rng.uniform(-0.04, 0.05))
            hs = rng.uniform(0.6, 1.0)
            sc.box(q + UP * 0.05 * hs, (0.09 * hs, 0.07 * hs, 0.08 * hs), yaw=a_, pitch=rng.uniform(-0.3, 0.3),
                   rnd=0.025, mat=5, k=0.0)
        sc.end()
        # the keeper kneels beside the fire in profile to the camera (never front-lit)
        seated(sc, fu - 0.9, fv + 0.35, X.YAW_E + 80.0, (0.03, 0.025, 0.02), h=0.98, lean=0.45, peak=True,
               hands_to=p + UP * 0.3, fold_phase=1.7 * k)
        fl = F.flicker(t, 60 + k)
        lights.append([p[0], p[1] + 0.8, p[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2], 7.0 * fl, 0.8])
    return sc, lights, heart, hc, br


# ------------------------------------------------------------------ frame ---
FINISH = dict(exposure=1.05, bloom_strength=0.07, bloom_threshold=1.0, streak_strength=0.0, vignette_amount=0.25)
_STARS = None


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
    PL = [[p[0], p[1], p[2], 1.1] for p in (on_ground(u, v) for u, v in FIRES)]
    WD.shade(C, D, Pp, CR, S, np.array(lights, np.float64).reshape(-1, 8), Lk, Q, amb, fogp, fr.img, fr.zb, fr.dist,
             np.array(PL, np.float64).reshape(-1, 4))
    SPa = sky_params(t)
    sky_pass(fr.img, fr.dist, C, SPa)
    if _STARS is None:
        _STARS = SK.make_stars(9000, 101, lum_scale=5.0)
    k = min(max((t - 80 / FPS) / ((NFR - 80) / FPS), 0.0), 1.0)
    mask = (fr.dist > 1e8).astype(np.float32)
    SK.splat_stars(fr.img, scam, _STARS, mask, t=t, gain=fr.ss * fr.ss * 0.35 * (1.0 - 0.8 * k),
                   scale=PI.src_scale(fr))
    draw_smoke(fr, t, SPa)
    SP.render(fr.img, fr.zb, C, *sc.arrays(), np.array(lights, np.float64), np.array([Lk[0], Lk[1], Lk[2], Lk[3], Lk[4],
              Lk[5], Q[0]]), amb * 1.2, fogp, scam.pos[1])
    for (fu, fv) in FIRES:
        p = on_ground(fu, fv)
        F2.flame(fr.img, fr.zb, scam, p + UP * 0.1, 1.2, 0.38, t, seed=int(fu), I=11.0, lean=0.1, zbias=0.5)
    X.draw_heart(fr.img, fr.zb, scam, heart, t, hc, 0.6 * br, PI.src_scale(fr))
    img, zb, di = PI.to_target(fr)
    return img


def _work(args):
    frames, scale, ss, out, threads = args
    import numba
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    for f in frames:
        t0 = time.time()
        img = PI.look.finish(render(f - CUT0, scale, ss), **FINISH)
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
    ap.add_argument('--threads', type=int, default=4)
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
        raise SystemExit(f'frames outside A19 (cut {CUT0}-{CUT0 + NFR - 1}): {bad[:5]}')
    if a.skip:
        frames = [f for f in frames if not os.path.exists(PI.look.frame_path(out, f))]
    if a.procs <= 1:
        _work((frames, a.scale, a.ss, out, a.threads))
    else:
        import multiprocessing as mp
        chunks = [frames[i::a.procs] for i in range(a.procs)]
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(c, a.scale, a.ss, out, 1) for c in chunks])


if __name__ == '__main__':
    main()
