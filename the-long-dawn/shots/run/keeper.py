"""B . THE KEEPER: the lifetime locked frame (R9, greybox) on her summit, in the Run world.

One locked frame near her summit, looking NNE over the cloud bay to the ring of ranges; the NE arete (the
path) runs off to the right and down to the cloud. World plates (a few lighting states) are rendered once in
the lens-shift SOURCE camera and cached; everything that lives in time is drawn over them per frame, in source
space (verticals vertical), then warped to the target camera:
  fire + firelight (a unit-light difference plate, so the snow and rock around the beacon take the fire),
  the counting cairn (one stone a year), her figure at every age, travellers and the thread of torches
  down the arete, far pinpricks answering on the ranges, village glow under the cloud sea, star trails,
  snow in the storm, and dissolves between the plates (time passes inside the frame, never by a cut).

Nothing in world.py is changed: the summit shelf is a `crag` row in this module's own CR table.

  python shots/run/keeper.py --plates              # render + cache the plates (quarter scale)
  python shots/run/keeper.py --frames 0,200,600    # test frames -> renders/run_b_tests/lifetime/
  python shots/run/keeper.py --all                 # the whole greybox animatic + mp4 + contact sheet
"""
import math
import os
import sys
import time

import cv2
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import fire2 as F2          # noqa: E402
import s1_peak as S1        # noqa: E402
from mt import sky as SK, fire as F, figure as FG, props as PR   # noqa: E402
from mt.noise import smoothstep   # noqa: E402

look = PI.look
CM = PI.CM
FPS = 24.0
ROOT = CM.ROOT
OUTDIR = os.path.join(ROOT, 'renders', 'run_b_tests')
CACHE = os.path.join(OUTDIR, 'cache')

# ------------------------------------------------------------------ the set ---
S_SUM = S1.summit()                                   # the point s1 aimed at (her beacon seen from the shepherd)
TOPXZ = np.array([S_SUM[0] - 8.0, S_SUM[2] + 1.0])    # the true top of her horn is 10 m west of it
TOP_Y = 301.5
ARETE_AZ = 40.0                   # the long arete: the way up and down, NE into the cloud


def _ar(r, az=ARETE_AZ):
    a = math.radians(az)
    return (TOPXZ[0] + r * math.sin(a), TOPXZ[1] + r * math.cos(a))


_A0, _A1, _A2 = _ar(4.0), _ar(1400.0), _ar(2700.0, ARETE_AZ + 3.0)
CR_B = np.array([WD.crag_row(TOPXZ[0], TOPXZ[1], TOP_Y, L=5.0, s_hi=1.45, s_lo=2.2, aniso=1.15,
                             ang=math.radians(-45.0), seed=5, k=4.0, detail=0.10, shelf=3.4, nf=5),
                 WD.ridge_row((_A0[0], TOP_Y - 0.6, _A0[1]), (_A1[0], 110.0, _A1[1]), wl=28.0, wr=28.0, seed=41,
                              k=6.0, detail=0.22, slope=1.3),
                 WD.ridge_row((_A1[0], 110.0, _A1[1]), (_A2[0], -640.0, _A2[1]), wl=40.0, wr=40.0, seed=43,
                              k=8.0, detail=0.25, slope=1.2)])

# the locked camera: SSW of the top, a little above the shelf, looking NNE over the bay
CAM_BEAR = 20.0                   # bearing camera -> top (deg)
CAM_DIST = 44.0
CAM_UP = 1.5                      # above the shelf: the cairn, her head and the flames stand against the sky
YAW = 31.0
PITCH = -2.6
HFOV = 44.0


def _dirxz(az):
    a = math.radians(az)
    return np.array([math.sin(a), 0.0, math.cos(a)])


TOP = np.array([TOPXZ[0], TOP_Y, TOPXZ[1]])
CAM_POS = TOP - CAM_DIST * _dirxz(CAM_BEAR) + np.array([0.0, CAM_UP, 0.0])
FWD = _dirxz(YAW)
RIGHT = np.array([FWD[2], 0.0, -FWD[0]])


def ground(x, z):
    return WD.ground(x, z, CR_B, 0.02)


def on_ground(p):
    return np.array([p[0], ground(p[0], p[2]), p[2]])


def shelf_pt(r, f):
    """A point on the shelf: r metres screen-right and f metres away from the camera, from the top."""
    return on_ground(TOP + RIGHT * r + FWD * f)


BEACON = shelf_pt(0.9, 0.6)          # the fire basket on its low dry-stone base
CAIRN = shelf_pt(-2.1, -1.3)         # the counting cairn: one stone at each grey dawn (diagonal to the beacon,
                                     # so neither hides the other from the lifetime or the hand-back camera)
KNEEL = shelf_pt(0.05, 0.35)         # where she kneels to strike / feed
STAND = shelf_pt(-0.55, 0.1)         # where she keeps the watch (between the fire and the cairn)
LIP = shelf_pt(3.6, 3.2)             # the NE lip: where the arete leaves the shelf (entries and exits)


def camera(W=480, H=201):
    return RC.RCam(CAM_POS, YAW, PITCH, 0.0, HFOV, W, H)


# ------------------------------------------------------------------ plates ---
# A plate = one lighting state of the locked frame, rendered once. Lk/amb/S/fogp/Q as world.shade takes them.

def _night(moon=True, key=0.55, amb_k=0.35, fog_a=5.0e-5, fog_b=1 / 1500.0, fogc='#2E3D66', fogk=0.95,
           zen='#070B1C', hor='#2A3866', glow=0.25, mdir=None, mist=2.2e-4, fwd=1.5):
    Lk, amb, S, fogp, Q = WD.night_light()
    if mdir is not None:
        md = np.asarray(mdir, np.float64)
        md = md / np.linalg.norm(md)
        Lk = np.r_[md, Lk[3:]]
    else:
        md = Lk[:3]
    Q[0] = key
    amb = CM.lin('#27335E') * amb_k
    S = SK.sky_params(zenith=zen, horizon=hor, moon_dir=md if moon else (0.0, -1.0, 0.2), moon_radius_deg=0.8,
                      halo_I=0.025 if moon else 0.0, halo_w=0.22, halo2_I=0.012 if moon else 0.0, halo2_w=0.7,
                      horizon_glow=glow, gain=1.0)
    fc = CM.lin(fogc) * fogk
    fogp = np.array([fog_a, fog_b, mist, 1 / 140.0, fwd, fc[0], fc[1], fc[2]])
    return Lk, amb, S, fogp, Q


def plate_light(name):
    if name == 'moon':        # s1's moon: the first night is the reveal's night
        return _night()
    if name == 'stars':       # moonless, clear: starlight only (the eye adapted), the far ranges faint
        return _night(moon=False, key=0.10, amb_k=0.22, mdir=(0.0, 1.0, 0.05), zen='#04060F', hor='#141C38',
                      glow=0.18, fogc='#1C2744', fogk=0.8)
    if name == 'storm':       # driving snow: no sky, the far world gone, visibility ~250 m
        return _night(moon=False, key=0.12, amb_k=0.34, mdir=(0.0, 1.0, 0.1), zen='#11151F', hor='#1E2433',
                      glow=0.0, fog_a=3.2e-3, fog_b=1 / 4000.0, fogc='#262D40', fogk=0.62, mist=0.0, fwd=0.0)
    if name == 'fog':         # inside the cloud: the summit alone, moonlit fog, visibility ~110 m
        return _night(moon=False, key=0.10, amb_k=0.30, mdir=(0.0, 1.0, 0.1), zen='#1A2236', hor='#232C44',
                      glow=0.0, fog_a=9.0e-3, fog_b=1 / 4000.0, fogc='#34426A', fogk=0.62, mist=0.0, fwd=0.0)
    if name == 'dawn':        # the grey dawn: no sun yet, cold light, the east (right) paling
        return _night(moon=True, key=0.30, amb_k=1.25, mdir=(0.85, 0.05, 0.52), zen='#232F4C', hor='#6E7894',
                      glow=0.55, fogc='#56627F', fogk=0.9)
    raise KeyError(name)


PLATES = ('moon', 'stars', 'storm', 'fog', 'dawn')
FIRE_REF = 40.0          # the unit fire light of the difference plates (world.shade LT intensity)


def _render_plate(name, scale, ss):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tc = camera(W, H)
    fr = PI.Frame(tc, ss)
    light = plate_light(name)
    PI.render_terrain(fr, 0, CR_B, light, np.zeros((0, 8)))
    base = fr.img.copy()
    fl = [BEACON[0], BEACON[1] + 1.6, BEACON[2], F.FIRE_LIGHT[0], F.FIRE_LIGHT[1], F.FIRE_LIGHT[2], FIRE_REF, 0.7]
    PI.render_terrain(fr, 0, CR_B, light, np.array(fl))
    fire = np.maximum(fr.img - base, 0.0)
    if name == 'dawn':       # the east pales: a cold band low on the right of the sky, rising to grey
        scam = fr.src
        v, u = np.mgrid[0:scam.H, 0:scam.W].astype(np.float64)
        sky = fr.dist > 1e8
        xr = (u + 0.5 - scam.cx) / scam.f
        yr = (scam.cyy - (v + 0.5)) / scam.f
        east = np.clip(0.5 + 0.9 * xr, 0, 1.4)
        band = np.exp(-np.maximum(yr, 0) / 0.05) * east
        warm = CM.lin('#B9A89C') * 0.22
        base = base + (sky * band)[..., None] * warm[None, None, :].astype(np.float32)
    return dict(img=base.astype(np.float32), fire=fire.astype(np.float32), zb=fr.zb.astype(np.float32),
                dist=fr.dist.astype(np.float32))


def plates(scale=0.25, ss=1.5, force=False):
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for n in PLATES:
        p = os.path.join(CACHE, f'plate_{n}_{scale:.3f}_{ss:.2f}.npz')
        if force or not os.path.exists(p):
            t0 = time.time()
            d = _render_plate(n, scale, ss)
            np.savez_compressed(p, **d)
            print(f'plate {n} {time.time() - t0:.1f}s', flush=True)
        z = np.load(p)
        out[n] = {k: z[k] for k in z.files}
    return out


# --------------------------------------------------------------- the world ---
_ST = {}


def _cached(key, fn):
    if key not in _ST:
        _ST[key] = fn()
    return _ST[key]


def path_points():
    """The route down: from the fire over the NE lip and down the arete's crest (lateral max search every 10 m)
    to the cloud sea. World points, top first."""
    def mk():
        pts = [BEACON.copy(), shelf_pt(2.0, 1.8), LIP]
        d = _dirxz(ARETE_AZ)
        n = np.array([d[2], 0.0, -d[0]])
        r = 12.0
        while r < 2700.0:
            az = ARETE_AZ + 3.0 * smoothstep(1400.0, 1600.0, r)
            d = _dirxz(az)
            n = np.array([d[2], 0.0, -d[0]])
            q = np.array([TOPXZ[0], 0.0, TOPXZ[1]]) + d * r
            best = None
            for off in np.linspace(-12, 12, 13):
                c = q + n * off
                h = ground(c[0], c[2])
                if best is None or h > best[0]:
                    best = (h, c)
            p = np.array([best[1][0], best[0], best[1][2]])
            if p[1] < WD.CLOUD_Y + 30:
                break
            pts.append(p)
            r += 10.0 if r < 200 else 25.0
        P = np.array(pts)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        return P, s
    return _cached('path', mk)


def path_at(sm):
    P, s = path_points()
    sm = np.clip(sm, 0, s[-1])
    out = np.stack([np.interp(sm, s, P[:, k]) for k in range(3)], -1)
    return out


def far_peaks(plate, scam):
    """Summits in frame, 1.5-45 km out, that the camera can see (depth test on the moon plate). Sorted nearest
    first; each gets the year it first answers."""
    def mk():
        rng = np.random.default_rng(7)
        cand = []
        for r in np.geomspace(1500.0, 45000.0, 70):
            for a in np.linspace(YAW - 26, YAW + 26, 44):
                d = _dirxz(a)
                x, z = CAM_POS[0] + d[0] * r, CAM_POS[2] + d[2] * r
                cand.append((x, z, r))
        peaks = []
        for x, z, r in cand:
            # climb to the local summit (coarse hill-climb at the local footprint)
            fp = r / 600.0
            step = max(r * 0.012, 40.0)
            bx, bz = x, z
            bh = WD.h_rock(bx, bz, fp, CR_B)
            for _it in range(12):
                imp = False
                for ddx, ddz in ((step, 0), (-step, 0), (0, step), (0, -step), (step, step), (-step, -step),
                                 (step, -step), (-step, step)):
                    h = WD.h_rock(bx + ddx, bz + ddz, fp, CR_B)
                    if h > bh:
                        bh, bx, bz, imp = h, bx + ddx, bz + ddz, True
                if not imp:
                    step *= 0.5
            if bh < WD.CLOUD_Y + 120:
                continue
            peaks.append((bx, bh, bz))
        P = np.array(peaks)
        # dedupe
        keep = []
        for p in P:
            if all(np.hypot(p[0] - q[0], p[2] - q[2]) > max(0.03 * np.hypot(p[0] - CAM_POS[0], p[2] - CAM_POS[2]), 120)
                   for q in keep):
                keep.append(p)
        P = np.array(keep)
        # visible from the camera: projected depth vs the plate depth
        dist = np.linalg.norm(P - CAM_POS, axis=1)
        curv = ((P[:, 0] - CAM_POS[0]) ** 2 + (P[:, 2] - CAM_POS[2]) ** 2) / (2 * WD.R_EARTH)
        Pc = P.copy()
        Pc[:, 1] = P[:, 1] - curv + 3.0
        sx, sy, z = scam.project(Pc)
        ok = []
        for i in range(len(P)):
            ix, iy = int(sx[i]), int(sy[i])
            if not (2 <= ix < scam.W - 2 and 2 <= iy < scam.H - 2) or z[i] <= 0:
                continue
            dd = plate['dist'][iy - 1:iy + 2, ix - 1:ix + 2].max()
            if dd > dist[i] * 0.97:
                ok.append(i)
        Pc = Pc[ok]
        dist = dist[ok]
        o = np.argsort(dist)
        Pc, dist = Pc[o], dist[o]
        n = len(Pc)
        # the year each first answers: the first at year 8 far off on the rim; then the ring fills in
        yr = np.zeros(n)
        order = list(rng.permutation(n))
        sx, sy, _ = scam.project(Pc)
        first = [i for i in range(n) if dist[i] > 12000 and 0.52 * scam.W < sx[i] < 0.75 * scam.W]
        if first:
            f0 = min(first, key=lambda i: sy[i])
            order.remove(f0)
            order.insert(0, f0)
        for k, i in enumerate(order):
            yr[i] = 8.0 + 52.0 * (k / max(n - 1, 1)) ** 0.8
        return Pc, dist, yr
    return _cached('peaks', mk)


def villages(plate, scam):
    """Warm glow under the cloud sea: centres on the cloud (world x, z) in frame, each with a first year."""
    def mk():
        rng = np.random.default_rng(11)
        v, u = np.mgrid[0:scam.H, 0:scam.W]
        d = plate['dist']
        # world points of the cloud-sea pixels
        ys = []
        rays = scam.params()
        pts = []
        for _k in range(4000):
            i = rng.integers(0, scam.W)
            j = rng.integers(int(scam.H * 0.35), scam.H)
            dd = d[j, i]
            if dd > 1e8 or dd < 1500 or dd > 30000:
                continue
            xo = i + 0.5 - scam.cx
            dxh = scam.fwd[0] * scam.f + scam.right[0] * xo
            dzh = scam.fwd[2] * scam.f + scam.right[2] * xo
            hl = math.hypot(dxh, dzh)
            vy = scam.cyy - (j + 0.5)
            nn = math.sqrt(hl * hl + vy * vy)
            x = scam.pos[0] + dxh / nn * dd
            y = scam.pos[1] + vy / nn * dd
            z = scam.pos[2] + dzh / nn * dd
            curv = ((x - scam.pos[0]) ** 2 + (z - scam.pos[2]) ** 2) / (2 * WD.R_EARTH)
            if y + curv < WD.CLOUD_Y + 170 and WD.h_rock(x, z, 20.0, CR_B) < WD.CLOUD_Y - 40:
                pts.append((x, z, dd))
        pts = np.array(pts)
        keep = []
        for p in pts[rng.permutation(len(pts))]:
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 900 for q in keep):
                keep.append(p)
            if len(keep) >= 22:
                break
        V = np.array(keep)
        o = np.argsort(V[:, 2])
        V = V[o]
        yr = np.linspace(17, 58, len(V)) + rng.normal(size=len(V)) * 2.0
        rad = rng.uniform(260, 520, len(V))
        return V, yr, rad
    return _cached('vill', mk)


@njit(cache=True, fastmath=True)
def _village_field(W, H, cx, cyy, f, px, py, pz, fx, fz, rx, rz, dist, VX, VZ, VR, VI, out):
    for j in range(H):
        for i in range(W):
            dd = dist[j, i]
            if dd > 1e8 or dd < 800.0:
                continue
            xo = i + 0.5 - cx
            dxh = fx * f + rx * xo
            dzh = fz * f + rz * xo
            hl = math.sqrt(dxh * dxh + dzh * dzh)
            vy = cyy - (j + 0.5)
            nn = math.sqrt(hl * hl + vy * vy)
            x = px + dxh / nn * dd
            z = pz + dzh / nn * dd
            y = py + vy / nn * dd + ((x - px) ** 2 + (z - pz) ** 2) / (2 * 6.371e6)
            low = 1.0 - min(max((y - (-650.0 + 60.0)) / 150.0, 0.0), 1.0)
            if low <= 0.0:
                continue
            s = 0.0
            for k in range(VX.shape[0]):
                if VI[k] <= 0.0:
                    continue
                ex = (x - VX[k]) / VR[k]
                ez = (z - VZ[k]) / VR[k]
                q = ex * ex + ez * ez
                if q < 9.0:
                    s += VI[k] * math.exp(-q)
            out[j, i] = s * low


# ------------------------------------------------------------------- stars ---
POLE = np.array([0.0, math.sin(math.radians(44.0)), math.cos(math.radians(44.0))])


def _rotmat(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def star_cat():
    def mk():
        st = SK.make_stars(14000, 101, lum_scale=7.0)
        o = np.argsort(-st['lum'])[:2600]
        return dict(dir=st['dir'][o], lum=st['lum'][o], col=st['col'][o])
    return _cached('stars', mk)


@njit(cache=True, fastmath=True)
def _trail(img, mask, xs, ys, lum, col, sig):
    """xs, ys: (n, K) screen samples along each trail, oldest first; brightness rises toward the head."""
    H, W = img.shape[0], img.shape[1]
    n, K = xs.shape
    for s in range(n):
        for k in range(K - 1):
            x0, y0, x1, y1 = xs[s, k], ys[s, k], xs[s, k + 1], ys[s, k + 1]
            L = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            m = int(L / 0.5) + 1
            w = lum[s] * (0.35 + 0.65 * ((k + 1.0) / (K - 1.0))) / m
            for q in range(m):
                u = (q + 0.5) / m
                x = x0 + (x1 - x0) * u
                y = y0 + (y1 - y0) * u
                ix = int(x)
                iy = int(y)
                for jj in range(iy - 1, iy + 2):
                    if jj < 0 or jj >= H:
                        continue
                    for ii in range(ix - 1, ix + 2):
                        if ii < 0 or ii >= W:
                            continue
                        dx = ii + 0.5 - x
                        dy = jj + 0.5 - y
                        g = math.exp(-(dx * dx + dy * dy) / (2 * sig * sig)) * w * mask[jj, ii]
                        img[jj, ii, 0] += g * col[s, 0]
                        img[jj, ii, 1] += g * col[s, 1]
                        img[jj, ii, 2] += g * col[s, 2]


def draw_stars(img, scam, mask, ang0, ang1, gain, K=10):
    """Stars wheeling about the pole from sky angle ang0 to ang1 (radians): a short trail behind each."""
    st = star_cat()
    angs = np.linspace(ang0, ang1, K)
    xs = np.zeros((len(st['lum']), K))
    ys = np.zeros_like(xs)
    vis = np.ones(len(st['lum']), bool)
    for k, a in enumerate(angs):
        R = _rotmat(POLE, a)
        d = st['dir'] @ R.T
        sx, sy, z = scam.project(scam.pos[None, :] + d * 1e7)
        xs[:, k], ys[:, k] = sx, sy
        vis &= (z > 0) & (d[:, 1] > -0.02)
    ok = vis & (xs.max(1) > -5) & (xs.min(1) < scam.W + 5) & (ys.max(1) > -5) & (ys.min(1) < scam.H + 5)
    lum = 0.02 * (st['lum'][ok] / 0.02) ** 0.55 * gain
    _trail(img, mask, xs[ok], ys[ok], lum.astype(np.float64), st['col'][ok], 0.55)


# ----------------------------------------------------------------- figures ---
M = FG.MATS.copy()
M = np.vstack([M, [0.30, 0.012, 0.010, 1.2, 0.0, 8.0, 0.10, 0, 0, 0, 0],      # 13 the red scarf
               [0.050, 0.046, 0.044, 1.2, 0.0, 8.0, 0.10, 0, 0, 0, 0],        # 14 grey wool
               [0.20, 0.20, 0.21, 1.4, 0.0, 8.0, 0.05, 0, 0, 0, 0],           # 15 white hair
               [0.16, 0.155, 0.15, 1.1, 0.2, 10.0, 0.05, 0, 0, 0, 0]])        # 16 weathered granite


def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def person(pose='stand', age=0.0, scarf=True, stick=False, torch=False, child=False, walk=0.0, face=0.0,
           wind=1.0, hold=None):
    """A hooded, cloaked figure from behind / three-quarter (2-D SDF puppet, metres, feet at origin).
    pose: stand | kneel | bend | sit | walk | shield | look ; age 0..1 (0 = young woman, 1 = very old: stoop,
    shorter step, a stick). face: -1..1 turns the hood (0 = away from camera). Returns (Drawing, pts)."""
    d = FG.Drawing()
    d.new_group()
    k = 0.035
    sc = 0.62 if child else 1.0
    stoop = 0.34 * age ** 1.3
    if pose == 'kneel':
        hip = np.array([0.02, 0.50])
        lean = 0.35 + stoop
    elif pose == 'sit':
        hip = np.array([0.0, 0.40])
        lean = 0.12 + stoop
    elif pose == 'bend':
        hip = np.array([0.0, 0.80])
        lean = 0.95
    else:
        hip = np.array([0.0, 0.92 - 0.06 * age])
        lean = 0.06 + stoop + (0.10 if pose == 'walk' else 0.0)
    hip = hip * sc
    up = _rot(np.array([0.0, 1.0]), -lean)
    side = _rot(np.array([1.0, 0.0]), -lean)
    chest = hip + up * 0.42 * sc
    neck = hip + up * 0.60 * sc
    # legs
    if pose == 'kneel':
        d.capsule(hip, (0.26, 0.10), 0.09, 0.07, k=k)
        d.capsule((0.26, 0.10), (-0.10, 0.06), 0.07, 0.055, k=k)
        d.capsule(hip + np.array([-0.05, 0]), (-0.18, 0.05), 0.09, 0.06, k=k)
    elif pose == 'sit':
        d.capsule(hip, (0.34, 0.36) * np.array([sc, sc]), 0.09 * sc, 0.075 * sc, k=k)
        d.capsule((0.34 * sc, 0.36 * sc), (0.38 * sc, 0.04), 0.075 * sc, 0.06 * sc, k=k)
        d.ellipse((-0.05, 0.18 * sc), 0.26 * sc, 0.2 * sc, k=0.05)          # the rock she sits on (dark)
    else:
        ph = walk * 2 * math.pi
        for sgn in (-1, 1):
            sw = 0.16 * math.sin(ph + (0 if sgn > 0 else math.pi)) * (1.0 if pose == 'walk' else 0.0) * (1 - 0.5 * age)
            a = hip + np.array([0.05 * sgn * sc, 0.0])
            ft = np.array([0.07 * sgn * sc + sw * sc, 0.03])
            kn = 0.5 * (a + ft) + np.array([0.03 * sc, 0.0])
            d.capsule(a, kn, 0.085 * sc, 0.068 * sc, k=k)
            d.capsule(kn, ft, 0.068 * sc, 0.055 * sc, k=k)
    # cloak: long, flared, the hem streaming a little downwind (+x)
    hem = hip + np.array([0.0, -0.55 * sc if pose in ('stand', 'walk', 'look', 'shield') else -0.25 * sc])
    hem_w = 0.30 * sc + (0.10 if pose == 'shield' else 0.0)
    d.trap(hem + np.array([0.04 * wind, 0.0]), chest + up * 0.05 * sc, hem_w, 0.19 * sc, rnd=0.03, k=0.06,
           mat=14 if age > 0.55 else 0, fuzz=0.01, ff=14.0)
    d.ellipse(chest + up * 0.05 * sc, 0.21 * sc, 0.12 * sc, ang=-lean, k=0.06, mat=14 if age > 0.55 else 0)
    # arms
    shL = chest + up * 0.05 * sc - side * 0.15 * sc
    shR = chest + up * 0.05 * sc + side * 0.15 * sc
    hands = {}
    if pose == 'shield':          # arms and cloak spread around the fire (to screen-right)
        for sh, sg in ((shL, -1), (shR, 1)):
            e = sh + np.array([0.28 * sg + 0.1, -0.05]) * sc
            w = e + np.array([0.22 * sg + 0.12, -0.12]) * sc
            d.capsule(sh, e, 0.07 * sc, 0.06 * sc, k=0.04)
            d.capsule(e, w, 0.06 * sc, 0.05 * sc, k=0.04)
        d.tri(shL, shR + np.array([0.55, -0.35]) * sc, hem + np.array([0.45, 0.1]) * sc, rnd=0.02, k=0.05)
    elif pose == 'kneel' or pose == 'bend':
        e = shR + _rot(np.array([0.22, -0.18]), 0) * sc
        w = e + np.array([0.20, -0.16]) * sc
        d.capsule(shR, e, 0.07 * sc, 0.06 * sc, k=0.04)
        d.capsule(e, w, 0.06 * sc, 0.05 * sc, k=0.04)
        hands['R'] = w
        e2 = shL + np.array([0.10, -0.25]) * sc
        w2 = e2 + np.array([0.18, -0.10]) * sc
        d.capsule(shL, e2, 0.07 * sc, 0.06 * sc, k=0.04)
        d.capsule(e2, w2, 0.06 * sc, 0.05 * sc, k=0.04)
        hands['L'] = w2
    else:
        for sh, sg in ((shL, -1), (shR, 1)):
            e = sh + np.array([0.04 * sg, -0.28]) * sc
            w = e + np.array([0.03 * sg + (0.10 if (hold is not None and sg < 0) else 0.0),
                              -0.24 + (0.08 if (torch and sg > 0) else 0.0)]) * sc
            d.capsule(sh, e, 0.07 * sc, 0.06 * sc, k=0.04)
            d.capsule(e, w, 0.06 * sc, 0.05 * sc, k=0.04)
            hands['L' if sg < 0 else 'R'] = w
    if stick:
        hR = hands.get('R', shR + np.array([0.1, -0.5]))
        tip = np.array([hR[0] + 0.12, 0.0])
        d.capsule(hR + np.array([0.0, 0.12]), tip, 0.018, 0.016, mat=4)
    if torch:
        hR = hands.get('R', shR + np.array([0.1, -0.5]))
        th = hR + np.array([0.10, 0.45]) * sc
        d.capsule(hR - np.array([0.02, 0.06]), th, 0.02, 0.022, mat=4)
        hands['torch'] = th
    # head: a hood, turned by `face`; the scarf wound at the neck with a tail streaming downwind
    hc = neck + up * 0.13 * sc + side * 0.02 * face
    d.ellipse(hc, 0.115 * sc, 0.135 * sc, ang=-lean, k=0.03, mat=14 if age > 0.55 else 0)
    if age > 0.7:
        d.ellipse(hc + side * 0.10 * face + up * 0.02, 0.05, 0.07, mat=15)      # white hair at the hood's edge
    if scarf:
        d.new_group()
        d.ellipse(neck + up * 0.02 * sc, 0.13 * sc, 0.06 * sc, ang=-lean, k=0.02, mat=13)
        tail = [neck + up * 0.02 * sc + np.array([0.08, 0.0]) * sc]
        for q in range(1, 5):
            tail.append(tail[-1] + np.array([0.09 * wind, -0.035 + 0.015 * math.sin(q * 1.3)]) * sc)
        d.chain(np.array(tail), 0.045 * sc, 0.025 * sc, k=0.02, mat=13)
    return d, dict(hands=hands, head=hc)


def stone_cairn(n, seed=3):
    """The counting cairn after n stones: a broad heap of field stones (2-D puppet) that grows self-similarly, a
    small cone of stones in the first years and nearly twice her height at sixty. Never a column: the base is as
    wide as the heap is tall, and the outline is made of the stones themselves (a smooth dark silhouette read as a
    thumb to a cold viewer). Stone i sits in the smallest heap containing i stones (rho = y/H + |x|/R sorted)."""
    def mk():
        rng = np.random.default_rng(seed)
        R0, H0, N = 1.55, 2.95, 64
        stones = []
        for i in range(1, N + 1):
            rho = (i / N) ** 0.75                        # steady growth: every dawn the heap is visibly bigger
            best = None
            for _t in range(12):                         # on the heap's surface, away from the stones already there
                v = rng.random() ** 1.3 * 0.92
                side = -1.0 if rng.random() < 0.5 else 1.0
                x = side * R0 * rho * (1.0 - v) * (0.92 + 0.08 * rng.random())
                y = H0 * rho * v
                dmin = min([((x - q[0]) ** 2 + ((y - q[1]) * 1.4) ** 2) for q in stones] or [9.0])
                if best is None or dmin > best[0]:
                    best = (dmin, x, y)
            _, x, y = best
            sw = 0.40 + 0.16 * rng.random()
            sh = 0.24 + 0.10 * rng.random()
            stones.append((x, max(y - 0.12, 0.0), sw, sh, (rng.random() - 0.5) * 0.5, rho))
        return stones
    stones = _cached(('cairn5', seed), mk)
    d = FG.Drawing()
    m = min(n, len(stones))
    if m > 3:                                  # dark chinks behind the stones, well inside the outline
        rho = stones[m - 1][5]
        d.new_group()
        d.tri((-1.55 * rho * 0.80, 0.02), (1.55 * rho * 0.80, 0.02), (0.0, 2.95 * rho * 0.80), rnd=0.04, mat=7)
    for i in range(m):
        cx, y, sw, h, tilt, rho = stones[i]
        d.new_group()
        p0 = (cx - math.sin(tilt) * h * 0.5, y + 0.01)
        p1 = (cx + math.sin(tilt) * h * 0.5, y + h - 0.01)
        d.trap(p0, p1, sw * 0.46, sw * 0.30, rnd=0.10, mat=16, fuzz=0.02, ff=12.0)
    return d


# ---------------------------------------------------------------- timeline ---
# One night = one bar of the chaconne (80 frames at 72 BPM): beat 1 she comes, three strikes, the catch; beat 2
# the fire up, the sky wheeling; beat 3 the watch (and whatever answers); beat 4 the grey dawn, the stone, gone.
# Cutaways (the far peak) sit between. ~10 nights, 3 lighting plates + the grey dawn.
NIGHTS = [
    # (kind, year, frames, plate, extras): the years quicken as she ages (a bar a night, then less), then the
    # sixtieth night slows to real time
    ('night', 0, 144, 'moon', {}),
    ('night', 1, 80, 'storm', {}),
    ('night', 2, 80, 'stars', {}),
    ('night', 3, 80, 'fog', {}),
    ('cut', 'M3', 56, None, {}),
    ('night', 8, 64, 'stars', {}),
    ('night', 14, 64, 'moon', {}),
    ('cut', 'M2', 40, None, {}),
    ('night', 20, 48, 'stars', {}),
    ('night', 26, 44, 'stars', {}),
    ('night', 32, 40, 'storm', {}),
    ('night', 38, 36, 'stars', {}),
    ('night', 44, 34, 'moon', {}),
    ('night', 50, 32, 'stars', {}),
    ('night', 55, 32, 'stars', {}),
    ('night', 60, 206, 'moon', {}),
]


def schedule():
    out = []
    f0 = 0
    for kind, yr, n, pl, ex in NIGHTS:
        out.append((f0, f0 + n, kind, yr, pl, ex))
        f0 += n
    return out, f0


def segment(frame):
    sch, total = schedule()
    for i, (a, b, kind, yr, pl, ex) in enumerate(sch):
        if a <= frame < b:
            return i, a, b, kind, yr, pl, ex
    a, b, kind, yr, pl, ex = sch[-1]
    return len(sch) - 1, a, b, kind, yr, pl, ex


def prev_night(i):
    sch, _ = schedule()
    for j in range(i - 1, -1, -1):
        if sch[j][2] == 'night':
            return sch[j]
    return None


def cairn_count(year, u):
    """Stones on the cairn during the night of `year` at night-phase u (the new one lands at u=0.9)."""
    return int(year) + (1 if u >= 0.905 else 0)


def env_night(u, year, plate, first=False):
    """Everything that is a function of the night phase u in [0, 1): fire size/light, her pose, weights."""
    e = {}
    # beats (quarters of the bar)
    strikes = [0.21, 0.245, 0.28]
    catch = 0.30
    if first:                      # the first night is already burning (B5 precedes)
        strikes = []
        catch = -1.0
    e['strikes'] = [s for s in strikes if abs(u - s) < 0.012]
    if u < catch:
        size = 0.0
    else:
        k = (u - catch) / 0.06
        size = min(1.0, 1 - math.exp(-k * 2.5)) * 1.0 + 0.3 * math.exp(-((k - 0.8) / 0.5) ** 2)
    # the fire sinks through the watch, is fed once (a pulse), then banked at dawn
    sink = 1.0 - 0.35 * smoothstep(0.35, 0.60, u)
    fed = 0.40 * math.exp(-((u - 0.63) / 0.03) ** 2)
    bank = 1.0 - 0.72 * smoothstep(0.80, 0.90, u)
    size = size * (sink + fed) * bank
    if plate == 'storm':           # the flame nearly dies in the gusts, and she shields it
        size *= 1.0 - 0.75 * math.exp(-((u - 0.42) / 0.05) ** 2)
    e['fire'] = max(size, 0.0)
    e['ember'] = smoothstep(0.84, 0.92, u) * (1 - smoothstep(0.985, 1.0, u))
    # dawn (the grey dawn plate) and the dissolve in from the previous dawn
    e['dawn'] = smoothstep(0.72, 0.92, u)
    e['in'] = smoothstep(0.0, 0.22, u) if not first else 1.0
    # sky wheel angle (radians of the pole rotation), dusk -> dawn
    e['sky'] = math.radians(-15.0 + 30.0 * u)
    # her presence and pose
    age = min(max((year - 5) / 55.0, 0.0), 1.0)
    e['age'] = age
    arrive = 0.16 + 0.04 * age
    leave = 0.97
    if first:
        arrive = -1.0
    if u < arrive:
        e['her'] = ('walk', 0.0, 0.0)
    elif u < catch + 0.03:
        e['her'] = ('kneel', 1.0, 0.0)
    elif plate == 'storm' and 0.36 < u < 0.52:
        e['her'] = ('shield', 1.0, 0.0)
    elif 0.58 < u < 0.67:
        e['her'] = ('kneel', 1.0, 0.0) if age < 0.8 else ('bend', 1.0, 0.0)
    elif 0.86 < u < 0.93:
        e['her'] = ('bend', 1.0, 1.0)          # at the cairn: the stone
    elif u < leave:
        e['her'] = ('sit', 1.0, 0.0) if (age > 0.62 and 0.3 < u < 0.80) else ('look', 1.0, 0.0)
    else:
        e['her'] = ('walk', 1.0 - smoothstep(0.97, 1.0, u), 2.0)
    e['arrive'] = arrive
    return e


# ------------------------------------------------------------------ render ---
_PL = {}


def get_plates(scale, ss):
    key = (scale, ss)
    if key not in _PL:
        _PL[key] = plates(scale, ss)
    return _PL[key]


def _fire_light_col():
    return np.asarray(F.FIRE_LIGHT, np.float64)


def _fig_lights(fire_I, pl):
    Lk, amb, S, fogp, Q = plate_light(pl)
    L = [dict(dir=Lk[:3], col=Lk[3:6], I=Q[0] * 0.8)]
    if fire_I > 0:
        L.append(dict(pos=BEACON + np.array([0, 1.3, 0]), col=F.FIRE_LIGHT, I=fire_I, r0=0.5))
    return L, amb * 1.3


def draw_person(img, zb, scam, feet, fig, lights, amb, t, alpha=1.0, flip=False):
    if alpha <= 0.01:
        return
    d, pts = fig
    if alpha >= 0.99:
        FG.render(img, zb, scam, d, feet, lights, amb=amb, mats=M, t=t, write_depth=False, zbias=0.3, flipx=flip)
        return
    tmp = img.copy()
    FG.render(tmp, zb, scam, d, feet, lights, amb=amb, mats=M, t=t, write_depth=False, zbias=0.3, flipx=flip)
    img[...] = img * (1 - alpha) + tmp * alpha


def render_night(frame, scale, ss, seg):
    i, a, b, kind, year, pl, ex = seg
    u = (frame - a) / float(b - a)
    first = year == 0
    PLT = get_plates(scale, ss)
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tc = camera(W, H)
    fr = PI.Frame(tc, ss)
    scam = fr.src
    e = env_night(u, year, pl, first)
    # --- the plate mix: previous dawn -> this night -> grey dawn
    pn = prev_night(i)
    w_in = e['in']
    w_dawn = e['dawn']
    P = PLT[pl]
    DAWN_K = 0.60                                   # the grey dawn held under its plate level (no slide-advance pump)
    img = P['img'] * (1 - w_dawn) + PLT['dawn']['img'] * DAWN_K * w_dawn
    fire_diff = P['fire'] * (1 - w_dawn) + PLT['dawn']['fire'] * w_dawn
    if w_in < 1.0:
        img = img * w_in + PLT['dawn']['img'] * DAWN_K * (1 - w_in)
    img = img.astype(np.float32).copy()
    zb = P['zb'].copy()
    dist = P['dist']
    t = frame / FPS
    # --- stars (clear plates only)
    clarity = {'moon': 1.0, 'stars': 1.6, 'storm': 0.0, 'fog': 0.0, 'dawn': 0.2}[pl]
    clarity *= (1 - smoothstep(0.70, 0.86, u)) * w_in
    if clarity > 0:
        mask = (dist > 1e8).astype(np.float32)
        draw_stars(img, scam, mask, e['sky'] - math.radians(0.25), e['sky'], clarity * ss * ss * 1.1, K=2)
    # --- villages under the cloud
    V, vyr, vrad = villages(PLT['moon'], scam)
    lit = np.clip((year - vyr) / 8.0, 0, 1) ** 1.2 * (0.5 + 0.5 * np.clip((year - vyr) / 30.0, 0, 1))
    if first:
        lit[:] = 0
    vis = {'moon': 1.0, 'stars': 1.0, 'storm': 0.0, 'fog': 0.0, 'dawn': 0.4}[pl]
    if lit.max() > 0 and vis > 0:
        fld = np.zeros(dist.shape, np.float32)
        C = scam.params()
        _village_field(scam.W, scam.H, C[8], C[9], C[7], scam.pos[0], scam.pos[1], scam.pos[2], C[3], C[4], C[5],
                       C[6], dist, V[:, 0], V[:, 1], vrad, lit * (1.0 - 0.5 * w_dawn), fld)
        vc = np.array([1.0, 0.55, 0.22], np.float32) * 0.12 * vis
        img += fld[..., None] * vc[None, None, :]
    # --- far pinpricks answering
    Pk, pdist, pyr = far_peaks(PLT['moon'], scam)
    ans_vis = {'moon': 1.0, 'stars': 1.0, 'storm': 0.0, 'fog': 0.0, 'dawn': 0.6}[pl]
    if ans_vis > 0 and not first:
        sx, sy, z = scam.project(Pk)
        for k in range(len(Pk)):
            if pyr[k] > year + 0.01:
                continue
            # each night they answer after hers has caught, the nearer first
            delay = 0.34 + 0.22 * (pdist[k] / pdist.max())
            on = smoothstep(delay, delay + 0.04, u) * (1 - smoothstep(0.86, 0.95, u))
            if on <= 0:
                continue
            fl = F.flicker(t * 0.7, k)
            en = 0.9 * on * ans_vis * fl * (0.45 + 0.55 * min(3000.0 / pdist[k], 1.0)) * ss * ss
            if pyr[k] < 8.5:
                en *= 5.0
                F2.halo(img, zb, sx[k], sy[k], 3.5 * ss, 0.02 * on * ans_vis, z=z[k], zbias=z[k] * 0.02)
            F2.glow(img, zb, sx[k], sy[k], 0.7 * ss, en, z=z[k], zbias=z[k] * 0.02)
            F2.halo(img, zb, sx[k], sy[k], 2.2 * ss, 0.004 * on * ans_vis, z=z[k], zbias=z[k] * 0.02)
    # --- the travellers and the thread of torches down the arete
    draw_travellers(img, zb, scam, u, year, pl, t, ss)
    # --- the counting cairn
    n_st = cairn_count(year, u) if not first else (1 if u >= 0.905 else 0)
    fire_I = 0.0
    if e['fire'] > 0:
        fire_I = 2.2 * e['fire'] * F.flicker(t, 3)
    fire_I += 0.35 * e['ember']
    lights, amb = _fig_lights(fire_I, pl)
    if n_st > 0:
        FG.render(img, zb, scam, stone_cairn(n_st), CAIRN, lights, amb=amb, mats=M, t=t, write_depth=True, zbias=0.3)
    # --- the beacon: dry-stone base + basket, the fire, its light on the snow
    back, front, fb, rb = _cached('beacon', lambda: PR.cairn(seed=9, height=0.85, base_w=1.2, top_w=0.95,
                                                            basket=True, basket_h=0.5, basket_w=1.1))
    FG.render(img, zb, scam, back, BEACON, lights, amb=amb, t=t, emissive_gain=min(e['fire'] + e['ember'], 1.2),
              write_depth=True, zbias=0.3)
    img += fire_diff * (fire_I * 1.5 / FIRE_REF)
    base = BEACON + np.array([0.0, fb, 0.0])
    if e['fire'] > 0.02:
        F2.flame(img, zb, scam, base, 1.25 * e['fire'] + 0.15, 0.42, t, seed=4, I=13.0 * min(e['fire'] * 1.3, 1.4),
                 lean=0.25, zbias=0.5, tongues=5, warp=1.2)
        sx, sy, z = scam.project(base + np.array([0, 0.6, 0]))
        fogk = {'moon': 1.0, 'stars': 1.0, 'storm': 2.0, 'fog': 3.0, 'dawn': 0.6}[pl]
        F2.halo(img, zb, sx, sy, (5.0 + 4.0 * fogk) * scam.f / z, 0.006 * fogk * e['fire'], z=z, zbias=3.0)
    if e['ember'] > 0.01:
        sx, sy, z = scam.project(base + np.array([0, 0.05, 0]))
        F2.glow(img, zb, sx, sy, 1.2 * ss, 3.0 * e['ember'] * ss * ss, z=z, zbias=0.5, col=np.array([1.0, 0.35, 0.08]))
    for s in e['strikes']:
        sx, sy, z = scam.project(KNEEL + np.array([0.35, 0.55, 0.0]))
        F2.glow(img, zb, sx, sy, 1.0 * ss, 60.0 * ss * ss, z=z, zbias=1.0, col=np.array([1.0, 0.85, 0.6]))
    FG.render(img, zb, scam, front, BEACON, lights, amb=amb, t=t, write_depth=False, zbias=0.3)
    # --- her: drawn at three nearby moments of the night and averaged (time-lapse ghosting, no pose pops)
    bgk = img.copy()
    acc = None
    for du, wgt in ((0.0, 0.55), (0.03, 0.27), (0.06, 0.18)):
        tmp = bgk.copy()
        draw_her(tmp, zb, scam, max(u - du, 0.0), year, pl, first, lights, amb, t)
        acc = tmp * wgt if acc is None else acc + tmp * wgt
    img = acc.astype(np.float32)
    # the child, the last year: beside her, holding her hand
    if year >= 60 and u > 0.12:
        fig = person('look' if not (0.3 < u < 0.58) else 'stand', 0.0, child=True, scarf=False, face=0.3)
        draw_person(img, zb, scam, STAND + RIGHT * 0.62 - FWD * 0.1, fig, lights, amb, t)
    # --- weather
    if pl == 'storm':
        draw_snow(img, scam, t, w=(1 - w_dawn) * w_in)
    fr.img, fr.zb, fr.dist = img, zb, dist
    out, zt, di = PI.to_target(fr)
    return out


def draw_her(img, zb, scam, u, year, pl, first, lights, amb, t):
    e = env_night(u, year, pl, first)
    pose, alpha, where = e['her']
    age = e['age']
    stick = age > 0.62
    if first or u >= e['arrive'] or pose != 'walk':
        if pose == 'walk' and where == 2.0:          # leaving down the arete, fading into the dark
            p = LIP + (STAND - LIP) * alpha
            fig = person('walk', age, stick=stick, walk=u * 9.0, face=0.6)
            draw_person(img, zb, scam, on_ground(p), fig, lights, amb, t, alpha=alpha)
        elif pose == 'walk':
            pass
        else:
            spot = {'kneel': KNEEL, 'shield': KNEEL + RIGHT * 0.15, 'bend': CAIRN + RIGHT * 0.85,
                    'look': STAND, 'sit': STAND + RIGHT * 0.3}[pose]
            if pose == 'bend' and where == 0.0:
                spot = KNEEL
            face = {'kneel': 1.0, 'shield': 1.0, 'bend': -1.0 if where == 1.0 else 1.0, 'look': 0.25, 'sit': 0.3}[pose]
            fig = person(pose, age, stick=stick and pose in ('look', 'walk'), face=face,
                         hold=('child' if year >= 60 else None))
            flip = pose == 'bend' and where == 1.0
            draw_person(img, zb, scam, spot, fig, lights, amb, t, flip=flip)
    # the return: each dusk she walks up the arete and over the lip to her fire
    if not first and u < e['arrive'] + 0.02:
        v = smoothstep(0.0, e['arrive'], u)
        P_, s_ = path_points()
        sd = 45.0 * (1.0 - v)
        p = path_at(sd) if sd > 8.0 else LIP + (STAND - LIP) * smoothstep(8.0, 0.0, sd)
        fig = person('walk', age, stick=stick, walk=u * 14.0, face=-0.6)
        draw_person(img, zb, scam, on_ground(p), fig, lights, amb, t, alpha=smoothstep(0.0, 0.25, v))


def draw_travellers(img, zb, scam, u, year, pl, t, ss):
    """From year 11 people climb to her fire with unlit torches, light them and carry them down: small figures
    at the shelf, then lights descending the arete (a longer thread each year)."""
    if year < 11:
        return
    vis = {'moon': 1.0, 'stars': 1.0, 'storm': 0.35, 'fog': 0.25, 'dawn': 0.5}[pl]
    n = int(min(1 + (year - 11) * 1.4, 60))
    Pp, s = path_points()
    L = s[-1]
    rng = np.random.default_rng(year)
    offs = rng.random(n)
    for k in range(n):
        # each traveller: arrives (0.30-0.45), lights (0.45-0.5), descends from 0.5 at a steady pace
        t_arr = 0.30 + 0.12 * offs[k]
        t_go = t_arr + 0.07
        if u < t_go:
            continue
        sd = (u - t_go) * (L * 1.9) * (0.8 + 0.4 * offs[k])
        if sd > L:
            continue
        p = path_at(sd)
        sx, sy, z = scam.project(p + np.array([0, 1.7, 0]))
        if z <= 1:
            continue
        near = sd < 12.0
        fade = 1.0 - smoothstep(0.93, 1.0, u)
        en = (1.2 if near else 0.6) * vis * fade * ss * ss * min(1.0, 300.0 / max(z, 1.0) + 0.25)
        F2.glow(img, zb, sx, sy, 0.8 * ss, en, z=z, zbias=max(2.0, z * 0.01))
    # the few who stand at the fire to light their torches (only the nearest moments are figures)
    lights, amb = _fig_lights(1.6, pl)
    nfig = min(n, 3 if year < 30 else 5)
    for k in range(nfig):
        a0 = 0.30 + 0.12 * offs[k]
        if not (a0 < u < a0 + 0.10):
            continue
        v = (u - a0) / 0.10
        spot = BEACON + RIGHT * (0.9 + 0.35 * k) + FWD * (0.4 + 0.25 * k)
        p = LIP + (spot - LIP) * smoothstep(0.0, 0.35, v) if v < 0.5 else spot + (LIP - spot) * smoothstep(0.6, 1.0, v)
        fig = person('walk' if (v < 0.35 or v > 0.6) else 'stand', 0.3, scarf=False, torch=True, walk=u * 11.0,
                     face=-0.5)
        al = smoothstep(0.0, 0.15, v) * (1 - smoothstep(0.85, 1.0, v))
        draw_person(img, zb, scam, on_ground(p), fig, lights, amb, t, alpha=al * vis, flip=True)
        if v > 0.35:
            hs = fig[1]['hands'].get('torch')
            if hs is not None:
                tw = on_ground(p) + RIGHT * (-hs[0]) + np.array([0, hs[1] + 0.05, 0])
                sx, sy, z = scam.project(tw)
                F2.glow(img, zb, sx, sy, 0.8 * ss, 5.0 * al * ss * ss, z=z, zbias=0.5)


def draw_snow(img, scam, t, w=1.0, n=900):
    """Driving snow: streaks of flakes blowing hard across the frame, near ones large and soft."""
    if w <= 0:
        return
    rng = np.random.default_rng(5)
    H, W = img.shape[:2]
    X = rng.random(n)
    Y = rng.random(n)
    Z = rng.uniform(4.0, 60.0, n)
    sp = 1.6 + rng.random(n)
    x = (X * W * 1.4 + t * 900.0 * sp * (8.0 / Z)) % (W * 1.4) - 0.2 * W
    y = (Y * H + t * 140.0 * sp * (8.0 / Z)) % H
    L = 5.0 * (8.0 / Z) * sp
    lum = 0.03 * w * np.clip(8.0 / Z, 0.2, 1.5)
    ov = np.zeros_like(img)
    for i in range(n):
        x0, y0 = int(x[i]), int(y[i])
        x1, y1 = int(x[i] - L[i]), int(y[i] - L[i] * 0.16)
        c = float(lum[i])
        cv2.line(ov, (x0, y0), (x1, y1), (c, c * 1.05, c * 1.15), max(1, int(round(8.0 / Z[i]))), cv2.LINE_AA)
    img += cv2.GaussianBlur(ov, (0, 0), 0.8)


# --------------------------------------------------------------- cutaways ---

def render_m3(frame_local, n, scale, ss):
    """B8 · THE FAR PEAK (greybox): s1's summit and camera; a parent and a small child; the child points at the
    one pinprick on the far horizon: hers."""
    u = frame_local / float(n)
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tilt0 = math.degrees(math.atan((S1.HORIZON_Y0 - 402.0) / S1.F_FULL))
    tc = RC.RCam(S1.CAM0 + np.array([0.0, 1.2, -22.0 + 0.4 * u]), 0.0, tilt0 - 1.0, 0.0, S1.HFOV, W, H)
    fr = PI.Frame(tc, ss)
    key = ('m3', scale, ss)
    if key not in _ST:
        light = WD.night_light()
        PI.render_terrain(fr, 1447, np.zeros((0, WD.NCR)), light, np.zeros((0, 8)))
        _ST[key] = (fr.img.copy(), fr.zb.copy(), fr.dist.copy())
    img, zb, dist = [a.copy() for a in _ST[key]]
    scam = fr.src
    t = frame_local / FPS
    mask = (dist > 1e8).astype(np.float32)
    SK.splat_stars(img, scam, _cached('s1stars', lambda: SK.make_stars(14000, 101, lum_scale=7.0)), mask, t=t,
                   gain=ss * ss, scale=PI.src_scale(fr))
    # her pinprick, on the far horizon
    sx, sy, z = scam.project(S1.summit())
    F2.glow(img, zb, sx, sy, 0.8 * ss, 1.6 * ss * ss * F.flicker(t, 2), z=z, zbias=z * 0.05)
    F2.halo(img, zb, sx, sy, 3.0 * ss, 0.006, z=z, zbias=z * 0.05)
    Lk, amb, S, fogp, Q = WD.night_light()
    lights = [dict(dir=Lk[:3], col=Lk[3:6], I=Q[0])]
    point = smoothstep(0.25, 0.45, u)
    par, _ = person('look', 0.5, scarf=False, face=0.1)
    ch, _ = person('look', 0.0, scarf=False, child=True, face=0.4)
    FG.render(img, zb, scam, par, np.array([-1.1, 0.0, 16.0]), lights, amb=amb * 1.3, mats=M, t=t,
              write_depth=False, zbias=0.3)
    FG.render(img, zb, scam, ch, np.array([-0.35, 0.0, 16.3]), lights, amb=amb * 1.3, mats=M, t=t,
              write_depth=False, zbias=0.3)
    if point > 0:        # the child's arm raised toward the light
        d = FG.Drawing()
        d.new_group()
        sh = np.array([0.08, 0.72])
        tip = sh + np.array([-0.30, 0.20]) * point + np.array([0.0, -0.25]) * (1 - point)
        d.capsule(sh, tip, 0.04, 0.03, k=0.02)
        FG.render(img, zb, scam, d, np.array([-0.35, 0.0, 16.3]), lights, amb=amb * 1.3, mats=M, t=t,
                  write_depth=False, zbias=0.3)
    fr.img, fr.zb, fr.dist = img, zb, dist
    out, _, _ = PI.to_target(fr)
    return out


def render_m2(frame_local, n, scale):
    """B10 · THE FAR PEAK, years later: the v2 shepherd take (REUSE), the turn and the ignition."""
    f = 1440 + int(round(frame_local * (1475 - 1440) / max(n - 1, 1)))      # he sees her light and turns; cut before the thrust
    p = look.frame_path(os.path.join(ROOT, 'renders', 'montage_v2'), f)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return im[..., ::-1].astype(np.float32) / 255.0     # display-referred already


FINISH = dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)


def render_frame(frame, scale=0.25, ss=1.5):
    seg = segment(frame)
    i, a, b, kind, yr, pl, ex = seg
    if kind == 'cut':
        if yr == 'M3':
            hdr = render_m3(frame - a, b - a, scale, ss)
            return look.finish(hdr, **FINISH)
        return render_m2(frame - a, b - a, scale)
    hdr = render_night(frame, scale, ss, seg)
    return look.finish(hdr, **FINISH)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--plates', action='store_true')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default='lifetime')
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--range', default=None, help='a-b (inclusive)')
    ap.add_argument('--procs', type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out) if a.out.startswith('renders/') else os.path.join(OUTDIR, a.out)
    os.makedirs(out, exist_ok=True)
    if a.plates:
        plates(a.scale, a.ss, force=True)
        return
    _, total = schedule()
    if a.all:
        frames = list(range(total))
    elif a.range:
        s0, s1 = a.range.split('-')
        frames = list(range(int(s0), int(s1) + 1))
    else:
        frames = [int(x) for x in a.frames.split(',')]
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.frame_path(out, f))]
    get_plates(a.scale, a.ss)          # build the plate cache once before any fork
    if a.procs <= 1:
        _work((frames, a.scale, a.ss, out))
    else:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(frames[i::a.procs], a.scale, a.ss, out) for i in range(a.procs)])


def _work(args):
    frames, scale, ss, out = args
    for f in frames:
        t0 = time.time()
        img = render_frame(f, scale, ss)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.2f}s', flush=True)


if __name__ == '__main__':
    main()
