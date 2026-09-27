"""A15 . R16 THE WATCHERS (cut A, bars 54-55): A cut frames 4240-4399 -> renders/watchers_A/ (EDIT-v3).

From behind a watcher at the seventh Run fire: a hooded, cloaked back, backlit by the fire just beyond it, looking
across the ranges toward the cold glow that breathes on the horizon. On the lower ridges between, more watch-fires,
each with a small figure turned the same way. Eyelines only: the camera stays at a standing eye's height, never above
them, and the fires lie in depth along the look, never round anything. The strings hold; nothing moves but fire, cloth,
smoke and the red throbbing under the cloud.

World, light, glow, red and fires are A14's (RUN-A-L's `beaconrun_a`: light(), CR, fires_table(), stars(), FINISH,
end_cam()) through RUN-A3's `nighta` kit, so the first frame holds from R3's last position. Here: the seventh watcher
(also drawn by A14 from the catch on, through draw_figures()), and the watch-fires of the ridges between with their
figures (watchers_a.npz: `fires` rows for nighta, `figs` rows; A14 stacks the fires into its table so they burn in
its frames too). Figures are sdfppl travellers standing still in the wind: closed cowls, no face exists.

  python shots/run/watchers_a.py --build                                        # the ridges-between table
  python shots/run/watchers_a.py --frames 4240,4320,4399 --scale 0.5 --out renders/_wa_tests/t1
  python shots/run/watchers_a.py --range 4240-4399 --procs 4 --skip                # finals -> renders/watchers_A
  env: W15_P7="x,y,z" (the seventh fire, until A14's FIRES are fixed), W15_CAM="x,y,z,yaw,pitch,hfov" (static cam)
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import falsedawn as FD      # noqa: E402
import nighta as NA         # noqa: E402
import sdfppl as SP         # noqa: E402  (RUN-A2's; read-only)
from mt import fire as F    # noqa: E402
from mt.noise import smoothstep   # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FPS = 24.0
FR0, NFR = 4240, 160
EXTRA_NPY = os.path.join(HERE, 'watchers_a_extra.npy')    # (N, 6) nighta fire rows: A14 stacks them (LIT+EXTRA+CHAIN)
FIGS_NPY = os.path.join(HERE, 'watchers_a_figs.npy')      # (N, 8) the extra fires' watchers
UP = np.array([0.0, 1.0, 0.0])
LOOK_AZ = NA.GLOW_AZ
WIND = np.array([1.6, 0.0, 0.4]) / np.linalg.norm([1.6, 0.0, 0.4])     # nighta's smoke wind: the cloaks' lee
CATCH7 = 4200                    # A14: the seventh fire catches on bar 53 b3, beside the one who lit it


def br():
    """A14's module (RUN-A-L): imported lazily, so beaconrun_a may import this module's draw_figures in turn."""
    import beaconrun_a as BR
    return BR


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _dir(az_deg):
    a = math.radians(az_deg)
    return np.array([math.sin(a), 0.0, math.cos(a)])


def ground(x, z, CR=None):
    CR = br().CR if CR is None else CR
    return float(WD.ground(float(x), float(z), CR))


def chain():
    """The seven Run fires (7, 6) in ignition order: A14's FIRES (RUN-A-L), or, for look-dev with W15_P7="x,y,z", a
    placeholder chain: 7 at P7, 5/3/1 on crests about 120 m, 500 m and 1.2 km beyond it toward the glow (alternating
    sides), 6/4/2 about 5, 9 and 15 km out."""
    s = os.environ.get('W15_P7')
    if not s:
        FI = np.asarray(br().FIRES, np.float64).reshape(-1, 6)
        if FI.shape[0] < 7:
            raise SystemExit('watchers_a: beaconrun_a.FIRES has %d rows (need 7); set W15_P7 for look-dev'
                             % FI.shape[0])
        return FI[:7].copy()
    P7 = np.array([float(v) for v in s.split(',')])
    CR = br().CR
    out = np.zeros((7, 6))
    w = _dir(LOOK_AZ)
    side = np.cross(UP, w)
    for k, (d, lat) in enumerate(((1200.0, 110.0), (15000.0, -1500.0), (500.0, -60.0), (9000.0, 900.0),
                                  (120.0, 18.0), (5000.0, -500.0), (0.0, 0.0))):
        q = P7 + w * d + side * lat
        if d > 0:
            q = _crest_near(q, CR, 40.0 if d < 2000 else 400.0)
        else:
            q = P7.copy()
            q[1] = ground(q[0], q[2], CR)
        out[k] = [q[0], q[1], q[2], 3960.0 + 40.0 * k, 1.0, 900.0 + k]
    return out


def _crest_near(q, CR, rad):
    xs = np.linspace(q[0] - rad, q[0] + rad, 41)
    zs = np.linspace(q[2] - rad, q[2] + rad, 41)
    X, Z = np.meshgrid(xs, zs)
    H = np.zeros(X.size)
    WD.heights(X.ravel().copy(), Z.ravel().copy(), rad / 20.0, CR, H)
    k = int(np.argmax(H))
    return np.array([X.ravel()[k], H[k], Z.ravel()[k]])


def seventh_fire():
    return chain()[6]


# ------------------------------------------------------------------ the seventh watcher ---
def lighter_spot(P, lat=0.8):
    """Where a fire's lighter stands: 1.7 m on the near side of it (away from the glow), `lat` m to its right, on the
    ground; facing the glow."""
    w = _dir(LOOK_AZ)
    s = np.cross(UP, w)                           # the figure's right in this world (sdfppl's note)
    p = np.asarray(P[:3], np.float64) - w * 1.7 + s * lat
    p[1] = ground(p[0], p[2])
    return p, w


# THE PLATE (A15; A14's END_CAM settles into it): the camera looks PLATE_YAW deg right of the glow, so the glow's core
# sits in the left third; the seventh's lighter stands WATCHER_BEARING deg right of the axis (the right third), 1.2 m
# on our side of the fire and 0.7 m to its right, so the flames show just left of the figure and rim it.
PLATE_YAW = 7.0
WATCHER_BEARING = 7.3
FIRE_GAP = (1.2, 0.72)


def _plate_axes():
    u = _dir(LOOK_AZ + PLATE_YAW + WATCHER_BEARING)          # camera -> watcher
    return u, np.cross(UP, u)


def seventh_spot(P7):
    u, r = _plate_axes()
    p = np.asarray(P7[:3], np.float64) - u * FIRE_GAP[0] + r * FIRE_GAP[1]
    p[1] = ground(p[0], p[2])
    return p, _dir(LOOK_AZ)


def lighter_pose(f, f_ign):
    """(crouch 0..1, reach 0..1, turn deg): crouched and reaching into the fire at its catch, rising by +22, standing
    and turned full to the glow by +36; then still, a watcher."""
    rise = smoothstep(f_ign + 4, f_ign + 22, f)
    crouch = 1.0 - rise
    reach = 1.0 - smoothstep(f_ign + 1, f_ign + 10, f)
    turn = 14.0 * (1.0 - smoothstep(f_ign + 12, f_ign + 36, f))     # from the fire toward the glow
    return crouch, reach, turn


def seventh_pose(f):
    return lighter_pose(f, CATCH7)


def _figure(sc, spot, w, f, seed, h=1.0, crouch=0.0, reach=None, rgb=(0.030, 0.026, 0.022), staff=False,
            gust=1.0):
    t = f / FPS
    rng = np.random.default_rng(int(seed))
    s = np.cross(UP, w)
    ph = rng.uniform(0, 6.28)
    # a planted stance, weight a little back on the heels; the breath lifts the chest by a few millimetres
    aL = spot + s * 0.13 * h + w * 0.05 * h + UP * 0.07
    aR = spot - s * 0.13 * h - w * 0.07 * h + UP * 0.07
    breath = 0.004 * math.sin(2 * math.pi * t / 3.6 + ph)
    pel = spot + UP * ((0.93 - 0.42 * crouch) * h + breath) - w * 0.02 * h
    # the wind: a steady lean of the cloth to the lee, with gusts; the hem flutters
    g = 0.14 + 0.05 * math.sin(2 * math.pi * t / 5.3 + ph) + 0.03 * math.sin(2 * math.pi * t / 1.7 + 2 * ph)
    wind = WIND * g * gust
    tip = None
    if staff:
        tip = spot + w * 0.30 * h - s * 0.34 * h
        tip[1] = spot[1]
    SP.traveller(sc, pel, w, aL, aR, rgb, h=h, lean=0.05 + 0.50 * crouch, hem=0.14, cloak=(0.205, 0.33),
                 folds=int(9 + 3 * rng.random()), fold_depth=0.030, fold_phase=6.28 * rng.random(), peak=True,
                 staff_tip=tip, reach=reach, wind=wind, flutter=2.3 * t + ph)


LIGHTERS = ((6, 0.8, 7007), (4, -0.7, 7005), (2, 0.75, 7003), (0, -0.8, 7001))   # (chain index, side m, seed)


def lighter_scene(sc, f, P, lat, seed):
    spot, w = lighter_spot(P, lat)
    crouch, reach, turn = lighter_pose(f, P[3])
    ww = _dir(LOOK_AZ + (turn if lat >= 0 else -turn))
    rp = (np.asarray(P[:3]) + UP * 0.3) if reach > 0.2 else None
    _figure(sc, spot, ww, f, seed, h=1.0, crouch=crouch, reach=rp, rgb=(0.030, 0.026, 0.022), staff=False)


def hearth_scene(sc, P, seed, scale=1.0):
    """The seventh fire's low cairn: weathered, rounded field stones (lumpy round cones at random axes, each its own
    size, lightly fused where they touch) in three uneven courses to about 0.64 m, where nighta's recipe puts a size-0.6
    fire's base. No cut blocks, no even gaps. There before the catch (A14)."""
    rng = np.random.default_rng(int(seed))
    base = np.asarray(P[:3], np.float64)
    sc.begin(rgb=(0.050, 0.047, 0.045))
    courses, top = 3, 0.64 * scale
    for c in range(courses):
        y0 = top * c / courses
        rr = (0.44 - 0.15 * c / (courses - 1)) * scale
        n = 9 - 2 * c
        for m in range(n):
            a_ = 2.0 * math.pi * (m + 0.5 * (c % 2)) / n + rng.uniform(-0.2, 0.2)
            r0 = rng.uniform(0.075, 0.125) * scale
            q = base + np.array([math.cos(a_), 0.0, math.sin(a_)]) * rr * rng.uniform(0.88, 1.08) \
                + UP * (y0 + r0 * rng.uniform(0.55, 0.9))
            tang = np.array([-math.sin(a_), rng.uniform(-0.35, 0.35), math.cos(a_)])
            tang = tang / np.linalg.norm(tang) * rng.uniform(0.03, 0.10) * scale
            sc.cone(q - tang, q + tang, r0, r0 * rng.uniform(0.55, 0.9), mat=5, k=0.018 * scale)
    sc.end()


def seventh_scene(sc, f, P7):
    hearth_scene(sc, P7, 7070)
    spot, w = seventh_spot(P7)
    crouch, reach, turn = lighter_pose(f, P7[3])
    rp = (np.asarray(P7[:3]) + UP * 0.3) if reach > 0.2 else None
    _figure(sc, spot, _dir(LOOK_AZ + turn), f, 7007, h=1.0, crouch=crouch, reach=rp, rgb=(0.030, 0.026, 0.022))


# ------------------------------------------------------------------ the ridges between ---
def build(CH, cam_pos, n_extra=4, seed=4815):
    """EXTRA: watch-fires on the lower ridges ahead of the seventh summit (toward the glow), beyond chain fire 5 and
    sparser than the chain (so the seven still read as A14's events), already burning (caught with hers), each with a
    watcher 1.4-1.9 m on its near side turned toward the glow. Crests only (local maxima), lower than the seventh, in a
    +-32 deg wedge about the look, 170-1600 m out, in sight of cam_pos, clear of the chain fires (> 140 m), spread in
    angle as seen from the camera (no row, no ring). -> watchers_a_extra.npy (fires) + watchers_a_figs.npy."""
    from scipy import ndimage as ndi
    CR = br().CR
    rng = np.random.default_rng(seed)
    P7 = np.asarray(CH[6, :3], np.float64)
    dx = 6.0
    R = 2300.0
    xs = P7[0] + np.arange(-R, R + dx, dx)
    zs = P7[2] + np.arange(-R, R + dx, dx)
    X, Z = np.meshgrid(xs, zs)
    H = np.zeros(X.size)
    WD.heights(X.ravel().copy(), Z.ravel().copy(), dx, CR, H)
    H = H.reshape(X.shape)
    mx = ndi.maximum_filter(H, size=7)
    mn = ndi.minimum_filter(H, size=15)
    cands = []
    for j, i in np.argwhere((H == mx) & (H > WD.CLOUD_Y + 40.0) & (H < P7[1] - 4.0)):
        x, z, y = X[j, i], Z[j, i], H[j, i]
        d = math.hypot(x - P7[0], z - P7[2])
        az = math.degrees(math.atan2(x - P7[0], z - P7[2]))
        da = (az - LOOK_AZ + 180.0) % 360.0 - 180.0
        dc = np.hypot(CH[:, 0] - x, CH[:, 2] - z).min()
        if 200.0 <= d <= 2200.0 and abs(da) <= 45.0 and y - mn[j, i] > 3.0 and dc > 140.0:
            cands.append((x, y, z, d, da, y - mn[j, i]))
    C = np.array(cands).reshape(-1, 6)
    ok = []
    for x, y, z, d, da, pr in C:
        tgt = np.array([x, y + 2.0, z])
        L = np.linalg.norm(tgt - cam_pos)
        q = np.geomspace(1.5 / L, 0.97, 160)
        Q = cam_pos[None] + (tgt - cam_pos)[None] * q[:, None]
        hq = np.zeros(len(q))
        WD.heights(Q[:, 0].copy(), Q[:, 2].copy(), 2.0, CR, hq)
        ok.append(not (hq > Q[:, 1] + 0.3).any())
    C = C[np.array(ok, bool)] if len(C) else C
    print('crests', len(ok), 'in sight', len(C), flush=True)
    # in the plate's frame: inside it with a margin, below the far skyline band, clear of the chain fires (>= 70 px)
    # and of the foreground watcher's silhouette (and its fire), so each extra reads as one more fire, not a clump
    pcam = camera(FR0)
    spot7, _ = seventh_spot(CH[6])
    wa = pcam.project(spot7 + UP * 1.9)
    wb = pcam.project(spot7)
    cx_, cy_, _z = pcam.project(CH[:, :3] + UP * 1.0)
    order = sorted(range(len(C)), key=lambda k: -C[k, 5] * rng.uniform(0.6, 1.4))
    picked = []
    for k in order:
        sx, sy, sz = pcam.project(C[k, :3] + UP * 1.0)
        if sz <= 0 or not (80.0 < sx < 1840.0 and 330.0 < sy < 780.0):
            continue
        if wa[0] - 140.0 < sx < wa[0] + 120.0 and sy > wa[1] - 60.0:
            continue
        if np.min(np.hypot(cx_ - sx, cy_ - sy)) < 70.0 or any(math.hypot(sx - q[1], sy - q[2]) < 90.0 for q in picked):
            continue
        picked.append((k, sx, sy))
        if len(picked) >= n_extra:
            break
    fires, figs = [], []
    for k, _sx, _sy in picked:
        x, y, z, d, da, pr = C[k]
        fires.append([x, y, z, rng.uniform(3690.0, 3760.0), rng.uniform(0.55, 0.8), float(rng.integers(0, 9999))])
        w = _dir(LOOK_AZ + rng.uniform(-8.0, 8.0))
        sd = np.cross(UP, w)
        sp = np.array([x, y, z]) - w * rng.uniform(1.4, 1.9) + sd * rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 0.9)
        sp[1] = ground(sp[0], sp[2], CR)
        figs.append([sp[0], sp[1], sp[2], LOOK_AZ + rng.uniform(-8.0, 8.0), rng.uniform(0.94, 1.05),
                     float(rng.random() < 0.4), float(rng.integers(0, 9999)), d])
    EX = np.array(fires).reshape(-1, 6)
    FG = np.array(figs).reshape(-1, 8)
    np.save(EXTRA_NPY, EX)
    np.save(FIGS_NPY, FG)
    print('extra watch-fires', len(EX), 'at', np.round(FG[:, 7]).astype(int), 'm ->', EXTRA_NPY, flush=True)
    return EX, FG


_TAB = None


def table():
    """(EXTRA fires (N, 6), their watchers (N, 8): x, y, z, face az, h, staff, seed, dist)."""
    global _TAB
    if _TAB is None:
        ex = np.load(EXTRA_NPY) if os.path.exists(EXTRA_NPY) else np.zeros((0, 6))
        fg = np.load(FIGS_NPY) if os.path.exists(FIGS_NPY) else np.zeros((0, 8))
        _TAB = (ex.reshape(-1, 6), fg.reshape(-1, 8))
    return _TAB


def extra_fires():
    """EXTRA, for A14's fires_table() = LIT + EXTRA + CHAIN."""
    return table()[0]


def figures_scene(f, near_only=False, CH=None):
    """Every watcher at frame f: the lighters of chain fires 7, 5, 3, 1 (each crouched at its own catch, then standing
    toward the glow) and the EXTRA fires' watchers (standing). near_only: the seventh's lighter alone."""
    CH = chain() if CH is None else CH
    sc = SP.Scene()
    seventh_scene(sc, f, CH[6])
    for idx, lat, seed in (() if near_only else LIGHTERS[1:]):
        lighter_scene(sc, f, CH[idx], lat, seed)
    if not near_only:
        for x, y, z, az, h, staff, seed, d in table()[1]:
            _figure(sc, np.array([x, y, z]), _dir(az), f, seed, h=h, staff=staff > 0.5,
                    rgb=(0.030 + 0.006 * (seed % 3), 0.026, 0.022), gust=1.0)
    return sc


def draw_figures(img, zb, scam, f, LT, light, near_only=False, CH=None):
    """The watchers in a lens-shift SOURCE camera: after the terrain and the stars, BEFORE the fires, so a flame and
    its air glow stay behind a figure standing in front of it (the backlit silhouette). LT: the frame's point lights;
    light: the world light tuple (Lk, amb, S, fogp, Q). A14 calls this for every frame where a figure is >= 3 px."""
    sc = figures_scene(f, near_only, CH)
    Pr, Ob = sc.arrays()
    Lk, amb, S, fogp, Q = light
    md = np.asarray(WD.MOON_DIR, np.float64)
    md = md / np.linalg.norm(md)
    moon = np.array([md[0], md[1], md[2], Lk[3], Lk[4], Lk[5], Q[0]])
    C = scam.params()
    SP.render(img, zb, C, Pr, Ob, np.asarray(LT, np.float64).reshape(-1, 8), moon, amb * 1.3, fogp,
              float(scam.pos[1]))


# ------------------------------------------------------------------ the camera ---
EYE = 1.90                       # a tall standing eye: level with the lighter's hood, never above it
BACK = 14.5                      # metres behind the watcher
HFOV = 40.0
PITCH = -1.8
PUSH = 3.5                       # metres of slow push toward the lighter, easing in from A14's hold


def plate(P7):
    """A15's plate = A14's END_CAM: (pos, yaw, pitch, hfov). At the seventh lighter's eye height (never above them;
    a standing eye if the ground behind rises), BACK m behind the figure on the camera->watcher bearing."""
    spot, w = seventh_spot(P7)
    u, r = _plate_axes()
    pos = spot - u * BACK
    pos[1] = max(ground(pos[0], pos[2]), spot[1]) + EYE
    return pos, LOOK_AZ + PLATE_YAW, PITCH, HFOV


def default_cam(P7):
    return plate(P7)


def plate_check(CH, W=1920, H=804):
    """For picking the seventh summit: what the plate sees. The lighter's size and place, the seventh fire's place,
    and for every chain fire its screen position and whether the terrain hides it (the seventh summit's own brink
    must not: the land has to fall away beyond the fire)."""
    CR = br().CR
    pos, yaw, pitch, hfov = plate(CH[6])
    cam = RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)
    spot, w = seventh_spot(CH[6])
    a = cam.project(spot + UP * 1.75)
    b = cam.project(spot)
    rep = {'cam': (np.round(pos, 2), yaw, pitch, hfov), 'watcher_px': float(b[1] - a[1]),
           'watcher_xy': (float(a[0]), float(a[1]), float(b[1]))}
    for k in range(7):
        tgt = CH[k, :3] + UP * 1.2
        sx, sy, z = cam.project(tgt)
        L = np.linalg.norm(tgt - pos)
        q = np.geomspace(0.5 / L, 0.985, 200)
        Q = pos[None] + (tgt - pos)[None] * q[:, None]
        hq = np.zeros(len(q))
        WD.heights(Q[:, 0].copy(), Q[:, 2].copy(), 1.0, CR, hq)
        dd = np.hypot(Q[:, 0] - pos[0], Q[:, 2] - pos[2])
        hid = bool((hq - dd * dd / (2 * WD.R_EARTH) > Q[:, 1] + 0.2).any())
        rep['fire%d' % (k + 1)] = (round(float(sx)), round(float(sy)), round(float(z)), 'HIDDEN' if hid else 'seen')
    return rep


def camera(f, W=1920, H=804):
    ov = os.environ.get('W15_CAM')
    if ov:
        x, y, z, yaw, pitch, hfov = [float(v) for v in ov.split(',')]
        return RC.RCam(np.array([x, y, z]), yaw, pitch, 0.0, hfov, W, H)
    P7 = seventh_fire()
    BR = br()
    if getattr(BR, 'END_CAM', None) is not None and os.environ.get('W15_OWN_CAM') != '1':
        pos, yaw, pitch, hfov = BR.end_cam()
    else:
        pos, yaw, pitch, hfov = default_cam(P7)
    # A14 holds still on this plate into the cut, so the push eases in from rest (quintic) and is still drifting
    # gently on the last frame (the cut to A16 follows the look)
    u = min(max((f - FR0) / float(NFR - 1), 0.0), 1.0)
    s5 = lambda x: x * x * x * (x * (6.0 * x - 15.0) + 10.0)
    e = s5(0.8 * u) / s5(0.8)
    pos = np.asarray(pos, np.float64) + _dir(yaw) * PUSH * e
    return RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)


# ------------------------------------------------------------------ the frame ---
def render(f, scale=1.0, ss=1.5):
    BR = br()
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tcam = camera(f, W, H)
    fr = PI.Frame(tcam, ss)
    CH = chain()
    FT = np.asarray(BR.fires_table(), np.float64).reshape(-1, 6)
    for grp in (CH, extra_fires()):                 # look-dev, or before A14 stacks them: add what is missing
        for row in grp:
            if not np.any(np.all(np.isclose(FT[:, :3], row[None, :3]), axis=1)):
                FT = np.vstack([FT, row[None]])
    LT = NA.lt_rows(FT, f)
    PL = NA.pl_rows(FT, max_dist=6000.0, cam_pos=tcam.pos)
    light = BR.light(f)
    Lk, amb, S, fogp, Q = light
    scam = fr.src
    C = scam.params()
    t = f / FPS
    Pp = np.array([scam.pos[0], scam.pos[2], t, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(Pp, BR.CR, C, 0.2, 90000.0, 0.0035, 0.35, FD.terrain_hmax(BR.CR), 9, D)
    fr.img = np.zeros((scam.H, scam.W, 3), np.float32)
    fr.zb = np.zeros((scam.H, scam.W), np.float32)
    fr.dist = np.zeros((scam.H, scam.W), np.float32)
    WD.shade(C, D, Pp, BR.CR, S, LT, Lk, Q, amb, fogp, fr.img, fr.zb, fr.dist, PL)
    WD.cloud_glow(C, D, Pp, BR.CR, NA.ug_rows(f), fogp, fr.img)
    kill = np.zeros(fr.dist.shape, np.float32)
    skl = NA.skyline(scam.pos, BR.CR, WD)
    NA.glow_pass(fr.img, fr.dist, kill, C, NA.glow_gp(f), skl[0], skl[1], skl[2], NA.HAZE_K, NA.HAZE_D)
    pxs = PI.src_scale(fr)
    from mt import sky as SK
    SK.splat_stars(fr.img, scam, BR.stars(), kill, t=t, gain=ss * ss, scale=pxs)
    draw_figures(fr.img, fr.zb, scam, f, LT, light, CH=CH)
    NA.fires_layer(fr.img, fr.zb, scam, FT, f, pxs, fogp=fogp, wmod=WD)
    if hasattr(BR, 'draw_flares'):
        BR.draw_flares(fr.img, fr.zb, scam, f, pxs, fogp)     # A14's catch flares run on into A15 (the last to 4260)
    img, zb, di = PI.to_target(fr)
    return img


def finish(img):
    return PI.look.finish(img, **br().FINISH)


def _work(args):
    frames, scale, out, skip, wid = args
    import cv2
    cv2.setNumThreads(1)
    for f in frames:
        path = PI.look.frame_path(out, f)
        if skip and (os.path.exists(path) or os.path.exists(path[:-4] + '.jpg')):
            continue
        t1 = time.time()
        PI.look.save_png(path, finish(render(f, scale)))
        print('[w%d] watchers_A f%d %.1fs' % (wid, f, time.time() - t1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='')
    ap.add_argument('--range', default='')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'renders', 'watchers_A'))
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--build', action='store_true')
    a = ap.parse_args()
    if a.build:
        CH = chain()
        pos = default_cam(CH[6])[0]
        BR = br()
        if getattr(BR, 'END_CAM', None) is not None and os.environ.get('W15_OWN_CAM') != '1':
            pos = np.asarray(BR.end_cam()[0], np.float64)
        build(CH, np.asarray(pos, np.float64))
        return
    frames = []
    if a.range:
        x0, x1 = a.range.split('-')
        frames = list(range(int(x0), int(x1) + 1))
    if a.frames:
        frames += [int(v) for v in a.frames.split(',') if v]
    frames = [f for f in frames if FR0 <= f < FR0 + NFR]
    if not frames:
        raise SystemExit('no frames in %d-%d' % (FR0, FR0 + NFR - 1))
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    if a.procs <= 1:
        _work((frames, a.scale, out, a.skip, 0))
        return
    import multiprocessing as mp
    chunks = [frames[i::a.procs] for i in range(a.procs)]
    with mp.get_context('fork').Pool(a.procs) as pool:
        pool.map(_work, [(c, a.scale, out, a.skip, i) for i, c in enumerate(chunks) if c])


if __name__ == '__main__':
    main()
