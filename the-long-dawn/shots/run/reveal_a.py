"""A13 . R2-A THE REVEAL (cut A, bar 47 b1 to bar 48 b3): A cut frames 3680-3799 -> renders/reveal_A/ (EDIT-v3).

H1's take (HEROINE's `shots/hills/beacon.py`, the roar and the pull-back to her lone fire) ends at src 1555 = cut 3679
with the camera at rest 46 m behind her, 5 m up. The reveal CONTINUES THAT TAKE, with H1's own renderer, so the cut
is seamless: the same summit, basket, fire, smoke, her figure and scarf, world layer, grade and finish. Out of that rest
the camera cranes gently back, up and left (toward the cold glow at az -30), and on every ridge to the horizon fires
catch in the same breath as hers (all at once, never in a wave out from her: these are lit together, not in reply).
Beyond the ranges the cold glow pulses once a bar; under the cloud sea dull red patches pulse on the race's beat. It
ends on her small silhouette (>= 40 px) beside her fire, turned toward the far horizon; the cut follows her look into
KARST. No figure but hers, no face; nothing but fire and the three lights of the night.

How: `beacon` is imported READ-ONLY and configured exactly as `shots/hills/render.py --shot beacon_v3_roar2` (the
current H1-A). Two runtime hooks replace module names in this process only (beacon.py is never edited):
  * `camera`: H1's own formula up to src 1555, then the crane (from rest, a quintic ease with a small settle drift);
  * `world_layer`: H1's world layer plus the A night from `nighta.py` (the fires' flames and their warm pools on the
    snow, the cold glow over the sky and the far haze with A2's star kill, the red under the cloud).
The take's simulations (scarf, hair, sparks) are extended past F1 = 1555 with the frames before it unchanged.
Source guards: if beacon.py's `camera` or `world_layer` change, this prints REVEAL_A GUARD and stops (set
REVEAL_A_ALLOW_DRIFT=1 to render anyway), because a stale copy would break the match at the cut.
Only the A night fades in over the first beats (the glow and the red were never in H1's frames): see RAMP.

  python shots/run/reveal_a.py --frames 3680,3740,3799 --scale 0.35 --out renders/_ra_tests/t1     # test stills
  python shots/run/reveal_a.py --range 3680-3799 --procs 4 --skip                                   # finals
  python shots/run/reveal_a.py --probe                                                              # poses, sizes
  python shots/run/reveal_a.py --build-fires                                                        # the fire table
"""
import argparse
import hashlib
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                   # shots/run
HILLS = os.path.abspath(os.path.join(HERE, '..', 'hills'))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))              # the-long-dawn
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if HILLS not in sys.path:
    sys.path.insert(0, HILLS)                                       # hills first: beacon's own imports
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

CUT0, CUT1 = 3660, 3800                  # A13 R2-A: from 3660 (A-FIX: after B's reveal 3612-3659) .. bar 48 b3 (KARST from 3800)
SRC_OFF = 3600 - 1476                    # cut = src + 2124 (the roar: src 1476 = cut 3600, bar 46 b1)
TAKE_END = 1555                          # H1's last src frame (cut 3679)
SRC1 = CUT1 - 1 - SRC_OFF                # 1675
FIRES_NPY = os.path.join(HERE, 'reveal_a_fires.npy')
# A-FIX (28 Sep): A13 now follows B's reveal of her by her fire (A 3612-3679), and the cut to these ranges must show
# fires ANSWERING across them: at 3-5 px they were invisible. FAR_PXS scales the unresolved fires' hot point and aura
# (nighta draws them at a fixed 0.95 / 3.4 / 11 px x pxs, energy x pxs^2). GLOW_RAYS 0: the glow's shadow rays fanned
# up from the tallest peak like a searchlight.
FAR_PXS = 3.0
GLOW_RAYS = 0.0

GUARD = {'camera': '49dd99955a91d12e61f657c861031c7f',  # A-FIX branch only (AFIX False here)
          'world_layer': 'f53a1f551995e845cb816731fb4be17b'}   # A-FIX branch only


def _guard():
    src = open(os.path.join(HILLS, 'beacon.py')).read()
    bad = []
    for name, h in GUARD.items():
        m = re.search(r'^def %s\(.*?(?=^def |^class |^# ----)' % name, src, re.S | re.M)
        if m is None or hashlib.md5(m.group(0).encode()).hexdigest() != h:
            bad.append(name)
    if bad:
        msg = ('REVEAL_A GUARD: shots/hills/beacon.py changed %s since the reveal copied it; re-check the hooks so '
               'the cut at 3679/3680 still matches H1' % ', '.join(bad))
        print(msg, file=sys.stderr, flush=True)
        if os.environ.get('REVEAL_A_ALLOW_DRIFT') != '1':
            raise SystemExit(msg)


_guard()
import beacon as BK          # noqa: E402  (HEROINE's H1 take; read-only)
BK.V3_REKEY = True           # == render.py make_shot('beacon_v3_roar2')
BK.use_master_timing()
BK.apply_h5_calls()
BK.apply_roar2()
BK.BEACON_SMOKE_AMB = (0.020, 0.026, 0.045)   # A-FIX: her smoke a pale moonlit veil, not a dark smudge over the sky
assert BK.ROAR == 1476 and BK.F1 == TAKE_END, (BK.ROAR, BK.F1)
_orig_s1_world = BK.s1_world

import nighta as NA          # noqa: E402
from core import Camera, fnoise1, smoothstep, look   # noqa: E402  (hills core)

FPS = 24.0
# the A night fades in: H1's frames never had the glow or the red, so both come up out of the take's rest
RAMP = (CUT0, CUT0 + 48)


def night_ramp(fc):
    u = min(max((fc - RAMP[0]) / float(RAMP[1] - RAMP[0]), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


# ------------------------------------------------------------------ the camera ---
def h1_camera(f, scale):
    """H1's own camera formula (beacon.camera, HEROINE_V2) with the take's F1 = 1555 (our F1 is extended)."""
    t = f / FPS
    hand = 0.004 * fnoise1(t * 0.9, 1.0), 0.003 * fnoise1(t * 0.8, 2.0)
    c_pos = np.array([0.42 + hand[0], 0.96 + hand[1], -2.15])
    c_tgt = np.array([0.40, 1.02, 0.0])
    w_pos = np.array([1.50, 5.2, -46.0])
    w_tgt = np.array([0.40, 0.0, 60.0])
    u = smoothstep(BK.ROAR, TAKE_END + 6, f)
    u = 1 - (1 - u) ** 3
    pos = c_pos + (w_pos - c_pos) * u
    tgt = c_tgt + (w_tgt - c_tgt) * (u ** 0.8)
    hfov = 40.0 + 26.0 * u
    cam = Camera(pos, hfov=hfov, scale=scale)
    yaw, pitch = cam.look_at(tgt)
    focus = float(np.linalg.norm(np.array([0.4, 1.0, 0.0]) - pos))
    return pos, yaw, pitch, hfov, focus, u


# the crane (her summit's local frame: +z north, +x east, y up from her summit's top; she faces north)
END_POS = np.array([8.2, 14.0, -59.2])
END_YAW = -14.0                          # deg: the glow (az -30) in the left third, her right of centre
END_PITCH = -4.5
END_HFOV = 60.0
EASE_A = 0.92                            # < 1: the crane still drifts a little on the last frame


def _s5(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def crane_e(f):
    s = (f - TAKE_END) / float(SRC1 - TAKE_END)
    return _s5(s * EASE_A) / _s5(EASE_A)


def camera(f, scale):
    """beacon.camera's signature: (Camera, focus, u)."""
    if f <= TAKE_END:
        pos, yaw, pitch, hfov, focus, u = h1_camera(f, scale)
        return Camera(pos, yaw=yaw, pitch=pitch, hfov=hfov, scale=scale), focus, u
    p0, yaw0, pitch0, hfov0, _, _ = h1_camera(TAKE_END, scale)
    e = crane_e(f)
    ey = e ** 0.85                                # the rise leads a little
    ea = min(max(e, 0.0), 1.0) ** 1.1             # the pan follows
    pos = p0 + (END_POS - p0) * np.array([e, ey, e])
    yaw = yaw0 + (math.radians(END_YAW) - yaw0) * ea
    pitch = pitch0 + (math.radians(END_PITCH) - pitch0) * ea
    hfov = hfov0 + (END_HFOV - hfov0) * e
    focus = float(np.linalg.norm(np.array([0.9, 1.0, 0.3]) - pos))
    return Camera(pos, yaw=yaw, pitch=pitch, hfov=hfov, scale=scale), focus, 1.0


# ------------------------------------------------------------------ the fires on every ridge ---
def _heights(WDm, xs, zs, fp):
    out = np.zeros(xs.size)
    WDm.heights(np.ascontiguousarray(xs, np.float64).ravel(), np.ascontiguousarray(zs, np.float64).ravel(), fp,
                np.zeros((0, WDm.NCR)), out)
    return out.reshape(np.shape(xs))


def _summit_blocks(X, Y, Z):
    """True where a ray (rows of local-coordinate samples) passes under her 10 cm summit grid (peaks.py)."""
    import peaks as PK
    S, _M = PK.summit_grids()
    out = np.zeros(X.shape[0], bool)
    for a in range(X.shape[0]):
        for b in range(X.shape[1]):
            if PK.summit_h(S, X[a, b], Z[a, b]) > Y[a, b] + 0.2:
                out[a] = True
                break
    return out


def _visible(WDm, eye, pts, off, lift=2.5, n=300):
    """Line of sight from eye to each point + lift (curvature included), log-sampled from 3 m out against the
    terrain and, near the eye, her summit grid."""
    ok = np.ones(len(pts), bool)
    R = WDm.R_EARTH
    for k0 in range(0, len(pts), 400):
        P = pts[k0:k0 + 400]
        tgt = P + np.array([0.0, lift, 0.0])
        L = np.linalg.norm(tgt - eye[None], axis=1)
        s = np.geomspace(3.0 / L, np.full(len(L), 0.985), n, axis=1)[..., None]
        Q = eye[None, None, :] + (tgt[:, None, :] - eye[None, None, :]) * s
        dist = np.hypot(Q[..., 0] - eye[0], Q[..., 2] - eye[2])
        h = _heights(WDm, Q[..., 0], Q[..., 2], 20.0) - dist * dist / (2 * R)
        blk = (h > Q[..., 1] + 0.5).any(axis=1)
        near = dist < 130.0
        ql = Q - off[None, None, :]
        Xl = np.where(near, ql[..., 0], 1e6)
        blk |= _summit_blocks(Xl, np.where(near, ql[..., 1], 0.0), np.where(near, ql[..., 2], 1e6))
        ok[k0:k0 + 400] = ~blk
    return ok


# (from m, to m, how many, screen spacing px at full res): the near needles, the mid ranges, the islands, the horizon
BANDS = ((300.0, 3000.0, 12, 90.0), (3000.0, 12000.0, 26, 54.0), (12000.0, 30000.0, 30, 36.0),
         (30000.0, 80000.0, 18, 40.0))


def build_fires(seed=4700):
    """Summits around her (world coords) that the crane can see, thinned to a spread of fires on every ridge to the
    horizon; ignition frames: those in H1's last frame catch between 3684 and 3740 (random, not ordered by distance),
    those the pan brings in caught with hers (already burning when they enter). -> reveal_a_fires.npy (N, 6):
    x, y, z, f_ign (A cut frame), size, seed."""
    from scipy import ndimage as ndi
    Wd = _orig_s1_world()
    WDm = Wd['WD']
    off = Wd['off']
    rng = np.random.default_rng(seed)
    cands = []
    for (r0, r1, dx) in ((350.0, 4000.0, 12.0), (3000.0, 20000.0, 50.0), (15000.0, 75000.0, 150.0)):
        n = int(2 * r1 / dx) + 1
        xs = off[0] + np.linspace(-r1, r1, n)
        zs = off[2] + np.linspace(-0.25 * r1, r1, int(1.25 * r1 / dx) + 1)
        X, Z = np.meshgrid(xs, zs)
        H = _heights(WDm, X, Z, dx)
        win = 9
        mx = ndi.maximum_filter(H, size=win)
        mn = ndi.minimum_filter(H, size=2 * win + 1)
        for j, i in np.argwhere((H == mx) & (H > WDm.CLOUD_Y + 60.0)):
            x, z = X[j, i], Z[j, i]
            d = math.hypot(x - off[0], z - off[2])
            if not (r0 <= d < r1):
                continue
            az = math.degrees(math.atan2(x - off[0], z - off[2]))
            if not (-58.0 <= az <= 40.0):
                continue
            cands.append((x, H[j, i], z, H[j, i] - mn[j, i], d, az))
    C = np.array(cands)
    print('candidates', len(C), flush=True)
    # the two ends of the crane, world coords
    p0 = h1_camera(TAKE_END, 1.0)[0] + off
    p1 = END_POS + off
    vis0 = _visible(WDm, p0, C[:, :3], off)
    vis1 = _visible(WDm, p1, C[:, :3], off)
    keep = vis0 | vis1
    C, vis0, vis1 = C[keep], vis0[keep], vis1[keep]
    print('visible', len(C), flush=True)
    # screen positions in the end camera (and the start) at full res, for a spread without clumps
    cam1, _, _ = camera(SRC1, 1.0)
    cam0, _, _ = camera(TAKE_END, 1.0)
    sx1, sy1, z1 = cam1.project(C[:, :3] - off)
    sx0, sy0, z0 = cam0.project(C[:, :3] - off)
    # "in H1's last frame" keeps a 60 px margin, so nothing already burning can bleed a halo into its edge at 3680
    in0 = vis0 & (z0 > 0) & (sx0 > -60) & (sx0 < 1980) & (sy0 > -60) & (sy0 < 864)
    in1 = vis1 & (z1 > 0) & (sx1 > -4) & (sx1 < 1924) & (sy1 > -4) & (sy1 < 808)
    # quotas per distance band, so every ridge carries fire and the far islands don't string a line of lamps along
    # the horizon; spacing jittered per pick (no even rhythm); prominence ranks crests within a band
    picked = []
    for d0, d1, quota, sep0 in BANDS:
        sel = np.nonzero((C[:, 4] >= d0) & (C[:, 4] < d1) & (in0 | in1))[0]
        if len(sel) == 0:
            continue
        score = np.log1p(np.maximum(C[sel, 3], 1.0)) * rng.uniform(0.5, 1.5, len(sel))
        n = 0
        for k in sel[np.argsort(-score)]:
            sx, sy = (sx1[k], sy1[k]) if in1[k] else (sx0[k], sy0[k])
            sep = sep0 * rng.uniform(0.7, 1.6)
            if any(math.hypot(sx - q[0], sy - q[1]) < sep for q in picked):
                continue
            picked.append((sx, sy, k))
            n += 1
            if n >= quota:
                break
        print('band %5.0f-%5.0f m: %d of %d' % (d0, d1, n, len(sel)), flush=True)
    rows = []
    for sx, sy, k in picked:
        x, y, z, prom, d, az = C[k]
        if in0[k]:
            f_ign = 3684.0 + 56.0 * rng.random() ** 1.4       # in H1's last frame: catch after the cut, at once
        else:
            f_ign = 3606.0 + 90.0 * rng.random()             # the pan brings them in already burning
        size = 1.0 if rng.random() < 0.8 else rng.uniform(1.3, 2.0)
        rows.append([x, y, z, f_ign, size, float(rng.integers(0, 10000))])
    F = np.array(rows)
    np.save(FIRES_NPY, F)
    print('fires', len(F), 'in H1 frame', int(sum(in0[k] for _, _, k in picked)), '->', FIRES_NPY, flush=True)
    return F


_FIRES = None


def fires():
    global _FIRES
    if _FIRES is None:
        _FIRES = np.load(FIRES_NPY) if os.path.exists(FIRES_NPY) else np.zeros((0, 6))
    return _FIRES


# ------------------------------------------------------------------ the world layer ---
def world_layer(cam, f, t, reveal):
    """beacon.world_layer (copied: same canvas, march, shade, resample, stars and grade) plus the A night."""
    import cv2
    Wd = _orig_s1_world()
    WD = Wd['WD']
    fc = f + SRC_OFF                                  # A cut frame
    ramp = night_ramp(fc)
    k = 0.25 if f < BK.ROAR + 4 else 1.0
    m = 32.0 * cam.scale
    fl = cam.f * k
    Wl = int(math.ceil((cam.W + 2.0 * m) * k))
    Hl = int(math.ceil((cam.H + 2.0 * m) * k))
    pos = cam.pos + Wd['off']
    mc = Wd['Cam'](pos, math.degrees(cam.yaw), math.degrees(cam.pitch),
                   math.degrees(2.0 * math.atan(0.5 * Wl / fl)), Wl, Hl)
    C = mc.params()
    P = np.array([pos[0], pos[2], t, 0.0])
    D = np.zeros((Hl, Wl))
    WD.march(P, Wd['CR'], C, 0.2, 90000.0, 0.0035, 0.35, 700.0, 9, D)
    Lk, amb, S, fogp, Q = Wd['light']
    if ramp > 0.0:
        # into the shared A night's cloud sea and aerial depth (nighta.night_light) with the rest of the night; never
        # the snow knobs (Q[18] skews the snow noise's domain: blended, the texture would slide), so H1's ranges keep
        # their snow exactly
        knobs = dict(NA.cloud_values())
        knobs['f0'] = NA.TERRAIN['f0']
        Q, fogp = NA.apply_knobs(Q, fogp, knobs, ramp)
    out = np.zeros((Hl, Wl, 3), np.float32)
    zb = np.zeros((Hl, Wl), np.float32)
    di = np.zeros((Hl, Wl), np.float32)
    FR = fires()
    near = FR[np.hypot(FR[:, 0] - pos[0], FR[:, 2] - pos[2]) < 12000.0] if len(FR) else FR
    LT = NA.lt_rows(near, fc)
    PL = NA.pl_rows(FR, 4000.0, pos)
    WD.shade(C, D, P, Wd['CR'], S, LT, Lk, Q, amb, fogp, out, zb, di, PL)
    # ---- the A night (nighta): the red under the cloud, the cold glow, the fires
    if ramp > 0.0:
        WD.cloud_glow(C, D, P, Wd['CR'], NA.ug_rows(fc, gain=ramp), fogp, out)
    kill = np.ones((Hl, Wl), np.float32)
    if ramp > 0.0:
        GP = NA.glow_gp(fc)
        GP[2] *= ramp
        GP[11] *= GLOW_RAYS
        skl = NA.skyline(pos, Wd['CR'], WD)
        NA.glow_pass(out, di, kill, C, GP, skl[0], skl[1], skl[2], NA.HAZE_K, NA.HAZE_D)
    pxs = cam.scale * k
    NA.fires_layer(out, zb, mc, FR, fc, pxs * FAR_PXS, fogp=fogp, wmod=WD)
    sky = (di > 1e8).astype(np.float32)
    starm = sky * np.where(di > 1e8, kill, 1.0).astype(np.float32)
    # ---- as beacon.world_layer: our pixel centres -> world rays -> the canvas
    jj, ii = np.mgrid[0:cam.H, 0:cam.W].astype(np.float64)
    x = (ii + 0.5 - cam.cx) / cam.f
    y = (cam.cy - (jj + 0.5)) / cam.f
    R = cam.R
    dx = R[0, 0] * x + R[0, 1] * y + R[0, 2]
    dy = R[1, 0] * x + R[1, 1] * y + R[1, 2]
    dz = R[2, 0] * x + R[2, 1] * y + R[2, 2]
    zf = dx * mc.fwd[0] + dz * mc.fwd[2]
    mx = (mc.cx + mc.f * (dx * mc.right[0] + dz * mc.right[2]) / zf - 0.5).astype(np.float32)
    my = (mc.cy + mc.shift - mc.f * dy / zf - 0.5).astype(np.float32)
    img = cv2.remap(out, mx, my, cv2.INTER_LINEAR if k < 1.0 else cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE)
    img = np.maximum(img, 0.0)
    skym = np.clip(cv2.remap(sky, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0.0, 1.0)
    starm = np.clip(cv2.remap(starm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0.0, 1.0)
    BK._splat_stars(img, cam, Wd, starm, t)
    ramp_g = smoothstep(BK.ROAR + 4, BK.ROAR + 58, f)
    md = np.asarray(WD.MOON_DIR, np.float64)
    md = md / np.linalg.norm(md)
    cth = np.maximum((dx * md[0] + dy * md[1] + dz * md[2]) / np.sqrt(dx * dx + dy * dy + dz * dz), 0.0)
    csk = BK.V3_CLOSE_SKY if BK.V3_H5 else BK.CLOSE_SKY
    cs = csk[0] * (1.0 + csk[1] * cth ** 4)
    sg = (0.25 + 0.75 * reveal) * (cs + (1.0 - cs) * ramp_g)
    g = (skym * sg + (1.0 - skym) * reveal) / 1.05
    return (img * g[..., None]).astype(np.float32)


# ------------------------------------------------------------------ the shot ---
_SHOT = None


def shot():
    """H1's FirstBeacon with the take extended to src SRC1 and our two hooks."""
    global _SHOT
    if _SHOT is None:
        BK.camera = camera
        BK.world_layer = world_layer
        # extend the take's simulations; the spindrift grains keep the take's own random draw
        BK.F1 = TAKE_END
        ref = BK.FirstBeacon()
        ref.simulate()
        snow, snow_ph = ref.snow, ref.snow_ph
        BK.F1 = SRC1 + 2
        s = BK.FirstBeacon()
        s.simulate()
        s.snow, s.snow_ph = snow, snow_ph
        _SHOT = s
    return _SHOT


def render(fc, scale=1.0):
    return shot().render(fc - SRC_OFF, scale)


def probe():
    for fc in (CUT0 - 1, CUT0, CUT0 + 40, CUT0 + 80, CUT1 - 1):
        cam, focus, u = camera(fc - SRC_OFF, 1.0)
        her = np.array([0.9, 1.7, 0.3])
        feet = np.array([0.9, 0.0, 0.3])
        a = cam.project(her)
        b = cam.project(feet)
        g = cam.pos + np.array([math.sin(math.radians(NA.GLOW_AZ)), 0.0, math.cos(math.radians(NA.GLOW_AZ))]) * 1e6
        gx = cam.project(g)
        print('cut %d src %d pos %s yaw %.2f pitch %.2f hfov %.2f | her head (%.0f, %.0f) feet (%.0f, %.0f) %.0f px | '
              'glow az-30 at x %.0f' % (fc, fc - SRC_OFF, np.round(cam.pos, 2), math.degrees(cam.yaw),
                                          math.degrees(cam.pitch), cam.hfov, a[0], a[1], b[0], b[1], b[1] - a[1],
                                          gx[0]), flush=True)


def _work(args):
    frames, scale, out, skip, wid = args
    os.environ['NUMBA_NUM_THREADS'] = os.environ.get('NUMBA_NUM_THREADS', '1')
    import cv2
    cv2.setNumThreads(1)
    done = 0
    for fc in frames:
        path = look.frame_path(out, fc)
        if skip and (os.path.exists(path) or os.path.exists(path[:-4] + '.jpg')):
            continue
        t1 = time.time()
        img = render(fc, scale)
        look.save_png(path, img)
        done += 1
        print('[w%d] reveal_A f%d %.1fs' % (wid, fc, time.time() - t1), flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='')
    ap.add_argument('--range', default='')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'renders', 'reveal_A'))
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--build-fires', action='store_true')
    a = ap.parse_args()
    if a.probe:
        probe()
        return
    if a.build_fires:
        build_fires()
        return
    frames = []
    if a.range:
        x0, x1 = a.range.split('-')
        frames = list(range(int(x0), int(x1) + 1))
    if a.frames:
        frames += [int(v) for v in a.frames.split(',') if v]
    frames = [fc for fc in frames if CUT0 <= fc < CUT1]
    if not frames:
        raise SystemExit('no frames in %d-%d' % (CUT0, CUT1 - 1))
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(FIRES_NPY):
        raise SystemExit('missing %s (run --build-fires and commit it)' % FIRES_NPY)
    if a.procs <= 1:
        _work((frames, a.scale, out, a.skip, 0))
        return
    import multiprocessing as mp
    chunks = [frames[i::a.procs] for i in range(a.procs)]
    with mp.get_context('fork').Pool(a.procs) as pool:
        pool.map(_work, [(c, a.scale, out, a.skip, i) for i, c in enumerate(chunks) if c])


if __name__ == '__main__':
    main()
