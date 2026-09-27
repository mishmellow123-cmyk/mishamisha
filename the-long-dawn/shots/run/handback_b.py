"""B13 . THE HAND-BACK (R11) on B's own landform: frames 3840-5199 of cut B (bars 49-65), one take.

  3840-4079  THE CRANE (R10 folded in): from the vigil's locked frame the camera rises and arcs round her summit
             (every beacon of the range alight beyond the two small figures, the ranges beyond to the curve of the
             world) and settles on bar 52 behind her, a little above, looking toward the east. Per-frame G-buffer
             (bworld.build with one moon channel) + moonlit shade.
  4080-5199  SETTLED (locked; a slow push-in bars 61-63 so the hand-off reads). The world is marched once
             (bworld.build: G-buffer + each pixel's clearance toward the sun's azimuth + the moon) and relit every
             frame: moonlight until the east greys (bar 53), then the sun's elevation e(f) solved from the bar map.
             First light on bar 54 b1; six far beacons pale on bars 55-60 (each by its own clearance), hers last on
             bar 61 b1; she opens her hands to the sun (61 b3); on bar 63 b1 the sun reaches the child and the red
             shawl and she presses her fire-steel into the sleeping child's open palm; on bar 64 b1 the child wakes,
             holding it, and looks at the sun.
H5: the sun is tamed (small soft disc, no streaks, modest bloom); no poster: the pair sit low and small in the
right third, seen from a little above and behind, the sun off to the left behind the shoulder, the far beacons
burning in the cloud sea (PROTECT). The puppets are RUN-B's silhouettes; the cloud HEROINE lane's hands can
replace them in comp (`--no-figures` renders the plate; cameras.json gives the frame).

  python shots/run/handback_b.py --frames 3840,3960,4080,4300,4800,4980,5100 --scale 0.25
  python shots/run/handback_b.py --build --scale 1.0 --ss 1.5           # cache the settled G-buffer (cloud setup)
  python shots/run/handback_b.py --range 4080-5199 --scale 1.0 --ss 1.5 --out renders/handback_B --skip
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bworld as BW         # noqa: E402
import bset as BS           # noqa: E402
import bfig as BF           # noqa: E402  (B's figure light: sun-side rim only, wool, folds, stones)
import bprops as BP         # noqa: E402  (the shared cairn3 + child3)
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import fire2 as F2          # noqa: E402
from mt import fire as F, figure as FG, sky as SK   # noqa: E402
from mt.noise import smoothstep   # noqa: E402

CM = PI.CM
look = PI.look
lin = CM.lin
CR = BW.CR_B
FPS = 24.0
CACHE = os.path.join(CM.ROOT, 'renders', 'run_b_tests', 'cache')
TESTS = os.path.join(CM.ROOT, 'renders', 'run_b_tests')

F0, F1 = 3840, 5200
F_SETTLE, F_GREY, F_FIRST, F_HER = 4080, 4160, 4240, 4800
PALE = [4320, 4400, 4480, 4560, 4640, 4720]
F_HANDS, F_CHILD_SUN, F_GIVE, F_WAKE = 4840, 4960, 4980, 5040
F_PUSH0, F_PUSH1 = 4800, 5000
# B14 TITLE plate (dawntitle_B, 5200-5439): the settled frame of 5199 lifts into the dawn sky (a rotation only), so
# EDIT's title (y ~360, kindling ~5260, fading into the light 5340-5400) sits over clean sky and high cloud
F_TITLE0, F_TITLE1 = 5200, 5440
TITLE_TILT, TITLE_EASE, TITLE_DRIFT = 6.8, 76, 0.45
MOON_END = BS.moon_vec(12.0, 305.0)         # the setting moon, low in the NW (the vigil's last hour)


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


# ------------------------------------------------------------------ the sun ---
def _shoulder():
    r = CR[3]                                          # the sunrise shoulder's dome row
    x, z = float(r[0]), float(r[1])
    h = BW.ground(x, z, CR, 2.0)
    step = 40.0
    for _ in range(120):
        imp = False
        for dx, dz in ((step, 0), (-step, 0), (0, step), (0, -step)):
            hh = BW.ground(x + dx, z + dz, CR, 2.0)
            if hh > h:
                h, x, z, imp = hh, x + dx, z + dz, True
        if not imp:
            step *= 0.5
            if step < 0.5:
                break
    return np.array([x, h, z])


SHOULDER = _shoulder()
SEAT = BS.SEAT
_d = SHOULDER - SEAT
SUN_BASE_AZ = math.degrees(math.atan2(_d[0], _d[2]))
SUN_AZ = None


def skyline_el(az_deg, eye):
    a = math.radians(az_deg)
    d = np.array([math.sin(a), math.cos(a)])
    best = -10.0
    for r in np.geomspace(20.0, 120000.0, 1100):
        x, z = eye[0] + d[0] * r, eye[2] + d[1] * r
        h = max(BW.ground(x, z, CR, r / 2000.0), float(BW.h_cloud(x, z, r / 2000.0, 0.0))) - r * r / (2 * BW.R_EARTH)
        best = max(best, math.degrees(math.atan2(h - eye[1], r)))
    return best


def _sun_az():
    """The sun comes up behind the shoulder's flank as seen from her: the azimuth where her skyline is ~ -0.1 deg
    (her summit is lit last, just after the disc clears the flank for the camera behind her)."""
    global SUN_AZ
    if SUN_AZ is None:
        p = os.path.join(CACHE, f'hbb_sunaz_{BW.VERSION}.npy')
        if os.path.exists(p):
            SUN_AZ = float(np.load(p))
        else:
            eye = SEAT + np.array([0.0, 1.0, 0.0])
            best = None
            for da in np.linspace(-10.0, 10.0, 81):
                el = skyline_el(SUN_BASE_AZ + da, eye)
                sc = abs(el + 0.52) + 0.004 * abs(da + 3.0)
                if best is None or sc < best[1]:
                    best = (SUN_BASE_AZ + da, sc)
            SUN_AZ = best[0]
            os.makedirs(CACHE, exist_ok=True)
            np.save(p, np.array(SUN_AZ))
    return SUN_AZ


# ------------------------------------------------------------------ cameras ---
SET_BACK, SET_UP, SET_HFOV, SET_TURN = 14.0, 2.6, 44.0, 11.0    # behind her, a little above; the sun left of centre
SET_SEAT_U, SET_SEAT_V = 0.62, 0.74                           # the pair low in the right third
PUSH_HFOV = 30.0                                                # the push-in (a zoom from the locked position)


def _aim(pos, target, hfov, W, H, u, v):
    d = target - pos
    bear = math.degrees(math.atan2(d[0], d[2]))
    el = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
    f = 0.5 * W / math.tan(math.radians(hfov) * 0.5)
    yaw = bear - math.degrees(math.atan((u - 0.5) * W / f))
    pitch = el + math.degrees(math.atan((v - 0.5) * H / f))
    return yaw, pitch


def settled_cam(W, H, frame=F_SETTLE):
    az = _sun_az() + SET_TURN
    pos = SEAT - BS.dirxz(az) * SET_BACK + np.array([0.0, SET_UP, 0.0])
    tgt = SEAT + np.array([0.0, 0.55, 0.0])
    w = _ease((frame - F_PUSH0) / float(F_PUSH1 - F_PUSH0))
    hfov = SET_HFOV + (PUSH_HFOV - SET_HFOV) * w
    u = SET_SEAT_U + (0.60 - SET_SEAT_U) * w
    v = SET_SEAT_V + (0.82 - SET_SEAT_V) * w
    yaw, pitch = _aim(pos, tgt, hfov, W, H, u, v)
    return RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)


def camera(frame, W, H):
    """The crane: from the vigil's locked frame, rising and arcing round her summit to the settled frame."""
    c0 = BS.camera(W, H)
    c1 = settled_cam(W, H)
    if frame >= F_SETTLE:
        return settled_cam(W, H, frame)
    u = _ease((frame - F0) / float(F_SETTLE - F0))

    def polar(p):
        d = p - SEAT
        return math.degrees(math.atan2(d[0], d[2])), math.hypot(d[0], d[2]), p[1] - SEAT[1]
    b0, r0, h0 = polar(c0.pos)
    b1, r1, h1 = polar(c1.pos)
    db = (b1 - b0 + 540.0) % 360.0 - 180.0
    b = b0 + db * u
    s = math.sin(math.pi * u) ** 2
    rr = r0 + (r1 - r0) * u + 40.0 * s                       # pulls back as it rises...
    hh = h0 + (h1 - h0) * u + 20.0 * s                       # ...to see the whole range alight, then settles
    a = math.radians(b)
    pos = SEAT + np.array([rr * math.sin(a), hh, rr * math.cos(a)])
    hfov = c0.hfov_d + (c1.hfov_d - c0.hfov_d) * u + 18.0 * s

    def seat_uv(c):
        x, y, z = c.project(SEAT + np.array([0.0, 0.55, 0.0]))
        return x / c.W, y / c.H
    (u0, v0), (u1, v1) = seat_uv(c0), seat_uv(c1)
    us, vs = u0 + (u1 - u0) * u, v0 + (v1 - v0) * u + 0.14 * s
    yaw, pitch = _aim(pos, SEAT + np.array([0.0, 0.55, 0.0]), hfov, W, H, us, min(vs, 0.90))
    return RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)


# ---------------------------------------------------------------- beacons ---
def beacon_catalogue():
    """Every summit around her (0.8-70 km, all directions) standing well above the cloud: one beacon each."""
    p = os.path.join(CACHE, f'hbb_beacons3_{BW.VERSION}.npy')
    if os.path.exists(p):
        return np.load(p)
    pts = []
    for r in np.geomspace(900.0, 70000.0, 64):
        n = int(min(max(2 * math.pi * r / max(r * 0.06, 350.0), 24), 150))
        for k in range(n):
            a = 2 * math.pi * (k + 0.5 * (int(r) % 2)) / n
            x, z = BS.TOP[0] + r * math.sin(a), BS.TOP[2] + r * math.cos(a)
            fp = max(r / 800.0, 2.0)
            step = max(r * 0.012, 40.0)
            bh = BW.h_rock(x, z, fp, CR)
            for _it in range(16):
                imp = False
                for ddx, ddz in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    h = BW.h_rock(x + ddx, z + ddz, fp, CR)
                    if h > bh:
                        bh, x, z, imp = h, x + ddx, z + ddz, True
                if not imp:
                    step *= 0.5
            rr = math.hypot(x - BS.TOP[0], z - BS.TOP[2])
            if bh > BW.CLOUD_Y + (160.0 if rr < 15000.0 else 330.0) and rr > 600.0:
                pts.append((x, bh, z))
    P = np.array(pts)
    keep = []
    for q in P[np.argsort(np.hypot(P[:, 0] - BS.TOP[0], P[:, 2] - BS.TOP[2]))]:
        dd = math.hypot(q[0] - BS.TOP[0], q[2] - BS.TOP[2])
        if all(math.hypot(q[0] - k[0], q[2] - k[2]) > max(0.07 * dd, 700.0) for k in keep):
            keep.append(q)
    B = np.array(keep)
    os.makedirs(CACHE, exist_ok=True)
    np.save(p, B)
    return B


def beacon_thresholds(B):
    p = os.path.join(CACHE, f'hbb_thresh_{BW.VERSION}_{_sun_az():.3f}.npy')
    if os.path.exists(p):
        return np.load(p)
    a = math.radians(_sun_az())
    P = np.array([BS.TOP[0], BS.TOP[2], 0.0, 0.0])
    th = np.zeros(len(B))
    for k, (x, y, z) in enumerate(B):
        curv = ((x - P[0]) ** 2 + (z - P[1]) ** 2) / (2 * BW.R_EARTH)
        c0 = BW.clearance0_fine(P, CR, x, y + 4.0 - curv, z, math.sin(a), math.cos(a), 2.0, 0.03, 4.0, 120000.0)
        th[k] = math.degrees(math.asin(min(max(-c0, -0.2), 0.2)))
    np.save(p, th)
    return th


def her_threshold():
    a = math.radians(_sun_az())
    P = np.array([BS.TOP[0], BS.TOP[2], 0.0, 0.0])
    b = BS.BEACON + np.array([0.0, 1.4, 0.0])
    c0 = BW.clearance0_fine(P, CR, b[0], b[1], b[2], math.sin(a), math.cos(a), 0.05, 0.01, 2.0, 60000.0)
    return math.degrees(math.asin(min(max(-c0, -0.2), 0.2)))


def villages_world():
    p = os.path.join(CACHE, f'hbb_villages_{BW.VERSION}.npy')
    if os.path.exists(p):
        return np.load(p)
    rng = np.random.default_rng(12)
    V = []
    tries = 0
    while len(V) < 44 and tries < 20000:
        tries += 1
        r = 1800.0 * math.exp(rng.random() * math.log(16000.0 / 1800.0))
        a = rng.random() * 2 * math.pi
        x, z = BS.TOP[0] + r * math.sin(a), BS.TOP[2] + r * math.cos(a)
        if BW.h_rock(x, z, 30.0, CR) > BW.CLOUD_Y - 60.0:
            continue
        if all(math.hypot(x - q[0], z - q[1]) > 1100.0 for q in V):
            V.append((x, z, rng.uniform(280, 560), rng.uniform(0.6, 1.3)))
    V = np.array(V)
    np.save(p, V)
    return V


# ------------------------------------------------------------------ light ---
def dawn_params(e, grey, pix_ang):
    """LP / SD / amb / fog at sun elevation e; grey = 0..1 how far the east has greyed. The sun is TAMED: a small,
    soft disc (never a starburst), a modest aureole."""
    a = math.radians(_sun_az())
    L = np.array([math.cos(math.radians(e)) * math.sin(a), math.sin(math.radians(e)),
                  math.cos(math.radians(e)) * math.cos(a)])
    up = smoothstep(-2.0, 1.0, e)
    LP = np.zeros(BW.NLP)
    LP[0:3] = L
    LP[3] = 5.2
    LP[4:7] = [0.016, 0.042, 0.090]
    LP[7] = 0.012
    LP[8] = 0.0026
    LP[9] = 0.95
    LP[10] = 2.2
    LP[12] = 0.12                     # a softer crust sheen (a strong one read as a lake under the sun)
    gd = np.array([L[0], 0.10, L[2]])
    LP[13:16] = gd / np.linalg.norm(gd)
    LP[16:19] = lin('#FFB98A') * 0.030 * up
    LP[19] = 1.0
    LP[20] = pix_ang
    LP[21] = 0.6
    LP[32] = -1
    LP[33] = -1
    LP[44] = 1.0
    SD = np.zeros(24)
    SD[0:3] = L
    SD[3:6] = np.array([1.0, 0.62, 0.36])
    SD[6] = 22.0 * smoothstep(-0.6, 0.2, e)                  # a small soft disc: warm, never clipped, never a starburst
    SD[7] = math.radians(0.27)
    SD[8:11] = lin('#2B53A0') * (0.20 + 0.40 * up) * grey
    SD[11:14] = lin('#FFC88A') * (0.22 + 0.60 * up) * grey
    SD[14:17] = lin('#A9A6CE') * (0.12 + 0.26 * up) * grey
    SD[17] = 0.55 * up
    SD[18] = 0.035 * up
    SD[19] = 0.034
    SD[20] = 0.55
    amb = lin('#6A86C8') * (0.14 + 0.30 * up) * max(grey, 0.3)
    fc = lin('#6F83B8') * (0.10 + 0.22 * up) * max(grey, 0.3)
    wc = lin('#FFC58A') * 0.7 * up
    fogp = np.zeros(16)
    fogp[:8] = [5.0e-5, 1 / 1500.0, 2.6e-4, 1 / 220.0, 0.6, fc[0], fc[1], fc[2]]    # valley haze: a deeper layer
    fogp[8] = 10.0
    fogp[9:12] = wc
    fogp[12:15] = L
    return LP, SD, amb, fogp


# ------------------------------------------------------------ the hand-back puppets ---
# (built here, not in bset, so the vigil's puppets can change independently; the join matches on the SAME calls)
def sleeve_arm(d, sh, el, wr, sg, hand_ang, palm=1.0, drape=0.11, mat=14):
    """A cloaked arm (local metres): a tapered upper arm and forearm, the wide sleeve hanging under the forearm, and
    a gloved hand (a mitten, its thumb out toward the body's centre line). palm 0 closed .. 1 open. Returns the
    centre of the palm."""
    sh, el, wr = np.asarray(sh, float), np.asarray(el, float), np.asarray(wr, float)
    d.new_group()
    d.capsule(sh, el, 0.080, 0.064, k=0.05, mat=mat, fuzz=0.008, ff=20.0)
    d.capsule(el, wr, 0.060, 0.044, k=0.05, mat=mat, fuzz=0.008, ff=20.0)
    fa = wr - el
    L = float(np.linalg.norm(fa)) + 1e-9
    u = fa / L
    cuff = el + u * L * 0.86
    low = el + u * L * 0.62 + np.array([0.0, -drape])
    d.tri(el + np.array([0.0, -0.04]), cuff + np.array([0.0, -0.02]), low, rnd=0.035, k=0.05, mat=mat)   # the sleeve
    d.ellipse(cuff, 0.052, 0.078, ang=math.atan2(u[1], u[0]), k=0.04, mat=mat)      # the cuff, flared
    d.new_group()
    hu = np.array([math.cos(hand_ang), math.sin(hand_ang)])
    hn = np.array([-hu[1], hu[0]])
    pc = wr + hu * 0.040
    d.ellipse(pc, 0.046, 0.031, ang=hand_ang, k=0.02, mat=19)                     # the palm
    d.ellipse(pc + hu * (0.040 + 0.015 * palm), 0.030, 0.024 - 0.004 * palm, ang=hand_ang, k=0.02, mat=19)
    side = -sg if hn[0] * sg > 0 else sg                                             # thumb toward the centre line
    th0 = wr + hu * 0.022 + hn * 0.022 * side
    d.capsule(th0, th0 + hu * 0.026 + hn * (0.016 + 0.022 * palm) * side, 0.013, 0.010, k=0.01, mat=19)
    return pc


def keeper_hb(hands=0.0, give=0.0, age=0.9, shawl=False, staff=True, wind=0.4):
    """Her, seated: bset.person2('sit', arms=False) (RUN-B2's shared silhouette) plus her own sleeved arms from its
    shoulders. hands 0..1: drawn from her sleeves and opened to the sun (forearms out at chest height, palms open;
    never a V, never hands-up); give 0..1: the right arm goes out and down to the child's open palm with the steel.
    At hands = give = 0 the arms rest in her lap. Returns (Drawing, pts) with pts['steel'] (local metres) or None."""
    kd, kp = BS.person2('sit', age=age, shawl=shawl, staff=staff, wind=wind, arms=False)
    pts = dict(steel=None)
    h = _ease(hands)
    g = _ease(give)
    for sg in (-1, 1):
        sh = np.asarray(kp.get('sh_R' if sg > 0 else 'sh_L', np.array([0.17 * sg, 0.64])), float)
        if sg > 0 and g > 0.0:
            el = sh + np.array([0.11 + 0.04 * g, -0.13 - 0.03 * g])
            wr = el + np.array([0.07 + 0.06 * g, -0.10 - 0.04 * g])
            pc = sleeve_arm(kd, sh, el, wr, sg, math.radians(-30.0 - 15.0 * g), palm=0.3, drape=0.10)
            pts['steel'] = pc + np.array([0.02, -0.03])
        else:
            rest_el = sh + np.array([0.06 * sg, -0.22])
            rest_wr = rest_el + np.array([-0.10 * sg, -0.05])
            open_el = sh + np.array([0.13 * sg, -0.19])
            open_wr = open_el + np.array([0.15 * sg, 0.02])
            el = rest_el + (open_el - rest_el) * h
            wr = rest_wr + (open_wr - rest_wr) * h
            a_out = math.radians(28.0) if sg > 0 else math.radians(152.0)
            a_rest = math.atan2(rest_wr[1] - rest_el[1], rest_wr[0] - rest_el[0])
            ang = a_rest + (a_out - a_rest) * smoothstep(0.2, 0.7, h)
            sleeve_arm(kd, sh, el, wr, sg, ang, palm=h, drape=0.12 + 0.05 * h)
    return kd, pts


def child_pos():
    """Where the child sits against her (world): shared with the vigil's last bars."""
    f = BS.dirxz(_sun_az())
    side = np.array([f[2], 0.0, -f[0]])
    fw = BS.dirxz(_sun_az() + SET_TURN)
    return BS.on_ground(SEAT + side * 0.78 - fw * 0.20)


def child_hb(wake=0.0, reach=0.0, hold=0.0):
    """The traveller's child, seated against her, wrapped head and shoulders in her red shawl, from behind: ONE
    continuous bell of drapery from the covered head to the lap (no neck notch, never a round head on a round body),
    a soft peak where the shawl falls over the crown. Asleep the head is bowed and leans toward her; waking it rises
    and turns to the sun. reach 0..1: a small gloved hand comes out of the wrap toward her, palm up; hold 0..1: the
    hand, closed on the steel, comes back to the chest. Returns (Drawing, pts) with pts['hand'] (palm centre)."""
    d = FG.Drawing()
    lean = 0.32 * (1.0 - _ease(wake))
    d.new_group()
    d.trap((0.02, 0.0), (0.0, 0.20), 0.25, 0.20, rnd=0.05, k=0.06, mat=0, fuzz=0.01, ff=14.0)        # lap, legs
    d.new_group()
    base = np.array([0.0, 0.12])
    tilt = np.array([-0.06 * lean, -0.03 * lean])                                  # asleep: bowed, leaning to her
    d.trap(base, np.array([-0.01, 0.39]) + 0.4 * tilt, 0.25, 0.17, rnd=0.05, k=0.06, mat=13, fuzz=0.008, ff=30.0)
    sho = np.array([-0.01, 0.37]) + 0.5 * tilt
    d.ellipse(sho, 0.165, 0.065, ang=0.25 * lean, k=0.06, mat=13, fuzz=0.008, ff=30.0)              # sloped shoulders
    head = np.array([-0.025 - 0.02 * wake, 0.505 + 0.02 * wake]) + tilt
    d.trap(sho, head + np.array([0.0, -0.02]), 0.13, 0.085, rnd=0.03, k=0.05, mat=13)                 # drape, head->shoulders
    d.ellipse(head, 0.074, 0.084, ang=0.35 * lean, k=0.05, mat=13, fuzz=0.008, ff=30.0)              # the covered head
    d.tri(head + np.array([-0.04, 0.03]), head + np.array([0.05, 0.04]),
          head + np.array([0.03 - 0.04 * lean, 0.10]), rnd=0.03, k=0.06, mat=13)                    # the crown's fold
    d.capsule(base + np.array([-0.20, 0.12]), base + np.array([0.21, 0.10]), 0.011, 0.011, mat=20)    # the weave
    d.chain(np.array([base + np.array([0.18, 0.13]), base + np.array([0.23, 0.03]), base + np.array([0.25, -0.08])]),
            0.030, 0.018, k=0.02, mat=13)                                                            # fringed end
    hand = None
    r = _ease(reach)
    if r > 0.0 or hold > 0.0:
        d.new_group()
        sh = base + np.array([-0.11, 0.19])
        out = sh + np.array([-0.13 - 0.07 * r, -0.07 + 0.03 * r])
        back = sh + np.array([-0.02, 0.02])
        hand = out + (back - out) * _ease(hold)
        d.capsule(sh, hand, 0.034, 0.026, k=0.03, mat=0)                                            # the small sleeve
        d.ellipse(hand + np.array([-0.028, 0.004]), 0.030, 0.019, ang=0.15, k=0.02, mat=19)         # mitten, palm up
    return d, dict(head=head, hand=hand)


# ------------------------------------------------------------------ shot ---
class HandBack:
    def __init__(self, scale=0.25, ss=1.5, figures=True):
        self.scale, self.ss, self.figures = scale, ss, figures
        self.W, self.H = int(round(1920 * scale)), int(round(804 * scale))
        _sun_az()
        B = beacon_catalogue()
        th = beacon_thresholds(B)
        self.B = B
        self.e_her = her_threshold()
        self.th_raw = th
        self.e_first0 = self.e_her - 0.95
        rk = np.argsort(np.argsort(th + 1e-6 * np.arange(len(th))))
        u = rk / max(len(th) - 1, 1)
        self.th = self.e_first0 + (self.e_her - 0.10 - self.e_first0) * u ** 0.9
        self.V = villages_world()
        self._settled = None
        self.stars = SK.make_stars(14000, 101, lum_scale=7.0)
        self.schedule()

    def schedule(self):
        from scipy.interpolate import PchipInterpolator
        tc = settled_cam(self.W, self.H)
        sx, sy, z = tc.project(self.B)
        ok = (z > 0) & (sx > 0.03 * self.W) & (sx < 0.97 * self.W) & (sy > 0) & (sy < self.H * 0.80)
        dist = np.linalg.norm(self.B - tc.pos, axis=1)
        e_first = self.e_first0
        cand = [(self.th[k], k) for k in np.nonzero(ok)[0] if e_first + 0.08 < self.th[k] < self.e_her - 0.06
                and dist[k] < 45000.0]
        targets = np.linspace(e_first + 0.12, self.e_her - 0.10, 6)
        chosen = []
        for tg in targets:
            best = None
            for th, k in cand:
                if k in chosen or (chosen and th <= self.th[chosen[-1]] + 0.02):
                    continue
                score = abs(th - tg) * 10.0 + dist[k] / 30000.0
                if best is None or score < best[0]:
                    best = (score, k)
            if best is not None:
                chosen.append(best[1])
        self.chosen = chosen
        fs = [F_GREY, F_FIRST] + PALE[:len(chosen)] + [F_HER, F1 + 40]
        es = [e_first - 2.2, e_first] + [float(self.th[k]) for k in chosen] + [self.e_her, self.e_her + 0.9]
        self.e_of = PchipInterpolator(np.array(fs, float), np.array(es, float))
        self.e_first = e_first

    def sun_el(self, frame):
        f2 = F1 + 40
        if frame > f2:              # the title plate: the sun keeps rising at the hand-back's last rate (seamless)
            return float(self.e_of(f2)) + float(self.e_of.derivative()(f2)) * (frame - f2)
        return float(self.e_of(max(frame, F_GREY)))

    def title_cam(self, frame):
        t0 = settled_cam(self.W, self.H, F_TITLE0 - 1)
        u = min(max((frame - F_TITLE0) / float(TITLE_EASE), 0.0), 1.0)
        tilt = TITLE_TILT * _ease(u) + TITLE_DRIFT * max(frame - F_TITLE0 - TITLE_EASE, 0) / float(F_TITLE1 - F_TITLE0 - TITLE_EASE)
        return RC.RCam(t0.pos, t0.yaw_d, t0.pitch_d + tilt, 0.0, t0.hfov_d, self.W, self.H)

    def title_src(self):
        """One tall source frame covering the whole lift (a rotation from a fixed position): marched once."""
        if getattr(self, '_title', None) is not None:
            return self._title
        t0 = settled_cam(self.W, self.H, F_TITLE0 - 1)
        hf = t0.hfov_d * 1.08
        Ws = int(round(self.W * self.ss * 1.08))
        fs = 0.5 * Ws / math.tan(math.radians(hf) * 0.5)
        ft = 0.5 * self.W / math.tan(math.radians(t0.hfov_d) * 0.5)
        vt = 2.0 * math.degrees(math.atan(0.5 * self.H / ft))
        span = vt + TITLE_TILT + TITLE_DRIFT + 1.6
        Hs = int(round(2.0 * fs * math.tan(math.radians(span) * 0.5)))
        scam = RC.RCam(t0.pos, t0.yaw_d, t0.pitch_d + 0.5 * (TITLE_TILT + TITLE_DRIFT), 0.0, hf, Ws, Hs)
        P = np.array([scam.pos[0], scam.pos[2], 0.0, 0.0])
        p = os.path.join(CACHE, f'hbt_G_{BW.VERSION}_{self.scale:.3f}_{self.ss:.2f}_{_sun_az():.2f}.npy')
        if os.path.exists(p):
            G = np.load(p)
        else:
            G = BW.build(scam, P, CR, _sun_az(), nsteps=88, tmax=120000.0, snow_bias=0.03, dmax=180000.0,
                         kstep=0.03, moons=[MOON_END], mk=10.0)
            os.makedirs(CACHE, exist_ok=True)
            tmp = p + f'.{os.getpid()}.tmp.npy'
            np.save(tmp, G)
            os.replace(tmp, p)
        self._title = dict(scam=scam, G=G)
        return self._title

    def render_title(self, frame):
        t = frame / FPS
        S = self.title_src()
        scam, G = S['scam'], S['G']
        e = self.sun_el(frame)
        LP, SD, amb, fogp = self.dawn(e, 1.0, scam)
        self.fire_light(LP, self.fire_level(frame), t)
        img = np.zeros((scam.H, scam.W, 3), np.float32)
        BW.shade(G, LP, SD, amb, fogp, float(scam.pos[1]), img)
        img = high_cloud(img, G, scam.pos, LP[0:3])
        dist = G[..., BW.G_DIST].copy()
        zb = dist.copy()
        self.layers(img, zb, dist, scam, frame, t, G)
        mx, my = RC.warp_maps(self.title_cam(frame).scaled(self.ss), scam)
        out = RC.warp(img, mx, my, cv2.INTER_LINEAR)
        return cv2.resize(out, (self.W, self.H), interpolation=cv2.INTER_AREA)

    def settled(self):
        """The settled frame's source camera covers the widest (un-pushed) view; the push-in is a zoom inside it."""
        if self._settled is not None:
            return self._settled
        tc = settled_cam(self.W, self.H)
        fr = PI.Frame(tc, self.ss)
        scam = fr.src
        P = np.array([scam.pos[0], scam.pos[2], 0.0, 0.0])
        p = os.path.join(CACHE, f'hbb_G_{BW.VERSION}_{self.scale:.3f}_{self.ss:.2f}_{_sun_az():.2f}.npy')
        if os.path.exists(p):
            G = np.load(p)
        else:
            G = BW.build(scam, P, CR, _sun_az(), nsteps=88, tmax=120000.0, snow_bias=0.03, dmax=180000.0,
                         kstep=0.03, moons=[MOON_END], mk=10.0)
            os.makedirs(CACHE, exist_ok=True)
            tmp = p + f'.{os.getpid()}.tmp.npy'
            np.save(tmp, G)
            os.replace(tmp, p)
        self._settled = dict(tc=tc, fr=fr, scam=scam, G=G, P=P)
        return self._settled

    def fire_level(self, frame):
        e = self.sun_el(frame) if frame >= F_GREY else -5.0
        her_sun = smoothstep(self.e_her - 0.04, self.e_her + 0.06, e) if frame >= F_GREY else 0.0
        return 1.0 - 0.97 * her_sun

    @staticmethod
    def fire_light(LP, lvl, t):
        """The fire lights the snow round the basket (the vigil's LP[36..43], so the join at 3839/3840 matches)."""
        LP[36] = 9.0 * lvl * F.flicker(t, 3)
        LP[37:40] = BS.BEACON + np.array([0.0, 1.3, 0.0])
        LP[40:43] = np.array(F.FIRE_LIGHT)
        LP[43] = 40.0

    def night(self, G, scam, grey=0.0, frame=None):
        try:
            LP, SN, amb, fogp = BS.night_params(MOON_END, 1.0 / scam.f, east=BS.dirxz(_sun_az()), grey=grey,
                                                horizon_match=True)       # the vigil's horizon (the join)
        except TypeError:
            LP, SN, amb, fogp = BS.night_params(MOON_END, 1.0 / scam.f, east=BS.dirxz(_sun_az()), grey=grey)
        LP[32] = 0
        LP[33] = 0
        if frame is not None:
            self.fire_light(LP, self.fire_level(frame), frame / FPS)
        out = np.zeros((scam.H, scam.W, 3), np.float32)
        BW.shade(G, LP, SN, amb, fogp, float(scam.pos[1]), out)
        return out

    def render(self, frame):
        t = frame / FPS
        if frame >= F_TITLE0:
            return self.render_title(frame)
        if frame < F_SETTLE:
            return self.render_crane(frame)
        S = self.settled()
        scam, fr = S['scam'], S['fr']
        grey = smoothstep(F_GREY, F_FIRST, frame)
        G = S['G']
        img = self.night(G, scam, grey, frame)
        if frame >= F_GREY:
            e = self.sun_el(frame)
            LP, SD, amb, fogp = self.dawn(e, max(grey, 0.02), scam)
            self.fire_light(LP, self.fire_level(frame), t)
            day = np.zeros((scam.H, scam.W, 3), np.float32)
            BW.shade(G, LP, SD, amb, fogp, float(scam.pos[1]), day)
            w = smoothstep(F_GREY, F_GREY + 80, frame)
            img = (img * (1.0 - w) + day * w).astype(np.float32)
        dist = G[..., BW.G_DIST].copy()
        zb = dist.copy()
        self.layers(img, zb, dist, scam, frame, t, G)
        # the push-in: a zoom inside the settled source frame
        tc = settled_cam(self.W, self.H, frame)
        t_ss = tc.scaled(self.ss)
        mx, my = RC.warp_maps(t_ss, scam)
        out = RC.warp(img, mx, my, cv2.INTER_LINEAR)
        return cv2.resize(out, (self.W, self.H), interpolation=cv2.INTER_AREA)

    def dawn(self, e, grey, scam):
        return dawn_params(e, grey, 1.0 / scam.f)

    def render_crane(self, frame):
        t = frame / FPS
        tc = camera(frame, self.W, self.H)
        fr = PI.Frame(tc, self.ss)
        scam = fr.src
        P = np.array([scam.pos[0], scam.pos[2], t, 0.0])
        G = BW.build(scam, P, CR, None, dmax=180000.0, moons=[MOON_END], mk=10.0)
        img = self.night(G, scam, 0.0, frame)
        dist = G[..., BW.G_DIST].copy()
        zb = dist.copy()
        self.layers(img, zb, dist, scam, frame, t, G)
        fr.img, fr.zb, fr.dist = img, zb, dist
        out, _, _ = PI.to_target(fr)
        return out

    def layers(self, img, zb, dist, scam, frame, t, G=None):
        ss = self.ss
        night = 1.0 - smoothstep(F_GREY, F_FIRST + 40, frame)
        e = self.sun_el(frame) if frame >= F_GREY else -5.0
        if night > 0.01 and G is not None:
            import vigil as VG          # lazy: vigil imports this module (the join shares one sky)
            BW.add_band(G, np.eye(3), VG.BAND, getattr(VG, 'BAND_GAIN', 0.05), night, img)
        if night > 0.01:
            mask = (dist > 1e8).astype(np.float32)
            import vigil as VG
            SK.splat_stars(img, scam, self.stars, mask, t=t, gain=ss * ss * night * getattr(VG, 'STAR_GAIN', 1.0),
                           scale=scam.f / (0.5 * 1920 / math.tan(math.radians(22.0))))
        # village glow under the cloud, fading into the day
        import keeper as KP
        vis = night * 0.9 + 0.1
        V = self.V
        fld = np.zeros(dist.shape, np.float32)
        C = scam.params()
        KP._village_field(scam.W, scam.H, C[8], C[9], C[7], scam.pos[0], scam.pos[1], scam.pos[2], C[3], C[4], C[5],
                          C[6], dist, V[:, 0], V[:, 1], V[:, 2], V[:, 3] * vis, fld)
        img += fld[..., None] * (np.array([1.0, 0.55, 0.22], np.float32) * 0.10)[None, None, :]
        # every beacon of the range; each pales as the sun reaches it
        Bc = self.B.copy()
        Bc[:, 1] += 3.0 - ((Bc[:, 0] - scam.pos[0]) ** 2 + (Bc[:, 2] - scam.pos[2]) ** 2) / (2 * BW.R_EARTH)
        sx, sy, z = scam.project(Bc)
        dd = np.linalg.norm(self.B - scam.pos, axis=1)
        for k in range(len(self.B)):
            if z[k] <= 0 or not (-4 <= sx[k] < scam.W + 4 and -4 <= sy[k] < scam.H + 4):
                continue
            ix, iy = int(min(max(sx[k], 0), scam.W - 1)), int(min(max(sy[k], 0), scam.H - 1))
            if dist[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].max() < dd[k] * 0.97:
                continue
            pale = smoothstep(self.th[k] - 0.05, self.th[k] + 0.07, e) if frame >= F_GREY else 0.0
            fl = F.flicker(t + 0.37 * k, k)
            near = min(2500.0 / dd[k], 1.0)
            big = 1.8 if k in self.chosen else 1.0
            en = 7.0 * big * fl * (0.4 + 0.6 * near) * ss * ss * (1.0 - 0.94 * pale)
            F2.glow(img, zb, sx[k], sy[k], 0.8 * ss, en, z=z[k], zbias=z[k] * 0.02, col=np.array([1.0, 0.52, 0.18]))
            F2.halo(img, zb, sx[k], sy[k], 3.0 * ss, 0.026 * big * (1.0 - 0.95 * pale) * (0.5 + 0.5 * near), z=z[k],
                    zbias=z[k] * 0.02, col=np.array([1.0, 0.45, 0.12]))
        # her summit: the cairn, the beacon and its fire, the two figures
        her_sun = smoothstep(self.e_her - 0.04, self.e_her + 0.06, e) if frame >= F_GREY else 0.0
        a = math.radians(_sun_az())
        ee = math.radians(max(e, -1.0))
        L = np.array([math.cos(ee) * math.sin(a), math.sin(ee), math.cos(ee) * math.cos(a)])
        fire_lvl = 1.0 - 0.97 * her_sun
        fire_I = 1.8 * fire_lvl * F.flicker(t, 3)
        amb = lin('#27335E') * 0.45 * night + lin('#6A86C8') * 0.14 * (1.0 - night)
        lights = [dict(dir=L, col=np.array([1.0, 0.62, 0.36]), I=2.2 * her_sun),
                  dict(pos=BS.BEACON + np.array([0, 1.3, 0]), col=F.FIRE_LIGHT, I=fire_I, r0=0.5),
                  dict(dir=MOON_END, col=lin('#A7BCE0'), I=0.45 * night)]
        BF.render(img, zb, scam, BP.cairn3(), BS.CAIRN, lights, amb=amb * 2.6, t=t, write_depth=True,   # snow = ground snow
                  zbias=0.3)
        back, front, fb = BS.beacon_base()
        BF.render(img, zb, scam, back, BS.BEACON, lights, amb=amb, t=t,
                  emissive_gain=0.95 * fire_lvl + 0.05, write_depth=True, zbias=0.3)
        base = BS.BEACON + np.array([0.0, fb, 0.0])
        F2.flame(img, zb, scam, base, 0.85, 0.34, t, seed=4, I=12.0 * (0.04 + 0.96 * fire_lvl), lean=0.2, zbias=0.5,
                 tongues=5, warp=1.2, absorb=0.7 * fire_lvl)      # a dying flame must not leave dark "ears" in the sun
        bx, by, bz = scam.project(base + np.array([0, 0.6, 0]))
        F2.halo(img, zb, bx, by, 5.0 * scam.f / bz, 0.005 * fire_lvl, z=bz, zbias=3.0)
        if her_sun > 0.3:              # in full sun her fire is only a shimmer of heat
            F.shimmer(img, scam, base + np.array([0, 0.9, 0]), 1.6, 0.5, t, amp_px=0.9 * ss * her_sun, seed=3)
        BF.render(img, zb, scam, front, BS.BEACON, lights, amb=amb, t=t, write_depth=False, zbias=0.3)
        if self.figures:
            self.figures_at(img, zb, scam, frame, t, lights, amb)

    def figures_at(self, img, zb, scam, frame, t, lights, amb):
        hands = smoothstep(F_HANDS - 10, F_HANDS + 30, frame) * (1.0 - smoothstep(F_CHILD_SUN - 40, F_CHILD_SUN - 10, frame))
        give = smoothstep(F_CHILD_SUN - 10, F_GIVE, frame) * (1.0 - smoothstep(F_WAKE - 20, F_WAKE + 10, frame))
        wake = smoothstep(F_WAKE - 6, F_WAKE + 40, frame)
        child_sun = smoothstep(F_CHILD_SUN - 20, F_CHILD_SUN + 10, frame)
        cpos = child_pos()
        lk = [dict(lights[0], I=lights[0]['I'] * child_sun)] + lights[1:]
        # the child (bprops.child3, shared with the vigil): asleep against her in her red shawl; the small hand out,
        # palm up, as she gives it; awake, holding the steel at the chest, looking at the sun
        hold = smoothstep(F_WAKE + 4, F_WAKE + 60, frame)
        reach = max(give, 0.8 * wake) * (1.0 - hold) + 0.0 * hold
        cd, cp = BP.child3(wake, reach=reach, hold=hold)
        if frame >= F_GIVE and cp['steel'] is not None:
            BS.fire_steel(cd, cp['steel'], ang=0.8, s=0.85)
        BF.render(img, zb, scam, cd, cpos, lk, amb=amb, t=t, write_depth=False, zbias=0.3)
        # her (bset.person2, RUN-B2's shared silhouette): its own arms at the join; from just before the hands open,
        # her own old hands (sleeve_arm) crossfade in and carry the gesture and the hand-off to the end
        kw = dict(age=0.9, shawl=False, staff=True, wind=0.4)
        w_mine = smoothstep(F_HANDS - 26, F_HANDS - 8, frame)
        steel_local = None
        if w_mine < 1.0:
            kd0, _ = BS.person2('sit', **kw)
            if w_mine <= 0.0:
                BF.render(img, zb, scam, kd0, SEAT, lights, amb=amb, t=t, write_depth=False, zbias=0.3)
            else:
                ia = img.copy()
                BF.render(ia, zb.copy(), scam, kd0, SEAT, lights, amb=amb, t=t, write_depth=False, zbias=0.3)
        if w_mine > 0.0:
            kd, kp = keeper_hb(hands=hands, give=give, **kw)
            if 0.0 < give and frame < F_GIVE and kp['steel'] is not None:
                steel_local = kp['steel']
                BS.fire_steel(kd, steel_local, ang=0.3, s=1.0)
            BF.render(img, zb, scam, kd, SEAT, lights, amb=amb, t=t, write_depth=False, zbias=0.3)
            if w_mine < 1.0:
                img[...] = ia * (1.0 - w_mine) + img * w_mine
        # the steel catches the new sun as it passes into the child's palm: one small warm glint (bar 63)
        gl = math.exp(-((frame - F_GIVE + 4) / 14.0) ** 2) * child_sun
        if gl > 0.01:
            rgt = np.array([scam.right[0], 0.0, scam.right[2]])
            if steel_local is not None:                   # still in her hand
                pw = SEAT + rgt * steel_local[0] + np.array([0.0, steel_local[1], 0.0])
            elif cp['steel'] is not None:                 # in the child's palm
                q = cp['steel']
                pw = cpos + rgt * q[0] + np.array([0.0, q[1], 0.0])
            else:
                pw = SEAT + rgt * 0.45 + np.array([0.0, 0.30, 0.0])
            gx, gy, gz = scam.project(pw)
            F2.glow(img, zb, gx, gy, 0.9 * self.ss, 6.0 * gl * self.ss * self.ss, z=gz, zbias=0.5,
                    col=np.array([1.0, 0.78, 0.45]))
            F2.halo(img, zb, gx, gy, 6.0 * self.ss, 0.012 * gl, z=gz, zbias=0.5, col=np.array([1.0, 0.7, 0.4]))

    def _side(self):
        f = BS.dirxz(_sun_az())
        return np.array([f[2], 0.0, -f[0]])


FINISH = dict(exposure=0.80, bloom_strength=0.05, bloom_threshold=1.8, streak_strength=0.0, vignette_amount=0.22)


def high_cloud(img, G, cam_pos, L):
    """A thin veil of high cirrus in the dawn sky (7.5 km up), lit from below by the low sun: gold toward the sun,
    peach and rose away from it; brighter than the sky behind it. Sky pixels only (the title's clean sky)."""
    from dusk import _fbm_arr
    sky = (G[..., BW.G_FLAG] == 0.0) & (G[..., BW.G_DY] > 0.010)
    jj, ii = np.nonzero(sky)
    if len(jj) == 0:
        return img
    dx, dy, dz = G[..., BW.G_DX][jj, ii], G[..., BW.G_DY][jj, ii], G[..., BW.G_DZ][jj, ii]
    t = (7500.0 - cam_pos[1]) / dy
    x = cam_pos[0] + dx * t
    z = cam_pos[2] + dz * t
    ca, sa = math.cos(math.radians(24.0)), math.sin(math.radians(24.0))
    u = ((x * ca + z * sa) / 11000.0).astype(np.float64)
    v = ((-x * sa + z * ca) / 1700.0).astype(np.float64)
    n = np.zeros(len(u))
    _fbm_arr(u, v, n)
    d = np.clip((n - 0.10) / 0.40, 0.0, 1.0)
    d = d * d * (3.0 - 2.0 * d) * np.clip(1.0 - t / 140000.0, 0.0, 1.0) ** 0.7
    cg = np.clip(dx * L[0] + dy * L[1] + dz * L[2], 0.0, 1.0)
    lift = (1.20 + 1.30 * cg ** 10)[:, None]
    tint = (np.array([1.0, 0.80, 0.62])[None, :] * (0.4 + 0.6 * cg ** 4)[:, None]
            + np.array([1.0, 0.70, 0.72])[None, :] * (0.6 - 0.6 * cg ** 4)[:, None])
    a = (0.42 * d)[:, None]
    px = img[jj, ii, :]
    img[jj, ii, :] = px * (1.0 - a) + px * lift * tint * a
    return img


def finish_at(f):
    """The crane opens in the vigil's grade (the join at 3839/3840) and eases into the hand-back's by the greying."""
    import vigil as VG
    w = _ease((f - F0) / float(F_GREY - F0))
    return {k: VG.FINISH.get(k, v) + (v - VG.FINISH.get(k, v)) * w for k, v in FINISH.items()}


def _work(args):
    frames, scale, ss, out, figs = args
    shot = HandBack(scale, ss, figs)
    for f in frames:
        t0 = time.time()
        img = look.finish(shot.render(f), **finish_at(f))
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.2f}s', flush=True)


def cameras_json(path, W=1920, H=804):
    """The frame for comp (HEROINE's hands): per frame the camera and where the two figures and hands sit."""
    rows = []
    for f in range(F0, F1):
        c = camera(f, W, H)
        rows.append(dict(f=f, pos=[float(v) for v in c.pos], yaw=float(c.yaw_d), pitch=float(c.pitch_d),
                         hfov=float(c.hfov_d), seat=[float(v) for v in c.project(SEAT)[:2]]))
    json.dump(dict(W=W, H=H, seat_world=[float(v) for v in SEAT], sun_az=_sun_az(), frames=rows), open(path, 'w'))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default='hbb')
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args()
    shot = HandBack(a.scale, a.ss, not a.no_figures)          # builds every cache once, before any fork
    if a.build:
        shot.settled()
        print(f'sun az {_sun_az():.2f}; first light e={shot.e_first:.3f}; hers e={shot.e_her:.3f}; '
              f'beacons {len(shot.B)}; chosen {[(int(k), round(float(shot.th[k]), 3)) for k in shot.chosen]}')
        return
    out = os.path.join(CM.ROOT, a.out) if a.out.startswith('renders/') else os.path.join(TESTS, a.out)
    os.makedirs(out, exist_ok=True)
    if a.range:
        s0, s1 = a.range.split('-')
        frames = list(range(int(s0), int(s1) + 1, a.step))
    else:
        frames = [int(x) for x in a.frames.split(',')]
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.find_frame(out, f))]   # find_frame never returns None
    if any(F_SETTLE <= f < F_TITLE0 for f in frames):
        shot.settled()
    if any(f >= F_TITLE0 for f in frames):
        shot.title_src()
    print(f'sun az {_sun_az():.2f}; first light e={shot.e_first:.3f}; hers e={shot.e_her:.3f}; beacons {len(shot.B)}; '
          f'chosen {[(int(k), round(float(shot.th[k]), 3)) for k in shot.chosen]}', flush=True)
    if a.procs <= 1:
        _work((frames, a.scale, a.ss, out, not a.no_figures))
    else:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(frames[i::a.procs], a.scale, a.ss, out, not a.no_figures) for i in range(a.procs)])


if __name__ == '__main__':
    main()
