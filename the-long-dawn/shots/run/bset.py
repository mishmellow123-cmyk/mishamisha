"""B's summit set on B's own landform (bworld): the shelf points, the vigil's locked camera, the NE path, and the
puppets that live on it (H5 s1, s6): the keeper in a hood and a red woven-wool shawl (old: a stoop and a staff,
never front-lit, 60-120 px in the locked frame), the child, the travellers; the counting cairn as a heap of visible,
irregular stones with a ragged top; the beacon's aged fire-basket (rust, soot, uneven bars) on its dry-stone base.
Used by vigil.py (bars 18-48) and handback_b.py (bars 49-65); the reveal and the climb share the same set.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bworld as BW         # noqa: E402
import rcam as RC           # noqa: E402
from mt import figure as FG  # noqa: E402

CR = BW.CR_B
TOP = np.array([BW.TX, BW.TOP_Y, BW.TZ])

# the vigil's locked camera (= the hand-back crane's first frame): WSW of the top, 6 m above it, looking ENE over
# the cloud sea and its rock islands (RUN-B2, 27 Sep; it was SSW looking NNE: half the frame a bare snow dome).
# The SET axes YAW / FWD / RIGHT stay frozen at 31 deg: shelf_pt() builds the basket, her places and the cairn from
# them, and the hand-back's staging depends on those.
CAM_BEAR = 62.0                   # bearing camera -> top (deg)
CAM_DIST = 55.0
CAM_UP = 6.0
CAM_YAW = 70.0
PITCH = -3.0
HFOV = 36.0
YAW = 31.0                        # the set axes (frozen; not the camera's yaw)


def dirxz(az):
    a = math.radians(az)
    return np.array([math.sin(a), 0.0, math.cos(a)])


CAM_POS = TOP - CAM_DIST * dirxz(CAM_BEAR) + np.array([0.0, CAM_UP, 0.0])
FWD = dirxz(YAW)
RIGHT = np.array([FWD[2], 0.0, -FWD[0]])


def ground(x, z):
    return BW.ground(x, z, CR, 0.02)


def on_ground(p):
    return np.array([p[0], ground(p[0], p[2]), p[2]])


def shelf_pt(r, f):
    """A point on the summit: r metres screen-right and f metres away from the camera, from the top."""
    return on_ground(TOP + RIGHT * r + FWD * f)


BEACON = shelf_pt(0.9, 0.6)          # the fire-basket on its low dry-stone base
CAIRN = None                         # set below: 3.2 m N of her seat (clear of her in both the vigil and the hand-back)
KNEEL = shelf_pt(0.05, 0.30)         # where she kneels to strike / feed
STAND = shelf_pt(-0.60, 0.15)        # where she keeps the watch (between the fire and the cairn)
SEAT = on_ground(STAND + RIGHT * 0.25)
CAIRN = on_ground(SEAT + 5.5 * dirxz(20.0) + 1.0 * dirxz(116.0))   # the counting cairn: one stone a longest night
LIP = shelf_pt(3.8, 3.0)             # the NE lip: where the path leaves the summit


def camera(W=1920, H=804):
    return RC.RCam(CAM_POS, CAM_YAW, PITCH, 0.0, HFOV, W, H)


# ------------------------------------------------------------------ the NE path ---
_PATH = None


def path_points(n=160):
    """The way up and down: from the lip along the NE ridge's crest (world points, 0 = the lip)."""
    global _PATH
    if _PATH is None:
        pts = [LIP]
        a0 = BW.polar(BW.RIDGE_AZ, 30.0)
        a1 = BW.polar(BW.RIDGE_AZ + 4.0, 1250.0)
        for u in np.linspace(0.0, 1.0, n)[1:]:
            x = a0[0] + (a1[0] - a0[0]) * u
            z = a0[1] + (a1[1] - a0[1]) * u
            # settle onto the crest: the highest point across the ridge within +-25 m
            best = None
            ax = np.array([a1[0] - a0[0], a1[1] - a0[1]])
            ax /= np.linalg.norm(ax)
            nx, nz = ax[1], -ax[0]
            for o in np.linspace(-25.0, 25.0, 11):
                h = ground(x + nx * o, z + nz * o)
                if best is None or h > best[0]:
                    best = (h, x + nx * o, z + nz * o)
            pts.append(np.array([best[1], best[0], best[2]]))
        P = np.array(pts)
        # arc length
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        _PATH = (P, np.r_[0.0, np.cumsum(seg)])
    return _PATH


def path_at(s):
    """World point s metres down the path from the lip."""
    P, S = path_points()
    s = min(max(s, 0.0), S[-1])
    k = int(np.searchsorted(S, s) - 1)
    k = min(max(k, 0), len(S) - 2)
    u = (s - S[k]) / max(S[k + 1] - S[k], 1e-6)
    return P[k] + (P[k + 1] - P[k]) * u


# ------------------------------------------------------------------ materials ---
M = FG.MATS.copy()
M = np.vstack([M,
               [0.30, 0.020, 0.016, 1.2, 0.0, 8.0, 0.16, 0, 0, 0, 0],       # 13 the red woven-wool shawl
               [0.046, 0.042, 0.040, 1.2, 0.0, 8.0, 0.12, 0, 0, 0, 0],      # 14 grey-brown wool (cloak, hood)
               [0.20, 0.20, 0.21, 1.4, 0.0, 8.0, 0.05, 0, 0, 0, 0],         # 15 white hair
               [0.14, 0.135, 0.13, 1.1, 0.2, 10.0, 0.05, 0, 0, 0, 0],       # 16 weathered granite
               [0.10, 0.098, 0.092, 1.0, 0.1, 10.0, 0.05, 0, 0, 0, 0],      # 17 darker granite
               [0.060, 0.034, 0.022, 0.8, 0.5, 14.0, 0.02, 0, 0, 0, 0],     # 18 rusted, sooted iron
               [0.022, 0.018, 0.016, 0.9, 0.2, 10.0, 0.10, 0, 0, 0, 0],     # 19 soft leather (gloves, boots)
               [0.24, 0.018, 0.014, 1.2, 0.0, 8.0, 0.16, 0, 0, 0, 0],       # 20 the shawl's darker weave
               [0.020, 0.019, 0.018, 1.0, 0.0, 8.0, 0.12, 0, 0, 0, 0]])     # 21 a cloak's fold shadow (RUN-B2)


def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ------------------------------------------------------------------ people ---
def person(pose='stand', age=0.85, shawl=True, staff=True, torch=False, child=False, walk=0.0, wind=1.0,
           reach=0.0, face=0.0, cloak=14):
    """A hooded, cloaked figure seen from behind / three-quarter (2-D SDF puppet, metres, feet at origin, x = screen
    right). The hood peaks at the back and flows into the shoulders (a cowl, never a round cap); the red shawl is a
    broad wrap of woven wool over the shoulders and back, its fringed ends hanging at the front edge.
    pose: stand | walk | kneel | feed | sit | look | shield | reach ; age 0..1 (stoop, shorter step, the staff).
    reach 0..1 (pose 'reach'): the right arm goes out and down to someone beside her (the hand-off).
    Returns (Drawing, pts) with pts['hands'], pts['head'], pts['torch']."""
    d = FG.Drawing()
    d.new_group()
    k = 0.035
    sc = 0.60 if child else 1.0
    stoop = 0.30 * age ** 1.3
    if pose in ('kneel', 'feed'):
        hip = np.array([0.02, 0.50])
        lean = 0.30 + stoop + (0.25 if pose == 'feed' else 0.0)
    elif pose == 'sit':
        hip = np.array([0.0, 0.34])
        lean = 0.14 + stoop
    else:
        hip = np.array([0.0, 0.92 - 0.06 * age])
        lean = 0.05 + stoop + (0.08 if pose == 'walk' else 0.0)
    hip = hip * sc
    up = _rot(np.array([0.0, 1.0]), -lean)
    side = _rot(np.array([1.0, 0.0]), -lean)
    chest = hip + up * 0.42 * sc
    neck = hip + up * 0.58 * sc
    # legs (mostly hidden by the cloak)
    if pose in ('kneel', 'feed'):
        d.capsule(hip, (0.26, 0.10), 0.09, 0.07, k=k)
        d.capsule((0.26, 0.10), (-0.10, 0.06), 0.07, 0.055, k=k)
        d.capsule(hip + np.array([-0.05, 0]), (-0.18, 0.05), 0.09, 0.06, k=k)
    elif pose == 'sit':
        d.capsule(hip, np.array([0.30, 0.30]) * sc, 0.09 * sc, 0.075 * sc, k=k)
        d.capsule(np.array([0.30, 0.30]) * sc, np.array([0.34, 0.04]) * sc, 0.075 * sc, 0.06 * sc, k=k)
    else:
        ph = walk * 2 * math.pi
        for sgn in (-1, 1):
            sw = 0.15 * math.sin(ph + (0 if sgn > 0 else math.pi)) * (1.0 if pose == 'walk' else 0.0) * (1 - 0.5 * age)
            a = hip + np.array([0.05 * sgn * sc, 0.0])
            ft = np.array([0.07 * sgn * sc + sw * sc, 0.03])
            kn = 0.5 * (a + ft) + np.array([0.03 * sc, 0.0])
            d.capsule(a, kn, 0.085 * sc, 0.068 * sc, k=k, mat=cloak)
            d.capsule(kn, ft, 0.068 * sc, 0.055 * sc, k=k, mat=19)
    # the cloak: long, heavy, flared a little downwind at the hem
    long_ = pose in ('stand', 'walk', 'look', 'shield', 'reach')
    hem = hip + np.array([0.0, (-0.74 if pose != 'walk' else -0.66) * sc if long_ else -0.26 * sc])
    hem_w = 0.31 * sc + (0.12 if pose == 'shield' else 0.0)
    d.trap(hem + np.array([0.05 * wind, 0.0]), chest + up * 0.06 * sc, hem_w, 0.20 * sc, rnd=0.03, k=0.06,
           mat=cloak, fuzz=0.012, ff=12.0)
    d.ellipse(chest + up * 0.06 * sc, 0.215 * sc, 0.13 * sc, ang=-lean, k=0.06, mat=cloak)
    # arms
    shL = chest + up * 0.06 * sc - side * 0.16 * sc
    shR = chest + up * 0.06 * sc + side * 0.16 * sc
    hands = {}
    if pose == 'shield':          # arms and cloak spread round the fire (to screen-right)
        for sh, sg in ((shL, -1), (shR, 1)):
            e = sh + np.array([0.28 * sg + 0.1, -0.05]) * sc
            w = e + np.array([0.22 * sg + 0.12, -0.12]) * sc
            d.capsule(sh, e, 0.07 * sc, 0.06 * sc, k=0.04, mat=cloak)
            d.capsule(e, w, 0.06 * sc, 0.05 * sc, k=0.04, mat=19)
        d.tri(shL, shR + np.array([0.55, -0.35]) * sc, hem + np.array([0.45, 0.1]) * sc, rnd=0.02, k=0.05, mat=cloak)
    elif pose in ('kneel', 'feed') and reach <= 0.0:
        e = shR + np.array([0.22, -0.18]) * sc
        w = e + np.array([0.20, -0.16 - (0.06 if pose == 'feed' else 0.0)]) * sc
        d.capsule(shR, e, 0.07 * sc, 0.06 * sc, k=0.04, mat=cloak)
        d.capsule(e, w, 0.06 * sc, 0.05 * sc, k=0.04, mat=19)
        hands['R'] = w
        e2 = shL + np.array([0.10, -0.25]) * sc
        w2 = e2 + np.array([0.18, -0.10]) * sc
        d.capsule(shL, e2, 0.07 * sc, 0.06 * sc, k=0.04, mat=cloak)
        d.capsule(e2, w2, 0.06 * sc, 0.05 * sc, k=0.04, mat=19)
        hands['L'] = w2
    else:
        for sh, sg in ((shL, -1), (shR, 1)):
            if (pose == 'reach' or reach > 0.0) and sg > 0:
                # the hand-off: the arm goes out and down to the one beside her, the gloved hand turned over
                e = sh + np.array([0.18 + 0.08 * reach, -0.20 - 0.06 * reach]) * sc
                w = e + np.array([0.14 + 0.10 * reach, -0.14 - 0.04 * reach]) * sc
            else:
                e = sh + np.array([0.04 * sg, -0.28]) * sc
                w = e + np.array([0.03 * sg, -0.24 + (0.08 if (torch and sg > 0) else 0.0)]) * sc
            d.capsule(sh, e, 0.07 * sc, 0.06 * sc, k=0.04, mat=cloak)
            d.capsule(e, w, 0.06 * sc, 0.05 * sc, k=0.04, mat=cloak)
            d.ellipse(w + np.array([0.01 * sg, -0.03]) * sc, 0.042 * sc, 0.05 * sc, k=0.02, mat=19)
            hands['L' if sg < 0 else 'R'] = w
    if staff and not child:
        hL = hands.get('L', shL + np.array([-0.1, -0.5]))
        tip = np.array([hL[0] - 0.10, 0.0])
        top = hL + np.array([0.02, 0.26])
        d.capsule(top, tip, 0.020, 0.017, mat=4)
    if torch:
        hR = hands.get('R', shR + np.array([0.1, -0.5]))
        th = hR + np.array([0.10, 0.45]) * sc
        d.capsule(hR - np.array([0.02, 0.06]), th, 0.02, 0.022, mat=4)
        hands['torch'] = th
    # the hood: a cowl peaking at the back and flowing into the shoulders (never a round cap)
    hc = neck + up * 0.14 * sc + side * 0.02 * face
    d.ellipse(hc, 0.118 * sc, 0.132 * sc, ang=-lean, k=0.05, mat=cloak)
    d.tri(hc + up * 0.06 * sc - side * 0.09 * sc, hc + up * 0.06 * sc + side * 0.09 * sc,
          hc + up * 0.19 * sc - side * 0.03 * sc, rnd=0.03, k=0.05, mat=cloak)
    d.tri(neck - side * 0.20 * sc, neck + side * 0.20 * sc, hc + up * 0.02 * sc, rnd=0.03, k=0.06, mat=cloak)
    if age > 0.7 and not child:
        d.ellipse(hc + side * (0.10 * face + 0.08) - up * 0.02, 0.035, 0.05, mat=15)     # white hair at the hood's edge
    if shawl:
        d.new_group()
        # the shawl: a broad woven wrap over the shoulders and down the back, its ends at the front edge
        top_ = neck + up * 0.02 * sc
        bot = chest - up * 0.16 * sc
        d.trap(bot + np.array([0.02 * wind, 0.0]), top_, 0.27 * sc, 0.21 * sc, rnd=0.03, k=0.03, mat=13,
               fuzz=0.008, ff=30.0)
        # the weave: two darker bands across the back
        for q in (0.35, 0.62):
            p = bot + (top_ - bot) * q
            d.capsule(p - side * 0.24 * sc, p + side * 0.24 * sc, 0.014 * sc, 0.014 * sc, mat=20)
        # a fringed end hanging at her right front edge, lifting a little in the wind
        e0 = bot + side * 0.22 * sc
        d.chain(np.array([e0, e0 + np.array([0.04 * wind, -0.10]) * sc, e0 + np.array([0.07 * wind, -0.20]) * sc]),
                0.035 * sc, 0.022 * sc, k=0.02, mat=13)
    return d, dict(hands=hands, head=hc, torch=hands.get('torch'))


def person2(pose='stand', age=0.85, shawl=True, staff=True, torch=False, child=False, walk=0.0, wind=1.0,
            reach=0.0, build=0.5, hood=0, pack=0, coat=1.0, cloak=14, lean_to=1.0, arms=True, weary=None):
    """Style 2 (RUN-B2): a person seen from behind at 40-120 px. A ROUNDED wool cowl that falls into the shoulders
    (never a pointed tip: a row of pointed hoods read as Nazguls), a stoop that lowers the head and rounds the back
    (never a sideways tilt), the red woven shawl draped as a triangle down the back from shoulder to shoulder with
    two darker woven chevrons and a ragged fringe (never a red box), a heavy cloak to mid-calf (travellers: coat to
    the knee, legs walking), gloves, a staff taller than her. Travellers vary: build 0..1 (slight..broad), hood 0 =
    cowl up / 1 = cowl down (a round head over a wool wrap) / 2 = a deep cowl with a fur edge, pack 0 none / 1 a
    corded load. lean_to: +1 the work (fire, child) is screen-right, -1 screen-left. arms=False leaves the arms
    out (another lane attaches its own at pts['sh_L'] / pts['sh_R']). weary 0..1 (default: the sit pose's all-night
    hunch): the head bowed, the back rounded.
    pose: stand | look | walk | kneel | feed | sit | shield | reach.  Returns (Drawing, pts)."""
    d = FG.Drawing()
    d.new_group()
    k = 0.03
    sc = 0.60 if child else (0.95 + 0.10 * build)
    ws = (1.12 if child else 1.0) * (0.90 + 0.28 * build)
    stoop = 0.0 if child else 0.30 * age ** 1.3
    if weary is None:
        weary = 0.8 if (pose == 'sit' and not child) else 0.0
    stoop = min(stoop + 0.25 * weary, 0.6)
    s = lean_to
    kneel = pose in ('kneel', 'feed', 'shield')
    sit = pose == 'sit'
    ph = walk * 2 * math.pi
    wk = 1.0 if pose == 'walk' else 0.0
    # body landmarks (metres, feet at the origin)
    if kneel:
        hip = np.array([0.0, 0.46])
        tilt = s * (0.30 + (0.18 if pose == 'feed' else 0.0) + (0.10 if pose == 'shield' else 0.0))
    elif sit:
        hip = np.array([0.0, 0.30])
        tilt = s * 0.06
    else:
        hip = np.array([0.0, 0.90])
        tilt = s * 0.03 * wk
    hip = hip * sc
    up = _rot(np.array([0.0, 1.0]), -tilt)
    side = _rot(np.array([1.0, 0.0]), -tilt)
    torso = 0.44 * sc * (1.0 - 0.10 * stoop)
    sho = hip + up * torso                                  # shoulder line centre
    shw = 0.20 * ws * sc / 0.95                               # half shoulder width
    head = sho + up * (0.17 * sc - 0.10 * stoop - 0.05 * weary) + side * (0.02 * s * stoop)
    # legs and boots
    if kneel:
        kn = hip + np.array([0.18 * s, -0.40 * sc])
        d.capsule(hip + side * 0.06 * s, kn, 0.085 * sc, 0.07 * sc, k=k, mat=cloak)
        d.capsule(kn, kn + np.array([-0.30 * s, -0.02]), 0.065 * sc, 0.05 * sc, k=k, mat=19)
        d.capsule(hip - side * 0.06 * s, np.array([-0.16 * s, 0.05]), 0.09 * sc, 0.06 * sc, k=k, mat=cloak)
    elif sit:
        d.capsule(hip + np.array([0.05, 0.0]), np.array([0.26, 0.10]) * sc, 0.09 * sc, 0.075 * sc, k=k, mat=cloak)
        d.capsule(hip - np.array([0.05, 0.0]), np.array([-0.24, 0.08]) * sc, 0.09 * sc, 0.075 * sc, k=k,
                  mat=cloak)
    else:
        for sg in (-1, 1):
            sw = 0.16 * math.sin(ph + (0.0 if sg > 0 else math.pi)) * wk * (1.0 - 0.5 * age)
            lift = 0.05 * max(0.0, math.sin(ph + (0.0 if sg > 0 else math.pi) + 1.2)) * wk
            a = hip + np.array([0.06 * sg * ws, 0.0])
            ft = np.array([0.08 * sg * ws + sw * sc, 0.03 + lift])
            kn_ = 0.5 * (a + ft) + np.array([0.0, 0.02])
            d.capsule(a, kn_, 0.085 * sc * ws, 0.065 * sc, k=k, mat=cloak)
            d.capsule(kn_, ft, 0.062 * sc, 0.050 * sc, k=k, mat=19)
            d.ellipse(ft + np.array([0.0, -0.005]), 0.055 * sc, 0.035 * sc, k=0.01, mat=19)
    # the cloak / coat
    if kneel or sit:
        hem_y = 0.04
        d.trap(np.array([0.10 * s * (1 if kneel else 0), hem_y]), sho + up * 0.02, 0.36 * ws * sc, shw * 0.95,
               rnd=0.05, k=0.06, mat=cloak, fuzz=0.010, ff=14.0)
    else:
        hem_y = hip[1] - (0.62 if coat >= 1.0 else 0.40) * sc
        flare = 0.05 * wind + 0.03 * math.sin(ph) * wk
        hw = (0.27 if coat >= 1.0 else 0.23) * ws * sc
        d.trap(np.array([flare, hem_y]), sho + up * 0.01, hw, shw * 0.96, rnd=0.04, k=0.06, mat=cloak, fuzz=0.010,
               ff=14.0)
        # the weight of wet wool: fuller over the hips, the hem heavy and a little uneven
        d.ellipse(np.array([flare * 0.6, hem_y + 0.34 * sc]), hw * 0.98, 0.30 * sc, k=0.08, mat=cloak)
        d.ellipse(np.array([flare, hem_y + 0.03]), hw * 1.02, 0.05 * sc, k=0.05, mat=cloak, fuzz=0.012, ff=10.0)
    # rounded shoulders and, with age, a rounded back
    d.ellipse(sho - up * 0.02, shw * 1.05, 0.085 * sc, ang=-tilt, k=0.07, mat=cloak)
    if stoop > 0.05:
        d.ellipse(sho - up * 0.07 * sc + up * 0.02, shw * 0.92, (0.10 + 0.10 * stoop) * sc, ang=-tilt, k=0.08,
                  mat=cloak)
    # arms
    hands = {}
    shL = sho - side * shw * 0.92
    shR = sho + side * shw * 0.92
    for sh, sg in (((shL, -1), (shR, 1)) if arms else ()):
        work = (sg == s)
        if pose == 'shield' and work:                      # the arm out round the fire, holding the cloak
            e = sh + np.array([0.26 * s, -0.04]) * sc
            w = e + np.array([0.22 * s, -0.16]) * sc
        elif pose == 'shield':
            e = sh + np.array([0.10 * s, -0.22]) * sc
            w = e + np.array([0.18 * s, -0.08]) * sc
        elif kneel and work:
            e = sh + np.array([0.20 * s, -0.20]) * sc
            w = e + np.array([0.20 * s, -0.12 - (0.08 if pose == 'feed' else 0.0)]) * sc
        elif kneel:
            e = sh + np.array([0.06 * s, -0.24]) * sc
            w = e + np.array([0.16 * s, -0.10]) * sc
        elif (pose == 'reach' or reach > 0.0) and work:
            e = sh + np.array([(0.16 + 0.08 * reach) * s, -0.20 - 0.05 * reach]) * sc
            w = e + np.array([(0.14 + 0.10 * reach) * s, -0.13 - 0.04 * reach]) * sc
        elif staff and not child and sg == -s:
            e = sh + np.array([-0.06 * s, -0.24]) * sc
            w = e + np.array([-0.05 * s, -0.08]) * sc
        elif torch and sg == s:
            e = sh + np.array([0.07 * s, -0.22]) * sc
            w = e + np.array([0.05 * s, 0.02]) * sc
        else:
            sw = -0.06 * math.sin(ph + (0.0 if sg > 0 else math.pi)) * wk
            e = sh + np.array([0.03 * sg + sw, -0.27]) * sc
            w = e + np.array([0.01 * sg + sw, -0.23]) * sc
        d.capsule(sh, e, 0.068 * sc, 0.058 * sc, k=0.05, mat=cloak)
        d.capsule(e, w, 0.058 * sc, 0.048 * sc, k=0.04, mat=cloak)
        d.ellipse(w + np.array([0.0, -0.035 * sc]), 0.040 * sc, 0.048 * sc, k=0.02, mat=19)
        hands['L' if sg < 0 else 'R'] = w
        if pose == 'shield' and work:                      # the cloak drawn out from the shoulder to the hand
            d.tri(sh + up * 0.02, w, np.array([w[0] + 0.05 * s, 0.03]), rnd=0.02, k=0.05, mat=cloak, fuzz=0.008,
                  ff=16.0)
            d.tri(sh + up * 0.02, np.array([w[0] + 0.05 * s, 0.03]), np.array([sh[0], 0.03]), rnd=0.02, k=0.05,
                  mat=cloak)
    if staff and not child and arms:
        hs = hands['R' if s < 0 else 'L']
        tip = np.array([hs[0] - 0.05 * s, 0.0])
        top = np.array([hs[0] - 0.01 * s, head[1] + 0.16])
        if kneel or sit:
            top = hs + np.array([-0.30 * s, 0.55])
            tip = hs + np.array([0.25 * s, -0.30])
        d.capsule(top, tip, 0.019, 0.016, mat=4)
    if torch and arms:
        hr = hands['R' if s > 0 else 'L']
        th = hr + np.array([0.08 * s, 0.50]) * sc
        d.capsule(hr - np.array([0.0, 0.05]), th, 0.018, 0.022, mat=4)
        hands['torch'] = th
    # packs, in the body's own silhouette (a separate outlined shape read as a disc): a bundle riding over one
    # shoulder, or a corded load high on the back that shows above the shoulders
    if pack == 1:
        pc = sho + up * 0.04 + side * (-0.12 * s) * sc
        d.ellipse(pc, 0.13 * sc, 0.12 * sc, ang=-tilt + 0.5 * s, k=0.04, mat=4, fuzz=0.010, ff=18.0)
    elif pack == 2:
        pc = sho + up * 0.02 * sc
        d.ellipse(pc, 0.19 * sc, 0.11 * sc, ang=-tilt, k=0.05, mat=0, fuzz=0.012, ff=16.0)
    # the head: a rounded cowl flowing into the shoulders (or cowl down: a round head over a wool wrap)
    d.new_group()
    hr_ = 0.108 * sc * (1.10 if child else 1.0)
    if hood == 1 and not child:
        d.ellipse(sho + up * 0.03, shw * 0.82, 0.075 * sc, ang=-tilt, k=0.05, mat=cloak, fuzz=0.012, ff=22.0)
        d.ellipse(head, 0.092 * sc, 0.110 * sc, ang=-tilt, k=0.05, mat=6, fuzz=0.006, ff=40.0)
    else:
        deep = 1.18 if hood == 2 else 1.0
        d.trap(sho + up * 0.01, head + up * 0.02, shw * 0.80, hr_ * 0.95 * deep, rnd=0.03, k=0.07, mat=cloak)
        d.ellipse(head + up * 0.012, hr_ * 1.06 * deep, hr_ * 1.14 * deep, ang=-tilt, k=0.07, mat=cloak,
                  fuzz=0.012 if hood == 2 else 0.0, ff=30.0)
        if age > 0.7 and not child:
            d.ellipse(head + side * (0.10 * s) - up * 0.035, 0.028, 0.045, mat=15)
    if shawl:
        d.new_group()
        # the red woven shawl: over both shoulders and down over the upper arms, its point low on the back
        Lt = sho - side * shw * 1.12 + up * 0.035
        Rt = sho + side * shw * 1.12 + up * 0.035
        La = sho - side * shw * 1.16 - up * (0.20 + 0.04 * s) * sc
        Ra = sho + side * shw * 1.16 - up * (0.20 - 0.04 * s) * sc
        drift = side * (0.04 * wind * (0.0 if kneel else 1.0))
        tip_ = sho - up * (0.50 * sc) + drift + side * (0.03 * s)
        for tr in ((Lt, Rt, tip_), (Lt, La, tip_), (Rt, Ra, tip_)):
            d.tri(tr[0], tr[1], tr[2], rnd=0.02, k=0.02, mat=13, fuzz=0.006, ff=36.0)
        d.capsule(Lt, Rt, 0.040 * sc, 0.040 * sc, k=0.03, mat=13)
        # one faint woven stripe across the shoulders
        d.capsule(Lt - up * 0.06 * sc + side * 0.02, Rt - up * 0.06 * sc - side * 0.02, 0.010 * sc, 0.010 * sc,
                  mat=20)
        # the fringe: short tassels along the lower edges toward the point
        for i in range(9):
            u = i / 8.0
            p0 = (La + (tip_ - La) * u) if u <= 0.5 else (tip_ + (Ra - tip_) * (u - 0.5) * 2.0)
            if i == 4:
                p0 = tip_
            d.capsule(p0, p0 - up * 0.045 + drift * 0.3, 0.011, 0.007, mat=13)
    return d, dict(hands=hands, head=head, torch=hands.get('torch'), sh_L=shL, sh_R=shR)


def child_asleep(wake=0.0, reach=0.0):
    """The traveller's child from behind, seated against her, wrapped head and shoulders in her red shawl: one small
    hooded lump (the head sunk into the wrap, no neck), asleep with the head bowed toward her; waking, the head rises
    and turns to the sun. reach > 0: a small gloved hand comes out of the wrap toward her, palm up."""
    d = FG.Drawing()
    d.new_group()
    lean = 0.30 * (1.0 - wake)
    # the lap and legs under the wrap (dark), the wrapped body and head as one smooth shape (red)
    d.trap((0.0, 0.0), (0.0, 0.24), 0.23, 0.18, rnd=0.05, k=0.06, mat=0, fuzz=0.01, ff=14.0)
    back = np.array([-0.07 * lean, 0.36])
    head = back + np.array([-0.13 * lean - 0.02 * wake, 0.16 + 0.05 * wake])
    d.new_group()
    d.ellipse(back, 0.175, 0.16, ang=0.35 * lean, k=0.10, mat=13, fuzz=0.008, ff=30.0)
    d.ellipse(head, 0.092, 0.105, k=0.10, mat=13, fuzz=0.008, ff=30.0)
    d.tri(head + np.array([-0.07, 0.03]), head + np.array([0.06, 0.05]), head + np.array([0.01 - 0.03 * lean, 0.15]),
          rnd=0.02, k=0.06, mat=13)
    # the shawl's weave (a darker band) and its fringed end over the small lap
    d.capsule(back + np.array([-0.16, -0.03]), back + np.array([0.16, -0.05]), 0.012, 0.012, mat=20)
    d.chain(np.array([back + np.array([0.13, -0.05]), back + np.array([0.18, -0.15]), back + np.array([0.20, -0.26])]),
            0.032, 0.020, k=0.02, mat=13)
    hand = None
    if reach > 0.0:
        d.new_group()
        sh = back + np.array([-0.12, -0.02])
        hand = sh + np.array([-0.10 - 0.08 * reach, -0.10 + 0.04 * reach])
        d.capsule(sh, hand, 0.035, 0.03, k=0.02, mat=13)
        d.ellipse(hand + np.array([-0.02, 0.0]), 0.032, 0.022, mat=19)
    return d, dict(head=head, hand=hand)


def fire_steel(d, at, ang=0.0, s=1.0, mat=18):
    """Her C-shaped fire-steel (a flat striker bent into a C), drawn into Drawing d at `at` (local metres)."""
    d.new_group()
    pts = []
    for u in np.linspace(-1.0, 1.0, 7):
        a = ang + 1.25 * u * math.pi * 0.5
        pts.append(np.array(at) + s * 0.045 * np.array([math.cos(a + math.pi), math.sin(a + math.pi)]))
    d.chain(np.array(pts), 0.010 * s, 0.010 * s, mat=mat)
    return d


# ------------------------------------------------------------------ the cairn ---
_CAIRN = {}


def rubble_cairn(seed=3, H=1.62, R=1.05):
    """The counting cairn: a heap of field stones of every size (angular, irregular, tilted), broad at the base and
    tapering to a RAGGED top where a few stones stand up on end; dark chinks between them. Never courses, never a
    column, never a smooth outline. Local metres, base centre at the origin."""
    if seed in _CAIRN:
        return _CAIRN[seed]
    rng = np.random.default_rng(seed)
    d = FG.Drawing()
    # the dark core behind the stones (chinks), well inside the outline
    d.new_group()
    d.tri((-R * 0.80, 0.02), (R * 0.80, 0.02), (0.05, H * 0.84), rnd=0.05, mat=7)

    def stone(cx, cy, r, tilt, elong, mat):
        d.new_group()
        nv = 5 + int(rng.random() * 3)
        ang0 = rng.random() * 2 * math.pi
        vs = []
        for i in range(nv):
            a = ang0 + 2 * math.pi * (i + 0.35 * (rng.random() - 0.5)) / nv
            rr = r * (0.72 + 0.4 * rng.random())
            p = np.array([rr * math.cos(a) * elong, rr * math.sin(a)])
            vs.append(_rot(p, tilt) + np.array([cx, cy]))
        c = np.array([cx, cy])
        for i in range(nv):
            d.tri(c, vs[i], vs[(i + 1) % nv], rnd=0.012 * r / 0.2, k=0.01, mat=mat)

    # courses of the heap from the bottom up: big stones low, smaller higher; the profile is a worn cone
    y = 0.0
    while y < H * 0.86:
        u = y / H
        half = R * (1.0 - u) ** 0.85 * (0.94 + 0.12 * rng.random())
        r = 0.23 * (1.0 - 0.45 * u) * (0.8 + 0.4 * rng.random())
        x = -half + r * 0.7
        while x < half - r * 0.5:
            rr = r * (0.7 + 0.6 * rng.random())
            stone(x + rng.normal(0, 0.03), y + rr * 0.75 + rng.normal(0, 0.03), rr, (rng.random() - 0.5) * 0.9,
                  1.0 + 0.5 * rng.random(), 16 if rng.random() < 0.7 else 17)
            x += rr * (1.25 + 0.35 * rng.random())
        y += r * (1.05 + 0.25 * rng.random())
    # the ragged top: two or three stones standing up on end, off-centre, leaning
    for i, (dx, hh) in enumerate(((-0.10, 0.34), (0.14, 0.24), (0.02, 0.16))):
        cy = y + hh * 0.5 - 0.04
        stone(dx + rng.normal(0, 0.04), cy, 0.11 + 0.03 * rng.random(), (rng.random() - 0.5) * 0.5 + math.pi * 0.5,
              1.7 + 0.5 * rng.random(), 16)
        y = max(y, cy + 0.02) if i == 0 else y
    _CAIRN[seed] = d
    return d


def beacon_base(seed=9, height=0.52, base_w=1.02, top_w=0.80, basket_h=0.42, basket_w=0.86):
    """The beacon: a low, rough plinth of field stones (never courses of bricks) and an old fire-basket (rusted,
    sooted, uneven bars, one bent) holding split wood. Returns (back, front, fire_base_y): back = plinth + logs
    (before the flame), front = the bars (after)."""
    rng = np.random.default_rng(seed)
    back = FG.Drawing()
    back.new_group()
    back.trap((0.0, 0.02), (0.0, height - 0.02), base_w * 0.44, top_w * 0.44, rnd=0.03, mat=12)   # the chinks

    def stone(cx, cy, r, tilt, elong, mat):
        back.new_group()
        nv = 5 + int(rng.random() * 3)
        a0_ = rng.random() * 2 * math.pi
        vs = []
        for i in range(nv):
            a = a0_ + 2 * math.pi * (i + 0.35 * (rng.random() - 0.5)) / nv
            rr = r * (0.75 + 0.35 * rng.random())
            vs.append(_rot(np.array([rr * math.cos(a) * elong, rr * math.sin(a)]), tilt) + np.array([cx, cy]))
        c = np.array([cx, cy])
        for i in range(nv):
            back.tri(c, vs[i], vs[(i + 1) % nv], rnd=0.008, k=0.008, mat=mat)
    y = 0.0
    while y < height - 0.06:
        u = y / height
        half = 0.5 * (base_w + (top_w - base_w) * u)
        r = 0.085 + 0.05 * rng.random()
        x = -half + r * 0.8
        while x < half - r * 0.6:
            rr = r * (0.75 + 0.5 * rng.random())
            stone(x + rng.normal(0, 0.012), y + rr * 0.7, rr, (rng.random() - 0.5) * 0.6, 1.2 + 0.6 * rng.random(),
                  16 if rng.random() < 0.6 else 17)
            x += rr * (1.5 + 0.4 * rng.random())
        y += r * (1.2 + 0.3 * rng.random())
    top = height
    fb = top + 0.07
    for i in range(7):
        back.new_group()
        side = -1 if i % 2 else 1
        a = side * (0.35 + 0.35 * rng.random())
        L = basket_w * (0.42 + 0.18 * rng.random())
        cx = (rng.random() - 0.5) * 0.20
        cy = top + 0.08 + 0.04 * i
        dx = math.cos(a) * L / 2
        dy = math.sin(a) * L / 2
        back.capsule((cx - dx, cy - dy), (cx + dx, cy + dy), 0.034, 0.028, mat=8)
    rng = np.random.default_rng(seed + 7)
    front = FG.Drawing()
    r0 = basket_w * 0.24
    r1 = basket_w * 0.52
    yb0 = top - 0.03
    nb = 8
    for i in range(nb):
        u = -1 + 2 * i / (nb - 1) + 0.06 * (rng.random() - 0.5)
        front.new_group()
        hb = basket_h * (0.86 + 0.24 * rng.random())
        bend = 0.05 * (rng.random() - 0.5) + (0.07 if i == 5 else 0.0)
        pa = (u * r0, yb0)
        pm = (u * (r0 + (r1 - r0) * 0.70) + bend, yb0 + hb * 0.55)
        pb = (u * r1 + bend * 1.6, yb0 + hb)
        w0 = 0.011 + 0.005 * rng.random()
        front.capsule(pa, pm, w0, w0 * 0.9, mat=18)
        front.capsule(pm, pb, w0 * 0.9, w0 * 0.8, mat=18)
    # the two rings, not quite level
    for yy, rr, tl in ((yb0 + basket_h * 0.55, r0 + (r1 - r0) * 0.70, 0.03), (yb0 + basket_h * 0.98, r1, -0.04)):
        front.new_group()
        front.capsule((-rr, yy - tl), (rr, yy + tl), 0.010, 0.010, mat=18)
    return back, front, fb


# ------------------------------------------------------------------ light presets (bworld.shade) ---
def _lin(h):
    import pipe as PI
    return PI.CM.lin(h)


def moon_vec(el_deg, az_deg):
    return BW.sun_vec(el_deg, az_deg)


def match_horizon(SN, fogp):
    """The far haze converges to the sky's own horizon colour, so the cloud sea's far edge melts into the sky (a fog
    brighter than the horizon sky left a hard, ragged seam along the horizon). In place; RUN-B2, additive."""
    SN[6:9] = fogp[5:8]
    fogp[9:12] = fogp[5:8]
    fogp[4] = 0.0


def night_params(moon, pix_ang, gain=1.0, east=None, grey=0.0, halo=1.0, stars_sky=1.0, horizon_match=False):
    """Moonlit silver (R2-B's grade, never crushed): LP / SN / amb / fogp for bworld.shade with a moon key whose
    visibility channels are set by the caller (LP[32:35]). grey = 0..1 the east greying (east = its direction)."""
    LP = np.zeros(BW.NLP)
    LP[19] = 2.0
    LP[20] = pix_ang
    LP[9] = 0.92
    LP[10] = 1.5
    LP[12] = 0.0
    LP[22] = 0.05
    LP[23] = 0.95 * gain
    LP[24:27] = moon
    LP[27:30] = _lin('#A7BCE0')
    LP[32] = -1
    LP[33] = -1
    LP[35] = 0.5
    LP[45] = 0.06 * gain
    SN = np.zeros(28)
    SN[0:3] = moon
    SN[3:6] = _lin('#060A1A') * gain
    SN[6:9] = _lin('#27365F') * gain
    SN[9] = 0.05 * halo * gain
    SN[10] = 0.10
    SN[11:14] = _lin('#9FB4DA')
    SN[14] = 0.0
    SN[15] = math.radians(0.6)
    SN[16] = 0.30
    SN[17] = 1.0
    if east is not None and grey > 0.0:
        SN[18:21] = _lin('#7E8CB0') * 0.20
        SN[21] = grey
        SN[22:25] = east
        SN[25] = 0.55
    amb = _lin('#2A3968') * 0.34 * gain
    fogc = _lin('#34466F') * 0.80 * gain
    fogp = np.zeros(16)
    fogp[:8] = [5.5e-5, 1 / 1600.0, 2.0e-4, 1 / 140.0, 0.8, fogc[0], fogc[1], fogc[2]]
    fogp[8] = 6.0
    fogp[9:12] = _lin('#6F86B8') * 0.9 * gain
    fogp[12:15] = moon
    if horizon_match:
        match_horizon(SN, fogp)
    return LP, SN, amb, fogp
