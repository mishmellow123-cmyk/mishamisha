"""ACCORD v3 scene (cut C, C frames 4480-5679): timeline, cameras, the council, her, the Ring, lights.

Plates (C numbering; bar n starts at (n-1)*80, a beat is 20 frames):
  P1 4480-5119  THE COUNCIL (AC1) + BRING OUT THE RING (AC4): one descent and slow orbit
  P2 5120-5379  THE BEARER (AC2): top-down and close; 5360-5379 is a handle for the cut into the melt
  P3 5520-5679  THE FIRE REMAINS (C23 bar 70); 5600-5679 is a handle under MAP's burn-through
"""
import math
import os
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import look  # noqa: E402

import geom3 as G  # noqa: E402

W, H = look.W, look.H
FOV_H = 50.0
F_FULL = (W / 2) / math.tan(math.radians(FOV_H / 2))

P1 = (4480, 5119)
P2 = (5120, 5379)
P3 = (5520, 5679)
RING_SET = 4840          # 61 b3
GILT_FIND = 5040         # 64 b1
TORCH_DOWN = 5120        # 65 b1
HAND_CLOSE = 5160        # 65 b3
FIRE_CATCH = 5200        # 66 b1
FINGERS_OPEN = 5356
WHITE_END = 5520


def plate(t):
    if t < P2[0]:
        return 1
    if t < P3[0] - 40:
        return 2
    return 3


def lin(h):
    return look.hexrgb(h)


FIRE_HOT = lin(look.PALETTE['fire_hot'])
FIRE_MID = lin(look.PALETTE['fire_mid'])
FIRE_CORE = lin(look.PALETTE['fire_core'])
MOON = lin(look.PALETTE['moonlight'])
NIGHT_MID = lin(look.PALETTE['night_mid'])
SCARF = np.array([0.070, 0.008, 0.010])          # madder-dyed wool: a deep, slightly brown red


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def ramp(t, a, b):
    return clamp01((t - a) / (b - a))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp01(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def ease_out(x, p=3.0):
    x = clamp01(x)
    return 1 - (1 - x) ** p


# ================================================================ the council ===
NEM = 13                     # emissaries; her index is NEM
HER = NEM
NFIG = NEM + 1
R_EM = 1.72                  # the circle
_rng = np.random.default_rng(2026092703)
PSI_HER = math.radians(200.0)                     # her place in the circle (world azimuth)
_slots = PSI_HER + np.radians(360.0 / NFIG * np.arange(1, NFIG)) + np.radians(_rng.uniform(-3.0, 3.0, NEM))
FIG_PSI = np.concatenate([_slots, [PSI_HER]])
FIG_R = np.concatenate([R_EM + _rng.uniform(-0.05, 0.10, NEM), [R_EM - 0.02]])
#                   0     1     2     3     4     5     6     7     8     9    10    11    12   her
FIG_TYPE = np.array([1, 5, 0, 3, 1, 1, 5, 0, 1, 3, 1, 0, 5, 1])
FIG_HS = np.array([1.10, 1.08, 0.98, 1.00, 0.94, 1.04, 1.02, 0.93, 1.06, 0.97, 1.12, 0.96, 1.00, 0.83])
FIG_WS = np.array([1.18, 0.96, 1.05, 1.10, 0.94, 1.02, 1.00, 0.98, 1.04, 1.00, 1.02, 1.06, 0.96, 0.86])
FIG_CAPE = np.array([0.44, 0.00, 0.40, 0.34, 0.00, 0.36, 0.30, 0.38, 0.00, 0.36, 0.40, 0.00, 0.32, 0.30])
FIG_TRAIN = np.array([0.24, 0.16, 0.22, 0.20, 0.14, 0.30, 0.18, 0.26, 0.20, 0.18, 0.28, 0.16, 0.20, 0.12])
FIG_BOW = np.radians([6, 4, 5, 3, 8, 5, 4, 18, 6, 7, 3, 9, 5, 10])
FIG_SIDE = np.array([-1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1], np.float64)
FIG_WEAVE = np.array([1.0, 1.6, 0.7, 1.3, 2.0, 0.8, 1.2, 1.5, 0.75, 1.1, 1.8, 0.9, 1.4, 1.1])
FIG_SHEENK = np.array([1.2, 0.6, 0.9, 1.4, 1.6, 0.7, 1.1, 0.5, 1.0, 1.3, 1.5, 0.8, 1.0, 0.9])
_CL = {'charcoal': (0.0135, 0.0130, 0.0126), 'umber': (0.0200, 0.0142, 0.0102), 'wool': (0.0300, 0.0262, 0.0205),
       'olive': (0.0150, 0.0158, 0.0112), 'bluegrey': (0.0112, 0.0130, 0.0175), 'oxblood': (0.0215, 0.0105, 0.0085),
       'black': (0.0102, 0.0100, 0.0098), 'grey': (0.0170, 0.0168, 0.0164), 'brown': (0.0175, 0.0130, 0.0100),
       'blackgreen': (0.0100, 0.0118, 0.0102)}
FIG_CLOTH = [_CL[k] for k in ('black', 'umber', 'charcoal', 'wool', 'brown', 'charcoal', 'bluegrey', 'grey',
                              'umber', 'umber', 'charcoal', 'brown', 'black', 'charcoal')]
FIG_CLOTH2 = [_CL[k] for k in ('black', 'umber', 'grey', 'brown', 'charcoal', 'umber', 'bluegrey', 'wool',
                               'brown', 'brown', 'black', 'brown', 'charcoal', 'charcoal')]
GILDED = 0                   # the forge that grasped: a broad figure in a heavy black mantle, beside her
FIG_BREATH_PH = _rng.uniform(0, 2 * np.pi, NFIG)
FIG_DOWN_DT = _rng.uniform(-1.5, 1.5, NFIG)       # the torches come down together (tiny human spread)
FIG_DIP_DT = _rng.uniform(-3.0, 3.0, NFIG)        # bar 70: each dips a moment apart
FIG_WALK_V = _rng.uniform(1.05, 1.35, NFIG)
# bar 70: the dips ripple round the circle from the far side (the fire passes round the council), ~5528-5552
_ord = np.argsort(((np.degrees(FIG_PSI[:13]) - 7.0) % 360.0))
P3_DIP = np.zeros(13)
P3_DIP[_ord] = 5528.0 + 1.9 * np.arange(13) + _rng.uniform(-1.2, 1.2, 13)


# ============================================================== the stones ===
NSTONE = 11
R_STONES = 7.5


def stones():
    """Eleven unhewn standing stones (MAP draws the same ring: eleven round a flat one)."""
    rng = np.random.default_rng(17)
    S = np.zeros((NSTONE, G.S_N), np.float64)
    for j in range(NSTONE):
        a = j * 2 * np.pi / NSTONE + rng.normal(0, 0.06) + 0.21
        r = R_STONES + rng.uniform(-0.35, 0.35)
        x, y = r * math.cos(a), r * math.sin(a)
        yaw = a + math.pi / 2 + rng.uniform(-0.35, 0.35)
        hx = rng.uniform(0.40, 0.72)
        hy = rng.uniform(0.24, 0.40)
        Hh = rng.uniform(2.0, 3.9)
        if j == 4:
            Hh = 1.35      # a broken stump
        if j == 8:
            Hh = 4.3       # one tall stone
        S[j, G.S_X], S[j, G.S_Y] = x, y
        S[j, G.S_C], S[j, G.S_S] = math.cos(yaw), math.sin(yaw)
        S[j, G.S_HX], S[j, G.S_HY], S[j, G.S_H] = hx, hy, Hh
        S[j, G.S_TAPER] = rng.uniform(0.12, 0.34)
        S[j, G.S_LX] = rng.uniform(-0.07, 0.07)
        S[j, G.S_LY] = rng.uniform(-0.07, 0.04)
        S[j, G.S_TNX], S[j, G.S_TNY] = rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3)
        S[j, G.S_TOP] = Hh - rng.uniform(0.05, 0.25)
        S[j, G.S_SEED] = rng.uniform(0, 10)
        S[j, G.S_ALB] = rng.uniform(0.16, 0.26)
        ext = max(hx, hy) + 0.3 + 0.1 * Hh
        S[j, G.S_BB:G.S_BB + 6] = (x - ext, x + ext, y - ext, y + ext, 0.0, Hh + 0.2)
    return S


# ============================================================== the hearth ===

def hearth_parts():
    """Kerb stones (irregular field stones, sunk), laid logs and sticks, a few big charcoal chunks (deterministic)."""
    rng = np.random.default_rng(31)
    KB = np.zeros((G.NKERB, 9))
    base = np.sort(rng.uniform(0, 2 * np.pi, G.NKERB) * 0.35 + np.arange(G.NKERB) * 2 * np.pi / G.NKERB * 0.65)
    for k in range(G.NKERB):
        a = base[k]
        r = G.KERB_R + rng.uniform(-0.05, 0.05)
        big = rng.uniform(0, 1)
        rx = 0.065 + 0.075 * big ** 1.5
        ry = rx * rng.uniform(0.60, 0.85)
        rz = rx * rng.uniform(0.55, 0.85)
        yaw = a + math.pi / 2 + rng.uniform(-0.6, 0.6)
        KB[k] = (r * math.cos(a), r * math.sin(a), rz * rng.uniform(0.15, 0.45), rx, ry, rz, math.cos(yaw),
                 math.sin(yaw), rng.uniform(0, 10))
    logs = []
    # eight split logs pushed in toward the stone like a star fire, half-burnt at their inner ends
    angs = np.arange(8) * 2 * np.pi / 8 + rng.uniform(-0.28, 0.28, 8) + 0.3
    for k in range(8):
        a = angs[k]
        r0, r1 = rng.uniform(0.42, 0.50), rng.uniform(0.70, 0.86)
        tw = rng.uniform(-0.35, 0.35)
        ra = rng.uniform(0.028, 0.050)
        z0 = ra * 0.85 + (0.03 if k in (2, 5) else 0.0)
        logs.append((r0 * math.cos(a), r0 * math.sin(a), z0, r1 * math.cos(a + tw * 0.4),
                     r1 * math.sin(a + tw * 0.4), ra * 0.9, ra * rng.uniform(0.75, 0.9), ra, rng.uniform(0, 6.28), 0))
    # kindling laid across the logs, some broken short
    for k in range(22):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.47, 0.80)
        L = rng.uniform(0.06, 0.22)
        yaw = a + math.pi / 2 + rng.uniform(-0.8, 0.8)
        cx, cy = r * math.cos(a), r * math.sin(a)
        dx, dy = 0.5 * L * math.cos(yaw), 0.5 * L * math.sin(yaw)
        rr = rng.uniform(0.006, 0.014)
        z = rng.uniform(0.012, 0.06)
        logs.append((cx - dx, cy - dy, z, cx + dx, cy + dy, z + rng.uniform(-0.015, 0.02), rr, rr * 0.8,
                     rng.uniform(0, 6.28), 1))
    LG = np.array(logs, np.float64)
    ch = []
    for k in range(9):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.46, 0.66)
        sz = rng.uniform(0.022, 0.038)
        ch.append((r * math.cos(a), r * math.sin(a), sz * 0.15, sz, sz * rng.uniform(0.6, 0.9), sz * 0.5,
                   rng.uniform(0, 10)))
    CH = np.array(ch, np.float64)
    return KB, LG, CH


def fuel_angles():
    """Where each emissary's torch meets the hearth (world azimuths): the fire catches there first."""
    return FIG_PSI[:NEM].copy()


# ============================================================ figure poses ===

def _two_bone(S, Hd, L1, L2, pole):
    d = Hd - S
    dist = np.linalg.norm(d)
    dist_c = min(max(dist, 0.08), L1 + L2 - 1e-3)
    u = d / max(dist, 1e-6)
    Hh = S + u * dist_c
    a = (L1 * L1 - L2 * L2 + dist_c * dist_c) / (2 * dist_c)
    hgt = math.sqrt(max(L1 * L1 - a * a, 0.0))
    p = pole - u * (pole @ u)
    p /= np.linalg.norm(p) + 1e-9
    E = S + u * a + p * hgt
    return E, Hh


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def shoulder_local(i, side, kneel, lean, ws=None):
    """A shoulder in the figure's local frame after the kneel and the lean (matches geom3.sd_fig's warp)."""
    hs = FIG_HS[i]
    ws = FIG_WS[i] if ws is None else ws
    zsh = 1.33 * hs - 0.43 * hs * kneel
    zhip = 0.62 * zsh
    x, y, z = 0.0, side * 0.200 * ws, zsh - 0.035 * hs
    lz = lean * smooth((z - (zhip - 0.20)) / 0.45)
    rz = z - zhip
    # sd_fig maps world -> body with a rotation by +lean; the body point maps back with -lean
    xw = math.cos(lz) * x + math.sin(lz) * rz
    zw = -math.sin(lz) * x + math.cos(lz) * rz + zhip
    return np.array([xw, y, zw])


def to_local(i, P, pos=None, ang=None):
    x0, y0 = (pos if pos is not None else (FIG_R[i] * math.cos(FIG_PSI[i]), FIG_R[i] * math.sin(FIG_PSI[i])))
    a = (FIG_PSI[i] + math.pi) if ang is None else ang
    c, s = math.cos(a), math.sin(a)
    dx, dy = P[0] - x0, P[1] - y0
    return np.array([dx * c + dy * s, -dx * s + dy * c, P[2]])


def to_world(i, p, pos=None, ang=None):
    x0, y0 = (pos if pos is not None else (FIG_R[i] * math.cos(FIG_PSI[i]), FIG_R[i] * math.sin(FIG_PSI[i])))
    a = (FIG_PSI[i] + math.pi) if ang is None else ang
    c, s = math.cos(a), math.sin(a)
    p = np.asarray(p, np.float64)
    return np.stack([x0 + p[..., 0] * c - p[..., 1] * s, y0 + p[..., 0] * s + p[..., 1] * c, p[..., 2]], -1)


def emissary_state(i, t):
    """Pose of emissary i at time t: position, facing, kneel, lean, walk, torch hand/axis (local), lit."""
    hs = FIG_HS[i]
    side = FIG_SIDE[i]
    pos = np.array([FIG_R[i] * math.cos(FIG_PSI[i]), FIG_R[i] * math.sin(FIG_PSI[i])])
    ang = FIG_PSI[i] + math.pi                        # facing the hearth
    kneel = 0.0
    lean = 0.0
    walk = 0.0
    phase = 0.0
    # --- the hold (P1): the torch held out before them at shoulder height, upright
    hand = np.array([0.36, side * 0.11, 1.10 * hs])
    tor = _unit([0.26, -side * 0.05, 1.0])
    breath = 0.004 * math.sin(2 * math.pi * t / 96.0 + FIG_BREATH_PH[i])
    hand = hand + np.array([0.0, 0.0, breath * 2.0]) + 0.003 * np.array(
        [math.sin(t * 0.21 + i), math.sin(t * 0.17 + 2 * i), math.sin(t * 0.23 + 3 * i)])
    lit = 1.0
    # --- P2: the torches come down together to the cold hearth, give their fire, and draw back spent
    if t >= TORCH_DOWN - 10 and t < P3[0] - 40:
        u = smoother(ramp(t, TORCH_DOWN + FIG_DOWN_DT[i], TORCH_DOWN + 26 + FIG_DOWN_DT[i]))
        back = smoother(ramp(t, 5236 + FIG_DOWN_DT[i], 5268 + FIG_DOWN_DT[i]))
        kneel = 0.62 * u * (1 - 0.35 * back)
        lean = math.radians(44.0) * u * (1 - 0.55 * back)
        tip = to_local(i, np.array([0.76 * math.cos(FIG_PSI[i]), 0.76 * math.sin(FIG_PSI[i]), 0.12]))
        tor_dn = _unit([0.52, -side * 0.02, -0.85])
        hand_dn = tip - 0.46 * tor_dn
        hand = hand * (1 - u) + hand_dn * u
        tor = _unit(tor * (1 - u) + tor_dn * u)
        # drawn back after the fire has taken: the spent torch held upright, low
        hand_bk = np.array([0.28, side * 0.10, 0.92 * hs])
        tor_bk = _unit([0.10, -side * 0.03, 1.0])
        hand = hand * (1 - back) + hand_bk * back
        tor = _unit(tor * (1 - back) + tor_bk * back)
        lit = 1.0 - smooth(ramp(t, FIRE_CATCH + 2, FIRE_CATCH + 16))
    # --- P3 (bar 70): spent torches dip into the fire that remains, catch, and are carried away
    if t >= P3[0] - 40:
        # they have closed in round the fire (a new plate after the white)
        pos = np.array([(FIG_R[i] - P3_STEP) * math.cos(FIG_PSI[i]), (FIG_R[i] - P3_STEP) * math.sin(FIG_PSI[i])])
        hand = np.array([0.28, side * 0.10, 0.92 * hs])
        tor = _unit([0.10, -side * 0.03, 1.0])
        lit = 0.0
        t0 = P3_DIP[i]
        u = smoother(ramp(t, t0, t0 + 14))                      # lean in
        v = smoother(ramp(t, t0 + 20, t0 + 34))                 # draw back, lit
        kneel = 0.34 * u * (1 - v)
        lean = math.radians(38.0) * u * (1 - v)
        # the head of the spent torch is lowered into the flank of the fire on the stone, steeply (a taper to a
        # flame, never a spit)
        tip = to_local(i, np.array([0.25 * math.cos(FIG_PSI[i]), 0.25 * math.sin(FIG_PSI[i]), 0.62]), pos=pos)
        tor_dn = _unit([0.60, -side * 0.02, -0.80])
        hand_dn = tip - 0.44 * tor_dn
        w_ = u * (1 - v)
        hand = hand * (1 - w_) + hand_dn * w_
        tor = _unit(tor * (1 - w_) + tor_dn * w_)
        lit = smooth(ramp(t, t0 + 12, t0 + 20))
        # carried: the torch raised a little higher, held out to the side as they turn away
        up = smooth(ramp(t, t0 + 26, t0 + 40))
        hand = hand + np.array([0.02, side * 0.03, 0.16]) * up
        # the turn (about 180 deg, the torch side leading) and the walk out
        tt0 = t0 + 30
        turn = smoother(ramp(t, tt0, tt0 + 18))
        ang = ang + side * math.pi * turn
        d = FIG_WALK_V[i] * max(t - (tt0 + 10), 0.0) / 24.0 * smooth(ramp(t, tt0 + 10, tt0 + 22))
        out = np.array([math.cos(FIG_PSI[i]), math.sin(FIG_PSI[i])])
        pos = pos + out * d
        walk = smooth(ramp(t, tt0 + 8, tt0 + 20))
        phase = d / 0.36 * math.pi
    return dict(pos=pos, ang=ang, kneel=kneel, lean=lean, hand=hand, tor=tor, lit=lit, walk=walk, phase=phase)


# --------------------------------------------------------------------- her ---
HER_KNEEL_R = 1.00           # where she kneels at the hearth's kerb
P3_STEP = 0.40               # bar 70: the bearers stand this much closer round the fire
RING_R = 0.22                # the Ring lies this far from the stone's centre, on her side


def ring_rest():
    a = PSI_HER
    x, y = RING_R * math.cos(a), RING_R * math.sin(a)
    th = math.atan2(y, x)
    z = G.flat_top(RING_R, th) + 0.5 * G.R_WIDTH
    return np.array([x, y, z])


_CROWD = {}


def _crowd3():
    """ACCORD-CROWD's module (shots/accord/crowd/crowd3.py), or None if it is missing or broken."""
    if 'm' not in _CROWD:
        try:
            d = os.path.join(HERE, 'crowd')
            if d not in sys.path:
                sys.path.insert(0, d)
            import crowd3
            _CROWD['m'] = crowd3
        except Exception as e:                    # the council renders without the crowd lane
            print('scene3: crowd3 unavailable:', repr(e)[:200], flush=True)
            _CROWD['m'] = None
    return _CROWD['m']


def her_state(t):
    """Her walk in (ACCORD-CROWD's her_walkin: along the Road through the crowd to her place by ~4704), then
    the walk to the hearth, the kneel, the reach that sets the Ring down (P1); kneeling at the hearth after."""
    out = np.array([math.cos(PSI_HER), math.sin(PSI_HER)])
    u_walk = smoother(ramp(t, 4800, 4830))
    r = FIG_R[HER] + (HER_KNEEL_R - FIG_R[HER]) * u_walk
    pos = out * r
    ang = PSI_HER + math.pi
    walk = math.sin(math.pi * u_walk) if 0 < u_walk < 1 else 0.0
    phase = (FIG_R[HER] - r) / 0.30 * math.pi
    if t < 4760:
        cm = _crowd3()
        wi = cm.her_walkin(t) if cm is not None else None
        if wi is not None:
            return dict(pos=np.asarray(wi['pos'], np.float64), ang=float(wi['ang']), kneel=0.0, lean=0.0,
                        walk=float(wi['walk']), phase=float(wi['phase']), arm2=0.0, hand2=None, fdir=None,
                        grip2=0.6, reach=0.0)
    kneel = smoother(ramp(t, 4826, 4840)) * 0.95
    lean = math.radians(12.0) * kneel
    arm2 = 0.0
    hand2 = None
    fdir = None
    grip2 = 0.6
    reach = 0.0
    if t >= 4822 and t < TORCH_DOWN:
        # the reach: from her breast, out over the ash, down to the stone; the hand opens on 4840 and comes back
        u = smoother(ramp(t, 4822, 4838))
        back = smoother(ramp(t, 4844, 4866))
        reach = u * (1 - back)
        lean = lean + math.radians(22.0) * reach
        arm2 = 1.0 if (u > 0.0 and back < 1.0) else 0.0
        R = ring_rest()
        tgt = to_local(HER, R + np.array([0, 0, 0.018]), pos=pos, ang=ang)
        chest = np.array([0.20, -0.06, 0.95])
        hand2 = chest * (1 - reach) + tgt * reach
        fdir = _unit([1.0, -0.15, -0.35 * reach])
        grip2 = 0.75 - 0.75 * smooth(ramp(t, 4838, 4842))
    if TORCH_DOWN <= t < P3[0] - 40:
        # P2: she bends in over the hearth as her hand goes back to the Ring, and stays bent over her fist
        lean = lean + math.radians(34.0) * smoother(ramp(t, 5128, 5154))
    return dict(pos=pos, ang=ang, kneel=kneel, lean=lean, walk=walk, phase=phase, arm2=arm2, hand2=hand2,
                fdir=fdir, grip2=grip2, reach=reach)


def ring_state(t):
    """RP: on, centre, axis, glow, hidden. Before 4840 it is in her hand (hidden); from 5160 in her fist."""
    RP = np.zeros(G.RP_N)
    c = ring_rest()
    RP[1:4] = c
    RP[4:7] = (0.0, 0.0, 1.0)
    if RING_SET - 1 <= t < HAND_CLOSE + 2:
        RP[0] = 1.0
        # it settles: a last tiny rock as it is laid down
        a = 0.06 * math.exp(-max(t - RING_SET, 0.0) / 3.0) * math.sin(max(t - RING_SET, 0.0) * 1.9)
        RP[4:7] = _unit([a, 0.3 * a, 1.0])
    elif t >= FINGERS_OPEN + 2 and t < P3[0]:
        # it falls out of her opened hand onto the stone (the melt takes it from 5364)
        f = hand_fist_pos(t)
        u = clamp01((t - (FINGERS_OPEN + 2)) / 6.0)
        z = f[2] - 0.01 + (c[2] - (f[2] - 0.01)) * u * u
        RP[0] = 1.0
        RP[1:4] = (f[0] * (1 - u) + c[0] * u, f[1] * (1 - u) + c[1] * u, z)
        tilt = 0.9 * (1 - u)
        RP[4:7] = _unit([math.sin(tilt), 0.0, math.cos(tilt)])
        RP[7] = 1.4 * smooth(ramp(t, 5366, 5374))
    return RP


# ------------------------------------------------------------ the hand rig ---
_FING = [  # base (u, v, w), spread (deg), segment lengths
    ((0.093, 0.027, 0.0), 9.0, (0.040, 0.024, 0.020)),
    ((0.097, 0.009, 0.0), 2.0, (0.044, 0.027, 0.021)),
    ((0.094, -0.009, 0.0), -4.0, (0.041, 0.026, 0.020)),
    ((0.087, -0.026, 0.0), -11.0, (0.033, 0.020, 0.018)),
]


def hand_joints(curl, thumb=0.0, spread=1.0, tremble=0.0, t=0.0):
    """Finger joints (5 x 4 x 3) of a right hand in hand-local (u, v, w). curl: scalar or 4 values."""
    J = np.zeros((5, 4, 3))
    cs = np.broadcast_to(np.asarray(curl, np.float64), (4,))
    for f, (base, sp, Ls) in enumerate(_FING):
        c = cs[f] + tremble * 0.04 * math.sin(t * 1.7 + f * 1.3)
        a = math.radians(sp * spread)
        e1 = np.array([math.cos(a), math.sin(a), 0.0])
        e2 = np.array([0.0, 0.0, -1.0])
        p = np.array(base)
        J[f + 1, 0] = p
        ang = 0.0
        for k, (L, mx) in enumerate(zip(Ls, (72.0, 98.0, 62.0))):
            ang += math.radians(mx) * c
            d = math.cos(ang) * e1 + math.sin(ang) * e2
            p = p + L * d
            J[f + 1, k + 1] = p
    # the thumb: from the heel of the palm, open along the side, curling across the palm with `thumb`
    b = np.array([0.014, 0.030, -0.012])
    d_open = _unit([0.62, 0.66, -0.25])
    d_close = _unit([0.45, -0.35, -0.82])
    p = b
    J[0, 0] = p
    for k, (L, bend) in enumerate(zip((0.040, 0.031, 0.026), (0.0, 0.35, 0.6))):
        d = _unit(d_open * (1 - thumb) + d_close * thumb)
        if k > 0:
            d = _unit(d + np.array([0.0, -0.4, -0.6]) * bend * thumb)
        p = p + L * d
        J[0, k + 1] = p
    return J


def hand_frame(wrist, a, c_hint, scale=1.0, forearm=0.26, gilt=0.0, sleeve=1.0, left=False):
    """HD array: wrist position, a = wrist->knuckles, c = back of the hand (orthogonalised), b = c x a
    (toward the thumb for a right hand; mirrored for a left hand)."""
    a = _unit(a)
    c = _unit(np.asarray(c_hint) - a * (np.asarray(c_hint) @ a))
    b = np.cross(c, a)
    if left:
        b = -b
    HD = np.zeros(G.HD_N)
    HD[0:3] = wrist
    HD[3:6] = a
    HD[6:9] = b
    HD[9:12] = c
    HD[12] = scale
    HD[13] = forearm
    HD[14] = gilt
    HD[15] = sleeve
    return HD


def hand_bounds(HD, J, extra=0.02):
    if HD[23] > 0.5:
        s = HD[12]
        pts = [HD[0:3] + (HD[3:6] * HD[j] + HD[6:9] * HD[j + 1] + HD[9:12] * HD[j + 2]) * s for j in (16, 19)]
        pts += [p + np.array(o) for p in pts for o in ((0.09, 0, 0), (-0.09, 0, 0), (0, 0.09, 0), (0, -0.09, 0),
                                                        (0, 0, 0.09), (0, 0, -0.09))]
    else:
        pts = [HD[0:3] - HD[3:6] * (HD[13] + 0.25)]
    for f in range(5):
        for k in range(4):
            u, v, w = J[f, k]
            pts.append(HD[0:3] + (HD[3:6] * u + HD[6:9] * v + HD[9:12] * w) * HD[12])
    pts.append(HD[0:3])
    P = np.array(pts)
    m = 0.06 * HD[12] + extra
    return np.array([P[:, 0].min() - m, P[:, 0].max() + m, P[:, 1].min() - m, P[:, 1].max() + m,
                     P[:, 2].min() - m, P[:, 2].max() + m])


def hand_fist_pos(t):
    """P2: where her fist holds the Ring above the stone (world)."""
    c = ring_rest()
    lift = smooth(ramp(t, HAND_CLOSE + 2, HAND_CLOSE + 22))
    tr = 0.0025 * math.sin(t * 1.3) + 0.0015 * math.sin(t * 2.9 + 1.0)
    grow = smooth(ramp(t, FIRE_CATCH, 5330))
    return c + np.array([tr * (1 + 2 * grow), tr * 0.7, 0.012 + 0.10 * lift + tr])


def her_hand(t):
    """P2: her right hand (gloved), in from the side: to the Ring, closing on it (5160), the fist held in the
    fire, forced open at 5356. Returns HD, J, or None."""
    if t < 5128 or t >= P3[0] - 40:
        return None
    out = np.array([math.cos(PSI_HER), math.sin(PSI_HER), 0.0])
    lat = np.array([-math.sin(PSI_HER), math.cos(PSI_HER), 0.0])
    c = ring_rest()
    # approach: from her kneeling place, low over the ash, to above the Ring
    u = smoother(ramp(t, 5134, 5156))
    start = np.array([out[0] * 0.95, out[1] * 0.95, 0.62]) + lat * 0.05
    over = c + np.array([0, 0, 0.055]) + out * 0.085
    fist = hand_fist_pos(t) + out * 0.080
    k = smoother(ramp(t, HAND_CLOSE, HAND_CLOSE + 10))
    wrist = start * (1 - u) + over * u
    wrist = wrist * (1 - k) + (fist + np.array([0, 0, 0.015])) * k
    # the hand points in toward the stone, palm down, tilting its fingers down to the Ring as it closes
    a = _unit(-out + np.array([0, 0, -0.35 - 0.3 * k]))
    chint = np.array([0.0, 0.0, 1.0]) + 0.3 * out
    # a living hand: relaxed, each finger curled a little more than the one before; then it closes on the Ring
    relax = np.array([0.20, 0.29, 0.37, 0.46]) + 0.06 * u
    cl = smoother(ramp(t, HAND_CLOSE - 3, HAND_CLOSE + 6))
    curl = relax + (np.array([0.90, 0.95, 0.98, 1.0]) - relax) * cl
    thumb = 0.28 + 0.67 * smoother(ramp(t, HAND_CLOSE - 2, HAND_CLOSE + 7))
    tremble = smooth(ramp(t, FIRE_CATCH, 5300)) * 1.0
    # forced open in the white heart: a spasm, the fingers spring out
    op = smoother(ramp(t, FINGERS_OPEN, FINGERS_OPEN + 5))
    curl = curl * (1 - op) + (-0.05) * op
    thumb = thumb * (1 - op) + 0.1 * op
    J = hand_joints(curl, thumb, spread=0.85 + 0.9 * op, tremble=tremble, t=t)
    HD = hand_frame(wrist, a, chint, scale=1.0, forearm=0.30, gilt=0.0, sleeve=1.0)
    # her arm, in her cloak's wool sleeve: from her right shoulder (bent in over the hearth) to the wrist
    st = her_state(t)
    S = to_world(HER, shoulder_local(HER, -1.0, st['kneel'], st['lean']), pos=st['pos'], ang=st['ang'])
    ang = st['ang']
    right = np.array([math.sin(ang), -math.cos(ang), 0.0])
    E, _ = _two_bone(S, wrist, 0.31, 0.30, right * 0.7 + np.array([0.0, 0.0, -0.6]))
    set_arm(HD, E, S)
    return HD, J


def set_arm(HD, elbow, shoulder):
    """A real arm for sd_hand: the elbow and shoulder (world) into hand-local units; arm mode on."""
    s = HD[12]
    for j, P in ((16, elbow), (19, shoulder)):
        q = np.asarray(P, np.float64) - HD[0:3]
        HD[j] = (q @ HD[3:6]) / s
        HD[j + 1] = (q @ HD[6:9]) / s
        HD[j + 2] = (q @ HD[9:12]) / s
    HD[23] = 1.0
    return HD


def gilded_hand(t, Fa):
    """P1/P2: the gilded, ember-scarred glove of the forge that grasped, closed round its torch."""
    i = GILDED
    if Fa[i, G.F_HEROHAND] < 0.5:
        return None
    hl = Fa[i, G.F_HX:G.F_HZ + 1]
    tl = Fa[i, G.F_TX:G.F_TZ + 1]
    sl = Fa[i, G.F_SHX:G.F_SHZ + 1]
    ang = Fa[i, G.F_ANG]
    pos = Fa[i, G.F_X:G.F_Y + 1]
    hw = to_world(i, hl, pos=pos, ang=ang)
    tw = to_world(i, hl + tl, pos=pos, ang=ang) - hw
    sw = to_world(i, sl, pos=pos, ang=ang)
    # hammer grip: the shaft runs across the palm (the hand's v axis ~ the shaft)
    fwd = _unit(hw - sw)
    a = _unit(fwd - tw * (fwd @ tw))
    # thumb side up the shaft (b = +shaft): right hand c = a x b, left hand mirrored
    c_hint = np.cross(a, _unit(tw)) * (1.0 if FIG_SIDE[i] < 0 else -1.0)
    HD = hand_frame(np.zeros(3), a, c_hint, scale=1.08, forearm=0.22, gilt=1.0, sleeve=1.0,
                    left=FIG_SIDE[i] > 0)
    cc = HD[9:12]
    # the shaft lies across the palm at the base of the fingers (u 0.090, w -0.026 in hand units)
    HD[0:3] = hw - a * 0.090 * 1.08 + cc * 0.026 * 1.08
    J = hand_joints([0.86, 0.9, 0.92, 0.95], thumb=0.85, spread=0.5)
    return HD, J


# ================================================================ figures ===

def figures(t):
    Fa = np.zeros((NFIG, G.F_N), np.float64)
    for i in range(NFIG):
        if i < NEM:
            st = emissary_state(i, t)
        else:
            st = her_state(t)
        pos, ang = st['pos'], st['ang']
        Fa[i, G.F_X], Fa[i, G.F_Y] = pos
        Fa[i, G.F_ANG] = ang
        Fa[i, G.F_C], Fa[i, G.F_S] = math.cos(ang), math.sin(ang)
        Fa[i, G.F_HS], Fa[i, G.F_WS] = FIG_HS[i], FIG_WS[i]
        Fa[i, G.F_TYPE] = FIG_TYPE[i]
        Fa[i, G.F_SEED] = 1.7 * i + 0.3
        Fa[i, G.F_R:G.F_B + 1] = FIG_CLOTH[i]
        Fa[i, G.F_R2:G.F_B2 + 1] = FIG_CLOTH2[i] if i != HER else SCARF
        Fa[i, G.F_CAPE] = FIG_CAPE[i] * FIG_HS[i]
        Fa[i, G.F_TRAIN] = FIG_TRAIN[i]
        Fa[i, G.F_WEAVE] = FIG_WEAVE[i]
        Fa[i, G.F_SHEENK] = FIG_SHEENK[i]
        Fa[i, G.F_SIDE] = FIG_SIDE[i]
        Fa[i, G.F_KNEEL] = st['kneel']
        Fa[i, G.F_LEAN] = st['lean']
        Fa[i, G.F_BREATH] = 0.006 * math.sin(2 * math.pi * t / 96.0 + FIG_BREATH_PH[i])
        Fa[i, G.F_BOW] = FIG_BOW[i] + math.radians(1.2) * math.sin(2 * math.pi * t / 131.0 + FIG_BREATH_PH[i])
        w = st.get('walk', 0.0)
        Fa[i, G.F_SWAY] = 0.018 * w * math.sin(st.get('phase', 0.0))
        Fa[i, G.F_HEM] = 0.05 * w
        Fa[i, G.F_PHASE] = st.get('phase', 0.0)
        side = FIG_SIDE[i]
        if i < NEM:
            Fa[i, G.F_TORCH] = 1.0
            sh = shoulder_local(i, side, st['kneel'], st['lean'])
            hand = st['hand']
            tor = st['tor']
            E, Hh = _two_bone(sh, hand, 0.31, 0.30, np.array([-0.35, side * 1.0, -0.9]))
            Fa[i, G.F_SHX:G.F_SHZ + 1] = sh
            Fa[i, G.F_EX:G.F_EZ + 1] = E
            Fa[i, G.F_HX:G.F_HZ + 1] = Hh
            Fa[i, G.F_TX:G.F_TZ + 1] = tor
            Fa[i, G.F_TLIT] = st['lit']
            Fa[i, G.F_GILT] = 1.0 if i == GILDED else 0.0
            Fa[i, G.F_HEROHAND] = 1.0 if i == GILDED else 0.0
            extra = [Hh + 0.60 * tor, Hh - 0.30 * tor, E, Hh, sh]
        else:
            Fa[i, G.F_SHAWL] = 1.0
            Fa[i, G.F_CAPE] = 0.30 * FIG_HS[i]
            extra = []
            if st['arm2'] > 0.5:
                sh2 = shoulder_local(i, -side, st['kneel'], st['lean'])
                E2, H2 = _two_bone(sh2, st['hand2'], 0.28, 0.27, np.array([-0.3, -side * 1.0, -0.8]))
                Fa[i, G.F_ARM2] = 1.0
                Fa[i, G.F_S2X:G.F_S2Z + 1] = sh2
                Fa[i, G.F_E2X:G.F_E2Z + 1] = E2
                Fa[i, G.F_H2X:G.F_H2Z + 1] = H2
                Fa[i, G.F_D2X:G.F_D2Z + 1] = st['fdir']
                Fa[i, G.F_GRIP2] = st['grip2']
                extra = [E2, H2, H2 + 0.12 * st['fdir'], sh2]
        # AABB
        hs, ws = FIG_HS[i], FIG_WS[i]
        pts = [np.array([sx, sy, sz]) for sx in (-(0.30 + FIG_TRAIN[i] + 0.2 * st['kneel']) * ws, 0.34 * ws)
               for sy in (-0.40 * ws, 0.40 * ws) for sz in (0.0, 1.97 * hs)]
        if st['lean'] > 0:
            pts.append(np.array([0.9 * math.sin(st['lean']) + 0.25, 0.0, 1.7 * hs]))
        pts += [np.asarray(p) for p in extra]
        wp = to_world(i, np.array(pts), pos=pos, ang=ang)
        m = 0.10
        Fa[i, G.F_BB:G.F_BB + 6] = (wp[:, 0].min() - m, wp[:, 0].max() + m, wp[:, 1].min() - m,
                                    wp[:, 1].max() + m, -0.02, wp[:, 2].max() + m)
    return Fa


def torch_heads(t, Fa):
    """World positions of lit torch heads: (n, 10) = base xyz, axis xyz, lit, seed, vel xy."""
    out = []
    for i in range(NEM):
        if Fa[i, G.F_TORCH] < 0.5:
            continue
        pos = Fa[i, G.F_X:G.F_Y + 1]
        ang = Fa[i, G.F_ANG]
        hl = Fa[i, G.F_HX:G.F_HZ + 1]
        tl = Fa[i, G.F_TX:G.F_TZ + 1]
        top = to_world(i, hl + 0.44 * tl, pos=pos, ang=ang)
        ax = to_world(i, hl + tl, pos=pos, ang=ang) - to_world(i, hl, pos=pos, ang=ang)
        out.append([top[0], top[1], top[2], ax[0], ax[1], ax[2], 0.0, 3.1 * i + 0.7, 0.0, 0.0])
    return np.array(out, np.float64) if out else np.zeros((0, 10))


def torch_lit(t):
    """Per emissary: flame strength 0..1 (P1 lit; P2 given to the hearth; P3 dipped and carried)."""
    return np.array([emissary_state(i, t)['lit'] for i in range(NEM)])


# ================================================================ the fire ===

def fire_state(t):
    """The fire everyone lit: strength, height, white-hot, the ring's ignition progress, hollow round her fist."""
    if t < FIRE_CATCH - 2:
        return dict(on=0.0, H=0.0, white=0.0, spread=0.0, hollow=0.0, conv=0.0, calm=0.0, age=0.0)
    if t < P3[0] - 40:
        age = t - FIRE_CATCH
        spread = smooth(ramp(t, FIRE_CATCH, FIRE_CATCH + 16))
        on = smooth(ramp(t, FIRE_CATCH - 2, FIRE_CATCH + 6))
        H = 0.25 + 1.05 * ease_out(ramp(t, FIRE_CATCH, FIRE_CATCH + 70), 2.0) + 0.35 * smooth(ramp(t, 5300, 5359))
        white = smooth(ramp(t, 5318, 5356)) ** 1.3 + 0.6 * smooth(ramp(t, 5356, 5379))
        conv = 0.35 + 0.45 * smooth(ramp(t, FIRE_CATCH + 10, FIRE_CATCH + 60))
        hollow = 1.0 - smooth(ramp(t, 5334, 5362))
        return dict(on=on, H=H, white=white, spread=spread, hollow=hollow, conv=conv, calm=0.0, age=age)
    # P3: out of the white, the fire settles to a warm, steady fire on the stone where the Ring was
    # (flame3.calm_density over calm_flames(t)); H is the fire's height factor (1 = settled)
    age = t - FIRE_CATCH
    settle = smooth(ramp(t, P3[0], P3[0] + 30))
    white = (1.0 - settle) ** 1.8
    H = 1.0 + 0.40 * (1.0 - settle)
    return dict(on=1.0, H=H, white=white, spread=1.0, hollow=0.0, conv=0.55, calm=1.0, age=age, p3=1.0,
                settle=settle)


def white_level(t):
    """Screen-level whiteout: P2 ends in white; P3 comes out of it."""
    if plate(t) == 2:
        return smooth(ramp(t, 5340, 5372)) ** 1.5
    if plate(t) == 3:
        return (1.0 - smooth(ramp(t, P3[0] - 1, P3[0] + 22))) ** 1.4
    return 0.0


def p2_flames(t):
    """AC2 (flame3.calm_density rows, mode 1): the fire everyone lit rises AROUND HER FIST. Each torch catches the
    laid fuel where it touched (r ~0.62 at its bearer's azimuth): a low flame leaps up there, bent in toward her
    fist. From it a runner runs in along the fuel to the stone, climbs onto it and stands up beside her fist; the
    thirteen runners arrive from every side at once and grow into one crown of fire round the fist (a clear hollow
    at its heart, where the fist is). The white takes it from ~5318. Returns (CF, 0)."""
    fs = fire_state(t)
    if fs['on'] <= 0.0:
        return np.zeros((1, 8)), 0
    fist = hand_fist_pos(max(t, HAND_CLOSE))
    fx, fy = fist[0], fist[1]
    Ts = t / 24.0
    grow = fs['H']
    rng = np.random.default_rng(66)
    rows = []

    def pump(seed, big):
        return (1.0 + big * (0.085 * math.sin(2 * math.pi * 3.1 * Ts + seed) + 0.060 * math.sin(2 * math.pi * 1.37 * Ts + 2.1 * seed)
                             + 0.045 * math.sin(2 * math.pi * 0.53 * Ts + 3.3 * seed)))

    for k in range(NEM):
        a = FIG_PSI[k]
        tc = FIRE_CATCH + 0.5 * FIG_DOWN_DT[k]
        age = t - tc
        g = ease_out(clamp01(age / 10.0), 2.0)
        if g <= 0.0:
            continue
        sd = 3.1 + 1.9 * k
        x, y = 0.62 * math.cos(a), 0.62 * math.sin(a)
        dx, dy = fx - x, fy - y
        D = math.hypot(dx, dy) + 1e-9
        # the catch: a low flame where the torch touched, bent in toward her fist (the ring behind stays low)
        Hh = (0.24 + 0.12 * min(grow, 1.3)) * (0.4 + 0.6 * g) * pump(sd, 1.0) * rng.uniform(0.85, 1.15)
        kk = 0.45 * D / max(Hh, 0.15)
        rows.append((x, y, 0.03, Hh, 0.085, kk * dx / D + 0.08, kk * dy / D - 0.03, sd))
        # the runner: in along the fuel, onto the stone, up beside her fist (one flame of the crown)
        u = smoother(clamp01((age - 2.0) / 20.0))
        if u > 0.0:
            ca, sa = dx / D, dy / D
            ex, ey = fx - 0.155 * ca, fy - 0.155 * sa            # its place on the crown, on its bearer's side
            px_, py_ = x + (ex - x) * u, y + (ey - y) * u
            z0 = 0.03 + (G.STONE_TOP - 0.045) * smooth(clamp01((u - 0.45) / 0.4))
            Hc = (0.20 + 1.00 * u * max(grow - 0.25, 0.0)) * pump(sd + 7.0, 1.1) * rng.uniform(0.88, 1.12)
            lean_in = 0.10 + 0.10 * u                            # over the fist a little, never onto it
            rows.append((px_, py_, z0, Hc, 0.075 + 0.045 * u, lean_in * ca + 0.05, lean_in * sa - 0.02, sd + 19.0))
    if not rows:
        return np.zeros((1, 8)), 0
    return np.array(rows, np.float64), 0


def calm_flames(t):
    """Bar 70 (flame3.calm_density rows: x, y, z0, Hf, Rf, sx, sy, seed): the fire that remains, standing on the
    stone where the Ring was. A tall heart and a crown of flames rise from the coals on the stone; low flames lick
    along the burning inner ends of the logs round it; all lean in a little and downwind, and each pumps and sways
    on its own. Returns (CF, nmain)."""
    fs = fire_state(t)
    grow = fs['H']
    rng = np.random.default_rng(70)
    wind = np.array([0.26, -0.10])
    Ts = t / 24.0
    zt = G.STONE_TOP - 0.025
    rows = []

    def pump(seed, big):
        return (1.0 + big * (0.085 * math.sin(2 * math.pi * 3.1 * Ts + seed) + 0.060 * math.sin(2 * math.pi * 1.37 * Ts + 2.1 * seed)
                             + 0.045 * math.sin(2 * math.pi * 0.53 * Ts + 3.3 * seed)))

    def sway(seed):
        return (0.10 * math.sin(2 * math.pi * 0.61 * Ts + seed), 0.10 * math.sin(2 * math.pi * 0.47 * Ts + 1.7 * seed))

    # the heart
    s0 = 1.3
    sw = sway(s0)
    rows.append((0.012, -0.008, zt, 0.92 * grow * pump(s0, 1.0), 0.165, wind[0] + sw[0], wind[1] + sw[1], s0))
    # the crown on the coals, leaning in toward the heart
    for k in range(5):
        a = 2 * math.pi * k / 5 + rng.uniform(-0.35, 0.35) + 0.4
        r = rng.uniform(0.12, 0.18)
        x, y = r * math.cos(a), r * math.sin(a)
        sd = 2.9 + 1.7 * k
        sw = sway(sd)
        Hh = rng.uniform(0.50, 0.72) * grow * pump(sd, 1.2)
        rows.append((x, y, zt, Hh, rng.uniform(0.105, 0.130), wind[0] - 0.24 * math.cos(a) + sw[0],
                     wind[1] - 0.24 * math.sin(a) + sw[1], sd))
    nmain = len(rows)
    # low flames along the burning inner ends of the eight logs, leaning in over the stone
    LG = hearth_parts()[1]
    for k in range(8):
        A = LG[k, 0:3]
        B = LG[k, 3:6]
        ra = LG[k, 7]
        for j, fr in enumerate((0.08, 0.40)):
            P = A + (B - A) * fr
            rr = math.hypot(P[0], P[1]) + 1e-9
            sd = 11.0 + 2.3 * k + 1.1 * j
            sw = sway(sd)
            # thin licking tongues that lean in over the stone (never round puffs)
            Hh = (0.36 - 0.12 * j) * rng.uniform(0.85, 1.15) * grow * pump(sd, 1.4)
            rows.append((P[0], P[1], P[2] + 0.6 * ra, Hh, (0.040 - 0.008 * j) * rng.uniform(0.9, 1.1),
                         wind[0] - 0.80 * P[0] / rr + 0.6 * sw[0], wind[1] - 0.80 * P[1] / rr + 0.6 * sw[1], sd))
    return np.array(rows, np.float64), nmain


# ================================================================== camera ===

def _cam_from(target, h, tilt_deg, phi_deg, scale, cy=None):
    th = math.radians(tilt_deg)
    ph = math.radians(phi_deg)
    C = np.asarray(target, np.float64) + h * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph),
                                                       math.cos(th)])
    Fw = _unit(np.asarray(target) - C)
    U = np.array([-math.cos(th) * math.cos(ph), -math.cos(th) * math.sin(ph), math.sin(th)])
    if tilt_deg < 0.5:
        U = np.array([-math.cos(ph), -math.sin(ph), 0.0])
    U = _unit(U - Fw * (U @ Fw))
    R = np.cross(Fw, U)
    f = F_FULL * scale
    return np.concatenate([C, R, U, Fw, [f, (W / 2) * scale, ((H / 2) if cy is None else cy) * scale]]).astype(np.float64)


# P1: the descent from the map's height, orbiting; then the council; the Ring; the gilded hand
_P1_H = PchipInterpolator([4470, 4480, 4540, 4600, 4660, 4720, 4780, 4840, 4900, 4960, 5000, 5040, 5080, 5130],
                          np.log([300.0, 300.0, 170.0, 62.0, 21.0, 10.4, 8.6, 7.6, 6.9, 6.3, 5.4, 4.2, 4.0, 3.9]))
PHI_GILT = math.degrees(FIG_PSI[5]) - 90.0 - 12.0   # the orbit's phase (kept: 4480 matches MAP's drawn ring)


def p1_phi(t):
    # a slow orbit, clockwise on screen, easing into bar 64 so the gilded hand lands and holds
    w0 = -0.075
    s = (t - 5040.0)
    return PHI_GILT + w0 * s * (0.55 + 0.45 * smooth(ramp(t, 4480, 4700))) * (1.0 - 0.6 * smooth(ramp(t, 5000, 5060)))


def p1_target(t):
    return np.array([0.0, 0.0, 0.34])


def _cam_look(C, T, Uh, scale):
    Fw = _unit(np.asarray(T, np.float64) - C)
    U = _unit(Uh - Fw * (Uh @ Fw))
    R = np.cross(Fw, U)
    f = F_FULL * scale
    return np.concatenate([C, R, U, Fw, [f, (W / 2) * scale, (H / 2) * scale]]).astype(np.float64)


GILT_FIND0, GILT_FIND1 = 4990, 5042
P3_PHI0 = 187.0        # bar 70 starts behind her right shoulder, between her and emissary 12


def gilt_knuckles(t):
    """World position of the gilded hand's knuckles at t."""
    Fa = figures(t)
    HD, J = gilded_hand(t, Fa)
    return HD[0:3] + HD[3:6] * 0.075 * HD[12] + HD[9:12] * 0.012


def gilt_find_cam(t, scale):
    """The orbit finds the gilded hand: close (2 m), 30 deg off vertical, from the side the back of the hand
    faces and a little toward the hearth, so the gold crust catches its own torch and the flame stands clear
    above it in frame; a slow drift in during the hold."""
    Fa = figures(t)
    HD, J = gilded_hand(t, Fa)
    kn = HD[0:3] + HD[3:6] * 0.075 * HD[12] + HD[9:12] * 0.012
    c = HD[9:12]
    az_back = math.atan2(c[1], c[0])
    inward = FIG_PSI[GILDED] + math.pi
    d = (inward - az_back + 3 * math.pi) % (2 * math.pi) - math.pi
    phi = math.degrees(az_back + 0.22 * d) + 5.0 * smooth(ramp(t, GILT_FIND1, 5119))
    h = 1.08 - 0.08 * smooth(ramp(t, GILT_FIND1, 5119))
    return _cam_from(kn, h, 34.0, phi, scale), kn


def camera(t, scale=1.0):
    p = plate(t)
    if p == 1:
        h = float(np.exp(_P1_H(min(t, GILT_FIND1))))
        tilt = 12.0 + 1.5 * smooth(ramp(t, 4700, 5040))
        co = _cam_from(p1_target(min(t, GILT_FIND1)), h, tilt, p1_phi(min(t, GILT_FIND1)), scale)
        if t <= GILT_FIND0:
            return co
        cf, kn = gilt_find_cam(t, scale)
        k = smoother(ramp(t, GILT_FIND0, GILT_FIND1))
        Co = co[0:3]
        To = Co + co[9:12] * h
        C = Co * (1 - k) + cf[0:3] * k
        T = To * (1 - k) + kn * k
        Uh = co[6:9] * (1 - k) + cf[6:9] * k
        return _cam_look(C, T, Uh, scale)
    if p == 2:
        phi = math.degrees(PSI_HER) + 90.0 + 0.035 * (t - P2[0])
        h = float(np.exp(np.interp(t, [5120, 5160, 5230, 5300, 5379], np.log([2.80, 2.25, 2.35, 2.60, 2.75]))))
        f = hand_fist_pos(max(t, 5160)) if t > 5150 else ring_rest()
        k = smooth(ramp(t, 5120, 5170))
        tgt = np.array([0.0, 0.0, 0.30]) * (1 - k) + np.array([f[0] * 0.55, f[1] * 0.55, 0.36]) * k
        tilt = 4.0 + 17.0 * smoother(ramp(t, 5205, 5300))
        return _cam_from(tgt, h, tilt, phi, scale)
    # P3: out of the white, low and oblique behind her shoulder (so the flames stand up as flames and the far
    # bearers are dark against them); then the crane up, turning to top-down as the flame is carried away, so by
    # 5600 the ring of stones lies where MAP's drawn ring will burn through
    phi = P3_PHI0 + 0.05 * (t - P3[0])
    h = float(np.exp(np.interp(t, [5520, 5548, 5566, 5582, 5600, 5620, 5640, 5660, 5679],
                               np.log([3.25, 3.45, 4.3, 6.2, 10.5, 17.0, 27.0, 42.0, 62.0]))))
    k = smoother(ramp(t, 5552, 5604))
    tilt = 5.0 + 34.0 * (1.0 - k)
    tgt = np.array([0.0, 0.0, 0.62 - 0.26 * k])
    return _cam_from(tgt, h, tilt, phi, scale)


def project(cam, P):
    C, R, U, Fw = cam[0:3], cam[3:6], cam[6:9], cam[9:12]
    f, cx, cy = cam[12], cam[13], cam[14]
    v = np.asarray(P, np.float64) - C
    zc = v @ Fw
    xs = cx + f * (v @ R) / zc
    ys = cy - f * (v @ U) / zc
    return np.stack([xs, ys], -1), zc


# =============================================================== exposure ===

def exposure(t):
    p = plate(t)
    if p == 1:
        # high over the dark plain, only the rivers of light: open up; down among the torches: normal;
        # close on the gilded hand under its own flame: stop down
        h = float(np.exp(_P1_H(t)))
        return (1.25 + 1.6 * smooth((math.log(h) - math.log(12.0)) / (math.log(120.0) - math.log(12.0)))
                - 0.40 * smoother(ramp(t, GILT_FIND0, GILT_FIND1)))
    if p == 2:
        # the torches come down close to everything: stop down; then again as the fire takes
        return (1.10 - 0.35 * smooth(ramp(t, TORCH_DOWN, TORCH_DOWN + 30)) - 0.30 * smooth(ramp(t, FIRE_CATCH, FIRE_CATCH + 40))
                - 0.15 * smooth(ramp(t, 5290, 5350)))
    return 0.62 + 0.40 * smooth(ramp(t, P3[0] + 4, P3[0] + 40)) + 0.25 * smooth(ramp(t, 5590, 5660))
