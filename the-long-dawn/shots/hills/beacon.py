"""FIRST BEACON 1200-1439: the Young Woman on a Himalayan summit, ~60 years earlier.

One continuous take. 1200-1235 black (a breath of blowing snow). Flint strikes at 1236,
1262, 1290: spark bursts light her hands, her profile, the red scarf. A spark lives in
the tinder; she blows; 1318 the kindling catches (face lit from below, breath vapour).
1360 the beacon ROARS; she rises and steps back; the camera pulls back and up (ease-out)
to reveal the snowy summit, spindrift, a sea of moonlit peaks under a vast starfield.
The moonlit world fades up from black as it is revealed (eye adaptation).
"""
import math

import cv2
import numpy as np

import characters as ch
import fire
from core import Camera, fbm1_np, fnoise1, look, over, over_region, smoothstep, track, splat_gauss
from hillscene import make_wind_fn, wind_at
from land import Ridge, make_profile, pack_ridges, render_ridges
import peaks
from puppet import Chain, Figure
from sky import Sky, dir_from_az_el, render_full_sky

FPS = 24.0
F0, F1 = 1200, 1439
STRIKES = (1236, 1262, 1290)
CATCH = 1318
ROAR = 1360
R_EARTH = 6371000.0
YW_X = 0.96                    # her hip x while kneeling
CAIRN = ch.Cairn(seed=11)
FIRE_BASE = np.array([0.0, CAIRN.bk_bot + 0.10, 0.0])
TINDER = np.array([0.20, CAIRN.bk_top - 0.03, -0.02])

MOON = np.array([0.55, 0.70, 0.95]) * 0.55          # moonlight on snow (linear)
# True 3-D moonlit range + rocky summit for the reveal (peaks.py). False = the old pyramid
# ridge cards + blob summit (land.py). Only this shot uses peaks.py.
TRUE_PEAKS = True

FOG = np.array([1.0 / 60000.0, 1.0 / 900.0, -1500.0, 260.0, 0.6, math.radians(3.0), 0.9,
                30.0, 1.0 / 60.0, MOON[0], MOON[1], MOON[2], 1.0], np.float64)

# v2 (Sep 2026): the Young Woman as a sculpted 3-D figure lit by the fire (heroine.py +
# heroine_sdf.py) instead of the 2-D card puppet. Only this take uses it; False = the v1 puppet.
HEROINE_V2 = True


# ------------------------------------------------------------------ world ---

def summit_profile(X):
    X = np.asarray(X, np.float64)
    a = np.abs(X)
    h = -0.025 * X ** 2
    h -= np.where(a > 2.5, 0.55 * (a - 2.5) ** 1.35, 0.0)
    h -= np.where(X > 0, 0.004 * X ** 3, 0.0)
    h += 0.05 * fbm1_np(X / 0.9 + 3.0, 4, 2.0, 0.5, 71) + 0.35 * fbm1_np(X / 9.0 + 1.0, 4, 2.0, 0.5, 72) * (np.abs(X) > 4)
    return h


def peak_profile(x0, x1, n, base, amp, spacing, seed, sharp=1.0):
    """Himalayan skyline: max of pyramid 'tents' + a little ridged detail."""
    rng = np.random.default_rng(seed)
    X = np.linspace(x0, x1, n)
    h = np.full(n, base - 0.2 * amp)
    xs = np.arange(x0, x1, spacing) + rng.uniform(0, spacing, int(np.ceil((x1 - x0) / spacing)))
    for xp in xs:
        hp = base + amp * rng.uniform(0.25, 1.0) ** 1.4
        sl = amp / spacing * rng.uniform(0.9, 2.2) * sharp
        sl_l = sl * rng.uniform(0.7, 1.3)
        sl_r = sl * rng.uniform(0.7, 1.3)
        t = np.where(X < xp, hp - sl_l * (xp - X), hp - sl_r * (X - xp))
        h = np.maximum(h, t)
    h += 0.04 * amp * fbm1_np(X / (spacing * 0.15) + seed, 5, 2.0, 0.5, seed)
    return h


def build_world():
    ridges = []
    # (z, top angle deg, amp deg, spacing (m), albedo, mist)
    L = [(900.0, -13.0, 5.5, 520.0, 0.9, 0.3),
         (2300.0, -8.5, 3.6, 1100.0, 0.8, 0.7),
         (5200.0, -5.8, 2.6, 2200.0, 0.7, 1.0),
         (11500.0, -4.1, 1.8, 4200.0, 0.6, 1.0),
         (25000.0, -3.1, 1.25, 8000.0, 0.5, 0.8),
         (56000.0, -2.8, 0.85, 15000.0, 0.45, 0.5)]
    for i, (z, top, amp, sp, al, mist) in enumerate(L):
        x0, x1 = -1.2 * z - 200, 1.2 * z + 200
        n = int(min(20000, max(3000, (x1 - x0) / (z / 3000.0))))
        drop = z * z / (2 * R_EARTH)
        base = z * math.tan(math.radians(top)) - drop
        ampm = z * math.tan(math.radians(amp))
        h = peak_profile(x0, x1, n, base - 0.55 * ampm, ampm, 2.4 * ampm, 500 + i * 13, sharp=1.6)
        ridges.append(Ridge(z, x0, x1, h, np.array([0.004, 0.006, 0.012]) * al, fog_mul=1.0, rim=0.05,
                            mist=mist, tex=0.0, name=f'M{i}', fog_el=math.radians(2.0),
                            mist_scale=1.0 / (2.0 * ampm), seed=3.1 * i, snow=1.0))
    X = np.linspace(-60, 60, 12000)
    summit = Ridge(0.02, -60, 60, summit_profile(X), np.array([0.004, 0.006, 0.012]), fog_mul=0.0, rim=0.0,
                   mist=0.0, tex=0.0, name='summit', fog_el=math.radians(2.0), snow=1.1, seed=9.0)
    return pack_ridges(ridges), pack_ridges([summit])


def sky():
    return Sky(sun=(0.0, -40.0), zenith=np.array([0.0005, 0.0010, 0.0065]),
               horizon=np.array([0.010, 0.018, 0.050]), amber=np.zeros(3), rose=np.zeros(3),
               violet=np.array([0.004, 0.006, 0.018]), base_fall=0.30, amber_fall=0.03, rose_fall=0.05,
               violet_fall=0.3, az_pow=40.0, moon=None, mw=1.6, mw_pole=dir_from_az_el(75.0, 40.0),
               star_gain=110.0, star_thresh=0.5, n_stars=26000, ring=None, seed=7)


# --------------------------------------------------------------- animation ---

def strike_env(f):
    """Spark-light envelope: sum of the three strikes (peak ~1 on the frame, fast decay)."""
    e = 0.0
    for k, s in enumerate(STRIKES):
        d = f - s
        if d >= 0:
            e += (0.55 + 0.45 * k) * math.exp(-d / (2.2 + 0.6 * k))
    return e


def ember_env(f):
    """The spark that lives in the tinder after strike 3; pulses as she blows."""
    if f < STRIKES[2] + 1 or f >= CATCH + 4:
        return 0.0
    tr = f - STRIKES[2]
    base = 0.12 + 0.10 * smoothstep(0, 25, tr)
    puff = 0.5 + 0.5 * math.sin((f - 1296) / 7.5 * 2 * math.pi) if f > 1296 else 0.0
    return base * (1.0 + 1.3 * puff)


def flame_level(f):
    """0 before the catch; small flame 1318-1359; roar after 1360."""
    if f < CATCH:
        return 0.0
    small = smoothstep(CATCH, CATCH + 14, f) * (0.35 + 0.65 * smoothstep(CATCH + 10, ROAR, f))
    if f < ROAR:
        return small
    tr = (f - ROAR) / FPS
    return 1.0 + 3.0 * fire.ignition(tr, overshoot=0.55, rise=0.30, settle=0.9)


def yw_pose(f):
    t = f / FPS
    br = math.sin(2 * math.pi * 0.28 * t)
    # strike cycle for the near (steel) hand: raise then snap down past the flint
    hx, hy = 0.34, 1.13
    for s in STRIKES:
        d = f - s
        if -10 <= d <= 6:
            if d < 0:
                a = smoothstep(-10, -2, d)
                hy += 0.07 * a
                hx += 0.02 * a
            else:
                a = 1 - smoothstep(0, 6, d)
                hy -= 0.035 * a
                hx -= 0.015 * a
            if -2 <= d < 0:
                hy -= 0.10 * smoothstep(-2, 0, d)
    # blowing on the tinder 1294-1320: lean in, head low
    lean = smoothstep(1292, 1302, f) * (1 - smoothstep(1322, 1336, f))
    rise = smoothstep(ROAR + 2, ROAR + 30, f)
    step = smoothstep(ROAR + 18, ROAR + 44, f)
    x = YW_X + 0.30 * step
    kneel = dict(hip_y=0.50, lumbar=18 + 8 * lean, thorax=30 + 14 * lean, neck=32 + 10 * lean,
                 head=22 + 12 * lean - 10 * smoothstep(CATCH + 2, CATCH + 16, f),
                 n_th=95.0, n_sh=182.0, f_th=178.0, f_sh=-88.0)
    stand = dict(hip_y=0.92, lumbar=-4.0, thorax=-6.0, neck=4.0, head=-6.0,
                 n_th=176.0, n_sh=182.0, f_th=186.0, f_sh=178.0)
    p = {}
    for k in kneel:
        p[k] = kneel[k] + (stand[k] - kneel[k]) * rise
    p['x'] = x
    p['breath'] = br
    p['flint'] = f < ROAR
    p['hem_wind'] = 0.10
    if f < ROAR:
        p['f_tgt'] = (0.30 + 0.01 * br, 1.08)
        p['f_bend'] = 1.0
        p['f_h'] = 70.0
        p['n_tgt'] = (hx, hy)
        p['n_bend'] = 1.0
        p['n_h'] = 110.0
    else:
        # arm raised against the heat, then lowered
        u = smoothstep(ROAR + 26, ROAR + 60, f)
        p['n_ua'] = 125.0 + 50.0 * u
        p['n_fa'] = 28.0 + 145.0 * u
        p['n_h'] = 20.0 + 150.0 * u
        p['f_ua'] = 165.0 + 12.0 * u
        p['f_fa'] = 120.0 + 55.0 * u
    return p


# ------------------------------------------------ v2 choreography (3-D figure) ---
# World metres. She kneels (far knee on a low summit rock, near foot forward) at the cairn's
# right, flint in her near (left) hand over the tinder, steel in her far (right) hand.

ROCK = ((0.78, 0.06, 0.10), (0.24, 0.09, 0.18))
KNEEL = dict(
    pelvis=(0.76, 0.62, 0.0), yaw=0.0, lean=4.0, chest=2.0, twist=0.0, neck=20.0, head=24.0, head_yaw=0.0,
    head_roll=0.0, shrug=0.0,
    hand_n=(0.32, 1.08, -0.07), hand_f=(0.37, 1.16, 0.03),
    elbow_n=(0.2, -1.0, -0.6), elbow_f=(0.3, -1.0, 0.4),
    fdir_n=(-1.0, 0.25, 0.1), palm_n=(0.0, 0.0, 1.0),
    fdir_f=(-0.6, -0.3, -0.2), palm_f=(0.3, 0.2, -1.0),
    curl_n=(0.62, 0.72, 0.78, 0.84), thumb_n=0.45, curl_f=(0.92, 0.92, 0.92, 0.92), thumb_f=0.75,
    spread_n=0.0, spread_f=0.0, thumbout_n=0.0, thumbout_f=0.0,
    foot_n=(0.36, 0.05, -0.22), knee_n=(-1.0, 0.5, 0.0), toe_n=(-1.0, 0.0, 0.0), sole_n=(0.0, 1.0, 0.0),
    foot_f=(1.05, 0.13, 0.09), knee_f=(-1.0, -0.3, 0.0), toe_f=(0.3, -1.0, 0.0), sole_f=(1.0, 0.1, 0.0),
    look=(0.23, 1.13, -0.03), mouth=0.0, purse=0.0, squint=0.0, blink=0.0, brow=0.0, hem=0.1,
)


def _kp(**kw):
    p = dict(KNEEL)
    p.update(kw)
    return p


READY = _kp(hand_f=(0.41, 1.10, 0.07), look=(0.25, 1.12, -0.04))
BLOW = _kp(pelvis=(0.66, 0.67, 0.0), lean=20.0, chest=4.0, neck=6.0, head=-6.0,
           hand_n=(0.36, 0.97, -0.11), fdir_n=(-1.0, 0.3, 0.25), hand_f=(0.40, 0.97, 0.10),
           elbow_n=(0.3, -1.0, -0.4), look=(0.20, 1.09, -0.02), brow=0.3)
WATCH = _kp(pelvis=(0.78, 0.62, 0.0), lean=2.0, chest=0.0, neck=14.0, head=21.0,
            hand_n=(0.37, 1.00, -0.12), hand_f=(0.45, 0.98, 0.10), elbow_n=(0.3, -1.0, -0.4),
            look=(0.20, 1.20, -0.02), brow=0.45)
WATCH2 = _kp(pelvis=(0.77, 0.62, 0.0), lean=5.0, chest=1.0, neck=16.0, head=18.0,
             hand_n=(0.36, 1.01, -0.12), hand_f=(0.44, 0.99, 0.10), elbow_n=(0.3, -1.0, -0.4),
             look=(0.18, 1.28, -0.02), brow=0.6)
_GUARD = dict(fdir_n=(0.05, 1.0, -0.05), palm_n=(-1.0, 0.0, 0.2), curl_n=(0.12, 0.10, 0.14, 0.18), thumb_n=0.0,
              thumbout_n=0.5, spread_n=0.40, elbow_n=(0.1, -1.0, -0.6))
FLINCH = _kp(pelvis=(0.81, 0.63, 0.0), lean=-9.0, chest=-6.0, neck=8.0, head=22.0, head_yaw=16.0, shrug=1.0,
             hand_n=(0.51, 1.21, -0.11), hand_f=(0.62, 1.00, 0.08), look=(0.9, 0.9, -1.0),
             squint=1.0, blink=0.9, mouth=0.3, brow=-1.0, **_GUARD)
RISE1 = _kp(pelvis=(0.89, 0.73, 0.0), lean=13.0, chest=0.0, neck=8.0, head=20.0, head_yaw=12.0, shrug=0.7,
            hand_n=(0.60, 1.30, -0.13), hand_f=(0.72, 0.95, 0.20), foot_n=(0.52, 0.06, -0.20),
            foot_f=(1.13, 0.06, 0.10), knee_f=(-1.0, 0.2, 0.0), toe_f=(-1.0, -0.1, 0.0), sole_f=(0.0, 1.0, 0.0),
            look=(0.5, 1.3, -0.8), squint=0.8, blink=0.4, mouth=0.2, brow=-0.6, **_GUARD)
RISE2 = _kp(pelvis=(1.02, 0.90, 0.0), lean=3.0, chest=-1.0, neck=8.0, head=8.0, head_yaw=5.0, shrug=0.3,
            hand_n=(0.82, 1.42, -0.17), hand_f=(1.02, 0.84, 0.20), foot_n=(0.92, 0.12, -0.16),
            foot_f=(1.22, 0.05, 0.12), knee_n=(-1.0, 0.1, 0.0), knee_f=(-1.0, 0.1, 0.0), toe_f=(-1.0, 0.0, 0.25),
            sole_f=(0.0, 1.0, 0.0), look=(0.1, 1.9, -0.2), squint=0.4, blink=0.1, mouth=0.1, brow=-0.2, **_GUARD)
STAND1 = _kp(pelvis=(1.08, 0.93, 0.0), lean=-1.0, chest=-2.0, neck=6.0, head=-2.0, head_yaw=0.0,
             hand_n=(1.00, 0.86, -0.22), fdir_n=(-0.1, -1.0, 0.0), palm_n=(0.3, 0.0, 1.0),
             curl_n=(0.35, 0.40, 0.55, 0.60), thumb_n=0.3, spread_n=0.1, elbow_n=(0.5, -0.3, -1.0),
             hand_f=(1.17, 0.80, 0.17), foot_n=(0.98, 0.05, -0.15), foot_f=(1.24, 0.05, 0.12),
             knee_n=(-1.0, 0.05, 0.0), knee_f=(-1.0, 0.05, 0.0), toe_f=(-0.9, 0.0, 0.3), sole_f=(0.0, 1.0, 0.0),
             look=(0.0, 1.7, 0.0), squint=0.25, brow=0.2)
# she turns from the fire to the far range (three-quarters away from camera) and simply watches
STAND = _kp(pelvis=(1.10, 0.93, 0.02), yaw=-58.0, lean=-1.0, chest=-2.0, neck=4.0, head=-3.0, head_yaw=-18.0,
            hand_n=(1.06, 0.84, -0.18), fdir_n=(0.0, -1.0, 0.1), palm_n=(0.6, 0.0, 0.8),
            curl_n=(0.35, 0.40, 0.55, 0.60), thumb_n=0.3, spread_n=0.1, elbow_n=(0.4, -0.3, -1.0),
            hand_f=(1.20, 0.83, 0.26), fdir_f=(0.0, -1.0, 0.0), palm_f=(-0.5, 0.0, -0.8), elbow_f=(0.3, -0.3, 1.0),
            curl_f=(0.6, 0.65, 0.7, 0.7), thumb_f=0.4,
            foot_n=(1.00, 0.05, -0.12), foot_f=(1.22, 0.05, 0.17), knee_n=(-0.5, 0.05, 0.8),
            knee_f=(-0.5, 0.05, 0.8), toe_n=(-0.5, 0.0, 0.85), toe_f=(-0.5, 0.0, 0.85), sole_f=(0.0, 1.0, 0.0),
            look=(-40.0, -2.0, 80.0), squint=0.2, brow=0.1)

V2_KEYS = [(1200, READY, 'smooth'), (1222, KNEEL, 'smooth'), (1293, KNEEL, 'smooth'), (1300, BLOW, 'io'),
           (1318, BLOW, 'smooth'), (1330, WATCH, 'io'), (1352, WATCH2, 'smooth'), (1360, WATCH2, 'smooth'),
           (1364, FLINCH, 'out3'),
           (1371, RISE1, 'io'), (1381, RISE2, 'io'), (1393, STAND1, 'io'), (1418, STAND, 'io'),
           (1439, STAND, 'smooth')]
BLOWS = ((1297, 1307), (1309, 1318))
BLINKS = (1209, 1247, 1273, 1334, 1351, 1402, 1427)
# exhales: (onset frame, duration s, kind) - lit only by what light is about (strikes, ember, flame)
BREATHS = ((1212, 1.6, 'out'), (1252, 1.7, 'out'), (1281, 1.5, 'out'), (1297, 0.55, 'blow'), (1309, 0.50, 'blow'),
           (1320, 1.8, 'out'), (1339, 1.7, 'out'), (1352, 1.4, 'out'))


def _mixp(a, b, u):
    out = {}
    for k, va in a.items():
        vb = b.get(k, va)
        if isinstance(va, (tuple, list, np.ndarray)):
            out[k] = np.asarray(va, np.float64) * (1 - u) + np.asarray(vb, np.float64) * u
        else:
            out[k] = va * (1 - u) + vb * u
    return out


def blow_env(f):
    """0..1 while she blows on the ember (two long, gentle breaths)."""
    e = 0.0
    for (a, b) in BLOWS:
        e = max(e, smoothstep(a, a + 2.5, f) * (1 - smoothstep(b - 3, b, f)))
    return e


def yw2_pose(f):
    from core import EASES
    ks = V2_KEYS
    if f <= ks[0][0]:
        p = dict(ks[0][1])
    elif f >= ks[-1][0]:
        p = dict(ks[-1][1])
    else:
        for i in range(1, len(ks)):
            if f <= ks[i][0]:
                u = EASES[ks[i][2]]((f - ks[i - 1][0]) / (ks[i][0] - ks[i - 1][0]))
                p = _mixp(ks[i - 1][1], ks[i][1], u)
                break
    t = f / FPS
    hf = np.asarray(p['hand_f'], np.float64).copy()
    hn = np.asarray(p['hand_n'], np.float64).copy()
    lean_add = 0.0
    # strikes: the steel's bar is driven along an explicit path relative to the flint's edge -
    # rest, wind up, snap down scraping the edge exactly on the strike frame, follow through,
    # recover. The wrist target is solved from the bar's offset inside the (fixed) grip.
    for k, s in enumerate(STRIKES):
        d = f - s
        if 0 <= d < 5:
            hn += np.array([0.002, -0.006, 0.0]) * (1 - d / 5.0)
            lean_add += 1.2 * (1 - d / 5.0)
    near_strike = [s for s in STRIKES if -13 <= f - s < 15]
    if near_strike and f < STRIKES[-1] + 15:
        import heroine as hero
        s = near_strike[0]
        k = STRIKES.index(s)
        d = f - s + 0.5            # the sparks are born on the half frame before each strike frame
        amp = 1.0 + 0.25 * k
        q = dict(p)
        q['hand_n'] = hn
        q['hand_f'] = hf
        fl = hero.cheap_anchors(q)['flint']
        J = hero.skeleton(q)
        Wf, a_, n_, sd_ = hero.hand_frame(q, J, 'f')
        off = a_ * 0.0969 + n_ * 0.0688 + sd_ * 0.0104
        rest = hf + off
        R = fl + np.array([0.050, 0.105, 0.0]) * amp
        Cc = fl + np.array([0.006, 0.010, 0.0])
        Fo = fl + np.array([-0.022, -0.075, 0.0])
        if -13 <= d < -3:
            u = smoothstep(-13, -4, d)
            S = rest * (1 - u) + R * u
        elif -3 <= d < 0:
            u = ((d + 3) / 3.0) ** 2
            S = R * (1 - u) + Cc * u
        elif 0 <= d < 4:
            u = 1 - (1 - d / 4.0) ** 2
            S = Cc * (1 - u) + Fo * u
        else:
            u = smoothstep(4, 15, d)
            S = Fo * (1 - u) + rest * u
        hf = S - off
    # the guard (1360+): an open hand up between the heat and her face, following her head as
    # she recoils and rises; lowered as she finds her feet (a working gesture, nothing held aloft)
    gw = smoothstep(ROAR, ROAR + 3, f) * (1 - smoothstep(ROAR + 16, ROAR + 30, f))
    if gw > 0:
        import heroine as hero
        q = dict(p)
        q['hand_n'] = hn
        q['hand_f'] = hf
        J = hero.skeleton(q)
        Fh = hero.head_frame(J['atlas'], J['hf'], J['hu'])
        guard = Fh.p(0.07, 0.0, 0.0) + np.array([-0.12, -0.065, -0.07])
        hn = hn * (1 - gw) + guard * gw
    p['hand_f'] = hf
    p['hand_n'] = hn
    # blowing: lips purse, the face eases toward the ember on each breath
    be = blow_env(f)
    p['purse'] = max(p['purse'], be)
    p['head'] = p['head'] + 3.0 * be
    lean_add += 1.5 * be
    # breathing + idle sway (never frozen)
    br = math.sin(2 * math.pi * t / 3.3)
    p['breath'] = br * (1 - 0.6 * be)
    p['lean'] = p['lean'] + lean_add + 0.6 * fnoise1(t * 0.35, 41.0, 2)
    p['head'] = p['head'] + 1.2 * fnoise1(t * 0.5, 43.0, 2)
    p['head_roll'] = p['head_roll'] + 1.5 * fnoise1(t * 0.3, 44.0, 2)
    # blinks (and a reflex blink at each strike flash)
    bl = 0.0
    for b in BLINKS + tuple(s + 1 for s in STRIKES):
        d = f - b
        if 0 <= d < 5:
            bl = max(bl, (1.0, 1.0, 0.8, 0.4, 0.1)[int(d)])
    p['blink'] = max(p['blink'], bl)
    p['rock'] = ROCK
    # she lets the flint go as she flinches (a 3 cm stone, gone inside the fast guard move)
    p['tools'] = 'both' if f < ROAR + 1 else 'steel'
    p['hem'] = 0.10 + 0.25 * smoothstep(ROAR, ROAR + 20, f)
    look = np.asarray(p.pop('look'), np.float64)
    p['expr'] = dict(mouth=p.pop('mouth'), purse=p.pop('purse'), squint=p.pop('squint'),
                     blink=min(1.0, p.pop('blink')), brow=p.pop('brow'))
    p['look_at'] = look
    return p


def ember_env2(f):
    """v2 ember: glows up on each of her two breaths; flares as the kindling catches."""
    if f < STRIKES[2] + 1 or f >= CATCH + 6:
        return 0.0
    tr = f - STRIKES[2]
    base = 0.10 + 0.06 * smoothstep(0, 20, tr)
    flare = 1.0 + 1.8 * smoothstep(CATCH - 2, CATCH + 1, f) * (1 - smoothstep(CATCH + 2, CATCH + 6, f))
    return base * (1.0 + 2.2 * blow_env(f - 1)) * flare * (1.0 + 0.15 * fnoise1(f / FPS * 6.0, 9.0, 2))


def camera(f, scale):
    t = f / FPS
    hand = 0.004 * fnoise1(t * 0.9, 1.0) , 0.003 * fnoise1(t * 0.8, 2.0)
    c_pos = np.array([0.42 + hand[0], 0.96 + hand[1], -2.15])
    c_tgt = np.array([0.40, 1.02, 0.0])
    if HEROINE_V2:
        w_pos = np.array([1.50, 5.2, -46.0])
        w_tgt = np.array([0.40, 0.0, 60.0])
    else:
        w_pos = np.array([1.00, 2.3, -23.0])
        w_tgt = np.array([0.30, 0.9, 40.0])
    u = smoothstep(ROAR, F1 + 6, f)
    u = 1 - (1 - u) ** 3            # explosive start, long deceleration
    pos = c_pos + (w_pos - c_pos) * u
    tgt = c_tgt + (w_tgt - c_tgt) * (u ** 0.8)
    hfov = 40.0 + (26.0 if HEROINE_V2 else 18.0) * u
    cam = Camera(pos, hfov=hfov, scale=scale)
    yaw, pitch = cam.look_at(tgt)
    focus = float(np.linalg.norm(np.array([0.4, 1.0, 0.0]) - pos))
    return Camera(pos, yaw=yaw, pitch=pitch, hfov=hfov, scale=scale), focus, u


_HA = {}


def hero_anchors(ff):
    """Cheap v2 rig anchors at (fractional) frame ff, memoised."""
    key = round(ff * 64) / 64.0
    r = _HA.get(key)
    if r is None:
        import heroine as hero
        r = hero.cheap_anchors(yw2_pose(key))
        if len(_HA) > 20000:
            _HA.clear()
        _HA[key] = r
    return r


# ------------------------------------------------------------------ shot ---

class FirstBeacon:
    def __init__(self):
        self.far, self.summit = build_world()
        self.sky = sky()
        self._sim = False

    def simulate(self):
        if self._sim:
            return
        rng = np.random.default_rng(21)

        def wbase(ff):
            return 7.0 + 2.5 * fnoise1(ff / FPS * 0.4, 3.0, 2)

        def anchor(ff):
            if HEROINE_V2:
                return hero_anchors(ff)['scarf_anchor'][:2]
            _, an = ch.young_woman(yw_pose(ff), ff / FPS, anchors_only=True)
            return an['scarf_anchor']
        if HEROINE_V2:
            # heavy wool whipping in hard wind: less drag, lower mean wind, strong travelling flutter
            self.scarf = Chain(14, 0.078, anchor, dir0=(1.0, -0.2), drag=4.2, iters=6, damp=0.99)
            self.scarf.simulate(F0, F1, make_wind_fn(1.0, 0.45, 5.0, lift=0.15, flutter=4.4,
                                                     extra=lambda ff: 0.72 * wbase(ff) - 1.0))
        else:
            self.scarf = Chain(14, 0.078, anchor, dir0=(1.0, 0.0), drag=6.0, iters=6, damp=0.99)
            self.scarf.simulate(F0, F1, make_wind_fn(1.0, 0.35, 5.0, lift=0.8, flutter=2.6,
                                                     extra=lambda ff: wbase(ff) - 1.0))
        self.hair = []
        for k in range(9):
            def ha(ff, k=k):
                if HEROINE_V2:
                    return hero_anchors(ff)['hair_root'][:2] + np.array([-0.010 + 0.004 * k, 0.030 - 0.010 * k])
                _, an = ch.young_woman(yw_pose(ff), ff / FPS, anchors_only=True)
                return an['hair_root'] + np.array([-0.01 + 0.004 * k, 0.03 - 0.012 * k])
            c = Chain(8, 0.045 + 0.006 * (k % 3), ha, dir0=(1.0, -0.1), drag=7.0, iters=4, damp=0.98, grav=4.0)
            c.simulate(F0, F1, make_wind_fn(1.0, 0.5, 20.0 + k, lift=0.5, flutter=(3.6 if HEROINE_V2 else 2.2),
                                            extra=lambda ff: wbase(ff) - 1.0))
            self.hair.append(c)
        # strike sparks (hot, fast, gravity) + beacon sparks/embers (buoyant)
        self.sparks = fire.ParticleSim(seed=31, cap=3000)

        def emit(sim, ff, dt):
            for k, s in enumerate(STRIKES):
                if s - 0.5 <= ff < s + 0.5:
                    n = int((70, 120, 220)[k] * dt * FPS * 1.0)
                    p = hero_anchors(s)['flint'] if HEROINE_V2 else ch.young_woman(yw_pose(s), s / FPS)[1]['flint']
                    pos = np.array([p[0] - 0.01, p[1], (p[2] if HEROINE_V2 else -0.01)]) + rng.normal(0, 0.004, (n, 3))
                    ang = rng.normal(-2.2, 0.55, n)          # mostly down-left into the basket
                    sp = rng.gamma(3.0, 0.9, n) + 0.6
                    vel = np.stack([np.cos(ang) * sp, np.sin(ang) * sp + 0.4, rng.normal(0, 0.5, n)], 1)
                    sim.spawn(pos, vel, rng.uniform(0.12, 0.55, n), rng.random(n) ** 2.2 * 2.0 + 0.3,
                              np.zeros(n, np.int64), 1.0)
            lv = flame_level(ff)
            if lv > 0:
                rate = 6.0 + 26.0 * max(0.0, lv - 1.0) + 90.0 * smoothstep(ROAR, ROAR + 4, ff) * (1 - smoothstep(ROAR + 10, ROAR + 30, ff))
                n = rng.poisson(rate * dt)
                if n > 0:
                    base = FIRE_BASE if ff >= ROAR else TINDER
                    spread = 0.20 if ff >= ROAR else 0.03
                    pos = base + np.stack([rng.normal(0, spread, n), rng.uniform(0.1, 0.6 if ff >= ROAR else 0.1, n),
                                           rng.normal(0, spread * 0.5, n)], 1)
                    vel = np.stack([rng.normal(0.5, 0.5, n), rng.uniform(1.0, 3.5, n), rng.normal(0, 0.3, n)], 1)
                    kind = (rng.random(n) > 0.3).astype(np.int64)
                    sim.spawn(pos, vel, rng.gamma(2.0, 0.6, n) + 0.3, rng.random(n) ** 2.5 * 1.8 + 0.2, kind,
                              np.where(kind == 0, 1.0, rng.uniform(0.6, 0.85, n)))

        def params(ff):
            return dict(wind=(wbase(ff) * 0.5, 0.0, 0.0), buoy=2.8, drag=1.2, turb=0.9, turb_sc=1.6,
                        grav=2.5, cool=1.0)

        self.sparks.run(F0, F1, emit, params, substeps=4)
        # spindrift snow grains near the camera path
        self.snow = rng.uniform([-6, -1, -8], [8, 7, 6], (2600, 3))
        self.snow_ph = rng.random(2600)
        self._sim = True

    def render(self, f, scale=0.5):
        self.simulate()
        t = f / FPS
        cam, focus, u = camera(f, scale)
        if HEROINE_V2:
            # dark adaptation: before the kindling catches the eye is used to the night, so the
            # moonlit summit and her cold rim are faintly there; the fire pulls it down; the
            # reveal brings the world back up
            floor = 0.17 - 0.13 * smoothstep(CATCH, CATCH + 14, f)
            reveal = floor + (1.0 - floor) * smoothstep(ROAR + 4, ROAR + 58, f)
        else:
            reveal = 0.03 + 0.97 * smoothstep(ROAR + 4, ROAR + 58, f)
        lv = flame_level(f)
        flick = fire.flicker(t, 3.3, 1.3)
        st_e = strike_env(f)
        em_e = ember_env2(f) if HEROINE_V2 else ember_env(f)
        # --- lights
        lights = []
        if HEROINE_V2:
            flint = hero_anchors(f)['flint']
        else:
            flint = ch.young_woman(yw_pose(f), t)[1]['flint']
        if st_e > 0.01:
            lights.append([flint[0] - 0.05, flint[1] - 0.05, -0.08, 3.2 * st_e, 2.6 * st_e, 1.9 * st_e, 0.30, 0.0])
        if em_e > 0:
            lights.append([TINDER[0], TINDER[1], -0.05, 1.4 * em_e, 0.45 * em_e, 0.10 * em_e, 0.18, 0.0])
        if lv > 0:
            if f < ROAR:
                I = lv * flick
                lights.append([TINDER[0], TINDER[1] + 0.06, -0.06, 2.6 * I, 1.25 * I, 0.40 * I, 0.35, 0.0])
            else:
                I = min(lv, 2.5) * flick
                lights.append([FIRE_BASE[0], FIRE_BASE[1] + 0.6, -0.1, 6.0 * I, 2.8 * I, 0.9 * I, 1.1, 0.0])
        lights = np.array(lights, np.float64).reshape(-1, 8)
        # --- sky + range (moonlit world scaled by the reveal)
        sk = self.sky
        img = render_full_sky(cam, sk, t, mw_col=np.array([0.80, 0.86, 1.0]) * 0.07) * (0.25 + 0.75 * reveal)
        fog = FOG.copy()
        if TRUE_PEAKS:
            # one ray-marched pass: far range + cloud sea (x reveal) and the summit (dim in the
            # close-up, full after the reveal), with the fire's warm pool on the summit snow
            fl = []
            if lv > 0 and f >= ROAR:
                I = min(lv, 2.5) * flick
                fl.append([FIRE_BASE[0], FIRE_BASE[1] + 0.45, FIRE_BASE[2] - 0.05, 0.11 * I, 0.047 * I, 0.014 * I, 1.5, 0])
            elif lv > 0:
                fl.append([TINDER[0], TINDER[1], TINDER[2] - 0.05, 0.035 * lv, 0.015 * lv, 0.005 * lv, 0.7, 0])
            pp = None
            if HEROINE_V2 and getattr(peaks, 'S1_WORLD', False):
                # the shepherd's world: s1's snow-hold rule, rock albedo and a softer moon
                pp = peaks.PARAMS.copy()
                pp[4] = 0.55
                pp[16] = 0.48
                pp[18] = 0.055
            trgb, ta = peaks.render(cam, sk.packed(), t, fl, reveal, 0.15 + 0.85 * reveal, ss=(2 if scale < 0.75 else 1.3),
                                    params=pp)
            over(img, trgb, ta)
        else:
            rs, meta, hs = self.far
            rgbp = np.zeros_like(img)
            a = np.zeros(img.shape[:2], np.float32)
            dep = np.full(img.shape[:2], 1e9, np.float32)
            render_ridges(rgbp, a, dep, cam.params(), meta, hs, sk.packed(), fog, np.zeros((0, 8)), np.zeros(1), 1.0, t)
            over(img, rgbp * reveal, a)
            # summit snow (+ warm pool from the fire)
            pool = []
            if lv > 0 and f >= ROAR:
                I = min(lv, 2.5) * flick
                pool.append([FIRE_BASE[0], 0.0, 0.02, 1.7, 0.06 * I, 0.026 * I, 0.008 * I, 0])
            elif lv > 0:
                pool.append([TINDER[0], 0.0, 0.02, 0.8, 0.02 * lv, 0.009 * lv, 0.003 * lv, 0])
            rs2, meta2, hs2 = self.summit
            srgb = np.zeros_like(img)
            sa = np.zeros(img.shape[:2], np.float32)
            sd = np.full(img.shape[:2], 1e9, np.float32)
            render_ridges(srgb, sa, sd, cam.params(), meta2, hs2, sk.packed(), fog,
                          np.array(pool, np.float64).reshape(-1, 8), np.zeros(1), 1.0, t)
            # the summit keeps a little moonlight even in the close-up (dim), full after reveal
            over(img, srgb * (0.15 + 0.85 * reveal), sa)
        # DOF on background in the close-up
        coc = 0.012 * cam.f / max(focus, 0.3) * (1 - u) ** 2
        if coc * 0.42 > 0.8:
            img = cv2.GaussianBlur(img, (0, 0), coc * 0.42)
        # --- beacon smoke (behind her)
        if f >= ROAR:
            sm_rgb = np.zeros_like(img)
            sm_a = np.zeros(img.shape[:2], np.float32)
            b = FIRE_BASE + np.array([0.0, 0.9, 0.05])
            pts = np.array([b + np.array([-1.0, -0.2, 0]), b + np.array([9.0, 8.0, 0])])
            sx, sy, z = cam.project(pts)
            if np.all(z > 0.05):
                I = min(lv, 2.5) * flick
                fire.render_smoke(sm_rgb, sm_a, cam.params(), float(b[0]), float(b[1]), float(b[2]), t, 1.7,
                                  7.0, 0.25, 0.30, 0.9, 0.9, 0.55 * smoothstep(ROAR, ROAR + 12, f),
                                  0.30 * I, 0.11 * I, 0.03 * I, 0.9, 0.004, 0.005, 0.009,
                                  int(min(sx)) - 20, int(min(sy)) - 20, int(max(sx)) + 20, int(max(sy)) + 20)
                over(img, sm_rgb, sm_a)
        # --- cairn + woman
        amb_top = np.array([0.010, 0.016, 0.040]) * reveal
        back = np.array([0.03, 0.05, 0.12]) * (0.3 + 0.7 * reveal)
        occ = None
        if HEROINE_V2:
            r = Figure([0.0, 0.0, 0.0], CAIRN.groups(x=0.0)).render(cam, lights, amb_top, np.zeros(3), back=back)
            if r is not None:
                over_region(img, *r)
            her = self._heroine(f, t, cam, scale, lv, flick, st_e, em_e, flint, reveal)
            ya = dict(mouth=hero_anchors(f)['mouth'])
            occ = np.zeros(img.shape[:2], np.float32)
            if her is not None:
                y0, x0, hrgb, hal, hdep = her
                over_region(img, y0, x0, hrgb, hal)
                # her pixels in front of the fire hide it (hands round the ember / flame)
                d_fl = float(np.linalg.norm((TINDER if f < ROAR else FIRE_BASE + np.array([0, 0.5, 0])) - cam.pos))
                hh, ww = hal.shape
                occ[y0:y0 + hh, x0:x0 + ww] = np.where(hdep < d_fl - 0.02, hal, 0.0)
        else:
            yp = yw_pose(f)
            hair = [c.at(f) for c in self.hair]
            yg, ya = ch.young_woman(yp, t, scarf_pts=self.scarf.at(f), hair_pts=hair)
            items = [Figure([0.0, 0.0, 0.0], CAIRN.groups(x=0.0)), Figure([0.0, 0.0, -0.01], yg)]
            blur = 0.0
            for it in items:
                r = it.render(cam, lights, amb_top, np.zeros(3), back=back, blur_px=blur)
                if r is not None:
                    y0, x0, rgb, al = r
                    over_region(img, y0, x0, rgb, al)
        # --- fire
        fa = np.zeros(img.shape[:2], np.float32)
        if lv > 0:
            fimg = np.zeros_like(img) if occ is not None else img
            if f < ROAR:
                h = 0.05 + 0.20 * lv
                fire.draw_flame(fimg, fa, cam, TINDER + np.array([0.0, -0.01, 0.0]), h, 0.028 + 0.05 * lv, 0.45,
                                t, 4.4, 6.5 * flick * (0.6 + 0.4 * lv), fire.TORCH_STYLE)
            else:
                grow = min(lv, 2.2)
                fire.draw_flame(fimg, fa, cam, FIRE_BASE, 0.55 + 0.75 * grow, 0.31, 0.55, t, 2.9,
                                5.0 + 5.0 * min(1.0, lv - 1.0 + 0.3), fire.BONFIRE_STYLE)
            if occ is not None:
                img += fimg * (1.0 - occ[..., None])
            fire.add_glow(img, cam, (TINDER if f < ROAR else FIRE_BASE + np.array([0, 0.7, 0])),
                          0.12 if f < ROAR else 0.9, (0.05 * lv if f < ROAR else 0.10 * min(lv, 2.5)) * flick)
        for k, s_ in enumerate(STRIKES):
            d = f - s_
            if 0 <= d <= 2 and flint is not None:
                fx, fy, fz = cam.project(np.array([flint[0], flint[1], -0.02]))
                e = (60.0 + 40.0 * k) * (1.0, 0.45, 0.15)[d] * cam.scale ** 2
                splat_gauss(img, float(fx), float(fy), max(0.6, 2.5 * cam.scale), e, e * 0.85, e * 0.6)
                fire.add_glow(img, cam, np.array([flint[0], flint[1], -0.02]), 0.06, (0.25 + 0.15 * k) * (1.0, 0.45, 0.15)[d])
        if em_e > 0:
            sx, sy, z = cam.project(TINDER)
            e = 6.0 * em_e * cam.scale ** 2
            eimg = np.zeros_like(img) if occ is not None else img
            splat_gauss(eimg, float(sx), float(sy), max(0.5, 2.0 * cam.scale), e, e * 0.35, e * 0.06)
            if occ is not None:
                img += eimg * (1.0 - occ[..., None])
            fire.add_glow(img, cam, TINDER, 0.05, 0.06 * em_e)
        # --- breath vapour after the catch (lit by the small flame)
        if HEROINE_V2:
            self._breath2(img, cam, f, lv, em_e, st_e, flint)
        elif CATCH <= f < ROAR + 10:
            self._breath(img, cam, f, ya['mouth'], lv)
        # --- sparks & embers
        P, V, T, S, K = self.sparks.snap[f]
        ap = 0.010 * cam.f * (1 - u)
        fire.render_sparks(img, cam.params(), P, V, T, S, K, 0.5 / FPS, 4.0 * cam.scale ** 2 * 6.0,
                           cam.scale * 1.4, focus, ap, 1.0, 1.4, 0.05)
        # --- spindrift grains
        self._spindrift(img, cam, f, reveal, lv)
        expo = 1.05
        out = look.finish(img, exposure=expo, bloom_strength=0.09, bloom_threshold=0.9, vignette_amount=0.25)
        return out

    def _heroine(self, f, t, cam, scale, lv, flick, st_e, em_e, flint, reveal):
        """The v2 Young Woman: 3-D figure, lit physically by the strike, ember, flame and roar."""
        import heroine as hero
        pose = yw2_pose(f)
        hair = [c.at(f) for c in self.hair]
        B, F, H, an = hero.build_figure(pose, t, scarf_pts=self.scarf.at(f), hair_pts=hair)
        L = []
        if st_e > 0.01:
            # the spark shower at the flint's edge: white-hot, brief, an extended source
            L.append([flint[0] + 0.015, flint[1] + 0.015, flint[2] - 0.045,
                      0.075 * st_e, 0.060 * st_e, 0.042 * st_e, 0.05, -10.0])
        if em_e > 0:
            L.append([TINDER[0], TINDER[1] + 0.006, TINDER[2] - 0.02, 0.070 * em_e, 0.023 * em_e, 0.0050 * em_e,
                      0.010, -12.0])
        if lv > 0:
            if f < ROAR:
                I = lv * flick
                L.append([TINDER[0], TINDER[1] + 0.03 + 0.06 * lv, TINDER[2] - 0.02,
                          0.30 * I, 0.135 * I, 0.036 * I, 0.03 + 0.04 * lv, -8.0])
            else:
                I = min(lv, 2.5) * flick
                L.append([FIRE_BASE[0], FIRE_BASE[1] + 0.55, FIRE_BASE[2] - 0.05, 0.75 * I, 0.34 * I, 0.095 * I,
                          0.28, 3.0])
        if reveal > 0.05:
            # moonlight (directional, no shadow ray): arrives as the world is revealed
            d = peaks.MOON_L
            mi = 0.30 * (reveal - 0.03) * 1e4
            c = np.array([0.55, 0.70, 0.95]) * mi
            p0 = np.array([0.9, 1.0, 0.0]) + d * 100.0
            L.append([p0[0], p0[1], p0[2], c[0], c[1], c[2], 0.0, 0.0])
        warm = (0.9 * lv * flick if f < ROAR else 0.35 * min(lv, 2.5) * flick)
        env = hero.env_vec(rim_dir=(0.55, 0.42, 0.72), rim=np.array([0.070, 0.100, 0.180]) * (1.0 + 1.0 * reveal),
                           amb=np.array([0.0035, 0.0050, 0.0100]) * (1.0 + 4.0 * reveal),
                           bounce=np.array([0.030, 0.012, 0.004]) * warm, ao=0.02)
        L = np.array(L, np.float64).reshape(-1, 8)
        res = hero.render(cam, B, H, L, env, ss=(3 if scale > 0.75 else 2))
        # fine hair strands along the simulated locks, and the lashes
        sets = [hero.hair_strands(cam, hair, F, t, L, env), hero.flyaways(cam, F, t, L, env),
                hero.lashes(cam, F, pose.get('expr'), L, env)]
        return hero.draw_fine(res, cam, sets)

    def _breath2(self, img, cam, f, lv, em_e, st_e, flint):
        """Breath fog. Each exhale leaves the lips as a thin plume that scatters whatever light
        reaches it (strike flash, ember, flame, fire); while she blows, it jets down to the
        tinder. Additive, noise-modulated puffs; invisible in the dark, as breath is."""
        import heroine_sdf as hsdf
        srcs = []
        if lv > 0 and f < ROAR:
            srcs.append((TINDER + np.array([0, 0.03 + 0.06 * lv, 0]), 0.30 * lv, np.array([1.0, 0.45, 0.12])))
        elif lv > 0:
            srcs.append((FIRE_BASE + np.array([0, 0.55, 0]), 0.75 * min(lv, 2.5), np.array([1.0, 0.45, 0.13])))
        if em_e > 0:
            srcs.append((TINDER, 0.070 * em_e, np.array([1.0, 0.33, 0.07])))
        if st_e > 0.01:
            srcs.append((flint, 0.075 * st_e, np.array([1.0, 0.8, 0.56])))
        if not srcs:
            return
        for (fs, dur, kind) in BREATHS:
            age = (f - fs) / FPS
            if age < 0 or age > dur:
                continue
            an = hero_anchors(fs)
            mouth = an['mouth']
            fwd = an['head'].U
            if kind == 'blow':
                tgt = TINDER + np.array([0.005, 0.012, 0.0])
                n_sub = 34
            else:
                n_sub = 26
            for q in range(n_sub):
                # sub-puffs released over the first part of the exhale, each ageing on its own
                rel = q / n_sub * (0.45 if kind != 'blow' else 0.40)
                a_ = age - rel
                if a_ <= 0 or a_ > (dur - rel):
                    continue
                if kind == 'blow':
                    d = tgt - mouth
                    k = min(1.0, a_ / 0.16)
                    pos = mouth + d * (1 - (1 - k) ** 2) + np.array([0.0, 0.05, 0.0]) * max(0.0, a_ - 0.16)
                    rad_m = 0.010 + 0.030 * min(a_, 0.5)
                    dens = 0.10 * math.exp(-a_ / 0.35)
                else:
                    v0 = fwd * 0.55 + np.array([0.0, -0.08, 0.0])
                    pos = mouth + v0 * (1 - math.exp(-a_ / 0.25)) * 0.25 \
                        + np.array([0.42, 0.10, 0.0]) * a_ * a_ * 0.8 + np.array([0.20, 0.02, 0.0]) * a_
                    rad_m = 0.008 + 0.045 * a_
                    dens = 0.085 * math.exp(-a_ / 0.6) * min(1.0, a_ / 0.08)
                sx, sy, z = cam.project(pos)
                if z <= 0.05:
                    continue
                Ls = np.zeros(3)
                for (sp, I, c) in srcs:
                    d2 = float(np.sum((pos - sp) ** 2)) + 0.003
                    Ls += c * (I / d2)
                Ls *= 0.55 * (1.0 - smoothstep(ROAR - 1, ROAR + 1, f))
                sig = max(0.8, cam.f * rad_m / z * 0.6)
                hsdf.splat_fog(img, float(sx), float(sy), sig, dens, Ls[0], Ls[1], Ls[2],
                               float(fs * 0.37), f / FPS * 0.8, max(2.0, sig * 1.1))

    def _breath(self, img, cam, f, mouth, lv):
        rng = np.random.default_rng(3)
        for k in range(4):
            f0 = CATCH + 4 + k * 11
            age = (f - f0) / FPS
            if age < 0 or age > 1.6:
                continue
            for q in range(6):
                off = rng.normal(0, 1, 2)
                p = np.array([mouth[0] + 0.55 * age + 0.02 * off[0] + 0.05 * q * age,
                              mouth[1] + 0.05 * age + 0.015 * off[1], -0.02])
                sx, sy, z = cam.project(p)
                if z <= 0:
                    continue
                rad = cam.f * (0.012 + 0.05 * age) / z
                fade = math.exp(-age / 0.6) * (1 - math.exp(-age / 0.08))
                e = 0.035 * fade * (0.4 + lv) * rad * rad
                splat_gauss(img, float(sx), float(sy), max(0.6, rad * 0.6), e * 1.0, e * 0.62, e * 0.40)

    def _spindrift(self, img, cam, f, reveal, lv):
        t = f / FPS
        P = self.snow.copy()
        speed = 7.5
        P[:, 0] = (P[:, 0] + speed * t + 3.0 * self.snow_ph) % 14.0 - 6.0
        P[:, 1] += 0.25 * np.sin(t * 2.0 + self.snow_ph * 20)
        V = np.zeros_like(P)
        V[:, 0] = speed
        sx, sy, z = cam.project(P)
        tx, ty, _ = cam.project(P - V * 0.5 / FPS)
        m = (z > 0.2) & (sx > -20) & (sx < cam.W + 20) & (sy > -20) & (sy < cam.H + 20)
        if not np.any(m):
            return
        d_fire = np.linalg.norm(P - (FIRE_BASE if f >= ROAR else TINDER), axis=1)
        warm = (0.08 + 1.2 * min(lv, 2.0)) / (1.0 + (d_fire / (0.5 + 0.6 * (f >= ROAR))) ** 2)
        cool = 0.03 * reveal
        e = (warm + cool)[m] * cam.scale ** 2 * 1.2 / np.maximum(z[m], 0.5)
        cols = np.stack([e * (0.25 + 0.75 * (warm[m] > cool)) + e * 0.2, e * 0.75, e * 0.7], 1)
        import fire as _f
        from core import splat_particles
        sig = np.full(m.sum(), max(0.35, 0.45 * cam.scale))
        splat_particles(img, tx[m].astype(np.float64), ty[m].astype(np.float64), sx[m].astype(np.float64),
                        sy[m].astype(np.float64), sig, cols.astype(np.float64))
