"""B13 . THE HAND-BACK (R11) in motion: frames 3840-5199 of cut B (bars 49-65), one take.

  3840-4079  THE CRANE (R10 folded in): from the vigil's locked frame the camera rises and swings round her summit,
             every beacon of the range alight beyond the two figures, and settles high behind her (bar 52).
             Per-frame march, night light.
  4080-5199  SETTLED (locked). The world is marched once (relight.build: G-buffer + each pixel's clearance toward the
             sun's azimuth) and relit every frame: the night plate until the east greys (bar 53), then the sun's
             elevation e(f) solved from the bar map. First light strikes the far skyline on bar 54 b1; the light walks
             across the range toward her and six chosen far beacons pale on bars 55-60 (each by its own clearance:
             B's paling rule); the disc breaks over the nearer shoulder's edge just before the light reaches her; her
             beacon pales last on bar 61 b1; she opens her hands; the sun reaches the child and the red scarf on bar
             63 b1; the child wakes on bar 64 b1.
Figures are greybox puppets (mt/figure): HILLS' H6 pass replaces them (`--no-figures` renders the plate for comp).

  python shots/run/handback_motion.py --frames 3840,4080,4300,4800,5100 --scale 0.25
  python shots/run/handback_motion.py --range 3840-5199 --scale 1.0 --out renders/handback_B --procs 4 --skip
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import keeper as KP         # noqa: E402
import relight as RL        # noqa: E402
import fire2 as F2          # noqa: E402
from mt import fire as F, figure as FG, props as PR, sky as SK   # noqa: E402
from mt.noise import smoothstep   # noqa: E402

CM = PI.CM
look = PI.look
CR = KP.CR_B
FPS = 24.0
F0, F1 = 3840, 5200
F_SETTLE, F_GREY, F_FIRST, F_HER = 4080, 4160, 4240, 4800
PALE = [4320, 4400, 4480, 4560, 4640, 4720]
F_HANDS, F_CHILD_SUN, F_TOUCH, F_WAKE = 4840, 4960, 5000, 5040


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


# ------------------------------------------------------------------ the set ---
def _shoulder():
    x, z = KP.S_SUM[0] + 1920.0, KP.S_SUM[2] - 1260.0
    h = WD.h_rock(x, z, 2.0, CR)
    step = 20.0
    for _ in range(80):
        imp = False
        for dx, dz in ((step, 0), (-step, 0), (0, step), (0, -step)):
            hh = WD.h_rock(x + dx, z + dz, 2.0, CR)
            if hh > h:
                h, x, z, imp = hh, x + dx, z + dz, True
        if not imp:
            step *= 0.5
            if step < 0.5:
                break
    return np.array([x, h, z])


SHOULDER = _shoulder()
SEAT = KP.on_ground(KP.STAND + KP.RIGHT * 0.25)             # where she sits at the end of the sixtieth night
_d = SHOULDER - SEAT
SUN_BASE_AZ = math.degrees(math.atan2(_d[0], _d[2]))       # her line to the shoulder's summit
SUN_AZ = None                                              # set by _sun_az(): just right of the summit, on the flank


def skyline_el(az_deg, eye):
    a = math.radians(az_deg)
    d = np.array([math.sin(a), math.cos(a)])
    best = -10.0
    for r in np.geomspace(20.0, 120000.0, 1100):
        x, z = eye[0] + d[0] * r, eye[2] + d[1] * r
        h = max(WD.ground(x, z, CR, r / 2000.0), WD.h_cloud(x, z, r / 2000.0, 0.0)) - r * r / (2 * WD.R_EARTH)
        best = max(best, math.degrees(math.atan2(h - eye[1], r)))
    return best


def _sun_az():
    """The sun comes up behind the shoulder's right flank as seen from her: the azimuth where her skyline is ~0.25
    deg (so her summit is lit last, a little after the disc clears the flank for the camera behind her)."""
    global SUN_AZ
    if SUN_AZ is None:
        eye = SEAT + np.array([0.0, 1.0, 0.0])
        best = None
        for da in np.linspace(0.3, 8.0, 50):
            el = skyline_el(SUN_BASE_AZ + da, eye)
            if best is None or abs(el + 0.12) < abs(best[1] + 0.12):
                best = (SUN_BASE_AZ + da, el)
        SUN_AZ = best[0]
    return SUN_AZ


# ------------------------------------------------------------------ camera ---
SET_BACK, SET_UP, SET_HFOV, SET_TURN = 27.0, 4.2, 50.0, 11.0     # the view runs 11 deg right of the sun
SET_SEAT_V = 0.70                                                 # her seat at 70% down the frame


def _look_pitch(pos, target, hfov, W, H, v):
    """Pitch that puts `target` at fraction v of the frame height."""
    d = target - pos
    el = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
    f = 0.5 * W / math.tan(math.radians(hfov) * 0.5)
    return el + math.degrees(math.atan((v - 0.5) * H / f))


def settled_cam(W, H):
    az = _sun_az() + SET_TURN
    f = KP._dirxz(az)
    pos = SEAT - f * SET_BACK + np.array([0.0, SET_UP, 0.0])
    pitch = _look_pitch(pos, SEAT + np.array([0.0, 0.6, 0.0]), SET_HFOV, W, H, SET_SEAT_V)
    return RC.RCam(pos, az, pitch, 0.0, SET_HFOV, W, H)


def camera(frame, W, H):
    """The crane: an arc round her summit from the vigil's locked frame to the settled frame, rising over it."""
    c0 = KP.camera(W, H)
    c1 = settled_cam(W, H)
    u = _ease((frame - F0) / float(F_SETTLE - F0))
    top = KP.TOP
    # arc in bearing (around her seat), distance and height
    def polar(p):
        d = p - SEAT
        return math.degrees(math.atan2(d[0], d[2])), math.hypot(d[0], d[2]), p[1] - SEAT[1]
    b0, r0, h0 = polar(c0.pos)
    b1, r1, h1 = polar(c1.pos)
    db = (b1 - b0 + 540.0) % 360.0 - 180.0
    b = b0 + db * u
    rr = r0 + (r1 - r0) * u
    hh = h0 + (h1 - h0) * u + 10.0 * math.sin(math.pi * u) ** 2          # rises over the arc, then settles
    a = math.radians(b)
    pos = SEAT + np.array([rr * math.sin(a), hh, rr * math.cos(a)])
    hfov = c0.hfov_d + (c1.hfov_d - c0.hfov_d) * u
    # aim: the seat's screen position eases from where the locked frame has it to where the settled frame has it
    def seat_uv(c):
        x, y, z = c.project(SEAT + np.array([0.0, 0.6, 0.0]))
        return x / c.W, y / c.H
    (u0, v0), (u1, v1) = seat_uv(c0), seat_uv(c1)
    us, vs = u0 + (u1 - u0) * u, v0 + (v1 - v0) * u
    d = SEAT + np.array([0.0, 0.6, 0.0]) - pos
    bear = math.degrees(math.atan2(d[0], d[2]))
    fpx = 0.5 * W / math.tan(math.radians(hfov) * 0.5)
    yaw = bear - math.degrees(math.atan((us - 0.5) * W / fpx))
    # the horizon leads the tilt (the range must stay in frame); she may sink low in frame, never below 90%
    pitch = c0.pitch_d + (c1.pitch_d - c0.pitch_d) * u - 2.5 * math.sin(math.pi * u) ** 2
    pitch = min(pitch, _look_pitch(pos, SEAT + np.array([0.0, 0.6, 0.0]), hfov, W, H, 0.90))
    if frame >= F_SETTLE:
        return c1
    return RC.RCam(pos, yaw, pitch, 0.0, hfov, W, H)


# ---------------------------------------------------------------- beacons ---
def beacon_catalogue():
    """Every summit around her (0.8-70 km, all directions) that stands above the cloud: one beacon each (world
    points, cached). The sixtieth night: all alight."""
    p = os.path.join(KP.CACHE, 'hb_beacons.npy')
    if os.path.exists(p):
        return np.load(p)
    pts = []
    for r in np.geomspace(800.0, 70000.0, 70):
        n = int(min(max(2 * math.pi * r / max(r * 0.05, 300.0), 24), 160))
        for k in range(n):
            a = 2 * math.pi * (k + 0.5 * (int(r) % 2)) / n
            x, z = KP.TOP[0] + r * math.sin(a), KP.TOP[2] + r * math.cos(a)
            fp = max(r / 800.0, 2.0)
            step = max(r * 0.01, 30.0)
            bh = WD.h_rock(x, z, fp, CR)
            for _it in range(14):
                imp = False
                for ddx, ddz in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    h = WD.h_rock(x + ddx, z + ddz, fp, CR)
                    if h > bh:
                        bh, x, z, imp = h, x + ddx, z + ddz, True
                if not imp:
                    step *= 0.5
            if bh > WD.CLOUD_Y + 110.0 and math.hypot(x - KP.TOP[0], z - KP.TOP[2]) > 400.0:
                pts.append((x, bh, z))
    P = np.array(pts)
    keep = []
    for q in P[np.argsort(np.hypot(P[:, 0] - KP.TOP[0], P[:, 2] - KP.TOP[2]))]:
        dd = math.hypot(q[0] - KP.TOP[0], q[2] - KP.TOP[2])
        if all(math.hypot(q[0] - k[0], q[2] - k[2]) > max(0.035 * dd, 150.0) for k in keep):
            keep.append(q)
    B = np.array(keep)
    os.makedirs(KP.CACHE, exist_ok=True)
    np.save(p, B)
    return B


def beacon_thresholds(B):
    """Per beacon: the sun elevation (deg) at which its summit is in the sun (clearance toward SUN_AZ)."""
    p = os.path.join(KP.CACHE, f'hb_thresh2_{_sun_az():.3f}.npy')
    if os.path.exists(p):
        return np.load(p)
    a = math.radians(_sun_az())
    P = np.array([KP.TOP[0], KP.TOP[2], 0.0, 0.0])
    th = np.zeros(len(B))
    for k, (x, y, z) in enumerate(B):
        curv = ((x - P[0]) ** 2 + (z - P[1]) ** 2) / (2 * WD.R_EARTH)
        c0 = RL.clearance0_fine(P, CR, x, y + 4.0 - curv, z, math.sin(a), math.cos(a), 2.0, 0.03, 4.0, 120000.0)
        th[k] = math.degrees(math.asin(min(max(-c0, -0.2), 0.2)))
    np.save(p, th)
    return th


def her_threshold():
    a = math.radians(_sun_az())
    P = np.array([KP.TOP[0], KP.TOP[2], 0.0, 0.0])
    b = KP.BEACON + np.array([0.0, 1.4, 0.0])
    c0 = RL.clearance0_fine(P, CR, b[0], b[1], b[2], math.sin(a), math.cos(a), 0.05, 0.01, 2.0, 60000.0)
    return math.degrees(math.asin(min(max(-c0, -0.2), 0.2)))


def villages_world():
    p = os.path.join(KP.CACHE, 'hb_villages.npy')
    if os.path.exists(p):
        return np.load(p)
    rng = np.random.default_rng(12)
    V = []
    tries = 0
    while len(V) < 44 and tries < 20000:
        tries += 1
        r = 1800.0 * math.exp(rng.random() * math.log(16000.0 / 1800.0))
        a = rng.random() * 2 * math.pi
        x, z = KP.TOP[0] + r * math.sin(a), KP.TOP[2] + r * math.cos(a)
        if WD.h_rock(x, z, 30.0, CR) > WD.CLOUD_Y - 60.0:
            continue
        if all(math.hypot(x - q[0], z - q[1]) > 1100.0 for q in V):
            V.append((x, z, rng.uniform(280, 560), rng.uniform(0.6, 1.3)))
    V = np.array(V)
    np.save(p, V)
    return V


# ----------------------------------------------------------------- figures ---
def keeper_seated(age=1.0, hands=0.0, touch=0.0):
    """Her, very old, seated, from behind: a rounded, stooped back (the head sunk forward below the hump of the
    shoulders), the cloak spread on the rock, the stick planted upright at her right side (her hand on it), and
    at `hands` > 0 her old hands drawn out of the sleeves and opened to the sun (arms out, forearms up).
    `touch` > 0: the right hand goes to the child's shoulder. Local metres, feet origin, x = screen right."""
    d = FG.Drawing()
    d.new_group()
    k = 0.04
    # the cloak on the rock: a wide low base
    d.trap((0.0, 0.0), (0.0, 0.42), 0.46, 0.34, rnd=0.05, k=0.06, mat=14, fuzz=0.012, ff=12.0)
    # the stooped back: a big rounded hump; the head sunk forward under it (only the hood's crown shows)
    d.ellipse((0.0, 0.62), 0.29, 0.25, k=0.08, mat=14, fuzz=0.012, ff=12.0)
    d.ellipse((0.0, 0.80), 0.22, 0.13, k=0.07, mat=14, fuzz=0.012, ff=12.0)
    d.ellipse((0.02, 0.86), 0.105, 0.10, k=0.03, mat=14)                       # the hood's crown, low
    d.ellipse((0.07, 0.86), 0.05, 0.06, mat=15)                                 # white hair at the hood's edge
    # arms: at rest the forearms are hidden in front of the body; opened, they come out wide and rise
    for sg in (-1, 1):
        if sg > 0 and touch > 0.0:
            continue
        sh = np.array([0.20 * sg, 0.74])
        if hands > 0.0:
            el = sh + np.array([0.20 * sg, -0.14 + 0.02 * hands]) * (0.4 + 0.6 * hands)
            wr = el + np.array([0.18 * sg, 0.10 + 0.14 * hands]) * hands
            d.capsule(sh, el, 0.07, 0.06, k=k, mat=14)
            d.capsule(el, wr, 0.06, 0.05, k=k, mat=14)
            d.ellipse(wr + np.array([0.03 * sg, 0.02]), 0.045, 0.035, ang=0.6 * sg, mat=2)   # an open palm
        else:
            el = sh + np.array([0.05 * sg, -0.22])
            d.capsule(sh, el, 0.07, 0.06, k=k, mat=14)
    if touch > 0.0:
        sh = np.array([0.20, 0.74])
        tip = sh + np.array([0.22 + 0.16 * touch, -0.12 - 0.06 * touch])
        d.capsule(sh, tip, 0.065, 0.05, k=k, mat=14)
        d.ellipse(tip, 0.045, 0.035, mat=2)
    # the stick: planted upright at her right, taller than she sits; it leans on her shoulder when her hands open
    d.new_group()
    base = np.array([0.40, 0.0])
    lean = 0.10 * max(hands, touch)
    tip = np.array([0.40 - 0.16 * lean * 3, 1.30 - 0.05 * lean])
    d.capsule(base, tip, 0.017, 0.015, mat=4)
    return d


def child_seated(wake=0.0, sun=0.0):
    """The child from behind, seated against her right side in her red scarf: asleep (head down on her arm), then
    waking: the head rises and turns to the sun."""
    d = FG.Drawing()
    d.new_group()
    d.trap((0.0, 0.0), (0.0, 0.30), 0.22, 0.17, rnd=0.04, k=0.05, mat=0, fuzz=0.01, ff=14.0)
    lean = 0.35 * (1.0 - wake)
    back = np.array([-0.10 * lean, 0.44])
    d.ellipse(back, 0.17, 0.15, ang=0.4 * lean, k=0.06, mat=0)
    head = back + np.array([-0.16 * lean, 0.14 + 0.06 * wake])
    d.ellipse(head, 0.085, 0.095, k=0.03, mat=6)
    d.new_group()
    # the red scarf: wound round the shoulders and head, its tail over her lap
    d.ellipse(back + np.array([0.0, 0.08]), 0.16, 0.07, ang=0.4 * lean, k=0.02, mat=13)
    d.ellipse(head + np.array([0.0, -0.04]), 0.10, 0.05, k=0.02, mat=13)
    d.chain(np.array([back + np.array([0.12, 0.06]), back + np.array([0.22, -0.02]), back + np.array([0.28, -0.14]),
                      back + np.array([0.30, -0.26])]), 0.04, 0.025, k=0.02, mat=13)
    return d


# ------------------------------------------------------------------- shot ---
class HandBack:
    def __init__(self, scale=0.25, ss=1.5, figures=True):
        self.scale, self.ss, self.figures = scale, ss, figures
        self.W, self.H = int(round(1920 * scale)), int(round(804 * scale))
        _sun_az()
        B = beacon_catalogue()
        th = beacon_thresholds(B)
        keep = B[:, 1] > WD.CLOUD_Y + 260.0                  # prominent summits only: a range alight, not a starfield
        self.B, self.th = B[keep], th[keep]
        self.e_her = her_threshold()
        self.th = np.minimum(self.th, self.e_her - 0.08)
        self.V = villages_world()
        self._settled = None
        self.stars = SK.make_stars(14000, 101, lum_scale=7.0)
        self.schedule()

    # --- the sun's elevation from the bar map
    def schedule(self):
        from scipy.interpolate import PchipInterpolator
        tc = settled_cam(self.W, self.H)
        sx, sy, z = tc.project(self.B)
        ok = (z > 0) & (sx > 0.03 * self.W) & (sx < 0.97 * self.W) & (sy > 0) & (sy < self.H * 0.75)
        dist = np.linalg.norm(self.B - tc.pos, axis=1)
        e_first = float(np.percentile(self.th[ok], 5)) if ok.any() else self.e_her - 1.0
        e_first = min(e_first, self.e_her - 0.9)
        # six beacons for the horns: spread in threshold between first light and hers, nearer ones later, prominent
        cand = [(self.th[k], k) for k in np.nonzero(ok)[0] if e_first + 0.08 < self.th[k] < self.e_her - 0.06
                and dist[k] < 30000.0]
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
        return float(self.e_of(max(frame, F_GREY)))

    # --- the settled frame: G-buffer + night plate, once
    def settled(self):
        if self._settled is not None:
            return self._settled
        tc = settled_cam(self.W, self.H)
        fr = PI.Frame(tc, self.ss)
        scam = fr.src
        P = np.array([scam.pos[0], scam.pos[2], 0.0, 0.0])
        p = os.path.join(KP.CACHE, f'hb_G3_{self.scale:.3f}_{self.ss:.2f}_{_sun_az():.2f}.npy')
        if os.path.exists(p):
            G = np.load(p)
        else:
            G = RL.build(scam, P, CR, _sun_az(), nsteps=88, tmax=120000.0, snow_bias=0.08, dmax=160000.0, kstep=0.03)
            np.save(p, G)
        pn = os.path.join(KP.CACHE, f'hb_night3_{self.scale:.3f}_{self.ss:.2f}_{_sun_az():.2f}.npz')
        if os.path.exists(pn):
            z = np.load(pn)
            night, zb, dist = z['img'], z['zb'], z['dist']
        else:
            PI.render_terrain(fr, 4080, CR, WD.night_light(), np.zeros((0, 8)))
            night, zb, dist = fr.img.copy(), fr.zb.copy(), fr.dist.copy()
            np.savez_compressed(pn, img=night, zb=zb, dist=dist)
        self._settled = dict(tc=tc, fr=fr, scam=scam, G=G, night=night, zb=zb, dist=dist, P=P)
        return self._settled

    def dawn_params(self, e, grey):
        """LP / sky / amb / fog for the relight at sun elevation e; grey = 0..1 how far the east has greyed."""
        lin = CM.lin
        a = math.radians(_sun_az())
        L = np.array([math.cos(math.radians(e)) * math.sin(a), math.sin(math.radians(e)),
                      math.cos(math.radians(e)) * math.cos(a)])
        up = smoothstep(-2.0, 1.0, e)                  # the day coming up
        LP = np.zeros(RL.NLP)
        LP[0:3] = L
        LP[3] = 5.6
        LP[4:7] = [0.016, 0.042, 0.090]
        LP[7] = 0.010
        LP[8] = 0.0026
        LP[9] = 0.95
        LP[10] = 2.2
        LP[12] = 1.4                                   # the snow glitters toward the low sun (seen from behind her)
        gd = np.array([L[0], 0.10, L[2]])
        LP[13:16] = gd / np.linalg.norm(gd)
        LP[16:19] = lin('#FFB98A') * 0.035 * up
        LP[19] = 1.0
        LP[20] = 1.0 / self.settled()['scam'].f
        LP[21] = 1.0
        SD = np.zeros(24)
        SD[0:3] = L
        SD[3:6] = np.array([1.0, 0.55, 0.26])
        SD[6] = 900.0 * smoothstep(-0.6, 0.2, e)
        SD[7] = math.radians(0.27)
        SD[8:11] = lin('#2B53A0') * (0.20 + 0.40 * up) * grey
        SD[11:14] = lin('#FFBE78') * (0.25 + 0.75 * up) * grey
        SD[14:17] = lin('#A9A6CE') * (0.12 + 0.26 * up) * grey
        SD[17] = 1.3 * up
        SD[18] = 0.045 * up
        SD[19] = 0.034
        SD[20] = 0.55
        amb = lin('#6A86C8') * (0.07 + 0.22 * up) * grey
        fc = lin('#6F83B8') * (0.10 + 0.22 * up) * grey
        wc = lin('#FFC58A') * 0.8 * up
        fogp = np.array([5.0e-5, 1 / 1500.0, 2.0e-4, 1 / 150.0, 1.2, fc[0], fc[1], fc[2], 30.0, wc[0], wc[1], wc[2]])
        return LP, SD, amb, fogp

    # --- frame
    def render(self, frame):
        t = frame / FPS
        if frame < F_SETTLE:
            return self.render_crane(frame)
        S = self.settled()
        scam, tc, fr = S['scam'], S['tc'], S['fr']
        grey = smoothstep(F_GREY, F_FIRST, frame)
        if frame < F_GREY:
            img = S['night'].copy()
        else:
            e = self.sun_el(frame)
            LP, SD, amb, fogp = self.dawn_params(e, max(grey, 0.02))
            day = np.zeros((scam.H, scam.W, 3), np.float32)
            RL.shade(S['G'], LP, SD, amb, fogp, float(scam.pos[1]), day)
            w = smoothstep(F_GREY, F_GREY + 60, frame)
            img = (S['night'] * (1.0 - w) + day * w).astype(np.float32)
        zb = S['zb'].copy()
        dist = S['dist']
        self.layers(img, zb, dist, scam, frame, t, settled=True)
        fr.img, fr.zb, fr.dist = img, zb, dist
        out, _, _ = PI.to_target(fr)
        return out

    def render_crane(self, frame):
        t = frame / FPS
        tc = camera(frame, self.W, self.H)
        fr = PI.Frame(tc, self.ss)
        PI.render_terrain(fr, frame, CR, WD.night_light(), np.zeros((0, 8)))
        img, zb, dist = fr.img, fr.zb, fr.dist
        self.layers(img, zb, dist, fr.src, frame, t, settled=False)
        fr.img = img
        out, _, _ = PI.to_target(fr)
        return out

    def layers(self, img, zb, dist, scam, frame, t, settled):
        ss = self.ss
        night = 1.0 - smoothstep(F_GREY, F_FIRST + 40, frame)
        e = self.sun_el(frame) if frame >= F_GREY else -5.0
        # stars, fading as the east greys
        if night > 0.01:
            mask = (dist > 1e8).astype(np.float32)
            SK.splat_stars(img, scam, self.stars, mask, t=t, gain=ss * ss * night, scale=scam.f / (0.5 * 1920 / math.tan(math.radians(22.0))))
        # village glow under the cloud, fading into the day
        vis = night * 0.9 + 0.1
        V = self.V
        fld = np.zeros(dist.shape, np.float32)
        C = scam.params()
        KP._village_field(scam.W, scam.H, C[8], C[9], C[7], scam.pos[0], scam.pos[1], scam.pos[2], C[3], C[4], C[5],
                          C[6], dist, V[:, 0], V[:, 1], V[:, 2], V[:, 3] * vis, fld)
        img += fld[..., None] * (np.array([1.0, 0.55, 0.22], np.float32) * 0.10)[None, None, :]
        # every beacon of the range; each pales as the sun reaches it
        Bc = self.B.copy()
        Bc[:, 1] += 3.0 - ((Bc[:, 0] - scam.pos[0]) ** 2 + (Bc[:, 2] - scam.pos[2]) ** 2) / (2 * WD.R_EARTH)
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
        L = np.array([math.cos(math.radians(max(e, -1))) * math.sin(a), math.sin(math.radians(max(e, -1.0))),
                      math.cos(math.radians(max(e, -1))) * math.cos(a)])
        fire_lvl = 1.0 - 0.9 * her_sun
        fire_I = 1.8 * fire_lvl * F.flicker(t, 3)
        amb = CM.lin('#27335E') * 0.45 * night + CM.lin('#6A86C8') * 0.14 * (1.0 - night)
        lights = [dict(dir=L, col=np.array([1.0, 0.62, 0.36]), I=2.2 * her_sun),
                  dict(pos=KP.BEACON + np.array([0, 1.3, 0]), col=F.FIRE_LIGHT, I=fire_I, r0=0.5)]
        if night > 0:
            Lk, _a, _S, _f, Q = WD.night_light()
            lights.append(dict(dir=Lk[:3], col=Lk[3:6], I=Q[0] * 0.7 * night))
        FG.render(img, zb, scam, KP.stone_cairn(60), KP.CAIRN, lights, amb=amb, mats=KP.M, t=t, write_depth=True,
                  zbias=0.3)
        back, front, fb, rb = PR.cairn(seed=9, height=0.85, base_w=1.2, top_w=0.95, basket=True, basket_h=0.5,
                                       basket_w=1.1)
        FG.render(img, zb, scam, back, KP.BEACON, lights, amb=amb, t=t, emissive_gain=0.9 * fire_lvl + 0.1,
                  write_depth=True, zbias=0.3)
        base = KP.BEACON + np.array([0.0, fb, 0.0])
        F2.flame(img, zb, scam, base, 1.05, 0.42, t, seed=4, I=12.0 * (0.12 + 0.88 * fire_lvl), lean=0.2, zbias=0.5,
                 tongues=5, warp=1.2)
        bx, by, bz = scam.project(base + np.array([0, 0.6, 0]))
        F2.halo(img, zb, bx, by, 5.0 * scam.f / bz, 0.005 * fire_lvl, z=bz, zbias=3.0)
        if her_sun > 0.3:              # in full sun her fire is only a shimmer of heat
            F.shimmer(img, scam, base + np.array([0, 0.9, 0]), 1.6, 0.5, t, amp_px=0.9 * ss * her_sun, seed=3)
        FG.render(img, zb, scam, front, KP.BEACON, lights, amb=amb, t=t, write_depth=False, zbias=0.3)
        if self.figures:
            hands = smoothstep(F_HANDS - 10, F_HANDS + 30, frame) * (1.0 - smoothstep(F_TOUCH - 30, F_TOUCH, frame))
            touch = smoothstep(F_TOUCH - 20, F_TOUCH + 8, frame)
            wake = smoothstep(F_WAKE - 6, F_WAKE + 40, frame)
            child_sun = smoothstep(F_CHILD_SUN - 20, F_CHILD_SUN + 10, frame)
            lk = [dict(lights[0], I=lights[0]['I'] * child_sun)] + lights[1:]
            FG.render(img, zb, scam, child_seated(wake), KP.on_ground(SEAT + self._side() * 0.62), lk, amb=amb,
                      mats=KP.M, t=t, write_depth=False, zbias=0.3)
            FG.render(img, zb, scam, keeper_seated(1.0, hands, touch), SEAT, lights, amb=amb, mats=KP.M, t=t,
                      write_depth=False, zbias=0.3)

    def _side(self):
        f = KP._dirxz(_sun_az())
        return np.array([f[2], 0.0, -f[0]])


FINISH = dict(exposure=0.78, bloom_strength=0.08, bloom_threshold=1.3, streak_strength=0.003, vignette_amount=0.22)


def _work(args):
    frames, scale, ss, out, figs = args
    shot = HandBack(scale, ss, figs)
    for f in frames:
        t0 = time.time()
        img = look.finish(shot.render(f), **FINISH)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.2f}s', flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default='handback_motion')
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args()
    out = os.path.join(CM.ROOT, a.out) if a.out.startswith('renders/') else os.path.join(KP.OUTDIR, a.out)
    os.makedirs(out, exist_ok=True)
    if a.range:
        s0, s1 = a.range.split('-')
        frames = list(range(int(s0), int(s1) + 1, a.step))
    else:
        frames = [int(x) for x in a.frames.split(',')]
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.frame_path(out, f))]
    shot = HandBack(a.scale, a.ss, not a.no_figures)          # builds every cache once, before any fork
    if any(f >= F_SETTLE for f in frames):
        shot.settled()
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
