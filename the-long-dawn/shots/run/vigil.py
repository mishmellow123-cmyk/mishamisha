"""B6-B12 . THE VIGIL (R9's fallback, now B): one longest night in one locked frame, frames 1520-3839 of cut B.

The night is the vow. One continuously moving sky (never a plate change): the stars wheel about the pole (a time-lapse
of ~9 hours), the moon crosses the northern sky above the frame from ENE to NW and its light moves over the land (K
precomputed moon-visibility channels, interpolated per frame), and the weather passes through as continuous layers
(hour 2 a snow squall, hour 4 fog rising out of the cloud sea, hour 9 fog again). The chaconne marks the hours: at
each hour's end she feeds the fire. Nothing answers for a long time; then one far pinprick on the line of sight (bar
35); a traveller climbs up the NE path with an unlit torch, takes flame and carries it down (bars 36-37); three
more (38-39) and the first village lights under the cloud (39); her fire flares and the far light answers as a full
beacon (40-41); the thread of torches down the ridge grows, more far tops burn, the valleys brighten (42-45); a
traveller's child stays and sleeps against her, wrapped in her red shawl (46-48). The hand-back's crane takes over
on bar 49 (handback_b.py) from this exact frame.
Figures are RUN-B's silhouettes (bset): hood, red woven shawl, old (a stoop, a staff), never front-lit, ~95 px tall.

  python shots/run/vigil.py --frames 1520,2060,2400,2720,2900,3200,3500,3800 --scale 0.25
  python shots/run/vigil.py --build --scale 1.0 --ss 1.5                 # cache the G-buffer (cloud setup)
  python shots/run/vigil.py --range 1520-2099 --scale 1.0 --ss 1.5 --out renders/vigil_B --skip
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bworld as BW         # noqa: E402
import bset as BS           # noqa: E402
import pipe as PI           # noqa: E402
import rcam as RC           # noqa: E402
import fire2 as F2          # noqa: E402
import bfig as BF           # noqa: E402  (RUN-B-3's B figure light: sun-side rim, wool, stones)
import bprops as BP         # noqa: E402  (RUN-B-3's cairn3 / child3, shared with the hand-back)
import keeper as KP         # noqa: E402  (generic helpers only: star trails, village field)
from mt import fire as F, figure as FG, sky as SK   # noqa: E402
from mt.noise import smoothstep   # noqa: E402

CM = PI.CM
look = PI.look
lin = CM.lin
CR = BW.CR_B
FPS = 24.0
CACHE = os.path.join(CM.ROOT, 'renders', 'run_b_tests', 'cache')
TESTS = os.path.join(CM.ROOT, 'renders', 'run_b_tests')

F0, F1 = 1520, 3840


def bar(b, beat=1.0):
    return int(round((b - 1) * 80 + (beat - 1) * 20))


# the feeds (the cues' stone_N events: "an hour gone; she feeds the fire") and the other sync points
FEEDS = [bar(25, 3), bar(27, 3), bar(29, 3), bar(31, 3), bar(35, 3), bar(37, 3), bar(39, 3), bar(43, 3),
         bar(45, 3), bar(47, 1)]
F_FIRST_ANSWER = bar(35)
F_SEEN = bar(33)            # 'child_points' (2560): far off, someone kindles a small light (faint, unsure)
F_TRAV1_LIGHT = bar(37)     # 'traveller' (2880): the first traveller's torch takes at her fire
F_VILLAGE = bar(39)
F_FLARE, F_HIS = bar(40, 2), bar(41)
F_CHILD_IN, F_SIT, F_SHAWL = bar(46) - 100, bar(48), bar(48, 2)   # the child tops the path with its parent on bar 46
F_JOIN_FIGS = bar(48, 2) + 30          # from here the hand-back draws the pair (the crane starts from this frame)


def HB_MOON_END():
    import handback_b as HB
    return HB.MOON_END


# ------------------------------------------------------------------ sky + weather ---
def moon_at(f):
    """The moon's path over the night: rising ENE, high in the north (backlight), setting NW (hand-back's MOON_END)."""
    u = min(max((f - F0) / float(F1 - F0), 0.0), 1.0)
    az = 80.0 - 135.0 * u
    el = 14.0 + 30.0 * math.sin(math.pi * u) - 2.0 * u
    return az, el


MOON_KEYS = np.linspace(F0, F1, 6)
MOON_DIRS = np.array([BS.moon_vec(moon_at(f)[1], moon_at(f)[0]) for f in MOON_KEYS])


def moon_mix(f):
    x = (f - F0) / float(F1 - F0) * (len(MOON_KEYS) - 1)
    a = int(min(max(math.floor(x), 0), len(MOON_KEYS) - 2))
    return a, a + 1, min(max(x - a, 0.0), 1.0)


def storm(f):
    return smoothstep(bar(25, 3), bar(26, 1) + 10, f) * (1.0 - smoothstep(bar(27, 3), bar(28, 1) + 10, f))


def fog(f):
    a = smoothstep(bar(29, 4), bar(30, 2), f) * (1.0 - smoothstep(bar(31, 3), bar(32, 1) + 20, f))
    b = smoothstep(bar(43, 4), bar(44, 2), f) * (1.0 - smoothstep(bar(45, 3), bar(46, 1) + 20, f))
    return max(a, b)


STAR_DEG = 132.0            # the sky wheels ~9 hours across the vigil
F_WHEEL_SLOW = bar(47)      # the wheel eases to a stop at the crane (the hand-back's stars are static)
_OMEGA = math.radians(STAR_DEG) / ((F_WHEEL_SLOW - F0) + 0.5 * (F1 - F_WHEEL_SLOW))


def theta(f):
    """The sky's rotation about the pole (radians) at frame f: 0 at the crane (f >= F1), negative before it; the
    rate is constant until bar 47, then eases to zero at F1. Continues linearly before F0 (the reveal)."""
    if f >= F1:
        return 0.0
    L = F1 - F_WHEEL_SLOW
    if f >= F_WHEEL_SLOW:
        u = (f - F_WHEEL_SLOW) / float(L)
        return -_OMEGA * L * (0.5 - u + u ** 3 - 0.5 * u ** 4)
    return -_OMEGA * ((F_WHEEL_SLOW - f) + 0.5 * L)


def sky_rot(f):
    return KP._rotmat(KP.POLE, theta(f))


def sky_angle(f):          # (kept for callers of the old API)
    return theta(f)


# the Milky Way in SKY coordinates = its world pose at the crane (R = I at F1); it wheels with the stars
def _band_from_start(b_start):
    """A band given as its pose at the vigil's start (F0) -> sky coordinates."""
    Rm = sky_rot(F0).T
    n = Rm @ np.asarray(b_start[0:3])
    c = Rm @ np.asarray(b_start[3:6])
    return np.r_[n, c, b_start[6:8]]


def _band_world(az_c, el_c, az_2, el_2, width=0.065, seed=117.0):     # 0.13 read as a cloud bank
    """A band with its galactic centre at (az_c, el_c) passing through (az_2, el_2) at the vigil's start."""
    def dv(az, el):
        a, e = math.radians(az), math.radians(el)
        return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
    c = dv(az_c, el_c)
    n = np.cross(c, dv(az_2, el_2))
    n /= np.linalg.norm(n)
    return _band_from_start(np.r_[n, c, width, seed])


BAND = _band_world(95.0, 2.0, 20.0, 15.0)      # the reveal's last wide: from the horizon right of her summit, arching
#                                                  up over it (low in the ENE at the vigil's start, then it wheels away)
BAND_GAIN = 0.10
STAR_GAIN = 1.6             # the low sky of a locked frame reads empty at the catalogue's own gain

_STARS = None


def star_cat():
    """The hand-back's catalogue (make_stars(14000, 101, 7.0)), so the sky is one sky across the join."""
    global _STARS
    if _STARS is None:
        _STARS = SK.make_stars(14000, 101, lum_scale=7.0)
    return _STARS


def draw_sky(img, G, scam, mask, f, clear, ss, th=None, band_gain=None):
    """The stars (points, a half-frame shutter) and the Milky Way, wheeling about the pole. th: a frozen sky angle
    (the reveal is real time); band_gain: override BAND_GAIN."""
    st = star_cat()
    t = f / FPS
    scale = scam.f / (0.5 * 1920 / math.tan(math.radians(22.0)))
    ths = np.linspace(theta(f), theta(f + 0.5), 3) if th is None else [th]
    for tk in ths:
        Rk = KP._rotmat(KP.POLE, tk)
        SK.splat_stars(img, scam, dict(st, dir=st['dir'] @ Rk.T), mask, t=t, gain=ss * ss * clear * STAR_GAIN / len(ths),
                       scale=scale)
    BW.add_band(G, KP._rotmat(KP.POLE, theta(f) if th is None else th), BAND,
                BAND_GAIN if band_gain is None else band_gain, clear, img)


def fire_level(f):
    """The fire: high after the roar, sinking each hour, restored by each feed; low in the squall."""
    lv = 0.55
    last = F0 - 200
    for q in FEEDS:
        if q <= f:
            last = q
    since = (f - last) / 160.0
    lv = 0.95 - 0.40 * min(since, 1.0) if last >= F0 - 100 else 0.9 - 0.35 * min((f - F0) / 440.0, 1.0)
    # the feed: a flare as the wood takes
    for q in FEEDS:
        if q - 4 <= f < q + 40:
            lv += 0.5 * math.exp(-((f - q - 8) / 10.0) ** 2)
    if F_FLARE - 4 <= f < F_FLARE + 60:
        lv += 0.9 * math.exp(-((f - F_FLARE - 12) / 18.0) ** 2)
    lv *= 1.0 - 0.45 * storm(f)
    return lv


# ------------------------------------------------------------------ the travellers' way ---
V_PATH_AZ = 80.0
_VPATH = None


def vpath():
    """The travellers' way (vigil only; the NE ridge stays her climb): from the summit's east lip (6 m out) down
    the ENE slope, winding, to the cloud sea ~2 km off. Returns (points, arc length)."""
    global _VPATH
    if _VPATH is None:
        p = BS.TOP + BS.dirxz(V_PATH_AZ) * 6.0
        pts = [BS.on_ground(p)]
        s = 0.0
        while s < 2100.0:
            ds = 6.0 if s < 200.0 else 15.0
            az = V_PATH_AZ + 11.0 * math.sin(s / 240.0) + 4.0 * math.sin(s / 71.0 + 1.3)
            p = p + BS.dirxz(az) * ds
            s += ds
            q = BS.on_ground(p)
            if q[1] < BW.CLOUD_Y + 20.0:
                break
            pts.append(q)
        P = np.array(pts)
        _VPATH = (P, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return _VPATH


def vpath_at(s):
    P, S = vpath()
    s = min(max(s, 0.0), S[-1])
    k = min(max(int(np.searchsorted(S, s) - 1), 0), len(S) - 2)
    u = (s - S[k]) / max(S[k + 1] - S[k], 1e-6)
    return P[k] + (P[k + 1] - P[k]) * u


# ------------------------------------------------------------------ the frame ---
# the locked frame (defaults: bset's vigil camera, which is also the hand-back crane's first frame); --cam
# overrides it for look-dev (bear, dist, up, yaw, pitch, hfov: bearing camera->top, metres out, metres above the top)
VCAM = dict(bear=BS.CAM_BEAR, dist=BS.CAM_DIST, up=BS.CAM_UP, yaw=BS.CAM_YAW, pitch=BS.PITCH, hfov=BS.HFOV)
_VK = ('bear', 'dist', 'up', 'yaw', 'pitch', 'hfov')


def camera(W=1920, H=804):
    pos = BS.TOP - VCAM['dist'] * BS.dirxz(VCAM['bear']) + np.array([0.0, VCAM['up'], 0.0])
    return RC.RCam(pos, VCAM['yaw'], VCAM['pitch'], 0.0, VCAM['hfov'], W, H)


def cam_key():
    """The locked frame's numbers (in every cache name, so a moved camera never loads a stale G-buffer)."""
    return 'c' + '_'.join(f'{VCAM[k]:g}' for k in _VK)


class Vigil:
    def __init__(self, scale=0.25, ss=1.5):
        self.scale, self.ss = scale, ss
        self.W, self.H = int(round(1920 * scale)), int(round(804 * scale))
        self.tc = camera(self.W, self.H)
        self.cam_yaw = VCAM['yaw']
        self.fr = PI.Frame(self.tc, ss)
        scam = self.fr.src
        # the cloud sea is frozen at the crane's own first time (handback_b builds the crane with t = frame/24), so
        # its billows are the same at the join (3839/3840)
        self.P = np.array([scam.pos[0], scam.pos[2], F1 / FPS, 0.0])
        p = os.path.join(CACHE, f'vig_G_{BW.VERSION}_{cam_key()}_t{F1 / FPS:.0f}_{scale:.3f}_{ss:.2f}.npy')
        if os.path.exists(p):
            self.G = np.load(p)
        else:
            self.G = BW.build(scam, self.P, CR, None, dmax=180000.0, moons=MOON_DIRS, mk=10.0)
            os.makedirs(CACHE, exist_ok=True)
            tmp = p + f'.{os.getpid()}.tmp.npy'
            np.save(tmp, self.G)
            os.replace(tmp, p)
        self.dist = self.G[..., BW.G_DIST].astype(np.float32)
        self.sky = (self.dist > 1e8).astype(np.float32)
        self.hb = self._handback()
        self.peaks = self._answers()
        self.vill = self._villages()
        self.rng = np.random.default_rng(5)

    def _handback(self):
        try:
            import handback_b as HB
            return HB.HandBack(self.scale, self.ss, figures=True)
        except Exception as e:          # the join needs it; the rest of the vigil does not
            print('vigil: no hand-back instance:', e, flush=True)
            return None

    # the far answers: the hand-back's beacon catalogue (every summit well above the cloud, all round her), those
    # this frame sees; the one on her line of sight answers first (bar 35), the rest one by one to bar 46. At the
    # crane every one of them burns exactly as the hand-back draws it.
    def _answers(self):
        import handback_b as HB
        scam = self.fr.src
        B = HB.beacon_catalogue()
        Bc = B.copy()
        Bc[:, 1] += 3.0 - ((Bc[:, 0] - scam.pos[0]) ** 2 + (Bc[:, 2] - scam.pos[2]) ** 2) / (2 * BW.R_EARTH)
        sx, sy, z = scam.project(Bc)
        dd = np.linalg.norm(B - scam.pos, axis=1)
        vis = []
        for k in range(len(B)):
            if z[k] <= 0 or not (2 <= sx[k] < scam.W - 2 and 2 <= sy[k] < scam.H - 2):
                continue
            ix, iy = int(sx[k]), int(sy[k])
            if self.dist[iy - 1:iy + 2, ix - 1:ix + 2].max() >= dd[k] * 0.97:
                vis.append(k)
        vis = np.array(vis, int)
        hx, hy, hz = scam.project(BS.TOP + np.array([0.0, 1.5, 0.0]))
        pref = [k for k in vis if dd[k] > 12000 and hx + 0.06 * scam.W < sx[k] < hx + 0.40 * scam.W]
        rng = np.random.default_rng(7)
        order = list(rng.permutation(vis)) if len(vis) else []
        first = min(pref, key=lambda k: abs(sx[k] - (hx + 0.2 * scam.W))) if pref else (order[0] if order else None)
        if first is not None:
            order.remove(first)
            order.insert(0, first)
        t_on = {}
        n = len(order)
        for i, k in enumerate(order):
            if i == 0:
                t_on[k] = F_FIRST_ANSWER
            else:
                u = i / max(n - 1, 1)
                t_on[k] = bar(36, 3) + (bar(46) - bar(36, 3)) * u ** 0.7 + rng.normal() * 20
        chosen = set(self.hb.chosen) if self.hb is not None else set()
        return dict(idx=vis, P=Bc[vis], sx=sx[vis], sy=sy[vis], z=z[vis], dist=dd[vis],
                    t_on=np.array([t_on[k] for k in vis]), first=first, chosen=chosen)

    def _villages(self):
        """The hand-back's village glows under the cloud (x, z, radius, gain); they wake one by one from bar 39."""
        import handback_b as HB
        V = HB.villages_world()
        scam = self.fr.src
        d = np.hypot(V[:, 0] - scam.pos[0], V[:, 1] - scam.pos[2])
        rng = np.random.default_rng(11)
        rank = np.argsort(np.argsort(d + rng.normal(size=len(d)) * 2500.0))
        t_on = F_VILLAGE + (bar(46) - F_VILLAGE) * (rank / max(len(V) - 1, 1)) ** 0.8
        return dict(V=V, t_on=t_on)

    # --- the world, relit
    def world(self, f):
        scam = self.fr.src
        a, b, w = moon_mix(f)
        md = MOON_DIRS[a] * (1.0 - w) + MOON_DIRS[b] * w
        md /= np.linalg.norm(md)
        st, fg = storm(f), fog(f)
        LP, SN, amb, fogp = BS.night_params(md, 1.0 / scam.f, gain=1.0 - 0.35 * st)
        LP[32], LP[33], LP[34] = a, b, w
        LP[23] *= 1.0 - 0.75 * st - 0.45 * fg
        LP[45] *= 1.0 - 0.8 * st
        # the fire lights the summit (a point light at the basket); over bar 48 it settles to the hand-back's full fire
        lv = fire_level(f)
        join = smoothstep(bar(47, 3), bar(48, 4), f)
        LP[36] = 9.0 * (lv + (1.0 - lv) * join) * F.flicker(f / FPS, 3)
        LP[37:40] = BS.BEACON + np.array([0.0, 1.3, 0.0])
        LP[40:43] = np.array(F.FIRE_LIGHT)
        LP[43] = 40.0
        # the squall: the air thickens near and far, the sky greys; the fog: a bank rising out of the cloud sea
        grey = lin('#3A4466') * 0.55
        fogp[0] *= 1.0 + 40.0 * st
        fogp[1] = 1.0 / (1600.0 + 20000.0 * st)
        fogp[5:8] = fogp[5:8] * (1.0 - st) + grey * st
        fogp[2] *= 1.0 + 60.0 * fg
        fogp[3] = 1.0 / (140.0 + 1500.0 * fg)
        fogp[5:8] = fogp[5:8] * (1.0 - 0.5 * fg) + lin('#5A6A90') * 0.6 * fg * 0.5
        SN[3:9] = SN[3:9] * (1.0 - 0.85 * st) + np.r_[grey * 0.55, grey * 0.9] * st
        SN[3:9] = SN[3:9] * (1.0 - 0.6 * fg) + np.r_[lin('#3C4A70') * 0.35, lin('#5A6A90') * 0.5] * 0.6 * fg
        SN[9] *= 1.0 - 0.8 * st
        BS.match_horizon(SN, fogp)
        out = np.zeros((scam.H, scam.W, 3), np.float32)
        BW.shade(self.G, LP, SN, amb, fogp, float(scam.pos[1]), out)
        return out, md, lv

    def render(self, f):
        t = f / FPS
        scam = self.fr.src
        img, md, lv = self.world(f)
        zb = self.dist.copy()
        st, fg = storm(f), fog(f)
        clear = (1.0 - st) * (1.0 - 0.85 * fg)
        # the stars and the Milky Way, wheeling (the hand-back's sky: identity at the crane)
        if clear > 0.02:
            draw_sky(img, self.G, scam, self.sky, f, clear, self.ss)
        # the far answers, one by one; the line-of-sight light flares into a beacon on bar 41
        self.answers(img, zb, scam, f, t, clear)
        # village glow under the cloud (the hand-back's villages, waking one by one)
        V = self.vill
        u_on = np.clip((f - (V['t_on'] - 30.0)) / 90.0, 0.0, 1.0)
        on = u_on * u_on * (3.0 - 2.0 * u_on) * (1.0 - 0.8 * fg) * (1.0 - 0.7 * st)
        if on.max() > 0:
            fld = np.zeros(self.dist.shape, np.float32)
            C = scam.params()
            VV = V['V']
            KP._village_field(scam.W, scam.H, C[8], C[9], C[7], scam.pos[0], scam.pos[1], scam.pos[2], C[3], C[4],
                              C[5], C[6], self.dist, VV[:, 0], VV[:, 1], VV[:, 2], VV[:, 3] * on, fld)
            img += fld[..., None] * (np.array([1.0, 0.55, 0.22], np.float32) * 0.10)[None, None, :]
        # the summit: the cairn, the beacon, the fire, the figures, the torches
        self.summit(img, zb, scam, f, t, md, lv)
        if st > 0.01:
            self.snow(img, scam, f, st)
        if fg > 0.01:
            self.fog_glow(img, scam, f, fg, lv)
        fr = self.fr
        fr.img, fr.zb, fr.dist = img, zb, self.dist
        out, _, _ = PI.to_target(fr)
        return out

    def answers(self, img, zb, scam, f, t, clear):
        K = self.peaks
        ss = self.ss
        join = smoothstep(bar(46, 3), bar(48, 3), f)        # into the hand-back's exact values by the crane
        for i, k in enumerate(K['idx']):
            if f < K['t_on'][i] - 10 and not (k == K['first'] and f >= F_SEEN - 4):
                continue
            on = smoothstep(K['t_on'][i] - 10, K['t_on'][i] + 30, f)
            fl = F.flicker(t + 0.37 * k, k)                   # k = the catalogue index: the hand-back's flicker
            if k == K['first'] and f < K['t_on'][i] + 30:     # bar 33: someone out there kindles a small light
                seen = smoothstep(F_SEEN - 4, F_SEEN + 20, f)
                unsure = 0.55 + 0.45 * math.sin(f * 0.21) * math.sin(f * 0.077 + 1.0)
                on = max(on, 0.30 * seen * unsure)
            near = min(2500.0 / K['dist'][i], 1.0)
            big_hb = 1.8 if k in K['chosen'] else 1.0
            big = big_hb
            if k == K['first']:                               # the first answer: the brightest far light; it
                big = max(big_hb, 2.3 + 1.2 * smoothstep(F_HIS - 6, F_HIS + 30, f))     # flares on bar 41
            big = big + (big_hb - big) * join
            wx = 0.25 + 0.75 * clear
            en = 7.0 * big * on * fl * (0.4 + 0.6 * near) * ss * ss * wx
            F2.glow(img, zb, K['sx'][i], K['sy'][i], 0.8 * ss, en, z=K['z'][i], zbias=K['z'][i] * 0.02,
                    col=np.array([1.0, 0.52, 0.18]))
            F2.halo(img, zb, K['sx'][i], K['sy'][i], 3.0 * ss, 0.026 * big * on * (0.5 + 0.5 * near) * wx,
                    z=K['z'][i], zbias=K['z'][i] * 0.02, col=np.array([1.0, 0.45, 0.12]))

    # --- her and the others
    def her_track(self):
        """Her night as a list of (frame, position, pose): she walks between places (never jumps)."""
        if getattr(self, '_track', None) is not None:
            return self._track
        LIPV = BS.on_ground(BS.STAND + BS.FWD * 1.6 + BS.RIGHT * 0.9)
        ev = [(F0, BS.STAND, 'look')]

        def go(f0, dst, pose, walk=24):
            ev.append((f0, None, 'walk'))
            ev.append((f0 + walk, dst, pose))
        go(bar(22) - 10, LIPV, 'look')                       # the second call: to the lip, searching
        go(bar(23, 3), BS.STAND, 'look')
        for q in FEEDS:
            if q >= bar(42):
                continue
            if q == bar(25, 3):                               # the squall comes: she feeds, then shields it
                go(q - 40, BS.KNEEL, 'feed', 20)
                ev.append((q + 30, BS.KNEEL, 'shield'))
                ev.append((bar(27, 3) - 16, BS.KNEEL, 'feed'))     # still kneeling: from shielding to feeding it
                go(bar(28, 1) + 40, BS.STAND, 'look', 20)
                continue
            if q == bar(27, 3):
                continue
            if q == bar(35, 3):                               # she sees the answer: hails it, then feeds her fire
                ev.append((F_FIRST_ANSWER + 2, LIPV, 'hail'))        # the staff up, until she hurries to feed it
                go(q - 16, BS.KNEEL, 'feed', 16)
                go(q + 34, BS.STAND, 'look', 22)
                continue
            go(q - 40, BS.KNEEL, 'feed', 20)
            if q == bar(39, 3):                               # she stays at the fire and builds it up to the flare
                continue
            go(q + 30, BS.STAND, 'look', 20)
            if q == bar(31, 3):                               # bars 32-35: at the lip, searching the black horizon
                go(bar(32) + 10, LIPV, 'look', 26)
        go(F_FLARE + 30, BS.STAND, 'look', 18)                # her light has flared: she stands and watches
        go(bar(42), BS.SEAT, 'sit', 22)                       # the young carry the flame down; she sits
        for q in (bar(43, 3), bar(45, 3), bar(47, 1)):        # and gets up to feed it each hour
            go(q - 30, BS.KNEEL, 'feed', 16)
            go(q + 34 if q != bar(47, 1) else F_SIT - 22, BS.SEAT, 'sit', 22)
        ev.sort(key=lambda e: e[0])
        self._track = ev
        return ev

    def her_state(self, f):
        """(pose, position, extras) for her at frame f."""
        ev = self.her_track()
        cur = ev[0]
        nxt = None
        for i, e in enumerate(ev):
            if e[0] <= f:
                cur = e
                nxt = ev[i + 1] if i + 1 < len(ev) else None
        if cur[2] == 'walk' and nxt is not None:
            # from the last placed position to the next one
            prev = [e for e in ev if e[0] <= cur[0] and e[1] is not None][-1]
            u = (f - cur[0]) / float(max(nxt[0] - cur[0], 1))
            p = prev[1] + (nxt[1] - prev[1]) * min(max(u, 0.0), 1.0)
            return 'walk', BS.on_ground(p), dict(walk=(f - cur[0]) / 26.0)
        pos = cur[1] if cur[1] is not None else BS.STAND
        ex = dict(since=f - cur[0])
        if cur[2] == 'feed':
            q = min(FEEDS + [F_FLARE], key=lambda x: abs(x - f))
            ex['feed'] = f - q
        return cur[2], pos, ex

    def summit(self, img, zb, scam, f, t, md, lv):
        night_amb = lin('#27335E') * 0.45
        join = smoothstep(bar(47, 3), bar(48, 4), f)          # into the hand-back's exact fire and light
        moon_I = 0.5 * (1.0 - 0.75 * storm(f) - 0.4 * fog(f))
        moon_I = moon_I + (0.45 - moon_I) * join
        fl3 = F.flicker(t, 3)
        fire_I = 1.8 * (lv + (1.0 - lv) * join) * fl3
        lights = [dict(pos=BS.BEACON + np.array([0, 1.3, 0]), col=F.FIRE_LIGHT, I=fire_I, r0=0.5),
                  dict(dir=md, col=lin('#A7BCE0'), I=moon_I)]
        BF.render(img, zb, scam, BP.cairn3(), BS.CAIRN, lights, amb=night_amb * 2.6, t=t,
                  write_depth=True, zbias=0.3)
        back, front, fb = BS.beacon_base()
        eg = 0.3 + 0.7 * min(lv, 1.0)
        BF.render(img, zb, scam, back, BS.BEACON, lights, amb=night_amb, t=t,
                  emissive_gain=eg + (1.0 - eg) * join, write_depth=True, zbias=0.3)
        base = BS.BEACON + np.array([0.0, fb, 0.0])
        wind = 0.2 + 0.9 * storm(f)
        # the vigil's fire, settling into the hand-back's (0.85 / 0.34 / I 12 / lean 0.2 / absorb 0.7) by the crane
        hf = 0.75 + 0.45 * min(lv, 1.4)
        fI = 12.0 * min(lv, 1.5)
        F2.flame(img, zb, scam, base, hf + (0.85 - hf) * join, 0.40 + (0.34 - 0.40) * join, t, seed=4,
                 I=fI + (12.0 - fI) * join, lean=wind + (0.2 - wind) * join, zbias=0.5, tongues=5, warp=1.2,
                 absorb=0.7 * join)
        bx, by, bz = scam.project(base + np.array([0, 0.6, 0]))
        F2.halo(img, zb, bx, by, 5.0 * scam.f / bz, 0.006 * lv + (0.005 - 0.006 * lv) * join, z=bz, zbias=3.0)
        BF.render(img, zb, scam, front, BS.BEACON, lights, amb=night_amb, t=t, write_depth=False,
                  zbias=0.3)
        # travellers on the path (behind/beside her: drawn before her)
        self.travellers(img, zb, scam, f, t, lights, night_amb)
        # her, and from bar 46 the traveller's child: at the crane the hand-back draws the pair itself
        if f >= F_JOIN_FIGS and self.hb is not None:
            hl = [dict(dir=np.array([0.0, -1.0, 0.0]), col=np.array([1.0, 0.62, 0.36]), I=0.0),
                  dict(pos=BS.BEACON + np.array([0, 1.3, 0]), col=F.FIRE_LIGHT, I=1.8 * fl3, r0=0.5),
                  dict(dir=HB_MOON_END(), col=lin('#A7BCE0'), I=0.45)]
            self.hb.figures_at(img, zb, scam, 3840, t, hl, night_amb)
            return
        give = smoothstep(F_SHAWL - 6, F_SHAWL + 24, f)       # she wraps the child in her shawl
        if f >= F_CHILD_IN:
            self.child(img, zb, scam, f, t, lights, night_amb, give)
        pose, pos, ex = self.her_state(f)
        kw = dict(age=0.9, shawl=give < 0.5, staff=pose in ('look', 'walk', 'sit', 'hail'),
                  wind=(0.4 if pose == 'sit' else 0.5) + 1.2 * storm(f), walk=ex.get('walk', 0.0) % 1.0,
                  reach=math.sin(math.pi * give) * 0.9 if pose == 'sit' else 0.0)
        d, pts = BS.person2('kneel' if pose == 'feed' else pose, **kw)
        since = ex.get('since', 99)
        p_prev = self.her_state(f - since - 1)[0] if since < 8 else pose
        if since < 8 and p_prev != pose and 'walk' not in (p_prev, pose):   # a pose change: an 8-frame crossfade
            d0, _ = BS.person2('kneel' if p_prev == 'feed' else p_prev, **dict(kw, staff=p_prev in ('look', 'sit', 'hail')))
            ia = img.copy()
            BF.render(ia, zb.copy(), scam, d0, pos, lights, amb=night_amb, t=t, write_depth=False, zbias=0.3)
            BF.render(img, zb, scam, d, pos, lights, amb=night_amb, t=t, write_depth=False, zbias=0.3)
            w = smoothstep(0.0, 8.0, since)
            img[:] = ia * (1.0 - w) + img * w
        else:
            BF.render(img, zb, scam, d, pos, lights, amb=night_amb, t=t, write_depth=False, zbias=0.3)
        if pose == 'feed':
            q = ex.get('feed', 99)
            if 0 <= q < 26:            # sparks as the wood takes
                rng = np.random.default_rng(int(f))
                for i in range(14):
                    age = (q + rng.random() * 6) / 26.0
                    p = base + np.array([rng.normal(0, 0.12), 0.3 + 1.6 * age + rng.random() * 0.2, rng.normal(0, 0.12)])
                    sx, sy, sz = scam.project(p)
                    F2.glow(img, zb, sx, sy, 0.5 * self.ss, 0.5 * self.ss * self.ss * (1.0 - age), z=sz, zbias=0.5,
                            col=np.array([1.0, 0.55, 0.2]))

    def child_pos(self):
        """Where the child sits against her: the hand-back's place (so the crane starts from this frame)."""
        try:
            import handback_b as HB
            return HB.child_pos()
        except Exception:
            return BS.on_ground(BS.SEAT + BS.RIGHT * 0.6)

    def child(self, img, zb, scam, f, t, lights, amb, give):
        """Bar 46: the child comes up the path beside a traveller; stays by her (bar 47); sits against her (bar 48);
        is wrapped in her shawl and falls asleep."""
        arrive = bar(46) + 50
        cp = self.child_pos()
        if f < arrive:
            s = 14.0 * (1.0 - (f - F_CHILD_IN) / float(arrive - F_CHILD_IN - 50))
            if s > 0.0:
                p = vpath_at(s) + BS.dirxz(self.cam_yaw + 90.0) * 0.55
            else:
                u = min((f - (arrive - 50)) / 50.0, 1.0)
                p = vpath_at(0.0) + (cp - vpath_at(0.0)) * u
            d, _ = BS.person2('walk', age=0.0, shawl=False, staff=False, child=True, walk=(f / 16.0) % 1.0)
        elif f < F_SIT:
            u = min((f - arrive) / 30.0, 1.0)
            p = cp
            d, _ = BS.person2('stand' if u >= 1.0 else 'walk', age=0.0, shawl=False, staff=False, child=True,
                              walk=(f / 16.0) % 1.0)
        else:
            p = cp
            if give >= 0.5:
                d, _ = BP.child3(0.0)                         # wrapped in her shawl, asleep against her
            else:
                d, _ = BS.person2('sit', age=0.0, shawl=False, staff=False, child=True, lean_to=-1.0)
        BF.render(img, zb, scam, d, BS.on_ground(p), lights, amb=amb, t=t, write_depth=False, zbias=0.3)

    # the travellers: (arrival frame, looks). They come up the last of the NE path with unlit torches, light them
    # at the basket and go home down the ridge: the thread of torches grows through the night.
    LOOKS = [dict(build=0.85, hood=0, pack=1, coat=0.6, cloak=0),
             dict(build=0.25, hood=1, pack=0, coat=0.6, cloak=14),
             dict(build=0.60, hood=2, pack=2, coat=1.0, cloak=0),
             dict(build=0.45, hood=0, pack=0, coat=1.0, cloak=14),
             dict(build=0.95, hood=1, pack=2, coat=0.6, cloak=14),
             dict(build=0.15, hood=0, pack=1, coat=0.6, cloak=0),
             dict(build=0.70, hood=2, pack=0, coat=0.6, cloak=14)]
    # (frame the group's first torch takes, how many). The first on 'traveller' (bar 37 b1); three together
    # (bar 38 b3); then more through bars 40-45; the last is the child's parent (the pair top the path on bar 46)
    TRIPS = [(F_TRAV1_LIGHT, 1), (bar(38, 3), 3), (bar(40, 4), 2), (bar(42, 2), 2), (bar(43, 2), 2),
             (bar(44, 2), 2), (bar(45, 2), 1), (bar(46) + 24, 1)]

    def travellers(self, img, zb, scam, f, t, lights, amb):
        ss = self.ss
        fg = fog(f)
        UP, AT = 100, 44
        n_look = 0
        for k0, (t0, n) in enumerate(self.TRIPS):
            for m in range(n):
                look = self.LOOKS[(n_look + 3 * k0) % len(self.LOOKS)]
                n_look += 1
                s0 = t0 - UP - 24 + m * 24                  # t0 = the moment this group's first torch takes
                if f < s0:
                    continue
                rs = BS.dirxz(self.cam_yaw + 90.0)          # screen-right on the ground
                side_off = rs * (0.45 * m - 0.2)
                lip = vpath_at(0.0)
                tgt = BS.on_ground(BS.BEACON + rs * (1.0 + 0.55 * m) + BS.dirxz(self.cam_yaw) * 0.5)
                if f < s0 + UP:                               # the last 14 m of the climb, unlit
                    u = (f - s0) / float(UP)
                    p = vpath_at(14.0 * (1.0 - u)) + side_off
                    lit = False
                    walk = (f - s0) / 22.0
                    pose = 'walk'
                elif f < s0 + UP + AT:                        # to the basket; the torch takes
                    u = (f - s0 - UP) / float(AT)
                    p = BS.on_ground(lip + (tgt - lip) * min(u * 2.0, 1.0))
                    lit = u >= 24.0 / AT
                    walk = (f - s0) / 22.0 if u < 0.5 else 0.0
                    pose = 'walk' if u < 0.5 else 'stand'
                else:                                         # home: back over the lip, then a light going down
                    g = f - s0 - UP - AT
                    if g < 30:
                        p = BS.on_ground(tgt + (lip - tgt) * (g / 30.0))
                    else:
                        gg = g - 30
                        s = 0.10 * gg + 0.0040 * gg * gg
                        if s > vpath()[1][-1] - 5.0:
                            continue
                        p = vpath_at(s) + side_off * 0.6
                    lit = True
                    walk = g / 22.0
                    pose = 'walk'
                dd = np.linalg.norm(p - scam.pos)
                rgt = np.array([scam.right[0], 0.0, scam.right[2]])
                fl = F.flicker(t + 0.53 * (k0 * 3 + m), k0 * 3 + m + 11)
                if dd < 220.0:
                    d, pts = BS.person2(pose, age=0.2, shawl=False, staff=False, torch=True, walk=walk % 1.0, **look)
                    tp = p + np.array([0.0, pts['torch'][1], 0.0]) + rgt * pts['torch'][0]
                    lk = list(lights)
                    if lit:                                    # the torch lights its bearer
                        lk.append(dict(pos=tp + np.array([0.0, 0.15, 0.0]), col=F.FIRE_LIGHT, I=0.55 * fl, r0=0.25))
                    BF.render(img, zb, scam, d, p, lk, amb=amb, t=t, write_depth=False, zbias=0.3)
                else:
                    tp = p + np.array([0.0, 1.7, 0.0])
                if lit:
                    sx, sy, sz = scam.project(tp)
                    if dd < 220.0:
                        F2.flame(img, zb, scam, tp, 0.30, 0.12, t + m, seed=9 + m, I=5.0, lean=0.3, zbias=0.3,
                                 tongues=3, warp=1.0)
                    en = 6.0 * fl * ss * ss * min(1.0, 300.0 / dd) ** 0.6
                    F2.glow(img, zb, sx, sy, 0.7 * ss, en, z=sz, zbias=sz * 0.02, col=np.array([1.0, 0.55, 0.2]))
                    F2.halo(img, zb, sx, sy, (2.5 + 9.0 * fg) * ss, (0.01 + 0.03 * fg) * min(1.0, 400.0 / dd),
                            z=sz, zbias=sz * 0.02, col=np.array([1.0, 0.45, 0.12]))

    def snow(self, img, scam, f, st):
        """Driving snow in the squall: fine short streaks blowing across the frame (left to right, falling), nearer
        flakes longer, brighter and softer; the flakes passing near her fire catch its warm light."""
        import cv2
        rng = np.random.default_rng(99)
        W, H = scam.W, scam.H
        k = W / 1920.0
        n = int(3400 * st)
        x0 = rng.random(n) * W * 1.3 - 0.15 * W
        y0 = rng.random(n) * H
        z = 0.6 + rng.random(n) ** 1.2 * 6.0                   # most flakes far and faint; a few near
        spd = (22.0 / z) * k
        x = (x0 + spd * f) % (W * 1.3) - 0.15 * W
        y = (y0 + spd * 0.42 * f + 3.0 * np.sin(f * 0.11 + x0 * 0.01)) % H
        L = np.minimum(spd * 0.40, 14.0 * k) * (0.6 + 0.8 * rng.random(n))
        a = (0.03 + 0.13 / z) * st * (0.5 + rng.random(n))
        lay = np.zeros((H, W), np.float32)
        for i in range(n):
            cv2.line(lay, (int(x[i]), int(y[i])), (int(x[i] + L[i]), int(y[i] + 0.42 * L[i])), float(a[i]), 1,
                     cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), 0.9 * max(k * 1.5, 0.5))
        bx, by, bz = scam.project(BS.BEACON + np.array([0.0, 1.0, 0.0]))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        warm = np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (0.09 * W) ** 2)) * min(fire_level(f), 1.2)
        col = lin('#9FB0D0')[None, None, :] * 0.55 * (1.0 - warm[..., None]) + \
            np.array(F.FIRE_LIGHT, np.float32)[None, None, :] * 1.4 * warm[..., None]
        img += lay[..., None] * col

    def fog_glow(self, img, scam, f, fg, lv):
        """In the fog the fire makes a great soft sphere of light round the summit."""
        bx, by, bz = scam.project(BS.BEACON + np.array([0.0, 1.2, 0.0]))
        H, W = img.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2) / (0.10 * W)
        g = np.exp(-r * r * 1.5) * fg * 0.035 * lv
        img += g[..., None] * np.array(F.FIRE_LIGHT, np.float32)[None, None, :]


FINISH = dict(exposure=1.15, bloom_strength=0.06, bloom_threshold=1.2, streak_strength=0.0, vignette_amount=0.22)


def set_cam(spec):
    if spec:
        for k, v in zip(_VK, spec.split(',')):
            VCAM[k] = float(v)


def _work(args):
    frames, scale, ss, out, cam = args
    set_cam(cam)
    shot = Vigil(scale, ss)
    for f in frames:
        t1 = time.time()
        img = look.finish(shot.render(f), **FINISH)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t1:.2f}s', flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--range', default=None)
    ap.add_argument('--frames', default=None)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default='vig_t')
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--cam', default=None, help='bear,dist,up,yaw,pitch,hfov (look-dev)')
    a = ap.parse_args()
    set_cam(a.cam)
    t0 = time.time()
    shot = Vigil(a.scale, a.ss)              # builds every cache once, before any fork
    print(f'G-buffer {time.time() - t0:.1f}s; answers {len(shot.peaks["idx"])}; villages {len(shot.vill["V"])}',
          flush=True)
    if a.build:
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
    if a.procs <= 1:
        _work((frames, a.scale, a.ss, out, a.cam))
    else:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(frames[i::a.procs], a.scale, a.ss, out, a.cam) for i in range(a.procs)])


if __name__ == '__main__':
    main()
