"""B2 . THE CLIMB (H4) on B's own landform: frames 640-879 of cut B (bars 9-11). Lane HEROINE-B.

The locked sheet: blue night. A tiny figure on a snow arete carries a clay fire-pot in both hands; its faint warm glow
is the only warm point in a blue world. Spindrift. On bar 11 b1 (800) we are closer, behind her: the red shawl and the
pot. Wind; the cello's last note held. "Someone carrying the last of the light up to where it left."

  WIDE  640-799  locked, from the NW side of the NE ridge, a little above the crest: her figure (40-60 px) walks right
                 along the moonlit crest toward the rounded summit, where the cold beacon and the counting cairn stand
                 on the skyline; spindrift streams off the crest toward us, backlit by the low moon; the pot's glow is
                 the only warm point.
  CLOSE 800-879  locked, behind her right shoulder and a little above: her back (hood, the red woven shawl, the long
                 cloak) in moonlight; the pot in both gloved hands past her right hip, its light leaking from under the
                 lid and pooling warm on the snow ahead of her; her long moon shadow; the summit ahead.

World = RUN-B's (read-only): bworld's landform (B's own), bset's set points / cairn / beacon, the vigil's moon (az 80,
el 14) and night grade (bset.night_params, the vigil's FINISH). Her = HEROINE's 3-D figure (heroine_v3.figure: the hood
/ wool cowl, the red woven shawl, thin leather gloves) with a long wool cloak, carrying B3's crafted clay pot
(heroine_v3.DeadEmber.pot, the same vessel), traced by hsdf3 in the target camera and depth-composited over the world.
Each locked camera marches bworld ONCE (G-buffer cached per BW.VERSION) and re-shades per frame.

  python3 shots/hills/climb_b.py --layout
  python3 shots/hills/climb_b.py --frames 640,720,800,860 --scale 0.25 --out climb_t1
  python3 shots/hills/climb_b.py --range 640-879 --scale 1.0 --out renders/climb_B --skip
"""
import math
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.abspath(os.path.join(HERE, '..', 'run'))
for _p in (RUN, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bworld as BW         # noqa: E402  (RUN-B-3's, read-only)
import bset as BS           # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import fire2 as F2          # noqa: E402
import keeper as KP         # noqa: E402
import vigil as VG          # noqa: E402
from mt import fire as MF, figure as FG   # noqa: E402

import heroine as hero      # noqa: E402  (HEROINE-L's, read-only)
import hsdf3 as H3          # noqa: E402
import heroine_v3 as HV     # noqa: E402
from core import Camera, smoothstep, fnoise1   # noqa: E402

CM = PI.CM
look = PI.look
lin = CM.lin
CR = BW.CR_B
FPS = 24.0
ROOT = CM.ROOT
TESTS = os.path.join(ROOT, 'renders', 'heroine_tests')
CACHE = os.path.join(TESTS, 'cache_climb')

F0, FC, F1 = 640, 800, 880            # WIDE 640-799 (bars 9-10), CLOSE 800-879 (bar 11 b1)

# the night: the vigil's moon (low in the ENE), its colour, the sky wheel frozen at the reveal's first frame
MOON_AZ, MOON_EL = VG.moon_at(VG.F0)
MOON = BS.moon_vec(MOON_EL, MOON_AZ)
MOON_LIN = lin('#A7BCE0')
SKY_ANG = VG.sky_angle(VG.F0) - math.radians(VG.STAR_DEG / (VG.F1 - VG.F0)) * (VG.F0 - 1360)
F_SKY = 1360                          # the climb is real time: the sky does not wheel; it is the reveal's first sky
FINISH = dict(exposure=1.15, bloom_strength=0.06, bloom_threshold=1.2, streak_strength=0.0, vignette_amount=0.22)
NIGHT_AMB = lin('#27335E') * 0.45
WIND_AZ = 314.0                       # the wind blows toward the NW: off the crest toward the wide's lens
WIND = BS.dirxz(WIND_AZ)


def dirxz(az):
    return BS.dirxz(az)


def az_of(v):
    return math.degrees(math.atan2(v[0], v[2])) % 360.0


# ------------------------------------------------------------------ her walk on the NE path ---
class Walk:
    """She trudges up the NE path toward the lip. Local frame per shot: origin = the path point at the shot's first
    frame (on the ground), +z = her heading (toward the lip), +x = her right, y up. world = base + R @ local."""
    SPEED = 0.30                       # m/s: slow and heavy, the pot held steady
    STEP = 1.25                        # s per stride pair

    def __init__(self, s0, f0):
        self.f0 = f0
        # bset's path is a straight chord from the lip to its first crest point (30 m): it floats up to ~1.3 m above
        # the rounded crest there, so put her on the ground under it
        self.base = BS.on_ground(BS.path_at(s0))
        ahead = BS.on_ground(BS.path_at(max(s0 - 6.0, 0.0)))
        self.heading = az_of(ahead - self.base)
        a = math.radians(self.heading)
        self.R = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])

    def to_world(self, p):
        return self.base + self.R @ np.asarray(p, np.float64)

    def to_local(self, P):
        return self.R.T @ (np.asarray(P, np.float64) - self.base)

    def ground(self, x, z):
        W = self.to_world([x, 0.0, z])
        return BS.ground(W[0], W[2]) - self.base[1]

    def pose(self, f):
        """(pose dict for heroine.build_figure, pelvis, pot base centre c, lid glow point) in the local frame."""
        t = (f - self.f0) / FPS
        zc = self.SPEED * t
        ph = (t / self.STEP) % 1.0
        stride = self.SPEED * self.STEP

        def foot(off, side):
            q = (ph + off) % 1.0
            k = math.floor(t / self.STEP + off)
            swing = smoothstep(0.55, 1.0, q)
            z = (k + swing - 0.5) * stride + 0.06
            x = 0.095 * side
            y = self.ground(x, z) + 0.01 + 0.08 * math.sin(math.pi * swing) * (q > 0.55)   # planted feet sink in
            return np.array([x, y, z])
        fl, fr = foot(0.0, -1.0), foot(0.5, 1.0)
        bob = 0.022 * math.cos(4 * math.pi * ph)
        gy = 0.5 * (self.ground(-0.1, zc) + self.ground(0.1, zc))
        pel = np.array([0.012 * math.sin(2 * math.pi * ph), gy + 0.86 + bob, zc])
        fwd = np.array([0.0, 0.0, 1.0])
        pot = pel + np.array([0.12, 0.27, 0.31])          # held out in front, a little to her right, both hands round it
        c = pot - np.array([0.0, 0.075, 0.0])            # the pot's foot (DeadEmber.pot's c)
        p = dict(
            pelvis=tuple(pel), yaw=-90.0, lean=18.0, chest=9.0, twist=2.0 * math.sin(2 * math.pi * ph), neck=12.0,
            head=26.0, head_yaw=0.0, head_roll=2.0 * math.sin(2 * math.pi * ph), shrug=0.5,
            hand_n=tuple(c + np.array([-0.084, 0.058, -0.006])), hand_f=tuple(c + np.array([0.086, 0.058, -0.004])),
            elbow_n=(-0.6, -1.0, -0.3), elbow_f=(0.6, -1.0, -0.3),
            fdir_n=tuple(HV.nrm([0.30, -0.25, 0.9])), palm_n=(1.0, 0.1, 0.0),
            fdir_f=tuple(HV.nrm([-0.30, -0.25, 0.9])), palm_f=(-1.0, 0.1, 0.0),
            curl_n=(0.50, 0.55, 0.60, 0.64), curl_f=(0.50, 0.55, 0.60, 0.64), thumb_n=0.35, thumb_f=0.35,
            foot_n=tuple(fl), foot_f=tuple(fr), knee_n=tuple(fwd + [0, 0.3, 0]), knee_f=tuple(fwd + [0, 0.3, 0]),
            toe_n=tuple(fwd), toe_f=tuple(fwd), sole_n=(0.0, 1.0, 0.0), sole_f=(0.0, 1.0, 0.0),
            hem=0.30, breath=math.sin(2 * math.pi * t / 2.6), tools='none', rock=None,
            expr=dict(squint=0.7), look_at=tuple(pel + np.array([0.0, 0.2, 3.0])),
        )
        glow = c + np.array([0.026, 0.150, -0.030])      # the chip in the lip: where the most light leaks
        return p, pel, c, glow, t, ph


# ------------------------------------------------------------------ her figure (3-D, hsdf3) ---
def build_her(walk, f, wind=1.0):
    """The Builder for her at frame f (local frame): HEROINE's figure + a long wool cloak + B3's crafted pot, lid on."""
    p, pel, c, glow, t, ph = walk.pose(f)
    B, F, Hp, anc, hair = HV.figure(p, t, hair_dir=(0.9, -0.3), scarf=False)
    # the shawl's tail: short, blown to her right by the wind from her left
    sa = anc['scarf_anchor']
    tail = HV.static_chain(np.array([sa[0], sa[1]]), (0.55, -0.80), 7, 0.070, droop=0.35, sway=(0.06, 0.10),
                           seed=1.0 - 3.1 * t)
    B.group('scarf')
    hero.scarf_tail(B, tail, t, z0=sa[2] - 0.03)
    # the long cloak: dark undyed wool hanging from under the hood's cape (so its top edge is covered) to the calves,
    # falling plumb behind her forward lean; open at the front, where her arms come forward to the pot (a back shell:
    # the cone cut by a half-space through her chest)
    J = hero.skeleton(p)
    C7, Ut, Vt = J['C7'], J['Ut'], J['Vt']
    sw = math.sin(2 * math.pi * ph)
    B.group('cloak', H3.M_HOOD, disp=1, amp=0.0015, scale=30.0, band=0.008)
    top = C7 - 0.065 * Ut - 0.045 * Vt
    hemc = np.array([pel[0] + 0.035 * wind + 0.02 * sw, walk.ground(0.0, pel[2] - 0.12) + 0.27, pel[2] - 0.12])
    B.cone(top, hemc, 0.118, 0.325, k=0.04,
           fold=dict(amp=0.016, wl=0.15, ph=0.7 + 0.9 * sw, s0=0.30, sw=0.70, wob=2.2, wph=1.3 + 0.8 * t))
    # half-space (the plane's local +x = outside): R[0] = +z keeps what is behind her chest
    B.plane(np.array([0.0, pel[1], pel[2] + 0.09]), np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            op=2, reach=1.6)
    # the pot: B3's crafted vessel (same geometry), lid on
    de = HV.DeadEmber()
    de.POT = c
    de.pot(B, c + np.array([0.0, 0.150, 0.0]), (0.0, 1.0))
    return B, Hp, anc, pel, c, glow


def local_camera(walk, tcam):
    """The hills Camera for tcam (a world RCam at output res), expressed in her local frame."""
    pos = walk.to_local(tcam.pos)
    yaw = math.radians(tcam.yaw_d - walk.heading)
    return Camera(pos, yaw=yaw, pitch=math.radians(tcam.pitch_d), hfov=tcam.hfov_d, scale=tcam.W / 1920.0)


def her_layer(walk, f, tcam, ss=3, pot_I=1.0):
    """RGB premultiplied + alpha + 3-D ray distance for her, in the target camera (output res)."""
    B, Hp, anc, pel, c, glow = build_her(walk, f)
    cam = local_camera(walk, tcam)
    moon_l = walk.R.T @ MOON
    Hp[40] = 0.0
    L = [HV.moon_light(0.55, moon_l),
         HV.light(glow + np.array([0.0, 0.004, 0.0]), (1.0, 0.36, 0.08), 0.0045 * pot_I, 0.02, 4.0)]
    env = hero.env_vec(rim_dir=moon_l, rim=np.array([0.10, 0.13, 0.20]), amb=np.array([0.005, 0.007, 0.013]),
                       bounce=np.array([0.010, 0.012, 0.018]), ao=0.03)
    res = H3.render(cam, B, Hp, np.asarray(L, np.float64), env, ss=ss,
                    sil=dict(skin=1.0, eyes=1.0, cap=0.3, cap_brim=0.3, hair=0.3))
    return res, walk.to_world(glow), walk.to_world(c)


# ------------------------------------------------------------------ the shots ---
class Shot:
    """One locked camera on bworld: the G-buffer marched once, re-shaded per frame."""
    name = ''

    def __init__(self, scale, ss, walk):
        self.scale, self.ss = scale, ss
        self.W, self.H = int(round(1920 * scale)), int(round(804 * scale))
        self.walk = walk
        self.tc = self.camera(self.W, self.H)
        self.fr = PI.Frame(self.tc, ss)
        scam = self.fr.src
        self.P = np.array([scam.pos[0], scam.pos[2], 0.0, 0.0])
        p = os.path.join(CACHE, f'{self.name}_G_{BW.VERSION}_{self.key()}_{scale:.3f}_{ss:.2f}.npy')
        if os.path.exists(p):
            self.G = np.load(p)
        else:
            self.G = BW.build(scam, self.P, CR, None, dmax=180000.0, moons=[MOON], mk=10.0)
            os.makedirs(CACHE, exist_ok=True)
            tmp = p + f'.{os.getpid()}.tmp.npy'
            np.save(tmp, self.G)
            os.replace(tmp, p)
        self.dist = self.G[..., BW.G_DIST].astype(np.float32)
        self.sky = (self.dist > 1e8).astype(np.float32)
        # per-pixel |ray| / |ray_h| of the target camera: horizontal range -> 3-D distance
        tt = self.tc
        us, vs = np.meshgrid(np.arange(tt.W) + 0.5, np.arange(tt.H) + 0.5)
        d = tt.ray(us, vs)
        self.k3 = (np.linalg.norm(d, axis=-1) / np.maximum(np.hypot(d[..., 0], d[..., 2]), 1e-9)).astype(np.float32)

    def key(self):
        c = self.camera(1920, 804)
        return f'{c.pos[0]:.1f}_{c.pos[1]:.1f}_{c.pos[2]:.1f}_{c.yaw_d:.2f}_{c.pitch_d:.2f}_{c.hfov_d:.1f}'

    # --- the world, relit per frame (the pot's light and her moon shadow added in source space)
    def world(self, f, glow_w, pot_I):
        scam = self.fr.src
        LP, SN, amb, fogp = BS.night_params(MOON, 1.0 / scam.f)
        if hasattr(BS, 'match_horizon'):           # RUN-B2's vigil: the far haze melts into the horizon sky
            BS.match_horizon(SN, fogp)
        LP[32], LP[33], LP[34] = 0, 0, 0.0
        img = np.zeros((scam.H, scam.W, 3), np.float32)
        BW.shade(self.G, LP, SN, amb, fogp, float(scam.pos[1]), img)
        # her moon shadow: re-shade the pixels near her without the moon and blend by her shadow mask
        box = self.near_box(scam, 9.0)
        if box is not None:
            y0, y1, x0, x1 = box
            Gc = np.ascontiguousarray(self.G[y0:y1, x0:x1])
            LP2 = LP.copy()
            LP2[23] = 0.0
            nm = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
            BW.shade(Gc, LP2, SN, amb, fogp, float(scam.pos[1]), nm)
            sh = self.her_shadow(Gc, f)
            sub = img[y0:y1, x0:x1]
            sub[:] = nm + (sub - nm) * (1.0 - sh[..., None])
            # the pot's light on the snow (her body and the pot itself shadow it)
            sub += self.pot_light(Gc, glow_w, pot_I, f)
        return img

    def near_box(self, scam, rad):
        pel = self.walk.to_world(self.walk.pose(self.walk.f0 + 40)[1])      # mid-shot
        pts = []
        for dx in (-rad, rad):
            for dz in (-rad, rad):
                for dy in (-3.0, 2.0):
                    pts.append(pel + np.array([dx, dy, dz]))
        pts = np.array(pts)
        sx, sy, z = scam.project(pts)
        if np.any(z <= 0.3):
            return 0, scam.H, 0, scam.W
        x0, x1 = int(max(0, sx.min() - 4)), int(min(scam.W, sx.max() + 4))
        y0, y1 = int(max(0, sy.min() - 4)), int(min(scam.H, sy.max() + 4))
        if x1 <= x0 or y1 <= y0:
            return None
        return y0, y1, x0, x1

    def her_body_world(self, f):
        """Her body for shadows: capsules (a, b, r) in world metres."""
        p, pel, c, glow, t, ph = self.walk.pose(f)
        g = self.walk.ground(0.0, pel[2])
        caps = [((0.0, g + 0.10, pel[2] - 0.06), (0.0, pel[1] + 0.52, pel[2] + 0.06), 0.24),   # cloak + torso
                ((0.0, pel[1] + 0.52, pel[2] + 0.08), (0.0, pel[1] + 0.78, pel[2] + 0.16), 0.13),  # hood
                (tuple(c + np.array([0.0, 0.07, 0.0])), tuple(c + np.array([0.0, 0.07, 0.0])), 0.085)]  # pot
        return [(self.walk.to_world(a), self.walk.to_world(b), r) for a, b, r in caps]

    def her_shadow(self, Gc, f):
        X = np.stack([Gc[..., BW.G_X], Gc[..., BW.G_Y], Gc[..., BW.G_Z]], -1).astype(np.float64)
        ok = (Gc[..., BW.G_DIST] < 1e8) & (Gc[..., BW.G_FLAG] != 2.0)
        caps = self.her_body_world(f)[:2]
        return _shadow_caps(X, ok, MOON, np.array([np.r_[a, b, r] for a, b, r in caps], np.float64), 0.10, 14.0)

    def pot_light(self, Gc, glow_w, pot_I, f):
        X = np.stack([Gc[..., BW.G_X], Gc[..., BW.G_Y], Gc[..., BW.G_Z]], -1).astype(np.float64)
        N = np.stack([Gc[..., BW.G_NX], Gc[..., BW.G_NY], Gc[..., BW.G_NZ]], -1).astype(np.float64)
        ok = (Gc[..., BW.G_DIST] < 1e8) & (Gc[..., BW.G_FLAG] != 2.0)
        caps = self.her_body_world(f)
        out = _point_light(X, N, ok, np.asarray(glow_w, np.float64),
                           np.array([np.r_[a, b, r] for a, b, r in caps], np.float64), 0.05)
        col = np.asarray(MF.FIRE_LIGHT, np.float32) * np.float32(0.030 * pot_I)
        return out[..., None].astype(np.float32) * col[None, None, :]

    def summit(self, img, zb, scam, t):
        """The counting cairn and the cold beacon, exactly as the vigil draws them (no fire yet)."""
        lights = [dict(dir=MOON, col=MOON_LIN, I=0.5)]
        FG.render(img, zb, scam, BS.rubble_cairn(), BS.CAIRN, lights, amb=NIGHT_AMB, mats=BS.M, t=t,
                  write_depth=True, zbias=0.3)
        back, front, fb = BS.beacon_base()
        FG.render(img, zb, scam, back, BS.BEACON, lights, amb=NIGHT_AMB, mats=BS.M, t=t, emissive_gain=0.0,
                  write_depth=True, zbias=0.3)
        FG.render(img, zb, scam, front, BS.BEACON, lights, amb=NIGHT_AMB, mats=BS.M, t=t, emissive_gain=0.0,
                  write_depth=False, zbias=0.3)

    def pot_I(self, f):
        t = f / FPS
        return 1.0 + 0.10 * fnoise1(t * 2.2, 3.0, 2) + 0.05 * fnoise1(t * 7.0, 11.0, 1)

    def render(self, f):
        t = f / FPS
        scam = self.fr.src
        pI = self.pot_I(f)
        res, glow_w, c_w = her_layer(self.walk, f, self.fr.t_out, ss=3, pot_I=pI)
        img = self.world(f, glow_w, pI)
        zb = self.dist.copy()
        if hasattr(VG, 'draw_sky'):                # RUN-B2's one sky (vigil + hand-back), frozen at the reveal's start
            VG.draw_sky(img, self.G, scam, self.sky, F_SKY, 1.0, self.ss)
        else:
            BW.add_band(self.G, KP._rotmat(KP.POLE, SKY_ANG), VG.BAND, 0.06, 1.0, img)
            KP.draw_stars(img, scam, self.sky, SKY_ANG - 0.0004, SKY_ANG, 1.6 * self.ss * self.ss, K=3)
        self.summit(img, zb, scam, t)
        fr = self.fr
        fr.img, fr.zb, fr.dist = img, zb, self.dist
        out, zt, _ = PI.to_target(fr)
        out = np.ascontiguousarray(out, np.float32)
        d3 = zt * self.k3
        # her, depth-tested against the snow (her boots sink a little into it)
        if res is not None:
            y0, x0, rgb, a, dd, _ = res
            h, w = a.shape
            vis = (dd < d3[y0:y0 + h, x0:x0 + w] + 0.04).astype(np.float32)
            aa = (a * vis).astype(np.float32)
            sub = out[y0:y0 + h, x0:x0 + w]
            sub *= (1.0 - aa[..., None])
            sub += (rgb * vis[..., None]).astype(np.float32)
            d3[y0:y0 + h, x0:x0 + w] = np.where(aa > 0.5, dd, d3[y0:y0 + h, x0:x0 + w])
        self.pot_glow(out, d3, glow_w, pI)
        self._glow_w = np.asarray(glow_w, np.float64)
        self.spindrift(out, d3, f)
        return out

    def pot_glow(self, out, d3, glow_w, pI):
        tc = self.fr.t_out
        sx, sy, z = tc.project(glow_w)
        if z <= 0.1:
            return
        z = float(np.linalg.norm(np.asarray(glow_w) - tc.pos))      # d3 holds 3-D distances
        s = tc.W / 1920.0
        ppm = tc.f / z
        # the light leaking under the lid: a small warm core and a faint veil round it
        MF.glow(out, d3, float(sx), float(sy), max(0.6, 0.012 * ppm), 1.2 * pI * s * s * self.core_gain,
                z=float(z), zbias=0.3, col=np.array([1.0, 0.50, 0.16]))
        MF.halo(out, d3, float(sx), float(sy), max(2.0, 0.45 * ppm), 0.020 * pI * self.halo_gain, z=float(z),
                zbias=0.6, col=np.array([1.0, 0.42, 0.12]))

    core_gain = 1.0
    halo_gain = 1.0

    # --- spindrift: grains torn off the snow by the wind (toward the NW), lit by the low moon (forward scatter) and,
    # near her, by the pot
    N_GRAINS = 0

    def sources(self, rng, n):
        """(n, 3) world emission points on the snow (override)."""
        raise NotImplementedError

    def gust(self, t, P):
        """0..1 gust strength at time t for emission points P (a wave travelling downwind along the ridge)."""
        u = (P[:, 0] * WIND[0] + P[:, 2] * WIND[2]) * 0.02 + (P[:, 0] * WIND[2] - P[:, 2] * WIND[0]) * 0.035
        g = 0.5 + 0.5 * np.sin(2 * math.pi * (0.16 * t - u) + 1.3) * np.sin(2 * math.pi * (0.07 * t + 0.4 * u) + 0.2)
        return np.clip((g - 0.25) / 0.75, 0.0, 1.0) ** 1.5

    def grains(self, f, shutter=0.5):
        """Positions at shutter open/close, energies and radii of the grains alive at frame f (deterministic)."""
        n = self.N_GRAINS
        if getattr(self, '_src', None) is None:
            self._src = self.sources(np.random.default_rng(self.SEED), n)
        P = self._src
        rng = np.random.default_rng(self.SEED + 1)
        LIFE = rng.uniform(*self.LIFE, n)
        T0 = rng.uniform(F0 / FPS - self.LIFE[1], F1 / FPS, n)           # emission times over the whole climb
        V = rng.uniform(*self.SPEED, n)
        LIFT = rng.uniform(*self.LIFT, n)
        SW = rng.uniform(0.2, 1.0, n) * self.SWIRL
        PH = rng.uniform(0, 2 * math.pi, (n, 3))
        OM = rng.uniform(2.0, 6.0, (n, 3))
        B = rng.random(n) ** 3.0                                          # a few bright grains, many faint
        gate = rng.random(n)
        t0 = f / FPS
        out = []
        for dt in (0.0, shutter / FPS):
            t = t0 + dt
            a = t - T0
            alive = (a > 0) & (a < LIFE) & (gate < self.gust(T0, P))
            side = np.array([WIND[2], 0.0, -WIND[0]])
            hor = V[:, None] * a[:, None] * WIND[None, :] * (1.0 - 0.25 * np.clip(a / LIFE, 0, 1))[:, None]
            sw = (SW[:, None] * (np.sin(OM[:, 0:1] * a[:, None] + PH[:, 0:1]) * side[None, :]
                                 + 0.4 * np.sin(OM[:, 1:2] * a[:, None] + PH[:, 1:2]) * WIND[None, :]))
            up = LIFT * (1.0 - np.exp(-a / 0.5)) - 0.35 * a + 0.25 * SW * np.sin(OM[:, 2] * a + PH[:, 2])
            Q = P + hor + sw
            Q[:, 1] += up
            out.append((Q, alive))
        (Q0, al0), (Q1, al1) = out
        a = t0 - T0
        fade = np.clip(a / 0.25, 0, 1) * np.clip((LIFE - a) / (0.45 * LIFE), 0, 1)
        m = al0 & al1
        return Q0[m], Q1[m], (B * fade)[m], m

    def spindrift(self, out, d3, f):
        if self.N_GRAINS <= 0:
            return
        Q0, Q1, e, m = self.grains(f)
        if len(Q0) == 0:
            return
        tc = self.fr.t_out
        s = tc.W / 1920.0
        # forward scatter of the moon (Henyey-Greenstein, g 0.55) toward the lens
        v = tc.pos[None, :] - Q0
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        cth = v @ MOON
        g = 0.55
        hg = (1 - g * g) / (1 + g * g - 2 * g * cth) ** 1.5
        col = MOON_LIN[None, :] * (self.GRAIN_E * e * (0.35 + 0.65 * hg))[:, None] * s * s
        # near the pot the grains catch its warm light
        gw = getattr(self, '_glow_w', None)
        if gw is not None:
            r2 = np.sum((Q0 - gw[None, :]) ** 2, axis=1)
            col += np.asarray(MF.FIRE_LIGHT)[None, :] * (self.GRAIN_WARM * e / (r2 + 0.05))[:, None] * s * s
        rad = np.full(len(Q0), max(0.55, self.GRAIN_R * s))
        C = np.r_[tc.pos, tc.fwd, tc.right, tc.up, tc.f, tc.cx, tc.cy].astype(np.float64)
        _splat_streaks(out, d3, C, Q0.astype(np.float64), Q1.astype(np.float64), col.astype(np.float64), rad, 0.05)


class Wide(Shot):
    """640-799: locked, down the ridge behind her right shoulder and above the crest: the crest runs up the frame to
    her tiny figure and on to the rounded summit; the moonlit SE flank on one side of it, the NW flank in shadow."""
    name = 'climbw'
    S0 = 34.0                    # her path position (m below the lip) at 640
    OFF = 48.0                   # camera bearing off her back axis, toward her right (90 = square to the ridge):
                                 # the pot's glow clears her right side; the crest climbs the frame to the summit
    DIST = 115.0                 # camera distance from her
    UP = 12.0                    # camera height above her feet
    HFOV = 30.0
    AIM_S = 22.0                 # the path point the camera aims at (between her and the summit)
    SU, SV = 0.50, 0.58          # its place in frame (a band of sky and stars over the far ranges)

    def __init__(self, scale=0.25, ss=1.5):
        Shot.__init__(self, scale, ss, Walk(self.S0, F0))

    def camera(self, W, H):
        wk = Walk(self.S0, F0)
        her = wk.base
        pos = her + dirxz((wk.heading + 180.0 - self.OFF) % 360.0) * self.DIST
        pos[1] = her[1] + self.UP
        tgt = BS.on_ground(BS.path_at(self.AIM_S)) + np.array([0.0, 1.0, 0.0])
        d = tgt - pos
        bear = math.degrees(math.atan2(d[0], d[2]))
        el = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
        fpx = 0.5 * W / math.tan(math.radians(self.HFOV) * 0.5)
        yaw = bear - math.degrees(math.atan((self.SU - 0.5) * W / fpx))
        pitch = el + math.degrees(math.atan((self.SV - 0.5) * H / fpx))
        return RC.RCam(pos, yaw, pitch, 0.0, self.HFOV, W, H)

    core_gain = 1.0
    halo_gain = 1.0
    # spindrift off the crest, streaming toward the lens over the NW flank
    N_GRAINS = 60000
    SEED = 41
    LIFE = (1.2, 3.2)
    SPEED = (5.0, 11.0)
    LIFT = (0.3, 2.2)
    SWIRL = 0.9
    GRAIN_E = 0.9
    GRAIN_WARM = 0.0
    GRAIN_R = 0.9

    def sources(self, rng, n):
        s = rng.uniform(-2.0, 115.0, n)
        P = np.array([BS.on_ground(BS.path_at(max(q, 0.0))) for q in np.linspace(0.0, 115.0, 231)])
        k = np.clip(((s - 0.0) / 115.0 * 230).astype(int), 0, 230)
        Q = P[k] + np.c_[rng.normal(0, 0.8, n), rng.uniform(0.02, 0.35, n), rng.normal(0, 0.8, n)]
        Q -= WIND[None, :] * rng.uniform(0.0, 4.0, n)[:, None]          # a little upwind of the crest line
        return Q


class Close(Shot):
    """800-879: locked, behind her right shoulder and a little above; the pot past her right hip."""
    name = 'climbc'
    S0 = 12.0
    BACK = 12.5                  # camera distance behind her
    SIDE = 33.0                  # degrees off her back axis, toward her right (the pot shows past her right hip)
    UP = 2.0                     # camera height above her feet
    HFOV = 44.0
    SU, SV = 0.26, 0.66          # where her pelvis sits in frame: the lip, the cold beacon and the cairn fit right

    def __init__(self, scale=0.25, ss=1.5):
        Shot.__init__(self, scale, ss, Walk(self.S0, FC))

    def camera(self, W, H):
        wk = Walk(self.S0, FC)
        mid = wk.to_world([0.0, 0.0, Walk.SPEED * 1.6])          # where she is mid-shot
        back = (wk.heading + 180.0 - self.SIDE) % 360.0          # behind-right (toward her right = heading + 90)
        pos = mid + dirxz(back) * self.BACK
        pos[1] = mid[1] + self.UP
        tgt = mid + np.array([0.0, 0.9, 0.0])
        d = tgt - pos
        bear = math.degrees(math.atan2(d[0], d[2]))
        el = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
        fpx = 0.5 * W / math.tan(math.radians(self.HFOV) * 0.5)
        yaw = bear - math.degrees(math.atan((self.SU - 0.5) * W / fpx))
        pitch = el + math.degrees(math.atan((self.SV - 0.5) * H / fpx))
        return RC.RCam(pos, yaw, pitch, 0.0, self.HFOV, W, H)

    # ground drift: grains skimming the snow from the left (upwind, SE) across and toward the lens; the pot warms them
    N_GRAINS = 26000
    SEED = 43
    LIFE = (0.8, 2.2)
    SPEED = (3.0, 7.5)
    LIFT = (0.02, 0.7)
    SWIRL = 0.35
    GRAIN_E = 0.028
    GRAIN_WARM = 0.0015
    GRAIN_R = 1.1

    def sources(self, rng, n):
        wk = self.walk
        x = rng.uniform(-14.0, 6.0, n)                   # her local x: left (-) is upwind
        z = rng.uniform(-12.0, 16.0, n)
        Q = np.array([wk.to_world([xx, 0.0, zz]) for xx, zz in zip(x, z)])
        g = np.array([BS.ground(q[0], q[2]) for q in Q])
        Q[:, 1] = g + rng.uniform(0.0, 0.12, n)
        return Q


# ------------------------------------------------------------------ kernels ---
@njit(cache=True, fastmath=True)
def _seg_seg(p0, p1, q0, q1):
    """Closest distance between segments p0-p1 and q0-q1."""
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = d1 @ d1
    e = d2 @ d2
    fq = d2 @ r
    if a < 1e-12 and e < 1e-12:
        return math.sqrt(r @ r)
    if a < 1e-12:
        s = 0.0
        tq = min(max(fq / e, 0.0), 1.0)
    else:
        c = d1 @ r
        if e < 1e-12:
            tq = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = min(max((b * fq - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            tq = (b * s + fq) / e
            if tq < 0.0:
                tq = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif tq > 1.0:
                tq = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)
    dd = p0 + d1 * s - (q0 + d2 * tq)
    return math.sqrt(dd @ dd)


@njit(cache=True, fastmath=True)
def _shadow_caps(X, ok, L, caps, soft, reach):
    """Soft shadow of capsules toward a directional light L, for world points X (h, w, 3)."""
    h, w = ok.shape
    out = np.zeros((h, w), np.float32)
    for j in range(h):
        for i in range(w):
            if not ok[j, i]:
                continue
            p0 = X[j, i]
            p1 = p0 + L * reach
            best = 0.0
            for k in range(caps.shape[0]):
                a = caps[k, 0:3]
                b = caps[k, 3:6]
                r = caps[k, 6]
                dd = _seg_seg(p0, p1, a, b)
                s = 1.0 - min(max((dd - r + soft) / (2.0 * soft), 0.0), 1.0)
                if s > best:
                    best = s
            out[j, i] = best
    return out


@njit(cache=True, fastmath=True)
def _point_light(X, N, ok, Lp, caps, soft):
    """Lambert * 1/r^2 from a point light at Lp onto points X with normals N; the capsules shadow it."""
    h, w = ok.shape
    out = np.zeros((h, w), np.float32)
    for j in range(h):
        for i in range(w):
            if not ok[j, i]:
                continue
            p = X[j, i]
            d = Lp - p
            r2 = d @ d
            if r2 > 64.0:
                continue
            r = math.sqrt(r2)
            l = d / r
            ndl = N[j, i] @ l
            if ndl <= 0.0:
                continue
            vis = 1.0
            for k in range(caps.shape[0]):
                a = caps[k, 0:3]
                b = caps[k, 3:6]
                rr = caps[k, 6]
                dd = _seg_seg(p + l * 0.02, Lp - l * 0.06, a, b)
                s = min(max((dd - rr + soft) / (2.0 * soft), 0.0), 1.0)
                vis = min(vis, s)
            fall = 1.0 / (r2 + 0.04) * (1.0 - smooth01(r / 8.0))
            out[j, i] = ndl * vis * fall
    return out


@njit(inline='always', fastmath=True)
def smooth01(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


@njit(cache=True, fastmath=True)
def _splat_streaks(img, d3, C, P0, P1, col, rad, zbias):
    """Motion-blurred grains: each a Gaussian of radius rad[k] px dragged from P0[k] to P1[k] (world), energy col[k]
    (linear RGB, total), depth-tested against d3 (3-D metres). C = cam pos(3) fwd(3) right(3) up(3) f cx cy."""
    H, W = d3.shape
    px, py, pz = C[0], C[1], C[2]
    for k in range(P0.shape[0]):
        ok = True
        sx = np.zeros(2)
        sy = np.zeros(2)
        zz = 0.0
        for e in range(2):
            Q = P0[k] if e == 0 else P1[k]
            dx, dy, dz = Q[0] - px, Q[1] - py, Q[2] - pz
            z = dx * C[3] + dy * C[4] + dz * C[5]
            if z < 0.05:
                ok = False
                break
            sx[e] = C[13] + C[12] * (dx * C[6] + dy * C[7] + dz * C[8]) / z
            sy[e] = C[14] - C[12] * (dx * C[9] + dy * C[10] + dz * C[11]) / z
            zz += 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
        if not ok:
            continue
        r = rad[k]
        L = math.hypot(sx[1] - sx[0], sy[1] - sy[0])
        n = int(L / max(0.5 * r, 0.35)) + 1
        if n > 64:
            n = 64
        x0 = min(sx[0], sx[1]) - 3.0 * r
        x1 = max(sx[0], sx[1]) + 3.0 * r
        y0 = min(sy[0], sy[1]) - 3.0 * r
        y1 = max(sy[0], sy[1]) + 3.0 * r
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            continue
        inv = 1.0 / (2.0 * r * r)
        norm = 1.0 / (6.2831853 * r * r * n)
        for m in range(n):
            u = (m + 0.5) / n
            cx = sx[0] + (sx[1] - sx[0]) * u
            cy = sy[0] + (sy[1] - sy[0]) * u
            i0 = max(int(cx - 3.0 * r), 0)
            i1 = min(int(cx + 3.0 * r) + 1, W - 1)
            j0 = max(int(cy - 3.0 * r), 0)
            j1 = min(int(cy + 3.0 * r) + 1, H - 1)
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    if zz > d3[j, i] + zbias:
                        continue
                    ddx = i + 0.5 - cx
                    ddy = j + 0.5 - cy
                    g = math.exp(-(ddx * ddx + ddy * ddy) * inv) * norm
                    img[j, i, 0] += col[k, 0] * g
                    img[j, i, 1] += col[k, 1] * g
                    img[j, i, 2] += col[k, 2] * g


# ------------------------------------------------------------------ driver ---
_SHOTS = {}


def shot_for(f, scale, ss):
    k = ('w' if f < FC else 'c', scale, ss)
    if k not in _SHOTS:
        _SHOTS[k] = (Wide if f < FC else Close)(scale, ss)
    return _SHOTS[k]


def render(f, scale=1.0, ss=1.5):
    return look.finish(shot_for(f, scale, ss).render(f), **FINISH)


def layout():
    for S in (Wide, Close):
        sh = S.__new__(S)
        sh.walk = Walk(S.S0, F0 if S is Wide else FC)
        tc = sh.camera(1920, 804)
        wk = sh.walk
        for f in ((F0, 720, FC - 1) if S is Wide else (FC, 840, F1 - 1)):
            p, pel, c, glow, t, ph = wk.pose(f)
            feet = wk.to_world([0.0, wk.ground(0.0, pel[2]), pel[2]])
            head = wk.to_world(pel + np.array([0.0, 0.80, 0.10]))
            fx, fy, fz = tc.project(feet)
            hx, hy, hz = tc.project(head)
            gx, gy, gz = tc.project(wk.to_world(glow))
            print(f'{S.__name__} f{f}: feet ({fx:.0f},{fy:.0f}) head ({hx:.0f},{hy:.0f}) height {fy - hy:.0f}px dist {fz:.1f}m'
                  f' | pot glow ({gx:.0f},{gy:.0f})')
        for nm, P in (('cairn', BS.CAIRN), ('beacon', BS.BEACON), ('lip', BS.LIP), ('top', BS.TOP)):
            x, y, z = tc.project(P)
            print(f'   {nm}: ({x:.0f},{y:.0f}) z {z:.1f}m')
        print(f'   cam pos {np.round(tc.pos, 1)} yaw {tc.yaw_d:.1f} pitch {tc.pitch_d:.1f} hfov {tc.hfov_d}; heading '
              f'{wk.heading:.1f}; moon az {MOON_AZ:.0f} el {MOON_EL:.0f}')


def _work(args):
    frames, scale, ss, out = args
    for f in frames:
        t0 = time.time()
        img = render(f, scale, ss)
        look.save_png(look.frame_path(out, f), img)
        print(f'frame {f} {time.time() - t0:.2f}s', flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--range', default=None)
    ap.add_argument('--frames', default=None)
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default='climb_t')
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--layout', action='store_true')
    a = ap.parse_args()
    if a.layout:
        layout()
        return
    out = os.path.join(ROOT, a.out) if a.out.startswith('renders/') else os.path.join(TESTS, a.out)
    os.makedirs(out, exist_ok=True)
    if a.range:
        s0, s1 = a.range.split('-')
        frames = list(range(int(s0), int(s1) + 1))
    else:
        frames = [int(x) for x in a.frames.split(',')]
    if a.skip:
        frames = [f for f in frames if not os.path.exists(look.find_frame(out, f))]
    if a.procs <= 1:
        _work((frames, a.scale, a.ss, out))
    else:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.procs) as pool:
            pool.map(_work, [(frames[i::a.procs], a.scale, a.ss, out) for i in range(a.procs)])


if __name__ == '__main__':
    main()
