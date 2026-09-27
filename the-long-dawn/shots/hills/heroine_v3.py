"""HEROINE v3 close-ups (BIBLE_V3 H2, H5 + REVISION 1): hands, pose and light. No faces.

Each shot is a class with render(f, scale) -> display sRGB (look.finish). Frames are seconds x 24 in the shot's own
cut, so they drop straight onto that cut's timeline (renders/heroine_B/, renders/heroine_C/):

  DeadEmber  B 40.0-50.0 (960-1199)    H5. Over her left shoulder from behind and above: she kneels at the summit
             cairn, lifts the clay fire-pot's lid, one red eye of ember on the ash; she blows; it greys under her
             breath; the breath hangs. Her face is turned down, away from the lens: the head is a silhouette.
  Find       C 148.2-153.5 (3557-3683) H2. Low on the snow by her knee: the second strike's flash finds a gold band in a
             melted hollow; in the dark again its letters are faintly awake; her gloved hand closes on it.
  FireTest   C 168.3-174.3 (4040-4183) H2, Bag End (REVISION 1). In her roaring beacon the Ring lies on the tip of her
             steel, unmarked, letters awake, not even warm; her gloved hand holds it there and cannot let it fall.

The Ring is ring.py's band and inscription (BIBLE_V3 H2; REVISION 1 gives the Ring close-ups to a Blender bake-off,
this band being its fallback). The hands are heroine.py's anatomical hands in leather gloves (hsdf3.gloves), the
coat, sleeves and scarf are the accepted v2b figure's. Rendered by hsdf3 (the v3 fork of the heroine tracer).

    python3 shots/hills/heroine_v3.py still deadember 1090 out.png [--scale 0.5]
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fire  # noqa: E402
import heroine as hero  # noqa: E402
import hsdf3 as H3  # noqa: E402
from core import Camera, look, smoothstep, fnoise1, splat_gauss  # noqa: E402

FPS = 24.0
MOON_DIR = np.array([0.55, 0.45, 0.70]) / np.linalg.norm([0.55, 0.45, 0.70])
MOON_COL = np.array([0.55, 0.70, 0.95])


def nrm(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


# ------------------------------------------------------------ environment ---

def env_dirs(w=512, h=256):
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    phi = (u - 0.5) * 2 * np.pi
    th = v * np.pi
    x = np.sin(th)[:, None] * np.sin(phi)[None, :]
    y = np.repeat(np.cos(th)[:, None], w, 1)
    z = np.sin(th)[:, None] * np.cos(phi)[None, :]
    return np.stack([x, y, z], -1)


def env_stack(img, levels=6, base=1.4):
    """Progressively blurred copies (same size, x wraps) for rough reflections: level k ~ sigma base * 2^k px."""
    out = np.zeros((levels,) + img.shape, np.float32)
    out[0] = img
    for k in range(1, levels):
        s = base * 2 ** (k - 1)
        pad = int(3 * s) + 2
        ext = np.concatenate([img[:, -pad:], img, img[:, :pad]], 1)
        out[k] = cv2.GaussianBlur(ext, (0, 0), s)[:, pad:-pad]
    return out


def night_env(snow=1.0, moon=1.0, fig_dirs=(), fig_dark=0.15):
    """The moonlit night seen from a point on her summit: a dark blue sky, the moon behind her, the snow below lit by
    it; `fig_dirs` = (direction, angular radius rad) blobs of her dark figure close by."""
    D = env_dirs()
    y = D[..., 1]
    sky = np.array([0.0020, 0.0035, 0.0100])[None, None] + np.array([0.004, 0.0065, 0.014])[None, None] * \
        np.clip(1 - y, 0, 1)[..., None] ** 3
    ground = MOON_COL[None, None] * 0.055 * snow * np.ones_like(D)
    img = np.where(y[..., None] > 0.0, sky, ground)
    cm = D @ MOON_DIR
    img += (MOON_COL * 40.0 * moon)[None, None] * (cm > math.cos(math.radians(1.2)))[..., None]
    img += (MOON_COL * 0.02 * moon)[None, None] * np.exp(-(1 - cm) / 0.004)[..., None]
    for d, rad in fig_dirs:
        c = D @ nrm(d)
        m = smoothstep(math.cos(rad * 1.15), math.cos(rad * 0.85), c)
        img = img * (1 - m[..., None] * (1 - fig_dark))
    return img.astype(np.float32)


# ---------------------------------------------------------------- the Ring ---

def ring_py():
    """accord/ring.py (the Ring's fallback design), imported by path."""
    import importlib.util
    if 'ring_py' in _MOD:
        return _MOD['ring_py']
    spec = importlib.util.spec_from_file_location('ring_accord', os.path.join(HERE, '..', 'accord', 'ring.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _MOD['ring_py'] = m
    return m


_MOD = {}
RING_INNER = 0.0088            # inner radius (m): a slim finger
_RP = None


def ring_dims():
    """ring.py's section (R_OUT 0.186, R_IN 0.138, BAND 0.084, ROUND 0.013) scaled to a finger."""
    RG = ring_py()
    s = RING_INNER / RG.R_IN
    R = 0.5 * (RG.R_OUT + RG.R_IN) * s
    tb = 0.5 * (RG.R_OUT - RG.R_IN) * s
    hb = 0.5 * RG.BAND * s
    return R, tb, hb, RG.ROUND * s


def add_ring(B, centre, rows):
    R, tb, hb, rnd = ring_dims()
    B.group('ring', H3.M_GOLD, band=0.0005)
    H3.band(B, centre, rows, R, tb, hb, rnd)
    return B


def ring_xp(XP, centre, rows, glow, engrave=1.0):
    R, tb, hb, rnd = ring_dims()
    return H3.ring_xp_accord(XP, centre, rows, R, tb, hb, glow, engrave)


# ------------------------------------------------------------------ lights ---

def light(p, col, inten, radius, k):
    p = np.asarray(p, np.float64)
    c = np.asarray(col, np.float64) * inten
    return [p[0], p[1], p[2], c[0], c[1], c[2], radius, k]


def moon_light(inten):
    p = MOON_DIR * 100.0
    c = MOON_COL * inten * 1e4
    return [p[0], p[1], p[2], c[0], c[1], c[2], 0.0, 0.0]


# ---------------------------------------------------------------- post ---

def dof(rgb, depth, focus, K, near_split=None, radii=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)):
    """Thin-lens depth of field by a blur stack: blur sigma (px) = K |1/d - 1/focus|. Pixels nearer than
    `near_split` (m) are pulled out as a layer and blurred with their alpha, so an out-of-focus foreground spreads
    over what is behind it."""
    def stack_blur(img, sig):
        out = np.zeros_like(img)
        sig = np.clip(sig, 0, radii[-1])
        prev = None
        for k in range(len(radii) - 1):
            lo, hi = radii[k], radii[k + 1]
            m = (sig >= lo) & ((sig < hi) | (k == len(radii) - 2))
            if not np.any(m):
                continue
            a = img if lo == 0 else cv2.GaussianBlur(img, (0, 0), lo)
            b = cv2.GaussianBlur(img, (0, 0), hi)
            w = ((sig - lo) / (hi - lo)).clip(0, 1)[..., None]
            out[m] = (a * (1 - w) + b * w)[m]
        return out
    inv = 1.0 / np.maximum(depth, 1e-3)
    sig = K * np.abs(inv - 1.0 / focus)
    if near_split is None:
        return stack_blur(rgb, sig)
    near = (depth < near_split).astype(np.float32)
    far_rgb = rgb * (1 - near[..., None])
    # fill the near layer's holes in the far layer with a blurred far image, so the spread has something behind it
    fill = cv2.GaussianBlur(far_rgb, (0, 0), 25) / np.maximum(cv2.GaussianBlur(1 - near, (0, 0), 25), 1e-3)[..., None]
    far_rgb = far_rgb + fill * near[..., None]
    far = stack_blur(far_rgb, np.where(near > 0, K * abs(1 / near_split - 1 / focus), sig))
    nsig = float(np.percentile(sig[near > 0], 60)) if np.any(near > 0) else 0.0
    if nsig > 0.3:
        nr = cv2.GaussianBlur(rgb * near[..., None], (0, 0), nsig)
        na = cv2.GaussianBlur(near, (0, 0), nsig)
    else:
        nr, na = rgb * near[..., None], near
    return far * (1 - na[..., None]) + nr


def finish(img, exposure=1.0, bloom=0.10, vign=0.28):
    return look.finish(img, exposure=exposure, bloom_strength=bloom, bloom_threshold=0.9, vignette_amount=vign)


def comp(bg, res):
    """Composite an hsdf3.render result over bg (in place); returns a full-frame depth buffer."""
    H_, W_ = bg.shape[:2]
    depth = np.full((H_, W_), 1e3, np.float32)
    if res is None:
        return depth
    y0, x0, rgb, a, d, _ = res
    h, w = a.shape
    sub = bg[y0:y0 + h, x0:x0 + w]
    sub *= (1 - a[..., None])
    sub += rgb
    depth[y0:y0 + h, x0:x0 + w] = np.where(a > 0.01, d, 1e3)
    return depth


# ------------------------------------------------------------ the figure ---

def static_chain(anchor, direction, n, seg, droop=0.3, sway=(0.0, 0.0), seed=0.0):
    """A still chain (scarf tail, hair lock) hanging from `anchor` (2-D x, y) along `direction`, bending under
    gravity; `sway` adds a lateral wave."""
    d = nrm(direction)
    P = [np.asarray(anchor, np.float64)]
    for i in range(1, n):
        s = i / (n - 1)
        dd = nrm(d * (1 - droop * s) + np.array([0.0, -1.0]) * droop * s
                 + np.array([sway[0], sway[1]]) * math.sin(s * 5.0 + seed))
        P.append(P[-1] + dd * seg)
    return np.array(P)


def figure(pose, t, scarf_dir=(0.7, -0.7), hair_dir=(0.5, -0.8), gloved=True, scarf_n=14, detail=1.0):
    an = hero.cheap_anchors(pose)
    sa = an['scarf_anchor'][:2]
    scarf = static_chain(sa, scarf_dir, scarf_n, 0.078, droop=0.55, sway=(0.12, 0.05), seed=1.0)
    hr = an['hair_root'][:2]
    hair = [static_chain(hr + np.array([-0.010 + 0.004 * k, 0.030 - 0.010 * k]), hair_dir, 8,
                         0.045 + 0.006 * (k % 3), droop=0.6, sway=(0.05, 0.03), seed=k * 1.3) for k in range(9)]
    B, F, Hp, anc = hero.build_figure(pose, t, scarf_pts=scarf, hair_pts=hair, detail=detail)
    if gloved:
        H3.gloves(B, Hp)
    return B, F, Hp, anc, hair


# ---------------------------------------------------------- shared props ---

def snow_ground(B, c, reach=1.2, hollow=None):
    """Summit snow: a gently rolling plane (y up) with felted relief; optional melted hollow
    (centre, radius, depth): a pit with a slumped lip, glazed with ice."""
    B.group('snow', H3.M_SNOW, disp=1, amp=0.0022, scale=26.0, band=0.02)
    R = np.stack([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    B.plane(np.asarray(c, np.float64), R, bend_y=0.04, bend_z=0.03, op=0, reach=reach)
    if hollow is not None:
        hc, hr, hd = hollow
        hc = np.asarray(hc, np.float64)
        B.torus(hc + np.array([0.0, 0.004, 0.0]), np.eye(3), hr * 1.02, hr * 0.28, 0.010, k=0.02)
        B.ell(hc + np.array([0.0, 0.2 * hd, 0.0]), np.array([hr * 1.02, hd * 1.2, hr * 0.98]), k=0.012, op=1)
        B.group('ice', H3.M_ICE, band=0.002)
        B.ell(hc + np.array([0.0, 0.2 * hd, 0.0]), np.array([hr * 1.02, hd * 1.2, hr * 0.98]) + 0.0025)
        B.ell(hc + np.array([0.0, 0.2 * hd, 0.0]), np.array([hr * 1.02, hd * 1.2, hr * 0.98]) - 0.0012, op=1)
        Rh = np.stack([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        B.plane(hc + np.array([0.0, -0.25 * hd, 0.0]), Rh, op=2, reach=hr * 1.6)
    return B


# ================================================================ H5 ====

class DeadEmber:
    """B 40.0-50.0 s (frames 960-1199). Keys (frames): kneel settled 960; lid lifts 984-1010; the red eye 1010-1040;
    blow 1 1040-1062 (the ember brightens a little: hope), blow 2 1072-1092 (nothing); it greys 1060-1130; the
    breath hangs 1092-1150; still to 1199."""
    F0, F1 = 960, 1199
    POT = np.array([0.34, 0.0, -0.03])

    def __init__(self):
        pass

    # ---- keys
    def life(self, f):
        e = 1.0 - 0.55 * smoothstep(1044, 1066, f) - 0.45 * smoothstep(1066, 1130, f)
        e += 0.18 * smoothstep(1041, 1047, f) * (1 - smoothstep(1049, 1062, f))       # a breath of hope
        return float(np.clip(e, 0.0, 1.2))

    def lid_u(self, f):
        return smoothstep(984, 1010, f)

    def pose(self, f):
        t = f / FPS
        u = self.lid_u(f)
        blow = max(smoothstep(1040, 1044, f) * (1 - smoothstep(1058, 1062, f)),
                   smoothstep(1072, 1076, f) * (1 - smoothstep(1088, 1092, f)))
        p = dict(
            pelvis=(0.74, 0.46, 0.0), yaw=0.0, lean=40.0 + 3.0 * blow, chest=10.0, twist=0.0, neck=16.0,
            head=34.0 + 6.0 * blow, head_yaw=0.0, head_roll=0.0, shrug=0.2,
            hand_f=(0.37, 0.115, 0.075), elbow_f=(0.2, -1.0, 0.8),
            fdir_f=(-0.25, -0.55, -0.80), palm_f=(0.15, 0.10, -1.0),
            curl_f=(0.30, 0.34, 0.38, 0.45), thumb_f=0.25, spread_f=0.15,
            elbow_n=(0.1, -1.0, -0.8),
            curl_n=(0.55, 0.62, 0.70, 0.74), thumb_n=0.55, spread_n=0.1,
            foot_n=(1.00, 0.07, -0.11), foot_f=(1.00, 0.07, 0.11), knee_n=(-1.0, -0.35, -0.05), knee_f=(-1.0, -0.35, 0.05),
            toe_n=(1.0, -0.2, 0.0), toe_f=(1.0, -0.2, 0.0), sole_n=(0.0, -1.0, 0.0), sole_f=(0.0, -1.0, 0.0),
            hem=0.05, breath=math.sin(2 * math.pi * t / 3.6) * (1 - blow),
        )
        # the near hand: on the lid's knob, lifting it off and holding it aside, tilted toward her
        rest = self.POT + np.array([0.0, 0.140, 0.0])
        held = self.POT + np.array([0.075, 0.215, -0.135])
        lc = rest * (1 - u) + held * u
        p['hand_n'] = tuple(lc + np.array([0.030, 0.050, -0.010]))
        p['fdir_n'] = tuple(nrm([-0.70, -0.55, 0.30]))
        p['palm_n'] = tuple(nrm([0.20, -0.85, 0.35]))
        p['tools'] = 'none'
        p['rock'] = None
        p['expr'] = dict(purse=blow, blink=0.6, brow=0.2)
        p['look_at'] = self.POT + np.array([0.0, 0.09, 0.0])
        return p, lc, u, blow

    # ---- props
    def pot(self, B, lid_c, lid_u):
        c = self.POT
        B.group('pot', H3.M_CLAY, disp=1, amp=0.0006, scale=90.0, band=0.004)
        B.ell(c + [0, 0.066, 0], np.array([0.070, 0.064, 0.070]))
        B.ell(c + [0, 0.012, 0], np.array([0.050, 0.014, 0.050]), k=0.02)                      # foot
        B.torus(c + [0, 0.122, 0], np.eye(3), 0.046, 0.0085, 0.0105, k=0.016)                  # rim
        B.cone(c + [0, 0.098, 0], c + [0, 0.122, 0], 0.052, 0.049, k=0.012)                   # shoulder to neck
        B.ell(c + [0, 0.070, 0], np.array([0.060, 0.056, 0.060]), op=1, k=0.004)               # hollow
        B.cone(c + [0, 0.090, 0], c + [0, 0.170, 0], 0.0395, 0.0395, op=1, k=0.004)            # mouth
        # soot: the rim blackened by years of carried fire (a thin dark skin just inside and over the lip)
        B.group('soot', H3.M_COAL, band=0.002)
        B.torus(c + [0, 0.1245, 0], np.eye(3), 0.0445, 0.0062, 0.0072, k=0.0)
        B.group('ash', H3.M_ASH, disp=1, amp=0.0012, scale=160.0, band=0.004)
        B.ell(c + [0, 0.062, 0], np.array([0.0575, 0.024, 0.0575]))
        B.group('ember', H3.M_EMBER, disp=1, amp=0.0016, scale=150.0, band=0.004)
        e = c + np.array([-0.006, 0.086, 0.004])
        B.ell(e, np.array([0.0135, 0.0085, 0.011]), R=np.stack([nrm([1, 0.1, 0.3]), nrm([-0.1, 1, 0]), nrm([-0.3, 0, 1])]))
        B.ell(e + [0.009, -0.002, 0.004], np.array([0.0075, 0.0060, 0.0070]), k=0.004)
        B.ell(e + [-0.008, -0.003, -0.005], np.array([0.0060, 0.0045, 0.0055]), k=0.004)
        # the lid: a shallow clay dome with a knob, lifted in her near hand and tipped toward her
        ax = nrm(np.array([0.0, 1.0, 0.0]) * (1 - lid_u) + nrm([0.55, 0.55, -0.62]) * lid_u)
        Rl = H3.ring_frame(ax, ref=(1.0, 0.0, 0.0))
        B.group('lid', H3.M_CLAY, disp=1, amp=0.0006, scale=90.0, band=0.004)
        B.ell(lid_c, np.array([0.054, 0.0105, 0.054]), R=Rl)
        B.ell(lid_c + ax * 0.012, np.array([0.011, 0.009, 0.011]), R=Rl, k=0.008)
        B.ell(lid_c - ax * 0.011, np.array([0.044, 0.008, 0.044]), R=Rl, op=1, k=0.004)
        return e

    def camera(self, f, scale):
        t = f / FPS
        pos = np.array([1.02 + 0.004 * fnoise1(t * 0.5, 3.0), 1.14 + 0.003 * fnoise1(t * 0.4, 5.0), -0.80])
        tgt = self.POT + np.array([0.035, 0.12, 0.0])
        cam = Camera(pos, hfov=30.0, scale=scale)
        yaw, pitch = cam.look_at(tgt)
        return Camera(pos, yaw=yaw, pitch=pitch, hfov=30.0, scale=scale), float(np.linalg.norm(tgt - pos))

    def render(self, f, scale=0.5):
        t = f / FPS
        cam, focus = self.camera(f, scale)
        p, lid_c, lid_u, blow = self.pose(f)
        B, F, Hp, anc, hair = figure(p, t, scarf_dir=(0.35, -0.9), hair_dir=(0.2, -1.0))
        epos = self.pot(B, lid_c, lid_u)
        snow_ground(B, self.POT + [0.1, 0.0, 0.0], reach=1.6)
        life = self.life(f)
        XP = np.zeros(64)
        XP[19] = life
        XP[20] = 1.6 * lid_u ** 0.5 + 0.6
        XP[24] = 260.0
        XP[25] = 1.0
        XP[26] = 1.0
        ENV = env_stack(night_env(moon=0.8))
        # light: B's silver moon from behind her; the ember inside the pot (its walls and her hands shadow it)
        glow = 0.0045 * (0.25 + life ** 1.6) * (0.35 + 0.65 * lid_u)
        L = [moon_light(0.30),
             light(epos + [0, 0.012, 0], (1.0, 0.30, 0.06), glow, 0.010, 6.0)]
        env = hero.env_vec(rim_dir=MOON_DIR, rim=np.array([0.10, 0.13, 0.20]), amb=np.array([0.004, 0.006, 0.011]),
                           bounce=np.array([0.010, 0.013, 0.020]), ao=0.02)
        res = H3.render(cam, B, Hp, L, env, XP, ENV, None, ss=(3 if scale > 0.75 else 2),
                        sil=dict(skin=1.0, eyes=1.0, cap=0.6, cap_brim=0.6, hair=0.6))
        img = np.zeros((cam.H, cam.W, 3), np.float32)
        img[:] = np.array([0.004, 0.006, 0.012], np.float32)
        depth = comp(img, res)
        # breath: from her hidden mouth down into the pot's glow, then hanging in the moonlight
        self._breath(img, cam, f, anc, epos, life, lid_u)
        img = dof(img, depth, focus, K=cam.f * 0.010, near_split=focus * 0.8)
        return finish(img, exposure=1.35)

    def _breath(self, img, cam, f, anc, epos, life, lid_u):
        import heroine_sdf as hsd
        mouth = anc['mouth']
        puffs = ((1040, 0.9, 'blow'), (1072, 0.8, 'blow'), (1006, 1.6, 'out'), (1100, 2.4, 'hang'))
        for (fs, dur, kind) in puffs:
            age = (f - fs) / FPS
            if age < 0 or age > dur + (1.8 if kind == 'hang' else 0.6):
                continue
            n_sub = 30
            for q in range(n_sub):
                rel = q / n_sub * 0.5
                a_ = age - rel
                if a_ <= 0:
                    continue
                if kind == 'blow':
                    d = (epos + np.array([0.0, 0.03, 0.0])) - mouth
                    k = min(1.0, a_ / 0.35)
                    pos = mouth + d * (1 - (1 - k) ** 2) + np.array([0.03, 0.02, -0.01]) * max(0.0, a_ - 0.35)
                    rad_m = 0.010 + 0.040 * min(a_, 0.9)
                    dens = 0.10 * math.exp(-a_ / 0.9) * min(1.0, a_ / 0.06)
                else:
                    pos = mouth + np.array([-0.03, -0.10, -0.02]) * (1 - math.exp(-a_ / 0.5)) + \
                        np.array([0.05, 0.03, -0.03]) * a_
                    rad_m = 0.02 + 0.05 * a_
                    dens = 0.05 * math.exp(-a_ / 1.4) * min(1.0, a_ / 0.2)
                sx, sy, z = cam.project(pos)
                if z <= 0.05:
                    continue
                d2 = float(np.sum((pos - epos) ** 2)) + 0.002
                Ls = np.array([1.0, 0.33, 0.07]) * 0.0045 * (0.25 + life ** 1.6) * lid_u / d2 * 0.6 + \
                    MOON_COL * 0.010
                sig = max(0.8, cam.f * rad_m / z * 0.6)
                hsd.splat_fog(img, float(sx), float(sy), sig, dens, Ls[0], Ls[1], Ls[2],
                              float(fs * 0.37 + q), f / FPS * 0.6, max(2.0, sig * 1.1))


SHOTS = dict(deadember=DeadEmber)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('shot')
    ap.add_argument('frame', type=int)
    ap.add_argument('out')
    ap.add_argument('--scale', type=float, default=0.5)
    a = ap.parse_args()
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    sh = SHOTS[a.shot]()
    img = sh.render(a.frame, a.scale)
    look.save_png(a.out, img)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
