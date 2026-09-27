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
        m = np.clip((c - math.cos(rad * 1.15)) / (math.cos(rad * 0.85) - math.cos(rad * 1.15)), 0, 1)
        m = m * m * (3 - 2 * m)
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


def moon_light(inten, d=None):
    p = (MOON_DIR if d is None else d) * 100.0
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


def debug_cam(cam, focus, scale):
    """V3_CAM='px,py,pz,tx,ty,tz,hfov' overrides the camera (and switches DOF off) for layout checks."""
    v = os.environ.get('V3_CAM')
    if not v:
        return cam, focus, False
    q = [float(x) for x in v.split(',')]
    c = Camera(q[:3], hfov=q[6], scale=scale)
    yaw, pitch = c.look_at(q[3:6])
    return Camera(q[:3], yaw=yaw, pitch=pitch, hfov=q[6], scale=scale), focus, True


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


def figure(pose, t, scarf_dir=(0.7, -0.7), hair_dir=(0.5, -0.8), gloved=True, scarf_n=14, detail=1.0, scarf=True):
    an = hero.cheap_anchors(pose)
    sa = an['scarf_anchor'][:2]
    if scarf:
        scarf = static_chain(sa, scarf_dir, scarf_n, 0.078, droop=0.55, sway=(0.12, 0.05), seed=1.0)
    else:
        scarf = None
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
        B.ell(hc + np.array([0.0, 0.2 * hd, 0.0]), np.array([hr * 1.02, hd * 1.2, hr * 0.98]) + 0.0015)
        Rh = np.stack([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        B.plane(hc + np.array([0.0, -0.25 * hd, 0.0]), Rh, op=2, reach=hr * 1.6)
    return B


# ================================================================ H5 ====

class DeadEmber:
    """B 40.0-50.0 s (frames 960-1199). Keys (frames): kneel settled 960; lid lifts 984-1010; the red eye 1010-1040;
    blow 1 1040-1062 (the ember brightens a little: hope), blow 2 1072-1092 (nothing); it greys 1060-1130; the
    breath hangs 1092-1150; still to 1199."""
    F0, F1 = 960, 1199
    POT = np.array([0.36, 0.28, -0.02])       # on the cairn's lowest course (top at 0.28 m)
    HFOV = 38.0
    POV = False
    MOON = np.array([0.20, 0.62, -0.76]) / np.linalg.norm([0.20, 0.62, -0.76])   # B's moon: high on her left

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
            pelvis=(0.80, 0.57, 0.0), yaw=0.0, lean=20.0 + 4.0 * blow, chest=6.0, twist=0.0, neck=12.0,
            head=40.0 + 6.0 * blow, head_yaw=0.0, head_roll=0.0, shrug=0.25,
            elbow_f=(0.2, -1.0, 0.9), elbow_n=(0.2, -1.0, -0.9),
            curl_f=(0.30, 0.34, 0.40, 0.46), thumb_f=0.25, spread_f=0.15,
            curl_n=(0.55, 0.62, 0.70, 0.74), thumb_n=0.55, spread_n=0.1,
            foot_n=(1.12, 0.07, -0.12), foot_f=(1.12, 0.07, 0.12), knee_n=(-1.0, -0.6, -0.05), knee_f=(-1.0, -0.6, 0.05),
            toe_n=(1.0, -0.2, 0.0), toe_f=(1.0, -0.2, 0.0), sole_n=(0.0, -1.0, 0.0), sole_f=(0.0, -1.0, 0.0),
            hem=0.05, breath=math.sin(2 * math.pi * t / 3.6) * (1 - blow),
        )
        # the lid: her near hand lifts it off by the knob and sets it on the stone at her left; then both hands cradle
        # the pot (the near one returns 1010-1024)
        rest = self.POT + np.array([0.0, 0.150, 0.0])
        down = self.POT + np.array([0.020, -0.004, -0.175])
        lift = self.POT + np.array([0.010, 0.230, -0.080])
        lc = (rest * (1 - smoothstep(0, 0.45, u)) + lift * smoothstep(0, 0.45, u)) * (1 - smoothstep(0.45, 1.0, u)) \
            + down * smoothstep(0.45, 1.0, u)
        back = smoothstep(1010, 1024, f)
        grip_lid = lc + np.array([0.040, 0.045, -0.020])
        cradle = np.array([0.66, 0.43, -0.13])        # her near hand comes to rest on her thigh
        hn = grip_lid * (1 - back) + cradle * back
        p['hand_n'] = tuple(hn)
        fl, pl = nrm([-0.75, -0.45, 0.35]), nrm([0.10, -0.80, 0.55])
        fc, pc = nrm([-0.80, -0.55, 0.10]), nrm([0.10, -1.0, 0.10])
        p['fdir_n'] = tuple(nrm(fl * (1 - back) + fc * back))
        p['palm_n'] = tuple(nrm(pl * (1 - back) + pc * back))
        p['curl_n'] = tuple(np.array([0.55, 0.62, 0.70, 0.74]) * (1 - back) + np.array([0.30, 0.34, 0.40, 0.46]) * back)
        # the far hand cradles the far side of the pot throughout
        p['hand_f'] = tuple(self.POT + np.array([0.050, 0.080, 0.085]))
        p['fdir_f'] = (-0.55, -0.30, -0.78)
        p['palm_f'] = tuple(nrm([-0.35, 0.05, -0.93]))
        lid_u = smoothstep(0.45, 1.0, u)
        p['tools'] = 'none'
        p['rock'] = None
        p['expr'] = dict(purse=blow, blink=0.6, brow=0.2)
        p['look_at'] = self.POT + np.array([0.0, 0.09, 0.0])
        return p, lc, lid_u, blow

    # ---- props
    def pot(self, B, lid_c, lid_u):
        """A round-bellied clay fire-pot (~15 cm), its lip sooted; ash inside with one ember; the lid in her hand."""
        c = self.POT
        B.group('pot', H3.M_CLAY, disp=1, amp=0.0006, scale=90.0, band=0.004)
        B.ell(c + [0, 0.070, 0], np.array([0.079, 0.068, 0.079]))
        B.ell(c + [0, 0.012, 0], np.array([0.052, 0.014, 0.052]), k=0.02)                      # foot
        B.cone(c + [0, 0.112, 0], c + [0, 0.136, 0], 0.050, 0.044, k=0.014)                   # neck
        B.torus(c + [0, 0.137, 0], np.eye(3), 0.041, 0.0078, 0.0088, k=0.010)                  # rim
        B.ell(c + [0, 0.070, 0], np.array([0.069, 0.059, 0.069]), op=1, k=0.004)               # hollow
        B.cone(c + [0, 0.100, 0], c + [0, 0.190, 0], 0.0335, 0.0335, op=1, k=0.004)            # mouth
        # soot: the lip blackened by years of carried fire
        B.group('soot', H3.M_COAL, band=0.002)
        B.torus(c + [0, 0.1395, 0], np.eye(3), 0.0395, 0.0056, 0.0066, k=0.0)
        B.group('ash', H3.M_ASH, disp=1, amp=0.0012, scale=160.0, band=0.004)
        B.ell(c + [0, 0.080, 0], np.array([0.066, 0.026, 0.066]))
        B.group('ember', H3.M_EMBER, disp=1, amp=0.0016, scale=150.0, band=0.004)
        e = c + np.array([0.006, 0.107, 0.018])
        B.ell(e, np.array([0.0135, 0.0085, 0.011]), R=np.stack([nrm([1, 0.1, 0.3]), nrm([-0.1, 1, 0]), nrm([-0.3, 0, 1])]))
        B.ell(e + [0.009, -0.002, 0.004], np.array([0.0075, 0.0060, 0.0070]), k=0.004)
        B.ell(e + [-0.008, -0.003, -0.005], np.array([0.0060, 0.0045, 0.0055]), k=0.004)
        # the lid: a shallow clay dome with a knob, lifted in her near hand and tipped toward her
        ax = nrm(np.array([0.0, 1.0, 0.0]) * (1 - lid_u) + nrm([0.25, 1.0, -0.15]) * lid_u)
        Rl = H3.ring_frame(ax, ref=(1.0, 0.0, 0.0))
        B.group('lid', H3.M_CLAY, disp=1, amp=0.0006, scale=90.0, band=0.004)
        B.ell(lid_c, np.array([0.049, 0.0100, 0.049]), R=Rl)
        B.ell(lid_c + ax * 0.012, np.array([0.011, 0.009, 0.011]), R=Rl, k=0.008)
        B.ell(lid_c - ax * 0.010, np.array([0.040, 0.008, 0.040]), R=Rl, op=1, k=0.004)
        return e

    def cairn(self, B):
        """The foot of the summit cairn: a flat course stone under the pot and the next course stepping up beyond it."""
        B.group('stone', H3.M_FLINT, disp=1, amp=0.004, scale=16.0, band=0.02)
        rng = np.random.default_rng(5)
        for (c, h, rot) in (((0.30, 0.235, -0.02), (0.20, 0.045, 0.26), 4.0), ((0.00, 0.19, -0.30), (0.16, 0.10, 0.14), -9.0),
                            ((-0.02, 0.36, -0.04), (0.14, 0.075, 0.20), 6.0), ((0.02, 0.35, 0.30), (0.15, 0.08, 0.13), -5.0),
                            ((0.33, 0.12, 0.05), (0.22, 0.10, 0.28), 2.0), ((-0.05, 0.50, 0.10), (0.12, 0.07, 0.18), 3.0)):
            a = math.radians(rot)
            R = np.stack([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
            R = R @ np.stack([nrm([1.0, rng.normal(0, 0.04), 0.0]), nrm([rng.normal(0, 0.04), 1.0, 0.0]), [0.0, 0.0, 1.0]])
            B.box(np.array(c), np.array(h), R=R, rnd=0.022, k=0.004)

    def camera(self, f, scale):
        """Observational (B watches her): high on her left, ~50 deg down into the pot, her head and shoulders above the
        frame. POV = True: her own point of view from between her eyes (the head is then not drawn)."""
        t = f / FPS
        hf = self.HFOV
        if self.POV:
            p, lc, lid_u, blow = self.pose(f)
            an = hero.cheap_anchors(p)
            pos = an['head'].p(0.070, 0.000, 0.0)
            tgt = self.POT + np.array([-0.035, 0.120, 0.005])
        else:
            pos = np.array([0.35, 0.98, -0.44])
            tgt = self.POT + np.array([0.020, 0.110, 0.0])
        pos = pos + np.array([0.004 * fnoise1(t * 0.6, 3.0), 0.003 * fnoise1(t * 0.5, 5.0), 0.0])
        cam = Camera(pos, hfov=hf, scale=scale)
        yaw, pitch = cam.look_at(tgt)
        return Camera(pos, yaw=yaw, pitch=pitch, hfov=hf, scale=scale), float(np.linalg.norm(self.POT + [0, 0.11, 0] - pos))

    def render(self, f, scale=0.5):
        t = f / FPS
        cam, focus = self.camera(f, scale)
        cam, focus, nodof = debug_cam(cam, focus, scale)
        p, lid_c, lid_u, blow = self.pose(f)
        B, F, Hp, anc, hair = figure(p, t, hair_dir=(0.30, -0.95), scarf=False)
        # the scarf's end hangs from the front of her neck as she bends over the pot, on its far side
        J = anc['J']
        front = J['C7'] + 0.030 * J['dn'] + 0.080 * J['Ut']
        tail = static_chain(front[:2], (-0.05, -1.0), 7, 0.074, droop=0.1, sway=(0.04, 0.0), seed=2.0)
        B.group('scarf')
        hero.scarf_tail(B, tail, t, z0=0.060)
        for g in B.groups:
            if self.POV and g['name'] in ('skin', 'eyes', 'cap', 'cap_brim', 'hair'):
                g['prims'] = []
        epos = self.pot(B, lid_c, lid_u)
        snow_ground(B, np.array([0.60, 0.0, 0.0]), reach=1.6)
        self.cairn(B)
        life = self.life(f)
        XP = np.zeros(64)
        XP[19] = life
        XP[20] = 0.16
        XP[24] = 260.0
        # the eye: the ember's top, turned a little toward the lens; it narrows as the ember dies
        XP[21:24] = epos + np.array([-0.004, 0.0075, -0.004])
        XP[31] = 0.0030 + 0.0026 * min(1.0, life)
        XP[25] = 1.0
        XP[26] = 1.0
        ENV = env_stack(night_env(moon=0.8))
        # light: B's silver moon from behind her; the ember inside the pot (its walls and her hands shadow it)
        glow = 0.0026 * (0.02 + life ** 1.6) * (0.35 + 0.65 * lid_u)
        L = [moon_light(0.45, self.MOON),
             light(epos + [0, 0.012, 0], (1.0, 0.30, 0.06), glow, 0.010, 6.0)]
        env = hero.env_vec(rim_dir=self.MOON, rim=np.array([0.10, 0.13, 0.20]), amb=np.array([0.004, 0.006, 0.011]),
                           bounce=np.array([0.010, 0.013, 0.020]), ao=0.02)
        res = H3.render(cam, B, Hp, L, env, XP, ENV, None, ss=(3 if scale > 0.75 else 2),
                        sil=dict(skin=1.0, eyes=1.0, cap=0.6, cap_brim=0.6, hair=0.6))
        img = np.zeros((cam.H, cam.W, 3), np.float32)
        img[:] = np.array([0.004, 0.006, 0.012], np.float32)
        depth = comp(img, res)
        # breath: from her hidden mouth down into the pot's glow, then hanging in the moonlight
        self._breath(img, cam, f, anc, epos, life, lid_u)
        if not nodof:
            img = dof(img, depth, focus, K=cam.f * 0.0055, near_split=focus * 0.75)
        return finish(img, exposure=float(os.environ.get('V3_EXPO', 1.35)))

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
                    dens = 0.035 * math.exp(-a_ / 0.7) * min(1.0, a_ / 0.06)
                else:
                    pos = mouth + np.array([-0.03, -0.10, -0.02]) * (1 - math.exp(-a_ / 0.5)) + \
                        np.array([0.05, 0.03, -0.03]) * a_
                    rad_m = 0.02 + 0.05 * a_
                    dens = 0.05 * math.exp(-a_ / 1.4) * min(1.0, a_ / 0.2)
                sx, sy, z = cam.project(pos)
                if z <= 0.05:
                    continue
                d2 = float(np.sum((pos - epos) ** 2)) + 0.002
                Ls = np.array([1.0, 0.30, 0.06]) * 0.0026 * (0.02 + life ** 1.6) * lid_u / d2 * 0.25 + \
                    MOON_COL * 0.010
                sig = max(0.8, cam.f * rad_m / z * 0.6)
                hsd.splat_fog(img, float(sx), float(sy), sig, dens, Ls[0], Ls[1], Ls[2],
                              float(fs * 0.37 + q), f / FPS * 0.6, max(2.0, sig * 1.1))


# ============================================================= C H2 ====

BK_BOT, BK_TOP, BK_RT = 0.828, 1.168, 0.30        # the beacon basket (characters.Cairn(seed=11))
FIRE_BASE = np.array([0.0, BK_BOT + 0.10, 0.0])


def fire_env(inten=1.0):
    """What the Ring sees inside her roaring beacon: fire below and all round the far side, the night above and
    toward her."""
    D = env_dirs()
    x, y, z = D[..., 0], D[..., 1], D[..., 2]
    img = np.zeros(D.shape, np.float32)
    toward_fire = np.clip(-x * 0.8 + 0.2 - 0.6 * y, 0, 1)
    img += (np.array([1.0, 0.42, 0.10]) * 6.0 * inten)[None, None] * (toward_fire ** 1.5)[..., None]
    img += (np.array([1.0, 0.55, 0.18]) * 3.0 * inten)[None, None] * np.clip(-y, 0, 1)[..., None] ** 0.7   # coals
    img += (np.array([1.0, 0.75, 0.40]) * 10.0 * inten)[None, None] * np.exp(-((x + 0.6) ** 2 + (y - 0.3) ** 2 + z ** 2) / 0.15)[..., None]
    sky = np.array([0.003, 0.005, 0.012])[None, None] * np.ones_like(D)
    img = np.where((y > 0.35)[..., None], img * 0.3 + sky, img)
    return img.astype(np.float32)


class FireTest:
    """C 168.3-174.3 s (frames 4040-4183). Bag End: after the roar she holds the Ring into her beacon on the tip of her
    steel. It hangs there in the flames, unmarked, its letters awake, not even warm. Her gloved hand trembles; the
    steel dips once toward the coals; she cannot let it fall; she draws it back (4160+)."""
    F0, F1 = 4040, 4183
    HFOV = 40.0

    def pose(self, f):
        t = f / FPS
        tremble = 0.0015 * fnoise1(t * 9.0, 7.0, 2) + 0.0008 * fnoise1(t * 17.0, 9.0, 1)
        dip = 0.012 * smoothstep(4100, 4118, f) * (1 - smoothstep(4122, 4140, f))
        back = smoothstep(4160, 4183, f)
        wrist = np.array([0.370 + 0.08 * back, 1.245 - dip + tremble, 0.010])
        p = dict(
            pelvis=(0.80, 0.63, 0.0), yaw=0.0, lean=10.0, chest=4.0, twist=6.0, neck=16.0, head=20.0, head_yaw=0.0,
            head_roll=0.0, shrug=0.4,
            hand_f=tuple(wrist), elbow_f=(0.3, -1.0, 0.6),
            fdir_f=tuple(nrm([-1.0, 0.02 - 1.2 * dip, -0.10])), palm_f=(0.0, -0.35, -1.0),
            curl_f=(0.80, 0.86, 0.90, 0.92), thumb_f=0.70,
            hand_n=(0.70, 0.72, -0.06), elbow_n=(0.2, -1.0, -0.6), fdir_n=(-0.3, -1.0, 0.0), palm_n=(0.2, 0.0, 1.0),
            curl_n=(0.7, 0.75, 0.8, 0.85), thumb_n=0.5,
            foot_n=(0.40, 0.05, -0.22), knee_n=(-1.0, 0.5, 0.0), toe_n=(-1.0, 0.0, 0.0), sole_n=(0.0, 1.0, 0.0),
            foot_f=(1.05, 0.13, 0.09), knee_f=(-1.0, -0.3, 0.0), toe_f=(0.3, -1.0, 0.0), sole_f=(1.0, 0.1, 0.0),
            hem=0.2, breath=math.sin(2 * math.pi * t / 3.0), tools='none', rock=((0.78, 0.06, 0.10), (0.24, 0.09, 0.18)),
            expr=dict(squint=0.6, brow=-0.3), look_at=(0.15, 1.12, -0.02),
        )
        return p, dip, back

    def steel(self, B, anc, dip):
        """Her steel held like a key: pinched between thumb and curled index, its bar pointing into the fire. Returns
        the tip and the bar's direction."""
        hf = anc['hand_f']
        a = hf['a']
        up = np.array([0.0, 1.0, 0.0])
        d = nrm(a + up * (0.10 - 1.5 * dip))
        p0 = hf['thumb'] + a * 0.004 - hf['n'] * 0.004
        p1 = p0 + d * 0.085
        side = nrm(np.cross(d, up))
        R = np.stack([d, nrm(np.cross(side, d)), side])
        B.group('steel', H3.M_IRON, band=0.001)
        B.box(0.5 * (p0 + p1), np.array([0.0425, 0.0035, 0.0075]), R=R, rnd=0.0012)
        return p1, d

    def ring_on_tip(self, tip, d):
        """Bag End: the Ring lies flat on the end of the flat steel, balanced; a tilt of her wrist would tip it into
        the coals."""
        R, tb, hb, rnd = ring_dims()
        up = np.array([0.0, 1.0, 0.0])
        side = nrm(np.cross(d, up))
        n = nrm(np.cross(side, d))                     # the steel's top face normal
        centre = tip - d * 0.009 + n * (0.0035 + hb)
        rows = H3.ring_frame(n, ref=d)
        return centre, rows

    def camera(self, f, scale, ring_c):
        t = f / FPS
        pos = ring_c + np.array([0.070, 0.105, -0.33]) + np.array([0.002 * fnoise1(t * 0.7, 2.0), 0.002 * fnoise1(t * 0.6, 4.0), 0])
        tgt = ring_c + np.array([0.060, -0.004, 0.0])
        cam = Camera(pos, hfov=self.HFOV, scale=scale)
        yaw, pitch = cam.look_at(tgt)
        return Camera(pos, yaw=yaw, pitch=pitch, hfov=self.HFOV, scale=scale), float(np.linalg.norm(ring_c - pos))

    def wood(self, B):
        """Burning split logs in the basket near its rim, and the nearest iron bars."""
        B.group('coals', H3.M_COAL, disp=1, amp=0.003, scale=40.0, band=0.01)
        rng = np.random.default_rng(3)
        for k in range(9):
            a = -0.9 + 0.25 * k + rng.normal(0, 0.08)
            r = 0.10 + 0.10 * rng.random()
            c = np.array([r * math.cos(a) * 0.9 - 0.04, BK_BOT + 0.04 + 0.06 * rng.random(), r * math.sin(a) + 0.08])
            d = nrm([rng.normal(0, 0.5), 0.9, rng.normal(0, 0.5)])
            B.cone(c - d * 0.14, c + d * 0.14, 0.030, 0.024, k=0.01)
        B.group('bars', H3.M_IRON, band=0.004)
        for k in range(7):
            a = math.radians(-120 + 25 * k)
            b0 = np.array([0.17 * math.cos(a), BK_BOT, 0.17 * math.sin(a)])
            b1 = np.array([BK_RT * math.cos(a), BK_TOP, BK_RT * math.sin(a)])
            B.cone(b0, b1, 0.007, 0.007)
        B.torus(np.array([0.0, BK_TOP, 0.0]), np.eye(3), BK_RT, 0.008, 0.008)

    def render(self, f, scale=0.5):
        t = f / FPS
        p, dip, back = self.pose(f)
        B, F, Hp, anc, hair = figure(p, t, hair_dir=(0.8, -0.3))
        tip, d = self.steel(B, anc, dip)
        ring_c, rows = self.ring_on_tip(tip, d)
        cam, focus = self.camera(f, scale, ring_c)
        cam, focus, nodof = debug_cam(cam, focus, scale)
        add_ring(B, ring_c, rows)
        self.wood(B)
        fl = fire.flicker(t, 3.3, 1.3)
        XP = np.zeros(64)
        ring_xp(XP, ring_c, rows, glow=2.6 + 0.3 * math.sin(t * 5.1))
        XP[25], XP[26] = 1.0, 1.0
        XP[28], XP[29], XP[30] = 0.9 * fl, 60.0, 0.85
        ENV = env_stack(fire_env(1.0 * fl))
        I = 2.2 * fl
        L = [light(FIRE_BASE + [0.0, 0.40, 0.0], (1.0, 0.45, 0.12), 0.75 * I, 0.28, 3.0),
             light(FIRE_BASE + [0.10, 0.12, -0.04], (1.0, 0.38, 0.08), 0.20 * I, 0.10, 3.0),
             light(ring_c + [-0.05, 0.10, 0.05], (1.0, 0.62, 0.25), 0.035 * I, 0.03, 0.0),
             moon_light(0.25)]
        env = hero.env_vec(rim_dir=MOON_DIR, rim=np.array([0.06, 0.08, 0.13]), amb=np.array([0.003, 0.004, 0.008]),
                           bounce=np.array([0.05, 0.02, 0.006]) * fl, ao=0.02)
        img = np.zeros((cam.H, cam.W, 3), np.float32)
        img[:] = np.array([0.002, 0.003, 0.007], np.float32)
        # the fire behind: the beacon's flame and tongues licking up round the Ring
        fimg = np.zeros_like(img)
        fa = np.zeros(img.shape[:2], np.float32)
        fire.draw_flame(fimg, fa, cam, FIRE_BASE, 1.9, 0.31, 0.55, t, 2.9, 10.0 * fl, fire.BONFIRE_STYLE)
        for k, (dx, dz, h, w) in enumerate(((0.14, 0.08, 0.50, 0.07), (0.05, 0.12, 0.62, 0.09), (0.20, 0.05, 0.44, 0.05),
                                           (-0.02, 0.16, 0.7, 0.10), (0.10, 0.20, 0.55, 0.08))):
            fire.draw_flame(fimg, fa, cam, np.array([dx, BK_BOT + 0.10, dz]), h, w, 0.5, t, 7.0 + k, 7.0 * fl,
                            fire.TORCH_STYLE)
        img += fimg
        res = H3.render(cam, B, Hp, L, env, XP, ENV, inscription_ins(), ss=(3 if scale > 0.75 else 2),
                        sil=dict(skin=1.0, eyes=1.0, cap=0.85, cap_brim=0.85, hair=0.85))
        depth = comp(img, res)
        # tongues of flame in front of and beside the Ring: it lies IN the fire
        fimg = np.zeros_like(img)
        for k, (dx, dz, h, w) in enumerate(((0.10, -0.10, 0.44, 0.05), (0.22, -0.02, 0.36, 0.04))):
            fire.draw_flame(fimg, fa, cam, np.array([dx, BK_BOT + 0.16, dz]), h, w, 0.5, t, 17.0 + k, 3.0 * fl,
                            fire.TORCH_STYLE)
        img += fimg * 0.6
        fire.add_glow(img, cam, FIRE_BASE + [0, 0.5, 0], 0.5, 0.06 * fl)
        if not nodof:
            img = dof(img, depth, focus, K=cam.f * 0.0022)
        return finish(img, exposure=float(os.environ.get('V3_EXPO', 0.55)), bloom=0.14)


class Find:
    """C 148.2-153.5 s (frames 3557-3683). Low on the snow by her near knee. 3557 the second strike: its flash (from the
    flint ~1 m above) finds a gold band in a melted hollow, sparks rain and die on the snow; dark again, the band's
    letters are faintly awake (its own light, 3562+); 3600 her gloved near hand comes down out of the dark, 3640 the
    fingers close on it, 3660 lift it away."""
    F0, F1 = 3557, 3683
    HOL = np.array([0.33, 0.0, -0.40])
    HFOV = 42.0
    STRIKE = 3557
    FLINT = np.array([0.32, 1.08, -0.07])

    HAND = False            # the hand closing on the band failed its test (claw read): the shot is the band alone

    def reach(self, f):
        if not self.HAND:
            return 0.0
        return smoothstep(3600, 3640, f) * (1 - smoothstep(3660, 3683, f))

    def pose(self, f):
        t = f / FPS
        r = self.reach(f)
        close = smoothstep(3636, 3650, f)
        W = self.HOL + np.array([0.058, 0.052, 0.018]) + np.array([0.10, 0.45, 0.10]) * (1 - r)
        p = dict(
            pelvis=(0.76, 0.42, -0.05), yaw=0.0, lean=72.0, chest=14.0, twist=0.0, neck=8.0, head=22.0, head_yaw=0.0,
            head_roll=0.0, shrug=0.1,
            hand_n=tuple(W), elbow_n=(0.3, -0.2, -1.0),
            fdir_n=tuple(nrm([-0.60, -0.75, 0.10])), palm_n=tuple(nrm([-0.55, 0.45, 0.70])),
            curl_n=tuple(np.array([0.30, 0.62, 0.82, 0.90]) * (1 - close) + np.array([0.62, 0.84, 0.92, 0.96]) * close),
            thumb_n=0.45 + 0.30 * close, spread_n=0.0, thumbout_n=0.25 * (1 - close),
            hand_f=(0.55, 0.62, 0.10), elbow_f=(0.3, -1.0, 0.4), fdir_f=(-0.6, -0.6, -0.2), palm_f=(0.3, 0.2, -1.0),
            curl_f=(0.9, 0.9, 0.9, 0.9), thumb_f=0.7,
            foot_n=(0.40, 0.05, -0.22), knee_n=(-1.0, 0.5, 0.0), toe_n=(-1.0, 0.0, 0.0), sole_n=(0.0, 1.0, 0.0),
            foot_f=(1.05, 0.13, 0.09), knee_f=(-1.0, -0.3, 0.0), toe_f=(0.3, -1.0, 0.0), sole_f=(1.0, 0.1, 0.0),
            hem=0.1, breath=math.sin(2 * math.pi * t / 3.3), tools='none', rock=None,
            expr=dict(blink=0.5), look_at=tuple(self.HOL),
        )
        return p, r, close

    def ring_pose(self, f, anc, close):
        """Lying tilted on the frozen melt; once her fingers close, it rides in them."""
        R, tb, hb, rnd = ring_dims()
        rest = self.HOL + np.array([0.004, -0.0075 + hb * 0.9 + 0.002, 0.002])
        rows_rest = H3.ring_frame(nrm([0.18, 1.0, -0.12]), ref=(1.0, 0.0, 0.0))
        lift = smoothstep(3650, 3683, f)
        if lift <= 0.0:
            return rest, rows_rest
        hn = anc['hand_n']
        grip = 0.5 * (hn['thumb'] + hn['tips'][0])
        c = rest * (1 - lift) + grip * lift
        return c, rows_rest

    def camera(self, f, scale):
        t = f / FPS
        pos = self.HOL + np.array([-0.050, 0.250, -0.255]) + np.array([0.0015 * fnoise1(t * 0.6, 2.0), 0.001 * fnoise1(t * 0.5, 4.0), 0])
        tgt = self.HOL + np.array([0.020, 0.010, 0.030])
        cam = Camera(pos, hfov=self.HFOV, scale=scale)
        yaw, pitch = cam.look_at(tgt)
        return Camera(pos, yaw=yaw, pitch=pitch, hfov=self.HFOV, scale=scale), float(np.linalg.norm(tgt - pos))

    def render(self, f, scale=0.5):
        t = f / FPS
        cam, focus = self.camera(f, scale)
        cam, focus, nodof = debug_cam(cam, focus, scale)
        p, r, close = self.pose(f)
        B, F, Hp, anc, hair = figure(p, t, hair_dir=(0.6, -0.8))
        ring_c, rows = self.ring_pose(f, anc, close)
        add_ring(B, ring_c, rows)
        snow_ground(B, self.HOL + np.array([0.2, 0.0, 0.0]), reach=3.0, hollow=(self.HOL, 0.045, 0.030))
        d = f - self.STRIKE
        flash = math.exp(-d / 2.4) if d >= 0 else 0.0
        # the letters, faintly awake in the dark; they warm as her fingers come near (the temptation is beautiful)
        # and light her glove from below, then dim in her closed hand
        near = smoothstep(3612, 3646, f) * (1 - smoothstep(3650, 3668, f)) if self.HAND else 0.0
        awake = 0.05 + 0.03 * math.sin(t * 2.3) + 0.45 * near
        XP = np.zeros(64)
        ring_xp(XP, ring_c, rows, glow=awake)
        XP[25], XP[26] = 1.0, 1.0
        # the night the band mirrors: sky, the moon behind her, the snow, her dark figure above
        ENV = env_stack(night_env(moon=1.0, fig_dirs=((np.array([0.35, 0.9, 0.3]), 0.55),)) + 0.0)
        L = [moon_light(0.30),
             light(ring_c + np.array([0.0, 0.006, 0.0]), (1.0, 0.40, 0.10), 0.00004 * awake / 0.05, 0.008, 4.0)]
        if flash > 0.01:
            L.append(light(self.FLINT + [0.015, 0.015, -0.045], (1.0, 0.82, 0.58), 0.30 * flash, 0.05, 0.0))
        env = hero.env_vec(rim_dir=MOON_DIR, rim=np.array([0.08, 0.11, 0.18]), amb=np.array([0.004, 0.006, 0.012]),
                           bounce=np.array([0.012, 0.016, 0.024]), ao=0.015)
        img = np.zeros((cam.H, cam.W, 3), np.float32)
        img[:] = np.array([0.003, 0.005, 0.011], np.float32)
        res = H3.render(cam, B, Hp, L, env, XP, ENV, inscription_ins(), ss=(3 if scale > 0.75 else 2),
                        sil=dict(skin=1.0, eyes=1.0, cap=0.85, cap_brim=0.85, hair=0.85))
        depth = comp(img, res)
        if 0 <= d < 30:
            self._sparks(img, cam, f)
        if not nodof:
            img = dof(img, depth, focus, K=cam.f * 0.0035)
        return finish(img, exposure=float(os.environ.get('V3_EXPO', 1.6)))

    def _sparks(self, img, cam, f):
        """A few of the strike's sparks fall into the frame and die on the snow round the hollow."""
        rng = np.random.default_rng(9)
        n = 14
        for k in range(n):
            t0 = rng.uniform(0.0, 0.25)
            life = rng.uniform(0.25, 0.7)
            land = self.HOL + np.array([rng.normal(0, 0.06), 0.002, rng.normal(0, 0.05)])
            age = (f - self.STRIKE) / FPS - t0
            if age < 0 or age > life + 0.4:
                continue
            fall = 0.35
            if age < fall:
                u = age / fall
                pos = land + np.array([0.0, 0.25 * (1 - u) ** 1.6, 0.0])
                prev = land + np.array([0.0, 0.25 * (1 - max(0.0, u - 0.08)) ** 1.6, 0.0])
                e = 1.0
            else:
                pos = prev = land
                e = math.exp(-(age - fall) / 0.12)
            sx, sy, z = cam.project(pos)
            px, py, _ = cam.project(prev)
            if z <= 0.02:
                continue
            en = 1.2 * e * cam.scale ** 2
            from core import splat_streak
            splat_streak(img, float(px), float(py), float(sx), float(sy), max(0.6, 1.1 * cam.scale), en, en * 0.55, en * 0.2)


def inscription_ins():
    return H3.inscription_accord()[0]


class DeadEmberPOV(DeadEmber):
    POV = True
    HFOV = 40.0


SHOTS = dict(deadember=DeadEmber, deadember_pov=DeadEmberPOV, firetest=FireTest, find=Find)


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
