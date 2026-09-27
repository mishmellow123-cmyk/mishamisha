"""HEROINE v3 close-ups (BIBLE_V3 H2, H5 + REVISION 1): hands, pose and light. No faces.

Each shot is a class with render(f, scale) -> display sRGB (look.finish). Frames are seconds x 24 in the shot's own
cut, so they drop straight onto that cut's timeline (renders/heroine_B/, renders/heroine_C/):

  DeadEmber  B 880-1119 (locked sheet, bars 12-14) H5. She kneels at the summit cairn, lifts the clay fire-pot's lid
             (the knock on 920), one red eye of ember on the ash; she blows (980); it greys; the last red point on
             1100 sits where H1's first spark is born (the match cut). Her head is a silhouette.
             -> renders/deadember_B (B frames).
  Find       C 3009-3059 H2 (fallback to the Blender find). Low on the snow by her knee: the second strike's flash
             finds a gold band in a melted hollow; in the dark again its letters are faintly awake.
  FireTest   C 3392-3599 H2, Bag End (fallback to the Blender fire test). In her roaring beacon the Ring lies on the
             up-turned tip of her C-shaped fire-steel, unmarked, letters awake, not even warm; it tips and does not fall.

Every Ring carries THE SCRIPT OF FIRE, the canonical inscription (assets/ring, MONTAGE-3D-2; outer and inner faces as
inscription.json maps them) at the canonical band proportions; ring.py's band and strip only if those assets are
missing. The hands are heroine.py's anatomical hands in thin leather gloves (hsdf3.gloves); the hood and woven shawl
are hsdf3.wardrobe_v3's. Rendered by hsdf3 (the v3 fork of the heroine tracer).

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
    """The canonical Ring's proportions (assets/ring/inscription.json: inner circumference / band width 11.358, outer
    14.137, so the script of fire keeps its letter shapes) at a slim finger; ring.py's section (R_OUT 0.186, R_IN 0.138,
    BAND 0.084) if the canonical assets are missing. The edge round is ring.py's."""
    RG = ring_py()
    s = RING_INNER / RG.R_IN
    can = H3.inscription_canon()
    if can is not None:
        w = RING_INNER * 2 * math.pi / can[1]['aspect_i']
        r_out = w * can[1]['aspect_o'] / (2 * math.pi)
        return 0.5 * (r_out + RING_INNER), 0.5 * (r_out - RING_INNER), 0.5 * w, RG.ROUND * s
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
    if H3.inscription_canon() is not None:           # the script of fire (MONTAGE-3D-2): one inscription on both rings
        return H3.ring_xp_canon(XP, centre, rows, R, tb, hb, glow, engrave)
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


def aim_at_screen(pos, tgt, P, px_full, hfov, iters=4):
    """A new look-at target (same camera position) that puts world point P at full-res pixel px_full."""
    tgt = np.asarray(tgt, np.float64).copy()
    for _ in range(iters):
        c = Camera(pos, hfov=hfov, scale=1.0)
        yaw, pitch = c.look_at(tgt)
        c = Camera(pos, yaw=yaw, pitch=pitch, hfov=hfov, scale=1.0)
        sx, sy, z = c.project(np.asarray(P, np.float64))
        d = float(np.linalg.norm(tgt - np.asarray(pos)))
        right, up = c.R[:, 0], c.R[:, 1]
        tgt = tgt + right * (sx - px_full[0]) / c.f * d - up * (sy - px_full[1]) / c.f * d
    return tgt


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
        scarf = static_chain(sa, scarf_dir, scarf_n, 0.078, droop=0.55, sway=(0.12, 0.05), seed=1.0 - 1.7 * t)
    else:
        scarf = None
    hr = an['hair_root'][:2]
    hair = [static_chain(hr + np.array([-0.010 + 0.004 * k, 0.030 - 0.010 * k]), hair_dir, 8,
                         0.045 + 0.006 * (k % 3), droop=0.6, sway=(0.05, 0.03), seed=k * 1.3 - 2.1 * t) for k in range(9)]
    B, F, Hp, anc = hero.build_figure(pose, t, scarf_pts=scarf, hair_pts=hair, detail=detail)
    if gloved:
        H3.gloves(B, Hp)
    # H5 calls: a hood / wool cowl (no beanie, no streaming hair) and the red scarf as a woven wool shawl
    H3.wardrobe_v3(B, F, anc['J'], anc['scarf_anchor'], scarf_pts=scarf, t=t)
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
    """B · THE DEAD EMBER, bars 12-14 on the locked sheet: B frames 880-1119. Keys: she kneels at the cairn, her far
    gloved hand on the lid's knob; it lifts the lid 888-904, carries it away from the lens and sets it down on the
    stone on the far side, the small knock on bar 12 b3 (920); the hand returns to steady the pot's rim (924-944).
    One red eye 920-980. From bar 13 b2 (980) she blows: the eye brightens a little (hope), then greys (1000-1100);
    a second breath (1016) does nothing. The last red point is on bar 14 b4 (1100) and goes out by 1116; the breath
    hangs. The camera drifts (1040-1100) so that last point sits where H1's first spark is born (the match cut on
    bar 15 b1)."""
    F0, F1 = 880, 1119
    POT = np.array([0.46, 0.66, -0.03])       # H5: HELD, cradled on her near palm in front of her chest (was on the cairn)
    HFOV = 38.0
    POV = False
    MOON = np.array([0.20, 0.62, -0.76]) / np.linalg.norm([0.20, 0.62, -0.76])   # B's moon: high on her left
    SPARK_PX = (757.0, 256.0)                 # H1's flint edge (strike 1) in the full-res frame: the match cut
    EYE = POT + np.array([0.002, 0.1145, 0.014])   # the ember's eye: its top, turned a little toward the lens
    KNOCK = 920
    CRAFTED = True                            # HEROINE-L: the crafted vessel (pot()); False = the 16:00Z ball pot

    def __init__(self):
        pass

    # ---- keys
    def life(self, f):
        e = 1.0 - 0.55 * smoothstep(1000, 1030, f) - 0.45 * smoothstep(1030, 1100, f)
        e += 0.18 * smoothstep(982, 988, f) * (1 - smoothstep(992, 1004, f))          # a breath of hope
        return float(np.clip(e, 0.0, 1.2))

    def last_point(self, f):
        """The last red point: a pinprick left in the eye once the rest has greyed; out by 1116."""
        return 0.9 * smoothstep(1080, 1098, f) * (1 - smoothstep(1100, 1116, f))

    def lid_u(self, f):
        return smoothstep(888, self.KNOCK, f)

    def blow_env(self, f):
        return max(smoothstep(980, 984, f) * (1 - smoothstep(1000, 1004, f)),
                   smoothstep(1016, 1020, f) * (1 - smoothstep(1032, 1036, f)))

    def pose(self, f):
        t = f / FPS
        u = self.lid_u(f)
        blow = self.blow_env(f)
        p = dict(
            pelvis=(0.80, 0.57, 0.0), yaw=0.0, lean=20.0 + 4.0 * blow, chest=6.0, twist=0.0, neck=12.0,
            head=40.0 + 6.0 * blow, head_yaw=0.0, head_roll=0.0, shrug=0.25,
            elbow_f=(0.2, -1.0, 0.9), elbow_n=(0.2, -1.0, -0.9),
            curl_n=(0.30, 0.34, 0.40, 0.46), thumb_n=0.3, spread_n=0.1,
            foot_n=(1.12, 0.07, -0.12), foot_f=(1.12, 0.07, 0.12), knee_n=(-1.0, -0.6, -0.05), knee_f=(-1.0, -0.6, 0.05),
            toe_n=(1.0, -0.2, 0.0), toe_f=(1.0, -0.2, 0.0), sole_n=(0.0, -1.0, 0.0), sole_f=(0.0, -1.0, 0.0),
            hem=0.05, breath=math.sin(2 * math.pi * t / 3.6) * (1 - blow),
        )
        # the lid: her FAR hand lifts it by the knob and sets it down on the stone beyond the pot (away from the lens:
        # no arm crosses the frame), then comes back to steady the pot's rim
        rest = self.POT + np.array([0.0, 0.150, 0.0])
        lift = self.POT + np.array([0.012, 0.215, 0.040])
        down = self.POT + np.array([0.060, -0.200, 0.230])      # set down on the cairn stone below, out of frame
        a = smoothstep(0.0, 0.5, u)
        b = smoothstep(0.5, 1.0, u)
        lc = rest * (1 - a) + lift * a
        lc = lc * (1 - b) + down * b
        back = smoothstep(self.KNOCK + 4, self.KNOCK + 24, f)
        grip = lc + np.array([0.030, 0.050, 0.020])
        steady = self.POT + np.array([0.118, 0.105, 0.020])       # the pot's right belly: clear of the mouth
        p['hand_f'] = tuple(grip * (1 - back) + steady * back)
        fg, pg = nrm([-0.45, -0.75, -0.40]), nrm([0.05, -0.90, -0.40])
        fs, ps = nrm([-0.20, -0.95, -0.10]), nrm([-0.97, 0.05, -0.20])
        p['fdir_f'] = tuple(nrm(fg * (1 - back) + fs * back))
        p['palm_f'] = tuple(nrm(pg * (1 - back) + ps * back))
        p['curl_f'] = tuple(np.array([0.55, 0.62, 0.70, 0.74]) * (1 - back) + np.array([0.30, 0.34, 0.40, 0.46]) * back)
        p['thumb_f'] = 0.55 * (1 - back) + 0.25 * back
        p['spread_f'] = 0.15 * back
        # H5: the vessel is HELD: her near gloved hand is round the pot's belly on the lens side throughout (it steadies
        # the pot as the lid comes off, and keeps hold while she blows)
        p['hand_n'] = tuple(self.POT + np.array([0.070, -0.030, -0.060]))
        p['fdir_n'] = tuple(nrm([-0.80, 0.12, 0.50]))
        p['palm_n'] = tuple(nrm([0.0, 1.0, 0.05]))
        p['curl_n'] = (0.30, 0.34, 0.40, 0.46)
        p['thumb_n'] = 0.20
        p['spread_n'] = 0.10
        p['elbow_n'] = (0.5, -1.0, -0.6)
        p['tools'] = 'none'
        p['rock'] = None
        p['expr'] = dict(purse=blow, blink=0.6, brow=0.2)
        p['look_at'] = self.POT + np.array([0.0, 0.09, 0.0])
        lid_flat = b                                   # tipped while carried, flat once set down
        tip = math.sin(math.pi * min(1.0, u)) * 0.6
        return p, lc, (tip, lid_flat), blow

    # ---- props
    def pot(self, B, lid_c, lid_u):
        """A round-bellied clay fire-pot (~15 cm), its lip sooted; ash inside with one ember; the lid in her hand."""
        c = self.POT
        if not self.CRAFTED:
            B.group('pot', H3.M_CLAY, disp=1, amp=0.0011, scale=55.0, band=0.005)
            tilt = np.stack([nrm([1.0, 0.05, 0.0]), nrm([-0.05, 1.0, 0.03]), nrm([0.0, -0.03, 1.0])])   # hand-thrown
            B.ell(c + [0, 0.070, 0], np.array([0.079, 0.068, 0.077]), R=tilt)
            B.ell(c + [0.012, 0.082, -0.010], np.array([0.068, 0.050, 0.070]), R=tilt, k=0.03)          # shoulder
            B.ell(c + [0, 0.012, 0], np.array([0.052, 0.014, 0.052]), k=0.02)                      # foot
            B.cone(c + [0, 0.112, 0], c + [0, 0.136, 0], 0.050, 0.044, k=0.014)                   # neck
            B.torus(c + [0, 0.137, 0], np.eye(3), 0.041, 0.0078, 0.0088, k=0.010)                  # rim
            B.ell(c + [0, 0.070, 0], np.array([0.069, 0.059, 0.069]), op=1, k=0.004)               # hollow
        else:
            # HEROINE-L (the H5 critic: "a coconut, a bowling ball"): a CRAFTED vessel, not a lumpy ball. A squat belly
            # under a carinated shoulder, a flat foot, a short neck and a rolled lip; two pierced lugs on the shoulder;
            # a leather thong knotted round the neck. Fired clay: a faint throwing unevenness, no lumps.
            B.group('pot', H3.M_CLAY, disp=1, amp=0.00035, scale=26.0, band=0.005)
            tilt = np.stack([nrm([1.0, 0.03, 0.0]), nrm([-0.03, 1.0, 0.02]), nrm([0.0, -0.02, 1.0])])   # thrown, not true
            B.cone(c + [0, 0.001, 0], c + [0, 0.016, 0], 0.047, 0.055, k=0.006)                   # the flat foot
            B.ell(c + [0, 0.064, 0], np.array([0.081, 0.058, 0.079]), R=tilt, k=0.012)            # squat belly
            B.ell(c + [0.004, 0.092, -0.003], np.array([0.068, 0.034, 0.067]), R=tilt, k=0.022)   # carinated shoulder
            B.cone(c + [0, 0.108, 0], c + [0, 0.133, 0], 0.047, 0.040, k=0.010)                   # neck
            B.torus(c + [0, 0.137, 0], np.eye(3), 0.0425, 0.0066, 0.0070, k=0.007)                 # the rolled lip
            for sg in (1.0, -1.0):                                                                  # pierced lugs
                Rl_ = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
                B.torus(c + [sg * 0.077, 0.089, 0.0], Rl_, 0.0082, 0.0030, 0.0036, k=0.004,
                        arc=((math.pi / 2 if sg > 0 else -math.pi / 2), 2.1))
            B.ell(c + [0, 0.064, 0], np.array([0.071, 0.050, 0.069]), op=1, k=0.004)               # hollow
        B.cone(c + [0, 0.095 if self.CRAFTED else 0.100, 0], c + [0, 0.190, 0], 0.0335, 0.0335, op=1, k=0.004)   # mouth
        B.ell(c + [0.026, 0.1475, -0.030], np.array([0.0055, 0.0040, 0.0050]), op=1, k=0.0015)   # an old chip in the lip
        # soot: the lip blackened by years of carried fire
        B.group('soot', H3.M_BOOT, band=0.002)          # dark soot (M_COAL would grey to ash with no fire in it)
        B.torus(c + [0, 0.1395, 0], np.eye(3), 0.0415 if self.CRAFTED else 0.0395, 0.0056, 0.0066, k=0.0)
        if self.CRAFTED:
            # the thong: once round the neck, a knot on her side, two short ends hanging over the shoulder
            B.group('thong', H3.M_BOOT, band=0.002)
            B.torus(c + [0, 0.121, 0], np.eye(3), 0.0452, 0.0021, 0.0023, k=0.0)
            kn = c + np.array([0.030, 0.121, -0.034])
            B.ell(kn, np.array([0.0048, 0.0040, 0.0045]), k=0.002)
            for dx, dz, ln in ((0.010, -0.004, 0.034), (-0.004, -0.010, 0.026)):
                B.cone(kn, kn + np.array([dx, -ln, dz - 0.010]), 0.0021, 0.0017, k=0.002)
        B.group('ash', H3.M_ASH, disp=1, amp=0.0012, scale=160.0, band=0.004)
        B.ell(c + [0, 0.080, 0], np.array([0.058, 0.026, 0.058]))
        B.group('ember', H3.M_EMBER, disp=1, amp=0.0016, scale=150.0, band=0.004)
        e = c + np.array([0.006, 0.107, 0.018])
        B.ell(e, np.array([0.0135, 0.0085, 0.011]), R=np.stack([nrm([1, 0.1, 0.3]), nrm([-0.1, 1, 0]), nrm([-0.3, 0, 1])]))
        B.ell(e + [0.009, -0.002, 0.004], np.array([0.0075, 0.0060, 0.0070]), k=0.004)
        B.ell(e + [-0.008, -0.003, -0.005], np.array([0.0060, 0.0045, 0.0055]), k=0.004)
        # the lid: a shallow clay dome with a knob, lifted in her near hand and tipped toward her
        tip, flat = lid_u
        ax = nrm(np.array([0.0, 1.0, 0.0]) * (1 - tip) + nrm([0.25, 1.0, 0.45]) * tip)
        Rl = H3.ring_frame(ax, ref=(1.0, 0.0, 0.0))
        B.group('lid', H3.M_CLAY, disp=1, amp=(0.00025 if self.CRAFTED else 0.0006), scale=(40.0 if self.CRAFTED else 90.0),
                band=0.004)
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
            # H5: about 45 deg down (40), from her front-left so her dark coat and knees fill the frame behind the pot
            # (no snow wedge), her head well above the frame
            tgt = self.POT + np.array([0.020, 0.095, 0.0])
            pos = tgt + np.array([-0.56, 0.56, -0.18])
        pos = pos + np.array([0.004 * fnoise1(t * 0.6, 3.0), 0.003 * fnoise1(t * 0.5, 5.0), 0.0])
        w = smoothstep(1040, 1100, f) if not self.POV else 0.0
        if w > 0.0:
            eye = self.EYE
            aimed = aim_at_screen(pos, tgt, eye, self.SPARK_PX, hf)
            tgt = tgt * (1 - w) + aimed * w
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
        tail = static_chain(front[:2], (-0.05, -1.0), 7, 0.074, droop=0.1, sway=(0.04, 0.0), seed=2.0 - 0.9 * t)
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
        XP[21:24] = self.EYE
        XP[31] = 0.0030 + 0.0026 * min(1.0, life)
        XP[32] = self.last_point(f)
        XP[20] *= 1.0 - smoothstep(1100, 1116, f)
        XP[25] = 1.0
        XP[26] = 1.0
        ENV = env_stack(night_env(moon=0.8))
        # light: B's silver moon from behind her; the ember inside the pot (its walls and her hands shadow it)
        open_ = smoothstep(888, 904, f)
        glow = 0.0026 * (0.02 + life ** 1.6) * (0.35 + 0.65 * open_) + 0.0004 * self.last_point(f)
        L = [moon_light(0.45, self.MOON),
             light(epos + [0, 0.012, 0], (1.0, 0.30, 0.06), glow, 0.010, 6.0)]
        env = hero.env_vec(rim_dir=self.MOON, rim=np.array([0.10, 0.13, 0.20]), amb=np.array([0.004, 0.006, 0.011]),
                           bounce=np.array([0.010, 0.013, 0.020]), ao=0.02)
        Mt = H3.material_table3()
        if self.CRAFTED:
            # fired clay, not husk: broad mottling, a faint burnish, almost no bump
            Mt[H3.M_CLAY, 3] = 0.70
            Mt[H3.M_CLAY, 12:15] = [24.0, 0.20, 0.03]
        res = H3.render(cam, B, Hp, L, env, XP, ENV, None, M=Mt, ss=(3 if scale > 0.75 else 2),
                        sil=dict(skin=1.0, eyes=1.0, cap=0.6, cap_brim=0.6, hair=0.6))
        img = np.zeros((cam.H, cam.W, 3), np.float32)
        img[:] = np.array([0.004, 0.006, 0.012], np.float32)
        depth = comp(img, res)
        # H5: no bright snow shapes behind her: the far ground (well behind the focus) graded down into the night
        far = np.clip((depth - (focus + 0.25)) / 0.35, 0.0, 1.0)
        img *= (1 - 0.55 * far)[..., None]
        # breath: from her hidden mouth down into the pot's glow, then hanging in the moonlight
        self._breath(img, cam, f, anc, epos, life, open_)
        if not nodof:
            img = dof(img, depth, focus, K=cam.f * 0.0055, near_split=focus * 0.75)
        return finish(img, exposure=float(os.environ.get('V3_EXPO', 1.35)))

    def _breath(self, img, cam, f, anc, epos, life, lid_u):
        import heroine_sdf as hsd
        mouth = anc['mouth']
        puffs = ((982, 1.0, 'blow'), (1016, 0.9, 'blow'), (930, 1.6, 'out'), (1040, 2.6, 'hang'))
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
                    dens = 0.085 * math.exp(-a_ / 0.8) * min(1.0, a_ / 0.06)
                else:
                    pos = mouth + np.array([-0.03, -0.10, -0.02]) * (1 - math.exp(-a_ / 0.5)) + \
                        np.array([0.05, 0.03, -0.03]) * a_
                    rad_m = 0.02 + 0.05 * a_
                    dens = 0.11 * math.exp(-a_ / 1.6) * min(1.0, a_ / 0.2)
                sx, sy, z = cam.project(pos)
                if z <= 0.05:
                    continue
                d2 = float(np.sum((pos - epos) ** 2)) + 0.002
                Ls = np.array([1.0, 0.30, 0.06]) * 0.0026 * (0.02 + life ** 1.6) * lid_u / d2 * 1.2 + \
                    MOON_COL * 0.14
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
    """C · THE FIRE TEST (Bag End) on the locked sheet: C frames 3392-3599 (the roar is H1's, bar 43 b1 = 3360). Her
    gloved fist brings her steel into the beacon (3392-3404) and on bar 43 b3 (3400) the Ring lies on its tip in the
    flames, unmarked, letters awake, not even warm; her hand trembles. On bar 44 b3 (3480) the steel tips (it dips and
    rolls), the Ring slides toward the edge and should fall into the fire, and does not: she cannot let it; she levels
    it (3500). On bar 45 b3 (3560) she draws it out of the fire (out of frame right by ~3590)."""
    F0, F1 = 3392, 3599
    HFOV = 40.0
    TIP = 3480
    OUT = 3560
    C_HB = 0.046                    # half length of the steel's back: both corners clear her fist

    def ref_ring(self):
        """Where the Ring rests during the hold (3440): the locked camera frames it."""
        if not hasattr(self, '_ref'):
            p, dip, back = self.pose(3440)
            an = hero.build_figure(p, 3440 / FPS)[3]
            p_, d, s_ = self.steel(hero.Builder(), an, dip)
            self._ref = self.ring_on_tip(p_, d, s_)[0]
        return self._ref

    def tipping(self, f):
        return smoothstep(self.TIP - 6, self.TIP + 4, f) * (1 - smoothstep(self.TIP + 10, self.TIP + 26, f))

    def pose(self, f):
        t = f / FPS
        tremble = (0.0015 * fnoise1(t * 9.0, 7.0, 2) + 0.0008 * fnoise1(t * 17.0, 9.0, 1)) * (1 + 1.5 * self.tipping(f))
        dip = 0.010 * self.tipping(f)
        enter = 1 - smoothstep(self.F0, self.F0 + 12, f)
        back = smoothstep(self.OUT - 4, self.OUT + 30, f)
        wrist = np.array([0.315 + 0.14 * enter + 0.20 * back, 1.225 - dip + tremble + 0.02 * back, 0.010])
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

    def steel(self, B, anc, dip, roll=0.0):
        """Her C-shaped fire-steel, the one she strikes with (H5 calls: never a flat bar): flat stock (7 x 2.4 mm)
        forged into a C standing in the plane of her reach, broad face to the lens. Its straight back lies in her fist;
        its upper end rounds forward into the long arm that ends in a small up-turned hook (the tip); its lower end
        rounds forward into a short arm rolled up in a scroll. Returns the point on the long arm's top edge at its end
        (where the Ring lies, round the hook), the reach direction d and the C's in-plane up s."""
        hf = anc['hand_f']
        a = hf['a']
        upw = np.array([0.0, 1.0, 0.0])
        d = nrm(a + upw * (0.06 - 1.5 * dip))
        s = nrm(hf['sd'] - np.dot(hf['sd'], d) * d)
        if s[1] < 0:
            s = -s
        if roll:
            s = nrm(hero.rot_about(s, d, roll))
        n = np.cross(d, s)
        g = hf['palm'] + a * 0.018 + hf['n'] * 0.022          # the back, inside the fist
        R = np.stack([d, n, s])        # local x = forward, y = the C's normal (the torus axis), z = in-plane up
        B.group('steel', H3.M_IRON, band=0.001)
        tp = 0.0012                    # half thickness, toward the lens
        wb, wa = 0.0036, 0.0029        # half widths in the C's plane: the back, the arms
        Hb, r = self.C_HB, 0.014       # half length of the back; the corners' radius
        L, rh = 0.040, 0.0048          # the long arm; the hook
        Ll, rs = 0.014, 0.0065         # the short arm; the scroll
        B.box(g, np.array([wb, tp, Hb]), R=R, rnd=0.0009)
        cu, cl = g + s * Hb + d * r, g - s * Hb + d * r
        B.torus(cu, R, r, wb * 0.95, tp, arc=(3 * math.pi / 4, math.pi / 4 + 0.03))
        B.torus(cl, R, r, wb * 0.95, tp, arc=(-3 * math.pi / 4, math.pi / 4 + 0.03))
        a_top = cu + s * r                                   # the long arm's root
        B.box(a_top + d * (0.5 * L), np.array([0.5 * L, tp, wa]), R=R, rnd=0.0009)
        B.torus(a_top + d * L + s * rh, R, rh, wa, tp, arc=(-0.61, 0.96))      # the tip: an up-turned hook
        l_bot = cl - s * r
        B.box(l_bot + d * (0.5 * Ll), np.array([0.5 * Ll, tp, wa]), R=R, rnd=0.0009)
        B.torus(l_bot + d * Ll + s * rs, R, rs, wa * 0.9, tp, arc=(0.62, 2.19))  # the scroll
        return a_top + d * L + s * wa, d, s

    def ring_on_tip(self, p, d, s, slide=0.0):
        """Bag End: the Ring lies flat on the long arm's top edge with the up-turned hook rising through its hole,
        balanced there; `slide` (m) edges it sideways, down the steel's roll, toward falling (the hook holds it)."""
        R, tb, hb, rnd = ring_dims()
        n = np.cross(d, s)
        sg = 1.0 if np.dot(np.array([0.0, -1.0, 0.0]), n) >= 0 else -1.0
        axis = nrm(s + n * sg * math.tan(math.radians(24.0 * slide / 0.0055)))
        centre = p - d * 0.0003 + axis * hb + n * sg * slide
        return centre, H3.ring_frame(axis, ref=d)

    def camera(self, f, scale, ring_c):
        t = f / FPS
        pos = ring_c + np.array([0.040, 0.030, -0.46]) + np.array([0.002 * fnoise1(t * 0.7, 2.0), 0.002 * fnoise1(t * 0.6, 4.0), 0])
        tgt = ring_c + np.array([0.042, -0.050, 0.015])        # the whole C and her fist; the Ring upper left
        cam = Camera(pos, hfov=self.HFOV, scale=scale)
        yaw, pitch = cam.look_at(tgt)
        return Camera(pos, yaw=yaw, pitch=pitch, hfov=self.HFOV, scale=scale), float(np.linalg.norm(ring_c - pos))

    def wood(self, B, half='all'):
        """Burning split logs in the basket near its rim, and the aged basket (`half` of it: the far half renders
        behind the flames, the near half with her)."""
        B.group('coals', H3.M_COAL, disp=1, amp=0.003, scale=40.0, band=0.01)
        rng = np.random.default_rng(3)
        for k in range(9):
            a = -0.9 + 0.25 * k + rng.normal(0, 0.08)
            r = 0.10 + 0.10 * rng.random()
            c = np.array([r * math.cos(a) * 0.9 - 0.04, BK_BOT - 0.01 + 0.04 * rng.random(), r * math.sin(a) + 0.10])
            d = nrm([rng.normal(0, 0.5), 0.9, rng.normal(0, 0.5)])
            B.cone(c - d * 0.14, c + d * 0.14, 0.030, 0.024, k=0.01)
        H3.basket_v3(B, BK_BOT, BK_TOP, 0.17, BK_RT, half=half)      # the same aged basket as the flint take

    def render(self, f, scale=0.5):
        t = f / FPS
        p, dip, back = self.pose(f)
        B, F, Hp, anc, hair = figure(p, t, hair_dir=(0.8, -0.3), scarf=False)
        tp = self.tipping(f)
        tip, d, s_ = self.steel(B, anc, dip, roll=9.0 * tp)
        ref = self.ref_ring()
        ring_c, rows = self.ring_on_tip(tip, d, s_, slide=0.0055 * smoothstep(self.TIP - 2, self.TIP + 8, f)
                                        * (1 - 0.35 * smoothstep(self.TIP + 14, self.TIP + 30, f)))
        cam, focus = self.camera(f, scale, ref)            # a locked frame: the steel moves through it
        cam, focus, nodof = debug_cam(cam, focus, scale)
        add_ring(B, ring_c, rows)
        H3.basket_v3(B, BK_BOT, BK_TOP, 0.17, BK_RT, half='front')
        fl = fire.flicker(t, 3.3, 1.3)
        XP = np.zeros(64)
        ring_xp(XP, ring_c, rows, glow=1.5 + 0.2 * math.sin(t * 5.1))      # letters awake; the gold not even warm
        XP[25], XP[26] = 1.0, 1.0
        XP[28], XP[29], XP[30] = 0.9 * fl, 60.0, 0.85
        XP[33] = BK_BOT + 0.12                                   # the basket's soot line
        ENV = env_stack(fire_env(0.45 * fl))                   # gold, not white-hot: the Ring is not even warm
        I = 2.2 * fl
        L = [light(FIRE_BASE + [0.0, 0.40, 0.0], (1.0, 0.45, 0.12), 0.75 * I, 0.28, 3.0),
             light(FIRE_BASE + [0.10, 0.12, -0.04], (1.0, 0.38, 0.08), 0.20 * I, 0.10, 3.0),
             light(ring_c + [-0.05, 0.10, 0.05], (1.0, 0.62, 0.25), 0.035 * I, 0.03, 0.0),
             moon_light(0.25)]
        env = hero.env_vec(rim_dir=MOON_DIR, rim=np.array([0.06, 0.08, 0.13]), amb=np.array([0.003, 0.004, 0.008]),
                           bounce=np.array([0.05, 0.02, 0.006]) * fl, ao=0.02)
        img = np.zeros((cam.H, cam.W, 3), np.float32)
        img[:] = np.array([0.002, 0.003, 0.007], np.float32)
        ss = 3 if scale > 0.75 else 2
        # the basket's far half and the burning logs, behind the flames
        Bb = hero.Builder()
        self.wood(Bb, half='back')
        XPb = XP.copy()
        XPb[1] = 0.0
        comp(img, H3.render(cam, Bb, Hp, L, env, XPb, ENV, None, ss=ss))
        # the fire: the beacon's flame, and tongues licking up behind the Ring (none behind her fist: no glow fringe)
        fimg = np.zeros_like(img)
        fa = np.zeros(img.shape[:2], np.float32)
        fire.draw_flame(fimg, fa, cam, FIRE_BASE, 1.9, 0.31, 0.55, t, 2.9, 10.0 * fl, fire.BONFIRE_STYLE)
        fire.draw_flame(fimg, fa, cam, np.array([0.15, BK_BOT + 0.06, 0.14]), 1.1, 0.16, 0.55, t, 5.3, 6.0 * fl,
                        fire.BONFIRE_STYLE)                    # the flames the Ring lies in
        for k, (dx, dz, h, w) in enumerate(((0.14, 0.08, 0.50, 0.07), (0.05, 0.12, 0.62, 0.09), (0.17, 0.04, 0.46, 0.045),
                                           (-0.02, 0.16, 0.7, 0.10), (0.10, 0.20, 0.55, 0.08), (0.19, 0.12, 0.40, 0.04))):
            fire.draw_flame(fimg, fa, cam, np.array([dx, BK_BOT + 0.10, dz]), h, w, 0.5, t, 7.0 + k, 7.0 * fl,
                            fire.TORCH_STYLE)
        img += fimg
        fire.add_glow(img, cam, FIRE_BASE + [0, 0.5, 0], 0.5, 0.06 * fl)
        Mft = H3.material_table3()
        Mft[H3.M_GLOVE, 8] = 0.15                               # H5: no warm fringe round the glove's silhouette
        res = H3.render(cam, B, Hp, L, env, XP, ENV, inscription_ins(), Mft, ss=ss,
                        sil=dict(skin=1.0, eyes=1.0, cap=0.85, cap_brim=0.85, hair=0.85))
        depth = comp(img, res)
        # her figure's mask (hand, sleeve, coat): the flames in front of the Ring never paint over it
        fig = np.zeros(img.shape[:2], np.float32)
        if res is not None:
            y0, x0, _, a_, _, grp = res
            keep = [gi for gi, g in enumerate(B.groups) if g['name'] not in ('ring', 'steel', 'basket_front')]
            fig[y0:y0 + a_.shape[0], x0:x0 + a_.shape[1]] = a_ * np.isin(grp, keep)
            fig = cv2.GaussianBlur(fig, (0, 0), 1.5 * cam.W / 1920.0 + 0.5)
        # tongues of flame in front of and beside the Ring: it lies IN the fire
        fimg = np.zeros_like(img)
        for k, (dx, dz, h, w) in enumerate(((0.10, -0.10, 0.44, 0.05), (0.16, -0.09, 0.34, 0.035))):
            fire.draw_flame(fimg, fa, cam, np.array([dx, BK_BOT + 0.16, dz]), h, w, 0.5, t, 17.0 + k, 3.0 * fl,
                            fire.TORCH_STYLE)
        img += fimg * 0.6 * (1 - fig[..., None])
        if not nodof:
            img = dof(img, depth, focus, K=cam.f * 0.0022)
        # H5: no glow fringe round the hand: the fire's bloom is held off her glove and sleeve
        img = img + (look.bloom(img, 0.04, 0.9) - img) * (1 - 0.9 * fig[..., None])
        return finish(img, exposure=float(os.environ.get('V3_EXPO', 0.55)), bloom=0.0)


class Find:
    """C 148.2-153.5 s (frames 3557-3135). Low on the snow by her near knee. 3557 the second strike: its flash (from the
    flint ~1 m above) finds a gold band in a melted hollow, sparks rain and die on the snow; dark again, the band's
    letters are faintly awake (its own light, 3562+); 3052 her gloved near hand comes down out of the dark, 3092 the
    fingers close on it, 3112 lift it away."""
    F0, F1 = 3009, 3059
    HOL = np.array([0.33, 0.0, -0.40])
    SHIFT = 3009 - 3557              # the locked sheet: strike 2 on C 3009 (bar 38 b2 + 29)
    HFOV = 42.0
    STRIKE = 3009
    FLINT = np.array([0.32, 1.08, -0.07])

    HAND = False            # the hand closing on the band failed its test (claw read): the shot is the band alone
    # H5 calls: if her glove is in this shot it comes in from the SIDE (frame right, low over the snow, palm down),
    # never a gauntlet descending from above; it slides in over the last beats and the edit cuts to her closed fist
    HAND_SIDE = os.environ.get('V3_FIND_HAND', 'side') == 'side'     # V3_FIND_HAND=none: the band alone

    def reach(self, f):
        if self.HAND_SIDE:
            return smoothstep(3036, 3058, f)
        if not self.HAND:
            return 0.0
        return smoothstep(3052, 3092, f) * (1 - smoothstep(3112, 3135, f))

    def pose(self, f):
        t = f / FPS
        r = self.reach(f)
        close = 0.0 if self.HAND_SIDE else smoothstep(3088, 3102, f)
        if self.HAND_SIDE:
            W = self.HOL + np.array([0.19, 0.040, 0.0]) + np.array([0.21, 0.012, -0.02]) * (1 - r)
            hand = dict(hand_n=tuple(W), elbow_n=(1.0, 0.25, -0.35), fdir_n=tuple(nrm([-1.0, -0.22, 0.05])),
                        palm_n=(0.15, -1.0, 0.15), curl_n=(0.26, 0.32, 0.40, 0.46), thumb_n=0.40, spread_n=0.0,
                        thumbout_n=0.05)
        else:
            W = self.HOL + np.array([0.058, 0.052, 0.018]) + np.array([0.10, 0.45, 0.10]) * (1 - r)
            hand = dict(hand_n=tuple(W), elbow_n=(0.3, -0.2, -1.0),
                        fdir_n=tuple(nrm([-0.60, -0.75, 0.10])), palm_n=tuple(nrm([-0.55, 0.45, 0.70])),
                        curl_n=tuple(np.array([0.30, 0.62, 0.82, 0.90]) * (1 - close) +
                                     np.array([0.62, 0.84, 0.92, 0.96]) * close),
                        thumb_n=0.45 + 0.30 * close, spread_n=0.0, thumbout_n=0.25 * (1 - close))
        p = dict(
            pelvis=(0.76, 0.42, -0.05), yaw=0.0, lean=72.0, chest=14.0, twist=0.0, neck=8.0, head=22.0, head_yaw=0.0,
            head_roll=0.0, shrug=0.1, **hand,
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
        lift = smoothstep(3102, 3135, f)
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
        near = smoothstep(3064, 3098, f) * (1 - smoothstep(3102, 3120, f)) if self.HAND else 0.0
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
        Mf = H3.material_table3()
        Mf[H3.M_GLOVE, 8] = 1.6                                  # the moon rims the leather: a glove, not a black cut-out
        res = H3.render(cam, B, Hp, L, env, XP, ENV, inscription_ins(), Mf, ss=(3 if scale > 0.75 else 2),
                        sil=dict(skin=1.0, eyes=1.0, cap=0.85, cap_brim=0.85, hair=0.85))
        depth = comp(img, res)
        self._sheen(img, B, res, t)
        if 0 <= d < 30:
            self._sparks(img, cam, f)
        if not nodof:
            img = dof(img, depth, focus, K=cam.f * 0.0035)
        return finish(img, exposure=float(os.environ.get('V3_EXPO', 1.6)))

    def _sheen(self, img, B, res, t):
        """H5: a skin of meltwater on the frozen disc. Seen this steeply it mirrors the moonlit far lip of the hollow: a
        pale cold sheen strongest along the far edge (the grazing side), fading toward us, with a slow ripple and one
        brighter line where the lip's reflection meets the water."""
        if res is None:
            return
        y0, x0, _, a_, _, grp = res
        gi = H3.group_index(B, 'ice')
        m = np.zeros(img.shape[:2], np.float32)
        m[y0:y0 + a_.shape[0], x0:x0 + a_.shape[1]] = a_ * (grp == gi)
        on = m > 0.5
        cols = np.where(on.any(0))[0]
        if len(cols) < 4:
            return
        H_ = img.shape[0]
        top = np.argmax(on, 0).astype(np.float32)
        bot = (H_ - 1 - np.argmax(on[::-1], 0)).astype(np.float32)
        ys = np.arange(H_, dtype=np.float32)[:, None]
        span = np.maximum(bot - top, 1.0)[None, :]
        u = np.clip((ys - top[None, :]) / span, 0.0, 1.0)                 # 0 at the far edge .. 1 at the near
        g = (1 - u) ** 5.0 + 0.45 * np.exp(-((u - 0.06) / 0.030) ** 2)   # the grazing gradient + the lip's line
        rng = np.random.default_rng(5)
        nz = cv2.GaussianBlur(rng.normal(0, 1, img.shape[:2]).astype(np.float32), (0, 0), img.shape[1] / 90.0)
        nz = nz / (np.abs(nz).max() + 1e-6)
        xx = np.arange(img.shape[1], dtype=np.float32)[None, :]
        rip = 1 + 0.18 * np.sin(xx * 0.045 * 960 / img.shape[1] + ys * 0.09 * 960 / img.shape[1] + t * 1.3) + 0.25 * nz
        col = np.array([0.020, 0.025, 0.034], np.float32)
        img += (m * g * rip)[..., None] * col[None, None]

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


# ================================================================ B H4 ====

class Climb:
    """B · THE CLIMB (H4) on the locked sheet, bars 9-11 (B 640-879). 640-799 is the wide: RUN-B's arête plate with her
    figure at 60 px or less, from `climb_layer()` (any camera, any place on the ridge). 800-879 (bar 11 b1) is this
    shot: closer, behind her, the red scarf and the clay pot's glow ahead of her as she trudges up a snow arête in
    the blue night; spindrift. Never her face: she walks away from the lens."""
    F0, F1 = 800, 879
    SLOPE = math.tan(math.radians(20.0))
    SPEED = 0.26                      # m/s up the crest: a slow, heavy trudge
    STEP = 1.30                       # s per stride pair
    HFOV = 36.0
    MOON = nrm([-0.45, 0.50, 0.74])   # B's moon ahead and to her left: her silhouette gets a silver edge
    WIND = np.array([-1.0, 0.1])      # blowing from her right (+x) across the crest
    REACH = 40.0
    WIDE = False

    def ground(self, x, z):
        return self.SLOPE * z - 0.9 * abs(x) ** 1.6

    def walker(self, t):
        """(pelvis xyz, pose dict, pot centre) for the trudge at time t (s)."""
        zc = self.SPEED * t
        ph = (t / self.STEP) % 1.0
        # feet: left (near, -x) plants on the half cycle, right on the whole; each lifts and swings forward
        def foot(off, side):
            q = (ph + off) % 1.0
            base = math.floor(t / self.STEP + off) * self.SPEED * self.STEP - 0.10 * side * 0
            swing = smoothstep(0.55, 1.0, q)
            z = base + self.SPEED * self.STEP * (swing - 0.5) + 0.05
            y = self.ground(0.0, z) + 0.06 + 0.08 * math.sin(math.pi * swing) * (q > 0.55)
            return np.array([0.085 * side, y, z])
        fl, fr = foot(0.0, -1.0), foot(0.5, 1.0)
        bob = 0.025 * math.cos(4 * math.pi * ph)
        pel = np.array([0.012 * math.sin(2 * math.pi * ph), self.ground(0.0, zc) + 0.90 + bob, zc])
        fwd = np.array([0.0, 0.0, 1.0])
        pot = pel + np.array([0.0, 0.30, 0.30])
        p = dict(
            pelvis=tuple(pel), yaw=-90.0, lean=16.0, chest=8.0, twist=2.0 * math.sin(2 * math.pi * ph), neck=10.0,
            head=24.0, head_yaw=0.0, head_roll=2.0 * math.sin(2 * math.pi * ph), shrug=0.5,
            hand_n=tuple(pot + np.array([-0.080, -0.030, -0.005])), hand_f=tuple(pot + np.array([0.080, -0.030, -0.005])),
            elbow_n=(-0.6, -1.0, -0.3), elbow_f=(0.6, -1.0, -0.3),
            fdir_n=tuple(nrm([0.35, -0.2, 0.9])), palm_n=(1.0, 0.1, 0.0),
            fdir_f=tuple(nrm([-0.35, -0.2, 0.9])), palm_f=(-1.0, 0.1, 0.0),
            curl_n=(0.45, 0.5, 0.55, 0.6), curl_f=(0.45, 0.5, 0.55, 0.6), thumb_n=0.3, thumb_f=0.3,
            foot_n=tuple(fl), foot_f=tuple(fr), knee_n=tuple(fwd + [0, 0.3, 0]), knee_f=tuple(fwd + [0, 0.3, 0]),
            toe_n=tuple(fwd), toe_f=tuple(fwd), sole_n=(0.0, 1.0, 0.0), sole_f=(0.0, 1.0, 0.0),
            hem=0.35, breath=math.sin(2 * math.pi * t / 2.6), tools='none', rock=None,
            expr=dict(squint=0.7), look_at=tuple(pel + np.array([0.0, 0.9, 3.0])),
        )
        return pel, p, pot

    def build(self, t, with_ridge=True):
        pel, p, pot = self.walker(t)
        B, F, Hp, anc, hair = figure(p, t, hair_dir=(0.9, -0.3), scarf=False)
        # the scarf's tail streams off to her left in the wind from the back of her neck
        sa = anc['scarf_anchor']
        # the wind comes from her right (-x): the tail streams to her left, toward the lens; a 2-D chain in x-y at a
        # z just behind her neck, like the accepted take's tail
        tail = static_chain(np.array([sa[0], sa[1]]), (0.70, -0.55), 10, 0.072, droop=0.30, sway=(0.05, 0.10),
                            seed=1.0 - 3.1 * t)
        B.group('scarf')
        hero.scarf_tail(B, tail, t, z0=sa[2] - 0.03)
        # the clay fire-pot in both hands, lid on; the ember's light leaks from under the lid and a vent
        c = pot - np.array([0.0, 0.075, 0.0])
        B.group('pot', H3.M_CLAY, disp=1, amp=0.0011, scale=55.0, band=0.005)
        B.ell(c + [0, 0.070, 0], np.array([0.079, 0.068, 0.077]))
        B.cone(c + [0, 0.112, 0], c + [0, 0.136, 0], 0.050, 0.044, k=0.014)
        B.torus(c + [0, 0.137, 0], np.eye(3), 0.041, 0.0078, 0.0088, k=0.010)
        B.group('lid', H3.M_CLAY, disp=1, amp=0.0006, scale=90.0, band=0.004)
        B.ell(c + [0, 0.150, 0], np.array([0.049, 0.0100, 0.049]))
        B.ell(c + [0, 0.162, 0], np.array([0.011, 0.009, 0.011]), k=0.008)
        glow_at = c + np.array([0.0, 0.160, 0.035])
        if with_ridge:
            # the arete: a sharp snow crest rising ahead, flanks falling away at ~40 deg
            B.group('snow', H3.M_SNOW, disp=1, amp=0.045, scale=1.1, band=0.12)
            up = nrm([0.0, 1.0, -self.SLOPE])
            for sg in (-1.0, 1.0):
                nrm_ = nrm([sg * 0.84, 1.0, -self.SLOPE])
                R = np.stack([nrm_, nrm(np.cross(nrm_, [0.0, 0.0, 1.0])), nrm(np.cross(nrm(np.cross(nrm_, [0.0, 0.0, 1.0])), nrm_))])
                B.plane(np.array([0.0, self.SLOPE * pel[2], pel[2]]), R, op=(0 if sg < 0 else 2), reach=self.REACH)
            # rocks breaking the crest ahead and on the flanks
            B.group('rocks', H3.M_FLINT, disp=1, amp=0.02, scale=3.0, band=0.05)
            rng = np.random.default_rng(31)
            for k in range(7):
                z = pel[2] + 1.2 + 1.3 * k + rng.uniform(-0.3, 0.3)
                x = rng.normal(0.0, 0.35)
                c = np.array([x, self.ground(x, z) + 0.02, z])
                Rr = H3.ring_frame(nrm([rng.normal(0, 0.3), 1.0, rng.normal(0, 0.3)]), ref=(1.0, 0.0, 0.2))
                B.box(c, np.array([0.18, 0.10, 0.14]) * rng.uniform(0.6, 1.5), R=Rr, rnd=0.05)
        return B, Hp, anc, pel, pot, glow_at

    def camera(self, f, scale):
        t = f / FPS
        pel = self.walker(t)[0]
        # close, behind her left shoulder: her back, the scarf blowing toward us, the pot's glow at her front-left
        back = 1.35 + 0.18 * (t - self.F0 / FPS)           # she draws away a little: the camera trails slower
        pos = pel + np.array([1.10, 0.38, -back * 0.85])
        tgt = pel + np.array([0.02, 0.36, 0.34])
        cam = Camera(pos, hfov=self.HFOV, scale=scale)
        yaw, pitch = cam.look_at(tgt)
        return Camera(pos, yaw=yaw, pitch=pitch, hfov=self.HFOV, scale=scale), float(np.linalg.norm(tgt - pos))

    def sky(self, cam):
        """The blue night behind her: a gradient, stars, the moon's glow; sampled per pixel from the env map."""
        H_, W_ = cam.H, cam.W
        env = night_env(moon=1.0)
        ys, xs = np.mgrid[0:H_, 0:W_]
        d = np.stack([(xs + 0.5 - cam.cx) / cam.f, -(ys + 0.5 - cam.cy) / cam.f, np.ones_like(xs, np.float64)], -1)
        d = d @ cam.R.T
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        u = ((np.arctan2(d[..., 0], d[..., 2]) / (2 * np.pi) + 0.5) * env.shape[1]).astype(int) % env.shape[1]
        v = (np.arccos(np.clip(d[..., 1], -1, 1)) / np.pi * env.shape[0]).astype(int).clip(0, env.shape[0] - 1)
        img = env[v, u] * 1.6
        rng = np.random.default_rng(77)
        n = int(2600 * cam.scale ** 2)
        sx, sy = rng.uniform(0, W_, n), rng.uniform(0, H_ * 0.75, n)
        mag = rng.random(n) ** 6
        for x, y, m in zip(sx, sy, mag):
            splat_gauss(img, float(x), float(y), max(0.5, 0.7 * cam.scale), 0.5 * m + 0.02, 0.55 * m + 0.02, 0.7 * m + 0.03)
        return img.astype(np.float32)

    def render(self, f, scale=0.5):
        t = f / FPS
        cam, focus = self.camera(f, scale)
        cam, focus, nodof = debug_cam(cam, focus, scale)
        B, Hp, anc, pel, pot, glow_at = self.build(t)
        XP = np.zeros(64)
        XP[25], XP[26] = 1.0, 1.0
        ENV = env_stack(night_env(moon=1.0))
        warm = 0.060 * (1.0 + 0.12 * fnoise1(t * 3.0, 5.0, 2))
        L = [moon_light(0.20, self.MOON),
             light(glow_at, (1.0, 0.36, 0.08), warm, 0.02, 4.0)]
        env = hero.env_vec(rim_dir=self.MOON, rim=np.array([0.14, 0.18, 0.28]), amb=np.array([0.006, 0.009, 0.016]),
                           bounce=np.array([0.012, 0.015, 0.022]), ao=0.03)
        img = self.sky(cam)
        res = H3.render(cam, B, Hp, L, env, XP, ENV, None, ss=(3 if scale > 0.75 else 2),
                        sil=dict(skin=1.0, eyes=1.0, cap=0.3, cap_brim=0.3, hair=0.3))
        depth = comp(img, res)
        fire.add_glow(img, cam, glow_at + np.array([0.0, 0.02, 0.0]), 0.12, 0.004)
        if self.WIDE:
            # at 60 px the pot is a point: its warm glow is the one warm thing in the frame
            fire.add_glow(img, cam, glow_at, 0.18, 0.30, col=(1.0, 0.42, 0.12))
        self._spindrift(img, cam, t)
        if not nodof:
            img = dof(img, depth, focus, K=cam.f * 0.004)
        return finish(img, exposure=float(os.environ.get('V3_EXPO', 1.5)))

    def _spindrift(self, img, cam, t):
        """Snow grains torn off the crest, streaking across in the wind, silver in the moon."""
        rng = np.random.default_rng(12)
        n = 900
        P0 = rng.uniform([-3.0, 0.0, -2.0], [3.0, 2.2, 6.0], (n, 3))
        ph = rng.random(n)
        pel = self.walker(t)[0]
        from core import splat_streak
        for k in range(n):
            x = (P0[k, 0] - 6.5 * t - 3.0 * ph[k]) % 6.0 - 3.0
            p = np.array([x, self.ground(0.0, pel[2] + P0[k, 2]) + P0[k, 1] * (0.3 + 0.7 * ph[k]) + 0.1, pel[2] + P0[k, 2]])
            q = p + np.array([0.20, 0.0, 0.0])
            sx, sy, z = cam.project(p)
            qx, qy, _ = cam.project(q)
            if z < 0.3:
                continue
            e = 0.9 * cam.scale ** 2 / max(z, 0.5) * (0.3 + ph[k])
            splat_streak(img, float(qx), float(qy), float(sx), float(sy), max(0.45, 0.6 * cam.scale), e * 0.55, e * 0.65, e * 0.85)


def climb_layer(cam, f, world_pos, heading_deg=0.0, glow=1.0, ss=3):
    """Her climbing figure (with the glowing pot) as an RGBA + depth layer for RUN-B's wide (B 640-799): placed at
    `world_pos` (the point on the crest under her) facing `heading_deg` (0 = +z in RUN-B's frame), in any camera.
    Lit by B's moon and the pot. Returns hsdf3.render's tuple (Y0, X0, rgb, a, depth, group)."""
    cl = Climb()
    t = f / FPS
    B, Hp, anc, pel, pot, glow_at = cl.build(t, with_ridge=False)
    a = math.radians(heading_deg)
    Ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    off = np.asarray(world_pos, np.float64) - np.array([pel[0], cl.ground(0.0, pel[2]), pel[2]])
    for g in B.groups:
        for row, bs in g['prims']:
            for sl in ((5, 8), (8, 11)) if int(row[0]) == 1 else ((5, 8),):
                pass
    # move the whole figure: world = Ry @ (p - pel_ground) + world_pos
    base = np.array([pel[0], cl.ground(0.0, pel[2]), pel[2]])
    for g in B.groups:
        for row, bs in g['prims']:
            typ = int(row[0])
            row[5:8] = Ry @ (row[5:8] - base) + world_pos
            if typ == 1:
                row[8:11] = Ry @ (row[8:11] - base) + world_pos
                if row[23] != 0.0:
                    row[11:14] = Ry @ row[11:14]
                    row[14:17] = Ry @ row[14:17]
            else:
                Rw = row[11:20].reshape(3, 3) @ Ry.T
                row[11:20] = Rw.reshape(-1)
            bs[:3] = Ry @ (bs[:3] - base) + world_pos
    Hp[40] = 0.0
    glow_w = Ry @ (glow_at - base) + world_pos
    L = [moon_light(0.55, Climb.MOON), light(glow_w, (1.0, 0.36, 0.08), 0.010 * glow, 0.02, 4.0)]
    env = hero.env_vec(rim_dir=Climb.MOON, rim=np.array([0.14, 0.18, 0.28]), amb=np.array([0.006, 0.009, 0.016]),
                       bounce=np.array([0.012, 0.015, 0.022]), ao=0.03)
    return H3.render(cam, B, Hp, L, env, ss=ss, sil=dict(skin=1.0, eyes=1.0, cap=0.3, cap_brim=0.3, hair=0.3))


def glove_hand_layer(cam, wrist, fdir, palm, curls=(0.9, 0.9, 0.9, 0.9), thumb=0.8, side='f', lights=(), env=None,
                     ss=3, forearm=0.22):
    """Her gloved hand (and sleeve cuff) alone as an RGBA + depth layer for another lane's camera (e.g. ACCORD's AC2:
    the top-down close of her fist over the Ring on the stone). side 'f' = her right hand, 'n' = her left.
    `lights` rows as hsdf3 ([x,y,z, r,g,b, radius, soft_k]). Returns hsdf3.render's tuple or None."""
    B = hero.Builder()
    a = nrm(fdir)
    W = np.asarray(wrist, np.float64)
    ha = hero.hand(B, 'hand_' + side, W + a * 0.004, a, np.asarray(palm, np.float64), -1.0 if side == 'f' else 1.0,
                   curls, thumb=thumb)
    B.group('hand_sleeves', H3.M_COAT, band=0.006)
    B.cone(W - a * forearm, W - a * 0.005, 0.050, 0.046, k=0.02)
    cf = hero.perp_frame(a)
    B.torus(W - a * 0.012, np.stack([cf[0], a, cf[1]]), 0.040, 0.009, 0.020, k=0.012)
    Hp = np.zeros(160)
    H3.gloves(B, Hp)
    if env is None:
        env = hero.env_vec()
    return H3.render(cam, B, Hp, np.asarray(lights, np.float64).reshape(-1, 8), env, ss=ss)


def inscription_ins():
    can = H3.inscription_canon()
    return can[0] if can is not None else H3.inscription_accord()[0]


class DeadEmberPOV(DeadEmber):
    POV = True
    HFOV = 40.0


class ClimbWideTest(Climb):
    """A stand-in for RUN-B's wide (B 640-799) to judge her figure at <= 60 px on a ridge: the plate is RUN-B's."""
    REACH = 400.0
    WIDE = True
    HFOV = 30.0

    def camera(self, f, scale):
        t = f / FPS
        pel = self.walker(t)[0]
        pos = pel + np.array([-38.0, 14.0, -72.0])
        tgt = pel + np.array([0.0, 2.0, 8.0])
        cam = Camera(pos, hfov=self.HFOV, scale=scale)
        yaw, pitch = cam.look_at(tgt)
        return Camera(pos, yaw=yaw, pitch=pitch, hfov=self.HFOV, scale=scale), float(np.linalg.norm(tgt - pos))


SHOTS = dict(deadember=DeadEmber, deadember_pov=DeadEmberPOV, firetest=FireTest, find=Find, climb=Climb,
             climb_wide_test=ClimbWideTest)


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
