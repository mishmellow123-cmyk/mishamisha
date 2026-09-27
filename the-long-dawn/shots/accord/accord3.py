"""THE LONG DAWN, cut C: ACCORD v3 (C frames 4480-5679) -> renders/accord_C3/f_%05d.png

  python shots/accord/accord3.py still 4960 [--scale 0.5] [--window x0,y0,w,h]   # test still -> renders/accord_C3/tests/
  python shots/accord/accord3.py range 4800 5119 --worker 0/4 [--scale 0.5]       # frames (full res unless --scale)
  python shots/accord/accord3.py sheet 4840,4960,5040 --scale 0.5 --out x.jpg     # quick labelled sheet

Plates: P1 4480-5119 (AC1 + AC4), P2 5120-5379 (AC2, 5360-5379 handle), P3 5520-5679 (C23 bar 70, 5600-5679 handle).
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault('NUMBA_NUM_THREADS', '2')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _invalidate_numba_cache():
    import glob
    import hashlib
    hsh = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(HERE, '*.py'))):
        with open(p, 'rb') as fh:
            hsh.update(fh.read())
    stamp = os.path.join(HERE, '__pycache__', 'src.sha1')
    os.makedirs(os.path.dirname(stamp), exist_ok=True)
    old = open(stamp).read() if os.path.exists(stamp) else ''
    if old != hsh.hexdigest():
        for f in glob.glob(os.path.join(HERE, '__pycache__', '*.nb[ic]')):
            try:
                os.remove(f)
            except OSError:
                pass
        with open(stamp, 'w') as fh:
            fh.write(hsh.hexdigest())


_invalidate_numba_cache()

import numpy as np  # noqa: E402
import cv2  # noqa: E402

import scene3 as SC  # noqa: E402
from scene3 import look, smooth, ramp  # noqa: E402
import geom3 as G  # noqa: E402
import shade3 as SH  # noqa: E402
import flame3 as FL3  # noqa: E402
import nbcore  # noqa: E402
import fire as FI  # noqa: E402

OUT = os.path.join(SC.ROOT, 'renders', 'accord_C')   # EDIT-v3 convention: C numbering (4480-5599; bar 70 = 5520-5599)
INSCRIPTION = os.path.join(SC.ROOT, 'assets', 'ring', 'inscription_outer.png')
_RES = {}
NOBAND = np.zeros(3)


def resources():
    if not _RES:
        _RES['stones'] = SC.stones()
        _RES['hearth'] = SC.hearth_parts()
        _RES['noise3'] = nbcore.make_noise3(64, 7)
        _RES['strip'] = inscription_mips()
        _RES['ang'] = SC.fuel_angles()
    return _RES


def inscription_mips():
    """MONTAGE-3D-2's canonical inscription (assets/ring), as a small mip chain for a ring of <= 60 px."""
    im = cv2.imread(INSCRIPTION, cv2.IMREAD_UNCHANGED)
    if im is None:
        z = np.zeros(4, np.float32)
        return z, np.array([[1, 4]], np.int64)
    a = im.astype(np.float32) / 65535.0
    h0 = 64
    w0 = int(round(a.shape[1] * h0 / a.shape[0]))
    lv = [cv2.resize(a, (w0, h0), interpolation=cv2.INTER_AREA)]
    while lv[-1].shape[0] > 4:
        lv.append(cv2.resize(lv[-1], (lv[-1].shape[1] // 2, lv[-1].shape[0] // 2), interpolation=cv2.INTER_AREA))
    flat = np.concatenate([l.ravel() for l in lv]).astype(np.float32)
    sz = np.array([[l.shape[0], l.shape[1]] for l in lv], np.int64)
    return flat, sz


# ================================================================ state ===

HAZE = 0.0025                           # smoke haze density (single scattering of every light): a clear night
_WIND = np.array([0.40, -0.16])     # a steady breeze: every flame streams the same way
TORCH_I = 12.0                      # torch flame emission (22 whited them to cotton under ACES)


def flames_and_lights(t, Fa):
    """Torch flames (FL rows) and point lights for the lit torches."""
    lit = SC.torch_lit(t)
    heads = SC.torch_heads(t, Fa)
    Fp = SC.figures(t - 1.0)
    headp = SC.torch_heads(t - 1.0, Fp)
    rng = np.random.default_rng(11)
    hf = 0.44 * rng.uniform(0.88, 1.12, SC.NEM)
    FL = np.zeros((SC.NEM, FL3.FL_N))
    L = []
    for i in range(SC.NEM):
        top = heads[i, 0:3]
        ax = heads[i, 3:6]
        base = top - 0.03 * ax / (np.linalg.norm(ax) + 1e-9)
        vel = (top - headp[i, 0:3]) * 24.0
        lean = _WIND - 0.35 * vel[:2]
        ln = np.linalg.norm(lean)
        if ln > 1.6:
            lean *= 1.6 / ln
        FL[i] = (base[0], base[1], base[2], hf[i], 0.088, lean[0], lean[1], TORCH_I, 3.7 * i + 1.3, lit[i])
        if lit[i] > 0.01:
            # the light pumps with the flame's own height (flame3.torch_flicker, same seed)
            fl = FL3.torch_flicker(3.7 * i + 1.3, float(t), resources()['noise3']) / 0.98
            I = 1.0 * lit[i] * (0.55 + 0.45 * fl)
            c = SC.FIRE_HOT
            L.append([base[0] + lean[0] * 0.08, base[1] + lean[1] * 0.08, base[2] + 0.22, 0.22,
                      I * c[0], I * c[1], I * c[2], float(i)])
    return FL, L


def hearth_state(t):
    fs = SC.fire_state(t)
    HP = np.zeros(FL3.HP_N)
    HP[FL3.HP_T] = t
    HP[FL3.HP_ON] = fs['on']
    HP[FL3.HP_H] = fs['H']
    HP[FL3.HP_WHITE] = fs['white']
    HP[FL3.HP_SPREAD] = fs['spread']
    HP[FL3.HP_HOLLOW] = fs['hollow']
    HP[FL3.HP_CONV] = fs['conv']
    HP[FL3.HP_CALM] = fs['calm']
    f = SC.hand_fist_pos(max(t, SC.HAND_CLOSE))
    HP[FL3.HP_FX:FL3.HP_FZ + 1] = f
    HP[FL3.HP_I] = 16.0 * fs['on'] * (1.0 - 0.35 * fs['calm'])
    CF = np.zeros((1, FL3.CF_N))
    if fs.get('p3', 0.0) > 0.0:
        CF, nmain = SC.calm_flames(t)
        HP[FL3.HP_P3] = 2.0
        HP[FL3.HP_NMAIN] = nmain
        HP[FL3.HP_I] = CALM_I
    elif fs['on'] > 0.0:
        # AC2: the crown of flames round her fist (the same flame machinery as bar 70)
        CF, nmain = SC.p2_flames(t)
        HP[FL3.HP_P3] = 1.0
        HP[FL3.HP_NMAIN] = 0
        HP[FL3.HP_I] = P2_I * fs['on']
    return fs, HP, CF


CALM_I = 11.0           # bar 70: the fire that remains (emission scale of flame3.calm_density)
P2_I = 12.0             # AC2: the fire everyone lit


def calm_lights(t, fs, CF):
    """Bar 70: the light of the fire that remains: a warm cluster over the coals on the stone (its heart pumping
    with the tall flame, <= ~4 % and slow, so the council breathes with it) and a low ring from the burning logs."""
    L = []
    wh = fs['white']
    col = np.array([1.0, 0.50, 0.14]) * (1 - wh) + np.array([1.0, 0.92, 0.8]) * wh
    boost = 1.0 + 5.0 * wh
    br = 1.0 + 0.025 * math.sin(t * 0.55) + 0.015 * math.sin(t * 1.3 + 1.1)
    zt = G.STONE_TOP
    I0 = CALM_LIGHT * boost * br
    # in the body of the flames (never a hand's breadth over the coals: that blew the stone's top out)
    for k in range(5):
        a = 2 * math.pi * k / 5 + 0.3
        I = I0 * 0.13
        L.append([0.13 * math.cos(a), 0.13 * math.sin(a), zt + 0.36 + 0.10 * (k % 2), 0.30,
                  I * col[0], I * col[1], I * col[2], -1.0])
    I = I0 * 0.30
    L.append([0.0, 0.0, zt + 0.55, 0.32, I * col[0], I * col[1], I * col[2], -1.0])
    # the log ring: low, redder
    cl = np.array([1.0, 0.42, 0.10])
    for k in range(6):
        a = 2 * math.pi * k / 6 + 0.1
        I = I0 * 0.035
        L.append([0.58 * math.cos(a), 0.58 * math.sin(a), 0.16, 0.15, I * cl[0], I * cl[1], I * cl[2], -1.0])
    return L


CALM_LIGHT = 9.0


def hearth_lights(t, fs):
    L = []
    if fs['on'] <= 0.0:
        return L
    if fs.get('p3', 0.0) > 0.0:
        return calm_lights(t, fs, None)
    ang = resources()['ang']
    n = 8
    wh = fs['white']
    col = np.array([1.0, 0.56, 0.17]) * (1 - wh) + np.array([1.0, 0.92, 0.8]) * wh
    base = 2.2 * fs['on'] * min(fs['H'] / 1.0, 1.3) * (1.0 + 5.0 * wh)
    for k in range(n):
        a = 2 * math.pi * k / n + 0.2
        # only the arcs that have caught give light
        dd = np.min(np.abs((a - ang + np.pi) % (2 * np.pi) - np.pi))
        on = smooth(ramp(fs['spread'] * (np.pi / len(ang)) * 1.3 + 0.1 - dd, -0.05, 0.1))
        on = max(on, fs['calm'])
        fl = 1.0 + 0.08 * math.sin(t * 0.8 + 1.7 * k) + 0.05 * math.sin(t * 1.9 + k)
        r = 0.55 * (1 - fs['calm']) + 0.25 * fs['calm']
        z = 0.25 + 0.2 * fs['H']
        I = base * on * fl / n * 1.6
        L.append([r * math.cos(a), r * math.sin(a), z, 0.22, I * col[0], I * col[1], I * col[2], -1.0])
    I = base * 0.55 * (1.0 - 0.5 * fs['calm'])
    L.append([0.0, 0.0, 0.35 + 0.45 * fs['H'], 0.3, I * col[0], I * col[1], I * col[2], -1.0])
    return L


def occluders(t, Fa, hands):
    oc = []
    for i in range(SC.NFIG):
        x0, y0 = Fa[i, G.F_X], Fa[i, G.F_Y]
        kneel = Fa[i, G.F_KNEEL]
        lean = Fa[i, G.F_LEAN]
        hs = SC.FIG_HS[i]
        zsh = 1.33 * hs - 0.43 * hs * kneel
        head_l = np.array([0.35 * math.sin(lean) * zsh * 0.8, 0.0, zsh + 0.18 * hs - 0.3 * (1 - math.cos(lean))])
        hw = SC.to_world(i, head_l, pos=(x0, y0), ang=Fa[i, G.F_ANG])
        r = 0.20 * SC.FIG_WS[i]
        oc.append([x0, y0, 0.15, hw[0], hw[1], hw[2] - 0.05, r, float(i)])
    S = resources()['stones']
    for j in range(S.shape[0]):
        oc.append([S[j, G.S_X], S[j, G.S_Y], 0.2, S[j, G.S_X], S[j, G.S_Y], S[j, G.S_H] - 0.4,
                   0.8 * max(S[j, G.S_HX], S[j, G.S_HY]), -2.0])
    for (HD, J) in hands:
        w = HD[0:3]
        a = HD[3:6]
        oc.append([*(w - a * 0.25), *(w + a * 0.08), 0.042, -3.0])
    return np.array(oc, np.float64)


def build(t, cam=None, scale=1.0):
    R = resources()
    Fa = SC.figures(t)
    FLm, Lt = flames_and_lights(t, Fa)
    fs, HP, CF = hearth_state(t)
    Lt += hearth_lights(t, fs)
    hands = []
    gh = SC.gilded_hand(t, Fa)
    if gh is not None:
        hands.append(gh)
    hh = SC.her_hand(t)
    if hh is not None:
        hands.append(hh)
    nh = len(hands)
    HDs = np.zeros((max(nh, 1), G.HD_N))
    Js = np.zeros((max(nh, 1), 5, 4, 3))
    HBB = np.zeros((max(nh, 1), 6))
    for k, (HD, J) in enumerate(hands):
        HDs[k] = HD
        Js[k] = J
        bb = SC.hand_bounds(HD, J)
        HBB[k] = bb
    OC = occluders(t, Fa, hands)
    LT = np.array(Lt, np.float64) if Lt else np.zeros((1, 8))
    PR = np.zeros(SH.P_NPARAM)
    PR[SH.P_T] = t
    el, az = math.radians(38.0), math.radians(222.0)
    PR[SH.P_MDX], PR[SH.P_MDY], PR[SH.P_MDZ] = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
    PR[SH.P_MI] = 0.10
    PR[SH.P_MCR:SH.P_MCB + 1] = SC.MOON
    PR[SH.P_SKI] = 1.0
    PR[SH.P_SKR:SH.P_SKB + 1] = SC.NIGHT_MID
    PR[SH.P_NL] = len(Lt)
    PR[SH.P_OWNK] = 0.35
    burn = smooth(ramp(t, SC.FIRE_CATCH, SC.FIRE_CATCH + 30)) if t >= SC.FIRE_CATCH else 0.0
    PR[SH.P_EMB] = burn * fs['on']
    PR[SH.P_EMBW] = fs['white']
    PR[SH.P_ASHG] = fs['spread']
    PR[SH.P_CROWD] = 0.9 * (1.0 - 0.8 * fs['white'])
    PR[SH.P_WI] = 0.0
    PR[SH.P_IGC_X0], PR[SH.P_IGC_CELL], PR[SH.P_IGF_X0], PR[SH.P_IGF_CELL] = -500.0, 250.0, -40.0, 20.0
    PR[SH.P_SHEEN] = 2.0
    PR[SH.P_FIGRIM] = 0.35 * fs['on'] * min(fs['H'], 1.0) * (1.0 + 2.0 * fs['white'])
    PR[SH.P_FIRE_Z] = 0.35 + 0.4 * fs['H']
    PR[SH.P_NHAND] = nh
    RP = SC.ring_state(t)
    PR[SH.P_RGLOW] = RP[7]
    PR[SH.P_NOC] = OC.shape[0]
    PR[SH.P_FIRE_I] = 0.35 * fs['on'] * min(fs['H'], 1.0) * (1.0 + 3.0 * fs['white'])
    PR[SH.P_COAL] = fs.get('p3', 0.0)
    return dict(Fa=Fa, FL=FLm, HP=HP, CF=CF, fs=fs, LT=LT, OC=OC, PR=PR, HDs=HDs, Js=Js, HBB=HBB, RP=RP)


# =============================================================== render ===

FIRE_LC = 1.9      # flame radiance knee after exposure: brighter than this rolls off without losing its hue


def fire_clip(F, lc):
    """Hue-preserving soft clip of a flame's own light (before the tonemap): ACES whites any colour pushed far past
    1, which turned every flame to cotton. Each pixel keeps its hue while its brightest channel rolls off toward
    lc (tanh); the white heart stays only where the flame truly is white-hot."""
    L = F.max(axis=2, keepdims=True)
    k = np.where(L > 1e-6, lc * np.tanh(L / lc) / np.maximum(L, 1e-6), 1.0)
    return (F * k).astype(np.float32)


def edge_mask(oid, rgb):
    m = np.zeros(oid.shape, bool)
    d = oid[:, 1:] != oid[:, :-1]
    m[:, 1:] |= d
    m[:, :-1] |= d
    d = oid[1:, :] != oid[:-1, :]
    m[1:, :] |= d
    m[:-1, :] |= d
    L = np.log(rgb.max(axis=2) + 1e-3)
    c = np.zeros(oid.shape, bool)
    dx = np.abs(L[:, 1:] - L[:, :-1]) > 0.25
    c[:, 1:] |= dx
    c[:, :-1] |= dx
    dy = np.abs(L[1:, :] - L[:-1, :]) > 0.25
    c[1:, :] |= dy
    c[:-1, :] |= dy
    obj = (oid >= 5)
    m |= c & obj
    m |= (oid == 5) | (oid >= 200)
    return m


def camera(t, scale, window=None):
    cam = SC.camera(t, scale)
    if window is not None:
        cam = cam.copy()
        cam[13] -= window[0] * scale
        cam[14] -= window[1] * scale
    return cam


def frame_size(scale, window=None):
    if window is not None:
        return int(round(window[2] * scale)), int(round(window[3] * scale))
    return int(round(SC.W * scale)), int(round(SC.H * scale))


def heat_haze(img, t, cam, scale, fs, window=None):
    if fs['on'] <= 0.0:
        return img
    Hd, Wd = img.shape[:2]
    (cx, cy), zc = SC.project(cam, np.array([0.0, 0.0, 0.3 + 0.5 * fs['H']]))
    rad = cam[12] * (0.9 + 0.4 * fs['H']) / max(zc, 0.3)
    ph = t * 0.35
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(rad, 1.0)
    fall = np.clip(1.0 - rr, 0, 1) ** 1.5
    if fall.max() <= 0:
        return img
    ox = window[0] * scale if window is not None else 0.0
    oy = window[1] * scale if window is not None else 0.0
    X = xx + ox
    Y = yy + oy
    amp = 2.0 * scale * fs['on'] * (1.0 + fs['white'])
    dx = amp * fall * (np.sin(X * 0.045 / scale + ph * 3.1) * np.cos(Y * 0.037 / scale - ph * 2.3)
                       + 0.5 * np.sin((X + Y) * 0.09 / scale + ph * 4.7))
    dy = amp * fall * (np.cos(X * 0.041 / scale - ph * 2.7) * np.sin(Y * 0.052 / scale + ph * 3.3)
                       + 0.5 * np.cos((X - Y) * 0.08 / scale - ph * 5.1))
    return cv2.remap(img, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def focus(t):
    """(focus distance from the camera in m, aperture k) per plate."""
    cam = SC.camera(t, 1.0)
    p = SC.plate(t)
    if p == 1:
        tgt = np.array([0.0, 0.0, 0.40])
        g = SC.figures(t)
        gi = SC.GILDED
        hand = SC.to_world(gi, g[gi, G.F_HX:G.F_HZ + 1], pos=g[gi, G.F_X:G.F_Y + 1], ang=g[gi, G.F_ANG])
        k = smooth(ramp(t, 5000, 5040))
        P = tgt * (1 - k) + hand * k
        amt = smooth(ramp(t, 4700, 4800)) * (0.6 + 0.4 * smooth(ramp(t, 4960, 5040)))
    elif p == 2:
        P = SC.hand_fist_pos(max(t, SC.HAND_CLOSE)) if t > 5150 else SC.ring_rest()
        amt = 1.0
    else:
        P = np.array([0.0, 0.0, 0.40])
        amt = 1.0 - smooth(ramp(t, 5575, 5610))
    zc = float((P - cam[0:3]) @ cam[9:12])
    return max(zc, 0.2), amt


def dof(rgb, depth, cam, t, scale, window=None):
    """Layered defocus from the depth buffer about the focus plane (a real lens: near and far)."""
    zf, amt = focus(t)
    if amt <= 0.01:
        return rgb
    Hd, Wd = depth.shape
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
    f, cx, cy = cam[12], cam[13], cam[14]
    sx = (xx + 0.5 - cx) / f
    sy = -(yy + 0.5 - cy) / f
    # ray length -> camera z
    ln = np.sqrt(1.0 + sx * sx + sy * sy)
    zc = depth / ln
    # aperture: a 50 mm-ish lens at f/2.8, scaled to the scene; in pixels at this scale
    A = 0.010 * amt
    coc = np.abs(A * f * (1.0 / np.maximum(zc, 0.05) - 1.0 / zf))
    coc = np.minimum(coc, 16.0 * scale)
    sig = np.array([1.2, 2.8, 5.5, 10.0]) * scale
    lv = np.interp(coc, [0.0, *sig], [0.0, 1.0, 2.0, 3.0, 4.0])
    out_c = np.zeros_like(rgb)
    out_a = np.zeros(depth.shape, np.float32)
    for k in range(4):
        w = np.clip(1.0 - np.abs(lv - (k + 1)), 0.0, 1.0).astype(np.float32)
        if k == 3:
            w = np.clip(lv - 3.0, 0.0, 1.0).astype(np.float32)
        if w.max() <= 0.0:
            continue
        out_c += cv2.GaussianBlur(rgb * w[..., None], (0, 0), float(sig[k]))
        out_a += cv2.GaussianBlur(w, (0, 0), float(sig[k]))
    a_ = np.clip(out_a, 0.0, 1.0)
    near = np.clip(lv, 0.0, 1.0).astype(np.float32)
    keep = 1.0 - near
    bgn = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), float(sig[2]))
    bgd = cv2.GaussianBlur(keep, (0, 0), float(sig[2]))[..., None]
    bg = rgb * keep[..., None] + (bgn / np.maximum(bgd, 1e-3)) * near[..., None]
    return (out_c + bg * (1.0 - a_[..., None])).astype(np.float32)


def render_frame(t, scale=1.0, aa=True, mb=True, window=None):
    R = resources()
    Wd, Hd = frame_size(scale, window)
    cam = camera(t, scale, window)
    st = build(t, cam, scale)
    Fa = st['Fa']
    S = R['stones']
    KB, LG, CH = R['hearth']
    stex, ssz = R['strip']
    igz = np.zeros((4, 4), np.float32)
    rgb = np.zeros((Hd, Wd, 3), np.float32)
    depth = np.zeros((Hd, Wd), np.float32)
    oid = np.zeros((Hd, Wd), np.int32)
    dummy = np.zeros((1, 1), np.bool_)
    args = (st['PR'], st['LT'], st['OC'], Fa, Fa.shape[0], S, S.shape[0], KB, LG, LG.shape[0], CH, CH.shape[0],
            st['HDs'], st['Js'], st['HBB'], st['RP'], stex, ssz, igz, igz)
    SH.render_surfaces(Wd, Hd, cam, *args, rgb, depth, oid, False, dummy, 1)
    if aa:
        m = edge_mask(oid, rgb)
        SH.render_surfaces(Wd, Hd, cam, *args, rgb, depth, oid, True, m, 4)
    fs = st['fs']
    # the hearth fire
    if fs['on'] > 0.0:
        fb = np.zeros_like(rgb)
        FL3.hearth_volume(Wd, Hd, cam, st['HP'], R['ang'], R['ang'].shape[0], R['noise3'], depth, fb,
                          int(28 + 14 * min(scale, 1.0)), st['CF'], st['CF'].shape[0])
        rgb = rgb * 1.0
        rgb += fire_clip(cv2.GaussianBlur(fb, (0, 0), 0.7 * max(scale, 0.5)), FIRE_LC / max(SC.exposure(t), 0.3))
    rgb = heat_haze(rgb, t, cam, scale, fs, window)
    # the firelight in the thin smoke haze over the council
    LTa = st['LT']
    # (bar 70: a clear night round a steady fire, only a breath of smoke)
    hz = HAZE * (0.35 if SC.plate(t) == 3 else 1.0) * (1.0 - 0.7 * smooth(ramp(t, SC.GILT_FIND0, SC.GILT_FIND0 + 30)) * (t < SC.P2[0]))
    FL3.airlight(rgb, depth, cam, LTa, int(st['PR'][SH.P_NL]), hz, 0.06, 0.35)
    # the torch flames (after the haze so they stay crisp)
    FLm = st['FL']
    tb = np.zeros_like(rgb)
    FL3.torch_flames(tb, depth, cam, FLm, FLm.shape[0], float(t), R['noise3'], 0.0)
    rgb += fire_clip(tb, FIRE_LC / max(SC.exposure(t), 0.3))
    # sparks
    SP = FL3.sparks(t, fs)
    if SP.shape[0]:
        FI.splat_streaks(rgb, depth, cam, SP, SP.shape[0], 0.0, 10.0, NOBAND, 30.0 * scale)
    rgb = dof(rgb, depth, cam, t, scale, window)
    if mb:
        camA = camera(t - 0.25, scale, window)
        camB = camera(t + 0.25, scale, window)
        out = np.empty_like(rgb)
        FI.motion_blur(rgb, out, depth, cam, camA, camB, 16, 60.0 * scale)
        rgb = out
    rgb = ring_glint(rgb, st, cam, depth, t, scale)
    # the white: the fire's heart swallows the frame (made of light, not a fade)
    wl = SC.white_level(t)
    if wl > 0.0:
        (cx, cy), zc = SC.project(cam, np.array([0.0, 0.0, 0.6]))
        yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (Wd * 0.55)
        g = np.clip(1.0 - rr, 0, 1) ** 0.7
        k = wl * (0.35 + 0.65 * g) * 60.0
        rgb = rgb + (k[..., None] * np.array([1.0, 0.93, 0.80], np.float32) * wl)
    return rgb, dict(depth=depth, oid=oid, cam=cam, st=st)


GLINT_K = 45.0       # the Ring's glint: energy per unit of torch irradiance at the band (full-res pixels)


def ring_glint(rgb, st, cam, depth, t, scale):
    """The Ring on the stone is ~6 px across in P1: too small for its highlights to survive sampling, so its
    glint is drawn here. A smooth rounded band has a specular point for every light that reaches it, each as
    bright as that light's irradiance at the band: the sum over the torches (and the fire), breathing with their
    flicker, gold-tinted, splatted as a tight gaussian after the lens so it stays crisp, hidden by anything in
    front of it (her hand while she lays it down)."""
    RP = st['RP']
    if RP[0] <= 0.0 or SC.plate(t) != 1:
        return rgb
    c = RP[1:4]
    (gx, gy), zc = SC.project(cam, c)
    Hd, Wd = rgb.shape[:2]
    if zc <= 0.05 or not (-4 <= gx < Wd + 4 and -4 <= gy < Hd + 4):
        return rgb
    ix, iy = int(min(max(gx, 0), Wd - 1)), int(min(max(gy, 0), Hd - 1))
    dist = float(np.linalg.norm(c - cam[0:3]))
    # occluded where the depth buffer holds something clearly nearer than the band
    y0, y1 = max(iy - 1, 0), min(iy + 2, Hd)
    x0, x1 = max(ix - 1, 0), min(ix + 2, Wd)
    vis = float(np.mean(depth[y0:y1, x0:x1] > dist - 0.03))
    if vis <= 0.0:
        return rgb
    LT = st['LT']
    E = 0.0
    for k in range(LT.shape[0]):
        d2 = float(np.sum((LT[k, 0:3] - c) ** 2))
        E += float(max(LT[k, 4], LT[k, 5], LT[k, 6])) / (d2 + LT[k, 3] ** 2 + 0.02)
    # it only glints once it is down and her hand has come away from it
    on = smooth(ramp(t, SC.RING_SET + 2, SC.RING_SET + 9))
    energy = GLINT_K * E * on * vis * (7.6 / max(dist, 0.5)) ** 2 * scale * scale
    if energy <= 1e-4:
        return rgb
    # a tight hot core and a soft gold halo round it (the core whites under the tonemap; the halo stays gold)
    for sig, frac, col in ((max(1.0 * scale, 0.6), 0.80, np.array([1.0, 0.82, 0.46], np.float32)),
                           (max(4.5 * scale, 1.5), 0.20, np.array([1.0, 0.66, 0.24], np.float32))):
        r = int(math.ceil(3 * sig)) + 1
        ys, xs = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
        g = np.exp(-((xs + 0.5 - gx) ** 2 + (ys + 0.5 - gy) ** 2) / (2 * sig * sig)) / (2 * math.pi * sig * sig)
        ok = (ys >= 0) & (ys < Hd) & (xs >= 0) & (xs < Wd)
        rgb[ys[ok], xs[ok]] += (frac * energy * g[ok])[:, None].astype(np.float32) * col
    return rgb


def grade(img, t):
    """The night piece: gold where firelight falls, umber-black shadows, blue only in the far dark."""
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sh = np.clip(1.0 - lum / 0.25, 0.0, 1.0)[..., None] ** 2
    umber = np.array([0.020, 0.014, 0.010], np.float32)
    out = img * (1.0 - 0.18 * sh) + umber * 0.18 * sh
    return np.clip(out, 0.0, 1.0)


def finish(hdr, t):
    img = look.finish(hdr, exposure=SC.exposure(t), bloom_strength=0.07, bloom_threshold=0.9, vignette_amount=0.22)
    return grade(img, t)


def still(t, scale, tag='', window=None):
    t0 = time.time()
    hdr, info = render_frame(t, scale, window=window)
    t1 = time.time()
    img = finish(hdr, t)
    tests = os.path.join(OUT, 'tests')
    os.makedirs(tests, exist_ok=True)
    w = '' if window is None else '_w' + '-'.join(str(int(v)) for v in window)
    p = os.path.join(tests, f'still_{int(t)}{tag}{w}_s{scale}.png')
    look.save_png(p, img)
    print(f'{p}  render {t1 - t0:.1f}s', flush=True)
    return p


def sheet(frames, scale, out, cols=2, tag=''):
    ims = []
    for f in frames:
        t0 = time.time()
        hdr, info = render_frame(float(f), scale)
        img = finish(hdr, float(f))
        im8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
        cv2.putText(im8, f'{f}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        ims.append(im8)
        print(f'frame {f} {time.time() - t0:.1f}s', flush=True)
    rows = []
    for r in range(0, len(ims), cols):
        row = ims[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(ims[0]))
        rows.append(np.concatenate(row, 1))
    grid = np.concatenate(rows, 0)
    cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['still', 'range', 'sheet'])
    ap.add_argument('a', nargs='?')
    ap.add_argument('b', nargs='?', type=float)
    ap.add_argument('--scale', type=float, default=None)
    ap.add_argument('--window', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--worker', default='0/1')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--cols', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    if args.cmd == 'still':
        win = tuple(float(v) for v in args.window.split(',')) if args.window else None
        still(float(args.a), args.scale if args.scale is not None else (1.0 if win else 0.5), args.tag, win)
    elif args.cmd == 'sheet':
        fr = [int(v) for v in args.a.split(',')]
        sheet(fr, args.scale or 0.5, args.out or os.path.join(OUT, 'tests', 'sheet.jpg'), args.cols)
    else:
        cv2.setNumThreads(1)
        k, n = [int(v) for v in args.worker.split('/')]
        if args.frames:
            frames = []
            for part in args.frames.split(','):
                if '-' in part:
                    a_, b_ = part.split('-')
                    frames += list(range(int(a_), int(b_) + 1))
                else:
                    frames.append(int(part))
        else:
            frames = list(range(int(float(args.a)), int(args.b) + 1))
        frames = frames[k::n]
        scale = args.scale if args.scale is not None else 1.0
        dest = args.outdir or (OUT if scale == 1.0 else os.path.join(OUT, 'draft'))
        dest = dest if os.path.isabs(dest) else os.path.join(SC.ROOT, dest)
        os.makedirs(dest, exist_ok=True)
        for fr in frames:
            p = look.frame_path(dest, fr)
            if (os.path.exists(p) or os.path.exists(p[:-4] + '.jpg')) and not args.force:
                continue
            t0 = time.time()
            hdr, info = render_frame(float(fr), scale)
            img = finish(hdr, float(fr))
            look.save_png(p, img)
            print(f'frame {fr} {time.time() - t0:.1f}s', flush=True)
