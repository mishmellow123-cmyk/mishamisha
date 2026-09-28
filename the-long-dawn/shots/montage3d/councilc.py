"""C19-C23 THE COUNCIL, rebuilt in Cycles (lane COUNCIL-C). C frames 4464-5679, the farm's h100-1 GPUs.

  farm look-dev:  python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/councilC_look.json --test 2
  local (tiny):   MT3D_ENGINE=CYCLES python3 council_local.py councilc --frames 5160 --scale 0.25 --samples 16 --out t_c0

ACCORD's staging is kept exactly: `prep` (venv) runs ACCORD-3's scene3 (cameras, the council's poses, the torches, her
hand, the Ring, the fire) and ACCORD-CROWD's crowd3 (the crowd, the walk-out) for the frames being rendered and writes
their state; Blender (`build`, `per_frame`) rebuilds the LOOK: heavy wool cloaks and deep hoods (council_geo.py),
leather gloves (glove.py), torches of pitch-soaked wound cloth with volumetric flames, a weathered natural slab with
soot and lichen, alligator-charred split logs, trampled heather and earth, standing stones.
H5: silhouettes and gloved hands only (the hood holds a void, never a face); no regional dress; nothing real.
"""
import json
import math
import os
import sys

try:
    import numpy as np
except ImportError:              # Blender's own Python on this Mac cannot load numpy: the Blender side never needs it
    np = None

FPS = 24.0
SAMPLES = 128
PREP_ONLY_FRAMES = True
START, END = 4464, 5679
HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.path.abspath(os.path.join(HERE, '..', 'accord'))
FINISH = dict(exposure=1.0, bloom_strength=0.045, bloom_threshold=1.4, streak_strength=0.0, vignette_amount=0.22,
              lift=0.003)
EXPO = float(os.environ.get('COUNCIL_EXPO', '1.25'))
WIND = (0.40, -0.16)
TORCH_W = float(os.environ.get('COUNCIL_TORCH_W', '6.0'))        # a torch flame's light (W, before flicker)
FIRE_W = float(os.environ.get('COUNCIL_FIRE_W', '90.0'))         # the hearth fire's light per m^2 of flame
FLAME_E = float(os.environ.get('COUNCIL_FLAME_E', '60.0'))        # flame volume emission scale
CLIP_LC = float(os.environ.get('COUNCIL_CLIP', '1.8'))
GLINT_W = float(os.environ.get('COUNCIL_GLINT_W', '2.0'))         # the Ring's own glint light (lights only the Ring)            # hue-preserving roll-off (after exposure)


def flame_specs():
    return []


def timing(frames):
    return {}


def _sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _smr(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def _ramp(t, a, b):
    return min(max((t - a) / (b - a), 0.0), 1.0)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


# ======================================================== venv: ACCORD's staging ===

def _acc():
    if ACC not in sys.path:
        sys.path.insert(0, ACC)
    import geom3 as G
    import scene3 as SC
    return SC, G


def _crowd():
    d = os.path.join(ACC, 'crowd')
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import crowd3
        return crowd3
    except Exception as e:                                          # the council renders without the crowd lane
        print('councilc: crowd3 unavailable:', repr(e)[:300], flush=True)
        return None


def _hammer_hd(SC, G, Fa, i, scale, gilt):
    """The hand frame of emissary i round its torch (hammer grip), as scene3.gilded_hand builds the gilded one."""
    hl = Fa[i, G.F_HX:G.F_HZ + 1]
    tl = Fa[i, G.F_TX:G.F_TZ + 1]
    sl = Fa[i, G.F_SHX:G.F_SHZ + 1]
    ang = Fa[i, G.F_ANG]
    pos = Fa[i, G.F_X:G.F_Y + 1]
    hw = SC.to_world(i, hl, pos=pos, ang=ang)
    tw = SC.to_world(i, hl + tl, pos=pos, ang=ang) - hw
    sw = SC.to_world(i, sl, pos=pos, ang=ang)
    fwd = _unit(hw - sw)
    a = _unit(fwd - tw * (fwd @ tw))
    left = SC.FIG_SIDE[i] > 0
    c_hint = np.cross(a, _unit(tw)) * (-1.0 if left else 1.0)
    HD = SC.hand_frame(np.zeros(3), a, c_hint, scale=scale, forearm=0.22, gilt=gilt, sleeve=1.0, left=left)
    HD[0:3] = hw - a * 0.090 * scale + HD[9:12] * 0.026 * scale
    return HD, left


def _focus(SC, G, t, cam):
    p = SC.plate(t)
    if p == 1:
        tgt = np.array([0.0, 0.0, 0.40])
        g = SC.figures(t)
        gi = SC.GILDED
        hand = SC.to_world(gi, g[gi, G.F_HX:G.F_HZ + 1], pos=g[gi, G.F_X:G.F_Y + 1], ang=g[gi, G.F_ANG])
        k = _sm(_ramp(t, 5000, 5040))
        P = tgt * (1 - k) + hand * k
        amt = _sm(_ramp(t, 4700, 4800)) * (0.6 + 0.4 * _sm(_ramp(t, 4960, 5040)))
    elif p == 2:
        P = SC.hand_fist_pos(max(t, SC.HAND_CLOSE)) if t > 5150 else SC.ring_rest()
        amt = 1.0
    else:
        P = np.array([0.0, 0.0, 0.40])
        amt = 1.0 - _sm(_ramp(t, 5575, 5610))
    zc = float((P - cam[0:3]) @ cam[9:12])
    return max(zc, 0.2), amt


def _her_pose(SC, t):
    """her_hand's finger curls (scene3 units) for the glove rig."""
    u = _smr(_ramp(t, 5134, 5156))
    relax = np.array([0.20, 0.29, 0.37, 0.46]) + 0.06 * u
    cl = _smr(_ramp(t, SC.HAND_CLOSE - 3, SC.HAND_CLOSE + 6))
    curl = relax + (np.array([0.90, 0.95, 0.98, 1.0]) - relax) * cl
    thumb = 0.28 + 0.67 * _smr(_ramp(t, SC.HAND_CLOSE - 2, SC.HAND_CLOSE + 7))
    op = _smr(_ramp(t, SC.FINGERS_OPEN, SC.FINGERS_OPEN + 5))
    curl = curl * (1 - op) + (-0.05) * op
    thumb = thumb * (1 - op) + 0.1 * op
    return curl, thumb, 0.85 + 0.9 * op


def _flicker(seed, t):
    Ts = t / 24.0
    return (1.0 + 0.10 * math.sin(2 * math.pi * 7.3 * Ts + seed) + 0.07 * math.sin(2 * math.pi * 4.1 * Ts + 2.3 * seed)
            + 0.05 * math.sin(2 * math.pi * 11.7 * Ts + 0.7 * seed) + 0.04 * math.sin(2 * math.pi * 1.3 * Ts + 5.1 * seed))


def _p2_rows(SC, G, t):
    """scene3.p2_flames with FIXED slots (catch 2k, runner 2k+1; H 0 while unlit) and fixed per-slot random factors,
    so no flame ever changes identity or height when another one catches (ACCORD's list grows as they catch)."""
    fs = SC.fire_state(t)
    rows = np.zeros((2 * SC.NEM, 8))
    if fs['on'] <= 0.0:
        return rows
    fist = SC.hand_fist_pos(max(t, SC.HAND_CLOSE))
    fx, fy = fist[0], fist[1]
    Ts = t / 24.0
    grow = fs['H']
    rng = np.random.default_rng(66)
    mc = rng.uniform(0.85, 1.15, SC.NEM)
    mr = rng.uniform(0.88, 1.12, SC.NEM)

    def pump(seed, big):
        return (1.0 + big * (0.085 * math.sin(2 * math.pi * 3.1 * Ts + seed) + 0.060 * math.sin(2 * math.pi * 1.37 * Ts + 2.1 * seed)
                             + 0.045 * math.sin(2 * math.pi * 0.53 * Ts + 3.3 * seed)))

    for k in range(SC.NEM):
        a = SC.FIG_PSI[k]
        age = t - (SC.FIRE_CATCH + 0.5 * SC.FIG_DOWN_DT[k])
        g = SC.ease_out(min(max(age / 10.0, 0.0), 1.0), 2.0)
        sd = 3.1 + 1.9 * k
        x, y = 0.62 * math.cos(a), 0.62 * math.sin(a)
        dx, dy = fx - x, fy - y
        D = math.hypot(dx, dy) + 1e-9
        if g > 0.0:
            Hh = (0.24 + 0.12 * min(grow, 1.3)) * (0.4 + 0.6 * g) * pump(sd, 1.0) * mc[k]
            kk = 0.45 * D / max(Hh, 0.15)
            rows[2 * k] = (x, y, 0.03, Hh * min(1.0, g * 4.0), 0.085, kk * dx / D + 0.08, kk * dy / D - 0.03, sd)
        u = _smr(min(max((age - 2.0) / 20.0, 0.0), 1.0))
        if u > 0.0:
            ca, sa = dx / D, dy / D
            ex, ey = fx - 0.155 * ca, fy - 0.155 * sa
            z0 = 0.03 + (G.STONE_TOP - 0.045) * _sm(min(max((u - 0.45) / 0.4, 0.0), 1.0))
            Hc = (0.20 + 1.00 * u * max(grow - 0.25, 0.0)) * pump(sd + 7.0, 1.1) * mr[k]
            li = 0.10 + 0.10 * u
            rows[2 * k + 1] = (x + (ex - x) * u, y + (ey - y) * u, z0, Hc * min(1.0, u * 6.0), 0.075 + 0.045 * u,
                               li * ca + 0.05, li * sa - 0.02, sd + 19.0)
    return rows


def _torch_world(SC, G, Fa, i):
    pos = Fa[i, G.F_X:G.F_Y + 1]
    ang = Fa[i, G.F_ANG]
    hl = Fa[i, G.F_HX:G.F_HZ + 1]
    tl = Fa[i, G.F_TX:G.F_TZ + 1]
    hw = SC.to_world(i, hl, pos=pos, ang=ang)
    ax = SC.to_world(i, hl + tl, pos=pos, ang=ang) - hw
    return hw, _unit(ax)


def _frame_state(SC, G, CR, t, hf, dz, crowd_keep):
    cam = SC.camera(t, 1.0)
    Fa = SC.figures(t)
    Fp = SC.figures(t - 1.0)
    H = SC.HER
    # her second arm always exists (tucked at her breast when she is not reaching): a fixed mesh topology
    if Fa[H, G.F_ARM2] < 0.5:
        st = SC.her_state(t)
        side = SC.FIG_SIDE[H]
        sh2 = SC.shoulder_local(H, -side, st['kneel'], st['lean'])
        E2, H2 = SC._two_bone(sh2, np.array([0.12, 0.02, 0.86 * SC.FIG_HS[H]]), 0.28, 0.27,
                              np.array([-0.3, -side * 1.0, -0.8]))
        Fa[H, G.F_S2X:G.F_S2Z + 1] = sh2
        Fa[H, G.F_E2X:G.F_E2Z + 1] = E2
        Fa[H, G.F_H2X:G.F_H2Z + 1] = H2
        Fa[H, G.F_ARM2] = 1.0
        tucked = 1.0
    else:
        tucked = 0.0
    torches = []
    for i in range(SC.NEM):
        hw, ax = _torch_world(SC, G, Fa, i)
        hp, axp = _torch_world(SC, G, Fp, i)
        top = hw + 0.44 * ax
        topp = hp + 0.44 * axp
        vel = (top - topp) * 24.0
        lean = np.array(WIND) - 0.35 * vel[:2]
        ln = np.linalg.norm(lean)
        if ln > 1.6:
            lean *= 1.6 / ln
        lit = float(Fa[i, G.F_TLIT])
        HD, left = _hammer_hd(SC, G, Fa, i, 1.08 if i == SC.GILDED else 1.03, 1.0 if i == SC.GILDED else 0.0)
        torches.append(dict(hw=hw.tolist(), ax=ax.tolist(), lit=lit, base=(top - 0.03 * ax).tolist(),
                            lean=lean.tolist(), hf=float(hf[i]), seed=3.7 * i + 1.3,
                            fl=_flicker(3.7 * i + 1.3, t), HD=HD.tolist(), left=bool(left)))
    her = SC.her_hand(t)
    herd = None
    if her is not None:
        HD, J = her
        HD = HD.copy()
        HD[2] += dz
        s = HD[12]
        E = HD[0:3] + (HD[3:6] * HD[16] + HD[6:9] * HD[17] + HD[9:12] * HD[18]) * s
        S = HD[0:3] + (HD[3:6] * HD[19] + HD[6:9] * HD[20] + HD[9:12] * HD[21]) * s
        curl, thumb, spread = _her_pose(SC, t)
        herd = dict(HD=HD.tolist(), E=E.tolist(), S=S.tolist(), curl=curl.tolist(), thumb=float(thumb),
                    spread=float(spread))
    RP = SC.ring_state(t)
    RP[3] += dz
    fs = SC.fire_state(t)
    CF, nmain = np.zeros((0, 8)), 0
    if fs.get('p3', 0.0) > 0.0:
        CF, nmain = SC.calm_flames(t)
    elif SC.plate(t) == 2 and t >= SC.FIRE_CATCH - 4:
        CF, nmain = _p2_rows(SC, G, t), 0
    zf, amt = _focus(SC, G, t, cam)
    out = dict(cam=cam[:12].tolist(), focus=[zf, amt], expo=float(SC.exposure(t)), white=float(SC.white_level(t)),
               plate=int(SC.plate(t)), figs=Fa.tolist(), tucked=tucked, torches=torches, her=herd,
               ring=RP.tolist(), fire={k: float(v) for k, v in fs.items()}, cf=CF.tolist(), nmain=int(nmain))
    if CR is not None and crowd_keep is not None and len(crowd_keep):
        cs = CR.state(t)
        CFr = cs['CF'][crowd_keep]
        FL = cs['FL'][crowd_keep]
        top = cs['top'][crowd_keep]
        axw = np.stack([np.cos(CFr[:, G.F_ANG]) * CFr[:, G.F_TX] - np.sin(CFr[:, G.F_ANG]) * CFr[:, G.F_TY],
                        np.sin(CFr[:, G.F_ANG]) * CFr[:, G.F_TX] + np.cos(CFr[:, G.F_ANG]) * CFr[:, G.F_TY],
                        CFr[:, G.F_TZ]], -1)
        axw = axw / (np.linalg.norm(axw, axis=1, keepdims=True) + 1e-12)
        hw = top - 0.44 * axw
        out['crowd'] = dict(F=CFr.tolist(), hw=hw.tolist(), ax=axw.tolist(), lit=cs['lit'][crowd_keep].tolist(),
                            base=FL[:, 0:3].tolist(), lean=FL[:, 5:7].tolist(), hf=FL[:, 3].tolist(),
                            seed=FL[:, 8].tolist(),
                            fl=[_flicker(float(s), t) for s in FL[:, 8]])
    return out


def _crowd_select(SC, G, CR, frames, cams):
    """Everyone of the crowd who is in (or near) any of these frames' views and within 45 m of the lens, with a
    level of detail per person from the nearest approach."""
    if CR is None:
        return None, None
    keep = {}
    for f, cam in zip(frames, cams):
        cs = CR.state(float(f), prev=False)
        if len(cs['x']) == 0:
            continue
        C, R, U, Fw = cam[0:3], cam[3:6], cam[6:9], cam[9:12]
        fpx = (1920 / 2) / math.tan(math.radians(25.0))
        for zz in (0.2, 1.0, 1.9):
            P = np.stack([cs['x'], cs['y'], np.full(len(cs['x']), zz)], -1)
            v = P - C
            zc = v @ Fw
            xs = 960 + fpx * (v @ R) / np.maximum(zc, 1e-3)
            ys = 402 - fpx * (v @ U) / np.maximum(zc, 1e-3)
            d = np.linalg.norm(v, axis=1)
            ok = (zc > 0.2) & (xs > -500) & (xs < 2420) & (ys > -400) & (ys < 1204) & (d < 45.0)
            for i in np.nonzero(ok)[0]:
                keep[i] = min(keep.get(i, 1e9), d[i])
    if not keep:
        return np.zeros(0, int), np.zeros(0)
    idx = np.array(sorted(keep), int)
    dist = np.array([keep[i] for i in idx])
    lod = np.where(dist < 7.0, 0.7, np.where(dist < 16.0, 0.45, 0.3))
    return idx, lod


def prep(frames, cache):
    import council_geo as CG
    SC, G = _acc()
    fr = sorted(set(int(f) for f in frames if int(f) >= START - 2))
    tag = f'{fr[0]}-{fr[-1]}_{len(fr)}'
    path = os.path.join(cache, f'state_{tag}.json')
    geo = os.path.join(cache, f'geo_{tag}.json')
    if os.path.exists(path) and os.path.exists(geo) and not os.environ.get('COUNCIL_REPREP'):
        return {'state': path, 'geo': geo}
    CR = _crowd() if not os.environ.get('COUNCIL_NOCROWD') else None
    ring0 = SC.ring_rest()
    dz = float(CG.slab_top(ring0[0], ring0[1]) + 0.5 * G.R_WIDTH - ring0[2])
    rng = np.random.default_rng(11)
    hf = 0.44 * rng.uniform(0.88, 1.12, SC.NEM)
    cams = [SC.camera(float(f), 1.0) for f in fr]
    keep, lod = _crowd_select(SC, G, CR, fr, cams)
    st = dict(frames=fr, stones=SC.stones()[:, :15].tolist(), ring_dz=dz)
    KB, LG, CH = SC.hearth_parts()
    st.update(kerb=KB.tolist(), logs=LG.tolist(), char=CH.tolist())
    st['crowd_lod'] = lod.tolist() if lod is not None else []
    per = {}
    for f in fr:
        per[str(f)] = _frame_state(SC, G, CR, float(f), hf, dz, keep)
    st['per'] = per
    os.makedirs(cache, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(st, fh)
    os.replace(tmp, path)
    geo = _bake(st, per, fr, cache, tag)
    print(f'councilc prep: {len(fr)} frames, crowd {0 if keep is None else len(keep)} kept -> {path}', flush=True)
    return {'state': path, 'geo': geo}


def post(f, hdr, depth, cam, scene):
    e = scene.get('expo', {}).get(str(f), 1.0)
    w = scene.get('white', {}).get(str(f), 0.0)
    hdr = hdr * (EXPO * e)
    L = hdr.max(axis=2, keepdims=True)
    k = np.where(L > 1e-6, CLIP_LC * np.tanh(L / CLIP_LC) / np.maximum(L, 1e-6), 1.0)
    hdr = hdr * k
    if w > 0.0:
        hdr = hdr * (1.0 - w) + w * np.array([8.0, 6.9, 5.4], np.float32)
    return hdr.astype(np.float32)



# ====================================================== venv: bake the geometry ===
# Blender's own Python on this Mac cannot load numpy (macOS 12), so every shape is computed here and handed to
# Blender as flat float32/int32 blobs (Blender side: array + foreach_set, numpy-free).

class _Blob:
    def __init__(self):
        self.buf = bytearray()
        self.idx = {}

    def put(self, arr, dtype):
        a = np.ascontiguousarray(np.asarray(arr), dtype=dtype).ravel()
        off = len(self.buf)
        self.buf += a.tobytes()
        return [off, int(a.size)]

    def mesh(self, name, V, Fc, Mt=None, attrs=None, **extra):
        import council_geo as CG
        start, tot, vidx = CG.faces_flat(Fc)
        Mt = np.zeros(len(tot), np.int32) if Mt is None or not len(Mt) else Mt
        e = dict(V=self.put(V, np.float32), S=self.put(start, np.int32), L=self.put(tot, np.int32),
                 I=self.put(vidx, np.int32), M=self.put(Mt, np.int32), A={})
        for k, v in (attrs or {}).items():
            v = np.asarray(v, np.float32)
            e['A'][k] = [self.put(v, np.float32), 1 if v.ndim == 1 else 3]
        e.update(extra)
        self.idx[name] = e
        return e


def _ground_geo():
    import council_geo as CG
    rings = np.concatenate([np.linspace(0.0, 3.0, 61)[1:], np.linspace(3.0, 30.0, 55)[1:],
                            np.geomspace(30.0, 400.0, 30)[1:]])
    nt = 256
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    R, T = np.meshgrid(rings, th, indexing='ij')
    X, Y = R * np.cos(T), R * np.sin(T)
    Z = (0.05 * np.sin(0.21 * X + 0.4) * np.sin(0.17 * Y + 1.1) + 0.8 * np.sin(0.023 * X + 0.3) * np.sin(0.019 * Y + 2.0)
         + 0.012 * np.sin(1.3 * X) * np.sin(1.1 * Y + 0.3)) * np.clip((R - 9.0) / 12.0, 0, 1)
    Z = Z + 0.008 * np.sin(3.1 * X + 0.2) * np.sin(2.7 * Y) * np.clip(R - 1.0, 0, 1) - 0.004
    m = CG.Mesh()
    base = m.add_grid(np.stack([X, Y, Z], -1), 0)
    m.add_fan(base, nt, (0.0, 0.0, -0.004), 0, flip=True)
    return m.arrays()


def _heather_points(rng, avoid, nvar, clear_r=1.05):
    """Tufts sparse and trodden flat inside the circle and on the crowd's ground, thicker out on the moor (to 26 m;
    further off the ground's own texture carries it). No tuft where a figure stands (it would come through the hem)."""
    pts, rot, scl, var = [], [], [], []
    avoid = np.asarray(avoid, np.float64).reshape(-1, 2) if len(avoid) else np.zeros((0, 2))
    for r0, r1, dens in ((clear_r, 2.2, 1.2), (2.2, 8.2, 4.0), (8.2, 15.5, 6.0), (15.5, 26.0, 14.0)):
        n = int(np.pi * (r1 * r1 - r0 * r0) * dens)
        rr = np.sqrt(rng.uniform(r0 * r0, r1 * r1, n))
        aa = rng.uniform(0, 2 * np.pi, n)
        x, y = rr * np.cos(aa), rr * np.sin(aa)
        ok = np.ones(n, bool)
        for k0 in range(0, len(avoid), 256):
            A = avoid[k0:k0 + 256]
            d2 = (x[:, None] - A[None, :, 0]) ** 2 + (y[:, None] - A[None, :, 1]) ** 2
            ok &= d2.min(1) > 0.45 ** 2
        x, y, rr = x[ok], y[ok], rr[ok]
        n = len(x)
        trod = ((rr < 15.5) & (rng.uniform(size=n) < 0.8)).astype(int)
        s_ = rng.uniform(0.6, 1.3, n) * np.where(trod, 0.8, 1.0)
        pts.append(np.stack([x, y, np.full(n, -0.01)], -1))
        rot.append(np.stack([rng.normal(0, 0.08, n), rng.normal(0, 0.08, n), rng.uniform(0, 6.28, n)], -1))
        scl.append(np.stack([s_, s_, s_ * np.where(trod, 0.55, 1.0)], -1))
        var.append(rng.integers(0, nvar, n) * 2 + trod)
    return np.concatenate(pts), np.concatenate(rot), np.concatenate(scl), np.concatenate(var)


def _litter_points(rng):
    """The floor round the hearth, as it is at 1 mm a pixel from above: pebbles, broken stems of dead heather,
    charcoal crumbs thick in the ash bed and thinning outward. Kinds: 0-2 pebbles, 3-6 stems, 7-9 crumbs."""
    import council_geo as CG
    P, R, S, K = [], [], [], []
    for r0, r1, dens, kinds in ((0.34, 2.6, 40.0, (0, 1, 2)), (0.34, 2.6, 70.0, (3, 4, 5, 6)),
                                (0.36, 1.1, 140.0, (7, 8, 9)), (1.1, 2.6, 10.0, (7, 8, 9))):
        n = int(np.pi * (r1 * r1 - r0 * r0) * dens)
        rr = np.sqrt(rng.uniform(r0 * r0, r1 * r1, n))
        aa = rng.uniform(0, 2 * np.pi, n)
        ok = rr > CG.slab_outline(aa) * 1.03
        rr, aa = rr[ok], aa[ok]
        n = len(rr)
        P.append(np.stack([rr * np.cos(aa), rr * np.sin(aa), np.full(n, -0.002)], -1))
        R.append(np.stack([rng.normal(0, 0.12, n), rng.normal(0, 0.12, n), rng.uniform(0, 6.28, n)], -1))
        sc = rng.uniform(0.6, 1.4, n)
        S.append(np.stack([sc, sc, sc], -1))
        K.append(rng.choice(kinds, n))
    return np.concatenate(P), np.concatenate(R), np.concatenate(S), np.concatenate(K)


def _ring_geo():
    R_IN, TH, WD, SQ = 0.0094, 0.0023, 0.0052, 2.8
    n_t, n_a = 96, 20
    th = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    a = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
    T, A = np.meshgrid(th, a, indexing='ij')
    ca, sa = np.cos(A), np.sin(A)
    rr = R_IN + 0.5 * TH + 0.5 * TH * np.sign(ca) * np.abs(ca) ** (2.0 / SQ)
    zz = 0.5 * WD * np.sign(sa) * np.abs(sa) ** (2.0 / SQ)
    V = np.stack([rr * np.cos(T), rr * np.sin(T), zz], -1).reshape(-1, 3)
    j, i = np.meshgrid(np.arange(n_t), np.arange(n_a), indexing='ij')
    j2, i2 = (j + 1) % n_t, (i + 1) % n_a
    q = np.stack([j * n_a + i, j * n_a + i2, j2 * n_a + i2, j2 * n_a + i], -1).reshape(-1, 4)
    return V, [q], np.zeros(len(q), np.int32)


def _her_arm_geo(h):
    import council_geo as CG
    m = CG.Mesh()
    CG._sleeve(m, h['S'], h['E'], h['HD'][0:3], 21.0, 1.0, short=-0.02)
    return m.arrays()


def _bake(st, per, fr, cache, tag):
    """Every mesh of the shot (static + the figures' topology) into one blob; the figures' per-frame vertex
    positions into figs_<tag>_<f>.bin (the dynamic objects in the index's order)."""
    import council_geo as CG
    B = _Blob()
    rng = np.random.default_rng(1970)
    f0 = fr[len(fr) // 2]
    s0 = per[str(f0)]
    V, Fc, Mt, _ = _ground_geo()
    B.mesh('ground', V, Fc, Mt)
    for k in range(3):
        V, Fc, Mt, _ = CG.pebble(200 + k, 0.010 + 0.007 * k)
        B.mesh(f'l0{k}', V, Fc, Mt, kind='peb')
    for k in range(4):
        V, Fc, Mt, _ = CG.twig(300 + k, 0.045 + 0.030 * k)
        B.mesh(f'l0{3 + k}', V, Fc, Mt, kind='twig')
    for k in range(3):
        sz = 0.005 + 0.004 * k
        V, Fc, Mt, _ = CG.chunk((0.0, 0.0, 0.3 * sz), (sz, sz * 0.7, sz * 0.5), 400.0 + k)
        B.mesh(f'l0{7 + k}', V, Fc, Mt, {'u': np.full(len(V), 0.9)}, kind='crumb')
    P, R, S, K = _litter_points(np.random.default_rng(77))
    B.idx['litter'] = dict(P=B.put(P, np.float32), R=B.put(R, np.float32), S=B.put(S, np.float32),
                           K=B.put(K, np.int32), n=len(P))
    if not os.environ.get('COUNCIL_NOHEATHER'):
        nvar = 6
        for k in range(nvar):
            h = rng.uniform(0.16, 0.26)
            ns = int(rng.integers(12, 22))
            for flat in (0, 1):
                V, Fc, Mt, _ = CG.heather_tuft(100 + k, height=h, n_stems=ns, flat=float(flat)).arrays()
                B.mesh(f'tuft{k}_{flat}', V, Fc, Mt)
        av = [row[0:2] for row in s0['figs']]
        if s0.get('crowd'):
            av += [row[0:2] for row in s0['crowd']['F']]
        P, R, S, K = _heather_points(rng, av, nvar)
        B.idx['heather'] = dict(P=B.put(P, np.float32), R=B.put(R, np.float32), S=B.put(S, np.float32),
                                K=B.put(K, np.int32), n=len(P))
    for j, S in enumerate(st['stones']):
        V, Fc, Mt, _ = CG.standing_stone(np.array(S))
        B.mesh(f'stone{j}', V, Fc, Mt, alb=float(S[14]))
    V, Fc, Mt, _ = CG.slab()
    B.mesh('slab', V, Fc, Mt)
    for k, K in enumerate(st['kerb']):
        x, y, z, rx, ry, rz, c, s, sd_ = K
        V, Fc, Mt, _ = CG.rock((0.0, 0.0, 0.0), (rx, ry, rz), sd_, flat_bottom=-0.6 * rz)
        W = np.stack([x + V[:, 0] * c - V[:, 1] * s, y + V[:, 0] * s + V[:, 1] * c, V[:, 2] + z - 0.25 * rz], -1)
        B.mesh(f'kerb{k}', W, Fc, Mt)
    LG = np.array(st['logs'])
    for k in range(LG.shape[0]):
        A_, B_, ra = LG[k, 0:3], LG[k, 3:6], LG[k, 7]
        if LG[k, 9] < 0.5:
            V, Fc, Mt, at = CG.log(A_, B_, ra, 7.0 + k, burn_a=0.40)
            B.mesh(f'log{k}', V, Fc, Mt, at, kind='log')
        else:
            V, Fc, Mt, at = CG.stick(A_, B_, LG[k, 6], 30.0 + k)
            B.mesh(f'stick{k}', V, Fc, Mt, at, kind='char' if k % 3 == 0 else 'bark')
    for k, Q in enumerate(st['char']):
        x, y, z, a, b, c, sd_ = Q
        V, Fc, Mt, _ = CG.chunk((x, y, z), (a, b, c), sd_)
        B.mesh(f'chunk{k}', V, Fc, Mt, {'u': np.full(len(V), 0.9)})
    if any(per[str(f)]['plate'] == 3 for f in fr):
        pile = CG.Mesh()
        for k in range(300):
            a = rng.uniform(0, 2 * np.pi)
            r = 0.27 * math.sqrt(rng.uniform(0, 1))
            x, y = r * math.cos(a), r * math.sin(a)
            sz = rng.uniform(0.012, 0.045) * (1.0 - 0.45 * r / 0.27)
            z = float(CG.slab_top(x, y)) + 0.25 * sz + 0.06 * max(0.0, 1.0 - (r / 0.20) ** 2) * rng.uniform(0.2, 1.0)
            V, Fc, Mt, _ = CG.chunk((x, y, z), (sz, sz * rng.uniform(0.6, 0.9), sz * 0.55), 50.0 + k)
            base = pile.n
            pile.V.append(V)
            pile._attrs({'u': np.full(len(V), rng.uniform(0.0, 0.3))}, len(V))
            pile.n += len(V)
            for f_ in Fc:
                pile.F.append(f_ + base)
                pile.M.append(np.zeros(len(f_), np.int32))
        V, Fc, Mt, at = pile.arrays()
        B.mesh('coalpile', V, Fc, Mt, at)
    V, Fc, Mt, at = CG.torch_mesh()
    B.mesh('torch', V, Fc, Mt, at)
    V, Fc, Mt = _ring_geo()
    B.mesh('ring', V, Fc, Mt)
    # ---- the figures: topology from the middle frame, positions per frame
    dyn = []
    Fa0 = np.array(s0['figs'])
    for i in range(Fa0.shape[0]):
        V, Fc, Mt, at = CG.figure(Fa0[i], lod=1.0, fist=False)
        at['rest'] = V
        B.mesh(f'fig{i}', CG.fig_to_world(V, Fa0[i]), Fc, Mt, at, c1=list(map(float, Fa0[i, 9:12])),
               c2=list(map(float, Fa0[i, 31:34])), sub=1, nv=len(V))
        dyn.append(('fig', i, 1.0, f'fig{i}'))
    if s0.get('crowd'):
        CF0 = np.array(s0['crowd']['F'])
        for j in range(CF0.shape[0]):
            lod = st['crowd_lod'][j]
            V, Fc, Mt, at = CG.figure(CF0[j], lod=lod, fist=True)
            at['rest'] = V
            B.mesh(f'crowd{j}', CG.fig_to_world(V, CF0[j]), Fc, Mt, at, c1=list(map(float, CF0[j, 9:12])),
                   c2=list(map(float, CF0[j, 31:34])), sub=1 if lod >= 0.7 else 0, nv=len(V))
            dyn.append(('crowd', j, lod, f'crowd{j}'))
    her_f = [f for f in fr if per[str(f)]['her'] is not None]
    if her_f:
        V, Fc, Mt, at = _her_arm_geo(per[str(her_f[0])]['her'])
        at['rest'] = V
        B.mesh('her_arm', V, Fc, Mt, at, c1=list(map(float, Fa0[13, 9:12])), c2=list(map(float, Fa0[13, 31:34])),
               sub=1, nv=len(V))
        dyn.append(('arm', 0, 1.0, 'her_arm'))
    B.idx['_dyn'] = [d[3] for d in dyn]
    blob = os.path.join(cache, f'geo_{tag}.bin')
    with open(blob + '.tmp', 'wb') as fh:
        fh.write(bytes(B.buf))
    os.replace(blob + '.tmp', blob)
    B.idx['_blob'] = blob
    last = {}
    for f in fr:
        s = per[str(f)]
        Fa = np.array(s['figs'])
        CF = np.array(s['crowd']['F']) if s.get('crowd') else None
        parts = []
        for kind, i, lod, name in dyn:
            if kind == 'fig':
                V = CG.fig_to_world(CG.figure(Fa[i], lod=1.0, fist=False)[0], Fa[i])
            elif kind == 'crowd':
                V = CG.fig_to_world(CG.figure(CF[i], lod=lod, fist=True)[0], CF[i])
            else:
                h = s['her'] or per[str(her_f[0] if f < her_f[0] else her_f[-1])]['her']
                V = _her_arm_geo(h)[0]
            if len(V) != B.idx[name]['nv']:
                V = last.get(name, None)
                if V is None:
                    raise SystemExit(f'councilc: {name} changed topology at {f}')
            last[name] = V
            parts.append(np.asarray(V, np.float32).ravel())
        p = os.path.join(cache, f'figs_{tag}_{f:05d}.bin')
        with open(p + '.tmp', 'wb') as fh:
            fh.write(np.concatenate(parts).astype(np.float32).tobytes())
        os.replace(p + '.tmp', p)
    B.idx['_figs'] = os.path.join(cache, f'figs_{tag}_%05d.bin')
    ip = os.path.join(cache, f'geo_{tag}.json')
    with open(ip + '.tmp', 'w') as fh:
        json.dump(B.idx, fh)
    os.replace(ip + '.tmp', ip)
    return ip


# ============================================================ Blender: materials ===

def _keyv(C, nb, name, fn, frames):
    v = nb.value(0.0, name)
    for f in frames:
        C.key_socket(v.outputs[0], f, fn(f))
    return v.outputs[0]


def _noise_col(nb, vec, scale, detail=3.0, rough=0.5):
    return nb.noise(vec, scale=scale, detail=detail, rough=rough)


# ---------------------------------------------------------------- materials ---

def mat_wool(new_material, name, cattr):
    """Heavy fulled wool, dark: fuzz and a fine twill, only a faint sheen at the grazing rims (never velvet or
    satin), damp and darker toward the hem."""
    m, nb = new_material(name)
    R = nb.attr('rest').outputs['Vector']
    fib = nb.noise(R, scale=900.0, detail=5.0, rough=0.7)
    twill = nb.wave(nb.vadd(R, nb.vscale(nb.noise(R, scale=40.0, detail=2.0).outputs['Color'], 0.004)), scale=260.0,
                    wtype='BANDS', direction='DIAGONAL')
    mott = nb.noise(R, scale=5.0, detail=3.0, rough=0.55)
    base = nb.attr(cattr, 'OBJECT').outputs['Color']
    k = nb.mul(nb.madd(mott.outputs['Fac'], 0.40, 0.80), nb.madd(fib.outputs['Fac'], 0.25, 0.87))
    hem = nb.attr('hem').outputs['Fac']
    k = nb.mul(k, nb.madd(nb.sstep(0.0, 1.0, hem), 0.40, 0.60))
    ridge = nb.attr('ridge').outputs['Fac']
    k = nb.mul(k, nb.madd(nb.clamp01(ridge), 0.05, 1.0))
    col = nb.colscale(base, nb.mul(k, 1.35))
    bs = nb.principled(Base_Color=col, Roughness=0.95)
    bs.inputs['Specular IOR Level'].default_value = 0.18
    bs.inputs['Sheen Weight'].default_value = 0.22
    bs.inputs['Sheen Roughness'].default_value = 0.30
    bs.inputs['Sheen Tint'].default_value = (0.95, 0.90, 0.85, 1.0)
    h = nb.add(nb.mul(fib.outputs['Fac'], 0.7), nb.mul(twill.outputs['Fac'], 0.3))
    nb.link(nb.bump(h, 0.30, 0.0012), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def mat_void(new_material):
    m, nb = new_material('void')
    bs = nb.principled(Base_Color=(0.0025, 0.0022, 0.0020), Roughness=1.0)
    bs.inputs['Specular IOR Level'].default_value = 0.0
    nb.output(surface=bs)
    return m


def mat_shawl(new_material):
    """Her shawl: a woven wool shawl, madder red (a deep, slightly brown red), plain weave."""
    m, nb = new_material('shawl')
    R = nb.attr('rest').outputs['Vector']
    x, y, z = nb.sep(R)
    wa = nb.math('SINE', nb.mul(nb.add(x, y), 2 * math.pi / 0.0035))
    wb = nb.math('SINE', nb.mul(z, 2 * math.pi / 0.0035))
    weave = nb.mul(nb.madd(wa, 0.5, 0.5), nb.madd(wb, 0.5, 0.5))
    fib = nb.noise(R, scale=700.0, detail=4.0, rough=0.7)
    mott = nb.noise(R, scale=9.0, detail=2.0)
    col = nb.colscale((0.155, 0.018, 0.020), nb.mul(nb.madd(mott.outputs['Fac'], 0.4, 0.8),
                                                   nb.madd(fib.outputs['Fac'], 0.3, 0.85)))
    bs = nb.principled(Base_Color=col, Roughness=0.9)
    bs.inputs['Sheen Weight'].default_value = 0.9
    bs.inputs['Sheen Roughness'].default_value = 0.4
    bs.inputs['Sheen Tint'].default_value = (1.0, 0.75, 0.7, 1.0)
    bs.inputs['Specular IOR Level'].default_value = 0.2
    nb.link(nb.bump(nb.add(weave, nb.mul(fib.outputs['Fac'], 0.6)), 0.3, 0.0010), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def mat_leather_simple(new_material, name='leather_far', base=(0.040, 0.026, 0.017)):
    m, nb = new_material(name)
    P = nb.texco().outputs['Object']
    g = nb.noise(P, scale=600.0, detail=3.0)
    bs = nb.principled(Base_Color=base, Roughness=nb.madd(g.outputs['Fac'], 0.2, 0.52))
    bs.inputs['Specular IOR Level'].default_value = 0.35
    bs.inputs['Coat Weight'].default_value = 0.06
    nb.link(nb.bump(g.outputs['Fac'], 0.2, 0.0005), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def mat_gilt(new_material, fr, C):
    """The forge that grasped: a dark leather glove crusted with gold over the knuckles and the back of the hand,
    cracked, blistered, with ember scars still alive in it."""
    m, nb = new_material('gilt_glove')
    R = nb.attr('rest').outputs['Vector']
    rx, ry, rz = nb.sep(R)
    grain = nb.voronoi(R, scale=2600.0, feature='SMOOTH_F1')
    fine = nb.noise(R, scale=900.0, detail=4.0, rough=0.6)
    back = nb.sstep(-0.002, 0.008, rz)
    kn = nb.mul(nb.sstep(0.035, 0.070, rx), nb.sstep(0.19, 0.12, rx))
    lob = nb.noise(R, scale=60.0, detail=4.0, rough=0.6)
    crust = nb.mul(nb.mul(back, kn), nb.sstep(0.36, 0.50, nb.add(lob.outputs['Fac'], nb.mul(nb.sstep(0.07, 0.10, rx), 0.12))))
    vo = nb.voronoi(nb.vadd(R, nb.vscale(nb.noise(R, scale=300.0).outputs['Color'], 0.002)), scale=520.0,
                    feature='DISTANCE_TO_EDGE')
    crack = nb.sub(1.0, nb.sstep(0.0, 0.08, vo.outputs['Distance']))
    gold = nb.mul(crust, nb.sub(1.0, nb.mul(crack, 0.85)))
    leather_col = nb.colscale((0.040, 0.026, 0.017), nb.madd(fine.outputs['Fac'], 0.3, 0.85))
    col = nb.mixcol(gold, leather_col, (1.0, 0.70, 0.32))
    bs = nb.principled(Base_Color=col, Roughness=nb.mixf(gold, 0.46, 0.24), Metallic=gold)
    bs.inputs['Coat Weight'].default_value = 0.12
    h = nb.add(nb.mul(grain.outputs['Distance'], 0.6), nb.mul(crust, 2.0))
    h = nb.sub(h, nb.mul(crack, nb.mul(crust, 1.5)))
    nb.link(nb.bump(h, 0.4, 0.0004), bs.inputs['Normal'])
    scar = nb.mul(nb.sstep(0.80, 0.90, nb.noise(R, scale=180.0, detail=2.0).outputs['Fac']), crack)
    bb = nb.n('ShaderNodeBlackbody')
    bb.inputs['Temperature'].default_value = 1250.0
    em = nb.emission(bb.outputs['Color'], nb.mul(nb.mul(scar, crust), 2.5))
    nb.output(surface=nb.addshader(bs, em))
    return m


def mat_gold(new_material):
    m, nb = new_material('ring_gold')
    P = nb.texco().outputs['Object']
    pol = nb.noise(P, scale=300.0, detail=2.0)
    bs = nb.principled(Base_Color=(1.0, 0.74, 0.34), Metallic=1.0, Roughness=nb.madd(pol.outputs['Fac'], 0.06, 0.07))
    nb.output(surface=bs)
    return m


def mat_torch(new_material):
    """Slot 0: the stave (dark worn wood, charred black toward the head). Slot 1: the head: strips of cloth wound
    and soaked in pitch, charred black with a sticky sheen; while lit its crevices glow; spent: cold, black, ash."""
    ms = []
    m, nb = new_material('torch_wood')
    P = nb.texco().outputs['Object']
    x, y, z = nb.sep(P)
    grain = nb.noise(nb.vmul(P, (60.0, 60.0, 3.0)), scale=1.0, detail=4.0, rough=0.6)
    soot = nb.sstep(-0.02, 0.20, nb.add(z, nb.mul(nb.sub(grain.outputs['Fac'], 0.5), 0.08)))
    col = nb.mixcol(soot, nb.colscale((0.034, 0.024, 0.016), nb.madd(grain.outputs['Fac'], 0.5, 0.75)),
                    (0.007, 0.006, 0.0055))
    bs = nb.principled(Base_Color=col, Roughness=0.80)
    bs.inputs['Specular IOR Level'].default_value = 0.3
    nb.link(nb.bump(grain.outputs['Fac'], 0.3, 0.0006), bs.inputs['Normal'])
    nb.output(surface=bs)
    ms.append(m)
    m, nb = new_material('torch_head')
    P = nb.texco().outputs['Object']
    x, y, z = nb.sep(P)
    strands = nb.noise(nb.vmul(P, (1.0, 1.0, 0.25)), scale=260.0, detail=4.0, rough=0.7)
    lump = nb.noise(P, scale=70.0, detail=3.0)
    pitch = nb.sstep(0.55, 0.75, lump.outputs['Fac'])
    lit = nb.attr('lit', 'OBJECT').outputs['Fac']
    ash = nb.attr('ash', 'OBJECT').outputs['Fac']
    ashm = nb.mul(ash, nb.sstep(0.45, 0.65, nb.noise(P, scale=120.0, detail=3.0).outputs['Fac']))
    col = nb.mixcol(ashm, (0.008, 0.007, 0.006), (0.12, 0.115, 0.11))
    bs = nb.principled(Base_Color=col, Roughness=nb.mixf(pitch, 0.80, 0.30))
    bs.inputs['Specular IOR Level'].default_value = 0.4
    nb.link(nb.bump(nb.add(strands.outputs['Fac'], nb.mul(lump.outputs['Fac'], 0.5)), 0.45, 0.0015), bs.inputs['Normal'])
    crev = nb.sstep(0.40, 0.27, strands.outputs['Fac'])
    fl = nb.noise(P, scale=40.0, detail=1.0, dims='4D', w=nb.attr('tseed', 'OBJECT').outputs['Fac'])
    glow = nb.mul(nb.mul(crev, lit), nb.madd(fl.outputs['Fac'], 1.2, 0.2))
    glow = nb.mul(glow, nb.sstep(0.33, 0.40, z))
    bb = nb.n('ShaderNodeBlackbody')
    bb.inputs['Temperature'].default_value = 1300.0
    em = nb.emission(bb.outputs['Color'], nb.mul(glow, 0.9))
    nb.output(surface=nb.addshader(bs, em))
    ms.append(m)
    return ms


def mat_flame(new_material, C, fr):
    """Volumetric flame (emission only) in the flame object's space (z up from its root, metres). Per object:
    fH height, fR radius, flx/fly lean (downwind shift per unit height), fI brightness, fseed. A turbulent field
    rising through it and a threshold that climbs with height break the top into tapering tongues with ragged,
    transparent tips; blackbody from a yellow-orange root (never white) to deep orange-red tips."""
    m, nb = new_material('flame')
    P = nb.texco().outputs['Object']
    px, py, pz = nb.sep(P)
    Hn = nb.attr('fH', 'OBJECT').outputs['Fac']
    Rn = nb.attr('fR', 'OBJECT').outputs['Fac']
    lx = nb.attr('flx', 'OBJECT').outputs['Fac']
    ly = nb.attr('fly', 'OBJECT').outputs['Fac']
    In = nb.attr('fI', 'OBJECT').outputs['Fac']
    sd = nb.attr('fseed', 'OBJECT').outputs['Fac']
    T = _keyv(C, nb, 'ftime', lambda f: f / FPS, fr)
    Hs = nb.mx(Hn, 0.02)
    h = nb.div(pz, Hs)
    hc = nb.clamp01(h)
    bend = nb.mul(nb.pw(hc, 1.6), Hs)
    qx = nb.sub(px, nb.mul(lx, bend))
    qy = nb.sub(py, nb.mul(ly, bend))
    lam = nb.madd(Hs, 0.24, 0.02)
    rise = nb.madd(Hs, 2.0, 0.45)
    tz = nb.sub(pz, nb.mul(T, rise))
    Q = nb.comb(nb.div(qx, lam), nb.div(qy, lam), nb.div(tz, lam))
    n1 = nb.noise(Q, scale=1.0, detail=1.0, rough=0.5, dims='4D', w=nb.add(nb.mul(T, 0.9), sd))
    dv = nb.vsub(n1.outputs['Color'], (0.5, 0.5, 0.5))
    amp = nb.mul(nb.mul(nb.pw(hc, 0.9), Hs), 0.50)
    dx, dy, dzz = nb.sep(dv)
    qx2 = nb.add(qx, nb.mul(dx, amp))
    qy2 = nb.add(qy, nb.mul(dy, amp))
    qz2 = nb.add(pz, nb.mul(dzz, nb.mul(amp, 0.5)))
    h2 = nb.div(qz2, Hs)
    h2c = nb.clamp01(h2)
    r = nb.math('SQRT', nb.add(nb.mul(qx2, qx2), nb.mul(qy2, qy2)))
    prof = nb.mul(nb.mul(Rn, 1.85), nb.mul(nb.pw(nb.mx(h2c, 0.002), 0.33), nb.pw(nb.sub(1.0, h2c), 0.85)))
    rr = nb.div(r, nb.mx(prof, 0.001))
    core = nb.sub(1.0, nb.sstep(0.35, 1.0, rr))
    l2 = nb.mul(lam, 0.75)
    Q2 = nb.comb(nb.add(nb.div(qx2, l2), sd), nb.div(qy2, l2), nb.div(nb.sub(qz2, nb.mul(T, nb.mul(rise, 1.3))), l2))
    n2 = nb.noise(Q2, scale=1.0, detail=2.0, rough=0.55)
    thr = nb.mul(h2c, 0.85)
    tongue = nb.sstep(nb.sub(thr, 0.03), nb.add(thr, 0.12), nb.madd(n2.outputs['Fac'], 1.25, -0.10))
    l3 = nb.mul(lam, 0.30)
    Q3 = nb.comb(nb.sub(nb.div(qx2, l3), sd), nb.div(qy2, l3), nb.div(nb.sub(qz2, nb.mul(T, nb.mul(rise, 1.6))), l3))
    n3 = nb.noise(Q3, scale=1.0, detail=1.0, rough=0.5)
    dens = nb.mul(nb.mul(core, tongue), nb.madd(n3.outputs['Fac'], 1.3, 0.35))
    dens = nb.mul(dens, nb.sstep(-0.03, 0.05, h2))
    dens = nb.mul(dens, nb.sstep(1.05, 0.90, h2))
    temp = nb.add(1100.0, nb.mul(1250.0, nb.mul(nb.pw(nb.sub(1.0, h2c), 1.2), nb.sub(1.0, nb.mul(nb.clamp01(rr), 0.45)))))
    bb = nb.n('ShaderNodeBlackbody')
    nb.link(temp, bb.inputs['Temperature'])
    corek = nb.madd(nb.mul(nb.pw(nb.sub(1.0, nb.clamp01(rr)), 2.0), nb.sub(1.0, h2c)), 1.5, 1.0)
    strength = nb.mul(nb.mul(nb.mul(nb.pw(dens, 1.3), In), nb.madd(nb.sub(1.0, h2c), 0.75, 0.25)), corek)
    em = nb.emission(bb.outputs['Color'], nb.mul(strength, FLAME_E))
    nb.output(volume=em)
    try:
        m.cycles.volume_step_rate = 0.40
        m.cycles.homogeneous_volume = False
    except Exception as e:
        print('flame step rate:', e, flush=True)
    return m


def mat_stone(new_material, name, base=(0.085, 0.080, 0.072), lichen=0.6, soot_fire=0.0):
    """Weathered gritstone: grain, pits, dark streaks; crusts of lichen (grey-green rosettes, a few ochre spots)
    where the weather reaches; soot on faces that look at the fire (soot_fire)."""
    m, nb = new_material(name)
    P = nb.texco().outputs['Object']
    N = nb.geo().outputs['Normal']
    grain = nb.voronoi(P, scale=900.0, feature='F1')
    gr, _, _ = nb.sep(grain.outputs['Color'])
    pits = nb.noise(P, scale=60.0, detail=5.0, rough=0.65)
    stain = nb.noise(P, scale=4.0, detail=4.0, rough=0.6)
    col = nb.colscale(base, nb.mul(nb.madd(stain.outputs['Fac'], 0.55, 0.72), nb.madd(gr, 0.25, 0.88)))
    lv = nb.voronoi(P, scale=22.0, feature='F1')
    ln = nb.noise(P, scale=35.0, detail=4.0)
    lic = nb.mul(nb.sstep(0.26, 0.12, nb.add(lv.outputs['Distance'], nb.mul(ln.outputs['Fac'], 0.12))),
                 nb.sstep(0.45, 0.62, nb.noise(P, scale=3.0, detail=2.0).outputs['Fac']))
    lic = nb.mul(lic, lichen)
    spots = nb.mul(nb.sstep(0.80, 0.86, nb.noise(P, scale=45.0, detail=2.0).outputs['Fac']), lichen * 0.8)
    col = nb.mixcol(lic, col, (0.26, 0.28, 0.23))
    col = nb.mixcol(spots, col, (0.30, 0.17, 0.05))
    rough = 0.88
    if soot_fire > 0:
        px, py, pz = nb.sep(P)
        nx, ny, nz = nb.sep(N)
        rl = nb.math('SQRT', nb.add(nb.mul(px, px), nb.mul(py, py)))
        facing = nb.clamp01(nb.div(nb.mul(-1.0, nb.add(nb.mul(nx, px), nb.mul(ny, py))), nb.mx(rl, 0.01)))
        soot = nb.mul(nb.mul(facing, soot_fire), nb.sstep(0.35, 0.7, nb.noise(P, scale=12.0, detail=3.0).outputs['Fac']))
        soot = nb.mx(soot, nb.mul(nb.clamp01(facing), soot_fire * 0.6))
        col = nb.mixcol(soot, col, (0.010, 0.009, 0.008))
    bs = nb.principled(Base_Color=col, Roughness=rough)
    bs.inputs['Specular IOR Level'].default_value = 0.25
    h = nb.add(nb.mul(pits.outputs['Fac'], 1.0), nb.mul(grain.outputs['Distance'], 0.5))
    h = nb.add(h, nb.mul(lic, 0.6))
    nb.link(nb.bump(h, 0.55, 0.004), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def mat_slab(new_material):
    """The council's flat stone: weathered dark-grey gritstone, a natural cleft top; a patchy soot stain in the middle
    where fires have stood, the broken sides sooted and dark, ash dust in the hollows, crusts of lichen on the
    weathered outer top. It never glows of itself."""
    m, nb = new_material('slab')
    P = nb.texco().outputs['Object']
    px, py, pz = nb.sep(P)
    nz = nb.sep(nb.geo().outputs['Normal'])[2]
    rl = nb.math('SQRT', nb.add(nb.mul(px, px), nb.mul(py, py)))
    grain = nb.voronoi(P, scale=1100.0, feature='F1')
    gr, _, _ = nb.sep(grain.outputs['Color'])
    pits = nb.noise(P, scale=90.0, detail=6.0, rough=0.65)
    stain = nb.noise(P, scale=9.0, detail=4.0, rough=0.6)
    col = nb.colscale((0.092, 0.087, 0.078), nb.mul(nb.madd(stain.outputs['Fac'], 0.60, 0.70), nb.madd(gr, 0.30, 0.85)))
    sootn = nb.noise(P, scale=11.0, detail=4.0, rough=0.6)
    soot = nb.mul(nb.sstep(0.15, 0.04, nb.add(rl, nb.mul(nb.sub(sootn.outputs['Fac'], 0.5), 0.16))), 0.45)
    streak = nb.mul(nb.sstep(0.55, 0.70, nb.noise(nb.vmul(P, (1.0, 1.0, 0.2)), scale=30.0, detail=2.0).outputs['Fac']),
                    nb.sstep(0.34, 0.18, rl))
    soot = nb.mx(soot, nb.mul(streak, 0.25))
    side = nb.sstep(0.80, 0.40, nz)
    soot = nb.mx(soot, nb.mul(side, 0.85))
    col = nb.mixcol(soot, col, (0.011, 0.010, 0.0095))
    dust = nb.mul(nb.sstep(0.52, 0.36, pits.outputs['Fac']), nb.mul(nb.sstep(0.16, 0.30, rl), nb.sub(1.0, side)))
    col = nb.mixcol(nb.mul(dust, 0.40), col, (0.15, 0.145, 0.14))
    lv = nb.voronoi(P, scale=26.0, feature='F1')
    lic = nb.mul(nb.sstep(0.24, 0.10, nb.add(lv.outputs['Distance'], nb.mul(nb.noise(P, scale=40.0).outputs['Fac'], 0.10))),
                 nb.sstep(0.27, 0.36, rl))
    lic = nb.mul(nb.mul(lic, nb.sstep(0.40, 0.62, nb.noise(P, scale=4.0, detail=2.0).outputs['Fac'])), nb.sub(1.0, soot))
    col = nb.mixcol(nb.mul(lic, 0.85), col, (0.20, 0.215, 0.17))
    bs = nb.principled(Base_Color=col, Roughness=nb.madd(soot, -0.08, 0.90))
    bs.inputs['Specular IOR Level'].default_value = 0.22
    flake = nb.noise(P, scale=7.0, detail=3.0, rough=0.5)
    steps = nb.sstep(0.48, 0.52, flake.outputs['Fac'])                   # layers flaked off the cleft face
    h = nb.add(pits.outputs['Fac'], nb.mul(grain.outputs['Distance'], 0.4))
    h = nb.add(nb.add(h, nb.mul(lic, 0.5)), nb.mul(steps, 1.6))
    nb.link(nb.bump(h, 0.6, 0.0030), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def mat_char(new_material, C, fr, heat_fn, name='char', cell=45.0, glow_k=3.0):
    """Charcoal: warped blocky alligator cells (no crack runs straight or meets another square), black with a dull
    silvery sheen on the cell faces and darker cracks; burning, the cracks and the underside glow softly near the
    burning end (attribute u: 0 = the inner, burnt end) under patchy white ash. Cold, nothing glows."""
    m, nb = new_material(name)
    P = nb.texco().outputs['Object']
    warp = nb.noise(P, scale=cell * 0.35, detail=3.0, rough=0.6)
    Pw = nb.vadd(P, nb.vscale(nb.vsub(warp.outputs['Color'], (0.5, 0.5, 0.5)), 1.6 / cell))
    vo = nb.voronoi(Pw, scale=cell, feature='DISTANCE_TO_EDGE', rand=1.0)
    vo2 = nb.voronoi(Pw, scale=cell * 2.3, feature='DISTANCE_TO_EDGE', rand=1.0)
    brk = nb.sstep(0.30, 0.60, nb.noise(P, scale=cell * 0.5, detail=2.0).outputs['Fac'])
    crack = nb.sub(1.0, nb.sstep(0.0, 0.10, vo.outputs['Distance']))
    crack2 = nb.mul(nb.sub(1.0, nb.sstep(0.0, 0.07, vo2.outputs['Distance'])), brk)
    cr = nb.mx(crack, nb.mul(crack2, 0.7))
    u = nb.attr('u').outputs['Fac']
    heat = _keyv(C, nb, name + '_heat', heat_fn, fr)
    near = nb.sstep(0.55, 0.05, u)
    nz = nb.sep(nb.geo().outputs['Normal'])[2]
    under = nb.sstep(0.2, -0.7, nz)
    hotn = nb.noise(P, scale=cell * 0.25, detail=3.0, dims='4D', w=_keyv(C, nb, name + '_t', lambda f: f / 24.0 * 0.4, fr))
    glow = nb.mul(nb.mx(nb.mul(cr, near), nb.mul(under, nb.mul(near, 0.5))), nb.sstep(0.35, 0.7, hotn.outputs['Fac']))
    glow = nb.mul(glow, heat)
    bb = nb.n('ShaderNodeBlackbody')
    bb.inputs['Temperature'].default_value = 1250.0
    em = nb.emission(bb.outputs['Color'], nb.mul(glow, glow_k))
    body = nb.colscale((0.015, 0.014, 0.0135), nb.madd(nb.noise(P, scale=cell * 3.0).outputs['Fac'], 0.5, 0.75))
    col = nb.mixcol(nb.mul(cr, 0.8), body, (0.004, 0.0037, 0.0035))
    ashp = nb.mul(nb.mul(nb.sstep(0.50, 0.66, nb.noise(P, scale=cell * 0.6, detail=3.0).outputs['Fac']),
                         nb.sstep(0.2, 0.7, nz)), nb.madd(heat, 0.45, 0.30))
    col = nb.mixcol(nb.mul(ashp, nb.sub(1.0, cr)), col, (0.15, 0.145, 0.14))
    bs = nb.principled(Base_Color=col, Roughness=nb.madd(cr, 0.30, 0.62))
    bs.inputs['Specular IOR Level'].default_value = 0.30
    h = nb.sub(nb.clamp01(nb.mul(vo.outputs['Distance'], 6.0)), nb.mul(crack2, 0.4))
    nb.link(nb.bump(h, 0.4, 0.0030), bs.inputs['Normal'])
    nb.output(surface=nb.addshader(bs, em))
    return m


def mat_bark(new_material):
    m, nb = new_material('kindling')
    P = nb.texco().outputs['Object']
    g = nb.noise(nb.vmul(P, (1.0, 1.0, 1.0)), scale=300.0, detail=4.0)
    ch = nb.sstep(0.45, 0.60, nb.noise(P, scale=25.0, detail=3.0).outputs['Fac'])
    col = nb.mixcol(ch, nb.colscale((0.085, 0.066, 0.050), nb.madd(g.outputs['Fac'], 0.5, 0.75)), (0.015, 0.013, 0.012))
    bs = nb.principled(Base_Color=col, Roughness=0.82)
    nb.link(nb.bump(g.outputs['Fac'], 0.4, 0.0008), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def mat_ground(new_material, C, fr, heat_fn):
    """The trampled floor: dark peaty earth, crumbs and grit, trodden litter of dead heather; round the hearth the
    ash bed (fine powdery grey ash, charcoal dust, no pellets), a few embers alive in it when the fire burns."""
    m, nb = new_material('ground')
    P = nb.texco().outputs['Object']
    px, py, pz = nb.sep(P)
    rl = nb.math('SQRT', nb.add(nb.mul(px, px), nb.mul(py, py)))
    crumbs = nb.noise(P, scale=160.0, detail=6.0, rough=0.65)
    grit = nb.voronoi(P, scale=420.0, feature='F1')
    grr, _, _ = nb.sep(grit.outputs['Color'])
    gritm = nb.mul(nb.sstep(0.965, 0.995, grr), nb.sstep(0.30, 0.05, grit.outputs['Distance']))
    broad = nb.noise(P, scale=0.6, detail=3.0, rough=0.5)
    mid = nb.noise(P, scale=3.5, detail=4.0, rough=0.55)
    earth = nb.colscale((0.034, 0.025, 0.018), nb.mul(nb.madd(broad.outputs['Fac'], 0.25, 0.88),
                                                     nb.mul(nb.madd(mid.outputs['Fac'], 0.30, 0.85),
                                                            nb.madd(crumbs.outputs['Fac'], 0.55, 0.72))))
    # trodden litter: fine broken stems of dead heather lying every way (short, crooked, many directions)
    lit_a = nb.noise(nb.vmul(P, (1.0, 7.0, 1.0)), scale=55.0, detail=3.0, rough=0.6)
    lit_b = nb.noise(nb.vmul(P, (7.0, 1.0, 1.0)), scale=55.0, detail=3.0, rough=0.6)
    lit_c = nb.noise(nb.vmul(P, (4.0, 4.0, 1.0)), scale=40.0, detail=3.0, rough=0.6)
    litter = nb.mx(nb.mx(nb.sstep(0.66, 0.74, lit_a.outputs['Fac']), nb.sstep(0.66, 0.74, lit_b.outputs['Fac'])),
                   nb.sstep(0.70, 0.78, lit_c.outputs['Fac']))
    lmask = nb.mul(nb.sstep(0.95, 1.6, rl), nb.madd(nb.noise(P, scale=2.0).outputs['Fac'], 0.8, 0.3))
    litter = nb.mul(litter, lmask)
    col = nb.mixcol(litter, earth, (0.045, 0.036, 0.028))
    col = nb.mixcol(gritm, col, (0.055, 0.052, 0.048))
    # the ash bed round the stone (0.38 .. 0.92 m), drifting, finer and paler toward the stone
    an = nb.noise(P, scale=5.0, detail=4.0, rough=0.6)
    ashm = nb.mul(nb.sstep(0.34, 0.44, rl), nb.sstep(0.96, 0.78, nb.add(rl, nb.mul(nb.sub(an.outputs['Fac'], 0.5), 0.20))))
    ashf = nb.noise(P, scale=260.0, detail=5.0, rough=0.7)
    ashc = nb.colscale((0.105, 0.100, 0.095), nb.mul(nb.madd(ashf.outputs['Fac'], 0.45, 0.78),
                                                 nb.madd(an.outputs['Fac'], 0.5, 0.75)))
    dust = nb.sstep(0.55, 0.70, nb.noise(P, scale=90.0, detail=3.0).outputs['Fac'])
    ashc = nb.mixcol(nb.mul(dust, 0.8), ashc, (0.018, 0.016, 0.015))
    col = nb.mixcol(ashm, col, ashc)
    heat = _keyv(C, nb, 'ground_heat', heat_fn, fr)
    emb = nb.mul(nb.mul(nb.sstep(0.84, 0.92, nb.noise(P, scale=60.0, detail=2.0).outputs['Fac']), ashm), heat)
    emb = nb.mul(emb, nb.sstep(0.70, 0.45, rl))
    bb = nb.n('ShaderNodeBlackbody')
    bb.inputs['Temperature'].default_value = 1200.0
    em = nb.emission(bb.outputs['Color'], nb.mul(emb, 1.6))
    bs = nb.principled(Base_Color=col, Roughness=0.96)
    bs.inputs['Specular IOR Level'].default_value = 0.18
    h = nb.add(nb.mul(crumbs.outputs['Fac'], 1.0), nb.mul(litter, 0.8))
    h = nb.add(h, nb.mul(gritm, 0.8))
    h = nb.mixf(ashm, h, nb.mul(ashf.outputs['Fac'], 0.5))
    nb.link(nb.bump(h, 0.6, 0.004), bs.inputs['Normal'])
    nb.output(surface=nb.addshader(bs, em))
    return m


def mat_heather(new_material):
    ms = []
    m, nb = new_material('heather_stem')
    bs = nb.principled(Base_Color=(0.050, 0.040, 0.032), Roughness=0.8)
    nb.output(surface=bs)
    ms.append(m)
    m, nb = new_material('heather_leaf')
    P = nb.texco().outputs['Object']
    oi = nb.objinfo()
    n = nb.noise(P, scale=900.0, detail=3.0)
    r = oi.outputs['Random']
    col = nb.mixcol(nb.sstep(0.55, 0.85, r), (0.026, 0.032, 0.018), (0.050, 0.036, 0.028))
    col = nb.mixcol(nb.mul(nb.sstep(0.62, 0.78, n.outputs['Fac']), 0.6), col, (0.075, 0.042, 0.050))
    bs = nb.principled(Base_Color=col, Roughness=0.7)
    bs.inputs['Subsurface Weight'].default_value = 0.0
    nb.link(nb.bump(n.outputs['Fac'], 0.6, 0.001), bs.inputs['Normal'])
    nb.output(surface=bs)
    ms.append(m)
    return ms


# ------------------------------------------------------------------- build ---


# ======================================================== Blender side (numpy-free) ===

def _arr(blob, typ, e):
    import array
    off, n = e
    a = array.array(typ)
    a.frombytes(blob[off:off + 4 * n])
    return a


def _mk(C, name, e, blob, mats, smooth=True, recalc=True, link=True):
    """A mesh object from a baked entry (world coordinates, identity transform)."""
    import bmesh
    import bpy
    V = _arr(blob, 'f', e['V'])
    S = _arr(blob, 'i', e['S'])
    L = _arr(blob, 'i', e['L'])
    I = _arr(blob, 'i', e['I'])
    M = _arr(blob, 'i', e['M'])
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V) // 3)
    me.vertices.foreach_set('co', V)
    me.loops.add(len(I))
    me.loops.foreach_set('vertex_index', I)
    me.polygons.add(len(S))
    me.polygons.foreach_set('loop_start', S)
    try:
        me.polygons.foreach_set('loop_total', L)
    except Exception:
        pass
    me.polygons.foreach_set('material_index', M)
    me.update(calc_edges=True)
    for m in mats:
        me.materials.append(m)
    if recalc:                                   # consistent winding (vertex order is kept: per-frame updates hold)
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
    for k, (ent, dim) in e.get('A', {}).items():
        a = _arr(blob, 'f', ent)
        if dim == 1:
            at = me.attributes.new(k, 'FLOAT', 'POINT')
            at.data.foreach_set('value', a)
        else:
            at = me.attributes.new(k, 'FLOAT_VECTOR', 'POINT')
            at.data.foreach_set('vector', a)
    if smooth:
        me.shade_smooth()
    me.update()
    ob = bpy.data.objects.new(name, me)
    return C.link_obj(ob) if link else ob


def _cam_matrix(cam):
    from mathutils import Matrix
    C, R, U, Fw = cam[0:3], cam[3:6], cam[6:9], cam[9:12]
    return Matrix(((R[0], U[0], -Fw[0], C[0]), (R[1], U[1], -Fw[1], C[1]), (R[2], U[2], -Fw[2], C[2]), (0, 0, 0, 1)))


def _heat_fn(per):
    def h(f):
        s = per.get(f)
        if s is None:
            return 0.0
        fs = s['fire']
        if fs.get('p3', 0.0) > 0:
            return 0.85
        return float(fs.get('on', 0.0)) * min(1.0, 0.3 + 0.7 * fs.get('spread', 0.0))
    return h


def _instancer(C, name, n, P, R, S, K, coll_src):
    """Geometry Nodes instancing of a collection's children on baked points (flat arrays)."""
    import bpy
    me = bpy.data.meshes.new(name + '_pts')
    me.vertices.add(n)
    me.vertices.foreach_set('co', P)
    for nm, arr_, typ in (('rot', R, 'FLOAT_VECTOR'), ('scl', S, 'FLOAT_VECTOR'), ('var', K, 'INT')):
        a = me.attributes.new(nm, typ, 'POINT')
        a.data.foreach_set('vector' if typ == 'FLOAT_VECTOR' else 'value', arr_)
    me.update()
    ob = C.link_obj(bpy.data.objects.new(name, me))
    ng = bpy.data.node_groups.new(name + '_GN', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nt, L = ng.nodes, ng.links
    gi, go = nt.new('NodeGroupInput'), nt.new('NodeGroupOutput')
    iop = nt.new('GeometryNodeInstanceOnPoints')
    ci = nt.new('GeometryNodeCollectionInfo')
    ci.inputs['Collection'].default_value = coll_src
    ci.inputs['Separate Children'].default_value = True
    ci.inputs['Reset Children'].default_value = True
    ci.transform_space = 'ORIGINAL'
    L.new(gi.outputs[0], iop.inputs['Points'])
    L.new(ci.outputs[0], iop.inputs['Instance'])
    iop.inputs['Pick Instance'].default_value = True
    for nm, dt, sock in (('var', 'INT', 'Instance Index'), ('rot', 'FLOAT_VECTOR', 'Rotation'),
                         ('scl', 'FLOAT_VECTOR', 'Scale')):
        na = nt.new('GeometryNodeInputNamedAttribute')
        na.data_type = dt
        na.inputs['Name'].default_value = nm
        L.new(na.outputs['Attribute'], iop.inputs[sock])
    L.new(iop.outputs['Instances'], go.inputs[0])
    mod = ob.modifiers.new('inst', 'NODES')
    mod.node_group = ng
    return ob


def _static(C, new_material, G, blob, per, fr):
    import bpy
    heat = _heat_fn(per)
    _mk(C, 'ground', G['ground'], blob, [mat_ground(new_material, C, fr, heat)])
    if 'heather' in G:
        ms = mat_heather(new_material)
        src = C.collection('heather_src', hide=True)
        for k in range(6):
            for flat in (0, 1):
                nm = f'tuft{k}_{flat}'
                ob = _mk(C, nm, G[nm], blob, ms, recalc=False, link=False)
                src.objects.link(ob)
        h = G['heather']
        _instancer(C, 'heather', h['n'], _arr(blob, 'f', h['P']), _arr(blob, 'f', h['R']), _arr(blob, 'f', h['S']),
                   _arr(blob, 'i', h['K']), src)
    if 'litter' in G:
        src2 = C.collection('litter_src', hide=True)
        lm = {'peb': mat_stone(new_material, 'pebble', base=(0.075, 0.070, 0.064), lichen=0.15),
              'twig': mat_bark(new_material),
              'crumb': mat_char(new_material, C, fr, lambda f: 0.0, 'crumb', cell=240.0)}
        for k in range(10):
            nm = f'l0{k}'
            ob = _mk(C, nm, G[nm], blob, [lm[G[nm]['kind']]], link=False)
            src2.objects.link(ob)
        h = G['litter']
        _instancer(C, 'litter', h['n'], _arr(blob, 'f', h['P']), _arr(blob, 'f', h['R']), _arr(blob, 'f', h['S']),
                   _arr(blob, 'i', h['K']), src2)
    for j in range(11):
        e = G.get(f'stone{j}')
        if e is None:
            continue
        a = e['alb']
        ob = _mk(C, f'stone{j}', e, blob, [mat_stone(new_material, f'stone{j}', base=(a * 0.42, a * 0.40, a * 0.37),
                                                     lichen=0.8)])
        md = ob.modifiers.new('sub', 'SUBSURF')
        md.levels, md.render_levels = 1, 2
    ob = _mk(C, 'slab', G['slab'], blob, [mat_slab(new_material)])
    md = ob.modifiers.new('sub', 'SUBSURF')
    md.levels, md.render_levels = 1, 1
    kerb_m = mat_stone(new_material, 'kerb', base=(0.07, 0.066, 0.06), lichen=0.25, soot_fire=0.9)
    char = mat_char(new_material, C, fr, heat, 'char', cell=28.0)
    bark = mat_bark(new_material)
    chm = mat_char(new_material, C, fr, heat, 'charchunk', cell=90.0)
    coal = None
    for nm, e in G.items():
        if nm.startswith('kerb'):
            ob = _mk(C, nm, e, blob, [kerb_m])
            md = ob.modifiers.new('sub', 'SUBSURF')
            md.levels, md.render_levels = 1, 1
        elif nm.startswith('log'):
            ob = _mk(C, nm, e, blob, [char])
            md = ob.modifiers.new('sub', 'SUBSURF')
            md.levels, md.render_levels = 1, 1
        elif nm.startswith('stick'):
            _mk(C, nm, e, blob, [char if e.get('kind') == 'char' else bark])
        elif nm.startswith('chunk'):
            _mk(C, nm, e, blob, [chm])
        elif nm.startswith('coal'):
            if coal is None:
                coal = mat_char(new_material, C, fr, lambda f: 1.0 if per[f]['plate'] == 3 else 0.0, 'coal', cell=70.0, glow_k=1.6)
            ob = _mk(C, nm, e, blob, [coal])
            for f in fr:
                ob.hide_render = per[f]['plate'] != 3
                ob.keyframe_insert('hide_render', frame=f)


_DYN = []


def _figures(C, G, blob, mats):
    global _DYN
    _DYN = []
    for nm in G['_dyn']:
        e = G[nm]
        ob = _mk(C, nm, e, blob, mats)
        ob['c1'] = e['c1']
        ob['c2'] = e['c2']
        if e.get('sub'):
            md = ob.modifiers.new('sub', 'SUBSURF')
            md.levels, md.render_levels = 0, 1
        _DYN.append((ob, e['nv']))
    return _DYN


def _torch_matrix(hw, ax):
    from mathutils import Matrix, Vector
    z = Vector(ax).normalized()
    ref = Vector((0.0, 0.0, 1.0)) if abs(z.z) < 0.95 else Vector((1.0, 0.0, 0.0))
    x = ref.cross(z).normalized()
    y = z.cross(x)
    return Matrix(((x[0], y[0], z[0], hw[0]), (x[1], y[1], z[1], hw[1]), (x[2], y[2], z[2], hw[2]), (0, 0, 0, 1)))


def _flame_obj(C, name, mat, Hmax, Rmax, lean_max, glossy=True):
    """A box domain round a flame's root (z up, metres, identity scale)."""
    import bpy
    w = 2.3 * Rmax + lean_max * Hmax * 0.9 + 0.02
    z0, z1 = -0.35 * Rmax - 0.02, 1.12 * Hmax + 0.03
    V = [(-w, -w, z0), (w, -w, z0), (w, w, z0), (-w, w, z0), (-w, -w, z1), (w, -w, z1), (w, w, z1), (-w, w, z1)]
    F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.update()
    me.materials.append(mat)
    ob = C.link_obj(bpy.data.objects.new(name, me))
    for attr in ('visible_diffuse', 'visible_shadow', 'visible_transmission', 'visible_volume_scatter'):
        try:
            setattr(ob, attr, False)
        except Exception:
            pass
    ob.visible_glossy = bool(glossy)
    return ob


def _key_prop(ob, key, f, v):
    ob[key] = float(v)
    ob.keyframe_insert(f'["{key}"]', frame=f)


def _torches(C, G, blob, mats_t, flame_m, per, fr):
    """Every torch (council + crowd in view): the stave and its wound head (keyed matrices), a flame (keyed volume
    parameters) and its light."""
    import bpy
    proto = _mk(C, 'torch_proto', G['torch'], blob, mats_t, link=False)
    tm = proto.data
    f0 = fr[0]
    n_c = len(per[f0]['torches'])
    cr0 = per[f0].get('crowd')
    n_k = len(cr0['F']) if cr0 else 0

    def tdata(f, k):
        s = per[f]
        if k < n_c:
            t = s['torches'][k]
            return t['hw'], t['ax'], t['lit'], t['base'], t['lean'], t['hf'], t['seed'], t['fl']
        j = k - n_c
        c = s['crowd']
        return c['hw'][j], c['ax'][j], c['lit'][j], c['base'][j], c['lean'][j], c['hf'][j], c['seed'][j], c['fl'][j]

    ring_vis = any(per[f]['ring'][0] > 0.5 for f in fr)
    for k in range(n_c + n_k):
        ob = C.link_obj(bpy.data.objects.new(f'torch{k}', tm))
        ob.rotation_mode = 'QUATERNION'
        lits = [tdata(f, k)[2] for f in fr]
        hfs = [tdata(f, k)[5] for f in fr]
        leans = [math.hypot(*tdata(f, k)[4]) for f in fr]
        fl_ob = (_flame_obj(C, f'flame{k}', flame_m, max(hfs) * 1.40, 0.075, max(leans) * 0.9, glossy=ring_vis)
                 if max(lits) > 0.01 else None)
        light = None
        if max(lits) > 0.01:
            ld = bpy.data.lights.new(f'tl{k}', 'POINT')
            ld.color = (1.0, 0.52, 0.20)
            ld.shadow_soft_size = 0.10
            light = C.link_obj(bpy.data.objects.new(f'tl{k}', ld))
            try:
                light.visible_camera = False
            except Exception:
                pass
        for f in fr:
            hw, ax, lit, base, lean, hf, seed, flk = tdata(f, k)
            loc, rot, _ = _torch_matrix(hw, ax).decompose()
            ob.location = loc
            ob.rotation_quaternion = rot
            ob.keyframe_insert('location', frame=f)
            ob.keyframe_insert('rotation_quaternion', frame=f)
            spent = 1.0 if (per[f]['plate'] >= 2 and f >= 5205 and lit < 0.5) else 0.0
            _key_prop(ob, 'lit', f, lit)
            _key_prop(ob, 'ash', f, spent)
            _key_prop(ob, 'tseed', f, seed + f / 24.0 * 1.5)
            if fl_ob is not None:
                fl_ob.location = (base[0] - 0.06 * ax[0], base[1] - 0.06 * ax[1], base[2] - 0.06 * ax[2])
                fl_ob.keyframe_insert('location', frame=f)
                _key_prop(fl_ob, 'fH', f, hf * 1.25 * (0.92 + 0.16 * (flk - 1.0) / 0.26))
                _key_prop(fl_ob, 'fR', f, 0.072)
                _key_prop(fl_ob, 'flx', f, lean[0] * 0.9)
                _key_prop(fl_ob, 'fly', f, lean[1] * 0.9)
                _key_prop(fl_ob, 'fI', f, lit * flk)
                _key_prop(fl_ob, 'fseed', f, seed)
                fl_ob.hide_render = lit < 0.005
                fl_ob.keyframe_insert('hide_render', frame=f)
            if light is not None:
                light.location = (base[0] + 0.10 * hf * lean[0], base[1] + 0.10 * hf * lean[1], base[2] + 0.35 * hf)
                light.keyframe_insert('location', frame=f)
                light.data.energy = TORCH_W * lit * flk
                light.data.keyframe_insert('energy', frame=f)


def _hearth_fire(C, flame_m, per, fr):
    """The fire on the stone and along the logs (scene3.p2_flames / calm_flames rows: x, y, z0, H, R, sx, sy, seed):
    each row a volumetric flame; four lights in the body of the fire."""
    import bpy
    nmax = max(len(per[f]['cf']) for f in fr)
    if nmax == 0:
        return
    obs = []
    for k in range(nmax):
        rk = [per[f]['cf'][k] for f in fr if k < len(per[f]['cf'])]
        Hk = max([r[3] for r in rk] + [0.05])
        Rk = max([r[4] for r in rk] + [0.03])
        Lk = max([math.hypot(r[5], r[6]) for r in rk] + [0.1])
        obs.append(_flame_obj(C, f'hf{k}', flame_m, Hk * 1.15, Rk, Lk / 0.9, glossy=False))
    lights = []
    for k in range(4):
        ld = bpy.data.lights.new(f'hl{k}', 'POINT')
        ld.color = (1.0, 0.50, 0.18)
        ld.shadow_soft_size = 0.25
        lo = C.link_obj(bpy.data.objects.new(f'hl{k}', ld))
        try:
            lo.visible_camera = False
        except Exception:
            pass
        lights.append(lo)
    for f in fr:
        rows = per[f]['cf']
        wh = per[f]['fire'].get('white', 0.0)
        tot, cx, cy, cz = 0.0, 0.0, 0.0, 0.0
        for k, ob in enumerate(obs):
            if k < len(rows) and rows[k][3] > 0.012:
                x, y, z0, Hf, Rf, sx, sy, sd = rows[k]
                ob.location = (x, y, z0)
                _key_prop(ob, 'fH', f, Hf)
                _key_prop(ob, 'fR', f, Rf)
                _key_prop(ob, 'flx', f, sx)
                _key_prop(ob, 'fly', f, sy)
                _key_prop(ob, 'fI', f, 0.6 + 2.0 * wh)
                _key_prop(ob, 'fseed', f, sd)
                ob.hide_render = False
                w = Hf * Rf
                tot += w
                cx += w * (x + 0.3 * sx * Hf)
                cy += w * (y + 0.3 * sy * Hf)
                cz += w * (z0 + 0.35 * Hf)
            else:
                _key_prop(ob, 'fI', f, 0.0)
                ob.hide_render = True
            ob.keyframe_insert('location', frame=f)
            ob.keyframe_insert('hide_render', frame=f)
        cen = (cx / tot, cy / tot, cz / tot) if tot > 0 else (0.0, 0.0, 0.5)
        for k, lo in enumerate(lights):
            a = k * 2 * math.pi / 4 + 0.4
            lo.location = (cen[0] + 0.10 * math.cos(a), cen[1] + 0.10 * math.sin(a), cen[2] + 0.05 * (k - 1.5))
            lo.keyframe_insert('location', frame=f)
            lo.data.energy = FIRE_W * tot * (1.0 + 3.0 * wh) * _flicker(1.3 + k, f) / 4.0
            lo.data.keyframe_insert('energy', frame=f)


def _hand_matrix(HD):
    from mathutils import Matrix, Vector
    a, c, w = Vector(HD[3:6]), Vector(HD[9:12]), HD[0:3]
    y = c.cross(a)
    return Matrix(((a[0], y[0], c[0], w[0]), (a[1], y[1], c[1], w[1]), (a[2], y[2], c[2], w[2]), (0, 0, 0, 1)))


def _glove_pose(curl, thumb, spread):
    import glove as GL
    cs = list(curl) if hasattr(curl, '__len__') else [curl] * 4
    rel, fist = GL.POSES['relaxed']['thumb'], GL.POSES['fist']['thumb']
    t = min(max(thumb, 0.0), 1.0)
    th = tuple(r + (q - r) * t for r, q in zip(rel, fist))
    return {'curl': {f: (72.0 * c, 98.0 * c, 62.0 * c) for f, c in zip(GL.FING, cs)}, 'thumb': th,
            'spread': 4.0 * (spread - 0.5)}


def _gloves(C, new_material, per, fr):
    """Leather gloves (glove.py rig): each emissary's torch hand in a hammer grip round the stave (the gilded one
    crusted with gold); her right hand in P2 (to the Ring, closing on it, the fist in the fire)."""
    import glove as GL
    leather = [mat_leather_simple(new_material, f'eleather{j}', base=b) for j, b in
               enumerate(((0.030, 0.020, 0.013), (0.026, 0.019, 0.014), (0.034, 0.022, 0.015)))]
    gilt = mat_gilt(new_material, fr, C)
    f0 = fr[0]
    n_c = len(per[f0]['torches'])
    grip = _glove_pose([0.80, 0.84, 0.87, 0.90], 0.80, 0.5)
    for i in range(n_c):
        t0 = per[f0]['torches'][i]
        mat = gilt if abs(t0['HD'][14] - 1.0) < 1e-6 else leather[i % 3]
        g = GL.Glove(f'g{i}', mat, None, mirror=bool(t0['left']), sub=1 if mat is not gilt else 2, sleeve=False)
        sc = t0['HD'][12]
        g.rig.scale = (sc, -sc if t0['left'] else sc, sc)
        g.key(fr[0], grip)
        for f in fr:
            g.key_place(f, _hand_matrix(per[f]['torches'][i]['HD']))
    her_f = [f for f in fr if per[f]['her'] is not None]
    if her_f:
        hl = GL.leather_material(new_material, base=(0.024, 0.015, 0.010), rough=0.56)
        for nd in hl.node_tree.nodes:
            if nd.type == 'BSDF_PRINCIPLED':
                nd.inputs['Specular IOR Level'].default_value = 0.40
                nd.inputs['Coat Weight'].default_value = 0.08
        g = GL.Glove('her_glove', hl, None, mirror=False, sub=2, sleeve=False)
        for f in fr:
            h = per[f]['her'] or per[her_f[0] if f < her_f[0] else her_f[-1]]['her']
            g.key(f, _glove_pose(h['curl'], h['thumb'], h['spread']))
            g.key_place(f, _hand_matrix(h['HD']))
            g.mesh.hide_render = per[f]['her'] is None
            g.mesh.keyframe_insert('hide_render', frame=f)


def _ring(C, new_material, G, blob, per, fr):
    from mathutils import Vector
    ob = _mk(C, 'ring', G['ring'], blob, [mat_gold(new_material)])
    ob.rotation_mode = 'QUATERNION'
    for f in fr:
        RP = per[f]['ring']
        ob.location = RP[1:4]
        ob.rotation_quaternion = Vector(RP[4:7]).to_track_quat('Z', 'Y')
        ob.keyframe_insert('location', frame=f)
        ob.keyframe_insert('rotation_quaternion', frame=f)
        ob.hide_render = RP[0] < 0.5
        ob.keyframe_insert('hide_render', frame=f)
    if any(per[f]['ring'][0] > 0.5 for f in fr):
        import bpy
        try:
            coll = bpy.data.collections.new('ring_recv')
            coll.objects.link(ob)
            ld = bpy.data.lights.new('ringglint', 'POINT')
            ld.color = (1.0, 0.62, 0.30)
            ld.shadow_soft_size = 0.03
            lo = C.link_obj(bpy.data.objects.new('ringglint', ld))
            lo.light_linking.receiver_collection = coll
            try:
                lo.visible_camera = False
            except Exception:
                pass
            for f in fr:
                RP = per[f]['ring']
                c = Vector(RP[1:4])
                cam = Vector(per[f]['cam'][0:3])
                d = (cam - c).normalized()
                side = d.cross(Vector((0.0, 0.0, 1.0)))
                if side.length < 1e-3:
                    side = Vector((1.0, 0.0, 0.0))
                lo.location = c + d * 0.22 + side.normalized() * 0.08 + Vector((0.0, 0.0, 0.06))
                lo.keyframe_insert('location', frame=f)
                ld.energy = GLINT_W if RP[0] > 0.5 else 0.0
                ld.keyframe_insert('energy', frame=f)
        except Exception as e:
            print('ring glint (light linking) unavailable:', e, flush=True)
            try:
                bpy.data.objects.remove(bpy.data.objects['ringglint'])
            except Exception:
                pass
    return ob


def build(job):
    import bpy
    from mathutils import Vector

    from kit import core as C
    from kit.nodes import new_material
    T = job['timing']
    st = json.load(open(T['state']))
    per = {int(k): v for k, v in st['per'].items()}
    fr = sorted(per)
    G = json.load(open(T['geo']))
    with open(G['_blob'], 'rb') as fh:
        blob = fh.read()
    sc = C.setup_render(scale=job['scale'], samples=job['samples'], mblur=True, shutter=0.5)
    C.use_cycles(sc, job['samples'])
    cy = sc.cycles
    cy.use_light_tree = True
    cy.max_bounces = 4
    cy.diffuse_bounces = 1
    cy.glossy_bounces = 1
    cy.transmission_bounces = 1
    cy.volume_bounces = 0
    cy.transparent_max_bounces = 8
    cy.sample_clamp_indirect = 3.0
    cy.volume_step_rate = 1.0
    cy.volume_max_steps = 256
    try:
        cy.denoising_prefilter = 'FAST'
    except Exception:
        pass
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    try:
        sc.render.motion_blur_position = 'CENTER'
    except Exception:
        pass
    sc.frame_start, sc.frame_end = fr[0], fr[-1]
    # the night: a faint cold sky, the moon high to one side
    w = bpy.data.worlds.new('W')
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.0016, 0.0022, 0.0042, 1.0)
    bg.inputs[1].default_value = 1.0
    C.sun('moon', Vector((-0.55, 0.50, 0.67)).normalized(), (0.55, 0.66, 1.0), 0.090, angle_deg=0.6)
    # ACCORD's camera exactly: 50 deg horizontal, principal point centred
    cam = C.make_camera('CAM', hfov=50.0, clip=(0.02, 3000.0))
    cam.rotation_mode = 'QUATERNION'
    cd = cam.data
    cd.dof.use_dof = True
    cd.dof.aperture_blades = 6
    for f in fr:
        loc, rot, _ = _cam_matrix(per[f]['cam']).decompose()
        cam.location = loc
        cam.rotation_quaternion = rot
        cam.keyframe_insert('location', frame=f)
        cam.keyframe_insert('rotation_quaternion', frame=f)
        zf, amt = per[f]['focus']
        cd.dof.focus_distance = zf
        cd.dof.aperture_fstop = min(22.0, 2.8 / max(amt, 0.13))
        cd.dof.keyframe_insert('focus_distance', frame=f)
        cd.dof.keyframe_insert('aperture_fstop', frame=f)
    _static(C, new_material, G, blob, per, fr)
    fig_mats = [mat_wool(new_material, 'wool1', 'c1'), mat_wool(new_material, 'wool2', 'c2'), mat_void(new_material),
                mat_shawl(new_material), mat_leather_simple(new_material)]
    _figures(C, G, blob, fig_mats)
    flame_m = mat_flame(new_material, C, fr)
    _torches(C, G, blob, mat_torch(new_material), flame_m, per, fr)
    _hearth_fire(C, flame_m, per, fr)
    _gloves(C, new_material, per, fr)
    _ring(C, new_material, G, blob, per, fr)
    job['_figs'] = G['_figs']
    return {'expo': {str(f): per[f]['expo'] for f in fr}, 'white': {str(f): per[f]['white'] for f in fr}}


def per_frame(job, f):
    """The figures' cloth for this frame (baked in the venv): one flat float32 file, the objects in order."""
    import array
    p = job.get('_figs', '') % f
    if not _DYN or not os.path.exists(p):
        return
    a = array.array('f')
    with open(p, 'rb') as fh:
        a.frombytes(fh.read())
    off = 0
    for ob, nv in _DYN:
        n = 3 * nv
        ob.data.vertices.foreach_set('co', a[off:off + n])
        ob.data.update()
        off += n
