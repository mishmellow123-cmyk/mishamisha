"""KARST (src 1640-1679, ignition 1650).

One idea: one spark in a sea of mist. Sandstone pillars (Zhangjiajie / Guilin) rise out of a moonlit
mist sea, layer after layer into haze; pines cling to crowns and ledges. On the crown of one pillar a
figure thrusts a torch into the kindling on the beat, steps back shielding the face; the beacon flares
and its glow blooms warm in the mist wrapped around the pillar.

Camera (as the original): lateral drift x +1.6 -> -2.2 m with 0.8 m forward, kick on the hit.
"""
import json
import math
import os

SRC0, SRC1, IGN_SRC = 1640, 1679, 1650
# KARST_SLOW=1: the same shot at 2/3 speed on its own 60-frame timeline (0-59, flare 15 frames in) for v3 cut A.
# Every time-driven thing is evaluated at SOURCE time sf(f), so the slow motion is real (no repeated frames).
SLOW = bool(os.environ.get('KARST_SLOW'))
RATE = 2.0 / 3.0 if SLOW else 1.0
START, END = (0, 59) if SLOW else (SRC0, SRC1)
IGN = START + int(round((IGN_SRC - SRC0) / RATE))
SAMPLES = 96


def sf(f):
    """Source frame (1640-1679 time base) for timeline frame f."""
    return SRC0 + (f - START) * RATE
FPS = 24.0
HFOV = 40.0
CAM0 = (0.0, 0.0, 30.0)
PITCH0 = 0.5
MOON_AZ, MOON_EL, MOON_I = -58.0, 16.0, 5.5
MOON_VOL = 0.4                       # the moon's weight in the volumes (mist brightness)
MIST_TOP = -8.0
# shapes are designed on [zb, zt]; only the window the camera can see is meshed (z_lo..z_hi, spacing dz)
HERO = dict(c=(28.0, 165.0), R0=10.0, zb=-60.0, zt=26.0, seed=11.3, nt=640, dz=0.09, z_lo=-7.0, dcap=0.2, cap=0.1,
            rock=0.8)
NEAR_L = dict(c=(-24.0, 58.0), R0=8.0, zb=-60.0, zt=140.0, seed=5.1, nt=512, dz=0.1, z_lo=10.0, z_hi=54.0)
MID_ZLO = -26.0                      # the mist sea is opaque a few metres under its top (-8 +- 4.5)
# pines on the hero crown, composed by hand: (right, away-from-camera) metres from the crown centre,
# variant, scale, lean azimuth (deg, world; 0 = +x). Behind and beside the beacon, never in front.
HERO_PINES = [((4.8, 3.4), 0, 1.0, 25.0), ((-5.4, 2.6), 1, 0.85, 150.0), ((8.4, -0.4), 5, 0.95, -5.0),
              ((-8.3, -1.2), 6, 0.75, 190.0), ((1.0, 6.8), 2, 1.1, 70.0), ((6.9, 5.6), 3, 0.8, 35.0),
              ((-2.6, 7.4), 4, 0.9, 120.0)]
NPINE_NEAR, NPINE_LEDGE, NPINE_FAR, NSHRUB = 5, 3, 4, 2   # variants: crown 0-4, ledge 5-7, far (low-poly) 8-11, shrubs 12-13
FAR_TREE_D = 450.0
# (centre, R0, top) in layers: 260-500 m, 550-900 m, 1-1.4 km
MIDS = [((-70, 300), 9, 62, 21.1), ((-160, 430), 24, 70, 22.7), ((95, 360), 13, 38, 23.3),
        ((10, 470), 7, 84, 24.9), ((175, 520), 17, 105, 25.1), ((-45, 650), 28, 48, 26.3),
        ((-240, 770), 15, 125, 27.7), ((80, 780), 10, 66, 28.1), ((240, 880), 26, 140, 29.9),
        ((-110, 1000), 20, 88, 31.3), ((150, 1150), 30, 58, 32.9), ((-310, 1250), 22, 150, 33.7),
        ((40, 250), 6.0, 20, 34.1), ((-5, 360), 5.5, 46, 35.9)]
# drifting mist banks between the layers (centre, radii, density)
BANKS = [((-40, 240, -2.0), (260, 70, 13), 0.028), ((110, 410, 6.0), (320, 90, 16), 0.022),
         ((-160, 600, 12.0), (420, 120, 20), 0.018), ((150, 950, 8.0), (600, 200, 24), 0.014)]
NFAR_VAR = 10
# v3 (H5): weathered rock towers (towers.py), no vegetation. KARST_V2=1 rebuilds the accepted v2 pillars + pines.
V3 = not os.environ.get('KARST_V2')
# the mid towers keep v2's places and layers, with varied, broader proportions (no forest of columns)
MIDS3 = [((-70, 300), 14, 52, 21.1), ((-160, 430), 30, 64, 22.7), ((95, 360), 13, 36, 23.3),
         ((10, 470), 14, 62, 24.9), ((175, 520), 24, 92, 25.1), ((-45, 650), 36, 44, 26.3),
         ((-240, 770), 24, 100, 27.7), ((80, 780), 16, 60, 28.1), ((240, 880), 34, 120, 29.9),
         ((-110, 1000), 26, 80, 31.3), ((150, 1150), 40, 54, 32.9), ((-310, 1250), 28, 140, 33.7),
         ((40, 250), 9.0, 24, 34.1), ((-5, 360), 10.0, 36, 35.9)]
FAR3 = [(12.0, 60.0), (18.0, 40.0), (9.0, 75.0), (22.0, 55.0), (14.0, 30.0), (10.0, 90.0), (26.0, 45.0),
        (15.0, 70.0), (20.0, 60.0), (11.0, 50.0)]
FINISH = dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.8, streak_strength=0.0, vignette_amount=0.25)


def ftime(f):
    return f / FPS


def moon_dir(az=None, el=None):
    a, e = math.radians(MOON_AZ if az is None else az), math.radians(MOON_EL if el is None else el)
    return (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))


# ======================================================================= venv side ===

def flame_specs():
    import fireparts as FP
    specs = [FP.FlameSpec('beacon', Hf=2.3, Rb=0.34, seed=8, I=26.0, tongues=5, lean=0.25, ppm=90,
                          t_ign=ftime(IGN_SRC)),
             FP.FlameSpec('torch', Hf=0.3, Rb=0.045, seed=23, I=16.0, tongues=3, lean=0.05, ppm=200, env=False)]
    if SLOW:
        for sp in specs:
            sp.params = (lambda frame, _p=sp.params: _p(sf(frame)))
    return specs


def timing(frames):
    import fireparts as FP
    out = dict(beacon=[], torch=[], ember=[])
    for f in frames:
        t = ftime(sf(f))
        s, i, l = FP.ignite_env(t, ftime(IGN_SRC))
        out['beacon'].append((f, l * FP.flicker(t, 21)))
        out['torch'].append((f, FP.flicker(t, 5)))
        e = 0.0 if sf(f) < IGN_SRC else min(1.0, (sf(f) - IGN_SRC + 1) / 10.0) * (0.8 + 0.2 * FP.flicker(t, 3))
        out['ember'].append((f, e))
    return out


def _far_list(rng):
    """Procedural far pillars (1-5 km) inside the view wedge: (x, y, variant, scale_xy, scale_z, rot)."""
    out = []
    tries = 0
    while len(out) < 150 and tries < 5000:
        tries += 1
        d = 950 + 4200 * rng.random() ** 1.4
        b = math.radians(rng.uniform(-30, 30))
        x, y = CAM0[0] + d * math.sin(b), CAM0[1] + d * math.cos(b)
        if any(math.hypot(x - q[0], y - q[1]) < 110 + 0.05 * d for q in out):
            continue
        s_xy = rng.uniform(0.9, 2.2) * (1 + d / 4000)
        top = rng.uniform(12, 70) + rng.random() ** 3 * 150 * (d / 3000)
        out.append((x, y, int(rng.integers(0, NFAR_VAR)), s_xy, top / 60.0, rng.uniform(0, 6.283)))
    return out


def _pillar_mesh(cx, cy, R0, zb, zt, seed, nt, dz, z_lo, z_hi=None, dcap=None, lump=0.07, fine=0.1, bed=11.0,
                 stri=0.035, core=0.55, cap=0.22, rock=0.0):
    """karstgen.pillar (same shape, same lobes/seed) meshed only on the visible window [z_lo, z_hi] with
    explicit resolution (nt around, dz between rows). The crown dome is closed only when z_hi >= zt.
    Kept here so karstgen.py (and its slow-to-compile numba cache) stays untouched."""
    import numpy as np
    import karstgen as KG
    rng = np.random.default_rng(int(abs(seed) * 1000) % (2 ** 31))
    H = zt - zb
    BL = KG.lobes_for(R0, zb, zt, rng)
    prm = np.array([R0, zb, zt, lump, fine, bed, stri, core], np.float64)
    th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
    closed = z_hi is None or z_hi >= zt
    top = zt if closed else z_hi
    nz = max(8, int(math.ceil((top - z_lo) / dz)) + 1)
    z = np.linspace(z_lo, top, nz)
    TH, Z = np.meshgrid(th, z)
    R = np.empty_like(TH)
    VEG = np.empty_like(TH)
    KG._pillar_r(TH, Z, prm, BL, float(seed % 97), R, VEG)
    ox = cx + 0.03 * R0 * np.sin(Z / 31.0 + seed)
    oy = cy + 0.03 * R0 * np.cos(Z / 27.0 + 2 * seed)
    V = [np.stack([ox + R * np.cos(TH), oy + R * np.sin(TH), Z], -1).reshape(-1, 3)]
    veg = [VEG.reshape(-1)]
    nc = 0
    if closed:
        rt_top = R[-1]
        ctop = zt + cap * R0
        nc = max(6, int(math.ceil(float(rt_top.mean()) * math.pi / 2 / (dcap or dz))))
        # rocky crown: a flattish dome + broken-rock undulation (a sum of plane waves, 1.8-7 m), faded to
        # zero at the rim so it joins the wall
        rngc = np.random.default_rng(int(abs(seed) * 313) % (2 ** 31))
        waves = [(rngc.uniform(0, 2 * math.pi), 2 * math.pi / rngc.uniform(1.8, 7.0), rngc.uniform(0, 2 * math.pi),
                  rngc.uniform(0.4, 1.0)) for _ in range(7)]
        wsum = sum(w[3] for w in waves)
        for c in range(1, nc + 1):
            a = c / nc * math.pi / 2
            rr = rt_top * math.cos(a)
            zz = zt + (ctop - zt) * math.sin(a)
            bump = 0.07 * R0 * np.sin(th * 3 + seed) * np.sin(th * 5 + 2 * seed) * math.sin(a)
            px = ox[-1] + rr * np.cos(th)
            py = oy[-1] + rr * np.sin(th)
            rocky = sum(A * np.sin(k * (math.cos(d) * px + math.sin(d) * py) + ph) for d, k, ph, A in waves) / wsum
            V.append(np.stack([px, py, zz + bump + rock * rocky * math.sin(a) ** 0.5], -1))
            veg.append(np.full(nt, 1.0))
        V.append(np.array([[ox[-1, 0], oy[-1, 0], ctop + 0.04 * R0]]))
        veg.append(np.array([1.0]))
    V = np.vstack(V).astype(np.float32)
    veg = np.concatenate(veg).astype(np.float32)
    rows = nz + nc
    i = np.arange(nt)
    i2 = (i + 1) % nt
    Q = [np.stack([j * nt + i, j * nt + i2, (j + 1) * nt + i2, (j + 1) * nt + i], 1) for j in range(rows - 1)]
    if closed:
        a = (rows - 1) * nt
        Q.append(np.stack([a + i, a + i2, np.full(nt, rows * nt), np.full(nt, -1)], 1))
    return V, np.vstack(Q).astype(np.int32), veg


def _res_for(d, R0):
    """Vertex spacing ~2 px (full res) at distance d; nt a multiple of 32."""
    s = max(0.1, d * 2.0 * math.tan(math.radians(HFOV / 2)) / 1920.0 * (2.0 if d < 600 else 2.6))
    nt = int(min(max(round(2 * math.pi * R0 * 0.95 / s / 32.0), 3), 20) * 32)
    return nt, s


def _tree_points(V, veg, up, rng, dens_ledge=0.08, zmin=-1e9, zmax=1e9, max_n=4000, r_crown=1.7, r_ledge=2.6):
    """karstgen.tree_points + a crown flag: rows (x, y, z, yaw, scale, tilt_x, tilt_y, crown)."""
    import numpy as np
    cand = np.nonzero(((veg > 0.95) | ((up > 0.35) & (veg > 0.2)) | (veg > 0.55)) & (V[:, 2] > zmin)
                      & (V[:, 2] < zmax))[0]
    rng.shuffle(cand)
    cell = min(r_crown, r_ledge)
    grid = {}
    out = []
    for vi in cand:
        crown = bool(veg[vi] > 0.95)
        p = 0.5 if crown else dens_ledge * (0.5 + max(float(up[vi]), 0.0))
        if rng.random() > p * 20:
            continue
        x, y, z = (float(c) for c in V[vi])
        r = r_crown if crown else r_ledge
        key = (int(x // cell), int(y // cell), int(z // cell))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for (qx, qy, qz) in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if (qx - x) ** 2 + (qy - y) ** 2 + (qz - z) ** 2 < r * r:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        grid.setdefault(key, []).append((x, y, z))
        s = rng.uniform(0.7, 1.3) * (1.05 if crown else 0.85)
        tx, ty = rng.normal(0, 0.06 if crown else 0.12, 2)
        out.append((x, y, z - 0.2, rng.uniform(0, 6.283), s, tx, ty, crown))
        if len(out) >= max_n:
            break
    return out


def prep(frames, cache):
    import numpy as np
    import karstgen as KG
    import towers as TW
    geo = os.path.join(cache, 'geo')
    os.makedirs(geo, exist_ok=True)
    info = dict(geo=geo, pillars=[], trees=[], far=[])
    redo = bool(os.environ.get('MT3D_REKARST'))
    nverts = [0]

    def make(name, cx, cy, R0, zt, seed, nt, dz, z_lo, z_hi=None, dcap=None, zb=-60.0, dens_ledge=0.08,
             crown_trees=True, cap=0.22, rock=0.0, style=None, keep=None, tw=None):
        path = os.path.join(geo, f'{name}.bin')
        tpath = os.path.join(geo, f'{name}_trees.json')
        d = math.hypot(cx - CAM0[0], cy - CAM0[1])
        if os.path.exists(path) and os.path.exists(tpath) and not redo:
            rows = json.load(open(tpath))
            import struct
            nverts[0] += struct.unpack('<ii', open(path, 'rb').read(8))[0]
        else:
            if V3:
                V, Q, veg = TW.tower_mesh(cx, cy, R0, zb, zt, seed, nt, dz, z_lo, z_hi=z_hi, dcap=dcap,
                                          style=style, keep=keep, **(tw or {}))
            else:
                V, Q, veg = _pillar_mesh(cx, cy, R0, zb, zt, seed, nt, dz, z_lo, z_hi=z_hi, dcap=dcap, cap=cap,
                                         rock=rock)
            up = KG.upness(V, Q)
            KG.write_mesh(path, V, Q, veg if V3 else np.maximum(veg, np.clip(up, 0, 1) * 0.9))
            nverts[0] += len(V)
            tr = _tree_points(V, veg, up, np.random.default_rng(int(seed * 100)), dens_ledge=dens_ledge,
                              zmin=max(z_lo + 1.0, -4.0), zmax=(z_hi - 1.0) if z_hi else 1e9, max_n=3000)
            rng = np.random.default_rng(int(seed * 77))
            rows = []
            for (x, y, z, yaw, s, tx, ty, crown) in tr:
                if crown and not crown_trees:
                    continue
                # cliff pines lean out over the drop: yaw toward the outside of the pillar (+ jitter)
                out_yaw = math.atan2(y - cy, x - cx) + rng.normal(0, 0.35)
                if d > FAR_TREE_D:
                    var = NPINE_NEAR + NPINE_LEDGE + (int(rng.integers(0, NPINE_FAR - 1)) if crown else NPINE_FAR - 1)
                else:
                    var = int(rng.integers(0, NPINE_NEAR)) if crown else NPINE_NEAR + int(rng.integers(0, NPINE_LEDGE))
                rows.append([x, y, z, float(yaw if crown else out_yaw), s, tx, ty, var])
            json.dump(rows, open(tpath, 'w'))
        info['pillars'].append(dict(name=name, path=path))
        if not V3:
            info['trees'].extend(rows)

    h = HERO
    # the hero keeps its crown whole on the camera side (the beacon and the figure stand there)
    to_cam = math.atan2(CAM0[1] - h['c'][1], CAM0[0] - h['c'][0])
    make('hero', h['c'][0], h['c'][1], h['R0'], h['zt'], h['seed'], h['nt'], h['dz'], h['z_lo'], dcap=h['dcap'],
         zb=h['zb'], dens_ledge=0.12, crown_trees=False, cap=h['cap'], rock=h['rock'], style='stack',
         keep=(to_cam, 1.3), tw=dict(bites=False, shoulder=0.07, cap=0.1, rock=0.9, taper=0.1))
    n = NEAR_L
    make('nearL', n['c'][0], n['c'][1], n['R0'], n['zt'], n['seed'], n['nt'], n['dz'], n['z_lo'], z_hi=n['z_hi'],
         zb=n['zb'], dens_ledge=0.12)
    for k, ((cx, cy), R0, top, seed) in enumerate(MIDS3 if V3 else MIDS):
        nt, s = _res_for(math.hypot(cx, cy), R0)
        make(f'mid{k}', cx, cy, R0, top, seed, nt, s, MID_ZLO, dcap=2.0 * s, dens_ledge=0.06)
    for v in range(NFAR_VAR):
        path = os.path.join(geo, f'far{v}.bin')
        if not os.path.exists(path) or redo:
            if V3:
                R0v, topv = FAR3[v]
                V, Q, veg = TW.tower_mesh(0.0, 0.0, R0v, -60.0, topv, 100.0 + v * 7.3, 96, 0.75, -60.0, dcap=1.5)
            else:
                V, Q, veg = _pillar_mesh(0.0, 0.0, 12.0, -60.0, 60.0, 100.0 + v * 7.3, 96, 0.75, -60.0, dcap=1.5)
            KG.write_mesh(path, V, Q, veg)
        import struct
        nverts[0] += struct.unpack('<ii', open(path, 'rb').read(8))[0]
    print(f'karst geometry: {nverts[0] / 1e6:.2f} M unique pillar verts', flush=True)
    info['far'] = _far_list(np.random.default_rng(77))
    # beacon on the hero crown, on the camera side; clear the trees around it
    hc = np.array(HERO['c'])
    toward = (np.array(CAM0[:2]) - hc)
    toward /= np.linalg.norm(toward)
    side = np.array([toward[1], -toward[0]])
    b_xy = hc + toward * HERO['R0'] * (0.30 if V3 else 0.38) + side * (1.0 if V3 else 1.2)
    Vh = _load_xyz(os.path.join(geo, 'hero.bin'))

    def top_z(xy, r=1.0, dflt=None):
        near = Vh[np.hypot(Vh[:, 0] - xy[0], Vh[:, 1] - xy[1]) < r]
        return float(near[:, 2].max()) if len(near) else dflt

    bz = top_z(b_xy, 1.2, HERO['zt'] + 1.5)
    info['beacon'] = [float(b_xy[0]), float(b_xy[1]), bz]
    fig_xy = b_xy + toward * 1.25 + side * 0.9
    info['figure_root'] = [float(fig_xy[0]), float(fig_xy[1]), top_z(fig_xy, 1.0, bz)]
    clear = [(b_xy, 2.6), (fig_xy, 1.6)]

    def in_corridor(t):
        v = np.array([t[0], t[1]]) - b_xy
        along = v @ toward
        lat = abs(v @ side)
        return 0.0 < along < HERO['R0'] * 1.4 and lat < 3.5 and t[2] > HERO['zt'] - 7.0
    info['trees'] = [t for t in info['trees']
                     if all(math.hypot(t[0] - c[0], t[1] - c[1]) > r for c, r in clear) and not in_corridor(t)]
    # the hero crown's pines, composed (right = -side, away = -toward)
    for (dx, dy), var, s, az in ([] if V3 else HERO_PINES):
        xy = hc - side * dx - toward * dy
        z = top_z(xy, 0.8, HERO['zt'])
        info['trees'].append([float(xy[0]), float(xy[1]), z - 0.25, math.radians(az), s, 0.0, 0.0, var])
    # low scrub on the crown (breaks the dome's outline): dart throwing, clear of the beacon, the figure and
    # the sight line; denser toward the rim
    rs = np.random.default_rng(515)
    crown_v = Vh[Vh[:, 2] > HERO['zt'] - 0.3]
    placed = [(p[0], p[1]) for p in info['trees'][-len(HERO_PINES):]] if not V3 else []
    shrubs = []
    for _ in range(0 if V3 else 4000):
        q = crown_v[int(rs.integers(0, len(crown_v)))]
        xy = q[:2]
        rim = np.hypot(*(xy - hc)) / HERO['R0']
        if rs.random() > 0.35 + 0.65 * min(rim, 1.0):
            continue
        if np.hypot(*(xy - b_xy)) < 5.0 or np.hypot(*(xy - fig_xy)) < 3.0:
            continue
        v = xy - b_xy
        if v @ toward > -1.0 and abs(v @ side) < 3.8:
            continue
        if any(math.hypot(xy[0] - a, xy[1] - b) < 1.9 for a, b in placed + shrubs):
            continue
        shrubs.append((float(xy[0]), float(xy[1])))
        z = top_z(xy, 0.6, HERO['zt'])
        info['trees'].append([float(xy[0]), float(xy[1]), z - 0.15, float(rs.uniform(0, 6.283)),
                              float(rs.uniform(0.55, 1.0)), 0.0, 0.0, NPINE_NEAR + NPINE_LEDGE + NPINE_FAR +
                              int(rs.integers(0, NSHRUB))])
    print(f'hero crown: {len(HERO_PINES)} pines, {len(shrubs)} shrubs', flush=True)
    # the figure: face the beacon; working gestures only
    yaw = math.degrees(math.atan2(-(b_xy[0] - fig_xy[0]), b_xy[1] - fig_xy[1]))
    info['figure_yaw'] = yaw
    info.update(_figure_frames(frames, cache, np.array(info['beacon']), np.array(info['figure_root']), yaw))
    return dict(karst=info)


def _load_xyz(path):
    import struct
    import numpy as np
    with open(path, 'rb') as f:
        nv, nq = struct.unpack('<ii', f.read(8))
        return np.frombuffer(f.read(nv * 12), np.float32).reshape(nv, 3)


FIG_KEYS = [
    # crouched at the cairn, torch toward the basket
    (START, dict(crouch=0.42, lean=0.45, head_pitch=-0.35, r_flex=0.9, r_abd=0.15, r_elbow=0.5, l_flex=0.35,
                 l_abd=0.25, l_elbow=0.8, r_hip=0.25, r_knee=0.55, l_hip=0.05, l_knee=0.2, stance=0.13)),
    (1645, dict(crouch=0.45, lean=0.5, head_pitch=-0.3)),
    (1649.5, dict(crouch=0.4, lean=0.62, head_pitch=-0.25, l_flex=0.25)),
    # flare: recoil, rise, step back, forearm up to shield the face
    (1653, dict(crouch=0.15, lean=-0.18, head_pitch=0.15, head_yaw=-0.25, r_flex=0.3, r_abd=0.35, r_elbow=0.6,
                l_flex=1.25, l_abd=0.35, l_elbow=1.9, r_hip=0.05, r_knee=0.2, l_hip=-0.1, l_knee=0.1)),
    (1660, dict(crouch=0.05, lean=-0.08, head_pitch=0.1, head_yaw=-0.12, l_flex=1.05, l_elbow=1.8)),
    (END, dict(crouch=0.0, lean=0.0, head_pitch=0.05, head_yaw=0.0, r_flex=0.15, r_abd=0.12, r_elbow=0.35,
               l_flex=0.35, l_abd=0.15, l_elbow=0.6, r_hip=0.02, r_knee=0.08, l_hip=0.0, l_knee=0.06, stance=0.1)),
]


def _figure_frames(frames, cache, beacon, root, yaw):
    """SDF figure meshes (figure-local) + torch grip/dir + root offset per frame (the step back)."""
    import numpy as np
    import figures as FG
    d = os.path.join(cache, 'figure')
    os.makedirs(d, exist_ok=True)
    a = math.radians(yaw)
    ca, sa = math.cos(a), math.sin(a)

    def to_local(P):
        q = np.asarray(P, float) - root
        return np.array([ca * q[0] + sa * q[1], -sa * q[0] + ca * q[1], q[2]])

    basket = to_local(beacon + np.array([0, 0, 1.2]))
    torch, step = {}, {}
    for f in frames:
        fs = sf(f)
        p = FG.interp_pose(fs, FIG_KEYS)
        J = FG.fk(p)
        back = 0.55 * FG.ease((fs - 1650.5) / 5.0)                # step back from the flare
        u_in = FG.ease((fs - 1644.0) / 5.5) * (1 - FG.ease((fs - 1650.5) / 3.0))
        hold = np.array([0.1, 0.8, 0.55])
        hold /= np.linalg.norm(hold)
        thr = basket - J['r_sh']
        thr /= np.linalg.norm(thr)
        tdir = hold * (1 - u_in) + thr * u_in
        tdir /= np.linalg.norm(tdir)
        if u_in > 1e-4:
            FG.ik_arm(J, 'r', J['r_hand'] * (1 - u_in) + (basket - thr * 0.55) * u_in, pole=(0.5, -0.3, -0.8))
        if fs >= 1650.5:
            grip, tdir = basket - thr * 0.55 + np.array([0, back, 0]), thr   # torch left in the basket
        else:
            grip = J['r_hand'].copy()
        # shield: left hand to the front of the face while recoiling
        sh = FG.ease((fs - 1650.0) / 3.0) * (1 - FG.ease((fs - 1664.0) / 12.0))
        if sh > 1e-3:
            face = J['head'] + J['R_head'] @ np.array([0.05, 0.16, -0.02])
            FG.ik_arm(J, 'l', J['l_hand'] * (1 - sh) + face * sh, pole=(-0.6, -0.2, -0.9))
        torch[str(f)] = [list(map(float, grip)), list(map(float, tdir))]
        step[str(f)] = -back
        path = os.path.join(d, f'fig_{f:05d}.bin')
        if not os.path.exists(path) or os.environ.get('MT3D_REFIG'):
            Vv, Q, M = FG.mesh_sdf(FG.hoodie_body(J, fs / FPS), h=0.012, disp=(0.003, 8.0, 21.0, 0.0))
            FG.write_mesh(path, Vv, Q, M)
    return dict(fig_dir=d, fig_torch=torch, fig_step=step)


_SPARKS = None


def post(frame, hdr, depth, cam, scene):
    import fireparts as FP
    global _SPARKS
    t = ftime(sf(frame))
    fb = scene['fire_base']
    if sf(frame) >= IGN_SRC:
        if _SPARKS is None:
            _SPARKS = FP.ZSparks(73, (fb[0], fb[1], fb[2] + 0.4), ftime(IGN_SRC), ftime(SRC1) + 0.1, burst=120, rate=30,
                                 ember_rate=10, wind=(1.4, 0.3, 0.0), I=24.0, radius=0.3)
        s, i, l = FP.ignite_env(t, ftime(IGN_SRC))
        FP.shimmer(hdr, cam, fb, 2.3 * s, 0.34, t, amp_px=0.8 * cam.W / 1920)
        _SPARKS.render(hdr, depth, cam, t, shutter=RATE / 48.0)
    return hdr


# ==================================================================== Blender side ===

def build(job):
    import bpy
    from mathutils import Matrix, Vector

    from kit import core as C
    from kit import fire as FK
    from kit import figure as FGk
    from kit.nodes import NB, new_material

    T = job['timing']
    K = T['karst']
    opts = job.get('opts', {})
    hero_c = HERO['c']
    dist_hero = math.hypot(hero_c[0] - CAM0[0], hero_c[1] - CAM0[1])
    C.setup_render(scale=job['scale'], samples=job['samples'],
                   vol=dict(start=40.0, end=1600.0, tile=opts.get('vol_tile', 4 if job.get('final') else 8), samples=96, dist=0.75,
                            shadow_samples=24),
                   mblur=bool(opts.get('mblur', True)), shutter=0.5)
    md = moon_dir(opts.get('moon_az'), opts.get('moon_el'))
    fire_base = Vector(K['beacon']) + Vector((0, 0, 1.08))
    # --------------------------------------------------------- mist + sky
    _atmos_karst(C, NB, md, fire_base, T['beacon'])
    C.sky_world(dict(zenith=C.hexlin('#070B1C'), horizon=C.hexlin('#2B3A6A'), below=C.hexlin('#2B3A6A'),
                     moon_dir=md, moon_I=0.0, halo=(0.12, 0.16), halo2=(0.07, 0.55),
                     hglow=((0.03, 0.036, 0.05), 8.0),
                     stars=dict(gain=0.12, ext0=0.05, ext1=0.3), light_gain=1.0))
    C.sun('moon', md, C.MOON, MOON_I * opts.get('moon_gain', 1.0), angle_deg=1.2, volume=opts.get('moon_vol', MOON_VOL),
          shadow=bool(opts.get('moon_shadow', True)))
    # ------------------------------------------------------------- camera
    cam = C.make_camera('CAM', hfov=HFOV, clip=(0.5, 20000.0))

    def ease_x(u):
        u = min(max(u, 0.0), 1.0)
        return u * 0.75 + 0.25 * u * u * (3 - 2 * u)

    def kick(t, t0, amp=1.0, freq=5.0, decay=5.0):
        if t < t0:
            return 0.0
        u = t - t0
        return amp * math.exp(-decay * u) * math.sin(2 * math.pi * freq * u)

    crop = opts.get('crop')            # [cx, cy, w] full-res pixels: same pixel density, narrower view
    if crop:
        cam.data.lens = 18.0 / (math.tan(math.radians(HFOV) / 2) * crop[2] / 1920.0)
    for f in range(START - 1, END + 2):
        u = (sf(f) - SRC0) / (SRC1 - SRC0)
        x = 1.6 + (-2.2 - 1.6) * ease_x(u)
        k = kick(ftime(sf(f)), ftime(IGN_SRC))
        shift = ((crop[0] - 960.0) / crop[2], -(crop[1] - 402.0) / crop[2]) if crop else (0.0, 0.0)
        C.key_camera(cam, f, (CAM0[0] + x, CAM0[1] + 0.8 * (sf(f) - SRC0) / 40.0, CAM0[2] + 0.02 * k),
                     0.0, PITCH0 + 0.06 * k, shift=shift)
    # a valley floor far below the mist top so every downward ray is fogged (the mist sea)
    R = 20000.0
    C.mesh_obj('valley', [(-R, -R, MIST_TOP - 40.0), (R, -R, MIST_TOP - 40.0), (R, R, MIST_TOP - 40.0), (-R, R, MIST_TOP - 40.0)], [(0, 1, 2, 3)],
               mat=C.surface_mat('valley', (0.02, 0.02, 0.02), rough=1.0), smooth=False)
    # ------------------------------------------------------------ pillars
    rock = _rock_material(C, new_material)
    nv = 0
    for p in K['pillars']:
        me = _mesh_fast(p['name'], p['path'], rock)
        nv += len(me.vertices)
        C.link_obj(bpy.data.objects.new(p['name'], me))
    far_col = C.collection('FAR_SRC', hide=True)
    for v in range(NFAR_VAR):
        me = _mesh_fast(f'far{v}', os.path.join(K['geo'], f'far{v}.bin'), rock)
        nv += len(me.vertices)
        C.link_obj(bpy.data.objects.new(f'far{v}', me), far_col)
    print(f'BUILD pillars {nv / 1e6:.2f} M verts', flush=True)
    far = K['far']
    C.instancer('far_pillars', [(x, y, 0.0) for x, y, v, sxy, sz, r in far], far_col,
                rot=[(0, 0, r) for x, y, v, sxy, sz, r in far], scl=[(sxy, sxy, sz) for x, y, v, sxy, sz, r in far],
                var=[v for x, y, v, sxy, sz, r in far])
    # -------------------------------------------------------------- trees
    tree_col = C.collection('TREES_SRC', hide=True)
    pmats = _pine_materials(C, new_material) if K['trees'] else None
    nfar = NPINE_NEAR + NPINE_LEDGE + NPINE_FAR
    for v in range(nfar + NSHRUB if K['trees'] else 0):
        kind = 'shrub' if v >= nfar else ('ledge' if (NPINE_NEAR <= v < NPINE_NEAR + NPINE_LEDGE or v == nfar - 1)
                                          else 'crown')
        _pine(C, FK, f'pine{v:02d}', tree_col, pmats, seed=v * 13 + 5, kind=kind,
              lod=1 if NPINE_NEAR + NPINE_LEDGE <= v < nfar else 0)
    tr = K['trees']
    print(f'BUILD trees {len(tr)}', flush=True)
    if tr:
        C.instancer('trees', [(t[0], t[1], t[2]) for t in tr], tree_col,
                    rot=[(t[5], t[6], t[3]) for t in tr], scl=[(t[4], t[4], t[4]) for t in tr],
                    var=[int(t[7]) for t in tr])
    # ------------------------------------------------------------- beacon
    bx, by, bz = K['beacon']
    B = FK.beacon('beacon', (bx, by, bz - 0.05), seed=41, height=1.0, ember_frames=T['ember'],
                  yaw=math.radians(20))
    fb = B['fire_base']
    FK.flame_card('beacon_flame', fb, T['sprites']['beacon']['card'], T['sprites']['beacon']['first'], cam)
    # the fire lights scatter extra hard in the air (volume_factor): a warm bloom in the mist around the
    # crown without over-lighting the rock and pines
    FK.fire_lights('beacon', fb, 2.3, T['beacon'], power=2600.0, radius=0.45, cutoff=400.0,
                   volume=opts.get('fire_vol', 2.2))
    FK.smoke_plume('beacon_smoke', fb + Vector((0, 0, 0.9)), ftime(IGN_SRC), height=16.0, r0=0.35, spread=0.32,
                   rise=1.6, wind=(1.6, 0.3), dens=1.4, albedo=0.35)
    # a veil of mist wrapping the hero's upper shaft just under the crown (lit from above by the fire)
    hc = HERO['c']
    vz = HERO['zt'] + opts.get('veil_dz', -4.5)
    _mist_bank(C, FK, new_material, 'veil', (hc[0] - 4.0, hc[1] - 6.0, vz), (36.0, 26.0, 5.0),
               opts.get('veil_dens', 0.05), seed=7.7, nscale=opts.get('veil_ns', 4.0))
    # thin air around the flames: the halo
    FK.glow_haze('beacon_halo', fb + Vector((0, 0, 1.0)), 17.0, density=opts.get('halo_dens', 0.013), aniso=0.4,
                 flat=0.8, noise_amt=0.5)
    for i, (c, rad, dens) in enumerate(BANKS):
        _mist_bank(C, FK, new_material, f'bank{i}', c, rad, dens, seed=i * 3.7)
    # ------------------------------------------------------------- figure
    rx_, ry_, rz_ = K['figure_root']
    yaw = K['figure_yaw']
    mat = FGk.figure_material('climber', [(0.012, 0.012, 0.014), (0.01, 0.011, 0.015), (0.02, 0.013, 0.01),
                                          (0.03, 0.03, 0.032)], sheen=0.4, folds=0.0)
    fwd = Vector((-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0.0))
    seq = FGk.MeshSeq('fig', K['fig_dir'], mat, Matrix.Identity(4), START - 1)
    torch = FGk.torch_obj('torch')
    heads = []
    for f in range(START - 1, END + 2):
        root = Matrix.Translation(Vector((rx_, ry_, rz_)) + fwd * K['fig_step'][str(f)]) @ \
            Matrix.Rotation(math.radians(-yaw) * -1.0, 4, 'Z')
        seq.ob.matrix_world = root
        seq.ob.keyframe_insert('location', frame=f)
        seq.ob.keyframe_insert('rotation_euler', frame=f)
        grip, tdir = K['fig_torch'][str(f)]
        g = root @ Vector(grip)
        dvec = (root.to_3x3() @ Vector(tdir)).normalized()
        torch.location = g
        torch.rotation_quaternion = dvec.to_track_quat('Z', 'Y')
        torch.keyframe_insert('location', frame=f)
        torch.keyframe_insert('rotation_quaternion', frame=f)
        heads.append((f, g + dvec * 0.6))
    tf = FK.flame_card('torch_flame', heads[0][1], T['sprites']['torch']['card'], T['sprites']['torch']['first'], cam)
    tl = C.point('torch_L', heads[0][1], C.FIRE_LIGHT, 0.0, radius=0.06, cutoff=60.0)
    tt = {f: v for f, v in T['torch']}
    for f, P in heads:
        tf.location = P
        tf.keyframe_insert('location', frame=f)
        tf.scale = (1, 1, 1) if sf(f) < IGN_SRC + 1 else (0.001, 0.001, 0.001)
        tf.keyframe_insert('scale', frame=f)
        tl.location = P + Vector((0, 0, 0.25))
        tl.keyframe_insert('location', frame=f)
        C.key(tl.data, 'energy', f, 40.0 * tt.get(f, 1.0) * (1.0 if sf(f) < IGN_SRC else 0.0))
    if SLOW:
        _retime_volumes(bpy)
    return dict(fire_base=list(fb))


def _retime_volumes(bpy):
    """The kit's volume 'time' nodes are keyed to seconds of the timeline; point them at SOURCE seconds."""
    from kit import core as C
    from kit import fire as FK
    trees = [m.node_tree for m in bpy.data.materials if m.node_tree] + list(bpy.data.node_groups)
    n = 0
    for nt in trees:
        nd = nt.nodes.get('time')
        if nd is None or nd.type != 'VALUE':
            continue
        if nt.animation_data:
            nt.animation_data_clear()
        C.key_socket(nd.outputs[0], START, ftime(sf(START)))
        C.key_socket(nd.outputs[0], END, ftime(sf(END)))
        for fc in FK._fcurves(nt.animation_data.action):
            fc.extrapolation = 'LINEAR'
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
        n += 1
    print(f'BUILD retimed {n} volume clocks', flush=True)


def per_frame(job, f):
    from kit import figure as FG
    FG.swap_all(f)


def _atmos_karst(C, NB, md, fire_base, beacon_tab):
    """'ATMOS' for the karst: an exponential mist sea below h0 (noise-modulated top) whose in-scatter
    colour is shaded where the ray enters the mist top (moonlit billows + the beacon's warm glow);
    thin moonlit haze above. Output: fogged shader + T."""
    import bpy
    from mathutils import Vector
    ng = bpy.data.node_groups.new('ATMOS', 'ShaderNodeTree')
    ng.interface.new_socket('Shader', in_out='INPUT', socket_type='NodeSocketShader')
    ng.interface.new_socket('Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
    ng.interface.new_socket('T', in_out='OUTPUT', socket_type='NodeSocketFloat')
    nb = NB(ng)
    gi = nb.n('NodeGroupInput')
    go = nb.n('NodeGroupOutput')
    g = nb.geo()
    d = nb.camdata().outputs['View Distance']
    P = g.outputs['Position']
    I = g.outputs['Incoming']
    px, py, pz = nb.sep(P)
    ix, iy, iz = nb.sep(I)
    zc = nb.madd(iz, d, pz)
    H0, HS, A_DEN, U_DEN = MIST_TOP, 5.0, 0.09, 1.0 / 1700.0
    hn = nb.noise(nb.comb(nb.div(px, 1.0), nb.div(py, 1.0), 0.0), scale=1.0 / 160.0, detail=3.0, rough=0.5)
    h0 = nb.madd(nb.sub(hn.outputs['Fac'], 0.5), 9.0, H0)

    def exp_tau(Hs, a):
        ec = nb.exp(nb.mn(nb.div(nb.sub(h0, zc), Hs), 30.0))
        x = nb.math('MAXIMUM', nb.math('MINIMUM', nb.div(nb.sub(pz, zc), Hs), 30.0), -30.0)
        xs = nb.add(x, 1e-5)
        gg = nb.div(nb.sub(1.0, nb.exp(nb.mul(xs, -1.0))), xs)
        return nb.mul(nb.mul(nb.mul(ec, gg), d), a)

    tau = nb.madd(d, U_DEN, exp_tau(HS, A_DEN))
    T = nb.exp(nb.mul(tau, -1.0))
    # entry into the mist top along the view ray
    dz = nb.mul(iz, -1.0)
    t_top = nb.div(nb.sub(zc, H0), nb.mx(nb.mul(dz, -1.0), 1e-4))
    Cx = nb.madd(ix, d, px)
    Cy = nb.madd(iy, d, py)
    ex = nb.sub(Cx, nb.mul(ix, t_top))
    ey = nb.sub(Cy, nb.mul(iy, t_top))
    enters = nb.mul(nb.math('LESS_THAN', t_top, d), nb.math('GREATER_THAN', iz, 0.0))
    # moonlit billows at the entry point (directional difference toward the moon)
    mxy = Vector((md[0], md[1], 0.0)).normalized()
    b1 = nb.noise(nb.comb(ex, ey, 0.0), scale=1.0 / 85.0, detail=5.0, rough=0.55, dist=0.25)
    b2 = nb.noise(nb.comb(nb.add(ex, mxy.x * 9.0), nb.add(ey, mxy.y * 9.0), 0.0), scale=1.0 / 85.0, detail=5.0,
                  rough=0.55, dist=0.25)
    lit = nb.clamp01(nb.madd(nb.sub(b2.outputs['Fac'], b1.outputs['Fac']), 11.0, 0.45))
    thick = nb.sstep(0.3, 0.7, b1.outputs['Fac'])
    mist_base = C.MOON
    mist_col = nb.colscale(mist_base, nb.mul(nb.madd(lit, 0.9, 0.35), nb.madd(thick, 0.35, 0.75)))
    mist_col = nb.colscale(nb.mixcol(0.35, mist_col, (0.8, 0.84, 0.9)), 0.13)
    # forward scatter toward the moon brightens the mist on the moon side
    cosm = nb.mul(nb.dot(I, tuple(Vector(md).normalized())), -1.0)
    ph = nb.madd(nb.pw(nb.mx(cosm, 0.0), 5.0), 2.2, 1.0)
    mist_col = nb.colscale(mist_col, ph)
    # the beacon's glow on the mist top around the pillar (keyed intensity)
    fe = nb.value(0.0, 'fireI')
    for f, v in beacon_tab:
        C.key_socket(fe.outputs[0], f, v)
    fx, fy, fz = fire_base
    dd2 = nb.add(nb.add(nb.pw(nb.sub(ex, fx), 2.0), nb.pw(nb.sub(ey, fy), 2.0)), (fz - H0) ** 2)
    glow = nb.div(nb.mul(fe.outputs[0], 1.0), nb.add(1.0, nb.div(dd2, 22.0 ** 2)))
    mist_col = nb.vadd(mist_col, nb.colscale(C.FIRE_LIGHT, nb.mul(glow, 0.09)))
    haze_col = nb.colscale(nb.mixcol(0.5, C.MOON, C.hexlin('#2B3A6A')), nb.mul(ph, 0.15))
    col = nb.mixcol(enters, haze_col, mist_col)
    em = nb.emission(col, 1.0)
    ms = nb.mixshader(nb.sub(1.0, T), gi.outputs[0], em)
    nb.link(ms, go.inputs[0])
    nb.link(T, go.inputs[1])
    return ng


def _rock_material(C, new_material):
    m, nb = new_material('rock')
    P = nb.geo().outputs['Position']
    x, y, z = nb.sep(P)
    veg = nb.attr('veg').outputs['Fac']
    n1 = nb.noise(P, scale=0.08, detail=4.0, rough=0.55)
    streak = (nb.noise(nb.comb(nb.mul(x, 0.12), nb.mul(y, 0.12), nb.mul(z, 1.6)), scale=0.6, detail=5.0, rough=0.6)
              if V3 else
              nb.noise(nb.comb(nb.mul(x, 1.0), nb.mul(y, 1.0), nb.mul(z, 0.08)), scale=0.6, detail=5.0, rough=0.6))
    tone = nb.madd(n1.outputs['Fac'], 0.5, 0.75)
    tone = nb.mul(tone, nb.madd(nb.sstep(0.35, 0.7, streak.outputs['Fac']), -0.45, 1.0))
    rockc = nb.colscale((0.29, 0.275, 0.25), tone)
    moss = nb.sstep(0.25, 0.65, nb.add(veg, nb.mul(nb.sub(n1.outputs['Fac'], 0.5), 0.6)))
    col = nb.mixcol(moss, rockc, (0.03, 0.04, 0.028))
    fine = nb.noise(P, scale=2.2, detail=6.0, rough=0.6)
    bs = nb.principled(Base_Color=col, Roughness=0.93)
    bs.inputs['Specular IOR Level'].default_value = 0.3
    nb.link(nb.bump(nb.add(fine.outputs['Fac'], nb.mul(streak.outputs['Fac'], 0.6)), 0.5, 0.15), bs.inputs['Normal'])
    nb.output(surface=C.fogged(nb, bs))
    return m


def _mesh_fast(name, path, mat):
    """figures.write_mesh binary -> Blender mesh via foreach_set (no per-vertex Python tuples: the old
    list-of-tuples loader doubled Blender's peak RAM on million-vertex pillars). 'mat' channel -> 'veg'."""
    import array
    import struct

    import bpy
    with open(path, 'rb') as f:
        nv, nq = struct.unpack('<ii', f.read(8))
        Va = array.array('f')
        Va.frombytes(f.read(nv * 12))
        Qa = array.array('i')
        Qa.frombytes(f.read(nq * 16))
        Ma = array.array('f')
        Ma.frombytes(f.read(nv * 4))
    starts = array.array('i', [0]) * nq
    loops = array.array('i')
    n = 0
    for i in range(nq):
        starts[i] = n
        k = 4 if Qa[4 * i + 3] >= 0 else 3
        loops.extend(Qa[4 * i:4 * i + k])
        n += k
    del Qa
    me = bpy.data.meshes.new(name)
    me.vertices.add(nv)
    me.vertices.foreach_set('co', Va)
    me.loops.add(n)
    me.loops.foreach_set('vertex_index', loops)
    me.polygons.add(nq)
    me.polygons.foreach_set('loop_start', starts)
    me.update(calc_edges=True)
    me.shade_smooth()
    a = me.attributes.new('veg', 'FLOAT', 'POINT')
    a.data.foreach_set('value', Ma)
    me.materials.append(mat)
    return me


def _pine_materials(C, new_material):
    """Bark + needles. Needle clouds are dense, dark and self-shadowed: AO darkens the gaps between the
    blobs and a fine tuft bump breaks up their round shading (no white sheen: that read as popcorn)."""
    wood = C.surface_mat('pine_bark', (0.028, 0.022, 0.017), rough=0.9)
    fm, nb = new_material('pine_needles')
    P = nb.texco().outputs['Object']
    nn = nb.noise(P, scale=3.0, detail=4.0, rough=0.7)
    tuft = nb.noise(P, scale=22.0, detail=3.0, rough=0.65)
    ao = nb.n('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 0.7
    ao.samples = 8
    occ = nb.pw(ao.outputs['AO'], 1.6)
    tone = nb.mul(nb.mul(nb.madd(nn.outputs['Fac'], 0.6, 0.7), nb.madd(tuft.outputs['Fac'], 0.7, 0.65)), occ)
    bs = nb.principled(Base_Color=nb.colscale((0.011, 0.018, 0.011), tone), Roughness=0.92)
    bs.inputs['Specular IOR Level'].default_value = 0.25
    bs.inputs['Sheen Weight'].default_value = 0.12
    bs.inputs['Sheen Tint'].default_value = (0.55, 0.75, 0.55, 1.0)
    nb.link(nb.bump(tuft.outputs['Fac'], 1.0, 0.06), bs.inputs['Normal'])
    nb.output(surface=C.fogged(nb, bs))
    return wood, fm


def _pine(C, FK, name, coll, mats, seed=1, kind='crown', lod=0):
    """Huangshan cliff pine. A crooked trunk that leans out (+X) and turns back up; near-horizontal
    branches in tiers, each ending in a flat, RAGGED needle cloud built from many overlapping noisy blobs
    (one smooth pad per branch reads as a mushroom cap; too few blobs read as a dead tree). Flat top.
    kind 'crown': 3.4-5.6 m, mild lean; 'ledge': 2-3.4 m, leaning 30-55 deg out of a crack;
    'shrub': low scrub cluster (0.6-1.2 m). Base at the origin. lod 1: far trees (fewer, bigger blobs)."""
    import random

    import bmesh
    from mathutils import Vector, noise
    rng = random.Random(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
    ico_v = [v.co.copy() for v in bm.verts]
    ico_f = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    parts, clouds = [], []
    if kind == 'shrub':
        for k in range(rng.randint(2, 3)):
            a = rng.uniform(0, 6.283)
            r = rng.uniform(0.0, 0.5)
            clouds.append((Vector((math.cos(a) * r, math.sin(a) * r, 0.25 + 0.15 * k)), rng.uniform(0.55, 0.9),
                           rng.uniform(0, 6.283), 0.8))
    else:
        if kind == 'crown':
            H, lean, ntier = rng.uniform(3.4, 5.6), rng.uniform(0.12, 0.38), rng.randint(5, 6)
        else:
            H, lean, ntier = rng.uniform(2.0, 3.4), rng.uniform(0.55, 0.95), rng.randint(3, 4)
        n = 7
        pts = [Vector((0.0, 0.0, -0.45))]
        for i in range(1, n + 1):
            u = i / n
            a = lean * (1.15 - 0.75 * u) + rng.uniform(-0.1, 0.1)
            b = rng.uniform(-0.3, 0.3)
            pts.append(pts[-1] + Vector((math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)))
                       * ((H + 0.45) / n))
        rb = 0.05 + 0.03 * H
        parts.append(FK.tube(pts, rb, 6 if lod == 0 else 4, radii=[rb * (1 - 0.8 * i / n) + 0.015 for i in range(n + 1)]))
        for k in range(ntier):
            t = 0.34 + 0.56 * k / max(ntier - 1, 1) + rng.uniform(-0.04, 0.04)
            fi = t * n
            i0 = min(int(fi), n - 1)
            base = pts[i0].lerp(pts[i0 + 1], fi - i0)
            # branches favour the lean side (light, the drop) and alternate a little
            az = rng.gauss(0.0, 1.0) * 0.9 + (math.pi * rng.uniform(0.6, 1.0) if k % 2 == 1 and rng.random() < 0.6 else 0.0)
            L = H * (0.28 + 0.2 * rng.random()) * (1.2 - 0.6 * t)
            tip = base + Vector((math.cos(az) * L, math.sin(az) * L, 0.1 * L))
            parts.append(FK.tube([base, base.lerp(tip, 0.5) + Vector((0, 0, 0.1 * L)), tip], rb * 0.3 * (1.2 - t) + 0.012,
                                 4 if lod == 0 else 3, cap=False))
            clouds.append((base.lerp(tip, 0.75) + Vector((0, 0, 0.05)), H * (0.17 + 0.08 * rng.random()) * (1.25 - 0.5 * t),
                           az, 0.5))
        clouds.append((pts[-1], H * (0.18 + 0.06 * rng.random()), rng.uniform(0, 6.283), 0.5))
    Vc, Fc = [], []
    for (c, Rc, az, flat) in clouds:
        nbl = int(10 + 16 * Rc) if lod == 0 else int(4 + 4 * Rc)
        ca, sa = math.cos(az), math.sin(az)
        for j in range(nbl):
            rr = math.sqrt(rng.random())
            ph = rng.uniform(0, 2 * math.pi)
            lx, ly = rr * math.cos(ph) * 1.3 * Rc, rr * math.sin(ph) * 0.9 * Rc
            x = c.x + lx * ca - ly * sa
            y = c.y + lx * sa + ly * ca
            z = c.z + Rc * 0.22 * (1 - rr * rr) + rng.uniform(-0.08, 0.08) * Rc
            br = Rc * (0.34 + 0.18 * rng.random()) * (1.1 - 0.35 * rr) * (1.35 if lod else 1.0)
            off = Vector((rng.random() * 50, rng.random() * 50, rng.random() * 50))
            o = len(Vc)
            for q in ico_v:
                s_ = 1.0 + 0.35 * noise.noise(q * 2.2 + off)
                Vc.append((x + q.x * br * s_, y + q.y * br * s_, z + q.z * br * (flat + 0.1) * s_))
            Fc.extend(tuple(o + i for i in f) for f in ico_f)
    Vw, Fw = FK.merge(parts) if parts else ([], [])
    o = len(Vw)
    ob = C.mesh_obj(name, Vw + Vc, Fw + [tuple(i + o for i in f) for f in Fc], mat=list(mats), coll=coll)
    ob.data.polygons.foreach_set('material_index', [1 if p.vertices[0] >= o else 0 for p in ob.data.polygons])
    return ob


def _mist_bank(C, FK, new_material, name, center, radii, dens, seed=0.0, nscale=1.0):
    """A drifting bank of mist: soft flattened ellipsoid of noisy density (billowy top, feathered
    edges), lit by the moon (forward scattering) and the fire."""
    import bmesh
    import bpy
    rx, ry, rz = radii
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    V = [(v.co.x * rx, v.co.y * ry, v.co.z * rz * 1.6) for v in bm.verts]
    F = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    ob = C.mesh_obj(name, V, F, smooth=False)
    ob.location = center
    m, nb = new_material(name + '_m')
    t = FK.time_value(nb)
    o = nb.texco().outputs['Object']
    x, y, z = nb.sep(o)
    # drift with the wind (+x), slowly
    q = nb.comb(nb.sub(x, nb.mul(t, 1.8)), y, z)
    r = nb.length(nb.comb(nb.div(x, rx), nb.div(y, ry), nb.div(z, rz)))
    n1 = nb.noise(q, scale=nscale / 55.0, detail=5.0, rough=0.58, dist=0.4, dims='4D', w=nb.madd(t, 0.02 * nscale, seed))
    n2 = nb.noise(q, scale=nscale / 14.0, detail=3.0, rough=0.5, dims='4D', w=nb.madd(t, 0.05 * nscale, seed + 2.0))
    top = nb.add(nb.mul(nb.sub(n1.outputs['Fac'], 0.5), 1.4), nb.mul(nb.sub(n2.outputs['Fac'], 0.5), 0.4))
    # billowy top: height threshold perturbed by noise; feathered horizontal edges
    zn = nb.div(z, rz)
    body = nb.sstep(0.55, -0.35, nb.sub(zn, top))
    edge = nb.sstep(1.0, 0.55, r)
    dd = nb.mul(nb.mul(body, edge), nb.madd(n2.outputs['Fac'], 0.8, 0.6))
    pv = nb.n('ShaderNodeVolumePrincipled')
    nb.set(pv.inputs['Density'], nb.mul(dd, dens))
    pv.inputs['Color'].default_value = (0.9, 0.93, 1.0, 1.0)
    pv.inputs['Absorption Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    pv.inputs['Anisotropy'].default_value = 0.55
    nb.output(volume=pv)
    ob.data.materials.append(m)
    ob.visible_shadow = False
    return ob
