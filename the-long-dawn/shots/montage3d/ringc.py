"""THE RING, cut C (v3, after the H5 critic): the find (C14), the fire test (C15) and the melt (C22). Cycles.

  RING_SHOT=find_a MT3D_ENGINE=CYCLES python render.py ringc --frames 3009,3040,3060 --scale 0.5 --samples 32 --out t_fa
  RING_SHOT=fire   MT3D_ENGINE=CYCLES python render.py ringc --final --outdir renders/ring_C --range 3360-3399

C frame numbering (BIBLE_V3 locked sheet; sync = music/v3/barmap_C.json). Delivery: renders/ring_C/f_%05d.
* find_a 3000-3079: dark snow; strike 2's sparks (3009) fall into a clean melted pit and show the band on its
  glazed ice disc; she stops; her thin leather glove comes in from the SIDE, low over the snow, closes on it (3060)
  and lifts it away. Lit only by the strike: the flash, the falling sparks, one spark dying on the ice; the night
  is a faint cold glint.
* find_b 3080-3149: her own eyes: the gloved fist, palm up, opens (3081-3090); for two seconds (3090-3138) a vision
  shimmers on the band's polished inner face (forge-towers bowing toward her); the fist closes (3140).
* fire 3360-3599: the roar; her C-shaped fire-steel carries the band into her beacon (3400): it lies there on the
  tip, unmarked, letters awake, while the steel's tip goes red; the steel tips (3480), the band slides to the very
  tip and does not fall; she draws it out (3560), tips it off into her other palm and closes her fist on it.
* melt 5360-5519: out of her opened hand the band drops onto the hearth stone in the white heart of the council
  fire (a crisp fall and bounce), glows, slumps and runs into one bright bead; its letters stay letters, flare
  once (5420) and go out (5440); the fire flares to white for ACCORD's bar 70 (5519).

The band is the canonical Ring (R_in 9.4 mm, 2.3 mm thick, 5.2 mm wide) with the script of fire
(assets/ring/inscription_{outer,inner}.png), awake only in fire: deep orange-red, never white.
"""
import json
import math
import os

FPS = 24.0
SAMPLES = 96
PREP_ONLY_FRAMES = True
SHOT = os.environ.get('RING_SHOT', 'find_a')
SHOTS = dict(find_a=(3000, 3079), find_b=(3080, 3149), fire=(3360, 3599), melt=(5360, 5519))
START, END = SHOTS[SHOT]
BEAT = dict(strike2=3009, close=3060, open=3090, vision=(3090, 3138), fist=3140,
            roar=3360, steel_in=3400, tip=3480, draw=3560,
            drop=5360, flare=5420, breath=5433, out=5440)
R_IN, THICK, WIDTH = 0.0094, 0.0023, 0.0052
SQ = 2.8
HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.abspath(os.path.join(HERE, '..', '..', 'assets', 'ring'))
_F = dict(find_a=dict(exposure=1.0, bloom_strength=0.035, bloom_threshold=1.2, streak_strength=0.0,
                      vignette_amount=0.34, lift=0.002),
          find_b=dict(exposure=1.0, bloom_strength=0.035, bloom_threshold=1.2, streak_strength=0.0,
                      vignette_amount=0.34, lift=0.002),
          fire=dict(exposure=0.9, bloom_strength=0.06, bloom_threshold=1.1, streak_strength=0.0,
                    vignette_amount=0.3, lift=0.003),
          melt=dict(exposure=0.55, bloom_strength=0.03, bloom_threshold=1.8, streak_strength=0.0,
                    vignette_amount=0.3, lift=0.002))
FINISH = _F[SHOT]


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def _eout(u):
    u = min(max(u, 0.0), 1.0)
    return 1.0 - (1.0 - u) ** 3


def _ein(u):
    u = min(max(u, 0.0), 1.0)
    return u ** 3


# ======================================================================= venv side ===

def flame_specs():
    import fireparts as FP
    if SHOT == 'fire':
        return [FP.FlameSpec('bgfire', Hf=0.30, Rb=0.09, seed=31, I=26.0, tongues=6, lean=0.02, ppm=700, env=False),
                FP.FlameSpec('lick', Hf=0.070, Rb=0.020, seed=7, I=22.0, tongues=5, lean=0.004, ppm=2600,
                             env=False)]
    if SHOT == 'melt':
        return [FP.FlameSpec('bgfire', Hf=0.26, Rb=0.08, seed=43, I=26.0, tongues=6, lean=0.01, ppm=700, env=False),
                FP.FlameSpec('lick', Hf=0.060, Rb=0.018, seed=19, I=22.0, tongues=5, lean=0.003, ppm=2600,
                             env=False)]
    return []


def timing(frames):
    return {}


def _write_grid(path, V, Q, CH):
    import struct

    import numpy as np
    with open(path, 'wb') as f:
        f.write(struct.pack('<iii', len(V), len(Q), CH.shape[1]))
        f.write(np.ascontiguousarray(V, np.float32).tobytes())
        f.write(np.ascontiguousarray(Q, np.int32).tobytes())
        f.write(np.ascontiguousarray(CH, np.float32).tobytes())


def _grid_quads(nx, ny):
    import numpy as np
    i = np.arange(nx - 1)
    return np.vstack([np.stack([j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i], 1)
                      for j in range(ny - 1)]).astype(np.int32)


PIT = dict(r_top=0.036, r_disc=0.019, depth=0.012)


def _snow_pit(path):
    """Snow (20 x 20 cm, 0.42 mm grid) with the clean pit the hot Ring melted: glazed walls, a flat disc of
    refrozen meltwater at the bottom, a soft crusted rim. Channels: 0 glaze (walls), 1 ice (the disc), 2 rim."""
    import numpy as np
    n = 520
    s_ = np.linspace(-1.0, 1.0, n)
    xs = 0.10 * s_ + 0.12 * s_ ** 3                            # dense at the pit (0.4 mm), sparse at the edges
    ys = 0.02 + 0.10 * s_ + 0.14 * s_ ** 3
    X, Y = np.meshgrid(xs, ys)
    rng = np.random.default_rng(11)

    def octave(k, amp, m=7):
        ph = rng.uniform(0, 2 * np.pi, (m,))
        d = rng.uniform(0, 2 * np.pi, (m,))
        return amp * sum(np.sin(k * (np.cos(d[i]) * X + np.sin(d[i]) * Y) + ph[i]) for i in range(m)) / m

    grain = octave(700.0, 0.00022) + octave(1900.0, 0.00009) + octave(260.0, 0.00045)
    soft = octave(45.0, 0.0035, 5) + octave(110.0, 0.0012)
    ang = np.arctan2(Y, X)
    wob = 1.0 + 0.035 * np.sin(3 * ang + 0.8) + 0.02 * np.sin(5 * ang + 2.1) + 0.01 * np.sin(8 * ang + 0.3)
    R = np.hypot(X, Y) / wob
    rt, rd, dp = PIT['r_top'], PIT['r_disc'], PIT['depth']
    t = np.clip((R - rd) / (rt - rd), 0.0, 1.0)
    wall = dp * (1.0 - t * t * (3 - 2 * t)) ** 1.25            # a smooth bowl wall
    bottom = -dp + 0.0006 * (R / rd) ** 2                      # the disc: flat, a hair concave
    Z = np.where(R < rd, bottom, -wall)
    rim = 0.0014 * np.exp(-((R - rt * 1.04) / (0.18 * rt)) ** 2) * (1.0 + 0.4 * np.sin(ang * 7.0 + 1.1))
    glaze = np.clip((rt * 1.02 - R) / (0.12 * rt), 0.0, 1.0)
    ice = np.clip((rd * 1.04 - R) / (0.16 * rd), 0.0, 1.0) ** 1.5
    Z = Z + rim + soft * np.clip((R - rt) / (0.8 * rt), 0.0, 1.0) + grain * (1.0 - 0.9 * glaze)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    CH = np.stack([glaze.reshape(-1), ice.reshape(-1), np.clip(rim / 0.0014, 0, 1).reshape(-1)], 1)
    _write_grid(path, V, _grid_quads(n, n), CH)


def _vision_strip(path, u):
    """The band's vision (find_b), as an 8-bit strip mapped on the inner face over the arc the camera sees:
    forge-towers of many kinds on a low horizon against a red forge-glow, bowing toward the middle (toward her)
    as u rises 0..1. No real place, no symbol, no Eye."""
    import cv2
    import numpy as np
    W, H = 1600, 360
    img = np.zeros((H, W, 3), np.float32)
    yy = (np.arange(H) + 0.5)[:, None] / H
    hz = 0.72                                                  # horizon (0 = top edge of the band)
    glow = np.clip(1.0 - np.abs(yy - hz) / 0.55, 0, 1) ** 1.6 * (yy < hz)
    img[:] = glow[..., None] * np.array([0.10, 0.28, 0.95], np.float32)[None, None, :] * 0.9   # BGR: red-gold
    ground = (yy >= hz).astype(np.float32)
    img *= (1.0 - ground[..., None])
    mask = np.zeros((H, W), np.uint8)
    rng = np.random.default_rng(23)
    xs = np.linspace(0.08, 0.92, 9) + rng.uniform(-0.025, 0.025, 9)
    for i, x0 in enumerate(xs):
        side = (x0 - 0.5) / 0.42                               # -1 .. 1 across the arc
        h = (0.38 + 0.2 * rng.uniform()) * H * (1.0 - 0.18 * abs(side))
        w = (0.018 + 0.01 * rng.uniform()) * W
        bow = -u * math.radians(24.0 + 10.0 * rng.uniform()) * math.copysign(abs(side) ** 0.7, side)
        kind = i % 3
        prof = {0: [(0.0, 1.0), (0.6, 0.85), (0.62, 1.05), (0.7, 0.7), (1.0, 0.55)],
                1: [(0.0, 1.0), (0.4, 0.9), (0.42, 0.6), (0.85, 0.5), (0.87, 0.75), (1.0, 0.7)],
                2: [(0.0, 0.9), (0.5, 0.8), (0.75, 0.6), (0.77, 0.9), (1.0, 0.35)]}[kind]
        left, right = [], []
        for t_, k in prof:
            a = bow * max(0.0, t_ - 0.15) ** 1.3
            cx = x0 * W + h * t_ * math.sin(a) * 1.2
            cy = hz * H - h * t_ * math.cos(a)
            left.append((cx - w * k * 0.5, cy))
            right.append((cx + w * k * 0.5, cy))
        poly = np.array(left + right[::-1], np.int32)
        cv2.fillPoly(mask, [poly], 255)
        # a lit window or two, the forge's own fire
        for k in range(2):
            t_ = 0.45 + 0.3 * k
            a = bow * max(0.0, t_ - 0.15) ** 1.3
            cx = int(x0 * W + h * t_ * math.sin(a) * 1.2)
            cy = int(hz * H - h * t_ * math.cos(a))
            cv2.circle(img, (cx, cy), max(2, int(w * 0.12)), (0.2, 0.55, 1.0), -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.4)
    win = img.copy()
    img = img * (1.0 - m[..., None] * 0.97)
    img = np.maximum(img, win * m[..., None] * (win.max(axis=2, keepdims=True) > 0.5))
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    cv2.imwrite(path, np.clip(img * 255.0, 0, 255).astype(np.uint8))


def prep(frames, cache):
    d = os.path.join(cache, 'data')
    os.makedirs(d, exist_ok=True)
    info = dict(shot=SHOT, tex=os.path.join(ADIR, 'inscription_outer.png'),
                tex_in=os.path.join(ADIR, 'inscription_inner.png'))
    redo = bool(os.environ.get('MT3D_RERING'))
    if SHOT in ('find_a', 'find_b'):
        p = os.path.join(d, 'snowpit.bin')
        if not os.path.exists(p) or redo:
            _snow_pit(p)
        info['snow'] = p
    if SHOT == 'find_b':
        vd = os.path.join(d, 'vision')
        os.makedirs(vd, exist_ok=True)
        v0, v1 = BEAT['vision']
        for f in range(v0 - 3, v1 + 4):
            vp = os.path.join(vd, f'vision_{f:05d}.png')
            if not os.path.exists(vp) or redo:
                _vision_strip(vp, min(1.0, max(0.0, (f - v0) / (v1 - v0))) ** 0.8)
        info['vision'] = os.path.join(vd, f'vision_{v0 - 3:05d}.png')
    return dict(ring=info)


# ==================================================================== Blender side ===

def _sgnpow(x, p):
    return math.copysign(abs(x) ** p, x)


def band_mesh(name, mat, n_th=384, n_a=48):
    """The canonical band: superellipse profile (THICK radial x WIDTH axial) swept round the local z axis.
    UV 'strip' (u once round, CCW from +x seen from +z; v across the width, 1 = the +z edge); attributes 'outer'
    and 'inner' (how much a vertex faces away from / toward the axis). Returns (object, params) where params
    [(th, a)] lets the melt deform the very same surface (the letters ride on the molten metal)."""
    import array

    import bpy
    R_mid = R_IN + THICK * 0.5
    V = array.array('f')
    outer = array.array('f')
    inner = array.array('f')
    par = []
    for j in range(n_th):
        th = 2 * math.pi * j / n_th
        ct, st = math.cos(th), math.sin(th)
        for i in range(n_a):
            a = 2 * math.pi * i / n_a
            rr = R_mid + 0.5 * THICK * _sgnpow(math.cos(a), 2.0 / SQ)
            zz = 0.5 * WIDTH * _sgnpow(math.sin(a), 2.0 / SQ)
            V.extend((rr * ct, rr * st, zz))
            outer.append(max(math.cos(a), 0.0))
            inner.append(max(-math.cos(a), 0.0))
            par.append((th, a))
    nq = n_th * n_a
    loops = array.array('i')
    uvs = array.array('f')
    for j in range(n_th):
        j2 = (j + 1) % n_th
        for i in range(n_a):
            i2 = (i + 1) % n_a
            for (jj, ii, uj, ui) in ((j, i, j, i), (j2, i, j + 1, i), (j2, i2, j + 1, i + 1), (j, i2, j, i + 1)):
                loops.append(jj * n_a + ii)
                a = 2 * math.pi * ui / n_a
                zz = 0.5 * WIDTH * _sgnpow(math.sin(a), 2.0 / SQ)
                uvs.extend((uj / n_th, zz / WIDTH + 0.5))
    me = bpy.data.meshes.new(name)
    me.vertices.add(n_th * n_a)
    me.vertices.foreach_set('co', V)
    me.loops.add(nq * 4)
    me.loops.foreach_set('vertex_index', loops)
    me.polygons.add(nq)
    me.polygons.foreach_set('loop_start', array.array('i', range(0, nq * 4, 4)))
    me.update(calc_edges=True)
    me.shade_smooth()
    uv = me.uv_layers.new(name='strip')
    uv.data.foreach_set('uv', uvs)
    for nm, arr in (('outer', outer), ('inner', inner)):
        at = me.attributes.new(nm, 'FLOAT', 'POINT')
        at.data.foreach_set('value', arr)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.pass_index = 1
    return ob, par


def _keyv(nb, name, table):
    from kit import core as C
    v = nb.value(0.0, name)
    for f, x in table:
        C.key_socket(v.outputs[0], f, x)
    return v.outputs[0]


def _tab(fn, f0=None, f1=None):
    return [(f, fn(f)) for f in range((START if f0 is None else f0) - 2, (END if f1 is None else f1) + 3)]


def _aov(nb, name, value):
    nd = nb.n('ShaderNodeOutputAOV')
    try:
        nd.aov_name = name
    except Exception:
        nd.name = name
    nb.set(nd.inputs['Value'], value)
    return nd


def gold(new_material, R, letters=None, deep=(1.0, 0.22, 0.03), bright=(1.0, 0.5, 0.12), flare=None, hot=None,
         vision=None, rough=0.05):
    """Polished yellow gold (no glitter: roughness varies only with slow polish marks). letters: keyed emission
    strength of the script (outer + inner faces); flare: keyed 0..1 shift deep -> bright; hot: keyed incandescence
    of the metal itself; vision: (image-sequence path, keyed strength) on the inner face."""
    import bpy
    m, nb = new_material('gold_c')
    P = nb.texco().outputs['Object']
    polish = nb.noise(P, scale=160.0, detail=2.0, rough=0.5)
    bs = nb.principled(Base_Color=(1.0, 0.71, 0.29), Metallic=1.0,
                       Roughness=nb.madd(polish.outputs['Fac'], 0.045, rough - 0.005))
    bs.inputs['Specular Tint'].default_value = (1.0, 0.93, 0.78, 1.0)
    shader = bs
    uvn = nb.n('ShaderNodeUVMap')
    uvn.uv_map = 'strip'
    u_, v_, _ = nb.sep(uvn.outputs['UV'])
    emis = []
    if letters:
        cov = None
        for key, path, flip in (('outer', R['tex'], False), ('inner', R['tex_in'], True)):
            img = bpy.data.images.load(path, check_existing=True)
            img.colorspace_settings.name = 'Non-Color'
            tx = nb.n('ShaderNodeTexImage')
            tx.image = img
            tx.interpolation = 'Cubic'
            tx.extension = 'REPEAT'
            uvv = nb.comb(nb.sub(1.0, u_) if flip else u_, v_, 0.0)
            nb.link(uvv, tx.inputs['Vector'])
            c = nb.mul(tx.outputs['Color'], nb.sstep(0.25, 0.6, nb.attr(key).outputs['Fac']))
            cov = c if cov is None else nb.add(cov, c)
        col = deep if flare is None else nb.mixcol(_keyv(nb, 'flare', flare), deep, bright)
        emis.append(nb.emission(col, nb.mul(cov, _keyv(nb, 'letters', letters))))
    if hot:
        bb = nb.n('ShaderNodeBlackbody')
        bb.inputs['Temperature'].default_value = 1350.0
        emis.append(nb.emission(bb.outputs['Color'], _keyv(nb, 'hot', hot)))
    if vision:
        vpath, vtab = vision
        tx = nb.n('ShaderNodeTexImage')
        vim = bpy.data.images.load(vpath, check_existing=False)
        vim.source = 'SEQUENCE'
        vim.colorspace_settings.name = 'sRGB'
        tx.image = vim
        tx.interpolation = 'Cubic'
        tx.extension = 'CLIP'
        iu = tx.image_user
        iu.frame_start = 1
        iu.frame_duration = 100000
        iu.frame_offset = 0
        iu.use_auto_refresh = True
        # the arc the camera sees (inner u 0.30..0.70 of the strip) fills the image; a heat shimmer ripples it
        t = _keyv(nb, 'vt', [(f, f / FPS) for f in range(START - 2, END + 3)])
        sh = nb.noise(nb.comb(nb.mul(u_, 90.0), nb.mul(v_, 14.0), nb.mul(t, 2.2)), scale=1.0, detail=2.0, dims='3D')
        du = nb.mul(nb.sub(sh.outputs['Fac'], 0.5), 0.010)
        vu = nb.add(nb.div(nb.sub(u_, 0.30), 0.40), du)
        vv = nb.add(v_, nb.mul(nb.sub(sh.outputs['Fac'], 0.5), 0.02))
        nb.link(nb.comb(vu, vv, 0.0), tx.inputs['Vector'])
        face = nb.sstep(0.3, 0.8, nb.attr('inner').outputs['Fac'])
        shimmer = nb.madd(nb.sstep(0.35, 0.65, nb.noise(nb.comb(nb.mul(u_, 30.0), v_, nb.mul(t, 1.3)),
                                                          scale=1.0).outputs['Fac']), 0.35, 0.65)
        k = nb.mul(nb.mul(_keyv(nb, 'vision', vtab), face), shimmer)
        emis.append(nb.emission(tx.outputs['Color'], k))
    for e in emis:
        shader = nb.addshader(shader, e)
    nb.output(surface=shader)
    _aov(nb, 'ringmask', 1.0)
    return m


def _camera(pos, target, lens, fstop, focus=None, roll=0.0):
    import bpy
    from mathutils import Quaternion, Vector
    cd = bpy.data.cameras.new('CAM')
    cd.sensor_fit = 'HORIZONTAL'
    cd.sensor_width = 36.0
    cd.lens = lens
    cd.clip_start, cd.clip_end = 0.003, 60.0
    ob = bpy.data.objects.new('CAM', cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = pos
    ob.rotation_mode = 'QUATERNION'
    q = (Vector(target) - Vector(pos)).to_track_quat('-Z', 'Y')
    if roll:
        q = q @ Quaternion((0.0, 0.0, 1.0), math.radians(roll))
    ob.rotation_quaternion = q
    bpy.context.scene.camera = ob
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = fstop
    cd.dof.aperture_blades = 7
    cd.dof.aperture_rotation = math.radians(12.0)
    cd.dof.focus_distance = focus or (Vector(target) - Vector(pos)).length
    return ob


def _load_grid(name, path, mat):
    """ringc grid binary -> mesh with channel attributes ch0, ch1, ch2 (foreach_set: no per-vertex tuples)."""
    import array
    import struct

    import bpy
    with open(path, 'rb') as f:
        nv, nq, nch = struct.unpack('<iii', f.read(12))
        Va = array.array('f')
        Va.frombytes(f.read(nv * 12))
        Qa = array.array('i')
        Qa.frombytes(f.read(nq * 16))
        Ca = array.array('f')
        Ca.frombytes(f.read(nv * nch * 4))
    me = bpy.data.meshes.new(name)
    me.vertices.add(nv)
    me.vertices.foreach_set('co', Va)
    me.loops.add(nq * 4)
    me.loops.foreach_set('vertex_index', Qa)
    me.polygons.add(nq)
    me.polygons.foreach_set('loop_start', array.array('i', range(0, nq * 4, 4)))
    me.update(calc_edges=True)
    me.shade_smooth()
    for c in range(nch):
        at = me.attributes.new(f'ch{c}', 'FLOAT', 'POINT')
        at.data.foreach_set('value', Ca[c::nch])
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def snow_material(new_material):
    """Night snow: granular, subsurface, a few crystal facets that catch a point light (sparkle only under the
    spark, never glitter); the pit's walls glazed and smoother; the disc a dark film of refrozen meltwater."""
    m, nb = new_material('snow_c')
    P = nb.texco().outputs['Object']
    glaze = nb.attr('ch0').outputs['Fac']
    ice = nb.attr('ch1').outputs['Fac']
    grain = nb.noise(P, scale=900.0, detail=3.0, rough=0.6)
    sugar = nb.voronoi(P, scale=1700.0, feature='SMOOTH_F1')
    snow = nb.principled(Base_Color=nb.mixcol(glaze, (0.80, 0.86, 0.95), (0.62, 0.70, 0.82)),
                         Roughness=nb.mixf(glaze, nb.madd(grain.outputs['Fac'], 0.22, 0.58), 0.2))
    snow.subsurface_method = 'BURLEY'
    snow.inputs['Subsurface Weight'].default_value = 1.0
    snow.inputs['Subsurface Radius'].default_value = (1.0, 1.35, 2.1)
    snow.inputs['Subsurface Scale'].default_value = 0.005
    snow.inputs['Specular IOR Level'].default_value = 0.5
    nb.link(nb.bump(nb.add(grain.outputs['Fac'], nb.mul(sugar.outputs['Distance'], 1.5)),
                    nb.mixf(glaze, 0.45, 0.06), 0.0005), snow.inputs['Normal'])
    # crystal facets: ~3 % of 0.6 mm cells are near-mirrors with their own tilt
    cell = nb.voronoi(P, scale=1650.0, feature='F1')
    cr, cg, cb = nb.sep(cell.outputs['Color'])
    pick = nb.mul(nb.math('LESS_THAN', cr, 0.03), nb.sub(1.0, glaze))
    tilt = nb.vmath('NORMALIZE', nb.vadd(nb.geo().outputs['Normal'],
                                         nb.vscale(nb.vsub(cell.outputs['Color'], (0.5, 0.5, 0.5)), 1.4)))
    gl = nb.n('ShaderNodeBsdfGlossy')
    gl.inputs['Roughness'].default_value = 0.04
    gl.inputs['Color'].default_value = (0.9, 0.95, 1.0, 1.0)
    nb.link(tilt, gl.inputs['Normal'])
    snowx = nb.addshader(snow, nb.mixshader(nb.sub(1.0, nb.mul(pick, 0.5)), gl, nb.n('ShaderNodeBsdfTransparent')))
    # the disc: dark, clear, wet ice with a film of meltwater
    cloud = nb.noise(P, scale=140.0, detail=4.0, rough=0.6)
    wet = nb.principled(Base_Color=nb.mixcol(nb.sstep(0.35, 0.7, cloud.outputs['Fac']), (0.20, 0.27, 0.36),
                                             (0.42, 0.50, 0.60)), Roughness=0.10)
    wet.subsurface_method = 'BURLEY'
    wet.inputs['Subsurface Weight'].default_value = 1.0
    wet.inputs['Subsurface Radius'].default_value = (1.0, 1.5, 2.4)
    wet.inputs['Subsurface Scale'].default_value = 0.003
    wet.inputs['Specular IOR Level'].default_value = 0.5
    wet.inputs['Coat Weight'].default_value = 1.0
    wet.inputs['Coat Roughness'].default_value = 0.012
    wet.inputs['Coat IOR'].default_value = 1.33
    ripple = nb.noise(P, scale=90.0, detail=2.0)
    nb.link(nb.bump(ripple.outputs['Fac'], 0.25, 0.0004), wet.inputs['Coat Normal'])
    nb.output(surface=nb.mixshader(ice, snowx, wet))
    return m


def _spark_mat(new_material, name, table):
    m, nb = new_material(name)
    nb.output(surface=nb.emission((1.0, 0.42, 0.10), _keyv(nb, 'g', table)))
    return m


def _sparks(C, new_material, paths, frames):
    """Short orange flint sparks: tiny incandescent beads that fall and curve through the frame (a quadratic
    path from entry through a bend to the landing, accelerating), streaked by Cycles' motion blur; they cool
    (bright -> red -> out) or die where they land. paths: (entry, bend, land, f_entry, f_land, glow_after_landing)."""
    from mathutils import Vector
    obs = []
    for k, (S, Q, L, f0, f1, land) in enumerate(paths):
        S, Q, L = Vector(S), Vector(Q), Vector(L)

        def at(f, S=S, Q=Q, L=L, f0=f0, f1=f1):
            u = min(1.0, max(0.0, (f - f0) / float(f1 - f0)))
            u = 0.55 * u + 0.45 * u * u                        # it accelerates as it falls
            return S * (1 - u) ** 2 + Q * 2 * u * (1 - u) + L * u * u

        def glow(f, f0=f0, f1=f1, land=land):
            if f < f0 - 1:
                return 0.0
            if f < f1:
                return 1400.0 * (0.55 + 0.45 * math.exp(-(f - f0) / 3.0))
            b_ = f - f1
            if not land or b_ >= land:
                return 0.0
            return 260.0 * math.exp(-b_ / (land * 0.38)) * (1.0 + 0.12 * math.sin(b_ * 1.9))

        tab = [(f, glow(f)) for f in range(frames[0] - 2, frames[1] + 3)]
        m = _spark_mat(new_material, f'spark{k}', tab)
        bm_v, bm_f = _uvsphere(0.00026, 8, 6)
        ob = C.mesh_obj(f'spark{k}', bm_v, bm_f, mat=m)
        ob.visible_shadow = False
        for f in range(frames[0] - 2, frames[1] + 3):
            alive = f >= f0 - 1 and (f < f1 or (land and f - f1 < land + 1))
            ob.location = at(f) if alive else S + Vector((0.0, 0.0, 3.0))
            ob.keyframe_insert('location', frame=f)
        obs.append((ob, L, f1, land))
    return obs


def _uvsphere(r, nu, nv):
    V, F = [], []
    for j in range(1, nv):
        ph = math.pi * j / nv
        for i in range(nu):
            th = 2 * math.pi * i / nu
            V.append((r * math.sin(ph) * math.cos(th), r * math.sin(ph) * math.sin(th), r * math.cos(ph)))
    top, bot = len(V), len(V) + 1
    V += [(0, 0, r), (0, 0, -r)]
    for j in range(nv - 2):
        for i in range(nu):
            i2 = (i + 1) % nu
            F.append((j * nu + i, j * nu + i2, (j + 1) * nu + i2, (j + 1) * nu + i))
    for i in range(nu):
        F.append((top, (i + 1) % nu, i))
        F.append((bot, (nv - 2) * nu + i, (nv - 2) * nu + (i + 1) % nu))
    return V, F


def _world(bg_col, strength):
    import bpy
    w = bpy.data.worlds.new('W')
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs[0].default_value = tuple(bg_col) + (1.0,)
    bg.inputs[1].default_value = strength
    return bg


def build(job):
    import bpy

    from kit import core as C
    from kit.nodes import new_material
    T = job['timing']
    R = T['ring']
    opts = job.get('opts', {})
    C.setup_render(scale=job['scale'], samples=job['samples'], mblur=True, shutter=0.5, light_threshold=1e-6,
                   shadow_pool='256')
    sc = bpy.context.scene
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    vl = sc.view_layers[0]
    try:
        a = vl.aovs.add()
        a.name = 'ringmask'
        a.type = 'VALUE'
    except Exception as e:
        print('AOV unavailable:', e, flush=True)
    if sc.render.engine == 'CYCLES':
        sc.cycles.light_sampling_threshold = 0.0
        sc.cycles.use_light_tree = True
    dict(find_a=_find_a, find_b=_find_b, fire=_fire, melt=_melt)[SHOT](C, new_material, R, opts, T)
    return {}


_PER_FRAME = []


def per_frame(job, f):
    for fn in _PER_FRAME:
        fn(f)


# ------------------------------------------------------------------------ C14 (a) ---

def _find_a(C, new_material, R, opts, T):
    """THE FIND (a), 3000-3079. The pit at the origin (snow surface z ~ 0, disc at -PIT depth)."""
    import bpy
    from mathutils import Euler, Matrix, Vector

    import glove as GL
    _world(opts.get('sky', (0.10, 0.16, 0.36)), opts.get('sky_w', 0.04))
    snow = _load_grid('snow', R['snow'], snow_material(new_material))
    snow.visible_shadow = True
    g = gold(new_material, R)
    ring, _ = band_mesh('ring', g)
    ring.rotation_mode = 'QUATERNION'
    z_disc = -PIT['depth'] + 0.0002
    tilt = math.radians(7.0)
    rest = (Matrix.Translation(Vector((0.002, 0.004, z_disc + 0.5 * WIDTH * math.cos(tilt) +
                                       (R_IN + THICK) * math.sin(tilt) * 0.0)))
            @ Euler((tilt, 0.0, math.radians(38.0))).to_matrix().to_4x4())
    # her left glove: from the left, low over the snow, palm down; closes on the band (3060) and lifts it
    leather = GL.leather_material(new_material, base=opts.get('leather', (0.075, 0.047, 0.029)))
    G = GL.Glove('glove', leather, GL.wool_material(new_material), mirror=True)
    close = BEAT['close']

    # a low scoop from the side: the hand skims in over the snow (palm just above it), the fingers reach down
    # into the hollow past the band, curl and draw it into the fist, and the fist lifts away to the left
    P0, P1, P2 = Vector((-0.40, 0.004, 0.026)), Vector((-0.100, 0.006, 0.024)), Vector((-0.094, 0.006, 0.021))
    P3 = Vector((-0.27, -0.004, 0.068))

    def place(f):
        yaw, pitch, roll = math.radians(-2.0), math.radians(4.0), math.radians(-5.0)
        if f <= 3033:
            p = P0
        elif f <= 3050:
            u = _eout((f - 3033) / 17.0)
            p = P0.lerp(P1, u)
        elif f <= close:
            u = _ease((f - 3050) / 10.0)
            p = P1.lerp(P2, u)
        else:
            u = _ein((f - close) / 21.0) ** 0.75
            p = P2.lerp(P3, u)
            pitch -= math.radians(10.0) * u
            roll += math.radians(-8.0) * u
        return Matrix.Translation(p) @ Euler((roll, pitch, yaw), 'XYZ').to_matrix().to_4x4()

    into = {'curl': {'index': (32, 16, 8), 'middle': (34, 18, 8), 'ring': (36, 20, 10), 'little': (40, 24, 12)},
            'thumb': (4, 6, 8, 6), 'spread': 7.0}
    glide = GL.pose_mix(GL.POSES['reach'], GL.POSES['flat'], 0.3)
    glide['spread'] = 9.0

    def pose(f):
        if f < 3042:
            return glide
        if f < 3050:
            return GL.pose_mix(glide, into, _ease((f - 3042) / 8.0))
        return GL.pose_mix(into, GL.POSES['fist'], _ease((f - 3050) / 10.0))

    for f in range(START - 2, END + 3):
        G.key_place(f, place(f))
        G.key(f, pose(f))
    # the band: at rest until the curling fingers reach it (3054), drawn up the wall into the fist's hollow by
    # the close (3060), then it rides the hand
    sc = bpy.context.scene
    sc.frame_set(close)
    pb = G.rig.pose.bones
    hol = (pb['m1'].head + pb['m2'].head + pb['m3'].head + pb['m3'].tail) / 4.0
    hol = Vector((hol.x, -hol.y, hol.z))                     # mirrored hand: armature space y is flipped
    Pc = place(close)
    in_fist = Pc @ Matrix.Translation(hol) @ Euler((math.radians(75.0), 0.0, math.radians(15.0))).to_matrix().to_4x4()
    Pc_inv = Pc.inverted()
    l0, r0, _ = rest.decompose()
    l1, r1, _ = in_fist.decompose()
    for f in range(START - 2, END + 3):
        if f <= close - 5:
            M = rest
        elif f <= close:
            u = _ease((f - (close - 5)) / 5.0)
            lift = Vector((0.0, 0.0, 0.006 * math.sin(math.pi * u)))
            M = Matrix.Translation(l0.lerp(l1, u) + lift) @ r0.slerp(r1, u).to_matrix().to_4x4()
        else:
            M = place(f) @ Pc_inv @ in_fist
        l, r, _ = M.decompose()
        ring.location = l
        ring.rotation_quaternion = r
        ring.keyframe_insert('location', frame=f)
        ring.keyframe_insert('rotation_quaternion', frame=f)
    print('fist hollow', tuple(round(c, 4) for c in l1), 'ring rest', tuple(round(c, 4) for c in l0), flush=True)
    # strike 2: a flash up at the basket (off frame, above left) and short sparks falling into the frame
    s2 = BEAT['strike2']
    fl = C.point('flash', (-0.30, 0.16, 0.62), (1.0, 0.52, 0.2), 0.0, radius=0.012)
    for f in range(START - 2, END + 3):
        e = 0.0 if f < s2 else 5.0 * math.exp(-(f - s2) / 1.6)
        C.key(fl.data, 'energy', f, e)
    zd = -PIT['depth'] + 0.0003
    paths = [((-0.052, 0.026, 0.036), (-0.030, 0.012, 0.030), (-0.0115, -0.0065, zd), s2 + 2, s2 + 6, 40),
             ((-0.060, 0.040, 0.040), (-0.020, 0.052, 0.034), (0.030, 0.056, 0.0015), s2 + 2, s2 + 7, 3),
             ((-0.066, 0.010, 0.038), (-0.058, -0.004, 0.024), (-0.046, -0.022, 0.0015), s2 + 3, s2 + 7, 2),
             ((-0.040, 0.060, 0.042), (0.010, 0.074, 0.040), (0.050, 0.086, 0.0015), s2 + 3, s2 + 9, 3),
             ((-0.056, 0.030, 0.044), (-0.040, 0.024, 0.036), (-0.030, 0.020, 0.030), s2 + 2, s2 + 5, 0)]
    sp = _sparks(C, new_material, paths, (START, END))
    # the spark that lands on the ice beside the band lights the pit while it dies
    ob, L, f1, land = sp[0]
    Le = C.point('ember', L + Vector((0, 0, 0.0012)), (1.0, 0.34, 0.07), 0.0, radius=0.0005)
    for f in range(START - 2, END + 3):
        b_ = f - f1
        e = 0.0 if b_ < 0 or b_ >= land else opts.get('ember_w', 0.0022) * math.exp(-b_ / (land * 0.38)) * (
            1.0 + 0.12 * math.sin(b_ * 1.9))
        C.key(Le.data, 'energy', f, e)
    # the night: a faint cold moon from behind right (form and a glint, not a fill)
    C.sun('moon', opts.get('moon_dir', (0.25, 0.88, 0.40)), (0.55, 0.66, 1.0), opts.get('moon', 0.26), angle_deg=1.5)
    if opts.get('debug_light'):
        C.sun('dbg', (-0.3, -0.6, 0.7), (1, 1, 1), opts['debug_light'], angle_deg=5.0)
    cp = Vector(opts.get('cam', [-0.030, -0.292, 0.157]))
    tgt = Vector(opts.get('tgt', [-0.006, 0.008, -0.004]))
    cam = _camera(cp, tgt, opts.get('lens', 70.0), opts.get('fstop', 13.0),
                  focus=(cp - Vector((0.002, 0.004, z_disc + 0.003))).length)
    # a slow push-in over the whole shot (5 %): the find draws us in
    for f in range(START - 2, END + 3):
        u = (f - START) / float(END - START)
        cam.location = cp.lerp(tgt, 0.05 * (0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, u)))))
        cam.keyframe_insert('location', frame=f)


# ------------------------------------------------------------------------ C14 (b) ---

def _find_b(C, new_material, R, opts, T):
    raise NotImplementedError


def _fire(C, new_material, R, opts, T):
    raise NotImplementedError


def _melt(C, new_material, R, opts, T):
    raise NotImplementedError
