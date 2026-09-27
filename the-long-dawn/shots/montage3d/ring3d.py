"""RING BAKE-OFF (v3 cut C, BIBLE_V3 REVISION 1): three macro stills of the Ring in Blender (EEVEE).

  RING_SHOT=find|fire|melt python render.py ring3d --frames 100 --scale 0.5 --samples 64 --out t_ring_find

* find  (C15): a gold band in a small melted hollow in snow, lit by a flint spark; a gloved hand about to close.
* fire  (C16): the band hanging on the tip of her steel over the beacon's coals, unmarked, its letters awake.
* melt  (C24): the band slumping into a bead in the white heart of the fire, its letters flaring once.

The band is an analytic surface (superellipse profile swept round the axis, 512 x 64, smooth normals):
a polished band only looks right if its reflections are unbroken, so no SDF/voxel mesh on the Ring
itself. The inscription is ring.py's invented broad-nib script (no real alphabet, no real text),
mapped once round the outer face; it glows only in fire (emission), the metal stays unmarked.
Reflections: EEVEE ray tracing (full-res screen trace) + a sphere probe at the Ring for everything
off screen. World units: metres (the band is 23.4 mm across).
"""
import json
import math
import os

SAMPLES = 96
FPS = 24.0
SHOT = os.environ.get('RING_SHOT', 'find')
# animated inserts on cut C's own timeline (BIBLE_V3 REVISION 1 LOCKED BEAT SHEETS; bar n starts at (n-1)*80)
SHOTS = dict(find_a=(3000, 3079), find_b=(3080, 3149), fire=(3360, 3599), melt=(5360, 5519),
             find=(100, 100), fire_still=(100, 100), melt_still=(100, 100))
START, END = SHOTS.get(SHOT, (100, 100))
IGN = START - 10
BEAT = dict(strike2=3009, close=3060, open=3090, vision=(3090, 3138), fist=3140,          # C14
            roar=3360, steel_in=3400, tip=3480, draw=3560,                               # C15
            drop=5360, flare=5420, out=5440, white=5470)                                  # C22
R_IN, THICK, WIDTH = 0.0094, 0.0023, 0.0052      # inner radius, radial thickness, band width (m)
SQ = 2.8                                          # profile superellipse exponent (comfort-fit, soft edges)
_F = dict(find_a=dict(exposure=0.9, bloom_strength=0.05, bloom_threshold=1.0, streak_strength=0.0,
                      vignette_amount=0.32),
          find_b=dict(exposure=0.95, bloom_strength=0.05, bloom_threshold=1.0, streak_strength=0.0,
                      vignette_amount=0.32),
          fire=dict(exposure=1.0, bloom_strength=0.07, bloom_threshold=1.0, streak_strength=0.0, vignette_amount=0.3),
          melt=dict(exposure=0.62, bloom_strength=0.035, bloom_threshold=1.6, streak_strength=0.0,
                    vignette_amount=0.28))
FINISH = _F.get(SHOT) or dict(find=dict(exposure=0.82, bloom_strength=0.06, bloom_threshold=0.9, streak_strength=0.0,
                        vignette_amount=0.3),
              fire=dict(exposure=1.0, bloom_strength=0.08, bloom_threshold=0.9, streak_strength=0.0,
                        vignette_amount=0.3),
              melt=dict(exposure=0.75, bloom_strength=0.05, bloom_threshold=1.2, streak_strength=0.0,
                        vignette_amount=0.25)).get(SHOT.replace('_still', ''), {})


def ftime(f):
    return f / FPS


# ======================================================================= venv side ===

def flame_specs():
    import fireparts as FP
    if SHOT.startswith('find'):
        return []
    return [FP.FlameSpec('bgfire', Hf=0.22, Rb=0.07, seed=31, I=26.0, tongues=6, lean=0.02, ppm=900, env=False),
            FP.FlameSpec('lick', Hf=0.045, Rb=0.014, seed=7, I=22.0, tongues=4, lean=0.004, ppm=5000, env=False)]


def timing(frames):
    return {}


def _write_quads(path, V, Q, CH):
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


def _snow(path, hollow=(0.0, 0.0), r=0.019, depth=0.0075):
    """Snow surface (16 x 16 cm, 0.4 mm grid): granular crust + a melted hollow with a raised, crusted
    rim and a glazed (refrozen) lining. Channel 0 = glaze (0..1), channel 1 = rim."""
    import numpy as np
    n = 400
    xs = np.linspace(-0.07, 0.09, n)
    ys = np.linspace(-0.06, 0.10, n)
    X, Y = np.meshgrid(xs, ys)
    rng = np.random.default_rng(3)

    def octave(k, amp):
        ph = rng.uniform(0, 2 * np.pi, (6,))
        d = rng.uniform(0, 2 * np.pi, (6,))
        return amp * sum(np.sin(k * (np.cos(d[i]) * X + np.sin(d[i]) * Y) + ph[i]) for i in range(6)) / 6.0

    Z = octave(90.0, 0.0012) + octave(260.0, 0.0005) + octave(700.0, 0.00022) + octave(1900.0, 0.00008)
    Z += 0.004 * np.sin(X * 23.0 + 0.6) * np.sin(Y * 17.0 + 1.0)
    ang = np.arctan2(Y - hollow[1], X - hollow[0])
    r_eff = r * (1.0 + 0.1 * np.sin(3 * ang + 1.0) + 0.06 * np.sin(5 * ang + 2.3) + 0.04 * np.sin(9 * ang + 0.4))
    R = np.hypot(X - hollow[0], (Y - hollow[1]) * 1.08)
    u = np.clip(R / r_eff, 0.0, 1.6)
    bowl = -depth * np.where(u < 1.0, (1.0 - u * u) ** 1.6, 0.0)
    rim = 0.0016 * np.exp(-((u - 1.02) / 0.16) ** 2) * (1.0 + 0.35 * np.sin(np.arctan2(Y, X) * 5.0 + 1.3))
    glaze = np.clip(1.15 - u, 0.0, 1.0) ** 0.7
    Z = Z * (1.0 - 0.85 * glaze) + bowl + rim
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    CH = np.stack([glaze.reshape(-1), np.clip(rim / 0.0016, 0, 1).reshape(-1)], 1)
    _write_quads(path, V, _grid_quads(n, n), CH)
    return float(np.interp(0.0, [0.0], [bowl.min()]))


FIST = (1.15, 1.55, 1.2)
REACH = [(0.25, 0.45, 0.35), (0.35, 0.55, 0.4), (0.5, 0.7, 0.5), (0.7, 0.85, 0.6)]
OPEN = [(0.04, 0.08, 0.05), (0.03, 0.07, 0.05), (0.06, 0.09, 0.06), (0.1, 0.12, 0.08)]
GRIP = (0.123, 0.004, -0.03)            # hand-local point the closed fingers wrap round


def _glove(path, curl=0.0, base=None):
    """A knitted glove reaching to close on something small: palm + four curling fingers + thumb, one
    smooth SDF body (figures.mesh_sdf). Hand-local: wrist at the origin, fingers toward +x, palm toward
    -z, thumb toward +y."""
    import numpy as np
    import figures as FG
    V_, cone, ell = FG.V, FG.cone, FG.ell
    P = [ell(V_(0.055, 0.0, 0.0), (0.05, 0.043, 0.0155), k=0.0, mat=0),
         cone(V_(0.0, 0.0, 0.0), V_(0.04, 0.0, 0.0), 0.028, 0.034, k=0.02, mat=0)]
    base = base or REACH
    fl = [tuple(b + curl * (f_ - b) for b, f_ in zip(base[k], FIST)) for k in range(4)]
    fingers = [(0.024, 0.0115, (0.042, 0.027, 0.021), fl[0]),
               (0.008, 0.012, (0.046, 0.030, 0.022), fl[1]),
               (-0.008, 0.0115, (0.044, 0.028, 0.021), fl[2]),
               (-0.023, 0.0102, (0.036, 0.022, 0.019), fl[3])]
    tips = []
    for (y0, rad, L, fl) in fingers:
        p = V_(0.1, y0, -0.002)
        a = 0.0
        for seg in range(3):
            a += fl[seg]
            d = V_(math.cos(a), 0.0, -math.sin(a))
            q = p + d * L[seg]
            P.append(cone(p, q, rad * (1.0 - 0.07 * seg), rad * (0.93 - 0.07 * seg), k=0.006, mat=0))
            p = q
        tips.append(list(map(float, p)))
    # thumb: from the palm's side, opposed, reaching under
    b = V_(0.035, 0.036, -0.008)
    t1 = b + V_(0.03, 0.022 - 0.012 * curl, -0.02 - 0.006 * curl)
    t2 = t1 + V_(0.026 - 0.004 * curl, -0.004 - 0.02 * curl, -0.022 + 0.006 * curl)
    P.append(cone(b, t1, 0.017, 0.0135, k=0.01, mat=0))
    P.append(cone(t1, t2, 0.0135, 0.0118, k=0.006, mat=0))
    Vv, Q, M = FG.mesh_sdf(P, h=0.0007, disp=(0.00035, 180.0, 420.0, 0.0), pad=0.004)
    FG.write_mesh(path, Vv, Q, M)
    import json
    json.dump(dict(tips=tips, tip_c=list(map(float, np.mean(np.array(tips[:2]), 0)))), open(path + '.json', 'w'))


def _melt_mesh(path, m=0.85, bead_deg=-120.0):
    """C24's slumped band as one smooth SDF: the far arc still standing but sagging and necking toward
    a flattened molten drop on the floor (the band's volume flows into it). Ring-local: axis up, floor
    at z = -WIDTH/2."""
    import numpy as np
    import figures as FG
    R_mid = R_IN + THICK * 0.5
    z0 = -0.5 * WIDTH
    bd = math.radians(bead_deg)
    P = []
    n = 54
    for j in range(n):
        th = bd + 2 * math.pi * (j + 0.5) / n
        d = math.atan2(math.sin(th - bd), math.cos(th - bd))
        phi = abs(d) / math.pi                               # 0 at the drop .. 1 opposite
        if phi < 0.3 * m:
            continue                                         # this part has run into the drop
        k = min(1.0, (phi - 0.3 * m) / 0.35)                  # necking toward the drop
        thin = 0.55 + 0.45 * k
        sag = m * (0.0022 * (1 - phi) + 0.0009)
        c = np.array([R_mid * math.cos(th), R_mid * math.sin(th), -sag])
        c[2] = max(c[2], z0 + 0.5 * WIDTH * thin)
        tng = np.array([-math.sin(th), math.cos(th), 0.0])
        rad = np.array([math.cos(th), math.sin(th), 0.0])
        Rm = np.stack([tng, rad, np.array([0.0, 0.0, 1.0])], 1)
        P.append(FG.ell(c, (2 * math.pi * R_mid / n * 1.2, 0.5 * THICK * thin, 0.5 * WIDTH * (0.75 + 0.25 * k)),
                        Rm, k=0.0012, mat=0))
    bx, by = 0.85 * R_mid * math.cos(bd), 0.85 * R_mid * math.sin(bd)
    rb = (0.0046 + 0.0014 * m) * max(m, 1e-3) ** (1.0 / 3.0)
    if rb > 0.0008:
        P.append(FG.ell(np.array([bx, by, z0 + 0.55 * rb]), (rb * 1.12, rb, rb * 0.62), None, k=0.0026 * min(1, m * 3),
                        mat=0))
        # a thin skin of metal spreading where the drop wets the coal
        P.append(FG.ell(np.array([bx * 1.02, by * 1.02, z0 + 0.0003]), (rb * 1.6, rb * 1.45, 0.0005), None,
                        k=0.0015, mat=0))
    Vv, Q, M = FG.mesh_sdf(P, h=0.00013, pad=0.001)
    FG.write_mesh(path, Vv, Q, M)


def _vision(path, u):
    """Equirect panorama (1024x512, linear HDR) for the band's vision, u 0..1: invented furnace-stacks round
    the horizon against a forge glow, bowing toward the viewer as u rises. No real place, no symbol."""
    import cv2
    import numpy as np
    W, H = 1024, 512
    el = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    sky = np.zeros((H, W, 3), np.float32)
    g = np.clip(1.0 - el / 0.9, 0, 1) ** 2.2 * (el > 0)
    for c, k in zip(range(3), (1.0, 0.34, 0.07)):
        sky[:, :, c] = (g * 1.6 * k)[:, None]
    mask = np.zeros((H, W), np.uint8)
    rng = np.random.default_rng(7)
    for i in range(9):
        az = -150 + i * 36 + rng.uniform(-8, 8)
        x0 = (az + 180) / 360 * W
        h = rng.uniform(0.34, 0.55) * H / 2
        w = rng.uniform(10, 18)
        bow = u * rng.uniform(0.35, 0.6) * (1 if x0 < W / 2 else -1)     # tops lean toward the centre
        pts = []
        for t_, half in ((0.0, w), (0.55, w * 0.8), (0.56, w * 0.95), (0.62, w * 0.7), (0.92, w * 0.55),
                         (0.93, w * 0.7), (1.0, w * 0.62)):
            ang = bow * max(0.0, t_ - 0.3) ** 1.5
            cx = x0 + h * t_ * math.sin(ang)
            cy = H / 2 - h * t_ * math.cos(ang)
            pts.append(((cx - half, cy), (cx + half, cy)))
        poly = [p[0] for p in pts] + [p[1] for p in pts[::-1]]
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.2)
    img = sky * (1.0 - m[..., None] * 0.96)
    img[H // 2:] = 0.0
    cv2.imwrite(path, img[..., ::-1].astype(np.float32))


def prep(frames, cache):
    import sys

    import cv2
    import numpy as np
    d = os.path.join(cache, 'data')
    os.makedirs(d, exist_ok=True)
    # the canonical inscription (assets/ring, ring_script.py): outer and inner faces
    adir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'ring')
    tex = os.path.abspath(os.path.join(adir, 'inscription_outer.png'))
    tex_in = os.path.abspath(os.path.join(adir, 'inscription_inner.png'))
    info = dict(tex=tex, tex_inner=tex_in, shot=SHOT)
    if SHOT == 'find':
        p = os.path.join(d, 'snow.bin')
        if not os.path.exists(p) or os.environ.get('MT3D_RERING'):
            _snow(p)
        g = os.path.join(d, 'glove.bin')
        if not os.path.exists(g) or os.environ.get('MT3D_RERING'):
            _glove(g)
        info.update(snow=p, glove=g, glove_tip=json.load(open(g + '.json'))['tip_c'])
    if SHOT == 'melt_still':
        p = os.path.join(d, 'melt.bin')
        if not os.path.exists(p) or os.environ.get('MT3D_RERING'):
            _melt_mesh(p)
        info.update(melt=p)
    redo = bool(os.environ.get('MT3D_RERING'))
    if SHOT in ('find_a', 'find_b'):
        p = os.path.join(d, 'snow.bin')
        if not os.path.exists(p) or redo:
            _snow(p)
        info.update(snow=p, hand={})
        for f in frames:
            c = hand_curl(f)
            g = os.path.join(d, f'glove_{SHOT}_{f:05d}.bin')
            if not os.path.exists(g) or redo:
                _glove(g, curl=c, base=OPEN if SHOT == 'find_b' else REACH)
            info['hand'][str(f)] = g
        if SHOT == 'find_b':
            vd = os.path.join(d, 'vision')
            os.makedirs(vd, exist_ok=True)
            for f in range(BEAT['vision'][0] - 4, BEAT['vision'][1] + 5):
                vp = os.path.join(vd, f'vision_{f:05d}.exr')
                if not os.path.exists(vp) or redo:
                    _vision(vp, min(1.0, max(0.0, (f - BEAT['vision'][0]) / 44.0)))
            info['vision'] = os.path.join(vd, f'vision_{BEAT["vision"][0] - 4:05d}.exr')
    if SHOT == 'melt':
        info['melt'] = {}
        for f in frames:
            m = melt_m(f)
            if m <= 0.0:
                continue
            p = os.path.join(d, f'melt_{f:05d}.bin')
            if not os.path.exists(p) or redo:
                _melt_mesh(p, m=m)
            info['melt'][str(f)] = p
    return dict(ring=info)


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def hand_curl(f):
    """0 = reaching / open, 1 = closed fist."""
    if SHOT == 'find_a':
        return _ease((f - 3050) / 10.0)
    return 1.0 - _ease((f - 3081) / 9.0) + _ease((f - BEAT['fist']) / 8.0)


def melt_m(f):
    """Slump progress for the melt: 0 until it has lain in the white heart a moment, then runs."""
    return 0.0 if f < 5380 else min(0.95, 0.95 * _ease((f - 5380) / 62.0) ** 0.8)


# ==================================================================== Blender side ===

def _sgnpow(x, p):
    return math.copysign(abs(x) ** p, x)


def ring_object(name, mat, n_th=512, n_a=64, melt=None):
    """The band: superellipse profile (radial THICK x axial WIDTH) swept round the local z axis.
    UV 'strip': u once round, v across the band width. Attribute 'outer': how much a vertex faces out.
    melt: dict(bead=(theta), m=progress 0..1, floor=z) slumps the band into a bead (keeps the UVs)."""
    import array

    import bpy
    R_mid = R_IN + THICK * 0.5
    V = array.array('f')
    outer = array.array('f')
    for j in range(n_th):
        th = 2 * math.pi * j / n_th
        ct, st = math.cos(th), math.sin(th)
        for i in range(n_a):
            a = 2 * math.pi * i / n_a
            rr = R_mid + 0.5 * THICK * _sgnpow(math.cos(a), 2.0 / SQ)
            zz = 0.5 * WIDTH * _sgnpow(math.sin(a), 2.0 / SQ)
            x, y, z = rr * ct, rr * st, zz
            if melt:
                x, y, z = _melt(x, y, z, th, melt)
            V.extend((x, y, z))
            outer.append(max(math.cos(a), 0.0))
    nv = n_th * n_a
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
    me.vertices.add(nv)
    me.vertices.foreach_set('co', V)
    me.loops.add(nq * 4)
    me.loops.foreach_set('vertex_index', loops)
    me.polygons.add(nq)
    me.polygons.foreach_set('loop_start', array.array('i', range(0, nq * 4, 4)))
    me.update(calc_edges=True)
    me.shade_smooth()
    uv = me.uv_layers.new(name='strip')
    uv.data.foreach_set('uv', uvs)
    at = me.attributes.new('outer', 'FLOAT', 'POINT')
    at.data.foreach_set('value', outer)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _slump_data(path, mat):
    ob = _slump_object('_tmp', path, mat)
    me = ob.data
    import bpy
    bpy.data.objects.remove(ob)
    return me


def _slump_object(name, path, mat):
    """Load the SDF slump; UV 'strip' from angle round the axis + height (the letters run on across the
    molten metal); 'outer' = how much the surface faces away from the axis."""
    import array

    import bpy

    import karst as KA
    me = KA._mesh_fast(name, path, mat)
    nv = len(me.vertices)
    co = array.array('f', [0.0]) * (nv * 3)
    me.vertices.foreach_get('co', co)
    lv = array.array('i', [0]) * len(me.loops)
    me.loops.foreach_get('vertex_index', lv)
    uvs = array.array('f', [0.0]) * (len(me.loops) * 2)
    for li, vi in enumerate(lv):
        x, y, z = co[3 * vi], co[3 * vi + 1], co[3 * vi + 2]
        uvs[2 * li] = (math.atan2(y, x) / (2 * math.pi)) % 1.0
        uvs[2 * li + 1] = z / WIDTH + 0.5
    uv = me.uv_layers.new(name='strip')
    uv.data.foreach_set('uv', uvs)
    me.update()
    nrm = array.array('f', [0.0]) * (nv * 3)
    me.vertices.foreach_get('normal', nrm)
    outer = array.array('f', [0.0]) * nv
    for i in range(nv):
        x, y = co[3 * i], co[3 * i + 1]
        r = math.hypot(x, y) + 1e-9
        outer[i] = max(0.0, (nrm[3 * i] * x + nrm[3 * i + 1] * y) / r)
    at = me.attributes.new('outer', 'FLOAT', 'POINT')
    at.data.foreach_set('value', outer)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    sm = ob.modifiers.new('smooth', 'SMOOTH')
    sm.factor = 0.6
    sm.iterations = 6
    return ob


def _melt(x, y, z, th, M):
    """Slump: the far arc sags and thins toward the bead; round the bead the band's own surface is drawn
    onto a flattened drop sitting on the floor. Local frame: the ring lies flat (axis up), floor z0."""
    bead_th, m, z0 = M['bead'], M['m'], M['floor']
    d = math.atan2(math.sin(th - bead_th), math.cos(th - bead_th))
    phi = abs(d) / math.pi                                 # 0 at the bead .. 1 opposite
    R_mid = R_IN + THICK * 0.5
    bx, by = R_mid * math.cos(bead_th) * 0.72, R_mid * math.sin(bead_th) * 0.72
    rb = 0.0062 * (0.55 + 0.45 * m)
    bz = z0 + rb * 0.55
    # sag: the whole band settles to the floor, more near the bead; thinning toward the bead
    w = max(0.0, 1.0 - phi / (0.45 + 0.4 * m))
    w = w * w * (3 - 2 * w)
    sag = m * (0.0016 + 0.0026 * (1 - phi))
    thin = 1.0 - 0.45 * m * (1 - phi)
    cx, cy = R_mid * math.cos(th), R_mid * math.sin(th)
    x = cx + (x - cx) * thin
    y = cy + (y - cy) * thin
    z = z * thin - sag
    if w > 0:
        dx, dy, dz = x - bx, y - by, (z - bz) / 0.55
        L = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
        px, py, pz = bx + dx / L * rb, by + dy / L * rb, bz + dz / L * rb * 0.55
        pz = max(pz, z0 + 0.0001)
        x, y, z = x + (px - x) * w, y + (py - y) * w, z + (pz - z) * w
    return x, y, max(z, z0 + 0.00005)


def gold_material(C, new_material, tex_path, letters=0.0, letter_col=(1.0, 0.3, 0.05), hot=0.0):
    """Polished yellow gold. No glitter: roughness varies only with slow polish marks. Letters: emission
    through ring.py's inscription, on the outer face only. hot: incandescent glow (the melt)."""
    import bpy
    m, nb = new_material('gold')
    P = nb.texco().outputs['Object']
    polish = nb.noise(P, scale=160.0, detail=2.0, rough=0.5)
    rough = nb.madd(polish.outputs['Fac'], 0.05, 0.045)
    bs = nb.principled(Base_Color=(1.0, 0.71, 0.29), Metallic=1.0, Roughness=rough)
    bs.inputs['Specular Tint'].default_value = (1.0, 0.93, 0.78, 1.0)
    em_s = None
    if letters > 0:
        img = bpy.data.images.load(tex_path, check_existing=True)
        img.colorspace_settings.name = 'Non-Color'
        tx = nb.n('ShaderNodeTexImage')
        tx.image = img
        tx.interpolation = 'Cubic'
        tx.extension = 'CLIP'
        uv = nb.n('ShaderNodeUVMap')
        uv.uv_map = 'strip'
        nb.link(uv.outputs['UV'], tx.inputs['Vector'])
        cov = nb.mul(tx.outputs['Color'], nb.sstep(0.25, 0.6, nb.attr('outer').outputs['Fac']))
        em_s = nb.mul(cov, letters)
        bs.inputs['Emission Color'].default_value = tuple(letter_col) + (1.0,)
        nb.set(bs.inputs['Emission Strength'], em_s)
    if hot > 0:
        glow = nb.n('ShaderNodeBlackbody')
        glow.inputs['Temperature'].default_value = 1450.0
        hot_em = nb.emission(glow.outputs['Color'], hot)
        nb.output(surface=nb.addshader(bs, hot_em))
    else:
        nb.output(surface=bs)
    return m


def build(job):
    import bpy

    from kit import core as C
    from kit.nodes import new_material
    T = job['timing']
    R = T['ring']
    opts = job.get('opts', {})
    shot = R['shot']
    C.setup_render(scale=job['scale'], samples=job['samples'], mblur=False, light_threshold=1e-6,
                   shadow_pool='256')
    sc = bpy.context.scene
    ro = sc.eevee.ray_tracing_options
    ro.resolution_scale = '1'
    ro.trace_max_roughness = 0.5
    sc.eevee.use_bokeh_jittered = True
    sc.eevee.bokeh_max_size = 120.0
    w = bpy.data.worlds.new('W')
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    if shot == 'find':
        _build_find(C, new_material, R, opts, bg)
    elif shot == 'fire_still':
        _build_fire(C, new_material, R, opts, bg, T)
    elif shot == 'melt_still':
        _build_melt(C, new_material, R, opts, bg, T)
    else:
        sc.render.use_motion_blur = True
        sc.render.motion_blur_shutter = 0.5
        dict(find_a=_anim_find_a, find_b=_anim_find_b, fire=_anim_fire, melt=_anim_melt)[shot](C, new_material, R, opts,
                                                                                            bg, T)
    return {}


# ================================================================ animated inserts ===

_SWAP = []


def per_frame(job, f):
    for ob, paths, loader in _SWAP:
        p = paths.get(str(f))
        if p is None:
            continue
        old = ob.data
        ob.data = loader(p)
        if old.users == 0:
            import bpy
            bpy.data.meshes.remove(old)


def _keyv(nb, name, table):
    from kit import core as C
    v = nb.value(0.0, name)
    for f, x in table:
        C.key_socket(v.outputs[0], f, x)
    return v.outputs[0]


def _tab(fn, f0=None, f1=None):
    return [(f, fn(f)) for f in range((START if f0 is None else f0) - 1, (END if f1 is None else f1) + 2)]


def gold_anim(C, new_material, tex_path, letters_tab=None, letter_col=(1.0, 0.3, 0.06), hot_tab=None,
              vision_path=None, vision_tab=None):
    import bpy
    m, nb = new_material('gold_anim')
    P = nb.texco().outputs['Object']
    polish = nb.noise(P, scale=160.0, detail=2.0, rough=0.5)
    bs = nb.principled(Base_Color=(1.0, 0.71, 0.29), Metallic=1.0,
                       Roughness=nb.madd(polish.outputs['Fac'], 0.05, 0.045))
    bs.inputs['Specular Tint'].default_value = (1.0, 0.93, 0.78, 1.0)
    shader = bs
    if letters_tab:
        img = bpy.data.images.load(tex_path, check_existing=True)
        img.colorspace_settings.name = 'Non-Color'
        tx = nb.n('ShaderNodeTexImage')
        tx.image = img
        tx.interpolation = 'Cubic'
        tx.extension = 'CLIP'
        uv = nb.n('ShaderNodeUVMap')
        uv.uv_map = 'strip'
        nb.link(uv.outputs['UV'], tx.inputs['Vector'])
        cov = nb.mul(tx.outputs['Color'], nb.sstep(0.25, 0.6, nb.attr('outer').outputs['Fac']))
        bs.inputs['Emission Color'].default_value = tuple(letter_col) + (1.0,)
        nb.set(bs.inputs['Emission Strength'], nb.mul(cov, _keyv(nb, 'letters', letters_tab)))
    if hot_tab:
        bb = nb.n('ShaderNodeBlackbody')
        bb.inputs['Temperature'].default_value = 1400.0
        shader = nb.addshader(shader, nb.emission(bb.outputs['Color'], _keyv(nb, 'hot', hot_tab)))
    if vision_path:
        # the vision lives IN the polish: the band's reflection vector looks into a panorama of bowing towers
        from kit import fire as FK
        env = nb.n('ShaderNodeTexEnvironment')
        vim = bpy.data.images.load(vision_path, check_existing=False)
        vim.source = 'SEQUENCE'
        vim.colorspace_settings.name = 'Linear Rec.709'
        env.image = vim
        iu = env.image_user
        iu.frame_start = 1
        iu.frame_duration = 100000
        iu.frame_offset = 0
        iu.use_auto_refresh = True
        nb.link(nb.texco().outputs['Reflection'], env.inputs['Vector'])
        t = FK.time_value(nb)
        sh = nb.noise(nb.vadd(nb.vscale(P, 700.0), nb.comb(0.0, 0.0, nb.mul(t, 3.0))), scale=1.0, detail=2.0)
        shimmer = nb.madd(nb.sstep(0.3, 0.7, sh.outputs['Fac']), 0.7, 0.3)
        k = nb.mul(_keyv(nb, 'vision', vision_tab), shimmer)
        shader = nb.addshader(shader, nb.emission(nb.colscale(env.outputs['Color'], (1.0, 0.85, 0.6)), k))
    nb.output(surface=shader)
    return m


def _wool(new_material):
    wool, nb = new_material('wool')
    P = nb.texco().outputs['Object']
    knit = nb.voronoi(nb.vmath('MULTIPLY', P, (1.0, 1.6, 1.0)), scale=520.0, feature='SMOOTH_F1')
    fuzz = nb.noise(P, scale=2600.0, detail=5.0, rough=0.75)
    bsw = nb.principled(Base_Color=nb.colscale((0.05, 0.036, 0.03), nb.madd(fuzz.outputs['Fac'], 0.5, 0.75)),
                        Roughness=0.95)
    bsw.inputs['Sheen Weight'].default_value = 1.0
    bsw.inputs['Sheen Roughness'].default_value = 0.45
    bsw.inputs['Sheen Tint'].default_value = (1.0, 0.85, 0.75, 1.0)
    nb.link(nb.bump(nb.add(nb.mul(knit.outputs['Distance'], 1.2), nb.mul(fuzz.outputs['Fac'], 0.4)), 0.5, 0.0006),
            bsw.inputs['Normal'])
    nb.output(surface=bsw)
    return wool


def _snow_mat(new_material):
    m, nb = new_material('snow')
    P = nb.texco().outputs['Object']
    uv = nb.n('ShaderNodeUVMap')
    uv.uv_map = 'crest'
    glaze, rimv, _ = nb.sep(uv.outputs['UV'])
    grain = nb.noise(P, scale=900.0, detail=3.0, rough=0.6)
    sugar = nb.voronoi(P, scale=1700.0, feature='SMOOTH_F1')
    bs = nb.principled(Base_Color=nb.mixcol(glaze, (0.84, 0.89, 0.97), (0.5, 0.58, 0.7)),
                       Roughness=nb.mixf(glaze, nb.madd(grain.outputs['Fac'], 0.25, 0.55), 0.22))
    bs.inputs['Subsurface Weight'].default_value = 1.0
    bs.inputs['Subsurface Radius'].default_value = (1.0, 1.4, 2.2)
    bs.inputs['Subsurface Scale'].default_value = 0.004
    bs.inputs['Specular IOR Level'].default_value = 0.5
    nb.link(nb.bump(nb.add(grain.outputs['Fac'], nb.mul(sugar.outputs['Distance'], 1.5)), nb.mixf(glaze, 0.45, 0.03),
                    0.0005), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def _hand_obj(R, wool):
    import bpy

    import karst as KA
    paths = R['hand']
    first = paths[str(START - 1)] if str(START - 1) in paths else paths[min(paths, key=int)]
    ob = bpy.data.objects.new('hand', KA._mesh_fast('hand', first, wool))
    bpy.context.scene.collection.objects.link(ob)
    _SWAP.append((ob, paths, lambda p: KA._mesh_fast('hand', p, wool)))
    return ob


def _anim_find_a(C, new_material, R, opts, bg, T=None):
    """C14 (a): dark snow; strike 2's spark (3009) flashes and falls, the band glints in its hollow; she stops;
    the gloved hand reaches in, closes on it (3060) and lifts it away."""
    import random

    import bpy
    from mathutils import Euler, Matrix, Vector

    import desert as DS
    from kit import fire as FK
    bg.inputs[0].default_value = (0.0012, 0.0018, 0.0042, 1.0)
    me = DS._mesh_ch('snow', R['snow'], _snow_mat(new_material))
    C.link_obj(bpy.data.objects.new('snow', me))
    gold = gold_anim(C, new_material, R['tex'])
    ring = ring_object('ring', gold)
    ring.rotation_mode = 'QUATERNION'
    rest = Matrix.Translation(Vector((0.0, 0.0, -0.0032))) @ Euler((math.radians(14.0), math.radians(-9.0),
                                                                     math.radians(30.0))).to_matrix().to_4x4()
    hand = _hand_obj(R, _wool(new_material))
    Rh = (Matrix.Rotation(math.radians(212.0), 4, 'Z') @ Matrix.Rotation(math.radians(10.0), 4, 'Y') @
          Matrix.Rotation(math.radians(-12.0), 4, 'X'))
    c0 = Vector((0.0, 0.0, -0.0032))
    far = c0 + Vector((0.075, 0.06, 0.07))
    above = c0 + Vector((0.004, 0.004, 0.014))
    grip = c0 + Vector((0.0, 0.0, 0.0025))
    lift = c0 + Vector((0.06, 0.045, 0.075))

    def H(f):
        if f <= 3036:
            p = far
        elif f <= 3052:
            p = far.lerp(above, _ease((f - 3036) / 16.0))
        elif f <= BEAT['close']:
            p = above.lerp(grip, _ease((f - 3052) / 8.0))
        else:
            p = grip.lerp(lift, _ease((f - BEAT['close'] - 2) / 16.0) ** 1.4)
        return Matrix.Translation(p) @ Rh @ Matrix.Translation(-Vector(GRIP))

    Hc_inv = H(BEAT['close']).inverted()
    for f in range(START - 1, END + 2):
        hand.matrix_world = H(f)
        hand.keyframe_insert('location', frame=f)
        hand.keyframe_insert('rotation_euler', frame=f)
        rm = rest if f <= BEAT['close'] else H(f) @ Hc_inv @ rest
        ring.matrix_world = rm
        ring.keyframe_insert('location', frame=f)
        ring.keyframe_insert('rotation_quaternion', frame=f)
    # strike 2: the flash and the falling sparks
    sp = Vector((-0.024, -0.012, 0.03))
    L = C.point('spark', sp, (1.0, 0.55, 0.22), 0.0, radius=0.0006)
    for f in range(START - 1, END + 2):
        e = 0.0 if f < BEAT['strike2'] else 0.05 * math.exp(-(f - BEAT['strike2']) / 2.2)
        C.key(L.data, 'energy', f, e)
    em, nb = new_material('sparkm')
    kv = _keyv(nb, 'glow', [(f, 0.0 if f < BEAT['strike2'] else 30.0 * math.exp(-(f - BEAT['strike2']) / 4.0))
                            for f in range(START - 1, END + 2)])
    nb.output(surface=nb.emission((1.0, 0.5, 0.16), kv))
    rng = random.Random(9)
    for k in range(9):
        bm_v, bm_f = FK.tube([(0, 0, -0.0006), (0, 0, 0.0006)], 0.00006, 4)
        mote = C.mesh_obj(f'mote{k}', bm_v, bm_f, mat=em)
        v0 = Vector((rng.uniform(-0.05, 0.35), rng.uniform(-0.1, 0.25), rng.uniform(-0.05, 0.3)))
        for f in range(START - 1, END + 2):
            tt = max(0.0, (f - BEAT['strike2']) / FPS)
            p = sp + v0 * tt + Vector((0, 0, -4.9 * tt * tt))
            p.z = max(p.z, -0.001)
            mote.location = p if f >= BEAT['strike2'] else sp + Vector((0, 0, 0.5))
            mote.keyframe_insert('location', frame=f)
    C.sun('moon', (-0.55, -0.35, 0.75), (0.42, 0.58, 1.0), opts.get('moon', 0.5), angle_deg=1.0)
    _probe((0.0, 0.0, 0.0), 0.12)
    cp = opts.get('cam', [0.022, -0.15, 0.095])
    _camera(C, cp, (0.0, 0.006, -0.004), opts.get('lens', 70.0), opts.get('fstop', 11.0),
            focus=(Vector(cp) - Vector((0.0, 0.0, -0.003))).length)


def _anim_find_b(C, new_material, R, opts, bg, T=None):
    """C14 (b): her gloved hand, palm up, opens (3090) on the band; for two seconds a vision shimmers in
    its polish (forge-stacks bowing toward her); the fist closes (3140)."""
    import bpy
    from mathutils import Matrix, Vector

    import desert as DS
    bg.inputs[0].default_value = (0.0012, 0.0018, 0.0042, 1.0)
    snow = C.link_obj(bpy.data.objects.new('snow', DS._mesh_ch('snow', R['snow'], _snow_mat(new_material))))
    snow.location = (0.0, 0.0, -0.14)
    v0, v1 = BEAT['vision']

    def vis(f):
        return 1.6 * _ease((f - v0) / 6.0) * (1.0 - _ease((f - (v1 - 7)) / 7.0))

    gold = gold_anim(C, new_material, R['tex'], vision_path=R['vision'], vision_tab=_tab(vis))
    hand = _hand_obj(R, _wool(new_material))
    Hb = (Matrix.Translation(Vector((0.0, 0.0, 0.0))) @ Matrix.Rotation(math.radians(200.0), 4, 'Z') @
          Matrix.Rotation(math.radians(-8.0), 4, 'Y') @ Matrix.Rotation(math.pi, 4, 'X') @
          Matrix.Translation(Vector((-0.07, 0.0, 0.0))))
    hand.matrix_world = Hb
    ring = ring_object('ring', gold)
    ring.matrix_world = Hb @ Matrix.Translation(Vector((0.07, 0.002, -0.0155 - 0.5 * WIDTH + 0.0004))) @ \
        Matrix.Rotation(math.radians(8.0), 4, 'Y')
    C.sun('moon', (-0.5, -0.4, 0.75), (0.42, 0.58, 1.0), opts.get('moon', 0.45), angle_deg=1.0)
    _probe((0.0, 0.0, 0.0), 0.2)
    cp = opts.get('cam', [0.035, -0.11, 0.1])
    _camera(C, cp, (0.0, 0.004, -0.002), opts.get('lens', 70.0), opts.get('fstop', 9.0))


def _key_gain(ob, table):
    """Key the emission strength of a flame card (kit.fire.flame_card material)."""
    from kit import core as C
    for nd in ob.data.materials[0].node_tree.nodes:
        if nd.type == 'EMISSION':
            base = nd.inputs['Strength'].default_value
            for f, v in table:
                C.key_socket(nd.inputs['Strength'], f, base * v)


def _anim_fire(C, new_material, R, opts, bg, T):
    """C15, Bag End: the roar (3360); on 3400 her steel carries the band into the flames; it lies there
    unmarked, letters awake; on 3480 the steel tips, the band slides to the tip and does not fall; on 3560 she
    draws it out."""
    import bpy
    from mathutils import Euler, Matrix, Vector

    from kit import fire as FK
    bg.inputs[0].default_value = (0.02, 0.006, 0.002, 1.0)
    _coals(C, new_material, 70, (0.1, 0.08), 11, 'fire', z0=-0.065)

    def letters(f):
        return 2.2 * _ease((f - BEAT['steel_in'] - 2) / 18.0) * (1.0 - _ease((f - BEAT['draw'] - 2) / 12.0))

    gold = gold_anim(C, new_material, R['tex'], letters_tab=_tab(letters), letter_col=(1.0, 0.3, 0.06))
    st, nb = new_material('steel')
    P = nb.texco().outputs['Object']
    x = nb.sep(P)[0]
    tint = nb.ramp(nb.sstep(0.07, 0.179, x), [(0.0, (0.06, 0.06, 0.065)), (0.55, (0.08, 0.075, 0.07)),
                                                (0.72, (0.3, 0.22, 0.1)), (0.84, (0.26, 0.12, 0.08)),
                                                (0.93, (0.1, 0.1, 0.22)), (1.0, (0.12, 0.12, 0.14))])
    bsx = nb.principled(Base_Color=tint, Metallic=1.0,
                        Roughness=nb.madd(nb.noise(P, scale=900.0, detail=3.0).outputs['Fac'], 0.2, 0.3))
    nb.link(nb.bump(nb.noise(P, scale=1400.0, detail=4.0).outputs['Fac'], 0.25, 0.0003), bsx.inputs['Normal'])
    nb.output(surface=bsx)
    # rig at her hand (out of frame left); the bar in rig space; the band hangs near the tip
    y_top = R_IN - 0.0012
    piv = Vector((-0.16, 0.012, y_top - 0.028))
    pts = [Vector(q) - piv for q in ((-0.16, 0.012, y_top - 0.028), (-0.06, 0.004, y_top - 0.01),
                                      (0.0, 0.0, y_top - 0.0019), (0.012, 0.0, y_top - 0.0012),
                                      (0.019, 0.0, y_top + 0.0012))]
    tv, tf = FK.tube(pts, 0.0021, 8, radii=[0.0024, 0.0022, 0.0019, 0.0016, 0.0011])
    rig = bpy.data.objects.new('rig', None)
    C.link_obj(rig)
    stl = C.mesh_obj('steel', tv, tf, mat=st)
    stl.parent = rig
    stl.scale = (1.0, 1.0, 0.7)
    ring = ring_object('ring', gold)
    ring.parent = rig
    ring.rotation_mode = 'XYZ'
    base_rot = Euler((math.radians(4.0), math.radians(56.0), math.radians(-18.0)), 'XYZ')
    along = (pts[4] - pts[2]).normalized()
    for f in range(START - 1, END + 2):
        if f < BEAT['steel_in']:
            dx = -0.3
        elif f < BEAT['steel_in'] + 12:
            dx = -0.3 * (1.0 - _ease((f - BEAT['steel_in']) / 12.0))
        elif f < BEAT['draw']:
            dx = 0.0
        else:
            dx = -0.32 * _ease((f - BEAT['draw']) / 18.0) ** 1.6
        tip = math.radians(9.0) * _ease((f - BEAT['tip']) / 8.0) * (1.0 - 0.6 * _ease((f - BEAT['tip'] - 22) / 18.0))
        rig.location = piv + Vector((dx, 0.0, 0.0))
        rig.rotation_euler = (0.0, tip, 0.0)
        rig.keyframe_insert('location', frame=f)
        rig.keyframe_insert('rotation_euler', frame=f)
        slide = 0.0042 * _ease((f - BEAT['tip'] - 2) / 7.0)
        ts = max(0.0, (f - BEAT['tip'] - 9) / FPS)
        sway = (math.radians(5.0) * math.exp(-ts * 2.2) * math.sin(ts * 11.0) if f >= BEAT['tip'] + 9 else 0.0)
        arr = max(0.0, (f - BEAT['steel_in'] - 12) / FPS)
        sway += math.radians(4.0) * math.exp(-arr * 2.5) * math.sin(arr * 10.0) if f >= BEAT['steel_in'] + 12 else 0.0
        ring.location = (-piv) + along * slide
        ring.rotation_euler = (base_rot.x + sway, base_rot.y, base_rot.z)
        ring.keyframe_insert('location', frame=f)
        ring.keyframe_insert('rotation_euler', frame=f)
    # the roar: the fire surges on 3360 and settles
    def roar(f):
        return 0.3 + 0.9 * _ease((f - BEAT['roar'] + 2) / 6.0) - 0.25 * _ease((f - BEAT['roar'] - 10) / 30.0)

    Ld = bpy.data.lights.new('bed', 'AREA')
    Ld.shape = 'DISK'
    Ld.size = 0.16
    Ld.color = (1.0, 0.33, 0.07)
    obl = C.link_obj(bpy.data.objects.new('bed', Ld))
    obl.location = (0.0, 0.0, -0.04)
    rim = C.point('rim', (0.02, 0.09, 0.03), (1.0, 0.55, 0.2), 0.0, radius=0.03)
    for f in range(START - 1, END + 2):
        C.key(Ld, 'energy', f, 6.0 * roar(f))
        C.key(rim.data, 'energy', f, 3.0 * roar(f))
    _probe((0.0, 0.0, 0.0), 0.25)
    cam = _camera(C, opts.get('cam', [0.1, -0.2, 0.03]), (0.004, 0.0, -0.004), opts.get('lens', 90.0),
                  opts.get('fstop', 5.6))
    specs = [('bgfire', (0.02, 0.2, -0.05), 0.45), ('bgfire', (-0.08, 0.26, -0.05), 0.4),
             ('bgfire', (0.1, 0.24, -0.05), 0.35), ('lick', (0.01, 0.03, -0.035), 1.0),
             ('lick', (-0.03, 0.02, -0.038), 0.9)]
    for i, (name, pos, gain) in enumerate(specs):
        sp_ = T['sprites'][name]
        ob = FK.flame_card(f'{name}{i}', Vector(pos), sp_['card'], sp_['first'], cam, gain=gain, fog=False)
        _key_gain(ob, _tab(roar))


def _anim_melt(C, new_material, R, opts, bg, T):
    """C22: the band drops into the white heart (5360+), lies there, slumps and runs into a drop; its letters
    flare once (5420) and go out (5440); the hearth flares to white."""
    import bpy
    from mathutils import Euler, Matrix, Vector

    from kit import core as Ck
    from kit import fire as FK

    def white(f):
        return 1.0 + 5.0 * _ease((f - BEAT['white']) / 45.0) ** 1.5

    def letters(f):
        a = 1.4 * _ease((f - 5368) / 8.0)
        fl = 7.5 * math.exp(-max(0.0, f - BEAT['flare']) / 4.0) * (f >= BEAT['flare'] - 1)
        return (a + fl) * (1.0 - _ease((f - BEAT['out']) / 3.0))

    def hot(f):
        return 0.28 * _ease((f - 5380) / 60.0)

    from kit.nodes import NB
    bg.inputs[0].default_value = (1.0, 0.3, 0.06, 1.0)
    wt = bpy.context.scene.world.node_tree
    wv = _keyv(NB(wt), 'white', [(f, 0.08 * white(f)) for f in range(START - 1, END + 2)])
    wt.links.new(wv, bg.inputs[1])
    coals = _coals(C, new_material, 70, (0.09, 0.08), 17, 'white', z0=-0.012, exclude=(0.0, 0.0, 0.017, -0.004))
    ct = coals.data.materials[0].node_tree
    emis = [nd for nd in ct.nodes if nd.type == 'EMISSION']
    nb_ = NB(ct)
    k = _keyv(nb_, 'whiteflare', [(f, white(f)) for f in range(START - 1, END + 2)])
    for nd in emis:
        src = nd.inputs['Strength'].links[0].from_socket
        ct.links.new(nb_.mul(src, k), nd.inputs['Strength'])
    C.mesh_obj('slab', [(-0.05, -0.05, -0.0027), (0.05, -0.05, -0.0027), (0.05, 0.05, -0.0027), (-0.05, 0.05, -0.0027)],
               [(0, 1, 2, 3)], mat=bpy.data.materials['coal_white'], smooth=False)
    gold = gold_anim(C, new_material, R['tex'], letters_tab=_tab(letters), letter_col=(1.0, 0.82, 0.5),
                     hot_tab=_tab(hot))
    # the drop: the intact band falls in and settles (analytic), then the slump meshes take over
    ring = ring_object('ring', gold)
    ring.rotation_mode = 'XYZ'
    z_rest = 0.5 * WIDTH - 0.0027
    for f in range(START - 1, END + 2):
        if f < 5364:
            z, rx = 0.09, 0.9
        elif f < 5368:
            u = (f - 5364) / 4.0
            z, rx = z_rest + (0.09 - z_rest) * (1 - u * u), 0.9 * (1 - u)
        else:
            u = (f - 5368) / FPS
            z, rx = z_rest + 0.002 * abs(math.sin(u * 22.0)) * math.exp(-u * 14.0), 0.05 * math.exp(-u * 10.0)
        ring.location = (0.0, 0.0, z)
        ring.rotation_euler = (rx, 0.3 * rx, 0.0)
        ring.keyframe_insert('location', frame=f)
        ring.keyframe_insert('rotation_euler', frame=f)
        ring.hide_render = bool(str(f) in R['melt'])
        ring.keyframe_insert('hide_render', frame=f)
    first = min(R['melt'], key=int)
    slump = _slump_object('slump', R['melt'][first], gold)
    slump.location = (0.0, 0.0, z_rest)
    for f in range(START - 1, END + 2):
        slump.hide_render = not bool(str(f) in R['melt'])
        slump.keyframe_insert('hide_render', frame=f)
    _SWAP.append((slump, R['melt'], lambda p: _slump_data(p, gold)))
    heart = C.point('heart', (0.0, 0.02, 0.03), (1.0, 0.8, 0.55), 0.0, radius=0.02)
    Ld = bpy.data.lights.new('bed', 'AREA')
    Ld.shape = 'DISK'
    Ld.size = 0.12
    Ld.color = (1.0, 0.6, 0.25)
    ob = C.link_obj(bpy.data.objects.new('bed', Ld))
    ob.location = (0.0, 0.0, -0.006)
    ob.rotation_euler = Euler((math.pi, 0.0, 0.0))
    for f in range(START - 1, END + 2):
        C.key(heart.data, 'energy', f, 0.25 * white(f))
        C.key(Ld, 'energy', f, 1.0 * white(f))
    _probe((0.0, 0.0, 0.0), 0.2)
    cam = _camera(C, opts.get('cam', [0.03, -0.12, 0.034]), (0.0, 0.0, 0.0), opts.get('lens', 90.0),
                  opts.get('fstop', 9.0))
    for i, (name, pos, gain) in enumerate([('lick', (0.014, 0.024, -0.004), 0.35), ('lick', (-0.022, 0.02, -0.006), 0.3),
                                           ('bgfire', (0.0, 0.14, -0.03), 0.3), ('bgfire', (-0.08, 0.18, -0.03), 0.25),
                                           ('bgfire', (0.08, 0.17, -0.03), 0.25)]):
        sp_ = T['sprites'][name]
        fc = FK.flame_card(f'{name}{i}', Vector(pos), sp_['card'], sp_['first'], cam, gain=gain, fog=False)
        _key_gain(fc, _tab(white))


def _camera(C, pos, target, lens, fstop, focus=None):
    import bpy
    from mathutils import Vector
    cd = bpy.data.cameras.new('CAM')
    cd.sensor_fit = 'HORIZONTAL'
    cd.sensor_width = 36.0
    cd.lens = lens
    cd.clip_start, cd.clip_end = 0.002, 50.0
    ob = bpy.data.objects.new('CAM', cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = pos
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = (Vector(target) - Vector(pos)).to_track_quat('-Z', 'Y')
    bpy.context.scene.camera = ob
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = fstop
    cd.dof.aperture_blades = 7
    cd.dof.focus_distance = focus or (Vector(target) - Vector(pos)).length
    return ob


def _probe(center, radius):
    import bpy
    pd = bpy.data.lightprobes.new('probe', 'SPHERE')
    try:
        pd.influence_distance = radius
        pd.clip_start = 0.001
    except Exception:
        pass
    ob = bpy.data.objects.new('probe', pd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = center
    return ob


def _build_find(C, new_material, R, opts, bg):
    import bpy
    from mathutils import Euler, Matrix, Vector

    import desert as DS
    from kit import figure as FGk
    import karst as KA
    bg.inputs[0].default_value = (0.0012, 0.0018, 0.0042, 1.0)
    bg.inputs[1].default_value = 1.0
    # snow
    m, nb = new_material('snow')
    P = nb.texco().outputs['Object']
    uv = nb.n('ShaderNodeUVMap')
    uv.uv_map = 'crest'
    glaze, rimv, _ = nb.sep(uv.outputs['UV'])
    grain = nb.noise(P, scale=900.0, detail=3.0, rough=0.6)
    sugar = nb.voronoi(P, scale=1700.0, feature='SMOOTH_F1')
    vor = nb.voronoi(P, scale=1500.0, feature='F1')
    cr, cg, cb = nb.sep(vor.outputs['Color'])
    xtal = nb.mul(nb.math('LESS_THAN', cr, 0.035), nb.sub(1.0, glaze))
    bs = nb.principled(Base_Color=nb.mixcol(glaze, (0.84, 0.89, 0.97), (0.5, 0.58, 0.7)),
                       Roughness=nb.mixf(glaze, nb.madd(grain.outputs['Fac'], 0.25, 0.55), 0.22))
    bs.inputs['Subsurface Weight'].default_value = 1.0
    bs.inputs['Subsurface Radius'].default_value = (1.0, 1.4, 2.2)
    bs.inputs['Subsurface Scale'].default_value = 0.004
    bs.inputs['Specular IOR Level'].default_value = 0.5
    bn = nb.bump(nb.add(grain.outputs['Fac'], nb.mul(sugar.outputs['Distance'], 1.5)), nb.mixf(glaze, 0.45, 0.03),
                 0.0005)
    geo_n = nb.geo().outputs['Normal']
    facet = nb.vmath('NORMALIZE', nb.vadd(geo_n, nb.vscale(nb.vsub(vor.outputs['Color'], (0.5, 0.5, 0.5)), 1.6)))
    nb.link(bn, bs.inputs['Normal'])
    nb.output(surface=bs)
    me = DS._mesh_ch('snow', R['snow'], m)
    C.link_obj(bpy.data.objects.new('snow', me))
    # the Ring: resting in the bottom of the hollow, a little askew
    gold = gold_material(C, new_material, R['tex'])
    ring = ring_object('ring', gold)
    ring.rotation_euler = Euler((math.radians(14.0), math.radians(-9.0), math.radians(30.0)))
    ring.location = (0.0, 0.0, opts.get('ring_z', -0.0032))
    # the glove, fingers curling down toward it from the upper right
    wool, nb = new_material('wool')
    P = nb.texco().outputs['Object']
    knit = nb.voronoi(nb.vmath('MULTIPLY', P, (1.0, 1.6, 1.0)), scale=520.0, feature='SMOOTH_F1')
    fuzz = nb.noise(P, scale=2600.0, detail=5.0, rough=0.75)
    bsw = nb.principled(Base_Color=nb.colscale((0.05, 0.036, 0.03), nb.madd(fuzz.outputs['Fac'], 0.5, 0.75)),
                        Roughness=0.95)
    bsw.inputs['Sheen Weight'].default_value = 1.0
    bsw.inputs['Sheen Roughness'].default_value = 0.45
    bsw.inputs['Sheen Tint'].default_value = (1.0, 0.85, 0.75, 1.0)
    nb.link(nb.bump(nb.add(nb.mul(knit.outputs['Distance'], 1.2), nb.mul(fuzz.outputs['Fac'], 0.4)), 0.5, 0.0006),
            bsw.inputs['Normal'])
    nb.output(surface=bsw)
    gme = KA._mesh_fast('glove', R['glove'], wool)
    gl = C.link_obj(bpy.data.objects.new('glove', gme))
    # placed by its fingertips: (target xyz for the curled fingertips' centroid, yaw, pitch, roll)
    gp = opts.get('glove', [0.013, 0.011, 0.004, 212.0, 10.0, -12.0])
    Rg = (Matrix.Rotation(math.radians(gp[3]), 4, 'Z') @ Matrix.Rotation(math.radians(gp[4]), 4, 'Y') @
          Matrix.Rotation(math.radians(gp[5]), 4, 'X'))
    gl.matrix_world = Matrix.Translation(Vector(gp[:3])) @ Rg @ Matrix.Translation(-Vector(R['glove_tip']))
    # the flint spark: a short shower of streaks up-left + its light (hard, tiny, warm)
    sp = Vector(opts.get('spark', (-0.022, -0.012, 0.026)))
    C.point('spark', sp, (1.0, 0.55, 0.22), opts.get('spark_w', 0.035), radius=0.0006)
    em, nb = new_material('sparkm')
    nb.output(surface=nb.emission((1.0, 0.5, 0.16), 22.0))
    import random
    rng = random.Random(5)
    parts = []
    from kit import fire as FK
    for k in range(7):
        a = rng.uniform(-2.8, 0.4)
        e = rng.uniform(-0.9, 0.1)
        L = rng.uniform(0.0015, 0.0045)
        dv = Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)))
        p0 = sp + dv * rng.uniform(0.0, 0.006)
        parts.append(FK.tube([p0, p0 + dv * L * 0.5, p0 + dv * L + Vector((0, 0, -0.0012))], 0.00005, 4,
                             radii=[0.00002, 0.00006, 0.00002]))
    Vs, Fs = FK.merge(parts)
    C.mesh_obj('sparks', Vs, Fs, mat=em)
    # cold sky fill from behind (moon) + the probe
    C.sun('moon', (-0.55, -0.35, 0.75), (0.42, 0.58, 1.0), opts.get('moon', 1.1), angle_deg=1.0)
    _probe((0.0, 0.0, 0.0), 0.12)
    cp = opts.get('cam', [0.022, -0.15, 0.095])
    _camera(C, cp, (0.0, 0.006, -0.004), opts.get('lens', 70.0), opts.get('fstop', 11.0),
            focus=(Vector(cp) - Vector((0.0, 0.0, -0.003))).length)


def _coals(C, new_material, n, spread, seed, heat, z0=0.0, exclude=None):
    """A bed of embers: noisy lumps with a dark char crust, pale ash on top and glowing cracks
    (voronoi edges) that glow brighter in the crevices. heat: 'fire' (orange) or 'white' (the heart)."""
    import random

    import bmesh
    from mathutils import Vector, noise
    rng = random.Random(seed)
    bm0 = bmesh.new()
    bmesh.ops.create_icosphere(bm0, subdivisions=3, radius=1.0)
    ico_v = [v.co.copy() for v in bm0.verts]
    ico_f = [tuple(v.index for v in f.verts) for f in bm0.faces]
    bm0.free()
    V, F = [], []
    placed = 0
    tries = 0
    while placed < n and tries < n * 20:
        tries += 1
        x, y = rng.uniform(-spread[0], spread[0]), rng.uniform(-spread[1], spread[1])
        if exclude and (math.hypot(x - exclude[0], y - exclude[1]) < exclude[2] or
                        (len(exclude) > 3 and y < exclude[3])):
            continue
        r = rng.uniform(0.006, 0.016)
        sx, sy, sz = r * rng.uniform(0.9, 1.5), r * rng.uniform(0.8, 1.3), r * rng.uniform(0.55, 0.9)
        z = z0 + sz * 0.45 + rng.uniform(-0.002, 0.006) * (1 - math.hypot(x / spread[0], y / spread[1]))
        off = Vector((rng.random() * 40, rng.random() * 40, rng.random() * 40))
        rot = rng.uniform(0, 6.283)
        cr, sr = math.cos(rot), math.sin(rot)
        o = len(V)
        for q in ico_v:
            d = 1.0 + 0.22 * noise.noise(q * 1.7 + off) + 0.08 * noise.noise(q * 5.0 + off)
            px, py, pz = q.x * sx * d, q.y * sy * d, q.z * sz * d
            V.append((x + px * cr - py * sr, y + px * sr + py * cr, z + pz))
        F.extend(tuple(o + i for i in f) for f in ico_f)
        placed += 1
    m, nb = new_material('coal_' + heat)
    P = nb.texco().outputs['Object']
    nz = nb.sep(nb.geo().outputs['Normal'])[2]
    warp = nb.noise(P, scale=90.0, detail=3.0, rough=0.6)
    Pw = nb.vadd(P, nb.vscale(nb.vsub(warp.outputs['Color'], (0.5, 0.5, 0.5)), 0.006))
    vor = nb.voronoi(Pw, scale=190.0, feature='DISTANCE_TO_EDGE')
    crack = nb.sstep(0.035, 0.0, vor.outputs['Distance'])
    hotn = nb.noise(P, scale=38.0, detail=4.0, rough=0.65)
    hot = nb.sstep(0.5, 0.78, hotn.outputs['Fac'])
    under = nb.sstep(0.4, -0.5, nz)                       # crevices / undersides glow most
    ash = nb.mul(nb.sstep(0.3, 0.8, nz), nb.sstep(0.42, 0.6, nb.noise(P, scale=120.0, detail=4.0).outputs['Fac']))
    base = nb.mixcol(ash, (0.018, 0.016, 0.015), (0.3, 0.29, 0.28))
    bs = nb.principled(Base_Color=base, Roughness=0.85)
    nb.link(nb.bump(nb.noise(P, scale=600.0, detail=4.0).outputs['Fac'], 0.4, 0.0008), bs.inputs['Normal'])
    if heat == 'white':
        glow_c, k_crack, k_surf = (1.0, 0.6, 0.2), 1.0, 2.0
    else:
        glow_c, k_crack, k_surf = (1.0, 0.2, 0.025), 6.0, 1.4
    if heat == 'white':
        bs.inputs['Base Color'].default_value = (0.01, 0.008, 0.007, 1.0)
        for lk in list(bs.inputs['Base Color'].links):
            nb.nt.links.remove(lk)
        top = nb.sstep(-0.2, 0.9, nz)
        heat_v = nb.clamp01(nb.madd(hotn.outputs['Fac'], 1.4, -0.25))
        g = nb.add(nb.mul(crack, k_crack), nb.mul(nb.madd(top, 0.7, 0.3), nb.mul(heat_v, k_surf)))
        col = nb.mixcol(nb.mul(top, heat_v), (0.95, 0.16, 0.02), glow_c)
        em = nb.emission(col, g)
    else:
        g = nb.add(nb.mul(nb.mul(crack, nb.madd(hot, 0.8, 0.2)), k_crack),
                   nb.mul(nb.mul(nb.madd(under, 0.8, 0.2), hot), k_surf))
        g = nb.mul(g, nb.sub(1.0, nb.mul(ash, 0.85)))
        em = nb.emission(glow_c, g)
    nb.output(surface=nb.addshader(bs, em))
    return C.mesh_obj('coals_' + heat, V, F, mat=m)


def _flames(C, T, cam, specs):
    """Sprite flame cards (the film's own bonfire sprites), camera-facing, additive."""
    from mathutils import Vector

    from kit import fire as FK
    for i, (name, pos, gain) in enumerate(specs):
        sp = T['sprites'][name]
        ob = FK.flame_card(f'{name}{i}', Vector(pos), sp['card'], sp['first'], cam, gain=gain, fog=False)
        ob.scale = (1.0, 1.0, 1.0)


def _build_fire(C, new_material, R, opts, bg, T):
    """C16: the band hanging on the tip of her steel, just lifted out of the beacon's coals. Unmarked,
    its letters awake; the coals below, the fire behind, both soft."""
    import bpy
    from mathutils import Euler, Vector

    from kit import fire as FK
    bg.inputs[0].default_value = (0.02, 0.006, 0.002, 1.0)
    bg.inputs[1].default_value = 1.0
    _coals(C, new_material, 70, (0.1, 0.08), 11, 'fire', z0=-0.065)
    gold = gold_material(C, new_material, R['tex'], letters=opts.get('letters', 2.2), letter_col=(1.0, 0.3, 0.06))
    ring = ring_object('ring', gold)
    # hangs from the steel: axis roughly along the bar, tilted toward the camera
    ring.rotation_euler = Euler((math.radians(4.0), math.radians(90.0 - 34.0), math.radians(-18.0)), 'XYZ')
    ring.location = (0.0, 0.0, 0.0)
    # the steel: a forged flat bar entering from the left, rising a little, the tip through the ring,
    # heat colours (straw -> bronze -> blue) toward the tip
    st, nb = new_material('steel')
    P = nb.texco().outputs['Object']
    x = nb.sep(P)[0]
    tint = nb.ramp(nb.sstep(-0.09, 0.012, x), [(0.0, (0.06, 0.06, 0.065)), (0.55, (0.08, 0.075, 0.07)),
                                                  (0.72, (0.3, 0.22, 0.1)), (0.84, (0.26, 0.12, 0.08)),
                                                  (0.93, (0.1, 0.1, 0.22)), (1.0, (0.12, 0.12, 0.14))])
    bsx = nb.principled(Base_Color=tint, Metallic=1.0,
                        Roughness=nb.madd(nb.noise(P, scale=900.0, detail=3.0).outputs['Fac'], 0.2, 0.3))
    nb.link(nb.bump(nb.noise(P, scale=1400.0, detail=4.0).outputs['Fac'], 0.25, 0.0003), bsx.inputs['Normal'])
    nb.output(surface=bsx)
    y_top = R_IN - 0.0012
    pts = [(-0.16, 0.012, y_top - 0.028), (-0.06, 0.004, y_top - 0.01), (0.0, 0.0, y_top - 0.0019),
           (0.012, 0.0, y_top - 0.0012), (0.019, 0.0, y_top + 0.0012)]
    tv, tf = FK.tube(pts, 0.0021, 8, radii=[0.0024, 0.0022, 0.0019, 0.0016, 0.0011])
    stl = C.mesh_obj('steel', tv, tf, mat=st)
    stl.scale = (1.0, 1.0, 0.7)
    # lights: the bed's glow from below, the fire behind (rim), the probe
    L = bpy.data.lights.new('bed', 'AREA')
    L.shape = 'DISK'
    L.size = 0.16
    L.color = (1.0, 0.33, 0.07)
    L.energy = opts.get('bed_w', 6.0)
    ob = C.link_obj(bpy.data.objects.new('bed', L))
    ob.location = (0.0, 0.0, -0.04)
    C.point('rim', (0.02, 0.09, 0.03), (1.0, 0.55, 0.2), opts.get('rim_w', 3.0), radius=0.03)
    _probe((0.0, 0.0, 0.0), 0.25)
    cam = _camera(C, opts.get('cam', [0.1, -0.2, 0.03]), (0.004, 0.0, -0.004), opts.get('lens', 90.0),
                  opts.get('fstop', 5.6))
    _flames(C, T, cam, [('bgfire', (0.02, 0.2, -0.05), 0.45), ('bgfire', (-0.08, 0.26, -0.05), 0.4),
                        ('bgfire', (0.1, 0.24, -0.05), 0.35), ('lick', (0.01, 0.03, -0.035), 1.0),
                        ('lick', (-0.03, 0.02, -0.038), 0.9)])


def _build_melt(C, new_material, R, opts, bg, T):
    """C24: in the white heart of the fire the band slumps and runs into a bead; the letters flare once."""
    import bpy
    from mathutils import Euler
    bg.inputs[0].default_value = (1.0, 0.3, 0.06, 1.0)
    bg.inputs[1].default_value = opts.get('bg_w', 0.1)
    _coals(C, new_material, 70, (0.09, 0.08), 17, 'white', z0=-0.012, exclude=(0.0, 0.0, 0.017, -0.004))
    slab = C.mesh_obj('slab', [(-0.05, -0.05, -0.0027), (0.05, -0.05, -0.0027), (0.05, 0.05, -0.0027), (-0.05, 0.05, -0.0027)],
                      [(0, 1, 2, 3)], mat=bpy.data.materials['coal_white'], smooth=False)
    # a flat coal the Ring lies on
    gold = gold_material(C, new_material, R['tex'], letters=opts.get('letters', 5.0), letter_col=(1.0, 0.82, 0.5),
                         hot=opts.get('hot', 0.1))
    ring = _slump_object('ring', R['melt'], gold)
    ring.location = (0.0, 0.0, 0.5 * WIDTH - 0.0027)
    C.point('heart', (0.0, 0.02, 0.03), (1.0, 0.8, 0.55), opts.get('heart_w', 0.25), radius=0.02)
    L = bpy.data.lights.new('bed', 'AREA')
    L.shape = 'DISK'
    L.size = 0.12
    L.color = (1.0, 0.6, 0.25)
    L.energy = opts.get('bed_w', 1.0)
    ob = C.link_obj(bpy.data.objects.new('bed', L))
    ob.location = (0.0, 0.0, -0.006)
    ob.rotation_euler = Euler((math.pi, 0.0, 0.0))
    _probe((0.0, 0.0, 0.0), 0.2)
    cam = _camera(C, opts.get('cam', [0.03, -0.12, 0.034]), (0.0, 0.0, 0.0), opts.get('lens', 90.0),
                  opts.get('fstop', 8.0))
    _flames(C, T, cam, [('lick', (0.014, 0.024, -0.004), 0.35), ('lick', (-0.022, 0.02, -0.006), 0.3),
                        ('bgfire', (0.0, 0.14, -0.03), 0.3), ('bgfire', (-0.08, 0.18, -0.03), 0.25),
                        ('bgfire', (0.08, 0.17, -0.03), 0.25)])
