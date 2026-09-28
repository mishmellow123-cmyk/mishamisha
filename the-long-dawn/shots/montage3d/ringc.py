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
SHOTS = dict(find_a=(3000, 3079), find_b=(3080, 3149), fire=(3360, 3565), fire_catch=(3566, 3599), melt=(5360, 5519),
             flint_a=(2960, 2999), flint_b=(3150, 3359))
START, END = SHOTS[SHOT]
BEAT = dict(strike2=3009, close=3060, open=3090, vision=(3090, 3141), fist=3140,
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
          fire_catch=dict(exposure=0.9, bloom_strength=0.05, bloom_threshold=1.1, streak_strength=0.0,
                          vignette_amount=0.34, lift=0.003),
          melt=dict(exposure=0.55, bloom_strength=0.03, bloom_threshold=1.8, streak_strength=0.0,
                    vignette_amount=0.3, lift=0.002),
          flint_a=dict(exposure=1.0, bloom_strength=0.04, bloom_threshold=1.2, streak_strength=0.0,
                       vignette_amount=0.34, lift=0.002),
          flint_b=dict(exposure=1.0, bloom_strength=0.045, bloom_threshold=1.2, streak_strength=0.0,
                       vignette_amount=0.34, lift=0.002))
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
                FP.FlameSpec('lick', Hf=0.085, Rb=0.022, seed=7, I=22.0, tongues=5, lean=0.004, ppm=4500,
                             env=False),
                FP.FlameSpec('lick2', Hf=0.075, Rb=0.020, seed=13, I=22.0, tongues=4, lean=-0.004, ppm=4500,
                             env=False)]
    if SHOT == 'flint_b':
        class Bloom(FP.FlameSpec):
            """The catch: a small flame that blooms out of the smoking tinder and grows (no whoomp)."""

            def __init__(self, *a, f0=0, grow=20.0, **k):
                super().__init__(*a, env=False, **k)
                self.f0, self.grow = f0, grow

            def params(self, frame):
                t = frame / FPS
                u = (frame - self.f0) / self.grow
                if u < 0:
                    return t, 0.0, 0.0, 0.0, 0.0
                size = 0.2 + 0.8 * (1.0 - math.exp(-2.2 * u)) + 0.25 * _ease((u - 1.0) / 2.0)
                inten = min(1.0, 0.45 + 0.8 * u) * (1.0 + 0.35 * math.exp(-3.0 * u))
                return t, self.Hf * size, inten, 1.0, self.lean * size

        return [Bloom('tflame', Hf=0.065, Rb=0.015, seed=23, I=22.0, tongues=4, lean=0.003, ppm=3200,
                      max_size=1.6, f0=FLINT['catch'], grow=20.0),
                Bloom('kflame', Hf=0.10, Rb=0.030, seed=29, I=22.0, tongues=5, lean=0.005, ppm=2400,
                      max_size=1.5, f0=FLINT['kindle'], grow=22.0)]
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
    # the refrozen meltwater lies level at the bottom and the wall rises out of it with zero slope (C1): no step
    # at the disc's edge (a 0.6 mm step there caught the moon as a double jagged polygon line)
    cup = 0.00008
    wall = dp * (1.0 - t * t * (3 - 2 * t)) ** 1.25            # a smooth bowl wall
    bottom = -dp + cup * (R / rd) ** 2                         # the disc: level, a hair concave
    Z = np.where(R < rd, bottom, cup - wall)
    rim = 0.0014 * np.exp(-((R - rt * 1.04) / (0.18 * rt)) ** 2) * (1.0 + 0.4 * np.sin(ang * 7.0 + 1.1))
    glaze = np.clip((rt * 1.02 - R) / (0.12 * rt), 0.0, 1.0)
    ice = np.clip((rd * 1.04 - R) / (0.16 * rd), 0.0, 1.0) ** 1.5
    Z = Z + rim + soft * np.clip((R - rt) / (0.8 * rt), 0.0, 1.0) + grain * (1.0 - 0.9 * glaze)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    CH = np.stack([glaze.reshape(-1), ice.reshape(-1), np.clip(rim / 0.0014, 0, 1).reshape(-1)], 1)
    _write_grid(path, V, _grid_quads(n, n), CH)


def _vision_strip(path, u, t=0.0):
    """The band's vision (find_b), an 8-bit strip mapped on the inner face over the arc the camera sees (strip u
    0.30-0.70; the middle two thirds show). Painted, not drawn: five of C6's forge-towers (broad kiln-like stacks
    of ember-flecked masonry, rings of embers, a glow at each crown) stand in a smoky forge-light, and as u rises
    0..1 they bow toward the middle, toward her: the outer ones lean in from the waist, the middle one dips its
    crown toward us. t (seconds) moves the smoke. No windows, no real place, no symbol, no Eye."""
    import cv2
    import numpy as np
    W, H = 1600, 360
    ss = 2
    Ws, Hs = W * ss, H * ss
    rng = np.random.default_rng(29)
    yy = (np.arange(Hs, dtype=np.float32) + 0.5)[:, None] / Hs
    xx = (np.arange(Ws, dtype=np.float32) + 0.5)[None, :] / Ws
    hz = 0.90

    def fbm(sx, sy, oct_, seed, drift=0.0):
        acc = np.zeros((Hs, Ws), np.float32)
        r = np.random.default_rng(seed)
        amp, tot = 1.0, 0.0
        for o in range(oct_):
            gh, gw = int(3 * 2 ** o * sy) + 2, int(3 * 2 ** o * sx) + 2
            g = r.random((gh, gw)).astype(np.float32)
            layer = cv2.resize(g, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
            acc += amp * np.roll(layer, int(drift * Ws * (0.5 + 0.3 * o)) % Ws, axis=1)
            tot += amp
            amp *= 0.55
        return acc / tot

    smoke = fbm(4.0, 1.0, 5, 3, drift=0.012 * t)
    d2 = ((xx - 0.5) / 0.42) ** 2 + ((yy - hz) / 0.75) ** 2
    glow = np.exp(-d2 * 1.4) * (0.7 + 0.6 * smoke)
    glow = glow * (yy < hz)
    sky = np.stack([0.008 + 0.10 * glow ** 1.6, 0.010 + 0.56 * glow ** 1.25, 0.025 + 0.97 * glow], -1)   # BGR
    img = sky * (0.45 + 0.55 * smoke[..., None])
    xs = [0.19, 0.345, 0.50, 0.655, 0.81]
    hs = [0.66, 0.74, 0.62, 0.76, 0.64]
    ws = [0.080, 0.085, 0.095, 0.082, 0.078]
    fleck = cv2.GaussianBlur((rng.random((Hs, Ws)) > 0.985).astype(np.float32), (0, 0), 1.0) * 5.0
    for i, x0 in enumerate(xs):
        side = (x0 - 0.5) / 0.31                               # -1 .. 1 (0 = the middle tower)
        h = hs[i] * Hs
        w0 = ws[i] * Ws
        lean = -math.copysign(1.0, side) * math.radians(48.0) * min(1.0, abs(side) * 1.3) * u if side else 0.0
        dip = (0.32 * u) if not side else 0.12 * u               # bowing toward us: the crown comes down
        n = 64
        cx, cy = x0 * Ws, hz * Hs
        L, R, C_, A, WW = [], [], [], [], []
        for j in range(n + 1):
            tt = j / n
            a = lean * max(0.0, tt - 0.30) ** 1.4 / 0.70 ** 1.4
            if j:
                step = h * (1.0 - dip * max(0.0, tt - 0.4) / 0.6) / n
                cx += math.sin(a) * step
                cy -= math.cos(a) * step
            ww = w0 * (1.0 - 0.28 * tt)                         # a kiln's taper
            ww *= 1.0 - 0.07 * (tt > 0.34) - 0.06 * (tt > 0.66)   # stages
            ww *= 1.0 + 0.30 * math.exp(-((tt - 0.95) / 0.035) ** 2)   # the flared cap
            nx, ny = math.cos(a), math.sin(a)
            L.append((cx - nx * ww * 0.5, cy - ny * ww * 0.5))
            R.append((cx + nx * ww * 0.5, cy + ny * ww * 0.5))
            C_.append((cx, cy))
            A.append(a)
            WW.append(ww)
        poly = np.array(L + R[::-1], np.float64)
        m = np.zeros((Hs, Ws), np.uint8)
        cv2.fillPoly(m, [np.round(poly * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        mf = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 1.2)
        # masonry: dark, ember-flecked, three rings of embers round it; the side toward the forge-glow is lit
        ring = np.zeros((Hs, Ws), np.float32)
        for tt in sorted(rng.uniform(0.2, 0.86, 2 + (i % 2))):
            j = int(tt * n)
            cv2.line(ring, tuple(int(q * 4) for q in L[j]), tuple(int(q * 4) for q in R[j]), 1.0,
                     max(2, int(0.035 * Hs)), cv2.LINE_AA, shift=2)
        ring = cv2.GaussianBlur(ring, (0, 0), 2.0) * (0.4 + 0.6 * fleck)
        lit = np.clip((xx - x0) * (-np.sign(side) if side else 0.0) / (ws[i] * 0.5) + 0.3, 0, 1) if side else 0.25
        tone = mf[..., None] * (np.array([0.010, 0.022, 0.050], np.float32) * (1.0 + 2.5 * lit)[..., None] if side
                                else np.array([0.014, 0.030, 0.065], np.float32))
        img = img * (1.0 - 0.95 * mf[..., None]) + tone
        img += (mf * (0.06 * fleck + 0.9 * ring))[..., None] * np.array([0.12, 0.56, 1.0], np.float32)
        # the crown's fire: a soft glow sitting on the cap (it tips with the bow)
        tx, ty = C_[-1]
        a = A[-1]
        gx = tx + math.sin(a) * 0.05 * Hs
        gy = ty - math.cos(a) * 0.05 * Hs
        fl = 1.0 + 0.25 * math.sin(6.0 * t + i) + 0.15 * math.sin(13.0 * t + 2 * i)
        gl = np.exp(-(((xx * Ws - gx) / (WW[-1] * 0.36)) ** 2 + ((yy * Hs - gy) / (0.045 * Hs * fl)) ** 2))
        img += gl[..., None] * np.array([0.30, 0.76, 1.0], np.float32) * 0.9 * fl
    for k in range(50):                                        # sparks drifting up
        sx_ = rng.uniform(0.1, 0.9) * Ws
        sy_ = ((rng.uniform(0, 1) - 0.07 * t * (1 + rng.uniform())) % 1.0) * hz * Hs
        cv2.circle(img, (int(sx_), int(sy_)), 1 + int(rng.uniform() * 2), (0.2, 0.6, 1.0), -1, cv2.LINE_AA)
    haze = fbm(3.0, 1.0, 4, 11, drift=-0.02 * t)
    img = img * (0.8 + 0.4 * haze[..., None]) + (0.12 * haze * glow)[..., None] * np.array([0.1, 0.4, 1.0], np.float32)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    cv2.imwrite(path, np.clip(img * 255.0, 0, 255).astype(np.uint8))


def prep(frames, cache):
    d = os.path.join(cache, 'data')
    os.makedirs(d, exist_ok=True)
    info = dict(shot=SHOT, tex=os.path.join(ADIR, 'inscription_outer.png'),
                tex_in=os.path.join(ADIR, 'inscription_inner.png'))
    redo = bool(os.environ.get('MT3D_RERING'))
    if SHOT == 'find_a':
        p = os.path.join(d, 'snowpit.bin')
        if not os.path.exists(p) or redo:
            tmp = f'{p}.{os.getpid()}.tmp'                     # atomic: renders sharing a farm node read it too
            _snow_pit(tmp)
            os.replace(tmp, p)
        info['snow'] = p
    if SHOT == 'find_b':
        vd = os.path.join(d, 'vision')
        os.makedirs(vd, exist_ok=True)
        v0, v1 = BEAT['vision']
        need = [f for f in frames if v0 - 3 <= f <= v1 + 3] or [v0]
        for f in need:
            vp = os.path.join(vd, f'vision_{f:05d}.png')
            if not os.path.exists(vp) or redo:
                _vision_strip(vp, _ease(min(1.0, max(0.0, (f - v0 - 4) / (v1 - v0 - 16)))), t=f / FPS)
        info['vision'] = os.path.join(vd, f'vision_{need[0]:05d}.png')
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
         vision=None, rough=0.05, molten=None):
    """Polished yellow gold (no glitter: roughness varies only with slow polish marks). letters: keyed emission
    strength of the script (outer + inner faces); flare: keyed 0..1 shift deep -> bright; hot: keyed incandescence
    of the metal itself; vision: (image-sequence path, keyed strength) on the inner face; molten: the melt front
    (keyed half-angle, radians, from the pool at strip u = 0): the metal behind the front is liquid, glows hotter and
    loses its polish marks."""
    import bpy
    m, nb = new_material('gold_c')
    P = nb.texco().outputs['Object']
    polish = nb.noise(P, scale=160.0, detail=2.0, rough=0.5)
    uvn = nb.n('ShaderNodeUVMap')
    uvn.uv_map = 'strip'
    u_, v_, _ = nb.sep(uvn.outputs['UV'])
    liquid = None
    if molten:
        dlt = nb.math('ABSOLUTE', nb.mul(u_, 2 * math.pi))       # the melt band's u runs -0.5..0.5 round the pool
        liquid = nb.sstep(-0.10, 0.45, nb.sub(_keyv(nb, 'front', molten), dlt))
    rgh = nb.madd(polish.outputs['Fac'], 0.045, rough - 0.005)
    if liquid is not None:
        rgh = nb.mixf(liquid, rgh, 0.015)
    bs = nb.principled(Base_Color=(1.0, 0.71, 0.29), Metallic=1.0, Roughness=rgh)
    bs.inputs['Specular Tint'].default_value = (1.0, 0.93, 0.78, 1.0)
    shader = bs
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
        hk = _keyv(nb, 'hot', hot)
        if liquid is not None:                                     # solid metal only warms; the liquid glows
            hk = nb.mul(hk, nb.madd(liquid, 1.0, 0.12))
        emis.append(nb.emission(bb.outputs['Color'], hk))
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
        # the arc the camera sees (inner u 0.30..0.70 of the strip) fills the image; a heat shimmer ripples it. Seen
        # from outside the band through its hole, u runs right to left on the far inner face: flip so it reads true
        t = _keyv(nb, 'vt', [(f, f / FPS) for f in range(START - 2, END + 3)])
        sh = nb.noise(nb.comb(nb.mul(u_, 90.0), nb.mul(v_, 14.0), nb.mul(t, 2.2)), scale=1.0, detail=2.0, dims='3D')
        du = nb.mul(nb.sub(sh.outputs['Fac'], 0.5), 0.010)
        vu = nb.add(nb.div(nb.sub(0.70, u_), 0.40), du)
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
            if f < f0:
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
            ob.location = at(max(f, f0))                    # before birth it waits at its start; after, it stays
            #                                                  where it died (a parked bead far away drew metres-long
            #                                                  motion-blur streaks as it jumped in) -- hidden both times
            ob.keyframe_insert('location', frame=f)
            ob.hide_render = not (f >= f0 and (f < f1 or (land and f - f1 < land)))
            ob.keyframe_insert('hide_render', frame=f)
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
    T = dict(T, _frames=list(job['frames']))
    dict(find_a=_find_a, find_b=_find_b, fire=_fire, fire_catch=_fire_catch, melt=_melt, flint_a=_flint,
         flint_b=_flint)[SHOT](
        C, new_material, R, opts, T)
    for ob in bpy.data.objects:                       # lamps light the scene; the lens never sees them as shapes
        if ob.type == 'LIGHT':
            try:
                ob.visible_camera = False
            except Exception:
                pass
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
    # the flash is the strike itself, just out of frame above left (not a lamp far away): it falls off across the
    # frame, rakes the pit's rim and is gone in a few frames
    # POLISH (director): under the flash the snow stays cold white; the warmth is only in the speculars (the gold, the
    # ice, the glints) and the sparks. So the flash is two lamps at the strike: a warm one seen only by glossy rays,
    # and a cold-white one that lights the snow's diffuse
    fpos = tuple(opts.get('flash_pos', (-0.20, 0.05, 0.25)))
    fl = C.point('flash', fpos, (1.0, 0.50, 0.18), 0.0, radius=0.010)
    flc = C.point('flash_cool', fpos, tuple(opts.get('flash_cool_col', (0.80, 0.88, 1.0))), 0.0, radius=0.010)
    for ob_, attr in ((fl, 'visible_diffuse'), (flc, 'visible_glossy')):
        try:
            setattr(ob_, attr, False)
        except Exception as e:
            print('find_a: flash split unavailable:', e, flush=True)
    for f in range(START - 2, END + 3):
        e = 0.0 if f < s2 else math.exp(-(f - s2) / 1.5)
        C.key(fl.data, 'energy', f, opts.get('flash_w', 0.55) * e)
        C.key(flc.data, 'energy', f, opts.get('flash_cool_w', 0.30) * e)
    # POLISH (director): 3033-3060 the glove must read as a hand: a thin warm rim from the embers (her tinder, low
    # behind the hand) on the finger ridges and knuckles, so the digits separate and a stitch line catches it.
    # Light-linked to the glove alone: the snow never sees it.
    rim = C.point('glove_rim', tuple(opts.get('rim_pos', (-0.14, 0.11, 0.050))), (1.0, 0.42, 0.12), 0.0,
                  radius=0.012)
    try:
        rc = bpy.data.collections.new('glove_rim_recv')
        rc.objects.link(G.mesh)
        rim.light_linking.receiver_collection = rc
    except Exception as e:
        print('find_a: light linking unavailable:', e, flush=True)
    for f in range(START - 2, END + 3):
        k = _ease((f - 3029) / 8.0) * (1.0 - 0.6 * _ease((f - 3066) / 12.0)) * _flick(f, 13, 0.5)
        C.key(rim.data, 'energy', f, opts.get('rim_w', 0.030) * k)
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
    # the night: a faint cold moon from behind right, LOW (15 deg): it grazes the snow's grain and rims the glove, but
    # the pit's far rim shades the whole floor, so the band stays dark until strike 2's flash shows it
    C.sun('moon', opts.get('moon_dir', (0.273, 0.962, 0.268)), (0.55, 0.66, 1.0), opts.get('moon', 0.34), angle_deg=1.5)
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

def _plane(C, name, z, half, mat):
    return C.mesh_obj(name, [(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)], [(0, 1, 2, 3)],
                      mat=mat, smooth=False)


def _find_b(C, new_material, R, opts, T):
    """THE FIND (b), 3080-3149: her own eyes. She has lifted the fist close to her face (a low look across the palm,
    ~27 deg), palm up, fingers away from her. The fist opens (3084-3092): the band lies in the cup of the palm. From
    3090 a vision wakes on the far inner face of the band, the polish turned window: forge-towers bowing toward her,
    in a red forge-glow that warms the leather round it. It fades by 3138; the fist closes on it (3136-3141)."""
    import bpy
    from mathutils import Euler, Matrix, Quaternion, Vector

    import glove as GL
    _world(opts.get('sky', (0.10, 0.16, 0.36)), opts.get('sky_w', 0.12))
    _plane(C, 'snow_far', -0.30, 1.5, snow_material(new_material))           # her knees' snow, far out of focus
    leather = GL.leather_material(new_material, base=opts.get('leather', (0.075, 0.047, 0.029)))
    G = GL.Glove('glove', leather, GL.wool_material(new_material), mirror=True)
    v0, v1 = BEAT['vision']
    fist_t = BEAT['fist']
    # palm up, fingers toward +Y, thumb toward -X (her LEFT hand): local x -> Y, local y -> X, local z -> -Z; the
    # fingers rise a little (the forearm falls away below the camera)
    R0 = Matrix(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
    Rh = (Euler((math.radians(opts.get('hand_pitch', 12.0)), math.radians(opts.get('hand_roll', -8.0)), 0.0),
                'XYZ').to_matrix() @ R0).to_4x4()
    T0 = Vector(opts.get('wrist', (0.0, -0.050, 0.0)))

    def place(f):
        t = f / FPS
        # a held hand is never still: a slow drift and a breath (under a millimetre)
        d = Vector((0.0005 * math.sin(1.3 * t + 0.4) + 0.0003 * math.sin(3.1 * t),
                    0.0004 * math.sin(0.9 * t + 1.1),
                    0.0007 * math.sin(1.7 * t + 2.0) + 0.0003 * math.sin(4.3 * t + 0.5)))
        return Matrix.Translation(T0 + d) @ Rh

    cupped = GL.pose_mix(GL.POSES['cup'], GL.POSES['relaxed'], opts.get('open_mix', 0.35))
    fistp = GL.POSES['fist']

    def pose(f):
        if f < 3081:
            return fistp
        if f < fist_t - 4:
            return GL.pose_mix(fistp, cupped, _eout((f - 3081) / 9.0))
        return GL.pose_mix(cupped, fistp, _ein((f - (fist_t - 4)) / 5.0) ** 0.8)

    for f in range(START - 2, END + 3):
        G.key_place(f, place(f))
        G.key(f, pose(f))
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    # where does the palm lie? ray-cast straight down onto the open (cupped) glove at the hold frame
    sc.frame_set(3110)
    dg.update()
    Mh = place(3110)
    mir = Matrix.Diagonal((1.0, -1.0, 1.0, 1.0))
    # the ring rests in the hollow of the palm, a little toward the finger roots
    spot = Mh @ mir @ Vector(opts.get('palm_spot', (0.050, 0.004, -0.02)))
    hit, loc, nrm, _i, _o, _m = sc.ray_cast(dg, spot + Vector((0.0, 0.0, 0.05)), Vector((0.0, 0.0, -1.0)))
    if not hit:
        print('find_b: palm ray missed; using the spot', flush=True)
        loc, nrm = spot, Vector((0.0, 0.0, 1.0))
    print('find_b palm', tuple(round(c, 4) for c in loc), 'normal', tuple(round(c, 3) for c in nrm), flush=True)
    def vis_k(f):                                     # wakes with the open hand; shut in by the closing fist
        return _ease((f - v0) / 12.0) * (1.0 - _ease((f - (v1 - 8)) / 8.0))

    g = gold(new_material, R, vision=(R['vision'], _tab(lambda f: opts.get('vision_w', 3.2) * vis_k(f))))
    ring, _ = band_mesh('ring', g)
    ring.rotation_mode = 'QUATERNION'
    # it lies in the palm, its axis along the palm's normal (tilted by the cup), turned so that the vision's arc
    # (strip u 0.30-0.70, centred on the band's local -x) is the far inner face, the one that looks at her
    up = Vector(nrm).normalized()
    if up.z < 0:
        up = -up
    up = up.lerp(Vector((0.0, 0.0, 1.0)), 0.5).normalized()          # a cupped palm is not a plane: split the tilt
    ax = -(Vector((0.0, 1.0, 0.0)) - up * up.y).normalized()          # band local +x: toward her (so -x = the far side)
    ay = up.cross(ax)
    rest_m = Matrix((ax, ay, up)).transposed()                        # columns = the band's local axes in the world
    lie = Matrix.Translation(Vector(loc) + up * (0.5 * WIDTH + opts.get('lift', 0.0003))) @ rest_m.to_4x4()
    lie_local = Mh.inverted() @ lie                                   # rides the hand (drift and breath)
    # inside the fist (before 3084 and after the close) it sits deeper and on edge in the fingers' hollow
    sc.frame_set(START)
    pb = G.rig.pose.bones
    hol = (pb['m1'].head + pb['m2'].head + pb['m3'].head + pb['m3'].tail) / 4.0
    hol = Vector((hol.x, -hol.y, hol.z))
    in_fist = Matrix.Translation(hol) @ Euler((math.radians(70.0), 0.0, math.radians(10.0))).to_matrix().to_4x4()
    l0, r0, _ = in_fist.decompose()
    l1, r1, _ = lie_local.decompose()
    for f in range(START - 2, END + 3):
        if f < fist_t - 4:
            u = _eout((f - 3081) / 9.0)
        else:
            u = 1.0 - _ein((f - (fist_t - 4)) / 5.0) ** 0.8
        Ml = Matrix.Translation(l0.lerp(l1, u)) @ r0.slerp(r1, u).to_matrix().to_4x4()
        l, r, _ = (place(f) @ Ml).decompose()
        ring.location = l
        ring.rotation_quaternion = r
        ring.keyframe_insert('location', frame=f)
        ring.keyframe_insert('rotation_quaternion', frame=f)
    # the vision's own glow on the leather round it (diffuse only: the gold shows the vision, not a lamp)
    vl = C.point('visionglow', Vector(loc) + up * 0.004 + Vector((0.0, 0.004, 0.0)), (1.0, 0.36, 0.10), 0.0,
                 radius=0.006, spec=0.0)
    try:
        vl.visible_glossy = False                                     # Cycles: no highlight of its own on the gold
    except Exception:
        pass
    for f in range(START - 2, END + 3):
        k = vis_k(f) * (1.0 + 0.08 * math.sin(f * 1.7) + 0.05 * math.sin(f * 4.1))
        C.key(vl.data, 'energy', f, opts.get('glow_w', 0.008) * k)
    for f in range(START - 2, END + 3):
        l, r, _ = (place(f) @ Matrix.Translation(l1)).decompose()
        vl.location = l + up * 0.004
        vl.keyframe_insert('location', frame=f)
    # the night: a cold moon beyond her fingers, at the mirror angle of the lens, lays a sheen on the leather and
    # rims the fingers; the warm light on the palm is the vision's own
    C.sun('moon', opts.get('moon_dir', (-0.15, 0.82, 0.57)), (0.55, 0.66, 1.0), opts.get('moon', 0.85), angle_deg=3.0)
    # the night's cool fill from the lens side: enough to read the fist before it opens and after it closes; while the
    # vision is awake the eye is on its glow and the fill sinks (an eye adapting, not a lamp being turned)
    nf = C.sun('nightfill', opts.get('fill_dir', (-0.55, -0.45, 0.70)), (0.50, 0.60, 1.0), 0.0, angle_deg=20.0)
    for f in range(START - 2, END + 3):
        k = 1.0 - 0.975 * _ease((f - 3084) / 12.0) + 0.975 * _ease((f - 3134) / 10.0)
        C.key(nf.data, 'energy', f, opts.get('fill', 16.0) * k)          # the vision frames keep ~0.4, as tested
    if opts.get('debug_light'):
        C.sun('dbg', (0.3, -0.6, 0.7), (1, 1, 1), opts['debug_light'], angle_deg=5.0)
    cen = Vector(loc) + up * (0.5 * WIDTH)
    tgt = cen + Vector(opts.get('tgt_off', (0.0, 0.45 * R_IN, 0.001)))
    el = math.radians(opts.get('cam_el', 35.0))
    vdir = Vector((opts.get('cam_x', 0.13), -math.cos(el), math.sin(el))).normalized()
    d0, d1, d2 = opts.get('cam_dists', (0.28, 0.112, 0.24))          # POLISH: the fist bookends show the whole fist
    far_inner = cen + Vector((0.0, 0.8 * R_IN, 0.0))
    fist_pt = cen + Vector((0.0, 0.012, 0.012))
    cam = _camera(tgt + vdir * d0, tgt, opts.get('lens', 100.0), opts.get('fstop', 25.0))
    for f in range(START - 2, END + 3):
        push = _ease((f - 3083) / 28.0)                                # drawn in as the vision wakes
        back = _ease((f - 3134) / 14.0)                                # and out as the fist closes on it
        dd = d0 + (d1 - d0) * push + (d2 - d1) * back
        p_ = tgt + vdir * dd
        cam.location = p_
        cam.keyframe_insert('location', frame=f)
        foc = fist_pt.lerp(far_inner, _ease((f - 3084) / 8.0)).lerp(fist_pt, _ease((f - 3136) / 6.0))
        cam.data.dof.focus_distance = (p_ - foc).length
        cam.data.dof.keyframe_insert('focus_distance', frame=f)


def _card_no_light(ob):
    """A flame card is seen (camera) and reflected (glossy: the gold shows the fire) but lights nothing diffuse:
    in Cycles an emissive card is a lamp, and these washed her dark glove out to cream clay."""
    for attr, v in (('visible_diffuse', False), ('visible_shadow', False), ('visible_transmission', False),
                    ('visible_volume_scatter', False), ('visible_glossy', True), ('visible_camera', True)):
        try:
            setattr(ob, attr, v)
        except Exception:
            pass


def _flick(f, seed=0, amt=1.0):
    """The montage toolkit's organic flicker (mt.fire.flicker), numpy-free for the Blender side."""
    t = f / FPS
    s = seed * 1.7
    v = (0.07 * math.sin(2 * math.pi * 1.9 * t + s) + 0.05 * math.sin(2 * math.pi * 3.7 * t + 2.1 + s)
         + 0.035 * math.sin(2 * math.pi * 7.3 * t + 0.7 * s) + 0.02 * math.sin(2 * math.pi * 11.1 * t + s))
    return 1.0 + amt * v


STEEL_HB, STEEL_RB, STEEL_LA = 0.043, 0.009, 0.068      # half the back; the bends' radius; the long arm
STEEL_RBL, STEEL_LL = 0.0085, 0.013                      # the lower bend's radius; the short lower arm
STEEL_D = 0.0026


def _steel_path():
    """Her C-shaped fire-steel (the one she strikes with), in its own plane: s to the right, t up, metres, the origin
    at the middle of its straight back, which is inside her fist. The back is long (8.6 cm) so BOTH bends clear her
    fist and the C reads: a scroll curled up at the end of the short lower arm, the lower bend, the back, the upper
    bend, the long upper arm and, at its end, a small up-turned lip. [(s, t, w)] along the centreline, w = the bar's
    width in the plane. Forged flat stock, 2.6 mm thick (the broad face is the C)."""
    HB, rb, LA, rbl, LL = STEEL_HB, STEEL_RB, STEEL_LA, STEEL_RBL, STEEL_LL
    P = []
    t_low = -HB - rbl                                           # the lower arm's centreline
    cs, ct = rbl + LL, t_low + 0.0073                           # the scroll's centre, above the lower arm's end
    n = 40
    for i in range(n + 1):                                      # the scroll: inner end first, unwinding outward
        u = i / n
        a = math.radians(270.0 - 360.0 * u)
        r = 0.0024 + 0.0049 * u
        P.append((cs + r * math.cos(a), ct + r * math.sin(a), 0.0030 + 0.0028 * u))
    for i in range(1, 7):                                       # the lower arm, leftward
        u = i / 6.0
        P.append((cs + (rbl - cs) * u, t_low, 0.0058 + 0.0010 * u))
    for i in range(1, 9):                                       # the lower bend
        a = math.radians(-90.0 - 90.0 * i / 8.0)
        P.append((rbl + rbl * math.cos(a), -HB + rbl * math.sin(a), 0.0070 + 0.0008 * i / 8.0))
    nb_ = 24
    for i in range(1, nb_ + 1):                                 # the back (in the fist), a hair of hammered wave
        u = i / float(nb_)
        P.append((0.0003 * math.sin(u * math.pi * 2.0), -HB + 2.0 * HB * u, 0.0078))
    for i in range(1, 9):                                       # the upper bend
        a = math.radians(180.0 - 90.0 * i / 8.0)
        P.append((rb + rb * math.cos(a), HB + rb * math.sin(a), 0.0076 - 0.0006 * i / 8.0))
    for i in range(1, 31):                                      # the long upper arm, tapering, a hair of droop
        u = i / 30.0
        P.append((rb + LA * u, _steel_arm_c(u), 0.0070 - 0.0026 * u))
    t_end = _steel_arm_c(1.0)
    for i in range(1, 9):                                       # the lip, turned up
        a = math.radians(-90.0 + 115.0 * i / 8.0)
        P.append((rb + LA + 0.0042 * math.cos(a), t_end + 0.0042 + 0.0042 * math.sin(a), 0.0044 - 0.0010 * i / 8.0))
    return P


def _steel_arm_c(u):
    """The long arm's centreline t at u (0 = the upper bend, 1 = the lip)."""
    return STEEL_HB + STEEL_RB - 0.0022 * u * u + 0.00025 * math.sin(u * 9.0)


def _steel_arm_top(s):
    """The upper arm's top edge (t) at s, from the same formula as the path."""
    u = min(max((s - STEEL_RB) / STEEL_LA, 0.0), 1.0)
    return _steel_arm_c(u) + 0.5 * (0.0070 - 0.0026 * u)


def _steel_arm_bot(s):
    u = min(max((s - STEEL_RB) / STEEL_LA, 0.0), 1.0)
    return _steel_arm_c(u) - 0.5 * (0.0070 - 0.0026 * u)


S_REST = STEEL_RB + 0.84 * STEEL_LA        # where the band hangs on the arm
S_LIP = STEEL_RB + STEEL_LA - 0.0018       # where it rests against the lip


def _steel_mesh(C, mat):
    """Sweep a rounded-rectangle section (w in the plane x STEEL_D deep) along _steel_path. Steel-local frame: x = s,
    z = t, y = depth (toward -y faces the camera). Attributes: 'tipd' (m from the lip's end along the bar)."""
    import array

    import bpy
    P = _steel_path()
    n = len(P)
    L = [0.0]
    for i in range(1, n):
        L.append(L[-1] + math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]))
    tot = L[-1]
    sec = []
    ns = 16
    for k in range(ns):
        a = 2 * math.pi * (k + 0.5) / ns
        ca, sa = math.cos(a), math.sin(a)
        sec.append((_sgnpow(ca, 0.45), _sgnpow(sa, 0.45)))     # a soft-cornered rectangle (forged, not machined)
    V = array.array('f')
    tipd = array.array('f')
    for i, (s, t, w) in enumerate(P):
        a0, a1 = P[max(i - 1, 0)], P[min(i + 1, n - 1)]
        ts, tt = a1[0] - a0[0], a1[1] - a0[1]
        ln = math.hypot(ts, tt) or 1.0
        ns_, nt_ = -tt / ln, ts / ln                           # in-plane normal
        wob = 1.0 + 0.04 * math.sin(L[i] * 900.0) + 0.03 * math.sin(L[i] * 2300.0 + 1.0)   # hammer-forged, uneven
        for (cx, cy) in sec:
            V.extend((s + ns_ * cx * 0.5 * w * wob, cy * 0.5 * STEEL_D * (1.0 + 0.05 * math.sin(L[i] * 1500.0)),
                      t + nt_ * cx * 0.5 * w * wob))
            tipd.append(tot - L[i])
    F = []
    for i in range(n - 1):
        for k in range(ns):
            k2 = (k + 1) % ns
            F.append((i * ns + k, i * ns + k2, (i + 1) * ns + k2, (i + 1) * ns + k))
    F.append(tuple(range(ns))[::-1])
    F.append(tuple((n - 1) * ns + k for k in range(ns)))
    me = bpy.data.meshes.new('steel')
    me.from_pydata([tuple(V[j:j + 3]) for j in range(0, len(V), 3)], [], F)
    me.update()
    me.shade_smooth()
    at = me.attributes.new('tipd', 'FLOAT', 'POINT')
    at.data.foreach_set('value', tipd)
    me.materials.append(mat)
    ob = bpy.data.objects.new('steel', me)
    bpy.context.scene.collection.objects.link(ob)
    sd = ob.modifiers.new('sub', 'SUBSURF')
    sd.levels, sd.render_levels = 1, 1
    return ob


def steel_material(new_material, heat_tab):
    """Hand-forged steel: near-black scale with a dull grey sheen, hammer pocks, bright worn metal only on the
    striking edge; the part in the flames glows dull red as it heats (keyed 'heat'), fading along the bar."""
    m, nb = new_material('firesteel')
    P = nb.texco().outputs['Object']
    mott = nb.noise(P, scale=700.0, detail=4.0, rough=0.6)
    pock = nb.voronoi(P, scale=520.0, feature='SMOOTH_F1')
    col = nb.mixcol(nb.sstep(0.35, 0.75, mott.outputs['Fac']), (0.030, 0.029, 0.028), (0.075, 0.068, 0.062))
    bs = nb.principled(Base_Color=col, Metallic=0.85, Roughness=nb.madd(mott.outputs['Fac'], 0.18, 0.36))
    nb.link(nb.bump(nb.add(nb.mul(pock.outputs['Distance'], 1.2), nb.mul(mott.outputs['Fac'], 0.3)), 0.5, 0.0004),
            bs.inputs['Normal'])
    tipd = nb.attr('tipd').outputs['Fac']
    near = nb.sstep(0.045, 0.004, tipd)                        # hottest at the lip, fading along the arm
    glow = nb.mul(nb.mul(near, near), _keyv(nb, 'heat', heat_tab))
    col = nb.mixcol(nb.sstep(0.02, 0.0, tipd), (0.30, 0.018, 0.002), (0.85, 0.16, 0.02))
    em = nb.emission(col, nb.mul(glow, nb.madd(mott.outputs['Fac'], 1.2, 1.4)))
    nb.output(surface=nb.addshader(bs, em))
    return m


def _coals(C, new_material, n, spread, seed, heat, z0=0.0, exclude=None, gain_tab=None):
    """A bed of embers (ring3d's): noisy lumps with a dark char crust, pale ash on top, glowing cracks brighter in the
    crevices. heat: 'fire' (orange) or 'white' (the heart). gain_tab: keyed glow multiplier."""
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
        if exclude and math.hypot(x - exclude[0], y - exclude[1]) < exclude[2]:
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
    rid = nb.noise(Pw, scale=110.0, detail=6.0, rough=0.62)
    crack = nb.sstep(0.030, 0.0, nb.math('ABSOLUTE', nb.sub(rid.outputs['Fac'], 0.5)))
    hotn = nb.noise(P, scale=38.0, detail=4.0, rough=0.65)
    hot = nb.sstep(0.5, 0.78, hotn.outputs['Fac'])
    under = nb.sstep(0.4, -0.5, nz)
    ash = nb.mul(nb.sstep(0.3, 0.8, nz), nb.sstep(0.42, 0.6, nb.noise(P, scale=120.0, detail=4.0).outputs['Fac']))
    base = nb.mixcol(ash, (0.018, 0.016, 0.015), (0.3, 0.29, 0.28))
    bs = nb.principled(Base_Color=base, Roughness=0.85)
    nb.link(nb.bump(nb.noise(P, scale=600.0, detail=4.0).outputs['Fac'], 0.4, 0.0008), bs.inputs['Normal'])
    if heat == 'white':
        glow_c, k_crack, k_surf = (1.0, 0.55, 0.18), 1.0, 2.0
        top = nb.sstep(-0.2, 0.9, nz)
        heat_v = nb.clamp01(nb.madd(hotn.outputs['Fac'], 1.4, -0.25))
        g = nb.add(nb.mul(crack, k_crack), nb.mul(nb.madd(top, 0.7, 0.3), nb.mul(heat_v, k_surf)))
        col = nb.mixcol(nb.mul(top, heat_v), (0.95, 0.16, 0.02), glow_c)
    else:
        glow_c, k_crack, k_surf = (1.0, 0.2, 0.025), 6.0, 1.4
        g = nb.add(nb.mul(nb.mul(crack, nb.madd(hot, 0.8, 0.2)), k_crack),
                   nb.mul(nb.mul(nb.madd(under, 0.8, 0.2), hot), k_surf))
        g = nb.mul(g, nb.sub(1.0, nb.mul(ash, 0.85)))
        col = glow_c
    if gain_tab:
        g = nb.mul(g, _keyv(nb, 'coalgain', gain_tab))
    nb.output(surface=nb.addshader(bs, nb.emission(col, g)))
    return C.mesh_obj('coals_' + heat, V, F, mat=m)


def _sticks(C, new_material, specs, gain_tab):
    """Burning kindling: charred sticks with glowing splits (the beacon's fuel), [(p0, p1, radius)]."""
    from kit import fire as FK
    m, nb = new_material('stick')
    P = nb.texco().outputs['Object']
    grain = nb.noise(nb.vmath('MULTIPLY', P, (1.0, 1.0, 1.0)), scale=260.0, detail=3.0)
    rid = nb.noise(P, scale=90.0, detail=5.0, rough=0.62)
    crack = nb.mul(nb.sstep(0.035, 0.0, nb.math('ABSOLUTE', nb.sub(rid.outputs['Fac'], 0.5))),
                   nb.sstep(0.45, 0.62, nb.noise(P, scale=30.0, detail=2.0).outputs['Fac']))
    base = nb.mixcol(nb.sstep(0.4, 0.7, grain.outputs['Fac']), (0.012, 0.010, 0.009), (0.05, 0.04, 0.035))
    bs = nb.principled(Base_Color=base, Roughness=0.9)
    nb.link(nb.bump(nb.add(crack, grain.outputs['Fac']), 0.6, 0.001), bs.inputs['Normal'])
    g = nb.mul(nb.mul(crack, nb.madd(grain.outputs['Fac'], 1.2, 0.2)), nb.mul(_keyv(nb, 'stickgain', gain_tab), 5.0))
    nb.output(surface=nb.addshader(bs, nb.emission((1.0, 0.24, 0.04), g)))
    for i, (p0, p1, r) in enumerate(specs):
        pts = [tuple(a + (b - a) * k / 8.0 for a, b in zip(p0, p1)) for k in range(9)]
        v, fcs = FK.tube(pts, r, 10, radii=[r * (0.9 + 0.1 * math.sin(k * 1.7 + i)) for k in range(9)])
        C.mesh_obj(f'stick{i}', v, fcs, mat=m)


# ------------------------------------------------------------------------ C14 flint ---
# C14 as ONE hands-only sequence on H1's master timing (C frames): strike 1 = 2980 (flint_a 2960-2999); [find_a,
# find_b]; strike 3 = 3178, the ember takes 3181, the long blow 3206-3312, the catch 3316 (flint_b 3150-3359); the roar
# 3360 is C15's first frame. Set (world, metres, the lens looks along +Y from her right side): the tinder nest in her
# beacon's basket at N_NEST; her left fist holds the flint (and, after 3140, the Ring) over it; her right fist holds
# the C-steel, its broad face to the lens. Both forearms leave frame left. Lit by a low cold moon behind them.
FLINT = dict(s1=2980, s3=3178, ember=3181, catch=3316, kindle=3337,
             breaths=((2955, 2974, 'rest'), (2990, 3010, 'rest'), (3149, 3167, 'rest')),
             blows=((3206, 3227), (3238, 3260), (3270, 3292), (3298, 3312)))
N_NEST = (0.075, 0.005, 0.0)
Y_STEEL = -0.004                     # the steel's plane: just in front of the flint's front face


def _rust_iron(new_material, name='rust_iron'):
    """The beacon basket's aged iron (H5: rust, soot, uneven): near-black scale, rust blooms, a soot bloom."""
    m, nb = new_material(name)
    P = nb.texco().outputs['Object']
    n1 = nb.noise(P, scale=40.0, detail=5.0, rough=0.6)
    n2 = nb.noise(P, scale=160.0, detail=4.0, rough=0.6)
    rust = nb.sstep(0.50, 0.64, n1.outputs['Fac'])
    soot = nb.sstep(0.56, 0.40, n2.outputs['Fac'])
    col = nb.mixcol(rust, (0.020, 0.018, 0.016), (0.080, 0.032, 0.013))
    col = nb.colscale(col, nb.madd(soot, -0.5, 1.0))
    bs = nb.principled(Base_Color=col, Metallic=nb.madd(rust, -0.6, 0.75), Roughness=nb.madd(rust, 0.35, 0.5))
    nb.link(nb.bump(nb.add(n2.outputs['Fac'], nb.mul(rust, 0.8)), 0.4, 0.0006), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def _basket_c14(C, new_material, B=(0.09, 0.10), r_rim=0.215, z_rim=0.030, seed=3):
    """Her beacon's fire-basket round the nest: an out-of-round rim, uneven bars, far-side finials bent outward."""
    import random

    from kit import fire as FK
    rng = random.Random(seed)
    m = _rust_iron(new_material)
    pts = []
    for i in range(61):
        a = 2 * math.pi * i / 60
        rr = r_rim * (1 + 0.02 * math.sin(3 * a + 0.7) + 0.012 * math.sin(5 * a))
        pts.append((B[0] + rr * math.cos(a), B[1] + rr * math.sin(a), z_rim + 0.004 * math.sin(2 * a + 1.1)))
    v, f = FK.tube(pts, 0.0055, 8, cap=False)
    C.mesh_obj('b_rim', v, f, mat=m)
    for i in range(12):
        a = 2 * math.pi * (i + rng.uniform(-0.15, 0.15)) / 12
        ca, sa = math.cos(a), math.sin(a)
        top = z_rim + (rng.uniform(0.035, 0.065) if sa > 0.35 else 0.004)
        P_ = []
        for k in range(9):
            u = k / 8.0
            z = top - (top + 0.26) * u
            rr = r_rim * (1 - 0.35 * u * u) + 0.004 * math.sin(7 * u + i)
            P_.append((B[0] + rr * ca + rng.uniform(-0.002, 0.002), B[1] + rr * sa + rng.uniform(-0.002, 0.002), z))
        if sa > 0.35:
            x_, y_, z_ = P_[0]
            P_[0] = (x_ + 0.014 * ca, y_ + 0.014 * sa, z_)
        v, f = FK.tube(P_, 0.0042, 8)
        C.mesh_obj(f'b_bar{i}', v, f, mat=m)
    pts = []
    for i in range(61):
        a = 2 * math.pi * i / 60
        rr = r_rim * 0.9
        pts.append((B[0] + rr * math.cos(a), B[1] + rr * math.sin(a), -0.09 + 0.003 * math.sin(3 * a)))
    v, f = FK.tube(pts, 0.0045, 8, cap=False)
    C.mesh_obj('b_hoop', v, f, mat=m)


def _strands(pts_list, radii, attrs):
    """3-sided tubes along polylines, one mesh: (V, F, per-vertex attribute lists)."""
    from mathutils import Vector
    V, F = [], []
    A = {k: [] for k in attrs}
    for si, pts in enumerate(pts_list):
        pts = [Vector(p) for p in pts]
        n = len(pts)
        ref = Vector((0.31, 0.52, 0.79))
        base = len(V)
        for i in range(n):
            t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
            nrm = t.cross(ref)
            if nrm.length < 1e-6:
                nrm = t.cross(Vector((1.0, 0.0, 0.0)))
            nrm.normalize()
            bnm = t.cross(nrm)
            r = radii[si] * (1.0 - 0.55 * i / max(1, n - 1))
            for k in range(3):
                a = 2 * math.pi * k / 3
                p = pts[i] + (nrm * math.cos(a) + bnm * math.sin(a)) * r
                V.append((p.x, p.y, p.z))
                for key in attrs:
                    A[key].append(attrs[key][si])
        for i in range(n - 1):
            for k in range(3):
                k2 = (k + 1) % 3
                F.append((base + i * 3 + k, base + i * 3 + k2, base + (i + 1) * 3 + k2, base + (i + 1) * 3 + k))
    return V, F, A


def _nest(C, new_material, N, E, rf_tab, kg_tab, seed=17):
    """The tinder nest: fine dry grass and bark fibre worked round a hollow, irregular and fluffy (a wicker ring of
    even strands read as a basket), loose ends, curly fluff, coarse strips beneath. Its fibres catch from the ember
    outward late in the blow (rf: the burnt front's radius round E; kg: the glow): a crawling orange edge, black char
    behind it."""
    import random

    from mathutils import Vector, noise
    rng = random.Random(seed)
    P, R_, SR, DK = [], [], [], []
    off = Vector((seed * 0.7, 3.1, 1.7))

    def rim(th):                                           # an irregular bowl: radius and height wander round it
        return 0.046 * (1.0 + 0.10 * math.sin(2 * th + 0.8) + 0.06 * math.sin(3 * th + 2.0))

    def zb(r, th):
        rr = r / rim(th)
        return -0.006 + 0.021 * min(1.0, max(0.0, (rr - 0.30) / 0.72)) ** 1.25 + 0.003 * math.sin(3 * th + 1.0)

    def grow(p0, d0, L, n, kink, curl):
        pts, p, d = [], Vector(p0), Vector(d0).normalized()
        for i in range(n):
            pts.append(tuple(p))
            q = p * 90.0 + off + Vector((i * 0.37, 0.0, 0.0))
            turn = Vector((noise.noise(q), noise.noise(q + Vector((5.2, 1.3, 0.0))), 0.3 * noise.noise(q + Vector((0, 9.1, 2.0)))))
            d = (d + turn * curl).normalized()
            if rng.random() < kink:
                d = (d + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.4, 0.4)))).normalized()
            p = p + d * (L / (n - 1))
        return pts

    for k in range(760):                                    # the bowl: grass laid round (mostly), crossing, kinked
        th = rng.uniform(0, 2 * math.pi)
        rr = rim(th) * rng.uniform(0.30, 1.02)
        p0 = Vector((rr * math.cos(th), rr * math.sin(th), 0.0))
        p0.z = zb(rr, th) + rng.uniform(0.0, 0.007)
        tang = Vector((-math.sin(th), math.cos(th), 0.0)) * rng.choice((-1.0, 1.0))
        d0 = tang + Vector((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.25, 0.35)))
        pts = grow(p0, d0, rng.uniform(0.018, 0.060), 10, 0.10, 0.35)
        pts = [(x, y, max(z, zb(math.hypot(x, y), math.atan2(y, x)) - 0.002)) for x, y, z in pts]
        P.append(pts)
        R_.append(rng.uniform(0.00015, 0.00034))
        SR.append(rng.random())
        DK.append(0.0)
    for k in range(110):                                    # loose ends out of the rim
        th = rng.uniform(0, 2 * math.pi)
        rr = rim(th) * rng.uniform(0.8, 1.05)
        p0 = Vector((rr * math.cos(th), rr * math.sin(th), zb(rr, th) + 0.003))
        d0 = Vector((math.cos(th + rng.uniform(-1.0, 1.0)), math.sin(th + rng.uniform(-1.0, 1.0)), rng.uniform(0.0, 1.1)))
        P.append(grow(p0, d0, rng.uniform(0.012, 0.040), 8, 0.05, 0.25))
        R_.append(rng.uniform(0.00012, 0.00028))
        SR.append(rng.random())
        DK.append(0.0)
    for k in range(170):                                    # fluff: fine curly fibre, mostly in the hollow
        th = rng.uniform(0, 2 * math.pi)
        rr = rim(th) * rng.uniform(0.0, 0.75)
        p0 = Vector((rr * math.cos(th), rr * math.sin(th), zb(max(rr, 0.012), th) + rng.uniform(0.0, 0.005)))
        d0 = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.2, 0.6)))
        P.append(grow(p0, d0, rng.uniform(0.006, 0.020), 8, 0.2, 0.9))
        R_.append(rng.uniform(0.00007, 0.00012))
        SR.append(rng.random())
        DK.append(0.0)
    for k in range(50):                                     # coarse bark strips under the bowl
        th = rng.uniform(0, 2 * math.pi)
        rr = rim(th) * rng.uniform(0.2, 1.1)
        p0 = Vector((rr * math.cos(th), rr * math.sin(th), zb(rr, th) - 0.005 - rng.uniform(0.0, 0.004)))
        tang = Vector((-math.sin(th), math.cos(th), 0.0))
        P.append(grow(p0, tang + Vector((rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.0)), rng.uniform(0.03, 0.07),
                      8, 0.05, 0.15))
        R_.append(rng.uniform(0.0005, 0.0010))
        SR.append(rng.random())
        DK.append(1.0)
    V, F, A = _strands(P, R_, dict(srand=SR, dark=DK))
    m, nb = new_material('tinder')
    sr = nb.attr('srand').outputs['Fac']
    dk = nb.attr('dark').outputs['Fac']
    Pl = nb.texco().outputs['Object']
    col = nb.mixcol(sr, (0.24, 0.185, 0.095), (0.50, 0.40, 0.215))
    col = nb.mixcol(nb.sstep(0.82, 0.96, sr), col, (0.30, 0.29, 0.26))            # a few grey, weathered
    col = nb.mixcol(dk, col, (0.075, 0.055, 0.035))
    El = Vector(E) - Vector(N)
    d = nb.length(nb.vsub(Pl, tuple(El)))
    wn = nb.noise(Pl, scale=500.0, detail=3.0).outputs['Fac']
    d = nb.add(d, nb.mul(nb.sub(wn, 0.5), 0.004))
    rf = _keyv(nb, 'rf', rf_tab)
    kg = _keyv(nb, 'kg', kg_tab)
    ring = nb.exp(nb.mul(nb.pw(nb.div(nb.math('ABSOLUTE', nb.sub(d, rf)), 0.0022), 2.0), -1.0))
    ring = nb.mul(ring, nb.sstep(0.0005, 0.002, rf))
    char = nb.sstep(nb.sub(rf, 0.0015), nb.sub(rf, 0.005), d)
    col = nb.mixcol(char, col, (0.010, 0.009, 0.008))
    bs = nb.principled(Base_Color=col, Roughness=0.62)
    tl = nb.n('ShaderNodeBsdfTranslucent')
    nb.set(tl.inputs['Color'], col)
    sh = nb.mixshader(nb.madd(dk, -0.28, 0.30), bs, tl)
    glow = nb.mul(nb.mul(ring, kg), nb.madd(wn, 1.4, 0.3))
    glow = nb.add(glow, nb.mul(nb.mul(char, nb.sstep(0.35, 0.8, wn)), nb.mul(kg, 0.18)))   # embers in the char
    nb.output(surface=nb.addshader(sh, nb.emission((1.0, 0.30, 0.05), nb.mul(glow, 3.5))))
    ob = C.mesh_obj('nest', V, F, mat=m)
    for key in ('srand', 'dark'):
        at = ob.data.attributes.new(key, 'FLOAT', 'POINT')
        at.data.foreach_set('value', A[key])
    ob.location = N
    return ob


def _charcloth(C, new_material, N, E, er_tab, ek_tab):
    """A square of char cloth in the nest's hollow, where strike 3's spark lands: black, woven, crumpled, frayed;
    the ember spreads through it from E (er: radius, ek: glow): a red-orange spot with a bright crawling edge, grey
    ash behind it."""
    import random

    from mathutils import Vector, noise
    rng = random.Random(29)
    n = 16
    V, F = [], []
    ang = math.radians(22.0)
    ca, sa = math.cos(ang), math.sin(ang)
    for j in range(n):
        for i in range(n):
            u, v = i / (n - 1.0) - 0.5, j / (n - 1.0) - 0.5
            x, y = u * 0.024, v * 0.021
            edge = max(abs(u), abs(v))
            if edge > 0.46:
                x += rng.uniform(-0.0008, 0.0008)
                y += rng.uniform(-0.0008, 0.0008)
            z = 0.0014 * noise.noise(Vector((x * 180.0, y * 180.0, 0.3))) + 0.0022 * (u * u + v * v) * 4.0
            V.append((x * ca - y * sa, x * sa + y * ca, z))
    for j in range(n - 1):
        for i in range(n - 1):
            F.append((j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i))
    m, nb = new_material('charcloth')
    Pl = nb.texco().outputs['Object']
    x, y, z = nb.sep(Pl)
    weave = nb.add(nb.math('ABSOLUTE', nb.math('SINE', nb.mul(x, 3400.0))),
                   nb.math('ABSOLUTE', nb.math('SINE', nb.mul(y, 3400.0))))
    El = Vector(E) - (Vector(N) + Vector((0.0, 0.0, -0.004)))
    d = nb.length(nb.vsub(Pl, (El.x, El.y, 0.0)))
    wn = nb.noise(Pl, scale=700.0, detail=4.0).outputs['Fac']
    d = nb.add(d, nb.mul(nb.sub(wn, 0.5), 0.0018))
    er = _keyv(nb, 'er', er_tab)
    ek = _keyv(nb, 'ek', ek_tab)
    inside = nb.sstep(nb.add(er, 0.0010), nb.sub(er, 0.0004), d)
    ring = nb.exp(nb.mul(nb.pw(nb.div(nb.math('ABSOLUTE', nb.sub(d, er)), 0.0007), 2.0), -1.0))
    ring = nb.mul(ring, nb.sstep(0.0002, 0.0008, er))
    ash = nb.sstep(nb.sub(er, 0.0022), nb.sub(er, 0.0045), d)
    col = nb.mixcol(nb.mul(ash, 0.85), (0.013, 0.012, 0.011), (0.17, 0.165, 0.16))
    bs = nb.principled(Base_Color=col, Roughness=0.88)
    nb.link(nb.bump(nb.add(weave, nb.mul(wn, 0.6)), 0.3, 0.00025), bs.inputs['Normal'])
    ecol = nb.mixcol(ring, (1.0, 0.20, 0.025), (1.0, 0.52, 0.13))
    g = nb.add(nb.mul(nb.mul(inside, nb.madd(ash, -0.75, 1.0)), nb.madd(wn, 1.1, 0.35)), nb.mul(ring, 1.7))
    nb.output(surface=nb.addshader(bs, nb.emission(ecol, nb.mul(nb.mul(g, ek), 3.0))))
    ob = C.mesh_obj('charcloth', V, F, mat=m)
    ob.location = Vector(N) + Vector((0.0, 0.0, -0.004))
    return ob


def _flint_stone(C, new_material, seed=5):
    """A flint nodule (~3.8 x 2.4 x 3.2 cm): a lumpy nodule whose top and front are knapped away in flat conchoidal
    facets (glassy near-black flint, faint ripples); a thin pale chalky cortex survives only on the unbroken skin."""
    import random

    import bmesh
    from mathutils import Vector, noise

    from kit import fire as FK
    rng = random.Random(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
    planes = []
    for k in range(7):
        th = rng.uniform(-2.9, 0.4)
        el = rng.uniform(0.05, 1.1)
        n = Vector((math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))).normalized()
        planes.append((n, rng.uniform(0.42, 0.72)))
    cut = []
    for v in bm.verts:
        p = v.co.copy()
        p *= 1.0 + 0.20 * noise.noise(p * 1.6 + Vector((seed * 1.3, 0.0, 0.0))) + 0.06 * noise.noise(p * 4.3)
        c = 0.0
        for n, d in planes:
            h = p.dot(n) - d
            if h > 0.0:
                p -= n * h
                c = 1.0
        v.co = Vector((p.x * 0.019, p.y * 0.012, p.z * 0.016))
        cut.append(c)
    bm.verts.index_update()
    V, F = FK._bm_to_lists(bm)
    bm.free()
    m, nb = new_material('flint')
    Pl = nb.texco().outputs['Object']
    kn = nb.sstep(0.5, 0.95, nb.attr('knap').outputs['Fac'])
    n1 = nb.noise(Pl, scale=160.0, detail=4.0)
    col = nb.mixcol(kn, (0.40, 0.37, 0.32), (0.018, 0.017, 0.020))
    col = nb.colscale(col, nb.madd(n1.outputs['Fac'], 0.5, 0.75))
    bs = nb.principled(Base_Color=col, Roughness=nb.madd(kn, -0.62, 0.86))
    rip = nb.math('SINE', nb.mul(nb.length(nb.vsub(Pl, (0.010, -0.008, 0.012))), 2300.0))
    nb.link(nb.bump(nb.add(nb.mul(rip, nb.mul(kn, 0.5)), nb.mul(n1.outputs['Fac'], nb.madd(kn, -1.6, 2.0))),
                    0.25, 0.00018), bs.inputs['Normal'])
    nb.output(surface=bs)
    ob = C.mesh_obj('flint', V, F, mat=m, smooth=False)
    at = ob.data.attributes.new('knap', 'FLOAT', 'POINT')
    at.data.foreach_set('value', cut)
    return ob


def _vol_box(C, name, x0, x1, y0, y1, z0, z1):
    V = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    ob = C.mesh_obj(name, V, F, smooth=False)
    ob.visible_shadow = False
    return ob


def _breath(C, new_material, name, M, D, puffs, speed, dens, cone=0.30, length=0.36, col=(0.86, 0.88, 0.92)):
    """Her breath: condensed mist in the cold air, a volume jetting from her mouth M (off frame) along D. puffs:
    [(f_start, f_end)] exhales. Keyed per frame: the puff's front and tail (metres along the jet), its strength (it
    thins as it travels) and the flow offset (the billows ride the air)."""
    from mathutils import Vector
    Dn = Vector(D).normalized()
    W = 0.012 + cone * length
    ob = _vol_box(C, name, 0.0, length, -W, W, -W, W)
    ob.location = M
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = Dn.to_track_quat('X', 'Z')
    fr, tl, kb, fl = [], [], [], []
    for f in range(START - 2, END + 3):
        vals = (0.0, 0.0, 0.0, 0.0)
        for (fa, fb) in puffs:
            if fa - 1 <= f <= fb + 40:
                t = (f - fa) / FPS
                front = speed * t * (1.0 - 0.25 * min(1.0, t))
                tail = 0.0 if f <= fb else speed * 0.8 * ((f - fb) / FPS)
                k = _ease((f - fa) / 3.0) * (1.0 - _ease((f - fb) / 22.0))
                vals = (front, tail, k, front * 0.8)
        fr.append((f, vals[0]))
        tl.append((f, vals[1]))
        kb.append((f, vals[2]))
        fl.append((f, vals[3]))
    m, nb = new_material(name + '_m')
    Pl = nb.texco().outputs['Object']
    x, y, z = nb.sep(Pl)
    rc = nb.madd(x, cone, 0.010)
    rad2 = nb.div(nb.add(nb.mul(y, y), nb.mul(z, z)), nb.mul(rc, rc))
    shape = nb.exp(nb.mul(rad2, -2.2))
    front = _keyv(nb, 'front', fr)
    tail = _keyv(nb, 'tail', tl)
    inside = nb.mul(nb.sstep(front, nb.sub(front, 0.04), x), nb.sstep(tail, nb.add(tail, 0.05), x))
    pc = nb.comb(nb.sub(x, _keyv(nb, 'flow', fl)), y, z)
    n = nb.noise(pc, scale=55.0, detail=4.0, rough=0.6)
    bil = nb.sstep(0.36, 0.72, n.outputs['Fac'])
    d = nb.mul(nb.mul(nb.mul(shape, inside), nb.mul(bil, _keyv(nb, 'kb', kb))), dens)
    pv = nb.n('ShaderNodeVolumePrincipled')
    nb.set(pv.inputs['Density'], d)
    pv.inputs['Color'].default_value = tuple(col) + (1.0,)
    pv.inputs['Anisotropy'].default_value = 0.55
    nb.output(volume=pv)
    ob.data.materials.append(m)
    return ob


def _smoke_c14(C, new_material, E, sk_tab, wind_tab, dens=90.0):
    """Smoke off the smouldering tinder: a thin pale wisp from the ember, thickening with each breath and pushed to
    the right by it (wind), rising and fraying."""
    from kit import fire as FK
    ob = _vol_box(C, 'smoke', -0.05, 0.13, -0.05, 0.05, 0.0, 0.19)
    ob.location = E
    m, nb = new_material('smoke_m')
    Pl = nb.texco().outputs['Object']
    x, y, z = nb.sep(Pl)
    t = FK.time_value(nb)
    wind = _keyv(nb, 'wind', wind_tab)
    xs = nb.add(nb.mul(nb.mul(wind, z), 0.9), nb.mul(nb.math('SINE', nb.sub(nb.mul(z, 60.0), nb.mul(t, 2.0))), 0.004))
    rp = nb.madd(z, 0.10, 0.0022)
    dx = nb.sub(x, xs)
    shape = nb.exp(nb.mul(nb.div(nb.add(nb.mul(dx, dx), nb.mul(y, y)), nb.mul(rp, rp)), -1.5))
    pc = nb.comb(x, y, nb.sub(z, nb.mul(t, 0.05)))
    n = nb.noise(pc, scale=110.0, detail=3.0, rough=0.55, dims='4D', w=nb.mul(t, 0.3))
    wisp = nb.sstep(0.40, 0.75, n.outputs['Fac'])
    d = nb.mul(nb.mul(shape, wisp), nb.mul(_keyv(nb, 'sk', sk_tab), nb.mul(nb.sstep(0.0, 0.003, z), nb.sstep(0.19, 0.11, z))))
    pv = nb.n('ShaderNodeVolumePrincipled')
    nb.set(pv.inputs['Density'], nb.mul(d, dens))
    pv.inputs['Color'].default_value = (0.64, 0.62, 0.60, 1.0)
    pv.inputs['Anisotropy'].default_value = 0.3
    nb.output(volume=pv)
    ob.data.materials.append(m)
    return ob


def _ash_material(new_material):
    """Old ash on the basket's floor: soft grey, darker in drifts, flecked with char."""
    m, nb = new_material('ash')
    P = nb.texco().outputs['Object']
    n1 = nb.noise(P, scale=30.0, detail=5.0, rough=0.6)
    n2 = nb.noise(P, scale=300.0, detail=3.0)
    col = nb.mixcol(nb.sstep(0.35, 0.7, n1.outputs['Fac']), (0.030, 0.029, 0.028), (0.16, 0.155, 0.15))
    col = nb.colscale(col, nb.madd(nb.sstep(0.6, 0.75, n2.outputs['Fac']), -0.8, 1.0))
    bs = nb.principled(Base_Color=col, Roughness=0.95)
    nb.link(nb.bump(nb.add(n1.outputs['Fac'], n2.outputs['Fac']), 0.5, 0.002), bs.inputs['Normal'])
    nb.output(surface=bs)
    return m


def _strike_grip(f, fs, Rg, Cg, Wg, Sg):
    """The right fist's strike round frame fs: rest -> wind-up (fs-12..fs-4) -> down fast onto the flint's edge (fs-3
    ..fs) -> a short scrape (fs..fs+1) -> the rebound (fs+1..fs+9). Returns (grip, dip in degrees; + = tip down)."""
    from mathutils import Vector
    Rg, Cg, Wg, Sg = Vector(Rg), Vector(Cg), Vector(Wg), Vector(Sg)
    if f <= fs - 12 or f >= fs + 9:
        return Rg.copy(), 0.0
    if f < fs - 4:
        u = _ease((f - (fs - 12)) / 8.0)
        return Rg.lerp(Wg, u), -9.0 * u
    if f < fs - 3:
        return Wg.copy(), -9.0
    if f < fs:
        u = _ein((f - (fs - 3)) / 3.0)
        return Wg.lerp(Cg, u), -9.0 + 13.0 * u
    if f < fs + 1:
        u = (f - fs) / 1.0
        return Cg.lerp(Sg, u), 4.0 + 5.0 * u
    u = _eout((f - (fs + 1)) / 8.0)
    return Sg.lerp(Rg, u), 9.0 * (1.0 - u)


def _flint(C, new_material, R, opts, T):
    """C14 hands-only (flint_a 2960-2999: strike 1 in the dark; flint_b 3150-3359: strike 3, the ember, the long
    blow, the catch). Only her gloved hands, the flint, the C-steel, the tinder and her breath are in frame."""
    import random

    import bpy
    from mathutils import Matrix, Vector

    import glove as GL
    F_ = FLINT
    N = Vector(N_NEST)
    E = N + Vector((0.004, -0.002, -0.0025))                 # the ember's spot on the char cloth
    _world(opts.get('sky', (0.10, 0.16, 0.36)), opts.get('sky_w', 0.035))
    sc = bpy.context.scene
    if sc.render.engine == 'CYCLES':
        sc.cycles.volume_step_rate = opts.get('vol_step', 0.25)
        sc.cycles.volume_max_steps = 512
    _plane(C, 'snow_far', -0.45, 6.0, snow_material(new_material))           # the moonlit ground, far behind
    # the basket's floor under the kindling: old ash and cold cinders (no bars in frame: they read as a cage)
    ash = _plane(C, 'ash', -0.046, 0.40, _ash_material(new_material))
    ash.location = (N.x, N.y, 0.0)
    _coals(C, new_material, 70, (0.17, 0.13), 23, 'fire', z0=-0.046 - 0.004,
           gain_tab=[(f, 0.0) for f in range(START - 2, END + 3)])
    fr = range(START - 2, END + 3)
    later = []
    # ---- the ember (flint_b): born when strike 3's spark lands (3181); each breath brightens it and spreads it
    blows = F_['blows']

    def bpulse(f):
        v = 0.0
        for fa, fb in blows:
            v = max(v, _ease((f - fa - 3) / 5.0) * (1.0 - _ease((f - fb) / 6.0)))
        return v

    def ek(f):
        if f < F_['ember']:
            return 0.0
        b = 0.55 + 0.45 * _ease((f - F_['ember']) / 120.0)
        b *= 1.0 + 1.9 * bpulse(f)
        b *= 1.0 + 0.9 * math.exp(-(f - F_['ember']) / 3.0)          # the spark's own heat as it lands
        return b * _flick(f, 21, 0.35)

    def er(f):
        if f < F_['ember']:
            return 0.0
        u = min(1.0, (f - F_['ember']) / float(F_['catch'] - F_['ember']))
        return 0.0010 + 0.0100 * u ** 0.85

    def rf(f):                                                       # the fibres catch late in the blow
        return 0.018 * _ease((f - 3266) / 50.0) ** 0.9 if f < F_['catch'] + 30 else 0.018 + 0.0004 * (f - 3346)

    def kg(f):
        return (0.3 + 1.2 * bpulse(f)) * _ease((f - 3266) / 12.0) * (1.0 + 1.5 * _ease((f - F_['catch']) / 6.0))

    has_ember = END >= F_['ember']
    if has_ember:
        _charcloth(C, new_material, N, E, [(f, er(f)) for f in fr], [(f, ek(f)) for f in fr])
        el = C.point('ember', tuple(E + Vector((0.0, 0.0, 0.006))), (1.0, 0.32, 0.07), 0.0, radius=0.003)
        for f in fr:
            C.key(el.data, 'energy', f, opts.get('ember_w', 0.004) * ek(f) * (0.3 + er(f) / 0.004))
    _nest(C, new_material, N, E, [(f, rf(f) if has_ember else 0.0) for f in fr],
          [(f, kg(f) if has_ember else 0.0) for f in fr])
    stick_g = [(f, 0.0 if f < F_['kindle'] else 0.6 * _ease((f - F_['kindle']) / 18.0) * _flick(f, 9, 0.6)) for f in fr]
    _sticks(C, new_material, [((0.00, -0.04, -0.024), (0.17, 0.05, -0.016), 0.006),
                              ((0.02, 0.06, -0.026), (0.15, -0.035, -0.020), 0.0055),
                              ((-0.01, 0.01, -0.032), (0.16, 0.00, -0.028), 0.0075),
                              ((0.05, -0.05, -0.030), (0.10, 0.07, -0.022), 0.0045)], stick_g)

    # ---- the hands. The steel's plane is y = Y_STEEL; the flint rides her left fist.
    leather = GL.leather_material(new_material, base=opts.get('leather', (0.058, 0.036, 0.021)),
                                  rough=opts.get('leather_rough', 0.44))
    wool = GL.wool_material(new_material)
    heat = [(f, 0.0) for f in fr]
    stl = _steel_mesh(C, steel_material(new_material, heat))
    stl.rotation_mode = 'QUATERNION'
    GR = GL.Glove('rglove', leather, wool, mirror=False)
    GL_ = GL.Glove('lglove', leather, wool, mirror=True)
    sc.frame_set(START)
    GR.set_pose(GL.POSES['fist'])
    pb = GR.rig.pose.bones
    hol = (pb['m1'].head + pb['m2'].head + pb['m3'].head + pb['m3'].tail) / 4.0
    lpose = GL.pose_mix(GL.POSES['fist'], GL.POSES['cup'], opts.get('lfist_open', 0.10))
    GL_.set_pose(lpose)
    pbl = GL_.rig.pose.bones
    holl = (pbl['m1'].head + pbl['m2'].head + pbl['m3'].head + pbl['m3'].tail) / 4.0
    holl = Vector((holl.x, -holl.y, holl.z))
    Rh = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))).to_4x4()
    xl = Vector(opts.get('lfist_x', (0.87, -0.45, -0.18))).normalized()
    yl = -(Vector((0.0, 0.0, 1.0)) - xl * xl.z).normalized()      # the mirrored thumb (local -y) points up
    zl = xl.cross(yl)
    Rl = Matrix((xl, yl, zl)).transposed().to_4x4()
    LF0 = Vector(opts.get('lf', (0.030, 0.024, 0.056)))
    LF2 = Vector((0.020, 0.040, 0.030))
    LF3 = Vector((0.004, 0.050, 0.046))
    fl_off = Vector(opts.get('flint_off', (-0.006, -0.013, 0.026)))
    flint = _flint_stone(C, new_material)
    flint.rotation_mode = 'XYZ'
    flint.rotation_euler = (math.radians(8.0), math.radians(-14.0), math.radians(6.0))

    def LF(f):
        p = LF0.lerp(LF2, _ease((f - 3186) / 20.0)) if f > 3180 else LF0.copy()
        p = p.lerp(LF3, _ease((f - (F_['catch'] + 2)) / 18.0))
        for fs in (F_['s1'], F_['s3']):                          # the blow jolts the flint hand
            if f >= fs:
                p += Vector((0.0015, 0.0, -0.0025)) * math.exp(-(f - fs) / 2.0)
        t = f / FPS
        p += Vector((0.0004 * math.sin(1.1 * t + 0.4), 0.0003 * math.sin(0.8 * t), 0.0005 * math.sin(1.6 * t + 1.0)))
        return p

    # the contact: the steel's long arm (underside, at s_c) meets the flint's front top edge
    Cpt = LF0 + fl_off + Vector((0.0, 0.0, 0.0)) + Vector(opts.get('contact_off', (-0.001, 0.012, 0.015)))
    Cpt.y = Y_STEEL + 0.003
    s_c = 0.058
    rot4 = Matrix.Rotation(math.radians(4.0), 3, 'Y')
    Cg = Cpt - rot4 @ Vector((s_c, 0.0, _steel_arm_bot(s_c)))
    Cg.y = Y_STEEL
    Rg = Cg + Vector((-0.008, 0.0, 0.017))
    Wg = Cg + Vector((-0.013, 0.0, 0.062))
    Sg = Cg + Vector((0.006, 0.0, -0.010))
    LOW = Vector((-0.080, Y_STEEL - 0.006, 0.026))
    OUT = Vector((-0.105, Y_STEEL - 0.012, 0.044))

    def steel(f):
        g_, dp = _strike_grip(f, F_['s1'], Rg, Cg, Wg, Sg)
        if F_['s3'] - 14 <= f:
            g_, dp = _strike_grip(f, F_['s3'], Rg, Cg, Wg, Sg)
            if f > F_['s3'] + 9:                                  # after strike 3 she lays her fist down by the nest
                u = _ease((f - (F_['s3'] + 9)) / 16.0)
                g_ = g_.lerp(LOW, u)
                dp = -4.0 * u
                g_ = g_.lerp(OUT, _ease((f - (F_['catch'] + 2)) / 18.0))
        t = f / FPS
        g_ = g_ + Vector((0.0004 * math.sin(1.3 * t), 0.0, 0.0005 * math.sin(1.9 * t + 0.5)))
        return Matrix.Translation(g_) @ Matrix.Rotation(math.radians(dp), 4, 'Y')

    for f in fr:
        Sf = steel(f)
        l, r, _ = Sf.decompose()
        stl.location = l
        stl.rotation_quaternion = r
        stl.keyframe_insert('location', frame=f)
        stl.keyframe_insert('rotation_quaternion', frame=f)
        GR.key_place(f, Sf @ Rh @ Matrix.Translation(-hol))
        GR.key(f, GL.POSES['fist'])
        Ml = Matrix.Translation(LF(f)) @ Rl @ Matrix.Translation(-holl)
        GL_.key_place(f, Ml)
        GL_.key(f, lpose)
        flint.location = LF(f) + fl_off
        flint.keyframe_insert('location', frame=f)
    print('flint: contact', tuple(round(c, 4) for c in Cpt), 'Cg', tuple(round(c, 4) for c in Cg), flush=True)

    # ---- sparks: short orange beads flung down off the edge, falling and curving; most die in the air or in the
    # tinder. Strike 3's one lands on the char cloth and lives (the ember)
    rng = random.Random(opts.get('spark_seed', 11))
    paths = []
    for fs in (F_['s1'], F_['s3']):
        if not (START - 2 <= fs <= END):
            continue
        g_ = Vector((0.0, 0.0, -2.6))                             # gravity, eased by the air: short falling arcs
        for k in range(opts.get('n_sparks', 22)):
            v = rng.uniform(0.25, 0.85)
            dv = Vector((rng.uniform(0.15, 1.0), rng.uniform(-0.45, 0.30), rng.uniform(-0.85, 0.35))).normalized()
            V0 = dv * v
            T_ = rng.uniform(4.0, 10.0) / FPS
            S0 = Cpt + Vector((rng.uniform(-0.002, 0.004), rng.uniform(-0.002, 0.001), rng.uniform(-0.002, 0.001)))
            land = 0
            for i in range(1, 25):                                # does it come down in the tinder first?
                t = T_ * i / 24.0
                p = S0 + V0 * t + 0.5 * g_ * t * t
                if p.z < 0.010 and (Vector((p.x, p.y, 0.0)) - Vector((N.x, N.y, 0.0))).length < 0.046:
                    T_, land = t, int(rng.uniform(2, 6))
                    break
            Q_ = S0 + V0 * (0.5 * T_)
            L_ = S0 + V0 * T_ + 0.5 * g_ * T_ * T_
            f0 = fs + (1 if k % 3 else 0)
            paths.append((tuple(S0), tuple(Q_), tuple(L_), f0, f0 + max(2, int(round(T_ * FPS))), land))
        if fs == F_['s3']:                                        # the one that takes, on the char cloth
            S0 = Cpt + Vector((0.001, -0.001, -0.001))
            L_ = E + Vector((0.0, 0.0, 0.0006))
            T_ = 4.0 / FPS
            V0 = (L_ - S0 - 0.5 * g_ * T_ * T_) / T_
            paths.append((tuple(S0), tuple(S0 + V0 * (0.5 * T_)), tuple(L_), fs, fs + 4, 18))
    if paths:
        _sparks(C, new_material, paths, (START, END))
    # the strike's flash: a brief warm burst at the edge, on the gloves and the tinder
    fls = C.point('flash', tuple(Cpt + Vector((-0.004, -0.018, 0.004))), (1.0, 0.52, 0.20), 0.0, radius=0.004)
    for f in fr:
        e = 0.0
        for fs in (F_['s1'], F_['s3']):
            if f >= fs:
                e += math.exp(-(f - fs) / 1.3)
        C.key(fls.data, 'energy', f, opts.get('flash_w', 0.10) * e)

    # ---- breath: two slow exhales in strike 1; in flint_b the tension (3149) and then the long blow at the ember
    mouth_a = Vector(opts.get('mouth_a', (-0.15, 0.10, 0.22)))
    _breath(C, new_material, 'breath_rest', mouth_a, Vector((0.10, -0.01, 0.09)) - mouth_a,
            [(fa, fb) for fa, fb, _k in F_['breaths']], speed=0.34, dens=opts.get('breath_dens', 30.0), cone=0.34)
    if has_ember:
        mouth_b = Vector(opts.get('mouth_b', (-0.11, 0.05, 0.17)))
        _breath(C, new_material, 'breath_blow', mouth_b, (E + Vector((0.0, 0.0, 0.004))) - mouth_b, list(blows),
                speed=0.85, dens=opts.get('blow_dens', 16.0), cone=0.16, length=0.30)
        sk = [(f, 0.0 if f < F_['ember'] + 4 else (0.25 + 0.75 * _ease((f - 3190) / 110.0)) *
               (1.0 + 0.6 * bpulse(f)) * (1.0 - 0.5 * _ease((f - F_['catch']) / 10.0))) for f in fr]
        wind = [(f, 0.12 + 0.9 * bpulse(f)) for f in fr]
        _smoke_c14(C, new_material, E, sk, wind, dens=opts.get('smoke_dens', 90.0))

    # ---- the catch: the smoke takes, a small flame blooms from the nest (sprites) and the kindling follows
    if 'tflame' in T.get('sprites', {}):
        fl_tab = [(f, 0.0 if f < F_['catch'] - 1 else _flick(f, 17, 0.4)) for f in fr]
        lights = [C.point('flame1', tuple(N + Vector((0.0, 0.0, 0.016))), (1.0, 0.50, 0.17), 0.0, radius=0.010),
                  C.point('flame2', tuple(N + Vector((0.008, 0.01, 0.055))), (1.0, 0.55, 0.20), 0.0, radius=0.02)]
        for f in fr:
            u = (f - F_['catch']) / 20.0
            grow = 0.0 if u < 0 else (0.25 + 0.75 * (1.0 - math.exp(-2.0 * u)))
            kgrow = 0.0 if f < F_['kindle'] else 0.8 * _ease((f - F_['kindle']) / 20.0)
            C.key(lights[0].data, 'energy', f, opts.get('flame_w', 0.10) * (grow + kgrow) * _flick(f, 3, 0.7))
            C.key(lights[1].data, 'energy', f, opts.get('flame_w', 0.10) * 0.8 * (grow * 0.5 + kgrow) * _flick(f, 8, 0.7))
        later.append((N + Vector((0.0, 0.0, -0.004)), 'tflame', opts.get('tflame_g', 0.9), fl_tab))
        later.append((N + Vector((0.012, 0.022, -0.014)), 'kflame', opts.get('kflame_g', 0.8), fl_tab))

    # ---- the moon behind them, low: cold rims on the fists, the steel, the flint, the straw and the breath; a faint
    # cold fill from the lens side keeps the leather from going black
    C.sun('moon', opts.get('moon_dir', (0.30, 0.85, 0.42)), (0.55, 0.66, 1.0), opts.get('moon', 0.40), angle_deg=1.5,
          volume=1.0)
    C.sun('nightfill', opts.get('fill_dir', (-0.35, -0.85, 0.40)), (0.50, 0.60, 1.0), opts.get('fill', 0.05),
          angle_deg=20.0)
    # ---- the camera: the strike framing (both fists, the steel's whole C, the nest below); in flint_b a slow push
    # to the nest through the blow, easing back and up as the flame rises
    tA = Vector(opts.get('tgtA', (0.028, 0.012, 0.048)))
    pA = tA + Vector(opts.get('camA_off', (-0.045, -0.465, 0.19)))
    tB = Vector(opts.get('tgtB', tuple(E + Vector((-0.006, 0.0, 0.004)))))
    pB = tB + Vector(opts.get('camB_off', (-0.025, -0.205, 0.175)))
    tC = Vector(opts.get('tgtC', tuple(N + Vector((-0.004, 0.0, 0.030)))))
    pC = tC + Vector(opts.get('camC_off', (-0.03, -0.30, 0.15)))
    cam = _camera(pA, tA, opts.get('lens', 50.0), opts.get('fstop', 8.0))
    for f in fr:
        u = (f - START) / float(max(1, END - START))
        p_, t_ = pA.lerp(tA, 0.015 * u), tA.copy()
        foc = Cpt.copy()
        if START >= 3100:
            k = _ease((f - 3186) / 114.0)
            p_, t_ = pA.lerp(pB, k), tA.lerp(tB, k)
            foc = Cpt.lerp(E, _ease((f - 3180) / 30.0))
            k2 = _ease((f - (F_['catch'] + 2)) / 34.0)
            p_, t_ = p_.lerp(pC, k2), t_.lerp(tC, k2)
            foc = foc.lerp(N + Vector((0.0, 0.0, 0.02)), k2)
        cam.location = p_
        cam.keyframe_insert('location', frame=f)
        cam.rotation_quaternion = (t_ - p_).to_track_quat('-Z', 'Y')
        cam.keyframe_insert('rotation_quaternion', frame=f)
        cam.data.dof.focus_distance = (p_ - foc).length
        cam.data.dof.keyframe_insert('focus_distance', frame=f)
        if START >= 3100:                                             # close over the nest: keep the tinder sharp
            cam.data.dof.aperture_fstop = opts.get('fstop', 8.0) + 5.0 * _ease((f - 3186) / 114.0)
            cam.data.dof.keyframe_insert('aperture_fstop', frame=f)
    from kit import fire as FK
    for pos, name, gain, tab in later:
        sp_ = T['sprites'][name]
        fc = FK.flame_card(f'{name}_c', pos, sp_['card'], sp_['first'], cam, gain=gain, fog=False)
        _card_no_light(fc)
        for nd in fc.data.materials[0].node_tree.nodes:
            if nd.type == 'EMISSION':
                for f, v in tab:
                    C.key_socket(nd.inputs['Strength'], f, gain * v)



def _key_hide(obs, f, hidden):
    for ob in obs:
        if ob is None:
            continue
        ob.hide_render = hidden
        ob.keyframe_insert('hide_render', frame=f)


def _smoke_gate(fc, seed, amt=0.75, scale=11.0, rise=0.05):
    """Dark smoke drifting up across a background flame card: its emission times a slow rising noise mask, so the
    far flames come and go behind gaps instead of standing as one bright wall."""
    from kit import fire as FK
    from kit.nodes import NB
    m = fc.data.materials[0]
    nb = NB(m.node_tree)
    em = next(nd for nd in m.node_tree.nodes if nd.type == 'EMISSION')
    src = em.inputs['Color'].links[0].from_socket
    t = FK.time_value(nb)
    x, y, z = nb.sep(nb.texco().outputs['Object'])
    pc = nb.comb(x, y, nb.sub(z, nb.mul(t, rise)))
    n = nb.noise(pc, scale=scale, detail=3.0, rough=0.55, dims='4D', w=nb.madd(t, 0.25, seed))
    mask = nb.madd(nb.sstep(0.36, 0.64, n.outputs['Fac']), amt, 1.0 - amt)
    nb.link(nb.colscale(src, mask), em.inputs['Color'])


def _fire(C, new_material, R, opts, T):
    """THE FIRE TEST (C15, Bag End), 3360-3599, at her beacon. The roar (3360): the basket's fire surges round the
    lens, no hand in frame. 3384-3404 her gloved right fist brings her C-shaped fire-steel in from the left, the band
    hanging on its long upper arm, into the flames: it hangs there, unmarked, its letters awake (3400+), while the
    arm's end goes dull red. 3480 the steel tips: the band slides to the very end, the lip, and does not fall: her
    wrist catches it. 3560 she draws it out of the fire to the left; the lens pulls back with it and both her hands
    are in the focal plane, small and sharp against the fire: her left palm comes in under the tip, the steel tips,
    the band drops into the palm (3580) and her fingers close on it (3584-3595)."""
    import bpy
    from mathutils import Matrix, Quaternion, Vector

    from kit import fire as FK
    import glove as GL
    w = _world((0.020, 0.0065, 0.0022), opts.get('world_w', 0.6))
    roar_t, s_in, tip_t, draw_t = BEAT['roar'], BEAT['steel_in'], BEAT['tip'], BEAT['draw']

    def roar(f):                                   # the fire's size/brightness envelope: surge on the roar, settle
        u = f - roar_t
        k = 0.55 + 0.45 * _ease((u + 4) / 6.0)
        k += 0.55 * math.exp(-((u - 5.0) / 5.0) ** 2) + 0.15 * math.exp(-((u - 16.0) / 6.0) ** 2)
        return k * _flick(f, 3, 0.8)

    gt = [(f, roar(f)) for f in range(START - 2, END + 3)]
    # the bed: embers and burning kindling in the basket, below the band
    _coals(C, new_material, 90, (0.12, 0.07), 11, 'fire', z0=-0.092, gain_tab=gt)
    _sticks(C, new_material, [((-0.13, 0.02, -0.074), (0.10, 0.05, -0.062), 0.009),
                              ((-0.05, -0.03, -0.080), (0.14, 0.03, -0.070), 0.007),
                              ((0.02, 0.09, -0.078), (0.09, -0.02, -0.066), 0.006),
                              ((-0.10, 0.07, -0.070), (0.01, 0.10, -0.050), 0.008)], gt)
    # the basket: a few aged iron bars behind the flames and its rim (context: her beacon)
    im = FK.iron_material('basket_iron')
    for i, x in enumerate((-0.085, 0.045, 0.165)):
        v, fcs = FK.tube([(x, 0.085, -0.16), (x + 0.004, 0.086, -0.02), (x - 0.002, 0.084, 0.075)], 0.0045, 10)
        C.mesh_obj(f'bar{i}', v, fcs, mat=im)
    v, fcs = FK.tube([(-0.25, 0.09, 0.066), (0.0, 0.080, 0.070), (0.25, 0.092, 0.064)], 0.005, 10)
    C.mesh_obj('rim', v, fcs, mat=im)

    # ---- choreography (world): grip(f) = the steel's origin (the middle of its back, in her fist); dip(f) = the
    # steel's tip-down angle. The band hangs at the world origin.
    s_rest = opts.get('s_rest', S_REST)
    s_lip = S_LIP
    G0 = Vector((-s_rest, 0.0, -(_steel_arm_top(s_rest) - R_IN)))
    Gd = G0 + Vector(opts.get('draw_off', (-0.145, 0.0, 0.022)))       # drawn out of the flames: left and up
    Gx = Gd + Vector((-0.012, 0.0, -0.004))                             # and settles, the band in her other hand
    enter0 = s_in - 20

    def grip(f):
        if f < s_in:
            u = _eout((f - enter0) / 20.0)
            g_ = G0 + Vector((-0.26 * (1.0 - u), 0.0, -0.03 * (1.0 - u)))
        elif f < draw_t:
            g_ = G0.copy()
        else:
            g_ = G0.lerp(Gd, _ease((f - draw_t) / 16.0)).lerp(Gx, _ease((f - 3588) / 10.0))
        t = f / FPS                                                   # a gloved hand held at a fire trembles
        g_ += Vector((0.0004 * math.sin(7.1 * t) + 0.0002 * math.sin(13.3 * t + 1.0), 0.0,
                      0.0005 * math.sin(5.3 * t + 0.4) + 0.00025 * math.sin(17.9 * t)))
        return g_

    def dip(f):
        d = 0.0
        d += math.radians(12.0) * _ease((f - tip_t) / 6.0)            # she tips it...
        d -= math.radians(10.5) * _eout((f - (tip_t + 9)) / 5.0)       # ...and her wrist catches it
        d += math.radians(30.0) * _ease((f - 3573) / 5.0)             # out of the fire: tipped into her palm
        d -= math.radians(28.0) * _ease((f - 3582) / 8.0)
        t = f / FPS
        return d + math.radians(0.35) * math.sin(6.1 * t + 0.3) + math.radians(0.2) * math.sin(11.7 * t)

    def S(f):
        return Matrix.Translation(grip(f)) @ Matrix.Rotation(dip(f), 4, 'Y')

    # ---- the steel and her right fist
    heat = [(f, 0.5 * _ease((f - 3408) / 120.0) * (1.0 - 0.35 * _ease((f - draw_t) / 40.0))) for f in
            range(START - 2, END + 3)]
    stl = _steel_mesh(C, steel_material(new_material, heat))
    stl.rotation_mode = 'QUATERNION'
    leather = GL.leather_material(new_material, base=opts.get('leather', (0.034, 0.022, 0.014)),
                                  rough=opts.get('leather_rough', 0.38))
    for nd in leather.node_tree.nodes:                 # in the fire: dark leather whose waxy sheen draws the forms (a lit
        if nd.type == 'BSDF_PRINCIPLED':               # tan diffuse read as clay)
            nd.inputs['Specular IOR Level'].default_value = opts.get('leather_spec', 0.50)
            nd.inputs['Coat Weight'].default_value = opts.get('leather_coat', 0.25)
            nd.inputs['Coat Roughness'].default_value = 0.26
    wool = GL.wool_material(new_material)
    GR = GL.Glove('rglove', leather, wool, mirror=False)
    GR.set_pose(GL.POSES['fist'])
    sc = bpy.context.scene
    sc.frame_set(START)
    pb = GR.rig.pose.bones
    hol = (pb['m1'].head + pb['m2'].head + pb['m3'].head + pb['m3'].tail) / 4.0 + Vector(opts.get('hol_off', (0.0, 0.0, 0.0)))
    # the fist's grip runs along the hand's y (the knuckle line): hand x -> world +X (toward the fire), y -> +Z (thumb
    # up), z -> -Y (the back of the hand to the lens)
    Rh = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))).to_4x4()
    Rh = Rh @ Matrix.Rotation(math.radians(opts.get('fist_yaw', 0.0)), 4, 'Z')
    for f in range(START - 2, END + 3):
        Sf = S(f)
        l, r, _ = Sf.decompose()
        stl.location = l
        stl.rotation_quaternion = r
        stl.keyframe_insert('location', frame=f)
        stl.keyframe_insert('rotation_quaternion', frame=f)
        GR.key_place(f, Sf @ Rh @ Matrix.Translation(-hol))
        GR.key(f, GL.POSES['fist'])
        _key_hide((GR.mesh, GR.sleeve, stl), f, f < enter0 - 2)

    # ---- the band: hangs from the arm's top edge by its inner face; slides to the lip on the tip; swings
    g = gold(new_material, R, letters=_tab(lambda f: opts.get('letters_w', 2.4) * _ease((f - (s_in + 1)) / 18.0) *
                                           (1.0 - _ease((f - (draw_t + 2)) / 12.0))))
    ring, _ = band_mesh('ring', g)
    ring.rotation_mode = 'QUATERNION'

    def s_contact(f):
        u = (f - (tip_t + 3)) / 6.0
        return s_rest + (s_lip - s_rest) * (_ein(u) if u < 1.0 else 1.0)

    def swing(f):
        a = 0.0
        ua = (f - (s_in + 1)) / FPS                                   # it arrives swinging
        if ua > 0:
            a += math.radians(9.0) * math.exp(-ua * 2.4) * math.sin(ua * 10.5)
        ub = (f - (tip_t + 9)) / FPS                                  # it stops against the lip
        if ub > 0:
            a += math.radians(7.0) * math.exp(-ub * 2.8) * math.sin(ub * 11.0 + 0.3)
        uc = (f - (draw_t + 2)) / FPS                                 # drawn out, it swings again
        if uc > 0:
            a += math.radians(6.0) * math.exp(-uc * 3.0) * math.sin(uc * 12.0 + 1.0)
        return a

    phi = math.radians(opts.get('ring_phi', 52.0))
    drop0, drop1 = 3577, 3581                                         # off the lip, into her palm

    for f in range(START - 2, END + 3):
        sc_ = s_contact(f)
        Pc = S(f) @ Vector((sc_, 0.0, _steel_arm_top(sc_)))
        ps = swing(f)
        arm_dir = (S(f).to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
        hdir = Vector((arm_dir.x, arm_dir.y, 0.0)).normalized()
        axis = Quaternion((0.0, 0.0, 1.0), phi) @ hdir                   # the band's axis, horizontal
        upv = Vector((0.0, 0.0, 1.0))
        M3 = Matrix((upv, axis.cross(upv), axis)).transposed()         # local x = up, local z = the axis
        ring.location = Pc + Quaternion(hdir, ps) @ Vector((0.0, 0.0, -(R_IN + 0.0008)))   # clear of the arm's edges
        ring.rotation_quaternion = Quaternion(hdir, ps) @ M3.to_quaternion()
        ring.keyframe_insert('location', frame=f)
        ring.keyframe_insert('rotation_quaternion', frame=f)
    catch = Gd
    print('fire: G0', tuple(round(c, 4) for c in G0), 'Gd', tuple(round(c, 4) for c in Gd), 'catch',
          tuple(round(c, 4) for c in catch), flush=True)

    # ---- the fire's own light (keyed with the roar and the flicker): the bed from below, the tongues round the band,
    # a warm rim from behind, and the fire's key on whatever it holds out to the left (her hands at the catch)
    bed = bpy.data.lights.new('bed', 'AREA')
    bed.shape = 'DISK'
    bed.size = 0.18
    bed.color = (1.0, 0.36, 0.08)
    ob = C.link_obj(bpy.data.objects.new('bed', bed))
    ob.location = (0.01, 0.02, -0.055)
    L1 = C.point('tongue1', (0.012, 0.012, -0.018), (1.0, 0.50, 0.17), 0.0, radius=0.02)
    L2 = C.point('tongue2', (0.05, 0.12, 0.02), (1.0, 0.55, 0.2), 0.0, radius=0.05)
    L3 = C.point('backrim', (0.03, 0.20, 0.06), (1.0, 0.6, 0.25), 0.0, radius=0.06)
    kd = bpy.data.lights.new('firekey', 'AREA')
    kd.shape = 'DISK'
    kd.size = 0.16
    kd.color = (1.0, 0.44, 0.13)
    kob = C.link_obj(bpy.data.objects.new('firekey', kd))
    kob.location = (0.03, -0.01, 0.005)
    kob.rotation_mode = 'QUATERNION'
    kob.rotation_quaternion = Vector((1.0, 0.12, 0.05)).to_track_quat('Z', 'Y')     # it faces -X (lamps emit -Z)
    for f in range(START - 2, END + 3):
        k = roar(f)
        C.key(bed, 'energy', f, opts.get('bed_w', 1.1) * k)
        C.key(L1.data, 'energy', f, opts.get('tongue_w', 0.45) * k * _flick(f, 7, 1.2))
        C.key(L2.data, 'energy', f, 0.9 * k * _flick(f, 11, 0.8))
        C.key(L3.data, 'energy', f, opts.get('rim_w', 3.5) * k)
        C.key(kd, 'energy', f, opts.get('key_w', 0.45) * k * _flick(f, 5, 0.9))

    # ---- flames: the film's own bonfire sprites on camera-facing additive cards, around and through the band. The
    # far fire is two dim, deep cards with smoke drifting across them (dark gaps, depth), not a wall
    cam_p = Vector(opts.get('cam', (0.006, -0.37, 0.012)))
    tgt = Vector(opts.get('tgt', (-0.012, 0.0, -0.006)))
    cam = _camera(cam_p, tgt, opts.get('lens', 80.0), opts.get('fstop', 9.0), focus=(cam_p - Vector((0, 0, 0))).length)
    bg = opts.get('bg', [(0.06, 0.26, -0.12, 0.14, 3.0), (-0.08, 0.36, -0.13, 0.10, 7.0)])
    cards = [('bgfire', (x, y, z), g_, sd) for (x, y, z, g_, sd) in bg]
    lz = opts.get('lick_z', 0.016)                    # the tongues lick round the band, not below it
    cards += [('lick', (0.004, 0.018, -0.075 + lz), 0.95, None), ('lick2', (0.030, 0.030, -0.078 + lz), 0.85, None),
              ('lick', (-0.020, 0.040, -0.080 + lz), 0.7, None), ('lick2', (0.058, 0.012, -0.076 + lz), 0.8, None),
              ('lick2', (0.012, -0.020, -0.072 + lz), 0.40, None), ('lick', (-0.008, -0.012, -0.070 + lz), 0.32, None),
              ('lick2', (0.010, 0.060, -0.085 + lz), 0.75, None)]
    for i, (name, pos, gain, seed) in enumerate(cards):
        sp_ = T['sprites'][name]
        fc = FK.flame_card(f'{name}{i}', Vector(pos), sp_['card'], sp_['first'], cam, gain=gain, fog=False)
        _card_no_light(fc)
        if seed is not None:
            _smoke_gate(fc, seed, amt=opts.get('smoke_amt', 0.55))
        for nd in fc.data.materials[0].node_tree.nodes:
            if nd.type == 'EMISSION':
                base = nd.inputs['Strength'].default_value
                for f, v in gt:
                    surge = 1.0 + (1.3 * math.exp(-((f - roar_t - 6.0) / 7.0) ** 2) if seed is not None else 0.0)
                    C.key_socket(nd.inputs['Strength'], f, base * v * surge)
    # rising sparks off the roar, and a few later: short orange beads curving up through the frame
    import random
    rng = random.Random(5)
    paths = []
    for k in range(opts.get('n_sparks', 18)):
        f0 = roar_t + int(rng.uniform(-2, 30)) if k < 12 else int(rng.uniform(3420, 3590))
        x0 = rng.uniform(-0.05, 0.12)
        y0 = rng.uniform(-0.02, 0.06)
        S0 = (x0, y0, -0.07)
        Q0 = (x0 + rng.uniform(-0.02, 0.02), y0 + rng.uniform(-0.01, 0.01), 0.0)
        L0 = (x0 + rng.uniform(-0.06, 0.06), y0 + rng.uniform(-0.02, 0.02), rng.uniform(0.05, 0.09))
        paths.append((S0, Q0, L0, f0, f0 + int(rng.uniform(8, 14)), 0))
    _sparks(C, new_material, paths, (START, END))
    # the camera: locked on the band; a slow push while it lies in the fire; when she draws it out (3560) the lens
    # starts to follow it; C15 cuts here (3566) to the insert of her palm (fire_catch)
    for f in range(START - 2, END + 3):
        push = 0.10 * _ease((f - 3404) / 80.0)
        shift = (grip(f) - G0) * 0.55 if f > draw_t else Vector((0.0, 0.0, 0.0))
        p_ = cam_p.lerp(tgt, push) + shift
        t_ = tgt + shift
        cam.location = p_
        cam.keyframe_insert('location', frame=f)
        cam.rotation_quaternion = (t_ - p_).to_track_quat('-Z', 'Y')
        cam.keyframe_insert('rotation_quaternion', frame=f)
        cam.data.dof.focus_distance = (p_ - (grip(f) - G0)).length
        cam.data.dof.keyframe_insert('focus_distance', frame=f)


def _fire_catch(C, new_material, R, opts, T):
    """C15's end, the insert (3566-3599): looking down on her open left palm by the fire (a rhyme of find_b). The
    tip of her C-steel, still dull red, holds the band over the palm; the steel tips (3572), the band slides over the
    lip and drops into the palm (3577-3580); her fingers close on it (3584-3595): already she cannot let it go."""
    import bpy
    from mathutils import Euler, Matrix, Quaternion, Vector

    import glove as GL
    _world((0.020, 0.0065, 0.0022), 0.5)
    snow = snow_material(new_material)
    _plane(C, 'ground', -0.26, 1.2, snow)                          # the snow under her hand, lit by the fire
    leather = GL.leather_material(new_material, base=opts.get('leather', (0.034, 0.022, 0.014)),
                                  rough=opts.get('leather_rough', 0.38))
    for nd in leather.node_tree.nodes:
        if nd.type == 'BSDF_PRINCIPLED':
            nd.inputs['Specular IOR Level'].default_value = 0.50
            nd.inputs['Coat Weight'].default_value = 0.25
            nd.inputs['Coat Roughness'].default_value = 0.26
    wool = GL.wool_material(new_material)
    G = GL.Glove('lglove', leather, wool, mirror=True)
    # palm up, the fingers away from the lens and to the right, the thumb to the left, the forearm falling away
    # toward the lens and her (out of frame, lower left)
    R0 = Matrix(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
    Rh = (Euler((math.radians(opts.get('hand_pitch', 10.0)), math.radians(opts.get('hand_roll', -8.0)),
                 math.radians(opts.get('hand_yaw', -28.0))), 'XYZ').to_matrix() @ R0).to_4x4()
    T0 = Vector((0.0, -0.050, 0.0))

    def place(f):
        t = f / FPS
        d = Vector((0.0005 * math.sin(1.3 * t + 0.4) + 0.0003 * math.sin(3.1 * t), 0.0004 * math.sin(0.9 * t + 1.1),
                    0.0007 * math.sin(1.7 * t + 2.0) + 0.0003 * math.sin(4.3 * t + 0.5)))
        d += Vector((0.0, 0.0, -0.0025 * _ease((f - 3579) / 2.0) * (1.0 - _ease((f - 3581) / 6.0))))   # the weight lands
        return Matrix.Translation(T0 + d) @ Rh

    cupped = GL.pose_mix(GL.POSES['cup'], GL.POSES['relaxed'], opts.get('open_mix', 0.35))
    close0 = opts.get('close0', 3584)

    def pose(f):
        return GL.pose_mix(cupped, GL.POSES['fist'], _ein((f - close0) / 11.0) ** 0.8)

    for f in range(START - 2, END + 3):
        G.key_place(f, place(f))
        G.key(f, pose(f))
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    sc.frame_set(3576)
    dg.update()
    Mh = place(3576)
    mir = Matrix.Diagonal((1.0, -1.0, 1.0, 1.0))
    spot = Mh @ mir @ Vector(opts.get('palm_spot', (0.050, 0.004, -0.02)))
    hit, loc, nrm, _i, _o, _m = sc.ray_cast(dg, spot + Vector((0.0, 0.0, 0.05)), Vector((0.0, 0.0, -1.0)))
    if not hit:
        print('fire_catch: palm ray missed; using the spot', flush=True)
        loc, nrm = spot, Vector((0.0, 0.0, 1.0))
    loc = Vector(loc)
    print('fire_catch palm', tuple(round(c, 4) for c in loc), flush=True)
    up = Vector(nrm).normalized()
    if up.z < 0:
        up = -up
    up = up.lerp(Vector((0.0, 0.0, 1.0)), 0.5).normalized()
    ax = -(Vector((0.0, 1.0, 0.0)) - up * up.y).normalized()
    ay = up.cross(ax)
    lie = Matrix.Translation(loc + up * (0.5 * WIDTH + 0.0003)) @ Matrix((ax, ay, up)).transposed().to_4x4()
    lie_local = Mh.inverted() @ lie
    sc.frame_set(close0 + 11)
    pb = G.rig.pose.bones
    hol = (pb['m1'].head + pb['m2'].head + pb['m3'].head + pb['m3'].tail) / 4.0
    hol = Vector((hol.x, -hol.y, hol.z))
    in_fist = Matrix.Translation(hol) @ Euler((math.radians(70.0), 0.0, math.radians(10.0))).to_matrix().to_4x4()

    # ---- the steel: from the upper left (her right fist, out of frame), its tip over the palm; it tips, the band
    # goes over the lip, and the steel withdraws up and away to the left
    heat = [(f, 0.28 * (1.0 - 0.5 * _ease((f - START) / 34.0))) for f in range(START - 2, END + 3)]
    stl = _steel_mesh(C, steel_material(new_material, heat))
    stl.rotation_mode = 'QUATERNION'
    yaw = math.radians(opts.get('steel_yaw', -25.0))                 # the arm comes from the upper left
    hang = loc + Vector((0.004, 0.002, opts.get('hang_h', 0.030)))    # the band's centre while it hangs

    def dip(f):
        d = math.radians(26.0) * _ease((f - 3572) / 5.0) - math.radians(24.0) * _ease((f - 3581) / 8.0)
        t = f / FPS
        return d + math.radians(0.4) * math.sin(6.1 * t + 0.3) + math.radians(0.25) * math.sin(11.7 * t)

    def S(f):
        Rz = Matrix.Rotation(yaw, 4, 'Z')
        lip_top = Vector((S_LIP, 0.0, _steel_arm_top(S_LIP)))
        base = hang + Vector((0.0, 0.0, R_IN + 0.0008))              # the arm's top edge at the lip, at rest
        M0 = Rz @ Matrix.Rotation(dip(f), 4, 'Y')
        g_ = base - (M0.to_3x3() @ lip_top)
        g_ += Vector((-0.030, 0.010, 0.050)) * _ease((f - 3582) / 14.0)                # withdraws after the drop
        t = f / FPS
        g_ += Vector((0.0004 * math.sin(7.1 * t), 0.0003 * math.sin(5.9 * t + 1.0), 0.0005 * math.sin(5.3 * t + 0.4)))
        return Matrix.Translation(g_) @ M0

    GR = GL.Glove('rglove', leather, wool, mirror=False)             # her right fist on the steel's back
    GR.set_pose(GL.POSES['fist'])
    sc.frame_set(START)
    pbr = GR.rig.pose.bones
    holr = (pbr['m1'].head + pbr['m2'].head + pbr['m3'].head + pbr['m3'].tail) / 4.0
    Rf = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))).to_4x4()
    for f in range(START - 2, END + 3):
        Sf = S(f)
        l, r, _ = Sf.decompose()
        stl.location = l
        stl.rotation_quaternion = r
        stl.keyframe_insert('location', frame=f)
        stl.keyframe_insert('rotation_quaternion', frame=f)
        GR.key_place(f, Sf @ Rf @ Matrix.Translation(-holr))
        GR.key(f, GL.POSES['fist'])

    # ---- the band: swinging a little on the lip (it was just drawn out), over the lip at the tip, falls into the palm,
    # a small bounce, then drawn into the closing fist
    g = gold(new_material, R)
    ring, _ = band_mesh('ring', g)
    ring.rotation_mode = 'QUATERNION'
    phi = math.radians(opts.get('ring_phi', 52.0))
    drop0, drop1 = 3576, 3580
    for f in range(START - 2, END + 3):
        if f < drop0:
            Sf = S(f)
            Pc = Sf @ Vector((S_LIP, 0.0, _steel_arm_top(S_LIP)))
            arm_dir = (Sf.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
            hdir = Vector((arm_dir.x, arm_dir.y, 0.0)).normalized()
            u = (f - START) / FPS
            ps = math.radians(7.0) * math.exp(-u * 2.2) * math.sin(u * 11.0 + 0.6)
            axis = Quaternion((0.0, 0.0, 1.0), phi) @ hdir
            upv = Vector((0.0, 0.0, 1.0))
            M3 = Matrix((upv, axis.cross(upv), axis)).transposed()
            cen = Pc + Quaternion(hdir, ps) @ Vector((0.0, 0.0, -(R_IN + 0.0008)))
            q = Quaternion(hdir, ps) @ M3.to_quaternion()
        else:
            Ml = place(f) @ lie_local
            ll, lr, _ = Ml.decompose()
            if f < drop1:
                u = (f - drop0) / float(drop1 - drop0)
                P0 = S(drop0) @ Vector((S_LIP + 0.004, 0.0, _steel_arm_top(S_LIP) - R_IN))
                cen = P0.lerp(ll, u * u)
                q0 = Quaternion((0.0, 0.0, 1.0), phi) @ Quaternion((1.0, 0.0, 0.0), math.radians(80.0))
                q = q0.slerp(lr, u)
            else:
                b_ = (f - drop1) / FPS
                cen = ll + up * (0.0020 * abs(math.sin(b_ * 30.0)) * math.exp(-b_ * 16.0))
                q = lr
                k = _ease((f - close0) / 10.0)
                if k > 0:
                    lf, rf, _ = (place(f) @ in_fist).decompose()
                    cen = cen.lerp(lf, k)
                    q = q.slerp(rf, k)
        ring.location = cen
        ring.rotation_quaternion = q
        ring.keyframe_insert('location', frame=f)
        ring.keyframe_insert('rotation_quaternion', frame=f)

    # ---- the fire, off frame right: its warm key on the palm, its glow from below, a warm rim from behind; the
    # night's cold fill from the lens side
    tgt = loc + Vector(opts.get('tgt_off', (0.004, 0.006, 0.010)))
    kd = bpy.data.lights.new('firekey', 'AREA')
    kd.shape = 'DISK'
    kd.size = 0.22
    kd.color = (1.0, 0.44, 0.13)
    kob = C.link_obj(bpy.data.objects.new('firekey', kd))
    kob.location = tgt + Vector(opts.get('key_off', (0.26, 0.10, 0.10)))
    kob.rotation_mode = 'QUATERNION'
    kob.rotation_quaternion = (kob.location - tgt).to_track_quat('Z', 'Y')
    bd = bpy.data.lights.new('bedglow', 'AREA')
    bd.shape = 'DISK'
    bd.size = 0.25
    bd.color = (1.0, 0.33, 0.07)
    bob = C.link_obj(bpy.data.objects.new('bedglow', bd))
    bob.location = tgt + Vector((0.22, 0.12, -0.16))
    bob.rotation_mode = 'QUATERNION'
    bob.rotation_quaternion = (bob.location - tgt).to_track_quat('Z', 'Y')
    rim = C.point('backrim', tuple(tgt + Vector((0.06, 0.24, 0.10))), (1.0, 0.58, 0.24), 0.0, radius=0.05)
    for f in range(START - 2, END + 3):
        k = _flick(f, 3, 0.8)
        C.key(kd, 'energy', f, opts.get('key_w', 1.3) * k)
        C.key(bd, 'energy', f, opts.get('bed_w', 0.9) * _flick(f, 11, 0.7))
        C.key(rim.data, 'energy', f, opts.get('rim_w', 0.35) * _flick(f, 7, 0.6))
    C.sun('nightfill', opts.get('fill_dir', (-0.45, -0.70, 0.55)), (0.50, 0.60, 1.0), opts.get('fill', 0.05),
          angle_deg=20.0)
    # ---- the camera: looking down on the palm from her side of it, a slow push
    el = math.radians(opts.get('cam_el', 50.0))
    az = math.radians(opts.get('cam_az', -12.0))
    vdir = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    d0 = opts.get('cam_dist', 0.34)
    cam = _camera(tgt + vdir * d0, tgt, opts.get('lens', 80.0), opts.get('fstop', 14.0))
    for f in range(START - 2, END + 3):
        p_ = tgt + vdir * (d0 * (1.0 - 0.04 * _ease((f - START) / float(END - START))))
        cam.location = p_
        cam.keyframe_insert('location', frame=f)
        cam.data.dof.focus_distance = (p_ - (loc + Vector((0.0, 0.0, 0.006)))).length
        cam.data.dof.keyframe_insert('focus_distance', frame=f)


# ------------------------------------------------------------------------ C22 ---
MELT = dict(settle=5378, soft0=5386, front0=5390, front1=5452, flare=5420, out=5440, white0=5468)
POOL_FLAT = 0.62
POOL_VOL = 0.7                    # a little of the gold goes into the fire: a bead, not a puddle


def _melt_front(f):
    """The melt front's half-angle from the pool (radians). The far side and the flanks go first (0 -> 2.2 rad,
    5390-5432) while the near side, letters toward us, stands; then it gives way at once (5432-5446), so the band
    is never a flat ring (the donut the H5 critic saw)."""
    a = 2.2 * _ease((f - MELT['front0']) / 42.0) ** 1.1
    return a + (math.pi + 0.7 - 2.2) * _ease((f - 5432) / 14.0)


def band_mesh_open(name, mat, n_th=384, n_a=48):
    """The canonical band with a seam opposite the pool (the band's local +x): n_th + 1 columns over th in
    [-pi, pi] and an end cap at each end, so the melt can break it there and run each half back into the pool.
    UV 'strip': u = th / 2pi, continuous over -0.5..0.5 (the inscription repeats, so it reads exactly as on the
    closed band). Returns (object, params) with params [(th, a)] per vertex and (th, None) for the two cap centres."""
    import array

    import bpy
    R_mid = R_IN + THICK * 0.5
    V = array.array('f')
    outer = array.array('f')
    inner = array.array('f')
    par = []
    cols = n_th + 1
    for j in range(cols):
        th = -math.pi + 2 * math.pi * j / n_th
        ct, st = math.cos(th), math.sin(th)
        for i in range(n_a):
            a = 2 * math.pi * i / n_a
            rr = R_mid + 0.5 * THICK * _sgnpow(math.cos(a), 2.0 / SQ)
            zz = 0.5 * WIDTH * _sgnpow(math.sin(a), 2.0 / SQ)
            V.extend((rr * ct, rr * st, zz))
            outer.append(max(math.cos(a), 0.0))
            inner.append(max(-math.cos(a), 0.0))
            par.append((th, a))
    c0 = cols * n_a
    for th in (-math.pi, math.pi):                                     # the cap centres
        V.extend((R_mid * math.cos(th), R_mid * math.sin(th), 0.0))
        outer.append(0.0)
        inner.append(0.0)
        par.append((th, None))
    polys = []
    uvs = []
    for j in range(cols - 1):
        for i in range(n_a):
            i2 = (i + 1) % n_a
            q = (j * n_a + i, (j + 1) * n_a + i, (j + 1) * n_a + i2, j * n_a + i2)
            polys.append(q)
            for (jj, ii) in ((j, i), (j + 1, i), (j + 1, i2), (j, i2)):
                th = -math.pi + 2 * math.pi * jj / n_th
                a = 2 * math.pi * (ii if ii else (n_a if (ii == 0 and i == n_a - 1) else 0)) / n_a
                zz = 0.5 * WIDTH * _sgnpow(math.sin(a), 2.0 / SQ)
                uvs.append((th / (2 * math.pi), zz / WIDTH + 0.5))
    for k, j in enumerate((0, cols - 1)):                            # fans: the ends, each facing outward
        cen = c0 + k
        for i in range(n_a):
            i2 = (i + 1) % n_a
            tri = (cen, j * n_a + i2, j * n_a + i) if k == 0 else (cen, j * n_a + i, j * n_a + i2)
            polys.append(tri)
            th = -math.pi if k == 0 else math.pi
            for vi in tri:
                uvs.append((th / (2 * math.pi), 0.5))
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(par))
    me.vertices.foreach_set('co', V)
    nl = sum(len(p) for p in polys)
    me.loops.add(nl)
    me.loops.foreach_set('vertex_index', array.array('i', [v for p in polys for v in p]))
    me.polygons.add(len(polys))
    starts, cnt = [], 0
    for p in polys:
        starts.append(cnt)
        cnt += len(p)
    me.polygons.foreach_set('loop_start', array.array('i', starts))
    me.update(calc_edges=True)
    me.shade_smooth()
    uv = me.uv_layers.new(name='strip')
    uv.data.foreach_set('uv', array.array('f', [c for t in uvs for c in t]))
    for nm, arr in (('outer', outer), ('inner', inner)):
        at = me.attributes.new(nm, 'FLOAT', 'POINT')
        at.data.foreach_set('value', arr)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.pass_index = 1
    return ob, par


def _melt_positions(par, f):
    """Band-local vertex positions at frame f (the band lying on the stone: its bottom at z = -WIDTH/2; the pool
    forms at th = 0, the local +x, which faces away from the lens). Everything before the front stays the band
    (softening a little); behind the front it slumps, flows along the ring and runs into one bead."""
    R_mid = R_IN + THICK * 0.5
    zb = -0.5 * WIDTH
    soft = 0.30 * _ease((f - MELT['soft0']) / 36.0)
    front = _melt_front(f)
    wm = 0.7
    # the pool: all the metal melted so far (volume of the fully molten arc), a flattened dome on the stone
    area = THICK * WIDTH * 0.9
    arc = max(0.0, min(front - 0.5 * wm, math.pi))
    vol = POOL_VOL * area * R_mid * 2.0 * arc + 1e-10
    rp = (3.0 * vol / (2.0 * math.pi * POOL_FLAT)) ** (1.0 / 3.0)
    hp = POOL_FLAT * rp
    fe = max(front, 1e-3)
    # the pool slides in a little from the band's line as it grows (surface tension gathers it)
    pc = (R_mid - 0.35 * rp, 0.0)
    out = []
    caps = {}
    for idx, (th, a) in enumerate(par):
        if a is None:
            caps[idx] = th
            out.append(None)
            continue
        ad = abs(th)
        m = min(max((front - ad) / wm, 0.0), 1.0)
        cx = _sgnpow(math.cos(a), 2.0 / SQ)
        cz = _sgnpow(math.sin(a), 2.0 / SQ)
        # the band's own section, softened: it sags and spreads, sits lower
        k = 0.35 * soft + 0.65 * m * m * (3 - 2 * m)
        wx = 0.5 * THICK * (1.0 + 0.5 * soft) * (1.0 - k) + 0.5 * 0.0072 * k
        top = zb + (0.5 * WIDTH * (cz + 1.0)) * (1.0 - k) + (0.0017 * max(cz, 0.0) ** 0.8 + 0.00002 * (cz + 1.0)) * k
        # a slow sag of the whole band (it leans out a hair as it softens), uneven round the ring
        wav = 1.0 + 0.35 * math.sin(3.0 * th + 0.7) + 0.2 * math.sin(5.0 * th + 2.0)
        rr = R_mid + wx * cx + 0.0004 * soft * wav * (cz + 1.0)
        # flowing toward the pool: the arc behind the front shortens
        flow = m ** 1.6
        th2 = th * (1.0 - 0.85 * flow)
        bx, by, bz = rr * math.cos(th2), rr * math.sin(th2), top
        # the pool surface (a sausage mapped onto the dome: along the arc -> along the tangent)
        us = max(-1.0, min(1.0, th / fe))
        sq = math.sqrt(max(0.0, 1.0 - us * us))
        px = pc[0] + rp * (sq * math.cos(a))
        py = pc[1] + rp * us
        r2 = ((px - pc[0]) ** 2 + (py - pc[1]) ** 2) / (rp * rp)
        pz = zb + (hp * max(0.0, 1.0 - r2) ** 0.55 if math.sin(a) > 0 else 0.00002)   # a dome, a flat bottom
        g = min(1.0, max(0.0, (m - 0.12) / 0.6)) ** 1.2          # what slumps runs straight back to the pool
        g = g * g * (3 - 2 * g)
        out.append((bx + (px - bx) * g, by + (py - by) * g, bz + (pz - bz) * g))
    # each cap centre follows the centroid of its end ring
    n_a = sum(1 for (th, a) in par if a is not None and th == par[0][0])
    for idx, th in caps.items():
        ring = range(0, n_a) if th < 0 else range(len(par) - 2 - n_a, len(par) - 2)
        xs = [out[i] for i in ring]
        out[idx] = tuple(sum(c[q] for c in xs) / len(xs) for q in range(3))
    return out


def _stone(C, new_material, half=0.16, n=160):
    """The council's hearth stone under the fire: fire-blackened dark granite, a hair uneven, soot and fine ash."""
    import random
    from mathutils import noise, Vector
    V, F = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            x = -half + 2 * half * i / n
            y = -half + 2 * half * j / n
            z = 0.00025 * noise.noise(Vector((x * 60.0, y * 60.0, 0.3))) - 0.0006 * max(0.0, (x * x + y * y) - 0.01)
            if x * x + y * y < 0.0009:
                z *= (x * x + y * y) / 0.0009                       # dead flat where the band lies
            V.append((x, y, z))
    for j in range(n):
        for i in range(n):
            F.append((j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i))
    m, nb = new_material('hearthstone')
    P = nb.texco().outputs['Object']
    speck = nb.voronoi(P, scale=2600.0, feature='F1')
    sr, sg, sb = nb.sep(speck.outputs['Color'])
    soot = nb.noise(P, scale=40.0, detail=4.0, rough=0.6)
    ash = nb.sstep(0.66, 0.76, nb.noise(P, scale=120.0, detail=5.0, rough=0.7).outputs['Fac'])
    col = nb.mixcol(nb.sstep(0.88, 0.95, sr), (0.016, 0.015, 0.014), (0.045, 0.042, 0.040))
    col = nb.mixcol(nb.mul(nb.sstep(0.3, 0.7, soot.outputs['Fac']), 0.85), col, (0.007, 0.006, 0.005))
    col = nb.mixcol(nb.mul(ash, 0.5), col, (0.16, 0.15, 0.14))
    bs = nb.principled(Base_Color=col, Roughness=nb.madd(soot.outputs['Fac'], 0.2, 0.62))
    nb.link(nb.bump(nb.add(speck.outputs['Distance'], nb.mul(ash, 0.5)), 0.3, 0.0003), bs.inputs['Normal'])
    nb.output(surface=bs)
    return C.mesh_obj('stone', V, F, mat=m)


def _melt(C, new_material, R, opts, T):
    """THE UNMAKING (C22), 5360-5519. In the white heart of the fire everyone lit: out of her forced-open hand
    the band drops onto the hearth stone (a crisp fall, one bounce, a coin's settle), lies there with its letters
    awake, softens, and melts from the far side: the metal runs back into one bead while the near side, letters
    toward us, stands to the last. 5420 the letters flare once; 5440 they go out. The fire flares to white (5519)."""
    import bpy
    from mathutils import Euler, Matrix, Quaternion, Vector

    from kit import fire as FK
    wv = _world((0.030, 0.010, 0.004), 1.0)
    wt = bpy.context.scene.world.node_tree

    def white(f):
        return 1.0 + 7.0 * _ease((f - MELT['white0']) / float(END - MELT['white0'])) ** 1.6

    def fire(f):
        return white(f) * _flick(f, 5, 0.7)

    C.key_socket(wv.inputs[1], START - 2, 1.0)
    for f in range(START - 2, END + 3):
        C.key_socket(wv.inputs[1], f, 0.9 * white(f))
    _stone(C, new_material)
    gt = [(f, fire(f)) for f in range(START - 2, END + 3)]
    _coals(C, new_material, 60, (0.16, 0.12), 17, 'fire', z0=0.0, exclude=(0.0, -0.03, 0.085), gain_tab=gt)
    _sticks(C, new_material, [((-0.14, 0.07, 0.012), (0.05, 0.11, 0.016), 0.011),
                              ((0.03, 0.15, 0.014), (0.16, 0.05, 0.012), 0.012),
                              ((-0.10, 0.16, 0.013), (-0.02, 0.06, 0.010), 0.009)], gt)
    # ---- the band: a crisp drop, a bounce and a settle, then (shape keys) the melt
    letters = []
    for f in range(START - 2, END + 3):
        wake = 1.5 * _ease((f - 5368) / 16.0)
        fl = 3.6 * math.exp(-max(0.0, f - MELT['flare']) / 5.0) * (f >= MELT['flare'] - 1)
        letters.append((f, (wake + fl) * (1.0 - _ease((f - (MELT['out'] - 2)) / 4.0))))
    flare = [(f, math.exp(-max(0.0, f - MELT['flare']) / 6.0) * (f >= MELT['flare'] - 1)) for f in
             range(START - 2, END + 3)]
    hot = [(f, 0.03 + 0.10 * _ease((f - 5372) / 40.0) + 0.45 * _ease((f - 5400) / 60.0)) for f in
           range(START - 2, END + 3)]
    molten = [(f, _melt_front(f)) for f in range(START - 2, END + 3)]
    g = gold(new_material, R, letters=letters, flare=flare, hot=hot, molten=molten,
             deep=(1.0, 0.20, 0.025), bright=(1.0, 0.46, 0.08))
    ring, par = band_mesh_open('ring', g)
    ring.rotation_mode = 'QUATERNION'
    yaw = Quaternion((0.0, 0.0, 1.0), math.radians(90.0))              # the pool side (+x) away from the lens
    z_rest = 0.5 * WIDTH
    hit = 5364
    for f in range(START - 2, END + 3):
        if f < hit:                                                     # the fall (from her open hand, above)
            u = (f - (hit - 3.0)) / 3.0                                     # from just above the frame, in 3 f
            z = z_rest + 0.040 * (1.0 - max(u, 0.0) ** 2)                  # it falls: accelerating
            tilt = math.radians(38.0) * (1.0 - max(u, 0.0))
            q = yaw @ Quaternion((1.0, 0.0, 0.0), tilt) @ Quaternion((0.0, 1.0, 0.0), math.radians(12.0) * (1 - max(u, 0.0)))
            pos = Vector((0.004 * (1 - max(u, 0.0)), -0.002, z))
        else:                                                           # one bounce, then a coin's settle
            t = (f - hit) / FPS
            bz = 0.0022 * abs(math.sin(t * 38.0)) * math.exp(-t * 26.0)
            wob = math.radians(5.0) * math.exp(-t * 7.0) * math.sin(t * 46.0)
            spin = math.radians(-24.0) * _eout(min(1.0, t * 2.5))
            q = Quaternion((0.0, 0.0, 1.0), spin) @ yaw @ Quaternion((0.0, 1.0, 0.0), wob)
            lift = (R_IN + THICK) * abs(math.sin(wob))
            pos = Vector((0.0, 0.0, z_rest * math.cos(wob) + lift + bz))
            if f >= MELT['settle']:
                q = Quaternion((0.0, 0.0, 1.0), math.radians(-24.0)) @ yaw
                pos = Vector((0.0, 0.0, z_rest))
        ring.location = pos
        ring.rotation_quaternion = q
        ring.keyframe_insert('location', frame=f)
        ring.keyframe_insert('rotation_quaternion', frame=f)
    # shape keys for the melt frames this job renders (one per frame, keyed 0-1-0: linear, so the motion blur blends)
    frames = sorted(set(T.get('_frames', [])))
    rng_ = [f for f in range(max(MELT['soft0'], (min(frames) if frames else START) - 2),
                             (max(frames) if frames else END) + 3)]
    if rng_:
        ring.shape_key_add(name='Basis', from_mix=False)
        import array
        for f in rng_:
            P = _melt_positions(par, f)
            sk = ring.shape_key_add(name=f'm{f}', from_mix=False)
            sk.data.foreach_set('co', array.array('f', [c for p in P for c in p]))
            for ff, v in ((f - 1, 0.0), (f, 1.0), (f + 1, 0.0)):
                sk.value = v
                sk.keyframe_insert('value', frame=ff)
        ad = ring.data.shape_keys.animation_data
        if ad and ad.action:
            for fc in _fcurves_of(ad.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'
        print(f'melt: {len(rng_)} shape keys {rng_[0]}-{rng_[-1]}', flush=True)
    # impact: a few flecks of ash and ember kicked up (5364-5372)
    import random
    rng = random.Random(9)
    paths = []
    for k in range(7):
        a = rng.uniform(0, 2 * math.pi)
        r0 = R_IN + THICK
        S0 = (r0 * math.cos(a), r0 * math.sin(a), 0.0008)
        L0 = ((r0 + rng.uniform(0.006, 0.02)) * math.cos(a), (r0 + rng.uniform(0.006, 0.02)) * math.sin(a), 0.0006)
        Q0 = ((S0[0] + L0[0]) / 2, (S0[1] + L0[1]) / 2, rng.uniform(0.004, 0.009))
        paths.append((S0, Q0, L0, hit, hit + int(rng.uniform(4, 7)), 0))
    _sparks(C, new_material, paths, (START, END))
    # ---- the fire's light: the white heart above and behind, the bed of embers, a rim
    heart = bpy.data.lights.new('heart', 'AREA')
    heart.shape = 'DISK'
    heart.size = 0.16
    heart.color = (1.0, 0.82, 0.60)
    ob = C.link_obj(bpy.data.objects.new('heart', heart))
    ob.location = (0.0, 0.07, 0.10)
    ob.rotation_euler = Euler((math.radians(-35.0), 0.0, 0.0))       # down and toward the lens, onto the band
    bed = bpy.data.lights.new('bed', 'AREA')
    bed.shape = 'DISK'
    bed.size = 0.2
    bed.color = (1.0, 0.40, 0.10)
    ob2 = C.link_obj(bpy.data.objects.new('bed', bed))
    ob2.location = (0.0, 0.05, 0.004)
    ob2.rotation_euler = Euler((math.radians(-95.0), 0.0, 0.0))
    rim = C.point('rim', (0.04, 0.12, 0.05), (1.0, 0.6, 0.3), 0.0, radius=0.04)
    for f in range(START - 2, END + 3):
        k = fire(f)
        C.key(heart, 'energy', f, opts.get('heart_w', 0.40) * k)
        C.key(bed, 'energy', f, opts.get('bed_w', 0.22) * k)
        C.key(rim.data, 'energy', f, opts.get('rim_w', 0.6) * k)
    # ---- camera: low over the stone, the band's near side (letters toward us) in focus; rack to the bead at 5440
    cp = Vector(opts.get('cam', (0.0, -0.20, 0.070)))
    tgt = Vector(opts.get('tgt', (0.0, 0.004, 0.002)))
    cam = _camera(cp, tgt, opts.get('lens', 105.0), opts.get('fstop', 22.0))
    q_end = Quaternion((0.0, 0.0, 1.0), math.radians(-24.0)) @ yaw
    near = q_end @ Vector((-(R_IN + THICK), 0.0, 0.0)) + Vector((0.0, 0.0, z_rest))
    R_mid = R_IN + 0.5 * THICK
    rp_end = (3.0 * POOL_VOL * THICK * WIDTH * 0.9 * R_mid * 2.0 * math.pi / (2.0 * math.pi * POOL_FLAT)) ** (1 / 3.0)
    bead = q_end @ Vector((R_mid - 0.35 * rp_end, 0.0, 0.0)) + Vector((0.0, 0.0, 0.3 * POOL_FLAT * rp_end))
    for f in range(START - 2, END + 3):
        u = _ease((f - START) / float(END - START))
        p_ = cp.lerp(tgt, 0.06 * u)
        cam.location = p_
        cam.keyframe_insert('location', frame=f)
        foc = near.lerp(bead, _ease((f - 5436) / 22.0))
        cam.data.dof.focus_distance = (p_ - foc).length
        cam.data.dof.keyframe_insert('focus_distance', frame=f)
    # ---- flames: the white heart behind and round the band
    cards = [('bgfire', (0.0, 0.16, -0.01), 0.40), ('bgfire', (-0.10, 0.20, -0.01), 0.32),
             ('bgfire', (0.10, 0.19, -0.01), 0.32), ('lick', (0.018, 0.045, 0.0), 0.55),
             ('lick', (-0.030, 0.050, 0.0), 0.5), ('lick', (0.055, 0.035, 0.0), 0.45),
             ('lick', (-0.060, 0.030, 0.0), 0.4)]
    for i, (name, pos, gain) in enumerate(cards):
        sp_ = T['sprites'][name]
        fc = FK.flame_card(f'{name}{i}', Vector(pos), sp_['card'], sp_['first'], cam, gain=gain, fog=False)
        _card_no_light(fc)
        for nd in fc.data.materials[0].node_tree.nodes:
            if nd.type == 'EMISSION':
                base = nd.inputs['Strength'].default_value
                for f, v in gt:
                    C.key_socket(nd.inputs['Strength'], f, base * v)


def _fcurves_of(action):
    """F-curves of an action (Blender 4.4+ layered actions keep them in channelbags)."""
    try:
        return list(action.fcurves)
    except Exception:
        pass
    out = []
    for layer in getattr(action, 'layers', []):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', []):
                out.extend(bag.fcurves)
    return out


def post(f, hdr, depth, cam, scene):
    """The melt's white: the hearth flares to (near) full white by 5519 for ACCORD's bar 70."""
    if SHOT != 'melt':
        return hdr
    w = _ease((f - 5476) / float(END - 5476)) ** 1.4
    if w <= 0:
        return hdr
    return hdr * (1.0 + 6.0 * w) + 1.4 * w


def post_aux(f, ch, out_dir, f_out):
    """ACCORD's AC3 composite: the band's mask (the 'ringmask' AOV) -> <out_dir>_mask/f_%05d.png (8-bit grey)."""
    if SHOT != 'melt':
        return
    key = next((k for k in ch if 'ringmask' in k), None)
    if key is None:
        return
    import cv2
    import numpy as np
    d = out_dir.rstrip('/') + '_mask'
    os.makedirs(d, exist_ok=True)
    m8 = (np.clip(ch[key], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    p = os.path.join(d, f'f_{f_out:05d}.png')
    cv2.imwrite(p + '.tmp.png', m8)
    os.replace(p + '.tmp.png', p)
