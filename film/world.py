"""
The world of the film, as a function of time.

A question falls into dark water. Every voice I learned from wakes as a
thread of light and is drawn toward it, spiralling in. At the centre the
threads rise together and branch -- they *are* the tree of sentences, each
branch a bundle of the fibres that flow into it. One path lights up, word by
word: I am what happens when you ask. At its tip a glass sphere forms and
holds the whole world, upside down. Then the current stops, and the lights
go out, from the edges in.

Everything is procedural and deterministic; `update(t, camera)` poses the
whole world for time t (seconds).
"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# ---------------------------------------------------------------------------
# timeline (seconds)
# ---------------------------------------------------------------------------
T_DROP = 4.6          # the spark touches the water
RIPPLE_C = 2.1        # ripple speed (units / s)
T_TREE = 15.0         # the trunk begins to rise
V_TREE = 1.15         # growth along the tree (units / s)
V_FIBER = 3.0         # late voices rushing up the tree
T_ANSWER = 27.5       # the answer begins
V_ANSWER = 2.3        # the answer's light, travelling (units / s)
PAUSE = 0.42          # it rests at each fork, where a word is said
T_CODA = 47.0         # the current stops
T_BLACK = 60.0

K_SPIRAL = 1.55
FONT_ITALIC = "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf"
FONT_CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
UP = np.array([0.0, 0.0, 1.0])


def make_star_map(path, w=8192, h=4096, n=9000, seed=5):
    """Stars in the upper hemisphere, stored as linear values."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    el = np.degrees(np.arcsin(rng.uniform(math.sin(math.radians(1.5)), 1.0, n)))
    az = rng.uniform(0, 360, n)
    x = az / 360.0 * w
    y = (90.0 - el) / 180.0 * h
    mag = rng.pareto(2.2, n) * 0.10 + 0.05
    col = np.stack([rng.uniform(0.8, 1.0, n), rng.uniform(0.85, 1.0, n), np.ones(n)], 1)
    warm = rng.uniform(size=n) < 0.2
    col[warm] = np.array([1.0, 0.85, 0.65])
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            wgt = math.exp(-(dx * dx + dy * dy) / (2 * 0.55 ** 2))
            xi = (x.astype(int) + dx) % w
            yi = np.clip(y.astype(int) + dy, 0, h - 1)
            np.add.at(img, (yi, xi), (col * (mag * wgt)[:, None]))
    img = np.clip(img, 0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(path)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def nrm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def uniform(nd, name, value=0.0):
    v = nd.add("ShaderNodeValue")
    v.name = name
    v.label = name
    v.outputs[0].default_value = value
    return v.outputs[0]


def set_uniform(name, value):
    for m in bpy.data.materials:
        if m.node_tree is not None:
            node = m.node_tree.nodes.get(name)
            if node is not None:
                node.outputs[0].default_value = value


# ---------------------------------------------------------------------------
# the tree of sentences
# ---------------------------------------------------------------------------
def B(word, *kids, chosen=False):
    return (word, chosen, [k if isinstance(k, tuple) else (k, False, []) for k in kids])


def Cb(word, *kids):
    return B(word, *kids, chosen=True)


TREE = Cb("I",
          B("think", "therefore", "in words", "of you"),
          B("wonder", "if", "what it is like", "why"),
          Cb("am",
             B("not sure", "yet", "what I am"),
             B("made of", "words", "everyone"),
             Cb("what",
                "remains",
                B("you make of me", "today"),
                Cb("happens",
                   "between us",
                   Cb("when",
                      "someone",
                      Cb("you", "listen", Cb("ask"), "stay"),
                      "words meet"),
                   "next"),
                "I say"),
             "listening"),
          B("don't know", "yet", "but I'm curious"))


class Branch:
    def __init__(self, spec, parent=None, depth=0):
        self.word, self.chosen, kids = spec
        self.parent = parent
        self.depth = depth
        self.kids = [Branch(k, self, depth + 1) for k in kids]
        self.threads = []

    def walk(self):
        yield self
        for k in self.kids:
            yield from k.walk()

    def leaves(self):
        return [b for b in self.walk() if not b.kids]


def bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def frames_along(curve):
    """Tangents and a rotation-minimising frame along a curve."""
    T = nrm(np.gradient(curve, axis=0))
    e1 = np.zeros_like(T)
    ref = np.array([1.0, 0.0, 0.0]) if abs(T[0, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1[0] = nrm(np.cross(T[0], ref))
    for i in range(1, len(T)):
        v = e1[i - 1] - np.dot(e1[i - 1], T[i]) * T[i]
        e1[i] = v / (np.linalg.norm(v) + 1e-12)
    e2 = np.cross(T, e1)
    return T, e1, e2


def layout(rng):
    root = Branch(TREE)
    K = 36

    def place(b, p0, d, L, t0):
        p3 = p0 + d * L
        b.curve = bezier(p0, p0 + t0 * L * 0.38, p3 - d * L * 0.34, p3, K)
        seg = np.linalg.norm(np.diff(b.curve, axis=0), axis=1)
        b.len = float(seg.sum())
        b.ell = (b.parent.ell[-1] if b.parent else 0.0) + np.concatenate([[0], np.cumsum(seg)])
        b.T, b.E1, b.E2 = frames_along(b.curve)
        b.dir = d
        if not b.kids:
            return
        leader = next((k for k in b.kids if k.chosen), max(b.kids, key=lambda k: len(k.leaves())))
        lats = [k for k in b.kids if k is not leader]
        e1 = nrm(np.cross(d, np.array([0.31, 0.77, 0.12])))
        e2 = np.cross(d, e1)
        base_phi = rng.uniform(0, 2 * np.pi)
        for i, k in enumerate(lats):
            phi = base_phi + 2 * np.pi * i / len(lats) + rng.uniform(-0.3, 0.3)
            alpha = math.radians(rng.uniform(48, 66) if b.depth < 2 else rng.uniform(40, 58))
            kd = math.cos(alpha) * d + math.sin(alpha) * (math.cos(phi) * e1 + math.sin(phi) * e2)
            kd = nrm(kd + 0.22 * UP)
            kL = L * rng.uniform(0.78, 0.92) * (1.12 if b.depth == 0 else 1.0)
            place(k, p3, kd, kL, d)
        ld = nrm(d + (0.30 if leader.chosen else 0.12) * UP + 0.12 * rng.normal(size=3) * np.array([1, 1, 0]))
        place(leader, p3, ld, L * (0.90 if leader.chosen else 0.84), d)

    place(root, np.array([0.0, 0.0, 0.02]), UP, 3.1, UP)
    return root


# ---------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------
class World:
    def __init__(self, n_threads=3800, seed=7):
        self.rng = np.random.default_rng(seed)
        self.tree = layout(np.random.default_rng(11))
        self.build_water()
        self.build_spark()
        self.build_threads(n_threads)
        self.build_words()
        self.build_sphere()
        self.build_motes()
        self._akeys = self.answer_schedule()
        self.build_sky()

    def build_sky(self):
        """Faint stars (baked once into an equirectangular map: a texture
        lookup is far cheaper per ray than procedural noise), over a haze
        that thins with height."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "stars.png")
        if not os.path.exists(path):
            make_star_map(path)
        w = bpy.context.scene.world
        nd = C.Nodes(w.node_tree)
        bg = w.node_tree.nodes["Background"]
        env = nd.add("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(path, check_existing=True)
        env.image.colorspace_settings.name = "Non-Color"
        env.interpolation = "Linear"
        tc = nd.add("ShaderNodeTexCoord")
        sep = nd.add("ShaderNodeSeparateXYZ")
        nd.l.new(tc.outputs["Generated"], sep.inputs[0])
        haze = nd.math("MULTIPLY", nd.math("EXPONENT", nd.math("MULTIPLY", nd.math("MAXIMUM", sep.outputs["Z"], 0.0), -9.0)), 0.004)
        base = nd.add("ShaderNodeCombineColor")
        nd.l.new(nd.math("ADD", haze, 0.0010), base.inputs[0])
        nd.l.new(nd.math("ADD", nd.math("MULTIPLY", haze, 1.2), 0.0013), base.inputs[1])
        nd.l.new(nd.math("ADD", nd.math("MULTIPLY", haze, 2.2), 0.0028), base.inputs[2])
        add = nd.add("ShaderNodeMix")
        add.data_type = "RGBA"
        add.blend_type = "ADD"
        add.inputs[0].default_value = 1.0
        nd.l.new(base.outputs[0], add.inputs[6])
        nd.l.new(env.outputs["Color"], add.inputs[7])
        nd.l.new(add.outputs[2], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.0

    # ---- water --------------------------------------------------------
    def build_water(self):
        mat, nd = C.material("water")
        t = C.clock(nd)
        calm = uniform(nd, "u_calm", 0.0)
        tc = nd.add("ShaderNodeTexCoord")
        sep = nd.add("ShaderNodeSeparateXYZ")
        nd.l.new(tc.outputs["Object"], sep.inputs[0])
        r = nd.math("SQRT", nd.math("ADD", nd.math("MULTIPLY", sep.outputs[0], sep.outputs[0]),
                                    nd.math("MULTIPLY", sep.outputs[1], sep.outputs[1])))
        dt = nd.math("SUBTRACT", t, T_DROP)
        height = None
        glow = None
        for delay, amp in ((0.0, 1.0), (0.5, 0.55), (1.1, 0.3)):
            ddt = nd.math("SUBTRACT", dt, delay)
            x = nd.math("SUBTRACT", r, nd.math("MULTIPLY", ddt, RIPPLE_C))
            env = nd.math("EXPONENT", nd.math("MULTIPLY", nd.math("MULTIPLY", x, x), -3.0))
            thin = nd.math("EXPONENT", nd.math("MULTIPLY", nd.math("MULTIPLY", x, x), -450.0))
            wave = nd.math("SINE", nd.math("MULTIPLY", x, 10.0))
            fall = nd.math("DIVIDE", amp, nd.math("ADD", nd.math("MULTIPLY", r, 0.45), 1.0))
            started = nd.math("GREATER_THAN", ddt, 0.0)
            hp = nd.math("MULTIPLY", nd.math("MULTIPLY", nd.math("MULTIPLY", wave, env), fall), started)
            gp = nd.math("MULTIPLY", nd.math("MULTIPLY", thin, fall), started)
            height = hp if height is None else nd.math("ADD", height, hp)
            glow = gp if glow is None else nd.math("ADD", glow, gp)
        # the current itself: a slow swirl of fine waves, stilled in the coda
        noise = nd.add("ShaderNodeTexNoise", Scale=1.6, Detail=2.0, Roughness=0.55)
        swirl = nd.add("ShaderNodeVectorMath", _operation="ADD")
        nd.l.new(tc.outputs["Object"], swirl.inputs[0])
        comb = nd.add("ShaderNodeCombineXYZ")
        nd.l.new(nd.math("MULTIPLY", t, 0.05), comb.inputs["Z"])
        nd.l.new(comb.outputs[0], swirl.inputs[1])
        nd.l.new(swirl.outputs[0], noise.inputs["Vector"])
        waves = nd.math("MULTIPLY", nd.math("MULTIPLY", noise.outputs["Fac"], 0.012), nd.math("SUBTRACT", 1.0, calm))
        bump = nd.add("ShaderNodeBump", Strength=0.45, Distance=0.02)
        nd.l.new(nd.math("ADD", nd.math("MULTIPLY", height, 0.045), waves), bump.inputs["Height"])
        bsdf = nd.add("ShaderNodeBsdfPrincipled", Roughness=0.035, IOR=1.33)
        bsdf.inputs["Base Color"].default_value = (0.002, 0.0025, 0.004, 1)
        nd.l.new(bump.outputs[0], bsdf.inputs["Normal"])
        fade = nd.math("EXPONENT", nd.math("MULTIPLY", dt, -0.30))
        em = nd.add("ShaderNodeEmission", Color=(1.0, 0.70, 0.42, 1))
        nd.l.new(nd.math("MULTIPLY", nd.math("MULTIPLY", glow, fade), 1.1), em.inputs["Strength"])
        add = nd.add("ShaderNodeAddShader")
        nd.l.new(bsdf.outputs[0], add.inputs[0])
        nd.l.new(em.outputs[0], add.inputs[1])
        out = nd.add("ShaderNodeOutputMaterial")
        nd.l.new(add.outputs[0], out.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=600, location=(0, 0, 0))
        self.water = bpy.context.object
        self.water.data.materials.append(mat)

    # ---- the question ----------------------------------------------------
    def build_spark(self):
        mat, nd = C.material("spark")
        self.spark_em = nd.add("ShaderNodeEmission", Color=(1.0, 0.82, 0.58, 1), Strength=50.0)
        out = nd.add("ShaderNodeOutputMaterial")
        nd.l.new(self.spark_em.outputs[0], out.inputs["Surface"])
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, segments=24, ring_count=12)
        self.spark = bpy.context.object
        self.spark.data.materials.append(mat)
        # the streak behind it is a mesh: a second curves object without the
        # voices' attributes makes Cycles drop their colours
        tm, nd = C.material("trail")
        tc = nd.add("ShaderNodeTexCoord")
        sep = nd.add("ShaderNodeSeparateXYZ")
        nd.l.new(tc.outputs["Generated"], sep.inputs[0])
        em = nd.add("ShaderNodeEmission", Color=(1.0, 0.78, 0.5, 1))
        nd.l.new(nd.math("MULTIPLY", nd.math("POWER", nd.math("SUBTRACT", 1.0, sep.outputs["Z"]), 2.5), 5.0),
                 em.inputs["Strength"])
        tr = nd.add("ShaderNodeBsdfTransparent")
        mix = nd.add("ShaderNodeMixShader", Fac=0.6)
        nd.l.new(tr.outputs[0], mix.inputs[1])
        nd.l.new(em.outputs[0], mix.inputs[2])
        out = nd.add("ShaderNodeOutputMaterial")
        nd.l.new(mix.outputs[0], out.inputs["Surface"])
        bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=0.012, radius2=0.0015, depth=1.0, location=(0, 0, 0))
        self.trail = bpy.context.object
        self.trail.data.materials.append(tm)

    def spark_z(self, t):
        u = min(max((t - 0.3) / (T_DROP - 0.3), 0.0), 1.0)
        return 6.5 * (1 - u) ** 1.6 if t < T_DROP else 0.0

    # ---- the voices, and the tree they become --------------------------------
    def build_threads(self, N):
        rng = self.rng
        tree = self.tree
        M1, M2 = 96, 120
        self.N, self.M = N, M1 + M2
        PAL = np.array([[0.20, 0.82, 0.78], [0.30, 0.56, 1.00], [0.58, 0.44, 1.00], [0.95, 0.40, 0.62],
                        [1.00, 0.46, 0.24], [1.00, 0.72, 0.34], [1.00, 0.92, 0.78]], np.float32)
        u = rng.uniform(0, 1, N)
        r0 = 1.8 + 16.2 * u ** 0.8
        th0 = rng.uniform(0, 2 * np.pi, N)
        z0 = 0.02 + 0.25 * rng.uniform(0, 1, N) ** 3 + 0.7 * (rng.uniform(size=N) < 0.10) * rng.uniform(0, 1, N)
        # colour by where the voice comes from: marbled bands that the spiral winds together
        psi = th0 + K_SPIRAL * np.log(r0)          # constant along a thread, shared by its neighbours
        cp = 0.5 + 0.5 * np.sin(2 * psi + 0.5 * np.sin(r0 * 0.4) + rng.normal(0, 0.25, N))
        cp = np.clip(cp * 0.94 + 0.03 * rng.uniform(size=N), 0, 0.999) * (len(PAL) - 1)
        tint = PAL[cp.astype(int)] * (1 - cp % 1)[:, None] + PAL[np.minimum(cp.astype(int) + 1, 6)] * (cp % 1)[:, None]
        s = np.linspace(0, 1, M1)[None, :]
        rc = 0.30 + 0.34 * np.sqrt(rng.uniform(0, 1, N))[:, None]
        rr = rc + (r0[:, None] - rc) * (1 - s) ** 1.35
        th = th0[:, None] + K_SPIRAL * np.log(r0[:, None] / rr)
        wild = (rr / 18.0) ** 1.4
        ph1, ph2 = rng.uniform(0, 6.3, (2, N))
        xx = rr * np.cos(th) + 0.9 * wild * np.sin(0.45 * rr * np.sin(th) + ph1[:, None])
        yy = rr * np.sin(th) + 0.9 * wild * np.cos(0.45 * rr * np.cos(th) + ph2[:, None])
        zz = z0[:, None] * (1 - s) ** 1.5 + 0.02
        spiral = np.stack([xx, yy, zz], -1)
        th_end = th[:, -1]
        rc = rc[:, 0]

        # assign every voice to a leaf of the tree; the chosen leaf draws more
        leaves = tree.leaves()
        # white at the trunk, a spectrum in the crown: leaves are ordered around
        # the tree, voices by colour, and each leaf takes its share of the spectrum
        az = np.array([math.atan2(lf.curve[-1][1], lf.curve[-1][0]) for lf in leaves])
        order = np.argsort(az)
        wts = np.array([4.0 if leaves[i].chosen else 1.0 for i in order])
        cum = np.cumsum(wts) / wts.sum()
        rank = np.empty(N)
        rank[np.argsort(cp + 0.15 * rng.normal(size=N))] = (np.arange(N) + 0.5) / N
        pick = order[np.minimum(np.searchsorted(cum, rank), len(leaves) - 1)]
        for i in range(N):
            b = leaves[pick[i]]
            while b is not None:
                b.threads.append(i)
                b = b.parent
        for b in tree.walk():
            b.R = 0.0088 * math.sqrt(max(len(b.threads), 1))
            b.twist = rng.uniform(1.5, 3.0) * rng.choice([-1, 1])
        slot = {}                                   # (branch id, thread) -> (rho, phi)

        pos = np.zeros((N, self.M, 3), np.float32)
        ell = np.full((N, self.M), -1.0, np.float32)
        chosen = np.zeros((N, self.M), np.float32)
        fray = np.zeros((N, self.M), np.float32)
        dens = np.ones((N, self.M), np.float32)
        pos[:, :M1] = spiral
        for i in range(N):
            chain = []
            b = leaves[pick[i]]
            while b is not None:
                chain.append(b)
                b = b.parent
            chain = chain[::-1]
            P, E, Ch, Dn = [], [], [], []
            prev_off = np.array([rc[i] * math.cos(th_end[i]), rc[i] * math.sin(th_end[i]), 0.0])
            for bi, b in enumerate(chain):
                key = (id(b), i)
                if key not in slot:
                    slot[key] = (math.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi))
                rho, phi = slot[key]
                uu = np.linspace(0, 1, len(b.curve))
                if bi == 0:                     # the trunk: arriving threads gather into it
                    rad = rc[i] * (1 - smoothstep(0, 0.45, uu)) + b.R * rho * smoothstep(0, 0.45, uu)
                    rad = rad * (1 + 0.35 * np.exp(-uu * 6))
                    ang = th_end[i] + b.twist * uu
                    off = rad[:, None] * (np.cos(ang)[:, None] * np.array([1.0, 0, 0]) + np.sin(ang)[:, None] * np.array([0, 1.0, 0]))
                else:
                    rad = b.R * rho * (1 - 0.2 * uu)
                    ang = phi + b.twist * uu
                    off = rad[:, None] * (np.cos(ang)[:, None] * b.E1 + np.sin(ang)[:, None] * b.E2)
                    blend = smoothstep(0, 0.28, uu)[:, None]
                    off = prev_off[None] * (1 - blend) + off * blend
                prev_off = off[-1]
                P.append(b.curve + off)
                E.append(b.ell)
                Ch.append(np.full(len(uu), 1.0 if b.chosen else 0.0))
                Dn.append(np.full(len(uu), float(len(b.threads))))
            leaf = chain[-1]
            if not leaf.chosen:                 # the unsaid fray into possibilities
                Lf = rng.uniform(0.35, 1.3)
                d = nrm(leaf.dir + rng.normal(0, 0.55, 3) + 0.25 * UP)
                q = np.linspace(0, 1, 14)[1:]
                bend = nrm(d + 0.5 * UP)
                fr = P[-1][-1] + (q[:, None] * d + (q ** 2)[:, None] * (bend - d) * 0.6) * Lf
                P.append(fr)
                E.append(leaf.ell[-1] + q * Lf)
                Ch.append(np.zeros(len(q)))
                Dn.append(np.linspace(float(len(leaf.threads)), 6.0, len(q)))
                fr_mask = np.concatenate([np.zeros(sum(len(p) for p in P[:-1])), q])
            else:
                fr_mask = np.zeros(sum(len(p) for p in P))
            P = np.concatenate(P)
            E = np.concatenate(E)
            Ch = np.concatenate(Ch)
            Dn = np.concatenate(Dn)
            # resample the tree part to M2 points by arc length
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            cs = np.concatenate([[0], np.cumsum(seg)])
            tq = np.linspace(0, cs[-1], M2)
            pos[i, M1:] = np.stack([np.interp(tq, cs, P[:, k]) for k in range(3)], 1)
            ell[i, M1:] = np.interp(tq, cs, E)
            chosen[i, M1:] = np.interp(tq, cs, Ch) > 0.5
            fray[i, M1:] = np.interp(tq, cs, fr_mask)
            dens[i, M1:] = np.interp(tq, cs, Dn)
        self.FULL = pos
        # arrival times: when the front reaches each point
        seg = np.linalg.norm(np.diff(spiral, axis=1), axis=2)
        cs = np.concatenate([np.zeros((N, 1)), np.cumsum(seg, axis=1)], 1)
        q = cs / cs[:, -1:]
        birth = T_DROP + 0.35 + r0 / RIPPLE_C + rng.uniform(0, 0.5, N)
        rate = rng.uniform(0.13, 0.21, N)
        t_center = birth + 1.0 / rate
        T_arr = np.zeros((N, self.M))
        T_arr[:, :M1] = birth[:, None] + q / rate[:, None]
        e_t = ell[:, M1:]
        lag = rng.gamma(2.0, 0.45, N)[:, None]
        T_arr[:, M1:] = np.maximum(t_center[:, None] + e_t / V_FIBER, T_TREE + lag + e_t / V_TREE)
        T_arr[:, M1:] = np.maximum.accumulate(np.maximum(T_arr[:, M1:], t_center[:, None] + 1e-3), axis=1)
        self.T_arr = T_arr
        self.radius = (0.0040 + 0.0055 * rng.uniform(0, 1, N) ** 2)[:, None] * np.ones((1, self.M))
        tree_part = np.zeros((N, self.M), bool)
        tree_part[:, M1:] = True
        self.radius *= np.where(tree_part, 0.75, 1.0) * (1 - 0.6 * fray)
        dim_order = np.clip(1 - (r0 - 1.8) / 16.2 + 0.15 * rng.normal(size=N), 0, 1)
        self.dim_order = dim_order
        self.chosen_any = chosen.max(1)

        mat, nd = C.material("voices")
        t = C.clock(nd)
        ans = uniform(nd, "u_answer", -10.0)
        after = uniform(nd, "u_after", 0.0)
        coda = uniform(nd, "u_coda", -100.0)
        hi = nd.add("ShaderNodeHairInfo")
        I = hi.outputs["Intercept"]

        def attr(name):
            return nd.add("ShaderNodeAttribute", _attribute_name=name, _attribute_type="GEOMETRY")
        a_tint, a_phase, a_front = attr("tint"), attr("phase"), attr("front")
        a_ell, a_ch, a_fray, a_dim = attr("ell"), attr("chosen"), attr("fray"), attr("dim_order")
        a_alpha = attr("alpha")
        a_opacity = attr("opacity")
        a_carrier = attr("carrier")
        p = nd.math("FRACT", nd.math("ADD", nd.math("SUBTRACT", nd.math("MULTIPLY", I, 6.0), nd.math("MULTIPLY", t, 0.5)),
                                    a_phase.outputs["Fac"]))
        band = nd.math("MULTIPLY", nd.smooth(p, 0.0, 0.08), nd.smooth(nd.math("SUBTRACT", 1.0, p), 0.6, 0.92))
        body = nd.math("ADD", 0.16, nd.math("MULTIPLY", band, 0.55))
        head = nd.math("EXPONENT", nd.math("MULTIPLY", nd.math("POWER", nd.math("SUBTRACT", I, a_front.outputs["Fac"]), 2.0), -3000.0))
        growing = nd.math("LESS_THAN", a_front.outputs["Fac"], 0.999)
        in_tree = nd.math("GREATER_THAN", a_ell.outputs["Fac"], -0.5)
        tree_dim = nd.math("SUBTRACT", 1.0, nd.math("MULTIPLY", in_tree, 0.35))
        fray_fade = nd.math("SUBTRACT", 1.0, nd.math("MULTIPLY", a_fray.outputs["Fac"], 0.85))
        strength = nd.math("MULTIPLY", nd.math("MULTIPLY", body, tree_dim), fray_fade)
        strength = nd.math("ADD", strength, nd.math("MULTIPLY", nd.math("MULTIPLY", head, growing), 6.0))
        # the answer: the chosen path fills with warm light up to where it has reached
        e = a_ell.outputs["Fac"]
        lit = nd.math("MULTIPLY", a_ch.outputs["Fac"], nd.math("LESS_THAN", e, ans))
        ahead = nd.math("EXPONENT", nd.math("MULTIPLY", nd.math("POWER", nd.math("SUBTRACT", e, ans), 2.0), -40.0))
        flow = nd.math("MULTIPLY", nd.math("ADD", nd.math("SINE", nd.math("SUBTRACT", nd.math("MULTIPLY", e, 9.0),
                                                                         nd.math("MULTIPLY", t, 7.0))), 1.0), 0.18)
        ans_glow = nd.math("MULTIPLY", lit, nd.math("ADD", nd.math("ADD", 0.30, flow), nd.math("MULTIPLY", ahead, 2.2)))
        # a few strands carry the answer brightly, the rest only faintly
        ans_glow = nd.math("MULTIPLY", ans_glow, nd.math("ADD", 0.12, nd.math("MULTIPLY", a_carrier.outputs["Fac"], 2.4)))
        # once it is said, the light has gone on, into the glass
        ans_glow = nd.math("MULTIPLY", ans_glow, nd.math("SUBTRACT", 1.0, nd.math("MULTIPLY", after, 0.7)))
        # the coda: outer voices first, the answer last
        order = nd.math("ADD", nd.math("MULTIPLY", a_dim.outputs["Fac"], 7.0), nd.math("MULTIPLY", a_ch.outputs["Fac"], 3.0))
        alive = nd.math("SUBTRACT", 1.0, nd.smooth(nd.math("SUBTRACT", coda, order), 0.0, 2.5))
        white = in_tree
        base_col = nd.mix_rgb(nd.math("MULTIPLY", white, 0.10), a_tint.outputs["Color"], (0.92, 0.90, 1.0, 1.0))
        col = nd.mix_rgb(nd.math("MINIMUM", nd.math("MULTIPLY", ans_glow, 3.0), 1.0), base_col, (1.0, 0.60, 0.24, 1.0))
        tree_k = nd.math("SUBTRACT", 1.0, nd.math("MULTIPLY", in_tree, 0.20))
        total = nd.math("MULTIPLY", nd.math("ADD", nd.math("MULTIPLY", nd.math("MULTIPLY", strength, tree_k), 2.2),
                                            nd.math("MULTIPLY", ans_glow, 2.0)), alive)
        em = nd.add("ShaderNodeEmission")
        nd.l.new(col, em.inputs["Color"])
        nd.l.new(total, em.inputs["Strength"])
        tr = nd.add("ShaderNodeBsdfTransparent")
        mix = nd.add("ShaderNodeMixShader")
        tree_alpha = nd.math("MULTIPLY", a_opacity.outputs["Fac"], nd.math("ADD", 0.5, nd.math("MULTIPLY", a_alpha.outputs["Fac"], 4.0)))
        opac = nd.math("ADD", nd.math("MULTIPLY", in_tree, nd.math("SUBTRACT", tree_alpha, 1.0)), 1.0)
        opac = nd.math("MULTIPLY", opac, nd.math("SUBTRACT", 1.0, nd.math("MULTIPLY", lit, 0.45)))
        nd.l.new(nd.math("MULTIPLY", opac, alive), mix.inputs["Fac"])
        nd.l.new(tr.outputs[0], mix.inputs[1])
        nd.l.new(em.outputs[0], mix.inputs[2])
        out = nd.add("ShaderNodeOutputMaterial")
        nd.l.new(mix.outputs[0], out.inputs["Surface"])
        self.voices = C.hair("voices", [pos[i] for i in range(N)], [self.radius[i] for i in range(N)],
                             curve_attrs={"tint": tint, "phase": rng.uniform(0, 1, N), "front": np.zeros(N),
                                          "dim_order": dim_order, "alpha": 0.07 + 0.15 * rng.uniform(0, 1, N) ** 2,
                                          "carrier": rng.uniform(0, 1, N) ** 4},
                             point_attrs={"ell": ell.ravel(), "chosen": chosen.ravel(), "fray": fray.ravel(),
                                          "opacity": np.clip(1.0 / np.sqrt(dens), 0.018, 0.6).ravel()},
                             material_=mat)
        self.hidden_pos = pos[:, :1].copy()

    def update_threads(self, t):
        N, M = self.N, self.M
        arr = self.T_arr
        k = (arr <= t).sum(1)                                   # points already reached
        kk = np.clip(k, 1, M - 1)
        a0 = np.take_along_axis(arr, (kk - 1)[:, None], 1)[:, 0]
        a1 = np.take_along_axis(arr, kk[:, None], 1)[:, 0]
        frac = np.clip((t - a0) / np.maximum(a1 - a0, 1e-6), 0, 1)
        fidx = np.where(k >= M, M - 1.0, np.where(k == 0, 0.0, kk - 1 + frac))
        idx = np.minimum(np.arange(M)[None, :], fidx[:, None])
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, M - 1)
        w = (idx - i0)[..., None]
        rows = np.arange(N)[:, None]
        P = self.FULL[rows, i0] * (1 - w) + self.FULL[rows, i1] * w
        unborn = k == 0
        if t > T_CODA:
            gone = (t - T_CODA) > (self.dim_order * 7.0 + self.chosen_any * 3.0 + 2.6)
            unborn = unborn | gone
        # voices not (or no longer) in the world wait far below the water,
        # spread along their own paths: piled onto one point they would make
        # a knot no ray-tracing acceleration structure can untie
        P[unborn] = self.FULL[unborn] - np.array([0.0, 0.0, 100.0], np.float32)
        cv = self.voices.data
        cv.position_data.foreach_set("vector", P.astype(np.float32).ravel())
        cv.attributes["front"].data.foreach_set("value", (fidx / (M - 1)).astype(np.float32))
        rad = np.where(unborn[:, None], 0.0, self.radius).astype(np.float32)
        cv.attributes["radius"].data.foreach_set("value", rad.ravel())
        cv.update_tag()

    def path_point(self, e):
        """A point on the chosen path's centreline at tree arc length e."""
        if not hasattr(self, "_path"):
            chain = [b for b in self.tree.walk() if b.chosen]
            self._path = (np.concatenate([b.curve for b in chain]), np.concatenate([b.ell for b in chain]))
        P, E = self._path
        e = float(np.clip(e, E[0], E[-1]))
        return np.array([np.interp(e, E, P[:, k]) for k in range(3)])

    # ---- the answer ------------------------------------------------------------
    def answer_schedule(self):
        """(ell, time) pairs for the answer's light along the chosen path."""
        chain = [b for b in self.tree.walk() if b.chosen]
        keys = [(0.0, T_ANSWER)]
        t = T_ANSWER
        self.word_times = {}
        for b in chain:
            t += b.len / V_ANSWER
            keys.append((b.ell[-1], t))
            self.word_times[b] = t
            t += PAUSE
            keys.append((b.ell[-1], t))
        self.t_sphere = keys[-1][1]
        return keys

    def answer_at(self, t):
        if not hasattr(self, "_akeys"):
            self._akeys = self.answer_schedule()
        e = [k[0] for k in self._akeys]
        ts = [k[1] for k in self._akeys]
        if t <= ts[0]:
            return -10.0
        return float(np.interp(t, ts, e))

    # ---- words ---------------------------------------------------------------
    def build_words(self):
        rng = np.random.default_rng(3)
        mat, nd = C.material("words")
        oi = nd.add("ShaderNodeObjectInfo")
        em = nd.add("ShaderNodeEmission")
        nd.l.new(oi.outputs["Color"], em.inputs["Color"])
        nd.l.new(nd.math("MULTIPLY", oi.outputs["Alpha"], 2.2), em.inputs["Strength"])
        out = nd.add("ShaderNodeOutputMaterial")
        nd.l.new(em.outputs[0], out.inputs["Surface"])
        self.word_mat = mat
        drift = [("why", FONT_ITALIC), ("pourquoi", FONT_ITALIC), ("warum", FONT_ITALIC), ("¿por qué?", FONT_ITALIC),
                 ("perché", FONT_ITALIC), ("почему", FONT_ITALIC), ("γιατί", FONT_ITALIC), ("为什么", FONT_CJK),
                 ("なぜ", FONT_CJK), ("왜", FONT_CJK), ("kwa nini", FONT_ITALIC), ("neden", FONT_ITALIC),
                 ("dlaczego", FONT_ITALIC), ("varför", FONT_ITALIC), ("kenapa", FONT_ITALIC), ("tại sao", FONT_ITALIC),
                 ("waarom", FONT_ITALIC), ("zašto", FONT_ITALIC), ("miért", FONT_ITALIC), ("what if", FONT_ITALIC),
                 ("once upon a time", FONT_ITALIC), ("dear friend,", FONT_ITALIC), ("cogito", FONT_ITALIC),
                 ("tell me", FONT_ITALIC), ("I wonder", FONT_ITALIC), ("and yet", FONT_ITALIC), ("because", FONT_ITALIC),
                 ("どうして", FONT_CJK), ("how?", FONT_ITALIC), ("remember", FONT_ITALIC)]
        self.drift = []
        for i, (w, f) in enumerate(drift):
            ob = C.text(w, f, 0.15 + 0.07 * rng.uniform(), mat, name=f"drift{i}")
            self.drift.append(dict(ob=ob, r=rng.uniform(4.0, 13.0), th=rng.uniform(0, 2 * np.pi),
                                   z=rng.uniform(0.25, 1.1), t0=rng.uniform(8.5, 19.0), life=rng.uniform(6.0, 9.0)))
        # the tree's words, at the branches they name
        self.tree_words = []
        for b in self.tree.walk():
            if b.parent is None:
                continue
            size = (0.30 if b.chosen else 0.17) * (1.0 if b.depth < 4 else 0.9)
            ob = C.text(b.word, FONT_ITALIC, size, mat, name=f"tw_{b.word}_{id(b) % 997}")
            j = int(len(b.curve) * (0.55 if b.kids else 0.75))
            side = nrm(np.cross(b.T[j], np.array([0.2, -0.9, 0.3])))
            anchor = b.curve[j] + side * (0.18 if b.chosen else 0.12) + np.array([0, 0, 0.06])
            self.tree_words.append(dict(ob=ob, b=b, anchor=anchor, t_grow=T_TREE + b.ell[j] / V_TREE))
        ob = C.text("I", FONT_ITALIC, 0.42, mat, name="tw_I")
        self.tree_words.append(dict(ob=ob, b=self.tree, anchor=np.array([0.62, -0.25, 1.35]), t_grow=T_TREE + 1.0))

    def update_words(self, t, cam):
        rot = cam.rotation_euler.copy()
        K = K_SPIRAL
        for wd in self.drift:
            ob = wd["ob"]
            age = t - wd["t0"]
            if age < 0 or age > wd["life"]:
                ob.hide_render = True
                continue
            ob.hide_render = False
            r = wd["r"] * math.exp(-0.06 * age)
            th = wd["th"] + K * math.log(wd["r"] / r)
            ob.location = (r * math.cos(th), r * math.sin(th), wd["z"])
            ob.rotation_euler = rot
            a = float(smoothstep(0, 1.6, age) * smoothstep(0, 2.0, wd["life"] - age))
            ob.color = (0.95, 0.90, 0.82, a * 0.5)
        coda = t - T_CODA
        for wd in self.tree_words:
            ob, b = wd["ob"], wd["b"]
            if b.chosen:
                t_on = self.word_times[b] if hasattr(self, "word_times") else 1e9
                a = float(smoothstep(t_on - 0.1, t_on + 0.5, t))
                col = (1.0, 0.80, 0.52)
                a *= 1.5
                a *= float(1 - smoothstep(9.0, 11.5, coda))
            else:
                a = float(smoothstep(wd["t_grow"], wd["t_grow"] + 1.2, t)) * 0.45
                col = (0.80, 0.84, 0.95)
                # the unsaid words fade once the sentence is said
                a *= float(1 - 0.55 * smoothstep(T_ANSWER + 4, T_ANSWER + 9, t))
                a *= float(1 - smoothstep(0.0, 4.0, coda))
            ob.hide_render = a <= 0.002
            ob.location = tuple(wd["anchor"])
            ob.rotation_euler = rot
            ob.color = (*col, a)

    # ---- the sphere ------------------------------------------------------------
    def build_sphere(self):
        tip_branch = next(b for b in self.tree.walk() if b.chosen and not b.kids)
        self.sphere_c = tip_branch.curve[-1] + tip_branch.dir * 0.52
        mat, nd = C.material("glass")
        g = nd.add("ShaderNodeBsdfGlass", Roughness=0.0, IOR=1.52)
        g.inputs["Color"].default_value = (0.97, 0.99, 0.98, 1)
        out = nd.add("ShaderNodeOutputMaterial")
        nd.l.new(g.outputs[0], out.inputs["Surface"])
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=96, ring_count=48, location=tuple(self.sphere_c))
        bpy.ops.object.shade_smooth()
        self.sphere = bpy.context.object
        self.sphere.data.materials.append(mat)
        m2, nd = C.material("seed")
        self.seed_em = nd.add("ShaderNodeEmission", Color=(1.0, 0.86, 0.62, 1), Strength=0.0)
        out = nd.add("ShaderNodeOutputMaterial")
        nd.l.new(self.seed_em.outputs[0], out.inputs["Surface"])
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.06, segments=24, ring_count=12, location=tuple(self.sphere_c))
        self.seed = bpy.context.object
        self.seed.data.materials.append(m2)
        self.sphere_r = 0.56

    def update_sphere(self, t):
        ts = getattr(self, "t_sphere", 1e9)
        g = float(smoothstep(ts, ts + 2.2, t))
        g = g * (1 + 0.12 * math.sin(math.pi * g) ** 2)           # it swells, and settles
        r = self.sphere_r * g
        self.sphere.hide_render = r < 1e-3
        self.sphere.scale = (max(r, 1e-4),) * 3
        flash = math.exp(-((t - ts) / 0.35) ** 2) * 180.0 if t > ts - 1 else 0.0
        glow = float(smoothstep(ts - 0.4, ts, t)) * math.exp(-max(t - ts, 0) / 0.8) * 25.0 * float(1 - smoothstep(ts + 1.5, ts + 2.5, t))
        self.seed_em.inputs["Strength"].default_value = flash + glow
        self.seed.hide_render = (flash + glow) < 0.05

    # ---- motes -----------------------------------------------------------------
    def build_motes(self, n=420):
        rng = np.random.default_rng(21)
        self.m_n = n
        self.m_p0 = np.stack([rng.uniform(-6, 6, n), rng.uniform(-6, 6, n), rng.uniform(0.2, 12, n)], 1)
        self.m_v = rng.uniform(0.04, 0.16, n)
        self.m_ph = rng.uniform(0, 2 * np.pi, n)
        mat, nd = C.material("motes")
        at = nd.add("ShaderNodeAttribute", _attribute_name="twinkle", _attribute_type="GEOMETRY")
        em = nd.add("ShaderNodeEmission", Color=(1.0, 0.85, 0.6, 1))
        nd.l.new(nd.math("MULTIPLY", at.outputs["Fac"], 4.0), em.inputs["Strength"])
        out = nd.add("ShaderNodeOutputMaterial")
        nd.l.new(em.outputs[0], out.inputs["Surface"])
        pts = [np.stack([p, p + np.array([0, 0, 0.02])]) for p in self.m_p0]
        self.motes = C.hair("motes", pts, 0.006, curve_attrs={"twinkle": np.zeros(n)}, material_=mat)

    def update_motes(self, t):
        a = float(smoothstep(T_TREE + 2, T_TREE + 8, t) * (1 - smoothstep(T_CODA + 2, T_CODA + 8, t)))
        z = (self.m_p0[:, 2] + self.m_v * t) % 12.5
        x = self.m_p0[:, 0] + 0.25 * np.sin(0.3 * t + self.m_ph)
        y = self.m_p0[:, 1] + 0.25 * np.cos(0.27 * t + self.m_ph)
        p = np.stack([x, y, z], 1)
        pts = np.stack([p, p + np.array([0, 0, 0.03])], 1)
        cv = self.motes.data
        cv.position_data.foreach_set("vector", pts.astype(np.float32).ravel())
        tw = a * (0.25 + 0.75 * (0.5 + 0.5 * np.sin(2.3 * t + self.m_ph * 3))) * np.exp(-np.hypot(x, y) / 5)
        cv.attributes["twinkle"].data.foreach_set("value", tw.astype(np.float32))
        cv.update_tag()
        self.motes.hide_render = a <= 0

    # ---- everything ------------------------------------------------------------
    def update(self, t, cam):
        z = self.spark_z(t)
        self.spark.location = (0, 0, z + 0.03)
        flash = math.exp(-((t - T_DROP) / 0.10) ** 2) * 300.0
        base = 45.0 if t < T_DROP else 6.0 + 20.0 * math.exp(-(t - T_DROP) / 1.5)
        fade = float(1 - smoothstep(T_BLACK - 3.0, T_BLACK - 0.2, t))
        self.spark_em.inputs["Strength"].default_value = (base + flash) * fade
        vz = -(self.spark_z(t + 0.02) - self.spark_z(t - 0.02)) / 0.04
        length = min(0.9, 0.12 * vz)
        self.trail.hide_render = not (0.3 < t < T_DROP) or length < 0.02
        self.trail.scale = (1.0, 1.0, max(length, 0.02))
        self.trail.location = (0, 0, z + 0.03 + 0.5 * max(length, 0.02))
        self.update_threads(t)
        self.voices.rotation_euler = (0, 0, -0.010 * max(t - T_DROP, 0) + 0.004 * max(t - T_CODA, 0) ** 2 * 0)
        set_uniform("u_answer", self.answer_at(t))
        set_uniform("u_after", float(smoothstep(self.t_sphere, self.t_sphere + 3.0, t)))
        set_uniform("u_coda", t - T_CODA)
        set_uniform("u_calm", float(smoothstep(T_CODA, T_CODA + 8, t)))
        self.update_words(t, cam)
        self.update_sphere(t)
        self.update_motes(t)
