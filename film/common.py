"""
Shared Blender (bpy) helpers for the film: scene setup, light-thread curves,
emission shaders with a clock, text, a smooth camera, and a resumable
frame-by-frame render loop (every frame is set up from Python, so anything
can be animated procedurally).
"""
import os
import time

import bpy
import numpy as np

FPS = 24
W, H = 1920, 816                       # 2.35:1


def reset(samples=12, width=W, height=H, denoise=False, bounces=2):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.render.resolution_x, sc.render.resolution_y = width, height
    sc.render.resolution_percentage = 100
    sc.render.use_persistent_data = True
    sc.render.fps = FPS
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.03
    sc.cycles.use_denoising = denoise
    sc.cycles.max_bounces = bounces
    sc.cycles.diffuse_bounces = min(bounces, 1)
    sc.cycles.glossy_bounces = bounces
    sc.cycles.transmission_bounces = max(bounces, 4)
    sc.cycles.transparent_max_bounces = 14
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles_curves.shape = "RIBBONS"
    sc.render.film_transparent = False
    im = sc.render.image_settings
    im.file_format = "OPEN_EXR"
    im.color_depth = "16"
    im.exr_codec = "DWAA"
    im.color_mode = "RGB"
    w = bpy.data.worlds.new("world")
    sc.world = w
    w.use_nodes = True
    w.cycles.sampling_method = "NONE"
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    return sc


# ---------------------------------------------------------------------------
# node helpers
# ---------------------------------------------------------------------------
class Nodes:
    def __init__(self, tree):
        self.t = tree
        self.n = tree.nodes
        self.l = tree.links

    def add(self, kind, **inputs):
        node = self.n.new(kind)
        for k, v in inputs.items():
            if k.startswith("_"):
                setattr(node, k[1:], v)
            else:
                self.set(node, k, v)
        return node

    def set(self, node, key, v):
        sock = node.inputs[key]
        if isinstance(v, bpy.types.NodeSocket):
            self.l.new(v, sock)
        else:
            sock.default_value = v

    def math(self, op, a, b=None, clamp=False):
        node = self.n.new("ShaderNodeMath")
        node.operation = op
        node.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, bpy.types.NodeSocket):
                self.l.new(v, node.inputs[i])
            else:
                node.inputs[i].default_value = v
        return node.outputs[0]

    def smooth(self, x, a, b):
        """smoothstep(a, b, x)"""
        node = self.n.new("ShaderNodeMapRange")
        node.interpolation_type = "SMOOTHSTEP"
        node.clamp = True
        self.l.new(x, node.inputs["Value"])
        node.inputs["From Min"].default_value = a
        node.inputs["From Max"].default_value = b
        node.inputs["To Min"].default_value = 0.0
        node.inputs["To Max"].default_value = 1.0
        return node.outputs["Result"]

    def mix_rgb(self, fac, a, b):
        node = self.n.new("ShaderNodeMix")
        node.data_type = "RGBA"
        self.l.new(fac, node.inputs[0]) if isinstance(fac, bpy.types.NodeSocket) else None
        if not isinstance(fac, bpy.types.NodeSocket):
            node.inputs[0].default_value = fac
        for idx, v in ((6, a), (7, b)):
            if isinstance(v, bpy.types.NodeSocket):
                self.l.new(v, node.inputs[idx])
            else:
                node.inputs[idx].default_value = v
        return node.outputs[2]

    def scale_color(self, col, s):
        node = self.n.new("ShaderNodeVectorMath")
        node.operation = "SCALE"
        self.l.new(col, node.inputs[0])
        if isinstance(s, bpy.types.NodeSocket):
            self.l.new(s, node.inputs["Scale"])
        else:
            node.inputs["Scale"].default_value = s
        return node.outputs[0]


def material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes.clear()
    # nothing in this world is lit diffusely: emitters need not be sampled as lights
    m.cycles.emission_sampling = "NONE"
    return m, Nodes(m.node_tree)


def clock(nd, value=0.0):
    """A Value node the render loop sets to the current time (seconds)."""
    v = nd.add("ShaderNodeValue")
    v.name = "clock"
    v.label = "clock"
    v.outputs[0].default_value = value
    return v.outputs[0]


def set_clocks(t):
    for m in bpy.data.materials:
        if m.node_tree is None:
            continue
        node = m.node_tree.nodes.get("clock")
        if node is not None:
            node.outputs[0].default_value = t
    w = bpy.context.scene.world
    if w and w.node_tree and w.node_tree.nodes.get("clock"):
        w.node_tree.nodes["clock"].outputs[0].default_value = t


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def hair(name, curves, radius, curve_attrs=None, point_attrs=None, material_=None):
    """curves: list of (m_i, 3) arrays. radius: scalar or list of (m_i,) arrays."""
    cv = bpy.data.hair_curves.new(name)
    sizes = [len(c) for c in curves]
    cv.add_curves(sizes)
    pos = np.concatenate(curves).astype(np.float32)
    cv.position_data.foreach_set("vector", pos.ravel())
    if np.isscalar(radius):
        r = np.full(len(pos), radius, np.float32)
    else:
        r = np.concatenate(radius).astype(np.float32)
    rad = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    rad.data.foreach_set("value", r)
    for key, arr in (curve_attrs or {}).items():
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 2:
            a = cv.attributes.new(key, "FLOAT_COLOR", "CURVE")
            if arr.shape[1] == 3:
                arr = np.concatenate([arr, np.ones((len(arr), 1), np.float32)], 1)
            a.data.foreach_set("color", arr.ravel())
        else:
            a = cv.attributes.new(key, "FLOAT", "CURVE")
            a.data.foreach_set("value", arr)
    for key, arr in (point_attrs or {}).items():
        arr = np.asarray(arr, np.float32)
        a = cv.attributes.new(key, "FLOAT", "POINT")
        a.data.foreach_set("value", arr)
    ob = bpy.data.objects.new(name, cv)
    bpy.context.scene.collection.objects.link(ob)
    if material_ is not None:
        cv.materials.append(material_)
    return ob


def text(body, font_path, size, material_=None, align="CENTER", extrude=0.0, name=None):
    fcurve = bpy.data.curves.new(name or ("text_" + body[:12]), type="FONT")
    fcurve.body = body
    fcurve.font = bpy.data.fonts.load(font_path, check_existing=True)
    fcurve.size = size
    fcurve.align_x = align
    fcurve.align_y = "CENTER"
    fcurve.extrude = extrude
    ob = bpy.data.objects.new(name or ("text_" + body[:12]), fcurve)
    bpy.context.scene.collection.objects.link(ob)
    if material_ is not None:
        fcurve.materials.append(material_)
    return ob


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------
def make_camera(lens=35.0, name="camera"):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.clip_start = 0.01
    cam.clip_end = 2000
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    return ob


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_io(t, p=2.2):
    t = min(max(t, 0.0), 1.0)
    return t ** p / (t ** p + (1 - t) ** p)


def catmull(points, u):
    """Centripetal-ish Catmull-Rom through key points, u in [0, 1]."""
    P = np.asarray(points, float)
    n = len(P) - 1
    x = min(max(u, 0.0), 1.0) * n
    i = min(int(x), n - 1)
    t = x - i
    p0 = P[max(i - 1, 0)]
    p1 = P[i]
    p2 = P[i + 1]
    p3 = P[min(i + 2, n)]
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def look_at(ob, target, roll=0.0):
    loc = np.asarray(ob.location, float)
    d = np.asarray(target, float) - loc
    d /= np.linalg.norm(d)
    # Blender cameras look down -Z with +Y up
    import mathutils
    rot = mathutils.Vector(d).to_track_quat("-Z", "Y").to_euler()
    ob.rotation_euler = rot
    if roll:
        ob.rotation_euler.rotate_axis("Z", roll)


class Path:
    """Camera keyframes (time, location, target, lens) with smooth interpolation."""

    def __init__(self, keys):
        self.keys = keys

    def at(self, t):
        ts = [k[0] for k in self.keys]
        if t <= ts[0]:
            k = self.keys[0]
            return np.array(k[1]), np.array(k[2]), k[3]
        if t >= ts[-1]:
            k = self.keys[-1]
            return np.array(k[1]), np.array(k[2]), k[3]
        i = max(j for j in range(len(ts)) if ts[j] <= t)
        a, b = self.keys[i], self.keys[i + 1]
        f = ease_io((t - a[0]) / (b[0] - a[0]), 1.6)
        # position through a spline of all keys, parameterised per segment
        locs = [k[1] for k in self.keys]
        tgts = [k[2] for k in self.keys]
        u = (i + f) / (len(self.keys) - 1)
        loc = catmull(locs, u)
        tgt = catmull(tgts, u)
        lens = a[3] + (b[3] - a[3]) * f
        return loc, tgt, lens


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_frames(outdir, frames, update, log_every=10):
    """Render the given frame numbers (resumable: existing files are skipped)."""
    os.makedirs(outdir, exist_ok=True)
    sc = bpy.context.scene
    t_start = time.time()
    done = 0
    for f in frames:
        path = os.path.join(outdir, f"{f:05d}.exr")
        if os.path.exists(path):
            continue
        t = f / FPS
        update(t, f)
        set_clocks(t)
        sc.render.filepath = path
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        done += 1
        if done % log_every == 1:
            el = time.time() - t_start
            print(f"frame {f} ({time.time() - t0:.1f}s, {el / done:.1f}s avg)", flush=True)
