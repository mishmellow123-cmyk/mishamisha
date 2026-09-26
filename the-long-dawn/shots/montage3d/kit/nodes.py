"""Tiny DSL for building Blender shader / geometry node trees from Python.

    nb = NB(mat.node_tree)
    tc = nb.n('ShaderNodeTexCoord')
    x = nb.math('MULTIPLY', nb.sep(tc.outputs['Object'])[0], 2.0)
    nb.link(x, some_node.inputs['Scale'])

Values passed where a socket is expected may be: a NodeSocket, a Node (its first output), a float,
or a tuple (vector / colour).
"""
import bpy


class NB:
    def __init__(self, nt, clear=False):
        self.nt = nt
        if clear:
            for n in list(nt.nodes):
                nt.nodes.remove(n)

    # ------------------------------------------------------------ basics ---
    def n(self, typ, inputs=None, **props):
        nd = self.nt.nodes.new(typ)
        for k, v in props.items():
            setattr(nd, k, v)
        if inputs:
            for k, v in inputs.items():
                self.set(nd.inputs[k], v)
        return nd

    def set(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        elif isinstance(v, bpy.types.Node):
            self.nt.links.new(v.outputs[0], sock)
        else:
            if isinstance(v, (tuple, list)):
                dv = sock.default_value
                n = len(dv)
                v = tuple(v) + (1.0,) * (n - len(v)) if len(v) < n else tuple(v)[:n]
            sock.default_value = v

    def link(self, a, b):
        if isinstance(a, bpy.types.Node):
            a = a.outputs[0]
        self.nt.links.new(a, b)

    @staticmethod
    def out(x, i=0):
        return x.outputs[i] if isinstance(x, bpy.types.Node) else x

    # ------------------------------------------------------------- math ---
    def math(self, op, a, b=None, c=None, clamp=False):
        nd = self.nt.nodes.new('ShaderNodeMath' if self._shader() else 'ShaderNodeMath')
        nd.operation = op
        nd.use_clamp = clamp
        self.set(nd.inputs[0], a)
        if b is not None:
            self.set(nd.inputs[1], b)
        if c is not None:
            self.set(nd.inputs[2], c)
        return nd.outputs[0]

    def add(self, a, b):
        return self.math('ADD', a, b)

    def sub(self, a, b):
        return self.math('SUBTRACT', a, b)

    def mul(self, a, b):
        return self.math('MULTIPLY', a, b)

    def div(self, a, b):
        return self.math('DIVIDE', a, b)

    def madd(self, a, b, c):
        return self.math('MULTIPLY_ADD', a, b, c)

    def exp(self, a):
        return self.math('EXPONENT', a)

    def pw(self, a, b):
        return self.math('POWER', a, b)

    def mn(self, a, b):
        return self.math('MINIMUM', a, b)

    def mx(self, a, b):
        return self.math('MAXIMUM', a, b)

    def clamp01(self, a):
        return self.math('MULTIPLY', a, 1.0, clamp=True)

    def sstep(self, e0, e1, x):
        """smoothstep via Map Range (SMOOTHSTEP)."""
        nd = self.nt.nodes.new('ShaderNodeMapRange')
        nd.interpolation_type = 'SMOOTHSTEP'
        self.set(nd.inputs['Value'], x)
        self.set(nd.inputs['From Min'], e0)
        self.set(nd.inputs['From Max'], e1)
        return nd.outputs['Result']

    def maprange(self, x, a0, a1, b0=0.0, b1=1.0, clamp=True, interp='LINEAR'):
        nd = self.nt.nodes.new('ShaderNodeMapRange')
        nd.interpolation_type = interp
        nd.clamp = clamp
        self.set(nd.inputs['Value'], x)
        self.set(nd.inputs['From Min'], a0)
        self.set(nd.inputs['From Max'], a1)
        self.set(nd.inputs['To Min'], b0)
        self.set(nd.inputs['To Max'], b1)
        return nd.outputs['Result']

    # ----------------------------------------------------------- vectors ---
    def vmath(self, op, a, b=None, c=None, scale=None):
        nd = self.nt.nodes.new('ShaderNodeVectorMath')
        nd.operation = op
        self.set(nd.inputs[0], a)
        if b is not None:
            self.set(nd.inputs[1], b)
        if c is not None:
            self.set(nd.inputs[2], c)
        if scale is not None:
            self.set(nd.inputs['Scale'], scale)
        if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE'):
            return nd.outputs['Value']
        return nd.outputs['Vector']

    def vadd(self, a, b):
        return self.vmath('ADD', a, b)

    def vsub(self, a, b):
        return self.vmath('SUBTRACT', a, b)

    def vmul(self, a, b):
        return self.vmath('MULTIPLY', a, b)

    def vscale(self, a, s):
        return self.vmath('SCALE', a, scale=s)

    def dot(self, a, b):
        return self.vmath('DOT_PRODUCT', a, b)

    def length(self, a):
        return self.vmath('LENGTH', a)

    def sep(self, v):
        nd = self.nt.nodes.new('ShaderNodeSeparateXYZ')
        self.set(nd.inputs[0], v)
        return nd.outputs[0], nd.outputs[1], nd.outputs[2]

    def comb(self, x, y, z):
        nd = self.nt.nodes.new('ShaderNodeCombineXYZ')
        self.set(nd.inputs[0], x)
        self.set(nd.inputs[1], y)
        self.set(nd.inputs[2], z)
        return nd.outputs[0]

    def rgb(self, r, g, b):
        nd = self.nt.nodes.new('ShaderNodeCombineColor')
        self.set(nd.inputs[0], r)
        self.set(nd.inputs[1], g)
        self.set(nd.inputs[2], b)
        return nd.outputs[0]

    def value(self, v, name=None):
        nd = self.nt.nodes.new('ShaderNodeValue')
        nd.outputs[0].default_value = v
        if name:
            nd.name = name
            nd.label = name
        return nd

    def mixcol(self, fac, a, b, blend='MIX'):
        nd = self.nt.nodes.new('ShaderNodeMix')
        nd.data_type = 'RGBA'
        nd.blend_type = blend
        self.set(nd.inputs['Factor'], fac)
        self.set(nd.inputs[6], a)       # A (colour)
        self.set(nd.inputs[7], b)       # B (colour)
        return nd.outputs[2]

    def mixf(self, fac, a, b):
        nd = self.nt.nodes.new('ShaderNodeMix')
        nd.data_type = 'FLOAT'
        self.set(nd.inputs['Factor'], fac)
        self.set(nd.inputs[2], a)
        self.set(nd.inputs[3], b)
        return nd.outputs[0]

    def mixv(self, fac, a, b):
        nd = self.nt.nodes.new('ShaderNodeMix')
        nd.data_type = 'VECTOR'
        self.set(nd.inputs['Factor'], fac)
        self.set(nd.inputs[4], a)
        self.set(nd.inputs[5], b)
        return nd.outputs[1]

    def colscale(self, col, s):
        """colour * scalar (via vector math scale; colours and vectors convert implicitly)."""
        return self.vmath('SCALE', col, scale=s)

    # ---------------------------------------------------------- textures ---
    def noise(self, vec, scale=1.0, detail=2.0, rough=0.5, lac=2.0, dist=0.0, dims='3D', w=None,
              ntype='FBM', normalize=True):
        nd = self.nt.nodes.new('ShaderNodeTexNoise')
        nd.noise_dimensions = dims
        try:
            nd.noise_type = ntype
        except Exception:
            pass
        nd.normalize = normalize
        self.set(nd.inputs['Vector'], vec)
        self.set(nd.inputs['Scale'], scale)
        self.set(nd.inputs['Detail'], detail)
        self.set(nd.inputs['Roughness'], rough)
        self.set(nd.inputs['Lacunarity'], lac)
        self.set(nd.inputs['Distortion'], dist)
        if w is not None:
            self.set(nd.inputs['W'], w)
        return nd

    def voronoi(self, vec, scale=1.0, feature='F1', metric='EUCLIDEAN', rand=1.0, dims='3D', w=None,
                detail=0.0):
        nd = self.nt.nodes.new('ShaderNodeTexVoronoi')
        nd.voronoi_dimensions = dims
        nd.feature = feature
        nd.distance = metric
        self.set(nd.inputs['Vector'], vec)
        self.set(nd.inputs['Scale'], scale)
        self.set(nd.inputs['Randomness'], rand)
        if 'Detail' in nd.inputs:
            self.set(nd.inputs['Detail'], detail)
        if w is not None:
            self.set(nd.inputs['W'], w)
        return nd

    def wave(self, vec, scale=1.0, dist=0.0, detail=2.0, dscale=1.0, wtype='BANDS', direction='X',
             profile='SIN', phase=0.0, rough=0.5):
        nd = self.nt.nodes.new('ShaderNodeTexWave')
        nd.wave_type = wtype
        if wtype == 'BANDS':
            nd.bands_direction = direction
        else:
            nd.rings_direction = direction
        nd.wave_profile = profile
        self.set(nd.inputs['Vector'], vec)
        self.set(nd.inputs['Scale'], scale)
        self.set(nd.inputs['Distortion'], dist)
        self.set(nd.inputs['Detail'], detail)
        self.set(nd.inputs['Detail Scale'], dscale)
        self.set(nd.inputs['Detail Roughness'], rough)
        self.set(nd.inputs['Phase Offset'], phase)
        return nd

    def white(self, vec, dims='3D', w=None):
        nd = self.nt.nodes.new('ShaderNodeTexWhiteNoise')
        nd.noise_dimensions = dims
        self.set(nd.inputs['Vector'], vec)
        if w is not None:
            self.set(nd.inputs['W'], w)
        return nd

    def ramp(self, fac, stops, interp='LINEAR'):
        """stops: [(pos, (r,g,b[,a])), ...]"""
        nd = self.nt.nodes.new('ShaderNodeValToRGB')
        cr = nd.color_ramp
        cr.interpolation = interp
        while len(cr.elements) > len(stops):
            cr.elements.remove(cr.elements[-1])
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for el, (p, c) in zip(cr.elements, stops):
            el.position = p
            c = tuple(c) + (1.0,) * (4 - len(c))
            el.color = c
        self.set(nd.inputs['Fac'], fac)
        return nd

    def bump(self, height, strength=1.0, distance=1.0, normal=None, invert=False):
        nd = self.nt.nodes.new('ShaderNodeBump')
        nd.invert = invert
        self.set(nd.inputs['Height'], height)
        self.set(nd.inputs['Strength'], strength)
        self.set(nd.inputs['Distance'], distance)
        if normal is not None:
            self.set(nd.inputs['Normal'], normal)
        return nd.outputs['Normal']

    def _shader(self):
        return True

    # ----------------------------------------------------------- shaders ---
    def principled(self, **inputs):
        nd = self.nt.nodes.new('ShaderNodeBsdfPrincipled')
        for k, v in inputs.items():
            self.set(nd.inputs[k.replace('_', ' ')], v)
        return nd

    def emission(self, col, strength=1.0):
        nd = self.nt.nodes.new('ShaderNodeEmission')
        self.set(nd.inputs['Color'], col)
        self.set(nd.inputs['Strength'], strength)
        return nd

    def mixshader(self, fac, a, b):
        nd = self.nt.nodes.new('ShaderNodeMixShader')
        self.set(nd.inputs[0], fac)
        self.set(nd.inputs[1], a)
        self.set(nd.inputs[2], b)
        return nd

    def addshader(self, a, b):
        nd = self.nt.nodes.new('ShaderNodeAddShader')
        self.set(nd.inputs[0], a)
        self.set(nd.inputs[1], b)
        return nd

    def output(self, surface=None, volume=None, displacement=None, target='ALL'):
        nd = None
        for n in self.nt.nodes:
            if n.bl_idname == 'ShaderNodeOutputMaterial':
                nd = n
        if nd is None:
            nd = self.nt.nodes.new('ShaderNodeOutputMaterial')
        nd.target = target
        if surface is not None:
            self.set(nd.inputs['Surface'], surface)
        if volume is not None:
            self.set(nd.inputs['Volume'], volume)
        if displacement is not None:
            self.set(nd.inputs['Displacement'], displacement)
        return nd

    def attr(self, name, kind='GEOMETRY'):
        nd = self.nt.nodes.new('ShaderNodeAttribute')
        nd.attribute_type = kind
        nd.attribute_name = name
        return nd

    def geo(self):
        return self.nt.nodes.new('ShaderNodeNewGeometry')

    def texco(self):
        return self.nt.nodes.new('ShaderNodeTexCoord')

    def objinfo(self):
        return self.nt.nodes.new('ShaderNodeObjectInfo')

    def camdata(self):
        return self.nt.nodes.new('ShaderNodeCameraData')

    def lightpath(self):
        return self.nt.nodes.new('ShaderNodeLightPath')


def new_material(name, clear=True):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nb = NB(m.node_tree, clear=clear)
    return m, nb
