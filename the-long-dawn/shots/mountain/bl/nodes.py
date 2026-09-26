"""Tiny helpers for building shader node trees from code."""
import bpy


class Tree:
    def __init__(self, nt):
        self.nt = nt
        self.x = 0

    def n(self, typ, **kw):
        node = self.nt.nodes.new(typ)
        node.location = (self.x, 0)
        self.x += 40
        for k, v in kw.items():
            if k.startswith('_'):
                setattr(node, k[1:], v)
            else:
                self.set(node.inputs[k], v)
        return node

    def set(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        elif isinstance(v, bpy.types.Node):
            self.nt.links.new(v.outputs[0], sock)
        else:
            if isinstance(v, (int, float)) and hasattr(sock, 'default_value') and \
                    hasattr(sock.default_value, '__len__'):
                sock.default_value = [v] * len(sock.default_value)
            elif isinstance(v, (tuple, list)) and hasattr(sock.default_value, '__len__'):
                n = len(sock.default_value)
                vv = list(v)[:n] + [1.0] * max(0, n - len(v))
                sock.default_value = vv
            else:
                sock.default_value = v

    def link(self, a, b):
        self.nt.links.new(a, b)

    # ---- math shortcuts (return output sockets) ----
    def math(self, op, a, b=None, c=None, clamp=False):
        m = self.nt.nodes.new('ShaderNodeMath')
        m.operation = op
        m.use_clamp = clamp
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            self.set(m.inputs[i], v)
        return m.outputs[0]

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

    def vmath(self, op, a, b=None, scale=None):
        m = self.nt.nodes.new('ShaderNodeVectorMath')
        m.operation = op
        if a is not None:
            self.set(m.inputs[0], a)
        if b is not None:
            self.set(m.inputs[1], b)
        if scale is not None:
            self.set(m.inputs[3], scale)
        out = m.outputs['Value'] if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE') else m.outputs['Vector']
        return out

    def sep(self, v):
        s = self.nt.nodes.new('ShaderNodeSeparateXYZ')
        self.set(s.inputs[0], v)
        return s.outputs

    def comb(self, x, y, z):
        c = self.nt.nodes.new('ShaderNodeCombineXYZ')
        self.set(c.inputs[0], x)
        self.set(c.inputs[1], y)
        self.set(c.inputs[2], z)
        return c.outputs[0]

    def mixc(self, fac, a, b):
        m = self.nt.nodes.new('ShaderNodeMix')
        m.data_type = 'RGBA'
        self.set(m.inputs['Factor'], fac)
        self.set(m.inputs['A'], a)
        self.set(m.inputs['B'], b)
        return m.outputs['Result']

    def mixf(self, fac, a, b):
        m = self.nt.nodes.new('ShaderNodeMix')
        m.data_type = 'FLOAT'
        self.set(m.inputs['Factor'], fac)
        self.set(m.inputs['A'], a)
        self.set(m.inputs['B'], b)
        return m.outputs['Result']

    def smooth(self, e0, e1, x):
        m = self.nt.nodes.new('ShaderNodeMapRange')
        m.interpolation_type = 'SMOOTHSTEP'
        self.set(m.inputs['Value'], x)
        self.set(m.inputs['From Min'], e0)
        self.set(m.inputs['From Max'], e1)
        return m.outputs['Result']

    def maprange(self, x, a0, a1, b0, b1, clamp=True):
        m = self.nt.nodes.new('ShaderNodeMapRange')
        m.clamp = clamp
        self.set(m.inputs['Value'], x)
        self.set(m.inputs['From Min'], a0)
        self.set(m.inputs['From Max'], a1)
        self.set(m.inputs['To Min'], b0)
        self.set(m.inputs['To Max'], b1)
        return m.outputs['Result']

    def cscale(self, col, f):
        """colour * float -> colour (via vector math scale)"""
        return self.vmath('SCALE', col, scale=f)

    def attr(self, name, kind='GEOMETRY', out='Fac'):
        a = self.nt.nodes.new('ShaderNodeAttribute')
        a.attribute_type = kind
        a.attribute_name = name
        return a.outputs[out]

    def noise(self, vec, scale=1.0, detail=2.0, rough=0.5, dims='3D', w=None, out='Fac',
              lac=2.0, distortion=0.0):
        n = self.nt.nodes.new('ShaderNodeTexNoise')
        n.noise_dimensions = dims
        self.set(n.inputs['Vector'], vec)
        self.set(n.inputs['Scale'], scale)
        self.set(n.inputs['Detail'], detail)
        self.set(n.inputs['Roughness'], rough)
        self.set(n.inputs['Lacunarity'], lac)
        self.set(n.inputs['Distortion'], distortion)
        if w is not None:
            self.set(n.inputs['W'], w)
        return n.outputs[out]
