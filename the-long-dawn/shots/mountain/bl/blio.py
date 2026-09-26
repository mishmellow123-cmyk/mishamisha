"""Blender-side IO helpers (Blender's bundled numpy cannot load on macOS 12, so everything here
uses the stdlib `array` module with foreach_set)."""
import array
import json

import bpy


def load_mesh(path, name, smooth=True):
    meta = json.load(open(path + '.json'))
    nv, nt = meta['nv'], meta['nt']
    raw = open(path + '.bin', 'rb').read()
    off = 0
    V = array.array('f')
    V.frombytes(raw[off:off + nv * 12])
    off += nv * 12
    T = array.array('i')
    T.frombytes(raw[off:off + nt * 12])
    off += nt * 12
    me = bpy.data.meshes.new(name)
    me.vertices.add(nv)
    me.vertices.foreach_set('co', V)
    me.loops.add(nt * 3)
    me.loops.foreach_set('vertex_index', T)
    me.polygons.add(nt)
    me.polygons.foreach_set('loop_start', array.array('i', range(0, nt * 3, 3)))
    me.update(calc_edges=True)
    for a in meta['attrs']:
        A = array.array('f')
        A.frombytes(raw[off:off + nv * 4])
        off += nv * 4
        at = me.attributes.new(a, 'FLOAT', 'POINT')
        at.data.foreach_set('value', A)
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob, meta


def read_floats(path):
    A = array.array('f')
    A.frombytes(open(path, 'rb').read())
    return A


def set_quads(ob, path):
    """Replace ob's mesh with the streak quads in `path` (int32 n_verts, float32 xyz, float32
    rgba) and a 'col' colour attribute."""
    raw = open(path, 'rb').read()
    n = array.array('i')
    n.frombytes(raw[:4])
    nv = n[0]
    me = ob.data
    me.clear_geometry()
    if nv == 0:
        return 0
    V = array.array('f')
    V.frombytes(raw[4:4 + nv * 12])
    C = array.array('f')
    C.frombytes(raw[4 + nv * 12:4 + nv * 12 + nv * 16])
    nq = nv // 4
    me.vertices.add(nv)
    me.vertices.foreach_set('co', V)
    me.loops.add(nv)
    me.loops.foreach_set('vertex_index', array.array('i', range(nv)))
    me.polygons.add(nq)
    me.polygons.foreach_set('loop_start', array.array('i', range(0, nv, 4)))
    me.update(calc_edges=True)
    if 'col' in me.color_attributes:
        me.color_attributes.remove(me.color_attributes['col'])
    ca = me.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', C)
    return nq
