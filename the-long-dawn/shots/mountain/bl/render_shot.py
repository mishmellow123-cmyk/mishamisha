"""Render a MOUNTAIN shot in Blender.

blender -b --factory-startup -P bl/render_shot.py -- --shot run --frames 1520,1581 \
        --scale 0.5 --samples 16 --out /path/dir [--markers]
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
from mathutils import Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blio
import scene as SC
import beacon as BC
from nodes import Tree

ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, 'cache')


def parse():
    argv = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--shot', default='run')
    ap.add_argument('--frames', default='1520')
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--out', required=True)
    ap.add_argument('--markers', action='store_true')
    ap.add_argument('--nomb', action='store_true')
    ap.add_argument('--nobeacons', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--land', default=None)
    ap.add_argument('--cloud', default=None)
    ap.add_argument('--cam', default=None)
    return ap.parse_args(argv)


def frames_of(spec):
    out = []
    for part in spec.split(','):
        if '-' in part[1:]:
            a, b = part.split('-')
            step = 1
            if ':' in b:
                b, step = b.split(':')
            out += list(range(int(a), int(b) + 1, int(step)))
        else:
            out.append(int(part))
    return out


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)


def setup_time():
    sc = bpy.context.scene
    sc['ld_time'] = 0.0
    sc.keyframe_insert('["ld_time"]', frame=0)
    sc['ld_time'] = 1.0
    sc.keyframe_insert('["ld_time"]', frame=24)
    for fc in sc.animation_data.action.fcurves:
        fc.extrapolation = 'LINEAR'
        for k in fc.keyframe_points:
            k.interpolation = 'LINEAR'


def setup_camera(shot):
    cj = json.load(open(os.path.join(CACHE, '%s_cam.json' % shot)))
    cd = bpy.data.cameras.new('cam')
    cd.sensor_fit = 'HORIZONTAL'
    cd.sensor_width = 36.0
    cd.clip_start = 0.5
    cd.clip_end = 600000.0
    cam = bpy.data.objects.new('cam', cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = 'QUATERNION'
    for c in cj['frames']:
        f = c['frame']
        cam.location = c['loc']
        cam.rotation_quaternion = Quaternion(c['quat'])
        cd.lens = 18.0 / math.tan(math.radians(c['hfov'] / 2))
        cam.keyframe_insert('location', frame=f)
        cam.keyframe_insert('rotation_quaternion', frame=f)
        cd.keyframe_insert('lens', frame=f)
    return cam, cj


def markers(shot):
    B = json.load(open(os.path.join(CACHE, '%s_beacons.json' % shot)))
    m = bpy.data.materials.new('marker')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    T = SC.Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    lit = T.attr('lit', 'OBJECT')
    em = T.n('ShaderNodeEmission', Color=(1.0, 0.45, 0.12, 1))
    T.link(T.add(T.mul(lit, 60.0), 0.3), em.inputs['Strength'])
    T.link(em.outputs[0], out.inputs['Surface'])
    for b in B:
        x, y, z = b['pos']
        d = math.hypot(x - 100, y - 500)
        r = max(1.2, d * 0.0012)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x, y, z + r), segments=12,
                                             ring_count=6)
        o = bpy.context.object
        o.data.materials.append(m)
        o['lit'] = 0.0
        o.keyframe_insert('["lit"]', frame=b['t'] - 1)
        o['lit'] = 1.0
        o.keyframe_insert('["lit"]', frame=b['t'])
        for fc in o.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = 'CONSTANT'


R_EARTH = 6.371e6


def spark_material():
    m = bpy.data.materials.new('Sparks')
    m.use_nodes = True
    m.surface_render_method = 'BLENDED'
    m.use_backface_culling = False
    nt = m.node_tree
    nt.nodes.clear()
    T = Tree(nt)
    out = T.n('ShaderNodeOutputMaterial')
    ca = T.n('ShaderNodeVertexColor')
    ca.layer_name = 'col'
    em = T.n('ShaderNodeEmission', Strength=1.0)
    T.link(ca.outputs['Color'], em.inputs['Color'])
    tr = T.n('ShaderNodeBsdfTransparent')
    add = T.n('ShaderNodeAddShader')
    T.link(em.outputs[0], add.inputs[0])
    T.link(tr.outputs[0], add.inputs[1])
    T.link(add.outputs[0], out.inputs['Surface'])
    return m


def setup_beacons(look, cam, cj, shot, ref, only=None):
    B = json.load(open(os.path.join(CACHE, '%s_beacons.json' % shot)))
    cam_path = {int(c['frame']): tuple(c['loc']) for c in cj['frames']}
    cam_right = {int(c['frame']): tuple(c['right']) for c in cj['frames']}
    frames = sorted(cam_path)
    out = []
    for k, b in enumerate(B):
        if only is not None and b['name'] not in only:
            continue
        x, y, z = b['pos']
        z -= ((x - ref[0]) ** 2 + (y - ref[1]) ** 2) / (2 * R_EARTH)
        hero = b['name'] in ('S0', 'B3')
        lit_frames = [f for f in frames if f >= b['t'] - 2]
        if not lit_frames:
            continue
        dmin = min(math.dist(cam_path[f], (x, y, z)) for f in lit_frames)
        bc = BC.Beacon(look, b['name'], (x, y, z), b['t'], b['scale'], cam, detail=hero,
                       shadows=hero, light=(dmin < 9000), seed=k + 1, sprite=not hero,
                       smoke=(dmin < 9000))
        bc.animate(frames, cam_path=cam_path, cam_right=cam_right)
        out.append(bc)
    return out


def main():
    a = parse()
    t0 = time.time()
    clear_scene()
    SC.render_settings(a.scale, a.samples, motion_blur=not a.nomb)
    look = SC.Look()
    import importlib.util
    spec = importlib.util.spec_from_file_location('world', os.path.join(ROOT, 'world_bl.py'))
    wb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wb)
    look.moon_dir = SC.dir_from(wb.MOON_AZ, wb.MOON_EL)
    SC.fog_group(look)
    SC.build_world(look)
    land, meta = blio.load_mesh(os.path.join(CACHE, a.land or ('%s_land' % a.shot)), 'land')
    land.data.materials.append(SC.land_material(look))
    cl, _ = blio.load_mesh(os.path.join(CACHE, a.cloud or ('%s_cloud' % a.shot)), 'cloud')
    cl.data.materials.append(SC.cloud_material(look))
    setup_time()
    cam, cj = setup_camera(a.cam or a.shot)
    ref = meta.get('ref', [0, 0, 0])
    if a.markers:
        markers(a.shot)
    elif not a.nobeacons:
        setup_beacons(look, cam, cj, a.cam or a.shot, ref)
    sparks = None
    sdir = os.path.join(CACHE, '%s_sparks' % (a.cam or a.shot))
    if os.path.isdir(sdir) and not a.markers and not a.nobeacons:
        me = bpy.data.meshes.new('sparks')
        sparks = bpy.data.objects.new('sparks', me)
        bpy.context.scene.collection.objects.link(sparks)
        sparks.data.materials.append(spark_material())
    print('scene built %.1fs' % (time.time() - t0), flush=True)
    os.makedirs(a.out, exist_ok=True)
    sc = bpy.context.scene
    for f in frames_of(a.frames):
        path = os.path.join(a.out, 'f_%05d.exr' % f)
        if a.skip_existing and os.path.exists(path):
            continue
        sc.frame_set(f)
        if sparks is not None:
            sp = os.path.join(sdir, 'f_%05d.bin' % f)
            if os.path.exists(sp):
                blio.set_quads(sparks, sp)
                if not sparks.data.materials:
                    sparks.data.materials.append(bpy.data.materials['Sparks'])
        sc.render.filepath = path
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print('frame %d %.1fs' % (f, time.time() - t1), flush=True)


if __name__ == '__main__':
    main()
