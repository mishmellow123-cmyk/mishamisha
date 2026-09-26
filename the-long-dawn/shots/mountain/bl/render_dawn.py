"""Render DAWN_C (cut C, v2 2400-2655) in Blender.

blender -b --factory-startup -P bl/render_dawn.py -- --frames 2400-2655 --scale 1 --samples 32 \
        --out DIR [--skip-existing]
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blio
import scene as SC
import beacon as BC
import dawn as DW
from render_shot import frames_of, setup_camera, setup_time, R_EARTH

ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, 'cache')


def parse():
    argv = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='2400')
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nomb', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    return ap.parse_args(argv)


def key_scene_prop(name, values):
    sc = bpy.context.scene
    for f, v in values:
        sc[name] = float(v)
        sc.keyframe_insert('["%s"]' % name, frame=f)


def main():
    a = parse()
    t0 = time.time()
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    SC.render_settings(a.scale, a.samples, motion_blur=not a.nomb)
    sun = json.load(open(os.path.join(CACHE, 'dawn_sun.json')))
    look = DW.DawnLook(math.radians(sun['az']))
    look.sun_disc = sun['radius']
    sc = bpy.context.scene
    sc['sun_az'] = math.radians(sun['az'])
    key_scene_prop('sun_el', sorted((int(k), v) for k, v in sun['elev'].items()))
    DW.dawn_fog_group(look)
    DW.build_dawn_world(look)
    land, meta = blio.load_mesh(os.path.join(CACHE, 'dawn_land'), 'land')
    land.data.materials.append(DW.dawn_land_material(look))
    cl, _ = blio.load_mesh(os.path.join(CACHE, 'dawn_cloud'), 'cloud')
    cl.data.materials.append(DW.dawn_cloud_material(look))
    setup_time()
    cam, cj = setup_camera('dawn')
    ref = meta.get('ref', [0, 0, 0])
    # the night's beacons, now smouldering and smoking
    DW.dawn_smoke_material(look)
    B = json.load(open(os.path.join(CACHE, 'dawn_beacons.json')))
    cam_path = {int(c['frame']): tuple(c['loc']) for c in cj['frames']}
    cam_right = {int(c['frame']): tuple(c['right']) for c in cj['frames']}
    frames = sorted(cam_path)
    for k, b in enumerate(B):
        x, y, z = b['pos']
        z -= ((x - ref[0]) ** 2 + (y - ref[1]) ** 2) / (2 * R_EARTH)
        bc = BC.Beacon(look, b['name'], (x, y, z), -10000, b['scale'] * 1.6, cam, detail=False,
                       light=False, seed=40 + k, sprite=True, smoke=True)
        bc.animate(frames, cam_path=cam_path, cam_right=cam_right, sprite_peak=2.5, glow=0.0)
    # eagles
    E = json.load(open(os.path.join(CACHE, 'dawn_eagles.json')))
    em = DW.eagle_material(look)
    ref_cam = [c for c in cj['frames'] if int(c['frame']) == 2490][0]
    C = Vector(ref_cam['loc'])
    Fw = Vector(ref_cam['fwd'])
    Rt = Vector(ref_cam['right'])
    Up = Vector(ref_cam['up'])
    th = math.tan(math.radians(ref_cam['hfov'] / 2))
    eagles = []
    for e in E:
        V, Fc, Wt = DW.eagle_rest(e['span'])
        me = bpy.data.meshes.new('eagle%d' % e['id'])
        me.from_pydata(V, [], Fc)
        me.update()
        ob = bpy.data.objects.new('eagle%d' % e['id'], me)
        ob.data.materials.append(em)
        sc.collection.objects.link(ob)
        eagles.append((e, ob, V, Wt))
    print('scene built %.1fs' % (time.time() - t0), flush=True)
    os.makedirs(a.out, exist_ok=True)

    def place_eagles(f):
        t = f / 24.0
        for e, ob, V, Wt in eagles:
            s = (f - e['t0']) / e['dur']
            if s < -0.05 or s > 1.05:
                ob.hide_render = True
                continue
            ob.hide_render = False
            d = e['dist']
            xl = (0.95 - 2.2 * s) * d * th
            bob = 0.012 * e['span'] * math.sin(2 * math.pi * (t / (e['flap_period'] / 24.0) + e['phase']))
            P = C + Fw * d + Rt * xl + Up * (e['height'] + bob)
            head = -Rt
            head.z = 0.0
            head.normalize()
            upv = Vector((0, 0, 1))
            rgt = head.cross(upv).normalized()
            bank = math.radians(4.0 * math.sin(2 * math.pi * (s * 0.7 + e['phase'])))
            M = Matrix((rgt, head, upv)).transposed()
            M = M @ Matrix.Rotation(bank, 3, 'Y')
            ob.matrix_world = Matrix.Translation(P) @ M.to_4x4()
            posed = DW.eagle_pose(V, Wt, t, e['flap_period'] / 24.0, e['phase'], e['glide'],
                                  e['span'])
            flat = [c for p in posed for c in p]
            ob.data.vertices.foreach_set('co', flat)
            ob.data.update()

    for f in frames_of(a.frames):
        path = os.path.join(a.out, 'f_%05d.exr' % f)
        if a.skip_existing and os.path.exists(path):
            continue
        sc.frame_set(f)
        place_eagles(f)
        sc.render.filepath = path
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print('frame %d %.1fs' % (f, time.time() - t1), flush=True)


if __name__ == '__main__':
    main()
