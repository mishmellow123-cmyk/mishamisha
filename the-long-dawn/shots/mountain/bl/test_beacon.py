"""Close-up test of the beacon asset (flame cards, glow, light) on a moonlit snow plane.
blender -b --factory-startup -P bl/test_beacon.py -- --out DIR --frames 1579,1580,1585,1600 --dist 40
"""
import argparse
import math
import os
import sys
import time

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scene as SC
import beacon as BC

argv = sys.argv[sys.argv.index('--') + 1:]
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--frames', default='1579,1580,1585,1600')
ap.add_argument('--dist', type=float, default=40.0)
ap.add_argument('--height', type=float, default=6.0)
ap.add_argument('--scale', type=float, default=0.5)
ap.add_argument('--bscale', type=float, default=1.7)
a = ap.parse_args(argv)

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
SC.render_settings(a.scale, 32, motion_blur=False)
look = SC.Look()
look.moon_dir = SC.dir_from(-118, 21)
SC.fog_group(look)
SC.build_world(look)
sc = bpy.context.scene
sc['ld_time'] = 0.0
sc.keyframe_insert('["ld_time"]', frame=0)
sc['ld_time'] = 1.0
sc.keyframe_insert('["ld_time"]', frame=24)
for fc in sc.animation_data.action.fcurves:
    fc.extrapolation = 'LINEAR'
    for k in fc.keyframe_points:
        k.interpolation = 'LINEAR'

# snowy knoll: a displaced grid
bpy.ops.mesh.primitive_grid_add(x_subdivisions=160, y_subdivisions=160, size=240, location=(0, 0, 0))
g = bpy.context.object
me = g.data
for v in me.vertices:
    x, y = v.co.x, v.co.y
    r = math.hypot(x, y)
    v.co.z = 200.0 - 0.0025 * r * r - 0.35 * r * (r > 8) + 0.6 * math.sin(x * 0.3) * math.cos(y * 0.23)
at = me.attributes.new('moon', 'FLOAT', 'POINT')
at.data.foreach_set('value', [1.0] * len(me.vertices))
for nm, val in (('snow', 0.85), ('ao', 1.0), ('cz', -50.0), ('slope', 20.0)):
    at = me.attributes.new(nm, 'FLOAT', 'POINT')
    at.data.foreach_set('value', [val] * len(me.vertices))
me.shade_smooth()
g.data.materials.append(SC.land_material(look))

# camera
cd = bpy.data.cameras.new('cam')
cd.sensor_fit = 'HORIZONTAL'
cd.sensor_width = 36.0
cd.lens = 18.0 / math.tan(math.radians(31))
cd.clip_start = 0.3
cd.clip_end = 100000
cam = bpy.data.objects.new('cam', cd)
sc.collection.objects.link(cam)
sc.camera = cam
cam.location = (0, -a.dist, 200.0 + a.height)
tgt = Vector((0, 0, 202.5))
cam.rotation_euler = (tgt - cam.location).to_track_quat('-Z', 'Y').to_euler()

b = BC.Beacon(look, 'T', (0, 0, 200.0), 1580, a.bscale, cam, detail=True, shadows=True, seed=3)
b.animate(range(1570, 1620))
os.makedirs(a.out, exist_ok=True)
for f in [int(v) for v in a.frames.split(',')]:
    sc.frame_set(f)
    sc.render.filepath = os.path.join(a.out, 'f_%05d.exr' % f)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print('frame', f, '%.1fs' % (time.time() - t0), flush=True)
