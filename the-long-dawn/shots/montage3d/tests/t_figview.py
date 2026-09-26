"""Debug: render a shot's figure alone, neutral grey light, from several views.
Blender -b --factory-startup -P tests/t_figview.py -- <job.json> <shot> <out_prefix> <frames,comma>"""
import importlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from kit import core as C, figure as FG, fire as FK  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:]
job = json.load(open(argv[0]))
shot = importlib.import_module(argv[1])
prefix = argv[2]
frames = [int(x) for x in argv[3].split(',')]
C.reset()
C.setup_render(scale=0.5, samples=16, raytrace=False)
sc = bpy.context.scene
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_depth = '8'
sc.view_settings.view_transform = 'Standard'
w = bpy.data.worlds.new('W')
sc.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.35, 0.36, 0.38, 1)
C.sun('key', (-0.4, -0.6, 0.7), (1, 1, 1), 2.5, angle_deg=5)
cam = C.make_camera('CAM', hfov=24.0)
fb = Vector(shot.DRUM) + Vector((0, 0, 0.91)) if hasattr(shot, 'DRUM') else Vector((0, 0, 0))
info = shot._young_person(C, FG, FK, cam, job['timing'], fb) if hasattr(shot, '_young_person') else \
    shot._robed_figure(C, FG, FK, cam, job['timing'], fb)
C.mesh_obj('floor', [(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)], [(0, 1, 2, 3)],
           mat=C.surface_mat('fl', (0.5, 0.5, 0.5), fog=False), smooth=False)
fx, fy, _ = shot.FEET
views = dict(back=(fx - 0.6, fy - 6.5, 1.1, 5.0, -1.0), side=(fx + 6.5, fy + 0.4, 1.1, -90 + 3.0, -1.0),
             q34=(fx - 4.8, fy - 4.2, 1.2, 49.0, -2.0))
for f in frames:
    sc.frame_set(f)
    if hasattr(shot, 'per_frame'):
        shot.per_frame(job, f)
    for vn, (x, y, z, yw, pt) in views.items():
        cam.animation_data_clear()
        C.key_camera(cam, f, (x, y, z), yw, pt)
        sc.frame_set(f)
        sc.render.filepath = f'{prefix}_{f}_{vn}.png'
        bpy.ops.render.render(write_still=True)
print('DONE')
