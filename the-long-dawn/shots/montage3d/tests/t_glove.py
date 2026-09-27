"""Glove turnaround in clay (Blender): poses side by side, three views.

  Blender -b --factory-startup -P tests/t_glove.py -- out.png [poses] [view]
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

from kit import core as C  # noqa: E402
from kit.nodes import new_material  # noqa: E402
import glove as GL  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:]
out = argv[0]
poses = (argv[1] if len(argv) > 1 else 'flat,relaxed,reach,cup,fist').split(',')
view = argv[2] if len(argv) > 2 else 'top'
C.reset()
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.cycles.device = 'GPU'
try:
    pr = bpy.context.preferences.addons['cycles'].preferences
    pr.compute_device_type = 'METAL'
    pr.get_devices()
    for d in pr.devices:
        d.use = d.type != 'CPU'
except Exception as e:
    print('no metal', e)
sc.cycles.samples = 24
sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = 1600, 640
sc.view_settings.view_transform = 'Standard'
w = bpy.data.worlds.new('W')
sc.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.25, 0.26, 0.28, 1)
w.node_tree.nodes['Background'].inputs[1].default_value = 0.6
if os.environ.get('CLAY'):
    m, nb = new_material('clay')
    nb.output(surface=nb.principled(Base_Color=(0.55, 0.5, 0.45), Roughness=0.55))
else:
    m = GL.leather_material(new_material, base=(0.2, 0.13, 0.08))
wool = GL.wool_material(new_material, base=(0.1, 0.1, 0.1))
n = len(poses)
for i, pn in enumerate(poses):
    g = GL.Glove(f'g{i}', m, wool, mirror=False, sleeve=False)
    g.set_pose(GL.POSES[pn])
    g.rig.location = (0.0, (i - (n - 1) / 2) * 0.13, 0.0)
    if view == 'side':
        g.rig.rotation_euler = (math.radians(90), 0, 0)
    elif view == 'palm':
        g.rig.rotation_euler = (math.radians(180), 0, 0)
C.sun('key', (0.4, -0.5, 0.8), (1.0, 0.97, 0.92), 3.0, angle_deg=3.0)
C.sun('fill', (-0.6, 0.4, 0.3), (0.8, 0.85, 1.0), 0.8, angle_deg=10.0)
cd = bpy.data.cameras.new('cam')
cd.type = 'ORTHO'
cd.ortho_scale = 0.13 * n + 0.02
cam = bpy.data.objects.new('cam', cd)
sc.collection.objects.link(cam)
cam.location = (0.04, 0.0, 1.0)
cam.rotation_euler = (0, 0, math.radians(-90))
sc.camera = cam
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print('WROTE', out)
