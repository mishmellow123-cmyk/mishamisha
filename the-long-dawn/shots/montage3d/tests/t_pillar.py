"""Debug: view a generated mesh (.bin) under neutral light. Blender -b -P tests/t_pillar.py -- mesh.bin out.png"""
import math, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import bpy
from kit import core as C, figure as FG
argv = sys.argv[sys.argv.index('--') + 1:]
C.reset()
C.setup_render(scale=0.5, samples=16, raytrace=False)
sc = bpy.context.scene
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_depth = '8'
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.05, 0.06, 0.08, 1)
C.sun('key', (-0.8, -0.3, 0.45), (1, 1, 1), 3.0, angle_deg=1)
m = C.surface_mat('rock', (0.3, 0.29, 0.27), rough=0.9, fog=False)
me = FG.mesh_from_bin('p', argv[0], m)
ob = C.link_obj(bpy.data.objects.new('p', me))
xs = [v.co.x for v in me.vertices]; ys = [v.co.y for v in me.vertices]; zs = [v.co.z for v in me.vertices]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
H = max(zs) - min(zs)
cam = C.make_camera('CAM', hfov=30.0)
d = H / (2 * math.tan(math.radians(6.2))) * 1.15
C.key_camera(cam, 1, (cx, cy - d, min(zs) + H * 0.5), 0.0, 0.0)
sc.frame_set(1)
sc.render.filepath = argv[1]
bpy.ops.render.render(write_still=True)
