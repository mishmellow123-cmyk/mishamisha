"""Quick look at a terrain mesh: blender -b --factory-startup -P bl_preview.py -- <mesh> <cam.json> <out.exr> [scale]"""
import json
import math
import os
import sys
import time

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blio

argv = sys.argv[sys.argv.index('--') + 1:]
mesh_path, cam_path, out_path = argv[0], argv[1], argv[2]
scale = float(argv[3]) if len(argv) > 3 else 0.5

sc = bpy.context.scene
for o in list(sc.objects):
    bpy.data.objects.remove(o)
sc.render.engine = 'BLENDER_EEVEE_NEXT'
sc.render.resolution_x = 1920
sc.render.resolution_y = 804
sc.render.resolution_percentage = int(scale * 100)
sc.eevee.taa_render_samples = 16
sc.view_settings.view_transform = 'Standard'
sc.render.image_settings.file_format = 'OPEN_EXR'
sc.render.image_settings.color_depth = '16'
sc.render.image_settings.exr_codec = 'ZIP'

t0 = time.time()
ob, meta = blio.load_mesh(mesh_path, 'land')
print('load', time.time() - t0)

# material: snow on gentle slopes, rock on steep
m = bpy.data.materials.new('land')
m.use_nodes = True
nt = m.node_tree
b = nt.nodes['Principled BSDF']
geo = nt.nodes.new('ShaderNodeNewGeometry')
sep = nt.nodes.new('ShaderNodeSeparateXYZ')
nt.links.new(geo.outputs['Normal'], sep.inputs[0])
mr = nt.nodes.new('ShaderNodeMapRange')
mr.inputs['From Min'].default_value = 0.55
mr.inputs['From Max'].default_value = 0.75
nt.links.new(sep.outputs['Z'], mr.inputs['Value'])
mix = nt.nodes.new('ShaderNodeMix')
mix.data_type = 'RGBA'
mix.inputs['A'].default_value = (0.05, 0.048, 0.045, 1)
mix.inputs['B'].default_value = (0.85, 0.88, 0.93, 1)
nt.links.new(mr.outputs['Result'], mix.inputs['Factor'])
nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
b.inputs['Roughness'].default_value = 0.8
ob.data.materials.append(m)

# cloud sea placeholder: big plane at z=0
bpy.ops.mesh.primitive_plane_add(size=300000, location=(0, 0, -20))
cl = bpy.context.object
cm = bpy.data.materials.new('cloud')
cm.use_nodes = True
cm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.7, 0.72, 0.78, 1)
cm.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 1.0
cl.data.materials.append(cm)

# world
w = bpy.data.worlds.new('W')
sc.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.004, 0.006, 0.016, 1)
w.node_tree.nodes['Background'].inputs[1].default_value = 1.0

# moon
ld = bpy.data.lights.new('moon', 'SUN')
ld.energy = 0.35
ld.color = (0.62, 0.72, 0.9)
ld.angle = math.radians(0.6)
lo = bpy.data.objects.new('moon', ld)
sc.collection.objects.link(lo)
az, el = math.radians(float(os.environ.get('MOON_AZ', 250))), math.radians(float(os.environ.get('MOON_EL', 22)))
d = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
lo.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()

# camera
cj = json.load(open(cam_path))
cd = bpy.data.cameras.new('cam')
cd.clip_start = 1.0
cd.clip_end = 400000.0
cd.sensor_fit = 'HORIZONTAL'
cd.angle = math.radians(cj['hfov'])
cam = bpy.data.objects.new('cam', cd)
sc.collection.objects.link(cam)
sc.camera = cam
p = Vector(cj['pos'])
t = Vector(cj['target'])
cam.location = p
cam.rotation_euler = (t - p).to_track_quat('-Z', 'Y').to_euler()

sc.render.filepath = out_path
t0 = time.time()
bpy.ops.render.render(write_still=True)
print('render', time.time() - t0)
