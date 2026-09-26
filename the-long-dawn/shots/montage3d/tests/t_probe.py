"""Probe: EEVEE Next headless on this Mac -- timing, EXR output, volumetrics."""
import sys
import time

import bpy

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
out = argv[0] if argv else '/tmp/probe.exr'
res = float(argv[1]) if len(argv) > 1 else 0.25
vol = int(argv[2]) if len(argv) > 2 else 0

sc = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
sc.render.engine = 'BLENDER_EEVEE_NEXT'
sc.render.resolution_x = 1920
sc.render.resolution_y = 804
sc.render.resolution_percentage = int(res * 100)
sc.render.image_settings.file_format = 'OPEN_EXR'
sc.render.image_settings.color_depth = '32'
sc.render.image_settings.exr_codec = 'ZIP'
sc.view_settings.view_transform = 'Standard'
sc.view_settings.look = 'None'
sc.view_settings.exposure = 0
sc.view_settings.gamma = 1
sc.render.film_transparent = False
ee = sc.eevee
ee.taa_render_samples = 64
print('eevee props:', [p.identifier for p in ee.bl_rna.properties][:200])

bpy.ops.mesh.primitive_plane_add(size=40)
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
bpy.ops.object.light_add(type='POINT', location=(2, -2, 3))
bpy.context.object.data.energy = 1000
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10), rotation=(0.8, 0.2, 0.3))
bpy.context.object.data.energy = 0.3
bpy.ops.object.camera_add(location=(8, -8, 4), rotation=(1.3, 0, 0.785))
sc.camera = bpy.context.object
w = bpy.data.worlds.new('W')
sc.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.01, 0.015, 0.03, 1)
if vol:
    bpy.ops.mesh.primitive_cube_add(size=20, location=(0, 0, 5))
    m = bpy.data.materials.new('V')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.remove(nt.nodes['Principled BSDF'])
    pv = nt.nodes.new('ShaderNodeVolumePrincipled')
    pv.inputs['Density'].default_value = 0.05
    nt.links.new(pv.outputs[0], nt.nodes['Material Output'].inputs['Volume'])
    bpy.context.object.data.materials.append(m)
sc.render.filepath = out
t0 = time.time()
bpy.ops.render.render(write_still=True)
print('RENDER_TIME %.2f' % (time.time() - t0))
