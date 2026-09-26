"""Probe 2: fire-lit smoke volume (custom range), additive blended card, passes, motion blur."""
import sys
import time

import bpy

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
out = argv[0]
tile = argv[1] if len(argv) > 1 else '4'
mblur = int(argv[2]) if len(argv) > 2 else 0
res = float(argv[3]) if len(argv) > 3 else 1.0

sc = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
sc.render.engine = 'BLENDER_EEVEE_NEXT'
sc.render.resolution_x = 1920
sc.render.resolution_y = 804
sc.render.resolution_percentage = int(res * 100)
s = sc.render.image_settings
s.file_format = 'OPEN_EXR_MULTILAYER'
s.color_depth = '32'
s.exr_codec = 'ZIP'
sc.view_settings.view_transform = 'Standard'
ee = sc.eevee
ee.taa_render_samples = 64
ee.use_volume_custom_range = True
ee.volumetric_start = 3.0
ee.volumetric_end = 20.0
ee.volumetric_tile_size = tile
ee.volumetric_samples = 64
ee.use_volumetric_shadows = True
ee.use_shadows = True
ee.shadow_pool_size = '256'
sc.render.use_motion_blur = bool(mblur)
vl = sc.view_layers[0]
vl.use_pass_z = True
vl.use_pass_mist = True
try:
    vl.use_pass_position = True
    print('position pass OK')
except Exception as e:
    print('no position pass', e)


def mat_basic(name, col, rough=0.8):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*col, 1)
    b.inputs['Roughness'].default_value = rough
    return m


bpy.ops.mesh.primitive_plane_add(size=60)
bpy.context.object.data.materials.append(mat_basic('ground', (0.05, 0.05, 0.05)))
bpy.ops.mesh.primitive_cylinder_add(radius=0.29, depth=0.88, location=(0, 0, 0.44))
bpy.context.object.data.materials.append(mat_basic('drum', (0.08, 0.06, 0.05), 0.5))
# fire light
bpy.ops.object.light_add(type='POINT', location=(0, 0, 1.3))
L = bpy.context.object
L.data.energy = 800
L.data.color = (1.0, 0.5, 0.17)
L.data.shadow_soft_size = 0.25
# moon
bpy.ops.object.light_add(type='SUN', rotation=(1.2, 0.0, 2.4))
bpy.context.object.data.energy = 0.08
bpy.context.object.data.color = (0.6, 0.7, 0.9)
# smoke volume
bpy.ops.mesh.primitive_cube_add(size=1, location=(0.3, 0, 3.0))
sm = bpy.context.object
sm.scale = (2.0, 2.0, 4.0)
m = bpy.data.materials.new('smoke')
m.use_nodes = True
nt = m.node_tree
nt.nodes.remove(nt.nodes['Principled BSDF'])
pv = nt.nodes.new('ShaderNodeVolumePrincipled')
tc = nt.nodes.new('ShaderNodeTexCoord')
nz = nt.nodes.new('ShaderNodeTexNoise')
nz.noise_dimensions = '4D'
nz.inputs['Scale'].default_value = 2.5
nz.inputs['Detail'].default_value = 6
nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
nz.inputs['W'].default_value = 0.0
nz.inputs['W'].keyframe_insert('default_value', frame=1)
nz.inputs['W'].default_value = 2.0
nz.inputs['W'].keyframe_insert('default_value', frame=40)
mr = nt.nodes.new('ShaderNodeMapRange')
mr.inputs['From Min'].default_value = 0.45
mr.inputs['From Max'].default_value = 0.75
mr.inputs['To Max'].default_value = 1.5
nt.links.new(nz.outputs['Fac'], mr.inputs['Value'])
nt.links.new(mr.outputs['Result'], pv.inputs['Density'])
pv.inputs['Color'].default_value = (0.3, 0.3, 0.3, 1)
nt.links.new(pv.outputs[0], nt.nodes['Material Output'].inputs['Volume'])
sm.data.materials.append(m)
# additive blended flame card
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 1.6), rotation=(1.5708, 0, 0))
card = bpy.context.object
card.scale = (0.6, 1.2, 1)
m = bpy.data.materials.new('flamecard')
m.use_nodes = True
nt = m.node_tree
nt.nodes.remove(nt.nodes['Principled BSDF'])
tr = nt.nodes.new('ShaderNodeBsdfTransparent')
em = nt.nodes.new('ShaderNodeEmission')
add = nt.nodes.new('ShaderNodeAddShader')
tc = nt.nodes.new('ShaderNodeTexCoord')
gr = nt.nodes.new('ShaderNodeTexGradient')
gr.gradient_type = 'SPHERICAL'
nt.links.new(tc.outputs['Object'], gr.inputs['Vector'])
em.inputs['Color'].default_value = (1.0, 0.6, 0.2, 1)
mul = nt.nodes.new('ShaderNodeMath')
mul.operation = 'MULTIPLY'
mul.inputs[1].default_value = 30.0
nt.links.new(gr.outputs['Fac'], mul.inputs[0])
nt.links.new(mul.outputs[0], em.inputs['Strength'])
nt.links.new(tr.outputs[0], add.inputs[0])
nt.links.new(em.outputs[0], add.inputs[1])
nt.links.new(add.outputs[0], nt.nodes['Material Output'].inputs['Surface'])
m.surface_render_method = 'BLENDED'
card.data.materials.append(m)
card.location.x = 0.0
card.keyframe_insert('location', frame=1)
card.location.x = 0.5
card.keyframe_insert('location', frame=2)

bpy.ops.object.camera_add(location=(0, -9, 1.6), rotation=(1.5708, 0, 0))
cam = bpy.context.object
cam.data.lens = 35
sc.camera = cam
w = bpy.data.worlds.new('W')
sc.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.005, 0.008, 0.02, 1)
sc.frame_set(1)
sc.render.filepath = out
t0 = time.time()
bpy.ops.render.render(write_still=True)
print('RENDER_TIME %.2f' % (time.time() - t0))
