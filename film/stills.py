"""Render stills of the world: python3 stills.py OUTDIR "t,x,y,z,tx,ty,tz,lens" ..."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import world as Wd  # noqa: E402

out = sys.argv[1]
views = [list(map(float, v.split(","))) for v in sys.argv[2:]]
C.reset(samples=int(os.environ.get("SAMPLES", 6)), width=int(os.environ.get("RW", 800)), height=int(os.environ.get("RH", 340)))
t0 = time.time()
world = Wd.World()
print("world built", round(time.time() - t0, 1), "sphere at", world.sphere_c, "answer done", round(world.t_sphere, 2))
cam = C.make_camera(35)
os.makedirs(out, exist_ok=True)
for i, (t, x, y, z, tx, ty, tz, lens) in enumerate(views):
    cam.location = (x, y, z)
    C.look_at(cam, (tx, ty, tz))
    cam.data.lens = lens
    world.update(t, cam)
    C.set_clocks(t)
    import bpy
    bpy.context.scene.render.filepath = os.path.join(out, f"{i:02d}_t{t:05.1f}.exr")
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print("view", i, "t", t, round(time.time() - t1, 1), "s", flush=True)
