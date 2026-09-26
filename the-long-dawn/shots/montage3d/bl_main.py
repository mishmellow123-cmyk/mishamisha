"""Blender entry point for MONTAGE-3D (runs inside Blender 4.5, headless).

  Blender -b --factory-startup -P bl_main.py -- <job.json>

job.json: {shot, frames, scale, samples, exr_dir, cache_dir, opts}
Builds the shot's scene (kit + shot module), then renders each frame to <exr_dir>/f_%05d.exr and
writes <exr_dir>/cam_%05d.json (exact camera for post) and <exr_dir>/scene.json (fire positions...).
"""
import importlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402

from kit import core as C  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    with open(argv[0]) as f:
        job = json.load(f)
    shot = importlib.import_module(job['shot'])
    t0 = time.time()
    C.reset()
    info = shot.build(job)
    sc = bpy.context.scene
    print(f'BUILD {time.time() - t0:.1f}s', flush=True)
    os.makedirs(job['exr_dir'], exist_ok=True)
    C.write_json(os.path.join(job['exr_dir'], 'scene.json'), info or {})
    if job.get('save_blend'):
        bpy.ops.wm.save_as_mainfile(filepath=job['save_blend'])
    for f in job['frames']:
        t1 = time.time()
        sc.frame_set(f)
        if hasattr(shot, 'per_frame'):
            shot.per_frame(job, f)
        cam = sc.camera
        C.write_json(os.path.join(job['exr_dir'], f'cam_{f:05d}.json'), C.camera_record(cam))
        sc.render.filepath = os.path.join(job['exr_dir'], f'f_{f:05d}.exr')
        bpy.ops.render.render(write_still=True)
        print(f'FRAME {f} {time.time() - t1:.1f}s', flush=True)
    print(f'DONE {time.time() - t0:.1f}s', flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    os._exit(1)
