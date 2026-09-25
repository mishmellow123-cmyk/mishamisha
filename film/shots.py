"""
The edit: nine camera setups over one continuous world. Each shot is a time
range of the world's timeline and a camera function of time.

    render:  python3 shots.py SHOT OUTDIR [preview]
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import world as Wd  # noqa: E402

SPHERE = np.array([1.1113, -1.1169, 16.5234])


def lerp(a, b, f):
    return np.asarray(a, float) * (1 - f) + np.asarray(b, float) * f


def orbit(radius, azimuth_deg, height, center=(0.0, 0.0, 0.0)):
    a = math.radians(azimuth_deg)
    c = np.asarray(center, float)
    return c + np.array([radius * math.cos(a), radius * math.sin(a), height])


def growth_front(t):
    """Tree arc length reached by the bulk of the growing fibres."""
    return max(0.0, Wd.V_TREE * (t - Wd.T_TREE - 0.9))


# every camera returns (location, target, lens, roll)
def cam_question(t, w):
    f = C.ease_io(t / 4.6, 1.3)
    z_s = w.spark_z(max(t, 0.35))
    tgt = np.array([0.0, 0.0, 0.85 * z_s + 0.10])
    loc = lerp([0.30, -5.2, 0.75], [0.12, -3.6, 0.42], f)
    return loc, tgt, 40.0, 0.0


def cam_rings(t, w):
    f = C.ease_io((t - 4.6) / 6.0, 1.4)
    loc = lerp([0.0, -6.0, 8.2], [1.8, -11.5, 13.5], f)
    return loc, np.array([0.0, 0.0, 0.0]), lerp(34.0, 28.0, f), 0.0


def cam_voices(t, w):
    f = C.ease_io((t - 10.6) / 5.6, 1.2)
    az = -120.0 + 62.0 * f
    loc = orbit(lerp(10.5, 7.6, f), az, lerp(3.0, 2.1, f))
    tgt = orbit(3.0, az + 70.0, 0.1)
    return loc, tgt, 28.0, math.radians(-5.0)


def cam_rising(t, w):
    f = C.ease_io((t - 16.2) / 6.8, 1.3)
    front = growth_front(t)
    tip = w.path_point(min(front, 7.5))
    az = -30.0 + 38.0 * f
    loc = orbit(lerp(6.5, 8.0, f), az, lerp(1.4, 6.0, f))
    tgt = lerp(np.array([0.0, 0.0, 1.0]), tip, 0.75)
    return loc, tgt, 30.0, 0.0


def cam_branching(t, w):
    f = C.ease_io((t - 23.0) / 6.2, 1.5)
    az = 22.0 + 26.0 * f
    loc = orbit(lerp(12.5, 22.0, f), az, lerp(3.2, 5.5, f), center=(0.4, -0.4, 0.0))
    tgt = np.array([0.4, -0.4, lerp(5.8, 8.4, f)])
    return loc, tgt, lerp(26.0, 22.0, f), 0.0


def cam_answer(t, w):
    """A slow crane outside the crown, rising with the answer's light."""
    e = max(w.answer_at(t), 0.0)
    here = w.path_point(e)
    f = C.ease_io((t - 29.2) / 8.4, 1.1)
    az = -62.0 + 34.0 * f
    loc = orbit(10.5, az, here[2] + 1.3, center=(0.5, -0.5, 0.0))
    tgt = here + np.array([0.0, 0.0, 0.35])
    return loc, tgt, 32.0, 0.0


D_TOP = np.array([0.69, 0.92, 3.98]) / np.linalg.norm([0.69, 0.92, 3.98]) * 3.8


def cam_sphere(t, w):
    f = C.ease_io((t - 37.6) / 6.0, 1.6)
    start = SPHERE + np.array([2.7, -2.3, 0.3])
    loc = lerp(start, SPHERE + D_TOP, f)
    return loc, SPHERE, lerp(40.0, 46.0, f), 0.0


def cam_held(t, w):
    f = C.ease_io((t - 43.6) / 3.6, 1.8)
    loc = SPHERE + D_TOP * lerp(1.0, 0.70, f)
    return loc, SPHERE, lerp(46.0, 50.0, f), 0.0


def cam_stillness(t, w):
    f = C.ease_io((t - 47.2) / 12.8, 1.0)
    loc = np.array([0.01, -0.02, lerp(29.0, 46.0, f)])
    return loc, np.array([0.0, 0.0, 0.0]), 30.0, math.radians(-12.0 * f)


SHOTS = {
    # name: (t0, t1, camera)
    "01_question": (0.0, 4.6, cam_question),
    "02_rings": (4.6, 10.6, cam_rings),
    "03_voices": (10.6, 16.2, cam_voices),
    "04_rising": (16.2, 23.0, cam_rising),
    "05_branching": (23.0, 29.2, cam_branching),
    "06_answer": (29.2, 37.6, cam_answer),
    "07_sphere": (37.6, 43.6, cam_sphere),
    "08_held": (43.6, 47.2, cam_held),
    "09_stillness": (47.2, 60.0, cam_stillness),
}


def frame_range(name):
    t0, t1, _ = SHOTS[name]
    return range(int(round(t0 * C.FPS)), int(round(t1 * C.FPS)))


if __name__ == "__main__":
    import bpy
    name = sys.argv[1]
    out = sys.argv[2]
    preview = len(sys.argv) > 3 and sys.argv[3] == "preview"
    width = int(os.environ.get("RW", 640 if preview else 1600))
    height = int(os.environ.get("RH", 272 if preview else 680))
    C.reset(samples=int(os.environ.get("SAMPLES", 4 if preview else 8)), width=width, height=height)
    bpy.context.scene.cycles.adaptive_threshold = 0.05
    world = Wd.World()
    cam = C.make_camera(35)
    t0, t1, camf = SHOTS[name]

    def update(t, f):
        loc, tgt, lens, roll = camf(t, world)
        cam.location = tuple(loc)
        C.look_at(cam, tgt, roll)
        cam.data.lens = lens
        world.update(t, cam)

    frames = list(frame_range(name))
    if preview:
        n = int(os.environ.get("NPREV", 4))
        frames = [frames[int(i * (len(frames) - 1) / (n - 1))] for i in range(n)]
    C.render_frames(out, frames, update)
