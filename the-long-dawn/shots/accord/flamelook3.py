"""Look-dev for the v3 torch flame: one flame from the side and from above (black ground), a few frames."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import cv2
import scene3 as SC
import flame3 as FL3
import nbcore
from scene3 import look


def cam_look(C, T, f, W, H):
    C = np.asarray(C, float)
    Fw = np.asarray(T, float) - C
    Fw /= np.linalg.norm(Fw)
    up = np.array([0, 0, 1.0]) if abs(Fw[2]) < 0.95 else np.array([1.0, 0, 0])
    R = np.cross(Fw, up)
    R /= np.linalg.norm(R)
    U = np.cross(R, Fw)
    return np.concatenate([C, R, U, Fw, [f, W / 2, H / 2]])


def render(views, frames, out, lean=(0.22, -0.10), Hf=0.36, Rf=0.08, I=26.0):
    n3 = nbcore.make_noise3(64, 7)
    W, H = 240, 300
    tiles = []
    for f in frames:
        row = []
        for (C, T) in views:
            img = np.zeros((H, W, 3), np.float32)
            depth = np.full((H, W), 1e6, np.float32)
            FL = np.array([[0.0, 0.0, 0.0, Hf, Rf, lean[0], lean[1], I, 1.3, 1.0]])
            cam = cam_look(C, T, 900.0, W, H)
            FL3.torch_flames(img, depth, cam, FL, 1, float(f), n3, 0.25)
            im = look.finish(img, exposure=1.2, bloom_strength=0.07, bloom_threshold=0.9, vignette_amount=0.0)
            row.append((np.clip(im, 0, 1) * 255).astype(np.uint8)[..., ::-1])
        tiles.append(np.concatenate(row, 1))
    cv2.imwrite(out, np.concatenate(tiles, 0))
    print(out)


if __name__ == '__main__':
    side = ((0.0, -1.6, 0.18), (0.0, 0.0, 0.18))
    top = ((0.0 + 1.6 * math.sin(math.radians(12)), 0.0, 1.6 * math.cos(math.radians(12))), (0.0, 0.0, 0.1))
    render([side, top], [4960, 4963, 4970], sys.argv[1] if len(sys.argv) > 1 else 'flamelook.jpg')
