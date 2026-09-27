"""Look-dev for the v3 hearth fire (the fire everyone lit): the fire volume alone over a dark disc, from a
P2 and a P3 camera, at a few frames.  python shots/accord/firelook3.py out.jpg [frames]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import cv2
import accord3 as A
import scene3 as SC
import flame3 as FL3
from scene3 import look


def render(frames, out, scale=0.35):
    R = A.resources()
    tiles = []
    for f in frames:
        t = float(f)
        fs, HP, CF = A.hearth_state(t)
        cam = SC.camera(t, scale)
        Wd, Hd = A.frame_size(scale)
        depth = np.full((Hd, Wd), 1e6, np.float32)
        # a dark ground plane for depth
        yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float64)
        f_ = cam[12]
        sx = (xx + 0.5 - cam[13]) / f_
        sy = -(yy + 0.5 - cam[14]) / f_
        dz = cam[11] + sx * cam[5] + sy * cam[8]
        dx = cam[9] + sx * cam[3] + sy * cam[6]
        dy = cam[10] + sx * cam[4] + sy * cam[7]
        l = np.sqrt(dx * dx + dy * dy + dz * dz)
        tg = -cam[2] / (dz / l)
        depth[:] = np.where(dz < 0, tg, 1e6)
        img = np.zeros((Hd, Wd, 3), np.float32) + np.array([0.004, 0.003, 0.002], np.float32)
        fb = np.zeros_like(img)
        FL3.hearth_volume(Wd, Hd, cam, HP, R['ang'], R['ang'].shape[0], R['noise3'], depth, fb, 40, CF, CF.shape[0])
        img += fb
        im = look.finish(img, exposure=SC.exposure(t), bloom_strength=0.07, bloom_threshold=0.9, vignette_amount=0.0)
        im8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
        cv2.putText(im8, str(f), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        tiles.append(im8)
    rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)]
    cv2.imwrite(out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(out)


if __name__ == '__main__':
    fr = [int(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else [5230, 5300, 5560, 5580]
    render(fr, sys.argv[1] if len(sys.argv) > 1 else 'firelook.jpg')
