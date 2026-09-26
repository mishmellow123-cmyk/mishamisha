"""Finish EXRs in a dir through look.finish and build a labelled sheet.
python sheet.py <dir> <out.jpg> [exposure] [cols] [width]"""
import glob
import os
import sys

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lib'))
import look

d, out = sys.argv[1], sys.argv[2]
exp = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 2
tw = int(sys.argv[5]) if len(sys.argv) > 5 else 960
ims = []
for p in sorted(glob.glob(os.path.join(d, 'f_*.exr'))):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1].astype(np.float32)
    s = look.finish(im, exposure=exp, bloom_strength=0.08, bloom_threshold=0.8, vignette_amount=0.2)
    png = p[:-4] + '.png'
    look.save_png(png, s)
    t = cv2.resize(cv2.imread(png), (tw, int(tw * 804 / 1920)), interpolation=cv2.INTER_AREA)
    cv2.putText(t, os.path.basename(p)[2:7], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    ims.append(t)
while len(ims) % cols:
    ims.append(np.zeros_like(ims[0]))
rows = [np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
