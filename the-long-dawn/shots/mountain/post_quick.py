"""EXR -> PNG through the project finish (for tests). python post_quick.py in.exr out.png [exposure]"""
import os
import sys

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lib'))
import look

src, dst = sys.argv[1], sys.argv[2]
exp = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
im = cv2.imread(src, cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1].astype(np.float32)
out = look.finish(im, exposure=exp, bloom_strength=0.08, bloom_threshold=0.8, vignette_amount=0.2)
look.save_png(dst, out)
