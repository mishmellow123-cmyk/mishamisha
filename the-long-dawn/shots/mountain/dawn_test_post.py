"""Apply the dawn post to test EXRs and build a sheet. python dawn_test_post.py DIR out.jpg [exp]"""
import glob, os, sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post_run as PR
d, out = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    PR.FINS['dawn']['exposure'] = float(sys.argv[3])
ims = []
for p in sorted(glob.glob(os.path.join(d, 'f_*.exr'))):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1].astype(np.float32)
    if im.shape[1] != 1920:
        im = cv2.resize(im, (1920, 804), interpolation=cv2.INTER_LINEAR)
    o = PR.process(im, 'dawn')
    PR.look.save_png(p[:-4] + '.png', o)
    t = cv2.resize(cv2.imread(p[:-4] + '.png'), (960, 402), interpolation=cv2.INTER_AREA)
    cv2.putText(t, os.path.basename(p)[2:7], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    ims.append(t)
while len(ims) % 2:
    ims.append(np.zeros_like(ims[0]))
cv2.imwrite(out, np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, len(ims), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
