"""Review sheet of delivered frames. python make_sheet.py <renders subdir> <first> <last> <step>
<out.jpg> [cols] [thumb_w]"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sub, a, b, step, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cols = int(sys.argv[6]) if len(sys.argv) > 6 else 4
tw = int(sys.argv[7]) if len(sys.argv) > 7 else 640
th = int(round(tw * 804 / 1920))
frames = list(range(a, b + 1, step))
if frames[-1] != b:
    frames.append(b)
ims = []
for f in frames:
    im = cv2.imread(os.path.join(ROOT, 'renders', sub, 'f_%05d.png' % f))
    if im is None:
        im = np.zeros((804, 1920, 3), np.uint8)
    t = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.putText(t, str(f), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    ims.append(t)
while len(ims) % cols:
    ims.append(np.zeros_like(ims[0]))
rows = [np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]
os.makedirs(os.path.dirname(out), exist_ok=True)
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(out, len(frames), 'frames')
