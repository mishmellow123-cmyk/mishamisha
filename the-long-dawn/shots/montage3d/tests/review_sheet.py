"""Review sheet for the director: v1 | v2 key frames side by side + a 1:1 full-res crop of v2.

  python tests/review_sheet.py <shot> <out.jpg> <frames,comma> <crop_frame> <x0,y0,w,h> [v2_dir]
  (v2_dir: e.g. a test folder of full-res stills; default renders/montage_v2)
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
V1 = os.path.join(ROOT, 'renders', 'montage')
V2 = os.path.join(ROOT, 'renders', 'montage_v2')
TW = 720


def tile(path, label):
    im = cv2.imread(path)
    if im is None:
        im = np.zeros((804, 1920, 3), np.uint8)
    th = int(round(TW * im.shape[0] / im.shape[1]))
    t = cv2.resize(im, (TW, th), interpolation=cv2.INTER_AREA)
    for c, w in (((0, 0, 0), 3), ((235, 235, 235), 1)):
        cv2.putText(t, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, w, cv2.LINE_AA)
    return t


def main():
    shot, out, frames, cf, box = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    global V2
    if len(sys.argv) > 6:
        V2 = sys.argv[6]
    rows = []
    for f in [int(x) for x in frames.split(',')]:
        rows.append(np.hstack([tile(os.path.join(V1, f'f_{f:05d}.png'), f'v1  {f}'),
                               tile(os.path.join(V2, f'f_{f:05d}.png'), f'v2  {f}')]))
    x0, y0, w, h = (int(v) for v in box.split(','))
    im = cv2.imread(os.path.join(V2, f'f_{cf:05d}.png'))
    crop = im[y0:y0 + h, x0:x0 + w]
    s = 2 * TW / crop.shape[1]
    crop = cv2.resize(crop, (2 * TW, int(round(crop.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else
                      cv2.INTER_CUBIC)
    for c, wd in (((0, 0, 0), 3), ((235, 235, 235), 1)):
        cv2.putText(crop, f'v2 {cf}  full-res detail ({w}x{h} px crop)', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c,
                    wd, cv2.LINE_AA)
    rows.append(crop)
    sheet = np.vstack(rows)
    cv2.putText(sheet, '', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, sheet.shape)


if __name__ == '__main__':
    main()
