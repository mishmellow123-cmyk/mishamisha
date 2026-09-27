"""Key-frame grid for an animated render folder, with optional beat labels.

  python tests/strip_sheet.py <out.jpg> <title> <dir> <frame[:label],frame[:label],...> [cols] [tile_w]
"""
import os
import sys

import cv2
import numpy as np


def main():
    out, title, d, spec = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    cols = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    tw = int(sys.argv[6]) if len(sys.argv) > 6 else 480
    tiles = []
    for item in spec.split(','):
        f, _, lab = item.partition(':')
        f = int(f)
        im = cv2.imread(os.path.join(d, f'f_{f:05d}.png'))
        if im is None:
            im = np.zeros((804, 1920, 3), np.uint8)
        t = cv2.resize(im, (tw, int(round(tw * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
        txt = f'{f}  {lab}'.strip()
        for c, w in (((0, 0, 0), 3), ((235, 235, 235), 1)):
            cv2.putText(t, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, w, cv2.LINE_AA)
        tiles.append(t)
    rows = []
    for i in range(0, len(tiles), cols):
        r = tiles[i:i + cols]
        while len(r) < cols:
            r.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(r))
    sheet = np.vstack(rows)
    head = np.full((30, sheet.shape[1], 3), 16, np.uint8)
    for c, w in (((0, 0, 0), 3), ((240, 240, 240), 1)):
        cv2.putText(head, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, w, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out)


if __name__ == '__main__':
    main()
