"""Before/after review sheet: the same frames from two render folders side by side, labelled.

  python shots/run/review_sheet.py --before <dir> --after renders/run_v2 --out review/run_fix_before_after.jpg
"""
import argparse
import os

import cv2
import numpy as np


def _thumb(path, w, h):
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((h, w, 3), np.uint8)
        cv2.putText(img, 'missing', (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 1, cv2.LINE_AA)
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True)
    ap.add_argument('--after', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', default='1520,1540,1560,1580,1592,1600,1612,1620,1640,1660,1679')
    ap.add_argument('--width', type=int, default=800)
    ap.add_argument('--labels', default='before: v2 finals,after: run_fix')
    a = ap.parse_args()
    frames = [int(f) for f in a.frames.split(',')]
    w = a.width
    h = int(round(w * 804 / 1920))
    gap, head, side = 8, 44, 70
    lb, la = a.labels.split(',')
    W = side + 2 * w + 3 * gap
    H = head + len(frames) * (h + gap) + gap
    sheet = np.full((H, W, 3), 16, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sheet, lb, (side + gap + 6, 30), font, 0.8, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(sheet, la, (side + 2 * gap + w + 6, 30), font, 0.8, (220, 220, 220), 1, cv2.LINE_AA)
    for i, f in enumerate(frames):
        y = head + i * (h + gap)
        cv2.putText(sheet, str(f), (8, y + 28), font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
        sheet[y:y + h, side + gap:side + gap + w] = _thumb(os.path.join(a.before, f'f_{f:05d}.png'), w, h)
        x = side + 2 * gap + w
        sheet[y:y + h, x:x + w] = _thumb(os.path.join(a.after, f'f_{f:05d}.png'), w, h)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(a.out, sheet.shape)


if __name__ == '__main__':
    main()
