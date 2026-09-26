"""Before/after review sheets for the v2 EMBERS work (writes to ~/mishamisha/_local_logs/review/).

    python review_sheets.py            # embers_v2.jpg (towers + vortex, v1 vs v2) and embers_C.jpg (Ring, Eye, grasp)
"""
import os

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.expanduser('~/mishamisha/_local_logs/review')
TW = 640


def load(dept, f):
    p = os.path.join(ROOT, 'renders', dept, 'f_%05d.png' % f)
    im = cv2.imread(p)
    if im is None:
        im = np.full((804, 1920, 3), 30, np.uint8)
        cv2.putText(im, 'MISSING %s %d' % (dept, f), (500, 420), cv2.FONT_HERSHEY_SIMPLEX, 3, (80, 80, 200), 4)
    return cv2.resize(im, (TW, int(TW * 804 / 1920)), interpolation=cv2.INTER_AREA)


def label(im, text):
    cv2.putText(im, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return im


def header(text, w):
    h = np.full((34, w, 3), 18, np.uint8)
    cv2.putText(h, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    return h


def pairs(rows_spec, before, after, title, cols=2):
    blocks = [header(title, TW * 2 * cols)]
    row = []
    for f, note in rows_spec:
        a = label(load(before, f), '%d  v1 %s' % (f, note))
        b = label(load(after, f), '%d  v2' % f)
        row.append(np.hstack([a, b]))
        if len(row) == cols:
            blocks.append(np.hstack(row))
            row = []
    if row:
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        blocks.append(np.hstack(row))
    return np.vstack(blocks)


def grid(frames, dept, title, cols=4, prefix=''):
    blocks = [header(title, TW * cols)]
    ims = [label(load(dept, f), '%s%d' % (prefix, f)) for f in frames]
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    for i in range(0, len(ims), cols):
        blocks.append(np.hstack(ims[i:i + cols]))
    return np.vstack(blocks)


def main():
    os.makedirs(OUT, exist_ok=True)
    fire = [(500, ''), (520, ''), (546, ''), (570, '')]
    towers = [(640, '(crown)'), (700, '(crown)'), (760, ''), (1000, '(grasp backdrop)')]
    vortex = [(812, ''), (824, ''), (850, ''), (870, '')]
    globe = [(890, ''), (916, ''), (932, ''), (946, '')]
    a = pairs(fire, 'embers', 'embers_v2', 'THE THINKING FIRE  (left: v1 delivered, right: v2 -- a flame, not a bulb)')
    b = pairs(towers, 'embers', 'embers_v2', 'TOWERS OF EMBERS, THE CROWN\'S LICKING TONGUES')
    c = pairs(vortex, 'embers', 'embers_v2', 'THE VORTEX  (v1 flat smear ~805-825 -> v2 opens toward the lens)')
    d = pairs(globe, 'embers', 'embers_v2', 'THE EMBER GLOBE  (v1 framed East Asia -> v2 from above the Arctic, fire reaches all at once)')
    im = np.vstack([a, b, c, d])
    im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, 'embers_v2.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    ring = grid([592, 600, 604, 612, 620, 628, 640, 700, 760], 'embers_C', 'CUT C: THE RING  (forged from the thinking '
                'fire 596-622, inscription burns in 616-642, floats above the race)', cols=3)
    eye = grid([832, 840, 846, 852, 862, 879], 'embers_C', 'CUT C: THE EYE  (resolves 836-858, holds to the cut at 880)',
               cols=3)
    grasp = grid([964, 990, 1016, 1024, 1030, 1034], 'embers_C', 'CUT C: THE GRASP ON THE RING', cols=3)
    im = np.vstack([ring, eye, grasp])
    im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, 'embers_C.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('wrote', os.path.join(OUT, 'embers_v2.jpg'), os.path.join(OUT, 'embers_C.jpg'))


if __name__ == '__main__':
    main()
