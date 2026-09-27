"""FINISH: one page to pick the look from: 1:1 crops of the moments that decide it (fire colour, grain in flat
mid-tones, shadows), BEFORE beside every candidate look, plus the fire-hue numbers. Reads lookdev.py's stills.

    python3 finish/picker.py            -> _local_logs/review/finish/PICK_the_look.jpg + fire_hue.txt
"""
import os

import cv2
import numpy as np

OUT = os.path.expanduser('~/mishamisha/_local_logs/review/finish')
ST = os.path.join(OUT, 'stills')
LOOKS = ['500T_2383', '500T_2383_fire', '250D_2383_fire', '500T_2393_fire']
# (hero, what it shows, crop box x, y, w, h at 1:1)
ROWS = [('A3', 'A H1 roar: fire colour', (560, 180, 420, 260)),
        ('A4', 'A KARST: the beacon and the mist (grain in mid-tones)', (1240, 330, 420, 260)),
        ('A2', 'A embers f1700: tower fires on black', (760, 360, 420, 260)),
        ('B6', 'B roar crop: fire colour', (700, 250, 420, 260)),
        ('B1', 'B dusk f120: the sky (grain, colour)', (700, 0, 420, 260)),
        ('B4', 'B H1 crop: glove, sparks, shadows', (1000, 300, 420, 260)),
        ('C2', 'C book: the letters kindle', (700, 300, 420, 260))]


def lab(img, text, y=18, s=0.5):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, s, (240, 240, 240), 1, cv2.LINE_AA)


def main():
    tiles = []
    for hid, what, (x, y, w, h) in ROWS:
        p = os.path.join(ST, f'{hid}_src.png')
        if not os.path.exists(p):
            continue
        row = []
        for n in ['src'] + LOOKS:
            q = os.path.join(ST, f'{hid}_{n}.png')
            im = cv2.imread(q) if os.path.exists(q) else np.zeros((804, 1920, 3), np.uint8)
            t = np.ascontiguousarray(im[y:y + h, x:x + w])
            lab(t, f'{hid} BEFORE' if n == 'src' else n)
            row.append(np.pad(t, ((0, 6), (0, 6), (0, 0)), constant_values=20))
        r = np.concatenate(row, 1)
        band = np.full((26, r.shape[1], 3), 20, np.uint8)
        lab(band, f'{hid}: {what}  (1:1 crops)', 18, 0.55)
        tiles.append(np.concatenate([band, r], 0))
    sheet = np.concatenate(tiles, 0)
    head = np.full((40, sheet.shape[1], 3), 20, np.uint8)
    lab(head, 'FINISH: pick ONE look. BEFORE | 500T>2383 | 500T>2383 +fire | 250D>2383 +fire | 500T>2393 +fire', 26,
        0.7)
    cv2.imwrite(os.path.join(OUT, 'PICK_the_look.jpg'), np.concatenate([head, sheet], 0),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    # fire hue numbers
    lines = ['mean hue (deg) and saturation of the warm, bright pixels of each hero (source hue < 60, sat > 0.2, '
             'value > 0.5); lower hue = more orange, higher = more yellow', '',
             f'{"hero":5} {"pixels":>8}  {"BEFORE":>12}  ' + '  '.join(f'{n:>15}' for n in LOOKS)]
    for hid in sorted({f.split('_')[0] for f in os.listdir(ST) if f.endswith('_src.png')}):
        src = cv2.imread(os.path.join(ST, f'{hid}_src.png'))[..., ::-1].astype(np.float32) / 255
        hs = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
        m = (hs[..., 0] < 60) & (hs[..., 1] > 0.2) & (hs[..., 2] > 0.5)
        if m.sum() < 200:
            continue
        cells = [f'{hs[..., 0][m].mean():5.1f} ({hs[..., 1][m].mean():.2f})']
        for n in LOOKS:
            q = os.path.join(ST, f'{hid}_{n}.png')
            if not os.path.exists(q):
                cells.append(' ' * 15)
                continue
            o = cv2.cvtColor(cv2.imread(q)[..., ::-1].astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
            cells.append(f'{o[..., 0][m].mean():5.1f} ({o[..., 1][m].mean():.2f})')
        lines.append(f'{hid:5} {int(m.sum()):8d}  {cells[0]:>12}  ' + '  '.join(f'{c:>15}' for c in cells[1:]))
    open(os.path.join(OUT, 'fire_hue.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
