"""FINISH look-dev: before/after pairs of the hero frames, exactly as the edit shows them (edit/assemble.py
Ctx.picture: take, crop, per-cut grade, book matte; no titles), through the candidate looks.

    python3 finish/lookdev.py                          # all heroes, all looks -> _local_logs/review/finish/
    python3 finish/lookdev.py --only A1 --looks 500T_2383
    python3 finish/lookdev.py --flicker A 1640 48      # 48-frame flicker check on a moving shot

Outputs: pairs/<hero>_<look>.jpg (full-res crop, BEFORE | AFTER), sheet_<cut>.jpg (every hero x every look, reduced),
stills/<hero>_{src,<look>}.png (full frames), flicker_<cut>_<f0>.txt/.png.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'edit'))
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(HERE, '.numba'))
import filmfinish as FF  # noqa: E402

OUT = os.path.expanduser('~/mishamisha/_local_logs/review/finish')

# hero: (id, cut, cut frame, label, detail crop (x, y, w, h) in pixels of the 1920x804 picture, look family)
HEROES = [
    ('A1', 'A', 1160, 'A5 embers_A3 f1160', None, 'film'),
    ('A2', 'A', 1700, 'A6 embers_A3 f1700', None, 'film'),
    ('A3', 'A', 3604, 'A H1 re-key h1_v3h5 src 1480 (roar)', None, 'film'),
    ('A4', 'A', 3830, 'A KARST v3 (karst_slow src 30)', None, 'film'),
    ('A5', 'A', 3890, 'A DESERT v3 (desert src 30)', None, 'film'),
    ('A6', 'A', 1400, 'A6 embers_A3 f1400 (towers)', None, 'film'),
    ('B1', 'B', 120, 'B1 dusk_B f120', None, 'film'),
    ('B2', 'B', 330, 'B1 dusk_B f330', None, 'film'),
    ('B3', 'B', 600, 'B1 dusk_B f600', None, 'film'),
    ('B4', 'B', 1178, 'B H1 crop, strike 3 (src 1294)', None, 'film'),
    ('B5', 'B', 1316, 'B H1 crop, the catch (src 1432)', None, 'film'),
    ('B6', 'B', 1364, 'B H1 roar crop (src 1480)', None, 'film'),
    ('C1', 'C', 175, 'MAP-L v3 book_C f175 (farm test, 19:57Z)', None, 'film', '_farmtest/map_v3_book/book_C'),
    ('C2', 'C', 728, 'MAP-L v3 book_C f728: the letters kindle (farm test)', None, 'film',
     '_farmtest/map_v3_book/book_C'),
    ('C3', 'C', 6600, 'MAP-L v3 book_C f6600: the Havens page (farm test)', None, 'film',
     '_farmtest/map_v3_book/book_C'),
    ('C4', 'C', 3316, 'C H1-C the catch, parchment grade (src 1432)', None, 'film'),
    ('C5', 'C', 114, 'RUN-C ink reveal_h f114 (farm test; plain sRGB, no Hill curve)', None, 'ink',
     '_farmtest/runC_r2/runC_reveal_h'),
    ('C6', 'C', 2480, 'RUN-C ink illum_d f2480 (farm test; plain sRGB, no Hill curve)', None, 'ink',
     '_farmtest/runC_r1/runC_illum_d'),
]
FILM_LOOKS = ['500T_2383', '500T_2383_fire', '250D_2383_fire', '500T_2393_fire']
INK_LOOKS = ['ink_grain', '500T_2383']

_CTX = {}


def picture(cut, f):
    import assemble as AS
    if cut not in _CTX:
        _CTX[cut] = AS.Ctx(cut, scale=1.0, clean=True)
    img, shot, status, src = _CTX[cut].picture(f)
    return np.clip(img, 0, 1).astype(np.float32), f"{shot['code']} {status}"


def raw_frame(stem, f):
    """A source frame straight from renders/<stem>/ (for heroes outside the current EDL)."""
    p = os.path.join(ROOT, 'renders', stem, f'f_{f:05d}.png')
    if not os.path.exists(p):
        p = p[:-4] + '.jpg'
    return cv2.imread(p)[..., ::-1].astype(np.float32) / 255.0


def to8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def save(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, to8(img)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93] if path.endswith('.jpg') else [])


def label(img8, text, scale=0.55, y=22):
    text = text.replace('\u00b7', '-')
    cv2.putText(img8, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img8, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)
    return img8


def detail_box(img, box=None, w=960, h=540):
    """The detail crop: the given box, else centred on the brightest warm region (the fire)."""
    H, W = img.shape[:2]
    if box is None:
        lin = img ** 2.2
        warm = np.clip(lin[..., 0] - lin[..., 2], 0, None) + 0.3 * lin.mean(axis=2)
        warm = cv2.GaussianBlur(warm, (0, 0), 25)
        cy, cx = np.unravel_index(np.argmax(warm), warm.shape)
        x0 = int(np.clip(cx - w // 2, 0, W - w))
        y0 = int(np.clip(cy - h // 2, 0, H - h))
        box = (x0, y0, w, h)
    return box


def crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def run(heroes, looks_film, looks_ink, overrides=None):
    os.makedirs(OUT, exist_ok=True)
    cache = {}
    looks = {}
    import filmfast
    for n in sorted(set(looks_film) | set(looks_ink)):
        looks[n] = filmfast.Fast(FF.make(n, **(overrides or {})))          # the delivery path itself
        print(f'look {n}: EV gain {np.log2(looks[n].look.gain):+.2f}', flush=True)
    rows = {}
    for hero in heroes:
        hid, cut, f, lab, box, fam = hero[:6]
        t0 = time.time()
        if len(hero) > 6:
            src, status = raw_frame(hero[6], f), f'raw renders/{hero[6]}'
        else:
            src, status = picture(cut, f)
        save(os.path.join(OUT, 'stills', f'{hid}_src.png'), src)
        names = looks_ink if fam == 'ink' else looks_film
        box = detail_box(src, box)
        outs = []
        for n in names:
            out = looks[n](src, cut, f)
            save(os.path.join(OUT, 'stills', f'{hid}_{n}.png'), out)
            pair = np.concatenate([crop(src, box), np.ones((box[3], 8, 3), np.float32), crop(out, box)], axis=1)
            p8 = to8(pair)
            label(p8, f'{hid} BEFORE  ({lab}; {status})')
            cv2.putText(p8, f'AFTER {n}', (box[2] + 18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(p8, f'AFTER {n}', (box[2] + 18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1,
                        cv2.LINE_AA)
            os.makedirs(os.path.join(OUT, 'pairs'), exist_ok=True)
            cv2.imwrite(os.path.join(OUT, 'pairs', f'{hid}_{n}.jpg'), p8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            outs.append((n, out))
        rows.setdefault(cut, []).append((hid, lab, src, outs))
        print(f'{hid} {cut} f{f} {status}: {time.time() - t0:.1f} s', flush=True)
    for cut, rr in rows.items():
        sheet(cut, rr)


def sheet(cut, rows, tw=480):
    th = int(round(tw * 804 / 1920))
    ncol = 1 + max(len(r[3]) for r in rows)
    pad = 6
    img = np.full((len(rows) * (th + pad) + 40, ncol * (tw + pad), 3), 18, np.uint8)
    film = {'A': 'A · EVERY STEP CLOSER', 'B': 'B · THE VIGIL', 'C': 'C · THE LAST PAGES'}[cut]
    label(img, f'FINISH look-dev {film}: BEFORE (as the edit shows it) | AFTER per look', 0.7, 28)
    for i, (hid, lab, src, outs) in enumerate(rows):
        y = 40 + i * (th + pad)
        for j, (n, im) in enumerate([('BEFORE', src)] + outs):
            t = to8(cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA))
            label(t, f'{hid} {n}' if j else f'{hid} BEFORE: {lab}', 0.45, 16)
            img[y:y + th, j * (tw + pad):j * (tw + pad) + tw] = t
    cv2.imwrite(os.path.join(OUT, f'sheet_{cut}.jpg'), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])


def flicker(cut, f0, n, look_name):
    """Mean luminance per frame, source vs finished, over n consecutive frames of a moving shot, and the grain's own
    frame-to-frame behaviour: a finish that flickers adds frame-to-frame variance the source does not have."""
    import filmfast
    L = filmfast.Fast(FF.make(look_name))                     # the delivery path itself
    rows = []
    prev = None
    for f in range(f0, f0 + n):
        src, status = picture(cut, f)
        out = L(src, cut, f)
        ls = float((src ** 2.2).mean())
        lo = float((out ** 2.2).mean())
        g = cv2.cvtColor(to8(out), cv2.COLOR_RGB2GRAY).astype(np.float32)
        corr = None
        if prev is not None:
            a = g - cv2.GaussianBlur(g, (0, 0), 2)
            b = prev - cv2.GaussianBlur(prev, (0, 0), 2)
            corr = float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean() + 1e-9))
        prev = g
        rows.append((f, ls, lo, corr, status))
        print(f'{f} src {ls:.5f} out {lo:.5f} ratio {lo / max(ls, 1e-9):.4f} hp-corr {corr}', flush=True)
    r = np.array([x[2] / max(x[1], 1e-9) for x in rows])
    d = np.abs(np.diff(r))
    dls = np.diff(np.log([max(x[1], 1e-9) for x in rows]))
    dlo = np.diff(np.log([max(x[2], 1e-9) for x in rows]))
    rep = [f'flicker check {cut} {f0}-{f0 + n - 1} ({n} frames), look {look_name}',
           f'out/src mean-luminance ratio: mean {r.mean():.4f}, std {r.std():.5f}, max step {d.max():.5f}',
           f'frame-to-frame |dlog L|: source max {np.abs(dls).max():.4f}; finished max {np.abs(dlo).max():.4f}; '
           f'ADDED by the finish max {np.abs(dlo - dls).max():.5f} ({100 * np.abs(dlo - dls).max():.3f}%), '
           f'mean {np.abs(dlo - dls).mean():.5f}',
           f'frame-to-frame high-pass correlation of the finished frames (grain renewal; ~source motion only): '
           f'median {np.median([x[3] for x in rows[1:]]):.3f}']
    for f, ls, lo, c, st in rows:
        rep.append(f'{f} {st} src {ls:.5f} out {lo:.5f} ratio {lo / max(ls, 1e-9):.4f} corr {c}')
    open(os.path.join(OUT, f'flicker_{cut}_{f0}.txt'), 'w').write('\n'.join(rep) + '\n')
    # plot
    W, H = 960, 300
    img = np.full((H, W, 3), 20, np.uint8)
    ls = np.array([x[1] for x in rows])
    lo = np.array([x[2] for x in rows])
    top = max(ls.max(), lo.max()) * 1.1
    for arr, col in ((ls, (160, 160, 160)), (lo, (80, 170, 255))):
        pts = np.stack([np.linspace(20, W - 20, len(arr)), H - 20 - arr / top * (H - 60)], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
    label(img, f'{cut} {f0}-{f0 + n - 1}: mean luminance, grey = source, orange = {look_name}; '
               f'ratio std {r.std():.5f}', 0.5, 20)
    cv2.imwrite(os.path.join(OUT, f'flicker_{cut}_{f0}.png'), img)
    print('\n'.join(rep[:4]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--looks', default='')
    ap.add_argument('--flicker', nargs=3, metavar=('CUT', 'F0', 'N'))
    ap.add_argument('--flicker-look', default='500T_2383')
    a = ap.parse_args()
    if a.flicker:
        flicker(a.flicker[0], int(a.flicker[1]), int(a.flicker[2]), a.flicker_look)
        sys.exit(0)
    hs = [h for h in HEROES if not a.only or h[0] in a.only.split(',')]
    lf = a.looks.split(',') if a.looks else FILM_LOOKS
    li = [x for x in (a.looks.split(',') if a.looks else INK_LOOKS)]
    run(hs, lf, li)
