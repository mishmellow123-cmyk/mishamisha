"""Evidence crops for A's measured events: the delivered picture around a point, frame by frame, as one sheet.

    cd the-long-dawn
    .../tools/onepy ~/ldfarm/venv/bin/python music/src/a_evidence_crops.py spec.json out.png

spec.json: [{"label": "ridge.34", "x": 696, "y": 211, "frames": [3685, 3686, 3688], "half": 20}, ...] (half-size
pixels). Frames are read one at a time through EDIT (AS._CTX.picture: take, finish, EDIT windows; no captions) at
half size; each crop is enlarged 4x (nearest) with its frame number burnt in. This is how a measured onset is looked
at by eye (the OBSERVED basis); it measures nothing itself. An entry may name "take": a candidate stem under renders/
read in place of the cut's take for its frames (as measure_a_points.py --take), and "zoom" (1 with a large "half"
gives a whole-frame view).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main(spec_path, out_path):
    sys.path.insert(0, str(ROOT / 'edit'))
    import assemble as AS
    AS._init('A', None, 0.5, True, True)
    spec = json.loads(Path(spec_path).read_text())
    need = sorted({(f, s.get('take')) for s in spec for f in s['frames']}, key=lambda v: (v[0], v[1] or ''))
    tiles = {}
    ctx = AS._CTX
    accepted = list(ctx.plans)
    for f, take in need:                              # one finished frame in memory at a time
        i, _ = ctx.shot_at(f)
        ctx.plans[i] = accepted[i] if not take else dict(
            kind='take', have=0, alt=0, take=dict(stem=take, off=0, mode='exact', note='evidence override', crop=None,
                                                  grade=None, matte=None, under=None, video=None, need=None, add=None,
                                                  final_eligible=False))
        img, _, status, _ = ctx.picture(f)
        pic = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1]
        del img
        for k, s in enumerate(spec):
            if f not in s['frames'] or s.get('take') != take:
                continue
            zoom = s.get('zoom', 4)
            h = s.get('half', 20)
            x, y = int(round(s['x'])), int(round(s['y']))
            crop = np.zeros((2 * h, 2 * h, 3), np.uint8)
            x0, y0 = x - h, y - h
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(pic.shape[1], x0 + 2 * h), min(pic.shape[0], y0 + 2 * h)
            if sx1 > sx0 and sy1 > sy0:
                crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = pic[sy0:sy1, sx0:sx1]
            if 'half_y' in s:
                crop = crop[max(0, h - s['half_y']):h + s['half_y']]
            crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, str(f), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles[(k, f)] = crop
        del pic
        ctx.plans[i] = accepted[i]
    rows = []
    width = max(len(s['frames']) for s in spec)
    for k, s in enumerate(spec):
        h = tiles[(k, s['frames'][0])].shape[0]
        label = np.zeros((h, 110, 3), np.uint8)
        cv2.putText(label, s['label'], (3, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cells = [label] + [tiles[(k, f)] for f in s['frames']]
        cells += [np.zeros_like(cells[-1])] * (width - len(s['frames']))
        rows.append(np.hstack([np.pad(c, ((0, 0), (0, 2), (0, 0))) for c in cells]))
    w = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 2), (0, w - r.shape[1]), (0, 0))) for r in rows])
    cv2.imwrite(str(out_path), sheet)
    print(out_path, sheet.shape)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
