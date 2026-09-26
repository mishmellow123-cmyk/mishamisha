"""Render key frames of a variant at a scale into a labelled sheet. usage: keys.py V out.jpg frames scale"""
import os, sys, time
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
sys.path.insert(0, os.path.expanduser('~/mishamisha/the-long-dawn/shots/accord'))
import numpy as np, cv2
import accord as A, scene as SC
V, out = sys.argv[1], sys.argv[2]
frames = [float(x) for x in sys.argv[3].split(',')]
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
SC.set_variant(V)
tiles = []
for t in frames:
    t0 = time.time()
    hdr, info = A.render_frame(t, scale)
    img = A.finish(hdr, t)
    im8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    cv2.putText(im8, f'{V} {int(t)}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im8, f'{V} {int(t)}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(im8)
    print(V, int(t), round(time.time() - t0, 1), 's', flush=True)
cols = 4
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
