"""Render key frames at low res and assemble a labelled sheet (quick review)."""
import os, sys, time
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import accord as A, scene as SC
from scene import look
frames = [float(x) for x in sys.argv[1].split(',')]
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
tiles = []
for t in frames:
    t0 = time.time()
    hdr, info = A.render_frame(t, scale)
    img = A.finish(hdr, t)
    im8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    cv2.putText(im8, str(int(t)), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(im8)
    print(int(t), round(time.time() - t0, 1), 's', flush=True)
cols = 4
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
os.makedirs(os.path.join(A.out_dir(), 'tests'), exist_ok=True); cv2.imwrite(os.path.join(A.out_dir(), 'tests', 'keys.png'), np.vstack(rows))
