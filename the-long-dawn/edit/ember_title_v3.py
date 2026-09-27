"""THE LONG DAWN v3: the X3 titles of A (A20) and B (B14), made of embers: the v2 ember-title engine
(edit/ember_title.py, unchanged) played in a v3 scene (edit/title_scene.py).

    python3 edit/ember_title_v3.py --cut A          # -> renders/title_A/f_06240..06479.exr (skips if up to date)
    python3 edit/ember_title_v3.py --cut B --force
    bash edit/title_v3.sh                            # both, through the render queue (the watcher runs it)

The scene: a static camera (no crane coupling), nothing hiding the sparks, sparks thrown by the fires in the
plate (title_scene.detect_fires on the plate's first frame; the stand-in fires until the plate lands), the title
set in titles.py's v3 style (Cinzel 92, weight 500, tracking 0.28) at y 360. The engine keeps its own clock; each
v3 frame maps onto it through title_scene.clock (A: kindles, holds, crumbles by 6440; B: kindles, holds, fades
into the light 5340-5400). Output: linear-light additive layers (half-float EXR, ZIP) that the edit composites as
linear_to_srgb(soft_clip(srgb_to_linear(picture) + layer)); a new set is rendered beside the old one and swapped
in whole, so no build ever reads a half-rendered title.
"""
import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, 'edit'), os.path.join(ROOT, 'lib')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import title_scene as TS  # noqa: E402

F_PX, Z = 1600.0, 100.0                   # the static camera: focal length (px) and the fires' depth


def out_dir(cut):
    return os.path.join(ROOT, 'renders', f'title_{cut}')


def plate_fires(cut):
    """(fires, source): the fires in the delivered plate's first frame, else the stand-in fires."""
    import assemble as AS
    shot = next(s for s in AS.EDL.EDL[cut] if s['kind'] == 'title')
    for take in shot['takes']:
        p, _ = AS.locate(take, cut, None, shot['f0'])
        if p and not isinstance(p, tuple):
            im = cv2.imread(p)
            if im is None:
                continue
            if im.shape[1] != TS.W:
                im = cv2.resize(im, (TS.W, TS.H), interpolation=cv2.INTER_CUBIC)
            fires = TS.detect_fires(im[..., ::-1].astype(np.float32) / 255.0)
            rel = os.path.relpath(p, ROOT)
            if len(fires) >= 3:
                return fires, f'{rel}: {len(fires)} fires found in the plate'
            return TS.standin_fires(cut), f'{rel}: {len(fires)} fires found, so the stand-in fires throw the sparks'
    return TS.standin_fires(cut), 'stand-in fires (no plate yet)'


def signature(cut, fires):
    eng = open(os.path.join(ROOT, 'edit', 'ember_title.py')).read()
    me = open(os.path.abspath(__file__)).read()
    sc = [inspect.getsource(TS.clock), inspect.getsource(TS.fade), TS.CLOCK[cut], TS.FADE.get(cut), TS.TITLE_Y,
          TS.SPAN[cut]]
    return hashlib.sha1(json.dumps([eng, me, sc, fires], default=str).encode()).hexdigest()[:16]


def setup(cut, fires):
    """Install the v3 scene into the engine (module globals it reads at call time)."""
    import ember_title as ET
    ET.TITLE_Y = TS.TITLE_Y
    ET.crane = lambda fv2: (np.zeros(3), np.eye(3), F_PX)             # static: no crane flow, no parallax
    ET.crest_y = lambda xs, fv2: np.full(np.shape(xs), TS.H + 500.0)  # nothing hides the sparks
    ET.occluder_boxes = lambda fv2: np.zeros((0, 4), np.float64)
    ET.FIRES = np.array([[(x - TS.W / 2) * Z / F_PX, -(y - TS.H / 2) * Z / F_PX, Z, 2700.0,
                          hp * Z / (1.35 * F_PX), 0.0] for x, y, hp in fires], np.float64)
    ET.CAIRN = -1
    ET._TITLE = None
    return ET


_ET = {}


def _init(cut, fires):
    cv2.setNumThreads(1)
    _ET['et'] = setup(cut, fires)


def _frame(job):
    cut, f, path = job
    img = _ET['et'].render(TS.clock(cut, f)) * np.float32(TS.fade(cut, f))
    tmp = path[:-4] + '.tmp.exr'
    ok = cv2.imwrite(tmp, np.ascontiguousarray(img[..., ::-1]).astype(np.float32),
                     [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
                      cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
    if not ok:
        raise RuntimeError('could not write ' + tmp)
    os.replace(tmp, path)
    return f


def build(cut, workers=2, force=False):
    fires, src = plate_fires(cut)
    sig = signature(cut, fires)
    d = out_dir(cut)
    a, b = TS.SPAN[cut]
    meta_p = os.path.join(d, 'meta.json')
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    have = all(os.path.exists(os.path.join(d, f'f_{f:05d}.exr')) for f in range(a, b))
    if not force and meta.get('sig') == sig and have:
        print(f'title_{cut}: up to date ({src})', flush=True)
        return False
    new = d + '.new'
    if os.path.isdir(new):
        shutil.rmtree(new)
    os.makedirs(new)
    jobs = [(cut, f, os.path.join(new, f'f_{f:05d}.exr')) for f in range(a, b)]
    t0 = time.time()
    print(f'title_{cut}: rendering {len(jobs)} frames ({src})', flush=True)
    with Pool(workers, initializer=_init, initargs=(cut, fires)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_frame, jobs, chunksize=4)):
            if i % 60 == 0:
                print(f'  {i}/{len(jobs)} {time.time() - t0:5.0f}s', flush=True)
    json.dump(dict(sig=sig, cut=cut, span=[a, b], fires=fires, source=src, clock=TS.CLOCK[cut],
                   fade=TS.FADE.get(cut), built=time.strftime('%d %b %H:%MZ', time.gmtime())),
              open(os.path.join(new, 'meta.json'), 'w'), indent=1)
    old = d + '.old'
    if os.path.isdir(old):
        shutil.rmtree(old)
    if os.path.isdir(d):
        os.rename(d, old)
    os.rename(new, d)
    if os.path.isdir(old):
        shutil.rmtree(old)
    print(f'title_{cut}: {len(jobs)} frames in {time.time() - t0:.0f}s -> {d}', flush=True)
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cut', required=True, type=str.upper, choices=['A', 'B'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    build(a.cut, a.workers, a.force)
