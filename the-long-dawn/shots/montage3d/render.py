"""MONTAGE-3D render driver (run with the project venv).

  source ~/.venvs/longdawn/env.sh
  python render.py city --frames 1680,1689,1690,1693,1719 --scale 0.5 --out t_city   # low-res key frames
  python render.py city --final                       # full res, all frames -> renders/montage_v2/
  python render.py city --final --range 1690-1699     # part of a shot

Per shot: venv makes the flame sprites + timing tables (montage toolkit envelope/flicker) ->
ONE Blender process builds the scene and renders EXRs (EEVEE Next, Metal) -> venv post (sparks,
heat shimmer, distant fire glows) -> look.finish() -> look.save_png().
GPU etiquette: never two of our Blender processes (lock file); before long full-res renders wait
while another department's Blender is running (poll every 3 min) unless --force.
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(HERE, 'cache', 'numba'))
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'lib'))

import numpy as np  # noqa: E402

import exr  # noqa: E402
import fireparts as FP  # noqa: E402
import look  # noqa: E402

BLENDER = os.path.expanduser('~/Applications/Blender.app/Contents/MacOS/Blender')
FINAL_DIR = os.path.join(ROOT, 'renders', 'montage_v2')
LOCK = os.path.join(HERE, 'cache', 'blender.lock')


def other_blenders():
    try:
        out = subprocess.run(['pgrep', '-fl', 'Blender.app'], capture_output=True, text=True).stdout
    except Exception:
        return []
    mine = set()
    if os.path.exists(LOCK):
        try:
            mine.add(int(open(LOCK).read().strip()))
        except Exception:
            pass
    return [l for l in out.splitlines() if l.strip() and int(l.split()[0]) not in mine]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def acquire_lock():
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
            if pid_alive(pid):
                raise SystemExit(f'another MONTAGE-3D Blender is running (pid {pid}); not starting a second one')
        except ValueError:
            pass


def post_frame(shot, f, exr_dir, out_dir, scene, keep_exr):
    p = os.path.join(exr_dir, f'f_{f:05d}.exr')
    ch = exr.read_exr(p)
    hdr = exr.layer_rgb(ch)
    depth = ch.get('ViewLayer.Depth.Z')
    if depth is None:
        depth = np.full(hdr.shape[:2], 1e9, np.float32)
    depth = np.ascontiguousarray(depth.astype(np.float32))
    with open(os.path.join(exr_dir, f'cam_{f:05d}.json')) as fh:
        rec = json.load(fh)
    cam = FP.BCam(rec, W=hdr.shape[1], H=hdr.shape[0])
    hdr = np.ascontiguousarray(hdr)
    if hasattr(shot, 'post'):
        hdr = shot.post(f, hdr, depth, cam, scene)
    img = look.finish(hdr, **shot.FINISH)
    look.save_png(look.frame_path(out_dir, f), img)
    if not keep_exr:
        os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shot')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--range', default=None)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--out', default=None, help='test name (-> tests/<name>/)')
    ap.add_argument('--final', action='store_true', help='full res -> renders/montage_v2/')
    ap.add_argument('--keep-exr', action='store_true')
    ap.add_argument('--force', action='store_true', help='ignore other departments\' Blender processes')
    ap.add_argument('--opts', default='{}', help='json dict passed to the shot build')
    ap.add_argument('--post-only', action='store_true', help='re-finish existing EXRs (needs --keep-exr run)')
    ap.add_argument('--save-blend', default=None)
    a = ap.parse_args()
    shot = importlib.import_module(a.shot)
    if a.frames:
        frames = [int(x) for x in a.frames.split(',')]
    elif a.range:
        s, e = a.range.split('-')
        frames = list(range(int(s), int(e) + 1, a.step))
    else:
        frames = list(range(shot.START, shot.END + 1, a.step))
    scale = 1.0 if a.final else a.scale
    out_dir = FINAL_DIR if a.final else os.path.join(HERE, 'tests', a.out or f'{a.shot}_test')
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(HERE, 'cache', a.shot)
    tag = 'final' if a.final else (a.out or 'test')
    exr_dir = os.path.join(cache, f'exr_{tag}')
    os.makedirs(exr_dir, exist_ok=True)
    samples = a.samples or (shot.SAMPLES if a.final else max(16, shot.SAMPLES // 2))

    if a.post_only:
        with open(os.path.join(exr_dir, 'scene.json')) as fh:
            scene = json.load(fh)
        for f in frames:
            post_frame(shot, f, exr_dir, out_dir, scene, True)
            print('post', f, flush=True)
        return

    # ---- venv prep: flame sprites (all frames of the shot) + timing tables
    t0 = time.time()
    spr_dir = os.path.join(cache, 'sprites')
    all_frames = list(range(shot.START - 2, shot.END + 3))
    specs = shot.flame_specs()
    for spec in specs:
        need = [f for f in all_frames if not os.path.exists(os.path.join(spr_dir, f'{spec.name}_{f:05d}.exr'))]
        if need or os.environ.get('MT3D_RESPRITE'):
            FP.write_sprites(spec, all_frames if os.environ.get('MT3D_RESPRITE') else need, spr_dir)
    timing = shot.timing(all_frames)
    if hasattr(shot, 'prep'):
        timing.update(shot.prep(all_frames, cache) or {})
    timing['sprites'] = {s.name: dict(card=s.card(), first=os.path.join(spr_dir, f'{s.name}_{all_frames[0]:05d}.exr'))
                         for s in specs}
    print(f'prep {time.time() - t0:.1f}s ({len(specs)} flame sequences)', flush=True)

    job = dict(shot=a.shot, frames=frames, scale=scale, samples=samples, exr_dir=exr_dir, cache_dir=cache,
               timing=timing, final=a.final, opts=json.loads(a.opts), save_blend=a.save_blend)
    job_path = os.path.join(cache, f'job_{tag}.json')
    with open(job_path, 'w') as fh:
        json.dump(job, fh)

    # ---- GPU etiquette
    acquire_lock()
    long_job = a.final and len(frames) > 3
    while long_job and not a.force:
        others = other_blenders()
        if not others:
            break
        print(time.strftime('%H:%M:%S'), 'GPU busy (another department is rendering):', others[0][:120],
              '-- waiting 3 min', flush=True)
        time.sleep(180)

    cmd = [BLENDER, '-b', '--factory-startup', '-noaudio', '-P', os.path.join(HERE, 'bl_main.py'), '--', job_path]
    log_path = os.path.join(cache, f'blender_{tag}.log')
    t1 = time.time()
    done = []
    scene = {}
    pending = []
    lock = threading.Lock()

    def poster():
        while True:
            with lock:
                item = pending.pop(0) if pending else None
            if item is None:
                if done and done[-1] == 'END':
                    return
                time.sleep(0.5)
                continue
            try:
                post_frame(shot, item, exr_dir, out_dir, scene, a.keep_exr)
                print(f'  posted {item}', flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print('POST FAILED', item, e, flush=True)

    th = threading.Thread(target=poster, daemon=True)
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with open(LOCK, 'w') as fh:
            fh.write(str(proc.pid))
        started = False
        for line in proc.stdout:
            log.write(line)
            if line.startswith(('BUILD', 'FRAME', 'DONE', 'Traceback', 'Error', 'ERROR')) or 'Error' in line:
                print(line.rstrip(), flush=True)
            if line.startswith('FRAME'):
                if not started:
                    with open(os.path.join(exr_dir, 'scene.json')) as fh:
                        scene.update(json.load(fh))
                    th.start()
                    started = True
                with lock:
                    pending.append(int(line.split()[1]))
        proc.wait()
    try:
        os.remove(LOCK)
    except Exception:
        pass
    done.append('END')
    if started:
        th.join()
    print(f'blender {time.time() - t1:.1f}s total, exit {proc.returncode}; log {log_path}', flush=True)
    if not a.final and len(frames) > 1:
        sheet = os.path.join(out_dir, 'sheet.jpg')
        make_sheet(out_dir, frames, sheet)
        print('sheet', sheet)


def make_sheet(d, frames, out, cols=None, tw=640):
    import cv2
    tiles = []
    for f in frames:
        im = cv2.imread(look.frame_path(d, f))
        if im is None:
            continue
        th = int(round(tw * im.shape[0] / im.shape[1]))
        t = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(f), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, str(f), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return
    cols = cols or (2 if len(tiles) <= 4 else 3)
    rows = []
    for i in range(0, len(tiles), cols):
        r = tiles[i:i + cols]
        while len(r) < cols:
            r.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(r))
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == '__main__':
    main()
