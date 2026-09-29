"""C v5 THE WATCH, 4480..4719: the existing ink scroll with every beacon already lit.

This separate driver leaves the accepted reveal/scroll/illumination unchanged.
It uses the last 240 camera poses of the original 320-frame scroll, with the
same world-space beacon sites and fire drawings. Only ignition times change.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

FIRST = 4480
LAST = 4719
SOURCE_FIRST = 80
ALREADY_LIT = -400.0


def source_frame(frame):
    if not FIRST <= frame <= LAST:
        raise ValueError(f"Watch frame {frame} outside {FIRST}..{LAST}")
    return int(frame) - FIRST + SOURCE_FIRST


def settled_beacons(beacons):
    """Copy the shared catalogue: no caller can retime the original scroll."""
    result = np.array(beacons, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 7 or not len(result):
        raise ValueError("Expected a nonempty seven-column beacon catalogue")
    result[:, 3] = ALREADY_LIT
    return result


def make_shot():
    import render_ink as RI
    shot = RI.Shot('scroll')
    shot.B = settled_beacons(shot.B)
    return shot


def render(frame, shot, scale=0.5, ss=2.0):
    import cv2
    import render_ink as RI
    import inkpass as IP
    # Keep the entire existing camera/fire/noise clock coherent in source time.
    source = source_frame(frame)
    aov = RI.render_aov(shot, source, scale, ss)
    aov['kpx'] = scale * ss
    image, _ = IP.compose(aov, B=shot.B, CR=shot.CR)
    width, height = int(round(1920 * scale)), int(round(804 * scale))
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if image.shape != (height, width, 3) or not np.isfinite(image).all():
        raise ValueError("Invalid watch render")
    return image


def write_image(path, image):
    import cv2
    from PIL import Image
    rgb = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    temporary = path.with_name(path.stem + '.part' + path.suffix)
    if path.suffix == '.jpg':
        Image.fromarray(rgb).save(temporary, format='JPEG', quality=95, subsampling=0)
    elif not cv2.imwrite(str(temporary), rgb[..., ::-1]):
        raise OSError(f"Failed to write {temporary}")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--frames', help='Comma-separated absolute C frames')
    selection.add_argument('--range', help='Inclusive absolute C frame range')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--ss', type=float, default=2.0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--format', choices=('png', 'jpg'), default='jpg')
    args = parser.parse_args()
    if args.scale <= 0 or args.ss <= 0:
        parser.error('scale and ss must be positive')
    if args.threads < 1:
        parser.error('threads must be positive')
    try:
        if args.frames:
            frames = [int(f) for f in args.frames.split(',')]
        else:
            lo, hi = (int(v) for v in args.range.split('-'))
            frames = list(range(lo, hi + 1))
        if not frames or len(frames) != len(set(frames)):
            raise ValueError('Supply distinct frames or a nonempty increasing range')
        for frame in frames:
            source_frame(frame)
    except ValueError as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [args.out / f'f_{frame:05d}.{args.format}' for frame in frames]
    if any(path.exists() for path in paths):
        parser.error('Output frame already exists; select a fresh review/final directory')
    receipt = dict(status='initializing', scale=args.scale, ss=args.ss,
                   source_camera_frames=[source_frame(f) for f in frames], frames=[])
    receipt_path = args.out / f'receipt_{frames[0]:05d}_{frames[-1]:05d}.json'
    if receipt_path.exists():
        parser.error('Receipt already exists; select a fresh review/final directory')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    try:
        os.environ.setdefault('NUMBA_NUM_THREADS', str(args.threads))
        import numba
        import cv2
        shot = make_shot()
        cv2.setNumThreads(0)
        numba.set_num_threads(min(args.threads, numba.config.NUMBA_NUM_THREADS))
        receipt.update(status='running',
                       catalogue_sha256=hashlib.sha256(shot.B.tobytes()).hexdigest(),
                       beacon_count=len(shot.B), ignition_frame=ALREADY_LIT,
                       numba_threads=numba.get_num_threads(), cv_threads=cv2.getNumThreads())
        for frame, path in zip(frames, paths):
            started = time.perf_counter()
            image = render(frame, shot, args.scale, args.ss)
            write_image(path, image)
            elapsed = time.perf_counter() - started
            receipt['frames'].append(dict(frame=frame, seconds=elapsed,
                shape=list(image.shape), file=path.name,
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                float_rgb_sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()))
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
            print(f'watch C{frame} source {source_frame(frame)}: {elapsed:.3f}s {path}', flush=True)
        receipt['status'] = 'complete'
    except BaseException:
        receipt['status'] = 'failed'
        receipt['traceback'] = traceback.format_exc()
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
