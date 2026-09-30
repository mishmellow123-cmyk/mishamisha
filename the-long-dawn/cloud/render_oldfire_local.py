#!/usr/bin/env python3
"""Native local Old Fire delivery through the farm's actual JPEG encoder.

Run under the project's onepy admission wrapper. This does not call a farm API.
PNG creation retains the accepted renderer's TPDF dither; its random seed is
not fixed, so decoded delivery comparisons are reported rather than hidden.
Resume requires the renderer source to remain frozen: receipts verify output
bytes and the selected take, but do not fingerprint the renderer's source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from types import SimpleNamespace

for _pool in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_pool] = '1'

ROOT = Path(__file__).resolve().parents[1]
START, END = 5840, 6080
SHAPE = (804, 1920)
ENCODER = 'cloud.farm_node.scan'

# Annex K JPEG reference tables, scaled to the farm's quality 95. Validation
# only: all delivery encoding is performed by farm_node.scan itself.
_LUMA = (16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,
         14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,
         18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,
         49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99)
_CHROMA = (17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,
           24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99,
           99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
           99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99)
Q95_TABLES = {i: [max(1, (x*10+50)//100) for x in table]
              for i, table in enumerate((_LUMA, _CHROMA))}


def safe_output(path):
    """Reject the repository's renders directory, including symlink aliases."""
    path = Path(path).expanduser().resolve()
    forbidden = (ROOT/'renders').resolve()
    if path == forbidden or forbidden in path.parents:
        raise ValueError('renders is read-only; choose a separate delivery folder')
    return path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_jpeg(path, shape=SHAPE):
    """Decode the full file and check shape, q95 tables and 4:4:4 sampling."""
    from PIL import Image, JpegImagePlugin
    import cv2
    import numpy as np
    path = Path(path)
    data = path.read_bytes()
    with Image.open(path) as picture:
        picture.load()
        if picture.format != 'JPEG' or picture.size != (shape[1], shape[0]):
            raise ValueError(f'JPEG dimensions/format do not match: {path.name}')
        if JpegImagePlugin.get_sampling(picture) != 0:
            raise ValueError(f'JPEG must use 4:4:4 sampling: {path.name}')
        tables = picture.quantization
        if tables != Q95_TABLES:
            raise ValueError(f'JPEG must use q95 quantisation tables: {path.name}')
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != (*shape, 3):
        raise ValueError(f'JPEG decode failed: {path.name}')
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                width=shape[1], height=shape[0], decode_checked=True,
                sampling='4:4:4', quantisation=tables)


def encode_pair(png_folders, ready_dir, frame, shape=SHAPE):
    """Call the same scan function farm.py ships to its nodes; no node starts."""
    import farm_node
    unit = SimpleNamespace(
        id='local-oldfire', want={(key, frame) for key in png_folders},
        done=set(), bad={}, todo={}, t={}, shape=shape,
        outdir={key: str(Path(value).resolve()) for key, value in png_folders.items()},
        ready_dir=str(ready_dir))
    farm_node.scan(unit, final=True)
    if unit.bad or unit.done != unit.want:
        raise RuntimeError(f'Farm encoder did not complete: {unit.bad}')
    results = {}
    for key in png_folders:
        path = Path(ready_dir)/str(key)/f'f_{frame:05d}.jpg'
        results[key] = (path, validate_jpeg(path, shape))
    return results


def compare_jpegs(actual, reference, shape=SHAPE):
    import cv2
    import numpy as np
    validate_jpeg(reference, shape)
    a = cv2.imread(str(actual), cv2.IMREAD_COLOR)
    b = cv2.imread(str(reference), cv2.IMREAD_COLOR)
    delta = np.abs(a.astype(np.int16)-b.astype(np.int16))
    return dict(reference_sha256=sha256(reference), actual_sha256=sha256(actual),
                byte_equal=Path(actual).read_bytes() == Path(reference).read_bytes(),
                maximum_channel_difference=int(delta.max()),
                mean_absolute_channel_difference=float(delta.mean()),
                percentile_99=float(np.percentile(delta, 99)),
                nonzero_channel_fraction=float(np.count_nonzero(delta)/delta.size))


def frames_of(value):
    result = []
    for item in value.split(','):
        parts = item.split('-')
        if len(parts) == 1:
            result.append(int(parts[0]))
        elif len(parts) == 2:
            first, last = map(int, parts)
            if first > last:
                raise ValueError('reversed frame range')
            result.extend(range(first, last+1))
        else:
            raise ValueError('invalid frame range')
    if not result or any(not START <= frame < END for frame in result):
        raise ValueError('frames must be absolute D5840..6079')
    return sorted(set(result))


def build_renderer(caption_hold):
    sys.path.insert(0, str(ROOT/'shots/map'))
    sys.path.append(str(ROOT/'lib'))
    import book_d_v2
    import cv2
    import numba
    cv2.setNumThreads(0)
    numba.set_num_threads(1)
    return book_d_v2.make_renderer('oldfire', ppc=110, caption_hold=caption_hold)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--caption-hold', action='store_true')
    parser.add_argument('--compare-source', type=Path)
    parser.add_argument('--resume', action='store_true',
                        help='skip receipt-verified files; requires the same frozen renderer source')
    args = parser.parse_args(argv)
    try:
        frames = frames_of(args.frames)
        output = safe_output(args.out)
        matte = safe_output(output.with_name(output.name+'_matte'))
        receipt = safe_output(output.with_name(output.name+'_delivery.jsonl'))
    except ValueError as exc:
        parser.error(str(exc))
    previous = {}
    if receipt.exists():
        for line in receipt.read_text().splitlines():
            row = json.loads(line)
            previous[row['frame']] = row
    pending = []
    for frame in frames:
        paths = {key: folder/f'f_{frame:05d}.jpg'
                 for key, folder in (('rgb', output), ('matte', matte))}
        if any(path.exists() for path in paths.values()):
            if not args.resume or not all(path.is_file() for path in paths.values()):
                parser.error(f'refusing existing or incomplete frame {frame}')
            old = previous.get(frame)
            if not old or old['caption_hold'] != args.caption_hold:
                parser.error(f'no matching receipt for frame {frame}')
            for key, path in paths.items():
                if validate_jpeg(path)['sha256'] != old['outputs'][key]['sha256']:
                    parser.error(f'changed output at frame {frame}')
        else:
            pending.append(frame)
        if args.compare_source:
            for folder in (args.compare_source,
                           args.compare_source.with_name(args.compare_source.name+'_matte')):
                validate_jpeg(folder/f'f_{frame:05d}.jpg')
    if not pending:
        print(json.dumps(dict(skipped=frames, verified=True)), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    matte.mkdir(parents=True, exist_ok=True)
    renderer = build_renderer(args.caption_hold)
    import book_c as BC
    import book_c_v5 as V
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='oldfire-encode-', dir=output.parent) as temporary:
        staging = Path(temporary)
        png_folders = {key: staging/key for key in ('rgb', 'matte')}
        ready = staging/'ready'
        for frame in pending:
            started = time.perf_counter()
            hdr, alpha = renderer.frame(frame)
            if hdr.shape != (*SHAPE, 3) or alpha.shape != SHAPE:
                raise ValueError('native RGB/matte dimensions changed')
            if not np.isfinite(hdr).all() or not np.isfinite(alpha).all():
                raise ValueError('non-finite renderer output')
            array_hashes = {name: hashlib.sha256(array.tobytes()).hexdigest()
                            for name, array in (('hdr', hdr), ('alpha', alpha))}
            rgb = BC.look.finish(hdr, exposure=1.15, bloom_strength=.06,
                                 bloom_threshold=1.2, vignette_amount=.32)
            V.save(png_folders['rgb']/f'f_{frame:05d}.png', rgb, 'png')
            V.save(png_folders['matte']/f'f_{frame:05d}.png',
                   np.repeat(alpha[..., None], 3, -1), 'png')
            results = encode_pair(png_folders, ready, frame)
            row = dict(frame=frame, caption_hold=args.caption_hold, encoder=ENCODER,
                       ppc=110, detail_ppc=renderer.detail_ppc, outputs={},
                       raw_array_sha256=array_hashes)
            for key, folder in (('rgb', output), ('matte', matte)):
                path, metadata = results[key]
                target = folder/path.name
                # A late collision must not replace another writer's frame.
                with target.open('xb') as destination:
                    destination.write(path.read_bytes())
                row['outputs'][key] = metadata
                if args.compare_source:
                    ref_folder = (args.compare_source if key == 'rgb' else
                                  args.compare_source.with_name(args.compare_source.name+'_matte'))
                    metadata['farm_comparison'] = compare_jpegs(target, ref_folder/path.name)
                path.unlink()
                (png_folders[key]/f'f_{frame:05d}.png').unlink()
            row['seconds'] = time.perf_counter()-started
            row['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == 'darwin' else 1024)
            if row['peak_rss_bytes'] > 3*1024**3:
                raise RuntimeError('native process exceeded 3 GiB peak RSS')
            with receipt.open('a') as report:
                report.write(json.dumps(row, sort_keys=True)+'\n')
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
