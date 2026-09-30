"""Bounded, reproducible assembler evidence for the A ignition polish lane.

Public commands acquire onepy themselves; do not wrap the orchestration in onepy.
`batch` assembles at most 12 output frames; `movie` runs such batches serially.
The hidden --worker flag is reserved for the child launched through onepy.
All frame numbers are absolute A timeline frames; range ends are exclusive.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
OWNER = Path(os.environ.get('LD_OWNER_OUT', REPO.parent.parent / 'outputs' / 'claude-owner-night'))  # host-specific: set LD_OWNER_OUT
ONEPY = OWNER / 'tools/onepy'
OUTPUT = OWNER / 'codex/polishign'
PYTHON = Path(os.environ.get('LD_PYTHON', sys.executable))
BASE_MODULES = ('afix_comp', 'edl_v3', 'assemble')
FPS = 24
LIMIT = 12


def frames_arg(value):
    frames = [int(x) for x in value.split(',') if x.strip()]
    if not frames or len(frames) != len(set(frames)) or min(frames) < 0:
        raise argparse.ArgumentTypeError('frames must be distinct nonnegative integers')
    return frames


def baseline_source(name, ref='HEAD'):
    return subprocess.check_output(
        ['git', 'show', f'{ref}:the-long-dawn/edit/{name}.py'], cwd=REPO)


def source_identity(version, ref='HEAD'):
    """Record exact source bytes, including added compositor modules in current view."""
    files = {}
    for path in sorted((ROOT / 'edit').glob('*.py')):
        data = baseline_source(path.stem, ref) if version == 'before' and path.stem in BASE_MODULES else path.read_bytes()
        # New modules are unused by the historical assembler, so do not let their
        # arrival invalidate already rendered historical evidence.
        if version == 'before' and path.stem not in BASE_MODULES:
            tracked = subprocess.run(['git', 'cat-file', '-e', f'{ref}:the-long-dawn/edit/{path.name}'],
                                     cwd=REPO, capture_output=True)
            if tracked.returncode:
                continue
        files[path.name] = hashlib.sha256(data).hexdigest()
    commit = subprocess.check_output(['git', 'rev-parse', ref], cwd=REPO, text=True).strip()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'version': version, 'reference': commit, 'source_sha256': digest, 'files': files}


def load_assembler(version, ref='HEAD'):
    sys.path.insert(0, str(ROOT / 'edit'))
    sys.path.insert(0, str(ROOT / 'lib'))
    if version == 'after':
        return importlib.import_module('assemble')
    for name in BASE_MODULES:
        module = types.ModuleType(name)
        # Relative resource lookups must stay rooted in this worktree, not /tmp.
        module.__file__ = str(ROOT / 'edit' / f'{name}.py')
        sys.modules[name] = module
        exec(compile(baseline_source(name, ref), module.__file__, 'exec'), module.__dict__)
    return sys.modules['assemble']


def checked_context(assembler):
    assembler._init('A', None, 1 / 3, True, True)
    ctx = assembler._CTX
    original_read, original_take = ctx.read, ctx.take_frame

    def read(ref, *args, **kwargs):
        value = original_read(ref, *args, **kwargs)
        if value is None:
            raise OSError(f'Cannot decode required source: {ref}')
        return value

    def take(take_spec, f):
        value = original_take(take_spec, f)
        if value is None:
            raise OSError(f'Missing source at A{f}: {take_spec["stem"]}')
        return value

    def slate(i, status):
        raise OSError(f'Refusing SLATE for {ctx.shots[i]["sec"]}: {status}')

    ctx.read, ctx.take_frame, ctx.slate = read, take, slate
    return ctx


def paths(out, version, frame):
    folder = out / f'{version}_frames'
    return folder / f'f_{frame:05d}.png', folder / f'f_{frame:05d}.json'


def read_record(out, version, frame):
    png, metadata = paths(out, version, frame)
    if not png.is_file() or not metadata.is_file():
        raise FileNotFoundError(f'Missing validated {version} frame {frame}')
    record = json.loads(metadata.read_text())
    if record.get('png_sha256') != hashlib.sha256(png.read_bytes()).hexdigest():
        raise ValueError(f'Stale or modified image: {png}')
    return png, record


def reusable(out, version, frame, identity):
    try:
        _, record = read_record(out, version, frame)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return record.get('source_sha256') == identity['source_sha256']


def batch(args):
    if len(args.frames) > LIMIT:
        raise ValueError(f'At most {LIMIT} assembler frames per process')
    from PIL import Image
    identity = source_identity(args.version, args.ref)
    assembler = load_assembler(args.version, args.ref)
    ctx = checked_context(assembler)
    started = time.monotonic()
    for frame in args.frames:
        png, metadata = paths(args.out, args.version, frame)
        png.parent.mkdir(parents=True, exist_ok=True)
        # An interrupted/failed rerender must never validate an old PNG.
        metadata.unlink(missing_ok=True)
        t = assembler.transition_at('A', frame)
        if t and assembler.transition_layers(t, frame) is None:
            raise OSError(f'Missing transition layer at A{frame}')
        rgb = ctx.frame(frame)
        if rgb.shape != (268, 640, 3):
            raise ValueError(f'Unexpected assembler dimensions: {rgb.shape}')
        tmp = png.with_name(f'{png.stem}.{os.getpid()}.part.png')
        Image.fromarray(rgb).save(tmp)
        os.replace(tmp, png)
        record = dict(identity, frame=frame, fps=FPS, width=640, height=268,
                      finish=True, captions=True, png_sha256=hashlib.sha256(png.read_bytes()).hexdigest())
        tmp_metadata = metadata.with_name(f'{metadata.stem}.{os.getpid()}.part.json')
        tmp_metadata.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(tmp_metadata, metadata)
    if source_identity(args.version, args.ref)['source_sha256'] != identity['source_sha256']:
        for frame in args.frames:
            paths(args.out, args.version, frame)[1].unlink(missing_ok=True)
        raise RuntimeError('Source changed during this batch; validation markers removed; rerun')
    print(json.dumps({'version': args.version, 'frames': args.frames,
                      'elapsed_seconds': round(time.monotonic() - started, 3)}), flush=True)


def run_worker(argv):
    subprocess.run([str(ONEPY), str(PYTHON), str(Path(__file__).resolve()), '--worker', *argv], check=True)


def movie(args):
    if args.end <= args.start:
        raise ValueError('end must exceed start')
    identity = source_identity(args.version, args.ref)
    sequence = range(args.start, args.end)
    pending = [f for f in sequence if not args.reuse or not reusable(args.out, args.version, f, identity)]
    started = time.monotonic()
    for j in range(0, len(pending), LIMIT):
        run_worker(['batch', '--version', args.version, '--ref', args.ref, '--out', str(args.out),
                    '--frames', ','.join(map(str, pending[j:j + LIMIT]))])
    # Every frame, including reused evidence, must match this source snapshot.
    for frame in sequence:
        if not reusable(args.out, args.version, frame, identity):
            raise RuntimeError(f'Source or image changed during movie render at frame {frame}; rerun')
    if args.wav:
        wav = args.wav.resolve(strict=True)
        output = args.out / f'{args.version}_{args.start}_{args.end - 1}.mp4'
        temp = output.with_suffix('.part.mp4')
        duration = (args.end - args.start) / FPS
        command = [str(ONEPY), 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                   '-framerate', str(FPS), '-start_number', str(args.start),
                   '-i', str(args.out / f'{args.version}_frames/f_%05d.png'),
                   '-ss', f'{args.start / FPS:.9f}', '-t', f'{duration:.9f}', '-i', str(wav),
                   '-map', '0:v:0', '-map', '1:a:0', '-frames:v', str(args.end - args.start),
                   '-t', f'{duration:.9f}', '-c:v', 'libx264', '-threads', '1', '-preset', 'medium',
                   '-crf', '18', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709',
                   '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', '192k',
                   '-movflags', '+faststart', str(temp)]
        subprocess.run(command, check=True)
        os.replace(temp, output)
        probe = json.loads(subprocess.check_output([
            str(ONEPY), 'ffprobe', '-v', 'error', '-count_frames', '-show_streams',
            '-show_format', '-of', 'json', str(output)], text=True))
        videos = [s for s in probe['streams'] if s['codec_type'] == 'video']
        audios = [s for s in probe['streams'] if s['codec_type'] == 'audio']
        if len(videos) != 1 or not audios or int(videos[0]['nb_read_frames']) != len(sequence):
            raise RuntimeError(f'Encoded movie did not retain expected picture/audio: {output}')
        report = dict(identity, start=args.start, end_exclusive=args.end, frames=len(sequence), fps=FPS,
                      wav=str(wav), video=str(output), elapsed_seconds=time.monotonic() - started,
                      ffprobe=probe)
        output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'video': str(output), 'frames': len(sequence),
                          'video_duration': videos[0]['duration'], 'audio_duration': audios[0].get('duration'),
                          'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    else:
        print(json.dumps({'frames_saved': len(sequence), 'elapsed_seconds': time.monotonic() - started}), flush=True)


def gray(rgb):
    import cv2
    return cv2.cvtColor(cv2.resize(rgb, (320, 134), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


def load_rgb(out, version, frame):
    import numpy as np
    from PIL import Image
    png, record = read_record(out, version, frame)
    with Image.open(png) as im:
        rgb = np.asarray(im.convert('RGB')).copy()
    return rgb, record


def measure(args):
    import numpy as np
    records = []
    controls = set(args.controls or [])
    for frame in sorted(set(args.frames) | controls):
        views, item = {}, {'frame': frame, 'control': frame in controls}
        for version in ('before', 'after'):
            rgb, record = load_rgb(args.out, version, frame)
            g = gray(rgb).astype(np.float32)
            view = {'mean_gray_320x134': float(g.mean()), 'step_mad_gray_320x134': None,
                    'step_from_frame': None, 'source_sha256': record['source_sha256']}
            try:
                prev, prior_record = load_rgb(args.out, version, frame - 1)
            except FileNotFoundError:
                pass
            else:
                if prior_record['source_sha256'] != record['source_sha256']:
                    raise ValueError(f'Mixed source revisions at {version} {frame - 1}/{frame}')
                view['step_mad_gray_320x134'] = float(np.abs(g - gray(prev).astype(np.float32)).mean())
                view['step_from_frame'] = frame - 1
            item[version] = view
            views[version] = rgb
        difference = np.abs(views['after'].astype(np.int16) - views['before'].astype(np.int16))
        item.update(bit_identical=bool(not difference.any()), max_channel_delta=int(difference.max()),
                    changed_pixels=int(np.any(difference, axis=2).sum()),
                    before_after_mad_rgb=float(difference.mean()))
        records.append(item)
    doc = {'measurement': '8-bit sRGB PNG, OpenCV INTER_AREA resize then RGB2GRAY; adjacent absolute frames only',
           'frames': records, 'controls_all_identical': all(x['bit_identical'] for x in records if x['control'])
           if controls else None}
    path = args.out / f'{args.name}_measurements.json'
    path.write_text(json.dumps(doc, indent=2) + '\n')
    print(path)
    if controls and not doc['controls_all_identical']:
        raise SystemExit('Negative-control frames changed; see measurement JSON')


def sheet(args):
    from PIL import Image, ImageDraw
    columns = 2
    tile_w, tile_h = 480, 225
    canvas = Image.new('RGB', (columns * tile_w, len(args.frames) * tile_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for row, frame in enumerate(args.frames):
        for col, version in enumerate(('before', 'after')):
            png, _ = read_record(args.out, version, frame)
            with Image.open(png) as image:
                tile = image.convert('RGB').resize((480, 201), Image.Resampling.LANCZOS)
            canvas.paste(tile, (col * tile_w, row * tile_h + 24))
            draw.text((col * tile_w + 8, row * tile_h + 5), f'{version.upper()}   A frame {frame}   {frame / FPS:.3f}s',
                      fill=(230, 230, 230))
    path = args.out / f'{args.name}_contact.png'
    canvas.save(path)
    print(path)


def master(args):
    """Read-only source-state comparison; codec/scaling differences remain in these deltas."""
    import numpy as np
    from PIL import Image, ImageDraw
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,r_frame_rate,codec_name', '-of', 'json', str(args.master)], text=True))
    stream = probe['streams'][0]
    if stream['r_frame_rate'] != '24/1':
        raise ValueError(f'Master is not on the expected 24 fps grid: {stream}')
    folder = args.out / 'owner_master_frames'
    folder.mkdir(exist_ok=True)
    canvas = Image.new('RGB', (960, len(args.frames) * 225), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    records = []
    for row, frame in enumerate(args.frames):
        # Seek to a whole-second boundary, then select an integer frame offset.
        # This avoids fractional seek rounding at the 24 fps frame boundary.
        data = subprocess.check_output([
            'ffmpeg', '-v', 'error', '-threads', '1', '-ss', str(frame // FPS), '-i', str(args.master),
            '-vf', f'select=eq(n\\,{frame % FPS}),scale=640:268:flags=area',
            '-frames:v', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
        if len(data) != 640 * 268 * 3:
            raise RuntimeError(f'Master frame {frame} did not decode completely')
        owner = np.frombuffer(data, np.uint8).reshape((268, 640, 3))
        before, baseline_record = load_rgb(args.out, 'before', frame)
        Image.fromarray(owner).save(folder / f'f_{frame:05d}.png')
        delta = np.abs(owner.astype(np.int16) - before.astype(np.int16))
        records.append({'frame': frame, 'master_vs_baseline_mad_rgb': float(delta.mean()),
                        'master_mean_gray': float(gray(owner).mean()),
                        'baseline_mean_gray': float(gray(before).mean()),
                        'baseline_source_sha256': baseline_record['source_sha256']})
        for col, (label, rgb) in enumerate((('OWNER MASTER', owner), ('HEAD ASSEMBLER', before))):
            tile = Image.fromarray(rgb).resize((480, 201), Image.Resampling.LANCZOS)
            canvas.paste(tile, (col * 480, row * 225 + 24))
            draw.text((col * 480 + 8, row * 225 + 5), f'{label}   A frame {frame}', fill=(230, 230, 230))
    canvas.save(args.out / 'owner_master_vs_baseline_contact.png')
    path = args.out / 'owner_master_vs_baseline.json'
    path.write_text(json.dumps({'master': str(args.master), 'probe': probe, 'frames': records,
                               'limitation': 'Master codec and scaling differences are included; '
                               'this is a source-state comparison, not a bit-identity test.'}, indent=2) + '\n')
    print(path)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('batch', 'movie', 'measure', 'sheet', 'master'):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, default=OUTPUT)
        if name in ('batch', 'movie'):
            p.add_argument('--version', choices=('before', 'after'), required=True)
            p.add_argument('--ref', default='HEAD')
        if name in ('batch', 'measure', 'sheet', 'master'):
            p.add_argument('--frames', type=frames_arg, required=True)
        if name in ('measure', 'sheet'):
            p.add_argument('--name', required=True)
        if name == 'measure':
            p.add_argument('--controls', type=frames_arg)
        if name == 'master':
            p.add_argument('--master', type=Path, default=OWNER / 'delivery/A_master.mov')
        if name == 'movie':
            p.add_argument('--start', type=int, default=960)
            p.add_argument('--end', type=int, default=1453)
            p.add_argument('--wav', type=Path)
            p.add_argument('--reuse', action='store_true')
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    if args.command in ('batch', 'master') and len(args.frames) > LIMIT:
        raise SystemExit(f'At most {LIMIT} assembler frames per process')
    if args.command != 'movie' and not args.worker:
        run_worker(argv)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
