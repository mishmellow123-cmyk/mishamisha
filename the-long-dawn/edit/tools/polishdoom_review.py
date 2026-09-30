"""Bounded, finished-assembler evidence for A2400–3359.

Run probes through the owner's onepy wrapper. Every probe refuses >12 frames;
``batches`` dispatches separate wrapped processes and never decodes a frame itself.
The before control executes all three historical edit modules from a pinned ref.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
BASELINE = '37e724baae5919b36b5ecf669c30aac9c757724f'
# host-specific: set LD_OWNER_OUT to the owner-night outputs folder (its tools/onepy guards memory)
OWNER = Path(os.environ.get('LD_OWNER_OUT', REPO.parent.parent / 'outputs' / 'claude-owner-night'))
ONEPY = OWNER / 'tools' / 'onepy'
# The moments of this lane (polishdoom): every beat of A8, both camera cuts, the white, the valley, the black, the
# ember, the piano's entry and the stars' return, each with the frames either side that show its shape.
GROUPS = {'2400_rim': [2399, 2400, 2401, 2402, 2403],
          '2420_pulse': [2419, 2420, 2421, 2422, 2423], '2440_pulse': [2439, 2440, 2441, 2442, 2443],
          '2460_cut': [2459, 2460, 2461, 2463, 2466],
          '2480_crown': [2479, 2480, 2481, 2482, 2485], '2500_pulse': [2499, 2500, 2501, 2503],
          '2520_updraft': [2519, 2520, 2521, 2522, 2523, 2525],
          '2540_pulse': [2539, 2540, 2541, 2542, 2544], '2560_pulse': [2559, 2560, 2561, 2562, 2564],
          '2580_pulse': [2579, 2580, 2581, 2582, 2584], '2600_fall': [2599, 2600, 2601, 2602, 2604, 2606],
          '2620_pulse': [2619, 2620, 2621, 2622, 2624],
          '2640_white': [2634, 2635, 2636, 2637, 2638, 2639, 2640],
          '2660_valley': [2640, 2641, 2648, 2656, 2660, 2664, 2672, 2684],
          '2800_black': [2762, 2780, 2796, 2799, 2800, 2801, 2820],
          '2836_ember': [2831, 2832, 2835, 2836, 2840, 2850],
          '2859_piano': [2858, 2859, 2860, 2862],
          '3120_adaptation': [3119, 3120, 3200, 3359]}
PROBE = sorted({f for frames in GROUPS.values() for f in frames})
OUTSIDE = [950, 1038, 1839, 1920, 2398, 2399, 3360, 3361, 4000, 4800, 5840, 6400]
AUDIO_GROUPS = {'2400_rim': (2380, 2450, [2400, 2420, 2440]),
                '2460_cut': (2440, 2490, [2460, 2480]),
                '2520_updraft': (2500, 2560, [2520, 2540]),
                '2600_fall': (2580, 2645, [2600, 2620, 2637, 2640]),
                '2640_impact': (2626, 2660, [2637, 2640]),
                '2660_valley': (2636, 2700, [2640, 2660, 2684]),
                '2800_black': (2750, 2850, [2762, 2796, 2800, 2832]),
                '2836_ember': (2820, 2870, [2832, 2836, 2848, 2860]),
                '2859_piano': (2845, 2930, [2860, 2920]),
                '3120_adaptation': (3090, 3200, [3120, 3130]),
                'range_2380_3380': (2380, 3380, [2400, 2460, 2520, 2600, 2640, 2660, 2800, 2832, 2860, 3120])}
Y = np.array([.2126, .7152, .0722], np.float32)


def chunks(frames, size=12):
    if not 1 <= size <= 12:
        raise ValueError('Assembler batch size must be 1..12')
    frames = sorted(set(frames))
    return [frames[n:n + size] for n in range(0, len(frames), size)]


def measured(rgb, previous=None):
    a = rgb.astype(np.float32)
    y = a @ Y
    result = {'mean_y': float(y.mean()), 'mean_rgb': a.mean(axis=(0, 1)).tolist(),
              'min_y': float(y.min()), 'max_y': float(y.max()),
              'white_fraction_y250': float((y >= 250).mean()),
              'nonzero_fraction': float(np.any(rgb != 0, axis=2).mean()),
              'sha256_rgb': hashlib.sha256(rgb.tobytes()).hexdigest()}
    if previous is not None:
        if previous.shape != rgb.shape:
            raise ValueError('Consecutive frames must have the same dimensions')
        before = previous.astype(np.float32)
        result['step_rgb_mad'] = float(np.abs(a - before).mean())
        result['step_y_mad'] = float(np.abs(y - before @ Y).mean())
        result['mean_y_delta'] = float((y - before @ Y).mean())
    return result


def code_source(name, ref):
    if ref:
        return subprocess.check_output(['git', 'show', f'{ref}:the-long-dawn/edit/{name}.py'],
                                       cwd=REPO)
    return (ROOT / 'edit' / f'{name}.py').read_bytes()


def assembler(ref=None):
    sys.path.insert(0, str(ROOT / 'edit'))
    hashes = {}
    for name in ('afix_comp', 'edl_v3', 'assemble'):
        source = code_source(name, ref)
        hashes[name] = hashlib.sha256(source).hexdigest()
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / 'edit' / f'{name}.py')
        sys.modules[name] = module
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module, hashes


def file_state(path):
    p = Path(path)
    stat = p.stat()
    return {'path': str(p), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def probe(out, phase, frames, ref=None, scale=.25):
    if len(frames) > 12 or not frames:
        raise ValueError('A probe must compose 1..12 assembler frames')
    if len(set(frames)) != len(frames):
        raise ValueError('A probe must not contain duplicate frames')
    dest = out / phase
    dest.mkdir(parents=True, exist_ok=True)
    AS, hashes = assembler(ref)
    AS._init('A', None, scale, True, True)
    ctx = AS._CTX
    import stage
    from PIL import features
    shared_code = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and str(path).startswith(str(ROOT)) and str(path).endswith('.py') and name not in hashes:
            shared_code[str(Path(path).relative_to(ROOT))] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    environment = {'python': sys.executable, 'pillow_raqm': features.check('raqm'),
                   'finish_code_id': stage.code_id(), 'finish_luts': str((ROOT / 'finish/luts').resolve()),
                   'finish_luts_is_symlink': (ROOT / 'finish/luts').is_symlink(),
                   'shared_code_hashes': shared_code}
    read, imread, picture = ctx.read, AS.cv2.imread, ctx.picture
    reads = []
    picture_status = []

    def observed_imread(path, *args, **kwargs):
        if Path(path).exists():
            reads.append(file_state(path))
        value = imread(path, *args, **kwargs)
        if value is None:
            raise OSError(f'Cannot decode source: {path}')
        return value

    def checked_read(path, *args, **kwargs):
        value = read(path, *args, **kwargs)
        if value is None:
            raise OSError(f'Cannot decode source: {path}')
        return value

    def checked_picture(f):
        value = picture(f)
        if value[2].startswith('SLATE'):
            raise RuntimeError(f'Cannot present slate as evidence: {f} {value[2]}')
        picture_status[:] = [value[2]]
        return value

    ctx.read = checked_read
    ctx.picture = checked_picture
    AS.cv2.imread = observed_imread
    plans = [{'section': s['sec'], 'f0': s['f0'], 'f1': s['f1'], 'plan': p}
             for s, p in zip(ctx.shots, ctx.plans)]
    records = {}
    for f in sorted(frames):
        reads.clear()
        rgb = ctx.frame(f)
        i, shot = ctx.shot_at(f)
        if ctx.plans[i]['kind'] == 'slate':
            raise RuntimeError(f'Cannot present slate as evidence: {f} {shot["sec"]}')
        previous = None
        prev_path = dest / f'f_{f - 1:05d}.png'
        prev_meta = dest / f'f_{f - 1:05d}.json'
        if prev_path.exists() and prev_meta.exists():
            metadata = json.loads(prev_meta.read_text())
            if metadata['code_hashes'] == hashes and metadata['scale'] == scale:
                with Image.open(prev_path) as image:
                    previous = np.asarray(image.convert('RGB'))
        record = {'frame': f, 'scale': scale, 'baseline_ref': ref, 'code_hashes': hashes,
                  'picture_status': picture_status[0],
                  'sources': list({r['path']: r for r in reads}.values()), **measured(rgb, previous)}
        Image.fromarray(rgb).save(dest / f'f_{f:05d}.png')
        (dest / f'f_{f:05d}.json').write_text(json.dumps(record, indent=2) + '\n')
        records[str(f)] = record
        print(f'{phase} f{f} Y={record["mean_y"]:.3f} step={record.get("step_y_mad")}', flush=True)
    result = {'phase': phase, 'frames': records, 'plans': plans, 'code_hashes': hashes,
              'environment': environment,
              'baseline_ref': ref, 'size': [ctx.W, ctx.H],
              'metric': 'Finished captioned RGB8; Y=.2126R+.7152G+.0722B, 0..255. '
                        'MAD compares consecutive whole frames. Source sizes/mtimes are actual decoder inputs.'}
    (dest / f'batch_{min(frames):05d}_{max(frames):05d}.json').write_text(json.dumps(result, indent=2) + '\n')


def batches(out, phase, frames, ref=None, scale=.25):
    for batch in chunks(frames):
        cmd = [str(ONEPY), sys.executable, __file__, 'probe', str(out), phase,
               '--frames', ','.join(map(str, batch)), '--scale', str(scale)]
        if ref:
            cmd += ['--ref', ref]
        subprocess.run(cmd, check=True)


def sheets(out, phases):
    width, height, bar = 480, 201, 34
    for name, frames in GROUPS.items():
        canvas = Image.new('RGB', (width * len(frames), (height + bar) * len(phases)), '#15151a')
        draw = ImageDraw.Draw(canvas)
        for row, phase in enumerate(phases):
            for col, f in enumerate(frames):
                path = out / phase / f'f_{f:05d}.png'
                meta = out / phase / f'f_{f:05d}.json'
                x, y = col * width, row * (height + bar)
                if not path.exists() or not meta.exists():
                    raise FileNotFoundError(f'Missing evidence image/record for {phase} f{f}')
                rec = json.loads(meta.read_text())
                with Image.open(path) as im:
                    rgb = np.asarray(im.convert('RGB'))
                    if measured(rgb)['sha256_rgb'] != rec['sha256_rgb']:
                        raise ValueError(f'Image differs from measurement: {path}')
                    canvas.paste(im.resize((width, height), Image.Resampling.LANCZOS), (x, y + bar))
                label = f'{phase} A{f} | Y {rec["mean_y"]:.2f}'
                if 'step_y_mad' in rec:
                    label += f' | step {rec["step_y_mad"]:.2f}'
                draw.text((x + 8, y + 10), label, fill='white')
        canvas.save(out / f'{"_".join(phases)}_{name}.jpg', quality=95)


def preview(out, phase, wav, start=2392, stop=3368):
    # Assembly is a separate bounded action; this validates every image record before encoding.
    code_versions = set()
    for f in range(start, stop):
        metadata = out / phase / f'f_{f:05d}.json'
        if not metadata.exists() or not (out / phase / f'f_{f:05d}.png').exists():
            raise FileNotFoundError(f'Assemble {phase} f{f} before encoding')
        code_versions.add(json.dumps(json.loads(metadata.read_text())['code_hashes'], sort_keys=True))
    if len(code_versions) != 1:
        raise ValueError('Preview would mix different edit code versions; reassemble changed frames')
    dest = out / f'{phase}_A{start}-{stop - 1}.mp4'
    subprocess.run([str(ONEPY), 'ffmpeg', '-y', '-v', 'error', '-framerate', '24',
                    '-start_number', str(start), '-i', str(out / phase / 'f_%05d.png'),
                    '-ss', str(start / 24), '-i', str(wav), '-frames:v', str(stop - start),
                    '-map', '0:v', '-map', '1:a', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '192k', '-t', str((stop - start) / 24),
                    '-movflags', '+faststart', str(dest)], check=True)
    print(dest)


def compare(out, frames):
    report = []
    for f in frames:
        before = json.loads((out / 'before' / f'f_{f:05d}.json').read_text())
        after = json.loads((out / 'after' / f'f_{f:05d}.json').read_text())
        bsrc, asrc = before['sources'], after['sources']
        report.append({'frame': f, 'same_pixels': before['sha256_rgb'] == after['sha256_rgb'],
                       'same_source_state': bsrc == asrc,
                       'before_y': before['mean_y'], 'after_y': after['mean_y'],
                       'before_step_y_mad': before.get('step_y_mad'),
                       'after_step_y_mad': after.get('step_y_mad')})
    (out / f'comparison_{min(frames)}_{max(frames)}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def window_metrics(records, start, end):
    """Inclusive arrival-frame bounds: value at f measures the step f-1 -> f."""
    rows = [records[f] for f in range(start, end + 1)]
    if any(r.get('step_y_mad') is None for r in rows):
        raise ValueError(f'Missing consecutive-frame measurement in [{start}, {end}]')
    peak = max(rows, key=lambda row: row['step_y_mad'])
    level = max(rows, key=lambda row: abs(row.get('mean_y_delta', 0.0)))
    return {'max_step_y_mad': peak['step_y_mad'], 'max_at_frame': peak['frame'],
            'mean_step_y_mad': float(np.mean([r['step_y_mad'] for r in rows])),
            'max_abs_mean_y_delta': abs(level.get('mean_y_delta', 0.0)), 'max_mean_delta_at_frame': level['frame'],
            'steps': [{'frame': r['frame'], 'step_y_mad': r['step_y_mad'], 'mean_y': r.get('mean_y'),
                       'mean_y_delta': r.get('mean_y_delta')} for r in rows]}


def summary(out):
    windows = [(f'beat_{f}', f, f + 10) for f in range(2400, 2621, 20)]
    windows += [('white', 2630, 2641), ('valley', 2640, 2692), ('valley_close', 2756, 2802),
                ('black_ember', 2800, 2870)]
    phases = {}
    for phase in ('before', 'after'):
        frames = set(f for _, start, end in windows for f in range(start, end + 1))
        records = {f: json.loads((out / phase / f'f_{f:05d}.json').read_text()) for f in frames}
        versions = {json.dumps(r['code_hashes'], sort_keys=True) for r in records.values()}
        if len(versions) != 1:
            raise ValueError(f'{phase} windows include multiple edit code versions')
        phases[phase] = records
    report = {'metric': 'Finished 480x201 RGB8 code-value Y=.2126R+.7152G+.0722B. '
                        'Y-MAD is mean(abs(Y[f]-Y[f-1])). Bounds are inclusive arrival frames; '
                        'e.g. [2400,2408] includes 2399->2400 through 2407->2408.',
              'windows': []}
    for name, start, end in windows:
        row = {'name': name, 'start_inclusive': start, 'end_inclusive': end}
        for phase, records in phases.items():
            row[phase] = window_metrics(records, start, end)
        report['windows'].append(row)
    outside = []
    for f in OUTSIDE:
        rows = {phase: json.loads((out / phase / f'f_{f:05d}.json').read_text())
                for phase in ('before', 'after')}
        outside.append({'frame': f, 'same_pixels': rows['before']['sha256_rgb'] == rows['after']['sha256_rgb'],
                        'same_source_state': rows['before']['sources'] == rows['after']['sources']})
    report['outside_controls'] = outside
    (out / 'window_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    for row in report['windows']:
        b, a = row['before'], row['after']
        print(f'{row["name"]} [{row["start_inclusive"]},{row["end_inclusive"]}]: MAD '
              f'{b["max_step_y_mad"]:.2f}@{b["max_at_frame"]} -> {a["max_step_y_mad"]:.2f}@{a["max_at_frame"]}; '
              f'|mean step| {b["max_abs_mean_y_delta"]:.2f}@{b["max_mean_delta_at_frame"]} -> '
              f'{a["max_abs_mean_y_delta"]:.2f}@{a["max_mean_delta_at_frame"]}')
    print(f'Outside controls: {sum(r["same_pixels"] and r["same_source_state"] for r in outside)}/{len(outside)} identical')


def envelope(samples, samplerate, start_sample=0):
    """Exact nonoverlapping 5 ms stereo RMS; zero has no finite dBFS value."""
    width = samplerate // 200
    if width * 200 != samplerate:
        raise ValueError('Sample rate does not support exact 5 ms windows')
    if samples.ndim == 1:
        samples = samples[:, None]
    windows = samples[:len(samples) // width * width].reshape(-1, width, samples.shape[1])
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=(1, 2)))
    peak = np.max(np.abs(windows), axis=(1, 2))
    result = []
    for n, (r, p) in enumerate(zip(rms, peak)):
        result.append({'time_s': (start_sample + n * width) / samplerate,
                       'rms': float(r), 'dbfs': float(20 * np.log10(r)) if r > 0 else None,
                       'peak': float(p), 'digital_zero': bool(p == 0)})
    return result


def audio_evidence(out, before_wav, after_wav=None):
    import soundfile as sf
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    waves = {'before': before_wav}
    if after_wav:
        waves['after'] = after_wav
    evidence = {'measurement': '5 ms nonoverlapping RMS over both channels, dBFS=20log10(RMS); '
                               'digital-zero dBFS is null and plotted at -120 only for display.',
                'files': {key: file_state(path) for key, path in waves.items()}, 'groups': {}}
    for name, (f0, f1, markers) in AUDIO_GROUPS.items():
        group = {}
        fig, ax = plt.subplots(figsize=(14, 4))
        for phase, path in waves.items():
            with sf.SoundFile(path) as wav:
                width = wav.samplerate // 200
                start = int(np.floor(f0 / 24 * wav.samplerate / width)) * width
                stop = int(np.ceil(f1 / 24 * wav.samplerate / width)) * width
                wav.seek(start)
                samples = wav.read(stop - start, dtype='float64', always_2d=True)
                if len(samples) != stop - start:
                    raise ValueError(f'WAV does not cover requested region: {path}')
                rows = envelope(samples, wav.samplerate, start)
                group[phase] = rows
            x = [row['time_s'] for row in rows]
            y = [row['dbfs'] if row['dbfs'] is not None else -120 for row in rows]
            ax.plot(x, y, linewidth=.85, alpha=.85, label=phase)
            zeros = [row['time_s'] for row in rows if row['digital_zero']]
            if zeros:
                ax.scatter(zeros, [-120] * len(zeros), s=3, marker='x', label=f'{phase} digital zero')
        for f in markers:
            ax.axvline(f / 24, color='#777777', linewidth=.6, alpha=.65)
            ax.text(f / 24, 1, f'A{f}', rotation=90, va='top', ha='right', fontsize=8)
        ax.set(title=name.replace('_', ' '), xlabel='Absolute film time (s)', ylabel='5 ms stereo RMS (dBFS)',
               xlim=(f0 / 24, f1 / 24), ylim=(-150, 3))
        ax.grid(alpha=.2)
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(out / f'{"_".join(waves)}_audio_{name}_5ms.png', dpi=140)
        plt.close(fig)
        evidence['groups'][name] = group
    (out / f'{"_".join(waves)}_audio_5ms.json').write_text(json.dumps(evidence, indent=2) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def manifest(out, wavs):
    """One edit-source hash set per phase across every image record, plus the audio files' SHA-256."""
    report = {'phases': {}, 'audio': {str(w): {'sha256': sha256(w), 'bytes': Path(w).stat().st_size} for w in wavs}}
    for phase in ('before', 'after'):
        versions, count = {}, 0
        for meta in sorted((out / phase).glob('f_*.json')):
            key = json.dumps(json.loads(meta.read_text())['code_hashes'], sort_keys=True)
            versions[key] = versions.get(key, 0) + 1
            count += 1
        report['phases'][phase] = {'records': count, 'code_hash_sets': [
            {'code_hashes': json.loads(k), 'records': n} for k, n in versions.items()]}
    (out / 'evidence_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['phases'], indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['probe', 'batches', 'sheets', 'preview', 'compare', 'audio', 'summary',
                                      'manifest'])
    p.add_argument('output', type=Path)
    p.add_argument('phase', nargs='?', default='before')
    p.add_argument('--frames', help='Comma-separated absolute frames, or start:stop (stop excluded)')
    p.add_argument('--ref')
    p.add_argument('--scale', type=float, default=.25)
    p.add_argument('--wav', type=Path)
    p.add_argument('--after-wav', type=Path)
    p.add_argument('--start', type=int, default=2392)
    p.add_argument('--stop', type=int, default=3368)
    args = p.parse_args()
    frames = PROBE
    if args.frames == 'moments':
        frames = sorted(set(PROBE + [f for group in GROUPS.values() for f in group] +
                            [2644, 2648, 2652, 2656, 2660, 2664, 2670, 2676]))
    elif args.frames == 'outside':
        frames = OUTSIDE
    elif args.frames:
        frames = list(range(*map(int, args.frames.split(':')))) if ':' in args.frames else list(map(int, args.frames.split(',')))
    if args.action in ('probe', 'batches'):
        globals()[args.action](args.output, args.phase, frames, args.ref, args.scale)
    elif args.action == 'sheets':
        sheets(args.output, args.phase.split(','))
    elif args.action == 'preview':
        preview(args.output, args.phase, args.wav, args.start, args.stop)
    elif args.action == 'audio':
        audio_evidence(args.output, args.wav, args.after_wav)
    elif args.action == 'summary':
        summary(args.output)
    elif args.action == 'manifest':
        manifest(args.output, [w for w in (args.wav, args.after_wav) if w])
    else:
        compare(args.output, frames)


if __name__ == '__main__':
    main()
