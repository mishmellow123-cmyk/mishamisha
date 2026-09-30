#!/usr/bin/env python3
"""Assemble the complete D23 RGB/matte delivery in an owner-named fresh folder.

The output argument names the RGB folder; its sibling receives ``_matte``.
The unchanged farm files are copied before --patch-start and the local files
from that frame onward. Source directories are never modified. Optional review
encoding produces a picture-only 240-frame 960x402, 24 fps clip.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from render_oldfire_local import START, END, safe_output, sha256, validate_jpeg


def assemble(farm_source, patch_source, patch_start, output):
    if not START <= patch_start < END:
        raise ValueError('patch start must fall in D5840..6079')
    output = safe_output(output)
    matte = safe_output(output.with_name(output.name+'_matte'))
    manifest = safe_output(output.with_name(output.name+'_assembly.json'))
    if any(path.exists() for path in (output, matte, manifest)):
        raise ValueError('assembly requires fresh output folders and manifest')
    rows = []
    for frame in range(START, END):
        source = Path(farm_source if frame < patch_start else patch_source).resolve()
        for key, destination in (('rgb', output), ('matte', matte)):
            folder = source if key == 'rgb' else source.with_name(source.name+'_matte')
            path = folder/f'f_{frame:05d}.jpg'
            metadata = validate_jpeg(path)
            rows.append(dict(frame=frame, kind=key, source=str(path),
                             origin='farm' if frame < patch_start else 'local',
                             target=str(destination/path.name), sha256=metadata['sha256']))
    # Preflight all 480 files before creating anything at the destination.
    output.mkdir(parents=True)
    matte.mkdir()
    for row in rows:
        with Path(row['source']).open('rb') as source, Path(row['target']).open('xb') as target:
            shutil.copyfileobj(source, target)
        if sha256(row['target']) != row['sha256']:
            raise RuntimeError('copy checksum mismatch')
    result = dict(first=START, last=END-1, frames=END-START, patch_start=patch_start,
                  rgb=str(output), matte=str(matte), files=rows)
    with manifest.open('x') as report:
        report.write(json.dumps(result, indent=2)+'\n')
    return result


def review_clip(source, output):
    output = safe_output(output)
    receipt = output.with_suffix('.json')
    if output.exists() or receipt.exists():
        raise ValueError('review clip or receipt already exists')
    subprocess.run(['ffmpeg', '-n', '-loglevel', 'error', '-threads', '1',
                    '-framerate', '24', '-start_number', str(START),
                    '-i', str(Path(source)/'f_%05d.jpg'), '-frames:v', str(END-START),
                    '-vf', 'scale=960:402:flags=lanczos', '-an', '-c:v', 'libx264',
                    '-threads', '1', '-crf', '16', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(output)], check=True)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(output)]))
    streams = probe['streams']
    if (len(streams) != 1 or streams[0]['codec_type'] != 'video'
            or (streams[0]['width'], streams[0]['height']) != (960, 402)
            or streams[0]['avg_frame_rate'] != '24/1'
            or int(streams[0]['nb_read_frames']) != END-START):
        raise RuntimeError('review clip failed stream/frame validation')
    probe['sha256'] = sha256(output)
    with receipt.open('x') as report:
        report.write(json.dumps(probe, indent=2)+'\n')
    return probe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--farm-source', required=True, type=Path)
    parser.add_argument('--patch-source', required=True, type=Path)
    parser.add_argument('--patch-start', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.review:
            clip = safe_output(args.review)
            if clip.exists() or clip.with_suffix('.json').exists():
                raise ValueError('review clip or receipt already exists')
        result = assemble(args.farm_source, args.patch_source, args.patch_start, args.out)
        if args.review:
            review_clip(args.out, args.review)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}), flush=True)


if __name__ == '__main__':
    main()
