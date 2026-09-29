"""Decode and hash all 400 frozen-source LAST BEACON delivery JPEGs."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, JpegImagePlugin

HERE = Path(__file__).resolve().parent
RENDERER = HERE.parents[1] / 'shots/map/last_beacon.py'
FROZEN_SHA = 'cee6cc2a7b6fd97c497503ace1578b1877333833f69c4026b95f64e096b41601'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory):
    assert digest(RENDERER) == FROZEN_SHA, 'renderer changed after native review'
    expected = {f'f_{f:05d}.jpg' for f in range(3440, 3840)}
    actual = {p.name for p in directory.glob('f_*.jpg')}
    assert actual == expected, {'missing': sorted(expected-actual), 'extra': sorted(actual-expected)}
    assert not list(directory.glob('*.part.*')), 'unfinished frame writes'
    quality_probe = io.BytesIO()
    Image.new('RGB', (8, 8)).save(quality_probe, format='JPEG', quality=95, subsampling=0)
    quality_probe.seek(0)
    with Image.open(quality_probe) as probe:
        quantization = probe.quantization
    frames = []
    for f in range(3440, 3840):
        p = directory / f'f_{f:05d}.jpg'
        with Image.open(p) as im:
            im.load()                       # fully decode, not just inspect headers
            assert im.format == 'JPEG' and im.mode == 'RGB' and im.size == (1920, 804), p
            assert JpegImagePlugin.get_sampling(im) == 0, f'not 4:4:4: {p}'
            assert im.quantization == quantization, f'not quality95 quantization: {p}'
            assert any(hi > lo for lo, hi in im.getextrema()), f'flat frame: {p}'
        sha = digest(p)
        reference = HERE / 'native' / p.name
        if f in (3724, 3792):
            assert reference.exists(), f'missing independently rendered native reference: {reference}'
            assert sha == digest(reference), f'delivery differs from independently rendered native still: {p}'
        frames.append(dict(frame=f, filename=p.name, bytes=p.stat().st_size, sha256=sha))
    manifest_sha = hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest()
    return dict(utc=datetime.now(timezone.utc).isoformat(), directory=str(directory.resolve()),
                frame_count=len(frames), first=3440, last=3839, dimensions=[1920, 804],
                mode='RGB', jpeg_sampling='4:4:4', quality95_quantization=True,
                full_decode=True, native_reference_frames=[3724, 3792],
                renderer_sha256=FROZEN_SHA, frame_manifest_sha256=manifest_sha, frames=frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--receipt', type=Path, default=HERE/'delivery.json')
    args = parser.parse_args()
    receipt = validate(args.directory)
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'frames'}, indent=2))
