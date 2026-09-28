"""Read-only checks and review candidates for an inclusive range of rendered frames.

Usage: python edit/tools/frame_qc.py renders/crossing_A --range 4880-5839 \
    --film A --json review/crossing_qc.json
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np


DEFAULT_JOINS = Path(__file__).with_name('frame_qc_joins.json')
FILMS = ('A', 'B', 'C')


@dataclass(frozen=True)
class Config:
    expected_width: int = 1920
    expected_height: int = 804
    thumbnail_width: int = 96
    black_max: float = 0.01
    near_constant_std: float = 0.002
    near_constant_range: float = 0.02
    pop_luma_floor: float = 0.08
    pop_structure_floor: float = 0.06
    pop_mad_multiplier: float = 6.0
    pop_window: int = 12
    pop_min_neighbors: int = 4

    def __post_init__(self):
        for name in ('expected_width', 'expected_height', 'thumbnail_width', 'pop_window', 'pop_min_neighbors'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('black_max', 'near_constant_std', 'near_constant_range', 'pop_luma_floor',
                     'pop_structure_floor', 'pop_mad_multiplier'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


def validate_joins(joins):
    if not isinstance(joins, list):
        raise ValueError('joins must be a list')
    result, seen = [], set()
    for item in joins:
        if not isinstance(item, dict):
            raise ValueError('each join must be an object')
        film, a, b = item.get('film'), item.get('from'), item.get('to')
        if film not in FILMS or type(a) is not int or type(b) is not int or a < 0 or b <= a:
            raise ValueError('each join needs film A/B/C and integer 0 <= from < to')
        key = film, a, b
        if key in seen:
            raise ValueError(f'duplicate join: {film} {a}->{b}')
        seen.add(key)
        label = item.get('label', f'{a}->{b}')
        if not isinstance(label, str) or not label.strip():
            raise ValueError('join label must be a nonempty string')
        result.append(dict(film=film, **{'from': a, 'to': b}, label=label))
    return result


def load_joins(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get('joins')
    return validate_joins(data)


def _read_frame(directory, number, config):
    paths = [directory / f'f_{number:05d}{suffix}' for suffix in ('.png', '.jpg')]
    found = [path for path in paths if path.is_file()]
    record = dict(frame=number, status='missing', path=None, alternatives=[], width=None, height=None,
                  flags=[], luminance_mean=None)
    if not found:
        return record, None, None
    path = found[0]
    record.update(path=str(path), alternatives=[str(p) for p in found[1:]])
    try:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        image = None
    if image is None or image.size == 0:
        record['status'] = 'decode_error'
        return record, None, None
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    record.update(width=w, height=h, channels=channels, dtype=str(image.dtype))
    if image.dtype not in (np.uint8, np.uint16) or channels not in (1, 3, 4):
        record['status'] = 'unsupported_format'
        return record, None, None
    record['status'] = 'ok' if (w, h) == (config.expected_width, config.expected_height) else 'wrong_dimensions'
    maximum = float(np.iinfo(image.dtype).max)
    color = image if image.ndim == 2 else image[..., :3]
    if channels == 1:
        luma = color.astype(np.float32) / maximum
        constant = bool(color.min() == color.max())
    else:
        # Display-encoded Rec.709-weighted RGB; not a physical/linear-light measurement.
        luma = (color[..., 0].astype(np.float32) * 0.0722 + color[..., 1] * 0.7152
                + color[..., 2] * 0.2126).astype(np.float32) / maximum
        constant = bool(np.all(color.min(axis=(0, 1)) == color.max(axis=(0, 1))))
    mean, std = float(luma.mean()), float(luma.std())
    spread = float(luma.max() - luma.min())
    record.update(luminance_mean=mean, luminance_std=std, luminance_range=spread)
    if float(color.max()) / maximum <= config.black_max:
        record['flags'].append('black')
    if constant:
        record['flags'].append('constant')
    elif std <= config.near_constant_std and spread <= config.near_constant_range:
        record['flags'].append('near_constant')
    if channels == 4 and not image[..., 3].any():
        record['flags'].append('transparent')
    digest = hashlib.sha256()
    digest.update(f'{image.dtype.str}:{image.shape}:'.encode())
    digest.update(memoryview(np.ascontiguousarray(image)))
    record['decoded_sha256'] = digest.hexdigest()
    tw = min(config.thumbnail_width, w)
    th = max(1, round(tw * h / w))
    thumb = cv2.resize(luma, (tw, th), interpolation=cv2.INTER_AREA)
    feature = dict(mean=mean, thumb=thumb, decoded_sha256=record['decoded_sha256'])
    return record, image, feature


def _pair(a, b):
    # Removing each thumbnail's own mean separates structure from an exposure shift.
    ca = a['thumb'] - a['thumb'].mean()
    cb = b['thumb'] - b['thumb'].mean()
    return dict(luminance_delta=abs(a['mean'] - b['mean']),
                structure_delta=float(np.abs(ca - cb).mean()))


def _pop_candidates(transitions, config):
    for i, item in enumerate(transitions):
        item['pop_candidates'] = []
        item['baseline'] = {}
        if item['declared_join']:
            continue
        neighbors = [other for other in transitions[max(0, i - config.pop_window):i + config.pop_window + 1]
                     if other is not item and not other['declared_join'] and other['_segment'] == item['_segment']
                     and abs(other['to'] - item['to']) <= config.pop_window]
        for name, metric, floor in (('luminance', 'luminance_delta', config.pop_luma_floor),
                                    ('structure', 'structure_delta', config.pop_structure_floor)):
            values = np.array([other[metric] for other in neighbors])
            median = float(np.median(values)) if len(values) else None
            mad = float(np.median(np.abs(values - median))) if len(values) else None
            enough = len(values) >= config.pop_min_neighbors
            threshold = max(floor, median + config.pop_mad_multiplier * 1.4826 * mad) if enough else floor
            item['baseline'][name] = dict(count=len(values), median=median, mad=mad, threshold=threshold,
                                         method='local_median_mad' if enough else 'absolute_floor_only')
            if item[metric] > threshold:
                item['pop_candidates'].append(name)
    for item in transitions:
        del item['_segment']


def analyze(directory, start, end, *, film=None, config=None, joins=None):
    """Scan inclusive frames and endpoints of joins intersecting that range.

    Explicit joins replace the defaults and require a film. Memory holds one prior
    decoded image, scalar results, and thumbnails only for declared join endpoints.
    """
    config = config or Config()
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise ValueError('range must have integer 0 <= start <= end')
    if film is not None and film not in FILMS:
        raise ValueError('film must be A, B or C')
    if joins is not None and film is None:
        raise ValueError('explicit joins require --film to select their film')
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f'not a render directory: {directory}')
    chosen = [] if film is None else [j for j in (load_joins(DEFAULT_JOINS) if joins is None
                                                 else validate_joins(joins)) if j['film'] == film]
    cut_pairs = {(j['from'], j['to']) for j in chosen}
    cut_starts = {j['to'] for j in chosen}
    intersecting = [j for j in chosen if j['from'] <= end and j['to'] >= start]
    endpoints = {n for j in intersecting for n in (j['from'], j['to'])}
    records, transitions, endpoint_features = [], [], {}
    previous_image = previous_feature = None
    previous_number = None
    segment = 0
    cv2.setNumThreads(1)
    for number in range(start, end + 1):
        record, image, feature = _read_frame(directory, number, config)
        records.append(record)
        if number in cut_starts:
            segment += 1
        if record['status'] != 'ok':
            previous_image = previous_feature = None
            previous_number = None
            segment += 1                    # do not bridge gaps, even for the outlier baseline
            continue
        if number in endpoints:
            endpoint_features[number] = feature
        if previous_number == number - 1:
            pair = _pair(previous_feature, feature)
            pair.update({'from': previous_number, 'to': number, '_segment': segment,
                         'declared_join': (previous_number, number) in cut_pairs,
                         'exact_duplicate': previous_image.dtype == image.dtype
                         and previous_image.shape == image.shape and bool(np.array_equal(previous_image, image))})
            transitions.append(pair)
        previous_number, previous_image, previous_feature = number, image, feature
    _pop_candidates(transitions, config)
    extra_records = {}
    for number in sorted(n for n in endpoints if not start <= n <= end):
        record, image, feature = _read_frame(directory, number, config)
        record['in_requested_range'] = False
        extra_records[number] = record
        if record['status'] == 'ok':
            endpoint_features[number] = feature
    del previous_image, image
    joins_report = []
    for join in chosen:
        item = dict(join, endpoints=[], metrics=None)
        active = join['from'] <= end and join['to'] >= start
        for n in (join['from'], join['to']):
            inside = start <= n <= end
            record = (records[n - start] if inside else extra_records.get(n)) if active else None
            status = record['status'] if record is not None else 'outside_range'
            item['endpoints'].append(dict(frame=n, status=status, path=record['path'] if record else None,
                                          in_requested_range=inside))
        states = [e['status'] for e in item['endpoints']]
        if 'outside_range' in states:
            item['status'] = 'outside_range'
        elif states != ['ok', 'ok']:
            item['status'] = 'unavailable'
        else:
            item['status'] = 'measured'
            a, b = endpoint_features[join['from']], endpoint_features[join['to']]
            item['metrics'] = dict(_pair(a, b), matching_decoded_hash=a['decoded_sha256'] == b['decoded_sha256'])
        joins_report.append(item)
    counts = {status: sum(r['status'] == status for r in records)
              for status in ('ok', 'missing', 'decode_error', 'wrong_dimensions', 'unsupported_format')}
    flags = {flag: [r['frame'] for r in records if flag in r['flags']]
             for flag in ('black', 'constant', 'near_constant', 'transparent')}
    return dict(schema_version=1, directory=str(directory), film=film,
                range=dict(start=start, end=end, inclusive=True), config=asdict(config),
                interpretation='Frame flags, holds and pops are review candidates, not artistic or quality verdicts. '
                               'Thresholds are selected defaults, not calibrated detection rates.',
                joins_selection='none: film not selected' if film is None else 'default' if joins is None else 'explicit',
                coverage=dict(expected_frames=end - start + 1, frame_status_counts=counts,
                              expected_adjacent_pairs=end - start, compared_adjacent_pairs=len(transitions),
                              skipped_adjacent_pairs=end - start - len(transitions),
                              joins_measured=sum(j['status'] == 'measured' for j in joins_report),
                              joins_unavailable=sum(j['status'] == 'unavailable' for j in joins_report),
                              joins_outside_range=sum(j['status'] == 'outside_range' for j in joins_report),
                              extra_join_endpoint_frames=len(extra_records)),
                frame_candidates=flags, frames=records, transitions=transitions, joins=joins_report,
                join_endpoint_frames=list(extra_records.values()))


def summary(report):
    cov = report['coverage']
    counts = cov['frame_status_counts']
    holds = sum(t['exact_duplicate'] and not t['declared_join'] for t in report['transitions'])
    pops = sum(bool(t['pop_candidates']) for t in report['transitions'])
    flags = report['frame_candidates']
    return '\n'.join([
        f"Frames {report['range']['start']}-{report['range']['end']} (inclusive): {counts['ok']}/{cov['expected_frames']} valid.",
        f"File issues: {counts['missing']} missing, {counts['decode_error']} undecodable, "
        f"{counts['wrong_dimensions']} wrong size, {counts['unsupported_format']} unsupported format.",
        f"Review candidates: {len(flags['black'])} black, {len(flags['constant'])} constant, "
        f"{len(flags['near_constant'])} near-constant, {len(flags['transparent'])} transparent; "
        f"{holds} duplicate adjacent pairs, {pops} pop transitions.",
        f"Adjacent coverage: {cov['compared_adjacent_pairs']}/{cov['expected_adjacent_pairs']}; "
        f"joins: {cov['joins_measured']} measured, {cov['joins_unavailable']} unavailable, "
        f"{cov['joins_outside_range']} outside range ({cov['extra_join_endpoint_frames']} extra endpoint reads). "
        'Candidates require review.'
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--range', required=True, dest='frame_range', metavar='START-END')
    parser.add_argument('--film', choices=FILMS)
    parser.add_argument('--json', type=Path, required=True, dest='output')
    parser.add_argument('--joins', type=Path, help='replace the default join list (requires --film)')
    parser.add_argument('--extra-joins', type=Path, help='append to the selected join list (requires --film)')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=804)
    parser.add_argument('--thumbnail-width', type=int, default=96)
    for name, default in (('black-max', 0.01), ('near-constant-std', 0.002), ('near-constant-range', 0.02),
                          ('pop-luma-floor', 0.08), ('pop-structure-floor', 0.06), ('pop-mad-multiplier', 6.0)):
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--pop-window', type=int, default=12)
    parser.add_argument('--pop-min-neighbors', type=int, default=4)
    args = parser.parse_args(argv)
    match = re.fullmatch(r'(\d+)-(\d+)', args.frame_range)
    if not match:
        parser.error('--range must be START-END, inclusive, with nonnegative frame numbers')
    if args.output.suffix.lower() != '.json' or args.output.resolve().suffix.lower() != '.json':
        parser.error('--json must name a .json report file')
    if (args.joins or args.extra_joins) and not args.film:
        parser.error('--joins and --extra-joins require --film')
    if args.output.resolve() in {p.resolve() for p in (DEFAULT_JOINS, args.joins, args.extra_joins) if p}:
        parser.error('the report must not overwrite a join configuration input')
    try:
        joins = load_joins(args.joins or DEFAULT_JOINS) if args.joins or args.extra_joins else None
        if args.extra_joins:
            joins += load_joins(args.extra_joins)
        config = Config(expected_width=args.width, expected_height=args.height,
                        **{name: getattr(args, name) for name in asdict(Config())
                           if name not in ('expected_width', 'expected_height')})
        report = analyze(args.directory, *map(int, match.groups()), film=args.film, config=config, joins=joins)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError) as exc:
        print(f'frame_qc: {exc}', file=sys.stderr)
        return 2
    print(summary(report))
    print(f'JSON: {args.output.resolve()}')
    coverage = report['coverage']
    return int(coverage['frame_status_counts']['ok'] != coverage['expected_frames'] or coverage['joins_unavailable'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
