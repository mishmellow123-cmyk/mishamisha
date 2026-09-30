"""Search A's sampled finished backdrops; run through the owner's onepy wrapper.

    python edit/tools/a_caption_search.py T5 --output /tmp/T5.json \
        --lines 'T5=Whoever held it first, they said,|would hold the world.'
    python edit/tools/a_caption_search.py --group T6a,T6b --output /tmp/T6.json

Only measured edge/texture box means are ranked. Every proposed placement still
needs composited glyph-ring contrast and exhaustive acceptance in a_caption_review.
Images are decoded one at a time; the only saved artifact is JSON.
"""
import argparse
import gc
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a_caption_review as AR
import c5_caption_backdrop as BD

AS = AR.AS


def prepared_rows(rows, breaks=None):
    """Validate proposed breaks against the actual current wording before loading picture."""
    breaks = breaks or {}
    unknown = set(breaks) - {r['id'] for r in rows}
    if unknown:
        raise ValueError(f'Line breaks name unselected captions: {sorted(unknown)}')
    result = []
    for row in rows:
        row = dict(row)
        if row['id'] in breaks:
            parts = tuple(breaks[row['id']])
            if not parts or any(not part for part in parts) or ' '.join(parts) != row['line']:
                raise ValueError(f"{row['id']}: line breaks must preserve the current words exactly")
            row['lines'] = parts
        result.append(row)
    return result


def requested_frames(rows, step=8, frames=None):
    """Each member keeps its own steady interval; a group includes its earlier solo phrase."""
    if step < 1:
        raise ValueError('step must be >= 1')
    steady = {r['id']: range(*BD.steady(r)) for r in rows}
    if frames is None:
        result = {rid: list(interval)[::step] for rid, interval in steady.items()}
    else:
        frames = sorted(set(frames))
        if not frames or any(not any(f in interval for interval in steady.values()) for f in frames):
            raise ValueError('Every explicit frame must be inside a selected caption steady interval')
        result = {rid: [f for f in frames if f in interval] for rid, interval in steady.items()}
    if any(not values for values in result.values()):
        raise ValueError('Every selected caption needs at least one steady sample')
    return result


def candidate_grid(lines, grid=40, margin=48, group=False, width=None, height=None):
    """Project integer TextV3 centre coordinates, retaining exact row x origins in group mode.

    The box used by backdrop() and the entire alpha canvas plus its 5px entry
    offset must both fit inside the margin. This also reserves the blur support.
    """
    width, height = BD.W if width is None else width, BD.H if height is None else height
    if grid < 1 or margin < 0:
        raise ValueError('grid must be >= 1 and margin must be >= 0')
    if not lines or (not group and len(lines) != 1):
        raise ValueError('Independent search requires one line; grouped search requires its row members')
    current_y = [float(line.y) for line in lines]
    ys = np.unique(np.r_[np.arange(margin, height - margin + 1, grid),
                         np.rint(current_y)]).astype(int)
    if group:
        xs, cy = np.zeros(len(ys), dtype=int), ys
    else:
        current_x = lines[0].x0 + lines[0].w / 2
        xgrid = np.unique(np.r_[np.arange(margin, width - margin + 1, grid),
                              round(current_x)]).astype(int)
        xs, cy = [v.ravel() for v in np.meshgrid(xgrid, ys)]
    valid = np.ones(len(cy), dtype=bool)
    boxes, origins = [], []
    distances = np.zeros(len(cy), dtype=np.float64)
    for line in lines:
        x0 = np.full(len(cy), line.x0, dtype=int) if group else np.rint(xs - line.w / 2).astype(int)
        y0 = np.rint(cy - line.h / 2).astype(int)
        gx0, gy0, gx1, gy1 = BD.glyph_extent(line)
        pad = int(round(line.size * 0.5))
        box = np.stack((x0 + gx0 - line.x0 - pad, y0 + gy0 - line.y0 - pad,
                        x0 + gx1 - line.x0 + pad, y0 + gy1 - line.y0 + pad), axis=1)
        valid &= ((box[:, 0] >= margin) & (box[:, 1] >= margin)
                  & (box[:, 2] <= width - margin) & (box[:, 3] <= height - margin)
                  & (x0 >= margin) & (y0 >= margin)
                  & (x0 + line.w <= width - margin) & (y0 + line.h + 5 <= height - margin))
        boxes.append(box)
        origins.append(np.stack((x0, y0), axis=1))
        distances = np.maximum(distances, np.hypot(x0 - line.x0, y0 - line.y0))
    if not valid.any():
        raise ValueError('No candidate fits the requested margin; change the line break or margin')
    return dict(x=xs[valid], y=cy[valid], boxes=np.stack(boxes, axis=1)[valid],
                origins=np.stack(origins, axis=1)[valid], distance=distances[valid])


def integral_means(image, boxes):
    """Measured backdrop means for every candidate box, with no caption pixels drawn."""
    luma = np.ascontiguousarray(image @ BD.LUMA, dtype=np.float32)
    edge = BD._integral(BD.edge_map(luma))
    texture = BD._integral(BD.texture_map(luma))
    corners = np.asarray(boxes, dtype=int).T
    return BD._box_mean(edge, *corners), BD._box_mean(texture, *corners)


def ranked_samples(samples, distance):
    """All members must pass both median thresholds; then worst edge, texture, distance."""
    if not samples:
        raise ValueError('Cannot rank an empty measurement')
    members = {}
    for rid, values in samples.items():
        if not values['edge'] or not values['texture']:
            raise ValueError(f'{rid}: no measured samples')
        edge, texture = np.asarray(values['edge']), np.asarray(values['texture'])
        if edge.shape != texture.shape or edge.ndim != 2 or edge.shape[1] != len(distance):
            raise ValueError(f'{rid}: sample shape does not match the candidate grid')
        if not np.isfinite(edge).all() or not np.isfinite(texture).all():
            raise ValueError(f'{rid}: nonfinite backdrop measurement')
        members[rid] = dict(median_edge=np.median(edge, axis=0), median_texture=np.median(texture, axis=0),
                            worst_edge=np.max(edge, axis=0), worst_texture=np.max(texture, axis=0))
    summary = {key: np.max([member[key] for member in members.values()], axis=0)
               for key in ('median_edge', 'median_texture', 'worst_edge', 'worst_texture')}
    summary['passes_sampled_busy_thresholds'] = ((summary['median_edge'] <= BD.BUSY_EDGE)
                                               & (summary['median_texture'] <= BD.BUSY_TEXTURE))
    summary['order'] = np.lexsort((distance, summary['worst_texture'], summary['worst_edge'],
                                  ~summary['passes_sampled_busy_thresholds']))
    summary['members'] = members
    return summary


def search(ctx, rows, lines, step=8, grid=40, margin=48, top=12, frames=None, group=False):
    frame_map = requested_frames(rows, step, frames)
    candidates = candidate_grid(lines, grid, margin, group)
    samples = {r['id']: dict(edge=[], texture=[]) for r in rows}
    sources = []
    for f in sorted(set().union(*map(set, frame_map.values()))):
        image, shot, status, src = ctx.picture(f)
        if status.startswith('SLATE'):
            raise RuntimeError(f'{f}: {status}; missing picture cannot support a backdrop search')
        if image.shape != (BD.H, BD.W, 3):
            raise ValueError(f'Expected {BD.W}x{BD.H} picture, got {image.shape}')
        # One pair of maps per image, shared across both row members. Flattened
        # boxes remain candidate-major, so reshape restores the member axis.
        edge, texture = integral_means(image, candidates['boxes'].reshape(-1, 4))
        edge, texture = edge.reshape(-1, len(lines)), texture.reshape(-1, len(lines))
        active = []
        for k, row in enumerate(rows):
            rid = row['id']
            if f in frame_map[rid]:
                samples[rid]['edge'].append(edge[:, k].copy())
                samples[rid]['texture'].append(texture[:, k].copy())
                active.append(rid)
        sources.append(dict(frame=f, status=status, captions=active))
        del image, edge, texture
        print(f"{','.join(frame_map)} {f}: sampled backdrop; peak RSS {AR.memory():.3f} GiB", flush=True)
    ranked = ranked_samples(samples, candidates['distance'])
    output = []
    for index in ranked['order'][:top]:
        placements = {row['id']: dict(y=int(candidates['y'][index])) for row in rows}
        if not group:
            placements[rows[0]['id']]['x'] = int(candidates['x'][index])
        output.append(dict(placements=placements, boxes=candidates['boxes'][index].tolist(),
                           origins=candidates['origins'][index].tolist(), moved=float(candidates['distance'][index]),
                           passes_sampled_busy_thresholds=bool(ranked['passes_sampled_busy_thresholds'][index]),
                           **{key: float(ranked[key][index]) for key in
                              ('median_edge', 'median_texture', 'worst_edge', 'worst_texture')},
                           members={rid: {key: float(value[index]) for key, value in metrics.items()}
                                    for rid, metrics in ranked['members'].items()}))
    return dict(ids=[r['id'] for r in rows], group=group, rows=rows, sampled_frames=frame_map,
                sources=sources, grid=grid, margin=margin, candidate_count=len(candidates['y']), candidates=output)


def run(output, ids=None, group=None, breaks=None, step=8, grid=40, margin=48, top=12, frames=None):
    if step < 1 or grid < 1 or margin < 0 or top < 1:
        raise ValueError('step, grid, and top must be positive; margin must be nonnegative')
    if bool(ids) == bool(group):
        raise ValueError('Select independent IDs or one group')
    selected = list(group or ids)
    if len(selected) != len(set(selected)):
        raise ValueError('Caption IDs must be unique')
    table = {r['id']: r for r in AS.titles.text_table('A')}
    unknown = set(selected) - set(table)
    if unknown or 'title' in selected:
        raise ValueError(f'Unknown or unsupported caption IDs: {sorted(unknown or {"title"})}')
    rows = prepared_rows([table[rid] for rid in selected], breaks)
    if group:
        if breaks or len(rows) < 2 or any(row['set'] != 'row' for row in rows):
            raise ValueError('Group search requires unchanged row members; it searches their common y only')
    elif any(row['set'] == 'row' for row in rows):
        raise ValueError('Use --group for row captions so their actual horizontal layout is preserved')
    batches = [rows] if group else [[row] for row in rows]
    for batch in batches:
        requested_frames(batch, step, frames)
    cv2.setNumThreads(1)
    AS._init('A', None, 1.0, True, True)
    ctx = AS._CTX
    actual = {line.id: line for line in ctx.lines}
    result = dict(scale=1.0, sample_step=step, explicit_frames=frames,
                  purpose='Sampled backdrop search; requires composited contrast and exhaustive review.',
                  thresholds=dict(edge=BD.BUSY_EDGE, texture=BD.BUSY_TEXTURE),
                  plans=[dict(code=s['code'], frames=[s['f0'], s['f1']], plan=p)
                         for s, p in zip(ctx.shots, ctx.plans)], searches=[])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for batch in batches:
        lines = [AS.titles.TextV3('A', row, 1.0) if row['id'] in (breaks or {}) else actual[row['id']]
                 for row in batch]
        result['searches'].append(search(ctx, batch, lines, step, grid, margin, top, frames, bool(group)))
        result['peak_rss_gib'] = AR.memory()
        AR.write_json(output, result)
        gc.collect()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rid', nargs='?', help='one independent caption ID')
    parser.add_argument('--ids', help='comma-separated independent caption IDs')
    parser.add_argument('--group', help='comma-separated row members, e.g. T6a,T6b; common y only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lines', action='append', default=[], help='ID=first line|second line')
    parser.add_argument('--frames', help='explicit comma-separated steady sample frames')
    parser.add_argument('--step', type=int, default=8)
    parser.add_argument('--grid', type=int, default=40)
    parser.add_argument('--margin', type=int, default=48)
    parser.add_argument('--top', type=int, default=12)
    args = parser.parse_args(argv)
    if sum(bool(value) for value in (args.rid, args.ids, args.group)) != 1:
        parser.error('use exactly one of rid, --ids, or --group')
    breaks = dict((rid, tuple(value.split('|'))) for rid, value in (part.split('=', 1) for part in args.lines))
    run(args.output, ids=(args.ids or args.rid).split(',') if args.ids or args.rid else None,
        group=args.group.split(',') if args.group else None, breaks=breaks,
        step=args.step, grid=args.grid, margin=args.margin, top=args.top,
        frames=[int(value) for value in args.frames.split(',')] if args.frames else None)


if __name__ == '__main__':
    main()
