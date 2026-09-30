"""A's captions on the finished master, one decoded frame at a time (run through onepy).

Full-size glyph cores/rings and eight slices use c5_caption_backdrop; pixels are
quantized as Ctx.frame before contrast is measured. Use the actual lines_v3
layout, including the two-part T6 row, and composite every active caption.
An explicit --take measures only delivered frames of an incoming/old take;
missing frames stay listed, never substituted or represented as a complete pass.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assemble as AS
import c5_caption_backdrop as BD
import lantern_review as LR


def geometry(line):
    """Count every nonzero alpha pixel, including antialiasing and A's 5px entry rise."""
    yy, xx = np.nonzero(line.alpha > 0)
    xs, ys = xx + line.x0, yy + line.y0
    clipped = (xs < 0) | (xs >= BD.W) | (ys < 0) | (ys >= BD.H)
    entry = (xs < 0) | (xs >= BD.W) | (ys + 5 < 0) | (ys + 5 >= BD.H)
    return dict(glyph_pixels=len(xs), clipped_pixels=int(clipped.sum()),
                entry_clipped_pixels=int(entry.sum()),
                extent=[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)])


def animated_geometry(line):
    """Exact local alpha support through entry/exit blur; no picture is loaded."""
    clipped_frames, extent, worst = [], None, 0
    for f in range(line.f_in, line.f_out):
        alpha, rise = line._fade(f)
        yy, xx = np.nonzero(alpha > 0)
        if not len(xx):
            continue
        xs, ys = xx + line.x0, yy + int(round(line.y0 + rise))
        count = int(((xs < 0) | (xs >= BD.W) | (ys < 0) | (ys >= BD.H)).sum())
        if count:
            clipped_frames.append(f)
        worst = max(worst, count)
        bounds = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        extent = bounds if extent is None else [min(extent[0], bounds[0]), min(extent[1], bounds[1]),
                                                max(extent[2], bounds[2]), max(extent[3], bounds[3])]
    return dict(animated_clipped_pixels=worst, clipped_frames=clipped_frames, animated_extent=extent)


def timing(row):
    a, b = row['f_in'], row['f_out']
    return dict(cuts=[s['f0'] for s in AS.EDL.EDL['A'] if a < s['f0'] < b],
                windows=[dict(kind=t['kind'], frames=[t['f0'], t['f1']], cut=t.get('cut'),
                              ready=t.get('ready', True), note=t.get('note'))
                         for t in AS.EDL.TRANS['A'] if max(a, t['f0']) < min(b, t['f1'])])


def flags(per, geom):
    if not per:
        return ['UNMEASURED']
    out = []
    if min(p['worst'] for p in per) < 3:
        out.append('LOW-CONTRAST')
    elif min(p['worst'] for p in per) < 4.5:
        out.append('MARGINAL')
    if np.median([p['edge'] for p in per]) > BD.BUSY_EDGE:
        out.append('BUSY-EDGE')
    if np.median([p['texture'] for p in per]) > BD.BUSY_TEXTURE:
        out.append('BUSY-TEXTURE')
    if geom['clipped_pixels'] or geom['entry_clipped_pixels'] or geom.get('animated_clipped_pixels', 0):
        out.append('CLIPPED')
    return out


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gib = peak / (1024 ** 3 if sys.platform == 'darwin' else 1024 ** 2)
    if gib >= 1.8:
        raise RuntimeError(f'Caption audit exceeded its 1.8 GiB budget: {gib:.3f}')
    return gib


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def run(output, ids=None, take=None, placements=None, step=1):
    cv2.setNumThreads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = AS.titles.text_table('A')
    if step < 1:
        raise ValueError('step must be >= 1')
    unknown = (ids or set()) - {r['id'] for r in rows}
    if unknown:
        raise ValueError(f'Unknown caption IDs: {sorted(unknown)}')
    # Override the production table only in this measurement process, so a
    # candidate gets the same row layout as the eventual production patch.
    if placements:
        AS.titles.A_PLACEMENT = {**AS.titles.A_PLACEMENT, **placements}
        rows = AS.titles.text_table('A')
    AS._init('A', None, 1.0, True, True)
    ctx = AS._CTX
    production_plans = [dict(code=s['code'], frames=[s['f0'], s['f1']], plan=p)
                        for s, p in zip(ctx.shots, ctx.plans)]
    lines = {ln.id: ln for ln in ctx.lines}
    if take:
        if not ids or len(ids) != 1:
            raise ValueError('--take requires exactly one --ids caption')
        row = next(r for r in rows if r['id'] in ids)
        i, shot = ctx.shot_at(row['f_in'])
        if row['id'] not in ('T1', 'T13'):
            raise ValueError('--take is restricted to A2/T1 and A18/T13 (identical source coordinates)')
        chosen = dict(shot['takes'][0], stem=take, mode='exact')
        chosen.pop('need', None)
        ctx.plans[i] = dict(kind='take', take=chosen, have=None, alt=0)
    results = dict(scale=1.0, step=step, take_override=take,
                   thresholds=dict(low=3, marginal=4.5, edge=BD.BUSY_EDGE, texture=BD.BUSY_TEXTURE),
                   plans=production_plans, captions=[],
                   excluded=[dict(id='title', reason='Delivered EXR title needs independently validated '
                                  'glyph registration and formed-text interval; TextV3 is its fallback only.')])
    for row in rows:
        if row['id'] == 'title' or (ids and row['id'] not in ids):
            continue
        line = lines[row['id']]
        geom = {**geometry(line), **animated_geometry(line)}
        if geom['clipped_pixels']:
            results['captions'].append(dict(row=row, geometry=geom, timing=timing(row),
                                           complete=False, measured=0, flags=['CLIPPED', 'UNMEASURED'],
                                           frames=[], reason='Clipped mask cannot support eight complete slices.'))
            write_json(output, results)
            continue
        box = tuple(int(v) for v in BD.box_of(line))
        x0, y0, x1, y1 = box
        core, ring = [m[y0:y1, x0:x1] for m in BD.glyph_masks(line)]
        slices = BD.band_slices(line, x0, y0)
        requested = list(range(*BD.steady(row)))[::step]
        per, missing, best_rgb, best_f, best_cr = [], [], None, None, float('inf')
        for f in requested:
            if take and not AS.locate(chosen, 'A', None, f)[0]:
                missing.append(f)
                continue
            img, shot, status, src = ctx.picture(f)
            if status.startswith('SLATE'):
                raise RuntimeError(f'{f}: {status}; cannot audit a missing picture')
            edge, texture = BD.backdrop(img, box)
            AS.titles.composite_v3(img, ctx.lines_nt if ctx._ember_on else ctx.lines, f)
            crop = LR.quantize(img[y0:y1, x0:x1]).astype(np.float32) / 255
            values = LR.caption_contrast(crop, core, ring, slices)
            per.append(dict(f=f, source=status, edge=edge, texture=texture, **values))
            if values['worst'] < best_cr:
                best_rgb = LR.quantize(cv2.resize(img, (960, 402), interpolation=cv2.INTER_AREA))
                best_f, best_cr = f, values['worst']
            del img, crop
            if len(per) % 24 == 0:
                print(f"{row['id']} {f}: {len(per)}/{len(requested)}; peak RSS {memory():.3f} GiB", flush=True)
        evidence = None
        if best_rgb is not None:
            evidence = output.with_name(f'{output.stem}_{row["id"]}_{best_f}.png')
            Image.fromarray(best_rgb).save(evidence)
        result = dict(row=row, geometry=geom, box=list(box), timing=timing(row),
                      steady=list(BD.steady(row)), requested=len(requested), measured=len(per),
                      missing=missing, complete=bool(per) and len(per) == len(requested) and step == 1,
                      median={k: float(np.median([p[k] for p in per])) for k in ('edge', 'texture')} if per else {},
                      worst=min(per, key=lambda p: p['worst']) if per else None,
                      flags=flags(per, geom), evidence=str(evidence) if evidence else None, frames=per)
        if missing:
            result['flags'].append('PARTIAL-COVERAGE')
        if step != 1:
            result['flags'].append('PROBE-ONLY')
        results['captions'].append(result)
        results['peak_rss_gib'] = memory()
        write_json(output, results)
        print(f"{row['id']}: {len(per)}/{len(requested)} worst {best_cr:.4f} {result['flags']}", flush=True)
        del core, ring, best_rgb
        gc.collect()
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--ids')
    ap.add_argument('--take')
    ap.add_argument('--place', action='append', default=[], help='ID=x,y')
    ap.add_argument('--step', type=int, default=1, help='1 for exhaustive acceptance; larger values are probes only')
    args = ap.parse_args()
    place = {k: dict(zip(('x', 'y'), map(int, v.split(','))))
             for k, v in (p.split('=', 1) for p in args.place)}
    run(args.output, set(args.ids.split(',')) if args.ids else None, args.take, place, args.step)
