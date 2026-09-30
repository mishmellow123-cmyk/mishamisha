"""Measure A's changed joins through the finished assembler, without rendering sources.

Run through the owner's tools/onepy; frames are decoded sequentially at 960x402.
The candidate view explicitly selects first-priority takes even while they are landing;
missing frames stay missing. Farm-test frames may fill only the same candidate's gaps.
The active view uses the real EDL coverage gates; accepted uses ALTERNATIVES' A takes.
All coordinates and areas are actual half-size pixels, never inferred master pixels.
"""
import argparse
import json
import resource
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assemble as AS

Y = np.float32([0.2126, 0.7152, 0.0722])
JOINS = (80, 128, 536, 560, 596, 672, 3920, 4240, 4400, 4880, 5840, 6240)
DETAIL_JOINS = (3936, 3960, 4000, 4040, 4080, 4120, 4160, 4200, 4230, 4960, 5180, 5440)
ACCEPTED = {'A2': AS.EDL.T('falsedawn'),
            'A14': AS.EDL.T('beaconrun_A_catches3', mode='exact'),
            'A15': AS.EDL.T('watchers_A_catches3', mode='exact'),
            'A18': AS.EDL.T('crossing')}
LOCATE = AS.locate
CURRENT_EDL, CURRENT_AFIX = AS.EDL, AS.AFIX


def baseline(ref):
    """Use actual historical edit code for before evidence, even while sources land."""
    prior = sys.modules['afix_comp']
    try:
        for name in ('afix_comp', 'edl_v3'):
            source = subprocess.check_output(['git', 'show', f'{ref}:the-long-dawn/edit/{name}.py'], text=True)
            module = types.ModuleType(name)
            module.__file__ = str(Path(AS.ROOT) / 'edit' / f'{name}.py')
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
            if name == 'afix_comp':
                AS.AFIX = module
                sys.modules['afix_comp'] = module
            else:
                AS.EDL = module
    finally:
        sys.modules['afix_comp'] = prior


def candidate_locate(take, cut, variant, f):
    ref, alt = LOCATE(take, cut, variant, f)
    if ref is None and take['stem'].startswith('cand_'):
        ref = AS.index(str(Path(AS.RENDERS) / '_farmtest' / take['stem'])).get(f + take['off'])
    return ref, alt


def context(view):
    AS.locate = LOCATE
    AS._INDEX.clear()  # Each pass gets a fresh landing snapshot; never re-use a stale directory index.
    AS._init('A', None, 0.5, True, True)
    ctx = AS._CTX
    read = ctx.read

    def checked_read(ref, *args, **kwargs):
        img = read(ref, *args, **kwargs)
        if img is None:
            raise OSError(f'Present frame could not be decoded: {ref}')
        return img

    ctx.read = checked_read
    if view == 'candidate':
        AS.locate = candidate_locate
    for n, shot in enumerate(ctx.shots):
        if shot['sec'] not in ACCEPTED or view == 'active':
            continue
        take = ACCEPTED[shot['sec']] if view == 'accepted' else shot['takes'][0]
        ctx.plans[n] = dict(kind='take', take=take, have=0, alt=0)
    return ctx


def dependencies(ctx, f):
    t = AS.transition_at('A', f)
    frames = [f] if not t or t['kind'] in AS.TKINDS_SHOT else [min(f, t['cut'] - 1), max(f, t['cut'])]
    found, missing = [], []
    for g in sorted(set(frames)):
        n, shot = ctx.shot_at(g)
        plan = ctx.plans[n]
        if plan['kind'] == 'black':
            continue
        take = plan.get('take')
        ref = AS.locate(take, 'A', None, g)[0] if take else None
        if ref is None:
            missing.append(dict(frame=g, shot=shot['sec'], take=take['stem'] if take else None))
        else:
            stat = Path(ref).stat()
            found.append(dict(frame=g, path=str(Path(ref).relative_to(AS.ROOT)), role='base_plate',
                              bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        if shot['kind'] == 'title':
            title = Path(AS.RENDERS) / 'title_A' / f'f_{g:05d}.exr'
            layer = dict(frame=g, path=str(title.relative_to(AS.ROOT)), role='optional_title', present=title.exists())
            if title.exists():
                stat = title.stat()
                layer.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            found.append(layer)
    return found, missing


def peak(y, mask):
    if not mask.any() or float(y[mask].max()) <= 0:
        return None
    row, col = np.unravel_index(np.argmax(np.where(mask, y, -1)), y.shape)
    return dict(xy=[int(col), int(row)], y=float(y[row, col]))


def metrics(rgb):
    a = rgb.astype(np.float32)
    y = a @ Y
    # Same warm colour definition as lantern_review.py; a brightness floor avoids
    # treating grain or a dim brown hillside as the eye's luminous fire anchor.
    warm = (a[..., 0] - a[..., 2] >= 20) & (a[..., 0] > a[..., 1]) & (y >= 63.75)
    return dict(mean_y=float(y.mean()), mean_rgb=a.mean(axis=(0, 1), dtype=np.float64).tolist(),
                warm_peak=peak(y, warm), global_peak=peak(y, np.ones(y.shape, bool)),
                warm_area=int(warm.sum()))


def distance(a, b, key):
    if a[key] is None or b[key] is None:
        return None
    return float(np.linalg.norm(np.array(a[key]['xy']) - b[key]['xy']))


def anchor_metrics(rgb, f):
    """Declared screen ROIs separate cream/white story lights from warmer fires.

    Boxes come from inspection of the initial finished frames, in half-size pixels.
    Peak positions can hop within a clipped core; also report its Y>=230 centroid.
    """
    if f in (4879, 4880):
        box = (450, 170, 510, 230)
    elif f == 5839:
        box = (745, 238, 790, 275)
    elif f in (5840, 5876, 5900):
        box = (845, 290, 900, 345)
    elif f == 3919 or 3930 <= f <= 3962:
        box = (540, 80, 650, 170)
    else:
        return None
    x0, y0, x1, y1 = box
    y = rgb.astype(np.float32) @ Y
    mask = np.zeros(y.shape, bool)
    mask[y0:y1, x0:x1] = True
    yy, xx = np.nonzero(mask & (y >= 230))
    return dict(box=list(box), peak=peak(y, mask), bright_area=int(xx.size),
                bright_centroid=[float(xx.mean()), float(yy.mean())] if xx.size else None)


def sheet(out, phase, cut, frames, views, records):
    w, h, bar = 480, 201, 36
    canvas = Image.new('RGB', (w * len(frames), (h + bar) * len(views)), '#14141a')
    draw = ImageDraw.Draw(canvas)
    for row, view in enumerate(views):
        for col, f in enumerate(frames):
            p = out / f'joinsA_{phase}_{view}_{f:05d}.png'
            x, y = col * w, row * (h + bar)
            label = f'{view} | A {f}'
            # A previous run's image is not evidence that this run composed it.
            if not records[view]['frames'][str(f)]['missing'] and p.exists():
                with Image.open(p) as im:
                    canvas.paste(im.resize((w, h), Image.Resampling.LANCZOS), (x, y + bar))
            else:
                label += ' | SOURCE MISSING'
            draw.text((x + 8, y + 8), label, fill='white')
    canvas.save(out / f'joinsA_{phase}_join_{cut:05d}.jpg', quality=95)


def run(out, phase, quick=False, views=('accepted', 'active', 'candidate'), frames=None, baseline_ref=None, radius=12):
    AS.EDL, AS.AFIX = CURRENT_EDL, CURRENT_AFIX
    if baseline_ref:
        baseline(baseline_ref)
    out.mkdir(parents=True, exist_ok=True)
    offsets = (-2, -1, 0, 1) if quick else range(-radius, radius + 1)
    samples = sorted(set(frames or [f + d for f in JOINS for d in offsets]))
    stills = set(frames or [f + d for f in JOINS for d in (-2, -1, 0, 1)])
    result = dict(phase=phase, baseline_ref=baseline_ref, size=[960, 402],
                  metric='sRGB code-value luma = .2126R + .7152G + .0722B, all values 0..255; '
                         'RGB means over entire finished captioned frame; XY/area at half size; '
                         'warm: R-B >= 20, R>G, Y>=63.75; ties in raster order', views={})
    for view in views:
        ctx = context(view)
        records, prev, prev_f, prev_m = {}, None, None, None
        plans = {str(s['f0']): p.get('take', {}).get('stem') if p.get('take') else p['kind']
                 for s, p in zip(ctx.shots, ctx.plans)}
        for number, f in enumerate(samples):
            found, missing = dependencies(ctx, f)
            rec = dict(frame=f, sources=found, missing=missing)
            records[str(f)] = rec
            if missing:
                prev, prev_f, prev_m = None, None, None
                continue
            rgb = ctx.frame(f)
            m = metrics(rgb)
            rec.update(m)
            rec['story_anchor'] = anchor_metrics(rgb, f)
            if prev_f == f - 1:
                rec['step'] = dict(rgb_mad=float(np.abs(rgb.astype(np.float32) - prev).mean()),
                                   mean_y_delta=m['mean_y'] - prev_m['mean_y'],
                                   warm_travel=distance(prev_m, m, 'warm_peak'),
                                   global_travel=distance(prev_m, m, 'global_peak'))
            if f in stills:
                Image.fromarray(rgb).save(out / f'joinsA_{phase}_{view}_{f:05d}.png')
            prev, prev_f, prev_m = rgb, f, m
            if (number + 1) % 50 == 0:
                print(f'{view}: {number + 1}/{len(samples)} sampled', flush=True)
        result['views'][view] = dict(plans=plans, frames=records)
        print(f'{view}: {len(records)} sampled, {sum(not r["missing"] for r in records.values())} composed', flush=True)
    if frames is None:
        for cut in JOINS:
            sheet(out, phase, cut, [cut + d for d in (-2, -1, 0, 1)], views, result['views'])
    else:
        for cut in JOINS + DETAIL_JOINS:
            selected = [cut + d for d in (-2, -1, 0, 1)]
            if all(f in samples for f in selected):
                sheet(out, phase, cut, selected, views, result['views'])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_rss_gib'] = rss / (1024 ** 3 if sys.platform == 'darwin' else 1024 ** 2)
    (out / f'joinsA_{phase}_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'peak RSS: {result["peak_rss_gib"]:.3f} GiB', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('phase')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--views', default='accepted,active,candidate')
    parser.add_argument('--frames', help='Explicit comma-separated absolute A frames')
    parser.add_argument('--baseline-ref', help='Read afix_comp and edl_v3 from this git ref for the before control')
    parser.add_argument('--radius', type=int, default=12, help='Consecutive samples on either side of each join')
    args = parser.parse_args()
    run(args.output, args.phase, args.quick, tuple(args.views.split(',')),
        [int(f) for f in args.frames.split(',')] if args.frames else None, args.baseline_ref, args.radius)
