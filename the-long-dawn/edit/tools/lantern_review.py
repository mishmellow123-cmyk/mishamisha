"""A18 -> A19 evidence, using the finished master path, one frame at a time.

Run via the owner's tools/onepy: python edit/tools/lantern_review.py OUTPUT_DIR before|after
The before phase loads the pre-lane afix_comp/titles modules from --baseline-ref (default ea7cdf3).
Half size for picture/eye trace; full size only for T14's 2-7 px glyph-ring measurement.
No renderer is invoked, and nothing is written under renders/. Areas count actual half-size
pixels; coordinates alone are multiplied by two to express master picture coordinates.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import types

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assemble as AS
import c5_caption_backdrop as BD

Y = np.float32([0.2126, 0.7152, 0.0722])
JOIN = (5836, 5837, 5838, 5839, 5840, 5846, 5852, 5864, 5876, 5888, 5900)
CAPTION = (6080, 6086, 6092, 6116, 6140, 6164, 6187, 6193, 6199)
POINTS = {'lantern': (1745, 640), 'near': (430, 752), 'far': (1330, 452),
          'farther': (180, 470), 'farthest': (1560, 350)}


def quantize(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def peak(y, mask):
    if not np.any(mask):
        return None
    py, px = np.unravel_index(np.argmax(np.where(mask, y, -1)), y.shape)
    return {'xy_master': [int(px * 2), int(py * 2)], 'luma_255': float(y[py, px])}


def light_metrics(rgb, f):
    a = rgb.astype(np.float32)
    y = a @ Y
    warm = (a[..., 0] - a[..., 2] >= 20) & (a[..., 0] > a[..., 1])
    bright = warm & (y >= 127.5)
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    out = {'f': f, 'warm_peak': peak(y, warm), 'global_peak': peak(y, np.ones(y.shape, bool))}
    regions = {name: (xx - x / 2) ** 2 + (yy - cy / 2) ** 2 <= 40 ** 2
               for name, (x, cy) in POINTS.items()}
    if f < 5840:
        # The carried great lantern in delivered cand_crossing_both_decal 5800-5839.
        regions = {'carried_lantern': (xx >= 745) & (xx <= 790) & (yy >= 238) & (yy <= 275)}
    for name, region in regions.items():
        out[name] = {'peak': peak(y, region), 'warm_peak': peak(y, region & warm),
                     'bright_warm_area_half_px': int(np.count_nonzero(region & bright)),
                     'warm_area_half_px': int(np.count_nonzero(region & warm & (y >= 63.75)))}
    other = ~np.logical_or.reduce(list(regions.values()))
    out['outside_lights'] = {'peak': peak(y, other), 'warm_peak': peak(y, other & warm),
                             'bright_warm_area_half_px': int(np.count_nonzero(other & bright))}
    return out


def sheet(out, frames, path, label):
    # Images are unaltered master output apart from this contact sheet's 2:1 downsample.
    w, h, bar, cols = 480, 201, 24, 3
    canvas = Image.new('RGB', (cols * w, ((len(frames) + cols - 1) // cols) * (h + bar)), '#101016')
    draw = ImageDraw.Draw(canvas)
    for n, f in enumerate(frames):
        x, y = n % cols * w, n // cols * (h + bar)
        with Image.open(out / f'lantern_{label}_{f}.png') as im:
            canvas.paste(im.resize((w, h), Image.Resampling.LANCZOS), (x, y + bar))
        draw.text((x + 8, y + 5), f'{label}  A / {f}', fill='white')
    canvas.save(path, quality=95)


def caption_contrast(comp, core, ring, slices):
    # Same glyph cores, ring, slice extent and 20-pixel denominator as the C5 tool;
    # keep the limiting slice's identity instead of discarding it in the aggregate.
    lum = BD.rel_lum(comp)
    values = [BD.ratio(float(lum[core].mean()), float(lum[ring].mean()))]
    counts = []
    for y0, y1, x0, x1 in slices:
        sl = np.s_[max(0, y0 - 8):y1 + 8, x0:x1]
        c, r, z = core[sl], ring[sl], lum[sl]
        counts.append([int(c.sum()), int(r.sum())])
        if min(counts[-1]) < 20:
            raise RuntimeError(f'T14 slice has insufficient glyph/ring pixels: {counts[-1]}')
        values.append(BD.ratio(float(z[c].mean()), float(z[r].mean())))
    return {'overall': values[0], 'slices': values[1:], 'worst': min(values),
            'limiting_slice': int(np.argmin(values)) - 1, 'slice_counts': counts}


def baseline(ref):
    """Read the actual pre-lane modules; labeling today's code 'before' is not a control."""
    modules = {}
    for name in ('afix_comp', 'titles'):
        source = subprocess.check_output(['git', 'show', f'{ref}:the-long-dawn/edit/{name}.py'], text=True)
        module = types.ModuleType(f'lantern_before_{name}')
        module.__file__ = str(Path(__file__).resolve().parents[1] / f'{name}.py')
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        modules[name] = module
    AS.titles = modules['titles']
    AS.AFIX = modules['afix_comp']
    AS.EDL.TRANS['A'] = [AS.AFIX.WATCHFIRES if t['kind'] == 'watchfires' else t for t in AS.EDL.TRANS['A']]


def memory():
    # macOS reports bytes. This is process high-water RSS, not a claimed current working set.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gib = peak / (1024 ** 3 if sys.platform == 'darwin' else 1024 ** 2)
    if gib >= 1.8:
        raise RuntimeError(f'Review exceeded its own 1.8 GiB budget (peak {gib:.3f} GiB)')
    return gib


def picture(f):
    result = AS._CTX.picture(f)
    _, _, status, src = result
    expected = 'cand_crossing_both_decal' if f < 5840 else 'dawnrev'
    if src is None or not status.startswith(expected) or (f >= 5840 and not status.endswith(' + watchfires')):
        raise RuntimeError(f'A {f}: expected {expected}' + (' + watchfires' if f >= 5840 else '') + f', got {status}')
    return result


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def run(out, phase, section='all', ref='ea7cdf3'):
    out.mkdir(parents=True, exist_ok=True)
    if section in ('all', 'picture'):
        checked = 0
        for path in out.glob('lantern_*'):
            if path.suffix.lower() in ('.png', '.jpg'):
                with Image.open(path) as im:
                    im.verify()
                checked += 1
        print(f'{phase}: verified {checked} existing evidence image files', flush=True)
    if phase == 'before':
        baseline(ref)
    if section in ('all', 'picture'):
        AS._init('A', None, 0.5, True, True)
    metrics = []
    frames = sorted(set(range(5800, 5901)) | set(CAPTION) | {6239, 6240, 6289, 6290, 6339, 6340, 6389, 6390, 6455, 6456, 6479})
    for f in frames if section in ('all', 'picture') else ():
        img, _, status, src = picture(f)
        if 5800 <= f <= 5900:
            metrics.append(dict(light_metrics(quantize(img), f), source=status))
        if f in JOIN or f in CAPTION or f >= 6239:
            comp = img.copy()
            AS.titles.composite_v3(comp, AS._CTX.lines_nt if AS._CTX._ember_on else AS._CTX.lines, f)
            rgb = quantize(comp)
            Image.fromarray(rgb).save(out / f'lantern_{phase}_{f}.png')
            del comp, rgb
        del img
        if f % 20 == 0:
            print(f'{phase}: picture {f}; peak RSS {memory():.3f} GiB', flush=True)
    if metrics:
        sheet(out, JOIN, out / f'lantern_{phase}_join.jpg', phase)
        sheet(out, CAPTION, out / f'lantern_{phase}_T14.jpg', phase)
        write_json(out / f'lantern_{phase}_picture.json', metrics)
    if section == 'picture':
        return
    if not metrics:
        metrics = json.loads((out / f'lantern_{phase}_picture.json').read_text())

    # Exact glyph measurement requires full-size type and its pixel-sized ring;
    # this is decoded delivered picture plus EDIT, never a full crossing/book render.
    AS._CTX = None  # release closure cycles from the half-size finish before building the full-size one
    AS.titles.render_line.cache_clear()
    AS.titles.render_block.cache_clear()
    AS.titles._font.cache_clear()
    gc.collect()
    AS._init('A', None, 1.0, True, True)
    row = next(r for r in AS.titles.text_table('A') if r['id'] == 'T14')
    line = AS.titles.TextV3('A', row, 1.0)
    box = BD.box_of(line)
    x0, y0, x1, y1 = box
    core, ring = [m[y0:y1, x0:x1] for m in BD.glyph_masks(line)]
    slices = BD.band_slices(line, x0, y0)
    contrast = []
    for f in range(*BD.steady(row)):
        img, _, status, src = picture(f)
        edge, texture = BD.backdrop(img, box)
        line.draw(img, f)
        # Quantize exactly as Ctx.frame does before reading the final glyph contrast.
        crop = quantize(img[y0:y1, x0:x1]).astype(np.float32) / 255
        contrast.append(dict(f=f, edge=edge, texture=texture, **caption_contrast(crop, core, ring, slices)))
        del img, crop
        if f % 20 == 0:
            print(f'{phase}: full-size glyph check {f}; peak RSS {memory():.3f} GiB', flush=True)
    report = {'phase': phase, 'picture_scale': 0.5, 'luma': 'Rec.709 weighted sRGB bytes (0..255)',
              'warm_mask': 'R-B >= 20 AND R>G; warm area Y>=63.75; bright warm area Y>=127.5',
              'roi': '80-master-px radius around each A19 light; outgoing great lantern x1490..1580 y476..550',
              'peak_rss_gib': memory(), 'baseline_ref': ref if phase == 'before' else None,
              'picture': metrics, 'caption': {'row': row, 'box': [int(v) for v in box],
              'steady': list(BD.steady(row)), 'frames': contrast,
              'worst': min(contrast, key=lambda p: p['worst'])}}
    path = out / f'lantern_{phase}_metrics.json'
    write_json(path, report)
    print(f'wrote {path}; T14 worst {report["caption"]["worst"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('phase', choices=('before', 'after'))
    parser.add_argument('--section', choices=('all', 'picture', 'caption'), default='all')
    parser.add_argument('--baseline-ref', default='ea7cdf3')
    args = parser.parse_args()
    run(args.output, args.phase, args.section, args.baseline_ref)
