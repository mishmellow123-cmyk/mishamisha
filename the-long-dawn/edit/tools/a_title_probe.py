"""Sample A's delivered EXR title; this diagnostic cannot certify its steady interval.

Run through onepy, with a captionsA_*.json output path outside renders/. Register
the current Cinzel alpha template to sharp strokes in one delivered EXR, then
measure selected finished-master frames using that translated template. The EXR
is additive RGB, not an alpha mask: neither a good match nor contrast proves that
the unversioned delivery used this exact raster or the current source clock.
All reported ratios are conditional template diagnostics. Until the mask is
independently validated, contrast_usable remains false and no verdict is emitted.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assemble as AS
import c5_caption_backdrop as BD
import lantern_review as LR


def sharp_strokes(image, sigma=3.0):
    """Signed high-pass detail; suppress broad glow without thresholding it into alpha."""
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Registration needs a finite two-dimensional image')
    return image - cv2.GaussianBlur(image, (0, 0), sigma)


def register_template(luma, alpha):
    """Integer translation from normalized cross-correlation of sharp stroke detail.

    The score describes this image/template comparison, not provenance or a
    pass threshold. A blank layer must fail instead of returning an arbitrary
    position from a constant correlation map.
    """
    if luma.shape[0] < alpha.shape[0] or luma.shape[1] < alpha.shape[1]:
        raise ValueError('Template is larger than the delivered layer')
    observed, template = sharp_strokes(luma), sharp_strokes(alpha)
    if float(np.std(observed)) == 0 or float(np.std(template)) == 0:
        raise ValueError('No sharp stroke variation to register')
    scores = cv2.matchTemplate(observed, template, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite registration scores')
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    return dict(x0=int(x), y0=int(y), score=float(score),
                method='TM_CCOEFF_NORMED on image minus GaussianBlur(sigma=3 px)',
                provenance_verified=False)


def template_geometry(line):
    """Only the registered source template's extent; delivered alpha is unavailable."""
    yy, xx = np.nonzero(line.alpha > 0)
    xs, ys = xx + line.x0, yy + line.y0
    return dict(extent=[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                glyph_pixels=int(len(xs)),
                clipped_pixels=int(((xs < 0) | (xs >= BD.W) | (ys < 0) | (ys >= BD.H)).sum()),
                scope='Registered current-source template only; no proof of delivered glyph clipping')


def source_record(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest())


def save_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def check_output(path):
    path = path.resolve()
    if not path.name.startswith('captionsA_') or path.suffix != '.json':
        raise ValueError('Output must be named captionsA_*.json')
    if path.is_relative_to(Path(AS.RENDERS).resolve()):
        raise ValueError('renders/ is read only')
    return path


def without_title(ctx, f, box):
    """Same finished picture path with only EXR addition disabled, then restore it."""
    ember, ember_on = ctx.ember, ctx._ember_on
    try:
        ctx.ember = lambda image, frame: (image, False)
        image, _, status, _ = ctx.picture(f)
        if status.startswith('SLATE'):
            raise RuntimeError(f'{f}: {status}')
        edge, texture = BD.backdrop(image, box)
        del image
        return dict(edge=edge, texture=texture, source=status,
                    method='Finished ctx.picture with EXR addition disabled for this call')
    finally:
        ctx.ember, ctx._ember_on = ember, ember_on


def conditional_measurement(values):
    """Keep the arithmetic without promoting an unvalidated mask into title contrast."""
    return dict(contrast_usable=False, mask_validation='pending independent validation',
                method='Glyph-core/ring arithmetic on the registered current-source template',
                caveat='These values do not establish delivered-title contrast or legibility.',
                values=dict(values))


def sample_label(frame):
    """Evidence labels must remain intelligible without the JSON's longer caveat."""
    return f'A {frame}  template diagnostic; contrast unusable'


def run(output, frames, register_frame=6360, backdrop=True):
    cv2.setNumThreads(1)
    output = check_output(output)
    frames = sorted(set(frames))
    shot = next(s for s in AS.EDL.EDL['A'] if s['kind'] == 'title')
    if not frames or any(not shot['f0'] <= f < shot['f1'] for f in frames + [register_frame]):
        raise ValueError('Registration and sample frames must be within A\'s title shot')
    output.parent.mkdir(parents=True, exist_ok=True)
    title_dir = Path(AS.RENDERS) / 'title_A'
    exr_path = title_dir / f'f_{register_frame:05d}.exr'
    source = source_record(exr_path)
    layer = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED)
    if layer is None or layer.shape != (BD.H, BD.W, 3):
        raise RuntimeError('Registration requires a decoded 1920x804 RGB EXR')
    layer = np.ascontiguousarray(layer[..., ::-1], dtype=np.float32)
    if not np.isfinite(layer).all():
        raise RuntimeError('Delivered EXR contains nonfinite values')
    row = next(r for r in AS.titles.text_table('A') if r['id'] == 'title')
    line = AS.titles.TextV3('A', row, 1.0)
    registration = register_template(layer @ BD.LUMA, line.alpha)
    line.x0, line.y0 = registration['x0'], registration['y0']
    geom = template_geometry(line)
    box = tuple(int(v) for v in BD.box_of(line))
    x0, y0, x1, y1 = box
    core, ring = [mask[y0:y1, x0:x1] for mask in BD.glyph_masks(line)]
    slices = BD.band_slices(line, x0, y0)
    # The display transform is diagnostic only. Correlation uses the EXR's
    # linear luma; alpha always comes from the unchanged Cinzel raster.
    display = LR.quantize(AS.look.linear_to_srgb(np.clip(layer[y0:y1, x0:x1], 0, 1)))
    ink = np.zeros((y1 - y0, x1 - x0), np.uint8)
    hh, ww = line.alpha.shape
    X0, Y0 = max(x0, line.x0), max(y0, line.y0)
    X1, Y1 = min(x1, line.x0 + ww), min(y1, line.y0 + hh)
    ink[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] = (
        line.alpha[Y0 - line.y0:Y1 - line.y0, X0 - line.x0:X1 - line.x0] > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    overlay = display.copy()
    cv2.drawContours(overlay, contours, -1, (40, 255, 110), 1)
    registration_path = output.with_name(f'{output.stem}_registration_{register_frame}.png')
    side_by_side = Image.new('RGB', (2 * display.shape[1], display.shape[0]))
    side_by_side.paste(Image.fromarray(display), (0, 0))
    side_by_side.paste(Image.fromarray(overlay), (display.shape[1], 0))
    side_by_side.save(registration_path)
    del layer, display, overlay, ink, contours, side_by_side
    gc.collect()
    results = dict(cut='A', scale=1.0, status='CONDITIONAL TEMPLATE DIAGNOSTIC; TITLE CONTRAST UNUSABLE',
                   complete=False, contrast_usable=False, mask_validation='pending independent validation',
                   steady_interval=None, steady_interval_proven=False,
                   mask_provenance='Current-source Cinzel alpha translated by image registration; '
                                   'equivalence to delivered unversioned EXR is unproven',
                   metadata_present=(title_dir / 'meta.json').is_file(),
                   words=row['line'], registration_frame=register_frame,
                   registration=registration, registration_source=source,
                   registration_evidence=str(registration_path),
                   registration_display='Left: EXR linear RGB clipped to [0,1], converted to sRGB; '
                                        'right: same with current-template alpha>0.5 contours',
                   template_geometry=geom, box=list(box), requested_frames=frames,
                   caveat='All ratios are conditional template measurements. The registered mask requires '
                          'independent validation before any delivered-title contrast or legibility conclusion; '
                          'the complete steady interval is also unproven.',
                   samples=[], peak_rss_gib=LR.memory())
    save_json(output, results)
    AS._init('A', None, 1.0, True, True)
    ctx = AS._CTX
    cols, tile_w, tile_h, bar = 2, 480, 201, 26
    contact = Image.new('RGB', (cols * tile_w, ((len(frames) + cols - 1) // cols) * (tile_h + bar)), '#101016')
    contact_draw = ImageDraw.Draw(contact)
    for n, f in enumerate(frames):
        delivered = source_record(title_dir / f'f_{f:05d}.exr')
        background = without_title(ctx, f, box) if backdrop else None
        image, _, status, _ = ctx.picture(f)
        if status.startswith('SLATE') or not ctx._ember_on:
            raise RuntimeError(f'{f}: delivered EXR title was not composited: {status}')
        image = np.ascontiguousarray(image, dtype=np.float32)
        AS.titles.composite_v3(image, ctx.lines_nt, f)
        rgb = LR.quantize(image)
        del image
        crop = rgb[y0:y1, x0:x1].astype(np.float32) / 255
        values = LR.caption_contrast(crop, core, ring, slices)
        del crop
        crop_path = output.with_name(f'{output.stem}_crop_{f}.png')
        frame_path = output.with_name(f'{output.stem}_frame_{f}.png')
        Image.fromarray(rgb[y0:y1, x0:x1]).save(crop_path)
        Image.fromarray(cv2.resize(rgb, (960, 402), interpolation=cv2.INTER_AREA)).save(frame_path)
        x, y = (n % cols) * tile_w, (n // cols) * (tile_h + bar)
        contact.paste(Image.fromarray(cv2.resize(rgb, (tile_w, tile_h), interpolation=cv2.INTER_AREA)), (x, y + bar))
        contact_draw.text((x + 8, y + 6), sample_label(f), fill='white')
        results['samples'].append(dict(f=f, source=status, exr=delivered, backdrop=background,
                                       evidence=str(frame_path), crop=str(crop_path),
                                       conditional_template_measurement=conditional_measurement(values)))
        results['peak_rss_gib'] = LR.memory()
        save_json(output, results)
        print(f'{sample_label(f)}; conditional minimum ratio {values["worst"]:.4f}:1; '
              f'peak RSS {results["peak_rss_gib"]:.3f} GiB',
              flush=True)
        del rgb
        gc.collect()
    contact_path = output.with_name(f'{output.stem}_contact.png')
    contact.save(contact_path)
    results['contact_sheet'] = str(contact_path)
    minimum = min(results['samples'],
                  key=lambda sample: sample['conditional_template_measurement']['values']['worst'])
    results['conditional_template_minimum'] = dict(
        f=minimum['f'], conditional_template_measurement=minimum['conditional_template_measurement'])
    results['measured'] = len(results['samples'])
    save_json(output, results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--frames', default='6340,6350,6360,6370,6380,6390', help='Comma-separated sample frames')
    parser.add_argument('--register-frame', type=int, default=6360)
    parser.add_argument('--no-backdrop', action='store_true')
    args = parser.parse_args()
    run(args.output, [int(f) for f in args.frames.split(',')], args.register_frame, not args.no_backdrop)
