"""A 3600-6479: point-light and warm-area statistics from the delivered picture, for the second half's sound table.

    cd the-long-dawn
    .../tools/onepy ~/ldfarm/venv/bin/python music/src/measure_a_points.py --section ridges --range 3660-3739 \
        --out <report dir>/asound-evidence/claude

Why a second collector. measure_a_events.py (the Codex lane, 29 Sep) records warm connected components
(R >= 100, R-G >= 15, R-B >= 35) over the whole frame. The far fires this lane must time are not warm at their cores:
the beacon run's link 2 is a pale pink point at 4000 (native crop, max RGB 155/146/147), which that predicate cannot
see, and a far ridge fire is 1-3 px at half size. A point light is a small bright feature on a smooth background
whatever its hue, so the statistic here is the white top-hat of luma (luma minus its 9x9 grey opening: everything
narrower than 9 px that stands above its surroundings), read at each fire's source-projected position (the ROI only
says WHERE to look; the frame decides WHETHER and WHEN). Snow glints and stars pass a luma top-hat too, but they are
bluer than the night around them, while a fire of any size adds warm light: the same top-hat of the chroma
R - (G+B)/2 (offset by 128) is kept beside it, so a fire is a point that stands out in BOTH (link 2 at 4000: its core
R-B is +8 against a surround at -68, where a snow glint's is about the same as its surround).

One frame at a time, half size (960x402), through EDIT (AS._CTX.picture: the take, the finish, EDIT's windows),
before captions: a caption is not a fire, and its glyphs would pass both detectors (5300's line crosses A18's ridge).
Where a caption is on screen its bounding box is recorded, so an ROI under it can be flagged. The delivered frame's
SHA-256 (AS._CTX.frame, captions included) is recorded too: it is the key the Codex receipts used, so the two
collections can be checked against each other frame by frame. No renderer, no audio; <= 80 frames per process.

--take STEM measures a candidate take the cut does not play yet (A14/A15's linked-fires, adopted 29 Sep night) through
the same finish: the section's shots read renders/STEM exactly (off 0), in this process only; the EDL is not edited.
Frames the candidate does not have yet are skipped and listed (a farm render still landing). --edit-root measures
through another worktree's EDIT (read only, no bytecode written): A19-A20's watch-fire comp changed on the owner's
branch after this worktree forked (c189b92: fire_scale 1.65, the lantern's arrival bloom), and the ending must be
measured as the master will draw it. The commit and the comp's SHA-256 are recorded with the batch.
"""
import argparse
import hashlib
import json
import math
import os
import resource
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]                     # the-long-dawn
EVIDENCE = ROOT.parents[2] / 'outputs' / 'claude-owner-night' / 'codex' / 'asound-evidence'
SECTIONS = {'ridges': (3660, 3799), 'inserts': (3800, 3919), 'run': (3920, 4239), 'watchers': (4240, 4399),
            'crossing': (4880, 5839), 'ending': (5840, 6479)}
TOPHAT_K = 9            # px at half size: wider than any far fire's core + aura shoulder (<= 5 px), narrower than terrain
POINT_MIN = 12          # luma levels: a global point candidate (the ROI reads keep every value, thresholds come later)
POINTS_MAX = 400        # per frame, strongest first


def luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def warm_mask(rgb, threshold=100, rg=15, rb=35):
    """The Codex collector's predicate, kept so both collections count the same pixels."""
    p = rgb.astype(np.int16)
    return ((p[..., 0] >= threshold) & (p[..., 0] - p[..., 1] >= rg) & (p[..., 0] - p[..., 2] >= rb)).astype(np.uint8)


def components(mask, min_area=1):
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    return [[int(s[4]), int(s[0]), int(s[1]), int(s[0] + s[2]), int(s[1] + s[3]), round(float(c[0]), 2),
             round(float(c[1]), 2)] for s, c in zip(stats[1:n], centers[1:n]) if s[4] >= min_area]


def rois_for(section, f, geo):
    """[(id, x, y, r, box or None)] in half-size pixels. Source projections are identity guides only."""
    out = []
    if geo is None:                                   # a candidate take: its fires are found by pixels alone
        return out
    if section == 'ridges':
        for fire in geo[f]:
            x, y, z = fire['hot_point']
            if z > 0 and -8 <= x < 968 and -8 <= y < 410:
                out.append((f"ridge.{fire['source_index']}", x, y, 3.0, None))
    elif section == 'run':
        for link in geo[f]:
            x, y, z = link['center']
            if z > 0 and -8 <= x < 968 and -8 <= y < 410:
                hw = max(link['nominal_halfwidth_px'], 1.0)
                box = None
                if hw > 2.5:                                        # links 5 and 7 are resolved flames
                    bx, by, _ = link['base']
                    tx, ty, _ = link['top']
                    box = [x - 2.5 * hw - 2, min(ty, by) - 1.5 * hw - 2, x + 2.5 * hw + 2, max(ty, by) + 0.5 * hw + 2]
                out.append((f"link.{link['id']}", x, y, 3.0, box))
    elif section == 'crossing':
        for wf in geo[f]:
            x, y, z = wf['center']
            hw = wf['nominal_halfwidth_px']
            bx, by, _ = wf['base']
            tx, ty, _ = wf['top']
            box = [x - 1.6 * hw - 3, ty - 0.6 * hw - 3, x + 1.6 * hw + 3, by + 0.3 * hw + 3]
            out.append((f"wf.{wf['id']}", x, y, max(3.0, 0.8 * hw), box))
    elif section == 'ending':
        for k, (x, y, sz, pale) in enumerate(geo):
            out.append((f"edit.{k + 1}", x * 0.5, y * 0.5, 3.0, None))
    elif section == 'watchers':
        out.append(('near_fire', 615.0, 280.0, 6.0, [560, 160, 700, 400]))
    return out


def roi_stats(rid, x, y, r, box, E, K, rgb, warm, dim, W, H):
    """Strongest luma top-hat pixel within r of (x, y); its colour; the strongest chroma top-hat within r; the
    footprint (luma top-hat >= 8 within r + 3); controls at +-12 px in x; warm/dim pixel counts inside box."""
    def peak(cx, cy, rad, M=E):
        x0, x1 = max(0, int(math.floor(cx - rad))), min(W, int(math.ceil(cx + rad)) + 1)
        y0, y1 = max(0, int(math.floor(cy - rad))), min(H, int(math.ceil(cy + rad)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
        if not inside.any():
            return None
        sub = np.where(inside, M[y0:y1, x0:x1], -1)
        j = int(np.argmax(sub))
        py, px = divmod(j, x1 - x0)
        return int(sub.flat[j]), x0 + px, y0 + py, inside, (x0, x1, y0, y1)
    row = dict(id=rid, x=round(float(x), 2), y=round(float(y), 2))
    p = peak(x, y, r)
    if p is None:
        row['off_frame'] = True
        return row
    v, px, py, _, _ = p
    row.update(pk=v, px=px, py=py, rgb=[int(c) for c in rgb[py, px]], ck_at=int(K[py, px]))
    c = peak(x, y, r, K)
    row['ck'] = c[0]
    q = peak(x, y, r + 3.0)
    if q is not None:
        _, _, _, inside, (x0, x1, y0, y1) = q
        row['area8'] = int(((E[y0:y1, x0:x1] >= 8) & inside).sum())
    for name, dx in (('ctl_l', -12.0), ('ctl_r', 12.0)):
        c = peak(x + dx, y, r)
        row[name] = None if c is None else c[0]
        c = peak(x + dx, y, r, K)
        row[name + '_ck'] = None if c is None else c[0]
    if box is not None:
        bx0, by0 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
        bx1, by1 = min(W, int(math.ceil(box[2]))), min(H, int(math.ceil(box[3])))
        row['box'] = [bx0, by0, bx1, by1]
        if bx1 > bx0 and by1 > by0:
            row['warm'] = int(warm[by0:by1, bx0:bx1].sum())
            row['dim'] = int(dim[by0:by1, bx0:bx1].sum())
            comps = components(warm[by0:by1, bx0:bx1])
            if comps:
                big = max(comps, key=lambda c: c[0])
                row['warm_largest'] = [big[0], big[1] + bx0, big[2] + by0, big[3] + bx0, big[4] + by0]
        else:
            row['warm'] = row['dim'] = 0
    return row


def _rel(path, root):
    """renders/... relative to the measured worktree (renders is a symlink in a lane worktree, a folder in the owner's)"""
    for p, r in ((Path(path), Path(root)), (Path(path).resolve(), Path(root).resolve())):
        try:
            return str(p.relative_to(r))
        except ValueError:
            pass
    return str(path)


def load_geometry(section):
    if section is None:                               # a candidate take: the accepted take's projections do not apply
        return None, None
    if section == 'ridges':
        d = json.loads((EVIDENCE / 'ridge_geometry.json').read_text())
        return {r['frame']: r['fires'] for r in d['frames']}, 'ridge_geometry.json'
    if section == 'run':
        d = json.loads((EVIDENCE / 'beacon_geometry.json').read_text())
        return {r['frame']: r['chain'] for r in d['frames']}, 'beacon_geometry.json'
    if section == 'crossing':
        d = json.loads((EVIDENCE / 'crossing_geometry.json').read_text())
        return {r['frame']: r['watchfires'] for r in d['frames']}, 'crossing_geometry.json'
    if section == 'ending':
        comp = sys.modules['afix_comp']               # the comp EDIT imported (edl_v3 -> afix_comp), whichever root
        return list(comp.WATCHFIRES['fires']), f'{comp.__file__} WATCHFIRES'
    if section == 'watchers':
        return 'fixed near-fire box', 'fixed box'
    return None, None


def collect(section, a, b, out, take=None, edit_root=None, frames=None, label=None):
    if (len(frames) if frames else b - a + 1) > 80:
        raise SystemExit('<= 80 frames per process (launch each batch through tools/onepy)')
    s0, s1 = SECTIONS[section]
    if not (s0 <= a <= b <= s1):
        raise SystemExit(f'{section} is {s0}-{s1}')
    eroot = Path(edit_root) if edit_root else ROOT
    sys.path.insert(0, str(eroot / 'edit'))
    import assemble as AS
    import titles
    if Path(AS.__file__).resolve().parent != (eroot / 'edit').resolve():
        raise SystemExit(f'imported {AS.__file__}, not {eroot}/edit')
    AS._init('A', None, 0.5, True, True)
    ctx = AS._CTX
    cv2.setNumThreads(1)
    edit_info = dict(root=str(eroot), assemble_sha256=hashlib.sha256(Path(AS.__file__).read_bytes()).hexdigest())
    for mod in ('afix_comp', 'edl_v3'):
        mp = eroot / 'edit' / f'{mod}.py'
        edit_info[f'{mod}_sha256'] = hashlib.sha256(mp.read_bytes()).hexdigest()
    head = eroot.parent / '.git'
    try:
        import subprocess
        edit_info['commit'] = subprocess.run(['git', '-C', str(eroot), 'rev-parse', 'HEAD'], capture_output=True,
                                             text=True, check=True).stdout.strip()
    except Exception as e:                            # recorded, not fatal: the SHA-256s above pin the code
        edit_info['commit'] = f'unknown ({e})'
    skipped = []
    if take:
        for i, shot in enumerate(ctx.shots):
            if shot['f0'] <= b and a < shot['f1']:
                ctx.plans[i] = dict(kind='take', have=0, alt=0,
                                    take=dict(stem=take, off=0, mode='exact', note=f'measurement override: {take}',
                                              crop=None, grade=None, matte=None, under=None, video=None, need=None,
                                              add=None, final_eligible=False))
    geo, geo_name = load_geometry(None if take else section)
    geo_sha = None
    if geo_name and geo_name.endswith('WATCHFIRES'):
        geo_sha = hashlib.sha256(Path(geo_name.split()[0]).read_bytes()).hexdigest()
    if geo_name and geo_name.endswith('.json'):
        geo_sha = hashlib.sha256((EVIDENCE / geo_name).read_bytes()).hexdigest()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_K, TOPHAT_K))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    ident = hashlib.sha256()
    for f in (frames if frames else range(a, b + 1)):
        i, shot = ctx.shot_at(f)
        plan = ctx.plans[i]
        if plan['kind'] != 'take':
            raise SystemExit(f'{f}: no delivered take ({plan["kind"]})')
        src, _ = AS.locate(plan['take'], 'A', None, f)
        if take and not src:
            skipped.append(f)                          # the candidate's frame has not landed
            continue
        if not src or isinstance(src, tuple):
            raise SystemExit(f'{f}: no still source')
        img, shot, status, _ = ctx.picture(f)
        if status.startswith('SLATE') or 'proxy' in status:
            raise SystemExit(f'{f}: {status}: refuse to measure a slate')
        img = np.ascontiguousarray(img, np.float32)
        pic = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        titles.composite_v3(img, ctx.lines_nt if ctx._ember_on else ctx.lines, f)
        fin = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        del img
        fin_sha = hashlib.sha256(fin.tobytes()).hexdigest()
        ident.update(f'{f} {fin_sha}\n'.encode())
        diff = np.any(fin != pic, axis=2)
        cap = None
        if diff.any():
            ys, xs = np.nonzero(diff)
            cap = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        del fin, diff
        Y = np.clip(luma(pic.astype(np.float32)) + 0.5, 0, 255).astype(np.uint8)
        E = cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, kernel).astype(np.int16)
        P = pic.astype(np.float32)
        C = np.clip(P[..., 0] - 0.5 * (P[..., 1] + P[..., 2]) + 128.5, 0, 255).astype(np.uint8)
        K = cv2.morphologyEx(C, cv2.MORPH_TOPHAT, kernel).astype(np.int16)
        del P, C
        H, W = Y.shape
        warm = warm_mask(pic)
        dim = warm_mask(pic, 60, 10, 20)
        # global point candidates: 3x3 local maxima of the top-hat, strongest first
        dil = cv2.dilate(E, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero((E >= POINT_MIN) & (E == dil))
        order = np.argsort(-E[ys, xs])[:POINTS_MAX]
        points = [[int(xs[k]), int(ys[k]), int(E[ys[k], xs[k]]), int(K[ys[k], xs[k]]), *[int(c) for c in pic[ys[k], xs[k]]]]
                  for k in order]
        row = dict(f=f, frame_sha256=fin_sha, picture_sha256=hashlib.sha256(pic.tobytes()).hexdigest(),
                   source=_rel(src, eroot), status=status, caption_box=cap,
                   tophat_noise=dict(p50=float(np.percentile(E, 50)), p99=float(np.percentile(E, 99)),
                                     p999=float(np.percentile(E, 99.9))),
                   points=points, warm=components(warm), dim=components(dim, 2))
        rois = []
        for rid, x, y, r, box in rois_for(section, f, geo):
            rois.append(roi_stats(rid, x, y, r, box, E, K, pic, warm, dim, W, H))
        row['rois'] = rois
        rows.append(row)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if (f - a) % 20 == 0:
            print(section, f, 'peak RSS MiB', round(rss / 2 ** 20, 1), flush=True)
        if rss > 1500 * 2 ** 20:
            raise MemoryError('stopping at 1500 MiB peak RSS, well under the 3 GiB lane guard')
        del pic, Y, E, K, warm, dim, dil
    dest = out / f'points_{label or section}_{a:05d}_{b:05d}.json'
    tmp = dest.with_suffix('.json.part')
    tmp.write_text(json.dumps(dict(section=section, first=a, last=b, frames_examined=len(rows), take_override=take,
                                   frames_skipped_not_landed=skipped, edit=edit_info,
                                   shape=[402, 960], space='960x402 finished picture before captions (AS._CTX.picture)',
                                   tophat_kernel_px=TOPHAT_K, point_min=POINT_MIN, geometry=geo_name,
                                   point_fields=['x', 'y', 'luma_tophat', 'chroma_tophat', 'R', 'G', 'B'],
                                   geometry_sha256=geo_sha,
                                   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   frame_set_sha256=ident.hexdigest(), rows=rows), separators=(',', ':')) + '\n')
    tmp.replace(dest)
    print('DONE', dest.name, len(rows), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--section', required=True, choices=sorted(SECTIONS))
    ap.add_argument('--range', required=True, help='inclusive a-b, at most 80 frames')
    ap.add_argument('--out', type=Path, default=EVIDENCE / 'claude')
    ap.add_argument('--take', help='a candidate stem under renders/ to measure in place of the cut\'s take')
    ap.add_argument('--edit-root', help='another worktree\'s the-long-dawn whose edit/ to measure through')
    ap.add_argument('--frames', help='comma-separated subset of the range (a candidate\'s test frames)')
    ap.add_argument('--label', help='output name instead of the section (e.g. run_linked)')
    args = ap.parse_args()
    a, b = map(int, args.range.split('-'))
    fr = [int(v) for v in args.frames.split(',')] if args.frames else None
    if fr and not all(a <= v <= b for v in fr):
        raise SystemExit('--frames must lie inside --range')
    collect(args.section, a, b, args.out, args.take, args.edit_root, fr, args.label)
