"""C5 caption backdrops: how busy, and how legible, is the picture under each caption EDIT draws?

For every ink or fire caption whose frames are all present on this machine, over its STEADY frames (written or
kindled, not yet leaving), at the master's 1920x804, reading the picture exactly as assemble.Ctx does:

  box       the glyphs' extent plus half the type size on every side: the backdrop the eye crosses
  edge      fraction of box pixels whose luma gradient exceeds EDGE_T per pixel (Sobel / 4 on sRGB luma after a
            1-pixel Gaussian): drawn lines, hatching, borders, script
  texture   mean over the box of the luma standard deviation in 15-pixel windows
  contrast  the WCAG 2.x contrast ratio between the glyph cores and a ring 2-7 px around them, both read from the
            COMPOSITED frame (a fire line's halo and glow count). "worst" is the lowest of 8 equal slices along the
            line: one word over a bright vein fails there while the line's mean passes.

A caption is BUSY when its median edge fraction exceeds BUSY_EDGE or its median texture exceeds BUSY_TEXTURE, and
LOW-CONTRAST when its worst slice falls under 3:1 (WCAG 2.x's floor for large text) in any steady frame. The busy
thresholds are EDIT-C5's, not a standard: on 29 Sep, before any caption moved, the eight delivered captions split
into a clean group (edge <= 0.015, texture <= 0.012: paper, sky, smoke, dark) and two over line art (R15 on the map,
0.047 / 0.019; R18 on the mine plate, 0.101 / 0.042); each threshold is the midpoint of that gap.

    python3 edit/tools/c5_caption_backdrop.py measure [--ids R15,R18] [--place R18=x,y] [--lines 'R18=a|b'] [--json F]
    python3 edit/tools/c5_caption_backdrop.py search R18 [--lines 'R18=a|b'] [--step 4] [--top 12]

search scores every centre on a 16-px grid with integral images over the caption's steady frames: the worst
frame's edge fraction and texture, and an ESTIMATED worst-slice contrast (the backdrop's slice luminance against
the caption's own colour, no halo). Take its best few to `measure --place`, which renders them for real, and look.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

EDIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EDIT)
import assemble as AS  # noqa: E402
import c5_readiness as RD  # noqa: E402

titles = AS.titles
CUT = 'C'
W, H = 1920, 804
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
EDGE_T = 0.05          # luma step per pixel that counts as a drawn edge
WIN = 15               # texture window, pixels
SLICES = 8
BUSY_EDGE = 0.03       # midpoints of the gap in C5's own frames, 29 Sep (see the docstring); not a standard
BUSY_TEXTURE = 0.015
CR_FLOOR = 3.0         # WCAG 2.x, large text


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def rel_lum(img):
    return srgb_to_linear(img) @ LUMA


def ratio(a, b):
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def steady(row):
    """[a, b): a caption's steady frames (ink is written by f_in + 26, fire kindled by f_in + 12; both leave over the
    last 12)."""
    return row['f_in'] + (26 if row['set'] == 'ink' else 12), row['f_out'] - 12


def glyph_extent(line):
    """(x0, y0, x1, y1) of the glyphs (alpha > 0.5) in picture pixels, half-open."""
    ys, xs = np.nonzero(line.alpha > 0.5)
    return line.x0 + xs.min(), line.y0 + ys.min(), line.x0 + xs.max() + 1, line.y0 + ys.max() + 1


def box_of(line):
    x0, y0, x1, y1 = glyph_extent(line)
    m = int(round(0.5 * line.size))
    return max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)


def edge_map(Y):
    b = cv2.GaussianBlur(Y, (0, 0), 1.0)
    g = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    return (g > EDGE_T).astype(np.float32)


def texture_map(Y):
    m = cv2.boxFilter(Y, -1, (WIN, WIN))
    m2 = cv2.boxFilter(Y * Y, -1, (WIN, WIN))
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def backdrop(img, box):
    """(edge fraction, mean texture) of the picture (no caption) inside box; maps computed with a 16-px apron."""
    x0, y0, x1, y1 = box
    a = 16
    X0, Y0, X1, Y1 = max(0, x0 - a), max(0, y0 - a), min(W, x1 + a), min(H, y1 + a)
    Y = np.ascontiguousarray(img[Y0:Y1, X0:X1] @ LUMA)
    sl = (slice(y0 - Y0, y1 - Y0), slice(x0 - X0, x1 - X0))
    return float(edge_map(Y)[sl].mean()), float(texture_map(Y)[sl].mean())


def glyph_masks(line):
    """(core, ring) boolean masks in picture coordinates for the line's glyphs."""
    core = np.zeros((H, W), bool)
    ink = np.zeros((H, W), np.uint8)
    hh, ww = line.alpha.shape
    X0, Y0 = max(0, line.x0), max(0, line.y0)
    X1, Y1 = min(W, line.x0 + ww), min(H, line.y0 + hh)
    sub = line.alpha[Y0 - line.y0:Y1 - line.y0, X0 - line.x0:X1 - line.x0]
    core[Y0:Y1, X0:X1] = sub > 0.9
    ink[Y0:Y1, X0:X1] = (sub > 0.05).astype(np.uint8)
    near = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    far = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)
    return core, far & ~near


def band_slices(line, X0=0, Y0=0):
    """[(y0, y1, x0, x1)]: SLICES equal slices along each text line's glyphs, in picture pixels minus (X0, Y0)."""
    out = []
    for by0, by1, bx0, bx1 in line.bands:
        edges = np.linspace(line.x0 + bx0, line.x0 + bx1, SLICES + 1).round().astype(int) - X0
        out += [(line.y0 + by0 - Y0, line.y0 + by1 - Y0, a, b) for a, b in zip(edges[:-1], edges[1:])]
    return out


def contrasts(comp, core, ring, slices):
    """(overall ratio, worst slice ratio) from a composited frame (all arrays in the same crop)."""
    L = rel_lum(comp)
    out = ratio(float(L[core].mean()), float(L[ring].mean()))
    worst = out
    for y0, y1, a, b in slices:
        y0, y1 = max(0, y0 - 8), y1 + 8                        # the ring reaches 7 px past the glyphs
        c, r, Ls = core[y0:y1, a:b], ring[y0:y1, a:b], L[y0:y1, a:b]
        if c.sum() >= 20 and r.sum() >= 20:
            worst = min(worst, ratio(float(Ls[c].mean()), float(Ls[r].mean())))
    return out, worst


def delivered_rows(ids=None):
    """The ink/fire rows whose frames are all present here (and in ids, when given)."""
    present = set()
    for a, b, _, _, st in RD.frame_runs():
        if st == 'present':
            present.update(range(a, b))
    rows = []
    for r in titles.text_table(CUT):
        # a line staged in parts (R15, 29 Sep) is measured part by part, as R15a, R15b ...
        for q in ([dict(r, id=f"{r['id']}{'abcdefgh'[k]}", **p) for k, p in enumerate(r['parts'])]
                  if r.get('parts') else [r]):
            if q['set'] not in ('ink', 'fire') or (ids and q['id'] not in ids and r['id'] not in ids):
                continue
            if all(f in present for f in range(q['f_in'], q['f_out'])):
                rows.append({k: v for k, v in q.items() if k != 'parts'})
    return rows


def measure_row(ctx, row, step=1):
    line = titles.TextV3(CUT, row, 1.0)
    box = box_of(line)
    X0, Y0, X1, Y1 = box                                      # the ring (7 px) lies well inside the box's margin
    core, ring = [m[Y0:Y1, X0:X1] for m in glyph_masks(line)]
    slices = band_slices(line, X0, Y0)
    per = []
    for f in range(*steady(row))[::step]:
        img = ctx.picture(f)[0]
        edge, tex = backdrop(img, box)
        comp = np.array(img, np.float32, copy=True)
        line.draw(comp, f)
        cr, worst = contrasts(comp[Y0:Y1, X0:X1], core, ring, slices)
        per.append(dict(f=f, edge=edge, texture=tex, contrast=cr, worst=worst))
    med = {k: float(np.median([p[k] for p in per])) for k in ('edge', 'texture', 'contrast', 'worst')}
    low = min(p['worst'] for p in per)
    flags = ([f'BUSY (edge {med["edge"]:.3f} > {BUSY_EDGE})'] if med['edge'] > BUSY_EDGE else []) + \
            ([f'BUSY (texture {med["texture"]:.3f} > {BUSY_TEXTURE})'] if med['texture'] > BUSY_TEXTURE else []) + \
            ([f'LOW-CONTRAST (worst slice {low:.2f}:1 < {CR_FLOOR:.0f}:1 in {sum(p["worst"] < CR_FLOOR for p in per)} '
              f'of {len(per)} frames)'] if low < CR_FLOOR else [])
    return dict(id=row['id'], set=row['set'], frames=[row['f_in'], row['f_out'] - 1], steady=list(steady(row)),
                measured=len(per), x=row.get('x'), y=row.get('y'), box=[int(v) for v in box], median=med,
                worst_frame=dict(edge=max(p['edge'] for p in per), texture=max(p['texture'] for p in per),
                                 worst=min(p['worst'] for p in per)), flags=flags)


def placed(rows, place=None, breaks=None):
    """rows with overrides applied: place {id: (x, y)} (the caption's centre, as titles.C5_TEXT's x and y) and breaks
    {id: (line, line, ...)} (an explicit break, as C5_TEXT's lines; () removes one)."""
    out = []
    for r in rows:
        r = dict(r)
        if r['id'] in (place or {}):
            r['x'], r['y'] = place[r['id']]
        if r['id'] in (breaks or {}):
            r.pop('lines', None)
            if breaks[r['id']]:
                r['lines'] = tuple(breaks[r['id']])
        out.append(r)
    return out


def measure(ids=None, place=None, step=1, breaks=None):
    ctx = AS.Ctx(CUT, None, 1.0, True)
    return [measure_row(ctx, r, step) for r in placed(delivered_rows(ids), place, breaks)]


def _integral(a):
    return cv2.integral(np.ascontiguousarray(a, np.float32), sdepth=cv2.CV_64F)


def _box_mean(I, x0, y0, x1, y1):
    return (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / np.maximum(1, (x1 - x0) * (y1 - y0))


def search(rid, step=4, top=12, grid=16, keep=80, breaks=None):
    """Rank centres for caption rid (as set, or with breaks) by: an ESTIMATED worst-slice contrast of at least 3:1
    first, then the worst frame's edge fraction, then its texture, then the distance moved; every candidate's box
    keeps `keep` pixels from the frame's edges. Returns C5_TEXT-ready (x, y) centres."""
    row = placed(delivered_rows({rid}), None, breaks)[0]
    line = titles.TextV3(CUT, row, 1.0)
    gx0, gy0, gx1, gy1 = glyph_extent(line)
    gw, gh, m = gx1 - gx0, gy1 - gy0, int(round(0.5 * line.size))
    dx = (line.x0 + line.w / 2.0) - (gx0 + gx1) / 2.0             # the row's centre relative to the glyphs' centre
    dy = (line.y0 + line.h / 2.0) - (gy0 + gy1) / 2.0
    cx0, cy0 = (gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0
    cxs = np.arange(gw / 2 + m + keep, W - gw / 2 - m - keep, grid)
    cys = np.arange(gh / 2 + m + keep, H - gh / 2 - m - keep, grid)
    CX, CY = [a.ravel() for a in np.meshgrid(cxs, cys)]
    bx0 = np.round(CX - gw / 2 - m).astype(int)
    by0 = np.round(CY - gh / 2 - m).astype(int)
    bx1, by1 = bx0 + gw + 2 * m, by0 + gh + 2 * m
    col = titles.IRON if row['set'] == 'ink' else titles._heat_rgb(np.array([0.72], np.float32))[0]  # steady heat
    text_L = float(rel_lum(np.asarray(col, np.float32)))
    rel = [(ty0 - gy0, ty1 - gy0, a - gx0, b - gx0) for ty0, ty1, a, b in band_slices(line)]   # vs the glyphs' corner
    worst_edge = np.zeros(len(CX))
    worst_tex = np.zeros(len(CX))
    worst_cr = np.full(len(CX), np.inf)
    ctx = AS.Ctx(CUT, None, 1.0, True)
    frames = list(range(*steady(row)))[::step]
    for f in frames:
        img = ctx.picture(f)[0]
        Y = np.ascontiguousarray(img @ LUMA)
        IE, IT, IL = _integral(edge_map(Y)), _integral(texture_map(Y)), _integral(rel_lum(img))
        worst_edge = np.maximum(worst_edge, _box_mean(IE, bx0, by0, bx1, by1))
        worst_tex = np.maximum(worst_tex, _box_mean(IT, bx0, by0, bx1, by1))
        ox, oy = np.round(CX - gw / 2).astype(int), np.round(CY - gh / 2).astype(int)
        for t0, t1, a, b in rel:
            Lb = _box_mean(IL, ox + a, oy + t0, ox + b, oy + t1)
            cr = np.where(Lb > text_L, (Lb + 0.05) / (text_L + 0.05), (text_L + 0.05) / (Lb + 0.05))
            worst_cr = np.minimum(worst_cr, cr)
    dist = np.hypot(CX - cx0, CY - cy0)
    order = np.lexsort((dist, worst_tex, worst_edge, worst_cr < CR_FLOOR))
    out = [dict(x=int(round(CX[i] + dx)), y=int(round(CY[i] + dy)), edge=float(worst_edge[i]),
                texture=float(worst_tex[i]), contrast_est=float(worst_cr[i]), moved=float(dist[i]))
           for i in order[:top]]
    return dict(id=rid, frames=len(frames), now=dict(x=int(round(cx0 + dx)), y=int(round(cy0 + dy))),
                lines=row.get('lines'), candidates=out)


def report(res):
    lines = [f'C5 caption backdrops at 1920x804 (BUSY: median edge > {BUSY_EDGE} or texture > {BUSY_TEXTURE}; '
             f'LOW-CONTRAST: a worst slice under {CR_FLOOR:.0f}:1 in any steady frame)']
    for r in res:
        m, w = r['median'], r['worst_frame']
        lines.append(f"{r['id']} {r['set']:4} {r['frames'][0]}-{r['frames'][1]} ({r['measured']} steady frames) "
                     f"box {r['box']}: edge {m['edge']:.3f} (worst {w['edge']:.3f}), texture {m['texture']:.3f} "
                     f"(worst {w['texture']:.3f}), contrast {m['contrast']:.2f}:1, worst slice {m['worst']:.2f}:1 "
                     f"(worst frame {w['worst']:.2f}:1)  " + ('; '.join(r['flags']) or 'clean'))
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['measure', 'search'])
    ap.add_argument('rid', nargs='?')
    ap.add_argument('--ids')
    ap.add_argument('--place', action='append', default=[], help='ID=x,y (the caption centre, as C5_TEXT x and y)')
    ap.add_argument('--lines', action='append', default=[], help='ID=first|second (an explicit break); ID= for none')
    ap.add_argument('--step', type=int, default=None)
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--keep', type=int, default=80, help='search: pixels kept clear of the frame edges')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    breaks = {k: tuple(t for t in v.split('|') if t) for k, v in (p.split('=', 1) for p in a.lines)}
    if a.cmd == 'search':
        res = search(a.rid, step=a.step or 4, top=a.top, keep=a.keep, breaks=breaks)
        print(f"{res['id']}{' as ' + repr(res['lines']) if res['lines'] else ''}: {res['frames']} frames searched; "
              f"now at x {res['now']['x']}, y {res['now']['y']}")
        for c in res['candidates']:
            print(f"  x {c['x']:4}, y {c['y']:3}: worst edge {c['edge']:.3f}, texture {c['texture']:.3f}, "
                  f"est. worst-slice contrast {c['contrast_est']:.2f}:1, moved {c['moved']:.0f} px")
    else:
        place = {}
        for p in a.place:
            k, v = p.split('=')
            place[k] = tuple(int(t) for t in v.split(','))
        ids = set(a.ids.split(',')) if a.ids else None
        res = measure(ids, place, a.step or 1, breaks)
        print(report(res))
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(res, fh, indent=1, default=list)
    return 0


if __name__ == '__main__':
    sys.exit(main())
