"""Carved-inscription maps for the ACCORD stone table.

Builds a Cartesian table-top texture (world x,y in metres, centred on the hearth)
with channels:
    0  carve depth in metres (V-cut chisel profile; >0 = into the stone)
    1  oath-letter coverage   (0..1)
    2  "together" coverage     (0..1)
    3  ornament coverage       (border rings, separators, hearth rays)
plus a mip pyramid, cached to renders/accord/cache/textmaps.npz.

Text is laid on arcs: letters' up = radially outward, reading clockwise seen
from above (so the block at the top of frame reads left->right).
"""
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(ROOT, 'renders', 'accord', 'cache')
CINZEL = os.path.join(ROOT, 'assets', 'fonts', 'Cinzel.ttf')
NOTO = '/usr/share/fonts/truetype/noto/'
NOTO_CJK = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'

# ------------------------------------------------------------------ layout ---
TEX_N = 4096
TEX_R = 2.56                      # texture covers [-TEX_R, TEX_R]^2 metres
TEXEL = 2 * TEX_R / TEX_N         # ~1.25 mm

OATH_CAP = 0.150                  # cap height (m)
OATH_R_IN = 1.36                  # baseline radius of 2nd line (inner)
OATH_LEAD = 1.50 * OATH_CAP       # baseline to baseline
OATH_R_OUT = OATH_R_IN + OATH_LEAD
OATH_TRACK = 0.05                 # extra tracking, in cap heights per char
OATHS = [('NO SINGLE HAND', 'SHALL HOLD IT'),
         ('NO FASTER THAN', 'WE CAN SEE'),
         ('NO FORGE', 'IN THE DARK'),
         ('WHAT IT GIVES,', 'IT GIVES TO ALL')]
# block centres: oath I at world +y (north), then clockwise in 90 deg steps
OATH_THETA = [math.pi / 2 - k * math.pi / 2 for k in range(4)]

BAND_IN = OATH_R_IN - 0.085       # carved border rings of the oath band
BAND_OUT = OATH_R_OUT + OATH_CAP + 0.085

TOG_R = (1.99, 2.215)             # baselines of the two "together" rows (inner, outer)
TOG_XH = 0.080                    # target Latin x-height (m); other scripts matched by eye
TOG_BORDER = (1.905, 2.405)
TABLE_R = 2.52

HEARTH_R = 0.50
RAY_R = (0.70, 1.16)              # 12 carved sun-rays around the hearth

MAX_DEPTH = 0.016                 # m, V-cut max depth
WALL_TAN = 1.35                   # depth per metre of inside-distance (V wall slope)

# "together" (bible section 8), in order, with font + scale + direction
def _nf(name):
    return os.path.join(NOTO, name)

TOGETHER = [
    ('together', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'en'),
    ('juntos', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'es'),
    ('ensemble', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'fr'),
    ('zusammen', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'de'),
    ('insieme', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'it'),
    ('вместе', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'ru'),
    ('разом', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'uk'),
    ('razem', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'pl'),
    ('μαζί', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'el'),
    ('一起', (NOTO_CJK, 2), 0.90, None, 'zh'),
    ('共に', (NOTO_CJK, 0), 0.90, None, 'ja'),
    ('함께', (NOTO_CJK, 1), 0.90, None, 'ko'),
    ('معًا', _nf('NotoNaskhArabic-SemiBold.ttf'), 1.12, 'rtl', 'ar'),
    ('ביחד', _nf('NotoSerifHebrew-SemiBold.ttf'), 0.88, 'rtl', 'he'),
    ('با هم', _nf('NotoNaskhArabic-SemiBold.ttf'), 1.12, 'rtl', 'fa'),
    ('एक साथ', _nf('NotoSerifDevanagari-SemiBold.ttf'), 0.95, None, 'hi'),
    ('ایک ساتھ', _nf('NotoNastaliqUrdu-Bold.ttf'), 0.82, 'rtl', 'ur'),
    ('একসাথে', _nf('NotoSerifBengali-SemiBold.ttf'), 0.95, None, 'bn'),
    ('ਇਕੱਠੇ', _nf('NotoSerifGurmukhi-SemiBold.ttf'), 0.95, None, 'pa'),
    ('ஒன்றாக', _nf('NotoSerifTamil-SemiBold.ttf'), 0.85, None, 'ta'),
    ('ด้วยกัน', _nf('NotoSerifThai-SemiBold.ttf'), 0.88, None, 'th'),
    ('cùng nhau', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'vi'),
    ('bersama', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'id'),
    ('pamoja', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'sw'),
    ('birlikte', _nf('NotoSerif-SemiBold.ttf'), 1.0, None, 'tr'),
    ('ერთად', _nf('NotoSerifGeorgian-SemiBold.ttf'), 1.0, None, 'ka'),
    ('միասին', _nf('NotoSerifArmenian-SemiBold.ttf'), 1.0, None, 'hy'),
]

SS = 4  # supersampling for glyph rasterisation / distance fields


def _font(spec, size):
    if isinstance(spec, tuple):
        return ImageFont.truetype(spec[0], size, index=spec[1], layout_engine=ImageFont.Layout.RAQM)
    return ImageFont.truetype(spec, size, layout_engine=ImageFont.Layout.RAQM)


def cinzel(size):
    f = ImageFont.truetype(CINZEL, size, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_axes([700])
    return f


# ------------------------------------------------------------ strip render ---

def _signed_dist(mask_ss):
    """mask at SSx res (bool) -> signed distance in 1x texels (+inside), downsampled."""
    m = mask_ss.astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sd = (din - dout + np.where(m > 0, -0.5, 0.5)).astype(np.float32) / SS
    h, w = sd.shape
    sd = cv2.resize(sd, (w // SS, h // SS), interpolation=cv2.INTER_AREA)
    return sd


def oath_strip(text, cap_m):
    """Render a Cinzel line with tracking. Returns (sdist[H,W] in texels, baseline_row, width_texels)."""
    cap_px = cap_m / TEXEL * SS
    size = int(round(cap_px / 0.70))
    f = cinzel(size)
    track = OATH_TRACK * cap_px
    # per-char positions using prefix lengths (keeps kerning)
    xs = [f.getlength(text[:i]) + i * track for i in range(len(text))]
    width = f.getlength(text) + (len(text) - 1) * track
    pad = int(0.6 * cap_px)
    Wd = int(width + 2 * pad)
    Wd += (-Wd) % SS
    Hd = int(2.2 * cap_px)
    Hd += (-Hd) % SS
    base = int(1.6 * cap_px)
    base -= base % SS
    img = Image.new('L', (Wd, Hd), 0)
    d = ImageDraw.Draw(img)
    for ch, x in zip(text, xs):
        d.text((pad + x, base), ch, font=f, fill=255, anchor='ls')
    m = np.asarray(img) > 127
    sd = _signed_dist(m)
    return sd, base // SS, width / SS, pad / SS


def word_strip(word, fontspec, scale, direction, lang, xh_m):
    """Render a shaped word. Size set so Latin x-height = xh_m * scale-correction."""
    # Noto Serif x-height ~0.536 em
    em_px = xh_m / 0.536 * scale / TEXEL * SS
    size = int(round(em_px))
    f = _font(fontspec, size)
    kw = {}
    if direction:
        kw['direction'] = direction
    if lang:
        kw['language'] = lang
    bb = f.getbbox(word, anchor='ls', **kw)
    pad = int(0.5 * em_px)
    Wd = int(bb[2] - bb[0] + 2 * pad)
    Wd += (-Wd) % SS
    top = min(bb[1], -int(0.9 * em_px))
    bot = max(bb[3], int(0.35 * em_px))
    Hd = int(bot - top + 2 * pad)
    Hd += (-Hd) % SS
    base = int(pad - top)
    base -= base % SS
    img = Image.new('L', (Wd, Hd), 0)
    d = ImageDraw.Draw(img)
    d.text((pad - bb[0], base), word, font=f, fill=255, anchor='ls', **kw)
    m = np.asarray(img) > 127
    sd = _signed_dist(m)
    ink_w = (bb[2] - bb[0]) / SS
    return sd, base // SS, ink_w, pad / SS, img


# ------------------------------------------------------------ arc warping ---

_rho = None
_theta = None


def _polar():
    global _rho, _theta
    if _rho is None:
        c = (np.arange(TEX_N, dtype=np.float32) + 0.5) * TEXEL - TEX_R
        x = c[None, :]
        y = -c[:, None]                  # row 0 = +y (north)
        _rho = np.sqrt(x * x + y * y).astype(np.float32)
        _theta = np.arctan2(y, x).astype(np.float32)
    return _rho, _theta


def place_on_arc(sd_strip, base_row, u_center, r_base, theta_c, rmin, rmax, out_sd):
    """Warp a strip signed-distance field onto the ring, max-combining into out_sd."""
    rho, theta = _polar()
    Hs, Ws = sd_strip.shape
    half_ang = (Ws * 0.5 + 4) / r_base * TEXEL
    # bounding box in texture space
    ri = int(max(0, (TEX_R - rmax) / TEXEL - 2))
    rf = int(min(TEX_N, (TEX_R + rmax) / TEXEL + 2))
    sub_rho = rho[ri:rf, ri:rf]
    sub_th = theta[ri:rf, ri:rf]
    dth = (theta_c - sub_th + np.pi) % (2 * np.pi) - np.pi
    sel = (sub_rho >= rmin) & (sub_rho <= rmax) & (np.abs(dth) <= half_ang)
    if not sel.any():
        return
    ys, xs = np.nonzero(sel)
    u = (u_center + dth[ys, xs] * r_base / TEXEL).astype(np.float32)
    v = (base_row - (sub_rho[ys, xs] - r_base) / TEXEL).astype(np.float32)
    n = u.size
    cols = 4096
    npad = (-n) % cols
    up = np.concatenate([u, np.zeros(npad, np.float32)]).reshape(-1, cols)
    vp = np.concatenate([v, np.zeros(npad, np.float32)]).reshape(-1, cols)
    vals = cv2.remap(sd_strip, up, vp, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=-1e3).reshape(-1)[:n]
    tgt = out_sd[ri:rf, ri:rf]
    cur = tgt[ys, xs]
    tgt[ys, xs] = np.maximum(cur, vals)


def oath_line_angles():
    """Angular extents (for camera roll + fill timing): list per oath of
    [(theta_start, theta_end) line1, line2] (theta_start > theta_end, clockwise)."""
    out = []
    for k, (l1, l2) in enumerate(OATHS):
        ang = []
        for text, rb in ((l1, OATH_R_OUT), (l2, OATH_R_IN)):
            f = cinzel(100)
            cap_px = 70.0
            w = (f.getlength(text) + (len(text) - 1) * OATH_TRACK * cap_px) / cap_px * OATH_CAP
            half = 0.5 * w / rb
            ang.append((OATH_THETA[k] + half, OATH_THETA[k] - half))
        out.append(ang)
    return out


def together_layout():
    """Returns list of (word_index, row, theta_centre, half_angle)."""
    strips = []
    for i, (w, fs, sc, dr, lg) in enumerate(TOGETHER):
        sd, base, ink_w, pad, _ = word_strip(w, fs, sc, dr, lg, TOG_XH)
        strips.append((sd, base, ink_w))
    # alternate rows: even index -> outer row, odd -> inner row
    rows = {0: [], 1: []}
    for i in range(len(TOGETHER)):
        rows[1 if i % 2 == 0 else 0].append(i)
    layout = []
    for row, idxs in rows.items():
        rb = TOG_R[row]
        widths = np.array([strips[i][2] * TEXEL for i in idxs])      # metres of ink
        circ = 2 * np.pi * rb
        gap = (circ - widths.sum()) / len(idxs)
        # first word centred on the top (theta=pi/2) for outer row; inner row staggered half a slot
        s = 0.0 if row == 1 else 0.5 * (widths[0] + gap)
        pos = []
        acc = -widths[0] / 2 + s
        for j, i in enumerate(idxs):
            cen = acc + widths[j] / 2
            pos.append(cen)
            acc += widths[j] + gap
        for j, i in enumerate(idxs):
            th = np.pi / 2 - pos[j] / rb
            layout.append((i, row, th, widths[j] / 2 / rb, rb))
    return strips, layout


# -------------------------------------------------------------- build all ---

def build(verbose=True):
    rho, theta = _polar()
    N = TEX_N
    sd_oath = np.full((N, N), -1e3, np.float32)
    sd_tog = np.full((N, N), -1e3, np.float32)

    # oaths
    for k, (l1, l2) in enumerate(OATHS):
        for text, rb in ((l1, OATH_R_OUT), (l2, OATH_R_IN)):
            sd, base, width, pad = oath_strip(text, OATH_CAP)
            ucen = pad + width / 2
            place_on_arc(sd, base, ucen, rb, OATH_THETA[k], rb - 0.06, rb + OATH_CAP * 1.35, sd_oath)
        if verbose:
            print('oath', k, 'placed', flush=True)

    # together
    strips, layout = together_layout()
    for (i, row, th, half, rb) in layout:
        sd, base, ink_w = strips[i]
        pad = (sd.shape[1] - ink_w) / 2
        # centre of ink = pad + ink_w/2 (word_strip centres ink between pads)
        ucen = sd.shape[1] / 2
        place_on_arc(sd, base, ucen, rb, th, rb - 0.09, rb + 0.19, sd_tog)
    if verbose:
        print('together placed', flush=True)

    # ornaments: border rings, separators, hearth rays (signed distance in texels)
    t = TEXEL
    sd_orn = np.full((N, N), -1e3, np.float32)

    def ring(r, hw):
        return (hw - np.abs(rho - r)) / t

    for r, hw in ((BAND_IN, 0.0065), (BAND_IN - 0.028, 0.003), (BAND_OUT, 0.0065), (BAND_OUT + 0.028, 0.003),
                  (TOG_BORDER[0], 0.005), (TOG_BORDER[1], 0.006)):
        sd_orn = np.maximum(sd_orn, ring(r, hw))
    # diamond separators between oath blocks (both lines' midline radius)
    rm = 0.5 * (OATH_R_IN + OATH_R_OUT + OATH_CAP)
    for k in range(4):
        th = OATH_THETA[k] - np.pi / 4
        cx, cy = rm * np.cos(th), rm * np.sin(th)
        X = (np.arange(N, dtype=np.float32) + 0.5) * t - TEX_R
        # rotate into radial/tangential frame
        dx = X[None, :] - cx
        dy = (-X)[:, None] - cy
        rr = dx * np.cos(th) + dy * np.sin(th)
        tt = -dx * np.sin(th) + dy * np.cos(th)
        dia = (0.07 - (np.abs(rr) / 1.0 + np.abs(tt) / 0.55)) * 0.55
        sd_orn = np.maximum(sd_orn, dia / t)
        for s in (-1, 1):   # two small dots flanking
            ddx = X[None, :] - (cx + s * 0.13 * -np.sin(th))
            ddy = (-X)[:, None] - (cy + s * 0.13 * np.cos(th))
            dot = 0.018 - np.sqrt(ddx * ddx + ddy * ddy)
            sd_orn = np.maximum(sd_orn, dot / t)
    # 12 tapered sun-rays around the hearth
    for k in range(12):
        th = np.pi / 2 + k * np.pi / 6
        dth = (theta - th + np.pi) % (2 * np.pi) - np.pi
        along = (rho - RAY_R[0]) / (RAY_R[1] - RAY_R[0])
        halfw = 0.018 * (1 - np.clip(along, 0, 1)) + 0.002
        dist_t = np.abs(dth) * rho
        inside = (halfw - dist_t)
        endcap = np.minimum(rho - RAY_R[0], RAY_R[1] - rho)
        sd_ray = np.minimum(inside, endcap)
        sd_orn = np.maximum(sd_orn, sd_ray / t)
    # 12 short hour ticks outside the together band (where the emissaries stand)
    for k in range(12):
        th = np.pi / 2 + k * np.pi / 6 + np.pi / 12
        dth = (theta - th + np.pi) % (2 * np.pi) - np.pi
        dist_t = np.abs(dth) * rho
        inside = 0.006 - dist_t
        endcap = np.minimum(rho - 2.425, 2.49 - rho)
        sd_orn = np.maximum(sd_orn, np.minimum(inside, endcap) / t)

    def depth_from(sd):
        d = np.clip(sd * t * WALL_TAN, 0, None)
        # soft cap (rounded groove bottom)
        return (MAX_DEPTH * (1 - np.exp(-d / MAX_DEPTH))).astype(np.float32)

    cov = lambda sd: np.clip(sd + 0.5, 0, 1).astype(np.float32)
    depth = np.maximum(np.maximum(depth_from(sd_oath), depth_from(sd_tog)), depth_from(sd_orn) * 0.8)
    tex = np.stack([depth, cov(sd_oath), cov(sd_tog), cov(sd_orn)], -1).astype(np.float32)
    return tex, layout


def mip_pyramid(tex, min_size=16):
    levels = [tex]
    cur = tex
    while cur.shape[0] > min_size:
        cur = cv2.resize(cur, (cur.shape[1] // 2, cur.shape[0] // 2), interpolation=cv2.INTER_AREA)
        levels.append(cur)
    sizes = np.array([l.shape[0] for l in levels], np.int64)
    offs = np.zeros(len(levels), np.int64)
    o = 0
    for i, l in enumerate(levels):
        offs[i] = o
        o += l.shape[0] * l.shape[1]
    flat = np.concatenate([l.reshape(-1, 4) for l in levels], 0).astype(np.float32)
    return flat, offs, sizes


def load(rebuild=False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'textmaps_v3.npz')
    if os.path.exists(path) and not rebuild:
        z = np.load(path)
        return z['flat'], z['offs'], z['sizes'], z['tog_layout']
    tex, layout = build()
    flat, offs, sizes = mip_pyramid(tex)
    tl = np.array([(i, row, th, half, rb) for (i, row, th, half, rb) in layout], np.float64)
    np.savez(path, flat=flat, offs=offs, sizes=sizes, tog_layout=tl)
    return flat, offs, sizes, tl


# ------------------------------------------------------------------- test ---

def script_test(out_png):
    """Zoomed sheet of every 'together' word as rasterised for carving (verify shaping/RTL/tofu)."""
    tiles = []
    for (w, fs, sc, dr, lg) in TOGETHER:
        sd, base, ink_w, pad, img = word_strip(w, fs, sc, dr, lg, TOG_XH)
        a = np.asarray(img)
        a = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)
        lab = np.zeros((40, a.shape[1]), np.uint8)
        cv2.putText(lab, lg, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 180, 2)
        tiles.append(np.vstack([lab, a]))
    Hm = max(t.shape[0] for t in tiles)
    rows, cur, curw = [], [], 0
    for t in tiles:
        t = np.pad(t, ((0, Hm - t.shape[0]), (0, 20)))
        if curw + t.shape[1] > 3000 and cur:
            rows.append(np.hstack(cur)); cur, curw = [], 0
        cur.append(t); curw += t.shape[1]
    rows.append(np.hstack(cur))
    Wm = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 10), (0, Wm - r.shape[1]))) for r in rows])
    cv2.imwrite(out_png, 255 - sheet)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        script_test(os.path.join(ROOT, 'renders', 'accord', 'tests', 'scripts_zoom.png'))
    else:
        flat, offs, sizes, tl = load(rebuild=True)
        print(flat.shape, offs, sizes)
