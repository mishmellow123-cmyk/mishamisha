"""Carved inscriptions and ornament for the ACCORD stone table and floor (v2).

Builds a Cartesian texture (world x,y in metres, centred on the hearth) with channels:
    0  carve depth in metres (V-cut chisel profile; >0 = into the stone)
    1  inner-band coverage: the four oaths (kind 'text', variants A and C) or, in the wordless
       variant B (kind 'orn'), a carved braid with a fire rising at each emissary's station
    2  frieze coverage: the carved stone band in the floor around the emissaries (it replaces
       v1's "together" ring) - a ring of 48 small carved fires on a running line, the grander one
       in each emissary's shadow
    3  ornament coverage (border rings, separators, hearth rays)
plus a mip pyramid, cached to renders/accord_cache/textmaps_<kind>_v7.npz.

Everything is laid on arcs: "up" = radially outward; text reads clockwise seen from above (so the
block at the top of frame reads left->right).
"""
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(ROOT, 'renders', 'accord_cache')
CINZEL = os.path.join(ROOT, 'assets', 'fonts', 'Cinzel.ttf')

# ------------------------------------------------------------------ layout ---
TEX_N = 4096
TEX_R = 3.62                      # texture covers [-TEX_R, TEX_R]^2 metres
TEXEL = 2 * TEX_R / TEX_N         # ~1.77 mm

OATH_CAP = 0.150                  # cap height (m)
OATH_R_IN = 1.40                  # baseline radius of 2nd line (inner)
OATH_LEAD = 1.50 * OATH_CAP       # baseline to baseline
OATH_R_OUT = OATH_R_IN + OATH_LEAD
OATH_TRACK = 0.035                # extra tracking, in cap heights per char
OATHS = [('NO SINGLE HAND', 'SHALL HOLD IT'),
         ('NO FASTER THAN', 'WE CAN SEE'),
         ('NO FORGE', 'IN THE DARK'),
         ('WHAT IT GIVES,', 'IT GIVES TO ALL')]
# block centres: oath I at world +y (north), then clockwise in 90 deg steps
OATH_THETA = [math.pi / 2 - k * math.pi / 2 for k in range(4)]

BAND_IN = OATH_R_IN - 0.085       # carved border rings of the inner band
BAND_OUT = OATH_R_OUT + OATH_CAP + 0.085

TABLE_R = 1.97
FLOOR_BAND = (2.64, 3.52)         # smooth stone ring set in the floor
FRIEZE_BORDER = (2.700, 3.455)    # its carved border rings
FRIEZE_R = 3.078                  # centre line of the frieze
N_MEDAL = 12                      # one medallion behind each emissary
MEDAL_THETA0 = math.radians(105.0)

HEARTH_R = 0.50
RAY_R = (0.70, 1.16)              # 12 carved sun-rays around the hearth

# variant B: the inner band is a braid of two strands; at each emissary's station the strands
# cross and a fire rises from the crossing
BRAID_R = 1.442                   # centre line of the braid
BRAID_H = 0.056                   # its amplitude
BRAID_W = 0.0145                  # strand half-width
STATION_FIRE = (1.47, 0.35)       # base radius, height of the station fires
FRIEZE_BASE = 2.878               # the outer frieze's running base line (its fires rise from here)
FRIEZE_FIRE_H = (0.46, 0.53)      # fire height, and the grander fire behind each emissary

MAX_DEPTH = 0.016                 # m, V-cut max depth
WALL_TAN = 1.35                   # depth per metre of inside-distance (V wall slope)
SS = 4                            # supersampling for rasterisation / distance fields


def kind_for(variant):
    return 'orn' if variant.upper() == 'B' else 'text'


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


def place_on_arc(sd_strip, base_row, u_center, r_base, theta_c, rmin, rmax, out_sd, half_ang=None):
    """Warp a strip signed-distance field onto the ring, max-combining into out_sd.
    Strip columns run clockwise (u = u_center + (theta_c - theta) * r_base / TEXEL); rows run
    inward (row = base_row - (rho - r_base) / TEXEL). half_ang limits the angular extent."""
    rho, theta = _polar()
    Hs, Ws = sd_strip.shape
    if half_ang is None:
        half_ang = (Ws * 0.5 + 4) / r_base * TEXEL
    ri = int(max(0, (TEX_R - rmax) / TEXEL - 2))
    rf = int(min(TEX_N, (TEX_R + rmax) / TEXEL + 2))
    # angular bounding box: only the rows/cols near this arc
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


# ------------------------------------------------------ ornament drawing ---

class Strip:
    """A supersampled drawing canvas in arc coordinates: s (m, clockwise along the arc at r_base)
    and v (m, radially outward). Strokes are unions of stamped discs (smooth tapered edges)."""

    def __init__(self, s0, s1, v0, v1):
        self.s0, self.s1, self.v0, self.v1 = s0, s1, v0, v1
        self.k = SS / TEXEL                                     # px per metre
        self.W = int(math.ceil((s1 - s0) * self.k / SS)) * SS
        self.H = int(math.ceil((v1 - v0) * self.k / SS)) * SS
        self.img = Image.new('L', (self.W, self.H), 0)
        self.d = ImageDraw.Draw(self.img)

    def px(self, s, v):
        return ((np.asarray(s) - self.s0) * self.k, (self.v1 - np.asarray(v)) * self.k)

    def stamp(self, pts, rad, fill=255):
        """Union of discs along a polyline with per-point radii (m)."""
        pts = np.asarray(pts, float)
        rad = np.asarray(rad, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        out_p, out_r = [pts[0]], [rad[0]]
        for k in range(len(seg)):
            n = max(1, int(math.ceil(seg[k] / max(0.25 * min(rad[k], rad[k + 1]), 0.0006))))
            for j in range(1, n + 1):
                a = j / n
                out_p.append(pts[k] * (1 - a) + pts[k + 1] * a)
                out_r.append(rad[k] * (1 - a) + rad[k + 1] * a)
        P = np.array(out_p)
        x, y = self.px(P[:, 0], P[:, 1])
        r = np.array(out_r) * self.k
        for xi, yi, ri in zip(x, y, r):
            self.d.ellipse([xi - ri, yi - ri, xi + ri, yi + ri], fill=fill)

    def disc(self, s, v, r, fill=255):
        x, y = self.px(s, v)
        rr = r * self.k
        self.d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill)

    def sdf(self):
        return _signed_dist(np.asarray(self.img) > 127)

    def base_row(self, v=0.0):
        """Row (1x texels) of the given v."""
        return (self.v1 - v) / TEXEL


def tongue(st, base, length, w0, h0=0.0, wave=0.35, flick=0.0, phase=0.0, n=120, fill=255, wpow=0.9):
    """A flame tongue: centre line by heading integration, heading(u) = h0 + wave*sin(1.7 pi u + phase)
    + flick*u^4 (h0 in rad from +v, + leans clockwise); width w0 tapering to a fine point."""
    u = np.linspace(0, 1, n)
    hd = h0 + wave * np.sin(2 * np.pi * 0.85 * u + phase) + flick * u ** 4
    ds = length / (n - 1)
    s = base[0] + np.concatenate([[0], np.cumsum(np.sin(hd[:-1]) * ds)])
    v = base[1] + np.concatenate([[0], np.cumsum(np.cos(hd[:-1]) * ds)])
    w = w0 * (1 - u) ** wpow * (0.78 + 0.22 * np.sin(np.pi * np.clip(1.8 * u, 0, 1))) + 0.0008
    st.stamp(np.stack([s, v], -1), w, fill=fill)


def fire3(st, s0, v0, H, W, lean=0.15):
    """A small carved fire: a tall centre tongue and two flanks that sweep out and flick up."""
    tongue(st, (s0, v0), H, 0.30 * W, h0=lean * 0.6, wave=0.22, flick=-0.35)
    tongue(st, (s0 - 0.10 * W, v0 + 0.02 * H), 0.62 * H, 0.21 * W, h0=-0.55 + lean * 0.5, wave=0.20,
           flick=0.9, phase=0.5)
    tongue(st, (s0 + 0.10 * W, v0 + 0.02 * H), 0.50 * H, 0.19 * W, h0=0.62 + lean * 0.5, wave=-0.18,
           flick=-0.9, phase=0.5)


def frieze_strip(grand):
    """One unit of the outer frieze (a fire, then a single small tongue, on a running base line),
    centred on its fire. Four units per emissary; the unit behind each emissary is `grand`.
    Returns (signed distance strip in 1x texels, row of r = FRIEZE_R, column of the fire, pitch)."""
    P = 2 * math.pi * FRIEZE_R / (4 * N_MEDAL)
    m = 0.08
    st = Strip(-P / 2 - m, P / 2 + m, -0.36, 0.36)
    vb = FRIEZE_BASE - FRIEZE_R
    s = np.linspace(-P / 2 - m, P / 2 + m, 400)
    st.stamp(np.stack([s, vb + 0.016 * np.sin(2 * np.pi * s / P + 0.5 * np.pi)], -1), np.full(s.size, 0.0105))
    H = FRIEZE_FIRE_H[1] if grand else FRIEZE_FIRE_H[0]
    fire3(st, 0.0, vb + 0.012, H, 0.24 if grand else 0.22)
    for e in (-1, 1):                               # the half-pitch tongue, shared with the neighbour
        tongue(st, (e * P / 2, vb + 0.012), 0.24, 0.030, h0=0.25, wave=0.25, flick=-0.3)
    if grand:
        for e in (-1, 1):
            st.disc(e * 0.10, vb - 0.055, 0.012)
    return st.sdf(), st.base_row(0.0), (0.0 - st.s0) / TEXEL, P


def braid_strip():
    """Variant B: one station of the inner band - a two-strand braid whose strands cross at the
    station, with a fire rising from the crossing. Returns (sd strip, row of r = BRAID_R, column of
    the station, pitch)."""
    L = 2 * math.pi * BRAID_R / N_MEDAL
    m = 0.08
    st = Strip(-L / 2 - m, L / 2 + m, CROWN_V0, CROWN_V1)
    D = L / 4.0                                     # crossings every quarter station (one at the station)
    s = np.linspace(-L / 2 - m, L / 2 + m, 900)
    a = BRAID_H * np.sin(np.pi * s / D)
    A = np.stack([s, a], -1)
    B = np.stack([s, -a], -1)
    w = np.full(s.size, BRAID_W)
    st.stamp(A, w)
    st.stamp(B, w)
    for j in range(-3, 4):                          # over/under alternate along each strand
        c = j * D
        over = A if j % 2 == 0 else B
        sel = np.abs(over[:, 0] - c) < 0.042
        if sel.sum() < 2:
            continue
        st.stamp(over[sel], np.full(sel.sum(), BRAID_W + 0.011), fill=0)
        sel = np.abs(over[:, 0] - c) < 0.07
        st.stamp(over[sel], np.full(sel.sum(), BRAID_W))
    base = STATION_FIRE[0] - BRAID_R
    fire3(st, 0.0, base, STATION_FIRE[1], 0.21, lean=0.10)
    for e in (-1, 1):                               # small tongues over the braid's outer crests
        tongue(st, (e * 1.5 * D, BRAID_H + BRAID_W + 0.01), 0.17, 0.024, h0=0.22, wave=0.22, flick=-0.3)
    return st.sdf(), st.base_row(0.0), (0.0 - st.s0) / TEXEL, L


CROWN_V0 = BAND_IN + 0.02 - BRAID_R          # braid strip extent (v) inside the inner band
CROWN_V1 = BAND_OUT - 0.02 - BRAID_R


# -------------------------------------------------------------- build all ---

def build(kind='text', verbose=True):
    rho, theta = _polar()
    N = TEX_N
    t = TEXEL
    sd_in = np.full((N, N), -1e3, np.float32)
    sd_fr = np.full((N, N), -1e3, np.float32)
    sd_orn = np.full((N, N), -1e3, np.float32)

    if kind == 'text':
        for k, (l1, l2) in enumerate(OATHS):
            for text, rb in ((l1, OATH_R_OUT), (l2, OATH_R_IN)):
                sd, base, width, pad = oath_strip(text, OATH_CAP)
                ucen = pad + width / 2
                place_on_arc(sd, base, ucen, rb, OATH_THETA[k], rb - 0.06, rb + OATH_CAP * 1.35, sd_in)
            if verbose:
                print('oath', k, 'placed', flush=True)
    else:
        sd, base, ucen, L = braid_strip()
        for k in range(N_MEDAL):
            th = MEDAL_THETA0 + k * 2 * math.pi / N_MEDAL
            place_on_arc(sd, base, ucen, BRAID_R, th, BAND_IN + 0.02, BAND_OUT - 0.02, sd_in,
                         half_ang=0.5 * L / BRAID_R)
        if verbose:
            print('braid placed', flush=True)

    # the frieze in the floor band: 4 fires per emissary, the grander one in each emissary's shadow
    for grand in (True, False):
        sd, base, ucen, P = frieze_strip(grand)
        for k in range(4 * N_MEDAL):
            if (k % 4 == 0) != grand:
                continue
            th = MEDAL_THETA0 + k * 2 * math.pi / (4 * N_MEDAL)
            place_on_arc(sd, base, ucen, FRIEZE_R, th, FRIEZE_BORDER[0] + 0.02, FRIEZE_BORDER[1] - 0.02, sd_fr,
                         half_ang=0.5 * P / FRIEZE_R)
    if verbose:
        print('frieze placed', flush=True)

    # ornaments: border rings, separators, hearth rays (signed distance in texels), computed in
    # row blocks to keep memory low
    def ring(r, hw, sl):
        return (hw - np.abs(rho[sl] - r)) / t

    X = (np.arange(N, dtype=np.float32) + 0.5) * t - TEX_R
    rings = [(BAND_IN, 0.0065), (BAND_IN - 0.028, 0.003), (BAND_OUT, 0.0065), (BAND_OUT + 0.028, 0.003),
             (FRIEZE_BORDER[0], 0.0055), (FRIEZE_BORDER[0] - 0.022, 0.0028),
             (FRIEZE_BORDER[1], 0.0065), (FRIEZE_BORDER[1] + 0.024, 0.003)]
    rm = 0.5 * (OATH_R_IN + OATH_R_OUT + OATH_CAP)
    for y0 in range(0, N, 512):
        sl = slice(y0, y0 + 512)
        blk = sd_orn[sl]
        for r, hw in rings:
            np.maximum(blk, ring(r, hw, sl), out=blk)
        Yb = (-X)[sl][:, None]
        if kind == 'text':
            # diamond separators between the oath blocks, with two small dots each
            for k in range(4):
                th = OATH_THETA[k] - np.pi / 4
                cx, cy = rm * np.cos(th), rm * np.sin(th)
                dx = X[None, :] - cx
                dy = Yb - cy
                rr = dx * np.cos(th) + dy * np.sin(th)
                tt = -dx * np.sin(th) + dy * np.cos(th)
                dia = (0.07 - (np.abs(rr) / 1.0 + np.abs(tt) / 0.55)) * 0.55
                np.maximum(blk, dia / t, out=blk)
                for s in (-1, 1):
                    ddx = X[None, :] - (cx + s * 0.13 * -np.sin(th))
                    ddy = Yb - (cy + s * 0.13 * np.cos(th))
                    dot = 0.018 - np.sqrt(ddx * ddx + ddy * ddy)
                    np.maximum(blk, dot / t, out=blk)
        # 12 tapered sun-rays around the hearth
        rh, thb = rho[sl], theta[sl]
        for k in range(12):
            th = np.pi / 2 + k * np.pi / 6
            dth = (thb - th + np.pi) % (2 * np.pi) - np.pi
            along = (rh - RAY_R[0]) / (RAY_R[1] - RAY_R[0])
            halfw = 0.018 * (1 - np.clip(along, 0, 1)) + 0.002
            inside = halfw - np.abs(dth) * rh
            endcap = np.minimum(rh - RAY_R[0], RAY_R[1] - rh)
            np.maximum(blk, np.minimum(inside, endcap) / t, out=blk)

    def depth_from(sd):
        d = np.clip(sd * t * WALL_TAN, 0, None)
        # soft cap (rounded groove bottom)
        return (MAX_DEPTH * (1 - np.exp(-d / MAX_DEPTH))).astype(np.float32)

    def cov(sd):
        return np.clip(sd + 0.5, 0, 1).astype(np.float32)

    tex = np.empty((N, N, 4), np.float32)
    tex[..., 0] = np.maximum(np.maximum(depth_from(sd_in), depth_from(sd_fr) * 0.9), depth_from(sd_orn) * 0.8)
    tex[..., 1] = cov(sd_in)
    del sd_in
    tex[..., 2] = cov(sd_fr)
    del sd_fr
    tex[..., 3] = cov(sd_orn)
    return tex


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


def load(kind='text', rebuild=False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f'textmaps_{kind}_v7.npy')
    meta = path[:-4] + '_meta.npz'
    if os.path.exists(path) and os.path.exists(meta) and not rebuild:
        flat = np.load(path, mmap_mode='r')          # memory-mapped: shared page cache between workers
        z = np.load(meta)
        return flat, z['offs'], z['sizes']
    tex = build(kind)
    flat, offs, sizes = mip_pyramid(tex)
    del tex
    tmp = path + '.tmp.npy'
    np.save(tmp, flat)
    os.replace(tmp, path)
    np.savez(meta, offs=offs, sizes=sizes)
    return np.load(path, mmap_mode='r'), offs, sizes


# ------------------------------------------------------------------- test ---

def preview(kind, out_png, r0=None, size=2048):
    """Top-down preview of the carved maps (depth shaded + coverage tint)."""
    flat, offs, sizes = load(kind)
    lvl = int(round(math.log2(TEX_N / size)))
    n = int(sizes[lvl])
    L = np.asarray(flat[offs[lvl]:offs[lvl] + n * n]).reshape(n, n, 4)
    dep = L[..., 0] / MAX_DEPTH
    gy, gx = np.gradient(dep)
    shade = np.clip(0.55 + 3.0 * (gx - gy), 0, 1)
    img = np.stack([shade * 0.62, shade * 0.60, shade * 0.56], -1)
    img[..., 0] += 0.9 * L[..., 1]
    img[..., 1] += 0.6 * L[..., 1]
    img[..., 0] += 0.8 * L[..., 2]
    img[..., 1] += 0.55 * L[..., 2] + 0.25 * L[..., 3]
    img[..., 2] += 0.25 * L[..., 3]
    cv2.imwrite(out_png, (np.clip(img, 0, 1)[..., ::-1] * 255).astype(np.uint8))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'preview':
        preview(sys.argv[2], sys.argv[3])
    else:
        for kd in (sys.argv[1:] or ['text', 'orn']):
            flat, offs, sizes = load(kd, rebuild=True)
            print(kd, flat.shape, offs, sizes)
