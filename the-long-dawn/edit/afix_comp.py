"""A-FIX comps for EDIT's EDL.TRANS windows (film A, 28 Sep). Owner: lane A-FIX; wired by EDIT (assemble._transitions).

Every function works on float32 sRGB frames in [0, 1], shape (H, W, 3), at any scale (positions and sizes in a
window's spec are in 1920-wide pixels and are scaled to the frame). They are pure functions of their inputs and the
cut frame f, so EDIT's parallel workers and segment keys stay deterministic.

Kinds (the window dict `t` carries the spec; `o` = the outgoing side, `i` = the incoming side, as _transitions
builds them: o holds its last frame after t['cut'], i holds its first frame before it):

  bloom      the push INTO the false dawn's glow: the outgoing plate is pushed in on the glow (digital zoom, eased
             in) while a bloom grows out of the glow's own position and screens the frame to white by t['cut'].
             A 536-560 (the slam at 560 was one frame, luma 64 -> 226).
  vision     THE FIRE'S VISIONS (A5 THE PROMISE, A9 THE DEAD VALLEY): the vision is seen only inside an organic
             window of the fire's light that opens out of the fire, breathes with it (once a bar), is bordered by a
             warm rim, shimmers with heat, and closes back into the fire. Outside the window: the outgoing side
             (A5: the calm flame on black) or a synthetic glow (A9: the fire's white, dimming to ember-dark).
  iceheart   THE BRINK's fall (A8): the warm orange "blast" is graded, as the camera falls, into the thinking fire's
             own ice-white (cool, desaturated, lifted), so it reads as falling into the fire, not an explosion.
  ember      E4: the drifting ember of A10/A11 made a story object (a 3-5 px point -> a flickering 12-20 px coal with
             a soft glow), found in each frame (the warmest point near its known track).
"""
import math

import cv2
import numpy as np

REF_W = 1920.0


def _lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _srgb(x):
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _ss(a, b, x):
    u = min(max((x - a) / float(b - a), 0.0), 1.0) if b != a else float(x >= a)
    return u * u * (3.0 - 2.0 * u)


def _key(keys, f):
    """Linear interpolation in a list of (frame, value...) keys (clamped)."""
    if f <= keys[0][0]:
        return np.asarray(keys[0][1:], np.float64)
    for (f0, *v0), (f1, *v1) in zip(keys, keys[1:]):
        if f <= f1:
            u = (f - f0) / float(f1 - f0)
            return np.asarray(v0, np.float64) * (1 - u) + np.asarray(v1, np.float64) * u
    return np.asarray(keys[-1][1:], np.float64)


def _grid(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return xx, yy


# ------------------------------------------------------------------------------------------------ bloom (TR 560)
def bloom(o, i, f, t):
    """t: f0, cut, center=(x, y) in 1920 px, zoom=1.14. Before the cut: the outgoing plate pushed in on the glow and
    screened to white by a bloom from the glow; from the cut on: the incoming frame (EMBERS' white)."""
    if f >= t['cut']:
        return i
    H, W = o.shape[:2]
    k = W / REF_W
    a = (f - t['f0'] + 1.0) / float(t['cut'] - t['f0'])          # (0, 1]: 1 on the frame before the cut
    cx, cy = (np.asarray(t.get('center', (1124.0, 464.0))) * k).tolist()
    z = 1.0 + (t.get('zoom', 1.14) - 1.0) * a ** 2.2                 # the push accelerates into the light
    M = np.float32([[z, 0.0, cx * (1 - z)], [0.0, z, cy * (1 - z)]])
    p = cv2.warpAffine(o, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    xx, yy = _grid(H, W)
    sig = (50.0 + 650.0 * a ** 2.0) * k
    d2 = ((xx - cx) ** 2 + ((yy - cy) * 1.35) ** 2) / (2.0 * sig * sig)
    I = 0.20 + 3.6 * a ** 2.6                                        # the core saturates and its white spreads out
    B = I * np.exp(-d2) + _ss(0.78, 1.0, a) * 1.02
    B = np.clip(B, 0.0, 1.0)[..., None]
    out = 1.0 - (1.0 - _lin(p)) * (1.0 - B * np.float32([0.98, 0.99, 1.0]))   # cold white, like the glow
    return np.clip(_srgb(out), 0.0, 1.0)


# ---------------------------------------------------------------------------------------------------- vision
def _window(H, W, f, t, k):
    """The vision window's alpha (0..1), its rim band, and its centre, at cut frame f."""
    cx, cy = (_key(t['track'], f) * k).tolist()                      # the fire (or the collapse point)
    op = _ss(t['open'][0], t['open'][1], f)
    cl = 1.0 - _ss(t['close'][0], t['close'][1], f)
    s = (op * cl) ** (0.8 if op < 1.0 else 1.2)                       # opens out of the fire, closes back into it
    br = 1.0 + t.get('breath', 0.035) * math.sin(2 * math.pi * (f - t.get('bar0', 0)) / 80.0 - math.pi / 2)
    rx0, ry0 = t.get('r0', (30.0, 60.0))
    rx1, ry1 = t['r1']
    rx = (rx0 + (rx1 - rx0) * s) * br * k
    ry = (ry0 + (ry1 - ry0) * s) * br * k
    oy = t.get('lift', 0.0) * s * k                                    # the window's centre rises off the fire as it opens
    xx, yy = _grid(H, W)
    dx, dy = (xx - cx) / max(rx, 1.0), (yy - (cy - oy)) / max(ry, 1.0)
    rr = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)
    tt = f / 24.0
    wob = (0.045 * np.sin(3 * th + 1.7 * tt) + 0.030 * np.sin(5 * th - 2.3 * tt + 1.1)
           + 0.020 * np.sin(8 * th + 3.1 * tt + 2.0) + 0.012 * np.sin(13 * th - 4.3 * tt))   # the edge flickers
    tongues = 0.035 * np.maximum(0.0, np.sin(21 * th + 6.1 * tt)) * np.maximum(0.0, np.sin(9 * th - 2.9 * tt + 0.7))
    edge = rr - (1.0 + wob * (0.4 + 0.6 * s) + tongues * s)
    fe = t.get('feather', 0.07)
    a = np.clip(0.5 - edge / fe, 0.0, 1.0)
    a = a * a * (3.0 - 2.0 * a)
    # the rim is the fire's own light, not an outline (director): thin, uneven along its length, flickering, with
    # flecks of ember riding the edge and lifting off it
    ph = 0.5 + 0.5 * (np.sin(7 * th + 2.1 * tt) * np.sin(17 * th - 3.3 * tt + 1.0))
    ph = np.clip(ph + 0.30 * np.sin(31 * th + 5.0 * tt + 0.4) + 0.15 * np.sin(53 * th - 7.9 * tt), 0.0, 1.0)
    uneven = 0.18 + 0.82 * ph ** 1.6
    line = np.exp(-(edge / (fe * 0.12)) ** 2)
    outer = 0.09 * np.exp(-np.maximum(edge, 0.0) / (fe * 0.9)) * (edge > 0) \
        * (0.6 + 0.4 * (0.5 + 0.5 * np.sin(3 * th + 1.3 * tt)))              # a low, smooth warmth outside
    H, W = xx.shape
    ix = (xx / (2.5 * k)).astype(np.int64)
    iy = ((yy + 0.9 * f * k) / (2.5 * k)).astype(np.int64)                # the flecks drift up off the edge
    hsh = (ix * 73856093) ^ (iy * 19349663) ^ (int(f // 3) * 83492791)
    hsh = (hsh & 0xFFFF).astype(np.float32) / 65535.0
    fleck = (hsh > 0.992) * np.exp(-((edge + 0.25 * fe) / (fe * 0.55)) ** 2) * 1.6
    rim = (line * uneven + outer + fleck) * s
    return a.astype(np.float32), rim.astype(np.float32), (cx, cy - oy), s


def _shimmer(img, f, amp_px, k):
    """Heat shimmer: a small, rising displacement."""
    if amp_px <= 0.0:
        return img
    H, W = img.shape[:2]
    xx, yy = _grid(H, W)
    tt = f / 24.0
    sx = amp_px * k * (np.sin(yy / (23.0 * k) + 5.1 * tt) * 0.6 + np.sin(yy / (9.0 * k) - 7.3 * tt + xx / (41.0 * k)) * 0.4)
    sy = amp_px * k * 0.5 * np.sin(xx / (31.0 * k) + 3.7 * tt + yy / (57.0 * k))
    return cv2.remap(img, xx + sx.astype(np.float32), yy + sy.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _flame_key(i, f, t, k):
    """The fire's own pixels in the incoming frame (A5): a luminance key in a box round the flame's track."""
    H, W = i.shape[:2]
    fx, fy = (_key(t['track'], f) * k).tolist()
    bx, by0, by1 = [v * k for v in t.get('flame_box', (95.0, 270.0, 90.0))]
    x0, x1 = max(0, int(fx - bx)), min(W, int(fx + bx))
    y0, y1 = max(0, int(fy - by0)), min(H, int(fy + by1))
    fm = np.zeros((H, W), np.float32)
    if x1 <= x0 or y1 <= y0:
        return fm
    L = i[y0:y1, x0:x1].mean(2)
    m = np.clip((L - 0.22) / 0.33, 0.0, 1.0)
    fm[y0:y1, x0:x1] = m * m * (3.0 - 2.0 * m)
    return cv2.GaussianBlur(fm, (0, 0), 1.5 * k)


def _fill(img_l, hole, k):
    """Fill a soft hole with its blurred surroundings (normalised convolution)."""
    w = (1.0 - np.clip(hole * 1.6, 0.0, 1.0))[..., None]
    num = cv2.GaussianBlur(img_l * w, (0, 0), 28.0 * k)
    den = cv2.GaussianBlur(w, (0, 0), 28.0 * k)[..., None] if w.ndim == 3 else cv2.GaussianBlur(w, (0, 0), 28.0 * k)
    den = den.reshape(num.shape[:2] + (1,))
    return img_l * w + (num / np.maximum(den, 1e-4)) * (1.0 - w)


def vision(o, i, f, t):
    """t: track=[(f, x, y), ...] (1920 px), open=(a, b), close=(a, b), r1=(rx, ry), r0, lift, breath, bar0,
    rim=(r, g, b) linear, rim_gain, inside_gain, inside_tint=(r, g, b), shimmer px, outside='o' | 'glow',
    glow_keys=[(f, r, g, b) linear] for outside='glow'."""
    H, W = i.shape[:2]
    k = W / REF_W
    a, rim, (cx, cy), s = _window(H, W, f, t, k)
    fm = _flame_key(i, f, t, k) if t.get('flame_key') else None
    src = i
    if fm is not None:                                                  # the fire stands in front of its vision
        src = np.clip(_srgb(_fill(_lin(i), cv2.dilate(fm, np.ones((5, 5), np.uint8)), k)), 0.0, 1.0)
    vis = _shimmer(src, f, t.get('shimmer', 1.6) * s, k)
    vis_l = _lin(vis) * np.float32(t.get('inside_tint', (1.0, 1.0, 1.0))) * t.get('inside_gain', 1.0)
    gl = t.get('inside_glow', 0.35)                                   # a vision is luminous: a soft glow of itself
    if gl > 0.0:
        vis_l = vis_l + gl * cv2.GaussianBlur(vis_l, (0, 0), 9.0 * k)
    if t.get('outside', 'o') == 'o':
        out_l = _lin(o)
    else:
        g = _key(t['glow_keys'], f).astype(np.float32)
        xx, yy = _grid(H, W)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (t.get('glow_r', 900.0) * k)
        out_l = (g[None, None, :] * (t.get('glow_floor', 0.55) + (1 - t.get('glow_floor', 0.55))
                                     * np.exp(-d * d))[..., None]).astype(np.float32)
    rim_c = np.float32(t.get('rim', (1.0, 0.72, 0.38))) * t.get('rim_gain', 0.35)
    out = out_l * (1.0 - a[..., None]) + vis_l * a[..., None] + rim[..., None] * rim_c
    if fm is not None:
        out = out * (1.0 - fm[..., None]) + _lin(i) * fm[..., None]
    return np.clip(_srgb(out), 0.0, 1.0)


# -------------------------------------------------------------------------------------------------- iceheart
def iceheart(o, i, f, t):
    """t: ramp=(a, b) cut frames over which the fall's warm light becomes the fire's ice-white; lift, sat."""
    img = i if f >= t['cut'] else o
    w = _ss(t['ramp'][0], t['ramp'][1], f)
    if w <= 0.0:
        return img
    L = _lin(img)
    Y = (0.2126 * L[..., 0] + 0.7152 * L[..., 1] + 0.0722 * L[..., 2])[..., None]
    ice = np.float32([0.86, 0.94, 1.0]) * Y * t.get('gain', 1.35)          # the thinking fire: ice-white
    gold = np.float32([1.0, 0.78, 0.45]) * Y                                  # a little of its gold edge kept
    tgt = ice * 0.85 + gold * 0.15
    tgt = 1.0 - (1.0 - tgt) * (1.0 - t.get('lift', 0.25) * w)                # and it lifts toward white
    out = L * (1.0 - w) + tgt * w
    return np.clip(_srgb(out), 0.0, 1.0)


# ----------------------------------------------------------------------------------------------------- ember
def ember(o, i, f, t):
    """t: track=[(f, x, y)] (1920 px; where the baked ember is expected), size px, gain. The baked 3-5 px point is
    found near the track (the warmest bright pixel) and given a coal's core, a flicker and a soft glow."""
    img = i if f >= t['cut'] else o
    H, W = img.shape[:2]
    k = W / REF_W
    ex, ey = (_key(t['track'], f) * k).tolist()
    r = int(40 * k) + 2
    x0, x1 = max(0, int(ex) - r), min(W, int(ex) + r + 1)
    y0, y1 = max(0, int(ey) - r), min(H, int(ey) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return img
    win = img[y0:y1, x0:x1]
    warm = win[..., 0] - win[..., 2] + 0.5 * win.mean(2)
    yy_, xx_ = np.unravel_index(int(np.argmax(warm)), warm.shape)
    if warm[yy_, xx_] < 0.25:                                              # no ember in this frame
        return img
    px, py = x0 + xx_ + 0.5, y0 + yy_ + 0.5
    fl = 1.0 + 0.18 * math.sin(f * 0.61) + 0.10 * math.sin(f * 1.37 + 1.0) + 0.06 * math.sin(f * 2.9 + 2.0)
    fade = _ss(t.get('in', (0, 1))[0], t.get('in', (0, 1))[1], f)
    xx, yy = _grid(H, W)
    d2 = (xx - px) ** 2 + (yy - py) ** 2
    sz = t.get('size', 7.0) * k
    core = np.exp(-d2 / (2 * (0.45 * sz) ** 2))
    halo = np.exp(-d2 / (2 * (2.2 * sz) ** 2))
    g = t.get('gain', 1.0) * fl * fade
    add = (core[..., None] * np.float32([1.0, 0.55, 0.20]) * 0.9 + halo[..., None] * np.float32([0.9, 0.35, 0.08]) * 0.10)
    out = _lin(img) + add.astype(np.float32) * g
    return np.clip(_srgb(out), 0.0, 1.0)


KINDS = dict(bloom=bloom, vision=vision, iceheart=iceheart, ember=ember)


def apply(kind, o, i, f, t):
    """EDIT's entry point: assemble._transitions calls this for any kind in KINDS."""
    return KINDS[kind](o, i, f, t)


# ------------------------------------------------------------------------------------ A's windows (EDL.TRANS['A'])
# EDIT: `EDL.TRANS['A'] = EDL.TRANS['A'] + afix_comp.A_TRANS`, and in assemble._transitions dispatch any kind in
# afix_comp.KINDS to afix_comp.apply(kind, o, i, f, t) ('dissolve' stays EDIT's own). Looked at on 28 Sep in
# _local_logs/review/a_fix/comp_t2.jpg (bloom) and comp_t3.jpg (visions, ember).
_PROMISE_TRACK = [(1280, 957, 404), (1300, 957, 409), (1320, 958, 414), (1340, 958, 445), (1360, 958, 478),
                  (1380, 957, 486), (1400, 957, 481), (1420, 958, 484), (1440, 958, 511)]   # the flame (measured)
_EMBER_TRACK = [(2836, 1030, 160), (2846, 1021, 215), (2850, 1009, 269), (2856, 989, 338), (2860, 976, 375),
                (2866, 960, 423), (2872, 953, 464), (2880, 947, 504), (2890, 954, 530), (2900, 962, 546),
                (3359, 958, 547)]                                    # A10's drifting ember, then A11's (measured)
A_TRANS = [
    # A2 -> A3: the push INTO the glow, which floods to white on bar 8 (was a one-frame slam at 560)
    dict(f0=536, f1=560, cut=560, kind='bloom', center=(1124, 464), zoom=1.14,
         note='A-FIX TR560: the push into the false dawn arrives at white'),
    # A3: the white falls evenly to the first letters (was a shrinking disc, a spotlight iris, 600-672)
    dict(f0=596, f1=672, cut=596, kind='dissolve', note='A-FIX TR600: held white -> the live frames, evenly'),
    # A5 THE PROMISE, bars 17-18: a vision in the fire's light (was the fire set down in a real valley)
    dict(f0=1280, f1=1440, cut=1280, kind='vision', track=_PROMISE_TRACK, open=(1280, 1318), close=(1396, 1436),
         r0=(40, 90), r1=(600, 300), lift=60, breath=0.035, bar0=1280, feather=0.07, flame_key=True,
         rim=(1.0, 0.75, 0.42), rim_gain=0.5, inside_glow=0.35, shimmer=1.6, outside='glow', glow_floor=1.0,
         glow_keys=[(1280, 0.004, 0.0036, 0.0032), (1440, 0.004, 0.0036, 0.0032)],
         note="A-FIX V1: THE PROMISE seen in the fire's light"),
    # A8 over the rim: the fall ends in the thinking fire's ice-white, not a warm blast
    # (the render's own white now lands ON 2640, the IMPACT: edge.py A-FIX; so no lift here, only the colour)
    dict(f0=2520, f1=2640, cut=2520, kind='iceheart', ramp=(2528, 2620), gain=1.35, lift=0.0,
         note='A-FIX E3: the fall into the ice-white fire'),
    # A9 THE DEAD VALLEY, bars 34-35: the fire's second vision, in the same window, closing to a point
    dict(f0=2640, f1=2800, cut=2684, kind='vision', track=[(2640, 958, 440), (2800, 958, 440)], open=(2656, 2692),
         close=(2762, 2796), r0=(10, 10), r1=(600, 300), lift=60, breath=0.035, bar0=2640, feather=0.07,
         rim=(1.0, 0.9, 0.8), rim_gain=0.7, inside_glow=0.25, shimmer=1.2, outside='glow', glow_r=520,
         glow_floor=0.0, glow_keys=[(2640, 1.0, 1.0, 1.0), (2664, 0.9, 0.86, 0.8), (2692, 0.12, 0.07, 0.035),
                                    (2730, 0.05, 0.028, 0.014), (2780, 0.02, 0.011, 0.005), (2800, 0.0, 0.0, 0.0)],
         note="A-FIX V2: THE DEAD VALLEY in the fire's light (the white's iris 2662-2676 is never seen)"),
    # A10-A11: the ember as a story object; at 3359 it sits where strike 1's spark lands at 3360 (h1_A)
    dict(f0=2836, f1=3360, cut=2836, kind='ember', track=_EMBER_TRACK, size=14, gain=1.0, **{'in': (2836, 2850)},
         note='A-FIX E4: the ember, 12-20 px with a soft glow; match cut to strike 1'),
]
