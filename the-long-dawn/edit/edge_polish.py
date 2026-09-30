"""A6/A7 light, lane polishedge (second pass): GAIN only, on the current plate only.

Owner's review of the first pass (30 Sep): no temporal pixel averaging (A7's sparks streaked, the towers ghosted)
and no cross-dissolves at the camera changes. So every output pixel here is the current plate's own pixel times a
gain; statistics of neighbouring plates decide the gains, and no pixel of another frame reaches the picture.

1. Exposure match at the four camera changes, which stay hard cuts on their beats. Near the cut a per-channel gain
   (scene-linear light) grades the frames next to it from near the other side's mean exposure back to their own
   over n frames (log-exposure, smoothstep): the incoming shot's first frames, the outgoing tail, or both sharing
   the gap (WINDOW['ramps']). The cut frame itself stays a cut; no pixel of the other side is used.
2. The beat surges (A6's forge throats before 1760; A7's crater and towers). A7's plates carry two kinds: the crater
   and fire jump on the beat frame (then decay slowly), and each near tower flares 2-3x for ONE frame at its own
   delay after the beat (scene_b's leap-frogging race, a height jump smeared by the shutter). A coarse
   grid of cells (GX x GY) follows each cell's light over time; a cell's one-frame spike keeps KEEP of its
   log-excess, and every rise and fall gets an ATTACK (three frames before it) and a RELEASE (~10 frames after it).
   The cell log-gains are clamped, smoothed and interpolated into a gain field for the current plate.

Statistics come from each plate's own file decoded at one fixed low resolution, whatever the output scale, so the
preview, the review frames and the master get the same gains. Gains act in scene-linear light through the finish's
own analytic inverse of the renderers' tone path (finish/filmfinish.py), so highlights roll off as the renderer's
own surges do; the change is added to the plate (plate + F(g v) - F(v)), which is exact where the gain is one.
Frames outside RAMPS and SURGES return None: the assembler then plays its plain frame, bit for bit.
"""
import math
import os
import sys

import cv2
import numpy as np

GX, GY = 48, 20            # the gain grid: cells of 40 x 40 px of the 1920x804 picture (a near tower is 2-3 wide)
CAP = 2.0                  # scene-linear luminance cap in a cell's statistic (display ~0.8): the fire's clipped core
                           # and passing sparks cannot dominate a cell
EPS = 0.02                 # scene-linear floor added before the log: a dark cell's noise is not a surge
ATTACK = (0.72, 0.42, 0.16)            # share of a coming rise reached 1, 2 and 3 frames before it
RELEASE = (0.87, 0.73, 0.60, 0.48, 0.37, 0.28, 0.20, 0.13, 0.08, 0.03)   # share of a fall still held 1..10 frames
                                       # after it: (1 - k / 11) ** 1.6
KEEP = 0.5                 # share of a one-frame spike's log-excess that stays on its own frame
MAX_LOG = math.log(3.0)    # no cell gain beyond x3 or below /3
SIGMA = 0.7                # spatial smoothing of the cell log-gains, in cells
EDGE = 10                  # a surge interval that does not begin or end on a camera change fades its field in or
                           # out over this many frames; the lane's last frame (2399) is always plain, so the doom
                           # lane's cut at 2400 meets the plain frame
BISECT = 32

WINDOW = dict(
    f0=1441, f1=2400, kind='edge_polish',
    # the camera changes inside the lane (edge.py/a3.py): 1760 behind a giant, 1840 THE EDGE, 1860 the gilding
    # insert (tower 7), 1920 back to the crater; 1440 and 2400 belong to the neighbouring lanes
    cameras=(1440, 1760, 1840, 1860, 1920, 2400),
    # (cut, n_out, n_in, split): split of the log-exposure gap closed by the incoming side (1: all 'in', 0: all 'out')
    # 1760 and 1840 cut to brighter views: the incoming shot blooms up from the outgoing's level. 1860 (bright edge ->
    # dark insert) and 1920 (dark insert -> bright crater, the downbeat): the gap is shared, so the edge's light pulls
    # back as the insert's throat enters bright, and the insert's throat swells into the downbeat as the crater lands
    ramps=((1760, 0, 6, 1.0), (1840, 0, 10, 1.0), (1860, 6, 6, 0.5), (1920, 6, 8, 0.5)),
    match=0.9,             # share of the exposure gap closed on the cut frame(s); the rest stays as the cut's step
    chroma=0.25,           # share of the colour-balance gap closed there (per-channel gain)
    # intervals whose beat surges are reshaped: A6's forge throats once the towers have risen, and A7's crater and
    # towers. Not A6's rise (1441-1485: moving towers, not light, change the cells, and the field measured noisier
    # there), not 1760-1839 (behind the giant: no beat pop, only camera motion), not the gilding insert (smooth swells)
    surges=((1486, 1760), (1840, 1860), (1920, 2400)),
    note='polishedge v2: exposure match at 1760/1840/1860/1920 (hard cuts kept); A7 surges reshaped by a '
         'current-frame gain field (3-frame attack, ~10-frame release); no pixels from other frames')


# --------------------------------------------------------------------------------------------- tone path
def _ff():
    """finish/filmfinish.py: the exact analytic inverse of lib/look.finish's tone path (imported when first used, so
    edl_v3 can import WINDOW without the finish on its path)."""
    if 'filmfinish' not in sys.modules:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(os.path.dirname(here), 'finish'))
    import filmfinish
    return filmfinish


def _grey(srgb):
    """OpenCV's grey (BT.601 weights on sRGB code values), 0..255, as the owner's scan measures a frame."""
    return float((srgb @ np.array([0.299, 0.587, 0.114], np.float32)).mean() * 255.0)


class Plate:
    """One frame's statistics, from its file decoded at the fixed statistics resolution (BGR uint8)."""

    def __init__(self, bgr):
        self.srgb = np.ascontiguousarray(bgr[..., ::-1], np.float32) / 255.0
        self._lin = self._fwd = self._cells = None

    @property
    def lin(self):                                        # approx. scene-linear Rec.709 (the renderer's light)
        if self._lin is None:
            self._lin = _ff().display_to_709(self.srgb).astype(np.float32)
        return self._lin

    @property
    def fwd(self):                                        # the tone path forward of lin (apply()'s subtrahend)
        if self._fwd is None:
            self._fwd = _ff().hill_forward(self.lin)
        return self._fwd

    @property
    def cells(self):                                      # GY x GX mean capped scene luminance
        if self._cells is None:
            y = np.minimum(self.lin @ np.array([0.2126, 0.7152, 0.0722], np.float32), CAP)
            self._cells = cv2.resize(y, (GX, GY), interpolation=cv2.INTER_AREA).astype(np.float64)
        return self._cells


# ------------------------------------------------------------------------------------------------ schedule
def ease(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def camera(f, spec):
    """[lo, hi): the camera interval holding f."""
    c = spec['cameras']
    return next((a, b) for a, b in zip(c, c[1:]) if a <= f < b)


def surge_interval(f, spec):
    return next(((a, b) for a, b in spec['surges'] if a <= f < b), None)


def ramps_at(f, spec):
    """[(cut, side, weight)] of the exposure ramps over f; weight 1 on the frame next to the cut."""
    out = []
    for cut, n_out, n_in, _ in spec['ramps']:
        if n_in and cut <= f < cut + n_in:
            out.append((cut, 'in', 1.0 - ease((f - cut) / n_in)))
        if n_out and cut - n_out <= f < cut:
            out.append((cut, 'out', ease((f - cut + n_out + 1) / n_out)))
    return out


def fill_frames(f, spec):
    """The frames whose cell statistics the surge field at f reads (inside f's camera interval)."""
    iv = surge_interval(f, spec)
    if iv is None:
        return ()
    lo, hi = iv
    return tuple(range(max(lo, f - len(RELEASE) - 1), min(hi, f + len(ATTACK) + 2)))


def source_frames(f, spec):
    """Every frame whose plate statistics decide frame f's gain (the delivery cache fingerprints them all)."""
    if not spec['f0'] <= f < spec['f1']:
        return ()
    got = set(fill_frames(f, spec))
    for cut, _, _ in ramps_at(f, spec):
        for g in (cut - 1, cut):
            got.add(g)
            got.update(fill_frames(g, spec))
    return tuple(sorted(got))


# ------------------------------------------------------------------------------------------------ surges
def surge_log_gain(f, spec, plate):
    """GY x GX log-gain of the surge reshaping at f (None outside SURGES). plate(g) -> Plate or None."""
    iv = surge_interval(f, spec)
    if iv is None:
        return None
    lo, hi = iv
    frames = fill_frames(f, spec)
    ell = {}
    for g in frames:
        p = plate(g)
        if p is None:
            return None
        ell[g] = np.log(p.cells + EPS)

    def soft(g):                                           # the cell log-light with one-frame spikes softened
        nb = [ell[h] for h in (g - 1, g + 1) if h in ell]
        if not nb:
            return ell[g]
        spike = np.maximum(ell[g] - np.maximum.reduce(nb), 0.0)
        return ell[g] - (1.0 - KEEP) * spike

    s = {g: soft(g) for g in frames if lo <= g < hi and (g - 1 in ell or g == lo) and (g + 1 in ell or g == hi - 1)}
    here = s[f]
    lift = np.zeros_like(here)
    for k, a in enumerate(ATTACK, 1):
        if f + k in s:
            lift = np.maximum(lift, a * (s[f + k] - here))
    for k, r in enumerate(RELEASE, 1):
        if f - k in s:
            lift = np.maximum(lift, r * (s[f - k] - here))
    return np.clip(here + lift - ell[f], -MAX_LOG, MAX_LOG) * taper(f, spec)


def taper(f, spec):
    """The surge field's weight: 0 outside SURGES; at an interval end that is not a camera change (and always at the
    lane's last frame) it eases from 0 on the end frame to 1 at EDGE frames in; at a camera change it starts at 1."""
    iv = surge_interval(f, spec)
    if iv is None:
        return 0.0
    lo, hi = iv
    w_in = 1.0 if lo in spec['cameras'] else ease((f - lo) / EDGE)
    w_out = 1.0 if (hi in spec['cameras'] and hi != spec['f1']) else ease((hi - 1 - f) / EDGE)
    return w_in * w_out


# ----------------------------------------------------------------------------------------------- exposure
def _field(log_cells, shape):
    """GY x GX log-gains -> a smooth H x W gain field."""
    lg = cv2.GaussianBlur(log_cells.astype(np.float32), (0, 0), SIGMA, borderType=cv2.BORDER_REPLICATE)
    return np.exp(cv2.resize(lg, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC))


def _graded(p, rgb, log_cells):
    """The plate's display (sRGB) after a per-channel gain rgb and a cell field, by the additive route of apply()."""
    ff = _ff()
    v = p.lin * np.asarray(rgb, np.float32)
    if log_cells is not None:
        v = v * _field(log_cells, v.shape[:2])[..., None]
    return p.srgb + (ff.hill_forward(v) - p.fwd)


def _balance(p, log_cells):
    """Display-linear channel means over luminance: the plate's colour balance after its own surge field."""
    d = _ff()._srgb_decode(np.clip(_graded(p, (1.0, 1.0, 1.0), log_cells), 0, 1)).reshape(-1, 3).mean(0)
    return d / max(float(d @ np.array([0.2126, 0.7152, 0.0722])), 1e-6)


def _solve(p, chroma, log_cells, target):
    """The luminance gain that brings the plate's grey (with chroma and its surge field) to target."""
    lo, hi = math.log(0.05), math.log(20.0)
    for _ in range(BISECT):
        mid = 0.5 * (lo + hi)
        if _grey(np.clip(_graded(p, math.exp(mid) * chroma, log_cells), 0, 1)) < target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def ramp_gain(cut, side, spec, plate):
    """The full per-channel gain on the frame next to the cut (weight 1), or None if a plate is missing."""
    entry = next(r for r in spec['ramps'] if r[0] == cut)
    split = entry[3] if side == 'in' else 1.0 - entry[3]
    me, other = (cut, cut - 1) if side == 'in' else (cut - 1, cut)
    pm, po = plate(me), plate(other)
    if pm is None or po is None:
        return None
    lm, lo_ = surge_log_gain(me, spec, plate), surge_log_gain(other, spec, plate)
    gm = _grey(np.clip(_graded(pm, (1.0, 1.0, 1.0), lm), 0, 1))
    go = _grey(np.clip(_graded(po, (1.0, 1.0, 1.0), lo_), 0, 1))
    target = math.exp(math.log(gm) + split * spec['match'] * (math.log(go) - math.log(gm)))
    chroma = (_balance(po, lo_) / _balance(pm, lm)) ** (split * spec['chroma'])
    chroma = chroma / float(chroma @ np.array([0.2126, 0.7152, 0.0722]))
    return _solve(pm, chroma, lm, target) * chroma


# ------------------------------------------------------------------------------------------------- the frame
def gain(f, spec, plate, memo=None):
    """(rgb gain (3,), GY x GX cell log-gain or None) for frame f, or None where the frame stays exactly as it is.

    plate(g) returns frame g's Plate (None if it has no plain file). memo, a dict, keeps the ramps' anchor gains."""
    if not spec['f0'] <= f < spec['f1']:
        return None
    ramps = ramps_at(f, spec)
    cells = surge_log_gain(f, spec, plate) if surge_interval(f, spec) else None
    if surge_interval(f, spec) and cells is None:
        return None                                        # a plate is missing: play the plain frame
    rgb = np.ones(3)
    for cut, side, w in ramps:
        key = (cut, side)
        full = memo.get(key) if memo is not None else None
        if full is None:
            full = ramp_gain(cut, side, spec, plate)
            if full is None:
                return None
            if memo is not None:
                memo[key] = full
        rgb = rgb * np.asarray(full, np.float64) ** w
    if cells is not None and not cells.any():
        cells = None                                       # a tapered end frame: no field at all
    if cells is None and not ramps:
        return None
    return rgb, cells


def apply(img, g):
    """The current plate (sRGB float, any size) with gain g from gain(); scene-linear, added as a difference."""
    ff = _ff()
    rgb, cells = g
    lin = ff.display_to_709(np.clip(img, 0, 1)).astype(np.float32)
    v = lin * np.asarray(rgb, np.float32)
    if cells is not None:
        v = v * _field(cells, img.shape[:2])[..., None]
    return (img + (ff.hill_forward(v) - ff.hill_forward(lin))).astype(np.float32)
