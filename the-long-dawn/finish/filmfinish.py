"""THE LONG DAWN · FINISH: the films' photographic finish (film negative -> print emulation, halation, grain).

Our frames on disk are DISPLAY-referred: every renderer ends in lib/look.finish() = bloom -> vignette -> the Hill
"ACES fitted" curve -> lift -> sRGB, saved as 8-bit PNG/JPG. The finish works on those frames as they are (no
re-render of approved shots):

    sRGB-Hill 8-bit --(exact analytic inverse of look.finish's tone path)--> approx. scene-linear Rec.709
      -> ACEScg -> ACEScct -> [L1] film log exposure  (+ HALATION: light spread back into the red layer)
      -> [L2] negative dye density                      (+ GRAIN: per-layer particle statistics, per-frame seed)
      -> [L3] print (2383/2393) + scan -> sRGB           -> restrained blend with the source (per look)

L1/L2/L3 are spektrafilm's baked 3-LUT bundle (finish/bake_luts.sh; CC BY-SA 4.0 LUTs, see edit/CREDITS.md).
This module needs only numpy + cv2 (numba if present), so it runs inside the edit's own venv; it never imports
spektrafilm. Exposure is FIXED per look (no auto-exposure, so nothing can pump or flicker). Grain is
deterministic per (cut, frame): the same frame always gets the same grain, and no two frames share it.

License: GPL-3.0-or-later (its halation and grain follow spektrafilm's published models; see edit/CREDITS.md).
"""
import os
import zlib

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LUTS = os.path.join(HERE, 'luts')

# ------------------------------------------------------------------------------------------------ inverse
# look.py (lib/look.py): the Hill "ACES fitted" curve used by every renderer, and the film-base lift.
_ACES_IN = np.array([[0.59719, 0.35458, 0.04823],
                     [0.07600, 0.90834, 0.01566],
                     [0.02840, 0.13383, 0.83777]], np.float64)
_ACES_OUT = np.array([[1.60475, -0.53108, -0.07367],
                      [-0.10208, 1.10813, -0.00605],
                      [-0.00327, -0.07276, 1.07602]], np.float64)
_A, _B, _C, _D, _E = 0.0245786, 0.000090537, 0.983729, 0.4329510, 0.238081
LIFT = 0.004
# linear Rec.709 (D65) -> ACEScg (AP1, D60, Bradford)
_709_TO_AP1 = np.array([[0.6130974, 0.3395231, 0.0473795],
                        [0.0701937, 0.9163539, 0.0134524],
                        [0.0206156, 0.1095698, 0.8698147]], np.float64)
_AP1_TO_709 = np.linalg.inv(_709_TO_AP1)
_OUT_INV = np.linalg.inv(_ACES_OUT)
# one matrix from the curve's own space straight to ACEScg
_CURVE_TO_AP1 = (_709_TO_AP1 @ np.linalg.inv(_ACES_IN)).astype(np.float32)
_CURVE_TO_709 = np.linalg.inv(_ACES_IN).astype(np.float32)
V_MAX = 32.0          # the curve reaches display 1.0 at ~25.7; a clipped code is "at least this bright"


def rrt_fit(v):
    return (v * (v + _A) - _B) / (v * (_C * v + _D) + _E)


def rrt_fit_inv(t):
    """Inverse of the per-channel Hill curve; linear below its black point, capped at V_MAX near the top."""
    t = np.asarray(t, np.float32)
    tc = np.clip(t, 0.0, 1.0 / _C - 1e-4)
    q = 1.0 - tc * _C
    p = _A - tc * _D
    r = _B + tc * _E
    v = (-p + np.sqrt(p * p + 4.0 * q * r)) / (2.0 * q)
    v0 = (-_A + np.sqrt(_A * _A + 4 * _B)) / 2.0                  # where the curve crosses 0
    slope0 = (2 * v0 + _A) / (_C * v0 * v0 + _D * v0 + _E)            # f'(v0), numerator is 0 there
    v = np.where(t < 0, v0 + t / slope0, v)
    return np.minimum(v, V_MAX).astype(np.float32)


def _srgb_decode(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _srgb_encode(x):
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def display_to_curve(srgb01):
    """sRGB display [0,1] -> the Hill curve's output t (per channel, pre output-matrix), float32."""
    y = _srgb_decode(srgb01)
    x = (y - LIFT) / (1.0 - LIFT)
    return x @ _OUT_INV.T.astype(np.float32)


def display_to_ap1(srgb01):
    """Approximate scene-linear ACEScg of a look.finish() frame (exact where no channel was clipped)."""
    v = rrt_fit_inv(display_to_curve(srgb01))
    return v @ _CURVE_TO_AP1.T


def display_to_709(srgb01):
    v = rrt_fit_inv(display_to_curve(srgb01))
    return v @ _CURVE_TO_709.T


def hill_forward(lin709):
    """look.tonemap + lift + sRGB (no bloom/vignette): used to measure the round trip."""
    x = np.asarray(lin709, np.float32) @ _ACES_IN.T.astype(np.float32)
    x = rrt_fit(x) @ _ACES_OUT.T.astype(np.float32)
    x = np.clip(x, 0, 1)
    return _srgb_encode(x + LIFT * (1 - x))


def acescct_encode(lin):
    lin = np.asarray(lin, np.float32)
    lo = 10.5402377416545 * lin + 0.0729055341958355
    hi = (np.log2(np.maximum(lin, 1e-10)) + 9.72) / 17.52
    return np.where(lin <= 0.0078125, lo, hi).astype(np.float32)


# ------------------------------------------------------------------------------------------------ 3D LUTs
def read_cube(path):
    """.cube -> float32 array [r, g, b, 3] (cached next to the file as .npy)."""
    npy = path + '.npy'
    if os.path.exists(npy) and os.path.getmtime(npy) >= os.path.getmtime(path):
        return np.load(npy)
    n, rows = None, []
    with open(path) as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s[0] == '#' or s[0].isalpha():
                if s.startswith('LUT_3D_SIZE'):
                    n = int(s.split()[1])
                continue
            rows.append(s)
    data = np.array([r.split() for r in rows], np.float32)
    lut = data.reshape(n, n, n, 3).transpose(2, 1, 0, 3).copy()     # file order: r fastest -> [b, g, r]
    np.save(npy, lut)
    return lut


try:                                                                # a numba kernel when available (10x faster)
    import numba as _nb

    @_nb.njit(cache=True, fastmath=True)
    def _tetra_nb(img, lut, out):
        n = lut.shape[0] - 1
        h, w = img.shape[0], img.shape[1]
        for yy in range(h):
            for xx in range(w):
                r = min(max(img[yy, xx, 0], 0.0), 1.0) * n
                g = min(max(img[yy, xx, 1], 0.0), 1.0) * n
                b = min(max(img[yy, xx, 2], 0.0), 1.0) * n
                i = min(int(r), n - 1)
                j = min(int(g), n - 1)
                k = min(int(b), n - 1)
                fr, fg, fb = r - i, g - j, b - k
                for c in range(3):
                    c000 = lut[i, j, k, c]
                    c111 = lut[i + 1, j + 1, k + 1, c]
                    if fr > fg:
                        if fg > fb:
                            v = (1 - fr) * c000 + (fr - fg) * lut[i + 1, j, k, c] + (fg - fb) * lut[i + 1, j + 1, k, c] + fb * c111
                        elif fr > fb:
                            v = (1 - fr) * c000 + (fr - fb) * lut[i + 1, j, k, c] + (fb - fg) * lut[i + 1, j, k + 1, c] + fg * c111
                        else:
                            v = (1 - fb) * c000 + (fb - fr) * lut[i, j, k + 1, c] + (fr - fg) * lut[i + 1, j, k + 1, c] + fg * c111
                    else:
                        if fb > fg:
                            v = (1 - fb) * c000 + (fb - fg) * lut[i, j, k + 1, c] + (fg - fr) * lut[i, j + 1, k + 1, c] + fr * c111
                        elif fb > fr:
                            v = (1 - fg) * c000 + (fg - fb) * lut[i, j + 1, k, c] + (fb - fr) * lut[i, j + 1, k + 1, c] + fr * c111
                        else:
                            v = (1 - fg) * c000 + (fg - fr) * lut[i, j + 1, k, c] + (fr - fb) * lut[i + 1, j + 1, k, c] + fb * c111
                    out[yy, xx, c] = v
        return out

    def apply_lut(img, lut):
        img = np.ascontiguousarray(img, np.float32)
        return _tetra_nb(img, lut, np.empty_like(img))
except Exception:                                                   # numpy trilinear fallback
    def apply_lut(img, lut):
        n = lut.shape[0] - 1
        x = np.clip(np.asarray(img, np.float32), 0, 1) * n
        i = np.minimum(x.astype(np.int32), n - 1)
        f = x - i
        out = np.zeros_like(x)
        for dr in (0, 1):
            wr = f[..., 0] if dr else 1 - f[..., 0]
            for dg in (0, 1):
                wg = f[..., 1] if dg else 1 - f[..., 1]
                for db in (0, 1):
                    wb = f[..., 2] if db else 1 - f[..., 2]
                    out += (wr * wg * wb)[..., None] * lut[i[..., 0] + dr, i[..., 1] + dg, i[..., 2] + db]
        return out


class Bundle:
    """A spektrafilm 3-LUT bundle (L1 ACEScct -> log10 E film, L2 -> CMY film density, L3 -> print+scan sRGB)."""

    def __init__(self, film='kodak_vision3_500t', print_='kodak_2383', root=LUTS):
        import json
        short = film.replace('kodak_', '').replace('_', '')
        d = next((os.path.join(root, x) for x in sorted(os.listdir(root))
                  if x.startswith(f'spektrafilm_v033_{short}_') and os.path.isdir(os.path.join(root, x))), None)
        if d is None:
            raise FileNotFoundError(f'no baked bundle for {film} in {root}: run finish/bake_luts.sh')
        meta = json.load(open(os.path.join(d, 'bundle.json')))
        w = meta['wires']
        self.le_min, self.le_max = w['log_e_film']['min'], w['log_e_film']['max']
        self.d_min = np.array(w['cmy_film']['d_min'], np.float32)
        self.d_max = np.array(w['cmy_film']['d_max'], np.float32)
        p = {(l['role'], l['print_profile']): l['path'] for l in meta['luts']}
        self.l1 = read_cube(os.path.join(d, p[('filming_expose', None)]))
        self.l2 = read_cube(os.path.join(d, p[('filming_develop', None)]))
        self.l3 = read_cube(os.path.join(d, p[('printing_combined', print_)]))
        self.film, self.print_ = film, print_

    # wires
    def le_encode(self, le):
        return (le - self.le_min) / (self.le_max - self.le_min)

    def le_decode(self, code):
        return code * (self.le_max - self.le_min) + self.le_min

    def d_encode(self, dens):
        return np.clip((dens - self.d_min) / (self.d_max - self.d_min), 0, 1)

    def d_decode(self, code):
        return code * (self.d_max - self.d_min) + self.d_min


# ------------------------------------------------------------------------------------------------ effects
PIX_UM = 12.97        # a 1920-px-wide 2.39:1 extraction from a Super 35 negative (24.9 mm) -> ~13 um per pixel


def halation(E, strength=(0.05, 0.015, 0.0), sigma_um=65.0, bounces=3, decay=0.5, amount=1.0, scale=1.0):
    """Light that passes the emulsion, reflects off the base and re-exposes it, mostly the red layer (spektrafilm's
    halation model, energy-conserving). E: linear film exposure per layer (H, W, 3)."""
    out = E.copy()
    sig = sigma_um / PIX_UM * scale
    wts = np.array([decay ** k for k in range(bounces)], np.float32)
    wts /= wts.sum()
    for c in range(3):
        s = strength[c] * amount
        if s <= 0:
            continue
        acc = np.zeros(E.shape[:2], np.float32)
        for k in range(bounces):
            acc += wts[k] * cv2.GaussianBlur(E[..., c], (0, 0), sig * (k + 1))
        out[..., c] = (1 - s) * E[..., c] + s * acc
    return out


def frame_seed(cut, frame, salt=0):
    return zlib.crc32(f'{cut}:{int(frame)}:{salt}'.encode()) & 0x7FFFFFFF


def grain(D, dmax, seed, amount=1.0, particle_um2=0.2, scale=(1.6, 1.6, 3.2), uniformity=(0.97, 0.99, 0.97),
          blur_px=0.65, rho=0.7):
    """Film grain in the negative's dye-density domain, from the statistics of spektrafilm's particle model
    (binomial development of a Poisson number of silver-halide crystals per pixel), in the Gaussian limit (hundreds of
    crystals per pixel at 13 um): var(D) = D * Dmax * (1 - p*u) / n, p = D / Dmax. The fast (large-grain) sub-layer
    carries the thin parts of the negative, so the crystal count falls toward low density. The dye-cloud blur is
    physical and lowers the variance (as in spektrafilm). rho: the share of one grain field common to the three
    layers (the layers are stacked and scanned through each other; rho < 1 keeps a little colour in the grain).
    Deterministic: numpy PCG64 seeded per (cut, frame)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    h, w = D.shape[:2]
    shared = rng.standard_normal((h, w), dtype=np.float32)
    own = rng.standard_normal((h, w, 3), dtype=np.float32)
    k = np.float32(np.sqrt(max(1.0 - rho * rho, 0.0)))
    area = PIX_UM * PIX_UM
    out = D.copy()
    for c in range(3):
        n = area / (particle_um2 * scale[c])
        d = np.maximum(D[..., c] + 0.03, 0.0)                               # + base fog density
        p = np.clip(d / dmax[c], 0.0, 1.0)
        n_eff = n * (0.5 + 0.5 * np.clip(p * 2.0, 0, 1))                      # thin negative: bigger grains
        sd = np.sqrt(np.maximum(d * dmax[c] * (1.0 - p * uniformity[c]), 0.0) / n_eff)
        g = rho * shared + k * own[..., c]
        if blur_px > 0.4:
            g = cv2.GaussianBlur(g, (0, 0), blur_px)
        out[..., c] = D[..., c] + amount * sd * g
    return out


# ------------------------------------------------------------------------------------------------ the look
class Look:
    """One finishing look. mix: how much of the film's tone and colour replaces the source's (in linear light);
    grain / halation: 1.0 = spektrafilm's physical defaults at 13 um per pixel. ev: a FIXED exposure offset (never
    auto). fire_guard: warm pixels may not drift toward yellow relative to the source (hue held, film tone kept)."""

    def __init__(self, name, film='kodak_vision3_500t', print_='kodak_2383', ev=0.0, mix=0.7, grain=1.0,
                 halation=1.0, fire_guard=True, fire_restore=0.0, grain_only=False, match_mid=True, balance=True,
                 black_lift=LIFT, white=1.0, grain_rho=0.85):
        self.name, self.film, self.print_ = name, film, print_
        self.ev, self.mix, self.grain_amt, self.hal_amt = ev, mix, grain, halation
        self.fire_guard, self.fire_restore, self.grain_only = fire_guard, fire_restore, grain_only
        self.black_lift, self.white, self.grain_rho = black_lift, white, grain_rho
        self.bundle = Bundle(film, print_)
        self.curves = None
        self.gain = 2.0 ** ev
        if match_mid:                                   # one fixed calibration: scene 0.18 prints as the source did
            self.gain *= self._mid_gain()
        if balance:
            self.curves = self._balance_curves()

    def _balance_curves(self):
        """Print grey balance: per-channel 1-D curves (display space) that make the film's neutral scale neutral
        (no crossover cast: the 2383 base prints warm, its toe cool) and let the print's white reach `white` with a
        smooth stretch above mid-grey (mid-grey itself is unchanged)."""
        x = np.exp(np.linspace(np.log(1e-4), np.log(64.0), 1024)).astype(np.float32)
        ramp = np.repeat(x[:, None, None], 3, axis=2).reshape(-1, 1, 3) * self.gain
        out = self._film(ramp, None, 0)[:, 0, :].astype(np.float64)        # (N, 3) display, per channel
        out = np.maximum.accumulate(out, axis=0)                             # guard against tiny non-monotony
        g = out[:, 1]
        t = np.clip((np.log2(x) - np.log2(0.18)) / (np.log2(64.0) - np.log2(0.18)), 0, 1)
        s = t * t * (3 - 2 * t)
        n = g + (self.white - g.max()) * s                                   # neutral target, top -> white
        n = np.maximum.accumulate(n)
        curves = []
        for c in range(3):
            xc, idx = np.unique(out[:, c], return_index=True)
            curves.append((xc.astype(np.float32), n[idx].astype(np.float32)))
        return curves

    def _apply_curves(self, img):
        out = np.empty_like(img)
        for c in range(3):
            xc, yc = self.curves[c]
            out[..., c] = np.interp(img[..., c], xc, yc).astype(np.float32)
        return out

    def _mid_gain(self):
        target = hill_forward(np.full((1, 1, 3), 0.18, np.float32))[0, 0, 1]
        lo, hi = -3.0, 3.0
        for _ in range(40):                                  # bisection on EV for the neutral mid-grey patch
            mid = 0.5 * (lo + hi)
            ap1 = np.full((1, 1, 3), 0.18 * 2 ** mid, np.float32)
            out = self._film(ap1, None, 0)[0, 0, 1]
            if out > target:
                hi = mid
            else:
                lo = mid
        return 2.0 ** (0.5 * (lo + hi))

    def _film(self, ap1, seed, hal_amt, grain_amt=0.0):
        b = self.bundle
        le = b.le_decode(apply_lut(acescct_encode(ap1), b.l1))            # log10 film exposure per layer
        if hal_amt > 0:
            E = np.power(10.0, le).astype(np.float32)
            le = np.log10(np.maximum(halation(E, amount=hal_amt), 1e-10)).astype(np.float32)
        dens = b.d_decode(apply_lut(b.le_encode(le), b.l2))              # negative dye density (CMY)
        if grain_amt > 0 and seed is not None:
            dens = grain(dens, b.d_max, seed, amount=grain_amt, rho=self.grain_rho)
        return apply_lut(b.d_encode(dens), b.l3)                          # print + scan, sRGB display

    def __call__(self, srgb01, cut='A', frame=0):
        src = np.clip(np.asarray(srgb01, np.float32), 0, 1)
        seed = frame_seed(cut, frame)
        if self.grain_only:
            return self._grain_only(src, seed)
        lin709 = display_to_709(src)
        ap1 = (lin709 @ _709_TO_AP1.T.astype(np.float32)) * self.gain
        film = self._film(ap1, seed, self.hal_amt, self.grain_amt)
        if self.curves is not None:
            film = self._apply_curves(film)
        f = _srgb_decode(film)
        if self.black_lift > 0:                          # the source's film-base black (look.finish lift)
            f = self.black_lift + (1.0 - self.black_lift) * f
        a = _srgb_decode(src)
        out = _srgb_encode(a + (min(self.mix, 1.0) * floor_weight(a))[..., None] * (f - a))
        if self.fire_guard or self.fire_restore > 0:     # on the final colour: the rule holds for what ships
            out = fire_hue(src, out, lin709, self.fire_restore if self.fire_restore > 0 else 0.0)
        return np.clip(out, 0, 1)

    def _grain_only(self, src, seed):
        """Ink and parchment: the page keeps its own tone; only the emulsion's grain, printed (density-domain noise
        mapped through a print-like contrast, strongest in the mid-tones)."""
        rng = np.random.Generator(np.random.PCG64(seed))
        h, w = src.shape[:2]
        g = cv2.GaussianBlur(rng.standard_normal((h, w), dtype=np.float32), (0, 0), 0.65) * 2.3
        gc = cv2.GaussianBlur(rng.standard_normal((h, w, 3), dtype=np.float32), (0, 0), 0.65) * 2.3
        lin = _srgb_decode(src)
        lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        amp = 0.045 * self.grain_amt * np.sqrt(np.clip(lum, 0, 1)) * (1.15 - np.clip(lum, 0, 1))
        noise = 0.8 * g[..., None] + 0.35 * gc
        return _srgb_encode(lin * np.exp(amp[..., None] * noise))


def floor_weight(lin):
    """1 at and above the renders' own film-base black (look.finish's lift, linear 0.004), fading to 0 at a fifth of
    it: whatever an edit grade, a fade or a slate took below the floor stays exactly as it was (never lifted)."""
    y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    t = np.clip((y - 0.2 * LIFT) / (0.8 * LIFT), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _hue_sat(rgb):
    hsv = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    return hsv


def fire_hue(src, film, lin709, restore=0.0, max_shift_deg=2.0, red_floor=22.0):
    """Keep fire ORANGE-GOLD. Where the source is warm (hue 0-60 deg, saturated, not dark):
    - restore > 0 turns the film's hue back toward the scene's own emission hue (the linear ratio before the Hill
      curve pushed bright orange toward yellow), by that fraction, never redder than red_floor degrees;
    - and the result may never sit more than max_shift_deg toward yellow of the source frame's hue.
    Value and saturation stay the film's."""
    hs = _hue_sat(src)
    hf = _hue_sat(film)
    warm = ((hs[..., 0] < 60) | (hs[..., 0] > 345)) & (hs[..., 1] > 0.2) & (hs[..., 2] > 0.08)
    if not warm.any():
        return film
    wrap = lambda h: np.where(h > 180, h - 360, h)                     # noqa: E731
    h_src = wrap(hs[..., 0])
    h_film = wrap(hf[..., 0])
    h = h_film
    if restore > 0:
        sc = lin709 / np.maximum(lin709.max(axis=2, keepdims=True), 1e-6)
        h_scene = wrap(_hue_sat(sc)[..., 0])
        target = np.maximum(h_scene, np.minimum(red_floor, h_film))
        h = np.where(h_film > target, h_film + restore * (target - h_film), h_film)
    h = np.minimum(h, h_src + max_shift_deg)
    w = (np.clip((hs[..., 1] - 0.2) / 0.2, 0, 1) * np.clip((hs[..., 2] - 0.08) / 0.1, 0, 1) * warm)[..., None]
    if not np.any(np.abs(h - h_film) > 0.05):
        return film
    hf2 = hf.copy()
    hf2[..., 0] = np.where(h < 0, h + 360, h)
    fixed = cv2.cvtColor(hf2, cv2.COLOR_HSV2RGB)
    return film * (1 - w) + fixed * w


# ------------------------------------------------------------------------------------------------ presets
LOOKS = {
    # stock -> print, restrained
    '500T_2383': dict(film='kodak_vision3_500t', print_='kodak_2383', mix=0.75, grain=0.6, halation=1.0),
    '500T_2383_fire': dict(film='kodak_vision3_500t', print_='kodak_2383', mix=0.75, grain=0.6, halation=1.0,
                           fire_restore=0.4),
    '250D_2383_fire': dict(film='kodak_vision3_250d', print_='kodak_2383', mix=0.75, grain=0.5, halation=1.0,
                           fire_restore=0.4),
    '500T_2393_fire': dict(film='kodak_vision3_500t', print_='kodak_2393', mix=0.75, grain=0.6, halation=1.0,
                           fire_restore=0.4),
    '50D_2383': dict(film='kodak_vision3_50d', print_='kodak_2383', mix=0.75, grain=0.4, halation=1.0),
    'ink_grain': dict(grain_only=True, grain=1.0),
}


def make(name, **over):
    kw = dict(LOOKS[name])
    kw.update(over)
    return Look(name, **kw)
