"""AP2's A6/A7 polish, confined to [1441, 2400) frames.

The stored score has already had its 200 ms breath gated and its hall tail cut.
Recover that short passage from the SAME cached performances, then conduct the
breath across the held chord and its release. No score note or sample changes.
Everything outside the replacement and fader windows remains sample-identical.
"""
import json
import os

import numpy as np

from timeline_v3 import FR, SR

SCOPE = (1441, 2400)
BREATH_WINDOW = (1822, 1854)
# Frame positions and dB are composition choices, not measured source levels.
BREATH_DB = ((1822, 0.0), (1835, -7.0), (1837, -7.0), (1840, 0.0), (1854, 0.0))
RIDE_DB = ((1840, 0.0), (1854, -2.0), (2000, -1.5), (2160, -0.9), (2320, -0.2), (2360, 0.0))
RECONSTRUCTION_EPS = 1e-5


def check_reconstruction(control, saved, eps=RECONSTRUCTION_EPS):
    """A cache/seating/IR mismatch must stop the render, not replace the score."""
    if control.shape != saved.shape or not np.all(np.isfinite(control)):
        raise ValueError("Edge control remix has an invalid shape or nonfinite samples")
    error = float(np.max(np.abs(control - saved)))
    if error > eps:
        raise ValueError(f"Edge control remix differs from saved score by {error:.8g} (limit {eps})")
    return error


def smooth_curve(frames, points, outside=0.0):
    """Cosine interpolation, including zero slope at each authored shoulder."""
    frames = np.asarray(frames, dtype=np.float64)
    out = np.full(frames.shape, outside, dtype=np.float64)
    for (f0, v0), (f1, v1) in zip(points, points[1:]):
        use = (frames >= f0) & (frames <= f1)
        u = (frames[use] - f0) / (f1 - f0)
        out[use] = v0 + (v1 - v0) * (0.5 - 0.5 * np.cos(np.pi * u))
    return out


def pulse_gain(frames):
    """A shallow fire-bed inhalation: three-frame attack, twelve-frame release.

    The audible score attacks remain on their original 20-frame beat grid. The
    fire bed recedes between them, less as the crater fills the shot; the bed's
    original level is the peak, so the accents do not add clipping pressure.
    """
    frames = np.asarray(frames, dtype=np.float64)
    phase = np.mod(frames - 1840.0, 20.0)
    pulse = np.zeros_like(frames)
    after = phase <= 12.0
    pulse[after] = 0.5 + 0.5 * np.cos(np.pi * phase[after] / 12.0)
    before = phase >= 17.0
    pulse[before] = 0.5 - 0.5 * np.cos(np.pi * (phase[before] - 17.0) / 3.0)
    depth = smooth_curve(frames, ((1840, 0.0), (1860, 2.0), (2160, 1.4), (2360, 0.7), (2390, 0.0)))
    return np.power(10.0, -depth * (1.0 - pulse) / 20.0).astype(np.float32)


def cached_score_window(bm, start_f, end_f, keep_breath=False):
    """Remix a bounded interval from the current score's existing dry caches.

    The original mix's seating, EQ, sends, hall and percussion limiter apply.
    Warmup covers the whole hall IR plus two seconds of filter/limiter history.
    Performances come from the saved premaster's manifest. Raw sampler assets are
    unnecessary: setup() supplies the same mixing seats without rebuilding notes.
    keep_breath is the comparison condition used by the validation probe, which
    checks the reconstruction against the saved premaster before adoption.
    """
    import kit_v3 as K
    import score_v3_A as A
    import score_v3_AP2 as AP2
    import mix as MX
    import render_v3 as RV
    from scipy import signal

    S = K.Score(bm.cut, bm)
    A.setup(S)
    S.groups = dict(A.GROUPS)
    for source, name in AP2.GO.items():
        S.add(name, source)
    manifest_path = os.path.join(RV.CACHE, "manifest_final_" + bm.cut + ".json")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    irs = RV.hall_ir()
    a = max(0, int(start_f * FR) - max(map(len, irs)) - 2 * SR)
    b = int(end_f * FR)
    n = b - a
    dry, send = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
    gdry = {g: np.zeros_like(dry) for g in S.groups}
    gsend = {g: np.zeros_like(send) for g in S.groups}
    wins = [(i0 - a, i1 - a, ex) for i0, i1, ex in RV.breath_windows(S.breaths)
            if a <= i0 < b and (keep_breath or i1 != 1840 * FR)]
    for name, key in manifest.items():
        p = S.parts[name]
        offset, src = RV.load_stem(name, key)
        lo, hi = max(a, offset), min(b, offset + len(src))
        if hi <= lo:
            continue
        y = np.array(src[lo - offset:hi - offset], dtype=np.float32)
        off = lo - a
        if wins:
            y *= RV.breath_env(len(y), wins, name=name, offset=off)[:, None]
        y = MX.pan_width(y * np.float32(10 ** (p.gain_db / 20)), p.pan, p.width)
        y = MX.depth_eq(y, p.depth)
        spec = S.eq.get(name, (None, None))
        if spec[0]:
            y = MX.highpass(y, spec[0])
        if spec[1]:
            bb, aa = signal.butter(2, spec[1] / (SR / 2))
            y = signal.lfilter(bb, aa, y, axis=0).astype(np.float32)
        if len(spec) > 2 and spec[2]:
            y += MX.highpass(y, spec[3] if len(spec) > 3 else 6000.0, order=2) * np.float32(
                10 ** (spec[2] / 20) - 1)
        group = next((g for g in S.groups if name.startswith(g)), None)
        (gdry[group] if group else dry)[off:off + len(y)] += y * (1.0 - 0.35 * p.depth)
        (gsend[group] if group else send)[off:off + len(y)] += y * p.send
    wet = RV.convolve_with_breaths(send, irs, wins, n)
    for g, cfg in S.groups.items():
        bus = gdry[g] + RV.convolve_with_breaths(gsend[g], irs, wins, n)
        bus *= RV.limiter_gain(bus, cfg.get("ceiling_db", -6.0),
                               release=cfg.get("release", 0.08))[:, None]
        dry += bus
    mixed = MX.highpass(dry + wet, 22.0)
    return mixed[int(start_f * FR) - a:]


def apply(score, effects, bm, replacement=None):
    """In-place AP2 premaster edit; other cuts return their original arrays.

    replacement exists for bounded tests; production always recovers the score
    from checked cached performances. The stored premaster is never overwritten.
    """
    if bm.cut != "AP2":
        return score, effects
    if bm.event("edge")["frame"] != 1840 or bm.event("vortex")["frame"] != 2400:
        raise ValueError("Edge polish needs the authored 1840/2400 frame anchors")
    start, end = BREATH_WINDOW
    a, b = start * FR, end * FR
    if replacement is None:
        control = cached_score_window(bm, start, end, keep_breath=True)
        error = check_reconstruction(control, score[a:b])
        print(f"  edge control remix: max sample error {error:.3g}", flush=True)
        uncut = cached_score_window(bm, start, end)
    else:
        uncut = replacement
    if uncut.shape != score[a:b].shape:
        raise ValueError("Edge replacement has the wrong sample extent")
    frame = np.arange(a, b, dtype=np.float64) / FR
    weight = smooth_curve(frame, ((start, 0.0), (1826, 1.0), (1846, 1.0), (end, 0.0)))
    recovered = uncut * np.power(10.0, smooth_curve(frame, BREATH_DB) / 20.0).astype(np.float32)[:, None]
    score[a:b] += (recovered - score[a:b]) * weight.astype(np.float32)[:, None]
    a, b = 1840 * FR, 2390 * FR
    frame = np.arange(a, b, dtype=np.float64) / FR
    score[a:b] *= np.power(10.0, smooth_curve(frame, RIDE_DB) / 20.0).astype(np.float32)[:, None]
    effects[a:b] *= pulse_gain(frame)[:, None]
    return score, effects
