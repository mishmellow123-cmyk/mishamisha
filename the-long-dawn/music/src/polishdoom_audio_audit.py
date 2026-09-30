"""Measure the doom candidate against the delivered AP2, with bounded reads (run through onepy).

    python polishdoom_audio_audit.py --before <delivered sound_AP2.wav> --after <new sound_AP2.wav>
        --before-sfxpre <sfxpre_AP2_before_polish.wav> --after-sfxpre <new sfxpre_AP2.wav>
        --scope <cache/v3/master_scope_sound_AP2.json> --json <out.json>

Outside [2400, 3360) the guarded effects premaster must equal the pinned reference sample for sample, and the master
must equal the delivered master times the single whole-film scalar the scoped master reports, to within the 24-bit
dither (TPDF, +-2 LSB after rescaling). Inside, it reports the moments this lane changed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import sound_polishdoom as P

LSB = 2.0 ** -23


def rms_db(y):
    return float(20 * np.log10(max(float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))), 1e-12)))


def read_frames(path, first, end):
    return sf.read(path, start=round(first * P.FR), stop=round(end * P.FR), dtype="float64", always_2d=True)[0]


def outside_identity(before, after, start=P.START, end=P.END):
    a, b = sf.info(before), sf.info(after)
    if (a.frames, a.samplerate, a.channels) != (b.frames, b.samplerate, b.channels):
        return dict(ok=False, error="WAV formats differ")
    count, peak = 0, 0.0
    for low, high in ((0, start * P.FR), (end * P.FR, a.frames)):
        for at in range(low, high, P.SR * 5):
            stop = min(high, at + P.SR * 5)
            x = sf.read(before, start=at, stop=stop, dtype="float32", always_2d=True)[0]
            y = sf.read(after, start=at, stop=stop, dtype="float32", always_2d=True)[0]
            d = np.abs(x - y)
            count += int(np.any(d != 0, axis=1).sum())
            peak = max(peak, float(d.max(initial=0)))
    return dict(ok=count == 0, different_sample_frames=count, max_abs_difference=peak,
                samples_compared=int(start * P.FR + (a.frames - end * P.FR)))


def outside_scaled(before, after, scalar, start=P.START, end=P.END, tolerance_lsb=2.0):
    """Outside the lane: after == scalar x before to within the requantisation and dither of a 24-bit write."""
    a = sf.info(before)
    worst, over, compared = 0.0, 0, 0
    for low, high in ((0, start * P.FR), (end * P.FR, a.frames)):
        for at in range(low, high, P.SR * 5):
            stop = min(high, at + P.SR * 5)
            x = sf.read(before, start=at, stop=stop, dtype="float64", always_2d=True)[0]
            y = sf.read(after, start=at, stop=stop, dtype="float64", always_2d=True)[0]
            d = np.abs(y - scalar * x) / LSB
            worst = max(worst, float(d.max(initial=0)))
            over += int((d > tolerance_lsb).sum())
            compared += d.size
    return dict(ok=over == 0, scalar=scalar, scalar_db=20 * np.log10(scalar), max_deviation_lsb=worst,
                samples_over_tolerance=over, tolerance_lsb=tolerance_lsb, samples_compared=compared)


def five_ms(y):
    n = 240
    y = y[:len(y) // n * n].reshape(-1, n, 2)
    return np.sqrt(np.mean(np.square(y, dtype=np.float64), axis=(1, 2)))


def onset(path, lo, hi, under_db=20.0):
    """First 1 ms block within under_db of the window's loudest 1 ms block, in frames."""
    x = read_frames(path, lo, hi)
    m = np.abs(x).max(1)
    k = len(m) // 48
    e = 20 * np.log10(np.maximum(m[:k * 48].reshape(k, 48).max(1), 1e-12))
    i = int(np.flatnonzero(e >= e.max() - under_db)[0])
    return round(lo + i * 48 / P.FR, 3), round(float(e.max()), 2)


def jump_db(path, frame, window=0.05):
    """50 ms RMS level after the frame minus the 50 ms before it."""
    n = round(window * P.SR)
    at = round(frame * P.FR)
    x = sf.read(path, start=at - n, stop=at + n, dtype="float64", always_2d=True)[0]
    return round(rms_db(x[n:]) - rms_db(x[:n]), 2)


def moment_metrics(path):
    points = [2400, 2460, 2520, 2600, 2620, 2636, 2637, 2638, 2639, 2640, 2641, 2660, 2700, 2762, 2780,
              2796, 2800, 2820, 2832, 2836, 2848, 2859, 2860, 2861, 2862, 2880, 2920, 3120, 3359]
    levels = {str(f): rms_db(read_frames(path, f, f + 1)) for f in points}
    black = five_ms(read_frames(path, 2797, 2848))
    breath = five_ms(read_frames(path, 2637, 2640))
    return dict(rms_one_frame_dbfs=levels,
                black_2797_2848_min_5ms_dbfs=float(20 * np.log10(max(black.min(), 1e-12))),
                black_2797_2848_max_5ms_dbfs=float(20 * np.log10(max(black.max(), 1e-12))),
                black_zero_5ms_blocks=int((black == 0).sum()),
                breath_2637_2640_min_5ms_dbfs=float(20 * np.log10(max(breath.min(), 1e-12))),
                breath_2637_2640_max_5ms_dbfs=float(20 * np.log10(max(breath.max(), 1e-12))),
                jump_50ms_db={str(f): jump_db(path, f) for f in (2637, 2640, 2800, 2832, 2848, 2860)},
                onset_impact=onset(path, 2632, 2646), onset_ember=onset(path, 2826, 2840),
                onset_piano=onset(path, 2856, 2866))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-sfxpre", type=Path, required=True)
    parser.add_argument("--after-sfxpre", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    scalar = json.loads(args.scope.read_text())["whole_film_scalar"]
    result = dict(before=moment_metrics(args.before), after=moment_metrics(args.after),
                  sfx_premaster_outside_identity=outside_identity(args.before_sfxpre, args.after_sfxpre),
                  master_outside_scaled=outside_scaled(args.before, args.after, scalar))
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    ok = result["sfx_premaster_outside_identity"]["ok"] and result["master_outside_scaled"]["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
