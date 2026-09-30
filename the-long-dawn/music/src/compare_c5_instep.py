"""Compare delivered C5P2 audio against private baseline clones, without rendering.

Run through the owner-night onepy wrapper: this streams the full audio duration.
python music/src/compare_c5_instep.py --out-dir /path/to/instepsound

The edit interval is [3800,4280). Every other video frame is compared. A saved
master envelope does not include the final true-peak scalar or head/tail fades;
the reconstruction accounts for both rather than treating its ratio as complete.
"""
import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf

from measure_c5_instep import clean, correlation_rows, sha256

MUSIC = Path(__file__).resolve().parents[1]
SR, FPS, FRAMES = 48000, 24, 5920
SPF = SR // FPS
N = FRAMES * SPF
BLOCK_FRAMES = 120
LSB = 2.0 ** -23
EDIT = (3800, 4280)
WAVS = {"master": "sound_C5P2.wav", "score": "sound_C5P2_score.wav",
        "effects": "sfx_C5P2.wav", "effects_pre": "sfxpre_C5P2.wav"}


def blocks():
    for f in range(0, FRAMES, BLOCK_FRAMES):
        yield f, min(BLOCK_FRAMES, FRAMES - f)


def outside(f, count):
    frames = np.arange(f, f + count)
    return (frames < EDIT[0]) | (frames >= EDIT[1])


def stats():
    return dict(sample_channel_values=0, different_sample_channel_values=0,
                sum_squares=0.0, max_abs=0.0, above_one_pcm24_lsb=0,
                above_four_pcm24_lsb=0)


def add(result, values):
    x = np.asarray(values, np.float64)
    result["sample_channel_values"] += x.size
    result["different_sample_channel_values"] += int(np.count_nonzero(x))
    result["sum_squares"] += float(np.sum(x * x))
    result["max_abs"] = max(result["max_abs"], float(np.max(np.abs(x), initial=0)))
    result["above_one_pcm24_lsb"] += int(np.count_nonzero(np.abs(x) > LSB))
    result["above_four_pcm24_lsb"] += int(np.count_nonzero(np.abs(x) > 4 * LSB))


def finish(result):
    result = dict(result)
    n, power = result["sample_channel_values"], result.pop("sum_squares")
    result["rms"] = np.sqrt(power / n) if n else None
    result["rms_dbfs"] = 10 * np.log10(power / n) if power and n else None
    result["max_abs_dbfs"] = 20 * np.log10(result["max_abs"]) if result["max_abs"] else None
    result["max_abs_pcm24_lsb"] = result["max_abs"] / LSB
    return result


def paths(baseline):
    return {"before": {**{k: baseline / v for k, v in WAVS.items()},
                       "env": baseline / "master_env_sound_C5P2.npy",
                       "premaster_score": baseline / "premaster_score_final_C5P2.npy"},
            "after": {**{k: MUSIC / "out/v3" / v for k, v in WAVS.items()},
                      "env": MUSIC / "cache/v3/master_env_sound_C5P2.npy",
                      "premaster_score": MUSIC / "cache/v3/premaster_score_final_C5P2.npy"}}


def frame_differences(old, new, name, writer):
    totals = {"outside": stats(), "before_edit": stats(), "after_edit": stats()}
    gain_residual = stats()
    changed_frames = []
    with sf.SoundFile(old) as a, sf.SoundFile(new) as b:
        if (a.frames, b.frames, a.samplerate, b.samplerate, a.channels, b.channels) != (N, N, SR, SR, 2, 2):
            raise ValueError(f"Unexpected audio format for {name}")
        for f, count in blocks():
            x = a.read(count * SPF, dtype="float32", always_2d=True).astype(np.float64)
            y = b.read(count * SPF, dtype="float32", always_2d=True).astype(np.float64)
            difference = (y - x).reshape(count, SPF, 2)
            mask = outside(f, count)
            add(totals["outside"], difference[mask])
            add(totals["before_edit"], difference[np.arange(f, f + count) < EDIT[0]])
            add(totals["after_edit"], difference[np.arange(f, f + count) >= EDIT[1]])
            if name == "effects_pre":
                # A shared per-sample gain leaves the two-channel vector collinear.
                # This is a measured compatibility test, not proof of a peak-guard cause.
                den = np.sum(x * x, axis=1)
                scale = np.ones(len(x), np.float64)
                np.divide(np.sum(x * y, axis=1), den, out=scale, where=den > 0)
                residual = (y - x * scale[:, None]).reshape(count, SPF, 2)
                add(gain_residual, residual[mask])
            for i, diff in enumerate(difference):
                peak = float(np.abs(diff).max())
                p = float(np.mean(diff * diff))
                if mask[i] and peak > 0:
                    changed_frames.append(f + i)
                writer.writerow(dict(stem=name, frame=f + i, outside_edit=bool(mask[i]),
                    differing_sample_channels=int(np.count_nonzero(diff)), max_abs_diff=peak,
                    rms_diff=np.sqrt(p), rms_diff_dbfs=10 * np.log10(p) if p else None))
    result = {k: finish(v) for k, v in totals.items()}
    result["outside_changed_frames"] = len(changed_frames)
    result["outside_first_changed_frame"] = min(changed_frames) if changed_frames else None
    result["outside_last_changed_frame"] = max(changed_frames) if changed_frames else None
    if name == "effects_pre":
        result["outside_residual_after_best_per_sample_stereo_gain"] = finish(gain_residual)
    return result


def envelope_comparison(p, writer):
    old, new = (np.load(p[k]["env"], mmap_mode="r") for k in ("before", "after"))
    minimum, maximum, sum_db, n, count_different = np.inf, -np.inf, 0.0, 0, 0
    for f, count in blocks():
        a, b = old[f * SPF:(f + count) * SPF], new[f * SPF:(f + count) * SPF]
        ratio_db = (20 * np.log10(b.astype(np.float64) / a)).reshape(count, SPF)
        mask = outside(f, count)
        values = ratio_db[mask]
        if len(values):
            minimum, maximum = min(minimum, values.min()), max(maximum, values.max())
            sum_db += float(values.sum())
            n += values.size
            count_different += int(np.count_nonzero((b != a).reshape(count, SPF)[mask]))
        for i, row in enumerate(ratio_db):
            writer.writerow(dict(frame=f + i, outside_edit=bool(mask[i]),
                ratio_db_min=float(row.min()), ratio_db_max=float(row.max()), ratio_db_mean=float(row.mean())))
    return dict(outside_samples=n, different_envelope_samples=count_different,
                ratio_db_min=float(minimum), ratio_db_max=float(maximum), ratio_db_mean=sum_db / n)


def faded(y, start):
    """Exact float32 fades used by render_v3.master, applied at global positions."""
    fi, fo = int(0.010 * SR), int(0.35 * SR)
    stop = start + len(y)
    if start < fi:
        n = min(fi, stop) - start
        y[:n] *= np.linspace(0, 1, fi, dtype=np.float32)[start:start + n, None]
    if stop > N - fo:
        a = max(start, N - fo)
        fade = (np.cos(np.linspace(0, np.pi / 2, fo)) ** 2).astype(np.float32)
        y[a - start:] *= fade[a - (N - fo):stop - (N - fo), None]
    return y


def fit_safety_scalar(p):
    """Least-squares fit from unchanged score and saved env; then round to float32."""
    score = np.load(p["premaster_score"], mmap_mode="r")
    env = np.load(p["env"], mmap_mode="r")
    b, a = signal.butter(1, 8.0 / (SR / 2), btype="high")
    state = np.zeros((1, 2), np.float64)
    numerator, denominator = 0.0, 0.0
    with sf.SoundFile(p["score"]) as output:
        for f, count in blocks():
            start, stop = f * SPF, (f + count) * SPF
            filtered, state = signal.lfilter(b, a, np.asarray(score[start:stop], np.float32), axis=0, zi=state)
            expected = faded(filtered.astype(np.float32) * env[start:stop, None], start).astype(np.float64)
            actual = output.read(stop - start, dtype="float32", always_2d=True).astype(np.float64)
            numerator += float(np.sum(expected * actual))
            denominator += float(np.sum(expected * expected))
    fit = numerator / denominator
    return dict(least_squares_gain=fit, nearest_float32_gain=float(np.float32(fit)),
                nearest_float32_gain_db=float(20 * np.log10(np.float32(fit))))


def reconstruction_chunks(p, scalar):
    score = np.load(p["premaster_score"], mmap_mode="r")
    env = np.load(p["env"], mmap_mode="r")
    b, a = signal.butter(1, 8.0 / (SR / 2), btype="high")
    score_state, effects_state = np.zeros((1, 2)), np.zeros((1, 2))
    with ExitStack() as stack:
        files = {name: stack.enter_context(sf.SoundFile(p[name])) for name in WAVS}
        for f, count in blocks():
            start, stop = f * SPF, (f + count) * SPF
            actual = {name: handle.read(stop - start, dtype="float32", always_2d=True)
                      for name, handle in files.items()}
            s, score_state = signal.lfilter(b, a, np.asarray(score[start:stop], np.float32), axis=0, zi=score_state)
            x, effects_state = signal.lfilter(b, a, actual["effects_pre"], axis=0, zi=effects_state)
            s = faded(s.astype(np.float32) * env[start:stop, None], start)
            x = faded(x.astype(np.float32) * env[start:stop, None], start)
            s *= np.float32(scalar)
            x *= np.float32(scalar)
            yield f, count, actual, {"score": s, "effects": x, "master": s + x}


def reconstruction(p, scalars, writer):
    conditions = {label: {stem: stats() for stem in ("master", "score", "effects")}
                  for label in ("before", "after")}
    delta_stats = stats()
    generators = [reconstruction_chunks(p[label], scalars[label]["nearest_float32_gain"])
                  for label in ("before", "after")]
    for old, new in zip(*generators):
        f, count = old[:2]
        mask = outside(f, count)
        for label, item in (("before", old), ("after", new)):
            for name in conditions[label]:
                residual = item[2][name].astype(np.float64) - item[3][name]
                add(conditions[label][name], residual.reshape(count, SPF, 2)[mask])
        actual_delta = new[2]["master"].astype(np.float64) - old[2]["master"]
        reconstructed_delta = new[3]["master"].astype(np.float64) - old[3]["master"]
        residual = (actual_delta - reconstructed_delta).reshape(count, SPF, 2)
        add(delta_stats, residual[mask])
        for i, row in enumerate(residual):
            writer.writerow(dict(frame=f + i, outside_edit=bool(mask[i]),
                max_abs_delta_residual=float(np.abs(row).max()), rms_delta_residual=float(np.sqrt(np.mean(row ** 2)))))
    return {"outside_export_residuals": {label: {stem: finish(v) for stem, v in results.items()}
                                         for label, results in conditions.items()},
            "outside_master_delta_unexplained_by_recorded_inputs_and_envelopes": finish(delta_stats)}


def gain_comparison(p, out):
    """Check common-gain shape and directly match each original hammer's waveform."""
    summary = dict(samples_with_old_stereo_rms_above_minus120_dbfs=0,
                   minimum_gain_ratio=np.inf, maximum_gain_ratio=-np.inf,
                   nonpositive_gain_ratios=0)
    with sf.SoundFile(p["before"]["effects_pre"]) as old, sf.SoundFile(p["after"]["effects_pre"]) as new:
        with (out / "outside-gain.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "outside_edit", "qualified_samples",
                                                        "gain_ratio_min", "gain_ratio_max", "nonpositive_ratios"])
            writer.writeheader()
            for f, count in blocks():
                a = old.read(count * SPF, dtype="float32", always_2d=True).astype(np.float64)
                b = new.read(count * SPF, dtype="float32", always_2d=True).astype(np.float64)
                power = np.sum(a * a, axis=1)
                gain = np.ones(len(a))
                np.divide(np.sum(a * b, axis=1), power, out=gain, where=power > 0)
                gain = gain.reshape(count, SPF)
                qualified = power.reshape(count, SPF) > 2e-12
                mask = outside(f, count)
                for i, row in enumerate(gain):
                    values = row[qualified[i]]
                    minimum = float(values.min()) if len(values) else None
                    maximum = float(values.max()) if len(values) else None
                    nonpositive = int(np.count_nonzero(values <= 0))
                    writer.writerow(dict(frame=f + i, outside_edit=bool(mask[i]), qualified_samples=len(values),
                        gain_ratio_min=minimum, gain_ratio_max=maximum, nonpositive_ratios=nonpositive))
                    if mask[i] and len(values):
                        summary["samples_with_old_stereo_rms_above_minus120_dbfs"] += len(values)
                        summary["minimum_gain_ratio"] = min(summary["minimum_gain_ratio"], minimum)
                        summary["maximum_gain_ratio"] = max(summary["maximum_gain_ratio"], maximum)
                        summary["nonpositive_gain_ratios"] += nonpositive
    original_events = json.loads((out / "before_events_C5P2.json").read_text())
    hammers = [e for e in original_events if e["id"].startswith(("C5.hammer.trap_", "C5.hammer.alone_"))]
    matches = []
    for event in hammers:
        frame = round(event["t"] * FPS)
        a = (frame - 2) * SPF
        b = (frame + 3) * SPF
        old = sf.read(p["before"]["effects_pre"], start=a, stop=b, dtype="float32", always_2d=True)[0]
        new = sf.read(p["after"]["effects_pre"], start=a, stop=b, dtype="float32", always_2d=True)[0]
        template = old[2 * SPF:2 * SPF + int(0.060 * SR)].astype(np.float64)
        match = correlation_rows(new, frame - 2, FPS, template, [frame])[0]
        matches.append(dict(id=event["id"], **match))
    result = {"method": "Best common stereo gain per sample, qualified only where old stereo RMS exceeds "
               "-120dBFS; report positive-gain range rather than allowing arbitrary sign changes unnoticed. "
               "For each of the9 original hammers, independently correlate the first60ms of its actual old "
               "delivered pre-effects waveform against the new pre-effects, searching +/-1frame at1sample steps.",
              "limits": "These are mixtures of all effects in each local window. Strong zero-lag matches "
               "support preserved waveform/timing but are not an isolation or listening check. A common gain "
               "is compatible with peak_guard; it does not independently rederive that function's gain envelope.",
              "outside_gain_summary": summary, "original_hammer_waveform_matches": matches}
    (out / "outside-gain.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gain-only", action="store_true", help="Add only the gain-shape/original-hammer diagnostic")
    args = parser.parse_args()
    out = args.out_dir
    if args.gain_only:
        print(json.dumps(clean(gain_comparison(paths(out / "baseline-full"), out)), indent=2, allow_nan=False))
        return
    destination = out / "outside-comparison.json"
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite {destination}")
    p = paths(out / "baseline-full")
    data = {"comparison_script_sha256": sha256(__file__), "edit_interval_frames_half_open": EDIT,
            "fps": FPS, "sample_rate": SR, "frames": FRAMES,
            "hashes": {label: {name: sha256(path) for name, path in values.items()} for label, values in p.items()},
            "methods": {
                "difference": "Exact decoded sample/channel comparisons; per-frame RMS averages both channels. "
                "1 PCM24 LSB = 2^-23. Outside means frames0..3799 and4280..5919.",
                "pre_effects": "sfxpre is already peak-guarded. Best per-sample stereo-gain residual tests "
                "whether the difference is compatible with a common gain on both channels; it does not prove cause.",
                "reconstruction": "Unchanged premaster score plus each delivered pre-effects stem pass through "
                "the production8Hz causal highpass, recorded float32 envelope, exact head/tail fades, and "
                "a final scalar fitted against the delivered score stem (nearestfloat32). Exports contain "
                "24-bit dither/quantization, so residuals are measured rather than asserted zero.",
                "limits": "This does not rerender any effect or rederive the compressor/limiter envelope. "
                "An envelope accounts for an output difference without proving why its gain changed. "
                "Any pre-effects waveform residual needs separate attribution; no listening check is included."}}
    data["premaster_score_byte_identical"] = data["hashes"]["before"]["premaster_score"] == data["hashes"]["after"]["premaster_score"]
    with (out / "outside-frame-differences.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "frame", "outside_edit", "differing_sample_channels",
                                                   "max_abs_diff", "rms_diff", "rms_diff_dbfs"])
        writer.writeheader()
        data["differences"] = {name: frame_differences(p["before"][name], p["after"][name], name, writer) for name in WAVS}
    with (out / "outside-envelope.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "outside_edit", "ratio_db_min", "ratio_db_max", "ratio_db_mean"])
        writer.writeheader()
        data["envelope"] = envelope_comparison(p, writer)
    data["fitted_final_safety_scalars"] = {label: fit_safety_scalar(v) for label, v in p.items()}
    with (out / "outside-reconstruction.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "outside_edit", "max_abs_delta_residual", "rms_delta_residual"])
        writer.writeheader()
        data["reconstruction"] = reconstruction(p, data["fitted_final_safety_scalars"], writer)
    data["gain_diagnostic"] = gain_comparison(p, out)
    destination.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")
    print(json.dumps(clean(data), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
