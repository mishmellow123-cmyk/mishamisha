"""Bounded-window measurements for C5P2's forge transition; never renders a stem.

python music/src/measure_c5_instep.py --label before --out-dir /path/to/evidence
python music/src/measure_c5_instep.py --label after --out-dir /path/to/evidence

Audio decoding is restricted to C3704..4260 (one second of filter warm-up,
three seconds of loudness pre-roll, the requested window, and one beat of
post-roll). Hashes stream files in 1 MiB chunks. No full-length audio is loaded.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy import signal
import soundfile as sf

from sound_v3 import kpow

MUSIC = Path(__file__).resolve().parents[1]
F0, F1 = 3800, 4240
SR = 48000
FILES = {"master": "sound_C5P2.wav", "effects": "sfx_C5P2.wav",
         "effects_pre": "sfxpre_C5P2.wav"}
SOURCES = ["src/sound_v3.py", "src/sound_c5_recipes.py", "src/sound_c5_table.py",
           "src/sound_recipes_C5P2.py", "src/score_v3_C5P2.py",
           "v3/barmap_C5P2.json", "v3/cues_C5P2.json", "sound/events_C5P2.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def rms_db(y):
    p = float(np.mean(np.asarray(y, np.float64) ** 2))
    return 10 * np.log10(p) if p > 0 else None


def longest_run(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    if not len(edges):
        return {"samples": 0, "start_offset_samples": None}
    lengths = edges[1::2] - edges[::2]
    i = int(np.argmax(lengths))
    return {"samples": int(lengths[i]), "start_offset_samples": int(edges[2 * i])}


def continuity(y, first_frame, fps):
    sample = lambda f: int(round((f - first_frame) * SR / fps))
    region = y[sample(3848):sample(4240)]
    rows = [{"frame": f, "rms_dbfs": rms_db(y[sample(f):sample(f + 1)]),
             "digital_zero": bool(np.all(y[sample(f):sample(f + 1)] == 0))}
            for f in range(F0, F1 + 1)]
    in_region = [r for r in rows if 3848 <= r["frame"] < 4240]
    nonzero = [r["rms_dbfs"] for r in in_region if r["rms_dbfs"] is not None]
    zero = longest_run(np.all(region == 0, axis=1))
    # Exhaustive sliding 10 ms windows, one-sample hop (not sparse probes).
    power = np.mean(region.astype(np.float64) ** 2, axis=1)
    n = int(round(0.010 * SR))
    # Local summation preserves low-level dither after much louder samples.
    # Subtracting two large prefix sums can round its positive power to zero.
    p10 = np.convolve(power, np.ones(n, np.float64) / n, mode="valid")
    i = int(np.argmin(p10))
    return {"interval_frames_half_open": [3848, 4240], "frame_rms": rows,
            "transition_frames": [r for r in rows if 3847 <= r["frame"] <= 3849],
            "digital_zero_frames": sum(r["digital_zero"] for r in in_region),
            "frames_below_minus90_dbfs_including_zero": sum(
                r["rms_dbfs"] is None or r["rms_dbfs"] < -90 for r in in_region),
            "min_nonzero_frame_rms_dbfs": min(nonzero) if nonzero else None,
            "max_frame_rms_dbfs": max(nonzero) if nonzero else None,
            "longest_digital_zero_run": {**zero, "seconds": zero["samples"] / SR,
                "start_cut_frame": (3848 + zero["start_offset_samples"] * fps / SR
                                    if zero["start_offset_samples"] is not None else None)},
            "minimum_sliding_10ms_rms_dbfs": 10 * np.log10(p10[i]) if p10[i] > 0 else None,
            "minimum_sliding_10ms_start_frame": 3848 + i * fps / SR}


def loudness_rows(y, first_frame, fps, beat_frames):
    power = kpow(y)
    sample = lambda f: int(round((f - first_frame) * SR / fps))

    def lufs(a, b):
        ia, ib = sample(a), sample(b)
        if not np.any(y[ia:ib]):
            return None
        p = float(np.mean(power[ia:ib]))
        return -0.691 + 10 * np.log10(p) if p > 0 else None

    return [{"frame": f, "seconds": f / fps,
             "bar": 1 + int(f // (4 * beat_frames)),
             "beat": 1 + (f % (4 * beat_frames)) / beat_frames,
             "short_term_3s_trailing_lufs": lufs(f - 3 * fps, f),
             "beat_interval_ungated_lufs": lufs(f, f + beat_frames)}
            for f in range(F0, F1 + 1, int(beat_frames))]


def hammer_template():
    """Build only the existing 650 ms event; never construct a full-length stem."""
    import sound_c5_recipes as recipes
    import sound_recipes_C as base
    import sound_v3 as sound

    recipe = recipes.INSTEP_RECIPES["C5.hammer.instep"]
    y, hit = sound.event(recipe, sound.rng_for("C5.hammer.instep"))
    if recipe.get("dist") or recipe.get("send"):
        raise ValueError("The onset template must match a dry, undelayed hammer")
    y = sound.limit_crest(y, recipe.get("crest", base.SPACE.get("event_crest")))
    y = sound.proc(y, hp=base.SPACE.get("stem_hp", 25), hp_order=2)
    template = y[hit:hit + int(round(0.060 * SR))].astype(np.float64)
    source_paths = [Path(sound.src_path(layer["src"][0])) for layer in recipe["layers"]]
    return template, {"recipe": recipe, "hit_sample": hit, "window_seconds": 0.060,
                      "source_hashes": {str(p.relative_to(MUSIC)): sha256(p) for p in source_paths},
                      "first_nonzero_sample_after_hit": int(np.flatnonzero(np.any(template != 0, axis=1))[0])}


def correlation_rows(y, clip_f0, fps, template, frames):
    """Stereo normalized cross-correlation, per-channel DC removed at each lag."""
    template = template - template.mean(axis=0, keepdims=True)
    energy = float(np.sum(template ** 2))
    n = len(template)
    radius = int(round(SR / fps))
    rows = []
    for frame in frames:
        expected = int(round((frame - clip_f0) * SR / fps))
        start = expected - radius
        segment = y[start:expected + radius + n].astype(np.float64)
        numerator = sum(signal.correlate(segment[:, ch], template[:, ch], mode="valid", method="fft")
                        for ch in range(2))
        sq = np.concatenate([np.zeros((1, 2)), np.cumsum(segment ** 2, axis=0)])
        sm = np.concatenate([np.zeros((1, 2)), np.cumsum(segment, axis=0)])
        window_sum = sm[n:] - sm[:-n]
        window_energy = np.sum(sq[n:] - sq[:-n] - window_sum ** 2 / n, axis=1)
        denominator = np.sqrt(np.maximum(window_energy, 0) * energy)
        corr = np.full(len(numerator), np.nan)
        np.divide(numerator, denominator, out=corr, where=denominator > 0)
        if np.any(np.isfinite(corr)):
            i = int(np.nanargmax(corr))
            lag = i - radius
            rows.append({"grid_frame": frame, "correlation": float(corr[i]),
                         "lag_samples": lag, "lag_ms": 1000 * lag / SR,
                         "lag_frames": lag * fps / SR,
                         "matched_template_hit_frame": frame + lag * fps / SR})
        else:
            rows.append({"grid_frame": frame, "correlation": None, "lag_samples": None,
                         "lag_ms": None, "lag_frames": None, "matched_template_hit_frame": None})
    return rows


def onset_comparison(out_dir, after_pre, clip_f0, fps, beat_frames):
    template, metadata = hammer_template()
    template_path = out_dir / "hammer_instep_first60ms.wav"
    sf.write(template_path, template, SR, subtype="FLOAT")
    metadata.update({"template_path": str(template_path), "template_sha256": sha256(template_path)})
    result = {"method": "Normalized stereo template correlation (per-channel DC removed), first 60ms "
              "of actual dry event after its production filters and crest limiter, then stem highpass. "
              "Search every sample within +/-1 video frame of each beat. The best correlation gives "
              "the template's placement lag; weak correlations are not proof of a hammer. "
              "Zero-energy windows return null. Offbeat controls are halfway between intended hits. "
              "This locates the composite unison recording; it cannot establish separate physical "
              "hammer contacts or audibility against the score.", "template": metadata}
    frames = range(int(np.ceil(3848 / beat_frames) * beat_frames), 4240, int(beat_frames))
    controls = [int(f + beat_frames / 2) for f in frames]
    sources = {"after": after_pre}
    baseline_path = out_dir / f"before_effects_pre_C{int(clip_f0)}_C{int(F1 + beat_frames)}.wav"
    if baseline_path.exists():
        sources["before"] = sf.read(baseline_path, dtype="float32", always_2d=True)[0]
    for label, y in sources.items():
        result[label] = {"beat_matches": correlation_rows(y, clip_f0, fps, template, frames),
                         "offbeat_controls": correlation_rows(y, clip_f0, fps, template, controls)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=["before", "after"])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / (args.label + ".json")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite measured evidence: {output}")
    bm = json.loads((MUSIC / "v3/barmap_C5P2.json").read_text())
    fps, bpm = bm["fps"], bm["bpm"]
    beat_frames = fps * 60 / bpm
    if not beat_frames.is_integer() or F0 % beat_frames or F1 % beat_frames:
        raise ValueError("This audit requires the requested endpoints on an integer-frame beat grid")
    read_f0 = F0 - 4 * fps
    clip_f0 = F0 - 3 * fps
    read_f1 = F1 + beat_frames
    data = {"label": args.label, "sample_rate": SR, "fps": fps, "bpm": bpm,
            "beat_frames": beat_frames, "grid_origin_frame": 0,
            "beat_frames_inclusive": [F0, F1],
            "decoded_interval_frames_half_open": [read_f0, read_f1],
            "saved_clip_interval_frames_half_open": [clip_f0, read_f1],
            "methods": {
                "loudness": "BS.1770 K-weighting from sound_v3.kpow, stereo channel powers summed; "
                "-0.691 + 10log10(mean power), ungated. Short-term: [beat time - 3s, beat time). "
                "Beat interval: [beat time, next beat). Digital silence is JSON null. "
                "K filter has 1s of additional decoded warm-up before the earliest measured interval.",
                "rms": "Unweighted mean-square across both channels; 0dBFS is per-channel unit amplitude. "
                "Frame interval is half-open. Digital zero means both channels exactly zero.",
                "minimum_10ms": "Every possible 480-sample window in C3848..4240, one-sample hop; "
                "local float64 mean-square convolution; no subtraction of large prefix sums.",
                "limits": "Effects include all audible effects; energy alone cannot identify forge presence. "
                "No listening or picture synchronization check is performed by this measurement."},
            "measurement_script_sha256": sha256(__file__),
            "sources_sha256": {name: sha256(MUSIC / name) for name in SOURCES}, "audio": {}}
    shutil.copyfile(MUSIC / "sound/events_C5P2.json",
                    args.out_dir / (args.label + "_events_C5P2.json"))
    rows = []
    for name, filename in FILES.items():
        path = MUSIC / "out/v3" / filename
        info = sf.info(path)
        if info.samplerate != SR or info.channels != 2:
            raise ValueError(f"Unexpected format: {path}: {info}")
        y, _ = sf.read(path, start=int(round(read_f0 * SR / fps)),
                       stop=int(round(read_f1 * SR / fps)), dtype="float32", always_2d=True)
        clip = y[int(round((clip_f0 - read_f0) * SR / fps)):]
        clip_path = args.out_dir / f"{args.label}_{name}_C{int(clip_f0)}_C{int(read_f1)}.wav"
        sf.write(clip_path, clip, SR, subtype="FLOAT")
        beats = loudness_rows(y, read_f0, fps, beat_frames)
        data["audio"][name] = {"path": str(path), "sha256": sha256(path),
                               "frames_samples": info.frames, "subtype": info.subtype,
                               "clip_path": str(clip_path), "clip_sha256": sha256(clip_path),
                               "beats": beats, "continuity": continuity(y, read_f0, fps)}
        rows += [{"stem": name, **row} for row in beats]
        if name == "effects_pre" and args.label == "after":
            data["onsets"] = onset_comparison(args.out_dir, clip, clip_f0, fps, beat_frames)
    output.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")
    with (args.out_dir / (args.label + ".csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))
    for name, item in data["audio"].items():
        c = item["continuity"]
        print(json.dumps(clean({"stem": name, "transition_frames": c["transition_frames"],
                               "zero_frames": c["digital_zero_frames"],
                               "below_minus90_frames": c["frames_below_minus90_dbfs_including_zero"],
                               "longest_zero_seconds": c["longest_digital_zero_run"]["seconds"],
                               "minimum_sliding_10ms_rms_dbfs": c["minimum_sliding_10ms_rms_dbfs"]})))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
