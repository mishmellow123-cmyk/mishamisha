"""Bounded PCM measurements around D cuts, matching the owner's mono method.

Each stem is measured at its delivered gain. Differences between stems' dB
readings are not component contributions to a mix. These metrics are not an
audition, and independently mastered stems must not be treated as a sum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import soundfile as sf


def rms_db(samples):
    value = float(np.sqrt(np.mean(np.asarray(samples, np.float64) ** 2))) if len(samples) else 0.
    return float(20 * np.log10(value)) if value > 1e-9 else -120.


def chroma(samples, sr):
    """Owner's one-window FFT energy chroma, A=440 bin zero (no STFT)."""
    if len(samples) < 4096:
        return None
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    frequencies = np.fft.rfftfreq(len(samples), 1 / sr)
    keep = (frequencies > 55) & (frequencies < 4000)
    result = np.zeros(12)
    np.add.at(result, (np.round(12 * np.log2(frequencies[keep] / 440.)) % 12).astype(int),
              spectrum[keep] ** 2)
    return result / (np.linalg.norm(result) + 1e-12)


def centroid(samples, sr):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    frequencies = np.fft.rfftfreq(len(samples), 1 / sr)
    return float((spectrum * frequencies).sum() / (spectrum.sum() + 1e-12))


def stereo_energy(samples):
    """Linear same-stem L/R identity; this is not score/effects attribution."""
    x = np.asarray(samples, np.float64)
    left, right = float(np.mean(x[:, 0] ** 2)), float(np.mean(x[:, 1] ** 2))
    cross = float(np.mean(x[:, 0] * x[:, 1]))
    return dict(left_mean_square=left, right_mean_square=right, lr_cross_mean=cross,
                stereo_mean_square=(left + right) / 2,
                mono_mean_square=(left + right + 2 * cross) / 4,
                stereo_rms_dbfs=rms_db(x), mono_rms_dbfs=rms_db(x.mean(axis=1)))


def step_guard(row, limit_db=6.):
    """Owner's 250ms limit in both directions; one second remains diagnostic."""
    if not np.isfinite(limit_db) or limit_db < 0:
        raise ValueError("step limit must be finite and nonnegative")
    values = {key: float(row[key]) for key in ("step250_db", "step1s_db")}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("step readings must be finite")
    return dict(limit_db=limit_db, window="250ms", ok=abs(values["step250_db"]) <= limit_db,
                exceedances=[dict(window=key, db=value,
                                  direction="jump" if value > 0 else "drop")
                             for key, value in values.items()
                             if key == "step250_db" and abs(value) > limit_db],
                diagnostic_1s=dict(db=values["step1s_db"],
                                   above_same_magnitude=abs(values["step1s_db"]) > limit_db,
                                   governs_owner_guard=False))


def measure_window(stereo, sr):
    """Exactly two seconds, cut at the centre; no padding short source audio."""
    x = np.asarray(stereo, np.float64)
    if sr <= 0 or sr % 4 or x.shape != (2 * sr, 2) or not np.isfinite(x).all():
        raise ValueError("need two finite stereo seconds at a positive rate divisible by four")
    mono = x.mean(axis=1)
    before, after = mono[:sr], mono[sr:]
    quarter = sr // 4
    cb, ca = chroma(before, sr), chroma(after, sr)
    result = dict(
        quarter_dbfs=[rms_db(mono[i * quarter:(i + 1) * quarter]) for i in range(8)],
        before_250_dbfs=rms_db(before[-quarter:]), after_250_dbfs=rms_db(after[:quarter]),
        before_1s_dbfs=rms_db(before), after_1s_dbfs=rms_db(after),
        step250_db=rms_db(after[:quarter]) - rms_db(before[-quarter:]),
        step1s_db=rms_db(after) - rms_db(before),
        chroma_before=None if cb is None else cb.tolist(),
        chroma_after=None if ca is None else ca.tolist(),
        chroma_cos=None if cb is None or ca is None else float(cb @ ca),
        chroma_silence_caveat=bool(np.max(np.abs(before)) < 1e-9 or np.max(np.abs(after)) < 1e-9),
        low_level_1s_below_minus90_dbfs=[rms_db(before) < -90., rms_db(after) < -90.],
        centroid_hz=[centroid(before, sr), centroid(after, sr)],
        stereo_before_250=stereo_energy(x[sr - quarter:sr]),
        stereo_after_250=stereo_energy(x[sr:sr + quarter]),
        stereo_before_1s=stereo_energy(x[:sr]), stereo_after_1s=stereo_energy(x[sr:]))
    result["step_guard"] = step_guard(result)
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure_file(path, cuts, fps=24):
    path = Path(path)
    if fps <= 0:
        raise ValueError("fps must be positive")
    digest = file_hash(path)
    rows = []
    with sf.SoundFile(path) as stream:
        if stream.channels != 2:
            raise ValueError("stereo input required")
        sr = stream.samplerate
        for cut in cuts:
            if type(cut) is not int or cut < 0:
                raise ValueError("cuts must be nonnegative integer frames")
            centre = int(cut / fps * sr)  # Owner's cut-to-sample conversion.
            if centre < sr or centre + sr > len(stream):
                raise ValueError(f"cut {cut}: full one-second handles unavailable")
            stream.seek(centre - sr)
            samples = stream.read(2 * sr, dtype="float64", always_2d=True)
            rows.append(dict(cut=cut, centre_sample=centre, **measure_window(samples, sr)))
        metadata = dict(name=path.name, sha256=digest, sample_rate=sr,
                        channels=stream.channels, sample_frames=len(stream), subtype=stream.subtype)
    if file_hash(path) != digest:
        raise ValueError("input changed during measurement")
    return dict(input=metadata, rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", action="append", required=True, metavar="LABEL=PATH")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--joins", type=Path, help="Owner JSON array with integer cut fields")
    group.add_argument("--cuts", help="Comma-separated integer D frames")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cuts = ([row["cut"] for row in json.loads(args.joins.read_text())] if args.joins
            else [int(value) for value in args.cuts.split(",")])
    if not cuts or len(cuts) != len(set(cuts)):
        parser.error("cuts must be nonempty and unique")
    stems = {}
    for item in args.wav:
        label, separator, path = item.partition("=")
        if not separator or not label or label in stems:
            parser.error("each WAV needs a distinct LABEL=PATH")
        stems[label] = measure_file(path, cuts, args.fps)
    result = dict(schema="long-dawn/join-measurements/1", fps=args.fps,
                  scope="Independent delivered stems at their own gains; no inferred mix fractions or audition.",
                  method="Owner mono mean(L,R), eight 250ms RMS windows, adjacent 250ms/1s steps; one-second Hann FFT chroma and centroids. Stereo energy/cross terms describe each stem's channels only.",
                  stems=stems, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for cut in cuts:
        values = []
        for name, stem in stems.items():
            row = next(row for row in stem["rows"] if row["cut"] == cut)
            cosine = "null" if row["chroma_cos"] is None else f"{row['chroma_cos']:.2f}"
            values.append(f"{name}: {row['step250_db']:+.2f}/{row['step1s_db']:+.2f}dB chroma={cosine}")
        print(f"D{cut}: " + "; ".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
