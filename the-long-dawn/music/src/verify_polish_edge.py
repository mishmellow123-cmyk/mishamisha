"""Bounded plots and a scoped listening export of AP2's polished edge.

Run through tools/onepy. This is an audit/export tool, never a pipeline hook:
the owner should integrate the lanes' premaster edits before mastering together.
The listening export splices only this lane into an immutable before master,
then applies one measured scalar gain to the whole film. No compressor runs on
that composite, so its out-of-scope sound differs only by gain and quantization.
"""
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
import soundfile as sf

from timeline_v3 import SR, FR
from sound_polish_edge import smooth_curve

CHUNK = SR * 5
SPLICE = ((1818, 0.0), (1822, 1.0), (2389, 1.0), (2399, 0.0))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def measure(path):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                             "loudnorm=I=-16:TP=-1:print_format=json", "-f", "null", "-"],
                            capture_output=True, text=True, check=True)
    info, _ = json.JSONDecoder().raw_decode(result.stderr[result.stderr.rfind("{"):])
    return {"lufs": float(info["input_i"]), "true_peak_dbtp": float(info["input_tp"])}


def blocks(before, after):
    with sf.SoundFile(before) as a, sf.SoundFile(after) as b:
        if a.frames != b.frames or a.frames != 6480 * FR or a.samplerate != SR or b.samplerate != SR:
            raise ValueError("The scoped audit requires two full 6480-frame, 48 kHz masters")
        if a.channels != 2 or b.channels != 2:
            raise ValueError("The scoped audit requires stereo masters")
        for start in range(0, a.frames, CHUNK):
            yield start, a.read(CHUNK, dtype="float32"), b.read(CHUNK, dtype="float32")


def outside_difference(before, after, exact_gain=None):
    """Residual outside [1441,2400), after removing one whole-film gain."""
    xx = xy = 0.0
    for start, a, b in blocks(before, after):
        index = np.arange(start, start + len(a))
        keep = (index < 1441 * FR) | (index >= 2400 * FR)
        x, y = a[keep].astype(np.float64), b[keep].astype(np.float64)
        xx += float(np.sum(x * x)); xy += float(np.sum(x * y))
    gain = xy / xx if exact_gain is None else exact_gain
    error2 = peak = count = 0
    for start, a, b in blocks(before, after):
        index = np.arange(start, start + len(a))
        keep = (index < 1441 * FR) | (index >= 2400 * FR)
        error = b[keep].astype(np.float64) - gain * a[keep].astype(np.float64)
        error2 += float(np.sum(error * error)); count += error.size
        peak = max(peak, float(np.max(np.abs(error))) if error.size else 0.0)
    return dict(gain_db=float(20 * np.log10(gain)), max_abs_residual=peak,
                rms_residual=float(np.sqrt(error2 / count)), compared_channel_samples=count)


def scoped_export(before, after, output):
    temp = output + ".float.wav"
    with sf.SoundFile(temp, mode="w", samplerate=SR, channels=2, subtype="FLOAT") as writer:
        for start, a, b in blocks(before, after):
            frame = np.arange(start, start + len(a), dtype=np.float64) / FR
            w = smooth_curve(frame, SPLICE).astype(np.float32)[:, None]
            writer.write(a + (b - a) * w)
    raw = measure(temp)
    # Centre in the allowed tolerance with 0.04 LU of peak margin; scalar only.
    gain_db = -16.04 - raw["lufs"]
    if raw["true_peak_dbtp"] + gain_db > -1.0:
        raise ValueError("Scoped composite cannot meet both loudness and peak targets with scalar gain")
    gain = float(10 ** (gain_db / 20.0))
    with sf.SoundFile(temp) as reader, sf.SoundFile(output, mode="w", samplerate=SR,
                                                   channels=2, subtype="PCM_24") as writer:
        for y in reader.blocks(CHUNK, dtype="float64"):
            writer.write(y * gain)
    final = measure(output)
    if abs(final["lufs"] + 16.0) > 0.08 or final["true_peak_dbtp"] > -1.0:
        raise ValueError("Scoped listening export failed measured loudness/peak target: " + str(final))
    return dict(raw=raw, gain_db=gain_db, final=final,
                outside=outside_difference(before, output, gain), float_intermediate=temp)


def envelope(path, f0, f1, hop=240):
    a = round(f0 * FR)
    y, sr = sf.read(path, start=a, stop=round(f1 * FR), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError("Envelope source is not 48 kHz")
    n = len(y) // hop
    energy = (y[:n * hop].astype(np.float64).reshape(n, hop, 2) ** 2).mean(axis=(1, 2))
    return (a + (np.arange(n) + 0.5) * hop) / FR, 10 * np.log10(energy + 1e-24)


def plots(before, after, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = (1760, 1840, 1860, 1920, 1940, 2000, 2200, 2380)
    fig, axs = plt.subplots(4, 2, figsize=(16, 12), constrained_layout=True)
    rows = []
    for moment, ax in zip(frames, axs.flat):
        lo, hi = moment - 20, moment + 20
        x, a = envelope(before, lo, hi)
        _, b = envelope(after, lo, hi)
        ax.plot(x, a, color="#6c737a", linewidth=0.85, label="Delivered before")
        ax.plot(x, b, color="#bd5327", linewidth=0.85, label="Scoped after")
        ax.axvline(moment, color="black", linewidth=0.6, alpha=0.6)
        ax.set(title=f"A frame {moment} · 5 ms stereo RMS", xlabel="Frame (24 fps)", ylabel="dBFS")
        ax.grid(alpha=0.17)
        rows.append(dict(frame=moment, before_min_5ms_dbfs=float(a.min()), after_min_5ms_dbfs=float(b.min()),
                         before_max_5ms_step_db=float(abs(np.diff(a)).max()),
                         after_max_5ms_step_db=float(abs(np.diff(b)).max())))
        np.savetxt(os.path.join(directory, f"envelope_{moment}_5ms.csv"), np.stack([x, a, b], axis=1),
                   delimiter=",", header="frame,before_dbfs,after_dbfs", comments="")
    axs.flat[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("A6–A7 · Delivered master and scoped listening export (5 ms windows)", fontsize=16)
    fig.savefig(os.path.join(directory, "sound_envelopes_5ms.png"), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    x, a = envelope(before, 1440, 2400, SR // 2)
    _, b = envelope(after, 1440, 2400, SR // 2)
    ax.plot(x, a, color="#6c737a", label="Delivered before")
    ax.plot(x, b, color="#bd5327", label="Scoped after")
    ax.set(title="A6–A7 build · 500 ms stereo RMS", xlabel="Frame (24 fps)", ylabel="dBFS")
    ax.grid(alpha=0.17); ax.legend()
    fig.savefig(os.path.join(directory, "sound_build.png"), dpi=160)
    plt.close(fig)
    return rows


def build_levels(before, after):
    """RMS over eight beats at entry and the same span approaching the brink."""
    rows = []
    for f0, f1 in ((1441, 1600), (1680, 1840), (1840, 2000), (2080, 2240), (2240, 2400)):
        row = dict(frames=[f0, f1])
        for label, path in (("before", before), ("after", after)):
            y, _ = sf.read(path, start=f0 * FR, stop=f1 * FR, dtype="float64", always_2d=True)
            row[label + "_rms_dbfs"] = float(10 * np.log10(np.mean(y * y) + 1e-24))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = dict(before_path=args.before, before_sha256=sha(args.before), after_path=args.after,
                  after_sha256=sha(args.after), full_remaster_outside=outside_difference(args.before, args.after))
    output = os.path.join(args.out, "after_sound_AP2_scoped.wav")
    report["scoped_export"] = scoped_export(args.before, args.after, output)
    report["scoped_export"]["path"] = output
    report["scoped_export"]["sha256"] = sha(output)
    report["envelopes"] = plots(args.before, output, args.out)
    report["build_levels"] = build_levels(args.before, output)
    with open(os.path.join(args.out, "sound_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
