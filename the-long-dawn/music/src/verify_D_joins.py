"""Read the delivered D mix at every edit boundary; no audio is rewritten.

Quarter-second cut windows are the owner's working numerical test. One-second
and frame-sliding results are diagnostics, not a listening or masking verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import soundfile as sf

SR, FPS, SAMPLES_PER_FRAME, FRAMES = 48000, 24, 2000, 9200
LSB = 2.0 ** -23
THRESHOLD_DB = 6.0
CUTS = (80, 560, 960, 1040, 1440, 1680, 1760, 2080, 2400, 2720,
        2728, 2758, 2945, 2960, 3200, 3440, 3520, 3760, 4080, 4240,
        4560, 5040, 5180, 5480, 5520, 5554, 5630, 5680, 5840, 6080,
        6400, 6640, 7040, 7360, 7840, 8080, 8320, 8640, 8880, 9120)
REVIEW_CUTS = (1440, 2080, 3520, 3760, 4080, 4240, 4560, 5840, 6080, 6640)
EXCEPTIONS = {
    3440: "Owner exception: isolated gap strike after the required vacuum.",
    3520: "Owner exception: refusal entrance after the nearly exhausted gap tail.",
    9120: "Owner exception: programme ending into silence.",
}
SILENCES = ((3400, 3440), (8660, 8740))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def join_rows(edl, owner_rows):
    """An exact population check prevents missing or extra joins passing quietly."""
    if (edl.get("cut"), edl.get("fps"), edl.get("frames")) != ("D", FPS, FRAMES):
        raise ValueError("not the complete 24 fps, 9200-frame D edit")
    shots = edl.get("shots", [])
    if not shots or shots[0].get("f0") != 0 or shots[-1].get("f1") != FRAMES:
        raise ValueError("incomplete D shot coverage")
    if any(a.get("f1") != b.get("f0") for a, b in zip(shots, shots[1:])):
        raise ValueError("non-contiguous D shot coverage")
    if any(type(s.get(k)) is not int for s in shots for k in ("f0", "f1")):
        raise ValueError("non-integer D shot boundary")
    if any(s["f0"] >= s["f1"] for s in shots):
        raise ValueError("empty or inverted D shot")
    cuts = [s["f0"] for s in shots[1:]]
    if cuts != list(CUTS):
        raise ValueError("D edit differs from the owner's exact 40-join population")
    if [r.get("cut") for r in owner_rows] != list(CUTS):
        raise ValueError("owner reference differs from the exact 40-join population")
    return [dict(cut=b["f0"], outgoing=a.get("code"), incoming=b.get("code"),
                 adopted_picture=[t for t in edl.get("transitions", [])
                                  if t.get("f0", -1) <= b["f0"] <= t.get("f1", -1)
                                  and t.get("kind") != "caption_grade"])
            for a, b in zip(shots, shots[1:])]


def level(x):
    x = np.asarray(x, dtype=np.float64)
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError("empty or nonfinite audio window")
    energy = float(np.mean(x * x, dtype=np.float64))
    peak = float(np.max(np.abs(x)))
    return dict(energy=energy, rms_dbfs=None if energy == 0 else float(10 * np.log10(energy)),
                owner_rms_dbfs=float(10 * np.log10(energy)) if energy > 1e-18 else -120.0,
                peak=peak, peak_pcm24_lsb=peak / LSB,
                at_or_below_four_pcm24_lsb=bool(peak <= 4 * LSB))


def step(before, after):
    a, b = level(before), level(after)
    signed = b["owner_rms_dbfs"] - a["owner_rms_dbfs"]
    actual = (None if a["rms_dbfs"] is None or b["rms_dbfs"] is None
              else b["rms_dbfs"] - a["rms_dbfs"])
    return dict(before=a, after=b, step_db=actual, owner_step_db=signed,
                absolute_owner_step_db=abs(signed), above_6_db=bool(abs(signed) > THRESHOLD_DB),
                floor_qualified=bool(a["at_or_below_four_pcm24_lsb"]
                                     or b["at_or_below_four_pcm24_lsb"]))


def modes(x, width):
    if x.shape != (2 * width, 2):
        raise ValueError("wrong two-sided stereo window length")
    return {name: step(y[:width], y[width:])
            for name, y in (("mono", x.mean(axis=1)), ("stereo", x))}


def validate_wav(handle):
    if (handle.samplerate, handle.channels, handle.subtype, len(handle)) != (
            SR, 2, "PCM_24", FRAMES * SAMPLES_PER_FRAME):
        raise ValueError("D mix must be complete 48 kHz stereo PCM24 with 18400000 sample frames")


def _read(handle, start, length):
    if start < 0 or start + length > len(handle):
        raise ValueError("audio window outside the programme")
    handle.seek(start)
    value = handle.read(length, dtype="float64", always_2d=True)
    if value.shape != (length, 2) or not np.all(np.isfinite(value)):
        raise ValueError("incomplete or nonfinite audio window")
    return value


def controls():
    """These deliberately violate the same numerical gate used on the delivery."""
    a = np.full(100, .01)
    rows = []
    for db in (6.01, -6.01):
        result = step(a, a * 10 ** (db / 20))
        if not result["above_6_db"]:
            raise AssertionError("synthetic signed step escaped the 6 dB gate")
        rows.append(dict(injected_db=db, measured_db=result["owner_step_db"], rejected=True))
    return rows


def audit(wav, edl_path, owner_path, expected_sha256=None):
    started = time.monotonic()
    wav, edl_path, owner_path = map(Path, (wav, edl_path, owner_path))
    identities = {"wav": sha256(wav), "edl": sha256(edl_path), "owner": sha256(owner_path)}
    if expected_sha256 is not None and identities["wav"] != expected_sha256:
        raise ValueError("WAV hash does not match the supplied delivery identity")
    rows = join_rows(json.loads(edl_path.read_text()), json.loads(owner_path.read_text()))
    sliding, silences = [], []
    with sf.SoundFile(wav) as handle:
        validate_wav(handle)
        for row in rows:
            center = row["cut"] * SAMPLES_PER_FRAME
            row["windows"] = {}
            for label, width in (("250ms", SR // 4), ("1s", SR)):
                result = modes(_read(handle, center - width, 2 * width), width)
                result["sample_windows"] = [[center - width, center], [center, center + width]]
                row["windows"][label] = result
            row["explicit_owner_exception"] = EXCEPTIONS.get(row["cut"])
            row["unexpected_modes_over_6_db"] = [
                mode for mode in ("mono", "stereo")
                if row["windows"]["250ms"][mode]["above_6_db"] and row["cut"] not in EXCEPTIONS]
            row["verdict"] = ("unexpected_step" if row["unexpected_modes_over_6_db"] else
                              "owner_exception" if row["cut"] in EXCEPTIONS else "pass")
        for cut in REVIEW_CUTS:
            pivots = range(cut - 3 * FPS, cut + 3 * FPS + 1)
            width = SR // 4
            start = (cut - 3 * FPS) * SAMPLES_PER_FRAME - width
            end = (cut + 3 * FPS) * SAMPLES_PER_FRAME + width
            y = _read(handle, start, end - start)
            values = []
            for frame in pivots:
                i = frame * SAMPLES_PER_FRAME - start
                values.append(dict(frame=frame, **modes(y[i-width:i+width], width)))
            sliding.append(dict(cut=cut, frame_pivots=values,
                                largest={m: max(values, key=lambda r: r[m]["absolute_owner_step_db"])["frame"]
                                         for m in ("mono", "stereo")},
                                over_6_db={m: [r["frame"] for r in values if r[m]["above_6_db"]]
                                           for m in ("mono", "stereo")}))
        for f0, f1 in SILENCES:
            y = _read(handle, f0 * SAMPLES_PER_FRAME, (f1-f0) * SAMPLES_PER_FRAME)
            silences.append(dict(frames=[f0, f1], nonzero_samples=int(np.count_nonzero(y)),
                                 peak=float(np.max(np.abs(y))), ok=bool(not np.any(y))))
    if identities != {"wav": sha256(wav), "edl": sha256(edl_path), "owner": sha256(owner_path)}:
        raise ValueError("an input changed during the audit")
    unexpected = [r["cut"] for r in rows if r["unexpected_modes_over_6_db"]]
    return dict(
        schema="sound_D_join_audit_v1", wav_name=wav.name, identities_sha256=identities,
        verifier_sha256=sha256(__file__), input_identity_unchanged=True,
        method="Every EDL shot boundary, including internal Deep/map joins; exact owner population required. "
               "Half-open adjacent 250 ms and 1 s windows. Mono is the arithmetic channel mean; stereo "
               "averages squared samples across both channels. Owner-compatible level is 10log10(E) for "
               "E > 1e-18, otherwise -120 dBFS. Actual zero levels/steps are null. A peak <= four PCM24 "
               "LSBs is separately marked, never silently exempted. Sliding comparisons use every "
               "integer-frame pivot within +/-3 s, with full 250 ms windows outside the endpoint pivots.",
        limits="RMS differences measure amplitude-energy steps. They cannot certify timbre, phrasing, "
               "perceived continuity or picture/sound quality without listening. Broad scans are diagnostics; "
               "they can include purposeful nearby effects and are not additional edit joins.",
        threshold_db=THRESHOLD_DB, rows=rows, sliding=sliding, required_silences=silences,
        signed_negative_controls=controls(),
        summary=dict(joins=len(rows), nonexempt_joins=len(rows)-len(EXCEPTIONS),
                     unexpected_cuts_over_6_db=unexpected,
                     owner_exceptions=[r["cut"] for r in rows if r["explicit_owner_exception"]],
                     ok=not unexpected and all(s["ok"] for s in silences)),
        elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                       * (1 if sys.platform == "darwin" else 1024),
        audio_files_created=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--edl", type=Path, default=Path("edit/edl/edl_D.json"))
    parser.add_argument("--owner-joins", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    doc = audit(args.wav, args.edl, args.owner_joins, args.expected_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps(doc["summary"]))
    return 0 if doc["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
