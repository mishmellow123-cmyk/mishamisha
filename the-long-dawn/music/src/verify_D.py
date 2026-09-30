"""Independent checks for the D draft's note data and exported audio.

Run the CLI through the production owner's ``onepy`` process guard.  This module
does not import either renderer (their import-time directory creation touches the
shared production cache).  The loudness and level-map definitions are copied from
``analyze_v3.st_loudness`` and its per-section battery; true peak follows
``render_v2.tp_peaks``: 4x polyphase oversampling, eight-second blocks, 256-sample
overlap.  These are numerical checks, not an audition or measured picture lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import wave

import numpy as np

SR, FPS, FRAME_N = 48000, 24, 2000
FRAMES, SAMPLE_FRAMES = 9200, 18_400_000
TARGET_LUFS, LUFS_TOLERANCE, TRUE_PEAK_LIMIT = -16.0, 0.1, -1.2


def true_peak_db(x, chunk=8 * SR, pad=256):
    """Stock 4x true peak without allocating the full per-sample peak vector."""
    from scipy import signal
    peak = 0.0
    for a in range(0, len(x), chunk):
        b = min(len(x), a + chunk)
        a0, b0 = max(0, a - pad), min(len(x), b + pad)
        y = signal.resample_poly(np.asarray(x[a0:b0], np.float64), 4, 1, axis=0)
        local = np.abs(y).max(axis=1).reshape(-1, 4).max(axis=1).astype(np.float32)
        peak = max(peak, float(local[a - a0:a - a0 + b - a].max()))
    return float(20 * np.log10(peak + 1e-12))


def lufs(x):
    import pyloudnorm as pyln
    return float(pyln.Meter(SR).integrated_loudness(np.asarray(x, np.float64)))


def st_loudness(x, win=3.0, hop=0.5):
    """Exactly analyze_v3's three-second windows and half-second hops."""
    import pyloudnorm as pyln
    meter, out, t = pyln.Meter(SR), [], 0.0
    while t + win <= len(x) / SR + 1e-6:
        seg = np.asarray(x[int(t * SR):int((t + win) * SR)], np.float64)
        v = meter.integrated_loudness(seg) if np.abs(seg).max() > 1e-6 else -120.0
        out.append((t + win / 2, float(v) if np.isfinite(v) else -120.0))
        t += hop
    return np.array(out) if out else np.zeros((0, 2))


def level_rows(track, sections, bands, hard_silences=()):
    """Engine section masks, with full-window containment for declared silence.

    The engine's one-second interior margin lets a three-second window include
    half a second of a preceding cue. Only a section wholly inside an explicit
    master silence uses full containment. Its legacy values remain in the row;
    exact-zero checks independently judge every sample of the whole silence.
    """
    band_by_id = {sid: (lo, hi) for sid, lo, hi in bands}
    anchor = float(track[:, 1].max()) if len(track) else None
    rows = []
    for sec in sections:
        sid = sec["id"]
        if sid not in band_by_id:
            continue
        t0, t1 = sec.get("t0", sec["f0"] / FPS), sec.get("t1", sec["f1"] / FPS)
        mask = (track[:, 0] >= t0 + 1.0) & (track[:, 0] <= t1 - 1.0)
        if len(track) and not np.any(mask):
            mask = np.abs(track[:, 0] - (t0 + t1) / 2) <= 1.6
        lo, hi = band_by_id[sid]
        policy, legacy = "engine_1s_interior", None
        if any(a <= sec["f0"] and sec["f1"] <= b for a, b in hard_silences):
            policy = "fully_contained_in_declared_silence"
            if anchor is not None and np.any(mask):
                vmax, vmed = float(track[mask, 1].max()), float(np.median(track[mask, 1]))
                legacy = dict(windows=int(mask.sum()), max_lufs=vmax, median_lufs=vmed,
                              relative_max_lu=vmax-anchor, relative_median_lu=vmed-anchor,
                              ok=bool(vmax-anchor <= hi+.5 and vmed-anchor >= lo-.5))
            mask = (track[:, 0] >= t0 + 1.5) & (track[:, 0] <= t1 - 1.5)
        if anchor is None or not np.any(mask):
            rows.append(dict(check="level band", section=sid, ok=False, reason="no windows measured",
                             window_policy=policy, legacy_engine_window=legacy))
            continue
        vmax, vmed = float(track[mask, 1].max()), float(np.median(track[mask, 1]))
        rmax, rmed = vmax - anchor, vmed - anchor
        rows.append(dict(check="level band", section=sid, ok=bool(rmax <= hi + .5 and rmed >= lo - .5),
                         windows=int(mask.sum()), anchor_lufs=anchor, max_lufs=vmax, median_lufs=vmed,
                         relative_max_lu=rmax, relative_median_lu=rmed, band_lu=[lo, hi], tolerance_lu=.5,
                         window_policy=policy, **({"legacy_engine_window": legacy} if legacy is not None else {})))
    return rows


def level_map(x, score, bm):
    track = st_loudness(x)
    return level_rows(track, bm.sections, score.levels, score.hard_silences), track


def pcm_prefix_sha256(path, frames=1440):
    """Hash actual packed WAV PCM bytes, not a float conversion of those bytes."""
    with wave.open(str(path), "rb") as fh:
        if (fh.getframerate(), fh.getnchannels(), fh.getsampwidth()) != (SR, 2, 3):
            raise ValueError("opening identity requires 48 kHz stereo 24-bit PCM")
        remain, digest = frames * FRAME_N, hashlib.sha256()
        while remain:
            take = min(remain, SR * 8)
            data = fh.readframes(take)
            if len(data) != take * 6:
                raise ValueError("WAV is shorter than the opening window")
            digest.update(data)
            remain -= take
        return digest.hexdigest()


def file_sha256(path):
    """Whole-file receipt identity, streamed without retaining the WAV in RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def silence_row(y, f0, f1, *, first_frame=0, label="silence"):
    a, b = int((f0 - first_frame) * FRAME_N), int((f1 - first_frame) * FRAME_N)
    if a < 0 or b > len(y) or b <= a:
        return dict(check="silence", label=label, frames=[f0, f1], ok=False, reason="missing full window")
    seg = y[a:b]
    finite = bool(np.isfinite(seg).all())
    peak = float(np.abs(seg).max()) if finite else None
    return dict(check="silence", label=label, frames=[f0, f1], exact_zero=finite and peak == 0,
                peak_dbfs=None if peak is None else float(20 * np.log10(peak + 1e-12)),
                ok=finite and peak == 0)


def _audible(part, note):
    return note.pitch is not None and note.gain_db + part.gain_db > -120 and (note.vel is None or note.vel > 0)


def ring_completions(score, families=None):
    """Find the actual five-note RING gesture, invariant to register and transposition.

    Search single instruments plus specified families; each family may distribute
    the melody among several desks.  The interval sequence and onset ratios are
    tested, so a stray Ab and a metadata label cannot establish completion.
    """
    groups = [(name,) for name in score.parts]
    # A fifth note can migrate into a new desk without changing phrase metadata.
    # Drum MIDI numbers select samples; they are not pitches in this melody.
    groups += [tuple(pn for pn, p in score.parts.items() if p.bus != "perc")]
    groups += [tuple(x) for x in (families if families is not None else getattr(score, "ring_families", []))]
    found = {}
    for names in groups:
        entries = [(round(n.start, 6), float(n.pitch), pn, i) for pn in names if pn in score.parts
                   for i, n in enumerate(score.parts[pn].notes) if _audible(score.parts[pn], n)]
        pitches = {}
        for t, p, pn, i in entries:
            pitches.setdefault(p, {}).setdefault(t, []).append((pn, i))
        for t, root, pn, i in entries:
            for second in pitches.get(root + 6, {}):
                unit = second - t
                if not .1 <= unit <= 16:
                    continue
                times = [round(t + k * unit, 6) for k in (0, 1, 2, 3, 3.5)]
                ps = [root + k for k in (0, 6, 12, 10, 6)]
                refs = [next((refs for onset, refs in pitches.get(p, {}).items() if abs(onset-tm) < 1e-4), [])
                        for tm, p in zip(times, ps)]
                if all(refs):
                    key = (t, root, round(unit, 6))
                    found[key] = dict(start_frame=t * 20, terminal_frame=times[-1] * 20, root_midi=root,
                                      unit_beats=unit, note_refs=[r[0] for r in refs])
    return sorted(found.values(), key=lambda d: (d["start_frame"], d["root_midi"]))


def ring_problems(score, families=None):
    hits = ring_completions(score, families)
    problems = []
    vision = []
    for hit in hits:
        a, b = hit["start_frame"], hit["terminal_frame"]
        if any(lo <= a and b < hi for lo, hi in ((1440, 1760), (5840, 6080))):
            continue
        if 3200 <= a and b < 3400:
            vision.append(hit)
        else:
            problems.append(f"RING COMPLETION OUTSIDE VISION: {a:g}..{b:g}, root {hit['root_midi']:g}")
    # Octave doublings are one gesture; separated or independently transposed completions are not.
    gestures = {(round(h["terminal_frame"], 5), h["root_midi"] % 12) for h in vision}
    if len(gestures) != 1:
        problems.append(f"VISION COMPLETION COUNT: {len(gestures)}; expected one gesture")
    return problems


def beacon_problems(score, dawn_frame=7360):
    """Find a complete major BEACON in the kit or C-illumination rhythm.

    CALL, ANSWER, HOME may pass among instruments. Transposition and uniform
    rhythmic augmentation do not hide an early full cadence. Other melodies
    and arbitrary reharmonizations are outside this deliberately stated scope.
    """
    by_pitch = {}
    for part in score.parts.values():
        if part.bus == "perc":
            continue
        for n in part.notes:
            if _audible(part, n):
                by_pitch.setdefault(n.pitch, set()).add(n.start)
    intervals = (0, 7, 12, 12, 11, 9, 4, 9, 7, 4, 0)
    rhythms = ((0, 1, 2, 4, 5, 5.5, 6, 8, 9, 10, 11),
               (0, 1, 2, 3, 4, 4.5, 5, 7, 8, 9, 10))
    early = set()
    for root, starts in by_pitch.items():
        for start in starts:
            if start * 20 >= dawn_frame:
                continue
            for second in by_pitch.get(root + 7, ()):
                unit = second - start
                if not .1 <= unit <= 8:
                    continue
                for rhythm in rhythms:
                    if all(any(abs(actual - (start + k * unit)) < 1e-4
                               for actual in by_pitch.get(root + interval, ()))
                           for k, interval in zip(rhythm, intervals)):
                        early.add((round(start * 20, 4), root))
    return [f"FULL BEACON BEFORE DAWN: {frame:g}, root {root:g}" for frame, root in sorted(early)]


def call_problems(score, bm, parts=None, events=("crowns_kindle", "two_fires")):
    names = parts if parts is not None else getattr(score, "call_parts", ())
    if len(names) != 2 or any(pn not in score.parts for pn in names):
        return ["CALL PAIR MISSING"]
    problems = []
    for pn, pan in zip(names, (-1., 1.)):
        part = score.parts[pn]
        if part.pan != pan:
            problems.append(f"CALL PAN: {pn}")
    for event in events:
        f = bm.event(event)["frame"]
        seqs = []
        for pn in names:
            part = score.parts[pn]
            notes = sorted([n for n in part.notes if _audible(part, n) and f - 1e-5 <= n.start * 20 < f + 80],
                           key=lambda n: n.start)
            seq = [(round(n.start * 20, 5), n.pitch) for n in notes[:3]]
            seqs.append(seq)
            if len(seq) != 3 or seq[0][0] != f or [p for _, p in seq] != [62, 69, 74]:
                problems.append(f"CALL OFF EVENT: {event}, {pn}")
        if seqs[0] != seqs[1]:
            problems.append(f"CALL NOT UNISON: {event}")
    return problems


def binding_problems(score, bm):
    """Validate each declared picture hit against its actual note and named event."""
    problems = []
    for hit in getattr(score, "sync_bindings", []):
        try:
            note = score.parts[hit["part"]].notes[hit["note_index"]]
            expected = bm.event(hit["event"])["frame"] + hit.get("offset_frames", 0)
        except (KeyError, IndexError):
            problems.append(f"BROKEN NAMED HIT: {hit}")
            continue
        if abs(note.start * 20 - expected) > 1e-4:
            problems.append(f"UNBOUND HIT FRAME: {hit['event']}, {hit['part']}")
    return problems


def watch_problems(score, parts=None, start=4560, end=7840):
    names = parts if parts is not None else getattr(score, "watch_parts", ())
    spans = sorted((max(start, n.start * 20), min(end, (n.start + n.dur) * 20))
                   for pn in names if pn in score.parts for n in score.parts[pn].notes
                   if _audible(score.parts[pn], n) and n.pitch == getattr(score, "watch_pitch", 50) and n.start * 20 < end
                   and (n.start + n.dur) * 20 > start)
    covered = start
    for lo, hi in spans:
        if lo > covered + 1e-4:
            return [f"WATCH GAP: {covered:g}..{lo:g}"]
        covered = max(covered, hi)
    return [] if covered >= end - 1e-4 else [f"WATCH GAP: {covered:g}..{end:g}"]


def source_hash_rows(baseline_path, root):
    """Validate a supplied lane baseline; no invented or regenerated baseline."""
    data = json.loads(Path(baseline_path).read_text())
    rows = []
    entries = data.get("files", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        entries = [dict(path=k, sha256=v if isinstance(v, str) else v.get("sha256")) for k, v in entries.items()]
    for record in entries:
        rel = record.get("path", record.get("file"))
        if not rel or not record.get("sha256"):
            continue
        path = Path(root) / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        rows.append(dict(check="source identity", path=rel, expected_sha256=record["sha256"],
                         sha256=actual, ok=actual == record["sha256"]))
    if not rows:
        rows.append(dict(check="source identity", ok=False, reason="baseline contained no checked files"))
    return rows


def verify(path, opening_path, score, bm, *, receipt_path=None):
    import soundfile as sf
    wav_before = file_sha256(path)
    info = sf.info(path)
    rows = [dict(check="WAV format and length", ok=(info.samplerate == SR and info.channels == 2
                 and info.subtype == "PCM_24" and info.frames == SAMPLE_FRAMES), sample_rate=info.samplerate,
                 channels=info.channels, subtype=info.subtype, sample_frames=info.frames,
                 expected_sample_frames=SAMPLE_FRAMES)]
    a, b = pcm_prefix_sha256(opening_path), pcm_prefix_sha256(path)
    rows.append(dict(check="AP2 opening raw PCM identity", frames=[0, 1440], source_sha256=a, output_sha256=b, ok=a == b))
    y, _ = sf.read(path, dtype="float32", always_2d=True)
    rows.append(dict(check="finite PCM", ok=bool(np.isfinite(y).all())))
    for window in score.hard_silences:
        if isinstance(window, dict):
            f0, f1 = window["f0"], window["f1"]
            label = window.get("name", window.get("label", "silence"))
        else:
            f0, f1 = window[:2]
            label = str(window[2]) if len(window) > 2 else "silence"
        rows.append(silence_row(y, f0, f1, label=label))
    loudness, peak = lufs(y), true_peak_db(y)
    rows += [dict(check="integrated loudness", lufs=loudness, target_lufs=TARGET_LUFS, tolerance_lu=LUFS_TOLERANCE,
                  ok=bool(np.isfinite(loudness) and abs(loudness - TARGET_LUFS) <= LUFS_TOLERANCE)),
             dict(check="true peak", dbtp=peak, ceiling_dbtp=TRUE_PEAK_LIMIT,
                  ok=bool(np.isfinite(peak) and peak <= TRUE_PEAK_LIMIT))]
    lev, track = level_map(y, score, bm)
    rows += lev
    if receipt_path:
        receipt = json.loads(Path(receipt_path).read_text())
        peak_rss = receipt.get("peak_rss_bytes")
        rows.append(dict(check="render peak RSS", peak_rss_bytes=peak_rss, limit_bytes=3 * 1024**3,
                         ok=isinstance(peak_rss, (int, float)) and 0 < peak_rss < 3 * 1024**3))
    wav_after = file_sha256(path)
    unchanged = wav_before == wav_after
    rows.append(dict(check="WAV unchanged during verification", before_sha256=wav_before,
                     after_sha256=wav_after, ok=unchanged))
    return dict(ok=all(r["ok"] for r in rows), checks=rows, short_term_loudness=track.tolist(),
                wav_sha256=wav_before if unchanged else None,
                events=[dict(e) for e in bm.events],
                measurement_scope="Audio values measured on this WAV; event picture frames retain their declared status.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--opening", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--receipt")
    ap.add_argument("--bindings", help="Optional measured event binding JSON accepted by DraftMap")
    ap.add_argument("--source-baseline", help="Pre-D lane source_inventory.json; all supplied hashes are checked")
    args = ap.parse_args(argv)
    from barmap_D_score import DraftMap
    import score_v3_D as D
    # Note analysis does not measure sample attacks or need to load sample audio.
    import kit_v3
    kit_v3.attack_time = lambda *a, **k: 10.0
    bm = DraftMap() if args.bindings is None else DraftMap(json.loads(Path(args.bindings).read_text()))
    score = D.build(bm)
    result = verify(args.wav, args.opening, score, bm, receipt_path=args.receipt)
    problems = D.check(score, bm) + ring_problems(score) + call_problems(score, bm) + watch_problems(score)
    problems += beacon_problems(score)
    problems += binding_problems(score, bm)
    result["checks"].append(dict(check="score rules", problems=problems, ok=not problems))
    if args.source_baseline:
        result["checks"] += source_hash_rows(args.source_baseline, Path(__file__).resolve().parents[2])
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    result["verification_peak_rss_bytes"] = peak_rss_bytes
    result["checks"].append(dict(check="verification peak RSS", peak_rss_bytes=peak_rss_bytes,
                                 limit_bytes=3 * 1024**3, ok=0 < peak_rss_bytes < 3 * 1024**3))
    result["ok"] = all(r["ok"] for r in result["checks"])
    Path(args.json).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["checks"]:
        print(("PASS " if row["ok"] else "FAIL ") + json.dumps(row))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
