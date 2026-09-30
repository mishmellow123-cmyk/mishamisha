"""Fail-closed numerical sync audit of delivered A/AP2 sound (no rendering or listening).

Run after sound_v3.py A, through the evening's onepy memory gate::

    python music/src/audit_sync_a.py --json /tmp/syncA.json

--cut auto chooses AP2 when sound_AP2.wav exists, otherwise A. It selects that cut's
matching sound mix, score/SFX stems, events and barmap; missing AP2 inputs fail without
falling back to A. --music-root selects an integrated owner's music tree read-only.
Every selected events_<cut>.json row must have one
picture_sync_A.json ``audit.events`` row with id, kind (hard/soft/bed), picture_frame,
search_frames [start, stop), control_frames [start, stop), and band_hz [low, high].
Search/control bounds may be fractional frames; they are rounded to the nearest sample.
The control must immediately precede the search. Null picture frames still permit
audio-onset measurement but remain unresolved with null timing offsets;
legacy ``t`` and event table timestamps are never substitutes for measured onsets.
A crossing at the film boundary has no preceding in-film negative control. A cue
scheduled at zero may cross later: an independently checked quiet opening interval
can then supply the control. Its scheduled start alone never certifies an onset.

The onset is the first trailing 5 ms stereo mean-square window, sampled every 1 ms,
above max(-60 dBFS, search peak - 20 dB), after a causal fourth-order Butterworth
bandpass. The control peak must be at least 10 dB below that threshold. Filter and
window delay are INCLUDED, not guessed away. Onsets in a continuous bed are therefore
NOT_ISOLATED. Hard tolerance is [-0.5, +1] frames (0..+1 ideal); soft/bed is +/-3.

``mix_onset_frame`` is ONLY a numerical masking proxy: the isolated band contribution
must exceed -60 dBFS and masker band energy minus 12 dB, as well as its onset threshold.
For a reconciled solo the masker is actual mix minus that solo, including every other
effect. Aggregate buses use the other bus as masker and require validated attribution.
The actual delivered mix must also exceed -60 dBFS, meet the same threshold, and retain
at least contribution energy minus 6 dB in that band; this conservative numerical
retention rule rejects cancellation that a stem-to-stem ratio would miss.
This is not a listening judgment. The full mix is independently checked, in 1-second
chunks, against score + SFX within independently dithered output/quantization bounds. Aggregate
SFX can establish an isolated band onset but cannot establish which overlapping cue
caused it; this limitation is recorded on every row.
An aggregate-SFX PASS also requires detector_validated=true and nonempty
detector_evidence in its picture record, plus detector_audio_sha256 with contribution
and mix keys containing the exact full-file SHA256 of those WAVs. Both must match the
current files; file hashes are reported in the format checks and streamed in 256 KiB
chunks. Missing/stale fingerprints make aggregate attribution UNRESOLVED. Reconciled
named cue contributions use their current reconstruction check instead. The measured feature is
threshold-crossing energy; a pre-scrape and the intended impact are not interchangeable.
Every barmap sync point also requires an audit.score_sync record with id, frame,
status="measured", measured_picture_frame, and kind (hard or soft). Its ``audio``
object must supply search_frames, control_frames, band_hz, detector_validated and
detector_evidence; the same detector measures the score bus with SFX as masker.
A reference-only point instead declares role="reference" and a reference_reason
(for example, an unused score field); this does not claim a rendered score arrival.
Missing score waveform checks fail, even when the metadata matches the picture.
Every K.walk note also requires audit.score_steps with id="feet.<frame>", frame,
status="measured", measured_picture_frame, visible_contact=true, contact_evidence,
and an audio detector object. Steps are always hard. The CLI builds note metadata from
the selected score source in a separate interpreter, without rendering: all sounding
feet/feet_* notes are covered, including AP2's feet_go and excluding its muted A notes.
Current source/cue fingerprints are reported; those fingerprints describe the selected
builder, not proof that a WAV came from that builder. Waveform checks remain required.
Legacy audit.score_sync and score_steps apply to A. AP2 requires explicit arrays at
audit.score_sync_by_cut.AP2 and audit.score_steps_by_cut.AP2; missing arrays fail without
inheriting A's evidence. Shared audit.events continues to use A.* effect identifiers.

Optional --cue-dir supplies postmaster cue WAVs (never rendered here), with manifest.json
``{"events": {"A.impact": {"path": "impact.wav", "offset_samples": 5200000}}}``.
These clips are zero outside their declared extent. Every event must have exactly one
clip, and their streamed sum must reconstruct the delivered SFX within the reported
conversion/dither/float32 budget before any clip can certify a named onset. The sum
checks the signal contribution; semantic attribution still depends on the manifest.
Missing/stale clips are failures, never a fallback to aggregate SFX. All audio reads, including
event envelopes, are streamed; no full-film audio array is materialized.
--solo-dir accepts the owner's sparse NPZ bank instead of cue-dir, only after its gain,
guard and filter reconstruction has passed against the selected delivered SFX.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.signal import butter, sosfilt

SR, FPS, FRAMES = 48000, 24, 6480
FR = SR // FPS
SAMPLES = FRAMES * FR
BLOCK = SR
WINDOW, HOP = SR // 200, SR // 1000
FLOOR_DB, UNDER_PEAK_DB, CONTROL_MARGIN_DB, MASK_DB = -60.0, 20.0, 10.0, 12.0
RETENTION_DB = 6.0
MUSIC = Path(__file__).resolve().parents[1]
METHOD = "causal Butterworth order 4; trailing 5 ms stereo energy; 1 ms hop"
# Reviewed on this cut. Re-check K.walk, score construction, the time grid and cue overrides before
# updating a fingerprint; automatic refresh would turn the source-contract gate into a no-op.
# Hash exact module bytes, including the whole kit: AST dumps differ between Python 3.12/3.13.
STEP_CONTRACT = {
    "src/score_v3_A.py": "3fc11ea1ee6ccbf15bcf9720101d601836d4c9313ce23d3499d0a43c7c7940fd",
    "src/kit_v3.py": "fbd46ec15ac2226626326af31e2e0fe113d498006d4b23e19f4359dc232b878b",
    # Reviewed 30 Sep (cutd): D-only status and CLI --all selection; the grid/score paths are unchanged.
    # Full BarMap state and validation compared equal before/after for A/B/C/C5/C5P2/AP2. Other pins unchanged.
    "src/timeline_v3.py": "e804f11da8d41867a14ad01cd1c74fba4c8d9586291ebe34df784ec6f6e6c5f5",
    "v3/cues_A.json": "0f8680be6cccb38fbffc95abad437e3ca674cdf85a01813b0a440225ff8c5e2f",
}


class InvalidAudio(ValueError):
    pass


class WaveAudio:
    """Seekable file, at most one bounded chunk returned by read()."""

    def __init__(self, path):
        import soundfile as sf
        self.path = str(path)
        self.file = sf.SoundFile(self.path)
        self.samplerate, self.channels = self.file.samplerate, self.file.channels
        self.frames, self.subtype = len(self.file), self.file.subtype
        self._fingerprint = None
        self._opened_stat = self._stat()

    def _stat(self):
        stat = Path(self.path).stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def file_sha256(self):
        """Full encoded-file identity; reuse the hash only while the opened file is unchanged."""
        if self._stat() != self._opened_stat:
            raise InvalidAudio("audio file changed after it was opened")
        if self._fingerprint is None:
            digest = hashlib.sha256()
            with open(self.path, "rb") as stream:
                for chunk in iter(lambda: stream.read(256 * 1024), b""):
                    digest.update(chunk)
            if self._stat() != self._opened_stat:
                raise InvalidAudio("audio file changed during fingerprinting")
            self._fingerprint = digest.hexdigest()
        return self._fingerprint

    def read(self, start, stop):
        self.file.seek(start)
        return self.file.read(stop - start, dtype="float64", always_2d=True)

    def close(self):
        self.file.close()


class OffsetAudio:
    """A cropped cue placed in a full-length, otherwise silent, film timeline."""

    def __init__(self, source, offset):
        self.source, self.offset = source, offset
        self.samplerate, self.channels = source.samplerate, source.channels
        self.frames, self.subtype = SAMPLES, source.subtype

    def read(self, start, stop):
        out = np.zeros((stop - start, self.channels), np.float64)
        a, b = max(start, self.offset), min(stop, self.offset + self.source.frames)
        if b > a:
            out[a - start:b - start] = self.source.read(a - self.offset, b - self.offset)
        return out


class DifferenceAudio:
    """The actual mix minus one reconciled contribution: every competing cue remains."""

    def __init__(self, mix, contribution):
        self.mix, self.contribution = mix, contribution
        self.samplerate, self.channels, self.frames = mix.samplerate, mix.channels, mix.frames
        self.subtype = "DOUBLE"

    def read(self, start, stop):
        return read_chunk(self.mix, start, stop) - read_chunk(self.contribution, start, stop)


def _finite_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _integer(value):
    return _finite_number(value) and int(value) == value


def quantization_bound(subtype):
    """One whole LSB per PCM file allows independent truncation as well as rounding."""
    bits = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
    if subtype in bits:
        return 2.0 ** (1 - bits[subtype])
    if subtype == "FLOAT":
        return 2 * np.finfo(np.float32).eps
    if subtype == "DOUBLE":
        return 2 * np.finfo(np.float64).eps
    raise InvalidAudio(f"unsupported quantization subtype {subtype!r}")


def check_format(source, name, expected_samples=SAMPLES):
    ok = source.samplerate == SR and source.channels == 2 and source.frames == expected_samples
    return dict(check="format", name=name, ok=bool(ok), samplerate=source.samplerate,
                channels=source.channels, samples=source.frames, expected_samples=expected_samples)


def read_chunk(source, start, stop):
    x = source.read(start, stop)
    if x.shape != (stop - start, 2) or not np.isfinite(x).all():
        raise InvalidAudio("missing, non-stereo, or non-finite audio samples")
    return x


def check_mix(mix, sfx, score, expected_samples=SAMPLES):
    """Scan every sample; a matching header alone cannot certify the rendered mix."""
    try:
        bounds = [quantization_bound(x.subtype) for x in (mix, sfx, score)]
        maximum, limit, checked = 0.0, 0.0, 0
        dither = 2.0 ** -23
        budget = {}
        for a in range(0, expected_samples, BLOCK):
            b = min(a + BLOCK, expected_samples)
            m, e, s = (read_chunk(x, a, b) for x in (mix, sfx, score))
            # FLOAT can represent samples beyond full scale; scale its rounding bound accordingly.
            peaks = [float(np.abs(x).max()) for x in (m, e, s)]
            conversion = sum(q * max(1.0, peak) if src.subtype in ("FLOAT", "DOUBLE") else q
                             for q, peak, src in zip(bounds, peaks, (mix, sfx, score)))
            # render_v3.master:253-260 dithers all three outputs independently; mix.tpdf_dither_24:183
            # adds two U(-.5,.5) draws times 2^-23, so each file has <=1 PCM24 LSB dither before
            # conversion. The float32 m_out=s_out+x_out and three x+dither additions add rounding.
            # A full epsilon per addition (twice unit roundoff), padded by conversion+dither,
            # conservatively bounds those four operations without fitting the observed residual.
            eps = np.finfo(np.float32).eps
            padded = [peak + q + dither for peak, q in zip(peaks, bounds)]
            arithmetic = eps * (max(1.0, padded[1] + padded[2]) + sum(max(1.0, p) for p in padded))
            budget = dict(conversion_bound=conversion, independent_tpdf_bound=3 * dither,
                          float32_arithmetic_bound=float(arithmetic))
            bound = sum(budget.values())
            error = float(np.abs(m - (s + e)).max())
            maximum, limit, checked = max(maximum, error), max(limit, bound), checked + b - a
            if error > bound:
                return dict(check="mix_equals_score_plus_sfx", ok=False, samples_checked=checked,
                            maximum_error=maximum, quantization_bound=bound, error_budget=budget,
                            first_failing_chunk_sample=a)
        return dict(check="mix_equals_score_plus_sfx", ok=True, samples_checked=checked,
                    maximum_error=maximum, quantization_bound=limit, last_chunk_error_budget=budget,
                    method="independent TPDF24 + output conversion + float32 mix/dither additions")
    except (InvalidAudio, ValueError, RuntimeError) as exc:
        return dict(check="mix_equals_score_plus_sfx", ok=False, error=str(exc))


def check_cue_reconstruction(sfx, cues, expected_ids, expected_samples=SAMPLES):
    """Trust no named clip until the exact cue set reconstructs the delivered SFX bus."""
    base = dict(check="cue_sum_equals_sfx", ok=False, cue_count=len(cues))
    if (not expected_ids or any(not isinstance(eid, str) for eid in expected_ids) or
            len(set(expected_ids)) != len(expected_ids) or set(cues) != set(expected_ids)):
        return dict(base, reason="cue manifest must cover every event exactly once, without unknown ids")
    try:
        sources = list(cues.values())
        quanta = [quantization_bound(src.subtype) for src in [sfx] + sources]
        maximum, largest_bound, checked = 0.0, 0.0, 0
        budget = {}
        for a in range(0, expected_samples, BLOCK):
            b = min(a + BLOCK, expected_samples)
            actual = read_chunk(sfx, a, b)
            total = np.zeros_like(actual)
            peaks = [float(np.abs(actual).max())]
            for src in sources:
                chunk = read_chunk(src, a, b)
                total += chunk
                peaks.append(float(np.abs(chunk).max()))
            conversion = sum(q * max(1.0, p) if src.subtype in ("FLOAT", "DOUBLE") else q
                             for q, p, src in zip(quanta, peaks, [sfx] + sources))
            dither = (len(sources) + 1) * 2 ** -23
            # Contract: N cue contributions under the same linear master gain as the bus.
            # N-1 additions, N+1 gain products and N+1 independent dither additions give
            # 3N+1 float32 operations; a full epsilon and sum-of-peaks bound each operation.
            arithmetic = (3 * len(sources) + 1) * np.finfo(np.float32).eps * max(1.0, sum(peaks) + conversion + dither)
            budget = dict(conversion_bound=conversion, independent_tpdf_bound=dither,
                          float32_arithmetic_bound=float(arithmetic))
            bound = sum(budget.values())
            error = float(np.abs(actual - total).max())
            maximum, largest_bound, checked = max(maximum, error), max(largest_bound, bound), checked + b - a
            if error > bound:
                return dict(base, maximum_error=maximum, reconstruction_bound=bound, error_budget=budget,
                            samples_checked=checked, first_failing_chunk_sample=a,
                            reason="named cue sum does not reconstruct delivered SFX")
        return dict(base, ok=True, maximum_error=maximum, reconstruction_bound=largest_bound,
                    samples_checked=checked, last_chunk_error_budget=budget)
    except (OSError, ValueError, RuntimeError) as exc:
        return dict(base, reason=str(exc))


def step_source_contract(source_root=MUSIC):
    """Pin module bytes across Python versions and normalized cue metadata; no renderer imports."""
    measured = {}
    try:
        for relative in STEP_CONTRACT:
            path = Path(source_root) / relative
            if path.suffix == ".py":
                canonical = path.read_bytes()
            else:
                canonical = json.dumps(json.loads(path.read_text()), sort_keys=True, separators=(",", ":")).encode()
            measured[relative] = hashlib.sha256(canonical).hexdigest()
    except (OSError, ValueError) as exc:
        return dict(check="score_step_source_contract", ok=False, reason=str(exc))
    return dict(check="score_step_source_contract", ok=bool(STEP_CONTRACT) and measured == STEP_CONTRACT,
                fingerprints=measured, reason="Source or cue changes require reviewed step-contract update.")


def expected_score_steps(barmap):
    """K.walk's two-beat loop at 20 frames/beat, on the reviewed current source contract."""
    if not isinstance(barmap, dict) or barmap.get("fps") != FPS or barmap.get("bpm") != 72:
        raise ValueError("step schedule requires current 24 fps / 72 BPM barmap")
    sync = barmap.get("sync", [])
    if not isinstance(sync, list):
        raise ValueError("invalid step-anchor registry")
    frames = {}
    for name in ("lantern", "narrowest", "watchfire_3", "breath_dawn"):
        matches = [row.get("f") for row in sync if isinstance(row, dict) and row.get("id") == name]
        if len(matches) != 1 or not _integer(matches[0]) or not 0 <= matches[0] < FRAMES:
            raise ValueError(f"step anchor {name} missing, duplicate, or invalid")
        frames[name] = int(matches[0])
    if not frames["lantern"] < frames["narrowest"] <= frames["watchfire_3"] < frames["breath_dawn"]:
        raise ValueError("invalid ordered walk intervals")
    beats = list(range(frames["lantern"], frames["narrowest"], 40))
    beats += list(range(frames["watchfire_3"], frames["breath_dawn"], 40))
    return [dict(id=f"feet.{frame}", f=frame) for frame in beats]


def derived_score_steps(cut, music_root, barmap_path):
    """Build note metadata in a fresh interpreter; never render or import sample/renderer code.

    AP2 keeps muted A notes for RNG continuity and adds feet_go. Reading the built notes
    handles that schedule without imposing A's 40-frame interval on the new first step.
    The subprocess also prevents an already-imported A module from another tree leaking in.
    The one reviewed sample-dependent helper, K.anticipate, changes only attack kwargs on
    non-feet parts; it is byte-pinned and skipped. This is scheduled-note metadata, never
    a claim about rendered anticipation or sample onsets.
    """
    root = Path(music_root).resolve()
    check = dict(check="score_step_source_contract", ok=False, cut=cut,
                 method="selected builder's feet/feet_* scheduled-note metadata; pinned non-feet anticipation skipped")
    script = r'''
import ast, hashlib, importlib, importlib.abc, json, pathlib, sys
root, cut, path = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
paths = [root / "src" / name for name in ("score_v3_A.py", "kit_v3.py", "dsl.py", "timeline_v3.py")]
if cut != "A":
    paths.append(root / "src" / ("score_v3_" + cut + ".py"))
paths += [path, root / "v3" / ("cues_" + cut + ".json")]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
class NoAudioImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sampler", "sampler_v2", "soundfile", "render_v3", "sound_v3", "sfx_v3"}:
            raise ValueError("audio module forbidden in scheduled-note metadata adapter: " + fullname)
sys.meta_path.insert(0, NoAudioImports())
sys.path.insert(0, str(root / "src"))
import timeline_v3 as T
import kit_v3 as K
if hasattr(K, "anticipate"):
    text = (root / "src/kit_v3.py").read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "anticipate")
    fingerprint = hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()
    if fingerprint != "601fc1bd19ba683bf322d1d60715f7eb4f3c9496c3fbbddbc0cb616c00234e23":
        raise ValueError("sample-dependent anticipation helper changed; review the metadata adapter")
    def skip_nonfeet_anticipation(score, part, *args, **kwargs):
        if part == "feet" or part.startswith("feet_"):
            raise ValueError("anticipation now touches the feet; scheduled-note adapter needs review")
    K.anticipate = skip_nonfeet_anticipation
mod = importlib.import_module("score_v3_" + cut)
bm = T.BarMap(cut, path=str(path), cues=str(root / "v3" / ("cues_" + cut + ".json")))
score = mod.build(bm)
parts = {name: part for name, part in score.parts.items() if name == "feet" or name.startswith("feet_")}
if "feet" not in parts or bm.frames != 6480 or T.FPS != 24:
    raise ValueError("missing feet part or incompatible film time grid")
notes = []
for name, part in parts.items():
    if part.inst != "giant_hand":
        raise ValueError("feet part instrument changed; review the step adapter")
    for note in part.notes:
        if note.pitch is None or note.gain_db <= -120.0 or note.vel <= 0:
            continue
        frame = note.start * T.BEAT_S * T.FPS
        if abs(frame - round(frame)) > 1e-7 or not 0 <= frame < 6480:
            raise ValueError("step is off the integral frame grid")
        frame = int(round(frame))
        notes.append(dict(id="feet." + str(frame), f=frame, part=name,
                          note_gain_db=note.gain_db, note_velocity=note.vel))
notes.sort(key=lambda note: (note["f"], note["part"]))
if not notes or len({n["id"] for n in notes}) != len(notes):
    raise ValueError("empty or duplicate feet note schedule")
if any(name in sys.modules for name in ("sampler", "render_v3", "sound_v3")):
    raise ValueError("score metadata build imported audio-render code; review before running audit")
fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if fingerprints != before:
    raise ValueError("score source changed during metadata build; run again on a stable tree")
print(json.dumps(dict(events=notes, fingerprints=fingerprints)))
'''
    try:
        if cut not in ("A", "AP2"):
            raise ValueError("unsupported score cut")
        # -B alone still reads stale timestamp-based bytecode. A fresh prefix makes
        # imports compile the exact source bytes fingerprinted above; -B prevents writes.
        with tempfile.TemporaryDirectory(prefix="syncA_score_metadata_") as cache_prefix:
            result = subprocess.run([sys.executable, "-B", "-X", "pycache_prefix=" + cache_prefix,
                                     "-c", script, str(root), cut, str(Path(barmap_path).resolve())],
                                    capture_output=True, text=True, timeout=60, check=True)
        data = json.loads(result.stdout)
        events = data["events"]
        if (not isinstance(events, list) or not events or
                any(not isinstance(e, dict) or not _integer(e.get("f")) or
                    e.get("id") != f"feet.{e['f']}" for e in events)):
            raise ValueError("invalid score-builder note metadata")
        return dict(check, ok=True, fingerprints=data["fingerprints"], expected_step_count=len(events)), events
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        return dict(check, reason=detail), []


def check_score_steps(barmap, picture, score=None, sfx=None, mix=None, *, step_contract=None, expected_steps=None):
    contract = step_source_contract() if step_contract is None else step_contract
    checks = [contract]
    try:
        expected = expected_score_steps(barmap) if expected_steps is None else expected_steps
        if not expected:
            raise ValueError("expected score note schedule unavailable")
    except ValueError as exc:
        return checks + [dict(check="score_steps_coverage", ok=False, status="UNRESOLVED", reason=str(exc))]
    audit = picture.get("audit", {}) if isinstance(picture, dict) else {}
    specs = audit.get("score_steps", []) if isinstance(audit, dict) else []
    registry, duplicates = {}, set()
    ids = {event["id"] for event in expected}
    if not isinstance(specs, list):
        specs = []
        checks.append(dict(check="score_steps_registry", ok=False, reason="audit.score_steps must be a list"))
    for spec in specs:
        eid = spec.get("id") if isinstance(spec, dict) else None
        if not isinstance(eid, str) or eid not in ids:
            checks.append(dict(check="score_steps_registry", ok=False, reason=f"unknown step id {eid!r}"))
        elif eid in registry:
            duplicates.add(eid)
        else:
            registry[eid] = spec
    for event in expected:
        spec = registry.get(event["id"], {})
        row = dict(check="score_step", id=event["id"], table_frame=event["f"],
                   measured_picture_frame=None, stem_onset_frame=None, mix_onset_frame=None,
                   offset_frames=None, status="UNRESOLVED", ok=False)
        contact = spec.get("measured_picture_frame")
        if not contract["ok"]:
            row["reason"] = "walk source contract changed; expected schedule needs review"
        elif spec.get("visible_contact") is False or spec.get("status") == "NO_VISIBLE_CONTACT":
            row.update(status="NO_VISIBLE_CONTACT", reason=spec.get("contact_evidence") or "no measured visible contact")
        elif (event["id"] in duplicates or spec.get("frame") != event["f"] or spec.get("status") != "measured" or
              not _integer(contact) or not 0 <= contact < FRAMES or spec.get("visible_contact") is not True or
              not isinstance(spec.get("contact_evidence"), str) or not spec["contact_evidence"].strip()):
            row["reason"] = "each scheduled step requires its own measured visible foot contact"
        elif not isinstance(spec.get("audio"), dict) or score is None or sfx is None or mix is None:
            row.update(measured_picture_frame=int(contact), reason="step waveform detector or actual audio unavailable")
        else:
            audio = dict(spec["audio"], kind="hard", picture_frame=int(contact))
            row = measure_event(dict(id=event["id"], t=event["f"] / FPS), audio, score, sfx, mix=mix)
            row.update(check="score_step", measured_picture_frame=int(contact), contact_evidence=spec["contact_evidence"],
                       contribution="score band", masker="SFX band", attribution="aggregate score step detector")
        checks.append(row)
    return checks


def band_envelope(source, start, stop, band):
    """Causal energy with filter warm-up, retaining only 1 ms envelope samples."""
    warm_start = max(0, start - SR)
    sos = butter(4, band, btype="bandpass", fs=SR, output="sos")
    state = np.zeros((len(sos), 2, 2), np.float64)
    history = np.zeros(WINDOW - 1, np.float64)
    positions, powers = [], []
    for a in range(warm_start, stop, BLOCK):
        b = min(a + BLOCK, stop)
        y, state = sosfilt(sos, read_chunk(source, a, b), axis=0, zi=state)
        energy = np.mean(y * y, axis=1)
        combined = np.concatenate((history, energy))
        accum = np.concatenate(([0.0], np.cumsum(combined)))
        mean = (accum[WINDOW:] - accum[:-WINDOW]) / WINDOW
        history = combined[-(WINDOW - 1):]
        # Global hop alignment makes independent source envelopes directly comparable.
        first = ((max(a, start) + HOP - 1) // HOP) * HOP
        idx = np.arange(first, b, HOP, dtype=np.int64)
        if len(idx):
            positions.append(idx)
            powers.append(mean[idx - a])
    if not positions:
        raise InvalidAudio("empty envelope interval")
    power = np.concatenate(powers)
    if not np.isfinite(power).all():
        raise InvalidAudio("non-finite filtered energy")
    return np.concatenate(positions), 10 * np.log10(np.maximum(power, 1e-30))


def _spec_problem(spec):
    if spec.get("kind") not in ("hard", "soft", "bed"):
        return "kind must be hard, soft, or bed"
    pf = spec.get("picture_frame")
    if pf is not None and (not _integer(pf) or not 0 <= pf < FRAMES):
        return "picture_frame is invalid"
    for field in ("search_frames", "control_frames"):
        pair = spec.get(field)
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(_finite_number(x) for x in pair):
            return f"{field} must contain two finite frame bounds"
        if not 0 <= pair[0] < pair[1] <= FRAMES:
            return f"invalid {field} interval"
    if spec["control_frames"][1] != spec["search_frames"][0]:
        return "control must immediately precede search"
    if pf is not None and not spec["search_frames"][0] <= pf < spec["search_frames"][1]:
        return "picture_frame lies outside search"
    band = spec.get("band_hz")
    if not isinstance(band, (list, tuple)) or len(band) != 2 or not all(_finite_number(x) for x in band):
        return "band_hz must contain two finite frequencies"
    if not 0 < band[0] < band[1] < SR / 2:
        return "invalid band_hz"
    return None


def table_row(event, spec=None):
    t = event.get("t", event.get("t0"))
    frame = float(t) * FPS if _finite_number(t) else None
    frame = frame if _finite_number(frame) else None
    return dict(id=event.get("id"), kind=None if spec is None else spec.get("kind"),
                table_frame=frame,
                picture_frame=None, stem_onset_frame=None, mix_onset_frame=None,
                offset_frames=None, stem_offset_frames=None, status="UNRESOLVED", ok=False,
                method=METHOD, table_time_role="metadata only; never a measured onset")


def measure_event(event, spec, contribution, score, *, mix=None, isolated_cue=False):
    spec = dict(spec)
    if "kind" not in spec:
        spec["kind"] = "bed" if event.get("kind") == "bed" else "hard"
    row = table_row(event, spec)
    problem = _spec_problem(spec)
    if problem:
        return dict(row, reason=problem)
    if row["table_frame"] is None or not np.isfinite(row["table_frame"]):
        return dict(row, reason="event table time is missing or non-finite")
    if mix is None:
        return dict(row, reason="actual delivered mix is required; stem ratios alone cannot detect cancellation")
    picture_frame = spec.get("picture_frame")
    row.update(picture_frame=int(picture_frame) if picture_frame is not None else None, band_hz=spec["band_hz"],
               search_frames=spec["search_frames"], control_frames=spec["control_frames"],
               attribution="postmaster cue supplied by manifest" if isolated_cue else "aggregate SFX band",
               limitation="Numerical masking proxy, not listening. " +
                          ("Cue provenance is supplied, not independently certified." if isolated_cue else
                           "Aggregate SFX cannot identify an overlapping cue's contribution."))
    start, split, stop = (round(float(f) * FR) for f in
                          (spec["control_frames"][0], spec["search_frames"][0], spec["search_frames"][1]))
    try:
        samples, env = band_envelope(contribution, start, stop, spec["band_hz"])
        # An isolated cue can be buried by another effect even when the score is silent.
        # Subtraction is justified only after the cue bank has reconstructed this SFX bus.
        masker = DifferenceAudio(mix, contribution) if isolated_cue else score
        score_samples, score_env = band_envelope(masker, start, stop, spec["band_hz"])
        mix_samples, mix_env = band_envelope(mix, start, stop, spec["band_hz"])
        if not np.array_equal(samples, score_samples) or not np.array_equal(samples, mix_samples):
            raise InvalidAudio("score and contribution envelope grids differ")
        search, control = samples >= split, samples < split
        if not search.any() or not control.any():
            raise InvalidAudio("search or control has no measured windows")
        peak, pre = float(env[search].max()), float(env[control].max())
        threshold = max(FLOOR_DB, peak - UNDER_PEAK_DB)
        row.update(search_peak_dbfs=peak, control_peak_dbfs=pre, threshold_dbfs=threshold,
                   control_margin_db=threshold - pre, floor_dbfs=FLOOR_DB,
                   mix_retention_db=RETENTION_DB,
                   masker="actual mix minus reconciled cue" if isolated_cue else "other aggregate stem",
                   mix_proxy="contribution >= threshold, > -60 dBFS, > masker - 12 dB; actual mix >= threshold, > -60 dBFS, >= contribution - 6 dB")
        if peak <= FLOOR_DB:
            return dict(row, status="SILENT", reason="no contribution above the declared measurement floor")
        if pre > threshold - CONTROL_MARGIN_DB:
            return dict(row, status="NOT_ISOLATED", reason="neighboring negative control lacks 10 dB separation")
        above = search & (env >= threshold)
        hit = np.flatnonzero(above)
        if not len(hit):
            return dict(row, status="UNRESOLVED", reason="no threshold crossing")
        onset = float(samples[hit[0]] / FR)
        row.update(stem_onset_frame=onset,
                   stem_offset_frames=onset - row["picture_frame"] if picture_frame is not None else None)
        audible = np.flatnonzero(above & (env > FLOOR_DB) & (env > score_env - MASK_DB) &
                                 (mix_env >= threshold) & (mix_env > FLOOR_DB) &
                                 (mix_env >= env - RETENTION_DB))
        if not len(audible):
            return dict(row, status="MASKED_PROXY", reason="no window meets the numerical mix masking proxy")
        arrival = float(samples[audible[0]] / FR)
        row.update(actual_mix_dbfs_at_proxy_onset=float(mix_env[audible[0]]),
                   contribution_dbfs_at_proxy_onset=float(env[audible[0]]))
        if picture_frame is None:
            return dict(row, mix_onset_frame=arrival, status="UNRESOLVED",
                        reason="audio onset measured; picture_frame remains unresolved so timing cannot pass")
        delta = arrival - row["picture_frame"]
        tolerance = (-0.5, 1.0) if spec["kind"] == "hard" else (-3.0, 3.0)
        ok = tolerance[0] <= delta <= tolerance[1]
        attributed = isolated_cue or (spec.get("detector_validated") is True and
                                     isinstance(spec.get("detector_evidence"), str) and
                                     bool(spec["detector_evidence"].strip()))
        if not attributed:
            return dict(row, mix_onset_frame=arrival, offset_frames=delta,
                        status="UNRESOLVED", reason="energy crossing lacks validated event attribution")
        if not isolated_cue:
            recorded = spec.get("detector_audio_sha256")
            if not isinstance(recorded, dict) or any(
                    not isinstance(recorded.get(key), str) or len(recorded[key]) != 64 or
                    any(c not in "0123456789abcdef" for c in recorded[key]) for key in ("contribution", "mix")):
                return dict(row, mix_onset_frame=arrival, offset_frames=delta, status="UNRESOLVED",
                            reason="aggregate attribution lacks exact contribution/mix full-file SHA256 fingerprints")
            if not callable(getattr(contribution, "file_sha256", None)) or not callable(getattr(mix, "file_sha256", None)):
                return dict(row, mix_onset_frame=arrival, offset_frames=delta, status="UNRESOLVED",
                            reason="current audio file fingerprints unavailable")
            current = dict(contribution=contribution.file_sha256(), mix=mix.file_sha256())
            row["current_audio_sha256"] = current
            if any(recorded[key] != current[key] for key in current):
                return dict(row, mix_onset_frame=arrival, offset_frames=delta, status="UNRESOLVED",
                            reason="aggregate attribution fingerprints do not match the current audio files")
        explanation = spec.get("timing_exception")
        explained = spec["kind"] != "hard" and isinstance(explanation, str) and bool(explanation.strip())
        return dict(row, mix_onset_frame=arrival, offset_frames=delta, tolerance_frames=list(tolerance),
                    ideal=0 <= delta <= 1 if spec["kind"] == "hard" else ok,
                    status="PASS_PROXY" if ok else "EXPLAINED_PROXY" if explained else "OFF",
                    timing_exception=explanation if explained else None, ok=bool(ok or explained))
    except (OSError, InvalidAudio, ValueError, RuntimeError) as exc:
        return dict(row, status="UNRESOLVED", reason=str(exc))


def audit_events(events, picture, sfx, score, cue_loader=None, *, mix=None):
    """Total coverage: retain every source row, including duplicates and absent picture records."""
    checks, rows = [], []
    if not isinstance(events, list):
        return [], [dict(check="event_registry", ok=False, error="events must be a list")]
    audit = picture.get("audit", {}) if isinstance(picture, dict) else {}
    specs = audit.get("events", []) if isinstance(audit, dict) else []
    if not isinstance(specs, list):
        specs = []
        checks.append(dict(check="picture_registry", ok=False, error="audit.events must be a list"))
    event_ids = [e.get("id") for e in events if isinstance(e, dict)]
    registry = {}
    duplicates = set()
    for spec in specs:
        eid = spec.get("id") if isinstance(spec, dict) else None
        if not isinstance(eid, str) or not eid or eid not in event_ids:
            checks.append(dict(check="picture_registry", ok=False, error=f"unknown/invalid picture event {eid!r}"))
            continue
        if eid in registry:
            duplicates.add(eid)
            checks.append(dict(check="picture_registry", ok=False, error=f"duplicate picture event {eid}"))
        registry[eid] = spec
    for event in events:
        if not isinstance(event, dict):
            rows.append(dict(table_row({}), reason="invalid event table row"))
            continue
        eid = event.get("id")
        if not isinstance(eid, str) or not eid or event_ids.count(eid) != 1 or eid in duplicates:
            rows.append(dict(table_row(event), reason="missing or duplicate event id"))
        elif eid not in registry:
            rows.append(dict(table_row(event), reason="no matching audit.events picture measurement"))
        else:
            spec = registry[eid]
            if cue_loader is None:
                rows.append(measure_event(event, spec, sfx, score, mix=mix))
            else:
                try:
                    with cue_loader(eid) as cue:
                        rows.append(measure_event(event, spec, cue, score, mix=mix, isolated_cue=True))
                except (OSError, ValueError, RuntimeError, KeyError) as exc:
                    rows.append(dict(table_row(event, spec), reason=f"cue unavailable: {exc}"))
    if not events:
        checks.append(dict(check="event_registry", ok=False, error="empty event table"))
    checks.append(dict(check="event_coverage", ok=bool(events) and len(rows) == len(events),
                       table_rows=len(events), report_rows=len(rows), picture_rows=len(specs)))
    return rows, checks


def check_score_sync(barmap, picture, score=None, sfx=None, mix=None):
    """Picture coverage is independent of SFX detection and may not silently shrink."""
    sync = barmap.get("sync", []) if isinstance(barmap, dict) else []
    audit = picture.get("audit", {}) if isinstance(picture, dict) else {}
    specs = audit.get("score_sync", []) if isinstance(audit, dict) else []
    if not isinstance(sync, list) or not sync or not isinstance(specs, list):
        return [dict(check="score_sync_coverage", ok=False, status="UNRESOLVED", reason="invalid/empty score registry")]
    checks, registry = [], {}
    ids = [s.get("id") for s in sync if isinstance(s, dict)]
    duplicates = set()
    for s in specs:
        eid = s.get("id") if isinstance(s, dict) else None
        if not isinstance(eid, str) or eid not in ids:
            checks.append(dict(check="score_sync_registry", ok=False, reason=f"unknown score sync id {eid!r}"))
        elif eid in registry:
            duplicates.add(eid)
            checks.append(dict(check="score_sync_registry", ok=False, reason=f"duplicate score sync id {eid}"))
        else:
            registry[eid] = s
    for sync_row in sync:
        event = sync_row if isinstance(sync_row, dict) else {}
        eid, frame = event.get("id"), event.get("f")
        spec = registry.get(eid, {}) if isinstance(eid, str) else {}
        measured = spec.get("measured_picture_frame")
        result = dict(check="score_sync_picture", id=eid, table_frame=frame if _finite_number(frame) else None,
                      measured_picture_frame=measured if _finite_number(measured) else None,
                      offset_frames=None, ok=False, status="UNRESOLVED",
                      method="barmap metadata versus measured picture; rendered score arrival not measured")
        kind = spec.get("kind")
        reference = (kind != "hard" and spec.get("role") == "reference" and
                     isinstance(spec.get("reference_reason"), str) and bool(spec["reference_reason"].strip()))
        if (isinstance(eid, str) and eid not in duplicates and ids.count(eid) == 1 and _integer(frame) and 0 <= frame < FRAMES
                and spec.get("status") == "measured" and spec.get("frame") == frame
                and _integer(measured) and 0 <= measured < FRAMES and (kind in ("hard", "soft") or reference)):
            delta = frame - measured
            result.update(offset_frames=delta)
            if reference:
                result.update(ok=True, status="REFERENCE_ONLY", reason=spec["reference_reason"])
            elif isinstance(spec.get("audio"), dict) and score is not None and sfx is not None:
                audio_spec = dict(spec["audio"], kind=kind, picture_frame=measured)
                measured_audio = measure_event(dict(id=eid, t=frame / FPS), audio_spec, score, sfx, mix=mix)
                measured_audio.update(check="score_sync_audio", masker="SFX band", contribution="score band",
                                      attribution="aggregate score band",
                                      limitation="Numerical masking proxy, not listening; aggregate score attribution requires detector evidence.",
                                      mix_proxy="score >= threshold, > -60 dBFS, > SFX - 12 dB; actual mix >= threshold, > -60 dBFS, >= score - 6 dB")
                result = measured_audio
            else:
                result.update(status="SCORE_AUDIO_UNRESOLVED", reason="rendered score waveform detector absent or audio unavailable")
        checks.append(result)
    return checks


def picture_for_cut(picture, cut):
    """Choose score evidence without changing the shared A.* effect measurements.

    Legacy score_sync/score_steps belong to A. AP2 must provide both explicit arrays;
    inheriting A's approvals would certify notes that AP2 muted and miss its first step.
    """
    selected = dict(picture) if isinstance(picture, dict) else {}
    audit = dict(selected.get("audit", {})) if isinstance(selected.get("audit", {}), dict) else {}
    selected["audit"] = audit
    missing, sources = [], {}
    for key in ("score_sync", "score_steps"):
        choices = audit.get(key + "_by_cut", {})
        if isinstance(choices, dict) and isinstance(choices.get(cut), list):
            audit[key] = choices[cut]
            sources[key] = key + "_by_cut." + cut
        elif cut == "A" and isinstance(choices, dict) and "A" not in choices:
            sources[key] = key + " (legacy A)"
        else:
            audit[key] = []
            missing.append(key + "_by_cut." + cut)
    return selected, dict(check="score_evidence_cut", ok=not missing, selected_cut=cut,
                          evidence_sources=sources, missing_or_invalid=missing)


def run(mix, sfx, score, picture, events, cue_dir=None, barmap=MUSIC / "v3/barmap_A.json", *,
        solo_dir=None, cut=None, music_root=MUSIC):
    from contextlib import contextmanager
    checks, rows = [], []
    metadata = {}
    for key, path, empty in (("events", events, []), ("picture", picture, {}), ("barmap", barmap, {})):
        try:
            metadata[key] = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            metadata[key] = empty
            checks.append(dict(check="input_metadata", input=key, ok=False, error=str(exc)))
    event_data, picture_data, barmap_data = (metadata[key] for key in ("events", "picture", "barmap"))
    step_kwargs = {}
    if cut is not None:
        tag_ok = isinstance(barmap_data, dict) and barmap_data.get("cut") == cut
        checks.append(dict(check="score_cut_metadata", ok=tag_ok, selected_cut=cut,
                           barmap_cut=barmap_data.get("cut") if isinstance(barmap_data, dict) else None,
                           events_path=str(events), barmap_path=str(barmap)))
        # Explicit paths are useful for an integrated tree, but a canonical filename
        # naming the other cut is positive evidence of a mismatched input.
        other = "A" if cut == "AP2" else "AP2"
        wrong_names = {f"sound_{other}.wav", f"sound_{other}_sfx.wav", f"sound_{other}_score.wav",
                       f"events_{other}.json", f"barmap_{other}.json"}
        mismatches = [str(path) for path in (mix, sfx, score, events, barmap) if Path(path).name in wrong_names]
        checks.append(dict(check="cut_input_names", ok=not mismatches, selected_cut=cut, mismatches=mismatches))
        contract, steps = derived_score_steps(cut, music_root, barmap)
        step_kwargs = dict(step_contract=contract, expected_steps=steps)
        picture_data, evidence_check = picture_for_cut(picture_data, cut)
        checks.append(evidence_check)
    if cue_dir is not None and solo_dir is not None:
        checks.append(dict(check="cue_input", ok=False, reason="choose cue-dir or solo-dir, not both"))
    with ExitStack() as stack:
        sources = {}
        for name, path in (("mix", mix), ("sfx", sfx), ("score", score)):
            try:
                src = WaveAudio(path)
                stack.callback(src.close)
                sources[name] = src
                format_check = dict(check_format(src, name), path=str(path))
                if format_check["ok"]:
                    format_check["sha256"] = src.file_sha256()
                checks.append(format_check)
            except (OSError, ValueError, RuntimeError) as exc:
                checks.append(dict(check="audio_input", name=name, path=str(path), ok=False, error=str(exc)))
        # Metadata failures remain red, but they must not suppress valid per-event measurements.
        audio_ok = len(sources) == 3 and all(check_format(source, name)["ok"] for name, source in sources.items())
        if audio_ok:
            checks.append(check_mix(sources["mix"], sources["sfx"], sources["score"]))
            loader = None
            if cue_dir is not None:
                try:
                    manifest = json.loads((Path(cue_dir) / "manifest.json").read_text())["events"]
                    if not isinstance(manifest, dict):
                        raise ValueError("cue manifest events must be an object")
                    unknown = set(manifest) - {e.get("id") for e in event_data if isinstance(e, dict)}
                    if unknown:
                        checks.append(dict(check="cue_manifest", ok=False, error=f"unknown cue ids: {sorted(unknown)}"))
                except (OSError, ValueError, KeyError) as exc:
                    manifest = {}
                    checks.append(dict(check="cue_manifest", ok=False, error=str(exc)))
                cue_sources = {}
                for eid, item in manifest.items():
                    try:
                        if not isinstance(item, dict):
                            raise InvalidAudio("cue manifest entry must be an object")
                        offset = item.get("offset_samples")
                        if not _integer(offset) or offset < 0 or not isinstance(item.get("path"), str):
                            raise InvalidAudio("invalid cue path or offset_samples")
                        cue = WaveAudio(Path(cue_dir) / item["path"])
                        stack.callback(cue.close)
                        if cue.samplerate != SR or cue.channels != 2 or cue.frames <= 0 or offset + cue.frames > SAMPLES:
                            raise InvalidAudio("cue must be 48 kHz stereo and fit inside the film")
                        cue_sources[eid] = OffsetAudio(cue, int(offset))
                    except (OSError, ValueError, RuntimeError) as exc:
                        checks.append(dict(check="cue_input", id=eid, ok=False, reason=str(exc)))
                expected_ids = [e.get("id") for e in event_data if isinstance(e, dict)] if isinstance(event_data, list) else []
                reconciled = check_cue_reconstruction(sources["sfx"], cue_sources, expected_ids)
                checks.append(reconciled)

                @contextmanager
                def loader(eid):
                    if not reconciled["ok"]:
                        raise InvalidAudio("cue contributions have not reconstructed the delivered SFX")
                    yield cue_sources[eid]
            elif solo_dir is not None:
                try:
                    from sync_a_solo import reconcile
                    expected_ids = [e.get("id") for e in event_data if isinstance(e, dict)] if isinstance(event_data, list) else []
                    reconciled, bank = reconcile(solo_dir, sources["sfx"], expected_ids)
                    checks.append(reconciled)
                    if bank is not None and callable(getattr(bank, "close", None)):
                        stack.callback(bank.close)
                except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
                    bank = None
                    reconciled = dict(check="solo_sum_equals_sfx", ok=False, reason=str(exc))
                    checks.append(reconciled)

                @contextmanager
                def loader(eid):
                    if not reconciled["ok"] or bank is None:
                        raise InvalidAudio("solo contributions have not reconstructed the delivered SFX")
                    with bank.open(eid) as cue:
                        yield cue
            rows, coverage = audit_events(event_data, picture_data, sources["sfx"], sources["score"], loader, mix=sources["mix"])
            checks.extend(coverage)
            checks.extend(check_score_sync(barmap_data, picture_data, sources["score"], sources["sfx"], sources["mix"]))
            checks.extend(check_score_steps(barmap_data, picture_data, sources["score"], sources["sfx"], sources["mix"], **step_kwargs))
        else:
            rows = [dict(table_row(e if isinstance(e, dict) else {}), reason="audio inputs missing or invalid")
                    for e in event_data] if isinstance(event_data, list) else []
            checks.extend(check_score_sync(barmap_data, picture_data))
            checks.extend(check_score_steps(barmap_data, picture_data, **step_kwargs))
    return dict(ok=bool(rows) and all(c["ok"] for c in checks) and all(r["ok"] for r in rows),
                fps=FPS, samplerate=SR, film_frames=FRAMES, cut=cut,
                judgment="Numerical sync/masking proxy only; not a listening assessment.", checks=checks, events=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cut", choices=("auto", "A", "AP2"), default="auto",
                        help="auto selects AP2 once sound_AP2.wav exists; missing AP2 siblings fail")
    parser.add_argument("--music-root", type=Path, default=MUSIC, help="selected tree's music directory")
    parser.add_argument("--mix", type=Path)
    parser.add_argument("--sfx", type=Path)
    parser.add_argument("--score", type=Path)
    parser.add_argument("--picture", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--barmap", type=Path)
    cues = parser.add_mutually_exclusive_group()
    cues.add_argument("--cue-dir", type=Path)
    cues.add_argument("--solo-dir", type=Path, help="owner NPZ solo bank; must reconstruct the selected SFX")
    parser.add_argument("--json", nargs="?", const="-", metavar="PATH", help="write JSON; bare --json prints it")
    args = parser.parse_args(argv)
    if args.cut == "auto":
        explicit_mix_cut = {"sound_A.wav": "A", "sound_AP2.wav": "AP2"}.get(args.mix.name) if args.mix else None
        cut = explicit_mix_cut or ("AP2" if (args.music_root / "out/v3/sound_AP2.wav").exists() else "A")
    else:
        cut = args.cut
    defaults = dict(mix=f"out/v3/sound_{cut}.wav", sfx=f"out/v3/sound_{cut}_sfx.wav",
                    score=f"out/v3/sound_{cut}_score.wav", picture="sound/picture_sync_A.json",
                    events=f"sound/events_{cut}.json", barmap=f"v3/barmap_{cut}.json")
    paths = {name: getattr(args, name) or args.music_root / relative for name, relative in defaults.items()}
    report = run(**paths, cue_dir=args.cue_dir, solo_dir=args.solo_dir, cut=cut, music_root=args.music_root)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.json == "-":
        print(encoded)
    else:
        for row in report["events"]:
            print(f"{row['status']:14s} {str(row['id']):20s} table={row['table_frame']} "
                  f"stem={row['stem_onset_frame']} mix_proxy={row['mix_onset_frame']} "
                  f"picture={row['picture_frame']} offset={row['offset_frames']}")
        for check in report["checks"]:
            if not check["ok"]:
                print("FAIL " + json.dumps(check, allow_nan=False))
        print("PASS (numerical proxy only)" if report["ok"] else "FAIL: unresolved or out-of-tolerance checks")
        if args.json:
            Path(args.json).write_text(encoded + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
