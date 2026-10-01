"""Bounded-memory numerical checks of exported D audio, independent of renderers.

Loudness follows pyloudnorm's default K weighting and BS.1770 block gates;
short-term windows and section masks follow analyze_v3. True peak follows the
production engine's 4x polyphase method. These checks do not measure picture sync
or replace listening. Run whole-film verification through the owner's onepy guard.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import wave

import numpy as np
import soundfile as sf
from scipy import signal
from pyloudnorm.iirfilter import IIRfilter

SR, FPS, FRAME_N = 48000, 24, 2000
FRAMES, SAMPLE_FRAMES, PREFIX_FRAMES = 9200, 18_400_000, 1440
CHUNK = 8 * SR
TARGET_LUFS, LUFS_TOLERANCE, TRUE_PEAK_LIMIT = -16.0, .1, -1.0
ZERO_WINDOWS = ((3400, 3440, "vision vacuum"), (8660, 8740, "voice void"))
SCORE_TAIL = (9120, 9200, "score tail")
VOICE_WINDOW = (8640, 8880)


def _audio(x):
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2 or not 1 <= x.shape[1] <= 5:
        raise ValueError("audio must have one through five channels")
    return x


def lufs(x):
    """Default pyloudnorm integrated LUFS, with persistent filters per chunk.

    Include the same rounded final block and full 0.4-second denominator as
    pyloudnorm. Keep at most one chunk plus a block of filtered audio in RAM.
    Filter state is continuous across chunks, but resets for each invocation.
    """
    x = _audio(x)
    block = .4
    if len(x) < block * SR:
        raise ValueError("audio must be at least 0.4 seconds long")
    stages = (IIRfilter(4., 1 / np.sqrt(2), 1500., SR, "high_shelf"),
              IIRfilter(0., .5, 38., SR, "high_pass"))
    states = [np.zeros((max(len(f.a), len(f.b)) - 1, x.shape[1])) for f in stages]
    count = int(np.round((len(x) / SR - block) / (block * .25)) + 1)
    energies = np.zeros((count, x.shape[1]), np.float64)
    carry, j = np.empty((0, x.shape[1])), 0
    for a in range(0, len(x), CHUNK):
        b = min(len(x), a + CHUNK)
        y = np.asarray(x[a:b], np.float64)
        if not np.isfinite(y).all():
            raise ValueError("nonfinite audio")
        for i, f in enumerate(stages):
            y, states[i] = signal.lfilter(f.b, f.a, y, axis=0, zi=states[i])
            y *= f.passband_gain
        start = a - len(carry)
        y = np.concatenate((carry, y), axis=0)
        while j < count:
            lo = int(block * (j * .25) * SR)
            hi = int(block * (j * .25 + 1) * SR)
            if hi > b and b != len(x):
                break
            window = y[lo - start:min(hi, b) - start]
            energies[j] = np.sum(window * window, axis=0) / (block * SR)
            j += 1
        carry = y[-(int(block * SR) + 2):].copy()
    weights = np.asarray([1., 1., 1., 1.41, 1.41])[:x.shape[1]]
    weighted = energies @ weights
    with np.errstate(divide="ignore"):
        levels = -.691 + 10 * np.log10(weighted)
    absolute = levels >= -70.
    if not np.any(absolute):
        return float("-inf")
    relative = -.691 + 10 * np.log10(np.mean(weighted[absolute])) - 10.
    admitted = (levels > relative) & (levels > -70.)
    if not np.any(admitted):
        return float("-inf")
    return float(-.691 + 10 * np.log10(np.mean(energies[admitted], axis=0) @ weights))


def true_peak_db(x, chunk=8 * SR, pad=256):
    """Engine 4x polyphase peak; halo removes artificial chunk-edge peaks."""
    if chunk <= 0 or pad < 0:
        raise ValueError("chunk must be positive and pad nonnegative")
    x, peak = _audio(x), 0.
    for a in range(0, len(x), chunk):
        b = min(len(x), a + chunk)
        a0, b0 = max(0, a - pad), min(len(x), b + pad)
        source = np.asarray(x[a0:b0], np.float64)
        if not np.isfinite(source).all():
            raise ValueError("nonfinite audio")
        y = signal.resample_poly(source, 4, 1, axis=0)
        local = np.abs(y).max(axis=1).reshape(-1, 4).max(axis=1).astype(np.float32)
        peak = max(peak, float(local[a-a0:a-a0+b-a].max()))
    return float(20 * np.log10(peak + 1e-12))


def st_loudness(x, win=3., hop=.5):
    """Exact analyze_v3 windows, threshold, finite fallback, and time centers."""
    if win < .4 or hop <= 0:
        raise ValueError("window must be at least .4 seconds and hop positive")
    out, t = [], 0.
    while t + win <= len(x) / SR + 1e-6:
        seg = np.asarray(x[int(t * SR):int((t + win) * SR)], np.float64)
        if not np.isfinite(seg).all():
            raise ValueError("nonfinite audio")
        v = lufs(seg) if np.abs(seg).max() > 1e-6 else -120.
        out.append((t + win / 2, v if np.isfinite(v) else -120.))
        t += hop
    return np.asarray(out) if out else np.zeros((0, 2))


def level_rows(track, sections, bands, hard_silences=()):
    """Engine section masks, with explicitly opted-in score silence containment.

    Missing/duplicate sections or bands fail; none are silently dropped. A score
    D34 caller may pass ((9120,9200),), retaining the legacy window evidence.
    No silence is inferred from a section name or from a quiet score stem.
    """
    rows, band_by_id, seen = [], {}, set()
    for sid, lo, hi in bands:
        if sid in band_by_id:
            rows.append(dict(check="level band", section=sid, ok=False, reason="duplicate band"))
        band_by_id[sid] = (float(lo), float(hi))
    valid = track.ndim == 2 and track.shape[1] == 2 and len(track) and np.isfinite(track).all()
    anchor = float(track[:, 1].max()) if valid else None
    for sec in sections:
        sid = sec["id"]
        if sid in seen:
            rows.append(dict(check="level band", section=sid, ok=False, reason="duplicate section"))
            continue
        seen.add(sid)
        if sid not in band_by_id:
            rows.append(dict(check="level band", section=sid, ok=False, reason="missing band"))
            continue
        lo, hi = band_by_id[sid]
        if not np.isfinite([lo, hi]).all() or lo > hi:
            rows.append(dict(check="level band", section=sid, ok=False, reason="invalid band"))
            continue
        if anchor is None:
            rows.append(dict(check="level band", section=sid, ok=False, reason="no finite windows measured"))
            continue
        t0 = sec["t0"] if "t0" in sec else sec["f0"] / FPS
        t1 = sec["t1"] if "t1" in sec else sec["f1"] / FPS
        mask = (track[:, 0] >= t0 + 1.) & (track[:, 0] <= t1 - 1.)
        if not np.any(mask):
            mask = np.abs(track[:, 0] - (t0+t1)/2) <= 1.6
        policy, legacy = "engine_1s_interior", None

        def metrics(m):
            vmax, vmed = float(track[m, 1].max()), float(np.median(track[m, 1]))
            return dict(windows=int(m.sum()), max_lufs=vmax, median_lufs=vmed,
                        relative_max_lu=vmax-anchor, relative_median_lu=vmed-anchor,
                        ok=bool(vmax-anchor <= hi+.5 and vmed-anchor >= lo-.5))

        if any(a <= sec.get("f0", t0*FPS) and sec.get("f1", t1*FPS) <= b for a, b in hard_silences):
            policy = "fully_contained_in_declared_silence"
            legacy = metrics(mask) if np.any(mask) else None
            mask = (track[:, 0] >= t0 + 1.5) & (track[:, 0] <= t1 - 1.5)
        row = dict(check="level band", section=sid, anchor_lufs=anchor, band_lu=[lo, hi],
                   tolerance_lu=.5, window_policy=policy)
        if policy != "engine_1s_interior":
            row["legacy_engine_window"] = legacy
        if np.any(mask):
            row.update(metrics(mask))
        else:
            row.update(ok=False, reason="no windows measured")
        rows.append(row)
    for sid in band_by_id.keys() - seen:
        rows.append(dict(check="level band", section=sid, ok=False, reason="missing section"))
    if not rows:
        rows.append(dict(check="level band", ok=False, reason="empty level map"))
    return rows


def _stats(x):
    finite, peak = True, 0.
    for a in range(0, len(x), CHUNK):
        seg = x[a:a+CHUNK]
        if not np.isfinite(seg).all():
            finite = False
        else:
            peak = max(peak, float(np.abs(seg).max()))
    return finite, peak if finite else None


def silence_row(x, f0, f1, *, first_frame=0, label="silence"):
    a, b = (f0-first_frame)*FRAME_N, (f1-first_frame)*FRAME_N
    row = dict(check="silence", label=label, frames=[f0, f1])
    if a < 0 or b > len(x) or b <= a:
        return dict(row, ok=False, reason="missing full window")
    finite, peak = _stats(x[a:b])
    return dict(row, ok=finite and peak == 0., exact_zero=finite and peak == 0.,
                peak_dbfs=None if peak is None else float(20*np.log10(peak+1e-12)))


def constraints(x, kind, *, first_frame=0):
    """Sample-domain checks; excerpt calls report uncovered windows as failures."""
    kind = kind.lower()
    if kind not in ("sfx", "mix", "score"):
        raise ValueError("kind must be sfx, mix, or score")
    finite, peak = _stats(x)
    rows = [dict(check="finite", ok=finite),
            dict(check="clipping", ok=finite and peak < .999, peak=peak)]
    windows = ZERO_WINDOWS + ((SCORE_TAIL,) if kind == "score" else ())
    rows.extend(silence_row(x, a, b, first_frame=first_frame, label=label) for a, b, label in windows)
    if kind == "sfx":
        a, b = ((f-first_frame)*FRAME_N for f in VOICE_WINDOW)
        row = dict(check="voice effects ceiling", frames=list(VOICE_WINDOW), limit_dbfs=-40.)
        if a < 0 or b > len(x) or b <= a:
            row.update(ok=False, reason="missing full window")
        else:
            valid, voice_peak = _stats(x[a:b])
            row.update(ok=valid and voice_peak <= 10**(-40/20),
                       peak_dbfs=None if voice_peak is None else float(20*np.log10(voice_peak+1e-12)))
        rows.append(row)
    return [dict(row, kind=kind) for row in rows]


def pcm_prefix_sha256(path, frames=PREFIX_FRAMES):
    """Identity of packed PCM bytes, independent of WAV metadata chunks."""
    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (SR, 2, 3):
            raise ValueError("opening identity requires 48 kHz stereo 24-bit PCM")
        remaining, digest = frames*FRAME_N, hashlib.sha256()
        while remaining:
            take = min(remaining, CHUNK)
            data = handle.readframes(take)
            if len(data) != take*6:
                raise ValueError("WAV is shorter than the opening window")
            digest.update(data)
            remaining -= take
        return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def verify_event_bindings(receipt, binding):
    """Audit nominal source-marker placement in metadata, never audio onsets.

    Coverage includes reused/new events and beds. ``sample_hit`` is the renderer's
    source marker; agreement cannot establish audible attack or picture sync.
    Crown differences/peaks are evidence recorded by the renderer, not remeasured
    here. No audio, source asset, or picture files are opened.
    """
    checks, matched = [], {}
    plan = binding.get("reuse_plan", {})
    expected_groups = {
        "events": list(plan.get("EXTRA_EVENTS", [])) + list(binding.get("new_events", [])),
        "beds": list(plan.get("BED_CROPS", [])) + list(binding.get("new_beds", [])),
    }

    def same(a, b):
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)

    for group, expected in expected_groups.items():
        actual = receipt.get(group, [])
        want, got = {}, {}
        for rows, index, side in ((expected, want, "binding"), (actual, got, "receipt")):
            for row in rows:
                rid = row.get("id")
                if not isinstance(rid, str) or not rid:
                    checks.append(dict(check="nominal binding coverage", group=group, ok=False,
                                       reason=f"missing {side} ID"))
                    continue
                index.setdefault(rid, []).append(row)
        for rid in sorted(want.keys() | got.keys()):
            coverage = len(want.get(rid, [])) == len(got.get(rid, [])) == 1
            checks.append(dict(check="nominal binding coverage", group=group, id=rid, ok=coverage,
                               binding_count=len(want.get(rid, [])), receipt_count=len(got.get(rid, []))))
            if not coverage:
                continue
            expected_row, recorded = want[rid][0], got[rid][0]
            if group == "events":
                matched[rid] = (expected_row, recorded)
            fields = ["source"]
            if group == "beds":
                fields += ["f0", "f1"]
            elif "source_sync" in expected_row:
                fields += ["source_sync"]
            for key in ("request_id", "hook", "picture_frame", "measured_ref", "measurement_scope", "measurement", "envelope_provenance",
                        "race_work_clock", "race_role", "design_revision", "texture_provenance", "stop_f",
                        "end_provenance", "continuity_provenance", "picture_revision",
                        "bridge_provenance", "post_gain_f", "source_span", "source_seed"):
                if key in expected_row:
                    fields.append(key)
            differences = [key for key in fields if key not in expected_row or key not in recorded
                           or not same(expected_row[key], recorded[key])]
            if "donor_crop" in expected_row and not same(expected_row["donor_crop"], recorded.get("donor_crop")):
                differences.append("donor_crop")
            if group == "beds" and "seed_id" in expected_row and expected_row["seed_id"] != recorded.get("seed"):
                differences.append("seed")
            anchor = expected_row.get("source_choice_provenance", {}).get("anchor")
            if anchor is not None:
                layers = recorded.get("source_recipe", {}).get("layers", [])
                if not layers or not same(anchor, layers[0].get("src")):
                    differences.append("source anchor")
            checks.append(dict(check="nominal binding provenance", group=group, id=rid,
                               ok=not differences, differing_fields=differences))
            if group != "events":
                continue
            frame, sample = expected_row.get("hit_f"), recorded.get("sample_hit")
            valid = number(frame) and number(sample) and float(sample).is_integer()
            expected_sample = int(round(frame*FRAME_N)) if number(frame) else None
            error = float(sample-expected_sample) if valid else None
            checks.append(dict(check="nominal source-marker alignment", id=rid,
                               ok=bool(valid and abs(error) <= FRAME_N),
                               expected_sample=expected_sample, recorded_sample=sample if number(sample) else None,
                               error_samples=error, tolerance_samples=FRAME_N,
                               measured_audio_onset=False, measured_picture=False))

    events = receipt.get("events", [])
    gap = [r for r in events if number(r.get("sample_hit")) and
           3440*FRAME_N <= r["sample_hit"] < 3520*FRAME_N]
    checks.append(dict(check="nominal gap single event", ok=len(gap) == 1 and
                       gap[0].get("request_id") == "D.new.gap.hammer",
                       event_ids=[r.get("id") for r in gap], frames=[3440, 3520],
                       scope="event metadata; score muting and wet-tail exclusion are separate checks"))
    crowns = {}
    for side, pan, inactive in (("left", -1, 1), ("right", 1, 0)):
        pairs = [(e, r) for e, r in matched.values() if e.get("request_id") == "D.new.crowns."+side]
        valid = len(pairs) == 1
        peaks = pairs[0][1].get("isolated_channel_peaks") if valid else None
        valid = (valid and pairs[0][0].get("pan") == pan and isinstance(peaks, list) and len(peaks) == 2
                 and all(number(p) for p in peaks) and peaks[inactive] == 0 and peaks[1-inactive] > 0)
        checks.append(dict(check="recorded isolated crown channels", side=side, ok=bool(valid), peaks=peaks,
                           scope="renderer-recorded isolated samples, before combined mastering"))
        if len(pairs) == 1:
            crowns[side] = pairs[0][1]
    pair = receipt.get("crown_pair", {})
    difference, correlation, compared = (pair.get(key) for key in
                                        ("max_channel_difference", "correlation", "compared_samples"))
    hits = all(side in crowns and pair.get(side+"_hit_sample") == crowns[side].get("sample_hit")
               for side in ("left", "right"))
    distinct = (number(difference) and difference > 0 and number(correlation) and
                number(compared) and compared > 0 and hits)
    checks.append(dict(check="recorded crown pair distinction", ok=bool(distinct),
                       max_channel_difference=difference if number(difference) else None,
                       correlation=correlation if number(correlation) else None,
                       compared_samples=compared if number(compared) else None,
                       recorded_hits_match_events=hits,
                       scope="renderer-recorded isolated source vectors; no final-mix onset measurement"))
    return dict(ok=bool(checks) and all(row["ok"] for row in checks), checks=checks,
                measurement_scope="nominal binding metadata audit; no audio-onset or picture measurement")


def gap_arrival(x, first_frame, expected_frame=3440):
    """Measure D's gap attack against its 40-frame vacuum, not C5's lookback.

    Same centered 5 ms energy envelope as verify_c5_render: first sample within
    6 dB of the search-window peak. Isolation uses [3400,expected-0.7 seconds)
    and requires its maximum 30 dB below the attack. Exact digital silence is
    checked separately by constraints(). The supplied excerpt must cover the
    whole half-open [3400,expected+12) range; absent samples are never padded.
    Pass the canonical binding's gap_contact_frame when using a native overlay.
    """
    if type(expected_frame) is not int or not 3440 <= expected_frame < 3520:
        raise ValueError('gap target must be an integer frame in the gap edit')
    end_frame = expected_frame + 12
    row = dict(check="gap measured audio arrival", expected_frame=expected_frame, coverage_frames=[3400, end_frame],
               search_seconds_relative=[-.7, .5], isolation_start_frame=3400,
               isolation_end_frame=expected_frame-.7*FPS, envelope_ms=5, under_peak_db=6,
               isolation_below_peak_db=30, tolerance_frames=1, measured_picture=False)
    lo, hi = (3400-first_frame)*FRAME_N, (end_frame-first_frame)*FRAME_N
    if lo < 0 or hi > len(x) or hi <= lo:
        return dict(row, ok=False, status="MISSING COVERAGE")
    segment = np.asarray(x[lo:hi], np.float64)
    if not np.isfinite(segment).all():
        return dict(row, ok=False, status="NONFINITE")
    mono = np.abs(segment).mean(axis=1) if segment.ndim == 2 else np.abs(segment)
    width = int(.005*SR)
    env = 10*np.log10(np.convolve(mono**2, np.ones(width)/width, mode="same")+1e-14)
    event = (expected_frame-3400)*FRAME_N
    start = event-int(.7*SR)
    window = env[start:event+int(.5*SR)]
    peak, pre = float(window.max()), float(env[:start].max())
    row.update(peak_db=peak, pre_db=pre)
    if peak < -100:
        return dict(row, ok=False, status="SILENT")
    arrived = start+int(np.argmax(window >= peak-6))
    delta = (arrived-event)/FRAME_N
    row.update(delta_frames=float(delta), measured_audio_frame=expected_frame+float(delta))
    if pre > peak-30:
        return dict(row, ok=False, status="NOT ISOLATED")
    return dict(row, ok=bool(abs(delta) <= 1), status="ON" if abs(delta) <= 1 else "OFF")


def verify_paths(sfx_path, mix_path, score_path, *, bands, sections, ap2_root, score_bands=None, out_json=None, resident=False):
    """Decode one file at a time to temporary mmap; emit only numerical evidence.

    The MIX level map contributes to acceptance. Score level rows are additional
    evidence (including D34's contained and legacy windows), not a MIX-silence
    requirement. ``score_bands`` preserves the source score contract when a MIX
    band is explicitly authored differently; omitted, it defaults to ``bands``.
    ``ap2_root`` contains the three delivered sound_AP2*.wav files.
    """
    paths = {"sfx": Path(sfx_path), "mix": Path(mix_path), "score": Path(score_path)}
    references = {"mix": "sound_AP2.wav", "sfx": "sound_AP2_sfx.wav", "score": "sound_AP2_score.wav"}
    checks, tracks, score_rows, identities = [], {}, [], {}
    if type(resident) is not bool:
        raise ValueError('resident verification must be explicitly boolean')
    bands, sections = list(bands), list(sections)
    score_bands = bands if score_bands is None else list(score_bands)
    with tempfile.TemporaryDirectory(prefix="verify-d-effects-") as temp:
        for kind, path in paths.items():
            try:
                info = sf.info(path)
                format_ok = (info.frames == SAMPLE_FRAMES and info.samplerate == SR and
                             info.channels == 2 and info.subtype == "PCM_24" and info.format == "WAV")
                checks.append(dict(check="WAV format", kind=kind, ok=format_ok, sample_frames=info.frames,
                                   sample_rate=info.samplerate, channels=info.channels,
                                   subtype=info.subtype, format=info.format))
                identities[kind] = dict(path=str(path), sha256=file_sha256(path))
                actual = pcm_prefix_sha256(path, PREFIX_FRAMES)
                expected = pcm_prefix_sha256(Path(ap2_root)/references[kind], PREFIX_FRAMES)
                checks.append(dict(check="packed AP2 prefix", kind=kind, ok=actual == expected,
                                   frames=[0, PREFIX_FRAMES], actual_sha256=actual, expected_sha256=expected))
                if not format_ok:
                    continue
                mmap_path = Path(temp)/(kind+".f32")
                x = (np.empty((info.frames, 2), np.float32) if resident else
                     np.memmap(mmap_path, mode="w+", dtype=np.float32, shape=(info.frames, 2)))
                try:
                    with sf.SoundFile(path) as handle:
                        for a in range(0, info.frames, CHUNK):
                            block = handle.read(min(CHUNK, info.frames-a), dtype="float32", always_2d=True)
                            if len(block) != min(CHUNK, info.frames-a):
                                raise ValueError("short audio read")
                            x[a:a+len(block)] = block
                    sample_checks = constraints(x, kind)
                    checks.extend(sample_checks)
                    if not all(row["ok"] for row in sample_checks if row["check"] == "finite"):
                        continue
                    tp = true_peak_db(x)
                    checks.append(dict(check="true peak", kind=kind, ok=tp < TRUE_PEAK_LIMIT,
                                       dbtp=tp, limit_dbtp=TRUE_PEAK_LIMIT, comparison="strictly below"))
                    loudness = lufs(x)
                    checks.append(dict(check="integrated loudness", kind=kind,
                                       ok=bool(np.isfinite(loudness) and abs(loudness-TARGET_LUFS) <= LUFS_TOLERANCE)
                                       if kind == "mix" else True,
                                       lufs=float(loudness) if np.isfinite(loudness) else None,
                                       target_lufs=TARGET_LUFS if kind == "mix" else None,
                                       tolerance_lu=LUFS_TOLERANCE if kind == "mix" else None))
                    if kind in ("mix", "score"):
                        track = st_loudness(x)
                        tracks[kind] = track.tolist()
                        if kind == "mix":
                            checks.extend(dict(row, kind=kind) for row in level_rows(track, sections, bands))
                        else:
                            score_rows = level_rows(track, sections, score_bands, hard_silences=(SCORE_TAIL[:2],))
                finally:
                    if not resident:
                        x._mmap.close()
                    del x
                    mmap_path.unlink(missing_ok=True)
            except (OSError, ValueError, RuntimeError, wave.Error) as exc:
                checks.append(dict(check="verification error", kind=kind, ok=False, reason=str(exc)))
    report = dict(ok=bool(checks) and all(row["ok"] for row in checks), checks=checks,
                  paths=identities, short_term_loudness=tracks, score_level_rows=score_rows,
                  storage="resident float32" if resident else "private float32 memmap",
                  measurement_scope="exported audio only; no picture-sync measurement or listening claim")
    if out_json is not None:
        Path(out_json).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report
