"""Synthetic audit proofs only: these numbers describe generated fixtures, never film A.

No WAVs, delivered frames, audio renderer, or cache are loaded. The fake source records
every read so whole-film coverage and bounded reads can be proved without a film array.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
import py_compile

import numpy as np
import pytest

import audit_sync_a as A


class SyntheticAudio:
    samplerate, channels, subtype = A.SR, 2, "PCM_24"

    def __init__(self, onset=None, amplitude=0.1, *, frames=A.SAMPLES, nan=False):
        self.onset, self.amplitude, self.frames, self.nan = onset, amplitude, frames, nan
        self.reads = []

    def read(self, start, stop):
        self.reads.append((start, stop))
        if stop > self.frames:
            stop = self.frames
        index = np.arange(start, stop)
        wave = np.zeros(len(index), np.float64)
        if self.onset is not None:
            active = index >= self.onset * A.FR
            wave[active] = self.amplitude * np.sin(2 * np.pi * 1000 * index[active] / A.SR)
        if self.nan and len(wave):
            wave[len(wave) // 2] = np.nan
        return np.repeat(wave[:, None], 2, axis=1)

    def file_sha256(self):
        # A deterministic stand-in for encoded-file identity in generated-only tests.
        # Real WaveAudio's full byte-stream hash is exercised separately below.
        identity = repr(("SYNTHETIC", self.onset, self.amplitude, self.frames, self.nan)).encode()
        return hashlib.sha256(identity).hexdigest()


class SumAudio(SyntheticAudio):
    def __init__(self, *sources):
        super().__init__()
        self.sources = sources

    def read(self, start, stop):
        return sum((source.read(start, stop) for source in self.sources), np.zeros((stop - start, 2)))

    def file_sha256(self):
        return hashlib.sha256(("SYNTHETIC SUM:" + ":".join(src.file_sha256() for src in self.sources)).encode()).hexdigest()


def fingerprinted(spec, contribution, mix):
    out = dict(spec)
    out.setdefault("detector_audio_sha256", dict(contribution=contribution.file_sha256(), mix=mix.file_sha256()))
    return out


def measure_event(event, spec, contribution, masker, **kwargs):
    """Supply the actual sum of these generated sources, never an inferred film mix."""
    mix = SumAudio(contribution, masker)
    return A.measure_event(event, fingerprinted(spec, contribution, mix), contribution, masker, mix=mix, **kwargs)


def audit_events(events, picture, sfx, score, *args):
    mix = SumAudio(sfx, score)
    picture = copy.deepcopy(picture)
    for spec in picture.get("audit", {}).get("events", []):
        spec.update(fingerprinted(spec, sfx, mix))
    return A.audit_events(events, picture, sfx, score, *args, mix=mix)


def check_score_sync(barmap, picture, score=None, sfx=None):
    mix = SumAudio(score, sfx) if score is not None and sfx is not None else None
    picture = copy.deepcopy(picture)
    if mix is not None:
        for spec in picture.get("audit", {}).get("score_sync", []):
            if isinstance(spec.get("audio"), dict):
                spec["audio"] = fingerprinted(spec["audio"], score, mix)
    return A.check_score_sync(barmap, picture, score, sfx, mix)


def event_spec(**changes):
    spec = dict(id="A.synthetic", kind="hard", picture_frame=48,
                search_frames=[44, 58], control_frames=[36, 44], band_hz=[500, 2000],
                detector_validated=True, detector_evidence="Synthetic isolated 1 kHz test tone; no film evidence")
    spec.update(changes)
    return spec


def table(**changes):
    event = dict(id="A.synthetic", kind="event", t=2.0)
    event.update(changes)
    return event


def measured(onset=48, spec=None, *, sfx=None, score=None):
    return measure_event(table(), event_spec() if spec is None else spec,
                           SyntheticAudio(onset) if sfx is None else sfx,
                           SyntheticAudio() if score is None else score)


def test_hard_on_frame_and_two_frames_late():
    on = measured(48)
    assert on["ok"] and on["status"] == "PASS_PROXY", on
    assert 0 <= on["offset_frames"] <= 1
    late = measured(50)
    assert not late["ok"] and late["status"] == "OFF", late
    assert late["offset_frames"] >= 2
    assert late["mix_onset_frame"] != late["table_frame"]


def test_hard_early_and_soft_tolerances():
    assert not measured(47)["ok"]
    assert measured(50, event_spec(kind="soft"))["ok"]
    assert not measured(52, event_spec(kind="soft"))["ok"]


def test_fractional_control_and_search_bounds_are_sample_resolved():
    spec = event_spec(control_frames=[42.2, 47.6], search_frames=[47.6, 58.1])
    assert measured(48, spec)["ok"]
    assert not measured(50, spec)["ok"]
    spec["control_frames"][1] = 47.7
    assert measured(48, spec)["status"] == "UNRESOLVED"


def test_frame_zero_onset_requires_unavailable_preceding_control():
    row = measured(0, event_spec(picture_frame=0, control_frames=[0, 1], search_frames=[1, 10]))
    assert not row["ok"] and row["status"] == "UNRESOLVED"


def test_timing_exception_is_only_for_measured_attributed_nonhard_event():
    assert measured(52, event_spec(kind="soft", timing_exception="Synthetic deliberate four-frame tail."))["ok"]
    assert not measured(52, event_spec(kind="hard", timing_exception="Cannot excuse hard timing."))["ok"]
    assert not measured(52, event_spec(kind="soft", detector_validated=False,
                                      timing_exception="Cannot excuse missing attribution."))["ok"]


def test_missing_kind_defaults_conservatively_from_event_table():
    spec = event_spec()
    del spec["kind"]
    assert not measured(50, spec)["ok"]
    assert measure_event(table(kind="bed"), spec, SyntheticAudio(50), SyntheticAudio())["ok"]


@pytest.mark.parametrize("picture_frame", [float("nan"), float("inf"), True, 48.5])
def test_unresolved_picture_cannot_pass(picture_frame):
    row = measured(spec=event_spec(picture_frame=picture_frame))
    assert not row["ok"] and row["status"] == "UNRESOLVED"
    assert row["mix_onset_frame"] is None
    json.dumps(row, allow_nan=False)


def test_unknown_picture_still_measures_audio_but_never_invents_offset_or_pass():
    on = measured(48, event_spec(picture_frame=None))
    late = measured(50, event_spec(picture_frame=None))
    for row in (on, late):
        assert not row["ok"] and row["status"] == "UNRESOLVED"
        assert row["stem_onset_frame"] is not None and row["mix_onset_frame"] is not None
        assert row["picture_frame"] is None and row["offset_frames"] is None and row["stem_offset_frames"] is None
        json.dumps(row, allow_nan=False)
    assert late["mix_onset_frame"] - on["mix_onset_frame"] >= 1.99


def test_silence_and_nonfinite_audio_do_not_invent_onsets():
    for source in (SyntheticAudio(), SyntheticAudio(48, nan=True)):
        row = measured(sfx=source)
        assert not row["ok"] and row["stem_onset_frame"] is None
        assert row["mix_onset_frame"] is None


def test_continuous_bed_and_high_control_do_not_claim_isolation():
    row = measured(sfx=SyntheticAudio(0), spec=event_spec(kind="bed"))
    assert not row["ok"] and row["status"] == "NOT_ISOLATED", row
    assert row["stem_onset_frame"] is None and row["mix_onset_frame"] is None


def test_masked_proxy_is_not_audible_claim():
    row = measured(sfx=SyntheticAudio(48, amplitude=0.005), score=SyntheticAudio(0, amplitude=0.5))
    assert not row["ok"] and row["status"] == "MASKED_PROXY", row
    assert row["stem_onset_frame"] is not None and row["mix_onset_frame"] is None


def test_unattributed_bus_energy_is_unresolved_even_when_on_time():
    row = measured(spec=event_spec(detector_validated=False))
    assert not row["ok"] and row["status"] == "UNRESOLVED"
    assert row["stem_onset_frame"] is not None


@pytest.mark.parametrize("changed", ["missing", "contribution", "mix"])
def test_aggregate_attribution_requires_current_full_file_fingerprints(changed):
    sfx, score = SyntheticAudio(48), SyntheticAudio()
    mix = SumAudio(sfx, score)
    spec = fingerprinted(event_spec(), sfx, mix)
    assert A.measure_event(table(), spec, sfx, score, mix=mix)["ok"]
    if changed == "missing":
        del spec["detector_audio_sha256"]
    else:
        spec["detector_audio_sha256"][changed] = "0" * 64
    row = A.measure_event(table(), spec, sfx, score, mix=mix)
    assert not row["ok"] and row["status"] == "UNRESOLVED"
    assert "fingerprint" in row["reason"]


def test_reconciled_cue_path_does_not_need_aggregate_fingerprints():
    cue, score = SyntheticAudio(48), SyntheticAudio()
    mix = SumAudio(cue, score)
    spec = event_spec(detector_validated=False)
    assert "detector_audio_sha256" not in spec
    assert A.check_cue_reconstruction(cue, {"A.synthetic": cue}, ["A.synthetic"], 3 * A.SR)["ok"]
    assert A.measure_event(table(), spec, cue, score, mix=mix, isolated_cue=True)["ok"]


def test_waveaudio_fingerprint_reads_full_bytes_in_bounded_chunks_and_rejects_changes(tmp_path, monkeypatch):
    # This arbitrary byte file is never opened or decoded as audio.
    payload = b"synthetic fingerprint bytes" * 25000
    path = tmp_path / "fingerprint-bytes.bin"
    path.write_bytes(payload)
    source = A.WaveAudio.__new__(A.WaveAudio)
    source.path, source._fingerprint = str(path), None
    source._opened_stat = source._stat()
    read_sizes = []
    builtin_open = open

    class ReadSpy:
        def __init__(self, *args):
            self.stream = builtin_open(*args)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def read(self, size):
            read_sizes.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(A, "open", ReadSpy, raising=False)
    assert source.file_sha256() == hashlib.sha256(payload).hexdigest()
    assert len(read_sizes) >= 3 and max(read_sizes) == 256 * 1024
    count = len(read_sizes)
    assert source.file_sha256() == hashlib.sha256(payload).hexdigest()
    assert len(read_sizes) == count
    path.write_bytes(payload + b"changed")
    with pytest.raises(A.InvalidAudio, match="changed"):
        source.file_sha256()


@pytest.mark.parametrize("changes", [dict(band_hz=[0, 1000]), dict(band_hz=[500, float("nan")]),
                                    dict(control_frames=[36, 43]), dict(control_frames=[44, 46]),
                                    dict(search_frames=[50, 58]), dict(kind="event")])
def test_invalid_detector_spec_fails(changes):
    row = measured(spec=event_spec(**changes))
    assert not row["ok"] and row["status"] == "UNRESOLVED"


def test_every_event_reported_and_unknown_picture_event_fails():
    events = [table(), table(id="A.missing")]
    picture = dict(audit=dict(events=[event_spec(), event_spec(id="A.unknown")]))
    rows, checks = audit_events(events, picture, SyntheticAudio(48), SyntheticAudio())
    assert len(rows) == len(events)
    assert rows[0]["ok"] and not rows[1]["ok"] and rows[1]["status"] == "UNRESOLVED"
    assert any(not check["ok"] for check in checks)


def test_legacy_picture_t_is_not_measurement_fallback():
    rows, _ = audit_events([table()], dict(t={"A.synthetic": {"f": 48}}), SyntheticAudio(48), SyntheticAudio())
    assert rows[0]["status"] == "UNRESOLVED"
    assert rows[0]["mix_onset_frame"] is None


def test_duplicate_picture_or_table_ids_fail():
    for events, specs in (([table()], [event_spec(), event_spec()]), ([table(), table()], [event_spec()])):
        rows, _ = audit_events(events, dict(audit=dict(events=specs)), SyntheticAudio(48), SyntheticAudio())
        assert rows and all(not row["ok"] for row in rows)


def test_missing_named_cue_is_unresolved_without_bus_fallback():
    @contextmanager
    def absent(_):
        raise KeyError("missing cue")
        yield  # pragma: no cover
    rows, _ = audit_events([table()], dict(audit=dict(events=[event_spec()])),
                             SyntheticAudio(48), SyntheticAudio(), absent)
    assert not rows[0]["ok"] and rows[0]["stem_onset_frame"] is None


def test_cue_attribution_can_supply_individually_isolated_onset():
    spec = event_spec(detector_validated=False)
    row = measure_event(table(), spec, SyntheticAudio(48), SyntheticAudio(), isolated_cue=True)
    assert row["ok"] and row["attribution"] == "postmaster cue supplied by manifest"


def test_named_cue_masker_includes_competing_effects_when_score_is_silent():
    cue = SyntheticAudio(48, amplitude=0.005)
    score = SyntheticAudio()
    other_effect = SyntheticAudio(0, amplitude=0.5)
    clear = A.measure_event(table(), event_spec(), cue, score, mix=SumAudio(cue, score), isolated_cue=True)
    buried = A.measure_event(table(), event_spec(), cue, score,
                             mix=SumAudio(cue, other_effect, score), isolated_cue=True)
    assert clear["ok"]
    assert not buried["ok"] and buried["status"] == "MASKED_PROXY"
    assert buried["masker"] == "actual mix minus reconciled cue"
    assert buried["stem_onset_frame"] is not None and buried["mix_onset_frame"] is None


def test_mix_checks_entire_film_in_bounded_chunks():
    mix, sfx, score = SyntheticAudio(48), SyntheticAudio(48), SyntheticAudio()
    result = A.check_mix(mix, sfx, score)
    assert result["ok"] and result["samples_checked"] == A.SAMPLES
    for source in (mix, sfx, score):
        assert source.reads[0][0] == 0 and source.reads[-1][1] == A.SAMPLES
        assert max(b - a for a, b in source.reads) <= A.BLOCK


def test_mix_mismatch_quantization_and_nan_fail():
    assert A.check_mix(SyntheticAudio(0), SyntheticAudio(0), SyntheticAudio(), A.SR)["ok"]
    assert not A.check_mix(SyntheticAudio(0, amplitude=0.11), SyntheticAudio(0), SyntheticAudio(), A.SR)["ok"]
    assert not A.check_mix(SyntheticAudio(nan=True), SyntheticAudio(), SyntheticAudio(), A.SR)["ok"]
    assert not A.check_mix(SyntheticAudio(frames=100), SyntheticAudio(), SyntheticAudio(), A.SR)["ok"]


def test_quantization_bound_includes_all_three_files():
    class Constant(SyntheticAudio):
        def read(self, start, stop):
            return np.full((stop - start, 2), self.amplitude)
    bound = 3 * (A.quantization_bound("PCM_24") + 2 ** -23) + 4 * np.finfo(np.float32).eps
    assert A.check_mix(Constant(amplitude=bound), Constant(amplitude=0), Constant(amplitude=0), 100)["ok"]
    assert not A.check_mix(Constant(amplitude=bound * 1.01), Constant(amplitude=0), Constant(amplitude=0), 100)["ok"]
    # Each file can differ by 1 LSB dither plus 1 LSB conversion, with opposite signs:
    # |mix - score - sfx| can therefore approach 6 LSB before float32 addition error.
    lsb = 2 ** -23
    extreme = A.check_mix(Constant(amplitude=2 * lsb), Constant(amplitude=-2 * lsb), Constant(amplitude=-2 * lsb), 100)
    assert extreme["ok"] and extreme["maximum_error"] == 6 * lsb


def test_actual_renderer_dither_on_synthetic_arrays_with_conversion():
    from mix import tpdf_dither_24
    rng = np.random.default_rng(42)
    sfx = rng.uniform(-0.4, 0.4, (A.BLOCK, 2)).astype(np.float32)
    score = rng.uniform(-0.4, 0.4, (A.BLOCK, 2)).astype(np.float32)
    mix = sfx + score
    lsb = 2 ** -23

    class ArraySource(SyntheticAudio):
        def __init__(self, data):
            super().__init__(frames=len(data))
            # A whole-LSB conversion bound permits truncation; no actual audio file is used.
            self.data = np.floor(tpdf_dither_24(data, rng).astype(np.float64) / lsb) * lsb
        def read(self, start, stop):
            return self.data[start:stop]
    mix, sfx, score = (ArraySource(x) for x in (mix, sfx, score))
    result = A.check_mix(mix, sfx, score, A.BLOCK)
    assert result["ok"] and result["maximum_error"] > 3 * lsb
    mix.data[0, 0] += 32 * lsb
    assert not A.check_mix(mix, sfx, score, A.BLOCK)["ok"]


def test_audio_format_contract_both_ways():
    source = SyntheticAudio()
    assert A.check_format(source, "synthetic")["ok"]
    for field, value in (("samplerate", 44100), ("channels", 1), ("frames", A.SAMPLES - 1)):
        altered = copy.copy(source)
        setattr(altered, field, value)
        assert not A.check_format(altered, "synthetic")["ok"]


def test_envelope_reads_are_bounded_and_causal():
    source = SyntheticAudio(48)
    samples, envelope = A.band_envelope(source, 36 * A.FR, 58 * A.FR, [500, 2000])
    assert max(b - a for a, b in source.reads) <= A.BLOCK
    assert np.all(envelope[samples < 48 * A.FR] < A.FLOOR_DB)
    assert np.all(np.diff(samples) == A.HOP) and A.HOP <= A.WINDOW


def test_score_coverage_requires_every_point_and_audio_evidence():
    barmap = dict(sync=[dict(id="hit", f=48), dict(id="missing", f=90)])
    picture = dict(audit=dict(score_sync=[dict(id="hit", frame=48, status="measured", measured_picture_frame=48, kind="hard")]))
    checks = check_score_sync(barmap, picture, SyntheticAudio(48), SyntheticAudio())
    assert len(checks) == 2 and all(not c["ok"] for c in checks)
    assert checks[0]["status"] == "SCORE_AUDIO_UNRESOLVED"
    assert checks[1]["status"] == "UNRESOLVED"


def test_score_waveform_shift_is_detected():
    audio = event_spec()
    picture = dict(audit=dict(score_sync=[dict(id="hit", frame=48, status="measured", measured_picture_frame=48,
                                             kind="hard", audio=audio)]))
    barmap = dict(sync=[dict(id="hit", f=48)])
    on = check_score_sync(barmap, picture, SyntheticAudio(48), SyntheticAudio())[0]
    late = check_score_sync(barmap, picture, SyntheticAudio(50), SyntheticAudio())[0]
    assert on["ok"] and not late["ok"] and late["status"] == "OFF"


def test_reference_only_requires_explicit_reason_and_measured_picture():
    spec = dict(id="boundary", frame=48, status="measured", measured_picture_frame=48, role="reference")
    barmap = dict(sync=[dict(id="boundary", f=48)])
    picture = dict(audit=dict(score_sync=[spec]))
    assert not A.check_score_sync(barmap, picture)[0]["ok"]
    spec["reference_reason"] = "Synthetic pure camera boundary; no score event is declared."
    assert A.check_score_sync(barmap, picture)[0]["ok"]
    spec["measured_picture_frame"] = None
    assert not A.check_score_sync(barmap, picture)[0]["ok"]
    spec.update(kind="hard", measured_picture_frame=48)
    assert not A.check_score_sync(barmap, picture)[0]["ok"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 1e308, 10 ** 400])
def test_invalid_event_time_fails_with_json_safe_null(value):
    row = measure_event(table(t=value), event_spec(), SyntheticAudio(48), SyntheticAudio())
    assert not row["ok"] and row["table_frame"] is None
    json.dumps(row, allow_nan=False)


def test_full_run_on_generated_sources_requires_score_waveform(tmp_path, monkeypatch):
    amplitudes = {"mix": 0.2, "sfx": 0.1, "score": 0.1}
    class Source(SyntheticAudio):
        def __init__(self, path):
            super().__init__(48, amplitude=amplitudes[str(path)])
        def read(self, start, stop):
            data = super().read(start, stop)
            frames = np.arange(start, stop) / A.FR
            data[~(((frames >= 48) & (frames < 60)) | ((frames >= 88) & (frames < 100)))] = 0
            return data
        def close(self):
            pass
    monkeypatch.setattr(A, "WaveAudio", Source)
    events, picture, barmap = (tmp_path / name for name in ("events.json", "picture.json", "barmap.json"))
    events.write_text(json.dumps([table()]))
    spec = dict(id="hit", frame=48, status="measured", measured_picture_frame=48, kind="hard", audio=event_spec())
    anchors = [dict(id=name, f=frame) for name, frame in
               (("lantern", 48), ("narrowest", 60), ("watchfire_3", 88), ("breath_dawn", 100))]
    references = [dict(id=e["id"], frame=e["f"], status="measured", measured_picture_frame=e["f"],
                       role="reference", reference_reason="Synthetic schedule boundary, no separate score hit.") for e in anchors]
    steps = [dict(id=f"feet.{f}", frame=f, status="measured", measured_picture_frame=f, visible_contact=True,
                  contact_evidence="Synthetic contact fixture, not film evidence.",
                  audio=event_spec(search_frames=[f - 4, f + 10], control_frames=[f - 12, f - 4])) for f in (48, 88)]
    registry = dict(audit=dict(events=[event_spec()], score_sync=[spec] + references, score_steps=steps))
    registry["audit"]["events"][0] = fingerprinted(registry["audit"]["events"][0], Source("sfx"), Source("mix"))
    spec["audio"] = fingerprinted(spec["audio"], Source("score"), Source("mix"))
    for step in steps:
        step["audio"] = fingerprinted(step["audio"], Source("score"), Source("mix"))
    picture.write_text(json.dumps(registry))
    barmap.write_text(json.dumps(dict(fps=24, bpm=72, sync=[dict(id="hit", f=48)] + anchors)))
    assert A.run("mix", "sfx", "score", picture, events, barmap=barmap)["ok"]
    amplitudes.update(mix=0, score=-0.1)
    cancelled = A.run("mix", "sfx", "score", picture, events, barmap=barmap)
    assert not cancelled["ok"] and cancelled["events"][0]["status"] == "MASKED_PROXY"
    assert any(c["check"] == "mix_equals_score_plus_sfx" and c["ok"] for c in cancelled["checks"])
    amplitudes.update(mix=0.2, score=0.1)
    del spec["audio"]
    picture.write_text(json.dumps(registry))
    result = A.run("mix", "sfx", "score", picture, events, barmap=barmap)
    assert not result["ok"] and result["events"][0]["ok"]
    assert any(c.get("status") == "SCORE_AUDIO_UNRESOLVED" for c in result["checks"])


def test_actual_mix_cancellation_cannot_pass_or_be_replaced_by_stem_ratios():
    sfx, score = SyntheticAudio(48, amplitude=0.1), SyntheticAudio(48, amplitude=-0.1)
    mix = SumAudio(sfx, score)
    assert A.check_mix(mix, sfx, score, 3 * A.SR)["ok"]
    row = A.measure_event(table(), event_spec(), sfx, score, mix=mix)
    assert not row["ok"] and row["status"] == "MASKED_PROXY" and row["mix_onset_frame"] is None
    assert row["stem_onset_frame"] is not None
    assert not A.measure_event(table(), event_spec(), sfx, score)["ok"]
    partly_cancelled = SumAudio(sfx, SyntheticAudio(48, amplitude=-0.09))
    row = A.measure_event(table(), event_spec(), sfx, score, mix=partly_cancelled)
    assert not row["ok"] and row["status"] == "MASKED_PROXY"


def test_cue_sum_rejects_stale_missing_extra_and_changed_cues():
    cue, bus = SyntheticAudio(0), SyntheticAudio(0)
    assert A.check_cue_reconstruction(bus, {"one": cue}, ["one"], 2 * A.SR)["ok"]
    assert max(b - a for a, b in cue.reads) <= A.BLOCK
    for actual, cues, expected in ((SyntheticAudio(), {"one": cue}, ["one"]),
                                   (bus, {}, ["one"]), (bus, {"one": cue, "unknown": cue}, ["one"]),
                                   (bus, {"one": SyntheticAudio(0, amplitude=0.11)}, ["one"]),
                                   (bus, {"one": cue}, ["one", "one"])):
        assert not A.check_cue_reconstruction(actual, cues, expected, 2 * A.SR)["ok"]


def test_stale_cue_does_not_gain_attributed_onset_in_full_run(tmp_path, monkeypatch):
    class Source(SyntheticAudio):
        def __init__(self, path):
            super().__init__(48, amplitude=0 if str(path) == "sfx" else 0.1)
        def close(self):
            pass
    monkeypatch.setattr(A, "WaveAudio", Source)
    events, picture, barmap = (tmp_path / name for name in ("events.json", "picture.json", "barmap.json"))
    events.write_text(json.dumps([table()]))
    picture.write_text(json.dumps(dict(audit=dict(events=[event_spec()]))))
    barmap.write_text(json.dumps(dict(sync=[dict(id="hit", f=48)])))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(events={"A.synthetic": dict(path="cue.wav", offset_samples=0)})))
    report = A.run("mix", "sfx", "score", picture, events, cue_dir=tmp_path, barmap=barmap)
    assert not report["ok"] and report["events"][0]["status"] == "UNRESOLVED"
    assert report["events"][0]["stem_onset_frame"] is None
    assert any(c["check"] == "cue_sum_equals_sfx" and not c["ok"] for c in report["checks"])


def test_current_walk_contract_and_every_expected_step_are_pinned():
    barmap = json.loads((A.MUSIC / "v3/barmap_A.json").read_text())
    expected = list(range(4880, 5360, 40)) + list(range(5520, 5833, 40))
    assert A.step_source_contract()["ok"]
    assert [event["f"] for event in A.expected_score_steps(barmap)] == expected
    checks = A.check_score_steps(barmap, {})
    rows = [row for row in checks if row["check"] == "score_step"]
    assert len(rows) == 20 and [row["id"] for row in rows] == [f"feet.{f}" for f in expected]
    assert all(not row["ok"] and row["status"] == "UNRESOLVED" for row in rows)


@pytest.mark.parametrize("changed", ["src/score_v3_A.py", "src/kit_v3.py", "v3/cues_A.json"])
def test_changed_step_source_or_cue_override_requires_contract_review(tmp_path, changed):
    for key in A.STEP_CONTRACT:
        relative = key.split(":")[0]
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((A.MUSIC / relative).read_text())
    relative = changed.split(":")[0]
    path = tmp_path / relative
    source = path.read_text()
    if relative == "src/score_v3_A.py":
        changed_source = source.replace('"feet", lan, narrow', '"feet", lan + 1, narrow')
    elif relative == "src/kit_v3.py":
        changed_source = source.replace("t += 2.0", "t += 3.0")
    else:
        data = json.loads(source)
        data["events"]["lantern"] = dict(frame=5000, t=5000 / A.FPS)
        changed_source = json.dumps(data)
    assert source != changed_source
    path.write_text(changed_source)
    assert not A.step_source_contract(tmp_path)["ok"]


def test_step_requires_contact_and_hard_waveform_even_with_all_sync_rows_present():
    barmap = dict(fps=24, bpm=72, sync=[dict(id=name, f=f) for name, f in
                 (("lantern", 48), ("narrowest", 60), ("watchfire_3", 88), ("breath_dawn", 100))])
    spec = dict(id="feet.48", frame=48, status="measured", measured_picture_frame=48,
                visible_contact=True, contact_evidence="Synthetic contact", audio=event_spec())
    picture = dict(audit=dict(score_steps=[spec]))
    score, sfx = SyntheticAudio(48), SyntheticAudio()
    spec["audio"] = fingerprinted(spec["audio"], score, SumAudio(score, sfx))
    rows = A.check_score_steps(barmap, picture, score, sfx, SumAudio(score, sfx))[1:]
    assert rows[0]["ok"] and not rows[1]["ok"]
    late = SyntheticAudio(50)
    assert not A.check_score_steps(barmap, picture, late, sfx, SumAudio(late, sfx))[1]["ok"]
    spec.update(visible_contact=False, status="NO_VISIBLE_CONTACT")
    assert A.check_score_steps(barmap, picture, score, sfx, SumAudio(score, sfx))[1]["status"] == "NO_VISIBLE_CONTACT"


def test_cli_nonzero_json_when_audio_missing(tmp_path, capsys):
    events, picture, barmap = (tmp_path / name for name in ("events.json", "picture.json", "barmap.json"))
    events.write_text(json.dumps([table()]))
    picture.write_text(json.dumps(dict(audit=dict(events=[event_spec()]))))
    barmap.write_text(json.dumps(dict(sync=[dict(id="hit", f=48)])))
    missing = str(tmp_path / "absent.wav")
    code = A.main(["--events", str(events), "--picture", str(picture), "--barmap", str(barmap),
                   "--mix", missing, "--sfx", missing, "--score", missing, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and not report["ok"] and len(report["events"]) == 1
    assert report["events"][0]["mix_onset_frame"] is None
    picture.unlink()
    code = A.main(["--events", str(events), "--picture", str(picture), "--barmap", str(barmap),
                   "--mix", missing, "--sfx", missing, "--score", missing, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and len(report["events"]) == 1


def test_cli_auto_and_explicit_cuts_select_matching_inputs_without_fallback(tmp_path, monkeypatch, capsys):
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return dict(ok=False, events=[], checks=[], cut=kwargs["cut"])

    monkeypatch.setattr(A, "run", capture)
    common = ["--music-root", str(tmp_path), "--json"]
    assert A.main(common) == 1
    assert calls[-1]["cut"] == "A"
    (tmp_path / "out/v3").mkdir(parents=True)
    (tmp_path / "out/v3/sound_AP2.wav").touch()  # Existence selector only: never decoded as audio.
    assert A.main(common) == 1
    chosen = calls[-1]
    assert chosen["cut"] == "AP2"
    assert chosen["mix"] == tmp_path / "out/v3/sound_AP2.wav"
    assert chosen["sfx"] == tmp_path / "out/v3/sound_AP2_sfx.wav"
    assert chosen["score"] == tmp_path / "out/v3/sound_AP2_score.wav"
    assert chosen["events"] == tmp_path / "sound/events_AP2.json"
    assert chosen["barmap"] == tmp_path / "v3/barmap_AP2.json"
    # Missing AP2 siblings remain AP2 inputs and therefore fail in run(); A is never substituted.
    assert not chosen["score"].exists()
    assert A.main(common + ["--cut", "A"]) == 1
    assert calls[-1]["cut"] == "A" and calls[-1]["events"].name == "events_A.json"
    assert A.main(common + ["--cut", "AP2", "--solo-dir", str(tmp_path / "solos")]) == 1
    assert calls[-1]["solo_dir"] == tmp_path / "solos"
    assert A.main(common + ["--mix", str(tmp_path / "sound_A.wav")]) == 1
    assert calls[-1]["cut"] == "A" and calls[-1]["score"].name == "sound_A_score.wav"
    capsys.readouterr()


def test_per_cut_evidence_selects_ap2_and_preserves_legacy_a_without_mutation():
    picture = dict(audit=dict(events=[event_spec()], score_sync=[dict(id="legacy")],
                             score_steps=[dict(id="feet.4880")],
                             score_sync_by_cut={"AP2": [dict(id="walk_setoff")]},
                             score_steps_by_cut={"AP2": [dict(id="feet.5180")]}))
    before = copy.deepcopy(picture)
    selected_a, check_a = A.picture_for_cut(picture, "A")
    selected_p, check_p = A.picture_for_cut(picture, "AP2")
    assert check_a["ok"] and check_p["ok"]
    assert selected_a["audit"]["score_steps"] == [dict(id="feet.4880")]
    assert selected_p["audit"]["score_steps"] == [dict(id="feet.5180")]
    assert selected_a["audit"]["score_sync"] == [dict(id="legacy")]
    assert selected_p["audit"]["score_sync"] == [dict(id="walk_setoff")]
    assert selected_p["audit"]["events"] == selected_a["audit"]["events"]
    assert picture == before


def test_missing_ap2_evidence_cannot_inherit_a_step_approval():
    spec = dict(id="feet.48", frame=48, status="measured", measured_picture_frame=48,
                visible_contact=True, contact_evidence="Synthetic A contact", audio=event_spec())
    score, sfx = SyntheticAudio(48), SyntheticAudio()
    mix = SumAudio(score, sfx)
    spec["audio"] = fingerprinted(spec["audio"], score, mix)
    picture = dict(audit=dict(score_steps=[spec]))
    kwargs = dict(step_contract=dict(check="score_step_source_contract", ok=True),
                  expected_steps=[dict(id="feet.48", f=48)])
    assert A.check_score_steps({}, picture, score, sfx, mix, **kwargs)[-1]["ok"]
    selected, check = A.picture_for_cut(picture, "AP2")
    assert not check["ok"] and set(check["missing_or_invalid"]) == {
        "score_sync_by_cut.AP2", "score_steps_by_cut.AP2"}
    row = A.check_score_steps({}, selected, score, sfx, mix, **kwargs)[-1]
    assert not row["ok"] and row["status"] == "UNRESOLVED" and row["mix_onset_frame"] is None


def test_derived_step_schedule_uses_selected_builder_and_excludes_muted_ap2_notes(tmp_path):
    src, v3 = tmp_path / "src", tmp_path / "v3"
    src.mkdir()
    v3.mkdir()
    for name in ("kit_v3.py", "dsl.py"):
        (src / name).write_text("# Synthetic metadata fixture, no audio.\n")
    (src / "timeline_v3.py").write_text(
        "import json\nFPS=24\nBEAT_S=5/6\n"
        "class BarMap:\n"
        " def __init__(self, cut, path, cues):\n"
        "  self.frames=json.load(open(path))['frames']\n")
    (src / "score_v3_A.py").write_text(
        "from types import SimpleNamespace as N\n"
        "def note(f,g=0): return N(start=f/20, pitch=1, gain_db=g, vel=0.3)\n"
        "def build(bm): return N(parts={'feet':N(inst='giant_hand',notes=[note(4880),note(4920)])})\n")
    (src / "score_v3_AP2.py").write_text(
        "from score_v3_A import N, note\n"
        "def build(bm): return N(parts={'feet':N(inst='giant_hand',notes=[note(4880,-999),note(5280)]),"
        "'feet_go':N(inst='giant_hand',notes=[note(5180),note(5240)])})\n")
    for cut in ("A", "AP2"):
        (v3 / f"cues_{cut}.json").write_text("{}")
        (v3 / f"barmap_{cut}.json").write_text(json.dumps(dict(cut=cut, frames=6480)))
    a_check, a_notes = A.derived_score_steps("A", tmp_path, v3 / "barmap_A.json")
    p_check, p_notes = A.derived_score_steps("AP2", tmp_path, v3 / "barmap_AP2.json")
    assert a_check["ok"] and [n["f"] for n in a_notes] == [4880, 4920]
    assert p_check["ok"] and [n["f"] for n in p_notes] == [5180, 5240, 5280]
    assert p_notes[0]["part"] == "feet_go"
    # Alter the actual composition: the adapter must read its irregular schedule, not an A-era loop.
    path = src / "score_v3_AP2.py"
    py_compile.compile(str(path), doraise=True)
    original_stat = path.stat()
    path.write_text(path.read_text().replace("note(5240)", "note(5247)"))
    # Same bytes length and timestamp would make Python trust the old 5240 bytecode.
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    changed, notes = A.derived_score_steps("AP2", tmp_path, v3 / "barmap_AP2.json")
    assert changed["ok"] and [n["f"] for n in notes] == [5180, 5247, 5280]
    assert changed["fingerprints"] != p_check["fingerprints"]
    # A future builder must not quietly turn this into an audio-reading operation.
    path.write_text("import soundfile\n" + path.read_text())
    rejected, notes = A.derived_score_steps("AP2", tmp_path, v3 / "barmap_AP2.json")
    assert not rejected["ok"] and notes == [] and "audio module forbidden" in rejected["reason"]
    path.write_text(path.read_text().removeprefix("import soundfile\n"))
    (src / "kit_v3.py").write_text("def anticipate(*args):\n raise RuntimeError('changed helper')\n")
    rejected, notes = A.derived_score_steps("AP2", tmp_path, v3 / "barmap_AP2.json")
    assert not rejected["ok"] and notes == [] and "anticipation helper changed" in rejected["reason"]
    path.unlink()
    failed, notes = A.derived_score_steps("AP2", tmp_path, v3 / "barmap_AP2.json")
    assert not failed["ok"] and notes == []
