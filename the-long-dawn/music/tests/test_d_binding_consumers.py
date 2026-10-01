"""Phase 2 binding-table entry points, with the Phase 1c path preserved.

Only notation and small synthetic PCM fixtures are built here. The table's
schema and measured-shot provenance have separate tests in this lane.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from conftest import _stub_sampler_deps


@pytest.mark.parametrize("flag", ["--bindings", "--events"])
def test_renderer_cli_routes_both_binding_flags(monkeypatch, flag):
    import render_D as R
    calls = []
    monkeypatch.setattr(R, "render", lambda *args, **kwargs: calls.append((args, kwargs)))
    R.main(["--render", "--output-dir", "/tmp/d-score-fixture", flag, "table.json"])
    assert len(calls) == 1
    assert calls[0][1]["event_file"] == "table.json"


@pytest.mark.parametrize("flag", [None, "--bindings", "--events"])
@pytest.mark.parametrize("transition", [False, True])
def test_verifier_cli_uses_shared_loader_and_preserves_no_flag(monkeypatch, tmp_path, flag, transition):
    _stub_sampler_deps()
    import bindings_D
    import score_v3_D as D
    import verify_D as V
    bm, score, loaded = object(), object(), []

    def load(value):
        loaded.append(value)
        return bm

    def build(value, **kwargs):
        assert value is bm
        assert kwargs == ({"transition_pass": True} if transition else {})
        return score

    def verify(wav, opening, given_score, given_map, **kwargs):
        assert given_score is score and given_map is bm
        return {"checks": []}

    monkeypatch.setattr(bindings_D, "load_map", load)
    monkeypatch.setattr(D, "build", build)
    monkeypatch.setattr(D, "check", lambda *a: [])
    monkeypatch.setattr(V, "verify", verify)
    for name in ("ring_problems", "call_problems", "watch_problems", "beacon_problems", "binding_problems"):
        monkeypatch.setattr(V, name, lambda *a: [])
    output = tmp_path / "verification.json"
    args = ["--wav", "unused.wav", "--opening", "unused.wav", "--json", str(output)]
    if flag:
        args += [flag, "table.json"]
    if transition:
        args += ["--transition-pass"]
    assert V.main(args) == 0
    assert loaded == (["table.json"] if flag else [None])
    assert json.loads(output.read_text())["ok"]


def test_render_function_passes_table_to_shared_loader_before_audio(monkeypatch, tmp_path):
    _stub_sampler_deps()
    import bindings_D
    import render_D as R
    import score_v3_D as D
    bm, calls = object(), []

    class StopBeforeAudio(Exception):
        pass

    def load(path):
        calls.append(path)
        return bm

    def build(given):
        assert given is bm
        raise StopBeforeAudio

    monkeypatch.setattr(bindings_D, "load_map", load)
    monkeypatch.setattr(R, "_configure_sampler", lambda *a: None)
    monkeypatch.setattr(R.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(D, "build", build)
    with pytest.raises(StopBeforeAudio):
        R.render(tmp_path, event_file="fixture-bindings.json")
    assert calls == ["fixture-bindings.json"]


def test_no_binding_flag_preserves_legacy_score_audio_identity():
    _stub_sampler_deps()
    from bindings_D import load_map
    from barmap_D_score import DraftMap
    import render_D as R
    import score_v3_D as D
    original = D.build(DraftMap())
    default = D.build(load_map(None))
    # Rebuilt from git 95f9dc101ec00195b3e02356382f1043521134bf with the
    # conftest 10-second attack stub. This pins the committed Phase 1c notation,
    # expression and mix inputs, not only two invocations of today's code.
    assert R._score_identity(default) == "c169666758acf37f99621b96adac750d013d250a95a3048ca5f05e4be1a9a1e9"
    assert R._score_identity(default) == R._score_identity(original)
    assert default.hard_silences == original.hard_silences
    assert default.crossing_source["destination_event_status"] == "provisional"
    corrupted = copy.deepcopy(default)
    corrupted.parts["gap_anvil"].notes[0].start += .1
    assert R._score_identity(corrupted) != R._score_identity(original)


def test_separate_gap_contact_moves_hit_but_keeps_editorial_silence():
    _stub_sampler_deps()
    from barmap_D_score import DraftMap
    import score_v3_D as D
    import verify_D as V
    bm = DraftMap()
    bm.events.append(dict(id="gap_hammer_lands", frame=3443, beat=3443 / 20, status="measured",
                          source="synthetic new-shot fixture", measured_ref="fixture://3443"))
    score = D.build(bm)
    assert score.parts["gap_anvil"].notes[0].start * 20 == pytest.approx(3443)
    assert (3400, 3440) in score.hard_silences
    assert any(hit["event"] == "gap_hammer_lands" for hit in score.sync_bindings)
    assert V.binding_problems(score, bm) == []
    broken = copy.deepcopy(score)
    broken.parts["gap_anvil"].notes[0].start = bm.ev("gap")
    assert any("gap_hammer_lands" in problem for problem in V.binding_problems(broken, bm))


def test_crossing_receipt_inherits_destination_status_and_evidence():
    _stub_sampler_deps()
    from barmap_D_score import DraftMap
    import score_v3_D as D
    bm = DraftMap()
    for event in bm.events:
        if event["id"].startswith("crossing_"):
            event.update(status="measured", source="AP2 fixture +1460",
                         measured_ref=f"fixture://{event['frame'] - 1460}")
    score = D.build(bm)
    source = score.crossing_source
    assert source["destination_event_status"] == "measured"
    assert len(source["destination_bindings"]) == 8
    for event in source["destination_bindings"]:
        assert event == bm.event(event["id"])
        assert event["measured_ref"].startswith("fixture://")
    # The receipt is a snapshot; later mutation of the map cannot rewrite it.
    bm.event("crossing_walk_second")["status"] = "estimated"
    assert source["destination_bindings"][2]["status"] == "measured"
    assert D.build(bm).crossing_source["destination_event_status"] == "mixed"


def test_audio_verification_carries_binding_table_provenance(tmp_path, monkeypatch):
    import verify_D as V
    path = tmp_path / "small.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setparams((2, 3, 48000, 0, "NONE", "not compressed"))
        fh.writeframes(bytes(2000 * 2 * 3))
    prefix = V.pcm_prefix_sha256
    monkeypatch.setattr(V, "pcm_prefix_sha256", lambda p: prefix(p, frames=1))
    monkeypatch.setattr(V, "lufs", lambda y: -16.)
    provenance = dict(table_path="music/v3/fixture.json", table_sha256="fixture-only",
                      source_sha256={"events": "fixture-source"})
    bm = SimpleNamespace(events=[], sections=[], binding_provenance=provenance)
    result = V.verify(path, path, SimpleNamespace(hard_silences=[], levels=[]), bm)
    assert result["binding_provenance"] == provenance


@pytest.fixture
def bound_receipt_case():
    """Synthetic musical markers and a receipt pinned to current render code."""
    import verify_D as V
    provenance = dict(table_path="fixture/current.json", table_sha256="table-original",
                      source_sha256={"editorial": "editorial-original"})
    events = [
        dict(id="crowns_kindle", frame=4320, status="estimated", source="fixture"),
        dict(id="gap_hammer_lands", frame=3440, status="estimated", source="fixture"),
        dict(id="giant_stroke_01", frame=2080, status="estimated", source="fixture"),
    ]
    bm = SimpleNamespace(events=events, sections=[], binding_provenance=provenance)
    here = Path(V.__file__).resolve().parent
    receipt = dict(measured=True, wav_sha256="a" * 64, events=copy.deepcopy(events),
                   binding_provenance=copy.deepcopy(provenance), peak_rss_bytes=1024,
                   code_sha256={name: V.file_sha256(here / name) for name in
                                ("render_D.py", "score_v3_D.py", "barmap_D_score.py", "bindings_D.py")})
    return bm, receipt


def _receipt_checks(receipt, bm, wav_sha256):
    import verify_D as V
    return {row["check"]: row for row in V.render_receipt_rows(receipt, bm, wav_sha256)}


def test_receipt_accepts_matching_wav_code_and_musical_frames(bound_receipt_case):
    bm, receipt = bound_receipt_case
    checks = _receipt_checks(receipt, bm, receipt["wav_sha256"])
    assert len(checks) == 4 and all(row["ok"] for row in checks.values())
    assert checks["render receipt musical bindings"]["table_hash_matches"] is True


def test_transition_receipt_requires_matching_option_and_module(bound_receipt_case):
    import verify_D as V
    bm, receipt = bound_receipt_case

    def checked(enabled):
        return {r["check"]: r for r in V.render_receipt_rows(
            receipt, bm, receipt["wav_sha256"], transition_pass=enabled)}

    assert not checked(True)["render receipt transition option"]["ok"]
    receipt["transition_pass"] = True
    assert not checked(True)["render receipt implementation"]["ok"]
    receipt["code_sha256"]["transitions_D.py"] = V.file_sha256(
        Path(V.__file__).parent / "transitions_D.py")
    assert all(r["ok"] for r in checked(True).values())
    assert not checked(False)["render receipt transition option"]["ok"]
    receipt["code_sha256"]["transitions_D.py"] = "f" * 64
    assert not checked(True)["render receipt implementation"]["ok"]


def test_renderer_cli_requires_explicit_transition_opt_in(monkeypatch):
    import render_D as R
    calls = []
    monkeypatch.setattr(R, "render", lambda *a, **kw: calls.append(kw))
    args = ["--render", "--output-dir", "fixture-output"]
    R.main(args)
    R.main(args + ["--transition-pass"])
    assert [call["transition_pass"] for call in calls] == [False, True]


@pytest.mark.parametrize("change", ["wrong_wav", "missing_hash", "not_measured"])
def test_receipt_rejects_unattested_wav(bound_receipt_case, change):
    bm, receipt = bound_receipt_case
    actual = receipt["wav_sha256"]
    if change == "wrong_wav":
        actual = "b" * 64
    elif change == "missing_hash":
        del receipt["wav_sha256"]
    else:
        receipt["measured"] = False
    checks = _receipt_checks(receipt, bm, actual)
    assert checks["render receipt WAV identity"]["ok"] is False
    assert checks["render receipt musical bindings"]["ok"] is True
    assert checks["render receipt implementation"]["ok"] is True


@pytest.mark.parametrize("event_id", ["crowns_kindle", "gap_hammer_lands"])
def test_receipt_rejects_a_moved_hit(bound_receipt_case, event_id):
    bm, receipt = bound_receipt_case
    next(event for event in bm.events if event["id"] == event_id)["frame"] += 1
    row = _receipt_checks(receipt, bm, receipt["wav_sha256"])["render receipt musical bindings"]
    assert row["ok"] is False
    assert row["changed_events"] == [event_id]


@pytest.mark.parametrize("which_side", ["current", "rendered"])
def test_receipt_rejects_changed_play_omit_action(bound_receipt_case, which_side):
    bm, receipt = bound_receipt_case
    events = bm.events if which_side == "current" else receipt["events"]
    next(event for event in events if event["id"] == "giant_stroke_01")["score_action"] = "omit"
    row = _receipt_checks(receipt, bm, receipt["wav_sha256"])["render receipt musical bindings"]
    assert row["ok"] is False
    assert row["changed_events"] == ["giant_stroke_01"]


@pytest.mark.parametrize("action", ["play", "omit"])
def test_receipt_accepts_matching_explicit_actions(bound_receipt_case, action):
    bm, receipt = bound_receipt_case
    for events in (bm.events, receipt["events"]):
        next(event for event in events if event["id"] == "giant_stroke_01")["score_action"] = action
    checks = _receipt_checks(receipt, bm, receipt["wav_sha256"])
    assert all(row["ok"] for row in checks.values())


def test_receipt_rejects_missing_musical_event(bound_receipt_case):
    bm, receipt = bound_receipt_case
    receipt["events"] = [event for event in receipt["events"] if event["id"] != "gap_hammer_lands"]
    row = _receipt_checks(receipt, bm, receipt["wav_sha256"])["render receipt musical bindings"]
    assert row["ok"] is False
    assert row["changed_events"] == ["gap_hammer_lands"]


@pytest.mark.parametrize("filename", ["render_D.py", "score_v3_D.py", "barmap_D_score.py", "bindings_D.py"])
@pytest.mark.parametrize("change", ["old_code", "missing_code"])
def test_receipt_rejects_old_or_unpinned_render_implementation(bound_receipt_case, filename, change):
    bm, receipt = bound_receipt_case
    if change == "old_code":
        receipt["code_sha256"][filename] = "0" * 64
    else:
        del receipt["code_sha256"][filename]
    checks = _receipt_checks(receipt, bm, receipt["wav_sha256"])
    assert checks["render receipt implementation"]["ok"] is False
    assert checks["render receipt implementation"]["changed_files"] == [filename]
    assert checks["render receipt WAV identity"]["ok"] is True
    assert checks["render receipt musical bindings"]["ok"] is True


@pytest.fixture
def small_audio_verification(tmp_path, monkeypatch):
    """Exercise real WAV hashing and receipts without a full-length audio render."""
    import verify_D as V
    path = tmp_path / "receipt-fixture.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setparams((2, 3, 48000, 0, "NONE", "not compressed"))
        fh.writeframes(bytes(2000 * 2 * 3))
    prefix = V.pcm_prefix_sha256
    monkeypatch.setattr(V, "SAMPLE_FRAMES", 2000)
    monkeypatch.setattr(V, "pcm_prefix_sha256", lambda p: prefix(p, frames=1))
    monkeypatch.setattr(V, "lufs", lambda y: -16.)
    monkeypatch.setattr(V, "true_peak_db", lambda y: -1.3)
    monkeypatch.setattr(V, "level_map", lambda *args: ([], V.np.zeros((0, 2))))
    return path, SimpleNamespace(hard_silences=[], levels=[])


def test_bound_audio_requires_a_render_receipt(bound_receipt_case, small_audio_verification):
    import verify_D as V
    bm, _ = bound_receipt_case
    path, score = small_audio_verification
    result = V.verify(path, path, score, bm)
    failures = [row for row in result["checks"] if not row["ok"]]
    assert result["ok"] is False
    assert failures == [dict(check="render receipt required for bound score", ok=False)]


def test_audio_verification_rejects_replaced_wav_bytes(
        bound_receipt_case, small_audio_verification, tmp_path):
    import verify_D as V
    bm, receipt = bound_receipt_case
    path, score = small_audio_verification
    receipt["wav_sha256"] = V.file_sha256(path)
    receipt_path = tmp_path / "original-render-receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    replaced = bytearray(path.read_bytes())
    replaced[-1] = 1  # A valid final PCM sample changes; the WAV header remains intact.
    path.write_bytes(replaced)
    result = V.verify(path, path, score, bm, receipt_path=receipt_path)
    failures = [row for row in result["checks"] if not row["ok"]]
    assert result["ok"] is False
    assert [row["check"] for row in failures] == ["render receipt WAV identity"]
    assert failures[0]["expected_sha256"] == receipt["wav_sha256"]
    assert failures[0]["actual_sha256"] == V.file_sha256(path)
    assert result["wav_sha256"] != receipt["wav_sha256"]


def test_metadata_promotion_reuses_audio_but_exposes_both_table_revisions(
        bound_receipt_case, small_audio_verification, tmp_path):
    import verify_D as V
    bm, receipt = bound_receipt_case
    path, score = small_audio_verification
    receipt["wav_sha256"] = V.file_sha256(path)
    original = copy.deepcopy(receipt["binding_provenance"])
    bm.binding_provenance.update(table_sha256="table-measured", table_path="fixture/measured.json")
    bm.events[0].update(status="measured", source="native fixture inspection",
                        measured_ref="fixture://crown-frame", measurement_scope="synthetic native frame")
    receipt_path = tmp_path / "render-receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    result = V.verify(path, path, score, bm, receipt_path=receipt_path)
    assert result["ok"] is True
    row = next(row for row in result["checks"] if row["check"] == "render receipt musical bindings")
    assert row["ok"] is True and row["changed_events"] == []
    assert row["table_hash_matches"] is False
    assert result["binding_provenance"] == bm.binding_provenance
    assert result["render_binding_provenance"] == original
    assert result["events"][0]["status"] == "measured"
    assert receipt["events"][0]["status"] == "estimated"
    assert result["wav_sha256"] == receipt["wav_sha256"]
