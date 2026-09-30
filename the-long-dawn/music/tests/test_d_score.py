"""D phase 1c: actual notes, provisional bindings, and measured-audio predicates.

Every musical guard has a corruption control. No source samples or shared cache
are needed; full-WAV measurements are the separate onepy-guarded verify_D CLI.
"""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from conftest import ROOT, SRC, _stub_sampler_deps


@pytest.fixture(scope="module")
def built_d():
    _stub_sampler_deps()
    from barmap_D_score import DraftMap
    import score_v3_D as D
    bm = DraftMap()
    return D.build(bm), bm


def test_draft_grid_is_contiguous_and_keeps_the_midbar_map_cut():
    from barmap_D_score import DraftMap
    bm = DraftMap()
    spans = [(s["f0"], s["f1"]) for s in bm.sections]
    assert len(spans) == 34 and spans[0][0] == 0 and spans[-1][1] == 9200
    assert all(a < b for a, b in spans)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert bm.bars * 80 == bm.frames == 9200
    assert bm.section("D21")["f0"] == bm.event("map")["frame"] == 5180
    assert {e["status"] for e in bm.events} == {"provisional"}
    assert len({e["id"] for e in bm.events}) == len(bm.events)


def test_adapter_does_not_write_cutd_files():
    from barmap_D_score import DraftMap
    files = [Path(ROOT) / "music/v3" / name for name in ("barmap_D.json", "cues_D.json")]
    before = [p.read_bytes() for p in files]
    DraftMap({"crowns_kindle": dict(frame=4321, status="measured", source="test image",
                                    measured_ref="test://frame4321")})
    assert [p.read_bytes() for p in files] == before


def test_measured_binding_requires_evidence_and_stays_in_its_bar():
    from barmap_D_score import DraftMap
    update = dict(frame=4321, status="measured", source="unit-test picture", measured_ref="test://4321")
    assert DraftMap({"crowns_kindle": update}).event("crowns_kindle")["status"] == "measured"
    for eid, bad in (("crowns_kindle", dict(update, frame=4400)),
                     ("crowns", dict(update, frame=4241)),
                     ("unknown", update),
                     ("crowns_kindle", {k: v for k, v in update.items() if k != "measured_ref"}),
                     ("crowns_kindle", {k: v for k, v in update.items() if k != "source"})):
        with pytest.raises(ValueError):
            DraftMap({eid: bad})


def test_complete_score_passes_rules_and_rate_checks(built_d):
    import score_v3_D as D
    import kit_v3 as K
    import verify_D as V
    s, bm = built_d
    assert D.check(s, bm) == []
    assert K.check_notes(s.used(), 460) == []
    assert K.check_rates(s.used()) == []
    assert V.ring_problems(s) == V.call_problems(s, bm) == V.watch_problems(s) == V.beacon_problems(s) == []
    assert V.binding_problems(s, bm) == []
    assert {r[0] for r in s.levels} == {sec["id"] for sec in bm.sections}


def test_withheld_note_is_a_gesture_not_a_pitch_class_ban(built_d):
    import verify_D as V
    s, _ = built_d
    hits = V.ring_completions(s)
    assert [(round(h["start_frame"]), round(h["terminal_frame"])) for h in hits] == [
        (1480, 1550), (1560, 1630), (3200, 3380), (5860, 6000)]
    assert V.ring_problems(s) == []
    harmless = copy.deepcopy(s)
    harmless.P("unfinished_ring").n("Ab4", 6200 / 20, .2, .2)
    assert V.ring_problems(harmless) == []
    missing = copy.deepcopy(s)
    missing.P("vision_stopped").notes.pop()
    assert any(x.startswith("VISION COMPLETION COUNT") for x in V.ring_problems(missing))


@pytest.mark.parametrize("split,root", [(False, 64), (True, 67)])
def test_extra_transposed_or_split_ring_completion_is_rejected(built_d, split, root):
    import verify_D as V
    s = copy.deepcopy(built_d[0])
    # New parts deliberately do not enter motif_phrases or ring_families.
    a = s.add("injected_a", "vla_q")
    b = s.add("injected_b", "vc_q")
    for i, (iv, offset) in enumerate(zip((0, 6, 12, 10, 6), (0, 1, 2, 3, 3.5))):
        (b if split and i % 2 else a).n(root + iv, 6100 / 20 + offset, .4, .3)
    assert any(x.startswith("RING COMPLETION OUTSIDE VISION") for x in V.ring_problems(s))


def test_ring_terminal_added_to_existing_unfinished_phrase_is_rejected(built_d):
    import verify_D as V
    s = copy.deepcopy(built_d[0])
    part = s.P("unfinished_ring")
    first = part.notes[0]
    part.n(first.pitch + 6, first.start + 7, .5, .3)
    assert any(x.startswith("RING COMPLETION OUTSIDE VISION") for x in V.ring_problems(s))


@pytest.mark.parametrize("event", ["crowns_kindle", "two_fires"])
def test_call_pair_unison_on_event_and_negative_controls(built_d, event):
    import verify_D as V
    s, bm = built_d
    assert V.call_problems(s, bm) == []
    early = copy.deepcopy(s)
    n = next(n for n in early.P(s.call_parts[1]).notes if n.start == bm.ev(event))
    n.start -= .1
    assert any(x.startswith("CALL OFF EVENT") for x in V.call_problems(early, bm))
    wrong_pitch = copy.deepcopy(s)
    n = next(n for n in wrong_pitch.P(s.call_parts[1]).notes if n.start == bm.ev(event))
    n.pitch += 1
    assert any(x.startswith("CALL NOT UNISON") for x in V.call_problems(wrong_pitch, bm))
    centered = copy.deepcopy(s)
    centered.P(s.call_parts[0]).pan = 0
    assert any(x.startswith("CALL PAN") for x in V.call_problems(centered, bm))


def test_named_hit_binding_detects_note_drift(built_d):
    import verify_D as V
    s, bm = built_d
    assert s.sync_bindings and V.binding_problems(s, bm) == []
    drift = copy.deepcopy(s)
    ref = next(r for r in drift.sync_bindings if r["event"] == "crowns_kindle")
    drift.parts[ref["part"]].notes[ref["note_index"]].start += .1
    assert any(x.startswith("UNBOUND HIT FRAME") for x in V.binding_problems(drift, bm))


def test_measured_crown_and_vision_binding_moves_actual_notes(built_d):
    import score_v3_D as D
    import verify_D as V
    from barmap_D_score import DraftMap
    bm = DraftMap({name: dict(frame=f, status="measured", source="unit test",
                              measured_ref=f"test://{f}")
                   for name, f in (("crowns_kindle", 4322), ("vision_complete", 3382))})
    s = D.build(bm)
    assert D.check(s, bm) == []
    assert V.call_problems(s, bm) == V.binding_problems(s, bm) == []
    vision = [h for h in V.ring_completions(s) if 3200 <= h["start_frame"] < 3400]
    assert len(vision) == 1 and vision[0]["terminal_frame"] == pytest.approx(3382, abs=.001)


def test_watch_reaches_through_crossing_and_dawn(built_d):
    import verify_D as V
    s = built_d[0]
    assert V.watch_problems(s) == []
    missing = copy.deepcopy(s)
    del missing.parts[s.watch_parts[0]].notes[15]
    assert any(x.startswith("WATCH GAP") for x in V.watch_problems(missing))


def test_crossing_uses_cutd_round3_interval_and_offset(built_d):
    """Cutd's final Round 3 report selects A5180..5579, replacing the provisional excerpt."""
    s = built_d[0]

    def matches_report(score):
        source = score.crossing_source
        assert source["source"] == "AP2" and (source["f0"], source["f1"]) == (5180, 5580)
        provenance = " ".join(str(source.get(key, "")) for key in ("reason", "selection_source")).lower()
        assert "cutd" in provenance and "round 3" in provenance
        reuse = [r for r in score.reuse if r["prefix"] == "crossing"]
        assert len(reuse) == 1
        assert (reuse[0]["source"], reuse[0]["source_f0"], reuse[0]["source_f1"],
                reuse[0]["destination_f0"]) == ("AP2", 5180, 5580, 6640)
        assert reuse[0]["destination_f0"] - reuse[0]["source_f0"] == 1460

    matches_report(s)
    wrong_interval = copy.deepcopy(s)
    wrong_interval.crossing_source.update(f0=4880, f1=5280)
    with pytest.raises(AssertionError):
        matches_report(wrong_interval)
    wrong_offset = copy.deepcopy(s)
    next(r for r in wrong_offset.reuse if r["prefix"] == "crossing")["destination_f0"] += 300
    with pytest.raises(AssertionError):
        matches_report(wrong_offset)


def test_crossing_retains_ap2_steps_and_narrow_stretch_rest(built_d):
    import score_v3_AP2 as AP2
    import verify_D as V
    s = built_d[0]
    expected = [6640, 6700, 6740, 6780, 6980, 7020]

    def walk_notes(score, base):
        names = ["crossing_" + base, "crossing_" + AP2.GO[base]]
        return sorted([n for pn in names if pn in score.parts for n in score.parts[pn].notes
                       if n.pitch is not None and n.gain_db > -120], key=lambda n: n.start)

    def keeps_walk(score):
        for base in AP2.WALK:
            notes = walk_notes(score, base)
            assert [n.start * 20 for n in notes] == pytest.approx(expected)
            assert all(n.start * 20 == pytest.approx(n.kw["donor_start_beat"] * 20 + 1460) for n in notes)
            assert not [n for n in notes if n.start * 20 < 6980 and (n.start + n.dur) * 20 > 6820 + 1e-5]
        assert V.watch_problems(score) == []

    keeps_walk(s)
    intruder = copy.deepcopy(s)
    intruder.parts["crossing_feet"].n(60, 6900 / 20, .5, .2, donor_start_beat=5440 / 20)
    with pytest.raises(AssertionError):
        keeps_walk(intruder)
    shifted = copy.deepcopy(s)
    for base in AP2.WALK:
        for note in walk_notes(shifted, base):
            note.start += 300 / 20  # the retired +1760-frame offset
    with pytest.raises(AssertionError):
        keeps_walk(shifted)


def test_trap_dropout_and_return_follow_off_grid_bindings():
    import score_v3_D as D
    from barmap_D_score import DraftMap
    bm = DraftMap({name: dict(frame=f, status="measured", source="unit test",
                              measured_ref=f"test://{f}")
                   for name, f in (("trap_drop", 3843), ("trap_return", 3963))})
    s = D.build(bm)
    lo, hi = bm.ev("trap_drop"), bm.ev("trap_return")

    def respects_dropout(score):
        notes = score.parts[D.TRAP_VOICE].notes
        assert not [n for n in notes if n.start < hi and n.start + n.dur > lo]
        assert min(n.start for n in notes if n.start >= hi) == pytest.approx(hi)
        # Other forge desks remain active during the dropout.
        for name in ("trap_others", "trap_bass", "trap_taiko"):
            assert [n for n in score.parts[name].notes if lo <= n.start < hi]

    respects_dropout(s)
    broken = copy.deepcopy(s)
    broken.parts[D.TRAP_VOICE].n("D3", lo + .2, .4, .3)
    with pytest.raises(AssertionError):
        respects_dropout(broken)
    broken = copy.deepcopy(s)
    broken.parts["trap_others"].notes = [n for n in broken.parts["trap_others"].notes if not lo <= n.start < hi]
    with pytest.raises(AssertionError):
        respects_dropout(broken)


def test_full_beacon_waits_for_dawn_even_if_distributed(built_d):
    import verify_D as V
    s = copy.deepcopy(built_d[0])
    assert V.beacon_problems(s) == []
    # C illumination's whole theme moved to an early scene, passed from horn to strings.
    a, b = s.add("early_beacon_call", "hn"), s.add("early_beacon_answer", "vln1")
    pitches = (62, 69, 74, 74, 73, 71, 66, 71, 69, 66, 62)
    times = (0, 2, 4, 6, 8, 9, 10, 14, 16, 18, 20)
    for i, (pitch, t) in enumerate(zip(pitches, times)):
        (a if i < 3 else b).n(pitch, 4500 / 20 + t, .5, .3)
    assert any(x.startswith("FULL BEACON BEFORE DAWN") for x in V.beacon_problems(s))


def test_hard_silence_note_and_master_contracts(built_d):
    import score_v3_D as D
    s, bm = built_d
    assert s.hard_silences == [(3400, 3440), (8660, 8740), (9120, 9200)]
    extra = copy.deepcopy(s)
    extra.P("vision_stopped").n("D4", 3420 / 20, .1, .3)
    assert any(x.startswith("NOTE IN HARD SILENCE") for x in D.check(extra, bm))
    mask = copy.deepcopy(s)
    mask.hard_silences.pop(0)
    assert "HARD SILENCE MASTER MASK MISSING" in D.check(mask, bm)


def test_silence_audio_guard_fires_on_a_single_nonzero_sample():
    from verify_D import FRAME_N, silence_row
    quiet = np.zeros((40 * FRAME_N, 2), np.float32)
    assert silence_row(quiet, 3400, 3440, first_frame=3400)["ok"]
    quiet[20 * FRAME_N, 1] = 2 ** -23  # one 24-bit least-significant bit
    assert not silence_row(quiet, 3400, 3440, first_frame=3400)["ok"]
    assert not silence_row(quiet[:-1], 3400, 3440, first_frame=3400)["ok"]


def test_relative_level_band_guard_fires_after_real_gain_change():
    from verify_D import SR, st_loudness, level_rows
    t = np.arange(16 * SR, dtype=np.float64) / SR
    tone = np.sin(2 * np.pi * 440 * t)
    tone[:8 * SR] *= .25
    tone[8 * SR:] *= .5
    sound = np.column_stack([tone, tone]).astype(np.float32)
    sections = [dict(id="quiet", f0=0, f1=8 * 24), dict(id="anchor", f0=8 * 24, f1=16 * 24)]
    # Use interior sections here so the transition does not set the quiet region's max.
    sections[0]["f1"] = 7 * 24
    sections[1]["f0"] = 9 * 24
    bands = [("quiet", -8, -4), ("anchor", -1, 0)]
    assert all(row["ok"] for row in level_rows(st_loudness(sound), sections, bands))
    sound[:8 * SR] *= 4
    assert not level_rows(st_loudness(sound), sections, bands)[0]["ok"]
    sound[:8 * SR] *= .001
    assert not level_rows(st_loudness(sound), sections, bands)[0]["ok"]
    assert not level_rows(np.zeros((0, 2)), sections, bands)[0]["ok"]


def test_declared_silence_level_window_excludes_previous_cue_and_keeps_legacy_values():
    from verify_D import FRAME_N, level_rows, silence_row
    track = np.array([(379.5, -10), (380, -30), (380.5, -50), (381, -65), (381.5, -120)], float)
    sections = [dict(id="D34", f0=9120, f1=9200)]
    band = [("D34", -120, -60)]
    original = level_rows(track, sections, band)[0]
    assert not original["ok"] and original["window_policy"] == "engine_1s_interior"
    scoped = level_rows(track, sections, band, [(9120, 9200)])[0]
    assert scoped["ok"] and scoped["window_policy"] == "fully_contained_in_declared_silence"
    assert scoped["band_lu"] == original["band_lu"] == [-120, -60]
    for field in ("windows", "max_lufs", "median_lufs", "relative_max_lu", "relative_median_lu", "ok"):
        assert scoped["legacy_engine_window"][field] == original[field]
    # Loudness inside the retained whole window still fails the unchanged band.
    bad = track.copy()
    bad[-1, 1] = -50
    assert not level_rows(bad, sections, band, [(9120, 9200)])[0]["ok"]
    # Energy even outside that window fails the separate full-interval PCM test.
    audio = np.zeros((80 * FRAME_N, 2), np.float32)
    assert silence_row(audio, 9120, 9200, first_frame=9120)["ok"]
    audio[0, 0] = 2 ** -23
    assert not silence_row(audio, 9120, 9200, first_frame=9120)["ok"]


def test_partial_silence_does_not_change_normal_section_level_mask():
    from verify_D import level_rows
    track = np.array([(360.5, -20), (361, -21), (361.5, -25), (362, -35), (362.5, -120),
                      (363, -65), (363.5, -25), (364, -20), (368, -25)], float)
    sections = [dict(id="D32", f0=8640, f1=8880)]
    band = [("D32", -40, -5)]
    assert level_rows(track, sections, band, [(8660, 8740)]) == level_rows(track, sections, band)


def test_true_peak_chunk_edges_match_full_polyphase_reference():
    from scipy import signal
    from verify_D import true_peak_db
    rng = np.random.default_rng(1040)
    audio = rng.normal(size=(2600, 2)).astype(np.float32) * .1
    full = signal.resample_poly(audio.astype(np.float64), 4, 1, axis=0)
    expected = 20 * np.log10(float(np.float32(np.abs(full).max())) + 1e-12)
    assert true_peak_db(audio, chunk=800) == pytest.approx(expected, abs=1e-6)
    assert true_peak_db(audio * 8, chunk=800) > -1.2  # ceiling control


def test_opening_identity_uses_packed_pcm_and_detects_one_changed_byte(tmp_path):
    from verify_D import pcm_prefix_sha256
    # One video frame is sufficient to exercise the exact same packed-byte helper.
    paths = [tmp_path / name for name in ("a.wav", "b.wav")]
    data = bytearray(2000 * 2 * 3)
    for path in paths:
        with wave.open(str(path), "wb") as fh:
            fh.setparams((2, 3, 48000, 0, "NONE", "not compressed"))
            fh.writeframes(data)
    assert pcm_prefix_sha256(paths[0], 1) == pcm_prefix_sha256(paths[1], 1)
    data[-1] = 1
    with wave.open(str(paths[1]), "wb") as fh:
        fh.setparams((2, 3, 48000, 0, "NONE", "not compressed"))
        fh.writeframes(data)
    assert pcm_prefix_sha256(paths[0], 1) != pcm_prefix_sha256(paths[1], 1)


def test_verification_receipt_rejects_wav_changed_during_measurement(tmp_path, monkeypatch):
    import verify_D as V
    path = tmp_path / "small.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setparams((2, 3, 48000, 0, "NONE", "not compressed"))
        fh.writeframes(bytes(2000 * 2 * 3))
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    # This small fixture intentionally fails the film-length check; exercise the
    # file/measurement binding without rendering a six-minute test soundtrack.
    prefix = V.pcm_prefix_sha256
    monkeypatch.setattr(V, "pcm_prefix_sha256", lambda p: prefix(p, frames=1))
    monkeypatch.setattr(V, "lufs", lambda y: -16.)
    score = SimpleNamespace(hard_silences=[], levels=[])
    bm = SimpleNamespace(events=[], sections=[])
    stable = V.verify(path, path, score, bm)
    assert stable["wav_sha256"] == original
    assert next(r for r in stable["checks"] if r["check"] == "WAV unchanged during verification")["ok"]

    def change_file_after_decode(y):
        with open(path, "r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"\x01")
        return -16.

    monkeypatch.setattr(V, "lufs", change_file_after_decode)
    changed = V.verify(path, path, score, bm)
    receipt = next(r for r in changed["checks"] if r["check"] == "WAV unchanged during verification")
    assert not changed["ok"] and not receipt["ok"]
    assert changed["wav_sha256"] is None
    assert receipt["before_sha256"] == original
    assert receipt["after_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() != original


def test_ac_score_source_bytes_match_pre_d_lane_receipt():
    # Taken from scored's source_inventory.json before any D implementation.
    baseline = {
        "score_v3_A.py": "3fc11ea1ee6ccbf15bcf9720101d601836d4c9313ce23d3499d0a43c7c7940fd",
        "score_v3_AP2.py": "a44aecc4826434dc44a8b9e7d322bb34fb0d2ede4bc27a069392ea7d7275bf06",
        "score_v3_C.py": "4f7506edab0ee9215f685b4f20bc6b4eb9e3f8a4a9473d5fb6d21b7cd94b1970",
        "score_v3_C5.py": "a18ff3ec51a0eadc3b18e8f8e73bc578fabc4bbae390b434e49cdf8f5f4764b1",
        "score_v3_C5P2.py": "ce8d1e445749bc96a311b0e85a2d34dedb5ff4cf6eee294f3fc61d6a5c26f588",
    }
    for filename, expected in baseline.items():
        data = (Path(SRC) / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        assert hashlib.sha256(data + b"\n").hexdigest() != expected
