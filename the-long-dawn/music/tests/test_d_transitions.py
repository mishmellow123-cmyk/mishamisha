"""Opt-in transition orchestration preserves musical/event contracts.

Rendered amplitude improvements are tested on the delivered WAV separately;
notation tests do not assert an unrendered 6 dB result or a listening verdict.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import _stub_sampler_deps


@pytest.fixture(scope="module")
def scores_d_transition():
    _stub_sampler_deps()
    from barmap_D_score import DraftMap
    import score_v3_D as D
    bm = DraftMap()
    return D.build(bm), D.build(bm, transition_pass=True), bm


def test_transition_opt_in_keeps_default_phase1c_identity(scores_d_transition):
    import render_D as R
    import score_v3_D as D
    plain, active, bm = scores_d_transition
    assert not getattr(plain, "transition_pass", False)
    assert not hasattr(plain, "transition_manifest")
    assert R._score_identity(plain) == "c169666758acf37f99621b96adac750d013d250a95a3048ca5f05e4be1a9a1e9"
    assert R._score_identity(D.build(bm, transition_pass=False)) == R._score_identity(plain)
    assert R._score_identity(active) != R._score_identity(plain)
    assert active.transition_pass is True


def test_active_pass_satisfies_original_musical_rules(scores_d_transition):
    import score_v3_D as D
    import kit_v3 as K
    import verify_D as V
    _, active, bm = scores_d_transition
    assert D.check(active, bm) == []
    assert K.check_notes(active.used(), 460) == []
    assert K.check_rates(active.used()) == []
    assert V.ring_problems(active) == []
    assert V.call_problems(active, bm) == []
    assert V.watch_problems(active) == []
    assert V.beacon_problems(active) == []
    assert V.binding_problems(active, bm) == []


def test_original_starts_pitches_calls_watch_and_protected_regions_survive(scores_d_transition):
    plain, active, _ = scores_d_transition
    for name, part in plain.parts.items():
        assert [(n.start, n.pitch) for n in active.parts[name].notes] == [(n.start, n.pitch) for n in part.notes]
    for name in plain.watch_parts:
        assert active.parts[name].to_dict() == plain.parts[name].to_dict()
    assert active.hard_silences == plain.hard_silences
    assert active.pcm_regions == plain.pcm_regions
    assert active.sync_bindings == plain.sync_bindings
    assert active.motif_phrases == plain.motif_phrases


def test_only_second_call_receives_equal_authorized_gains_and_fade(scores_d_transition):
    plain, active, bm = scores_d_transition
    kindle, second = bm.ev("crowns_kindle"), bm.ev("two_fires")
    paired_phrases = []
    for name in plain.call_parts:
        before = plain.parts[name].to_dict()
        actual = active.parts[name].to_dict()
        expected = copy.deepcopy(before)
        original_kindle = [n for n in before["notes"] if kindle <= n["start"] < kindle + 4]
        actual_kindle = [n for n in actual["notes"] if kindle <= n["start"] < kindle + 4]
        assert len(original_kindle) == 3
        assert actual_kindle == original_kindle  # Includes first-note gain, fade and all kwargs.
        allowed = [n for n in expected["notes"] if second <= n["start"] < second + 4]
        phrase = [n for n in actual["notes"] if second <= n["start"] < second + 4]
        assert len(allowed) == len(phrase) == 3
        for note, delta_db in zip(allowed, (-8., -4., 0.)):
            note["gain_db"] += delta_db
        allowed[0]["kw"]["fadein"] = .5
        assert actual == expected  # Instrument, seed, pan, sync, duration and all other fields.
        paired_phrases.append(phrase)
    assert paired_phrases[0] == paired_phrases[1]
    assert [active.parts[name].pan for name in plain.call_parts] == [-1., 1.]


def test_each_treated_join_has_a_sustained_carrier_or_existing_attack_fade(scores_d_transition):
    _, active, _ = scores_d_transition
    joins = {row["cut"]: row for row in active.transition_manifest["joins"]}
    assert set(joins) == {2080, 3520, 3760, 4080, 4560, 5840, 6080, 6640}
    for cut, row in joins.items():
        assert row["window"][0] <= cut < row["window"][1]
        notes = [n for name in row["carriers"] for n in active.parts[name].notes]
        if cut == 3520:
            assert any(n.start * 20 == cut and n.kw.get("fadein", 0) > 0 for n in notes)
        else:
            assert any(n.start * 20 < cut < (n.start + n.dur) * 20 for n in notes)
    for name, part in active.parts.items():
        if not name.startswith("transition_"):
            continue
        for note in part.notes:
            start, end = note.start * 20, (note.start + note.dur) * 20
            assert start >= 1440
            assert all(end <= lo or start >= hi for lo, hi in active.hard_silences)


def _assert_release_fades(part):
    from dsl import dyn_at
    last = max(part.notes, key=lambda note: note.start)
    end_frame = (last.start + last.dur) * 20
    release_frames = last.kw["rel"] * 24
    probes = np.unique(np.r_[end_frame, end_frame + .005, end_frame + .01,
                             np.linspace(end_frame, end_frame + release_frames, 33)])
    levels = [dyn_at(part.dyn, frame / 20) for frame in probes]
    assert all(after <= before + 1e-10 for before, after in zip(levels, levels[1:])), levels


def test_carried_leader_fade_does_not_restore_gain_during_release(scores_d_transition):
    _, active, _ = scores_d_transition
    part = active.parts["trap_leaders"]
    _assert_release_fades(part)
    restored = copy.deepcopy(part)
    last = max(restored.notes, key=lambda note: note.start)
    restore_beat = last.start + last.dur + .01 / 20
    restored.dyn = sorted([(b, v) for b, v in restored.dyn if abs(b - restore_beat) > 1e-8]
                          + [(restore_beat, 1.)])
    with pytest.raises(AssertionError):
        _assert_release_fades(restored)


def test_forging_preroll_envelope_and_nominal_entry_are_quiet(scores_d_transition):
    from dsl import dyn_at
    plain, active, _ = scores_d_transition

    def assert_quiet(candidate):
        checked = 0
        for name in ("race_low", "race_trem"):
            before, after = plain.parts[name], candidate.parts[name]
            for original, treated in zip(before.notes, after.notes):
                if original.start * 20 != 2080:
                    continue
                checked += 1
                if original.vel is not None:
                    assert treated.vel <= original.vel * .38 + 1e-10
                # The sampler's anticipated sound precedes its nominal note.
                # Inspect the whole authored preroll envelope, not note.start.
                for frame in (2072, 2076, 2079, 2080):
                    beat = frame / 20
                    assert dyn_at(after.dyn, beat) <= dyn_at(before.dyn, beat) * .38 + 1e-10
        assert checked > 0

    assert_quiet(active)
    with pytest.raises(AssertionError):
        assert_quiet(plain)  # The untreated anticipations must fail.


def test_glow_common_tones_continue_across_the_crowns_join(scores_d_transition):
    _, active, _ = scores_d_transition

    def spans_both_cuts(score):
        tones = []
        for name in ("transition_glow_0", "transition_glow_1"):
            note, = score.parts[name].notes
            assert note.start * 20 < 4080 < 4240 < (note.start + note.dur) * 20
            tones.append(note.pitch)
        assert tones == [50, 69]  # D3 and A4; no claim about rendered level.

    spans_both_cuts(active)
    short = copy.deepcopy(active)
    note = short.parts["transition_glow_0"].notes[0]
    note.dur = 4240 / 20 - note.start
    with pytest.raises(AssertionError):
        spans_both_cuts(short)


@pytest.mark.parametrize("corruption,match", [
    ("attack", "original attack"),
    ("pitch", "original attack"),
    ("call_pan", "CALL pair"),
    ("kindle_gain", "CALL pair"),
    ("kindle_attack", "original attack"),
    ("second_gain_asymmetry", "CALL pair"),
    ("second_fade_asymmetry", "CALL pair"),
    ("watch_duration", "continuous WATCH"),
    ("silence", "protected silence"),
    ("opening", "opening PCM"),
    ("event", "picture binding"),
])
def test_protection_guards_fire_on_corrupted_transition(scores_d_transition, monkeypatch, corruption, match):
    import transitions_D as T
    plain, _, bm = scores_d_transition
    score, clock = copy.deepcopy(plain), copy.deepcopy(bm)
    original_bridge = T._bridge

    def faulty_bridge(S, stem, *args, **kwargs):
        result = original_bridge(S, stem, *args, **kwargs)
        if stem != "crossing":
            return result
        if corruption == "attack":
            S.parts["race_taiko"].notes[0].start += .1
        elif corruption == "pitch":
            S.parts["race_ring"].notes[0].pitch += 1
        elif corruption == "call_pan":
            S.parts["crowns_left"].pan = 0.
        elif corruption in ("kindle_gain", "kindle_attack"):
            note = next(n for n in S.parts["crowns_left"].notes
                        if n.start == clock.ev("crowns_kindle"))
            if corruption == "kindle_gain":
                note.gain_db -= .1
            else:
                note.start += .1
        elif corruption in ("second_gain_asymmetry", "second_fade_asymmetry"):
            note = next(n for n in S.parts["crowns_left"].notes
                        if n.start == clock.ev("two_fires"))
            if corruption == "second_gain_asymmetry":
                note.gain_db -= 1.
            else:
                note.kw["fadein"] = .4
        elif corruption == "watch_duration":
            S.parts["watch_D"].notes[0].dur -= .1
        elif corruption == "silence":
            S.hard_silences[0] = (3401, 3440)
        elif corruption == "opening":
            S.pcm_regions.append(dict(corrupt=True))
        else:
            clock.events[0]["frame"] += 1
        return result

    monkeypatch.setattr(T, "_bridge", faulty_bridge)
    with pytest.raises(ValueError, match=match):
        T.apply(score, clock)


def test_reapplication_and_wrong_grid_are_rejected(scores_d_transition):
    import transitions_D as T
    plain, active, bm = scores_d_transition
    with pytest.raises(ValueError, match="twice"):
        T.apply(copy.deepcopy(active), bm)
    with pytest.raises(ValueError, match="fixed v2 grid"):
        T.apply(copy.deepcopy(plain), SimpleNamespace(bars=114, frames=9200))


def test_missing_outgoing_carrier_fails_instead_of_inventing_an_attack(scores_d_transition):
    import transitions_D as T
    plain, _, bm = scores_d_transition
    corrupt = copy.deepcopy(plain)
    corrupt.parts["inscription_bed_cb_q"].notes = []
    with pytest.raises(ValueError, match="no existing note"):
        T.apply(corrupt, bm)


def test_existing_sync_and_ring_guards_still_reject_corruption_after_pass(scores_d_transition):
    import verify_D as V
    _, active, bm = scores_d_transition
    wrong_hit = copy.deepcopy(active)
    hit = next(x for x in wrong_hit.sync_bindings if x["event"] == "crowns_kindle")
    wrong_hit.parts[hit["part"]].notes[hit["note_index"]].start += .1
    assert any("UNBOUND HIT FRAME" in problem for problem in V.binding_problems(wrong_hit, bm))
    wrong_motif = copy.deepcopy(active)
    part = wrong_motif.P("unfinished_ring")
    first = part.notes[0]
    part.n(first.pitch + 6, first.start + 7, .5, .3)
    assert any("RING COMPLETION OUTSIDE VISION" in problem for problem in V.ring_problems(wrong_motif))
