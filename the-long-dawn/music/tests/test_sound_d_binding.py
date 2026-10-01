"""Round 4 effects binding: no audio or picture render."""
from copy import deepcopy
import hashlib
import json

import pytest

import sound_d_binding as B
import sound_d_table as T
from sound_d_designs import build as build_design, design


@pytest.fixture
def plan():
    return B.build()


def named(rows, name):
    return next(r for r in rows if r["id"] == "D.new." + name)


def test_exact_edl_selection_and_inherited_audio_clocks(plan):
    assert plan["edit_selection"] == {
        "crossing_start": 5180, "map_pieces": [[3440, 3740, 5180], [3740, 3816, 5554]]}
    reuse = plan["reuse_plan"]
    events = {r["id"]: r for r in reuse["EXTRA_EVENTS"]}
    assert events["D.21.C5.map.beacon8"]["hit_f"] == 5600
    assert events["D.30.C5.pen.tree"]["hit_f"] == 8160
    assert events["D.33.C5.burn.title"]["hit_f"] == 8915
    assert events["D.15.C5.pen.refusal_offer"]["hit_f"] == 3529
    assert all(r["source"] == "derived" for r in events.values() if r["id"] != "D.11.C5.burn.deep")
    assert events["D.11.C5.burn.deep"]["source"] == "est."
    assert events["D.11.C5.burn.deep"]["hit_f"] == 2739
    assert "rebaked" in events["D.11.C5.burn.deep"]["source_sync"]
    assert "measured:" in events["D.15.C5.pen.refusal_offer"]["source_sync"]
    cross = [r for r in reuse["BED_CROPS"] if r["id"].startswith("D.26.")]
    assert len(cross) == 2
    assert all(r["donor_crop"] == [5180, 5580] and (r["f0"], r["f1"]) == (6640, 7040) for r in cross)
    assert not any("watchfire_far" in r["id"] for r in cross)
    assert reuse["PCM_COPIES"][0]["donor_crop"] == [0, 1440]


def test_every_phase1_request_is_bound_once_or_expanded_honestly(plan):
    expected = {r["id"] for r in T.build()["requests"]}
    rows = plan["new_events"] + plan["new_beds"] + plan["controls"]
    assert {r["request_id"] for r in rows} == expected == set(plan["bound_request_ids"])
    assert len({r["id"] for r in rows}) == len(rows)
    for key, kind in (("new_events", "event"), ("new_beds", "bed"), ("controls", "control")):
        for r in plan[key]:
            assert design(r["design"])["kind"] == kind
            assert r["source"] == ("derived" if r["id"] in ("D.new.ring_burn", "D.new.map.burn") else "est.")
            assert r["picture_frame"] is None
            if r["source"] == "est.":
                assert r["measured_ref"] is None
            assert r["hook"] and r["hook_source"]
            assert (r.get("hit_f", r.get("f0"))) == r["hook_frame"] + r["offset_f"]
    burn = named(plan["new_events"], "ring_burn")
    assert burn["hit_f"] == 1683 and burn["stop_f"] == 1748
    assert burn["inherited_measurement_ref"] == "burn.first_open_on_sample_grid"
    assert "480" in burn["inherited_measurement_scope"]
    assert "native" in burn["timing_note"]
    mapped = named(plan["new_events"], "map.burn")
    assert (mapped["hit_f"], mapped["stop_f"], mapped["source"]) == (5183, 5212, "derived")
    assert mapped["measured_ref"] == "ridge_burn.first_open_on_sample_grid"
    assert mapped["recipe_overrides"]["pre"] == 0
    assert mapped["supersedes"]["hit_f"] == 5180
    assert plan["phase"] == 3 and len(plan["input_sha256"]) == 5


def test_hook_repeat_and_motion_estimates_keep_their_actual_basis(plan):
    giant = [r for r in plan["new_events"] if r["request_id"] == "D.new.forging.giants"]
    assert [r["hit_f"] for r in giant] == list(range(2120, 2400, 40))
    assert all(r["hook"] == "forging_hammer_strokes" and r["spacing_f"] == 40 for r in giant)
    feet = [r for r in plan["new_events"] if r["request_id"] == "D.new.crossing.feet"]
    assert [r["hit_f"] for r in feet] == [6672, 6712] + list(range(6736, 7040, 40))
    assert all(r["source"] == "est." and r["hook_timing_status"] == "inherited_measured" for r in feet)
    assert named(plan["new_events"], "holdout.hammer")["hit_f"] == 5496
    assert not any(r["hook"] == "hammer_alone" for r in plan["new_events"])


def test_constraints_and_controls_cover_tails_and_voice(plan):
    c = plan["constraints"]
    assert c["vision_silence"] == [3400, 3440]
    assert c["voice_region"] == [8640, 8880] and c["voice_peak_dbfs"] == -40
    assert c["voice_silence"] == [8660, 8740]
    assert c["enforce_after_wet_and_master"] and c["final_zero_frame"] == 9200
    crown = [named(plan["new_events"], "crowns." + side) for side in ("left", "right")]
    assert [(r["hit_f"], r["pan"], r["dist"]) for r in crown] == [(4320, -1.0, 0), (4320, 1.0, 0)]
    assert all(r["recipe_overrides"]["send"] == 0 for r in crown)
    assert named(plan["new_beds"], "vision.pressure")["f1"] == 3400
    assert named(plan["new_events"], "gap.hammer")["hit_f"] == 3440
    assert named(plan["controls"], "inscription.stop")["targets"] == ["D.new.inscription.hiss", "D.new.inscription.ring"]
    leaf = named(plan["controls"], "leaf.stop")
    assert leaf["hit_f"] == 8500 and leaf["targets"] == ["D.new.leaf.quill"]
    assert leaf["includes_wet"]
    assert named(plan["new_events"], "leaf.quill")["stop_f"] == leaf["hit_f"]
    assert named(plan["new_events"], "page.to_blank")["stop_f"] == 8660
    assert named(plan["new_beds"], "hearth.out")["f1"] == 9200
    wind = named(plan["new_beds"], "watch.wind")
    dawn = named(plan["new_beds"], "dawn.wind")
    assert wind["f1"] > dawn["f0"]  # real overlap across7360, not paired fade-to-zero


def test_crowns_use_distinct_established_anchors_with_identical_processing(plan):
    import sound_recipes_C as C
    recipes = []
    for side, donor, expected in (("left", "C.beacon1", ("fs:595483", 9.944)),
                                  ("right", "C.beacon2", ("fs:595483", 26.558))):
        row = named(plan["new_events"], "crowns." + side)
        rc = build_design(row["design"])
        rc.update(deepcopy(row["recipe_overrides"]))
        assert rc["layers"][0]["src"] == C.RECIPES[donor]["layers"][0]["src"] == expected
        assert rc["layers"][0]["pre"] == 0 and rc["send"] == 0 and rc["no_breath"]
        assert row["hit_f"] == 4320 and row["source"] == "est."
        assert row["source_choice_provenance"]["anchor"] == expected
        assert row["source_choice_provenance"]["source_recipe"].endswith(donor + ".layers[0].src")
        recipes.append(rc)
    assert recipes[0]["layers"][0]["src"] != recipes[1]["layers"][0]["src"]
    recipes[1]["layers"][0]["src"] = recipes[0]["layers"][0]["src"]
    assert recipes[0] == recipes[1]


def test_first_render_response_trims_keep_measured_mix_provenance(plan):
    expected = [(named(plan["new_events"], "crowns." + side), -7, "D18", -4) for side in ("left", "right")]
    expected += [(named(plan["new_beds"], "crowns.fire"), -5, "D18", -2),
                 (named(plan["new_beds"], "ridges.wind"), -3, "D19", -3)]
    catches = [r for r in plan["new_events"] if r["request_id"] == "D.new.ridges.catches"]
    for row in catches:
        original = -8 if ".pair" in row["id"] else -8 + 2 * (int(row["id"].rsplit("answer", 1)[1]) - 1)
        expected.append((row, original - 3, "D19", -3))
    for row, trim, section, delta in expected:
        assert row["level_trim_db"] == trim
        assert row["source"] == "est."
        p = row["level_adjustment_provenance"]
        assert p["kind"] == "authored mix adjustment" and p["trim_delta_db"] == delta
        assert p["reference_artifact"] == "render_effects_D_first_receipt.json" and p["section"] == section
        assert p["measured_relative_max_lu"] == {"D18": -1.96378392041502, "D19": -2.4880370780276184}[section]
        assert p["permitted_relative_max_lu"] == -2.5
        assert len(p["reference_sha256"]) == len(p["measured_mix_sha256"]) == 64
        if section == 'D18':
            revision = row['phase3_level_adjustment_provenance']
            assert revision['trim_delta_db'] == -3
            assert trim == delta + revision['trim_delta_db']
            assert revision['measured_relative_max_lu'] > revision['permitted_relative_max_lu']
            assert revision['reference_artifact'].startswith('phase3/')
            assert revision['one_db_trial']['passed'] is False
        else:
            assert 'phase3_level_adjustment_provenance' not in row


@pytest.mark.parametrize("mutation", ["new_measured", "missing_request", "shift_hit", "voice_louder",
                                    "shorter_silence", "crown_send", "crown_same_source", "crown_trim", "lost_donor_clock", "changed_hook"])
def test_negative_binding_changes_fail_guard(plan, mutation):
    if mutation == "new_measured":
        plan["new_events"][0]["source"] = "measured"
    elif mutation == "missing_request":
        plan["new_events"] = [r for r in plan["new_events"] if r["request_id"] != "D.new.ring_burn"]
    elif mutation == "shift_hit":
        plan["new_events"][0]["hit_f"] += 2
    elif mutation == "voice_louder":
        plan["constraints"]["voice_peak_dbfs"] = -39
    elif mutation == "shorter_silence":
        plan["constraints"]["vision_silence"] = [3401, 3440]
    elif mutation == "crown_send":
        named(plan["new_events"], "crowns.left")["recipe_overrides"]["send"] = .08
    elif mutation == "crown_same_source":
        named(plan["new_events"], "crowns.right")["recipe_overrides"]["layers"] = build_design("crown_kindle")["layers"]
    elif mutation == "crown_trim":
        named(plan["new_events"], "crowns.left")["level_trim_db"] = 0
    elif mutation == "lost_donor_clock":
        plan["reuse_plan"]["BED_CROPS"][0]["donor_crop"][0] += 1
    elif mutation == "changed_hook":
        plan["new_events"][0]["hook"] = "drawn_ring_burn"
    assert B.problems(plan)


@pytest.mark.parametrize("mutation", ["crossing_offset", "map_hold", "book_clock", "burn_clock", "old_edl"])
def test_wrong_editorial_source_cannot_silently_reuse_audio(mutation):
    edl = json.loads((T.ROOT / B.FILES["edl"]).read_text())
    shots = {r["code"]: r for r in edl["shots"]}
    if mutation == "crossing_offset":
        shots["D26"]["takes"][0]["off"] -= 1
    elif mutation == "map_hold":
        shots["D21c"]["takes"][0]["hold"] += 1
    elif mutation == "book_clock":
        shots["D30"]["takes"][0]["off"] = -2880
    elif mutation == "burn_clock":
        next(t for t in edl["transitions"] if t.get("cover") == "x1_refusal_C5_cover")["layer_off"] -= 1
    elif mutation == "old_edl":
        edl["frames"] = 5920
    with pytest.raises(ValueError):
        B.build(edl=edl)


@pytest.mark.parametrize("mutation", ["missing_hook", "missing_cue", "repeat", "crown_frame", "false_cue_provenance"])
def test_wrong_hook_metadata_fails_before_sound_binding(mutation):
    bm = json.loads((T.ROOT / B.FILES["barmap"]).read_text())
    cues = json.loads((T.ROOT / B.FILES["cues"]).read_text())
    hooks = {r["id"]: r for r in bm["sync"]}
    if mutation == "missing_hook":
        bm["sync"].remove(hooks["inscription_scripts"])
    elif mutation == "missing_cue":
        cues["events"].pop("inscription_scripts")
    elif mutation == "repeat":
        hooks["forging_hammer_strokes"]["repeat"]["spacing_frames"] = None
    elif mutation == "crown_frame":
        hooks["crown_beacons"]["f"] += 1
    elif mutation == "false_cue_provenance":
        cues["events"]["drawn_ring_burn_opens"]["timing_status"] = "measured_finished_picture"
    with pytest.raises(ValueError):
        B.build(barmap=bm, cues=cues)


def test_bound_copies_and_json_roundtrip_are_stable(plan):
    assert B.problems(plan) == []
    assert B.problems(json.loads(json.dumps(plan))) == []
    again = B.build()
    plan["new_events"][0]["recipe_overrides"]["pre"] = 99
    assert again["new_events"][0]["recipe_overrides"]["pre"] == 0
    supplied = {k: json.loads((T.ROOT / p).read_text()) for k, p in B.FILES.items()}
    before = deepcopy(supplied)
    B.build(**supplied)
    assert supplied == before


def test_build_preserves_existing_files_and_donor_recipe_objects():
    import sound_recipes_A as A
    import sound_recipes_C as C
    paths = [T.ROOT / p for p in (*T.INPUTS, *B.FILES.values())]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    recipes = deepcopy((A.RECIPES, C.RECIPES))
    B.build()
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert recipes == (A.RECIPES, C.RECIPES)
