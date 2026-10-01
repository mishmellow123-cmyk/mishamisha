"""Phase 1 provenance/crop contracts; no fabricated D measurement or audio render."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import sound_d_table as T
import sound_recipes_D_effects as R


@pytest.fixture
def table():
    return T.build()


def row(table, suffix):
    return next(r for r in table["reuse"] if r["id"].endswith(suffix))


def test_v2_covers_every_frame_and_preserves_offbar_map_entry(table):
    assert T.problems(table) == []
    assert len(table["beats"]) == 34
    assert table["beats"][0]["f0"] == 0
    assert table["beats"][-1]["f1"] == 9200
    assert all(a["f1"] == b["f0"] for a, b in zip(table["beats"], table["beats"][1:]))
    assert table["beats"][20]["f0"] == 5180
    assert table["frames"] * T.SAMPLES_PER_FRAME == 18_400_000


def test_reuse_uses_audio_clock_and_keeps_actual_source_provenance(table):
    expected = {"C5.riffle": 1446, "C5.pen.mountain": 1475, "C5.pen.T1": 1520,
                "C5.burn.deep": 2739, "C5.pen.deep": 2748,
                "C5.pen.refusal_offer": 3529, "C5.pen.refusal_figure": 3596,
                "C5.burn.to_trap": 3770, "C5.pen.tree": 8160, "C5.burn.title": 8915}
    for rid, expected_frame in expected.items():
        r = row(table, rid)
        assert r["hit_f"] == expected_frame
        assert r["source"] == "derived" and r["source_sync"] == r["donor"]["sync"]
    assert "contact sheet" in row(table, "C5.pen.tree")["source_sync"]
    assert row(table, "C5.pen.refusal_offer")["donor"]["dur_f"] == 62
    assert row(table, "C5.pen.refusal_figure")["donor"]["dur_f"] == 94
    assert row(table, "C5.riffle")["review"]  # original measurement predates candidate
    assert not any(r.get("donor", {}).get("id") == "C5.pen.T7" for r in table["reuse"])


def test_ridge_inventory_is_complete_with_original_pan_and_relative_levels(table):
    import sound_a_table as A
    donor = {r["id"]: r for r in A.build()["events"] if r["id"].startswith("A.ridge.")}
    actual = {r["donor"]["id"]: r for r in table["reuse"] if r.get("donor", {}).get("id", "").startswith("A.ridge.")}
    assert len(actual) == len(donor) == 45
    for rid, d in donor.items():
        assert actual[rid]["donor"] == d
        assert actual[rid]["hit_f"] == d["hit_f"] + 1380
    assert min(r["hit_f"] for r in actual.values()) == 5066
    assert max(r["hit_f"] for r in actual.values()) == 5118


def test_new_shots_do_not_inherit_c_hammers_or_fabricate_times(table):
    assert all(r["source"] == "est." and r["hit_f"] is None and r["f0"] is None and r["f1"] is None
               and r["measured_ref"] is None for r in table["requests"])
    crowns = [r for r in table["requests"] if r["id"].startswith("D.new.crowns.") and r["kind"] == "event"]
    assert [(r["prescribed_f"], r["pan"]) for r in crowns] == [(4320, -1.0), (4320, 1.0)]
    assert not any(r.get("donor", {}).get("id", "").startswith("C5.hammer") for r in table["reuse"])


def test_defaults_leave_map_and_crossing_unbound(table):
    pending = [r for r in table["reuse"] if r["status"] == "awaiting_edit"]
    assert len(pending) == 10
    assert all(r.get("hit_f") is None and r.get("f0") is None and r.get("f1") is None for r in pending)
    assert all(r["selection"] is None for r in pending if r["beat"] == 26)
    with pytest.raises(ValueError, match="unbound rows"):
        R.assemble(table)


def test_explicit_piecewise_map_and_crossing_selection():
    t = T.build(crossing_start=5180, map_pieces=[(3440, 3740, 5180), (3740, 3816, 5554)])
    assert T.problems(t) == []
    assert row(t, "C5.map.beacon2")["hit_f"] == 5208
    assert row(t, "C5.map.beacon7")["hit_f"] == 5458
    assert row(t, "C5.map.beacon8")["hit_f"] == 5600
    cross = row(t, "A.watchfire_1")
    assert cross["delta_f"] == 1460 and cross["donor_crop"] == [5180, 5580]
    assert (cross["f0"], cross["f1"]) == (6640, 7040)
    assert not any(r["id"].endswith("A.watchfire_far") for r in t["reuse"])


@pytest.mark.parametrize("kwargs", [
    {"crossing_start": 4879}, {"crossing_start": 5441}, {"crossing_start": 5180.5},
    {"map_pieces": [(3440, 3816, 5180)]},  # no holdout despite complete source coverage
    {"map_pieces": [(3440, 3740, 5200), (3740, 3816, 5554)]},
    {"map_pieces": [(3440, 3700, 5180), (3700, 3816, 5554)]},  # insert before dark interval
    {"map_pieces": [(3440, 3740, 5180), (3741, 3816, 5554)]},
])
def test_invalid_editorial_binding_is_rejected(kwargs):
    with pytest.raises(ValueError):
        T.build(**kwargs)


def test_partial_assembly_preserves_polish_and_donor_bed_history(table):
    import sound_recipes_A as A
    import sound_recipes_C as C
    before = deepcopy((A.RECIPES, C.RECIPES))
    result = R.assemble(table, allow_partial=True)
    assert not result["deliverable"] and result["PENDING"]
    riffle = result["RECIPES"][row(table, "C5.riffle")["id"]]
    assert riffle["crest"] == 8.0 and riffle["level"] == -36.7
    assert riffle["env_after"][-1] == [0.95, 0.0]
    hearth = next(r for r in result["BED_CROPS"] if r["id"] == "D.06.C5.hearth.open")
    assert hearth["donor_crop"] == [320, 560]
    assert (hearth["donor_row"]["f0"], hearth["donor_row"]["f1"]) == (0, 600)
    assert hearth["donor_row"]["env_f"][0] == [0, 12.0]
    assert hearth["seed_id"] == "C5.hearth.open"
    assert result["PCM_COPIES"][0]["render_mode"] == "copy_final_packed_pcm_after_D_master"
    assert before == (A.RECIPES, C.RECIPES)
    result["RECIPES"][row(table, "C5.riffle")["id"]]["env_after"][0][1] = 999
    assert before == (A.RECIPES, C.RECIPES)


def test_every_request_names_a_compatible_palette_type(table):
    from sound_d_designs import design
    kinds = {"event": "event", "series": "event", "bed": "bed", "bed_family": "bed", "control": "control"}
    for r in table["requests"]:
        assert design(r["design"])["kind"] == kinds[r["kind"]], r["id"]


@pytest.mark.parametrize("mutation", ["offbar", "one_frame", "coherent_wrong_offset", "false_measured",
    "missing_ridge", "missing_refusal", "missing_requests", "invented_frame", "lost_seed",
    "wrong_prefix", "weaken_voice", "leak_silence", "birds", "missing_input", "stale_source"])
def test_negative_controls_fail_the_real_guard(table, mutation):
    if mutation == "offbar": table["beats"][20]["f0"] = 5200
    elif mutation == "one_frame": row(table, "C5.riffle")["hit_f"] += 1
    elif mutation == "coherent_wrong_offset": row(table, "C5.riffle").update(hit_f=10326, delta_f=10000)
    elif mutation == "false_measured": row(table, "C5.riffle")["source"] = "measured"
    elif mutation == "missing_ridge": table["reuse"] = [r for r in table["reuse"] if not r["id"].endswith("A.ridge.fire_34")]
    elif mutation == "missing_refusal": table["reuse"].remove(row(table, "C5.pen.refusal_offer"))
    elif mutation == "missing_requests": table["requests"] = []
    elif mutation == "invented_frame": table["requests"][0]["hit_f"] = 1688
    elif mutation == "lost_seed": row(table, "A.air.blue")["seed_id"] = "D.air.blue"
    elif mutation == "wrong_prefix": table["reuse"][0]["path"] = "music/out/v3/final_AP2_sfx.wav"
    elif mutation == "weaken_voice": table["constraints"]["voice_peak_dbfs"] = -39
    elif mutation == "leak_silence": table["constraints"]["vision_silence"] = [3401, 3440]
    elif mutation == "birds": table["constraints"]["birds_allowed"] = True
    elif mutation == "missing_input": table["input_sha256"].pop("music/sound/c5_sound_events.json")
    elif mutation == "stale_source": table["input_sha256"]["music/sound/c5_sound_events.json"] = "0" * 64
    assert T.problems(table), mutation
    with pytest.raises(ValueError, match="REFUSED"):
        R.assemble(table, allow_partial=True)


def test_build_does_not_modify_the_editorial_scaffold_or_any_existing_recipe():
    paths = [T.ROOT / p for p in T.INPUTS] + [T.ROOT / p for p in
        ("edit/edl_v3.py", "music/v3/barmap_D.json", "music/v3/cues_D.json")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    R.assemble(allow_partial=True)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
