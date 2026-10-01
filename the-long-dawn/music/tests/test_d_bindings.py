"""Binding provenance and delivery replacement, with synthetic D-frame assets.

The fixture's one-pixel PNGs exercise receipts and note movement. They provide
no evidence about a production picture's onset; the fixture says so explicitly.
"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from conftest import ROOT, _stub_sampler_deps
from bindings_D import (BoundMap, EDITORIAL_FILES, TABLE, _editorial_receipts,
                        load_map, movements)
from barmap_D_score import EVENT_TABLE


FIXTURE = Path(__file__).parent / "fixtures/d_binding_render_measurements.json"
PROTECTED = [Path(ROOT) / rel for rel in EDITORIAL_FILES] + [
    Path(ROOT) / "music/src" / f"score_v3_{cut}.py" for cut in ("A", "AP2", "C", "C5", "C5P2")]


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}


@pytest.fixture
def document():
    return json.loads(TABLE.read_text())


@pytest.fixture
def delivery(tmp_path, document):
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["synthetic"] is True
    patches = {}
    render_root = tmp_path / "renders"
    # Isolate the receipt path from production frame availability. Any native
    # rows already promoted in the live table receive explicitly synthetic
    # assets for this fixture; production receipts are checked separately.
    sample = fixture["measurements"][0]
    payload = base64.b64decode(sample["asset_base64"])
    for event in document["events"].values():
        for row in (event, event.get("absence_evidence", {})):
            if not row.get("measurement"):
                continue
            receipt = row["measurement"]
            asset = render_root / receipt["stem"] / receipt["frame_file"]
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(payload)
            receipt.update(sha256=hashlib.sha256(payload).hexdigest(),
                           method="SYNTHETIC TEST FIXTURE: receipt-path test only")
            row.update(source="SYNTHETIC TEST FIXTURE: no production timing evidence",
                       measured_ref="fixture://existing-native-row",
                       measurement_scope="Synthetic bytes; no picture measurement")
    for row in fixture["measurements"]:
        patch = row["patch"]
        receipt = patch["measurement"]
        asset = render_root / receipt["stem"] / receipt["frame_file"]
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(base64.b64decode(row["asset_base64"]))
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == receipt["sha256"]
        assert "SYNTHETIC TEST FIXTURE" in patch["source"]
        patches[row["event"]] = patch
    return document, render_root, patches


def make_score(bm):
    _stub_sampler_deps()
    import score_v3_D as D
    import verify_D as V
    score = D.build(bm)
    assert D.check(score, bm) == []
    assert V.call_problems(score, bm) == []
    assert V.binding_problems(score, bm) == []
    assert V.watch_problems(score) == []
    return score


def test_table_covers_every_editorial_and_score_event_with_honest_status(document):
    bar = json.loads((Path(ROOT) / EDITORIAL_FILES[0]).read_text())
    canonical = {event["id"] for event in bar["sync"]}
    original = {event["id"] for event in EVENT_TABLE}
    bm = BoundMap()
    assert len(canonical) == 154
    additional = {"old_fire_letters_flare", "old_fire_letters_full", "old_fire_letters_out",
                  "trap_neighbour_rise"}
    assert {event["id"] for event in bm.events} == canonical | original | additional
    assert {event["status"] for event in bm.events} == {"measured", "estimated"}
    # Later native deliveries may promote more rows through the same table.
    inherited_measured = {e["id"] for e in bar["sync"]
                          if e["timing_status"] in {
                              "inherited_measured", "inherited_cover_measurement",
                              "measured_D_cover", "measured_source_cover", "measured_D_composite"}}
    assert len(inherited_measured) == 22
    assert all(bm.event(name)["status"] == "measured" for name in inherited_measured)
    for event in bm.events:
        assert event["source"]
        if event["status"] == "measured":
            assert event["measured_ref"] and event["measurement_scope"]
    # AP2's pass-1 score markers do not become picture measurements by reuse.
    for name in ("ap2_wind", "ap2_false_dawn", "ap2_point", "ap2_ignition"):
        assert bm.event(name)["status"] == "estimated"
        assert bm.event(name)["editorial"]["timing_status"] == "inherited_pass 1"
    assert bm.binding_provenance["table_path"] == "music/v3/score_D_bindings.json"
    assert bm.binding_provenance["table_sha256"] == hashlib.sha256(TABLE.read_bytes()).hexdigest()
    assert load_map(document).events == bm.events
    assert load_map(TABLE).events == bm.events


def test_burn_cover_and_crossing_keep_distinct_measurement_scopes():
    bm = BoundMap()
    for name, frame in (("drawn_ring_burn_opens", 1683), ("drawn_ring_burn_page_gone", 1748)):
        event = bm.event(name)
        assert event["frame"] == frame and event["status"] == "measured"
        assert "480×201" in event["measurement_scope"]
        assert "not native-resolution" in event["measurement_scope"]
    assert bm.event("old_burn")["frame"] == 1680  # chord entry stays at the cut
    assert bm.event("crossing_walk_full")["frame"] == 6736
    assert bm.event("crossing_walk_full")["status"] == "measured"
    assert "camera-projection" in bm.event("crossing_walk_full")["measurement_scope"]
    assert bm.event("crossing_walk_full")["editorial"]["source_to_d"] == 1460
    assert bm.event("crossing_walk_full1")["frame"] == 6740
    assert bm.event("crossing_walk_full1")["status"] == "estimated"
    # Reaching walking speed and the retained musical step are separate events.
    score = make_score(bm)
    step_frames = [round(n.start * 20) for n in score.parts["crossing_feet"].notes]
    assert 6740 in step_frames and 6736 not in step_frames


def test_initial_reconciliation_can_be_rebuilt_from_editorial_evidence(document):
    # Native replacements must retain the original evidence, so the Phase 2
    # reconciliation can still be reconstructed after later farm deliveries.
    for row in document["events"].values():
        for key in ("score_action", "omission_reason", "omission_kind", "absence_evidence"):
            row.pop(key, None)
        if row.get("measurement") and row.get("editorial"):
            old = row["editorial"]
            row.update(frame=old["f"], status="estimated", source=old["source"],
                       measured_ref=None, measurement_scope=None, picture_frame=None)
            del row["measurement"]
    document["events"]["trap_surge"] = {"anchor": "trap_gold_passes"}
    original = {e["id"]: e for e in EVENT_TABLE}
    for name, row in document["events"].items():
        if row.get("measurement") and name in original and not row.get("editorial"):
            row.update(frame=original[name]["frame"], status="estimated",
                       source="Reconstructed original score marker", measured_ref=None,
                       measurement_scope=None, picture_frame=None)
            del row["measurement"]
    rows = movements(BoundMap(document))
    assert {r["event"]: (r["old_frame"], r["frame"], r["delta_frames"]) for r in rows} == {
        "inscription_stop": (2040, 1984, -56), "trap_drop": (3840, 3800, -40),
        "trap_surge": (3880, 3840, -40), "trap_return": (3960, 3940, -20),
        "leaders_climb": (4000, 4020, 20), "lamps": (6240, 6200, -40),
        "last_leaf_break": (8520, 8500, -20)}
    assert all(r["status"] == "estimated" for r in rows)


def test_round4_cover_scopes_do_not_move_the_composed_cut_entries():
    bm = BoundMap()
    expected = {
        "brink_burnthrough": (2953, "measured_source_cover", "before the D-only exit-tail envelope"),
        "brink_burn_clear": (2980, "measured_D_composite", "before finish"),
        "last_kingdom_burn": (5183, "measured_D_cover", "480×201"),
        "last_kingdom_burn_ridge_gone": (5212, "measured_D_cover", "480×201"),
    }
    for name, (frame, timing_status, scope) in expected.items():
        event = bm.event(name)
        assert event["frame"] == frame and event["status"] == "measured"
        assert event["editorial"]["timing_status"] == timing_status
        assert scope in event["measurement_scope"]
        assert event["picture_frame"] is None
    assert bm.event("last_kingdom_burn_t_open")["frame"] == 5182
    assert bm.event("last_kingdom_burn_t_open")["status"] == "estimated"
    assert bm.event("deep_red_glow")["status"] == "estimated"
    assert bm.event("brink")["frame"] == 2960
    assert bm.event("map")["frame"] == 5180
    assert "music/v3/events_D_round4_measured.json" in bm.binding_provenance["editorial_sha256"]


@pytest.mark.parametrize("failure,expected", [
    ("missing_leaf", "missing editorial receipt reference"),
    ("ridge_leaf", "receipt disagrees with the bar map"),
    ("parameter_leaf", "receipt disagrees with the bar map"),
    ("source_leaf_null", "receipt disagrees with the bar map"),
    ("ridge_samples", "ridge opening:.*summary disagrees"),
    ("source_clearance", "Deep source clearance:.*summary disagrees"),
    ("source_frame", "source frame disagrees"),
    ("tail_cover", "Deep tail clearance before finish:.*summary disagrees"),
    ("tail_identity", "Deep tail clearance before finish:.*summary disagrees"),
    ("glow_invented", "cannot establish a visible glow onset"),
])
def test_round4_receipt_rejects_changed_leaf_or_unsupported_summary(failure, expected):
    bar = json.loads((Path(ROOT) / EDITORIAL_FILES[0]).read_text())
    canonical = {event["id"]: event for event in bar["sync"]}
    receipts = {p: json.loads((Path(ROOT) / p).read_text()) for p in EDITORIAL_FILES[2:]}
    round4 = receipts[EDITORIAL_FILES[3]]
    if failure == "missing_leaf":
        del round4["ridge_burn"]["first_open_on_sample_grid"]
    elif failure == "ridge_leaf":
        round4["ridge_burn"]["first_open_on_sample_grid"] += 1
    elif failure == "parameter_leaf":
        round4["ridge_burn"]["parameters"]["t_open"] += 1
    elif failure == "source_leaf_null":
        round4["deep_exit_source"]["first_open_on_selected_source_masks"] = None
    elif failure == "ridge_samples":
        round4["ridge_burn"]["frames"][0]["open_pixels"] = 1
    elif failure == "source_clearance":
        round4["deep_exit_source"]["first_fully_gone_on_selected_source_masks"] = 2980
    elif failure == "source_frame":
        round4["deep_exit_source"]["first_open_source_frame"] += 1
    elif failure == "tail_cover":
        round4["deep_exit_tail"]["frames"][4]["effective_cover_max"] = 0.1
    elif failure == "tail_identity":
        round4["deep_exit_tail"]["frames"][4]["after_vs_incoming_max"] = 0.1
    elif failure == "glow_invented":
        round4["deep_exit_source"]["visible_glow_onset"] = 2953
    with pytest.raises(ValueError, match=expected):
        _editorial_receipts(canonical, receipts)


@pytest.mark.parametrize("name", ["brink_burnthrough", "brink_burn_clear",
                                  "last_kingdom_burn", "last_kingdom_burn_ridge_gone"])
def test_round4_source_scope_cannot_be_promoted_to_finished_picture(document, name):
    document["events"][name]["measurement_scope"] = "Finished native D picture onset"
    with pytest.raises(ValueError, match="retain the source evidence"):
        BoundMap(document)


@pytest.mark.parametrize("name,alias,part,frame", [
    ("crown_beacons", "crowns_kindle", "crowns_left", 4324),
    ("gap_hammer_lands", "gap_hammer_lands", "gap_anvil", 3443),
    ("holdout_first_contact", "holdout_hammer", "holdout_anvil", 5499),
    ("last_beacon", "last_beacon_catch", "map_last_answer", 5602),
])
def test_one_canonical_row_replacement_moves_actual_score_hit(delivery, name, alias, part, frame):
    document, render_root, patches = delivery
    before_table, before_sources = deepcopy(document), hashes()
    document["events"][name].update(patches[name])
    assert [key for key in document["events"] if document["events"][key] != before_table["events"][key]] == [name]
    bm = BoundMap(document, render_root=render_root)
    score = make_score(bm)
    assert bm.event(alias)["frame"] == frame and bm.event(alias)["status"] == "measured"
    assert bm.event(alias)["picture_frame"] == frame
    hit = next(ref for ref in score.sync_bindings if ref["event"] == alias and ref["part"] == part)
    assert score.parts[part].notes[hit["note_index"]].start * 20 == pytest.approx(frame)
    assert (3400, 3440) in score.hard_silences and bm.event("gap")["frame"] == 3440
    assert bm.event("all_lit")["frame"] == 5605
    assert hashes() == before_sources


def test_delivery_receipts_together_preserve_call_unison_and_grid(delivery):
    document, render_root, patches = delivery
    for name, patch in patches.items():
        document["events"][name].update(patch)
    bm = BoundMap(document, render_root=render_root)
    score = make_score(bm)
    for name in score.call_parts:
        assert [(n.start * 20, n.pitch) for n in score.parts[name].notes[:3]] == [
            (4324, 62), (4344, 69), (4364, 74)]
    assert (bm.bars, bm.frames, bm.d["fps"]) == (115, 9200, 24)
    assert score.pcm_regions[0]["f0"] == 0 and score.pcm_regions[0]["f1"] == 1440


def retime_fixture(patch, render_root, frame):
    """Copy synthetic bytes to another D frame; never production evidence."""
    patch = deepcopy(patch)
    receipt = patch["measurement"]
    original = render_root / receipt["stem"] / receipt["frame_file"]
    receipt.update(frame=frame, frame_file=f"frame_{frame:06d}.png")
    (original.parent / receipt["frame_file"]).write_bytes(original.read_bytes())
    patch.update(frame=frame, measured_ref=f"fixture://{receipt['stem']}/{receipt['frame_file']}")
    return patch


def test_native_delivery_may_cross_estimated_bar_inside_committed_window(delivery):
    document, render_root, patches = delivery
    document["events"]["crown_beacons"].update(retime_fixture(patches["crown_beacons"], render_root, 4401))
    bm = BoundMap(document, render_root=render_root)
    assert bm.event("crowns_kindle")["frame"] == 4401
    score = make_score(bm)
    assert score.parts["crowns_left"].notes[0].start * 20 == pytest.approx(4401)


@pytest.mark.parametrize("failure,expected", [
    ("unmeasured_crossbar", "canonical bar"),
    ("outside_window", "outside editorial timing window"),
    ("invalid_crossbar_receipt", "hash mismatch"),
])
def test_crossbar_delivery_still_requires_native_evidence_and_window(delivery, failure, expected):
    document, render_root, patches = delivery
    row = document["events"]["crown_beacons"]
    if failure == "unmeasured_crossbar":
        row.pop("measurement", None)
        row["status"] = "estimated"
        row["frame"] = 4401
    else:
        frame = 4560 if failure == "outside_window" else 4401
        row.update(retime_fixture(patches["crown_beacons"], render_root, frame))
        if failure == "invalid_crossbar_receipt":
            row["measurement"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=expected):
        BoundMap(document, render_root=render_root)


def test_new_measured_marker_uses_explicit_window_and_section(delivery):
    document, render_root, patches = delivery
    row = retime_fixture(patches["crown_beacons"], render_root, 6000)
    row.update(baseline_frame=5840, timing_window=[5840, 6080])
    document["events"]["synthetic_oldfire_event"] = row
    assert BoundMap(document, render_root=render_root).event("synthetic_oldfire_event")["frame"] == 6000
    # A window cannot grant permission to leave the source marker's section.
    row.update(retime_fixture(patches["crown_beacons"], render_root, 6100))
    row["timing_window"] = [5840, 6200]
    with pytest.raises(ValueError, match="new score marker cannot leave its section"):
        BoundMap(document, render_root=render_root)


@pytest.mark.parametrize("failure,expected", [
    ("missing_hash", "hash mismatch"), ("bad_asset", "hash mismatch"),
    ("missing_asset", "asset is missing"), ("wrong_stem", "different shot"),
    ("no_evidence", "reference and scope"), ("no_receipt", "render_D receipt"),
    ("wrong_receipt_frame", "receipt frame differs"), ("wrong_filename", "filename"),
    ("estimated_receipt", "labelled measured"),
])
def test_new_shot_measurement_rejects_broken_evidence(delivery, failure, expected):
    document, render_root, patches = delivery
    row = document["events"]["crown_beacons"]
    row.update(patches["crown_beacons"])
    receipt = row["measurement"]
    asset = render_root / receipt["stem"] / receipt["frame_file"]
    if failure == "missing_hash":
        del receipt["sha256"]
    elif failure == "bad_asset":
        asset.write_bytes(asset.read_bytes() + b"changed")
    elif failure == "missing_asset":
        asset.unlink()
    elif failure == "wrong_stem":
        receipt["stem"] = "embers_D_gap"
    elif failure == "no_evidence":
        row["measured_ref"] = None
    elif failure == "no_receipt":
        del row["measurement"]
    elif failure == "wrong_receipt_frame":
        receipt["frame"] += 1
    elif failure == "wrong_filename":
        receipt["frame_file"] = "frame_004325.png"
    elif failure == "estimated_receipt":
        row["status"] = "estimated"
    with pytest.raises(ValueError, match=expected):
        BoundMap(document, render_root=render_root)


@pytest.mark.parametrize("failure,expected", [
    ("alias_duplicate", "duplicate alias"), ("missing_name", "missing events"),
    ("out_of_bar", "canonical bar"), ("source_hash", "different editorial source"),
    ("alias_cycle", "alias cycle"), ("grid", "115 bars"),
    ("protected_opening", "protected opening"), ("fixed_cut", "fixed edit boundary"),
    ("changed_editorial", "editorial evidence"), ("promoted_pass1", "render_D receipt"),
    ("unscoped_inheritance", "reference and scope"),
    ("promoted_burn_scope", "retain the source evidence"),
    ("promoted_burn_source", "retain the source evidence"),
    ("promoted_burn_ref", "retain the source evidence"),
    ("estimated_picture_frame", "picture_frame is derived"),
    ("inherited_picture_frame", "picture_frame is derived"),
    ("canonical_cut_alias", "canonical events cannot be aliases"),
    ("canonical_measured_alias", "canonical events cannot be aliases"),
])
def test_table_rejects_conflicting_or_unproven_bindings(document, failure, expected):
    rows = document["events"]
    if failure == "alias_duplicate":
        rows["crowns_kindle"]["frame"] = 4320
    elif failure == "missing_name":
        del rows["crown_beacons"]
    elif failure == "out_of_bar":
        rows["crown_beacons"].pop("measurement", None)
        rows["crown_beacons"]["status"] = "estimated"
        rows["crown_beacons"]["frame"] = 4400
    elif failure == "source_hash":
        document["editorial_sha256"][EDITORIAL_FILES[0]] = "0" * 64
    elif failure == "alias_cycle":
        rows["crowns_kindle"]["anchor"] = "crowns_kindle"
    elif failure == "grid":
        document["bars"] = 114
    elif failure == "protected_opening":
        rows["ap2_ignition"]["frame"] = 1041
    elif failure == "fixed_cut":
        rows["drawn_ring_burn"]["frame"] = 1681
    elif failure == "changed_editorial":
        rows["crown_beacons"]["editorial"]["f"] = 4324
    elif failure == "promoted_pass1":
        rows["ap2_point"]["status"] = "measured"
    elif failure == "unscoped_inheritance":
        rows["drawn_ring_burn_opens"]["measurement_scope"] = None
    elif failure == "promoted_burn_scope":
        rows["drawn_ring_burn_opens"]["measurement_scope"] = "native D picture measurement"
    elif failure == "promoted_burn_source":
        rows["drawn_ring_burn_opens"]["source"] = "finished native picture inspection"
    elif failure == "promoted_burn_ref":
        rows["drawn_ring_burn_opens"]["measured_ref"] = "renders/native.png"
    elif failure == "estimated_picture_frame":
        rows["crown_beacons"]["picture_frame"] = 4320
    elif failure == "inherited_picture_frame":
        rows["last_beacon"]["picture_frame"] = 5550
    elif failure == "canonical_cut_alias":
        rows["holdout_insert"] = {"anchor": "holdout_first_contact",
                                   "editorial": rows["holdout_insert"]["editorial"]}
    elif failure == "canonical_measured_alias":
        rows["drawn_ring_burn_opens"] = {"anchor": "old_story_riffle",
                                          "editorial": rows["drawn_ring_burn_opens"]["editorial"]}
    with pytest.raises(ValueError, match=expected):
        BoundMap(document)


def test_duplicate_json_row_cannot_silently_overwrite_an_anchor(tmp_path):
    text = TABLE.read_text()
    needle = '"crown_beacons": {'
    assert text.count(needle) == 1
    path = tmp_path / "duplicate-binding.json"
    path.write_text(text.replace(needle, '"crown_beacons": {},\n    ' + needle, 1))
    with pytest.raises(ValueError, match="duplicate binding key: crown_beacons"):
        load_map(path)


OMITTED_FORGING_IDS = tuple(f"giant_stroke_{i:02d}" for i in (1, 2, 3, 4, 8))


def omission_fixture(document, render_root, patches):
    """Invent only test evidence, explicitly labelled; retain authored frames."""
    original = {event["id"]: event for event in EVENT_TABLE}
    for name in OMITTED_FORGING_IDS:
        frame = original[name]["frame"]
        patch = retime_fixture(patches["crown_beacons"], render_root, frame)
        receipt = patch["measurement"]
        source_asset = render_root / receipt["stem"] / receipt["frame_file"]
        receipt["stem"] = "embers_D_forging"
        asset = render_root / receipt["stem"] / receipt["frame_file"]
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(source_asset.read_bytes())
        row = document["events"][name]
        row.pop("measurement", None)
        row.pop("anchor", None)
        row.update(frame=frame, status="estimated", picture_frame=None,
                   source="SYNTHETIC TEST FIXTURE: authored accent position",
                   measured_ref=None, measurement_scope=None,
                   render_stem="embers_D_forging", score_action="omit",
                   omission_reason="SYNTHETIC TEST FIXTURE: no corresponding impact in the inspected window",
                   absence_evidence={
                       "source": "SYNTHETIC TEST FIXTURE: no production absence claim",
                       "measured_ref": f"fixture://embers_D_forging/{receipt['frame_file']}",
                       "measurement_scope": "Synthetic bytes exercise omission receipts; no image observation",
                       "measurement": receipt,
                   })
    return document


def test_explicit_absence_omits_only_named_glass_accents(delivery):
    document, render_root, patches = delivery
    # The no-omission comparison remains meaningful after real receipts land.
    for row in document["events"].values():
        for key in ("score_action", "omission_reason", "omission_kind", "absence_evidence"):
            row.pop(key, None)
    before = make_score(BoundMap(document, render_root=render_root))
    before_hashes = hashes()
    omission_fixture(document, render_root, patches)
    bm = BoundMap(document, render_root=render_root)
    after = make_score(bm)
    omitted_frames = {bm.event(name)["frame"] for name in OMITTED_FORGING_IDS}
    assert set(before.parts) == set(after.parts)
    for name in before.parts:
        expected = [n.to_dict() for n in before.parts[name].notes
                    if name != "giant_glass" or round(n.start * 20) not in omitted_frames]
        assert [n.to_dict() for n in after.parts[name].notes] == expected
    assert len(before.parts["giant_glass"].notes) - len(after.parts["giant_glass"].notes) == 5
    for name in OMITTED_FORGING_IDS:
        event = bm.event(name)
        assert event["score_action"] == "omit" and event["status"] == "estimated"
        assert event.get("picture_frame") is None and not event.get("measurement")
        assert not [row for row in after.picture_bindings + after.sync_bindings if row["event"] == name]
    for name in ("giant_stroke_05", "giant_stroke_06", "giant_stroke_07"):
        hit = next(row for row in after.sync_bindings if row["event"] == name)
        assert after.parts["giant_glass"].notes[hit["note_index"]].start * 20 == pytest.approx(bm.event(name)["frame"])
    assert hashes() == before_hashes


@pytest.mark.parametrize("failure,expected", [
    ("unknown_action", "unknown score_action"),
    ("no_action", "requires score_action omit"),
    ("non_giant", "only direct nonfixed original giant_stroke"),
    ("fixed", "only direct nonfixed original giant_stroke"),
    ("canonical", "only direct nonfixed original giant_stroke"),
    ("alias", "only direct nonfixed original giant_stroke"),
    ("inherited_alias", "omission cannot be inherited"),
    ("no_reason", "needs a rationale"),
    ("empty_reason", "needs a rationale"),
    ("no_absence", "needs absence evidence"),
    ("no_source", "needs absence evidence"),
    ("no_scope", "absence needs source, reference and scope"),
    ("no_ref", "absence needs source, reference and scope"),
    ("no_method", "observation method"),
    ("bad_asset", "hash mismatch"),
    ("missing_asset", "asset is missing"),
    ("wrong_stem", "different shot"),
    ("wrong_receipt_frame", "receipt frame differs"),
    ("wrong_filename", "filename"),
    ("promoted_status", "absence cannot be promoted"),
    ("promoted_picture", "absence cannot be promoted"),
    ("moved_frame", "retain its original frame"),
])
def test_omission_rejects_unscoped_or_corrupt_absence(delivery, failure, expected):
    document, render_root, patches = delivery
    omission_fixture(document, render_root, patches)
    rows = document["events"]
    row = rows["giant_stroke_01"]
    absence = row["absence_evidence"]
    receipt = absence["measurement"]
    asset = render_root / receipt["stem"] / receipt["frame_file"]
    if failure == "unknown_action":
        row["score_action"] = "mute_everything"
    elif failure == "no_action":
        del row["score_action"]
    elif failure in ("non_giant", "fixed", "canonical"):
        target = {"non_giant": "leaders_climb", "fixed": "forging", "canonical": "gap_hammer_lands"}[failure]
        for key in ("score_action", "omission_reason", "absence_evidence"):
            rows[target][key] = deepcopy(row[key])
    elif failure == "alias":
        row["anchor"] = "giant_stroke_05"
    elif failure == "inherited_alias":
        rows["giant_stroke_02"] = {"anchor": "giant_stroke_01"}
    elif failure == "no_reason":
        del row["omission_reason"]
    elif failure == "empty_reason":
        row["omission_reason"] = "  "
    elif failure == "no_absence":
        del row["absence_evidence"]
    elif failure == "no_source":
        del absence["source"]
    elif failure == "no_scope":
        absence["measurement_scope"] = "  "
    elif failure == "no_ref":
        del absence["measured_ref"]
    elif failure == "no_method":
        receipt["method"] = "  "
    elif failure == "bad_asset":
        asset.write_bytes(asset.read_bytes() + b"changed")
    elif failure == "missing_asset":
        asset.unlink()
    elif failure == "wrong_stem":
        receipt["stem"] = "embers_D_crowns"
    elif failure == "wrong_receipt_frame":
        receipt["frame"] += 1
    elif failure == "wrong_filename":
        receipt["frame_file"] = "frame_999999.png"
    elif failure == "promoted_status":
        row["status"] = "measured"
    elif failure == "promoted_picture":
        row["picture_frame"] = row["frame"]
    elif failure == "moved_frame":
        row["frame"] += 1
    with pytest.raises(ValueError, match=expected):
        BoundMap(document, render_root=render_root)


OMITTED_BRINK_IDS = tuple(f"giant_stroke_{i:02d}" for i in range(17, 23))


def brink_omission_fixture(document, render_root, patches):
    """Synthetic pulse assets; the unchanged gap comes from pinned source code."""
    original = {event["id"]: event for event in EVENT_TABLE}
    geometry_path = "shots/embers/d_vision.py"
    geometry = dict(path=geometry_path,
                    sha256=hashlib.sha256((Path(ROOT) / geometry_path).read_bytes()).hexdigest())
    for name in OMITTED_BRINK_IDS:
        frame = original[name]["frame"]
        patch = retime_fixture(patches["crown_beacons"], render_root, frame)
        receipt = patch["measurement"]
        source_asset = render_root / receipt["stem"] / receipt["frame_file"]
        receipt["stem"] = "embers_D_brink"
        asset = render_root / receipt["stem"] / receipt["frame_file"]
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(source_asset.read_bytes())
        document["events"][name].update(
            frame=frame, status="measured", picture_frame=None,
            source="SYNTHETIC TEST FIXTURE: no production pulse measurement",
            measured_ref=f"fixture://embers_D_brink/{receipt['frame_file']}",
            measurement_scope="Synthetic bytes exercise a native pulse receipt",
            measurement=receipt, render_stem="embers_D_brink",
            score_action="omit", omission_kind="no_gap_narrowing",
            omission_reason="Glass follows narrowing; the Brink source keeps the gap at55 degrees",
            absence_evidence={
                "source": "SYNTHETIC TEST FIXTURE: pulse evidence plus current source geometry",
                "measured_ref": f"fixture://embers_D_brink/{receipt['frame_file']}",
                "measurement_scope": "Synthetic pulse bytes; no gap-angle pixel measurement",
                "measurement": deepcopy(receipt), "geometry_source": deepcopy(geometry),
            })
    return document


def test_brink_omission_preserves_measured_pulse_and_other_notation(delivery):
    document, render_root, patches = delivery
    brink_omission_fixture(document, render_root, patches)
    before_document = deepcopy(document)
    for name in OMITTED_BRINK_IDS:
        for key in ("score_action", "omission_reason", "omission_kind", "absence_evidence"):
            before_document["events"][name].pop(key, None)
    before = make_score(BoundMap(before_document, render_root=render_root))
    bm = BoundMap(document, render_root=render_root)
    after = make_score(bm)
    omitted_frames = {bm.event(name)["frame"] for name in OMITTED_BRINK_IDS}
    for name, part in before.parts.items():
        expected = [n.to_dict() for n in part.notes
                    if name != "giant_glass" or round(n.start * 20) not in omitted_frames]
        assert [n.to_dict() for n in after.parts[name].notes] == expected
    assert len(before.parts["giant_glass"].notes) - len(after.parts["giant_glass"].notes) == 6
    for name in OMITTED_BRINK_IDS:
        event = bm.event(name)
        assert event["status"] == "measured" and event["picture_frame"] == event["frame"]
        assert event["measurement"] == before_document["events"][name]["measurement"]
        assert event["score_action"] == "omit" and event["omission_kind"] == "no_gap_narrowing"
        assert not [hit for hit in after.picture_bindings + after.sync_bindings if hit["event"] == name]


@pytest.mark.parametrize("failure,expected", [
    ("unknown_kind", "unknown omission_kind"),
    ("wrong_section", "restricted to original Brink strokes"),
    ("no_kind", "absence cannot be promoted"),
    ("no_action", "requires score_action omit"),
    ("downgraded_pulse", "must retain its native measured pulse"),
    ("missing_pulse", "must retain its native measured pulse"),
    ("bad_pulse_hash", "hash mismatch"),
    ("bad_absence_hash", "hash mismatch"),
    ("missing_geometry", "pinned Brink geometry source"),
    ("wrong_geometry_path", "pinned Brink geometry source"),
    ("missing_geometry_hash", "geometry source hash mismatch"),
    ("changed_geometry_hash", "geometry source hash mismatch"),
])
def test_brink_omission_requires_real_pulse_and_pinned_geometry(delivery, failure, expected):
    document, render_root, patches = delivery
    brink_omission_fixture(document, render_root, patches)
    row = document["events"]["giant_stroke_17"]
    if failure == "unknown_kind":
        row["omission_kind"] = "weak_flash"
    elif failure == "wrong_section":
        row = document["events"]["giant_stroke_05"]
        row.update({key: deepcopy(document["events"]["giant_stroke_17"][key]) for key in
                    ("score_action", "omission_reason", "omission_kind", "absence_evidence")})
    elif failure == "no_kind":
        del row["omission_kind"]
    elif failure == "no_action":
        del row["score_action"]
    elif failure == "downgraded_pulse":
        row["status"] = "estimated"
    elif failure == "missing_pulse":
        del row["measurement"]
    elif failure == "bad_pulse_hash":
        row["measurement"]["sha256"] = "0" * 64
    elif failure == "bad_absence_hash":
        row["absence_evidence"]["measurement"]["sha256"] = "0" * 64
    elif failure == "missing_geometry":
        del row["absence_evidence"]["geometry_source"]
    elif failure == "wrong_geometry_path":
        row["absence_evidence"]["geometry_source"]["path"] = "shots/embers/openring_d.py"
    elif failure == "missing_geometry_hash":
        del row["absence_evidence"]["geometry_source"]["sha256"]
    elif failure == "changed_geometry_hash":
        row["absence_evidence"]["geometry_source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=expected):
        BoundMap(document, render_root=render_root)


def test_trap_surge_binds_the_actual_horn_entry_with_negative_control():
    import verify_D as V
    bm = BoundMap()
    score = make_score(bm)
    hit = next(row for row in score.sync_bindings
               if row["event"] == "trap_surge" and row["part"] == "trap_horns")
    note = score.parts[hit["part"]].notes[hit["note_index"]]
    assert note.start * 20 == pytest.approx(bm.event("trap_surge")["frame"])
    assert bm.event("trap_surge")["anchor"] == "trap_neighbour_rise"
    note.start += 1 / 20
    assert "UNBOUND HIT FRAME: trap_surge, trap_horns" in V.binding_problems(score, bm)


@pytest.mark.parametrize("boundary", ["inscription_stop", "trap_drop", "trap_return"])
def test_native_silent_boundaries_match_notation_with_negative_controls(boundary):
    import score_v3_D as D
    bm = BoundMap()
    score = make_score(bm)
    frame = bm.event(boundary)["frame"]

    def endpoint(candidate):
        if boundary == "inscription_stop":
            note = max(candidate.parts["inscription_ring"].notes, key=lambda n: n.start + n.dur)
            return (note.start + note.dur) * 20, note
        notes = candidate.parts[D.TRAP_VOICE].notes
        if boundary == "trap_drop":
            note = max((n for n in notes if n.start * 20 < frame), key=lambda n: n.start + n.dur)
            return (note.start + note.dur) * 20, note
        note = min((n for n in notes if n.start * 20 >= frame - 1e-5), key=lambda n: n.start)
        return note.start * 20, note

    value, note = endpoint(score)
    assert value == pytest.approx(frame)
    if boundary == "trap_return":
        note.start += 1 / 20
    else:
        note.dur += 1 / 20
    with pytest.raises(AssertionError):
        assert endpoint(score)[0] == pytest.approx(frame)
