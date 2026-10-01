"""Evidence-gate tests; synthetic image fixtures never enter production metadata."""
from copy import deepcopy
from io import BytesIO
import json

import pytest
from PIL import Image

import sound_d_binding as B
import sound_d_measurements as M


def document():
    return dict(schema=M.SCHEMA, cut="D", fps=24, frames=9200,
                input_sha256=M.raw_input_hashes(), hooks={}, events={}, series={}, suppressions={})


def evidence(tmp_path, first, last, *, claim="contact"):
    """Generate conspicuously synthetic test-only plates, with real file hashes."""
    buf = BytesIO()
    Image.new("RGB", (1920, 804), (23, 41, 71)).save(buf, format="PNG")
    data = buf.getvalue()
    folder = tmp_path / "renders" / "synthetic_test_fixture"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(first, last + 1):
        path = folder / f"f_{frame:05d}.png"
        path.write_bytes(data)
        rows.append(dict(frame=frame, path=str(path.relative_to(tmp_path)), sha256=M.sha256(path)))
    return dict(method="Synthetic fixture for evidence-gate tests; no production observation", claim=claim,
                frame_size=[1920, 804], frames=rows,
                coverage=dict(first=first, last=last, complete=True, missing=[]),
                predicate="test-only predicate; not an asserted production measurement")


def point(tmp_path, frame, *, claim="contact", first=None, last=None):
    return dict(frame=frame, status="measured", source="synthetic test fixture only", measured_ref="test-only",
                measurement_scope=M.NATIVE_SCOPE,
                measurement=evidence(tmp_path, frame if first is None else first, frame if last is None else last, claim=claim))


def test_native_contact_moves_without_moving_fixed_edit_or_silence(tmp_path):
    doc = document()
    doc["hooks"]["gap_hammer_lands"] = point(tmp_path, 3442)
    doc["hooks"]["crown_beacons"] = point(tmp_path, 4321, claim="onset", first=4320)
    plan = B.build(measurements=doc, frame_root=tmp_path)
    by_id = {r["id"]: r for r in plan["new_events"]}
    gap = by_id["D.new.gap.hammer"]
    assert gap["hit_f"] == gap["picture_frame"] == 3442
    assert gap["source"] == "measured" and gap["measurement"]["claim"] == "contact"
    assert gap["editorial_reference"]["frame"] == 3440
    assert plan["constraints"]["gap_frame"] == 3440
    assert plan["constraints"]["vision_silence"] == [3400, 3440]
    assert next(r for r in plan["new_beds"] if r["id"] == "D.new.vision.pressure")["f1"] == 3400
    assert [(by_id["D.new.crowns." + s]["hit_f"], by_id["D.new.crowns." + s]["pan"])
            for s in ("left", "right")] == [(4321, -1), (4321, 1)]
    assert B.problems(plan, measurements=doc, frame_root=tmp_path) == []
    assert B.problems(plan)  # measured plans cannot validate against an absent overlay


@pytest.mark.parametrize("mutation", ["hash", "metadata", "scope", "missing_scope", "parameter", "fixed_cut", "path",
                                     "wrong_file_frame", "fake_complete", "no_prior_onset", "no_predicate", "outside_window"])
def test_native_claims_fail_closed(tmp_path, mutation):
    doc = document()
    p = point(tmp_path, 4321, claim="onset", first=4320)
    doc["hooks"]["crown_beacons"] = p
    if mutation == "hash":
        p["measurement"]["frames"][0]["sha256"] = "0" * 64
    elif mutation == "metadata":
        doc["input_sha256"][M.INPUT_FILES[-1]] = "0" * 64
    elif mutation == "scope":
        p["measurement_scope"] = "native_finished_picture"
    elif mutation == "missing_scope":
        p.pop("measurement_scope")
    elif mutation == "parameter":
        doc["hooks"] = {"last_kingdom_burn_t_open": point(tmp_path, 5182)}
    elif mutation == "fixed_cut":
        doc["hooks"] = {"gap": point(tmp_path, 3442)}
    elif mutation == "path":
        p["measurement"]["frames"][0]["path"] = "../renders/synthetic_test_fixture/f_04320.png"
    elif mutation == "wrong_file_frame":
        p["measurement"]["frames"][0]["source_frame"] = 4319
    elif mutation == "fake_complete":
        p["measurement"]["frames"].pop(0)
    elif mutation == "no_prior_onset":
        p["measurement"] = evidence(tmp_path, 4321, 4321, claim="onset")
    elif mutation == "no_predicate":
        p["measurement"].pop("predicate")
    elif mutation == "outside_window":
        doc["hooks"]["crown_beacons"] = point(tmp_path, 4560)
    with pytest.raises(ValueError):
        B.build(measurements=doc, frame_root=tmp_path)


def test_observed_state_and_unknown_frame_remain_evidence_not_onsets(tmp_path):
    doc = document()
    doc["hooks"]["crown_beacons"] = point(tmp_path, 4330, claim="observed_state")
    doc["hooks"]["gap_hammer_lands"] = dict(frame=None, status="unmeasured", reason="No plate coverage in synthetic case")
    plan = B.build(measurements=doc, frame_root=tmp_path)
    assert next(r for r in plan["new_events"] if r["id"] == "D.new.crowns.left")["source"] == "est."
    assert plan["measurement_overlay"]["observations"]["crown_beacons"]["frame"] == 4330
    assert plan["measurement_overlay"]["gaps"][0]["id"] == "gap_hammer_lands"


def test_measured_envelope_anchor_preserves_estimated_bed_and_authored_gain(tmp_path):
    doc = document()
    doc["hooks"]["dawn_sunlight"] = point(tmp_path, 7727, claim="onset", first=7726)
    plan = B.build(measurements=doc, frame_root=tmp_path)
    wind = next(r for r in plan["new_beds"] if r["id"] == "D.new.dawn.wind")
    assert wind["source"] == "est." and wind["picture_frame"] is None
    point_row = next(p for p in wind["envelope_provenance"] if p["hook"] == "dawn_sunlight")
    assert point_row["frame"] == 7727 and point_row["source"] == "measured"
    assert point_row["measurement_scope"] == M.NATIVE_SCOPE
    assert point_row["gain_db"] == -8 and point_row["gain_source"] == "authored envelope"
    assert [p["frame"] for p in wind["envelope_provenance"]] == [p[0] for p in wind["env_f"]]
    assert not plan["constraints"]["no_new_picture_measurement"]


def test_one_off_landmark_does_not_conflate_ring_fall_and_letters_flare(tmp_path):
    doc = document()
    p = point(tmp_path, 5901, claim="onset", first=5900)
    p.update(landmark="old_fire_letters_flare", editorial_reference="old_fire_ring_falls")
    doc["events"]["D.new.oldfire.flare"] = p
    plan = B.build(measurements=doc, frame_root=tmp_path)
    row = next(r for r in plan["new_events"] if r["id"] == "D.new.oldfire.flare")
    assert row["hook"] == "old_fire_letters_flare" and row["hit_f"] == 5901
    assert row["editorial_reference"]["hook"] == "old_fire_ring_falls"
    assert row["editorial_reference"]["previous_binding"]["hit_f"] == 5900


def test_reused_event_keeps_its_donor_gain_clock_when_picture_is_remeasured(tmp_path):
    before = B.build()
    original = next(r for r in before["reuse_plan"]["EXTRA_EVENTS"] if r["id"] == "D.11.C5.burn.deep")
    doc = document()
    p = point(tmp_path, 2740, claim="onset", first=2739)
    p.update(landmark="deep_native_sweep", editorial_reference="deep_ember_sweep")
    doc["events"][original["id"]] = p
    row = next(r for r in B.build(measurements=doc, frame_root=tmp_path)["reuse_plan"]["EXTRA_EVENTS"] if r["id"] == original["id"])
    assert row["hit_f"] == 2740 and row["source"] == "measured"
    assert row["donor_hit_f"] == original["donor_hit_f"] and row["seed_id"] == original["seed_id"]
    assert row["inherited_delta_f"] == original["delta_f"]


def test_complete_physical_series_replaces_estimated_grid_and_preserves_inventory(tmp_path):
    doc = document()
    search = evidence(tmp_path, 5680, 5839)
    contacts = []
    for n, frame in enumerate((5688, 5708)):
        p = point(tmp_path, frame)
        p["id"] = "contact" + str(n)
        p.update(pan=-0.25, spatial_basis="Synthetic test of an authored mapping from screen centroid")
        contacts.append(p)
    doc["series"]["D.new.instep.hammers"] = dict(events=contacts, measurement=search, coverage=search["coverage"])
    plan = B.build(measurements=doc, frame_root=tmp_path)
    rows = [r for r in plan["new_events"] if r["request_id"] == "D.new.instep.hammers"]
    assert [r["hit_f"] for r in rows] == [5688, 5708]
    assert all(r["source"] == "measured" and "spacing_f" not in r for r in rows)
    assert all(r["pan"] == -0.25 and "Authored" in r["distance_basis"] and "not measured" in r["level_trim_basis"] for r in rows)
    assert all(r["dist"] == r["design_reference"]["dist"] and r["level_trim_db"] == r["design_reference"]["level_trim_db"] for r in rows)
    assert "D.new.instep.hammers" in plan["bound_request_ids"]
    bad = deepcopy(doc)
    bad["series"]["D.new.instep.hammers"]["measurement"]["frames"].pop()
    with pytest.raises(ValueError, match="denominator"):
        B.build(measurements=bad, frame_root=tmp_path)


def test_partial_series_retains_estimates_and_explicit_gap(tmp_path):
    doc = document()
    search = evidence(tmp_path, 5680, 5681)
    search["coverage"]["complete"] = False
    doc["series"]["D.new.instep.hammers"] = dict(events=[dict(id="unknown", frame=None, status="unmeasured", reason="No contact yet")],
        measurement=search, coverage=search["coverage"], reason="Only two native frames available")
    plan = B.build(measurements=doc, frame_root=tmp_path)
    rows = [r for r in plan["new_events"] if r["request_id"] == "D.new.instep.hammers"]
    assert len(rows) == 8 and all(r["source"] == "est." for r in rows)
    assert len(plan["measurement_overlay"]["gaps"]) == 2


def test_suppression_retires_rows_without_losing_request_accounting(tmp_path):
    doc = document()
    for request in ("D.new.leaf.quill", "D.new.leaf.stop"):
        doc["suppressions"][request] = dict(reason="Synthetic test of explicit static-picture decision",
            source="test-only", measurement=evidence(tmp_path, 8320, 8321, claim="static_state"))
    plan = B.build(measurements=doc, frame_root=tmp_path)
    assert not any(r["request_id"] in doc["suppressions"] for r in plan["new_events"] + plan["controls"])
    assert set(doc["suppressions"]).issubset(plan["bound_request_ids"])
    assert {r["request_id"] for r in plan["suppressed_requests"]} == set(doc["suppressions"])
    assert all(r["retired_rows"] for r in plan["suppressed_requests"])


def test_suppression_cited_source_code_must_still_match(tmp_path):
    doc = document()
    code = tmp_path / "shots" / "synthetic_fixture.py"
    code.parent.mkdir()
    code.write_text("# Synthetic fixture, not production picture source.\n")
    proof = evidence(tmp_path, 8320, 8320, claim="static_state")
    proof["source_sha256"] = {"shots/synthetic_fixture.py": M.sha256(code)}
    doc["suppressions"]["D.new.leaf.quill"] = dict(reason="test-only source-backed decision", source="synthetic fixture", measurement=proof)
    assert B.build(measurements=doc, frame_root=tmp_path)["suppressed_requests"]
    code.write_text("# Changed synthetic fixture.\n")
    with pytest.raises(ValueError, match="source evidence SHA256"):
        B.build(measurements=doc, frame_root=tmp_path)
    proof["source_sha256"] = {"../synthetic_fixture.py": "0" * 64}
    with pytest.raises(ValueError, match="relative"):
        B.build(measurements=doc, frame_root=tmp_path)


@pytest.mark.parametrize("other", ["events", "series"])
def test_conflicting_replacements_and_suppression_are_rejected_before_applying(tmp_path, other):
    doc = document()
    key = "D.new.holdout.hammer"
    p = point(tmp_path, 5496)
    p.update(landmark="holdout_contact", editorial_reference="holdout_first_contact")
    doc["events"][key] = p
    if other == "series":
        doc["series"][key] = dict(events=[dict(p, id="one")], coverage=p["measurement"]["coverage"])
    else:
        doc["suppressions"][key] = dict(reason="synthetic contradiction", source="test-only", measurement=p["measurement"])
    with pytest.raises(ValueError, match="conflicting"):
        B.build(measurements=doc, frame_root=tmp_path)


@pytest.mark.parametrize("mutation", ["cover_summary", "source_summary", "source_clear", "glow", "denominator", "scope"])
def test_editorial_masks_cannot_be_promoted_or_detached_from_samples(mutation):
    r4 = M.read(M.ROOT / M.INPUT_FILES[4])
    bm = M.read(M.ROOT / M.INPUT_FILES[1])
    if mutation == "cover_summary":
        r4["ridge_burn"]["first_open_on_sample_grid"] = 5182
    elif mutation == "source_summary":
        r4["deep_exit_source"]["first_open_on_selected_source_masks"] = 2945
    elif mutation == "source_clear":
        r4["deep_exit_source"]["first_fully_gone_on_selected_source_masks"] = 2980
    elif mutation == "glow":
        r4["deep_exit_source"]["visible_glow_onset"] = 2945
    elif mutation == "denominator":
        r4["deep_exit_source"]["frames"].pop()
    elif mutation == "scope":
        next(h for h in bm["sync"] if h["id"] == "brink_burn_clear")["timing_status"] = "measured_native_plate"
    with pytest.raises(ValueError):
        B.build(round4_receipts=r4, barmap=bm)


def test_duplicate_overlay_keys_and_wrong_size_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError, match="duplicate"):
        M.read(path)
    doc = document()
    p = point(tmp_path, 4320)
    row = p["measurement"]["frames"][0]
    image = tmp_path / row["path"]
    Image.new("RGB", (480, 201)).save(image)
    row["sha256"] = M.sha256(image)
    doc["hooks"]["crown_beacons"] = p
    with pytest.raises(ValueError, match="dimensions"):
        B.build(measurements=doc, frame_root=tmp_path)


def composite_case(tmp_path, frame=2740):
    """Synthetic composite namespace, with actual file and EDL dependencies."""
    doc = document()
    p = point(tmp_path, frame, claim="onset", first=frame-1)
    p.update(landmark="synthetic_deep_composite", editorial_reference="deep_ember_sweep",
             measurement_scope=M.COMPOSITE_SCOPE)
    deep = tmp_path / "deep_evidence"
    (deep / "composites").mkdir(parents=True)
    for row in p["measurement"]["frames"]:
        original = tmp_path / row["path"]
        target = deep / "composites" / original.name
        target.write_bytes(original.read_bytes())
        row.update(root="soundd_deep_composites", path=target.relative_to(deep).as_posix())
    local = tmp_path / "local_assets"
    local.mkdir()
    source = local / "synthetic_coefficient.bin"
    source.write_bytes(b"Synthetic dependency, never a production coefficient.")
    p["measurement"]["dependencies"] = [dict(root="cutd_local_renders", path=source.name, sha256=M.sha256(source))]
    edl = M.read(M.ROOT / M.INPUT_FILES[0])
    p["measurement"]["edl_dependencies"] = {
        key:[deepcopy(r) for r in edl[key] if r["f0"] <= frame and r["f1"] > frame-1]
        for key in ("shots", "transitions")}
    doc["events"]["D.11.C5.burn.deep"] = p
    return doc, p, edl, dict(soundd_deep_composites=deep, cutd_local_renders=local)


def validate_composite(doc, edl, roots, tmp_path):
    bm = M.read(M.ROOT / M.INPUT_FILES[1])
    return M.validate_overlay(doc, hooks={h["id"]:h for h in bm["sync"]}, request_ids=set(),
        reuse_ids={"D.11.C5.burn.deep"}, frame_root=tmp_path, evidence_roots=roots, edl=edl)


def test_composite_scope_and_portable_roots_preserve_unfinished_picture_claim(tmp_path):
    doc, point_row, edl, roots = composite_case(tmp_path)
    result = validate_composite(doc, edl, roots, tmp_path)
    assert result["events"]["D.11.C5.burn.deep"]["measurement_scope"] == M.COMPOSITE_SCOPE
    plan = B.build(measurements=doc, frame_root=tmp_path, evidence_roots=roots)
    bound = next(r for r in plan["reuse_plan"]["EXTRA_EVENTS"] if r["id"] == "D.11.C5.burn.deep")
    assert bound["measurement_scope"] == M.COMPOSITE_SCOPE
    assert bound["donor_hit_f"] == 1699 and bound["hit_f"] == 2740


def test_composite_hook_binding_cannot_relabel_pre_finish_evidence_as_raw_plate(tmp_path):
    doc, p, edl, roots = composite_case(tmp_path, frame=4320)
    doc["events"] = {}
    doc["hooks"]["crown_beacons"] = p
    plan = B.build(measurements=doc, frame_root=tmp_path, evidence_roots=roots)
    crowns = [r for r in plan["new_events"] if r["request_id"] in ("D.new.crowns.left", "D.new.crowns.right")]
    assert len(crowns) == 2
    assert all(r["source"] == "measured" and r["measurement_scope"] == M.COMPOSITE_SCOPE for r in crowns)
    assert all(r["measurement"]["frames"] == p["measurement"]["frames"] for r in crowns)
    p["measurement_scope"] = M.NATIVE_SCOPE
    with pytest.raises(ValueError, match="repo-relative"):
        B.build(measurements=doc, frame_root=tmp_path, evidence_roots=roots)


@pytest.mark.parametrize("mutation", ["file_hash", "dependency_hash", "dependency_path", "edl_take",
                                     "omitted_shot", "new_transition", "promoted_scope", "raw_scope", "path"])
def test_composite_dependencies_fail_closed(tmp_path, mutation):
    doc, p, edl, roots = composite_case(tmp_path)
    proof = p["measurement"]
    if mutation == "file_hash": proof["frames"][0]["sha256"] = "0" * 64
    elif mutation == "dependency_hash": proof["dependencies"][0]["sha256"] = "0" * 64
    elif mutation == "dependency_path": proof["dependencies"][0]["path"] = "../synthetic_coefficient.bin"
    elif mutation == "edl_take": proof["edl_dependencies"]["shots"][0]["takes"][0]["off"] += 1
    elif mutation == "omitted_shot": proof["edl_dependencies"]["shots"] = []
    elif mutation == "new_transition": edl["transitions"].append(dict(kind="synthetic", f0=2738, f1=2742))
    elif mutation == "promoted_scope": p["measurement_scope"] = "native_finished_picture"
    elif mutation == "raw_scope": p["measurement_scope"] = M.NATIVE_SCOPE
    elif mutation == "path": proof["frames"][0]["path"] = "../composites/f_02739.png"
    with pytest.raises(ValueError): validate_composite(doc, edl, roots, tmp_path)


def test_null_composite_anchor_keeps_evidence_and_still_checks_its_hashes(tmp_path):
    doc, p, edl, roots = composite_case(tmp_path)
    p.update(frame=None, status="unmeasured", reason="Visible onset does not establish the donor envelope crest")
    result = validate_composite(doc, edl, roots, tmp_path)
    assert not result["events"] and result["event_observations"]["D.11.C5.burn.deep"] == p
    p["measurement"]["frames"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frame SHA256"): validate_composite(doc, edl, roots, tmp_path)


def test_composite_env_root_is_explicit_and_path_escape_is_rejected(tmp_path, monkeypatch):
    doc, p, edl, roots = composite_case(tmp_path)
    monkeypatch.delenv("SOUNDD_DEEP_EVIDENCE", raising=False)
    monkeypatch.delenv("CUTD_LOCAL_RENDERS", raising=False)
    with pytest.raises(ValueError, match="unconfigured evidence root"):
        validate_composite(doc, edl, {}, tmp_path)
    monkeypatch.setenv("SOUNDD_DEEP_EVIDENCE", str(roots["soundd_deep_composites"]))
    monkeypatch.setenv("CUTD_LOCAL_RENDERS", str(roots["cutd_local_renders"]))
    assert validate_composite(doc, edl, {}, tmp_path)["events"]
    row = p["measurement"]["frames"][0]
    path = roots["soundd_deep_composites"] / row["path"]
    original = tmp_path / "renders/synthetic_test_fixture" / path.name
    path.unlink()
    path.symlink_to(original)
    with pytest.raises(ValueError, match="escapes configured root"):
        validate_composite(doc, edl, {}, tmp_path)


def test_raw_source_stem_and_suppression_scope_are_checked(tmp_path):
    doc = document()
    p = point(tmp_path, 4320)
    doc["hooks"]["crown_beacons"] = p
    p["measurement"]["frames"][0]["source_stem"] = "different_synthetic_fixture"
    with pytest.raises(ValueError, match="source stem"):
        B.build(measurements=doc, frame_root=tmp_path)
    doc["hooks"] = {}
    doc["suppressions"]["D.new.leaf.quill"] = dict(reason="synthetic test only", source="test fixture",
        measurement_scope="native_finished_picture", measurement=evidence(tmp_path, 8320, 8320, claim="static_state"))
    with pytest.raises(ValueError, match="measurement scope"):
        B.build(measurements=doc, frame_root=tmp_path)


def test_native_editorial_suppression_checks_its_transition_dependency(tmp_path):
    doc = document()
    edl = M.read(M.ROOT / M.INPUT_FILES[0])
    proof = evidence(tmp_path, 8639, 8640, claim="static_state")
    proof["edl_dependencies"] = {field: [r for r in edl[field] if r["f0"] <= 8640 and r["f1"] > 8639]
                                 for field in ("shots", "transitions")}
    doc["suppressions"]["D.new.page.to_blank"] = dict(reason="Synthetic selected dissolve decision",
        source="Synthetic test only", measurement=proof)
    plan = B.build(measurements=doc, frame_root=tmp_path)
    assert any(r["request_id"] == "D.new.page.to_blank" for r in plan["suppressed_requests"])
    proof["edl_dependencies"]["transitions"][0]["f0"] += 1
    with pytest.raises(ValueError, match="EDL dependency"):
        B.build(measurements=doc, frame_root=tmp_path)
