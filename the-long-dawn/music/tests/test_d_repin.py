"""Editorial re-pin accepts the unbound page turn and rejects changed sources."""
from copy import deepcopy
import hashlib
import json

import pytest

from bindings_D import EDITORIAL_FILES, ROOT, TABLE, sha256
import repin_D as R


@pytest.fixture
def inputs():
    read = lambda rel: json.loads((ROOT / rel).read_text())
    document = json.loads(TABLE.read_text())
    document.pop("edit_revision", None)
    edl = read(R.EDIT_FILES[1])
    return dict(document=document, edl=edl, generated=deepcopy(edl),
                bar=read(EDITORIAL_FILES[0]), cues=read(EDITORIAL_FILES[1]),
                receipts={p: read(p) for p in EDITORIAL_FILES[2:]},
                music_hashes={p: sha256(ROOT / p) for p in EDITORIAL_FILES},
                edit_hashes={p: sha256(ROOT / p) for p in R.EDIT_FILES})


def establish(inputs):
    revision = R.build_revision(**inputs)
    inputs["document"]["edit_revision"] = revision
    return revision


def test_build_keeps_events_and_editorial_evidence_immutable(inputs):
    before = deepcopy(inputs)
    revision = R.build_revision(**inputs)
    assert inputs == before
    assert revision["sha256"] == inputs["edit_hashes"]
    assert revision["allowed_transition_window"] == [8320, 8880]
    assert len(revision["protected_signature"]["shots"]) == len(inputs["edl"]["shots"])
    assert "native onset" in revision["scope"]


def test_future_page_turn_can_change_without_rewriting_source_mapping(inputs):
    previous = establish(inputs)
    transition = next(t for t in inputs["edl"]["transitions"] if t.get("cut") == 8640)
    transition.update(f0=8628, f1=8654, tilt=11.0)
    inputs["generated"] = deepcopy(inputs["edl"])
    inputs["edit_hashes"] = {p: "f" * 64 for p in R.EDIT_FILES}
    updated = R.build_revision(**inputs)
    assert updated["sha256"] != previous["sha256"]
    assert updated["protected_signature"] == previous["protected_signature"]


@pytest.mark.parametrize("change", ["offset", "stem", "need", "baked_text", "hold", "shot_bound", "transition"])
def test_source_or_protected_timing_changes_cannot_be_re_pinned(inputs, change):
    establish(inputs)
    shot = next(s for s in inputs["edl"]["shots"] if s["code"] == "D23")
    if change == "shot_bound":
        shot["f0"] += 1
    elif change == "transition":
        next(t for t in inputs["edl"]["transitions"] if t.get("cut") == 8880)["f0"] -= 1
    else:
        take = shot["takes"][0]
        take[{"offset": "off", "stem": "stem", "need": "need", "baked_text": "baked_text", "hold": "hold"}[change]] = {
            "offset": 1, "stem": "different_plate", "need": [5841, 6079],
            "baked_text": [], "hold": 5900}[change]
    inputs["generated"] = deepcopy(inputs["edl"])
    with pytest.raises(ValueError, match="source mapping|fixed grid"):
        R.build_revision(**inputs)


def test_page_turn_change_cannot_escape_its_unbound_window(inputs):
    establish(inputs)
    next(t for t in inputs["edl"]["transitions"] if t.get("cut") == 8640)["f0"] = 8319
    inputs["generated"] = deepcopy(inputs["edl"])
    with pytest.raises(ValueError, match="bounded transition"):
        R.build_revision(**inputs)


@pytest.mark.parametrize("extra", [
    dict(f0=8350, f1=8600, kind="floor", k0=1, k1=1),
    dict(f0=8632, f1=8656, cut=8650, kind="page_turn"),
    dict(f0=8632, f1=8656, cut=8640, kind="page_turn"),
])
def test_unrelated_or_duplicate_transition_inside_page_window_is_rejected(inputs, extra):
    establish(inputs)
    inputs["edl"]["transitions"].append(extra)
    inputs["generated"] = deepcopy(inputs["edl"])
    with pytest.raises(ValueError, match="protected transition changed|bounded transition"):
        R.build_revision(**inputs)


@pytest.mark.parametrize("patch", [dict(cut=8639), dict(f0=8641), dict(f1=8639)])
def test_page_turn_exception_must_span_the_correct_shot_boundary(inputs, patch):
    establish(inputs)
    next(t for t in inputs["edl"]["transitions"] if t.get("cut") == 8640).update(patch)
    inputs["generated"] = deepcopy(inputs["edl"])
    with pytest.raises(ValueError, match="bounded transition"):
        R.build_revision(**inputs)


def test_stale_export_is_rejected_even_before_first_pin(inputs):
    inputs["generated"]["shots"][0]["desc"] += " changed"
    with pytest.raises(ValueError, match="export differs"):
        R.build_revision(**inputs)


def test_grid_change_is_rejected_even_when_generator_matches(inputs):
    inputs["edl"]["frames"] = 9201
    inputs["generated"] = deepcopy(inputs["edl"])
    with pytest.raises(ValueError, match="EDL grid changed"):
        R.build_revision(**inputs)


@pytest.mark.parametrize("path", EDITORIAL_FILES)
def test_music_json_pins_are_never_blindly_refreshed(inputs, path):
    inputs["music_hashes"][path] = "f" * 64
    with pytest.raises(ValueError, match="music editorial JSON changed"):
        R.build_revision(**inputs)


def test_canonical_evidence_cannot_be_refreshed_by_repin(inputs):
    inputs["bar"]["sync"][0]["f"] += 1
    with pytest.raises(ValueError, match="canonical evidence changed"):
        R.build_revision(**inputs)


def test_cues_must_still_agree_with_the_map(inputs):
    inputs["cues"]["events"]["ap2_wind"]["source"] = "changed source"
    with pytest.raises(ValueError, match="map/cue disagreement"):
        R.build_revision(**inputs)


def test_measurement_summary_must_still_agree_with_its_samples(inputs):
    inputs["receipts"][EDITORIAL_FILES[3]]["deep_exit_source"]["visible_glow_onset"] = 2953
    with pytest.raises(ValueError, match="cannot establish a visible glow"):
        R.build_revision(**inputs)


def test_repin_splices_only_revision_bytes(inputs):
    # Deliberate non-canonical formatting makes reserialising the events fail.
    text = '{\n  "events": { "a": {"note":"escaped \\\"text\\\"", "frame":1}},\n "x": [1, 2]\n}\n'
    revision = R.build_revision(**inputs)
    updated = R.replace_revision(text, revision)
    old_span, _ = R._member_spans(text)
    new_span, _ = R._member_spans(updated)
    assert text[slice(*old_span["events"])] == updated[slice(*new_span["events"])]
    assert json.loads(updated)["edit_revision"] == revision
    assert R.replace_revision(updated, revision) == updated
    revision["sha256"][R.EDIT_FILES[0]] = "f" * 64
    again = R.replace_revision(updated, revision)
    next_span, _ = R._member_spans(again)
    assert text[slice(*old_span["events"])] == again[slice(*next_span["events"])]


@pytest.fixture
def isolated_command(tmp_path, monkeypatch, inputs):
    table = tmp_path / "bindings.json"
    table.write_text(json.dumps(inputs["document"], indent=2, ensure_ascii=False) + "\n")
    revision = R.build_revision(**inputs)
    monkeypatch.setattr(R, "current_revision", lambda document: deepcopy(revision))
    # Unit fixture isolates the write path; the production test below checks
    # actual BoundMap receipts, including every native JPEG hash.
    monkeypatch.setattr(R, "BoundMap", lambda document: None)
    monkeypatch.setattr(R, "scoped_picture_receipts", lambda document: None)
    return table, revision


def test_command_requires_write_then_check_and_is_idempotent(isolated_command):
    table, revision = isolated_command
    before = table.read_text()
    with pytest.raises(ValueError, match="pins are absent or stale"):
        R.repin(table)
    assert table.read_text() == before
    assert R.repin(table, write=True)["changed"] is True
    assert json.loads(table.read_text())["edit_revision"] == revision
    pinned = table.read_bytes()
    assert R.repin(table)["changed"] is False
    assert R.repin(table, write=True)["changed"] is False
    assert table.read_bytes() == pinned


def test_command_check_rejects_stale_edit_hashes(isolated_command):
    table, _ = isolated_command
    R.repin(table, write=True)
    document = json.loads(table.read_text())
    document["edit_revision"]["sha256"][R.EDIT_FILES[0]] = "0" * 64
    table.write_text(json.dumps(document))
    before = table.read_bytes()
    with pytest.raises(ValueError, match="pins are absent or stale"):
        R.repin(table)
    assert table.read_bytes() == before


def test_invalid_native_receipt_blocks_repin_without_writing(isolated_command, monkeypatch):
    table, _ = isolated_command
    before = table.read_bytes()
    def fail(document):
        raise ValueError("measured frame asset is missing")
    monkeypatch.setattr(R, "BoundMap", fail)
    with pytest.raises(ValueError, match="measured frame asset is missing"):
        R.repin(table, write=True)
    assert table.read_bytes() == before


def test_command_will_not_overwrite_a_parallel_lane_edit(isolated_command, monkeypatch):
    table, revision = isolated_command
    def concurrent(document):
        table.write_text(table.read_text() + "\n")
        return revision
    monkeypatch.setattr(R, "current_revision", concurrent)
    with pytest.raises(ValueError, match="changed during preflight"):
        R.repin(table, write=True)
    assert "edit_revision" not in json.loads(table.read_text())


def test_production_editorial_pins_match_current_generator_export_and_receipts():
    receipt = R.repin(write=False)
    assert receipt["ok"] and not receipt["changed"]
    assert set(receipt["sha256"]) == set(R.EDIT_FILES)


@pytest.fixture
def scoped_clock(tmp_path):
    """Synthetic bytes test provenance only; they are not picture measurements."""
    render_root = tmp_path / "renders"
    asset = render_root / "embers_D_race/f_02400.jpg"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"SYNTHETIC FRAME FIXTURE: no optical evidence")
    clock = tmp_path / R.CLOCK_SOURCE
    clock.parent.mkdir(parents=True)
    clock.write_text("# SYNTHETIC CLOCK FIXTURE: no production schedule\n")
    row = dict(frame=2400, role="source_clock", status="estimated", picture_frame=None,
               render_stem="embers_D_race",
               frame_evidence=dict(frame=2400, source="SYNTHETIC TEST FIXTURE",
                                   measured_ref="fixture://clock-frame",
                                   measurement_scope="Synthetic receipt validation; no optical onset",
                                   measurement=dict(kind="render_D", stem="embers_D_race", frame=2400,
                                                    frame_file=asset.name,
                                                    sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                                                    method="Synthetic bytes only")),
               clock_source=dict(path=R.CLOCK_SOURCE, sha256=hashlib.sha256(clock.read_bytes()).hexdigest()))
    return dict(events={"race_beat_surges": row}), render_root, tmp_path


def test_scoped_clock_validates_frame_and_source_without_promotion(scoped_clock):
    document, renders, source = scoped_clock
    before = deepcopy(document)
    R.scoped_picture_receipts(document, render_root=renders, source_root=source)
    assert document == before
    assert document["events"]["race_beat_surges"]["status"] == "estimated"
    assert document["events"]["race_beat_surges"]["picture_frame"] is None


@pytest.mark.parametrize("failure, message", [
    ("role", "requires role source_clock"),
    ("status", "must remain a direct estimate"),
    ("onset", "must remain a direct estimate"),
    ("picture_frame", "must remain a direct estimate"),
    ("alias", "must remain a direct estimate"),
    ("missing_evidence", "matching frame evidence"),
    ("frame", "matching frame evidence"),
    ("source", "source, reference and scope"),
    ("reference", "source, reference and scope"),
    ("scope", "source, reference and scope"),
    ("method", "method, reference and scope"),
    ("receipt_frame", "receipt frame differs"),
    ("missing_asset", "frame asset is missing"),
    ("asset_hash", "frame asset hash mismatch"),
    ("missing_clock", "explicit race clock source"),
    ("clock_path", "explicit race clock source"),
    ("clock_hash", "clock source is missing or its hash changed"),
    ("clock_asset", "clock source is missing or its hash changed"),
])
def test_scoped_clock_negative_controls(scoped_clock, failure, message):
    document, renders, source = scoped_clock
    row = document["events"]["race_beat_surges"]
    evidence = row["frame_evidence"]
    if failure == "role":
        row["role"] = "picture"
    elif failure == "status":
        row["status"] = "measured"
    elif failure == "onset":
        row["measurement"] = deepcopy(evidence["measurement"])
    elif failure == "picture_frame":
        row["picture_frame"] = 2400
    elif failure == "alias":
        row["anchor"] = "another_event"
    elif failure == "missing_evidence":
        row.pop("frame_evidence")
    elif failure == "frame":
        evidence["frame"] += 1
    elif failure in ("source", "reference", "scope"):
        evidence[{"source": "source", "reference": "measured_ref", "scope": "measurement_scope"}[failure]] = ""
    elif failure == "method":
        evidence["measurement"]["method"] = ""
    elif failure == "receipt_frame":
        evidence["measurement"]["frame"] += 1
    elif failure == "missing_asset":
        (renders / "embers_D_race/f_02400.jpg").unlink()
    elif failure == "asset_hash":
        (renders / "embers_D_race/f_02400.jpg").write_bytes(b"corrupted synthetic bytes")
    elif failure == "missing_clock":
        row.pop("clock_source")
    elif failure == "clock_path":
        row["clock_source"]["path"] = "shots/embers/another_clock.py"
    elif failure == "clock_hash":
        row["clock_source"]["sha256"] = "0" * 64
    elif failure == "clock_asset":
        (source / R.CLOCK_SOURCE).unlink()
    with pytest.raises(ValueError, match=message):
        R.scoped_picture_receipts(document, render_root=renders, source_root=source)


def test_write_invokes_scoped_picture_validation_before_changing_pins(isolated_command, monkeypatch):
    table, _ = isolated_command
    before = table.read_bytes()
    def reject(document):
        raise ValueError("scoped frame failed")
    monkeypatch.setattr(R, "scoped_picture_receipts", reject)
    with pytest.raises(ValueError, match="scoped frame failed"):
        R.repin(table, write=True)
    assert table.read_bytes() == before


def reviewed_fixture(inputs, cuts=(4080, 4240, 4560, 5840, 6080, 6640)):
    """Synthetic transition windows exercise authority, not picture decisions.

    The window must differ from whatever the adopted EDL holds at the cut: round 8 adopted cut-6..cut+6
    dissolves at 4240 and 4560, which a symmetric synthetic window would merely restate (owner, 30 Sep)."""
    previous = establish(inputs)
    changes = []
    for cut in cuts:
        before = [t for t in previous["protected_signature"]["protected_transitions"]
                  if t.get("cut") == cut]
        after = [dict(f0=cut - 5, f1=cut + 7, cut=cut, kind="dissolve")]
        assert after != before, f"D{cut}: synthetic window restates the adopted record"
        changes.append(dict(cut=cut, before=deepcopy(before), after=after))
        inputs["edl"]["transitions"] = [t for t in inputs["edl"]["transitions"] if t.get("cut") != cut]
        inputs["edl"]["transitions"].extend(deepcopy(after))
    inputs["generated"] = deepcopy(inputs["edl"])
    inputs["edit_hashes"] = {p: "e" * 64 for p in R.EDIT_FILES}
    return dict(schema=R.REVIEW_SCHEMA, changes=changes)


def test_reviewed_six_join_records_update_signature_without_changing_music(inputs):
    approval = reviewed_fixture(inputs)
    before = deepcopy(inputs)
    updated = R.build_revision(**inputs, reviewed_transitions=approval)
    assert inputs == before
    assert updated["protected_signature"] == R._signature(inputs["edl"])
    assert updated["transition_review"]["changes"] == approval["changes"]
    assert len(updated["transition_review"]["approval_sha256"]) == 64
    # Subsequent ordinary preflight retains the audit without reusing approval.
    inputs["document"]["edit_revision"] = updated
    assert R.build_revision(**inputs) == updated
    with pytest.raises(ValueError, match="before records do not match"):
        R.build_revision(**inputs, reviewed_transitions=approval)


def test_changed_six_joins_still_fail_without_explicit_approval(inputs):
    reviewed_fixture(inputs)
    with pytest.raises(ValueError, match="protected transition changed"):
        R.build_revision(**inputs)


@pytest.mark.parametrize("side", ["before", "after"])
def test_reviewed_record_must_match_both_revisions_exactly(inputs, side):
    # Mutating a 'before' record needs a reviewable cut that currently has one (6640 became a straight cut in r8).
    current = R._signature(inputs["edl"])["protected_transitions"]
    cut = next(c for c in sorted(R.REVIEWABLE_CUTS) if any(t.get("cut") == c for t in current))
    approval = reviewed_fixture(inputs, (cut,))
    approval["changes"][0][side][0]["f0"] += 1
    with pytest.raises(ValueError, match=f"{side} records do not match"):
        R.build_revision(**inputs, reviewed_transitions=approval)


@pytest.mark.parametrize("failure", ["unlisted", "duplicate", "omitted", "unchanged", "too_wide", "bad_schema"])
def test_review_scope_cannot_expand_or_hide_unapproved_changes(inputs, failure):
    approval = reviewed_fixture(inputs)
    if failure == "unlisted":
        approval["changes"][0]["cut"] = 3760
    elif failure == "duplicate":
        approval["changes"].append(deepcopy(approval["changes"][0]))
    elif failure == "omitted":
        approval["changes"].pop()
    elif failure == "unchanged":
        approval["changes"][0]["after"] = deepcopy(approval["changes"][0]["before"])
    elif failure == "too_wide":
        approval["changes"][0]["after"][0]["f0"] = 0
        next(t for t in inputs["edl"]["transitions"] if t.get("cut") == 4080)["f0"] = 0
        inputs["generated"] = deepcopy(inputs["edl"])
    elif failure == "bad_schema":
        approval["schema"] = "unrecognised"
    with pytest.raises(ValueError, match="unlisted|duplicate|unreviewed|must change|inside its|invalid reviewed"):
        R.build_revision(**inputs, reviewed_transitions=approval)


@pytest.mark.parametrize("change", ["source", "map", "grid"])
def test_transition_approval_cannot_override_other_source_guards(inputs, change):
    approval = reviewed_fixture(inputs)
    if change == "source":
        next(s for s in inputs["edl"]["shots"] if s["code"] == "D23")["takes"][0]["off"] += 1
        inputs["generated"] = deepcopy(inputs["edl"])
    elif change == "map":
        inputs["music_hashes"][EDITORIAL_FILES[0]] = "0" * 64
    else:
        inputs["edl"]["frames"] += 1
        inputs["generated"] = deepcopy(inputs["edl"])
    with pytest.raises(ValueError, match="source mapping|music editorial JSON|EDL grid"):
        R.build_revision(**inputs, reviewed_transitions=approval)


def test_transition_approval_cannot_bootstrap_its_own_before_evidence(inputs):
    approval = reviewed_fixture(inputs)
    inputs["document"].pop("edit_revision")
    with pytest.raises(ValueError, match="requires existing editorial pins"):
        R.build_revision(**inputs, reviewed_transitions=approval)


def test_reviewed_transition_command_loads_private_file_and_preserves_event_bytes(
        inputs, tmp_path, monkeypatch):
    approval = reviewed_fixture(inputs)
    table = tmp_path / "bindings.json"
    table.write_text(json.dumps(inputs["document"], indent=2) + "\n")
    private = tmp_path / "review.json"
    private.write_text(json.dumps(approval))
    old_text = table.read_text()
    span, _ = R._member_spans(old_text)
    monkeypatch.setattr(R, "BoundMap", lambda document: None)
    monkeypatch.setattr(R, "scoped_picture_receipts", lambda document: None)
    def current(document, *, reviewed_transitions=None):
        values = dict(inputs, document=document)
        return R.build_revision(**values, reviewed_transitions=reviewed_transitions)
    monkeypatch.setattr(R, "current_revision", current)
    with pytest.raises(ValueError, match="requires --write"):
        R.repin(table, reviewed_transitions=private)
    result = R.repin(table, write=True, reviewed_transitions=private)
    new_text = table.read_text()
    new_span, _ = R._member_spans(new_text)
    assert result["changed"] and "transition_review" in result
    assert old_text[slice(*span["events"])] == new_text[slice(*new_span["events"])]
    assert R.repin(table)["changed"] is False
    # Approval is not a wildcard accepted again on the already-updated baseline.
    with pytest.raises(ValueError, match="before records do not match"):
        R.repin(table, write=True, reviewed_transitions=private)
    assert table.read_text() == new_text
