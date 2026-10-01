"""Re-pin adversarial controls with explicitly synthetic source evidence."""
from copy import deepcopy
import hashlib
import json

from PIL import Image
import pytest

import repin_sound_d as R
import sound_d_measurements as M


def encoded(document):
    return json.dumps(document, indent=2) + "\n"


@pytest.fixture
def case(tmp_path):
    take = dict(stem="synthetic_test_fixture", mode="exact", off=0, final_eligible=True,
                crop=None, grade=None, matte=None, under=None, video=None, add=None, need=[4240, 4559])
    old = dict(cut="D", fps=24, frames=9200, shots=[dict(code="D18", f0=4240, f1=4560, takes=[])], transitions=[])
    current = deepcopy(old)
    current["shots"][0]["takes"] = [take]
    bm = dict(sync=[dict(id="crown_beacons", f=4320, timing_status="estimated", timing_window=[4240, 4560])])
    for name, value in zip(M.INPUT_FILES, [current, bm, {}, {}, {}]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded(value))
    frames = []
    for frame in (4319, 4320):
        path = tmp_path / "renders/synthetic_test_fixture" / f"f_{frame:05d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 804), (12, 34, 56)).save(path)
        frames.append(dict(frame=frame, path=path.relative_to(tmp_path).as_posix(), sha256=M.sha256(path)))
    proof = dict(method="Synthetic test fixture only", claim="onset", predicate="test-only",
                 frame_size=[1920, 804], frames=frames, coverage=dict(first=4319, last=4320, complete=True, missing=[]))
    point = dict(frame=4320, status="measured", source="Synthetic test only", measured_ref="test-only",
                 measurement_scope=M.NATIVE_SCOPE, measurement=proof)
    base = dict(schema=M.SCHEMA, cut="D", fps=24, frames=9200, input_sha256=M.raw_input_hashes(tmp_path),
                hooks={"crown_beacons":point}, events={}, series={}, suppressions={})
    old_text = encoded(old)
    base["input_sha256"][M.INPUT_FILES[0]] = hashlib.sha256(old_text.encode()).hexdigest()
    return dict(root=tmp_path, base=base, before=old_text, current=current)


def run(case, **kwargs):
    return R.repin(case["base"], root=case["root"], base_edl_text=case["before"],
                   request_ids={"D.new.crowns.left"}, reuse_ids=set(), **kwargs)


def test_adoption_repin_preserves_measurements_and_source_inputs(case):
    before = deepcopy(case["base"])
    result = run(case)
    assert result["hooks"] == before["hooks"]
    assert case["base"] == before
    assert result["input_sha256"] == M.raw_input_hashes(case["root"])
    assert result["repin"]["base_overlay_sha256"] == R.digest(before)
    assert result["repin"]["adopted_raw_evidence"] == [dict(shot="D18", stem="synthetic_test_fixture", offset_f=0)]


@pytest.mark.parametrize("mutation", ["wrong_stem", "offset", "hold", "crop", "ineligible", "multiple", "bounds", "transition"])
def test_changed_take_or_unsupported_editorial_edit_is_not_hash_laundered(case, mutation):
    doc = case["current"]
    take = doc["shots"][0]["takes"][0]
    if mutation == "wrong_stem": take["stem"] = "different_fixture"
    elif mutation == "offset": take["off"] = 1
    elif mutation == "hold": take["hold"] = 4320
    elif mutation == "crop": take["crop"] = [0, 0, 100, 100]
    elif mutation == "ineligible": take["final_eligible"] = False
    elif mutation == "multiple": doc["shots"][0]["takes"].append(deepcopy(take))
    elif mutation == "bounds": doc["shots"][0]["f0"] += 1
    elif mutation == "transition": doc["transitions"] = [dict(kind="dissolve", f0=4280, f1=4360)]
    (case["root"] / M.INPUT_FILES[0]).write_text(encoded(doc))
    with pytest.raises(ValueError): run(case)


def test_later_d31_transition_edit_uses_verified_snapshot(case):
    first = run(case)
    case["current"]["transitions"] = [dict(kind="page_turn", f0=8620, f1=8660)]
    (case["root"] / M.INPUT_FILES[0]).write_text(encoded(case["current"]))
    second = R.repin(first, root=case["root"], request_ids=set(), reuse_ids=set())
    assert second["hooks"] == first["hooks"]
    assert second["repin"]["previous_repin_sha256"] == R.digest(first["repin"])
    bad = deepcopy(first)
    bad["repin"]["edl_source_text"] += " "
    with pytest.raises(ValueError, match="snapshot"):
        R.repin(bad, root=case["root"], request_ids=set(), reuse_ids=set())


def test_stale_frame_fails_until_explicit_replacement_arrives(case):
    old_point = deepcopy(case["base"]["hooks"]["crown_beacons"])
    file = case["root"] / old_point["measurement"]["frames"][1]["path"]
    Image.new("RGB", (1920, 804), (71, 41, 23)).save(file)
    with pytest.raises(ValueError, match="frame SHA256"): run(case)
    replacement = deepcopy(case["base"])
    replacement["input_sha256"] = M.raw_input_hashes(case["root"])
    replacement["observations"] = {"unmeasured_fixture":dict(frame=None, reason="No onset measured in this synthetic test")}
    replacement["hooks"]["crown_beacons"]["measurement"]["frames"][1]["sha256"] = M.sha256(file)
    result = run(case, replacements=[replacement])
    change = result["repin"]["replacements"][0]
    assert change["previous_sha256"] == R.digest(old_point)
    assert result["hooks"] == replacement["hooks"]
    assert result["repin"]["replacement_annotations"][0]["annotations"]["observations"] == replacement["observations"]
    with pytest.raises(ValueError, match="multiple replacements"):
        run(case, replacements=[replacement, replacement])


def test_changed_source_code_rejects_repin(case):
    path = case["root"] / "shots/synthetic_fixture.py"
    path.parent.mkdir()
    path.write_text("# Synthetic fixture only.\n")
    proof = case["base"]["hooks"]["crown_beacons"]["measurement"]
    proof["source_sha256"] = {"shots/synthetic_fixture.py": M.sha256(path)}
    assert run(case)
    path.write_text("# Changed fixture.\n")
    with pytest.raises(ValueError, match="source evidence SHA256"): run(case)


@pytest.mark.parametrize("name", M.INPUT_FILES[1:])
def test_non_edl_change_requires_exact_review_and_keeps_evidence_checks(case, name):
    path = case["root"] / name
    original = path.read_text()
    path.write_text(original + "\n")  # Even a raw-byte-only change must be reviewed.
    with pytest.raises(ValueError, match="explicit retained-evidence review"): run(case)
    review = dict(schema=R.REVIEW_SCHEMA, retained_evidence_review="Test-only whitespace review",
                  files={name:dict(from_sha256=case["base"]["input_sha256"][name],
                                   to_sha256=M.sha256(path), reason="Only JSON whitespace changed in this fixture")})
    result = run(case, review=review)
    assert result["repin"]["metadata_review"] == review
    assert result["hooks"] == case["base"]["hooks"]
    review["files"][name]["to_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash/rationale"): run(case, review=review)


def test_no_snapshot_or_wrong_snapshot_is_refused(case):
    with pytest.raises(ValueError, match="base-edl-ref"):
        R.repin(case["base"], root=case["root"], request_ids=set(), reuse_ids=set())
    case["before"] += " "
    with pytest.raises(ValueError, match="snapshot"): run(case)


def test_raw_edl_snapshot_preserves_line_ending_bytes(case):
    path = case["root"] / M.INPUT_FILES[0]
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    first = run(case)
    assert "\r\n" in first["repin"]["edl_source_text"]
    second = R.repin(first, root=case["root"], request_ids=set(), reuse_ids=set())
    assert second["input_sha256"] == first["input_sha256"]


def test_review_loaded_from_file_is_serializable(case):
    path = case["root"] / M.INPUT_FILES[2]
    path.write_text(path.read_text() + "\n")
    review = dict(schema=R.REVIEW_SCHEMA, retained_evidence_review="Synthetic whitespace-only review",
                  files={M.INPUT_FILES[2]:dict(from_sha256=case["base"]["input_sha256"][M.INPUT_FILES[2]],
                    to_sha256=M.sha256(path), reason="Test fixture only")})
    review_path = case["root"] / "review.json"
    review_path.write_text(encoded(review))
    result = run(case, review=review_path)
    assert json.loads(json.dumps(result))["repin"]["metadata_review"] == review


def test_cli_cannot_overwrite_base_or_metadata(case, capsys):
    path = case["root"] / "base.json"
    path.write_text(encoded(case["base"]))
    before = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        R.main(["--base", str(path), "--out", str(path)])
    assert exc.value.code == 2 and path.read_bytes() == before
    assert "preserve every input" in capsys.readouterr().err


def test_explicit_end_bounds_survive_repin_as_validated_landmarks(case):
    replacement=deepcopy(case['base'])
    replacement['hooks']={}
    point=deepcopy(case['base']['hooks']['crown_beacons'])
    point['measurement']['claim']='completion'
    point['offset_f']=1
    replacement['bounds']={'D.new.crowns.left':{'end':point}}
    result=run(case,replacements=[replacement])
    assert result['bounds']==replacement['bounds']
    assert result['repin']['replacements'][0]['field']=='bounds'
    assert 'bounds' not in result['repin']['replacement_annotations'][0]['annotations']
    point['measurement']['frames'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='SHA256'):
        run(case,replacements=[replacement])


def test_cli_forwards_picture_revision_to_binding(case,tmp_path,monkeypatch):
    import sound_d_binding as B
    result=dict(repin=dict(replacements=[],adopted_raw_evidence=[],scope='Synthetic CLI contract test'),
                input_sha256={})
    calls=[]
    monkeypatch.setattr(R,'repin',lambda *a,**k:result)
    monkeypatch.setattr(B,'build',lambda **kw:calls.append(kw) or {'phase':5})
    base=tmp_path/'base.json';base.write_text('{}\n')
    out=tmp_path/'out.json'
    R.main(['--base',str(base),'--race-revision','round5','--picture-revision','locked','--out',str(out)])
    assert calls[0]['picture_revision']=='locked' and calls[0]['race_revision']=='round5'
    receipt=json.loads(out.with_suffix('.receipt.json').read_text())
    assert receipt['picture_revision']=='locked'


@pytest.fixture
def d32_case(tmp_path):
    """Synthetic pixels/code with the production D32 selector, never real evidence."""
    take = dict(stem="cand_pen_soft_spine_metal", off=-3200, mode="exact", final_eligible=True,
                matte="cand_pen_soft_spine_metal_matte", under=None, crop=None, grade=None,
                add=None, video=None, need=None)
    shot = dict(code="D32", f0=8640, f1=8880, takes=[take])
    edl = dict(cut="D", fps=24, frames=9200, shots=[shot], transitions=[])
    for name, value in zip(M.INPUT_FILES, [edl, dict(sync=[]), {}, {}, {}]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded(value))
    assembler = tmp_path / "edit/assemble.py"
    assembler.write_text("# Synthetic dependency fixture; no renderer is executed.\n")
    dependencies = [dict(root="repo", path="edit/assemble.py", sha256=M.sha256(assembler))]
    frames = []
    for f in (8640, 8641):
        source = f - 3200
        for stem in (take["stem"], take["matte"]):
            path = tmp_path / "renders" / stem / f"f_{source:05d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1920, 804), (32, 64, 96)).save(path)
            row = dict(root="repo", path=path.relative_to(tmp_path).as_posix(), sha256=M.sha256(path))
            if stem == take["matte"]:
                dependencies.append(row)
            else:
                frames.append(dict(row, frame=f, source_frame=source))
    proof = dict(method="Synthetic dependency fixture only", claim="observed_state",
                 frame_size=[1920, 804], frames=frames,
                 coverage=dict(first=8640, last=8641, complete=True, missing=[]),
                 dependencies=dependencies, edl_dependencies=dict(shots=[deepcopy(shot)], transitions=[]))
    decision = dict(source="Synthetic fixture only", reason="Test raw scope preservation",
                    measurement_scope=M.NATIVE_SCOPE, measurement=proof)
    base = dict(schema=M.SCHEMA, cut="D", fps=24, frames=9200, input_sha256=M.raw_input_hashes(tmp_path),
                suppressions={"D.new.page.to_blank":decision}, hooks={}, events={}, series={})
    return dict(root=tmp_path, base=base, before=encoded(edl), current=edl, decision=decision, proof=proof)


def run_d32(case):
    return R.repin(case["base"], root=case["root"], base_edl_text=case["before"],
                   request_ids={"D.new.page.to_blank"}, reuse_ids=set())


def test_d32_raw_matte_allowance_preserves_exact_native_scope(d32_case):
    result = run_d32(d32_case)
    assert result["suppressions"] == d32_case["base"]["suppressions"]
    assert result["suppressions"]["D.new.page.to_blank"]["measurement_scope"] == M.NATIVE_SCOPE
    assert result["repin"]["adopted_raw_evidence"] == [
        dict(shot="D32", stem="cand_pen_soft_spine_metal", offset_f=-3200)]


@pytest.mark.parametrize("mutation", ["missing_matte", "stale_matte", "wrong_matte_clock", "duplicate_matte",
                                     "missing_assembler", "stale_assembler", "missing_shot", "stale_shot",
                                     "missing_edl_dependencies"])
def test_d32_raw_matte_allowance_requires_live_exact_dependencies(d32_case, mutation):
    proof = d32_case["proof"]
    dependencies = proof["dependencies"]
    if mutation == "missing_matte":
        dependencies.pop()
    elif mutation == "stale_matte":
        path = d32_case["root"] / dependencies[-1]["path"]
        Image.new("RGB", (1920, 804), (96, 64, 32)).save(path)
    elif mutation == "wrong_matte_clock":
        dependencies[-1]["path"] = dependencies[-2]["path"]
    elif mutation == "duplicate_matte":
        dependencies.append(deepcopy(dependencies[-1]))
    elif mutation == "missing_assembler":
        dependencies.pop(0)
    elif mutation == "stale_assembler":
        (d32_case["root"] / "edit/assemble.py").write_text("# Changed synthetic fixture.\n")
    elif mutation == "missing_shot":
        proof["edl_dependencies"]["shots"] = []
    elif mutation == "stale_shot":
        proof["edl_dependencies"]["shots"][0]["takes"][0]["off"] += 1
    else:
        proof.pop("edl_dependencies")
    with pytest.raises(ValueError):
        run_d32(d32_case)


@pytest.mark.parametrize("field", ["crop", "grade", "screen_transform", "under", "video", "add", "linear_mix"])
def test_d32_allowance_does_not_admit_other_image_treatments(d32_case, field):
    edl = d32_case["current"]
    edl["shots"][0]["takes"][0][field] = "synthetic_changed_treatment"
    d32_case["proof"]["edl_dependencies"]["shots"] = deepcopy(edl["shots"])
    (d32_case["root"] / M.INPUT_FILES[0]).write_text(encoded(edl))
    with pytest.raises(ValueError, match="changed image treatment"):
        run_d32(d32_case)


@pytest.mark.parametrize("mutation", ["different_shot", "different_matte", "composite_scope"])
def test_d32_allowance_cannot_expand_to_another_source_or_scope(d32_case, mutation):
    edl = d32_case["current"]
    if mutation == "composite_scope":
        d32_case["decision"]["measurement_scope"] = M.COMPOSITE_SCOPE
    elif mutation == "different_shot":
        edl["shots"][0]["code"] = "D33"
    else:
        edl["shots"][0]["takes"][0]["matte"] = "different_synthetic_matte"
    d32_case["proof"]["edl_dependencies"]["shots"] = deepcopy(edl["shots"])
    with pytest.raises(ValueError, match="restricted to native D32"):
        R.adopted_evidence(d32_case["base"], edl, root=d32_case["root"])


def transition_case():
    """Synthetic six-join approval matching the owner's structural schema."""
    before=dict(cut='D',fps=24,frames=9200,shots=[],transitions=[
        dict(f0=6636,f1=6644,cut=6640,kind='dissolve',note='Synthetic prior')])
    after=deepcopy(before)
    after['transitions']=[dict(f0=a,f1=b,cut=c,kind='dissolve',note='Synthetic adopted')
        for c,a,b in [(4080,4068,4092),(4240,4234,4246),(4560,4554,4566),(5840,5828,5852)]]
    after['transitions'].append(dict(f0=6080,f1=6120,cut=6080,kind='ring_burn',
        center=[1013.,187.],t_open=6082.,speed=10.,seed=12))
    approval=dict(schema=R.TRANSITION_REVIEW_SCHEMA,changes=R._transition_changes(before,after))
    return before,after,approval


def test_owner_six_join_approval_pins_every_operational_field():
    before,after,approval=transition_case()
    result=R._editorial_compatibility(before,after,approval)
    assert result['reviewed_transitions']==approval
    assert result['reviewed_transitions'] is not approval
    after['transitions'][0]['note']='Different non-operational note'
    assert R._editorial_compatibility(before,after,approval)


@pytest.mark.parametrize('mutation',['missing','stale','partial','extra','wrong_schema','missing_join',
    'unrelated','duplicate','wrong_center','wrong_clock','stale_replay','missing_cut'])
def test_transition_approval_rejects_incomplete_or_unrelated_edit(mutation):
    before,after,approval=transition_case()
    if mutation=='missing': approval=None
    elif mutation=='stale': approval['changes'][0]['after'][0]['f0']-=1
    elif mutation=='partial': approval['changes'].pop()
    elif mutation=='extra': approval['changes'].append(dict(cut=8320,before=[],after=[]))
    elif mutation=='wrong_schema': approval['schema']='wrong'
    elif mutation=='missing_join': after['transitions'].pop(0)
    elif mutation=='unrelated': after['transitions'].append(dict(f0=8310,f1=8330,cut=8320,kind='dissolve'))
    elif mutation=='duplicate': approval['changes'].append(deepcopy(approval['changes'][0]))
    elif mutation=='wrong_center': after['transitions'][-1]['center'][0]+=1
    elif mutation=='wrong_clock': after['frames']+=1
    elif mutation=='stale_replay': before=deepcopy(after)
    elif mutation=='missing_cut': after['transitions'].append(dict(f0=8000,f1=8020,kind='dissolve'))
    with pytest.raises(ValueError): R._editorial_compatibility(before,after,approval)


def test_new_burn_request_requires_explicit_transition_pass(case):
    point=deepcopy(case['base']['hooks']['crown_beacons'])
    point.update(landmark='synthetic_onset',editorial_reference='crown_beacons')
    case['base']['events']={'D.new.unfinished.burn':point}
    with pytest.raises(ValueError,match='unknown sound event'):run(case)
    result=run(case,transition_pass=True)
    assert result['events']['D.new.unfinished.burn']==point
    with pytest.raises(ValueError,match='explicitly boolean'):run(case,transition_pass=1)


def test_cli_forwards_transition_pass_and_owner_review(case,tmp_path,monkeypatch):
    import sound_d_binding as B
    result=dict(repin=dict(replacements=[],adopted_raw_evidence=[],scope='Synthetic CLI contract test'),
                input_sha256={})
    calls=[];binding_calls=[]
    monkeypatch.setattr(R,'repin',lambda *a,**kw:calls.append(kw) or result)
    monkeypatch.setattr(B,'build',lambda **kw:binding_calls.append(kw) or {'phase':6})
    base=tmp_path/'base.json';base.write_text('{}\n')
    review=tmp_path/'review.json';review.write_text('{}\n')
    out=tmp_path/'out.json'
    R.main(['--base',str(base),'--race-revision','round5','--picture-revision','locked',
            '--transition-pass','--transition-review',str(review),'--out',str(out)])
    assert calls[0]['transition_pass'] is True and calls[0]['transition_review']==review
    assert binding_calls[0]['transition_pass'] is True
    assert json.loads(out.with_suffix('.receipt.json').read_text())['transition_pass'] is True
