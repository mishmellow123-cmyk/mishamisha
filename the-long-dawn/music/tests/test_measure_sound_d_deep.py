"""Small synthetic checks; these tests neither load source media nor render."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import measure_sound_d_deep as M


def test_subtraction_does_not_wrap_uint8_or_count_unchanged_warm_race():
    base = np.array([[[240, 90, 20], [255, 0, 0]]], dtype=np.uint8)
    assert M.changed_pixels(base, base)["new_warm_pixels"]["1"] == 0
    rgb = np.array([[[250, 100, 30], [0, 0, 0]]], dtype=np.uint8)
    result = M.changed_pixels(rgb, base)
    assert result["max_channel_change"] == 255
    assert result["changed_pixels"]["16"] == 1
    assert result["new_warm_pixels"]["8"] == 1
    assert result["new_warm_pixels"]["16"] == 0


def test_page_white_and_blue_are_not_the_saturated_warm_proxy():
    rgb = np.array([[[200,150,90], [255,255,255], [0,40,255]]], dtype=np.uint8)
    assert M.changed_pixels(rgb, np.zeros_like(rgb))["new_warm_pixels"]["1"] == 0


def test_mismatched_and_float_inputs_fail():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        M.changed_pixels(rgb, rgb.astype(float))
    with pytest.raises(ValueError):
        M.changed_pixels(rgb, rgb[:1])


def test_page_proxy_excludes_original_page_colored_pixels():
    clean=np.full((1,3,3),180,dtype=np.uint8)
    baseline=clean.copy()
    baseline[0,1]=0
    result=M.page_resemblance(clean,baseline,clean)
    assert result["8"] == dict(pixels=1,centre_column_pixels=1)
    assert M.page_resemblance(baseline,baseline,clean)["8"]["pixels"] == 0


def test_core_mask_rejects_warm_page_fire_and_new_nearby_white_spark():
    baseline=np.array([[[240,240,245],[0,0,0],[230,230,240]]],dtype=np.uint8)
    rgb=np.array([[[230,160,70],[255,255,255],[255,190,80]]],dtype=np.uint8)
    result=M.core_stats(rgb,baseline,(0,0,3,1))
    assert result["220"] == dict(baseline_pixels=2,white_pixels=0)
    assert M.core_stats(baseline,baseline,(0,0,3,1))["220"]["white_pixels"]==2


def test_core_clearance_cannot_certify_missing_prefix_or_changed_baseline():
    rows=[dict(frame=f,core={str(t):dict(baseline_pixels=10,white_pixels=v) for t in M.CORE_FLOORS})
          for f,v in ((10,10),(11,6),(12,0),(13,0))]
    result=M.core_clearance(rows,10,13)
    assert all(r["last_visible_frame"]==11 and r["first_trailing_absent_frame"]==12 for r in result.values())
    with pytest.raises(ValueError,match="coverage"):
        M.core_clearance(rows[1:],10,13)
    rows[1]["core"]["180"]["baseline_pixels"]=11
    with pytest.raises(ValueError,match="mask changed"):
        M.core_clearance(rows,10,13)


def test_core_clearance_does_not_invent_fire_or_clearance_outside_scan():
    for value in (0,10):
        rows=[dict(frame=f,core={str(t):dict(baseline_pixels=value,white_pixels=value) for t in M.CORE_FLOORS})
              for f in (10,11)]
        assert all(r["first_trailing_absent_frame"] is None for r in M.core_clearance(rows,10,11).values())


def test_counterfactual_rejects_white_burn_light_even_on_exact_core_position():
    baseline=np.full((1,1,3),240,dtype=np.uint8)
    white_burn=np.full((1,1,3),255,dtype=np.uint8)
    got=M.core_stats(white_burn,baseline,(0,0,1,1),counterfactual=white_burn)
    assert got["220"]["white_pixels"]==1
    assert all(v==0 for v in got["220"]["attributed_white_pixels"].values())
    surviving=M.core_stats(baseline,baseline,(0,0,1,1),counterfactual=np.zeros_like(baseline))
    assert all(v==1 for v in surviving["220"]["attributed_white_pixels"].values())


def test_page_rate_keeps_ties_without_inventing_a_crest():
    rows=[dict(frame=f,page_resemblance={str(t):dict(pixels=v,centre_column_pixels=v)
                                        for t in (8,12,16)}) for f,v in ((10,0),(11,7),(12,14))]
    got=M.page_candidates(rows)["8"]
    assert got["maximum_rate_frames"] == [11,12]
    assert got["maximum_new_page_proxy_pixels_per_frame"] == 7
    assert got["first_centre_column_page_proxy"] == 11


def sample(frame, value):
    return dict(frame=frame, changed_pixels={"1": value})


def test_complete_prefix_finds_first_change():
    out = M.first_change([sample(10,0), sample(11,7), sample(12,20)], "changed_pixels", 1, 10, 12)
    assert out["frame"] == 11 and out["claim"] == "onset"


@pytest.mark.parametrize("rows", [[sample(11,7), sample(12,20)], [sample(10,0), sample(12,20)]])
def test_sparse_search_does_not_claim_onset(rows):
    out = M.first_change(rows, "changed_pixels", 1, 10, 12)
    assert out["frame"] is None and not out["coverage"]["complete"]


@pytest.mark.parametrize("values", [[3,4,5], [0,0,0]])
def test_already_positive_and_absence_are_not_onsets(values):
    assert M.first_change([sample(i+10,v) for i,v in enumerate(values)], "changed_pixels", 1, 10, 12)["frame"] is None


def test_duplicate_frame_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        M.first_change([sample(10,0),sample(10,1)], "changed_pixels", 1, 10, 11)


def test_clearance_ignores_early_zero_and_fails_closed_on_missing_frame():
    rows=[dict(frame=i+10,new_warm_pixels={"1":v}) for i,v in enumerate([0,8,0,7,0,0])]
    assert M.trailing_clearance(rows,1,10,15)["frame"] == 14
    assert M.trailing_clearance(rows[:-1],1,10,15)["frame"] is None
    assert M.trailing_clearance(rows[:4],1,10,13)["frame"] is None
    for r in rows:
        r["new_warm_pixels"]["1"]=0
    assert M.trailing_clearance(rows,1,10,15)["frame"] is None


def test_native_header_guard_rejects_silent_upscale(tmp_path):
    path = tmp_path/"small.png"
    Image.new("RGB", (480,201)).save(path)
    with pytest.raises(ValueError, match="native"):
        M.native_rgb(path)


def test_incomplete_runtime_receipt_fails_before_loading_media(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        M.validate_refresh(dict(status="pending"), tmp_path)
    with pytest.raises(ValueError, match="denominator"):
        M.validate_refresh(dict(status="complete", native_runtime_ran=True, runtime=dict(frames=[])), tmp_path)


def test_stale_held_race_hash_fails_before_image_decode(tmp_path):
    race=tmp_path/"renders/embers_D_race/f_02719.jpg"
    race.parent.mkdir(parents=True)
    race.write_bytes(b"changed")
    receipt=dict(status="complete",native_runtime_ran=True,farm_sha256="0"*64,
                 runtime=dict(frames=[dict(frame=f) for f in range(M.FIRST,M.LAST+1)]))
    with pytest.raises(ValueError, match="held race"):
        M.validate_refresh(receipt,tmp_path,root=tmp_path)


def test_changed_edl_and_new_transition_fail():
    edl=dict(shots=[dict(code="D10",f0=2400,f1=2720),dict(code="D11a",f0=2720,f1=2728),
                    dict(code="D11b",f0=2728,f1=2758)],transitions=[])
    assert len(M.selected_edl(edl)["shots"]) == 3
    altered=deepcopy(edl)
    altered["transitions"]=[dict(f0=2719,f1=2724)]
    with pytest.raises(ValueError, match="transition"):
        M.selected_edl(altered)
    altered=deepcopy(edl)
    altered["shots"][1]["code"]="other"
    with pytest.raises(ValueError, match="layout"):
        M.selected_edl(altered)


def test_observed_pixel_onset_cannot_replace_sound_crest_and_output_is_isolated():
    result=dict(retained_frames=[dict(frame=f,path=f"composites/f_{f:05d}.png") for f in M.KEEP],
                input_sha256={},onsets={"changed_pixels":{"1":{"frame":2721}}},
                dependencies=[dict(root="repo",path="edit/assemble.py",sha256="a"*64)],
                edl_dependencies={"shots":[],"transitions":[]})
    out=M.overlay(result)
    assert set(out["events"]) == {M.EVENT}
    assert not out["hooks"] and not out["series"] and not out["suppressions"]
    event=out["events"][M.EVENT]
    assert event["frame"] is None and event["status"] == "unmeasured"
    assert event["measurement_scope"] == "native_pre_finish_composite"
    assert {r["frame"] for r in event["measurement"]["frames"]} == {2720,2721,2722}
    event["measurement"]["dependencies"][0]["sha256"]="changed"
    assert result["dependencies"][0]["sha256"] == "a"*64


def core_result():
    rows=[dict(frame=f,core={str(t):dict(baseline_pixels=20,white_pixels=value,
            attributed_white_pixels={str(k):value for k in (1,4,8)}) for t in M.CORE_FLOORS})
          for f,value in ((2740,20),(2741,9),(2742,0),(2743,0))]
    return dict(retained_frames=[dict(frame=f,path=f"composites/f_{f:05d}.png")
                                for f in (2720,2721,2722,2740,2741,2742,2743)],
        input_sha256={},onsets={},dependencies=[],edl_dependencies={"shots":[],"transitions":[]},
        thinking_fire_core=dict(method="synthetic actual and counterfactual controls",roi=list(M.CORE_ROI),
                                clearance=M.core_clearance(rows,2740,2743)))


def test_core_endpoint_keeps_burn_anchor_null_and_copies_evidence():
    result=core_result()
    out=M.overlay(result)
    point=out["bounds"]["D.new.forging.fire"]["end"]
    assert point["frame"]==2742 and point["offset_f"]==0
    assert point["measurement"]["claim"]=="completion"
    assert [r["frame"] for r in point["measurement"]["frames"]]==[2741,2742,2743]
    assert out["events"][M.EVENT]["frame"] is None
    point["measurement"]["frames"][0]["path"]="mutated"
    assert all(r["path"]!="mutated" for r in result["retained_frames"])


def test_core_endpoint_refuses_threshold_disagreement_and_color_only_evidence():
    result=core_result()
    result["thinking_fire_core"]["clearance"]["minRGB140_pixels1"]["first_trailing_absent_frame"]=2743
    assert "bounds" not in M.overlay(result)
    result=core_result()
    result["thinking_fire_core"]["clearance"].pop("minRGB140_pixels1_sourceDelta1")
    assert "bounds" not in M.overlay(result)
    result=core_result()
    result["thinking_fire_core"]["clearance"]={k:v for k,v in result["thinking_fire_core"]["clearance"].items()
                                                if "sourceDelta" not in k}
    assert "bounds" not in M.overlay(result)


def test_core_endpoint_requires_retained_completion_bracket():
    result=core_result()
    result["retained_frames"]=[r for r in result["retained_frames"] if r["frame"]!=2741]
    with pytest.raises(ValueError,match="retained native bracket"):
        M.overlay(result)
