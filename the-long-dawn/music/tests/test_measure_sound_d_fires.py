"""Measurement guards use synthetic pixels; no source renders or audio loaded."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import measure_sound_d_fires as M


def test_warm_pixels_separate_light_from_blue_white_and_dark():
    rgb = np.array([[[200,80,20], [255,255,255], [10,10,10], [20,40,200]]], dtype=np.uint8)
    got = M.warm_stats(rgb, [0,0,4,1])
    assert got["pixels"] == 1
    assert got["centroid_xy"] == [0., 0.]
    assert M.warm_stats(rgb, [1,0,4,1])["pixels"] == 0


@pytest.mark.parametrize("roi", [[-1,0,2,2], [0,0,3,2], [1,1,1,2]])
def test_roi_does_not_silently_wrap_or_truncate(roi):
    with pytest.raises(ValueError, match="ROI"):
        M.warm_stats(np.zeros((2,2,3), dtype=np.uint8), roi)


def test_float_image_rejected():
    with pytest.raises(ValueError, match="uint8"):
        M.warm_stats(np.ones((2,2,3)), [0,0,2,2])


def test_solar_predicate_rejects_rose_sky_and_orange_fire():
    rgb=np.array([[[255,249,246],[230,160,140],[255,180,80],[185,180,175]]],dtype=np.uint8)
    assert M.solar_stats(rgb,[0,0,4,1])["pixels"] == 1
    assert M.solar_stats(rgb,[1,0,4,1])["pixels"] == 0


def test_gap_free_dark_prefix_reports_first_onset():
    out = M.first_onset({10:0, 11:0, 12:5, 13:8}, 10, 13)
    assert out["frame"] == 12 and out["claim"] == "onset"
    assert out["coverage"] == dict(complete=True, first=10, last=12, missing=[])


@pytest.mark.parametrize("missing", [10, 11])
def test_missing_early_frame_prevents_first_onset_even_if_pair_exists(missing):
    rows = {10:0, 11:0, 12:0, 13:5}
    del rows[missing]
    out = M.first_onset(rows, 10, 13)
    assert out["frame"] is None and out["first_observed_lit"] == 13
    assert out["coverage"]["missing"] == [missing]


def test_already_lit_start_is_only_observed_state():
    out = M.first_onset({10:5,11:7},10,11)
    assert out["frame"] is None and out["claim"] == "observed_state"


def test_no_light_does_not_invent_onset():
    assert M.first_onset({10:0,11:0},10,11)["frame"] is None


def test_inventory_has_missing_indices_and_rejects_duplicate_formats(tmp_path):
    d=tmp_path/"renders/embers_D_crowns"
    d.mkdir(parents=True)
    (d/"f_04320.jpg").touch()
    inv=M.inventory(tmp_path)["embers_D_crowns"]
    assert inv["present"] == [4320] and 4319 in inv["missing"]
    assert not inv["complete"]
    (d/"f_04320.png").touch()
    with pytest.raises(ValueError, match="duplicate"):
        M.inventory(tmp_path)


def test_scan_does_not_promote_wrong_resolution_plate(tmp_path):
    d=tmp_path/"renders/embers_D_crowns"
    d.mkdir(parents=True)
    Image.new("RGB", (20,20), "orange").save(d/"f_04240.jpg")
    result=M.scan(tmp_path,"embers_D_crowns",4240,4241,{"small":[0,0,10,10]})
    assert not result["frames"]
    assert result["errors"][0]["frame"] == 4240
    assert "native 1920x804" in result["errors"][0]["error"]
    assert result["detections"]["small"]["110"]["frame"] is None


def agreed_result():
    detection=M.first_onset({10:0,11:4},10,11)
    return dict(detections={s:{str(t):deepcopy(detection) for t in M.THRESHOLDS} for s in ("left","right")},
                frames=[dict(frame=f,path=f"renders/embers_D_crowns/f_{f:05d}.jpg",sha256="a"*64,
                             source_stem="embers_D_crowns",source_frame=f) for f in (10,11)],
                rois=deepcopy(M.CROWN_ROIS),thresholds=list(M.THRESHOLDS),chroma_rule="test")


def test_two_baskets_and_thresholds_must_agree():
    result=agreed_result()
    assert M.crown_hook(result)["frame"] == 11
    result["detections"]["right"]["130"]["frame"] = 12
    assert M.crown_hook(result) is None
    result=agreed_result()
    result["detections"]["right"]["90"]["frame"] = None
    assert M.crown_hook(result) is None


def test_hook_copies_mutable_evidence():
    result=agreed_result()
    hook=M.crown_hook(result)
    hook["measurement"]["roi"]["left"][0]=0
    hook["measurement"]["coverage"]["missing"].append(99)
    assert result["rois"]["left"][0] == 713
    assert result["detections"]["left"]["90"]["coverage"]["missing"] == []


def test_output_refuses_repo_and_shared_render_symlink(tmp_path):
    root=tmp_path/"repo"; root.mkdir()
    shared=tmp_path/"shared"; shared.mkdir()
    (root/"renders").symlink_to(shared, target_is_directory=True)
    for path in (root/"music", shared/"bad", root/"renders/bad"):
        with pytest.raises(ValueError, match="external"):
            M.external_output(path, root)
    assert M.external_output(tmp_path/"owned-output", root).is_dir()


def test_import_and_inventory_never_write_baseline(tmp_path):
    root=tmp_path/"repo";root.mkdir()
    before=list(root.rglob("*"))
    M.inventory(root)
    assert list(root.rglob("*")) == before


def ridge_result():
    result=dict(scan_range_inclusive=[4560,5039],rois=deepcopy(M.RIDGE_ROIS),
                thresholds=list(M.THRESHOLDS),chroma_rule="test",frames=[],detections={})
    # Synthetic contact frames test only the full-search guard and mapping.
    for f in range(4560,5040):
        stats={n:{str(t):dict(pixels=4 if f>=4700+i*10 else 0,centroid_xy=[100.+i,550.])
                  for t in M.THRESHOLDS} for i,n in enumerate(M.RIDGE_ROIS)}
        result["frames"].append(dict(frame=f,path=f"renders/test/f_{f:05d}.jpg",sha256="a"*64,
                                     source_stem="test",source_frame=f,stats=stats))
    for name in M.RIDGE_ROIS:
        result["detections"][name]={str(t):M.first_onset({r["frame"]:r["stats"][name][str(t)]["pixels"]
                                   for r in result["frames"]},4560,5039) for t in M.THRESHOLDS}
    return result


def test_ridge_replacement_has_only_detected_catches_and_full_search_evidence():
    result=ridge_result()
    series=M.ridge_series(result)
    assert [e["frame"] for e in series["events"]] == [4700,4710,4720,4730,4740]
    assert all("pair" not in e["id"] for e in series["events"])
    assert len(series["measurement"]["frames"]) == 480
    assert series["events"][0]["pan"] == 200/1919-1
    # Missing even a post-catch frame cannot certify the whole search.
    result["frames"].pop()
    assert M.ridge_series(result) is None


def test_ridge_threshold_disagreement_blocks_series():
    result=ridge_result()
    result["detections"]["ridge1"]["130"]["frame"] = 4701
    assert M.ridge_series(result) is None


def test_solar_onset_requires_threshold_agreement_and_persistent_next_frame():
    frames=[]
    for f in (7360,7361,7362):
        frames.append(dict(frame=f,path=f"renders/test/f_{f:05d}.jpg",sha256="b"*64,
                           source_stem="test",source_frame=f,
                           stats={"sun":{str(t):dict(pixels=int(f>=7361)) for t in M.SUN_THRESHOLDS}}))
    result=dict(frames=frames,scan_range_inclusive=[7360,7839],rois=deepcopy(M.SUN_ROI),
                detections={"sun":{str(t):M.first_onset({7360:0,7361:1,7362:1},7360,7362,min_pixels=1)
                                   for t in M.SUN_THRESHOLDS}})
    hook=M.solar_hook(result)
    assert hook["frame"] == 7361 and hook["measurement"]["coverage"]["last"] == 7362
    result["frames"][-1]["stats"]["sun"]["230"]["pixels"] = 0
    assert M.solar_hook(result) is None
    result["frames"].pop()
    assert M.solar_hook(result) is None
    result["detections"]["sun"]["230"]["frame"] = None
    assert M.solar_hook(result) is None
