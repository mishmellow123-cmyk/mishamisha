"""Synthetic race-evidence controls; no production images or render calls."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import measure_sound_d_race as M


def trace():
    return dict(coverage=M.A.coverage({10:None, 11:None, 12:None}, 10, 12),
                trace=[dict(frame=10, ring=dict(positive_change=None)),
                       dict(frame=11, ring=dict(positive_change=.07)),
                       dict(frame=12, ring=dict(positive_change=.01))])


def test_candidates_reject_missing_and_reordered_frames():
    good = trace()
    assert M.candidates(good, "ring", .02) == [11]
    broken_rows, broken_coverage = deepcopy(good), deepcopy(good)
    broken_rows["trace"].pop(1)
    broken_coverage["coverage"]["complete"] = False
    broken_coverage["coverage"]["missing"] = [11]
    for broken in (broken_rows, broken_coverage):
        with pytest.raises(ValueError, match="contiguous"):
            M.candidates(broken, "ring", .02)
    good["trace"].reverse()
    with pytest.raises(ValueError, match="contiguous"):
        M.candidates(good, "ring", .02)


def test_removed_flash_and_invalid_threshold_cannot_make_events():
    data = trace()
    data["trace"][1]["ring"]["positive_change"] = .001
    assert M.candidates(data, "ring", .02) == []
    for threshold in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(ValueError, match="threshold"):
            M.candidates(data, "ring", threshold)


def test_isolated_pulse_does_not_confuse_smooth_reveal_with_contact():
    values = [None,.006,.01,.03,.01,.006,.014,.015,.016,.017,.018]
    data = dict(coverage=M.A.coverage(dict.fromkeys(range(len(values))),0,len(values)-1),
                trace=[dict(frame=i,ring=dict(positive_change=v)) for i,v in enumerate(values)])
    assert M.isolated_responses(data) == [3]
    data["trace"][3]["ring"]["positive_change"] = .01
    assert M.isolated_responses(data) == []
    for minimum, ratio in ((float("nan"),1.5),(.01,1)):
        with pytest.raises(ValueError,match="thresholds"):
            M.isolated_responses(data,minimum=minimum,ratio=ratio)


def test_work_requires_both_field_regions_even_when_ring_pulses():
    values = [None,.001,.02,.001,.001]
    data = dict(coverage=M.A.coverage(dict.fromkeys(range(5)),0,4),trace=[
        dict(frame=i,ring=dict(positive_change=v),left_field=dict(mean=.02 if i==2 else 0),
             right_field=dict(mean=.03 if i==2 else 0)) for i,v in enumerate(values)])
    assert M.work_responses(data) == [2]
    data["trace"][2]["right_field"]["mean"] = 0
    assert M.isolated_responses(data) == [2]
    assert M.work_responses(data) == []
    with pytest.raises(ValueError,match="bilateral threshold"):
        M.work_responses(data,float("nan"))


def test_old_new_deltas_do_not_invent_nearest_neighbor_drift():
    assert M.deltas([2400, 2440], [2400, 2420, 2440]) == dict(
        retained=[2400, 2440], removed=[], added=[2420], old_count=2, new_count=3)
    assert M.deltas([2400, 2440], [2401, 2440])["removed"] == [2400]
    for old, new in (([2, 1], [1]), ([1], [1, 1])):
        with pytest.raises(ValueError, match="unique and ordered"):
            M.deltas(old, new)


def test_partial_scan_never_decodes(monkeypatch, tmp_path):
    monkeypatch.setattr(M.A, "frame_files", lambda *args: {10:Path("a"),12:Path("c")})
    monkeypatch.setattr(M.A, "read_frame", lambda *args: pytest.fail("decoded partial coverage"))
    result = M.scan(tmp_path, "race", {"ring":(0,0,2,2)}, 10, 12)
    assert result["frame"] is None and result["coverage"]["missing"] == [11]


def test_separate_regions_are_measured_without_event_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(M.A, "frame_files", lambda *args: {f:Path(str(f)) for f in (10,11,12)})
    def read(path, frame, root):
        pixels = np.zeros((4,4,3), np.uint8)
        pixels[:2,:2] = 255 if frame == 11 else 0
        pixels[2:,2:] = 255 if frame == 12 else 0
        return Image.fromarray(pixels), dict(frame=frame,sha256=str(frame))
    monkeypatch.setattr(M.A, "read_frame", read)
    data = M.scan(tmp_path, "race", {"ring":(0,0,2,2),"field":(2,2,4,4)},10,12)
    assert data["claim"] == "pixel_trace_only" and "events" not in data
    assert M.candidates(data,"ring",.5) == [11]
    assert M.candidates(data,"field",.5) == [12]
    assert data["trace"][2]["ring"]["positive_change"] == 0
    with pytest.raises(ValueError, match="reversed"):
        M.scan(tmp_path,"race",first=12,last=10)
    with pytest.raises(ValueError, match="ROI"):
        M.scan(tmp_path,"race",{"ring":(4,0,2,2)},10,12)


def test_output_guard_follows_shared_render_symlink(tmp_path):
    shared = tmp_path/"shared"
    shared.mkdir()
    root = tmp_path/"repo"
    root.mkdir()
    (root/"renders").symlink_to(shared, target_is_directory=True)
    with pytest.raises(ValueError, match="shared renders"):
        M.output_dir(root/"renders/new-report",root)
    with pytest.raises(ValueError, match="shared renders"):
        M.output_dir(shared/"new-report",root)
    assert list(shared.iterdir()) == []
    output = M.output_dir(tmp_path/"reports",root)
    assert output.is_dir()


def test_white_core_rejects_gold_grains_and_absent_fire():
    rgb = np.zeros((20,24,3),np.uint8)
    rgb[3:9,7:13] = (220,230,225)
    core = M.white_core(rgb,(2,1,22,19))
    assert core["present"] and core["pixels"] == 36
    assert core["centroid"] == [9.5,5.5]
    assert core["bbox"] == [7,3,13,9]
    rgb[3:9,7:13] = (255,170,30)  # bright warm gold is not a white core
    assert not M.white_core(rgb,(2,1,22,19))["present"]
    rgb[3,7] = 255  # one bright grain cannot restore the fire
    assert not M.white_core(rgb,(2,1,22,19))["present"]
    rgb[:] = 0
    assert M.white_core(rgb,(2,1,22,19))["centroid"] is None


def test_white_core_preserves_position_and_never_bridges_disconnected_grains():
    rgb = np.zeros((20,24,3),np.uint8)
    rgb[3:9,7:13] = 255
    before = M.white_core(rgb,(0,0,24,20))
    rgb[:] = 0
    rgb[3:9,10:16] = 255
    after = M.white_core(rgb,(0,0,24,20))
    assert after["centroid"][0]-before["centroid"][0] == 3
    rgb[:] = 0
    rgb[2:5,2:7] = 255
    rgb[10:13,12:17] = 255
    assert not M.white_core(rgb,(0,0,24,20))["present"]
    for threshold in (float('nan'),0,1):
        with pytest.raises(ValueError,match="threshold"):
            M.white_core(rgb,(0,0,24,20),threshold=threshold)
    with pytest.raises(ValueError,match="ROI"):
        M.white_core(rgb,(20,0,30,20))


def test_core_measurement_is_opt_in_and_keeps_other_pixel_traces(monkeypatch,tmp_path):
    monkeypatch.setattr(M.A,"frame_files",lambda *args:{f:Path(str(f)) for f in (10,11,12)})
    def read(path,frame,root):
        rgb=np.zeros((20,24,3),np.uint8)
        rgb[3:9,7:13]=255 if frame!=11 else 0
        return Image.fromarray(rgb),dict(frame=frame,sha256=str(frame))
    monkeypatch.setattr(M.A,"read_frame",read)
    plain=M.scan(tmp_path,"race",{"ring":(0,0,24,20)},10,12)
    measured=M.scan(tmp_path,"race",{"ring":(0,0,24,20)},10,12,core_roi=(0,0,24,20))
    assert "core_predicate" not in plain
    assert all("white_core" not in r for r in plain["trace"])
    assert [r["ring"] for r in plain["trace"]] == [r["ring"] for r in measured["trace"]]
    assert [r["white_core"]["present"] for r in measured["trace"]] == [True,False,True]
    with pytest.raises(ValueError,match="requires a core ROI"):
        M.scan(tmp_path,"race",{"ring":(0,0,24,20)},10,12,core_anchor=(10,5))


def test_core_tracking_cannot_jump_to_larger_ring_or_restore_missing_fire():
    rgb=np.zeros((70,50,3),np.uint8)
    rgb[2:20,10:40]=255  # larger bright Ring far from the inspected fire
    rgb[50:56,22:28]=255
    core=M.white_core(rgb,(0,0,50,70),anchor=(25,53),max_distance=8)
    assert core["present"] and core["pixels"] == 36
    assert core["centroid"] == [24.5,52.5]
    rgb[50:56,22:28]=0
    assert not M.white_core(rgb,(0,0,50,70),anchor=core["centroid"],max_distance=8)["present"]
    for options in ({"anchor":(25,53)}, {"max_distance":8},
                    {"anchor":(25,float('nan')),"max_distance":8},
                    {"anchor":(25,53),"max_distance":0}):
        with pytest.raises(ValueError,match="tracking"):
            M.white_core(rgb,(0,0,50,70),**options)
