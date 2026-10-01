"""Small synthetic controls; no production images or renderer imports."""
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

SPEC = importlib.util.spec_from_file_location("measure_sound_d_actions", Path(__file__).parents[1]/"src/measure_sound_d_actions.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_coverage_cannot_claim_completeness_from_endpoints():
    complete = M.coverage({40: None, 41: None, 42: None}, 40, 42)
    assert complete["complete"] and complete["required_count"] == 3
    partial = M.coverage({40: None, 42: None}, 40, 42)
    assert not partial["complete"] and partial["missing"] == [41]
    assert partial["present_count"] == 2
    assert M.runs([42, 40, 40, 44]) == [[40, 40], [42, 42], [44, 44]]


def test_inventory_is_read_only_and_rejects_ambiguous_formats(tmp_path):
    frames = tmp_path/"renders/embers_D_gap"
    frames.mkdir(parents=True)
    path = frames/"f_03440.jpg"
    path.write_bytes(b"synthetic filename fixture")
    before = M.sha256(path)
    row = M.inventory(tmp_path)["shots"]["gap"]
    assert row["coverage"]["present_count"] == 1
    assert not row["coverage"]["complete"]
    assert M.sha256(path) == before
    (frames/"f_03440.png").write_bytes(b"ambiguous duplicate")
    with pytest.raises(ValueError, match="ambiguous"):
        M.frame_files(tmp_path, "gap")


def test_missing_trace_refuses_to_decode_or_bridge(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "frame_files", lambda root, shot: {3440: Path("first"), 3442: Path("last")})
    monkeypatch.setattr(M, "read_frame", lambda *args: pytest.fail("partial trace decoded"))
    result = M.scan_roi(tmp_path, "gap", 3440, 3442, (0, 0, 2, 2))
    assert result["frame"] is None and result["status"] == "unmeasured"
    assert result["coverage"]["missing"] == [3441]


def test_trace_detects_changed_frame_without_making_event_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "frame_files", lambda root, shot: {f: Path(str(f)) for f in (40, 41, 42)})
    def read(path, frame, root):
        pixels = np.full((2, 2, 3), 255 if frame == 41 else 0, np.uint8)
        return Image.fromarray(pixels), {"frame": frame, "sha256": str(frame)}
    monkeypatch.setattr(M, "read_frame", read)
    result = M.scan_roi(tmp_path, "gap", 40, 42, (0, 0, 2, 2))
    assert result["claim"] == "pixel_trace_only" and "frame" not in result
    rows = result["trace"]
    assert rows[0]["mean_abs_change"] is None
    assert rows[1]["positive_change"] == 1
    assert rows[2]["positive_change"] == 0
    assert rows[2]["mean_abs_change"] == 1
    assert len(result["frames"]) == result["coverage"]["required_count"] == 3


def test_wrong_native_size_is_rejected(tmp_path):
    path = tmp_path/"small.jpg"
    Image.new("RGB", (8, 4)).save(path)
    with pytest.raises(ValueError, match="non-native"):
        M.read_frame(path, 3440, tmp_path)


def test_roi_metrics_black_control_and_color_are_independent():
    dark = np.zeros((3, 3, 3), np.uint8)
    red = dark.copy()
    red[:, :, 0] = 255
    assert M.roi_values(dark, dark)["changed_fraction"] == 0
    assert M.roi_values(red, dark)["changed_fraction"] == 1
    assert M.roi_values(red)["gold_mean"] == 1
    assert M.roi_values(np.full_like(red, 255))["gold_mean"] == 0


def test_response_candidates_fail_missing_frame_and_respond_to_removed_flash():
    trace = dict(coverage=M.coverage({40: None, 41: None, 42: None}, 40, 42),
                 trace=[dict(frame=40, positive_change=None), dict(frame=41, positive_change=.01),
                        dict(frame=42, positive_change=.08)])
    assert M.response_frames(trace, .02) == [42]
    trace["trace"][2]["positive_change"] = .01
    assert M.response_frames(trace, .02) == []
    trace["trace"].pop(1)
    with pytest.raises(ValueError, match="contiguous"):
        M.response_frames(trace, .02)


def test_take_comparison_refuses_missing_old_or_new_frames_before_decode(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "frame_files", lambda *args: {40: Path("selected")})
    monkeypatch.setattr(M, "stem_files", lambda *args: {40: Path("old"), 41: Path("old")})
    monkeypatch.setattr(M, "read_frame", lambda *args: pytest.fail("partial pair decoded"))
    with pytest.raises(ValueError, match="both takes"):
        M.compare_take(tmp_path, "gap", 40, 41, (0, 0, 2, 2), "embers_D_gap_old")
    monkeypatch.setattr(M, "frame_files", lambda *args: {40: Path("selected"), 41: Path("selected")})
    monkeypatch.setattr(M, "stem_files", lambda *args: {40: Path("old")})
    with pytest.raises(ValueError, match="both takes"):
        M.compare_take(tmp_path, "gap", 40, 41, (0, 0, 2, 2), "embers_D_gap_old")


def test_take_comparison_separates_old_evidence_and_does_not_invent_onset(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "frame_files", lambda *args: {f: Path("selected") for f in (40, 41, 42)})
    monkeypatch.setattr(M, "stem_files", lambda *args: {f: Path("old") for f in (40, 41, 42)})
    def read(path, frame, root):
        changed = path.name == "selected" and frame == 41
        value = 255 if changed else 0
        return Image.fromarray(np.full((2, 2, 3), value, np.uint8)), dict(
            frame=frame, sha256="white" if changed else "black", source_stem=path.name)
    monkeypatch.setattr(M, "read_frame", read)
    result = M.compare_take(tmp_path, "gap", 40, 42, (0, 0, 2, 2), "embers_D_gap_old")
    assert result["changed_runs"] == [[41, 41]]
    assert result["unchanged_runs"] == [[40, 40], [42, 42]]
    assert result["claim"] == "pixel_trace_only" and "frame" not in result
    assert all(r["source_stem"] == "selected" for r in result["frames"])
    assert all(r["source_stem"] == "old" for r in result["comparison_frames"])
    assert result["trace"][1]["comparison_max_abs_code_change"] == 255
    assert result["trace"][2]["comparison_max_abs_code_change"] == 0


@pytest.mark.parametrize("stem", ["../embers_D_gap", "embers_D_gap/old", "/embers_D_gap"])
def test_comparison_stem_cannot_escape_renders(tmp_path, stem):
    with pytest.raises(ValueError, match="stem"):
        M.stem_files(tmp_path, stem)
