"""The audit must distinguish light identity and never display a stale sample."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import joins_a_review as J


def test_black_has_no_light_and_cold_core_is_separate_from_warm_peak():
    rgb = np.zeros((5, 8, 3), np.uint8)
    black = J.metrics(rgb)
    assert black['warm_peak'] is None and black['global_peak'] is None
    # Explicit synthetic control: the warm mask must not claim the brighter white core.
    rgb[1, 6] = (255, 255, 255)
    rgb[3, 2] = (240, 180, 90)
    measured = J.metrics(rgb)
    assert measured['global_peak']['xy'] == [6, 1]
    assert measured['warm_peak']['xy'] == [2, 3]
    assert measured['warm_area'] == 1
    assert J.distance(black, measured, 'warm_peak') is None


def test_missing_sample_cannot_reuse_a_previous_runs_png(tmp_path, monkeypatch):
    # An existing path alone must never turn missing coverage into a picture.
    stale = tmp_path / 'joinsA_probe_candidate_00080.png'
    stale.write_bytes(b'stale evidence must not be opened')
    records = {'candidate': {'frames': {'80': {'missing': [{'frame': 80}]}}}}
    saved = []
    monkeypatch.setattr(J.Image.Image, 'save', lambda self, *a, **k: saved.append(self.copy()))
    J.sheet(tmp_path, 'probe', 80, [80], ['candidate'], records)
    assert saved[0].getpixel((240, 136)) == (20, 20, 26)
    # Negative control: claiming the stale file is current really does try to open it.
    records['candidate']['frames']['80']['missing'] = []
    with pytest.raises(OSError):
        J.sheet(tmp_path, 'probe', 80, [80], ['candidate'], records)


def test_story_anchor_does_not_follow_a_brighter_remote_fire():
    rgb = np.zeros((402, 960, 3), np.uint8)
    rgb[200, 480] = (240, 240, 240)
    rgb[180, 870] = (255, 255, 255)
    assert J.anchor_metrics(rgb, 4880)['bright_centroid'] == [480.0, 200.0]
    rgb[200, 480] = 0
    assert J.anchor_metrics(rgb, 4880)['bright_centroid'] is None
