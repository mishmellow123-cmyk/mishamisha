"""A's actual text-table path and the arrival review's two measurement discriminators."""
from pathlib import Path
import sys

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EDIT), str(EDIT / 'tools')]
import titles
import c5_caption_backdrop as BD
import lantern_review as LR


def test_a_t14_placement_reaches_the_renderer():
    table = titles.text_table('A')
    row = next(r for r in table if r['id'] == 'T14')
    placed = titles.TextV3('A', row, 0.5)
    legacy = titles.TextV3('A', {k: v for k, v in row.items() if k not in ('x', 'y')}, 0.5)
    assert (row['f_in'], row['f_out'], row['line']) == (6080, 6200, 'No one got there first.')
    def assert_lantern_placement(line):
        assert abs(line.x0 + line.w / 2 - 710) <= 0.5
        assert abs(line.y0 + line.h / 2 - 285) <= 0.5

    assert_lantern_placement(placed)
    assert placed.x0 - legacy.x0 > 200
    # The old table path (silently dropping coordinates) must fail the same placement criterion.
    with pytest.raises(AssertionError):
        assert_lantern_placement(legacy)


def test_light_metric_distinguishes_lantern_from_competing_fire_and_cold_sky():
    frame = np.zeros((402, 960, 3), np.uint8)
    frame[320, 872] = [255, 230, 170]
    frame[226, 665] = [230, 180, 80]
    frame[20, 20] = [255, 255, 255]
    good = LR.light_metrics(frame, 5840)
    assert good['warm_peak']['xy_master'] == [1744, 640]
    assert good['global_peak']['xy_master'] == [40, 40]
    assert good['lantern']['bright_warm_area_half_px'] == 1
    frame[320, 872] = 0  # control: absence of the lantern must hand the warm peak to the other fire
    bad = LR.light_metrics(frame, 5840)
    assert bad['warm_peak']['xy_master'] == [1330, 452]
    assert bad['lantern']['bright_warm_area_half_px'] == 0


def test_caption_worst_slice_finds_a_bad_word_hidden_by_the_mean():
    row = next(r for r in titles.text_table('A') if r['id'] == 'T14')
    line = titles.TextV3('A', row, 1.0)
    core, ring = BD.glyph_masks(line)
    slices = BD.band_slices(line)
    frame = np.full((804, 1920, 3), 0.05, np.float32)
    frame[core] = 0.9
    good = LR.caption_contrast(frame, core, ring, slices)
    assert len(good['slices']) == 8 and good['worst'] > 3
    y0, y1, x0, x1 = slices[-1]
    frame[max(0, y0 - 8):y1 + 8, x0:x1] = 0.9
    bad = LR.caption_contrast(frame, core, ring, slices)
    assert bad['overall'] > 3
    assert bad['worst'] < 3 and bad['limiting_slice'] == 7
