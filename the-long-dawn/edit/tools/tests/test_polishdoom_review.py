"""Controls for evidence batching and frame measurement, independent of source assets."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polishdoom_review as P


def test_batches_are_bounded_without_dropping_or_repeating_frames():
    frames = list(range(2400, 3360))
    batches = P.chunks(frames)
    assert max(map(len, batches)) == 12
    assert sum(batches, []) == frames
    with pytest.raises(ValueError):
        P.chunks(frames, 13)


def test_metrics_distinguish_movement_from_mean_level_and_black_from_floor():
    black = np.zeros((2, 2, 3), np.uint8)
    floor = np.full_like(black, 1)
    assert P.measured(black)['nonzero_fraction'] == 0
    assert P.measured(floor)['nonzero_fraction'] == 1
    a, b = black.copy(), black.copy()
    a[0, 0] = 255
    b[1, 1] = 255
    same = P.measured(a, a)
    moved = P.measured(b, a)
    assert same['step_y_mad'] == 0
    assert moved['step_y_mad'] == 127.5
    assert moved['mean_y_delta'] == 0
    assert moved['sha256_rgb'] != same['sha256_rgb']


def test_probe_refuses_thirteenth_frame_before_loading_assembler(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'assembler', lambda ref: pytest.fail('Must reject before loading'))
    with pytest.raises(ValueError, match='1..12'):
        P.probe(tmp_path, 'before', list(range(13)))


def test_exact_5ms_envelopes_keep_true_zero_separate_from_quiet_sound():
    samples = np.zeros((480, 2))
    samples[240:] = .000001
    rows = P.envelope(samples, 48000, 48000)
    assert len(rows) == 2
    assert rows[0]['time_s'] == 1
    assert rows[1]['time_s'] == 1.005
    assert rows[0]['digital_zero'] and rows[0]['dbfs'] is None
    assert not rows[1]['digital_zero']
    assert rows[1]['dbfs'] == pytest.approx(-120)
    with pytest.raises(ValueError, match='exact 5 ms'):
        P.envelope(samples, 44100)


def test_baseline_loader_executes_historical_assembler_with_historical_dependencies(monkeypatch):
    sources = {'afix_comp': b'marker = "old_afix"', 'edl_v3': b'marker = "old_edl"',
               'assemble': b'import afix_comp, edl_v3\nmarker = ("old_assembler", afix_comp.marker, edl_v3.marker)'}
    refs = []
    def historical(name, ref):
        refs.append((name, ref))
        return sources[name]
    monkeypatch.setattr(P, 'code_source', historical)
    # Register mutations for restoration, even if another test imported the real modules earlier.
    for name in sources:
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    module, hashes = P.assembler('pinned_ref')
    assert module.marker == ('old_assembler', 'old_afix', 'old_edl')
    assert refs == [('afix_comp', 'pinned_ref'), ('edl_v3', 'pinned_ref'), ('assemble', 'pinned_ref')]
    assert len(hashes) == 3


def test_preview_refuses_mixed_edit_versions_before_encoding(tmp_path, monkeypatch):
    dest = tmp_path / 'after'
    dest.mkdir()
    for f in (1, 2):
        (dest / f'f_{f:05d}.png').touch()
        (dest / f'f_{f:05d}.json').write_text(json.dumps({'code_hashes': {'assemble': 'same'}}))
    encoded = []
    monkeypatch.setattr(P.subprocess, 'run', lambda *a, **k: encoded.append(a))
    P.preview(tmp_path, 'after', tmp_path / 'sound.wav', 1, 3)
    assert len(encoded) == 1
    (dest / 'f_00002.json').write_text(json.dumps({'code_hashes': {'assemble': 'changed'}}))
    with pytest.raises(ValueError, match='mix different edit code versions'):
        P.preview(tmp_path, 'after', tmp_path / 'sound.wav', 1, 3)
    assert len(encoded) == 1


def test_window_metric_includes_both_bounds_and_refuses_missing_steps():
    records = {f: {'frame': f, 'step_y_mad': value} for f, value in ((9, 100), (10, 2), (11, 9), (12, 3), (13, 100))}
    measured = P.window_metrics(records, 10, 12)
    assert measured['max_at_frame'] == 11
    assert measured['max_step_y_mad'] == 9
    assert len(measured['steps']) == 3
    records[12]['step_y_mad'] = 10
    assert P.window_metrics(records, 10, 12)['max_at_frame'] == 12
    del records[11]['step_y_mad']
    with pytest.raises(ValueError, match='Missing consecutive-frame'):
        P.window_metrics(records, 10, 12)


def test_window_metric_reports_the_largest_mean_level_step_in_either_direction():
    records = {f: {'frame': f, 'step_y_mad': 1.0, 'mean_y_delta': d} for f, d in ((10, 1.5), (11, -6.0), (12, 4.0))}
    measured = P.window_metrics(records, 10, 12)
    assert measured['max_abs_mean_y_delta'] == 6.0 and measured['max_mean_delta_at_frame'] == 11
    records[11]['mean_y_delta'] = -2.0
    assert P.window_metrics(records, 10, 12)['max_mean_delta_at_frame'] == 12
