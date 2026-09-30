"""Backdrop-search geometry and ranking; synthetic pixels only, no assembler initialization."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EDIT), str(EDIT / 'tools')]
import a_caption_search as SEARCH


def test_integral_means_match_independent_box_slices_and_backdrop():
    image = np.full((SEARCH.BD.H, SEARCH.BD.W, 3), 0.3, np.float32)
    image[::5, :150] = 0.8
    image[:100, ::9] = 0.1
    boxes = np.array([[25, 25, 70, 60], [90, 50, 140, 85], [200, 150, 250, 190]])
    edge, texture = SEARCH.integral_means(image, boxes)
    direct = np.asarray([SEARCH.BD.backdrop(image, tuple(box)) for box in boxes])

    def assert_means(actual):
        np.testing.assert_allclose(actual, direct, rtol=0, atol=1e-7)

    assert_means(np.stack((edge, texture), axis=1))
    with pytest.raises(AssertionError):
        assert_means(np.stack((edge[::-1], texture[::-1]), axis=1))


def test_candidate_centres_round_trip_through_textv3_and_keep_all_margins():
    row = next(r for r in SEARCH.AS.titles.text_table('A') if r['id'] == 'T5')
    parts = ('Whoever held it first, they said,', 'would hold the world.')
    broken = SEARCH.prepared_rows([row], {'T5': parts})[0]
    line = SEARCH.AS.titles.TextV3('A', broken, 1.0)
    grid = SEARCH.candidate_grid([line], grid=160, margin=48)

    def assert_margins(boxes, origins):
        assert np.all(boxes[:, 0, :2] >= 48)
        assert np.all(boxes[:, 0, 2:] <= [SEARCH.BD.W - 48, SEARCH.BD.H - 48])
        assert np.all(origins[:, 0] >= 48)
        assert np.all(origins[:, 0] + [line.w, line.h + 5] <= [SEARCH.BD.W - 48, SEARCH.BD.H - 48])

    assert_margins(grid['boxes'], grid['origins'])
    bad = grid['boxes'].copy()
    bad[0, 0, 0] = 47
    with pytest.raises(AssertionError):
        assert_margins(bad, grid['origins'])

    for index in (0, len(grid['y']) // 2, len(grid['y']) - 1):
        placed = SEARCH.AS.titles.TextV3('A', dict(broken, x=int(grid['x'][index]), y=int(grid['y'][index])), 1.0)
        assert [placed.x0, placed.y0] == grid['origins'][index, 0].tolist()
        assert list(SEARCH.BD.box_of(placed)) == grid['boxes'][index, 0].tolist()
    with pytest.raises(AssertionError):
        assert [line.x0, line.y0] == grid['origins'][0, 0].tolist()
    with pytest.raises(ValueError, match='preserve the current words'):
        SEARCH.prepared_rows([row], {'T5': ('Whoever held it first,', 'would hold the world.')})


def test_group_preserves_the_masters_x_origins_and_searches_a_common_y():
    lines = [line for line in SEARCH.AS.titles.lines_v3('A', 1.0) if line.id in ('T6a', 'T6b')]
    grid = SEARCH.candidate_grid(lines, grid=80, margin=48, group=True)

    def assert_x_origins(origins):
        assert len(origins) > 0
        for k, line in enumerate(lines):
            assert np.all(origins[:, k, 0] == line.x0)

    assert_x_origins(grid['origins'])
    for k, line in enumerate(lines):
        np.testing.assert_array_equal(grid['origins'][:, k, 1], np.rint(grid['y'] - line.h / 2))
    broken = grid['origins'].copy()
    broken[:, 0, 0] += 1
    with pytest.raises(AssertionError):
        assert_x_origins(broken)
    with pytest.raises(ValueError, match='Independent search requires one line'):
        SEARCH.candidate_grid(lines, group=False)


def test_group_samples_each_members_steady_interval_including_the_solo_phrase():
    rows = [r for r in SEARCH.AS.titles.text_table('A') if r['id'] in ('T6a', 'T6b')]
    frames = SEARCH.requested_frames(rows, step=8)
    assert frames['T6a'][0] == 1652
    assert frames['T6b'][0] == 1732
    assert 1652 not in frames['T6b']
    assert SEARCH.requested_frames(rows, frames=[1652, 1732]) == dict(T6a=[1652, 1732], T6b=[1732])
    with pytest.raises(ValueError, match='at least one steady sample'):
        SEARCH.requested_frames(rows, frames=[1652])
    with pytest.raises(ValueError, match='inside a selected caption steady interval'):
        SEARCH.requested_frames(rows, frames=[1652, 1732, 9999])


def test_ranking_prioritizes_both_medians_then_worst_values_then_distance():
    samples = {'A': dict(edge=[[0.04, 0.02, 0.01, 0.01], [0.04, 0.02, 0.01, 0.01],
                               [0.00, 0.06, 0.01, 0.01]],
                         texture=[[0, 0, 0.01, 0.01]] * 3)}
    distance = np.array([0, 0, 10, 3])
    ranked = SEARCH.ranked_samples(samples, distance)

    def assert_order(value):
        assert list(value) == [3, 2, 1, 0]

    assert_order(ranked['order'])
    assert ranked['passes_sampled_busy_thresholds'].tolist() == [False, True, True, True]
    with pytest.raises(AssertionError):
        assert_order(np.argsort(ranked['worst_edge']))
    # The second phrase's bad texture must disqualify candidate 3 even though
    # the first phrase passed there; a group average would hide that member.
    samples['B'] = dict(edge=[[0] * 4] * 3, texture=[[0, 0, 0, 0.02]] * 3)
    grouped = SEARCH.ranked_samples(samples, distance)
    assert grouped['passes_sampled_busy_thresholds'].tolist() == [False, True, True, False]
    assert grouped['order'][:2].tolist() == [2, 1]
    with pytest.raises(AssertionError):
        assert grouped['passes_sampled_busy_thresholds'][3]


def test_empty_and_nonfinite_metrics_cannot_be_ranked():
    good = {'A': dict(edge=[[0.0]], texture=[[0.0]])}
    assert SEARCH.ranked_samples(good, [0])['order'].tolist() == [0]
    with pytest.raises(ValueError, match='empty measurement'):
        SEARCH.ranked_samples({}, [0])
    with pytest.raises(ValueError, match='no measured samples'):
        SEARCH.ranked_samples({'A': dict(edge=[], texture=[])}, [0])
    with pytest.raises(ValueError, match='nonfinite'):
        SEARCH.ranked_samples({'A': dict(edge=[[np.nan]], texture=[[0]])}, [0])


def test_search_measures_synthetic_picture_and_never_emits_contrast(monkeypatch):
    monkeypatch.setattr(SEARCH.BD, 'W', 64)
    monkeypatch.setattr(SEARCH.BD, 'H', 48)
    monkeypatch.setattr(SEARCH.AR, 'memory', lambda: 0.0)
    row = dict(id='T1', f_in=0, f_out=28, set='lower', line='synthetic')
    line = SimpleNamespace(id='T1', x0=28, y0=20, y=22, w=8, h=4, size=4,
                           alpha=np.ones((4, 8), np.float32))
    image = np.full((48, 64, 3), 0.2, np.float32)
    image[30::2] = 0.8
    calls = []

    def picture(f):
        calls.append(f)
        return image.copy(), {}, 'synthetic', None

    ctx = SimpleNamespace(picture=picture)
    result = SEARCH.search(ctx, [row], [line], step=2, grid=8, margin=4)
    assert calls == [12, 14]
    assert result['sampled_frames'] == {'T1': [12, 14]}
    assert result['candidate_count'] > 1
    assert result['candidates'][0]['boxes'][0][3] <= 30
    assert result['candidates'][0]['passes_sampled_busy_thresholds']

    def assert_measured_fields(value):
        if isinstance(value, dict):
            assert not any('contrast' in key for key in value)
            for item in value.values():
                assert_measured_fields(item)
        elif isinstance(value, list):
            for item in value:
                assert_measured_fields(item)

    assert_measured_fields(result)
    with pytest.raises(AssertionError):
        assert_measured_fields(dict(result, contrast_est=10))
    ctx.picture = lambda f: (image, {}, 'SLATE missing', None)
    with pytest.raises(RuntimeError, match='missing picture'):
        SEARCH.search(ctx, [row], [line], step=2, grid=8, margin=4)
