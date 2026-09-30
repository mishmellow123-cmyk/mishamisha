"""Round 8 changes transition windows without retiming the accepted D picture or words.

The row and caption baselines are from accepted ecb106b. No production images are loaded.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assemble as AS
import edl_v3 as EDL
import ring_burn as RB
import titles


FIXED_DISSOLVES = {4080: (4068, 4092), 4240: (4234, 4246),
                   4560: (4554, 4566), 5840: (5828, 5852)}
# All 41 rows, including Deep helper boundaries and the map's insert/holds.
ACCEPTED_ROWS = [
    ('D01', 0, 80), ('D02', 80, 560), ('D03', 560, 960), ('D04', 960, 1040),
    ('D05', 1040, 1440), ('D06', 1440, 1680), ('D07', 1680, 1760),
    ('D08', 1760, 2080), ('D09', 2080, 2400), ('D10', 2400, 2720),
    ('D11a', 2720, 2728), ('D11b', 2728, 2758), ('D11c', 2758, 2945),
    ('D11d', 2945, 2960), ('D12', 2960, 3200), ('D13', 3200, 3440),
    ('D14', 3440, 3520), ('D15', 3520, 3760), ('D16', 3760, 4080),
    ('D17', 4080, 4240), ('D18', 4240, 4560), ('D19', 4560, 5040),
    ('D20', 5040, 5180), ('D21a', 5180, 5480), ('D21b', 5480, 5520),
    ('D21c', 5520, 5554), ('D21d', 5554, 5630), ('D21e', 5630, 5680),
    ('D22', 5680, 5840), ('D23', 5840, 6080), ('D24', 6080, 6400),
    ('D25', 6400, 6640), ('D26', 6640, 7040), ('D27', 7040, 7360),
    ('D28', 7360, 7840), ('D29', 7840, 8080), ('D30', 8080, 8320),
    ('D31', 8320, 8640), ('D32', 8640, 8880), ('D33', 8880, 9120),
    ('D34', 9120, 9200),
]
CAPTION_SHA256 = 'f517ff3db22a4cf321177d9539a7ecc789df27423c215dca39691cd8ed0386d5'
OLDFIRE_BURN = dict(f0=6080, f1=6120, cut=6080, kind='ring_burn',
                    center=(1013., 187.), t_open=6082., speed=10., seed=12)
# Conservative support around the delivered D6079 caption, measured in native 1920x804 pixels.
OLDFIRE_CAPTION_BOX = (449, 681, 1488, 734)


def assert_window(windows, cut, start, end, kind='dissolve'):
    matches = [t for t in windows if t.get('cut') == cut]
    assert len(matches) == 1
    assert tuple(matches[0][key] for key in ('f0', 'f1', 'cut', 'kind')) == (start, end, cut, kind)


@pytest.mark.parametrize('cut', FIXED_DISSOLVES)
def test_four_authored_dissolves_have_exact_half_open_windows(cut):
    assert_window(EDL.TRANS['D'], cut, *FIXED_DISSOLVES[cut])


@pytest.mark.parametrize('cut', FIXED_DISSOLVES)
@pytest.mark.parametrize('fault', ['f0', 'f1', 'cut', 'kind', 'missing', 'duplicate'])
def test_each_dissolve_pin_rejects_wrong_clock_kind_and_missing_or_duplicate_window(cut, fault):
    start, end = FIXED_DISSOLVES[cut]
    correct = dict(f0=start, f1=end, cut=cut, kind='dissolve')
    assert_window([correct], cut, start, end)
    broken = deepcopy(correct)
    if fault in ('f0', 'f1', 'cut'):
        broken[fault] += 1
    elif fault == 'kind':
        broken['kind'] = 'page_turn'
    windows = [] if fault == 'missing' else [broken, correct] if fault == 'duplicate' else [broken]
    with pytest.raises(AssertionError):
        assert_window(windows, cut, start, end)


def assert_row_clocks(rows, total):
    assert total == 9200
    assert [(r['code'], r['f0'], r['f1']) for r in rows] == ACCEPTED_ROWS


def test_all_accepted_row_boundaries_and_total_remain_fixed():
    assert_row_clocks(EDL.D, EDL.TOTAL['D'])
    for index in range(len(EDL.D)):
        for clock in ('f0', 'f1'):
            broken = deepcopy(EDL.D)
            broken[index][clock] += 1
            with pytest.raises(AssertionError):
                assert_row_clocks(broken, EDL.TOTAL['D'])
    with pytest.raises(AssertionError):
        assert_row_clocks(EDL.D, 9201)


def assert_caption_document(rows):
    encoded = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(encoded).hexdigest() == CAPTION_SHA256


def test_all_accepted_caption_words_windows_and_baked_status_remain_fixed():
    rows = titles.text_table('D')
    assert_caption_document(rows)
    for index in range(len(rows)):
        for clock in ('f_in', 'f_out'):
            broken = deepcopy(rows)
            broken[index][clock] += 1
            with pytest.raises(AssertionError):
                assert_caption_document(broken)
    for field, value in [('line', 'changed words'), ('set', 'lower')]:
        broken = deepcopy(rows)
        next(row for row in broken if row['id'] == 'D14')[field] = value
        with pytest.raises(AssertionError):
            assert_caption_document(broken)


def assert_disjoint(windows):
    ordered = sorted(windows, key=lambda t: t['f0'])
    assert all(t['f0'] < t['f1'] for t in ordered)
    assert all(a['f1'] <= b['f0'] for a, b in zip(ordered, ordered[1:]))


def test_no_d_window_can_be_shadowed_by_an_overlapping_earlier_dispatch():
    assert_disjoint(EDL.TRANS['D'])
    for start, end in [(1681, 1683), (5000, 5000)]:
        broken = deepcopy(EDL.TRANS['D'])
        broken.append(dict(f0=start, f1=end, cut=start, kind='dissolve'))
        with pytest.raises(AssertionError):
            assert_disjoint(broken)


@pytest.mark.parametrize('cut', FIXED_DISSOLVES)
def test_dissolve_source_handles_freeze_only_beyond_the_existing_cut(cut):
    start, end = FIXED_DISSOLVES[cut]
    spec = dict(f0=start, f1=end, cut=cut, kind='dissolve')
    expected = [(f, cut) if f < cut else (cut - 1, f) for f in range(start, end)]
    assert [AS.transition_source_frames(spec, f) for f in range(start, end)] == expected
    with pytest.raises(AssertionError):
        assert [AS.transition_source_frames(dict(spec, cut=cut + 1), f)
                for f in range(start, end)] == expected


def assert_straight_cut(windows, cut):
    assert not any(t.get('cut') == cut or t['f0'] <= cut < t['f1'] for t in windows)


def test_crossing_uses_a_straight_cut_after_the_frozen_procession_review():
    assert_straight_cut(EDL.TRANS['D'], 6640)
    for start, end in [(6636, 6644), (6628, 6652)]:
        rejected = list(EDL.TRANS['D']) + [dict(f0=start, f1=end, cut=6640, kind='dissolve')]
        with pytest.raises(AssertionError):
            assert_straight_cut(rejected, 6640)


@pytest.mark.parametrize('cut', FIXED_DISSOLVES)
def test_live_dispatch_uses_authored_window_and_frozen_source_handles(cut, monkeypatch):
    start, end = FIXED_DISSOLVES[cut]
    calls = []
    source = {'stem': 'synthetic'}
    shot = {'code': 'synthetic'}
    plain = np.full((2, 3, 3), .9, np.float32)

    def raw(self, frame):
        calls.append(frame)
        return np.full_like(plain, .1 if frame < cut else .3), shot, 'take', source

    monkeypatch.setattr(AS.Ctx, 'picture', raw)
    ctx = SimpleNamespace(cut='D', W=3, H=2,
                          picture=lambda f: (plain.copy(), shot, 'plain', source))
    picture = AS._transitions(ctx)
    for frame in (start, cut - 1, cut, end - 1):
        calls.clear()
        image, _, status, _ = picture(frame)
        expected = [frame, cut] if frame < cut else [cut - 1, frame]
        assert calls == expected
        assert status == 'take + dissolve'
        assert np.all((image > .1) & (image < .3))
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(image, plain)
    for frame in (start - 1, end):
        calls.clear()
        image, _, status, _ = picture(frame)
        assert calls == [] and status == 'plain'
        np.testing.assert_array_equal(image, plain)


def assert_oldfire_burn(spec):
    assert {k: v for k, v in spec.items() if k != 'note'} == OLDFIRE_BURN


def test_oldfire_burn_uses_the_delivered_flame_and_measured_clock():
    matches = [t for t in EDL.TRANS['D'] if t.get('cut') == 6080]
    assert len(matches) == 1
    assert_oldfire_burn(matches[0])
    for key, value in [('f0', 6079), ('f1', 6119), ('cut', 6081), ('kind', 'dissolve'),
                       ('center', (998., 92.)), ('t_open', 6080.), ('speed', 9.4), ('seed', 13)]:
        with pytest.raises(AssertionError):
            assert_oldfire_burn(dict(matches[0], **{key: value}))


def assert_oldfire_sources(rows, spec):
    by_code = {row['code']: row for row in rows}
    page, incoming = (by_code[code]['takes'][0] for code in ('D23', 'D24'))
    assert (page['stem'], page['mode'], page['need'], page['baked_text']) == (
        'book_D_oldfire', 'exact', (5840, 6079), ('D14',))
    assert (incoming['stem'], incoming['mode'], incoming['need']) == (
        'embers_D_unfinished', 'exact', (6080, 6399))
    for frame in range(6080, 6120):
        before, after = AS.transition_source_frames(spec, frame)
        assert (before, after) == (6079, frame)
        assert EDL.source_frame(page, before) == 6079
        assert EDL.source_frame(incoming, after) == frame


def test_oldfire_holds_the_baked_words_on_6079_while_unfinished_advances():
    assert_oldfire_sources(EDL.D, OLDFIRE_BURN)
    for code, key, value in [('D23', 'off', -1), ('D23', 'baked_text', ()),
                              ('D23', 'need', (5840, 6078)), ('D24', 'hold', 6080)]:
        broken = deepcopy(EDL.D)
        next(row for row in broken if row['code'] == code)['takes'][0][key] = value
        with pytest.raises(AssertionError):
            assert_oldfire_sources(broken, OLDFIRE_BURN)
    assert next(row for row in titles.text_table('D') if row['id'] == 'D15')['f_in'] == 6104


def test_oldfire_first_frame_keeps_the_complete_held_page(monkeypatch):
    outgoing = np.full((2, 3, 3), .6, np.float32)
    incoming = np.full_like(outgoing, .2)
    monkeypatch.setattr(RB, 'layers', lambda *a: pytest.fail('first frame must not burn the held page'))
    result = RB.composite(outgoing, incoming, 6080, OLDFIRE_BURN)
    np.testing.assert_array_equal(result, outgoing)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(result, incoming)


def assert_untouched_caption(layers):
    glow, keep, cover = layers
    np.testing.assert_array_equal(keep, np.ones_like(keep))
    np.testing.assert_array_equal(cover, np.ones_like(cover))
    np.testing.assert_array_equal(glow, np.zeros_like(glow))


def assert_no_outgoing_or_burn(layers):
    for layer in layers:
        np.testing.assert_array_equal(layer, np.zeros_like(layer))


def test_oldfire_caption_survives_until_6093_and_clears_for_caption15():
    # Quarter-size real procedural masks retain native screen coordinates; no delivered image is read.
    width, height = 480, 201
    x = (np.arange(width) + .5) * 1920 / width
    y = (np.arange(height) + .5) * 804 / height
    x0, y0, x1, y1 = OLDFIRE_CAPTION_BOX
    roi = np.ix_((y >= y0) & (y < y1), (x >= x0) & (x < x1))
    layers = {f: RB.layers(f, OLDFIRE_BURN, width, height)
              for f in (*range(6081, 6095), 6103, 6104, 6108, 6109)}
    caption = lambda f: tuple(layer[roi] for layer in layers[f])
    for frame in range(6081, 6094):
        assert_untouched_caption(caption(frame))
    with pytest.raises(AssertionError):
        assert_untouched_caption(caption(6094))
    assert_no_outgoing_or_burn(caption(6104))
    with pytest.raises(AssertionError):
        assert_no_outgoing_or_burn(caption(6103))
    assert_no_outgoing_or_burn(layers[6109])
    with pytest.raises(AssertionError):
        assert_no_outgoing_or_burn(layers[6108])
