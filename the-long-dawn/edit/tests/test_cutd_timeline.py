"""Treatment v2 ranges, restored A opening, source holds and unreleased NEW picture for the115-bar film."""
from copy import deepcopy
import re
from pathlib import Path
import sys

import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS  # noqa: E402
import edl_v3 as EDL  # noqa: E402
from cutd_test_fixtures import edl_namespace


# The34 half-open beat ranges from the approved treatment v2, not inferred from the EDL under test.
BEATS = [(0, 80), (80, 560), (560, 960), (960, 1040), (1040, 1440), (1440, 1680),
         (1680, 1760), (1760, 2080), (2080, 2400), (2400, 2720), (2720, 2960), (2960, 3200),
         (3200, 3440), (3440, 3520), (3520, 3760), (3760, 4080), (4080, 4240), (4240, 4560),
         (4560, 5040), (5040, 5180), (5180, 5680), (5680, 5840), (5840, 6080), (6080, 6400),
         (6400, 6640), (6640, 7040), (7040, 7360), (7360, 7840), (7840, 8080), (8080, 8320),
         (8320, 8640), (8640, 8880), (8880, 9120), (9120, 9200)]
# D07 is the separately authorized drawn-ring burn; its provisional/replacement routing has dedicated tests.
NEW = {'D08': 'embers_D_inscription', 'D09': 'embers_D_forging',
       'D13': 'embers_D_vision', 'D14': 'embers_D_gap',
       'D16': 'embers_D_trap', 'D17': 'falsedawn_brink', 'D18': 'embers_D_crowns',
       'D19': 'falsedawn_twofires', 'D21b': 'embers_D_holdout', 'D22': 'embers_D_instep',
       'D23': 'book_D_oldfire', 'D24': 'embers_D_unfinished', 'D27': 'falsedawn_watch',
       'D28': 'falsedawn_truedawn', 'D31': 'book_lastleaf_open'}


def assert_treatment(rows):
    assert len(rows) == 41
    cursor = 0
    for row in rows:
        assert row['f0'] == cursor and row['f1'] > cursor
        assert re.fullmatch(r'D(?:0[1-9]|[12][0-9]|3[0-4])[a-e]?', row['code'])
        cursor = row['f1']
    assert cursor == 9200
    for number, (start, end) in enumerate(BEATS, 1):
        parts = [row for row in rows if row['code'][:3] == f'D{number:02d}']
        assert parts and (parts[0]['f0'], parts[-1]['f1']) == (start, end)


def test_real_d_tiles_the34_treatment_beats_without_a_gap_or_overlap():
    assert EDL.TOTAL['D'] == 9200
    assert_treatment(EDL.D)
    bad = deepcopy(EDL.D)
    bad[3]['f0'] += 1
    with pytest.raises(AssertionError):
        assert_treatment(bad)
    assert [(r['sec'], r['f0']) for i, r in enumerate(EDL.D) if i == 0 or r['sec'] != EDL.D[i - 1]['sec']] == [
        ('D0', 0), ('D1', 1440), ('D2', 2400), ('D3', 3520), ('D4', 4080), ('D5', 5840),
        ('D6', 7040), ('D7', 7360)]


def assert_opening(rows, windows):
    for number, (a, d) in enumerate(zip(EDL.A[:5], rows[:5]), 1):
        assert {k: v for k, v in d.items() if k not in ('sec', 'code')} == {
            k: v for k, v in a.items() if k not in ('sec', 'code')}
        assert (d['sec'], d['code']) == ('D0', f'D{number:02d}')
    assert [t for t in windows if t['f1'] <= 1440] == [t for t in EDL.TRANS['A'] if t['f1'] <= 1440]


def test_entire_a_opening_and_its_six_windows_are_restored_independently():
    assert_opening(EDL.D, EDL.TRANS['D'])
    assert len([t for t in EDL.TRANS['D'] if t['f1'] <= 1440]) == 6
    bad = deepcopy(EDL.D)
    bad[4]['takes'][0]['off'] += 1
    with pytest.raises(AssertionError):
        assert_opening(bad, EDL.TRANS['D'])
    bad_windows = deepcopy(EDL.TRANS['D'])
    bad_windows.pop(0)
    with pytest.raises(AssertionError):
        assert_opening(EDL.D, bad_windows)
    assert EDL.D[4]['takes'] is not EDL.A[4]['takes']
    av = next(t for t in EDL.TRANS['A'] if t['kind'] == 'vision' and t['f0'] == 1280)
    dv = next(t for t in EDL.TRANS['D'] if t['kind'] == 'vision')
    assert dv['track'] is not av['track']


def assert_new_is_slate(row):
    assert not row['takes']
    assert AS.plan_shot(row, 'D', None)['kind'] == 'slate'


def test_new_stems_cannot_auto_promote_when_a_folder_arrives(monkeypatch):
    # Every filesystem lookup claims all frames exist. Unadopted NEW picture still remains an explicit slate.
    monkeypatch.setattr(AS, 'complete', lambda *args: True)
    monkeypatch.setattr(AS, 'locate', lambda *args: ('synthetic-available-frame', False))
    by_code = {row['code']: row for row in edl_namespace()['D']}
    source = (EDIT / 'edl_v3.py').read_text()
    for code, expected in NEW.items():
        row = by_code[code]
        assert expected in row['desc']
        if code=='D08':
            assert "D_NEW_TAKES['D08']; its need=(1680, 2079)" in source
        else:
            assert f'expected stem {expected}, off=0, need=({row["f0"]}, {row["f1"] - 1})' in source
        assert_new_is_slate(row)
        bad = dict(row, takes=[EDL.T(expected, mode='exact')])
        with pytest.raises(AssertionError):
            assert_new_is_slate(bad)


# Explicit harvesting ranges; D01..D05 are already pinned to A's complete row/take chain above.
REUSE = {
    'D06': ('cand_t1_current-words-held', 320, 559),
    'D11a': ('cand_sweep_soft-entry', 1680, 1687), 'D11b': ('book_C_ft', 1688, 1717),
    'D11c': ('book_C', 1718, 1904), 'D11d': ('book_C_ft', 1905, 1919),
    'D15': ('book_C5_refusal', 2080, 2319), 'D20': ('reveal_A', 3660, 3799),
    'D21a': ('cand_map_beacon-falloff', 3440, 3739), 'D21c': ('cand_map_beacon-falloff', 3739, 3739),
    'D21d': ('cand_map_beacon-falloff', 3740, 3815), 'D21e': ('cand_map_beacon-falloff', 3815, 3815),
    'D25': ('cand_deep_leaned_ladders', 4240, 4479),
    'D26': ('cand_crossing_both_decal_cap', 5180, 5579), 'D29': ('dawnrev_A', 6000, 6239),
    'D30': ('book_C', 6160, 6399), 'D32': ('cand_pen_soft_spine_metal', 5440, 5679),
    'D33': ('book_C', 6960, 7199),
}


def assert_source_range(row, expected):
    # D11's original takes remain the explicit unconfigured fallback behind its local re-bake.
    take = row['takes'][-1]
    stem, first, last = expected
    assert take['mode'] == 'exact' and take['stem'] == stem
    frames = [EDL.source_frame(take, f) for f in range(row['f0'], row['f1'])]
    assert (frames[0], frames[-1]) == (first, last)
    assert frames == ([first] * len(frames) if first == last else list(range(first, last + 1)))


def test_reuse_source_clocks_and_holds_match_the_treatment():
    by_code = {row['code']: row for row in EDL.D}
    for code, expected in REUSE.items():
        row = by_code[code]
        assert_source_range(row, expected)
        bad = deepcopy(row)
        key = 'hold' if 'hold' in bad['takes'][-1] else 'off'
        bad['takes'][-1][key] += 1
        with pytest.raises(AssertionError):
            assert_source_range(bad, expected)
    assert EDL.source_frame(by_code['D21d']['takes'][0], 5600) == 3786
    assert by_code['D11c']['takes'][-1]['under'] == ('hold', 'embers_C3', 1679)
    assert by_code['D11d']['takes'][-1]['under'] == ('hold', 'embers_C3', 1920)
    assert EDL.D_CROSSING_SOURCE_START == 5180
    assert by_code['D26']['takes'][0]['need'] == (5180, 5579)
    map_rows = [r for r in EDL.D if r['code'].startswith('D21')]
    assert sum(r['f1'] - r['f0'] for r in map_rows) == 500
    moving = [r for r in map_rows if r['takes'] and 'hold' not in r['takes'][0]
              and r['takes'][0]['stem'] == 'cand_map_beacon-falloff']
    assert sum(r['f1'] - r['f0'] for r in moving) == 376
    assert sum(r['f1'] - r['f0'] for r in map_rows if r['takes'] and 'hold' in r['takes'][0]) == 84


def assert_holdout_adoption(rows):
    by_code = {row['code']: row for row in rows}
    holdout = by_code['D21b']
    assert (holdout['f0'],holdout['f1']) == (5480,5520)
    assert holdout['takes'] and holdout['takes'][0]['stem'] == 'embers_D_holdout'
    take = holdout['takes'][0]
    assert take['need'] == (5480,5519) and 'hold' not in take
    assert [EDL.source_frame(take,f) for f in range(5480,5520)] == list(range(5480,5520))
    advancing = [row for row in rows if row['code'].startswith('D21') and row['takes']
                 and 'hold' not in row['takes'][0]]
    assert sum(row['f1']-row['f0'] for row in advancing) == 416
    assert {row['code'] for row in advancing} == {'D21a','D21b','D21d'}


def test_holdout_adopts_40_new_picture_frames_without_retiming_the_376_map_frames():
    assert_holdout_adoption(EDL.D)
    for change in ({'takes':[]},
                   {'takes':[EDL.T('embers_D_holdout',0,'exact',hold=5480,need=(5480,5519))]},
                   {'takes':[EDL.T('embers_D_holdout',1,'exact',need=(5480,5519))]},
                   {'takes':[EDL.T('embers_D_holdout',0,'exact',need=(5480,5518))]}):
        bad=[dict(row,**change) if row['code']=='D21b' else row for row in EDL.D]
        with pytest.raises(AssertionError):
            assert_holdout_adoption(bad)


def assert_rebake_required(row):
    plate='embers_D_brink D2960' if row['code']=='D11d' else 'embers_D_race D2719'
    assert 're-bake' in row['desc'] and plate in row['desc']
    for take in row['takes']:
        assert 're-bake' in take['note'] and plate in take['note']
        assert not EDL.is_final_take(take)


def test_deep_reuse_does_not_certify_the_retired_closed_ring_picture():
    for row in (r for r in EDL.D if r['code'].startswith('D11')):
        assert_rebake_required(row)
        bad = deepcopy(row)
        bad['takes'][0]['final_eligible'] = True
        with pytest.raises(AssertionError):
            assert_rebake_required(bad)
    glow = next(r for r in EDL.D if r['code'] == 'D11d')
    assert 'embers_D_brink D2960' in glow['desc'] and 'embers_D_brink D2960' in glow['takes'][0]['note']


def test_hold_is_opt_in_and_never_changes_legacy_take_documents():
    legacy = EDL.T('book_C', off=-1920, mode='exact')
    assert 'hold' not in legacy
    assert EDL.source_frame(legacy, 8880) == 6960
    assert EDL.T('book_C', off=-1920, mode='exact', hold=None) == legacy
    held = EDL.T('cand_map_beacon-falloff', mode='exact', hold=3739)
    assert {EDL.source_frame(held, f) for f in range(5520, 5554)} == {3739}
    for invalid in (-1, 1.5, True, '3739'):
        with pytest.raises(ValueError, match='hold'):
            EDL.T('cand_map_beacon-falloff', hold=invalid)


def assert_transition_clocks(windows):
    # D07's burn is covered by its integration tests and may add a separately authored window at1680–1760.
    retained = [t for t in windows if t['f0'] >= 1440 and not (t['f0'] == 1680 and t['f1'] == 1760)]
    assert [(t['f0'], t['f1'], t['kind']) for t in retained] == [
        (1520, 1660, 'caption_grade'), (2945, 2981, 'deep_reveal'), (3750, 3786, 'burn'),
        (5180, 5220, 'ring_burn'), (6388, 6412, 'dissolve'),
        (6636, 6644, 'dissolve'), (8628, 8652, 'dissolve'), (8868, 8892, 'dissolve'),
        (8948, 9080, 'caption_grade'), (9084, 9120, 'floor')]
    assert next(t for t in windows if t['f0'] == 8628)['cut'] == 8640
    burn = next(t for t in windows if t.get('cut') == 3760)
    assert burn['layer_off'] == -1440
    assert (burn['f0'] + burn['layer_off'], burn['f1'] + burn['layer_off']) == (2310, 2346)
    assert {t['id']: (t['source_off'], t['band_frame_off']) for t in windows if t['kind'] == 'caption_grade'} == {
        'R02': (-1120, -1120), 'title': (-1920, -3200)}
    assert not any(t['kind'] in ('dawn_dissolve', 'dawn_sweep', 'finish_ramp') for t in windows)


def test_transition_windows_follow_d_clocks_and_keep_c_layer_clocks():
    assert_transition_clocks(EDL.TRANS['D'])
    for key, value in (('f0', 8630), ('f1', 8651), ('kind', 'page_turn'), ('cut', 8641)):
        bad = deepcopy(EDL.TRANS['D'])
        next(t for t in bad if t.get('cut') == 8640)[key] = value
        with pytest.raises(AssertionError):
            assert_transition_clocks(bad)
    bad = deepcopy(EDL.TRANS['D'])
    next(t for t in bad if t.get('cut') == 3760)['layer_off'] = 0
    with pytest.raises(AssertionError):
        assert_transition_clocks(bad)
    bad = deepcopy(EDL.TRANS['D'])
    next(t for t in bad if t['kind'] == 'caption_grade')['band_frame_off'] = 0
    with pytest.raises(AssertionError):
        assert_transition_clocks(bad)
