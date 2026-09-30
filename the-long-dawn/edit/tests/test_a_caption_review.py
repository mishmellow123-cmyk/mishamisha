"""A caption audit contracts, using glyph masks and small synthetic arrays only.

The assembler's real _init and picture readers never run. Each accepting contract
also sees a negative control and must reject it with an AssertionError.
"""
from pathlib import Path
from copy import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EDIT), str(EDIT / 'tools')]
import a_caption_review as AR


def assert_clear(geom):
    assert geom['clipped_pixels'] == 0
    assert geom['entry_clipped_pixels'] == 0
    assert geom.get('animated_clipped_pixels', 0) == 0


def assert_flags(per, geom, expected):
    assert AR.flags(per, geom) == expected


def assert_complete(result):
    assert result['complete']
    assert result['requested'] > 0
    assert result['measured'] == result['requested']
    assert result['missing'] == []


def line_at(x, y, alpha=None):
    return SimpleNamespace(x0=x, y0=y,
                           alpha=np.ones((2, 4), np.float32) if alpha is None else alpha)


def sample(worst=5.0, edge=0.0, texture=0.0):
    return dict(worst=worst, edge=edge, texture=texture)


@pytest.fixture(scope='module')
def a_lines():
    # These are only font masks; no delivered picture or assembler is initialized.
    return {line.id: line for line in AR.AS.titles.lines_v3('A', 1.0)}


def test_audit_uses_actual_context_lines(monkeypatch, tmp_path, a_lines):
    rows = [r for r in AR.AS.titles.text_table('A') if r['id'] in ('T6a', 'T6b')]
    master = [copy(a_lines[r['id']]) for r in rows]
    # A context-specific translation distinguishes using ctx.lines from
    # rebuilding TextV3 from the same row (which matches today's stacked T6).
    for line in master:
        line.x0 += 37
        line.y0 += 13
    independent = [AR.AS.titles.TextV3('A', r, 1.0) for r in rows]

    # Empty steady ranges exercise mask selection without composing even a
    # synthetic frame. run() must retain the context's actual objects.
    short_rows = [dict(row, f_out=row['f_in'] + 24) for row in rows]
    ctx = SimpleNamespace(lines=master, shots=[], plans=[])
    monkeypatch.setattr(AR.AS, '_init', lambda *args: None)
    monkeypatch.setattr(AR.AS, '_CTX', ctx)
    monkeypatch.setattr(AR.AS.titles, 'text_table', lambda cut: short_rows)
    monkeypatch.setattr(AR, 'memory', lambda: 0.0)
    result = AR.run(tmp_path / 'row.json')

    def assert_master_extents(audits):
        assert [a['row']['id'] for a in audits] == [r['id'] for r in rows]
        for audit, actual in zip(audits, master):
            assert audit['geometry']['extent'] == AR.geometry(actual)['extent']

    assert_master_extents(result['captions'])
    with pytest.raises(AssertionError):
        assert_master_extents([])
    for index, centered in enumerate(independent):
        incorrect = [dict(a) for a in result['captions']]
        incorrect[index]['geometry'] = AR.geometry(centered)
        with pytest.raises(AssertionError):
            assert_master_extents(incorrect)


def test_legacy_shared_row_differs_from_independently_centered_phrases(monkeypatch):
    titles = AR.AS.titles
    monkeypatch.setitem(titles.SET_AS, 'A', {**titles.SET_AS['A'], 'T6a': 'row', 'T6b': 'row'})
    monkeypatch.setattr(titles, 'A_PLACEMENT', {rid: place for rid, place in titles.A_PLACEMENT.items()
                                             if rid not in ('T6a', 'T6b')})
    rows = [r for r in titles.text_table('A') if r['id'] in ('T6a', 'T6b')]
    paired = [line for line in titles.lines_v3('A', 1.0) if line.id in ('T6a', 'T6b')]

    def assert_shared_row(lines):
        left, right = lines
        assert right.x0 - (left.x0 + left.w) == pytest.approx(0.9 * left.size, abs=1)
        assert (left.x0 + right.x0 + right.w) / 2 == pytest.approx(AR.BD.W / 2, abs=0.5)
        assert left.y0 == right.y0

    assert_shared_row(paired)
    with pytest.raises(AssertionError):
        assert_shared_row([titles.TextV3('A', row, 1.0) for row in rows])


def test_a_placement_changes_preserve_barmap_timing_and_authorized_words():
    titles = AR.AS.titles
    locked = json.loads((EDIT.parent / 'music' / 'v3' / 'barmap_A.json').read_text())['text']
    expected = {row['id']: (titles.H5_WORDS['A'].get(row['id'], row['line']), row['f_in'], row['f_out'])
                for row in locked}
    rows = titles.text_table('A')

    def assert_script_preserved(actual):
        assert len(actual) == len(expected)
        assert {row['id']: (row['line'], row['f_in'], row['f_out']) for row in actual} == expected
        for row in actual:
            if row.get('lines'):
                assert ' '.join(row['lines']) == row['line']

    assert_script_preserved(rows)
    for changes in (dict(line='synthetic changed wording'), dict(f_in=rows[0]['f_in'] + 1)):
        wrong = [dict(row) for row in rows]
        wrong[0].update(changes)
        with pytest.raises(AssertionError):
            assert_script_preserved(wrong)
    wrong_break = [dict(row) for row in rows]
    broken = next(row for row in wrong_break if row.get('lines'))
    broken['lines'] = tuple(reversed(broken['lines']))
    with pytest.raises(AssertionError):
        assert_script_preserved(wrong_break)
    with pytest.raises(AssertionError):
        assert_script_preserved(rows[:-1])


def test_confirmed_a_blocks_and_stack_reach_the_renderer(monkeypatch, a_lines):
    # Receipt for the confirmed placement patch; coordinates are TextV3
    # centres in the 1920x804 master, and the third value counts text lines.
    expected = {'T5': (440, 600, 2), 'T6a': (440, 600, 1), 'T6b': (440, 680, 1),
                'T7': (1600, 648, 2), 'T8': (1600, 648, 2), 'T9': (960, 724, 1),
                'T13': (960, 160, 1), 'T14': (1420, 570, 1)}

    def assert_placements(lines):
        for rid, (x, y, bands) in expected.items():
            line = lines[rid]
            assert abs(line.x0 + line.w / 2 - x) <= 0.5, rid
            assert abs(line.y0 + line.h / 2 - y) <= 0.5, rid
            assert len(line.bands) == bands, rid
            assert line.kind == 'lower', rid

    def assert_reading_stack(top, bottom):
        assert abs(top.x0 + top.w / 2 - bottom.x0 - bottom.w / 2) <= 1
        assert AR.geometry(top)['extent'][3] < AR.geometry(bottom)['extent'][1]

    assert_placements(a_lines)
    assert_reading_stack(a_lines['T6a'], a_lines['T6b'])
    with pytest.raises(AssertionError):
        assert_reading_stack(a_lines['T6b'], a_lines['T6a'])
    titles = AR.AS.titles
    monkeypatch.setattr(titles, 'A_PLACEMENT', {'T14': titles.A_PLACEMENT['T14']})
    monkeypatch.setitem(titles.SET_AS, 'A', {**titles.SET_AS['A'], 'T6a': 'row', 'T6b': 'row'})
    legacy = {line.id: line for line in titles.lines_v3('A', 1.0)}
    with pytest.raises(AssertionError):
        assert_placements(legacy)
    with pytest.raises(AssertionError):
        assert_reading_stack(legacy['T6a'], legacy['T6b'])


def test_all_actual_a_caption_glyphs_fit_at_rest_and_on_entry(a_lines):
    captions = [line for rid, line in a_lines.items() if rid != 'title']
    assert captions
    for line in captions:
        assert_clear(AR.geometry(line))
    clipped = line_at(-1, 30)
    with pytest.raises(AssertionError):
        assert_clear(AR.geometry(clipped))


@pytest.mark.parametrize('x,y', [(-1, 20), (AR.BD.W - 3, 20), (20, -1), (20, AR.BD.H - 1)])
def test_geometry_rejects_each_frame_edge(x, y):
    assert_clear(AR.geometry(line_at(20, 20)))
    bad = AR.geometry(line_at(x, y))
    assert bad['clipped_pixels'] > 0
    with pytest.raises(AssertionError):
        assert_clear(bad)


def test_geometry_counts_antialias_fringe_and_five_pixel_entry_rise():
    fringe = np.array([[0.01, 1.0]], np.float32)
    assert_clear(AR.geometry(line_at(0, 20, fringe)))
    bad_fringe = AR.geometry(line_at(-1, 20, fringe))
    assert bad_fringe['glyph_pixels'] == 2
    assert bad_fringe['clipped_pixels'] == 1
    with pytest.raises(AssertionError):
        assert_clear(bad_fringe)

    # At H-2 the 2px glyph is wholly inside the frame at rest, but its
    # downward entry offset clips. The last safe origin is H-2-5.
    assert_clear(AR.geometry(line_at(20, AR.BD.H - 7)))
    entry = AR.geometry(line_at(20, AR.BD.H - 2))
    assert entry['clipped_pixels'] == 0
    assert entry['entry_clipped_pixels'] == 8
    with pytest.raises(AssertionError):
        assert_clear(entry)
    assert_flags([sample()], entry, ['CLIPPED'])
    with pytest.raises(AssertionError):
        assert_flags([sample()], entry, [])


def test_entry_blur_can_clip_even_when_the_static_glyph_fits():
    alpha = np.zeros((12, 12), np.float32)
    alpha[5:7, 5:7] = 1.0

    def animated_line(x):
        line = line_at(x, 20, alpha)
        line.f_in, line.f_out, line.s = 0, 30, 1.0
        line.stag = np.zeros_like(alpha)
        line._fade = lambda f: AR.AS.titles.TextV3._fade(line, f)
        return line

    clear_line, clipping_line = animated_line(0), animated_line(-3)
    # Both solid glyphs fit. The real A envelope blurs the second one's
    # nonzero alpha into the negative columns during its entry/exit.
    assert_clear(AR.geometry(clipping_line))
    clear = {**AR.geometry(clear_line), **AR.animated_geometry(clear_line)}
    clipped = {**AR.geometry(clipping_line), **AR.animated_geometry(clipping_line)}
    assert_clear(clear)
    assert clear['clipped_frames'] == []
    assert clipped['animated_clipped_pixels'] > 0
    assert clipped['clipped_frames']
    assert clipped['animated_extent'][0] < clipped['extent'][0]
    with pytest.raises(AssertionError):
        assert_clear(clipped)
    assert_flags([sample()], clipped, ['CLIPPED'])
    with pytest.raises(AssertionError):
        assert_flags([sample()], clipped, [])


@pytest.mark.parametrize('ratio,expected', [
    (np.nextafter(3.0, 0.0), ['LOW-CONTRAST']),
    (3.0, ['MARGINAL']),
    (np.nextafter(4.5, 0.0), ['MARGINAL']),
    (4.5, []),
])
def test_contrast_flags_use_the_worst_frame_and_exact_boundaries(ratio, expected):
    geom = AR.geometry(line_at(20, 20))
    per = [sample(10.0), sample(ratio), sample(10.0)]
    assert_flags(per, geom, expected)
    wrong = [] if expected else ['MARGINAL']
    with pytest.raises(AssertionError):
        assert_flags(per, geom, wrong)


@pytest.mark.parametrize('metric,threshold,flag', [
    ('edge', AR.BD.BUSY_EDGE, 'BUSY-EDGE'),
    ('texture', AR.BD.BUSY_TEXTURE, 'BUSY-TEXTURE'),
])
def test_busyness_uses_median_and_strict_threshold(metric, threshold, flag):
    geom = AR.geometry(line_at(20, 20))
    at_limit = [sample(**{metric: threshold}) for _ in range(3)]
    assert_flags(at_limit, geom, [])
    isolated = [sample(), sample(), sample(**{metric: threshold * 10})]
    assert_flags(isolated, geom, [])
    over = [sample(), sample(**{metric: np.nextafter(threshold, np.inf)}),
            sample(**{metric: threshold * 10})]
    assert_flags(over, geom, [flag])
    with pytest.raises(AssertionError):
        assert_flags(over, geom, [])
    with pytest.raises(AssertionError):
        assert_flags(at_limit, geom, [flag])
    with pytest.raises(AssertionError):
        assert_flags(isolated, geom, [flag])


def test_unmeasured_caption_cannot_have_clean_flags():
    geom = AR.geometry(line_at(20, 20))
    assert_flags([sample()], geom, [])
    assert_flags([], geom, ['UNMEASURED'])
    with pytest.raises(AssertionError):
        assert_flags([], geom, [])


def test_timing_finds_hard_cuts_and_transition_overlap_with_half_open_bounds(monkeypatch):
    monkeypatch.setitem(AR.AS.EDL.EDL, 'A', [dict(f0=0), dict(f0=100), dict(f0=200)])
    window = dict(kind='dissolve', f0=190, f1=210, cut=200)
    monkeypatch.setitem(AR.AS.EDL.TRANS, 'A', [window])

    def assert_no_cut(row):
        assert AR.timing(row)['cuts'] == []

    def assert_no_window(row):
        assert AR.timing(row)['windows'] == []

    assert_no_cut(dict(f_in=100, f_out=200))
    hard = dict(f_in=99, f_out=101)
    assert AR.timing(hard) == dict(cuts=[100], windows=[])
    with pytest.raises(AssertionError):
        assert_no_cut(hard)
    for row in (dict(f_in=180, f_out=190), dict(f_in=210, f_out=220)):
        assert_no_window(row)
    overlap = dict(f_in=189, f_out=191)
    overlaps = AR.timing(overlap)['windows']
    assert len(overlaps) == 1
    assert {k: overlaps[0][k] for k in ('kind', 'frames', 'cut')} == dict(
        kind='dissolve', frames=[190, 210], cut=200)
    with pytest.raises(AssertionError):
        assert_no_window(overlap)


def test_actual_a_captions_do_not_cross_shot_boundaries():
    rows = [r for r in AR.AS.titles.text_table('A') if r['id'] != 'title']

    def assert_no_crossings(captions):
        crossings = {r['id']: AR.timing(r)['cuts'] for r in captions if AR.timing(r)['cuts']}
        assert crossings == {}

    assert_no_crossings(rows)
    cut = AR.AS.EDL.EDL['A'][1]['f0']
    with pytest.raises(AssertionError):
        assert_no_crossings([dict(rows[0], f_in=cut - 1, f_out=cut + 1)])


@pytest.fixture
def synthetic_run(monkeypatch, tmp_path):
    """Run coverage bookkeeping with a 64x32 array; no real init or file lookup."""
    monkeypatch.setattr(AR.BD, 'W', 64)
    monkeypatch.setattr(AR.BD, 'H', 32)
    row = dict(id='T1', line='synthetic test', set='lower', f_in=0, f_out=28)
    line = line_at(20, 10)
    line.id, line.size, line.bands = row['id'], 4, [(0, 2, 0, 4)]
    line.f_in, line.f_out = row['f_in'], row['f_out']
    line._fade = lambda f: (line.alpha, 0.0)
    shot = dict(code='SYNTHETIC', f0=0, f1=100, takes=[dict(stem='old', mode='exact')])
    reads = []

    def picture(f):
        reads.append(f)
        return np.zeros((32, 64, 3), np.float32), shot, 'synthetic', 'synthetic'

    ctx = SimpleNamespace(lines=[line], lines_nt=[line], _ember_on=False, shots=[shot],
                          plans=[dict(kind='take')], shot_at=lambda f: (0, shot), picture=picture)
    monkeypatch.setattr(AR.AS, '_init', lambda *args: None)
    monkeypatch.setattr(AR.AS, '_CTX', ctx)
    monkeypatch.setattr(AR.AS.titles, 'text_table', lambda cut: [row])
    monkeypatch.setattr(AR.AS.titles, 'composite_v3', lambda *args: None)
    monkeypatch.setattr(AR.BD, 'backdrop', lambda *args: (0.0, 0.0))
    monkeypatch.setattr(AR.LR, 'caption_contrast', lambda *args: dict(overall=5.0, worst=5.0))
    monkeypatch.setattr(AR, 'memory', lambda: 0.0)

    def run(delivered, step=1, f_out=28):
        reads.clear()
        row['f_out'] = f_out
        line.f_out = f_out
        monkeypatch.setattr(AR.AS, 'locate',
                            lambda take, cut, variant, f: (f'synthetic-{f}' if f in delivered else None, None))
        output = AR.run(tmp_path / 'synthetic.json', {row['id']}, 'incoming', step=step)
        return output['captions'][0], list(reads)

    return run


def test_incoming_take_gaps_are_reported_and_never_certified(synthetic_run):
    good, reads = synthetic_run({12, 13, 14, 15})
    assert_complete(good)
    assert reads == [12, 13, 14, 15]
    partial, reads = synthetic_run({12, 14, 15})
    assert partial['requested'] == 4
    assert partial['measured'] == 3
    assert partial['missing'] == [13]
    assert not partial['complete']
    assert partial['flags'] == ['PARTIAL-COVERAGE']
    assert reads == [12, 14, 15]
    with pytest.raises(AssertionError):
        assert_complete(partial)
    missing, reads = synthetic_run(set())
    assert missing['measured'] == 0 and missing['worst'] is None
    assert missing['missing'] == [12, 13, 14, 15]
    assert missing['flags'] == ['UNMEASURED', 'PARTIAL-COVERAGE']
    assert reads == []
    with pytest.raises(AssertionError):
        assert_complete(missing)


def test_strided_probe_does_not_certify_an_exhaustive_audit(synthetic_run):
    good, _ = synthetic_run({12, 13, 14, 15})
    assert_complete(good)
    probe, reads = synthetic_run({12, 13, 14, 15}, step=2)
    assert reads == [12, 14]
    assert probe['requested'] == probe['measured'] == 2
    assert probe['missing'] == [] and not probe['complete']
    assert probe['flags'] == ['PROBE-ONLY']
    with pytest.raises(AssertionError):
        assert_complete(probe)


def test_empty_steady_range_is_unmeasured_and_incomplete(synthetic_run):
    good, _ = synthetic_run({12, 13, 14, 15})
    assert_complete(good)
    empty, reads = synthetic_run(set(), f_out=24)
    assert empty['requested'] == empty['measured'] == 0
    assert empty['missing'] == [] and reads == []
    assert empty['flags'] == ['UNMEASURED']
    assert not empty['complete']
    with pytest.raises(AssertionError):
        assert_complete(empty)


@pytest.mark.parametrize('kwargs,message', [
    (dict(step=0), 'step must be >= 1'),
    (dict(step=-1), 'step must be >= 1'),
    (dict(ids={'UNKNOWN'}), 'Unknown caption IDs'),
])
def test_invalid_selection_is_rejected_before_initializing_picture(
        monkeypatch, tmp_path, synthetic_run, kwargs, message):
    good, _ = synthetic_run({12, 13, 14, 15})
    assert_complete(good)

    def forbidden_init(*args):
        raise AssertionError('Invalid audit request initialized the assembler')

    monkeypatch.setattr(AR.AS, '_init', forbidden_init)
    with pytest.raises(ValueError, match=message):
        AR.run(tmp_path / 'invalid.json', **kwargs)
