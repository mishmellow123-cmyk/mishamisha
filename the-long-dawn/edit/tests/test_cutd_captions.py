"""The locked D treatment's caption words, timing, and opt-in lower-third rendering.

Only glyph masks and small synthetic pictures are drawn; no production renders are read.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS
import edl_v3 as EDL
import titles as T
from cutd_test_fixtures import edl_namespace


WORDS = [
    'At first it looked like morning.',
    'Our Ring was kindled from every tale we had ever told.',
    'It read every word we had ever written, and learned to answer.',
    'It could show us anything, except itself.',
    'In the old story, a Dark Lord forges a Ring to rule the world.',
    'In our story, there was no Dark Lord, only smiths in every kingdom, each racing to finish it first.',
    'They dug deeper every year, for the gold ran deeper still.',
    'But whoever won the race, the Ring would rule us all.',
    'In the old story, the wise refuse the Ring.',
    'In ours, no smith could refuse it alone.',
    'So the two furthest ahead lit the first beacons, together.',
    'Each fire said: I will wait, if you will.',
    'When the last beacon caught, every forge fell into step.',
    'In the old story, the Ring is unmade in the fire that forged it.',
    'Ours was left unfinished, and they lit lamps to read it by.',
    'It took longer than anyone wanted.',
    'And the beacons burned on, so no forge could be lit in secret.',
    'Even unfinished, the Ring brought the dawn.',
    'Perhaps one day they would read it well enough to finish it. Perhaps never.',
    'The last pages were left for us.',
]
RANGES = [(152, 320), (684, 932), (1028, 1256), (1330, 1438), (1520, 1660), (2104, 2376), (2772, 2948), (3224, 3416), (3544, 3740), (3798, 4040), (4332, 4536), (4584, 4824), (5612, 5816), (5864, 6056), (6104, 6376), (6664, 6880), (7064, 7336), (7384, 7656), (8344, 8616), (8664, 8840)]


def narration():
    return [row for row in T.text_table('D') if row['id'] != 'title']


def assert_treatment(rows):
    assert [row['id'] for row in rows] == [f'D{i:02d}' for i in range(1, 21)]
    assert [row['line'] for row in rows] == WORDS
    assert [(row['f_in'], row['f_out']) for row in rows] == RANGES
    assert all(row['f_out'] - row['f_in'] >= 108 for row in rows)


def test_twenty_exact_treatment_captions_and_reading_windows():
    rows = narration()
    assert_treatment(rows)
    for change in ('wording', 'timing', 'count'):
        broken = deepcopy(rows)
        if change == 'wording':
            broken[5]['line'] = broken[5]['line'].replace('Lord, only', 'Lord. Only')
        elif change == 'timing':
            broken[12]['f_out'] = broken[12]['f_in'] + 107
        else:
            broken.pop()
        with pytest.raises(AssertionError):
            assert_treatment(broken)


def test_caption_entrances_follow_shots_and_measured_catches():
    rows = {row['row']: row for row in narration()}
    # Settles: opening128, glyph white-fall672, point1016, vision1318, Deep2758,
    # refusal burn3786, crown ignition4320, map catch5600, crossing cut6640, last-leaf dissolve8652.
    settled = {1: 128, 2: 672, 3: 1016, 4: 1318, 6: 2080, 7: 2758, 8: 3200, 9: 3520,
               10: 3786, 11: 4320, 12: 4560, 13: 5600, 14: 5840, 15: 6080, 16: 6640,
               17: 7040, 18: 7360, 19: 8320, 20: 8652}
    assert all(12 <= rows[number]['f_in'] - frame <= 24 for number, frame in settled.items())
    assert rows[13]['f_in'] < 5680 < rows[13]['f_out'] <= 5840


def test_old_story_baked_caption_maps_to_the_existing_source_window():
    row = next(row for row in narration() if row['row'] == 5)
    assert row['set'] == 'in_picture'
    shot = next(shot for shot in EDL.D if shot['f0'] <= row['f_in'] < shot['f1'])
    take = next(take for take in shot['takes'] if take['stem'] == 'cand_t1_current-words-held')
    baked = next(item for item in T.BAKED_TEXT if item['stem'] == take['stem'])
    assert take['off'] == -1120
    assert (row['f_in'] + take['off'], row['f_out'] + take['off']) == baked['src']
    assert row['line'] == baked['line']
    drawn = {line.id for line in T.lines_v3('D', .125)}
    assert row['id'] not in drawn
    old_fire = next(shot for shot in EDL.D if shot['code'] == 'D23')
    plan = AS.plan_shot(old_fire, 'D', None)
    assert plan['kind'] == 'take' and plan['have'] == 240
    assert plan['take']['stem'] == 'book_D_oldfire' and plan['take']['baked_text'] == ('D14',)
    assert 'D14' not in drawn  # the owner's complete D23 adoption now supplies these words in the page
    assert next(row for row in narration() if row['row'] == 14)['set'] == 'in_picture'


def test_multiline_rows_preserve_exact_case_and_punctuation():
    rows = {row['row']: row for row in narration()}
    assert rows[6]['lines'] == ('In our story, there was no Dark Lord,',
                               'only smiths in every kingdom, each racing to finish it first.')
    assert rows[19]['lines'] == ('Perhaps one day they would read it well enough to finish it.',
                                'Perhaps never.')
    for row in rows.values():
        if row.get('lines'):
            assert ' '.join(row['lines']) == row['line']
            rendered = T.TextV3('D', row, .25)
            assert len(rendered.bands) == 2


def test_all_d_narration_glyphs_fit_in_the_lower_third():
    for row in narration():
        if row['set'] == 'in_picture':
            continue
        line = T.TextV3('D', row, 1.)
        assert line.font == T.ITALIC
        assert line.y == T.Y_LOWER
        gy, gx = np.nonzero(line.alpha > .5)
        assert line.x0 + gx.min() >= 64, row['id']
        assert line.x0 + gx.max() < T.W - 64, row['id']
        assert line.y0 + gy.min() >= T.H * 2 / 3, row['id']
        assert line.y0 + gy.max() < T.H - 48, row['id']


@pytest.mark.parametrize('number', [1, 6, 14, 19])
def test_d_lower_rows_use_twelve_frame_fades_and_not_fire(number, monkeypatch):
    # D14 still needs this lower-third fallback whenever its complete baked plate is not adopted.
    monkeypatch.setattr(EDL, 'D', edl_namespace()['D'])
    row = narration()[number - 1]
    assert row['set'] == 'lower'
    d, a = T.TextV3('D', row, .25), T.TextV3('A', row, .25)
    monkeypatch.setattr(d, '_fire', lambda *args: pytest.fail('D narration kindled as fire'))
    monkeypatch.setattr(d, '_ink', lambda *args: pytest.fail('D narration used a pen wipe'))
    plate = np.full((201, 480, 3), .08, np.float32)
    for frame in [row['f_in'], row['f_in'] + 6, row['f_in'] + 12, row['f_out'] - 12, row['f_out'] - 1]:
        np.testing.assert_array_equal(d.draw(plate.copy(), frame), a.draw(plate.copy(), frame))
    np.testing.assert_array_equal(d._fade(row['f_in'] + 12)[0], d.alpha)
    np.testing.assert_array_equal(d._fade(row['f_out'] - 12)[0], d.alpha)
    np.testing.assert_array_equal(d.draw(plate.copy(), row['f_in']), plate)
    np.testing.assert_array_equal(d.draw(plate.copy(), row['f_out']), plate)
    assert not np.array_equal(d.draw(plate.copy(), row['f_in'] + 6), plate)
    # The retired D scaffold's C-style fire route is a concrete wrong-style control.
    wrong = T.TextV3('C', row, .25).draw(plate.copy(), row['f_in'] + 6)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(wrong, d.draw(plate.copy(), row['f_in'] + 6))


def test_d_caption_is_composited_over_a_new_shot_slate(monkeypatch):
    monkeypatch.setattr(EDL, 'D', edl_namespace()['D'])
    row = narration()[13]
    assert row['set'] == 'lower'
    ctx = AS.Ctx.__new__(AS.Ctx)
    ctx.lines = T.lines_v3('D', .125)
    assert 'D14' in {line.id for line in ctx.lines}
    ctx._ember_on, ctx.clean = False, True
    plate = np.zeros((100, 240, 3), np.float32)
    ctx.picture = lambda frame: (plate.copy(), {}, 'SLATE · NEW old fire', None)
    image = ctx.frame(row['f_in'] + 12)
    assert np.count_nonzero(image) > 0
    ctx.lines = []
    assert not np.count_nonzero(ctx.frame(row['f_in'] + 12))


def test_d_table_changes_cannot_mutate_the_c_alternative(monkeypatch):
    before = deepcopy(T.C5_TEXT)
    changed = deepcopy(T.D_TEXT)
    changed[4]['line'] = 'Synthetic independent edit.'
    changed[4]['lines'] = ('Synthetic independent edit.',)
    monkeypatch.setattr(T, 'D_TEXT', changed)
    assert T.C5_TEXT == before
    assert T.text_table('C') != T.text_table('D')


def test_one_caption_string_drives_both_export_and_two_row_layout():
    import ast
    tree = ast.parse((EDIT / 'titles.py').read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'D_TEXT' for t in n.targets))
    calls = assignment.value.elts[:-1]
    assert len(calls) == 20
    assert all(isinstance(n, ast.Call) and n.func.id == '_d_caption' and n.lineno == n.end_lineno for n in calls)
    changed = T._d_caption(6, 2104, 2376, 'First new row.\nSecond new row.')
    assert changed['line'] == 'First new row. Second new row.'
    assert changed['lines'] == ('First new row.', 'Second new row.')
