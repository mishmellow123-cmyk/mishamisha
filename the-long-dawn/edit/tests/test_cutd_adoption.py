"""Explicit D adoption, complete source gates, baked-caption routing and filename-only inventory."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS
import cutd_adoption as A
import deliver as DV
import edl_v3 as EDL
import titles
from cutd_test_fixtures import adoption_lines, commented_source, edl_namespace


def assert_catalogue(catalog):
    assert len(catalog) == 17
    assert len({row[1] for row in catalog}) == 17
    by_stem = {stem: (first, last) for _, stem, first, last in catalog}
    assert by_stem['embers_D_inscription'] == (1680, 2079), 'inscription gate includes all400 frames'
    assert by_stem['embers_D_holdout'] == (5480, 5519), 'holdout uses the committed40-frame insert'


def test_every_new_plate_has_one_literal_explicit_adoption_line():
    assert_catalogue(A.NEW_PLATES)
    calls = adoption_lines()
    assert set(calls) == {code for code, *_ in A.NEW_PLATES + A.OPTIONAL_PLATES}
    for code, stem, first, last in A.NEW_PLATES + A.OPTIONAL_PLATES:
        take, = calls[code]
        assert (take['stem'], take['off'], take['mode'], take['need']) == (stem, 0, 'exact', (first, last))
    assert calls['D23'][0]['baked_text'] == ('D14',)
    assert 'baked_text' not in EDL.T('legacy')
    expected = {'D08', 'D09', 'D10', 'D12', 'D13', 'D14', 'D16', 'D17', 'D18',
                'D19', 'D21b', 'D22', 'D23', 'D24', 'D27', 'D28', 'D31'}
    assert set(EDL.D_NEW_TAKES) == expected
    assert EDL.D_NEW_TAKES == {code: calls[code] for code in expected}
    title = next(s for s in EDL.D if s['code'] == 'D33')
    assert title['takes'] == [EDL.book('C title source6960-7199; caption source6980-7159', off=-1920)]
    for bad_span in ((1760, 2079), (1680, 1759)):
        broken = list(A.NEW_PLATES)
        broken[0] = (*broken[0][:2], *bad_span)
        with pytest.raises(AssertionError, match='all400'):
            assert_catalogue(broken)


def test_uncommenting_one_inscription_line_routes_both_rows_without_changing_the_standin():
    live = (EDIT / 'edl_v3.py').read_bytes()
    source = commented_source()
    before = "    # 'D08': [T('embers_D_inscription', 0, 'exact', need=(1680, 2079))],"
    assert source.count(before) == 1
    unadopted = edl_namespace()
    assert not unadopted['D_NEW_TAKES']
    assert next(s for s in unadopted['D'] if s['code'] == 'D07')['takes'] == [EDL.D_BURN_PLATE]
    assert next(s for s in unadopted['D'] if s['code'] == 'D08')['takes'] == []
    namespace = edl_namespace(adopt=('D08',))
    rows = {s['code']: s for s in namespace['D']}
    assert rows['D07']['takes'] == rows['D08']['takes'] == namespace['D_NEW_TAKES']['D08']
    assert rows['D07']['takes'][0]['need'] == (1680, 2079)
    assert namespace['D_BURN_PLATE'] == EDL.D_BURN_PLATE
    assert namespace['A'] == EDL.A and namespace['B'] == EDL.B and namespace['C'] == EDL.C
    broken = "    'D08': [T('first', 0, 'exact'), T('fallback', 0, 'exact')],"
    with pytest.raises(ValueError, match='exactly one explicit take'):
        exec(compile(source.replace(before, broken), '<invalid adoption>', 'exec'),
             {'__file__': str(EDIT / 'edl_v3.py'), '__name__': 'cutd_invalid_adoption'})
    assert (EDIT / 'edl_v3.py').read_bytes() == live


def test_inscription_gate_requires_burn_preroll_and_body_together(monkeypatch):
    take = adoption_lines()['D08'][0]
    present = set(range(1760, 2080))
    monkeypatch.setattr(AS, 'chain', lambda *args: ['fixture'])
    monkeypatch.setattr(AS, 'index', lambda folder: {f: f'f_{f:05d}.png' for f in present})
    burn = EDL.S('D1', 1680, 1760, 'D07', 'burn', 'fixture', '', [take])
    body = EDL.S('D1', 1760, 2080, 'D08', 'body', 'fixture', '', [take])
    assert AS.plan_shot(body, 'D', None)['kind'] == 'slate'
    assert AS.plan_shot(burn, 'D', None)['kind'] == 'slate'
    present.update(range(1680, 1760))
    assert all(AS.plan_shot(s, 'D', None)['kind'] == 'take' for s in (burn, body))
    present.remove(2079)
    assert all(AS.plan_shot(s, 'D', None)['kind'] == 'slate' for s in (burn, body))
    # Negative control: a body-only gate would incorrectly pass the unrendered pre-roll.
    body_only = deepcopy(body)
    body_only['takes'][0]['need'] = (1760, 2079)
    present.add(2079)
    present.remove(1680)
    assert AS.plan_shot(body_only, 'D', None)['kind'] == 'take'
    assert AS.plan_shot(body, 'D', None)['kind'] == 'slate'


def test_book_adoption_changes_caption_set_only_after_complete_selection(monkeypatch):
    row = deepcopy(next(s for s in edl_namespace()['D'] if s['code'] == 'D23'))
    monkeypatch.setattr(EDL, 'D', [row])
    present = set(range(5840, 6080))
    monkeypatch.setattr(AS, 'chain', lambda *args: ['fixture'])
    monkeypatch.setattr(AS, 'index', lambda folder: {f: f'f_{f:05d}.png' for f in present})
    caption = lambda: next(r for r in titles.text_table('D') if r['id'] == 'D14')
    baseline = caption()
    assert baseline['set'] == 'lower'  # all files present does not declare an adoption
    row['takes'] = adoption_lines()['D23']
    baked = caption()
    assert baked == dict(baseline, set='in_picture')
    assert len(titles.text_table('D')) == 21
    assert A.baked_caption_ids(titles.D_TEXT) == {'D14'}
    assert 'D14' not in {line.id for line in titles.lines_v3('D', .05)}
    present.remove(6079)
    assert caption() == baseline
    assert 'D14' in {line.id for line in titles.lines_v3('D', .05)}
    present.add(6079)
    row['takes'][0].pop('baked_text')
    assert caption() == baseline  # a complete plate without the baked-text declaration keeps the overlay
    row['takes'][0]['baked_text'] = ('D14',)
    row['f1'] = 6000
    assert caption() == baseline  # a declaration cannot suppress words outside its selected row


def test_partial_selected_take_cannot_hide_a_caption():
    take = EDL.T('book_D_oldfire', 0, 'exact', baked_text=('D14',))
    shot = EDL.S('D5', 5840, 6080, 'D23', 'oldfire', 'fixture', '', [take])
    def planner(row, cut, variant):
        return dict(kind='take', take=take, have=239)
    assert A.baked_caption_ids(titles.D_TEXT, [shot], planner) == set()
    def complete(row, cut, variant):
        return dict(kind='take', take=take, have=240)
    assert A.baked_caption_ids(titles.D_TEXT, [shot], complete) == {'D14'}


def test_baked_caption_set_invalidates_the_delivery_segment(monkeypatch):
    shot = deepcopy(next(s for s in EDL.D if s['code'] == 'D23'))
    take = adoption_lines()['D23'][0]
    shot['takes'] = [take]
    plan = dict(kind='take', take=take, have=240, alt=0)
    monkeypatch.setattr(EDL, 'D', [shot])
    monkeypatch.setattr(AS, 'plan_shot', lambda *args: plan)
    monkeypatch.setattr(DV, 'frame_sources', lambda *args: [])
    monkeypatch.setattr(AS, 'transition_at', lambda *args: None)
    code = {k: 'fixture' for k in ('frame', 'text', 'slate', 'x2', 'title')}
    def key():
        return DV.segment_key('D', None, DV.PROFILES['master'], 0, shot, plan, code, titles.text_table('D'))
    baked = key()
    take.pop('baked_text')
    overlay = key()
    assert baked != overlay
    take['baked_text'] = ('D14',)
    assert key() == baked


def test_abc_tables_do_not_consult_d_adoption(monkeypatch):
    expected = {cut: titles.text_table(cut) for cut in 'ABC'}
    monkeypatch.setattr(A, 'baked_caption_ids', lambda *args: pytest.fail('ABC consulted D adoption'))
    assert {cut: titles.text_table(cut) for cut in 'ABC'} == expected


def test_inventory_counts_unique_accepted_in_range_frames_without_adopting(tmp_path, monkeypatch):
    unadopted = edl_namespace()
    monkeypatch.setattr(EDL, 'D', unadopted['D'])
    folder = tmp_path / 'embers_D_holdout'
    folder.mkdir()
    for f in range(5480, 5520):
        (folder / f'f_{f:05d}.jpg').touch()
    # A duplicate extension is one frame; PNG has the assembler's preference.
    Image.new('RGB', (8, 4)).save(folder / 'f_05480.png')
    for name in ('f_05480.jpeg', 'f_05520.png', 'f_05479.png', 'f_05481.PNG', 'notes.png'):
        (folder / name).touch()
    monkeypatch.setattr(AS, '_INDEX', {})
    rows = A.availability(tmp_path, dimensions=True)
    assert len(rows) == 19
    holdout = next(r for r in rows if r['stem'] == folder.name)
    assert (holdout['needed'], holdout['available'], holdout['missing']) == (40, 40, 0)
    assert holdout['complete'] and holdout['adopted_rows'] == []
    assert holdout['dimensions_sample'] == dict(frame=5480, width=8, height=4)
    assert next(r for r in rows if r['stem'] == 'embers_D_inscription')['needed'] == 400
    (folder / 'f_05519.jpg').unlink()
    (folder / 'f_05519.jpeg').touch()  # ignored extension must not rescue the missing frame
    holdout = next(r for r in A.availability(tmp_path) if r['stem'] == folder.name)
    assert (holdout['available'], holdout['missing'], holdout['complete']) == (39, 1, False)
    assert len(list(folder.iterdir())) > 40  # file count alone would falsely clear the gate
    generated = next(r for r in rows if r['kind'] == 'generated')
    assert generated['stem'] == 'x1burn_D_drawnring' and generated['available'] is None
    assert generated['missing'] is None and generated['needed'] == 80
    assert next(r for r in rows if r['stem'] == 'book_D_title')['optional']


def test_cli_reports_missing_frames_as_inventory_without_writes(tmp_path, capsys):
    rows = A.main(['--renders', str(tmp_path), '--json'])
    assert json.loads(capsys.readouterr().out) == rows
    assert all(r['available'] == 0 and r['missing'] == r['needed'] for r in rows if r['kind'] == 'plate')
    assert list(tmp_path.iterdir()) == []
