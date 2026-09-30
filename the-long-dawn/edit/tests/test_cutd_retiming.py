"""Retimed sources must agree across decoded composites, coverage, provenance and cache keys."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS
import caption_grade as CG
import deliver as DV
import edl_v3 as EDL


def test_hold_selects_one_source_frame_for_colour_matte_add_and_metadata(monkeypatch):
    folders = {stem: {17: f'{stem}/17.jpg'} for stem in ('plate', 'matte', 'add', 'under')}
    monkeypatch.setattr(AS, 'index', lambda path: folders.get(Path(path).name, {}))
    monkeypatch.setattr(DV, '_stat', lambda path: path)
    take = EDL.T('plate', -100, 'exact', matte='matte', add='add', under=('hold', 'under', 17), hold=17)
    shot = EDL.S('fixture', 100, 120, 'test', 'test', 'EDIT', '', takes=[take])
    plan = AS.plan_shot(shot, 'D', None)
    assert plan['have'] == 20
    ctx = object.__new__(AS.Ctx)
    ctx.cut, ctx.variant = 'D', None
    reads = []

    def read(path, crop=None, gray=False):
        reads.append(path)
        return np.full((2, 2) if gray else (2, 2, 3), 0.2, np.float32)

    ctx.read = read
    for frame in (100, 119):
        reads.clear()
        ctx.take_frame(take, frame)
        assert reads == ['plate/17.jpg', 'matte/17.jpg', 'under/17.jpg', 'add/17.jpg']
        assert set(DV.frame_sources('D', None, plan, frame)) == set(reads)
    # Removing the fixed coordinate leaves only one available frame, exposing the broken hold.
    moving = dict(take)
    moving.pop('hold')
    assert AS.plan_shot(dict(shot, takes=[moving]), 'D', None)['have'] == 1
    del folders['add'][17]
    assert AS.plan_shot(shot, 'D', None)['kind'] == 'slate'


def test_hold_is_absent_by_default_and_changes_the_segment_identity(monkeypatch):
    assert 'hold' not in EDL.T('plate')
    monkeypatch.setattr(DV, 'frame_sources', lambda *a: [])
    monkeypatch.setattr(AS, 'transition_at', lambda *a: None)
    take = EDL.T('plate', mode='exact', hold=17)
    shot = EDL.S('fixture', 100, 102, 'test', 'test', 'EDIT', '', takes=[take])
    code = dict.fromkeys(('frame', 'text', 'slate', 'x2', 'title'), 'fixture')
    def key():
        plan = dict(kind='take', take=take, have=2, alt=False)
        return DV.segment_key('D', None, DV.PROFILES['animatic'], 0, shot, plan, code, [])
    before = key()
    take['hold'] = 18
    assert key() != before


def test_shifted_burn_reads_original_layer_numbers_and_missing_layer_disables_it(monkeypatch):
    folders = {stem: {2310: stem + '/2310.png', 2345: stem + '/2345.png'} for stem in ('glow', 'keep', 'cover')}
    monkeypatch.setattr(AS, 'index', lambda path: folders.get(Path(path).name, {}))
    spec = dict(glow='glow', keep='keep', cover='cover', layer_off=-1440)
    assert AS.transition_layers(spec, 3750) == {k: k + '/2310.png' for k in folders}
    assert AS.transition_layers(spec, 3785) == {k: k + '/2345.png' for k in folders}
    assert AS.transition_layers(dict(spec, layer_off=0), 3750) is None
    del folders['cover'][2345]
    assert AS.transition_layers(spec, 3785) is None


@pytest.mark.parametrize('ident,delta', [('R02', 1120), ('title', 3200)])
def test_d_baked_grade_matches_same_physical_c_plate_and_rejects_wrong_remap(ident, delta):
    c = next(t for t in EDL.TRANS['C'] if t.get('id') == ident)
    d = next(t for t in EDL.TRANS['D'] if t.get('id') == ident)
    image = np.full((201, 480, 3), 0.4, np.float32)
    frame = c['full0']
    ct = EDL.T(c['source_stem'], c['source_off'], 'exact')
    dt = EDL.T(d['source_stem'], d['source_off'], 'exact')
    np.testing.assert_array_equal(CG.apply(image, frame, c, ct), CG.apply(image, frame + delta, d, dt))
    bad = deepcopy(d)
    bad.pop('band_frame_off')
    with pytest.raises(ValueError, match='mapping mismatch'):
        CG.apply(image, frame + delta, bad, dt)
    with pytest.raises(ValueError, match='mapping mismatch'):
        CG.apply(image, frame + delta, d, dict(dt, hold=EDL.source_frame(dt, frame + delta)))
    for invalid in (dict(dt, clamp=(400, 400)), dict(dt, screen_transform={'target0': (990, 90)})):
        with pytest.raises(ValueError, match='mapping mismatch'):
            CG.apply(image, frame + delta, d, invalid)
    with pytest.raises(ValueError, match='must be an integer'):
        CG.apply(image, frame + delta, dict(d, band_frame_off=0.5), dt)


@pytest.mark.parametrize('cut_frame,slated_side', [(3760, 'incoming'), (6400, 'outgoing'), (8640, 'outgoing')])
def test_d_pair_windows_remain_plain_cuts_when_a_neighbour_is_new(cut_frame, slated_side, monkeypatch):
    from types import SimpleNamespace
    transition = next(t for t in EDL.TRANS['D'] if t.get('cut') == cut_frame)
    monkeypatch.setattr(AS, 'transition_at', lambda cut, frame: transition)
    monkeypatch.setattr(AS, 'transition_layers', lambda *args: {})
    def picture(frame):
        incoming = frame >= cut_frame
        slated = incoming if slated_side == 'incoming' else not incoming
        return (np.full((3, 4, 3), 0.2 if incoming else 0.7, np.float32), {},
                'SLATE' if slated else 'source', None if slated else {'stem': 'fixture'})
    def forbidden(*args):
        raise AssertionError('A slate neighbour must suppress the pair compositor')
    monkeypatch.setitem(AS.TKINDS_PAIR, transition['kind'], forbidden)
    monkeypatch.setattr(AS.Ctx, 'picture', lambda self, frame: picture(frame))
    dispatch = AS._transitions(SimpleNamespace(cut='D', picture=picture), False)
    for frame in (cut_frame-1, cut_frame, cut_frame+1):
        expected, actual = picture(frame), dispatch(frame)
        np.testing.assert_array_equal(expected[0], actual[0])
        assert expected[2:] == actual[2:]
