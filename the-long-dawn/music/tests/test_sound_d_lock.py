"""Final-picture treatment and endpoint gates, using synthetic evidence only."""
from copy import deepcopy

import pytest

import sound_d_binding as B
import sound_d_lock as L
from test_sound_d_measurements import document, point


def bound_case(tmp_path):
    doc = document()
    end = point(tmp_path, 2742, claim='completion', first=2741)
    end.update(landmark='synthetic_fire_hidden', offset_f=0)
    ink = point(tmp_path, 5983, claim='completion', first=5982)
    ink.update(landmark='synthetic_ink_complete', offset_f=1)
    doc['bounds'] = {'D.new.forging.fire': {'end': end}, 'D.new.oldfire.quill': {'end': ink}}
    start = point(tmp_path, 5960, claim='onset', first=5959)
    start.update(landmark='synthetic_caption_start', editorial_reference='old_fire')
    doc['events']['D.new.oldfire.quill'] = start
    return doc


def test_measured_bounds_are_applied_through_table_with_exclusive_offset(tmp_path):
    doc = bound_case(tmp_path)
    plan = B.build(measurements=doc, frame_root=tmp_path)
    forge = next(r for r in plan['new_beds'] if r['request_id'] == 'D.new.forging.fire')
    ink = next(r for r in plan['new_events'] if r['request_id'] == 'D.new.oldfire.quill')
    assert forge['f1'] == 2742 and forge['end_offset_f'] == 0
    assert ink['hit_f'] == 5960 and ink['stop_f'] == 5984
    assert ink['end_provenance']['frame'] == 5983
    assert B.problems(plan, measurements=doc, frame_root=tmp_path) == []
    ink['stop_f'] += 1
    assert B.problems(plan, measurements=doc, frame_root=tmp_path)


@pytest.mark.parametrize('mutation', ['offset', 'claim', 'missing_prior', 'unknown', 'inverted', 'fake_hash', 'wrong_side'])
def test_bounds_reject_false_or_unsupported_endpoints(tmp_path, mutation):
    doc = bound_case(tmp_path)
    end = doc['bounds']['D.new.oldfire.quill']['end']
    if mutation == 'offset': end['offset_f'] = 8
    elif mutation == 'claim': end['measurement']['claim'] = 'contact'
    elif mutation == 'missing_prior': end['measurement']['frames'].pop(0)
    elif mutation == 'unknown': doc['bounds']['unknown'] = doc['bounds'].pop('D.new.oldfire.quill')
    elif mutation == 'inverted': doc['bounds']['D.new.oldfire.quill'] = doc['bounds']['D.new.forging.fire']
    elif mutation == 'fake_hash': end['measurement']['frames'][0]['sha256'] = '0'*64
    elif mutation == 'wrong_side': doc['bounds']['D.new.oldfire.quill']['start'] = deepcopy(end)
    with pytest.raises(ValueError): B.build(measurements=doc, frame_root=tmp_path)


def test_locked_treatment_has_continuous_fire_and_requires_retired_page_turn():
    events = [dict(id='ink', request_id='D.new.oldfire.quill', hit_f=5960, stop_f=5984,
                   source='measured', end_provenance={'synthetic': True}, recipe_overrides={})]
    beds = [dict(id='fire', request_id='D.new.forging.fire', f0=2080, f1=2742,
                 end_provenance={'synthetic': True})]
    overlay = dict(bounds={'D.new.forging.fire': {}, 'D.new.oldfire.quill': {}},
                   suppressions={'D.new.page.to_blank': {}})
    result = L.apply(events, beds, overlay)
    assert result['caption_interval'] == [5960, 5984]
    assert beds[0]['f0'] < 2400 < beds[0]['f1']
    assert beds[0]['continuity_provenance']['restart_at_2400'] is False
    assert events[0]['recipe_overrides']['post'] == 1
    with pytest.raises(ValueError, match='page-turn suppression'):
        L.apply(events, beds, dict(overlay, suppressions={}))
    with pytest.raises(ValueError, match='measured end bound'):
        L.apply(events, beds, dict(overlay, bounds={}))
    with pytest.raises(ValueError, match='locked picture requires'):
        B.build(picture_revision='locked')
    events[0]['source'] = 'est.'
    with pytest.raises(ValueError, match='measured ink onset'):
        L.apply(events, beds, overlay)


def test_end_bound_alone_counts_as_picture_measurement(tmp_path):
    doc = bound_case(tmp_path)
    doc['events'] = {}
    doc['bounds'].pop('D.new.oldfire.quill')
    assert B.build(measurements=doc, frame_root=tmp_path)['constraints']['no_new_picture_measurement'] is False
    doc['bounds'] = {}
    assert B.build(measurements=doc, frame_root=tmp_path)['constraints']['no_new_picture_measurement'] is True
