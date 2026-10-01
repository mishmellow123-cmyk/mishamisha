"""Authored bridge constraints; synthetic rows do not assert picture measurements."""
from copy import deepcopy

import pytest

import sound_d_binding as B
import sound_d_bridges as D


def case():
    plan = B.build()
    events, beds, reuse = plan['new_events'], plan['new_beds'], plan['reuse_plan']
    events.append(dict(id=D.TRANSITION_REQUEST['id'], request_id=D.TRANSITION_REQUEST['id'],
                       hit_f=6083, stop_f=6109, source='measured', end_provenance={'test_only':True}))
    # Production native trap series is not needed to test the authored gain curve.
    edl = B._load('edl', None)
    return events, beds, reuse, {}, edl


def test_bridge_opt_in_preserves_default_binding_and_requires_lock():
    assert B.build() == B.build(transition_pass=False)
    for flag in (True, 'false', 1):
        with pytest.raises(ValueError, match='transition pass'):
            B.build(transition_pass=flag)
    assert not any(r['id']==D.TRANSITION_REQUEST['id'] for r in B.build()['new_events'])


def test_authored_bridges_keep_markers_and_same_clock_donor_handles():
    events, beds, reuse, overlay, edl = case()
    before = {r['id']:r['hit_f'] for r in events+reuse['EXTRA_EVENTS']}
    result = D.apply(events, beds, reuse, overlay, edl)
    assert before == {r['id']:r['hit_f'] for r in events+reuse['EXTRA_EVENTS']}
    rows = {r['id']:r for r in beds+reuse['BED_CROPS']}
    assert rows['D.new.glow.wind']['source_span'] == rows['D.new.ridges.wind']['source_span'] == [4068,5180]
    assert rows['D.new.glow.wind']['source_seed'] == rows['D.new.ridges.wind']['source_seed']
    assert rows['D.26.A.wind.crossing']['donor_crop'] == [5168,5580]
    assert rows['D.26.A.wind.crossing']['f0'] == 6628
    assert rows['D.new.instep.fire']['f1'] == 5852
    assert rows['D.new.oldfire.hearth']['f0'] == 5828
    assert rows['D.new.oldfire.hearth']['f1'] == 6109
    assert rows['D.new.lamps.forge']['f0'] == 6083
    assert all(r['authored'] for r in result['changes'])
    assert all(r['f0']>=1440 for r in beds+reuse['BED_CROPS'])


@pytest.mark.parametrize('mutation', ['window','center','crossing','unmeasured','no_end','outside'])
def test_bridges_reject_stale_edit_or_missing_burn_evidence(mutation):
    events, beds, reuse, overlay, edl = case()
    if mutation == 'window': next(t for t in edl['transitions'] if t.get('cut')==4080)['f0']+=1
    elif mutation == 'center': next(t for t in edl['transitions'] if t.get('cut')==6080)['center'][0]+=1
    elif mutation == 'crossing': edl['transitions'].append(dict(cut=6640))
    elif mutation == 'unmeasured': events[-1]['source']='est.'
    elif mutation == 'no_end': events[-1].pop('end_provenance')
    else: events[-1]['stop_f']=6121
    with pytest.raises(ValueError): D.apply(events,beds,reuse,overlay,edl)
