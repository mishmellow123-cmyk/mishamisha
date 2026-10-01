"""Synthetic pixel controls; no test fixture is production picture evidence."""
from copy import deepcopy
import numpy as np
import pytest

import measure_sound_d_transitions as T


def rows():
    return [dict(frame=f,incoming_contribution={str(t):int(f>=6083) for t in T.LEVELS},
        outgoing_contribution={str(t):int(f<6109) for t in T.LEVELS},
        caption_change={str(t):int(f>=6094) for t in T.LEVELS}) for f in range(6080,6120)]


def test_source_contribution_thresholds_measure_both_edges():
    data=rows()
    assert {r['first_positive'] for r in T.boundaries(data,'incoming_contribution').values()}=={6083}
    assert {r['first_trailing_zero'] for r in T.boundaries(data,'outgoing_contribution').values()}=={6109}
    assert {r['first_positive'] for r in T.boundaries(data,'caption_change').values()}=={6094}
    a=np.zeros((3,4,3),np.uint8); b=a.copy();b[1,2]=[4,0,0]
    assert T.counts(a,b)=={'1':1,'4':1,'8':0}


@pytest.mark.parametrize('mutation',['missing','duplicate','disordered','outside'])
def test_no_landmark_without_whole_transition_denominator(mutation):
    data=rows()
    if mutation=='missing':data.pop(5)
    elif mutation=='duplicate':data[2]=deepcopy(data[1])
    elif mutation=='disordered':data[1],data[2]=data[2],data[1]
    else:data[-1]['frame']=6120
    with pytest.raises(ValueError,match='coverage'):T.boundaries(data,'incoming_contribution')


def result():
    data=rows()
    return dict(input_sha256={},method='Synthetic test only',dependencies=[{'synthetic':True}],
        edl_dependencies={'synthetic':True},
        retained_frames=[dict(frame=f,synthetic=True) for f in range(6080,6120)],
        burn={k:T.boundaries(data,k) for k in ('incoming_contribution','outgoing_contribution','caption_change')})


def test_overlay_preserves_observed_scope_and_authored_end_convention():
    r=result();out=T.overlay(r)
    onset=out['events'][T.EVENT]; end=out['bounds'][T.EVENT]['end']
    assert (onset['frame'],end['frame'],end['offset_f'])==(6083,6109,0)
    assert onset['measurement_scope']=='native_pre_finish_composite'
    assert onset['measurement']['claim']=='onset' and end['measurement']['claim']=='completion'
    onset['measurement']['dependencies'][0]['synthetic']=False
    assert r['dependencies'][0]['synthetic'] is True


@pytest.mark.parametrize('mutation',['disagree_onset','disagree_clear','no_onset','no_clear','missing_bracket'])
def test_overlay_refuses_uncertain_boundary(mutation):
    r=result()
    if mutation=='disagree_onset':r['burn']['incoming_contribution']['4']['first_positive']+=1
    elif mutation=='disagree_clear':r['burn']['outgoing_contribution']['4']['first_trailing_zero']+=1
    elif mutation=='no_onset':r['burn']['incoming_contribution']['4']['first_positive']=None
    elif mutation=='no_clear':r['burn']['outgoing_contribution']['4']['first_trailing_zero']=None
    else:r['retained_frames']=[x for x in r['retained_frames'] if x['frame']!=6082]
    with pytest.raises(ValueError):T.overlay(r)


def test_boundary_censored_positive_is_not_new_onset():
    data=rows()
    for row in data:row['incoming_contribution']={str(t):1 for t in T.LEVELS}
    assert all(x['first_positive'] is None for x in T.boundaries(data,'incoming_contribution').values())


def test_selected_construction_includes_held_outgoing_shot():
    edl=dict(shots=[dict(code='old',f0=5840,f1=6080),dict(code='new',f0=6080,f1=6400)],
        transitions=[dict(cut=6080,f0=6080,f1=6120,kind='ring_burn')])
    assert {r['code'] for r in T.selected(edl,6082,6085)['shots']}=={'old','new'}
    assert T.selected(edl,6082,6085)['transitions']==edl['transitions']


def test_counterfactuals_share_the_actual_field_and_exclude_added_light():
    outgoing=np.full((2,3,3),.4,np.float32)
    incoming=np.full_like(outgoing,.2)
    glow=np.full_like(outgoing,.1)
    keep=np.full_like(outgoing,.5)
    cover=np.full((2,3),.25,np.float32)
    calls=[]
    def layers(*args):calls.append(args);return glow,keep,cover
    spec={'f0':6080}
    real,no_new,no_old=T.counterfactuals(outgoing,incoming,6083,spec,layers)
    assert len(calls)==1
    np.testing.assert_array_equal(real,outgoing*keep+incoming*(1-cover[...,None])+glow)
    np.testing.assert_allclose(real-no_new,incoming*(1-cover[...,None]),atol=1e-7)
    np.testing.assert_allclose(real-no_old,outgoing*keep,atol=1e-7)
    a,b,c=T.counterfactuals(outgoing,incoming,6080,spec,layers)
    assert len(calls)==1 and np.array_equal(a,outgoing) and np.array_equal(b,outgoing) and not c.any()
    a[:]=0
    assert outgoing.any()
