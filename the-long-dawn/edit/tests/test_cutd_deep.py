"""D-only independent filmed coefficients, explicit held under and retimed exit; synthetic file fixtures."""
from copy import deepcopy
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import assemble as AS
import cutd_deep as DEEP
import deliver as DV
import edl_v3 as EDL


def png(root,stem,f,value):
    path=root/stem/f'f_{f:05d}.png'
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path),np.full((4,6,3),value,np.uint8))
    return str(path)


def coeff(root,f,gain=0.,base=.25):
    folder=root/'cutd_coeff'
    folder.mkdir(exist_ok=True)
    path=DEEP.coeff_path(folder,f)
    np.savez_compressed(path,gain=np.full((4,6,3),gain,np.float16),base=np.full((4,6,3),base,np.float16))
    return path


def fixture(monkeypatch,tmp_path):
    monkeypatch.setenv('CUTD_LOCAL_RENDERS',str(tmp_path))
    monkeypatch.setattr(AS,'_INDEX',{})
    race=EDL.T('cutd_race',0,'exact',hold=2719,need=(2719,2719),final_eligible=False)
    entry=EDL.T('cutd_clean',0,'exact',under=('take',race,2719,'D'),linear_mix='cutd_coeff',final_eligible=False)
    ctx=object.__new__(AS.Ctx)
    ctx.cut,ctx.variant,ctx.W,ctx.H,ctx.scale='D',None,6,4,1.
    return ctx,race,entry


def test_original_clipped_soft_entry_envelope_and_exact_first_frame(tmp_path):
    path=coeff(tmp_path,2724,gain=.5,base=8.)
    race=np.full((4,6,3),.2,np.float32)
    np.testing.assert_array_equal(DEEP.sweep(race,path,2720),race)
    expected=DEEP.to_srgb(DEEP.to_lin(race)*.5+np.full_like(race,.5))
    np.testing.assert_allclose(DEEP.sweep(race,path,2724),expected,atol=1e-7)
    wrong=DEEP.to_srgb(DEEP.to_lin(race)*.5+(DEEP.to_lin(race)*.5+8.)*.5)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(wrong,expected)


def test_required_explicit_under_and_coefficients_gate_readiness(monkeypatch,tmp_path):
    ctx,race,entry=fixture(monkeypatch,tmp_path)
    png(tmp_path,'cutd_clean',2720,100)
    assert AS.locate(entry,'D',None,2720)==(None,False)
    race_path=png(tmp_path,'cutd_race',2719,51)
    AS._INDEX.clear()
    assert AS.locate(entry,'D',None,2720)==(None,False)
    path=coeff(tmp_path,2720)
    assert AS.locate(entry,'D',None,2720)[0]
    np.testing.assert_array_equal(ctx.take_frame(entry,2720),np.full((4,6,3),.2,np.float32))
    assert AS.locate_under(entry['under'],2720)==(race_path,'cutd_race')
    assert AS.provisional_sources(entry,'D',None,2720)==('cutd_clean','under:cutd_race')
    race['need']=(2719,2720)
    assert AS.locate(entry,'D',None,2720)==(None,False)
    assert ctx.take_frame(entry,2720) is None
    assert ctx.under(entry['under'],2720) is None
    with pytest.raises(ValueError):
        AS.under_take(('take',dict(race,under=('hold','x',0)),2719,'D'))


def test_required_under_decode_failure_slates_while_legacy_under_retains_behavior(monkeypatch,tmp_path):
    ctx,race,entry=fixture(monkeypatch,tmp_path)
    foreground=png(tmp_path,'cutd_clean',2720,100)
    png(tmp_path,'cutd_race',2719,51)
    png(tmp_path,'cutd_matte',2720,128)
    take=EDL.T('cutd_clean',0,'exact',matte='cutd_matte',under=entry['under'])
    original=ctx.read
    ctx.read=lambda p,*a,**kw:None if 'cutd_race' in p else original(p,*a,**kw)
    assert ctx.take_frame(take,2720) is None
    legacy=dict(take,under=('hold','cutd_race',2719))
    np.testing.assert_array_equal(ctx.take_frame(legacy,2720),original(foreground))
    ctx.read=lambda p,*a,**kw:None if 'cutd_matte' in p else original(p,*a,**kw)
    assert ctx.take_frame(take,2720) is None
    np.testing.assert_array_equal(ctx.take_frame(legacy,2720),original(foreground))
    with pytest.raises(ValueError):
        EDL.T('x',linear_mix='cutd_coeff',under=entry['under'],grade='C')


def test_complete_entry_coefficients_and_exit_mattes_are_required_atomically(monkeypatch,tmp_path):
    ctx,race,entry=fixture(monkeypatch,tmp_path)
    png(tmp_path,'cutd_race',2719,51)
    entry['need']=(2720,2757)
    for f in range(2720,2758):
        png(tmp_path,'cutd_clean',f,100)
        if f<2728:
            coeff(tmp_path,f)
    shot=EDL.S('D2',2720,2728,'D11a','Entry','test','test',[entry])
    assert not AS.complete(entry,'D',None,2720,2757)
    assert AS.plan_shot(shot,'D',None)['kind']=='slate'
    for f in range(2728,2758):
        coeff(tmp_path,f)
    assert AS.plan_shot(shot,'D',None)['kind']=='take'
    exit_take=EDL.T('cutd_clean',0,'exact',matte='cutd_matte',under=entry['under'],need=(2720,2757))
    shot['takes']=[exit_take]
    assert AS.locate(exit_take,'D',None,2720)==(None,False)
    for f in range(2720,2757):
        png(tmp_path,'cutd_matte',f,128)
    AS._INDEX.clear()
    assert AS.plan_shot(shot,'D',None)['kind']=='slate'
    png(tmp_path,'cutd_matte',2757,128)
    AS._INDEX.clear()
    assert AS.plan_shot(shot,'D',None)['kind']=='take'


def test_new_race_pixels_and_cache_sources_follow_the_same_explicit_take(monkeypatch,tmp_path):
    ctx,race,entry=fixture(monkeypatch,tmp_path)
    png(tmp_path,'cutd_clean',2728,100)
    p0=png(tmp_path,'cutd_race',2719,30)
    path=coeff(tmp_path,2728,gain=.5,base=.1)
    image0=ctx.take_frame(entry,2728)
    monkeypatch.setattr(DV,'_stat',lambda p:p)
    keys0=DV.frame_sources('D',None,dict(kind='take',take=entry),2728)
    assert p0 in keys0 and path in keys0
    p1=png(tmp_path,'cutd_new_race',2719,190)
    entry['under']=('take',EDL.T('cutd_new_race',0,'exact',hold=2719),2719,'D')
    image1=ctx.take_frame(entry,2728)
    keys1=DV.frame_sources('D',None,dict(kind='take',take=entry),2728)
    assert p1 in keys1 and p0 not in keys1
    assert keys0!=keys1 and not np.array_equal(image0,image1)
    assert AS.provisional_sources(entry,'D',None,2728)==('cutd_clean',)


def test_new_under_dependencies_are_traced_for_video_takes_too(monkeypatch,tmp_path):
    ctx,race,entry=fixture(monkeypatch,tmp_path)
    p=png(tmp_path,'cutd_race',2719,30)
    coefficients=coeff(tmp_path,2728)
    video=dict(entry,mode='video',video='synthetic.mov')
    monkeypatch.setattr(AS,'video_frames',lambda take:3000)
    monkeypatch.setattr(DV,'_stat',lambda path:path)
    traced=DV.frame_sources('D',None,dict(kind='take',take=video),2728)
    assert traced[0].endswith('synthetic.mov#2728')
    assert p in traced and coefficients in traced
    legacy=dict(video)
    del legacy['linear_mix'],legacy['under']
    assert DV.frame_sources('D',None,dict(kind='take',take=legacy),2728)==[traced[0]]


def test_exit_crosses_row_boundary_once_and_finishes_after_composition(monkeypatch,tmp_path):
    rgb=png(tmp_path,'cutd_exit',2960,25)
    cover=png(tmp_path,'cutd_cover',2960,128)
    incoming=np.full((4,6,3),.4,np.float32)
    expected=np.full_like(incoming,25/255+(1-128/255)*.4)
    np.testing.assert_allclose(DEEP.reveal(incoming,rgb,cover),expected,atol=1e-7)
    monkeypatch.setattr(AS,'transition_layers',lambda *a:dict(glow=rgb,cover=cover))
    t={'under_start':2960}
    fin=lambda img,*a:img*.8
    np.testing.assert_array_equal(AS._tk_deep_reveal(incoming,{},2959,t,fin,'D'),incoming*.8)
    np.testing.assert_allclose(AS._tk_deep_reveal(incoming,{},2960,t,fin,'D'),expected*.8,atol=1e-7)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(DEEP.reveal(expected,rgb,cover),expected)


def test_exit_tail_clear_preserves_prior_frames_and_lands_exactly_on_live_under(monkeypatch,tmp_path):
    rgb=png(tmp_path,'cutd_exit',2976,25)
    cover=png(tmp_path,'cutd_cover',2976,128)
    incoming=np.full((4,6,3),.4,np.float32)
    original=DEEP.reveal(incoming,rgb,cover)
    for f in (2945,2960,2975,2976):
        np.testing.assert_array_equal(DEEP.reveal(incoming,rgb,cover,frame=f,tail_clear=(2976,2980)),original)
    assert [DEEP.tail_weight(f,(2976,2980)) for f in range(2976,2981)]==[1.,.84375,.5,.15625,0.]
    end=DEEP.reveal(incoming,rgb,cover,frame=2980,tail_clear=(2976,2980))
    np.testing.assert_array_equal(end,incoming)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(original,incoming)
    middle=DEEP.reveal(incoming,rgb,cover,frame=2978,tail_clear=(2976,2980))
    np.testing.assert_allclose(middle,original*.5+incoming*.5,atol=1e-7)
    monkeypatch.setattr(AS,'transition_layers',lambda *a:dict(glow=rgb,cover=cover))
    fin=lambda im,*a:im*.8
    t=dict(under_start=2960,tail_clear=(2976,2980))
    np.testing.assert_array_equal(AS._tk_deep_reveal(incoming,{},2980,t,fin,'D'),fin(incoming))


def test_deep_edl_holds_correct_d_plates_and_full_ranges():
    rows={r['code']:r for r in EDL.D}
    assert rows['D10']['takes'][0]==EDL.D_DEEP_RACE_SOURCE
    assert rows['D12']['takes'][0]==EDL.D_DEEP_BRINK_SOURCE
    for code in ('D11a','D11b','D11c'):
        take=rows[code]['takes'][0]
        assert AS.under_take(take['under'])==(EDL.D_DEEP_RACE_SOURCE,2719,'D')
        assert not EDL.is_final_take(take)
    assert AS.under_take(rows['D11d']['takes'][0]['under'])==(EDL.D_DEEP_BRINK_SOURCE,2960,'D')
    assert EDL.D_DEEP_ENTRY['need']==(2720,2757)
    assert EDL.D_DEEP_EXIT['need']==(2945,2980)
    t=next(t for t in EDL.TRANS['D'] if t['kind']=='deep_reveal')
    assert (t['f0'],t['f1'],t['under_start'])==(2945,2981,2960)
    assert t['tail_clear']==(2976,2980)
    assert 'def reveal' in AS.transition_code('deep_reveal')


def test_builder_rejects_shared_output_including_a_symlink(monkeypatch,tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    import rebake_cutd_deep as builder
    monkeypatch.setattr(builder,'ROOT',tmp_path)
    shared=tmp_path/'renders'
    shared.mkdir()
    linked=tmp_path/'looks_local'
    linked.symlink_to(shared,target_is_directory=True)
    for path in (shared,shared/'nested',linked/'nested'):
        with pytest.raises(ValueError,match='shared renders'):
            builder.validate_output(path)
    builder.validate_output(tmp_path/'local')
