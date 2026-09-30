"""D07's explicit burn and source/cache contracts; tiny synthetic images, no render corpus reads."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assemble as AS
import deliver as DV
import edl_v3 as EDL
import ring_burn as RB
from cutd_test_fixtures import adoption_lines, edl_namespace


def burn():
    return next(t for t in EDL.TRANS['D'] if t['kind'] == 'ring_burn')


def assert_source_contract(take):
    assert [EDL.source_frame(take,f) for f in (1680,1719,1720,1759)] == [1400,1439,1439,1439]
    assert take['need'] == (1400,1439)
    assert not EDL.is_final_take(take)


def test_explicit_standin_and_future_one_line_adoption():
    unadopted = edl_namespace()
    row = next(s for s in unadopted['D'] if s['code'] == 'D07')
    assert row['takes'] == [EDL.D_BURN_PLATE]
    assert_source_contract(row['takes'][0])
    for key, value in [('clamp',(1400,1440)), ('need',(1400,1438)), ('final_eligible',True)]:
        rejected = dict(row['takes'][0], **{key:value})
        with pytest.raises(AssertionError):
            assert_source_contract(rejected)
    future = adoption_lines()['D08'][0]
    assert EDL.source_frame(future,1759) == 1759
    assert 'screen_transform' not in future
    assert EDL.is_final_take(future)
    assert future['need'] == (1680,2079)
    assert next(s for s in unadopted['D'] if s['code']=='D08')['takes'] == []
    assert all(row['takes'][0][k] is None for k in ('matte','under','add'))
    opted=[(cut,s['code']) for cut,rows in unadopted['EDL'].items() for s in rows for t in s['takes']
           if 'clamp' in t or 'screen_transform' in t]
    assert opted==[('D','D07')]
    live = {s['code']:s for s in EDL.D}
    assert live['D07']['takes'] == live['D08']['takes'] == [future]
    assert not any('clamp' in t or 'screen_transform' in t for s in EDL.D for t in s['takes'])


@pytest.mark.parametrize('missing_frame',(1680,1700,2079))
def test_future_arrival_alone_does_not_adopt_and_explicit_swap_requires_all_frames(monkeypatch,missing_frame):
    current=next(s for s in edl_namespace()['D'] if s['code']=='D07')
    adopted_rows={s['code']:s for s in edl_namespace(adopt=('D08',))['D']}
    sources={'embers_A3':{f:f'A/{f}' for f in range(1400,1440)},
             'embers_D_inscription':{f:f'D/{f}' for f in range(1680,2080)}}
    monkeypatch.setattr(AS,'index',lambda p:sources.get(Path(p).name,{}))
    assert AS.plan_shot(current,'D',None)['take']['stem']=='embers_A3'
    for code in ('D07','D08'):
        assert AS.plan_shot(adopted_rows[code],'D',None)['take']['stem']=='embers_D_inscription'
    del sources['embers_D_inscription'][missing_frame]
    for code in ('D07','D08'):
        assert AS.plan_shot(adopted_rows[code],'D',None)['kind']=='slate'


def test_layers_call_original_v2_kernel_in_native_screen_units(monkeypatch):
    calls={}
    def params(origin,**kwargs):
        calls['origin'],calls['params']=origin,kwargs
        return 'params'
    def kernel(u,v,glow,keep,cover,bf,seconds):
        calls.update(u=u,v=v,bf=bf,seconds=seconds)
        glow[:]=[.7,.3,.1]
        keep[:]=[.4,.2,.1]
        cover[:]=.6
    def finish(glow,**kwargs):
        calls['finish']=kwargs
        return glow
    engine=SimpleNamespace(UNIT=100.,BURN=SimpleNamespace(params=params,v2=lambda p:'v2:'+p),
                           _kernel=kernel,look=SimpleNamespace(finish=finish))
    monkeypatch.setattr(RB,'_engine',lambda:engine)
    glow,keep,cover=RB.layers(1700,burn(),48,20)
    assert calls['origin']==(9.98,.92)
    assert calls['bf']=='v2:params'
    assert calls['seconds']==1700/24
    assert calls['params']['t_start']==1682/24
    assert calls['params']['speed']==2.8
    assert calls['u'][0,0]==.2 and calls['u'][-1,-1]==19.
    np.testing.assert_allclose([calls['v'][0,0],calls['v'][-1,-1]],[.201,7.839],atol=1e-12)
    assert calls['finish']==dict(exposure=1.,bloom_strength=.05,bloom_threshold=1.,vignette_amount=0.,lift=0.)
    np.testing.assert_array_equal(glow[0,0],np.float32([.7,.3,.1]))
    np.testing.assert_array_equal(keep[0,0],np.float32([.4,.2,.1]))
    assert cover.shape==(20,48)
    with pytest.raises(AssertionError):
        assert calls['origin']==(9.6,4.02)


@pytest.mark.parametrize('kwargs', [dict(clamp=(-1,2)),dict(clamp=(2,1)),dict(clamp=(0,1.5)),
                                    dict(clamp=(0,1),hold=0)])
def test_invalid_clamp_rejected(kwargs):
    with pytest.raises(ValueError):
        EDL.T('test',**kwargs)


def test_default_take_source_clock_and_pixels_unchanged(monkeypatch):
    take = EDL.T('legacy',-20,'exact')
    assert 'clamp' not in take and 'screen_transform' not in take
    assert EDL.source_frame(take,30) == 10
    image = np.arange(90,dtype=np.float32).reshape(5,6,3)/100
    ctx = object.__new__(AS.Ctx)
    ctx.cut, ctx.variant = 'C', None
    ctx.read = lambda *a,**kw: image.copy()
    monkeypatch.setattr(AS,'locate',lambda *a: ('source',False))
    monkeypatch.setattr(RB,'screen_transform',lambda image,*a: image*.5)
    np.testing.assert_array_equal(ctx.take_frame(take,30),image)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(ctx.take_frame(dict(take,screen_transform={'on':True}),30),image)


def test_rgb_keep_is_not_hole_coverage(monkeypatch):
    o=np.full((4,5,3),.6,np.float32)
    i=np.full_like(o,.4)
    glow=np.full_like(o,.02)
    keep=np.broadcast_to(np.float32([.2,.1,.05]),o.shape)
    cover=np.full(o.shape[:2],.8,np.float32)
    monkeypatch.setattr(RB,'layers',lambda *a:(glow,keep,cover))
    actual=RB.composite(o,i,1688,burn())
    expected=o*keep+i*.2+glow
    np.testing.assert_allclose(actual,expected,atol=1e-7)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(o*keep+i*(1-keep)+glow,expected,atol=1e-7)
    np.testing.assert_array_equal(RB.composite(o,i,1680,burn()),o)


def test_pair_reads_held_page_and_advancing_then_held_flame():
    t=burn()
    assert (t['f0'],t['f1'],t['cut'],t['center']) == (1680,1760,1680,(998.,92.))
    assert AS.transition_source_frames(t,1759) == (1679,1759)
    page=next(s for s in EDL.D if s['code']=='D06')['takes'][0]
    assert EDL.source_frame(page,1679)==559
    assert AS.transition_layers(t,1700)=={}
    with pytest.raises(AssertionError):
        assert AS.transition_source_frames(dict(t,cut=1760),1759)==(1679,1759)


def test_transform_places_flame_anchor_and_preserves_colour():
    spec=dict(f0=0,f1=10,source_f0=0,source_f1=5,anchor0=(960,402),anchor1=(960,402),
              target0=(480,201),target1=(1440,603),scale0=1.,scale1=1.,border=(0.,0.,0.))
    image=np.zeros((804,1920,3),np.float32)
    image[400:405,958:963]=[.3,.6,.9]
    for f,(x,y) in [(0,(480,201)),(10,(1440,603))]:
        result=RB.screen_transform(image,f,spec)
        np.testing.assert_array_equal(result[y,x],image[402,960])
        np.testing.assert_array_equal(result[402,960],np.zeros(3))
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(image[y,x],image[402,960])


def test_source_trace_and_cache_identity_cover_clamp_transform_and_burn_code(monkeypatch):
    monkeypatch.setattr(AS,'locate',lambda take,cut,variant,f:(f"source/{EDL.source_frame(take,f)}",False))
    monkeypatch.setattr(DV,'_stat',lambda p:p)
    plan=dict(kind='take',take=deepcopy(EDL.D_BURN_PLATE))
    before=DV.frame_sources('D',None,plan,1759)
    assert before[0]=='source/1439'
    plan['take']['clamp']=(1400,1440)
    assert DV.frame_sources('D',None,plan,1759)!=before
    plan['take']=deepcopy(EDL.D_BURN_PLATE)
    plan['take']['screen_transform']['target0']=(990,90)
    assert DV.frame_sources('D',None,plan,1759)!=before
    legacy=DV.frame_sources('A',None,dict(kind='take',take=EDL.T('legacy')),20)
    assert legacy==['source/20']
    assert 'BURN.field2' in AS.transition_code('ring_burn')
    assert 'def field2' in AS.transition_code('ring_burn')
    assert 'def gnoise' in AS.transition_code('ring_burn')
    with pytest.raises(AssertionError):
        assert 'BURN.field2' in AS.transition_code('burn')


def test_context_dispatches_real_burn_and_suppresses_on_slate(monkeypatch):
    o=np.full((5,6,3),.8,np.float32)
    i=np.full_like(o,.2)
    shot={'code':'fixture'}
    source={'stem':'fixture'}
    monkeypatch.setattr(AS.Ctx,'picture',lambda self,f:(o if f==1679 else i,shot,'take',source))
    ctx=SimpleNamespace(cut='D',W=6,H=5,picture=lambda f:(i,shot,'plain',source))
    monkeypatch.setattr(RB,'composite',lambda out,inc,f,t:out*.25+inc*.75)
    image,_,status,_=AS._transitions(ctx)(1700)
    np.testing.assert_allclose(image,.35)
    assert status.endswith('ring_burn')
    monkeypatch.setattr(AS.Ctx,'picture',lambda self,f:(i,shot,'SLATE',None))
    image,_,status,_=AS._transitions(ctx)(1700)
    np.testing.assert_array_equal(image,i)
    assert status=='plain'


def test_video_transform_and_its_helper_change_cache_identity(monkeypatch):
    monkeypatch.setattr(AS,'locate',lambda *a:(('video','clip.mp4',12),False))
    monkeypatch.setattr(DV,'_stat',lambda p:p)
    take=EDL.T('video',mode='video',video='clip.mp4')
    plan=dict(kind='take',take=take)
    assert DV.frame_sources('D',None,plan,1700)==['clip.mp4#12']
    take['screen_transform']=deepcopy(EDL.D_BURN_PLATE['screen_transform'])
    before=DV.frame_sources('D',None,plan,1700)
    assert len(before)==3
    take['screen_transform']['target0']=(980.,92.)
    assert DV.frame_sources('D',None,plan,1700)!=before
    take['screen_transform']=deepcopy(EDL.D_BURN_PLATE['screen_transform'])
    monkeypatch.setattr(RB,'screen_transform',lambda image,*a:image)
    assert DV.frame_sources('D',None,plan,1700)!=before
    with pytest.raises(AssertionError):
        assert ['clip.mp4#12']==before
