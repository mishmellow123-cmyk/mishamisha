"""Explicit external D assets: all consumers resolve the same files; no legacy stem overlay."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import assemble as AS
import deliver as DV
import edl_v3 as EDL


def write(root,stem,frame,value):
    folder=root/stem
    folder.mkdir(exist_ok=True)
    path=folder/f'f_{frame:05d}.png'
    cv2.imwrite(str(path),np.full((4,6,3),value,np.uint8))
    return str(path)


def test_local_lookup_is_reserved_explicit_and_does_not_touch_d07(monkeypatch,tmp_path):
    monkeypatch.delenv('CUTD_LOCAL_RENDERS',raising=False)
    baseline=AS.chain(EDL.D_BURN_PLATE,'D')
    assert AS.render_dir('cutd_test')==str(Path(AS.RENDERS)/'cutd_test')
    monkeypatch.setenv('CUTD_LOCAL_RENDERS',str(tmp_path))
    assert AS.render_dir('cutd_test')==str(tmp_path/'cutd_test')
    assert AS.chain(EDL.D_BURN_PLATE,'D')==baseline
    for stem in ('book_C_ft','cand_sweep_soft-entry','embers_A3','renders/cutd_foo'):
        assert AS.render_dir(stem)==str(Path(AS.RENDERS)/stem)
    with pytest.raises(AssertionError):
        assert AS.render_dir('embers_A3')==str(tmp_path/'embers_A3')
    with pytest.raises(ValueError):
        AS.render_dir('cutd_foo/../../outside')


def test_rgb_matte_under_add_and_trace_share_local_root(monkeypatch,tmp_path):
    monkeypatch.setenv('CUTD_LOCAL_RENDERS',str(tmp_path))
    monkeypatch.setattr(AS,'_INDEX',{})
    rgb=write(tmp_path,'cutd_rgb',10,20)
    matte=write(tmp_path,'cutd_matte',10,128)
    under=write(tmp_path,'cutd_under',7,50)
    add=write(tmp_path,'cutd_add',10,10)
    take=EDL.T('cutd_rgb',0,'exact',matte='cutd_matte',under=('hold','cutd_under',7),add='cutd_add',
               final_eligible=False)
    ctx=object.__new__(AS.Ctx)
    ctx.cut,ctx.variant,ctx.W,ctx.H,ctx.scale='D',None,6,4,1.
    expected=np.float32((20+(1-128/255)*50+10)/255)
    np.testing.assert_allclose(ctx.take_frame(take,10),expected,atol=1e-7)
    assert AS.locate_under(take['under'],10)==(under,'cutd_under')
    monkeypatch.setattr(DV,'_stat',lambda p:p)
    assert DV.frame_sources('D',None,dict(kind='take',take=take),10)==[rgb,add,matte,under]
    assert AS.provisional_sources(take,'D',None,10)==('cutd_rgb',)
    monkeypatch.delenv('CUTD_LOCAL_RENDERS')
    assert AS.locate(take,'D',None,10)==(None,False)
    assert ctx.take_frame(take,10) is None


def test_transition_layer_lookup_and_root_change_invalidate_sources(monkeypatch,tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    a.mkdir(); b.mkdir()
    for root in (a,b):
        write(root,'cutd_glow',20,10)
        write(root,'cutd_cover',20,128)
    t=dict(f0=20,f1=21,kind='x1',glow='cutd_glow',cover='cutd_cover')
    monkeypatch.setenv('CUTD_LOCAL_RENDERS',str(a))
    first=AS.transition_layers(t,20)
    monkeypatch.setenv('CUTD_LOCAL_RENDERS',str(b))
    second=AS.transition_layers(t,20)
    assert first!=second
    assert all(str(b) in path for path in second.values())
    monkeypatch.delenv('CUTD_LOCAL_RENDERS')
    assert AS.transition_layers(t,20) is None
