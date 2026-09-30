"""Generated fixtures only; no film audio, frames, renderer or shared cache access."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import sync_a_solo as S


class GeneratedAudio:
    samplerate,channels,subtype=S.SR,2,'PCM_24'
    def __init__(self,path,data):
        self.path=str(path);self.data=data;self.frames=len(data);self.reads=[]
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_bytes(b'synthetic identity, not a WAV')
    def read(self,a,b):
        self.reads.append(b-a)
        return self.data[a:b].astype(np.float64)


def test_shared_gain_is_not_independent_channel_ratio():
    t=np.arange(4800)/S.SR
    x=np.column_stack([.1*np.sin(2*np.pi*1000*t),.07*np.cos(2*np.pi*1200*t)])
    clean=S.shared_gain_metrics(x,x*.5)
    assert clean['vector_maximum_error']<1e-14
    assert clean['cross_channel']['0_to_1']['maximum_error']<1e-14
    stale=S.shared_gain_metrics(x,x*np.array([.5,.7]))
    assert stale['vector_maximum_error']>.005
    assert stale['cross_channel']['0_to_1']['maximum_error']>.01


def test_stream_loudness_equals_library():
    pyln=pytest.importorskip('pyloudnorm')
    rng=np.random.default_rng(42)
    score=(rng.normal(size=(3*S.SR,2))*.01).astype(np.float32)
    effects=(rng.normal(size=score.shape)*.004).astype(np.float32)
    expect=pyln.Meter(S.SR).integrated_loudness((score+effects).astype(np.float64))
    assert S.integrated_loudness(score,effects)==pytest.approx(expect,abs=1e-10)
    assert abs(S.integrated_loudness(score*2,effects*2)-expect)>5


def make_fixture(tmp_path):
    pytest.importorskip('pyloudnorm')
    rng=np.random.default_rng(23);n=3*S.SR
    folder=tmp_path/'solos';folder.mkdir()
    cue=(rng.normal(size=(n,2))*.01).astype(np.float32)
    score=(rng.normal(size=(n,2))*.006).astype(np.float32)
    env=np.full(n,.5,np.float32)
    item=dict(id='A.synthetic',start=0,stop=n)
    index=dict(n=n,sr=S.SR,items=[item])
    (folder/'index.json').write_text(json.dumps(index))
    np.save(folder/'full_pre_guard.npy',cue)
    np.savez(folder/'A.synthetic.npz',start=0,data=cue)
    cache=tmp_path/'music'/'cache'/'v3';cache.mkdir(parents=True)
    np.save(cache/'master_env_sound_TEST.npy',env)
    np.save(cache/'premaster_score_final_TEST.npy',score)
    loudness=S.integrated_loudness(score,cue)
    guard,_=S.guard_curve(score,cue,loudness)
    out=np.concatenate([x for _,x in S.candidate_chunks(cue,guard,env)])
    sfx=GeneratedAudio(tmp_path/'music'/'out'/'v3'/'sound_TEST_sfx.wav',out)
    return folder,index,cue,sfx


def test_complete_reconciliation_and_adapter(tmp_path):
    folder,index,cue,sfx=make_fixture(tmp_path)
    check,bank=S.reconcile(folder,sfx,['A.synthetic'])
    assert check['ok'],check
    assert max(sfx.reads)<=S.BLOCK
    with bank.open('A.synthetic') as source:
        assert source.frames==sfx.frames
        for a in [0,100,48000,2*S.SR]:
            assert np.max(abs(source.read(a,a+200)-sfx.read(a,a+200)))<1e-6
    bank.close()


@pytest.mark.parametrize('damage',['missing','duplicate','misaligned','changed_full_and_solo','one_channel','nan'])
def test_reconciliation_rejects_stale_or_bad_inputs(tmp_path,damage):
    folder,index,cue,sfx=make_fixture(tmp_path)
    ids=['A.synthetic']
    if damage=='missing':ids=['A.wrong']
    if damage=='duplicate':ids*=2
    if damage=='misaligned':
        np.savez(folder/'A.synthetic.npz',start=1,data=cue[:-1])
    if damage=='changed_full_and_solo':
        stale=np.roll(cue,2000,axis=0)
        np.save(folder/'full_pre_guard.npy',stale)
        np.savez(folder/'A.synthetic.npz',start=0,data=stale)
    if damage=='one_channel':sfx.data[:,1]*=.8
    if damage=='nan':
        cue[20,0]=np.nan;np.savez(folder/'A.synthetic.npz',start=0,data=cue)
    check,bank=S.reconcile(folder,sfx,ids)
    assert not check['ok'],check
    assert bank is None


def test_decomposition_detects_shift_before_master_fit(tmp_path):
    folder,index,cue,sfx=make_fixture(tmp_path)
    np.savez(folder/'A.synthetic.npz',start=0,data=np.roll(cue,2000,axis=0))
    result=S.decomposition(folder,index,cue)
    assert not result['ok']
    assert result['maximum_error']>.01


def test_guard_matches_source_equation_on_generated_samples():
    rng=np.random.default_rng(19);score=(rng.normal(size=(S.SR,2))*.02).astype(np.float32)
    full=(rng.normal(size=(S.SR,2))*.10).astype(np.float32)
    actual,_=S.guard_curve(score,full,-25.)
    gain=10**(9/20)*10**(1.2/20);lim=10**(-2.3/20)/gain
    sa=np.abs(score).max(1);total=np.abs(score+full).max(1);over=(total>lim)&(sa<lim)
    need=np.ones(len(full),np.float32)
    need[over]=np.clip((lim*.97-sa[over])/(np.abs(full).max(1)[over]+1e-12),10**(-10/20),1.)
    gr=S.minimum_filter1d(20*np.log10(need),size=481)
    pole=np.exp(-1/(.12*S.SR));smooth=signal.lfilter([1-pole],[1,-pole],gr)
    expected=(10**(np.minimum(gr,smooth)/20)).astype(np.float32)
    np.testing.assert_array_equal(actual,expected)
    assert np.max(abs(actual-1))>.1


def test_diagnostic_bank_cannot_promote_failed_identity(tmp_path):
    folder,index,cue,sfx=make_fixture(tmp_path)
    sfx.data[:,1]*=.8
    check,bank=S.reconcile(folder,sfx,['A.synthetic'],diagnostic_bank=True)
    assert not check['ok']
    assert bank is not None and bank.attribution_validated is False
    bank.close()
    check,bank=S.reconcile(folder,sfx,['A.wrong'],diagnostic_bank=True)
    assert not check['ok'] and bank is None


def test_processed_sum_rejects_broken_adapter_gain(tmp_path):
    folder,index,cue,sfx=make_fixture(tmp_path)
    check,bank=S.reconcile(folder,sfx,['A.synthetic'])
    assert check['ok'] and check['processed_solo_sum']['ok']
    bank.guard*=.8
    check=S.processed_decomposition(bank,sfx)
    assert not check['ok']
    assert check['maximum_error']>.001


def test_bank_rejects_input_change_after_reconciliation(tmp_path):
    folder,index,cue,sfx=make_fixture(tmp_path)
    check,bank=S.reconcile(folder,sfx,['A.synthetic'])
    assert check['ok']
    path=folder/'A.synthetic.npz'
    path.write_bytes(b'replaced after reconciliation')
    with pytest.raises(ValueError,match='input changed'):
        with bank.open('A.synthetic'):
            pytest.fail('stale bank opened')


def test_late_exception_does_not_retain_earlier_true_verdict(tmp_path,monkeypatch):
    folder,index,cue,sfx=make_fixture(tmp_path)
    def fail_after_chain(*args):
        raise ValueError('synthetic input mutation at final sum')
    monkeypatch.setattr(S,'processed_decomposition',fail_after_chain)
    check,bank=S.reconcile(folder,sfx,['A.synthetic'])
    assert check['maximum_error']<S.RECONSTRUCTION_ATOL
    assert not check['ok'] and bank is None
    assert 'synthetic input mutation' in check['reason']


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_delivered_quiet_chunk_is_not_hidden_by_max(tmp_path,bad):
    folder,index,cue,sfx=make_fixture(tmp_path)
    cue[2*S.SR:]=0
    np.save(folder/'full_pre_guard.npy',cue)
    np.savez(folder/'A.synthetic.npz',start=0,data=cue)
    cache=Path(sfx.path).parent.parent.parent/'cache'/'v3'
    score=np.load(cache/'premaster_score_final_TEST.npy')
    env=np.load(cache/'master_env_sound_TEST.npy')
    guard,_=S.guard_curve(score,cue,S.integrated_loudness(score,cue))
    sfx.data=np.concatenate([x for _,x in S.candidate_chunks(cue,guard,env)])
    sfx.data[-1,0]=bad
    check,bank=S.reconcile(folder,sfx,['A.synthetic'])
    assert not check['ok'] and bank is None
    assert 'non-finite' in check['reason']
    json.dumps(check,allow_nan=False)


def test_processed_sum_rejects_nonfinite_actual_mix(tmp_path):
    folder,index,cue,sfx=make_fixture(tmp_path)
    check,bank=S.reconcile(folder,sfx,['A.synthetic'])
    assert check['ok']
    sfx.data[-1,0]=np.nan
    with pytest.raises(ValueError,match='non-finite'):
        S.processed_decomposition(bank,sfx)


def test_zero_effects_failure_report_is_strict_json(tmp_path):
    folder,index,cue,sfx=make_fixture(tmp_path)
    cue[:]=0;sfx.data[:]=0
    np.save(folder/'full_pre_guard.npy',cue)
    np.savez(folder/'A.synthetic.npz',start=0,data=cue)
    check,bank=S.reconcile(folder,sfx,['A.synthetic'])
    assert not check['ok'] and bank is None
    assert check['safety_gain'] is None
    assert check['global_ls_gain'] is None
    json.dumps(check,allow_nan=False)


def test_master_chain_overflow_is_rejected():
    full=np.ones((S.SR,2),np.float32)
    full[::2]*=-1
    env=np.full(S.SR,np.finfo(np.float32).max,np.float32)
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError,match='non-finite master-chain output'):
            list(S.candidate_chunks(full,np.full(S.SR,2.,np.float32),env))
