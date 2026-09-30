"""Bounded ignition treatment; controls recreate the old hole or disable the new hook."""
import numpy as np
import pytest

import polish_ignition_A as P


def _assert_breath(gain):
    # The held floor is sustained; its release is distributed across multiple frames.
    assert gain[1020] == pytest.approx(gain[1026])
    assert -20 < gain[1020] < -8
    assert np.max(np.abs(np.diff(gain[1000:1042]))) < 2.5
    assert gain[1040] > gain[1026] + 8


def test_release_hold_and_inhale_have_no_gate():
    frames = np.arange(1140)
    gain = P.curve(frames, P.GAIN_POINTS)
    _assert_breath(gain)
    old = gain.copy()
    old[1020:1040] = -45
    with pytest.raises(AssertionError):
        _assert_breath(old)
    assert np.array_equal(gain[:960], np.zeros(960))
    assert np.array_equal(gain[1120:], np.zeros(20))


def _assert_local(before, after):
    assert np.array_equal(before[:960*P.FR], after[:960*P.FR])
    assert np.array_equal(before[1120*P.FR:], after[1120*P.FR:])
    assert np.max(np.abs(after[1020*P.FR:1026*P.FR])) > 0.01


def test_replacement_restores_floor_with_exact_exterior_identity():
    source = np.full((1124*P.FR, 2), 0.25, np.float32)
    source[1020*P.FR:1040*P.FR] *= 10**(-45/20)
    candidate = P.blend_score(source.copy(), np.full(((P.END-P.START)*P.FR, 2), 0.25, np.float32))
    _assert_local(source, candidate)
    with pytest.raises(AssertionError):
        _assert_local(source, source)  # The original gated breath fails the restored-floor assertion.
    leaked = candidate.copy()
    leaked[950*P.FR] += 0.001
    with pytest.raises(AssertionError):
        _assert_local(source, leaked)
    assert np.array_equal(source[960*P.FR], candidate[960*P.FR])
    assert np.max(np.abs(np.diff(candidate[1119*P.FR:1121*P.FR], axis=0))) < 1e-6
    with pytest.raises(ValueError, match='wrong length'):
        P.blend_score(source, np.zeros((1, 2), np.float32))


def test_recorded_air_rises_on_visual_swell_and_releases(monkeypatch):
    import sound_v3 as SV
    monkeypatch.setattr(SV, 'load', lambda ref, start, end: np.ones((round((end-start)*P.SR), 2), np.float32))
    monkeypatch.setattr(SV, 'proc', lambda y, **kw: y)
    source = np.zeros((1120*P.FR, 2), np.float32)
    candidate = P.inhale(source.copy())
    def check(y):
        assert np.any(y[1020*P.FR:1026*P.FR])
        assert np.mean(y[1040*P.FR:1041*P.FR]) > 2*np.mean(y[1026*P.FR:1027*P.FR])
        assert not np.any(y[:1004*P.FR]) and not np.any(y[1088*P.FR:])
        assert np.max(np.abs(np.diff(y, axis=0))) < 1e-5
    check(candidate)
    with pytest.raises(AssertionError):
        check(source)  # Disabling the new air leaves the held breath empty.
    monkeypatch.setattr(SV, 'load', lambda ref, start, end: np.zeros((round((end-start)*P.SR), 2), np.float32))
    with pytest.raises(ValueError, match='silent'):
        P.inhale(source)


def test_hook_applies_to_ap2_only_and_is_forwarded(monkeypatch):
    import sound_recipes_A as A
    import sound_recipes_AP2 as AP2
    calls = []
    monkeypatch.setattr(P, 'local_score', lambda cut: calls.append(cut) or np.array([[2.0]]))
    monkeypatch.setattr(P, 'blend_score', lambda s, r: s.__setitem__((0, 0), r[0, 0]))
    monkeypatch.setattr(P, 'inhale', lambda s: s.__setitem__((0, 0), 3.0))
    assert AP2.prepare_master is A.prepare_master is P.prepare_master
    for cut in ('AP2',):
        s, x = np.zeros((1, 1)), np.zeros((1, 1))
        P.prepare_master(cut, s, x)
        assert s[0, 0] == 2 and x[0, 0] == 3
    assert calls == ['AP2']
    for cut in ('A', 'B', 'C5', 'C5P2'):
        s, x = np.zeros((1, 1)), np.zeros((1, 1))
        P.prepare_master(cut, s, x)
        assert not s.any() and not x.any()
    assert calls == ['AP2']


def test_bounded_master_retains_reference_exterior_and_rejects_controls():
    import render_v3 as RV
    reference = np.linspace(0.7, 1.1, 100*P.FR, dtype=np.float32)
    current = reference*1.14
    actual = RV.bounded_master_env(current, reference, ((20,80),))
    def check(env):
        assert np.array_equal(env[:20*P.FR],reference[:20*P.FR])
        assert np.array_equal(env[80*P.FR:],reference[80*P.FR:])
        assert env[50*P.FR] == current[50*P.FR]
        assert np.max(np.abs(np.diff(env[19*P.FR:21*P.FR]))) < 1e-5
    check(actual)
    with pytest.raises(AssertionError):
        check(current)  # Re-solving the unbounded master changes every exterior sample.
    with pytest.raises(AssertionError):
        check(reference)  # Freezing the whole master would suppress the new local dynamics.
    with pytest.raises(ValueError, match='invalid reference'):
        RV.bounded_master_env(current, reference[:-1], ((20,80),))
    with pytest.raises(ValueError, match='explicit edit windows'):
        RV.bounded_master_env(current, reference, ())
    with pytest.raises(ValueError, match='outside audio'):
        RV.bounded_master_env(current, reference, ((-1,80),))


def test_guarded_sfx_restores_exterior_and_references_are_hash_checked(tmp_path,monkeypatch):
    import hashlib
    import json
    import soundfile as sf
    import render_v3 as RV
    monkeypatch.setattr(RV,'CACHE',str(tmp_path))
    monkeypatch.setattr(P,'__file__',str(tmp_path/'polish_ignition_A.py'))
    n=1448*P.FR
    sf.write(tmp_path/'reference.wav',np.full((n,2),0.125,np.float32),P.SR,subtype='FLOAT')
    np.save(tmp_path/'reference.npy',np.ones(n,np.float32))
    files={kind:dict(filename=name,sha256=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest())
           for kind,name in [('envelope','reference.npy'),('sfx','reference.wav')]}
    (tmp_path/'polish_ignition_reference.json').write_text(json.dumps(dict(files=files)))
    candidate=np.full((n,2),0.5,np.float32)
    result,options=P.bound_master('AP2',candidate)
    assert np.all(result[:960*P.FR]==0.125)
    assert np.all(result[1440*P.FR:]==0.125)
    assert np.all(result[960*P.FR:1440*P.FR]==0.5)
    assert options['edit_windows']==((960,1440),)
    assert np.all(options['reference_env']==1)
    untouched=np.zeros((1,2),np.float32)
    assert P.bound_master('B',untouched)==(untouched,{})
    (tmp_path/'reference.npy').write_bytes(b'changed reference')
    with pytest.raises(ValueError,match='hash mismatch'):
        P.bound_master('AP2',candidate)
    (tmp_path/'reference.npy').unlink()
    with pytest.raises(ValueError,match='missing ignition reference'):
        P.bound_master('AP2',candidate)


def test_breath_report_describes_polished_score_and_preserves_other_breaths():
    score=np.full((1046*P.FR,2),.01,np.float32)
    probes=[dict(start_s=42.5,end_s=1040/24,before_db=-25,inside_db=-80,after_db=-39),
            dict(start_s=76.46,end_s=76.66,before_db=-25,inside_db=-80,after_db=-23)]
    result=P.refresh_breath_probes('AP2',score,probes)
    assert result[0]['inside_db']==pytest.approx(-40)
    assert result[1]==probes[1]
    assert probes[0]['inside_db']==-80
    assert P.refresh_breath_probes('B',score,probes) is probes
    with pytest.raises(ValueError,match='missing or ambiguous'):
        P.refresh_breath_probes('AP2',score,probes[1:])
