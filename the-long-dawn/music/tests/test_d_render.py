"""D renderer kernels, private-cache identity, and protected PCM boundaries.

Fixtures are synthetic test signals; no donor samples or shared caches are read.
The stock references below use mix.py directly, never a renderer import with
directory side effects. Corruptions prove that the comparison predicates fire.
"""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d

import mix as MX
import render_D as D


@pytest.fixture(scope="module")
def loud_signal():
    n = D.SR*9
    t = np.arange(n)/D.SR
    rng = np.random.default_rng(917)
    y = np.column_stack((np.sin(2*np.pi*997*t), np.sin(2*np.pi*1319*t+.3)))
    y *= (1.1+.5*np.sin(2*np.pi*.37*t))[:,None]
    y += rng.normal(0, .015, y.shape)
    return y.astype(np.float32)


def test_glue_streaming_matches_stock_when_compression_actually_fires(loud_signal):
    got = np.repeat(D.glue_gain(loud_signal), 48)[:len(loud_signal)]
    want = MX.glue_comp_gain(loud_signal, thresh_db=-16, ratio=1.5, attack=.04, release=.4)
    assert want.min() < .8  # A below-threshold fixture would never test compression.
    np.testing.assert_allclose(got, want, atol=2e-6, rtol=0)
    bad = got.copy()
    bad[D.CHUNK-12:D.CHUNK+12] = 1.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(bad, want, atol=2e-6, rtol=0)


def test_true_peak_blocks_match_one_resample_across_boundary(loud_signal):
    want = np.abs(signal.resample_poly(loud_signal.astype(np.float64), 4, 1, axis=0))
    want = want.max(1).reshape(-1, 4).max(1)
    got = D.true_peak_envelope(loud_signal)
    np.testing.assert_allclose(got, want, atol=2e-7, rtol=0)
    bad = got.copy()
    bad[D.CHUNK-2:D.CHUNK+2] = 0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(bad, want, atol=2e-7, rtol=0)


def test_limiter_matches_stock_control_and_reduces_loud_input(loud_signal):
    need = np.minimum(1., 10**(-1.3/20)/np.maximum(D.true_peak_envelope(loud_signal), 1e-9))
    need = minimum_filter1d(need, size=2*int(.006*D.SR)+1)
    db = 20*np.log10(need)
    want_gain = (10**(np.minimum(MX._smooth_gain(db, .0005, .15), db)/20)).astype(np.float32)
    want = loud_signal*want_gain[:,None]
    got = loud_signal.copy()
    reduction = D.limit_inplace(got)
    assert reduction < -3.
    np.testing.assert_allclose(got, want, atol=2e-7, rtol=0)
    assert D.true_peak_db(got) <= -1.2
    assert D.true_peak_db(loud_signal) > -1.2  # bypass is the negative control


def test_filter_state_and_part_processing_cross_chunk_seam(loud_signal):
    from dsl import Part
    p = Part("fixture", "vla", pan=-.37, width=.6, gain_db=-4, depth=.6)
    eq = {p.name: (100., 4200., 2., 2000.)}
    gain = np.float32(10**(p.gain_db/20))
    want = MX.pan_width(loud_signal*gain, p.pan, p.width)
    want = MX.depth_eq(want, p.depth)
    want = MX.highpass(want, 100.)
    b,a = signal.butter(2, 4200/(D.SR/2))
    want = signal.lfilter(b,a,want,axis=0).astype(np.float32)
    want += MX.highpass(want, 2000., order=2)*np.float32(10**(2/20)-1)
    got = loud_signal.copy()
    D._process_part(got, p, eq, [])
    np.testing.assert_allclose(got, want, atol=2e-7, rtol=0)
    bad = got.copy()
    bad[D.CHUNK:D.CHUNK+100] = 0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(bad, want, atol=2e-7, rtol=0)


def test_hall_overlap_add_preserves_chunk_tails(loud_signal):
    rng = np.random.default_rng(193)
    irs = [(rng.normal(size=1024)*np.exp(-np.arange(1024)/180)*.01).astype(np.float32)
           for _ in range(4)]
    want = MX.convolve_stereo(loud_signal, irs)
    got = np.zeros_like(loud_signal)
    D.convolve_add(got, loud_signal, irs, [])
    np.testing.assert_allclose(got, want, atol=5e-7, rtol=0)
    bad = got.copy()
    bad[D.CHUNK:D.CHUNK+1024] = 0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(bad, want, atol=5e-7, rtol=0)


def test_protected_pcm_and_join_never_read_a6(tmp_path):
    first = D.FRAME_N
    prefix = np.column_stack((np.arange(first)%73-36, np.arange(first)%97-48)).astype(np.float32)/2**18
    source = tmp_path/"source.wav"
    sf.write(source, np.vstack((prefix, np.full_like(prefix, .75))), D.SR, subtype="PCM_24")
    opening = D._protected_opening(source, frames=1)
    assert len(opening) == first
    target = np.full((2*first,2), -.1, np.float32)
    D._splice_opening(target, opening, first=first)
    out = tmp_path/"output.wav"
    sf.write(out, target, D.SR, subtype="PCM_24")
    assert D.pcm_sha256(source, 0, 1) == D.pcm_sha256(out, 0, 1)
    np.testing.assert_array_equal(target[first], opening[-1])
    np.testing.assert_array_equal(target[first+480:], np.full((first-480,2), -.1, np.float32))
    # Arbitrarily change A6, keeping the protected prefix: output cannot change.
    sf.write(source, np.vstack((prefix, np.full_like(prefix, -.8))), D.SR, subtype="PCM_24")
    other = np.full((2*first,2), -.1, np.float32)
    D._splice_opening(other, D._protected_opening(source, frames=1), first=first)
    np.testing.assert_array_equal(other, target)
    bad = target.copy()
    bad[first-1,0] += 2**-20
    sf.write(out, bad, D.SR, subtype="PCM_24")
    assert D.pcm_sha256(source, 0, 1) != D.pcm_sha256(out, 0, 1)


def test_silence_reapplied_after_dither_is_digital_zero():
    import verify_D as V
    source = np.full((8*D.FRAME_N,2), .2, np.float32)
    D._silence(source, [(2,4)])
    assert V.silence_row(source, 2, 4)["exact_zero"]
    assert np.all(source[:D.FRAME_N] == .2)
    dithered = MX.tpdf_dither_24(source, np.random.default_rng(0))
    assert not V.silence_row(dithered, 2, 4)["exact_zero"]
    D._silence(dithered, [(2,4)], fade=False)
    assert V.silence_row(dithered, 2, 4)["exact_zero"]
    dithered[3*D.FRAME_N,0] = 2**-23
    assert not V.silence_row(dithered, 2, 4)["exact_zero"]


def _tiny_score():
    from dsl import Part
    p = Part("fixture", "vla")
    p.n("D4", 72., .5, .3)
    return SimpleNamespace(used=lambda: {"fixture":p}, eq={}, groups={}, breaths=[],
                           push=[], fader=[], hard_silences=[(3400,3440)])


def test_master_rides_change_only_master_and_do_not_invalidate_premaster():
    score = _tiny_score()
    identity = D._score_identity(score)
    score.master_rides = [(4200,0.), (4240,-1.), (4720,-1.), (4760,0.)]
    assert D._score_identity(score) == identity
    score.master_rides[1] = (4240,-2.)
    assert D._score_identity(score) == identity
    # The corresponding PREMASTER ride belongs to the cache identity.
    score.fader = [(4200/24,0.), (4240/24,-1.), (4720/24,-1.), (4760/24,0.)]
    assert D._score_identity(score) != identity
    score.fader = []
    assert D._score_identity(score) == identity
    score.used()["fixture"].gain_db -= 1.
    assert D._score_identity(score) != identity


@pytest.mark.parametrize("change", ["note", "gain", "eq", "silence", "fader"])
def test_master_only_rejects_changed_score_before_loading_audio(tmp_path, monkeypatch, change):
    score = _tiny_score()
    before = D._score_identity(score)
    assert D._score_identity(score) == before
    if change == "note":
        score.used()["fixture"].notes[0].pitch += 1
    elif change == "gain":
        score.used()["fixture"].gain_db += 1
    elif change == "eq":
        score.eq["fixture"] = (100, None)
    elif change == "fader":
        score.fader = [(175.,0.), (176.,-1.)]
    else:
        score.hard_silences = [(3401,3440)]
    assert D._score_identity(score) != before
    (tmp_path/"premaster_score_D.identity.json").write_text(json.dumps({"score_identity":before}))
    monkeypatch.setattr(D, "_configure_sampler", lambda *a,**k: None)
    monkeypatch.setattr(D.shutil, "disk_usage", lambda *a: SimpleNamespace(free=2*D.MAX_DISK))
    monkeypatch.setitem(sys.modules, "score_v3_D", SimpleNamespace(build=lambda bm:score, check=lambda *a:[]))
    with pytest.raises(ValueError, match="premaster does not match"):
        D.render(tmp_path, master_only=True)
    assert not (tmp_path/"premaster_score_D.npy").exists()
    assert not (tmp_path/"score_D_draft.wav").exists()


@pytest.mark.parametrize("shared", ["music/out", "music/cache", "renders"])
def test_shared_output_protection_precedes_any_sampler_or_audio_work(shared, monkeypatch):
    def should_not_run(*a, **k):
        raise AssertionError("sampler invoked before private-path guard")
    monkeypatch.setattr(D, "_configure_sampler", should_not_run)
    for location in (D.ROOT/shared, D.ROOT/shared/"D-test-forbidden"):
        with pytest.raises(ValueError, match="private output directory required"):
            D.render(location)


def test_rss_guard_fails_at_the_limit(monkeypatch):
    monkeypatch.setattr(D, "peak_rss_bytes", lambda: D.MAX_RSS-1)
    assert D.memory_guard() == D.MAX_RSS-1
    monkeypatch.setattr(D, "peak_rss_bytes", lambda: D.MAX_RSS)
    with pytest.raises(MemoryError, match="3 GiB lane limit"):
        D.memory_guard()
