"""D's opt-in borrowed audio, own duration and cross-cut source selection; tiny synthetic assets only."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS
import deliver as DV
import edl_v3 as EDL


@pytest.fixture
def audio_env(monkeypatch, tmp_path):
    monkeypatch.setattr(AS, 'CACHE', str(tmp_path / 'cache'))
    monkeypatch.setitem(EDL.TOTAL, 'D', 24)
    monkeypatch.delenv('LD_D_AUDIO', raising=False)
    monkeypatch.setattr(AS, '_audio_candidates', lambda cut: ())
    return tmp_path


def mix(path, samples=12000, value=0.125):
    sf.write(path, np.full((samples, 2), value, np.float32), 48000, subtype='PCM_24')
    return str(path)


def assert_exact_silence(path, frames):
    audio, sr = sf.read(path, always_2d=True)
    assert sr == 48000
    assert audio.shape == (frames * 2000, 2), 'silence must fit D, independently of C'
    assert not np.any(audio), 'silence must contain no sync clicks or borrowed samples'


def test_default_and_explicit_silence_fit_d_and_reject_a_click(audio_env):
    path, label = AS.resolve_audio('D')
    assert label.startswith('STAND-IN silence')
    assert_exact_silence(path, 24)
    assert AS.resolve_audio('D', 'silence')[0] == path
    assert 'STAND-IN' in AS.resolve_audio_label('D')
    bad = mix(audio_env / 'click.wav', 48000)
    with pytest.raises(AssertionError, match='no sync clicks'):
        assert_exact_silence(bad, 24)
    with pytest.raises(AssertionError, match='fit D'):
        assert_exact_silence(path, 48)


def test_explicit_mix_pads_trims_and_precedes_environment(audio_env, monkeypatch):
    short = mix(audio_env / 'short.wav')
    long = mix(audio_env / 'long.wav', 96000, 0.25)
    monkeypatch.setenv('LD_D_AUDIO', long)
    path, label = AS.resolve_audio('D', short)
    audio, _ = sf.read(path, always_2d=True)
    assert audio.shape == (48000, 2)
    np.testing.assert_array_equal(audio[:12000], 0.125)
    assert not audio[12000:].any()
    assert label.startswith('STAND-IN mix') and 'short.wav' in label
    trimmed, label = AS.resolve_audio('D')
    audio, _ = sf.read(trimmed, always_2d=True)
    assert audio.shape == (48000, 2)
    np.testing.assert_array_equal(audio, 0.25)
    assert 'long.wav' in label
    assert_exact_silence(AS.resolve_audio('D', 'silence')[0], 24)


def test_cache_and_watcher_follow_mix_and_d_duration(audio_env, monkeypatch):
    source = mix(audio_env / 'source.wav')
    monkeypatch.setenv('LD_D_AUDIO', source)
    first = AS.resolve_audio('D')[0]
    signature = AS.audio_signature('D')
    assert AS.resolve_audio('D')[0] == first
    monkeypatch.setitem(EDL.TOTAL, 'D', 48)
    second = AS.resolve_audio('D')[0]
    assert second != first and AS.audio_signature('D') != signature
    assert sf.info(second).frames == 96000
    signature = AS.audio_signature('D')
    mix(Path(source), 12001, 0.375)
    third = AS.resolve_audio('D')[0]
    assert third != second and AS.audio_signature('D') != signature
    assert Path(third).parent == audio_env / 'cache'


def test_bad_requested_mix_fails_without_falling_back(audio_env, monkeypatch):
    monkeypatch.setenv('LD_D_AUDIO', str(audio_env / 'absent.wav'))
    with pytest.raises(FileNotFoundError, match='does not exist'):
        AS.resolve_audio('D')
    with pytest.raises(ValueError, match='only enabled for cut D'):
        AS.standin_audio('C')
    with pytest.raises(ValueError, match='only enabled for cut D'):
        AS.resolve_audio('A', 'silence')


@pytest.mark.parametrize('failure', ['wrong_duration', 'moving_source'])
def test_failed_normalization_never_publishes_a_cached_mix(audio_env, monkeypatch, failure):
    source = mix(audio_env / 'source.wav')

    def incomplete_or_racing_encode(cmd, **kwargs):
        mix(Path(cmd[-1]), 48001 if failure == 'wrong_duration' else 48000)
        if failure == 'moving_source':
            mix(Path(source), 12001)

    monkeypatch.setattr(AS.subprocess, 'run', incomplete_or_racing_encode)
    if failure == 'wrong_duration':
        with pytest.raises(ValueError, match='declared cut length'):
            AS.standin_audio('D', source)
    else:
        with pytest.raises(RuntimeError, match='changed during normalization'):
            AS.standin_audio('D', source)
    assert not list((audio_env / 'cache').iterdir())


def test_real_d_master_requires_exact_d_length(audio_env, monkeypatch):
    master = mix(audio_env / 'final_D.wav', 48000)
    monkeypatch.setattr(AS, '_audio_candidates', lambda cut: ((master, 'COMPOSER master'),))
    assert AS.resolve_audio('D')[0] == master
    assert 'STAND-IN' not in AS.resolve_audio('D')[1]
    mix(Path(master), 48001)
    assert not AS.audio_fits(master, 'D')
    path, label = AS.resolve_audio('D')
    assert path != master and label.startswith('STAND-IN silence')
    assert_exact_silence(path, 24)


@pytest.mark.parametrize('cut', 'ABC')
def test_d_environment_does_not_change_accepted_cut_audio(cut, audio_env, monkeypatch):
    monkeypatch.setitem(EDL.TOTAL, cut, 24)
    original = mix(audio_env / f'original_{cut}.wav', EDL.TOTAL[cut] * 2000)
    monkeypatch.setattr(AS, '_audio_candidates', lambda cut: ((original, 'original master'),))
    before = (AS.audio_choice(cut), AS.audio_signature(cut), AS.resolve_audio_label(cut), AS.resolve_audio(cut))
    monkeypatch.setenv('LD_D_AUDIO', str(audio_env / 'missing-D-override.wav'))
    after = (AS.audio_choice(cut), AS.audio_signature(cut), AS.resolve_audio_label(cut), AS.resolve_audio(cut))
    assert before == after


def test_offsets_reuse_a_and_c_and_pick_one_take_for_whole_shot(monkeypatch):
    folders = {
        'embers_A3': {f: f'A/{f}.jpg' for f in range(1200, 1204)},
        'book_C5': {f: f'C/{f}.jpg' for f in range(400, 404)},
    }
    monkeypatch.setattr(AS, 'index', lambda path: folders.get(os.path.basename(path), {}))
    a = EDL.T('embers_A3', off=1100, mode='exact')
    c = EDL.T('book_C5', off=300, mode='exact')
    shot = EDL.S('D_TEST', 100, 104, 'test', 'test', 'EDIT', '', takes=[a, c])
    plan = AS.plan_shot(shot, 'D', None)
    assert plan['take'] is a
    assert [AS.locate(plan['take'], 'D', None, f)[0] for f in range(100, 104)] == [
        'A/1200.jpg', 'A/1201.jpg', 'A/1202.jpg', 'A/1203.jpg']
    del folders['embers_A3'][1203]
    plan = AS.plan_shot(shot, 'D', None)
    assert plan['take'] is c and plan['have'] == 4
    assert AS.locate(plan['take'], 'D', None, 100)[0] == 'C/400.jpg'
    del folders['book_C5'][400]
    plan = AS.plan_shot(shot, 'D', None)
    assert plan['take'] is a and plan['have'] == 3
    # C has the missing frame, but a partial shot keeps one take and leaves the missing frame for a slate.
    assert AS.locate(c, 'D', None, 103)[0] == 'C/403.jpg'
    assert AS.locate(plan['take'], 'D', None, 103)[0] is None
    wrong_offset = dict(a, off=0)
    assert AS.plan_shot(dict(shot, takes=[wrong_offset]), 'D', None)['kind'] == 'slate'


def test_qc_marks_standin_even_if_audio_peaks_and_loudness_pass(audio_env, monkeypatch):
    total = EDL.TOTAL['D']
    mov = audio_env / 'D.mov'
    mov.write_bytes(b'synthetic media')
    monkeypatch.setattr(DV, 'ff_info', lambda path: ('h264 yuv420p 1920x804 24 fps bt709', '48000 Hz stereo'))
    monkeypatch.setattr(DV, 'count_video_frames', lambda path: total)
    monkeypatch.setattr(DV, 'decode_audio', lambda path: np.full((total * 2000, 2), 0.01, np.float32))
    monkeypatch.setattr(DV, 'true_peak_db', lambda x: -2.0)
    monkeypatch.setattr(DV, 'picture_stats', lambda path, n: (np.full(n, 0.5), np.zeros((n, 1)), n))
    monkeypatch.setattr(DV, 'flashes', lambda grid: (0, None))
    monkeypatch.setattr(DV, 'status_frames', lambda *args: (0, [], [], 0, []))
    monkeypatch.setattr(DV, 'provisional_frames', lambda *args: (0, []))
    monkeypatch.setitem(sys.modules, 'pyloudnorm', SimpleNamespace(Meter=lambda sr: SimpleNamespace(
        integrated_loudness=lambda x: -16.0)))
    build = dict(segments=1, encoded=0, frames_encoded=0, seconds=0)
    txt, result = DV.qc('D', None, str(mov), None, 'STAND-IN mix', build)
    assert result['result'] == 'WARN' and result['audio_stand_in']
    assert '[WARN] sound source: STAND-IN mix' in txt
    _, finished = DV.qc('D', None, str(mov), None, 'COMPOSER master', build)
    assert finished['result'] == 'PASS' and not finished['audio_stand_in']
    _, unknown = DV.qc('D', None, str(mov), None, DV.recorded_audio_label(str(mov)), build)
    assert unknown['result'] == 'WARN' and unknown['audio_stand_in'] is None


def test_qc_only_audio_receipt_rejects_missing_or_replaced_media(audio_env):
    mov = audio_env / 'D.mov'
    mov.write_bytes(b'first mux')
    assert 'UNKNOWN' in DV.recorded_audio_label(str(mov))
    DV.write_audio_receipt(str(mov), 'STAND-IN borrowed C mix')
    assert DV.recorded_audio_label(str(mov)) == 'STAND-IN borrowed C mix'
    mov.write_bytes(b'a replacement mux')
    assert 'UNKNOWN' in DV.recorded_audio_label(str(mov))


def test_offset_under_keeps_rgb_matte_background_and_delivery_dependencies_together(monkeypatch):
    folders = {'foreground_C': {1100: 'rgb'}, 'matte_C': {1100: 'matte'},
               'background_C': {100: 'same', 1100: 'offset', 1679: 'held'},
               'background_C_half': {1100: 'half'}}
    values = {'rgb': 0.0, 'matte': 0.0, 'same': 0.125, 'offset': 0.25, 'held': 0.5, 'half': 0.375}
    reads = []
    monkeypatch.setattr(AS, 'index', lambda path: folders.get(os.path.basename(path), {}))
    monkeypatch.setattr(DV, '_stat', lambda path: path)
    monkeypatch.setitem(EDL.UNDER_FINAL_ELIGIBILITY, 'background_C_half', False)
    ctx = AS.Ctx.__new__(AS.Ctx)
    ctx.cut, ctx.variant = 'D', None

    def read(path, crop=None, gray=False):
        reads.append(path)
        return np.full((2, 3) if gray else (2, 3, 3), values[path], np.float32)

    ctx.read = read

    def check(spec, expected_path, expected_value):
        take = EDL.T('foreground_C', 1000, 'exact', matte='matte_C', under=spec)
        reads.clear()
        picture = ctx.take_frame(take, 100)
        np.testing.assert_array_equal(picture, expected_value)
        assert reads == ['rgb', 'matte', expected_path]
        assert AS.locate_under(spec, 100)[0] == expected_path
        assert DV.frame_sources('D', None, dict(kind='take', take=take), 100) == ['rgb', 'matte', expected_path]
        return take

    # Existing same/hold deliberately keep their original frame domains; off never silently changes them.
    check(('same', 'background_C'), 'same', 0.125)
    check(('hold', 'background_C', 1679), 'held', 0.5)
    check(('offset', 'background_C', 1000), 'offset', 0.25)
    del folders['background_C'][1100]
    take = check(('offset', 'background_C', 1000), 'half', 0.375)
    assert AS.provisional_sources(take, 'D', None, 100) == ('under:background_C_half',)
    # An off-by-one offset cannot borrow the timeline's same-frame background or its adjacent source frame.
    bad = ('offset', 'background_C', 999)
    assert AS.locate_under(bad, 100) == (None, None)
    assert ctx.under(bad, 100) is None
    bad_take = dict(take, under=bad)
    assert DV.frame_sources('D', None, dict(kind='take', take=bad_take), 100)[-1] == 'under:none'
    np.testing.assert_array_equal(ctx.take_frame(bad_take, 100), 0)


@pytest.mark.parametrize('spec', [('offset', 'source'), ('offset', 'source', 0.5), ('offset', 'source', 1, 2)])
def test_under_offset_requires_explicit_integer_in_renderer_and_metadata(spec):
    ctx = AS.Ctx.__new__(AS.Ctx)
    for lookup in (AS.locate_under, ctx.under):
        with pytest.raises(ValueError, match='integer frame offset'):
            lookup(spec, 100)


@pytest.mark.parametrize('partial', [False, True])
def test_missing_d_title_is_a_slate_and_ab_title_proxies_survive(monkeypatch, partial):
    shot = EDL.S('D_TITLE', 100, 102, 'title', 'TITLE', 'EDIT', '',
                 takes=[EDL.T('borrowed_A_title', mode='exact')], kind='title')
    frames = {100: 'present'} if partial else {}
    monkeypatch.setattr(AS, 'index', lambda path: frames)
    monkeypatch.setitem(EDL.EDL, 'D', [shot])
    monkeypatch.setitem(EDL.TRANS, 'D', [])
    plan = AS.plan_shot(shot, 'D', None)
    assert plan['kind'] == ('take' if partial else 'slate')
    assert DV.status_frames('D', None)[0] == (1 if partial else 2)
    assert DV.status_frames('D', None)[3] == 0
    ctx = AS.Ctx.__new__(AS.Ctx)
    ctx.cut, ctx.variant, ctx.W, ctx.H = 'D', None, 3, 2
    ctx.shots, ctx.plans = [shot], [plan]
    ctx.take_frame = lambda take, frame: None
    slate = np.full((2, 3, 3), 0.125, np.float32)
    ctx.slate = lambda index, reason: slate.copy()
    ctx.ember = lambda image, frame: (image, False)
    monkeypatch.setattr(AS, 'draw_slate_clock', lambda *args: None)

    def forbidden_sky(*args):
        raise AssertionError('D has no approved sky proxy')

    monkeypatch.setattr(AS.TS, 'standin_sky', forbidden_sky)
    picture, _, status, src = ctx.picture(101)
    np.testing.assert_array_equal(picture, slate)
    assert status.startswith('SLATE') and src is None
    if not partial:
        for cut in 'AB':
            assert AS.plan_shot(shot, cut, None)['kind'] == 'titlesky'
    else:
        # Both A/B retain their gap proxy behavior; actual picture dispatch must still call the sky renderer.
        for cut in 'AB':
            ctx.cut = cut
            with pytest.raises(AssertionError, match='no approved sky proxy'):
                ctx.picture(101)
