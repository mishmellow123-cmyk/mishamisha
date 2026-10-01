"""Synthetic D renderer helper tests; no assets, full render, or shared writes."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import render_D_effects as R


def test_score_contract_requires_verified_matching_bytes_and_all_bands(tmp_path):
    wav = tmp_path/'synthetic.wav'
    wav.write_bytes(b'Synthetic identity fixture; not audio')
    contract = tmp_path/'verification.json'
    doc = dict(ok=True, wav_sha256=R.sha(wav), checks=[
        dict(check='level band', section=f'D{i:02d}', band_lu=[-20, -5], ok=True)
        for i in range(1, 35)])
    contract.write_text(json.dumps(doc))
    assert len(R.load_contract(contract, wav)[0]) == 34
    for mutation in ('failed', 'missing_status', 'failed_check', 'hash', 'missing_band'):
        changed = json.loads(json.dumps(doc))
        if mutation == 'failed': changed['ok'] = False
        elif mutation == 'missing_status': changed.pop('ok')
        elif mutation == 'failed_check': changed['checks'][0]['ok'] = False
        elif mutation == 'hash': changed['wav_sha256'] = '0'*64
        else: changed['checks'].pop()
        contract.write_text(json.dumps(changed))
        with pytest.raises(ValueError): R.load_contract(contract, wav)


def test_delivery_provenance_preserves_boundary_censored_native_scope():
    row=dict(source='measured',picture_frame=3440,measurement_scope='native_delivered_plate',
             measured_ref='actual.json',measurement=dict(claim='contact',onset_measured=False,boundary_censored=True),
             inherited_measurement_scope='source-mask')
    result=R.picture_provenance(row)
    assert result['measured_picture'] is True
    assert result['measurement']==row['measurement']
    assert result['inherited_measurement_scope']=='source-mask'
    result['measurement']['onset_measured']=True
    assert row['measurement']['onset_measured'] is False
    assert R.picture_provenance(dict(source='derived',measurement_scope='cover-grid')) == dict(measurement_scope='cover-grid',measured_picture=False)
    assert R.picture_provenance(dict(source='est.'))==dict(measured_picture=False)


def small_blocks(monkeypatch, size):
    def chunks(n, size=size):
        for a in range(0, n, size):
            yield a, min(n, a+size)
    monkeypatch.setattr(R, "blocks", chunks)


@pytest.mark.parametrize("offset,stop,expected", [
    (-2, None, [3, 4, 5, 0, 0, 0]),
    (4, None, [0, 0, 0, 0, 1, 2]),
    (2, 4, [0, 0, 1, 2, 0, 0]),
    (-2, 2, [3, 4, 0, 0, 0, 0]),
    (7, None, [0, 0, 0, 0, 0, 0]),
    (2, 1, [0, 0, 0, 0, 0, 0]),
])
def test_put_clips_both_ends_and_obeys_exclusive_stop(offset, stop, expected):
    y = np.repeat(np.arange(1, 6, dtype=np.float32)[:, None], 2, axis=1)
    stem = np.zeros((6, 2), np.float32)
    R.put(stem, y, offset, stop=stop)
    np.testing.assert_array_equal(stem[:, 0], expected)
    np.testing.assert_array_equal(stem[:, 1], expected)
    # Placement adds to the destination; a second cue must not overwrite it.
    R.put(stem, y, offset, stop=stop)
    np.testing.assert_array_equal(stem[:, 0], 2*np.asarray(expected))


def test_effective_stop_controls_expanded_request_tails():
    row = {"id": "expanded.3", "request_id": "vision.riser", "stop_f": 25}
    controls = [
        {"action": "cut_existing_layers", "targets": ["vision.riser"], "hit_f": 20},
        {"action": "cut_existing_layers", "targets": ["other.request"], "hit_f": 3},
        {"action": "different_action", "targets": ["vision.riser"], "hit_f": 1},
    ]
    stop = R.effective_stop(row, controls)
    assert stop == 20
    stem = np.zeros((40, 2), np.float32)
    wet_tail = np.ones((35, 2), np.float32)
    R.put(stem, wet_tail, 5, stop=stop)
    assert np.all(stem[5:20] == 1)
    assert np.count_nonzero(stem[20:]) == 0
    # Removing the targeting control reveals the event's own later boundary.
    assert R.effective_stop(row, controls[1:]) == 25
    assert R.effective_stop(dict(row, stop_f=10), controls) == 10
    assert R.effective_stop(dict(row, stop_f=None), controls) == 20
    assert R.effective_stop(dict(row, stop_f=None), controls[1:]) is None
    assert R.effective_stop({"request_id": "vision.riser"}, controls) == 20


def test_hard_zero_releases_one_lsb_before_the_cut_and_preserves_after():
    lsb = np.float32(2**-23)
    x = np.full((20, 2), lsb, np.float32)
    R.hard_zero(x, 10, 13, release=4)
    np.testing.assert_array_equal(x[:6], lsb)
    np.testing.assert_allclose(x[6:10, 0], lsb*np.array([1, 2/3, 1/3, 0]), atol=1e-14, rtol=0)
    assert x[8, 0] > 0 and x[9, 0] == 0
    assert np.count_nonzero(x[10:13]) == 0
    np.testing.assert_array_equal(x[13:], lsb)
    # Without a release, the last pre-cut sample remains nonzero.
    control = np.full_like(x, lsb)
    R.hard_zero(control, 10, 13, release=0)
    assert control[9, 0] == lsb
    assert np.count_nonzero(control[10:13]) == 0


def test_hard_zero_release_at_start_cannot_wrap_to_the_tail():
    x = np.ones((8, 2), np.float32)
    R.hard_zero(x, 2, 4, release=480)
    np.testing.assert_array_equal(x[:, 0], [1, 0, 0, 0, 1, 1, 1, 1])
    R.hard_zero(x, 0, 2)
    np.testing.assert_array_equal(x[:, 0], [0, 0, 0, 0, 1, 1, 1, 1])


def peak4(x):
    return float(np.abs(signal.resample_poly(np.asarray(x, np.float64), 4, 1, axis=0)).max())


def test_linked_limit_protects_prefix_and_controls_processing_boundaries(monkeypatch):
    monkeypatch.setattr(R, "SR", 2000)
    monkeypatch.setattr(R, "FR", 1)
    monkeypatch.setattr(R, "N", 7440)
    small_blocks(monkeypatch, 503)
    t = np.arange(R.N)/R.SR
    mono = .08*np.sin(2*np.pi*173*t)
    # Peaks on both sides of two processing boundaries and the suffix entrance.
    for i in (1440, 1440+502, 1440+503, 1440+2*503-1, 1440+2*503):
        mono[i:i+3] = [1.5, -1.7, 1.2]
    mix = np.column_stack((mono, -.6*mono)).astype(np.float32)
    score, sfx = .625*mix, .375*mix
    score_before, sfx_before, mix_before = score.copy(), sfx.copy(), mix.copy()
    assert peak4(mix) > 10**(-1.35/20)
    reduction = R.linked_limit(score, sfx, mix)
    assert reduction < 0
    np.testing.assert_array_equal(score[:1440], score_before[:1440])
    np.testing.assert_array_equal(sfx[:1440], sfx_before[:1440])
    np.testing.assert_array_equal(mix, mix_before)
    # Both stems and both channels receive one common envelope.
    np.testing.assert_allclose(score*.375, sfx*.625, atol=2e-8, rtol=0)
    np.testing.assert_allclose(score[:, 1], -.6*score[:, 0], atol=3e-8, rtol=0)
    limited = score+sfx
    assert peak4(limited) <= 10**(-1/20)
    # Ordinary quiet material remains audible; limiting cannot pass by muting it.
    assert np.max(np.abs(limited[-1000:])) > .05
    assert np.max(np.abs(limited[1440:])) < np.max(np.abs(mix_before[1440:]))


def test_linked_limit_below_ceiling_is_identity(monkeypatch):
    monkeypatch.setattr(R, "SR", 1000)
    monkeypatch.setattr(R, "FR", 1)
    monkeypatch.setattr(R, "N", 2600)
    small_blocks(monkeypatch, 257)
    mix = np.full((R.N, 2), .05, np.float32)
    score, sfx = mix*.7, mix*.3
    before = (score.copy(), sfx.copy())
    assert R.linked_limit(score, sfx, mix) == pytest.approx(0, abs=1e-10)
    np.testing.assert_array_equal(score, before[0])
    np.testing.assert_array_equal(sfx, before[1])


def test_linked_limit_constrains_individual_stems_when_mix_cancels(monkeypatch):
    monkeypatch.setattr(R, "SR", 2000)
    monkeypatch.setattr(R, "FR", 1)
    monkeypatch.setattr(R, "N", 7440)
    small_blocks(monkeypatch, 503)
    t = np.arange(R.N)/R.SR
    source = (1.6*np.sin(2*np.pi*173*t)).astype(np.float32)
    source[:1440] *= .02
    score = np.column_stack((source, -.6*source))
    sfx = -.99*score
    mix = score+sfx
    before = (score.copy(), sfx.copy())
    assert peak4(mix) < .1
    assert peak4(score) > 1
    assert peak4(sfx) > 1
    assert R.linked_limit(score, sfx, mix) < 0
    for stem, original in zip((score, sfx), before):
        np.testing.assert_array_equal(stem[:1440], original[:1440])
        assert peak4(stem) < 10**(-1/20)
        assert np.max(np.abs(stem[1440:])) > .5
    np.testing.assert_allclose(sfx, -.99*score, atol=1e-7, rtol=0)
    assert peak4(score+sfx) < .1


def test_suffix_gain_preserves_prefix_and_links_stems(monkeypatch):
    monkeypatch.setattr(R, "FR", 1)
    monkeypatch.setattr(R, "N", 1700)
    small_blocks(monkeypatch, 71)
    score = np.full((R.N, 2), .2, np.float32)
    sfx = np.full((R.N, 2), -.05, np.float32)
    R.gain_suffix(score, sfx, 20*np.log10(2))
    np.testing.assert_array_equal(score[:1440], np.float32(.2))
    np.testing.assert_array_equal(sfx[:1440], np.float32(-.05))
    np.testing.assert_array_equal(score[1440:], np.float32(.4))
    np.testing.assert_array_equal(sfx[1440:], np.float32(-.1))


def test_constraints_cover_full_voice_and_leave_hearth_effects(monkeypatch):
    monkeypatch.setattr(R, "FR", 1)
    monkeypatch.setattr(R, "SR", 1000)
    monkeypatch.setattr(R, "N", 9200)
    score = np.full((R.N, 2), .4, np.float32)
    sfx = np.full((R.N, 2), .6, np.float32)
    # A peak after the exact void catches a ceiling applied only to the void.
    sfx[8879, 1] = 1.
    receipt = R.constraints_inplace(sfx, score)
    assert receipt["voice_effect_gain_db"] < 0
    assert np.max(np.abs(sfx[8640:8880])) <= 10**(-45/20)+1e-9
    for x in (sfx, score):
        assert np.count_nonzero(x[3400:3440]) == 0
        assert np.count_nonzero(x[8660:8740]) == 0
        np.testing.assert_array_equal(x[:1440], np.float32(.6 if x is sfx else .4))
    assert np.count_nonzero(score[3440:3520]) == 0
    assert np.count_nonzero(score[9120:]) == 0
    assert sfx[3440, 0] > 0
    assert sfx[9120, 0] > 0
    assert sfx[-1, 0] == 0
    assert sfx[8879, 1] > 0
    # A quiet voice bed should not be amplified by the cap.
    quiet = np.full((R.N, 2), .001, np.float32)
    result = R.constraints_inplace(quiet)
    assert result["voice_effect_gain_db"] == pytest.approx(0)
    assert np.max(np.abs(quiet[8640:8880])) <= .001000001


def pcm(path, count=None):
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes() if count is None else count)


def test_write_final_copies_packed_prefix_and_masks_dither_at_silence(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FR", 1)
    monkeypatch.setattr(R, "N", 1700)
    small_blocks(monkeypatch, 113)
    donor = tmp_path/"donor.wav"
    target = tmp_path/"final.wav"
    # Exact 24-bit integer values exercise both sign and low PCM bits.
    integers = np.random.default_rng(42).integers(-(2**23), 2**23, (R.N, 2), dtype=np.int32)
    sf.write(donor, integers*256, R.SR, subtype="PCM_24")
    x = np.zeros((R.N, 2), np.float32)
    R.write_final(target, x, donor, [(1499, 1537)])
    info = sf.info(target)
    assert (info.frames, info.channels, info.samplerate, info.subtype) == (R.N, 2, R.SR, "PCM_24")
    assert pcm(target, 1440) == pcm(donor, 1440)
    assert any(pcm(donor, 1440))  # zero-output cannot satisfy identity accidentally
    decoded, _ = sf.read(target, dtype="float32", always_2d=True)
    assert np.count_nonzero(decoded[1499:1537]) == 0
    assert np.count_nonzero(decoded[-1]) == 0
    assert np.count_nonzero(decoded[1440:1499]) > 0  # dither exists outside masks
    assert np.count_nonzero(x) == 0  # writer does not dither its source array
    assert pcm(target)[1499*6:1537*6] == bytes((1537-1499)*6)


def test_read_into_respects_partial_range_and_does_not_rescale_pcm(tmp_path, monkeypatch):
    small_blocks(monkeypatch, 17)
    donor = tmp_path/"donor.wav"
    integers = np.arange(200, dtype=np.int32).reshape(100, 2)*256
    sf.write(donor, integers, R.SR, subtype="PCM_24")
    target = np.full((100, 2), -99., np.float32)
    R.read_into(donor, target, start=13, stop=79)
    decoded, _ = sf.read(donor, dtype="float32", always_2d=True)
    np.testing.assert_array_equal(target[13:79], decoded[13:79])
    np.testing.assert_array_equal(target[:13], -99)
    np.testing.assert_array_equal(target[79:], -99)


@pytest.mark.parametrize("frames,channels,rate", [(19, 2, 48000), (20, 1, 48000), (20, 2, 44100)])
def test_read_into_rejects_short_or_wrong_format_donors(tmp_path, frames, channels, rate):
    donor = tmp_path/"bad.wav"
    sf.write(donor, np.zeros((frames, channels)), rate, subtype="PCM_24")
    target = np.full((20, 2), -99., np.float32)
    with pytest.raises(ValueError, match="Invalid donor"):
        R.read_into(donor, target)
    np.testing.assert_array_equal(target, -99)


def synthetic_sound():
    """Finite synthetic DSP adapter; imports no asset/render/cache modules."""
    return SimpleNamespace(
        event=lambda rc, rng: (np.ones((12, 2), np.float32), 3),
        rng_for=lambda seed: seed,
        # Deliberately leaves stereo leakage, as a source-space panner can.
        pan=lambda y, p: y.copy(),
        limit_crest=lambda y, crest: y,
        match_gain=lambda y, level, kind, trim: 0.,
        db=lambda value: 10**(np.asarray(value)/20),
        proc=lambda y, **kwargs: y,
    )


@pytest.mark.parametrize("pan,opposite", [(-1, 1), (1, 0)])
def test_hard_pan_clamps_dry_opposite_channel_but_preserves_wet(monkeypatch, pan, opposite):
    sound = synthetic_sound()
    convolved = []
    def wet(y, ir):
        convolved.append(y.copy())
        return np.ones_like(y)
    sound.convolve = wet
    monkeypatch.setattr(R, "readonly_space", lambda name: [np.ones(4), np.ones(4)])
    result, hit = R.event_clip({"pan": pan, "level": -20, "send": .5}, "synthetic", {}, sound)
    assert hit == 3
    assert np.count_nonzero(convolved[0][:, opposite]) == 0
    np.testing.assert_array_equal(convolved[0][:12, 1-opposite], 1.)
    # The post-pan wet path may cross channels; clamping the final output would fail.
    np.testing.assert_array_equal(result[:12, opposite], .5)
    np.testing.assert_array_equal(result[:12, 1-opposite], 1.5)
    ordinary, _ = R.event_clip({"pan": .5, "level": -20, "send": 0}, "synthetic", {}, sound)
    assert np.all(ordinary[:, opposite] > 0)


@pytest.mark.parametrize("no_breath", [False, True])
def test_donor_event_uses_source_breath_offset_and_preserves_wet_history(monkeypatch, no_breath):
    sound = synthetic_sound()
    calls, forwarded = [], {}
    windows = [(100, 105, "leave-room")]
    dry_env = np.full(12, .25, np.float32)
    dry_env[3:5] = 0
    def breath_env(n, breaths, offset):
        calls.append((n, breaths, offset))
        return dry_env
    sound.R2 = SimpleNamespace(breath_windows=lambda beats: windows, breath_env=breath_env)
    class BarMap:
        def __init__(self, cut):
            assert cut == "AP2"
        def breath_beats(self):
            return ["synthetic beat"]
    monkeypatch.setitem(sys.modules, "timeline_v3", SimpleNamespace(BarMap=BarMap))
    monkeypatch.setattr(R, "FR", 2)
    def spatial(y, rc, space, sound, **kwargs):
        forwarded.update(kwargs)
        return y
    monkeypatch.setattr(R, "spatial_and_tail", spatial)
    rc = {"level": -20, "send": .1, "no_breath": no_breath}
    y, hit = R.event_clip(rc, "synthetic", {}, sound, donor_cut="AP2", donor_hit_f=51)
    assert hit == 3
    assert forwarded == {"breaths": windows, "source_offset": 99}
    if no_breath:
        assert calls == []
        np.testing.assert_array_equal(y, 1.)
    else:
        assert calls == [(12, windows, 99)]
        np.testing.assert_array_equal(y[:, 0], dry_env)
        np.testing.assert_array_equal(y[:, 1], dry_env)


def test_wet_breath_windows_clip_in_source_coordinates_and_skip_nonoverlap(monkeypatch):
    sound = synthetic_sound()
    segmented, ordinary = [], []
    def with_breaths(y, ir, windows, n):
        segmented.append((windows, n))
        return np.full_like(y, .25)
    def convolve(y, ir):
        ordinary.append(len(y))
        return np.full_like(y, .75)
    sound.R2 = SimpleNamespace(convolve_with_breaths=with_breaths)
    sound.convolve = convolve
    monkeypatch.setattr(R, "readonly_space", lambda name: [np.ones(4), np.ones(4)])
    breaths = [(98, 102, "a"), (110, 120, "b"), (90, 100, "c"), (116, 120, "d")]
    y = np.ones((12, 2), np.float32)
    result = R.spatial_and_tail(y, {"send": .5}, {}, sound, breaths=breaths, source_offset=100)
    assert segmented == [([(0, 2, "a"), (10, 16, "b")], 16)]
    assert ordinary == []
    np.testing.assert_array_equal(result[:12], 1.125)
    control = R.spatial_and_tail(y, {"send": .5}, {}, sound, breaths=breaths, source_offset=200)
    assert ordinary == [16]
    np.testing.assert_array_equal(control[:12], 1.375)


def test_donor_gain_filters_at_8hz_before_saved_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", tmp_path)
    small_blocks(monkeypatch, 13)
    cache = tmp_path/"music/cache/v3"
    cache.mkdir(parents=True)
    envelope = np.linspace(.1, 1.8, 64).astype(np.float32)
    np.save(cache/"master_env_sound_AP2.npy", envelope)
    t = np.arange(80)/R.SR
    original = np.column_stack((.5+np.sin(2*np.pi*10*t), .3+np.cos(2*np.pi*150*t))).astype(np.float32)
    indices = np.clip(np.arange(len(original))-3, 0, len(envelope)-1)
    sos = signal.butter(1, 8, "high", fs=R.SR, output="sos")
    expected = signal.sosfilt(sos, original, axis=0).astype(np.float32)*envelope[indices, None]
    y = original.copy()
    returned = R.donor_gain(y, -3, "AP2")
    assert returned is y
    np.testing.assert_array_equal(y, expected)
    assert not np.allclose(y, original*envelope[indices, None], atol=1e-6, rtol=0)
    wrong_order = signal.sosfilt(sos, original*envelope[indices, None], axis=0)
    assert not np.allclose(y, wrong_order, atol=1e-6, rtol=0)


@pytest.mark.parametrize("same_source", [False, True])
def test_rendered_crown_pair_records_distinct_combined_channels(tmp_path, monkeypatch, same_source):
    """Exercise the real event/pan/placement/receipt path with two tiny sources."""
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "FR", 1)
    cache = tmp_path/"music/cache/v3"
    cache.mkdir(parents=True)
    np.save(cache/"master_env_sound_C5P2.npy", np.ones(1, np.float32))
    sound = synthetic_sound()
    sound._LOADED = {}
    sources = {"synthetic-left": np.linspace(-.5, .5, 12, dtype=np.float32),
               "synthetic-right": np.sin(np.arange(12)).astype(np.float32)*.4}
    def event(rc, seed):
        source = sources[rc["layers"][0]["src"]]
        return np.column_stack((source, source)), 3
    sound.event = event
    monkeypatch.setitem(sys.modules, "sound_v3", sound)
    monkeypatch.setitem(sys.modules, "sound_d_designs", SimpleNamespace(
        build=lambda name: {"level": 0, "send": 0, "layers": [{"src": "synthetic-left"}]}))
    monkeypatch.setitem(sys.modules, "sound_d_foley", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sound_recipes_C", SimpleNamespace(SPACE={}))
    rows = []
    for side, pan in (("left", -1), ("right", 1)):
        source = "synthetic-left" if same_source or side == "left" else "synthetic-right"
        rows.append(dict(id="crown."+side, request_id="D.new.crowns."+side,
                         design="fixture", hit_f=4320, pan=pan, dist=0,
                         recipe_overrides={"layers": [{"src": source}]},
                         source="synthetic test fixture", hook="synthetic"))
    binding = dict(reuse_plan={"EXTRA_EVENTS": [], "BED_CROPS": [], "RECIPES": {}},
                   new_events=rows, new_beds=[], controls=[])
    receipt = dict(events=[], beds=[])
    stem = np.zeros((4400, 2), np.float32)
    if same_source:
        with pytest.raises(ValueError, match="Identical simultaneous crowns"):
            R.render_effects(stem, binding, receipt)
        assert receipt["crown_pair"]["max_channel_difference"] == 0
        np.testing.assert_array_equal(stem[:, 0], stem[:, 1])
        return
    assert R.render_effects(stem, binding, receipt) is stem
    np.testing.assert_array_equal(stem[4317:4329, 0], sources["synthetic-left"])
    np.testing.assert_array_equal(stem[4317:4329, 1], sources["synthetic-right"])
    pair = receipt["crown_pair"]
    assert pair["source_hit_offsets"] == [3, 3]
    assert pair["left_hit_sample"] == pair["right_hit_sample"] == 4320
    assert pair["compared_samples"] == 12
    expected_difference = np.max(np.abs(sources["synthetic-left"]-sources["synthetic-right"]))
    assert pair["max_channel_difference"] == float(expected_difference)
    assert np.isfinite(pair["correlation"])
    left, right = receipt["events"]
    assert left["isolated_channel_peaks"][1] == right["isolated_channel_peaks"][0] == 0
    assert left["source_recipe"]["layers"] != right["source_recipe"]["layers"]


def test_resident_audio_is_explicit_and_preserves_float32_processing(tmp_path):
    n = 9600
    resident = R.allocate_audio(tmp_path/'resident.f32', n, resident=True)
    disk = R.allocate_audio(tmp_path/'disk.f32', n)
    try:
        assert not isinstance(resident, np.memmap) and not (tmp_path/'resident.f32').exists()
        assert isinstance(disk, np.memmap) and (tmp_path/'disk.f32').exists()
        signal = np.column_stack((np.linspace(-.2, .3, n), np.linspace(.1, -.15, n))).astype(np.float32)
        resident[:] = disk[:] = signal
        R.hard_zero(resident, 20, 40); R.hard_zero(disk, 20, 40)
        np.testing.assert_array_equal(resident, disk)
        sf.write(tmp_path/'resident.wav', resident, R.SR, subtype='PCM_24')
        sf.write(tmp_path/'disk.wav', disk, R.SR, subtype='PCM_24')
        assert (tmp_path/'resident.wav').read_bytes() == (tmp_path/'disk.wav').read_bytes()
        resident[0] = 0
        assert not np.array_equal(resident, disk)
    finally:
        disk._mmap.close()
    with pytest.raises(ValueError, match='explicitly boolean'):
        R.allocate_audio(tmp_path/'bad.f32', n, resident='false')


def test_authored_gain_is_after_processing_and_absent_path_is_bit_identical():
    x = np.full((4*R.FR, 2), .25, np.float32)
    before = x.tobytes()
    assert R.post_gain(x, {}, 1440*R.FR) is x and x.tobytes() == before
    R.post_gain(x, {'post_gain_f':[[1440,-12],[1444,0]]}, 1440*R.FR)
    assert x[0,0] == pytest.approx(.25*10**(-12/20))
    assert .24 < x[-1,0] < .25
    assert x.tobytes() != before
    for points in ([[1440,0],[1440,-6]], [[1440,float('nan')],[1444,0]], [[1440,0]]):
        with pytest.raises(ValueError): R.post_gain(x, {'post_gain_f':points}, 0)


def test_shared_bed_crops_preserve_source_clock_and_reject_restarts(monkeypatch):
    monkeypatch.setattr(R,'SR',48);monkeypatch.setattr(R,'FR',2)
    def rng(seed): return np.random.default_rng(sum(seed.encode()))
    sound = SimpleNamespace(rng_for=rng, limit_crest=lambda y,c:y, db=lambda n:10**(n/20),
        match_gain=lambda *a,**k:0., proc=lambda y,**k:y,
        bed=lambda rc,dur,r:r.normal(size=(round(dur*48),2)).astype(np.float32))
    rc={'level':0,'send':0};space={}
    a=dict(f0=10,f1=20,source_span=[10,30],source_seed='same',fade_in_s=0,fade_out_s=0)
    b=dict(a,f0=15,f1=25)
    left=R.bed_clip(rc,a,'row_a',space,sound)
    right=R.bed_clip(rc,b,'row_b',space,sound)
    np.testing.assert_array_equal(left[12:18],right[2:8])
    wrong=R.bed_clip(rc,dict(b,source_seed='restart'),'row_b',space,sound)
    assert not np.array_equal(left[12:18],wrong[2:8])
    with pytest.raises(ValueError,match='containing source span'):
        R.bed_clip(rc,dict(b,source_span=[16,30]),'row_b',space,sound)
