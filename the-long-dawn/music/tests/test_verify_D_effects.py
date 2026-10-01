"""Small synthetic controls; no production renders or shared cache writes."""
from pathlib import Path
from copy import deepcopy
import sys

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import verify_D_effects as V


def tone(seconds, amplitude=.1, frequency=997.):
    t = np.arange(round(seconds*V.SR))/V.SR
    mono = amplitude*np.sin(2*np.pi*frequency*t)
    return np.column_stack((mono, .73*mono))


@pytest.mark.parametrize("seconds", [.4, .451, 3.027, 8.451])
def test_streamed_lufs_matches_pyloudnorm(seconds):
    x = tone(seconds)
    # Quiet blocks and a level step exercise absolute and relative gates.
    x[:len(x)//5] *= 1e-5
    x[len(x)//2:] *= .1
    expected = pyln.Meter(V.SR).integrated_loudness(x)
    assert V.lufs(x) == pytest.approx(expected, abs=1e-10)
    assert V.lufs(2*x)-V.lufs(x) == pytest.approx(20*np.log10(2), abs=1e-10)


def test_lufs_filter_state_across_small_chunks(monkeypatch):
    x = np.random.default_rng(12).normal(0, .01, (4*V.SR+139, 2))
    monkeypatch.setattr(V, "CHUNK", 17777)
    assert V.lufs(x) == pytest.approx(pyln.Meter(V.SR).integrated_loudness(x), abs=1e-10)


def test_lufs_silence_short_and_nonfinite_controls():
    assert V.lufs(np.zeros((V.SR, 2))) == -np.inf
    with pytest.raises(ValueError, match="0.4 seconds"):
        V.lufs(np.zeros((100, 2)))
    x = tone(1)
    x[101, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        V.lufs(x)


def test_true_peak_matches_whole_polyphase_with_boundary_transient():
    x = tone(.2, frequency=17000)
    x[4095:4098] = [[.75, -.4], [-.8, .6], [.6, -.8]]
    y = signal.resample_poly(x, 4, 1, axis=0)
    peak = np.abs(y).max(axis=1).reshape(-1, 4).max(axis=1).astype(np.float32).max()
    expected = 20*np.log10(float(peak)+1e-12)
    assert V.true_peak_db(x, chunk=4096) == pytest.approx(expected, abs=1e-10)
    assert V.true_peak_db(x/4, chunk=4096) < -1
    assert V.true_peak_db(x*2, chunk=4096) > -1


def test_short_term_matches_engine_windows_and_silent_fallback():
    x = tone(4.7)
    expected = []
    t = 0.
    while t+3 <= len(x)/V.SR+1e-6:
        expected.append((t+1.5, pyln.Meter(V.SR).integrated_loudness(x[int(t*V.SR):int((t+3)*V.SR)])))
        t += .5
    np.testing.assert_allclose(V.st_loudness(x), expected, atol=1e-10, rtol=0)
    quiet = V.st_loudness(np.zeros_like(x))
    np.testing.assert_array_equal(quiet[:, 0], np.asarray(expected)[:, 0])
    assert np.all(quiet[:, 1] == -120)
    assert V.st_loudness(x[:V.SR]).shape == (0, 2)


def test_level_map_exact_masks_and_drift_failure():
    track = np.array([[1.5, -12], [2, -14], [2.5, -15], [3, -16], [3.5, -18]])
    sections = [{"id": "D1", "f0": 0, "f1": 72}, {"id": "D2", "f0": 72, "f1": 96}]
    rows = V.level_rows(track, sections, [("D1", -3, 0), ("D2", -5, -1)])
    assert all(row["ok"] for row in rows)
    assert rows[0]["windows"] == 2
    assert rows[0]["relative_median_lu"] == -1
    # The short D2 section invokes analyze_v3's +/-1.6s midpoint fallback.
    assert rows[1]["windows"] == 4
    assert rows[1]["relative_max_lu"] == -2
    wrong = V.level_rows(track, sections, [("D1", -3, -2), ("D2", 0, 1)])
    assert not any(row["ok"] for row in wrong)


@pytest.mark.parametrize("sections,bands,reason", [
    ([{"id": "D1", "f0": 0, "f1": 96}], [], "missing band"),
    ([], [("D1", -3, 0)], "missing section"),
    ([{"id": "D1", "f0": 0, "f1": 96}], [("D1", -3, 0), ("D1", -3, 0)], "duplicate band"),
    ([{"id": "D1", "f0": 0, "f1": 96}]*2, [("D1", -3, 0)], "duplicate section"),
    ([{"id": "D1", "f0": 0, "f1": 96}], [("D1", 1, -1)], "invalid band"),
    ([], [], "empty level map"),
])
def test_missing_or_invalid_level_contract_fails(sections, bands, reason):
    rows = V.level_rows(np.array([[1.5, -16], [2, -16]]), sections, bands)
    assert any(not row["ok"] and row.get("reason") == reason for row in rows)


def test_d34_containment_is_explicit_retains_legacy_and_never_mutes_mix():
    track = np.array([[1.5, -16], [381, -30], [381.5, -120], [382, -120]])
    sections = [{"id": "D34", "f0": 9120, "f1": 9200}]
    bands = [("D34", -105, -103)]
    standard = V.level_rows(track, sections, bands)[0]
    assert not standard["ok"]
    contained = V.level_rows(track, sections, bands, hard_silences=((9120, 9200),))[0]
    assert contained["ok"]
    assert contained["windows"] == 1
    assert not contained["legacy_engine_window"]["ok"]
    track[2, 1] = -30
    assert not V.level_rows(track, sections, bands, hard_silences=((9120, 9200),))[0]["ok"]


def test_authored_mix_d34_band_accepts_hearth_and_rejects_excess():
    # This is an authored MIX contract, distinct from SCORE's (-120,-60).
    sections = [{"id": "D34", "f0": 9120, "f1": 9200}]
    track = np.array([[1.5, -16], [381, -52], [381.5, -54], [382, -120]])
    row = V.level_rows(track, sections, [("D34", -120, -35)])[0]
    assert row["ok"]
    assert row["window_policy"] == "engine_1s_interior"
    assert not V.level_rows(track, sections, [("D34", -120, -60)])[0]["ok"]
    track[1, 1] = -50
    assert not V.level_rows(track, sections, [("D34", -120, -35)])[0]["ok"]


def test_silence_leak_boundary_missing_and_nonfinite_controls():
    x = np.zeros((40*V.FRAME_N, 2), np.float32)
    assert V.silence_row(x, 3400, 3440, first_frame=3400)["ok"]
    x[-1, 1] = 2**-23
    assert not V.silence_row(x, 3400, 3440, first_frame=3400)["ok"]
    x[-1, 1] = np.nan
    assert not V.silence_row(x, 3400, 3440, first_frame=3400)["ok"]
    assert not V.silence_row(x[:-1], 3400, 3440, first_frame=3400)["ok"]


def test_voice_ceiling_covers_entire_window_and_clipping():
    x = np.zeros((240*V.FRAME_N, 2), np.float32)
    def row(name):
        return next(r for r in V.constraints(x, "sfx", first_frame=8640) if r["check"] == name)
    assert row("voice effects ceiling")["ok"]
    x[-1, 0] = .011
    assert not row("voice effects ceiling")["ok"]
    x[-1, 0] = 1.
    assert not row("clipping")["ok"]
    x[-1, 0] = np.inf
    assert not row("finite")["ok"]
    assert not row("voice effects ceiling")["ok"]


def test_packed_prefix_detects_single_pcm_bit_but_ignores_later_audio(tmp_path):
    n = 2*V.FRAME_N
    a = tmp_path/"a.wav"
    b = tmp_path/"b.wav"
    x = np.zeros((n, 2))
    sf.write(a, x, V.SR, subtype="PCM_24")
    x[-1, 0] = .5
    sf.write(b, x, V.SR, subtype="PCM_24")
    assert V.pcm_prefix_sha256(a, 1) == V.pcm_prefix_sha256(b, 1)
    x[0, 0] = 2**-23
    sf.write(b, x, V.SR, subtype="PCM_24")
    assert V.pcm_prefix_sha256(a, 1) != V.pcm_prefix_sha256(b, 1)
    with pytest.raises(ValueError, match="shorter"):
        V.pcm_prefix_sha256(a, 3)
    sf.write(b, x, V.SR, subtype="PCM_16")
    with pytest.raises(ValueError, match="24-bit"):
        V.pcm_prefix_sha256(b, 1)


@pytest.fixture
def mini_delivery(tmp_path, monkeypatch):
    """Reduce only the timeline contract; keep real DSP and real PCM I/O."""
    monkeypatch.setattr(V, "SAMPLE_FRAMES", 4*V.SR)
    monkeypatch.setattr(V, "PREFIX_FRAMES", 1)
    monkeypatch.setattr(V, "ZERO_WINDOWS", ((24, 25, "vacuum"), (48, 49, "voice")))
    monkeypatch.setattr(V, "VOICE_WINDOW", (48, 50))
    monkeypatch.setattr(V, "SCORE_TAIL", (88, 96, "score tail"))
    mix = tone(4)
    for a, b in ((24, 25), (48, 49)):
        mix[a*V.FRAME_N:b*V.FRAME_N] = 0
    mix *= 10**((-16-V.lufs(mix))/20)
    score = mix.copy()
    score[88*V.FRAME_N:] = 0
    arrays = {"mix": mix, "score": score, "sfx": np.zeros_like(mix)}
    refs = tmp_path/"ap2"
    refs.mkdir()
    paths = {}
    for kind, x in arrays.items():
        paths[kind] = tmp_path/(kind+".wav")
        suffix = "" if kind == "mix" else "_"+kind
        sf.write(paths[kind], x, V.SR, subtype="PCM_24")
        sf.write(refs/("sound_AP2"+suffix+".wav"), x, V.SR, subtype="PCM_24")
    kwargs = dict(bands=[("D1", -120, 1)], sections=[{"id": "D1", "f0": 0, "f1": 96}], ap2_root=refs)
    def verify():
        return V.verify_paths(paths["sfx"], paths["mix"], paths["score"], **kwargs)
    return paths, arrays, verify, kwargs


def test_verify_paths_real_pcm_success_and_cleanup(mini_delivery, tmp_path, monkeypatch):
    paths, _, verify, _ = mini_delivery
    monkeypatch.setattr(V.tempfile, "tempdir", str(tmp_path))
    report = verify()
    assert report["ok"], [r for r in report["checks"] if not r["ok"]]
    assert set(report["paths"]) == {"sfx", "mix", "score"}
    assert not list(tmp_path.glob("verify-d-effects-*"))
    assert all(p.exists() for p in paths.values())


@pytest.mark.parametrize("mutation,failed_check", [
    ("drift", "integrated loudness"), ("leak", "silence"),
    ("prefix", "packed AP2 prefix"), ("peak", "true peak"),
    ("format", "WAV format"),
])
def test_verify_paths_rejects_deliberate_defects(mini_delivery, mutation, failed_check):
    paths, arrays, verify, _ = mini_delivery
    x = arrays["mix"].copy()
    subtype = "PCM_24"
    if mutation == "drift":
        x *= 10**(.5/20)
    elif mutation == "leak":
        x[24*V.FRAME_N, 0] = 2**-23
    elif mutation == "prefix":
        x[0, 0] += 2**-23
    elif mutation == "peak":
        x[V.SR:V.SR+4] = .99
    else:
        subtype = "PCM_16"
    sf.write(paths["mix"], x, V.SR, subtype=subtype)
    report = verify()
    assert not report["ok"]
    assert any(r["check"] == failed_check and not r["ok"] for r in report["checks"])


def test_verify_paths_json_contains_no_nonfinite_numbers(mini_delivery, tmp_path):
    import json
    paths, _, _, kwargs = mini_delivery
    output = tmp_path/"report.json"
    report = V.verify_paths(paths["sfx"], paths["mix"], paths["score"], out_json=output, **kwargs)
    assert json.loads(output.read_text()) == report
    sfx_level = next(r for r in report["checks"] if r["check"] == "integrated loudness" and r["kind"] == "sfx")
    assert sfx_level["lufs"] is None


def test_verify_paths_keeps_separate_score_bands(mini_delivery):
    paths, _, _, kwargs = mini_delivery
    source_score_bands = [("D1", -120, -35)]
    report = V.verify_paths(paths["sfx"], paths["mix"], paths["score"],
                            score_bands=source_score_bands, **kwargs)
    assert report["ok"]
    assert report["score_level_rows"][0]["band_lu"] == [-120., -35.]
    assert not report["score_level_rows"][0]["ok"]
    mix_row = next(r for r in report["checks"] if r["check"] == "level band")
    assert mix_row["band_lu"] == [-120., 1.]
    assert mix_row["ok"]


@pytest.fixture
def nominal_binding():
    reused = dict(id="donor.event", hit_f=2000, source="derived", source_sync="inherited fixture")
    events = [dict(id="gap", request_id="D.new.gap.hammer", hit_f=3440, source="est.", hook="gap")]
    for side, pan in (("left", -1), ("right", 1)):
        events.append(dict(id=side, request_id="D.new.crowns."+side, hit_f=4320, source="est.",
                           pan=pan, hook="crowns", source_choice_provenance={"anchor": ["fixture", pan]}))
    bed = dict(id="bed", request_id="new.bed", f0=4000, f1=4200, source="est.", hook="bed")
    binding = dict(reuse_plan={"EXTRA_EVENTS": [reused], "BED_CROPS": []}, new_events=events, new_beds=[bed])
    rows = [dict(e, sample_hit=e["hit_f"]*V.FRAME_N) for e in [reused]+events]
    for row, pan in ((rows[2], -1), (rows[3], 1)):
        row["isolated_channel_peaks"] = [.5, 0.] if pan == -1 else [0., .4]
        row["source_recipe"] = {"layers": [{"src": ["fixture", pan]}]}
    receipt = dict(events=rows, beds=[deepcopy(bed)], crown_pair=dict(
        left_hit_sample=4320*V.FRAME_N, right_hit_sample=4320*V.FRAME_N,
        max_channel_difference=.2, correlation=.1, compared_samples=100))
    return receipt, binding


def test_nominal_binding_audit_is_explicit_about_measurement_scope(nominal_binding):
    receipt, binding = nominal_binding
    report = V.verify_event_bindings(receipt, binding)
    assert report["ok"]
    assert "no audio-onset or picture measurement" in report["measurement_scope"]
    assert all(not r["measured_audio_onset"] for r in report["checks"] if "measured_audio_onset" in r)
    receipt["events"][0]["sample_hit"] += V.FRAME_N
    assert V.verify_event_bindings(receipt, binding)["ok"]
    receipt["events"][0]["sample_hit"] += 1
    assert not V.verify_event_bindings(receipt, binding)["ok"]


@pytest.mark.parametrize('key', ['picture_frame','measured_ref','measurement_scope','measurement'])
def test_native_provenance_cannot_be_lost_or_promoted_in_delivery(nominal_binding, key):
    receipt,binding=nominal_binding
    provenance=dict(source='measured',picture_frame=3440,measured_ref='frame_receipt.json',
                    measurement_scope='native_delivered_plate',
                    measurement=dict(claim='contact',onset_measured=False,boundary_censored=True))
    binding['new_events'][0].update(deepcopy(provenance))
    receipt['events'][1].update(deepcopy(provenance))
    assert V.verify_event_bindings(receipt,binding)['ok']
    receipt['events'][1].pop(key)
    assert not V.verify_event_bindings(receipt,binding)['ok']
    receipt['events'][1][key] = 'finished_film' if key!='measurement' else dict(claim='onset')
    assert not V.verify_event_bindings(receipt,binding)['ok']


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "missing_id", "source",
                                      "missing_source", "same_crown", "crown_leak", "anchor",
                                      "gap_double", "bed_missing", "bed_source"])
def test_nominal_binding_rejects_missing_changed_or_duplicated_evidence(nominal_binding, mutation):
    receipt, binding = nominal_binding
    rows = receipt["events"]
    if mutation == "missing":
        rows.pop(0)
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "extra":
        rows.append(dict(rows[0], id="unplanned"))
    elif mutation == "missing_id":
        rows[0].pop("id")
    elif mutation == "source":
        rows[0]["source"] = "measured"
    elif mutation == "missing_source":
        rows[0].pop("source")
    elif mutation == "same_crown":
        receipt["crown_pair"]["max_channel_difference"] = 0
    elif mutation == "crown_leak":
        rows[2]["isolated_channel_peaks"][1] = 2**-23
    elif mutation == "anchor":
        rows[3]["source_recipe"] = deepcopy(rows[2]["source_recipe"])
    elif mutation == "gap_double":
        rows.append(dict(rows[1], id="second_gap"))
    elif mutation == "bed_missing":
        receipt["beds"].clear()
    else:
        receipt["beds"][0]["source"] = "measured"
    assert not V.verify_event_bindings(receipt, binding)["ok"]


def gap_fixture(delay_frames=0):
    # Loud earlier vision is outside D's declared vacuum and cannot contaminate it.
    x = np.zeros((72*V.FRAME_N, 2), np.float32)
    x[:20*V.FRAME_N] = .9
    onset = int((60+delay_frames)*V.FRAME_N)
    x[onset:onset+4*V.FRAME_N] = .5
    return x


def test_gap_arrival_uses_only_declared_vacuum_and_measures_arrival():
    row = V.gap_arrival(gap_fixture(), 3380)
    assert row["ok"] and row["status"] == "ON"
    assert abs(row["delta_frames"]) < .1
    assert row["pre_db"] < row["peak_db"]-30
    assert not row["measured_picture"]


def test_gap_arrival_tracks_a_remeasured_contact_without_moving_edit_silence():
    x=np.pad(gap_fixture(2),((0,2*V.FRAME_N),(0,0)))
    assert V.gap_arrival(x,3380)['status']=='OFF'
    row=V.gap_arrival(x,3380,expected_frame=3442)
    assert row['ok'] and abs(row['measured_audio_frame']-3442)<.1
    assert row['isolation_start_frame']==3400
    assert row['coverage_frames']==[3400,3454]
    assert V.gap_arrival(x[:-1],3380,expected_frame=3442)['status']=='MISSING COVERAGE'


@pytest.mark.parametrize('frame',[3439,3520,3440.5,True,None])
def test_gap_arrival_rejects_invalid_binding_target(frame):
    with pytest.raises(ValueError): V.gap_arrival(gap_fixture(),3380,expected_frame=frame)


@pytest.mark.parametrize("mutation,status", [("missing", "MISSING COVERAGE"), ("silence", "SILENT"),
                                            ("preleak", "NOT ISOLATED"), ("delay", "OFF"),
                                            ("nonfinite", "NONFINITE")])
def test_gap_arrival_negative_controls(mutation, status):
    x = gap_fixture(2 if mutation == "delay" else 0)
    if mutation == "missing":
        x = x[:-1]
    elif mutation == "silence":
        x[20*V.FRAME_N:] = 0
    elif mutation == "preleak":
        x[30*V.FRAME_N:31*V.FRAME_N] = .1
    elif mutation == "nonfinite":
        x[30*V.FRAME_N, 0] = np.nan
    row = V.gap_arrival(x, 3380)
    assert not row["ok"] and row["status"] == status


@pytest.mark.parametrize("key", ["race_work_clock", "race_role", "design_revision", "texture_provenance", "stop_f"])
def test_race_design_delivery_provenance_is_checked(nominal_binding, key):
    receipt, binding = nominal_binding
    design = dict(race_work_clock={"parameter_frames": [2400, 2420]}, race_role="synthetic field fixture",
                  design_revision="round5", texture_provenance={"layer_count": 4}, stop_f=2720)
    binding["new_events"][0].update(deepcopy(design))
    receipt["events"][1].update(deepcopy(design))
    assert V.verify_event_bindings(receipt, binding)["ok"]
    receipt["events"][1].pop(key)
    assert not V.verify_event_bindings(receipt, binding)["ok"]


def test_resident_verification_matches_disk_and_writes_no_float_scratch(mini_delivery, monkeypatch):
    paths, _, verify, kwargs = mini_delivery
    disk = verify()
    def forbidden(*args, **kw):
        raise AssertionError('resident verifier must not allocate a disk memmap')
    monkeypatch.setattr(V.np, 'memmap', forbidden)
    resident = V.verify_paths(paths['sfx'], paths['mix'], paths['score'], **kwargs, resident=True)
    assert resident['ok'] and resident['storage'] == 'resident float32'
    assert resident['checks'] == disk['checks']
    assert resident['short_term_loudness'] == disk['short_term_loudness']
    with pytest.raises(ValueError, match='explicitly boolean'):
        V.verify_paths(paths['sfx'], paths['mix'], paths['score'], **kwargs, resident='false')


@pytest.mark.parametrize("key", ["end_provenance", "continuity_provenance", "picture_revision",
                                 "bridge_provenance", "post_gain_f", "source_span", "source_seed"])
def test_locked_delivery_cannot_drop_endpoint_or_continuity_evidence(nominal_binding, key):
    receipt, binding = nominal_binding
    fields = dict(end_provenance={"frame": 2742, "scope": "synthetic"},
                  continuity_provenance={"restart_at_2400": False}, picture_revision="locked",
                  bridge_provenance=[{"authored":True}], post_gain_f=[[0,-6],[1,0]],
                  source_span=[0,4], source_seed='synthetic')
    binding["new_beds"][0].update(deepcopy(fields))
    receipt["beds"][0].update(deepcopy(fields))
    assert V.verify_event_bindings(receipt, binding)["ok"]
    receipt["beds"][0].pop(key)
    assert not V.verify_event_bindings(receipt, binding)["ok"]
