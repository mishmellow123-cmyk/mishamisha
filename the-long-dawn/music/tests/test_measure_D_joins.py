"""Numerical join probes: explicit failures for jumps, drops and hidden stereo energy."""
import json

import numpy as np
import pytest
import soundfile as sf

import measure_D_joins as J


def stereo_tone(sr=8000, frequency=440., gain=.1):
    y = gain * np.sin(2 * np.pi * frequency * np.arange(sr) / sr)
    return np.column_stack((y, y))


def test_owner_window_layout_and_amplitude_steps():
    one = stereo_tone()
    whole = np.concatenate((one, one * 2))
    row = J.measure_window(whole, 8000)
    assert row["quarter_dbfs"][:4] == pytest.approx([-23.0102999566] * 4)
    assert row["quarter_dbfs"][4:] == pytest.approx([-16.9897000434] * 4)
    assert row["step250_db"] == pytest.approx(20 * np.log10(2))
    assert row["step1s_db"] == pytest.approx(20 * np.log10(2))
    assert row["chroma_cos"] == pytest.approx(1.)
    assert row["centroid_hz"] == pytest.approx([440., 440.], abs=.02)
    assert not row["step_guard"]["ok"]  # A doubled signal exceeds 6 dB.


@pytest.mark.parametrize("value,direction", [(6.001, "jump"), (-6.001, "drop")])
@pytest.mark.parametrize("window", ["step250_db", "step1s_db"])
def test_six_db_guard_negative_controls_both_signs_and_diagnostic_scope(value, direction, window):
    row = dict(step250_db=0., step1s_db=0.)
    assert J.step_guard(row)["ok"]
    row[window] = value
    result = J.step_guard(row)
    if window == "step250_db":
        assert not result["ok"]
        assert result["exceedances"] == [dict(window=window, db=value, direction=direction)]
    else:
        assert result["ok"] and result["exceedances"] == []
        assert result["diagnostic_1s"]["above_same_magnitude"]
        assert not result["diagnostic_1s"]["governs_owner_guard"]


def test_six_db_boundary_passes_and_invalid_values_fail():
    assert J.step_guard(dict(step250_db=6., step1s_db=-6.))["ok"]
    with pytest.raises(ValueError):
        J.step_guard(dict(step250_db=float("nan"), step1s_db=0.))
    with pytest.raises(ValueError):
        J.step_guard(dict(step250_db=0., step1s_db=0.), -1.)


def test_chroma_detects_changed_pitch_with_level_unchanged():
    row = J.measure_window(np.concatenate((stereo_tone(), stereo_tone(frequency=466.))), 8000)
    assert abs(row["step1s_db"]) < .01
    assert row["chroma_cos"] < .01
    assert row["step_guard"]["ok"]  # RMS gate does not claim harmonic continuity.


def test_antiphase_negative_control_reveals_mono_blind_spot():
    tone = stereo_tone()
    same = J.stereo_energy(tone)
    tone[:, 1] *= -1
    opposite = J.stereo_energy(tone)
    assert opposite["stereo_rms_dbfs"] == pytest.approx(same["stereo_rms_dbfs"])
    assert same["mono_rms_dbfs"] > -24.
    assert opposite["mono_rms_dbfs"] == -120.
    assert opposite["lr_cross_mean"] < 0
    assert opposite["mono_mean_square"] == pytest.approx(0.)


def test_silence_is_explicit_and_short_handles_are_rejected():
    x = np.concatenate((np.zeros((8000, 2)), stereo_tone()))
    row = J.measure_window(x, 8000)
    assert row["before_1s_dbfs"] == -120.
    assert row["chroma_silence_caveat"]
    assert row["low_level_1s_below_minus90_dbfs"] == [True, False]
    for invalid in (x[:-1], x[:, :1], x * float("nan")):
        with pytest.raises(ValueError):
            J.measure_window(invalid, 8000)


def test_streamed_file_selects_cut_window_and_pins_bytes(tmp_path):
    x = np.concatenate((stereo_tone(), stereo_tone(), stereo_tone() * 2))
    path = tmp_path / "tone.wav"
    sf.write(path, x, 8000, subtype="FLOAT")
    result = J.measure_file(path, [48])
    assert result["input"]["sha256"] == J.file_hash(path)
    assert result["input"]["sample_frames"] == 24000
    assert result["rows"][0]["centre_sample"] == 16000
    assert result["rows"][0]["step1s_db"] == pytest.approx(20 * np.log10(2))
    with pytest.raises(ValueError, match="handles"):
        J.measure_file(path, [12])
    with pytest.raises(ValueError, match="integer"):
        J.measure_file(path, [24.5])


def test_file_changed_during_measurement_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "tone.wav"
    sf.write(path, np.concatenate((stereo_tone(), stereo_tone())), 8000)
    hashes = iter(["a", "b"])
    monkeypatch.setattr(J, "file_hash", lambda path: next(hashes))
    with pytest.raises(ValueError, match="changed"):
        J.measure_file(path, [24])


def test_cli_uses_owner_cut_list_and_reports_independent_stems(tmp_path):
    path, joins, output = (tmp_path / name for name in ("tone.wav", "cuts.json", "result.json"))
    sf.write(path, np.concatenate((stereo_tone(), stereo_tone())), 8000)
    joins.write_text(json.dumps([dict(cut=24)]))
    assert J.main(["--wav", f"score={path}", "--joins", str(joins), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["stems"]["score"]["rows"][0]["cut"] == 24
    assert "no inferred mix fractions" in result["scope"]
    assert str(tmp_path) not in output.read_text()
    with pytest.raises(SystemExit):
        J.main(["--wav", f"score={path}", "--cuts", "24,24", "--output", str(output)])
