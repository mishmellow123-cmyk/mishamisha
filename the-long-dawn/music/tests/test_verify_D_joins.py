"""Population, signed-step, waveform and silence controls for the D join audit."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import verify_D_joins as V


def edit():
    bounds = (0,) + V.CUTS + (V.FRAMES,)
    return dict(cut="D", fps=24, frames=V.FRAMES, transitions=[],
                shots=[dict(code=f"shot{i}", f0=a, f1=b)
                       for i, (a, b) in enumerate(zip(bounds, bounds[1:]))])


def owners():
    return [dict(cut=c) for c in V.CUTS]


def test_population_exact_and_internal_shot_joins_present():
    rows = V.join_rows(edit(), owners())
    assert len(rows) == 40
    assert [r["cut"] for r in rows] == list(V.CUTS)
    assert {2728, 2758, 2945, 5554, 5630}.issubset(r["cut"] for r in rows)


@pytest.mark.parametrize("mutation", ["missing_owner", "duplicate_owner", "missing_shot",
                                     "shifted_join", "gapped_shot", "wrong_fps", "wrong_length"])
def test_population_cannot_silently_shrink_or_move(mutation):
    edl, owner = edit(), owners()
    if mutation == "missing_owner":
        del owner[0]
    elif mutation == "duplicate_owner":
        owner.append(owner[-1])
    elif mutation == "missing_shot":
        edl["shots"][0]["f1"] = edl["shots"][1]["f1"]
        del edl["shots"][1]
    elif mutation == "shifted_join":
        edl["shots"][0]["f1"] += 1
        edl["shots"][1]["f0"] += 1
    elif mutation == "gapped_shot":
        edl["shots"][1]["f0"] += 1
    elif mutation == "wrong_fps":
        edl["fps"] = 25
    else:
        edl["frames"] -= 1
    with pytest.raises(ValueError):
        V.join_rows(edl, owner)


@pytest.mark.parametrize("gain_db", [6.01, -6.01, 12, -12])
def test_signed_excess_step_is_detected(gain_db):
    a = np.ones(100) * .03
    result = V.step(a, a * 10 ** (gain_db/20))
    assert result["above_6_db"]
    assert result["owner_step_db"] == pytest.approx(gain_db, abs=1e-12)
    assert not result["floor_qualified"]


def test_floor_never_automatically_passes_a_large_entrance():
    result = V.step(np.zeros(100), np.ones(100) * .01)
    assert result["step_db"] is None
    assert result["owner_step_db"] == 80
    assert result["above_6_db"] and result["floor_qualified"]
    silent = V.step(np.zeros(100), np.zeros(100))
    assert silent["owner_step_db"] == 0 and not silent["above_6_db"]


def test_mono_cancellation_does_not_hide_stereo_energy_step():
    a = np.ones(100) * .01
    whole = np.r_[a, a * 4]
    x = np.column_stack((whole, -whole))
    result = V.modes(x, len(a))
    assert not result["mono"]["above_6_db"]
    assert result["stereo"]["above_6_db"]
    assert result["stereo"]["owner_step_db"] == pytest.approx(20*np.log10(4))


@pytest.mark.parametrize("x", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_invalid_windows_fail(x):
    with pytest.raises(ValueError, match="empty or nonfinite"):
        V.level(x)


def test_wrong_window_geometry_fails():
    with pytest.raises(ValueError, match="window length"):
        V.modes(np.zeros((30, 2)), 20)


@pytest.fixture
def mini_delivery(tmp_path, monkeypatch):
    # Same frame/sample mapping and programme length, at 1/200 sample density.
    monkeypatch.setattr(V, "SR", 240)
    monkeypatch.setattr(V, "SAMPLES_PER_FRAME", 10)
    wav, edl, owner = (tmp_path / name for name in ("mix.wav", "edl.json", "owner.json"))
    x = np.full((V.FRAMES * 10, 2), .01)
    for a, b in V.SILENCES:
        x[a*10:b*10] = 0
    sf.write(wav, x, V.SR, subtype="PCM_24")
    edl.write_text(json.dumps(edit()))
    owner.write_text(json.dumps(owners()))
    return wav, edl, owner, x


def test_full_audit_reads_every_join_and_every_sliding_pivot(mini_delivery):
    wav, edl, owner, _ = mini_delivery
    doc = V.audit(wav, edl, owner, V.sha256(wav))
    assert doc["summary"]["ok"]
    assert doc["summary"]["joins"] == 40
    assert doc["summary"]["nonexempt_joins"] == 37
    assert len(doc["sliding"]) == 10
    assert all(len(r["frame_pivots"]) == 145 for r in doc["sliding"])
    assert all(s["ok"] for s in doc["required_silences"])
    assert all(r["rejected"] for r in doc["signed_negative_controls"])
    assert doc["audio_files_created"] == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_actual_audio_step_corruption_trips_final_gate(mini_delivery, sign):
    wav, edl, owner, x = mini_delivery
    x[2080*10:] *= 10 ** (sign*6.1/20)
    sf.write(wav, x, V.SR, subtype="PCM_24")
    doc = V.audit(wav, edl, owner)
    assert not doc["summary"]["ok"]
    assert 2080 in doc["summary"]["unexpected_cuts_over_6_db"]


def test_known_exceptions_do_not_hide_an_extra_unmotivated_step(mini_delivery):
    wav, edl, owner, x = mini_delivery
    x[3520*10:] *= 4
    x[4240*10:] /= 4
    sf.write(wav, x, V.SR, subtype="PCM_24")
    doc = V.audit(wav, edl, owner)
    by_cut = {r["cut"]: r for r in doc["rows"]}
    assert by_cut[3520]["verdict"] == "owner_exception"
    assert by_cut[4240]["verdict"] == "unexpected_step"
    assert not doc["summary"]["ok"]


def test_displaced_step_is_visible_in_diagnostic(mini_delivery):
    wav, edl, owner, x = mini_delivery
    x[2100*10:] *= 4
    sf.write(wav, x, V.SR, subtype="PCM_24")
    doc = V.audit(wav, edl, owner)
    assert doc["summary"]["ok"]
    slide = next(r for r in doc["sliding"] if r["cut"] == 2080)
    assert 2100 in slide["over_6_db"]["mono"]


def test_single_lsb_breaks_required_silence(mini_delivery):
    wav, edl, owner, x = mini_delivery
    x[8665*10, 1] = V.LSB
    sf.write(wav, x, V.SR, subtype="PCM_24")
    doc = V.audit(wav, edl, owner)
    assert not doc["summary"]["ok"]
    assert doc["required_silences"][1]["nonzero_samples"] == 1


def test_wrong_identity_fails_before_audit(mini_delivery):
    with pytest.raises(ValueError, match="WAV hash"):
        V.audit(*mini_delivery[:3], expected_sha256="0"*64)


def test_input_changed_mid_audit_is_rejected(mini_delivery, monkeypatch):
    wav, edl, owner, _ = mini_delivery
    original_read = V._read
    modified = False

    def read_and_change_reference(*args):
        nonlocal modified
        result = original_read(*args)
        if not modified:
            owner.write_text(owner.read_text() + "\n")
            modified = True
        return result

    monkeypatch.setattr(V, "_read", read_and_change_reference)
    with pytest.raises(ValueError, match="input changed"):
        V.audit(wav, edl, owner)


@pytest.mark.parametrize("kind", ["short", "mono", "rate", "float"])
def test_wrong_wave_format_or_length_fails(mini_delivery, kind):
    wav, edl, owner, x = mini_delivery
    if kind == "short":
        x = x[:-1]
    if kind == "mono":
        x = x[:, 0]
    sf.write(wav, x, V.SR + (1 if kind == "rate" else 0),
             subtype="FLOAT" if kind == "float" else "PCM_24")
    with pytest.raises(ValueError, match="D mix must"):
        V.audit(wav, edl, owner)
