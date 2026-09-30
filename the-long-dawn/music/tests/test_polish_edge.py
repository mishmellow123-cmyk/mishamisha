"""The scoped AP2 breath/edge edit, including controls that break each contract."""
from types import SimpleNamespace

import numpy as np
import pytest

import sound_polish_edge as P


def bm(cut="AP2", edge=1840):
    return SimpleNamespace(cut=cut, event=lambda k: {"frame": edge if k == "edge" else 2400})


def apply_fixture(cut="AP2", edge=1840):
    # Low sample rate surrogate keeps these structural tests small. The actual
    # source/render probe runs the same function at 48 kHz.
    n = 2500
    score = np.ones((n, 2), np.float32)
    fx = np.ones_like(score)
    replacement = np.full((32, 2), 0.8, np.float32)
    return P.apply(score, fx, bm(cut, edge), replacement)


def test_changed_interval_and_boundary_controls(monkeypatch):
    monkeypatch.setattr(P, "FR", 1)
    score, fx = apply_fixture()
    for y in (score, fx):
        assert np.array_equal(y[:1441], np.ones_like(y[:1441]))
        assert np.array_equal(y[2399:], np.ones_like(y[2399:]))
    assert score[1836, 0] < 0.5 and fx[1873, 0] < 0.85
    # Negative controls: the original all-unity score and bed cannot satisfy the
    # implemented breath/beat assertions; either hook silently omitted fails.
    assert not (np.ones_like(score)[1836, 0] < 0.5)
    assert not (np.ones_like(fx)[1873, 0] < 0.85)


def test_other_cut_is_exact_and_anchor_drift_is_refused(monkeypatch):
    monkeypatch.setattr(P, "FR", 1)
    score, fx = apply_fixture("A")
    assert np.array_equal(score, np.ones_like(score))
    assert np.array_equal(fx, np.ones_like(fx))
    with pytest.raises(ValueError, match="anchors"):
        apply_fixture(edge=1841)


def test_breath_is_nonzero_and_reaches_downbeat():
    f = np.arange(1822, 1841, 0.12)  # 5 ms bins on the 24 fps grid
    gain = 10 ** (P.smooth_curve(f, P.BREATH_DB) / 20)
    assert gain.min() > 0.4
    assert np.max(np.abs(np.diff(20 * np.log10(gain)))) < 0.5
    assert P.smooth_curve([1837, 1840], P.BREATH_DB).tolist() == [-7.0, 0.0]
    # Original breath's -45 dB floor violates the floor requirement.
    assert not (10 ** (-45 / 20) > 0.4)


def test_pulse_attack_release_and_return_to_unity():
    f = np.arange(1917.0, 1933.0, 0.12)
    g = P.pulse_gain(f)
    peak = int(np.argmax(g))
    assert abs(f[peak] - 1920) <= 0.12
    assert np.all(np.diff(g[:peak + 1]) >= -1e-6)
    assert np.all(np.diff(g[peak:][f[peak:] < 1931.8]) <= 1e-6)
    assert np.max(np.abs(np.diff(20 * np.log10(g)))) < 0.15
    assert np.array_equal(P.pulse_gain([1440, 1839, 2390, 2399, 2400]), np.ones(5, np.float32))
    # A one-frame step, the rejected contour, cannot meet the bounded slope.
    pop = np.where(f < 1920, 10 ** (-2 / 20), 1.0)
    assert not (np.max(np.abs(np.diff(20 * np.log10(pop)))) < 0.15)


def test_ride_escalates_and_finishes_before_boundary():
    f = np.array([1854, 2000, 2160, 2320, 2360, 2400])
    g = P.smooth_curve(f, P.RIDE_DB)
    assert g[0] == -2 and g[-2] == g[-1] == 0
    assert np.all(np.diff(g) >= 0)
    assert not np.all(np.diff(g[::-1]) >= 0)  # a receding ride fails


def test_hook_is_adopted_by_ap2_only():
    import sound_recipes_AP2 as ap2
    import sound_recipes_A as a
    assert ap2.PREMASTER_POLISH is P.apply
    assert not hasattr(a, "PREMASTER_POLISH")


def test_stale_performance_or_ir_cannot_pass_reconstruction_guard():
    saved = np.full((100, 2), 0.1, np.float32)
    assert P.check_reconstruction(saved + 1e-7, saved) < P.RECONSTRUCTION_EPS
    wrong = saved.copy()
    wrong[50, 1] += 0.001
    with pytest.raises(ValueError, match="differs"):
        P.check_reconstruction(wrong, saved)
    with pytest.raises(ValueError, match="shape"):
        P.check_reconstruction(saved[:-1], saved)
    wrong[50, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        P.check_reconstruction(wrong, saved)


def test_listening_export_blend_cannot_reach_boundary_or_adjacent_lane():
    import verify_polish_edge as V
    f = np.arange(6480, dtype=float)
    w = P.smooth_curve(f, V.SPLICE)
    assert w[1822] == 1 and w[2389] == 1
    assert not w[:1818].any() and not w[2399:].any()
    # A finish one frame into the adjacent lane would fail the scope assertion.
    drift = P.smooth_curve(f, ((1818, 0), (1822, 1), (2389, 1), (2401, 0)))
    assert drift[2400] > 0


def test_outside_report_distinguishes_global_gain_from_compressor_change(tmp_path, monkeypatch):
    import soundfile as sf
    import verify_polish_edge as V
    monkeypatch.setattr(V, "FR", 1)
    rng = np.random.default_rng(34)
    a = rng.normal(0, 0.05, (6480, 2)).astype(np.float32)
    before, after = str(tmp_path / "before.wav"), str(tmp_path / "after.wav")
    sf.write(before, a, V.SR, subtype="FLOAT")
    sf.write(after, a * 1.2, V.SR, subtype="FLOAT")
    assert V.outside_difference(before, after)["max_abs_residual"] < 3e-8
    different = a * 1.2
    different[:500] *= 0.75
    sf.write(after, different, V.SR, subtype="FLOAT")
    assert V.outside_difference(before, after)["max_abs_residual"] > 0.01
