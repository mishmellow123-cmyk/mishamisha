"""A 2400-3360 sound contracts (polishdoom): the updraft rises into the white and survives the old gate, the ash wind
is up on the white and follows the valley's closing, the black keeps a floor, the ember arrives on its first light,
and the silence piano's first note enters softened without any later swell on its ringing tail.

Recipe/page tests need no recordings or renders; every acceptance check is also run against a rejected configuration
(the delivered recipes or the first pass's) and must fail there. The render audit measures the mix itself.
"""
import copy

import numpy as np
import pytest

import sound_polishdoom as P
import sound_recipes_A as A
import sound_recipes_AP2 as AP2


def interp_db(recipe, t0, frames):
    points = np.array(recipe["env"])
    return np.interp((np.asarray(frames) / 24.0) - t0, points[:, 0], points[:, 1])


def assert_fall_continues(recipes, timings):
    storm = recipes["A.storm.brink"]
    assert storm.get("no_breath"), "updraft must survive the 2637-2640 gate"
    assert timings["A.storm.brink"]["t1"] * 24 > P.WHITE
    rise = interp_db(storm, 100.0, [2520, 2600, 2620, 2636, 2637])
    assert np.all(np.diff(rise) >= 0) and rise[-1] >= rise[0] + 4, "the updraft must rise with the light"
    inhale = interp_db(storm, 100.0, np.linspace(2637, 2640, 13))
    assert np.all(np.diff(inhale) <= 0), "the suck before the impact is one continuous inhale"
    assert rise[-1] - inhale[-1] >= 6, "the impact must land on an inhale, not be buried under the updraft"
    assert inhale[-1] > -20, "the inhale keeps a floor"


def test_updraft_rises_inhales_and_lets_the_impact_land():
    assert_fall_continues(A.RECIPES, A.BED_TIMING)
    bad = copy.deepcopy(A.RECIPES)
    bad["A.storm.brink"]["no_breath"] = False
    with pytest.raises(AssertionError, match="survive"):
        assert_fall_continues(bad, A.BED_TIMING)
    flat = copy.deepcopy(A.RECIPES)
    flat["A.storm.brink"]["env"] = [(0, 0), (10, 0)]
    with pytest.raises(AssertionError, match="rise with the light"):
        assert_fall_continues(flat, A.BED_TIMING)
    # the first pass: +12 dB held through the impact (render 1, audited: +1.6 dB in 50 ms at 2640)
    buried = copy.deepcopy(A.RECIPES)
    buried["A.storm.brink"]["env"] = [(0, 0), (120 / 24, 0), (200 / 24, 5), (220 / 24, 9), (237 / 24, 12),
                                      (240 / 24, 12), (258 / 24, 0)]
    with pytest.raises(AssertionError, match="buried"):
        assert_fall_continues(buried, A.BED_TIMING)


def assert_ash_follows_valley(recipe, timing):
    t0 = timing["t0"]
    assert (P.WHITE / 24 - t0) >= timing["fade_in"], "ash is still fading in on the white"
    assert recipe.get("no_breath"), "ash was cut by the shared gate"
    white, held = interp_db(recipe, t0, [2640]), interp_db(recipe, t0, [2688, 2700, 2762])
    closing, black = interp_db(recipe, t0, [2779]), interp_db(recipe, t0, [2796])
    assert white[0] >= -4.6, "ash must be present on the white (at most 4.5 dB under its bed level)"
    assert np.all(held >= -0.1), "ash must hold its bed level while the valley is open (2688-2762)"
    assert -24.1 < closing[0] < -0.1, "ash must fall with the window closing (2762-2796)"
    assert black[0] <= -24, "ash must be down to the black's floor by the end of the close"


def test_ash_is_present_on_the_white_and_follows_the_valley_closing():
    assert_ash_follows_valley(A.RECIPES["A.wind.ash"], A.BED_TIMING["A.wind.ash"])
    # the delivered bed began at 2640 at -20 dB (reaching its level ~1.2 s later); the first pass pulled it down
    # from 2736, 26 frames before the window starts closing
    late = dict(A.RECIPES["A.wind.ash"], env=[(0.0, -20.0), (0.667, -20.0), (2.0, 0.0), (5.083, 0.0),
                                              (6.5, -24.0), (6.667, -40.0)])
    with pytest.raises(AssertionError):
        assert_ash_follows_valley(late, dict(t0=2640 / 24, fade_in=1.0))
    early = dict(A.RECIPES["A.wind.ash"], env=[(0, -6), (12 / 24, 0), (108 / 24, 0), (144 / 24, -24),
                                               (172 / 24, -40)])
    with pytest.raises(AssertionError, match="hold its bed level"):
        assert_ash_follows_valley(early, A.BED_TIMING["A.wind.ash"])


def assert_floor(recipes, beds):
    b = next((b for b in beds if b["id"] == "A.doom.black_air"), None)
    assert b is not None, "missing floor"
    r = recipes[b["id"]]
    assert b["t0"] * 24 + b["fade_in"] * 24 <= 2797, "the floor must be up before the black"
    assert b["t1"] * 24 - b["fade_out"] * 24 > 2848, "the floor must hold until the night bed starts"
    assert r["no_breath"] and r["send"] == 0


def test_recorded_floor_spans_the_black():
    assert_floor(A.RECIPES, A.EXTRA_BEDS)
    with pytest.raises(AssertionError, match="missing floor"):
        assert_floor(A.RECIPES, [])
    short = [dict(b, t1=2840 / 24) if b["id"] == "A.doom.black_air" else b for b in A.EXTRA_BEDS]
    with pytest.raises(AssertionError, match="night bed"):
        assert_floor(A.RECIPES, short)


def assert_ember_arrival(recipes, events):
    e = next((e for e in events if e["id"] == "A.doom.ember"), None)
    assert e is not None, "the ember has no sound"
    assert round(e["t"] * 24, 6) == P.EMBER == 2832, "the ember's sound must land on its first light (2832)"
    r = recipes["A.doom.ember"]
    pop, breath = r["layers"]
    assert pop["dt"] == 0 and pop["pre"] <= 0.005 and pop["fi"] <= pop["pre"], "the hit is the pop's own onset"
    assert pop["post"] <= 0.5, "the arrival is one small pop, not a burst"
    assert breath["dt"] == 0 and breath["fi"] <= 18 / 24 + 1e-9, "the breath rises with the glow (2836-2850)"
    end = P.EMBER + (breath["post"] - breath["pre"]) * 24
    assert end < P.END, "the ember's sound ends inside the lane"


def test_ember_arrives_small_and_exact_on_its_first_light():
    assert_ember_arrival(A.RECIPES, A.EXTRA_EVENTS)
    with pytest.raises(AssertionError, match="no sound"):
        assert_ember_arrival(A.RECIPES, [])
    late = [dict(e, t=2859 / 24) if e["id"] == "A.doom.ember" else e for e in A.EXTRA_EVENTS]
    with pytest.raises(AssertionError, match="first light"):
        assert_ember_arrival(A.RECIPES, late)


def assert_piano_entry(gain_of):
    onset = gain_of(np.array([P.PIANO]))[0]
    opened = gain_of(np.array([P.PIANO + 1.5]))[0]
    assert onset < opened, "the first hammer must be softened relative to the ring"
    assert 20 * np.log10(opened) <= -5.9, "the first ring stays small"
    ring = gain_of(np.linspace(P.PIANO + 1.5, P.PIANO_2 - 1e-6, 400))
    assert np.ptp(ring) == 0, "no swell or dip may ride a ringing piano tail"
    after = gain_of(np.linspace(P.PIANO_2 + 1, 3400, 400))
    assert np.all(after == 1), "the score is back at its level under the next note's attack"
    back = gain_of(np.linspace(P.PIANO_2, P.PIANO_2 + 1, 50))
    assert np.all(np.diff(back) >= 0), "the return rides the next note's attack"


def first_pass_gain(frames):
    """The rejected first-pass ride: -12 dB ring from 2862, -6 dB from 2912, unity only at 3120."""
    g6, g12 = 10 ** (-6 / 20), 10 ** (-12 / 20)
    return P.eased_gain(frames, [(2858, 1), (2859, 0), (2862, g12), (2885, g12), (2912, g6), (3096, g6), (3120, 1)])


def test_piano_enters_softened_and_no_ringing_tail_swells():
    assert_piano_entry(P.score_gain)
    with pytest.raises(AssertionError, match="swell or dip"):
        assert_piano_entry(first_pass_gain)
    with pytest.raises(AssertionError, match="softened"):
        assert_piano_entry(lambda f: np.ones(np.shape(f), np.float32))


def test_score_ride_is_bit_identical_outside_its_frames_and_does_not_change_notes():
    # At 240 Hz, ten actual samples represent each frame; no sampler/cache needed.
    rng = np.random.default_rng(23)
    before = rng.normal(size=(6480 * 10, 2)).astype(np.float32)
    after = P.polish_score(before.copy(), sr=240)
    assert np.array_equal(after[:2858 * 10], before[:2858 * 10])
    assert np.array_equal(after[2921 * 10:], before[2921 * 10:])
    assert not np.array_equal(after[2860 * 10:2870 * 10], before[2860 * 10:2870 * 10])
    unbounded = before * np.float32(0.5)
    assert not np.array_equal(unbounded[:2400 * 10], before[:2400 * 10])


LANE_BEDS = {"A.storm.brink": 2400, "A.wind.ash": 2640}      # the delivered beds this lane re-times (their t0 frame)


def assert_inside(bed_timing, extra_beds, extra_events):
    for rid, tm in bed_timing.items():
        assert rid in LANE_BEDS, f"{rid} is not one of this lane's beds"
        assert P.START <= tm.get("t0", LANE_BEDS[rid] / 24) * 24 and tm.get("t1", 0) * 24 <= P.END, rid
    for b in extra_beds:
        if b["id"].startswith("A.doom."):
            assert P.START <= b["t0"] * 24 and b["t1"] * 24 <= P.END, b["id"]
    for e in extra_events:
        if e["id"].startswith("A.doom."):
            assert P.START <= e["t"] * 24 < P.END, e["id"]


def test_every_changed_cue_lies_inside_the_lane():
    assert_inside(A.BED_TIMING, A.EXTRA_BEDS, A.EXTRA_EVENTS)
    with pytest.raises(AssertionError, match="not one of this lane"):
        assert_inside(dict(A.BED_TIMING, **{"A.wind.night": dict(t0=2832 / 24)}), A.EXTRA_BEDS, A.EXTRA_EVENTS)
    with pytest.raises(AssertionError, match="A.doom.late"):
        assert_inside(A.BED_TIMING, A.EXTRA_BEDS + [dict(id="A.doom.late", t0=3300 / 24, t1=3400 / 24)],
                      A.EXTRA_EVENTS)


def test_ap2_forwards_timing_and_score_polish_hooks():
    assert AP2.BED_TIMING is A.BED_TIMING
    assert AP2.polish_score is A.polish_score
    assert AP2.EXTRA_EVENTS is A.EXTRA_EVENTS
    assert P.score_gain([2400, 2857, 2921, 3120, 3360]).tolist() == [1, 1, 1, 1, 1]


def test_delivered_premaster_audit_detects_one_changed_sample_outside(tmp_path):
    import soundfile as sf
    from polishdoom_audio_audit import outside_identity
    original = np.zeros((6 * P.FR, 2), np.float32)
    before, after = tmp_path / "before.wav", tmp_path / "after.wav"
    sf.write(before, original, P.SR, subtype="FLOAT")
    candidate = original.copy()
    candidate[3 * P.FR] = 0.1
    sf.write(after, candidate, P.SR, subtype="FLOAT")
    assert outside_identity(before, after, start=2, end=4)["ok"]
    for sample in (P.FR, 5 * P.FR):
        broken = candidate.copy()
        broken[sample, 0] = 1e-7
        sf.write(after, broken, P.SR, subtype="FLOAT")
        result = outside_identity(before, after, start=2, end=4)
        assert not result["ok"] and result["different_sample_frames"] == 1
