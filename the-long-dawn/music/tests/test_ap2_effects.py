"""AP2 follows A's live effects attributes, including hooks added by the owner's other lanes.

These proofs import recipe data only: no sound engine, samples, frames, cache or output directories. The frozen
namespaces are deliberate broken adapters; each contract below is seen rejecting its corresponding failure.
"""
from types import SimpleNamespace

import pytest

import sound_recipes_A as A
import sound_recipes_AP2 as AP2


def _same_attribute(adapter, name):
    assert getattr(adapter, name, None) is getattr(A, name), f"{name} did not follow A"


def _missing_hook(adapter, name):
    assert not hasattr(adapter, name), f"{name} was invented by the adapter"


@pytest.mark.parametrize("name", ["RECIPES", "PICTURE", "EXTRA_BEDS", "EXTRA_EVENTS", "SPACE"])
def test_effects_follow_replaced_a_tables(name, monkeypatch):
    _same_attribute(AP2, name)
    frozen = SimpleNamespace(**{name: getattr(A, name)})
    monkeypatch.setattr(A, name, object())
    _same_attribute(AP2, name)
    # A from-import or copied table goes stale when a timing lane replaces the attribute.
    with pytest.raises(AssertionError, match=f"{name} did not follow A"):
        _same_attribute(frozen, name)


@pytest.mark.parametrize("name", ["SILENCE", "extra_cues", "FUTURE_SOUND_HOOK"])
def test_optional_hooks_follow_addition_replacement_and_removal(name, monkeypatch):
    monkeypatch.delattr(A, name, raising=False)
    _missing_hook(AP2, name)
    # An invented default changes hasattr(), which sound_v3 uses to activate optional hooks.
    with pytest.raises(AssertionError, match=f"{name} was invented by the adapter"):
        _missing_hook(SimpleNamespace(**{name: None}), name)

    monkeypatch.setattr(A, name, lambda *args: ([], []), raising=False)
    _same_attribute(AP2, name)
    with pytest.raises(AssertionError, match=f"{name} did not follow A"):
        _same_attribute(SimpleNamespace(), name)

    frozen = SimpleNamespace(**{name: getattr(A, name)})
    monkeypatch.setattr(A, name, lambda *args: ([], []))
    _same_attribute(AP2, name)
    with pytest.raises(AssertionError, match=f"{name} did not follow A"):
        _same_attribute(frozen, name)

    monkeypatch.delattr(A, name)
    _missing_hook(AP2, name)
    with pytest.raises(AssertionError, match=f"{name} was invented by the adapter"):
        _missing_hook(frozen, name)


# ---------------------------------------------------------------------------------------------------- AP2's own two
def test_ap2_matches_its_effects_to_As_reference_levels_and_never_writes_the_shared_caches():
    assert AP2.REF_LEVEL_CUT == "A" and AP2.CACHE_READ_ONLY is True
    # control: A itself has neither, so sound_v3 A behaves as it always has (its own keys, caches written)
    assert not hasattr(A, "REF_LEVEL_CUT") and not hasattr(A, "CACHE_READ_ONLY")


def _cue(t0=1.0):
    return dict(id="A.wind.test", fx="wind", params={}, gain_db=-20, pan=None, dist=None, t0=t0, t1=t0 + 2.0)


def test_sound_v3_ref_levels_keys_on_the_reference_cut_and_honours_read_only(tmp_path, monkeypatch):
    import json
    import sound_v3 as SV
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(SV, "ref_level", lambda it, kind: calls.append(it["id"]) or dict(I=-30.0, S3=-28.0, M=-26.0,
                                                                                          pk=-20.0))
    cache = tmp_path / "ref_levels_v2.json"
    monkeypatch.setattr(SV, "_REF_CACHE", str(cache))
    bm = SimpleNamespace(cut="AP2")
    # read-only, keyed on A: a miss is computed, and NOTHING is written
    monkeypatch.setattr(SV, "_CACHE_READ_ONLY", True)
    SV.ref_levels(bm, [(_cue(), "bed")], "A")
    assert calls == ["A.wind.test"] and not cache.exists()
    # control: the same call as any other cut makes (read-write) writes the cache, under the reference cut's key
    monkeypatch.setattr(SV, "_CACHE_READ_ONLY", False)
    SV.ref_levels(bm, [(_cue(), "bed")], "A")
    assert cache.exists()
    stored = json.load(open(cache))
    # a hit on A's key: no new computation; AP2's own key (no ref_cut) would miss
    calls.clear()
    monkeypatch.setattr(SV, "_CACHE_READ_ONLY", True)
    SV.ref_levels(bm, [(_cue(), "bed")], "A")
    assert calls == []
    SV.ref_levels(bm, [(_cue(), "bed")])
    assert calls == ["A.wind.test"] and json.load(open(cache)) == stored


def test_sound_v3_src_loudness_read_only(tmp_path, monkeypatch):
    import sound_v3 as SV
    path = tmp_path / "src_loudness.json"
    monkeypatch.setattr(SV, "_SRC_L_PATH", str(path))
    monkeypatch.setattr(SV, "_SRC_L", {"x": 0.0})
    monkeypatch.setattr(SV, "load", lambda ref, a, b: None)
    monkeypatch.setattr(SV, "loud_int", lambda y: -23.0)
    monkeypatch.setattr(SV, "_CACHE_READ_ONLY", True)
    assert SV.src_loudness("fs:1", 0.0, 1.0) == -23.0 and not path.exists()
    monkeypatch.setattr(SV, "_CACHE_READ_ONLY", False)                     # control: read-write writes it
    assert SV.src_loudness("fs:2", 0.0, 1.0) == -23.0 and path.exists()
