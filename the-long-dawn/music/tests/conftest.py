"""Shared fixtures for the music tests (SOUND-SCORE-C, 29 Sep 2026).

    python -m pytest the-long-dawn/music/tests -q

The score modules import sampler.py, which imports `soxr` (its resampler) at module level, and K.anticipate reads
every sample's attack from the VSCO-2-CE WAVs. The note-data proofs need neither: when `soxr` is absent a stub module
stands in (nothing is resampled here), and every attack time is 10 s, so every anticipation sits at its cap (the
worst case for the hard silence; and the same numbers on every machine, samples or not).
"""
import importlib.util
import os
import sys
import types

import pytest

MUSIC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(MUSIC, "src")
ROOT = os.path.dirname(MUSIC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _stub_sampler_deps():
    if "soxr" not in sys.modules and importlib.util.find_spec("soxr") is None:
        sys.modules["soxr"] = types.ModuleType("soxr")
    import kit_v3
    kit_v3.attack_time = lambda pkey, pitch, level: 10.0
    return kit_v3


@pytest.fixture(scope="session")
def scores():
    """{'C5': (S, bm), 'C5P2': (S, bm)}: both passes built once, under the worst-case anticipation stub"""
    _stub_sampler_deps()
    import score_v3_C5
    import score_v3_C5P2
    from timeline_v3 import BarMap
    out = {}
    for cut, mod in (("C5", score_v3_C5), ("C5P2", score_v3_C5P2)):
        bm = BarMap(cut)
        out[cut] = (mod.build(bm), bm)
    return out


@pytest.fixture(scope="session")
def patches():
    _stub_sampler_deps()
    import score_v3_C5P2
    p = score_v3_C5P2._patches()
    assert p is not None and "vla_q" in p, "sampler.P (+ sampler_v2's quiet strings) must import for these tests"
    return p


@pytest.fixture
def fresh_scores():
    """a function building a fresh (mutable) pass, for the negative tests"""
    _stub_sampler_deps()
    import score_v3_C5
    import score_v3_C5P2
    from timeline_v3 import BarMap

    def make(cut):
        bm = BarMap(cut)
        return (score_v3_C5P2 if cut == "C5P2" else score_v3_C5).build(bm), bm
    return make
