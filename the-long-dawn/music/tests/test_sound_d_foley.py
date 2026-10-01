"""Static contracts for opt-in, locally sourced D crossing foley."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import SRC
import sound_d_foley as F
import sound_recipes_B as B


def test_both_foley_roles_use_the_existing_recorded_wood_source():
    assert set(F.RECIPES) == {"crossing_footfall", "crossing_creak"}
    for name in F.RECIPES:
        item = F.selection(name)
        recipe = item["recipe"]
        assert recipe["src"][0] == "fs:734628"
        assert "logs" in item["source"]["recorded_action"]
        assert item["source"]["title"] == "Wood logs - woodshed"
        assert item["source"]["license"] == "http://creativecommons.org/publicdomain/zero/1.0/"
        assert item["auditioned"] is False
        assert "layers" not in recipe  # No inherited fire or any synthetic material.


def test_footfall_keeps_bs_known_contact_but_has_its_own_cropped_treatment():
    recipe = F.build("crossing_footfall")
    assert recipe["src"] == B.feed(3)["layers"][0]["src"] == ("fs:734628", 2.136)
    assert recipe["pre"] < B.feed(3)["layers"][0]["pre"]
    assert recipe["post"] < B.feed(3)["layers"][0]["post"]


def test_creak_edit_anchor_is_not_claimed_to_be_a_measured_onset():
    item = F.selection("crossing_creak")
    assert "not asserted to be a natural acoustic onset" in item["source_anchor_basis"]
    assert item["recipe"]["src"] == ("fs:734628", 8.85)
    assert item["recipe"]["fi"] > 0 and item["recipe"]["fo"] > 0


def test_source_cuts_fit_inside_the_verified_header():
    for name in F.RECIPES:
        item = F.selection(name)
        recipe = item["recipe"]
        header = item["source"]["header"]
        start = recipe["src"][1] - recipe["pre"]
        end = recipe["src"][1] + recipe["post"]
        assert (start, end) == pytest.approx(item["selected_source_range_s"])
        assert 0 <= start < end <= header["frames"] / header["samplerate"]


def test_level_targets_are_copied_from_cited_donor_rows():
    rows = {row["id"]: row for row in json.loads((Path(SRC).parent / "sound/events_B.json").read_text())}
    for name in F.RECIPES:
        item = F.selection(name)
        row = rows[item["level_from"].split(":")[1]]
        assert row["kind"] == "event"
        assert item["recipe"]["level"] == row["level"]
        assert "target" in item["level_basis"]


def test_spatial_binding_and_metadata_are_detached():
    before = deepcopy((F.RECIPES, F.SELECTION, B.RECIPES))
    recipe = F.build("crossing_footfall", pan=-0.3, dist=0.2)
    assert recipe["pan"] == -0.3 and recipe["dist"] == 0.2
    recipe["src"] = ("wrong", 0)
    item = F.selection("crossing_creak")
    item["source"]["header"]["frames"] = 1
    item["recipe"]["level"] = 0
    assert before == (F.RECIPES, F.SELECTION, B.RECIPES)
    for name in F.RECIPES:
        assert not {"pan", "dist", "hit_f", "f0", "f1", "t", "t0", "t1"}.intersection(F.build(name))
    with pytest.raises(TypeError):
        F.build("crossing_footfall", hit_f=100)


@pytest.mark.parametrize("key,value", [("pan", -1.01), ("pan", 1.01), ("pan", float("nan")),
                                        ("dist", -0.01), ("dist", 1.01), ("dist", float("inf")),
                                        ("pan", True), ("dist", "far")])
def test_invalid_spatial_bindings_are_refused(key, value):
    with pytest.raises(ValueError, match=key):
        F.build("crossing_creak", **{key: value})


def test_foley_is_opt_in_and_does_not_resolve_phase_one_templates():
    import sound_d_designs as phase_one
    for name in F.RECIPES:
        assert phase_one.design(name)["recipe"] is None
    code = ("import sys; import sound_d_foley as F; "
            "assert 'sound_v3' not in sys.modules; assert 'soundfile' not in sys.modules; "
            "assert 'numpy' not in sys.modules; assert 'freesound_v3' not in sys.modules")
    run = subprocess.run([sys.executable, "-B", "-c", code], cwd=SRC, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
