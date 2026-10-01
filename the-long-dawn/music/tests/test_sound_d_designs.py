"""D's opt-in palette copies donor sources without assigning picture timing."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import SRC
import sound_c5_recipes as C5
import sound_d_designs as D
import sound_recipes_C as C


def leaves(recipe):
    if "layers" in recipe:
        return [leaf for layer in recipe["layers"] for leaf in leaves(layer)]
    return [recipe]


def refs(recipe):
    return {part[0] for leaf in leaves(recipe)
            for part in (leaf["src"] if isinstance(leaf["src"], list) else [leaf["src"]])}


EXPECTED = {
    "ring_burn": {"fs:528662"}, "inscription_hiss": {"fs:870180"},
    "inscription_ring": {C.ANVIL}, "giant_ring": {C.ANVIL}, "dull_hammer": {C.ANVIL},
    "glass_tick": {"fs:185609"}, "vision_pressure": {"fs:754256"}, "gap_hammer": {C.ANVIL},
    "crown_kindle": {"fs:595483"}, "ridge_catch": {"fs:595483"}, "holdout_hammer": {C.ANVIL_SOFT},
    "instep_hammer": {"fs:386116", "fs:386115"}, "forge_steady": {"fs:595483", "fs:172630"},
    "old_fire": {"fs:528662", "fs:595483"}, "lamp": {"fs:681366"},
    "page_quill": {"fs:194905"},
    "leaf_settle": {"sn:slow_page_turns"}, "ridge_wind": {"fs:725630"},
    "hearth_low": {"fs:681366"}, "dawn_wind": {"fs:725630"},
}
UNRESOLVED = {"inscription_stop", "crossing_creak", "crossing_footfall"}


def test_palette_has_every_required_effect_and_only_inspected_sources():
    assert set(D.DESIGN) == set(EXPECTED) | UNRESOLVED
    for name, sources in EXPECTED.items():
        assert refs(D.build(name)) == sources
        assert "fs:725219" not in sources  # B's robin cannot enter D.


def test_each_event_keeps_a_donor_sample_position_and_beds_are_ranges():
    donors = [recipe for recipe in list(C.RECIPES.values()) + list(C5.INSTEP_RECIPES.values())
              if not recipe.get("skip")]
    positions = {tuple(leaf["src"]) for recipe in donors for leaf in leaves(recipe)
                 if isinstance(leaf["src"], tuple)}
    for name in EXPECTED:
        item = D.design(name)
        for leaf in leaves(item["recipe"]):
            if item["kind"] == "event":
                assert tuple(leaf["src"]) in positions, name
            else:
                assert isinstance(leaf["src"], list), name
                assert all(len(part) == 4 and part[1] < part[2] for part in leaf["src"])
    steam = C.RECIPES["C.snow_hiss"]["layers"][1]
    assert D.build("inscription_hiss")["src"][0][1:3] == (steam["src"][1], steam["src"][1] + steam["post"])


def test_levels_copy_the_cited_row_and_keep_design_provenance():
    sound = Path(SRC).parent / "sound"
    approved = {r["id"]: r for r in json.loads((sound / "events_C.json").read_text())}
    c5 = {r["id"]: r for r in json.loads((sound / "c5_sound_events.json").read_text())["events"]}
    for name in EXPECTED:
        item = D.design(name)
        path, row_id = item["level_from"].split(":")
        row_id = row_id.removesuffix(" + trim_db")
        row = (c5 if "c5_sound_events" in path else approved)[row_id]
        assert item["kind"] == row["kind"]
        assert item["level"] == item["recipe"]["level"] == row["level"] + row.get("trim_db", 0.0)
        assert item["auditioned"] is False and item["design"] is True
        if row.get("design"):
            assert item["level_basis"] == "inherited design target"


def test_event_level_cannot_be_silently_reused_as_a_bed_measurement():
    with pytest.raises(ValueError, match="bed requires a bed level donor"):
        D._proposal("bed", "C.snow_hiss", {}, "C.snow_hiss", "negative control")


def test_factory_and_metadata_copies_cannot_mutate_donors_or_other_instances():
    before = deepcopy((C.RECIPES, C5.INSTEP_RECIPES, D.DESIGN))
    a, b = D.build("old_fire"), D.build("old_fire")
    a["layers"][0]["env"].append((999.0, 999.0))
    a["layers"][1]["src"] = ("bad:replacement", 0)
    meta = D.design("forge_steady")
    meta["recipe"]["layers"][0]["src"].append(("bad:replacement", 0, 1, 1))
    assert b == D.build("old_fire")
    assert before == (C.RECIPES, C5.INSTEP_RECIPES, D.DESIGN)


def test_picture_timing_and_spatial_positions_are_not_inherited():
    timing_keys = {"hit_f", "f0", "f1", "t", "t0", "t1", "frame", "frames"}
    for name in EXPECTED:
        item = D.design(name)
        assert timing_keys.isdisjoint(item)
        for recipe in [item["recipe"], *leaves(item["recipe"])]:
            assert timing_keys.isdisjoint(recipe)
            assert "pan" not in recipe and "dist" not in recipe
    left, right = D.build("crown_kindle", pan=-1.0), D.build("crown_kindle", pan=1.0)
    assert left.pop("pan") == -1 and right.pop("pan") == 1
    assert left == right == D.build("crown_kindle")
    assert D.build("ridge_catch", pan=0.25, dist=0.75)["dist"] == 0.75
    with pytest.raises(TypeError):
        D.build("ridge_catch", hit_f=123)


@pytest.mark.parametrize("key,value", [("pan", -1.01), ("pan", 1.01), ("pan", float("nan")),
                                        ("dist", -0.01), ("dist", 1.01), ("dist", float("inf")),
                                        ("pan", True), ("dist", "near")])
def test_invalid_spatial_binding_is_refused(key, value):
    with pytest.raises(ValueError, match=key):
        D.build("ridge_catch", **{key: value})


@pytest.mark.parametrize("name", sorted(UNRESOLVED))
def test_missing_recordings_and_control_landmark_do_not_render(name):
    item = D.design(name)
    assert item["recipe"] is None and item["level"] is None
    with pytest.raises(ValueError, match="no renderable recipe"):
        D.build(name)
    assert D.design("inscription_stop")["action"] == "cut_existing_layers"
    assert D.design("inscription_stop")["kind"] == "control"


def test_crowns_explicitly_disable_the_inherited_outdoor_send():
    for pan in (-1.0, 1.0):
        recipe = D.build("crown_kindle", pan=pan)
        assert recipe.get("send", C.SPACE["event_send"]) == 0.0
        # Removing the explicit override restores C's nonzero outdoor send.
        del recipe["send"]
        assert recipe.get("send", C.SPACE["event_send"]) > 0.0


def test_page_quill_retains_the_existing_take_and_level():
    recipe = D.build("page_quill")
    assert recipe["src"] == C.RECIPES["C+.pen.deep"]["src"]
    assert recipe["post"] == C.RECIPES["C+.pen.deep"]["post"]
    assert D.design("page_quill")["level_from"] == "music/sound/events_C.json:C+.pen.deep"


def test_continuous_materials_and_instep_pair_keep_their_contracts():
    assert D.design("vision_pressure")["kind"] == D.design("inscription_hiss")["kind"] == "bed"
    assert D.design("lamp")["kind"] == "bed"
    rc = D.build("instep_hammer")
    assert rc["send"] == 0 and rc["no_breath"]
    assert all(layer.get("dt", 0) == 0 and layer["pre"] == 0 for layer in rc["layers"])
    assert D.build("giant_ring")["lp"] > D.build("dull_hammer")["lp"]
    assert "fs:172630" not in refs(D.build("old_fire"))


def test_import_does_not_load_the_renderer_or_modify_baseline_recipes():
    code = (
        "from copy import deepcopy; import sys; import sound_c5_recipes as C5; "
        "import sound_recipes_C as C; before=deepcopy((C.RECIPES,C5.INSTEP_RECIPES)); "
        "import sound_d_designs as D; assert before == (C.RECIPES,C5.INSTEP_RECIPES); "
        "assert 'sound_v3' not in sys.modules; assert 'sound_recipes_D' not in sys.modules; "
        "assert 'soundfile' not in sys.modules; assert 'numpy' not in sys.modules"
    )
    run = subprocess.run([sys.executable, "-B", "-c", code], cwd=SRC, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
