"""The measured C5 event map (music/v3/events_C5_measured.json): schema, provenance, the load-bearing facts, and
negative mutations that the validator must catch. With the delivered frames present (~/ldfarm/out or $LD_FRAMES),
two shots are re-measured and their frame-set hashes recomputed."""
import copy
import json
import os

import pytest

import measure_c5_events as M

TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v3", "events_C5_measured.json")


@pytest.fixture(scope="module")
def table():
    with open(TABLE_PATH) as fh:
        text = fh.read()
    return json.loads(text), text


def ev(d, eid):
    return next(e for e in d["events"] if e["id"] == eid)


def test_schema_and_invariants_hold(table):
    d, text = table
    assert M.validate_table(d, text) == []


def test_generated_by_this_measurer(table):
    assert M.is_current(table[0]), "events_C5_measured.json is stale: re-run measure_c5_events.py"


def test_every_delivered_shot_is_covered(table):
    d, _ = table
    assert set(d["shots"]) == set(M.SHOTS)
    assert {e["shot"] for e in d["events"]} == set(M.SHOTS)


def test_load_bearing_facts(table):
    d, _ = table
    assert ev(d, "reveal.both_fires_ignite")["frames"]["frame"] == 2880
    assert ev(d, "cold.forges_off")["frames"]["frame"] == 3848
    last = ev(d, "map.last_catch")["frames"]
    assert (last["first"], last["full"]) == (3786, 3791)
    dark = ev(d, "map.one_dark")["frames"]
    assert (dark["first"], dark["last"]) == (3722, 3785)
    assert ev(d, "map.all_lit")["frames"]["first"] == 3791
    assert [ev(d, f"map.beacon_{k}")["frames"]["first"] for k in range(2, 9)] == [3468, 3523, 3584, 3647, 3680,
                                                                                 3718, 3786]
    hold = ev(d, "unfinished.still_hold")["frames"]
    assert (hold["first"], hold["last"]) == (4217, 4239)
    assert ev(d, "trap.hammer_strikes")["frames"]["found"] == []


def test_retired_catch_is_listed_as_a_disagreement(table):
    d, _ = table
    rows = [r for r in d["barmap_C5_vs_picture"] if r["barmap_id"] == "last_beacon"]
    assert rows and rows[0]["barmap_frame"] == 3760 and rows[0]["measured_frame"] == 3786


def test_no_private_paths(table):
    _, text = table
    for bad in ("/Users/", "/home/", "/private/", "Downloads/"):
        assert bad not in text


@pytest.mark.parametrize("mutate, expect", [
    (lambda d: ev(d, "cold.forges_off")["frames"].update(frame=4000), "outside its shot"),
    (lambda d: ev(d, "reveal.both_fires_ignite").pop("region"), "missing"),
    (lambda d: ev(d, "map.last_catch").update(basis=["INFERRED"]), "lacks MEASURED"),
    (lambda d: ev(d, "map.last_catch").update(confidence="certain"), "confidence"),
    (lambda d: d["events"].append(copy.deepcopy(d["events"][0])), "duplicate id"),
    (lambda d: d["shots"]["map"].update(first=3400), "identity"),
    (lambda d: d["shots"]["cold"].update(frame_set_sha256="00"), "sha256"),
    (lambda d: d.update(schema="long-dawn/c5-events-measured/0"), "schema"),
    (lambda d: ev(d, "map.last_catch").update(evidence="/Users/someone/frames"), "private path"),
    (lambda d: ev(d, "map.last_catch").update(observed_stale=True), "another frame set"),
])
def test_validator_catches(table, mutate, expect):
    d = copy.deepcopy(table[0])
    mutate(d)
    bad = M.validate_table(d)
    assert any(expect in b for b in bad), bad


def _frames_root():
    root = M.frames_root()
    stem, a, b, _ = M.SHOTS["cold"]
    return root if os.path.exists(M.fpath(root, stem, a)) else None


@pytest.mark.skipif(_frames_root() is None, reason="the delivered C5 frames are not on this machine")
@pytest.mark.parametrize("shot", ["cold", "reveal"])
def test_remeasure_matches_table(table, shot):
    """the table is reproducible from the delivered frames: same frame set (sha256), same events"""
    d, _ = table
    root = _frames_root()
    events, identity = M.run([shot], root, verbose=False)
    assert identity[shot]["frame_set_sha256"] == d["shots"][shot]["frame_set_sha256"]
    got = {e["id"]: e["frames"] for e in events}
    want = {e["id"]: e["frames"] for e in d["events"] if e["shot"] == shot}
    assert json.loads(json.dumps(got, default=str)) == want
