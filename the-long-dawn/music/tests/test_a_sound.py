"""A's second half (3600-6479): the effects table stays on the measured picture (29 Sep 2026, night).

    python -m pytest the-long-dawn/music/tests/test_a_sound.py -q

No recording, sound cache or frame is opened: these tests read the measured file (music/v3/events_A_measured.json),
the table (music/sound/a_sound_events.json), picture_sync_A.json and the recipes module's inputs to sound_v3.
"""
import copy
import json
import os
import subprocess

import pytest

import sound_a_table as T

ROOT = os.path.dirname(os.path.dirname(T.MUSIC))            # the git worktree


@pytest.fixture(scope="module")
def measured():
    return T._load(T.MEASURED)


@pytest.fixture(scope="module")
def table():
    return T.build()


def _frames(r):
    return [("hit_f", r["hit_f"])] if r["kind"] == "event" else [("f0", r["f0"]), ("f1", r["f1"])]


def test_committed_table_is_what_the_measurements_give(table):
    """a_sound_events.json is regenerated from events_A_measured.json, never hand-edited"""
    assert open(T.OUT).read() == T.dumps(table)
    assert table["measured_sha256"] == T.sha(T.MEASURED)


def test_every_row_keeps_its_contract(table, measured):
    assert T.problems(table, measured) == []


def test_every_frame_is_pinned_to_its_measurement(measured):
    """the COMMITTED table (not a rebuild): each hit, and each bed's first frame, is the measured frame its `sync`
    names; a bed ends on the frame after its measured last (half-open)"""
    committed = T._load(T.OUT)
    for r in committed["events"]:
        if r["kind"] == "event":
            assert r["hit_f"] == T.resolve(measured, r["sync"]), r["id"]
        else:
            s0, s1 = r["sync"].split(" .. ")
            assert r["f0"] == T.resolve(measured, s0), r["id"]
            assert r["f1"] == T.resolve(measured, s1.split(" ")[0]) + 1, r["id"]


def test_a_one_frame_drift_in_the_table_is_refused(table, measured):
    """negative control: every frame of every row, moved by one, fails the contract"""
    for src in table["events"]:
        for field, _ in _frames(src):
            for step in (-1, 1):
                t = copy.deepcopy(table)
                row = next(r for r in t["events"] if r["id"] == src["id"])
                row[field] += step
                assert any("drifted" in p for p in T.problems(t, measured)), (src["id"], field, step)


def test_a_re_measured_frame_makes_the_table_stale(table, measured):
    """negative control from the other side: if a catch is re-measured one frame later, the committed table no longer
    matches its measurement, and the rebuild moves with it"""
    m = copy.deepcopy(measured)
    ev = next(e for e in m["events"] if e["id"] == "run.link_7")
    ev["frames"]["first"] += 1
    assert any("A.run.link_7" in p and "drifted" in p for p in T.problems(table, m))
    rebuilt = next(r for r in T.build(m)["events"] if r["id"] == "A.run.link_7")
    assert rebuilt["hit_f"] == ev["frames"]["first"]


def test_the_run_comes_nearer_and_louder(table):
    """the beacon run far to near: each link louder and no farther than the one before, the last the loudest"""
    run = sorted((r for r in table["events"] if r["id"].startswith("A.run.link_") and not r["id"].endswith("_flare")),
                 key=lambda r: int(r["id"].rsplit("_", 1)[1]))
    assert len(run) == 7
    assert all(b["level"] > a["level"] and b["dist"] <= a["dist"] for a, b in zip(run, run[1:]))
    assert run[-1]["level"] == max(r["level"] for r in table["events"] if r["id"].startswith("A.run."))
    lv = T.levels()
    assert [r["level"] for r in run] == sorted(lv[k] for k in T.RUN_LEVELS)


def test_a_receding_run_is_refused(table, measured):
    """negative control: the retired ramp (link 7 the farthest) breaks the contract"""
    t = copy.deepcopy(table)
    next(r for r in t["events"] if r["id"] == "A.run.link_7")["dist"] = 0.65
    assert any("nearer and louder" in p for p in T.problems(t, measured))


def test_the_ridge_catches_together_sit_at_the_approved_flare(table, measured):
    ridge = [r for r in table["events"] if r["id"].startswith("A.ridge.")]
    catches = [e for e in measured["events"] if e["id"].startswith("ridges.fire_") and e.get("kind") == "catch"]
    assert len(ridge) == len(catches) > 0
    assert all(r["pre"] == 0.0 for r in ridge)              # nothing is heard before the first light
    t = copy.deepcopy(table)
    for r in t["events"]:
        if r["id"].startswith("A.ridge."):
            r["level"] += 3.0
    assert any("ridge catches together" in p for p in T.problems(t, measured))


def test_levels_are_approved(table):
    """every level is its approved reference, or that reference plus a stated rule (design); beds sit inside the
    approved fire-bed range (A.fire.blue -69.8 .. C+.fire.rises -33.6), events inside the approved flare range except
    the ridge catches, whose SUM is the approved flare"""
    lv = T.levels()
    fire_beds = [lv[k] for k in ("A.fire.blue", "A.x.take", "C+.fire.rises", "C+.fire.remains", "C.fire.born",
                                 "C.hearth.open", "C.hearth.end", "C.hearth.council")]
    flares = [lv[f"C.beacon{k}"] for k in range(1, 8)]
    for r in table["events"]:
        ref = r["level_from"].split()[1]
        assert ref in lv, r["id"]
        assert r.get("design") or r["level"] == lv[ref], r["id"]
        if r["kind"] == "bed":
            assert min(fire_beds) <= r["level"] <= max(fire_beds), r["id"]
        elif not r["id"].startswith("A.ridge."):
            assert min(flares) <= r["level"] <= max(flares), r["id"]


def test_pinned_A_levels_are_the_committed_render_report():
    """A_APPROVED copies SOUND's last A render (events_A.json as committed); a render rewrites the working file"""
    try:
        head = subprocess.run(["git", "-C", ROOT, "show", "HEAD:the-long-dawn/music/sound/events_A.json"],
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"git HEAD not readable here: {e}")
    committed = {e["id"]: e.get("level") for e in json.loads(head)}
    for k, v in T.A_APPROVED.items():
        assert committed[k] == v, k


def test_recipes_play_the_table_and_retire_the_bar_grid():
    """sound_recipes_A (and AP2, which forwards to it) hands sound_v3 exactly these rows; the replaced bar-grid cues
    are skipped with a reason; PICTURE moves none of the rows; the first half is untouched"""
    import sound_recipes_A as R
    import sound_v3 as SV
    from timeline_v3 import BarMap, _resolve_t
    t = R.A_TABLE
    assert t["events"] == T._load(T.OUT)["events"]
    for rid, why in t["replaced"].items():
        assert R.RECIPES[rid].get("skip") and R.RECIPES[rid]["why"] == why
    ev = {e["id"]: e for e in R.EXTRA_EVENTS}
    beds = {b["id"]: b for b in R.EXTRA_BEDS}
    sync = SV.sync_table("A")
    for r in t["events"]:
        rc = R.RECIPES[r["id"]]
        assert rc["level"] == r["level"] and not rc.get("skip") and "level_from" not in rc
        if r["kind"] == "event":
            f = round(_resolve_t(ev[r["id"]]["t"], sync) * 24)
            assert f == r["hit_f"] and R.PICTURE.get(r["id"], f) == f
            assert rc["pre"] == 0.0 and rc["dist"] == r["dist"]
        else:
            b = beds[r["id"]]
            assert (round(b["t0"] * 24), round(b["t1"] * 24)) == (r["f0"], r["f1"])
            assert b["fade_in"] == r["fade_in"]
    bm = BarMap("A")
    live = {c["id"] for c in bm.d["sfx"] + bm.d["ambience"] if not R.RECIPES.get(c["id"], {}).get("skip")}
    assert {"A.impact", "A.strike1", "A.blow", "A.roar", "A.karst_flare", "A.desert_fire", "A.wind.watch",
            "A.wind.crossing", "A.air.blue"} <= live
    assert not live & set(t["replaced"])
    assert R.PICTURE["A.karst_flare"] == 3815 and R.PICTURE["A.desert_fire"] == 3880


def test_picture_sync_is_current_and_nothing_is_pending(table):
    cur = T._load(T.PSYNC)
    assert T.picture_sync(table, cur) == cur
    assert "pending" not in cur and "not_measurable" not in cur
    rows = cur["second_half"]["rows"]
    assert set(rows) == {r["id"] for r in table["events"]}
    text = json.dumps({k: v for k, v in cur.items() if k != "historical_previous_record"}).lower()
    assert "slate" not in text


def test_the_ending_guard_reads_only_the_watchfire_path():
    """the ending (A19-A20) was measured through the owner's afix_comp; an edit elsewhere in that module (22:10: A2's
    bloom) must not refuse it, while any edit to what draws the watch-fires must (negative controls both ways)"""
    import measure_a_analysis as M
    src = ("import math\nREF_W = 1920.0\n\n\ndef _lin(x):\n    return x\n\n\ndef bloom(o, i, f, t):\n    return o\n\n\n"
           "def watchfires(o, i, f, t):\n    return _lin(o)\n\n\nKINDS = dict(bloom=bloom, watchfires=watchfires)\n"
           "WATCHFIRES = dict(fires=[(430, 752, 2.0, 6240)])\nA_TRANS = [dict(f0=536, kind='bloom')]\n")
    base = M.watchfire_path(src)[0]
    assert M.watchfire_path(src.replace("return o\n", "return i\n"))[0] == base          # bloom's body
    assert M.watchfire_path(src.replace("f0=536", "f0=540"))[0] == base                  # A's other windows
    assert M.watchfire_path(src.replace("6240", "6241"))[0] != base                      # a pale frame
    assert M.watchfire_path(src.replace("return _lin(o)", "return o"))[0] != base        # the drawing
    assert M.watchfire_path(src.replace("return x\n", "return 2 * x\n"))[0] != base      # a shared helper
    assert M.watchfire_path(src.replace("import math\n", "import math\nimport cv2\n"))[0] != base


def test_measured_file_is_what_the_evidence_gives():
    """events_A_measured.json is regenerated from the evidence batches (outside the repo: skipped where absent)"""
    import measure_a_analysis as M
    if not M.POINTS.exists():
        pytest.skip(f"evidence batches not on this machine ({M.POINTS})")
    assert M.dumps(M.build()) == M.OUT.read_text()
