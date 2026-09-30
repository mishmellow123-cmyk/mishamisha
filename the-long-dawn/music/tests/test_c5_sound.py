"""The stale-artefact guards (audio_guard_v3 and its three call sites) and the C5 sound event table: every guard is
shown refusing the stale case and passing the current one."""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from conftest import SRC

import audio_guard_v3 as AG
import sound_c5_table as T

PY = sys.executable
FR = AG.FR                                  # 2,000 samples per frame at 48 kHz / 24 fps


# ------------------------------------------------------------------ the saved pre-master
@pytest.fixture
def pm(tmp_path):
    """a 3-frame 'pre-master' with its bar map and cue sheet, as render_v3 leaves them"""
    npy = tmp_path / "premaster_score_final_X.npy"
    np.save(npy, np.zeros((3 * FR, 2), np.float32))
    bm, cues = tmp_path / "barmap_X.json", tmp_path / "cues_X.json"
    bm.write_text('{"frames": 3}')
    cues.write_text('{"breaths": []}')
    return npy, bm, cues


def test_premaster_of_another_length_is_refused(pm):
    npy, bm, cues = pm
    with pytest.raises(AG.StaleArtefact, match="another cut"):
        AG.check_premaster(str(npy), "C5", 4, str(bm), str(cues))


def test_the_retired_7200_premaster_is_refused_as_c5(tmp_path):
    npy = tmp_path / "premaster_score_final_C5.npy"
    np.lib.format.open_memmap(npy, mode="w+", dtype=np.float32, shape=(7200 * FR, 2)).flush()
    bm = tmp_path / "b.json"
    bm.write_text("{}")
    with pytest.raises(AG.StaleArtefact, match="14400000 samples"):
        AG.check_premaster(str(npy), "C5", 5920, str(bm))


def test_premaster_without_sidecar_warns_for_old_cuts_and_is_refused_for_strict(pm):
    npy, bm, cues = pm
    w = AG.check_premaster(str(npy), "C5", 3, str(bm), str(cues))
    assert len(w) == 1 and w[0].startswith("WARNING")
    with pytest.raises(AG.StaleArtefact, match="requires it"):
        AG.check_premaster(str(npy), "C5P2", 3, str(bm), str(cues))
    assert AG.check_premaster(str(npy), "C5P2", 3, str(bm), str(cues), allow_unverified=True)


def test_premaster_sidecar_pins_bar_map_cues_and_cut(pm, tmp_path):
    npy, bm, cues = pm
    score = tmp_path / "score.py"
    score.write_text("# a score module\n")
    AG.write_premaster_identity(str(npy), "final_C5P2", "C5P2", 3, 3 * FR, str(bm), str(cues), str(score))
    assert AG.check_premaster(str(npy), "C5P2", 3, str(bm), str(cues)) == []
    with pytest.raises(AG.StaleArtefact, match="rendered for cut 'C5P2'"):
        AG.check_premaster(str(npy), "C5", 3, str(bm), str(cues))
    cues.write_text('{"breaths": [1]}')
    with pytest.raises(AG.StaleArtefact, match="cue sheet"):
        AG.check_premaster(str(npy), "C5P2", 3, str(bm), str(cues))
    cues.write_text('{"breaths": []}')
    bm.write_text('{"frames": 3, "edited": true}')
    with pytest.raises(AG.StaleArtefact, match="bar map"):
        AG.check_premaster(str(npy), "C5P2", 3, str(bm), str(cues))


def test_retired_cut_needs_an_explicit_flag():
    with pytest.raises(AG.StaleArtefact, match="retired"):
        AG.check_cut_allowed("C")
    AG.check_cut_allowed("C", allow_retired=True)
    for cut in ("A", "B", "C5", "C5P2"):
        AG.check_cut_allowed(cut)


def test_wav_of_another_length_or_rate_is_refused(tmp_path):
    p = tmp_path / "x.wav"
    sf.write(p, np.zeros((3 * FR, 2), np.float32), 48000, subtype="PCM_24")
    assert AG.check_wav(str(p), 3)
    with pytest.raises(AG.StaleArtefact, match="is 3.00 frames"):
        AG.check_wav(str(p), 4)
    q = tmp_path / "y.wav"
    sf.write(q, np.zeros((6000, 2), np.float32), 44100, subtype="PCM_24")
    with pytest.raises(AG.StaleArtefact, match="44100 Hz"):
        AG.check_wav(str(q), 3)


def test_video_frame_count_guard():
    with pytest.raises(AG.StaleArtefact, match="7200 frames"):
        AG.check_video_frames(7200, "C5", 5920)
    AG.check_video_frames(5920, "C5", 5920)
    AG.check_video_frames(123, "Z", None)


# ------------------------------------------------------------------ the three call sites
def test_sound_v3_cli_refuses_the_retired_cut():
    r = subprocess.run([PY, os.path.join(SRC, "sound_v3.py"), "C"], cwd=SRC, capture_output=True, text=True)
    assert r.returncode != 0 and "REFUSED" in r.stderr and "retired" in r.stderr, r.stderr[-500:]


def test_sound_v3_write_refuses_a_stale_premaster(tmp_path, monkeypatch):
    import sound_v3 as SV
    for d in ("out", "cache", "sound", "v3"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(SV, "OUT", str(tmp_path / "out"))
    monkeypatch.setattr(SV, "CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(SV, "MUSIC", str(tmp_path))
    monkeypatch.setattr(SV, "V3", str(tmp_path / "v3"))
    np.save(tmp_path / "cache" / "premaster_score_final_C5P2.npy", np.zeros((4 * FR, 2), np.float32))
    (tmp_path / "v3" / "barmap_C5P2.json").write_text("{}")

    class BM:
        n, frames, path = 3 * FR, 3, str(tmp_path / "v3" / "barmap_C5P2.json")
    with pytest.raises(SystemExit, match="REFUSED: premaster_score_final_C5P2.npy holds 8000 samples"):
        SV.write("C5P2", np.zeros((3 * FR, 2), np.float32), [], BM())
    np.save(tmp_path / "cache" / "premaster_score_final_C5P2.npy", np.zeros((3 * FR, 2), np.float32))
    with pytest.raises(SystemExit, match="requires it"):                        # right length, no sidecar
        SV.write("C5P2", np.zeros((3 * FR, 2), np.float32), [], BM())


def test_sync_audit_refuses_a_master_of_another_cut(tmp_path, monkeypatch):
    import sync_audit_v3 as SA
    monkeypatch.setattr(SA, "W", 2)
    monkeypatch.setattr(SA, "H", 2)
    monkeypatch.setattr(SA, "NF", {"C5": 5920})
    raw = tmp_path / "C5.rgb"
    monkeypatch.setattr(SA, "raw", lambda cut: str(raw))
    raw.write_bytes(bytes(7200 * 12))
    with pytest.raises(SystemExit, match="REFUSED: the C5 master decodes to 7200 frames"):
        SA.frames("C5")
    raw.write_bytes(bytes(5920 * 12))
    assert SA.frames("C5").shape == (5920, 2, 2, 3)


def test_apply_silence_zeroes_only_its_window():
    import sound_v3 as SV
    sr = SV.SR
    y = np.ones((2 * sr, 2), np.float32)
    SV.apply_silence(y, [(0.5, 1.0)])
    i0, i1 = sr // 2, sr
    assert not y[i0:i1].any()
    assert (y[:i0 - int(0.005 * sr)] == 1).all() and (y[i1:] == 1).all()
    assert 0 < y[i0 - 10, 0] < 1                                              # the 5 ms fade into it
    z = np.ones((sr, 2), np.float32)
    assert (SV.apply_silence(z, ()) == 1).all()


# ------------------------------------------------------------------ the C5 sound event table
@pytest.fixture(scope="module")
def table():
    return T.build("C5P2")


def test_table_snapshot_is_current():
    r = subprocess.run([PY, os.path.join(SRC, "sound_c5_table.py"), "--check"], cwd=SRC, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_table_contract_holds(table):
    assert T.problems(table) == []


def test_table_rows_sit_on_the_measured_frames(table):
    with open(os.path.join(os.path.dirname(SRC), "v3", "events_C5_measured.json")) as fh:
        E = {e["id"]: e["frames"] for e in json.load(fh)["events"]}
    R = {r["id"]: r for r in table["events"]}
    assert R["C5.fire.first_near"]["hit_f"] == R["C5.fire.first_far"]["hit_f"] == E["reveal.both_fires_ignite"]["frame"]
    assert R["C5.forge.surge"]["hit_f"] == E["trap.forges_surge"]["first"]
    assert R["C5.forge.low_returns"]["hit_f"] == E["trap.low_forge_returns"]["first"]
    for k in range(2, 9):
        assert R[f"C5.map.beacon{k}"]["hit_f"] == E[f"map.beacon_{k}"]["first"]
    assert R["C5.forge.race"]["f1"] == 3872
    assert R["C5.forge.instep"]["f0"] == 3848 and R["C5.forge.instep"]["f1"] > 4240
    assert R["C5.pen.refusal_offer"]["hit_f"] == E["refusal.ink_begins"]["first"]
    assert R["C5.pen.refusal_figure"]["hit_f"] == E["refusal.figure_begins"]["first"]
    assert (R["C5.storm.unfinished"]["f0"], R["C5.storm.unfinished"]["f1"]) == (4000, 4216)
    assert "C5.silence" not in R and table["hard_silence"] is None


def test_table_hammers_follow_each_pass_pulse(table):
    p1 = {r["id"]: r for r in T.build("C5")["events"]}
    assert [p1[f"C5.hammer.trap_{k}"]["hit_f"] for k in range(1, 8)] == [2320, 2360, 2440, 2480, 2520, 2560, 2600]
    p2 = {r["id"]: r for r in table["events"]}
    assert [p2[f"C5.hammer.trap_{k}"]["hit_f"] for k in range(1, 8)] == [2320, 2360, 2400, 2483, 2523, 2563, 2603]
    bm = json.load(open(os.path.join(T.V3, "barmap_C5P2.json")))
    beat = 60 * bm["fps"] / bm["bpm"]
    strikes = [r["hit_f"] for r in table["events"] if r["id"].startswith("C5.hammer.instep_")]
    assert strikes == list(range(3860, 4240, 20))
    assert all(f / beat == int(f / beat) for f in strikes)
    assert p1["C5.forge.cold"]["f1"] == p1["C5.silence"]["f0"] == 3848


@pytest.mark.parametrize("mutate, expect", [
    (lambda t: next(r for r in t["events"] if r["id"] == "C5.forge.instep").update(f1=3999), "continuous"),
    (lambda t: next(r for r in t["events"] if r["id"] == "C5.forge.race").update(no_breath=False), "continuous"),
    (lambda t: next(r for r in t["events"] if r["id"] == "C5.forge.surge").update(level=None), "ready without"),
    (lambda t: t["events"].append(copy.deepcopy(t["events"][0])), "duplicate"),
    (lambda t: next(r for r in t["events"] if r["id"] == "C5.burn.title").update(hit_f=6000), "outside the film"),
    (lambda t: next(r for r in t["events"] if r["id"] == "C5.hammer.instep_1").update(hit_f=3861), "every score beat"),
    (lambda t: next(r for r in t["events"] if r["id"] == "C5.hammer.instep_1").update(no_breath=False), "bypass"),
    (lambda t: t["events"].append(dict(id="old.silence", kind="silence", f0=3848, f1=4000, status="ready")), "silence contradicts"),
])
def test_table_contract_catches(table, mutate, expect):
    t = copy.deepcopy(table)
    mutate(t)
    assert any(expect in p for p in T.problems(t)), T.problems(t)


def test_instep_bed_bypasses_real_renderer_breath(monkeypatch):
    """Exercise build's bed path across a real breath envelope; removing the bypass makes this fail."""
    import types
    import sound_v3 as SV
    import sound_c5_recipes as CR
    r = next(r for r in T.build()["events"] if r["id"] == "C5.forge.instep")
    recipe = CR._recipe(r)
    module = types.SimpleNamespace(RECIPES={"test.bed": recipe}, SPACE={}, EXTRA_EVENTS=[], SILENCE=[],
                                   EXTRA_BEDS=[dict(id="test.bed", t0=0.0, t1=1.0, fade_in=0, fade_out=0)])
    bm = types.SimpleNamespace(n=SV.SR, d={}, breath_beats=lambda: [(0.24, 0.96, [])])
    monkeypatch.setitem(sys.modules, "sound_recipes_INSTEP_TEST", module)
    monkeypatch.setattr(SV, "BarMap", lambda cut: bm)
    monkeypatch.setattr(SV, "sync_table", lambda cut: {})
    monkeypatch.setattr(SV, "space_irs", lambda name: None)
    monkeypatch.setattr(SV, "ref_levels", lambda *a: {})
    monkeypatch.setattr(SV, "bed", lambda *a: np.full((SV.SR, 2), 0.1, np.float32))
    monkeypatch.setattr(SV, "limit_crest", lambda y, *a: y)
    monkeypatch.setattr(SV, "match_gain", lambda *a: 0.0)
    monkeypatch.setattr(SV, "proc", lambda y, **kw: y)
    _, continuous, _ = SV.build("INSTEP_TEST", verbose=False)
    recipe["no_breath"] = False
    _, ducked, _ = SV.build("INSTEP_TEST", verbose=False)
    assert np.max(np.abs(continuous[SV.SR // 2])) > 0.09
    assert np.max(np.abs(ducked[SV.SR // 2])) < 0.001


def test_new_c5p2_samples_resolve_inside_existing_sound_cache():
    import sound_c5_recipes as CR
    import sound_v3 as SV
    recipes, _, _, _, _ = CR.assemble(T.build(), "test")

    def refs(spec):
        if "layers" in spec:
            return [ref for layer in spec["layers"] for ref in refs(layer)]
        src = spec["src"]
        return [s[0] for s in src] if isinstance(src, list) else [src[0]]

    for eid, recipe in recipes.items():
        if not (eid.startswith(("C5.forge.race", "C5.forge.instep", "C5.hammer.gather_", "C5.hammer.instep_"))):
            continue
        for ref in refs(recipe):
            path = os.path.realpath(SV.src_path(ref))
            assert os.path.commonpath([path, os.path.realpath(SV.LIB)]) == os.path.realpath(SV.LIB), ref


def test_approved_hammers_keep_original_recipes_and_sources():
    import sound_c5_recipes as CR
    import sound_recipes_C as RC
    import sound_v3 as SV
    rows = {r["id"]: r for r in T.build()["events"]}
    for prefix, count, recipe, filename in (("trap", 7, "C5.hammer", "Anvil_Hit1_v3_Sum.wav"),
                                            ("alone", 2, "C5.hammer.faint", "Anvil_Hit1_v2_Sum.wav")):
        for k in range(1, count + 1):
            row = rows[f"C5.hammer.{prefix}_{k}"]
            assert row["recipe"] == recipe
            expected = dict(RC.RECIPES[recipe], level=row["level"])
            assert CR._recipe(row) == expected
            assert expected["src"] == (f"vsco:Percussion/{filename}", 0.002)
            assert os.path.isfile(SV.src_path(expected["src"][0]))


def test_instep_pair_has_no_scattered_layer_delays():
    from sound_c5_recipes import INSTEP_RECIPES
    rc = INSTEP_RECIPES["C5.hammer.instep"]
    assert len(rc["layers"]) == 2
    assert all(layer.get("dt", 0.0) == 0.0 and layer["pre"] == 0.0 for layer in rc["layers"])
    assert rc["send"] == 0.0 and rc["no_breath"]


def test_edl_of_the_retired_cut_is_refused(tmp_path):
    p = tmp_path / "edl_C.json"
    p.write_text(json.dumps(dict(frames=7200, shots=[])))           # the retired cut, as a fixture, not repo state
    edl, note = T.c5_edl(str(p))
    assert edl is None and "7200-frame" in note


def test_committed_edl_is_the_c5_cut():
    """Since EDIT-C5 (975d644) the committed edl_C.json is the 5,920-frame cut, and the table reads it."""
    edl, note = T.c5_edl()
    assert edl is not None and edl["frames"] == T.FRAMES, note


def _c5_edl(tmp_path, pieces):
    shots = [dict(sec="C12", f0=a, f1=b, kind="black" if s0 is None else None,
                  takes=[] if s0 is None else [dict(stem="ring_C", off=s0 - a)]) for a, b, s0 in pieces]
    p = tmp_path / "edl_C.json"
    p.write_text(json.dumps(dict(frames=5920, shots=shots, decisions=[dict(id="C12_FLINT", choice="A")])))
    return str(p)


def test_flint_lands_through_a_c5_edl(tmp_path):
    """EDIT-C5's candidate A (975d644 edl_v3.FLINT_CANDIDATES): its own event table gives strike1 2650, strike3
    2698, catch 2836; the sound table must agree"""
    p = _c5_edl(tmp_path, ((2640, 2670, 2970), (2670, 2880, 3150)))
    R = {r["id"]: r for r in T.build("C5P2", edl_path=p)["events"]}
    got = {k: R[f"C5.flint.{k}"]["hit_f"] for k in ("strike1", "strike3", "blow", "catch")}
    assert got == dict(strike1=2650, strike3=2698, blow=2724, catch=2836)
    assert all(R[f"C5.flint.{k}"]["status"] == "ready" for k in got)
    assert "C5.flint.strike2" not in R


def test_flint_waits_while_edit_has_not_chosen(tmp_path):
    p = tmp_path / "edl_C.json"
    p.write_text(json.dumps(dict(frames=5920, shots=[dict(sec="C12", f0=2640, f1=2880, kind="decision", takes=[])],
                                 decisions=[dict(id="C12_FLINT", choice=None)])))
    R = {r["id"]: r for r in T.build("C5P2", edl_path=str(p))["events"]}
    assert all(R[f"C5.flint.{k}"]["hit_f"] is None and R[f"C5.flint.{k}"]["status"].startswith("needs frame")
               for k in ("strike1", "strike3", "blow", "catch"))
    assert "decision_required" in R["C5.flint.strike1"]["sync"]


def test_no_flint_source_frame_reads_the_retired_find(tmp_path, monkeypatch):
    """strike 2 (v1 3009) lies in ring_C's retired find/vision 3000-3149: dropped from C5, and refused even if an EDL
    put a take across it"""
    assert not [k for k, f in T.FLINT_SOURCE.items() if T.RETIRED_SOURCE[0] <= f < T.RETIRED_SOURCE[1]]
    assert [d["id"] for d in T.build("C5P2")["dropped"]] == ["C5.flint.strike2"]
    edl = json.load(open(_c5_edl(tmp_path, ((2640, 2880, 2900),))))       # a (forbidden) take over 2900-3139
    monkeypatch.setitem(T.FLINT_SOURCE, "strike2", 3009)
    assert T.flint_frame(edl, "strike2") is None
    assert T.flint_frame(edl, "strike1") == 2720


# ------------------------------------------------------------------ the recipes modules
def _import(mod, env_extra=None):
    env = dict(os.environ)
    env.pop("LD_SOUND_ALLOW_UNRESOLVED", None)
    env.update(env_extra or {})
    return subprocess.run([PY, "-c", f"import {mod} as R; print(len(R.RECIPES), len(R.SKIPPED), R.SILENCE)"],
                          cwd=SRC, capture_output=True, text=True, env=env)


@pytest.mark.parametrize("mod", ["sound_recipes_C5P2", "sound_recipes_C5"])
def test_every_c5_row_is_resolved(mod):
    """29 Sep: the burns are built (hits on their measured half-open frames), the page turn and R22 are placed, and the
    forge hammers play the VSCO anvil: both tables import with nothing skipped. The refusal itself is pinned below on a
    synthetic row, so it stays tested now that the real tables have nothing to refuse."""
    r = _import(mod)
    assert r.returncode == 0, r.stderr[-400:]
    n_recipes, n_skipped, silence = r.stdout.strip().splitlines()[-1].split(" ", 2)
    assert int(n_recipes) > 0 and int(n_skipped) == 0
    assert silence == str([] if mod.endswith("C5P2") else [(3848 / 24, 4000 / 24)])


def test_an_unresolved_row_is_refused_unless_told(monkeypatch):
    code = ("import sound_c5_table as T, sound_c5_recipes as R; t = T.build('C5P2'); "
            "t = dict(t, events=t['events'] + [dict(id='X.unresolved', kind='event', hit_f=100, sync='test', recipe=None, "
            "level=None, level_from=None, status='needs source (test)', design=False)]); "
            "rc, ev, beds, sk, de = R.assemble(t, 'T'); print(len(rc), [r['id'] for r in sk])")
    env = dict(os.environ)
    env.pop("LD_SOUND_ALLOW_UNRESOLVED", None)
    r = subprocess.run([PY, "-c", code], cwd=SRC, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "REFUSED: T sound rows not ready" in r.stderr and "X.unresolved" in r.stderr, r.stderr[-400:]
    env["LD_SOUND_ALLOW_UNRESOLVED"] = "1"
    r = subprocess.run([PY, "-c", code], cwd=SRC, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip().splitlines()[-1].endswith("['X.unresolved']")
