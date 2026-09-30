"""C5P2's polished sound edges (polish_c5_sound, lane polishcsound 30 Sep): every new behaviour is shown both ways.

The release, the dawn's breath and the bridge are tested on the note data and on synthetic audio; the bounded master
on synthetic audio against a frozen copy of render_v3.master as it was before the exact-exterior option, so A's AP2
path (reference envelope, re-normalised) and every other cut (no reference) are proven unchanged."""
import json
import os

import numpy as np
import pytest

import polish_c5_sound as P

FR, SR = P.FR, P.SR


# ------------------------------------------------------------------ scope and wiring
def test_hooks_are_wired_for_c5p2_and_leave_every_other_cut_alone(monkeypatch):
    import sound_recipes_C5 as R5
    import sound_recipes_C5P2 as R
    assert R.prepare_master is P.prepare_master and R.bound_master is P.bound_master
    assert R.refresh_breath_probes is P.refresh_breath_probes
    assert not hasattr(R5, "prepare_master") and not hasattr(R5, "bound_master")     # pass 1 stays historical
    monkeypatch.setattr(P, "polished_score", lambda *a: pytest.fail("polished another cut"))
    monkeypatch.setattr(P, "dawn_air_bridge", lambda *a: pytest.fail("bridged another cut"))
    monkeypatch.setattr(P, "references", lambda: pytest.fail("bounded another cut"))
    s, x = np.ones((4, 2), np.float32), np.ones((4, 2), np.float32)
    for cut in ("C5", "AP2", "A", "B"):
        s2, x2 = P.prepare_master(cut, s, x)
        assert s2 is s and x2 is x
        x3, opts = P.bound_master(cut, x)
        assert x3 is x and opts == {}
        assert P.refresh_breath_probes(cut, s, [{"start_s": 1.0}]) == [{"start_s": 1.0}]


def test_windows_hold_every_edit_with_room_for_the_joins():
    P.check_windows(5920 * FR)
    for bad in (((300, 424), (3790, 4000), (4672, 4816)),         # the release's first frames outside full weight
                ((300, 424), (3760, 4000), (4700, 4816)),         # the dawn's start inside the envelope's join
                ((300, 424), (4672, 4816), (3760, 4000)),         # unsorted
                ((300, 424), (3760, 4000), (4672, 5921))):        # past the film
        with pytest.raises(ValueError):
            P.check_windows(5920 * FR, bad)


# ------------------------------------------------------------------ the release on the shutdown
def _changed_notes(S, parts):
    out = []
    for pn, pd in parts.items():
        old = S.P(pn).to_dict()
        assert pd["dyn"] == old["dyn"] and len(pd["notes"]) == len(old["notes"])
        for a, b in zip(old["notes"], pd["notes"]):
            if a != b:
                out.append((pn, a, b))
    return out


def test_release_changes_only_the_notes_cut_on_the_shutdown(fresh_scores):
    S, bm = fresh_scores("C5P2")
    parts, log = P.release_parts(S, bm)
    cold = bm.event("forges_cold")["t"] / P.BEAT_S
    changed = _changed_notes(S, parts)
    assert len(changed) == len(S.cut_at_shutdown) == len(log) == 15
    assert sorted(parts) == sorted({pn for pn, *_ in S.cut_at_shutdown})
    for pn, a, b in changed:
        assert {k for k in a if a[k] != b[k]} == {"dur", "kw"}                  # pitch, start, level, seat untouched
        assert b["start"] + b["dur"] == pytest.approx(cold)                    # ends ON the shutdown
        assert b["kw"]["rel"] == P.RELEASE_S[pn] and a["kw"]["rel"] == 0.03   # a release, not the 30 ms cut
        assert {k: v for k, v in b["kw"].items() if k != "rel"} == {k: v for k, v in a["kw"].items() if k != "rel"}
    # the source score is not mutated
    assert all(n.kw.get("rel") == 0.03 for pn, beat, *_ in S.cut_at_shutdown for n in S.P(pn).notes
               if round(n.start, 4) == beat and n.pitch is not None)


def test_release_refuses_an_undesigned_part_or_a_missing_note(fresh_scores, monkeypatch):
    S, bm = fresh_scores("C5P2")
    monkeypatch.setattr(P, "RELEASE_S", {k: v for k, v in P.RELEASE_S.items() if k != "tuba"})
    with pytest.raises(ValueError, match="no release designed for tuba"):
        P.release_parts(S, bm)
    monkeypatch.undo()
    S.cut_at_shutdown = list(S.cut_at_shutdown) + [("vla", 1.0, 2.0, 3.0)]
    with pytest.raises(ValueError, match="not found"):
        P.release_parts(S, bm)


# ------------------------------------------------------------------ the local re-mix and its breaths
class _Part:
    gain_db, pan, width, depth, send = 0.0, 0.0, 1.0, 0.0, 0.0

    def to_dict(self):
        return {}


class _Score:
    def __init__(self, breaths):
        self.breaths, self.eq, self.push, self.groups, self.fader = breaths, {}, [], {}, None

    def used(self):
        return {"pad": _Part()}


def _fake_mix(monkeypatch):
    import render_v3 as RV
    t = np.arange(700 * FR) / SR
    stem = np.repeat((0.5 * np.sin(2 * np.pi * 440 * t))[:, None], 2, 1).astype(np.float32)
    monkeypatch.setattr(RV, "hall_ir", lambda: [np.zeros(8, np.float32)] * 4)
    monkeypatch.setattr(RV, "load_stem", lambda name, key: (0, stem))
    return _Score([(12.0, 12.33, set())])                              # a breath over beats 12-12.33: f240-246.6


def test_local_mix_releases_only_the_named_breath(monkeypatch):
    S = _fake_mix(monkeypatch)
    gated = P.local_mix(S, None, {"pad": "k"}, 200, 300)
    open_ = P.local_mix(S, None, {"pad": "k"}, 200, 300, drop=(12.0,))
    i = (243 - 200) * FR
    lvl = lambda y: 20 * np.log10(np.abs(y[i:i + FR]).mean() + 1e-12)       # noqa: E731
    assert lvl(gated) < -45 and lvl(open_) > -15                             # the -45 dB gate, then no gate
    assert np.array_equal(gated[:(239 - 200) * FR], open_[:(239 - 200) * FR])   # identical before the breath
    with pytest.raises(ValueError, match="not in the cue sheet"):
        P.local_mix(S, None, {"pad": "k"}, 200, 300, drop=(13.0,))


def test_local_mix_refuses_a_passage_the_fader_ride_reaches(monkeypatch):
    S = _fake_mix(monkeypatch)
    S.breaths = []
    S.fader = [(1.0, 0.0), (5.0, -3.0), (9.0, 0.0)]                           # 24..216 f: reaches 200-300's preroll
    with pytest.raises(ValueError, match="fader"):
        P.local_mix(S, None, {"pad": "k"}, 200, 300)
    S.fader = [(1.0, 0.0), (2.0, 0.0)]                                        # over long before the preroll
    P.local_mix(S, None, {"pad": "k"}, 200, 300)


def _smooth_join(y, a, b):
    """no sample step larger than the signal's own at the join points"""
    d = np.abs(np.diff(y[a - 2:a + 2, 0])).max(), np.abs(np.diff(y[b - 2:b + 2, 0])).max()
    return max(d) < 1e-3


def test_blend_is_exact_outside_and_joins_smoothly():
    src = np.zeros((500 * FR, 2), np.float32)
    rep = np.full(((300 - 100) * FR, 2), 0.5, np.float32)
    out = P.blend(src.copy(), 100, 300, rep)
    assert np.array_equal(out[:100 * FR], src[:100 * FR]) and np.array_equal(out[300 * FR:], src[300 * FR:])
    assert np.allclose(out[(100 + P.RAMP_F) * FR:(300 - P.RAMP_F) * FR], 0.5)
    assert _smooth_join(out, 100 * FR, 300 * FR)
    spliced = src.copy()
    spliced[100 * FR:300 * FR] = rep                                          # a hard splice fails the same test
    assert not _smooth_join(spliced, 100 * FR, 300 * FR)
    with pytest.raises(ValueError, match="shape"):
        P.blend(src, 100, 300, rep[:-1])


# ------------------------------------------------------------------ the dawn air's bridge
def test_bridge_fills_the_crossfade_null_without_an_attack(monkeypatch):
    import sound_v3 as SV
    rng = np.random.default_rng(1)
    monkeypatch.setattr(SV, "load", lambda ref, t0, t1: rng.normal(0, 0.1, (round((t1 - t0) * SR), 2)).astype(np.float32))
    monkeypatch.setattr(SV, "proc", lambda y, **kw: y)
    first, last = P.BRIDGE["frames"]
    y = P.dawn_air_bridge(np.zeros((4800 * FR, 2), np.float32))
    assert not np.any(y[:first * FR]) and not np.any(y[last * FR:])
    mid = y[4716 * FR:4724 * FR]
    assert 10 * np.log10(np.mean(mid.astype(np.float64) ** 2)) == pytest.approx(P.BRIDGE["rms_db"], abs=0.5)
    head = np.abs(y[first * FR:first * FR + FR]).max()
    assert head < 0.05 * np.abs(mid).max()                                    # it rises: no attack on 4700
    monkeypatch.setattr(SV, "load", lambda ref, t0, t1: np.zeros((round((t1 - t0) * SR), 2), np.float32))
    with pytest.raises(ValueError, match="silent"):
        P.dawn_air_bridge(np.zeros((4800 * FR, 2), np.float32))
    monkeypatch.setattr(SV, "load", lambda ref, t0, t1: np.ones((10, 2), np.float32))
    with pytest.raises(ValueError, match="shorter"):
        P.dawn_air_bridge(np.zeros((4800 * FR, 2), np.float32))


def test_bridge_reads_a_stretch_neither_bed_plays_there(monkeypatch):
    import sound_recipes_C5P2 as R
    import sound_v3 as SV
    ref, t0 = P.BRIDGE["src"]
    t1 = t0 + (P.BRIDGE["frames"][1] - P.BRIDGE["frames"][0]) / 24
    assert all(s[0] == ref for rid in ("C5.wind.watch", "C5.air.dawn") for s in R.RECIPES[rid]["src"])
    # the stretches the two beds actually play (sound_v3.bed's own segment choice, with their own seeds; nothing is
    # loaded): the bridge must overlap neither, or it would comb with them where they meet
    read = []
    monkeypatch.setattr(SV, "load", lambda r, a, b: read.append((r, a, b)) or np.zeros((round((b - a) * SR), 2), np.float32))
    monkeypatch.setattr(SV, "src_loudness", lambda r, a, b: -30.0)
    monkeypatch.setattr(SV, "proc", lambda y, *a, **k: y)
    for rid, f0, f1 in (("C5.wind.watch", 4480, 4720), ("C5.air.dawn", 4720, 5200)):
        SV.bed(R.RECIPES[rid], (f1 - f0) / 24, SV.rng_for(rid))
    assert read and all(r == ref for r, _, _ in read)
    assert all(t1 <= a or t0 >= b for _, a, b in read), read


# ------------------------------------------------------------------ the table rows (effects half)
def test_riffle_and_wind_rows_change_for_c5p2_only():
    import sound_c5_table as T
    import sound_c5_recipes as CR
    rows = {cut: {r["id"]: r for r in T.build(cut)["events"]} for cut in ("C5P2", "C5")}
    rif, rif1 = rows["C5P2"]["C5.riffle"], rows["C5"]["C5.riffle"]
    assert rif["crest"] == 8.0 and rif["trim_db"] == -3.0 and rif["design"]
    env = np.array(rif["env_after"])
    assert env[0, 1] < env[1, 1] <= env[2, 1] < env[3, 1] < env[4, 1] == 0.0   # rises, flutters, swells to the landing
    assert 326 + env[4, 0] * 24 == pytest.approx(348.8, abs=0.1)
    assert not {"crest", "env_after", "trim_db"} & set(rif1)                   # pass 1 keeps SOUND-C's riffle
    assert rows["C5P2"]["C5.wind.watch"].get("no_breath") and not rows["C5"]["C5.wind.watch"].get("no_breath")
    assert CR._recipe(rif)["crest"] == 8.0 and CR._recipe(rif1)["crest"] == 13.0   # a row's crest overrides


# ------------------------------------------------------------------ the bounded master
def test_bound_master_restores_the_exterior_exactly(tmp_path, monkeypatch):
    import soundfile as sf
    n = 400 * FR
    ref = np.random.default_rng(2).normal(0, 0.1, (n, 2)).astype(np.float32)
    sf.write(tmp_path / "ref_sfx.wav", ref, SR, subtype="FLOAT")
    env = np.full(n, 3.0, np.float32)
    np.save(tmp_path / "ref_env.npy", env)
    monkeypatch.setattr(P, "references", lambda: {"sfx": tmp_path / "ref_sfx.wav", "envelope": tmp_path / "ref_env.npy"})
    monkeypatch.setattr(P, "MASTER_WINDOWS", ((50, 150), (250, 350)))
    monkeypatch.setattr(P, "check_windows", lambda n: None)
    new = ref + 0.05                                                          # changed everywhere: inside AND out
    out, opts = P.bound_master("C5P2", new.copy())
    for a, b in ((0, 50), (150, 250), (350, 400)):
        assert np.array_equal(out[a * FR:b * FR], ref[a * FR:b * FR])        # the delivered effects, bit for bit
    for a, b in ((50, 150), (250, 350)):
        assert np.allclose(out[(a + P.SFX_RAMP_F) * FR:(b - P.SFX_RAMP_F) * FR],
                           new[(a + P.SFX_RAMP_F) * FR:(b - P.SFX_RAMP_F) * FR])
        assert np.abs(out[a * FR:(a + 1) * FR] - ref[a * FR:(a + 1) * FR]).max() < 0.01   # joined, not switched
        assert np.abs(out[(b - 1) * FR:b * FR] - ref[(b - 1) * FR:b * FR]).max() < 0.01
    assert P.REPORT["sfx_exterior_max_abs_before_restore"] == pytest.approx(0.05, abs=1e-6)
    assert opts["renormalize"] is False and opts["tp_limit"] == -1.2 and opts["edit_windows"] == P.MASTER_WINDOWS
    assert np.array_equal(np.asarray(opts["reference_env"]), env)


def test_references_are_pinned_by_hash(tmp_path, monkeypatch):
    import render_v3 as RV
    (tmp_path / "a.bin").write_bytes(b"delivered")
    import hashlib
    pin = {"files": {"sfx": {"filename": "a.bin", "sha256": hashlib.sha256(b"delivered").hexdigest()}}}
    (tmp_path / "pin.json").write_text(json.dumps(pin))
    monkeypatch.setattr(P, "REFERENCE_JSON", tmp_path / "pin.json")
    monkeypatch.setattr(RV, "CACHE", str(tmp_path))
    assert P.references() == {"sfx": tmp_path / "a.bin"}
    (tmp_path / "a.bin").write_bytes(b"re-rendered")
    with pytest.raises(ValueError, match="hash mismatch"):
        P.references()
    os.remove(tmp_path / "a.bin")
    with pytest.raises(ValueError, match="missing"):
        P.references()


def test_the_pinned_manifest_names_the_delivered_c5p2_references():
    pin = json.loads(P.REFERENCE_JSON.read_text())
    assert pin["cut"] == "C5P2" and pin["samples"] == 5920 * FR
    assert {k: v["filename"] for k, v in pin["files"].items()} == {
        "envelope": "polishcsound_reference_master_C5P2.npy", "sfx": "polishcsound_reference_sfx_C5P2.wav"}
    assert all(len(v["sha256"]) == 64 for v in pin["files"].values())


# render_v3.master exactly as it was at b429b7b (before the exact-exterior option), for the differential tests
MASTER_AT_B429B7B = '''
def master(score, sfx, total_n, name, target=TARGET_LUFS, ceil_db=CEIL_DB, fade_out=0.35,
           reference_env=None, edit_windows=()):
    score = MX.highpass(score[:total_n].astype(np.float32), 8.0, order=1)
    sfx = None if sfx is None else MX.highpass(sfx[:total_n].astype(np.float32), 8.0, order=1)
    pre = score if sfx is None else score + sfx
    G = 10 ** ((target - lufs(pre)) / 20)
    for it in range(6):
        x = pre * np.float32(G)
        comp = MX.glue_comp_gain(x, thresh_db=-16.0, ratio=1.5, attack=0.04, release=0.4)
        y = x * comp[:, None]
        lim = limiter_gain(y, ceil_db)
        if sfx is not None:
            lim = np.minimum(lim, limiter_gain(score * np.float32(G) * comp[:, None], ceil_db))
        y *= lim[:, None]
        L = lufs(y)
        del x, y
        if abs(target - L) < 0.08:
            break
        G *= 10 ** ((target - L) / 20)
    env = (np.float32(G) * comp * lim).astype(np.float32)[:, None]
    del comp, lim
    if reference_env is not None:
        env = bounded_master_env(env[:, 0], reference_env, edit_windows)[:, None]
        correction = np.float32(10 ** ((target - lufs(pre * env)) / 20))
        env *= correction
    s_out = score * env
    x_out = None if sfx is None else sfx * env
    fi, fo = int(0.010 * SR), int(fade_out * SR)
    for y in (s_out, x_out):
        if y is None:
            continue
        y[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
        y[-fo:] *= (np.cos(np.linspace(0, np.pi / 2, fo)) ** 2).astype(np.float32)[:, None]
    m_out = s_out if x_out is None else s_out + x_out
    tp = max(true_peak_db(m_out), true_peak_db(s_out))
    tp_limit = -1.0 if reference_env is not None else -1.2
    if tp > tp_limit:
        k = np.float32(10 ** ((tp_limit - 0.05 - tp) / 20))
        s_out *= k
        if x_out is not None:
            x_out *= k
        if reference_env is not None:
            env *= k
        m_out = s_out if x_out is None else s_out + x_out
    rng = np.random.default_rng(0)
    paths = {}
    outs = [(name, m_out)] if x_out is None else [(name + "_score", s_out), (name + "_sfx", x_out), (name, m_out)]
    for nm, y in outs:
        yd = MX.tpdf_dither_24(y, rng)
        assert len(yd) == total_n
        paths[nm] = os.path.join(OUT, nm + ".wav")
        sf.write(paths[nm], yd, SR, subtype="PCM_24")
    np.save(os.path.join(CACHE, f"master_env_{name}.npy"), env[:, 0])
    return paths
'''


def _program(seconds=4.0, loud=0.3, clicks=False):
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(3)
    score = (loud * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t)))[:, None] * [1.0, 0.9]
    sfx = rng.normal(0, 0.05, (len(t), 2)) * (t > 1.5)[:, None]
    if clicks:                          # sharp transients outside the edit window: the true-peak branch must act
        for c in (0.5, 3.0, 3.5):
            i = int(c * SR)
            sfx[i:i + 24] += 0.9 * np.hanning(24)[:, None]
    return score.astype(np.float32), sfx.astype(np.float32)


def _run_both(tmp_path, monkeypatch, clicks=False, **kw):
    import render_v3 as RV
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir(), new_dir.mkdir()
    ns = dict(vars(RV), OUT=str(old_dir), CACHE=str(old_dir))
    exec(MASTER_AT_B429B7B, ns)
    score, sfx = _program(loud=0.05 if clicks else 0.3, clicks=clicks)     # quiet: the clicks end over -1 dBTP
    ns["master"](score.copy(), sfx.copy(), len(score), "x", **kw)
    monkeypatch.setattr(RV, "OUT", str(new_dir))
    monkeypatch.setattr(RV, "CACHE", str(new_dir))
    RV.master(score.copy(), sfx.copy(), len(score), "x", **kw)
    return old_dir, new_dir


def _same(old_dir, new_dir):
    names = sorted(p.name for p in old_dir.iterdir())
    return names == sorted(p.name for p in new_dir.iterdir()) and all(
        (old_dir / nm).read_bytes() == (new_dir / nm).read_bytes() for nm in names)


def _true_peak(path):
    import render_v3 as RV
    import soundfile as sf
    return RV.true_peak_db(sf.read(str(path), dtype="float32", always_2d=True)[0])


@pytest.mark.parametrize("clicks", [False, True])
def test_master_without_a_reference_is_unchanged(tmp_path, monkeypatch, clicks):
    old_dir, new_dir = _run_both(tmp_path, monkeypatch, clicks=clicks)
    assert _same(old_dir, new_dir)


@pytest.mark.parametrize("clicks", [False, True])
def test_master_on_the_ap2_path_is_unchanged(tmp_path, monkeypatch, clicks):
    # polish_ignition_A.bound_master passes exactly these two options
    ref = np.full(4 * SR, 5.0, np.float32)
    old_dir, new_dir = _run_both(tmp_path, monkeypatch, clicks=clicks, reference_env=ref, edit_windows=((30, 60),))
    assert _same(old_dir, new_dir)
    e = np.load(new_dir / "master_env_x.npy")
    assert not np.array_equal(e[:30 * FR], ref[:30 * FR])        # AP2 keeps its re-normalisation scalar outside
    if clicks:                                                    # and its true-peak safety at -1.0 dBTP (-0.05)
        assert _true_peak(new_dir / "x.wav") == pytest.approx(-1.05, abs=0.02)


def test_exact_exterior_master_keeps_the_reference_envelope_bit_for_bit(tmp_path, monkeypatch):
    import render_v3 as RV
    monkeypatch.setattr(RV, "OUT", str(tmp_path))
    monkeypatch.setattr(RV, "CACHE", str(tmp_path))
    score, sfx = _program()
    ref = np.full(len(score), 1.0, np.float32)
    RV.master(score.copy(), sfx.copy(), len(score), "x", reference_env=ref, edit_windows=((30, 60),),
              renormalize=False, tp_limit=-1.2)
    e = np.load(tmp_path / "master_env_x.npy")
    assert np.array_equal(e[:30 * FR], ref[:30 * FR]) and np.array_equal(e[60 * FR:], ref[60 * FR:])
    assert not np.array_equal(e[35 * FR:55 * FR], ref[35 * FR:55 * FR])
    # the re-normalised variant moves the exterior by one scalar: the option is what makes it exact
    RV.master(score.copy(), sfx.copy(), len(score), "y", reference_env=ref, edit_windows=((30, 60),))
    assert not np.array_equal(np.load(tmp_path / "master_env_y.npy")[:30 * FR], ref[:30 * FR])


def test_exact_exterior_master_refuses_a_whole_film_safety_gain(tmp_path, monkeypatch):
    import render_v3 as RV
    monkeypatch.setattr(RV, "OUT", str(tmp_path))
    monkeypatch.setattr(RV, "CACHE", str(tmp_path))
    score, sfx = _program()
    hot = np.full(len(score), 30.0, np.float32)                       # a reference envelope that clips outside
    with pytest.raises(ValueError, match="exact exterior"):
        RV.master(score.copy(), sfx.copy(), len(score), "x", reference_env=hot, edit_windows=((30, 60),),
                  renormalize=False, tp_limit=-1.2)
    RV.master(score.copy(), sfx.copy(), len(score), "y", reference_env=hot, edit_windows=((30, 60),))  # AP2: scales


# ------------------------------------------------------------------ the breath probes
def test_breath_probes_are_remeasured_on_the_polished_premaster():
    probes = [dict(start_s=64.8, end_s=65.0, inside_db=-85.0), dict(start_s=160.303333, end_s=161.533333, inside_db=-200.0),
              dict(start_s=196.391667, end_s=196.666667, inside_db=-103.0)]
    score = np.full((int(200 * SR), 2), 0.01, np.float32)
    out = P.refresh_breath_probes("C5P2", score, probes)
    assert out[0] == probes[0]                                        # the other breaths keep their measurements
    assert all(r["inside_db"] == pytest.approx(-40.0, abs=0.01) and "post-polish" in r["measurement"] for r in out[1:])
    with pytest.raises(ValueError, match="anchor"):
        P.refresh_breath_probes("C5P2", score, probes[:2])
