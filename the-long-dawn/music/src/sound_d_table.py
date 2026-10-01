"""D treatment v2 effects plan; explicit Phase 1 API, independent of D's old scaffold.

No imports render audio or write caches. `build()` returns inherited recipes and
unbound picture requests. A supplied crossing/map selection is an editorial
binding, never a measurement. Source offsets below use D = donor + delta (the
opposite sign to an EDL take's `off`). All ranges are half-open.

    python music/src/sound_d_table.py --output <private-output>/effects_plan.json

This is not a sound_v3 cut. Phase 2 must adopt/measure the delivered edit and
implement the crop/master instructions before delivering a WAV.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FPS, SR, FRAMES = 24, 48000, 9200
SAMPLES_PER_FRAME = SR // FPS
BEATS = (
    ("BLACK", 0, 80), ("FALSE DAWN", 80, 560), ("GLYPHS", 560, 960),
    ("POINT", 960, 1040), ("IGNITION / PROMISE", 1040, 1440),
    ("OLD STORY", 1440, 1680), ("RING BURN", 1680, 1760),
    ("INSCRIPTION", 1760, 2080), ("FORGING", 2080, 2400), ("RACE", 2400, 2720),
    ("DEEP", 2720, 2960), ("BRINK", 2960, 3200), ("VISION", 3200, 3440),
    ("GAP", 3440, 3520), ("REFUSAL", 3520, 3760), ("TRAP", 3760, 4080),
    ("GLOW AT BRINK", 4080, 4240), ("TWO CROWNS", 4240, 4560),
    ("TWO FIRES", 4560, 5040), ("EVERY RIDGE", 5040, 5180),
    ("LAST KINGDOM", 5180, 5680), ("IN STEP", 5680, 5840),
    ("OLD FIRE", 5840, 6080), ("UNFINISHED / LAMPS", 6080, 6400),
    ("ABANDONED DEEP", 6400, 6640), ("CROSSING", 6640, 7040),
    ("WATCH", 7040, 7360), ("TRUE DAWN", 7360, 7840),
    ("TERRACES", 7840, 8080), ("PLENTY", 8080, 8320),
    ("LAST LEAF", 8320, 8640), ("LAST PAGES", 8640, 8880),
    ("TITLE", 8880, 9120), ("HEARTH OUT", 9120, 9200),
)
INPUTS = (
    "music/src/sound_recipes_AP2.py", "music/src/sound_recipes_A.py",
    "music/src/sound_a_table.py", "music/src/sound_c5_table.py",
    "music/src/sound_c5_recipes.py", "music/src/sound_recipes_C.py",
    "music/src/sound_recipes_C5P2.py", "music/src/polish_ignition_A.py",
    "music/src/sound_polishdoom.py", "music/src/sound_polish_edge.py",
    "music/src/polish_c5_sound.py", "music/v3/events_A_measured.json",
    "music/v3/events_C5_measured.json", "music/sound/picture_sync_A.json",
    "music/sound/events_C.json", "music/v3/barmap_AP2.json",
    "music/v3/cues_AP2.json", "music/v3/barmap_C5P2.json",
    "music/sound/c5_sound_events.json",
    "music/src/sound_d_designs.py",
)


def _move(row, cut, delta, span, beat, review=None):
    """Keep donor row/seed/envelope intact; bed crops occur AFTER donor rendering."""
    donor = copy.deepcopy(row)
    out = dict(id=f"D.{beat:02d}.{row['id']}", beat=beat, kind=row["kind"],
               source="derived", donor_cut=cut, donor=donor, delta_f=delta,
               selection=list(span), source_sync=row.get("sync", "donor cue sheet"),
               seed_id=row["id"], review=review,
               status="awaiting_edit" if delta is None else "inherited_reuse")
    if row["kind"] == "event":
        out["hit_f"] = None if delta is None else row["hit_f"] + delta
        out["render_mode"] = "event_recipe"
    else:
        a, b = max(span[0], row["f0"]), min(span[1], row["f1"])
        if a >= b:
            return None
        out.update(f0=None if delta is None else a + delta,
                   f1=None if delta is None else b + delta,
                   donor_crop=[a, b], render_mode="render_full_donor_bed_then_crop")
    return out


def _selection(crossing_start, map_pieces):
    if crossing_start is not None and (type(crossing_start) is not int or not 4880 <= crossing_start <= 5440):
        raise ValueError("crossing requires 400 source frames within A[4880,5840)")
    if map_pieces is not None:
        if len(map_pieces) < 2:
            raise ValueError("map requires an explicit holdout insertion inside the dark stretch")
        end, last_d = 3440, 5180
        insertions = []
        for a, b, d in map_pieces:
            if any(type(v) is not int for v in (a, b, d)):
                raise ValueError("map pieces require integer frames")
            if a != end or not a < b <= 3816 or d < last_d or d + b - a > 5680:
                raise ValueError("map pieces must cover C[3440,3816) once in D[5180,5680), in order")
            if d > last_d:
                insertions.append(a)
            end, last_d = b, d + b - a
        if end != 3816 or not map_pieces or map_pieces[0][2] != 5180:
            raise ValueError("map source coverage or entrance is incomplete")
        if len(insertions) != 1 or not 3722 <= insertions[0] < 3786:
            raise ValueError("map holdout insertion must interrupt the measured one-dark interval")


def build(*, crossing_start=None, map_pieces=None):
    """Return data only. Unfinished cutd Round 3 choices are deliberately not defaults."""
    _selection(crossing_start, map_pieces)
    import sound_a_table as AT
    import sound_c5_table as CT
    from timeline_v3 import BarMap

    aa, cc = AT.build(), CT.build("C5P2")
    bad = AT.problems(aa) + CT.problems(cc)
    if bad:
        raise ValueError("Donor tables failed: " + "; ".join(bad))
    arows = {r["id"]: r for r in aa["events"]}
    crows = {r["id"]: r for r in cc["events"]}
    # Already resolved donor cues; do not read D's 5,920-frame scaffold.
    for r in BarMap("AP2").d["ambience"]:
        if r["id"] in ("A.wind.watch", "A.wind.crossing", "A.air.blue"):
            arows[r["id"]] = dict(r, kind="bed", f0=r["t0"] * FPS, f1=r["t1"] * FPS,
                                  sync="cue:AP2 ambience; not a measured discrete event")
    rows = [dict(id="D.opening.AP2", beat=[1, 2, 3, 4, 5], kind="pcm_copy", source="derived",
                 donor_cut="AP2", path="music/out/v3/sound_AP2_sfx.wav", donor_crop=[0, 1440],
                 f0=0, f1=1440, delta_f=0, status="requires_donor_pcm_identity",
                 render_mode="copy_final_packed_pcm_after_D_master",
                 source_sync="treatment v2 unchanged delivered AP2 opening; includes ignition master hooks")]

    def add(ids, cut, delta, span, beat, review=None):
        source_rows = arows if cut == "AP2" else crows
        for rid in ids:
            r = _move(source_rows[rid], cut, delta, span, beat, review)
            if r is not None:
                rows.append(r)

    add(("C5.hearth.open", "C5.riffle", "C5.pen.mountain", "C5.pen.T1"), "C5P2", 1120,
        (320, 560), 6, "candidate page: confirm original drawing/riffle timing on delivered D")
    add(("C5.burn.deep", "C5.pen.deep"), "C5P2", 1040, (1680, 1920), 11,
        "sweep/backing rebake: inherited anchors require remeasurement")
    add(("C5.hearth.refusal", "C5.pen.refusal_offer", "C5.pen.refusal_figure"), "C5P2", 1440,
        (2080, 2320), 15)
    add(("C5.burn.to_trap",), "C5P2", 1440, (2310, 2346), 16,
        "only valid if source burn layers keep off=-1440; first/half/full at D3756/3770/3780")
    add(["A.wind.watch"] + [rid for rid in arows if rid.startswith("A.ridge.")], "AP2", 1380,
        (3660, 3800), 20, "rendered D still needs sync check; inherited A source positions retained")
    for rid, r in crows.items():
        if rid.startswith("C5.map.beacon"):
            delta = None
            if map_pieces is not None:
                delta = next(d - a for a, b, d in map_pieces if a <= r["hit_f"] < b)
            add((rid,), "C5P2", delta, (3440, 3816), 21,
                "candidate map + piecewise edit; confirm catch pixels and pans in D")
    add(("C5.hearth.deep",), "C5P2", 2160, (4240, 4480), 25,
        "candidate Deep: still page; no invented digging or pen stroke")
    cs = 4880 if crossing_start is None else crossing_start
    add(("A.wind.crossing", "A.watchfire_1", "A.watchfire_far"), "AP2",
        None if crossing_start is None else 6640 - cs, (cs, cs + 400), 26,
        "crossing source selection unbound" if crossing_start is None else
        "explicit editorial selection; preserve original bed duration and envelope before crop")
    # Pending crossing rows' crops are NOT a selected 4880-based excerpt.
    if crossing_start is None:
        for r in rows:
            if r["beat"] == 26:
                r["selection"] = r["donor_crop"] = None
    add(("A.air.blue", "A.fire.blue.measured"), "AP2", 1840, (6000, 6240), 29,
        "fire fade measured on dawnrev_A plus A EDIT overlay; do not assert D overlay identity")
    add(("C5.hearth.end", "C5.pen.tree"), "C5P2", 2880, (5200, 5440), 30,
        "book picture clock 6160:6400; tree pen is inherited contact-sheet timing, not measured onset")
    add(("C5.hearth.end",), "C5P2", 3200, (5440, 5680), 32,
        "post-master voice mask overrides this quiet donor bed")
    add(("C5.hearth.end", "C5.burn.title"), "C5P2", 3200, (5680, 5920), 33,
        "book picture clock 6960:7200; sound hit8915 is not first-light frame8902")

    requests = []

    def need(rid, beat, design, *, kind="event", count=1, prescribed=None, pan=None, note=""):
        _, lo, hi = BEATS[beat - 1]
        requests.append(dict(id="D.new." + rid, beat=beat, design=design, kind=kind, source="est.",
                             status="unbound", window=[lo, hi], hit_f=None,
                             f0=None, f1=None, prescribed_f=prescribed, count=count, pan=pan,
                             measured_ref=None, note=note))

    need("ring_burn", 7, "ring_burn", note="measure first char, opening and full aperture at drawn gilt ring")
    need("inscription.hiss", 8, "inscription_hiss", kind="bed")
    need("inscription.ring", 8, "inscription_ring", kind="series", count=None,
         note="ringing front contacts only as visible; sustained hiss carries the writing")
    need("inscription.stop", 8, "inscription_stop", kind="control", note="cut writing layers mid-letter; no invented impact")
    for beat, name in ((9, "forging"), (10, "race"), (12, "brink"), (16, "trap")):
        need(name + ".fire", beat, "forge_steady", kind="bed")
        need(name + ".giants", beat, "giant_ring", kind="series", count=None,
             note="one metallic response per pictured giant stroke that reaches a Ring end; measured screen pan")
        need(name + ".anvils", beat, "dull_hammer", kind="series", count=None,
             note="other forges duller; measure contacts; do not translate C's unpictured musical grid")
    need("gap.narrows", 12, "glass_tick", kind="series", count=None)
    need("vision.pressure", 13, "vision_pressure", kind="bed", note="rising pressure ends before hard silence")
    need("gap.hammer", 14, "gap_hammer", prescribed=3440,
         note="one impact at cut; coordinate with score anvil so no double attack")
    need("trap.surges", 16, "vision_pressure", kind="bed",
         note="shape old surge recordings around measured sink/surge/return; retain no C trap frame")
    need("glow.wind", 17, "ridge_wind", kind="bed")
    need("crowns.left", 18, "crown_kindle", prescribed=4320, pan=-1.0)
    need("crowns.right", 18, "crown_kindle", prescribed=4320, pan=1.0)
    need("crowns.fire", 18, "forge_steady", kind="bed", note="work continues beneath the two baskets")
    need("ridges.wind", 19, "ridge_wind", kind="bed", note="continue through every ridge5180; crossfade donor A wind")
    need("ridges.catches", 19, "ridge_catch", kind="series", count=None,
         note="include paired first fires and following catches; pans/nearer-louder hierarchy from measured picture")
    need("map.burn", 21, "ring_burn", note="NEW ridge-to-map burn; C3449 is not a D measurement")
    need("holdout.hammer", 21, "holdout_hammer", kind="series", count=None,
         note="visible insert strokes only; C3742/3762 are retired musical placements")
    need("instep.fire", 22, "forge_steady", kind="bed")
    need("instep.hammers", 22, "instep_hammer", kind="series", count=None,
         note="measure convergence/shared pulse; no shutdown breath")
    need("oldfire.hearth", 23, "hearth_low", kind="bed")
    need("oldfire.flare", 23, "old_fire", note="small drawn Ring unmaking; paper and low flare")
    need("oldfire.quill", 23, "page_quill", note="follow the book-hand writing only while it appears")
    need("lamps", 24, "lamp", kind="bed_family", count=None, note="one small close bed per visible lamp; no giant forge strikes")
    need("lamps.forge", 24, "forge_steady", kind="bed", note="lower steady work beneath studying lamps")
    need("crossing.feet", 26, "crossing_footfall", kind="series", count=None,
         note="physical footsteps under wind; source and contact frames pending, no score-WALK substitution")
    need("crossing.creak", 26, "crossing_creak", kind="series", count=None)
    need("watch.wind", 27, "ridge_wind", kind="bed", note="continuous into true dawn; no silent reset at7360")
    need("watch.fires", 27, "lamp", kind="bed_family", count=None, note="burning lights; measure envelopes for any feeding")
    need("dawn.wind", 28, "dawn_wind", kind="bed", note="wind drops with measured light; no birds")
    need("leaf.hearth", 31, "hearth_low", kind="bed")
    need("leaf.quill", 31, "page_quill", note="writing stops mid-word; do not sound a resting pen")
    need("leaf.stop", 31, "inscription_stop", kind="control", note="truncate the page quill at the visible mid-word stop")
    need("leaf.settle", 31, "leaf_settle", note="one settling page; measure its motion")
    need("page.to_blank", 32, "leaf_settle", prescribed=8640,
         note="new page-turn geometry; effects <=-40dBFS throughout8640:8880 and mute8660:8740")
    need("hearth.out", 34, "hearth_low", kind="bed", note="new final fade to exact zero at9200")
    return dict(schema="long-dawn/d-effects-phase1/1", cut="D", phase=1, fps=FPS, sr=SR,
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                frames=FRAMES, beats=[dict(n=i+1, name=n, f0=a, f1=b) for i, (n, a, b) in enumerate(BEATS)],
                edit_selection=dict(crossing_start=crossing_start, map_pieces=map_pieces),
                input_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in INPUTS},
                reuse=rows, requests=requests,
                constraints=dict(vision_silence=[3400, 3440], vision_silence_source="score-lane design; verify/adopt in Phase2",
                                 voice_region=[8640, 8880], voice_peak_dbfs=-40.0,
                                 voice_silence=[8660, 8740], voice_silence_source="proposed shared score/effects reservation (80f)",
                                 gap_frame=3440, crown_frame=4320, birds_allowed=False,
                                 lufs_integrated=-16.0, true_peak_under_dbtp=-1.0),
                excluded=["C5.pen.T7: D uses overlay caption; inspect before adding quill sound",
                          "C5.hammer.alone_1/2: old musical beats, not holdout contacts",
                          "C5.burn.to_map: NEW burn must be measured",
                          "C storm Eye, shutdown breath, C flint, A crater/impact, birds",
                          "old full-length C5 hearth.end: split around inserted D leaf/voice/final fade"])


def problems(table):
    """Structural/timing guard, not proof of delivered-picture sync or audio quality."""
    errors = []
    try:
        canonical = build(**table["edit_selection"])
    except (ValueError, TypeError, KeyError) as exc:
        return ["invalid edit selection: " + str(exc)]
    # This Phase 1 artifact has no override path: new measured bindings belong to
    # the Phase 2 schema. Compare against the complete donor-derived inventory so
    # deleting rows or consistently moving hit AND delta cannot pass validation.
    for field in ("reuse", "requests", "constraints", "beats"):
        if table.get(field) != canonical[field]:
            errors.append("changed Phase1 " + field + " inventory/contract")
    if table.get("generator_sha256") != canonical["generator_sha256"]:
        errors.append("changed Phase1 table generator")
    if table.get("frames") != FRAMES or table.get("fps") != FPS or table.get("sr") != SR:
        errors.append("D v2 format/length changed")
    actual = [(r["name"], r["f0"], r["f1"]) for r in table["beats"]]
    if actual != list(BEATS):
        errors.append("D v2 beat coverage changed")
    ids = [r["id"] for r in table["reuse"] + table["requests"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate sound id")
    for r in table["reuse"]:
        if r.get("source") != "derived":
            errors.append(r["id"] + ": inherited D placement is derived, not measured")
        if r["kind"] == "pcm_copy":
            if (r["f0"], r["f1"], r["donor_crop"], r["path"], r["render_mode"]) != (
                    0, 1440, [0, 1440], "music/out/v3/sound_AP2_sfx.wav", "copy_final_packed_pcm_after_D_master"):
                errors.append("AP2 opening preservation changed")
            continue
        delta = r["delta_f"]
        if delta is None:
            if any(r.get(k) is not None for k in ("hit_f", "f0", "f1")):
                errors.append(r["id"] + ": unbound edit supplied a frame")
            continue
        if r["kind"] == "event":
            if r["hit_f"] != r["donor"]["hit_f"] + delta:
                errors.append(r["id"] + ": wrong donor-to-D offset")
        else:
            a, b = r["donor_crop"]
            if (r["f0"], r["f1"]) != (a + delta, b + delta) or not r["donor"]["f0"] <= a < b <= r["donor"]["f1"]:
                errors.append(r["id"] + ": invalid donor bed crop")
            if r["seed_id"] != r["donor"]["id"] or r["render_mode"] != "render_full_donor_bed_then_crop":
                errors.append(r["id"] + ": donor bed history lost")
    if not any(r["id"] == "D.opening.AP2" for r in table["reuse"]):
        errors.append("AP2 opening missing")
    expected_ridges = {r["id"] for r in __import__("sound_a_table").build()["events"] if r["id"].startswith("A.ridge.")}
    actual_ridges = {r["donor"]["id"] for r in table["reuse"] if r.get("donor", {}).get("id", "").startswith("A.ridge.")}
    if actual_ridges != expected_ridges:
        errors.append("incomplete inherited ridge catches")
    for r in table["requests"]:
        if r["source"] != "est." or r["status"] != "unbound" or any(r.get(k) is not None for k in ("hit_f", "f0", "f1", "measured_ref")):
            errors.append(r["id"] + ": Phase1 request falsely bound or measured")
    if table["constraints"]["voice_region"] != [8640, 8880] or table["constraints"]["voice_peak_dbfs"] > -40:
        errors.append("voice-region effects ceiling weakened")
    if table["constraints"]["vision_silence"] != [3400, 3440] or table["constraints"]["voice_silence"] != [8660, 8740]:
        errors.append("reserved silence changed")
    if table["constraints"]["birds_allowed"]:
        errors.append("birds are forbidden")
    for p, digest in table["input_sha256"].items():
        if p not in INPUTS or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest:
            errors.append("changed donor input: " + p)
    if set(table["input_sha256"]) != set(INPUTS):
        errors.append("incomplete donor input inventory")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    table = build()
    bad = problems(table)
    if bad:
        raise SystemExit("REFUSED: " + "; ".join(bad))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2) + "\n")
    print(f"Phase 1: {len(table['beats'])} beats, {len(table['reuse'])} donor rows, {len(table['requests'])} unbound requests; no WAV")
