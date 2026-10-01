"""Opt-in D effects timing against the adopted D edit and scoped picture evidence.

Reads metadata only. Reused effects retain the Phase 1 donor recipe, sample
clock and derived provenance. Scoped cover/mask measurements remain distinct
from native delivered plates; neither establishes an audible D arrival.
Unmeasured spacing and spatial choices remain explicitly authored sound design.

build() returns reuse_plan, new_events, new_beds, controls and constraints.
Event stop_f is exclusive; bed env_f uses absolute D frames. recipe_overrides
applies after copying a palette recipe. Global silence and voice limits must
be enforced after processing, including wet tails. No rendering or writes.
"""
from copy import deepcopy
import hashlib
import json

import sound_d_table as T
import sound_recipes_D_effects as R
from sound_d_designs import build as build_design
import sound_recipes_C as C
import sound_d_measurements as M


FILES = {"edl": "edit/edl/edl_D.json", "barmap": "music/v3/barmap_D.json",
         "cues": "music/v3/cues_D.json", "event_receipts": "music/v3/events_D_measured.json",
         "round4_receipts": "music/v3/events_D_round4_measured.json"}


def _first_mix_adjustment(section, delta_db):
    """Authored response to the archived first combined render, not a source level."""
    return dict(kind="authored mix adjustment", trim_delta_db=delta_db,
        reference_artifact="render_effects_D_first_receipt.json",
        reference_sha256="8bfaa22bf83d875d5bd0641b9923b4fcf1182be6f8ab00370df189db10b2f784",
        measured_mix_sha256="8eb8d9d4501aae0ee550376123ee178a27cafb93561ba993c33289fa622bb73f",
        section=section, measured_relative_max_lu={"D18": -1.96378392041502, "D19": -2.4880370780276184}[section],
        permitted_relative_max_lu=-2.5,
        note="Mix-design trim after the first combined-render level-band failure; source recordings and estimated picture timing are unchanged. Revised levels require a new rendered verification.")


def _phase3_crown_adjustment():
    """An authored reduction, following two measured mixes of the new score."""
    return dict(kind="authored mix adjustment", trim_delta_db=-3,
        reference_artifact="phase3/latest_score_first_verify_effects_D.json",
        reference_sha256="e32d62ef5178735af6cc292c8dfd015c4dd177e448059a982300301e204d0da5",
        measured_mix_sha256="01d18ebc8f8e262965be6a2f15be6f8cba5c0b2cf223905d70354fc4ba74f30c",
        score_sha256="dbf629969f9bd1a1aa0ef2580b4746e486a8dc83d1dc95b073f52f8152b55059",
        section="D18", measured_relative_max_lu=-2.4005954774125033,
        permitted_relative_max_lu=-2.5,
        one_db_trial=dict(reference_artifact="phase3/crown_minus1_verify_effects_D.json",
            reference_sha256="aaa9c5262f28f467740081b98c6c814b90a4f07677981f2b641623d4fa7a3a16",
            measured_mix_sha256="907d9deb0f727873453b01912c668adcfde24175fd096bae8310d96115321b45",
            measured_relative_max_lu=-2.465083270754363, passed=False),
        note="Additional crown-effects reduction after mixing the verified Phase3 score. A new full export must verify the resulting programme; timings and source recordings are unchanged.")


def _load(kind, supplied):
    return deepcopy(supplied) if supplied is not None else json.loads((T.ROOT / FILES[kind]).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _editorial_selection(edl):
    """Check every fixed Phase 1 reuse clock against the current chosen takes."""
    if (edl.get("cut"), edl.get("frames"), edl.get("fps")) != ("D", 9200, 24):
        raise ValueError("D effects require the 9200-frame / 24fps v2 EDL")
    shots = {s["code"]: s for s in edl["shots"]}
    if len(shots) != len(edl["shots"]):
        raise ValueError("duplicate EDL shot code")
    if not edl["shots"] or edl["shots"][0]["f0"] != 0 or edl["shots"][-1]["f1"] != 9200:
        raise ValueError("incomplete D EDL coverage")
    for a, b in zip(edl["shots"], edl["shots"][1:]):
        if a["f1"] != b["f0"]:
            raise ValueError("D EDL gap or overlap")

    def take(code, bounds, stem, off=None, hold=None):
        s = shots[code]
        if (s["f0"], s["f1"]) != bounds or not s.get("takes"):
            raise ValueError(f"{code}: changed reuse interval or no chosen take")
        t = s["takes"][0]
        if t["stem"] != stem or t.get("hold") != hold or (off is not None and t.get("off", 0) != off):
            raise ValueError(f"{code}: changed source clock/identity; rebind the donor effects")
        return s, t

    for code, bounds, stem, off in (
        ("D02", (80, 560), "cand_falsedawn_clear_high_deck", 0),
        ("D03", (560, 960), "embers_A3", 0), ("D04", (960, 1040), "embers_A3", 0),
        ("D05", (1040, 1440), "embers_A3", 0),
        ("D06", (1440, 1680), "cand_t1_current-words-held", -1120),
        ("D11c", (2758, 2945), "book_C", -1040),
        ("D15", (3520, 3760), "book_C5_refusal", -1440),
        ("D20", (5040, 5180), "reveal_A", -1380),
        ("D25", (6400, 6640), "cand_deep_leaned_ladders", -2160),
        ("D29", (7840, 8080), "dawnrev_A", -1840),
        ("D30", (8080, 8320), "book_C", -1920),
        ("D32", (8640, 8880), "cand_pen_soft_spine_metal", -3200),
        ("D33", (8880, 9120), "book_C", -1920),
    ):
        take(code, bounds, stem, off)
    for code, bounds, stem, fallback in (
        ("D11a", (2720, 2728), "cutd_deep_clean", "cand_sweep_soft-entry"),
        ("D11b", (2728, 2758), "cutd_deep_clean", "book_C_ft"),
        ("D11d", (2945, 2960), "cutd_deep_exit", "book_C_ft"),
    ):
        s, t = take(code, bounds, stem, 0)
        if len(s["takes"]) < 2 or (s["takes"][1]["stem"], s["takes"][1].get("off")) != (fallback, -1040):
            raise ValueError(code + ": changed inherited Deep fallback clock")
        expected_need = [2945, 2980] if code == "D11d" else [2720, 2757]
        if t.get("need") != expected_need:
            raise ValueError(code + ": changed Deep rebake source window")
    burns = [t for t in edl["transitions"] if t.get("cover") == "x1_refusal_C5_cover"]
    if len(burns) != 1 or (burns[0]["f0"], burns[0]["f1"], burns[0]["cut"], burns[0].get("layer_off")) != (3750, 3786, 3760, -1440):
        raise ValueError("Refusal burn changed: inherited cover sync requires rebinding")
    s, ct = take("D26", (6640, 7040), "cand_crossing_both_decal_cap", -1460)
    crossing_start = s["f0"] + ct["off"]
    if ct.get("need") != [crossing_start, crossing_start + 399]:
        raise ValueError("crossing source need disagrees with its selected offset")
    pieces = []
    for code, bounds, off in (("D21a", (5180, 5480), -1740), ("D21d", (5554, 5630), -1814)):
        s, t = take(code, bounds, "cand_map_beacon-falloff", off)
        pieces.append([s["f0"] + t["off"], s["f1"] + t["off"], s["f0"]])
    if (shots["D21b"]["f0"], shots["D21b"]["f1"]) != (5480, 5520):
        raise ValueError("holdout insertion changed")
    take("D21c", (5520, 5554), "cand_map_beacon-falloff", hold=3739)
    take("D21e", (5630, 5680), "cand_map_beacon-falloff", hold=3815)
    return dict(crossing_start=crossing_start, map_pieces=pieces)


def build(*, edl=None, barmap=None, cues=None, event_receipts=None, round4_receipts=None,
          measurements=None, frame_root=T.ROOT, evidence_roots=None, race_revision=None, picture_revision=None,
          transition_pass=False):
    """Bind all Phase 1 requests; supplied dictionaries are never mutated."""
    if picture_revision not in (None, "locked") or (picture_revision and race_revision != "round5"):
        raise ValueError("locked picture requires the round5 race sound treatment")
    if race_revision not in (None, "round5"):
        raise ValueError("unknown race sound revision")
    if type(transition_pass) is not bool or (transition_pass and picture_revision != "locked"):
        raise ValueError("transition pass requires the locked-picture treatment and an explicit boolean")
    edl, bm, cues = _load("edl", edl), _load("barmap", barmap), _load("cues", cues)
    old_receipt, r4_receipt = _load("event_receipts", event_receipts), _load("round4_receipts", round4_receipts)
    selection = _editorial_selection(edl)
    if race_revision == "round5":
        race_shot = next(s for s in edl["shots"] if s["code"] == "D10")
        take = race_shot["takes"][0]
        if (race_shot["f0"], race_shot["f1"], take["stem"], take.get("off", 0)) != (2400, 2720, "embers_D_race", 0):
            raise ValueError("round5 sound requires the adopted native race at its original D clock")
    if (bm.get("cut"), bm.get("frames"), bm.get("fps"), bm.get("bpm"), cues.get("cut")) != ("D", 9200, 24, 72, "D"):
        raise ValueError("D binding requires the v2 D bar map and cue sheet")
    hooks = {h["id"]: h for h in bm["sync"]}
    if len(hooks) != len(bm["sync"]):
        raise ValueError("duplicate D sync hook")
    scoped = M.editorial_receipts(edl, bm, cues, old_receipt, r4_receipt)
    phase1 = T.build(**selection)
    reuse = R.assemble(phase1, allow_partial=True)
    requests = {r["id"].removeprefix("D.new."): r for r in phase1["requests"]}
    if transition_pass:
        from sound_d_bridges import TRANSITION_REQUEST
        requests['unfinished.burn'] = deepcopy(TRANSITION_REQUEST)
    input_hashes = M.raw_input_hashes()
    documents = dict(edl=edl, barmap=bm, cues=cues, event_receipts=old_receipt, round4_receipts=r4_receipt)
    if measurements is not None and any(_digest(value) != _digest(_load(key, None)) for key, value in documents.items()):
        raise ValueError("native overlay cannot bind modified in-memory editorial metadata")
    overlay = M.validate_overlay(measurements, hooks=hooks, request_ids={r["id"] for r in requests.values()},
        reuse_ids={r["id"] for r in reuse["EXTRA_EVENTS"]}, frame_root=frame_root, input_hashes=input_hashes, evidence_roots=evidence_roots, edl=edl)
    original_hooks = deepcopy(hooks)
    for name, point in overlay["hooks"].items():
        hooks[name] = dict(hooks[name], f=point["frame"], source=point["source"], timing_status="measured_" + point["measurement_scope"],
                          measured_ref=point["measured_ref"], measurement_scope=point["measurement_scope"],
                          picture_frame=point["frame"], native_measurement=deepcopy(point))
    for row in reuse["EXTRA_EVENTS"]:
        if row["id"] == "D.11.C5.burn.deep":
            row.update(source="est.", inherited_source_sync=row["source_sync"],
                       source_sync="Provisional donor crest C1699 +1040 = D2739; the adopted Deep entry is rebaked over a new held background. The recipe anchors its loudest mid-sweep point, not its first visible pixel. That crest landmark remains unmeasured.",
                       review="Old donor crest retained provisionally, not promoted to a measured D rebake event.")
    events, beds, controls, used = [], [], [], set()

    def h(name):
        if name not in hooks or name not in cues["events"]:
            raise ValueError("missing D timing hook: " + name)
        out = hooks[name]
        if type(out["f"]) is not int or not 0 <= out["f"] < T.FRAMES:
            raise ValueError("invalid frame for D hook: " + name)
        if cues["events"][name].get("timing_status") != original_hooks[name].get("timing_status"):
            raise ValueError("cue/bar-map provenance disagrees for " + name)
        return out

    def base(name, hook, offset=0, suffix="", **kw):
        q, hook_data = requests[name], h(hook)
        used.add(q["id"])
        row = dict(id=q["id"] + suffix, request_id=q["id"], beat=q["beat"], design=q["design"],
                   source="est.", hook=hook, hook_frame=hook_data["f"], offset_f=offset,
                   hook_timing_status=hook_data.get("timing_status"), hook_source=hook_data["source"],
                   picture_frame=None, measured_ref=None, pan=0.0, dist=0.0, level_trim_db=0.0,
                   timing_note="Authored sound placement from the named hook; no finished D picture/audio sync measurement.")
        if hook_data.get("measured_ref"):
            row.update(inherited_measurement_ref=hook_data["measured_ref"],
                       inherited_measurement_file=hook_data.get("measurement_file"),
                       inherited_measurement_scope=hook_data.get("measurement_scope", "Inherited hook provenance only; new sound timing remains estimated."))
        if offset == 0 and hook in scoped:
            row.update(source="derived", measured_ref=hook_data["measured_ref"],
                       measurement_scope=hook_data["measurement_scope"],
                       timing_note="Sound placement derived from the explicitly scoped editorial receipt; native delivered picture onset and audible arrival are not measured.")
        if offset == 0 and hook in overlay["hooks"]:
            row.update(source="measured", picture_frame=hook_data["f"], measured_ref=hook_data["measured_ref"],
                       measurement_scope=hook_data["measurement_scope"], measurement=deepcopy(overlay["hooks"][hook]["measurement"]),
                       editorial_reference=dict(hook=hook, frame=original_hooks[hook]["f"], source=original_hooks[hook]["source"]),
                       timing_note="Observed landmark within its recorded picture scope; final assembled D picture and audible arrival remain unverified.")
        row.update(kw)
        return row

    def event(name, hook, offset=0, suffix="", **kw):
        row = base(name, hook, offset, suffix, **kw)
        row["hit_f"] = row["hook_frame"] + offset
        if not 0 <= row["hit_f"] < T.FRAMES:
            raise ValueError(row["id"] + ": estimated hit outside D")
        events.append(row)
        return row

    def bed(name, hook, end_hook, *, offset=0, end_offset=0, suffix="", **kw):
        row = base(name, hook, offset, suffix, **kw)
        row.update(f0=row["hook_frame"] + offset, f1=h(end_hook)["f"] + end_offset,
                   end_hook=end_hook, end_hook_frame=h(end_hook)["f"], end_offset_f=end_offset,
                   end_hook_timing_status=h(end_hook)["timing_status"])
        if not 0 <= row["f0"] < row["f1"] <= T.FRAMES:
            raise ValueError(row["id"] + ": estimated bed outside D or empty")
        row.setdefault("fade_in_s", 0.5)
        row.setdefault("fade_out_s", 0.5)
        beds.append(row)
        return row

    def series(name, hook, end_hook, *, spacing=40, offset=0, repeat=False, pans=(-0.55, 0.55), trim=0.0):
        start, stop = h(hook)["f"] + offset, h(end_hook)["f"]
        if repeat:
            rep = h(hook).get("repeat", {})
            step = rep.get("spacing_frames")
            if type(step) is not int or step <= 0 or rep.get("from") != original_hooks[hook]["f"] or rep.get("to") != original_hooks[end_hook]["f"]:
                raise ValueError(hook + ": missing or changed beat-grid repeat")
            spacing = step * 2
        if type(spacing) is not int or spacing <= 0 or start >= stop:
            raise ValueError(name + ": invalid estimated series")
        for i, f in enumerate(range(start, stop, spacing)):
            row = event(name, hook, f - h(hook)["f"], suffix=f".{i + 1:02d}", pan=pans[i % len(pans)],
                  level_trim_db=trim, spacing_f=spacing, sequence_end_hook=end_hook,
                  spacing_basis="D hook repeat, every second planned beat" if repeat else "Authored provisional sound spacing; contacts unmeasured")
            row.update(source="est.", picture_frame=None, measured_ref=None,
                       timing_note="Provisional series timing; an observed starting hook does not measure subsequent contacts.")
            row.pop("measurement", None)

    def cutoff(name, hook, targets):
        row = base(name, hook)
        row.update(action="cut_existing_layers", hit_f=row["hook_frame"],
                   targets=[requests[n]["id"] for n in targets], de_click_frames=1, includes_wet=True)
        controls.append(row)

    def envelope(points):
        values, provenance = [], []
        for hook, offset, gain in points:
            data = h(hook)
            frame = data["f"] + offset
            source = "measured" if hook in overlay["hooks"] and offset == 0 else "derived" if hook in scoped and offset == 0 else "est."
            values.append([frame, gain])
            provenance.append(dict(hook=hook, hook_frame=data["f"], frame=frame, offset_f=offset,
                gain_db=gain, source=source, hook_source=data["source"], hook_timing_status=data["timing_status"],
                measured_ref=data.get("measured_ref"), measurement_scope=data.get("measurement_scope"),
                gain_source="authored envelope"))
        if any(a[0] > b[0] for a, b in zip(values, values[1:])):
            raise ValueError("measured anchors invert an authored envelope; explicitly revise its design")
        return dict(env_f=values, envelope_provenance=provenance)

    event("ring_burn", "drawn_ring_burn_opens", stop_f=h("drawn_ring_burn_page_gone")["f"],
          recipe_overrides={"pre": 0.0},
          timing_note="Estimated sound onset uses the measured 480x201 generated-cover onset1683; native-resolution and finished-picture onset remain unmeasured.")
    bed("inscription.hiss", "inscription_front", "inscription_stops_midletter", fade_out_s=1 / T.FPS)
    series("inscription.ring", "inscription_scripts", "inscription_stops_midletter", spacing=40, trim=-10)
    cutoff("inscription.stop", "inscription_stops_midletter", ["inscription.hiss", "inscription.ring"])
    for name, start, attacks, end, rep in (
        ("forging", "forging", "forging_hammer_strokes", "race", True),
        ("race", "race", "race_beat_surges", "deep", True),
        ("brink", "brink", "brink", "if_it_closed", False),
        ("trap", "trap", "trap", "glow_at_brink", False),
    ):
        bed(name + ".fire", start, end)
        series(name + ".giants", attacks, end, repeat=rep)
        series(name + ".anvils", attacks, end, repeat=rep, offset=20, pans=(-0.3, 0.2, 0.4, -0.15), trim=-6)
    event("gap.narrows", "brink_gap_narrowest", level_trim_db=-8)
    bed("vision.pressure", "if_it_closed", "gap", end_offset=-40,
        fade_out_s=1 / T.FPS,
        **envelope([("if_it_closed", 0, -18), ("vision_ice_eye", 0, -6),
                    ("vision_slit_push", 0, 0), ("gap", -40, 0)]))
    event("gap.hammer", "gap_hammer_lands", recipe_overrides={"send": 0.0, "no_breath": True})
    bed("trap.surges", "trap_forge_slows", "glow_at_brink", level_trim_db=-6,
        **envelope([("trap_forge_slows", 0, -18), ("trap_gold_passes", 0, -12),
                    ("trap_forge_overtaken", 0, 0), ("trap_forge_rejoins", 0, 3), ("glow_at_brink", 0, 0)]))
    bed("glow.wind", "glow_at_brink", "two_fires")
    for side, pan in (("left", -1.0), ("right", 1.0)):
        overrides = {"send": 0.0, "no_breath": True}
        source_recipe = "C.beacon1" if side == "left" else "C.beacon2"
        anchor = deepcopy(C.RECIPES[source_recipe]["layers"][0]["src"])
        if side == "right":
            # Keep crown processing, including zero pre-roll; borrow only the
            # second established source anchor, not beacon2's stretch or gain.
            layers = build_design(requests["crowns.right"]["design"])["layers"]
            layers[0]["src"] = anchor
            overrides["layers"] = layers
        event("crowns." + side, "crown_beacons", pan=pan, recipe_overrides=overrides, level_trim_db=-7,
              level_adjustment_provenance=_first_mix_adjustment("D18", -4),
              phase3_level_adjustment_provenance=_phase3_crown_adjustment(),
              source_choice_provenance=dict(kind="authored source selection", anchor=anchor,
                  source_recipe="music/src/sound_recipes_C.py:" + source_recipe + ".layers[0].src",
                  note="Established fire excerpt; two distinct anchors with identical crown processing. Timing status and picture evidence are recorded on the bound row. No audio audition asserted."),
              spatial_basis="Owner-mandated hard left/right, not measured screen positions")
    bed("crowns.fire", "two_crowns", "two_fires", level_trim_db=-5,
        level_adjustment_provenance=_first_mix_adjustment("D18", -2),
        phase3_level_adjustment_provenance=_phase3_crown_adjustment())
    bed("ridges.wind", "two_fires", "last_kingdom", level_trim_db=-3,
        level_adjustment_provenance=_first_mix_adjustment("D19", -3),
        fade_out_s=(h("last_kingdom")["f"] - h("every_ridge")["f"]) / T.FPS,
        timing_note="Wind overlaps inherited A ridge bed across5040:5180; crossfade in the renderer, do not restart donor source.")
    for i, pan in enumerate((-0.65, 0.65), 1):
        event("ridges.catches", "ridge_pair_match", suffix=f".pair{i}", pan=pan, dist=0.6, level_trim_db=-11,
              level_adjustment_provenance=_first_mix_adjustment("D19", -3),
              spatial_basis="Proposed symmetric distant pair; positions unmeasured")
    # No count/spacing is supplied by ridge_answers. This five-answer bar grid
    # is an authored listening draft, not a detected fire inventory.
    for i, f in enumerate(range(h("ridge_answers")["f"], h("every_ridge")["f"], 80)):
        event("ridges.catches", "ridge_answers", f - h("ridge_answers")["f"], suffix=f".answer{i + 1}",
              pan=(-0.5, 0.5, -0.3, 0.3, 0.0)[i % 5], dist=max(0.1, 0.55 - i * 0.1), level_trim_db=-11 + 2 * i,
              level_adjustment_provenance=_first_mix_adjustment("D19", -3),
              spacing_f=80, spacing_basis="Authored one-answer-per-bar draft; visible catch count is unknown",
              spatial_basis="Authored far-to-near progression; no measured D screen position/size")
    event("map.burn", "last_kingdom_burn", stop_f=h("last_kingdom_burn_ridge_gone")["f"],
          recipe_overrides={"pre": 0.0},
          supersedes=dict(phase=2, hit_f=5180, source="est.", reason="Round 4 measured cover opening supersedes the planned edit boundary; renderer t_open5182 is not a measured onset."))
    event("holdout.hammer", "holdout_first_contact", stop_f=h("map_dark_hold")["f"])
    bed("instep.fire", "in_step_shared_work", "old_fire", recipe_overrides={"no_breath": True})
    series("instep.hammers", "in_step_shared_work", "old_fire", spacing=20, pans=(0.0,))
    bed("oldfire.hearth", "old_fire", "ring_unfinished")
    event("oldfire.flare", "old_fire_ring_falls")
    event("oldfire.quill", "old_fire", offset=24, stop_f=h("ring_unfinished")["f"] - 24,
          recipe_overrides={"post": (h("ring_unfinished")["f"] - h("old_fire")["f"] - 48) / T.FPS},
          timing_note="Proposed page-writing window; D14 is currently an EDIT lower caption, so actual drawn writing remains unverified.")
    if transition_pass:
        event('unfinished.burn', 'ring_unfinished', stop_f=6120)
    for i, pan in enumerate((-0.5, 0.0, 0.5), 1):
        bed("lamps", "unfinished_lamps_rise", "deep_abandoned", suffix=f".{i}", pan=pan, level_trim_db=-5,
            spatial_basis="Three proposed spatial beds; not a counted lamp inventory")
    bed("lamps.forge", "ring_unfinished", "deep_abandoned", level_trim_db=-6)
    # Gradual set-off, then a restrained 40-frame walking pulse. These foley
    # estimates use inherited AP2 motion anchors, not measured foot contacts.
    for i, (hook, off) in enumerate((("crossing_walk_setoff", 32), ("crossing_walk_setoff", 72))):
        event("crossing.feet", hook, off, suffix=f".start{i + 1}", pan=(-0.15, 0.15)[i], level_trim_db=-6)
    series("crossing.feet", "crossing_walk_full", "watch", spacing=40, pans=(-0.15, 0.15), trim=-6)
    for i, hook in enumerate(("crossing_walk_setoff", "crossing_walk_full", "crossing_narrowest", "crossing_watchfire_3"), 1):
        event("crossing.creak", hook, suffix=f".{i}", level_trim_db=-8,
              timing_note="Estimated lantern-motion accent at inherited motion hook; no creak contact has been measured.")
    bed("watch.wind", "watch", "dawn_rose", fade_in_s=0.0, fade_out_s=40 / T.FPS,
        recipe_overrides={"no_breath": True})
    bed("watch.fires", "watch_bar_breath", "terraces", level_trim_db=-3,
        recipe_overrides={"no_breath": True},
        timing_note="One aggregate quiet fire bed through dawn; no new catch or feeding invented.")
    bed("dawn.wind", "long_dawn", "terraces", fade_in_s=40 / T.FPS, fade_out_s=0.5,
        recipe_overrides={"no_breath": True},
        **envelope([("long_dawn", 0, 0), ("dawn_rose", 0, -2), ("dawn_sunlight", 0, -8), ("terraces", 0, -10)]))
    bed("leaf.hearth", "last_written_leaf", "last_pages")
    event("leaf.quill", "last_written_leaf", offset=24, stop_f=h("last_leaf_line_breaks")["f"],
          recipe_overrides={"post": (h("last_leaf_line_breaks")["f"] - h("last_written_leaf")["f"] - 24) / T.FPS})
    cutoff("leaf.stop", "last_leaf_line_breaks", ["leaf.quill"])
    event("leaf.settle", "last_written_leaf", offset=8, level_trim_db=-6)
    event("page.to_blank", "last_pages", stop_f=8660, recipe_overrides={"send": 0.0},
          timing_note="Estimated page turn at EDIT cut8640; suppressed slate transition is not observed motion. Final voice mask applies.")
    bed("hearth.out", "hearth_out", "hearth_out", end_offset=80, fade_in_s=0.0, fade_out_s=80 / T.FPS,
        **envelope([("hearth_out", 0, 0), ("hearth_out", 80, -120)]), recipe_overrides={"no_breath": True})

    missing = set(q["id"] for q in phase1["requests"]) - used
    if missing:
        raise ValueError("unbound Phase 1 requests: " + ", ".join(sorted(missing)))
    def measured_event(row, point, landmark=None):
        before = {k: deepcopy(row.get(k)) for k in ("hit_f", "hook", "hook_frame", "source", "source_sync")}
        row.update(hit_f=point["frame"], source="measured", picture_frame=point["frame"],
                   measured_ref=point["measured_ref"], measurement_scope=point["measurement_scope"],
                   measurement=deepcopy(point["measurement"]),
                   hook=point.get("landmark", landmark or row.get("hook")), hook_frame=point["frame"], offset_f=0,
                   hook_source=point["source"], hook_timing_status="measured_" + point["measurement_scope"],
                   editorial_reference=dict(hook=point.get("editorial_reference", before["hook"]),
                                            previous_binding=before),
                   timing_note="Observed landmark within the recorded picture scope; finished D image and audible arrival remain unverified.")
        if "donor_hit_f" in row:
            row.update(inherited_delta_f=row["delta_f"], delta_f=point["frame"] - row["donor_hit_f"],
                       t=point["frame"] / T.FPS, source_sync=point["source"])
        if "pan" in point:
            # Crown hard-panning is an explicit owner requirement, independent
            # of the measured screen centroids retained in the receipt.
            if row.get("request_id") not in ("D.new.crowns.left", "D.new.crowns.right"):
                row.update(pan=point["pan"], spatial_basis=point["spatial_basis"], pan_basis=point["spatial_basis"])
        row.update(distance_basis="Authored sound perspective retained from the prior design; image pixels do not measure world distance.",
                   level_trim_basis="Authored sound/mix trim retained from the prior design, not measured from picture brightness.")
        if row.get("stop_f") is not None and row["stop_f"] <= row["hit_f"]:
            raise ValueError(row["id"] + ": measured hit is outside its sound cutoff")

    for request_id, point in overlay["events"].items():
        candidates = [r for r in events + reuse["EXTRA_EVENTS"]
                      if r.get("request_id", r["id"]) == request_id]
        if len(candidates) != 1:
            raise ValueError(request_id + ": one-off replacement needs exactly one event; use series for repeated contacts")
        row = candidates[0]
        window = original_hooks[point["editorial_reference"]].get("timing_window")
        if window and not window[0] <= point["frame"] < window[1]:
            raise ValueError(request_id + ": replacement is outside its editorial interval")
        measured_event(row, point)

    replaced_series = []
    for request_id, group in overlay["series"].items():
        existing = [r for r in events if r["request_id"] == request_id]
        if not existing:
            raise ValueError(request_id + ": replacement series has no event design")
        request = requests[request_id.removeprefix("D.new.")]
        first, stop = T.BEATS[request["beat"] - 1][1:]
        coverage = group["coverage"]
        if coverage["first"] > first or coverage["last"] < stop - 1:
            raise ValueError(request_id + ": series search does not cover its whole picture beat")
        events[:] = [r for r in events if r["request_id"] != request_id]
        for point in group["events"]:
            if not first <= point["frame"] < stop:
                raise ValueError(request_id + ": contact outside the request's picture beat")
            row = deepcopy(min(existing, key=lambda r: abs(r["hit_f"] - point["frame"])))
            row["design_reference"] = {k: deepcopy(row[k]) for k in ("id", "hit_f", "pan", "dist", "level_trim_db")}
            row["id"] = request_id + ".measured." + point["id"]
            row.pop("spacing_f", None)
            row["spacing_basis"] = "Explicit measured landmarks from a complete native-plate search; each row names its observed predicate."
            measured_event(row, point, point.get("landmark", point["id"]))
            events.append(row)
        replaced_series.append(dict(request_id=request_id, retired_rows=existing, coverage=deepcopy(coverage),
                                    count=len(group["events"])))

    suppressed = []
    for request_id, decision in overlay["suppressions"].items():
        retired = []
        for rows in (events, beds, controls, reuse["EXTRA_EVENTS"]):
            selected = [r for r in rows if r.get("request_id", r["id"]) == request_id]
            retired.extend(deepcopy(selected))
            rows[:] = [r for r in rows if r not in selected]
        if not retired:
            raise ValueError(request_id + ": suppression has no active rows")
        suppressed.append(dict(request_id=request_id, decision=deepcopy(decision), retired_rows=retired,
                               status="suppressed by explicit evidence-based sound decision"))

    race_design = None
    if race_revision == "round5":
        from sound_d_race import apply as apply_race
        race_design = apply_race(events, beds, overlay)

    for request_id, sides in overlay["bounds"].items():
        candidates = [r for r in events + beds if r["request_id"] == request_id]
        if len(candidates) != 1:
            raise ValueError(request_id + ": end bound needs one event or bed")
        row, point = candidates[0], sides["end"]
        end = point["frame"] + point.get("offset_f", 0)
        if not row.get("hit_f", row.get("f0")) < end <= T.FRAMES:
            raise ValueError(request_id + ": measured end precedes sound start or exceeds film")
        row["end_provenance"] = deepcopy(point)
        if "f0" in row:
            row.update(f1=end, end_hook=point.get("landmark", "measured_completion"),
                       end_hook_frame=point["frame"], end_offset_f=point.get("offset_f", 0),
                       end_hook_timing_status="measured_" + point["measurement_scope"])
        else:
            row["stop_f"] = end
    locked_design = None
    if picture_revision == "locked":
        from sound_d_lock import apply as apply_lock
        locked_design = apply_lock(events, beds, overlay)
    bridge_design = None
    if transition_pass:
        from sound_d_bridges import apply as apply_bridges
        bridge_design = apply_bridges(events, beds, reuse, overlay, edl)

    all_new = events + beds + controls
    if len({r["id"] for r in all_new}) != len(all_new):
        raise ValueError("duplicate D binding id")
    for name, frame in (("crown_beacons", 4320), ("gap_hammer_lands", 3440), ("gap", 3440), ("last_pages", 8640), ("title", 8880)):
        if original_hooks[name]["f"] != frame:
            raise ValueError("mandated D sound frame changed: " + name)
    constraints = deepcopy(phase1["constraints"])
    constraints.update(enforce_after_wet_and_master=True, final_zero_frame=9200,
                       new_timing_source="per-row provenance", no_new_picture_measurement=not bool(overlay["hooks"] or overlay["events"] or overlay["series"] or overlay["bounds"]),
                       gap_contact_frame=next((r["hit_f"] for r in events if r["request_id"] == "D.new.gap.hammer"), None),
                       crown_contact_frames=[r["hit_f"] for r in events if r["request_id"] in ("D.new.crowns.left", "D.new.crowns.right")])
    return dict(schema="long-dawn/d-effects-binding/1", phase=6 if transition_pass else 5 if picture_revision else 4 if race_revision else 3, cut="D", fps=T.FPS, sr=T.SR, frames=T.FRAMES,
                **(dict(bridge_design=bridge_design) if transition_pass else {}),
                edit_selection=selection, race_design=race_design, locked_picture_design=locked_design, reuse_plan=reuse, new_events=events, new_beds=beds, controls=controls,
                constraints=constraints, bound_request_ids=sorted(used), suppressed_requests=suppressed,
                replaced_series=replaced_series, editorial_measurements=scoped,
                measurement_overlay={k: deepcopy(v) for k, v in overlay.items() if k != "document"},
                input_sha256=input_hashes, input_document_sha256={FILES[k]: _digest(v) for k, v in documents.items()},
                input_hash_method="sha256 of raw metadata file bytes; canonical JSON hashes of supplied documents also recorded",
                status="Per-row estimated, scoped derived, or native delivered-plate timing; finished D picture and rendered arrivals remain unverified")


def problems(plan, *, edl=None, barmap=None, cues=None, event_receipts=None, round4_receipts=None,
             measurements=None, frame_root=T.ROOT, evidence_roots=None, race_revision=None, picture_revision=None,
             transition_pass=False):
    """Reject missing requests, altered sync/provenance and weakened constraints."""
    try:
        expected = build(edl=edl, barmap=barmap, cues=cues, event_receipts=event_receipts,
                         round4_receipts=round4_receipts, measurements=measurements, frame_root=frame_root,
                         evidence_roots=evidence_roots, race_revision=race_revision, picture_revision=picture_revision,
                         transition_pass=transition_pass)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        return ["cannot bind current D metadata: " + str(exc)]
    # Palette source tuples become JSON arrays on export; that serialization
    # change is not a timing/provenance change.
    return [] if _digest(plan) == _digest(expected) else ["changed D binding: timing, source inventory, provenance or constraints"]
