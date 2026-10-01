"""Opt-in D effects palette; no picture frame is assigned and no audio is opened.

The source selections come from C/C5's existing recipes. Filter, envelope and
layer changes are unauditioned proposals. A copied level is a donor target, never
a measurement of D. The D table owns picture timing, silence and spatial evidence.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import sound_c5_recipes as C5
import sound_recipes_C as C


_SOUND = Path(__file__).resolve().parents[1] / "sound"
with (_SOUND / "events_C.json").open() as _fh:
    _C_LEVELS = {row["id"]: row for row in json.load(_fh) if "level" in row}
with (_SOUND / "c5_sound_events.json").open() as _fh:
    _C5_LEVELS = {row["id"]: row for row in json.load(_fh)["events"]}


def _clean(recipe):
    """Remove donor placement and matching instructions; retain its source cuts."""
    recipe = deepcopy(recipe)
    for key in ("pan", "dist", "trim", "level_from", "skip", "why"):
        recipe.pop(key, None)
    if "layers" in recipe:
        recipe["layers"] = [_clean(layer) for layer in recipe["layers"]]
    return recipe


def _proposal(kind, donor, recipe, level_id, note, *, c5=False, **metadata):
    row = (_C5_LEVELS if c5 else _C_LEVELS)[level_id]
    if row["kind"] != kind:
        raise ValueError(f"{donor}: {kind} requires a {kind} level donor")
    level = row["level"] + row.get("trim_db", 0.0)
    recipe = _clean(recipe)
    recipe["level"] = level
    table = "c5_sound_events.json" if c5 else "events_C.json"
    return dict(kind=kind, donor_recipe=donor, recipe=recipe, level=level,
                level_from=f"music/sound/{table}:{level_id}" + (" + trim_db" if row.get("trim_db") else ""),
                level_basis="inherited design target" if c5 and row.get("design") else "approved donor target",
                design=True, auditioned=False, note=note, **metadata)


def _unresolved(kind, note, *, action=None):
    return dict(kind=kind, donor_recipe=None, recipe=None, level=None, level_from=None,
                level_basis=None, design=True, auditioned=False, note=note, action=action)


# The steam layer already selects [0.3, 2.6] in C.snow_hiss. Its usable
# continuous portion is looped as a bed; 0.3 is not asserted to be an onset.
_steam = C.RECIPES["C.snow_hiss"]["layers"][1]
_steam_ref, _steam_start = _steam["src"]
_hiss = dict(src=[(_steam_ref, _steam_start, _steam_start + _steam["post"], 1.0)],
             seg=(0.6, 1.0), xf=0.2, hp=800, lp=6500, width=0.5)
_bright = dict(C.RECIPES["C5.hammer"], hp=350, lp=9000)
_dull = dict(C.RECIPES["C5.hammer"], lp=1800, post=0.65, fo=0.20)
_glass = deepcopy(C.RECIPES["C+.coldtick"]["layers"][0])
for _key in ("g", "dt", "stretch"):
    _glass.pop(_key, None)
_old_paper = dict(C.RECIPES["C+.burn.remains"], g=0.0)
_old_flare = dict(C.RECIPES["C.beacon1"]["layers"][0], g=-14.0, lp=2200)


DESIGN = {
    "ring_burn": _proposal("event", "C+.burn.map", C.RECIPES["C+.burn.map"], "C+.burn.map",
        "Paper burn from the drawn gilt ring; bind its chosen picture landmark after measuring the burn."),
    "inscription_hiss": _proposal("bed", "C.snow_hiss.layers[1]", _hiss, "C.storm.eye",
        "Loop the existing steam cut under the moving front. The storm-bed level is an audition target for this "
        "new bed, not a steam measurement; stop it on the measured unfinished letter.",
        level_relation="same-kind donor; different material"),
    "inscription_ring": _proposal("event", "C5.hammer", dict(_bright, post=0.65, fo=0.25),
        "C5.hammer.trap_1", "A short high-passed anvil ring from the writing front; the donor onset is retained.",
        c5=True),
    "inscription_stop": _unresolved("control",
        "A control landmark: truncate the inscription hiss and ring on the measured stop, with a short de-click "
        "ending at that boundary. Do not add a new impact.", action="cut_existing_layers"),
    "giant_ring": _proposal("event", "C5.hammer", _bright, "C5.hammer.trap_1",
        "Bright anvil ring for a giant's stroke reaching the band; one instance per measured contact.", c5=True),
    "dull_hammer": _proposal("event", "C5.hammer", _dull, "C5.hammer.trap_1",
        "The same anvil attack, shortened and low-passed, for a stroke which does not reach the band.", c5=True),
    "glass_tick": _proposal("event", "C+.coldtick.layers[0]", _glass, "C+.coldtick",
        "One metal tick proposed as the glass-like gap detail; the source is metal, not a glass recording. "
        "The donor sequence's target level requires audition for this single tick."),
    "vision_pressure": _proposal("bed", "C.storm.eye", dict(C.RECIPES["C.storm.eye"], lp=1500, width=0.65),
        "C.storm.eye", "Filtered storm pressure rises over the measured vision interval. Scale the envelope "
        "fractions to that interval; hard silence is enforced separately by the D table.",
        envelope_fraction=((0.0, -18.0), (0.8, 0.0), (1.0, 0.0))),
    "gap_hammer": _proposal("event", "C5.hammer", dict(_bright, send=0.0, no_breath=True),
        "C5.hammer.trap_1", "One dry bright stroke at the hard cut after the vision silence.", c5=True),
    "crown_kindle": _proposal("event", "C.beacon1",
        dict(C.RECIPES["C.beacon1"], send=0.0,
             layers=[dict(C.RECIPES["C.beacon1"]["layers"][0], pre=0.0, fi=0.005)]),
        "C.beacon1", "Two caller-owned copies start together, hard left and hard right. "
        "No pre-roll before kindling and no outdoor send crossing into the opposite channel."),
    "ridge_catch": _proposal("event", "C.beacon2",
        dict(C.RECIPES["C.beacon2"], layers=[dict(C.RECIPES["C.beacon2"]["layers"][0], pre=0.0, fi=0.005)]),
        "C.beacon2", "One catch per measured fire; pan, distance and relative gain follow the delivered picture."),
    "holdout_hammer": _proposal("event", "C5.hammer.faint", C.RECIPES["C5.hammer.faint"],
        "C5.hammer.alone_1", "The soft anvil's lone stroke, with its inherited C5 design target.", c5=True),
    "instep_hammer": _proposal("event", "C5.hammer.instep", C5.INSTEP_RECIPES["C5.hammer.instep"],
        "C5.hammer.instep_1", "The existing simultaneous cached-anvil pair, with no layer delays or wet send; "
        "D's delivered convergence determines each strike.", c5=True),
    "forge_steady": _proposal("bed", "C5.forge.steady", C5.INSTEP_RECIPES["C5.forge.steady"],
        "C5.forge.instep", "The working forge continues under the open Ring; the C5 target includes its saved trim.",
        c5=True),
    "old_fire": _proposal("event", "C+.burn.remains + C.beacon1.layers[0]",
        dict(layers=[_old_paper, _old_flare], crest=12.0, send=0.0), "C+.burn.remains",
        "Paper with a low wood-fire flare. The -14 dB flare layer is an unauditioned design choice; no lava or "
        "volcanic roar source is used."),
    "lamp": _proposal("bed", "C.hearth.open", dict(C.RECIPES["C.hearth.open"], lp=4800, width=0.35),
        "C.hearth.open", "Small warm crackle from the existing close hearth; a bed, with no invented ignition hit."),
    "crossing_creak": _unresolved("event",
        "No lantern or rope creak donor found in the inspected A/B/C recipes and SFX_CREDITS inventory. "
        "Select and measure a recording before placing a creak; C+.cock is cock_hahn.wav, not a creak."),
    "crossing_footfall": _unresolved("event",
        "No physical footfall donor found in the inspected A/B/C recipes and SFX_CREDITS inventory. "
        "AP2's WALK is score material; it does not establish a recorded footfall source or source onset."),
    "page_quill": _proposal("event", "C+.pen.deep", C.RECIPES["C+.pen.deep"], "C+.pen.deep",
        "The existing quill cut for the old-fire hand caption or final written leaf. Bind each contact from "
        "picture and shorten the take to its measured writing duration; retain the donor source position."),
    "leaf_settle": _proposal("event", "C.page.blank_2",
        dict(C.RECIPES["C.page.blank_2"], pre=0.0, fo=0.5,
             env=[(0.0, 0.0), (0.2, -6.0), (0.8, -30.0)]), "C.page.blank_2",
        "A curtailed single leaf gesture from the existing page take; bind after observing the last leaf settle."),
    "ridge_wind": _proposal("bed", "C.wind.fall", C.RECIPES["C.wind.fall"], "C.wind.fall",
        "Continuous mountain wind across the ridge sequence; measured gain changes belong to the D table."),
    "hearth_low": _proposal("bed", "C.hearth.end", dict(C.RECIPES["C.hearth.end"], lp=4000, width=0.5),
        "C.hearth.end", "The existing ending hearth with a narrower, darker proposed treatment. "
        "This donor LUFS target does not prove compliance with the voice window's dBFS ceiling."),
    "dawn_wind": _proposal("bed", "C.air.dawn", C.RECIPES["C.air.dawn"], "C.air.dawn",
        "The existing quiet wind range, with the inherited trim already represented in the target; no birds."),
}


def design(name):
    """Return a caller-owned copy of all provenance and recipe metadata."""
    return deepcopy(DESIGN[name])


def build(name, *, pan=None, dist=None):
    """Return sound_v3 parameters only. The caller must bind measured D timing.

    Unknown sources and control landmarks cannot masquerade as renderable audio.
    Spatial values are supplied by the caller; donor picture positions are removed.
    """
    item = design(name)
    if item["recipe"] is None:
        raise ValueError(f"{name}: no renderable recipe; {item['note']}")
    recipe = item["recipe"]
    for key, value, lo, hi in (("pan", pan, -1.0, 1.0), ("dist", dist, 0.0, 1.0)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name}: {key} must be finite and within [{lo}, {hi}]")
            recipe[key] = value
    return recipe
