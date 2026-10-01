"""Opt-in D crossing foley made from the existing cached wood recording.

These are designed footfalls and lantern-support creaks. The source is heavy
logs moved onto other wood, not a recording of shoes or a lantern. Selection
used the cached source metadata and waveform/spectrogram inspection; it has
not been auditioned. No audio is read or written when this module is imported.
"""
from copy import deepcopy
import json
import math
from pathlib import Path


_LEVEL_FILE = Path(__file__).resolve().parents[1] / "sound/events_B.json"
with _LEVEL_FILE.open() as _fh:
    _LEVELS = {row["id"]: row["level"] for row in json.load(_fh) if "level" in row}

SOURCE = {
    "ref": "fs:734628",
    "cache_path": "music/cache/sound/fs/734628_orig.wav",
    "metadata_path": "music/cache/sound/json/8f9ea125b1daf874ff2c.json",
    "title": "Wood logs - woodshed",
    "author": "Vrymaa",
    "source_url": "https://freesound.org/s/734628/",
    "license": "http://creativecommons.org/publicdomain/zero/1.0/",
    "recorded_action": "Heavy logs thrown onto other logs and wood elements, including a pallet",
    "header": {"samplerate": 44100, "channels": 1, "frames": 698377, "subtype": "PCM_16"},
}

RECIPES = {
    "crossing_footfall": dict(src=(SOURCE["ref"], 2.136), pre=0.016, post=0.22,
        hp=90, lp=2200, fi=0.005, fo=0.09, width=0.3, crest=8.0, send=0.0,
        level=_LEVELS["B.feed.stone_5"]),
    "crossing_creak": dict(src=(SOURCE["ref"], 8.85), pre=0.0, post=0.75,
        hp=180, lp=2800, fi=0.07, fo=0.16, width=0.35, crest=10.0, send=0.0,
        level=_LEVELS["B.lid"]),
}

SELECTION = {
    "crossing_footfall": dict(kind="event", source=deepcopy(SOURCE),
        donor_recipe="sound_recipes_B.feed(3).layers[0]",
        source_anchor_basis="Existing B KNOCKS[3] loudest wood contact at 2.136 s; source position retained",
        selected_source_range_s=(2.12, 2.356),
        role="Designed soft footfall from recorded wood contact",
        design_note="Shorten and low-pass the existing contact; its body supplies the tread while the crest "
                    "cap restrains the hard log click. No fire layer from B's feed is retained.",
        level_from="music/sound/events_B.json:B.feed.stone_5",
        level_basis="Copied donor event target; applying it to isolated footfall foley is an unauditioned design",
        auditioned=False, picture_timing="caller-owned; no measured D contact supplied"),
    "crossing_creak": dict(kind="event", source=deepcopy(SOURCE),
        donor_recipe=None,
        source_anchor_basis="Authored crop boundary at 8.85 s within continuous recorded wood movement; "
                            "not asserted to be a natural acoustic onset",
        selected_source_range_s=(8.85, 9.60),
        role="Designed lantern-support creak from recorded wood movement",
        design_note="Use the sustained movement texture between the stronger impacts, with a soft entry "
                    "and exit and a restrained high-frequency band. The creak role is a design inference "
                    "from the material and source's squeaky tag, not an auditory verification.",
        level_from="music/sound/events_B.json:B.lid",
        level_basis="Copied quiet prop-event target; this does not measure the new wood creak",
        auditioned=False, picture_timing="caller-owned; no measured D lantern movement supplied"),
}


def selection(name):
    """Return a detached provenance record and its proposed sound_v3 recipe."""
    item = deepcopy(SELECTION[name])
    item["recipe"] = deepcopy(RECIPES[name])
    return item


def build(name, *, pan=None, dist=None):
    """Return a caller-owned recipe; no frame, cadence or picture pan is inferred."""
    recipe = deepcopy(RECIPES[name])
    for key, value, lo, hi in (("pan", pan, -1.0, 1.0), ("dist", dist, 0.0, 1.0)):
        if value is not None:
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or not lo <= value <= hi):
                raise ValueError(f"{key} must be finite and within [{lo}, {hi}]")
            recipe[key] = value
    return recipe
