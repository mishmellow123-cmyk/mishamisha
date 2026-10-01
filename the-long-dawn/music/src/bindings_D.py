"""Opt-in D score bindings; the editorial JSON remains read-only.

The single editable table is music/v3/score_D_bindings.json. Its canonical
rows retain cutd's original evidence; aliases point to those rows instead of
duplicating their frame values. A new D render measurement replaces one row's
frame/status/source and adds its measured frame receipt. No pixel onset is
inferred merely because a render directory exists.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from barmap_D_score import DraftMap, EVENT_TABLE
from timeline_v3 import BAR_F, BEAT_F, FPS

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "music/v3/score_D_bindings.json"
SCHEMA = "long-dawn/score-D-bindings/1"
EDITORIAL_FILES = ("music/v3/barmap_D.json", "music/v3/cues_D.json",
                   "music/v3/events_D_measured.json", "music/v3/events_D_round4_measured.json")
MEASURED_ORIGINS = {"inherited_measured", "inherited_cover_measurement", "measured_D_cover",
                    "measured_source_cover", "measured_D_composite"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate binding key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _inherited_evidence(event):
    reference = event.get("measured_ref") or "music/v3/barmap_D.json#sync." + event["id"]
    scope = event.get("measurement_scope")
    if not scope:
        scope = ("Inherited C generated-cover measurement; not remeasured on D."
                 if event["timing_status"] == "inherited_cover_measurement" else
                 "Inherited donor measurement with the recorded source-to-D offset; not remeasured on D.")
    return event["source"], reference, scope


def _editorial_receipts(canonical, receipts):
    """Check the cited leaf and what its samples actually establish.

    Source-mask clearance and the D composite's tail clearance are different
    quantities. A nonzero source remnant cannot become a measured clear frame
    merely because the D-only envelope later removes it.
    """
    for name, event in canonical.items():
        receipt = receipts.get(event.get("measurement_file"))
        if receipt is None:
            continue
        reference = event.get("measured_ref", "")
        value = receipt
        try:
            for key in reference.split("."):
                value = value[key]
        except (KeyError, TypeError):
            raise ValueError(f"{name}: missing editorial receipt reference {reference}") from None
        if value is None or isinstance(value, bool) or value != event["f"]:
            raise ValueError(f"{name}: editorial receipt disagrees with the bar map")

    def first(rows, predicate):
        return next((row["frame"] for row in rows if predicate(row)), None)

    old = receipts[EDITORIAL_FILES[2]]["burn"]
    ridge = receipts[EDITORIAL_FILES[3]]["ridge_burn"]
    source = receipts[EDITORIAL_FILES[3]]["deep_exit_source"]
    tail = receipts[EDITORIAL_FILES[3]]["deep_exit_tail"]
    summaries = (
        ("drawn-ring opening", old["first_open_on_sample_grid"],
         first(old["frames"], lambda r: r["hole_fraction"] > 0)),
        ("drawn-ring clearance", old["first_fully_gone_on_sample_grid"],
         first(old["frames"], lambda r: r["page_remaining_max"] == 0)),
        ("ridge opening", ridge["first_open_on_sample_grid"],
         first(ridge["frames"], lambda r: r["open_pixels"] > 0)),
        ("ridge clearance", ridge["first_fully_gone_on_sample_grid"],
         first(ridge["frames"], lambda r: r["remaining_cover_pixels"] == 0)),
        ("Deep source opening", source["first_open_on_selected_source_masks"],
         first(source["frames"], lambda r: r["cover_mean"] < 1)),
        ("Deep source clearance", source["first_fully_gone_on_selected_source_masks"],
         first(source["frames"], lambda r: r["cover_max"] == 0)),
        ("Deep tail clearance before finish", tail["clear_frame_before_finish"],
         first(tail["frames"], lambda r: r.get("effective_cover_max") == 0
               and r.get("after_vs_incoming_max") == 0)),
    )
    for name, claimed, sampled in summaries:
        if claimed != sampled:
            raise ValueError(f"{name}: editorial receipt summary disagrees with its samples")
    opening = next(r for r in source["frames"]
                   if r["frame"] == source["first_open_on_selected_source_masks"])
    if opening["source_frame"] != source["first_open_source_frame"]:
        raise ValueError("Deep source opening: source frame disagrees with its sample")
    if source["visible_glow_onset"] is not None:
        raise ValueError("Deep source masks cannot establish a visible glow onset")


def _frame_receipt(name, row, render_root):
    receipt = row.get("measurement")
    if not isinstance(receipt, dict) or receipt.get("kind") != "render_D":
        raise ValueError(f"{name}: a new measured frame needs a render_D receipt")
    stem = receipt.get("stem", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", stem):
        raise ValueError(f"{name}: invalid render stem")
    if row.get("render_stem") and stem != row["render_stem"]:
        raise ValueError(f"{name}: measurement belongs to a different shot")
    if receipt.get("frame") != row["frame"]:
        raise ValueError(f"{name}: receipt frame differs from binding")
    filename = receipt.get("frame_file", "")
    match = re.fullmatch(r"[^/\\]*?(\d+)\.(png|jpg|jpeg|exr)", filename, re.I)
    if not match or int(match[1]) != row["frame"]:
        raise ValueError(f"{name}: frame filename must carry the D frame number")
    if not receipt.get("method") or not row.get("measured_ref") or not row.get("measurement_scope"):
        raise ValueError(f"{name}: measured frame needs method, reference and scope")
    path = (Path(render_root) / stem / filename).resolve()
    shot = (Path(render_root) / stem).resolve()
    if path.parent != shot or not path.is_file():
        raise ValueError(f"{name}: measured frame asset is missing")
    if sha256(path) != receipt.get("sha256"):
        raise ValueError(f"{name}: measured frame asset hash mismatch")


def _omission_receipt(name, row, original, canonical, render_root):
    """Keep pulse evidence separate from the reason a glass accent is omitted."""
    action = row.get("score_action")
    if action is None:
        if any(field in row for field in ("absence_evidence", "omission_reason", "omission_kind")):
            raise ValueError(f"{name}: omission evidence requires score_action omit")
        return
    if action != "omit":
        raise ValueError(f"{name}: unknown score_action {action!r}")
    old = original.get(name)
    if (name in canonical or "anchor" in row or not old or old.get("fixed")
            or "giant_stroke" not in old.get("tags", [])):
        raise ValueError(f"{name}: only direct nonfixed original giant_stroke rows may omit")
    if row.get("frame") != old["frame"]:
        raise ValueError(f"{name}: an omitted accent must retain its original frame")
    kind = row.get("omission_kind", "no_strike")
    if kind == "no_strike":
        if row.get("status") != "estimated" or row.get("measurement") or row.get("picture_frame") is not None:
            raise ValueError(f"{name}: absence cannot be promoted to a measured onset")
    elif kind == "no_gap_narrowing":
        if not 2960 <= old["frame"] < 3200:
            raise ValueError(f"{name}: no_gap_narrowing is restricted to original Brink strokes")
        receipt = row.get("measurement")
        if row.get("status") != "measured" or not isinstance(receipt, dict) or receipt.get("kind") != "render_D":
            raise ValueError(f"{name}: no_gap_narrowing must retain its native measured pulse")
    else:
        raise ValueError(f"{name}: unknown omission_kind {kind!r}")
    reason = row.get("omission_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{name}: omission needs a rationale")
    absence = row.get("absence_evidence")
    if not isinstance(absence, dict) or not absence.get("source"):
        raise ValueError(f"{name}: omission needs absence evidence")
    if kind == "no_gap_narrowing":
        geometry = absence.get("geometry_source")
        if not isinstance(geometry, dict) or geometry.get("path") != "shots/embers/d_vision.py":
            raise ValueError(f"{name}: no_gap_narrowing needs the pinned Brink geometry source")
        if geometry.get("sha256") != sha256(ROOT / geometry["path"]):
            raise ValueError(f"{name}: Brink geometry source hash mismatch")
    for field in ("source", "measured_ref", "measurement_scope"):
        if not isinstance(absence.get(field), str) or not absence[field].strip():
            raise ValueError(f"{name}: absence needs source, reference and scope")
    method = absence.get("measurement", {}).get("method") if isinstance(absence.get("measurement"), dict) else None
    if not isinstance(method, str) or not method.strip():
        raise ValueError(f"{name}: absence needs an observation method")
    evidence = deepcopy(absence)
    evidence["frame"] = row["frame"]
    if row.get("render_stem"):
        evidence["render_stem"] = row["render_stem"]
    _frame_receipt(name + " absence", evidence, render_root)


class BoundMap(DraftMap):
    """Resolve one D table, with explicit measurement scope and aliases."""

    def __init__(self, table=TABLE, *, render_root=None):
        super().__init__()
        path = Path(table).resolve() if isinstance(table, (str, Path)) else None
        document = _read(path) if path else deepcopy(table)
        if not isinstance(document, dict) or document.get("schema") != SCHEMA:
            raise ValueError("unrecognised D binding table schema")
        if tuple(document.get(k) for k in ("cut", "bars", "frames", "fps")) != ("D", 115, 9200, 24):
            raise ValueError("D binding grid must remain 115 bars / 9200 frames / 24 fps")
        actual_hashes = {p: sha256(ROOT / p) for p in EDITORIAL_FILES}
        if document.get("editorial_sha256") != actual_hashes:
            raise ValueError("binding table belongs to a different editorial source revision")
        bar = _read(ROOT / EDITORIAL_FILES[0])
        cues = _read(ROOT / EDITORIAL_FILES[1])
        receipts = {p: _read(ROOT / p) for p in EDITORIAL_FILES[2:]}
        if (bar["bars"], bar["frames"], bar["fps"]) != (115, 9200, 24):
            raise ValueError("editorial grid differs from score")
        canonical = {e["id"]: e for e in bar["sync"]}
        rows = document.get("events", {})
        original = {e["id"]: e for e in EVENT_TABLE}
        missing = (canonical.keys() | original.keys()) - rows.keys()
        if missing:
            raise ValueError(f"binding table is missing events: {sorted(missing)}")
        for name, event in canonical.items():
            cue = cues["events"][name]
            if "anchor" in rows[name]:
                raise ValueError(f"{name}: canonical events cannot be aliases")
            if any(cue[k] != event[k] for k in cue.keys() & event.keys()):
                raise ValueError(f"{name}: editorial map/cue disagreement")
            if rows[name].get("editorial") != event:
                raise ValueError(f"{name}: original editorial evidence must be preserved")
        _editorial_receipts(canonical, receipts)

        resolved, visiting = {}, set()
        render_root = render_root or ROOT / "renders"

        def resolve(name):
            if name in resolved:
                return resolved[name]
            if name in visiting or name not in rows:
                raise ValueError(f"binding alias cycle or missing anchor: {name}")
            visiting.add(name)
            row = rows[name]
            _omission_receipt(name, row, original, canonical, render_root)
            for related in row.get("related_editorial_events", []):
                if related not in canonical:
                    raise ValueError(f"{name}: unknown related editorial event {related}")
            if "anchor" in row:
                if any(k in row for k in ("frame", "status", "measurement")):
                    raise ValueError(f"{name}: edit the anchor, not a duplicate alias frame")
                anchor = resolve(row["anchor"])
                if anchor.get("score_action") == "omit":
                    raise ValueError(f"{name}: omission cannot be inherited through an alias")
                event = deepcopy(anchor)
                event.update(id=name, anchor=row["anchor"], binding_note=row.get("note", ""))
                # Aliases have no independent musical tags; only the original
                # score names below may drive score generators.
                event["tags"] = []
            else:
                frame = row.get("frame")
                if type(frame) is not int or not 0 <= frame <= self.frames:
                    raise ValueError(f"{name}: invalid integer D frame")
                if row.get("status") not in ("measured", "estimated") or not row.get("source"):
                    raise ValueError(f"{name}: status and source are required")
                if row.get("picture_frame") is not None:
                    raise ValueError(f"{name}: picture_frame is derived from a validated render receipt")
                base = canonical.get(name, original.get(name))
                baseline = base.get("f", base.get("frame")) if base else row.get("baseline_frame")
                if type(baseline) is not int:
                    raise ValueError(f"{name}: baseline frame is required")
                fixed = (base or {}).get("timing_status") in ("planned_edit_boundary", "renderer_parameter")
                fixed = fixed or ((base or {}).get("fixed", False) and name not in canonical)
                if (fixed or baseline < 1440) and frame != baseline:
                    raise ValueError(f"{name}: fixed edit boundary or protected opening")
                # An estimate remains in its original bar. Delivered native
                # evidence may cross that bar inside the editorial window;
                # the render receipt is checked below before this can load.
                window = (base or {}).get("timing_window")
                if base is None:
                    window = row.get("timing_window")
                    section = self.sec_at(baseline / BEAT_F)
                    if not section["f0"] <= frame < section["f1"]:
                        raise ValueError(f"{name}: new score marker cannot leave its section")
                native = (row["status"] == "measured" and isinstance(row.get("measurement"), dict)
                          and row["measurement"].get("kind") == "render_D")
                if frame // BAR_F != baseline // BAR_F and not (native and window):
                    raise ValueError(f"{name}: replacement must stay in its canonical bar without a measured window")
                if window and not window[0] <= frame < window[1]:
                    raise ValueError(f"{name}: frame outside editorial timing window")
                if row["status"] == "measured":
                    inherited = (name in canonical and frame == canonical[name]["f"]
                                 and canonical[name]["timing_status"] in MEASURED_ORIGINS
                                 and not row.get("measurement"))
                    if inherited:
                        claimed = tuple(row.get(k) for k in ("source", "measured_ref", "measurement_scope"))
                        if claimed != _inherited_evidence(canonical[name]):
                            raise ValueError(f"{name}: inherited measurement reference and scope must retain the source evidence")
                    else:
                        _frame_receipt(name, row, render_root)
                elif row.get("measurement"):
                    raise ValueError(f"{name}: a render measurement must be labelled measured")
                event = deepcopy(row)
                event.update(id=name, fixed=fixed, kind=cues["events"].get(name, {}).get("kind", "music"), tags=[])
                if row.get("measurement"):
                    event["picture_frame"] = frame
            if name in original:
                old = original[name]
                if old["fixed"] and event["frame"] != old["frame"]:
                    raise ValueError(f"{name}: score edit boundary cannot move through an alias")
                section = self.sec_at(old["frame"] / BEAT_F)
                if old["frame"] < self.frames and not section["f0"] <= event["frame"] < section["f1"]:
                    raise ValueError(f"{name}: score marker cannot leave its section")
                event["tags"] = deepcopy(original[name]["tags"])
            event.update(beat=event["frame"] / BEAT_F, t=event["frame"] / FPS,
                         bar=event["frame"] // BAR_F + 1,
                         bar_beat=1 + event["frame"] % BAR_F / BEAT_F)
            event["section"] = self.sec_at(event["beat"])["id"]
            visiting.remove(name)
            resolved[name] = event
            return event

        self.events = sorted((resolve(name) for name in rows), key=lambda e: (e["frame"], e["id"]))
        if path:
            label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
            digest = sha256(path)
        else:
            label = "in-memory binding table"
            digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
        self.binding_provenance = dict(schema=SCHEMA, table_path=label, table_sha256=digest,
                                       editorial_sha256=actual_hashes,
                                       note="Measured status retains per-event scope; inherited timings are not D remeasurements.")
        self.d.update(status="D bound", events=self.events, binding_provenance=self.binding_provenance)


def load_map(table=None):
    """Shared renderer/verifier entry; omitting the flag preserves Phase 1c."""
    if table is None:
        return DraftMap()
    document = _read(table) if isinstance(table, (str, Path)) else table
    if isinstance(document, dict) and document.get("schema") == SCHEMA:
        return BoundMap(table)
    return DraftMap(document)


def movements(bound, previous=None):
    """Named score-marker changes; these are not audible-onset measurements."""
    previous = previous or DraftMap()
    return [dict(event=e["id"], old_frame=e["frame"], frame=bound.event(e["id"])["frame"],
                 delta_frames=bound.event(e["id"])["frame"] - e["frame"],
                 status=bound.event(e["id"])["status"])
            for e in previous.events if bound.event(e["id"])["frame"] != e["frame"]]
