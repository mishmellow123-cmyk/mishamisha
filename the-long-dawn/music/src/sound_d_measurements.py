"""Read-only evidence gate for D sound bindings.

The two editorial receipts describe cover grids, source masks and an unfinished
composite. Native incoming plates are a separate scope, never finished D film.
One optional overlay carries hooks, one-off events, complete event series and
explicit suppressions. No renderer parameter can satisfy a pixel receipt.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "long-dawn/d-effects-measurements/1"
INPUT_FILES = ("edit/edl/edl_D.json", "music/v3/barmap_D.json", "music/v3/cues_D.json",
               "music/v3/events_D_measured.json", "music/v3/events_D_round4_measured.json")
NATIVE_SCOPE = "native_delivered_plate"
COMPOSITE_SCOPE = "native_pre_finish_composite"


def evidence_roots_config(frame_root=ROOT, supplied=None):
    """Resolve portable evidence namespaces; serialized paths stay relative."""
    roots = {"repo": Path(frame_root)}
    for name, variable in (("cutd_local_renders", "CUTD_LOCAL_RENDERS"),
                           ("soundd_deep_composites", "SOUNDD_DEEP_EVIDENCE")):
        if os.environ.get(variable):
            roots[name] = Path(os.environ[variable])
    for name, path in (supplied or {}).items():
        _require(name in ("cutd_local_renders", "soundd_deep_composites"), "unknown evidence root: " + name)
        roots[name] = Path(path)
    return roots


def _relative_path(name):
    _require(isinstance(name, str) and bool(name) and "\\" not in name, "invalid evidence path")
    p = PurePosixPath(name)
    _require(not p.is_absolute() and ".." not in p.parts and str(p) == name,
             "evidence path must be normalized and relative")
    return p


def _edl_dependencies(receipt, edl):
    """Check the selected construction, allowing unrelated editorial changes."""
    dependencies = receipt.get("edl_dependencies")
    _require(isinstance(dependencies, dict) and dependencies.get("shots"),
             "composite evidence needs selected EDL shot dependencies")
    _require(isinstance(dependencies.get("transitions"), list), "composite evidence needs transition dependencies")
    for field, key in (("shots", "code"), ("transitions", None)):
        rows = dependencies[field]
        _require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows), "invalid EDL dependencies")
        ids = [r.get(key) if key else json.dumps(r, sort_keys=True) for r in rows]
        _require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), "ambiguous EDL dependencies")
        for row in rows:
            selected = [r for r in edl[field] if r.get(key) == row[key]] if key else [r for r in edl[field] if r == row]
            _require(selected == [row], "stale composite EDL dependency: " + str(row.get(key, field)))
    coverage = receipt.get("coverage", {})
    first, last = coverage.get("first"), coverage.get("last")
    _require(_frame(first) and _frame(last) and first <= last, "invalid composite coverage")
    for field in ("shots", "transitions"):
        selected = [r for r in edl[field] if r["f0"] <= last and r["f1"] > first]
        _require(all(r in dependencies[field] for r in selected), "composite omits overlapping EDL " + field)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_input_hashes(root=ROOT):
    return {name: sha256(Path(root) / name) for name in INPUT_FILES}


def read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate measurement key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _frame(value):
    return type(value) is int and 0 <= value < 9200


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def editorial_receipts(edl, barmap, cues, original, round4):
    """Validate recorded samples and the exact scope of each cited hook."""
    docs = {INPUT_FILES[3]: original, INPUT_FILES[4]: round4}
    _require(original.get("schema") == "long-dawn/d-events-measured/1", "wrong original D receipt schema")
    _require(round4.get("schema") == "long-dawn/d-round4-events-measured/1", "wrong Round 4 receipt schema")
    for doc in docs.values():
        _require((doc.get("cut"), doc.get("fps")) == ("D", 24), "wrong D receipt clock")
    burn, ridge = original["burn"], round4["ridge_burn"]
    source, tail = round4["deep_exit_source"], round4["deep_exit_tail"]
    for label, obj, first, stop, shape in (("drawn ring", burn, 1680, 1760, [480, 201]),
                                          ("ridge", ridge, 5180, 5220, [480, 201]),
                                          ("Deep source", source, 2945, 2981, [1920, 804])):
        _require(obj["sample_grid"] == shape, label + ": changed measurement grid")
        _require([r["frame"] for r in obj["frames"]] == list(range(first, stop)),
                 label + ": incomplete measurement denominator")
    _require([r["frame"] for r in tail["frames"]] == list(range(2976, 2982)), "incomplete Deep tail samples")
    _require(tail["native_frame"] == [1920, 804], "Deep tail is not the recorded native composite")
    first = lambda rows, predicate: next((r["frame"] for r in rows if predicate(r)), None)
    comparisons = (
        (burn["first_open_on_sample_grid"], first(burn["frames"], lambda r: r["hole_fraction"] > 0)),
        (burn["first_fully_gone_on_sample_grid"], first(burn["frames"], lambda r: r["page_remaining_max"] == 0)),
        (ridge["first_open_on_sample_grid"], first(ridge["frames"], lambda r: r["open_pixels"] > 0)),
        (ridge["first_fully_gone_on_sample_grid"], first(ridge["frames"], lambda r: r["remaining_cover_pixels"] == 0)),
        (source["first_open_on_selected_source_masks"], first(source["frames"], lambda r: r["cover_mean"] < 1)),
        (source["first_fully_gone_on_selected_source_masks"], first(source["frames"], lambda r: r["cover_max"] == 0)),
        (tail["clear_frame_before_finish"], first(tail["frames"], lambda r: r.get("effective_cover_max") == 0
                                                and r.get("after_vs_incoming_max") == 0)),
    )
    _require(all(a == b for a, b in comparisons), "D receipt summary disagrees with actual samples")
    _require(source["visible_glow_onset"] is None, "source masks cannot establish visible glow onset")
    for r in source["frames"]:
        _require(r["source_frame"] == round(1905 + (r["frame"] - 2945) * 86 / 35), "changed Deep source clock")
        _require(_hash(r["rgb_sha256"]) and _hash(r["cover_sha256"]), "Deep source hashes missing")
    opening = next(r for r in source["frames"] if r["frame"] == source["first_open_on_selected_source_masks"])
    _require(opening["source_frame"] == source["first_open_source_frame"], "Deep source onset clock disagrees")
    transitions = edl["transitions"]
    for obj, cut, parameters in ((original, 1680, original["parameters"]), (ridge, 5180, ridge["parameters"])):
        selected = [t for t in transitions if t.get("cut") == cut and t["kind"] == "ring_burn"]
        _require(len(selected) == 1 and all(selected[0].get(k) == v for k, v in parameters.items()),
                 "burn receipt no longer matches selected D transition")
    deep = [t for t in transitions if t["kind"] == "deep_reveal"]
    _require(len(deep) == 1 and deep[0]["tail_clear"] == tail["tail_clear"]
             and (deep[0]["f0"], deep[0]["f1"]) == (2945, 2981), "Deep tail receipt no longer matches EDL")
    checked = {}
    allowed = {"drawn_ring_burn_opens": "measured_D_cover", "drawn_ring_burn_page_gone": "measured_D_cover",
               "last_kingdom_burn": "measured_D_cover", "last_kingdom_burn_ridge_gone": "measured_D_cover",
               "brink_burnthrough": "measured_source_cover", "brink_burn_clear": "measured_D_composite"}
    for hook in barmap["sync"]:
        if hook["id"] not in allowed:
            continue
        _require(hook.get("timing_status") == allowed[hook["id"]] and hook.get("picture_frame") is None,
                 hook["id"] + ": source measurement scope was promoted")
        _require(bool(hook.get("measurement_scope")), hook["id"] + ": measurement scope missing")
        value = docs[hook["measurement_file"]]
        for key in hook["measured_ref"].split("."):
            value = value[key]
        _require(type(value) is int and value == hook["f"], hook["id"] + ": receipt disagrees with bar map")
        cue = cues["events"][hook["id"]]
        for key in ("source", "timing_status", "measured_ref", "measurement_scope", "measurement_file"):
            _require(cue.get(key) == hook.get(key), hook["id"] + ": cue scope differs from bar map")
        checked[hook["id"]] = deepcopy(hook)
    _require(set(checked) == set(allowed), "missing scoped D receipt hook")
    return checked


def _coverage(obj, frames, *, complete_required=False):
    """A complete denominator means every physical image is cited and hashed."""
    _require(isinstance(obj, dict), "native evidence needs explicit coverage")
    first, last = obj.get("first"), obj.get("last")
    _require(_frame(first) and _frame(last) and first <= last, "invalid inclusive evidence coverage")
    _require(type(obj.get("complete")) is bool, "coverage complete must be boolean")
    numbers = {r["frame"] for r in frames}
    _require(all(first <= f <= last for f in numbers), "evidence frame outside coverage")
    missing = sorted(set(range(first, last + 1)) - numbers)
    _require(obj.get("missing") == missing, "coverage missing list does not match physical evidence denominator")
    _require(not obj["complete"] or not missing, "complete coverage has missing images")
    _require(not complete_required or obj["complete"], "series replacement needs complete physical coverage")


def _evidence(receipt, root, cache, *, evidence_roots=None, edl=None, scope=NATIVE_SCOPE):
    _require(scope in (NATIVE_SCOPE, COMPOSITE_SCOPE), "unknown evidence measurement scope")
    _require(isinstance(receipt, dict) and bool(receipt.get("method")), "native measurement needs a method")
    _require(receipt.get("frame_size") == [1920, 804], "native measurement must use 1920x804 delivered plates")
    roots = evidence_roots_config(root, evidence_roots)
    sources = receipt.get("source_sha256", {})
    _require(isinstance(sources, dict), "source evidence hashes must be a repo-path mapping")
    for name, expected in sources.items():
        p = _relative_path(name)
        _require(not p.is_absolute() and ".." not in p.parts and p.suffix == ".py"
                 and p.parts[0] in ("shots", "edit", "lib", "music"),
                 "source evidence must cite a repo-relative Python file")
        path = root / name
        _require(_hash(expected) and path.is_file() and sha256(path) == expected,
                 "source evidence SHA256 mismatch: " + name)
    dependencies = receipt.get("dependencies", [])
    _require(isinstance(dependencies, list), "evidence dependencies must be a list")
    if scope == COMPOSITE_SCOPE:
        _require(bool(dependencies), "composite evidence needs construction file dependencies")
    if scope == COMPOSITE_SCOPE or "edl_dependencies" in receipt:
        _edl_dependencies(receipt, edl if edl is not None else read(root / INPUT_FILES[0]))
    for dependency in dependencies:
        _require(isinstance(dependency, dict), "invalid evidence dependency")
        namespace, name = dependency.get("root", "repo"), dependency.get("path")
        p = _relative_path(name)
        _require(namespace in roots, "unconfigured evidence root: " + str(namespace))
        if namespace == "repo":
            _require(p.parts[0] in ("shots", "edit", "lib", "music", "renders"), "invalid repo construction dependency")
        path = roots[namespace] / p
        if namespace != "repo":
            _require(path.resolve().is_relative_to(roots[namespace].resolve()), "evidence dependency escapes configured root")
        _require(_hash(dependency.get("sha256")) and path.is_file()
                 and sha256(path) == dependency["sha256"], "construction dependency SHA256 mismatch: " + name)
    frames = receipt.get("frames")
    _require(isinstance(frames, list) and frames, "native measurement needs actual frame evidence")
    keys = set()
    from PIL import Image
    for item in frames:
        _require(isinstance(item, dict) and _frame(item.get("frame")), "invalid evidence D frame")
        name = item.get("path", "")
        p = _relative_path(name)
        namespace = item.get("root", "repo")
        if scope == COMPOSITE_SCOPE:
            _require(namespace == "soundd_deep_composites" and len(p.parts) == 2 and p.parts[0] == "composites",
                     "pre-finish composite evidence must use soundd_deep_composites/composites/<frame>")
        else:
            _require(namespace == "repo" and len(p.parts) == 3 and p.parts[0] == "renders",
                     "frame evidence path must be repo-relative renders/<stem>/<frame>")
            _require(item.get("source_stem", p.parts[1]) == p.parts[1], "evidence source stem disagrees with its path")
        _require(namespace in roots, "unconfigured evidence root: " + namespace)
        match = re.fullmatch(r"f_(\d+)\.(jpg|jpeg|png)", p.name, re.I)
        _require(match is not None and int(match[1]) == item.get("source_frame", item["frame"]),
                 "evidence filename disagrees with source frame")
        _require(_hash(item.get("sha256")), "missing frame SHA256")
        key = (item["frame"], namespace, name)
        _require(key not in keys, "duplicate native frame evidence")
        keys.add(key)
        path = roots[namespace] / name
        if namespace != "repo":
            _require(path.resolve().is_relative_to(roots[namespace].resolve()), "frame evidence escapes configured root")
        _require(path.is_file(), "missing native frame: " + name)
        stat = path.stat()
        stamp = (str(path), stat.st_size, stat.st_mtime_ns)
        if stamp not in cache:
            with Image.open(path) as im:
                size = list(im.size)
            cache[stamp] = (sha256(path), size)
        digest, size = cache[stamp]
        _require(digest == item["sha256"], "native frame SHA256 mismatch: " + name)
        _require(size == [1920, 804], "native frame dimensions disagree: " + name)
    _coverage(receipt.get("coverage"), frames)
    return frames


def validate_overlay(document, *, hooks, request_ids, reuse_ids=(), frame_root=ROOT, input_hashes=None,
                     evidence_roots=None, edl=None):
    """Validate without writes; absent observations never become estimated facts."""
    if document is None:
        return dict(hooks={}, events={}, series={}, suppressions={}, bounds={}, gaps=[])
    doc = read(document) if isinstance(document, (str, Path)) else deepcopy(document)
    _require((doc.get("schema"), doc.get("cut"), doc.get("fps"), doc.get("frames")) == (SCHEMA, "D", 24, 9200),
             "invalid D measurement overlay schema/clock")
    _require(doc.get("input_sha256") == (input_hashes or raw_input_hashes()), "stale measurement overlay metadata hashes")
    root, cache = Path(frame_root), {}
    known = set(request_ids) | set(reuse_ids)
    for field in ("hooks", "events", "series", "suppressions", "bounds"):
        doc.setdefault(field, {})
        _require(isinstance(doc[field], dict), field + " must be keyed objects")
    event_ids, series_ids, suppression_ids = (set(doc[k]) for k in ("events", "series", "suppressions"))
    _require(not (event_ids & series_ids) and not ((event_ids | series_ids) & suppression_ids),
             "request has conflicting replacement/suppression")
    gaps = []

    def landmark(name, row):
        _require(isinstance(row, dict), name + ": landmark must be an object")
        def evidence():
            scope = row.get("measurement_scope")
            _require(scope in (NATIVE_SCOPE, COMPOSITE_SCOPE), name + ": wrong native measurement scope")
            return _evidence(row.get("measurement"), root, cache, evidence_roots=evidence_roots, edl=edl, scope=scope)
        if row.get("frame") is None:
            _require(row.get("status") in ("unmeasured", "estimated") and bool(row.get("reason")),
                     name + ": null landmark needs an unmeasured reason")
            if row.get("measurement"):
                evidence()
            gaps.append(dict(deepcopy(row), id=name))
            return False
        _require(_frame(row["frame"]) and row.get("status") == "measured", name + ": only measured frames override timing")
        _require(bool(row.get("source")) and bool(row.get("measured_ref")), name + ": source/ref missing")
        frames = evidence()
        _require(row["frame"] in {r["frame"] for r in frames}, name + ": measured frame is not in evidence")
        claim = row["measurement"].get("claim")
        _require(claim in ("onset", "contact", "completion", "observed_state", "static_state", "absence"),
                 name + ": explicit measurement claim required")
        if claim in ("onset", "completion"):
            _require(row["frame"] - 1 in {r["frame"] for r in frames}, name + ": onset/completion lacks preceding image")
            _require(bool(row["measurement"].get("predicate") or row["measurement"].get("threshold")),
                     name + ": onset/completion needs an observed predicate")
        if row.get("pan") is not None:
            _require(type(row["pan"]) in (int, float) and -1 <= row["pan"] <= 1 and bool(row.get("spatial_basis")),
                     name + ": measured pan needs a spatial basis")
        return claim in ("onset", "contact", "completion")

    applied_hooks, applied_events, applied_series = {}, {}, {}
    for name, row in doc["hooks"].items():
        _require(name in hooks, "unknown D timing hook: " + name)
        if landmark(name, row):
            original = hooks[name]
            _require(original.get("timing_status") not in ("renderer_parameter", "planned_edit_boundary"),
                     name + ": parameter/edit boundary cannot be promoted to a measured landmark")
            window = original.get("timing_window")
            _require(not window or window[0] <= row["frame"] < window[1], name + ": native landmark outside timing window")
            applied_hooks[name] = row
    for name, row in doc["events"].items():
        _require(name in known, "unknown sound event: " + name)
        if landmark(name, row):
            _require(bool(row.get("landmark")), name + ": replacement needs its own landmark name")
            _require(row.get("editorial_reference") in hooks, name + ": replacement needs its editorial reference hook")
            applied_events[name] = row
    for name, group in doc["series"].items():
        _require(name in request_ids and isinstance(group, dict), "unknown/malformed sound series: " + name)
        rows = group.get("events")
        _require(isinstance(rows, list), name + ": series events must be a list")
        all_frames, ids, valid = [], set(), []
        for row in rows:
            _require(bool(row.get("id")) and row["id"] not in ids, name + ": series IDs must be unique")
            ids.add(row["id"])
            if landmark(name + "." + row["id"], row):
                valid.append(row)
                all_frames.extend(row["measurement"]["frames"])
        # A collector must cite the complete searched window, including frames
        # without events; event count cannot stand in for image coverage.
        search = group.get("measurement")
        if search:
            all_frames.extend(_evidence(search, root, cache, evidence_roots=evidence_roots, edl=edl,
                                        scope=group.get("measurement_scope", NATIVE_SCOPE)))
        _coverage(group.get("coverage"), all_frames)
        if not group["coverage"]["complete"]:
            _require(bool(group.get("reason")), name + ": partial series needs a coverage reason")
            gaps.append(dict(id=name, status="unmeasured", reason=group["reason"], coverage=deepcopy(group["coverage"])))
            continue
        _require(len(valid) == len(rows), name + ": incomplete series landmarks cannot replace estimates")
        _require(all(a["frame"] <= b["frame"] for a, b in zip(valid, valid[1:])), name + ": series frames must be ordered")
        applied_series[name] = group
    for name, decision in doc["suppressions"].items():
        _require(name in known and bool(decision.get("reason")) and bool(decision.get("source")),
                 "suppression needs known request, source and reason")
        _evidence(decision.get("measurement"), root, cache, evidence_roots=evidence_roots, edl=edl,
                  scope=decision.get("measurement_scope", NATIVE_SCOPE))
    applied_bounds = {}
    for name, sides in doc["bounds"].items():
        _require(name in known and name not in suppression_ids and isinstance(sides, dict)
                 and set(sides) == {"end"}, "bound needs active known request and one end landmark")
        point = sides["end"]
        if landmark(name + ".end", point):
            _require(point["measurement"]["claim"] == "completion", "end bound needs a measured completion")
            _require(type(point.get("offset_f", 0)) is int and point.get("offset_f", 0) in (0, 1),
                     "end bound offset must be zero or one frame")
            applied_bounds[name] = sides
    return dict(hooks=applied_hooks, events=applied_events, series=applied_series, bounds=applied_bounds,
                suppressions=doc["suppressions"], gaps=gaps, document=doc,
                observations={name: deepcopy(row) for name, row in doc["hooks"].items() if name not in applied_hooks},
                event_observations={name: deepcopy(row) for name, row in doc["events"].items() if name not in applied_events},
                sha256=hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
