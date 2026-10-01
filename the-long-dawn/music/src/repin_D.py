"""Explicit D editorial re-pin; never changes a musical event or measurement.

The frozen music JSONs must still match their original pins. The current D EDL
must reproduce from its generator. Once bootstrapped, source clocks and shot
bounds cannot change through re-pinning. The unbound D31-to-D32 transition may
change; six further owner-reviewed joins require exact before/after records.
Native frame receipts remain the responsibility of BoundMap.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from bindings_D import (BoundMap, EDITORIAL_FILES, ROOT, TABLE, _editorial_receipts,
                        _frame_receipt, _read, sha256)

EDIT_FILES = ("edit/edl_v3.py", "edit/edl/edl_D.json")
REVISION_SCHEMA = "long-dawn/score-D-edit-revision/1"
UNBOUND_TRANSITION_WINDOW = (8320, 8880)
CLOCK_SOURCE = "shots/embers/c_d.py"
REVIEW_SCHEMA = "long-dawn/score-D-reviewed-transitions/1"
REVIEWABLE_CUTS = frozenset((4080, 4240, 4560, 5840, 6080, 6640))


def scoped_picture_receipts(document, *, render_root=None, source_root=None):
    """Validate a clock's inspected frame without claiming an optical onset."""
    render_root = render_root or ROOT / "renders"
    source_root = source_root or ROOT
    for name, row in document["events"].items():
        has_evidence = "frame_evidence" in row or "clock_source" in row
        if row.get("role") != "source_clock":
            if has_evidence:
                raise ValueError(f"{name}: scoped frame evidence requires role source_clock")
            continue
        if (row.get("status") != "estimated" or row.get("measurement")
                or row.get("picture_frame") is not None or "anchor" in row):
            raise ValueError(f"{name}: source_clock must remain a direct estimate without an onset receipt")
        evidence = row.get("frame_evidence")
        if not isinstance(evidence, dict) or evidence.get("frame") != row.get("frame"):
            raise ValueError(f"{name}: source_clock needs matching frame evidence")
        if type(evidence.get("frame")) is not int:
            raise ValueError(f"{name}: scoped frame must be an integer")
        for key in ("source", "measured_ref", "measurement_scope"):
            if not isinstance(evidence.get(key), str) or not evidence[key].strip():
                raise ValueError(f"{name}: scoped frame needs source, reference and scope")
        inspected = deepcopy(evidence)
        if row.get("render_stem"):
            inspected["render_stem"] = row["render_stem"]
        _frame_receipt(name + " scoped frame", inspected, render_root)
        clock = row.get("clock_source")
        if not isinstance(clock, dict) or clock.get("path") != CLOCK_SOURCE:
            raise ValueError(f"{name}: source_clock needs the explicit race clock source")
        path = Path(source_root) / CLOCK_SOURCE
        if not path.is_file() or sha256(path) != clock.get("sha256"):
            raise ValueError(f"{name}: race clock source is missing or its hash changed")


def _signature(edl):
    """The source clocks and timing that a re-pin may not silently alter."""
    shots = []
    for shot in edl["shots"]:
        takes = []
        for take in shot["takes"]:
            # Human descriptions do not affect the source selected by EDIT.
            takes.append({k: deepcopy(v) for k, v in take.items() if k != "note"})
        shots.append({**{k: shot[k] for k in ("sec", "f0", "f1", "code", "kind")},
                      "takes": takes})
    lo, hi = UNBOUND_TRANSITION_WINDOW
    page_turn = [t for t in edl["transitions"] if t.get("cut") == 8640]
    shot_by_code = {s["code"]: s for s in edl["shots"]}
    if (len(page_turn) != 1 or "D31" not in shot_by_code or "D32" not in shot_by_code
            or shot_by_code["D31"]["f1"] != 8640 or shot_by_code["D32"]["f0"] != 8640
            or not lo <= page_turn[0]["f0"] < 8640 < page_turn[0]["f1"] <= hi):
        raise ValueError("D31-to-D32 transition must be one bounded transition spanning cut 8640")
    transitions = [{k: deepcopy(v) for k, v in transition.items() if k != "note"}
                   for transition in edl["transitions"]
                   if transition.get("cut") != 8640]
    return dict(shots=shots, protected_transitions=transitions)


def _review_transitions(previous, signature, review):
    """An approval must describe exactly the old and new protected records.

    These are signature records: every transition field except the human note.
    Empty lists describe a hard cut. Approval is consumed against the previous
    revision, so reusing it after adoption fails rather than widening its scope.
    """
    if (not isinstance(review, dict) or set(review) != {"schema", "changes"}
            or review.get("schema") != REVIEW_SCHEMA
            or not isinstance(review.get("changes"), list) or not review["changes"]):
        raise ValueError("invalid reviewed-transitions approval")
    old = previous["protected_transitions"]
    new = signature["protected_transitions"]
    approved = set()
    for change in review["changes"]:
        if not isinstance(change, dict) or set(change) != {"cut", "before", "after"}:
            raise ValueError("reviewed transition requires cut, before and after records")
        cut = change["cut"]
        if type(cut) is not int or cut not in REVIEWABLE_CUTS or cut in approved:
            raise ValueError("unlisted or duplicate reviewed transition cut")
        before, after = change["before"], change["after"]
        if (not isinstance(before, list) or not isinstance(after, list)
                or len(before) > 1 or len(after) > 1 or before == after):
            raise ValueError("reviewed transition must change zero or one record per side")
        if before != [t for t in old if t.get("cut") == cut]:
            raise ValueError(f"D{cut}: reviewed before records do not match prior pins")
        if after != [t for t in new if t.get("cut") == cut]:
            raise ValueError(f"D{cut}: reviewed after records do not match current EDL")
        outgoing = next((s for s in signature["shots"] if s["f1"] == cut), None)
        incoming = next((s for s in signature["shots"] if s["f0"] == cut), None)
        if outgoing is None or incoming is None:
            raise ValueError("reviewed transition must use a fixed shot boundary")
        for record in after:
            if (type(record.get("f0")) is not int or type(record.get("f1")) is not int
                    or not outgoing["f0"] <= record["f0"] <= cut < record["f1"] <= incoming["f1"]):
                raise ValueError("reviewed transition must remain inside its two fixed shots")
        approved.add(cut)
    if ([t for t in old if t.get("cut") not in approved]
            != [t for t in new if t.get("cut") not in approved]):
        raise ValueError("unreviewed protected transition changed")
    encoded = json.dumps(review, sort_keys=True, separators=(",", ":")).encode()
    return dict(schema=REVIEW_SCHEMA, approval_sha256=hashlib.sha256(encoded).hexdigest(),
                changes=deepcopy(review["changes"]),
                scope="Exact protected transition records reviewed; no picture or audio onset is inferred.")


def build_revision(document, *, edl, generated, bar, cues, receipts,
                   music_hashes, edit_hashes, reviewed_transitions=None):
    """Validate before returning a new pin; inputs and event rows are immutable."""
    if document.get("editorial_sha256") != music_hashes:
        raise ValueError("music editorial JSON changed: reconcile its evidence; re-pin cannot accept it")
    if edl != generated:
        raise ValueError("D EDL export differs from its generator; regenerate it before re-pinning")
    if (edl.get("cut"), edl.get("fps"), edl.get("frames"), edl.get("bars")) != ("D", 24, 9200, 115):
        raise ValueError("D EDL grid changed")
    if (bar.get("cut"), bar.get("fps"), bar.get("frames"), bar.get("bars")) != ("D", 24, 9200, 115):
        raise ValueError("D bar-map grid changed")
    canonical = {row["id"]: row for row in bar["sync"]}
    if len(canonical) != len(bar["sync"]) or set(cues["events"]) != set(canonical):
        raise ValueError("D canonical event population changed or disagrees with cues")
    for name, event in canonical.items():
        if document.get("events", {}).get(name, {}).get("editorial") != event:
            raise ValueError(f"{name}: canonical evidence changed; reconcile it instead of re-pinning")
        cue = cues["events"][name]
        if any(cue[k] != event[k] for k in cue.keys() & event.keys()):
            raise ValueError(f"{name}: editorial map/cue disagreement")
    _editorial_receipts(canonical, receipts)
    cursor = 0
    for shot in edl["shots"]:
        if (type(shot["f0"]) is not int or type(shot["f1"]) is not int
                or shot["f0"] != cursor or shot["f1"] <= cursor):
            raise ValueError("D EDL shots do not tile the fixed grid")
        cursor = shot["f1"]
    if cursor != 9200:
        raise ValueError("D EDL shots do not end on the fixed grid")
    for section in bar["sections"]:
        shots = [s for s in edl["shots"] if s["sec"] == section["id"]]
        if not shots or (shots[0]["f0"], shots[-1]["f1"]) != (section["f0"], section["f1"]):
            raise ValueError("D EDL section boundaries disagree with the music bar map")
    # The original 34 treatment entries remain boundaries even when a shot has
    # subsequently been split (the holdout is D21a/b/c).
    starts = {s["f0"] for s in edl["shots"]}
    if any(e["f"] not in starts for e in canonical.values() if e.get("picture_name")):
        raise ValueError("D EDL moved a treatment picture entry")
    signature = _signature(edl)
    previous = document.get("edit_revision")
    review_receipt = None
    if previous is not None:
        if previous.get("schema") != REVISION_SCHEMA:
            raise ValueError("unrecognised D edit-revision schema")
        old_signature = previous.get("protected_signature", {})
        if reviewed_transitions is not None:
            if old_signature.get("shots") != signature["shots"]:
                raise ValueError("D source mapping changed; transition approval cannot change sources")
            review_receipt = _review_transitions(old_signature, signature, reviewed_transitions)
        elif old_signature != signature:
            raise ValueError("D source mapping or protected transition changed; explicit binding review required")
        else:
            review_receipt = deepcopy(previous.get("transition_review"))
    elif reviewed_transitions is not None:
        raise ValueError("transition approval requires existing editorial pins")
    if set(edit_hashes) != set(EDIT_FILES):
        raise ValueError("D edit revision must pin the generator and its export")
    revision = dict(schema=REVISION_SCHEMA, sha256=deepcopy(edit_hashes),
                protected_signature=signature,
                allowed_transition_window=list(UNBOUND_TRANSITION_WINDOW),
                scope="Source/timing preflight only. D31-to-D32 transition is unbound; no caption write-on or native onset is measured by re-pinning.")
    if review_receipt is not None:
        revision["transition_review"] = review_receipt
    return revision


def _member_spans(text):
    """Locate top-level JSON values without rewriting any event bytes."""
    decoder = json.JSONDecoder()
    pos = text.index("{") + 1
    spans = {}
    while True:
        while text[pos].isspace() or text[pos] == ",":
            pos += 1
        if text[pos] == "}":
            return spans, pos
        key, pos = decoder.raw_decode(text, pos)
        while text[pos].isspace():
            pos += 1
        if text[pos] != ":":
            raise ValueError("invalid binding JSON member")
        pos += 1
        while text[pos].isspace():
            pos += 1
        start = pos
        _, pos = decoder.raw_decode(text, pos)
        spans[key] = (start, pos)


def replace_revision(text, revision):
    spans, close = _member_spans(text)
    encoded = json.dumps(revision, indent=2, ensure_ascii=False).replace("\n", "\n  ")
    if "edit_revision" in spans:
        start, end = spans["edit_revision"]
        return text[:start] + encoded + text[end:]
    return text[:close].rstrip() + ',\n  "edit_revision": ' + encoded + "\n" + text[close:]


def current_revision(document, *, reviewed_transitions=None):
    # edl_doc is the production export path, not a second implementation of it.
    sys.path.insert(0, str(ROOT / "edit"))
    import assemble
    hashes = {p: sha256(ROOT / p) for p in EDITORIAL_FILES}
    return build_revision(document, edl=_read(ROOT / EDIT_FILES[1]),
                          generated=assemble.edl_doc("D"),
                          bar=_read(ROOT / EDITORIAL_FILES[0]),
                          cues=_read(ROOT / EDITORIAL_FILES[1]),
                          receipts={p: _read(ROOT / p) for p in EDITORIAL_FILES[2:]},
                          music_hashes=hashes,
                          edit_hashes={p: sha256(ROOT / p) for p in EDIT_FILES},
                          reviewed_transitions=reviewed_transitions)


def repin(table=TABLE, *, write=False, reviewed_transitions=None):
    """Return a privacy-safe receipt; --write changes only edit_revision."""
    table = Path(table)
    text = table.read_text()
    document = _read(table)
    # A stale/missing native image must be remeasured, not blessed by this tool.
    BoundMap(document)
    scoped_picture_receipts(document)
    if reviewed_transitions is not None and not write:
        raise ValueError("reviewed-transitions approval requires --write; subsequent --check needs no approval")
    if reviewed_transitions is None:
        revision = current_revision(document)
    else:
        review = _read(reviewed_transitions) if isinstance(reviewed_transitions, (str, Path)) else reviewed_transitions
        revision = current_revision(document, reviewed_transitions=review)
    previous = document.get("edit_revision")
    if not write and previous != revision:
        raise ValueError("D editorial pins are absent or stale; run repin_D.py --write after review")
    changed = previous != revision
    if write and changed:
        updated = replace_revision(text, revision)
        if json.loads(updated)["events"] != document["events"]:
            raise ValueError("re-pin attempted to change an event")
        before_spans, _ = _member_spans(text)
        after_spans, _ = _member_spans(updated)
        if text[slice(*before_spans["events"])] != updated[slice(*after_spans["events"])]:
            raise ValueError("re-pin attempted to rewrite event bytes")
        # Refuse to overwrite a simultaneous lane edit.
        if table.read_text() != text:
            raise ValueError("binding table changed during preflight; retry against the current table")
        temp = table.with_name(table.name + ".repin.tmp")
        try:
            temp.write_text(updated)
            temp.replace(table)
        finally:
            if temp.exists():
                temp.unlink()
    receipt = dict(ok=True, mode="write" if write else "check", changed=changed,
                prior_sha256=(previous or {}).get("sha256"), sha256=revision["sha256"],
                table_sha256=sha256(table), events_byte_identical=True,
                music_editorial_sha256=document["editorial_sha256"],
                scope=revision["scope"])
    if "transition_review" in revision:
        receipt["transition_review"] = revision["transition_review"]
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, default=TABLE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="validate and replace only the edit_revision pins")
    mode.add_argument("--check", action="store_true", help="validate existing pins without modifying files")
    parser.add_argument("--receipt", type=Path, help="optional private JSON audit receipt")
    parser.add_argument("--reviewed-transitions", type=Path,
                        help="private exact before/after approval for the six owner-reviewed joins")
    args = parser.parse_args(argv)
    result = repin(args.bindings, write=args.write, reviewed_transitions=args.reviewed_transitions)
    payload = json.dumps(result, indent=2) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
