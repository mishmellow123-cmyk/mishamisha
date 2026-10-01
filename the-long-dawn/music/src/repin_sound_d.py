"""Reviewable re-pin of D sound evidence after an editorial-only update.

Run from the film repository through the owner's onepy wrapper, for example::

    "$O/tools/onepy" "$PY" -B music/src/repin_sound_d.py \
      --base "$O/codex/soundd/phase3/effects_measurements.json" \
      --base-edl-ref 3858589 --overlay "$RACE_OVERLAY" --overlay "$DEEP_OVERLAY" \
      --race-revision round5 --out "$O/codex/soundd/phase4/repin/effects_measurements.json"

Set SOUNDD_DEEP_EVIDENCE to the owned Deep evidence directory and
CUTD_LOCAL_RENDERS to the adopted local asset directory. These absolute locations
are runtime configuration, never serialized evidence paths. For the next EDL-only
change, use the preceding output as --base; its verified EDL snapshot replaces
--base-edl-ref. Supply further --overlay arguments only for explicitly replaced
observations. Every retained frame and cited source is revalidated.
For the final adopted picture, add --picture-revision locked and the reviewed
book/transition overlays. Their measured end bounds are merged explicitly.

Non-EDL metadata changes require --review with schema
long-dawn/d-effects-repin-review/1, retained_evidence_review (nonempty rationale),
and files mapping each changed path to {from_sha256,to_sha256,reason}. This is an
explicit review of retained evidence, not permission to skip frame verification.
No EDL, bar map, cue, source image, baseline overlay or sound source is written.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

import sound_d_measurements as M

FIELDS = ("hooks", "events", "series", "suppressions", "bounds")
REVIEW_SCHEMA = "long-dawn/d-effects-repin-review/1"
TRANSITION_REVIEW_SCHEMA = "long-dawn/score-D-reviewed-transitions/1"
REVIEWED_CUTS = (4080, 4240, 4560, 5840, 6080, 6640)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _document(value):
    return M.read(value) if isinstance(value, (str, Path)) else deepcopy(value)


def _pins(document):
    pins = document.get("input_sha256")
    M._require(isinstance(pins, dict) and set(pins) == set(M.INPUT_FILES)
               and all(M._hash(v) for v in pins.values()), "overlay requires all five valid raw metadata hashes")
    M._require((document.get("schema"), document.get("cut"), document.get("fps"), document.get("frames"))
               == (M.SCHEMA, "D", 24, 9200), "wrong source overlay schema/clock")
    return pins


def _base_edl(base, supplied):
    text = supplied if supplied is not None else base.get("repin", {}).get("edl_source_text")
    M._require(isinstance(text, str), "base overlay needs --base-edl-ref or its own verified EDL snapshot")
    M._require(hashlib.sha256(text.encode()).hexdigest() == base["input_sha256"][M.INPUT_FILES[0]],
               "base EDL snapshot does not match overlay pin")
    return json.loads(text)


def _metadata_reviews(sources, current, review):
    required = {(name, old, current[name]) for doc in sources for name, old in _pins(doc).items()
                if name != M.INPUT_FILES[0] and old != current[name]}
    if not required:
        M._require(review is None or not review.get("files"), "metadata review names no current change")
        return []
    M._require(isinstance(review, dict) and review.get("schema") == REVIEW_SCHEMA
               and bool(review.get("retained_evidence_review")),
               "non-EDL metadata changed; explicit retained-evidence review required")
    files = review.get("files")
    M._require(isinstance(files, dict) and set(files) == {r[0] for r in required}, "review must cover exactly the changed non-EDL files")
    for name, old, new in required:
        row = files[name]
        M._require(isinstance(row, dict) and (row.get("from_sha256"), row.get("to_sha256")) == (old, new)
                   and bool(row.get("reason")), "metadata review hash/rationale mismatch: " + name)
    return sorted(required)


def _changes(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        return [p for key in sorted(before.keys() | after.keys())
                for p in _changes(before.get(key), after.get(key), path + "/" + key)]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [p for i, (a, b) in enumerate(zip(before, after)) for p in _changes(a, b, path + "/" + str(i))]
    return [path] if before != after else []


def _transition_changes(before, after):
    """Match the owner's approval schema, retaining every operational field."""
    def at(doc, cut):
        return [{k: deepcopy(v) for k, v in row.items() if k != "note"}
                for row in doc["transitions"] if row.get("cut") == cut]
    cuts = sorted({r.get("cut") for doc in (before, after) for r in doc["transitions"]
                   if type(r.get("cut")) is int})
    return [dict(cut=cut, before=at(before, cut), after=at(after, cut))
            for cut in cuts if at(before, cut) != at(after, cut)]


def _editorial_compatibility(before, after, transition_review=None):
    M._require(all(before.get(k) == after.get(k) == v for k, v in (("cut", "D"), ("fps", 24), ("frames", 9200))),
               "re-pin cannot change the D clock")
    shape = lambda doc: [(s["code"], s["f0"], s["f1"]) for s in doc["shots"]]
    M._require(shape(before) == shape(after), "shot boundaries changed; explicit sound rebind required")
    # Owner page-turn edits may change how the same raw leaf evidence is
    # presented. They cannot invalidate a claim about that source plate alone.
    changed = [r for doc, other in ((before, after), (after, before))
               for r in doc["transitions"] if r not in other["transitions"]]
    reviewed = None
    if transition_review is not None:
        reviewed = _document(transition_review)
        actual = _transition_changes(before, after)
        M._require(isinstance(reviewed, dict) and set(reviewed) == {"schema", "changes"}
                   and reviewed.get("schema") == TRANSITION_REVIEW_SCHEMA,
                   "wrong reviewed-transition schema")
        M._require([r.get("cut") for r in actual] == list(REVIEWED_CUTS)
                   and reviewed.get("changes") == actual,
                   "transition review must match exactly the six adopted join changes")
        M._require(all(type(r.get("cut")) is int and r["cut"] in REVIEWED_CUTS
                       for r in changed), "transition review contains unrelated construction change")
    else:
        M._require(all(8320 <= r["f0"] < r["f1"] <= 8880 for r in changed),
                   "transition outside D31/D32 changed; obtain reviewed construction evidence")
    return dict(changed_paths=_changes(before, after),
                reviewed_transitions=reviewed,
                transition_policy="D31/D32 raw-scope changes auto-repin; six Round8 joins require the exact owner approval. Evidence and selected construction still validate.")


def _measurements(value, scope=M.NATIVE_SCOPE):
    """Yield every actual receipt, including null landmarks and series searches."""
    if isinstance(value, dict):
        scope = value.get("measurement_scope", scope)
        if isinstance(value.get("frames"), list) and "frame_size" in value and "coverage" in value:
            yield value, scope
        else:
            for child in value.values():
                yield from _measurements(child, scope)
    elif isinstance(value, list):
        for child in value:
            yield from _measurements(child, scope)


def _d32_raw_matte(receipt, scope, shot, image, edl, root):
    """Preserve D32's raw-source claim with its exact inert matte construction.

    This exception does not certify the finished composite. D32 has no under
    layer; the cited assembler therefore leaves the RGB plate unchanged by its
    matte. All other image treatments remain forbidden below. Requiring every
    observed source-clock mask makes a future construction change fail closed.
    """
    take = shot["takes"][0]
    M._require(scope == M.NATIVE_SCOPE
               and (shot["code"], shot["f0"], shot["f1"]) == ("D32", 8640, 8880)
               and (take.get("stem"), take.get("off"), take.get("matte"))
               == ("cand_pen_soft_spine_metal", -3200, "cand_pen_soft_spine_metal_matte"),
               "raw matte allowance is restricted to native D32 source evidence")
    M._edl_dependencies(receipt, edl)
    M._require(shot in receipt["edl_dependencies"]["shots"],
               "D32 raw matte evidence needs its exact selected shot dependency")
    matte = "renders/" + take["matte"] + "/" + PurePosixPath(image["path"]).name
    for name in ("edit/assemble.py", matte):
        matching = [d for d in receipt.get("dependencies", [])
                    if d.get("root", "repo") == "repo" and d.get("path") == name]
        M._require(len(matching) == 1, "D32 raw matte evidence needs exact dependency: " + name)
        path = Path(root) / name
        expected = matching[0].get("sha256")
        M._require(M._hash(expected) and path.is_file() and M.sha256(path) == expected,
                   "D32 raw matte dependency SHA256 mismatch: " + name)


def adopted_evidence(document, edl, *, root=M.ROOT):
    """Raw plate coordinates must still select that exact source take."""
    checked = set()
    for receipt, scope in _measurements({k: document.get(k, {}) for k in FIELDS}):
        for image in receipt["frames"]:
            if image.get("root", "repo") != "repo":
                # Composite construction dependencies are checked by M.
                continue
            p = PurePosixPath(image["path"])
            M._require(len(p.parts) == 3 and p.parts[0] == "renders", "invalid raw plate path")
            frame, source = image["frame"], image.get("source_frame", image["frame"])
            shots = [s for s in edl["shots"] if s["f0"] <= frame < s["f1"]]
            M._require(len(shots) == 1, "evidence has no unique selected shot")
            shot = shots[0]
            takes = shot.get("takes", [])
            M._require(len(takes) == 1, "raw plate evidence needs one adopted take: " + shot["code"])
            take = takes[0]
            M._require(take.get("stem") == p.parts[1] and take.get("mode") == "exact"
                       and take.get("hold") is None and take.get("clamp") is None
                       and take.get("off", 0) + frame == source and take.get("final_eligible") is True,
                       "raw plate evidence disagrees with adopted take/clock: " + shot["code"])
            M._require(all(take.get(k) is None for k in ("crop", "grade", "screen_transform", "under", "video", "add", "linear_mix")),
                       "raw evidence take has changed image treatment: " + shot["code"])
            if take.get("matte") is not None:
                _d32_raw_matte(receipt, scope, shot, image, edl, root)
            need = take.get("need")
            M._require(need is None or need[0] <= source <= need[1], "evidence outside adopted take source range")
            checked.add((shot["code"], take["stem"], take.get("off", 0)))
    return [dict(shot=s, stem=t, offset_f=o) for s, t, o in sorted(checked)]


def repin(base, replacements=(), *, root=M.ROOT, base_edl_text=None, review=None,
          evidence_roots=None, request_ids=None, reuse_ids=None, transition_review=None,
          transition_pass=False):
    """Merge explicit revisions, check evidence/adoption, and return new JSON."""
    root = Path(root)
    base = _document(base)
    sources = [base] + [_document(p) for p in replacements]
    current = M.raw_input_hashes(root)
    _pins(base)
    old_edl = _base_edl(base, base_edl_text)
    edl_text = (root / M.INPUT_FILES[0]).read_bytes().decode("utf-8")
    edl = json.loads(edl_text)
    compatibility = _editorial_compatibility(old_edl, edl, transition_review)
    review = _document(review) if review is not None else None
    reviewed = _metadata_reviews(sources, current, review)
    merged = deepcopy(base)
    overwritten, supplied = [], set()
    for i, revision in enumerate(sources[1:], 1):
        for field in FIELDS:
            M._require(isinstance(revision.get(field, {}), dict), "replacement fields must be keyed objects")
            target = merged.setdefault(field, {})
            for name, row in revision.get(field, {}).items():
                key = (field, name)
                M._require(key not in supplied, "multiple replacements for " + field + ":" + name)
                supplied.add(key)
                overwritten.append(dict(field=field, id=name, previous_sha256=digest(target[name]) if name in target else None,
                                        replacement_sha256=digest(row), overlay_index=i))
                target[name] = deepcopy(row)
    merged["input_sha256"] = current
    if request_ids is None or reuse_ids is None:
        import sound_d_table as T
        table = T.build()
        request_ids = {r["id"] for r in table["requests"]} if request_ids is None else request_ids
        reuse_ids = {r["id"] for r in table["reuse"] if r["kind"] == "event"} if reuse_ids is None else reuse_ids
    M._require(type(transition_pass) is bool, "transition pass must be explicitly boolean")
    if transition_pass:
        from sound_d_bridges import TRANSITION_REQUEST
        request_ids = set(request_ids) | {TRANSITION_REQUEST["id"]}
    bm = M.read(root / M.INPUT_FILES[1])
    M.validate_overlay(merged, hooks={h["id"]: h for h in bm["sync"]}, request_ids=request_ids,
                       reuse_ids=reuse_ids, frame_root=root, input_hashes=current,
                       evidence_roots=evidence_roots, edl=edl)
    adoption = adopted_evidence(merged, edl, root=root)
    record = dict(schema="long-dawn/d-effects-repin/1", base_overlay_sha256=digest(base),
                  replacement_overlay_sha256=[digest(p) for p in sources[1:]],
                  replacement_annotations=[dict(overlay_index=i, annotations={k:deepcopy(v) for k, v in p.items()
                      if k not in (*FIELDS, "schema", "cut", "fps", "frames", "input_sha256", "repin")})
                      for i, p in enumerate(sources[1:], 1)],
                  previous_input_sha256=base["input_sha256"], current_input_sha256=current,
                  replacements=overwritten, reviewed_non_edl_changes=reviewed, metadata_review=review,
                  editorial_compatibility=compatibility, adopted_raw_evidence=adoption,
                  edl_source_text=edl_text,
                  scope="Evidence scope unchanged; re-pin verifies byte identity and adopted source clocks, not finished-film synchronization.")
    record["previous_repin_sha256"] = digest(base["repin"]) if "repin" in base else None
    merged["repin"] = record
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-edl-ref", help="Git revision whose exact EDL bytes match the base overlay pin")
    parser.add_argument("--overlay", action="append", type=Path, default=[])
    parser.add_argument("--review", type=Path, help="Explicit non-EDL metadata compatibility review JSON")
    parser.add_argument("--transition-review", type=Path, help="Owner's exact six-join Round8 transition approval JSON")
    parser.add_argument("--race-revision", choices=("round5",))
    parser.add_argument("--picture-revision", choices=("locked",))
    parser.add_argument("--transition-pass", action="store_true")
    parser.add_argument("--evidence-root", action="append", default=[], metavar="NAMESPACE=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binding-out", type=Path)
    args = parser.parse_args(argv)
    sources = [args.base, *args.overlay] + ([args.review] if args.review else []) + ([args.transition_review] if args.transition_review else [])
    outputs = [args.out, args.out.with_suffix(".receipt.json")] + ([args.binding_out] if args.binding_out else [])
    forbidden = {p.resolve() for p in sources + [M.ROOT / p for p in M.INPUT_FILES]}
    if (len({p.resolve() for p in outputs}) != len(outputs)
            or any(p.resolve() in forbidden or p.resolve().is_relative_to((M.ROOT / "renders").resolve()) for p in outputs)):
        parser.error("outputs must be distinct and preserve every input, metadata file and shared render")
    try:
        roots = {}
        for value in args.evidence_root:
            name, path = value.split("=", 1)
            M._require(name not in roots and bool(path), "duplicate or empty evidence root")
            roots[name] = Path(path)
        base_text = None
        if args.base_edl_ref:
            M._require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/~-]*", args.base_edl_ref) is not None, "invalid Git revision")
            base_text = subprocess.check_output(["git", "show", args.base_edl_ref + ":./" + M.INPUT_FILES[0]], cwd=M.ROOT).decode()
        result = repin(args.base, args.overlay, base_edl_text=base_text,
                       review=args.review, evidence_roots=roots, transition_review=args.transition_review,
                       transition_pass=args.transition_pass)
        import sound_d_binding as B
        binding = B.build(measurements=result, evidence_roots=roots, race_revision=args.race_revision,
                          picture_revision=args.picture_revision, transition_pass=args.transition_pass)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        receipt = dict(ok=True, output=args.out.name, output_sha256=M.sha256(args.out),
                       canonical_overlay_sha256=digest(result), binding_sha256=digest(binding),
                       race_revision=args.race_revision, picture_revision=args.picture_revision,
                       transition_pass=args.transition_pass,
                       replaced_rows=result["repin"]["replacements"],
                       adopted_raw_evidence=result["repin"]["adopted_raw_evidence"],
                       input_sha256=result["input_sha256"],
                       scope=result["repin"]["scope"])
        args.out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        if args.binding_out:
            args.binding_out.parent.mkdir(parents=True, exist_ok=True)
            args.binding_out.write_text(json.dumps(binding, indent=2, allow_nan=False) + "\n")
        print(json.dumps(receipt))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, "re-pin refused: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
