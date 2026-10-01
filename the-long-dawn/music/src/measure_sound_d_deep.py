"""Measure the rebuilt Deep entry before the film finish, with an explicit root.

Run through the shared production slot. The existing burn's source marker is
an authored mid-sweep envelope crest; an observed first pixel change does not
replace it. This command writes only to its explicit external output directory.
It never edits a binding, EDL, source asset, or sound file.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
FIRST, LAST = 2720, 2757
SIZE = (1920, 804)
INPUTS = ("edit/edl/edl_D.json", "music/v3/barmap_D.json", "music/v3/cues_D.json",
          "music/v3/events_D_measured.json", "music/v3/events_D_round4_measured.json")
EVENT = "D.11.C5.burn.deep"
KEEP = {2720, 2721, 2722, 2738, 2739, 2740, 2741, 2742, 2743, 2745, 2746, 2747, 2757}
THRESHOLDS = (1, 4, 8, 16)
# Localized on the actual held R7 D2719/D2720 image, below the bright Ring.
# This ROI is a search region, not a claim derived from renderer parameters.
CORE_ROI = (935, 320, 985, 395)
CORE_FLOORS = (140, 180, 220)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def native_rgb(path):
    with Image.open(path) as image:
        if image.size != SIZE:
            raise ValueError("requires native 1920x804 input")
        return np.asarray(image.convert("RGB")).copy()


def changed_pixels(rgb, baseline):
    """Exact decoded display-pixel differences; warm color is only a proxy.

    The warm statistic excludes existing held-race pixels unless red increases.
    It is not a measurement of physical fire energy or the desired sound crest.
    """
    if (rgb.shape != baseline.shape or rgb.ndim != 3 or rgb.shape[-1] != 3
            or rgb.dtype != np.uint8 or baseline.dtype != np.uint8):
        raise ValueError("equal-shaped uint8 RGB inputs required")
    delta = rgb.astype(np.int16) - baseline.astype(np.int16)
    difference = np.max(np.abs(delta), axis=-1)
    r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
    warm = (r >= 180) & (g >= 50) & (r >= 1.65*g) & (g >= 1.5*b)
    return dict(max_channel_change=int(difference.max()),
                mean_channel_change=float(np.abs(delta).mean()),
                changed_pixels={str(t): int(np.count_nonzero(difference >= t)) for t in THRESHOLDS},
                new_warm_pixels={str(t): int(np.count_nonzero(warm & (delta[..., 0] >= t)))
                                 for t in THRESHOLDS})


def first_change(rows, field, threshold, first=FIRST, last=LAST):
    """A missing member anywhere in the search prevents an onset claim."""
    frames = [r["frame"] for r in rows]
    if len(frames) != len(set(frames)) or any(type(f) is not int or not first <= f <= last for f in frames):
        raise ValueError("duplicate or out-of-range frame")
    missing = sorted(set(range(first, last+1)) - set(frames))
    candidates = sorted(r["frame"] for r in rows if r[field][str(threshold)] > 0)
    observed = candidates[0] if candidates else None
    frame = observed if not missing and observed is not None and observed > first else None
    return dict(frame=frame, first_observed_positive=observed,
                claim="onset" if frame is not None else "observed_state",
                coverage=dict(first=first, last=last, complete=not missing, missing=missing))


def core_stats(rgb, baseline, roi=CORE_ROI, counterfactual=None):
    """White-core survival on masks fixed from the actual held-race pixels.

    A warm page/flame is excluded by the near-neutral color condition. These
    counts describe the displayed core, not physical fire extinction or alpha.
    """
    if rgb.shape != baseline.shape or rgb.dtype != np.uint8 or baseline.dtype != np.uint8:
        raise ValueError("equal uint8 core inputs required")
    x0,y0,x1,y1=roi
    if not (0<=x0<x1<=rgb.shape[1] and 0<=y0<y1<=rgb.shape[0]):
        raise ValueError("core ROI outside native image")
    now=rgb[y0:y1,x0:x1].astype(np.int16)
    old=baseline[y0:y1,x0:x1].astype(np.int16)
    if counterfactual is not None:
        if counterfactual.shape!=rgb.shape or counterfactual.dtype!=np.uint8:
            raise ValueError("counterfactual must match actual RGB image")
        difference=np.max(np.abs(now-counterfactual[y0:y1,x0:x1].astype(np.int16)),axis=-1)
    out={}
    for floor in CORE_FLOORS:
        mask=(old.min(axis=-1)>=floor)&(np.ptp(old,axis=-1)<=60)
        white=(now.min(axis=-1)>=floor)&(np.ptp(now,axis=-1)<=60)
        out[str(floor)]=dict(baseline_pixels=int(mask.sum()),white_pixels=int((mask&white).sum()))
        if counterfactual is not None:
            out[str(floor)]["attributed_white_pixels"]={str(t):int((mask&white&(difference>=t)).sum())
                                                       for t in (1,4,8)}
    return out


def core_clearance(rows, first=FIRST, last=LAST):
    frames=[r["frame"] for r in rows]
    if sorted(frames)!=list(range(first,last+1)):
        raise ValueError("white-core clearance needs complete unique coverage")
    out={}
    for floor in CORE_FLOORS:
        mask_pixels={r["core"][str(floor)]["baseline_pixels"] for r in rows}
        if len(mask_pixels)!=1:
            raise ValueError("white-core baseline mask changed during scan")
        for minimum in (1,3,5):
            positive=[r["frame"] for r in rows if r["core"][str(floor)]["white_pixels"]>=minimum]
            last_positive=max(positive) if positive else None
            out[f"minRGB{floor}_pixels{minimum}"]=dict(baseline_pixels=next(iter(mask_pixels)),
                last_visible_frame=last_positive,
                first_trailing_absent_frame=last_positive+1 if last_positive is not None and last_positive<last else None,
                criterion_pixels=minimum)
            if all("attributed_white_pixels" in r["core"][str(floor)] for r in rows):
                for contribution in (1,4,8):
                    surviving=[r["frame"] for r in rows
                        if r["core"][str(floor)]["attributed_white_pixels"][str(contribution)]>=minimum]
                    last_surviving=max(surviving) if surviving else None
                    out[f"minRGB{floor}_pixels{minimum}_sourceDelta{contribution}"]=dict(
                        baseline_pixels=next(iter(mask_pixels)),last_visible_frame=last_surviving,
                        first_trailing_absent_frame=last_surviving+1 if last_surviving is not None and last_surviving<last else None,
                        criterion_pixels=minimum,source_contribution_floor=contribution)
    return out


def page_resemblance(rgb, baseline, clean):
    """Visible-page proxy, not an alpha measurement: RGB within a clean-page tolerance.

    Pixels already resembling the page on the held race are excluded. Warm
    lighting, char and fire can prevent page pixels satisfying this predicate.
    """
    if rgb.shape != clean.shape or clean.dtype != np.uint8:
        raise ValueError("clean page must match the composite RGB shape")
    distance = np.max(np.abs(rgb.astype(np.int16)-clean.astype(np.int16)), axis=-1)
    original_distance = np.max(np.abs(baseline.astype(np.int16)-clean.astype(np.int16)), axis=-1)
    changed = original_distance > 32
    centre = rgb.shape[1]//2
    return {str(t): dict(pixels=int(np.count_nonzero((distance <= t) & changed)),
                        centre_column_pixels=int(np.count_nonzero(((distance <= t) & changed)[:,centre])))
            for t in (8, 12, 16)}


def page_candidates(rows):
    """List all tied maxima of a declared proxy; do not promote to a sound cue."""
    result = {}
    for threshold in (8, 12, 16):
        counts = [r["page_resemblance"][str(threshold)]["pixels"] for r in rows]
        differences = [(rows[i]["frame"], counts[i]-counts[i-1]) for i in range(1,len(rows))]
        maximum = max(value for _, value in differences)
        result[str(threshold)] = dict(
            maximum_new_page_proxy_pixels_per_frame=maximum,
            maximum_rate_frames=[f for f, value in differences if value == maximum],
            first_centre_column_page_proxy=next((r["frame"] for r in rows
                if r["page_resemblance"][str(threshold)]["centre_column_pixels"] > 0), None))
    return result


def trailing_clearance(rows, threshold, first=FIRST, last=LAST):
    """First trailing zero after the final positive; requires complete coverage."""
    onset = first_change(rows, "new_warm_pixels", threshold, first, last)
    positive = [r["frame"] for r in rows if r["new_warm_pixels"][str(threshold)] > 0]
    last_positive = max(positive) if positive else None
    frame = last_positive+1 if (onset["coverage"]["complete"] and last_positive is not None
                                and last_positive < last) else None
    return dict(frame=frame, last_positive=last_positive, coverage=onset["coverage"],
                claim="warm-proxy trailing clearance within searched interval; not physical flame extinction")


def selected_edl(edl):
    shots = [s for s in edl["shots"] if s["f0"] < LAST+1 and s["f1"] > FIRST]
    background = [s for s in edl["shots"] if s["code"] == "D10"]
    if [s["code"] for s in shots] != ["D11a", "D11b"] or len(background) != 1:
        raise ValueError("Deep entry shot layout changed")
    transitions = [t for t in edl["transitions"] if t["f0"] < LAST+1 and t["f1"] > FIRST]
    if transitions:
        raise ValueError("Deep entry gained an edit transition; revise measurement method")
    return deepcopy(dict(shots=background+shots, transitions=transitions))


def validate_refresh(receipt, local, root=ROOT):
    """Validate native inputs before the assembler can silently resize them."""
    if receipt.get("status") != "complete" or not receipt.get("native_runtime_ran"):
        raise ValueError("refresh runtime is incomplete")
    if [r["frame"] for r in receipt["runtime"]["frames"]] != list(range(FIRST, LAST+1)):
        raise ValueError("refresh receipt lacks the complete Deep denominator")
    farm = "renders/embers_D_race/f_02719.jpg"
    if digest(Path(root)/farm) != receipt["farm_sha256"]:
        raise ValueError("held race no longer matches refreshed input")
    native_rgb(Path(root)/farm)
    wanted = {f"{stem}/f_{f:05d}.{ext}" for f in range(FIRST, LAST+1)
              for stem, ext in (("cutd_deep_clean", "png"), ("cutd_deep_sweep_coeff", "npz"))}
    by_name = {r["file"]: r for r in receipt["reused_assets"]}
    if len(by_name) != len(receipt["reused_assets"]) or not wanted <= set(by_name):
        raise ValueError("missing or duplicate Deep construction assets")
    dependencies = [dict(root="repo", path=farm, sha256=receipt["farm_sha256"])]
    for name in sorted(wanted):
        p = PurePosixPath(name)
        path = Path(local)/p
        if not path.resolve().is_relative_to(Path(local).resolve()):
            raise ValueError("Deep construction asset escapes its root")
        sha = digest(path)
        if sha != by_name[name]["sha256"]:
            raise ValueError("Deep construction hash mismatch: "+name)
        if p.suffix == ".png":
            native_rgb(path)
        else:
            with np.load(path, allow_pickle=False) as data:
                for key in ("gain", "base"):
                    array = data[key]
                    if array.shape != (804, 1920, 3) or not np.isfinite(array).all():
                        raise ValueError("invalid native Deep coefficient: "+name)
        dependencies.append(dict(root="cutd_local_renders", path=name, sha256=sha))
    for name in ("edit/assemble.py", "edit/cutd_deep.py", "music/src/measure_sound_d_deep.py",
                 "music/src/sound_c5_table.py", "music/src/sound_recipes_C.py", "music/sound/picture_sync_C.json"):
        dependencies.append(dict(root="repo", path=name, sha256=digest(Path(root)/name)))
    return dependencies


def overlay(result):
    """Null event anchor retains the measured transition evidence for review."""
    retained = [r for r in result["retained_frames"] if FIRST <= r["frame"] <= FIRST+2]
    document=dict(schema="long-dawn/d-effects-measurements/1", cut="D", fps=24, frames=9200,
                input_sha256=deepcopy(result["input_sha256"]), hooks={}, series={}, suppressions={},
                events={EVENT: dict(frame=None, status="unmeasured",
                    reason="The source marker is an authored mid-sweep envelope crest. First changed composite pixels do not identify that crest; D2739 remains estimated.",
                    source="native rebuilt Deep entry before master finish",
                    measured_ref="deep_measurements.json",
                    measurement_scope="native_pre_finish_composite",
                    measurement=dict(method="Decoded 8-bit RGB comparison against the exact held D2719 race; full-frame search with sensitivity controls.",
                        claim="observed_state", frame_size=list(SIZE), frames=deepcopy(retained),
                        coverage=dict(first=FIRST, last=FIRST+2, complete=True, missing=[]),
                        observed_transition_onset=deepcopy(result["onsets"]),
                        predicate="At least one pixel changes by the stated channel threshold relative to held race; warm proxy additionally requires R>=180, G>=50, R>=1.65G, G>=1.5B and red increase.",
                        dependencies=deepcopy(result["dependencies"]),
                        edl_dependencies=deepcopy(result["edl_dependencies"]),
                        limitations="No claim of master-finished visibility, physical fire energy, source-hit timing or sound-envelope crest. Full38-frame metrics are in the receipt; only the three-frame onset prefix is retained here."))})
    core=result.get("thinking_fire_core")
    if core:
        clearances=core["clearance"]
        values={r["first_trailing_absent_frame"] for r in clearances.values()}
        # Require both actual-color and counterfactual source-attribution controls.
        if len(clearances)==36 and len(values)==1 and None not in values and any("sourceDelta" in k for k in clearances):
            frame=next(iter(values))
            bracket=deepcopy([r for r in result["retained_frames"] if frame-1<=r["frame"]<=frame+1])
            if {r["frame"] for r in bracket}!={frame-1,frame,frame+1}:
                raise ValueError("white-core completion needs retained native bracket")
            measurement=deepcopy(document["events"][EVENT]["measurement"])
            measurement.update(method=core["method"],claim="completion",frames=bracket,
                coverage=dict(first=frame-1,last=frame+1,complete=True,missing=[]),roi=core["roi"],
                predicate="All36 white-core color/count/source-contribution controls are zero from this frame through D2757; preceding frame remains positive.",
                controls=deepcopy(clearances),
                limitations="Completion of the displayed held white core in the native pre-finish composite. Physical extinction, finished-film visibility, sound level and fade shape are not measured.")
            measurement.pop("observed_transition_onset",None)
            document["bounds"]={"D.new.forging.fire":{"end":dict(frame=frame,status="measured",offset_f=0,
                source="native pre-finish held thinking-fire white-core occlusion",
                measured_ref="deep_measurements.json#thinking_fire_core",measurement_scope="native_pre_finish_composite",
                measurement=measurement)}}
    return document


def scan(local, receipt_path, out):
    started = time.perf_counter()
    local, out = Path(local).resolve(), Path(out).resolve()
    for forbidden in (ROOT, ROOT/"renders", local):
        if out.is_relative_to(forbidden.resolve()):
            raise ValueError("output must be external to repository and source roots")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["CUTD_LOCAL_RENDERS"] = str(local)
    os.environ["NUMBA_CACHE_DIR"] = str(out/"numba_cache")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
    receipt = json.loads(Path(receipt_path).read_text())
    dependencies = validate_refresh(receipt, local)
    inputs = {name: digest(ROOT/name) for name in INPUTS}
    edl = selected_edl(json.loads((ROOT/INPUTS[0]).read_text()))
    sys.path.insert(0, str(ROOT/"edit"))
    import assemble as AS
    AS.CACHE = str(out/"assembler_cache")
    AS._INDEX.clear()
    if selected_edl(AS.edl_doc("D")) != edl:
        raise ValueError("runtime EDL differs from exported selected construction")
    runtime_edl_sha = digest(ROOT/"edit/edl_v3.py")
    AS._init("D", None, 1., True, False)
    ctx = AS._CTX
    for code in ("D11a", "D11b"):
        index = next(i for i, s in enumerate(ctx.shots) if s["code"] == code)
        if ctx.plans[index]["take"] != AS.EDL.D_DEEP_ENTRY:
            raise ValueError("assembler did not select independent Deep construction")
    farm = ROOT/"renders/embers_D_race/f_02719.jpg"
    pillow_baseline = native_rgb(farm)
    baseline = (np.clip(ctx.read(str(farm)), 0., 1.)*255.+.5).astype(np.uint8)
    decoder_comparison = changed_pixels(baseline, pillow_baseline)
    coreless_race=baseline.astype(np.float32)/255.
    x0,y0,x1,y1=CORE_ROI
    core_crop=baseline[y0:y1,x0:x1].astype(np.int16)
    core_mask=(core_crop.min(axis=-1)>=min(CORE_FLOORS))&(np.ptp(core_crop,axis=-1)<=60)
    coreless_race[y0:y1,x0:x1][core_mask]=0.
    sheet = Image.new("RGB", (4*384, 10*184), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    core_sheet=Image.new("RGB",(6*150,7*246),(12,12,12))
    core_draw=ImageDraw.Draw(core_sheet)
    control_sheet=Image.new("RGB",(6*200,7*171),(12,12,12))
    control_draw=ImageDraw.Draw(control_sheet)
    rows, retained = [], []
    (out/"composites").mkdir(exist_ok=True)
    for f in range(FIRST, LAST+1):
        picture, _, status, source = ctx.picture(f)
        if source != AS.EDL.D_DEEP_ENTRY or status.startswith("SLATE"):
            raise ValueError("assembler returned a fallback or slate")
        pixels = (np.clip(picture, 0., 1.)*255.+.5).astype(np.uint8)
        if pixels.shape != (804, 1920, 3):
            raise ValueError("assembler produced nonnative composite")
        if f == FIRST and not np.array_equal(pixels, baseline):
            raise ValueError("D2720 is not the exact held race")
        counter=AS.DEEP.sweep(coreless_race,str(local/"cutd_deep_sweep_coeff"/f"f_{f:05d}.npz"),f)
        counter_pixels=(np.clip(counter,0.,1.)*255.+.5).astype(np.uint8)
        row = dict(frame=f, pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                   core=core_stats(pixels,baseline,counterfactual=counter_pixels),
                   coreless_counterfactual_pixel_sha256=hashlib.sha256(counter_pixels.tobytes()).hexdigest(),
                   page_resemblance=page_resemblance(pixels, baseline,
                       native_rgb(local/"cutd_deep_clean"/f"f_{f:05d}.png")),
                   **changed_pixels(pixels, baseline))
        rows.append(row)
        image = Image.fromarray(pixels)
        if f in KEEP:
            relative = f"composites/f_{f:05d}.png"
            image.save(out/relative)
            retained.append(dict(frame=f, source_frame=f, root="soundd_deep_composites", path=relative,
                                 sha256=digest(out/relative), pixel_sha256=row["pixel_sha256"]))
        x, y = ((f-FIRST) % 4)*384, ((f-FIRST)//4)*184
        sheet.paste(image.resize((384, 161), Image.Resampling.LANCZOS), (x, y+21))
        draw.text((x+5, y+3), f"D{f} pre-finish", fill="white")
        cx,cy=((f-FIRST)%6)*150,((f-FIRST)//6)*246
        core_sheet.paste(image.crop(CORE_ROI).resize((150,225),Image.Resampling.NEAREST),(cx,cy+21))
        core_draw.text((cx+4,cy+3),f"D{f} core 3x",fill="white")
        cx,cy=((f-FIRST)%6)*200,((f-FIRST)//6)*171
        control_sheet.paste(image.crop(CORE_ROI).resize((100,150),Image.Resampling.NEAREST),(cx,cy+21))
        control_sheet.paste(Image.fromarray(counter_pixels).crop(CORE_ROI).resize((100,150),Image.Resampling.NEAREST),(cx+100,cy+21))
        control_draw.text((cx+2,cy+3),f"{f} actual | core removed",fill="white")
    # Detect concurrent source/metadata changes before publishing the receipt.
    for dependency in dependencies:
        root = ROOT if dependency["root"] == "repo" else local
        if digest(root/dependency["path"]) != dependency["sha256"]:
            raise ValueError("construction changed during scan")
    if inputs != {name: digest(ROOT/name) for name in INPUTS}:
        raise ValueError("metadata changed during scan")
    if runtime_edl_sha != digest(ROOT/"edit/edl_v3.py"):
        raise ValueError("runtime EDL changed during scan")
    sheet.save(out/"deep_contact_sheet.jpg", quality=90)
    core_sheet.save(out/"thinking_fire_core_sheet.png")
    control_sheet.save(out/"thinking_fire_counterfactual_sheet.png")
    result = dict(scope="native_pre_finish_composite", frame_size=list(SIZE),
                  coverage=dict(first=FIRST, last=LAST, complete=True, missing=[]),
                  frames=rows, retained_frames=retained, dependencies=dependencies,
                  edl_dependencies=edl, input_sha256=inputs,
                  refresh_receipt_sha256=digest(receipt_path),
                  baseline_decoder="assembler Ctx.read/OpenCV; same decoder as composite",
                  baseline_pixel_sha256=hashlib.sha256(baseline.tobytes()).hexdigest(),
                  pillow_vs_assembler_jpeg_decode=decoder_comparison,
                  runtime_edl_source_sha256=runtime_edl_sha,
                  prior_runtime_scope="master-finished pre-title; those pixel hashes are not compared to this pre-finish scan",
                  onsets={field: {str(t): first_change(rows, field, t) for t in THRESHOLDS}
                          for field in ("changed_pixels", "new_warm_pixels")},
                  page_proxy_candidates=page_candidates(rows),
                  warm_proxy_clearance={str(t): trailing_clearance(rows,t) for t in THRESHOLDS},
                  thinking_fire_core=dict(roi=list(CORE_ROI),thresholds=list(CORE_FLOORS),
                    method="Fixed white-core masks from held D2719 in x935:985,y320:395; min(R,G,B)>=140/180/220, max-min<=60, with at least1/3/5 matching pixels. Counterfactual zeroes only held core pixels before the identical sweep, and requires actual-vs-counterfactual max-channel contribution >=1/4/8 to reject unrelated white burn-edge light. Scan all38 actual and controlled pre-finish composites. No physical extinction or finished-film claim.",
                    clearance=core_clearance(rows)),
                  page_proxy_method="RGB max-channel distance to corresponding clean page <=8/12/16, excluding pixels whose held-race distance to that page is <=32; centre column x960. Page lighting, char and fire can violate resemblance. This is not alpha or source footage coverage.",
                  anchor=dict(event=EVENT, old_frame=2739, measured_frame=None, adopted_frame=2739,
                              delta_frames=0, status="estimated", reason="Donor marker is mid-sweep envelope crest, not first-visible transition."),
                  total_seconds=time.perf_counter()-started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/"deep_measurements.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"deep_overlay.json").write_text(json.dumps(overlay(result), indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-renders", required=True, type=Path)
    parser.add_argument("--refresh-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = scan(args.local_renders, args.refresh_receipt, args.out)
    print(json.dumps({key: result[key] for key in ("scope", "coverage", "onsets", "anchor", "total_seconds", "peak_rss_bytes")}), flush=True)


if __name__ == "__main__":
    main()
