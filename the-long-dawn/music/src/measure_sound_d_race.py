"""Opt-in, read-only race plate evidence collection.

Source work clocks are hypotheses. This tool decodes native images and emits
pixel traces; classification and inspected brackets remain explicit receipts.
Run image modes through the production owner's onepy guard.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy import ndimage

import measure_sound_d_actions as A

ROOT = Path(__file__).resolve().parents[2]
FIRST, LAST = 2400, 2719
STEMS = ("race", "race_r4old")
SOURCES = ("shots/embers/c_d.py", "shots/embers/openring_d.py",
           "shots/embers/d_race_embers.py", "shots/embers/d_thinking_fire.py")
REGIONS = {"ring": (780, 40, 1160, 490),
           "field": (0, 400, 1920, 804),
           "left_field": (0, 400, 760, 804),
           "right_field": (1160, 400, 1920, 804)}


def complete_rows(trace):
    cov = trace["coverage"]
    if (not cov["complete"] or cov["missing"] or
            [row["frame"] for row in trace["trace"]] != list(range(cov["first"], cov["last"] + 1))):
        raise ValueError("requires complete contiguous pixel coverage")
    return trace["trace"]


def candidates(trace, region, threshold, field="positive_change"):
    """Pixel candidates only; no automatic contact or work-onset claim."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    return [row["frame"] for row in complete_rows(trace)
            if row[region].get(field) is not None and row[region][field] >= threshold]


def isolated_responses(trace, region="ring", minimum=.012, ratio=1.5):
    """Interior isolated pulse candidates, independent of the renderer clock.

    A smooth camera reveal can exceed an absolute difference threshold. Requiring
    prominence above both adjacent changes separates isolated pulse responses;
    the result still needs visual classification and does not prove contact.
    """
    if not np.isfinite(minimum) or minimum <= 0 or not np.isfinite(ratio) or ratio <= 1:
        raise ValueError("invalid response prominence thresholds")
    rows = complete_rows(trace)
    selected = []
    for index in range(2, len(rows)-1):
        values = [rows[j][region]["positive_change"] for j in (index-1,index,index+1)]
        if any(v is None or not np.isfinite(v) for v in values):
            raise ValueError("invalid adjacent pixel response")
        before, value, after = values
        if value >= minimum and value >= ratio * max(before, after):
            selected.append(rows[index]["frame"])
    return selected


def work_responses(trace, minimum_bilateral_jump=.004):
    """Measured field brightening synchronous with an isolated Ring response.

    This predicate concerns two field regions, not individual hammer contacts.
    It excludes secondary brightness waves without claiming they are absent.
    The first plate frame has no preceding same-plate evidence and is omitted.
    """
    if not np.isfinite(minimum_bilateral_jump) or minimum_bilateral_jump <= 0:
        raise ValueError("invalid bilateral threshold")
    pulses = set(isolated_responses(trace))
    rows = complete_rows(trace)
    return [row["frame"] for index, row in enumerate(rows) if index and row["frame"] in pulses
            and all(row[k]["mean"]-rows[index-1][k]["mean"] >= minimum_bilateral_jump
                    for k in ("left_field", "right_field"))]


def deltas(old, new):
    """Set comparison avoids pairing a new work event to an unrelated old hit."""
    if old != sorted(set(old)) or new != sorted(set(new)):
        raise ValueError("event frames must be unique and ordered")
    return dict(retained=sorted(set(old) & set(new)), removed=sorted(set(old)-set(new)),
                added=sorted(set(new)-set(old)), old_count=len(old), new_count=len(new))


def white_core(rgb, roi, threshold=.3, minimum_pixels=30, *, anchor=None, max_distance=None):
    """Connected white component in an explicitly inspected fire ROI.

    This measures visible core presence and location, not physical emitter
    identity. Warm gold and isolated bright grains are negative controls.
    By default choose the largest; an optional anchor tracks the nearest within
    a stated distance. Coordinates are native pixels; bounding boxes are half-open.
    """
    if not np.isfinite(threshold) or not 0 < threshold < 1 or minimum_pixels < 1:
        raise ValueError("invalid white-core threshold")
    if (anchor is None) != (max_distance is None):
        raise ValueError("core tracking requires anchor and maximum distance")
    if anchor is not None and (len(anchor) != 2 or not np.isfinite(anchor).all()
                              or not np.isfinite(max_distance) or max_distance <= 0):
        raise ValueError("invalid core tracking anchor or distance")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
        raise ValueError("invalid core ROI")
    crop = np.asarray(rgb)[y0:y1, x0:x1]
    mask = crop.min(axis=2)/255. > threshold
    labels, count = ndimage.label(mask)  # four-connected, no morphology/gap fill
    areas = np.bincount(labels.ravel())[1:]
    eligible = np.flatnonzero(areas >= minimum_pixels) + 1
    distances = {}
    if anchor is not None:
        for label in eligible:
            yy, xx = np.nonzero(labels == label)
            distances[int(label)] = float(np.hypot(xx.mean()+x0-anchor[0], yy.mean()+y0-anchor[1]))
        eligible = [label for label in eligible if distances[int(label)] <= max_distance]
    if not len(eligible):
        return dict(present=False, pixels=0, centroid=None, bbox=None,
                    largest_candidate_pixels=int(areas.max()) if count else 0)
    chosen = (min(eligible, key=lambda label: distances[int(label)]) if anchor is not None
              else max(eligible, key=lambda label: areas[label-1]))
    selected = labels == chosen
    yy, xx = np.nonzero(selected)
    return dict(present=True, pixels=int(selected.sum()),
                centroid=[float(xx.mean()+x0), float(yy.mean()+y0)],
                bbox=[int(xx.min()+x0),int(yy.min()+y0),int(xx.max()+x0+1),int(yy.max()+y0+1)],
                mean_rgb=(crop[selected].mean(axis=0)/255.).tolist(),
                largest_candidate_pixels=int(areas.max()),
                anchor_distance=distances.get(int(chosen)))


def scan(root, shot, regions=REGIONS, first=FIRST, last=LAST, *, core_roi=None,
         core_anchor=None, core_max_distance=20):
    if first > last:
        raise ValueError("reversed frame range")
    if core_anchor is not None and core_roi is None:
        raise ValueError("core tracking requires a core ROI")
    for roi in regions.values():
        if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1920 and 0 <= roi[1] < roi[3] <= 804):
            raise ValueError("invalid native ROI")
    files = A.frame_files(root, shot)
    cov = A.coverage(files, first, last)
    if not cov["complete"]:
        return dict(status="unmeasured", frame=None, reason="incomplete delivered race coverage",
                    coverage=cov, frames=[], trace=[])
    rows, frames, previous = [], [], {}
    anchor = core_anchor
    for frame in range(first, last+1):
        im, record = A.read_frame(files[frame], frame, root)
        pixels = np.asarray(im)
        row = dict(frame=frame)
        for name, (x0, y0, x1, y1) in regions.items():
            crop = pixels[y0:y1, x0:x1]
            row[name] = A.roi_values(crop, previous.get(name))
            previous[name] = crop.copy()
        if core_roi is not None:
            row["white_core"] = white_core(pixels, core_roi, anchor=anchor,
                                            max_distance=core_max_distance if anchor is not None else None)
            if anchor is not None and row["white_core"]["present"]:
                anchor = row["white_core"]["centroid"]
        rows.append(row)
        frames.append(record)
    result = dict(measurement_scope="native_delivered_plate", claim="pixel_trace_only",
                method="Adjacent native RGB-code ROI traces; thresholds select candidates, not automatic event labels",
                frame_size=[1920, 804], regions={k:list(v) for k,v in regions.items()},
                coverage=cov, frames=frames, trace=rows)
    if core_roi is not None:
        result["core_predicate"] = dict(roi=list(core_roi),min_rgb_exclusive=.3,
                                       minimum_connected_pixels=30,connectivity=4)
        if core_anchor is not None:
            result["core_predicate"].update(initial_anchor=list(core_anchor),
                max_centroid_step=core_max_distance,
                selection='Nearest qualifying component to previous measured centroid; missing frames remain absent')
    return result


def output_dir(path, root):
    resolved = path.resolve()
    if resolved.is_relative_to((root/"renders").resolve()):
        raise ValueError("output must not write shared renders")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "scan", "sheet"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shot", choices=STEMS, default="race")
    parser.add_argument("--frames", default="2400,2420,2440,2460,2520,2560,2640,2680,2719")
    parser.add_argument("--roi")
    parser.add_argument("--core-roi", help="optional native fire ROI x0,y0,x1,y1; records core presence only")
    parser.add_argument("--core-anchor", help="optional inspected first core center x,y; tracks measured centroid within20px")
    args = parser.parse_args(argv)
    start = time.monotonic()
    out = output_dir(args.out, args.root)
    if args.mode == "inventory":
        result = {shot:A.coverage(A.frame_files(args.root, shot), FIRST, LAST) for shot in STEMS}
    elif args.mode == "scan":
        core_roi = tuple(map(int,args.core_roi.split(","))) if args.core_roi else None
        core_anchor = tuple(map(float,args.core_anchor.split(","))) if args.core_anchor else None
        if core_anchor is not None and core_roi is None:
            parser.error("--core-anchor requires --core-roi")
        result = scan(args.root, args.shot, core_roi=core_roi, core_anchor=core_anchor)
    else:
        roi = tuple(map(int, args.roi.split(","))) if args.roi else None
        result = A.contact_sheet(args.root, args.shot, list(map(int, args.frames.split(","))),
                                 out/(args.shot+"_sheet.jpg"), roi)
    result.update(utc=datetime.now(timezone.utc).isoformat(),
                  input_sha256={p:A.sha256(args.root/p) for p in A.INPUTS},
                  source_sha256={p:A.sha256(args.root/p) for p in SOURCES},
                  seconds=time.monotonic()-start,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024))
    path = out/(args.shot+"_"+args.mode+".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(path), seconds=result["seconds"], peak_rss_bytes=result["peak_rss_bytes"])), flush=True)


if __name__ == "__main__":
    main()
