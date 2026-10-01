"""Read-only delivered D action-plate inventory and opt-in pixel measurements.

Run image operations through the production owner's onepy guard. This module
never imports a renderer or edits shared frames. Rendering schedules can guide
an explicit search window; only decoded image evidence can establish an event.
All native frame ranges in reports are inclusive unless named half_open.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SHOTS = {
    "inscription": (1680, 2080), "forging": (2080, 2400), "race": (2400, 2720),
    "brink": (2960, 3200), "vision": (3200, 3440), "gap": (3440, 3520),
    "trap": (3760, 4080), "holdout": (5480, 5520), "instep": (5680, 5840),
    "unfinished": (6080, 6400),
}
INPUTS = ("edit/edl/edl_D.json", "music/v3/barmap_D.json", "music/v3/cues_D.json",
          "music/v3/events_D_measured.json", "music/v3/events_D_round4_measured.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runs(frames):
    out = []
    for frame in sorted(set(frames)):
        if out and out[-1][1]+1 == frame:
            out[-1][1] = frame
        else:
            out.append([frame, frame])
    return out


def stem_files(root, stem):
    if not re.fullmatch(r"embers_D_[A-Za-z0-9_]+", stem):
        raise ValueError("invalid action plate stem")
    files = {}
    for path in sorted((Path(root)/"renders"/stem).glob("f_*")):
        match = re.fullmatch(r"f_(\d+)\.(jpg|png)", path.name)
        if not match:
            continue
        frame = int(match[1])
        if frame in files:
            raise ValueError(f"ambiguous delivered frame {stem}:{frame}")
        files[frame] = path
    return files


def frame_files(root, shot):
    return stem_files(root, "embers_D_"+shot)


def coverage(files, first, last):
    required = range(first, last+1)
    missing = [frame for frame in required if frame not in files]
    return dict(complete=not missing, first=first, last=last, missing=missing,
                required_count=last-first+1, present_count=last-first+1-len(missing))


def inventory(root=ROOT):
    rows = {}
    for shot, (first, stop) in SHOTS.items():
        files = frame_files(root, shot)
        rows[shot] = dict(stem="embers_D_"+shot, coverage=coverage(files, first, stop-1),
                          present_runs=runs(files), extra_frames=sorted(set(files)-set(range(first, stop))))
    return dict(utc=datetime.now(timezone.utc).isoformat(), measurement_scope="filename inventory only",
                shots=rows)


def read_frame(path, frame, root=ROOT):
    before = sha256(path)
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
    after = sha256(path)
    if before != after:
        raise ValueError(f"frame changed during read: {frame}")
    if image.size != (1920, 804):
        raise ValueError(f"non-native delivered frame {frame}: {image.size}")
    evidence = dict(frame=frame, source_frame=frame, source_stem=path.parent.name,
                    path=path.relative_to(root).as_posix(), sha256=before, frame_size=list(image.size))
    return image, evidence


def contact_sheet(root, shot, frames, output, roi=None, columns=3, cell_width=640):
    """Selected states only. A contact sheet alone never claims onset coverage."""
    files, evidence, thumbs = frame_files(root, shot), [], []
    for frame in frames:
        if frame not in files:
            raise ValueError(f"missing contact-sheet frame {shot}:{frame}")
        image, record = read_frame(files[frame], frame, root)
        if roi is not None:
            image = image.crop(tuple(roi))
        height = round(image.height*cell_width/image.width)
        image = image.resize((cell_width, height), Image.Resampling.LANCZOS)
        label = Image.new("RGB", (cell_width, height+26), "#171717")
        label.paste(image, (0, 26))
        ImageDraw.Draw(label).text((8, 6), f"{shot} D{frame}", fill="white")
        thumbs.append(label)
        evidence.append(record)
    if not thumbs:
        raise ValueError("no contact-sheet frames")
    height = max(i.height for i in thumbs)
    sheet = Image.new("RGB", (cell_width*columns, height*((len(thumbs)+columns-1)//columns)), "black")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % columns)*cell_width, (i//columns)*height))
    sheet.save(output)
    return dict(measurement_scope="native_delivered_plate", claim="observed_state", roi=roi,
                sheet=Path(output).name, frames=evidence)


def roi_values(rgb, previous=None):
    """Uncalibrated RGB-code traces; thresholds must be explicitly declared."""
    x = np.asarray(rgb, np.float32)/255.
    value = x.max(axis=2)
    gold = x[:, :, 0]-np.maximum(x[:, :, 1], x[:, :, 2])
    out = dict(mean=float(value.mean()), p99=float(np.quantile(value, .99)),
               hot_fraction=float(np.mean(value >= .75)), gold_mean=float(gold.mean()))
    if previous is not None:
        delta = x-np.asarray(previous, np.float32)/255.
        out.update(mean_abs_change=float(np.abs(delta).mean()),
                   positive_change=float(np.maximum(delta, 0).mean()),
                   changed_fraction=float(np.mean(np.max(np.abs(delta), axis=2) >= 8/255)))
    else:
        out.update(mean_abs_change=None, positive_change=None, changed_fraction=None)
    return out


def response_frames(trace, threshold, field="positive_change"):
    """Thresholded visible-response candidates, requiring contiguous coverage.

    This helper does not label an action. Contact classification still needs an
    inspected bracket, because lighting and camera motion also change pixels.
    """
    cov, rows = trace["coverage"], trace["trace"]
    if (not cov["complete"] or cov["missing"] or
            [r["frame"] for r in rows] != list(range(cov["first"], cov["last"]+1))):
        raise ValueError("response candidates require complete contiguous trace")
    return [r["frame"] for r in rows if r.get(field) is not None and r[field] >= threshold]


def scan_roi(root, shot, first, last, roi):
    """Complete per-frame ROI trace; refuse missing frames instead of bridging."""
    files = frame_files(root, shot)
    cov = coverage(files, first, last)
    if not cov["complete"]:
        return dict(measurement_scope="native_delivered_plate", status="unmeasured", frame=None,
                    reason="incomplete requested frame coverage", coverage=cov, frames=[])
    rows, evidence, previous = [], [], None
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 804):
        raise ValueError("invalid native ROI")
    for frame in range(first, last+1):
        image, record = read_frame(files[frame], frame, root)
        crop = np.array(image)[y0:y1, x0:x1]
        rows.append(dict(frame=frame, **roi_values(crop, previous)))
        previous = crop
        evidence.append(record)
    return dict(measurement_scope="native_delivered_plate", claim="pixel_trace_only", coverage=cov,
                method="RGB code values, adjacent-frame difference in explicit native ROI; no automatic event claim",
                frame_size=[1920, 804], roi=list(roi), frames=evidence, trace=rows)


def compare_take(root, shot, first, last, roi, comparison_stem):
    """Decode both complete native takes; a changed hash is not an onset.

    The selected take alone supplies ``frames``. Superseded plates are kept in
    a separate comparison field so callers cannot accidentally bind to them.
    No automatic action, camera-direction or lighting-cause claim is made.
    """
    selected = frame_files(root, shot)
    comparison = stem_files(root, comparison_stem)
    cov = coverage(selected, first, last)
    old_cov = coverage(comparison, first, last)
    if not cov["complete"] or not old_cov["complete"]:
        raise ValueError("paired take trace requires complete coverage in both takes")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 804):
        raise ValueError("invalid native ROI")
    rows, evidence, old_evidence, previous = [], [], [], None
    for frame in range(first, last+1):
        image, record = read_frame(selected[frame], frame, root)
        old_image, old_record = read_frame(comparison[frame], frame, root)
        crop = np.asarray(image)[y0:y1, x0:x1]
        old_crop = np.asarray(old_image)[y0:y1, x0:x1]
        difference = crop.astype(np.int16)-old_crop.astype(np.int16)
        rows.append(dict(frame=frame, **roi_values(crop, previous),
                         bytes_changed=record["sha256"] != old_record["sha256"],
                         comparison_mean_abs_code_change=float(np.abs(difference).mean()),
                         comparison_max_abs_code_change=int(np.abs(difference).max())))
        previous = crop.copy()
        evidence.append(record)
        old_evidence.append(old_record)
    return dict(measurement_scope="native_delivered_plate", claim="pixel_trace_only",
                method="Complete paired native decode and SHA256 comparison; selected-take RGB trace in explicit ROI. No automatic event claim.",
                frame_size=[1920, 804], coverage=cov, comparison_coverage=old_cov,
                roi=list(roi), frames=evidence, comparison_frames=old_evidence,
                changed_runs=runs(r["frame"] for r in rows if r["bytes_changed"]),
                unchanged_runs=runs(r["frame"] for r in rows if not r["bytes_changed"]),
                trace=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "sheet", "trace"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shot", choices=SHOTS)
    parser.add_argument("--frames", help="comma-separated absolute D frames (sheet)")
    parser.add_argument("--first", type=int)
    parser.add_argument("--last", type=int)
    parser.add_argument("--roi", help="native x0,y0,x1,y1")
    args = parser.parse_args(argv)
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    roi = tuple(map(int, args.roi.split(","))) if args.roi else None
    if args.mode == "inventory":
        result = inventory(args.root)
        filename = "inventory.json"
    elif args.mode == "sheet":
        if not args.shot or not args.frames:
            parser.error("sheet needs --shot and --frames")
        filename = args.shot+"_sheet.json"
        result = contact_sheet(args.root, args.shot, list(map(int, args.frames.split(","))),
                               args.out/(args.shot+"_sheet.jpg"), roi)
    else:
        if not args.shot or args.first is None or args.last is None or roi is None:
            parser.error("trace needs --shot, --first, --last, and --roi")
        filename = f"{args.shot}_{args.first}_{args.last}_trace.json"
        result = scan_roi(args.root, args.shot, args.first, args.last, roi)
    result["input_sha256"] = {path: sha256(args.root/path) for path in INPUTS}
    result["seconds"] = time.monotonic()-started
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)
    (args.out/filename).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.out/filename), seconds=result["seconds"], peak_rss_bytes=result["peak_rss_bytes"])), flush=True)


if __name__ == "__main__":
    main()
