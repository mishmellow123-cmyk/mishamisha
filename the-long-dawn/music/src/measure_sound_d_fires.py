"""Opt-in, bounded-memory observations of delivered D fire plates.

Renderer constants only select search regions. An onset is reported only after
decoding every frame from the declared coverage start through that onset and
its preceding frame. No renderer imports, audio, source writes, or estimates.
Run this CLI through the production slot tool; outputs require an explicit
external directory. Source plates are not assembled/graded D film frames.
"""
from copy import deepcopy
from io import BytesIO
import argparse
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
RANGES = {"embers_D_crowns": (4240, 4560), "falsedawn_brink": (4080, 4240),
          "falsedawn_twofires": (4560, 5040), "falsedawn_watch": (7040, 7360),
          "falsedawn_truedawn": (7360, 7840)}
# Search regions around the renderer's basket roots; these are not measured
# screen positions. Below-rim furnace light is excluded from the search.
CROWN_ROIS = {"left": [713, 441, 754, 467], "right": [1366, 433, 1408, 459]}
# Localized from delivered D4869 and D5029, then visually checked against
# those plates. The whole shot is searched; renderer ignition constants are
# deliberately absent from the measurement logic.
RIDGE_ROIS = {"ridge1": [1609, 548, 1644, 598], "ridge2": [562, 546, 601, 603],
              "ridge3": [833, 525, 876, 586], "ridge4": [1472, 528, 1515, 595],
              "ridge5": [1518, 510, 1563, 585]}
# The direct sun disc in delivered D7834 is centred near x1072/y502. This
# region includes its approach to the skyline, above the foreground beacon.
SUN_ROI = {"sun": [1045, 475, 1100, 535]}
SUN_THRESHOLDS = (190, 210, 230)
THRESHOLDS = (90, 110, 130)
INPUTS = ("edit/edl/edl_D.json", "music/v3/barmap_D.json", "music/v3/cues_D.json",
          "music/v3/events_D_measured.json", "music/v3/events_D_round4_measured.json")


def inventory(root=ROOT):
    """Read filenames only; ranges are half-open and include missing indices."""
    out = {}
    for stem, (first, end) in RANGES.items():
        frames = {}
        for path in sorted((Path(root)/"renders"/stem).glob("f_*")):
            match = re.fullmatch(r"f_(\d+)\.(jpg|png)", path.name)
            if not match:
                continue
            frame = int(match[1])
            if not first <= frame < end:
                continue
            if frame in frames:
                raise ValueError(f"duplicate delivered frame: {stem}/{frame}")
            frames[frame] = path.relative_to(root).as_posix()
        missing = [f for f in range(first, end) if f not in frames]
        out[stem] = dict(first=first, end=end, present=sorted(frames), missing=missing,
                         complete=not missing, paths=frames)
    return out


def warm_stats(rgb, roi, threshold=110):
    """Measure bright warm pixels, with thresholds in decoded 8-bit RGB units."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
        raise ValueError("ROI outside image")
    if not 0 <= threshold <= 255:
        raise ValueError("invalid brightness threshold")
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    r, g, b = np.moveaxis(crop, -1, 0)
    mask = (r >= threshold) & (g >= 30) & (r >= 1.25*g) & (g >= 1.25*b)
    yy, xx = np.nonzero(mask)
    centroid = [float(xx.mean()+x0), float(yy.mean()+y0)] if len(xx) else None
    return dict(pixels=int(mask.sum()), centroid_xy=centroid,
                maximum_rgb=crop.reshape(-1, 3).max(axis=0).astype(int).tolist())


def solar_stats(rgb, roi, threshold=210):
    """Bright nearly neutral pixels distinguish the disc from the rose sky.

    The full earlier prefix remains a negative control for the neighbouring
    persistent fire. This defines a decoded-pixel landmark, not light's first
    physical contribution to a continuously brightening scene.
    """
    warm_stats(rgb, roi, threshold)  # share input/ROI validation
    x0,y0,x1,y1=roi
    crop=np.asarray(rgb)[y0:y1,x0:x1].astype(np.int16)
    mask=(crop.min(axis=-1)>=threshold)&(np.ptp(crop,axis=-1)<=30)
    yy,xx=np.nonzero(mask)
    return dict(pixels=int(mask.sum()),
                centroid_xy=[float(xx.mean()+x0),float(yy.mean()+y0)] if len(xx) else None,
                maximum_rgb=crop.reshape(-1,3).max(axis=0).tolist())


def first_onset(samples, first, last, *, min_pixels=2):
    """Return a point only for a fully covered, dark-to-lit sequence prefix.

    Sparse positive frames remain observed states. A missing earlier frame
    prevents a first-onset assertion even if the immediate pair is present.
    """
    if first >= last or min_pixels < 1:
        raise ValueError("invalid onset search")
    if any(type(f) is not int or not first <= f <= last for f in samples):
        raise ValueError("sample outside declared coverage")
    lit = sorted(f for f, value in samples.items() if value >= min_pixels)
    candidate = lit[0] if lit else None
    missing = [f for f in range(first, (candidate if candidate is not None else last)+1) if f not in samples]
    complete = not missing
    onset = candidate if candidate is not None and complete and candidate > first and samples[candidate-1] < min_pixels else None
    return dict(frame=onset, first_observed_lit=candidate,
                claim="onset" if onset is not None else "observed_state",
                coverage=dict(complete=complete, first=first,
                              last=candidate if candidate is not None else last, missing=missing),
                reason="complete dark-to-lit prefix" if onset is not None else
                       "missing earlier frames" if missing else
                       "already lit at coverage start" if candidate == first else "no onset detected")


def scan(root, stem, first, last, rois, thresholds=THRESHOLDS, *, statistic=warm_stats,
         chroma_rule="R>=1.25G; G>=30; G>=1.25B", min_pixels=2):
    """Decode one delivered frame at a time and hash those exact source bytes."""
    root = Path(root)
    snapshot = inventory(root)[stem]
    if not snapshot["first"] <= first < last < snapshot["end"]:
        raise ValueError("scan outside shot")
    frames, errors = [], []
    for frame in range(first, last+1):
        relative = snapshot["paths"].get(frame)
        if relative is None:
            continue
        try:
            raw = (root/relative).read_bytes()
            with Image.open(BytesIO(raw)) as image:
                if image.size != (1920, 804):
                    raise ValueError("requires native 1920x804 delivered plate")
                rgb = np.asarray(image.convert("RGB"))
            stats = {name: {str(t): statistic(rgb, roi, t) for t in thresholds}
                     for name, roi in rois.items()}
            frames.append(dict(frame=frame, path=relative, sha256=hashlib.sha256(raw).hexdigest(),
                               source_stem=stem, source_frame=frame, stats=stats))
        except (OSError, ValueError) as exc:
            errors.append(dict(frame=frame, path=relative, error=str(exc)))
    detections = {name: {str(t): first_onset({r["frame"]: r["stats"][name][str(t)]["pixels"] for r in frames},
                                          first, last, min_pixels=min_pixels) for t in thresholds} for name in rois}
    return dict(stem=stem, scan_range_inclusive=[first, last], rois=deepcopy(rois), thresholds=list(thresholds),
                min_pixels=min_pixels, chroma_rule=chroma_rule, frame_size=[1920, 804],
                scope="native_delivered_plate", detections=detections, frames=frames, errors=errors)


def crown_hook(result, ref="crowns_measurements.json"):
    """Promote only an agreed two-basket onset across all sensitivity controls."""
    outcomes = [d for side in result["detections"].values() for d in side.values()]
    onsets = {d["frame"] for d in outcomes}
    if len(outcomes) != 6 or len(onsets) != 1 or None in onsets:
        return None
    frame = onsets.pop()
    evidence = [{k: r[k] for k in ("frame", "path", "sha256", "source_stem", "source_frame")}
                for r in result["frames"] if r["frame"] <= frame]
    return dict(frame=frame, status="measured", source="delivered-plate warm-pixel onset, both crown baskets",
                measured_ref=ref, measurement_scope="native_delivered_plate",
                measurement=dict(method="Full-prefix native RGB scan; independent left/right ROIs; three brightness thresholds",
                    claim="onset", frame_size=[1920, 804], frames=evidence,
                    coverage=deepcopy(outcomes[0]["coverage"]), roi=deepcopy(result["rois"]),
                    threshold=dict(red_min=list(result["thresholds"]), warm_pixels=2, chroma=result["chroma_rule"]),
                    uncertainty_frames=0, uncertainty_scope="integer decoded-plate frame at specified thresholds; no subframe or audible-arrival claim"))


def ridge_series(result, ref="ridges_measurements.json"):
    """Return all new catches only after a complete search of the whole shot.

    The two crowns are already alight at the shot's first frame, as separately
    inspected. They receive no attack in this replacement series.
    """
    first, last = result["scan_range_inclusive"]
    seen = {r["frame"] for r in result["frames"]}
    missing = [f for f in range(first,last+1) if f not in seen]
    if (first,last) != (4560,5039) or missing or set(result["rois"]) != set(RIDGE_ROIS):
        return None
    events = []
    fields = ("frame", "path", "sha256", "source_stem", "source_frame")
    by_frame = {r["frame"]:r for r in result["frames"]}
    for name in RIDGE_ROIS:
        outcomes = list(result["detections"][name].values())
        values = {r["frame"] for r in outcomes}
        if len(outcomes) != 3 or len(values) != 1 or None in values:
            return None
        frame = values.pop()
        centroid = by_frame[frame]["stats"][name]["110"]["centroid_xy"]
        measurement = dict(method="Full-shot native RGB warm-pixel scan with three brightness thresholds",
                           claim="onset", frame_size=[1920,804],
                           frames=[{k:by_frame[f][k] for k in fields} for f in range(first,frame+1)],
                           coverage=deepcopy(outcomes[0]["coverage"]), roi=result["rois"][name].copy(),
                           threshold=dict(red_min=list(result["thresholds"]), warm_pixels=2, chroma=result["chroma_rule"]),
                           uncertainty_frames=0, centroid_xy=centroid.copy(),
                           uncertainty_scope="integer decoded plate at specified thresholds, not audible arrival")
        events.append(dict(id="D.new.ridges.catches."+name, frame=frame, status="measured",
                           source="delivered-plate localized ridge-fire onset", measured_ref=ref,
                           measurement_scope="native_delivered_plate", measurement=measurement,
                           pan=2*centroid[0]/1919-1,
                           spatial_basis="Authored linear stereo mapping 2*x/1919-1 of the measured first-lit warm-pixel centroid; no world distance inferred"))
    coverage = dict(complete=True, first=first, last=last, missing=[])
    return dict(events=events, coverage=coverage,
                measurement=dict(method="Full-shot search of five observed ridge-fire regions; pre-existing crown pair excluded",
                                 frame_size=[1920,804], frames=[{k:r[k] for k in fields} for r in result["frames"]],
                                 coverage=deepcopy(coverage), roi=deepcopy(result["rois"]),
                                 threshold=dict(red_min=list(result["thresholds"]), warm_pixels=2, chroma=result["chroma_rule"])))


def solar_hook(result, ref="sun_measurements.json"):
    """First bright, near-neutral solar-disc pixel with next-frame persistence.

    One native pixel is intentional for this first-emergence landmark. The
    entire preceding shot prefix must be negative at all three thresholds;
    the following delivered frame must remain positive at all thresholds.
    """
    outcomes=list(result["detections"]["sun"].values())
    values={r["frame"] for r in outcomes}
    if len(outcomes)!=3 or len(values)!=1 or None in values:
        return None
    frame=values.pop()
    refs={r["frame"]:r for r in result["frames"]}
    if frame+1 not in refs or any(refs[frame+1]["stats"]["sun"][str(t)]["pixels"]<1 for t in SUN_THRESHOLDS):
        return None
    fields=("frame","path","sha256","source_stem","source_frame")
    first=result["scan_range_inclusive"][0]
    return dict(frame=frame,status="measured",source="first bright near-neutral direct solar-disc pixel in delivered native plates",
                measured_ref=ref,measurement_scope="native_delivered_plate",
                measurement=dict(method="Full-prefix native RGB scan of localized sun region; three thresholds and next-frame persistence",
                    claim="onset",frame_size=[1920,804],frames=[{k:refs[f][k] for k in fields} for f in range(first,frame+2)],
                    coverage=dict(complete=True,first=first,last=frame+1,missing=[]),roi=deepcopy(result["rois"]),
                    threshold=dict(min_all_rgb=list(SUN_THRESHOLDS),max_channel_range=30,min_pixels=1,next_frame_persistence=True),
                    uncertainty_frames=0,
                    uncertainty_scope="first qualifying native pixel at specified thresholds; this does not claim first ambient illumination or assembled-film visibility"))


def ridge_sheet(root, result, output):
    """Before/first-lit/after crops; sparse candidates remain visibly labelled."""
    sheet=Image.new("RGB",(960,180*len(RIDGE_ROIS)),(18,18,18))
    draw=ImageDraw.Draw(sheet)
    refs={r["frame"]:r for r in result["frames"]}
    for row,(name,roi) in enumerate(RIDGE_ROIS.items()):
        detection=result["detections"][name]["110"]
        candidate=detection["first_observed_lit"]
        if candidate is None:
            draw.text((0,row*180),name+" no lit frame available",fill="white")
            continue
        prior=[f for f in refs if f<candidate]
        later=[f for f in refs if f>candidate]
        chosen=[max(prior) if prior else None,candidate,min(later) if later else None]
        for col,frame in enumerate(chosen):
            if frame is None:
                continue
            with Image.open(Path(root)/refs[frame]["path"]) as im:
                crop=im.convert("RGB").crop(roi)
                crop.thumbnail((280,140))
                crop=crop.resize((crop.width*2,crop.height*2))
                sheet.paste(crop,(col*320,row*180+25))
            draw.text((col*320,row*180),f"{name} D{frame} {detection['claim']}",fill="white")
    sheet.save(output,quality=93)


def solar_sheet(root,result,output):
    refs={r["frame"]:r for r in result["frames"]}
    candidates=[r["first_observed_lit"] for r in result["detections"]["sun"].values()
                if r["first_observed_lit"] is not None]
    if not candidates:
        return
    chosen=sorted(set(f for c in candidates for f in sorted(refs,key=lambda f:abs(f-c))[:7]))
    sheet=Image.new("RGB",(800,275*len(chosen)),(18,18,18))
    draw=ImageDraw.Draw(sheet)
    for row,frame in enumerate(chosen):
        with Image.open(Path(root)/refs[frame]["path"]) as im:
            im=im.convert("RGB")
            sheet.paste(im.crop((1030,480,1110,560)).resize((240,240),Image.Resampling.NEAREST),(0,row*275+25))
            sheet.paste(im.resize((540,226)),(260,row*275+25))
        counts={t:r["pixels"] for t,r in refs[frame]["stats"]["sun"].items()}
        draw.text((0,row*275+3),f"D{frame} bright pixels {counts}",fill="white")
    sheet.save(output,quality=94)


def contact_sheet(root, result, output):
    frames = [4240, 4318, 4319, 4320, 4321, 4322, 4326, 4336]
    sheet = Image.new("RGB", (900, 145*len(frames)), (18,18,18))
    draw = ImageDraw.Draw(sheet)
    refs = {r["frame"]: r for r in result["frames"]}
    for row, frame in enumerate(frames):
        if frame not in refs:
            draw.text((0,row*145), f"D{frame} MISSING", fill="white")
            continue
        with Image.open(Path(root)/refs[frame]["path"]) as im:
            im = im.convert("RGB")
            for col, x in enumerate((733,1387)):
                crop = im.crop((x-70,385,x+70,490)).resize((187,140))
                sheet.paste(crop,(col*200,row*145))
            sheet.paste(im.resize((480,201)).crop((0,0,480,140)), (410,row*145))
        draw.text((2,row*145+2),f"D{frame}",fill="white")
    sheet.save(output, quality=93)


def external_output(path, root=ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path.is_relative_to(root) or path.is_relative_to((root/"renders").resolve()):
        raise ValueError("measurement output must be external to repository and shared renders")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--mode", choices=("crowns","ridges","sun"), default="crowns")
    args = parser.parse_args(argv)
    started = time.monotonic()
    out = external_output(args.out, args.root)
    inv = inventory(args.root)
    (out/"inventory.json").write_text(json.dumps(inv, indent=2)+"\n")
    mode=args.mode
    if mode == "crowns":
        result=scan(args.root,"embers_D_crowns",4240,4336,CROWN_ROIS)
    elif mode == "ridges":
        result=scan(args.root,"falsedawn_twofires",4560,5039,RIDGE_ROIS)
    else:
        result=scan(args.root,"falsedawn_truedawn",7360,7839,SUN_ROI,SUN_THRESHOLDS,
                    statistic=solar_stats,chroma_rule="min(R,G,B)>=threshold; max(R,G,B)-min(R,G,B)<=30",min_pixels=1)
    result["visual_review"] = "pending; inspect "+mode+"_measurement_sheet.jpg"
    (out/(mode+"_measurements.json")).write_text(json.dumps(result, indent=2)+"\n")
    if mode == "crowns":
        contact_sheet(args.root, result, out/"crowns_measurement_sheet.jpg")
    elif mode == "ridges":
        ridge_sheet(args.root,result,out/"ridges_measurement_sheet.jpg")
    else:
        solar_sheet(args.root,result,out/"sun_measurement_sheet.jpg")
    hook = crown_hook(result) if mode == "crowns" else solar_hook(result) if mode == "sun" else None
    series = ridge_series(result) if mode == "ridges" else None
    overlay = dict(schema="long-dawn/d-effects-measurements/1", cut="D", fps=24, frames=9200,
                   input_sha256={p: hashlib.sha256((args.root/p).read_bytes()).hexdigest() for p in INPUTS},
                   hooks={"crown_beacons" if mode=="crowns" else "dawn_sunlight":hook} if hook else {},
                   series={"D.new.ridges.catches":series} if series else {})
    (out/(mode+"_overlay.json")).write_text(json.dumps(overlay, indent=2)+"\n")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = dict(seconds=time.monotonic()-started, peak_rss_bytes=int(rss if sys.platform == "darwin" else rss*1024),
                   scanned_frames=len(result["frames"]), decoded_errors=result["errors"], measured_frame=hook["frame"] if hook else None,
                   ridge_frames=[r["frame"] for r in series["events"]] if series else None,
                   analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scope="native delivered plates; no assembled D, listening or sound-arrival assertion")
    (out/(mode+"_scan_receipt.json")).write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
