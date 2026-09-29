"""THE LONG DAWN C5: the MEASURED event map of the delivered C5 picture (SOUND-SCORE-C, 29 Sep 2026).

    python measure_c5_events.py                         # every shot -> music/v3/events_C5_measured.json
    python measure_c5_events.py --shots map,cold --dry  # print a subset, write nothing
    python measure_c5_events.py --frames-root DIR       # default: $LD_FRAMES, else project renders/

Why this exists. barmap_C5.json was written before the picture existed: its frames inside the new shots are
musical estimates ("(est.)"), its holdout catch sits at 3760 and its wording has one leader and one first fire.
The nine original C5 shots use absolute source-frame numbers; nine added ranges use EDIT's EDL resolver
and compositor to map the reused sources into cut numbering. This script reads them one frame at a time (never a whole shot in memory), measures pixels, and
writes one table: per event its frame(s), the method, the region, the numbers that decided it, and a confidence.
It never edits barmap_C5.json or cues_C5.json; `disagreements` lists where they differ from the picture.

Basis labels. MEASURED = a pixel statistic decided the frame (the method field says which). A human-readable
check of stills is recorded separately in OBSERVED (filled from inspected crops; `--dry` prints what to inspect).
Nothing here watched continuous motion or heard anything.

Frame identity. Each shot's frame set is hashed (sha256 over "name sha256(file)" lines, sorted), so a later reader
can tell whether a re-render moved the picture under this table.
"""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
V3 = os.path.join(MUSIC, "v3")
OUT_JSON = os.path.join(V3, "events_C5_measured.json")
FPS = 24
W, H = 1920, 804

# shot key -> (delivered directory stem, first frame, last frame inclusive, the C5 section it plays in)
SHOTS = {
    "refusal": ("book_C5_refusal", 2080, 2319, "C10"),
    "trap": ("embers_C5_trap", 2320, 2639, "C11"),
    "reveal": ("runC_reveal_pair_v5", 2880, 3119, "C13"),
    "map": ("map_last_beacon_C", 3440, 3839, "C15"),
    "cold": ("embers_C5_cold", 3840, 3999, "C16"),
    "unfinished": ("embers_C5_unfinished", 4000, 4239, "C17"),
    "deep": ("book_C5_deep_abandoned", 4240, 4479, "C18"),
    "watch": ("runC_watch_v5", 4480, 4719, "C19"),
    "pen": ("book_C5_pen", 5440, 5679, "C22"),
}
from measure_c5_delivered import EXTRA_SHOTS
SHOTS.update(EXTRA_SHOTS)
# Keep imports from helper modules attached to this same module when run as a script.
if __name__ == "__main__":
    sys.modules["measure_c5_events"] = sys.modules[__name__]

RSS_ABORT = 1_500_000_000   # lane A: one sequential reader, at most 1.5 GB


# ------------------------------------------------------------------------------------------------ io + guards
def frames_root(arg=None):
    return os.path.expanduser(arg or os.environ.get("LD_FRAMES") or os.path.join(os.path.dirname(MUSIC), "renders"))


def fpath(root, stem, f):
    png = os.path.join(root, stem, f"f_{f:05d}.png")
    return png if os.path.exists(png) else os.path.join(root, stem, f"f_{f:05d}.jpg")


_N_READ = [0]


def _guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    if rss > RSS_ABORT:
        raise SystemExit(f"ABORT: peak RSS {rss / 2 ** 20:.0f} MiB > 1.5 GB")
    _N_READ[0] += 1
    if _N_READ[0] % 100 == 0 and sys.platform == "darwin":
        try:
            lvl = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True,
                                     text=True).stdout.strip() or 0)
        except (OSError, ValueError):
            lvl = 0
        if lvl >= 4:
            raise SystemExit(f"ABORT: memory pressure level {lvl}")


def read(root, stem, f, half=False):
    """float32 RGB of one delivered frame; half=True decodes at 960x402 (JPEG DCT scaling, exact halves)"""
    from PIL import Image
    im = Image.open(fpath(root, stem, f))
    if half:
        im.draft("RGB", (W // 2, H // 2))
    im = im.convert("RGB")
    if half and im.size != (W // 2, H // 2):
        im = im.resize((W // 2, H // 2), Image.BILINEAR)
    _guard()
    return np.asarray(im, np.float32)


def luma(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def blobs(mask, min_area=1):
    import cv2
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = int(st[i][4])
        if a >= min_area:
            x0, y0, w, h = (int(v) for v in st[i][:4])
            out.append(dict(cx=float(cen[i][0]), cy=float(cen[i][1]), area=a, x0=x0, y0=y0, x1=x0 + w, y1=y0 + h))
    return out


def frame_set_sha256(root, stem, a, b):
    h = hashlib.sha256()
    for f in range(a, b + 1):
        p = fpath(root, stem, f)
        h.update(f"{os.path.basename(p)} {hashlib.sha256(open(p, 'rb').read()).hexdigest()}\n".encode())
    return h.hexdigest()


def file_sha(root, stem, f):
    return hashlib.sha256(open(fpath(root, stem, f), "rb").read()).hexdigest()


def track(per_frame, radius, min_len=1, key=("cx", "cy"), wy=1.0):
    """nearest-neighbour tracks through [(frame, [blob, ...]), ...]: each blob joins the nearest live track
    within `radius` px (by its last position; vertical distance weighted by wy, so a flame whose crown rises
    with its growing tower keeps its track), else starts a track. -> [{frames: [...], blobs: [...]}, ...]"""
    tracks = []
    for f, bl in per_frame:
        taken = set()
        order = sorted(range(len(bl)), key=lambda i: -bl[i]["area"])
        for i in order:
            b = bl[i]
            best, bd = None, radius
            for k, t in enumerate(tracks):
                if k in taken or f - t["frames"][-1] > 12:
                    continue
                lb = t["blobs"][-1]
                d = float(np.hypot(b[key[0]] - lb[key[0]], wy * (b[key[1]] - lb[key[1]])))
                if d < bd:
                    best, bd = k, d
            if best is None:
                tracks.append(dict(frames=[f], blobs=[b]))
                taken.add(len(tracks) - 1)
            else:
                tracks[best]["frames"].append(f)
                tracks[best]["blobs"].append(b)
                taken.add(best)
    return [t for t in tracks if len(t["frames"]) >= min_len]


def series_of(t, a, b, field="area"):
    """a track's field as a dense array over [a, b] (0 where the track has no blob)"""
    s = np.zeros(b - a + 1)
    for f, bl in zip(t["frames"], t["blobs"]):
        s[f - a] = bl[field]
    return s


def ramp(s, a, lo_frac=0.1, hi_frac=0.9, base=None, top=None):
    """(first frame s passes lo_frac of the way from base to top, first frame it passes hi_frac, fastest frame,
    first frame past halfway); s is dense from frame a; a falling ramp works too (top < base)"""
    base = float(s[0] if base is None else base)
    top = float(s[-1] if top is None else top)
    span = top - base
    if abs(span) <= 1e-9:
        return None, None, None, None
    u = (np.asarray(s, float) - base) / span

    def cross(q):
        return int(np.argmax(u >= q)) if np.any(u >= q) else None
    d = np.diff(u)
    i_fast = int(np.argmax(d)) + 1 if len(d) and np.max(d) > 0 and np.any(u >= lo_frac) else None
    f = (lambda i: None if i is None else a + i)
    return f(cross(lo_frac)), f(cross(hi_frac)), f(i_fast), f(cross(0.5))


def win(s, a, f, k=3, nd=1):
    """evidence: {frame: value} for frames f-k..f+k of a dense series starting at a"""
    return {str(g): round(float(s[g - a]), nd) for g in range(f - k, f + k + 1) if 0 <= g - a < len(s)}


def ev(eid, shot, event, frames, method, region, evidence, confidence, note="", **kw):
    d = dict(id=eid, shot=shot, stem=SHOTS[shot][0], event=event, frames=frames, method=method, region=region,
             evidence=evidence, confidence=confidence, basis=["MEASURED"])
    if note:
        d["note"] = note
    d.update(kw)
    return d


# ------------------------------------------------------------------------------------------------ the shots
FLAME = "flame pixels: luma > 150 and R > 180 (the forge flames' yellow-white cores), 960x402 decode"


def _flame_mask(x):
    return (luma(x) > 150) & (x[..., 0] > 180)


def m_trap(root):
    """forge flames: the one forge that sinks and comes back, the others' surge, the two leaders, any strike"""
    stem, a, b, _ = SHOTS["trap"]
    per, tot, dY, prev = [], np.zeros(b - a + 1), np.zeros(b - a + 1), None
    for f in range(a, b + 1):
        x = read(root, stem, f, half=True)
        per.append((f, blobs(_flame_mask(x), 3)))
        Y = luma(x)
        if prev is not None:
            dY[f - a] = float(np.abs(Y[160:, :] - prev[160:, :]).mean())
        prev = Y
    # the Ring's own gold (two small blobs of ~70-100 px) sits in the flame mask: found on the first frame as the
    # blobs in the top third, and left out of every forge measure
    ring = [q for q in per[0][1] if q["cy"] < 140]
    rc = (float(np.mean([q["cx"] for q in ring])), float(np.mean([q["cy"] for q in ring]))) if ring else (478., 104.)

    def is_ring(q):
        return np.hypot(q["cx"] - rc[0], q["cy"] - rc[1]) < 22 and q["area"] < 160
    for f, bl in per:
        tot[f - a] = sum(q["area"] for q in bl if not is_ring(q))
    # flames tracked on their BASE (lowest row, centre column), vertical distance weighted 0.3: a crown rises with
    # its growing tower, and neighbouring towers stand side by side
    per_b = [(f, [dict(q, bx=q["cx"], by=float(q["y1"])) for q in bl if q["area"] >= 20 and not is_ring(q)])
             for f, bl in per]
    tr = track(per_b, radius=16, min_len=10, key=("bx", "by"), wy=0.3)
    events = []
    from measure_c5_analysis import control

    def ctrl(values, interval, threshold, direction=1):
        return control([dict(f=a+i, signal=float(v)) for i,v in enumerate(values)],
                       "signal", interval, threshold, direction, "fixed pre-event control; original detector threshold")

    # 1. the withdrawing forge: a full-size flame from the first frame that goes out for a stretch and returns
    low = None
    for t in (t for t in tr if t["frames"][0] == a):
        sl = series_of(t, a, b)
        base = float(np.median(sl[:60]))
        if base > 60 and t["frames"][-1] < a + 200:
            low = (t, sl, base)
            break
    if low:
        t, sl, base = low
        i_gone = int(np.argmax(sl[60:] < 0.1 * base)) + 60
        i_start = int(np.flatnonzero(sl[:i_gone] >= 0.9 * base)[-1]) + 1
        lb = t["blobs"][-1]
        events.append(ev("trap.low_forge_sinks", "trap", "one forge's flame sinks almost out (it tries to stop "
                         "alone)", dict(first=a + i_start, out=a + i_gone),
                         FLAME + "; its flame tracked on the base; first = first frame under 90% of its median area "
                         "over the shot's first 60 frames, out = first frame under 10% (the flame's core leaves the "
                         "mask; the renderer keeps a 6% flame)",
                         dict(x=round(lb["bx"] * 2), y=round(lb["by"] * 2), space="1920x804, flame base"),
                         dict(area_base_px960=round(base), area=win(sl, a, a + i_start, 2, 0) | win(sl, a, a + i_gone, 3, 0)),
                         "high"))
        events[-1]["diagnostic_controls"] = [ctrl(sl, (2320,2379), -0.9*base, -1)]
        events[-1]["diagnostic_controls"][0]["selection_reason"] = "90% alone also detects ordinary flicker; the full sink rule additionally requires a subsequent extinction crossing"
        events[-1]["negative_controls"] = [ctrl(sl, (2320,2379), -0.1*base, -1)]
        events[-1]["negative_controls"][0]["selection_reason"] = "no extinction crossing below 10% in the opening plateau; the 90% onset is assigned only before that crossing"
        back = [u for u in tr if u["frames"][0] > a + i_gone and abs(u["blobs"][0]["bx"] - lb["bx"]) < 20]
        back = max(back, key=lambda u: len(u["frames"])) if back else None
        if back:
            sb = series_of(back, a, b)
            fb = back["frames"][0]
            ref = float(np.median(sb[fb - a + 12:fb - a + 30]))
            i90 = int(np.argmax(sb[fb - a:] >= 0.9 * ref)) + fb - a
            events.append(ev("trap.low_forge_returns", "trap", "the low forge flares back up and races after the "
                             "others (its tower climbs)", dict(first=fb, full=a + i90),
                             FLAME + "; the longest new track within 20 px (x) of where the low flame went out; full "
                             "= first frame at >= 90% of its median area 12-30 frames after it reappears",
                             dict(x=round(back["blobs"][0]["bx"] * 2), y=round(back["blobs"][0]["by"] * 2),
                                  space="1920x804, flame base"),
                             dict(area_ref_px960=round(ref), area=win(sb, a, fb, 3, 0),
                                  base_y960_first_last=[round(back["blobs"][0]["by"]), round(back["blobs"][-1]["by"])]),
                             "high"))
            events[-1]["negative_controls"] = [ctrl(sb, (2500,2530), 20)]
    # 2. the others surge: the forge-flame total steps up; three steady single flames show it is simultaneous
    base_tot, hi_tot = float(np.median(tot[:80])), float(np.median(tot[190:226]))
    lo_f, hi_f, fast_f, half_f = ramp(tot[120:226], a + 120, base=base_tot, top=hi_tot)
    steady = sorted((t for t in tr if len(t["frames"]) == b - a + 1 and (low is None or t is not low[0])),
                    key=lambda t: -float(np.median(series_of(t, a, b)[:80])))[:3]
    each = []
    for t in steady:
        st = series_of(t, a, b)
        l0, h0, fa, _ = ramp(st[120:226], a + 120, base=float(np.median(st[:80])), top=float(np.median(st[190:226])))
        each.append(dict(x=round(t["blobs"][0]["bx"] * 2), first=l0, fastest=fa, full=h0))
    events.append(ev("trap.forges_surge", "trap", "at once the other forges surge up (every flame taller)",
                     dict(first=lo_f, fastest=fast_f, half=half_f, full=hi_f),
                     "sum of " + FLAME + " (Ring blobs excluded) per frame; first/full = 10%/90% of the way from its "
                     "median over frames 0-79 to its median over 190-225; fastest = largest one-frame rise; the same "
                     "ramp on the three largest steady single flames (evidence.each) shows the surge is simultaneous",
                     dict(whole="forge band", space="1920x804"),
                     dict(total_base_px960=round(base_tot), total_after_px960=round(hi_tot),
                          total=win(tot, a, fast_f, 4, 0), each=each), "high"))
    events[-1]["negative_controls"] = [ctrl(tot, (2320,2399), base_tot+.1*(hi_tot-base_tot))]
    # 3. the two leaders: the tallest flame (its top row) either side of the Ring, per frame
    for k, (x0, x1, side) in enumerate(((rc[0] - 110, rc[0] - 8, "left"), (rc[0] + 8, rc[0] + 110, "right"))):
        topy, area = np.zeros(b - a + 1), np.zeros(b - a + 1)
        for f, bl in per:
            c = [q for q in bl if x0 <= q["cx"] < x1 and q["area"] >= 20 and not is_ring(q)]
            if c:
                q = min(c, key=lambda q: q["y0"])
                topy[f - a], area[f - a] = q["y0"], q["area"]
        y_before, y_end = float(np.median(topy[200:240])), float(np.median(topy[-8:]))
        lo, hi, fast, half = ramp(-topy[230:], a + 230, base=-y_before, top=-y_end)
        events.append(ev(f"trap.leader_{side}_pulls_ahead", "trap", f"the {side} of the two front-runners (same form, "
                         "uncoded) climbs toward the Ring", dict(first=lo, half=half, full=hi),
                         FLAME + f"; per frame the flame with the highest top within x {x0 * 2:.0f}-{x1 * 2:.0f} "
                         "(1920 px) of the Ring; first/full = 10%/90% of its top's climb from the median over frames "
                         "200-239 to the last 8 frames' median; half = its 50% crossing (a one-frame 'fastest' is "
                         "not reported: the tallest flame can switch to a neighbour's blob for a frame)",
                         dict(x0=round(x0 * 2), x1=round(x1 * 2), space="1920x804, beside the Ring"),
                         dict(top_y960_before=round(y_before), top_y960_end=round(y_end),
                              top_y960=win(topy, a, half, 3, 0), area_px960_end=round(float(np.median(area[-8:])))),
                         "high"))
        events[-1]["negative_controls"] = [ctrl(topy, (2520,2559), -y_before+.1*(y_before-y_end), -1)]
    # 4. hammer strikes: a strike would be an impulse (a flash or spark burst at a crown): a one-frame spike in the
    # frame difference over the forge band against its running median
    from scipy.ndimage import median_filter
    med = median_filter(dY, size=15, mode="nearest")
    spikes = [int(a + i) for i in range(1, len(dY)) if dY[i] > 3.0 * med[i] + 0.5]
    events.append(ev("trap.hammer_strikes", "trap", "visible hammer strikes (flash or spark impulses at the crowns)",
                     dict(found=spikes), "mean |luma difference| to the previous frame over the forge band (rows "
                     "160-401 of 960x402); a strike would be a one-frame spike > 3x its 15-frame running median + 0.5",
                     dict(rows="160-401 of 402", space="960x402"),
                     dict(dY_median=round(float(np.median(dY[1:])), 3), dY_max=round(float(dY[1:].max()), 3),
                          dY_max_frame=int(a + 1 + np.argmax(dY[1:])), dY_around_max=win(dY, a, int(a + 1 + np.argmax(dY[1:])), 3, 2)),
                     "high" if not spikes else "medium",
                     note=("no impulse: the hammers are not drawn; they are SOUND's, placed musically (not on a "
                           "visible strike)") if not spikes else "inspect these frames"))
    events[-1]["negative_controls"] = [ctrl(dY-3*med, (2321,2399), .5)]
    return events, dict(total=tot, dY=dY)


def m_cold(root):
    stem, a, b, _ = SHOTS["cold"]
    tot = np.zeros(b - a + 1)
    for f in range(a, b + 1):
        x = read(root, stem, f, half=True)
        m = _flame_mask(x)
        tot[f - a] = sum(q["area"] for q in blobs(m, 3))
    d = np.diff(tot)
    i = int(np.argmin(d)) + 1
    after = float(np.median(tot[i:i + 20]))
    before = float(np.median(tot[max(0, i - 8):i]))
    return [ev("cold.forges_off", "cold", "every forge goes dark at the same instant", dict(frame=a + i),
               "sum of " + FLAME + " per frame (the Ring's gold included, so the floor is the Ring); the frame of the "
               "largest one-frame drop", dict(whole="frame", space="960x402"),
               dict(flame_px960=win(tot, a, a + i, 3, 0), median_before=round(before), median_after=round(after),
                    drop_ratio=round(after / max(before, 1), 3)), "high")], dict(total=tot)


def m_map(root):
    """the beacons are small bright flames on the parchment: local-contrast blobs, tracked"""
    import cv2
    stem, a, b, _ = SHOTS["map"]
    per = []
    for f in range(a, b + 1):
        x = read(root, stem, f)
        Y = luma(x)
        bg = cv2.GaussianBlur(Y, (0, 0), 12)
        m = ((Y - bg) > 40) & (x[..., 0] > x[..., 1]) & (x[..., 1] > x[..., 2])
        per.append((f, blobs(m, 4)))
    tr = [t for t in track(per, radius=40, min_len=20)]
    beacons = []
    for t in tr:
        s = series_of(t, a, b)
        first = t["frames"][0]
        ref = float(np.median(s[first - a + 10:first - a + 40])) if first - a + 40 <= len(s) else float(np.median(s[first - a:]))
        i90 = int(np.argmax(s[first - a:] >= 0.9 * ref)) + first - a
        d = np.diff(s[first - a:first - a + 12])
        fast = first + int(np.argmax(d)) + 1 if len(d) else first
        lit_through = t["frames"][-1]
        beacons.append(dict(first=first, fastest=fast, full=a + i90, ref=ref, s=s, t=t, last=lit_through))
    beacons.sort(key=lambda q: q["first"])
    lit = np.zeros(b - a + 1, int)
    for q in beacons:
        lit[q["first"] - a:q["last"] - a + 1] += 1
    events = []
    MAPM = ("bright warm local-contrast blobs (luma minus its 12 px Gaussian > 40, R > G > B), native 1920x804, "
            "tracked (40 px, >= 20 frames); first = first frame with a blob of >= 4 px, fastest = largest one-frame "
            "area rise in its first 12 frames, full = first frame at >= 90% of its median area 10-40 frames after "
            "it appears")
    for k, q in enumerate(beacons):
        pre = q["first"] <= a
        bl = q["t"]["blobs"][min(len(q["t"]["blobs"]) - 1, 12)]
        fr = dict(lit_from=a, lit_through=q["last"]) if pre else dict(first=q["first"], fastest=q["fastest"],
                                                                          full=q["full"], lit_through=q["last"])
        events.append(ev(f"map.beacon_{k + 1}", "map", ("beacon 1 is already burning when the map opens" if pre else
                          f"beacon {k + 1} of {len(beacons)} catches"), fr, MAPM,
                         dict(x=round(bl["cx"]), y=round(bl["cy"]), space="1920x804, 12 frames after it appears"),
                         dict(area_ref_px=round(q["ref"]), area=win(q["s"], a, max(a + 3, q["first"]), 4, 0)),
                         "high"))
    n = len(beacons)
    last, prev = beacons[-1], beacons[-2]
    events.append(ev("map.one_dark", "map", "exactly one kingdom is still dark: the long pause before the last "
                     "beacon", dict(first=prev["full"], last=last["first"] - 1, seventh_first_visible=prev["first"]),
                     "from the 2nd-to-last beacon's full frame to the frame before the last beacon's first visible "
                     "frame (" + MAPM + ")", dict(whole="frame", space="1920x804"),
                     dict(beacons=n, lit_count=win(lit.astype(float), a, prev["first"], 2, 0) |
                          win(lit.astype(float), a, last["first"], 2, 0)), "high",
                     note="the renderer states one-dark C3724-3783 (its 7th catch ramp COMPLETES at 3724); by "
                          "visibility 7 are lit from 3718"))
    events.append(ev("map.last_catch", "map", "the last kingdom's beacon catches (the holdout joins)",
                     dict(first=last["first"], fastest=last["fastest"], full=last["full"]),
                     "as map.beacon_k, for the last beacon to appear",
                     dict(x=round(last["t"]["blobs"][min(12, len(last["t"]["blobs"]) - 1)]["cx"]),
                          y=round(last["t"]["blobs"][min(12, len(last["t"]["blobs"]) - 1)]["cy"]), space="1920x804"),
                     dict(area=win(last["s"], a, last["first"], 4, 0), area_ref_px=round(last["ref"])), "high",
                     note="a faint glow is already visible on the first frame before (OBSERVED); the renderer "
                          "states the catch as C3784-3791"))
    events.append(ev("map.all_lit", "map", "every beacon burning (to the end of the shot)",
                     dict(first=last["full"], last=b, all_visible_from=last["first"]),
                     "from the last beacon's full frame; every tracked beacon stays lit to the shot's last frame",
                     dict(whole="frame", space="1920x804"), dict(beacons=n, lit_at_end=int(lit[-1])), "high"))
    # Track births are the catch detector; already-burning tracks are not new catches.
    from measure_c5_analysis import control
    births = [dict(f=f, births=sum(q["first"] == f for q in beacons)) for f in range(a,b+1)]
    c = control(births, "births", (3792,3839), 1, why="all eight established fires, no new catch")
    for event in events:
        event["negative_controls"] = [c]
    return events, dict(lit=lit)


def m_refusal(root):
    """the ink drawing: where and when ink arrives (no pen and no drawing hand are in the picture)"""
    stem, a, b, _ = SHOTS["refusal"]
    L = b - a + 1
    n, cx, cy = np.zeros(L), np.full(L, -1.0), np.full(L, -1.0)
    bbox = [None] * L
    prev = None
    for f in range(a, b + 1):
        Y = luma(read(root, stem, f, half=True))
        if prev is not None:
            dk = (Y - prev) < -18
            n[f - a] = dk.sum()
            if n[f - a]:
                ys, xs = np.nonzero(dk)
                cx[f - a], cy[f - a] = xs.mean() * 2, ys.mean() * 2
                bbox[f - a] = (int(xs.min() * 2), int(ys.min() * 2), int(xs.max() * 2), int(ys.max() * 2))
        prev = Y
    INK = ("ink arrival: pixels whose luma DROPS by > 18 from the previous frame (960x402, frame to frame, so a slow "
           "drift never accumulates); a stroke frame has >= 10 such pixels; positions in 1920x804")
    stroke = n >= 10
    # segments: runs of stroke frames separated by >= 2 frames without a stroke
    segs, cur, gap = [], [], 0
    for i in range(L):
        if stroke[i]:
            if cur and gap >= 2:
                segs.append(cur)
                cur = []
            cur.append(i)
            gap = 0
        elif cur:
            gap += 1
    if cur:
        segs.append(cur)

    def seg_info(sg):
        w = n[sg]
        return dict(first=int(a + sg[0]), last=int(a + sg[-1]), frames=len(sg), ink_px960=int(w.sum()),
                    centroid=[int(np.average(cx[sg], weights=w)), int(np.average(cy[sg], weights=w))])

    def union(idx):
        bx = [bbox[i] for i in idx if bbox[i]]
        return dict(x0=min(q[0] for q in bx), y0=min(q[1] for q in bx), x1=max(q[2] for q in bx),
                    y1=max(q[3] for q in bx), space="1920x804")
    info = [seg_info(sg) for sg in segs]
    first, last = int(a + segs[0][0]), int(a + segs[-1][-1])
    fig_k = next(k for k, q in enumerate(info) if q["centroid"][0] > 1000)
    ring_k = fig_k - 1                                   # the segment right before the figure: the Ring on the palm
    ring_sg, fig_sg = segs[ring_k], segs[fig_k]
    pause = (int(a + ring_sg[-1] + 1), int(a + fig_sg[0] - 1))
    # the raised hand: after the figure begins, the first run of stroke frames centred up and left of the figure's
    # body (x < 1190, y < 300: the hand stands beside the hood)
    after = [i for i in range(fig_sg[0] + 21, L) if stroke[i] and cx[i] < 1190 and cy[i] < 300]
    hand = [after[0]] if after else []
    for i in after[1:]:
        if i == hand[-1] + 1:
            hand.append(i)
        else:
            break
    ev_ = [ev("refusal.ink_begins", "refusal", "the first stroke: the offering sleeve and arm begin, left of the page",
              dict(first=first), INK, dict(x=int(cx[first - a]), y=int(cy[first - a]), space="1920x804, centroid"),
              dict(pixels=win(n, a, first, 2, 0)), "high"),
           ev("refusal.offering_hand", "refusal", "the offering arm is hatched and its open hand drawn",
              dict(first=info[ring_k - 1]["first"], last=info[ring_k - 1]["last"]), INK + "; the stroke segment "
              "before the Ring's", union(segs[ring_k - 1]),
              dict(segment=info[ring_k - 1], pixels=win(n, a, info[ring_k - 1]["first"], 2, 0)), "high"),
           ev("refusal.ring_on_palm", "refusal", "the Ring is drawn on the open palm (the old story's offer)",
              dict(first=info[ring_k]["first"], last=info[ring_k]["last"]), INK + "; the stroke segment just "
              "before the pause and the figure (segments = stroke runs split by >= 2 empty frames)", union(ring_sg),
              dict(segment=info[ring_k], pixels=win(n, a, info[ring_k]["first"], 2, 0)), "high"),
           ev("refusal.pause_before_figure", "refusal", "the pen pauses: no stroke between the Ring and the figure",
              dict(first=pause[0], last=pause[1]), INK + "; the frames between the two segments",
              dict(whole="page", space="1920x804"), dict(pixels=win(n, a, pause[0], 2, 0)), "high"),
           ev("refusal.figure_begins", "refusal", "the hooded figure begins, right of the page, already turned away",
              dict(first=info[fig_k]["first"]), INK + "; the first segment centred right of x 1000",
              union(fig_sg[:12]), dict(segment=info[fig_k], pixels=win(n, a, info[fig_k]["first"], 2, 0)), "high"),
           ev("refusal.raised_hand", "refusal", "the figure's hand is raised against it (the refusal gesture)",
              dict(first=int(a + hand[0]), last=int(a + hand[-1])) if hand else dict(), INK + "; more than 20 "
              "frames after the figure begins, the first run of stroke frames centred at x < 1190, y < 300 (beside "
              "the hood)", union(hand) if hand else {},
              dict(pixels=win(n, a, int(a + hand[0]), 2, 0) if hand else {},
                   centroids=[[int(cx[i]), int(cy[i])] for i in hand]), "high"),
           ev("refusal.ink_ends", "refusal", "the last stroke (the cloak's hatching); the drawing then holds still to "
              "the end of the shot", dict(last=last, still_from=last + 1, still_to=b), INK,
              dict(whole="page", space="1920x804"),
              dict(pixels=win(n, a, last, 2, 0), max_pixels_after=int(n[last - a + 1:].max()) if last < b else 0,
                   segments=info), "high")]
    from measure_c5_analysis import control
    c = control([dict(f=a+i, ink=float(v)) for i,v in enumerate(n)], "ink", (2251,2319), 10,
                why="completed drawing, no more ink strokes")
    for event in ev_:
        event["negative_controls"] = [c]
    return ev_, dict(n=n)


def m_reveal(root):
    """both first fires: the catch glow (warm wash) around each summit, and the near flame's own ink outline"""
    stem, a, b, _ = SHOTS["reveal"]
    near, far = (950, 640), (1390, 390)                # 1920x804: the two warm blobs of C2880 (see the probe notes)
    wn, wf, dn = np.zeros(b - a + 1), np.zeros(b - a + 1), np.zeros(b - a + 1)
    for f in range(a, b + 1):
        x = read(root, stem, f)
        R, B = x[..., 0], x[..., 2]
        warm = (R - B > 80) & (R > 170)
        Y = luma(x)
        for (cx, cy), w in ((near, wn), (far, wf)):
            w[f - a] = warm[cy - 110:cy + 60, cx - 110:cx + 110].sum()
        dn[f - a] = (Y[near[1] - 110:near[1] + 60, near[0] - 110:near[0] + 110] < 120).sum()
    WARM = ("warm wash: R - B > 80 and R > 170 (1920x804) counted in a 220x170 px window around each fire; the "
            "flame glyph's ink outline: luma < 120 in the near window")
    out = []
    for nm, w in (("near", wn), ("far", wf)):
        settled = float(np.median(w[-40:]))
        half = int(np.argmax(w <= (w[0] + settled) / 2))
        out.append(dict(name=nm, peak_frame=int(a + np.argmax(w)), peak=int(w.max()), settled=int(settled),
                        half_frame=int(a + half), monotone_first_12=bool(np.all(np.diff(w[:12]) <= 0))))
    g_lo, g_hi, _, g_half = ramp(dn[:40], a, base=float(dn[0]), top=float(np.median(dn[8:30])))
    ev_ = [ev("reveal.both_fires_ignite", "reveal", "both first beacons ignite at the same moment (hers near, the "
              "rival's on the far peak)", dict(frame=a, near_glow_peak=out[0]["peak_frame"],
                                               far_glow_peak=out[1]["peak_frame"]),
              WARM + "; the ignition frame = the frame of each window's largest warm wash (the catch glow flashes, "
              "then settles)", dict(near=dict(x=near[0], y=near[1]), far=dict(x=far[0], y=far[1]), space="1920x804"),
              dict(near=out[0], far=out[1], near_window=win(wn, a, a, 3, 0) | win(wn, a, a + 8, 1, 0),
                   far_window=win(wf, a, a, 3, 0) | win(wf, a, a + 8, 1, 0)), "high",
              note="2880 is the shot's FIRST frame: its glow peak there is the latest the ignition can be; whether "
                   "it began inside FLINT (2640-2879, not on this Mac) cannot be measured here"),
           ev("reveal.near_flame_grows", "reveal", "her flame's drawn glyph grows to full size", dict(first=g_lo,
              half=g_half, full=g_hi), WARM + "; ink-outline pixel count in the near window, 10%/50%/90% of the way "
              "from frame 2880 to its median over 2888-2909", dict(x=near[0], y=near[1], space="1920x804"),
              dict(ink_px=win(dn, a, a + 3, 3, 0)), "medium",
              note="the far flame is ~30 px tall: its outline is too small for this count; OBSERVED crops show it "
                   "growing over the same frames")]
    from measure_c5_analysis import control
    ev_[0]["negative_controls"] = [control([dict(f=a+i, wash=float(v)) for i,v in enumerate(w)],
        "wash", (3000,3119), .9*float(w.max()), why="settled already-burning fire; fixed 90% peak wash")
        for w in (wn,wf)]
    ev_[0]["note"] = "First-frame glow peak is left-censored: 2880 is the latest possible ignition, not proof of a within-shot onset. The control tests recurrence of that peak wash."
    ev_[1]["negative_controls"] = [control([dict(f=a+i, ink=float(v)) for i,v in enumerate(dn)],
        "ink", (2880,2880), float(dn[0])+.1*(float(np.median(dn[8:30]))-float(dn[0])),
        why="only one available pre-growth frame; limited negative-control coverage")]
    return ev_, dict(wn=wn, wf=wf, dn=dn)


def m_unfinished(root):
    stem, a, b, _ = SHOTS["unfinished"]
    warmth, sky = np.zeros(b - a + 1), np.zeros(b - a + 1)
    for f in range(a, b + 1):
        x = read(root, stem, f, half=True)
        R, B = x[..., 0], x[..., 2]
        Y = luma(x)
        box = (slice(60, 180), slice(420, 540))           # the Ring, 960x402 (it hangs still above the towers)
        m = Y[box] > 110
        warmth[f - a] = float((R[box] - B[box])[m].mean()) if m.any() else 0.0
        sky[f - a] = float((R - B)[:200].mean())
    r_lo, r_hi, _, r_half = ramp(warmth, a, base=float(warmth[0]), top=float(np.median(warmth[-20:])))
    s_lo, s_hi, _, s_half = ramp(sky, a, base=float(sky[0]), top=float(np.median(sky[-20:])))
    sh = [file_sha(root, stem, f) for f in range(b - 40, b + 1)]
    i = len(sh) - 1
    while i > 0 and sh[i - 1] == sh[-1]:
        i -= 1
    hold = b - 40 + i
    return [ev("unfinished.ring_drains", "unfinished", "the Ring's gold drains to dull grey",
               dict(first=r_lo, half=r_half, full=r_hi),
               "mean R - B of the Ring's bright pixels (luma > 110 in x 840-1080, y 120-360 of 1920x804); 10%/50%/"
               "90% of the way from frame 4000 to the last 20 frames' median", dict(x0=840, y0=120, x1=1080, y1=360,
                                                                                   space="1920x804"),
               dict(ring_R_minus_B=win(warmth, a, a, 0, 1) | win(warmth, a, r_half, 0, 1) | win(warmth, a, b, 0, 1)),
               "high"),
            ev("unfinished.storm_thins", "unfinished", "the storm's red thins away to nothing",
               dict(first=s_lo, half=s_half, full=s_hi), "mean R - B over the top 400 rows (1920x804); 10%/50%/90% "
               "of the way from frame 4000 to the last 20 frames' median", dict(rows="0-399", space="1920x804"),
               dict(sky_R_minus_B=win(sky, a, a, 0, 1) | win(sky, a, s_half, 0, 1) | win(sky, a, b, 0, 1)), "high"),
            ev("unfinished.still_hold", "unfinished", "the picture holds byte-identical to the end (intentional)",
               dict(first=hold, last=b), "sha256 of each delivered file, frames 4199-4239: the first frame of the "
               "final run of identical files", dict(whole="frame"), dict(identical_frames=b - hold + 1), "high")], \
        dict(warmth=warmth, sky=sky)


def _static(root, key, what):
    """a shot with no discrete event: per-frame mean |luma difference|, its largest frame against its median"""
    stem, a, b, _ = SHOTS[key]
    d, prev = np.zeros(b - a + 1), None
    for f in range(a, b + 1):
        Y = luma(read(root, stem, f, half=True))
        if prev is not None:
            d[f - a] = float(np.abs(Y - prev).mean())
        prev = Y
    q = d[1:]
    med = float(np.median(q))
    i = int(np.argmax(q)) + 1
    spikes = [int(a + 1 + j) for j in range(len(q)) if q[j] > 3 * med + 0.5]
    return [ev(f"{key}.no_discrete_event", key, what, dict(spikes=spikes),
               "mean |luma difference| to the previous frame over the whole frame (960x402); an event would be a "
               "frame > 3x the shot's median + 0.5", dict(whole="frame", space="960x402"),
               dict(dY_median=round(med, 3), dY_max=round(float(q.max()), 3), dY_max_frame=a + i), "high")], dict(d=d)


def m_deep(root):
    return _static(root, "deep", "the abandoned mine page holds still (the vein, the ladders and the lantern do not "
                                 "move; only a slow camera drift and paper grain change): no discrete sync event")


def m_pen(root):
    return _static(root, "pen", "the resting dip pen on the blank pages holds still (no hand, no page turn; only a "
                                "slow drift): no discrete sync event")


def m_watch(root):
    """the watch: every beacon lit; the camera drifts, so beacons enter at the frame edge already burning"""
    stem, a, b, _ = SHOTS["watch"]
    per = []
    for f in range(a, b + 1):
        x = read(root, stem, f, half=True)
        R, B = x[..., 0], x[..., 2]
        per.append((f, blobs((R - B > 80) & (R > 170), 30)))
    tr = track(per, radius=25, min_len=10)
    entering, inframe, cv, dx = [], [], [], []
    for t in tr:
        s = series_of(t, a, b)
        med = float(np.median(s[s > 0]))
        fb = t["blobs"][0]
        at_edge = fb["x1"] >= 955 or fb["x0"] <= 5 or t["frames"][0] == a
        grows = s[t["frames"][0] - a] < 0.3 * med and len(t["frames"]) > 20
        (inframe if (grows and not at_edge) else entering).append(
            dict(first=t["frames"][0], x=round(fb["cx"] * 2), y=round(fb["cy"] * 2), area0=int(fb["area"]),
                 median_area=round(med)))
        if len(t["frames"]) >= 60:
            cv.append(float(np.std(s[s > 0]) / med))
            dx.append(float(np.median(np.diff([q["cx"] for q in t["blobs"]]))) * 2)
    return [ev("watch.beacons_burn_on", "watch", "every beacon keeps burning; none catches or goes out in shot "
               "(as the camera drifts, beacons enter at the frame edge or come out from behind a ridge, already lit)",
               dict(appear_in_frame=[q["first"] for q in inframe]),
               "warm blobs (R - B > 80 and R > 170, >= 30 px at 960x402) tracked (25 px, >= 10 frames); a track "
               "that starts inside the frame at < 30% of its median area is listed in appear_in_frame: an ignition "
               "OR a flame uncovered by the foreground (the pixels cannot tell which; see OBSERVED)",
               dict(whole="frame", space="960x402"),
               dict(tracks=len(tr), entering=len(entering), appear_in_frame=inframe,
                    flicker_cv_median=round(float(np.median(cv)), 3) if cv else None,
                    camera_dx_px_per_frame_1920=round(float(np.median(dx)), 2) if dx else None), "high",
               note="flicker_cv = the flame area's standard deviation / median, per long track: the fires flicker "
                    "continuously (a bed), with no discrete hit")], dict()


# ------------------------------------------------------------------------------------------------ OBSERVED
# What a person (Claude, SOUND-SCORE-C, 29 Sep) saw on native-resolution crops of the delivered stills, keyed by event.
# Stills only: nobody on this Mac watched the shots in motion. Each note is tied to the frame set it was made on
# (the first 12 hex of its frame_set_sha256, computed on the night); if the shot is re-rendered the note is
# dropped from the table as stale. (On 29 Sep all 2,320 measured files matched their delivery receipts' sha256.)
OBSERVED = {
    "refusal": ("3a29c217e783", {
        "refusal.ink_begins": ("2088, 2089, 2095, 2116", "2088 is a blank page; 2089 the first short stroke (the "
                               "sleeve); 2095 the sleeve's outline; 2116 the sleeve hatched. No pen or drawing hand "
                               "is visible: the lines appear on the page by themselves"),
        "refusal.offering_hand": ("2116, 2134", "2116 the sleeve hatched; 2134 the open hand complete, empty"),
        "refusal.ring_on_palm": ("2134, 2138, 2150, 2155", "2134 the open palm complete, empty; 2138 a first small "
                                 "mark on it; 2150 the Ring complete on the palm; 2155 unchanged"),
        "refusal.figure_begins": ("2156, 2168", "2156 a first stroke at the right; 2168 the hooded figure's "
                                  "outline, turned away"),
        "refusal.raised_hand": ("2194, 2195, 2201", "2194-2195 the hood's shadow filled, no raised hand; 2201 the "
                                "hand raised against the Ring"),
        "refusal.pause_before_figure": ("2150, 2155", "the same drawing: hand and Ring, no figure yet"),
        "refusal.ink_ends": ("2249, 2250, 2319", "2249 and 2250 the cloak hatched; 2319 the same drawing"),
    }),
    "trap": ("37cddbda0bb0", {
        "trap.low_forge_sinks": ("2418, 2423, 2425, 2427, 2428, 2432, 2440", "the flame on one crown (left of "
                                 "centre, 1920x804 x~676) shrinks from 2418 through 2427, is a sliver at 2428 and "
                                 "all but gone by 2432; 2440 its crown is dark but for its ember-lit windows"),
        "trap.low_forge_returns": ("2540, 2545, 2546, 2548, 2557", "2540 dark crown; 2545 a small flame; 2548 "
                                   "larger; 2557 a tall flame on a visibly higher crown"),
        "trap.forges_surge": ("2478, 2482, 2484, 2486, 2488, 2492", "every other flame grows taller together from "
                              "2482-2484 to 2492"),
        "trap.leader_left_pulls_ahead": ("2560, 2580, 2590, 2600, 2610, 2620, 2626, 2639", "two towers either side "
                                         "of the Ring climb together from ~2590; by 2620 their flames reach the "
                                         "Ring's height and by 2626-2639 stand beside it; the same form, no "
                                         "marking tells them apart"),
        "trap.leader_right_pulls_ahead": ("as trap.leader_left_pulls_ahead", "the right-hand one of the pair; it "
                                          "climbs with the left one"),
        "trap.hammer_strikes": ("overview 2320-2620 every 20; the crops above", "no hammer, anvil flash or spark "
                                "burst is drawn: sparks drift continuously; nothing strikes"),
    }),
    "reveal": ("14da848fad57", {
        "reveal.both_fires_ignite": ("2880, 2881, 2882, 2884, 2888, 2896, 2920, 3000", "on 2880 both summits carry "
                                     "a small drawn flame inside a broad warm glow; the flames are larger on 2881 "
                                     "and 2882 and full by 2884; the glow fades over the next seconds; the far "
                                     "fire is a small gold mark on its peak"),
        "reveal.near_flame_grows": ("2880-2888", "as above"),
    }),
    "map": ("44fe23611860", {
        "map.beacon_1": ("overview 3440-3815 every 25", "a flame already burns on the range at 3440"),
        **{f"map.beacon_{k}": ("first - 1, first, full", "on the frame before `first` a faint glow is already on "
                               "the unlit ink stack; on `first` the flame's outline begins; on `full` the drawn "
                               "flame stands in its glow") for k in (2, 3, 4, 5, 6)},
        "map.beacon_8": ("as map.last_catch", "as map.last_catch"),
        "map.all_lit": ("3792 and overview 3790-3815", "eight flames burn"),
        "map.beacon_7": ("3716, 3718, 3720, 3722, 3724, 3783", "3716 only the unlit ink stack; 3718 a small "
                         "flame; 3722 full with its glow; steady to 3783"),
        "map.last_catch": ("3716, 3783, 3785, 3786, 3788, 3789, 3791, 3792", "the stack is unlit through 3783; "
                           "3785 a faint glow; 3786 a small flame; 3788 the flame's outline drawn; 3791-3792 full"),
        "map.one_dark": ("overview 3440-3815 every 25, and the crops above", "the parchment greys toward night "
                         "through the shot; seven flames burn and one site stays an unlit stack until 3785"),
    }),
    "cold": ("cf73c14d3185", {
        "cold.forges_off": ("3846, 3847, 3848, 3849", "3846-3847 every forge burning; 3848 every flame and window "
                            "out at once, the Ring still gold over dark towers; 3849 the same"),
    }),
    "unfinished": ("8a72077f1b63", {
        "unfinished.ring_drains": ("4000, 4040, 4072, 4100, 4135, 4184, 4216, 4239", "4000 gold with falling gold "
                                   "beads; 4072 paler; 4100 pale gold-grey; 4135 grey; grey to the end"),
        "unfinished.storm_thins": ("the same", "the red storm behind the Ring fades from ~4100; by 4184 only a "
                                   "trace; 4216-4239 dark"),
        "unfinished.still_hold": ("4216, 4239", "no visible difference: a grey Ring over dark towers"),
    }),
    "deep": ("d0353d9b8315", {
        "deep.no_discrete_event": ("4240, 4360, 4479", "the ink page of the mine: arched galleries, ladders, the "
                                   "gold vein, a small roofed lantern-like mark at the top left; identical content "
                                   "in all three"),
    }),
    "watch": ("e8886465b38f", {
        "watch.beacons_burn_on": ("4600-4630 and 4660-4690 (the two appear_in_frame tracks)", "both are flames "
                                  "coming out from behind a ridge as the camera drifts, not new ignitions; the "
                                  "renderer lights every beacon before the shot (shots/run/watch_c_v5.py "
                                  "ALREADY_LIT)"),
    }),
    "pen": ("67fe59598bd1", {
        "pen.no_discrete_event": ("5440, 5560, 5679", "a dip pen resting across the blank pages in the gutter's "
                                  "shadow; no hand, no page turn; identical content"),
    }),
}

# The old maps' frames against the delivered picture. barmap_C5 / cues_C5 are left untouched (pass 1 renders from
# them); these rows say where they disagree. (barmap id, its frame, measured event, which frame of it, what differs)
BARMAP_VS_PICTURE = [
    ("old_story", "refusal.ring_on_palm", "first", "the Ring is drawn on the palm 2138-2150; the open hand before it "
     "2114-2135 (est. 2120)"),
    ("turns_away", "refusal.raised_hand", "first", "the figure does not turn (it is drawn already turned away from "
     "2156); the refusal gesture, the raised hand, is drawn 2195-2202"),
    ("trap", None, None, "the barmap has 'hammers ringing': no hammer strike is drawn anywhere in the Trap (see "
     "trap.hammer_strikes); hammers are a sound design choice, not a sync point"),
    ("low_fire", "trap.low_forge_sinks", "first", "the one forge sinks 2423-2428 (flame gone from the mask; "
     "shrinking visibly from ~2420)"),
    ("surge", "trap.forges_surge", "first", "the others surge 2483-2490, 43 frames after the barmap's estimate"),
    ("flares_back", "trap.low_forge_returns", "first", "the low forge comes back 2546-2557, 66 frames later"),
    ("pulls_ahead", "trap.leader_left_pulls_ahead", "first", "TWO front-runners (not one tower) climb together "
     "2588-2626"),
    ("flint_black", None, None, "not measurable here: FLINT (2640-2879) is not on this Mac; the Trap's last frame "
     "2639 is not black"),
    ("catch", None, None, "not measurable here: FLINT is not on this Mac"),
    ("reveal", "reveal.both_fires_ignite", "frame", "the frame agrees (2880); the barmap's wording has ONE fire "
     "('her fire, one point of light'): the picture has two, igniting together"),
    ("promise", None, None, "a caption time (EDIT's), not a picture event"),
    ("run", None, None, "not measurable here: the BEACON RUN (runC_scroll, 3120-3439) and its beacon_1..7 are not "
     "on this Mac"),
    ("one_dark", "map.one_dark", "first", "one kingdom is dark only from the 7th beacon's catch (3718 visible, 3722 "
     "full; the renderer states 3724) to 3785, not from 3600"),
    ("hammer_alone", None, None, "no hammer is drawn on the map: a lone hammer under the pause is SOUND's choice "
     "inside map.one_dark"),
    ("last_beacon", "map.last_catch", "first", "the holdout catches 3786-3791 (a faint glow on 3785; the renderer "
     "states 3784-3791), not 3760: 26 frames (1.08 s) later"),
    ("forges_cold", "cold.forges_off", "frame", "agrees: every forge goes dark on 3848, in one frame"),
    ("smoke", None, None, "no discrete smoke event: smoke rises continuously from the shutdown"),
    ("storm_gone", "unfinished.storm_thins", "full", "the storm thins 4087-4181 (half 4138); the Ring's gold drains "
     "4025-4135 (half 4072); the picture then holds byte-identical 4217-4239"),
    ("havens", None, None, "the PEN shot (5440-5679) has no page turn in its frames: EDIT still owes the incoming "
     "turn, so the page-turn sound has no measured frame yet"),
]
CUES_VS_PICTURE = [
    "breaths: 'forges_cold' cuts the hall for 1.2 s (3848 -> 3876.8) only; nothing in the cue sheet or the level "
    "map can fail on sound between 3877 and 3999 (C16's band is -99..-1 LU relative), so the hard silence through "
    "3999 was a property of the note list, unchecked. Pass 1's Ring entry at 4000 may start up to 0.52 s early "
    "(ANTIC_HI for the quiet strings), i.e. inside the silence",
    "events: the run's beacon_1..7 tags (call/answer) are on fires this Mac cannot see; the map's measured catches "
    "(3468-3791) have no cue-sheet events at all",
    "sfx/ambience: empty by design (the effects are SOUND's); see sound/c5_sound_events.json",
]


def disagreements(events):
    by = {e["id"]: e for e in events}
    d = json.load(open(os.path.join(V3, "barmap_C5.json")))
    sync = {e["id"]: e for e in d["sync"]}
    out = []
    for bid, mid, field, what in BARMAP_VS_PICTURE:
        old = sync.get(bid, {})
        row = dict(barmap_id=bid, barmap_frame=old.get("f"), barmap_says=old.get("what"), measured=mid, what=what)
        if mid and mid in by:
            fr = by[mid]["frames"].get(field)
            row["measured_frame"] = fr
            row["delta_frames"] = None if fr is None or old.get("f") is None else int(fr - old["f"])
        out.append(row)
    for event in events:
        sid = event.get("sync_id")
        if sid:
            old = sync[sid]
            f = event["frames"].get(event["sync_field"])
            out = [r for r in out if r["barmap_id"] != sid]
            out.append(dict(barmap_id=sid, barmap_frame=old["f"], barmap_says=old["what"],
                            measured=event["id"], measured_frame=f,
                            delta_frames=None if f is None else f-old["f"],
                            what=event.get("note", event["event"])))
    out = [r for r in out if r["barmap_id"] != "beacon_1..7"]
    from measure_c5_analysis import PASS1_REASONS
    for row in out:
        if row["barmap_id"] in PASS1_REASONS and not row.get("measured_frame"):
            row["what"] = PASS1_REASONS[row["barmap_id"]]
    return out


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build_table(events, identity):
    for e in events:
        sh = e["shot"]
        tag, notes = OBSERVED.get(sh, ("", {}))
        if e["id"] in notes:
            fr, saw = notes[e["id"]]
            if tag and identity[sh]["frame_set_sha256"].startswith(tag):
                e["observed"] = dict(frames=fr, saw=saw)
                e["basis"] = ["MEASURED", "OBSERVED"]
            else:
                e["observed_stale"] = "an OBSERVED note exists for an older frame set of this shot: re-inspect"
    return dict(
        schema="long-dawn/c5-events-measured/1",
        cut="C5", fps=FPS, frames="absolute C5 cut frame numbers; reused-source offsets and composite layers are recorded in shots",
        generated_by="music/src/measure_c5_events.py", generator_sha256=sha_file(os.path.abspath(__file__)),
        basis_labels=dict(MEASURED="a pixel statistic decided the frame(s) (see method)",
                          OBSERVED="checked by eye on native-resolution stills (never in motion)",
                          INFERRED="not used for any frame in this table"),
        sync_definitions=dict(
            catch="a fire catching: first = the first frame its flame is visible, fastest = its fastest growth, "
                  "full = 90% of its size; a sound's burst ARRIVES on first..fastest, a score's resolution on full",
            ramp="a gradual change: first/half/full = 10%/50%/90% of the way",
            ink="a drawing stroke: frames where ink arrives (a pen/quill sound follows them)"),
        shots=identity,
        not_on_this_mac=[],
        generator_dependencies={name: sha_file(os.path.join(HERE, name)) for name in
                                ("measure_c5_delivered.py", "measure_c5_analysis.py")},
        events=events,
        barmap_C5_vs_picture=disagreements(events),
        cues_C5_vs_picture=[CUES_VS_PICTURE[0],
                           "Beacon Run catches are now measured from delivered frames; map catches remain measured.",
                           CUES_VS_PICTURE[2]])


SCHEMA = "long-dawn/c5-events-measured/1"
EVENT_KEYS = ("id", "shot", "stem", "event", "frames", "method", "region", "evidence", "confidence", "basis")


def _frame_values(x):
    if isinstance(x, bool):
        return []
    if isinstance(x, int):
        return [x]
    if isinstance(x, list):
        return [v for y in x for v in _frame_values(y)]
    return []


def validate_table(d, text=None):
    """the table's schema and invariants -> [problems] (empty when sound): every event names a delivered shot, its
    frames lie inside that shot, its basis says MEASURED, and no private path leaked into a public file"""
    out = []
    for k in ("schema", "cut", "fps", "generator_sha256", "shots", "events", "barmap_C5_vs_picture"):
        if k not in d:
            out.append(f"missing top-level {k}")
    if d.get("schema") != SCHEMA:
        out.append(f"schema {d.get('schema')!r} != {SCHEMA!r}")
    for s, (stem, a, b, sec) in SHOTS.items():
        got = d.get("shots", {}).get(s)
        if not got or got.get("stem") != stem or (got.get("first"), got.get("last")) != (a, b):
            out.append(f"shot {s}: identity missing or not {stem} {a}-{b}")
        elif len(got.get("frame_set_sha256", "")) != 64:
            out.append(f"shot {s}: no frame-set sha256")
    seen = set()
    for e in d.get("events", []):
        eid = e.get("id", "?")
        miss = [k for k in EVENT_KEYS if k not in e]
        if miss:
            out.append(f"{eid}: missing {miss}")
            continue
        if eid in seen:
            out.append(f"{eid}: duplicate id")
        seen.add(eid)
        if e["shot"] not in SHOTS or e["stem"] != SHOTS[e["shot"]][0]:
            out.append(f"{eid}: unknown shot/stem {e['shot']}/{e['stem']}")
            continue
        a, b = SHOTS[e["shot"]][1:3]
        for k, v in e["frames"].items():
            for f in _frame_values(v):
                if not a <= f <= b:
                    out.append(f"{eid}: frames.{k} = {f} outside its shot {a}-{b}")
        if e["confidence"] not in ("high", "medium", "low"):
            out.append(f"{eid}: confidence {e['confidence']!r}")
        if "MEASURED" not in e["basis"]:
            out.append(f"{eid}: basis {e['basis']} lacks MEASURED")
        if "observed_stale" in e:
            out.append(f"{eid}: its OBSERVED note is for another frame set")
        if "OBSERVED" in e["basis"] and not (e.get("observed") or {}).get("saw"):
            out.append(f"{eid}: OBSERVED without a note")
    text = text if text is not None else json.dumps(d)
    for bad in ("/Users/", "/home/", "/private/"):
        if bad in text:
            out.append(f"a private path ({bad}...) in the table")
    return out


def is_current(d):
    """True when the table was written by this very script (a stale table after an edit of the measurer fails)"""
    return (d.get("generator_sha256") == sha_file(os.path.abspath(__file__)) and
            d.get("generator_dependencies") == {name: sha_file(os.path.join(HERE, name)) for name in
            ("measure_c5_delivered.py", "measure_c5_analysis.py")})


def inherited_controls(key, events, stats, root):
    """Audit controls for the previously measured shots, retaining their original event frames."""
    from measure_c5_analysis import control
    a, b = SHOTS[key][1:3]
    def test(values, interval, threshold, direction=1, why="pre-event interval, same fixed threshold"):
        return control([dict(f=a+i, signal=float(v)) for i,v in enumerate(values)],
                       "signal", interval, threshold, direction, why)
    if key == "cold":
        values = stats["total"]
        drop = np.r_[0, -np.diff(values)]
        # Argmin alone always selects a frame. Require a drop of half the pre-shutdown median.
        threshold = .5 * float(np.median(values[:8]))
        assert drop.max() >= threshold, "no abrupt shutdown: argmin alone is insufficient"
        events[0]["method"] += "; accepted only if the drop exceeds half the median area C3840-3847"
        events[0]["negative_controls"] = [test(drop, (3860,3999), threshold,
            why="post-shutdown Ring and cold forges; same absolute drop threshold")]
    elif key == "unfinished":
        for event, field in zip(events[:2], ("warmth", "sky")):
            v = stats[field]
            threshold = -v[0] + .1*(v[0]-float(np.median(v[-20:])))
            event["negative_controls"] = [test(v, (4000,4010), threshold, -1)]
        hs = [file_sha(root, SHOTS[key][0], f) for f in range(4199,4217)]
        same = [dict(f=4200+i, same=int(x==y)) for i,(x,y) in enumerate(zip(hs,hs[1:]))]
        events[2]["negative_controls"] = [control(same,"same",(4200,4216),1,
            why="before the final held frame, adjacent files must still differ")]
    elif key in ("deep", "pen"):
        v = stats["d"]
        events[0]["negative_controls"] = [test(v, (a+1,b), 3*float(np.median(v[1:]))+.5,
            why="held illustration, all frame differences examined")]
    elif key == "watch":
        hits = events[0]["frames"]["appear_in_frame"]
        events[0]["negative_controls"] = [control(
            [dict(f=f,births=hits.count(f)) for f in range(a,b+1)], "births", (a,b), 1,
            why="already-burning beacons across the full shot; occlusion remains a possible false positive"),
            control([dict(f=f,births=hits.count(f)) for f in range(a,b+1)], "births", (4480,4600), 1,
                    why="opening already-burning landscape, before either later ridge-uncovering candidate")]


def run(shots, root, verbose=True, trace_dir=None):
    fns = {"refusal": m_refusal, "trap": m_trap, "reveal": m_reveal, "map": m_map, "cold": m_cold,
           "unfinished": m_unfinished, "deep": m_deep, "watch": m_watch, "pen": m_pen}
    events, identity = [], {}
    picture = None
    for s in shots:
        stem, a, b, sec = SHOTS[s]
        if s in EXTRA_SHOTS:
            from measure_c5_delivered import Picture, scan
            from measure_c5_analysis import analyze
            picture = picture or Picture(root)
            rows = scan(picture, s)
            if trace_dir:
                with open(os.path.join(trace_dir, s+"_trace.json"), "w") as fh:
                    json.dump(rows, fh, separators=(",", ":"))
            e = analyze(s, rows)
            identity[s] = picture.identity(s)
        else:
            e, stats = fns[s](root)
            inherited_controls(s, e, stats, root)
            identity[s] = dict(stem=stem, first=a, last=b, section=sec,
                               frame_set_sha256=frame_set_sha256(root, stem, a, b))
        events += e
        if verbose:
            for x in e:
                print(json.dumps({k: x[k] for k in ("id", "frames", "confidence")}, default=str), flush=True)
    return events, identity


def main():
    import cv2
    cv2.setNumThreads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=",".join(SHOTS))
    ap.add_argument("--frames-root", default=None)
    ap.add_argument("--dry", action="store_true", help="print, write nothing")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--trace-dir", help="optional existing directory for reproducible scalar traces")
    a = ap.parse_args()
    root = frames_root(a.frames_root)
    shots = [s for s in a.shots.split(",") if s]
    unknown = [s for s in shots if s not in SHOTS]
    if unknown:
        ap.error(f"unknown shots {unknown}; known: {list(SHOTS)}")
    for s in shots:
        if s in EXTRA_SHOTS:
            continue  # Picture.sources validates EDIT's resolved sources and every composite layer.
        stem, f0, f1, _ = SHOTS[s]
        missing = [f for f in range(f0, f1 + 1) if not os.path.exists(fpath(root, stem, f))]
        if missing:
            raise SystemExit(f"{stem}: {len(missing)} of {f1 - f0 + 1} frames missing under {root} "
                             f"(first {missing[0]}): refusing to measure a partial shot")
    events, identity = run(shots, root, trace_dir=a.trace_dir)
    if a.dry:
        return 0
    if set(shots) != set(SHOTS):
        raise SystemExit("the table is written only for every shot at once (use --dry for a subset)")
    table = build_table(events, identity)
    with open(a.out, "w") as fh:
        json.dump(table, fh, indent=1, default=str)
        fh.write("\n")
    print(f"wrote {a.out}: {len(events)} events over {len(identity)} shots")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print(f"peak RSS: {rss} bytes; limit {RSS_ABORT} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
