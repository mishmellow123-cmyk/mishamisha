"""THE LONG DAWN C5 - verify a RENDERED score pass 2 (and its sound master) against the measured picture.

Written on a machine that could not render (no VSCO-2-CE samples, no `soxr`, no effect recordings): nothing here
has been run on a real render. It is the numerical half of the morning's check, after

    python render_v3.py C5P2          # -> cache/v3/premaster_score_final_C5P2.npy (+ .json), manifest, part stems,
                                      #    out/v3/final_C5P2.wav
    python sound_v3.py C5P2           # (optional) -> out/v3/sound_C5P2.wav
    python verify_c5_render.py C5P2   # exit 0 only if every check below passes; --json writes the report

Each check is MEASURED on the rendered arrays; none of them is listening.

  length      the premaster and every master are exactly the cut's length (5,920 frames = 11,840,000 samples)
  silence     the peak of every frame 3848..3999 of the premaster and the masters: frames 3849..3999 must stay
              under --floor-db (default -90 dBFS; the masters' 24-bit dither sits near -138 dBFS). Frame 3848 is
              reported, not judged: the cut lands inside it
  arrivals    each part that enters FROM SILENCE on a re-timed event (the two first fires, the trap's measured forges,
              the far beacons' answers, the resolution on all_lit): its ARRIVAL, the first point within -0.7..+0.5 s
              of the event where the part's 5 ms energy envelope comes within 6 dB of the entry's early peak (the
              definition kit_v3.anticipate aims at), must lie within +-1 frame of the event. A part not quiet
              (30 dB under that peak) in the 1.5 s before the search window is NOT ISOLATED: a failure to look at,
              never a pass
  swells      the violins' climb on pulls_ahead is a bowed SWELL (render 1 of pass 2: -63 to -45 dB across its
              0.44 s first note, with a flat stretch at -53 dB around the frame), so its arrival depends on where the
              threshold meets that stretch: render 2 moved the note 10 ms and this module's own 8 dB crossing 2.8
              frames. Its arrival zone is bounded by the two instruments that judge it: the probe the score declares
              for it (analyze_v3.measure3 'arrivew', 8 dB under the peak on the 60 Hz-highpassed 10 ms envelope, run
              on the same stem) and this module's 6 dB arrival. The frame must lie inside that zone, widened by
              +-1 frame, so a swell wholly before or after its frame still fails
  bloom onsets the resolution's brass on all_lit is a BLOOM by design (score_v3_C5.py's battery: "a bloom's ONSET
              sits on its beat", anticipation fixed at 0.05 s, then a crescendo from 0.1 to 0.25 over a beat), so
              its -6 dB point is meant to fall after the frame (render 1 of pass 2: +1.1 to +6.3 frames). Its ONSET,
              the first point in the same window where the envelope comes within 30 dB of the entry's early peak,
              must lie within +-1 frame of the event; the same isolation rule applies
  not before  the Ring's entry (cb_q, vc_q, vla_q): no sample of those parts above --floor-db before frame 4000
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
CACHE = os.path.join(MUSIC, "cache", "v3")
PARTS = os.path.join(CACHE, "parts")
OUT = os.path.join(MUSIC, "out", "v3")
V3 = os.path.join(MUSIC, "v3")
SR, FPS = 48000, 24
FR = SR // FPS
SHUTDOWN, RING_CUT = 3848, 4000
ARRIVALS = [("reveal", "hn3"), ("reveal", "hn_far"), ("low_fire", "hn3"), ("surge", "tbn"), ("flares_back", "hn3"),
            ("map_beacon_3", "hn_far"), ("map_beacon_4", "hn_farther"), ("one_dark", "vln1")]
SWELLS = [("pulls_ahead", "vln1"), ("pulls_ahead", "vln2")]
SWELL_W = 0.25                     # the climb's sync window as the score declares it (score_v3_C5P2.trap)
BLOOMS = [("all_lit", pn) for pn in ("hn", "hn2", "hn3", "tbn", "tuba")]
BLOOM_ONSET_DB = 30.0
RING_ENTRY = ("cb_q", "vc_q", "vla_q")
PRE_S, PRE_GAP_S, WIN_BEFORE_S, WIN_AFTER_S = 1.5, 0.7, 0.7, 0.5


def db(x):
    return 20.0 * np.log10(max(float(x), 1e-12))


def check_length(n, frames, what):
    ok = n == frames * FR
    return dict(check="length", what=what, samples=int(n), expected=frames * FR, ok=bool(ok))


def frame_peaks(y, first_frame, f0, f1):
    """{frame: peak dBFS} for frames f0..f1-1 of y, whose sample 0 is frame first_frame's first sample"""
    out = {}
    for f in range(f0, f1):
        a, b = (f - first_frame) * FR, (f - first_frame + 1) * FR
        seg = y[max(a, 0):max(b, 0)]
        out[f] = db(np.abs(seg).max()) if len(seg) else None
    return out


def check_silence(y, what, floor_db, first_frame=0):
    pk = frame_peaks(y, first_frame, SHUTDOWN, RING_CUT)
    judged = {f: v for f, v in pk.items() if f > SHUTDOWN}
    missing = [f for f, v in judged.items() if v is None]
    loud = {f: round(v, 1) for f, v in judged.items() if v is not None and v > floor_db}
    worst = max((v for v in judged.values() if v is not None), default=None)
    first = pk[SHUTDOWN]
    return dict(check="silence", what=what, floor_db=floor_db,
                first_frame_peak_db=None if first is None else round(first, 1),
                worst_db=None if worst is None else round(worst, 1), over_floor=loud, frames_missing=missing,
                ok=not loud and not missing)


def envelope_db(x):
    mono = np.abs(np.asarray(x, np.float64)).mean(1) if x.ndim == 2 else np.abs(np.asarray(x, np.float64))
    w = int(0.005 * SR)
    return 10 * np.log10(np.convolve(mono ** 2, np.ones(w) / w, mode="same") + 1e-14)


def part_window(y, offset, i0, i1):
    """samples [i0, i1) of a part stem stored from `offset` (zeros outside it)"""
    out = np.zeros((i1 - i0,) + y.shape[1:], np.float32)
    a, b = max(i0, offset), min(i1, offset + len(y))
    if b > a:
        out[a - i0:b - i0] = y[a - offset:b - offset]
    return out


def arrival(y, offset, frame, under_db=6.0):
    """-> dict(status, delta_frames, ...): see the module doc (under_db=BLOOM_ONSET_DB measures a bloom's onset)"""
    i_ev = frame * FR
    i0 = i_ev - int((PRE_S + PRE_GAP_S) * SR)
    x = part_window(y, offset, i0, i_ev + int(WIN_AFTER_S * SR))
    env = envelope_db(x)
    k_pre = int(PRE_S * SR)
    k_win = k_pre + int((PRE_GAP_S - WIN_BEFORE_S) * SR)
    win = env[k_win:]
    peak = float(win.max())
    if peak < -100.0:
        return dict(status="SILENT", peak_db=round(peak, 1))
    pre = float(env[:k_pre].max())
    j = int(np.argmax(win >= peak - under_db))
    delta_s = (i0 + k_win + j - i_ev) / SR
    d = dict(delta_frames=round(delta_s * FPS, 2), peak_db=round(peak, 1), pre_db=round(pre, 1))
    if pre > peak - 30.0:
        return dict(d, status="NOT ISOLATED")
    return dict(d, status="ON" if abs(delta_s) <= 1.0 / FPS else "OFF")


def arrivew(y, offset, frame, W=SWELL_W):
    """analyze_v3.measure3's 'arrivew' (analyze_v3.py, the probe the score declares for a swell) on this stem, as
    the render's own sync report runs it: -> frames from the event, or None"""
    import analyze_v3 as A3
    T = frame / FPS
    a = max(0, int((T - W - 0.6) * SR))
    x = part_window(y, offset, a, int((T + W + 1.0) * SR))
    tm, _ = A3.measure3(x, T - a / SR, W, "arrivew")
    return None if tm is None else round((tm + a / SR - T) * FPS, 2)


def check_not_before(y, offset, frame, floor_db, first_from=SHUTDOWN):
    i0, i1 = first_from * FR, frame * FR
    x = part_window(y, offset, i0, i1)
    mag = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    loud = np.flatnonzero(mag > 10 ** (floor_db / 20))
    first = None if not len(loud) else round((i0 + int(loud[0])) / FR, 2)
    return dict(first_sound_frame=first, ok=first is None)


def load_part(name, key, parts_dir=PARTS):
    base = os.path.join(parts_dir, f"{name}__{key}")
    with open(base + ".json") as fh:
        meta = json.load(fh)
    return int(meta["offset"]), np.load(base + ".npy", mmap_mode="r")


def barmap_frames(cut):
    with open(os.path.join(V3, f"barmap_{cut}.json")) as fh:
        d = json.load(fh)
    return {e["id"]: e["f"] for e in d["sync"]}, d


def run(cut="C5P2", floor_db=-90.0, cache=CACHE, out_dir=OUT, parts_dir=PARTS):
    import soundfile as sf
    ev, bmj = barmap_frames(cut)
    frames = int(bmj["frames"])
    rep = []
    pm = os.path.join(cache, f"premaster_score_final_{cut}.npy")
    if not os.path.exists(pm):
        return [dict(check="inputs", ok=False, what=f"no {os.path.basename(pm)}: render first (render_v3.py {cut})")]
    y = np.load(pm, mmap_mode="r")
    rep.append(check_length(len(y), frames, os.path.basename(pm)))
    rep.append(check_silence(y, os.path.basename(pm), floor_db))
    for nm in (f"final_{cut}.wav", f"sound_{cut}.wav"):
        p = os.path.join(out_dir, nm)
        if not os.path.exists(p):
            rep.append(dict(check="inputs", what=nm, ok=nm.startswith("sound_"), note="absent"))
            continue
        info = sf.info(p)
        rep.append(check_length(info.frames, frames, nm) if info.samplerate == SR else
                   dict(check="length", what=nm, ok=False, note=f"{info.samplerate} Hz"))
        seg, _ = sf.read(p, start=SHUTDOWN * FR, stop=RING_CUT * FR, dtype="float32", always_2d=True)
        rep.append(check_silence(seg, nm, floor_db, first_frame=SHUTDOWN))
    man = os.path.join(cache, f"manifest_final_{cut}.json")
    with open(man) as fh:
        manifest = json.load(fh)
    for eid, pn in ARRIVALS:
        if pn not in manifest:
            rep.append(dict(check="arrival", event=eid, part=pn, ok=False, status="PART NOT RENDERED"))
            continue
        off, yp = load_part(pn, manifest[pn], parts_dir)
        a = arrival(yp, off, int(ev[eid]))
        rep.append(dict(check="arrival", event=eid, frame=ev[eid], part=pn, ok=a["status"] == "ON", **a))
    for eid, pn in SWELLS:
        if pn not in manifest:
            rep.append(dict(check="swell", event=eid, part=pn, ok=False, status="PART NOT RENDERED"))
            continue
        off, yp = load_part(pn, manifest[pn], parts_dir)
        late = arrival(yp, off, int(ev[eid]))
        early = arrivew(yp, off, int(ev[eid]))
        if "delta_frames" not in late or late["status"] == "NOT ISOLATED" or early is None:
            rep.append(dict(check="swell", event=eid, frame=ev[eid], part=pn, ok=False,
                            status="NOT FOUND" if early is None else late["status"]))
            continue
        zone = (early, late["delta_frames"])
        ok = zone[0] - 1.0 <= 0.0 <= zone[1] + 1.0
        rep.append(dict(check="swell", event=eid, frame=ev[eid], part=pn, ok=ok, status="ON" if ok else "OFF",
                        zone_frames=zone, peak_db=late["peak_db"], pre_db=late["pre_db"]))
    for eid, pn in BLOOMS:
        if pn not in manifest:
            rep.append(dict(check="bloom onset", event=eid, part=pn, ok=False, status="PART NOT RENDERED"))
            continue
        off, yp = load_part(pn, manifest[pn], parts_dir)
        a = arrival(yp, off, int(ev[eid]), under_db=BLOOM_ONSET_DB)
        rep.append(dict(check="bloom onset", event=eid, frame=ev[eid], part=pn, ok=a["status"] == "ON", **a))
    for pn in RING_ENTRY:
        off, yp = load_part(pn, manifest[pn], parts_dir)
        rep.append(dict(check="not before", event="ring_unfinished", frame=RING_CUT, part=pn,
                        **check_not_before(yp, off, RING_CUT, floor_db)))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cut", nargs="?", default="C5P2")
    ap.add_argument("--floor-db", type=float, default=-90.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rep = run(a.cut, a.floor_db)
    for r in rep:
        print(("ok   " if r["ok"] else "FAIL ") + json.dumps({k: v for k, v in r.items() if k != "ok"}))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    bad = [r for r in rep if not r["ok"]]
    print(f"{len(rep) - len(bad)}/{len(rep)} checks pass" + ("" if not bad else f"; {len(bad)} FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
