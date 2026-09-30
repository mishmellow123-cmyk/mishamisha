"""A score pass 2 (AP2): what the RENDERED files say, read-only (after render_v3.py AP2 and sound_v3.py AP2).

    python verify_ap2_render.py            # prints the checks; exit 1 if any fails
    python verify_ap2_render.py --json F   # also writes them to F

The page checks (score_v3_AP2.check, music/tests/test_ap2_pass2.py) prove the walk's timing on the NOTE DATA. This
measures the audio the render made:

  1. LENGTH: every AP2 master and stem is exactly 6,480 frames = 12,960,000 samples, 48 kHz, stereo.
  2. PROVENANCE: the premaster's identity sidecar names cut AP2 and the sha256 of the bar map, cue sheet and score
     module on disk now (audio_guard_v3 checks the first two; the score's is checked here).
  3. THE WALK, ON ITS OWN RENDERED STEMS (the render's dry part stems, cache/v3/parts, named by manifest_final_AP2):
     for each walk instrument (the pizzicato: cb_pizz + cb_pizz_go; the drum: feet + feet_go), the summed stem is
     exactly 0.0 in every sample from the lantern (4880) to one frame before the set-off, and its first onset (the
     first sample within ONSET_UNDER_DB of the first step's peak) lands within +-1 frame of the measured set-off.
  4. PASS 1 OUTSIDE THE WALK'S WINDOW: the AP2 premaster (the score, mixed, before the master) against pass 1's own
     delivered premaster (premaster_score_final_A.npy): the first and last frame at which they differ by more than
     1e-6 (-120 dBFS) must lie inside [lantern - 1, watch-fire 3): nothing before the lantern differs and pass 1's
     walk from watch-fire 3 (5520-5833) is pass 1's, sample for sample.
  5. LOUDNESS (report only): integrated LUFS (ITU-R BS.1770, pyloudnorm) and true peak (4x oversampled, render_v2) of
     every AP2 master found.

Arrays are memory-mapped and read a window at a time (the machine is shared); nothing is written but --json.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timeline_v3 import FR, SR, V3, BarMap  # noqa: E402

MUSIC = os.path.dirname(HERE)
CACHE = os.path.join(MUSIC, "cache", "v3")
OUT = os.path.join(MUSIC, "out", "v3")
CUT = "AP2"
FRAMES = 6480
SAMPLES = FRAMES * FR
ONSET_UNDER_DB = 30.0
TOL_FRAMES = 1.0
IDENT_EPS = 1e-6                       # -120 dBFS: rounding of two mixes of the same parts is ~1e-7 (measured)
INSTRUMENTS = {"pizzicato": ("cb_pizz", "cb_pizz_go"), "drum": ("feet", "feet_go")}
MASTERS = ("final_AP2", "final_AP2_score", "final_AP2_sfx", "score_AP2", "sound_AP2", "sound_AP2_score",
           "sound_AP2_sfx", "sfx_AP2")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lengths(out_dir=OUT):
    rows = []
    for name in MASTERS:
        p = os.path.join(out_dir, name + ".wav")
        if not os.path.exists(p):
            rows.append(dict(check="length", what=name, ok=None, note="not rendered yet"))
            continue
        i = sf.info(p)
        rows.append(dict(check="length", what=name, samples=i.frames, samplerate=i.samplerate, channels=i.channels,
                         ok=bool(i.frames == SAMPLES and i.samplerate == SR and i.channels == 2)))
    return rows


def provenance(cache=CACHE):
    p = os.path.join(cache, "premaster_score_final_AP2.json")
    d = json.load(open(p))
    want = dict(cut=CUT, frames=FRAMES, samples=SAMPLES,
                barmap_sha256=_sha(os.path.join(V3, "barmap_AP2.json")),
                cues_sha256=_sha(os.path.join(V3, "cues_AP2.json")),
                score_sha256=_sha(os.path.join(HERE, "score_v3_AP2.py")))
    bad = [k for k, v in want.items() if d.get(k) != v]
    return dict(check="provenance", what=os.path.basename(p), ok=not bad, mismatched=bad, rendered_utc=d.get("utc"))


def stem(name, key, parts=os.path.join(CACHE, "parts")):
    """(offset, array) of a rendered part: render_v3 stores the active span only, zero outside it by construction"""
    base = os.path.join(parts, f"{name}__{key}")
    meta = json.load(open(base + ".json"))
    y = np.load(base + ".npy", mmap_mode="r")
    if y.ndim != 2 or y.shape[1] != 2 or len(y) != meta["n"]:
        raise ValueError(f"{name}: the cached stem disagrees with its metadata")
    return int(meta["offset"]), y


def window(parts, a, b):
    """the sum of several (offset, array) stems over samples [a, b)"""
    out = np.zeros((b - a, 2), np.float64)
    for off, y in parts:
        lo, hi = max(a, off), min(b, off + len(y))
        if hi > lo:
            out[lo - a:hi - a] += y[lo - off:hi - off]
    return out


def entry(parts, lan_f, go_f, tol=TOL_FRAMES, under_db=ONSET_UNDER_DB):
    """-> dict: the summed stem's peak |x| over [lantern, set-off - 1 frame) (must be exactly 0), and its first onset
    (frames) in [set-off - 0.5 s, set-off + 0.5 s): the first sample within under_db of the peak of that window"""
    a, b = int(lan_f * FR), int((go_f - 1.0) * FR)
    quiet = float(np.abs(window(parts, a, b)).max()) if b > a else 0.0
    w0, w1 = int(go_f * FR - SR // 2), int(go_f * FR + SR // 2)
    x = np.abs(window(parts, w0, w1)).max(1)
    pk = float(x.max())
    if pk <= 0.0:
        return dict(quiet_peak=quiet, onset_frame=None, ok=False)
    k = int(np.flatnonzero(x >= pk * 10 ** (-under_db / 20))[0])
    on = (w0 + k) / FR
    return dict(quiet_peak=quiet, onset_frame=round(on, 3), offset_frames=round(on - go_f, 3),
                peak_dbfs=round(20 * np.log10(pk), 2), ok=bool(quiet == 0.0 and abs(on - go_f) <= tol))


def walk_checks(bm, cache=CACHE):
    man = json.load(open(os.path.join(cache, "manifest_final_AP2.json")))
    lan_f, go_f = bm.event("lantern")["frame"], bm.event("walk_setoff")["frame"]
    rows = []
    for inst, names in INSTRUMENTS.items():
        parts = [stem(n, man[n]) for n in names]
        r = entry(parts, lan_f, go_f)
        rows.append(dict(check="walk entry", what=f"{inst} ({' + '.join(names)})", set_off=go_f, **r))
    return rows


def pass1_identity(bm, cache=CACHE, eps=IDENT_EPS, chunk=240 * FR):
    a = np.load(os.path.join(cache, "premaster_score_final_A.npy"), mmap_mode="r")
    p = np.load(os.path.join(cache, "premaster_score_final_AP2.npy"), mmap_mode="r")
    if a.shape != p.shape:
        return dict(check="pass 1 outside the walk", ok=False, note=f"shapes {a.shape} vs {p.shape}")
    first = last = None
    worst_out = 0.0
    lan_f, wf3_f = bm.event("lantern")["frame"], bm.event("watchfire_3")["frame"]
    for s in range(0, len(a), chunk):
        d = np.abs(np.asarray(p[s:s + chunk], np.float64) - np.asarray(a[s:s + chunk], np.float64)).max(1)
        idx = np.flatnonzero(d > eps)
        if len(idx):
            first = (s + idx[0]) / FR if first is None else first
            last = (s + idx[-1]) / FR
    ok = first is None or (first >= lan_f - 1.0 and last < wf3_f)
    for s0, s1 in ((0, int((lan_f - 1) * FR)), (int(wf3_f * FR), len(a))):
        for s in range(s0, s1, chunk):
            e = min(s1, s + chunk)
            worst_out = max(worst_out, float(np.abs(np.asarray(p[s:e], np.float64) - np.asarray(a[s:e], np.float64))
                                            .max()))
    return dict(check="pass 1 outside the walk", ok=bool(ok), eps=eps,
                first_frame_differing=None if first is None else round(first, 3),
                last_frame_differing=None if last is None else round(last, 3),
                max_abs_diff_before_lantern_and_from_watchfire_3=worst_out,
                window_allowed=[lan_f - 1, wf3_f])


def loudness(out_dir=OUT):
    import pyloudnorm as pyln
    import render_v2 as R2
    rows = []
    m = pyln.Meter(SR)
    for name in MASTERS:
        p = os.path.join(out_dir, name + ".wav")
        if not os.path.exists(p):
            continue
        y, _ = sf.read(p, dtype="float32", always_2d=True)
        rows.append(dict(check="loudness", what=name, lufs_integrated=round(float(m.integrated_loudness(
            y.astype(np.float64))), 2), true_peak_dbtp=round(float(R2.true_peak_db(y)), 2), ok=None))
        del y
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--no-loudness", action="store_true")
    a = ap.parse_args(argv)
    bm = BarMap(CUT)
    rows = lengths() + [provenance()] + walk_checks(bm) + [pass1_identity(bm)]
    if not a.no_loudness:
        rows += loudness()
    bad = [r for r in rows if r.get("ok") is False]
    for r in rows:
        flag = {True: "PASS", False: "FAIL", None: "info"}[r.get("ok")]
        print(f"{flag:4s} {r['check']:24s} {r['what'] if 'what' in r else ''}: " +
              ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("check", "what", "ok")))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1, default=float)
    print(f"{len(bad)} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
