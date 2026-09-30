"""C5P2's three sound edges as shaped gestures (lane polishcsound, 30 Sep 2026), applied by sound_v3.write.

Measured on the delivered sound_C5P2.wav (00:13; 5 ms stereo RMS, 50 ms windows at 5 ms hops):
  13.745 s (f329.9)  -12.8 dB  the riffle's flap at f329.3 (-14.6 dBFS) decaying into a gap in the take; the score
                               is continuous there. Fixed in the effects table (sound_c5_table.py, C5.riffle): the
                               riffle rises with the blur and swells into the page's landing, its flaps tamer.
  160.325 s (f3847.8) -18.6 dB  score_v3_C5P2.end_on_the_cut stops the D-major swell with a 30 ms release, and the
                               breath at forges_cold (cues_C5P2.json) cuts the hall: the orchestra falls from -13 to
                               digital zero in 40 ms while the forges it scores fall into step.
  196.435 s (f4714.4) -51.0 dB  the breath 275 ms before the sunrise (cues_C5P2.json) gates score AND effects to
                               -85..-96 dBFS in the middle of the C19-C20 dissolve (4712-4728); the strings then
                               enter +52.5 dB in 50 ms.

What this module does (C5P2 only; every other cut passes through untouched):
  * RELEASE (premaster, frames 3776-3984). The fifteen notes that end_on_the_cut shortened (strings, brass,
    timpani roll: pitches, starts, dynamics and samples unchanged) are re-rendered by the sampler ending ON the
    shutdown with a natural release (RELEASE_S), and the passage is re-mixed with the renderer's own seats, EQ
    and hall, without the forges_cold breath: the swell peaks on 3848 and settles with the forges by ~3872 while
    the hall rings under the unison hammers. The re-rendered parts must equal their cached performances up to the
    old cut (checked sample for sample), and the original passage re-mixed WITH its breath must reproduce the
    saved premaster (checked, MAX_RECON_ERROR), or the render stops.
  * DAWN (premaster, frames 4688-4800). The same re-mix without the sunrise breath: the chorale's release and
    hall carry the dissolve, and the strings' anticipated entry blooms onto 4720 as the kit placed it, instead of
    being gated to it. The effects: the table's no_breath on C5.wind.watch, and BRIDGE, a thread of the same
    recorded air (fs:725630, the take both beds play, from a stretch neither plays here) across 4700-4744, where
    the wind's fade and the dawn air's fade met at digital zero (4720).
  * BOUNDED MASTERING. The effects premaster outside MASTER_WINDOWS is restored from the delivered one and the
    master envelope from the delivered one (both pinned by SHA-256 in polish_c5_reference.json), without a
    re-normalisation scalar: outside the windows the new master is the delivered master, sample for sample.

No note is added, moved or re-pitched; the score module, its cue sheet and its cached premaster are unchanged
(verify_c5_render.py's premaster silence check still reads the score as composed; the delivered sound master
now carries the release).
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from timeline_v3 import SR, BEAT_N, BEAT_S, BarMap

FPS = 24
FR = SR // FPS
CUT = "C5P2"
SHUTDOWN, SETTLED, RING_CUT, SUNRISE = 3848, 3872, 4000, 4720

RIFFLE = (323, 392)                  # what the table's C5.riffle row can change: its clip (hit 326 - 0.1 s ..
                                     # + 1.6 s) and its send's 1.06 s outdoor tail (the effects premaster)
RELEASE = (3776, 3984)               # premaster edits, half-open frames; joins lie wholly inside
DAWN = (4688, 4800)
RAMP_F = 16                          # quintic join at each end of a premaster edit
# the master envelope and the effects are re-done only here: each premaster/effects edit plus 16 frames each
# side (render_v3.bounded_master_env's joins) and room for the 8 Hz master high-pass to settle
MASTER_WINDOWS = ((300, 424), (3760, 4000), (4672, 4816))
SFX_RAMP_F = 8                       # the effects' join back to the delivered stem, inside each window

# the natural release each family gets on the shutdown (the sampler's own shape, exp(-4.5u)*sqrt(1-u) over rel):
# strings -11 dB at the first unison hammer (3860) and -22 dB when the forges have settled (3872); brass and the
# timpani roll let go sooner, under them
RELEASE_S = {"cb": 2.0, "vc": 2.0, "vla": 2.0, "vln1": 2.0, "vln2": 2.0,
             "hn": 1.0, "hn2": 1.0, "hn3": 1.0, "tbn": 1.0, "tuba": 1.2, "timp_roll": 1.5}
# the dawn air's thread across the dissolve: the take both beds play (C5.wind.watch reads it at 7.7-23.8 s, C5.air.dawn
# at 121.5-147.1 s), from 112.0 s, which neither plays here, filtered as C.air.dawn is; rising 4700-4712, holding
# through the dissolve's middle, gone by 4744 (the air is at its own plateau, -53.5 dBFS premaster, from ~4756).
# -61 dBFS RMS (premaster) keeps the effects' floor through 4720 within ~8 dB of the air that follows
BRIDGE = dict(src=("fs:725630", 112.0), frames=(4700, 4744), rise_f=12, fall_f=16, rms_db=-61.0, hp=60, lp=2500)
MAX_RECON_ERROR = 1e-5               # |local re-mix with the original breaths - saved premaster|, full scale
IDENTICAL_BEFORE_S = 0.2             # a re-rendered part must equal its cached stem until this long before 3848
REFERENCE_JSON = Path(__file__).with_name("polish_c5_reference.json")
REPORT = {}                          # the last render's checks (read by the lane's audit)


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def join_weight(first, last, ramp_f, n=None):
    """per-sample weight over frames [first, last): 0 at both ends, 1 inside, quintic ramps of ramp_f frames"""
    n = (last - first) * FR if n is None else n
    f = np.arange(n, dtype=np.float64) / FR
    return (smooth(f / ramp_f) * smooth((last - first - f) / ramp_f)).astype(np.float32)


# ------------------------------------------------------------------ the score, its cache, the re-rendered releases
def score_and_manifest(cut=CUT):
    """(S, bm, manifest): the score as built now, refused unless every part's cache key matches the saved premaster's
    manifest (a stale cache would re-mix a different performance)"""
    import importlib
    import render_v3 as RV
    bm = BarMap(cut)
    S = importlib.import_module(f"score_v3_{cut}").build(bm)
    manifest = json.loads((Path(RV.CACHE) / f"manifest_final_{cut}.json").read_text())
    for name, part in S.used().items():
        if RV.part_key(part.to_dict(), bm.render_n) != manifest.get(name):
            raise ValueError(f"polish C5: stale cached part {name}; render the current score first")
    if S.push or S.groups:
        raise ValueError("polish C5: the local re-mix models no push or group bus")
    return S, bm, manifest


def release_parts(S, bm):
    """{part: part dict} for the parts end_on_the_cut shortened, those notes ending ON the shutdown and releasing
    over RELEASE_S; -> (parts, [(part, start beat, old end, new end, release s)]). Nothing else in a part changes."""
    cold = bm.event("forges_cold")["t"]
    cut = {}
    for pn, beat, _old_end, new_end in S.cut_at_shutdown:
        cut.setdefault(pn, []).append((beat, new_end))
    parts, log = {}, []
    for pn, rows in cut.items():
        if pn not in RELEASE_S:
            raise ValueError(f"polish C5: no release designed for {pn}, which the shutdown cuts")
        pd = copy.deepcopy(S.P(pn).to_dict())
        todo = list(rows)
        for nt in pd["notes"]:
            if nt["pitch"] is None:
                continue
            key = (round(nt["start"], 4), round(nt["start"] + nt["dur"], 4))
            if key in todo:
                todo.remove(key)
                old_end = nt["start"] + nt["dur"]
                nt["dur"] = cold / BEAT_S - nt["start"]
                nt["kw"] = dict(nt["kw"], rel=RELEASE_S[pn])
                log.append((pn, key[0], round(old_end, 4), round(nt["start"] + nt["dur"], 4), RELEASE_S[pn]))
        if todo:
            raise ValueError(f"polish C5: notes of {pn} cut on the shutdown were not found: {todo}")
        parts[pn] = pd
    return parts, log


def render_releases(parts, bm, manifest, a, b):
    """{part: slice [a, b) samples} of each part re-rendered with its release; each must equal its cached stem
    up to IDENTICAL_BEFORE_S before the shutdown (the same performance, only the ending changed)"""
    import render_v3 as RV
    import sampler
    import sampler_v2  # noqa: F401  (registers the quiet strings)
    sampler.RAW_BUDGET, sampler.RS_BUDGET = 140e6, 220e6
    out, first_diff = {}, {}
    limit = int(round((bm.event("forges_cold")["t"] - IDENTICAL_BEFORE_S) * SR))
    for name, pd in parts.items():
        buf = np.nan_to_num(sampler.render_part(pd, bm.render_n)).astype(np.float32)
        off, cached = RV.load_stem(name, manifest[name])
        ref = np.zeros((b - a, 2), np.float32)
        lo, hi = max(a, off), min(b, off + len(cached))
        if hi > lo:
            ref[lo - a:hi - a] = cached[lo - off:hi - off]
        seg = np.ascontiguousarray(buf[a:b])
        del buf
        d = np.flatnonzero(np.abs(seg - ref).max(1) > 0)
        first = a + int(d[0]) if len(d) else None
        if first is not None and first < limit:
            raise ValueError(f"polish C5: re-rendered {name} departs from its cached performance at sample {first}, "
                             f"before the shutdown")
        first_diff[name] = None if first is None else round(first / FR, 2)
        out[name] = seg
    sampler.clear_caches()
    REPORT["release_first_difference_frame"] = first_diff
    return out


def local_mix(S, bm, manifest, first, last, drop=(), replace=None):
    """the score premaster over frames [first, last), re-mixed exactly as render_v3.mix_score mixes it (seats,
    depth EQ, EQ, dry/send, hall, the breaths that remain, the 22 Hz high-pass), with the breaths whose start beat
    is in `drop` removed and `replace` = {part: slice from the preroll start} standing in for cached stems.
    The preroll covers the hall IR plus 1 s, so filters and reverb start from the same state as the full mix."""
    import mix as MX
    import render_v2 as R2
    import render_v3 as RV
    from scipy import signal
    irs = RV.hall_ir()
    a = first * FR - max(map(len, irs)) - SR
    b = last * FR
    n = b - a
    keep = [(i0, i1, ex) for i0, i1, ex in R2.breath_windows(S.breaths)
            if not any(abs(i0 - int(round(d * BEAT_N))) <= 1 for d in drop)]
    if len(keep) != len(S.breaths) - len(drop):
        raise ValueError("polish C5: a breath to release is not in the cue sheet")
    if any(i0 < a < i1 for i0, i1, _ in keep):
        raise ValueError("polish C5: the re-mix preroll starts inside a breath")
    local = [(i0 - a, i1 - a, ex) for i0, i1, ex in keep if i1 > a and i0 < b]
    dry = np.zeros((n, 2), np.float32)
    send = np.zeros((n, 2), np.float32)
    for name, part in S.used().items():
        if replace and name in replace:
            y = np.array(replace[name], dtype=np.float32)
            off = a
        else:
            off, raw = RV.load_stem(name, manifest[name])
            lo, hi = max(a, off), min(b, off + len(raw))
            if hi <= lo:
                continue
            y = np.array(raw[lo - off:hi - off], dtype=np.float32)
            off = lo
        if not np.any(y):
            continue
        if local:
            y *= R2.breath_env(len(y), local, name=name, offset=off - a)[:, None]
        y = MX.pan_width(y * np.float32(10 ** (part.gain_db / 20)), part.pan, part.width)
        y = MX.depth_eq(y, part.depth)
        spec = S.eq.get(name, (None, None))
        if spec[0]:
            y = MX.highpass(y, spec[0])
        if spec[1]:
            bb, aa = signal.butter(2, spec[1] / (SR / 2))
            y = signal.lfilter(bb, aa, y, axis=0).astype(np.float32)
        if len(spec) > 2 and spec[2]:
            y = y + MX.highpass(y, spec[3] if len(spec) > 3 else 6000.0, order=2) * np.float32(10 ** (spec[2] / 20) - 1)
        i = off - a
        dry[i:i + len(y)] += y * (1.0 - 0.35 * part.depth)
        send[i:i + len(y)] += y * part.send
    wet = R2.convolve_with_breaths(send, irs, local, n) if local else MX.convolve_stereo(send, irs)
    out = MX.highpass(dry + wet, 22.0)
    fader = getattr(S, "fader", None)
    if fader and not (b <= round(fader[0][0] * SR) or a >= round(fader[-1][0] * SR)):
        raise ValueError("polish C5: the score's fader ride reaches this passage; the re-mix does not model it")
    return out[first * FR - a:]


def breath_beat(S, bm, event, before_s):
    """the start beat of the cue sheet's breath `before_s` seconds before `event`"""
    t = bm.event(event)["t"] - before_s
    hit = [b0 for b0, _, _ in S.breaths if abs(b0 * BEAT_S - t) < 0.01]
    if len(hit) != 1:
        raise ValueError(f"polish C5: no single breath {before_s} s before {event}")
    return hit[0]


def blend(score, first, last, replacement):
    """score[first:last] frames += join weight * (replacement - score): exact identity outside, smooth joins"""
    a, b = first * FR, last * FR
    if replacement.shape != (b - a, 2):
        raise ValueError("polish C5: replacement has the wrong shape")
    w = join_weight(first, last, RAMP_F)
    region = score[a:b]
    region += ((replacement - region) * w[:, None]).astype(np.float32)
    return score


def polished_score(score, cut=CUT):
    """the premaster with the release and the dawn re-mixed in (in place); the checks land in REPORT"""
    S, bm, manifest = score_and_manifest(cut)
    cold = breath_beat(S, bm, "forges_cold", 0.03)
    dawn = breath_beat(S, bm, "sunrise", 0.275)
    for (first, last), label in ((RELEASE, "release"), (DAWN, "dawn")):
        err = float(np.abs(local_mix(S, bm, manifest, first, last) - score[first * FR:last * FR]).max())
        REPORT[f"{label}_reconstruction_max_abs"] = err
        if not err <= MAX_RECON_ERROR:
            raise ValueError(f"polish C5: the {label} re-mix does not reproduce the saved premaster ({err:.2e})")
    import render_v3 as RV
    a = RELEASE[0] * FR - max(map(len, RV.hall_ir())) - SR
    parts, log = release_parts(S, bm)
    REPORT["release_notes"] = log
    stems = render_releases(parts, bm, manifest, a, RELEASE[1] * FR)
    new = local_mix(S, bm, manifest, *RELEASE, drop=(cold,), replace=stems)
    blend(score, *RELEASE, new)
    new = local_mix(S, bm, manifest, *DAWN, drop=(dawn,))
    blend(score, *DAWN, new)
    print(f"  polish C5: re-mix reproduces the saved premaster to {REPORT['release_reconstruction_max_abs']:.2e} "
          f"(release) and {REPORT['dawn_reconstruction_max_abs']:.2e} (dawn); {len(log)} notes released on the "
          f"shutdown, re-rendered parts first depart at frames {sorted(set(REPORT['release_first_difference_frame'].values()), key=str)}",
          flush=True)
    return score


def dawn_air_bridge(sfx):
    """BRIDGE added to the effects premaster (in place): a fixed recorded excerpt, quintic in and out, no attack"""
    import sound_v3 as SV
    first, last = BRIDGE["frames"]
    n = (last - first) * FR
    ref, t0 = BRIDGE["src"]
    y = SV.load(ref, t0, t0 + n / SR).copy()
    if len(y) != n:
        raise ValueError("polish C5: the dawn air's bridge excerpt is shorter than its window")
    y = SV.proc(y, hp=BRIDGE["hp"], lp=BRIDGE["lp"])
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms <= 1e-10:
        raise ValueError("polish C5: the dawn air's bridge excerpt is silent")
    y *= np.float32(10 ** (BRIDGE["rms_db"] / 20) / rms)
    f = first + np.arange(n) / FR
    env = smooth((f - first) / BRIDGE["rise_f"]) * smooth((last - f) / BRIDGE["fall_f"])
    sfx[first * FR:last * FR] += (y * env[:, None]).astype(np.float32)
    return sfx


def prepare_master(cut, score, sfx):
    """sound_v3.write's hook, after the untouched premaster and effects stem are loaded, before the peak guard"""
    if cut != CUT:
        return score, sfx
    import sound_v3 as SV
    SV._LOADED.clear()          # the effects stem is built; its recordings need not stay resident while we render
    return polished_score(score, cut), dawn_air_bridge(sfx)


# ------------------------------------------------------------------ bounded mastering
def references():
    """{kind: path} of the pinned delivered references, checked against their SHA-256"""
    import render_v3 as RV
    manifest = json.loads(REFERENCE_JSON.read_text())
    files = {}
    for kind, row in manifest["files"].items():
        path = Path(RV.CACHE) / row["filename"]
        if not path.is_file():
            raise ValueError(f"missing C5 polish reference {path.name}; copy the lane's delivery into music/cache/v3")
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for data in iter(lambda: fh.read(1 << 20), b""):
                digest.update(data)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError(f"C5 polish reference hash mismatch: {path.name}")
        files[kind] = path
    return files


def check_windows(n, windows=MASTER_WINDOWS):
    """the master windows must be sorted, disjoint, inside the film, and hold every edit with room for its joins"""
    spans = [(a * FR, b * FR) for a, b in windows]
    if any(a < 0 or b > n or b <= a for a, b in spans) or any(b0 > a1 for (_, b0), (a1, _) in zip(spans, spans[1:])):
        raise ValueError("polish C5: master windows must be sorted, disjoint and inside the film")
    edits = [RIFFLE, RELEASE, DAWN, BRIDGE["frames"]]
    if not all(any(w0 + 16 <= e0 and e1 <= w1 - 16 for w0, w1 in windows) for e0, e1 in edits):
        raise ValueError("polish C5: an edit is not inside a master window's full-weight span")


def bound_master(cut, sfx):
    """the delivered effects outside MASTER_WINDOWS (joined smoothly inside them) and the master options that keep
    the delivered master envelope outside them, with no re-normalisation: the exterior is the delivered master"""
    if cut != CUT:
        return sfx, {}
    import soundfile as sf
    check_windows(len(sfx))
    files = references()
    with sf.SoundFile(files["sfx"]) as ref:
        if ref.frames != len(sfx) or ref.samplerate != SR or ref.channels != 2:
            raise ValueError("C5 polish reference effects disagree with the render format")
        out = ref.read(dtype="float32", always_2d=True)
    # what the restore undoes outside the windows (the peak guard re-solves its gain for the whole film, so its
    # ducking moves wherever the sum sat at its threshold), measured span by span to keep memory flat
    undone, cursor = 0.0, 0
    for first, last in sorted(MASTER_WINDOWS) + [(len(sfx) // FR, None)]:
        a = first * FR
        for k in range(cursor, a, 10 * SR):
            b = min(a, k + 10 * SR)
            undone = max(undone, float(np.abs(sfx[k:b] - out[k:b]).max()))
        cursor = len(sfx) if last is None else last * FR
    REPORT["sfx_exterior_max_abs_before_restore"] = undone
    print(f"  polish C5: effects outside the windows restored from the delivered stem (largest change undone "
          f"{undone:.2e} full scale)", flush=True)
    for first, last in MASTER_WINDOWS:
        a, b = first * FR, last * FR
        w = join_weight(first, last, SFX_RAMP_F)[:, None]
        out[a:b] += (sfx[a:b] - out[a:b]) * w
    return out, dict(reference_env=np.load(files["envelope"], mmap_mode="r"), edit_windows=MASTER_WINDOWS,
                     renormalize=False, tp_limit=-1.2)


def refresh_breath_probes(cut, score, probes):
    """the forges_cold and sunrise probes re-measured on the polished premaster (dry + hall), as render_v3 measures"""
    if cut != CUT:
        return probes
    out = [dict(p) for p in probes]
    for t_event, before in ((SHUTDOWN / FPS, 0.03), (SUNRISE / FPS, 0.275)):
        rows = [p for p in out if abs(p["start_s"] - (t_event - before)) < 0.01]
        if len(rows) != 1:
            raise ValueError("polish C5: breath probe anchor missing or ambiguous")
        row = rows[0]
        a, b = int(round(row["start_s"] * SR)), int(round(row["end_s"] * SR))

        def rms(lo, hi):
            return float(10 * np.log10(np.mean(np.asarray(score[lo:hi], np.float64) ** 2) + 1e-20))
        row.update(before_db=rms(a - int(0.3 * SR), a), inside_db=rms(a + int(0.036 * SR), b - int(0.006 * SR)),
                   after_db=rms(b, b + int(0.2 * SR)), measurement="post-polish score premaster, dry plus hall "
                   "(polish_c5_sound: the breath is released, not gated)")
    return out
