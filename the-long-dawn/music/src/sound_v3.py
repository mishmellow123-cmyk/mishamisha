"""THE LONG DAWN v3 - SOUND lane: the REAL effects stem, from recordings (Freesound originals, Sonniss GDC 2026 picks;
ElevenLabs only for the gaps), frame-aligned to the locked bar map and the cue sheet.

    python sound_v3.py B                 # -> music/out/v3/sfx_B.wav (+ sfxpre_B.wav, sound_B.wav mix check, battery)
    python sound_v3.py B --only B.lid,B.strike1 --solo out.wav    # audition some cues alone (pre-master)

What it does, per cue of music/v3/cues_<cut>.json (never edited) plus this lane's own EXTRA events:
  * every bed / event is rendered from a RECIPE (recordings, the segment, the sync point inside it, filters);
  * its level is MATCHED to the synthesized design COMPOSER approved for that cue (same gain_db, same fades, same
    envelope): beds by integrated K-weighted loudness, events by their loudest 400 ms; then this lane's `trim` (dB);
  * distance = air absorption + a MEASURED impulse response (no synthesized tails); the outdoor space is a measured
    IR too (ir_v3.py);
  * the breaths are honoured (every layer is silent in each breath window, like the score);
  * pre-master stem = music/out/v3/sfxpre_<cut>.wav (the domain render_v3.master() takes);
  * post-master stem = pre x the cut's master envelope (cache/v3/master_env_final_<cut>.npy): music/out/v3/sfx_<cut>.wav,
    a drop-in for final_<cut>_sfx.wav beside final_<cut>_score.wav; a true-peak guard keeps score + sfx <= -1.25 dBTP.
"""
import json
import os
import sys
import zlib

import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_v2 as R2  # noqa: E402  (breaths, true peak)
from timeline_v3 import SR, BarMap, _resolve_t  # noqa: E402

MUSIC = os.path.dirname(HERE)
LIB = os.path.join(MUSIC, "cache", "sound")
OUT = os.path.join(MUSIC, "out", "v3")
CACHE = os.path.join(MUSIC, "cache", "v3")
V3 = os.path.join(MUSIC, "v3")


def rng_for(name):
    return np.random.default_rng(zlib.crc32(("SOUND:" + name).encode()) % (2 ** 31))


# ------------------------------------------------------------------ sources
def src_path(ref):
    """'fs:499027' (Freesound original, else its HQ preview), 'el:B_roar_02' (ElevenLabs), 'sn:pine_branches'
    (a Sonniss GDC 2026 pick, a trimmed 48 kHz copy in cache/sound/sn/), 'vsco:<path>' (VSCO-2-CE, CC0)"""
    kind, name = ref.split(":", 1)
    if kind == "fs":
        d = os.path.join(LIB, "fs")
        for f in sorted(os.listdir(d)):
            if f.startswith(name + "_orig."):
                return os.path.join(d, f)
        p = os.path.join(d, name + "_hq.ogg")
        if os.path.exists(p):
            return p
    elif kind == "el":
        p = os.path.join(LIB, "el", name + ".wav")
        if os.path.exists(p):
            return p
    elif kind == "sn":
        p = os.path.join(LIB, "sn", name + ".wav")
        if os.path.exists(p):
            return p
    elif kind == "vsco":                       # 'vsco:Percussion/Anvil_Hit1_v3_Sum.wav': the score's CC0 sample library
        p = os.path.join(MUSIC, "samples", "VSCO-2-CE", name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(ref)


_LOADED = {}


def load(ref, t0=0.0, t1=None):
    """-> float32 [n, 2] at 48 kHz (resampled with a polyphase filter; mono duplicated; AmbiX B-format decoded to a
    pair of cardioids facing +-60 degrees)"""
    key = (ref, round(t0, 4), None if t1 is None else round(t1, 4))
    if key in _LOADED:
        return _LOADED[key]
    p = src_path(ref)
    info = sf.info(p)
    sr = info.samplerate
    margin = int(0.05 * sr) if sr != SR else 0
    a = max(0, int(round(t0 * sr)) - margin)
    b = None if t1 is None else int(round(t1 * sr)) + margin
    x, _ = sf.read(p, dtype="float32", always_2d=True, start=a, stop=b)
    if x.shape[1] >= 4:                        # AmbiX (ACN/SN3D): W, Y, Z, X
        W, Y, X = x[:, 0], x[:, 1], x[:, 3]
        th = np.deg2rad(60)
        L = 0.5 * (W + np.sin(th) * Y + np.cos(th) * X)
        R = 0.5 * (W - np.sin(th) * Y + np.cos(th) * X)
        x = np.stack([L, R], 1)
    elif x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    else:
        x = x[:, :2]
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
        head = int(round((t0 * sr - a) / sr * SR))
        x = x[head:]
        if t1 is not None:
            x = x[: int(round((t1 - t0) * SR))]
    x = np.ascontiguousarray(x, np.float32)
    _LOADED[key] = x
    return x


# ------------------------------------------------------------------ filters and loudness
def biquad(kind, f0, q=0.707, gain_db=0.0):
    """RBJ cookbook biquad at 48 kHz -> sos row"""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    c = np.cos(w)
    if kind == "lowshelf":
        b = [A * ((A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al), 2 * A * ((A - 1) - (A + 1) * c),
             A * ((A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al)]
        a = [(A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al]
    elif kind == "highshelf":
        b = [A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
             A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)]
        a = [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al]
    elif kind == "peak":
        b = [1 + al * A, -2 * c, 1 - al * A]
        a = [1 + al / A, -2 * c, 1 - al / A]
    else:
        raise ValueError(kind)
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return np.concatenate([b, a])[None, :]


def proc(y, hp=None, lp=None, shelves=(), hp_order=2, lp_order=2):
    """zero-phase-free (causal) filters: hp / lp Butterworth, then [(kind, f0, gain_db, q)] shelves / peaks"""
    y = np.asarray(y, np.float32)
    sos = []
    if hp:
        sos.append(signal.butter(hp_order, hp, "high", fs=SR, output="sos"))
    if lp:
        sos.append(signal.butter(lp_order, lp, "low", fs=SR, output="sos"))
    for s in shelves:
        kind, f0, g = s[:3]
        q = s[3] if len(s) > 3 else 0.707
        sos.append(biquad(kind, f0, q, g))
    if not sos:
        return y
    return signal.sosfilt(np.concatenate(sos, 0), y, axis=0).astype(np.float32)


_K = np.concatenate([np.array([[1.53512485958697, -2.69169618940638, 1.19839281085285,
                                1.0, -1.69065929318241, 0.73248077421585]]),
                     np.array([[1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]])], 0)


def kpow(y):
    z = signal.sosfilt(_K, np.asarray(y, np.float64), axis=0)
    return (z ** 2).sum(1)


def loud_int(y):
    """BS.1770 integrated loudness (gated), LUFS"""
    import pyloudnorm as pyln
    if len(y) < int(0.5 * SR):
        return loud_max(y)
    v = pyln.Meter(SR).integrated_loudness(np.asarray(y, np.float64))
    return float(v) if np.isfinite(v) else -120.0


def loud_max(y, win=0.4, hop=0.05):
    """the loudest 400 ms (momentary loudness max), LUFS"""
    p = kpow(y)
    n, h = int(win * SR), int(hop * SR)
    if len(p) < n:
        return -0.691 + 10 * np.log10(p.mean() + 1e-20)
    c = np.cumsum(np.concatenate([[0.0], p]))
    m = (c[n::h] - c[:-n:h]) / n
    return float(-0.691 + 10 * np.log10(m.max() + 1e-20))


def db(x):
    return 10 ** (x / 20)


def fades(y, fi=0.0, fo=0.0, shape="sin"):
    y = y.copy()
    n = len(y)
    if fi > 0:
        k = min(n, max(1, int(fi * SR)))
        r = np.linspace(0, 1, k, dtype=np.float32)
        y[:k] *= (np.sin(r * np.pi / 2) ** 2 if shape == "sin" else r)[:, None]
    if fo > 0:
        k = min(n, max(1, int(fo * SR)))
        r = np.linspace(1, 0, k, dtype=np.float32)
        y[-k:] *= (np.sin(r * np.pi / 2) ** 2 if shape == "sin" else r)[:, None]
    return y


def width(y, w):
    """stereo width: 0 = mono, 1 = as recorded"""
    m = y.mean(1, keepdims=True)
    return (m + (y - m) * w).astype(np.float32)


def pan(y, p):
    """constant-power balance of a stereo clip, p in -1..1"""
    th = (p + 1) * np.pi / 4
    gl, gr = np.cos(th) * np.sqrt(2), np.sin(th) * np.sqrt(2)
    return (y * np.array([gl, gr], np.float32)).astype(np.float32)


def limit_crest(y, crest_db, look=0.0015, release=0.04):
    """a transparent look-ahead peak limiter that holds a clip's peaks within crest_db of its loudest 400 ms.
    A real page's snap or a wood pop carries 18-25 dB of peak over its loudness (a synthesized one ~10 dB), so
    matched by loudness it would spike the master; limited this way it can sit at its designed level instead.
    Acts only on the rare peaks (1.5 ms look-ahead, 40 ms release): the crackle's texture is untouched."""
    if crest_db is None or len(y) < 16:
        return y
    from scipy.ndimage import minimum_filter1d
    thr = db(loud_max(y) + crest_db)
    a = np.abs(y).max(1)
    if a.max() <= thr:
        return y
    gr = 20 * np.log10(np.minimum(1.0, thr / (a + 1e-12)))
    gr = minimum_filter1d(gr, size=2 * int(look * SR) + 1)
    k = np.exp(-1.0 / (release * SR))
    sm = signal.lfilter([1 - k], [1, -k], gr)
    gr = np.minimum(gr, sm)
    return (y * db(gr).astype(np.float32)[:, None]).astype(np.float32)


# ------------------------------------------------------------------ building blocks
_SRC_L = {}
_SRC_L_PATH = os.path.join(LIB, "src_loudness.json")


# ADDITIVE (AP2, 29 Sep): a cut whose recipes set CACHE_READ_ONLY computes a cache miss in memory and never writes
# the shared caches in LIB (src_loudness.json, ref_levels_v2.json); every other cut caches as before
_CACHE_READ_ONLY = False


def src_loudness(ref, a, b):
    """integrated loudness of a source's usable range (cached): beds stitch sources at one common level"""
    global _SRC_L
    if not _SRC_L:
        try:
            _SRC_L = json.load(open(_SRC_L_PATH))
        except Exception:
            _SRC_L = {}
    k = f"{ref}:{a:.2f}:{b:.2f}"
    if k not in _SRC_L:
        _SRC_L[k] = loud_int(load(ref, a, b))
        if _CACHE_READ_ONLY:
            print(f"  src loudness {k}: computed, not cached (read-only caches)")
        else:
            json.dump(_SRC_L, open(_SRC_L_PATH, "w"))
    return _SRC_L[k]


def bed(spec, dur, rng):
    """a continuous bed of `dur` s stitched from recordings: segments of seg=(lo, hi) s chosen at random inside each
    source's usable range, each source first brought to one common loudness (-30 LUFS over its range), joined with
    equal-power crossfades of xf s (never the same stretch twice in a row).  spec["layers"] = [spec, ...] sums
    several such beds (each with its own gain g in dB): a fire's body and its crackle, a wind and its gusts."""
    if "layers" in spec:
        out = None
        for k, ly in enumerate(spec["layers"]):
            y = bed(ly, dur, np.random.default_rng(rng.integers(1 << 30) + k)) * np.float32(db(ly.get("g", 0.0)))
            out = y if out is None else out + y
        return proc(out, spec.get("hp"), spec.get("lp"), spec.get("shelves", ()))
    n = int(round(dur * SR))
    xf = spec.get("xf", 2.5)
    lo, hi = spec.get("seg", (8.0, 16.0))
    srcs = spec["src"]                             # [(ref, t0, t1, weight)]
    w = np.array([s[3] if len(s) > 3 else 1.0 for s in srcs], float)
    w /= w.sum()
    out = np.zeros((n + int(xf * SR) + 8, 2), np.float32)
    pos = 0
    k = int(xf * SR)
    last = None
    first = True
    while pos < n:
        i = rng.choice(len(srcs), p=w)
        ref, a, b = srcs[i][:3]
        norm = np.float32(db(-30.0 - src_loudness(ref, a, b)))
        remain = (n - pos) / SR + xf + 0.05
        seglen = max(0.5, min(rng.uniform(lo, hi), b - a - xf - 0.1, remain))
        for _ in range(8):
            st = rng.uniform(a, max(a, b - seglen - xf))
            if last is None or last[0] != ref or abs(st - last[1]) > seglen:
                break
        last = (ref, st)
        y = load(ref, st, st + seglen + xf) * norm
        m = min(len(y), len(out) - pos)
        y = y[:m]
        if first:
            out[:m] += y
            first = False
        else:
            kk = min(k, m)
            r = np.linspace(0, 1, kk, dtype=np.float32)
            gi, go = np.sin(r * np.pi / 2)[:, None], np.cos(r * np.pi / 2)[:, None]
            out[pos:pos + kk] = out[pos:pos + kk] * go + y[:kk] * gi
            out[pos + kk:pos + m] = y[kk:m]
        pos = pos + m - k if m > k else pos + m
    y = out[:n]
    return proc(y, spec.get("hp"), spec.get("lp"), spec.get("shelves", ()))


def event(spec, rng):
    """-> (clip [n, 2], hit index).  spec: src=(ref, hit_s_in_source), pre, post, fi, fo, hp, lp, shelves, or
    layers=[spec, ...] summed on their hits (each with its own gain `g` in dB and `dt` offset in s)"""
    if "layers" in spec:
        parts = [(event(s, rng), s.get("g", 0.0), s.get("dt", 0.0)) for s in spec["layers"]]
        pre = max(h / SR - dt for (c, h), g, dt in parts)
        end = max((len(c) - h) / SR + dt for (c, h), g, dt in parts)
        n = int(round((pre + end) * SR)) + 2
        y = np.zeros((n, 2), np.float32)
        hit = int(round(pre * SR))
        for (c, h), g, dt in parts:
            i0 = hit - h + int(round(dt * SR))
            y[i0:i0 + len(c)] += c * db(g)
        return y, hit
    ref, h = spec["src"]
    pre, post = spec.get("pre", 0.1), spec.get("post", 1.0)
    t0 = max(0.0, h - pre)
    y = load(ref, t0, h + post)
    hit = int(round((h - t0) * SR))
    y = proc(y, spec.get("hp"), spec.get("lp"), spec.get("shelves", ()))
    if spec.get("stretch"):                        # a gentle speed change (pitch follows, like tape)
        from math import gcd
        f = spec["stretch"]
        up, dn = int(round(1000 * f)), 1000
        g = gcd(up, dn)
        y = signal.resample_poly(y, up // g, dn // g, axis=0).astype(np.float32)
        hit = int(round(hit * f))
    y = fades(y, spec.get("fi", 0.005), spec.get("fo", 0.08))
    if spec.get("env"):                            # [(t_rel_to_hit_s, dB), ...]
        pts = np.array(spec["env"], float)
        tt = (np.arange(len(y)) - hit) / SR
        y = y * db(np.interp(tt, pts[:, 0], pts[:, 1]))[:, None].astype(np.float32)
    if spec.get("width") is not None:
        y = width(y, spec["width"])
    return y.astype(np.float32), hit


# ------------------------------------------------------------------ space (measured impulse responses)
def space_irs(name):
    import ir_v3
    return ir_v3.irs(name)


def convolve(y, irs):
    LL, LR, RL, RR = irs
    n = len(y)
    wl = signal.oaconvolve(y[:, 0], LL)[:n] + signal.oaconvolve(y[:, 1], RL)[:n]
    wr = signal.oaconvolve(y[:, 0], LR)[:n] + signal.oaconvolve(y[:, 1], RR)[:n]
    return np.stack([wl, wr], 1).astype(np.float32)


def distance(y, dist, irs):
    """dist 0 (close) .. 1 (far): air absorption, less direct sound, more of a MEASURED space, narrower image"""
    if not dist or dist <= 0.01:
        return y
    fc = 9000 * (1 - dist) ** 1.6 + 700
    y = proc(y, lp=fc)
    pad = np.zeros((int(1.5 * SR), 2), np.float32)
    z = np.concatenate([y, pad], 0)
    wet = convolve(z, irs)
    wet = proc(wet, lp=fc * 0.8)
    rd = np.sqrt((z ** 2).mean() / ((wet ** 2).mean() + 1e-12))
    out = z * (1.0 - 0.7 * dist) + wet * (0.3 + 0.8 * dist) * rd
    return width(out, 1 - 0.6 * dist)


# ------------------------------------------------------------------ the synthesized reference (level matching)
def loud_s3(y):
    return loud_max(y, win=3.0, hop=0.25)


def levels(y):
    return dict(I=loud_int(y), S3=loud_s3(y), M=loud_max(y), pk=float(20 * np.log10(np.abs(y).max() + 1e-12)))


def ref_level(item, kind):
    """COMPOSER's synthesized design for this cue, scaled exactly as sfx_v3.build_stem scales it:
    -> {I: integrated, S3: loudest 3 s, M: loudest 400 ms, pk: sample peak} (dB)"""
    import sfx_v3 as SX
    rng = SX.rng_for(item["id"])
    if kind == "bed":
        dur = item["t1"] - item["t0"]
        a, _ = SX.DESIGNS[item["fx"]](rng, dur, **item.get("params", {}))
        a = SX._norm_rms(SX._st(a)[: int(dur * SR)])
        a = a * np.float32(db(item.get("gain_db", 0)))
        return levels(a)
    a, hit = SX.DESIGNS[item["fx"]](rng, **item.get("params", {}))
    a = SX._st(a)
    if item.get("pan") is not None:
        import sfx as S1
        a = S1.panner(a.mean(1), item["pan"]).astype(np.float32)
    if item.get("dist"):
        a = SX.distance(a, item["dist"], rng)
    a = SX._norm_peak(a) * np.float32(db(item.get("gain_db", 0)))
    return levels(a)


_REF_CACHE = os.path.join(LIB, "ref_levels_v2.json")


def ref_levels(bm, items, ref_cut=None):
    """ref_cut (ADDITIVE, AP2): the cut whose designs these cues are (a pass 2 that plays another cut's effects table
    matches them to that cut's own cached reference levels: the same cue, the same design, the same numbers)"""
    try:
        cache = json.load(open(_REF_CACHE))
    except Exception:
        cache = {}
    out = {}
    changed = 0
    for it, kind in items:
        key = f"{ref_cut or bm.cut}:{it['id']}:" + json.dumps({k: it.get(k) for k in ("fx", "params", "gain_db", "pan",
                                                                                        "dist", "t0", "t1")},
                                                                 sort_keys=True)
        h = str(zlib.crc32(key.encode()))
        if h not in cache:
            cache[h] = ref_level(it, kind)
            changed += 1
        out[it["id"]] = cache[h]
    if changed and _CACHE_READ_ONLY:
        print(f"  ref levels: {changed} of {len(out)} cues computed, not cached (read-only caches)")
    elif changed:
        json.dump(cache, open(_REF_CACHE, "w"))
    return out


def match_gain(y, ref, kind, trim=0.0, peak_room=0.0):
    """the gain that puts a real recording at the synthesized design's level, never louder on ANY scale:
    beds by integrated loudness, capped so the loudest 3 s stays within 1 dB of the design's loudest 3 s;
    events by their loudest 400 ms, capped by the loudest 3 s and by the design's sample peak (+peak_room dB)
    (a real transient carries far more peak per unit of loudness than a synthesized one, and the master's
    limiter must not have to catch it).  `ref` may be a plain number (an EXTRA cue's absolute level)."""
    if not isinstance(ref, dict):
        return (ref + trim) - (loud_int(y) if kind == "bed" else loud_max(y))
    L = levels(y)
    if kind == "bed":
        # the loudest 3 s may not pass the design's loudest 3 s by more than 1 dB; a sparse synthesized crackle
        # can read LOWER on its loudest 3 s than its gated integrated loudness, so the cap never sits under I + 3
        return min(ref["I"] + trim - L["I"], max(ref["S3"], ref["I"] + 3.0) + 1.0 + trim - L["S3"])
    return min(ref["M"] + trim - L["M"], ref["S3"] + trim - L["S3"], ref["pk"] + peak_room + trim - L["pk"])


def sync_table(cut):
    d = json.load(open(os.path.join(V3, f"barmap_{cut}.json")))
    return {e["id"]: e for e in d["sync"]}


def apply_silence(stem, windows, fade=0.005):
    """a HARD silence in the effects stem: zero over each [t0, t1) s window, a 5 ms fade into it (C5: the shutdown
    3848 to the Ring's cut 4000; nothing else is touched). Windows come from a recipes module's SILENCE."""
    for t0, t1 in windows:
        i0, i1 = int(round(t0 * SR)), min(len(stem), int(round(t1 * SR)))
        k = min(int(fade * SR), i0)
        stem[i0 - k:i0] *= np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
        stem[i0:i1] = 0.0
    return stem


def build(cut, only=None, verbose=True):
    import importlib
    global _CACHE_READ_ONLY
    R = importlib.import_module(f"sound_recipes_{cut}")
    bm = BarMap(cut)
    ref_cut = getattr(R, "REF_LEVEL_CUT", None)         # ADDITIVE (AP2): see ref_levels
    _CACHE_READ_ONLY = bool(getattr(R, "CACHE_READ_ONLY", False))
    sync = sync_table(cut)
    total = bm.n + SR
    stem = np.zeros((total, 2), np.float32)            # near: events + beds
    send = np.zeros((total, 2), np.float32)            # to the measured outdoor space
    wins = R2.breath_windows(bm.breath_beats())
    irs_dist = space_irs(R.SPACE.get("distance", "forest20"))
    beds = [dict(b) for b in bm.d.get("ambience", [])] + [dict(b, extra=True) for b in R.EXTRA_BEDS]
    evs = [dict(e) for e in bm.d.get("sfx", [])] + [dict(e, extra=True) for e in R.EXTRA_EVENTS]
    if hasattr(R, "extra_cues"):                      # a composer's own in-memory effects (e.g. score_v3_C's C+.*)
        have = {x["id"] for x in beds + evs}
        fx, amb = R.extra_cues(bm)
        evs += [dict(e) for e in fx if e["id"] not in have]
        beds += [dict(b) for b in amb if b["id"] not in have]
    for b in beds:
        if b.get("extra"):
            b["t0"], b["t1"] = _resolve_t(b["t0"], sync), _resolve_t(b["t1"], sync)
    for e in evs:
        if e.get("extra"):
            e["t"] = _resolve_t(e["t"], sync)
    pic = getattr(R, "PICTURE", {})                    # SOUND-C: the cue's hit moved onto the MEASURED picture
    for e in evs:                                      # event (sound/picture_sync_<cut>.json); levels unaffected
        if e["id"] in pic:
            e["cue_t"], e["t"] = e["t"], pic[e["id"]] / 24.0
    live =lambda it: (not it.get("extra") and it["id"] in R.RECIPES and not R.RECIPES[it["id"]].get("skip")
                       and (not only or it["id"] in only or any(
                           R.RECIPES.get(o, {}).get("level_from") == it["id"] for o in only)))
    refs = ref_levels(bm, [(b, "bed") for b in beds if live(b)] + [(e, "event") for e in evs if live(e)], ref_cut)

    def ref_of(it, rc):
        if rc.get("level_from"):
            base = refs.get(rc["level_from"])
            if base is None:
                src = next(x for x in evs + beds if x["id"] == rc["level_from"])
                base = ref_levels(bm, [(src, "bed" if "t0" in src else "event")], ref_cut)[rc["level_from"]]
            return base
        if it.get("extra"):
            return rc["level"]
        return refs[it["id"]]
    meta = []
    for b in beds:
        rid = b["id"]
        if only and rid not in only:
            continue
        rc = R.RECIPES.get(rid)
        if rc is None or rc.get("skip"):
            meta.append(dict(id=rid, kind="bed", status="SKIP" if rc else "NO RECIPE", why=(rc or {}).get("why", "")))
            continue
        rng = rng_for(rid)
        # A polish (polishdoom): timing is applied after reference lookup, so the design's measured
        # loudness stays the reference when a bed is given a longer release.
        b.update(getattr(R, "BED_TIMING", {}).get(rid, {}))
        dur = b["t1"] - b["t0"]
        y = limit_crest(bed(rc, dur, rng), rc.get("crest", R.SPACE.get("bed_crest", 18.0)))
        ref = ref_of(b, rc)
        g = match_gain(y, ref, "bed", rc.get("trim", 0.0))
        target = (ref["I"] if isinstance(ref, dict) else ref) + rc.get("trim", 0.0)
        y = y * np.float32(db(g))
        fi, fo = b.get("fade_in", 1.5), b.get("fade_out", 1.5)
        tt = np.arange(len(y)) / SR
        env = np.clip(tt / max(fi, 1e-3), 0, 1) * np.clip((dur - tt) / max(fo, 1e-3), 0, 1)
        env = np.sin(env * np.pi / 2) ** 2
        if b.get("env"):
            pts = np.array(b["env"], float)
            env = env * db(np.interp(tt, pts[:, 0], pts[:, 1]))
        if rc.get("env"):
            pts = np.array(rc["env"], float)
            env = env * db(np.interp(tt, pts[:, 0], pts[:, 1]))
        y = y * env[:, None].astype(np.float32)
        if rc.get("pan") is not None:
            y = pan(y, rc["pan"])
        if rc.get("width") is not None:
            y = width(y, rc["width"])
        i0 = int(round(b["t0"] * SR))
        e = min(total, i0 + len(y))
        y = y[: e - i0]
        if wins and not rc.get("no_breath"):
            y = y * R2.breath_env(len(y), wins, offset=i0)[:, None]
        stem[i0:e] += y
        send[i0:e] += y * np.float32(rc.get("send", R.SPACE.get("bed_send", 0.0)))
        meta.append(dict(id=rid, kind="bed", t0=round(b["t0"], 3), t1=round(b["t1"], 3), level=round(target, 1),
                         gain=round(float(g), 1),
                         src=sorted({x[0] for ly in rc.get("layers", [rc]) for x in ly["src"]})))
    for ev in evs:
        rid = ev["id"]
        if only and rid not in only:
            continue
        rc = R.RECIPES.get(rid)
        if rc is None or rc.get("skip"):
            meta.append(dict(id=rid, kind="event", status="SKIP" if rc else "NO RECIPE", why=(rc or {}).get("why", "")))
            continue
        rng = rng_for(rid)
        y, hit = event(rc, rng)
        if rc.get("env_after"):                   # ADDITIVE (owner night, 29 Sep): [(t_rel_to_hit_s, dB)] over the whole
            pts = np.array(rc["env_after"], float)          # event, layers included; inert unless a recipe sets it
            tt = (np.arange(len(y)) - hit) / SR
            y = y * db(np.interp(tt, pts[:, 0], pts[:, 1]))[:, None].astype(np.float32)
        if rc.get("pan") is not None:
            y = pan(y, rc["pan"])
        dist = rc.get("dist", ev.get("dist"))
        if dist:
            y = distance(y, dist, irs_dist)
        y = limit_crest(y, rc.get("crest", R.SPACE.get("event_crest")))
        ref = ref_of(ev, rc)
        g = match_gain(y, ref, "event", rc.get("trim", 0.0), rc.get("peak_room", R.SPACE.get("peak_room", 0.0)))
        target = (ref["M"] if isinstance(ref, dict) else ref) + rc.get("trim", 0.0)
        y = y * np.float32(db(g))
        i0 = int(round(ev["t"] * SR)) - hit
        if i0 < 0:
            y, i0 = y[-i0:], 0
        e = min(total, i0 + len(y))
        y = y[: e - i0]
        if wins and not rc.get("no_breath"):      # ADDITIVE (owner night, 29 Sep): a row may sound through a breath
            y = y * R2.breath_env(len(y), wins, offset=i0)[:, None]
        stem[i0:e] += y
        send[i0:e] += y * np.float32(rc.get("send", R.SPACE.get("event_send", 0.0)))
        meta.append(dict(id=rid, kind="event", t=round(ev["t"], 4), level=round(target, 1), gain=round(float(g), 1),
                         clip_start_s=round(i0 / SR, 4), hit_offset_s=round(hit / SR, 4),
                         src=[s["src"][0] for s in rc["layers"]] if "layers" in rc else [rc["src"][0]]))
    if np.any(send):
        irs = space_irs(R.SPACE.get("outdoor", "forest20"))
        wet = R2.convolve_with_breaths(send, irs, wins, total) if wins else convolve(send, irs)
        stem = stem + proc(wet, hp=R.SPACE.get("wet_hp", 150), lp=R.SPACE.get("wet_lp", 9000))
    stem = proc(stem, hp=R.SPACE.get("stem_hp", 25), hp_order=2)
    stem = apply_silence(stem, getattr(R, "SILENCE", ()))   # ADDITIVE (SOUND-SCORE-C): inert unless a recipe sets it
    if verbose:
        for m in meta:
            if m.get("status"):
                print(f"  {m['id']:22s} {m['status']:9s} {m.get('why', '')}")
    return bm, stem, meta


def levels_report(cut, only=None):
    """per cue: the synthesized design's levels vs the real clip's after matching, and which cap decided the gain"""
    import importlib
    R = importlib.import_module(f"sound_recipes_{cut}")
    bm = BarMap(cut)
    beds = [dict(b) for b in bm.d.get("ambience", [])]
    evs = [dict(e) for e in bm.d.get("sfx", [])]
    if hasattr(R, "extra_cues"):
        fx, amb = R.extra_cues(bm)
        have = {x["id"] for x in beds + evs}
        evs += [dict(e) for e in fx if e["id"] not in have]
        beds += [dict(b) for b in amb if b["id"] not in have]
    live = lambda x: x["id"] in R.RECIPES and not R.RECIPES[x["id"]].get("skip") and (not only or x["id"] in only)
    global _CACHE_READ_ONLY
    _CACHE_READ_ONLY = bool(getattr(R, "CACHE_READ_ONLY", False))
    refs = ref_levels(bm, [(b, "bed") for b in beds if live(b)] + [(e, "event") for e in evs if live(e)],
                      getattr(R, "REF_LEVEL_CUT", None))
    irs_dist = space_irs(R.SPACE.get("distance", "forest20"))
    print(f"{'cue':20s} {'kind':5s} | synth I / S3 / M / pk       | real, after gain: I / S3 / M / pk | gain  bound")
    for it, kind in [(b, "bed") for b in beds] + [(e, "event") for e in evs]:
        rid = it["id"]
        if not live(it) or rid not in refs:
            continue
        rc = R.RECIPES[rid]
        rng = rng_for(rid)
        if kind == "bed":
            y = limit_crest(bed(rc, it["t1"] - it["t0"], rng), rc.get("crest", R.SPACE.get("bed_crest", 18.0)))
        else:
            y, _ = event(rc, rng)
            if rc.get("pan") is not None:
                y = pan(y, rc["pan"])
            d = rc.get("dist", it.get("dist"))
            if d:
                y = distance(y, d, irs_dist)
            y = limit_crest(y, rc.get("crest", R.SPACE.get("event_crest")))
        r = refs[rid]
        L = levels(y)
        tr = rc.get("trim", 0.0)
        if kind == "bed":
            cands = {"I": r["I"] + tr - L["I"], "S3cap": max(r["S3"], r["I"] + 3.0) + 1.0 + tr - L["S3"]}
        else:
            pr = rc.get("peak_room", R.SPACE.get("peak_room", 0.0))
            cands = {"M": r["M"] + tr - L["M"], "S3cap": r["S3"] + tr - L["S3"], "pkcap": r["pk"] + pr + tr - L["pk"]}
        bound = min(cands, key=cands.get)
        g = cands[bound]
        print(f"{rid:20s} {kind:5s} | {r['I']:6.1f} {r['S3']:6.1f} {r['M']:6.1f} {r['pk']:6.1f} | "
              f"{L['I'] + g:6.1f} {L['S3'] + g:6.1f} {L['M'] + g:6.1f} {L['pk'] + g:6.1f} | {g:6.1f} {bound}")


def write(cut, stem, meta, bm, allow_unverified=False):
    """the pre-master stem, then the REAL master: COMPOSER's own chain (render_v3.master: -16 LUFS, glue comp,
    limiter with the score-stem ceiling, fades, true-peak safety) run on the saved pre-master score of
    final_<cut> + this stem -> sound_<cut>.wav = sound_<cut>_score.wav + sound_<cut>_sfx.wav; sfx_<cut>.wav is
    the effects stem of that master (a copy of sound_<cut>_sfx.wav)"""
    n = bm.n
    pre = np.ascontiguousarray(stem[:n], np.float32)
    sf.write(os.path.join(OUT, f"sfxpre_{cut}.wav"), pre, SR, subtype="FLOAT")
    json.dump(meta, open(os.path.join(MUSIC, "sound", f"events_{cut}.json"), "w"), indent=1)
    sp = os.path.join(CACHE, f"premaster_score_final_{cut}.npy")
    if not os.path.exists(sp):
        print(f"  no {os.path.basename(sp)} yet: pre-master stem only")
        return pre, None
    import shutil
    import audio_guard_v3 as AG          # SOUND-SCORE-C: a saved premaster must provably be this cut's current one
    try:
        for w in AG.check_premaster(sp, cut, bm.frames, bm.path, os.path.join(V3, f"cues_{cut}.json"),
                                    allow_unverified):
            print("  " + w)
    except AG.StaleArtefact as e:
        raise SystemExit(f"REFUSED: {e}")
    import render_v3 as RV
    score = np.load(sp)
    import importlib
    recipe = importlib.import_module(f"sound_recipes_{cut}")
    if hasattr(recipe, "prepare_master"):
        score, pre = recipe.prepare_master(cut, score, pre)
    # polishedge: AP2's A6/A7 breath recovery and dynamics (sound_polish_edge), after the ignition's premaster edit
    # (the two touch disjoint frames); the source caches stay intact, and both enter before peak guard and mastering
    polish = getattr(recipe, "PREMASTER_POLISH", None)
    if polish is not None:
        score, pre = polish(score, pre, bm)
    # polishdoom: bounded fader rides on the score (the piano's first note near 2860); the saved note render is untouched
    if hasattr(recipe, "polish_score"):
        recipe.polish_score(score)
    G = db(-16.0 - RV.lufs(score[:n] + pre)) * db(1.2)          # the master's gain, a little on the safe side
    pre = peak_guard(score[:n], pre, G)
    master_options = {}
    if hasattr(recipe, "bound_master"):
        pre, master_options = recipe.bound_master(cut, pre)
    sf.write(os.path.join(OUT, f"sfxpre_{cut}.wav"), pre, SR, subtype="FLOAT")
    RV.master(score, pre, n, f"sound_{cut}", **master_options)
    for f in (f"manifest_final_{cut}.json", f"breath_probe_final_{cut}.json"):
        src = os.path.join(CACHE, f)
        if os.path.exists(src):
            dst = os.path.join(CACHE, f.replace("final_", "sound_"))
            if f.startswith("breath_probe_") and hasattr(recipe, "refresh_breath_probes"):
                probes = recipe.refresh_breath_probes(cut, score, json.load(open(src)))
                json.dump(probes, open(dst, "w"), indent=1)
            else:
                shutil.copyfile(src, dst)
    del score
    shutil.copyfile(os.path.join(OUT, f"sound_{cut}_sfx.wav"), os.path.join(OUT, f"sfx_{cut}.wav"))
    post, _ = sf.read(os.path.join(OUT, f"sfx_{cut}.wav"), dtype="float32", always_2d=True)
    return pre, post


def peak_guard(score, sfx, G, ceil_db=-2.3, floor_db=-10.0):
    """pre-master: duck the EFFECTS only, where score*G + sfx*G would pass ceil_db (the master limiter's -1.3 dBFS
    ceiling less 1 dB for the glue compressor and inter-sample peaks), so the limiter never has to act on the effects
    (a limiter reacting to a crackle imprints a click on the score stem).  Where the score alone already passes the
    ceiling, the effects are left alone (the limiter acts there as it always did).  5 ms look-ahead, 120 ms release,
    at most floor_db of ducking."""
    from scipy.ndimage import minimum_filter1d
    lim = db(ceil_db) / G
    s_abs = np.abs(score).max(1)
    tot = np.abs(score + sfx).max(1)
    over = (tot > lim) & (s_abs < lim)
    if not np.any(over):
        print("  peak guard: effects never push the master into its limiter")
        return sfx
    x_abs = np.abs(sfx).max(1) + 1e-12
    need = np.ones(len(sfx), np.float32)
    need[over] = np.clip((lim * 0.97 - s_abs[over]) / x_abs[over], db(floor_db), 1.0)
    gr = 20 * np.log10(need)
    gr = minimum_filter1d(gr, size=int(0.010 * SR) + 1)                    # +-5 ms: the attack arrives early
    a = np.exp(-1.0 / (0.12 * SR))
    sm = signal.lfilter([1 - a], [1, -a], gr)                             # the release
    gr = np.minimum(gr, sm)
    idx = np.where(over)[0]
    brk = np.where(np.diff(idx) > int(0.05 * SR))[0]
    spans = [(idx[a] / SR, idx[b] / SR) for a, b in zip(np.r_[0, brk + 1], np.r_[brk, len(idx) - 1])]
    print(f"  peak guard: {int(over.sum())} samples ({over.sum() / SR * 1000:.0f} ms) would drive the limiter; "
          f"effects ducked there by up to {-gr.min():.1f} dB, at " +
          ", ".join(f"{a:.2f}-{b:.2f} s" for a, b in spans[:8]) + (" ..." if len(spans) > 8 else ""))
    return (sfx * db(gr).astype(np.float32)[:, None]).astype(np.float32)


def report(cut, pre, post):
    import pyloudnorm as pyln
    m = pyln.Meter(SR)
    L = lambda x: m.integrated_loudness(np.asarray(x, np.float64))
    op = os.path.join(OUT, f"final_{cut}_sfx.wav")
    mix, _ = sf.read(os.path.join(OUT, f"sound_{cut}.wav"), dtype="float32", always_2d=True)
    if os.path.exists(op):
        old, _ = sf.read(op, dtype="float32", always_2d=True)
        print(f"  real sfx stem {L(post):6.2f} LUFS  TP {R2.true_peak_db(post):6.2f} dBTP   | old synthesized sfx stem "
              f"{L(old):6.2f} LUFS")
    else:                                   # a cut scored without synthesized effects (C5, C5P2)
        print(f"  real sfx stem {L(post):6.2f} LUFS  TP {R2.true_peak_db(post):6.2f} dBTP   | (no synthesized stem)")
    print(f"  sound_{cut}.wav (score + real sfx, mastered): {L(mix):6.2f} LUFS  TP {R2.true_peak_db(mix):6.2f} dBTP")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cut")
    ap.add_argument("--only", default="")
    ap.add_argument("--solo", default="")
    ap.add_argument("--no-battery", action="store_true")
    ap.add_argument("--levels", action="store_true")
    ap.add_argument("--retired-c-7200", action="store_true", help="allow the retired 7,200-frame C (writes sound_C.*)")
    ap.add_argument("--premaster-unverified", action="store_true",
                    help="accept a saved premaster with no identity sidecar for a strict cut (C5P2)")
    a = ap.parse_args()
    cut = a.cut.upper()
    only = set(x for x in a.only.split(",") if x) or None
    if a.levels:
        levels_report(cut, only)
        sys.exit(0)
    import audio_guard_v3 as AG
    if not a.solo:                          # SOUND-SCORE-C: the retired cut's outputs are what the edit picks by name
        try:
            AG.check_cut_allowed(cut, a.retired_c_7200)
        except AG.StaleArtefact as e:
            sys.exit(f"REFUSED: {e}")
    bm, stem, meta = build(cut, only)
    if a.solo:
        sf.write(a.solo, stem[: bm.n], SR, subtype="PCM_24")
        print("solo ->", a.solo)
        sys.exit(0)
    pre, post = write(cut, stem, meta, bm, allow_unverified=a.premaster_unverified)
    if post is not None:
        report(cut, pre, post)
        if not a.no_battery:
            import analyze_v3 as A
            S = __import__(f"score_v3_{cut}").build(bm)
            A.main(bm, S, f"sound_{cut}")
