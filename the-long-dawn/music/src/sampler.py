"""VSCO-2-CE sampler.

* SFZ-mapped patches (key ranges, velocity layers, round robins) + manual
  drum "kits" for un-mapped percussion.
* Velocity layers are crossfaded (equal power) - statically from the note
  velocity or dynamically from the part's dynamics curve (CC1-style).
* Round robins rotate per key; repeated notes also get micro detune/gain/
  start-offset variation.
* Pitch shift + 44.1k->48k conversion in one soxr pass (only a few semitones:
  the SFZ key ranges keep shifts small).
* Sample onsets are detected once and aligned to the note time, so hits
  land on the frame.  Sustained articulations can be anticipated a little so
  the perceived attack lands on the beat.
* Notes longer than the recording are extended with crossfaded sustain
  segments; note-off applies a release envelope; legato notes skip the
  attack and fade in under the previous note's release.
"""
import glob
import hashlib
import json
import os
import re
from collections import OrderedDict

import numpy as np
import soundfile as sf
import soxr

from dsl import SR, BEAT_S, dyn_array, level_db
from sfz import parse_sfz, VSCO, MUSIC

CACHE = os.path.join(MUSIC, "cache")
META_PATH = os.path.join(CACHE, "sample_meta.json")
os.makedirs(CACHE, exist_ok=True)

# ---------------------------------------------------------------------------
# patches
# ---------------------------------------------------------------------------
# kind: 'sus' (held, released at note-off, extendable), 'one' (one-shot,
# rings naturally), 'pno' (one-shot but damped at note-off)
P = {}


def patch(key, sfz=None, kind="sus", rel=0.35, antic=0.0, lskip=0.07,
          lfade=0.07, gain=0.0, fadein=0.004, maxlen=None, kit=None,
          vary=True):
    P[key] = dict(sfz=sfz, kind=kind, rel=rel, antic=antic, lskip=lskip,
                  lfade=lfade, gain=gain, fadein=fadein, maxlen=maxlen,
                  kit=kit, vary=vary)


# woodwinds
patch("flute", "FluteSusVib", rel=0.30, antic=0.035, lskip=0.09, lfade=0.06)
patch("flute_nv", "FluteSusNV", rel=0.30, antic=0.035, lskip=0.09, lfade=0.06)
patch("flute_exp", "FluteExpVib", rel=0.30, antic=0.035, lskip=0.09)
patch("flute_stac", "FluteStac", kind="one")
patch("oboe", "OboeSusVib", rel=0.25, antic=0.03)
patch("clarinet", "ClarinetSus", rel=0.25, antic=0.03)
patch("bassoon", "BassoonSus", rel=0.25, antic=0.03)
# brass
patch("horn", "FHornSus", rel=0.35, antic=0.04, lskip=0.08, lfade=0.07)
patch("horn_stac", "FHornStac", kind="one")
patch("trumpet", "TrumpetSus", rel=0.30, antic=0.03, lskip=0.07)
patch("trumpet_vib", "TrumpetSusVib", rel=0.30, antic=0.03, lskip=0.07)
patch("trumpet_stac", "TrumpetStac", kind="one")
patch("trombone", "TromboneSus", rel=0.35, antic=0.035, lskip=0.08)
patch("trombone_stac", "TromboneStac", kind="one")
patch("tuba", "TubaSus", rel=0.4, antic=0.04)
patch("tuba_stac", "TubaStac", kind="one")
# strings
patch("vln", "ViolinEnsSusVib", rel=0.45, antic=0.06, lskip=0.10, lfade=0.09)
patch("vln_spic", "ViolinEnsSpic", kind="one", maxlen=1.6)
patch("vln_trem", "ViolinEnsTrem", rel=0.4, antic=0.04)
patch("vln_pizz", "ViolinEnsPizz", kind="one")
patch("vla", "ViolaEnsSusVib", rel=0.45, antic=0.06, lskip=0.10, lfade=0.09)
patch("vla_spic", "ViolaEnsSpic", kind="one", maxlen=1.6)
patch("vla_trem", "ViolaEnsTrem", rel=0.4, antic=0.04)
patch("vc", "CelloEnsSusVib", rel=0.5, antic=0.06, lskip=0.10, lfade=0.09)
patch("vc_spic", "CelloEnsSpic", kind="one", maxlen=1.8)
patch("vc_trem", "CelloEnsTrem", rel=0.45, antic=0.04)
patch("vc_pizz", "CelloEnsPizz", kind="one")
patch("cb", "ContrabassSusVB", rel=0.5, antic=0.06, lskip=0.10)
patch("cb_nv", "ContrabassSusNV", rel=0.5, antic=0.06, lskip=0.10)
patch("cb_spic", "ContrabassSpic", kind="one", maxlen=1.8)
patch("cb_trem", "ContrabassTrem", rel=0.45, antic=0.04)
patch("cb_pizz", "ContrabassPizz", kind="one")
patch("svln", "SViolinVib", rel=0.4, antic=0.05, lskip=0.10)
patch("harp", "Harp", kind="one", maxlen=9.0)
# keys / pitched percussion
patch("piano", "UprightPiano", kind="pno", rel=0.5)
patch("organ", "OrganLoud", rel=0.6, antic=0.02)
patch("organ_q", "OrganQuiet", rel=0.6, antic=0.02)
patch("organ_ped", "OrganLoudPedal", rel=0.7, antic=0.02)
patch("glock", "Glockenspiel", kind="one", maxlen=6.0)
patch("tubular", "TubularBells", kind="one", maxlen=14.0)
patch("timp", "Timpani", kind="one", maxlen=8.0)
patch("timp_roll", "TimpaniRolls", rel=0.5, antic=0.0, vary=False)
patch("marimba", "Marimba", kind="one")

# ---- drum kits: list of (glob, lovel, hivel) relative to VSCO root --------
PERC = "Percussion/"
V1 = "VSCO 1 Percussion/"
KITS = {
    "bdrum": [(PERC + f"BDrumNewhit_v{i}_rr*_Sum.wav", int((i - 1) * 127 / 7),
               int(i * 127 / 7)) for i in range(1, 8)],
    "bdrum2": [(V1 + "drums/bass/bdrum_ppp_*.wav", 0, 30),
               (V1 + "drums/bass/bdrum_pp_*.wav", 31, 50),
               (V1 + "drums/bass/bdrum_mp_*.wav", 51, 75),
               (V1 + "drums/bass/bdrum_f_*.wav", 76, 95),
               (V1 + "drums/bass/bdrum_ff_*.wav", 96, 110),
               (V1 + "drums/bass/bdrum_fff_*.wav", 111, 127)],
    "giant_mallet": [(V1 + "drums/other/ethnic/giant/mallet/*_pp_*.wav", 0, 50),
                     (V1 + "drums/other/ethnic/giant/mallet/*_mf_*.wav", 51, 80),
                     (V1 + "drums/other/ethnic/giant/mallet/*_f_*.wav", 81, 100),
                     (V1 + "drums/other/ethnic/giant/mallet/*_ff_*.wav", 101, 127)],
    "giant_sticks": [(V1 + "drums/other/ethnic/giant/sticks/*_ppp_*.wav", 0, 25),
                     (V1 + "drums/other/ethnic/giant/sticks/*_pp_*.wav", 26, 40),
                     (V1 + "drums/other/ethnic/giant/sticks/*_p_*.wav", 41, 55),
                     (V1 + "drums/other/ethnic/giant/sticks/*_mp_*.wav", 56, 70),
                     (V1 + "drums/other/ethnic/giant/sticks/*_mf_*.wav", 71, 90),
                     (V1 + "drums/other/ethnic/giant/sticks/*_f_*.wav", 91, 110),
                     (V1 + "drums/other/ethnic/giant/sticks/*_fff_*.wav", 111, 127)],
    "giant_hand": [(V1 + "drums/other/ethnic/giant/hand/*_hit_ppp_*.wav", 0, 25),
                   (V1 + "drums/other/ethnic/giant/hand/*_hit_pp_*.wav", 26, 45),
                   (V1 + "drums/other/ethnic/giant/hand/*_hit_p_*.wav", 46, 60),
                   (V1 + "drums/other/ethnic/giant/hand/*_hit_mp_*.wav", 61, 80),
                   (V1 + "drums/other/ethnic/giant/hand/*_hit_f_*.wav", 81, 105),
                   (V1 + "drums/other/ethnic/giant/hand/*_hit_ff_*.wav", 106, 127)],
    "tenor_lo": [(V1 + "drums/tenor/tenor_lower/tenor_ppp_*.wav", 0, 20),
                 (V1 + "drums/tenor/tenor_lower/tenor_pp_*.wav", 21, 40),
                 (V1 + "drums/tenor/tenor_lower/tenor_mp_*.wav", 41, 60),
                 (V1 + "drums/tenor/tenor_lower/tenor_mf_*.wav", 61, 80),
                 (V1 + "drums/tenor/tenor_lower/tenor_f_*.wav", 81, 95),
                 (V1 + "drums/tenor/tenor_lower/tenor_ff_*.wav", 96, 110),
                 (V1 + "drums/tenor/tenor_lower/tenor_fff_*.wav", 111, 127)],
    "tenor_hi": [(V1 + "drums/tenor/tenor_higher/tenorH_pp_*.wav", 0, 30),
                 (V1 + "drums/tenor/tenor_higher/tenorH_p_*.wav", 31, 50),
                 (V1 + "drums/tenor/tenor_higher/tenorH_mp_*.wav", 51, 65),
                 (V1 + "drums/tenor/tenor_higher/tenorH_mf_*.wav", 66, 80),
                 (V1 + "drums/tenor/tenor_higher/tenorH_f_*.wav", 81, 95),
                 (V1 + "drums/tenor/tenor_higher/tenorH_ff_*.wav", 96, 112),
                 (V1 + "drums/tenor/tenor_higher/tenorH_fff_*.wav", 113, 127)],
    "snare": [(PERC + "Snare2-HitNS_v1_rr*_Sum.wav", 0, 40),
              (PERC + "Snare2-HitNS_v3_rr*_Sum.wav", 41, 70),
              (PERC + "Snare2-HitNS_v5_rr*_Sum.wav", 71, 100),
              (PERC + "Snare2-HitNS_v6_rr*_Sum.wav", 101, 127)],
    "snare_roll": [(PERC + "Snare2-rollNS_v1_rr1_Sum.wav", 0, 50),
                   (PERC + "Snare2-rollNS_v3_rr1_Sum.wav", 51, 90),
                   (PERC + "Snare2-rollNS_v5_rr1_Sum.wav", 91, 127)],
    "crash": [(PERC + "cymbal-crash1_pp_rr*.wav", 0, 40),
              (PERC + "cymbal-crash1_mp_rr*.wav", 41, 70),
              (PERC + "cymbal-crash1_mf_rr*.wav", 71, 100),
              (PERC + "cymbal-crash1_ff_rr*.wav", 101, 127)],
    "clash": [(V1 + "varMetal/Cymbals/clash/crash_hit_pp_loose.wav", 0, 40),
              (V1 + "varMetal/Cymbals/clash/crash_hit_mp_loose.wav", 41, 80),
              (V1 + "varMetal/Cymbals/clash/crash_hit_ff_loose.wav", 81, 110),
              (V1 + "varMetal/Cymbals/clash/crash_hit_fff_loose*.wav", 111, 127)],
    "suscym": [(PERC + "susCymb1-hit_pp_rr*.wav", 0, 40),
               (PERC + "susCymb1-hit_mp_rr*.wav", 41, 75),
               (PERC + "susCymb1-hit_f_rr*.wav", 76, 105),
               (PERC + "susCymb1-hit_fff_rr*.wav", 106, 127)],
    "gong": [(PERC + "gongHit_p.wav", 0, 45), (PERC + "gongHit_mf.wav", 46, 80),
             (PERC + "gongHit_f.wav", 81, 110), (PERC + "gongHit_fff.wav", 111, 127)],
    "triangle": [(PERC + "Triangle3-Hit_v1_rr*_Sum.wav", 0, 70),
                 (PERC + "Triangle3-Hit_v2_rr*_Sum.wav", 71, 127)],
    "belltree": [(PERC + "BellTree_Stroke*_v1_Sum.wav", 0, 127)],
    # swells (special: aligned by their loudness peak, see render)
    "cym_swell_long": [(PERC + "susCymb1-cresc-Long_v1.wav", 0, 127)],
    "cym_swell_med": [(PERC + "susCymb1-cresc-Median_v1.wav", 0, 127)],
    "cym_swell_short": [(PERC + "susCymb1-cresc-Short_v1.wav", 0, 127)],
    "cym_roll_cresc": [(V1 + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", 0, 127)],
    "bdrum_roll": [(V1 + "drums/bass/bdrum_roll_long1.wav", 0, 127)],
}
for k in KITS:
    patch(k, kit=k, kind="one", maxlen=None, vary=True)
P["bdrum_roll"]["kind"] = "sus"
P["bdrum_roll"]["rel"] = 0.8
for k in ("cym_swell_long", "cym_swell_med", "cym_swell_short", "cym_roll_cresc"):
    P[k]["align"] = "peak"
P["crash"]["maxlen"] = 9.0
P["gong"]["maxlen"] = 16.0

# ---------------------------------------------------------------------------
# sample metadata (onset, loudness reference)
# ---------------------------------------------------------------------------
_META = None


def _load_meta():
    global _META
    if _META is None:
        _META = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    return _META


def _save_meta():
    tmp = META_PATH + f".{os.getpid()}.tmp"
    json.dump(_META, open(tmp, "w"))
    os.replace(tmp, META_PATH)


_RAW = OrderedDict()
_RAW_BYTES = [0]
RAW_BUDGET = 350e6


def load_raw(path):
    if path in _RAW:
        _RAW.move_to_end(path)
        return _RAW[path]
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    _RAW[path] = (x, sr)
    _RAW_BYTES[0] += x.nbytes
    while _RAW_BYTES[0] > RAW_BUDGET and len(_RAW) > 1:
        _, (y, _) = _RAW.popitem(last=False)
        _RAW_BYTES[0] -= y.nbytes
    return x, sr


def meta(path):
    """onset (s), loudness ref (dB, max 50ms RMS in first 3 s), peak time,
    active end (s, -40 dB)."""
    M = _load_meta()
    k = os.path.relpath(path, VSCO)
    if k in M:
        return M[k]
    x, sr = load_raw(path)
    mono = np.abs(x).max(axis=1)
    w = max(1, int(sr * 0.002))
    env = np.convolve(mono, np.ones(w) / w, mode="same")
    pk = env.max() + 1e-9
    above = np.where(env > pk * 10 ** (-32 / 20))[0]
    on = int(above[0]) if len(above) else 0
    # refine: walk back to where env is < -50 dB of peak (start of attack)
    j = on
    floor = pk * 10 ** (-50 / 20)
    while j > 0 and env[j] > floor and on - j < int(0.02 * sr):
        j -= 1
    onset = j / sr
    w50 = int(sr * 0.05)
    seg = (x[on:on + int(3 * sr)] ** 2).mean(axis=1)
    if len(seg) > w50:
        rms = np.sqrt(np.convolve(seg, np.ones(w50) / w50, mode="valid"))
        L = 20 * np.log10(rms.max() + 1e-9)
        pkt = (on + int(np.argmax(rms)) + w50 // 2) / sr
    else:
        L = 20 * np.log10(np.sqrt(seg.mean()) + 1e-9)
        pkt = onset
    # loudness peak over whole file (for swells)
    segw = (x ** 2).mean(axis=1)
    w200 = int(sr * 0.2)
    if len(segw) > w200:
        r2 = np.convolve(segw, np.ones(w200) / w200, mode="same")
        gpeak = int(np.argmax(r2)) / sr
    else:
        gpeak = pkt
    act = np.where(env > pk * 10 ** (-40 / 20))[0]
    end = (act[-1] / sr) if len(act) else len(x) / sr
    Lg = 20 * np.log10(np.sqrt(r2.max()) + 1e-9) if len(segw) > w200 else L
    M[k] = dict(onset=onset, L=float(L), Lg=float(Lg), pk=pkt, gpeak=gpeak,
                end=end, dur=len(x) / sr, sr=sr)
    return M[k]


# ---------------------------------------------------------------------------
# region lookup
# ---------------------------------------------------------------------------
_REG = {}


def regions(pkey):
    if pkey in _REG:
        return _REG[pkey]
    p = P[pkey]
    if p["kit"]:
        regs = []
        for pat, lo, hi in KITS[p["kit"]]:
            files = sorted(glob.glob(os.path.join(VSCO, pat)))
            if not files:
                raise FileNotFoundError(pat)
            for i, f in enumerate(files):
                regs.append(dict(path=f, lokey=0, hikey=127, center=60, lovel=lo,
                                 hivel=hi, volume=0.0, tune=0.0, transpose=0,
                                 seq_length=len(files), seq_position=i + 1,
                                 release=0.5, offset=0, pitch_keytrack=0))
    else:
        regs = parse_sfz(p["sfz"])
    _REG[pkey] = regs
    return regs


def pick(pkey, pitch):
    """Return (layers, shift_of_each) for a pitch: layers = list of
    (vcenter, [regions for rr]) sorted by velocity."""
    regs = regions(pkey)
    ip = int(round(pitch))
    cand = [r for r in regs if r["lokey"] <= ip <= r["hikey"]]
    if not cand:
        # nearest by key range
        def dist(r):
            return 0 if r["lokey"] <= ip <= r["hikey"] else min(abs(ip - r["lokey"]), abs(ip - r["hikey"]))
        dmin = min(dist(r) for r in regs)
        cand = [r for r in regs if dist(r) == dmin]
    layers = {}
    for r in cand:
        layers.setdefault((r["lovel"], r["hivel"]), []).append(r)
    # collapse overlapping "all velocity" layers (0-127) when others exist
    keys = sorted(layers)
    if len(keys) > 1 and (0, 127) in keys:
        keys.remove((0, 127))
    out = []
    for k in keys:
        out.append(((k[0] + k[1]) / 2.0, layers[k]))
    return out


def layer_weights(centers, v):
    """Equal-power crossfade weights. v scalar or array (0..127)."""
    v = np.asarray(v, dtype=np.float64)
    k = len(centers)
    w = np.zeros((k,) + v.shape)
    if k == 1:
        w[0] = 1.0
        return w
    for i in range(k - 1):
        c0, c1 = centers[i], centers[i + 1]
        u = np.clip((v - c0) / (c1 - c0), 0, 1)
        mask = (v >= c0) & (v < c1)
        w[i] += np.where(mask, np.cos(u * np.pi / 2), 0)
        w[i + 1] += np.where(mask, np.sin(u * np.pi / 2), 0)
    w[0] = np.where(v < centers[0], 1.0, w[0])
    w[-1] = np.where(v >= centers[-1], 1.0, w[-1])
    return w


# ---------------------------------------------------------------------------
# resampled sample cache
# ---------------------------------------------------------------------------
_RS = OrderedDict()
_RS_BYTES = [0]
RS_BUDGET = 700e6
PRE = 0.004  # seconds of pre-roll kept before detected onset


def resampled(path, cents):
    """Sample trimmed to (onset - PRE), pitch-shifted by `cents` and
    converted to 48 kHz.  Returns dict(y, pre, gpk, norm, gnorm)."""
    key = (path, int(round(cents)))
    if key in _RS:
        _RS.move_to_end(key)
        return _RS[key]
    x, sr = load_raw(path)
    mt = meta(path)
    s0 = max(0, int((mt["onset"] - PRE) * sr))
    ratio = 2 ** (key[1] / 1200.0)
    y = soxr.resample(x[s0:], sr * ratio, SR, quality="HQ").astype(np.float32)
    conv = SR / (sr * ratio)
    val = dict(y=y,
               pre=int(round((mt["onset"] * sr - s0) * conv)),
               gpk=int(round((mt["gpeak"] * sr - s0) * conv)),
               norm=10 ** ((-20.0 - mt["L"]) / 20.0),
               gnorm=10 ** ((-20.0 - mt.get("Lg", mt["L"])) / 20.0))
    _RS[key] = val
    _RS_BYTES[0] += y.nbytes
    while _RS_BYTES[0] > RS_BUDGET and len(_RS) > 1:
        _, v = _RS.popitem(last=False)
        _RS_BYTES[0] -= v["y"].nbytes
    return val


def clear_caches():
    _RS.clear()
    _RS_BYTES[0] = 0
    _RAW.clear()
    _RAW_BYTES[0] = 0


def _xfade_extend(y, start, need, rng, xf=0.25):
    """Take y[start:] and extend to `need` samples by crossfading random
    sustain segments of y (avoids the recorded release at the end)."""
    body = y[start:]
    if len(body) >= need:
        return body[:need]
    n = len(y)
    xfn = int(xf * SR)
    usable = n - start
    a = int(start + 0.25 * usable)
    b = int(start + 0.72 * usable)
    seg_len = max(int(0.35 * usable), xfn * 3)
    if b - a < seg_len // 2 + 1:
        a, b = start, max(start + 2, n - seg_len)
    cut = int(start + 0.8 * usable)
    out = [y[start:cut]]
    total = cut - start
    fo = np.cos(np.linspace(0, np.pi / 2, xfn))[:, None].astype(np.float32)
    fi = np.sin(np.linspace(0, np.pi / 2, xfn))[:, None].astype(np.float32)
    guard = 0
    while total < need + xfn and guard < 400:
        guard += 1
        s = int(rng.integers(a, max(a + 1, b - seg_len // 2)))
        chunk = y[s:s + seg_len]
        if len(chunk) < xfn * 2:
            continue
        prev = out[-1]
        mixed = prev[-xfn:] * fo + chunk[:xfn] * fi
        out[-1] = prev[:-xfn]
        out.append(mixed)
        out.append(chunk[xfn:])
        total += len(chunk) - xfn
    res = np.concatenate(out, axis=0)
    if len(res) < need:
        res = np.concatenate([res, np.zeros((need - len(res), 2), np.float32)])
    return res[:need]


# ---------------------------------------------------------------------------
# note rendering
# ---------------------------------------------------------------------------
class RR:
    def __init__(self):
        self.c = {}

    def next(self, key, n):
        v = self.c.get(key, -1) + 1
        self.c[key] = v
        return v % max(1, n)


def _db_curve(levels):
    return 30.0 * np.log10(np.maximum(levels, 0.03))


def render_note(pkey, pitch, start_beat, dur_beats, vel, dyn_pts, rr, rng,
                legato=False, gain_db=0.0, kw=None):
    """Render one note.  Returns (audio[n,2], offset) where offset is the
    sample offset of audio[0] relative to the nominal note time."""
    from dsl import dyn_at, BEAT_N
    kw = kw or {}
    p = P[pkey]
    layers = pick(pkey, pitch)
    centers = [c for c, _ in layers]
    dur_s = dur_beats * BEAT_S
    kind = kw.get("kind", p["kind"])
    rel = kw.get("rel", p["rel"])
    align_peak = p.get("align") == "peak"
    if kind == "sus":
        body_len = int((dur_s + rel + p["antic"] + 0.1) * SR)
    else:
        ml = kw.get("maxlen", p["maxlen"]) or 30.0
        if kind == "pno":
            ml = min(ml, dur_s + rel)
        body_len = int(ml * SR)
    antic = kw.get("antic", p["antic"])
    fadein = kw.get("fadein", p["fadein"])
    skip = 0.0
    if legato:
        skip = kw.get("lskip", p["lskip"])
        fadein = kw.get("lfade", p["lfade"])
        antic = antic + fadein * 0.5

    follow = kw.get("follow_dyn", kind == "sus") and bool(dyn_pts)
    if follow:
        curve = dyn_array(dyn_pts, start_beat - antic / BEAT_S, body_len, BEAT_N)
        if vel is not None:
            c0 = max(1e-3, float(curve[min(len(curve) - 1, int(antic * SR))]))
            curve = np.clip(curve * (vel / c0), 0.02, 1.0)
        w = layer_weights(centers, curve[::240] * 127.0)
        w = np.stack([np.repeat(wi, 240)[:body_len] for wi in w])
        if w.shape[1] < body_len:
            w = np.pad(w, ((0, 0), (0, body_len - w.shape[1])), mode="edge")
        gcurve = (10 ** (_db_curve(curve) / 20.0)).astype(np.float32)
    else:
        lev0 = vel if vel is not None else dyn_at(dyn_pts, start_beat)
        w = layer_weights(centers, lev0 * 127.0)
        gcurve = None
        gscalar = 10 ** (_db_curve(lev0) / 20.0)

    vary = p["vary"] and kw.get("vary", True)
    cents_var = float(rng.choice([-6, -3, 0, 3, 6])) if vary else 0.0
    gain_var = float(rng.normal(0, 0.5)) if vary else 0.0
    off_var = float(rng.uniform(0, 0.003)) if (vary and kind == "one" and not align_peak) else 0.0

    out = np.zeros((body_len, 2), np.float32)
    used = 0
    ref = None
    for li, (c, regs) in enumerate(layers):
        wl = w[li]
        if (np.ndim(wl) == 0 and wl < 0.04) or (np.ndim(wl) > 0 and wl.max() < 0.04):
            continue
        seql = max(r["seq_length"] for r in regs)
        pos = rr.next((pkey, regs[0]["lokey"], c), seql) + 1
        rsel = [r for r in regs if r["seq_position"] == pos] or regs
        r = rsel[0]
        if p["kit"]:
            shift = kw.get("tune", 0.0)
        else:
            shift = (pitch - r["center"]) + r["tune"] / 100.0 + r["transpose"]
        S = resampled(r["path"], shift * 100 + cents_var)
        y = S["y"]
        if legato:
            st = S["pre"] + int(skip * SR)
            r_ref = 0
        elif align_peak:
            st = 0
            r_ref = S["gpk"]
        else:
            st = int(off_var * SR)
            r_ref = S["pre"] - st
        if kind == "sus" and len(y) - st < body_len:
            seg = _xfade_extend(y, st, body_len, rng)
        else:
            seg = y[st:st + body_len]
        g = S["gnorm"] if align_peak else S["norm"]
        if np.ndim(wl) == 0:
            out[:len(seg)] += seg * np.float32(g * wl)
        else:
            out[:len(seg)] += seg * (g * wl[:len(seg), None]).astype(np.float32)
        used = max(used, len(seg))
        if ref is None:
            ref = r_ref
    if ref is None:
        return None, 0
    if kind != "sus" and used < body_len:
        out = out[:used]
        body_len = used
        if gcurve is not None:
            gcurve = gcurve[:used]
    if gcurve is not None:
        out *= gcurve[:, None]
    else:
        out *= np.float32(gscalar)
    out *= np.float32(10 ** ((gain_db + gain_var + p["gain"]) / 20.0))
    fi = max(1, int(fadein * SR))
    ramp = np.linspace(0, 1, fi, dtype=np.float32)
    out[:fi] *= (ramp if legato else np.sqrt(ramp))[:, None]
    if kind in ("sus", "pno"):
        rs = int((dur_s + antic) * SR)
        rn = max(1, int(rel * SR))
        if rs < body_len:
            k = min(rn, body_len - rs)
            e = np.ones(body_len, np.float32)
            tt = np.linspace(0, 1, k, dtype=np.float32)
            e[rs:rs + k] = np.exp(-4.5 * tt) * (1 - tt) ** 0.5
            e[rs + k:] = 0
            out *= e[:, None]
    tf = min(len(out), int(0.06 * SR))
    out[-tf:] *= np.linspace(1, 0, tf, dtype=np.float32)[:, None]
    offset = -int(round(antic * SR)) - int(ref)
    return out, offset


def render_part(part, total_n, rng_seed=None):
    """Render a sampler Part to a stereo buffer (before pan/mix)."""
    rng = np.random.default_rng(part["seed"] if rng_seed is None else rng_seed)
    rr = RR()
    buf = np.zeros((total_n, 2), np.float32)
    hum = part["humanize_ms"] / 1000.0
    for nt in sorted(part["notes"], key=lambda x: x["start"]):
        if nt["pitch"] is None:
            continue
        kw = dict(nt.get("kw") or {})
        pkey = nt["art"] or part["inst"]
        y, off = render_note(pkey, nt["pitch"], nt["start"], nt["dur"],
                             nt["vel"], part["dyn"], rr, rng,
                             legato=bool(nt["legato"]), gain_db=nt["gain_db"],
                             kw=kw)
        if y is None:
            continue
        t = nt["start"] * BEAT_S
        if not nt["sync"] and hum > 0:
            t += float(np.clip(rng.normal(0, hum * 0.6), -hum * 1.5, hum * 1.5))
        i0 = int(round(t * SR)) + off
        if nt["pan"] is not None:
            y = pan_stereo(y, nt["pan"], kw.get("width", 0.6))
        a, b = max(0, i0), min(total_n, i0 + len(y))
        if b > a:
            buf[a:b] += y[a - i0:b - i0]
    return buf


def pan_stereo(y, pan, width):
    mid = (y[:, 0] + y[:, 1]) * 0.5
    side = (y[:, 0] - y[:, 1]) * 0.5 * width
    l, r = mid + side, mid - side
    th = (pan + 1) * np.pi / 4
    gl, gr = np.cos(th) * np.sqrt(2), np.sin(th) * np.sqrt(2)
    # balance-style: never boost beyond +3 dB
    gl, gr = min(gl, 1.25), min(gr, 1.25)
    return np.stack([l * gl, r * gr], axis=1).astype(np.float32)


def part_hash(part):
    s = json.dumps(part, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:16]
