"""THE LONG DAWN v3 - C . THE SAMPLED-ORCHESTRA A/B (owner: COMPOSER-C2).

Renders the SAME note events as score_v3_C (read from its build(), never edited) through the reference SFZ player,
sfizz_render (sfizz 1.2.3, BSD-2-Clause; the official macOS build, x86_64 under Rosetta: nothing is compiled), on
VSCO-2-CE (CC0), with EVERY voice a real recording: the orchestra on VSCO's own samples (as in A, where our numpy
sampler plays them) and the 11 synthesised colour voices mapped to VSCO's real percussion, harp and strings.

    source ~/.venvs/longdawn/env.sh && cd music/src
    python sfizz_v3.py export        # every part's note events -> out/v3/midi/C_score.mid (musical time, 72 BPM,
                                     # one track a part) + C_events.json (exact placed times, articulation, levels)
    python sfizz_v3.py test          # one-voice checks of the sfizz path (timing, loops, CC dynamics, legato)
    python sfizz_v3.py ab            # the A/B pairs -> ~/Downloads/The Long Dawn v3 - PREVIEWS/music_AB/

How B keeps A's performance (so the A/B hears the player and the voices, not a re-performance):
  * timing: a derived SFZ starts every region 4 ms before its detected onset (`offset`), and each note-on is
    placed where A places that onset (the note time minus its anticipation; legato notes start past the attack,
    as A's do, through CC102 -> offset/attack); MIDI ticks are samples (PPQ 24000 at 120 BPM = 48,000 ticks/s)
  * level: every region is loudness-normalised exactly as A's sampler normalises it (-20 dB ref), and the part's
    dynamics curve drives the gain (CC21, 30*log10 of the level over 48 dB) and the layer crossfade (CC20,
    equal-power between VSCO's velocity layers), as A's sampler does; one-shots take both from the velocity
  * notes that need different curves at once (a per-note velocity, gain or pan) go to separate passes (sfizz's CC
    state is global), each rendered alone and summed
  * long notes loop inside the sustain (loop_continuous, a 250 ms crossfade) where A crossfades sustain segments
"""
import hashlib
import json
import os
import struct
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sampler as SM  # noqa: E402
import sampler_v2  # noqa: E402,F401  (registers the v2 patches)
from dsl import SR, BEAT_S, BEAT_N, dyn_array, dyn_at  # noqa: E402
from sfz import VSCO, MUSIC  # noqa: E402

TOOLS = os.path.join(MUSIC, "samples", "tools", "sfizz-1.2.3")
SFIZZ = os.path.join(TOOLS, "bin", "sfizz_render")
WORK = os.path.join(MUSIC, "cache", "sfizz")
SFZ_OUT = os.path.join(WORK, "sfz")
PASS_DIR = os.path.join(WORK, "pass")
for _d in (SFZ_OUT, PASS_DIR):
    os.makedirs(_d, exist_ok=True)

PPQ, TEMPO_US = 24000, 500000     # 48,000 ticks a second: one tick is one sample at 48 kHz
CC_XF, CC_EXP, CC_LEG = 20, 21, 102
EXP_DB = 48.0                     # CC21 spans -48..0 dB
SFIZZ_DB = 7.35                   # sfizz_render's own output level vs the raw sample at CC7 = 127 (measured)
PRE = SM.PRE                      # A keeps 4 ms before each detected onset


# ---------------------------------------------------------------------------
# Standard MIDI Files
# ---------------------------------------------------------------------------
def _vlq(n):
    n = int(n)
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def _meta(kind, data):
    return b"\xff" + bytes([kind]) + _vlq(len(data)) + data


def write_smf(path, tracks, ppq=PPQ, tempo_us=TEMPO_US, end_tick=None):
    """tracks: [(name, [(tick, order, bytes)])]; order breaks ties (CCs, then note-offs, then note-ons)."""
    chunks = []
    for k, (name, evs) in enumerate(tracks):
        trk = bytearray(b"\x00" + _meta(0x03, name.encode("utf-8")))
        if k == 0:
            trk += b"\x00" + _meta(0x51, struct.pack(">I", tempo_us)[1:])
        last = 0
        for tick, _, data in sorted(evs, key=lambda e: (e[0], e[1])):
            trk += _vlq(max(0, tick - last)) + data
            last = max(last, tick)
        trk += _vlq(max(0, (end_tick or last) - last)) + _meta(0x2F, b"")
        chunks.append(b"MTrk" + struct.pack(">I", len(trk)) + bytes(trk))
    fmt = 1 if len(tracks) > 1 else 0
    with open(path, "wb") as fh:
        fh.write(b"MThd" + struct.pack(">IHHH", 6, fmt, len(tracks), ppq) + b"".join(chunks))


def note_on(ch, key, vel):
    return bytes([0x90 | ch, int(key) & 0x7F, int(np.clip(vel, 1, 127))])


def note_off(ch, key):
    return bytes([0x80 | ch, int(key) & 0x7F, 0])


def cc(ch, num, val):
    return bytes([0xB0 | ch, num & 0x7F, int(np.clip(round(val), 0, 127))])


# ---------------------------------------------------------------------------
# B's real voices for A's synthesised colours (registered in memory only: sampler.py is untouched)
# ---------------------------------------------------------------------------
_PC = "Percussion/"
SM.KITS["anvil_real"] = [(_PC + "Anvil_Hit1_v1_Sum.wav", 0, 42), (_PC + "Anvil_Hit1_v2_Sum.wav", 43, 84),
                         (_PC + "Anvil_Hit1_v3_Sum.wav", 85, 127)]
SM.patch("anvil_real", kit="anvil_real", kind="one", maxlen=2.4)
KEYTRACK = {"anvil_real": 88}      # the real anvil keeps the tick's E6 / D#6 alternation (as shifts from E6)
# A's synth voice -> B's real recording.  tamswell (already a real gong, reversed and forward) becomes the real
# players' version: a suspended-cymbal roll crescendo peaking where A's swell peaks + the gong's bloom from there.
REAL = {"fmbell": "glock", "hharm": "harp", "harmonic": "vln_q", "anvil": "anvil_real", "lowbell": "tubular",
        "taiko": "bdrum", "riser": "cym_swell_long", "revcym": "cym_swell_med"}


def _note_kw_real(inst, nt):
    """A synth note's kw -> the sampler kw of its real voice."""
    kw = dict(nt.get("kw") or {})
    out = {}
    if inst == "anvil":
        out["maxlen"] = 0.45 if float(kw.get("size", 1.0)) < 0.5 else 2.4
    elif inst == "harmonic":
        out["fadein"] = float(kw.get("atk", 0.9))
        out["rel"] = float(kw.get("rel", 1.6))
    elif inst in ("riser", "revcym"):
        out["align_end"] = True
    return out


# ---------------------------------------------------------------------------
# derived SFZ: VSCO's regions, played the way A's sampler plays them
# ---------------------------------------------------------------------------
def _segments(regs):
    """Split the keyboard where the set of covering regions changes; per segment, A's pick(): the layers by
    (lovel, hivel), an all-velocity layer dropped when others exist, the first region per round-robin slot."""
    edges = sorted({r["lokey"] for r in regs} | {r["hikey"] + 1 for r in regs} | {0, 128})
    segs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        cand = [r for r in regs if r["lokey"] <= lo and r["hikey"] >= hi - 1]
        if not cand:
            continue
        layers = {}
        for r in cand:
            layers.setdefault((r["lovel"], r["hivel"]), []).append(r)
        keys = sorted(layers)
        if len(keys) > 1 and (0, 127) in keys:
            keys.remove((0, 127))
        lay = []
        for k in keys:
            seen, rs = set(), []
            for r in layers[k]:
                if r["seq_position"] not in seen:
                    seen.add(r["seq_position"])
                    rs.append(r)
            lay.append(((k[0] + k[1]) / 2.0, rs))
        segs.append([lo, hi - 1, lay])
    if segs:                                  # A plays the nearest mapped range beyond the ends
        segs[0][0], segs[-1][1] = 0, 127
        for a, b in zip(segs[:-1], segs[1:]):
            if b[0] > a[1] + 1:               # a gap: split it between its neighbours
                mid = (a[1] + b[0]) // 2
                a[1], b[0] = mid, mid + 1
    return segs


def derive_sfz(pkey, mode, rel=0.35, fadein=0.004, lskip=0.07, lfade=0.07, skip_s=0.0, vol_db=0.0):
    """Write (once) the SFZ that makes sfizz play patch `pkey` as A's sampler does.  mode 'sus' (layers on CC20,
    gain applied after rendering, legato on CC102, loops in the sustain) or 'one' (layers and gain on velocity).
    Returns its path."""
    spec = json.dumps([pkey, mode, round(rel, 4), round(fadein, 4), round(lskip, 4), round(lfade, 4),
                       round(skip_s, 4), round(vol_db, 2), 3], sort_keys=True)
    path = os.path.join(SFZ_OUT, f"{pkey}_{hashlib.sha1(spec.encode()).hexdigest()[:10]}.sfz")
    if os.path.exists(path):
        return path
    p = SM.P[pkey]
    kit = bool(p["kit"])
    peak = p.get("align") == "peak"
    lines = [f"// derived from VSCO-2-CE for sfizz by sfizz_v3.py: {spec}", "<control>", "<global>",
             f"ampeg_attack={max(fadein, 0.001):.4f}", f"ampeg_release={rel:.4f}"]
    if mode == "sus":
        lines += ["amp_veltrack=0", f"ampeg_attack_oncc{CC_LEG}={max(0.0, lfade - fadein):.4f}"]
    else:
        lines += ["amp_veltrack=100"] + [f"amp_velcurve_{v}={max(v / 127.0, 0.03) ** 1.5:.6f}"
                                         for v in (1, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
                                                   112, 120, 127)]
    for lo, hi, lay in _segments(SM.regions(pkey)):
        cs = [c for c, _ in lay]
        for i, (c, rs) in enumerate(lay):
            xf = []
            opc = "cc%d" % CC_XF if mode == "sus" else "vel"
            if len(cs) > 1:
                if i > 0:
                    xf += [f"xfin_lo{opc}={int(round(cs[i - 1]))}", f"xfin_hi{opc}={int(round(c))}"]
                if i < len(cs) - 1:
                    xf += [f"xfout_lo{opc}={int(round(c))}", f"xfout_hi{opc}={int(round(cs[i + 1]))}"]
                xf += ["xf_cccurve=power" if mode == "sus" else "xf_velcurve=power"]
            for r in rs:
                mt = SM.meta(r["path"])
                sr = mt["sr"]
                L = mt.get("Lg", mt["L"]) if peak else mt["L"]
                off = int(round((mt["onset"] - PRE + skip_s) * sr))
                reg = ["<region>", f"lokey={lo}", f"hikey={hi}", "lovel=0", "hivel=127"]
                if kit:
                    kc = KEYTRACK.get(pkey)
                    reg += [f"pitch_keycenter={kc}", "pitch_keytrack=100"] if kc else ["pitch_keycenter=60",
                                                                                        "pitch_keytrack=0"]
                    tune = 0.0
                else:
                    reg += [f"pitch_keycenter={r['center']}", f"transpose={r['transpose']}"]
                    tune = r["tune"] + SM.pitch_fix(r["path"], pkey)
                reg += [f"tune={tune:.2f}", f"volume={-20.0 - L + vol_db + SFIZZ_DB:.3f}"]
                reg += [f"offset={max(0, off)}"] + ([f"delay={-off / sr:.6f}"] if off < 0 else [])
                if r["seq_length"] > 1:
                    reg += [f"seq_length={r['seq_length']}", f"seq_position={r['seq_position']}"]
                if mode == "sus":
                    st = mt["onset"] * sr
                    use = mt["dur"] * sr - st
                    reg += [f"offset_oncc{CC_LEG}={int(round((PRE + lskip) * sr))}", "loop_mode=loop_continuous",
                            f"loop_start={int(st + 0.30 * use)}", f"loop_end={int(st + 0.78 * use)}",
                            "loop_crossfade=0.25"]
                else:
                    reg += ["loop_mode=no_loop"]
                lines += reg + xf + [f"sample={r['path']}"]
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# A's performance of a part, as B's events
# ---------------------------------------------------------------------------
def events(pd, pkey_over=None, kw_fn=None, seed_off=0):
    """Every note of part dict pd as one event: where A's sampler puts it (seconds), what it asks of the
    sample (patch, articulation variant, layer control) and its post gains.  pkey_over / kw_fn map a synth part
    onto its real voice."""
    rng = np.random.default_rng(pd["seed"] + 7919 + seed_off)
    hum = pd["humanize_ms"] / 1000.0
    dyn = pd["dyn"]
    out = []
    for nt in sorted(pd["notes"], key=lambda x: x["start"]):
        if nt["pitch"] is None:
            continue
        pkey = pkey_over or nt["art"] or pd["inst"]
        p = SM.P[pkey]
        kw = dict(nt.get("kw") or {}) if pkey_over is None else kw_fn(nt)
        kind = kw.get("kind", p["kind"])
        mode = "sus" if kind == "sus" else "one"
        rel = kw.get("rel", p["rel"])
        antic = kw.get("antic", p["antic"])
        fadein = kw.get("fadein", p["fadein"])
        lskip, lfade = kw.get("lskip", p["lskip"]), kw.get("lfade", p["lfade"])
        legato = bool(nt["legato"]) and mode == "sus"
        if legato:
            antic = antic + lfade * 0.5
        t = nt["start"] * BEAT_S
        if not nt["sync"] and hum > 0:
            t += float(np.clip(rng.normal(0, hum * 0.6), -hum * 1.5, hum * 1.5))
        dur_s = nt["dur"] * BEAT_S
        key = int(round(nt["pitch"] + SM.KEY_OFFSET.get(pkey, 0)))
        if abs(nt["pitch"] - round(nt["pitch"])) > 0.01:
            print(f"  WARNING {pd['name']}: microtonal pitch {nt['pitch']} rounded", flush=True)
        ev = dict(part=pd["name"], pkey=pkey, mode=mode, key=key, legato=legato, t=t, gain_db=nt["gain_db"] or 0.0,
                  pan=nt["pan"], width=kw.get("width", 0.6),
                  var=(rel, fadein, lskip, lfade, 0.0) if mode == "sus" else (0.06, fadein, 0.07, 0.07, 0.0))
        if mode == "sus":
            on = t - antic - (0.0 if legato else PRE)
            ev.update(on=on, off=on + dur_s + antic, tail=rel + 0.15)      # A's release starts dur + antic in
            follow = kw.get("follow_dyn", True) and bool(dyn)
            if follow:
                scale = 1.0
                if nt["vel"] is not None:
                    scale = float(nt["vel"]) / max(1e-3, float(dyn_at(dyn, nt["start"])))
                ev["ctl"] = ("f", round(scale, 4))
            else:
                lev0 = nt["vel"] if nt["vel"] is not None else dyn_at(dyn, nt["start"])
                ev["ctl"] = ("c", round(float(lev0), 4))
            ev["vel"] = 100
        else:
            lev0 = nt["vel"] if nt["vel"] is not None else dyn_at(dyn, nt["start"])
            ml = kw.get("maxlen", p["maxlen"]) or 30.0
            on = t - antic - PRE
            if p.get("align") == "peak" or kw.get("align_end"):
                r0 = SM.regions(pkey)[0]
                mt = SM.meta(r0["path"])
                target = t + dur_s if kw.get("align_end") else t - antic
                on = target - (mt["gpeak"] - mt["onset"]) - PRE
                ml = min(ml, mt["dur"])
            ev.update(on=on, off=on + ml - 0.06, tail=0.12, vel=int(np.clip(round(lev0 * 127), 1, 127)), ctl=None)
        out.append(ev)
    return out


def _control(pd, ctl, t0, n):
    """the layer level c(t) of a sustained pass over [t0, t0 + n / SR)"""
    if ctl[0] == "c":
        return np.full(n, ctl[1], np.float32)
    c = dyn_array(pd["dyn"], t0 / BEAT_S, n, BEAT_N).astype(np.float32)
    if ctl[1] != 1.0:
        c = np.clip(c * ctl[1], 0.02, 1.0)
    return c


def passes(evs):
    """Group events that can share one sfizz run: same patch/variant/layer control/post gain/pan, and no key
    re-struck while it still sounds (a note-off would release both)."""
    groups = {}
    for ev in evs:
        base = (ev["pkey"], ev["mode"], ev["var"], ev["ctl"], round(ev["gain_db"], 3),
                None if ev["pan"] is None else (round(ev["pan"], 3), round(ev["width"], 3)))
        k = 0
        while True:
            g = groups.setdefault(base + (k,), [])
            if all(not (e["key"] == ev["key"] and e["on"] < ev["off"] + ev["tail"] and ev["on"] < e["off"] + e["tail"])
                   for e in g):
                g.append(ev)
                break
            k += 1
    return groups


def _gate(y, lsb=3.0 / 32768, ramp=0.005):
    """sfizz_render writes 16-bit: silence the quantisation floor between and after the notes (below 3 LSB,
    -81 dBFS in the pass, where every note sits near -26 dBFS) so the passes' floors never sum into a hiss"""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    w = int(ramp * SR)
    m = (maximum_filter1d(np.abs(y).max(1), size=2 * w + 1) > lsb).astype(np.float32)
    m = uniform_filter1d(m, size=w)
    return y * m[:, None]


def render_pass(pd, gkey, evs, t_lo, n_out, tag):
    """One sfizz_render run; returns (buffer over [t_lo, t_lo + n_out / SR), clipped?)."""
    pkey, mode, var, ctl, gain_db, panw, _ = gkey
    rel, fadein, lskip, lfade, skip_s = var
    t0 = min(e["on"] for e in evs) - 0.02
    t1 = max(e["off"] + e["tail"] for e in evs) + 0.3
    poly = max(sum(1 for e in evs if e["on"] <= x["on"] < e["off"] + e["tail"]) for x in evs)
    vol = -(6.0 + 10 * np.log10(max(1, poly)))
    if mode == "one":
        vol -= 20 * np.log10(max(max(e["vel"] for e in evs) / 127.0, 0.03) ** 1.5)
    tk = lambda s: max(0, int(round((s - t0) * SR)))  # noqa: E731
    ev = [(0, 0, cc(0, 7, 127))]
    for e in evs:
        if mode == "sus":
            ev.append((tk(e["on"]), 1, cc(0, CC_LEG, 127 if e["legato"] else 0)))
        ev.append((tk(e["on"]), 3, note_on(0, e["key"], e["vel"])))
        ev.append((tk(e["off"]), 2, note_off(0, e["key"])))
    n = int(round((t1 - t0) * SR))
    g = None
    if mode == "sus":
        c = _control(pd, ctl, t0, n)
        step = 240
        last = -1
        for i in range(0, n, step):
            v = int(np.clip(round(float(c[i]) * 127), 0, 127))
            if v != last:
                ev.append((i, 0, cc(0, CC_XF, v)))
                last = v
        g = (np.maximum(c, 0.03) ** 1.5).astype(np.float32)
    # sfizz_render --use-eot stops at the last MIDI EVENT (not the End of Track's time), which would cut the last
    # note's release: an inert controller at the pass's end keeps it rendering through every tail
    ev.append((n - 1, 9, cc(0, 119, 0)))
    mid, wav = os.path.join(PASS_DIR, tag + ".mid"), os.path.join(PASS_DIR, tag + ".wav")
    write_smf(mid, [(tag, ev)], end_tick=n)
    for attempt in range(3):
        sfz = derive_sfz(pkey, mode, rel, fadein, lskip, lfade, skip_s, vol)
        r = subprocess.run(["arch", "-x86_64", SFIZZ, "--sfz", sfz, "--midi", mid, "--wav", wav, "-s", str(SR),
                            "-q", "3", "-b", "1024", "-p", "256", "--use-eot"], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(wav):
            raise RuntimeError(f"sfizz_render failed on {tag}: {r.stderr[-400:]}")
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        os.remove(wav)
        assert sr == SR
        clipped = bool(np.abs(y).max() >= 0.9999)
        if not clipped:
            break
        vol -= 12.0
    os.remove(mid)
    y = _gate(y[:n])
    if g is not None:
        y *= g[:len(y), None]
    y *= np.float32(10 ** ((gain_db + SM.P[pkey]["gain"] - vol) / 20))
    if panw is not None:
        y = SM.pan_stereo(y, panw[0], panw[1])
    out = np.zeros((n_out, 2), np.float32)
    a = int(round((t0 - t_lo) * SR))
    lo, hi = max(0, a), min(n_out, a + len(y))
    if hi > lo:
        out[lo:hi] = y[lo - a:hi - a]
    return out, clipped


def render_part_b(pd, t_lo, t_hi, inst_map=True):
    """B's render of one part over [t_lo, t_hi) (seconds): (buffer, info).  Synth parts go to their real voice."""
    n_out = int(round((t_hi - t_lo) * SR))
    real = None
    if pd["kind"] != "sampler":
        if pd["inst"] == "tamswell":
            return render_tam_b(pd, t_lo, t_hi), dict(real="cym_swell_long + gong")
        real = REAL.get(pd["inst"])
        if real is None:
            return None, dict(real=None)
        evs = events(pd, pkey_over=real, kw_fn=lambda nt: _note_kw_real(pd["inst"], nt))
    else:
        evs = events(pd)
    evs = [e for e in evs if e["on"] < t_hi and e["off"] + e["tail"] + 0.2 > t_lo]
    if not evs:
        return None, dict(real=real, passes=0)
    buf = np.zeros((n_out, 2), np.float32)
    clips = 0
    grp = passes(evs)
    for k, (gk, ge) in enumerate(sorted(grp.items(), key=lambda kv: min(e["on"] for e in kv[1]))):
        y, clipped = render_pass(pd, gk, ge, t_lo, n_out, f"{pd['name']}_{k}")
        if clipped:
            clips += 1
        buf += y
    return buf, dict(real=real, passes=len(grp), notes=len(evs), clipped=clips)


def render_tam_b(pd, t_lo, t_hi):
    """A's tam swell (the gong's wash reversed into the note's end, then its bloom) by real players: a suspended
    cymbal's roll crescendo peaking on the note's end, and the gong struck there with the mallet taken out."""
    n_out = int(round((t_hi - t_lo) * SR))
    buf = np.zeros((n_out, 2), np.float32)
    for nt in pd["notes"]:
        if nt["pitch"] is None:
            continue
        t_end = (nt["start"] + nt["dur"]) * BEAT_S
        lev = nt["vel"] if nt["vel"] is not None else dyn_at(pd["dyn"], nt["start"])
        v = int(np.clip(round(lev * 127), 1, 127))
        sw = SM.meta(SM.regions("cym_swell_long")[0]["path"])
        on = t_end - (sw["gpeak"] - sw["onset"]) - PRE
        bloom = float(nt["kw"].get("rel", 5.0))
        for pkey, e_on, e_off, var in (("cym_swell_long", on, on + sw["dur"] - 0.06, (0.06, 0.004, 0.07, 0.07, 0.0)),
                                       ("gong", t_end - PRE, t_end - PRE + bloom - 1.2, (1.2, 0.05, 0.07, 0.07, 0.35))):
            ev = dict(part=pd["name"], pkey=pkey, mode="one", key=60, legato=False, t=t_end, gain_db=0.0, pan=None,
                      width=0.6, var=var, on=e_on, off=e_off, tail=var[0] + 0.06, vel=v, ctl=None)
            gk = (pkey, "one", var, None, 0.0, None, 0)
            y, _ = render_pass(pd, gk, [ev], t_lo, n_out, f"{pd['name']}_{pkey}")
            buf += y
    return buf


# ---------------------------------------------------------------------------
# the score, a window of it, A's mix chain and A's master
# ---------------------------------------------------------------------------
def score_C():
    from timeline_v3 import BarMap
    import score_v3_C as C
    bm = BarMap("C")
    S = C.build(bm)
    return bm, S, S.used()


def seat_chain(name, y, p, eq):
    """render_v3.mix_score's per-part chain (gain, pan/width, depth, eq); returns (dry, send) contributions"""
    import mix as MX
    y = MX.pan_width(y * np.float32(10 ** (p.gain_db / 20)), p.pan, p.width)
    y = MX.depth_eq(y, p.depth)
    spec = eq.get(name, (None, None))
    if spec[0]:
        y = MX.highpass(y, spec[0])
    if spec[1]:
        from scipy import signal as _sg
        bb, aa = _sg.butter(2, spec[1] / (SR / 2))
        y = _sg.lfilter(bb, aa, y, axis=0).astype(np.float32)
    if len(spec) > 2 and spec[2]:
        y = y + MX.highpass(y, spec[3] if len(spec) > 3 else 6000.0, order=2) * np.float32(10 ** (spec[2] / 20) - 1)
    return y * np.float32(1.0 - 0.35 * p.depth), y * np.float32(p.send)


class Window:
    """render_v3.mix_score over [t_lo, t_hi) seconds: add parts one at a time (low memory), then close()."""

    def __init__(self, S, t_lo, t_hi):
        import render_v3 as R
        self.S, self.t_lo = S, t_lo
        self.n = int(round((t_hi - t_lo) * SR))
        off = int(round(t_lo * SR))
        self.wins = [(i0 - off, i1 - off, ex) for i0, i1, ex in R.breath_windows(S.breaths)
                     if i0 - off >= 0 and i1 - off <= self.n]
        self.dry = np.zeros((self.n, 2), np.float32)
        self.send = np.zeros((self.n, 2), np.float32)

    def add(self, name, y, p):
        import render_v3 as R
        y = y.astype(np.float32, copy=True)
        if self.wins:
            y *= R.breath_env(self.n, self.wins, name=name, offset=0)[:, None]
        d, s = seat_chain(name, y, p, self.S.eq)
        self.dry += d
        self.send += s

    def close(self):
        import mix as MX
        import render_v3 as R
        wet = R.convolve_with_breaths(self.send, R.hall_ir(), self.wins, self.n)
        out = MX.highpass(self.dry + wet, 22.0)
        self.dry = self.send = None
        return out


def a_stem(name, key, t_lo, n):
    """A's cached dry part stem over the window"""
    import render_v3 as R
    off, y = R.load_stem(name, key)
    out = np.zeros((n, 2), np.float32)
    a = int(round(t_lo * SR))
    lo, hi = max(a, off), min(a + n, off + len(y))
    if hi > lo:
        out[lo - a:hi - a] = y[lo - off:hi - off]
    return out


def ride_curve(t_lo, n, pts):
    """a fader ride: (t_s, dB) breakpoints, cosine-interpolated; 0 dB outside"""
    t = t_lo + np.arange(n) / SR
    g = np.zeros(n)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        m = (t >= x0) & (t < x1)
        u = (t[m] - x0) / (x1 - x0)
        g[m] = y0 + (y1 - y0) * (1 - np.cos(np.pi * u)) / 2
    return (10 ** (g / 20)).astype(np.float32)


def master_window(pre, G, ride=None):
    """render_v3.master's chain on the score alone (the film's settings): gain G, glue comp, true-peak limiter"""
    import mix as MX
    import render_v3 as R
    x = MX.highpass(pre.astype(np.float32), 8.0, order=1) * np.float32(G)
    if ride is not None:
        x *= ride[:, None]
    comp = MX.glue_comp_gain(x, thresh_db=-16.0, ratio=1.5, attack=0.04, release=0.4)
    y = x * comp[:, None]
    lim = R.limiter_gain(y, R.CEIL_DB)
    y *= lim[:, None]
    return y, comp, lim


def master_G(name="final_C"):
    import render_v3 as R
    env = np.load(os.path.join(R.CACHE, f"master_env_{name}.npy"), mmap_mode="r")
    return float(np.max(env))


def lufs(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(np.asarray(x, np.float64))


def write_pair_file(path, y, fade_in=0.25, fade_out=1.2):
    import mix as MX
    y = y.astype(np.float32, copy=True)
    fi, fo = int(fade_in * SR), int(fade_out * SR)
    y[:fi] *= np.sin(np.linspace(0, np.pi / 2, fi, dtype=np.float32))[:, None] ** 2
    y[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo, dtype=np.float32))[:, None] ** 2
    sf.write(path, MX.tpdf_dither_24(y, np.random.default_rng(0)), SR, subtype="PCM_24")


# ---------------------------------------------------------------------------
# export: every part's note events
# ---------------------------------------------------------------------------
def export(out_dir=None):
    """out/v3/midi/C_score.mid (format 1, one track a part, 72 BPM, PPQ 960: the WRITTEN notes, velocity =
    the level, CC11 = the part's dynamics curve) + C_events.json (every note: written beat/duration, velocity,
    articulation, legato, sync, and where A's sampler places it in seconds)."""
    bm, S, parts = score_C()
    out_dir = out_dir or os.path.join(MUSIC, "out", "v3", "midi")
    os.makedirs(out_dir, exist_ok=True)
    ppq = 960
    tracks, js = [], {}
    for k, (name, p) in enumerate(parts.items()):
        pd = p.to_dict()
        ch = k % 16 if k % 16 != 9 else 15
        ev, rows = [], []
        for nt in pd["notes"]:
            if nt["pitch"] is None:
                continue
            lev = nt["vel"] if nt["vel"] is not None else dyn_at(pd["dyn"], nt["start"])
            a, b = int(round(nt["start"] * ppq)), int(round((nt["start"] + nt["dur"]) * ppq))
            key = int(round(nt["pitch"]))
            ev += [(a, 3, note_on(ch, key, round(lev * 127))), (max(a + 1, b), 2, note_off(ch, key))]
            rows.append(dict(pitch=nt["pitch"], beat=round(nt["start"], 6), dur_beats=round(nt["dur"], 6),
                             start_s=round(nt["start"] * BEAT_S, 6), dur_s=round(nt["dur"] * BEAT_S, 6),
                             level=round(float(lev), 4), vel_given=nt["vel"], art=nt["art"] or pd["inst"],
                             legato=bool(nt["legato"]), sync=bool(nt["sync"]), gain_db=nt["gain_db"],
                             kw={kk: vv for kk, vv in (nt["kw"] or {}).items() if isinstance(vv, (int, float, str))}))
        if pd["dyn"]:
            last = -1
            b0, b1 = pd["dyn"][0][0], pd["dyn"][-1][0]
            for i in range(int(b0 * 8), int(b1 * 8) + 1):
                v = int(round(dyn_at(pd["dyn"], i / 8) * 127))
                if v != last:
                    ev.append((i * ppq // 8, 0, cc(ch, 11, v)))
                    last = v
        tracks.append((f"{name} ({pd['inst']}, {pd['kind']})", ev))
        placed = {}
        if pd["kind"] == "sampler":
            for e in events(pd):
                placed.setdefault(round(e["t"], 4), []).append(round(e["on"] + (0 if e["legato"] else PRE), 6))
        js[name] = dict(inst=pd["inst"], kind=pd["kind"], bus=pd["bus"], seat=dict(pan=pd["pan"], width=pd["width"],
                        gain_db=pd["gain_db"], depth=pd["depth"], send=pd["send"]), dyn=pd["dyn"], notes=rows)
    write_smf(os.path.join(out_dir, "C_score.mid"), tracks, ppq=ppq, tempo_us=int(round(60e6 / 72)))
    with open(os.path.join(out_dir, "C_events.json"), "w") as fh:
        json.dump(dict(cut="C", bpm=72, beat_s=BEAT_S, sr=SR, bars_note="bar 1 beat 1 = 0 s; 4 beats a bar",
                       sync_points=[dict(t_s=round(s[0], 4), label=s[1], part=s[2]) for s in S.sync],
                       parts=js), fh, indent=1, default=str)
    n = sum(len(v["notes"]) for v in js.values())
    print(f"export: {len(js)} parts, {n} notes -> {out_dir}/C_score.mid + C_events.json", flush=True)


# ---------------------------------------------------------------------------
# the A/B
# ---------------------------------------------------------------------------
DEST = os.path.expanduser("~/Downloads/The Long Dawn v3 - PREVIEWS/music_AB")
PREROLL, TAIL = 10.0, 4.0
# the climax's headroom fix: a slow fader ride that takes over most of the master's gain reduction at the Eye and
# the slit (up to -10.4 dB there in the delivered master: comp 5.0 + limiter 5.6), so the limiter only trims
CLIMAX_RIDE = [(79.0, 0.0), (80.5, -1.0), (81.6, -2.5), (82.6, -5.0), (83.25, -7.0), (84.6, -6.5), (85.4, -3.0),
               (86.6, -0.5), (87.6, 0.0)]
PAIRS = [
    dict(id="1_horn_solo", t0=139.0, t1=162.5,
         what="the lone horn cries the CALL over a low D, the Ring circling beneath (bassoon, cellos), then the call "
              "echoes off the ranges, farther each time (bars 43-48, 2:20-2:40)"),
    dict(id="2_climax", t0=70.0, t1=92.0, ride=CLIMAX_RIDE,
         what="swept to parchment (the anvil tick), THE EYE, the slit at 1:23 (C's loudest), the tam-tam's bloom, "
              "into the Mirror (bars 22-28, 1:10-1:32)"),
    dict(id="3_dawn", t0=236.0, t1=258.0,
         what="THE ILLUMINATION: the bloom out of silence, the CALL on one horn, the ANSWER and HOME in octave "
              "violins with the solo violin, home on bar 76 (bars 72-77, 3:56-4:18)"),
]


def _rms_db(y):
    a = np.abs(y).max(1) if y.ndim > 1 else np.abs(y)
    act = a > 10 ** (-70 / 20) * (a.max() + 1e-12)
    if not act.any():
        return -200.0
    return float(10 * np.log10((y[act] ** 2).mean() + 1e-20))


def ab(ids=None, dest=DEST):
    import render_v3 as R
    bm, S, parts = score_C()
    man = json.load(open(os.path.join(R.CACHE, "manifest_final_C.json")))
    G = master_G()
    pm_all = np.load(os.path.join(R.CACHE, "premaster_score_final_C.npy"), mmap_mode="r")
    a_file = os.path.join(R.OUT, "final_C_score.wav")
    os.makedirs(dest, exist_ok=True)
    log = []
    for pr in PAIRS:
        if ids and pr["id"] not in ids:
            continue
        t_lo, t_hi = pr["t0"] - PREROLL, pr["t1"] + TAIL
        n = int(round((t_hi - t_lo) * SR))
        WA, WB = Window(S, t_lo, t_hi), Window(S, t_lo, t_hi)
        rows = []
        for name, p in parts.items():
            pd = p.to_dict()
            ya = a_stem(name, man[name], t_lo, n)
            if not np.any(ya):
                continue
            yb, info = render_part_b(pd, t_lo, t_hi)
            la = 10 * np.log10(float((ya.astype(np.float64) ** 2).sum()) + 1e-20)
            lb = 10 * np.log10(float((yb.astype(np.float64) ** 2).sum()) + 1e-20) if yb is not None else -200.0
            cal = 0.0
            if yb is not None and lb > -150:
                cal = la - lb                   # one gain a part: B keeps A's balance (the raw difference is logged)
                yb *= np.float32(10 ** (cal / 20))
            rows.append((name, pd["kind"], info.get("real"), info.get("passes"), round(la, 1), round(-cal, 2)))
            WA.add(name, ya, p)
            if yb is not None:
                WB.add(name, yb, p)
            del ya, yb
        preA, preB = WA.close(), WB.close()
        np.save(os.path.join(WORK, f"pre_{pr['id']}_B.npy"), preB)
        i0 = int(round(t_lo * SR))
        pm = np.asarray(pm_all[i0:i0 + n], dtype=np.float32)
        a0, a1 = int(round(PREROLL * SR)), int(round((PREROLL + pr["t1"] - pr["t0"]) * SR))
        err = 10 * np.log10(((preA - pm)[a0:a1] ** 2).mean() / ((pm[a0:a1] ** 2).mean() + 1e-20) + 1e-20)
        A_del, sr = sf.read(a_file, start=int(round(pr["t0"] * SR)), stop=int(round(pr["t1"] * SR)), dtype="float32",
                            always_2d=True)
        LA = lufs(A_del)
        ride = ride_curve(t_lo, n, pr["ride"]) if pr.get("ride") else None
        outs = {"A_current": (A_del, None)}
        if ride is not None:
            y, comp, lim = master_window(pm, G, ride)
            outs["A_current_headroomfix"] = (y[a0:a1], (comp[a0:a1], lim[a0:a1]))
        y, comp, lim = master_window(preB, G, ride)
        outs["B_samples" + ("_headroomfix" if ride is not None else "")] = (y[a0:a1], (comp[a0:a1], lim[a0:a1]))
        yA0, compA0, limA0 = master_window(pm, G, None)       # A's own chain on the score alone (for the GR report)
        gr = {"A_current": (20 * np.log10(compA0[a0:a1].min()), 20 * np.log10(limA0[a0:a1].min()))}
        del yA0
        line = [f"{pr['id']}: A re-mix vs A premaster {err:.1f} dB (chain check); A {LA:.2f} LUFS"]
        for k, (y, cl) in outs.items():
            if cl is not None:
                gr[k] = (20 * np.log10(cl[0].min()), 20 * np.log10(cl[1].min()))
                y = y * np.float32(10 ** ((LA - lufs(y)) / 20))
            tp = R.true_peak_db(y)
            write_pair_file(os.path.join(dest, f"{pr['id']}_{k}.wav"), y)
            line.append(f"{k}: {lufs(y):.2f} LUFS, TP {tp:.2f} dBTP, master comp/lim GR max "
                        f"{gr[k][0]:.1f}/{gr[k][1]:.1f} dB")
        print("\n  ".join(line), flush=True)
        for r in rows:
            print(f"    {r[0]:12s} {r[1]:7s} real={str(r[2]):15s} passes={str(r[3]):4s} A {r[4]:6.1f} dB  "
                  f"B-A before calibration {r[5]:+.2f} dB", flush=True)
        log.append(dict(pair=pr["id"], what=pr["what"], lines=line, parts=rows))
        del preA, preB, pm, WA, WB, outs
    with open(os.path.join(WORK, "ab_report.json"), "w") as fh:
        json.dump(log, fh, indent=1, default=str)
    return log


def check_part(name, t0, t1):
    """one part, B vs A: level, onset timing (cross-correlation of envelopes), passes"""
    import render_v3 as R
    bm, S, parts = score_C()
    man = json.load(open(os.path.join(R.CACHE, "manifest_final_C.json")))
    pd = parts[name].to_dict()
    n = int(round((t1 - t0) * SR))
    ya = a_stem(name, man[name], t0, n)
    yb, info = render_part_b(pd, t0, t1)
    ea = np.abs(ya).max(1)
    eb = np.abs(yb).max(1)
    k = 48
    ea = ea[:len(ea) // k * k].reshape(-1, k).max(1)
    eb = eb[:len(eb) // k * k].reshape(-1, k).max(1)
    xc = np.correlate(ea - ea.mean(), eb - eb.mean(), "full")
    lag = (np.argmax(xc) - (len(eb) - 1)) * k / SR * 1000
    print(f"{name}: {info}; A {_rms_db(ya):.1f} dB, B {_rms_db(yb):.1f} dB; envelope lag (A vs B) {lag:+.1f} ms",
          flush=True)
    return ya, yb


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if cmd == "export":
        export()
    elif cmd == "check":
        check_part(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    elif cmd == "ab":
        ab(sys.argv[2].split(",") if len(sys.argv) > 2 else None)
