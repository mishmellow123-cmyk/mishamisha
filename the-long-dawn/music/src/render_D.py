"""Opt-in D draft renderer; all writes stay below --output-dir.

Uses the v3 sampler, synth voices, seating, hall and master dynamics. Parts render
serially; EQ, summation, convolution and peak detection use bounded blocks. The
delivered AP2 opening is inserted after mastering and is never gain-adjusted.
No legacy renderer is imported (those modules create shared output directories).

Run through the project's onepy wrapper, for example::

    onepy python music/src/render_D.py --render --output-dir /tmp/d-score

--master-only reuses a matching private premaster after a change to master rides.
The receipt distinguishes measured audio/memory values from provisional events.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import wave

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SR = 48000
FPS = 24
FRAME_N = SR // FPS
BEAT_N = 40000
CHUNK = 8 * SR
MAX_RSS = 3 * 1024 ** 3
MAX_DISK = int(1.5 * 1024 ** 3)


def peak_rss_bytes():
    """ru_maxrss is bytes on macOS and KiB on Linux."""
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform == "darwin" else n * 1024)


def memory_guard():
    n = peak_rss_bytes()
    if n >= MAX_RSS:
        raise MemoryError(f"D render peak RSS {n / 1024**3:.3f} GiB exceeds the 3 GiB lane limit")
    return n


def blocks(n, chunk=CHUNK):
    for a in range(0, n, chunk):
        yield a, min(n, a + chunk)


def _new_map(path, n):
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, 2))


def _remove_map(y, path):
    y.flush()
    y._mmap.close()
    Path(path).unlink(missing_ok=True)


def true_peak_envelope(x, chunk=CHUNK, pad=256):
    """The stock render_v2 4x, 8-second, overlap-padded true-peak method."""
    out = np.empty(len(x), np.float32)
    for a, b in blocks(len(x), chunk):
        a0, b0 = max(0, a-pad), min(len(x), b+pad)
        y = signal.resample_poly(np.asarray(x[a0:b0], np.float64), 4, 1, axis=0)
        pk = np.abs(y).max(1).reshape(-1, 4).max(1)
        out[a:b] = pk[a-a0:a-a0+b-a]
    return out


def true_peak_db(x):
    return float(20 * np.log10(float(true_peak_envelope(x).max()) + 1e-12))


def lufs(x):
    """Same pyloudnorm float64 integrated measurement as render_v2."""
    import pyloudnorm as pyln
    return float(pyln.Meter(SR).integrated_loudness(np.asarray(x, np.float64)))


def _smooth_control(g, attack, release):
    """Stock mix._smooth_gain's 1 ms control filter, without a full-rate copy."""
    aa, ar = np.exp(-1/(attack*1000)), np.exp(-1/(release*1000))
    out = np.empty_like(g)
    state = 0.0
    for i, v in enumerate(g):
        c = aa if v < state else ar
        state = c * state + (1-c) * v
        out[i] = state
    return out


def glue_gain(x, thresh_db=-16, ratio=1.5, attack=.04, release=.4, knee=6):
    """Stock 20 ms centered RMS detector and 1 ms minimum control, in blocks."""
    hop, w = 48, int(.02 * SR)
    ctrl = np.empty((len(x)+hop-1)//hop, np.float64)
    # CHUNK is divisible by the engine's 48-sample control hop.
    for a, b in blocks(len(x)):
        a0, b0 = max(0, a-w//2), min(len(x), b+w//2)
        mono = np.mean(np.asarray(x[a0:b0], np.float64)**2, axis=1)
        rms = np.sqrt(np.convolve(mono, np.ones(w)/w, mode="same") + 1e-12)
        over = 20*np.log10(rms[a-a0:a-a0+b-a]) - thresh_db
        gr = np.where(over <= -knee/2, 0., np.where(over >= knee/2,
                      over*(1/ratio-1), (1/ratio-1)*(over+knee/2)**2/(2*knee)))
        if len(gr) % hop:
            gr = np.pad(gr, (0, hop-len(gr)%hop), mode="edge")
        ctrl[a//hop:(b+hop-1)//hop] = gr.reshape(-1, hop).min(1)
    return (10 ** (_smooth_control(ctrl, attack, release)/20)).astype(np.float32)


def apply_control(y, control):
    for a, b in blocks(len(y)):
        g = np.repeat(control[a//48:(b+47)//48], 48)[:b-a]
        y[a:b] *= g[:, None]


def limit_inplace(y, ceiling_db=-1.3, release=.15):
    need = np.minimum(1., 10**(ceiling_db/20) / np.maximum(true_peak_envelope(y), 1e-9))
    need = minimum_filter1d(need, size=2*int(.006*SR)+1)
    gdb = 20*np.log10(need)
    hop = 48
    ctrl = gdb[:len(gdb)//hop*hop].reshape(-1, hop).min(1)
    if len(gdb) % hop:
        ctrl = np.r_[ctrl, gdb[len(ctrl)*hop:].min()]
    smooth = _smooth_control(ctrl, .0005, release)
    min_gain = 1.
    for a, b in blocks(len(y)):
        gs = np.repeat(smooth[a//hop:(b+hop-1)//hop], hop)[:b-a]
        g = (10 ** (np.minimum(gs, gdb[a:b])/20)).astype(np.float32)
        min_gain = min(min_gain, float(g.min()))
        y[a:b] *= g[:, None]
    return float(20*np.log10(min_gain))


def _filter_inplace(y, b, a):
    zi = np.zeros((max(len(a), len(b))-1, 2), np.float64)
    for lo, hi in blocks(len(y)):
        z, zi = signal.lfilter(b, a, y[lo:hi], axis=0, zi=zi)
        y[lo:hi] = z.astype(np.float32)


def _hp_inplace(y, fc, order=2):
    b, a = signal.butter(order, fc/(SR/2), btype="high")
    _filter_inplace(y, b, a)


def _breath_env(n, wins, name, offset):
    env = np.ones(n, np.float32)
    fd, fu = int(.030*SR), int(.005*SR)
    floor = 10**(-45/20)
    for i0, i1, exempt in wins:
        if name in exempt:
            continue
        a, b = max(i0, offset), min(i1, offset+n)
        if b <= a:
            continue
        indices = np.arange(a-i0, b-i0)
        seg = np.full(b-a, floor, np.float32)
        down = indices < fd
        up = indices >= i1-i0-fu
        seg[down] = floor+(1-floor)*np.cos(indices[down]/max(1, fd-1)*np.pi/2)**2
        seg[up] = floor+(1-floor)*np.sin((indices[up]-(i1-i0-fu))/max(1, fu-1)*np.pi/2)**2
        env[a-offset:b-offset] = np.minimum(env[a-offset:b-offset], seg)
    return env


def _process_part(y, p, eq, wins):
    import mix as MX
    # Stateful filters are applied across the complete part after block panning.
    for a, b in blocks(len(y)):
        z = np.asarray(y[a:b]) * np.float32(10**(p.gain_db/20))
        z *= _breath_env(b-a, wins, p.name, a)[:, None]
        y[a:b] = MX.pan_width(z, p.pan, p.width)
    if p.depth > .05:
        b, a = signal.butter(1, 16000*(1-.55*p.depth)/(SR/2))
        _filter_inplace(y, b, a)
    spec = eq.get(p.name, (None, None))
    if spec[0]:
        _hp_inplace(y, spec[0])
    if spec[1]:
        b, a = signal.butter(2, spec[1]/(SR/2))
        _filter_inplace(y, b, a)
    if len(spec) > 2 and spec[2]:
        b, a = signal.butter(2, (spec[3] if len(spec)>3 else 6000.)/(SR/2), btype="high")
        zi = np.zeros((2, 2), np.float64)
        for lo, hi in blocks(len(y)):
            hp, zi = signal.lfilter(b, a, y[lo:hi], axis=0, zi=zi)
            y[lo:hi] += hp * np.float32(10**(spec[2]/20)-1)


def convolve_add(dry, send, irs, wins):
    """Engine overlap-add hall, blockwise, with its breath tail truncation."""
    starts = [0] + [i1 for _, i1, _ in wins]
    stops = [i0 for i0, _, _ in wins] + [None]
    fd, floor = int(.030*SR), 10**(-45/20)
    LL, LR, RL, RR = irs
    for k, start in enumerate(starts):
        end = starts[k+1] if k+1<len(starts) else len(send)
        cutoff = stops[k]
        for a in range(start, end, CHUNK):
            b = min(end, a+CHUNK)
            seg = np.asarray(send[a:b])
            if not np.any(seg):
                continue
            # oaconvolve is the engine's own bounded FFT convolution method.
            left = signal.oaconvolve(seg[:,0], LL) + signal.oaconvolve(seg[:,1], RL)
            right = signal.oaconvolve(seg[:,0], LR) + signal.oaconvolve(seg[:,1], RR)
            stop = min(len(dry), a+len(left))
            if cutoff is not None:
                stop = min(stop, cutoff+fd)
            if stop <= a:
                continue
            z = np.stack((left[:stop-a], right[:stop-a]), axis=1).astype(np.float32)
            if cutoff is not None and stop > cutoff:
                j0 = max(0, cutoff-a)
                t = np.arange(max(a, cutoff)-cutoff, stop-cutoff)
                z[j0:] *= (floor+(1-floor)*np.cos(t/max(1, fd-1)*np.pi/2)**2)[:,None]
            dry[a:stop] += z


def _apply_push(y, pushes):
    """The engine's local zero-phase split-band push; no full-film arrays."""
    sos = signal.butter(2, 150/(SR/2), "low", output="sos")
    for frame, db_hi, db_lo, hold, rel in pushes:
        T = int(round(frame*FRAME_N))
        a, b = max(0, T-int(.5*SR)), min(len(y), T+int((hold+rel+1)*SR))
        seg = np.asarray(y[a:b], np.float64)
        lo = signal.sosfiltfilt(sos, seg, axis=0)
        t = (np.arange(b-a)-(T-a-int(.005*SR)))/SR
        env = np.where(t<0, 0., np.where(t<hold, 1., np.cos(np.clip((t-hold)/rel, 0, 1)*np.pi/2)**2))
        y[a:b] = ((seg-lo)*10**(db_hi*env[:,None]/20)+lo*10**(db_lo*env[:,None]/20)).astype(np.float32)


def _apply_rides(y, rides):
    """Explicit cosine-interpolated (frame, dB) master rides, zero outside."""
    for (f0, db0), (f1, db1) in zip(rides, rides[1:]):
        a, b = max(0, round(f0*FRAME_N)), min(len(y), round(f1*FRAME_N))
        if b <= a:
            continue
        for lo in range(a, b, CHUNK):
            hi = min(b, lo+CHUNK)
            u = (np.arange(lo, hi)-a)/(b-a)
            db = db0+(db1-db0)*(1-np.cos(np.pi*u))/2
            y[lo:hi] *= (10**(db/20)).astype(np.float32)[:,None]


def _silence(y, windows, fade=True):
    for f0, f1 in windows:
        a, b = round(f0*FRAME_N), round(f1*FRAME_N)
        if fade:
            n = min(int(.030*SR), a)
            if n:
                y[a-n:a] *= np.cos(np.linspace(0, np.pi/2, n, dtype=np.float32))[:,None]**2
        y[a:b] = 0


def pcm_sha256(path, first=0, last=1440):
    with wave.open(str(path), "rb") as w:
        w.setpos(first*FRAME_N)
        return hashlib.sha256(w.readframes((last-first)*FRAME_N)).hexdigest()


def _protected_opening(source, frames=1440):
    with sf.SoundFile(source) as f:
        if f.samplerate != SR or f.channels != 2 or f.subtype != "PCM_24":
            raise ValueError("protected opening must be 48 kHz stereo PCM_24")
        return f.read(frames*FRAME_N, dtype="float32", always_2d=True)


def _splice_opening(y, opening, first=1440*FRAME_N):
    y[:first] = opening[:first]
    n = min(480, len(y)-first)
    if n:
        # Only the NEW side of the boundary changes; the protected prefix is exact.
        u = np.linspace(0, 1, n, dtype=np.float32)
        w = u*u*(3-2*u)
        y[first:first+n] = opening[first-1]*(1-w[:,None])+y[first:first+n]*w[:,None]


def _configure_sampler(output_dir, sample_root=None):
    import sampler
    import sampler_v2  # noqa: F401
    import sfz
    sampler.RAW_BUDGET, sampler.RS_BUDGET = 140e6, 220e6
    # Retain read-only metadata/pitch corrections; only a future _save_meta would
    # target this private directory. Loading the originals avoids repeated probes.
    original_cache = Path(sampler.CACHE)
    sampler._load_meta()
    fp = original_cache / "pitch_fix.json"
    sampler._FIX = json.loads(fp.read_text()) if fp.exists() else {}
    sampler.CACHE = str(output_dir)
    sampler.META_PATH = str(output_dir / "sample_meta.json")
    if sample_root:
        sampler.VSCO = sfz.VSCO = str(Path(sample_root).resolve())
        sampler._REG.clear()
    return sampler


def _score_identity(S):
    payload = dict(parts={k:p.to_dict() for k,p in S.used().items()}, eq=S.eq,
                   groups=S.groups, breaths=S.breaths, push=S.push,
                   fader=getattr(S, "fader", []), hard_silences=S.hard_silences,
                   mix_pipeline_revision=1,
                   engine_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                  for name in ("dsl.py", "sampler.py", "sampler_v2.py", "sfz.py",
                                               "synth.py", "synth_v2.py", "synth_v3.py",
                                               "mix.py", "voices_v3_C.py")})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def render(output_dir, *, master_only=False, opening_source=None, sample_root=None,
           event_file=None, rides_file=None, keep_intermediates=False):
    import mix as MX
    output_dir = Path(output_dir).resolve()
    # Shared legacy caches/outputs must never be accidentally selected.
    for rel in ("music/out", "music/cache", "renders"):
        forbidden = (ROOT/rel).resolve()
        if output_dir == forbidden or forbidden in output_dir.parents:
            raise ValueError(f"private output directory required; {rel} is shared")
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir).free < MAX_DISK:
        raise OSError("less than 1.5 GiB free before D audio render")
    sampler = _configure_sampler(output_dir, sample_root)
    import score_v3_D
    from barmap_D_score import DraftMap
    bm = DraftMap() if event_file is None else DraftMap(json.loads(Path(event_file).read_text()))
    S = score_v3_D.build(bm)
    problems = score_v3_D.check(S, bm)
    if problems:
        raise ValueError("score contract failed: " + "; ".join(problems))
    n = int(bm.frames * FRAME_N)
    if n != 18400000:
        raise ValueError("D v2 must be exactly 9200 frames / 18,400,000 sample frames")
    render_n = n+SR
    windows = list(S.hard_silences)
    identity = _score_identity(S)
    receipt_path = output_dir/"render_D_receipt.json"
    receipt = dict(frames=bm.frames, sample_frames=n, sample_rate=SR, channels=2,
                   subtype="PCM_24", score_identity=identity, stages=[],
                   events=bm.events, hard_silences=windows,
                   code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), HERE/"score_v3_D.py", HERE/"barmap_D_score.py")},
                   method="serial engine parts; block EQ/summation/overlap-add hall; bounded master controls",
                   measured=False)
    t0 = last_stage = time.monotonic()

    def stage(name):
        nonlocal last_stage
        now = time.monotonic()
        receipt["stages"].append(dict(name=name, seconds=now-last_stage,
                                     peak_rss_bytes=memory_guard()))
        last_stage = now
        receipt_path.write_text(json.dumps(receipt, indent=2, default=str)+"\n")
        print(f"{name}: {now-t0:.1f}s elapsed; peak RSS {peak_rss_bytes()/1024**3:.3f} GiB", flush=True)

    premaster_path = output_dir/"premaster_score_D.npy"
    identity_path = output_dir/"premaster_score_D.identity.json"
    if master_only:
        stored = json.loads(identity_path.read_text())
        if stored["score_identity"] != identity:
            raise ValueError("private premaster does not match the current score; run --render")
        pre = np.load(premaster_path, mmap_mode="r")
        if pre.shape != (render_n, 2):
            raise ValueError("private premaster has wrong length")
    else:
        pre = _new_map(premaster_path, render_n)
        wins = sorted((round(b0*BEAT_N), round(b1*BEAT_N), set(ex)) for b0,b1,ex in S.breaths)
        ir_path = ROOT/"music/cache/ir_hall.npy"
        irs = list(np.load(ir_path)) if ir_path.exists() else MX.make_ir(rt_mid=3.1)
        parts = S.used()
        groups = dict(S.groups)
        report = []
        for group in [None] + list(groups):
            dp, sp = output_dir/".D_bus_dry.npy", output_dir/".D_bus_send.npy"
            dry, send = _new_map(dp, render_n), _new_map(sp, render_n)
            for name, p in parts.items():
                grp = next((g for g in groups if name.startswith(g)), None)
                if grp != group:
                    continue
                started = time.monotonic()
                pd = p.to_dict()
                if p.kind == "sampler":
                    y = sampler.render_part(pd, render_n)
                else:
                    import synth_v3
                    y = synth_v3.render_part(pd, render_n)
                np.nan_to_num(y, copy=False)
                _process_part(y, p, S.eq, wins)
                energy = 0.
                for a, b in blocks(render_n):
                    z = y[a:b]
                    dry[a:b] += z*np.float32(1-.35*p.depth)
                    send[a:b] += z*np.float32(p.send)
                    energy += float(np.sum(z.astype(np.float64)**2))
                del y
                memory_guard()
                report.append(dict(part=name, seconds=time.monotonic()-started,
                                   rms_db=float(10*np.log10(energy/(2*render_n)+1e-24))))
                print(f"  {name}: {report[-1]['seconds']:.1f}s; RSS peak {peak_rss_bytes()/1024**3:.3f} GiB", flush=True)
            sampler.clear_caches()
            convolve_add(dry, send, irs, wins)
            _remove_map(send, sp)
            if group is not None:
                cfg = groups[group]
                limit_inplace(dry, cfg.get("ceiling_db", -6.), cfg.get("release", .08))
            for a,b in blocks(render_n):
                pre[a:b] += dry[a:b]
            _remove_map(dry, dp)
            stage("mix bus " + (group or "main"))
        _hp_inplace(pre, 22.)
        _apply_push(pre, S.push)
        if getattr(S, "fader", None):
            _apply_rides(pre, [(t*FPS, db) for t, db in S.fader])
        _silence(pre, windows, fade=False)
        pre.flush()
        identity_path.write_text(json.dumps(dict(score_identity=identity, parts=report), indent=2)+"\n")
        stage("premaster")
    sampler.clear_caches()
    gc.collect()
    if opening_source is None:
        opening_source = ROOT/"music/out/v3/sound_AP2_score.wav"
    opening_source = Path(opening_source).resolve()
    opening = _protected_opening(opening_source)
    opening_tp = true_peak_db(opening[:1440*FRAME_N])
    if opening_tp > -1.2:
        raise ValueError(f"protected opening itself is {opening_tp:.4f} dBTP; cannot preserve PCM and meet -1.2 dBTP")
    receipt["opening"] = dict(source_pcm_sha256=pcm_sha256(opening_source),
                              frames=1440, true_peak_db=opening_tp, join_after_boundary_samples=480,
                              join_method="10 ms smoothstep from final protected sample into Mountain; no A6 source")
    rides = (json.loads(Path(rides_file).read_text()) if rides_file
             else list(getattr(S, "master_rides", [])))
    receipt["master_rides"] = rides
    temp_path = output_dir/".D_master.npy"
    y = _new_map(temp_path, n)
    pre_lufs = lufs(pre[:n])
    receipt["premaster"] = dict(lufs=pre_lufs, true_peak_db=true_peak_db(pre[:n]))
    gain = 10**((-16-pre_lufs)/20)
    iterations = []
    for it in range(12):
        for a,b in blocks(n):
            y[a:b] = pre[a:b]*np.float32(gain)
        _hp_inplace(y, 8., order=1)
        if rides:
            _apply_rides(y, rides)
        comp = glue_gain(y)
        apply_control(y, comp)
        min_comp = float(20*np.log10(comp.min()))
        del comp
        min_limit = limit_inplace(y)
        _silence(y, windows)
        _splice_opening(y, opening)
        measured = lufs(y)
        iterations.append(dict(iteration=it, lufs=measured, gain_db=float(20*np.log10(gain)),
                               compression_min_db=min_comp, limiter_min_db=min_limit))
        print(f"  master {it}: {measured:.3f} LUFS; gain {20*np.log10(gain):+.3f} dB", flush=True)
        memory_guard()
        if abs(measured+16) < .055:
            break
        gain *= 10**((-16-measured)/20)
    if abs(measured+16) > .15:
        raise ValueError(f"D master did not converge to -16 LUFS: {measured:.3f}")
    tp = true_peak_db(y)
    if tp > -1.2:
        factor = np.float32(10**((-1.25-tp)/20))
        for a,b in blocks(n-1440*FRAME_N):
            y[1440*FRAME_N+a:1440*FRAME_N+b] *= factor
        _splice_opening(y, opening)
        tp = true_peak_db(y)
    if tp > -1.2:
        raise ValueError(f"post-splice D master exceeds -1.2 dBTP: {tp:.4f}")
    output = output_dir/"score_D_draft.wav"
    rng = np.random.default_rng(0)
    with sf.SoundFile(output, "w", samplerate=SR, channels=2, subtype="PCM_24") as f:
        for a,b in blocks(n):
            z = MX.tpdf_dither_24(np.asarray(y[a:b]), rng)
            # Do not dither the protected PCM or the deliberately digital silences.
            if a < 1440*FRAME_N:
                k = min(b, 1440*FRAME_N)-a
                z[:k] = y[a:a+k]
            for f0,f1 in windows:
                lo,hi = max(a, f0*FRAME_N), min(b, f1*FRAME_N)
                if hi>lo:
                    z[lo-a:hi-a] = 0.
            f.write(z)
    receipt["opening"]["output_pcm_sha256"] = pcm_sha256(output)
    if receipt["opening"]["output_pcm_sha256"] != receipt["opening"]["source_pcm_sha256"]:
        raise ValueError("protected opening PCM changed")
    receipt["master"] = dict(lufs=lufs(y), true_peak_db=tp, iterations=iterations)
    receipt["output"] = output.name
    receipt["measured"] = True
    _remove_map(y, temp_path)
    del pre, opening
    if not keep_intermediates:
        # Keep the one premaster for cheap measured-frame/master revisions; no part cache.
        for path in output_dir.glob(".D_bus_*.npy"):
            path.unlink()
    stage("master and write")
    receipt["peak_rss_bytes"] = memory_guard()
    receipt["peak_rss_gib"] = peak_rss_bytes()/1024**3
    receipt["elapsed_seconds"] = time.monotonic()-t0
    receipt["output_directory_bytes"] = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    if receipt["output_directory_bytes"] > MAX_DISK:
        raise OSError("private score output exceeds 1.5 GiB disk limit")
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str)+"\n")
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--master-only", action="store_true")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--opening-source")
    ap.add_argument("--sample-root")
    ap.add_argument("--events", help="explicit named event overrides JSON")
    ap.add_argument("--rides", help="optional ordered [frame, dB] points for new material")
    ap.add_argument("--keep-intermediates", action="store_true")
    args = ap.parse_args()
    render(args.output_dir, master_only=args.master_only, opening_source=args.opening_source,
           sample_root=args.sample_root, event_file=args.events, rides_file=args.rides,
           keep_intermediates=args.keep_intermediates)


if __name__ == "__main__":
    main()
