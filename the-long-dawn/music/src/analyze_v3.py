"""v3 analysis battery = my ears (red team 5.4).  Called by render_v3; or:
    python analyze_v3.py B [--fallback]

Writes music/analysis/v3/<name>/:
  report.txt     format (length, stems sum, clipping, DC), loudness (whole + per section) against the
                 level map, true peak, the spectral-centroid arc against the intended arc, the sync
                 table (expected vs measured onsets), the breaths, note checks (stuck / overlapping /
                 too fast / one sample repeated > 8 per second), clicks
  overview.png   short-term loudness of final / score / effects with sections, level-map bands, sync marks
  spec.png       log-frequency spectrogram of the score stem, one row per ~60 s, sections marked
  roll.png       piano roll of the score coloured by family, sections and bar-map events marked
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyloudnorm as pyln  # noqa: E402

import analyze_v2 as A2  # noqa: E402  (measure(): onset detection, clicks())
from timeline_v3 import SR, FPS, BarMap, bar_beat, bt  # noqa: E402

MUSIC = os.path.dirname(HERE)
OUT = os.path.join(MUSIC, "out", "v3")
CACHE = os.path.join(MUSIC, "cache", "v3")
FAM = {"strings": "#3b7dd8", "brass": "#d8843b", "winds": "#3bb07d", "keys": "#9b59b6", "perc": "#555555",
       "synth": "#d83b8e"}
DYN_LUFS = {"niente": (-80, -45), "ppp": (-60, -32), "pp": (-45, -27), "p": (-38, -22), "mp": (-31, -18),
            "mf": (-26, -14), "f": (-22, -11)}


def st_loudness(x, win=3.0, hop=0.5):
    meter = pyln.Meter(SR)
    out = []
    t = 0.0
    while t + win <= len(x) / SR + 1e-6:
        seg = np.asarray(x[int(t * SR): int((t + win) * SR)], np.float64)
        v = meter.integrated_loudness(seg) if np.abs(seg).max() > 1e-6 else -120.0
        out.append((t + win / 2, v if np.isfinite(v) else -120.0))
        t += hop
    return np.array(out) if out else np.zeros((0, 2))


def centroid_track(x, hop=0.05, n=4096):
    mono = np.asarray(x, np.float64).mean(1)
    f, t, Z = signal.stft(mono, SR, nperseg=n, noverlap=n - int(hop * SR))
    P = np.abs(Z) ** 2
    e = P.sum(0)
    c = (P * f[:, None]).sum(0) / (e + 1e-20)
    lvl = 10 * np.log10(e / (n / 2) + 1e-20)
    return t, c, lvl


def measure3(x, T, W, kind):
    """onsets for the v3 sync table.  'hit' / 'soft': analyze_v2.measure.  'bloom': the first audible energy
    after a silence (35 dB under the window's peak).  'pitch:<midi>': a legato pitch change, found in the band of
    the new note (its fundamental and second partial, +-60 cents) where its energy rises half-way (in dB) from
    before the change to after it."""
    if kind in ("hit", "soft"):
        return A2.measure(x, T, W, kind)
    if kind == "bloom":
        t, e = A2.env_db(x, T - W - 0.05, T + W + 1.5, hp=40.0, win=0.005, hop=0.001)
        thr = e.max() - 35
        idx = np.where((e > thr) & (t >= T - W))[0]
        return (float(t[idx[0]]) if len(idx) else None), float(e.max() - np.median(e))
    if kind.startswith("arrive:"):
        # a soft entry ARRIVES when its fundamental band reaches 8 dB under the note's peak
        f0 = 440 * 2 ** ((int(kind.split(":")[1]) - 69) / 12)
        a, b = max(0, int((T - W - 0.4) * SR)), min(len(x), int((T + W + 0.8) * SR))
        seg = np.asarray(x[a:b], np.float64).mean(1)
        sos = signal.butter(4, [f0 * 2 ** -0.05 / (SR / 2), f0 * 2 ** 0.05 / (SR / 2)], btype="band", output="sos")
        band = signal.sosfiltfilt(sos, seg) ** 2
        w = int(0.01 * SR)
        env = 10 * np.log10(np.convolve(band, np.ones(w) / w, mode="same") + 1e-14)
        t = (a + np.arange(len(env))) / SR
        m = (t >= T - W) & (t <= T + 0.8)
        pk = env[m].max()
        idx = np.where((env > pk - 8) & (t >= T - W))[0]
        return (float(t[idx[0]]) if len(idx) else None), 8.0
    if kind.startswith("band:"):
        # a soft entry measured in the new note's fundamental band (the part is still ringing another note)
        f0 = 440 * 2 ** ((int(kind.split(":")[1]) - 69) / 12)
        a, b = max(0, int((T - W - 0.4) * SR)), min(len(x), int((T + W + 0.6) * SR))
        seg = np.asarray(x[a:b], np.float64).mean(1)
        sos = signal.butter(4, [f0 * 2 ** -0.05 / (SR / 2), f0 * 2 ** 0.05 / (SR / 2)], btype="band", output="sos")
        band = signal.sosfiltfilt(sos, seg) ** 2
        w = int(0.005 * SR)
        env = 10 * np.log10(np.convolve(band, np.ones(w) / w, mode="same") + 1e-14)
        t = (a + np.arange(len(env))) / SR
        pk = env[(t >= T - W) & (t <= T + W + 0.6)].max()
        idx = np.where((env > pk - 20) & (t >= T - W))[0]
        return (float(t[idx[0]]) if len(idx) else None), 20.0
    if kind.startswith("pitch:"):
        f0 = 440 * 2 ** ((int(kind.split(":")[1]) - 69) / 12)
        a, b = max(0, int((T - W - 0.4) * SR)), min(len(x), int((T + W + 0.6) * SR))
        seg = np.asarray(x[a:b], np.float64).mean(1)
        lo, hi = f0 * 2 ** (-0.05), f0 * 2 ** (0.05)          # the fundamental only (an octave leap would
        sos = signal.butter(4, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")   # leak into h2)
        band = signal.sosfiltfilt(sos, seg) ** 2
        w = int(0.02 * SR)
        env = 10 * np.log10(np.convolve(band, np.ones(w) / w, mode="same") + 1e-14)
        t = (a + np.arange(len(env))) / SR
        pre = np.median(env[(t >= T - W - 0.4) & (t < T - W)]) if np.any(t < T - W) else env.min()
        post = env[(t >= T) & (t <= T + W + 0.6)].max()
        thr = pre + max(6.0, 0.35 * (post - pre))
        idx = np.where((env > thr) & (t >= T - W))[0]
        return (float(t[idx[0]]) if len(idx) else None), float(post - pre)
    return A2.measure(x, T, W, "soft")


def main(bm, S, name):
    if isinstance(bm, str):
        bm = BarMap(bm)
    d = os.path.join(MUSIC, "analysis", "v3", name)
    os.makedirs(d, exist_ok=True)
    rep = []
    P = rep.append
    fin, sr = sf.read(os.path.join(OUT, name + ".wav"), dtype="float32", always_2d=True)
    info = sf.info(os.path.join(OUT, name + ".wav"))
    sc_p = os.path.join(OUT, name + "_score.wav")
    has_stems = os.path.exists(sc_p)
    sco = sf.read(sc_p, dtype="float32", always_2d=True)[0] if has_stems else fin
    sfx = sf.read(os.path.join(OUT, name + "_sfx.wav"), dtype="float32", always_2d=True)[0] if has_stems else None
    P(f"THE LONG DAWN v3 - {name}  (bar map {bm.path}, status {bm.d.get('status', '?')})")
    P("=" * 100)
    # ---- format
    P("FORMAT")
    ok_len = len(fin) == bm.n
    P(f"  {info.samplerate} Hz, {info.subtype}, {info.channels} ch, {len(fin)} samples = {len(fin) / SR:.3f} s "
      f"(bar map: {bm.n} = {bm.bars} bars) -> {'PASS' if ok_len and sr == SR else 'FAIL'}")
    pk = float(np.abs(fin).max())
    P(f"  sample peak {20 * np.log10(pk + 1e-12):.2f} dBFS -> {'PASS (no clipping)' if pk < 0.999 else 'FAIL (clipping)'}")
    P(f"  DC offset L/R {fin[:, 0].mean():.2e} / {fin[:, 1].mean():.2e}")
    if has_stems:
        err = float(np.abs(sco + sfx - fin).max())
        P(f"  stems: score + sfx - final max error {err:.2e} -> {'PASS' if err < 1e-5 else 'FAIL'}")
    head = np.abs(fin[: int(0.005 * SR)]).max()
    tail = np.abs(fin[-int(0.005 * SR):]).max()
    P(f"  from silence {20 * np.log10(head + 1e-12):.0f} dBFS in the first 5 ms, to silence "
      f"{20 * np.log10(tail + 1e-12):.0f} dBFS in the last 5 ms")
    # ---- loudness
    meter = pyln.Meter(SR)
    P("\nLOUDNESS (EBU R128 integrated / true peak)")
    for lab, x in (("final", fin), ("score", sco if has_stems else None), ("sfx", sfx)):
        if x is None:
            continue
        L = meter.integrated_loudness(np.asarray(x, np.float64))
        P(f"  {lab:6s} {L:7.2f} LUFS   TP {A2.true_peak_db(x):6.2f} dBTP")
    stf = st_loudness(fin)
    sts = st_loudness(sco) if has_stems else stf
    lev = {k: (lo, hi) for k, lo, hi in S.levels}
    anchor = float(stf[:, 1].max()) if len(stf) else 0.0
    at_s = float(stf[int(np.argmax(stf[:, 1])), 0]) if len(stf) else 0.0
    P(f"\n  the film's loudest 3 s: {anchor:.1f} LUFS at {at_s:.1f} s (the level map is relative to it)")
    P("  per section (final: integrated / short-term max / median, and max / median relative to the loudest;"
      " score short-term median) vs the level map")
    fails = 0
    for s in bm.sections:
        a, b = int(s["t0"] * SR), int(s["t1"] * SR)
        seg = np.asarray(fin[a:b], np.float64)
        Li = meter.integrated_loudness(seg) if (b - a) > 0.5 * SR and np.abs(seg).max() > 1e-6 else -120
        Li = Li if np.isfinite(Li) else -120
        m = (stf[:, 0] >= s["t0"] + 1.0) & (stf[:, 0] <= s["t1"] - 1.0) if len(stf) else []
        if len(stf) and not np.any(m):
            m = (np.abs(stf[:, 0] - (s["t0"] + s["t1"]) / 2) <= 1.6)
        vmax = float(stf[m, 1].max()) if len(stf) and np.any(m) else -120
        vmed = float(np.median(stf[m, 1])) if len(stf) and np.any(m) else -120
        smed = float(np.median(sts[m, 1])) if len(sts) and np.any(m) else -120
        tp = A2.true_peak_db(fin[a:b]) if b > a else -120
        verdict = ""
        rmax, rmed = vmax - anchor, vmed - anchor
        if s["id"] in lev:
            lo, hi = lev[s["id"]]
            ok = (rmax <= hi + 0.5) and (rmed >= lo - 0.5)
            verdict = f"target {lo:+.1f}..{hi:+.1f} LU -> {'PASS' if ok else 'FAIL'}"
            fails += 0 if ok else 1
        P(f"  {s['id']:5s} {s.get('name', '')[:22]:22s} {Li:7.2f} / {vmax:6.1f} / {vmed:6.1f}  rel {rmax:+5.1f} / {rmed:+5.1f}"
          f"  score {smed:6.1f}  TP {tp:6.1f}  {verdict}")
    P(f"  level map: {fails} section(s) outside their band")
    for (label, t0, t1, max_rel) in getattr(S, "rules", []):
        m = (stf[:, 0] >= t0 + 1.5) & (stf[:, 0] <= t1 - 1.5)
        v = float(stf[m, 1].max()) - anchor if np.any(m) else -99
        P(f"  RULE {label}: {v:+.1f} LU (limit {max_rel:+.1f}) -> {'PASS' if v <= max_rel + 0.05 else 'FAIL'}")
    # ---- centroid
    P("\nSPECTRAL CENTROID of the score stem (median over frames above -60 dB) vs the intended arc")
    t, c, lvl = centroid_track(sco)
    cmap = {k: (lo, hi) for k, lo, hi in S.centroid}
    cf = 0
    arc = []
    for s in bm.sections:
        m = (t >= s["t0"]) & (t < s["t1"]) & (lvl > -60)
        med = float(np.median(c[m])) if np.any(m) else float("nan")
        arc.append(med)
        v = ""
        if s["id"] in cmap and np.isfinite(med):
            lo, hi = cmap[s["id"]]
            ok = lo <= med <= hi
            cf += 0 if ok else 1
            v = f"target {lo:.0f}-{hi:.0f} Hz -> {'PASS' if ok else 'FAIL'}"
        P(f"  {s['id']:5s} {med:8.0f} Hz  {v}")
    P(f"  centroid arc: {cf} section(s) off")
    # ---- sync
    P("\nSYNC (expected vs measured).  Hits: steepest onset, +-10 ms.  Soft entries (arrive): each is started early by\n"
      "  its own sample's attack time, and is measured where its fundamental ARRIVES (8 dB under the note's peak),\n"
      "  +-50 ms.  Blooms: first audible energy after a silence, -70..+35 ms.  Legato pitch changes: where the new\n"
      "  note's fundamental has risen 6 dB, -90..+60 ms (a legato change is a crossfade centred on the beat).")
    man_p = os.path.join(CACHE, f"manifest_{name}.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {}
    import render_v3 as R
    sf_n = 0
    rows = []
    for (T, label, part, W, kind) in S.sync:
        if part == "final":
            x = fin
            off = 0
        elif part == "score":
            x = sco
            off = 0
        elif part == "sfx" and sfx is not None:
            x = sfx
            off = 0
        elif all(q in man for q in part.split("+")):
            a = max(0, int((T - W - 0.6) * SR))
            b = int((T + W + 1.0) * SR)
            x = np.zeros((b - a, 2), np.float32)
            for q in part.split("+"):                      # "voice+voice_core": the layers as heard, summed
                o, y = R.load_stem(q, man[q])
                g = np.float32(10 ** (S.parts[q].gain_db / 20)) if q in S.parts else np.float32(1)
                s0, s1 = max(a, o), min(b, o + len(y))
                if s1 > s0:
                    x[s0 - a:s1 - a] += y[s0 - o:s1 - o] * g
            off = a
        else:
            P(f"  {T:8.3f} s {label}: part {part} missing")
            continue
        tm, strength = measure3(x, T - off / SR, W, kind)
        if tm is None:
            P(f"  {T:8.3f} s  {label:44s} {part:12s} NOT FOUND")
            sf_n += 1
            continue
        tm += off / SR
        dms = (tm - T) * 1000
        ok = (abs(dms) <= 10 if kind == "hit" else (-90 <= dms <= 60) if kind.startswith("pitch") else
              (-50 <= dms <= 50) if kind.startswith("arrive") else (-70 <= dms <= 35))
        sf_n += 0 if ok else 1
        b, bb = bar_beat(T / (60 / 72))
        rows.append((T, dms))
        P(f"  {T:8.3f} s  bar {b:3d}.{bb:<4.2f} {label:44s} {part:12s} {dms:+7.1f} ms ({kind}) {'PASS' if ok else 'FAIL'}")
    P(f"  sync: {sf_n} failure(s) of {len(S.sync)}")
    # ---- breaths
    bp = os.path.join(CACHE, f"breath_probe_{name}.json")
    if os.path.exists(bp):
        P("\nBREATHS (score dry + hall around each window, pre-master)")
        for b in json.load(open(bp)):
            P(f"  {b['start_s']:.3f}-{b['end_s']:.3f} s: before {b['before_db']:.1f} dB, inside {b['inside_db']:.1f} dB, "
              f"after {b['after_db']:.1f} dB -> depth {b['before_db'] - b['inside_db']:.1f} dB")
    # ---- notes
    import kit_v3 as K
    parts = S.used()
    w = K.check_notes(parts, bm.bars * 4) + K.check_rates(parts)
    P(f"\nNOTES: {len(w)} warning(s) (stuck / overlapping / too fast / repeated samples)")
    for x in w[:60]:
        P("  " + x)
    # ---- clicks
    ck = A2.clicks(sco)
    P(f"\nCLICKS in the score stem: {len(ck)} candidate(s)" +
      ("" if not len(ck) else f" at {', '.join(f'{i / SR:.3f}' for i in ck[:12])} s") +
      "   (the effects' crackle is intentional and not scanned)")
    open(os.path.join(d, "report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))
    # ---- images
    _overview(d, bm, S, stf, sts, st_loudness(sfx) if sfx is not None else None, name)
    _spec(d, bm, sco, name)
    _roll(d, bm, S, name)
    print("analysis ->", d)


def _marks(ax, bm, ymin, ymax, events=True):
    for s in bm.sections:
        ax.axvline(s["t0"], color="k", lw=0.6, alpha=0.6)
        ax.text(s["t0"] + 0.3, ymax - (ymax - ymin) * 0.06, s["id"], fontsize=7)
    if events:
        for e in bm.events:
            if e.get("lock") == "hard":
                ax.axvline(e["t"], color="#c0392b", lw=0.4, alpha=0.5)


def _overview(d, bm, S, stf, sts, stx, name):
    fig, ax = plt.subplots(figsize=(18, 4.2))
    if len(stf):
        ax.plot(stf[:, 0], stf[:, 1], color="k", lw=1.2, label="final")
    if sts is not None and len(sts):
        ax.plot(sts[:, 0], sts[:, 1], color="#3b7dd8", lw=0.9, label="score")
    if stx is not None and len(stx):
        ax.plot(stx[:, 0], stx[:, 1], color="#27ae60", lw=0.8, label="effects")
    lev = {k: (lo, hi) for k, lo, hi in S.levels}
    for s in bm.sections:
        if s["id"] in lev:
            lo, hi = lev[s["id"]]
            ax.fill_between([s["t0"], s["t1"]], lo, hi, color="#f39c12", alpha=0.12)
    for (T, label, part, W, kind) in S.sync:
        ax.plot([T], [-58], marker="|", color="#8e44ad", ms=8)
    _marks(ax, bm, -60, -8)
    ax.set_ylim(-60, -8)
    ax.set_xlim(0, bm.seconds)
    ax.set_ylabel("short-term LUFS (3 s)")
    ax.set_xlabel("s")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{name}: short-term loudness; orange = level map; red = hard sync points; purple = measured syncs")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "overview.png"), dpi=70)
    plt.close(fig)


def _spec(d, bm, x, name, row_s=60.0):
    mono = np.asarray(x, np.float64).mean(1)
    rows = int(np.ceil(len(mono) / SR / row_s))
    fig, axs = plt.subplots(rows, 1, figsize=(18, 2.6 * rows), squeeze=False)
    f, t, Z = signal.stft(mono, SR, nperseg=4096, noverlap=4096 - 1200)
    Pdb = 10 * np.log10(np.abs(Z) ** 2 + 1e-14)
    mx = np.percentile(Pdb, 99.9)
    fm = (f >= 30) & (f <= 16000)
    for r in range(rows):
        ax = axs[r, 0]
        m = (t >= r * row_s) & (t < (r + 1) * row_s)
        if not np.any(m):
            continue
        ax.pcolormesh(t[m], f[fm], Pdb[np.ix_(fm, m)], vmin=mx - 80, vmax=mx, cmap="magma", shading="auto")
        ax.set_yscale("log")
        ax.set_ylim(30, 16000)
        _marks(ax, bm, 30, 16000, events=False)
        ax.set_xlim(r * row_s, min((r + 1) * row_s, bm.seconds))
    axs[0, 0].set_title(f"{name}: score stem, log-frequency spectrogram (80 dB range)")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "spec.png"), dpi=60)
    plt.close(fig)


def _roll(d, bm, S, name):
    parts = S.used()
    fig, ax = plt.subplots(figsize=(22, 8))
    for pn, p in parts.items():
        col = FAM.get(p.bus, "#888")
        for nt in p.notes:
            if nt.pitch is None:
                continue
            lvl = nt.vel if nt.vel is not None else p.level_at(nt.start)
            ax.add_patch(plt.Rectangle((bt(nt.start), nt.pitch - 0.4), bt(nt.dur), 0.8, color=col,
                                       alpha=0.2 + 0.7 * min(1, lvl)))
    _marks(ax, bm, 20, 104)
    for e in bm.events:
        if e.get("lock") == "hard":
            ax.text(e["t"], 21, e["id"].split(".")[-1][:14], rotation=90, fontsize=5, va="bottom", color="#c0392b")
    ax.set_xlim(0, bm.seconds)
    ax.set_ylim(20, 104)
    ax.set_yticks(range(24, 104, 12))
    ax.set_yticklabels([f"C{o}" for o in range(1, 8)])
    ax.set_xlabel("s")
    ax.set_title(f"{name}: piano roll (blue strings, orange brass, green winds, purple keys, pink synth)")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "roll.png"), dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    cut = sys.argv[1].upper()
    fb_ = "--fallback" in sys.argv
    bm_ = BarMap(cut)
    if fb_:
        import fallback_v3
        S_ = fallback_v3.build(bm_)
        main(bm_, S_, f"fallback_{cut}")
    else:
        S_ = __import__(f"score_v3_{cut}").build(bm_)
        main(bm_, S_, f"final_{cut}")


def review_sheet(name, out_path, title=None):
    """one PNG for the director: the loudness overview, the piano roll and the spectrogram, stacked, with the
    report's key lines on top"""
    from PIL import Image, ImageDraw
    d = os.path.join(MUSIC, "analysis", "v3", name)
    ims = [Image.open(os.path.join(d, f)).convert("RGB") for f in ("overview.png", "roll.png", "spec.png")
           if os.path.exists(os.path.join(d, f))]
    W = max(i.width for i in ims)
    rep = open(os.path.join(d, "report.txt")).read().splitlines()
    keys = [ln for ln in rep if any(k in ln for k in ("LUFS   TP", "-> PASS", "-> FAIL", "failure", "level map:",
                                                     "centroid arc:", "NOTES:", "CLICKS", "depth"))][:26]
    head = 18 * (len(keys) + 2)
    sheet = Image.new("RGB", (W, head + sum(i.height for i in ims)), "white")
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 4), title or name, fill="black")
    for k, ln in enumerate(keys):
        dr.text((10, 22 + 18 * k), ln[:180], fill=("#b03a2e" if "FAIL" in ln else "black"))
    y = head
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(out_path, quality=85)
    return out_path
