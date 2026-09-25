"""Synthesised instruments (everything VSCO-2-CE lacks).

Each voice renders a Part (dict) into a stereo float32 buffer.  All use the
part's notes (pitch, start beat, dur beats, vel, kw) and dynamics curve.
"""
import numpy as np
from scipy import signal

from dsl import SR, BEAT_S, BEAT_N, dyn_array, dyn_at

TWOPI = 2 * np.pi


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def _t(n):
    return np.arange(n, dtype=np.float64) / SR


def _place(buf, y, i0):
    a, b = max(0, i0), min(len(buf), i0 + len(y))
    if b > a:
        buf[a:b] += y[a - i0:b - i0]


def _pan(mono, pan):
    th = (pan + 1) * np.pi / 4
    return np.stack([mono * np.cos(th), mono * np.sin(th)], axis=1) * np.sqrt(2)


def _smooth_noise(rng, n, rate_hz, fs=SR):
    """Band-limited random walk (for drift / jitter), ~unit std."""
    k = max(2, int(n * rate_hz / fs) + 3)
    pts = rng.normal(0, 1, k)
    x = np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)
    return x


def _env_adsr(n, a, r, rel_start=None):
    e = np.ones(n)
    an = max(1, int(a * SR))
    an = min(an, n)
    e[:an] = np.sin(np.linspace(0, np.pi / 2, an)) ** 2
    if rel_start is not None and rel_start < n:
        rn = max(1, n - rel_start)
        e[rel_start:] *= np.exp(-np.linspace(0, 6, rn)) * np.linspace(1, 0, rn) ** 0.5
    return e


def _biquad_bp(f, bw):
    """RBJ bandpass (constant 0 dB peak gain)."""
    w0 = TWOPI * f / SR
    q = f / bw
    alpha = np.sin(w0) / (2 * q)
    b = np.array([alpha, 0, -alpha])
    a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
    return b / a[0], a / a[0]


# ---------------------------------------------------------------------------
# CHOIR  (source-filter: detuned glottal wavetables -> formant bank)
# ---------------------------------------------------------------------------
FORMANTS = {
    # vowel: register: (freqs, amps dB, bws)
    "a": {
        "bass": ([600, 1040, 2250, 2450, 2750], [0, -7, -9, -9, -20], [60, 70, 110, 120, 130]),
        "tenor": ([650, 1080, 2650, 2900, 3250], [0, -6, -7, -8, -22], [80, 90, 120, 130, 140]),
        "alto": ([800, 1150, 2800, 3500, 4950], [0, -4, -20, -36, -60], [80, 90, 120, 130, 140]),
        "sop": ([800, 1150, 2900, 3900, 4950], [0, -6, -32, -20, -50], [80, 90, 120, 130, 140]),
    },
    "o": {
        "bass": ([400, 750, 2400, 2600, 2900], [0, -11, -21, -20, -40], [40, 80, 100, 120, 120]),
        "tenor": ([400, 800, 2600, 2800, 3000], [0, -10, -12, -12, -26], [40, 80, 100, 120, 120]),
        "alto": ([450, 800, 2830, 3500, 4950], [0, -9, -16, -28, -55], [70, 80, 100, 130, 135]),
        "sop": ([450, 800, 2830, 3800, 4950], [0, -11, -22, -22, -50], [70, 80, 100, 130, 135]),
    },
    "u": {
        "bass": ([350, 600, 2400, 2675, 2950], [0, -20, -32, -28, -36], [40, 80, 100, 120, 120]),
        "tenor": ([350, 600, 2700, 2900, 3300], [0, -20, -17, -14, -26], [40, 60, 100, 120, 120]),
        "alto": ([325, 700, 2530, 3500, 4950], [0, -12, -30, -40, -64], [50, 60, 170, 180, 200]),
        "sop": ([325, 700, 2700, 3800, 4950], [0, -16, -35, -40, -60], [50, 60, 170, 180, 200]),
    },
}


def _register(midi):
    if midi < 50:
        return "bass"
    if midi < 60:
        return "tenor"
    if midi < 69:
        return "alto"
    return "sop"


_WT = {}


def _glottal_table(nharm, tilt, seed):
    key = (nharm, round(tilt, 2), seed)
    if key in _WT:
        return _WT[key]
    rng = np.random.default_rng(seed)
    N = 4096
    ph = np.arange(N) / N
    k = np.arange(1, nharm + 1)
    amps = k ** (-tilt)
    phases = rng.uniform(0, TWOPI, nharm)
    tab = (amps[:, None] * np.sin(TWOPI * k[:, None] * ph[None, :] + phases[:, None])).sum(0)
    tab /= np.abs(tab).max()
    tab = np.append(tab, tab[0]).astype(np.float32)
    _WT[key] = tab
    return tab


def _osc(tab, freq_arr, phase0=0.0):
    ph = (phase0 + np.cumsum(freq_arr) / SR) % 1.0
    idx = ph * (len(tab) - 1)
    i0 = idx.astype(np.int64)
    u = (idx - i0).astype(np.float32)
    return tab[i0] * (1 - u) + tab[np.minimum(i0 + 1, len(tab) - 1)] * u


def choir(part, total_n):
    rng = np.random.default_rng(part["seed"])
    prm = part["params"]
    vowel = prm.get("vowel", "a")
    nvoices = prm.get("voices", 8)
    breath = prm.get("breath", 0.05)
    spread = prm.get("spread", 0.8)
    center = part.get("pan", 0.0)
    out = np.zeros((total_n, 2), np.float32)
    # group sources per register so each register is filtered once
    groups = {}
    for nt in part["notes"]:
        reg = nt["kw"].get("reg") or _register(nt["pitch"])
        groups.setdefault(reg, []).append(nt)
    for reg, notes in groups.items():
        srcL = np.zeros(total_n, np.float32)
        srcR = np.zeros(total_n, np.float32)
        for nt in notes:
            f0 = hz(nt["pitch"])
            b0 = nt["start"]
            atk = nt["kw"].get("atk", prm.get("atk", 0.35))
            rel = nt["kw"].get("rel", prm.get("rel", 0.8))
            vw = nt["kw"].get("vowel", vowel)
            dur_s = nt["dur"] * BEAT_S
            n = int((dur_s + rel + 0.3) * SR)
            i0 = int(round(b0 * BEAT_S * SR - atk * 0.35 * SR))
            if nt["vel"] is not None and not part["dyn"]:
                lev = np.full(n, nt["vel"], np.float32)
            else:
                lev = dyn_array(part["dyn"], b0 - atk * 0.35 / BEAT_S, n, BEAT_N)
                if nt["vel"] is not None:
                    lev = lev * (nt["vel"] / max(1e-3, float(lev[int(atk * 0.35 * SR)])))
            g = (10 ** (30 * np.log10(np.maximum(lev, 0.03)) / 20)).astype(np.float32)
            nh = max(4, int(min(9000, SR / 2.2) / f0))
            tilt = 1.25 - 0.25 * float(np.mean(lev))
            nv = nt["kw"].get("voices", nvoices)
            for v in range(nv):
                tab = _glottal_table(nh, tilt, int(rng.integers(0, 4)))
                det = rng.normal(0, 7) / 1200.0
                vr = rng.uniform(4.6, 6.0)
                vd = rng.uniform(10, 26) / 1200.0
                vdel = rng.uniform(0.25, 0.7)
                tt = _t(n)
                vib_env = np.clip((tt - vdel) / 0.6, 0, 1)
                drift = _smooth_noise(rng, n, 1.2) * (4 / 1200.0)
                jit = _smooth_noise(rng, n, 25.0) * (1.5 / 1200.0)
                fr = f0 * 2 ** (det + vd * vib_env * np.sin(TWOPI * vr * tt + rng.uniform(0, 6.28)) + drift + jit)
                s = _osc(tab, fr, rng.uniform(0, 1))
                # aspiration noise (breathy), modulated by glottal cycle roughly
                if breath > 0:
                    s = s + breath * rng.normal(0, 1, n).astype(np.float32) * (0.6 + 0.4 * s)
                shimmer = 1 + 0.06 * _smooth_noise(rng, n, 6.0)
                vo = rng.uniform(-0.04, 0.12)
                e = _env_adsr(n, atk * rng.uniform(0.8, 1.3), rel,
                              rel_start=int((dur_s + atk * 0.35 + vo) * SR))
                y = (s * e * shimmer * g).astype(np.float32) / np.sqrt(nv)
                pv = np.clip(center + spread * rng.uniform(-1, 1), -1, 1)
                th = (pv + 1) * np.pi / 4
                off = int(rng.uniform(0, 0.05) * SR)
                _place(srcL, y * np.cos(th) * 1.414, i0 + off)
                _place(srcR, y * np.sin(th) * 1.414, i0 + off)
        # formant filter bank (per vowel of the first note - vowels per group)
        vw = notes[0]["kw"].get("vowel", vowel)
        fs_, amps, bws = FORMANTS[vw][reg]
        shift = 1.0 + prm.get("formant_shift", 0.0)
        yL = np.zeros(total_n, np.float32)
        yR = np.zeros(total_n, np.float32)
        for f, a, bw in zip(fs_, amps, bws):
            b, aa = _biquad_bp(min(f * shift, SR * 0.45), bw * 1.3)
            gain = 10 ** (a / 20)
            yL += (signal.lfilter(b, aa, srcL) * gain).astype(np.float32)
            yR += (signal.lfilter(b, aa, srcR) * gain).astype(np.float32)
        # a touch of the raw source for body (low end), lowpassed
        bl, al = signal.butter(2, 1400 / (SR / 2))
        yL += 0.08 * signal.lfilter(bl, al, srcL).astype(np.float32)
        yR += 0.08 * signal.lfilter(bl, al, srcR).astype(np.float32)
        out[:, 0] += yL * 3.0
        out[:, 1] += yR * 3.0
    return out


# ---------------------------------------------------------------------------
# ORGAN (additive flue ranks)
# ---------------------------------------------------------------------------
def organ(part, total_n):
    rng = np.random.default_rng(part["seed"])
    prm = part["params"]
    # ranks: (footage ratio, level, n harmonics, harmonic tilt)
    reg = prm.get("reg", "full")
    if reg == "soft":
        ranks = [(1.0, 1.0, 3, 2.5), (2.0, 0.35, 2, 3.0)]
    elif reg == "pedal":
        ranks = [(0.5, 0.9, 3, 2.0), (1.0, 0.8, 5, 1.6), (2.0, 0.3, 4, 2.0)]
    else:  # principal chorus
        ranks = [(0.5, 0.45, 4, 1.8), (1.0, 1.0, 8, 1.3), (2.0, 0.6, 6, 1.5),
                 (3.0, 0.22, 3, 2.0), (4.0, 0.3, 4, 1.8), (6.0, 0.12, 2, 2.0),
                 (8.0, 0.1, 2, 2.0)]
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        f0 = hz(nt["pitch"])
        dur_s = nt["dur"] * BEAT_S
        rel = 0.25
        n = int((dur_s + rel + 0.05) * SR)
        i0 = int(round(nt["start"] * BEAT_S * SR))
        tt = _t(n)
        if part["dyn"]:
            lev = dyn_array(part["dyn"], nt["start"], n, BEAT_N)
        else:
            lev = np.full(n, nt["vel"] if nt["vel"] is not None else 0.6)
        g = 10 ** (30 * np.log10(np.maximum(lev, 0.03)) / 20)
        yl = np.zeros(n)
        yr = np.zeros(n)
        for ratio, lvl, nh, tilt in ranks:
            fr = f0 * ratio
            if fr > 9000:
                continue
            for side in (0, 1):
                det = 2 ** (rng.normal(0, 1.6) / 1200)
                ph0 = rng.uniform(0, TWOPI)
                s = np.zeros(n)
                for k in range(1, nh + 1):
                    if fr * k > 16000:
                        break
                    s += np.sin(TWOPI * fr * k * det * tt + ph0 * k) * k ** (-tilt)
                # slow wind unsteadiness
                s *= 1 + 0.01 * _smooth_noise(rng, n, 3.0)
                if side == 0:
                    yl += lvl * s
                else:
                    yr += lvl * s
        # attack: pipes speak in ~30-80 ms (lower pipes slower) + chiff
        atk = float(np.clip(0.09 - 0.012 * np.log2(f0 / 65), 0.025, 0.1))
        e = _env_adsr(n, atk, rel, rel_start=int(dur_s * SR))
        chiff = rng.normal(0, 1, n) * np.exp(-tt / 0.025) * 0.08
        bc, ac = _biquad_bp(min(f0 * 4, 8000), f0 * 2)
        chiff = signal.lfilter(bc, ac, chiff)
        yl = (yl + chiff) * e * g
        yr = (yr + chiff) * e * g
        y = np.stack([yl, yr], 1).astype(np.float32) * 0.12
        _place(out, y, i0)
    return out


# ---------------------------------------------------------------------------
# DRUMS: taiko ensemble, synthesized hits
# ---------------------------------------------------------------------------
def _drum_hit(rng, f_end, decay, vel, skin=0.5, pitch_drop=1.7, modes=(1.0, 1.59, 2.14),
              mode_amps=(1.0, 0.35, 0.18), drop_t=0.035, noise_bp=(250, 2500), length=None):
    n = int((length or decay * 5) * SR)
    tt = _t(n)
    f = f_end * (1 + (pitch_drop - 1) * np.exp(-tt / drop_t))
    y = np.zeros(n)
    for mr, ma in zip(modes, mode_amps):
        ph = TWOPI * np.cumsum(f * mr) / SR
        y += ma * np.sin(ph) * np.exp(-tt / (decay / (mr ** 0.7)))
    # skin / stick attack
    nz = rng.normal(0, 1, n) * np.exp(-tt / 0.012)
    b, a = signal.butter(2, [noise_bp[0] / (SR / 2), noise_bp[1] / (SR / 2)], btype="band")
    y += skin * signal.lfilter(b, a, nz) * 2.0
    # soft saturation for weight
    y = np.tanh(y * (0.8 + 0.8 * vel)) / np.tanh(0.8 + 0.8 * vel)
    a0 = int(0.0015 * SR)
    y[:a0] *= np.linspace(0, 1, a0)
    return y * vel


def taiko(part, total_n):
    """Taiko ensemble: kw 'drum' in {'o' (odaiko), 'n' (nagado), 's' (shime)}.
    Each hit = 2-3 players slightly apart in time/pitch/pan."""
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    spec = {
        "o": dict(f=48, decay=0.55, skin=0.35, players=2, drop=1.8, bp=(150, 1500)),
        "n": dict(f=92, decay=0.28, skin=0.55, players=3, drop=1.6, bp=(250, 2800)),
        "s": dict(f=330, decay=0.07, skin=0.9, players=2, drop=1.25, bp=(900, 6000)),
    }
    for nt in part["notes"]:
        d = spec[nt["kw"].get("drum", "n")]
        vel = nt["vel"] if nt["vel"] is not None else dyn_at(part["dyn"], nt["start"])
        i0 = int(round(nt["start"] * BEAT_S * SR))
        for pl in range(d["players"]):
            fe = d["f"] * 2 ** (rng.normal(0, 0.6) / 12) * nt["kw"].get("tune", 1.0)
            y = _drum_hit(rng, fe, d["decay"] * rng.uniform(0.9, 1.1), vel * rng.uniform(0.9, 1.0),
                          skin=d["skin"], pitch_drop=d["drop"], noise_bp=d["bp"])
            pan = np.clip(part.get("pan", 0) + rng.uniform(-0.5, 0.5), -1, 1)
            dt = 0 if (nt["sync"] and pl == 0) else int(abs(rng.normal(0, 0.004)) * SR)
            _place(out, _pan(y, pan).astype(np.float32) * 0.5, i0 + dt)
    return out


# ---------------------------------------------------------------------------
# FM glass bells / celesta
# ---------------------------------------------------------------------------
def fmbell(part, total_n):
    rng = np.random.default_rng(part["seed"])
    prm = part["params"]
    ratio = prm.get("ratio", 3.5)
    index = prm.get("index", 2.2)
    decay = prm.get("decay", 1.4)
    bright = prm.get("bright", 1.0)
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        f0 = hz(nt["pitch"])
        vel = nt["vel"] if nt["vel"] is not None else dyn_at(part["dyn"], nt["start"])
        dec = nt["kw"].get("decay", decay) * (1.0 if f0 < 800 else (800 / f0) ** 0.35)
        n = int(min(dec * 4.5, 8.0) * SR)
        tt = _t(n)
        i0 = int(round(nt["start"] * BEAT_S * SR))
        chans = []
        for side in (-1, 1):
            det = 2 ** (side * rng.uniform(2, 6) / 1200)
            fc = f0 * det
            idx_env = index * bright * (0.35 + 0.65 * np.exp(-tt / 0.12)) * (0.6 + 0.6 * vel)
            mod = np.sin(TWOPI * fc * ratio * tt) * idx_env
            car = np.sin(TWOPI * fc * tt + mod)
            # pure partial an octave up (glass) + slight inharmonic shimmer
            glass = 0.25 * np.sin(TWOPI * fc * 2.0 * tt + 0.3 * np.sin(TWOPI * fc * 5.02 * tt) * np.exp(-tt / 0.05))
            env = np.exp(-tt / dec) * (1 - np.exp(-tt / 0.0015))
            env2 = np.exp(-tt / (dec * 0.35))
            chans.append((car * env + glass * env2) * vel)
        y = np.stack(chans, 1).astype(np.float32) * 0.35
        pan = nt["pan"] if nt["pan"] is not None else part.get("pan", 0)
        if pan:
            y = y * np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)], np.float32) * 1.414
        _place(out, y, i0)
    return out


# ---------------------------------------------------------------------------
# risers, shepard tone, reverse cymbal, glissandi
# ---------------------------------------------------------------------------
def riser(part, total_n):
    """Noise riser: bandpass sweep + exponential swell, ends at note end."""
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        n = int(nt["dur"] * BEAT_S * SR)
        i0 = int(round(nt["start"] * BEAT_S * SR))
        tt = np.linspace(0, 1, n)
        f_lo, f_hi = nt["kw"].get("f0", 180.0), nt["kw"].get("f1", 9000.0)
        fc = f_lo * (f_hi / f_lo) ** (tt ** 1.3)
        ys = []
        for ch in range(2):
            x = rng.normal(0, 1, n)
            # time-varying one-pole bandpass approximation in blocks
            y = np.zeros(n)
            blk = 1024
            zi_h = None
            zi_l = None
            for s in range(0, n, blk):
                f = fc[min(s + blk // 2, n - 1)]
                bh, ah = signal.butter(2, max(20, f * 0.5) / (SR / 2), btype="high")
                bl, al = signal.butter(2, min(SR * 0.45, f * 1.8) / (SR / 2))
                if zi_h is None:
                    zi_h = signal.lfilter_zi(bh, ah) * 0
                    zi_l = signal.lfilter_zi(bl, al) * 0
                seg, zi_h = signal.lfilter(bh, ah, x[s:s + blk], zi=zi_h)
                seg, zi_l = signal.lfilter(bl, al, seg, zi=zi_l)
                y[s:s + blk] = seg
            ys.append(y)
        env = (tt ** nt["kw"].get("curve", 2.5))
        vel = nt["vel"] if nt["vel"] is not None else 0.7
        y = (np.stack(ys, 1) * env[:, None] * vel * 0.5).astype(np.float32)
        f = int(0.01 * SR)
        y[-f:] *= np.linspace(1, 0, f)[:, None]
        _place(out, y, i0)
    return out


def shepard(part, total_n):
    """Endlessly rising Shepard tone over each note's span."""
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        n = int(nt["dur"] * BEAT_S * SR)
        i0 = int(round(nt["start"] * BEAT_S * SR))
        tt = _t(n)
        rate = nt["kw"].get("oct_per_s", 0.45)
        base = nt["kw"].get("base", 55.0)
        comps = 7
        y = np.zeros((n, 2))
        for c in range(comps):
            pos = (c + rate * tt) % comps          # octave position 0..comps
            f = base * 2 ** pos
            lf = np.log2(f / (base * 2 ** (comps / 2)))
            amp = np.exp(-0.5 * (lf / 1.1) ** 2)
            ph = TWOPI * np.cumsum(f) / SR
            s = np.sin(ph) + 0.25 * np.sin(2 * ph + 0.5)
            pan = np.sin(c * 1.7)
            y[:, 0] += s * amp * np.cos((pan * 0.6 + 1) * np.pi / 4)
            y[:, 1] += s * amp * np.sin((pan * 0.6 + 1) * np.pi / 4)
        env = np.linspace(0, 1, n) ** nt["kw"].get("curve", 1.8)
        vel = nt["vel"] if nt["vel"] is not None else 0.6
        y = y * env[:, None] * vel * 0.25
        f = int(0.01 * SR)
        y[-f:] *= np.linspace(1, 0, f)[:, None]
        _place(out, y.astype(np.float32), i0)
    return out


def revcym(part, total_n):
    """Reversed crash cymbal (VSCO crash) + reversed reverb smear, peak at note end."""
    import sampler as S
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    regs = S.regions("crash")
    for nt in part["notes"]:
        r = [x for x in regs if x["lovel"] >= 71][int(rng.integers(0, 2))]
        Sd = S.resampled(r["path"], 0)
        y = Sd["y"][:int(4.5 * SR)] * Sd["norm"]
        # smear: convolve with a short noise tail for "reverse reverb"
        ir = rng.normal(0, 1, (int(1.2 * SR), 2)) * np.exp(-np.linspace(0, 7, int(1.2 * SR)))[:, None]
        ir /= np.sqrt((ir ** 2).sum(0))
        wet = np.stack([signal.fftconvolve(y[:, c], ir[:, c]) for c in range(2)], 1)
        y2 = np.zeros_like(wet)
        y2[:len(y)] = y
        y2 = 0.6 * y2 + 0.8 * wet
        y2 = y2[::-1]
        n = int(nt["dur"] * BEAT_S * SR)
        y2 = y2[-n:] if len(y2) > n else y2
        y2 = y2 * (np.linspace(0, 1, len(y2)) ** 1.5)[:, None]
        end = int(round((nt["start"] + nt["dur"]) * BEAT_S * SR))
        vel = nt["vel"] if nt["vel"] is not None else 0.7
        f = int(0.004 * SR)
        y2[-f:] *= np.linspace(1, 0, f)[:, None]
        _place(out, (y2 * vel).astype(np.float32), end - len(y2))
    return out


def gliss(part, total_n):
    """String tremolo glissando from a sampler patch with variable-rate playback.
    note.pitch = start pitch, kw['to'] = end pitch, kw['patch']."""
    import sampler as S
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        pk = nt["kw"].get("patch", "vln_trem")
        p0, p1 = nt["pitch"], nt["kw"]["to"]
        mid = 0.5 * (p0 + p1)
        layers = S.pick(pk, int(round(mid)))
        c, regs = layers[-1] if nt["kw"].get("loud", True) else layers[0]
        r = regs[0]
        Sd = S.resampled(r["path"], 0)
        src = Sd["y"][Sd["pre"]:] * Sd["norm"]
        n = int(nt["dur"] * BEAT_S * SR)
        tt = np.linspace(0, 1, n)
        curve = nt["kw"].get("curve", 1.6)
        pitch = p0 + (p1 - p0) * tt ** curve
        rate = 2 ** ((pitch - r["center"]) / 12.0)
        pos = np.cumsum(rate)
        # reflect-loop within the sustain region if we run out
        L = len(src) - 2
        a = int(0.2 * L)
        span = max(1, L - a)
        pos = np.where(pos < L, pos, a + np.abs(((pos - a) % (2 * span)) - span))
        i0f = np.floor(pos).astype(np.int64)
        u = (pos - i0f)[:, None]
        y = src[i0f] * (1 - u) + src[np.minimum(i0f + 1, len(src) - 1)] * u
        lev = dyn_array(part["dyn"], nt["start"], n, BEAT_N) if part["dyn"] else np.full(n, nt["vel"] or 0.7)
        g = 10 ** (30 * np.log10(np.maximum(lev, 0.03)) / 20)
        y = y * g[:, None]
        f = int(0.03 * SR)
        y[:f] *= np.linspace(0, 1, f)[:, None]
        y[-f:] *= np.linspace(1, 0, f)[:, None]
        _place(out, y.astype(np.float32), int(round(nt["start"] * BEAT_S * SR)))
    return out


# ---------------------------------------------------------------------------
# impacts / braam / sub
# ---------------------------------------------------------------------------
def impact(part, total_n):
    """Big hybrid hit: sub thump + boom + crack (sample layers are separate parts)."""
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        vel = nt["vel"] if nt["vel"] is not None else 1.0
        size = nt["kw"].get("size", 1.0)
        n = int(6.0 * SR)
        tt = _t(n)
        f = 38 + 55 * np.exp(-tt / 0.06)
        sub = np.sin(TWOPI * np.cumsum(f) / SR) * np.exp(-tt / (1.1 * size))
        sub = np.tanh(sub * 1.6) / np.tanh(1.6)
        boom = rng.normal(0, 1, n) * np.exp(-tt / (0.18 * size))
        b, a = signal.butter(2, 180 / (SR / 2))
        boom = signal.lfilter(b, a, boom) * 6
        crack = rng.normal(0, 1, n) * np.exp(-tt / 0.012)
        b, a = signal.butter(2, [1500 / (SR / 2), 9000 / (SR / 2)], btype="band")
        crack = signal.lfilter(b, a, crack) * 0.6
        y = (sub * 0.9 + boom * 0.5 + crack * nt["kw"].get("crack", 1.0))
        a0 = int(0.001 * SR)
        y[:a0] *= np.linspace(0, 1, a0)
        side = rng.normal(0, 1, n) * np.exp(-tt / 0.3)
        b, a = signal.butter(2, [300 / (SR / 2), 3000 / (SR / 2)], btype="band")
        side = signal.lfilter(b, a, side) * 0.15
        st = np.stack([y + side, y - side], 1) * vel * 0.5
        _place(out, st.astype(np.float32), int(round(nt["start"] * BEAT_S * SR)))
    return out


def subdrop(part, total_n):
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        n = int(nt["dur"] * BEAT_S * SR)
        tt = _t(n)
        f0, f1 = nt["kw"].get("f0", 62.0), nt["kw"].get("f1", 24.0)
        f = f1 + (f0 - f1) * np.exp(-tt / (n / SR / 3.5))
        y = np.sin(TWOPI * np.cumsum(f) / SR)
        y = np.tanh(1.3 * y) / np.tanh(1.3)          # a little harmonic content
        env = (1 - np.exp(-tt / 0.004)) * np.exp(-tt / (n / SR / 2.2))
        f = int(0.05 * SR)
        env[-f:] *= np.linspace(1, 0, f)
        vel = nt["vel"] if nt["vel"] is not None else 0.9
        y = (y * env * vel * 0.6).astype(np.float32)
        _place(out, np.stack([y, y], 1), int(round(nt["start"] * BEAT_S * SR)))
    return out


def braam(part, total_n):
    """Detuned saw stack through an opening lowpass + saturation."""
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        f0 = hz(nt["pitch"])
        dur_s = nt["dur"] * BEAT_S
        rel = nt["kw"].get("rel", 2.2)
        n = int((dur_s + rel) * SR)
        tt = _t(n)
        vel = nt["vel"] if nt["vel"] is not None else 1.0
        chans = []
        for side in range(2):
            s = np.zeros(n)
            for v in range(5):
                det = 2 ** (rng.normal(0, 9) / 1200)
                ph = (rng.uniform(0, 1) + f0 * det * tt) % 1.0
                saw = 2 * ph - 1
                s += saw
            s /= 5
            chans.append(s)
        y = np.stack(chans, 1)
        # time-varying lowpass: cutoff env
        cut = 250 + 2600 * np.exp(-tt / 0.35) * vel + 700 * vel
        yf = np.zeros_like(y)
        blk = 512
        zi = None
        for s0 in range(0, n, blk):
            c = min(cut[min(s0 + blk // 2, n - 1)], SR * 0.45)
            b, a = signal.butter(2, c / (SR / 2))
            if zi is None:
                zi = np.zeros((2, max(len(a), len(b)) - 1))
            for ch in range(2):
                yf[s0:s0 + blk, ch], zi[ch] = signal.lfilter(b, a, y[s0:s0 + blk, ch], zi=zi[ch])
        env = (1 - np.exp(-tt / 0.03))
        rs = int(dur_s * SR)
        env[rs:] *= np.exp(-(tt[rs:] - tt[rs]) / (rel / 4))
        yf = np.tanh(yf * env[:, None] * 2.2 * vel) * 0.5
        f = int(0.02 * SR)
        yf[-f:] *= np.linspace(1, 0, f)[:, None]
        _place(out, yf.astype(np.float32), int(round(nt["start"] * BEAT_S * SR)))
    return out


def sub(part, total_n):
    """Sine sub-bass with soft 2nd harmonic, follows dynamics curve."""
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        f0 = hz(nt["pitch"])
        dur_s = nt["dur"] * BEAT_S
        rel = nt["kw"].get("rel", 1.2)
        atk = nt["kw"].get("atk", 1.0)
        n = int((dur_s + rel) * SR)
        tt = _t(n)
        y = np.sin(TWOPI * f0 * tt) + 0.12 * np.sin(2 * TWOPI * f0 * tt)
        lev = dyn_array(part["dyn"], nt["start"], n, BEAT_N) if part["dyn"] else np.full(n, nt["vel"] or 0.5)
        g = 10 ** (30 * np.log10(np.maximum(lev, 0.03)) / 20)
        e = _env_adsr(n, atk, rel, rel_start=int(dur_s * SR))
        y = (y * g * e * 0.35).astype(np.float32)
        _place(out, np.stack([y, y], 1), int(round(nt["start"] * BEAT_S * SR)))
    return out


def tick(part, total_n):
    """Clock tick: dry metallic click; alternates tick/tock."""
    rng = np.random.default_rng(part["seed"])
    out = np.zeros((total_n, 2), np.float32)
    n = int(0.08 * SR)
    tt = _t(n)
    for i, nt in enumerate(sorted(part["notes"], key=lambda x: x["start"])):
        vel = nt["vel"] if nt["vel"] is not None else dyn_at(part["dyn"], nt["start"])
        f = 4200 if i % 2 == 0 else 3500
        ping = np.sin(TWOPI * f * tt) * np.exp(-tt / 0.006)
        ping += 0.5 * np.sin(TWOPI * f * 1.63 * tt) * np.exp(-tt / 0.004)
        clk = rng.normal(0, 1, n) * np.exp(-tt / 0.0012)
        b, a = signal.butter(2, 2500 / (SR / 2), btype="high")
        y = (ping * 0.6 + signal.lfilter(b, a, clk) * 0.8) * vel
        pan = 0.25 if i % 2 == 0 else -0.25
        _place(out, _pan(y, pan).astype(np.float32) * 0.5, int(round(nt["start"] * BEAT_S * SR)))
    return out


def lowbell(part, total_n):
    """Church-bell additive model: hum, prime, tierce, quint, nominal..."""
    rng = np.random.default_rng(part["seed"])
    partials = [(0.5, 1.0, 6.0), (1.0, 0.8, 3.5), (1.2, 0.55, 2.6), (1.5, 0.3, 2.0),
                (2.0, 0.65, 1.8), (2.5, 0.25, 1.2), (2.67, 0.2, 1.0), (3.0, 0.18, 0.9),
                (4.0, 0.12, 0.6), (5.33, 0.06, 0.4)]
    out = np.zeros((total_n, 2), np.float32)
    for nt in part["notes"]:
        base = hz(nt["pitch"])       # perceived strike pitch = prime partial
        vel = nt["vel"] if nt["vel"] is not None else 0.7
        n = int(7.0 * SR)
        tt = _t(n)
        y = np.zeros((n, 2))
        for r, a, d in partials:
            for ch in range(2):
                det = 2 ** (rng.normal(0, 1.5) / 1200)
                beat = 1 + 0.03 * np.sin(TWOPI * rng.uniform(0.3, 1.2) * tt)
                y[:, ch] += a * np.sin(TWOPI * base * r * det * tt + rng.uniform(0, 6)) * np.exp(-tt / d) * beat
        strike = rng.normal(0, 1, n) * np.exp(-tt / 0.006)
        b, a = signal.butter(2, [800 / (SR / 2), 6000 / (SR / 2)], btype="band")
        strike = signal.lfilter(b, a, strike) * 0.3
        y += strike[:, None]
        a0 = int(0.002 * SR)
        y[:a0] *= np.linspace(0, 1, a0)[:, None]
        _place(out, (y * vel * 0.18).astype(np.float32), int(round(nt["start"] * BEAT_S * SR)))
    return out


VOICES = dict(choir=choir, organ=organ, taiko=taiko, fmbell=fmbell, riser=riser,
              shepard=shepard, revcym=revcym, gliss=gliss, impact=impact,
              subdrop=subdrop, braam=braam, sub=sub, tick=tick, lowbell=lowbell)


def render_part(part, total_n):
    return VOICES[part["inst"]](part, total_n)
