"""THE LONG DAWN v3 - SOUND lane: ElevenLabs text-to-sound-effects, for the gaps the recordings leave.

The key stays in its file (~/.config/longdawn-elevenlabs/api_key) and in a request header: never printed, logged or
committed.  Every generation is logged (prompt, seconds, influence, format, credits) to music/sound/eleven_log.json,
and music/SFX_CREDITS.md is regenerated from the logs by sound_credits_v3.py.  No voices, no music.

    python eleven_v3.py gen <tag> <seconds> "<prompt>" [--takes 2] [--influence 0.5] [--loop]
    python eleven_v3.py credits
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
EL_DIR = os.path.join(MUSIC, "cache", "sound", "el")
LOG = os.path.join(MUSIC, "sound", "eleven_log.json")
KEY = os.path.expanduser("~/.config/longdawn-elevenlabs/api_key")
API = "https://api.elevenlabs.io/v1"
BUDGET = 30000
# best first: 48 kHz PCM needs a higher tier on some plans, so fall back in order
FORMATS = ["pcm_48000", "opus_48000_192", "opus_48000_128", "mp3_44100_128"]
os.makedirs(EL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def _key():
    return open(KEY).read().strip()


def _req(path, data=None, method="GET", timeout=180):
    h = {"xi-api-key": _key(), "User-Agent": "longdawn-sound/1.0"}
    if data is not None:
        h["Content-Type"] = "application/json"
        data = json.dumps(data).encode()
    req = urllib.request.Request(API + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        try:
            body = e.read()[:400].decode("utf-8", "replace")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {e.code} on {path.split('?')[0]}: {body}") from None


def subscription():
    d, _ = _req("/user/subscription")
    return json.loads(d)


def used():
    s = subscription()
    return s.get("character_count"), s.get("character_limit"), s.get("tier")


def log():
    try:
        return json.load(open(LOG))
    except Exception:
        return []


def spent():
    return sum(x.get("credits") or 0 for x in log())


def gen(tag, seconds, prompt, influence=0.5, loop=False, fmt=None):
    """one generation -> path of the decoded 48 kHz WAV (float), logged with its credit cost"""
    if spent() >= BUDGET:
        raise RuntimeError(f"ElevenLabs budget reached ({spent()} >= {BUDGET})")
    before = used()[0]
    body = {"text": prompt, "duration_seconds": float(seconds), "prompt_influence": float(influence),
            "model_id": "eleven_text_to_sound_v2"}
    if loop:
        body["loop"] = True
    last = None
    for f in ([fmt] if fmt else FORMATS):
        try:
            audio, ctype = _req(f"/sound-generation?output_format={f}", body, "POST")
            fmt = f
            break
        except RuntimeError as e:
            last = e
            if "HTTP 4" in str(e) and ("tier" in str(e).lower() or "format" in str(e).lower() or "403" in str(e)
                                      or "422" in str(e) or "400" in str(e)):
                continue
            raise
    else:
        raise last
    n = len([x for x in log() if x["tag"] == tag]) + 1
    base = os.path.join(EL_DIR, f"{tag}_{n:02d}")
    if fmt.startswith("pcm_"):
        import numpy as np
        import soundfile as sf
        sr = int(fmt.split("_")[1])
        y = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
        if abs(len(y) - 2 * seconds * sr) < abs(len(y) - seconds * sr):     # interleaved stereo
            y = y[: len(y) // 2 * 2].reshape(-1, 2)
        path = base + ".wav"
        sf.write(path, y, sr, subtype="PCM_16")
    else:
        ext = "ogg" if fmt.startswith("opus") else "mp3"
        raw = base + "." + ext
        open(raw, "wb").write(audio)
        path = base + ".wav"
        import subprocess
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-ar", "48000", "-c:a", "pcm_f32le", path], check=True)
        os.remove(raw)
    time.sleep(1.0)
    after = used()[0]
    credits = (after - before) if (after is not None and before is not None) else None
    entry = dict(tag=tag, take=n, file=os.path.relpath(path, MUSIC), prompt=prompt, seconds=float(seconds),
                 influence=float(influence), loop=bool(loop), format=fmt, credits=credits,
                 when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    lg = log()
    lg.append(entry)
    json.dump(lg, open(LOG, "w"), indent=1)
    return path, entry


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("tag", nargs="?")
    ap.add_argument("seconds", nargs="?", type=float)
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--influence", type=float, default=0.5)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--fmt", default=None)
    a = ap.parse_args()
    if a.cmd == "credits":
        c, lim, tier = used()
        print(f"account: {c} / {lim} credits used (tier {tier});  this lane's log: {spent()} credits in "
              f"{len(log())} generations (budget {BUDGET})")
    elif a.cmd == "gen":
        for k in range(a.takes):
            p, e = gen(a.tag, a.seconds, a.prompt, a.influence, a.loop, a.fmt)
            print(f"{e['tag']} take {e['take']}: {e['seconds']} s, {e['format']}, {e['credits']} credits -> {e['file']}")
