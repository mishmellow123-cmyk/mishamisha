"""THE LONG DAWN C5 - synchronized listening excerpts for the unheard decisions of score pass 2 (and its effects).

For the morning after a render exists; nothing here has been listened to, and no audio existed when it was written.

    python listening_excerpts_c5.py --list                                   # the windows and their open questions
    python listening_excerpts_c5.py OUTDIR ../out/v3/final_C5P2.wav ../out/v3/final_C5.wav    # A/B, one per file

For every window and every audio file: OUTDIR/<window>_<f0>-<f1>__<audio>.mov and a .txt with the window's open
questions and the audio's sha256. Picture: the DELIVERED shot frames (measure_c5_events.SHOTS; $LD_FRAMES, else
~/ldfarm/out), scaled to 960x402, hard cuts between shots (NOT the edit's transitions); a frame with no delivered
shot is a conspicuous slate naming what is missing. Audio: the file's own samples [f0 * 2000, (f1 + 1) * 2000) as
24-bit PCM, so the first sample belongs to the first frame (no codec priming, no offset). Only a 5,920-frame file
is accepted: the retired 7,200-frame sound_C.wav is refused by its length. Excerpts carry film frames: keep them
out of the public repository.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import measure_c5_events as M

FPS, SR = 24, 48000
FR = SR // FPS
FRAMES = 5920
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
MISSING = [(2640, 2880, "C12 FLINT (EDIT's decision; not on this Mac)"), (3120, 3440, "C14 THE BEACON RUN (runC_scroll)"),
           (4720, 5200, "C20 THE ILLUMINATION"), (5200, 5440, "C21 PLENTY"), (5680, 5920, "C23 THE TITLE"),
           (0, 2080, "C1-C9, the first half")]


def barmap_frames(cut="C5P2"):
    import json
    with open(os.path.join(M.V3, f"barmap_{cut}.json")) as fh:
        return {e["id"]: e["f"] for e in json.load(fh)["sync"]}


def windows(ev=None):
    """[(id, f0, f1 inclusive, [questions])]; every frame in a question comes from the pass-2 bar map"""
    e = ev or barmap_frames()
    low, back, ahead, reach, rev = e["low_fire"], e["flares_back"], e["pulls_ahead"], e["leaders_reach"], e["reveal"]
    b3, b4, b5, b6 = (e[f"map_beacon_{k}"] for k in (3, 4, 5, 6))
    dark, last, lit, cold, ring, storm = (e["one_dark"], e["last_beacon"], e["all_lit"], e["forges_cold"],
                                          e["ring_unfinished"], e["storm_gone"])
    darken = dark - 80                         # score_v3_C5P2.last_beacon: the darkening starts 4 beats before
    return [
        ("refusal", 2080, 2319, [
            "2089-2150 and 2156-2249 (effects): two quill passages follow the drawing. On the ink, or a separate event?",
            "the music is pass 1's, unchanged: its cues sit on the offering hand (2120; drawn 2114-2135) and the raised "
            "hand (2200; drawn 2195-2202). Should the old story's cue move to the Ring on the palm (2138)?"]),
        ("trap", 2400, 2659, [
            f"{low}: one forge sinks; a lone horn tries the refusal (A4). One voice trying to stop alone?",
            f"{back}-{ahead}: the returning horn call, pass 1's (1, 1, 2) squeezed into {ahead - back} frames (0.525 of "
            "its written length). Hurried, or urgent?",
            f"{ahead}-{reach}: the violins climb D Eb F Ab and land on A5 at {reach}, as both front-runners reach the "
            "Ring. Does the landing meet the picture?",
            "no hammer is heard: the library has no hammer or anvil recording (sound table rows C5.hammer.*)",
            "2640-2659 is a slate: FLINT is not on this Mac and EDIT has not chosen its frames"]),
        ("first_fires", 2860, 3059, [
            f"{rev}: two calls at once, near (hn3: D4 A4 D5) and far (hn_far: a fifth lower, the rival's fire, right). "
            "Two fires answering each other, or one chord?",
            f"{rev + 20}: the far call's E4 over the D-major pad (an added ninth) and the parallel fifths. Open, or a "
            "clash?",
            f"{rev + 80} and {rev + 160}: pass 1's echoes off the ranges follow the pair. Too many calls now?"]),
        ("map", 3440, 3859, [
            f"{b3} and {b4}: the far horns answer beacons 3 and 4 as they catch. On the flame?",
            f"{darken} and {darken + 40}: the harmony darkens (Gm7, then Eb maj7) while beacons 5 ({b5}) and 6 ({b6}) "
            f"are still catching. Should F major hold until the pause at {dark}?",
            f"{dark}-{lit - 1}: A7sus4 held through the dark kingdom's pause. Does the hold carry the wait?",
            f"{lit}: D when the holdout's flame is full; its first flame shows at {last}. Late by {lit - last} frames, "
            "or right?",
            f"{cold}: the swell stops dead (every note released by the cut; no residue under the breath). Clean, or a "
            "click?"]),
        ("silence_ring", 3820, 4199, [
            f"{cold}-{ring - 1} must be silent: read verify_c5_render.py's numbers before judging by ear",
            f"{ring}: the Ring's strings start on the cut, unanticipated, so their bows bloom up to 12 frames after it. "
            "Late, or right for a suspended Ring?",
            f"{ring}-{storm}: the storm bed at a -9 dB design trim, thinning with the measured storm; the cellos' Ab "
            f"drains to {storm} plus half a beat. Balance?"]),
    ]


def frame_source(root, f):
    """the delivered JPEG showing C5 frame f, or None"""
    for stem, a, b, _ in M.SHOTS.values():
        if a <= f <= b:
            p = M.fpath(root, stem, f)
            return p if os.path.exists(p) else None
    return None


def missing_label(f):
    for a, b, what in MISSING:
        if a <= f < b:
            return what
    return "no delivered frame"


def slate(path, f):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (1920, 804), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((24, 24, 1896, 780), outline=(255, 0, 0), width=8)
    font = ImageFont.load_default(size=56)
    for k, line in enumerate(("MISSING PICTURE", missing_label(f), f"C5 frame {f}")):
        d.text((80, 220 + 110 * k), line, fill=(255, 255, 255), font=font)
    im.save(path, quality=90)


def excerpt(root, audio, wid, f0, f1, questions, outdir, frames_total=FRAMES):
    import audio_guard_v3 as AG
    import soundfile as sf
    AG.check_wav(audio, frames_total)
    name = f"{wid}_{f0:05d}-{f1:05d}__{os.path.splitext(os.path.basename(audio))[0]}"
    slates = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, f in enumerate(range(f0, f1 + 1)):
            dst = os.path.join(tmp, f"seq_{k:05d}.jpg")
            src = frame_source(root, f)
            if src:
                os.symlink(src, dst)
            else:
                slate(dst, f)
                slates.append(f)
        y, sr = sf.read(audio, start=f0 * FR, stop=(f1 + 1) * FR, dtype="float32", always_2d=True)
        assert sr == SR and len(y) == (f1 + 1 - f0) * FR, (sr, len(y))
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, y, SR, subtype="PCM_24")
        out = os.path.join(outdir, name + ".mov")
        r = subprocess.run([FFMPEG, "-y", "-v", "error", "-threads", "2", "-framerate", str(FPS), "-i",
                            os.path.join(tmp, "seq_%05d.jpg"), "-i", wav, "-map", "0:v", "-map", "1:a", "-vf",
                            "scale=960:402", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt",
                            "yuv420p", "-c:a", "pcm_s24le", out], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"ffmpeg failed for {name}: {r.stderr[-400:]}")
    h = hashlib.sha256(open(audio, "rb").read()).hexdigest()
    with open(os.path.join(outdir, name + ".txt"), "w") as fh:
        fh.write(f"{name}\nC5 frames {f0}-{f1} ({(f1 - f0 + 1) / FPS:.2f} s); audio {os.path.basename(audio)} "
                 f"sha256 {h}\npicture: delivered shot frames, hard cuts (not the edit)"
                 + (f"; SLATES (missing picture) on {len(slates)} frames {slates[0]}-{slates[-1]}" if slates else "")
                 + "\nUNHEARD when written. Questions:\n" + "".join(f"- {q}\n" for q in questions))
    return out, slates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("audio", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--frames-root", default=None)
    a = ap.parse_args()
    ws = [w for w in windows() if not a.only or w[0] in a.only.split(",")]
    if a.list or not a.outdir:
        for wid, f0, f1, qs in ws:
            print(f"{wid} {f0}-{f1}")
            for q in qs:
                print(f"  - {q}")
        return 0
    if not a.audio:
        ap.error("give at least one rendered audio file")
    os.makedirs(a.outdir, exist_ok=True)
    root = M.frames_root(a.frames_root)
    for audio in a.audio:
        for wid, f0, f1, qs in ws:
            out, sl = excerpt(root, audio, wid, f0, f1, qs, a.outdir)
            print(f"{out}" + (f"  ({len(sl)} slate frames)" if sl else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
