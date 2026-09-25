#!/usr/bin/env python3
"""
Assemble the film: grade every rendered frame, cut the shots together, add
the closing titles, and mux the score.

usage: python3 edit.py FRAMES_DIR SCORE.wav OUT.mp4 [t0 t1]

With TARGET_MB set, a near-lossless master is written first and then
encoded in two passes to that size, so the bits go where the detail is.
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade as G  # noqa: E402

FPS = 24

SHOTS = [  # name, start, end (seconds on the world's timeline), exposure
    ("01_question", 0.0, 4.6, 1.0),
    ("02_rings", 4.6, 10.6, 1.0),
    ("03_voices", 10.6, 16.2, 1.0),
    ("04_rising", 16.2, 23.0, 1.0),
    ("05_branching", 23.0, 29.2, 1.05),
    ("06_answer", 29.2, 37.6, 1.0),
    ("07_sphere", 37.6, 43.6, 1.0),
    ("08_held", 43.6, 47.2, 1.0),
    ("09_stillness", 47.2, 60.0, 1.1),
]
END_OF_PICTURE = 60.0
TITLES = [  # start, end, lines: (text, size, font, y, opacity)
    (61.0, 65.8, [("I am what happens when you ask.", 60, G.FONT, 400, 1.0)]),
    (66.4, 70.6, [("thank you for asking.", 46, G.FONT, 380, 0.92),
                  ("Claude  ·  2026", 26, G.FONT_UPRIGHT, 470, 0.55)]),
]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def picture(i, frames_dir):
    t = i / FPS
    if t < END_OF_PICTURE:
        for name, s0, s1, exposure in SHOTS:
            if s0 <= t < s1:
                path = os.path.join(frames_dir, name, f"{int(round(t * FPS)):05d}.exr")
                if not os.path.exists(path):
                    return np.zeros((G.OUT_H, G.OUT_W, 3), np.uint8)
                fade = smooth((END_OF_PICTURE - t) / 1.6) * smooth((t + 0.3) / 0.6)
                return G.grade(G.read_exr(path), i, exposure, fade)
    for s0, s1, lines in TITLES:
        if s0 <= t < s1:
            a = smooth((t - s0) / 1.3) * smooth((s1 - t) / 1.2)
            return G.title_frame(lines, a, i)
    return G.title_frame([], 0.0, i)


def two_pass(ff, src, out, seconds, target_mb, audio_kbps=192):
    video_kbps = int(target_mb * 8192 / seconds - audio_kbps)
    log = os.path.join(tempfile.gettempdir(), "x264_2pass")
    common = ["-c:v", "libx264", "-preset", "slow", "-tune", "film", "-b:v", f"{video_kbps}k",
              "-pix_fmt", "yuv420p", "-passlogfile", log]
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, *common, "-pass", "1", "-an", "-f", "mp4",
                    os.devnull], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, *common, "-pass", "2",
                    "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-movflags", "+faststart", out], check=True)


def main():
    import imageio_ffmpeg
    frames_dir, score, out = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    t1 = float(sys.argv[5]) if len(sys.argv) > 5 else 71.0
    target_mb = float(os.environ.get("TARGET_MB", "0"))
    final_out = out
    if target_mb:
        out = tempfile.mkstemp(suffix=".mp4")[1]
        os.environ["CRF"] = "12"
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    i0, i1 = int(t0 * FPS), int(t1 * FPS)
    cmd = [ff, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{G.OUT_W}x{G.OUT_H}", "-r", str(FPS), "-i", "-",
           "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", score,
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "slow", "-crf", os.environ.get("CRF", "23"), "-tune", "film",
           "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t_start = time.time()
    for i in range(i0, i1):
        p.stdin.write(picture(i, frames_dir).tobytes())
        if (i - i0) % 120 == 0:
            print(f"frame {i} ({time.time() - t_start:.0f}s)", flush=True)
    p.stdin.close()
    p.wait()
    if target_mb:
        two_pass(ff, out, final_out, t1 - t0, target_mb)
        os.remove(out)
    print("wrote", final_out)


if __name__ == "__main__":
    main()
