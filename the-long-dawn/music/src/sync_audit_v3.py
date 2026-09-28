"""SOUND-C sync audit: where the PICTURE's events actually happen, measured from the frames themselves.

  python sync_audit_v3.py extract C          # the latest delivery master -> cache/sync/C_192x80.rgb (niced, 2 threads)
  python sync_audit_v3.py signals C          # per-frame signals -> cache/sync/C_sig.npz
  python sync_audit_v3.py show C name:a:b[:sig,...] ...   # print signals over [a, b)
  python sync_audit_v3.py sheet C out.jpg f1,f2,...       # a labelled contact sheet of master frames (small)

Signals per frame (192x80 area-scaled): L mean luma; D mean |luma diff| to the previous frame (x10 in `show`);
Dk mean darkening (ink arriving); Wn % warm pixels (fire/gold); Br % pixels over luma 200.
"""
import os, sys, subprocess, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MUSIC))
CACHE = os.path.join(MUSIC, "cache", "sync")
DELIV = os.path.join(ROOT, "_local_logs", "delivery")
FF = os.path.expanduser("~/.venvs/longdawn/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
W, H = 192, 80
NF = {"A": 6480, "B": 5440, "C": 7200}


def master(cut):
    c = sorted(glob.glob(os.path.join(DELIV, f"{cut}_master*.mp4")), key=os.path.getmtime)
    return c[-1]


def raw(cut):
    return os.path.join(CACHE, f"{cut}_{W}x{H}.rgb")


def extract(cut):
    os.makedirs(CACHE, exist_ok=True)
    src = master(cut)
    print("master:", src)
    subprocess.run(["nice", "-n", "10", FF, "-v", "error", "-threads", "2", "-i", src, "-an", "-vf",
                    f"scale={W}:{H}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-y", raw(cut)], check=True)


def frames(cut):
    n = os.path.getsize(raw(cut)) // (W * H * 3)
    return np.memmap(raw(cut), dtype=np.uint8, mode="r", shape=(n, H, W, 3))


def luma(f):
    f = f.astype(np.float32)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def signals(cut):
    v = frames(cut)
    n = len(v)
    out = {k: np.zeros(n, np.float32) for k in ("L", "D", "Dk", "Wn", "Br")}
    prev = None
    for i in range(n):
        f = v[i].astype(np.float32)
        y = luma(f)
        warm = (f[..., 0] > 140) & (f[..., 0] > 1.35 * f[..., 2] + 10) & (y > 70)
        out["L"][i], out["Wn"][i], out["Br"][i] = y.mean(), warm.mean() * 100, (y > 200).mean() * 100
        if prev is not None:
            d = y - prev
            out["D"][i], out["Dk"][i] = np.abs(d).mean(), np.clip(-d, 0, None).mean()
        prev = y
    np.savez(os.path.join(CACHE, f"{cut}_sig.npz"), **out)


SCALE = {"L": 1, "D": 10, "Dk": 10, "Wn": 10, "Br": 10}


def show(cut, specs):
    s = np.load(os.path.join(CACHE, f"{cut}_sig.npz"))
    for arg in specs:
        name, a, b, *sg = arg.split(":")
        a, b = int(a), int(b)
        print(f"== {name} [{a},{b})  col k = frame {a}+k")
        for k in (sg[0].split(",") if sg else ("L", "D", "Wn")):
            print(f"{k:3s}", " ".join(str(int(round(x))) for x in s[k][a:b] * SCALE[k]))


def sheet(cut, out, fl, cols=6, tw=320, crop=None):
    """a labelled contact sheet of master frames (decoded one at a time by seeking)"""
    from PIL import Image, ImageDraw
    src = master(cut)
    cw, ch = (crop[2], crop[3]) if crop else (1920, 804)
    th = int(tw * ch / cw)
    fl = [int(x) for x in fl]
    vf = (f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else "") + f"scale={tw}:{th}"
    rows = (len(fl) + cols - 1) // cols
    S = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    for k, f in enumerate(fl):
        p = subprocess.run([FF, "-v", "error", "-threads", "2", "-ss", f"{f / 24:.4f}", "-i", src, "-frames:v", "1",
                            "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                           capture_output=True, check=True).stdout
        im = Image.frombytes("RGB", (tw, th), p[: tw * th * 3])
        ImageDraw.Draw(im).text((4, 2), str(f), fill=(255, 255, 0))
        S.paste(im, ((k % cols) * tw, (k // cols) * th))
    S.save(out, quality=80)


if __name__ == "__main__":
    cmd, cut = sys.argv[1], sys.argv[2].upper()
    if cmd == "extract":
        extract(cut)
    elif cmd == "signals":
        signals(cut)
    elif cmd == "show":
        show(cut, sys.argv[3:])
    elif cmd == "sheet":
        sheet(cut, sys.argv[3], sys.argv[4].split(","))


def ink(cut, a, b, w=960, thr=18.0):
    """fine ink/pen measurement over [a, b): per frame, the number of pixels that DARKENED by > thr luma vs the
    previous frame (ink arriving), and that BRIGHTENED (glow, burn); decoded at w px wide, grayscale"""
    h = int(round(w * 804 / 1920 / 2)) * 2
    src = master(cut)
    p = subprocess.run(["nice", "-n", "10", FF, "-v", "error", "-threads", "2", "-ss", f"{(a - 1) / 24:.4f}", "-i", src,
                        "-frames:v", str(b - a + 1), "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", "gray", "-f",
                        "rawvideo", "-"], capture_output=True, check=True).stdout
    v = np.frombuffer(p, np.uint8).reshape(-1, h, w).astype(np.int16)
    d = v[1:] - v[:-1]
    dk = (d < -thr).sum((1, 2))
    br = (d > thr).sum((1, 2))
    return dk, br


if __name__ == "__main__" and sys.argv[1] == "ink":
    for arg in sys.argv[3:]:
        name, a, b = arg.split(":")
        dk, br = ink(sys.argv[2].upper(), int(a), int(b))
        print(f"== {name} [{a},{b}) ink-dark / brighten pixel counts (960 px wide), col k = frame {a}+k")
        print("dk", " ".join(str(int(x)) for x in dk))
        print("br", " ".join(str(int(x)) for x in br))
