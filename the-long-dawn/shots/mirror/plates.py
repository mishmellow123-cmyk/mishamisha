"""C10 THE MIRROR: bake the two vision plates the water shows into small committed files, so farm nodes
(which only get the git tip, never renders/) can render the shot.

    python3 shots/mirror/plates.py            # -> shots/mirror/plates/fire.mp4 + dawn.jpg (+ a preview sheet)

FIRE ("the lands burning"): EMBERS-C's C7 THE RACE (C 1560-1679: the surges, the gold rain, the walls of red).
  Source = renders/embers_C3 when it holds all of 1560-1679 (the H5 re-dress), else the pre-H5
  renders/embers_C3_half. The Ring hangs at the top centre of every race frame and must NOT be in the water, so
  the plate is the frame's two flanks (left and right of the Ring) joined with a soft seam: towers, sparks and the
  red walls, no Ring. Stored at the half-res height (402) as H.264 (the water blurs it anyway).
DAWN ("a golden dawn"): RUN-C's C24 THE ILLUMINATION (the drawn world at sunrise, gold wash laid on the ink),
  so the Mirror's glimpse is the very image the film pays off at C24. Source = renders/runC_illum/f_02480 when the
  final exists, else RUN-C's farm test of the same frame.
Re-run this after EMBERS-C's H5 race or RUN-C's final illumination lands, check the sheet, commit the plates.
"""
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import look  # noqa: E402

OUT = os.path.join(HERE, 'plates')
FIRE_SRC = (1560, 1680)                      # C7 frames used (the surge half of the race)
FIRE_H = 402                                 # stored height
# the flanks, as fractions of the source width: left of the Ring and right of it (pre-H5 race: Ring at x 0.39-0.61)
FLANK_L = (0.0, 0.385)
FLANK_R = (0.615, 1.0)
SEAM = 0.05                                  # cross-fade width at the join, fraction of the source width
DAWN_FRAME = 2480                            # RUN-C illum local frame: the sun just up, the gold wash laid
FFMPEG = os.path.expanduser('~/.venvs/longdawn/bin/ffmpeg')


def fire_source():
    full = os.path.join(ROOT, 'renders', 'embers_C3')
    if all(os.path.exists(look.find_frame(full, f)) for f in range(*FIRE_SRC)):
        return full
    return os.path.join(ROOT, 'renders', 'embers_C3_half')


def dawn_source():
    for p in (look.find_frame(os.path.join(ROOT, 'renders', 'runC_illum'), DAWN_FRAME),
              os.path.join(ROOT, 'renders', '_farmtest', 'runC_r1', 'runC_illum_d', f'f_{DAWN_FRAME:05d}.jpg')):
        if os.path.exists(p):
            return p
    raise SystemExit('no dawn source frame')


def flanks(img):
    """Join the frame's left and right flanks (the Ring between them is dropped) with a soft seam."""
    h, w = img.shape[:2]
    if h != FIRE_H:
        img = cv2.resize(img, (int(round(w * FIRE_H / h)), FIRE_H), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    a = img[:, int(FLANK_L[0] * w):int(FLANK_L[1] * w)].astype(np.float32)
    b = img[:, int(FLANK_R[0] * w):int(FLANK_R[1] * w)].astype(np.float32)
    s = int(SEAM * w)
    ramp = np.linspace(0, 1, s, dtype=np.float32)[None, :, None]
    ramp = ramp * ramp * (3 - 2 * ramp)
    mid = a[:, -s:] * (1 - ramp) + b[:, :s] * ramp
    out = np.concatenate([a[:, :-s], mid, b[:, s:]], axis=1)
    wo = out.shape[1] // 2 * 2
    return np.clip(out[:, :wo], 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    src = fire_source()
    frames = [flanks(cv2.imread(look.find_frame(src, f))) for f in range(*FIRE_SRC)]
    h, w = frames[0].shape[:2]
    mp4 = os.path.join(OUT, 'fire.mp4')
    p = subprocess.Popen([FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
                          '-r', '24', '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '15',
                          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4], stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    cap = cv2.VideoCapture(mp4)
    n = 0
    while cap.read()[0]:
        n += 1
    print(f'fire.mp4 {w}x{h} {n} frames from {os.path.relpath(src, ROOT)} ({os.path.getsize(mp4) / 1e6:.1f} MB)')
    assert n == len(frames), 'fire plate does not decode to every frame'
    d = dawn_source()
    dawn = cv2.imread(d)
    cv2.imwrite(os.path.join(OUT, 'dawn.jpg'), dawn, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f'dawn.jpg {dawn.shape[1]}x{dawn.shape[0]} from {os.path.relpath(d, ROOT)}')
    sheet = [cv2.resize(frames[i], (w // 2, h // 2)) for i in (0, 40, 80, 119)]
    cv2.imwrite(os.path.join(OUT, '..', 'plates_preview.jpg'),
                np.vstack([np.hstack(sheet[:2]), np.hstack(sheet[2:])]))


if __name__ == '__main__':
    main()
