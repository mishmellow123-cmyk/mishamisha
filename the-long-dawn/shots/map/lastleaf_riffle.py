"""One opt-in opening-leaf study, using C2's existing camera and hearth.

Current C2 has no page turn; its riffle is in C3. This proposes one new forward
turn at C150 (6.25 s in the film), with the last written leaf's open Ring on its
back and the next recto blank. It renders one shutter instant, not animation.
The still cannot establish turn timing or motion quality.

Example: python shots/map/lastleaf_riffle.py --out /tmp/opening-leaf.png
"""
import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np

import book as B
import book_c as BC
import ringpage


ROOT = Path(__file__).resolve().parents[2]
FRAME = 150
PHASE = .80
CENTER = (10.2, 7.2)
RADIUS = 3.3


def validate(frame, phase, scale):
    if isinstance(frame, bool) or not isinstance(frame, int) or not 80 <= frame < 320:
        raise ValueError('The opening study uses an absolute C2 frame, 80–319')
    if not math.isfinite(phase) or not 0 < phase < 1:
        raise ValueError('A turning leaf requires a phase strictly between 0 and 1')
    if scale not in (.5, 1.):
        raise ValueError('The still supports half or full delivery resolution')


def setup(frame=FRAME, phase=PHASE, scale=.5):
    """Retain the C2 camera, page blocks, and hearth at the requested instant."""
    validate(frame, phase, scale)
    scene = BC.Book3(int(1920 * scale), int(804 * scale))
    t = (frame - 80) / 24.
    bk = scene.book(2.6, 1.4)
    cam = scene.cam_red_book(bk, t)
    up = scene.hearth_up(t)
    light = scene.light((-55, 55, 38), 2.2 * up, t)
    return scene, bk, cam, light, t, up


def render_still(frame=FRAME, phase=PHASE, scale=.5):
    scene, bk, cam, light, t, up = setup(frame, phase, scale)
    # At phi=0 this leaf lies on the right: text faces up. As it turns left,
    # its back becomes the Ring-bearing verso; the following recto is blank.
    # The older verso beneath it is also written. Avoid C2's 140 ppc pair of
    # full-page textures: this study needs one 70 ppc drawing and 45 ppc text.
    written = scene.tex_text(41, ppc=45)
    last = ringpage.RingLeaf(ppc=70, center=CENTER, radius=RADIUS).texture()
    blank = B.blank_tex(20., 29.)
    bounce = 2200. * BC.ramp(t, 5., 9.)
    lights = np.array([[70., -35., 30., bounce, bounce * .62, bounce * .36]])
    return scene.finish_layer(
        bk, cam, light, written, blank, t, xl=lights,
        leaf=(phase, written, last), lk=up,
    )


def output_path(value):
    path = Path(value).expanduser()
    if path.suffix.lower() != '.png':
        raise ValueError('The study output must be a PNG file')
    if path.resolve().is_relative_to((ROOT / 'renders').resolve()):
        raise ValueError('The shared renders directory is read-only; choose a study output path')
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=output_path)
    ap.add_argument('--frame', type=int, default=FRAME)
    ap.add_argument('--phase', type=float, default=PHASE)
    ap.add_argument('--scale', type=float, choices=(.5, 1.), default=.5)
    args = ap.parse_args(argv)
    try:
        validate(args.frame, args.phase, args.scale)
    except ValueError as exc:
        ap.error(str(exc))
    start = time.perf_counter()
    hdr, _ = render_still(args.frame, args.phase, args.scale)
    rgb = BC.look.finish(hdr, exposure=1.15, bloom_strength=.06,
                         bloom_threshold=1.2, vignette_amount=.32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    BC.look.save_png(args.out, rgb)
    print(json.dumps(dict(
        frame=args.frame, film_seconds=args.frame / 24., turn_phase=args.phase,
        proposed_new_C2_turn=True, motion_verified=False,
        seconds=time.perf_counter() - start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == 'darwin' else 1024),
    )), flush=True)


if __name__ == '__main__':
    main()
