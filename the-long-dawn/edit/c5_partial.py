"""C5 PARTIAL REVIEW CUT: a conspicuously labelled, silent assembly of THE LAST PAGES C5 from whatever this machine
holds. Every delivered frame goes through the real assembler (assemble.Ctx: takes, mattes, EDIT transitions, the C5
captions); every frame whose source is missing becomes a large slate, "MISSING ON THIS MAC: <shot> C<a>-<b>". No
finish, no sound, no delivery QC: this is review evidence, never a master.

    python3 edit/c5_partial.py --out DIR                  # DIR/C5_PARTIAL_review_960x402.mp4 + manifest + storyboard
    python3 edit/c5_partial.py --out DIR --frames 2319,2320   # just those frames as PNGs (checks and joins)
    python3 edit/c5_partial.py --out DIR --no-video --storyboard

The readiness gate runs first in --partial mode; a FAIL (the structure is wrong) stops the build, GAPs are what the
slates stand for. Captions set in_picture (baked into a render) are drawn by EDIT only over their shot's slate, as a
stand-in for pixels this machine lacks; C's iron-gall ink lines go parchment-white on a slate (there is no paper).
The storyboard (one frame mid-caption per line, one per slate, in order) carries no explanation for a blind reader:
its README lives beside it, not in it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

EDIT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EDIT)
sys.path.insert(0, EDIT)
import assemble as AS  # noqa: E402
import c5_readiness as RD  # noqa: E402

EDL = AS.EDL
titles = AS.titles
CUT, FPS = 'C', 24
FFMPEG = os.environ.get('FFMPEG') or shutil.which('ffmpeg') or 'ffmpeg'
SLATE_BG = (16, 16, 21)
LABEL = 'C5 PARTIAL REVIEW CUT  ·  NOT FINAL'
SB_LABEL = 'PARTIAL  ·  NOT FINAL'


def section_names():
    return {s['id']: s['name'] for s in RD.load_barmap()['sections']}


def slate_spans(runs):
    """[(a, b, section, label)]: consecutive missing or undecided frames, merged within their bar-map section."""
    names, out = section_names(), []
    for a, b, sec, _, st in runs:
        if st not in ('missing', 'decision'):
            continue
        if out and out[-1][1] == a and out[-1][2] == sec:
            out[-1][1] = b
        else:
            out.append([a, b, sec, names.get(sec, sec)])
    return [tuple(x) for x in out]


class PartialCtx(AS.Ctx):
    """assemble.Ctx for C5 with missing frames slated conspicuously and the partial label burned in."""

    def __init__(self, runs, scale=0.5, storyboard=False):
        super().__init__(CUT, None, scale, True)
        self.storyboard = storyboard
        self.spans = slate_spans(runs)
        self.missing = {}
        for a, b, sec, label in self.spans:
            for f in range(a, b):
                self.missing[f] = (a, b, sec, label)
        rows = titles.text_table(CUT)
        # in_picture lines over a slate: EDIT draws a stand-in (R02 as ink on its page, the title as C's title)
        self.standins = []
        for r in rows:
            if r['set'] == 'in_picture' and any(f in self.missing for f in range(r['f_in'], r['f_out'])):
                self.standins.append(titles.TextV3(CUT, dict(r, set='title' if r['id'] == 'title' else 'ink'), scale))
        self.caption_bands = [(r['f_in'], r['f_out'], r.get('y', 402 if r['id'] == 'title' else titles.Y_LOWER))
                              for r in rows]
        self._slate_cache = {}
        self._label = self._make_label(SB_LABEL if storyboard else LABEL)
        if EDL.TRANS.get(CUT):
            self.picture = AS._transitions(self, None)           # the real EDIT transitions, exactly as a build

    # ------------------------------------------------------------------------------------------------ slates
    def _slate_y(self, a, b):
        """The slate text's centre row (1920x804 units), clear of every caption shown over this span."""
        ys = [y for f0, f1, y in self.caption_bands if f0 < b and a < f1]
        for cy in (322, 200, 470):
            if all(abs(cy - y) > 150 for y in ys):
                return cy
        return 322

    def missing_slate(self, f):
        a, b, sec, label = self.missing[f]
        key = (a, b)
        if key not in self._slate_cache:
            W, H, s = self.W, self.H, self.W / 1920.0
            im = Image.new('RGB', (W, H), SLATE_BG)
            d = ImageDraw.Draw(im)
            m, top = int(round(22 * s)), int(round(64 * s))       # the top edge clears the label and frame number
            d.rectangle([m, top, W - 1 - m, H - 1 - m], outline=(92, 88, 80), width=max(1, int(round(3 * s))))
            cy = self._slate_y(a, b) * s
            line1, line2 = 'MISSING ON THIS MAC:', f'{label} C{a}-{b - 1}'
            f1 = AS._fit(d, line1, 'Cinzel.ttf', 76 * s, W * 0.86)
            f2 = AS._fit(d, line2, 'Cinzel.ttf', 58 * s, W * 0.86)
            d.text((W / 2, cy - 44 * s), line1, font=f1, fill=(236, 226, 206), anchor='mm')
            d.text((W / 2, cy + 44 * s), line2, font=f2, fill=(214, 196, 160), anchor='mm')
            self._slate_cache[key] = np.asarray(im, np.float32) / 255.0
        return self._slate_cache[key].copy()

    def picture(self, f):                                         # replaced by _transitions' wrapper (outer = this)
        if f in self.missing:
            i, shot = self.shot_at(f)
            return self.missing_slate(f), shot, 'SLATE (missing on this machine)', None
        return AS.Ctx.picture(self, f)

    # ------------------------------------------------------------------------------------------------ labels
    def _make_label(self, text):
        s = self.W / 1920.0
        size = max(9, int(round(26 * s)))
        font = AS._pil_font('Cinzel.ttf', size)
        probe = Image.new('L', (10, 10))
        tw = int(ImageDraw.Draw(probe).textlength(text, font=font)) + 2 * size
        th = int(size * 1.8)
        a = Image.new('L', (tw, th), 0)
        ImageDraw.Draw(a).text((size, th / 2), text, font=font, fill=255, anchor='lm')
        return np.asarray(a, np.float32) / 255.0

    def _burn_label(self, img, f):
        a = self._label
        h, w = a.shape
        y0 = int(round(8 * self.W / 1920.0))
        x0 = self.W - w - y0 if self.storyboard else y0
        reg = img[y0:y0 + h, x0:x0 + w]
        reg *= (1 - 0.55 * np.clip(cv2.GaussianBlur(a, (0, 0), 3), 0, 1))[..., None]
        reg[:] = reg * (1 - a[..., None] * 0.85) + np.array([0.93, 0.86, 0.72], np.float32) * a[..., None] * 0.85
    def _burn_number(self, out, f):
        """The video (not the storyboard) carries its C5 frame number, top right, for review notes."""
        txt = f'C {f:04d}'
        fs = 0.42 * self.W / 960.0
        y0 = int(round(8 * self.W / 1920.0))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        org = (self.W - tw - y0 * 2, y0 + th + 4)
        cv2.putText(out, txt, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (230, 217, 191), 1, cv2.LINE_AA)

    def frame(self, f):
        img, shot, status, src = self.picture(f)
        img = np.ascontiguousarray(img, np.float32)
        slate = status.startswith('SLATE')
        titles.composite_v3(img, self.lines + (self.standins if slate else []), f, dark_ground=slate)
        self._burn_label(img, f)
        out = np.ascontiguousarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
        if not self.storyboard:
            self._burn_number(out, f)
        return out


def git_state():
    def g(*a):
        try:
            return subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    dirty = g('status', '--porcelain', '--', 'edit')
    return dict(head=g('rev-parse', '--short', 'HEAD'), branch=g('rev-parse', '--abbrev-ref', 'HEAD'),
                edit_dirty=bool(dirty))


def storyboard_frames(ctx):
    """[(frame, what)] in cut order: one mid-caption frame per line, one caption-free frame per slate if it has one."""
    rows = titles.text_table(CUT)
    picks = [((r['f_in'] + r['f_out']) // 2, f"caption {r['id']}") for r in rows]
    busy = {f for r in rows for f in range(r['f_in'], r['f_out'])}
    for a, b, sec, label in ctx.spans:
        mid = (a + b) // 2
        free = sorted((f for f in range(a, b) if f not in busy), key=lambda f: abs(f - mid))
        picks.append((free[0] if free else mid, f'slate {label} C{a}-{b - 1}'))
    return sorted(picks)


def build(out_dir, scale=0.5, video=True, storyboard=True, frames=None):
    rep = RD.run(partial=True)
    if rep.failed():
        print(RD.render_text(rep))
        raise SystemExit('readiness FAILED in --partial mode: the structure is wrong; nothing built')
    os.makedirs(out_dir, exist_ok=True)
    ctx = PartialCtx(rep.runs, scale)
    W, H = ctx.W, ctx.H
    t0 = time.time()
    written = {}
    if frames:
        for f in frames:
            p = os.path.join(out_dir, f'C5_PARTIAL_f{f:04d}.png')
            cv2.imwrite(p, ctx.frame(f)[..., ::-1])
            written[f] = p
        print(f'wrote {len(frames)} frames to {out_dir} ({time.time() - t0:.1f} s)')
        return dict(frames=written)
    manifest = dict(what='C5 PARTIAL REVIEW CUT (silent, labelled; NOT a master, NOT final)', built_utc=time.strftime(
        '%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source=git_state(), size=[W, H], fps=FPS, frames=EDL.TOTAL[CUT],
        readiness=dict(mode='partial', fail=rep.count('FAIL'), gap=rep.count('GAP'), warn=rep.count('WARN'),
                       items=[i for i in rep.items if i['level'] in ('FAIL', 'GAP', 'WARN')]),
        slates=[dict(frames=[a, b - 1], section=sec, text=f'MISSING ON THIS MAC: {label} C{a}-{b - 1}')
                for a, b, sec, label in ctx.spans],
        captions=[dict(id=r['id'], row=r['row'], frames=[r['f_in'], r['f_out'] - 1], set=r['set'], line=r['line'],
                       drawn=('stand-in over the slate (baked in the missing render)' if r['set'] == 'in_picture'
                              else 'EDIT')) for r in titles.text_table(CUT)],
        transitions=[dict(frames=[t['f0'], t['f1'] - 1], kind=t['kind'], ready=t.get('ready', True),
                          in_this_build=('designed, not built: hard cut' if not t.get('ready', True) else
                                         'plain cut (a side is slated)' if any(f in ctx.missing for f in
                                                                               range(t['f0'], t['f1'])) else
                                         'no finish in a partial: none' if t['kind'] == 'finish_ramp' else 'rendered'))
                     for t in EDL.TRANS[CUT]])
    mp4 = None
    if video:
        mp4 = os.path.join(out_dir, f'C5_PARTIAL_review_{W}x{H}.mp4')
        tmp = mp4 + '.part.mp4'
        cmd = [FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-framerate', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
               '-movflags', '+faststart', tmp]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in range(EDL.TOTAL[CUT]):
            proc.stdin.write(ctx.frame(f).tobytes())
            if f % 480 == 0:
                print(f'  frame {f}/{EDL.TOTAL[CUT]}  {time.time() - t0:5.0f}s', flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit('ffmpeg failed')
        os.replace(tmp, mp4)
        manifest['video'] = os.path.basename(mp4)
        print(f'wrote {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB, {time.time() - t0:.0f} s)', flush=True)
    if storyboard:
        sb = os.path.join(out_dir, 'storyboard_C5_PARTIAL')
        os.makedirs(sb, exist_ok=True)
        sctx = PartialCtx(rep.runs, scale, storyboard=True)
        picks = storyboard_frames(sctx)
        manifest['storyboard'] = []
        for k, (f, what) in enumerate(picks, 1):
            name = f'{k:02d}_C{f:04d}.png'
            cv2.imwrite(os.path.join(sb, name), sctx.frame(f)[..., ::-1])
            manifest['storyboard'].append(dict(file=name, frame=f, what=what))
        print(f'wrote {len(picks)} storyboard frames to {sb}')
    with open(os.path.join(out_dir, 'C5_PARTIAL_manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1, default=list)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--frames', default=None, help='comma-separated C5 frames: PNGs only')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--storyboard', action='store_true', help='with --no-video: the storyboard alone')
    a = ap.parse_args(argv)
    frames = [int(x) for x in a.frames.split(',')] if a.frames else None
    build(os.path.abspath(a.out), a.scale, video=not a.no_video and not frames,
          storyboard=(a.storyboard or not a.no_video) and not frames, frames=frames)
    return 0


if __name__ == '__main__':
    sys.exit(main())
