"""THE LONG DAWN v3: the H9 critic kit, rebuilt from the current EDL and renders by one command.

    bash edit/h9_kit.sh                         # all three films (a few minutes)
    python3 edit/h9_kit.py --cuts A --workers 2 # one film

Writes ~/mishamisha/_local_logs/review/h9/ (the critic judges from these without watching video):
    INDEX.md                         what is here, how to read it, the legend, the gate's four questions
    COVERAGE.md                      one line per section: rendered, stand-in, slate or variant (and the ALT master)
    TEXT.md                          every line as it reads on screen, where it lands and over what
    <cut>_overview.jpg               the whole film on one page: one tiny frame per bar, a status stripe per bar
    <cut>_bars_<a>-<b>.jpg           the contact sheets: one frame per bar, 12 bars a page, captions with the bar,
                                     frame, time, section and shot, status and the words on screen
    A_ALT_codedtowers_bars_<a>-<b>.jpg   the producer alternate: MAIN beside ALT on the bars where they differ
    stills/<cut>/<cut>_bar<NN>_f<NNNNN>.jpg   the same frames at full resolution (1920x804), for zooming

The frame for a bar is its beat 3 (bar start + 40); when a line is on screen in the bar, it is the frame nearest
beat 3 where the most lines are fully drawn. Frames are clean (no burn-ins) with the text composited as in the film.
"""
import argparse
import math
import os
import re
import shutil
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'edit'))
import assemble as AS  # noqa: E402  (Ctx, plan_shot, locate, coverage, the audio resolver)

EDL = AS.EDL
titles = AS.titles
OUT = os.path.expanduser('~/mishamisha/_local_logs/review/h9')
FPS, BAR = 24, 80
TW, TH = 480, 201                                   # sheet thumbnail: a quarter of the 1920x804 picture
FILM = {'A': 'EVERY STEP CLOSER', 'B': 'THE VIGIL', 'C': 'THE LAST PAGES'}
AVENIR = '/System/Library/Fonts/Avenir Next Condensed.ttc'
FACE = {'bold': 0, 'demi': 2, 'medium': 5, 'italic': 6, 'regular': 7}
PER_PAGE, COLS = 12, 3

# status of a frame, as the critic should read it
COL = dict(RENDERED=(104, 196, 112), STAND_IN=(230, 172, 64), VARIANT=(104, 168, 240), SLATE=(170, 78, 78),
           BLACK=(128, 128, 138), PROXY=(176, 140, 226))
LABEL = dict(RENDERED='RENDERED', STAND_IN='STAND-IN', VARIANT='ALT VARIANT', SLATE='SLATE', BLACK='BLACK (by design)',
             PROXY='EDIT PROXY')
MEANING = dict(
    RENDERED='the department\'s current render of the shot',
    STAND_IN='rendered, but a test, preview or pre-H5 look that a newer render will replace (judge it, knowing that)',
    SLATE='not rendered yet: a dark card naming the intended shot, at its exact length (the story still reads)',
    BLACK='planned black: the films open, and breathe, on black',
    PROXY='an edit-built stand-in (the X2 star field) until the department\'s plate lands',
    VARIANT='the producer\'s labelled alternate for A (the two giants as coded towers); shown beside MAIN')
STAND_IN_RX = re.compile(r'\b(pre-h5|test|preview|quarter-res|half-res|stand-in)\b', re.I)
SET_NAME = dict(lower='lower third', row='lower third, one row (T6b joins T6a)', black_top='on black, upper line',
                black_bottom='on black, lower line', ink='ink write-on, lower third', ink_page='ink on the page, centre',
                fire='fire letters, lower third', fire_black_top='fire letters on black, upper line',
                fire_black_bottom='fire letters on black, lower line', title='main title (Cinzel)',
                in_picture='main title, burned into the page in the render (book space)')

_FONTS = {}


def font(face, size):
    key = (face, size)
    if key not in _FONTS:
        try:
            _FONTS[key] = ImageFont.truetype(AVENIR, size, index=FACE[face])
        except OSError:
            name = 'EBGaramond-Italic.ttf' if face == 'italic' else 'EBGaramond.ttf'
            _FONTS[key] = ImageFont.truetype(os.path.join(AS.FONTS, name), size)
    return _FONTS[key]


def tc(f):
    s = f / FPS
    return f'{int(s // 60)}:{s % 60:05.2f}'


def bb(f):
    return f // BAR + 1, (f % BAR) / 20.0 + 1.0


def bbs(f):
    """'24 b1.4': bar and beat, the beat without trailing zeros."""
    bar, beat = bb(f)
    return f'{bar} b{beat:g}'


# --------------------------------------------------------------------------------------------- the text
def legible(r, f):
    """The line is fully drawn at f (past its write-on / kindle / fade-in, before its fade-out)."""
    title = r['set'] in ('title', 'in_picture')
    lead = 26 if r['set'].startswith('ink') else 30 if title else 14
    tail = 50 if title else 14
    return r['f_in'] + lead <= f < r['f_out'] - tail


def choose(bar, table):
    """Beat 3 of the bar, or the frame nearest it where the most lines are fully drawn."""
    b0 = (bar - 1) * BAR
    mid = b0 + 40
    best = (0, 0, mid)
    for f in range(b0, b0 + BAR):
        key = (sum(legible(r, f) for r in table), -abs(f - mid))
        if key > best[:2]:
            best = key + (f,)
    return best[2]


# -------------------------------------------------------------------------------------- status per frame
class Film:
    """One cut's plan (as the animatic uses it) and each frame's status."""

    def __init__(self, cut, variant=None):
        self.cut, self.variant = cut, variant
        self.shots = EDL.EDL[cut]
        self.plans = [AS.plan_shot(s, cut, variant) for s in self.shots]
        self.table = titles.text_table(cut)
        self.n = EDL.TOTAL[cut]
        self.bars = self.n // BAR

    def shot_i(self, f):
        for i, s in enumerate(self.shots):
            if s['f0'] <= f < s['f1']:
                return i
        raise IndexError(f)

    def status(self, f):
        """(klass, alt) for cut frame f."""
        i = self.shot_i(f)
        plan = self.plans[i]
        if plan['kind'] == 'black':
            return 'BLACK', False
        if plan['kind'] == 'x2':
            return 'PROXY', False
        if plan['kind'] != 'take':
            return 'SLATE', False
        p, alt = AS.locate(plan['take'], self.cut, self.variant, f)
        if not p:
            return 'SLATE', False
        if alt:
            return 'VARIANT', True
        return ('STAND_IN' if STAND_IN_RX.search(plan['take'].get('note') or '') else 'RENDERED'), False

    def note(self, f):
        plan = self.plans[self.shot_i(f)]
        return (plan['take'] or {}).get('note') or '' if plan['kind'] == 'take' else ''

    def src(self, f):
        plan = self.plans[self.shot_i(f)]
        if plan['kind'] != 'take':
            return ''
        t = plan['take']
        return f"{t['stem']} {f + t['off']}"

    def words(self, f):
        """[(id, line, fully drawn)] of the lines on screen at f."""
        return [(r['id'], r['line'], legible(r, f)) for r in self.table if r['f_in'] <= f < r['f_out']]

    def counts(self):
        c = {}
        for f in range(self.n):
            k, _ = self.status(f)
            c[k] = c.get(k, 0) + 1
        return c


# ------------------------------------------------------------------------------------------ still frames
_CTX = {}


def _still(job):
    cut, variant, f, path = job
    key = (cut, variant)
    if key not in _CTX:
        _CTX[key] = AS.Ctx(cut, variant, 1.0, True)
    ctx = _CTX[key]
    img = ctx.frame(f)
    if len(ctx._slates) > 3:                         # full-res slate cards are 4.6 MB each
        ctx._slates.clear()
    tmp = path + '.part.jpg'
    cv2.imwrite(tmp, img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    os.replace(tmp, path)
    return path


def thumb(path, w=TW, h=TH):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(img[..., ::-1])


# -------------------------------------------------------------------------------------------- drawing
def wrap(d, text, fnt, width, max_lines=2):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if d.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and d.textlength(lines[-1] + ' …', font=fnt) > width:
            lines[-1] = lines[-1].rsplit(' ', 1)[0] if ' ' in lines[-1] else lines[-1][:-1]
        lines[-1] += ' …'
    return lines


def fit(d, text, face, size, width):
    while size > 9 and d.textlength(text, font=font(face, size)) > width:
        size -= 1
    return font(face, size)


def header(d, W, left, right, y=14):
    d.text((12, y), left, font=font('demi', 21), fill=(232, 232, 238))
    d.text((W - 12, y + 3), right, font=font('regular', 15), fill=(150, 150, 162), anchor='ra')


def draw_cell(im, d, x, y, it, film, first_of_shot):
    im.paste(it['img'], (x, y))
    k = it['klass']
    d.rectangle([x, y + TH, x + TW - 1, y + TH + 3], fill=COL[k])
    badge = str(it['bar'])
    bw = int(d.textlength(badge, font=font('bold', 22))) + 14
    d.rectangle([x, y, x + bw, y + 28], fill=(0, 0, 0))
    d.text((x + 7, y + 2), badge, font=font('bold', 22), fill=(255, 255, 255))
    f = it['f']
    bar, beat = bb(f)
    cy = y + TH + 8
    d.text((x, cy), f'bar {bar} · b{beat:.2f} · f {f} · {tc(f)}', font=font('medium', 16), fill=(205, 205, 212))
    d.text((x + TW, cy), LABEL[k], font=font('demi', 16), fill=COL[k], anchor='ra')
    cy += 20
    s = film.shots[film.shot_i(f)]
    shot = f"{s['sec']} · {s['code']} {s['name']}"
    d.text((x, cy), shot, font=fit(d, shot, 'medium', 16, TW), fill=(172, 172, 182))
    cy += 21
    words = film.words(f)
    if words:
        txt = '  /  '.join(f'“{w}”' + ('' if ok else ' (drawing on)') for _, w, ok in words)
        ids = ' '.join(i for i, _, _ in words)
        for ln in wrap(d, f'{txt}  [{ids}]', font('italic', 17), TW):
            d.text((x, cy), ln, font=font('italic', 17), fill=(250, 244, 226))
            cy += 20
    elif k == 'SLATE':
        txt = s['desc'] if first_of_shot else f"(slate continues: {s['name']})"
        for ln in wrap(d, txt, font('italic', 15), TW):
            d.text((x, cy), ln, font=font('italic', 15), fill=(150, 150, 160))
            cy += 18
    elif k in ('STAND_IN', 'PROXY'):
        txt = film.note(f) or 'edit-built star field until the RUN-A plate lands'
        for ln in wrap(d, txt, font('italic', 15), TW):
            d.text((x, cy), ln, font=font('italic', 15), fill=(206, 160, 76))
            cy += 18


CAP = 86                                              # caption height under a thumbnail


def sheet_page(film, items, path, page, pages, stamp):
    gap, head = 12, 50
    rows = math.ceil(len(items) / COLS)
    W = COLS * TW + (COLS + 1) * gap
    H = head + rows * (TH + CAP + gap) + gap
    im = Image.new('RGB', (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    b0, b1 = items[0]['bar'], items[-1]['bar']
    header(d, W, f'THE LONG DAWN v3 · {film.cut} · {FILM[film.cut]} · bars {b0}–{b1} of {film.bars}',
           f'sheet {page}/{pages} · one frame per bar (80 f = 3.33 s) · built {stamp}')
    for k, it in enumerate(items):
        r, c = divmod(k, COLS)
        draw_cell(im, d, gap + c * (TW + gap), head + r * (TH + CAP + gap), it, film, it['first_of_shot'])
    im.save(path, quality=88)


def overview(film, items, path, stamp, summary):
    cols, tw, th, gap, lab, stripe = 10, 150, 63, 6, 17, 5
    rows = math.ceil(len(items) / cols)
    W = cols * tw + (cols + 1) * gap
    head, legend = 50, 34
    H = head + rows * (th + stripe + lab + gap) + gap + legend
    im = Image.new('RGB', (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    header(d, W, f'THE LONG DAWN v3 · {film.cut} · {FILM[film.cut]} · every bar', f'{summary} · built {stamp}')
    sec_first = {}
    for s in film.shots:
        sec_first.setdefault(s['sec'], s['f0'] // BAR + 1)
    for k, it in enumerate(items):
        r, c = divmod(k, cols)
        x, y = gap + c * (tw + gap), head + r * (th + stripe + lab + gap)
        im.paste(it['img'].resize((tw, th), Image.LANCZOS), (x, y))
        b0 = (it['bar'] - 1) * BAR
        for j in range(BAR):                               # the stripe: every frame's status in this bar
            kk, _ = film.status(b0 + j)
            xx = x + int(tw * j / BAR)
            d.rectangle([xx, y + th, x + int(tw * (j + 1) / BAR), y + th + stripe - 1], fill=COL[kk])
        label = str(it['bar'])
        secs = [sid for sid, b in sec_first.items() if b == it['bar']]
        ly = y + th + stripe + 1
        d.text((x, ly), label, font=font('demi', 14), fill=(205, 205, 212))
        if secs:                                            # the section(s) starting in this bar, beside its number
            lx = x + int(d.textlength(label + ' ', font=font('demi', 14)))
            d.text((lx, ly), ' '.join(secs), font=font('demi', 14), fill=(236, 190, 110))
        if film.words(it['f']):
            d.text((x + tw, ly), 'text', font=font('italic', 13), fill=(250, 244, 226), anchor='ra')
    ly = H - legend + 8
    x = 12
    for kk in ('RENDERED', 'STAND_IN', 'PROXY', 'SLATE', 'BLACK'):
        d.rectangle([x, ly + 4, x + 18, ly + 12], fill=COL[kk])
        d.text((x + 24, ly), LABEL[kk], font=font('medium', 15), fill=(200, 200, 208))
        x += 34 + int(d.textlength(LABEL[kk], font=font('medium', 15))) + 14
    d.text((x + 10, ly), 'amber id = the section that starts in this bar · "text" = a line is on screen',
           font=font('italic', 15), fill=(150, 150, 162))
    im.save(path, quality=88)


def alt_pages(main, alt, pairs, stamp, out):
    """MAIN | ALT side by side, 5 bars a page."""
    paths = []
    gap, head, cap = 12, 50, 34
    for p0 in range(0, len(pairs), 5):
        chunk = pairs[p0:p0 + 5]
        W = 2 * TW + 3 * gap
        H = head + len(chunk) * (TH + cap + gap) + gap
        im = Image.new('RGB', (W, H), (18, 18, 22))
        d = ImageDraw.Draw(im)
        b0, b1 = chunk[0]['bar'], chunk[-1]['bar']
        header(d, W, f'A · ALT (producer alternate, coded towers) · bars {b0}–{b1}', f'left MAIN · right ALT · {stamp}')
        for r, it in enumerate(chunk):
            y = head + r * (TH + cap + gap)
            for c, (img, lab, k) in enumerate(((it['main'], 'MAIN', it['kmain']), (it['alt'], 'ALT', 'VARIANT'))):
                x = gap + c * (TW + gap)
                im.paste(img, (x, y))
                d.rectangle([x, y + TH, x + TW - 1, y + TH + 3], fill=COL[k])
                d.rectangle([x, y, x + 44, y + 28], fill=(0, 0, 0))
                d.text((x + 7, y + 2), str(it['bar']), font=font('bold', 22), fill=(255, 255, 255))
                s = main.shots[main.shot_i(it['f'])]
                d.text((x, y + TH + 8), f"{lab} · bar {it['bar']} · f {it['f']} · {s['sec']} {s['name']}",
                       font=fit(d, f"{lab} · bar {it['bar']} · f {it['f']} · {s['sec']} {s['name']}", 'medium', 16, TW),
                       fill=COL[k] if lab == 'ALT' else (205, 205, 212))
        p = os.path.join(out, f'A_ALT_codedtowers_bars_{b0:02d}-{b1:02d}.jpg')
        im.save(p, quality=88)
        paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------- the texts
def summary_line(film, counts):
    n = film.n
    pic = counts.get('RENDERED', 0)
    si = counts.get('STAND_IN', 0)
    sl = counts.get('SLATE', 0)
    parts = [(pic, 'rendered'), (si, 'stand-in'), (sl, 'slate')]
    return ' · '.join(f'{name} {100 * v / n:.0f}%' for v, name in parts if v)


def coverage_md(films, alt, stamp, audio):
    L = [f'# H9 kit · coverage, section by section', '',
         f'_Built {stamp} by `edit/h9_kit.sh` from `edit/edl_v3.py` and the renders on disk._', '',
         'Status of the picture only. ' + ' '.join(f'**{LABEL[k]}**: {MEANING[k]}.' for k in
                                                   ('RENDERED', 'STAND_IN', 'SLATE', 'BLACK', 'PROXY')), '']
    for cut, film in films.items():
        c = film.counts()
        runtime = tc(film.n)
        L.append(f'## {cut} · {FILM[cut]} · {film.bars} bars · {film.n:,} f · {runtime}')
        L.append('')
        L.append('Picture: ' + ' · '.join(f'{LABEL[k]} {c[k]:,} f ({100 * c[k] / film.n:.0f}%)' for k in
                                          ('RENDERED', 'STAND_IN', 'PROXY', 'BLACK', 'SLATE') if c.get(k)) +
                 f'. Sound in the animatic: {audio[cut]}.')
        L.append('')
        by_sec = {}
        for i, s in enumerate(film.shots):
            by_sec.setdefault(s['sec'], []).append(i)
        for sec, idx in by_sec.items():
            s0, s1 = film.shots[idx[0]], film.shots[idx[-1]]
            b0, b1 = bb(s0['f0'])[0], bb(s1['f1'] - 1)[0]
            bars = f'bar {b0}' if b0 == b1 else f'bars {b0}-{b1}'
            parts = []
            for i in idx:
                s = film.shots[i]
                ks = {}
                for f in range(s['f0'], s['f1']):
                    k, _ = film.status(f)
                    ks[k] = ks.get(k, 0) + 1
                n = s['f1'] - s['f0']
                if len(ks) == 1:
                    st = LABEL[next(iter(ks))]
                else:
                    st = ', '.join(f'{LABEL[k]} {v}/{n} f' for k, v in sorted(ks.items(), key=lambda kv: -kv[1]))
                extra = ''
                if 'SLATE' in ks:
                    extra = f' — _{s["desc"]}_'
                elif film.plans[i]['kind'] == 'take' and film.plans[i]['take'].get('note'):
                    extra = f' ({film.plans[i]["take"]["note"]})'
                parts.append(f'{s["code"]} {s["name"]}: **{st}**{extra}')
            L.append(f'- **{sec}** · {bars} · ' + ' · '.join(parts))
        if cut == 'A' and alt is not None:
            L.append('')
            diff = {}
            for f in range(alt.n):
                k, a = alt.status(f)
                if a:
                    sec = alt.shots[alt.shot_i(f)]['sec']
                    diff[sec] = diff.get(sec, 0) + 1
            if diff:
                L.append('**ALT master** (the producer\'s labelled alternate, `_alt_codedtowers`: the two giants as coded '
                         'towers): differs from MAIN in ' + ', '.join(
                             f'{sec} ({v} f)' for sec, v in diff.items()) + '; everywhere else it is MAIN. '
                         'See `A_ALT_codedtowers_bars_*.jpg`.')
            else:
                L.append('**ALT master**: no `_alt_codedtowers` frames yet, so it is identical to MAIN.')
        L.append('')
    return '\n'.join(L)


def text_md(films, stamp):
    L = ['# H9 kit · the text', '',
         f'_Built {stamp} from the bar maps (`music/v3/barmap_*.json`) with the director\'s H5 wording, exactly as '
         'the films draw it (`edit/titles.py`)._', '',
         'Frames are the film\'s own (24 fps; 1 bar = 80 f = 3.33 s). **Over** is the shot under the line at its '
         'midpoint, with its status (a line over a SLATE is drawn over the dark card in the sheets).', '']
    style = {'A': 'Cormorant Garamond italic, lower third; T10a/T10b centred on black; the title kindles in the sky.',
             'B': 'Wordless: the title only, kindling in the dawn sky and fading into the light.',
             'C': 'Ink lines (EB Garamond italic) write on with a pen nib; fire lines kindle and crumble into sparks; '
                  'T8a/T8b in fire on black; the title burns onto the blank page in the render itself and cools to '
                  'ink.'}
    for cut, film in films.items():
        L.append(f'## {cut} · {FILM[cut]}')
        L.append('')
        L.append(style[cut])
        L.append('')
        L.append('| id | line (as on screen) | frames | bar, beat: in → out | time in | dur | set | over |')
        L.append('|---|---|---|---|---|---|---|---|')
        for r in film.table:
            mid = (r['f_in'] + r['f_out']) // 2
            s = film.shots[film.shot_i(mid)]
            k, _ = film.status(mid)
            bi, bo = bbs(r['f_in']), bbs(r['f_out'])
            line = r['line'] if r['id'] != 'title' else 'THE LONG DAWN'
            L.append(f"| {r['id']} | *{line}* | {r['f_in']}–{r['f_out']} | {bi} → {bo} | "
                     f"{tc(r['f_in'])} | {(r['f_out'] - r['f_in']) / FPS:.1f} s | {SET_NAME.get(r['set'], r['set'])} | "
                     f"{s['sec']} {s['name']} ({LABEL[k]}) |")
        L.append('')
    return '\n'.join(L)


def index_md(films, stamp, audio, sheets, alt_sheets, summaries):
    L = ['# THE LONG DAWN v3 · H9 critic kit', '',
         f'_Built {stamp} by `bash ~/mishamisha/the-long-dawn/edit/h9_kit.sh` from the current edit decision lists and '
         'whatever has been rendered. Rebuild it the same way; every file here is regenerated._', '',
         'Three short films on one grid: 24 fps, 72 BPM, a bar is 80 frames (3.33 s). Anything not yet rendered '
         'appears as a SLATE, a dark card naming the intended shot at its exact length, so each film is complete '
         'end to end and the story can be told back from the sheets alone.', '']
    for cut, film in films.items():
        L.append(f'- **{cut} · {FILM[cut]}**: {film.bars} bars, {tc(film.n)}. Picture: {summaries[cut]}. '
                 f'Sound in the animatic: {audio[cut]}.')
    L += ['', '## Read in this order', '',
          '1. `<film>_overview.jpg`: the whole film on one page, one small frame per bar; the stripe under each '
          'frame shows the status of every frame in that bar.',
          '2. `<film>_bars_<a>-<b>.jpg`: the contact sheets, one frame per bar, 12 bars a page. Under each frame: '
          'the bar number (also on the frame), beat, frame and time; the section and shot; the status; the words '
          'on screen, if any (for a slate, the intended shot instead).',
          '3. `TEXT.md`: every line as it reads on screen, where it lands and what it plays over.',
          '4. `COVERAGE.md`: each section, rendered, stand-in, slate or variant.',
          '5. `stills/<film>/`: the same frames at full resolution (1920x804) for zooming in; the file name gives the '
          'bar and frame.']
    if alt_sheets:
        L.append('6. `A_ALT_codedtowers_bars_*.jpg`: the producer\'s labelled alternate master of A (the two giants '
                 'dressed as coded towers), MAIN beside ALT on the bars where they differ.')
    L += ['', '## Legend (the stripe and status colours)', '']
    for k in ('RENDERED', 'STAND_IN', 'SLATE', 'BLACK', 'PROXY', 'VARIANT'):
        L.append(f'- **{LABEL[k]}**: {MEANING[k]}.')
    L += ['', '## Which frame stands for a bar', '',
          'Beat 3 of the bar (its start + 40 frames). When a line is on screen in that bar, the frame nearest beat 3 '
          'where the most lines are fully drawn, so every line appears on the sheets as the audience reads it. The '
          'frames are clean (no timecode burn-ins); the text is composited exactly as in the films.', '',
          '## The gate (REVISION 1)', '',
          'A fresh-context critic answers four fixed questions for each film, from the sheets, the stills and the text:',
          '',
          '1. Tell the story back.',
          '2. Name any real company, country, deal or institution it evokes.',
          '3. Which moment would get a laugh?',
          '4. Does any frame read as a doll, clip-art, a poster or a tech demo?', '',
          '## Files', '']
    for p in sheets + alt_sheets:
        L.append(f'- `{os.path.basename(p)}`')
    L.append('- `COVERAGE.md`, `TEXT.md`, `stills/`')
    L.append('')
    return '\n'.join(L)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuts', default='ABC')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    cuts = [c for c in a.cuts.upper() if c in 'ABC']
    out = a.out
    EDL.check(os.path.join(ROOT, 'music', 'v3'))
    os.makedirs(out, exist_ok=True)
    stamp = time.strftime('%d %b %H:%MZ', time.gmtime())
    t0 = time.time()

    films = {c: Film(c) for c in cuts}
    alt = Film('A', 'codedtowers') if 'A' in cuts else None
    jobs, plan = [], {}
    for c, film in films.items():
        d = os.path.join(out, 'stills', c)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        plan[c] = []
        for bar in range(1, film.bars + 1):
            f = choose(bar, film.table)
            p = os.path.join(d, f'{c}_bar{bar:02d}_f{f:05d}.jpg')
            plan[c].append(dict(bar=bar, f=f, path=p))
            jobs.append((c, None, f, p))
    pairs = []
    if alt is not None:
        d = os.path.join(out, 'stills', 'A_ALT')
        if os.path.isdir(d):
            shutil.rmtree(d)
        for it in plan['A']:
            b0 = (it['bar'] - 1) * BAR
            fs = [f for f in range(b0, b0 + BAR) if alt.status(f)[1]]
            if not fs:
                continue
            os.makedirs(d, exist_ok=True)
            f = it['f'] if it['f'] in fs else min(fs, key=lambda x: abs(x - it['f']))
            pm = it['path'] if f == it['f'] else os.path.join(d, f"A_bar{it['bar']:02d}_f{f:05d}_MAIN.jpg")
            pa = os.path.join(d, f"A_bar{it['bar']:02d}_f{f:05d}_ALT.jpg")
            if pm != it['path']:
                jobs.append(('A', None, f, pm))
            jobs.append(('A', 'codedtowers', f, pa))
            pairs.append(dict(bar=it['bar'], f=f, pm=pm, pa=pa))
    print(f'h9 kit: {len(jobs)} frames at full resolution -> {out}/stills', flush=True)
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_still, jobs, chunksize=4)):
            if i % 60 == 0:
                print(f'  {i}/{len(jobs)}  {time.time() - t0:4.0f}s', flush=True)

    for old in os.listdir(out):                        # the sheets are rebuilt from scratch
        if old.endswith('.jpg') and re.match(r'([ABC]_(bars|overview)|A_ALT_codedtowers)', old):
            if old[0] in cuts:
                os.remove(os.path.join(out, old))
    sheets, alt_sheets, summaries, audio = [], [], {}, {}
    for c, film in films.items():
        counts = film.counts()
        summaries[c] = summary_line(film, counts)
        audio[c] = AS.resolve_audio_label(c).replace('`', '')
        items = []
        seen = set()
        for it in plan[c]:
            k, _ = film.status(it['f'])
            i = film.shot_i(it['f'])
            items.append(dict(bar=it['bar'], f=it['f'], img=thumb(it['path']), klass=k, first_of_shot=i not in seen))
            seen.add(i)
        p = os.path.join(out, f'{c}_overview.jpg')
        overview(film, items, p, stamp, summaries[c])
        sheets.append(p)
        pages = math.ceil(len(items) / PER_PAGE)
        for pg in range(pages):
            chunk = items[pg * PER_PAGE:(pg + 1) * PER_PAGE]
            p = os.path.join(out, f"{c}_bars_{chunk[0]['bar']:02d}-{chunk[-1]['bar']:02d}.jpg")
            sheet_page(film, chunk, p, pg + 1, pages, stamp)
            sheets.append(p)
    if pairs:
        for it in pairs:
            it['main'], it['alt'] = thumb(it['pm']), thumb(it['pa'])
            it['kmain'] = films['A'].status(it['f'])[0]
        alt_sheets = alt_pages(films['A'], alt, pairs, stamp, out)

    open(os.path.join(out, 'COVERAGE.md'), 'w').write(coverage_md(films, alt, stamp, audio))
    open(os.path.join(out, 'TEXT.md'), 'w').write(text_md(films, stamp))
    open(os.path.join(out, 'INDEX.md'), 'w').write(index_md(films, stamp, audio, sheets, alt_sheets, summaries))
    for c in cuts:
        print(f'{c}: {summaries[c]} · sound: {audio[c]}')
    print(f'H9 kit: {len(sheets) + len(alt_sheets)} sheets, {len(jobs)} stills, {time.time() - t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
