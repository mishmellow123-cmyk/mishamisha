"""THE LONG DAWN v3: preview exports of the finished stretches (a standing request from the producer).

Every continuous, fully rendered stretch of a film of about 30 s or more is cut frame-exactly from that film's
current master (picture + score) into ~/Downloads/The Long Dawn v3 - PREVIEWS/<film>_bars<a>-<b>_<what>.mp4, and
README.txt there says, one line per file, what it is and what is still missing or temporary.

    python3 edit/previews.py          # (the watcher runs it after the masters; bash edit/previews.sh queues it)

A stretch qualifies when every frame is rendered (stand-in renders and EDIT proxies only as a small part, planned
black allowed, trimmed to 1 s at the edges), it lasts >= 25 s (B's 27 s DUSK counts), and >= 60% of its non-black
frames are final renders. A stretch that grows replaces its old file; one whose inputs change (any source frame,
the text, the score) is re-exported under the same name. Each new file prints `NEW: <name>` (the watcher then
wakes EDIT, who tells the director).
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'edit'))
import assemble as AS  # noqa: E402
import deliver as D  # noqa: E402
from h9_kit import Film, FILM  # noqa: E402

OUT = os.path.expanduser('~/Downloads/The Long Dawn v3 - PREVIEWS')
MANIFEST = os.path.join(OUT, '.previews.json')
FPS, MIN_F, EDGE_BLACK = 24, 600, 24
OK = {'RENDERED', 'STAND_IN', 'PROXY', 'BLACK'}
VUI = D.VUI


def tc(f):
    s = f / FPS
    return f'{int(s // 60)}:{s % 60:04.1f}'


def slug(name):
    short = name.split('·')[-1].strip()
    words = [w for w in short.replace(',', ' ').split() if w.upper() not in ('THE', 'A', 'OF')]
    return (words[-1] if words else short).lower().strip("'")


def stretches(film):
    st = [film.status(f)[0] for f in range(film.n)]
    out, f = [], 0
    while f < film.n:
        if st[f] not in OK:
            f += 1
            continue
        g = f
        while g < film.n and st[g] in OK:
            g += 1
        a, b = f, g
        lead = next((k for k in range(a, b) if st[k] != 'BLACK'), b) - a
        a += max(0, lead - EDGE_BLACK)
        trail = b - 1 - next((k for k in range(b - 1, a - 1, -1) if st[k] != 'BLACK'), a - 1)
        b -= max(0, trail - EDGE_BLACK)
        c = Counter(st[a:b])
        n = b - a
        if n >= MIN_F and c['RENDERED'] >= 0.6 * max(1, n - c['BLACK']) and c['PROXY'] <= 0.25 * n:
            out.append((a, b, c))
        f = g
    return out


def describe(film, a, b):
    idx = [i for i, s in enumerate(film.shots) if s['f0'] < b and s['f1'] > a]
    shots = [film.shots[i] for i in idx]
    what = slug(shots[0]['name']) if len(shots) == 1 or slug(shots[0]['name']) == slug(shots[-1]['name']) else \
        f"{slug(shots[0]['name'])}-to-{slug(shots[-1]['name'])}"
    temp = []
    for i in idx:
        s, pl = film.shots[i], film.plans[i]
        ks = Counter(film.status(f)[0] for f in range(max(a, s['f0']), min(b, s['f1'])))
        if ks.get('STAND_IN'):
            temp.append(f"{s['sec']} {s['name']} is a stand-in ({(pl['take'] or {}).get('note', '')})")
        if ks.get('PROXY'):
            temp.append(f"{s['sec']} {s['name']} is an EDIT proxy until its plate lands")
    return idx, what, temp


def seg_keys(film, idx):
    """Per shot, the keys its master segment may have now: unfinished, and (rendered shots, when the masters
    carry the film finish) finished; the FINISH backlog may still hold the unfinished one."""
    code = D._code_hash()
    prof = D.PROFILES['master']
    table = AS.titles.text_table(film.cut)
    fin = D._finish_id() if prof.get('finish') else None
    out = []
    for i in idx:
        s, pl = film.shots[i], film.plans[i]
        ks = {D.segment_key(film.cut, None, prof, i, s, pl, code, table)}
        if fin and pl['kind'] == 'take' and pl['have'] > 0:
            ks.add(D.segment_key(film.cut, None, prof, i, s, pl, code, table, fin))
        out.append(ks)
    return out


def master_segments(film, idx):
    man = os.path.join(D.DELIVERY, 'cache', 'master', 'manifests.json')
    segs = json.load(open(man)).get(f'{film.cut}_master', []) if os.path.exists(man) else []
    return [os.path.basename(segs[i])[:-5] if i < len(segs) else None for i in idx]


def master_current(film, idx, keys):
    """Was the film's master built from these renders (else it is older than them: wait for it)?"""
    return all(m in ks for m, ks in zip(master_segments(film, idx), keys))


def key(film, idx, audio):
    """What the preview is cut from: the master's own segments for these shots (finish included) and the sound."""
    st = os.stat(audio) if os.path.isfile(audio) else None
    return json.dumps([master_segments(film, idx), audio, st.st_mtime_ns if st else 0])


def export(mov, a, b, path):
    d = (b - a) / FPS
    tmp = path + '.part.mp4'
    cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{a / FPS:.6f}', '-i', mov, '-t', f'{d:.6f}',
           '-map', '0:v', '-map', '0:a',
           '-vf', f'fade=t=in:st=0:d=0.25,fade=t=out:st={d - 0.25:.3f}:d=0.25',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', '-bf', '2',
           '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
           '-bsf:v', VUI, '-af', f'afade=t=in:d=0.25,afade=t=out:st={d - 0.5:.3f}:d=0.5', '-c:a', 'aac_at',
           '-b:a', '320k', '-movflags', '+faststart', tmp]
    if subprocess.run(cmd).returncode != 0:
        cmd[cmd.index('aac_at')] = 'aac'
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit(f'preview export failed: {path}')
    os.replace(tmp, path)


def main():
    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    stamp = time.strftime('%d %b %H:%MZ', time.gmtime())
    lines, keep, new = [], set(), []
    for cut in 'ABC':
        mov = os.path.join(D.DELIVERY, f'{cut}_master.mov')
        if not os.path.exists(mov):
            continue
        film = Film(cut)
        audio, label = AS.resolve_audio(cut)
        score = label.split(' (')[0] + ' ' + os.path.basename(audio) if 'CLICK' not in label.upper() else 'no score yet'
        for a, b, c in stretches(film):
            idx, what, temp = describe(film, a, b)
            b0, b1 = a // 80 + 1, (b - 1) // 80 + 1
            name = f'{cut}_bars{b0:02d}-{b1:02d}_{what}.mp4'
            path = os.path.join(OUT, name)
            keys = seg_keys(film, idx)
            if not master_current(film, idx, keys):        # the master is older than these renders: wait for it
                if name in man and os.path.exists(path):
                    keep.add(name)
                continue
            k = key(film, idx, audio)
            if man.get(name, {}).get('key') != k or not os.path.exists(path):
                export(mov, a, b, path)
                if name not in man:
                    new.append(name)
                man[name] = dict(key=k, cut=cut, f0=a, f1=b, at=stamp)
            keep.add(name)
            secs = ', '.join(f"{film.shots[i]['sec']} {film.shots[i]['name']}" for i in idx)
            lines.append(f"{name}: {cut} · {FILM[cut]} · bars {b0}-{b1} · {tc(a)}-{tc(b)} ({(b - a) / FPS:.1f} s) · "
                         f"{secs} · sound: {score} · temporary: " + ('; '.join(temp) if temp else 'nothing beyond the '
                                                                     'general notes above') + f" · exported {man[name]['at']}")
    for name in list(man):                              # a stretch that grew or vanished: its old file goes
        if name not in keep:
            p = os.path.join(OUT, name)
            if os.path.exists(p):
                os.remove(p)
            del man[name]
    json.dump(man, open(MANIFEST + '.part', 'w'), indent=1)
    os.replace(MANIFEST + '.part', MANIFEST)
    head = [f'THE LONG DAWN v3 · PREVIEWS · updated {stamp}',
            'Each file is a continuous, fully rendered stretch of one film, cut frame-exactly from that film\'s current '
            'master with its sound. Files update automatically when a stretch grows or its renders or score improve.',
            'General notes for every file: work in progress; no film-look pass yet (grain, final grade); the rest of '
            'each film is still being rendered.', '']
    open(os.path.join(OUT, 'README.txt'), 'w').write('\n'.join(head + (lines or ['(no finished stretch yet)'])) + '\n')
    for n in new:
        print(f'NEW: {n}')
    print(f'previews: {len(keep)} files in {OUT}')


if __name__ == '__main__':
    main()
