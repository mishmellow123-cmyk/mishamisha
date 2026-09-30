"""C5 local asset map: point the C5 cut's sources at where THIS machine keeps them, explicitly, one stem at a time.

The assembler reads renders/<stem>/f_%05d.(png|jpg). On a machine whose frames live elsewhere (the M4 keeps the nine
Codex C5 deliveries in its farm's out/ directory), this tool makes renders/<stem> a symlink to a directory NAMED in a
gitignored map file, after checking that the directory holds exactly the frames the C5 EDL reads from that stem.
Nothing is discovered: no globbing, no name variants, no newest-by-mtime; a stem missing from the map stays missing
(a slate in a --partial build, a FAIL in a full one).

    python3 edit/tools/c5_assets.py init --out-root DIR   # write the local map for the delivered stems
    python3 edit/tools/c5_assets.py link                  # verify each mapped directory, then symlink renders/<stem>
    python3 edit/tools/c5_assets.py check                 # report only (exit 1 if a mapped stem fails)

The map (edit/cache/c5_asset_map.local.json: edit/cache/ is gitignored, so machine paths never enter the public
repo; `init` regenerates it):
    {"version": 1, "stems": {"book_C5_refusal": "/abs/dir/book_C5_refusal", ...}}
The environment variable LD_C5_ASSET_MAP names another map file. A manifest of what was linked (targets, frame
counts, bytes) goes to edit/cache/c5_assets_manifest.json (gitignored).
"""
import argparse
import json
import os
import re
import sys

EDIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(EDIT)
RENDERS = os.path.join(ROOT, 'renders')
MAP_DEFAULT = os.path.join(EDIT, 'cache', 'c5_asset_map.local.json')          # gitignored (edit/cache/)
MANIFEST = os.path.join(EDIT, 'cache', 'c5_assets_manifest.json')
FRAME = re.compile(r'f_(\d+)\.(png|jpg)$')
ALLOWED_EXTRA = re.compile(r'receipt_\d+_\d+\.json$')       # RUN deliveries keep their receipts beside the frames

# The nine Codex C5 deliveries (PRs 11-14; CODEX-MAILBOX-HANDBACK): stem -> inclusive C5 frames, absolute numbering.
DELIVERED = {
    'book_C5_refusal': (2080, 2319), 'book_C5_refusal_matte': (2080, 2319),
    'embers_C5_trap': (2320, 2639),
    'runC_reveal_pair_v5': (2880, 3119),
    'map_last_beacon_C': (3440, 3839),
    'embers_C5_cold': (3840, 3999),
    'embers_C5_unfinished': (4000, 4239),
    'book_C5_deep_abandoned': (4240, 4479), 'book_C5_deep_abandoned_matte': (4240, 4479),
    'runC_watch_v5': (4480, 4719),
    'book_C5_pen': (5440, 5679), 'book_C5_pen_matte': (5440, 5679),
    # Codex PR #15's candidates, farm-rendered 29 Sep (5bc5fe9); adopted in edl_v3 but for the Reveal's (an alternative)
    'cand_t1_current-words': (320, 559), 'cand_t1_current-words_matte': (320, 559),
    'cand_t1_current-words-held': (320, 559), 'cand_t1_current-words-held_matte': (320, 559),
    'cand_trap_front_smoke_near': (2320, 2639),
    'cand_reveal_night-fire': (2880, 3119),
    'cand_map_beacon-falloff': (3440, 3839),
    'cand_cold_lead24': (3816, 3999),
    'cand_deep_leaned_ladders': (4240, 4479), 'cand_deep_leaned_ladders_matte': (4240, 4479),
    'cand_watch_night-fire': (4480, 4719),
    'cand_pen_soft_spine_metal': (5440, 5679), 'cand_pen_soft_spine_metal_matte': (5440, 5679),
}


def map_path():
    return os.environ.get('LD_C5_ASSET_MAP') or MAP_DEFAULT


def load_map(path=None):
    path = path or map_path()
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get('version') != 1 or not isinstance(doc.get('stems'), dict):
        raise SystemExit(f'{path}: expected {{"version": 1, "stems": {{...}}}}')
    return doc['stems']


def needed_frames(cut='C'):
    """{stem: set of source frames} the cut's PRIMARY takes read (take, matte, add, under), from the EDL itself."""
    sys.path.insert(0, EDIT)
    import edl_v3 as EDL
    need = {}
    for s in EDL.EDL[cut]:
        if not s['takes']:
            continue
        t = s['takes'][0]
        if t['mode'] != 'exact':
            continue
        for f in range(s['f0'], s['f1']):
            src = f + t['off']
            for stem in (t['stem'], t.get('matte'), t.get('add')):
                if stem:
                    need.setdefault(stem, set()).add(src)
            if t.get('under'):
                how, stem, *rest = t['under']
                need.setdefault(stem, set()).add(rest[0] if how == 'hold' and rest else f)
    return need


def survey(d):
    """(frames {n: filename}, other file names) of a directory, without reading any image."""
    frames, other = {}, []
    for n in sorted(os.listdir(d)):
        m = FRAME.match(n)
        if m:
            k = int(m.group(1))
            if k in frames:
                other.append(n)                                   # a png beside a jpg: ambiguous, reported
            else:
                frames[k] = n
        elif not n.startswith('.'):
            other.append(n)
    return frames, other


def split_need(stem, need):
    """(frames inside the recorded delivery, frames beyond it). Frames the EDL reads beyond a delivery are OWED by a new
    delivery (a lit lead-in once edl_v3.COLD_CUT moves earlier, say), not a fault of the folder that was delivered."""
    if stem not in DELIVERED:
        return set(need), set()
    a, b = DELIVERED[stem]
    inside = {f for f in need if a <= f <= b}
    return inside, set(need) - inside


def fmt_frames(frames):
    """'3816-3839, 4100' for a set of ints."""
    runs = []
    for f in sorted(frames):
        if runs and runs[-1][1] == f - 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return ', '.join(f'{a}-{b}' if b > a else f'{a}' for a, b in runs)


def verify(stem, d, need):
    """Problems (a list of strings) with directory d as stem's source; [] when it is exactly right. A delivered stem is
    judged against its delivery: frames needed beyond it are OWED (split_need), reported by the callers."""
    if not os.path.isdir(d):
        return [f'{stem}: {d} is not a directory']
    need, _ = split_need(stem, need)
    frames, other = survey(d)
    probs = []
    miss = sorted(need - set(frames))
    if miss:
        probs.append(f'{stem}: {len(miss)} of {len(need)} needed frames missing (first {miss[:5]})')
    if stem in DELIVERED:                                         # a delivery is exact: no gap, no stray frame
        a, b = DELIVERED[stem]
        want = set(range(a, b + 1))
        extra = sorted(set(frames) - want)
        if extra:
            probs.append(f'{stem}: {len(extra)} frames outside the delivered {a}-{b} (first {extra[:5]})')
        gaps = sorted(want - set(frames))
        if gaps and not miss:
            probs.append(f'{stem}: {len(gaps)} delivered frames absent (first {gaps[:5]})')
    bad_other = [n for n in other if not ALLOWED_EXTRA.match(n)]
    if bad_other:
        probs.append(f'{stem}: unexpected files {bad_other[:5]}')
    empty = [n for k, n in frames.items() if k in need and os.path.getsize(os.path.join(d, n)) == 0]
    if empty:
        probs.append(f'{stem}: {len(empty)} empty frame files (first {empty[:3]})')
    return probs


def link_state(stem, target):
    """('ok'|'absent'|'elsewhere'|'real', detail) for renders/<stem> against the mapped target."""
    p = os.path.join(RENDERS, stem)
    if os.path.islink(p):
        cur = os.path.realpath(p)
        return ('ok', cur) if cur == os.path.realpath(target) else ('elsewhere', cur)
    if os.path.exists(p):
        return 'real', p
    return 'absent', p


def run(cmd, out_root=None, dry=False):
    need = needed_frames()
    if cmd == 'init':
        if not out_root:
            raise SystemExit('init needs --out-root')
        root = os.path.abspath(os.path.expanduser(out_root))
        path = map_path()
        if os.path.exists(path):
            raise SystemExit(f'{path} exists; edit it by hand or remove it first (never overwritten)')
        stems = {stem: os.path.join(root, stem) for stem in DELIVERED}
        doc = dict(version=1, note='C5 local asset map (gitignored). One explicit directory per stem.', stems=stems)
        if not dry:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as fh:
                json.dump(doc, fh, indent=1)
        print(('would write ' if dry else 'wrote ') + path + f' ({len(stems)} stems under {root})')
        return 0
    stems = load_map()
    if not stems:
        print(f'no asset map at {map_path()}: nothing mapped (run init, or set LD_C5_ASSET_MAP)')
        return 1
    rc, rows = 0, []
    for stem, d in sorted(stems.items()):
        if stem not in need:
            print(f'  SKIP {stem}: the C5 EDL reads nothing from it')
            continue
        probs = verify(stem, d, need[stem])
        _, owed = split_need(stem, need[stem])
        state, detail = link_state(stem, d)
        frames, _ = survey(d) if os.path.isdir(d) else ({}, [])
        size = sum(os.path.getsize(os.path.join(d, n)) for n in frames.values()) if frames else 0
        row = dict(stem=stem, target=d, frames=len(frames), needed=len(need[stem]), bytes=size,
                   first=min(frames) if frames else None, last=max(frames) if frames else None, link=state,
                   problems=probs, owed=fmt_frames(owed) if owed else None)
        if owed:
            a, b = DELIVERED[stem]
            print(f'  OWED {stem}: the EDL reads {fmt_frames(owed)} ({len(owed)} frames) beyond the recorded delivery '
                  f'{a}-{b}; they need a new delivery, whose range then goes in DELIVERED')
        if probs:
            rc = 1
            print(f'  FAIL {stem}: ' + '; '.join(probs))
        elif state == 'real':
            rc = 1
            print(f'  FAIL {stem}: renders/{stem} is a real directory; not replaced (move it aside by hand)')
        elif state == 'elsewhere':
            rc = 1
            print(f'  FAIL {stem}: renders/{stem} already links to {detail}; not replaced')
        elif cmd == 'link' and state == 'absent':
            if not dry:
                os.makedirs(RENDERS, exist_ok=True)
                os.symlink(os.path.realpath(d), os.path.join(RENDERS, stem))
                row['link'] = 'ok'
            print(f"  {'WOULD LINK' if dry else 'LINKED'} renders/{stem} -> {d} ({len(frames)} frames)")
        else:
            print(f'  {state.upper():6} {stem}: {len(frames)} frames, {len(need[stem])} needed')
        rows.append(row)
    unmapped = sorted(set(need) - set(stems))
    print(f'unmapped C5 sources ({len(unmapped)} of {len(need)} stems the EDL reads): {", ".join(unmapped)}')
    if not dry:
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        with open(MANIFEST, 'w') as fh:
            json.dump(dict(map=map_path(), rows=rows, unmapped=unmapped), fh, indent=1)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['init', 'link', 'check'])
    ap.add_argument('--out-root')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    return run(a.cmd, a.out_root, a.dry_run)


if __name__ == '__main__':
    sys.exit(main())
