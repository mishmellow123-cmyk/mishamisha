"""C5 readiness: THE LAST PAGES C5 (5,920 f, 74 bars) may be mastered only when the generator (edl_v3.py), its
export (edl/edl_C.json), the bar map (music/v3/barmap_C5.json), the captions (titles.C5_TEXT against HANDOVER.md's
"Script v5.2"), the words already baked into the pixels (titles.BAKED_TEXT), the transitions (edl_v3.TRANS['C']) and
every source frame agree.

    python3 edit/c5_readiness.py                  # FULL gate: exit 1 on any FAIL or GAP
    python3 edit/c5_readiness.py --partial        # PARTIAL: FAILs still fail; GAPs (what a labelled partial build may
                                                  #   stand in for) are listed and exit 0
    python3 edit/c5_readiness.py --json out.json  # the report as data (edit/c5_partial.py reads it)

Levels: FAIL breaks both modes (the structure is wrong: tiling, JSON drift, a caption not verbatim or outside its shot,
a flag that would silently drop a caption, a mistyped source). GAP breaks FULL only: a source frame not on this
machine, the open Flint decision, a designed transition not yet built, a provisional source, a re-render owed for
baked words. WARN and INFO never fail. Counts print their denominators. Files are checked for presence and size;
nothing is decoded (the deliveries' own validators decoded and hashed theirs).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

EDIT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EDIT)
sys.path.insert(0, EDIT)
import assemble as AS  # noqa: E402

EDL = AS.EDL
titles = AS.titles
CUT, FPS, BAR = 'C', 24, 80
HANDOVER = os.path.join(ROOT, 'HANDOVER.md')
JSON_PATH = os.path.join(EDIT, 'edl', 'edl_C.json')
BARMAP_DIR = os.path.join(ROOT, 'music', 'v3')

# The paired-rivals amendment (NIGHT_PLAN 29 Sep); HANDOVER's table must already carry it.
AMENDMENT = {12: None, 13: 'So the two furthest ahead lit the first beacons, together.',
             14: 'It was a promise to stop, if all the others would.'}
# script v5.2 row -> the bar-map sections that picture it (row 1 is the black and the red book)
ROW_SECTIONS = {1: ('C1', 'C2'), 2: ('C3',), 3: ('C4',), 4: ('C4',), 5: ('C5',), 6: ('C6',), 7: ('C7',), 8: ('C8',),
                9: ('C9',), **{n: (f'C{n}',) for n in range(10, 24)}}
# The C5 shot map, stated here independently of edl_v3 so a mistyped stem or offset fails: section -> (f0, f1, the
# primary take's stem, off). Delivered shots are absolute (off 0); reused ones read the source frames the 7,200-frame
# cut played (verified 29 Sep against edl_C.json at 69a1788: runC_scroll 0-319, runC_illum 2398-2877, book_C
# 6160-6399 Plenty and 6960-7199 the title).
SHOT_MAP = {
    'C10': (2080, 2320, 'book_C5_refusal', 0), 'C11': (2320, 2640, 'embers_C5_trap', 0),
    'C13': (2880, 3120, 'runC_reveal_pair_v5', 0), 'C14': (3120, 3440, 'runC_scroll', -3120),
    'C15': (3440, 3840, 'map_last_beacon_C', 0), 'C16': (3840, 4000, 'embers_C5_cold', 0),
    'C17': (4000, 4240, 'embers_C5_unfinished', 0), 'C18': (4240, 4480, 'book_C5_deep_abandoned', 0),
    'C19': (4480, 4720, 'runC_watch_v5', 0), 'C20': (4720, 5200, 'runC_illum', 2398 - 4720),
    'C21': (5200, 5440, 'book_C', 960), 'C22': (5440, 5680, 'book_C5_pen', 0), 'C23': (5680, 5920, 'book_C', 1280),
}
# The one movable picture cut, edl_v3.COLD_CUT (the last beacon -> the forges): anywhere from 3792 to the section line
# 3840. Measured 29 Sep from the delivered frames, stated here independently: map_last_beacon_C's last kingdom catches
# 3785-3791, so an earlier cut loses the catch; embers_C5_cold goes dark at 3848 whatever the cut.
COLD_HOLD = (3792, 3840)
# The Pages mattes delivered with C5 are opaque (255 in every pixel of all 240 frames of each, decoded 29 Sep): a
# transition that took its hole from one would reveal nothing.
OPAQUE_MATTES = ('book_C5_refusal_matte', 'book_C5_deep_abandoned_matte', 'book_C5_pen_matte')
FT_CONVENTION = {(841, 1040): dict(add='embers_C3_e15', under=None),             # BURN_NOTES: page + fire, e15
                 (1680, 1718): dict(add=None, under=None),                       # the sweep: race baked, matte 1
                 (1905, 1992): dict(add=None, under=('same', 'embers_C3'))}      # the Eye: the live storm under
# THE GLOW, 1905-1919: book_C_ft_matte is opaque in every pixel of all 15 frames (decoded 29 Sep) and the storm was
# never rendered there (embers_C3 holds 1040-1679 and 1920-2079), so the row holds the storm's first live frame, 1920.
# The composite img + (1 - m) * under is then the same whatever is held. The gate decodes the matte again on every
# run, so a re-delivered matte with any hole in these frames fails here instead of showing a frozen storm.
FT_HELD = {(1905, 1920): ('hold', 'embers_C3', 1920)}


class Report:
    def __init__(self, partial):
        self.partial, self.items, self.rows, self.runs = partial, [], [], []

    def add(self, level, what, msg, **data):
        self.items.append(dict(level=level, what=what, msg=msg, **data))

    def failed(self):
        return any(i['level'] == 'FAIL' or (i['level'] == 'GAP' and not self.partial) for i in self.items)

    def count(self, level):
        return sum(i['level'] == level for i in self.items)


def runs_of(frames):
    """[(a, b)] half-open runs of a sorted iterable of ints."""
    out = []
    for f in frames:
        if out and out[-1][1] == f:
            out[-1][1] = f + 1
        else:
            out.append([f, f + 1])
    return [tuple(r) for r in out]


def fmt_runs(rs, n=6):
    s = ', '.join(f'{a}-{b - 1}' for a, b in rs[:n])
    return s + (f' (+{len(rs) - n} more)' if len(rs) > n else '')


def rows_of(sec):
    return [s for s in EDL.EDL[CUT] if s['sec'] == sec]


def load_barmap():
    with open(EDL.barmap_path(CUT, BARMAP_DIR)) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------ the checks
def check_generator_json(rep):
    """The JSON on disk must be exactly what the generator emits now."""
    doc = AS.edl_doc(CUT)
    try:
        with open(JSON_PATH) as fh:
            disk = json.load(fh)
    except (OSError, ValueError) as e:
        rep.add('FAIL', 'edl_C.json', f'unreadable ({e}); run python3 edit/assemble.py --edl')
        return
    if disk != doc:
        keys = [k for k in sorted(set(doc) | set(disk)) if doc.get(k) != disk.get(k)]
        rep.add('FAIL', 'edl_C.json', f'stale: differs from the generator in {keys}; run python3 edit/assemble.py --edl '
                                      '(never edit the JSON alone)')
    else:
        rep.add('INFO', 'edl_C.json', f"matches the generator ({len(doc['shots'])} rows, {len(doc['text'])} lines, "
                                      f"{len(doc['transitions'])} transition windows)")


def check_grid(rep):
    bm = load_barmap()
    total = EDL.TOTAL[CUT]
    ok = (total == 5920 and bm['frames'] == total and bm.get('bars') == 74 and bm['bars'] * BAR == total
          and bm.get('fps') == FPS and bm.get('bpm') == 72)
    rep.add('INFO' if ok else 'FAIL', 'grid', f"EDL {total} f; bar map {bm['frames']} f = {bm.get('bars')} bars x {BAR} "
                                              f"at {bm.get('fps')} fps, {bm.get('bpm')} BPM (C5: 5920 = 74 x 80, 24, 72)")
    secs = bm['sections']
    f, bad = 0, []
    for s in secs:
        if s['f0'] != f or s['f1'] <= s['f0']:
            bad.append(f"{s['id']} {s['f0']}-{s['f1']} (expected to start at {f})")
        if s['f0'] % BAR or s['f1'] % BAR:
            bad.append(f"{s['id']} {s['f0']}-{s['f1']} is off the bar grid")
        f = s['f1']
    if f != total:
        bad.append(f'sections end at {f}, not {total}')
    rep.add('FAIL' if bad else 'INFO', 'bar map sections',
            '; '.join(bad) if bad else f'{len(secs)} sections tile 0-{total - 1} on bar lines')


def check_tiling(rep):
    shots = EDL.EDL[CUT]
    total = EDL.TOTAL[CUT]
    cover = np.zeros(total + 1, np.int32)
    bad = []
    for s in shots:
        if not (0 <= s['f0'] < s['f1'] <= total):
            bad.append(f"{s['sec']} {s['f0']}-{s['f1']} outside 0-{total}")
            continue
        cover[s['f0']:s['f1']] += 1
    gaps = runs_of(np.flatnonzero(cover[:total] == 0).tolist())
    over = runs_of(np.flatnonzero(cover[:total] > 1).tolist())
    if gaps:
        bad.append(f'gaps at {fmt_runs(gaps)}')
    if over:
        bad.append(f'overlaps at {fmt_runs(over)}')
    try:
        EDL.check(BARMAP_DIR)
    except AssertionError as e:
        bad.append(f'edl_v3.check: {e}')
    n = int((cover[:total] == 1).sum())
    rep.add('FAIL' if bad else 'INFO', 'coverage',
            '; '.join(bad) if bad else f'{len(shots)} rows cover {n} of {total} frames exactly once (no gap, no overlap)')
    off = sorted({s['f0'] for s in shots if s['f0'] % BAR} | {s['f1'] for s in shots if s['f1'] % BAR})
    rep.add('INFO', 'sub-row boundaries', f'{len(off)} row boundaries fall between bar lines, each inside a section '
                                          f'(the bar map fixes only the section lines): {off}')


def expected_rows(sec):
    """[(f0, f1, stem, off)] a section must hold: SHOT_MAP's one row, except C15 once COLD_CUT moves before 3840
    (the map, then Cold's lit lead-in at its own C5 numbers)."""
    f0, f1, stem, off = SHOT_MAP[sec]
    cc = EDL.COLD_CUT
    if sec == 'C15' and cc != f1:
        return [(f0, cc, stem, off), (cc, f1, 'embers_C5_cold', 0)]
    return [(f0, f1, stem, off)]


def check_shot_map(rep):
    bad = []
    cc = EDL.COLD_CUT
    if not COLD_HOLD[0] <= cc <= COLD_HOLD[1]:
        bad.append(f"edl_v3.COLD_CUT {cc} is outside {COLD_HOLD[0]}-{COLD_HOLD[1]}: before {COLD_HOLD[0]} the cut loses "
                   f"the last kingdom's catch (3785-3791), and {COLD_HOLD[1]} is the bar map's section line")
    for sec in SHOT_MAP:
        want = expected_rows(sec)
        rs = rows_of(sec)
        if [(r['f0'], r['f1']) for r in rs] != [(a, b) for a, b, _, _ in want]:
            bad.append(f'{sec}: rows {[(r["f0"], r["f1"]) for r in rs]}, expected {[(a, b) for a, b, _, _ in want]}'
                       + (f' (edl_v3.COLD_CUT = {cc})' if sec == 'C15' else ''))
            continue
        for r, (_, _, stem, off) in zip(rs, want):
            t = r['takes'][0] if r['takes'] else None
            if not t or (t['stem'], t['off'], t['mode']) != (stem, off, 'exact'):
                bad.append(f"{sec} {r['f0']}-{r['f1']}: primary take "
                           f"{None if not t else (t['stem'], t['off'], t['mode'])}, expected ({stem}, {off}, 'exact')")
            if len(r['takes']) != 1:
                bad.append(f"{sec} {r['f0']}-{r['f1']}: {len(r['takes'])} takes; a C5 shot has exactly one source, "
                           'no fallback')
    for s in EDL.EDL[CUT]:
        if s['f0'] >= 2080:
            for t in s['takes']:
                if t['mode'] != 'exact':
                    bad.append(f"{s['sec']} {s['f0']}-{s['f1']}: take {t['stem']} mode {t['mode']!r} (C5 reads exact "
                               'folder names only)')
    # the first half's filmed burns: exact ranges, their own comp conventions, never the superseded book_C burn
    for s in EDL.EDL[CUT]:
        for t in s['takes']:
            if t['stem'] == 'book_C_ft':
                rng = next((r for r in FT_CONVENTION if r[0] <= s['f0'] and s['f1'] <= r[1]), None)
                if rng is None:
                    bad.append(f"{s['sec']} {s['f0']}-{s['f1']}: book_C_ft outside the filmed ranges {list(EDL.FT_RANGES)}")
                    continue
                want = FT_CONVENTION[rng]
                got_under = tuple(t['under']) if t.get('under') else None
                want = dict(want, under=FT_HELD.get((s['f0'], s['f1']), want['under']))
                if t.get('matte') != 'book_C_ft_matte' or t.get('add') != want['add'] or got_under != want['under']:
                    bad.append(f"{s['sec']} {s['f0']}-{s['f1']}: book_C_ft comp (matte {t.get('matte')}, add "
                               f"{t.get('add')}, under {got_under}) is not BURN_NOTES' {want} for {rng[0]}-{rng[1] - 1}")
            if t['stem'] == 'book_C' and t['off'] == 0:
                clash = [r for r in EDL.FT_RANGES if s['f0'] < r[1] and r[0] < s['f1']]
                if clash:
                    bad.append(f"{s['sec']} {s['f0']}-{s['f1']}: the superseded book_C burn can play inside the filmed "
                               f'range {clash}')
    if tuple(EDL.FT_RANGES) != tuple(FT_CONVENTION):
        bad.append(f'edl_v3.FT_RANGES {EDL.FT_RANGES} differ from BURN_NOTES {list(FT_CONVENTION)}')
    rep.add('FAIL' if bad else 'INFO', 'shot map',
            '; '.join(bad) if bad else f'{len(SHOT_MAP)} C5 shots map to their exact sources and offsets (the map -> '
                                       f'Cold cut at edl_v3.COLD_CUT = {cc}); the filmed '
                                       f'burns ({", ".join(f"{a}-{b - 1}" for a, b in EDL.FT_RANGES)}) keep their '
                                       'BURN_NOTES comps and no superseded book_C burn can play there')


def _matte_holes(paths):
    """The frames (keys of {frame: path}) whose matte lets any of the under-layer through: a pixel below 255, or a file
    that does not decode (unverifiable counts as a hole)."""
    import cv2
    holes = []
    for f, p in sorted(paths.items()):
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None or (m < 255).any():
            holes.append(f)
    return holes


def check_held_under(rep):
    """FT_HELD's rows hold an under-layer frame, which is sound only while the matte hides it completely."""
    bad, n = [], 0
    for (a, b), under in FT_HELD.items():
        rows = [s for s in EDL.EDL[CUT] if (s['f0'], s['f1']) == (a, b)]
        takes = [t for s in rows for t in s['takes'] if t.get('under') and tuple(t['under']) == under]
        if not takes:
            continue                                          # no row holds this under: check_shot_map reports it
        have = AS.index(os.path.join(AS.RENDERS, takes[0]['matte']))
        missing = [f for f in range(a, b) if f not in have]
        if missing:
            rep.add('GAP', f'held under {a}-{b - 1}', f"{takes[0]['matte']} lacks {fmt_runs(runs_of(missing))}: "
                    'cannot confirm the held storm is hidden')
            continue
        holes = _matte_holes({f: have[f] for f in range(a, b)})
        n += b - a
        if holes:
            bad.append(f"{a}-{b - 1} holds {under[1]} {under[2]}, but {takes[0]['matte']} lets it through at "
                       f'{fmt_runs(runs_of(holes))}: render {under[1]} there or cut the hold')
    rep.add('FAIL' if bad else 'INFO', 'held under', '; '.join(bad) if bad else
            f'{n} frames hold an under-layer frame behind a matte decoded opaque in every pixel: the hold cannot show')


def _size(path):
    """A file's size, or -1 when it cannot be read (a dangling link, a vanished file)."""
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def _frame_ok(take, f):
    """(ok, provisional sources, missing layer names) for cut frame f through this take and its layers."""
    p, _ = AS.locate(take, CUT, None, f)
    if not p:
        return False, (), [take['stem']] + ([take['add']] if take.get('add') else [])
    miss = []
    src = f + take['off']
    if isinstance(p, str) and _size(p) <= 0:
        miss.append(take['stem'] + (' (empty file)' if _size(p) == 0 else ' (unreadable)'))
    if take.get('matte'):
        mp = AS.index(os.path.join(AS.RENDERS, take['matte'])).get(src)
        if not mp or _size(mp) <= 0:
            miss.append(take['matte'])
        elif take.get('under'):
            up, _ = AS.locate_under(take['under'], f)
            if not up:
                miss.append('under:' + take['under'][1])
    return not miss, AS.provisional_sources(take, CUT, None, f), miss


def check_sources(rep):
    """Every frame of every row: the primary take and all its layers present; provisional/fallback sources flagged."""
    total_present, total_needed = 0, 0
    for s in EDL.EDL[CUT]:
        n = s['f1'] - s['f0']
        row = dict(sec=s['sec'], name=s['name'], f0=s['f0'], f1=s['f1'], kind=s['kind'])
        if s['kind'] == 'black':
            row.update(status='black', present=n)
            rep.rows.append(row)
            total_present += n
            total_needed += n
            continue
        if s['kind'] == 'decision':
            row.update(status='decision', present=0, missing=[(s['f0'], s['f1'])])
            rep.rows.append(row)
            total_needed += n
            continue
        prim = s['takes'][0]
        plan = AS.plan_shot(s, CUT, None)
        ok, bad, prov_frames, layers = [], [], [], set()
        for f in range(s['f0'], s['f1']):
            good, prov, miss = _frame_ok(prim, f)
            if good and not prov:
                ok.append(f)
            else:
                bad.append(f)
                layers.update(miss)
                if good and prov:
                    prov_frames.append(f)
        total_needed += n
        total_present += len(ok)
        chosen = plan['take']['stem'] if plan['kind'] == 'take' else None
        row.update(primary=prim['stem'], off=prim['off'], present=len(ok), missing=runs_of(bad),
                   plays=chosen if chosen != prim['stem'] else None)
        if not bad:
            row['status'] = 'present'
        else:
            row['status'] = 'missing' if not ok else 'partial'
            what = f"{s['sec']} {s['name']} {s['f0']}-{s['f1'] - 1}"
            src = f"{prim['stem']} {s['f0'] + prim['off']}-{s['f1'] - 1 + prim['off']}"
            msg = (f'{len(bad)} of {n} frames not resolvable from {src}' +
                   (f' (missing: {", ".join(sorted(layers))})' if layers else '') +
                   (f'; {len(prov_frames)} use provisional sources' if prov_frames else '') +
                   (f'; would play fallback {chosen} (provisional)' if chosen and chosen != prim['stem'] else ''))
            rep.add('GAP', what, msg, frames=len(bad))
        rep.rows.append(row)
    rep.add('INFO', 'sources', f'{total_present} of {total_needed} frames resolvable from their primary sources on this '
                               f'machine (black included; presence and size only)')


def check_flint(rep):
    fl, cands, choice = EDL.FLINT, EDL.FLINT_CANDIDATES, EDL.FLINT_CHOICE
    a0, b0 = fl['slot']
    bad = []
    for k, c in cands.items():                                 # every written candidate must be a legal selection
        f = a0
        for a, b, s0 in c['pieces']:
            if a != f or b <= a:
                bad.append(f'candidate {k}: piece {a}-{b} does not follow {f}')
            f = b
            if s0 is not None:
                s1 = s0 + (b - a)
                if not any(ta <= s0 and s1 <= tb for ta, tb in fl['takes'].values()):
                    bad.append(f'candidate {k}: source {s0}-{s1 - 1} is not inside one flint take {fl["takes"]}')
                if s0 < fl['retired'][1] and fl['retired'][0] < s1:
                    bad.append(f'candidate {k}: source {s0}-{s1 - 1} reads the RETIRED find/vision '
                               f'{fl["retired"][0]}-{fl["retired"][1] - 1}')
        if f != b0:
            bad.append(f'candidate {k}: pieces end at {f}, not {b0}')
    rows = rows_of('C12')
    if choice is None:
        if [(r['f0'], r['f1'], r['kind']) for r in rows] != [(a0, b0, 'decision')]:
            bad.append(f'C12 rows {[(r["f0"], r["f1"], r["kind"]) for r in rows]} are not the single decision row')
        summary = '; '.join(f"{k}: " + ' + '.join((f'black {a}-{b - 1}' if s0 is None else
                                                  f'ring_C {s0}-{s0 + b - a - 1} -> {a}-{b - 1}')
                                                 for a, b, s0 in c['pieces'])
                            + f" (catch {EDL.flint_events(k)['catch']})" for k, c in cands.items())
        rep.add('GAP', 'C12 FLINT 2640-2879: decision_required',
                'nobody has chosen the 240 frames from the 250 of ring_C (flint_a 2960-2999 + flint_b 3150-3359) '
                'and no still of either take exists on this machine; set edl_v3.FLINT_CHOICE after looking at the '
                f'frames. Written candidates: {summary}', decision='C12_FLINT')
    else:
        if choice not in cands:
            bad.append(f'FLINT_CHOICE {choice!r} is not a written candidate')
        else:
            want = [(a, b, s0) for a, b, s0 in cands[choice]['pieces']]
            got = [(r['f0'], r['f1'], None if r['kind'] == 'black' else r['f0'] + r['takes'][0]['off']) for r in rows]
            if got != want:
                bad.append(f'C12 rows {got} do not realise candidate {choice} {want}')
            else:
                rep.add('INFO', 'C12 FLINT', f'candidate {choice} chosen: {want}')
    if bad:
        rep.add('FAIL', 'C12 FLINT selection', '; '.join(bad))


def script_v52(path=HANDOVER):
    """{row: words or None} from HANDOVER.md's '### Script v5.2' table (trailing italic notes dropped)."""
    with open(path) as fh:
        txt = fh.read()
    i = txt.index('### Script v5.2')
    j = txt.find('\n### ', i + 5)
    rows = {}
    for line in txt[i:j if j > 0 else len(txt)].splitlines():
        m = re.match(r'\|\s*(\d+)\s*\|[^|]*\|(.*)\|\s*$', line)
        if m:
            words = re.sub(r'\s*\*\([^)]*\)\*\s*$', '', m.group(2).strip()).strip()
            rows[int(m.group(1))] = None if words.lower() == 'none' else words
    return rows


def hard_cuts():
    """{frame: (source before, source after)}: row boundaries where the picture's source changes with no transition
    window over them. The filmed-burn edges are left out (BURN_NOTES: there the burn continues the same page)."""
    shots = EDL.EDL[CUT]
    ft_edges = {x for r in EDL.FT_RANGES for x in r}
    out = {}
    for s, n in zip(shots, shots[1:]):
        c = n['f0']
        a = s['takes'][0]['stem'] if s['takes'] else s['kind']
        b = n['takes'][0]['stem'] if n['takes'] else n['kind']
        if a != b and c not in ft_edges and not any(t['f0'] <= c < t['f1'] for t in EDL.TRANS[CUT]):
            out[c] = (a, b)
    return out


def check_captions(rep):
    table = titles.text_table(CUT)
    try:
        script = script_v52()
    except (OSError, ValueError) as e:
        rep.add('FAIL', 'captions', f'cannot read the script table in {HANDOVER}: {e}')
        return
    bad, warn = [], []
    if sorted(script) != list(range(1, 24)):
        bad.append(f'HANDOVER.md v5.2 table rows {sorted(script)}, expected 1-23')
    for n, words in AMENDMENT.items():
        if script.get(n) != words:
            bad.append(f'HANDOVER.md row {n} {script.get(n)!r} is not the paired-rivals amendment {words!r}')
    by_row = {}
    for r in table:
        by_row.setdefault(r['row'], []).append(r)
    for n, words in sorted(script.items()):
        mine = by_row.get(n, [])
        if words is None and mine:
            bad.append(f'row {n} has no words in the script but captions {[r["id"] for r in mine]}')
        elif words is not None and len(mine) != 1:
            bad.append(f'row {n} {words!r}: {len(mine)} captions (expected 1)')
        elif words is not None and mine[0]['line'] != words:
            bad.append(f'row {n} {mine[0]["id"]}: {mine[0]["line"]!r} is not verbatim {words!r}')
    secs = {s['id']: s for s in load_barmap()['sections']}
    wins = [t for t in EDL.TRANS[CUT]]
    cuts = hard_cuts()
    for r in sorted(table, key=lambda r: r['f_in']):
        if not 0 <= r['f_in'] < r['f_out'] <= EDL.TOTAL[CUT]:
            bad.append(f"{r['id']} {r['f_in']}-{r['f_out']} outside the cut")
            continue
        home = [secs[k] for k in ROW_SECTIONS.get(r['row'], ()) if k in secs]
        inside = [s for s in home if s['f0'] <= r['f_in'] and r['f_out'] <= s['f1']]
        if not inside:
            bad.append(f"{r['id']} {r['f_in']}-{r['f_out'] - 1} is not inside one of its row's shots "
                       f"{[(s['id'], s['f0'], s['f1'] - 1) for s in home]}")
        if r['set'] not in ('fire', 'ink', 'in_picture'):
            bad.append(f"{r['id']}: set {r['set']!r} (C5 uses fire, ink, in_picture)")
        if not (0 <= r.get('x', 960) < 1920 and 0 <= r.get('y', 402) < 804):
            bad.append(f"{r['id']}: placement {r.get('x')}, {r.get('y')} outside the 1920x804 picture")
        if r.get('lines') and ' '.join(r['lines']) != r['line']:
            bad.append(f"{r['id']}: its line breaks {list(r['lines'])} do not rejoin to the verbatim line")
        elif r['set'] in ('ink', 'fire'):                        # EDIT draws it: every glyph must land in the picture
            ln = titles.TextV3(CUT, r, 1.0)
            ys, xs = np.nonzero(ln.alpha > 0.5)
            ext = (ln.x0 + xs.min(), ln.y0 + ys.min(), ln.x0 + xs.max(), ln.y0 + ys.max())
            if ext[0] < 0 or ext[1] < 0 or ext[2] >= 1920 or ext[3] >= 804:
                bad.append(f"{r['id']}: its glyphs span x {ext[0]}-{ext[2]}, y {ext[1]}-{ext[3]}, outside the "
                           '1920x804 picture')
        for t in wins:
            if t['f0'] < r['f_out'] and r['f_in'] < t['f1']:
                warn.append(f"{r['id']} overlaps the {t['kind']} window {t['f0']}-{t['f1'] - 1}")
        for c, (a, b) in sorted(cuts.items()):
            if r['f_in'] < c < r['f_out']:
                warn.append(f"{r['id']} {r['f_in']}-{r['f_out'] - 1} runs across the hard cut at {c} ({a} -> {b}): "
                            f'its words change picture mid-line; end it by {c}, or keep it across on purpose')
    srt = sorted(table, key=lambda r: r['f_in'])
    for a, b in zip(srt, srt[1:]):
        if b['f_in'] < a['f_out']:
            bad.append(f"{a['id']} ({a['f_in']}-{a['f_out'] - 1}) and {b['id']} ({b['f_in']}-{b['f_out'] - 1}) overlap")
    n_words = sum(w is not None for w in script.values())
    rep.add('FAIL' if bad else 'INFO', 'captions', '; '.join(bad) if bad else
            f'{len(table)} lines = the {n_words} worded rows of HANDOVER v5.2, verbatim, each inside its own shot, none '
            'overlapping (PROVISIONAL words)')
    for w in warn:
        rep.add('WARN', 'captions', w)


def _take_at(f):
    s = next(s for s in EDL.EDL[CUT] if s['f0'] <= f < s['f1'])
    return s, (s['takes'][0] if s['takes'] else None)


def _baked(stem, src):
    return [b for b in titles.BAKED_TEXT if b['stem'] == stem and b['src'][0] <= src < b['src'][1]]


def check_baked(rep):
    """in_picture lines must match the words in the pixels; EDIT's lines must not sit on baked words; baked words the
    cut shows must belong to an in_picture line."""
    table = titles.text_table(CUT)
    fails, gaps = [], []
    for r in table:
        frames = range(r['f_in'], r['f_out'])
        seen = {}
        for f in frames:
            s, t = _take_at(f)
            if t is None or s['kind'] == 'black':
                seen.setdefault(('none', None, None), []).append(f)
                continue
            src = f + t['off']
            recs = _baked(t['stem'], src)
            key = (t['stem'], recs[0]['line'] if recs else None, recs[0]['by'] if recs else None)
            seen.setdefault(key, []).append(f)
        for (stem, words, by), fs in seen.items():
            rs = fmt_runs(runs_of(fs))
            if r['set'] == 'in_picture':
                if words is None:
                    fails.append(f"{r['id']} {r['line']!r} is in_picture, but {stem} bakes no words at C5 {rs}: EDIT "
                                 'would draw nothing and the line would silently vanish (set it ink/fire, or bake it)')
                elif words != r['line']:
                    s, t = _take_at(fs[0])
                    gaps.append(f"{r['id']} is in_picture but {stem} {fs[0] + t['off']}-{fs[-1] + t['off']} bakes "
                                f"{words!r} (by {by}), not the v5.2 {r['line']!r}: RE-RENDER {stem} "
                                f"{fs[0] + t['off']}-{fs[-1] + t['off']} with the v5.2 words (MAP/PAGES), or clear them "
                                f"from the page and set {r['id']} to ink")
            elif words is not None:
                s, t = _take_at(fs[0])
                gaps.append(f"EDIT draws {r['id']} over words already baked in {stem} "
                            f"{fs[0] + t['off']}-{fs[-1] + t['off']} ({words!r}): double text; re-render without them")
    owned = {f for r in table if r['set'] == 'in_picture' for f in range(r['f_in'], r['f_out'])}
    stale = {}                                                # baked words on frames no in_picture line claims
    for f in range(EDL.TOTAL[CUT]):
        s, t = _take_at(f)
        if t is None or s['kind'] in ('black', 'decision') or f in owned:
            continue
        for b in _baked(t['stem'], f + t['off']):
            stale.setdefault((b['line'], t['stem']), []).append(f)
    for (words, stem), fs in stale.items():
        fails.append(f'the cut shows words baked in {stem} ({words!r}) at C5 {fmt_runs(runs_of(fs))} that no '
                     'in_picture line owns (a wrong source range, or a line missing from C5_TEXT)')
    for m in fails:
        rep.add('FAIL', 'baked text', m)
    for m in gaps:
        rep.add('GAP', 'baked text', m)
    if not fails and not gaps:
        rep.add('INFO', 'baked text', 'every in_picture line matches its pixels; no EDIT line sits on baked words')
    # the registry records the PIXELS; say so when the baking source has moved on (a re-render may be owed or done)
    try:
        with open(os.path.join(ROOT, 'shots', 'map', 'inkline.py')) as fh:
            src = fh.read()
        for key in ('T1', 'T14'):
            m = re.search(rf"'{key}': \('([^']*)', (\d+), (\d+)", src)
            rec = next((b for b in titles.BAKED_TEXT if f"LINES['{key}']" in b['by']), None)
            if m and rec and (m.group(1), int(m.group(2)), int(m.group(3))) != (rec['line'], *rec['src']):
                rep.add('WARN', 'baked text', f"inkline.py LINES['{key}'] now reads {m.group(1)!r} "
                                              f'{m.group(2)}-{m.group(3)}; BAKED_TEXT (the pixels) says {rec["line"]!r}: '
                                              'update the registry only after the frames are re-rendered and looked at')
    except OSError:
        rep.add('WARN', 'baked text', 'shots/map/inkline.py not readable: the registry was not cross-checked')


def check_transitions(rep):
    shots = EDL.EDL[CUT]
    bounds = {s['f0']: s for s in shots}
    kinds = set(AS.TKINDS_PAIR) | set(AS.TKINDS_SHOT) | set(AS.AFIX.KINDS)
    bad, built, planned = [], 0, 0
    wins = sorted(EDL.TRANS[CUT], key=lambda t: t['f0'])
    for a, b in zip(wins, wins[1:]):
        if b['f0'] < a['f1']:
            bad.append(f"windows {a['f0']}-{a['f1'] - 1} and {b['f0']}-{b['f1'] - 1} overlap")
    for t in wins:
        span = f"{t['kind']} {t['f0']}-{t['f1'] - 1}"
        if not 0 <= t['f0'] < t['f1'] <= EDL.TOTAL[CUT]:
            bad.append(f'{span}: outside the cut')
            continue
        if t['kind'] not in kinds:
            bad.append(f'{span}: unknown kind')
        if t['kind'] in AS.TKINDS_SHOT:
            s0 = next(s for s in shots if s['f0'] <= t['f0'] < s['f1'])
            s1 = next(s for s in shots if s['f0'] <= t['f1'] - 1 < s['f1'])
            if s0['sec'] != s1['sec']:
                bad.append(f'{span}: a one-shot kind across {s0["sec"]}|{s1["sec"]}')
        else:
            c = t.get('cut')
            prev = next((s for s in shots if c is not None and s['f0'] <= c - 1 < s['f1']), None)
            if (c is None or not t['f0'] <= c <= t['f1'] or c not in bounds or prev is None
                    or prev['sec'] == bounds[c]['sec']):
                bad.append(f'{span}: cut {c} is not a boundary between two shots inside the window')
        opaque = [t[k] for k in ('glow', 'keep', 'cover') if t.get(k) in OPAQUE_MATTES]
        if opaque:
            bad.append(f'{span}: takes a layer from {opaque}, a delivered page matte that is opaque on every frame (no '
                       'hole to burn or turn through); build the hole in the edit or render its own layers')
        if not t.get('ready', True):
            planned += 1
            rep.add('GAP', f'transition {span}', 'DESIGNED, NOT BUILT (plays as a hard cut): ' + t.get('note', ''))
            continue
        missing = {}
        for k in ('glow', 'keep', 'cover'):
            if t.get(k):
                have = AS.index(os.path.join(AS.RENDERS, t[k]))
                miss = [f for f in range(t['f0'], t['f1']) if f not in have]
                if miss:
                    missing[t[k]] = len(miss)
        if missing:
            rep.add('GAP', f'transition {span}', f'layer frames missing {missing} of {t["f1"] - t["f0"]}: plays as a '
                                                 'hard cut')
        else:
            built += 1
    rep.add('FAIL' if bad else 'INFO', 'transitions', '; '.join(bad) if bad else
            f'{len(wins)} windows: {built} ready (none waits on layer frames; each still needs both sides\' frames), '
            f'{planned} designed but not built')


def check_assets(rep):
    """Delivered folders must be exact (their contract); say where each resolves on this machine."""
    sys.path.insert(0, os.path.join(EDIT, 'tools'))
    import c5_assets as CA
    need = CA.needed_frames(CUT)
    for stem, (a, b) in CA.DELIVERED.items():
        _, owed = CA.split_need(stem, need.get(stem, set()))
        if owed:
            rep.add('GAP', f'asset {stem}', f'the EDL reads {CA.fmt_frames(owed)} ({len(owed)} frames) beyond the '
                                            f'recorded delivery {a}-{b}: OWED by a new delivery (then record its range '
                                            'in tools/c5_assets.DELIVERED)', frames=len(owed))
        p = os.path.join(AS.RENDERS, stem)
        if not os.path.isdir(p):
            continue
        probs = CA.verify(stem, p, need.get(stem, set()))
        where = os.path.realpath(p) if os.path.islink(p) else 'renders/' + stem
        if probs:
            rep.add('FAIL', f'asset {stem}', '; '.join(probs))
        else:
            rep.add('INFO', f'asset {stem}', f'{b - a + 1} frames {a}-{b} exact, via {"symlink" if os.path.islink(p) else "folder"}'
                                             f' ({"mapped" if os.path.islink(p) else "local"}: {os.path.basename(where)})')


def check_audio(rep):
    path, label, _ = AS.audio_choice(CUT)
    rep.add('WARN' if path is None else 'INFO', 'sound (separate gate)',
            'no C sound file of exactly 5920 frames: a master would carry the click track (sound_C.wav and its '
            'siblings belong to the retired 7,200-frame cut and are refused by length)' if path is None else
            f'{label}: {os.path.relpath(path, ROOT)}')


def frame_runs():
    """[(a, b, section, name, status)] runs over the whole cut: present / missing / black / decision."""
    out = []
    for s in EDL.EDL[CUT]:
        if s['kind'] in ('black', 'decision'):
            out.append((s['f0'], s['f1'], s['sec'], s['name'], s['kind']))
            continue
        prim = s['takes'][0]
        st = []
        for f in range(s['f0'], s['f1']):
            good, prov, _ = _frame_ok(prim, f)
            st.append('present' if good and not prov else 'missing')
        a = s['f0']
        for k in range(1, len(st) + 1):
            if k == len(st) or st[k] != st[k - 1]:
                out.append((a, s['f0'] + k, s['sec'], s['name'], st[k - 1]))
                a = s['f0'] + k
    return out


CHECKS = (check_generator_json, check_grid, check_tiling, check_shot_map, check_sources, check_flint, check_captions,
          check_baked, check_transitions, check_assets, check_audio, check_held_under)


def run(partial=False):
    rep = Report(partial)
    for chk in CHECKS:
        try:
            chk(rep)
        except Exception as e:                                # a check that cannot run is itself a failure
            rep.add('FAIL', chk.__name__, f'{type(e).__name__}: {e}')
    rep.runs = frame_runs()
    return rep


def render_text(rep):
    lines = [f"C5 READINESS ({'PARTIAL' if rep.partial else 'FULL'} mode): "
             f"{'FAILED' if rep.failed() else ('PARTIAL OK (not a final gate)' if rep.partial else 'READY')}  "
             f"FAIL {rep.count('FAIL')}  GAP {rep.count('GAP')}  WARN {rep.count('WARN')}  INFO {rep.count('INFO')}"]
    for lvl in ('FAIL', 'GAP', 'WARN', 'INFO'):
        for i in rep.items:
            if i['level'] == lvl:
                lines.append(f"[{lvl}] {i['what']}: {i['msg']}")
    present = sum(b - a for a, b, _, _, st in rep.runs if st in ('present', 'black'))
    lines.append(f'frames: {present} of {EDL.TOTAL[CUT]} playable from real sources (black included); '
                 f'{EDL.TOTAL[CUT] - present} would be slates')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--partial', action='store_true', help='allow GAPs (a labelled partial build); FAILs still fail')
    ap.add_argument('--json', help='write the report as JSON here')
    a = ap.parse_args(argv)
    rep = run(a.partial)
    print(render_text(rep))
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(dict(mode='partial' if a.partial else 'full', failed=rep.failed(), items=rep.items,
                           rows=rep.rows, runs=rep.runs), fh, indent=1, default=list)
    return 1 if rep.failed() else 0


if __name__ == '__main__':
    sys.exit(main())
