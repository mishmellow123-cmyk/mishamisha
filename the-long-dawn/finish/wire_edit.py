"""FINISH -> EDIT: wire the film finish into the edit's delivery chain (idempotent; anchored edits that assert).

    python3 finish/wire_edit.py [edit_dir]        # default: the-long-dawn/edit

assemble.py: `_init(..., finish=None)` wraps the worker's Ctx.picture with the finish (finish/stage.py): the last picture
  stage, after the take (crop, per-cut grade, book matte, add layers) and before titles and burn-ins. No Ctx method
  changes, so deliver's frame-code hash (and every existing segment's key) is untouched.
deliver.py: the master profile gets finish=True. A segment with rendered frames is keyed with the finish identity
  (the look + stage.code_id()); segments that must be encoded anyway (new renders) are finished at once; segments whose
  only change is the finish are the BACKLOG, finished FINISH_BUDGET frames per film per run (default 1200), so the
  one-time cost spreads over the watcher's refreshes. FINISH_ALL=1 clears the backlog in one run. Slates, black and
  EDIT proxies keep their keys (the finish never touches them).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EDIT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), 'edit')


def patch(path, edits):
    s = open(path).read()
    done = 0
    for old, new, marker in edits:
        if marker in s:
            continue
        assert s.count(old) == 1, f'{os.path.basename(path)}: anchor not found once: {old[:70]!r}'
        s = s.replace(old, new)
        done += 1
    if done:
        tmp = path + '.wire.tmp'
        open(tmp, 'w').write(s)
        os.replace(tmp, path)
    print(f'{os.path.basename(path)}: {done} edit(s) applied, {len(edits) - done} already in place')


A_INIT_OLD = '''def _init(cut, variant, scale, clean):
    global _CTX
    _CTX = Ctx(cut, variant, scale, clean)
'''
A_INIT_NEW = '''def _init(cut, variant, scale, clean, finish=None):
    global _CTX
    _CTX = Ctx(cut, variant, scale, clean)
    if finish:                                                # FINISH (finish/stage.py): the masters' film finish
        _CTX.picture = _finishing(_CTX, None if finish is True else finish)


def _finishing(ctx, look=None):
    """FINISH: the last picture stage (finish/stage.py; lane FINISH). It runs after the take frame (crop, per-cut
    grade, book matte, add layers) and before titles and burn-ins. Slates, black and EDIT proxies pass untouched.
    Grain is seeded per (cut, cut frame)."""
    sys.path.insert(0, os.path.join(ROOT, 'finish'))
    import stage
    fin = stage.Finisher(look)
    picture = Ctx.picture.__get__(ctx)

    def finished(f):
        img, shot, status, src = picture(f)
        if src is not None and not status.startswith('SLATE'):
            img = fin(img, src, ctx.cut, f)
        return img, shot, status, src
    return finished
'''

D_PROF_OLD = "    'master': dict(scale=1.0, clean=True, crf=14, preset='slow', out=DELIVERY),\n"
D_PROF_NEW = "    'master': dict(scale=1.0, clean=True, crf=14, preset='slow', out=DELIVERY, finish=True),   # FINISH\n"

D_KEY_OLD = '''def segment_key(cut, variant, prof, i, shot, plan, code, table):
    h = hashlib.sha1()'''
D_KEY_NEW = '''def _finish_id():
    """FINISH: the finish's identity (the look, its code, its baked LUTs) for the master segments' keys."""
    sys.path.insert(0, os.path.join(ROOT, 'finish'))
    import stage
    return f'{stage.LOOK}:{stage.code_id()}'


def segment_key(cut, variant, prof, i, shot, plan, code, table, fin=None):
    h = hashlib.sha1()'''

D_HEAD_OLD = '''            plan['kind'], tdesc, plan['have'], bool(plan['alt'])]
    h.update(json.dumps(head, sort_keys=True, default=str).encode())'''
D_HEAD_NEW = '''            plan['kind'], tdesc, plan['have'], bool(plan['alt'])]
    if fin:                                                   # FINISH: finished segments carry the finish's identity
        head.append(['finish', fin])
    h.update(json.dumps(head, sort_keys=True, default=str).encode())'''

D_BUILD_OLD = '''    code, table = _code_hash(), titles.text_table(cut)
    segs, todo = [], []
    for i, (s, pl) in enumerate(zip(shots, plans)):
        p = os.path.join(cache, segment_key(cut, variant, prof, i, s, pl, code, table) + '.h264')
        segs.append(p)
        if not os.path.exists(p):
            todo.append((i, s, p))
    t0 = time.time()
    nf = sum(s['f1'] - s['f0'] for _, s, _ in todo)
    log(f'{name_of(cut, variant, profile)}: {len(shots)} segments, {len(todo)} to encode ({nf} f)')
    if todo:
        with Pool(workers, initializer=AS._init, initargs=(cut, variant, prof['scale'], prof['clean'])) as pool:'''
D_BUILD_NEW = '''    code, table = _code_hash(), titles.text_table(cut)
    fin = _finish_id() if prof.get('finish') else None       # FINISH (see finish/wire_edit.py)
    budget = None if os.environ.get('FINISH_ALL') == '1' else int(os.environ.get('FINISH_BUDGET', '1200'))
    segs, todo, backlog = [], [], []
    for i, (s, pl) in enumerate(zip(shots, plans)):
        sfin = fin if (pl['kind'] == 'take' and pl['have'] > 0) else None
        p = os.path.join(cache, segment_key(cut, variant, prof, i, s, pl, code, table, sfin) + '.h264')
        if sfin and not os.path.exists(p):
            q = os.path.join(cache, segment_key(cut, variant, prof, i, s, pl, code, table) + '.h264')
            n = s['f1'] - s['f0']
            if os.path.exists(q):                             # only the finish is missing: the backlog, on a budget
                if budget is not None and budget <= 0:          # (overshoots by at most one segment: always progress)
                    backlog.append(n)
                    segs.append(q)
                    continue
                if budget is not None:
                    budget -= n
        segs.append(p)
        if not os.path.exists(p):
            todo.append((i, s, p))
    t0 = time.time()
    nf = sum(s['f1'] - s['f0'] for _, s, _ in todo)
    log(f'{name_of(cut, variant, profile)}: {len(shots)} segments, {len(todo)} to encode ({nf} f)'
        + (f'; finish {fin}, backlog {len(backlog)} segments ({sum(backlog)} f) left' if fin else ''))
    if todo:
        with Pool(workers, initializer=AS._init,
                  initargs=(cut, variant, prof['scale'], prof['clean'], bool(fin))) as pool:'''

D_STATS_OLD = '''    return segs, dict(segments=len(shots), encoded=len(todo), frames_encoded=nf, seconds=round(time.time() - t0, 1))'''
D_STATS_NEW = '''    return segs, dict(segments=len(shots), encoded=len(todo), frames_encoded=nf, seconds=round(time.time() - t0, 1),
                      finish=fin, finish_backlog_segments=len(backlog), finish_backlog_frames=sum(backlog))'''

D_QC_OLD = """    check('INFO', 'build', f"{build['encoded']} of {build['segments']} segments encoded "
                           f"({build['frames_encoded']:,} f) in {build['seconds']:.0f} s")
"""
D_QC_NEW = """    check('INFO', 'build', f"{build['encoded']} of {build['segments']} segments encoded "
                           f"({build['frames_encoded']:,} f) in {build['seconds']:.0f} s")
    if build.get('finish'):                                   # FINISH: the film finish and what is still unfinished
        nb = build.get('finish_backlog_segments', 0)
        check('WARN' if nb else 'INFO', 'finish',
              f"{build['finish']}" + (f"; {nb} segments ({build.get('finish_backlog_frames', 0):,} f) still "
                                      f"unfinished (the budgeted backlog; FINISH_ALL=1 clears it)" if nb else
                                      '; every rendered frame finished'))
"""

if __name__ == '__main__':
    patch(os.path.join(EDIT, 'assemble.py'), [(A_INIT_OLD, A_INIT_NEW, 'def _finishing(ctx, look=None):')])
    patch(os.path.join(EDIT, 'deliver.py'), [(D_PROF_OLD, D_PROF_NEW, 'finish=True),   # FINISH'),
                                              (D_KEY_OLD, D_KEY_NEW, 'def _finish_id():'),
                                              (D_HEAD_OLD, D_HEAD_NEW, "head.append(['finish', fin])"),
                                              (D_BUILD_OLD, D_BUILD_NEW, "FINISH_BUDGET"),
                                              (D_STATS_OLD, D_STATS_NEW, 'finish_backlog_frames='),
                                              (D_QC_OLD, D_QC_NEW, "check('WARN' if nb else 'INFO', 'finish',")])
