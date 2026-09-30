#!/bin/bash
# THE LONG DAWN v3: keep the animatics, the H9 kit and the masters fresh as renders and scores land.
# Run it in the background:  bash edit/refresh_watch.sh      (ONCE=1: exit after one refresh)
#   - polls every 60 s: which take each shot plays, its frames and its folders' mtimes (ALT included) + each cut's
#     sound master and its mtime
#   - a change must hold still for 3 polls (or has waited 30 min: a progress refresh); masters are snapshotted
#     once settled, so a COMPOSER render in progress cannot tear the sound
#   - then, for the cuts that changed: edit/animatic.sh, edit/h9_kit.sh, edit/deliver.sh (incremental masters + QC),
#     each through the render queue, and it goes on watching
#   - then the finished stretches go to ~/Downloads/The Long Dawn v3 - PREVIEWS/ (edit/previews.sh)
#   - it EXITS (waking whoever launched it) only when a step fails, a QC says FAIL, a new preview file appears
#     (exit 3), a film's picture sources become complete (no slates, EDIT proxies or provisional sources),
#     or nothing changes for MAX_H hours (default 6). Completion is not creative approval or audio/finish QC.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/.venvs/longdawn/env.sh
STATE=edit/cache/refresh_watch.sig
MAX_H="${MAX_H:-6}"
DELIV="$HOME/mishamisha/_local_logs/delivery"
RUN=edit/cache/refresh_watch.run
sig() {
  python3 - <<'EOF'
import os, re, sys
sys.path.insert(0, 'edit')
import assemble as AS
import hashlib
def dmt(d):
    try:
        return os.stat(d).st_mtime_ns
    except OSError:
        return 0
def source_dirs(t, cut, variant):
    dirs = AS.chain(t, cut, variant) if t and t['mode'] != 'video' else []
    dirs += [AS.render_dir(x) for x in (t.get('matte'), t.get('add'), t.get('linear_mix')) if x]
    spec = t.get('under')
    if (t.get('matte') or t.get('linear_mix')) and spec:
        if spec[0] == 'take':
            under_take, held_frame, source_cut = AS.under_take(spec)
            dirs += source_dirs(under_take, source_cut, None)
        else:
            stem = spec[1]
            dirs += [AS.render_dir(x) for x in (stem, stem + '_half')]
    return dirs
def local_sources(dirs):
    # Native D plates/coefficients may be overwritten in place without changing their folder mtime.
    records = []
    for folder in dirs:
        files = []
        if os.path.basename(folder).startswith('cutd_'):
            try:
                for name in sorted(os.listdir(folder)):
                    if re.fullmatch(r'f_\d+\.(png|jpg|npz)', name):
                        try:
                            st = os.stat(os.path.join(folder, name))
                            files.append((name, st.st_size, st.st_mtime_ns))
                        except OSError:
                            pass  # a landing file will be observed on the next poll
            except OSError:
                pass
        records.append((folder, files))
    return records
for c in os.environ.get('FILMS', 'AC'):                # film B was dropped (user, 28 Sep)
    h, have = hashlib.sha1(), 0
    for v in ((None, 'codedtowers') if c == 'A' else (None,)):
        for s in AS.EDL.EDL[c]:
            pl = AS.plan_shot(s, c, v)
            have += pl['have'] if v is None and s['kind'] != 'black' else 0
            t = pl['take'] or {}
            dirs = source_dirs(t, c, v)
            if c == 'D' and any(os.path.basename(d).startswith('cutd_') for d in dirs):
                h.update(repr(('D source folders', local_sources(dirs))).encode())
            h.update(repr((s['f0'], pl['kind'], t.get('stem'), pl['have'], pl['alt'],
                           AS.EDL.is_final_take(pl['take']), [dmt(d) for d in dirs])).encode())
            if pl['kind'] == 'take':
                for f in range(s['f0'], s['f1']):
                    h.update(repr(AS.provisional_sources(t, c, v, f)).encode())
    for t in AS.EDL.TRANS.get(c, ()):             # EDIT transition windows (EDL.TRANS) and their layer folders
        layers = [AS.render_dir(t[k]) for k in ('glow', 'keep', 'cover') if t.get(k)]
        if c == 'D' and any(os.path.basename(d).startswith('cutd_') for d in layers):
            h.update(repr(('D transition folders', local_sources(layers))).encode())
        h.update(repr((t, [dmt(d) for d in layers])).encode())
    if c == 'A':                                         # lane A-FIX iterates on its comps: a change refreshes A
        h.update(repr(dmt(os.path.join(AS.ROOT, 'edit', 'afix_comp.py'))).encode())
    alt = h.hexdigest()[:10]                    # which take each shot plays and when its folders last changed
    # the sound a master would carry now, through assemble.audio_choice (C: only a file of exactly 5920 frames;
    # the 7,200-frame cut's sound_C / final_C / fallback_C are refused by length, so they can neither be taken nor
    # trigger a refresh)
    audio = AS.audio_signature(c)
    print(f'{c}:{have}+{alt}:{audio}')
EOF
}
complete() {  # QC's picture-source completeness signal, one master name per line
  python3 -c "
import glob, json, os
for p in sorted(glob.glob(os.path.join('$DELIV', '*_QC.json'))):
    try:
        if json.load(open(p)).get('complete'): print(os.path.basename(p)[:-8])
    except Exception: pass"
}
step() {  # run one rebuild step, keep its output for the checks, print the lines that matter
  "$@" > "$RUN" 2>&1; local rc=$?
  cat "$RUN" >> "$RUN.all"
  grep -E '^\*\*|^wrote|warning|RESULT|^\[FAIL\]|H9 kit|Traceback|Error|failed|^NEW:|^previews:|^title_' "$RUN" | cut -c1-170
  return $rc
}
now=$(sig) || exit 1
[ -s "$STATE" ] || echo "$now" > "$STATE"
last=$(cat "$STATE")
done0=$(complete)
while :; do
  t0=$(date +%s); seen=""; stable=0; since=0
  while :; do
    cur=$(sig) || { sleep 60; continue; }
    if [ "$cur" != "$last" ]; then
      [ -z "$seen" ] && since=$(date +%s)
      if [ "$cur" = "$seen" ]; then stable=$((stable + 1)); else stable=0; fi
      seen="$cur"
      if [ "$stable" -ge 3 ] || [ $(( $(date +%s) - since )) -ge 1800 ]; then break; fi
    elif [ $(( $(date +%s) - t0 )) -ge $(( MAX_H * 3600 )) ]; then
      echo "refresh_watch: nothing new in ${MAX_H} h ($(date -u +%H:%MZ))"; exit 0
    fi
    sleep 60
  done
  cuts=$(diff <(echo "$last" | tr ' ' '\n') <(echo "$cur" | tr ' ' '\n') | sed -n 's/^> \([ABCD]\):.*/\1/p' | sort -u | tr '\n' ' ')
  echo "== refresh_watch: $(date -u +%H:%MZ) changed: ${cuts}"
  : > "$RUN.all"
  export CUTS="${cuts% }"
  fail=0
  tc=$(echo "$CUTS" | tr ' ' '\n' | grep -E '^A$' | tr '\n' ' ' || true)
  if [ -n "${tc// /}" ]; then                 # the ember titles (A20, B14) follow their plates' fires
    export TITLE_CUTS="${tc% }"; step bash edit/title_v3.sh || fail=1; unset TITLE_CUTS
  fi
  step bash edit/animatic.sh || fail=1
  step bash edit/h9_kit.sh || fail=1
  step bash edit/deliver.sh || fail=1
  unset CUTS
  step bash edit/previews.sh --cuts "${FILMS:-AC}" || fail=1  # include D when this watcher explicitly tracks D
  echo "$cur" > "$STATE"; last="$cur"
  if [ "$fail" = "1" ] || grep -q 'RESULT: FAIL' "$RUN.all"; then
    echo "refresh_watch: a step or a QC FAILED ($(date -u +%H:%MZ)); see $RUN.all"; exit 2
  fi
  newp=$(grep '^NEW:' "$RUN.all" | sed 's/^NEW: //' | tr '\n' ' ')
  if [ -n "${newp// /}" ]; then echo "refresh_watch: NEW PREVIEW: $newp($(date -u +%H:%MZ))"; exit 3; fi
  done1=$(complete)
  new=$(comm -13 <(echo "$done0") <(echo "$done1") | tr '\n' ' ')
  if [ -n "${new// /}" ]; then echo "refresh_watch: PICTURE SOURCES COMPLETE (no slates, proxies or provisional sources): $new($(date -u +%H:%MZ))"; exit 0; fi
  done0="$done1"
  [ "${ONCE:-0}" = "1" ] && exit 0
done
