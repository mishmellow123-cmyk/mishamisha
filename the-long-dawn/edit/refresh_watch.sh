#!/bin/bash
# THE LONG DAWN v3: keep the animatics, the H9 kit and the masters fresh as renders and scores land.
# Run it in the background:  bash edit/refresh_watch.sh      (ONCE=1: exit after one refresh)
#   - polls every 60 s: which take each shot plays, its frames and its folders' mtimes (ALT included) + each cut's
#     sound master and its mtime
#   - a change must hold still for 3 polls (or has waited 30 min: a progress refresh); masters are snapshotted
#     once settled, so a COMPOSER render in progress cannot tear the sound
#   - then, for the cuts that changed: edit/animatic.sh, edit/h9_kit.sh, edit/deliver.sh (incremental masters + QC),
#     each through the render queue, and it goes on watching
#   - it EXITS (waking whoever launched it) only when a step fails, a QC says FAIL, a film's master becomes
#     complete (no slates left), or nothing changes for MAX_H hours (default 6). Log: one block per refresh.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/.venvs/longdawn/env.sh
STATE=edit/cache/refresh_watch.sig
MAX_H="${MAX_H:-6}"
DELIV="$HOME/mishamisha/_local_logs/delivery"
RUN=edit/cache/refresh_watch.run
sig() {
  python3 - <<'EOF'
import os, sys
sys.path.insert(0, 'edit')
import assemble as AS
tab = AS.masters_table()
v3 = os.path.join(AS.ROOT, 'music', 'out', 'v3')
import hashlib
def dmt(d):
    try:
        return os.stat(d).st_mtime_ns
    except OSError:
        return 0
for c in 'ABC':
    h, have = hashlib.sha1(), 0
    for v in ((None, 'codedtowers') if c == 'A' else (None,)):
        for s in AS.EDL.EDL[c]:
            pl = AS.plan_shot(s, c, v)
            have += pl['have'] if v is None and s['kind'] != 'black' else 0
            t = pl['take'] or {}
            dirs = AS.chain(t, c, v) if t and t['mode'] != 'video' else []
            dirs += [os.path.join(AS.RENDERS, x) for x in (t.get('matte'), t.get('add')) if x]
            h.update(repr((s['f0'], pl['kind'], t.get('stem'), pl['have'], pl['alt'], [dmt(d) for d in dirs])).encode())
    alt = h.hexdigest()[:10]                    # which take each shot plays and when its folders last changed
    audio = 'click'
    for p in (tab.get((c, 'score')), os.path.join(v3, f'final_{c}.wav'), tab.get((c, 'fallback')),
              os.path.join(v3, f'fallback_{c}.wav')):
        if p and os.path.isfile(p):
            audio = f'{os.path.basename(p)}@{int(os.path.getmtime(p))}'
            break
    print(f'{c}:{have}+{alt}:{audio}')
EOF
}
complete() {  # the masters whose QC says complete (no slates), one name per line
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
  grep -E '^\*\*|^wrote|warning|RESULT|^\[FAIL\]|H9 kit|Traceback|Error|failed' "$RUN" | cut -c1-170
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
  cuts=$(diff <(echo "$last" | tr ' ' '\n') <(echo "$cur" | tr ' ' '\n') | sed -n 's/^> \([ABC]\):.*/\1/p' | sort -u | tr '\n' ' ')
  echo "== refresh_watch: $(date -u +%H:%MZ) changed: ${cuts}"
  : > "$RUN.all"
  export CUTS="${cuts% }"
  fail=0
  step bash edit/animatic.sh || fail=1
  step bash edit/h9_kit.sh || fail=1
  step bash edit/deliver.sh || fail=1
  unset CUTS
  echo "$cur" > "$STATE"; last="$cur"
  if [ "$fail" = "1" ] || grep -q 'RESULT: FAIL' "$RUN.all"; then
    echo "refresh_watch: a step or a QC FAILED ($(date -u +%H:%MZ)); see $RUN.all"; exit 2
  fi
  done1=$(complete)
  new=$(comm -13 <(echo "$done0") <(echo "$done1") | tr '\n' ' ')
  if [ -n "${new// /}" ]; then echo "refresh_watch: COMPLETE (no slates left): $new($(date -u +%H:%MZ))"; exit 0; fi
  done0="$done1"
  [ "${ONCE:-0}" = "1" ] && exit 0
done
