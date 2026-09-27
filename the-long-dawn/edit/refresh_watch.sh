#!/bin/bash
# THE LONG DAWN v3: wait for new picture or a new master, let it settle, rebuild what changed, then exit.
# Run it in the background (one refresh per run; re-arm after each):  bash edit/refresh_watch.sh
#   - polls every 60 s: frames found per cut (the EDL's own lookup, ALT included) + each cut's audio file and mtime
#   - a change must hold still for 3 polls (or has waited 30 min: a progress refresh); masters are snapshotted
#     once settled, so a COMPOSER render in progress cannot tear the sound
#   - then, for the cuts that changed: edit/animatic.sh, edit/h9_kit.sh, edit/deliver.sh (incremental masters + QC),
#     each through the render queue; exits (MAX_H hours idle: exits too)
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/.venvs/longdawn/env.sh
STATE=edit/cache/refresh_watch.sig
MAX_H="${MAX_H:-4}"
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
now=$(sig) || exit 1
[ -s "$STATE" ] || echo "$now" > "$STATE"
last=$(cat "$STATE")
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
echo "refresh_watch: $(date -u +%H:%MZ) changed: ${cuts}| was: $(echo $last) | now: $(echo $cur)"
CUTS="${cuts% }" bash edit/animatic.sh | grep -E '^\*\*|wrote|warning' | cut -c1-160
bash edit/h9_kit.sh | tail -1
CUTS="${cuts% }" bash edit/deliver.sh | grep -E '^RESULT|^\[FAIL\]|^wrote|encode' | cut -c1-160
echo "$cur" > "$STATE"
