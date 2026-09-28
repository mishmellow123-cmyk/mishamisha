#!/bin/bash
# THE LONG DAWN v3: the full-res masters (A, A's ALT, B, C) with QC, incrementally (see edit/deliver.py).
# Runs through the local render queue (renderq.py). Outputs: ~/mishamisha/_local_logs/delivery/
#   {A,B,C}_master.mov (+ .mp4 screener, _QC.txt, _QC.json), A_master_codedtowers.*
# Options (env): CUTS="A C" (default; film B was dropped 28 Sep), WORKERS=3, ALT=0 to skip A's ALT master.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
if [ "${EDIT_Q:-0}" != "1" ]; then           # every local render goes through the render queue (COMMON.md)
  export EDIT_Q=1
  exec python3 "$HOME/mishamisha/_local_logs/renderq.py" -- bash "$SELF" "$@"
fi
source ~/.venvs/longdawn/env.sh
if [ ! -s edit/cache/x1_letters_C_test.mp4 ]; then
  mkdir -p edit/cache
  git -C .. show origin/claude/v3-map:the-long-dawn/review/v3/map_X1_letters_to_fire.mp4 \
    > edit/cache/x1_letters_C_test.mp4 2>/dev/null || rm -f edit/cache/x1_letters_C_test.mp4
fi
for c in ${CUTS:-A C}; do
  python3 edit/deliver.py --cut "$c" --workers "${WORKERS:-3}" | grep -v '^  '
  if [ "$c" = "A" ] && [ "${ALT:-1}" = "1" ] && ls renders/embers_A3_alt_codedtowers/f_* >/dev/null 2>&1; then
    python3 edit/deliver.py --cut A --variant codedtowers --workers "${WORKERS:-3}" | grep -v '^  '
  fi
done
ls -la "$HOME/mishamisha/_local_logs/delivery/" | grep -E 'master'
