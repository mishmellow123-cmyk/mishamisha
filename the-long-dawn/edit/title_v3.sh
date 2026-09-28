#!/bin/bash
# THE LONG DAWN v3: (re)render the ember titles of A (A20) and B (B14) when their inputs change: the fires in the
# plate (or the stand-ins), the clock, the engine (edit/ember_title_v3.py; up to date = a few seconds).
# Runs through the local render queue. Options (env): TITLE_CUTS="A" (default; B was dropped 28 Sep), WORKERS=2.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
if [ "${EDIT_Q:-0}" != "1" ]; then           # every local render goes through the render queue (COMMON.md)
  export EDIT_Q=1
  exec python3 "$HOME/mishamisha/_local_logs/renderq.py" -- bash "$SELF" "$@"
fi
source ~/.venvs/longdawn/env.sh
export NUMBA_NUM_THREADS=2
for c in ${TITLE_CUTS:-A}; do
  python3 edit/ember_title_v3.py --cut "$c" --workers "${WORKERS:-2}" | grep -v '^  '
done
