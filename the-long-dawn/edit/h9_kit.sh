#!/bin/bash
# THE LONG DAWN v3: rebuild the H9 critic kit from the current EDL and renders (about 3 min for all three films).
# Everything in the kit is regenerated; the critic judges from it without watching video:
#   ~/mishamisha/_local_logs/review/h9/  INDEX.md  COVERAGE.md  TEXT.md
#     {A,B,C}_overview.jpg  {A,B,C}_bars_<a>-<b>.jpg  A_ALT_codedtowers_bars_<a>-<b>.jpg  stills/<film>/*.jpg
# Options (env): CUTS="A C" (default; film B was dropped 28 Sep), WORKERS=3.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
if [ "${EDIT_Q:-0}" != "1" ]; then           # every local render goes through the render queue (COMMON.md)
  export EDIT_Q=1
  exec python3 "$HOME/mishamisha/_local_logs/renderq.py" -- bash "$SELF" "$@"
fi
source ~/.venvs/longdawn/env.sh
# MAP-v3's X1 letters-to-fire motion test stands in for C4/C5 until renders/book_C lands (as in animatic.sh)
if [ ! -s edit/cache/x1_letters_C_test.mp4 ]; then
  mkdir -p edit/cache
  git -C .. show origin/claude/v3-map:the-long-dawn/review/v3/map_X1_letters_to_fire.mp4 \
    > edit/cache/x1_letters_C_test.mp4 2>/dev/null || rm -f edit/cache/x1_letters_C_test.mp4
fi
CUTS="${CUTS:-A C}"
python3 edit/h9_kit.py --cuts "${CUTS// /}" --workers "${WORKERS:-3}"
