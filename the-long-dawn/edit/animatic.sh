#!/bin/bash
# THE LONG DAWN v3: rebuild the three half-res animatics (960x402, H.264 CRF 23) from whatever has been rendered.
# Re-run it as renders land; anything missing is a slate at the right duration. Outputs:
#   ~/mishamisha/_local_logs/animatic/{A,B,C}_animatic.mp4   (+ A_animatic_codedtowers.mp4 once ALT frames exist)
#   edit/NOTES_v3.md COVERAGE block, edit/edl/edl_{A,B,C}.json
# Options (env): CUTS="A C" (default; B dropped 28 Sep), WORKERS=3, CLEAN=1 (no burn-ins), VARIANT=1 (force A's ALT master).
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
if [ "${EDIT_Q:-0}" != "1" ]; then           # every local render goes through the render queue (COMMON.md)
  export EDIT_Q=1
  exec python3 "$HOME/mishamisha/_local_logs/renderq.py" -- bash "$SELF" "$@"
fi
source ~/.venvs/longdawn/env.sh
mkdir -p edit/cache "$HOME/mishamisha/_local_logs/animatic"
# MAP-v3's X1 letters-to-fire motion test (C 560-906) stands in for C4/C5 until renders/book_C lands
if [ ! -s edit/cache/x1_letters_C_test.mp4 ]; then
  git -C .. show origin/claude/v3-map:the-long-dawn/review/v3/map_X1_letters_to_fire.mp4 \
    > edit/cache/x1_letters_C_test.mp4 2>/dev/null || rm -f edit/cache/x1_letters_C_test.mp4
fi
CUTS="${CUTS:-A C}"
WORKERS="${WORKERS:-3}"
EXTRA=""
[ "${CLEAN:-0}" = "1" ] && EXTRA="--clean"
python3 edit/assemble.py --edl --coverage --notes > /dev/null
# the incremental engine (edit/deliver.py): only shots whose inputs changed are re-encoded; OLD=1 = assemble.py
PROFILE=animatic
[ "${CLEAN:-0}" = "1" ] && PROFILE=animatic_clean
anim() {
  if [ "${OLD:-0}" = "1" ]; then python3 edit/assemble.py --cut "$1" --animatic --workers "$WORKERS" $EXTRA ${2:+--variant $2}
  else python3 edit/deliver.py --cut "$1" --profile "$PROFILE" --workers "$WORKERS" ${2:+--variant $2} | grep -v '^  '; fi
}
for c in $CUTS; do
  anim "$c"
done
# A's alternate master (the coded pair) only when _alt_codedtowers frames exist on A's own timeline
if [[ " $CUTS " == *" A "* ]] && { [ "${VARIANT:-0}" = "1" ] || ls renders/embers_A3_alt_codedtowers/f_* >/dev/null 2>&1; }; then
  anim A codedtowers
fi
python3 edit/assemble.py --coverage --notes | grep -E '^\*\*'
ls -la "$HOME/mishamisha/_local_logs/animatic/"
