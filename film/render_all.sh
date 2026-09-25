#!/bin/sh
# Render every shot of the film (resumable: finished frames are skipped).
# usage: sh render_all.sh OUTDIR
OUT=${1:-frames}
DIR=$(cd "$(dirname "$0")" && pwd)
for s in 01_question 02_rings 03_voices 04_rising 05_branching 06_answer 07_sphere 08_held 09_stillness; do
  echo "=== $s $(date +%H:%M:%S)"
  python3 "$DIR/shots.py" "$s" "$OUT/$s" 2>&1 | grep --line-buffered -E "^frame|Error|Traceback"
done
echo "=== done $(date +%H:%M:%S)"
