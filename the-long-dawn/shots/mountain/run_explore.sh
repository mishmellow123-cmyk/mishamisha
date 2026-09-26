#!/bin/bash
# usage: ./run_explore.sh NAME key=val ...
cd "$(dirname "$0")"
S=/private/tmp/claude-501/-Users-mishasalahshoor/d71369b9-7742-4da4-abf2-0e5d5e46099b/scratchpad
source ~/.venvs/longdawn/env.sh
export NUMBA_NUM_THREADS=2
N=$1
python explore_land.py "$@" 2>&1 | grep -E "BASE|land points|cloud saved|Error|Trace|line"
rm -rf $S/x_$N
~/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -P bl/render_shot.py -- --shot x$N --frames 1,2,3 --scale 0.5 --samples 16 --out $S/x_$N --nomb 2>&1 | grep -E "^frame|Error|Trace"
python sheet.py $S/x_$N $S/x_$N.jpg 1.0 1 960
