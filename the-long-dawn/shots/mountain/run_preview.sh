#!/bin/bash
# usage: CAMPOS=.. CAMTGT=.. ./run_preview.sh name [exposure]
cd "$(dirname "$0")"
S=/private/tmp/claude-501/-Users-mishasalahshoor/d71369b9-7742-4da4-abf2-0e5d5e46099b/scratchpad
source ~/.venvs/longdawn/env.sh
export NUMBA_NUM_THREADS=2
python preview_land.py | grep -E "points|thinning|heights"
~/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -P bl/bl_preview.py -- cache/preview_land cache/preview_cam.json $S/$1.exr ${SCALE:-0.5} 2>&1 | grep -E "^render|Error"
python post_quick.py $S/$1.exr $S/$1.png ${2:-6.0}
