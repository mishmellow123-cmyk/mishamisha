#!/bin/bash
# Full-res production render of DAWN_C (cut C, v2 2400-2655) -> EXR -> renders/dawn_C/*.png
cd "$(dirname "$0")"
source ~/.venvs/longdawn/env.sh
export NUMBA_NUM_THREADS=2
mkdir -p cache/dawn_exr
rm -f cache/dawn_exr/DONE
( python post_run.py dawn --watch > cache/post_dawn.log 2>&1 ) &
caffeinate -i ~/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -P bl/render_dawn.py -- \
    --frames ${FRAMES:-2400-2655} --scale 1.0 --samples ${SAMPLES:-32} \
    --out cache/dawn_exr --skip-existing > cache/render_dawn.log 2>&1
touch cache/dawn_exr/DONE
wait
echo ALLDONE >> cache/render_dawn.log
