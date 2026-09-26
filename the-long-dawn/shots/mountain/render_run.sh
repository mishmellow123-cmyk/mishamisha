#!/bin/bash
# Full-res production render of THE BEACON RUN (v2 1520-1679) -> EXR -> renders/run/*.png
cd "$(dirname "$0")"
source ~/.venvs/longdawn/env.sh
export NUMBA_NUM_THREADS=2
mkdir -p cache/run_exr
rm -f cache/run_exr/DONE
( python post_run.py run --watch > cache/post_run.log 2>&1 ) &
caffeinate -i ~/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -P bl/render_shot.py -- \
    --shot run --frames ${FRAMES:-1520-1679} --scale 1.0 --samples ${SAMPLES:-32} \
    --out cache/run_exr --skip-existing > cache/render_run.log 2>&1
touch cache/run_exr/DONE
wait
echo ALLDONE >> cache/render_run.log
