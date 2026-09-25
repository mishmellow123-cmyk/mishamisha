#!/bin/bash
# Full-resolution render of EMBERS (global frames 300-1199) with two single-threaded workers.
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=../../renders/embers/logs
python3 render.py ${1:-300-759} > $LOG/worker_a.log 2>&1 &
python3 render.py ${2:-760-1199} > $LOG/worker_b.log 2>&1 &
wait
