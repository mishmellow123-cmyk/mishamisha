#!/bin/sh
# numba's on-disk cache does not track cross-module inlined functions; clear it after
# editing core.py / sky.py / any shared @njit helper.
find "$(dirname "$0")" -name '*.nbi' -delete
find "$(dirname "$0")" -name '*.nbc' -delete
