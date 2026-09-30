#!/bin/bash
# THE LONG DAWN v3: export every finished stretch (>= ~30 s, fully rendered) of each film from its current master,
# with its sound, to ~/Downloads/The Long Dawn v3 - PREVIEWS/ with README.txt (see edit/previews.py).
# Runs through the local render queue. Prints NEW: <file> for each new file.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
if [ "${EDIT_Q:-0}" != "1" ]; then           # every local render goes through the render queue (COMMON.md)
  export EDIT_Q=1
  exec python3 "$HOME/mishamisha/_local_logs/renderq.py" -- bash "$SELF" "$@"
fi
source ~/.venvs/longdawn/env.sh
python3 edit/previews.py "$@"
