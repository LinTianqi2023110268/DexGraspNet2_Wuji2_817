#!/usr/bin/env bash
set -euo pipefail

# Open a visible Isaac Lab terminal while Isaac Sim runs in its own GUI.
# The complete console output is also saved for later inspection.

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
RUNNER="$ROOT/08_dual_arm_scene_layout/isaaclab_control/runtime/launchers/run_full_pick_place_25s_dog_candidate3800.sh"
LOG_DIR="$ROOT/08_dual_arm_scene_layout/isaaclab_control/outputs/full_pick_place_25s_dog_candidate3800"
LOG_FILE="$LOG_DIR/visible_terminal.log"

mkdir -p "$LOG_DIR"

if pgrep -f "^/home/lin/miniconda3/envs/isaaclab22_sim50/bin/python .*10_run_full_pick_place.py" >/dev/null; then
  echo "[REFUSED] Another full-pipeline Isaac process is already running." >&2
  exit 3
fi

exec gnome-terminal \
  --title="Isaac Lab - DGN2 Full Pick Place (25 s)" \
  -- /bin/bash -lc "
    set -o pipefail
    cd '$ROOT'
    bash '$RUNNER' 2>&1 | tee '$LOG_FILE'
    printf '\n[PROCESS FINISHED]\n'
    printf 'Log: %s\n' '$LOG_FILE'
    printf 'The terminal will stay open for inspection.\n'
    exec /bin/bash
  "
