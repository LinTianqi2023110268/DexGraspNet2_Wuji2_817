#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
RUNNER="$ROOT/08_dual_arm_scene_layout/isaaclab_control/runtime/launchers/run_replay_25s.sh"
LOG="$ROOT/08_dual_arm_scene_layout/isaaclab_control/outputs/full_pick_place_25s_dog_candidate3800/replay_terminal.log"

exec gnome-terminal \
  --title="Isaac Lab - Physical Replay (24.85 s)" \
  -- /bin/bash -lc "
    set -o pipefail
    cd '$ROOT'
    bash '$RUNNER' 2>&1 | tee '$LOG'
    printf '\n[REPLAY PROCESS FINISHED]\nLog: %s\n' '$LOG'
    exec /bin/bash
  "
