#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
RUNNER="$ROOT/08_dual_arm_scene_layout/isaaclab_control/history/launchers/run_full_pick_place_live_dog_candidate3800.sh"

# Start one visible terminal.  That terminal owns the Isaac Lab process and
# prints the live state-machine telemetry while Isaac Sim renders the motion.
exec gnome-terminal \
  --title="Isaac Lab - DGN2 Dog Candidate3800 Live Monitor" \
  -- bash -lc "cd '$ROOT'; exec '$RUNNER' --stop-after-lift"
