#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
PYTHON="/home/lin/miniconda3/envs/graspnet2.0/bin/python"
SCRIPT="$ROOT/08_dual_arm_scene_layout/isaaclab_control/report_demo/scripts/02_interactive_report_demo.py"

exec gnome-terminal \
  --title="DGN2 Wuji2 Interactive Report Demo" \
  --geometry=92x46+1060+20 \
  -- bash -lc "cd '$ROOT'; exec '$PYTHON' '$SCRIPT'"
