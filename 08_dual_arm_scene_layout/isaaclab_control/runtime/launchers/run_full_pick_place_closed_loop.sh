#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
LAB_ROOT="/home/lin/Projects/Wuji2_DexGraspNet_Portable_Dataset_Factory/01_environment/vendor/IsaacLab-2.2.0"
CONDA_SETUP="/home/lin/miniconda3/etc/profile.d/conda.sh"
ISAAC_SIM_ENV="/home/lin/isaacsim/setup_conda_env.sh"
SCRIPT="$ROOT/08_dual_arm_scene_layout/isaaclab_control/runtime/scripts/10_run_full_pick_place.py"

if pgrep -f "^/home/lin/miniconda3/envs/isaaclab22_sim50/bin/python .*10_run_full_pick_place.py" >/dev/null; then
  echo "[REFUSED] Another full-pipeline Isaac process is already running. Close it before retrying." >&2
  exit 3
fi

cd "$ROOT"
export TERM=xterm-256color PYTHONUNBUFFERED=1
source "$CONDA_SETUP"
conda activate isaaclab22_sim50
set +u
source "$ISAAC_SIM_ENV"
set -u
export PYTHONPATH="$LAB_ROOT/source/isaaclab:$LAB_ROOT/source/isaaclab_assets:$LAB_ROOT/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"
exec bash "$LAB_ROOT/isaaclab.sh" -p "$SCRIPT" "$@"
