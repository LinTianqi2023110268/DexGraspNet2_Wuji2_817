#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
LAB_ROOT="/home/lin/Projects/Wuji2_DexGraspNet_Portable_Dataset_Factory/01_environment/vendor/IsaacLab-2.2.0"
CONDA_SETUP="/home/lin/miniconda3/etc/profile.d/conda.sh"
ISAAC_SIM_ENV="/home/lin/isaacsim/setup_conda_env.sh"
SCRIPT="$ROOT/08_dual_arm_scene_layout/isaaclab_control/runtime/scripts/10_run_full_pick_place.py"
CASE_ROOT="$ROOT/06_leap_to_wuji2_final_pipeline/01_cases/live_dynamic_scene0000_dog_candidate3800"
CONFIG="$ROOT/08_dual_arm_scene_layout/isaaclab_control/runtime/config/full_pick_place_25s_dog_candidate3800.json"

if pgrep -f "^/home/lin/miniconda3/envs/wuji2_factory/bin/python .*10_run_full_pick_place.py" >/dev/null; then
  echo "[REFUSED] Another full-pipeline Isaac process is already running. Close it before retrying." >&2
  exit 3
fi

cd "$ROOT"
export TERM=xterm-256color PYTHONUNBUFFERED=1
source "$CONDA_SETUP"
conda activate wuji2_factory
set +u
source "$ISAAC_SIM_ENV"
set -u
exec bash "$LAB_ROOT/isaaclab.sh" -p "$SCRIPT" \
  --case-root "$CASE_ROOT" \
  --config "$CONFIG" \
  "$@"
