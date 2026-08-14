#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
LAB_ROOT="/home/lin/Projects/Wuji2_DexGraspNet_Portable_Dataset_Factory/01_environment/vendor/IsaacLab-2.2.0"
CONDA_SETUP="/home/lin/miniconda3/etc/profile.d/conda.sh"
ISAAC_SIM_ENV="/home/lin/isaacsim/setup_conda_env.sh"
SCRIPT="$ROOT/08_dual_arm_scene_layout/isaaclab_control/runtime/scripts/10_run_full_pick_place.py"
CASE_ROOT="$ROOT/06_leap_to_wuji2_final_pipeline/99_archive/regenerable_cases/live_scene0000_ashtray_fullflow_candidate0597"

cd "$ROOT"
export TERM=xterm-256color PYTHONUNBUFFERED=1
source "$CONDA_SETUP"
conda activate wuji2_factory
set +u
source "$ISAAC_SIM_ENV"
set -u
exec bash "$LAB_ROOT/isaaclab.sh" -p "$SCRIPT" \
  --case-root "$CASE_ROOT" \
  --config "$ROOT/08_dual_arm_scene_layout/isaaclab_control/runtime/config/full_pick_place.json" \
  "$@"
