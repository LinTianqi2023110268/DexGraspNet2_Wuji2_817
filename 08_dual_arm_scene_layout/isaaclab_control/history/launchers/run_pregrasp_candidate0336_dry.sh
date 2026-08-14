#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
CASE_ROOT="$PROJECT_ROOT/06_leap_to_wuji2_final_pipeline/99_archive/regenerable_cases/live_scene0000_ashtray_armreachable_candidate0336"
CONDA_SETUP="/home/lin/miniconda3/etc/profile.d/conda.sh"
ISAAC_SIM_ENV="/home/lin/isaacsim/setup_conda_env.sh"
ISAACLAB_ROOT="/home/lin/Projects/Wuji2_DexGraspNet_Portable_Dataset_Factory/01_environment/vendor/IsaacLab-2.2.0"

cd "$PROJECT_ROOT"
export TERM=xterm-256color PYTHONUNBUFFERED=1
source "$CONDA_SETUP"
conda activate wuji2_factory
set +u
source "$ISAAC_SIM_ENV"
set -u

exec bash "$ISAACLAB_ROOT/isaaclab.sh" \
  -p 08_dual_arm_scene_layout/isaaclab_control/runtime/scripts/01_run_short_motion.py \
  --config 08_dual_arm_scene_layout/isaaclab_control/diagnostics/config/short_motion_ft04_z20.json \
  --joint-target-npz "$CASE_ROOT/07_arm_execution/pregrasp_read_only_ik.npz" \
  --waypoints-npz "$CASE_ROOT/06_isaacsim/final_waypoints.npz" \
  --flange-targets-npz "$CASE_ROOT/07_arm_execution/arm_flange_targets.npz" \
  --joint-duration-s 22 \
  --endpoint-refine-s 6 \
  --position-tolerance-mm 5 \
  --orientation-tolerance-deg 5 \
  --output-directory 08_dual_arm_scene_layout/isaaclab_control/outputs/pregrasp_candidate0336_dry \
  --headless
