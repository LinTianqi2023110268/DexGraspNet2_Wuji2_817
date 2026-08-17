#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
CONDA_SETUP="/home/lin/miniconda3/etc/profile.d/conda.sh"
ISAAC_SIM_ENV="/home/lin/isaacsim/setup_conda_env.sh"
ISAACLAB_ROOT="/home/lin/Projects/Wuji2_DexGraspNet_Portable_Dataset_Factory/01_environment/vendor/IsaacLab-2.2.0"
PROGRAM="${PROJECT_ROOT}/08_dual_arm_scene_layout/isaaclab_control/tools/11_settle_and_capture_dynamic_scene.py"

for required in "$CONDA_SETUP" "$ISAAC_SIM_ENV" "$ISAACLAB_ROOT/isaaclab.sh" "$PROGRAM"; do
  [[ -f "$required" ]] || { echo "[ERROR] missing: $required" >&2; exit 2; }
done

export TERM=xterm-256color PYTHONUNBUFFERED=1
source "$CONDA_SETUP"
conda activate isaaclab22_sim50
set +u
source "$ISAAC_SIM_ENV"
set -u
export PYTHONPATH="$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_assets:$ISAACLAB_ROOT/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"

# Important: unlike the old diagnostic launcher, this does NOT pass --settle-only.
# The closed loop needs an aligned RGB-D frame on every cycle.
exec bash "$ISAACLAB_ROOT/isaaclab.sh" -p "$PROGRAM" --headless --enable_cameras "$@"
