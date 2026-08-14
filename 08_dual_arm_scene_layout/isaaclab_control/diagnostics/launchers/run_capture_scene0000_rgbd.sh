#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
CONDA_SETUP="/home/lin/miniconda3/etc/profile.d/conda.sh"
ISAAC_SIM_ENV="/home/lin/isaacsim/setup_conda_env.sh"
ISAACLAB_ROOT="/home/lin/Projects/Wuji2_DexGraspNet_Portable_Dataset_Factory/01_environment/vendor/IsaacLab-2.2.0"
PROGRAM="${PROJECT_ROOT}/08_dual_arm_scene_layout/isaaclab_control/tools/02_capture_scene0000_rgbd.py"

export TERM=xterm-256color PYTHONUNBUFFERED=1
source "${CONDA_SETUP}"
conda activate wuji2_factory
set +u
source "${ISAAC_SIM_ENV}"
set -u

exec bash "${ISAACLAB_ROOT}/isaaclab.sh" -p "${PROGRAM}" --headless --enable_cameras "$@"
