#!/usr/bin/env bash
set -eo pipefail

# 唯一启动入口：先进入项目固定的Conda环境，再注入本机Isaac Sim 5.0环境，
# 最后由Isaac Lab的isaaclab.sh和AppLauncher启动唯一一个Kit进程。
PROJECT_ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
CONDA_SETUP="/home/lin/miniconda3/etc/profile.d/conda.sh"
ISAAC_SIM_ENV="/home/lin/isaacsim/setup_conda_env.sh"
ISAACLAB_ROOT="/home/lin/Projects/Wuji2_DexGraspNet_Portable_Dataset_Factory/01_environment/vendor/IsaacLab-2.2.0"
PROGRAM="${PROJECT_ROOT}/08_dual_arm_scene_layout/isaaclab_control/runtime/scripts/01_run_short_motion.py"

if [[ ! -f "${CONDA_SETUP}" ]]; then
    echo "Missing Conda setup script: ${CONDA_SETUP}" >&2
    exit 1
fi
if [[ ! -f "${ISAAC_SIM_ENV}" ]]; then
    echo "Missing Isaac Sim environment script: ${ISAAC_SIM_ENV}" >&2
    exit 1
fi
if [[ ! -f "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    echo "Missing Isaac Lab launcher: ${ISAACLAB_ROOT}/isaaclab.sh" >&2
    exit 1
fi

export TERM=xterm-256color PYTHONUNBUFFERED=1
source "${CONDA_SETUP}"
conda activate isaaclab22_sim50
set +u
source "${ISAAC_SIM_ENV}"
set -u
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_tasks${PYTHONPATH:+:${PYTHONPATH}}"
exec bash "${ISAACLAB_ROOT}/isaaclab.sh" -p "${PROGRAM}" "$@"
