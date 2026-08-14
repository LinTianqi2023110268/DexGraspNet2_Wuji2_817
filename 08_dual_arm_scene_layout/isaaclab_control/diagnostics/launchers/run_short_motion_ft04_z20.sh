#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control"
CONFIG="${ROOT}/diagnostics/config/short_motion_ft04_z20.json"

exec bash "${ROOT}/diagnostics/launchers/run_short_motion.sh" --config "${CONFIG}" "$@"
