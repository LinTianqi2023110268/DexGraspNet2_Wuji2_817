#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control"
CONFIG="${ROOT}/diagnostics/config/diagnostic_grouped_pd_zero_gravity.json"

exec bash "${ROOT}/diagnostics/launchers/run_initial_stability.sh" --config "${CONFIG}" "$@"
