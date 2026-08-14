#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG_JSON [extra AppLauncher args]"
  exit 2
fi

CONFIG="$1"
shift || true

exec "${ROOT}/diagnostics/launchers/run_initial_stability.sh" --config "${CONFIG}" "$@"
