#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config-json> [extra AppLauncher args...]" >&2
    exit 2
fi

CONFIG="$1"
shift

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control"
exec bash "${ROOT}/diagnostics/launchers/run_initial_stability.sh" --config "${CONFIG}" "$@"
