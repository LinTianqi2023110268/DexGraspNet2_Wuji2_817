#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="$ROOT/08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py"

if [[ ! -f "$ORCH" ]]; then
  echo "[ERROR] Missing closed-loop orchestrator: $ORCH" >&2
  exit 2
fi

exec /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python "$ORCH" --project-root "$ROOT" "$@"
