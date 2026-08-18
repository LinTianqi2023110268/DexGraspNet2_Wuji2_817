#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="$ROOT/08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py"
[[ -f "$ORCH" ]] || { echo "✗ 缺少总控脚本：$ORCH" >&2; exit 2; }
exec /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python "$ORCH" --project-root "$ROOT" "$@"
