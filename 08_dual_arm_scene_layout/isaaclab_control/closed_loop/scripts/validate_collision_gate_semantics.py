#!/usr/bin/env python3
"""Regression tests for self-collision and continuous-path gate semantics.

The goal is not to tune thresholds.  It verifies that the current worker reports
the expected sign for:

* a known safe right-arm state from the previous GPU IK PASS case;
* a deliberately folded self-colliding state;
* a trivial continuous path that stays at the safe state;
* a deliberate path ending at the self-colliding state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
sys.path.insert(0, str(CONTROL_ROOT))

from core.bridge import CuroboWorkerClient  # noqa: E402
from core.config import RIGHT_ARM_NAMES  # noqa: E402


SAFE_Q = np.asarray([
    0.8562305569648743,
    -1.2129706144332886,
    -0.01392145361751318,
    0.6837326288223267,
    0.5853013396263123,
    0.020781924948096275,
    0.445459246635437,
], dtype=np.float64)

SELF_COLLIDING_Q = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def arm_state(q: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(RIGHT_ARM_NAMES, q)}


def complete_state(defaults: dict[str, float], q: np.ndarray) -> dict[str, float]:
    state = {str(name): float(value) for name, value in defaults.items()}
    state.update(arm_state(q))
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()

    with CuroboWorkerClient(root, seeds=8, batch_size=8) as client:
        safe_self = client.check_self_collision(arm_state(SAFE_Q))
        colliding_self = client.check_self_collision(arm_state(SELF_COLLIDING_Q))
        sphere_info = client.robot_spheres(arm_state(SAFE_Q))
        defaults = sphere_info["default_joint_positions_by_name"]
        safe_complete = complete_state(defaults, SAFE_Q)
        colliding_complete = complete_state(defaults, SELF_COLLIDING_Q)
        safe_path = client.check_joint_path(
            np.stack([SAFE_Q, SAFE_Q]),
            safe_complete,
            check_observed_map=False,
        )
        colliding_path = client.check_joint_path(
            np.stack([SAFE_Q, SELF_COLLIDING_Q]),
            safe_complete,
            joint_positions_by_node=[safe_complete, colliding_complete],
            check_observed_map=False,
        )

    status = (
        safe_self["self_collision_pass"] is True
        and colliding_self["self_collision_pass"] is False
        and safe_path["path_pass"] is True
        and colliding_path["path_pass"] is False
    )
    out = {
        "schema_version": 1,
        "status": "PASS" if status else "FAIL",
        "safe_q_rad": SAFE_Q.tolist(),
        "self_colliding_q_rad": SELF_COLLIDING_Q.tolist(),
        "self_collision_negative_safe_state": safe_self,
        "self_collision_positive_folded_state": colliding_self,
        "continuous_path_negative_safe_path": safe_path,
        "continuous_path_positive_self_collision_path": colliding_path,
        "note": (
            "continuous path test is self-collision semantics only; observed-ESDF "
            "path checking is exercised by candidate gates after build_map."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "safe_self_collision_pass": safe_self["self_collision_pass"],
        "folded_self_collision_pass": colliding_self["self_collision_pass"],
        "safe_path_pass": safe_path["path_pass"],
        "colliding_path_pass": colliding_path["path_pass"],
        "output": str(args.output),
    }, ensure_ascii=False))
    if not status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
