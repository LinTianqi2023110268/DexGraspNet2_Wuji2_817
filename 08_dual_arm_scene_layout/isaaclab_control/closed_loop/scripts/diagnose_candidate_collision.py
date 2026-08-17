#!/usr/bin/env python3
"""Detailed collision diagnosis for one retargeted pick-stage candidate."""
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


PICK_STAGES = ["pregrasp", "cover", "grasp", "squeeze", "lift"]
PHASE_FOR = {
    "pregrasp": "pregrasp",
    "cover": "cover",
    "grasp": "grasp",
    "squeeze": "squeeze",
    "lift": "lift",
}
HAND_FOR = {
    "pregrasp": "pregrasp",
    "cover": "cover",
    "grasp": "grasp",
    "squeeze": "squeeze",
    "lift": "squeeze",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def load_pick_inputs(case_root: Path, robot_state: Path, project_root: Path) -> tuple[np.ndarray, np.ndarray, list[dict], list[str], np.ndarray]:
    state = load_json(robot_state)
    measured = {str(k): float(v) for k, v in state["joint_positions_by_name"].items()}
    q_current = np.asarray(state["right_arm_q_current_rad"], dtype=np.float64)
    T_world_base = world_from_base(project_root)
    T_base_from_world = np.linalg.inv(T_world_base)
    with np.load(case_root / "07_arm_execution/arm_flange_targets.npz", allow_pickle=False) as z:
        names = [str(x) for x in z["waypoint_names"].tolist()]
        flange_world = np.asarray(z["world_from_right_flange"], dtype=np.float64)
    if names[:5] != PICK_STAGES:
        raise RuntimeError(f"unexpected stages: {names[:5]}")
    with np.load(case_root / "06_isaacsim/final_waypoints.npz", allow_pickle=False) as z:
        hand_names = [str(x) for x in z["finger_joint_names"].tolist()]
        hand_stage_names = [str(x) for x in z["waypoint_names"].tolist()]
        hand_q = np.asarray(z["waypoint_joint_positions"][0], dtype=np.float64)
    hand_index = {name: i for i, name in enumerate(hand_stage_names)}
    states = []
    phases = []
    for stage in PICK_STAGES:
        named = dict(measured)
        qh = hand_q[hand_index[HAND_FOR[stage]]]
        for joint_name, q in zip(hand_names, qh):
            named[joint_name] = float(q)
        states.append(named)
        phases.append(PHASE_FOR[stage])
    targets_base = np.stack([T_base_from_world @ T for T in flange_world[:5]])
    return targets_base, q_current, states, phases, T_world_base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    case_root = args.case_root.expanduser().resolve()
    capture_root = args.capture_root.expanduser().resolve()
    targets_base, q_current, states, phases, T_world_base = load_pick_inputs(
        case_root, args.robot_state.expanduser().resolve(), project_root
    )
    with CuroboWorkerClient(project_root, seeds=48, batch_size=64) as client:
        map_report = client.build_map(
            capture_root / "depth_m.npy",
            capture_root / "intrinsics.npy",
            capture_root / "T_world_camera.npy",
            args.mask.expanduser().resolve(),
        )
        diag = client.diagnose_ik_collisions(
            targets_base,
            q_current,
            {
                "phases": phases,
                "joint_positions_by_name": states[0],
                "joint_positions_by_target": states,
                "T_world_base": T_world_base,
                "margin_m": 0.0,
                "include_return_to_reference": False,
            },
            top_k=args.top_k,
        )
    out = {
        "schema_version": 1,
        "case_root": str(case_root),
        "capture_root": str(capture_root),
        "stages": PICK_STAGES,
        "map": map_report,
        "diagnosis": diag,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output": str(args.output),
        "stage_summary": [
            {
                "phase": x["phase"],
                "raw": x["raw_ik_solutions"],
                "threshold": x["threshold_accepted"],
                "self": x["rejected_by_self_collision"],
                "scene": x["rejected_by_scene_esdf"],
                "target": x["rejected_by_target_esdf"],
                "multiple": x["rejected_by_multiple_causes"],
                "survivors": x["final_surviving_solutions"],
            }
            for x in diag["stage_reports"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
