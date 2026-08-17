#!/usr/bin/env python3
"""Fail-closed cuRobo gate for one retargeted DGN2 candidate.

Current core Route-C V2 already provides endpoint IK + observed RGB-D ESDF.
This gate additionally EXPECTS future worker fields for:
  - self_collision_pass
  - path_pass (continuous interpolated path, not only waypoint endpoints)

Until Codex implements those fields, ``feasible`` remains False.  This is
intentional: the patch must not turn an unfinished safety check into a PASS.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTROL_ROOT = PROJECT_ROOT/"08_dual_arm_scene_layout/isaaclab_control"
sys.path.insert(0, str(CONTROL_ROOT))
from core.bridge import CuroboWorkerClient

def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case-root", type=Path, required=True)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--robot-state", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--block-unknown", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    case_root = a.case_root.resolve()
    route_path = case_root/"07_arm_execution/full_arm_waypoint_ik.npz"
    hand_path = case_root/"06_isaacsim/final_waypoints.npz"
    with np.load(route_path, allow_pickle=False) as z:
        names = [str(x) for x in z["waypoint_names"].tolist()]
        flange_world = np.asarray(z["world_from_right_flange"], dtype=np.float64)
    with np.load(hand_path, allow_pickle=False) as z:
        hand_names = [str(x) for x in z["finger_joint_names"].tolist()]
        hand_stage_names = [str(x) for x in z["waypoint_names"].tolist()]
        hand_q5 = np.asarray(z["waypoint_joint_positions"][0], dtype=np.float64)
    state = load_json(a.robot_state)
    q_current = np.asarray(state["right_arm_q_current_rad"], dtype=np.float64)
    measured = {str(k):float(v) for k,v in state["joint_positions_by_name"].items()}

    layout = load_json(PROJECT_ROOT/"08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    world_from_base = np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"], dtype=np.float64
    ).T
    base_from_world = np.linalg.inv(world_from_base)
    targets_base = np.stack([base_from_world @ T for T in flange_world])

    hand_index = {name:i for i,name in enumerate(hand_stage_names)}
    hand_for = {
        "pregrasp":"pregrasp","cover":"cover","grasp":"grasp","squeeze":"squeeze","lift":"squeeze",
        "transfer":"squeeze","place":"squeeze","release":"pregrasp","retreat":"pregrasp",
    }
    phase_for = {
        "pregrasp":"pregrasp","cover":"cover","grasp":"grasp","squeeze":"squeeze","lift":"lift",
        "transfer":"lift","place":"lift","release":"lift","retreat":"lift",
    }
    states, phases = [], []
    for stage in names:
        if stage not in hand_for:
            raise RuntimeError(f"No hand/phase policy for {stage}")
        named = dict(measured)
        qh = hand_q5[hand_index[hand_for[stage]]]
        for jn, q in zip(hand_names, qh):
            named[jn] = float(q)
        states.append(named)
        phases.append(phase_for[stage])

    capture = a.capture_root.resolve()
    with CuroboWorkerClient(PROJECT_ROOT) as client:
        map_report = client.build_map(
            capture/"depth_m.npy", capture/"intrinsics.npy", capture/"T_world_camera.npy", a.mask
        )
        solve = client.solve_ik(
            targets_base, q_current, select_chain=True,
                collision_context={
                    "phases": phases,
                    "joint_positions_by_name": measured,
                    "joint_positions_by_target": states,
                    "T_world_base": world_from_base,
                    "margin_m": 0.0,
                },
        )

    ik_pass = bool(solve.get("ik_pass") and solve.get("selected") is not None)
    observed_pass = bool(solve.get("observed_scene_collision_pass"))
    unknown_list = solve.get("unknown_space_exposure") or []
    unknown_exposure = bool(any(bool(x) for x in unknown_list))
    unknown_policy_pass = (not unknown_exposure) if a.block_unknown else True
    self_collision_pass = solve.get("self_collision_pass")
    self_collision_policy = "REPORT_ONLY_UNRESOLVED"
    path_pass = solve.get("path_pass")
    feasible = bool(
        ik_pass and observed_pass and unknown_policy_pass
        and path_pass is True
    )
    out = {
        "schema_version": 1,
        "status": "PASS" if feasible else "FAIL",
        "feasible": feasible,
        "ik_pass": ik_pass,
        "observed_scene_collision_pass": observed_pass,
        "unknown_space_exposure": unknown_exposure,
        "unknown_policy_pass": unknown_policy_pass,
        "self_collision_pass": self_collision_pass,
        "self_collision_policy": self_collision_policy,
        "SELF_COLLISION_POLICY": self_collision_policy,
        "path_pass": path_pass,
        "map": map_report,
        "solve": solve,
        "fail_closed_reason": None if feasible else (
            "Endpoint IK/ESDF and continuous observed-scene path gates must be explicit True; "
            "self-collision is report-only pending model-semantics audit"
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k not in {"map","solve"}}, ensure_ascii=False))

if __name__ == "__main__":
    main()
