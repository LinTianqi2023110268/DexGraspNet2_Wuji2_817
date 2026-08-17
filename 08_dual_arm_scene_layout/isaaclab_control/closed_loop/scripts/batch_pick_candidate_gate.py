#!/usr/bin/env python3
"""Chunked pick-stage cuRobo gate for closed-loop candidate screening.

This is the production screening primitive that replaces the old one-process
per-candidate gate for PREGRASP/COVER/GRASP/SQUEEZE/LIFT:

* one persistent cuRobo worker per closed-loop cycle;
* one RGB-D Mapper/TSDF/ESDF build per captured frame;
* DGN2 candidates are supplied in official score order and processed in chunks;
* each chunk sends ``chunk_size * 5`` flange targets in one grouped GPU IK
  request, while branch continuity is still selected independently per
  candidate from q_current.

The script does not build LEAP/Wuji2 cases.  Upstream code should materialize
only scratch cases for the current chunk, not permanent ``01_cases/active``
directories for rejected candidates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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


def load_case_pick_contract(case_root: Path, measured: dict, T_base_from_world: np.ndarray) -> dict:
    route_path = case_root / "07_arm_execution/arm_flange_targets.npz"
    hand_path = case_root / "06_isaacsim/final_waypoints.npz"
    if not route_path.is_file():
        raise FileNotFoundError(route_path)
    if not hand_path.is_file():
        raise FileNotFoundError(hand_path)
    with np.load(route_path, allow_pickle=False) as z:
        names = [str(x) for x in z["waypoint_names"].tolist()]
        flange_world = np.asarray(z["world_from_right_flange"], dtype=np.float64)
    if names[:5] != PICK_STAGES:
        raise RuntimeError(f"{case_root}: first stages are not Route-C pick stages: {names[:5]}")
    with np.load(hand_path, allow_pickle=False) as z:
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
    case_json = load_json(case_root / "case.json")
    return {
        "case_root": str(case_root),
        "case_id": case_root.name,
        "candidate_index": int(case_json.get("source_candidate_index", -1)),
        "official_score": float(case_json.get("official_score", "nan")),
        "targets_base": targets_base,
        "states": states,
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, action="append", required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-unknown", action="store_true")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    project_root = args.project_root.expanduser().resolve()
    capture_root = args.capture_root.expanduser().resolve()
    state = load_json(args.robot_state.expanduser().resolve())
    measured = {str(k): float(v) for k, v in state["joint_positions_by_name"].items()}
    q_current = np.asarray(state["right_arm_q_current_rad"], dtype=np.float64)
    T_world_base = world_from_base(project_root)
    T_base_from_world = np.linalg.inv(T_world_base)

    cases = [
        load_case_pick_contract(case_root.expanduser().resolve(), measured, T_base_from_world)
        for case_root in args.case_root
    ]

    chunks = []
    selected = None
    worker_start_count = 1
    map_build_count = 0
    started = time.perf_counter()
    with CuroboWorkerClient(project_root, seeds=48, batch_size=max(64, args.chunk_size * 5)) as client:
        map_started = time.perf_counter()
        map_report = client.build_map(
            capture_root / "depth_m.npy",
            capture_root / "intrinsics.npy",
            capture_root / "T_world_camera.npy",
            args.mask.expanduser().resolve(),
        )
        map_wall_s = time.perf_counter() - map_started
        map_build_count += 1

        for start in range(0, len(cases), args.chunk_size):
            chunk_cases = cases[start:start + args.chunk_size]
            targets = np.concatenate([item["targets_base"] for item in chunk_cases], axis=0)
            states = [state for item in chunk_cases for state in item["states"]]
            phases = [phase for item in chunk_cases for phase in item["phases"]]
            solve_started = time.perf_counter()
            solve = client.solve_ik_groups(
                targets,
                q_current,
                group_sizes=[len(PICK_STAGES)] * len(chunk_cases),
                select_chain=True,
                collision_context={
                    "phases": phases,
                    "joint_positions_by_name": measured,
                    "joint_positions_by_target": states,
                    "T_world_base": T_world_base,
                    "margin_m": 0.0,
                    "include_return_to_reference": False,
                },
            )
            solve_wall_s = time.perf_counter() - solve_started
            group_rows = []
            for local_index, (case, group) in enumerate(zip(chunk_cases, solve["groups"])):
                unknown = group.get("unknown_space_exposure") or []
                unknown_policy_pass = (not any(bool(x) for x in unknown)) if args.block_unknown else True
                feasible = bool(
                    group.get("ik_pass") is True
                    and group.get("observed_scene_collision_pass") is True
                    and group.get("path_pass") is True
                    and unknown_policy_pass
                )
                row = {
                    "score_order_index": start + local_index,
                    "case_id": case["case_id"],
                    "case_root": case["case_root"],
                    "candidate_index": case["candidate_index"],
                    "official_score": case["official_score"],
                    "feasible": feasible,
                    "unknown_policy_pass": unknown_policy_pass,
                    **group,
                }
                group_rows.append(row)
                if selected is None and feasible:
                    selected = row
            chunks.append({
                "chunk_index": len(chunks),
                "score_order_start": start,
                "candidate_count": len(chunk_cases),
                "pose_count": int(len(targets)),
                "solve_wall_s": solve_wall_s,
                "worker_solve_time_s": solve.get("solve_time_s"),
                "groups": group_rows,
            })
            if selected is not None:
                break

    total_wall_s = time.perf_counter() - started
    tested = sum(int(chunk["candidate_count"]) for chunk in chunks)
    out = {
        "schema_version": 1,
        "status": "PASS" if selected is not None else "FAIL",
        "scope": "chunked pick-stage candidate screening only",
        "SELF_COLLISION_POLICY": "REPORT_ONLY_UNRESOLVED",
        "candidate_order": "caller-supplied official DGN2 score descending order",
        "pick_stages": PICK_STAGES,
        "selected": selected,
        "chunks": chunks,
        "metrics": {
            "worker_start_count": worker_start_count,
            "map_build_count": map_build_count,
            "chunk_size": args.chunk_size,
            "tested_candidate_count": tested,
            "map_wall_s": map_wall_s,
            "total_wall_s": total_wall_s,
            "mean_wall_s_per_tested_candidate": total_wall_s / max(tested, 1),
            "poses_per_full_chunk": args.chunk_size * len(PICK_STAGES),
        },
        "map": map_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "selected_candidate_index": None if selected is None else selected["candidate_index"],
        "tested_candidate_count": tested,
        "worker_start_count": worker_start_count,
        "map_build_count": map_build_count,
        "chunk_size": args.chunk_size,
        "poses_per_full_chunk": args.chunk_size * len(PICK_STAGES),
        "mean_wall_s_per_tested_candidate": out["metrics"]["mean_wall_s_per_tested_candidate"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
