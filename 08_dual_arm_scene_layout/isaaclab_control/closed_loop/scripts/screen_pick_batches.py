#!/usr/bin/env python3
"""Lazy chunked pick-stage screening for one closed-loop cycle.

This script is intentionally cycle-scoped:

* starts exactly one persistent cuRobo worker;
* builds Mapper/TSDF/ESDF exactly once for the provided RGB-D frame;
* processes official DGN2 candidates in score order by lazy chunks;
* materializes/retargets only the current chunk;
* sends each chunk as one grouped GPU request of ``chunk_size * 5`` poses;
* stops at the first pick-stage PASS candidate.

It does not allocate placement, build TRANSFER/PLACE/RELEASE/RETREAT, or run
physical motion.  HOME belongs to final full-route validation, not this
pick-stage screen.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
PIPELINE_SCRIPTS = PROJECT_ROOT / "06_leap_to_wuji2_final_pipeline/02_scripts"
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


def run(label: str, cmd: list, *, cwd: Path, env: dict | None = None, capture_json: bool = False):
    print(f"\n--- {label} ---", flush=True)
    print("$", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    completed = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        lines = completed.stdout.splitlines()
        preview = "\n".join(lines[-20:])
        print(preview, flush=True)
    if completed.returncode:
        raise RuntimeError(f"{label} failed: {completed.returncode}")
    if not capture_json:
        return None
    for line in reversed([x for x in completed.stdout.splitlines() if x.strip()]):
        try:
            return json.loads(line)
        except Exception:
            pass
    raise RuntimeError(f"{label} did not emit JSON")


def candidate_order(prediction: Path, limit: int) -> list[dict]:
    with np.load(prediction, allow_pickle=False) as z:
        order = np.asarray(z["target_score_descending_candidate_index"], dtype=np.int64)
        score = np.asarray(z["score"], dtype=np.float64)
        graspness = np.asarray(z["graspness"], dtype=np.float64)
        log_prob = np.asarray(z["log_prob"], dtype=np.float64)
    rows = []
    for rank, idx in enumerate(order[: max(0, limit)]):
        i = int(idx)
        rows.append({
            "target_rank": rank,
            "candidate_index": i,
            "score": float(score[i]),
            "graspness": float(graspness[i]),
            "log_prob": float(log_prob[i]),
        })
    return rows


def world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def retarget_case(
    *,
    project_root: Path,
    case_root: Path,
    case_id: str,
    candidate_index: int,
    prediction: Path,
    network_input: Path,
    capture_root: Path,
    settled_manifest: Path,
    sim_target_segmentation_id: int,
    network_python: Path,
    retarget_python: Path,
    planner_python: Path,
) -> dict:
    build = run(
        "build candidate LEAP scratch case",
        [
            network_python,
            CONTROL_ROOT / "closed_loop/scripts/build_candidate_case.py",
            "--case-id",
            case_id,
            "--case-root",
            case_root,
            "--candidate-index",
            str(candidate_index),
            "--prediction",
            prediction,
            "--network-input",
            network_input,
            "--capture-root",
            capture_root,
            "--settled-manifest",
            settled_manifest,
            "--sim-target-segmentation-id",
            str(sim_target_segmentation_id),
            "--replace",
        ],
        cwd=project_root,
        capture_json=True,
    )
    env = os.environ.copy()
    env["DGN2_CASE_ROOT"] = str(case_root)
    for label, py, script in (
        ("LEAP -> Wuji2 GRASP", retarget_python, PIPELINE_SCRIPTS / "01_retarget_grasp_official.py"),
        ("Wuji2 root 6D alignment", retarget_python, PIPELINE_SCRIPTS / "02_align_root6d.py"),
        ("Wuji2 SQUEEZE retarget", retarget_python, PIPELINE_SCRIPTS / "03_retarget_squeeze_official.py"),
        ("build final Wuji2 waypoints", network_python, PIPELINE_SCRIPTS / "05_build_isaacsim_validation.py"),
    ):
        run(label, [py, script], cwd=project_root, env=env)
    run(
        "build 5-stage arm flange targets",
        [
            planner_python,
            project_root / "08_dual_arm_scene_layout/isaaclab_control/tools/03_build_arm_execution_targets.py",
            "--case-root",
            case_root,
        ],
        cwd=project_root,
    )
    return build


def load_case_pick_contract(case_root: Path, measured: dict, T_base_from_world: np.ndarray) -> dict:
    route_path = case_root / "07_arm_execution/arm_flange_targets.npz"
    hand_path = case_root / "06_isaacsim/final_waypoints.npz"
    with np.load(route_path, allow_pickle=False) as z:
        names = [str(x) for x in z["waypoint_names"].tolist()]
        flange_world = np.asarray(z["world_from_right_flange"], dtype=np.float64)
    if names[:5] != PICK_STAGES:
        raise RuntimeError(f"{case_root}: first stages are not {PICK_STAGES}: {names[:5]}")
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
    return {
        "targets_base": np.stack([T_base_from_world @ T for T in flange_world[:5]]),
        "states": states,
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--network-input", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--settled-manifest", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--sim-target-segmentation-id", type=int, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-python", type=Path, required=True)
    parser.add_argument("--retarget-python", type=Path, required=True)
    parser.add_argument("--planner-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--candidate-case-prefix", default="closedloop")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--block-unknown", action="store_true")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    project_root = args.project_root.expanduser().resolve()
    scratch_root = args.scratch_root.expanduser().resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    capture_root = args.capture_root.expanduser().resolve()
    candidates = candidate_order(args.prediction.expanduser().resolve(), args.limit)
    if not candidates:
        raise RuntimeError("no candidates to screen")
    state = load_json(args.robot_state.expanduser().resolve())
    q_current = np.asarray(state["right_arm_q_current_rad"], dtype=np.float64)
    measured = {str(k): float(v) for k, v in state["joint_positions_by_name"].items()}
    T_world_base = world_from_base(project_root)
    T_base_from_world = np.linalg.inv(T_world_base)

    chunks = []
    selected = None
    worker_start_count = 1
    map_build_count = 0
    materialized_count = 0
    total_started = time.perf_counter()

    with CuroboWorkerClient(
        project_root,
        seeds=48,
        batch_size=max(64, args.chunk_size * len(PICK_STAGES)),
    ) as client:
        map_started = time.perf_counter()
        map_report = client.build_map(
            capture_root / "depth_m.npy",
            capture_root / "intrinsics.npy",
            capture_root / "T_world_camera.npy",
            args.mask.expanduser().resolve(),
        )
        map_wall_s = time.perf_counter() - map_started
        map_build_count += 1

        for start in range(0, len(candidates), args.chunk_size):
            chunk_started = time.perf_counter()
            chunk_items = candidates[start:start + args.chunk_size]
            chunk_cases = []
            for item in chunk_items:
                rank = int(item["target_rank"])
                idx = int(item["candidate_index"])
                case_id = f"{args.candidate_case_prefix}_r{rank:03d}_cand{idx:04d}"
                case_root = scratch_root / f"chunk_{len(chunks):03d}" / case_id
                print(
                    f"\n[LAZY MATERIALIZE] rank={rank} candidate={idx} score={item['score']:.6f}",
                    flush=True,
                )
                retarget_case(
                    project_root=project_root,
                    case_root=case_root,
                    case_id=case_id,
                    candidate_index=idx,
                    prediction=args.prediction.expanduser().resolve(),
                    network_input=args.network_input.expanduser().resolve(),
                    capture_root=capture_root,
                    settled_manifest=args.settled_manifest.expanduser().resolve(),
                    sim_target_segmentation_id=args.sim_target_segmentation_id,
                    network_python=args.network_python.expanduser().resolve(),
                    retarget_python=args.retarget_python.expanduser().resolve(),
                    planner_python=args.planner_python.expanduser().resolve(),
                )
                materialized_count += 1
                contract = load_case_pick_contract(case_root, measured, T_base_from_world)
                chunk_cases.append({"item": item, "case_root": case_root, "contract": contract})

            targets = np.concatenate([x["contract"]["targets_base"] for x in chunk_cases], axis=0)
            states = [state for x in chunk_cases for state in x["contract"]["states"]]
            phases = [phase for x in chunk_cases for phase in x["contract"]["phases"]]
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
            groups = []
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
                    "target_rank": int(case["item"]["target_rank"]),
                    "candidate_index": int(case["item"]["candidate_index"]),
                    "official_score": float(case["item"]["score"]),
                    "case_root": str(case["case_root"]),
                    "feasible": feasible,
                    "unknown_policy_pass": unknown_policy_pass,
                    **group,
                }
                groups.append(row)
                if selected is None and feasible:
                    selected = row
            chunks.append({
                "chunk_index": len(chunks),
                "score_order_start": start,
                "candidate_count": len(chunk_items),
                "pose_count": int(len(targets)),
                "materialize_and_solve_wall_s": time.perf_counter() - chunk_started,
                "solve_wall_s": solve_wall_s,
                "worker_solve_time_s": solve.get("solve_time_s"),
                "groups": groups,
            })
            if selected is not None:
                break

    total_wall_s = time.perf_counter() - total_started
    tested = sum(int(chunk["candidate_count"]) for chunk in chunks)
    out = {
        "schema_version": 1,
        "status": "PASS" if selected is not None else "FAIL",
        "scope": "pick-stage planning-only lazy batch screening",
        "SELF_COLLISION_POLICY": "REPORT_ONLY_UNRESOLVED",
        "pick_path_contract": "q_current->PREGRASP->COVER->GRASP->SQUEEZE->LIFT only; no LIFT->HOME",
        "candidate_order": "official DGN2 score descending",
        "pick_stages": PICK_STAGES,
        "selected": selected,
        "chunks": chunks,
        "metrics": {
            "worker_start_count": worker_start_count,
            "map_build_count": map_build_count,
            "chunk_size": args.chunk_size,
            "limit": args.limit,
            "materialized_candidate_count": materialized_count,
            "tested_candidate_count": tested,
            "poses_per_full_chunk": args.chunk_size * len(PICK_STAGES),
            "map_wall_s": map_wall_s,
            "total_wall_s": total_wall_s,
            "mean_wall_s_per_tested_candidate": total_wall_s / max(tested, 1),
        },
        "map": map_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "selected_candidate_index": None if selected is None else selected["candidate_index"],
        "worker_start_count": worker_start_count,
        "map_build_count": map_build_count,
        "chunk_size": args.chunk_size,
        "tested_candidate_count": tested,
        "materialized_candidate_count": materialized_count,
        "poses_per_full_chunk": args.chunk_size * len(PICK_STAGES),
        "mean_wall_s_per_tested_candidate": out["metrics"]["mean_wall_s_per_tested_candidate"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
