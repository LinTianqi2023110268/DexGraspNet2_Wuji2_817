#!/usr/bin/env python3
"""Regression check for a known-good Route-C V2 five-stage cuRobo IK case.

This script intentionally does not build DGN2 candidates, run retargeting, or
materialize new 01_cases entries.  It replays the archived candidate3800
PREGRASP/COVER/GRASP/SQUEEZE/LIFT flange targets against the current core
worker so we can separate IK/base/tool-frame regressions from closed-loop
candidate-screening architecture issues.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def matrix_max_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--case-root", type=Path, default=Path(
        "06_leap_to_wuji2_final_pipeline/99_archive/route_c_provenance/"
        "live_dynamic_scene0000_dog_candidate3800"
    ))
    ap.add_argument("--capture-root", type=Path, default=Path(
        "08_dual_arm_scene_layout/captures/live_dynamic_scene0000"
    ))
    ap.add_argument("--old-report", type=Path, default=Path(
        "08_dual_arm_scene_layout/isaaclab_control/outputs/"
        "full_pick_place_25s_dog_candidate3800/route_c_v2_planning.json"
    ))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    project_root = args.project_root.expanduser().resolve()
    control_root = project_root / "08_dual_arm_scene_layout/isaaclab_control"
    sys.path.insert(0, str(control_root))

    from core.bridge import CuroboWorkerClient

    case_root = (project_root / args.case_root).resolve() if not args.case_root.is_absolute() else args.case_root.resolve()
    capture_root = (project_root / args.capture_root).resolve() if not args.capture_root.is_absolute() else args.capture_root.resolve()
    old_report_path = (project_root / args.old_report).resolve() if not args.old_report.is_absolute() else args.old_report.resolve()

    old = load_json(old_report_path)
    stages = ["pregrasp", "cover", "grasp", "squeeze", "lift"]
    old_stage_names = list(old["stage_names"][:5])
    if old_stage_names != stages:
        raise RuntimeError(f"old report first five stages mismatch: {old_stage_names}")

    q_current = np.asarray(old["q_current_rad"], dtype=np.float64)
    old_raw = list(old["solve"]["raw_success_per_target"][:5])
    old_accepted = list(old["solve"]["accepted_per_target"][:5])
    old_ik_accepted = list(old["solve"].get("ik_accepted_per_target", old_accepted)[:5])

    target_npz = case_root / "07_arm_execution/arm_flange_targets.npz"
    with np.load(target_npz, allow_pickle=False) as z:
        waypoint_names = [str(x) for x in z["waypoint_names"].tolist()]
        world_from_flange = np.asarray(z["world_from_right_flange"], dtype=np.float64)
    if waypoint_names != stages:
        raise RuntimeError(f"archived flange stages mismatch: {waypoint_names}")

    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    world_from_base = np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"], dtype=np.float64
    ).T
    base_from_world = np.linalg.inv(world_from_base)
    base_from_flange = np.stack([base_from_world @ T for T in world_from_flange])

    # Compare against old selected_collision stage target provenance where possible:
    old_targets = []
    for item in old["solve"].get("selected_collision", [])[:5]:
        old_targets.append({
            "target_index": item["target_index"],
            "phase": item.get("phase"),
            "position_error_m": item.get("position_error_m"),
            "orientation_error_rad": item.get("orientation_error_rad"),
            "inner_limit_margin_rad": item.get("inner_limit_margin_rad"),
        })

    t0 = time.perf_counter()
    with CuroboWorkerClient(project_root, seeds=48, batch_size=64) as client:
        worker_started_s = time.perf_counter() - t0
        solve_t0 = time.perf_counter()
        ik_only = client.solve_ik(base_from_flange, q_current, select_chain=True)
        solve_wall_s = time.perf_counter() - solve_t0
    total_wall_s = time.perf_counter() - t0

    out = {
        "schema_version": 1,
        "status": "PASS" if all(int(x) > 0 for x in ik_only["raw_success_per_target"]) else "FAIL",
        "scope": "known-good candidate3800 five-stage pure IK regression",
        "case_root": str(case_root),
        "capture_root": str(capture_root),
        "old_report": str(old_report_path),
        "stages": stages,
        "q_current_rad": q_current.tolist(),
        "old": {
            "raw_success_per_target": old_raw,
            "accepted_per_target": old_accepted,
            "ik_accepted_per_target": old_ik_accepted,
            "selected_collision_summary": old_targets,
        },
        "current": {
            "raw_success_per_target": ik_only["raw_success_per_target"],
            "accepted_per_target": ik_only["accepted_per_target"],
            "ik_accepted_per_target": ik_only.get("ik_accepted_per_target"),
            "ik_pass": ik_only.get("ik_pass"),
            "solve_time_s": ik_only.get("solve_time_s"),
            "selected": ik_only.get("selected"),
        },
        "target_contract": {
            "base": "arm_base_link",
            "tool": "arm_r_link_tf",
            "input_targets": "base_from_right_flange 4x4, first five stages only",
            "world_from_base": world_from_base.tolist(),
            "base_from_flange": base_from_flange.tolist(),
            "max_abs_self_delta_base_from_flange": matrix_max_abs_delta(base_from_flange, base_from_flange),
        },
        "timing": {
            "worker_startup_plus_ping_wall_s": worker_started_s,
            "solve_request_wall_s": solve_wall_s,
            "total_wall_s": total_wall_s,
            "worker_start_count": 1,
            "map_build_count": 0,
            "pose_count": int(base_from_flange.shape[0]),
        },
        "regression_gate": {
            "old_raw_all_positive": all(int(x) > 0 for x in old_raw),
            "current_raw_all_positive": all(int(x) > 0 for x in ik_only["raw_success_per_target"]),
            "stop_if_current_raw_all_zero": all(int(x) == 0 for x in ik_only["raw_success_per_target"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "old_raw": old_raw,
        "current_raw": out["current"]["raw_success_per_target"],
        "old_accepted": old_accepted,
        "current_accepted": out["current"]["accepted_per_target"],
        "worker_start_count": 1,
        "map_build_count": 0,
        "pose_count": int(base_from_flange.shape[0]),
        "solve_request_wall_s": solve_wall_s,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0 if out["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
