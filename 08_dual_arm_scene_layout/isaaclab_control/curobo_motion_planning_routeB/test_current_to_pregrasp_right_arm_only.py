#!/usr/bin/env python3
"""Standalone Route B true right-arm-only current -> PREGRASP test."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
THIS_DIR = Path(__file__).resolve().parent
RIGHT_ARM_CORE_ROOT = THIS_DIR / "routeB_right_arm_only_core_v1"
for path in (ISAACLAB_CONTROL_ROOT, THIS_DIR, RIGHT_ARM_CORE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter
from curobo_motion_planning_routeB.routeB_adapter import (
    DEFAULT_ENABLE_GRAPH_ATTEMPT,
    DEFAULT_LAYOUT_JSON,
    DEFAULT_ROBOT_FILE,
)
from right_arm_only_core import RIGHT_ARM_JOINTS, plan_right_arm_only
from core.perception_collision.esdf_collision import query_spheres
from core.perception_collision.robot_spheres import CuroboRobotSphereModel
from test_trajopt_feasibility_audit import _metrics_audit


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_pregrasp_q(route_plan_path: Path) -> np.ndarray:
    with np.load(route_plan_path, allow_pickle=True) as data:
        names = [str(x) for x in data["waypoint_names"].tolist()]
        if "pregrasp" not in names:
            raise KeyError(f"{route_plan_path} does not contain a pregrasp waypoint")
        return np.asarray(data["arm_q_rad"][names.index("pregrasp")], dtype=np.float32)


def tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def constraint_summary(raw_result: Any, planner: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scene_collision": {"max_value": None, "positive_count": None},
        "cspace": {"max_value": None, "positive_count": None},
        "failed_constraints": [],
    }
    try:
        solution = getattr(raw_result, "solution", None)
        if solution is None:
            raise RuntimeError("raw_result.solution is unavailable")
        action = solution.reshape(-1, solution.shape[-2], solution.shape[-1])
        expected = int(getattr(planner.trajopt_solver.config, "num_seeds", action.shape[0]))
        if action.shape[0] == 1 and expected > 1:
            action = action.repeat(expected, 1, 1)
        raw_metrics = planner.trajopt_solver.metrics_rollout.compute_metrics_from_action(action)
        raw_report = _metrics_audit(raw_metrics)
        interp_report = {"present": False, "reason": "right-arm-only postcheck uses raw 7DOF optimizer metrics"}
        out["raw_metrics"] = json_safe(raw_report)
        out["interpolated_metrics"] = json_safe(interp_report)
        for report in (raw_report, interp_report):
            for item in report.get("constraints", []) + report.get("hybrid_constraints", []):
                name = item.get("name")
                if name not in {"scene_collision", "cspace"}:
                    continue
                current = out[name]
                max_value = item.get("max_value")
                positive_count = item.get("positive_count")
                if max_value is not None and (
                    current.get("max_value") is None or float(max_value) > float(current["max_value"])
                ):
                    out[name] = {
                        "shape": item.get("shape"),
                        "max_value": float(max_value),
                        "sum_value": item.get("sum_value"),
                        "positive_count": int(positive_count or 0),
                    }
                if positive_count:
                    out["failed_constraints"].append(name)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["failed_constraints"] = sorted(set(out["failed_constraints"]))
    return out


def full_q_from_right(adapter: RouteBMotionPlannerAdapter, q_by_name: dict[str, float], q_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_current = adapter._coerce_q(q_by_name)
    q_goal = adapter._coerce_q(q_right, base_q=q_current)
    return q_current, q_goal


def trajectory_postcheck_factory(
    *,
    scene,
    full_joint_names: list[str],
    adapter: RouteBMotionPlannerAdapter,
    device: str,
):
    sphere_model = CuroboRobotSphereModel(adapter.robot_file, device=device)
    full_index = {name: i for i, name in enumerate(full_joint_names)}

    def postcheck(traj, contract, planner, raw_result):
        full_q = []
        for q_right in traj.q_rad:
            q = np.zeros(len(full_joint_names), dtype=np.float32)
            for name, value in contract.lock_joints.items():
                q[full_index[name]] = float(value)
            for value, joint_name in zip(q_right, RIGHT_ARM_JOINTS):
                q[full_index[joint_name]] = float(value)
            full_q.append(q)
        full_q_arr = np.asarray(full_q, dtype=np.float32)

        min_clearance = float("inf")
        collision_count = 0
        worst = None
        for t, q_t in enumerate(full_q_arr):
            q_map = {name: float(q_t[i]) for i, name in enumerate(full_joint_names)}
            spheres = sphere_model.spheres_from_named_joints(q_map)
            batch = query_spheres(scene.voxel[0], spheres[:, :3], spheres[:, 3], margin_m=0.0)
            clearance = np.asarray(batch.distance_m, dtype=np.float64) - spheres[:, 3]
            i = int(np.argmin(clearance))
            if float(clearance[i]) < min_clearance:
                min_clearance = float(clearance[i])
                worst = {
                    "timestep": int(t),
                    "sphere_index": i,
                    "link_name": sphere_model.sphere_link_names[i]
                    if i < len(sphere_model.sphere_link_names)
                    else None,
                    "clearance_m": float(clearance[i]),
                }
            if bool(np.any(batch.collision)):
                collision_count += 1

        lower, upper = adapter._motion_planner_position_bounds_for(planner)
        q_active = np.asarray(traj.q_rad, dtype=np.float64)
        lower = np.asarray(lower, dtype=np.float64)[: q_active.shape[1]]
        upper = np.asarray(upper, dtype=np.float64)[: q_active.shape[1]]
        joint_violation = np.maximum(lower.reshape(1, -1) - q_active, q_active - upper.reshape(1, -1))
        joint_limit_pass = bool(np.count_nonzero(joint_violation > 0.0) == 0)

        dynamics = {
            "velocity_max_abs_rad_s": float(np.max(np.abs(traj.qd_rad_s))),
            "acceleration_max_abs_rad_s2": float(np.max(np.abs(traj.qdd_rad_s2))),
            "jerk_max_abs_rad_s3": float(np.max(np.abs(traj.jerk_rad_s3))),
            "velocity_pass": bool(np.isfinite(traj.qd_rad_s).all()),
            "acceleration_pass": bool(np.isfinite(traj.qdd_rad_s2).all()),
            "jerk_pass": bool(np.isfinite(traj.jerk_rad_s3).all()),
        }
        constraints = constraint_summary(raw_result, planner)
        return {
            "environment_collision": bool(collision_count > 0),
            "environment_collision_pass": bool(collision_count == 0),
            "min_environment_clearance_m": float(min_clearance),
            "environment_collision_sample_count": int(collision_count),
            "environment_worst_sample": worst,
            "scene_collision_max": constraints.get("scene_collision", {}).get("max_value"),
            "scene_collision_positive_count": constraints.get("scene_collision", {}).get("positive_count"),
            "cspace_max": constraints.get("cspace", {}).get("max_value"),
            "cspace_positive_count": constraints.get("cspace", {}).get("positive_count"),
            "constraint_summary": constraints,
            "joint_limit_pass": joint_limit_pass,
            "joint_limit_violation_count": int(np.count_nonzero(joint_violation > 0.0)),
            "joint_limit_worst_violation_rad": float(np.max(joint_violation)) if joint_violation.size else 0.0,
            **dynamics,
        }

    return postcheck


def run(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    route_plan = Path(args.route_plan).expanduser().resolve()
    output_dir = (
        capture_dir / "curobo_test_result"
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "filtered_depth": capture_dir / "planning/filtered_depth.npy",
        "intrinsics": capture_dir / "intrinsics.npy",
        "T_world_camera": capture_dir / "T_world_camera.npy",
        "robot_state": capture_dir / "robot_state.json",
        "route_plan": route_plan,
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Route B right-arm-only test inputs: " + ", ".join(missing))

    cfg = {
        "routeB": {
            "device": args.device,
            "robot_file": str(args.robot_file),
            "layout_json": str(args.layout_json),
            "collision": {"environment_collision": True, "self_collision": False},
            "use_cuda_graph": args.use_cuda_graph,
            "num_ik_seeds": args.num_ik_seeds,
            "num_trajopt_seeds": args.num_trajopt_seeds,
            "max_attempts": args.max_attempts,
            "enable_graph_attempt": args.enable_graph_attempt,
            "warmup_iterations": args.warmup_iterations,
            "interpolation_dt_s": args.interpolation_dt_s,
        }
    }
    adapter = RouteBMotionPlannerAdapter(cfg)
    scene = adapter.build_pick_scene(inputs["filtered_depth"], inputs["intrinsics"], inputs["T_world_camera"])

    # Use the existing full-DOF adapter only to obtain the canonical full joint
    # order and the already-validated numerical-bound planning state. It does
    # not produce a 35DOF trajectory here.
    adapter.create_planner(scene)
    q_pregrasp = load_pregrasp_q(route_plan)
    robot_state = load_json(inputs["robot_state"])
    q_current_raw, q_goal_raw = full_q_from_right(
        adapter,
        robot_state["joint_positions_by_name"],
        q_pregrasp,
    )
    q_current_planning, q_pregrasp_planning, sanitization_report = adapter.sanitize_planning_joint_states(
        q_current_raw,
        q_goal_raw,
    )
    full_joint_names = list(adapter.joint_names)
    robot_source = load_yaml(adapter.robot_file)

    planner_reports: dict[str, Any] = {}

    def planner_factory(locked_robot_cfg):
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg

        device_cfg = DeviceCfg(device=torch.device(adapter.device), dtype=torch.float32)
        motion_cfg = MotionPlannerCfg.create(
            robot=locked_robot_cfg,
            scene_model=scene,
            device_cfg=device_cfg,
            self_collision_check=adapter.self_collision_check,
            num_ik_seeds=adapter.num_ik_seeds,
            num_trajopt_seeds=adapter.num_trajopt_seeds,
            use_cuda_graph=adapter.use_cuda_graph,
            interpolation_dt=adapter.interpolation_dt_s,
        )
        planner_reports["collision_policy"] = adapter._apply_collision_policy_to_motion_cfg(motion_cfg)
        planner = MotionPlanner(motion_cfg)
        planner_reports["voxel_shape_contract"] = adapter._normalize_voxel_shape_contract(planner, scene)
        planner.warmup(
            enable_graph=adapter.use_cuda_graph,
            num_warmup_iterations=adapter.warmup_iterations,
        )
        return planner

    def planner_fixup(_planner):
        return {
            **planner_reports,
            "environment_collision": True,
            "self_collision": False,
            "note": "collision policy and voxel shape fixups are applied in planner_factory before warmup",
        }

    t0 = time.time()
    result = plan_right_arm_only(
        robot_source=robot_source,
        full_joint_names=full_joint_names,
        q_current_planning=q_current_planning,
        q_pregrasp_planning=q_pregrasp_planning,
        planner_factory=planner_factory,
        planner_fixup=planner_fixup,
        trajectory_postcheck=trajectory_postcheck_factory(
            scene=scene,
            full_joint_names=full_joint_names,
            adapter=adapter,
            device=args.device,
        ),
        output_dir=output_dir,
        max_attempts=args.max_attempts,
        enable_graph_attempt=args.enable_graph_attempt,
    )
    wall = time.time() - t0

    report = load_json(result.report_path)
    report.update(
        {
            "capture_dir": str(capture_dir),
            "route_plan": str(route_plan),
            "robot_state": str(inputs["robot_state"]),
            "q_current_source": "capture/robot_state.json",
            "q_pregrasp_source": "flexible_route_plan.npz:arm_q_rad[pregrasp]",
            "joint_state_sanitization": sanitization_report,
            "routeB_adapter_scene_report": adapter.last_scene_report,
            "raw_result_solution_shape": [
                int(x) for x in getattr(result.raw_result, "solution").shape
            ],
            "total_wall_time_s": float(wall),
        }
    )
    result.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[Route B right-arm-only current->PREGRASP]")
    print(f"success={report['success']}")
    print(f"action_dim={report['planner']['action_dim']}")
    print(f"active_joints={report['planner']['active_joint_names']}")
    print(f"trajectory_points={report['trajectory']['point_count']}")
    print(f"min_environment_clearance_m={report['postcheck']['min_environment_clearance_m']:.6f}")
    print(f"trajectory={result.trajectory_path}")
    print(f"report={result.report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--route-plan", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--robot-file", default=str(DEFAULT_ROBOT_FILE))
    parser.add_argument("--layout-json", default=str(DEFAULT_LAYOUT_JSON))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--num-ik-seeds", type=int, default=32)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--enable-graph-attempt", type=int, default=DEFAULT_ENABLE_GRAPH_ATTEMPT)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--interpolation-dt-s", type=float, default=0.025)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
