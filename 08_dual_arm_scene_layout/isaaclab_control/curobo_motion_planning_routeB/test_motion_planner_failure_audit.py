#!/usr/bin/env python3
"""Diagnose why Route B MotionPlanner current->PREGRASP returns success=false.

Diagnostic only:
- no Route A changes
- no parameter tuning
- no Isaac launch/execution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
if str(ISAACLAB_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_CONTROL_ROOT))

from core.perception_collision.esdf_collision import query_spheres
from core.perception_collision.robot_spheres import CuroboRobotSphereModel
from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter
from curobo_motion_planning_routeB.routeB_adapter import DEFAULT_ENABLE_GRAPH_ATTEMPT
from test_current_to_pregrasp import load_json, load_pregrasp_q


def _tensor_summary(value: Any, *, max_items: int = 16) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
        flat = arr.reshape(-1)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "values": flat[:max_items].tolist(),
            "min": float(np.nanmin(flat)) if flat.size else None,
            "max": float(np.nanmax(flat)) if flat.size else None,
        }
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _count_success(result: Any) -> int | None:
    success = getattr(result, "success", None)
    if success is None:
        return None
    try:
        return int(success.detach().cpu().bool().sum().item())
    except Exception:
        return None


def _shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return [int(x) for x in value.shape]
    return None


def _joint_state_from_q(planner, q: np.ndarray):
    import torch
    from curobo.types import JointState

    return JointState.from_position(
        torch.as_tensor(q, device=planner.device_cfg.device, dtype=torch.float32).reshape(1, -1),
        joint_names=planner.joint_names,
    )


def _result_report(result: Any) -> dict[str, Any]:
    if result is None:
        return {"present": False}
    fields = [name for name in dir(result) if not name.startswith("_")]
    report = {
        "present": True,
        "type": f"{type(result).__module__}.{type(result).__name__}",
        "available_result_fields": fields,
        "success": _tensor_summary(getattr(result, "success", None)),
        "success_count": _count_success(result),
        "solve_time": _tensor_summary(getattr(result, "solve_time", None)),
        "total_time": _tensor_summary(getattr(result, "total_time", None)),
        "debug_info": _tensor_summary(getattr(result, "debug_info", None)),
        "position_error": _tensor_summary(getattr(result, "position_error", None)),
        "rotation_error": _tensor_summary(getattr(result, "rotation_error", None)),
        "feasible": _tensor_summary(getattr(result, "feasible", None)),
        "cspace_error": _tensor_summary(getattr(result, "cspace_error", None)),
        "seed_cost": _tensor_summary(getattr(result, "seed_cost", None)),
        "total_cost_reshaped": _tensor_summary(getattr(result, "total_cost_reshaped", None)),
        "solution_shape": _shape(getattr(result, "solution", None)),
        "optimized_seeds_shape": _shape(getattr(result, "optimized_seeds", None)),
    }
    js = getattr(result, "js_solution", None)
    report["js_solution"] = {
        "present": js is not None,
        "position_shape": _shape(getattr(js, "position", None)) if js is not None else None,
        "velocity_shape": _shape(getattr(js, "velocity", None)) if js is not None else None,
        "acceleration_shape": _shape(getattr(js, "acceleration", None)) if js is not None else None,
        "jerk_shape": _shape(getattr(js, "jerk", None)) if js is not None else None,
        "dt": _tensor_summary(getattr(js, "dt", None)) if js is not None else None,
    }
    report["metrics"] = _metrics_report(getattr(result, "metrics", None))
    report["interpolated_metrics"] = _metrics_report(getattr(result, "interpolated_metrics", None))
    return report


def _collection_report(collection: Any) -> dict[str, Any] | None:
    if collection is None:
        return None
    names = list(getattr(collection, "names", []) or [])
    values = list(getattr(collection, "values", []) or [])
    out = []
    for name, value in zip(names, values):
        summary = _tensor_summary(value)
        item = {"name": str(name)}
        if isinstance(summary, dict):
            item.update(summary)
        else:
            item["value"] = summary
        out.append(item)
    return {"names": names, "items": out}


def _metrics_report(metrics: Any) -> dict[str, Any] | None:
    if metrics is None:
        return None
    cc = getattr(metrics, "costs_and_constraints", None)
    report: dict[str, Any] = {
        "type": f"{type(metrics).__module__}.{type(metrics).__name__}",
        "feasible_attr": _tensor_summary(getattr(metrics, "feasible", None)),
        "convergence": _collection_report(getattr(metrics, "convergence", None)),
    }
    if cc is not None:
        report["costs"] = _collection_report(getattr(cc, "costs", None))
        report["constraints"] = _collection_report(getattr(cc, "constraints", None))
        report["hybrid_costs_constraints"] = _collection_report(
            getattr(cc, "hybrid_costs_constraints", None)
        )
        try:
            report["feasible_sum_horizon"] = _tensor_summary(
                cc.get_feasible(include_all_hybrid=False, sum_horizon=True)
            )
        except Exception as exc:
            report["feasible_sum_horizon_error"] = f"{type(exc).__name__}: {exc}"
        try:
            report["constraint_sum_horizon"] = _tensor_summary(
                cc.get_sum_constraint(include_all_hybrid=False, sum_horizon=True)
            )
        except Exception as exc:
            report["constraint_sum_horizon_error"] = f"{type(exc).__name__}: {exc}"
    return report


def _full_q_from_right(planner, q_current_by_name: dict[str, float], q_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current_q = planner.default_joint_state.position.detach().cpu().numpy().reshape(-1).astype(np.float32)
    name_to_i = {str(name): i for i, name in enumerate(planner.joint_names)}
    for name, value in q_current_by_name.items():
        if name in name_to_i:
            current_q[name_to_i[name]] = float(value)
    goal_q = current_q.copy()
    for j, value in enumerate(np.asarray(q_right, dtype=np.float32).reshape(-1)):
        name = f"arm_r_joint_{j + 1}"
        goal_q[name_to_i[name]] = float(value)
    return current_q, goal_q


def _linear_path_audit(
    *,
    model: CuroboRobotSphereModel,
    scene_grid: Any,
    joint_names: list[str],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    sample_count: int,
) -> dict[str, Any]:
    min_clearance = float("inf")
    min_index = None
    first_collision = None
    collision_count = 0
    worst = None
    for i, alpha in enumerate(np.linspace(0.0, 1.0, int(sample_count))):
        q = (1.0 - alpha) * q_start + alpha * q_goal
        q_by_name = {name: float(q[j]) for j, name in enumerate(joint_names)}
        spheres = model.spheres_from_named_joints(q_by_name)
        batch = query_spheres(scene_grid, spheres[:, :3], spheres[:, 3], margin_m=0.0)
        clearance = np.asarray(batch.distance_m, dtype=np.float64) - spheres[:, 3]
        local_min_i = int(np.argmin(clearance))
        local_min = float(clearance[local_min_i])
        if local_min < min_clearance:
            min_clearance = local_min
            min_index = i
            worst = {
                "sample_index": i,
                "alpha": float(alpha),
                "sphere_index": local_min_i,
                "link_name": (
                    model.sphere_link_names[local_min_i]
                    if local_min_i < len(model.sphere_link_names)
                    else None
                ),
                "sphere_radius_m": float(spheres[local_min_i, 3]),
                "esdf_signed_distance_m": float(batch.distance_m[local_min_i]),
                "clearance_m": local_min,
                "inside_grid": bool(batch.inside_grid[local_min_i]),
                "collision": bool(batch.collision[local_min_i]),
            }
        if bool(np.any(batch.collision)):
            collision_count += 1
            if first_collision is None:
                idx = int(np.nonzero(batch.collision)[0][0])
                first_collision = {
                    "sample_index": i,
                    "alpha": float(alpha),
                    "sphere_index": idx,
                    "link_name": model.sphere_link_names[idx] if idx < len(model.sphere_link_names) else None,
                    "sphere_radius_m": float(spheres[idx, 3]),
                    "esdf_signed_distance_m": float(batch.distance_m[idx]),
                    "clearance_m": float(clearance[idx]),
                }
    return {
        "sample_count": int(sample_count),
        "collision_free": first_collision is None,
        "first_collision_index": None if first_collision is None else first_collision["sample_index"],
        "first_collision": first_collision,
        "collision_sample_count": int(collision_count),
        "min_environment_clearance_m": float(min_clearance),
        "min_clearance_index": min_index,
        "worst_sample": worst,
    }


def _run_graph(planner, q_start: np.ndarray, q_goal: np.ndarray) -> dict[str, Any]:
    graph = getattr(planner, "graph_planner", None)
    if graph is None:
        return {
            "configured": False,
            "instance_present": False,
            "success": False,
            "success_count": 0,
            "waypoint_count": 0,
            "status": "GRAPH_PLANNER_NOT_CONFIGURED",
        }
    import torch
    from curobo._src.util.trajectory import TrajInterpolationType

    x_start = torch.as_tensor(q_start, device=planner.device_cfg.device, dtype=torch.float32).reshape(1, -1)
    x_goal = torch.as_tensor(q_goal, device=planner.device_cfg.device, dtype=torch.float32).reshape(1, -1)
    t0 = time.time()
    result = graph.find_path(
        x_start,
        x_goal,
        interpolate_waypoints=True,
        interpolation_steps=planner.trajopt_solver.action_horizon,
        interpolation_type=TrajInterpolationType.LINEAR,
        validate_interpolated_trajectory=False,
    )
    graph_time = time.time() - t0
    success_count = _count_success(result) or 0
    waypoint_count = 0
    if getattr(result, "interpolated_waypoints", None) is not None:
        waypoint_count = int(result.interpolated_waypoints.shape[-2])
    elif getattr(result, "plan_waypoints", None):
        first = next((x for x in result.plan_waypoints if x is not None), None)
        if first is not None and hasattr(first, "shape"):
            waypoint_count = int(first.shape[-2]) if len(first.shape) > 1 else int(first.shape[0])
    return {
        "configured": True,
        "instance_present": True,
        "success": bool(success_count > 0),
        "success_count": int(success_count),
        "graph_time_s": float(graph_time),
        "waypoint_count": int(waypoint_count),
        "debug_info": _tensor_summary(getattr(result, "debug_info", None)),
        "valid_query": _tensor_summary(getattr(result, "valid_query", None)),
        "raw_result": _result_report(result),
    }


def _attempt_audit(planner, current_state, goal_state, max_attempts: int, enable_graph_attempt: int) -> tuple[list[dict[str, Any]], Any]:
    import torch

    attempts = []
    final_result = None
    total_time = 0.0
    solve_time = 0.0
    num_seeds = int(planner.trajopt_solver.config.num_seeds)
    og_current = current_state.clone()
    for attempt_i in range(int(max_attempts)):
        cur = og_current.clone()
        seed_traj = None
        graph_requested = bool(attempt_i >= int(enable_graph_attempt))
        graph_report = None
        if graph_requested and getattr(planner, "graph_planner", None) is not None:
            goal_configs = goal_state.position.view(1, 1, -1).repeat(1, num_seeds, 1)
            graph_report = _run_graph(planner, cur.position.reshape(-1).detach().cpu().numpy(), goal_configs[0, 0].detach().cpu().numpy())
            if graph_report["success"]:
                seed_config = goal_configs
                graph_starts = cur.position.view(1, planner.trajopt_solver.action_dim).repeat(num_seeds, 1)
                graph_goals = seed_config.view(num_seeds, planner.trajopt_solver.action_dim)
                result = planner.graph_planner.find_path(
                    graph_starts.clone(),
                    graph_goals.clone(),
                    interpolate_waypoints=True,
                    interpolation_steps=planner.trajopt_solver.action_horizon,
                    validate_interpolated_trajectory=False,
                )
                if torch.count_nonzero(result.success) > 0:
                    seed_traj = result.interpolated_waypoints[result.success, :, :].unsqueeze(0)

        t0 = time.time()
        result = planner.trajopt_solver.solve_cspace(
            goal_state,
            cur,
            seed_traj=seed_traj,
            finetune_attempts=3,
            finetune_dt_scale=0.75,
        )
        wall = time.time() - t0
        total_time += float(getattr(result, "total_time", 0.0) or 0.0)
        solve_time += float(getattr(result, "solve_time", 0.0) or 0.0)
        success_count = _count_success(result) or 0
        attempts.append(
            {
                "attempt": int(attempt_i),
                "source": "graph_seed_plus_trajopt" if seed_traj is not None else "trajopt",
                "graph_requested": graph_requested,
                "graph_available": getattr(planner, "graph_planner", None) is not None,
                "graph_used": seed_traj is not None,
                "graph_seed_generated": seed_traj is not None,
                "graph_success_count": None if graph_report is None else graph_report.get("success_count"),
                "graph_report": graph_report,
                "trajopt_success_count": int(success_count),
                "trajopt_total_seeds": int(num_seeds),
                "wall_time_s": float(wall),
                "solve_time_s": _tensor_summary(getattr(result, "solve_time", None)),
                "trajopt_result": _result_report(result),
            }
        )
        final_result = result
        if success_count > 0:
            break
    if final_result is not None:
        final_result.total_time = total_time
        final_result.solve_time = solve_time
    return attempts, final_result


def _diagnose(report: dict[str, Any]) -> str:
    attempts = report["attempts"]
    graph = report["graph_planner"]
    linear = report["linear_joint_path"]
    if not report["planner_config"]["graph_planner_instance_present"]:
        return "A: graph planner is not configured/instantiated; only TrajOpt attempts ran."
    if not any(a["graph_requested"] for a in attempts):
        if linear["collision_free"]:
            return "D/E: ordinary TrajOpt failed and graph fallback did not occur under current enable_graph_attempt; linear path is ESDF-free, so optimizer/config constraints are most likely."
        return "D/E: ordinary TrajOpt failed and graph fallback did not occur; linear path collides with ESDF, so graph or stronger trajectory optimization is needed."
    if any(a["graph_requested"] for a in attempts) and not graph["success"]:
        return "B: graph planner is present/requested but did not find a path."
    if graph["success"] and not any((a.get("trajopt_success_count") or 0) > 0 for a in attempts):
        return "C/E: graph found a path/seed, but TrajOpt failed to produce a successful trajectory."
    return "E: TrajOpt returned candidate data but success=false; see feasibility/convergence metrics."


def run(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    route_plan = Path(args.route_plan).expanduser().resolve()
    output_dir = capture_dir / "curobo_test_result"
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
        raise FileNotFoundError("missing motion planner audit inputs: " + ", ".join(missing))

    robot_state = load_json(inputs["robot_state"])
    q_pregrasp_right = load_pregrasp_q(route_plan)
    adapter = RouteBMotionPlannerAdapter(
        {
            "routeB": {
                "device": args.device,
                "collision": {"environment_collision": True, "self_collision": False},
                "max_attempts": args.max_attempts,
                "enable_graph_attempt": args.enable_graph_attempt,
                "num_ik_seeds": args.num_ik_seeds,
                "num_trajopt_seeds": args.num_trajopt_seeds,
            }
        }
    )
    scene = adapter.build_pick_scene(inputs["filtered_depth"], inputs["intrinsics"], inputs["T_world_camera"])
    planner = adapter.create_planner(scene)
    q_current, q_goal = _full_q_from_right(planner, robot_state["joint_positions_by_name"], q_pregrasp_right)
    current_state = _joint_state_from_q(planner, q_current)
    goal_state = _joint_state_from_q(planner, q_goal)

    attempts, final_result = _attempt_audit(
        planner,
        current_state,
        goal_state,
        args.max_attempts,
        args.enable_graph_attempt,
    )
    graph_report = _run_graph(planner, q_current, q_goal)
    model = CuroboRobotSphereModel(adapter.robot_file, device=args.device)
    linear_report = _linear_path_audit(
        model=model,
        scene_grid=scene.voxel[0],
        joint_names=list(planner.joint_names),
        q_start=q_current,
        q_goal=q_goal,
        sample_count=args.linear_samples,
    )

    planner_config = {
        "max_attempts": int(args.max_attempts),
        "enable_graph_attempt": int(args.enable_graph_attempt),
        "graph_planner_config_present": planner.config.graph_planner_config is not None,
        "graph_planner_instance_present": getattr(planner, "graph_planner", None) is not None,
        "trajopt_num_seeds": int(planner.trajopt_solver.config.num_seeds),
        "trajopt_action_horizon": int(planner.trajopt_solver.action_horizon),
        "interpolation_dt": float(planner.trajopt_solver.config.interpolation_dt),
        "self_collision_disabled": bool(
            adapter.last_collision_policy_report.get("all_self_collision_rollouts_disabled")
        ),
        "scene_collision_enabled": bool(
            adapter.last_collision_policy_report.get("environment_scene_collision_cfg_present")
        ),
        "collision_policy_report": adapter.last_collision_policy_report,
        "scene_report": adapter.last_scene_report,
    }
    report = {
        "schema_version": 1,
        "route": "RouteB",
        "audit": "motion_planner_current_to_pregrasp_failure",
        "inputs": {k: str(v) for k, v in inputs.items()},
        "planner_config": planner_config,
        "attempts": attempts,
        "graph_planner": graph_report,
        "linear_joint_path": linear_report,
        "trajopt": _result_report(final_result),
    }
    report["diagnosis"] = _diagnose(report)

    out = output_dir / "motion_planner_failure_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[Route B MotionPlanner failure audit]")
    print(f"graph_instance_present={planner_config['graph_planner_instance_present']}")
    print(f"attempt_count={len(attempts)}")
    print(f"enable_graph_attempt={args.enable_graph_attempt}")
    print(f"graph_success={graph_report.get('success')}")
    print(f"linear_collision_free={linear_report['collision_free']}")
    print(f"trajopt_success_count={report['trajopt'].get('success_count')}")
    print(f"diagnosis={report['diagnosis']}")
    print(f"report={out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--route-plan", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--enable-graph-attempt", type=int, default=DEFAULT_ENABLE_GRAPH_ATTEMPT)
    parser.add_argument("--num-ik-seeds", type=int, default=32)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--linear-samples", type=int, default=151)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
