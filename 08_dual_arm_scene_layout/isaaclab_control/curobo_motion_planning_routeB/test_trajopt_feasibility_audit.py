#!/usr/bin/env python3
"""Audit the exact feasibility constraint behind Route B TrajOpt success=false.

Diagnostic only:
- no Route A changes
- no Route B adapter changes
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
from test_motion_planner_failure_audit import _full_q_from_right, _joint_state_from_q


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _float_or_none(value: Any) -> float | None:
    arr = _to_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def _tensor_shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return [int(x) for x in value.shape]
    return None


def _success_count(result: Any) -> int:
    arr = _to_numpy(getattr(result, "success", None))
    if arr is None:
        return 0
    return int(np.asarray(arr, dtype=bool).sum())


def _constraint_items(collection: Any, *, sphere_link_names: list[str] | None = None) -> list[dict[str, Any]]:
    if collection is None:
        return []
    names = list(getattr(collection, "names", []) or [])
    values = list(getattr(collection, "values", []) or [])
    out: list[dict[str, Any]] = []
    for name, value in zip(names, values):
        arr = _to_numpy(value)
        if arr is None:
            out.append({"name": str(name), "available": False})
            continue
        arr = np.asarray(arr, dtype=np.float64)
        flat = arr.reshape(-1)
        positive = flat > 0.0
        if flat.size:
            worst_flat = int(np.nanargmax(flat))
            worst_unravel = list(np.unravel_index(worst_flat, arr.shape))
        else:
            worst_flat = None
            worst_unravel = None
        first_pos = None
        if np.any(positive):
            pos_flat = int(np.nonzero(positive)[0][0])
            first_pos = list(np.unravel_index(pos_flat, arr.shape))
        out.append(
            {
                "name": str(name),
                "available": True,
                "shape": list(arr.shape),
                "max_value": float(np.nanmax(flat)) if flat.size else None,
                "min_value": float(np.nanmin(flat)) if flat.size else None,
                "sum_value": float(np.nansum(flat)) if flat.size else None,
                "positive_count": int(np.count_nonzero(positive)),
                "first_positive_timestep": first_pos,
                "worst_timestep": worst_unravel,
                "worst_value": None if worst_flat is None else float(flat[worst_flat]),
            }
        )
        if str(name) == "scene_collision" and worst_unravel and sphere_link_names:
            sphere_i = int(worst_unravel[-1])
            out[-1]["worst_sphere_index"] = sphere_i
            out[-1]["worst_link_name"] = (
                sphere_link_names[sphere_i] if sphere_i < len(sphere_link_names) else None
            )
    return out


def _metrics_audit(metrics: Any, *, sphere_link_names: list[str] | None = None) -> dict[str, Any]:
    if metrics is None:
        return {"present": False, "feasible": None, "constraints": []}
    cc = getattr(metrics, "costs_and_constraints", None)
    constraints: list[dict[str, Any]] = []
    hybrid: list[dict[str, Any]] = []
    feasible = None
    sum_constraint = None
    if cc is not None:
        constraints = _constraint_items(
            getattr(cc, "constraints", None),
            sphere_link_names=sphere_link_names,
        )
        hybrid = _constraint_items(
            getattr(cc, "hybrid_costs_constraints", None),
            sphere_link_names=sphere_link_names,
        )
        try:
            feasible_arr = _to_numpy(cc.get_feasible(include_all_hybrid=False, sum_horizon=True))
            feasible = None if feasible_arr is None else bool(np.asarray(feasible_arr, dtype=bool).all())
        except Exception as exc:
            feasible = f"{type(exc).__name__}: {exc}"
        try:
            sum_arr = _to_numpy(cc.get_sum_constraint(include_all_hybrid=False, sum_horizon=True))
            if sum_arr is not None:
                sum_constraint = {
                    "shape": list(np.asarray(sum_arr).shape),
                    "max": float(np.asarray(sum_arr).max()),
                    "min": float(np.asarray(sum_arr).min()),
                    "values": np.asarray(sum_arr).reshape(-1).tolist(),
                }
        except Exception as exc:
            sum_constraint = {"error": f"{type(exc).__name__}: {exc}"}
    failed = [x["name"] for x in constraints + hybrid if int(x.get("positive_count") or 0) > 0]
    return {
        "present": True,
        "type": f"{type(metrics).__module__}.{type(metrics).__name__}",
        "feasible": feasible,
        "sum_constraint": sum_constraint,
        "constraints": constraints,
        "hybrid_costs_constraints": hybrid,
        "failed_constraint_names": failed,
    }


def _recompute_metrics(
    planner,
    result: Any,
    *,
    sphere_link_names: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    raw_metrics = None
    interpolated_metrics = None

    solution = getattr(result, "solution", None)
    if solution is not None:
        action = solution.reshape(-1, solution.shape[-2], solution.shape[-1])
        expected = int(getattr(planner.trajopt_solver.config, "num_seeds", action.shape[0]))
        # TrajOptSolverResult.get_topk_seeds() keeps only the top seed, while
        # the metrics rollout goal buffer still carries the original seed count.
        # Repeat the selected top seed so cuRobo's metrics_transition_model shape
        # contract is satisfied. This is diagnostic-only and does not alter the
        # trajectory being evaluated.
        if action.shape[0] == 1 and expected > 1:
            action = action.repeat(expected, 1, 1)
        raw_metrics = planner.trajopt_solver.metrics_rollout.compute_metrics_from_action(action)

    js_solution = getattr(result, "js_solution", None)
    if js_solution is not None:
        interp = planner.trajopt_solver.additional_metrics_rollouts.get("interpolated_rollout")
        if interp is not None:
            from curobo.types import JointState

            def state_field(name: str):
                value = getattr(js_solution, name, None)
                if value is None:
                    return None
                return value.reshape(-1, value.shape[-2], value.shape[-1])

            pos = state_field("position")
            vel = state_field("velocity")
            acc = state_field("acceleration")
            jerk = state_field("jerk")
            expected = int(getattr(planner.trajopt_solver.config, "num_seeds", pos.shape[0]))
            if pos.shape[0] == 1 and expected > 1:
                pos = pos.repeat(expected, 1, 1)
                if vel is not None:
                    vel = vel.repeat(expected, 1, 1)
                if acc is not None:
                    acc = acc.repeat(expected, 1, 1)
                if jerk is not None:
                    jerk = jerk.repeat(expected, 1, 1)
            dt = getattr(js_solution, "dt", None)
            if dt is not None:
                dt = dt.reshape(-1)
                if dt.shape[0] == 1 and expected > 1:
                    dt = dt.repeat(expected)
                dt = dt[: pos.shape[0]]
            js_view = JointState(
                position=pos,
                velocity=vel,
                acceleration=acc,
                jerk=jerk,
                joint_names=planner.joint_names,
                dt=dt,
            )
            interp_state = interp.transition_model.compute_augmented_state(js_view)
            interpolated_metrics = interp.compute_metrics_from_state(interp_state)

    return (
        _metrics_audit(raw_metrics, sphere_link_names=sphere_link_names),
        _metrics_audit(interpolated_metrics, sphere_link_names=sphere_link_names),
        raw_metrics,
        interpolated_metrics,
    )


def _trajectory_array(result: Any) -> np.ndarray:
    js = getattr(result, "js_solution", None)
    pos = _to_numpy(getattr(js, "position", None) if js is not None else None)
    if pos is None:
        sol = _to_numpy(getattr(result, "solution", None))
        if sol is None:
            return np.empty((0, 0), dtype=np.float64)
        pos = sol
    pos = np.asarray(pos, dtype=np.float64)
    return pos.reshape(-1, pos.shape[-1])


def _dt_array(result: Any, n_steps: int) -> np.ndarray:
    js = getattr(result, "js_solution", None)
    dt = _to_numpy(getattr(js, "dt", None) if js is not None else None)
    if dt is None:
        return np.full(max(1, n_steps - 1), 0.025, dtype=np.float64)
    flat = np.asarray(dt, dtype=np.float64).reshape(-1)
    if flat.size == 1:
        return np.full(max(1, n_steps - 1), float(flat[0]), dtype=np.float64)
    return flat[: max(1, n_steps - 1)]


def _limit_array_from_robot_yaml(robot_file: Path, key: str, joint_names: list[str]) -> np.ndarray | None:
    try:
        import yaml

        data = yaml.safe_load(robot_file.read_text(encoding="utf-8"))
        kin = data.get("kinematics", data)
        cspace = kin.get("cspace", {})
        names = [str(x) for x in cspace.get("joint_names", [])]
        values = cspace.get(key)
        if not names or values is None:
            return None
        by_name = {name: float(value) for name, value in zip(names, values)}
        if not all(name in by_name for name in joint_names):
            return None
        return np.asarray([by_name[name] for name in joint_names], dtype=np.float64)
    except Exception:
        return None


def _position_limits(planner, joint_names: list[str]) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    for attr in ("state_bounds", "action_bounds"):
        bounds = _to_numpy(getattr(planner.trajopt_solver.metrics_rollout, attr, None))
        if bounds is not None:
            bounds = np.asarray(bounds, dtype=np.float64)
            if bounds.shape[0] == 2 and bounds.shape[1] >= len(joint_names):
                return (
                    bounds[0, : len(joint_names)],
                    bounds[1, : len(joint_names)],
                    f"planner.trajopt_solver.metrics_rollout.{attr}",
                )
    candidates = [
        getattr(getattr(planner, "kinematics", None), "joint_limits", None),
        getattr(getattr(planner, "rollout_fn", None), "kinematics", None),
        getattr(getattr(planner.trajopt_solver.metrics_rollout, "kinematics", None), "joint_limits", None),
        getattr(getattr(planner.trajopt_solver.metrics_rollout, "robot_model", None), "joint_limits", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        lower = _to_numpy(getattr(obj, "position_lower", None) or getattr(obj, "lower", None))
        upper = _to_numpy(getattr(obj, "position_upper", None) or getattr(obj, "upper", None))
        if lower is not None and upper is not None:
            lower = np.asarray(lower, dtype=np.float64).reshape(-1)
            upper = np.asarray(upper, dtype=np.float64).reshape(-1)
            if lower.size >= len(joint_names) and upper.size >= len(joint_names):
                return lower[: len(joint_names)], upper[: len(joint_names)], f"{type(obj).__module__}.{type(obj).__name__}"
    return None, None, "unavailable"


def _joint_limit_audit(q: np.ndarray, lower: np.ndarray | None, upper: np.ndarray | None, joint_names: list[str]) -> dict[str, Any]:
    if lower is None or upper is None:
        return {"available": False, "joint_limit_violation_count": None}
    low_violation = lower.reshape(1, -1) - q
    high_violation = q - upper.reshape(1, -1)
    violation = np.maximum(low_violation, high_violation)
    positive = violation > 0.0
    count = int(np.count_nonzero(positive))
    if violation.size:
        worst_flat = int(np.argmax(violation))
        t, j = np.unravel_index(worst_flat, violation.shape)
        worst = {
            "timestep": int(t),
            "joint_index": int(j),
            "joint_name": joint_names[int(j)],
            "value_rad": float(q[int(t), int(j)]),
            "lower_rad": float(lower[int(j)]),
            "upper_rad": float(upper[int(j)]),
            "violation_rad": max(0.0, float(violation[int(t), int(j)])),
        }
    else:
        worst = None
    return {
        "available": True,
        "joint_limit_violation_count": count,
        "worst_joint": worst,
    }


def _dynamic_audit(
    q: np.ndarray,
    dt: np.ndarray,
    joint_names: list[str],
    limits: np.ndarray | None,
    kind: str,
) -> dict[str, Any]:
    if q.shape[0] < 2:
        return {"available": False, "reason": "trajectory too short"}
    if kind == "velocity":
        values = np.diff(q, axis=0) / dt[: q.shape[0] - 1].reshape(-1, 1)
    elif kind == "acceleration":
        v = np.diff(q, axis=0) / dt[: q.shape[0] - 1].reshape(-1, 1)
        if v.shape[0] < 2:
            return {"available": False, "reason": "trajectory too short for acceleration"}
        values = np.diff(v, axis=0) / dt[: v.shape[0] - 1].reshape(-1, 1)
    elif kind == "jerk":
        v = np.diff(q, axis=0) / dt[: q.shape[0] - 1].reshape(-1, 1)
        if v.shape[0] < 3:
            return {"available": False, "reason": "trajectory too short for jerk"}
        a = np.diff(v, axis=0) / dt[: v.shape[0] - 1].reshape(-1, 1)
        values = np.diff(a, axis=0) / dt[: a.shape[0] - 1].reshape(-1, 1)
    else:
        raise ValueError(kind)
    max_abs = np.max(np.abs(values), axis=0)
    report: dict[str, Any] = {
        "available": True,
        "measured_max_abs_by_joint": {
            name: float(max_abs[i]) for i, name in enumerate(joint_names)
        },
        "max_abs": float(max_abs.max()) if max_abs.size else None,
        "worst_joint": None,
        "limit_available": limits is not None,
        "violations": [],
    }
    if max_abs.size:
        j = int(np.argmax(max_abs))
        report["worst_joint"] = {
            "joint_index": j,
            "joint_name": joint_names[j],
            "measured": float(max_abs[j]),
            "limit": None if limits is None else float(limits[j]),
            "ratio": None if limits is None or limits[j] == 0 else float(max_abs[j] / limits[j]),
        }
    if limits is not None:
        ratio = max_abs / limits
        bad = np.nonzero(ratio > 1.0)[0]
        report["violation_count"] = int(len(bad))
        report["violations"] = [
            {
                "joint_index": int(j),
                "joint_name": joint_names[int(j)],
                "measured": float(max_abs[int(j)]),
                "limit": float(limits[int(j)]),
                "ratio": float(ratio[int(j)]),
            }
            for j in bad
        ]
    return report


def _environment_audit(
    *,
    q: np.ndarray,
    scene_grid: Any,
    model: CuroboRobotSphereModel,
    joint_names: list[str],
) -> dict[str, Any]:
    min_clearance = float("inf")
    worst = None
    first_collision = None
    collision_count = 0
    for t, q_t in enumerate(q):
        q_by_name = {name: float(q_t[j]) for j, name in enumerate(joint_names)}
        spheres = model.spheres_from_named_joints(q_by_name)
        batch = query_spheres(scene_grid, spheres[:, :3], spheres[:, 3], margin_m=0.0)
        clearance = np.asarray(batch.distance_m, dtype=np.float64) - spheres[:, 3]
        local = int(np.argmin(clearance))
        if float(clearance[local]) < min_clearance:
            min_clearance = float(clearance[local])
            worst = {
                "timestep": int(t),
                "sphere_index": local,
                "link_name": model.sphere_link_names[local] if local < len(model.sphere_link_names) else None,
                "sphere_radius_m": float(spheres[local, 3]),
                "esdf_signed_distance_m": float(batch.distance_m[local]),
                "clearance_m": float(clearance[local]),
                "inside_grid": bool(batch.inside_grid[local]),
                "collision": bool(batch.collision[local]),
            }
        if bool(np.any(batch.collision)):
            collision_count += 1
            if first_collision is None:
                idx = int(np.nonzero(batch.collision)[0][0])
                first_collision = {
                    "timestep": int(t),
                    "sphere_index": idx,
                    "link_name": model.sphere_link_names[idx] if idx < len(model.sphere_link_names) else None,
                    "sphere_radius_m": float(spheres[idx, 3]),
                    "esdf_signed_distance_m": float(batch.distance_m[idx]),
                    "clearance_m": float(clearance[idx]),
                }
    return {
        "trajectory_environment_collision": first_collision is not None,
        "collision_sample_count": int(collision_count),
        "first_collision": first_collision,
        "min_clearance_m": float(min_clearance),
        "worst_timestep": None if worst is None else worst["timestep"],
        "worst_link": None if worst is None else worst["link_name"],
        "worst_sphere": None if worst is None else worst["sphere_index"],
        "worst_sample": worst,
    }


def _root_cause(raw: dict[str, Any], interp: dict[str, Any], joint_limits: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    failed_raw = raw.get("failed_constraint_names", []) or []
    failed_interp = interp.get("failed_constraint_names", []) or []
    names = sorted(set(failed_raw + failed_interp))
    if failed_raw and failed_interp:
        stage = "both"
    elif failed_raw:
        stage = "raw"
    elif failed_interp:
        stage = "interpolated"
    else:
        stage = "unknown"
    if names:
        summary = f"cuRobo feasibility is false because constraint(s) are positive: {', '.join(names)}."
    elif env.get("trajectory_environment_collision"):
        summary = "Project ESDF post-check found environment collision on returned TrajOpt trajectory."
    elif joint_limits.get("joint_limit_violation_count"):
        summary = "Project joint-limit post-check found position joint-limit violation."
    else:
        summary = "No positive constraint was exposed by recomputed public metrics; failure is likely hidden in convergence/valid-query bookkeeping or unsupported metric recomputation path."
    return {
        "failed_constraint_names": names,
        "stage": stage,
        "summary": summary,
    }


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
        raise FileNotFoundError("missing TrajOpt feasibility audit inputs: " + ", ".join(missing))

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
                "interpolation_dt_s": args.interpolation_dt_s,
            }
        }
    )
    scene = adapter.build_pick_scene(inputs["filtered_depth"], inputs["intrinsics"], inputs["T_world_camera"])
    planner = adapter.create_planner(scene)
    q_current_raw, q_goal_raw = _full_q_from_right(
        planner, robot_state["joint_positions_by_name"], q_pregrasp_right
    )
    q_current, q_goal, sanitization_report = adapter.sanitize_planning_joint_states(
        q_current_raw, q_goal_raw
    )
    adapter.last_joint_state_sanitization_report = sanitization_report
    current_state = _joint_state_from_q(planner, q_current)
    goal_state = _joint_state_from_q(planner, q_goal)

    t0 = time.time()
    result = planner.trajopt_solver.solve_cspace(
        goal_state,
        current_state,
        seed_traj=None,
        finetune_attempts=3,
        finetune_dt_scale=0.75,
    )
    wall = time.time() - t0

    model = CuroboRobotSphereModel(adapter.robot_file, device=args.device)
    raw_report, interp_report, _raw_metrics, _interp_metrics = _recompute_metrics(
        planner,
        result,
        sphere_link_names=list(model.sphere_link_names),
    )
    q = _trajectory_array(result)
    dt = _dt_array(result, q.shape[0])
    joint_names = [str(x) for x in planner.joint_names]
    lower, upper, limit_source = _position_limits(planner, joint_names)
    joint_report = _joint_limit_audit(q, lower, upper, joint_names)
    vel_limits = _limit_array_from_robot_yaml(adapter.robot_file, "max_velocity", joint_names)
    acc_limits = _limit_array_from_robot_yaml(adapter.robot_file, "max_acceleration", joint_names)
    jerk_limits = _limit_array_from_robot_yaml(adapter.robot_file, "max_jerk", joint_names)
    velocity_report = _dynamic_audit(q, dt, joint_names, vel_limits, "velocity")
    acceleration_report = _dynamic_audit(q, dt, joint_names, acc_limits, "acceleration")
    jerk_report = _dynamic_audit(q, dt, joint_names, jerk_limits, "jerk")
    env_report = _environment_audit(q=q, scene_grid=scene.voxel[0], model=model, joint_names=joint_names)

    report = {
        "schema_version": 1,
        "route": "RouteB",
        "audit": "trajopt_feasibility_current_to_pregrasp",
        "inputs": {k: str(v) for k, v in inputs.items()},
        "collision_policy": {
            "environment_collision": True,
            "self_collision": False,
            "policy_report": adapter.last_collision_policy_report,
        },
        "joint_state_sanitization": sanitization_report,
        "planner": {
            "max_attempts": int(args.max_attempts),
            "enable_graph_attempt": int(args.enable_graph_attempt),
            "trajopt_num_seeds": int(planner.trajopt_solver.config.num_seeds),
            "trajopt_action_horizon": int(planner.trajopt_solver.action_horizon),
            "interpolation_dt_s": float(args.interpolation_dt_s),
            "solve_wall_time_s": float(wall),
            "result_success": bool(_success_count(result) > 0),
            "result_success_count": int(_success_count(result)),
            "solution_shape": _tensor_shape(getattr(result, "solution", None)),
            "js_solution_position_shape": _tensor_shape(getattr(getattr(result, "js_solution", None), "position", None)),
            "position_error": _float_or_none(getattr(result, "position_error", None)),
            "rotation_error": _float_or_none(getattr(result, "rotation_error", None)),
            "seed_cost": _float_or_none(getattr(result, "seed_cost", None)),
        },
        "raw_metrics": raw_report,
        "interpolated_metrics": interp_report,
        "joint_limits": {
            "limit_source": limit_source,
            **joint_report,
        },
        "velocity": velocity_report,
        "acceleration": acceleration_report,
        "jerk": jerk_report,
        "environment_collision": env_report,
    }
    report["root_cause"] = _root_cause(raw_report, interp_report, report["joint_limits"], env_report)

    out = output_dir / "trajopt_feasibility_audit.json"
    out.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    print("[Route B TrajOpt feasibility audit]")
    print(f"result_success={report['planner']['result_success']}")
    print(f"raw_feasible={raw_report.get('feasible')}")
    print(f"interpolated_feasible={interp_report.get('feasible')}")
    print(f"failed_constraints={report['root_cause']['failed_constraint_names']}")
    print(f"environment_collision={env_report['trajectory_environment_collision']}")
    print(f"min_clearance_m={env_report['min_clearance_m']:.6f}")
    print(f"joint_limit_violations={joint_report.get('joint_limit_violation_count')}")
    print(f"root_cause={report['root_cause']['summary']}")
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
    parser.add_argument("--interpolation-dt-s", type=float, default=0.025)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
