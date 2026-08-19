#!/usr/bin/env python3
"""Standalone Route B current -> PREGRASP MotionPlanner smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
if str(ISAACLAB_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_CONTROL_ROOT))

from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter
from curobo_motion_planning_routeB.routeB_adapter import DEFAULT_ENABLE_GRAPH_ATTEMPT
from core.perception_collision.esdf_collision import query_spheres
from core.perception_collision.robot_spheres import CuroboRobotSphereModel


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pregrasp_q(route_plan_path: Path) -> np.ndarray:
    with np.load(route_plan_path, allow_pickle=True) as data:
        names = [str(x) for x in data["waypoint_names"].tolist()]
        if "pregrasp" not in names:
            raise KeyError(f"{route_plan_path} does not contain a pregrasp waypoint")
        return np.asarray(data["arm_q_rad"][names.index("pregrasp")], dtype=np.float32)


def _trajectory_environment_postcheck(q: np.ndarray, scene_grid, model: CuroboRobotSphereModel, joint_names: list[str]) -> dict:
    min_clearance = float("inf")
    collision_count = 0
    worst = None
    for t, q_t in enumerate(q):
        spheres = model.spheres_from_named_joints({name: float(q_t[i]) for i, name in enumerate(joint_names)})
        batch = query_spheres(scene_grid, spheres[:, :3], spheres[:, 3], margin_m=0.0)
        clearance = np.asarray(batch.distance_m, dtype=np.float64) - spheres[:, 3]
        idx = int(np.argmin(clearance))
        if float(clearance[idx]) < min_clearance:
            min_clearance = float(clearance[idx])
            worst = {
                "timestep": int(t),
                "sphere_index": idx,
                "link_name": model.sphere_link_names[idx] if idx < len(model.sphere_link_names) else None,
                "clearance_m": float(clearance[idx]),
                "esdf_signed_distance_m": float(batch.distance_m[idx]),
                "sphere_radius_m": float(spheres[idx, 3]),
            }
        if bool(np.any(batch.collision)):
            collision_count += 1
    return {
        "trajectory_environment_collision": bool(collision_count > 0),
        "collision_sample_count": int(collision_count),
        "min_clearance_m": float(min_clearance),
        "worst_sample": worst,
    }


def _joint_limit_postcheck(q: np.ndarray, lower: np.ndarray, upper: np.ndarray, joint_names: list[str]) -> dict:
    violation = np.maximum(lower.reshape(1, -1) - q, q - upper.reshape(1, -1))
    positive = violation > 0.0
    worst = None
    if violation.size:
        t, j = np.unravel_index(int(np.argmax(violation)), violation.shape)
        worst = {
            "timestep": int(t),
            "joint_name": joint_names[int(j)],
            "value_rad": float(q[int(t), int(j)]),
            "lower_rad": float(lower[int(j)]),
            "upper_rad": float(upper[int(j)]),
            "violation_rad": max(0.0, float(violation[int(t), int(j)])),
        }
    return {
        "joint_limit_violation_count": int(np.count_nonzero(positive)),
        "worst_joint": worst,
    }


def _kinematic_postcheck(q: np.ndarray, dt_s: float) -> dict:
    def stat(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"available": False}
        return {
            "available": True,
            "max_abs": float(np.max(np.abs(values))),
            "finite": bool(np.isfinite(values).all()),
        }

    velocity = np.diff(q, axis=0) / dt_s if q.shape[0] >= 2 else np.empty((0, q.shape[1]))
    acceleration = np.diff(velocity, axis=0) / dt_s if velocity.shape[0] >= 2 else np.empty((0, q.shape[1]))
    jerk = np.diff(acceleration, axis=0) / dt_s if acceleration.shape[0] >= 2 else np.empty((0, q.shape[1]))
    return {
        "dt_s_assumed": float(dt_s),
        "velocity": stat(velocity),
        "acceleration": stat(acceleration),
        "jerk": stat(jerk),
    }


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
        raise FileNotFoundError("missing Route B test inputs: " + ", ".join(missing))

    robot_state = load_json(inputs["robot_state"])
    q_current_by_name = robot_state["joint_positions_by_name"]
    q_pregrasp = load_pregrasp_q(route_plan)

    cfg = {
        "routeB": {
            "device": args.device,
            "robot_file": str(args.robot_file),
            "layout_json": str(args.layout_json),
            "collision": {
                "environment_collision": True,
                "self_collision": args.self_collision_check,
            },
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
    scene = adapter.build_pick_scene(
        inputs["filtered_depth"],
        inputs["intrinsics"],
        inputs["T_world_camera"],
    )
    result = adapter.plan_current_to_pregrasp(q_current_by_name, q_pregrasp, scene)

    traj_path = output_dir / "trajectory.npz"
    report_path = output_dir / "report.json"
    np.savez_compressed(
        traj_path,
        q_rad=result.trajectory_q_rad,
        joint_names=np.asarray(result.joint_names),
        start_q_rad=result.start_q_rad,
        goal_q_rad=result.goal_q_rad,
        source_route_plan=str(route_plan),
        source_capture_dir=str(capture_dir),
    )
    report = result.to_report()
    raw_start = np.asarray(
        [float(q_current_by_name.get(name, result.start_q_rad[i])) for i, name in enumerate(result.joint_names)],
        dtype=np.float32,
    )
    trajectory_start = (
        np.asarray(result.trajectory_q_rad[0], dtype=np.float32)
        if result.trajectory_q_rad.size
        else np.asarray(result.start_q_rad, dtype=np.float32)
    )
    report["motion_planner_graph_seed"] = {
        "enable_graph_attempt": int(args.enable_graph_attempt),
        "max_attempts": int(args.max_attempts),
        "enabled_in_this_run": bool(args.enable_graph_attempt < args.max_attempts),
    }
    returned_trajectory_postcheck = None
    if result.trajectory_q_rad.size:
        sphere_model = CuroboRobotSphereModel(adapter.robot_file, device=args.device)
        joint_names = [str(x) for x in result.joint_names]
        lower, upper = adapter._motion_planner_position_bounds()
        q_traj = np.asarray(result.trajectory_q_rad, dtype=np.float32)
        returned_trajectory_postcheck = {
            "environment_collision": _trajectory_environment_postcheck(
                q=q_traj,
                scene_grid=scene.voxel[0],
                model=sphere_model,
                joint_names=joint_names,
            ),
            "joint_limits": {
                "limit_source": "RouteBMotionPlannerAdapter._motion_planner_position_bounds",
                **_joint_limit_postcheck(q_traj, lower, upper, joint_names),
            },
            **_kinematic_postcheck(q_traj, float(args.interpolation_dt_s)),
        }
    report.update(
        {
            "capture_dir": str(capture_dir),
            "route_plan": str(route_plan),
            "filtered_depth": str(inputs["filtered_depth"]),
            "intrinsics": str(inputs["intrinsics"]),
            "T_world_camera": str(inputs["T_world_camera"]),
            "robot_state": str(inputs["robot_state"]),
            "trajectory_npz": str(traj_path),
            "output_dir": str(output_dir),
            "q_current_source": "capture/robot_state.json",
            "q_pregrasp_source": "flexible_route_plan.npz:arm_q_rad[pregrasp]",
            "finite_trajectory": bool(np.isfinite(result.trajectory_q_rad).all()),
            "joint_limit_pass": bool(result.success and np.isfinite(result.trajectory_q_rad).all()),
            "scene_report": adapter.last_scene_report,
            "collision_policy_report": adapter.last_collision_policy_report,
            "voxel_shape_contract": adapter.last_voxel_shape_contract_report,
            "returned_trajectory_postcheck": returned_trajectory_postcheck,
            "start_matches_q_current_raw": bool(
                all(
                    name in q_current_by_name
                    and abs(float(result.start_q_rad[i]) - float(q_current_by_name[name])) <= 1.0e-6
                    for i, name in enumerate(result.joint_names)
                )
            ),
            "start_matches_q_current": bool(np.allclose(result.start_q_rad, raw_start, atol=1.0e-6)),
            "trajectory_start_vs_raw_q_current_max_abs_rad": float(
                np.max(np.abs(trajectory_start - raw_start))
            ),
            "trajectory_start_vs_planning_q_current_max_abs_rad": float(
                np.max(np.abs(trajectory_start - result.start_q_rad))
            ),
            "q_current_raw_to_planning_max_abs_rad": float(
                (report.get("joint_state_sanitization") or {}).get(
                    "q_current_planning_diff_max_abs_rad", 0.0
                )
            ),
            "q_pregrasp_raw_to_planning_max_abs_rad": float(
                (report.get("joint_state_sanitization") or {}).get(
                    "q_pregrasp_planning_diff_max_abs_rad", 0.0
                )
            ),
            "end_matches_q_pregrasp": bool(
                result.goal_q_rad.shape[0] >= 7
                and np.allclose(result.goal_q_rad[:7], q_pregrasp, atol=1.0e-6)
            ),
        }
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[Route B current->PREGRASP]")
    print(f"success={result.success}")
    print(f"planning_time_s={result.planning_time_s:.3f}")
    print(f"trajectory_points={result.waypoint_count}")
    print(f"trajectory={traj_path}")
    print(f"report={report_path}")
    return 0 if result.success else 2


def main() -> int:
    default_robot = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--route-plan", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--robot-file", default=str(default_robot))
    parser.add_argument(
        "--layout-json",
        default=str(PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--self-collision-check", action="store_true")
    parser.add_argument("--num-ik-seeds", type=int, default=32)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--enable-graph-attempt", type=int, default=DEFAULT_ENABLE_GRAPH_ATTEMPT)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--interpolation-dt-s", type=float, default=0.025)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
