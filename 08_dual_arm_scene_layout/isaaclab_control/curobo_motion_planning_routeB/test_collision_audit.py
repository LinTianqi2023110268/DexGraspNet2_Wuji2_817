#!/usr/bin/env python3
"""Route B endpoint collision audit.

This script is diagnostic-only.  It does not run MotionPlanner trajectory
optimization and does not modify Route A/Route B planning logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
if str(ISAACLAB_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_CONTROL_ROOT))

from core.config import RIGHT_ARM_NAMES
from core.perception_collision.esdf_collision import query_spheres
from core.perception_collision.robot_spheres import CuroboRobotSphereModel
from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter
from test_current_to_pregrasp import load_json, load_pregrasp_q


def _full_pregrasp_state(q_current_by_name: dict[str, float], q_pregrasp_right: np.ndarray) -> dict[str, float]:
    q = {str(k): float(v) for k, v in q_current_by_name.items()}
    for name, value in zip(RIGHT_ARM_NAMES, np.asarray(q_pregrasp_right, dtype=np.float64).reshape(-1)):
        q[name] = float(value)
    return q


def _self_summary(model: CuroboRobotSphereModel, q_by_name: dict[str, float]) -> dict[str, Any]:
    result = model.check_self_collision(q_by_name)
    return {
        "collision": bool(not result["self_collision_pass"]),
        "min_distance": None,
        "details": result.get("self_collision_top_pairs", []),
        "pair_count": int(result.get("self_collision_pair_count", 0)),
        "max_penetration_m": float(result.get("self_collision_max_penetration_m", 0.0)),
        "note": "cuRobo SelfCollisionCost exposes penetration pairs, not a signed minimum free distance.",
    }


def _env_summary(
    *,
    model: CuroboRobotSphereModel,
    scene_grid: Any,
    q_by_name: dict[str, float],
    right_arm_only: bool = False,
    top_k: int = 20,
) -> dict[str, Any]:
    spheres = model.spheres_from_named_joints(q_by_name)
    link_names = list(model.sphere_link_names)
    if len(link_names) != len(spheres):
        link_names = [f"unknown_link_{i}" for i in range(len(spheres))]

    keep = np.ones(len(spheres), dtype=bool)
    if right_arm_only:
        keep = np.asarray(
            [
                name.startswith("arm_r_link_") or name in {"arm_r_link_tf", "arm_r_link_d405"}
                for name in link_names
            ],
            dtype=bool,
        )
    original_indices = np.nonzero(keep)[0]
    spheres_eval = spheres[keep]
    links_eval = [link_names[i] for i in original_indices]
    if len(spheres_eval) == 0:
        return {
            "collision": False,
            "min_distance": None,
            "details": [],
            "sphere_count": 0,
            "colliding_sphere_count": 0,
            "inside_grid_count": 0,
            "right_arm_only": bool(right_arm_only),
        }

    batch = query_spheres(scene_grid, spheres_eval[:, :3], spheres_eval[:, 3], margin_m=0.0)
    clearance = np.asarray(batch.distance_m, dtype=np.float64) - spheres_eval[:, 3]
    collision = np.asarray(batch.collision, dtype=bool)
    inside = np.asarray(batch.inside_grid, dtype=bool)
    order = np.argsort(clearance)[:top_k]
    details = []
    for local_i in order:
        details.append(
            {
                "sphere_index": int(original_indices[int(local_i)]),
                "link_name": str(links_eval[int(local_i)]),
                "sphere_center": spheres_eval[int(local_i), :3].astype(float).tolist(),
                "sphere_radius_m": float(spheres_eval[int(local_i), 3]),
                "esdf_signed_distance_m": float(batch.distance_m[int(local_i)]),
                "clearance_m": float(clearance[int(local_i)]),
                "inside_grid": bool(inside[int(local_i)]),
                "collision": bool(collision[int(local_i)]),
            }
        )
    return {
        "collision": bool(np.any(collision)),
        "min_distance": float(np.min(clearance)),
        "details": details,
        "sphere_count": int(len(spheres_eval)),
        "colliding_sphere_count": int(np.count_nonzero(collision)),
        "inside_grid_count": int(np.count_nonzero(inside)),
        "right_arm_only": bool(right_arm_only),
    }


def _combine(self_result: dict[str, Any], env_result: dict[str, Any] | None) -> dict[str, Any]:
    env_collision = False if env_result is None else bool(env_result["collision"])
    details = []
    if self_result["collision"]:
        details.extend(
            {
                "type": "self_collision",
                **item,
            }
            for item in self_result.get("details", [])
        )
    if env_result is not None and env_result["collision"]:
        details.extend(
            {
                "type": "environment_esdf",
                **item,
            }
            for item in env_result.get("details", [])
            if item.get("collision")
        )
    return {
        "collision": bool(self_result["collision"] or env_collision),
        "min_distance": None if env_result is None else env_result["min_distance"],
        "details": details[:20],
        "self_collision": self_result,
        "environment_collision": env_result,
    }


def _state_pass(state: dict[str, Any]) -> bool:
    return not bool(state.get("collision", False))


def run(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    route_plan = Path(args.route_plan).expanduser().resolve()
    output_dir = capture_dir / "curobo_test_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_file = Path(args.robot_file).expanduser()
    if not robot_file.is_absolute():
        robot_file = (PROJECT_ROOT / robot_file).resolve()

    inputs = {
        "filtered_depth": capture_dir / "planning/filtered_depth.npy",
        "intrinsics": capture_dir / "intrinsics.npy",
        "T_world_camera": capture_dir / "T_world_camera.npy",
        "robot_state": capture_dir / "robot_state.json",
        "route_plan": route_plan,
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing collision audit inputs: " + ", ".join(missing))

    robot_state = load_json(inputs["robot_state"])
    q_current = {str(k): float(v) for k, v in robot_state["joint_positions_by_name"].items()}
    q_pregrasp = _full_pregrasp_state(q_current, load_pregrasp_q(route_plan))

    adapter = RouteBMotionPlannerAdapter(
        {
            "routeB": {
                "device": args.device,
                "robot_file": str(robot_file),
                "layout_json": args.layout_json,
                "collision": {
                    "environment_collision": True,
                    "self_collision": False,
                },
                "mapper": {
                    "voxel_size_m": args.voxel_size_m,
                    "esdf_voxel_size_m": args.esdf_voxel_size_m,
                },
            }
        }
    )
    scene = adapter.build_pick_scene(
        inputs["filtered_depth"],
        inputs["intrinsics"],
        inputs["T_world_camera"],
    )
    scene_grid = scene.voxel[0]
    model = CuroboRobotSphereModel(robot_file, device=args.device)

    empty = {
        "q_current": _self_summary(model, q_current),
        "q_pregrasp": _self_summary(model, q_pregrasp),
    }
    esdf_env = {
        "q_current": _env_summary(model=model, scene_grid=scene_grid, q_by_name=q_current),
        "q_pregrasp": _env_summary(model=model, scene_grid=scene_grid, q_by_name=q_pregrasp),
    }
    esdf_combined = {
        name: _combine(empty[name], esdf_env[name])
        for name in ("q_current", "q_pregrasp")
    }
    self_disabled = {
        "q_current": esdf_env["q_current"],
        "q_pregrasp": esdf_env["q_pregrasp"],
    }
    right_arm_only = {
        "q_current": _env_summary(model=model, scene_grid=scene_grid, q_by_name=q_current, right_arm_only=True),
        "q_pregrasp": _env_summary(model=model, scene_grid=scene_grid, q_by_name=q_pregrasp, right_arm_only=True),
    }

    test_pass = {
        "empty_scene": _state_pass(empty["q_current"]) and _state_pass(empty["q_pregrasp"]),
        "esdf_scene": _state_pass(esdf_combined["q_current"]) and _state_pass(esdf_combined["q_pregrasp"]),
        "self_collision_disabled": _state_pass(self_disabled["q_current"]) and _state_pass(self_disabled["q_pregrasp"]),
        "right_arm_only": _state_pass(right_arm_only["q_current"]) and _state_pass(right_arm_only["q_pregrasp"]),
    }

    report = {
        "schema_version": 1,
        "route": "RouteB",
        "audit": "current_to_pregrasp_endpoint_collision",
        "inputs": {k: str(v) for k, v in inputs.items()},
        "environment": {
            "esdf_enabled": True,
            "voxel_size": float(adapter.last_scene_report.get("voxel_size_m", args.voxel_size_m)),
            "esdf_voxel_size": float(adapter.last_scene_report.get("esdf_voxel_size_m", args.esdf_voxel_size_m)),
            "scene_report": adapter.last_scene_report,
        },
        "q_current": esdf_combined["q_current"],
        "q_pregrasp": esdf_combined["q_pregrasp"],
        "tests": test_pass,
        "subtests": {
            "empty_scene_self_collision_only": empty,
            "esdf_scene_self_plus_environment": esdf_combined,
            "self_collision_disabled_environment_only": self_disabled,
            "right_arm_only_environment_only": right_arm_only,
        },
        "api_notes": {
            "environment_details": "Computed via project CuroboRobotSphereModel FK plus ESDF query_spheres, giving link/sphere/clearance details.",
            "self_collision_details": "cuRobo SelfCollisionCost exposes pair penetration and guessed sphere/link pairs; it does not expose a simple signed minimum free distance.",
            "motionplanner_start_end_details": "MotionPlanner.plan_cspace only printed 'Start or End state in collision' here; exact MotionPlanner-internal start/end collision object requires cuRobo rollout/collision checker internals.",
        },
    }

    out = output_dir / "collision_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[Route B collision audit]")
    print(f"empty_scene={test_pass['empty_scene']}")
    print(f"esdf_scene={test_pass['esdf_scene']}")
    print(f"self_collision_disabled={test_pass['self_collision_disabled']}")
    print(f"right_arm_only={test_pass['right_arm_only']}")
    for state_name in ("q_current", "q_pregrasp"):
        state = esdf_combined[state_name]
        print(
            f"{state_name}: collision={state['collision']} "
            f"min_clearance_m={state['min_distance']}"
        )
        for item in state.get("details", [])[:5]:
            print(f"  {item}")
    print(f"report={out}")
    return 0


def main() -> int:
    default_robot = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml"
    default_layout = PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--route-plan", required=True)
    parser.add_argument("--robot-file", default=str(default_robot))
    parser.add_argument("--layout-json", default=str(default_layout))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size-m", type=float, default=0.01)
    parser.add_argument("--esdf-voxel-size-m", type=float, default=0.02)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
