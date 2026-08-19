#!/usr/bin/env python3
"""One-to-one audit of cuRobo scene_collision semantics vs project ESDF query.

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
from test_esdf_ground_truth_audit import _curobo_semantic_sdf
from test_motion_planner_failure_audit import _full_q_from_right, _joint_state_from_q
from test_trajopt_feasibility_audit import _json_default, _recompute_metrics, _to_numpy


SAMPLES = [
    {
        "label": "worst_curobo_scene_collision",
        "timestep": 76,
        "sphere_index": 112,
        "expected_link": "arm_r_link_7",
    },
    {
        "label": "first_curobo_scene_collision_positive",
        "timestep": 61,
        "sphere_index": 184,
        "expected_link": "r_wrist",
    },
    {
        "label": "project_min_clearance_sample",
        "timestep": 64,
        "sphere_index": 185,
        "expected_link": "r_wrist",
    },
]


CUROBO_SOURCE_LOCATIONS = {
    "SceneCollisionCost.forward": (
        "curobo/_src/cost/cost_scene_collision.py:"
        "SceneCollisionCost.forward -> _discrete_fn"
    ),
    "discrete_query": (
        "SceneCollisionCost._discrete_fn calls "
        "SceneCollision.get_sphere_collision when convert_to_binary=True, otherwise get_sphere_distance"
    ),
    "SceneCollision": (
        "curobo/_src/geom/collision/collision_scene.py:"
        "SceneCollision.get_sphere_collision/get_sphere_distance"
    ),
    "CollisionChecker": (
        "curobo/_src/geom/collision/checker_collision.py:"
        "CollisionChecker.get_sphere_distance"
    ),
    "Warp_kernel": (
        "curobo/_src/geom/collision/wp_collision_kernel.py:"
        "penetration = -local_sdf + (sphere_radius + activation_distance); "
        "constraint/cost = weight * activation(penetration)"
    ),
    "Voxel_SDF": (
        "curobo/_src/geom/data/data_voxel.py:"
        "compute_local_sdf_with_grad uses float16 voxel features and trilinear interpolation"
    ),
}


def _float(value: Any) -> float | None:
    arr = _to_numpy(value)
    if arr is None:
        return None
    flat = np.asarray(arr).reshape(-1)
    if flat.size == 0:
        return None
    return float(flat[0])


def _tensor_scalar(tensor: Any, index: tuple[int, ...]) -> float:
    arr = _to_numpy(tensor)
    if arr is None:
        raise RuntimeError("expected tensor-like value")
    return float(np.asarray(arr)[index])


def _scene_grid_report(scene_grid: Any) -> dict[str, Any]:
    feature = getattr(scene_grid, "feature_tensor", None)
    feature_np = _to_numpy(feature)
    pose = _to_numpy(getattr(scene_grid, "pose", None))
    return {
        "type": f"{type(scene_grid).__module__}.{type(scene_grid).__name__}",
        "frame": "arm_base_link",
        "voxel_size_m": float(getattr(scene_grid, "voxel_size", np.nan)),
        "pose_center_base_m": None if pose is None else np.asarray(pose).reshape(-1)[:3].tolist(),
        "feature_shape": None if feature_np is None else list(feature_np.shape),
        "feature_dtype": None if feature is None else str(getattr(feature, "dtype", type(feature))),
        "feature_min": None if feature_np is None else float(np.nanmin(feature_np)),
        "feature_max": None if feature_np is None else float(np.nanmax(feature_np)),
    }


def _find_scene_cost(planner):
    manager = planner.trajopt_solver.metrics_rollout.metrics_constraint_manager
    costs = getattr(manager, "costs", {})
    if "scene_collision" not in costs:
        raise RuntimeError("metrics_constraint_manager has no scene_collision cost")
    return costs["scene_collision"]


def _scene_cost_config_report(scene_cost: Any) -> dict[str, Any]:
    cfg = scene_cost.config
    return {
        "class": f"{type(scene_cost).__module__}.{type(scene_cost).__name__}",
        "config_class": f"{type(cfg).__module__}.{type(cfg).__name__}",
        "weight": _float(getattr(scene_cost, "_weight", None)),
        "activation_distance_m": _float(getattr(cfg, "activation_distance", None)),
        "use_sweep": bool(getattr(cfg, "use_sweep", False)),
        "sum_distance": bool(getattr(cfg, "sum_distance", False)),
        "convert_to_binary": bool(getattr(cfg, "convert_to_binary", False)),
        "use_grad_input": bool(getattr(cfg, "use_grad_input", False)),
        "use_speed_metric": bool(getattr(cfg, "use_speed_metric", False)),
        "num_spheres": int(getattr(cfg, "num_spheres", -1)),
        "scene_collision_checker_class": f"{type(cfg.scene_collision_checker).__module__}.{type(cfg.scene_collision_checker).__name__}",
        "source_locations": CUROBO_SOURCE_LOCATIONS,
    }


def _curobo_internal_voxel_report(scene_cost: Any) -> dict[str, Any]:
    data = scene_cost.config.scene_collision_checker.data
    vox = getattr(data, "voxels", None)
    if vox is None:
        return {"present": False}
    out: dict[str, Any] = {
        "present": True,
        "class": f"{type(vox).__module__}.{type(vox).__name__}",
        "names": getattr(vox, "names", None),
    }
    for attr in ("params", "dims", "inv_pose", "features", "count"):
        value = getattr(vox, attr, None)
        arr = _to_numpy(value)
        out[attr] = {
            "present": value is not None,
            "shape": None if arr is None else list(np.asarray(arr).shape),
            "first_values": None if arr is None else np.asarray(arr).reshape(-1)[:32].tolist(),
            "dtype": None if value is None else str(getattr(value, "dtype", type(value))),
        }
    try:
        grid = vox.get_voxel_grid("block_sparse_esdf_grid", 0)
        out["get_voxel_grid"] = {
            "pose": getattr(grid, "pose", None),
            "voxel_size": getattr(grid, "voxel_size", None),
            "dims": getattr(grid, "dims", None),
            "feature_shape": list(grid.feature_tensor.shape)
            if getattr(grid, "feature_tensor", None) is not None
            else None,
        }
    except Exception as exc:
        out["get_voxel_grid_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _raw_curobo_collision_value(scene_cost: Any, sphere: np.ndarray, device: str) -> dict[str, Any]:
    import torch
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer

    sph = torch.as_tensor(sphere, device=device, dtype=torch.float32).reshape(1, 1, 1, 4)
    buffer = CollisionBuffer.from_shape(tuple(sph.shape), scene_cost.device_cfg)
    dist_same_weight = scene_cost.config.scene_collision_checker.get_sphere_distance_raw(
        sph,
        buffer,
        scene_cost._weight,
        scene_cost.config.activation_distance,
        return_loss=scene_cost.config.use_grad_input,
    )
    buffer_unit = CollisionBuffer.from_shape(tuple(sph.shape), scene_cost.device_cfg)
    unit_weight = torch.ones_like(scene_cost._weight)
    dist_unit_weight = scene_cost.config.scene_collision_checker.get_sphere_distance_raw(
        sph,
        buffer_unit,
        unit_weight,
        scene_cost.config.activation_distance,
        return_loss=scene_cost.config.use_grad_input,
    )
    return {
        "raw_cost_same_weight": float(dist_same_weight.detach().cpu().reshape(-1)[0].item()),
        "raw_cost_unit_weight": float(dist_unit_weight.detach().cpu().reshape(-1)[0].item()),
        "buffer_distance_same_weight": float(buffer.distance.detach().cpu().reshape(-1)[0].item()),
        "buffer_distance_unit_weight": float(buffer_unit.distance.detach().cpu().reshape(-1)[0].item()),
    }


def _infer_curobo_signed_distance(
    *,
    radius: float,
    activation_distance: float,
    weight: float,
    constraint_value: float,
) -> dict[str, float | None]:
    if weight is None or weight == 0:
        return {"penetration_m": None, "signed_distance_m": None}
    # For eta=0, apply_collision_activation(penetration, 0) is linear:
    # cost = weight * penetration for penetration > 0.
    if abs(float(activation_distance)) > 1.0e-12:
        return {"penetration_m": None, "signed_distance_m": None}
    penetration = float(constraint_value) / float(weight)
    signed_distance = float(radius) + float(activation_distance) - penetration
    return {
        "penetration_m": penetration,
        "signed_distance_m": signed_distance,
    }


def _sample_report(
    *,
    sample: dict[str, Any],
    robot_spheres: np.ndarray,
    scene_constraint: np.ndarray,
    scene_grid: Any,
    scene_cost: Any,
    sphere_link_names: list[str],
    device: str,
) -> dict[str, Any]:
    t = int(sample["timestep"])
    s = int(sample["sphere_index"])
    sphere = np.asarray(robot_spheres[0, t, s], dtype=np.float64)
    center = sphere[:3]
    radius = float(sphere[3])
    project = query_spheres(scene_grid, center.reshape(1, 3), np.asarray([radius]), margin_m=0.0)
    project_signed = float(project.distance_m[0])
    project_clearance = project_signed - radius
    curobo_semantic_sdf = float(_curobo_semantic_sdf(
        scene_cost,
        center.reshape(1, 3),
    )[0][0])
    grid_center = np.asarray(getattr(scene_grid, "pose", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)[:3]
    dims = np.asarray(getattr(scene_grid, "feature_tensor").shape, dtype=np.float64)
    voxel_size = float(getattr(scene_grid, "voxel_size"))
    local = center - grid_center
    voxel_continuous = local / voxel_size + dims / 2.0 - 0.5
    constraint_value = float(scene_constraint[0, t, s])
    weight = _float(getattr(scene_cost, "_weight", None))
    activation = _float(getattr(scene_cost.config, "activation_distance", None))
    inferred = _infer_curobo_signed_distance(
        radius=radius,
        activation_distance=float(activation or 0.0),
        weight=float(weight or 0.0),
        constraint_value=constraint_value,
    )
    raw = _raw_curobo_collision_value(scene_cost, sphere, device)
    unit_inferred = _infer_curobo_signed_distance(
        radius=radius,
        activation_distance=float(activation or 0.0),
        weight=1.0,
        constraint_value=raw["raw_cost_unit_weight"],
    )
    return {
        "label": sample["label"],
        "timestep": t,
        "sphere_index": s,
        "link_name": sphere_link_names[s] if s < len(sphere_link_names) else None,
        "expected_link": sample.get("expected_link"),
        "sphere_center_base_m": center.tolist(),
        "sphere_radius_m": radius,
        "scene_grid_frame": "arm_base_link",
        "project_grid_local_center_m": local.tolist(),
        "project_grid_continuous_voxel_xyz": voxel_continuous.tolist(),
        "project_query": {
            "signed_distance_m": project_signed,
            "clearance_m": project_clearance,
            "inside_grid": bool(project.inside_grid[0]),
            "collision": bool(project.collision[0]),
            "formula": "collision = inside_grid and signed_distance <= sphere_radius",
        },
        "curobo_semantic_query": {
            "signed_distance_m": curobo_semantic_sdf,
            "clearance_m": curobo_semantic_sdf - radius,
            "difference_vs_project_signed_distance_m": curobo_semantic_sdf - project_signed,
        },
        "curobo_rollout": {
            "constraint_value": constraint_value,
            "activation_distance_m": activation,
            "weight": weight,
            "inferred_penetration_m": inferred["penetration_m"],
            "inferred_signed_distance_m": inferred["signed_distance_m"],
            "formula_eta0": "constraint = weight * max(radius - signed_distance, 0)",
        },
        "curobo_raw_checker": {
            **raw,
            "unit_weight_inferred_penetration_m": unit_inferred["penetration_m"],
            "unit_weight_inferred_signed_distance_m": unit_inferred["signed_distance_m"],
        },
        "semantic_checks": {
            "distance_sign_convention_difference": (
                "no: both treat positive SDF as outside/free, negative as inside; "
                "cuRobo then converts to positive penetration/cost"
            ),
            "unit_difference_suspected": False,
            "sphere_radius_double_count_suspected": bool(
                inferred["signed_distance_m"] is not None
                and abs(project_signed - radius - inferred["signed_distance_m"]) < 0.01
            ),
            "frame_transform_double_application_suspected": bool(
                abs(project_signed - (inferred["signed_distance_m"] or project_signed)) > 0.05
            ),
            "inside_outside_grid_project": bool(project.inside_grid[0]),
        },
    }


def _root_cause(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[str, str]:
    worst = samples[0]
    project = worst["project_query"]
    cu = worst["curobo_rollout"]
    raw = worst["curobo_raw_checker"]
    a_b_consistent = abs(raw["raw_cost_same_weight"] - cu["constraint_value"]) < 1.0e-4
    if project["clearance_m"] > 0 and (cu["inferred_penetration_m"] or 0.0) > 0:
        root = (
            "cuRobo scene_collision raw checker and rollout agree, but they disagree with "
            "the project ESDF query for the same base-frame sphere. The mismatch is in the "
            "scene-collision checker/scene representation path, not in TrajOpt postprocessing."
        )
        if cfg.get("convert_to_binary"):
            root += " The configured metric uses get_sphere_collision with convert_to_binary=True."
        fix = (
            "Next minimal fix is to audit the cuRobo VoxelGrid obstacle transform/features handed "
            "to SceneCollision against the project VoxelGrid pose/feature_tensor, then align the "
            "Route B scene construction so cuRobo's collision checker queries the same base-frame "
            "ESDF used by project query_spheres. Do not change thresholds before that."
        )
    elif a_b_consistent:
        root = "A≈B and B→C follows cuRobo activation/weight semantics."
        fix = "Inspect activation/weight only if the resulting clearance is intentionally too conservative."
    else:
        root = "cuRobo raw checker and rollout constraint are inconsistent; inspect collision buffer/cost manager state."
        fix = "Recompute rollout metrics and raw checker from the exact same KinematicsState/collision buffer."
    return root, fix


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
        raise FileNotFoundError("missing scene collision audit inputs: " + ", ".join(missing))

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
    solve_wall_time_s = time.time() - t0

    sphere_model = CuroboRobotSphereModel(adapter.robot_file, device=args.device)
    raw_report, _interp_report, raw_metrics, _interp_metrics = _recompute_metrics(
        planner,
        result,
        sphere_link_names=list(sphere_model.sphere_link_names),
    )
    if raw_metrics is None:
        raise RuntimeError("failed to recompute raw metrics")
    scene_cost = _find_scene_cost(planner)
    scene_constraint_items = [
        item for item in raw_report.get("constraints", []) if item.get("name") == "scene_collision"
    ]
    if not scene_constraint_items:
        raise RuntimeError("raw metrics did not contain scene_collision constraint")
    costs_and_constraints = raw_metrics.costs_and_constraints
    constraints = costs_and_constraints.constraints
    scene_idx = list(constraints.names).index("scene_collision")
    scene_constraint = _to_numpy(constraints.values[scene_idx])
    robot_spheres = _to_numpy(raw_metrics.state.robot_spheres)
    if scene_constraint is None or robot_spheres is None:
        raise RuntimeError("failed to read scene constraint or robot spheres")

    samples = [
        _sample_report(
            sample=sample,
            robot_spheres=np.asarray(robot_spheres),
            scene_constraint=np.asarray(scene_constraint),
            scene_grid=scene.voxel[0],
            scene_cost=scene_cost,
            sphere_link_names=list(sphere_model.sphere_link_names),
            device=args.device,
        )
        for sample in SAMPLES
    ]
    cfg = _scene_cost_config_report(scene_cost)
    root, fix = _root_cause(samples, cfg)
    report = {
        "schema_version": 1,
        "route": "RouteB",
        "audit": "scene_collision_semantics_current_to_pregrasp",
        "inputs": {k: str(v) for k, v in inputs.items()},
        "joint_state_sanitization": sanitization_report,
        "solve_wall_time_s": solve_wall_time_s,
        "result_success": bool(_to_numpy(getattr(result, "success", None)).astype(bool).any()),
        "scene_grid": _scene_grid_report(scene.voxel[0]),
        "curobo_internal_voxel_data": _curobo_internal_voxel_report(scene_cost),
        "curobo_scene_collision_config": cfg,
        "distance_convention": (
            "Project query_spheres reports signed ESDF distance and collision when "
            "signed_distance <= radius. cuRobo voxel SDF is also positive outside and negative "
            "inside, but SceneCollisionCost converts it to positive constraint/cost via "
            "penetration = -signed_distance + radius + activation_distance."
        ),
        "activation_rule": (
            "For activation_distance=0, cuRobo apply_collision_activation is linear: "
            "constraint = weight * max(radius - signed_distance, 0)."
        ),
        "samples": samples,
        "scene_constraint_summary": scene_constraint_items[0],
        "root_cause": root,
        "minimal_fix": fix,
    }
    out = output_dir / "scene_collision_semantics_audit.json"
    out.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    print("[Route B scene_collision semantics audit]")
    for item in samples:
        print(
            f"{item['label']}: t={item['timestep']} sphere={item['sphere_index']} "
            f"link={item['link_name']} project_clearance={item['project_query']['clearance_m']:.6f} "
            f"curobo_constraint={item['curobo_rollout']['constraint_value']:.6f} "
            f"curobo_inferred_sdf={item['curobo_rollout']['inferred_signed_distance_m']:.6f}"
        )
    print(f"root_cause={root}")
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
