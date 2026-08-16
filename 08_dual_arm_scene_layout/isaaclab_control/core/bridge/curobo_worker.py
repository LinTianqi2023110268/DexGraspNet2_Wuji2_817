#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

_PROTOCOL = "__CUROBO_CORE__"


def emit(payload: dict) -> None:
    print(_PROTOCOL + json.dumps(payload, separators=(",", ":")), flush=True)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stdio", action="store_true")
    args = parser.parse_args()

    core_parent = args.project_root / "08_dual_arm_scene_layout/isaaclab_control"
    sys.path.insert(0, str(core_parent))
    from core.config import (
        IKConfig,
        MapperConfig,
        DEFAULT_INITIAL_RIGHT_ARM_DEG,
        RIGHT_ARM_NAMES,
    )
    from core.ik import CuroboGpuIK, select_waypoint_chain, select_solution
    from core.perception_collision import (
        RGBDFrame, CuroboRGBDMapper, CuroboRobotSphereModel,
    )

    robot_urdf = (
        args.project_root
        / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
    )
    ik = CuroboGpuIK(
        robot_urdf,
        IKConfig(
            device=args.device,
            num_seeds=args.seeds,
            batch_size=args.batch_size,
            return_seeds=args.seeds,
        ),
    )
    mapper = CuroboRGBDMapper(MapperConfig(device=args.device))
    observed_map = None
    robot_sphere_model = None
    robot_collision_config = (
        args.project_root
        / "08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml"
    )

    def get_robot_sphere_model():
        nonlocal robot_sphere_model
        if robot_sphere_model is None:
            robot_sphere_model = CuroboRobotSphereModel(
                robot_collision_config, device=args.device
            )
        return robot_sphere_model

    def ik_solution_record(result, target_index: int, solution_index: int) -> dict:
        i, k = int(target_index), int(solution_index)
        return {
            "target_index": i,
            "solution_index": k,
            "q_rad": result.q_rad[i, k].tolist(),
            "position_error_m": float(result.position_error_m[i, k]),
            "orientation_error_rad": float(result.orientation_error_rad[i, k]),
            "inner_limit_margin_rad": float(result.inner_limit_margin_rad[i, k]),
        }

    def collision_filter_ik(result, context: dict) -> tuple[list[list[dict]], list[list[dict]]]:
        if observed_map is None:
            raise RuntimeError("build_map must be called before collision-aware solve_ik")
        phases = [str(x) for x in context["phases"]]
        if len(phases) != result.batch_size:
            raise ValueError(
                f"collision phases must match IK batch: {len(phases)} != {result.batch_size}"
            )
        states = context.get("joint_positions_by_target")
        if states is None:
            baseline = context["joint_positions_by_name"]
            states = [baseline for _ in range(result.batch_size)]
        if len(states) != result.batch_size:
            raise ValueError(
                "joint_positions_by_target must contain one named state per IK target"
            )
        T_world_base = np.asarray(context["T_world_base"], dtype=np.float64)
        margin_m = float(context.get("margin_m", 0.0))
        model = get_robot_sphere_model()
        required_names = set(model.joint_names)
        for i, state in enumerate(states):
            missing = sorted(required_names - set(state))
            if missing:
                raise KeyError(
                    "production collision state must provide every active joint; "
                    f"target_index={i}, missing={missing}"
                )

        ik_accepted = result.accepted.copy()
        audited: list[list[dict]] = []
        feasible: list[list[dict]] = []
        for i in range(result.batch_size):
            target_audit: list[dict] = []
            target_feasible: list[dict] = []
            for k in np.flatnonzero(ik_accepted[i]):
                named = {str(name): float(value) for name, value in states[i].items()}
                for name, value in zip(RIGHT_ARM_NAMES, result.q_rad[i, k]):
                    named[name] = float(value)
                spheres = model.spheres_from_named_joints(named, T_world_base)
                collision = observed_map.check_spheres(
                    spheres[:, :3], spheres[:, 3], phases[i], margin_m
                )
                scene_count = int(np.count_nonzero(collision["scene_collision"]))
                target_count = int(np.count_nonzero(collision["target_collision"]))
                blocking_count = int(np.count_nonzero(collision["blocking_collision"]))
                unknown_count = int(np.count_nonzero(collision["unknown"]))
                record = ik_solution_record(result, i, int(k))
                record.update({
                    "phase": phases[i],
                    "observed_scene_collision_pass": blocking_count == 0,
                    "unknown_space_exposure": unknown_count > 0,
                    "blocking_collision_sphere_count": blocking_count,
                    "scene_collision_sphere_count": scene_count,
                    "target_collision_sphere_count": target_count,
                    "unknown_sphere_count": unknown_count,
                    "robot_sphere_count": int(len(spheres)),
                })
                target_audit.append(record)
                if blocking_count == 0:
                    target_feasible.append(record)
                else:
                    result.accepted[i, k] = False
            audited.append(target_audit)
            feasible.append(target_feasible)
        return audited, feasible

    if not args.stdio:
        print(f"cuRobo={ik.version}; joints={ik.joint_names}")
        return 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "ping":
                emit({"ok": True, "op": "pong", "curobo_version": ik.version, "joint_names": list(ik.joint_names)})
            elif op == "shutdown":
                emit({"ok": True, "op": "shutdown"})
                return 0
            elif op == "solve_ik":
                targets = np.asarray(req["targets"], dtype=np.float64)
                q_ref = np.asarray(req.get("q_reference_rad", np.deg2rad(DEFAULT_INITIAL_RIGHT_ARM_DEG)), dtype=np.float64)
                result = ik.solve(targets)
                ik_accepted_per_target = result.accepted.sum(axis=1).tolist()
                collision_context = req.get("collision_context")
                if collision_context is None:
                    ik_accepted_solutions = [
                        [ik_solution_record(result, i, int(k)) for k in np.flatnonzero(result.accepted[i])]
                        for i in range(result.batch_size)
                    ]
                    feasible_solutions = ik_accepted_solutions
                else:
                    ik_accepted_solutions, feasible_solutions = collision_filter_ik(
                        result, collision_context
                    )
                if bool(req.get("select_chain", True)):
                    selected = select_waypoint_chain(result, q_ref)
                else:
                    selected = [select_solution(result, i, q_ref) for i in range(result.batch_size)]
                selected_collision = None
                if selected is not None and collision_context is not None:
                    selected_collision = []
                    for pick in selected:
                        match = next(
                            x for x in feasible_solutions[pick.target_index]
                            if x["solution_index"] == pick.solution_index
                        )
                        selected_collision.append(match)
                emit({
                    "ok": True,
                    "op": "solve_ik",
                    "accepted_per_target": result.accepted.sum(axis=1).tolist(),
                    "ik_accepted_per_target": ik_accepted_per_target,
                    "raw_success_per_target": result.raw_success.sum(axis=1).tolist(),
                    "ik_accepted_solutions": ik_accepted_solutions,
                    "feasible_solutions": feasible_solutions,
                    "selected": None if selected is None else [None if x is None else x.to_jsonable() for x in selected],
                    "selected_collision": selected_collision,
                    "ik_pass": bool(all(int(x) > 0 for x in ik_accepted_per_target)),
                    "observed_scene_collision_pass": (
                        None if collision_context is None else selected is not None
                    ),
                    "unknown_space_exposure": (
                        None if selected_collision is None
                        else [bool(x["unknown_space_exposure"]) for x in selected_collision]
                    ),
                    "solve_time_s": result.solve_time_s,
                })
            elif op == "build_map":
                frame = RGBDFrame.from_npy(
                    req["depth_path"],
                    req["intrinsics_path"],
                    req["T_world_camera_path"],
                    req.get("target_mask_path"),
                )
                observed_map = mapper.build(frame)
                emit({
                    "ok": True,
                    "op": "build_map",
                    "map_id": observed_map.map_id,
                    "grid_center_world": observed_map.grid_center_world.tolist(),
                    "extent_meters_xyz": observed_map.extent_meters_xyz.tolist(),
                    "has_target_layer": observed_map.target_grid is not None,
                })
            elif op == "query_spheres":
                if observed_map is None:
                    raise RuntimeError("build_map must be called before query_spheres")
                result = observed_map.check_spheres(
                    np.asarray(req["centers_world"], dtype=np.float64),
                    np.asarray(req["radii_m"], dtype=np.float64),
                    req["phase"],
                    float(req.get("margin_m", 0.0)),
                )
                emit({"ok": True, "op": "query_spheres", "result": jsonable(result)})
            elif op == "robot_spheres":
                model = get_robot_sphere_model()
                T_world_base = req.get("T_world_base")
                spheres = model.spheres_from_named_joints(
                    {str(k): float(v) for k, v in req["joint_positions_by_name"].items()},
                    None if T_world_base is None else np.asarray(T_world_base, dtype=np.float64),
                )
                emit({
                    "ok": True,
                    "op": "robot_spheres",
                    "joint_names": list(model.joint_names),
                    "sphere_count": int(len(spheres)),
                    "spheres_world_xyzw_radius": spheres.tolist(),
                })
            elif op == "check_robot_state":
                if observed_map is None:
                    raise RuntimeError("build_map must be called before check_robot_state")
                model = get_robot_sphere_model()
                spheres = model.spheres_from_named_joints(
                    {str(k): float(v) for k, v in req["joint_positions_by_name"].items()},
                    np.asarray(req["T_world_base"], dtype=np.float64),
                )
                result = observed_map.check_spheres(
                    spheres[:, :3],
                    spheres[:, 3],
                    req["phase"],
                    float(req.get("margin_m", 0.0)),
                )
                emit({
                    "ok": True,
                    "op": "check_robot_state",
                    "sphere_count": int(len(spheres)),
                    "result": jsonable(result),
                })
            else:
                raise ValueError(f"unknown worker op: {op}")
        except Exception as exc:
            emit({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
