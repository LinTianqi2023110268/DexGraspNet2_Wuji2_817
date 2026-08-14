"""Run the first safe Isaac Lab closed-loop test for the right arm.

The program loads the already calibrated stage, audits the 35-DOF assembly,
holds every non-arm joint at its measured initial position, and moves the right
flange 20 mm upward and back.  The end-effector path uses quintic translation,
quaternion SLERP and damped-least-squares differential IK.

No root teleport, network inference, grasping or object motion is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/diagnostics/config/short_motion.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--translation-z-mm",
        type=float,
        default=None,
        help="Override only the world-Z smoke-test displacement from the JSON config.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=None,
        help="Override only the report directory from the JSON config.",
    )
    parser.add_argument(
        "--joint-target-npz",
        type=Path,
        default=None,
        help="Run a slow joint-space PREGRASP dry motion using pregrasp_read_only_ik.npz.",
    )
    parser.add_argument(
        "--waypoints-npz",
        type=Path,
        default=None,
        help="final_waypoints.npz containing the matching Wuji2 PREGRASP q20.",
    )
    parser.add_argument(
        "--flange-targets-npz",
        type=Path,
        default=None,
        help="arm_flange_targets.npz containing the matching PREGRASP flange pose.",
    )
    parser.add_argument("--joint-duration-s", type=float, default=22.0)
    parser.add_argument("--endpoint-refine-s", type=float, default=6.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=None)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_arguments()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import torch  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, quat_inv, subtract_frame_transforms  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402

from motion_math import (  # noqa: E402
    JointCommandLimiter,
    quaternion_error_deg,
    quaternion_slerp,
    quintic_time_scale,
)


TRACE_FIELDS = [
    "time_s",
    "state",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "actual_x_m",
    "actual_y_m",
    "actual_z_m",
    "position_error_m",
    "orientation_error_deg",
    "max_actual_joint_velocity_rad_s",
    "max_command_joint_velocity_rad_s",
]


def load_config(path: Path) -> dict:
    config = json.loads(path.resolve().read_text(encoding="utf-8"))
    if ARGS.translation_z_mm is not None:
        config["test_world_translation_m"] = [0.0, 0.0, float(ARGS.translation_z_mm) / 1000.0]
    if ARGS.output_directory is not None:
        config["output_directory"] = ARGS.output_directory
    if ARGS.position_tolerance_mm is not None:
        config["position_tolerance_m"] = float(ARGS.position_tolerance_mm) / 1000.0
    if ARGS.orientation_tolerance_deg is not None:
        config["orientation_tolerance_deg"] = float(ARGS.orientation_tolerance_deg)
    required = {
        "stage",
        "robot_prim",
        "flange_body",
        "right_arm_joints",
        "expected_total_actuated_joints",
        "physics_dt_s",
        "test_world_translation_m",
        "output_directory",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")
    if len(config["right_arm_joints"]) != 7:
        raise ValueError("right_arm_joints must contain exactly seven joints")
    return config


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def open_calibrated_stage(config: dict) -> Path:
    stage_path = project_path(config["stage"]).resolve()
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    if not stage_utils.open_stage(str(stage_path)):
        raise RuntimeError(f"Isaac Sim could not open stage: {stage_path}")
    return stage_path


def create_simulation(config: dict) -> SimulationContext:
    simulation_cfg = sim_utils.SimulationCfg(
        dt=float(config["physics_dt_s"]),
        render_interval=int(config["render_interval"]),
        device=ARGS.device,
    )
    return SimulationContext(simulation_cfg)


def set_force_drive_type_on_joint_prims(config: dict) -> None:
    """Switch only the right-arm USD joint drives to Force Drive before reset."""

    if config.get("right_arm_drive_type") != "force":
        return
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    robot_prim_path = config["robot_prim"]
    requested = set(config["right_arm_joints"])
    changed: list[str] = []
    for prim in stage.Traverse():
        if not prim.GetPath().pathString.startswith(robot_prim_path + "/"):
            continue
        if prim.GetName() not in requested:
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError(f"Right-arm joint prim is not a RevoluteJoint: {prim.GetPath()}")
        drive_api = UsdPhysics.DriveAPI(prim, "angular")
        if not drive_api:
            drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive_api.CreateTypeAttr("force")
        if not PhysxSchema.PhysxJointAPI(prim):
            PhysxSchema.PhysxJointAPI.Apply(prim)
        changed.append(prim.GetName())
    missing = sorted(requested.difference(changed))
    if missing:
        raise RuntimeError(f"Could not locate right-arm joint prims for force-drive switch: {missing}")
    print(f"[DRIVE TYPE OVERRIDE] right-arm USD DriveAPI type='force' for joints: {changed}")


def create_robot(config: dict) -> Articulation:
    groups = config.get("right_arm_force_natural_frequency_groups", [])
    if not groups:
        # Preserve all runtime drive values and physical limits from the USD.
        actuators = {
            "usd_native": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
            )
        }
    else:
        # Only J1-J7 get the tuned force-drive gains.  Left arm and Wuji2 keep
        # their authored USD drive settings.
        actuators = {
            "native_left_and_wuji2": ImplicitActuatorCfg(
                joint_names_expr=["arm_l_.*", "r_.*"],
                stiffness=None,
                damping=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
            )
        }
        for group in groups:
            actuators[group["name"]] = ImplicitActuatorCfg(
                joint_names_expr=group["joint_names_expr"],
                stiffness=None,
                damping=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
            )
    robot_cfg = ArticulationCfg(
        prim_path=config["robot_prim"],
        spawn=None,
        actuators=actuators,
    )
    return Articulation(robot_cfg)


def apply_force_natural_frequency_gains(robot: Articulation, arm_joint_ids: list[int], config: dict) -> list[dict]:
    """Apply ft04-style Force Drive K/D from the generalized mass diagonal."""

    groups = config.get("right_arm_force_natural_frequency_groups", [])
    if not groups:
        return []
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[0].to(robot.device)
    stiffness = robot.data.joint_stiffness.clone()
    damping = robot.data.joint_damping.clone()
    resolved: list[dict] = []
    covered: set[int] = set()

    for group in groups:
        joint_ids, joint_names = robot.find_joints(group["joint_names_expr"], preserve_order=True)
        joint_ids = [int(joint_id) for joint_id in joint_ids]
        for joint_id in joint_ids:
            if joint_id not in arm_joint_ids:
                raise RuntimeError(
                    f"Force gain group {group['name']} matched non-right-arm joint: {robot.joint_names[joint_id]}"
                )
        frequency = float(group["natural_frequency_rad_s"])
        zeta = float(group.get("damping_ratio", 1.0))
        if frequency <= 0.0 or zeta <= 0.0:
            raise ValueError(f"Invalid Force Drive natural-frequency group: {group}")
        equivalent_mass = torch.clamp(torch.diag(mass_matrix)[joint_ids], min=1.0e-6)
        k_values = equivalent_mass * frequency * frequency
        d_values = 2.0 * zeta * frequency * equivalent_mass
        stiffness[0, joint_ids] = k_values
        damping[0, joint_ids] = d_values
        covered.update(joint_ids)
        resolved.append(
            {
                **group,
                "matched_joint_names": joint_names,
                "equivalent_mass_diagonal": [float(value) for value in equivalent_mass.detach().cpu()],
                "stiffness_by_joint": {name: float(value) for name, value in zip(joint_names, k_values.detach().cpu())},
                "damping_by_joint": {name: float(value) for name, value in zip(joint_names, d_values.detach().cpu())},
                "gain_source": "force_drive_mass_matrix_natural_frequency",
            }
        )
    if sorted(covered) != sorted(arm_joint_ids):
        missing = [robot.joint_names[joint_id] for joint_id in arm_joint_ids if joint_id not in covered]
        raise RuntimeError(f"Force natural-frequency groups did not cover all right-arm joints: {missing}")

    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.reset()
    print(f"[FORCE DRIVE GAINS] Applied tuned gains: {resolved}")
    return resolved


def audit_robot(robot: Articulation, config: dict) -> tuple[list[int], int]:
    expected_count = int(config["expected_total_actuated_joints"])
    if robot.num_joints != expected_count:
        raise RuntimeError(f"Expected {expected_count} joints, found {robot.num_joints}")
    if not robot.is_fixed_base:
        raise RuntimeError("The assembled dual arm must be fixed-base")

    joint_ids, joint_names = robot.find_joints(config["right_arm_joints"], preserve_order=True)
    if joint_names != config["right_arm_joints"]:
        raise RuntimeError(f"Right-arm joint order mismatch: {joint_names}")

    body_ids, body_names = robot.find_bodies([config["flange_body"]], preserve_order=True)
    if body_names != [config["flange_body"]]:
        raise RuntimeError(f"Flange body mismatch: {body_names}")

    print("[AUDIT PASS] fixed-base articulation")
    print(f"[AUDIT PASS] total actuated joints: {robot.num_joints}")
    print(f"[AUDIT PASS] right arm joints: {joint_names}")
    print(f"[AUDIT PASS] flange body: {body_names[0]}")
    return joint_ids, body_ids[0]


def physics_step(sim: SimulationContext, robot: Articulation, dt: float) -> None:
    robot.write_data_to_sim()
    sim.step()
    robot.update(dt)


def hold_initial_state(
    sim: SimulationContext,
    robot: Articulation,
    hold_target: torch.Tensor,
    duration_s: float,
    dt: float,
) -> None:
    step_count = max(1, round(duration_s / dt))
    robot.set_joint_position_target(hold_target)
    for _ in range(step_count):
        physics_step(sim, robot, dt)


def initialize_joint_state(
    sim: SimulationContext,
    robot: Articulation,
    arm_joint_ids: list[int],
    config: dict,
) -> torch.Tensor:
    """Write one deterministic state before the first physics step.

    This follows Isaac Lab's official reset sequence.  It avoids asking the
    USD drives to move from their runtime reset pose to the calibrated pose.
    """
    joint_position = robot.data.joint_pos.clone()
    right_arm_rad = torch.deg2rad(
        torch.tensor(
            config["initial_right_arm_joint_deg"],
            device=robot.device,
            dtype=joint_position.dtype,
        )
    ).reshape(1, 7)
    joint_position[:, arm_joint_ids] = right_arm_rad
    joint_velocity = torch.zeros_like(joint_position)

    robot.write_joint_state_to_sim(joint_position, joint_velocity)
    robot.reset()
    robot.set_joint_position_target(joint_position)
    physics_step(sim, robot, float(config["physics_dt_s"]))
    return joint_position


def audit_settled_arm(robot: Articulation, arm_joint_ids: list[int], config: dict) -> None:
    print("[RUNTIME DRIVE AUDIT] right-arm values after initial hold")
    for local_index, joint_id in enumerate(arm_joint_ids):
        name = config["right_arm_joints"][local_index]
        print(
            f"  {name}: "
            f"q={float(robot.data.joint_pos[0, joint_id]):+.4f} rad, "
            f"qd={float(robot.data.joint_vel[0, joint_id]):+.4f} rad/s, "
            f"target={float(robot.data.joint_pos_target[0, joint_id]):+.4f} rad, "
            f"K={float(robot.data.joint_stiffness[0, joint_id]):.3f}, "
            f"D={float(robot.data.joint_damping[0, joint_id]):.3f}, "
            f"effort_limit={float(robot.data.joint_effort_limits[0, joint_id]):.3f}, "
            f"velocity_limit={float(robot.data.joint_vel_limits[0, joint_id]):.3f}"
        )
    max_velocity = float(torch.max(torch.abs(robot.data.joint_vel[:, arm_joint_ids])))
    print(f"[STARTUP] raw reported max right-arm joint velocity: {max_velocity:.4f} rad/s")
    print("[STARTUP] raw qdot is diagnostic only; ft04 validation uses position span and finite-difference velocity.")


def flange_pose(robot: Articulation, flange_body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    pose = robot.data.body_pose_w[:, flange_body_id]
    return pose[:, :3].clone(), pose[:, 3:7].clone()


def jacobian_index(robot: Articulation, flange_body_id: int) -> int:
    return flange_body_id - 1 if robot.is_fixed_base else flange_body_id


def clamp_to_joint_limits(
    position: torch.Tensor,
    robot: Articulation,
    arm_joint_ids: list[int],
    margin_rad: float,
) -> torch.Tensor:
    limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids]
    lower = limits[:, :, 0] + float(margin_rad)
    upper = limits[:, :, 1] - float(margin_rad)
    return torch.minimum(torch.maximum(position, lower), upper)


def append_trace(
    trace: list[dict],
    simulation_time_s: float,
    state: str,
    target_position: torch.Tensor,
    target_quaternion: torch.Tensor,
    actual_position: torch.Tensor,
    actual_quaternion: torch.Tensor,
    actual_joint_velocity: torch.Tensor,
    command_joint_velocity: torch.Tensor,
) -> tuple[float, float]:
    position_error = float(torch.linalg.vector_norm(target_position - actual_position))
    orientation_error = quaternion_error_deg(actual_quaternion, target_quaternion)
    trace.append(
        {
            "time_s": simulation_time_s,
            "state": state,
            "target_x_m": float(target_position[0, 0]),
            "target_y_m": float(target_position[0, 1]),
            "target_z_m": float(target_position[0, 2]),
            "actual_x_m": float(actual_position[0, 0]),
            "actual_y_m": float(actual_position[0, 1]),
            "actual_z_m": float(actual_position[0, 2]),
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
            "max_actual_joint_velocity_rad_s": float(torch.max(torch.abs(actual_joint_velocity))),
            "max_command_joint_velocity_rad_s": float(torch.max(torch.abs(command_joint_velocity))),
        }
    )
    return position_error, orientation_error


def run_segment(
    name: str,
    start_position_w: torch.Tensor,
    start_quaternion_w: torch.Tensor,
    goal_position_w: torch.Tensor,
    goal_quaternion_w: torch.Tensor,
    duration_s: float,
    sim: SimulationContext,
    robot: Articulation,
    controller: DifferentialIKController,
    limiter: JointCommandLimiter,
    static_target_bias: torch.Tensor,
    hold_target: torch.Tensor,
    arm_joint_ids: list[int],
    flange_body_id: int,
    config: dict,
    trace: list[dict],
    start_time_s: float,
) -> float:
    dt = float(config["physics_dt_s"])
    step_count = max(2, round(duration_s / dt))
    telemetry_stride = max(1, round(1.0 / (float(config["telemetry_hz"]) * dt)))
    jacobi_index = jacobian_index(robot, flange_body_id)

    print(f"\n>>> STATE: {name} | duration={duration_s:.2f}s | steps={step_count}")
    for step in range(step_count + 1):
        progress = step / step_count
        alpha = quintic_time_scale(progress)
        target_position_w = start_position_w + alpha * (goal_position_w - start_position_w)
        target_quaternion_w = quaternion_slerp(start_quaternion_w, goal_quaternion_w, alpha)

        root_pose_w = robot.data.root_pose_w
        target_position_b, target_quaternion_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            target_position_w,
            target_quaternion_w,
        )
        controller.set_command(torch.cat((target_position_b, target_quaternion_b), dim=1))

        actual_position_w, actual_quaternion_w = flange_pose(robot, flange_body_id)
        actual_position_b, actual_quaternion_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            actual_position_w,
            actual_quaternion_w,
        )
        # PhysX reports this geometric Jacobian in the world frame.  The target
        # and measured flange poses above are expressed in the articulation-root
        # frame, so rotate both the linear and angular Jacobian blocks into that
        # same frame.  This follows Isaac Lab 2.2's DifferentialInverseKinematicsAction.
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_index, :, arm_joint_ids].clone()
        world_to_root_rotation = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jacobian[:, :3, :] = torch.bmm(world_to_root_rotation, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(world_to_root_rotation, jacobian[:, 3:, :])
        current_arm_position = robot.data.joint_pos[:, arm_joint_ids]
        requested_arm_position = controller.compute(
            actual_position_b,
            actual_quaternion_b,
            jacobian,
            current_arm_position,
        )
        # At the gravity-loaded static equilibrium, q_target is intentionally
        # different from q_actual.  That small position error is what produces
        # the holding torque of the implicit Force Drive.  Differential IK
        # returns q_actual + delta_q; applying it directly would erase the
        # static holding error on the first motion step.  Preserve the measured
        # equilibrium target bias while adding the IK correction.
        requested_arm_position = requested_arm_position + static_target_bias
        requested_arm_position = clamp_to_joint_limits(
            requested_arm_position,
            robot,
            arm_joint_ids,
            config["joint_limit_margin_rad"],
        )
        limited_arm_position = limiter.step(requested_arm_position)

        hold_target[:, arm_joint_ids] = limited_arm_position
        robot.set_joint_position_target(hold_target)
        physics_step(sim, robot, dt)

        actual_position_w, actual_quaternion_w = flange_pose(robot, flange_body_id)
        position_error, orientation_error = append_trace(
            trace,
            start_time_s + step * dt,
            name,
            target_position_w,
            target_quaternion_w,
            actual_position_w,
            actual_quaternion_w,
            robot.data.joint_vel[:, arm_joint_ids],
            limiter.velocity,
        )
        if step % telemetry_stride == 0 or step == step_count:
            print(
                f"\r{name:<10} {100.0 * progress:6.1f}% | "
                f"position_error={1000.0 * position_error:6.2f} mm | "
                f"rotation_error={orientation_error:5.2f} deg",
                end="",
                flush=True,
            )
    print()
    return start_time_s + (step_count + 1) * dt


def run_joint_target_segment(
    name: str,
    start_arm_target: torch.Tensor,
    goal_arm_target: torch.Tensor,
    start_flange_pose: tuple[torch.Tensor, torch.Tensor],
    goal_flange_pose: tuple[torch.Tensor, torch.Tensor],
    duration_s: float,
    sim: SimulationContext,
    robot: Articulation,
    hold_target: torch.Tensor,
    arm_joint_ids: list[int],
    flange_body_id: int,
    config: dict,
    trace: list[dict],
    start_time_s: float,
) -> float:
    """Execute one bounded quintic joint-position segment.

    This mode deliberately bypasses Differential IK because the endpoint was
    already solved and audited offline.  It still uses the same ft04 implicit
    Force Drive and the same physical effort limits as the validated baseline.
    """

    dt = float(config["physics_dt_s"])
    step_count = max(2, round(duration_s / dt))
    telemetry_stride = max(1, round(1.0 / (float(config["telemetry_hz"]) * dt)))
    previous_command = start_arm_target.clone()

    print(f"\n>>> STATE: {name} | joint-space quintic | duration={duration_s:.2f}s | steps={step_count}")
    for step in range(step_count + 1):
        progress = step / step_count
        alpha = quintic_time_scale(progress)
        command = start_arm_target + alpha * (goal_arm_target - start_arm_target)
        command_velocity = (command - previous_command) / dt if step else torch.zeros_like(command)
        previous_command = command.clone()

        hold_target[:, arm_joint_ids] = command
        robot.set_joint_position_target(hold_target)
        physics_step(sim, robot, dt)

        target_position = start_flange_pose[0] + alpha * (goal_flange_pose[0] - start_flange_pose[0])
        target_quaternion = quaternion_slerp(start_flange_pose[1], goal_flange_pose[1], alpha)
        actual_position, actual_quaternion = flange_pose(robot, flange_body_id)
        position_error, orientation_error = append_trace(
            trace,
            start_time_s + step * dt,
            name,
            target_position,
            target_quaternion,
            actual_position,
            actual_quaternion,
            robot.data.joint_vel[:, arm_joint_ids],
            command_velocity,
        )
        if step % telemetry_stride == 0 or step == step_count:
            print(
                f"\r{name:<10} {100.0 * progress:6.1f}% | "
                f"position_error={1000.0 * position_error:6.2f} mm | "
                f"rotation_error={orientation_error:5.2f} deg",
                end="",
                flush=True,
            )
    print()
    return start_time_s + (step_count + 1) * dt


def move_hand_to_pregrasp(
    sim: SimulationContext,
    robot: Articulation,
    hold_target: torch.Tensor,
    waypoint_path: Path,
    duration_s: float,
    dt: float,
) -> None:
    """Slowly form the exact generated PREGRASP while the right arm is parked."""

    with np.load(waypoint_path, allow_pickle=False) as archive:
        names = [str(value) for value in archive["finger_joint_names"].tolist()]
        stages = np.asarray(archive["waypoint_names"])
        matches = np.flatnonzero(stages == "pregrasp")
        if matches.size != 1:
            raise RuntimeError("PREGRASP hand waypoint is ambiguous")
        goal_values = np.asarray(
            archive["waypoint_joint_positions"][0, int(matches[0])], dtype=np.float32
        )
    joint_ids, matched_names = robot.find_joints(names, preserve_order=True)
    if matched_names != names:
        raise RuntimeError(f"Wuji2 PREGRASP joint order mismatch: {matched_names}")
    start = hold_target[:, joint_ids].clone()
    goal = torch.as_tensor(goal_values, device=robot.device, dtype=start.dtype).reshape(1, -1)
    steps = max(2, round(duration_s / dt))
    print(f"\n>>> STATE: FORM_HAND_PREGRASP | duration={duration_s:.2f}s | steps={steps}")
    for step in range(steps + 1):
        alpha = quintic_time_scale(step / steps)
        hold_target[:, joint_ids] = start + alpha * (goal - start)
        robot.set_joint_position_target(hold_target)
        physics_step(sim, robot, dt)
    print("[HAND] PREGRASP target reached at parked arm pose")


def write_outputs(
    config: dict,
    stage_path: Path,
    trace: list[dict],
    start_pose: tuple[torch.Tensor, torch.Tensor],
    end_pose: tuple[torch.Tensor, torch.Tensor],
    resolved_force_drive_gains: list[dict],
) -> Path:
    output_dir = project_path(config["output_directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(trace)

    final_position_error = float(torch.linalg.vector_norm(end_pose[0] - start_pose[0]))
    final_orientation_error = quaternion_error_deg(end_pose[1], start_pose[1])
    measured_velocity = [row["max_actual_joint_velocity_rad_s"] for row in trace]
    command_velocity = [row["max_command_joint_velocity_rad_s"] for row in trace]
    passed = (
        final_position_error <= float(config["position_tolerance_m"])
        and final_orientation_error <= float(config["orientation_tolerance_deg"])
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "stage": str(stage_path),
        "right_arm_drive_type": config.get("right_arm_drive_type", "native"),
        "resolved_force_drive_gains": resolved_force_drive_gains,
        "test_world_translation_m": config["test_world_translation_m"],
        "final_return_position_error_m": final_position_error,
        "final_return_orientation_error_deg": final_orientation_error,
        "max_raw_reported_joint_velocity_rad_s": max(measured_velocity) if measured_velocity else 0.0,
        "max_command_joint_velocity_rad_s": max(command_velocity) if command_velocity else 0.0,
        "velocity_note": (
            "Raw PhysX joint velocity is diagnostic only. ft04 static audit showed a TGS telemetry anomaly; "
            "use Cartesian tracking, finite-difference motion, and final return error for this minimal IK smoke test."
        ),
        "limits": {
            "max_joint_velocity_rad_s": config["max_joint_velocity_rad_s"],
            "max_joint_acceleration_rad_s2": config["max_joint_acceleration_rad_s2"],
            "position_tolerance_m": config["position_tolerance_m"],
            "orientation_tolerance_deg": config["orientation_tolerance_deg"],
        },
        "trace_csv": str(trace_path),
    }
    pregrasp_rows = [
        row for row in trace if row["state"] in {"TO_PREGRASP", "REFINE_PREGRASP"}
    ]
    if pregrasp_rows:
        endpoint = pregrasp_rows[-1]
        report["pregrasp_endpoint_position_error_m"] = endpoint["position_error_m"]
        report["pregrasp_endpoint_orientation_error_deg"] = endpoint["orientation_error_deg"]
        report["status"] = (
            "PASS"
            if passed
            and endpoint["position_error_m"] <= float(config["position_tolerance_m"])
            and endpoint["orientation_error_deg"] <= float(config["orientation_tolerance_deg"])
            else "FAIL"
        )
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def run() -> Path:
    config = load_config(ARGS.config)
    stage_path = open_calibrated_stage(config)
    set_force_drive_type_on_joint_prims(config)
    sim = create_simulation(config)
    robot = create_robot(config)
    sim.reset()

    dt = float(config["physics_dt_s"])
    robot.update(dt)
    arm_joint_ids, flange_body_id = audit_robot(robot, config)
    resolved_force_drive_gains = apply_force_natural_frequency_gains(robot, arm_joint_ids, config)

    hold_target = initialize_joint_state(sim, robot, arm_joint_ids, config)
    hold_initial_state(sim, robot, hold_target, config["initial_hold_s"], dt)
    audit_settled_arm(robot, arm_joint_ids, config)
    start_pose = flange_pose(robot, flange_body_id)

    # Freeze the gravity-loaded target offset established by the validated ft04
    # static hold.  The short-motion smoke test must start from this equilibrium
    # instead of resetting the drive target to the measured joint position.
    settled_arm_position = robot.data.joint_pos[:, arm_joint_ids].clone()
    static_target_bias = hold_target[:, arm_joint_ids].clone() - settled_arm_position
    print(
        "[STATIC TARGET BIAS] q_target - q_actual (deg): "
        f"{torch.rad2deg(static_target_bias).detach().cpu().tolist()[0]}"
    )

    if ARGS.joint_target_npz is not None:
        if ARGS.waypoints_npz is None or ARGS.flange_targets_npz is None:
            raise ValueError("joint-target mode requires --waypoints-npz and --flange-targets-npz")
        joint_target_path = ARGS.joint_target_npz.resolve()
        waypoint_path = ARGS.waypoints_npz.resolve()
        flange_target_path = ARGS.flange_targets_npz.resolve()
        for path in (joint_target_path, waypoint_path, flange_target_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        move_hand_to_pregrasp(sim, robot, hold_target, waypoint_path, 4.0, dt)
        hold_initial_state(sim, robot, hold_target, 2.0, dt)
        start_pose = flange_pose(robot, flange_body_id)
        with np.load(joint_target_path, allow_pickle=False) as archive:
            if not bool(np.asarray(archive["reachable"]).item()):
                raise RuntimeError("refusing motion: read-only PREGRASP IK did not pass")
            solved_q = np.asarray(archive["solved_right_arm_q_rad"], dtype=np.float32)
        with np.load(flange_target_path, allow_pickle=False) as archive:
            stages = np.asarray(archive["waypoint_names"])
            matches = np.flatnonzero(stages == "pregrasp")
            if matches.size != 1:
                raise RuntimeError("PREGRASP flange target is ambiguous")
            goal_matrix = np.asarray(
                archive["world_from_right_flange"][int(matches[0])], dtype=np.float32
            )
        goal_arm_target = torch.as_tensor(
            solved_q, device=robot.device, dtype=hold_target.dtype
        ).reshape(1, 7) + static_target_bias
        goal_position = torch.as_tensor(
            goal_matrix[:3, 3], device=robot.device, dtype=start_pose[0].dtype
        ).reshape(1, 3)
        goal_rotation = torch.as_tensor(
            goal_matrix[:3, :3], device=robot.device, dtype=start_pose[0].dtype
        ).reshape(1, 3, 3)
        goal_quaternion = quat_from_matrix(goal_rotation)

        initial_arm_target = hold_target[:, arm_joint_ids].clone()
        trace: list[dict] = []
        time_s = run_joint_target_segment(
            "TO_PREGRASP",
            initial_arm_target,
            goal_arm_target,
            start_pose,
            (goal_position, goal_quaternion),
            float(ARGS.joint_duration_s),
            sim,
            robot,
            hold_target,
            arm_joint_ids,
            flange_body_id,
            config,
            trace,
            0.0,
        )
        controller_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": float(config["dls_damping"])},
        )
        controller = DifferentialIKController(controller_cfg, num_envs=1, device=sim.device)
        controller.reset()
        refine_limiter = JointCommandLimiter(
            hold_target[:, arm_joint_ids].clone(),
            dt,
            config["max_joint_velocity_rad_s"],
            config["max_joint_acceleration_rad_s2"],
        )
        time_s = run_segment(
            "REFINE_PREGRASP",
            goal_position,
            goal_quaternion,
            goal_position,
            goal_quaternion,
            float(ARGS.endpoint_refine_s),
            sim,
            robot,
            controller,
            refine_limiter,
            static_target_bias,
            hold_target,
            arm_joint_ids,
            flange_body_id,
            config,
            trace,
            time_s,
        )
        hold_initial_state(sim, robot, hold_target, 3.0, dt)
        time_s += 3.0
        refined_arm_target = hold_target[:, arm_joint_ids].clone()
        time_s = run_joint_target_segment(
            "RETURN",
            refined_arm_target,
            initial_arm_target,
            (goal_position, goal_quaternion),
            start_pose,
            float(ARGS.joint_duration_s),
            sim,
            robot,
            hold_target,
            arm_joint_ids,
            flange_body_id,
            config,
            trace,
            time_s,
        )
        hold_initial_state(sim, robot, hold_target, config["final_hold_s"], dt)
        end_pose = flange_pose(robot, flange_body_id)
        return write_outputs(
            config, stage_path, trace, start_pose, end_pose, resolved_force_drive_gains
        )

    controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": float(config["dls_damping"])},
    )
    controller = DifferentialIKController(controller_cfg, num_envs=1, device=sim.device)
    controller.reset()

    initial_arm_position = hold_target[:, arm_joint_ids].clone()
    limiter = JointCommandLimiter(
        initial_arm_position,
        dt,
        config["max_joint_velocity_rad_s"],
        config["max_joint_acceleration_rad_s2"],
    )

    translation = torch.tensor(
        config["test_world_translation_m"],
        device=sim.device,
        dtype=start_pose[0].dtype,
    ).reshape(1, 3)
    upper_pose = (start_pose[0] + translation, start_pose[1].clone())

    trace: list[dict] = []
    time_s = 0.0
    time_s = run_segment(
        "OUTBOUND",
        *start_pose,
        *upper_pose,
        config["outbound_duration_s"],
        sim,
        robot,
        controller,
        limiter,
        static_target_bias,
        hold_target,
        arm_joint_ids,
        flange_body_id,
        config,
        trace,
        time_s,
    )
    hold_initial_state(sim, robot, hold_target, config["endpoint_hold_s"], dt)
    time_s += float(config["endpoint_hold_s"])
    time_s = run_segment(
        "RETURN",
        *upper_pose,
        *start_pose,
        config["return_duration_s"],
        sim,
        robot,
        controller,
        limiter,
        static_target_bias,
        hold_target,
        arm_joint_ids,
        flange_body_id,
        config,
        trace,
        time_s,
    )
    hold_initial_state(sim, robot, hold_target, config["final_hold_s"], dt)

    end_pose = flange_pose(robot, flange_body_id)
    return write_outputs(config, stage_path, trace, start_pose, end_pose, resolved_force_drive_gains)


def main() -> int:
    try:
        report_path = run()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"\n[SHORT MOTION {report['status']}] {report_path}")
        print(f"return position error: {1000.0 * report['final_return_position_error_m']:.3f} mm")
        print(f"return orientation error: {report['final_return_orientation_error_deg']:.3f} deg")
        return 0 if report["status"] == "PASS" else 2
    except Exception as error:
        print(f"\n[SHORT MOTION FAILED] {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        SIMULATION_APP.close()


if __name__ == "__main__":
    raise SystemExit(main())
