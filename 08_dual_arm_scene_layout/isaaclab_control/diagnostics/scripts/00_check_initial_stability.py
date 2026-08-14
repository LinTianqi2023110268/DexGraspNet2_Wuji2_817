"""Static startup stability gate for the calibrated dual arm and Wuji2 hand.

The saved 35-DOF state is written once and then held for 12 seconds.  This
program never runs IK, moves the articulation root, grasps an object, or
overrides drive/dynamics/contact values authored in the assembled USD.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/diagnostics/config/initial_stability.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_arguments()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import torch  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402


TRACE_FIELDS = [
    "time_s", "joint_name", "group", "position_rad", "target_rad", "error_rad",
    "velocity_rad_s", "physx_actuation_force_nm", "physx_projected_joint_force_nm",
    "effort_limit_nm",
    "explicit_pd_p_nm", "explicit_pd_d_nm", "gravity_compensation_nm",
    "explicit_raw_torque_nm", "explicit_applied_torque_nm", "explicit_saturation_ratio",
]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def acceleration_natural_frequency_to_pd(group: dict) -> dict:
    """Convert acceleration-drive natural frequency gains to explicit K/D fields."""
    natural_frequency = float(group["natural_frequency_rad_s"])
    damping_ratio = float(group.get("damping_ratio", 1.0))
    if natural_frequency <= 0.0:
        raise ValueError(f"natural_frequency_rad_s must be positive: {group}")
    if damping_ratio <= 0.0:
        raise ValueError(f"damping_ratio must be positive: {group}")
    converted = dict(group)
    converted["stiffness"] = natural_frequency * natural_frequency
    converted["damping"] = 2.0 * damping_ratio * natural_frequency
    converted["gain_source"] = "acceleration_natural_frequency"
    converted["gain_formula"] = "K=f^2, D=2*zeta*f, zeta fixed by config"
    return converted


def resolve_right_arm_pd_groups(config: dict) -> list[dict]:
    if config.get("right_arm_pd_groups") and config.get("right_arm_natural_frequency_groups"):
        raise ValueError("Use either right_arm_pd_groups or right_arm_natural_frequency_groups, not both")
    if config.get("right_arm_natural_frequency_groups"):
        return [acceleration_natural_frequency_to_pd(group) for group in config["right_arm_natural_frequency_groups"]]
    return config.get("right_arm_pd_groups", [])


def set_force_drive_type_on_joint_prims(config: dict) -> None:
    """Switch only the configured right-arm USD joint drives to Force Drive before PhysX reset."""
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
        physx_joint_api = PhysxSchema.PhysxJointAPI(prim)
        if not physx_joint_api:
            PhysxSchema.PhysxJointAPI.Apply(prim)
        changed.append(prim.GetName())
    missing = sorted(requested.difference(changed))
    if missing:
        raise RuntimeError(f"Could not locate right-arm joint prims for force-drive switch: {missing}")
    print(f"[DRIVE TYPE OVERRIDE] right-arm USD DriveAPI type='force' for joints: {changed}")


def create_robot(config: dict) -> Articulation:
    explicit = config.get("right_arm_explicit_pd")
    pd_groups = resolve_right_arm_pd_groups(config)
    if explicit and (pd_groups or config.get("right_arm_force_natural_frequency_groups")):
        raise ValueError("Explicit right-arm PD cannot be mixed with implicit right-arm PD/Force NF groups")
    if explicit:
        gains = explicit["gains_by_joint"]
        limits = explicit["effort_limits_nm_by_joint"]
        stiffness = {name: float(values["kp"]) for name, values in gains.items()}
        damping = {name: float(values["kd"]) for name, values in gains.items()}
        effort_limits = {name: float(limits[name]) for name in gains}
        actuators = {
            "native_left_and_wuji2": ImplicitActuatorCfg(
                joint_names_expr=["arm_l_.*", "r_.*"],
                stiffness=None, damping=None,
                effort_limit_sim=None, velocity_limit_sim=None,
            ),
            "right_arm_explicit_ideal_pd": IdealPDActuatorCfg(
                joint_names_expr=list(gains.keys()),
                stiffness=stiffness,
                damping=damping,
                effort_limit=effort_limits,
                # Keep the PhysX solver safety limit equal to the real motor limit.
                # The IdealPDActuator already clips the command before writing it;
                # this second layer prevents accidental oversized external effort.
                effort_limit_sim=effort_limits,
                velocity_limit_sim=None,
            ),
        }
    elif not pd_groups:
        # None preserves every drive and limit value already authored in the USD.
        actuators = {
            "native_usd": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=None, damping=None,
                effort_limit_sim=None, velocity_limit_sim=None,
            )
        }
    else:
        # Only the seven right-arm drive gains are replaced for this A/B run.
        # Left arm and all 20 Wuji2 joints keep their official USD drive values.
        actuators = {
            "native_left_and_wuji2": ImplicitActuatorCfg(
                joint_names_expr=["arm_l_.*", "r_.*"],
                stiffness=None, damping=None,
                effort_limit_sim=None, velocity_limit_sim=None,
            )
        }
        for group in pd_groups:
            actuators[group["name"]] = ImplicitActuatorCfg(
                joint_names_expr=group["joint_names_expr"],
                stiffness=float(group["stiffness"]),
                damping=float(group["damping"]),
                effort_limit_sim=None,
                velocity_limit_sim=None,
            )
    return Articulation(
        ArticulationCfg(
            prim_path=config["robot_prim"], spawn=None,
            actuators=actuators,
        )
    )


def apply_force_natural_frequency_gains(robot: Articulation, arm_ids: list[int], config: dict) -> list[dict]:
    """Apply Force Drive K/D using the current generalized mass diagonal and natural-frequency formula."""
    groups = config.get("right_arm_force_natural_frequency_groups", [])
    if not groups:
        return []
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[0].to(robot.device)
    stiffness = robot.data.joint_stiffness.clone()
    damping = robot.data.joint_damping.clone()
    resolved: list[dict] = []
    all_ids = set()
    for group in groups:
        joint_ids, joint_names = robot.find_joints(group["joint_names_expr"], preserve_order=True)
        joint_ids = [int(joint_id) for joint_id in joint_ids]
        for joint_id in joint_ids:
            if joint_id not in arm_ids:
                raise RuntimeError(
                    f"Force gain group {group['name']} matched non-right-arm joint: {robot.joint_names[joint_id]}"
                )
        f = float(group["natural_frequency_rad_s"])
        zeta = float(group.get("damping_ratio", 1.0))
        if f <= 0.0 or zeta <= 0.0:
            raise ValueError(f"Invalid Force Drive natural-frequency group: {group}")
        m_eq = torch.clamp(torch.diag(mass_matrix)[joint_ids], min=1.0e-6)
        k_values = m_eq * f * f
        d_values = 2.0 * zeta * f * m_eq
        stiffness[0, joint_ids] = k_values
        damping[0, joint_ids] = d_values
        all_ids.update(joint_ids)
        resolved.append(
            {
                **group,
                "matched_joint_names": joint_names,
                "equivalent_mass_diagonal": [float(value) for value in m_eq.detach().cpu()],
                "stiffness_by_joint": {name: float(value) for name, value in zip(joint_names, k_values.detach().cpu())},
                "damping_by_joint": {name: float(value) for name, value in zip(joint_names, d_values.detach().cpu())},
                "gain_source": "force_drive_mass_matrix_natural_frequency",
                "gain_formula": "K=Mii*f^2, D=2*zeta*Mii*f using current generalized mass matrix diagonal",
            }
        )
    if sorted(all_ids) != sorted(arm_ids):
        missing = [robot.joint_names[joint_id] for joint_id in arm_ids if joint_id not in all_ids]
        raise RuntimeError(f"Force natural-frequency groups did not cover all right-arm joints: {missing}")
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.reset()
    print(f"[FORCE DRIVE GAINS] Applied mass-matrix natural-frequency gains: {resolved}")
    return resolved


def explicit_pd_config_by_joint(config: dict, robot: Articulation, arm_ids: list[int]) -> dict[int, dict]:
    explicit = config.get("right_arm_explicit_pd")
    if not explicit:
        return {}
    gains = explicit["gains_by_joint"]
    limits = explicit["effort_limits_nm_by_joint"]
    missing = sorted(set(robot.joint_names[joint_id] for joint_id in arm_ids).difference(gains))
    extra = sorted(set(gains).difference(robot.joint_names[joint_id] for joint_id in arm_ids))
    if missing or extra:
        raise RuntimeError(f"Explicit right-arm gains mismatch. missing={missing}, extra={extra}")
    resolved = {}
    for joint_id in arm_ids:
        name = robot.joint_names[joint_id]
        resolved[joint_id] = {
            "kp": float(gains[name]["kp"]),
            "kd": float(gains[name]["kd"]),
            "limit": float(limits[name]),
        }
    return resolved


def apply_explicit_right_arm_command(
    robot: Articulation,
    arm_ids: list[int],
    target: torch.Tensor,
    explicit_by_joint: dict[int, dict],
    use_gravity_compensation: bool,
) -> tuple[dict[int, dict], torch.Tensor]:
    """Set explicit feed-forward command for J1-J7 and return PD audit values.

    For IdealPDActuator, Isaac Lab itself computes:

        Kp * (q_target - q) + Kd * (qd_target - qd) + joint_effort_target

    and clips the result by actuator effort_limit.  Therefore the effort target
    written here must be the feed-forward term only, not the already-combined
    torque.  Only the right-arm effort targets are populated.  Wuji2 and the
    left arm keep their normal commands untouched by this helper.
    """
    if not explicit_by_joint:
        return {}, torch.zeros((1, robot.num_joints), device=robot.device, dtype=target.dtype)

    q = robot.data.joint_pos[0]
    qd = robot.data.joint_vel[0]
    target_q = target[0]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[0].to(robot.device)
    effort_target = torch.zeros((1, robot.num_joints), device=robot.device, dtype=target.dtype)
    audit: dict[int, dict] = {}

    for joint_id in arm_ids:
        cfg = explicit_by_joint[joint_id]
        kp = float(cfg["kp"])
        kd = float(cfg["kd"])
        limit = float(cfg["limit"])
        error = target_q[joint_id] - q[joint_id]
        velocity_error = -qd[joint_id]
        p_term = kp * error
        d_term = kd * velocity_error
        g_term = gravity[joint_id] if use_gravity_compensation else torch.zeros_like(error)
        raw = p_term + d_term + g_term
        applied = torch.clamp(raw, min=-limit, max=limit)
        effort_target[0, joint_id] = g_term
        audit[joint_id] = {
            "p": float(p_term.detach().cpu()),
            "d": float(d_term.detach().cpu()),
            "gravity": float(g_term.detach().cpu()),
            "raw": float(raw.detach().cpu()),
            "applied": float(applied.detach().cpu()),
            "limit": limit,
            "saturation_ratio": float((torch.abs(applied) / max(limit, 1.0e-9)).detach().cpu()),
        }

    robot.set_joint_effort_target(effort_target[:, arm_ids], joint_ids=arm_ids)
    return audit, effort_target


def find_exact_joint_ids(robot: Articulation, names: list[str]) -> list[int]:
    ids, found = robot.find_joints(names, preserve_order=True)
    if found != names:
        raise RuntimeError(f"Joint order mismatch. requested={names}, found={found}")
    return ids


def audit_structure(robot: Articulation, config: dict) -> tuple[list[int], list[int], int, int]:
    expected = int(config["expected_total_actuated_joints"])
    if robot.num_joints != expected:
        raise RuntimeError(f"Expected {expected} joints, found {robot.num_joints}")
    if not robot.is_fixed_base:
        raise RuntimeError("The assembled dual arm must be fixed-base")

    arm_ids = find_exact_joint_ids(robot, config["right_arm_joints"])
    prefixes = tuple(config["right_hand_joint_prefixes"])
    hand_names = [name for name in robot.joint_names if name.startswith(prefixes)]
    if len(hand_names) != int(config["expected_right_hand_joints"]):
        raise RuntimeError(f"Expected 20 Wuji2 joints, found {len(hand_names)}: {hand_names}")
    hand_ids = find_exact_joint_ids(robot, hand_names)
    flange_ids, flange_names = robot.find_bodies([config["flange_body"]], preserve_order=True)
    wrist_ids, wrist_names = robot.find_bodies([config["wrist_body"]], preserve_order=True)
    if flange_names != [config["flange_body"]] or wrist_names != [config["wrist_body"]]:
        raise RuntimeError(f"Body lookup failed: flange={flange_names}, wrist={wrist_names}")
    if set(arm_ids).intersection(hand_ids):
        raise RuntimeError("Right-arm and Wuji2-hand joint groups overlap")

    print(f"[AUDIT PASS] fixed-base articulation, joints={robot.num_joints}, bodies={robot.num_bodies}")
    print(f"[AUDIT PASS] right arm joints: {config['right_arm_joints']}")
    print(f"[AUDIT PASS] Wuji2 hand joints: {hand_names}")
    return arm_ids, hand_ids, flange_ids[0], wrist_ids[0]


def build_saved_state(robot: Articulation, manifest: dict) -> torch.Tensor:
    saved = manifest["revolute_joint_positions_deg"]
    missing = sorted(set(robot.joint_names).difference(saved))
    extra = sorted(set(saved).difference(robot.joint_names))
    if missing or extra:
        raise RuntimeError(f"Saved joint map mismatch. missing={missing}, extra={extra}")
    values = [float(saved[name]) for name in robot.joint_names]
    return torch.deg2rad(
        torch.tensor(values, device=robot.device, dtype=robot.data.joint_pos.dtype)
    ).reshape(1, robot.num_joints)


def physx_tensor(robot: Articulation, method_name: str) -> torch.Tensor:
    value = getattr(robot.root_physx_view, method_name)()
    if isinstance(value, torch.Tensor):
        return value.to(robot.device)
    if hasattr(value, "numpy"):
        return torch.as_tensor(value.numpy(), device=robot.device)
    return torch.as_tensor(value, device=robot.device)


def body_pose_cpu(robot: Articulation, body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    pose = robot.data.body_pose_w[0, body_id]
    return pose[:3].detach().cpu().clone(), pose[3:7].detach().cpu().clone()


def quaternion_distance_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


def group_label(joint_id: int, arm_ids: list[int], hand_ids: list[int]) -> str:
    if joint_id in arm_ids:
        return "right_arm"
    if joint_id in hand_ids:
        return "wuji2_hand"
    return "held_other"


def summarize_group(
    ids: list[int], robot: Articulation, samples: dict, settled: torch.Tensor,
    error_limit_deg: float, config: dict,
) -> dict:
    idx = torch.tensor(ids, dtype=torch.long)
    q = torch.stack(samples["q"])[:, idx]
    qd = torch.stack(samples["qd"])[:, idx]
    target = torch.stack(samples["target"])[:, idx]
    actuation = torch.stack(samples["actuation"])[:, idx]
    projected = torch.stack(samples["projected"])[:, idx]
    explicit_applied = torch.stack(samples["explicit_applied"])[:, idx]
    explicit_raw = torch.stack(samples["explicit_raw"])[:, idx]
    explicit_gravity = torch.stack(samples["explicit_gravity"])[:, idx]
    effort_limits = samples["effort_limits"][idx].reshape(1, -1)

    settled_error_deg = torch.rad2deg(torch.abs(q[settled] - target[settled]))
    settled_speed = torch.abs(qd[settled])
    max_error = float(torch.max(settled_error_deg))
    max_speed = float(torch.max(settled_speed))
    passed = (
        math.isfinite(max_error)
        and math.isfinite(max_speed)
        and max_error <= error_limit_deg
        and max_speed <= float(config["limits"]["settled_max_joint_speed_rad_s"])
    )

    per_joint = []
    for local, joint_id in enumerate(ids):
        per_joint.append(
            {
                "name": robot.joint_names[joint_id],
                "final_error_deg": float(torch.rad2deg(torch.abs(q[-1, local] - target[-1, local]))),
                "settled_max_speed_rad_s": float(torch.max(torch.abs(qd[settled, local]))),
                "peak_physx_actuation_force_nm": float(torch.max(torch.abs(actuation[:, local]))),
                "peak_physx_projected_joint_force_nm": float(torch.max(torch.abs(projected[:, local]))),
                "peak_explicit_raw_torque_nm": float(torch.max(torch.abs(explicit_raw[:, local]))),
                "peak_explicit_applied_torque_nm": float(torch.max(torch.abs(explicit_applied[:, local]))),
                "peak_gravity_compensation_nm": float(torch.max(torch.abs(explicit_gravity[:, local]))),
                "peak_explicit_saturation_ratio": float(
                    torch.max(torch.abs(explicit_applied[:, local]) / effort_limits[0, local].clamp(min=1.0e-9))
                ),
                "effort_limit_nm": float(effort_limits[0, local]),
            }
        )
    return {
        "status": "PASS" if passed else "FAIL",
        "settled_max_target_error_deg": max_error,
        "settled_max_joint_speed_rad_s": max_speed,
        "all_time_max_joint_speed_rad_s": float(torch.max(torch.abs(qd))),
        "all_time_peak_physx_actuation_force_nm": float(torch.max(torch.abs(actuation))),
        "all_time_peak_physx_projected_joint_force_nm": float(torch.max(torch.abs(projected))),
        "all_time_peak_explicit_raw_torque_nm": float(torch.max(torch.abs(explicit_raw))),
        "all_time_peak_explicit_applied_torque_nm": float(torch.max(torch.abs(explicit_applied))),
        "all_time_peak_gravity_compensation_nm": float(torch.max(torch.abs(explicit_gravity))),
        "per_joint": per_joint,
    }


def summarize_body(name: str, positions: torch.Tensor, quaternions: torch.Tensor,
                   settled: torch.Tensor, config: dict) -> dict:
    settled_pos = positions[settled]
    settled_quat = quaternions[settled]
    translation_span = float(
        torch.max(torch.linalg.vector_norm(settled_pos - settled_pos[0:1], dim=1))
    )
    rotation_span = float(
        torch.max(quaternion_distance_deg(settled_quat, settled_quat[0:1].expand_as(settled_quat)))
    )
    passed = (
        translation_span <= float(config["limits"]["settled_body_translation_span_m"])
        and rotation_span <= float(config["limits"]["settled_body_rotation_span_deg"])
    )
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "settled_translation_span_m": translation_span,
        "settled_rotation_span_deg": rotation_span,
        "all_time_translation_from_first_sample_m": float(
            torch.max(torch.linalg.vector_norm(positions - positions[0:1], dim=1))
        ),
        "all_time_rotation_from_first_sample_deg": float(
            torch.max(quaternion_distance_deg(quaternions, quaternions[0:1].expand_as(quaternions)))
        ),
    }


def write_outputs(config: dict, stage_path: Path, robot: Articulation, arm_ids: list[int],
                  hand_ids: list[int], samples: dict, trace_rows: list[dict]) -> Path:
    output_dir = project_path(config["output_directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "joint_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(trace_rows)

    times = torch.tensor(samples["time_s"])
    settled = times >= float(config["duration_s"] - config["settled_window_s"])
    arm = summarize_group(
        arm_ids, robot, samples, settled,
        float(config["limits"]["right_arm_settled_max_target_error_deg"]), config,
    )
    hand = summarize_group(
        hand_ids, robot, samples, settled,
        float(config["limits"]["right_hand_settled_max_target_error_deg"]), config,
    )
    flange = summarize_body(
        config["flange_body"], torch.stack(samples["flange_pos"]),
        torch.stack(samples["flange_quat"]), settled, config,
    )
    wrist = summarize_body(
        config["wrist_body"], torch.stack(samples["wrist_pos"]),
        torch.stack(samples["wrist_quat"]), settled, config,
    )
    passed = all(item["status"] == "PASS" for item in (arm, hand, flange, wrist))
    resolved_pd_groups = resolve_right_arm_pd_groups(config)
    explicit = config.get("right_arm_explicit_pd")
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "meaning": (
            "Initial static pose is stable"
            if passed else "Initial static pose is not stable; IK remains disabled"
        ),
        "stage": str(stage_path),
        "robot_prim": config["robot_prim"],
        "duration_s": float(config["duration_s"]),
        "settled_window_s": float(config["settled_window_s"]),
        "physics_dt_s": float(config["physics_dt_s"]),
        "parameter_policy": (
            "Only right-arm J1-J7 are explicit IdealPDActuator torque control; left arm and Wuji2 remain native implicit USD"
            if explicit else (
                "Only right-arm K/D gains overridden; Wuji2 and all physical limits unchanged"
                if resolved_pd_groups
                else "Native assembled USD values preserved; no drive/dynamics/contact override"
            )
        ),
        "right_arm_explicit_pd": explicit or {},
        "right_arm_explicit_runtime_audit": samples.get("explicit_runtime_audit", {}),
        "right_arm_pd_groups": resolved_pd_groups,
        "right_arm_natural_frequency_groups": config.get("right_arm_natural_frequency_groups", []),
        "right_arm_drive_type_override": config.get("right_arm_drive_type", "native"),
        "right_arm_force_natural_frequency_groups": config.get("right_arm_force_natural_frequency_groups", []),
        "resolved_force_drive_gains": samples.get("resolved_force_drive_gains", []),
        "initial_gravity_compensation_by_joint_nm": samples["initial_gravity_compensation"],
        "physx_drive_type_by_joint": samples["drive_types"],
        "physx_measurements": {
            "actuation": "ArticulationView.get_dof_actuation_forces; this is the external actuation-force buffer and is normally zero for implicit USD drives",
            "projected_joint": "ArticulationView.get_dof_projected_joint_forces",
            "explicit_applied_torque": "For right-arm explicit IdealPDActuator runs only: this is the clipped torque command written to PhysX through set_dof_actuation_forces",
        },
        "limits": config["limits"],
        "right_arm": arm,
        "wuji2_hand": hand,
        "right_flange": flange,
        "wuji2_wrist": wrist,
        "joint_trace_csv": str(trace_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def run() -> Path:
    config = load_json(ARGS.config)
    stage_path = project_path(config["stage"]).resolve()
    manifest_path = project_path(config["layout_manifest"]).resolve()
    if not stage_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"stage={stage_path}, manifest={manifest_path}")

    if not stage_utils.open_stage(str(stage_path)):
        raise RuntimeError(f"Isaac Sim could not open stage: {stage_path}")
    set_force_drive_type_on_joint_prims(config)
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=float(config["physics_dt_s"]),
            render_interval=int(config["render_interval"]),
            device=ARGS.device,
            gravity=tuple(config.get("gravity_m_s2", [0.0, 0.0, -9.81])),
        )
    )
    robot = create_robot(config)
    sim.reset()
    dt = float(config["physics_dt_s"])
    robot.update(dt)
    arm_ids, hand_ids, flange_id, wrist_id = audit_structure(robot, config)

    target = build_saved_state(robot, load_json(manifest_path))
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.reset()
    robot.update(dt)
    resolved_force_drive_gains = apply_force_natural_frequency_gains(robot, arm_ids, config)
    explicit_by_joint = explicit_pd_config_by_joint(config, robot, arm_ids)
    explicit_use_gravity = bool(config.get("right_arm_explicit_pd", {}).get("use_gravity_compensation", False))
    robot.set_joint_position_target(target)

    resolved_pd_groups = resolve_right_arm_pd_groups(config)
    if explicit_by_joint:
        print("[PARAMETER POLICY] Right-arm J1-J7 explicit IdealPDActuator torque control")
        print(f"[PARAMETER POLICY] gravity feed-forward enabled={explicit_use_gravity}")
        print("[PARAMETER POLICY] Left arm and Wuji2 drives remain native USD implicit values")
    elif resolved_pd_groups:
        print(f"[PARAMETER POLICY] Right-arm grouped PD only: {resolved_pd_groups}")
        print("[PARAMETER POLICY] Wuji2 drives and all physical limits remain native USD values")
    else:
        print("[PARAMETER POLICY] Native USD values unchanged")
    print(f"[TEST] gravity={config.get('gravity_m_s2', [0.0, 0.0, -9.81])} m/s^2")
    print("[TEST] Exact 35-DOF hold only; no IK, root command, or grasp")
    for joint_id in arm_ids:
        print(
            f"  {robot.joint_names[joint_id]}: "
            f"K={float(robot.data.joint_stiffness[0, joint_id]):.3f}, "
            f"D={float(robot.data.joint_damping[0, joint_id]):.3f}, "
            f"effort_limit={float(robot.data.joint_effort_limits[0, joint_id]):.3f}"
        )

    samples = {
        "time_s": [], "q": [], "qd": [], "target": [], "actuation": [], "projected": [],
        "explicit_p": [], "explicit_d": [], "explicit_gravity": [],
        "explicit_raw": [], "explicit_applied": [],
        "flange_pos": [], "flange_quat": [], "wrist_pos": [], "wrist_quat": [],
        "effort_limits": robot.data.joint_effort_limits[0].detach().cpu().clone(),
        "initial_gravity_compensation": {},
        "drive_types": {},
        "resolved_force_drive_gains": resolved_force_drive_gains,
        "explicit_runtime_audit": {
            "enabled": bool(explicit_by_joint),
            "gravity_feedforward": explicit_use_gravity,
            "right_arm_stiffness_in_physx_after_setup": {},
            "right_arm_damping_in_physx_after_setup": {},
            "right_arm_effort_limit_in_physx_after_setup": {},
            "right_arm_ideal_pd_limit": {},
            "effort_limit_policy": (
                "IdealPDActuator clips computed torque by effort_limit; effort_limit_sim is also set to the same real limit as a PhysX safety clamp."
                if explicit_by_joint else ""
            ),
        },
    }
    trace_rows: list[dict] = []
    steps = max(1, round(float(config["duration_s"]) / dt))
    stride = max(1, round(1.0 / (float(config["telemetry_hz"]) * dt)))

    for step in range(steps):
        robot.set_joint_position_target(target)
        explicit_audit, explicit_effort_target = apply_explicit_right_arm_command(
            robot, arm_ids, target, explicit_by_joint, explicit_use_gravity
        )
        robot.write_data_to_sim()
        if step == 0:
            gravity_force = physx_tensor(robot, "get_gravity_compensation_forces")[0].detach().cpu()
            drive_types = physx_tensor(robot, "get_drive_types")[0].detach().cpu()
            if not explicit_by_joint:
                physx_target = physx_tensor(robot, "get_dof_position_targets")[0].detach().cpu()
                command_error = torch.max(torch.abs(physx_target - target[0].detach().cpu()))
                print(f"[TARGET AUDIT] max PhysX target mapping error={float(command_error):.9f} rad")
            else:
                print("[TARGET AUDIT] explicit right arm: q_target is held in Isaac Lab controller, not written as a PhysX drive target")
            for joint_id in arm_ids:
                samples["initial_gravity_compensation"][robot.joint_names[joint_id]] = float(
                    gravity_force[joint_id]
                )
                samples["drive_types"][robot.joint_names[joint_id]] = int(drive_types[joint_id])
                samples["explicit_runtime_audit"]["right_arm_stiffness_in_physx_after_setup"][
                    robot.joint_names[joint_id]
                ] = float(robot.data.joint_stiffness[0, joint_id])
                samples["explicit_runtime_audit"]["right_arm_damping_in_physx_after_setup"][
                    robot.joint_names[joint_id]
                ] = float(robot.data.joint_damping[0, joint_id])
                samples["explicit_runtime_audit"]["right_arm_effort_limit_in_physx_after_setup"][
                    robot.joint_names[joint_id]
                ] = float(samples["effort_limits"][joint_id])
                if joint_id in explicit_by_joint:
                    samples["explicit_runtime_audit"]["right_arm_ideal_pd_limit"][
                        robot.joint_names[joint_id]
                    ] = float(explicit_by_joint[joint_id]["limit"])
                print(
                    f"  {robot.joint_names[joint_id]}: requested={float(target[0, joint_id]):+.6f}, "
                    f"gravity_comp={float(gravity_force[joint_id]):+.3f} N*m, "
                    f"limit={float(samples['effort_limits'][joint_id]):.3f} N*m, "
                    f"drive_type={int(drive_types[joint_id])}, "
                    f"K_runtime={float(robot.data.joint_stiffness[0, joint_id]):.3f}, "
                    f"D_runtime={float(robot.data.joint_damping[0, joint_id]):.3f}"
                )
        sim.step()
        robot.update(dt)

        q = robot.data.joint_pos[0].detach().cpu().clone()
        qd = robot.data.joint_vel[0].detach().cpu().clone()
        target_cpu = target[0].detach().cpu().clone()
        actuation = physx_tensor(robot, "get_dof_actuation_forces")[0].detach().cpu().clone()
        projected = physx_tensor(robot, "get_dof_projected_joint_forces")[0].detach().cpu().clone()
        p_values = torch.zeros(robot.num_joints)
        d_values = torch.zeros(robot.num_joints)
        gravity_values = torch.zeros(robot.num_joints)
        raw_values = torch.zeros(robot.num_joints)
        applied_values = torch.zeros(robot.num_joints)
        for joint_id, audit in explicit_audit.items():
            p_values[joint_id] = audit["p"]
            d_values[joint_id] = audit["d"]
            gravity_values[joint_id] = audit["gravity"]
            raw_values[joint_id] = audit["raw"]
            applied_values[joint_id] = audit["applied"]
        flange_pos, flange_quat = body_pose_cpu(robot, flange_id)
        wrist_pos, wrist_quat = body_pose_cpu(robot, wrist_id)
        time_s = (step + 1) * dt

        samples["time_s"].append(time_s)
        samples["q"].append(q)
        samples["qd"].append(qd)
        samples["target"].append(target_cpu)
        samples["actuation"].append(actuation)
        samples["projected"].append(projected)
        samples["explicit_p"].append(p_values)
        samples["explicit_d"].append(d_values)
        samples["explicit_gravity"].append(gravity_values)
        samples["explicit_raw"].append(raw_values)
        samples["explicit_applied"].append(applied_values)
        samples["flange_pos"].append(flange_pos)
        samples["flange_quat"].append(flange_quat)
        samples["wrist_pos"].append(wrist_pos)
        samples["wrist_quat"].append(wrist_quat)

        for joint_id, joint_name in enumerate(robot.joint_names):
            trace_rows.append(
                {
                    "time_s": time_s,
                    "joint_name": joint_name,
                    "group": group_label(joint_id, arm_ids, hand_ids),
                    "position_rad": float(q[joint_id]),
                    "target_rad": float(target_cpu[joint_id]),
                    "error_rad": float(q[joint_id] - target_cpu[joint_id]),
                    "velocity_rad_s": float(qd[joint_id]),
                    "physx_actuation_force_nm": float(actuation[joint_id]),
                    "physx_projected_joint_force_nm": float(projected[joint_id]),
                    "effort_limit_nm": float(samples["effort_limits"][joint_id]),
                    "explicit_pd_p_nm": float(p_values[joint_id]),
                    "explicit_pd_d_nm": float(d_values[joint_id]),
                    "gravity_compensation_nm": float(gravity_values[joint_id]),
                    "explicit_raw_torque_nm": float(raw_values[joint_id]),
                    "explicit_applied_torque_nm": float(applied_values[joint_id]),
                    "explicit_saturation_ratio": (
                        float(abs(applied_values[joint_id]) / max(float(samples["effort_limits"][joint_id]), 1.0e-9))
                    ),
                }
            )

        if step % stride == 0 or step == steps - 1:
            arm_speed = float(torch.max(torch.abs(qd[arm_ids])))
            hand_speed = float(torch.max(torch.abs(qd[hand_ids])))
            arm_error = float(torch.max(torch.rad2deg(torch.abs(q[arm_ids] - target_cpu[arm_ids]))))
            hand_error = float(torch.max(torch.rad2deg(torch.abs(q[hand_ids] - target_cpu[hand_ids]))))
            print(
                f"\r[STATIC] t={time_s:5.2f}s | arm error={arm_error:7.3f} deg, "
                f"speed={arm_speed:7.4f} rad/s | hand error={hand_error:7.3f} deg, "
                f"speed={hand_speed:7.4f} rad/s",
                end="", flush=True,
            )
    print()
    return write_outputs(config, stage_path, robot, arm_ids, hand_ids, samples, trace_rows)


def main() -> int:
    try:
        report_path = run()
        report = load_json(report_path)
        print(f"\n[INITIAL STABILITY {report['status']}] {report_path}")
        for label, key in (
            ("right arm", "right_arm"), ("Wuji2 hand", "wuji2_hand"),
            ("right flange", "right_flange"), ("Wuji2 wrist", "wuji2_wrist"),
        ):
            print(f"  {label}: {report[key]['status']}")
        return 0 if report["status"] == "PASS" else 2
    except Exception as error:
        print(f"\n[INITIAL STABILITY ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        SIMULATION_APP.close()


if __name__ == "__main__":
    raise SystemExit(main())
