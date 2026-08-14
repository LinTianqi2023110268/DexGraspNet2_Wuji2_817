"""Runtime mass/COM/gravity audit for the mass-fixed Arm + Wuji2 A/B asset.

This program performs exactly one physics step.  It does not run IK, a
trajectory, a camera task, or a grasp.  J6 downstream bodies are discovered
from active USD joint body0/body1 relationships rather than a hand-written list.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/isaaclab_control/diagnostics/config/initial_stability_grouped_pd_round1_mass_fixed.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/isaaclab_control/outputs/mass_tree_audit_round1_mass_fixed/report.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
from isaaclab.sim import SimulationContext  # noqa: E402
from pxr import UsdPhysics  # noqa: E402


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def physx_tensor(robot: Articulation, method_name: str) -> torch.Tensor:
    value = getattr(robot.root_physx_view, method_name)()
    if isinstance(value, torch.Tensor):
        return value.to(robot.device)
    if hasattr(value, "numpy"):
        value = value.numpy()
    return torch.as_tensor(value, device=robot.device)


def create_robot(config: dict) -> Articulation:
    actuators = {
        "native_left_and_wuji2": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_.*", "r_.*"],
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        )
    }
    for group in config["right_arm_pd_groups"]:
        actuators[group["name"]] = ImplicitActuatorCfg(
            joint_names_expr=group["joint_names_expr"],
            stiffness=float(group["stiffness"]),
            damping=float(group["damping"]),
            effort_limit_sim=None,
            velocity_limit_sim=None,
        )
    return Articulation(
        ArticulationCfg(prim_path=config["robot_prim"], spawn=None, actuators=actuators)
    )


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


def active_joint_edges(robot_prim: str) -> tuple[dict[str, set[str]], dict[str, tuple[str, str]]]:
    """Return active body-parent graph and joint-name edge lookup from USD."""
    stage = stage_utils.get_current_stage()
    root = stage.GetPrimAtPath(robot_prim)
    if not root.IsValid():
        raise RuntimeError(f"Robot prim is missing: {robot_prim}")
    children: dict[str, set[str]] = {}
    joints: dict[str, tuple[str, str]] = {}
    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.GetPath().HasPrefix(root.GetPath()):
            continue
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            continue
        parent = body0[0].name
        child = body1[0].name
        children.setdefault(parent, set()).add(child)
        joints[prim.GetName()] = (parent, child)
    return children, joints


def descendants(children: dict[str, set[str]], start: str) -> list[str]:
    found: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in found:
            continue
        found.add(current)
        stack.extend(sorted(children.get(current, set())))
    return sorted(found)


def vector(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.tolist()]


def run() -> Path:
    config = load_json(ARGS.config)
    stage_path = project_path(config["stage"]).resolve()
    manifest = load_json(project_path(config["layout_manifest"]))
    if not stage_utils.open_stage(str(stage_path)):
        raise RuntimeError(f"Cannot open stage: {stage_path}")

    dt = float(config["physics_dt_s"])
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=dt,
            render_interval=int(config["render_interval"]),
            device=ARGS.device,
            gravity=tuple(config.get("gravity_m_s2", [0.0, 0.0, -9.81])),
        )
    )
    robot = create_robot(config)
    sim.reset()
    robot.update(dt)

    if robot.num_joints != int(config["expected_total_actuated_joints"]):
        raise RuntimeError(f"Expected 35 joints, found {robot.num_joints}")
    if not robot.is_fixed_base:
        raise RuntimeError("Expected fixed-base articulation")

    arm_ids, arm_names = robot.find_joints(config["right_arm_joints"], preserve_order=True)
    if arm_names != config["right_arm_joints"]:
        raise RuntimeError(f"Right-arm order mismatch: {arm_names}")

    target = build_saved_state(robot, manifest)
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.reset()
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()

    gravity_before = physx_tensor(robot, "get_gravity_compensation_forces")[0].cpu()
    drive_types = physx_tensor(robot, "get_drive_types")[0].cpu()

    sim.step()
    robot.update(dt)
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()

    gravity_after = physx_tensor(robot, "get_gravity_compensation_forces")[0].cpu()
    masses = physx_tensor(robot, "get_masses")[0].cpu()
    local_coms = physx_tensor(robot, "get_coms")[0, :, :3].cpu()
    world_coms = robot.data.body_com_pos_w[0].detach().cpu()

    graph, joint_edges = active_joint_edges(config["robot_prim"])
    if "arm_r_joint_6" not in joint_edges:
        raise RuntimeError("USD graph is missing arm_r_joint_6")
    _, j6_child = joint_edges["arm_r_joint_6"]
    downstream_names = descendants(graph, j6_child)
    missing_bodies = sorted(set(downstream_names).difference(robot.body_names))
    if missing_bodies:
        raise RuntimeError(f"J6 descendants missing from PhysX body list: {missing_bodies}")

    body_index = {name: index for index, name in enumerate(robot.body_names)}
    downstream_ids = [body_index[name] for name in downstream_names]
    downstream_masses = masses[downstream_ids]
    total_mass = torch.sum(downstream_masses)
    combined_world_com = torch.sum(
        downstream_masses[:, None] * world_coms[downstream_ids], dim=0
    ) / total_mass

    body_rows = []
    for name, body_id in zip(downstream_names, downstream_ids):
        body_rows.append(
            {
                "name": name,
                "mass_kg": float(masses[body_id]),
                "com_local_m": vector(local_coms[body_id]),
                "com_world_m": vector(world_coms[body_id]),
            }
        )

    arm_runtime = []
    for name, joint_id in zip(arm_names, arm_ids):
        arm_runtime.append(
            {
                "name": name,
                "drive_type": int(drive_types[joint_id]),
                "stiffness": float(robot.data.joint_stiffness[0, joint_id]),
                "damping": float(robot.data.joint_damping[0, joint_id]),
                "effort_limit_nm": float(robot.data.joint_effort_limits[0, joint_id]),
                "gravity_comp_before_first_step_nm": float(gravity_before[joint_id]),
                "gravity_comp_after_first_step_nm": float(gravity_after[joint_id]),
            }
        )

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "One-variable runtime audit after removing six non-physical default-mass frames",
        "stage": str(stage_path),
        "robot_prim": config["robot_prim"],
        "fixed_base": bool(robot.is_fixed_base),
        "joint_count": robot.num_joints,
        "body_count": robot.num_bodies,
        "frozen_right_arm_pd_groups": config["right_arm_pd_groups"],
        "right_arm_runtime": arm_runtime,
        "j6_child_body": j6_child,
        "j6_downstream_body_count": len(downstream_names),
        "j6_downstream_total_mass_kg": float(total_mass),
        "j6_downstream_combined_com_world_m": vector(combined_world_com),
        "j6_downstream_bodies": body_rows,
    }
    output = ARGS.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[MASS AUDIT] bodies={robot.num_bodies}, J6 downstream={len(body_rows)}")
    print(f"[MASS AUDIT] J6 downstream total mass={float(total_mass):.9f} kg")
    print(f"[MASS AUDIT] combined COM world={vector(combined_world_com)} m")
    for row in arm_runtime:
        print(
            f"  {row['name']}: gravity={row['gravity_comp_after_first_step_nm']:+.6f} N*m, "
            f"drive={row['drive_type']}, K={row['stiffness']:.3f}, "
            f"D={row['damping']:.3f}, limit={row['effort_limit_nm']:.3f}"
        )
    print(f"[MASS AUDIT COMPLETE] {output}")
    return output


def main() -> int:
    try:
        run()
        return 0
    except Exception as error:
        print(f"[MASS AUDIT ERROR] {type(error).__name__}: {error}")
        return 1
    finally:
        SIMULATION_APP.close()


if __name__ == "__main__":
    raise SystemExit(main())
