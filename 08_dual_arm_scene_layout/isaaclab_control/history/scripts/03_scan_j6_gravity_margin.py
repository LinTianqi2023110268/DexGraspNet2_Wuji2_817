"""Scan only right-arm J6 for a gravity-torque-feasible startup posture.

No simulation step is advanced.  The other 34 joint positions are fixed to the
accepted layout values.  The result is diagnostic and does not edit the stage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/history/config/initial_stability_grouped_pd_round2.json"

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
APP = AppLauncher(ARGS)
SIMULATION_APP = APP.app


import torch  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(path(cfg["layout_manifest"]).read_text(encoding="utf-8"))
    if not stage_utils.open_stage(str(path(cfg["stage"]))):
        raise RuntimeError("Could not open calibrated stage")
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=float(cfg["physics_dt_s"]), device=ARGS.device)
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path=cfg["robot_prim"], spawn=None,
            actuators={
                "native": ImplicitActuatorCfg(
                    joint_names_expr=[".*"], stiffness=None, damping=None,
                    effort_limit_sim=None, velocity_limit_sim=None,
                )
            },
        )
    )
    sim.reset()
    robot.update(float(cfg["physics_dt_s"]))
    if robot.num_joints != 35:
        raise RuntimeError(f"Expected 35 joints, found {robot.num_joints}")

    q = torch.deg2rad(
        torch.tensor(
            [float(manifest["revolute_joint_positions_deg"][name]) for name in robot.joint_names],
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )
    ).reshape(1, -1)
    arm_ids, arm_names = robot.find_joints(cfg["right_arm_joints"], preserve_order=True)
    if arm_names != cfg["right_arm_joints"]:
        raise RuntimeError(arm_names)
    j6_id = arm_ids[5]
    effort = robot.data.joint_effort_limits[0, arm_ids].detach().cpu()
    limits = robot.data.joint_pos_limits[0, j6_id].detach().cpu()

    lower_deg = float(torch.rad2deg(limits[0]))
    upper_deg = float(torch.rad2deg(limits[1]))
    values_deg = torch.arange(
        int(round(lower_deg * 2.0)), int(round(upper_deg * 2.0)) + 1,
        device=robot.device, dtype=torch.float32,
    ) / 2.0
    rows = []
    zero_velocity = torch.zeros_like(q)
    for q6_deg in values_deg:
        candidate = q.clone()
        candidate[0, j6_id] = torch.deg2rad(q6_deg)
        robot.write_joint_state_to_sim(candidate, zero_velocity)
        gravity = robot.root_physx_view.get_gravity_compensation_forces()[0].detach().cpu()
        arm_gravity = gravity[arm_ids]
        ratio = torch.abs(arm_gravity) / effort
        rows.append(
            {
                "j6_deg": float(q6_deg),
                "arm_gravity_compensation_nm": {
                    name: float(arm_gravity[i]) for i, name in enumerate(arm_names)
                },
                "arm_effort_ratio": {name: float(ratio[i]) for i, name in enumerate(arm_names)},
                "max_arm_effort_ratio": float(torch.max(ratio)),
                "j6_effort_ratio": float(ratio[5]),
            }
        )

    feasible = [row for row in rows if row["max_arm_effort_ratio"] <= 0.80]
    feasible.sort(key=lambda row: (abs(row["j6_deg"]), row["max_arm_effort_ratio"]))
    best = feasible[0] if feasible else min(rows, key=lambda row: row["max_arm_effort_ratio"])
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FEASIBLE" if feasible else "NO_80_PERCENT_MARGIN",
        "fixed_joint_deg": manifest["revolute_joint_positions_deg"],
        "only_scanned_joint": "arm_r_joint_6",
        "scan_range_deg": [lower_deg, upper_deg],
        "step_deg": 0.5,
        "selected": best,
        "all_samples": rows,
    }
    output_path = PROJECT_ROOT / (
        "08_dual_arm_scene_layout/isaaclab_control/outputs/j6_gravity_margin_scan.json"
    )
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "selected": best}, indent=2))
    print(f"wrote {output_path}")
    return 0 if feasible else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        SIMULATION_APP.close()
