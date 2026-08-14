#!/usr/bin/env python3
"""Read-only bounded IK audit for the selected PREGRASP flange target.

This script uses the assembled robot URDF only for forward kinematics.  It does
not start Isaac Sim, write a joint target, or certify collision-free motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASE = (
    PROJECT_ROOT
    / "06_leap_to_wuji2_final_pipeline/01_cases"
    / "live_scene0000_ashtray_isaaclab_candidate0274"
)
ROBOT_URDF = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2"
    / "urdf/dual_arm_right_wuji2.urdf"
)
LAYOUT_JSON = PROJECT_ROOT / "08_dual_arm_scene_layout/outputs/manual_layout_calibrated.json"
RIGHT_ARM_NAMES = [f"arm_r_joint_{index}" for index in range(1, 8)]
INITIAL_RIGHT_ARM_DEG = np.asarray([50.0, -70.0, 0.0, 40.0, 35.0, 0.0, 25.0])


def world_from_robot_base() -> np.ndarray:
    layout = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))
    mount = layout["transforms"]["dual_arm_mount"]
    # OpenUSD stores Gf matrices in row-vector convention.  Transpose once to
    # obtain the column-vector convention used by Pinocchio and this project.
    transform = np.asarray(mount["Gf_local_to_world_row_major"], dtype=np.float64).T
    return transform


def pose_error(current: pin.SE3, target: pin.SE3) -> tuple[float, float]:
    position_mm = 1000.0 * float(np.linalg.norm(current.translation - target.translation))
    orientation_deg = float(np.degrees(np.linalg.norm(pin.log3(current.rotation.T @ target.rotation))))
    return position_mm, orientation_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--starts", type=int, default=24)
    args = parser.parse_args()

    case_root = args.case_root.resolve()
    target_path = case_root / "07_arm_execution/arm_flange_targets.npz"
    if not target_path.is_file():
        raise FileNotFoundError(f"run 03_build_arm_execution_targets.py first: {target_path}")
    with np.load(target_path, allow_pickle=False) as archive:
        names = np.asarray(archive["waypoint_names"])
        targets_world = np.asarray(archive["world_from_right_flange"], dtype=np.float64)
    matches = np.flatnonzero(names == "pregrasp")
    if matches.size != 1:
        raise RuntimeError(f"expected one PREGRASP target, found {matches.size}")

    base_from_world = np.linalg.inv(world_from_robot_base())
    target_base_matrix = base_from_world @ targets_world[int(matches[0])]
    target = pin.SE3(target_base_matrix[:3, :3], target_base_matrix[:3, 3])

    model = pin.buildModelFromUrdf(str(ROBOT_URDF))
    data = model.createData()
    flange_frame = model.getFrameId("arm_r_link_tf")
    if flange_frame >= len(model.frames):
        raise RuntimeError("arm_r_link_tf frame is missing from assembled URDF")
    joint_ids = [model.getJointId(name) for name in RIGHT_ARM_NAMES]
    q_indices = np.asarray([model.joints[joint_id].idx_q for joint_id in joint_ids])
    q_template = pin.neutral(model)
    initial = np.deg2rad(INITIAL_RIGHT_ARM_DEG)
    lower = model.lowerPositionLimit[q_indices] + 0.01
    upper = model.upperPositionLimit[q_indices] - 0.01

    def frame_at(right_q: np.ndarray) -> pin.SE3:
        q = q_template.copy()
        q[q_indices] = right_q
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[flange_frame].copy()

    def residual(right_q: np.ndarray) -> np.ndarray:
        current = frame_at(right_q)
        position = (current.translation - target.translation) / 0.01
        orientation = pin.log3(current.rotation.T @ target.rotation) / 0.10
        return np.concatenate((position, orientation))

    initial_pose = frame_at(initial)
    initial_position_mm, initial_orientation_deg = pose_error(initial_pose, target)
    rng = np.random.default_rng(20260813)
    starts = [np.clip(initial, lower, upper)]
    starts.extend(rng.uniform(lower, upper) for _ in range(max(0, args.starts - 1)))

    solutions = []
    for start_index, start in enumerate(starts):
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            method="trf",
            max_nfev=800,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        solved_pose = frame_at(result.x)
        position_mm, orientation_deg = pose_error(solved_pose, target)
        solutions.append(
            {
                "start_index": start_index,
                "q_rad": result.x.copy(),
                "position_error_mm": position_mm,
                "orientation_error_deg": orientation_deg,
                "cost": float(result.cost),
                "nfev": int(result.nfev),
            }
        )
    best = min(solutions, key=lambda item: item["position_error_mm"] + item["orientation_error_deg"])
    margin = np.minimum(best["q_rad"] - lower, upper - best["q_rad"])
    reachable = best["position_error_mm"] <= 2.0 and best["orientation_error_deg"] <= 2.0

    output_root = case_root / "07_arm_execution"
    solution_path = output_root / "pregrasp_read_only_ik.npz"
    np.savez_compressed(
        solution_path,
        right_arm_joint_names=np.asarray(RIGHT_ARM_NAMES),
        initial_right_arm_q_rad=initial,
        solved_right_arm_q_rad=best["q_rad"],
        lower_limit_rad=lower,
        upper_limit_rad=upper,
        limit_margin_rad=margin,
        target_base_from_flange=target_base_matrix,
        position_error_mm=np.asarray(best["position_error_mm"]),
        orientation_error_deg=np.asarray(best["orientation_error_deg"]),
        reachable=np.asarray(reachable),
    )
    report = {
        "schema_version": 1,
        "status": "PASS" if reachable else "FAIL",
        "audit_scope": "kinematic reachability only; no simulator command and no path collision claim",
        "issues_robot_commands": False,
        "robot_urdf": str(ROBOT_URDF),
        "target_source": str(target_path),
        "target_stage": "pregrasp",
        "initial_target_gap": {
            "position_mm": initial_position_mm,
            "orientation_deg": initial_orientation_deg,
        },
        "best_solution": {
            "right_arm_joint_names": RIGHT_ARM_NAMES,
            "right_arm_joint_deg": np.degrees(best["q_rad"]).tolist(),
            "position_error_mm": best["position_error_mm"],
            "orientation_error_deg": best["orientation_error_deg"],
            "minimum_joint_limit_margin_deg": float(np.degrees(np.min(margin))),
            "start_index": best["start_index"],
            "iterations": best["nfev"],
        },
        "acceptance": {"position_error_mm_max": 2.0, "orientation_error_deg_max": 2.0},
        "next_gate": (
            "path collision and interpolated joint-limit audit"
            if reachable
            else "select another official collision-valid grasp candidate or revise the arm/table layout"
        ),
        "output_npz": str(solution_path),
    }
    report_path = output_root / "pregrasp_read_only_ik_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[READ-ONLY IK {report['status']}] no Isaac Sim command was issued")
    print(f"initial gap: {initial_position_mm:.2f} mm, {initial_orientation_deg:.2f} deg")
    print(
        f"best error: {best['position_error_mm']:.3f} mm, "
        f"{best['orientation_error_deg']:.3f} deg"
    )
    print(f"q solution (deg): {np.round(np.degrees(best['q_rad']), 3).tolist()}")
    print(f"minimum limit margin: {np.degrees(np.min(margin)):.3f} deg")
    print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()
