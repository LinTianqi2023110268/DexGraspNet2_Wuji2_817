#!/usr/bin/env python3
"""Screen official collision-valid grasps for PREGRASP and COVER reachability.

This is a read-only coarse gate.  It reuses the audited LEAP-to-Wuji2 rigid
root mapping only to avoid retargeting hundreds of candidates.  Every selected
candidate must still pass the full Wuji2 retargeting and exact all-waypoint IK
audit before it can be commanded in Isaac Lab.
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
CAPTURE_TARGET = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/captures/isaaclab_scene0000_smoke/dgn2/ashtray"
)
REFERENCE_CASE = (
    PROJECT_ROOT
    / "06_leap_to_wuji2_final_pipeline/01_cases"
    / "live_scene0000_ashtray_isaaclab_candidate0274"
)
ROBOT_ROOT = PROJECT_ROOT / "01_environment/vendor/wuji-description"
ROBOT_URDF = ROBOT_ROOT / "dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
ASSEMBLY_SPEC = ROBOT_ROOT / "dual_arm_right_wuji2/config/assembly_spec.json"
LAYOUT_JSON = PROJECT_ROOT / "08_dual_arm_scene_layout/outputs/manual_layout_calibrated.json"
RIGHT_ARM_NAMES = [f"arm_r_joint_{index}" for index in range(1, 8)]
INITIAL_Q = np.deg2rad([50.0, -70.0, 0.0, 40.0, 35.0, 0.0, 25.0])
STAGES = ("pregrasp", "cover")


def pose_error(current: pin.SE3, target: pin.SE3) -> tuple[float, float]:
    position_mm = 1000.0 * float(np.linalg.norm(current.translation - target.translation))
    rotation_vector = pin.log3(current.rotation.T @ target.rotation)
    return position_mm, float(np.degrees(np.linalg.norm(rotation_vector)))


class RightArmIk:
    """Small bounded Pinocchio IK helper for the seven right-arm joints."""

    def __init__(self) -> None:
        self.model = pin.buildModelFromUrdf(str(ROBOT_URDF))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("arm_r_link_tf")
        joint_ids = [self.model.getJointId(name) for name in RIGHT_ARM_NAMES]
        self.q_indices = np.asarray([self.model.joints[joint].idx_q for joint in joint_ids])
        self.template = pin.neutral(self.model)
        self.lower = self.model.lowerPositionLimit[self.q_indices] + 0.01
        self.upper = self.model.upperPositionLimit[self.q_indices] - 0.01

    def frame_at(self, right_q: np.ndarray) -> pin.SE3:
        q = self.template.copy()
        q[self.q_indices] = right_q
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id].copy()

    def solve(self, target_matrix: np.ndarray, starts: list[np.ndarray]) -> dict:
        target = pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])

        def residual(right_q: np.ndarray) -> np.ndarray:
            current = self.frame_at(right_q)
            return np.concatenate(
                (
                    (current.translation - target.translation) / 0.01,
                    pin.log3(current.rotation.T @ target.rotation) / 0.10,
                )
            )

        best = None
        for start in starts:
            result = least_squares(
                residual,
                np.clip(start, self.lower, self.upper),
                bounds=(self.lower, self.upper),
                max_nfev=260,
                xtol=1.0e-9,
                ftol=1.0e-9,
                gtol=1.0e-9,
            )
            position_mm, orientation_deg = pose_error(self.frame_at(result.x), target)
            record = {
                "q_rad": result.x.copy(),
                "position_error_mm": position_mm,
                "orientation_error_deg": orientation_deg,
                "metric": position_mm + 2.0 * orientation_deg,
            }
            if best is None or record["metric"] < best["metric"]:
                best = record
        assert best is not None
        return best


def transforms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))
    world_from_source = np.eye(4)
    world_from_source[:3, 3] = layout["transforms"]["source_zone"]["position_world_m"]
    world_from_base = np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T

    assembly = json.loads(ASSEMBLY_SPEC.read_text(encoding="utf-8"))
    mount = assembly["mount_transform_parent_to_child"]
    flange_from_wrist = np.eye(4)
    flange_from_wrist[:3, :3] = Rotation.from_euler("xyz", mount["rpy_rad"]).as_matrix()
    flange_from_wrist[:3, 3] = mount["xyz_m"]
    return world_from_source, np.linalg.inv(world_from_base), np.linalg.inv(flange_from_wrist)


def approximate_root_mapping() -> np.ndarray:
    reference = REFERENCE_CASE / "06_isaacsim/final_waypoints.npz"
    with np.load(reference, allow_pickle=False) as archive:
        leap = np.asarray(archive["source_leap_waypoint_pose_world"][0, 0], dtype=np.float64)
        wuji = np.asarray(archive["waypoint_pose_world"][0, 0], dtype=np.float64)
    return np.linalg.inv(leap) @ wuji


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-target", type=Path, default=CAPTURE_TARGET)
    parser.add_argument("--reference-case", type=Path, default=REFERENCE_CASE)
    parser.add_argument("--keep", type=int, default=30)
    parser.add_argument("--position-mm", type=float, default=8.0)
    parser.add_argument("--orientation-deg", type=float, default=8.0)
    args = parser.parse_args()

    capture_target = args.capture_target.resolve()
    reference_case = args.reference_case.resolve()
    collision_path = capture_target / "official_leap_target_collision_filtered.npz"
    prediction_path = capture_target / "official_leap_1024_target_ranked.npz"
    with np.load(collision_path, allow_pickle=False) as archive:
        collision = {key: archive[key] for key in archive.files}
    with np.load(prediction_path, allow_pickle=False) as archive:
        scores = np.asarray(archive["score"], dtype=np.float64)

    valid = np.asarray(collision["valid_candidate_index"], dtype=np.int64)
    target = np.asarray(collision["target_candidate_index"], dtype=np.int64)
    local_by_candidate = {int(candidate): index for index, candidate in enumerate(target)}
    order = sorted(valid.tolist(), key=lambda candidate: -scores[candidate])

    world_from_source, base_from_world, wrist_from_flange = transforms()
    reference = reference_case / "06_isaacsim/final_waypoints.npz"
    with np.load(reference, allow_pickle=False) as archive:
        leap = np.asarray(archive["source_leap_waypoint_pose_world"][0, 0], dtype=np.float64)
        wuji = np.asarray(archive["waypoint_pose_world"][0, 0], dtype=np.float64)
    wuji_from_leap = np.linalg.inv(leap) @ wuji
    ik = RightArmIk()
    center = 0.5 * (ik.lower + ik.upper)
    rows = []

    print(f"[APPROACH SCREEN] official collision-valid candidates={len(order)}")
    for rank, candidate in enumerate(order):
        local = local_by_candidate[candidate]
        leap_waypoints = np.asarray(collision["waypoint_pose_source"][local], dtype=np.float64)
        stage_results = {}
        previous = INITIAL_Q
        for stage_index, stage_name in enumerate(STAGES):
            target_base_flange = (
                base_from_world
                @ world_from_source
                @ leap_waypoints[stage_index]
                @ wuji_from_leap
                @ wrist_from_flange
            )
            result = ik.solve(target_base_flange, [previous, INITIAL_Q, center])
            previous = result["q_rad"]
            stage_results[stage_name] = result

        passed = all(
            result["position_error_mm"] <= args.position_mm
            and result["orientation_error_deg"] <= args.orientation_deg
            for result in stage_results.values()
        )
        if passed:
            rows.append(
                {
                    "official_score_rank": rank,
                    "candidate_index": candidate,
                    "official_score": float(scores[candidate]),
                    "pregrasp_position_error_mm": stage_results["pregrasp"]["position_error_mm"],
                    "pregrasp_orientation_error_deg": stage_results["pregrasp"]["orientation_error_deg"],
                    "cover_position_error_mm": stage_results["cover"]["position_error_mm"],
                    "cover_orientation_error_deg": stage_results["cover"]["orientation_error_deg"],
                    "pregrasp_q_deg": np.degrees(stage_results["pregrasp"]["q_rad"]).tolist(),
                    "cover_q_deg": np.degrees(stage_results["cover"]["q_rad"]).tolist(),
                }
            )
            print(
                f"  PASS candidate={candidate:04d} score={scores[candidate]:.3f} | "
                f"pre={stage_results['pregrasp']['position_error_mm']:.2f}mm/"
                f"{stage_results['pregrasp']['orientation_error_deg']:.2f}deg | "
                f"cover={stage_results['cover']['position_error_mm']:.2f}mm/"
                f"{stage_results['cover']['orientation_error_deg']:.2f}deg"
            )
            if len(rows) >= args.keep:
                break
        if (rank + 1) % 100 == 0:
            print(f"  screened {rank + 1}/{len(order)}")

    report = {
        "schema_version": 1,
        "status": "PASS" if rows else "FAIL",
        "scope": "coarse PREGRASP+COVER arm reachability; full retarget and exact IK still required",
        "issues_robot_commands": False,
        "thresholds": {
            "position_error_mm_max": args.position_mm,
            "orientation_error_deg_max": args.orientation_deg,
        },
        "collision_valid_candidates": len(order),
        "kept_candidates": len(rows),
        "candidates": rows,
    }
    output = capture_target / "arm_approach_reachability_shortlist.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[APPROACH SCREEN {report['status']}] kept={len(rows)}")
    print(f"[REPORT] {output}")
    if not rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
