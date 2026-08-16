#!/usr/bin/env python3
"""Coarsely screen all collision-valid grasps for right-arm reachability.

The expensive finger retargeting is intentionally not repeated for all 708
candidates.  This first pass reuses the audited LEAP-to-Wuji2 PREGRASP root
alignment from candidate 274, solves bounded 7-DOF flange IK, then emits a
shortlist.  Shortlisted candidates still require full Wuji2 retargeting and a
second exact IK audit before any simulator motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares


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
ROBOT_URDF = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2"
    / "urdf/dual_arm_right_wuji2.urdf"
)
ASSEMBLY_SPEC = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2"
    / "config/assembly_spec.json"
)
LAYOUT_JSON = PROJECT_ROOT / "08_dual_arm_scene_layout/outputs/manual_layout_calibrated.json"
RIGHT_ARM_NAMES = [f"arm_r_joint_{index}" for index in range(1, 8)]
INITIAL_RIGHT_ARM_DEG = np.asarray([50.0, -70.0, 0.0, 40.0, 35.0, 0.0, 25.0])


def column_transform(row_matrix: list[list[float]]) -> np.ndarray:
    return np.asarray(row_matrix, dtype=np.float64).T


def pose_error(current: pin.SE3, target: pin.SE3) -> tuple[float, float]:
    position_mm = 1000.0 * float(np.linalg.norm(current.translation - target.translation))
    orientation_deg = float(np.degrees(np.linalg.norm(pin.log3(current.rotation.T @ target.rotation))))
    return position_mm, orientation_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=int, default=40)
    parser.add_argument("--refine-starts", type=int, default=12)
    args = parser.parse_args()

    collision_path = CAPTURE_TARGET / "official_leap_target_collision_filtered.npz"
    prediction_path = CAPTURE_TARGET / "official_leap_1024_target_ranked.npz"
    reference_path = REFERENCE_CASE / "06_isaacsim/final_waypoints.npz"
    with np.load(collision_path, allow_pickle=False) as archive:
        collision = {key: archive[key] for key in archive.files}
    with np.load(prediction_path, allow_pickle=False) as archive:
        score_all = np.asarray(archive["score"], dtype=np.float64)
    with np.load(reference_path, allow_pickle=False) as archive:
        reference_leap = np.asarray(archive["source_leap_waypoint_pose_world"][0, 0], dtype=np.float64)
        reference_wuji = np.asarray(archive["waypoint_pose_world"][0, 0], dtype=np.float64)
    approximate_wuji_from_leap = np.linalg.inv(reference_leap) @ reference_wuji

    layout = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))
    world_from_source = np.eye(4, dtype=np.float64)
    world_from_source[:3, 3] = np.asarray(
        layout["transforms"]["source_zone"]["position_world_m"], dtype=np.float64
    )
    world_from_base = column_transform(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"]
    )
    base_from_world = np.linalg.inv(world_from_base)
    assembly = json.loads(ASSEMBLY_SPEC.read_text(encoding="utf-8"))
    mount = assembly["mount_transform_parent_to_child"]
    flange_from_wrist = np.eye(4, dtype=np.float64)
    from scipy.spatial.transform import Rotation
    flange_from_wrist[:3, :3] = Rotation.from_euler("xyz", mount["rpy_rad"]).as_matrix()
    flange_from_wrist[:3, 3] = np.asarray(mount["xyz_m"], dtype=np.float64)

    valid_candidate = np.asarray(collision["valid_candidate_index"], dtype=np.int64)
    target_candidate = np.asarray(collision["target_candidate_index"], dtype=np.int64)
    local_by_candidate = {int(candidate): local for local, candidate in enumerate(target_candidate)}
    pregrasp_leap = np.stack(
        [collision["waypoint_pose_source"][local_by_candidate[int(candidate)], 0] for candidate in valid_candidate]
    ).astype(np.float64)
    target_base_matrices = (
        base_from_world[None]
        @ world_from_source[None]
        @ pregrasp_leap
        @ approximate_wuji_from_leap[None]
        @ np.linalg.inv(flange_from_wrist)[None]
    )

    model = pin.buildModelFromUrdf(str(ROBOT_URDF))
    data = model.createData()
    frame_id = model.getFrameId("arm_r_link_tf")
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
        return data.oMf[frame_id].copy()

    def solve(target_matrix: np.ndarray, starts: list[np.ndarray], max_nfev: int) -> dict:
        target = pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])

        def residual(right_q: np.ndarray) -> np.ndarray:
            current = frame_at(right_q)
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
                np.clip(start, lower, upper),
                bounds=(lower, upper),
                max_nfev=max_nfev,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
            )
            position_mm, orientation_deg = pose_error(frame_at(result.x), target)
            record = {
                "q": result.x.copy(),
                "position_mm": position_mm,
                "orientation_deg": orientation_deg,
                "metric": position_mm + 2.0 * orientation_deg,
            }
            if best is None or record["metric"] < best["metric"]:
                best = record
        assert best is not None
        return best

    print(f"[COARSE SCREEN] {len(valid_candidate)} official collision-valid candidates")
    coarse = []
    center = 0.5 * (lower + upper)
    for index, (candidate, target_matrix) in enumerate(zip(valid_candidate, target_base_matrices)):
        result = solve(target_matrix, [initial, center], max_nfev=180)
        result["candidate"] = int(candidate)
        result["target_matrix"] = target_matrix
        result["score"] = float(score_all[int(candidate)])
        coarse.append(result)
        if (index + 1) % 100 == 0:
            print(f"  coarse {index + 1}/{len(valid_candidate)}")
    coarse.sort(key=lambda item: item["metric"])

    rng = np.random.default_rng(20260813)
    random_starts = [rng.uniform(lower, upper) for _ in range(max(0, args.refine_starts - 2))]
    refined = []
    for item in coarse[: args.shortlist]:
        starts = [initial, item["q"], *random_starts]
        result = solve(item["target_matrix"], starts, max_nfev=600)
        result["candidate"] = item["candidate"]
        result["score"] = item["score"]
        refined.append(result)
    refined.sort(key=lambda item: item["metric"])

    output_root = REFERENCE_CASE / "07_arm_execution"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, item in enumerate(refined):
        rows.append(
            {
                "reachability_rank": rank,
                "candidate_index": item["candidate"],
                "official_score": item["score"],
                "position_error_mm": item["position_mm"],
                "orientation_error_deg": item["orientation_deg"],
                "coarse_reachable": item["position_mm"] <= 3.0 and item["orientation_deg"] <= 3.0,
                "right_arm_joint_deg": np.degrees(item["q"]).tolist(),
            }
        )
    report = {
        "schema_version": 1,
        "status": "SHORTLIST_READY",
        "scope": "coarse arm reachability using candidate274 root alignment; full per-candidate retarget still required",
        "issues_robot_commands": False,
        "collision_valid_candidates": int(len(valid_candidate)),
        "coarse_candidates_refined": len(rows),
        "coarse_reachable_count": int(sum(row["coarse_reachable"] for row in rows)),
        "candidates": rows,
    }
    report_path = output_root / "arm_reachability_shortlist.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_root / "arm_reachability_shortlist.npz",
        candidate_index=np.asarray([row["candidate_index"] for row in rows], dtype=np.int64),
        official_score=np.asarray([row["official_score"] for row in rows]),
        position_error_mm=np.asarray([row["position_error_mm"] for row in rows]),
        orientation_error_deg=np.asarray([row["orientation_error_deg"] for row in rows]),
        coarse_reachable=np.asarray([row["coarse_reachable"] for row in rows]),
        right_arm_joint_deg=np.asarray([row["right_arm_joint_deg"] for row in rows]),
    )
    print(f"[PASS] refined={len(rows)}, coarse reachable={report['coarse_reachable_count']}")
    for row in rows[:10]:
        print(
            f"  rank={row['reachability_rank']:02d} candidate={row['candidate_index']:04d} "
            f"error={row['position_error_mm']:.2f}mm/{row['orientation_error_deg']:.2f}deg "
            f"score={row['official_score']:.3f}"
        )
    print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()
