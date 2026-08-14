#!/usr/bin/env python3
"""Solve exact bounded right-arm IK for all generated Wuji2 wrist waypoints.

The program is offline and read-only with respect to Isaac Sim.  It uses the
assembled dual-arm + Wuji2 URDF, preserves the calibrated initial posture, and
writes one ordered q7 target through placement and return.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SCRIPTS = (
    PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/runtime/scripts"
)
sys.path.insert(0, str(RUNTIME_SCRIPTS))

from placement_allocator import allocate_placement, load_json
ROBOT_URDF = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2"
    / "urdf/dual_arm_right_wuji2.urdf"
)
LAYOUT_JSON = PROJECT_ROOT / "08_dual_arm_scene_layout/outputs/manual_layout_calibrated.json"
PLACEMENT_POLICY_JSON = (
    PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/runtime/config/placement_policy.json"
)
RIGHT_ARM_NAMES = [f"arm_r_joint_{index}" for index in range(1, 8)]
INITIAL_Q = np.deg2rad([50.0, -70.0, 0.0, 40.0, 35.0, 0.0, 25.0])


def pose_error(current: pin.SE3, target: pin.SE3) -> tuple[float, float]:
    position_mm = 1000.0 * float(np.linalg.norm(current.translation - target.translation))
    orientation_deg = float(np.degrees(np.linalg.norm(pin.log3(current.rotation.T @ target.rotation))))
    return position_mm, orientation_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--position-mm", type=float, default=5.0)
    parser.add_argument("--orientation-deg", type=float, default=5.0)
    parser.add_argument("--placement-policy", type=Path, default=PLACEMENT_POLICY_JSON)
    parser.add_argument(
        "--release-hand-height-offset-m", "--release-clearance-m",
        dest="release_clearance_m", type=float,
        help="Override release hand Z above its GRASP height for a controlled audit.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        help="Override the normal case 07_arm_execution directory (useful for non-destructive audits).",
    )
    parser.add_argument(
        "--placement-slot-index", type=int,
        help="Choose the Nth currently free footprint-aware slot; omit for the first free slot.",
    )
    parser.add_argument(
        "--grasp-only",
        action="store_true",
        help="Screen only PREGRASP..LIFT; skip transfer/place generation until a grasp is reachable.",
    )
    args = parser.parse_args()

    case_root = args.case_root.resolve()
    target_path = case_root / "07_arm_execution/arm_flange_targets.npz"
    with np.load(target_path, allow_pickle=False) as archive:
        names = np.asarray(archive["waypoint_names"])
        targets_world = np.asarray(archive["world_from_right_flange"], dtype=np.float64)
        wrist_targets_world = np.asarray(archive["world_from_wuji2_wrist"], dtype=np.float64)
        world_from_source = np.asarray(archive["world_from_source_zone"], dtype=np.float64)

    layout = load_json(LAYOUT_JSON)
    world_from_base = np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T
    base_from_world = np.linalg.inv(world_from_base)

    # Keep the grasp-time object-to-flange transform fixed during transport.
    # A footprint-aware allocator replaces the old hard-coded placement XY.
    # It preserves the object's stable orientation and computes root Z from the
    # real lower surface, not from an arbitrary centre-height assumption.
    manifest_paths = sorted((case_root / "01_input").glob("scene_*_manifest.json"))
    if len(manifest_paths) != 1:
        raise RuntimeError(f"Expected one scene manifest, got {manifest_paths}")
    scene_manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    case_record = json.loads((case_root / "case.json").read_text(encoding="utf-8"))
    target_segmentation_id = int(case_record["target_segmentation_id"])
    target_object = next(
        item for item in scene_manifest["objects"]
        if int(item["segmentation_id"]) == target_segmentation_id
    )
    world_from_object_initial = world_from_source @ np.asarray(
        target_object["pose_world_object"], dtype=np.float64
    )
    grasp_index = int(np.flatnonzero(names == "grasp")[0])
    flange_from_object = np.linalg.inv(targets_world[grasp_index]) @ world_from_object_initial
    surface_path = case_root / "01_input" / f"object_{target_segmentation_id:03d}_surface_points.npy"
    if not surface_path.is_file():
        raise FileNotFoundError(surface_path)
    placement_policy = load_json(args.placement_policy)
    if args.release_clearance_m is not None:
        if args.release_clearance_m < 0.0:
            raise ValueError("--release-clearance-m must be non-negative")
        placement_policy = dict(placement_policy)
        placement_policy["release_hand_height_above_grasp_m"] = float(args.release_clearance_m)
    placement_plan = allocate_placement(
        project_root=PROJECT_ROOT,
        layout=layout,
        policy=placement_policy,
        surface_points_object=np.load(surface_path),
        world_from_object_initial=world_from_object_initial,
        requested_slot_index=args.placement_slot_index,
    )
    world_from_object_place = world_from_object_initial.copy()
    world_from_object_place[:3, 3] = np.asarray(
        placement_plan["object_root_place_world_m"], dtype=np.float64
    )
    world_from_object_transfer = world_from_object_place.copy()
    world_from_object_transfer[2, 3] += float(placement_plan["transfer_clearance_m"])
    release_hand_world_z = float(
        wrist_targets_world[grasp_index][2, 3]
        + placement_plan["release_hand_height_above_grasp_m"]
    )
    placement_plan["hand_height_frame"] = "Wuji2 r_wrist origin"
    placement_plan["grasp_hand_world_z_m"] = float(wrist_targets_world[grasp_index][2, 3])
    placement_plan["release_hand_world_z_m"] = release_hand_world_z
    # Nominal placement frames.  The right arm cannot retain the exact GRASP
    # rotation at the green zone, therefore placement IK below constrains the
    # required translation and selects the feasible solution with the smallest
    # rotation/joint change from the preceding stage.  RELEASE reuses PLACE q7
    # exactly; RETREAT is a world +Z translation with minimum rotation change.
    grasp_flange = targets_world[grasp_index].copy()
    grasp_rotation = grasp_flange[:3, :3].copy()
    flange_from_wrist = np.linalg.inv(grasp_flange) @ wrist_targets_world[grasp_index]

    # TRANSFER carries the rigidly held object above the allocated slot while
    # retaining its grasp orientation.
    transfer = world_from_object_transfer @ np.linalg.inv(flange_from_object)

    # PLACE/RELEASE obey the user-selected rule exactly:
    #   Wuji2 r_wrist world Z = GRASP r_wrist world Z + 10 mm.
    # The object XY is the allocated slot.  With flange orientation frozen,
    # these three scalar constraints determine the flange translation.
    place = grasp_flange.copy()
    object_offset_world = grasp_rotation @ flange_from_object[:3, 3]
    wrist_offset_world = grasp_rotation @ flange_from_wrist[:3, 3]
    place[0, 3] = world_from_object_place[0, 3] - object_offset_world[0]
    place[1, 3] = world_from_object_place[1, 3] - object_offset_world[1]
    place[2, 3] = release_hand_world_z - wrist_offset_world[2]
    release = place.copy()

    # RETREAT is a pure world +Z translation.  Orientation is identical to
    # RELEASE, so opening fingers cannot be followed by a wrist flip.
    retreat = release.copy()
    retreat[2, 3] += float(placement_plan["retreat_clearance_m"])

    induced_object_place = place @ flange_from_object
    placement_plan["induced_object_root_release_world_m"] = (
        induced_object_place[:3, 3].tolist()
    )
    placement_plan["transport_orientation_policy"] = (
        "minimum_stage_to_stage_rotation; RELEASE_identical_to_PLACE; RETREAT_world_plus_Z"
    )
    if not args.grasp_only:
        names = np.concatenate((names, np.asarray(["transfer", "place", "release", "retreat"])))
        targets_world = np.concatenate(
            (targets_world, transfer[None], place[None], release[None], retreat[None]), axis=0
        )

    model = pin.buildModelFromUrdf(str(ROBOT_URDF))
    data = model.createData()
    flange_frame = model.getFrameId("arm_r_link_tf")
    wrist_frame = model.getFrameId("r_wrist")
    joint_ids = [model.getJointId(name) for name in RIGHT_ARM_NAMES]
    q_indices = np.asarray([model.joints[joint_id].idx_q for joint_id in joint_ids])
    q_template = pin.neutral(model)
    lower = model.lowerPositionLimit[q_indices] + 0.01
    upper = model.upperPositionLimit[q_indices] - 0.01

    def frame_at(right_q: np.ndarray) -> pin.SE3:
        q = q_template.copy()
        q[q_indices] = right_q
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[flange_frame].copy()

    def wrist_at(right_q: np.ndarray) -> pin.SE3:
        q = q_template.copy()
        q[q_indices] = right_q
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[wrist_frame].copy()

    rng = np.random.default_rng(20260813)
    solved = []
    previous = INITIAL_Q.copy()
    reports = []
    previous_achieved = None
    achieved_place_world = None
    for name, target_world in zip(names, targets_world):
        stage_name = str(name)
        target_matrix = base_from_world @ target_world
        target = pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])
        placement_stage = str(name) in {"transfer", "place", "release", "retreat"}

        # Opening the hand must not move the arm.  This removes the old
        # PLACE->RELEASE pose change and makes the release semantics auditable.
        if stage_name == "release":
            best_q = previous.copy()
            achieved = frame_at(best_q)
            base_from_flange = np.eye(4, dtype=np.float64)
            base_from_flange[:3, :3] = achieved.rotation
            base_from_flange[:3, 3] = achieved.translation
            achieved_world = world_from_base @ base_from_flange
            targets_world[len(solved)] = achieved_world
            solved.append(best_q.copy())
            minimum_margin_deg = float(np.degrees(np.min(
                np.minimum(best_q - lower, upper - best_q)
            )))
            reports.append({
                "stage": stage_name,
                "status": "PASS",
                "position_error_mm": 0.0,
                "orientation_error_deg": 0.0,
                "orientation_change_from_previous_deg": 0.0,
                "minimum_limit_margin_deg": minimum_margin_deg,
                "right_arm_joint_deg": np.degrees(best_q).tolist(),
                "placement_plan_applies": True,
                "arm_motion_policy": "identical_to_PLACE; hand_only_release",
            })
            previous_achieved = achieved.copy()
            print(
                f"[PASS] {stage_name:8s} 0.000 mm / 0.000 deg step | "
                f"margin={minimum_margin_deg:.2f} deg"
            )
            continue

        held_object_target = (
            world_from_object_transfer[:3, 3]
            if stage_name == "transfer"
            else world_from_object_place[:3, 3]
        )

        def residual(right_q: np.ndarray) -> np.ndarray:
            current = frame_at(right_q)
            if placement_stage:
                base_from_flange = np.eye(4, dtype=np.float64)
                base_from_flange[:3, :3] = current.rotation
                base_from_flange[:3, 3] = current.translation
                current_world = world_from_base @ base_from_flange
                if stage_name == "retreat":
                    if achieved_place_world is None:
                        raise RuntimeError("RETREAT requires an achieved PLACE pose")
                    target_xyz = achieved_place_world[:3, 3].copy()
                    target_xyz[2] += float(placement_plan["retreat_clearance_m"])
                    return (current_world[:3, 3] - target_xyz) / 0.01
                world_from_held_object = current_world @ flange_from_object
                if stage_name == "place":
                    current_wrist = wrist_at(right_q)
                    base_from_wrist = np.eye(4, dtype=np.float64)
                    base_from_wrist[:3, :3] = current_wrist.rotation
                    base_from_wrist[:3, 3] = current_wrist.translation
                    wrist_world_z = float((world_from_base @ base_from_wrist)[2, 3])
                    return np.concatenate((
                        (world_from_held_object[:2, 3] - held_object_target[:2]) / 0.01,
                        np.asarray([wrist_world_z - release_hand_world_z]) / 0.01,
                    ))
                return (world_from_held_object[:3, 3] - held_object_target) / 0.01
            position = (current.translation - target.translation) / 0.01
            orientation = pin.log3(current.rotation.T @ target.rotation) / 0.10
            return np.concatenate((position, orientation))

        random_count = 120 if placement_stage else max(0, args.starts - 3)
        random_starts = [rng.uniform(lower, upper) for _ in range(random_count)]
        starts = [previous, INITIAL_Q, 0.5 * (lower + upper), *random_starts]
        candidates = []
        for start in starts:
            result = least_squares(
                residual,
                np.clip(start, lower, upper),
                bounds=(lower, upper),
                max_nfev=800,
                xtol=1.0e-11,
                ftol=1.0e-11,
                gtol=1.0e-11,
            )
            if placement_stage:
                position_mm = 10.0 * float(np.linalg.norm(residual(result.x)))
                reference_rotation = (
                    previous_achieved.rotation
                    if previous_achieved is not None
                    else frame_at(previous).rotation
                )
                orientation_deg = float(np.degrees(np.linalg.norm(
                    pin.log3(reference_rotation.T @ frame_at(result.x).rotation)
                )))
            else:
                position_mm, orientation_deg = pose_error(frame_at(result.x), target)
            motion = float(np.linalg.norm(result.x - previous))
            margin_deg = float(
                np.degrees(np.min(np.minimum(result.x - lower, upper - result.x)))
            )
            margin_penalty = 20.0 * max(0.0, 3.0 - margin_deg)
            feasible = position_mm <= args.position_mm and margin_deg >= 3.0
            continuity_cost = orientation_deg + 0.10 * float(np.degrees(motion))
            # Feasibility is lexicographically dominant.  Once feasible,
            # minimise stage-to-stage rotation and joint movement.
            primary = 0.0 if feasible else 1.0e6 + 1000.0 * position_mm
            candidates.append(
                (
                    primary + continuity_cost + margin_penalty,
                    result.x,
                    position_mm,
                    orientation_deg,
                )
            )
        _, best_q, position_mm, orientation_deg = min(candidates, key=lambda item: item[0])
        margin = np.minimum(best_q - lower, upper - best_q)
        minimum_margin_deg = float(np.degrees(np.min(margin)))
        stage_pass = (
            position_mm <= args.position_mm
            and (placement_stage or orientation_deg <= args.orientation_deg)
            and minimum_margin_deg >= 3.0
        )
        solved.append(best_q.copy())
        # Store the complete achieved flange pose.  Runtime telemetry therefore
        # compares against the pose that corresponds to this exact q7 solution.
        if placement_stage:
            achieved = frame_at(best_q)
            base_from_flange = np.eye(4, dtype=np.float64)
            base_from_flange[:3, :3] = achieved.rotation
            base_from_flange[:3, 3] = achieved.translation
            achieved_world = world_from_base @ base_from_flange
            targets_world[len(solved) - 1] = achieved_world
            previous_achieved = achieved.copy()
            if stage_name == "place":
                achieved_place_world = achieved_world.copy()
                placement_plan["achieved_object_root_release_world_m"] = (
                    (achieved_world @ flange_from_object)[:3, 3].tolist()
                )
        previous = best_q.copy()
        reports.append(
            {
                "stage": str(name),
                "status": "PASS" if stage_pass else "FAIL",
                "position_error_mm": position_mm,
                "orientation_error_deg": orientation_deg,
                "orientation_change_from_previous_deg": (
                    orientation_deg if placement_stage else 0.0
                ),
                "minimum_limit_margin_deg": minimum_margin_deg,
                "right_arm_joint_deg": np.degrees(best_q).tolist(),
                "placement_plan_applies": placement_stage,
            }
        )
        print(
            f"[{reports[-1]['status']}] {str(name):8s} "
            f"{position_mm:.3f} mm / {orientation_deg:.3f} deg | "
            f"margin={reports[-1]['minimum_limit_margin_deg']:.2f} deg"
        )

    passed = all(item["status"] == "PASS" for item in reports)
    output_root = (
        args.output_root.resolve() if args.output_root is not None
        else case_root / "07_arm_execution"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_npz = output_root / "full_arm_waypoint_ik.npz"
    np.savez_compressed(
        output_npz,
        waypoint_names=names,
        right_arm_joint_names=np.asarray(RIGHT_ARM_NAMES),
        initial_right_arm_q_rad=INITIAL_Q,
        solved_right_arm_q_rad=np.asarray(solved),
        lower_limit_rad=lower,
        upper_limit_rad=upper,
        position_error_mm=np.asarray([x["position_error_mm"] for x in reports]),
        orientation_error_deg=np.asarray([x["orientation_error_deg"] for x in reports]),
        world_from_right_flange=targets_world,
        placement_object_root_world_m=np.asarray(placement_plan["object_root_place_world_m"]),
        placement_footprint_world_xy_min_m=np.asarray(placement_plan["footprint_world_xy_min_m"]),
        placement_footprint_world_xy_max_m=np.asarray(placement_plan["footprint_world_xy_max_m"]),
        all_reachable=np.asarray(passed),
    )
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "scope": (
            "exact offline PREGRASP-through-LIFT kinematic reachability screening"
            if args.grasp_only
            else "exact offline grasp-and-place waypoint kinematic reachability; no path or physics claim"
        ),
        "issues_robot_commands": False,
        "robot_urdf": str(ROBOT_URDF),
        "thresholds": {
            "position_error_mm_max": args.position_mm,
            "orientation_error_deg_max": args.orientation_deg,
        },
        "waypoints": reports,
        "placement_policy": str(args.placement_policy.resolve()),
        "placement_plan": placement_plan,
        "output_npz": str(output_npz),
    }
    report_path = output_root / "full_arm_waypoint_ik_report.json"
    placement_path = output_root / "placement_plan.json"
    placement_path.write_text(json.dumps(placement_plan, indent=2) + "\n", encoding="utf-8")
    report["placement_plan_file"] = str(placement_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[FULL WAYPOINT IK {report['status']}] {report_path}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
