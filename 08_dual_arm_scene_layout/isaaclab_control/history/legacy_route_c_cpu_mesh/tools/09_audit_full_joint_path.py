#!/usr/bin/env python3
"""Audit the complete HOME->PREGRASP->...->LIFT joint path offline.

The arm q7 and Wuji2 q20 are interpolated together according to the generated
waypoints.  HPP-FCL checks the table, all six scene objects, and newly-created
self-collision pairs.  No simulator is started and no command is issued.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import hppfcl
import numpy as np
import pinocchio as pin


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBOT_ROOT = PROJECT_ROOT / "01_environment/vendor/wuji-description"
ROBOT_URDF = ROBOT_ROOT / "dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
WUJI_PACKAGE_SOURCE = ROBOT_ROOT / "hand2/hand2_beta1/body"
LAYOUT_JSON = PROJECT_ROOT / "08_dual_arm_scene_layout/outputs/manual_layout_calibrated.json"
RIGHT_ARM_NAMES = [f"arm_r_joint_{index}" for index in range(1, 8)]


def active_pairs(model: pin.GeometryModel, data: pin.GeometryData) -> set[tuple[str, str]]:
    result = set()
    for pair, collision in zip(model.collisionPairs, data.collisionResults):
        if collision.isCollision():
            a = model.geometryObjects[pair.first].name
            b = model.geometryObjects[pair.second].name
            result.add(tuple(sorted((a, b))))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--samples-per-segment", type=int, default=81)
    parser.add_argument(
        "--through-stage", default="lift",
        choices=["pregrasp", "cover", "grasp", "squeeze", "lift", "retreat"],
        help="Last stage to audit; lift is the physical grasp gate used before transfer.",
    )
    args = parser.parse_args()
    case_root = args.case_root.resolve()

    with np.load(case_root / "07_arm_execution/full_arm_waypoint_ik.npz") as archive:
        if not bool(np.asarray(archive["all_reachable"]).item()):
            raise RuntimeError("all-waypoint IK has not passed")
        names = [str(x) for x in archive["waypoint_names"]]
        arm_initial = np.asarray(archive["initial_right_arm_q_rad"], dtype=np.float64)
        arm_waypoints = np.asarray(archive["solved_right_arm_q_rad"], dtype=np.float64)
    with np.load(case_root / "07_arm_execution/arm_flange_targets.npz") as archive:
        world_from_source = np.asarray(archive["world_from_source_zone"], dtype=np.float64)
    stop_index = names.index(args.through_stage) + 1 if args.through_stage != "retreat" else len(names)
    names = names[:stop_index]
    arm_waypoints = arm_waypoints[:stop_index]
    with np.load(case_root / "06_isaacsim/final_waypoints.npz") as archive:
        hand_names = [str(x) for x in archive["finger_joint_names"]]
        hand_waypoints = np.asarray(archive["waypoint_joint_positions"][0], dtype=np.float64)
        target_segmentation_id = int(np.asarray(archive["target_segmentation_id"]).reshape(-1)[0])
    manifest_paths = sorted((case_root / "01_input").glob("scene_*_manifest.json"))
    if len(manifest_paths) != 1:
        raise RuntimeError(f"Expected one scene manifest, got {manifest_paths}")
    scene = json.loads(manifest_paths[0].read_text())
    layout = json.loads(LAYOUT_JSON.read_text())

    model = pin.buildModelFromUrdf(str(ROBOT_URDF))
    data = model.createData()
    q = pin.neutral(model)
    arm_indices = np.asarray([model.joints[model.getJointId(x)].idx_q for x in RIGHT_ARM_NAMES])
    hand_indices = np.asarray([model.joints[model.getJointId(x)].idx_q for x in hand_names])

    with tempfile.TemporaryDirectory(prefix="dgn2_full_path_") as temp_root:
        alias = Path(temp_root) / "wuji_hand2_description"
        alias.symlink_to(WUJI_PACKAGE_SOURCE, target_is_directory=True)
        geometry_model = pin.buildGeomFromUrdf(
            model,
            str(ROBOT_URDF),
            pin.GeometryType.COLLISION,
            package_dirs=[str(ROBOT_ROOT), temp_root],
        )

        world_from_base = np.asarray(
            layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
            dtype=np.float64,
        ).T
        base_from_world = np.linalg.inv(world_from_base)
        table_position = np.asarray(layout["transforms"]["table"]["position_world_m"])
        table_size = np.asarray(layout["geometry"]["table_size_m"])
        base_from_table = base_from_world.copy()
        base_from_table[:3, 3] += base_from_world[:3, :3] @ table_position
        geometry_model.addGeometryObject(
            pin.GeometryObject(
                "__layout_table__",
                0,
                hppfcl.Box(*table_size.tolist()),
                pin.SE3(base_from_table[:3, :3], base_from_table[:3, 3]),
            )
        )

        loader = hppfcl.MeshLoader()
        scene_names = set()
        for record in scene["objects"]:
            name = f"__scene_object_{int(record['segmentation_id']):03d}__"
            scene_names.add(name)
            base_from_object = (
                base_from_world
                @ world_from_source
                @ np.asarray(record["pose_world_object"], dtype=np.float64)
            )
            geometry_model.addGeometryObject(
                pin.GeometryObject(
                    name,
                    0,
                    loader.load(str(Path(record["visual_mesh"]).resolve())),
                    pin.SE3(base_from_object[:3, :3], base_from_object[:3, 3]),
                )
            )
        geometry_model.addAllCollisionPairs()
        geometry_data = pin.GeometryData(geometry_model)

        # URDF-to-geometry helpers such as ``addAllCollisionPairs`` include
        # geometry attached to the same joint and directly adjacent parent/
        # child links.  PhysX articulations disable these local pairs by
        # default; treating them as blockers creates false positives such as
        # r_wrist versus r_thumb_proximal.  Mirror that articulation rule.
        ignored_adjacent_pairs: set[tuple[str, str]] = set()
        for pair in geometry_model.collisionPairs:
            first = geometry_model.geometryObjects[pair.first]
            second = geometry_model.geometryObjects[pair.second]
            joint_a = int(first.parentJoint)
            joint_b = int(second.parentJoint)
            adjacent = (
                joint_a == joint_b
                or int(model.parents[joint_a]) == joint_b
                or int(model.parents[joint_b]) == joint_a
            )
            if adjacent:
                ignored_adjacent_pairs.add(tuple(sorted((first.name, second.name))))

        def collisions(arm_q: np.ndarray, hand_q: np.ndarray) -> set[tuple[str, str]]:
            q[arm_indices] = arm_q
            q[hand_indices] = hand_q
            pin.forwardKinematics(model, data, q)
            pin.updateGeometryPlacements(model, data, geometry_model, geometry_data, q)
            pin.computeCollisions(geometry_model, geometry_data, False)
            return active_pairs(geometry_model, geometry_data)

        target_name = f"__scene_object_{target_segmentation_id:03d}__"

        def target_clearance(arm_q: np.ndarray, hand_q: np.ndarray) -> dict:
            """Return the nearest robot-to-target signed HPP-FCL distance."""
            q[arm_indices] = arm_q
            q[hand_indices] = hand_q
            pin.forwardKinematics(model, data, q)
            pin.updateGeometryPlacements(model, data, geometry_model, geometry_data, q)
            pin.computeDistances(geometry_model, geometry_data)
            candidates = []
            for pair, result in zip(geometry_model.collisionPairs, geometry_data.distanceResults):
                first = geometry_model.geometryObjects[pair.first].name
                second = geometry_model.geometryObjects[pair.second].name
                if target_name not in {first, second}:
                    continue
                other = second if first == target_name else first
                if other in scene_names or other == "__layout_table__":
                    continue
                item = {
                    "signed_distance_m": float(result.min_distance),
                    "robot_geometry": other,
                    "target_geometry": target_name,
                }
                candidates.append(item)
            if not candidates:
                raise RuntimeError(f"No robot distance pair found for {target_name}")
            candidates.sort(key=lambda item: item["signed_distance_m"])
            return {
                **candidates[0],
                "nearest_robot_geometries": candidates[:20],
            }

        baseline = collisions(arm_initial, hand_waypoints[0])
        segments = []
        start_arm = arm_initial
        start_hand = hand_waypoints[0]
        for index, stage in enumerate(names):
            end_arm = arm_waypoints[index]
            if index < len(hand_waypoints):
                end_hand = hand_waypoints[index]
            elif stage in {"transfer", "place"}:
                end_hand = hand_waypoints[3]  # keep SQUEEZE while carrying
            elif stage in {"release", "retreat"}:
                end_hand = hand_waypoints[0]  # reopen before retreat
            else:
                raise RuntimeError(f"No hand waypoint policy for extended stage: {stage}")
            first = {}
            for sample, alpha in enumerate(np.linspace(0.0, 1.0, args.samples_per_segment)):
                arm = (1.0 - alpha) * start_arm + alpha * end_arm
                hand = (1.0 - alpha) * start_hand + alpha * end_hand
                for pair in collisions(arm, hand) - baseline:
                    first.setdefault(pair, sample)
            # COVER is the physical approach/closure segment.  First contact
            # with the selected object is allowed there, but never with the
            # table or any distractor object.
            allow_target_contact = stage in {"cover", "grasp", "squeeze", "lift"}
            blockers = []
            for pair, sample in sorted(first.items()):
                if pair in ignored_adjacent_pairs:
                    continue
                involves_table = "__layout_table__" in pair
                involved_scene = [x for x in pair if x in scene_names]
                allowed = allow_target_contact and involved_scene == [target_name]
                if involves_table or (involved_scene and not allowed) or not involved_scene:
                    blockers.append({"pair": list(pair), "first_sample": sample})
            clearance = target_clearance(end_arm, end_hand)
            segments.append(
                {
                    "to_stage": stage,
                    "status": "PASS" if not blockers else "FAIL",
                    "blocking_collision_pairs": blockers,
                    "nearest_target_clearance": clearance,
                }
            )
            print(
                f"[{segments[-1]['status']}] path -> {stage}: blockers={len(blockers)}; "
                f"target clearance={1000.0 * clearance['signed_distance_m']:+.3f} mm "
                f"({clearance['robot_geometry']})"
            )
            start_arm, start_hand = end_arm, end_hand

    passed = all(x["status"] == "PASS" for x in segments)
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "scope": "offline linear joint interpolation; selected-object contact allowed from COVER",
        "through_stage": args.through_stage,
        "issues_robot_commands": False,
        "samples_per_segment": args.samples_per_segment,
        "ignored_same_or_parent_child_pairs": len(ignored_adjacent_pairs),
        "segments": segments,
    }
    output = case_root / "07_arm_execution/full_joint_path_collision_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[FULL PATH {report['status']}] {output}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
