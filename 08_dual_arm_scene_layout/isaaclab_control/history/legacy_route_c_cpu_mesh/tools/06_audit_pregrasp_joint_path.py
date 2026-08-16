#!/usr/bin/env python3
"""Offline collision audit of the initial-to-PREGRASP joint interpolation.

The Wuji2 hand is held at the generated PREGRASP q20 while the right arm is
linearly interpolated from the calibrated initial q7 to the bounded IK q7.
The script reports table collisions and self-collision pairs that were not
already present at the start.  It never starts Isaac Sim or commands a robot.
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
DEFAULT_CASE = (
    PROJECT_ROOT
    / "06_leap_to_wuji2_final_pipeline/01_cases"
    / "live_scene0000_ashtray_armreachable_candidate0336"
)
ROBOT_ROOT = PROJECT_ROOT / "01_environment/vendor/wuji-description"
ROBOT_URDF = ROBOT_ROOT / "dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
WUJI_PACKAGE_SOURCE = ROBOT_ROOT / "hand2/hand2_beta1/body"
LAYOUT_JSON = PROJECT_ROOT / "08_dual_arm_scene_layout/outputs/manual_layout_calibrated.json"
RIGHT_ARM_NAMES = [f"arm_r_joint_{index}" for index in range(1, 8)]


def active_pairs(geometry_model: pin.GeometryModel, geometry_data: pin.GeometryData) -> set[tuple[str, str]]:
    collisions: set[tuple[str, str]] = set()
    for pair, result in zip(geometry_model.collisionPairs, geometry_data.collisionResults):
        if result.isCollision():
            first = geometry_model.geometryObjects[pair.first].name
            second = geometry_model.geometryObjects[pair.second].name
            collisions.add(tuple(sorted((first, second))))
    return collisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--samples", type=int, default=121)
    parser.add_argument(
        "--include-scene-objects",
        action="store_true",
        help="Also add all six scene object meshes using their SourceZone poses.",
    )
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")

    case_root = args.case_root.resolve()
    ik_path = case_root / "07_arm_execution/pregrasp_read_only_ik.npz"
    waypoints_path = case_root / "06_isaacsim/final_waypoints.npz"
    with np.load(ik_path, allow_pickle=False) as archive:
        if not bool(np.asarray(archive["reachable"]).item()):
            raise RuntimeError("PREGRASP IK has not passed")
        arm_initial = np.asarray(archive["initial_right_arm_q_rad"], dtype=np.float64)
        arm_target = np.asarray(archive["solved_right_arm_q_rad"], dtype=np.float64)
    with np.load(waypoints_path, allow_pickle=False) as archive:
        waypoint_names = np.asarray(archive["waypoint_names"])
        matches = np.flatnonzero(waypoint_names == "pregrasp")
        if matches.size != 1:
            raise RuntimeError("PREGRASP waypoint is ambiguous")
        hand_names = [str(value) for value in archive["finger_joint_names"].tolist()]
        hand_pregrasp = np.asarray(
            archive["waypoint_joint_positions"][0, int(matches[0])], dtype=np.float64
        )
    scene_manifest = json.loads(
        (case_root / "01_input/scene_0000_manifest.json").read_text(encoding="utf-8")
    )

    model = pin.buildModelFromUrdf(str(ROBOT_URDF))
    data = model.createData()
    q = pin.neutral(model)
    arm_q_indices = np.asarray(
        [model.joints[model.getJointId(name)].idx_q for name in RIGHT_ARM_NAMES]
    )
    hand_q_indices = np.asarray(
        [model.joints[model.getJointId(name)].idx_q for name in hand_names]
    )
    q[hand_q_indices] = hand_pregrasp

    with tempfile.TemporaryDirectory(prefix="dgn2_pin_packages_") as temp_root:
        package_alias = Path(temp_root) / "wuji_hand2_description"
        package_alias.symlink_to(WUJI_PACKAGE_SOURCE, target_is_directory=True)
        geometry_model = pin.buildGeomFromUrdf(
            model,
            str(ROBOT_URDF),
            pin.GeometryType.COLLISION,
            package_dirs=[str(ROBOT_ROOT), temp_root],
        )

        layout = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))
        world_from_base = np.asarray(
            layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
            dtype=np.float64,
        ).T
        base_from_world = np.linalg.inv(world_from_base)
        table_position_world = np.asarray(
            layout["transforms"]["table"]["position_world_m"], dtype=np.float64
        )
        table_size = np.asarray(layout["geometry"]["table_size_m"], dtype=np.float64)
        world_from_table = np.eye(4, dtype=np.float64)
        world_from_table[:3, 3] = table_position_world
        base_from_table = base_from_world @ world_from_table
        table = pin.GeometryObject(
            "__layout_table__",
            0,
            hppfcl.Box(*table_size.tolist()),
            pin.SE3(base_from_table[:3, :3], base_from_table[:3, 3]),
        )
        table_geometry_index = geometry_model.addGeometryObject(table)
        scene_geometry_names: set[str] = set()
        if args.include_scene_objects:
            mesh_loader = hppfcl.MeshLoader()
            world_from_source = np.eye(4, dtype=np.float64)
            world_from_source[:3, 3] = np.asarray(
                layout["transforms"]["source_zone"]["position_world_m"], dtype=np.float64
            )
            for record in scene_manifest["objects"]:
                name = f"__scene_object_{int(record['segmentation_id']):03d}__"
                scene_geometry_names.add(name)
                source_from_object = np.asarray(record["pose_world_object"], dtype=np.float64)
                base_from_object = base_from_world @ world_from_source @ source_from_object
                geometry = mesh_loader.load(str(Path(record["visual_mesh"]).resolve()))
                geometry_object = pin.GeometryObject(
                    name,
                    0,
                    geometry,
                    pin.SE3(base_from_object[:3, :3], base_from_object[:3, 3]),
                )
                geometry_model.addGeometryObject(geometry_object)
        geometry_model.addAllCollisionPairs()
        geometry_data = pin.GeometryData(geometry_model)

        def collisions_at(right_arm_q: np.ndarray) -> set[tuple[str, str]]:
            q[arm_q_indices] = right_arm_q
            pin.forwardKinematics(model, data, q)
            pin.updateGeometryPlacements(model, data, geometry_model, geometry_data, q)
            pin.computeCollisions(geometry_model, geometry_data, False)
            return active_pairs(geometry_model, geometry_data)

        baseline = collisions_at(arm_initial)
        new_pair_first_sample: dict[tuple[str, str], int] = {}
        table_pair_first_sample: dict[tuple[str, str], int] = {}
        scene_pair_first_sample: dict[tuple[str, str], int] = {}
        for sample_index, alpha in enumerate(np.linspace(0.0, 1.0, args.samples)):
            right_arm_q = (1.0 - alpha) * arm_initial + alpha * arm_target
            current = collisions_at(right_arm_q)
            for pair in current - baseline:
                new_pair_first_sample.setdefault(pair, sample_index)
            for pair in current:
                if "__layout_table__" in pair:
                    table_pair_first_sample.setdefault(pair, sample_index)
                if any(name in pair for name in scene_geometry_names):
                    scene_pair_first_sample.setdefault(pair, sample_index)

        baseline_without_table = sorted(pair for pair in baseline if "__layout_table__" not in pair)
        new_self = sorted(
            pair for pair in new_pair_first_sample if "__layout_table__" not in pair
        )
        table_pairs = sorted(table_pair_first_sample)
        scene_pairs = sorted(scene_pair_first_sample)
        passed = not new_self and not table_pairs and not scene_pairs
        report = {
            "schema_version": 1,
            "status": "PASS" if passed else "FAIL",
            "scope": "offline straight joint interpolation; hand fixed at PREGRASP; no simulator command",
            "issues_robot_commands": False,
            "samples": args.samples,
            "baseline_self_collision_pairs_ignored": [list(pair) for pair in baseline_without_table],
            "new_self_collision_pairs": [
                {"pair": list(pair), "first_sample": new_pair_first_sample[pair]}
                for pair in new_self
            ],
            "table_collision_pairs": [
                {"pair": list(pair), "first_sample": table_pair_first_sample[pair]}
                for pair in table_pairs
            ],
            "scene_object_collision_pairs": [
                {"pair": list(pair), "first_sample": scene_pair_first_sample[pair]}
                for pair in scene_pairs
            ],
            "next_gate": (
                "slow Isaac Lab dry motion"
                if passed
                else "plan a collision-free waypoint path; do not execute this straight interpolation"
            ),
            "notes": [
                "Pairs already colliding at the initial PREGRASP-hand configuration are listed separately and are not counted as path-created collisions.",
                "This conservative audit does not certify dynamic stability or scene-object clearance.",
                f"table_geometry_index={table_geometry_index}",
            ],
        }
    output_root = case_root / "07_arm_execution"
    report_path = output_root / "pregrasp_joint_path_collision_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[JOINT PATH {report['status']}] samples={args.samples}; no Isaac Sim command")
    print(f"baseline self pairs={len(baseline_without_table)}")
    print(f"new self pairs={len(new_self)}; table pairs={len(table_pairs)}")
    print(f"scene object pairs={len(scene_pairs)}")
    for item in report["new_self_collision_pairs"][:10]:
        print(f"  self sample={item['first_sample']}: {item['pair']}")
    for item in report["table_collision_pairs"][:10]:
        print(f"  table sample={item['first_sample']}: {item['pair']}")
    for item in report["scene_object_collision_pairs"][:10]:
        print(f"  scene sample={item['first_sample']}: {item['pair']}")
    print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()
