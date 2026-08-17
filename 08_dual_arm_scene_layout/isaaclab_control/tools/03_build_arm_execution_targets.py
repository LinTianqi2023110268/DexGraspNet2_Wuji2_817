#!/usr/bin/env python3
"""Convert the selected Wuji2 hand waypoints into dual-arm flange targets.

Input coordinate contract
-------------------------
``final_waypoints.npz`` stores ``T_SourceZone_r_wrist``.  The calibrated scene
stores the SourceZone origin in the layout world.  The assembly specification
stores the fixed ``T_flange_r_wrist`` connection.  Therefore this script uses

    T_world_flange = T_world_SourceZone @ T_SourceZone_r_wrist
                      @ inverse(T_flange_r_wrist)

No simulator is started and no robot command is issued.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASE = (
    PROJECT_ROOT
    / "06_leap_to_wuji2_final_pipeline/01_cases"
    / "live_scene0000_ashtray_isaaclab_candidate0274"
)
ASSEMBLY_SPEC = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2"
    / "config/assembly_spec.json"
)
LAYOUT_JSON = (
    PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
)


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def euler_xyz_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def quaternion_xyzw_from_matrix(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat.tolist()


def transform_from_xyz_rpy(xyz: list[float], rpy: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_xyz_matrix(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def source_zone_transform(layout: dict) -> np.ndarray:
    source = layout["transforms"]["source_zone"]
    transform = np.eye(4, dtype=np.float64)
    # The zone has no authored rotation.  Its recorded matrix also contains the
    # display scale, which must never be applied to a pose transform.
    transform[:3, 3] = np.asarray(source["position_world_m"], dtype=np.float64)
    return transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--source-yaw-deg", type=float, default=0.0)
    parser.add_argument("--source-offset-x-m", type=float, default=0.0)
    parser.add_argument("--source-offset-y-m", type=float, default=0.0)
    args = parser.parse_args()

    case_root = args.case_root.resolve()
    waypoint_path = case_root / "06_isaacsim/final_waypoints.npz"
    output_root = case_root / "07_arm_execution"
    output_root.mkdir(parents=True, exist_ok=True)

    assembly = json.loads(ASSEMBLY_SPEC.read_text(encoding="utf-8"))
    layout = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))
    mount = assembly["mount_transform_parent_to_child"]
    world_from_source = source_zone_transform(layout)
    world_from_source[:3, :3] = rotation_z(math.radians(args.source_yaw_deg))
    world_from_source[:2, 3] += np.asarray(
        [args.source_offset_x_m, args.source_offset_y_m], dtype=np.float64
    )
    flange_from_wrist = transform_from_xyz_rpy(mount["xyz_m"], mount["rpy_rad"])

    with np.load(waypoint_path, allow_pickle=False) as archive:
        waypoint_names = np.asarray(archive["waypoint_names"])
        source_from_wrist = np.asarray(archive["waypoint_pose_world"][0], dtype=np.float64)

    world_from_wrist = world_from_source[None] @ source_from_wrist
    world_from_flange = world_from_wrist @ np.linalg.inv(flange_from_wrist)[None]

    output_npz = output_root / "arm_flange_targets.npz"
    np.savez_compressed(
        output_npz,
        waypoint_names=waypoint_names,
        world_from_source_zone=world_from_source,
        flange_from_wuji2_wrist=flange_from_wrist,
        source_zone_from_wuji2_wrist=source_from_wrist,
        world_from_wuji2_wrist=world_from_wrist,
        world_from_right_flange=world_from_flange,
    )

    records = []
    for name, wrist, flange in zip(waypoint_names.tolist(), world_from_wrist, world_from_flange):
        records.append(
            {
                "stage": str(name),
                "wuji2_wrist_position_world_m": wrist[:3, 3].tolist(),
                "right_flange_position_world_m": flange[:3, 3].tolist(),
                "right_flange_quaternion_xyzw": quaternion_xyzw_from_matrix(flange[:3, :3]),
            }
        )
    report = {
        "schema_version": 1,
        "status": "READY_FOR_READ_ONLY_IK_AUDIT",
        "case_root": str(case_root),
        "source_waypoints": str(waypoint_path),
        "assembly_spec": str(ASSEMBLY_SPEC),
        "layout_calibration": str(LAYOUT_JSON),
        "formula": "T_world_flange = T_world_SourceZone @ T_SourceZone_r_wrist @ inverse(T_flange_r_wrist)",
        "scene_placement": {
            "source_yaw_deg": args.source_yaw_deg,
            "source_offset_xy_m": [args.source_offset_x_m, args.source_offset_y_m],
            "world_from_source_zone": world_from_source.tolist(),
        },
        "issues_robot_commands": False,
        "waypoints": records,
        "output_npz": str(output_npz),
    }
    report_path = output_root / "arm_flange_targets.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[PASS] built {len(records)} flange targets without starting Isaac Sim")
    print(f"[OUTPUT] {output_npz}")
    print(f"[AUDIT]  {report_path}")
    for record in records:
        print(f"  {record['stage']:<8} flange={np.round(record['right_flange_position_world_m'], 6)}")


if __name__ == "__main__":
    main()
