#!/usr/bin/env python3
"""Strict LEAP -> Wuji2 coordinate-frame invariant audit.

Convention
----------
T_A_B maps coordinates expressed in frame B into frame A.

Frames
------
W = calibrated layout world
S = SourceZone rigid frame
L = LEAP hand_base_link
H = Wuji2 r_wrist
F = arm_r_link_tf

This script is read-only with respect to retargeting outputs.  It does not run
Isaac, does not recompute q20, and does not alter Kabsch/root alignment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from case_paths import PROJECT_ROOT, active_case_root  # noqa: E402


LAYOUT_JSON = PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
ASSEMBLY_SPEC = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/config/assembly_spec.json"
)
COMBINED_URDF = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
)
LEAP_TIPS = np.asarray([4, 8, 12, 16], dtype=np.int64)
FINGERS = ("thumb", "index", "middle", "ring")


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


def transform_from_xyz_rpy(xyz: list[float], rpy: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_xyz_matrix(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_angle_error_rad(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(cosine))


def transform_error(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "position_error_m": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        "rotation_error_rad": rotation_angle_error_rad(a, b),
        "rotation_error_deg": math.degrees(rotation_angle_error_rad(a, b)),
        "matrix_frobenius": float(np.linalg.norm(a - b)),
    }


def rotation_audit(transform: np.ndarray) -> dict:
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    return {
        "orthogonality_error_fro": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
        "det": float(np.linalg.det(rotation)),
    }


def source_zone_transform() -> np.ndarray:
    layout = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))
    source = layout["transforms"]["source_zone"]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(source["position_world_m"], dtype=np.float64)
    return transform


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def official_dgn2_pose(case_root: Path, candidate_index: int) -> np.ndarray | None:
    path = case_root / "01_input/official_leap_1024.npz"
    if not path.is_file():
        return None
    data = load_npz(path)
    if "rotation_world" not in data or "translation_world" not in data:
        return None
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(data["rotation_world"][candidate_index], dtype=np.float64)
    pose[:3, 3] = np.asarray(data["translation_world"][candidate_index], dtype=np.float64)
    return pose


def urdf_mount_transform() -> dict:
    root = ET.parse(COMBINED_URDF).getroot()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            parent is not None
            and child is not None
            and parent.attrib.get("link") == "arm_r_link_tf"
            and child.attrib.get("link") == "r_wrist"
        ):
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
            return {
                "joint_name": joint.attrib.get("name"),
                "xyz_m": xyz,
                "rpy_rad": rpy,
                "matrix": transform_from_xyz_rpy(xyz, rpy),
            }
    raise RuntimeError("combined URDF missing fixed joint arm_r_link_tf -> r_wrist")


def kabsch_audit(source: np.ndarray, target: np.ndarray) -> dict:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(0)
    target_center = target.mean(0)
    covariance = (source - source_center).T @ (target - target_center)
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return {
        "singular_values": singular_values.tolist(),
        "rank": int(np.linalg.matrix_rank(covariance)),
        "det_R": float(np.linalg.det(rotation)),
        "orthogonality_error_fro": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=active_case_root())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    case_root = args.case_root.resolve()
    root_alignment = load_npz(case_root / "03_root_alignment/root_alignment.npz")
    final = load_npz(case_root / "06_isaacsim/final_waypoints.npz")
    waypoint_names = [str(x) for x in final["waypoint_names"].tolist()]
    grasp_i = waypoint_names.index("grasp")

    T_W_S = source_zone_transform()
    T_S_W = np.linalg.inv(T_W_S)
    T_S_L = np.asarray(final["source_leap_waypoint_pose_world"][0, grasp_i], dtype=np.float64)
    T_S_H = np.asarray(final["waypoint_pose_world"][0, grasp_i], dtype=np.float64)
    T_L_H = np.linalg.inv(T_S_L) @ T_S_H
    T_W_H = T_W_S @ T_S_H

    assembly = json.loads(ASSEMBLY_SPEC.read_text(encoding="utf-8"))
    spec = assembly["mount_transform_parent_to_child"]
    T_F_H_spec = transform_from_xyz_rpy(spec["xyz_m"], spec["rpy_rad"])
    urdf = urdf_mount_transform()
    T_F_H_urdf = urdf["matrix"]
    T_W_F = T_W_H @ np.linalg.inv(T_F_H_spec)

    candidate_index = int(np.asarray(final["source_candidate_index"]).reshape(-1)[0])
    T_W_L = official_dgn2_pose(case_root, candidate_index)
    dgn2_round_trip = {
        "candidate_index": candidate_index,
        "input_pose_available": T_W_L is not None,
    }
    if T_W_L is not None:
        T_S_L_from_world = T_S_W @ T_W_L
        direct_source_error = transform_error(T_W_L, T_S_L)
        world_roundtrip_error = transform_error(T_W_S @ T_S_L, T_W_L)
        if (
            direct_source_error["position_error_m"] < 1.0e-6
            and direct_source_error["rotation_error_rad"] < math.radians(0.1)
            and world_roundtrip_error["position_error_m"] > 1.0e-3
        ):
            dgn2_round_trip.update(
                {
                    "detected_input_frame": "SourceZone",
                    "note": (
                        "official_leap_1024 rotation_world/translation_world are "
                        "legacy-labeled as world but numerically equal T_SourceZone_LEAP "
                        "for this case; no DGN2 world-to-source round-trip is available."
                    ),
                    "legacy_label_T_world_L_directly_equals_case_T_S_L": direct_source_error,
                    "T_W_S_times_case_T_S_L_vs_legacy_labeled_pose": world_roundtrip_error,
                }
            )
        else:
            dgn2_round_trip.update({"detected_input_frame": "World"})
            dgn2_round_trip.update(
                {
                    "T_S_L_equals_case_T_S_L": transform_error(T_S_L_from_world, T_S_L),
                    "T_W_S_times_case_T_S_L_equals_T_W_L": world_roundtrip_error,
                }
            )
    else:
        dgn2_round_trip["note"] = "No original T_W_L available; case pose is treated as T_SourceZone_LEAP."

    leap_local = np.asarray(root_alignment["raw_leap_grasp_mediapipe21_m"], dtype=np.float64)[LEAP_TIPS]
    p_S_leap = transform_points(T_S_L, leap_local)
    p_H_wuji = np.asarray(root_alignment["wuji2_four_tip_local_m"], dtype=np.float64)
    p_S_wuji = transform_points(T_S_H, p_H_wuji)
    tip_errors_mm = np.linalg.norm(p_S_wuji - p_S_leap, axis=1) * 1000.0

    p_W_tip_from_H = transform_points(T_W_H, p_H_wuji)
    p_W_tip_from_S = transform_points(T_W_S, p_S_wuji)
    world_tip_delta_m = np.linalg.norm(p_W_tip_from_H - p_W_tip_from_S, axis=1)

    axis_audit = {}
    if "wuji2_semantic_palm_approach_axis_source" in final:
        axis_source = np.asarray(final["wuji2_semantic_palm_approach_axis_source"], dtype=np.float64)
        axis_world_expected = T_W_S[:3, :3] @ axis_source
        axis_world_expected /= np.linalg.norm(axis_world_expected)
        axis_world = np.asarray(final["wuji2_semantic_palm_approach_axis_world"], dtype=np.float64)
        axis_world /= np.linalg.norm(axis_world)
        axis_audit = {
            "source_field_present": True,
            "world_from_source_rotation_error_norm": float(np.linalg.norm(axis_world_expected - axis_world)),
            "axis_source_norm": float(np.linalg.norm(axis_source)),
            "axis_world_norm": float(np.linalg.norm(axis_world)),
        }
    else:
        axis_audit = {
            "source_field_present": False,
            "note": "Legacy final_waypoints lacks wuji2_semantic_palm_approach_axis_source; generator has been patched for new outputs.",
        }

    rotations = {
        "T_W_S": rotation_audit(T_W_S),
        "T_S_L": rotation_audit(T_S_L),
        "T_S_H": rotation_audit(T_S_H),
        "T_L_H": rotation_audit(T_L_H),
        "T_F_H_spec": rotation_audit(T_F_H_spec),
        "T_F_H_urdf": rotation_audit(T_F_H_urdf),
        "T_W_H": rotation_audit(T_W_H),
        "T_W_F": rotation_audit(T_W_F),
    }
    if T_W_L is not None:
        rotations["T_W_L"] = rotation_audit(T_W_L)

    audit = {
        "schema_version": 1,
        "case_root": str(case_root),
        "coordinate_convention": "T_A_B maps coordinates from frame B into frame A",
        "frames": {
            "W": "layout world",
            "S": "SourceZone rigid frame",
            "L": "LEAP hand_base_link",
            "H": "Wuji2 r_wrist",
            "F": "arm_r_link_tf",
        },
        "metadata_present": {
            "waypoint_pose_frame": str(np.asarray(final["waypoint_pose_frame"]).item())
            if "waypoint_pose_frame" in final
            else "MISSING_LEGACY_OUTPUT",
            "coordinate_convention": str(np.asarray(final["coordinate_convention"]).item())
            if "coordinate_convention" in final
            else "MISSING_LEGACY_OUTPUT",
        },
        "invariants": {
            "T_W_S_times_T_S_W_identity_fro": float(np.linalg.norm(T_W_S @ T_S_W - np.eye(4))),
            "rotations": rotations,
            "dgn2_round_trip": dgn2_round_trip,
            "leap_tip_transform_source_error_m": float(
                np.linalg.norm(p_S_leap - np.asarray(root_alignment["leap_four_tip_world_m"], dtype=np.float64))
            ),
            "kabsch": kabsch_audit(p_H_wuji, p_S_leap),
            "tip_alignment": {
                finger: {
                    "error_mm": float(error),
                    "p_S_leap": p_S_leap[i].tolist(),
                    "p_S_wuji": p_S_wuji[i].tolist(),
                }
                for i, (finger, error) in enumerate(zip(FINGERS, tip_errors_mm))
            },
            "tip_alignment_rms_mm": float(np.sqrt(np.mean(tip_errors_mm ** 2))),
            "tip_alignment_max_mm": float(np.max(tip_errors_mm)),
            "bridge_T_L_H": {
                "T_S_L_times_T_L_H_equals_T_S_H": transform_error(T_S_L @ T_L_H, T_S_H),
            },
            "T_F_H_spec_vs_urdf": {
                "spec_xyz_m": spec["xyz_m"],
                "spec_rpy_rad": spec["rpy_rad"],
                "urdf_joint_name": urdf["joint_name"],
                "urdf_xyz_m": urdf["xyz_m"],
                "urdf_rpy_rad": urdf["rpy_rad"],
                "error": transform_error(T_F_H_spec, T_F_H_urdf),
            },
            "final_arm_target": {
                "T_W_F_times_T_F_H_equals_T_W_H": transform_error(T_W_F @ T_F_H_spec, T_W_H),
            },
            "world_tip_consistency": {
                finger: {"error_m": float(error)}
                for finger, error in zip(FINGERS, world_tip_delta_m)
            },
            "world_tip_consistency_max_m": float(np.max(world_tip_delta_m)),
            "approach_axis": axis_audit,
            "mediapipe_rotation_scope": (
                "x=0 y=180 z=-90 belongs only to Retargeter local keypoint preprocessing; "
                "it is not multiplied into T_S_L, T_S_H, T_W_H, or T_W_F."
            ),
            "retarget_offset_scope": (
                "wrist_offset_cm/thumb_offset_cm are in MediaPipe-normalized retarget-local frame "
                "after mediapipe_rotation, not W/S/L/H/F frames."
            ),
        },
    }

    output = args.output
    if output is None:
        output = case_root / "08_audit/coordinate_frame_audit.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[AUDIT] {output}")
    print(f"T_W_S @ T_S_W identity fro = {audit['invariants']['T_W_S_times_T_S_W_identity_fro']:.3e}")
    print(f"Kabsch RMS/MAX mm = {audit['invariants']['tip_alignment_rms_mm']:.3f} / {audit['invariants']['tip_alignment_max_mm']:.3f}")
    print(f"T_F_H spec-vs-URDF fro = {audit['invariants']['T_F_H_spec_vs_urdf']['error']['matrix_frobenius']:.3e}")
    print(f"world tip consistency max m = {audit['invariants']['world_tip_consistency_max_m']:.3e}")


if __name__ == "__main__":
    main()
