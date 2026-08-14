#!/usr/bin/env python3
"""Stage 02: solve only Wuji2 root 6D from four GRASP fingertips."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from case_paths import PROJECT_ROOT, SHARED_ROOT, active_case_root  # noqa: E402

CASE_ROOT = active_case_root()
SOURCE = CASE_ROOT / "02_retargeting/grasp_official.npz"
WUJI_URDF = PROJECT_ROOT / "01_environment/vendor/wuji-description/hand2/hand2_beta1/body/urdf/right.urdf"
OUTPUT = CASE_ROOT / "03_root_alignment/root_alignment.npz"
REPORT = CASE_ROOT / "03_root_alignment/root_alignment_report.json"
sys.path.insert(0, str(SHARED_ROOT / "lib"))
from common_kinematics import UrdfKinematicModel  # noqa: E402


FINGERS = ("thumb", "index", "middle", "ring")
LEAP_TIPS = np.asarray([4, 8, 12, 16], dtype=np.int64)
WUJI_TIPS = (
    "r_thumb_tip",
    "r_index_finger_tip",
    "r_middle_finger_tip",
    "r_ring_finger_tip",
)


def world_points(pose, points):
    return np.asarray(points, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]


def rigid_fit(source, target):
    source, target = np.asarray(source), np.asarray(target)
    source_center, target_center = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = target_center - rotation @ source_center
    return pose


def main() -> None:
    for path in (SOURCE, WUJI_URDF):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(SOURCE, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    names = [str(value) for value in source["wuji2_joint_names"].tolist()]
    q20 = np.asarray(source["wuji2_q20_grasp"], dtype=np.float64)
    leap_pose = np.asarray(source["source_leap_pose_world"], dtype=np.float64)
    leap_tip_world = world_points(
        leap_pose,
        np.asarray(source["raw_leap_grasp_mediapipe21_m"], dtype=np.float64)[LEAP_TIPS],
    )
    model = UrdfKinematicModel(WUJI_URDF, "r_wrist")
    fk = model.forward_kinematics(dict(zip(names, q20)))
    wuji_tip_local = np.asarray([fk[link][:3, 3] for link in WUJI_TIPS])
    root_pose = rigid_fit(wuji_tip_local, leap_tip_world)
    wuji_tip_world = world_points(root_pose, wuji_tip_local)
    errors_mm = 1000.0 * np.linalg.norm(wuji_tip_world - leap_tip_world, axis=1)
    bridge = np.linalg.inv(leap_pose) @ root_pose

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        wuji2_q20_grasp=q20.astype(np.float32),
        wuji2_joint_names=np.asarray(names),
        leap_q16_grasp=source["leap_q16_grasp"],
        leap_q16_squeeze=source["leap_q16_squeeze"],
        leap_joint_names=source["leap_joint_names"],
        raw_leap_grasp_mediapipe21_m=source["raw_leap_grasp_mediapipe21_m"],
        raw_leap_squeeze_mediapipe21_m=source["raw_leap_squeeze_mediapipe21_m"],
        source_leap_pose_world=leap_pose.astype(np.float32),
        aligned_wuji2_root_pose_world=root_pose.astype(np.float32),
        T_LEAP_hand_base_link_from_WUJI2_r_wrist=bridge.astype(np.float32),
        leap_four_tip_world_m=leap_tip_world.astype(np.float32),
        wuji2_four_tip_local_m=wuji_tip_local.astype(np.float32),
        wuji2_four_tip_world_m=wuji_tip_world.astype(np.float32),
        four_tip_error_mm=errors_mm.astype(np.float32),
        finger_names=np.asarray(FINGERS),
        source_candidate_index=source["source_candidate_index"],
        target_segmentation_id=source["target_segmentation_id"],
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "equal-weight four-tip Kabsch; q20 frozen",
        "q20_unchanged": True,
        "tip_error_mm": dict(zip(FINGERS, errors_mm.tolist())),
        "rms_mm": float(np.sqrt(np.mean(errors_mm ** 2))),
        "max_mm": float(np.max(errors_mm)),
        "output": str(OUTPUT),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] root-only four-tip RMS={report['rms_mm']:.3f} mm")
    print(f"[CASE] {CASE_ROOT.name}")
    print(f"[OK] {OUTPUT}")


if __name__ == "__main__":
    main()
