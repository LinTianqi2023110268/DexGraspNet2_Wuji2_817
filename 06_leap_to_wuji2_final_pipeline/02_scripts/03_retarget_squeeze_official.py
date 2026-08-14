#!/usr/bin/env python3
"""Retry only SQUEEZE with official Wuji retargeting YAML parameters.

The accepted GRASP q20 and aligned Wuji2 root pose are immutable. The new
SQUEEZE endpoint is produced by the official analytical optimizer, not by a
custom IK or post-refinement step.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from case_paths import PROJECT_ROOT, SHARED_ROOT, active_case_root  # noqa: E402

CASE_ROOT = active_case_root()
ALIGNED = CASE_ROOT / "03_root_alignment/root_alignment.npz"
BASE_CONFIG = SHARED_ROOT / "config/grasp_retarget.yaml"
OVERRIDES = SHARED_ROOT / "config/squeeze_retry_overrides.yaml"
LEAP_WAYPOINTS = CASE_ROOT / "01_input/leap_official_waypoints.npz"
WUJI_URDF = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/hand2/hand2_beta1/body/urdf/right.urdf"
)
OUTPUT = CASE_ROOT / "04_squeeze/squeeze_official.npz"
REPORT = CASE_ROOT / "04_squeeze/squeeze_official_report.json"
PINKY_POLICY = SHARED_ROOT / "config/pinky_ring_coupling.json"
sys.path.insert(0, str(SHARED_ROOT / "lib"))

from common_kinematics import UrdfKinematicModel  # noqa: E402
from four_finger_official_adapter import install_four_finger_optimizer  # noqa: E402
from leap_mediapipe import leap_qpos_to_mediapipe21  # noqa: E402
from pinky_ring_coupling import (  # noqa: E402
    apply_pinky_ring_coupling,
    load_pinky_policy,
)
from wuji_retargeting import Retargeter  # noqa: E402


FINGERS = ("thumb", "index", "middle", "ring")
LEAP_TIP_INDICES = np.asarray([4, 8, 12, 16], dtype=np.int64)
WUJI_TIP_LINKS = (
    "r_thumb_tip",
    "r_index_finger_tip",
    "r_middle_finger_tip",
    "r_ring_finger_tip",
)
INTERPOLATION_FRAMES = 41


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def world_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]


def tip_local(model, names, q20):
    fk = model.forward_kinematics(dict(zip(names, np.asarray(q20, dtype=np.float64))))
    return np.asarray([fk[link][:3, 3] for link in WUJI_TIP_LINKS], dtype=np.float64)


def main() -> None:
    for path in (
        ALIGNED,
        BASE_CONFIG,
        OVERRIDES,
        LEAP_WAYPOINTS,
        WUJI_URDF,
        PINKY_POLICY,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(ALIGNED, allow_pickle=False) as archive:
        aligned = {key: archive[key] for key in archive.files}
    with np.load(LEAP_WAYPOINTS, allow_pickle=False) as archive:
        leap = {key: archive[key] for key in archive.files}

    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    override = yaml.safe_load(OVERRIDES.read_text(encoding="utf-8"))["retarget"]
    apply_filter = bool(override.pop("apply_filter_at_static_endpoint"))
    config["retarget"].update(override)
    config["__yaml_dir"] = str(BASE_CONFIG.parent.resolve())

    stage_names = [str(value) for value in leap["waypoint_names"].tolist()]
    leap_names = [str(value) for value in leap["finger_joint_names"].tolist()]
    q16_grasp = np.asarray(
        leap["waypoint_joint_positions"][0, stage_names.index("grasp")], dtype=np.float64
    )
    q16_squeeze = np.asarray(
        leap["waypoint_joint_positions"][0, stage_names.index("squeeze")], dtype=np.float64
    )
    raw_squeeze, _ = leap_qpos_to_mediapipe21(dict(zip(leap_names, q16_squeeze)))

    joint_names = [str(value) for value in aligned["wuji2_joint_names"].tolist()]
    q20_grasp = np.asarray(aligned["wuji2_q20_grasp"], dtype=np.float64).copy()
    root_pose = np.asarray(aligned["aligned_wuji2_root_pose_world"], dtype=np.float64).copy()
    leap_pose = np.asarray(aligned["source_leap_pose_world"], dtype=np.float64)

    retargeter = Retargeter(config, hand_side="right")
    install_four_finger_optimizer(retargeter, config)
    retargeter.reset()
    q_optimizer, verbose = retargeter.retarget_verbose(raw_squeeze, apply_filter=apply_filter)
    optimizer_names = list(retargeter.optimizer.robot.dof_joint_names)
    q_by_name = dict(zip(optimizer_names, np.asarray(q_optimizer, dtype=np.float64)))
    q20_squeeze_raw = np.asarray(
        [q_by_name[name] for name in joint_names], dtype=np.float64
    )
    if not np.allclose(q20_squeeze_raw[16:20], 0.0, atol=1.0e-7):
        raise RuntimeError("inactive Wuji2 pinky moved inside the official solver")

    model = UrdfKinematicModel(WUJI_URDF, "r_wrist")
    limits = np.asarray([model.joint_limits(name) for name in joint_names], dtype=np.float64)
    q20_squeeze, pinky_audit = apply_pinky_ring_coupling(
        q20_squeeze_raw,
        joint_names,
        limits,
        load_pinky_policy(PINKY_POLICY),
    )
    margin = np.minimum(q20_squeeze - limits[:, 0], limits[:, 1] - q20_squeeze)
    if float(np.min(margin)) < -1.0e-7:
        raise RuntimeError(f"SQUEEZE violates official limits by {-float(np.min(margin))} rad")

    alpha = np.linspace(0.0, 1.0, INTERPOLATION_FRAMES, dtype=np.float64)
    q20_path = q20_grasp[None] + alpha[:, None] * (q20_squeeze - q20_grasp)[None]
    q16_path = q16_grasp[None] + alpha[:, None] * (q16_squeeze - q16_grasp)[None]
    leap_tip_world = []
    wuji_tip_world = []
    for q16, q20 in zip(q16_path, q20_path):
        raw, _ = leap_qpos_to_mediapipe21(dict(zip(leap_names, q16)))
        leap_tip_world.append(world_points(leap_pose, raw[LEAP_TIP_INDICES]))
        wuji_tip_world.append(world_points(root_pose, tip_local(model, joint_names, q20)))
    leap_tip_world = np.asarray(leap_tip_world)
    wuji_tip_world = np.asarray(wuji_tip_world)
    error_mm = 1000.0 * np.linalg.norm(wuji_tip_world - leap_tip_world, axis=2)

    # The previous accepted GRASP and root contracts must remain exact.
    if not np.array_equal(q20_grasp.astype(np.float32), aligned["wuji2_q20_grasp"]):
        raise RuntimeError("accepted GRASP q20 changed")
    if not np.array_equal(root_pose.astype(np.float32), aligned["aligned_wuji2_root_pose_world"]):
        raise RuntimeError("accepted root 6D changed")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        wuji2_q20_grasp=q20_grasp.astype(np.float32),
        wuji2_q20_squeeze=q20_squeeze.astype(np.float32),
        wuji2_q20_path=q20_path.astype(np.float32),
        wuji2_joint_names=np.asarray(joint_names),
        leap_q16_grasp=q16_grasp.astype(np.float32),
        leap_q16_squeeze=q16_squeeze.astype(np.float32),
        leap_q16_path=q16_path.astype(np.float32),
        leap_joint_names=np.asarray(leap_names),
        path_alpha=alpha.astype(np.float32),
        fixed_wuji2_root_pose_world=root_pose.astype(np.float32),
        source_leap_pose_world=leap_pose.astype(np.float32),
        leap_four_tip_world_m=leap_tip_world.astype(np.float32),
        wuji2_four_tip_world_m=wuji_tip_world.astype(np.float32),
        four_tip_error_mm=error_mm.astype(np.float32),
        finger_names=np.asarray(FINGERS),
        transformed_squeeze_keypoints_m=np.asarray(verbose["mediapipe_kp"], dtype=np.float32),
        optimization_cost=np.asarray(float(verbose["cost"]), dtype=np.float64),
    )
    report = {
        "schema_version": 1,
        "method": "official wuji-retargeting static SQUEEZE endpoint; no custom IK",
        "immutable_contract": {
            "grasp_q20_unchanged": True,
            "root_6d_unchanged": True,
            "pinky_ring_coupled": True,
        },
        "base_config": str(BASE_CONFIG),
        "base_config_sha256": sha256(BASE_CONFIG),
        "override_config": str(OVERRIDES),
        "override_config_sha256": sha256(OVERRIDES),
        "effective_overrides": override,
        "static_endpoint_filter_enabled": apply_filter,
        "interpolation_frames": INTERPOLATION_FRAMES,
        "squeeze_tip_error_mm": {
            finger: float(error_mm[-1, index]) for index, finger in enumerate(FINGERS)
        },
        "squeeze_rms_mm": float(np.sqrt(np.mean(error_mm[-1] ** 2))),
        "squeeze_max_mm": float(np.max(error_mm[-1])),
        "trajectory_rms_mm": float(np.sqrt(np.mean(error_mm ** 2))),
        "maximum_joint_delta_from_grasp_rad": float(
            np.max(np.abs(q20_squeeze - q20_grasp))
        ),
        "maximum_interpolated_step_rad": float(np.max(np.abs(np.diff(q20_path, axis=0)))),
        "minimum_official_joint_limit_margin_rad": float(np.min(margin)),
        "optimization_cost": float(verbose["cost"]),
        "pinky_policy_config": str(PINKY_POLICY),
        "pinky_policy_audit": pinky_audit,
        "output": str(OUTPUT),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] output={OUTPUT}")
    print(f"[OK] report={REPORT}")
    print(f"[CASE] {CASE_ROOT.name}")
    print("SQUEEZE four-tip errors (mm)")
    for index, finger in enumerate(FINGERS):
        print(f"  {finger:6s}: {error_mm[-1,index]:8.3f}")
    print(f"  RMS   : {np.sqrt(np.mean(error_mm[-1]**2)):8.3f}")
    print(f"  MAX   : {np.max(error_mm[-1]):8.3f}")


if __name__ == "__main__":
    main()
