#!/usr/bin/env python3
"""Stage 01: map the official LEAP GRASP to Wuji2 q20.

Only the official wuji-retargeting optimizer and the reviewed four-finger
correspondence are used. The output contains no Wuji2 world/root pose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from case_paths import PROJECT_ROOT, SHARED_ROOT, active_case_root  # noqa: E402

CASE_ROOT = active_case_root()
CONFIG = SHARED_ROOT / "config/grasp_retarget.yaml"
LEAP_INPUT = CASE_ROOT / "01_input/leap_official_waypoints.npz"
OUTPUT = CASE_ROOT / "02_retargeting/grasp_official.npz"
REPORT = CASE_ROOT / "02_retargeting/grasp_official_report.json"
PINKY_POLICY = SHARED_ROOT / "config/pinky_ring_coupling.json"
WUJI_URDF = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/hand2/hand2_beta1/body/urdf/right.urdf"
)
sys.path.insert(0, str(SHARED_ROOT / "lib"))

from common_kinematics import UrdfKinematicModel  # noqa: E402
from four_finger_official_adapter import install_four_finger_optimizer  # noqa: E402
from leap_mediapipe import leap_qpos_to_mediapipe21  # noqa: E402
from pinky_ring_coupling import (  # noqa: E402
    apply_pinky_ring_coupling,
    load_pinky_policy,
)
from wuji_retargeting import Retargeter  # noqa: E402


CONSUMER_ORDER = [
    "r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip",
    "r_index_finger_mcp_flex", "r_index_finger_mcp_abd", "r_index_finger_pip", "r_index_finger_dip",
    "r_middle_finger_mcp_flex", "r_middle_finger_mcp_abd", "r_middle_finger_pip", "r_middle_finger_dip",
    "r_ring_finger_mcp_flex", "r_ring_finger_mcp_abd", "r_ring_finger_pip", "r_ring_finger_dip",
    "r_pinky_mcp_flex", "r_pinky_mcp_abd", "r_pinky_pip", "r_pinky_dip",
]


def main() -> None:
    for path in (CONFIG, LEAP_INPUT, PINKY_POLICY, WUJI_URDF):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["__yaml_dir"] = str(CONFIG.parent.resolve())
    with np.load(LEAP_INPUT, allow_pickle=False) as archive:
        leap = {key: archive[key] for key in archive.files}

    stages = [str(value) for value in leap["waypoint_names"].tolist()]
    leap_names = [str(value) for value in leap["finger_joint_names"].tolist()]
    grasp_i, squeeze_i = stages.index("grasp"), stages.index("squeeze")
    q16_grasp = np.asarray(leap["waypoint_joint_positions"][0, grasp_i], dtype=np.float64)
    q16_squeeze = np.asarray(leap["waypoint_joint_positions"][0, squeeze_i], dtype=np.float64)
    raw_grasp, _ = leap_qpos_to_mediapipe21(dict(zip(leap_names, q16_grasp)))
    raw_squeeze, _ = leap_qpos_to_mediapipe21(dict(zip(leap_names, q16_squeeze)))

    retargeter = Retargeter(config, hand_side="right")
    install_four_finger_optimizer(retargeter, config)
    retargeter.reset()
    q_optimizer, verbose = retargeter.retarget_verbose(raw_grasp, apply_filter=False)
    optimizer_names = list(retargeter.optimizer.robot.dof_joint_names)
    by_name = dict(zip(optimizer_names, np.asarray(q_optimizer, dtype=np.float64)))
    q20_raw = np.asarray([by_name[name] for name in CONSUMER_ORDER], dtype=np.float64)
    if not np.allclose(q20_raw[16:20], 0.0, atol=1.0e-7):
        raise RuntimeError("inactive Wuji2 pinky moved")
    model = UrdfKinematicModel(WUJI_URDF, "r_wrist")
    limits = np.asarray(
        [model.joint_limits(name) for name in CONSUMER_ORDER], dtype=np.float64
    )
    q20, pinky_audit = apply_pinky_ring_coupling(
        q20_raw,
        CONSUMER_ORDER,
        limits,
        load_pinky_policy(PINKY_POLICY),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        wuji2_q20_grasp=q20.astype(np.float32),
        wuji2_joint_names=np.asarray(CONSUMER_ORDER),
        leap_q16_grasp=q16_grasp.astype(np.float32),
        leap_q16_squeeze=q16_squeeze.astype(np.float32),
        leap_joint_names=np.asarray(leap_names),
        raw_leap_grasp_mediapipe21_m=raw_grasp.astype(np.float32),
        raw_leap_squeeze_mediapipe21_m=raw_squeeze.astype(np.float32),
        source_leap_pose_world=np.asarray(
            leap["waypoint_pose_world"][0, grasp_i], dtype=np.float32
        ),
        source_candidate_index=np.asarray(leap["source_candidate_index"], dtype=np.int64),
        target_segmentation_id=np.asarray(leap["target_segmentation_id"], dtype=np.int64),
        optimization_cost=np.asarray(float(verbose["cost"]), dtype=np.float64),
    )
    REPORT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "method": "official wuji-retargeting GRASP with reviewed four-finger correspondence",
                "filter_enabled": False,
                "pinky_ring_coupled": True,
                "pinky_policy_config": str(PINKY_POLICY),
                "pinky_policy_audit": pinky_audit,
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] official GRASP q20; cost={float(verbose['cost']):.6f}")
    print(f"[CASE] {CASE_ROOT.name}")
    print(f"[OK] {OUTPUT}")


if __name__ == "__main__":
    main()
