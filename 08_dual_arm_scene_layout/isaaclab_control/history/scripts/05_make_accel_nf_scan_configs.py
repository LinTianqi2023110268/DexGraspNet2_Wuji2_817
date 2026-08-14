"""Create Acceleration Drive natural-frequency static-test configs.

This only prepares JSON files.  It does not run Isaac Lab.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
BASE_CONFIG = CONTROL_ROOT / "history/config/accel_nf_scan_base.json"
OUTPUT_DIR = CONTROL_ROOT / "history/config/accel_nf_scan"


SCAN_STEPS = [
    {"name": "nf08", "j1": 8.0, "j2_j4": 8.0, "j5_j7": 8.0},
    {"name": "nf12", "j1": 12.0, "j2_j4": 12.0, "j5_j7": 12.0},
    {"name": "nf16", "j1": 16.0, "j2_j4": 16.0, "j5_j7": 16.0},
    {"name": "nf20", "j1": 20.0, "j2_j4": 20.0, "j5_j7": 20.0},
    {"name": "nf25", "j1": 25.0, "j2_j4": 25.0, "j5_j7": 25.0},
    {"name": "nf30", "j1": 30.0, "j2_j4": 30.0, "j5_j7": 30.0},
    {"name": "nf35", "j1": 35.0, "j2_j4": 35.0, "j5_j7": 35.0},
    {"name": "nf40", "j1": 40.0, "j2_j4": 40.0, "j5_j7": 40.0},
]


def natural_frequency_groups(step: dict) -> list[dict]:
    return [
        {
            "name": "right_arm_j1",
            "joint_names_expr": ["arm_r_joint_1"],
            "natural_frequency_rad_s": step["j1"],
            "damping_ratio": 1.0,
        },
        {
            "name": "right_arm_j2_j4",
            "joint_names_expr": ["arm_r_joint_[2-4]"],
            "natural_frequency_rad_s": step["j2_j4"],
            "damping_ratio": 1.0,
        },
        {
            "name": "right_arm_j5_j7",
            "joint_names_expr": ["arm_r_joint_[5-7]"],
            "natural_frequency_rad_s": step["j5_j7"],
            "damping_ratio": 1.0,
        },
    ]


def main() -> int:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for step in SCAN_STEPS:
        config = deepcopy(base)
        config["purpose"] = f"Acceleration Drive natural-frequency scan {step['name']}; zeta=1; no IK"
        config["right_arm_natural_frequency_groups"] = natural_frequency_groups(step)
        config["output_directory"] = (
            "08_dual_arm_scene_layout/isaaclab_control/outputs/"
            f"accel_nf_scan/{step['name']}"
        )
        path = OUTPUT_DIR / f"{step['name']}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        written.append(path)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
