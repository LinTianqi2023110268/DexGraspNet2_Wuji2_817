"""Generate Force Drive natural-frequency static-test configs.

This is phase B of the static-stability audit:

- the assembled mass-fixed USD and initial pose stay unchanged;
- only J1-J7 drive type is switched to Force Drive;
- effort limits remain the robot's own 130/70/12 N*m limits;
- K/D are computed at runtime from the current generalized mass matrix:
  K = Mii * f^2, D = 2 * zeta * Mii * f.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE_CONFIG = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/history/config/force_nf_scan_base.json"
OUTPUT_DIR = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/history/config/force_nf_scan"

SCAN_STEPS = [8, 12, 16, 20, 25, 30, 35, 40]


def build_groups(natural_frequency: float) -> list[dict]:
    return [
        {
            "name": "right_arm_j1",
            "joint_names_expr": ["arm_r_joint_1"],
            "natural_frequency_rad_s": natural_frequency,
            "damping_ratio": 1.0,
        },
        {
            "name": "right_arm_j2_j4",
            "joint_names_expr": ["arm_r_joint_[2-4]"],
            "natural_frequency_rad_s": natural_frequency,
            "damping_ratio": 1.0,
        },
        {
            "name": "right_arm_j5_j7",
            "joint_names_expr": ["arm_r_joint_[5-7]"],
            "natural_frequency_rad_s": natural_frequency,
            "damping_ratio": 1.0,
        },
    ]


def main() -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for step in SCAN_STEPS:
        config = dict(base)
        config["purpose"] = f"Force Drive natural-frequency scan nf{step:02d}; zeta=1; no IK"
        config["right_arm_force_natural_frequency_groups"] = build_groups(float(step))
        config["output_directory"] = (
            f"08_dual_arm_scene_layout/isaaclab_control/outputs/force_nf_scan/nf{step:02d}"
        )
        path = OUTPUT_DIR / f"nf{step:02d}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
