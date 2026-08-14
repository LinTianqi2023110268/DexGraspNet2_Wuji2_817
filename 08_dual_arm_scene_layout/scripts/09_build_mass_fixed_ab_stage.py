#!/usr/bin/env python3
"""Clone the calibrated scene and switch only its robot reference.

This produces the mass-tree A/B scene.  No table, layout, camera, physics-scene,
joint state, or solver attribute is changed.
"""

from pathlib import Path

from pxr import Usd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_STAGE = PROJECT_ROOT / "08_dual_arm_scene_layout/scenes/manual_layout_calibrated.usda"
OUTPUT_STAGE = PROJECT_ROOT / "08_dual_arm_scene_layout/scenes/manual_layout_calibrated_mass_fixed.usda"
ROBOT_PRIM = "/World/Layout/DualArmMount/DualArm"
MASS_FIXED_USD = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/usd/dual_arm_right_wuji2_mass_fixed.usda"
)


def build() -> None:
    if not SOURCE_STAGE.is_file() or not MASS_FIXED_USD.is_file():
        raise FileNotFoundError(f"source={SOURCE_STAGE}, robot={MASS_FIXED_USD}")
    source = Usd.Stage.Open(str(SOURCE_STAGE))
    if source is None:
        raise RuntimeError(f"Cannot open {SOURCE_STAGE}")
    source.GetRootLayer().Export(str(OUTPUT_STAGE))

    output = Usd.Stage.Open(str(OUTPUT_STAGE))
    robot = output.GetPrimAtPath(ROBOT_PRIM)
    if not robot.IsValid():
        raise RuntimeError(f"Missing robot prim: {ROBOT_PRIM}")
    relative_robot = MASS_FIXED_USD.relative_to(PROJECT_ROOT)
    relative_from_scene = Path("../..") / relative_robot
    robot.GetReferences().ClearReferences()
    robot.GetReferences().AddReference(str(relative_from_scene))
    output.GetRootLayer().Save()
    print(f"generated: {OUTPUT_STAGE}")


if __name__ == "__main__":
    build()
