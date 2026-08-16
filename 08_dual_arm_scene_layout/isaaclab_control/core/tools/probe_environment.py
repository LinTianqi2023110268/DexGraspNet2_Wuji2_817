#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Read-only cuRobo core environment probe")
    parser.add_argument("--project-root", type=Path, default=Path.home()/"Projects/DexGraspNet2_Wuji2")
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root/"08_dual_arm_scene_layout/isaaclab_control"))
    import torch
    import curobo
    from curobo.perception import Mapper, MapperCfg
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.robot_builder import RobotBuilder
    print(f"project_root={root}")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"curobo={getattr(curobo, '__version__', 'unknown')}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("Mapper/MapperCfg import: OK")
    print("InverseKinematics import: OK")
    print("Kinematics/KinematicsCfg import: OK")
    print("RobotBuilder import: OK")


if __name__ == '__main__':
    main()
