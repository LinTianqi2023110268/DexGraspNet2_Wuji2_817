#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.home()/"Projects/DexGraspNet2_Wuji2")
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root/"08_dual_arm_scene_layout/isaaclab_control"))
    from core.bridge import CuroboWorkerClient
    with CuroboWorkerClient(root) as client:
        print(client.request({"op":"ping"}))


if __name__ == '__main__':
    main()
