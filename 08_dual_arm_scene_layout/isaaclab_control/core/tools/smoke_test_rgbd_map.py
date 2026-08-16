#!/usr/bin/env python3
"""Build one cuRobo V2 observed-scene TSDF/ESDF from project RGB-D files.

Run this only inside the dedicated ``curobo_v2`` environment.  It never starts
Isaac Sim and never modifies project inputs.  The output is a compact JSON
summary intended to catch path, calibration, depth-scale and Mapper API issues
before Codex connects the mapper to the production runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.home()/"Projects/DexGraspNet2_Wuji2")
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--camera-pose", type=Path, required=True, help="T_world_camera.npy")
    parser.add_argument("--target-mask", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    sys.path.insert(0, str(root/"08_dual_arm_scene_layout/isaaclab_control"))
    from core.config import MapperConfig
    from core.perception_collision import RGBDFrame, CuroboRGBDMapper

    paths = [args.depth, args.intrinsics, args.camera_pose]
    if args.target_mask is not None:
        paths.append(args.target_mask)
    paths = [p.expanduser().resolve() for p in paths]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    frame = RGBDFrame.from_npy(
        args.depth.expanduser().resolve(),
        args.intrinsics.expanduser().resolve(),
        args.camera_pose.expanduser().resolve(),
        None if args.target_mask is None else args.target_mask.expanduser().resolve(),
    )
    cfg = MapperConfig(device=args.device)
    observed = CuroboRGBDMapper(cfg).build(frame)

    valid = frame.valid_depth_mask(cfg)
    summary = {
        "status": "PASS",
        "depth_shape": list(frame.depth_m.shape),
        "valid_depth_pixels": int(np.count_nonzero(valid)),
        "depth_min_valid_m": float(frame.depth_m[valid].min()),
        "depth_max_valid_m": float(frame.depth_m[valid].max()),
        "grid_center_world": observed.grid_center_world.tolist(),
        "extent_meters_xyz": observed.extent_meters_xyz.tolist(),
        "scene_esdf_shape": list(observed.scene_grid.feature_tensor.shape),
        "scene_esdf_voxel_size_m": float(observed.scene_grid.voxel_size),
        "has_target_layer": bool(observed.target_grid is not None),
        "target_esdf_shape": None if observed.target_grid is None else list(observed.target_grid.feature_tensor.shape),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
