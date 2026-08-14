#!/usr/bin/env python3
"""Build one DexGraspNet2 network input from one official GraspNet camera view."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as scio
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.pc import depth_image_to_point_cloud, get_workspace_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True, help="For example: scene_0000")
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_dir = REPO_ROOT / "data" / "scenes" / args.scene_id / args.camera
    view_name = f"{args.view:04d}"
    depth_path = camera_dir / "depth_gt" / f"{view_name}.png"
    label_path = camera_dir / "label_gt" / f"{view_name}.png"
    meta_path = camera_dir / "meta" / f"{view_name}.mat"
    edge_path = camera_dir / "edge_gt" / f"{view_name}.png"
    for required in (depth_path, label_path, meta_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    depth = np.asarray(Image.open(depth_path))
    segmentation = np.asarray(Image.open(label_path))
    edge = np.asarray(Image.open(edge_path)) if edge_path.is_file() else np.zeros_like(depth)
    meta = scio.loadmat(meta_path)
    camera_poses = np.load(camera_dir / "camera_poses.npy")
    align_mat = np.load(camera_dir / "cam0_wrt_table.npy")
    if args.view < 0 or args.view >= len(camera_poses):
        raise IndexError(f"view {args.view} is outside [0, {len(camera_poses)})")

    dense_cloud = depth_image_to_point_cloud(
        depth, meta["intrinsic_matrix"], meta["factor_depth"]
    )
    extrinsics = align_mat @ camera_poses[args.view]
    valid = (depth > 0) & get_workspace_mask(dense_cloud, segmentation, extrinsics)
    points = dense_cloud[valid].astype(np.float32)
    labels = segmentation[valid].astype(np.int64)
    edges = edge[valid].astype(np.int64)
    if len(points) == 0:
        raise RuntimeError("No valid depth points remain after workspace masking")

    rng = np.random.default_rng(args.seed)
    # Match the official preprocessing: sample exactly N points with replacement.
    indices = rng.choice(len(points), args.num_points, replace=True)
    points = points[indices]
    labels = labels[indices]
    edges = edges[indices]

    output = args.output or camera_dir / f"network_input_view_{view_name}.npz"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pc=points[None],
        seg=labels[None],
        edge=edges[None],
        extrinsics=extrinsics.astype(np.float32)[None],
        scene_id=np.asarray(args.scene_id),
        camera=np.asarray(args.camera),
        source_view=np.asarray(args.view, dtype=np.int64),
    )
    object_ids, counts = np.unique(labels[labels > 0], return_counts=True)
    print(f"wrote {output}")
    print(f"depth={depth.shape}, valid_before_sampling={valid.sum()}, pc={(1, len(points), 3)}")
    print(f"visible_object_ids={dict(zip(object_ids.tolist(), counts.tolist()))}")
    print("pc frame=camera; extrinsics maps camera coordinates to table/world coordinates")


if __name__ == "__main__":
    main()
