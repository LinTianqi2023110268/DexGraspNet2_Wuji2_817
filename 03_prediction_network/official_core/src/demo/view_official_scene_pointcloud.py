#!/usr/bin/env python3
"""Visualize one official single-view DexGraspNet2 point-cloud input with Trimesh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.demo.trimesh_scene_utils import (
    add_axis,
    add_point_cloud,
    add_table,
    show_or_export,
    transform_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frame", choices=("camera", "world"), default="camera")
    parser.add_argument("--max-points", type=int, default=40000)
    parser.add_argument("--export", type=Path, help="Export a .glb instead of opening a GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = np.load(args.input)
    points = np.asarray(payload["pc"][0])
    segmentation = np.asarray(payload["seg"][0])
    extrinsics = np.asarray(payload["extrinsics"][0])
    scene = trimesh.Scene()
    if args.frame == "world":
        points = transform_points(points, extrinsics)
        add_axis(scene, name="world_frame")
        add_axis(scene, extrinsics, name="camera_frame")
        add_table(scene)
    else:
        add_axis(scene, name="camera_frame")
    add_point_cloud(
        scene,
        points,
        segmentation,
        name="single_view_network_input",
        max_points=args.max_points,
    )
    print(f"points={points.shape}, frame={args.frame}")
    print(f"segment_counts={dict(zip(*np.unique(segmentation, return_counts=True)))}")
    print("gray=table/background; other colors=visible object instances")
    show_or_export(scene, args.export, "Official DexGraspNet2 single-view point cloud")


if __name__ == "__main__":
    main()
