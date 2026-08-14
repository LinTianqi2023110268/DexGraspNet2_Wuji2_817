#!/usr/bin/env python3
"""Overlay predicted LEAP grasps on an official single-view point cloud."""

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
    add_leap_hand,
    add_point_cloud,
    add_table,
    as_transform,
    load_leap_robot_model,
    show_or_export,
    transform_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--grasps", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Draw every prediction in the file (for example, one per segmented object)",
    )
    parser.add_argument("--max-points", type=int, default=40000)
    parser.add_argument("--export", type=Path, help="Export a .glb instead of opening a GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    network = np.load(args.input)
    grasps = np.load(args.grasps)
    prediction_count = len(grasps["translation"])
    if args.index < 0 or args.index >= prediction_count:
        raise IndexError(f"index {args.index} is outside prediction output")
    selected_indices = list(range(prediction_count)) if args.all else [args.index]
    source_view = (
        int(grasps["view_index"][selected_indices[0]]) if "view_index" in grasps else 0
    )
    if "view_index" in grasps and any(
        int(grasps["view_index"][index]) != source_view for index in selected_indices
    ):
        raise ValueError("--all currently requires all predictions to come from one view")
    points_world = transform_points(network["pc"][source_view], network["extrinsics"][source_view])
    segmentation = network["seg"][source_view]

    robot_model = load_leap_robot_model()
    scene = trimesh.Scene()
    add_axis(scene, name="world_frame")
    add_axis(scene, network["extrinsics"][source_view], name="camera_frame", axis_length=0.10)
    add_table(scene)
    add_point_cloud(
        scene,
        points_world,
        segmentation,
        name="single_view_network_input",
        max_points=args.max_points,
    )
    palette = [
        (60, 220, 110, 205),
        (245, 120, 55, 205),
        (70, 145, 245, 205),
        (225, 75, 180, 205),
        (245, 205, 55, 205),
        (75, 210, 215, 205),
        (155, 100, 235, 205),
        (195, 225, 90, 205),
        (235, 105, 105, 205),
    ]
    for display_index, prediction_index in enumerate(selected_indices):
        color = palette[display_index % len(palette)]
        qpos = {
            name: float(grasps[name][prediction_index]) for name in robot_model.joint_names
        }
        hand_pose = as_transform(
            grasps["rotation"][prediction_index], grasps["translation"][prediction_index]
        )
        add_axis(
            scene,
            hand_pose,
            name=f"predicted_hand_frame_{prediction_index}",
            axis_length=0.045,
        )
        add_leap_hand(
            scene,
            robot_model,
            hand_pose,
            qpos,
            f"prediction_{prediction_index}",
            color,
        )
        if "seed_point" in grasps:
            seed = trimesh.creation.icosphere(radius=0.005)
            seed.apply_translation(grasps["seed_point"][prediction_index])
            seed.visual.face_colors = color
            scene.add_geometry(seed, geom_name=f"selected_seed_point_{prediction_index}")
        if not args.all and "voxel_center" in grasps:
            voxel_size = float(grasps["voxel_size"])
            box = trimesh.creation.box(extents=[voxel_size] * 3)
            box.apply_translation(grasps["voxel_center"][prediction_index])
            box.visual.face_colors = [250, 210, 40, 190]
            scene.add_geometry(box, geom_name=f"seed_voxel_{prediction_index}")

        details = [f"index={prediction_index}", f"source_view={source_view}"]
        for key in ("object_index", "score", "graspness", "log_prob"):
            if key in grasps:
                details.append(f"{key}={grasps[key][prediction_index]}")
        print(", ".join(details))
    if args.all:
        print("colored hands/seeds=one selected prediction per segmented object")
    else:
        print("colored hand/seed=selected prediction; yellow=seed voxel")
    show_or_export(scene, args.export, "Official scene DexGraspNet2 prediction")


if __name__ == "__main__":
    main()
