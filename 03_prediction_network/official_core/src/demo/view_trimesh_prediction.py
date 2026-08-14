#!/usr/bin/env python3
"""Visualize one predicted LEAP grasp in the two-object world scene."""

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
    add_scene_objects,
    add_table,
    as_transform,
    load_json,
    load_leap_robot_model,
    show_or_export,
    transform_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasps", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--no-point-cloud", action="store_true")
    parser.add_argument("--export", type=Path, help="Export .glb instead of opening a GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grasps = np.load(args.grasps)
    manifest = load_json(args.scene_manifest)
    count = len(grasps["translation"])
    if args.index < 0 or args.index >= count:
        raise IndexError("index {} is outside [0, {})".format(args.index, count))

    robot_model = load_leap_robot_model()
    qpos = {name: float(grasps[name][args.index]) for name in robot_model.joint_names}
    hand_pose = as_transform(grasps["rotation"][args.index], grasps["translation"][args.index])
    scene = trimesh.Scene()
    add_axis(scene, name="world_frame")
    add_axis(scene, hand_pose, name="predicted_hand_frame", axis_length=0.07)
    add_table(scene)
    add_scene_objects(scene, manifest, alpha=125)
    add_leap_hand(scene, robot_model, hand_pose, qpos, "prediction", (60, 220, 110, 220))

    if not args.no_point_cloud:
        network = np.load(manifest["network_input"])
        view = int(grasps["view_index"][args.index]) if "view_index" in grasps else 0
        points_world = transform_points(network["pc"][view], network["extrinsics"][view])
        add_point_cloud(
            scene, points_world, network["seg"][view],
            name="network_point_cloud", max_points=args.max_points,
        )

    details = ["index={}".format(args.index)]
    for key in ("object_index", "score", "graspness", "log_prob"):
        if key in grasps:
            details.append("{}={}".format(key, grasps[key][args.index]))
    print(", ".join(details))
    print("green mesh=predicted articulated LEAP Hand; hand axes show T_world_hand")
    show_or_export(scene, args.export, "DexGraspNet2 predicted grasp")


if __name__ == "__main__":
    main()
