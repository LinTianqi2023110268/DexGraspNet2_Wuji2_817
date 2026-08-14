#!/usr/bin/env python3
"""Visualize the five LEAP waypoints prepared for Isaac Sim 5.0 execution."""

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
    add_scene_objects,
    add_table,
    load_json,
    load_leap_robot_model,
    show_or_export,
)


WAYPOINT_NAMES = ("pregrasp", "cover", "grasp", "squeeze", "lift")
WAYPOINT_COLORS = (
    (120, 120, 255, 75),
    (60, 210, 220, 90),
    (70, 230, 90, 150),
    (255, 170, 50, 110),
    (235, 65, 80, 105),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--waypoint", choices=("all",) + WAYPOINT_NAMES, default="all")
    parser.add_argument("--export", type=Path, help="Export .glb instead of opening a GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job = np.load(args.job)
    manifest = load_json(args.job.with_suffix(".json"))
    count = len(job["waypoint_pose_world"])
    if args.index < 0 or args.index >= count:
        raise IndexError("index {} is outside [0, {})".format(args.index, count))

    robot_model = load_leap_robot_model()
    saved_names = [str(value) for value in job["finger_joint_names"].tolist()]
    if saved_names != robot_model.joint_names:
        raise ValueError("Waypoint joint order does not match the LEAP URDF")
    selected = range(5) if args.waypoint == "all" else [WAYPOINT_NAMES.index(args.waypoint)]

    scene = trimesh.Scene()
    add_axis(scene, name="world_frame")
    add_table(scene)
    add_scene_objects(scene, manifest, alpha=130)
    for waypoint_index in selected:
        pose = job["waypoint_pose_world"][args.index, waypoint_index]
        values = job["waypoint_joint_positions"][args.index, waypoint_index]
        qpos = dict(zip(saved_names, values))
        name = WAYPOINT_NAMES[waypoint_index]
        add_leap_hand(
            scene, robot_model, pose, qpos,
            "{}_{}".format(waypoint_index, name), WAYPOINT_COLORS[waypoint_index],
        )
        add_axis(scene, pose, name="{}_frame".format(name), axis_length=0.045)

    print("grasp index={}, pregrasp_valid={}".format(
        args.index, bool(job["pregrasp_valid"][args.index])
    ))
    print("waypoint colors: pregrasp=blue, cover=cyan, grasp=green, squeeze=orange, lift=red")
    print("The five poses are targets; Isaac Sim interpolates between them during execution.")
    show_or_export(scene, args.export, "DexGraspNet2 five execution waypoints")


if __name__ == "__main__":
    main()
