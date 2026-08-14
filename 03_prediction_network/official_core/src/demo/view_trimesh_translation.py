#!/usr/bin/env python3
"""Visualize seed point, voxel center, translation offset and predicted hand."""

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
    parser.add_argument("--no-hand", action="store_true", help="Hide the LEAP mesh but retain its root frame")
    parser.add_argument("--export", type=Path, help="Export .glb instead of opening a GUI")
    return parser.parse_args()


def colored_sphere(center: np.ndarray, radius: float, color) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.apply_translation(center)
    mesh.visual.face_colors = color
    return mesh


def colored_cylinder(start: np.ndarray, end: np.ndarray, radius: float, color) -> trimesh.Trimesh:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        return colored_sphere(start, radius * 1.5, color)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], vector / length)
    transform[:3, 3] = (np.asarray(start) + np.asarray(end)) / 2
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=24, transform=transform)
    mesh.visual.face_colors = color
    return mesh


def add_voxel_wireframe(
    scene: trimesh.Scene,
    transform: np.ndarray,
    size: float,
    color=(255, 225, 20, 255),
) -> None:
    """Draw an actual-size camera-aligned voxel without hiding its seed point."""

    half = size / 2
    corners_local = np.asarray(
        [[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)],
        dtype=np.float64,
    )
    corners_world = transform_points(corners_local, transform)
    edges = []
    for i, a in enumerate(corners_local):
        for j in range(i + 1, len(corners_local)):
            b = corners_local[j]
            if np.count_nonzero(a != b) == 1:
                edges.append((i, j))
    for edge_index, (start, end) in enumerate(edges):
        scene.add_geometry(
            colored_cylinder(
                corners_world[start], corners_world[end],
                radius=max(size * 0.065, 0.00025), color=color,
            ),
            geom_name="voxel_edge_{:02d}".format(edge_index),
        )


def main() -> None:
    args = parse_args()
    grasps = np.load(args.grasps)
    manifest = load_json(args.scene_manifest)
    required = {
        "rotation", "translation", "seed_point", "voxel_center",
        "delta_translation_scaled", "voxel_size", "trans_scale",
    }
    missing = sorted(required.difference(grasps.files))
    if missing:
        raise KeyError(
            "grasp file lacks teaching fields {}. Rerun predict_custom_scene.py.".format(missing)
        )
    count = len(grasps["translation"])
    if args.index < 0 or args.index >= count:
        raise IndexError("index {} is outside [0, {})".format(args.index, count))

    index = args.index
    seed = np.asarray(grasps["seed_point"][index], dtype=np.float64)
    center = np.asarray(grasps["voxel_center"][index], dtype=np.float64)
    hand_translation = np.asarray(grasps["translation"][index], dtype=np.float64)
    hand_pose = as_transform(grasps["rotation"][index], hand_translation)
    voxel_size = float(grasps["voxel_size"])
    trans_scale = float(grasps["trans_scale"])
    delta_scaled = np.asarray(grasps["delta_translation_scaled"][index], dtype=np.float64)

    scene = trimesh.Scene()
    add_axis(scene, name="world_frame")
    add_axis(scene, hand_pose, name="predicted_hand_frame", axis_length=0.07)
    add_table(scene)
    add_scene_objects(scene, manifest, alpha=105)
    if not args.no_hand:
        robot_model = load_leap_robot_model()
        qpos = {name: float(grasps[name][index]) for name in robot_model.joint_names}
        add_leap_hand(scene, robot_model, hand_pose, qpos, "prediction", (60, 220, 110, 125))

    # Magenta is the original surface sample.  The yellow cube is the center of
    # its 5 mm camera-aligned voxel.  The cyan segment is delta / trans_scale.
    scene.add_geometry(
        colored_sphere(seed, max(voxel_size * 0.22, 0.0008), [255, 20, 220, 255]),
        geom_name="seed_point_magenta",
    )
    view = int(grasps["view_index"][index]) if "view_index" in grasps else 0
    network = np.load(manifest["network_input"])
    camera_to_world = np.asarray(network["extrinsics"][view], dtype=np.float64)
    voxel_transform = camera_to_world.copy()
    voxel_transform[:3, 3] = center
    add_voxel_wireframe(scene, voxel_transform, voxel_size)
    scene.add_geometry(
        colored_sphere(center, max(voxel_size * 0.14, 0.00055), [255, 235, 20, 255]),
        geom_name="voxel_center_yellow",
    )
    scene.add_geometry(
        colored_cylinder(center, hand_translation, 0.0018, [20, 220, 235, 255]),
        geom_name="translation_offset_cyan",
    )

    if not args.no_point_cloud:
        points_world = transform_points(network["pc"][view], camera_to_world)
        add_point_cloud(
            scene, points_world, network["seg"][view],
            name="network_point_cloud", max_points=args.max_points,
        )

    reconstructed = center + camera_to_world[:3, :3] @ (delta_scaled / trans_scale)
    print("index={}".format(index))
    print("magenta seed point (world)       = {}".format(seed.tolist()))
    print("yellow voxel center (world)     = {}".format(center.tolist()))
    print("delta_translation_scaled(camera)= {}".format(delta_scaled.tolist()))
    print("trans_scale                      = {}".format(trans_scale))
    print("green hand root (world)         = {}".format(hand_translation.tolist()))
    print("reconstruction error (m)        = {:.3e}".format(
        float(np.linalg.norm(reconstructed - hand_translation))
    ))
    print("colors: seed=magenta, voxel=yellow, offset=center-to-hand cyan, hand=green")
    show_or_export(scene, args.export, "DexGraspNet2 translation parameterization")


if __name__ == "__main__":
    main()
