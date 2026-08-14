#!/usr/bin/env python3
"""Visualize DexGraspNet2 network input, segmentation, frames and source meshes."""

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
    add_scene_objects,
    add_table,
    load_json,
    show_or_export,
    transform_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--frame", choices=("camera", "world"), default="world")
    parser.add_argument(
        "--viewpoint",
        choices=("sensor", "orbit"),
        help=(
            "How the Trimesh viewer observes the displayed coordinates. "
            "Defaults to sensor for --frame camera and orbit for --frame world."
        ),
    )
    parser.add_argument("--max-points", type=int, default=40000)
    parser.add_argument("--export", type=Path, help="Export .glb instead of opening a GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = np.load(args.input)
    manifest = load_json(args.scene_manifest)
    view = args.view
    if view < 0 or view >= len(payload["pc"]):
        raise IndexError("view {} is outside [0, {})".format(view, len(payload["pc"])))
    points_camera = payload["pc"][view]
    segmentation = payload["seg"][view]
    camera_to_world = np.asarray(payload["extrinsics"][view], dtype=np.float64)
    viewpoint = args.viewpoint or ("sensor" if args.frame == "camera" else "orbit")
    if viewpoint == "sensor" and args.frame != "camera":
        raise ValueError("--viewpoint sensor requires --frame camera")

    scene = trimesh.Scene()
    if args.frame == "world":
        points = transform_points(points_camera, camera_to_world)
        world_to_display = np.eye(4)
        add_axis(scene, name="world_frame")
        add_axis(scene, camera_to_world, name="camera_frame", axis_length=0.08)
        add_table(scene)
    else:
        world_to_display = np.linalg.inv(camera_to_world)
        points = points_camera
        # The coordinate-frame axes originate at the sensor itself, so they are
        # useful in the free orbit view but would sit on the eye in sensor view.
        if viewpoint == "orbit":
            add_axis(scene, name="camera_frame")
        add_table(scene, world_to_display)

    add_point_cloud(scene, points, segmentation, max_points=args.max_points)
    add_scene_objects(scene, manifest, world_to_display=world_to_display)
    if viewpoint == "sensor":
        # Trimesh/OpenGL cameras look along local -Z with +Y upward.  The input
        # follows OpenCV (+Z forward, +Y downward), so a 180-degree rotation
        # about X makes the GUI optical axis and screen directions match the
        # camera which produced ``pc``.  The teaching NPZ has no intrinsics;
        # 60x45 degrees is therefore an illustrative field of view.
        scene.camera.resolution = np.asarray([1280, 720], dtype=np.int64)
        scene.camera.fov = np.asarray([60.0, 45.0], dtype=np.float64)
        scene.camera_transform = np.diag([1.0, -1.0, -1.0, 1.0])
    print("frame={}, view={}, points={}, seg_ids={}".format(
        args.frame, view, len(points), np.unique(segmentation).tolist()
    ))
    print("axis colors: X=red, Y=green, Z=blue; object 000=orange, 001=blue")
    print("viewer viewpoint={}".format(viewpoint))
    if viewpoint == "sensor":
        print("sensor view: screen right=+X_camera, screen down=+Y_camera, depth=+Z_camera")
    show_or_export(scene, args.export, "DexGraspNet2 input ({})".format(args.frame))


if __name__ == "__main__":
    main()
