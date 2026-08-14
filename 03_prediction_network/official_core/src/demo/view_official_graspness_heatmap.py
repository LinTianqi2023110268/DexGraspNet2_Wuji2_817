#!/usr/bin/env python3
"""Show official scene_0000/view_0000 raw graspness as a Trimesh heatmap."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as scio
import trimesh
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.pc import depth_image_to_point_cloud, get_workspace_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scene_0000")
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=40000)
    parser.add_argument(
        "--score-space",
        choices=("log", "raw"),
        default="log",
        help="Paper/training GS is log(raw + 1e-3); raw is kept for auditing.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--export", type=Path, default=None)
    return parser.parse_args()


def set_camera(
    scene: trimesh.Scene, world_from_camera: np.ndarray, intrinsic: np.ndarray
) -> None:
    width, height = 1280, 720
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    scene.camera.resolution = [width, height]
    scene.camera.fov = [
        np.degrees(2.0 * np.arctan(width / (2.0 * fx))),
        np.degrees(2.0 * np.arctan(height / (2.0 * fy))),
    ]
    opencv_from_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    scene.camera_transform = world_from_camera @ opencv_from_opengl


def main() -> None:
    args = parse_args()
    view_name = f"{args.view:04d}"
    camera_dir = REPO_ROOT / "data" / "scenes" / args.scene / args.camera
    graspness_path = (
        REPO_ROOT
        / "data"
        / "dex_graspness_new"
        / args.scene
        / args.camera
        / f"{view_name}.npy"
    )
    depth = np.asarray(Image.open(camera_dir / "depth_gt" / f"{view_name}.png"))
    segmentation = np.asarray(
        Image.open(camera_dir / "label_gt" / f"{view_name}.png")
    )
    meta = scio.loadmat(camera_dir / "meta" / f"{view_name}.mat")
    intrinsic = np.asarray(meta["intrinsic_matrix"], dtype=np.float64)
    world_from_camera = (
        np.load(camera_dir / "cam0_wrt_table.npy")
        @ np.load(camera_dir / "camera_poses.npy")[args.view]
    )
    dense = depth_image_to_point_cloud(
        depth, intrinsic, meta["factor_depth"]
    )
    valid = (depth > 0) & get_workspace_mask(
        dense, segmentation, world_from_camera
    )
    cloud_camera = dense[valid].astype(np.float32)
    labels = segmentation[valid].astype(np.int64)
    raw_graspness = np.load(graspness_path).reshape(-1).astype(np.float32)
    if len(raw_graspness) != len(cloud_camera):
        raise RuntimeError(
            f"graspness/cloud mismatch: {len(raw_graspness)} != {len(cloud_camera)}"
        )
    indices = np.random.default_rng(args.seed).choice(
        len(cloud_camera), args.num_points, replace=True
    )
    cloud_camera = cloud_camera[indices]
    labels = labels[indices]
    raw_graspness = raw_graspness[indices]
    prepared_path = camera_dir / f"network_input_view_{view_name}.npz"
    if prepared_path.is_file() and args.seed == 0 and args.num_points == 40000:
        with np.load(prepared_path) as prepared:
            if not np.array_equal(cloud_camera, prepared["pc"][0]):
                raise RuntimeError("Heatmap sampling differs from prepared network input")
            if not np.array_equal(labels, prepared["seg"][0]):
                raise RuntimeError("Heatmap segmentation differs from prepared network input")

    cloud_world = (
        cloud_camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    )
    object_mask = labels > 0
    if not object_mask.any():
        raise RuntimeError("Official view has no visible object points")
    colors = np.tile(np.asarray([155, 155, 155, 180], np.uint8), (len(labels), 1))
    display_score = (
        np.log(raw_graspness + 1.0e-3)
        if args.score_space == "log"
        else raw_graspness
    )
    colors[object_mask] = trimesh.visual.color.interpolate(
        display_score[object_mask], color_map="viridis"
    )
    colors[object_mask, 3] = 255
    display = trimesh.Scene()
    display.add_geometry(
        trimesh.points.PointCloud(cloud_world, colors=colors),
        geom_name=f"official_scene_0000_view_0000_{args.score_space}_graspness_40000_points",
    )
    set_camera(display, world_from_camera, intrinsic)
    destination = (
        args.export.resolve()
        if args.export is not None
        else camera_dir
        / f"official_{args.scene}_view_{view_name}_{args.score_space}_graspness_heatmap.glb"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    display.export(destination)
    object_values = raw_graspness[object_mask]
    metadata = {
        "source": "official DexGraspNet2.0 dex_graspness_new",
        "scene": args.scene,
        "camera": args.camera,
        "view": args.view,
        "sampling_seed": args.seed,
        "score_space": args.score_space,
        "point_count": int(len(cloud_world)),
        "table_point_count": int((~object_mask).sum()),
        "object_point_count": int(object_mask.sum()),
        "positive_graspness_point_count": int((object_values > 0).sum()),
        "raw_graspness_object_min_median_max": [
            float(object_values.min()),
            float(np.median(object_values)),
            float(object_values.max()),
        ],
        "display_score_object_min_median_max": [
            float(display_score[object_mask].min()),
            float(np.median(display_score[object_mask])),
            float(display_score[object_mask].max()),
        ],
        "color_contract": (
            "table gray; object points log(raw + 1e-3) with per-scene linear Viridis scale"
            if args.score_space == "log"
            else "table gray; object points raw graspness with per-scene linear Viridis scale"
        ),
    }
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"exported: {destination}")
    print(f"metadata: {metadata_path}")
    if args.show:
        display.show(caption="Official DexGraspNet2 scene_0000/view_0000 graspness")


if __name__ == "__main__":
    main()
