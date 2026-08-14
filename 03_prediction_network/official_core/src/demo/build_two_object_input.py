#!/usr/bin/env python3
"""Build a small, simulator-free two-object input for learning DexGraspNet 2.0.

This is deliberately a transparent teaching input, not a replacement for the
paper's rendered GraspNet scenes.  It composes surface samples from objects 000
and 001 plus a table plane, transforms them into one OpenCV-style camera frame,
and writes the same ``network_input.npz`` keys consumed by the predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "two_object_demo"
REQUIRED_ASSET_FILES = (
    "simplified.obj",
    "nontextured_simplified.ply",
    "nontextured_simplified.urdf",
    "surface_points_1000.npy",
    "textured.obj",
    "textured.mtl",
    "texture_map.png",
    "textured.sdf",
    "coacd/decomposed.obj",
    "coacd/coacd.urdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-points", type=int, default=40000)
    parser.add_argument("--table-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def look_at_opencv(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return T_world_camera with camera axes x-right, y-down, z-forward."""

    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.stack([right, down, forward], axis=1)
    transform[:3, 3] = eye
    return transform


def transformed_surface_points(
    mesh_path: Path,
    pose_world_object: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    # trimesh's sampler uses NumPy's legacy global RNG.
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    points, _ = trimesh.sample.sample_surface(mesh, count)
    return points @ pose_world_object[:3, :3].T + pose_world_object[:3, 3]


def main() -> None:
    args = parse_args()
    if args.num_points <= args.table_points or args.table_points < 0:
        raise ValueError("Require num_points > table_points >= 0")

    mesh_root = REPO_ROOT / "data" / "meshdata"
    for code in ("000", "001"):
        missing = [name for name in REQUIRED_ASSET_FILES if not (mesh_root / code / name).is_file()]
        if missing:
            raise FileNotFoundError("Object {} is incomplete: {}".format(code, missing))

    rng = np.random.default_rng(args.seed)
    object_total = args.num_points - args.table_points
    object_counts = [object_total // 2, object_total - object_total // 2]
    x_positions = (-0.075, 0.075)

    world_points = []
    segmentations = []
    objects = []
    for seg_id, (code, count, x_pos) in enumerate(
        zip(("000", "001"), object_counts, x_positions), start=1
    ):
        mesh_path = mesh_root / code / "simplified.obj"
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [x_pos, 0.0, -float(mesh.bounds[0, 2])]
        points = transformed_surface_points(mesh_path, pose, count, rng)
        world_points.append(points)
        segmentations.append(np.full(count, seg_id, dtype=np.int64))
        objects.append(
            {
                "code": code,
                "segmentation_id": seg_id,
                "pose_world_object": pose.tolist(),
                "visual_mesh": str(mesh_path.resolve()),
                "simulation_urdf": str((mesh_root / code / "coacd" / "coacd.urdf").resolve()),
                "surface_points": str((mesh_root / code / "surface_points_1000.npy").resolve()),
            }
        )

    if args.table_points:
        table_xy = rng.uniform([-0.28, -0.24], [0.28, 0.24], size=(args.table_points, 2))
        table = np.column_stack([table_xy, np.zeros(args.table_points, dtype=np.float64)])
        world_points.append(table)
        segmentations.append(np.zeros(args.table_points, dtype=np.int64))

    points_world = np.concatenate(world_points, axis=0)
    seg = np.concatenate(segmentations, axis=0)
    permutation = rng.permutation(len(points_world))
    points_world = points_world[permutation]
    seg = seg[permutation]

    extrinsics = look_at_opencv(
        eye=np.asarray([0.0, -0.72, 0.52], dtype=np.float64),
        target=np.asarray([0.0, 0.0, 0.09], dtype=np.float64),
    )
    points_camera = (points_world - extrinsics[:3, 3]) @ extrinsics[:3, :3]
    if np.any(points_camera[:, 2] <= 0):
        raise RuntimeError("Teaching scene contains points behind the camera")

    # Zero means "not an object boundary".  The official rendered scenes use
    # a real edge image; zero is intentional here so no object surface samples
    # are suppressed during this first, inspectable lesson.
    edge = np.zeros_like(seg)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    network_input_path = output / "network_input.npz"
    np.savez_compressed(
        network_input_path,
        pc=points_camera.astype(np.float32)[None],
        seg=seg[None],
        edge=edge[None],
        extrinsics=extrinsics.astype(np.float32)[None],
    )

    manifest = {
        "schema_version": 1,
        "purpose": "simulator-free teaching input; not a paper benchmark scene",
        "coordinate_contract": {
            "pc": "camera frame, metres, OpenCV x-right/y-down/z-forward",
            "extrinsics": "T_world_camera; p_world = R_world_camera @ p_camera + t_world_camera",
            "object_pose": "T_world_object",
        },
        "network_input": str(network_input_path),
        "num_views": 1,
        "num_points_per_view": int(args.num_points),
        "camera_extrinsics_world_from_camera": extrinsics.tolist(),
        "objects": objects,
        "limitations": [
            "uses complete mesh-surface samples instead of depth-rendered visible surfaces",
            "uses an all-zero edge mask",
            "is for tracing input/output and smoke inference, not reporting paper metrics",
        ],
    }
    manifest_path = output / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote {}".format(network_input_path))
    print("wrote {}".format(manifest_path))
    print("pc shape={}, seg ids={}".format((1, args.num_points, 3), np.unique(seg).tolist()))


if __name__ == "__main__":
    main()
