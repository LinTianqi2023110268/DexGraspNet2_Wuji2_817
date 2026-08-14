#!/usr/bin/env python3
"""Visualize Wuji2 DexGraspNet2 predictions with Trimesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import trimesh

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.project import source_path  # noqa: E402
from wuji2_dgn2.visual import (  # noqa: E402
    DEFAULT_URDF,
    PALETTE,
    Wuji2VisualModel,
    load_object_mesh,
)


DEFAULT_DATA_ROOT = source_path("active_training_scene_dataset")
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "05_inference/outputs/scene_0036_view_0000_all.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--top-hands", type=int, default=1)
    parser.add_argument(
        "--hide-seeds",
        action="store_true",
        help="Hide the 1024 candidate seed overlay and show only the per-point heatmap.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10000,
        help="Maximum displayed cloud points; use 0 only when explicitly requesting all points.",
    )
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def normalized(values: np.ndarray, low=2.0, high=98.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    prediction_path = args.predictions.resolve()
    with np.load(prediction_path) as archive:
        prediction = {key: archive[key] for key in archive.files}
    scene_index = int(prediction["scene_index"])
    view_index = int(prediction["view_index"])
    manifest_path = (
        args.data_root.resolve()
        / "scenes"
        / f"scene_{scene_index:04d}"
        / "scene_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranking = np.asarray(prediction["ranking"], dtype=np.int64)
    top_hands = max(0, min(args.top_hands, len(ranking)))

    display = trimesh.Scene()
    display.add_geometry(
        trimesh.creation.axis(origin_size=0.004, axis_length=0.10),
        geom_name="world_frame",
    )
    table = trimesh.creation.box(extents=manifest["table"]["size_m"])
    table.apply_translation(
        [
            0.0,
            0.0,
            float(manifest["table"]["top_z_m"])
            - 0.5 * float(manifest["table"]["size_m"][2]),
        ]
    )
    table.visual.face_colors = [145, 145, 145, 55]
    display.add_geometry(table, geom_name="table")
    for scene_object in manifest["objects"]:
        mesh = load_object_mesh(scene_object["asset"])
        mesh.apply_transform(scene_object["T_world_centered_object"])
        color = PALETTE[
            (int(scene_object["segmentation_id"]) - 1) % len(PALETTE)
        ].copy()
        color[3] = 38
        mesh.visual.face_colors = color
        display.add_geometry(
            mesh,
            geom_name=f"object_{int(scene_object['segmentation_id']):03d}_transparent",
        )

    points = np.asarray(prediction["point_cloud_world"], dtype=np.float64)
    predicted_objectness = np.asarray(prediction["predicted_objectness"], dtype=np.int64)
    predicted_graspness = np.asarray(prediction["predicted_graspness_log"], dtype=np.float64)
    if args.max_points > 0 and len(points) > args.max_points:
        shown = np.linspace(0, len(points) - 1, args.max_points, dtype=np.int64)
        points = points[shown]
        predicted_objectness = predicted_objectness[shown]
        predicted_graspness = predicted_graspness[shown]
    colors = np.full((len(points), 4), [135, 135, 135, 65], dtype=np.uint8)
    object_mask = predicted_objectness == 1
    if object_mask.any():
        heat = normalized(predicted_graspness[object_mask])
        rgba = matplotlib.colormaps["viridis"](heat)
        colors[object_mask] = np.rint(rgba * 255).astype(np.uint8)
        colors[object_mask, 3] = 225
    display.add_geometry(
        trimesh.points.PointCloud(points, colors=colors),
        geom_name=f"predicted_objectness_and_graspness_{len(points)}_points",
    )

    score = np.asarray(prediction["score"], dtype=np.float64)
    if not args.hide_seeds:
        seed_points = np.asarray(prediction["seed_point_world"], dtype=np.float64)
        seed_heat = normalized(score)
        seed_colors = np.rint(matplotlib.colormaps["plasma"](seed_heat) * 255).astype(np.uint8)
        seed_colors[:, 3] = 255
        display.add_geometry(
            trimesh.points.PointCloud(seed_points, colors=seed_colors),
            geom_name="all_1024_candidate_seed_points_score_colored",
        )

    model = Wuji2VisualModel(args.urdf)
    joint_order = [str(value) for value in prediction["joint_order"].tolist()]
    for rank, candidate_index in enumerate(ranking[:top_hands]):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = prediction["rotation_world"][candidate_index]
        transform[:3, 3] = prediction["translation_world"][candidate_index]
        qpos = dict(
            zip(joint_order, prediction["qpos"][candidate_index].astype(float).tolist())
        )
        color = np.asarray([40, 230, 90, 220] if rank == 0 else [70, 150, 245, 150])
        for name, mesh in model.scene_geometry(
            transform, qpos, color, f"prediction_rank_{rank + 1:02d}"
        ):
            display.add_geometry(mesh, geom_name=name)
        display.add_geometry(
            trimesh.creation.axis(
                origin_size=0.0025,
                axis_length=0.045,
                transform=transform,
            ),
            geom_name=f"r_base_link_frame_rank_{rank + 1:02d}",
        )

    if args.export is None:
        destination = (
            PROJECT_ROOT / "05_inference/outputs/visualizations"
            / f"scene_{scene_index:04d}_view_{view_index:04d}_wuji2_network_prediction.glb"
        )
    else:
        destination = args.export.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    display.export(destination)
    best = int(ranking[0])
    print(
        f"scene={scene_index:04d} view={view_index:04d} candidates={len(ranking)} "
        f"best={best} seed_object={int(prediction['seed_segmentation'][best])} "
        f"score={float(score[best]):.6f} top_hands={top_hands}"
    )
    print("point cloud: gray=predicted background, viridis=predicted object graspness")
    print(
        "candidate seeds: hidden"
        if args.hide_seeds
        else "candidate seeds: plasma by combined score; best hand=green, other top hands=blue"
    )
    print(f"exported: {destination}")
    if args.show:
        display.show(caption="Wuji2 DexGraspNet2 network prediction")


if __name__ == "__main__":
    main()
