#!/usr/bin/env python3
"""Arrange one object's top 35 filtered Wuji2 grasps in Trimesh grids.

Input
-----
``01_cases/scene_xxxx_view_xxxx/02_predictions/``
``balanced_filtered_predictions.npz`` and the matching scene manifest.

Output
------
Page 1 contains ranks 1--25 in a 5 x 5 grid.  Page 2 contains ranks
26--35 in a 5 x 2 grid.  A JSON index records rank, source candidate and
score for every cell.  The hand links use convex hull display meshes only to
keep the 35-pose audit responsive; transforms and q20 values are unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import write_json_atomic  # noqa: E402
from wuji2_dgn2.visual import (  # noqa: E402
    DEFAULT_URDF,
    PALETTE,
    Wuji2VisualModel,
    load_object_mesh,
)


SELECTION = PIPELINE_ROOT / "00_config/test_5scene.json"
CASE_ROOT = PIPELINE_ROOT / "01_cases"
COLORS = {
    "hand": np.asarray([25, 205, 235, 205], dtype=np.uint8),
    "target": np.asarray([255, 188, 25, 215], dtype=np.uint8),
    "other": np.asarray([145, 155, 170, 75], dtype=np.uint8),
    "table": np.asarray([180, 180, 180, 45], dtype=np.uint8),
    "label": np.asarray([245, 55, 75, 255], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--object-id", type=int, default=14)
    parser.add_argument("--count", type=int, default=35)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--points-per-cell", type=int, default=2500)
    parser.add_argument("--show", choices=("none", "page1", "page2"), default="none")
    return parser.parse_args()


def selected_scene(scene_index: int) -> dict:
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    matches = [
        item for item in payload["scenes"]
        if int(item["scene_index"]) == scene_index
    ]
    if len(matches) != 1:
        raise KeyError(f"scene {scene_index} is not unique in {SELECTION}")
    return matches[0]


def offset_transform(offset: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = offset
    return transform


def add_rank_digits(scene: trimesh.Scene, rank: int, offset: np.ndarray) -> None:
    """Draw a compact two-digit seven-segment rank label on each tile."""
    patterns = {
        "0": "abcedf", "1": "bc", "2": "abdeg", "3": "abcdg",
        "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
        "8": "abcdefg", "9": "abcdfg",
    }
    # Segment centers and XY extents.  Labels lie just above the table.
    horizontal = (0.014, 0.0022, 0.0012)
    vertical = (0.0022, 0.014, 0.0012)
    definitions = {
        "a": ((0.0, 0.014, 0.0), horizontal),
        "b": ((0.007, 0.007, 0.0), vertical),
        "c": ((0.007, -0.007, 0.0), vertical),
        "d": ((0.0, -0.014, 0.0), horizontal),
        "e": ((-0.007, -0.007, 0.0), vertical),
        "f": ((-0.007, 0.007, 0.0), vertical),
        "g": ((0.0, 0.0, 0.0), horizontal),
    }
    text = f"{rank:02d}"
    origin = offset + np.asarray([-0.225, 0.125, 0.024], dtype=np.float64)
    for digit_index, digit in enumerate(text):
        digit_origin = origin + np.asarray([digit_index * 0.020, 0.0, 0.0])
        for segment in patterns[digit]:
            center, extents = definitions[segment]
            box = trimesh.creation.box(extents=extents)
            box.apply_translation(digit_origin + np.asarray(center))
            box.visual.face_colors = COLORS["label"]
            scene.add_geometry(
                box,
                geom_name=f"rank_{rank:02d}_digit_{digit_index}_{segment}",
            )


def build_hand_hulls(model: Wuji2VisualModel) -> dict[str, list[tuple[trimesh.Trimesh, np.ndarray]]]:
    result = {}
    for link_name, entries in model.meshes.items():
        result[link_name] = [
            (source.convex_hull, visual_origin)
            for source, visual_origin in entries
        ]
    return result


def add_hand(
    scene: trimesh.Scene,
    model: Wuji2VisualModel,
    hulls: dict,
    world_from_base: np.ndarray,
    qpos: dict[str, float],
    offset: np.ndarray,
    rank: int,
) -> None:
    fk = model.forward_kinematics(qpos)
    tile = offset_transform(offset)
    for link_name, entries in hulls.items():
        for visual_index, (source, visual_origin) in enumerate(entries):
            mesh = source.copy()
            mesh.apply_transform(
                tile @ world_from_base @ fk[link_name] @ visual_origin
            )
            mesh.visual.face_colors = COLORS["hand"]
            scene.add_geometry(
                mesh,
                geom_name=f"rank_{rank:02d}_hand_{link_name}_{visual_index}",
            )


def add_cell_scene(
    display: trimesh.Scene,
    manifest: dict,
    object_meshes: list[tuple[int, trimesh.Trimesh]],
    points: np.ndarray,
    segmentation: np.ndarray,
    point_indices: np.ndarray,
    target_id: int,
    offset: np.ndarray,
    rank: int,
) -> None:
    tile = offset_transform(offset)
    table = trimesh.creation.box(extents=manifest["table"]["size_m"])
    table.apply_translation(
        [
            0.0,
            0.0,
            float(manifest["table"]["top_z_m"])
            - 0.5 * float(manifest["table"]["size_m"][2]),
        ]
    )
    table.apply_transform(tile)
    table.visual.face_colors = COLORS["table"]
    display.add_geometry(table, geom_name=f"rank_{rank:02d}_table")

    for seg_id, source in object_meshes:
        mesh = source.copy()
        mesh.apply_transform(tile)
        mesh.visual.face_colors = (
            COLORS["target"] if seg_id == target_id else COLORS["other"]
        )
        display.add_geometry(
            mesh,
            geom_name=f"rank_{rank:02d}_object_{seg_id:03d}",
        )

    shown_points = points[point_indices] + offset[None, :]
    shown_seg = segmentation[point_indices]
    colors = np.full((len(shown_points), 4), COLORS["other"], dtype=np.uint8)
    colors[shown_seg == target_id] = COLORS["target"]
    display.add_geometry(
        trimesh.points.PointCloud(shown_points, colors=colors),
        geom_name=f"rank_{rank:02d}_single_view_points",
    )
    add_rank_digits(display, rank, offset)


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.columns != 5:
        raise ValueError("this audit is defined for a positive count and 5 columns")
    scene_cfg = selected_scene(args.scene)
    view_index = int(scene_cfg["view_index"])
    source_root = CASE_ROOT / f"scene_{args.scene:04d}_view_{view_index:04d}"
    prediction_path = source_root / "02_predictions/balanced_filtered_predictions.npz"
    manifest_path = PROJECT_ROOT / scene_cfg["scene_manifest"]
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    target = np.asarray(data["target_segmentation_id"], dtype=np.int64)
    target_indices = np.flatnonzero(target == args.object_id)
    if len(target_indices) < args.count:
        raise RuntimeError(
            f"object {args.object_id} has only {len(target_indices)} filtered poses"
        )
    ordered = target_indices[np.argsort(-np.asarray(data["score"])[target_indices])]
    ordered = ordered[: args.count]

    object_meshes = []
    for record in manifest["objects"]:
        mesh = load_object_mesh(record["asset"])
        mesh.apply_transform(record["T_world_centered_object"])
        object_meshes.append((int(record["segmentation_id"]), mesh))

    model = Wuji2VisualModel(DEFAULT_URDF)
    hulls = build_hand_hulls(model)
    joint_order = [str(value) for value in data["joint_order"].tolist()]
    points = np.asarray(data["point_cloud_world"], dtype=np.float64)
    segmentation = np.asarray(data["ground_truth_segmentation"], dtype=np.int64)
    shown_count = min(max(args.points_per_cell, 0), len(points))
    point_indices = (
        np.linspace(0, len(points) - 1, shown_count, dtype=np.int64)
        if shown_count else np.empty(0, dtype=np.int64)
    )

    output_root = source_root / "03_visualization/top35_grid"
    output_root.mkdir(parents=True, exist_ok=True)
    pages = [ordered[:25], ordered[25:35]]
    index_records = []
    page_paths = []
    spacing = np.asarray([0.62, 0.42, 0.0], dtype=np.float64)
    for page_number, page_indices in enumerate(pages, start=1):
        if not len(page_indices):
            continue
        display = trimesh.Scene()
        for local_index, data_index in enumerate(page_indices):
            rank = (page_number - 1) * 25 + local_index + 1
            row, column = divmod(local_index, args.columns)
            offset = np.asarray(
                [column * spacing[0], -row * spacing[1], 0.0], dtype=np.float64
            )
            add_cell_scene(
                display, manifest, object_meshes, points, segmentation,
                point_indices, args.object_id, offset, rank,
            )
            qpos = dict(
                zip(
                    joint_order,
                    np.asarray(data["qpos"][data_index], dtype=float).tolist(),
                )
            )
            add_hand(
                display,
                model,
                hulls,
                np.asarray(data["T_world_r_base_link"][data_index], dtype=np.float64),
                qpos,
                offset,
                rank,
            )
            index_records.append(
                {
                    "rank": rank,
                    "page": page_number,
                    "row_1_based": row + 1,
                    "column_1_based": column + 1,
                    "source_candidate_index": int(data["source_candidate_index"][data_index]),
                    "score": float(data["score"][data_index]),
                    "graspness": float(data["graspness"][data_index]),
                    "log_prob": float(data["log_prob"][data_index]),
                    "final_table_clearance_m": float(data["final_table_clearance_m"][data_index]),
                    "final_non_target_clearance_m": float(data["final_non_target_clearance_m"][data_index]),
                }
            )
        rows = int(np.ceil(len(page_indices) / args.columns))
        destination = output_root / (
            f"scene_{args.scene:04d}_object_{args.object_id:03d}_"
            f"ranks_{(page_number - 1) * 25 + 1:02d}_"
            f"{(page_number - 1) * 25 + len(page_indices):02d}_"
            f"{args.columns}x{rows}.glb"
        )
        display.export(destination)
        page_paths.append(destination)
        print(f"page {page_number}: poses={len(page_indices)} geometries={len(display.geometry)}")
        print(f"output={destination}")
        if args.show == f"page{page_number}":
            display.show(caption=f"scene {args.scene:04d} object {args.object_id} top35 page {page_number}")

    index_path = output_root / (
        f"scene_{args.scene:04d}_object_{args.object_id:03d}_top{args.count}_index.json"
    )
    write_json_atomic(
        index_path,
        {
            "schema_version": 1,
            "scene_index": args.scene,
            "view_index": view_index,
            "target_segmentation_id": args.object_id,
            "prediction_source": str(prediction_path.relative_to(PROJECT_ROOT)),
            "ranking_rule": "descending score after balanced collision filtering",
            "layout": "page1=5x5 ranks1-25; page2=5x2 ranks26-35",
            "hand_display_geometry": "per-link convex hull; exact root transform and q20",
            "pages": [str(path.relative_to(PROJECT_ROOT)) for path in page_paths],
            "poses": index_records,
        },
    )
    print(f"index={index_path}")


if __name__ == "__main__":
    main()
