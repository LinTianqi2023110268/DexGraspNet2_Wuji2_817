#!/usr/bin/env python3
"""用Trimesh查看已选抓取位姿。

输入：
  outputs/scene_xxxx/view_xxxx/selected_top2.npz。
  对应测试场景scene_manifest.json。

输出：
  visualizations/下的GLB；加入--show时同时打开交互窗口。

模式：
  overview：点云、6个物体以及全部12个掌心位置，不加载完整手。
  object：显示指定物体的pose 0和pose 1两只完整手。
  pose：只显示指定物体、指定rank的一只完整手。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.visual import (  # noqa: E402
    DEFAULT_URDF,
    PALETTE,
    Wuji2VisualModel,
    load_object_mesh,
)


SELECTION = PROJECT_ROOT / "07_wuji2_network_3p3r_sim/00_config/test_5scene.json"
OUTPUT_ROOT = PROJECT_ROOT / "07_wuji2_network_3p3r_sim/01_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--mode", choices=("overview", "object", "pose"), default="overview")
    parser.add_argument("--object-id", type=int, default=None)
    parser.add_argument("--pose-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--stage", choices=("pregrasp", "grasp"), default="grasp")
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def selected_case(scene_index: int) -> dict:
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    matches = [item for item in payload["scenes"] if int(item["scene_index"]) == scene_index]
    if len(matches) != 1:
        raise KeyError(f"scene {scene_index}不在test_5scene.json中")
    return matches[0]


def add_scene_geometry(display: trimesh.Scene, manifest: dict) -> None:
    table = trimesh.creation.box(extents=manifest["table"]["size_m"])
    table.apply_translation(
        [
            0.0,
            0.0,
            float(manifest["table"]["top_z_m"])
            - 0.5 * float(manifest["table"]["size_m"][2]),
        ]
    )
    table.visual.face_colors = [145, 145, 145, 70]
    display.add_geometry(table, geom_name="table")
    for item in manifest["objects"]:
        mesh = load_object_mesh(item["asset"])
        mesh.apply_transform(item["T_world_centered_object"])
        color = PALETTE[(int(item["segmentation_id"]) - 1) % len(PALETTE)].copy()
        color[3] = 70
        mesh.visual.face_colors = color
        display.add_geometry(
            mesh, geom_name=f"object_{int(item['segmentation_id']):03d}"
        )


def main() -> None:
    args = parse_args()
    case = selected_case(args.scene)
    view = int(case["view_index"])
    case_root = OUTPUT_ROOT / f"scene_{args.scene:04d}_view_{view:04d}"
    prediction_root = case_root / "02_predictions"
    visual_root = case_root / "03_visualization"
    selected_path = prediction_root / "selected_top2.npz"
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"缺少{selected_path}；请先运行02_select_top2_per_object.py"
        )
    with np.load(selected_path, allow_pickle=False) as archive:
        selected = {key: archive[key] for key in archive.files}
    manifest_path = PROJECT_ROOT / case["scene_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    display = trimesh.Scene()
    display.add_geometry(
        trimesh.creation.axis(origin_size=0.004, axis_length=0.10),
        geom_name="world_frame",
    )
    add_scene_geometry(display, manifest)

    points = np.asarray(selected["point_cloud_world"], dtype=np.float64)
    segmentation = np.asarray(
        selected["ground_truth_segmentation"], dtype=np.int64
    )
    if args.max_points > 0 and len(points) > args.max_points:
        shown = np.linspace(0, len(points) - 1, args.max_points, dtype=np.int64)
        points = points[shown]
        segmentation = segmentation[shown]
    point_colors = np.full((len(points), 4), [120, 120, 120, 80], dtype=np.uint8)
    for object_id in np.unique(segmentation):
        if object_id <= 0:
            continue
        color = PALETTE[(int(object_id) - 1) % len(PALETTE)].copy()
        color[3] = 220
        point_colors[segmentation == object_id] = color
    display.add_geometry(
        trimesh.points.PointCloud(points, colors=point_colors),
        geom_name="single_view_point_cloud",
    )

    target = np.asarray(selected["target_segmentation_id"], dtype=np.int64)
    pose_rank = np.asarray(selected["pose_rank_within_object"], dtype=np.int64)
    if args.mode == "overview":
        palms = np.asarray(selected["palm_center_grasp_world"], dtype=np.float64)
        for index, palm in enumerate(palms):
            color = [255, 210, 0, 255] if int(pose_rank[index]) == 0 else [255, 80, 220, 255]
            marker = trimesh.creation.icosphere(radius=0.005, subdivisions=2)
            marker.apply_translation(palm)
            marker.visual.face_colors = color
            display.add_geometry(
                marker,
                geom_name=(
                    f"object_{int(target[index]):03d}_pose_{int(pose_rank[index])}_palm"
                ),
            )
        suffix = "overview"
    else:
        if args.object_id is None:
            raise ValueError("--mode object/pose时必须指定--object-id")
        indices = np.flatnonzero(target == args.object_id)
        if args.mode == "pose":
            indices = indices[pose_rank[indices] == args.pose_rank]
        if not len(indices):
            raise KeyError(
                f"scene={args.scene} object={args.object_id} pose={args.pose_rank}不存在"
            )
        model = Wuji2VisualModel(DEFAULT_URDF)
        joint_order = [str(value) for value in selected["joint_order"].tolist()]
        for index in indices:
            rank = int(pose_rank[index])
            if args.stage == "pregrasp":
                transform = np.asarray(
                    selected["pregrasp_T_world_r_base_link"][index], dtype=np.float64
                )
                q_values = selected["pregrasp_qpos"][index]
            else:
                transform = np.asarray(
                    selected["T_world_r_base_link"][index], dtype=np.float64
                )
                q_values = selected["qpos"][index]
            qpos = dict(zip(joint_order, np.asarray(q_values, float).tolist()))
            hand_color = np.asarray(
                [40, 220, 100, 170] if rank == 0 else [70, 140, 255, 170]
            )
            for name, mesh in model.scene_geometry(
                transform,
                qpos,
                hand_color,
                f"object_{args.object_id:03d}_pose_{rank}_{args.stage}",
            ):
                display.add_geometry(mesh, geom_name=name)
            display.add_geometry(
                trimesh.creation.axis(
                    origin_size=0.0025, axis_length=0.045, transform=transform
                ),
                geom_name=f"hand_root_object_{args.object_id:03d}_pose_{rank}",
            )
        suffix = (
            f"object_{args.object_id:03d}_{args.stage}"
            if args.mode == "object"
            else f"object_{args.object_id:03d}_pose_{args.pose_rank}_{args.stage}"
        )

    destination = (
        args.output.resolve()
        if args.output is not None
        else visual_root / f"scene_{args.scene:04d}_view_{view:04d}_{suffix}.glb"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    display.export(destination)
    print(
        f"scene={args.scene:04d} view={view:04d} mode={args.mode} "
        f"geometries={len(display.geometry)}"
    )
    print(f"output={destination}")
    if args.show:
        display.show(caption=f"Wuji2 test scene {args.scene:04d} {suffix}")


if __name__ == "__main__":
    main()
