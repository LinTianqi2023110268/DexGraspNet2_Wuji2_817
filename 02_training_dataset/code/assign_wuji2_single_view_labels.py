#!/usr/bin/env python3
"""Stage 04: map complete-surface Wuji2 labels to configured single-view inputs.

This reproduces the final nearest-neighbour assignment in
``DexGraspNet2/src/preprocess/dex_graspness.py`` and records which scene
grasps can be matched to each visible single-view point within the official
6 mm training-center limit from ``src/utils/dataset.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ADAPTER_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config, write_json_atomic  # noqa: E402
from wuji2_dgn2.project import project_path  # noqa: E402


STAGE_03 = "03_reference_points_and_surface_graspness"
STAGE_NAME = "04_single_view_training_labels"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ADAPTER_ROOT
        / "config"
        / "wuji2_train60_100seminal_256view_v1.json",
    )
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def camera_from_world(world_from_camera: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(world_from_camera, dtype=np.float64)).astype(
        np.float32
    )


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (
        np.asarray(points, dtype=np.float32) @ transform[:3, :3].T
        + transform[:3, 3]
    ).astype(np.float32)


@torch.inference_mode()
def nearest(
    query: np.ndarray, reference: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    if device.type == "cpu":
        # ``knn_points`` is efficient on CUDA but its CPU implementation is
        # needlessly slow for Stage04.  cKDTree returns the same exact L2
        # nearest-neighbour contract without changing any label threshold.
        distance, indices = cKDTree(
            np.asarray(reference, dtype=np.float64)
        ).query(np.asarray(query, dtype=np.float64), k=1, workers=1)
        return (
            np.asarray(distance, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
        )
    query_tensor = torch.as_tensor(query, device=device).unsqueeze(0)
    reference_tensor = torch.as_tensor(reference, device=device).unsqueeze(0)
    squared, indices, _ = knn_points(query_tensor, reference_tensor, K=1)
    distance = torch.sqrt(squared[0, :, 0].clamp_min(0.0))
    return distance.cpu().numpy(), indices[0, :, 0].cpu().numpy()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    label_cfg = config["grasp_label_generation"]
    train_cfg = config["network_training"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    output_root = Path(config["paths"]["output_root"])
    scene_root = output_root / "scenes" / f"scene_{args.scene:04d}"
    network_path = scene_root / "network_input.npz"
    with np.load(network_path) as archive:
        network = {key: archive[key] for key in archive.files}
    expected = (
        int(config["scope"]["views_per_scene"]),
        int(config["scope"]["points_per_view"]),
    )
    if network["pc"].shape[:2] != expected or network["seg"].shape != expected:
        raise RuntimeError(f"network_input shape mismatch: {network['pc'].shape}")
    stage03_root = (
        output_root
        / label_cfg["stage_directory_name"]
        / STAGE_03
        / f"scene_{args.scene:04d}"
    )
    stage03_manifest_path = stage03_root / "stage_manifest.json"
    stage03 = json.loads(stage03_manifest_path.read_text(encoding="utf-8"))
    object_labels = {}
    surface_points, surface_graspness, surface_ids = [], [], []
    for record in stage03["object_records"]:
        object_id = int(record["segmentation_id"])
        with np.load(project_path(record["grasp_npz"])) as archive:
            grasp = {key: archive[key] for key in archive.files}
        with np.load(project_path(record["surface_graspness_npz"])) as archive:
            surface = {key: archive[key] for key in archive.files}
        object_labels[object_id] = grasp
        if int(record["grasp_count"]) > 0:
            points = surface["surface_points_world"].astype(np.float32)
            surface_points.append(points)
            surface_graspness.append(surface["graspness"].astype(np.float32))
            surface_ids.append(np.full(len(points), object_id, dtype=np.int64))
    if not surface_points:
        raise RuntimeError("No collision-free grasp labels are available in this scene")
    perfect_points_world = np.concatenate(surface_points)
    perfect_graspness = np.concatenate(surface_graspness)
    perfect_ids = np.concatenate(surface_ids)
    assignment_threshold = float(label_cfg["graspness_assignment_threshold_m"])
    center_threshold = float(
        train_cfg["maximum_reference_to_view_point_distance_m"]
    )
    stage04_root = (
        output_root
        / label_cfg["stage_directory_name"]
        / STAGE_NAME
        / f"scene_{args.scene:04d}"
    )
    view_records = []
    total_matched_points = 0
    for view in range(expected[0]):
        cloud_camera = network["pc"][view].astype(np.float32)
        segmentation = network["seg"][view].astype(np.int64)
        world_from_camera = network["extrinsics"][view].astype(np.float32)
        world_cloud = transform_points(world_from_camera, cloud_camera)
        distance, index = nearest(world_cloud, perfect_points_world, device)
        same_object = segmentation == perfect_ids[index]
        assigned = (distance < assignment_threshold) & same_object
        raw_graspness = np.zeros(expected[1], dtype=np.float32)
        raw_graspness[assigned] = perfect_graspness[index[assigned]]
        objectness = (segmentation > 0).astype(np.int64)
        available_by_object = {}
        nearest_center_by_object = {}
        camera_from_world_matrix = camera_from_world(world_from_camera)
        for object_id, grasp in object_labels.items():
            if not len(grasp["point"]):
                available_by_object[str(object_id)] = np.empty((0,), np.int64)
                nearest_center_by_object[str(object_id)] = np.empty((0,), np.int64)
                continue
            reference_camera = transform_points(
                camera_from_world_matrix, grasp["point"]
            )
            candidate_mask = segmentation == object_id
            candidate_indices = np.flatnonzero(candidate_mask)
            if not len(candidate_indices):
                available_by_object[str(object_id)] = np.empty((0,), np.int64)
                nearest_center_by_object[str(object_id)] = np.empty((0,), np.int64)
                continue
            center_distance, local_index = nearest(
                reference_camera, cloud_camera[candidate_mask], device
            )
            available = np.flatnonzero(center_distance <= center_threshold).astype(
                np.int64
            )
            centers = candidate_indices[local_index[available]].astype(np.int64)
            available_by_object[str(object_id)] = available
            nearest_center_by_object[str(object_id)] = centers
        available_counts = {
            key: int(len(value)) for key, value in available_by_object.items()
        }
        view_path = stage04_root / f"view_{view:04d}.npz"
        arrays = {
            "point_clouds": cloud_camera,
            "coors": cloud_camera / float(train_cfg["voxel_size_m"]),
            "feats": np.ones_like(cloud_camera, dtype=np.float32),
            "seg": segmentation,
            "objectness": objectness,
            "graspness_raw": raw_graspness,
            "graspness_log_target": np.log(raw_graspness + 1.0e-3).astype(
                np.float32
            ),
            "edge": network["edge"][view].astype(np.int64),
            "T_world_camera": world_from_camera,
        }
        for object_id in sorted(object_labels):
            arrays[f"object_{object_id:03d}_available_grasp_indices"] = (
                available_by_object[str(object_id)]
            )
            arrays[f"object_{object_id:03d}_center_point_indices"] = (
                nearest_center_by_object[str(object_id)]
            )
        atomic_savez(view_path, **arrays)
        total_matched_points += int(assigned.sum())
        view_records.append(
            {
                "view_index": view,
                "reference_official_view_index": int(
                    config["camera"]["reference_view_indices"][view]
                ),
                "output_npz": str(view_path.resolve()),
                "visible_object_point_count": int(objectness.sum()),
                "graspness_assigned_point_count": int(assigned.sum()),
                "available_grasp_count_by_object": available_counts,
                "total_available_grasp_count": int(sum(available_counts.values())),
            }
        )
        print(
            f"[VIEW {view:04d}] object_points={int(objectness.sum())} "
            f"graspness_points={int(assigned.sum())} "
            f"available_grasps={int(sum(available_counts.values()))}",
            flush=True,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "single_scene_training_labels_ready",
        "training_ready": True,
        "scope_note": "One scene of the configured multi-scene/multi-view dataset; view count is read from the active config.",
        "scene_index": int(args.scene),
        "network_input": str(network_path.resolve()),
        "input_stage_manifest": str(stage03_manifest_path.resolve()),
        "official_benchmark": {
            "graspness_source": "DexGraspNet2/src/preprocess/dex_graspness.py",
            "loader_source": "DexGraspNet2/src/utils/dataset.py",
            "points_per_view": expected[1],
            "voxel_size_m": float(train_cfg["voxel_size_m"]),
            "graspness_assignment_distance_m": assignment_threshold,
            "grasp_center_match_distance_m": center_threshold,
            "graspness_training_target": "log(raw_graspness + 1e-3)",
        },
        "source_code_distance_note": "pytorch3d.knn_points returns squared L2 distance, while official dex_graspness.py compares it directly with 0.015 despite describing a 1.5 cm threshold. This adapter takes sqrt first so 0.015 retains its documented meter meaning.",
        "coordinate_contract": {
            "point_clouds": "OpenCV camera frame",
            "surface_and_grasp_labels": "world frame in Stage 03",
            "assignment": "single-view camera points transformed by T_world_camera before world-frame nearest-neighbour matching",
        },
        "view_records": view_records,
        "total_graspness_assigned_points_over_views": total_matched_points,
    }
    manifest_path = stage04_root / "stage_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"[COMPLETE] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
