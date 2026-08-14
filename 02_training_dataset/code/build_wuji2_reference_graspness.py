#!/usr/bin/env python3
"""Stage 03: compute Wuji2 reference points and complete-surface graspness.

The cone allocation and decay equations reproduce
``DexGraspNet2/src/preprocess/dex_graspness.py``.  Hand-specific keypoints are
adapted explicitly: the Wuji2 semantic palm center is the cone origin, and the
midpoint of the URDF's empty thumb/middle tip-marker links defines its axis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ADAPTER_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config, write_json_atomic  # noqa: E402
from wuji2_dgn2.project import project_path  # noqa: E402


STAGE_02 = "02_scene_table_collision_filtered"
STAGE_NAME = "03_reference_points_and_surface_graspness"
SCHEMA_VERSION = 1
WUJI2_MODEL_PATH = Path(
    PROJECT_ROOT
    / "02_training_dataset/assets/wuji2_factory/04_pipeline/engine/"
    "configurable_object/grasp_generation/utils/wuji2_hand_model.py"
)


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
    parser.add_argument(
        "--selection-mask",
        choices=("paper_keep_mask", "wuji2_safe_keep_mask"),
        default=None,
        help="Choose which retained Stage-02 mask contributes training labels.",
    )
    return parser.parse_args()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def select_per_grasp_arrays(
    arrays: dict[str, np.ndarray], mask_name: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    total = int(arrays["qpos"].shape[0])
    if mask_name not in arrays:
        raise KeyError(
            f"Requested selection mask {mask_name!r} is absent; available keys: "
            f"{sorted(arrays)}"
        )
    mask = np.asarray(arrays[mask_name], dtype=bool)
    if mask.shape != (total,):
        raise RuntimeError(
            f"{mask_name} shape {mask.shape} does not match grasp count {total}"
        )
    indices = np.flatnonzero(mask).astype(np.int64)
    selected = {}
    non_row_keys = {
        "paper_keep_indices",
        "wuji2_safe_keep_indices",
        "scene_segmentation_ids",
        "scene_segmentation_ids_path_filter",
    }
    for key, value in arrays.items():
        if (
            key not in non_row_keys
            and value.ndim > 0
            and value.shape[0] == total
        ):
            selected[key] = value[mask]
        else:
            selected[key] = value
    selected["source_index_selected_from_retained"] = indices
    selected["training_selection_mask_name"] = np.asarray(mask_name)
    return selected, indices


def load_wuji2_module():
    name = "wuji2_hand_model_for_dgn2_graspness"
    spec = importlib.util.spec_from_file_location(name, WUJI2_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {WUJI2_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_surface_points(path: Path, count: int, seed: int) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise RuntimeError(f"Invalid surface samples: {path}")
    return points


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (
        points @ np.asarray(transform[:3, :3], dtype=np.float32).T
        + np.asarray(transform[:3, 3], dtype=np.float32)
    ).astype(np.float32)


def world_link_transforms_from_base_pose(model, base_pose, qpos):
    base_fk = model.forward_kinematics_base(qpos)
    world_wrist = base_pose @ torch.linalg.inv(base_fk["r_base_link"])
    transforms = {
        link: world_wrist @ wrist_link for link, wrist_link in base_fk.items()
    }
    if not torch.allclose(
        transforms["r_base_link"], base_pose, atol=2.0e-6, rtol=0.0
    ):
        raise RuntimeError("r_base_link to r_wrist bridge failed")
    return transforms


def transform_local_point(transform: torch.Tensor, local_xyz: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,j->bi", transform[:, :3, :3], local_xyz) + transform[:, :3, 3]


@torch.inference_mode()
def compute_keypoints(model, module, arrays, batch_size, device):
    total = int(arrays["qpos"].shape[0])
    palm_all, thumb_all, middle_all, midpoint_all = [], [], [], []
    palm_local = torch.tensor(
        module.PALM_CENTER_LOCAL_OFFSET, dtype=torch.float32, device=device
    )
    origin = torch.zeros(3, dtype=torch.float32, device=device)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        base_pose = torch.as_tensor(
            arrays["T_world_r_base_link"][start:stop], device=device
        )
        qpos = torch.as_tensor(arrays["qpos"][start:stop], device=device)
        transforms = world_link_transforms_from_base_pose(model, base_pose, qpos)
        palm = transform_local_point(transforms[module.PALM_FRAME_LINK], palm_local)
        thumb = transform_local_point(transforms["r_thumb_tip"], origin)
        middle = transform_local_point(transforms["r_middle_finger_tip"], origin)
        midpoint = 0.5 * (thumb + middle)
        palm_all.append(palm.cpu())
        thumb_all.append(thumb.cpu())
        middle_all.append(middle.cpu())
        midpoint_all.append(midpoint.cpu())
    return tuple(
        torch.cat(items, dim=0).numpy().astype(np.float32)
        for items in (palm_all, thumb_all, middle_all, midpoint_all)
    )


@torch.inference_mode()
def allocate_graspness(
    surface_points_world: np.ndarray,
    palm_center_world: np.ndarray,
    fingertip_midpoint_world: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    points = torch.as_tensor(surface_points_world, device=device)
    palms = torch.as_tensor(palm_center_world, device=device)
    midpoints = torch.as_tensor(fingertip_midpoint_world, device=device)
    total = int(palms.shape[0])
    result = torch.zeros(points.shape[0], dtype=torch.float32, device=device)
    best_indices = []
    batch_size = int(cfg["graspness_batch_size"])
    cone_angle = np.deg2rad(float(cfg["graspness_cone_half_angle_deg"]))
    height_band = float(cfg["graspness_cone_height_band_m"])
    angle_coef = -180.0 / np.pi * np.log(2.0) / float(
        cfg["graspness_angle_half_decay_deg"]
    )
    height_coef = -np.log(2.0) / float(cfg["graspness_height_half_decay_m"])
    radial_decay = float(cfg["graspness_radial_decay_per_m"])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        palm = palms[start:stop]
        midpoint = midpoints[start:stop]
        x_axis = midpoint - palm
        norm = torch.linalg.norm(x_axis, dim=1, keepdim=True)
        if (norm <= 1.0e-8).any():
            raise RuntimeError("Thumb-middle midpoint coincides with palm center")
        x_axis = x_axis / norm
        y_axis = torch.zeros_like(x_axis)
        least_aligned = x_axis.abs().argmin(dim=1)
        y_axis[torch.arange(len(x_axis), device=device), least_aligned] = 1.0
        y_axis = y_axis - (y_axis * x_axis).sum(dim=1, keepdim=True) * x_axis
        y_axis = y_axis / torch.linalg.norm(y_axis, dim=1, keepdim=True)
        z_axis = torch.cross(x_axis, y_axis, dim=1)
        rotation = torch.stack((x_axis, y_axis, z_axis), dim=2)
        local = torch.einsum(
            "bji,bnj->bni", rotation, points.unsqueeze(0) - palm.unsqueeze(1)
        )
        heights = local[:, :, 0]
        radius = torch.linalg.norm(local, dim=2).clamp_min(1.0e-12)
        angle = torch.arccos((heights.abs() / radius).clamp(0.0, 1.0))
        positive = (heights > 0.0) & (angle < cone_angle)
        valid_count = positive.sum(dim=1)
        if (valid_count == 0).any():
            bad = (valid_count == 0).nonzero(as_tuple=False)[:, 0] + start
            raise RuntimeError(
                "No complete-surface point lies in the graspness cone for "
                f"grasp indices {bad[:10].tolist()}"
            )
        masked_height = heights.masked_fill(~positive, 100.0)
        minimum_height = masked_height.amin(dim=1, keepdim=True)
        positive = positive & ((heights - minimum_height).abs() < height_band)
        raw = torch.exp(
            angle_coef * angle + height_coef * (heights - minimum_height)
        ) * positive
        best = raw.argmax(dim=1)
        best_indices.append(best.cpu())
        reference = points[best]
        distance = torch.linalg.norm(
            points.unsqueeze(0) - reference.unsqueeze(1), dim=2
        )
        result += torch.pow(10.0, -radial_decay * distance).sum(dim=0)
    return (
        result.cpu().numpy().astype(np.float32),
        torch.cat(best_indices).numpy().astype(np.int64),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cfg = config["grasp_label_generation"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    output_root = Path(config["paths"]["output_root"])
    enhanced_cfg = config.get("enhanced_palm_path_filter", {})
    input_stage = (
        str(enhanced_cfg["stage_name"])
        if enhanced_cfg.get("enabled")
        else STAGE_02
    )
    stage02_root = (
        output_root
        / cfg["stage_directory_name"]
        / input_stage
        / f"scene_{args.scene:04d}"
    )
    stage02_manifest_path = stage02_root / "stage_manifest.json"
    stage02 = json.loads(stage02_manifest_path.read_text(encoding="utf-8"))
    if stage02.get("diagnostic_prefix_limit") is not None:
        raise RuntimeError(f"Input stage {input_stage} is a prefix diagnostic")
    scene_path = output_root / "scenes" / f"scene_{args.scene:04d}" / "scene_manifest.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_objects = {int(x["segmentation_id"]): x for x in scene["objects"]}
    selection_mask = args.selection_mask or str(
        cfg.get(
            "training_selection_mask",
            "wuji2_safe_keep_mask"
            if enhanced_cfg.get("enabled")
            else "paper_keep_mask",
        )
    )
    module = load_wuji2_module()
    model = module.Wuji2HandKinematics(
        module.ORIGINAL_HAND_URDF, device=device, dtype=torch.float32
    )
    stage03_root = (
        output_root
        / cfg["stage_directory_name"]
        / STAGE_NAME
        / f"scene_{args.scene:04d}"
    )
    records = []
    total = 0
    for record in stage02["object_records"]:
        object_id = int(record["segmentation_id"])
        source_path = project_path(record["output_npz"])
        with np.load(source_path) as archive:
            retained_arrays = {key: archive[key] for key in archive.files}
        retained_count = int(retained_arrays["qpos"].shape[0])
        arrays, selected_indices = select_per_grasp_arrays(
            retained_arrays, selection_mask
        )
        count = int(arrays["qpos"].shape[0])
        scene_object = scene_objects[object_id]
        local_surface = deterministic_surface_points(
            project_path(scene_object["asset"]["centered_combined_obj"]),
            int(cfg["graspness_surface_points_per_object"]),
            int(cfg["collision_random_seed"]) + object_id,
        )
        surface_world = transform_points(
            np.asarray(scene_object["T_world_centered_object"], dtype=np.float32),
            local_surface,
        )
        if count:
            palm, thumb, middle, midpoint = compute_keypoints(
                model,
                module,
                arrays,
                int(cfg["graspness_batch_size"]),
                device,
            )
            graspness, best = allocate_graspness(
                surface_world, palm, midpoint, cfg, device
            )
            reference = surface_world[best]
        else:
            palm = np.empty((0, 3), np.float32)
            thumb = np.empty((0, 3), np.float32)
            middle = np.empty((0, 3), np.float32)
            midpoint = np.empty((0, 3), np.float32)
            best = np.empty((0,), np.int64)
            reference = np.empty((0, 3), np.float32)
            graspness = np.zeros(len(surface_world), np.float32)
        enriched = dict(arrays)
        enriched.update(
            {
                "point": reference,
                "reference_surface_index": best,
                "palm_center_world": palm,
                "thumb_tip_world": thumb,
                "middle_tip_world": middle,
                "fingertip_midpoint_world": midpoint,
            }
        )
        output_path = stage03_root / source_path.name
        surface_path = stage03_root / source_path.name.replace(
            ".npz", "_surface_graspness.npz"
        )
        atomic_savez(output_path, **enriched)
        atomic_savez(
            surface_path,
            surface_points_world=surface_world,
            surface_points_centered_object=local_surface,
            graspness=graspness,
        )
        records.append(
            {
                "segmentation_id": object_id,
                "object_code": record["object_code"],
                "retained_eligible_grasp_count": retained_count,
                "selection_mask": selection_mask,
                "grasp_count": count,
                "grasp_npz": str(output_path.resolve()),
                "surface_graspness_npz": str(surface_path.resolve()),
                "nonzero_surface_graspness_count": int((graspness > 0).sum()),
                "maximum_surface_graspness": float(graspness.max()),
            }
        )
        total += count
        print(
            f"[OBJECT {object_id:03d}] grasps={count} "
            f"nonzero_surface={int((graspness > 0).sum())} "
            f"max_graspness={float(graspness.max()):.6f}",
            flush=True,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "complete_surface_labels_intermediate_not_training_ready",
        "training_ready": False,
        "scene_index": int(args.scene),
        "scene_manifest": str(scene_path.resolve()),
        "input_stage_manifest": str(stage02_manifest_path.resolve()),
        "input_filter_stage": input_stage,
        "training_selection_mask": selection_mask,
        "selection_policy": (
            "Stage 01/02/02b retain all eligible grasps; Stage 03 is the first "
            "stage that materializes a chosen training subset"
        ),
        "official_benchmark": {
            "source": "DexGraspNet2/src/preprocess/dex_graspness.py",
            "surface_points_per_object": int(cfg["graspness_surface_points_per_object"]),
            "cone_half_angle_deg": float(cfg["graspness_cone_half_angle_deg"]),
            "cone_height_band_m": float(cfg["graspness_cone_height_band_m"]),
            "assignment_threshold_m_for_next_stage": float(
                cfg["graspness_assignment_threshold_m"]
            ),
        },
        "wuji2_keypoint_adaptation": {
            "cone_origin": "semantic palm center on r_wrist",
            "cone_axis_endpoint": "midpoint of r_thumb_tip and r_middle_finger_tip URDF marker origins",
            "reference_point": "best of 1000 complete target-object surface points inside the cone",
        },
        "remaining_required_stages": [
            "assign complete-surface graspness and grasps to each single-view point cloud"
        ],
        "object_records": records,
        "total_grasps": total,
    }
    manifest_path = stage03_root / "stage_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        f"[COMPLETE] scene={args.scene:04d} grasps={total} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
