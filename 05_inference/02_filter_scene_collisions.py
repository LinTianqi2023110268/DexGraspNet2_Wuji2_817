#!/usr/bin/env python3
"""Filter Wuji2 network predictions before physics execution.

The first two masks reproduce the two collision policies present in the
DexGraspNet2 source:

* ``official_endpoint_keep`` is SimulationEvaluator's zero-penetration rule.
* ``strict_training_keep`` is dex_graspness.py's 2.5 mm clearance rule.

The final mask additionally applies the explicitly marked Wuji2 enhancement:
the zero-radius semantic-palm centerline along the reviewed tiger-mouth
PREGRASP-to-GRASP path may not cross
the table clearance plane or a non-target object triangle mesh.  This is not
claimed to be part of the paper and is not a full articulated swept-volume
test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config, write_json_atomic  # noqa: E402
from wuji2_dgn2.palm_path import (  # noqa: E402
    base_to_semantic_palm_center,
    load_world_mesh,
    segment_mesh_intersections,
    transform_one_point,
)
from wuji2_dgn2.collision import (  # noqa: E402
    build_scene_object_points,
    collision_metrics,
    load_hand_link_vertices,
    load_wuji2_module,
    minimum_table_clearance,
    world_link_transforms_from_base_pose,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "02_training_dataset/config/wuji2_train60_100seminal_256view_v1.json"
)
DEFAULT_PREDICTION = (
    SCRIPT_DIR / "outputs/scene_0036_view_0000_all.npz"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "outputs/scene_0036_view_0000_collision_filtered.npz"
)
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Check only the highest-scoring N candidates for a safe diagnostic run.",
    )
    parser.add_argument(
        "--collision-batch-size",
        type=int,
        default=None,
        help="Override the config batch size; smaller values reduce peak GPU memory.",
    )
    return parser.parse_args()


def resolve_adapter_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def candidate_subset(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    total = int(arrays["qpos"].shape[0])
    result: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key == "ranking":
            continue
        if value.ndim > 0 and value.shape[0] == total:
            result[key] = value[indices]
        else:
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    prediction_path = resolve_adapter_path(args.prediction)
    output_path = resolve_adapter_path(args.output)
    config_path = resolve_adapter_path(args.config)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")

    config = load_config(config_path)
    label_cfg = config["grasp_label_generation"]
    enhanced_cfg = config["enhanced_palm_path_filter"]
    if not enhanced_cfg.get("enabled", False):
        raise RuntimeError("enhanced_palm_path_filter is disabled in the config")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    with np.load(prediction_path) as archive:
        source = {key: archive[key] for key in archive.files}
    required = {"qpos", "score", "joint_order", "scene_index", "view_index"}
    missing = sorted(required.difference(source))
    if missing:
        raise KeyError(f"Prediction is missing keys: {missing}")
    total = int(source["qpos"].shape[0])
    if source["qpos"].shape[1] != 20:
        raise RuntimeError("Expected 20 Wuji2 joint values")

    score_order = np.argsort(-source["score"]).astype(np.int64)
    tested_source_indices = score_order[
        : total if args.max_candidates is None else min(args.max_candidates, total)
    ]
    arrays = candidate_subset(source, tested_source_indices)
    tested = len(tested_source_indices)
    if "T_world_r_base_link" in arrays:
        final_pose = np.asarray(arrays["T_world_r_base_link"], dtype=np.float32)
    elif "rotation_world" in arrays and "translation_world" in arrays:
        final_pose = np.repeat(np.eye(4, dtype=np.float32)[None], tested, axis=0)
        final_pose[:, :3, :3] = arrays["rotation_world"]
        final_pose[:, :3, 3] = arrays["translation_world"]
    else:
        raise KeyError(
            "Input needs T_world_r_base_link or rotation_world+translation_world"
        )
    collision_arrays = {
        "T_world_r_base_link": final_pose,
        "qpos": np.asarray(arrays["qpos"], dtype=np.float32),
    }

    scene_index = int(np.asarray(source["scene_index"]).item())
    view_index = int(np.asarray(source["view_index"]).item())
    output_root = Path(config["paths"]["output_root"])
    scene_path = output_root / "scenes" / f"scene_{scene_index:04d}" / "scene_manifest.json"
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_objects = {int(record["segmentation_id"]): record for record in scene["objects"]}
    scene_ids = np.asarray(sorted(scene_objects), dtype=np.int64)

    module = load_wuji2_module()
    if list(np.asarray(source["joint_order"]).tolist()) != list(module.RIGHT_HAND_JOINT_ORDER):
        raise RuntimeError("Prediction joint order differs from the Wuji2 hand model")
    model = module.Wuji2HandKinematics(
        module.ORIGINAL_HAND_URDF, device=device, dtype=torch.float32
    )
    local_hand_vertices = load_hand_link_vertices(module, device)
    surface_count = int(label_cfg["collision_surface_points_per_object"])
    scene_points_cpu = build_scene_object_points(
        scene, surface_count, int(label_cfg["collision_random_seed"])
    )
    batch_size = int(
        args.collision_batch_size
        if args.collision_batch_size is not None
        else label_cfg["collision_grasp_batch_size"]
    )
    if batch_size <= 0:
        raise ValueError("collision batch size must be positive")
    table_top_z = float(scene["table"]["top_z_m"])

    if "seed_segmentation" in arrays:
        target_id = np.asarray(arrays["seed_segmentation"], dtype=np.int64)
    elif "target_segmentation_id" in arrays:
        target_id = np.asarray(arrays["target_segmentation_id"], dtype=np.int64)
    else:
        raise KeyError("Input needs seed_segmentation or target_segmentation_id")
    selected_source_candidate = np.asarray(
        arrays.get("source_candidate_index", tested_source_indices),
        dtype=np.int64,
    )
    (
        scene_clearance,
        table_clearance,
        per_object_clearance,
        pregrasp_pose,
        pregrasp_qpos,
    ) = collision_metrics(
        model=model,
        module=module,
        arrays=collision_arrays,
        target_id=0,
        scene_points_cpu=scene_points_cpu,
        local_hand_vertices=local_hand_vertices,
        table_top_z=table_top_z,
        batch_size=batch_size,
        device=device,
        label_cfg=label_cfg,
        approach_mode="tiger_mouth",
    )

    official_endpoint_keep = (scene_clearance > 0.0) & (table_clearance > 0.0)
    strict_clearance = float(label_cfg["scene_and_table_clearance_m"])
    strict_training_keep = (
        (scene_clearance > strict_clearance)
        & (table_clearance > strict_clearance)
    )

    # The official pregrasp check alone can admit a final hand pose that is
    # several centimetres through the table.  Target-object contact is allowed,
    # but the final hand must clear the table and every non-target object.
    final_table_clearance = np.empty(tested, dtype=np.float32)
    final_per_object_clearance = np.full(
        (tested, len(scene_ids)), np.nan, dtype=np.float32
    )
    final_non_target_clearance = np.full(tested, np.inf, dtype=np.float32)
    for start in range(0, tested, batch_size):
        stop = min(start + batch_size, tested)
        transforms = world_link_transforms_from_base_pose(
            model,
            torch.as_tensor(final_pose[start:stop], device=device),
            torch.as_tensor(collision_arrays["qpos"][start:stop], device=device),
        )
        final_table_clearance[start:stop] = (
            minimum_table_clearance(transforms, local_hand_vertices, table_top_z)
            .cpu().numpy()
        )
        for column, object_id in enumerate(scene_ids.tolist()):
            signed = model._batch_signed_distance_to_hand_transforms(
                scene_points_cpu[object_id].to(device), transforms, stop - start
            )
            clearance = (-signed.amax(dim=1)).cpu().numpy()
            final_per_object_clearance[start:stop, column] = clearance
            rows = np.arange(start, stop)
            non_target = target_id[start:stop] != object_id
            final_non_target_clearance[rows[non_target]] = np.minimum(
                final_non_target_clearance[rows[non_target]], clearance[non_target]
            )
    final_table_keep = final_table_clearance > 0.0
    final_non_target_keep = final_non_target_clearance > 0.0

    valid_target = np.isin(target_id, scene_ids)
    palm_in_base = base_to_semantic_palm_center(module)
    final_palm = transform_one_point(final_pose.astype(np.float64), palm_in_base)
    pregrasp_palm = transform_one_point(pregrasp_pose.astype(np.float64), palm_in_base)
    path_minimum_z = np.minimum(final_palm[:, 2], pregrasp_palm[:, 2])
    path_minimum_allowed_z = float(
        table_top_z + enhanced_cfg["minimum_table_clearance_m"]
    )
    path_table_keep = path_minimum_z > path_minimum_allowed_z

    hit_matrix = np.zeros((tested, len(scene_ids)), dtype=bool)
    if enhanced_cfg.get("check_non_target_object_mesh_intersection", True):
        for column, object_id in enumerate(scene_ids.tolist()):
            mesh = load_world_mesh(scene_objects[object_id])
            hit = segment_mesh_intersections(pregrasp_palm, final_palm, mesh)
            # Contact/enclosure of the intended target is allowed.
            hit[target_id == object_id] = False
            hit_matrix[:, column] = hit
    path_other_object_keep = ~hit_matrix.any(axis=1)
    enhanced_keep = (
        valid_target
        & strict_training_keep
        & path_table_keep
        & path_other_object_keep
        & final_table_keep
        & final_non_target_keep
    )

    keep_positions = np.flatnonzero(enhanced_keep)
    filtered: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if value.ndim > 0 and value.shape[0] == tested:
            filtered[key] = value[keep_positions]
        else:
            filtered[key] = value
    filtered.update(
        {
            "ranking": np.arange(len(keep_positions), dtype=np.int64),
            "source_candidate_index": selected_source_candidate[keep_positions],
            "target_segmentation_id": target_id[keep_positions],
            "T_world_r_base_link": final_pose[keep_positions],
            "pregrasp_T_world_r_base_link": pregrasp_pose[keep_positions],
            "pregrasp_qpos": pregrasp_qpos[keep_positions],
            "minimum_scene_clearance_m": scene_clearance[keep_positions],
            "minimum_table_clearance_m": table_clearance[keep_positions],
            "scene_clearance_by_segmentation_m": per_object_clearance[keep_positions],
            "final_table_clearance_m": final_table_clearance[keep_positions],
            "final_non_target_clearance_m": final_non_target_clearance[keep_positions],
            "final_clearance_by_segmentation_m": final_per_object_clearance[keep_positions],
            "palm_center_pregrasp_world": pregrasp_palm[keep_positions].astype(np.float32),
            "palm_center_grasp_world": final_palm[keep_positions].astype(np.float32),
            "palm_path_minimum_world_z_m": path_minimum_z[keep_positions].astype(np.float32),
            "scene_segmentation_ids_collision": scene_ids,
        }
    )
    atomic_savez(output_path, **filtered)

    diagnostics_path = output_path.with_name(output_path.stem + "_all_diagnostics.npz")
    atomic_savez(
        diagnostics_path,
        source_candidate_index=selected_source_candidate,
        target_segmentation_id=target_id,
        valid_target_mask=valid_target,
        official_endpoint_keep_mask=official_endpoint_keep,
        strict_training_keep_mask=strict_training_keep,
        enhanced_keep_mask=enhanced_keep,
        minimum_scene_clearance_m=scene_clearance,
        minimum_table_clearance_m=table_clearance,
        scene_clearance_by_segmentation_m=per_object_clearance,
        final_table_clearance_m=final_table_clearance,
        final_non_target_clearance_m=final_non_target_clearance,
        final_clearance_by_segmentation_m=final_per_object_clearance,
        final_table_keep_mask=final_table_keep,
        final_non_target_keep_mask=final_non_target_keep,
        scene_segmentation_ids=scene_ids,
        pregrasp_T_world_r_base_link=pregrasp_pose,
        pregrasp_qpos=pregrasp_qpos,
        palm_center_pregrasp_world=pregrasp_palm.astype(np.float32),
        palm_center_grasp_world=final_palm.astype(np.float32),
        palm_path_minimum_world_z_m=path_minimum_z.astype(np.float32),
        palm_path_table_keep_mask=path_table_keep,
        palm_path_other_object_keep_mask=path_other_object_keep,
        palm_path_intersection_by_segmentation=hit_matrix,
    )

    rejection_counts = {
        "invalid_or_background_target": int((~valid_target).sum()),
        "official_scene_collision": int((scene_clearance <= 0.0).sum()),
        "official_table_collision": int((table_clearance <= 0.0).sum()),
        "strict_scene_clearance": int((scene_clearance <= strict_clearance).sum()),
        "strict_table_clearance": int((table_clearance <= strict_clearance).sum()),
        "enhanced_palm_path_table": int((~path_table_keep).sum()),
        "enhanced_palm_path_non_target_object": int((~path_other_object_keep).sum()),
        "enhanced_final_table_collision": int((~final_table_keep).sum()),
        "enhanced_final_non_target_collision": int((~final_non_target_keep).sum()),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "wuji2_predictions_collision_filtered_not_yet_simulated",
        "scene_index": scene_index,
        "view_index": view_index,
        "input_prediction": str(prediction_path),
        "scene_manifest": str(scene_path.resolve()),
        "tested_candidates": tested,
        "input_candidates": total,
        "diagnostic_prefix_limit": args.max_candidates,
        "tested_in_descending_network_score_order": True,
        "official_endpoint_kept": int(official_endpoint_keep.sum()),
        "strict_training_clearance_kept": int(strict_training_keep.sum()),
        "enhanced_final_kept": int(enhanced_keep.sum()),
        "rejection_counts_not_mutually_exclusive": rejection_counts,
        "policies": {
            "official_simulation_evaluator": "opened/retreated pregrasp; scene and table signed penetration < 0",
            "official_training_label": f"opened/retreated pregrasp; scene and table clearance > {strict_clearance} m",
            "wuji2_enhancement": "strict pregrasp clearance, valid target, palm path clearance, and full final-hand clearance from table and non-target objects; target-object contact remains allowed",
            "pregrasp": (
                "open fingertips "
                f"{float(label_cfg['pregrasp_fingertip_opening_m']):.4f} m "
                "with configured-step IK and retreat root "
                f"{float(label_cfg['pregrasp_retreat_m']):.4f} m opposite approach"
            ),
            "approach": "Wuji2 semantic-palm center toward the current GRASP thumb/index-tip midpoint",
            "target_assignment": "seed_segmentation from the labeled synthetic view; background ID 0 is invalid",
            "guarantee_limit": "endpoint hand collision plus palm centerline only; not a full articulated swept-volume proof",
        },
        "output_npz": str(output_path),
        "diagnostics_npz": str(diagnostics_path),
    }
    summary_path = output_path.with_suffix(".json")
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
