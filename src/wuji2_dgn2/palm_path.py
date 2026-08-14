#!/usr/bin/env python3
"""Stage 02B: enhanced semantic-palm centerline path filtering.

This is deliberately separate from the published DexGraspNet2 filter.  For
each paper-filtered grasp, it joins the semantic palm center at pregrasp and at
the final grasp with a finite line segment.  The segment must stay above the
table and must not intersect any non-target object triangle mesh.

Passing this stage does not prove that the complete articulated hand swept
volume is collision-free; it certifies only the requested zero-radius palm
reference-point path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import transforms3d
import trimesh


from .adapter_common import load_config, write_json_atomic
from .project import PROJECT_ROOT, project_path, source_path

ADAPTER_ROOT = PROJECT_ROOT


STAGE_02 = "02_scene_table_collision_filtered"
SCHEMA_VERSION = 1
OUTPUT_POLICY_REVISION = "wuji2-nondestructive-path-mask-v1"
WUJI2_MODEL_PATH = source_path("wuji2_factory") / (
    "04_pipeline/engine/configurable_object/grasp_generation/utils/"
    "wuji2_hand_model.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ADAPTER_ROOT
        / "configs"
        / "wuji2_train60_100seminal_256view_v1.json",
    )
    parser.add_argument("--scene", type=int, default=0)
    return parser.parse_args()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_wuji2_module():
    name = "wuji2_hand_model_for_palm_path_adapter"
    spec = importlib.util.spec_from_file_location(name, WUJI2_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {WUJI2_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected xyz/rpy triplet, got {text!r}")
    return values


def base_to_semantic_palm_center(module) -> np.ndarray:
    """Return the semantic palm origin expressed in r_base_link."""
    root = ET.parse(module.ORIGINAL_HAND_URDF).getroot()
    fixed = next(
        (
            joint
            for joint in root.findall("joint")
            if joint.attrib.get("name") == "r_wrist_fixed"
        ),
        None,
    )
    if fixed is None:
        raise RuntimeError("Wuji2 URDF has no r_wrist_fixed joint")
    origin = fixed.find("origin")
    xyz = parse_xyz(origin.attrib.get("xyz") if origin is not None else None)
    rpy = parse_xyz(origin.attrib.get("rpy") if origin is not None else None)
    base_to_wrist = np.eye(4, dtype=np.float64)
    base_to_wrist[:3, :3] = transforms3d.euler.euler2mat(*rpy, axes="sxyz")
    base_to_wrist[:3, 3] = xyz
    wrist_palm = np.asarray(
        [*module.PALM_CENTER_LOCAL_OFFSET, 1.0], dtype=np.float64
    )
    point = base_to_wrist @ wrist_palm
    if not np.isclose(point[3], 1.0):
        raise RuntimeError("Invalid homogeneous semantic palm point")
    return point[:3]


def transform_one_point(transforms: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.einsum("nij,j->ni", transforms[:, :3, :3], point) + transforms[:, :3, 3]


def load_world_mesh(scene_object: dict) -> trimesh.Trimesh:
    mesh = trimesh.load(
        project_path(scene_object["asset"]["centered_combined_obj"], must_exist=True),
        force="mesh",
        process=False,
    )
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.apply_transform(
        np.asarray(scene_object["T_world_centered_object"], dtype=np.float64)
    )
    if not len(mesh.faces):
        raise RuntimeError(f"Object mesh has no faces: {scene_object['object_code']}")
    return mesh


def segment_mesh_intersections(
    starts: np.ndarray,
    ends: np.ndarray,
    mesh: trimesh.Trimesh,
    batch_size: int = 128,
    epsilon: float = 1.0e-10,
) -> np.ndarray:
    """Exact finite-segment/triangle test via batched Moller-Trumbore."""
    starts = np.asarray(starts, dtype=np.float64)
    directions = np.asarray(ends, dtype=np.float64) - starts
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    vertex0 = triangles[:, 0]
    edge1 = triangles[:, 1] - vertex0
    edge2 = triangles[:, 2] - vertex0
    result = np.zeros(len(starts), dtype=bool)
    for begin in range(0, len(starts), batch_size):
        stop = min(begin + batch_size, len(starts))
        origin = starts[begin:stop]
        direction = directions[begin:stop]
        pvec = np.cross(direction[:, None, :], edge2[None, :, :])
        determinant = np.einsum("fj,bfj->bf", edge1, pvec)
        nonsingular = np.abs(determinant) > epsilon
        inverse = np.zeros_like(determinant)
        inverse[nonsingular] = 1.0 / determinant[nonsingular]
        tvec = origin[:, None, :] - vertex0[None, :, :]
        barycentric_u = np.einsum("bfj,bfj->bf", tvec, pvec) * inverse
        qvec = np.cross(tvec, edge1[None, :, :])
        barycentric_v = np.einsum("bj,bfj->bf", direction, qvec) * inverse
        segment_t = np.einsum("fj,bfj->bf", edge2, qvec) * inverse
        hit = (
            nonsingular
            & (barycentric_u >= -epsilon)
            & (barycentric_v >= -epsilon)
            & (barycentric_u + barycentric_v <= 1.0 + epsilon)
            & (segment_t >= -epsilon)
            & (segment_t <= 1.0 + epsilon)
        )
        result[begin:stop] = hit.any(axis=1)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    enhanced = config["enhanced_palm_path_filter"]
    if not enhanced.get("enabled"):
        raise RuntimeError("enhanced_palm_path_filter is disabled")
    output_root = Path(config["paths"]["output_root"])
    label_root = output_root / config["grasp_label_generation"]["stage_directory_name"]
    stage02_root = label_root / STAGE_02 / f"scene_{args.scene:04d}"
    stage02_manifest_path = stage02_root / "stage_manifest.json"
    stage02 = json.loads(stage02_manifest_path.read_text(encoding="utf-8"))
    if stage02.get("diagnostic_prefix_limit") is not None:
        raise RuntimeError("Stage 02 is only a prefix diagnostic")
    scene_path = output_root / "scenes" / f"scene_{args.scene:04d}" / "scene_manifest.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_objects = {int(item["segmentation_id"]): item for item in scene["objects"]}
    object_column = {
        object_id: column for column, object_id in enumerate(sorted(scene_objects))
    }
    world_meshes = {
        object_id: load_world_mesh(record)
        for object_id, record in scene_objects.items()
    }
    module = load_wuji2_module()
    palm_in_base = base_to_semantic_palm_center(module)
    stage_name = str(enhanced["stage_name"])
    stage02b_root = label_root / stage_name / f"scene_{args.scene:04d}"
    table_top_z = float(scene["table"]["top_z_m"])
    table_clearance = float(enhanced["minimum_table_clearance_m"])
    minimum_z = table_top_z + table_clearance
    records = []
    total_input = 0
    total_kept = 0
    for record in stage02["object_records"]:
        object_id = int(record["segmentation_id"])
        source_path = project_path(record["output_npz"])
        with np.load(source_path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        count = int(arrays["qpos"].shape[0])
        if "paper_keep_mask" in arrays:
            paper_keep = np.asarray(arrays["paper_keep_mask"], dtype=bool)
            if paper_keep.shape != (count,):
                raise RuntimeError(
                    f"paper_keep_mask shape mismatch for {source_path}: "
                    f"{paper_keep.shape} != {(count,)}"
                )
        else:
            # Backward compatibility: legacy Stage 02 physically removed every
            # paper-rejected grasp, so all remaining rows were paper-valid.
            paper_keep = np.ones(count, dtype=bool)
        final_palm = transform_one_point(
            arrays["T_world_r_base_link"].astype(np.float64), palm_in_base
        )
        pregrasp_palm = transform_one_point(
            arrays["pregrasp_T_world_r_base_link"].astype(np.float64), palm_in_base
        )
        path_minimum_z = np.minimum(pregrasp_palm[:, 2], final_palm[:, 2])
        pregrasp_below_table = pregrasp_palm[:, 2] <= minimum_z
        final_below_table = final_palm[:, 2] <= minimum_z
        below_table = np.zeros(len(final_palm), dtype=bool)
        if enhanced.get("reject_pregrasp_below_table", True):
            below_table |= pregrasp_below_table
        if enhanced.get("reject_final_palm_below_table", True):
            below_table |= final_below_table
        hit_matrix = np.zeros((len(final_palm), len(scene_objects)), dtype=bool)
        if enhanced.get("check_non_target_object_mesh_intersection", True):
            for other_id, mesh in world_meshes.items():
                if other_id == object_id:
                    continue
                hit_matrix[:, object_column[other_id]] = segment_mesh_intersections(
                    pregrasp_palm, final_palm, mesh
                )
        hit_other = hit_matrix.any(axis=1)
        path_geometry_keep = ~below_table & ~hit_other
        wuji2_safe_keep = paper_keep & path_geometry_keep
        reject_reason_bits = np.asarray(
            arrays.get("paper_reject_reason_bits", np.zeros(count, np.uint8)),
            dtype=np.uint8,
        ).copy()
        reject_reason_bits[below_table] |= np.uint8(4)
        reject_reason_bits[hit_other] |= np.uint8(8)

        # Preserve all Stage-02 rows.  This stage only appends independent path
        # masks and diagnostics so thresholds can be changed later.
        preserved = dict(arrays)
        preserved.update(
            {
                "source_index_stage02": np.arange(count, dtype=np.int64),
                "enhanced_path_geometry_keep_mask": path_geometry_keep,
                "wuji2_safe_keep_mask": wuji2_safe_keep,
                "wuji2_safe_keep_indices": np.flatnonzero(
                    wuji2_safe_keep
                ).astype(np.int64),
                "wuji2_reject_reason_bits": reject_reason_bits,
                "palm_center_pregrasp_world": pregrasp_palm.astype(np.float32),
                "palm_center_grasp_world": final_palm.astype(np.float32),
                "palm_path_minimum_world_z_m": path_minimum_z.astype(np.float32),
                "palm_path_intersection_by_segmentation": hit_matrix,
                "scene_segmentation_ids_path_filter": np.asarray(
                    sorted(scene_objects), dtype=np.int64
                ),
            }
        )
        output_path = stage02b_root / source_path.name
        diagnostics_path = stage02b_root / source_path.name.replace(
            ".npz", "_all_path_diagnostics.npz"
        )
        atomic_savez(output_path, **preserved)
        atomic_savez(
            diagnostics_path,
            paper_keep_mask=paper_keep,
            enhanced_path_geometry_keep_mask=path_geometry_keep,
            wuji2_safe_keep_mask=wuji2_safe_keep,
            wuji2_reject_reason_bits=reject_reason_bits,
            palm_center_pregrasp_world=pregrasp_palm.astype(np.float32),
            palm_center_grasp_world=final_palm.astype(np.float32),
            palm_path_minimum_world_z_m=path_minimum_z.astype(np.float32),
            pregrasp_palm_below_table_clearance_mask=pregrasp_below_table,
            final_palm_below_table_clearance_mask=final_below_table,
            below_table_or_clearance_mask=below_table,
            palm_path_intersection_by_segmentation=hit_matrix,
            scene_segmentation_ids=np.asarray(sorted(scene_objects), dtype=np.int64),
        )
        kept = int(wuji2_safe_keep.sum())
        total_input += count
        total_kept += kept
        records.append(
            {
                "segmentation_id": object_id,
                "object_code": record["object_code"],
                "input_count": count,
                "preserved_count": count,
                "paper_keep_count": int(paper_keep.sum()),
                "path_geometry_keep_count": int(path_geometry_keep.sum()),
                "kept_count": kept,
                "wuji2_safe_keep_count": kept,
                "rejected_below_table_or_clearance_count": int(below_table.sum()),
                "rejected_non_target_mesh_intersection_count": int(hit_other.sum()),
                "rejected_union_count": int((~wuji2_safe_keep).sum()),
                "output_npz": str(output_path.resolve()),
                "all_diagnostics_npz": str(diagnostics_path.resolve()),
            }
        )
        print(
            f"[OBJECT {object_id:03d}] preserved={count} "
            f"paper_keep={int(paper_keep.sum())} wuji2_safe_keep={kept} "
            f"below_table={int(below_table.sum())} "
            f"other_mesh_hit={int(hit_other.sum())}",
            flush=True,
        )
    manifest = {
        "schema_version": 2,
        "stage": stage_name,
        "status": "enhanced_path_metrics_complete_all_eligible_grasps_preserved",
        "training_ready": False,
        "output_policy_revision": OUTPUT_POLICY_REVISION,
        "storage_policy": (
            "non-destructive: retain every Stage-02 row and append "
            "wuji2_safe_keep_mask"
        ),
        "scene_index": int(args.scene),
        "scene_manifest": str(scene_path.resolve()),
        "input_stage_manifest": str(stage02_manifest_path.resolve()),
        "relationship_to_paper": "additional Wuji2 safety filter; not claimed by DexGraspNet2",
        "path_contract": {
            "start": "semantic palm center at opened, 0.10 m retreated pregrasp",
            "end": "semantic palm center at final predicted/training grasp root pose",
            "minimum_world_z_m": minimum_z,
            "table_top_world_z_m": table_top_z,
            "minimum_table_clearance_m": table_clearance,
            "table_rule": f"both endpoints and therefore the linear segment must satisfy world z > {minimum_z} m",
            "object_rule": "finite centerline must not intersect any non-target object triangle mesh",
            "target_object": "excluded",
            "guarantee_limit": "zero-radius palm centerline only, not complete-hand swept volume",
        },
        "selection_fields": {
            "paper_reproduction": "paper_keep_mask",
            "wuji2_safe_training": "wuji2_safe_keep_mask",
            "reject_reason_bits": {
                "1": "paper PREGRASP scene clearance failed",
                "2": "paper PREGRASP table clearance failed",
                "4": "semantic-palm path/table clearance failed",
                "8": "semantic-palm centerline intersects a non-target object",
            },
        },
        "object_records": records,
        "total_input": total_input,
        "total_kept": total_kept,
    }
    manifest_path = stage02b_root / "stage_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        f"[COMPLETE] scene={args.scene:04d} preserved={total_input} "
        f"wuji2_safe_keep={total_kept} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
from .project import project_path
