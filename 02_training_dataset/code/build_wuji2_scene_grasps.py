#!/usr/bin/env python3
"""Stage 01: transform validated Wuji2 object grasps into a scene world frame.

This is an isolated Wuji2 adaptation of the object-to-scene transform in
DexGraspNet2 ``src/preprocess/dex_graspness.py``.  It deliberately does not
perform scene/table collision filtering or graspness assignment yet.  Its
outputs are therefore diagnostic intermediate data, not training labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ADAPTER_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config, write_json_atomic  # noqa: E402


STAGE_NAME = "01_transformed_object_grasps"
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
    parser.add_argument(
        "--max-per-object",
        type=int,
        default=None,
        help="Optional deterministic prefix for a small diagnostic run.",
    )
    return parser.parse_args()


def require_rigid_transform(matrix: np.ndarray, label: str) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must be 4x4, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7):
        raise ValueError(f"{label} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-4):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-4):
        raise ValueError(f"{label} rotation determinant is not +1")


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_scene_manifest(config: dict, scene_index: int) -> tuple[Path, dict]:
    scene_dir = (
        Path(config["paths"]["output_root"])
        / "scenes"
        / f"scene_{scene_index:04d}"
    )
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("objects", [])) != int(
        config["scope"]["objects_per_scene"]
    ):
        raise RuntimeError("Scene object count does not match the experiment config")
    return manifest_path, manifest


def validate_object_contract(
    configured: dict, scene_object: dict, source_payload: dict, source_manifest: dict
) -> None:
    code = configured["code"]
    observed = (
        scene_object.get("object_code"),
        source_payload.get("object_code"),
        source_manifest.get("object_code"),
    )
    if observed != (code, code, code):
        raise RuntimeError(f"Object-code contract failed for {code}: {observed}")
    scene_asset = scene_object["asset"]
    expected_scale = float(configured["scale"])
    scales = (
        float(scene_asset["scale"]),
        float(source_manifest["object_scale"]),
    )
    if not all(np.isclose(value, expected_scale, atol=0.0, rtol=0.0) for value in scales):
        raise RuntimeError(f"Scale contract failed for {code}: {scales}")
    hashes = (
        scene_asset["source_obj_sha256"],
        source_manifest["source_mesh_sha256"],
    )
    if hashes[0] != hashes[1]:
        raise RuntimeError(f"Mesh SHA256 contract failed for {code}: {hashes}")


def grasp_is_eligible(grasp: dict, label_cfg: dict) -> bool:
    physics = grasp["physics_validation"]
    diagnostics = grasp["diagnostics"]
    if label_cfg["require_paper_sim_success"] and not bool(
        physics["paper_sim_success"]
    ):
        return False
    if label_cfg["require_all_gravity_directions"] and int(
        physics["passed_gravity_directions"]
    ) != int(physics["required_gravity_directions"]):
        return False
    return float(diagnostics["reverse_penetration_max_depth_mm"]) < float(
        label_cfg["maximum_object_penetration_mm"]
    )


def transform_object_grasps(
    object_pose_world: np.ndarray,
    grasps: list[dict],
    label_cfg: dict,
    joint_count: int,
) -> dict[str, np.ndarray]:
    world_matrices = []
    object_matrices = []
    training_qpos = []
    pre_force_qpos = []
    validation_qpos = []
    force_delta = []
    success_rank = []
    candidate_id = []
    energy_rank = []
    penetration_mm = []
    training_field = label_cfg["training_joint_field"]
    validation_field = label_cfg["physics_validation_joint_field"]
    pre_force_field = label_cfg.get("pre_force_joint_field", training_field)
    for grasp in grasps:
        object_hand = np.asarray(
            grasp["hand_root_pose_in_centered_object"]["matrix"],
            dtype=np.float64,
        )
        require_rigid_transform(object_hand, "T_centered_object_r_base_link")
        q_train = np.asarray(grasp[training_field], dtype=np.float64)
        q_pre_force = np.asarray(grasp[pre_force_field], dtype=np.float64)
        q_valid = np.asarray(grasp[validation_field], dtype=np.float64)
        q_delta = np.asarray(grasp["force_adjustment"]["joint_delta_rad"], dtype=np.float64)
        if (
            q_train.shape != (joint_count,)
            or q_pre_force.shape != (joint_count,)
            or q_valid.shape != (joint_count,)
        ):
            raise ValueError(
                f"Expected {joint_count} joints, got "
                f"{q_train.shape}/{q_pre_force.shape}/{q_valid.shape}"
            )
        if q_delta.shape != (joint_count,):
            raise ValueError(f"Expected {joint_count} force deltas, got {q_delta.shape}")
        if not np.allclose(q_pre_force + q_delta, q_valid, atol=2.0e-6, rtol=0.0):
            raise RuntimeError("Force-adjusted qpos does not equal pre-force qpos + delta")
        world_hand = object_pose_world @ object_hand
        require_rigid_transform(world_hand, "T_world_r_base_link")
        recovered = np.linalg.inv(object_pose_world) @ world_hand
        if not np.allclose(recovered, object_hand, atol=2.0e-7, rtol=0.0):
            raise RuntimeError("Object-to-world transform round trip failed")
        world_matrices.append(world_hand)
        object_matrices.append(object_hand)
        training_qpos.append(q_train)
        pre_force_qpos.append(q_pre_force)
        validation_qpos.append(q_valid)
        force_delta.append(q_delta)
        success_rank.append(int(grasp["success_rank"]))
        candidate_id.append(int(grasp["candidate_id"]))
        energy_rank.append(int(grasp["energy_rank"]))
        penetration_mm.append(
            float(grasp["diagnostics"]["reverse_penetration_max_depth_mm"])
        )
    world = np.stack(world_matrices).astype(np.float32)
    object_local = np.stack(object_matrices).astype(np.float32)
    return {
        "rotation": world[:, :3, :3],
        "translation": world[:, :3, 3],
        "T_world_r_base_link": world,
        "T_centered_object_r_base_link": object_local,
        "qpos": np.stack(training_qpos).astype(np.float32),
        "qpos_pre_force": np.stack(pre_force_qpos).astype(np.float32),
        "qpos_force_adjusted": np.stack(validation_qpos).astype(np.float32),
        "force_adjustment_delta": np.stack(force_delta).astype(np.float32),
        "success_rank": np.asarray(success_rank, dtype=np.int64),
        "candidate_id": np.asarray(candidate_id, dtype=np.int64),
        "energy_rank": np.asarray(energy_rank, dtype=np.int64),
        "object_penetration_mm": np.asarray(penetration_mm, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    if args.scene < 0:
        raise ValueError("--scene must be non-negative")
    if args.max_per_object is not None and args.max_per_object <= 0:
        raise ValueError("--max-per-object must be positive")
    config = load_config(args.config)
    label_cfg = config["grasp_label_generation"]
    source_root = Path(config["paths"]["single_object_output_root"])
    scene_manifest_path, scene_manifest = load_scene_manifest(config, args.scene)
    scene_objects = {
        item["object_code"]: item for item in scene_manifest["objects"]
    }
    stage_root = (
        Path(config["paths"]["output_root"])
        / label_cfg["stage_directory_name"]
        / STAGE_NAME
        / f"scene_{args.scene:04d}"
    )
    object_records = []
    common_joint_order = None
    total_source = 0
    total_eligible = 0
    configured_by_code = {item["code"]: item for item in config["objects"]}
    if len(configured_by_code) != len(config["objects"]):
        raise RuntimeError("Configured object pool contains duplicate object codes")
    for scene_object in scene_manifest["objects"]:
        code = scene_object["object_code"]
        if code not in configured_by_code:
            raise RuntimeError(f"Scene contains an object outside the pool: {code}")
        configured = configured_by_code[code]
        object_root = source_root / "objects" / code
        source_manifest_path = object_root / "manifest.json"
        source_path = object_root / label_cfg["source_grasp_relative_path"]
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        validate_object_contract(
            configured, scene_object, source_payload, source_manifest
        )
        joint_order = tuple(source_payload["joint_order"])
        if len(joint_order) != int(label_cfg["joint_count"]):
            raise RuntimeError(f"{code}: expected 20 joint names, got {len(joint_order)}")
        if common_joint_order is None:
            common_joint_order = joint_order
        elif joint_order != common_joint_order:
            raise RuntimeError(f"{code}: joint order differs from the first object")
        source_grasps = list(source_payload["grasps"])
        eligible = [
            grasp
            for grasp in source_grasps
            if grasp_is_eligible(grasp, label_cfg)
        ]
        eligible_full_count = len(eligible)
        if args.max_per_object is not None:
            eligible = eligible[: args.max_per_object]
        if not eligible:
            raise RuntimeError(f"{code}: no grasp survived the Stage 01 filter")
        object_pose_world = np.asarray(
            scene_objects[code]["T_world_centered_object"], dtype=np.float64
        )
        require_rigid_transform(object_pose_world, f"{code}: T_world_centered_object")
        arrays = transform_object_grasps(
            object_pose_world,
            eligible,
            label_cfg,
            int(label_cfg["joint_count"]),
        )
        output_path = stage_root / f"object_{int(configured['id']):03d}_{code}.npz"
        atomic_savez(output_path, **arrays)
        total_source += len(source_grasps)
        total_eligible += len(eligible)
        object_records.append(
            {
                "segmentation_id": int(configured["id"]),
                "object_code": code,
                "source_grasp_file": str(source_path.resolve()),
                "source_grasp_count": len(source_grasps),
                "eligible_count_before_optional_limit": eligible_full_count,
                "exported_count": len(eligible),
                "output_npz": str(output_path.resolve()),
                "T_world_centered_object": object_pose_world.tolist(),
                "maximum_exported_object_penetration_mm": float(
                    arrays["object_penetration_mm"].max()
                ),
            }
        )
        print(
            f"[OBJECT {int(configured['id']):03d}] {code}: "
            f"source={len(source_grasps)} eligible={eligible_full_count} "
            f"exported={len(eligible)}",
            flush=True,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "diagnostic_intermediate_not_training_ready",
        "training_ready": False,
        "scene_index": int(args.scene),
        "scene_manifest": str(scene_manifest_path.resolve()),
        "coordinate_contract": {
            "source_object_pose": "T_centered_object_r_base_link",
            "scene_object_pose": "T_world_centered_object",
            "output_hand_pose": "T_world_r_base_link = T_world_centered_object @ T_centered_object_r_base_link",
        },
        "label_contract": {
            "hand_root_link": label_cfg["hand_root_link"],
            "training_joint_field": label_cfg["training_joint_field"],
            "pre_force_joint_field": label_cfg.get(
                "pre_force_joint_field", label_cfg["training_joint_field"]
            ),
            "physics_validation_joint_field": label_cfg[
                "physics_validation_joint_field"
            ],
            "joint_order": list(common_joint_order),
            "joint_count": int(label_cfg["joint_count"]),
            "maximum_object_penetration_mm_strict_less_than": float(
                label_cfg["maximum_object_penetration_mm"]
            ),
            "paper_sim_success_required": bool(
                label_cfg["require_paper_sim_success"]
            ),
            "all_gravity_directions_required": bool(
                label_cfg["require_all_gravity_directions"]
            ),
        },
        "remaining_required_stages": [
            "Wuji2 scene/table collision filtering",
            "Wuji2 fingertip/palm reference-point computation",
            "surface and single-view graspness assignment",
        ],
        "object_records": object_records,
        "total_source_grasps": total_source,
        "total_exported_grasps": total_eligible,
    }
    manifest_path = stage_root / "stage_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        f"[COMPLETE] scene={args.scene:04d} exported={total_eligible} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
