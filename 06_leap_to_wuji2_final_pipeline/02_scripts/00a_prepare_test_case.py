#!/usr/bin/env python3
"""Freeze one test-set scene/view into a new isolated pipeline case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
CASES_ROOT = PIPELINE_ROOT / "01_cases/active"
ACTIVE_CASE = PIPELINE_ROOT / "active_case.json"
TEST_ROOT = (
    PROJECT_ROOT
    / "02_training_dataset/data/scene_datasets/"
    "wuji2_test60_10upright_10view_v1/scenes"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [item.copy() for item in loaded.geometry.values()]
        if not meshes:
            raise RuntimeError(f"mesh has no geometry: {path}")
        return trimesh.util.concatenate(meshes)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--view-index", type=int, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    scene_id = f"scene_{args.scene_index:04d}"
    view_id = f"view_{args.view_index:04d}"
    if "/" in args.case_id or args.case_id in {"", ".", ".."}:
        raise ValueError(f"invalid case id: {args.case_id!r}")
    source_scene = (TEST_ROOT / scene_id).resolve()
    source_npz = source_scene / "network_input.npz"
    source_manifest_path = source_scene / "scene_manifest.json"
    case_root = (CASES_ROOT / args.case_id).resolve()
    if case_root.parent != CASES_ROOT.resolve():
        raise RuntimeError(f"case escaped 01_cases/active: {case_root}")
    input_root = case_root / "01_input"
    view_input = input_root / f"{view_id}_network_input.npz"
    scene_manifest = input_root / f"{scene_id}_manifest.json"
    for path in (source_npz, source_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (case_root / "case.json").exists():
        raise FileExistsError(
            f"case already exists; refusing to overwrite: {case_root}"
        )
    input_root.mkdir(parents=True, exist_ok=False)
    for stage in (
        "02_retargeting", "03_root_alignment", "04_squeeze",
        "05_visualization", "06_isaacsim",
    ):
        (case_root / stage).mkdir()

    with np.load(source_npz, allow_pickle=False) as archive:
        required = ("pc", "seg", "edge", "extrinsics", "pixel_indices")
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"network_input missing {missing}")
        pc = archive["pc"]
        if pc.ndim != 3 or pc.shape[1:] != (40000, 3):
            raise ValueError(f"unexpected point cloud tensor: {pc.shape}")
        if not 0 <= args.view_index < pc.shape[0]:
            raise IndexError(f"view {args.view_index} outside {pc.shape[0]} views")
        frozen = {
            key: np.asarray(archive[key][args.view_index:args.view_index + 1])
            for key in required
        }
    np.savez_compressed(view_input, **frozen)

    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    objects = []
    rng = np.random.default_rng(0)
    dataset_root = source_manifest_path.parents[2]
    for record in source["objects"]:
        seg_id = int(record["segmentation_id"])
        mesh_path = Path(record["asset"]["centered_combined_obj"]).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh = load_mesh(mesh_path)
        # trimesh's sampler uses NumPy's global generator; preserve caller state.
        state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        try:
            points, _ = trimesh.sample.sample_surface(mesh, 4096)
        finally:
            np.random.set_state(state)
        sample_path = input_root / f"object_{seg_id:03d}_surface_points.npy"
        np.save(sample_path, np.asarray(points, dtype=np.float32))
        pool_index = int(record["object_pool_index"])
        object_usd = (
            dataset_root / f"usd_cache/object_{pool_index:03d}/flat/"
            f"object_{pool_index:03d}_editable.usd"
        )
        if not object_usd.is_file():
            raise FileNotFoundError(object_usd)
        objects.append(
            {
                "segmentation_id": seg_id,
                "object_pool_index": pool_index,
                "code": str(record["object_code"]),
                "pose_world_object": record["T_world_centered_object"],
                "surface_points": str(sample_path),
                "visual_mesh": str(mesh_path),
                "simulation_usd": str(object_usd.resolve()),
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment": "official LEAP to Wuji2 multi-case pipeline",
        "scene_index": int(source["scene_index"]),
        "view_index": int(args.view_index),
        "source_scene_manifest": str(source_manifest_path),
        "source_network_input": str(source_npz),
        "frozen_view_input": str(view_input),
        "frozen_view_input_sha256": sha256(view_input),
        "coordinate_contract": source["coordinate_contract"],
        "table": source["table"],
        "camera": source["camera"],
        "objects": objects,
    }
    scene_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    visible_ids, visible_counts = np.unique(frozen["seg"], return_counts=True)
    case = {
        "schema_version": 1,
        "case_id": args.case_id,
        "scene_id": scene_id,
        "view_id": view_id,
        "target_segmentation_id": None,
        "target_object_code": None,
        "source_candidate_index": None,
        "selection_policy": "official_network_raw_score_rank0",
        "source_hand": "LEAP Hand",
        "target_hand": "Wuji Hand 2 Beta1 right",
        "pipeline_status": "scene_and_point_cloud_ready",
        "physics_status": "not_tested",
        "point_cloud_shape": list(frozen["pc"].shape),
        "visible_point_count_by_segmentation_id": {
            str(int(key)): int(value)
            for key, value in zip(visible_ids, visible_counts)
        },
    }
    (case_root / "case.json").write_text(
        json.dumps(case, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (case_root / "README.md").write_text(
        f"# {args.case_id}\n\n"
        f"- 测试场景：`{scene_id}`\n"
        f"- 单视角：`{view_id}`\n"
        f"- 网络输入：`01_input/{view_id}_network_input.npz`，形状 `(1,40000,3)`\n"
        f"- 点云预览：`01_input/single_view_point_cloud.glb`\n"
        f"- 目标物体和候选编号：运行官方网络后写入 `case.json`\n"
        f"- 四手对比：`05_visualization/four_hand_final.glb`\n"
        f"- Isaac Sim：依次运行 `06_isaacsim/01_import.py` 和 `02_execute.py`\n\n"
        "本目录只保存这个场景/视角/候选的输入与结果；算法代码位于 "
        "`../../02_scripts`，公共模型和Isaac执行器位于 `../../00_shared`。\n",
        encoding="utf-8",
    )
    if args.activate:
        ACTIVE_CASE.write_text(
            json.dumps(
                {"schema_version": 1, "active_case_id": args.case_id},
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
    print(f"[PASS] case={args.case_id}; scene={scene_id}; view={view_id}")
    print(f"[PASS] point_cloud={frozen['pc'].shape}; objects={len(objects)}")
    print(f"[PASS] visible={case['visible_point_count_by_segmentation_id']}")
    print(f"[OK] {case_root}")


if __name__ == "__main__":
    main()
