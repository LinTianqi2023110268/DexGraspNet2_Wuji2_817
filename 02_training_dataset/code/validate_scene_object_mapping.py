#!/usr/bin/env python3
"""Validate Wuji2 single-object, scene, view and Stage01-04 identity mapping.

This checker is intentionally read-only.  It verifies identity fields and relative
file layout rather than trusting absolute provenance paths stored in manifests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SINGLE_ROOT = PROJECT_ROOT / "02_training_dataset/data/single_object_dataset"
DEFAULT_SCENE_ROOT = (
    PROJECT_ROOT
    / "02_training_dataset/data/scene_datasets/"
    "wuji2_train60_100seminal_256view_v1"
)
DEFAULT_POOL_CONFIG = (
    PROJECT_ROOT / "02_training_dataset/config/wuji2_train60_object_pool_v1.json"
)
STAGES_WITH_OBJECTS = (
    "01_transformed_object_grasps",
    "02_scene_table_collision_filtered",
    "02b_enhanced_palm_center_path_filtered",
    "03_reference_points_and_surface_graspness",
)
STAGE04 = "04_single_view_training_labels"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-root", type=Path, default=DEFAULT_SINGLE_ROOT)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--pool-config", type=Path, default=DEFAULT_POOL_CONFIG)
    parser.add_argument("--scene-count", type=int, default=100)
    parser.add_argument("--views-per-scene", type=int, default=256)
    parser.add_argument(
        "--check-network-segmentation",
        action="store_true",
        help="Load every network_input.npz segmentation array (slower).",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def object_pairs(records: list[dict]) -> list[tuple[int, str]]:
    return [
        (int(record["segmentation_id"]), str(record["object_code"]))
        for record in records
    ]


def main() -> None:
    args = arguments()
    single_root = args.single_root.expanduser().resolve()
    scene_root = args.scene_root.expanduser().resolve()
    pool_config = args.pool_config.expanduser().resolve()
    errors: list[str] = []

    pool = read_json(pool_config)
    configured = {
        int(record["id"]): str(record["code"]) for record in pool["objects"]
    }
    if len(configured) != 60 or len(set(configured.values())) != 60:
        fail(errors, "pool config must contain 60 unique IDs and object codes")

    dataset_index = read_json(single_root / "dataset_index.json")
    index_text = json.dumps(dataset_index, ensure_ascii=False)
    for object_id, code in configured.items():
        object_root = single_root / "objects" / code
        manifest_path = object_root / "manifest.json"
        grasp_path = object_root / "grasps/successful_grasps.json"
        surface_path = object_root / "pointcloud/object_surface_20000.npy"
        for required in (manifest_path, grasp_path, surface_path):
            if not required.is_file():
                fail(errors, f"object {object_id:03d} {code}: missing {required}")
        if code not in index_text:
            fail(errors, f"object {object_id:03d} {code}: absent from dataset_index.json")

        prepared = scene_root / "prepared_assets" / f"object_{object_id:03d}"
        prepared_manifest = prepared / "asset_manifest.json"
        if not prepared_manifest.is_file():
            fail(errors, f"object {object_id:03d} {code}: missing prepared asset manifest")
        else:
            actual = str(read_json(prepared_manifest).get("object_code"))
            if actual != code:
                fail(
                    errors,
                    f"object_{object_id:03d}: prepared code {actual!r} != {code!r}",
                )

    seen_ids: Counter[int] = Counter()
    seen_codes: Counter[str] = Counter()
    total_stage04_views = 0
    empty_stage04_views = 0
    total_available_grasps = 0

    for scene_index in range(args.scene_count):
        scene_name = f"scene_{scene_index:04d}"
        scene_dir = scene_root / "scenes" / scene_name
        scene_manifest_path = scene_dir / "scene_manifest.json"
        network_input_path = scene_dir / "network_input.npz"
        if not scene_manifest_path.is_file():
            fail(errors, f"{scene_name}: missing scene_manifest.json")
            continue
        if not network_input_path.is_file():
            fail(errors, f"{scene_name}: missing network_input.npz")

        scene_manifest = read_json(scene_manifest_path)
        if int(scene_manifest.get("scene_index", -1)) != scene_index:
            fail(errors, f"{scene_name}: scene_index mismatch")
        objects = scene_manifest.get("objects", [])
        pairs = object_pairs(objects)
        if len(pairs) != 6 or len(set(pairs)) != 6:
            fail(errors, f"{scene_name}: expected 6 unique objects, got {pairs}")
        expected_pairs = set(pairs)
        expected_ids = {item[0] for item in pairs}
        for record in objects:
            segmentation_id = int(record["segmentation_id"])
            pool_index = int(record["object_pool_index"])
            code = str(record["object_code"])
            if segmentation_id != pool_index + 1:
                fail(errors, f"{scene_name}: segmentation/pool index mismatch for {code}")
            if configured.get(segmentation_id) != code:
                fail(errors, f"{scene_name}: ID {segmentation_id} maps to wrong code {code}")
            asset_code = str(record.get("asset", {}).get("object_code"))
            if asset_code != code:
                fail(errors, f"{scene_name}: nested asset code mismatch for {code}")
            seen_ids[segmentation_id] += 1
            seen_codes[code] += 1

        camera_dirs = sorted((scene_dir / "camera").glob("view_*"))
        if len(camera_dirs) != args.views_per_scene:
            fail(errors, f"{scene_name}: camera view count {len(camera_dirs)}")
        for view_index, camera_dir in enumerate(camera_dirs):
            if camera_dir.name != f"view_{view_index:04d}":
                fail(errors, f"{scene_name}: non-contiguous camera view {camera_dir.name}")
            for filename in (
                "depth_m.npy",
                "segmentation.npy",
                "semantic_rgba.npy",
                "sample_pixel_indices.npy",
            ):
                if not (camera_dir / filename).is_file():
                    fail(errors, f"{scene_name}/{camera_dir.name}: missing {filename}")

        if args.check_network_segmentation and network_input_path.is_file():
            with np.load(network_input_path, allow_pickle=False) as archive:
                segmentation_key = "seg" if "seg" in archive.files else "segmentation"
                if segmentation_key not in archive.files:
                    fail(errors, f"{scene_name}: network input has no segmentation field")
                else:
                    actual_ids = set(np.unique(archive[segmentation_key]).astype(int).tolist())
                    unexpected = actual_ids - expected_ids - {0}
                    if unexpected:
                        fail(errors, f"{scene_name}: network segmentation IDs {unexpected}")

        for stage in STAGES_WITH_OBJECTS:
            stage_dir = scene_root / "grasp_label_stages" / stage / scene_name
            manifest_path = stage_dir / "stage_manifest.json"
            if not manifest_path.is_file():
                fail(errors, f"{scene_name}/{stage}: missing stage manifest")
                continue
            manifest = read_json(manifest_path)
            if int(manifest.get("scene_index", -1)) != scene_index:
                fail(errors, f"{scene_name}/{stage}: scene_index mismatch")
            stage_pairs = set(object_pairs(manifest.get("object_records", [])))
            if stage_pairs != expected_pairs:
                fail(
                    errors,
                    f"{scene_name}/{stage}: object set mismatch "
                    f"missing={expected_pairs-stage_pairs} extra={stage_pairs-expected_pairs}",
                )
            for segmentation_id, code in expected_pairs:
                prefix = f"object_{segmentation_id:03d}_{code}"
                if not any(path.name.startswith(prefix) for path in stage_dir.glob("*.npz")):
                    fail(errors, f"{scene_name}/{stage}: no NPZ matching {prefix}")

        stage04_dir = scene_root / "grasp_label_stages" / STAGE04 / scene_name
        stage04_manifest_path = stage04_dir / "stage_manifest.json"
        if not stage04_manifest_path.is_file():
            fail(errors, f"{scene_name}/{STAGE04}: missing stage manifest")
            continue
        stage04_manifest = read_json(stage04_manifest_path)
        records = stage04_manifest.get("view_records", [])
        if len(records) != args.views_per_scene:
            fail(errors, f"{scene_name}/{STAGE04}: view record count {len(records)}")
        view_files = sorted(stage04_dir.glob("view_*.npz"))
        if len(view_files) != args.views_per_scene:
            fail(errors, f"{scene_name}/{STAGE04}: view NPZ count {len(view_files)}")
        for view_index, record in enumerate(records):
            if int(record.get("view_index", -1)) != view_index:
                fail(errors, f"{scene_name}/{STAGE04}: view index mismatch at {view_index}")
            label_ids = {int(key) for key in record.get("available_grasp_count_by_object", {})}
            if not label_ids.issubset(expected_ids):
                fail(errors, f"{scene_name}/view_{view_index:04d}: label IDs outside scene")
            available = int(record.get("total_available_grasp_count", 0))
            total_available_grasps += available
            empty_stage04_views += int(available == 0)
            total_stage04_views += 1

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "single_object_pool_count": len(configured),
        "scene_count": args.scene_count,
        "objects_per_scene": 6,
        "scene_object_assignments": sum(seen_ids.values()),
        "covered_object_ids": len(seen_ids),
        "covered_object_codes": len(seen_codes),
        "uncovered_pool_objects": [
            {"id": object_id, "code": configured[object_id]}
            for object_id in sorted(set(configured) - set(seen_ids))
        ],
        "stage04_view_count": total_stage04_views,
        "empty_stage04_views": empty_stage04_views,
        "usable_stage04_views": total_stage04_views - empty_stage04_views,
        "total_available_grasp_count": total_available_grasps,
        "error_count": len(errors),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print("\nFirst mapping errors:")
        for message in errors[:100]:
            print("-", message)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
