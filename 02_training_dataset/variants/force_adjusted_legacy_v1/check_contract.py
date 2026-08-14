#!/usr/bin/env python3
"""Audit the strict single-variable contract of the force-adjusted dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OLD_ROOT = PROJECT_ROOT / (
    "02_training_dataset/data/scene_datasets/"
    "wuji2_train60_100seminal_256view_v1"
)
NEW_ROOT = PROJECT_ROOT / (
    "02_training_dataset/data/scene_datasets/"
    "wuji2_train60_100seminal_256view_force_adjusted_legacy_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, default=17)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def object_archives(root: Path, stage: str, scene: int) -> list[Path]:
    return sorted(
        path
        for path in (
            root / "grasp_label_stages" / stage / f"scene_{scene:04d}"
        ).glob("object_*.npz")
        if "surface_graspness" not in path.name
        and "diagnostics" not in path.name
    )


def main() -> None:
    args = parse_args()
    scene_name = f"scene_{args.scene:04d}"
    old_scene = OLD_ROOT / "scenes" / scene_name
    new_scene = NEW_ROOT / "scenes" / scene_name
    old_manifest = json.loads((old_scene / "scene_manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((new_scene / "scene_manifest.json").read_text(encoding="utf-8"))
    old_poses = [item["T_world_centered_object"] for item in old_manifest["objects"]]
    new_poses = [item["T_world_centered_object"] for item in new_manifest["objects"]]
    if old_poses != new_poses:
        raise RuntimeError("Scene object poses changed")
    old_network_hash = digest(old_scene / "network_input.npz")
    new_network_hash = digest(new_scene / "network_input.npz")
    if old_network_hash != new_network_hash:
        raise RuntimeError("network_input.npz changed")

    stage01 = object_archives(NEW_ROOT, "01_transformed_object_grasps", args.scene)
    if len(stage01) != len(new_manifest["objects"]):
        raise RuntimeError("Stage01 object count mismatch")
    stage01_counts = {}
    for path in stage01:
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["qpos"], archive["qpos_force_adjusted"]):
                raise RuntimeError(f"qpos is not force-adjusted: {path.name}")
            if not np.allclose(
                archive["qpos_pre_force"] + archive["force_adjustment_delta"],
                archive["qpos"],
                atol=2.0e-6,
                rtol=0.0,
            ):
                raise RuntimeError(f"pre+delta identity failed: {path.name}")
            stage01_counts[path.name] = int(len(archive["qpos"]))

    stage03 = object_archives(
        NEW_ROOT, "03_reference_points_and_surface_graspness", args.scene
    )
    for path in stage03:
        stage01_path = (
            NEW_ROOT
            / "grasp_label_stages/01_transformed_object_grasps"
            / scene_name
            / path.name
        )
        with np.load(stage01_path, allow_pickle=False) as source, np.load(
            path, allow_pickle=False
        ) as final:
            source_by_id = {
                int(value): index for index, value in enumerate(source["candidate_id"])
            }
            for index, candidate in enumerate(final["candidate_id"]):
                source_index = source_by_id[int(candidate)]
                if not np.array_equal(final["qpos"][index], source["qpos"][source_index]):
                    raise RuntimeError(f"Stage03 qpos drift: {path.name}")

    report = {
        "schema_version": 1,
        "status": "pass",
        "scene_index": args.scene,
        "network_input_sha256": new_network_hash,
        "scene_pose_identity": True,
        "qpos_equals_qpos_force_adjusted": True,
        "qpos_pre_force_plus_delta_equals_qpos": True,
        "stage03_qpos_traceable_to_stage01": True,
        "stage01_counts": stage01_counts,
    }
    destination = NEW_ROOT / "audits" / f"{scene_name}_force_adjusted_contract.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
