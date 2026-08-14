#!/usr/bin/env python3
"""Create the compact, isolated force-adjusted dataset input tree.

This does not regenerate scenes or cameras.  It copies the exact accepted scene
manifests and network_input.npz tensors from the frozen optimizer-target dataset,
then rewrites only paths for files that are physically copied into the new root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config, write_json_atomic  # noqa: E402


CONFIG = (
    PROJECT_ROOT
    / "02_training_dataset/config/"
    "wuji2_train60_100seminal_256view_force_adjusted_legacy_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, action="append")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    if args.all == bool(args.scene):
        raise ValueError("Choose exactly one of --all or one/more --scene values")
    config = load_config(CONFIG)
    source_root = PROJECT_ROOT / config["dataset_derivation"]["source_dataset"]
    output_root = Path(config["paths"]["output_root"])
    scene_indices = (
        list(range(int(config["scope"]["scene_count"])))
        if args.all
        else sorted(set(args.scene))
    )
    invalid = [x for x in scene_indices if not 0 <= x < int(config["scope"]["scene_count"])]
    if invalid:
        raise ValueError(f"Scene indices outside configured range: {invalid}")

    source_assets = source_root / "prepared_assets"
    output_assets = output_root / "prepared_assets"
    if not output_assets.exists():
        shutil.copytree(source_assets, output_assets)

    source_prefix = project_relative(source_root)
    output_prefix = project_relative(output_root)
    for scene_index in scene_indices:
        source_scene = source_root / "scenes" / f"scene_{scene_index:04d}"
        output_scene = output_root / "scenes" / f"scene_{scene_index:04d}"
        copy_file(source_scene / "network_input.npz", output_scene / "network_input.npz")
        manifest = json.loads((source_scene / "scene_manifest.json").read_text(encoding="utf-8"))
        manifest["camera_output"]["network_input"] = project_relative(
            output_scene / "network_input.npz"
        )
        for scene_object in manifest["objects"]:
            asset = scene_object["asset"]
            for key in ("centered_combined_obj", "urdf"):
                value = str(asset[key])
                if not value.startswith(source_prefix + "/prepared_assets/"):
                    raise RuntimeError(f"Unexpected source asset path: {value}")
                asset[key] = output_prefix + value[len(source_prefix):]
        manifest["dataset_derivation"] = {
            "source_scene_manifest": project_relative(
                source_scene / "scene_manifest.json"
            ),
            "policy": "Exact scene and camera tensors copied; only downstream grasp labels are recomputed.",
            "training_joint_target": "joint_positions_rad (Wuji2 1.0 force-adjusted target)",
        }
        write_json_atomic(output_scene / "scene_manifest.json", manifest)
        print(f"[INPUT READY] scene={scene_index:04d}", flush=True)

    dataset_manifest = {
        "schema_version": 1,
        "status": "input_ready_labels_pending",
        "config": project_relative(CONFIG),
        "source_dataset": project_relative(source_root),
        "output_root": project_relative(output_root),
        "initialized_scene_indices": scene_indices,
        "training_target": "joint_positions_rad",
        "pre_force_provenance": "pre_force_joint_positions_rad",
        "strict_ab_contract": "Scenes, point clouds, legacy URDF geometry and filters unchanged; only qpos supervision changes.",
    }
    write_json_atomic(output_root / "dataset_derivation_manifest.json", dataset_manifest)
    print(f"[COMPLETE] compact input root: {output_root}", flush=True)


if __name__ == "__main__":
    main()
