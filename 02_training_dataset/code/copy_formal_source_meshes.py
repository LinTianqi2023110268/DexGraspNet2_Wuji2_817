#!/usr/bin/env python3
"""Copy and hash-verify the 60 formal source meshes into this project."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL = PROJECT_ROOT / "02_training_dataset/config/wuji2_train60_object_pool_v1.json"
DEFAULT_TARGET = (
    PROJECT_ROOT / "02_training_dataset/assets/wuji2_factory/03_dexgraspnet_objects/meshdata"
)
DEFAULT_SCENES = (
    PROJECT_ROOT
    / "02_training_dataset/data/scene_datasets/"
    "wuji2_train60_100seminal_256view_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="One-time migration source meshdata directory; no external path is assumed.",
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENES)
    args = parser.parse_args()

    pool = json.loads(args.pool.resolve().read_text(encoding="utf-8"))
    source_root = args.source.expanduser().resolve()
    target_root = args.target.expanduser().resolve()
    scene_root = args.scene_root.expanduser().resolve()
    records = []
    for item in pool["objects"]:
        object_id = int(item["id"])
        code = str(item["code"])
        source = source_root / code / "coacd/decomposed.obj"
        target = target_root / code / "coacd/decomposed.obj"
        prepared_manifest = json.loads(
            (
                scene_root
                / "prepared_assets"
                / f"object_{object_id:03d}"
                / "asset_manifest.json"
            ).read_text(encoding="utf-8")
        )
        expected = str(prepared_manifest["source_obj_sha256"])
        actual = sha256(source)
        if actual != expected:
            raise RuntimeError(
                f"source hash mismatch for object_{object_id:03d} {code}: "
                f"{actual} != {expected}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied = sha256(target)
        if copied != expected:
            raise RuntimeError(f"copied hash mismatch for {target}")
        records.append(
            {
                "id": object_id,
                "code": code,
                "relative_obj": target.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": copied,
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "complete_hash_verified_copy",
        "object_count": len(records),
        "id_semantics": "id equals scene segmentation_id and object_pool_index + 1",
        "objects": records,
    }
    output = target_root / "mesh_pool_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"copied_and_verified={len(records)} manifest={output}")


if __name__ == "__main__":
    main()
