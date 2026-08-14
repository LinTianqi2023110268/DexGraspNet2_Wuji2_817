#!/usr/bin/env python3
"""选择5个测试场景各自物体可见性最好的一个视角。

输入：
  02_training_dataset/data/scene_datasets/wuji2_test60_10upright_10view_v1/
  每个场景的scene_manifest.json与network_input.npz。

输出：
  07_wuji2_network_3p3r_sim/00_config/test_5scene.json。

本程序只读取分割标签，不运行网络、不修改测试集。视角按以下顺序比较：
可见物体数、最少物体点数、全部物体点数；三者均越大越好。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = PROJECT_ROOT / (
    "02_training_dataset/data/scene_datasets/"
    "wuji2_test60_10upright_10view_v1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "07_wuji2_network_3p3r_sim/00_config/test_5scene.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-count", type=int, default=5)
    parser.add_argument("--minimum-object-points", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.scene_count <= 10:
        raise ValueError("--scene-count必须位于1到10之间")

    selected = []
    for scene_index in range(args.scene_count):
        scene_dir = TEST_ROOT / "scenes" / f"scene_{scene_index:04d}"
        manifest_path = scene_dir / "scene_manifest.json"
        network_path = scene_dir / "network_input.npz"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        object_ids = [int(item["segmentation_id"]) for item in manifest["objects"]]
        with np.load(network_path, allow_pickle=False) as archive:
            segmentation = np.asarray(archive["seg"], dtype=np.int64)

        view_records = []
        for view_index, view_seg in enumerate(segmentation):
            counts = [int(np.count_nonzero(view_seg == value)) for value in object_ids]
            view_records.append(
                {
                    "view_index": view_index,
                    "visible_object_count": int(
                        sum(value >= args.minimum_object_points for value in counts)
                    ),
                    "minimum_object_points": min(counts),
                    "total_object_points": sum(counts),
                    "object_point_counts": dict(zip(map(str, object_ids), counts)),
                }
            )
        best = max(
            view_records,
            key=lambda item: (
                item["visible_object_count"],
                item["minimum_object_points"],
                item["total_object_points"],
            ),
        )
        if best["visible_object_count"] != len(object_ids):
            raise RuntimeError(
                f"scene_{scene_index:04d}没有一个视角满足全部物体可见"
            )
        selected.append(
            {
                "scene_index": scene_index,
                "view_index": best["view_index"],
                "scene_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
                "network_input": network_path.relative_to(PROJECT_ROOT).as_posix(),
                "objects": [
                    {
                        "segmentation_id": int(item["segmentation_id"]),
                        "object_pool_index": int(item["object_pool_index"]),
                        "object_code": str(item["object_code"]),
                        "visible_point_count": int(
                            best["object_point_counts"][str(item["segmentation_id"])]
                        ),
                    }
                    for item in manifest["objects"]
                ],
            }
        )

    payload = {
        "schema_version": 1,
        "status": "five_scenes_one_visible_view_selected",
        "test_dataset": TEST_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "selection_rule": (
            "maximize visible object count, then minimum per-object point count, "
            "then total object point count"
        ),
        "minimum_visible_object_points": args.minimum_object_points,
        "scenes": selected,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for item in selected:
        counts = [obj["visible_point_count"] for obj in item["objects"]]
        print(
            f"scene={item['scene_index']:04d} view={item['view_index']:04d} "
            f"objects=6 min_points={min(counts)}"
        )
    print(f"output={output}")


if __name__ == "__main__":
    main()
