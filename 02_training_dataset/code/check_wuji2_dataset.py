#!/usr/bin/env python3
"""Load and collate Wuji2 samples, then verify the official tensor contract."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ADAPTER_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "04_training"))

from wuji2_dataset import Wuji2SceneDataset, minkowski_collate_fn  # noqa: E402
from wuji2_dgn2.adapter_common import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ADAPTER_ROOT / "config/wuji2_train60_100seminal_256view_v1.json",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument(
        "--scene",
        type=int,
        action="append",
        help="Check only these scene indices instead of the complete split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(config["paths"]["output_root"])
    split_key = f"{args.split}_scene_indices"
    scene_indices = tuple(
        config["dataset_split"][split_key]
        if args.scene is None
        else args.scene
    )
    if not scene_indices:
        raise RuntimeError(
            f"No scenes are configured for split={args.split!r}; "
            "choose a non-empty split or pass --scene explicitly"
        )
    dataset = Wuji2SceneDataset(root, scene_indices=scene_indices, is_train=False)
    sample = dataset[0]
    print(
        f"split={args.split} scenes={list(scene_indices)} "
        f"single_view_samples={len(dataset)}"
    )
    for key, value in sample.items():
        print(f"  {key:28s} {value.shape} {value.dtype}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=minkowski_collate_fn,
    )
    batch = next(iter(loader))
    print("collated batch")
    for key, value in batch.items():
        print(f"  {key:28s} {tuple(value.shape)} {value.dtype}")
    assert tuple(batch["point_clouds"].shape) == (1, 40000, 3)
    assert tuple(batch["rot"].shape) == (1, 64, 3, 3)
    assert tuple(batch["trans"].shape) == (1, 64, 3)
    assert tuple(batch["qpos"].shape) == (1, 64, 20)
    assert tuple(batch["centers"].shape) == (1, 64)
    print("[PASS] official loader contract with Wuji2 joint_num=20")


if __name__ == "__main__":
    main()
