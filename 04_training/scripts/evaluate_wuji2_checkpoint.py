#!/usr/bin/env python3
"""Evaluate one Wuji2 checkpoint on a scene-disjoint validation or test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ADAPTER_ROOT.parent
OFFICIAL_ROOT = PROJECT_ROOT / "03_prediction_network/official_core"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(ADAPTER_ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT))

from wuji2_dgn2.adapter_common import load_config as load_adapter_config  # noqa: E402
from wuji2_dataset import Wuji2SceneDataset, minkowski_collate_fn  # noqa: E402
from src.network.model import get_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.util import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "02_training_dataset/config/wuji2_train60_100seminal_256view_v1.json",
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path, base: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def joint_target_field(config: dict) -> str:
    """Return the explicit 20-joint supervision field of a resolved dataset config."""
    pose_policy = config.get("pose_policy", {})
    field = pose_policy.get("training_joint_field")
    if not field:
        raise RuntimeError(
            "Dataset config does not declare pose_policy.training_joint_field"
        )
    return str(field)


def checkpoint_training_config(checkpoint_path: Path) -> Path:
    """Read checkpoint provenance without modifying the historical checkpoint file."""
    run_config_path = checkpoint_path.parent.parent / "run_config.json"
    if not run_config_path.is_file():
        raise RuntimeError(
            "Checkpoint has no sibling run_config.json, so its joint-label semantics "
            "cannot be audited by this evaluator"
        )
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    value = run_config.get("adapter_config")
    if not value:
        raise RuntimeError(f"Missing adapter_config in {run_config_path}")
    path = Path(value)
    if path.is_file():
        return path.resolve()
    fallback = PROJECT_ROOT / "02_training_dataset/config" / path.name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Checkpoint training config is unavailable: {value}")


def main() -> None:
    args = parse_args()
    adapter_config_path = resolve(args.config, PROJECT_ROOT)
    adapter_config = load_adapter_config(adapter_config_path)
    split_key = f"{args.split}_scene_indices"
    scene_indices = tuple(int(x) for x in adapter_config["dataset_split"][split_key])
    data_root = resolve(Path(adapter_config["paths"]["output_root"]), PROJECT_ROOT)
    checkpoint_path = resolve(args.ckpt, PROJECT_ROOT)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    training_config_path = checkpoint_training_config(checkpoint_path)
    training_adapter_config = load_adapter_config(training_config_path)
    evaluation_joint_target = joint_target_field(adapter_config)
    training_joint_target = joint_target_field(training_adapter_config)
    if evaluation_joint_target != training_joint_target:
        raise RuntimeError(
            "Joint-supervision mismatch: checkpoint was trained against "
            f"{training_joint_target!r}, but this evaluation dataset exposes "
            f"{evaluation_joint_target!r}. Build a label-matched held-out dataset "
            "before reporting loss_joint or total loss."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("joint_num", -1)) != 20:
        raise RuntimeError("Checkpoint is not a Wuji2 20-joint checkpoint")
    joint_order = tuple(checkpoint.get("joint_order", ()))
    if len(joint_order) != 20 or len(set(joint_order)) != 20:
        raise RuntimeError("Checkpoint joint_order must contain 20 unique joints")

    os.chdir(OFFICIAL_ROOT)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    config = load_config(OFFICIAL_ROOT / "configs/network/train_dex_ours.yaml")
    config.model.joint_num = 20
    config.model.voxel_size = float(config.data.voxel_size)
    model = get_model(config.model)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    dataset = Wuji2SceneDataset(
        data_root,
        scene_indices=scene_indices,
        is_train=False,
        sample_total=int(config.data.sample_total),
        k=int(config.data.k),
        max_point_dis=float(config.data.max_point_dis),
        voxel_size=float(config.data.voxel_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=minkowski_collate_fn,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    totals: dict[str, float] = {}
    sample_count = 0
    per_batch = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            _, result = model(batch)
            count = int(batch["point_clouds"].shape[0])
            metrics = {
                key: float(value.detach().mean().cpu())
                for key, value in result.items()
            }
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
            per_batch.append(
                {
                    "batch": batch_index,
                    "sample_count": count,
                    "scene_indices": batch["scene"].detach().cpu().reshape(-1).tolist(),
                    "view_indices": batch["view"].detach().cpu().reshape(-1).tolist(),
                    **metrics,
                }
            )
            sample_count += count
            print(
                f"[{args.split.upper()}] batch={batch_index + 1:03d}/{len(loader):03d} "
                f"samples={sample_count:03d}/{len(dataset):03d} loss={metrics['loss']:.6f}",
                flush=True,
            )
    means = {key: value / sample_count for key, value in totals.items()}
    report = {
        "status": "evaluation_complete",
        "split": args.split,
        "scene_indices": list(scene_indices),
        "view_count": len(dataset),
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
        "checkpoint_training_config": str(training_config_path),
        "joint_target_field": training_joint_target,
        "strict_checkpoint_load": True,
        "joint_num": 20,
        "joint_order": list(joint_order),
        "seed": args.seed,
        "mean_metrics": means,
        "elapsed_seconds": time.time() - started,
        "cuda_peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "per_batch": per_batch,
    }
    output_path = (
        resolve(args.output, PROJECT_ROOT)
        if args.output is not None
        else checkpoint_path.parent.parent / f"{args.split}_evaluation.json"
    )
    allowed_root = (ADAPTER_ROOT / "experiments").resolve()
    try:
        output_path.relative_to(allowed_root)
    except ValueError as exc:
        raise RuntimeError(f"Evaluation output must remain under {allowed_root}") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report, "per_batch": "omitted from console"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
