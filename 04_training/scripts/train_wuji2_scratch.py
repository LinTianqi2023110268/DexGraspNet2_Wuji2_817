#!/usr/bin/env python3
"""Train a new 20-joint Wuji2 DexGraspNet2 network from scratch.

The author's LEAP checkpoint is never loaded.  Its SHA256 is checked before
and after training, while all Wuji2 logs and checkpoints are written only
under this project's ``04_training/experiments`` directory.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ADAPTER_ROOT.parent
OFFICIAL_ROOT = PROJECT_ROOT / "03_prediction_network/official_core"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from wuji2_dgn2.project import source_path  # noqa: E402

OFFICIAL_CKPT = source_path("official_dexgraspnet2") / "experiments/dex_ours/ckpt/ckpt_50000.pth"
OFFICIAL_CKPT_SHA256 = "a081f8ed57bbf855d63fe29b2f06ae71995c1eea57d78884a501a4c69cab6a4b"
sys.path.insert(0, str(ADAPTER_ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT))

from wuji2_dataset import Wuji2SceneDataset, minkowski_collate_fn  # noqa: E402
from wuji2_dgn2.adapter_common import load_config as load_adapter_config  # noqa: E402
from src.network.model import get_model  # noqa: E402
from src.utils.config import load_config, to_dict  # noqa: E402
from src.utils.util import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "02_training_dataset/config/wuji2_train60_100seminal_256view_v1.json",
        help="Adapter experiment config containing the data root and scene split.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override official max_iter (default: 50000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override official batch_size (default: 8).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override official num_workers (default: 16).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Override official save_every (default: 5000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.set_defaults(live_monitor=True)
    parser.add_argument(
        "--live-monitor",
        dest="live_monitor",
        action="store_true",
        help="Open the non-blocking live loss window (default).",
    )
    parser.add_argument(
        "--no-live-monitor",
        dest="live_monitor",
        action="store_false",
        help="Do not open the live loss window.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def metrics_to_float(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().mean().cpu())
        for key, value in values.items()
    }


def save_checkpoint(path, model, optimizer, scheduler, iteration, config, joint_order):
    torch.save(
        {
            "training_origin": (
                "random initialization using the author's model constructors and "
                "initialization rules; author checkpoint not loaded"
            ),
            "joint_num": 20,
            "joint_order": list(joint_order),
            "iteration": int(iteration),
            "model_config": to_dict(config.model),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def launch_live_monitor(log_path: Path, expected_iterations: int) -> int | None:
    """Launch a detached CPU-only plotter; failure never aborts training."""

    if not os.environ.get("DISPLAY"):
        print("[MONITOR] DISPLAY is unset; live loss window was not opened", flush=True)
        return None
    monitor_script = PROJECT_ROOT / "04_training/scripts/monitor_wuji2_training_loss.py"
    monitor_log = log_path.with_name("loss_monitor.log")
    stream = monitor_log.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(monitor_script),
                "--log",
                str(log_path),
                "--expected-iterations",
                str(expected_iterations),
            ],
            cwd=ADAPTER_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        stream.close()
    print(
        f"[MONITOR] live loss window pid={process.pid} log={monitor_log}",
        flush=True,
    )
    return int(process.pid)


def evaluate_model(model, loader, device, seed: int) -> dict[str, float]:
    """Evaluate every view while preserving the training RNG streams exactly."""

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.eval()
    totals: dict[str, float] = {}
    sample_count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                _, result = model(batch)
                current_batch = int(batch["point_clouds"].shape[0])
                for key, value in metrics_to_float(result).items():
                    totals[key] = totals.get(key, 0.0) + value * current_batch
                sample_count += current_batch
    finally:
        if was_training:
            model.train()
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    if sample_count == 0:
        raise RuntimeError("Validation loader produced no samples")
    return {key: value / sample_count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    # Some official dependencies change cwd while importing.  CLI-relative
    # project paths must therefore never depend on the process cwd here.
    adapter_config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (PROJECT_ROOT / args.config).resolve()
    )
    adapter_config = load_adapter_config(adapter_config_path)
    split_config = adapter_config["dataset_split"]
    # The official YAML is the single source of truth for network and optimizer
    # defaults.  The only architectural change below is LEAP 16 joints ->
    # Wuji2 20 joints.  CLI options remain available for explicit experiments.
    config = load_config(OFFICIAL_ROOT / "configs/network/train_dex_ours.yaml")
    iterations = int(
        config.max_iter
        if args.iterations is None
        else args.iterations
    )
    batch_size = int(
        config.batch_size
        if args.batch_size is None
        else args.batch_size
    )
    save_every = int(
        config.save_every
        if args.save_every is None
        else args.save_every
    )
    num_workers = int(
        config.num_workers
        if args.num_workers is None
        else args.num_workers
    )
    seed = int(config.seed if args.seed is None else args.seed)
    learning_rate = float(config.lr if args.lr is None else args.lr)
    learning_rate_min = float(config.lr_min if args.lr_min is None else args.lr_min)
    log_every = int(config.log_every if args.log_every is None else args.log_every)
    validation_every = int(config.val_every)
    train_scenes = tuple(int(value) for value in split_config["train_scene_indices"])
    validation_scenes = tuple(
        int(value) for value in split_config["validation_scene_indices"]
    )
    test_scenes = tuple(int(value) for value in split_config["test_scene_indices"])
    all_scenes = train_scenes + validation_scenes + test_scenes
    if len(set(all_scenes)) != len(all_scenes):
        raise ValueError("Train/validation/test scene splits must be disjoint")
    if (
        iterations <= 0
        or batch_size <= 0
        or save_every <= 0
        or log_every <= 0
        or validation_every <= 0
        or num_workers < 0
        or learning_rate <= 0
        or learning_rate_min < 0
    ):
        raise ValueError(
            "official/override training parameters are invalid"
        )
    if args.output_dir is None:
        output_dir = (
            ADAPTER_ROOT
            / "experiments"
            / f"{adapter_config['experiment_name']}_scratch"
        ).resolve()
    else:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir.is_absolute()
            else (PROJECT_ROOT / args.output_dir).resolve()
        )
    allowed_output_root = (ADAPTER_ROOT / "experiments").resolve()
    try:
        output_dir.relative_to(allowed_output_root)
    except ValueError as exc:
        raise RuntimeError(
            "Wuji2 checkpoints must stay under this project's training directory: "
            f"{allowed_output_root}; requested {output_dir}"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (output_dir / ".training.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.seek(0)
        owner = lock_stream.read().strip() or "unknown process"
        raise RuntimeError(
            f"Another trainer already owns {output_dir}: {owner}"
        ) from exc
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lock_stream.flush()
    before_hash = sha256(OFFICIAL_CKPT)
    if before_hash != OFFICIAL_CKPT_SHA256:
        raise RuntimeError(
            "Author checkpoint SHA256 changed; refusing to train until audited: "
            f"{before_hash}"
        )
    os.chdir(OFFICIAL_ROOT)
    set_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    config.model.joint_num = 20
    config.model.voxel_size = float(config.data.voxel_size)
    data_root = Path(adapter_config["paths"]["output_root"]).resolve()
    dataset = Wuji2SceneDataset(
        data_root,
        scene_indices=train_scenes,
        is_train=True,
        repeat_length=max(iterations * batch_size * 2, 10000),
        sample_total=int(config.data.sample_total),
        k=int(config.data.k),
        max_point_dis=float(config.data.max_point_dis),
        voxel_size=float(config.data.voxel_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=minkowski_collate_fn,
    )
    iterator = iter(loader)
    validation_dataset = None
    validation_loader = None
    if validation_scenes:
        validation_dataset = Wuji2SceneDataset(
            data_root,
            scene_indices=validation_scenes,
            is_train=False,
            sample_total=int(config.data.sample_total),
            k=int(config.data.k),
            max_point_dis=float(config.data.max_point_dis),
            voxel_size=float(config.data.voxel_size),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=minkowski_collate_fn,
        )

    # No torch.load call occurs anywhere in this training program.
    model = get_model(config.model).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=learning_rate_min
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    validation_log_path = output_dir / "validation_metrics.jsonl"
    run_path = output_dir / "run_config.json"
    run_config = {
        "purpose": "new Wuji2 20-joint network trained from random initialization",
        "author_checkpoint_loaded": False,
        "author_checkpoint_path": str(OFFICIAL_CKPT),
        "author_checkpoint_sha256_guard": OFFICIAL_CKPT_SHA256,
        "adapter_config": str(adapter_config_path),
        "data_root": str(data_root),
        "train_scenes": list(train_scenes),
        "validation_scenes_reserved": list(validation_scenes),
        "test_scenes_reserved": list(test_scenes),
        "views_per_scene": int(adapter_config["scope"]["views_per_scene"]),
        "training_view_samples": len(train_scenes)
        * int(adapter_config["scope"]["views_per_scene"]),
        "iterations": iterations,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "save_every": save_every,
        "seed": seed,
        "learning_rate": learning_rate,
        "learning_rate_min": learning_rate_min,
        "gradient_clip": float(config.grad_clip),
        "log_every": log_every,
        "validation_every": validation_every,
        "joint_num": 20,
        "joint_order": list(dataset.joint_order),
        "base_architecture_config": str(
            OFFICIAL_ROOT / "configs/network/train_dex_ours.yaml"
        ),
        "output_dir": str(output_dir),
        "split_policy": split_config["policy"],
        "initialization_policy": (
            "Official DexGraspNet2 constructors/default initializers with seed "
            f"{seed}; joint output dimension changed from 16 to 20; no checkpoint loaded"
        ),
        "official_parameter_source": {
            "max_iter": int(config.max_iter),
            "batch_size": int(config.batch_size),
            "num_workers": int(config.num_workers),
            "save_every": int(config.save_every),
            "log_every": int(config.log_every),
            "val_every": int(config.val_every),
            "lr": float(config.lr),
            "lr_min": float(config.lr_min),
            "grad_clip": float(config.grad_clip),
            "seed": int(config.seed),
        },
    }
    run_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Truncate only our new experiment log before this from-scratch run.
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("")
    with validation_log_path.open("w", encoding="utf-8") as stream:
        stream.write("")
    monitor_pid = (
        launch_live_monitor(log_path, iterations) if args.live_monitor else None
    )
    print(
        "[START] random_init=True author_ckpt_loaded=False "
        f"train_scenes={len(train_scenes)} "
        f"train_views={len(train_scenes) * int(adapter_config['scope']['views_per_scene'])} "
        f"val_scenes={len(validation_scenes)} test_scenes={len(test_scenes)} "
        f"batch={batch_size} iterations={iterations}",
        flush=True,
    )
    start = time.time()
    first_metrics = None
    last_metrics = None
    best_validation_loss = float("inf")
    best_validation_iteration = None
    best_validation_metrics = None
    for iteration in range(1, iterations + 1):
        batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, result = model(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at iteration {iteration}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
        optimizer.step()
        scheduler.step()
        metrics = metrics_to_float(result)
        record = {
            "iteration": iteration,
            "elapsed_seconds": time.time() - start,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **metrics,
        }
        if first_metrics is None:
            first_metrics = record
        last_metrics = record
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
        if iteration == 1 or iteration % log_every == 0:
            print(
                f"iter={iteration:04d}/{iterations} "
                f"loss={metrics['loss']:.6f} "
                f"obj={metrics['loss_objectness']:.5f} "
                f"gs={metrics['loss_graspness']:.5f} "
                f"diff={metrics['loss_diffusion']:.5f} "
                f"joint={metrics['loss_joint']:.5f} "
                f"acc={metrics['acc_objectness']:.3f} "
                f"lr={record['learning_rate']:.7f}",
                flush=True,
            )
        if iteration % save_every == 0 or iteration == iterations:
            checkpoint_path = checkpoint_dir / f"wuji2_scratch_{iteration:06d}.pth"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                iteration,
                config,
                dataset.joint_order,
            )
        if validation_loader is not None and (
            iteration % validation_every == 0 or iteration == iterations
        ):
            validation_metrics = evaluate_model(
                # Fixed labels, diffusion timesteps and noise at every
                # validation pass make validation losses directly comparable.
                model, validation_loader, device, seed + 100000
            )
            validation_record = {
                "iteration": iteration,
                "view_count": len(validation_dataset),
                **validation_metrics,
            }
            with validation_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(validation_record, ensure_ascii=False) + "\n")
                stream.flush()
            print(
                f"[VAL] iter={iteration:06d} views={len(validation_dataset)} "
                f"loss={validation_metrics['loss']:.6f} "
                f"obj={validation_metrics['loss_objectness']:.5f} "
                f"gs={validation_metrics['loss_graspness']:.5f} "
                f"diff={validation_metrics['loss_diffusion']:.5f} "
                f"joint={validation_metrics['loss_joint']:.5f} "
                f"acc={validation_metrics['acc_objectness']:.3f}",
                flush=True,
            )
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_validation_iteration = iteration
                best_validation_metrics = validation_metrics
                save_checkpoint(
                    checkpoint_dir / "wuji2_best_validation.pth",
                    model,
                    optimizer,
                    scheduler,
                    iteration,
                    config,
                    dataset.joint_order,
                )

    after_hash = sha256(OFFICIAL_CKPT)
    if after_hash != before_hash:
        raise RuntimeError("Author checkpoint changed during Wuji2 training")
    summary = {
        "status": "training_complete",
        "author_checkpoint_unchanged": True,
        "author_checkpoint_sha256": after_hash,
        "first_metrics": first_metrics,
        "last_metrics": last_metrics,
        "elapsed_seconds": time.time() - start,
        "cuda_peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "final_checkpoint": str(
            checkpoint_dir / f"wuji2_scratch_{iterations:06d}.pth"
        ),
        "best_validation_checkpoint": (
            str(checkpoint_dir / "wuji2_best_validation.pth")
            if best_validation_iteration is not None
            else None
        ),
        "best_validation_iteration": best_validation_iteration,
        "best_validation_metrics": best_validation_metrics,
        "live_metrics": str(log_path),
        "validation_metrics": str(validation_log_path),
        "live_monitor_pid_at_start": monitor_pid,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    lock_stream.close()


if __name__ == "__main__":
    main()
