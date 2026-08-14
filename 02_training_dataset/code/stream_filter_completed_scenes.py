#!/usr/bin/env python3
"""Continuously annotate completed Wuji2 scenes without deleting any grasp.

The Isaac Sim generator writes ``scene_manifest.json`` only after a scene has
passed physical stability checks and all configured views have been captured.
This watcher treats that file as the completion marker and runs, in order:

1. object-frame eligible q_opt -> scene-world transformation (CPU),
2. paper-style PREGRASP scene/table collision metrics (CPU by default),
3. the separately labelled Wuji2 semantic-palm centerline safety metrics.

Stage 02 and 02B retain every Stage-01 eligible row.  They append masks,
clearances and reject-reason bits; no grasp is physically removed.  The default
CPU mode and positive niceness are intentionally safe to run beside the active
Isaac Sim camera producer.  Use CUDA only after that producer stops, unless the
operator explicitly opts into GPU sharing.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from wuji2_dgn2.adapter_common import load_config, write_json_atomic  # noqa: E402


STAGE_SPECS = (
    (
        "transform",
        "01_transformed_object_grasps",
        "build_wuji2_scene_grasps.py",
        None,
    ),
    (
        "collision",
        "02_scene_table_collision_filtered",
        "filter_wuji2_scene_collisions.py",
        "wuji2-nondestructive-paper-mask-v1",
    ),
    (
        "path",
        "02b_enhanced_palm_center_path_filtered",
        "filter_wuji2_palm_center_paths.py",
        "wuji2-nondestructive-path-mask-v1",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "02_training_dataset/config/wuji2_train60_100seminal_256view_v1.json",
    )
    parser.add_argument(
        "--through",
        choices=tuple(spec[0] for spec in STAGE_SPECS),
        default="path",
        help="Last stage to run for each completed scene.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Collision device. CPU is safe beside Isaac Sim; use cuda:0 later.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Stage-02 collision batch size; 16 is a conservative CPU default.",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument(
        "--scene",
        type=int,
        action="append",
        help="Restrict to a scene index; may be repeated.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process currently completed scenes once instead of watching.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute stages even when a compatible manifest already exists.",
    )
    parser.add_argument(
        "--allow-gpu-sharing",
        action="store_true",
        help="Permit CUDA filtering while the Isaac scene generator is active.",
    )
    return parser.parse_args()


def configured_scene_indices(config: dict) -> list[int]:
    split = config["dataset_split"]
    result = []
    for key in (
        "train_scene_indices",
        "validation_scene_indices",
        "test_scene_indices",
    ):
        result.extend(int(value) for value in split[key])
    if len(result) != len(set(result)):
        raise ValueError("Dataset scene splits overlap")
    return sorted(result)


def completed_scene_indices(config: dict, selected: set[int] | None) -> list[int]:
    root = Path(config["paths"]["output_root"])
    result = []
    for scene_index in configured_scene_indices(config):
        if selected is not None and scene_index not in selected:
            continue
        marker = root / "scenes" / f"scene_{scene_index:04d}" / "scene_manifest.json"
        if marker.is_file():
            result.append(scene_index)
    return result


def stage_manifest_path(config: dict, stage_directory: str, scene: int) -> Path:
    return (
        Path(config["paths"]["output_root"])
        / config["grasp_label_generation"]["stage_directory_name"]
        / stage_directory
        / f"scene_{scene:04d}"
        / "stage_manifest.json"
    )


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def stage_is_current(
    config: dict,
    scene: int,
    stage_directory: str,
    required_revision: str | None,
) -> bool:
    manifest_path = stage_manifest_path(config, stage_directory, scene)
    manifest = read_json(manifest_path)
    if int(manifest.get("scene_index", -1)) != scene:
        return False
    if manifest.get("stage") != stage_directory:
        return False
    if manifest.get("diagnostic_prefix_limit") is not None:
        return False
    if required_revision is not None and manifest.get(
        "output_policy_revision"
    ) != required_revision:
        return False
    scene_manifest = (
        Path(config["paths"]["output_root"])
        / "scenes"
        / f"scene_{scene:04d}"
        / "scene_manifest.json"
    )
    if not scene_manifest.is_file() or manifest_path.stat().st_mtime < scene_manifest.stat().st_mtime:
        return False
    records = manifest.get("object_records")
    if not isinstance(records, list) or not records:
        return False
    for record in records:
        output = record.get("output_npz")
        if output is None or not Path(output).is_file():
            return False
    return True


def isaac_scene_generator_running() -> bool:
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if b"generate_scenes_and_views.py" in command:
            return True
    return False


def run_stage(
    args: argparse.Namespace,
    config_path: Path,
    scene: int,
    stage_name: str,
    script_name: str,
) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / script_name),
        "--config",
        str(config_path),
        "--scene",
        str(scene),
    ]
    if stage_name == "collision":
        command.extend(
            ("--device", args.device, "--batch-size", str(args.batch_size))
        )
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] "
        f"scene={scene:04d} stage={stage_name} command={' '.join(command)}",
        flush=True,
    )
    if args.dry_run:
        return
    environment = os.environ.copy()
    threads = str(args.torch_threads)
    environment.update(
        {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
        }
    )
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def process_scene(
    args: argparse.Namespace,
    config: dict,
    config_path: Path,
    scene: int,
) -> list[str]:
    last_index = next(
        index for index, spec in enumerate(STAGE_SPECS) if spec[0] == args.through
    )
    completed = []
    for stage_name, stage_directory, script_name, revision in STAGE_SPECS[
        : last_index + 1
    ]:
        current = stage_is_current(
            config, scene, stage_directory, revision
        )
        if current and not args.force:
            completed.append(stage_name)
            continue
        run_stage(args, config_path, scene, stage_name, script_name)
        if not args.dry_run and not stage_is_current(
            config, scene, stage_directory, revision
        ):
            raise RuntimeError(
                f"Stage {stage_name} did not produce a valid manifest for scene {scene:04d}"
            )
        completed.append(stage_name)
    return completed


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.torch_threads <= 0:
        raise ValueError("--batch-size and --torch-threads must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    expected = set(configured_scene_indices(config))
    selected = set(args.scene) if args.scene else None
    if selected is not None:
        unknown = selected - expected
        if unknown:
            raise ValueError(f"Scenes outside configured splits: {sorted(unknown)}")
    if args.device.startswith("cuda") and isaac_scene_generator_running() and not args.allow_gpu_sharing:
        raise RuntimeError(
            "Isaac Sim scene generation is active. Use the default --device cpu, "
            "wait for generation to finish, or explicitly pass --allow-gpu-sharing."
        )

    label_root = (
        Path(config["paths"]["output_root"])
        / config["grasp_label_generation"]["stage_directory_name"]
    )
    label_root.mkdir(parents=True, exist_ok=True)
    lock_path = label_root / ".stream_filter.lock"
    progress_path = label_root / "stream_filter_progress.json"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another streaming filter already owns {lock_path}"
            ) from error
        if args.nice > 0 and not args.dry_run:
            os.nice(args.nice)
        print(
            f"[STREAM FILTER] config={config_path} through={args.through} "
            f"device={args.device} batch={args.batch_size} threads={args.torch_threads} "
            f"once={args.once} dry_run={args.dry_run}",
            flush=True,
        )
        failures: dict[str, str] = {}
        while True:
            complete = completed_scene_indices(config, selected)
            processed = []
            for scene in complete:
                try:
                    stages = process_scene(args, config, config_path, scene)
                    processed.append({"scene": scene, "stages": stages})
                    failures.pop(str(scene), None)
                except Exception as error:  # keep the watcher auditable and alive
                    failures[str(scene)] = f"{type(error).__name__}: {error}"
                    print(
                        f"[SCENE {scene:04d} ERROR] {failures[str(scene)]}",
                        file=sys.stderr,
                        flush=True,
                    )
            report = {
                "schema_version": 1,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "config": str(config_path),
                "output_root": config["paths"]["output_root"],
                "through": args.through,
                "device": args.device,
                "completed_scene_markers": complete,
                "processed_or_already_current": processed,
                "failures": failures,
                "dry_run": bool(args.dry_run),
            }
            if not args.dry_run:
                write_json_atomic(progress_path, report)
            if args.once or args.dry_run:
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
