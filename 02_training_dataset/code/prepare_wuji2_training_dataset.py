#!/usr/bin/env python3
"""Run Wuji2 scene-label preprocessing for every configured dataset scene.

The stages remain separate and auditable, but this driver makes a larger
dataset reproducible without manually issuing five commands per scene.
Run it in the ``graspnet2.0`` environment after Isaac Sim has accepted and
captured all configured scenes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ADAPTER_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config  # noqa: E402

STAGES = (
    ("01_transform", "build_wuji2_scene_grasps.py", False),
    ("02_collision", "filter_wuji2_scene_collisions.py", True),
    ("02b_palm_path", "filter_wuji2_palm_center_paths.py", False),
    ("03_graspness", "build_wuji2_reference_graspness.py", True),
    ("04_single_view", "assign_wuji2_single_view_labels.py", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ADAPTER_ROOT
        / "config"
        / "wuji2_train60_100seminal_256view_v1.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scene",
        type=int,
        action="append",
        help="Process only this scene; may be supplied more than once.",
    )
    parser.add_argument(
        "--start-stage",
        choices=tuple(stage[0] for stage in STAGES),
        default=STAGES[0][0],
        help="Resume from this stage, inclusively.",
    )
    return parser.parse_args()


def configured_scenes(config: dict) -> list[int]:
    split = config["dataset_split"]
    result = []
    for key in (
        "train_scene_indices",
        "validation_scene_indices",
        "test_scene_indices",
    ):
        result.extend(int(value) for value in split[key])
    if len(result) != len(set(result)):
        raise ValueError("Configured dataset scene splits overlap")
    return result


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    scenes = configured_scenes(config) if args.scene is None else args.scene
    expected = configured_scenes(config)
    unknown = sorted(set(scenes) - set(expected))
    if unknown:
        raise ValueError(f"Scenes are outside the configured split: {unknown}")

    start_index = next(
        index for index, stage in enumerate(STAGES) if stage[0] == args.start_stage
    )
    stages = STAGES[start_index:]
    print(
        f"[DATASET PREP] scenes={scenes} stages={[stage[0] for stage in stages]} "
        f"device={args.device}",
        flush=True,
    )
    for scene_index in scenes:
        for stage_name, script_name, needs_device in stages:
            command = [
                sys.executable,
                str(SCRIPT_DIR / script_name),
                "--config",
                str(config_path),
                "--scene",
                str(scene_index),
            ]
            if needs_device:
                command.extend(("--device", args.device))
            print(
                f"[RUN] scene={scene_index:04d} stage={stage_name}",
                flush=True,
            )
            subprocess.run(command, check=True, cwd=ADAPTER_ROOT)

    # Preserve the historical training-set behavior, but do not hard-code an
    # empty ``train`` split for independently generated validation/test sets.
    # ``--scene`` still overrides the selected split inside the checker.
    split_to_key = {
        "train": "train_scene_indices",
        "validation": "validation_scene_indices",
        "test": "test_scene_indices",
    }
    check_split = next(
        (
            split_name
            for split_name, split_key in split_to_key.items()
            if config["dataset_split"][split_key]
        ),
        None,
    )
    if check_split is None:
        raise RuntimeError("No non-empty dataset split is configured")
    check_command = [
        sys.executable,
        str(SCRIPT_DIR / "check_wuji2_dataset.py"),
        "--config",
        str(config_path),
        "--split",
        check_split,
    ]
    if args.scene is not None:
        for scene_index in scenes:
            check_command.extend(("--scene", str(scene_index)))
    subprocess.run(
        check_command,
        check=True,
        cwd=ADAPTER_ROOT,
    )
    print(
        f"[COMPLETE] all configured Wuji2 labels are ready; "
        f"verified_split={check_split}",
        flush=True,
    )


if __name__ == "__main__":
    main()
