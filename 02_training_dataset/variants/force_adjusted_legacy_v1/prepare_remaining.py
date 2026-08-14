#!/usr/bin/env python3
"""Resume the force-adjusted Stage01-04 pipeline without redoing complete stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
CODE_DIR = PROJECT_ROOT / "02_training_dataset/code"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config, write_json_atomic  # noqa: E402


CONFIG = PROJECT_ROOT / (
    "02_training_dataset/config/"
    "wuji2_train60_100seminal_256view_force_adjusted_legacy_v1.json"
)
STAGES = (
    ("01_transform", "01_transformed_object_grasps", "build_wuji2_scene_grasps.py", False),
    ("02_collision", "02_scene_table_collision_filtered", "filter_wuji2_scene_collisions.py", True),
    ("02b_palm_path", "02b_enhanced_palm_center_path_filtered", "filter_wuji2_palm_center_paths.py", False),
    ("03_graspness", "03_reference_points_and_surface_graspness", "build_wuji2_reference_graspness.py", True),
    ("04_single_view", "04_single_view_training_labels", "assign_wuji2_single_view_labels.py", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", type=int, action="append")
    parser.add_argument(
        "--stop-after-stage",
        choices=tuple(stage[0] for stage in STAGES),
        default="03_graspness",
        help=(
            "Default stops after per-scene reference/graspness generation. "
            "Choose 04_single_view only when 256-view training labels are required."
        ),
    )
    return parser.parse_args()


def stage_complete(output_root: Path, directory: str, scene: int) -> bool:
    return (
        output_root
        / "grasp_label_stages"
        / directory
        / f"scene_{scene:04d}"
        / "stage_manifest.json"
    ).is_file()


def counts(output_root: Path, scene_total: int) -> dict[str, int]:
    return {
        name: sum(stage_complete(output_root, directory, scene) for scene in range(scene_total))
        for name, directory, _script, _gpu in STAGES
    }


def main() -> None:
    args = parse_args()
    config = load_config(CONFIG)
    output_root = Path(config["paths"]["output_root"])
    scene_total = int(config["scope"]["scene_count"])
    scenes = list(range(scene_total)) if args.scene is None else sorted(set(args.scene))
    stop_index = next(
        index for index, stage in enumerate(STAGES)
        if stage[0] == args.stop_after_stage
    )
    requested_stages = STAGES[: stop_index + 1]
    progress_path = output_root / "force_adjusted_label_progress.json"
    for scene in scenes:
        if not 0 <= scene < scene_total:
            raise ValueError(f"Scene outside configured range: {scene}")
        for stage_name, directory, script, needs_device in requested_stages:
            if stage_complete(output_root, directory, scene):
                print(f"[SKIP COMPLETE] scene={scene:04d} stage={stage_name}", flush=True)
                continue
            progress = {
                "status": "running",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "current_scene": scene,
                "current_stage": stage_name,
                "device": args.device,
                "completed": counts(output_root, scene_total),
            }
            write_json_atomic(progress_path, progress)
            command = [
                sys.executable,
                str(CODE_DIR / script),
                "--config",
                str(CONFIG),
                "--scene",
                str(scene),
            ]
            if needs_device:
                command.extend(("--device", args.device))
            print(f"[RUN] scene={scene:04d} stage={stage_name}", flush=True)
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    write_json_atomic(
        progress_path,
        {
            "status": (
                "scene_grasp_filtering_complete"
                if args.stop_after_stage == "03_graspness"
                else "complete"
            ),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "device": args.device,
            "stop_after_stage": args.stop_after_stage,
            "completed": counts(output_root, scene_total),
        },
    )
    print(
        f"[COMPLETE] requested force-adjusted scenes through {args.stop_after_stage}",
        flush=True,
    )


if __name__ == "__main__":
    main()
