#!/usr/bin/env python3
"""Read-only GUI for per-scene force-adjusted filtering through Stage03."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config  # noqa: E402


CONFIG = PROJECT_ROOT / (
    "02_training_dataset/config/"
    "wuji2_train60_100seminal_256view_force_adjusted_legacy_v1.json"
)
STAGES = (
    ("Stage01 transform q_force", "01_transformed_object_grasps", "#60a5fa"),
    ("Stage02 scene/table collision", "02_scene_table_collision_filtered", "#818cf8"),
    ("Stage02b palm-path filter", "02b_enhanced_palm_center_path_filtered", "#c084fc"),
    ("Stage03 reference + graspness", "03_reference_points_and_surface_graspness", "#22c55e"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def completed(root: Path) -> int:
    return sum(1 for path in root.glob("scene_*/stage_manifest.json") if path.is_file())


def read_progress(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    args = parse_args()
    config = load_config(CONFIG)
    output_root = Path(config["paths"]["output_root"])
    labels = output_root / "grasp_label_stages"
    total = int(config["scope"]["scene_count"])
    progress_path = output_root / "force_adjusted_label_progress.json"

    plt.style.use("dark_background")
    figure, axes = plt.subplots(5, 1, figsize=(11.5, 8.2))
    figure.canvas.manager.set_window_title("Wuji2 force-adjusted scene filtering")
    heading = figure.suptitle(
        "Wuji2 q_force scene filtering (no simulation, no per-view loop)",
        fontsize=15,
        weight="bold",
    )
    footer = figure.text(0.5, 0.025, "", ha="center", fontsize=10)

    def draw(_frame: int) -> None:
        values = [completed(labels / directory) for _name, directory, _color in STAGES]
        overall = sum(values) / (len(STAGES) * total)
        percentages = [100.0 * overall] + [100.0 * value / total for value in values]
        colors = ["#f59e0b"] + [item[2] for item in STAGES]
        titles = [f"Overall per-scene pipeline: {sum(values)}/{len(STAGES) * total}"] + [
            f"{name}: {value}/{total} scenes"
            for (name, _directory, _color), value in zip(STAGES, values)
        ]
        for axis, percentage, color, title in zip(axes, percentages, colors, titles):
            axis.clear()
            axis.set_xlim(0, 100)
            axis.set_yticks([])
            axis.set_xticks([0, 25, 50, 75, 100])
            axis.grid(axis="x", alpha=0.18)
            axis.barh([0], [percentage], height=0.55, color=color)
            axis.text(min(percentage + 1.0, 94.0), 0, f"{percentage:.1f}%", va="center", weight="bold")
            axis.set_title(title, fontsize=11)
        progress = read_progress(progress_path)
        footer.set_text(
            f"status={progress.get('status', 'unknown')}  "
            f"current=scene_{int(progress.get('current_scene', 0)):04d}/"
            f"{progress.get('current_stage', '-')}  "
            f"updated={datetime.now().strftime('%H:%M:%S')}  "
            "closing this window does not stop filtering"
        )
        if values[-1] == total:
            heading.set_text("Wuji2 q_force scene filtering COMPLETE")
            heading.set_color("#4ade80")
        figure.tight_layout(rect=(0.04, 0.07, 0.98, 0.92))

    animation = FuncAnimation(
        figure,
        draw,
        interval=max(250, int(args.interval * 1000)),
        cache_frame_data=False,
    )
    draw(0)
    plt.show()
    _ = animation


if __name__ == "__main__":
    main()
