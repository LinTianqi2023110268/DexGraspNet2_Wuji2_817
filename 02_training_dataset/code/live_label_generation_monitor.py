#!/usr/bin/env python3
"""Show read-only Stage03/Stage04 progress for a Wuji2 dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def completed(root: Path) -> list[int]:
    result = []
    for path in root.glob("scene_*/stage_manifest.json"):
        try:
            result.append(int(path.parent.name.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return sorted(result)


def current_views(root: Path, done: set[int]) -> tuple[int | None, int]:
    rows = []
    for path in root.glob("scene_*"):
        try:
            index = int(path.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        if index not in done:
            rows.append((index, len(list(path.glob("view_*.npz")))))
    return max(rows, default=(None, 0), key=lambda row: row[0])


def eta_text(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    minutes, second = divmod(max(0, int(seconds)), 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def gpu_text() -> str:
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=1.5,
        ).splitlines()[0]
        util, used, total, temp = [value.strip() for value in line.split(",")]
        return f"GPU {util}%   VRAM {used}/{total} MiB   {temp}°C"
    except Exception:
        return "GPU status unavailable"


def main() -> None:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    configured_output = Path(config["paths"]["output_root"])
    output = (
        configured_output.resolve()
        if configured_output.is_absolute()
        else (PROJECT_ROOT / configured_output).resolve()
    )
    scene_total = int(config["scope"]["scene_count"])
    views_per_scene = int(config["scope"]["views_per_scene"])
    view_total = scene_total * views_per_scene
    labels = output / "grasp_label_stages"
    stage01 = labels / "01_transformed_object_grasps"
    stage02 = labels / "02_scene_table_collision_filtered"
    stage02b = labels / "02b_enhanced_palm_center_path_filtered"
    stage03 = labels / "03_reference_points_and_surface_graspness"
    stage04 = labels / "04_single_view_training_labels"

    plt.style.use("dark_background")
    figure, axes = plt.subplots(6, 1, figsize=(11.5, 9.2))
    figure.canvas.manager.set_window_title("Wuji2 DexGraspNet2 label progress")
    heading = figure.suptitle(
        "Wuji2 DexGraspNet2 — training-label generation", fontsize=16, weight="bold"
    )
    footer = figure.text(0.5, 0.03, "", ha="center", fontsize=11)
    history: deque[tuple[float, int]] = deque(maxlen=240)

    def draw(_frame: int) -> None:
        done01 = completed(stage01)
        done02 = completed(stage02)
        done02b = completed(stage02b)
        done03 = completed(stage03)
        done04 = completed(stage04)
        scene, partial = current_views(stage04, set(done04))
        view_count = min(view_total, len(done04) * views_per_scene + partial)
        history.append((time.monotonic(), view_count))
        eta = None
        if len(history) > 1 and history[-1][1] > history[0][1]:
            rate = (history[-1][1] - history[0][1]) / (history[-1][0] - history[0][0])
            eta = (view_total - view_count) / rate

        percentages = (
            100.0
            * (len(done01) + len(done02) + len(done02b) + len(done03) + view_count)
            / (4 * scene_total + view_total),
            100.0 * len(done01) / scene_total,
            100.0 * len(done02) / scene_total,
            100.0 * len(done02b) / scene_total,
            100.0 * len(done03) / scene_total,
            100.0 * view_count / view_total,
        )
        colors = ("#f59e0b", "#60a5fa", "#818cf8", "#c084fc", "#3b82f6", "#22c55e")
        current = (
            f"scene_{scene:04d}, {partial}/{views_per_scene} views"
            if scene is not None
            else "between scenes"
        )
        titles = (
            "Overall label pipeline",
            f"Stage01 force-adjusted scene grasps: {len(done01)}/{scene_total} scenes",
            f"Stage02 scene/table collision: {len(done02)}/{scene_total} scenes",
            f"Stage02b enhanced palm path: {len(done02b)}/{scene_total} scenes",
            f"Stage03 reference points + graspness: {len(done03)}/{scene_total} scenes",
            f"Stage04 single-view labels: {view_count}/{view_total} views\n"
            f"Current: {current}    ETA: {eta_text(eta)}",
        )
        for axis, percent, color, title in zip(axes, percentages, colors, titles):
            axis.clear()
            axis.set_xlim(0, 100)
            axis.set_yticks([])
            axis.set_xticks([0, 25, 50, 75, 100])
            axis.grid(axis="x", alpha=0.18)
            axis.barh([0], [percent], height=0.55, color=color)
            axis.text(min(percent + 1.0, 94.0), 0, f"{percent:.1f}%", va="center", weight="bold")
            axis.set_title(title, fontsize=12)
        footer.set_text(
            f"{gpu_text()}    Updated {datetime.now().strftime('%H:%M:%S')}    "
            "Closing this window does not stop generation"
        )
        if (
            len(done01) == scene_total
            and len(done02) == scene_total
            and len(done02b) == scene_total
            and len(done03) == scene_total
            and len(done04) == scene_total
        ):
            heading.set_text("Wuji2 DexGraspNet2 — label generation COMPLETE")
            heading.set_color("#4ade80")
        figure.tight_layout(rect=(0.04, 0.08, 0.98, 0.91))

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
