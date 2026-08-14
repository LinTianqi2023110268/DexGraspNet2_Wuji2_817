#!/usr/bin/env python3
"""Low-overhead live dashboard for Wuji2 DexGraspNet2 JSONL metrics.

This process is read-only. Closing the window never stops or changes training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_LOG = (
    PROJECT_ROOT
    / "04_training/experiments/"
    "wuji2_dexgraspnet2_train60_100seminal_256view_v1_scratch/metrics.jsonl"
)
METRIC_KEYS = (
    "loss",
    "loss_objectness",
    "loss_graspness",
    "loss_joint",
    "loss_diffusion",
    "acc_objectness",
    "abs_graspness",
    "abs_dis_joint",
    "learning_rate",
    "elapsed_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--refresh-ms", type=int, default=1000)
    parser.add_argument("--smooth-window", type=int, default=30)
    parser.add_argument(
        "--plot-window",
        type=int,
        default=2000,
        help="Show only the newest N iterations; 0 shows the complete history.",
    )
    parser.add_argument("--expected-iterations", type=int, default=50000)
    parser.add_argument("--stale-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", help="Save one PNG and exit.")
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args()


class IncrementalJsonlReader:
    """Read only newly appended complete JSONL lines after the first refresh."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = ""
        self.file_identity: tuple[int, int] | None = None

    def read_new(self) -> list[dict]:
        if not self.path.is_file():
            return []
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.file_identity != identity or stat.st_size < self.offset:
            self.offset = 0
            self.pending = ""
            self.file_identity = identity
        with self.path.open("r", encoding="utf-8") as stream:
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        if not chunk:
            return []
        text = self.pending + chunk
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.pending = lines.pop()
        else:
            self.pending = ""
        records: list[dict] = []
        for line in lines:
            try:
                record = json.loads(line)
                iteration = int(record["iteration"])
                record["iteration"] = iteration
                records.append(record)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A malformed monitoring line must never affect training.
                continue
        return records


class MetricStore:
    def __init__(self) -> None:
        self.records: dict[int, dict] = {}

    def extend(self, records: list[dict]) -> None:
        for record in records:
            self.records[int(record["iteration"])] = record

    def ordered(self) -> list[dict]:
        return [self.records[key] for key in sorted(self.records)]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.copy()
    width = min(int(window), int(values.size))
    sums = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty_like(values, dtype=float)
    result[: width - 1] = sums[1:width] / np.arange(1, width)
    result[width - 1 :] = (sums[width:] - sums[:-width]) / width
    return result


def finite_float(record: dict, key: str, default: float = math.nan) -> float:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def estimate_rate_and_eta(
    records: list[dict], expected_iterations: int
) -> tuple[float | None, float | None]:
    if len(records) < 2:
        return None, None
    sample = records[-min(300, len(records)) :]
    first, last = sample[0], sample[-1]
    dt = finite_float(last, "elapsed_seconds") - finite_float(first, "elapsed_seconds")
    di = int(last["iteration"]) - int(first["iteration"])
    if not math.isfinite(dt) or dt <= 0 or di <= 0:
        return None, None
    rate = di / dt
    remaining = max(0, expected_iterations - int(last["iteration"]))
    return rate, remaining / rate


def robust_log_limits(value_groups: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([group for group in value_groups if group.size])
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return 1.0e-5, 1.0e2
    low = max(float(np.percentile(values, 1)) * 0.55, 1.0e-8)
    high = max(float(np.percentile(values, 99)) * 1.8, low * 20.0)
    return low, high


def main() -> None:
    args = parse_args()
    log_path = args.log.resolve()
    cache = log_path.parent / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))

    import matplotlib

    matplotlib.use("Agg" if args.once else "TkAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    # Explicit CJK fallback keeps Chinese labels readable in both the live
    # Xorg window and headless PNG snapshots.
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.style.use("dark_background")
    figure = plt.figure(figsize=(13.2, 8.4), facecolor="#10151d")
    figure.canvas.manager.set_window_title("Wuji2 DexGraspNet2 训练监控（关闭不影响训练）")
    grid = figure.add_gridspec(
        3, 2, height_ratios=(0.30, 1.0, 1.0), hspace=0.34, wspace=0.23
    )
    dashboard = figure.add_subplot(grid[0, :])
    ax_total = figure.add_subplot(grid[1, 0])
    ax_components = figure.add_subplot(grid[1, 1])
    ax_diffusion = figure.add_subplot(grid[2, 0])
    ax_quality = figure.add_subplot(grid[2, 1])

    dashboard.set_xlim(0, 1)
    dashboard.set_ylim(0, 1)
    dashboard.axis("off")
    title = dashboard.text(
        0.0, 0.91, "Wuji2 DexGraspNet2 — 等待训练日志", fontsize=16,
        weight="bold", color="#f2f5f8", va="top",
    )
    status_text = dashboard.text(0.0, 0.52, "状态：等待", fontsize=11, color="#ffcc66")
    detail_text = dashboard.text(0.0, 0.16, "", fontsize=10.5, color="#cbd5e1")
    progress_bg = Rectangle((0.57, 0.45), 0.41, 0.22, color="#293341", ec="#536273")
    progress_fill = Rectangle((0.57, 0.45), 0.0, 0.22, color="#2dbf78")
    dashboard.add_patch(progress_bg)
    dashboard.add_patch(progress_fill)
    progress_text = dashboard.text(
        0.775, 0.56, "0.00%", ha="center", va="center", fontsize=11,
        weight="bold", color="white",
    )
    path_text = dashboard.text(
        0.57, 0.16, str(log_path), fontsize=8.2, color="#8ea0b5",
        ha="left", va="center",
    )

    for axis in (ax_total, ax_components, ax_diffusion, ax_quality):
        axis.set_facecolor("#151d27")
        axis.grid(True, which="both", color="#64748b", alpha=0.19)
        axis.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#354255")

    total_raw, = ax_total.plot([], [], color="#58a6ff", alpha=0.19, lw=0.7)
    total_smooth, = ax_total.plot([], [], color="#58a6ff", lw=2.1, label="总损失（平滑）")
    component_specs = (
        ("loss_objectness", "物体分类", "#f59e0b"),
        ("loss_graspness", "抓取评分", "#a78bfa"),
        ("loss_joint", "20关节", "#34d399"),
    )
    component_lines = {
        key: ax_components.plot([], [], color=color, lw=1.8, label=label)[0]
        for key, label, color in component_specs
    }
    diffusion_raw, = ax_diffusion.plot([], [], color="#fb7185", alpha=0.18, lw=0.7)
    diffusion_smooth, = ax_diffusion.plot(
        [], [], color="#fb7185", lw=2.1, label="扩散损失（原始数值）"
    )
    accuracy_line, = ax_quality.plot(
        [], [], color="#22d3ee", lw=2.0, label="物体分类准确率"
    )
    grasp_abs_line, = ax_quality.plot(
        [], [], color="#fde047", lw=1.5, alpha=0.9,
        label="抓取评分绝对误差",
    )

    ax_total.set_title("总损失", color="#e2e8f0", weight="bold")
    ax_components.set_title("分项监督损失", color="#e2e8f0", weight="bold")
    ax_diffusion.set_title("姿态扩散损失（不再乘 10）", color="#e2e8f0", weight="bold")
    ax_quality.set_title("分类准确率与抓取评分误差", color="#e2e8f0", weight="bold")
    for axis in (ax_total, ax_components, ax_diffusion):
        axis.set_yscale("log")
        axis.set_ylabel("loss（对数坐标）", color="#cbd5e1")
        axis.legend(loc="upper right", fontsize=8, framealpha=0.25)
    ax_quality.set_ylabel("数值", color="#cbd5e1")
    ax_quality.legend(loc="upper right", fontsize=8, framealpha=0.25)
    for axis in (ax_diffusion, ax_quality):
        axis.set_xlabel("训练迭代", color="#cbd5e1")

    reader = IncrementalJsonlReader(log_path)
    store = MetricStore()
    last_seen_wall_time: float | None = None

    def update(_frame: int) -> tuple:
        nonlocal last_seen_wall_time
        new_records = reader.read_new()
        if new_records:
            store.extend(new_records)
            last_seen_wall_time = time.time()
        records = store.ordered()
        if not records:
            status_text.set_text("状态：等待 metrics.jsonl 出现新记录")
            status_text.set_color("#ffcc66")
            return ()

        last = records[-1]
        iteration = int(last["iteration"])
        expected = max(1, int(args.expected_iterations))
        fraction = float(np.clip(iteration / expected, 0.0, 1.0))
        rate, eta = estimate_rate_and_eta(records, expected)
        elapsed = finite_float(last, "elapsed_seconds")
        age = None if last_seen_wall_time is None else time.time() - last_seen_wall_time
        if iteration >= expected:
            state, state_color = "训练完成", "#34d399"
        elif age is not None and age > args.stale_seconds:
            state, state_color = f"日志暂未更新（{age:.0f} 秒）", "#fb7185"
        else:
            state, state_color = "训练中", "#34d399"

        title.set_text(f"Wuji2 DexGraspNet2 — 第 {iteration:,} / {expected:,} 步")
        status_text.set_text(f"状态：{state}")
        status_text.set_color(state_color)
        detail_text.set_text(
            f"总损失 {finite_float(last, 'loss'):.5f}    "
            f"物体准确率 {finite_float(last, 'acc_objectness'):.3f}    "
            f"速度 {rate:.2f} iter/s" if rate is not None else
            f"总损失 {finite_float(last, 'loss'):.5f}    速度 --"
        )
        if rate is not None:
            detail_text.set_text(
                detail_text.get_text()
                + f"    已用 {format_duration(elapsed)}    预计剩余 {format_duration(eta)}"
            )
        progress_fill.set_width(0.41 * fraction)
        progress_text.set_text(f"{100.0 * fraction:.2f}%")

        visible = records[-args.plot_window :] if args.plot_window > 0 else records
        steps = np.asarray([int(record["iteration"]) for record in visible], dtype=float)
        values = {
            key: np.asarray([finite_float(record, key) for record in visible], dtype=float)
            for key in METRIC_KEYS
        }
        total_mean = moving_average(values["loss"], args.smooth_window)
        diffusion_mean = moving_average(values["loss_diffusion"], args.smooth_window)
        total_raw.set_data(steps, values["loss"])
        total_smooth.set_data(steps, total_mean)
        diffusion_raw.set_data(steps, values["loss_diffusion"])
        diffusion_smooth.set_data(steps, diffusion_mean)
        for key, line in component_lines.items():
            line.set_data(steps, moving_average(values[key], args.smooth_window))
        accuracy_line.set_data(
            steps, moving_average(values["acc_objectness"], args.smooth_window)
        )
        grasp_abs_line.set_data(
            steps, moving_average(values["abs_graspness"], args.smooth_window)
        )

        left = max(0.0, float(steps[0]) - 1.0)
        right = max(left + 10.0, float(steps[-1]) + max(5.0, 0.02 * len(steps)))
        for axis in (ax_total, ax_components, ax_diffusion, ax_quality):
            axis.set_xlim(left, right)
        ax_total.set_ylim(*robust_log_limits([values["loss"], total_mean]))
        ax_components.set_ylim(
            *robust_log_limits([values[key] for key in component_lines])
        )
        ax_diffusion.set_ylim(
            *robust_log_limits([values["loss_diffusion"], diffusion_mean])
        )
        quality = np.concatenate((values["acc_objectness"], values["abs_graspness"]))
        quality = quality[np.isfinite(quality)]
        if quality.size:
            qlow, qhigh = np.percentile(quality, (1, 99))
            margin = max(0.05, 0.10 * float(qhigh - qlow))
            ax_quality.set_ylim(max(0.0, float(qlow) - margin), float(qhigh) + margin)
        return ()

    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.075, top=0.98)

    if args.once:
        update(0)
        output = (args.save or log_path.with_name("training_loss.png")).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())
        print(output)
        plt.close(figure)
        return

    animation = FuncAnimation(
        figure, update, interval=max(250, args.refresh_ms), cache_frame_data=False
    )
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
