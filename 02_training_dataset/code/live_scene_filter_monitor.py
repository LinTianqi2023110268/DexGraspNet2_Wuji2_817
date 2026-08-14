#!/usr/bin/env python3
"""Read-only Tk visualization for the Wuji2 scene producer and grasp filter.

This program is read-only. Closing the window never stops either background job.
It intentionally uses Tk instead of a GPU plotting backend, so it can remain open
while Isaac Sim owns the RTX GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "02_training_dataset/config/wuji2_train60_100seminal_256view_v1.json"
)
STAGE_DIRECTORY = "grasp_label_stages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--refresh-seconds", type=float, default=2.0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def tail_text(path: Path, byte_limit: int = 48_000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - byte_limit), os.SEEK_SET)
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def process_running(fragment: bytes) -> bool:
    proc = Path("/proc")
    try:
        entries = proc.iterdir()
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if fragment in command:
            return True
    return False


def manifest_totals(paths: list[Path]) -> tuple[int, int, int]:
    preserved = 0
    paper = 0
    safe = 0
    for path in paths:
        data = read_json(path)
        for record in data.get("object_records", []):
            preserved += int(record.get("preserved_count", record.get("input_count", 0)))
            paper += int(record.get("paper_keep_count", 0))
            safe += int(record.get("wuji2_safe_keep_count", record.get("kept_count", 0)))
    return preserved, paper, safe


class Dashboard:
    COLORS = {
        "background": "#151a21",
        "panel": "#202832",
        "text": "#e8eef5",
        "muted": "#9cabb9",
        "green": "#43c581",
        "blue": "#4ca3ff",
        "orange": "#f6ad55",
        "red": "#ff6b6b",
    }

    def __init__(self, root: tk.Tk, config_path: Path, refresh_seconds: float):
        self.root = root
        self.config_path = config_path.expanduser().resolve()
        self.refresh_ms = max(500, int(refresh_seconds * 1000))
        self.config = read_json(self.config_path)
        configured_output = Path(self.config["paths"]["output_root"])
        self.output_root = (
            configured_output.resolve()
            if configured_output.is_absolute()
            else (PROJECT_ROOT / configured_output).resolve()
        )
        self.target_scenes = int(self.config.get("scope", {}).get("scene_count", 100))
        self.label_root = self.output_root / STAGE_DIRECTORY
        self.log_path = self.label_root / "stream_filter.log"
        self._last_completed_count = -1
        self._last_totals = (0, 0, 0)
        self._closed = False

        root.title("Wuji2 DexGraspNet2 — 场景生成与位姿筛选实时监控")
        root.geometry("1120x760")
        root.minsize(900, 620)
        root.configure(bg=self.COLORS["background"])
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "Generated.Horizontal.TProgressbar",
            troughcolor="#303946",
            background=self.COLORS["blue"],
            bordercolor="#303946",
        )
        style.configure(
            "Filtered.Horizontal.TProgressbar",
            troughcolor="#303946",
            background=self.COLORS["green"],
            bordercolor="#303946",
        )
        self._build()
        self.refresh()

    def _label(self, parent: tk.Widget, text: str, size: int = 12, **kwargs) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            bg=kwargs.pop("bg", self.COLORS["panel"]),
            fg=kwargs.pop("fg", self.COLORS["text"]),
            font=("DejaVu Sans", size, kwargs.pop("weight", "normal")),
            **kwargs,
        )

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=self.COLORS["background"])
        header.pack(fill="x", padx=24, pady=(18, 10))
        self._label(
            header,
            "Wuji2 DexGraspNet2 实时数据流水线",
            size=20,
            weight="bold",
            bg=self.COLORS["background"],
        ).pack(side="left")
        self.clock = self._label(
            header, "", size=11, bg=self.COLORS["background"], fg=self.COLORS["muted"]
        )
        self.clock.pack(side="right")

        status_panel = tk.Frame(self.root, bg=self.COLORS["panel"], padx=18, pady=14)
        status_panel.pack(fill="x", padx=24, pady=6)
        self.generator_status = self._label(status_panel, "场景生成：检查中", size=13, weight="bold")
        self.generator_status.grid(row=0, column=0, sticky="w", padx=(0, 45))
        self.filter_status = self._label(status_panel, "位姿筛选：检查中", size=13, weight="bold")
        self.filter_status.grid(row=0, column=1, sticky="w")
        self.current_stage = self._label(
            status_panel, "当前任务：读取日志中", size=11, fg=self.COLORS["muted"]
        )
        self.current_stage.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        progress_panel = tk.Frame(self.root, bg=self.COLORS["panel"], padx=18, pady=14)
        progress_panel.pack(fill="x", padx=24, pady=6)
        self.generated_label = self._label(progress_panel, "完整场景 0 / 100", size=12, weight="bold")
        self.generated_label.pack(anchor="w")
        self.generated_bar = ttk.Progressbar(
            progress_panel,
            style="Generated.Horizontal.TProgressbar",
            maximum=self.target_scenes,
        )
        self.generated_bar.pack(fill="x", pady=(5, 12), ipady=5)
        self.filtered_label = self._label(progress_panel, "筛选完成 0 / 100", size=12, weight="bold")
        self.filtered_label.pack(anchor="w")
        self.filtered_bar = ttk.Progressbar(
            progress_panel,
            style="Filtered.Horizontal.TProgressbar",
            maximum=self.target_scenes,
        )
        self.filtered_bar.pack(fill="x", pady=(5, 0), ipady=5)

        cards = tk.Frame(self.root, bg=self.COLORS["background"])
        cards.pack(fill="x", padx=18, pady=6)
        cards.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="cards")
        self.card_values: dict[str, tk.Label] = {}
        card_specs = (
            ("backlog", "等待筛选场景", self.COLORS["orange"]),
            ("preserved", "累计保留原始位姿", self.COLORS["blue"]),
            ("paper", "论文筛选通过", self.COLORS["green"]),
            ("safe", "Wuji2安全筛选通过", self.COLORS["green"]),
        )
        for column, (key, title, color) in enumerate(card_specs):
            frame = tk.Frame(cards, bg=self.COLORS["panel"], padx=12, pady=12)
            frame.grid(row=0, column=column, sticky="nsew", padx=6)
            self._label(frame, title, size=10, fg=self.COLORS["muted"]).pack()
            value = self._label(frame, "0", size=19, weight="bold", fg=color)
            value.pack(pady=(5, 0))
            self.card_values[key] = value

        log_panel = tk.Frame(self.root, bg=self.COLORS["panel"], padx=14, pady=12)
        log_panel.pack(fill="both", expand=True, padx=24, pady=(6, 18))
        self._label(log_panel, "最近筛选日志", size=12, weight="bold").pack(anchor="w", pady=(0, 7))
        text_frame = tk.Frame(log_panel, bg=self.COLORS["panel"])
        text_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            text_frame,
            bg="#11161c",
            fg="#d7e1ea",
            insertbackground="#d7e1ea",
            relief="flat",
            font=("DejaVu Sans Mono", 10),
            wrap="none",
            padx=10,
            pady=8,
        )
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=yscroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        self.log_text.configure(state="disabled")

    def _latest_stage(self, log_lines: list[str]) -> str:
        stage_pattern = re.compile(r"scene=(\d{4}) stage=(\w+) command=")
        complete_pattern = re.compile(r"\[COMPLETE\] scene=(\d{4})")
        for line in reversed(log_lines):
            match = stage_pattern.search(line)
            if match:
                names = {"transform": "位姿转换", "collision": "论文碰撞筛选", "path": "虎口路径筛选"}
                return f"scene_{match.group(1)} · {names.get(match.group(2), match.group(2))}"
            match = complete_pattern.search(line)
            if match:
                return f"scene_{match.group(1)} 已完成，等待下一阶段"
        return "等待已完成场景"

    def refresh(self) -> None:
        if self._closed:
            return
        generated = sorted((self.output_root / "scenes").glob("scene_*/scene_manifest.json"))
        path_manifests = sorted(
            (self.label_root / "02b_enhanced_palm_center_path_filtered").glob(
                "scene_*/stage_manifest.json"
            )
        )
        generated_count = len(generated)
        filtered_count = len(path_manifests)
        if filtered_count != self._last_completed_count:
            self._last_totals = manifest_totals(path_manifests)
            self._last_completed_count = filtered_count
        preserved, paper, safe = self._last_totals

        generator_running = process_running(b"generate_scenes_and_views.py")
        filter_running = process_running(b"stream_filter_completed_scenes.py")
        self.generator_status.configure(
            text=f"场景生成：{'运行中' if generator_running else '未运行'}",
            fg=self.COLORS["green"] if generator_running else self.COLORS["red"],
        )
        self.filter_status.configure(
            text=f"位姿筛选：{'运行中（CPU）' if filter_running else '未运行'}",
            fg=self.COLORS["green"] if filter_running else self.COLORS["red"],
        )
        self.generated_label.configure(
            text=f"完整场景  {generated_count} / {self.target_scenes}"
        )
        self.filtered_label.configure(
            text=f"完成全部筛选  {filtered_count} / {self.target_scenes}"
        )
        self.generated_bar["value"] = generated_count
        self.filtered_bar["value"] = filtered_count
        self.card_values["backlog"].configure(text=f"{max(0, generated_count-filtered_count):,}")
        self.card_values["preserved"].configure(text=f"{preserved:,}")
        self.card_values["paper"].configure(text=f"{paper:,}")
        self.card_values["safe"].configure(text=f"{safe:,}")

        log = tail_text(self.log_path)
        lines = [line for line in log.splitlines() if line.strip()]
        self.current_stage.configure(text=f"当前任务：{self._latest_stage(lines)}")
        visible_log = "\n".join(lines[-20:]) if lines else "尚未产生筛选日志。"
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", visible_log)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.clock.configure(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(self.refresh_ms, self.refresh)

    def close(self) -> None:
        self._closed = True
        self.root.destroy()


def main() -> None:
    args = parse_args()
    if args.refresh_seconds <= 0:
        raise ValueError("--refresh-seconds must be positive")
    if not args.config.expanduser().is_file():
        raise FileNotFoundError(args.config)
    root = tk.Tk()
    Dashboard(root, args.config, args.refresh_seconds)
    root.mainloop()


if __name__ == "__main__":
    main()
