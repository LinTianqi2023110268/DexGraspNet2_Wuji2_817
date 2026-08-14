#!/usr/bin/env python3
"""Read-only status report for the currently generated Wuji2 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.project import source_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=source_path("active_training_scene_dataset"),
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    progress = read_json(root / "run_progress.json")
    manifest = read_json(root / "run_manifest.json")
    target_scenes = int(progress.get("scene_count", 0))
    views_per_scene = int(progress.get("view_count", 0))
    scene_root = root / "scenes"
    scene_dirs = sorted(
        path
        for path in scene_root.glob("scene_[0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    complete_scenes = sum(
        (path / "scene_manifest.json").is_file() for path in scene_dirs
    )
    view_pattern = (
        "scene_[0-9][0-9][0-9][0-9]/camera/"
        "view_[0-9][0-9][0-9][0-9]/sample_pixel_indices.npy"
    )
    complete_views = sum(1 for _ in scene_root.glob(view_pattern))
    current_scene = scene_dirs[-1] if scene_dirs else None
    current_views = 0
    if current_scene is not None:
        current_views = sum(
            1
            for _ in (current_scene / "camera").glob(
                "view_[0-9][0-9][0-9][0-9]/sample_pixel_indices.npy"
            )
        )
    target_views = target_scenes * views_per_scene
    label_root = root / "grasp_label_stages"
    stage03_root = label_root / "03_reference_points_and_surface_graspness"
    stage04_root = label_root / "04_single_view_training_labels"
    complete_stage03_scenes = sum(
        1 for _ in stage03_root.glob("scene_[0-9][0-9][0-9][0-9]/stage_manifest.json")
    )
    stage04_manifests = sorted(
        stage04_root.glob("scene_[0-9][0-9][0-9][0-9]/stage_manifest.json")
    )
    complete_stage04_scenes = len(stage04_manifests)
    stage04_view_files = sum(
        1
        for _ in stage04_root.glob(
            "scene_[0-9][0-9][0-9][0-9]/view_[0-9][0-9][0-9][0-9].npz"
        )
    )
    empty_label_views = 0
    for path in stage04_manifests:
        stage04 = read_json(path)
        empty_label_views += sum(
            int(record.get("total_available_grasp_count", 0)) <= 0
            for record in stage04.get("view_records", [])
        )
    usable_training_views = stage04_view_files - empty_label_views
    disk = shutil.disk_usage(root)
    status = str(manifest.get("status") or progress.get("status") or "unknown")
    report = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "output_root": str(root),
        "target_scenes": target_scenes,
        "started_scenes": len(scene_dirs),
        "complete_scenes": complete_scenes,
        "views_per_scene": views_per_scene,
        "target_views": target_views,
        "complete_views": complete_views,
        "complete_stage03_scenes": complete_stage03_scenes,
        "complete_stage04_scenes": complete_stage04_scenes,
        "stage04_view_files": stage04_view_files,
        "empty_label_views": empty_label_views,
        "usable_training_views": usable_training_views,
        "completion_percent_by_view": (
            100.0 * complete_views / target_views if target_views else 0.0
        ),
        "current_scene": current_scene.name if current_scene else None,
        "current_scene_complete_views": current_views,
        "rejected_candidate_count": len(progress.get("rejected_candidates", [])),
        "free_disk_gib": disk.free / (1024 ** 3),
        "safe_to_train": bool(
            status == "complete"
            and target_scenes > 0
            and complete_scenes == target_scenes
            and complete_views == target_views
            and complete_stage03_scenes == target_scenes
            and complete_stage04_scenes == target_scenes
            and stage04_view_files == target_views
            and usable_training_views > 0
        ),
    }
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(f"状态: {report['status']}")
    print(
        f"场景: {report['complete_scenes']}/{report['target_scenes']} 完整，"
        f"{report['started_scenes']} 个已开始"
    )
    print(
        f"视角: {report['complete_views']}/{report['target_views']} "
        f"({report['completion_percent_by_view']:.2f}%)"
    )
    print(
        f"标签: Stage03 {report['complete_stage03_scenes']}/{report['target_scenes']}，"
        f"Stage04 {report['complete_stage04_scenes']}/{report['target_scenes']}，"
        f"可训练视角 {report['usable_training_views']}/{report['stage04_view_files']}"
    )
    print(
        f"当前: {report['current_scene']} "
        f"{report['current_scene_complete_views']}/{report['views_per_scene']}"
    )
    print(f"拒绝候选: {report['rejected_candidate_count']}")
    print(f"磁盘剩余: {report['free_disk_gib']:.1f} GiB")
    print(f"允许训练: {'是' if report['safe_to_train'] else '否'}")
    print(f"只读数据目录: {report['output_root']}")


if __name__ == "__main__":
    main()
