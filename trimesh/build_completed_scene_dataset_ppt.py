#!/usr/bin/env python3
"""Generate a ppt5-style summary of the completed Wuji2 scene dataset."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from statistics import mean, median
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / (
    "02_training_dataset/data/scene_datasets/"
    "wuji2_train60_100seminal_256view_v1"
)
DEFAULT_POOL = PROJECT_ROOT / (
    "02_training_dataset/config/wuji2_train60_object_pool_v1.json"
)
DEFAULT_TEMPLATE = SCRIPT_DIR / "assets/ppt5.odp"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs/ppt6.odp"

PICTURES = {
    "top": "Pictures/100000010000051C000003CDE9174E56A4365A22.png",
    "middle": "Pictures/100000010000026D000001BA601B2BAEEA010098.png",
    "right_bottom": "Pictures/10000001000003430000027AFCE66504473A010D.png",
    "chart_left": "Pictures/100000010000040500000249699AE2BF989FA2FE.png",
    "chart_right": "Pictures/10000001000004640000036B19ED679B729B69ED.png",
}
NS = {
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}


def qname(prefix: str, local: str) -> str:
    return f"{{{NS[prefix]}}}{local}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--object-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=SCRIPT_DIR / "outputs/completed_scene_dataset_ppt",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_statistics(data_root: Path, pool_path: Path) -> dict:
    pool = read_json(pool_path)["objects"]
    scene_files = sorted((data_root / "scenes").glob("scene_*/scene_manifest.json"))
    occurrences: Counter[int] = Counter()
    candidate_indices = []
    upright_scenes = 0
    forced_upright = 0
    classified_upright = 0
    for path in scene_files:
        manifest = read_json(path)
        occurrences.update(int(obj["object_pool_index"]) for obj in manifest["objects"])
        physics = manifest["physics_acceptance"]
        candidate_indices.append(int(physics["candidate_index"]))
        upright = physics["candidate_layout_sampling"].get("upright_pose_bias", {})
        upright_scenes += int(bool(upright.get("active", False)))
        forced_upright += len(upright.get("forced_upright_object_indices", []))
        classified_upright += int(upright.get("classified_upright_object_count", 0))

    raw_grasps = paper_grasps = safe_grasps = 0
    stage02b = data_root / "grasp_label_stages/02b_enhanced_palm_center_path_filtered"
    for path in sorted(stage02b.glob("scene_*/stage_manifest.json")):
        manifest = read_json(path)
        raw_grasps += int(manifest["total_input"])
        safe_grasps += int(manifest["total_kept"])
        paper_grasps += sum(
            int(record["paper_keep_count"]) for record in manifest["object_records"]
        )

    stage03_files = sorted(
        (data_root / "grasp_label_stages/03_reference_points_and_surface_graspness")
        .glob("scene_*/stage_manifest.json")
    )
    stage03_grasps = sum(read_json(path)["total_grasps"] for path in stage03_files)

    available = []
    visible = []
    zero_views = []
    stage04_files = sorted(
        (data_root / "grasp_label_stages/04_single_view_training_labels")
        .glob("scene_*/stage_manifest.json")
    )
    for path in stage04_files:
        manifest = read_json(path)
        for record in manifest["view_records"]:
            count = int(record["total_available_grasp_count"])
            available.append(count)
            visible.append(int(record["visible_object_point_count"]))
            if count == 0:
                zero_views.append(
                    {
                        "scene": int(manifest["scene_index"]),
                        "view": int(record["view_index"]),
                        "visible_object_points": int(record["visible_object_point_count"]),
                    }
                )

    occurrence_values = [int(occurrences[index]) for index in range(len(pool))]
    missing_objects = [
        {
            "pool_index": index,
            "id": int(obj["id"]),
            "code": obj["code"],
            "category": obj["category"],
        }
        for index, obj in enumerate(pool)
        if occurrence_values[index] == 0
    ]
    run_manifest = read_json(data_root / "run_manifest.json")
    disk_kib = int(
        subprocess.check_output(["du", "-sk", str(data_root)], text=True).split()[0]
    )
    total_views = len(available)
    trainable_views = sum(count > 0 for count in available)
    return {
        "dataset_status": run_manifest["status"],
        "scene_count": len(scene_files),
        "object_pool_size": len(pool),
        "represented_object_count": sum(count > 0 for count in occurrence_values),
        "object_placements": sum(occurrence_values),
        "object_occurrences": occurrence_values,
        "object_occurrence_min": min(occurrence_values),
        "object_occurrence_max": max(occurrence_values),
        "object_occurrence_mean": mean(occurrence_values),
        "missing_objects": missing_objects,
        "views_per_scene": 256,
        "points_per_view": 40000,
        "total_views": total_views,
        "total_sampled_points": total_views * 40000,
        "stage03_scene_count": len(stage03_files),
        "stage04_scene_count": len(stage04_files),
        "trainable_views": trainable_views,
        "trainable_view_rate": trainable_views / total_views,
        "zero_grasp_views": zero_views,
        "available_grasps_min": min(available),
        "available_grasps_max": max(available),
        "available_grasps_mean": mean(available),
        "available_grasps_median": median(available),
        "visible_object_points_mean": mean(visible),
        "raw_scene_grasps": raw_grasps,
        "paper_keep_grasps": paper_grasps,
        "wuji2_safe_grasps": safe_grasps,
        "stage03_grasps": stage03_grasps,
        "rejected_candidate_count": len(run_manifest["rejected_candidates"]),
        "accepted_candidate_min": min(candidate_indices),
        "accepted_candidate_max": max(candidate_indices),
        "upright_biased_accepted_scenes": upright_scenes,
        "upright_forced_placements": forced_upright,
        "upright_classified_placements": classified_upright,
        "disk_gib": disk_kib / (1024.0 * 1024.0),
    }


def configure_plotting() -> None:
    cjk_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(cjk_font))
    cjk_family = font_manager.FontProperties(fname=cjk_font).get_name()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [cjk_family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def label_path(data_root: Path, scene: int, view: int) -> Path:
    return data_root / (
        "grasp_label_stages/04_single_view_training_labels/"
        f"scene_{scene:04d}/view_{view:04d}.npz"
    )


def render_point_cloud(
    data_root: Path,
    scene: int,
    view: int,
    output: Path,
    size_px: tuple[int, int],
    heatmap: bool,
    caption: str,
) -> None:
    with np.load(label_path(data_root, scene, view)) as archive:
        points_camera = np.asarray(archive["point_clouds"], dtype=np.float64)
        transform = np.asarray(archive["T_world_camera"], dtype=np.float64)
        segmentation = np.asarray(archive["seg"], dtype=np.int64)
        objectness = np.asarray(archive["objectness"], dtype=np.int64)
        graspness = np.asarray(archive["graspness_log_target"], dtype=np.float64)
    points = trimesh.transform_points(points_camera, transform)

    width, height = size_px
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    background = np.flatnonzero(objectness == 0)[::8]
    foreground = np.flatnonzero(objectness == 1)[::2]
    if background.size:
        ax.scatter(
            points[background, 0], points[background, 1], points[background, 2],
            s=0.25, c="#a7abb0", alpha=0.13, depthshade=False,
        )
    if foreground.size:
        if heatmap:
            values = graspness[foreground]
            finite = values[np.isfinite(values)]
            lo, hi = np.percentile(finite, [2.0, 98.0]) if finite.size else (0.0, 1.0)
            colors = np.clip((values - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)
            ax.scatter(
                points[foreground, 0], points[foreground, 1], points[foreground, 2],
                s=1.2, c=colors, cmap="viridis", alpha=0.95, depthshade=False,
            )
        else:
            colors = plt.get_cmap("tab20")((segmentation[foreground] * 3) % 20)
            ax.scatter(
                points[foreground, 0], points[foreground, 1], points[foreground, 2],
                s=1.2, c=colors, alpha=0.92, depthshade=False,
            )
    table = [
        (-0.25, -0.15, 0.0), (0.25, -0.15, 0.0),
        (0.25, 0.15, 0.0), (-0.25, 0.15, 0.0),
    ]
    ax.add_collection3d(
        Poly3DCollection([table], facecolor="#d4d6d8", alpha=0.22, edgecolor="#888")
    )
    ax.set(xlim=(-0.27, 0.27), ylim=(-0.17, 0.17), zlim=(-0.01, 0.23))
    ax.set_box_aspect((0.54, 0.34, 0.24))
    ax.view_init(elev=27, azim=-58)
    ax.set_axis_off()
    ax.text2D(
        0.03, 0.94, caption, transform=ax.transAxes,
        fontsize=11, weight="bold", color="#202020",
    )
    fig.savefig(output, dpi=100, facecolor="white", transparent=False)
    plt.close(fig)


def render_occurrences(stats: dict, output: Path, size_px: tuple[int, int]) -> None:
    width, height = size_px
    values = np.asarray(stats["object_occurrences"], dtype=int)
    colors = np.where(values == 0, "#e45756", "#4c78a8")
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax.bar(np.arange(len(values)) + 1, values, color=colors, width=0.82)
    ax.axhline(values.mean(), color="#f28e2b", lw=2.0, ls="--")
    ax.set(xlim=(0, len(values) + 1), ylim=(0, max(values) * 1.2))
    ax.set_title("60物体池的实际场景覆盖", fontsize=12, weight="bold", pad=5)
    ax.set_xlabel("物体池编号", fontsize=9)
    ax.set_ylabel("出现次数", fontsize=9)
    ax.text(
        0.98, 0.94,
        f"实际覆盖 {stats['represented_object_count']}/60\n"
        f"范围 {stats['object_occurrence_min']}–{stats['object_occurrence_max']}，"
        f"均值 {stats['object_occurrence_mean']:.1f}",
        ha="right", va="top", transform=ax.transAxes, fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout(pad=0.45)
    fig.savefig(output, dpi=100, facecolor="white")
    plt.close(fig)


def render_filter_funnel(stats: dict, output: Path, size_px: tuple[int, int]) -> None:
    width, height = size_px
    labels = ["场景变换后", "论文碰撞通过", "Wuji2路径安全"]
    values = [
        stats["raw_scene_grasps"],
        stats["paper_keep_grasps"],
        stats["wuji2_safe_grasps"],
    ]
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    bars = ax.barh(labels, values, color=["#9aa0a6", "#59a14f", "#2f7f5f"], height=0.54)
    ax.invert_yaxis()
    ax.set_title("训练抓取姿态筛选", fontsize=12, weight="bold", pad=5)
    ax.set_xlim(0, max(values) * 1.24)
    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}", va="center", fontsize=10, weight="bold",
        )
    ax.text(
        0.98, 0.04,
        f"最终保留率 {stats['wuji2_safe_grasps']/stats['raw_scene_grasps']:.1%}",
        transform=ax.transAxes, ha="right", fontsize=9,
    )
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.55)
    fig.savefig(output, dpi=100, facecolor="white")
    plt.close(fig)


def replace_text_box(
    box: ET.Element,
    lines: list[str],
    title: bool,
    paragraph_style: str | None = None,
) -> None:
    for child in list(box):
        box.remove(child)
    for index, line in enumerate(lines):
        paragraph = ET.SubElement(
            box, qname("text", "p"),
            {
                qname("text", "style-name"): (
                    paragraph_style or ("P1" if title else "P4")
                )
            },
        )
        span = ET.SubElement(
            paragraph, qname("text", "span"),
            {qname("text", "style-name"): "T1" if title and index == 0 else "T2"},
        )
        span.text = line


def build_odp(
    template: Path, output: Path, images: dict[str, Path], stats: dict
) -> None:
    with zipfile.ZipFile(template, "r") as source:
        content = source.read("content.xml")
        for _event, (prefix, uri) in ET.iterparse(
            io.BytesIO(content), events=("start-ns",)
        ):
            try:
                ET.register_namespace(prefix, uri)
            except ValueError:
                pass
        root = ET.fromstring(content)
        page = root.find(f".//{qname('draw', 'page')}")
        if page is None:
            raise RuntimeError("ppt5 template has no slide")
        text_boxes = page.findall(
            f"./{qname('draw', 'frame')}/{qname('draw', 'text-box')}"
        )
        if len(text_boxes) < 3:
            raise RuntimeError("ppt5 template text layout changed")
        text_frames = [
            frame
            for frame in page.findall(f"./{qname('draw', 'frame')}")
            if frame.find(f"./{qname('draw', 'text-box')}") is not None
        ]
        text_frames[2].set(qname("svg", "x"), "0.25cm")
        text_frames[2].set(qname("svg", "width"), "18.25cm")

        missing_category = stats["missing_objects"][0]["category"]
        left_lines = [
            "场景训练数据已完成",
            f"{stats['object_pool_size']}个物体池，实际覆盖{stats['represented_object_count']}种；"
            f"{stats['scene_count']}个完整场景；",
            f"每场景6个不同物体，共{stats['object_placements']}次摆放；",
            f"每场景256视角，共{stats['total_views']:,}个视角；",
            f"每视角40,000点，共{stats['total_sampled_points']/1e8:.2f}亿采样点；",
            f"Stage-03/04均完成：{stats['stage03_scene_count']}/100；",
            f"可训练视角：{stats['trainable_views']:,}/{stats['total_views']:,} "
            f"({stats['trainable_view_rate']:.3%})；约{stats['disk_gib']:.0f} GiB。",
        ]
        middle_lines = [
            "场景构建与监督标签：",
            "桌面有效区域：30 cm × 50 cm；",
            "Trimesh稳定姿态 + 二维足迹无碰撞摆放；",
            "Isaac Sim 5.0稳定性验收：100场景通过，12候选拒绝；",
            f"立放偏置进入{stats['upright_biased_accepted_scenes']}个场景，"
            f"强制{stats['upright_forced_placements']}次立放；",
            "监督位姿：q_opt / pre_force_joint_positions_rad；",
            "不是SQUEEZE后的关节位姿；",
            f"场景变换后{stats['raw_scene_grasps']:,}条 → 论文碰撞过滤"
            f"{stats['paper_keep_grasps']:,}条；",
            f"Wuji2掌心路径安全过滤后保留{stats['wuji2_safe_grasps']:,}条；",
            f"单视角平均可匹配抓取{stats['available_grasps_mean']:.1f}条。",
        ]
        bottom_line = (
            f"审计：实际覆盖59/60，缺少{missing_category}；"
            "scene_0043有2个零抓取视角。建议补齐后再冻结训练集。"
        )
        replace_text_box(text_boxes[0], left_lines, title=True)
        replace_text_box(text_boxes[1], middle_lines, title=False)
        replace_text_box(
            text_boxes[2], [bottom_line], title=False, paragraph_style="P7"
        )
        new_content = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        replacements = {
            PICTURES[key]: images[key].read_bytes() for key in PICTURES
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w") as destination:
            for info in source.infolist():
                if info.filename == "content.xml":
                    payload = new_content
                else:
                    payload = replacements.get(info.filename, source.read(info.filename))
                destination.writestr(info, payload)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    stats = collect_statistics(data_root, args.object_pool.resolve())
    (artifacts / "dataset_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    images = {
        "top": artifacts / "scene_0000_view_0000_segmentation.png",
        "middle": artifacts / "scene_0043_view_0144_graspness.png",
        "right_bottom": artifacts / "scene_0099_view_0000_segmentation.png",
        "chart_left": artifacts / "object_occurrences.png",
        "chart_right": artifacts / "grasp_filter_funnel.png",
    }
    render_point_cloud(
        data_root, 0, 0, images["top"], (1308, 973), False,
        "scene_0000 · 立放偏置场景",
    )
    render_point_cloud(
        data_root, 43, 144, images["middle"], (621, 442), True,
        "scene_0043 · graspness",
    )
    render_point_cloud(
        data_root, 99, 0, images["right_bottom"], (835, 634), False,
        "scene_0099 · 普通稳定姿态",
    )
    render_occurrences(stats, images["chart_left"], (1029, 585))
    render_filter_funnel(stats, images["chart_right"], (1124, 875))
    build_odp(args.template.resolve(), args.output.resolve(), images, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"ODP: {args.output.resolve()}")
    print(f"artifacts: {artifacts}")


if __name__ == "__main__":
    main()
