#!/usr/bin/env python3
"""Compare optimizer-target and force-adjusted-target Wuji2 training labels.

Outputs two GLB files:
1. an in-place hand overlay (blue=q_opt, orange=q_force_adjusted);
2. a side-by-side single-view graspness comparison using one shared color scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.project import project_path  # noqa: E402
from wuji2_dgn2.visual import Wuji2VisualModel  # noqa: E402


OLD_ROOT = PROJECT_ROOT / (
    "02_training_dataset/data/scene_datasets/"
    "wuji2_train60_100seminal_256view_v1"
)
NEW_ROOT = PROJECT_ROOT / (
    "02_training_dataset/data/scene_datasets/"
    "wuji2_train60_100seminal_256view_force_adjusted_legacy_v1"
)
LEGACY_URDF = PROJECT_ROOT / (
    "02_training_dataset/assets/wuji2_factory/02_wuji2_hand/"
    "original_wuji2_right/body/urdf/right.urdf"
)
OUTPUT_DIR = PROJECT_ROOT / "trimesh/outputs/force_adjusted_training_ab"
STAGE01 = "01_transformed_object_grasps"
STAGE03 = "03_reference_points_and_surface_graspness"
STAGE04 = "04_single_view_training_labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, default=17)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--object-id", type=int, default=2)
    parser.add_argument("--candidate-id", type=int, default=983)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def stage_object_path(root: Path, stage: str, scene: int, object_id: int) -> Path:
    matches = sorted(
        (root / "grasp_label_stages" / stage / f"scene_{scene:04d}").glob(
            f"object_{object_id:03d}_*.npz"
        )
    )
    matches = [path for path in matches if "surface_graspness" not in path.name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one object archive, got {matches}")
    return matches[0]


def set_color(mesh: trimesh.Trimesh, color: list[int] | np.ndarray) -> None:
    mesh.visual.face_colors = np.asarray(color, dtype=np.uint8)


def centered_object_mesh(scene_object: dict) -> trimesh.Trimesh:
    asset = scene_object["asset"]
    loaded = trimesh.load(project_path(asset["source_obj"]), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = loaded.copy()
    mesh.vertices = (
        np.asarray(mesh.vertices, dtype=np.float64)
        - np.asarray(asset["native_aabb_center"], dtype=np.float64)
    ) * float(asset["scale"])
    return mesh


def add_table_and_objects(
    display: trimesh.Scene, manifest: dict, offset: np.ndarray, suffix: str
) -> None:
    size = np.asarray(manifest["table"]["size_m"], dtype=np.float64)
    table = trimesh.creation.box(extents=size)
    table.apply_translation(
        offset
        + np.asarray(
            [0.0, 0.0, float(manifest["table"]["top_z_m"]) - size[2] / 2.0]
        )
    )
    set_color(table, [145, 145, 145, 35])
    display.add_geometry(table, geom_name=f"table_{suffix}")
    for record in manifest["objects"]:
        mesh = centered_object_mesh(record)
        transform = np.asarray(record["T_world_centered_object"], dtype=np.float64).copy()
        transform[:3, 3] += offset
        mesh.apply_transform(transform)
        set_color(mesh, [175, 175, 175, 28])
        display.add_geometry(
            mesh,
            geom_name=f"object_{int(record['segmentation_id']):03d}_{suffix}",
        )


def add_sphere(
    display: trimesh.Scene, center: np.ndarray, color: list[int], name: str
) -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.0035)
    sphere.apply_translation(center)
    set_color(sphere, color)
    display.add_geometry(sphere, geom_name=name)


def select_common_candidate(
    old: dict[str, np.ndarray], new: dict[str, np.ndarray], requested: int
) -> tuple[int, int, int]:
    old_ids = np.asarray(old["candidate_id"], dtype=np.int64)
    new_ids = np.asarray(new["candidate_id"], dtype=np.int64)
    common = np.intersect1d(old_ids, new_ids)
    if not len(common):
        raise RuntimeError("No grasp candidate survives both label variants")
    candidate = requested if requested in common else None
    if candidate is None:
        old_map = {int(value): index for index, value in enumerate(old_ids)}
        new_map = {int(value): index for index, value in enumerate(new_ids)}
        candidate = max(
            common.tolist(),
            key=lambda value: float(
                np.linalg.norm(
                    new["qpos"][new_map[int(value)]]
                    - old["qpos"][old_map[int(value)]]
                )
            ),
        )
    old_index = int(np.flatnonzero(old_ids == candidate)[0])
    new_index = int(np.flatnonzero(new_ids == candidate)[0])
    return int(candidate), old_index, new_index


def build_posture_overlay(args: argparse.Namespace, manifest: dict) -> tuple[trimesh.Scene, dict]:
    old = load_npz(stage_object_path(OLD_ROOT, STAGE03, args.scene, args.object_id))
    new = load_npz(stage_object_path(NEW_ROOT, STAGE03, args.scene, args.object_id))
    candidate, old_index, new_index = select_common_candidate(
        old, new, args.candidate_id
    )
    joint_manifest = json.loads(
        (
            OLD_ROOT
            / "grasp_label_stages"
            / STAGE01
            / f"scene_{args.scene:04d}"
            / "stage_manifest.json"
        ).read_text(encoding="utf-8")
    )
    joint_order = joint_manifest["label_contract"]["joint_order"]
    old_q = np.asarray(old["qpos"][old_index], dtype=np.float64)
    new_q = np.asarray(new["qpos"][new_index], dtype=np.float64)
    old_root = np.asarray(old["T_world_r_base_link"][old_index], dtype=np.float64)
    new_root = np.asarray(new["T_world_r_base_link"][new_index], dtype=np.float64)
    if not np.allclose(old_root, new_root, atol=1.0e-7, rtol=0.0):
        raise RuntimeError("Strict A/B contract broken: hand root pose changed")

    display = trimesh.Scene()
    add_table_and_objects(display, manifest, np.zeros(3), "shared")
    model = Wuji2VisualModel(LEGACY_URDF)
    for prefix, q_values, color in (
        ("OLD_q_opt_BLUE", old_q, np.asarray([35, 115, 245, 95], np.uint8)),
        ("NEW_q_force_ORANGE", new_q, np.asarray([255, 130, 20, 150], np.uint8)),
    ):
        qpos = dict(zip(joint_order, q_values.tolist()))
        for name, mesh in model.scene_geometry(old_root, qpos, color, prefix):
            mesh.visual.face_colors[:, 3] = color[3]
            display.add_geometry(mesh, geom_name=name)
    old_point = np.asarray(old["point"][old_index], dtype=np.float64)
    new_point = np.asarray(new["point"][new_index], dtype=np.float64)
    add_sphere(display, old_point, [35, 115, 245, 255], "OLD_reference_BLUE")
    add_sphere(display, new_point, [255, 220, 20, 255], "NEW_reference_YELLOW")
    line = trimesh.load_path(np.stack([old_point, new_point]))
    display.add_geometry(line, geom_name="reference_point_shift")
    display.add_geometry(
        trimesh.creation.axis(origin_size=0.004, axis_length=0.08),
        geom_name="world_frame",
    )
    stats = {
        "candidate_id": candidate,
        "old_index": old_index,
        "new_index": new_index,
        "joint_delta_l2_rad": float(np.linalg.norm(new_q - old_q)),
        "joint_delta_max_abs_rad": float(np.abs(new_q - old_q).max()),
        "reference_point_shift_mm": float(np.linalg.norm(new_point - old_point) * 1000.0),
    }
    return display, stats


def build_graspness_side_by_side(
    args: argparse.Namespace, manifest: dict
) -> tuple[trimesh.Scene, dict]:
    old_path = OLD_ROOT / "grasp_label_stages" / STAGE04 / f"scene_{args.scene:04d}" / f"view_{args.view:04d}.npz"
    new_path = NEW_ROOT / "grasp_label_stages" / STAGE04 / f"scene_{args.scene:04d}" / f"view_{args.view:04d}.npz"
    old = load_npz(old_path)
    new = load_npz(new_path)
    if not np.array_equal(old["point_clouds"], new["point_clouds"]):
        raise RuntimeError("Strict A/B contract broken: point cloud changed")
    if not np.array_equal(old["seg"], new["seg"]):
        raise RuntimeError("Strict A/B contract broken: segmentation changed")
    points_camera = np.asarray(old["point_clouds"], dtype=np.float64)
    world_from_camera = np.asarray(old["T_world_camera"], dtype=np.float64)
    points_world = trimesh.transform_points(points_camera, world_from_camera)
    foreground = np.asarray(old["objectness"], dtype=bool)
    old_score = np.asarray(old["graspness_log_target"], dtype=np.float64)
    new_score = np.asarray(new["graspness_log_target"], dtype=np.float64)
    joined = np.concatenate([old_score[foreground], new_score[foreground]])
    low, high = np.percentile(joined[np.isfinite(joined)], [2.0, 98.0])
    if high <= low:
        high = low + 1.0

    display = trimesh.Scene()
    offsets = {
        "OLD_q_opt_LEFT": np.asarray([-0.32, 0.0, 0.0]),
        "NEW_q_force_RIGHT": np.asarray([0.32, 0.0, 0.0]),
    }
    for label, score in (
        ("OLD_q_opt_LEFT", old_score),
        ("NEW_q_force_RIGHT", new_score),
    ):
        offset = offsets[label]
        add_table_and_objects(display, manifest, offset, label)
        normalized = np.clip((score - low) / (high - low), 0.0, 1.0)
        colors = np.full((len(score), 4), [115, 115, 115, 45], dtype=np.uint8)
        colors[foreground] = np.rint(
            matplotlib.colormaps["viridis"](normalized[foreground]) * 255.0
        ).astype(np.uint8)
        colors[foreground, 3] = 235
        display.add_geometry(
            trimesh.points.PointCloud(points_world + offset, colors=colors),
            geom_name=f"{label}_graspness",
        )
        axis = trimesh.creation.axis(origin_size=0.004, axis_length=0.07)
        axis.apply_translation(offset)
        display.add_geometry(axis, geom_name=f"{label}_world_frame")
    stats = {
        "shared_color_scale_percentile_2": float(low),
        "shared_color_scale_percentile_98": float(high),
        "old_max_log_graspness": float(old_score.max()),
        "new_max_log_graspness": float(new_score.max()),
        "old_nonzero_raw_points": int((old["graspness_raw"] > 0).sum()),
        "new_nonzero_raw_points": int((new["graspness_raw"] > 0).sum()),
    }
    return display, stats


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (OLD_ROOT / "scenes" / f"scene_{args.scene:04d}" / "scene_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    posture, posture_stats = build_posture_overlay(args, manifest)
    heatmap, heatmap_stats = build_graspness_side_by_side(args, manifest)
    posture_path = OUTPUT_DIR / f"scene{args.scene:04d}_object{args.object_id:03d}_posture_ab.glb"
    heatmap_path = OUTPUT_DIR / f"scene{args.scene:04d}_view{args.view:04d}_graspness_ab.glb"
    posture.export(posture_path)
    heatmap.export(heatmap_path)
    report = {
        "scene": args.scene,
        "view": args.view,
        "object_id": args.object_id,
        "color_legend": {
            "posture_blue": "old q_opt training target",
            "posture_orange": "new q_force_adjusted training target",
            "old_reference_blue": "old complete-surface reference point",
            "new_reference_yellow": "new complete-surface reference point",
            "heatmap_left": "old q_opt-derived graspness",
            "heatmap_right": "new q_force_adjusted-derived graspness",
            "heatmap": "viridis purple=low, yellow=high; shared percentile scale",
        },
        "posture": posture_stats,
        "graspness": heatmap_stats,
        "outputs": [str(posture_path), str(heatmap_path)],
    }
    report_path = OUTPUT_DIR / f"scene{args.scene:04d}_ab_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.show:
        posture.show(caption="Blue q_opt versus orange q_force_adjusted")
        heatmap.show(caption="Left old graspness versus right force-adjusted graspness")


if __name__ == "__main__":
    main()
