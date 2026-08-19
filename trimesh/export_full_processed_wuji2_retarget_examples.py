#!/usr/bin/env python3
"""Export fully processed LEAP -> Wuji2 retarget examples.

This viewer consumes existing ``final_waypoints.npz`` files after:

1. LEAP -> Wuji2 hand retargeting
2. wrist/root coordinate migration
3. fingertip alignment validation/finalization

It does not run or modify any retargeting code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from export_leap_to_wuji2_retarget_examples import (  # noqa: E402
    AXIS_LENGTH_M,
    DEFAULT_CASES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WUJI2_URDF,
    UrdfVisualModel,
    add_floor_tile,
    add_root_marker,
    load_case,
)


WUJI2_COLOR = np.asarray([45, 170, 90, 230], dtype=np.uint8)
TARGET_TIP_COLOR = np.asarray([245, 70, 70, 255], dtype=np.uint8)
ACTUAL_TIP_COLOR = np.asarray([50, 190, 245, 255], dtype=np.uint8)
LINK_COLOR = np.asarray([250, 220, 80, 255], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        dest="cases",
        type=Path,
        action="append",
        default=None,
        help="final_waypoints.npz to visualize; may be passed multiple times.",
    )
    parser.add_argument("--wuji2-urdf", type=Path, default=DEFAULT_WUJI2_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--export-name",
        default="leap_to_wuji2_full_processed_examples.glb",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["pregrasp", "cover", "grasp", "squeeze", "lift"],
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def make_sphere(center: np.ndarray, radius: float, color: np.ndarray) -> trimesh.Trimesh:
    sphere = trimesh.creation.uv_sphere(radius=radius, count=[16, 8])
    sphere.apply_translation(center)
    sphere.visual.face_colors = color
    return sphere


def transform_points_between_roots(
    points_world: np.ndarray,
    original_root: np.ndarray,
    display_root: np.ndarray,
) -> np.ndarray:
    original_from_world = np.linalg.inv(original_root)
    points_h = np.concatenate(
        [np.asarray(points_world, dtype=np.float64), np.ones((len(points_world), 1))],
        axis=1,
    )
    points_local = (original_from_world @ points_h.T).T
    return (display_root @ points_local.T).T[:, :3]


def add_tip_alignment_markers(
    scene: trimesh.Scene,
    payload,
    display_grasp_root: np.ndarray,
    prefix: str,
) -> dict[str, object]:
    target_tips = np.asarray(payload.extra["four_real_tip_target_world_m"], dtype=np.float64)
    actual_tips = np.asarray(payload.extra["four_real_tip_actual_world_m"], dtype=np.float64)
    grasp_index = payload.waypoint_names.index("grasp")
    original_grasp_root = np.asarray(payload.wuji2_pose[grasp_index], dtype=np.float64)
    target_display = transform_points_between_roots(
        target_tips, original_grasp_root, display_grasp_root
    )
    actual_display = transform_points_between_roots(
        actual_tips, original_grasp_root, display_grasp_root
    )
    errors_mm = np.linalg.norm(actual_tips - target_tips, axis=1) * 1000.0
    for tip_index, (target, actual) in enumerate(zip(target_display, actual_display)):
        scene.add_geometry(
            make_sphere(target, 0.010, TARGET_TIP_COLOR),
            geom_name=f"{prefix}_target_tip_{tip_index}",
        )
        scene.add_geometry(
            make_sphere(actual, 0.007, ACTUAL_TIP_COLOR),
            geom_name=f"{prefix}_actual_tip_{tip_index}",
        )
        line = trimesh.load_path(np.asarray([[target, actual]], dtype=np.float64))
        scene.add_geometry(line, geom_name=f"{prefix}_tip_error_link_{tip_index}")
    return {
        "target_tip_color": "red",
        "actual_tip_color": "cyan",
        "tip_error_mm": errors_mm.round(4).tolist(),
        "max_tip_error_mm": float(np.max(errors_mm)),
        "mean_tip_error_mm": float(np.mean(errors_mm)),
    }


def load_case_with_extra(path: Path):
    payload = load_case(path)
    with np.load(path, allow_pickle=True) as data:
        payload.extra = {
            "four_real_tip_target_world_m": np.asarray(data["four_real_tip_target_world_m"]),
            "four_real_tip_actual_world_m": np.asarray(data["four_real_tip_actual_world_m"]),
        }
    return payload


def main() -> None:
    args = parse_args()
    case_paths = args.cases if args.cases is not None else DEFAULT_CASES
    case_paths = [path.resolve() for path in case_paths if path.exists()]
    if not case_paths:
        raise RuntimeError("No final_waypoints.npz cases found.")

    wuji2_model = UrdfVisualModel(args.wuji2_urdf)
    scene = trimesh.Scene()
    manifest: dict[str, object] = {
        "output_contract": (
            "Fully processed Wuji2 hand after wrist/root migration and fingertip alignment. "
            "Red spheres are LEAP target fingertips; cyan spheres are final Wuji2 fingertips."
        ),
        "wuji2_urdf": str(args.wuji2_urdf.resolve()),
        "cases": [],
    }

    x_spacing = 0.50
    y_spacing = 0.38
    for case_index, case_path in enumerate(case_paths):
        payload = load_case_with_extra(case_path)
        wanted = [stage for stage in args.stages if stage in payload.waypoint_names]
        display_grasp_root = None
        case_record = {
            "case_path": str(payload.path),
            "source_candidate_index": payload.source_candidate_index,
            "score": payload.score,
            "stages": [],
        }
        for stage_index, stage_name in enumerate(wanted):
            waypoint_index = payload.waypoint_names.index(stage_name)
            cell_origin = np.asarray([stage_index * x_spacing, -case_index * y_spacing, 0.0])
            add_floor_tile(scene, cell_origin, f"full_tile_case{case_index:02d}_{stage_name}")
            root = np.array(payload.wuji2_pose[waypoint_index], dtype=np.float64)
            root[:3, 3] = cell_origin
            if stage_name == "grasp":
                display_grasp_root = root.copy()
            qpos = dict(zip(payload.wuji2_joint_names, payload.wuji2_q[waypoint_index].tolist()))
            for name, mesh in wuji2_model.scene_geometry(
                root, qpos, WUJI2_COLOR, f"full_case{case_index:02d}_{stage_name}_wuji2"
            ):
                scene.add_geometry(mesh, geom_name=name)
            add_root_marker(scene, root, f"full_case{case_index:02d}_{stage_name}_root")
            case_record["stages"].append(
                {
                    "name": stage_name,
                    "waypoint_index": waypoint_index,
                    "display_xyz": root[:3, 3].round(6).tolist(),
                }
            )
        if display_grasp_root is not None:
            case_record["fingertip_alignment"] = add_tip_alignment_markers(
                scene, payload, display_grasp_root, f"full_case{case_index:02d}_grasp"
            )
        manifest["cases"].append(case_record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_path = args.output_dir / args.export_name
    manifest_path = args.output_dir / "leap_to_wuji2_full_processed_manifest.json"
    scene.export(export_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EXPORT] {export_path}")
    print(f"[MANIFEST] {manifest_path}")
    print(f"[CASES] {len(case_paths)}")
    if args.show:
        scene.show()


if __name__ == "__main__":
    main()
