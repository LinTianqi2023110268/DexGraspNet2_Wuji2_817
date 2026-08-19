#!/usr/bin/env python3
"""Export LEAP and final Wuji2 hands together after full processing.

Each waypoint cell shows:
- orange translucent LEAP source hand
- green final Wuji2 hand after wrist/root migration and fingertip alignment
- red LEAP target fingertips and cyan Wuji2 actual fingertips at GRASP

The script only reads existing ``final_waypoints.npz`` files.
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

from export_full_processed_wuji2_retarget_examples import (  # noqa: E402
    ACTUAL_TIP_COLOR,
    TARGET_TIP_COLOR,
    add_tip_alignment_markers,
    load_case_with_extra,
)
from export_leap_to_wuji2_retarget_examples import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_LEAP_URDF,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WUJI2_URDF,
    UrdfVisualModel,
    add_floor_tile,
    add_root_marker,
)


LEAP_COLOR = np.asarray([245, 145, 35, 145], dtype=np.uint8)
WUJI2_COLOR = np.asarray([45, 170, 90, 220], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", dest="cases", type=Path, action="append", default=None)
    parser.add_argument("--leap-urdf", type=Path, default=DEFAULT_LEAP_URDF)
    parser.add_argument("--wuji2-urdf", type=Path, default=DEFAULT_WUJI2_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--export-name", default="leap_and_wuji2_full_processed_together.glb")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["pregrasp", "cover", "grasp", "squeeze", "lift"],
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def display_transform_for_wuji2_root(
    original_wuji2_root: np.ndarray,
    display_xyz: np.ndarray,
) -> np.ndarray:
    display_wuji2_root = np.array(original_wuji2_root, dtype=np.float64)
    display_wuji2_root[:3, 3] = display_xyz
    return display_wuji2_root @ np.linalg.inv(original_wuji2_root), display_wuji2_root


def main() -> None:
    args = parse_args()
    case_paths = args.cases if args.cases is not None else DEFAULT_CASES
    case_paths = [path.resolve() for path in case_paths if path.exists()]
    if not case_paths:
        raise RuntimeError("No final_waypoints.npz cases found.")

    leap_model = UrdfVisualModel(args.leap_urdf)
    wuji2_model = UrdfVisualModel(args.wuji2_urdf)
    leap_urdf_joints = set(leap_model.actuated_joint_names)

    scene = trimesh.Scene()
    manifest: dict[str, object] = {
        "output_contract": (
            "LEAP and final Wuji2 are shown together in the same migrated wrist/root "
            "coordinate relation. Orange=LEAP, green=Wuji2, red=LEAP target tips, "
            "cyan=Wuji2 actual tips."
        ),
        "leap_urdf": str(args.leap_urdf.resolve()),
        "wuji2_urdf": str(args.wuji2_urdf.resolve()),
        "target_tip_color": TARGET_TIP_COLOR.tolist(),
        "actual_tip_color": ACTUAL_TIP_COLOR.tolist(),
        "cases": [],
    }

    x_spacing = 0.50
    y_spacing = 0.40
    for case_index, case_path in enumerate(case_paths):
        payload = load_case_with_extra(case_path)
        if len(payload.leap_joint_names) != 16 or set(payload.leap_joint_names) != leap_urdf_joints:
            raise RuntimeError(f"Invalid LEAP q16 order for {payload.path}: {payload.leap_joint_names}")
        wanted = [stage for stage in args.stages if stage in payload.waypoint_names]
        case_record = {
            "case_path": str(payload.path),
            "source_candidate_index": payload.source_candidate_index,
            "score": payload.score,
            "leap_joint_order": payload.leap_joint_names,
            "stages": [],
        }
        display_grasp_root = None
        for stage_index, stage_name in enumerate(wanted):
            waypoint_index = payload.waypoint_names.index(stage_name)
            cell_origin = np.asarray([stage_index * x_spacing, -case_index * y_spacing, 0.0])
            add_floor_tile(scene, cell_origin, f"together_tile_case{case_index:02d}_{stage_name}")

            original_wuji2_root = np.asarray(payload.wuji2_pose[waypoint_index], dtype=np.float64)
            group_transform, display_wuji2_root = display_transform_for_wuji2_root(
                original_wuji2_root, cell_origin
            )
            original_leap_root = np.asarray(payload.leap_pose[waypoint_index], dtype=np.float64)
            display_leap_root = group_transform @ original_leap_root
            if stage_name == "grasp":
                display_grasp_root = display_wuji2_root.copy()

            leap_q = dict(zip(payload.leap_joint_names, payload.leap_q[waypoint_index].tolist()))
            for name, mesh in leap_model.scene_geometry(
                display_leap_root,
                leap_q,
                LEAP_COLOR,
                f"together_case{case_index:02d}_{stage_name}_leap",
            ):
                scene.add_geometry(mesh, geom_name=name)
            add_root_marker(
                scene,
                display_leap_root,
                f"together_case{case_index:02d}_{stage_name}_leap_root",
            )

            wuji2_q = dict(zip(payload.wuji2_joint_names, payload.wuji2_q[waypoint_index].tolist()))
            for name, mesh in wuji2_model.scene_geometry(
                display_wuji2_root,
                wuji2_q,
                WUJI2_COLOR,
                f"together_case{case_index:02d}_{stage_name}_wuji2",
            ):
                scene.add_geometry(mesh, geom_name=name)
            add_root_marker(
                scene,
                display_wuji2_root,
                f"together_case{case_index:02d}_{stage_name}_wuji2_root",
            )

            root_link = trimesh.load_path(
                np.asarray([[display_leap_root[:3, 3], display_wuji2_root[:3, 3]]])
            )
            scene.add_geometry(root_link, geom_name=f"together_case{case_index:02d}_{stage_name}_root_link")
            case_record["stages"].append(
                {
                    "name": stage_name,
                    "waypoint_index": waypoint_index,
                    "leap_root_display_xyz": display_leap_root[:3, 3].round(6).tolist(),
                    "wuji2_root_display_xyz": display_wuji2_root[:3, 3].round(6).tolist(),
                }
            )
        if display_grasp_root is not None:
            case_record["fingertip_alignment"] = add_tip_alignment_markers(
                scene, payload, display_grasp_root, f"together_case{case_index:02d}_grasp"
            )
        manifest["cases"].append(case_record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_path = args.output_dir / args.export_name
    manifest_path = args.output_dir / "leap_and_wuji2_full_processed_together_manifest.json"
    scene.export(export_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EXPORT] {export_path}")
    print(f"[MANIFEST] {manifest_path}")
    print(f"[CASES] {len(case_paths)}")
    if args.show:
        scene.show()


if __name__ == "__main__":
    main()
