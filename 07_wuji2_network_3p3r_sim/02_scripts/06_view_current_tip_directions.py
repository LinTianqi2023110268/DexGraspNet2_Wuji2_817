#!/usr/bin/env python3
"""Audit current GRASP-to-SQUEEZE fingertip frames and directions in Trimesh.

This script reads the exact currently generated native-Wuji2 task.  It does
not recompute or modify SQUEEZE.  The official companion URDF is used for both
the displayed hand and fingertip forward kinematics, so the frame is
``r_wrist`` exactly like the Isaac Sim job.

Colors
------
blue transparent hand: GRASP
green transparent hand: SQUEEZE
red/green/blue thin arrows: fingertip local +X/+Y/+Z at GRASP
magenta thick arrow: configured local inward direction, length 30 mm
white sphere: GRASP fingertip origin
yellow sphere: requested 30 mm target
orange sphere/line: achieved SQUEEZE fingertip and target error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config  # noqa: E402
from wuji2_dgn2.collision import load_wuji2_module  # noqa: E402
from wuji2_dgn2.official_asset import canonical_asset_paths  # noqa: E402
from wuji2_dgn2.visual import Wuji2VisualModel, load_object_mesh  # noqa: E402


CASE_ROOT = PIPELINE_ROOT / "01_cases/selected_native_case"
JOB_PATH = CASE_ROOT / "03_waypoints/native_wuji2_3p3r_waypoints.npz"
CASE_PATH = CASE_ROOT / "case.json"
CONFIG_PATH = PIPELINE_ROOT / "00_config/test_runtime_config.json"
COLORS = {
    "grasp": np.asarray([40, 125, 255, 105], dtype=np.uint8),
    "squeeze": np.asarray([40, 220, 105, 105], dtype=np.uint8),
    "x": np.asarray([245, 45, 45, 255], dtype=np.uint8),
    "y": np.asarray([30, 210, 70, 255], dtype=np.uint8),
    "z": np.asarray([35, 80, 245, 255], dtype=np.uint8),
    "normal": np.asarray([235, 20, 205, 255], dtype=np.uint8),
    "origin": np.asarray([250, 250, 250, 255], dtype=np.uint8),
    "target": np.asarray([255, 220, 20, 255], dtype=np.uint8),
    "actual": np.asarray([255, 125, 15, 255], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def add_sphere(scene: trimesh.Scene, point: np.ndarray, radius: float, color, name: str) -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.apply_translation(point)
    sphere.visual.face_colors = color
    scene.add_geometry(sphere, geom_name=name)


def add_tube(
    scene: trimesh.Scene,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color,
    name: str,
) -> None:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-9:
        return
    direction = vector / length
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    transform[:3, 3] = 0.5 * (np.asarray(start) + np.asarray(end))
    tube = trimesh.creation.cylinder(
        radius=radius, height=length, sections=12, transform=transform
    )
    tube.visual.face_colors = color
    scene.add_geometry(tube, geom_name=name)


def add_arrow(
    scene: trimesh.Scene,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    color,
    name: str,
) -> None:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    head_length = min(0.005, 0.28 * length)
    shaft_end = np.asarray(start) + direction * (length - head_length)
    end = np.asarray(start) + direction * length
    add_tube(scene, start, shaft_end, radius, color, f"{name}_shaft")
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    transform[:3, 3] = shaft_end
    cone = trimesh.creation.cone(
        radius=2.4 * radius, height=head_length, sections=16, transform=transform
    )
    cone.visual.face_colors = color
    scene.add_geometry(cone, geom_name=f"{name}_head")
    # An endpoint marker makes overlapping +X and inward arrows legible.
    add_sphere(scene, end, 1.6 * radius, color, f"{name}_endpoint")


def main() -> None:
    args = parse_args()
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    with np.load(JOB_PATH, allow_pickle=False) as archive:
        job = {key: archive[key] for key in archive.files}
    waypoint_names = [str(x) for x in job["waypoint_names"].tolist()]
    grasp_i = waypoint_names.index("grasp")
    squeeze_i = waypoint_names.index("squeeze")
    q = np.asarray(job["waypoint_joint_positions"][0], dtype=np.float32)
    wrist_pose = np.asarray(job["waypoint_pose_world"][0, grasp_i], dtype=np.float64)
    joint_names = [str(x) for x in job["finger_joint_names"].tolist()]

    _official_usd, official_urdf = canonical_asset_paths()
    visual_model = Wuji2VisualModel(official_urdf)
    module = load_wuji2_module()
    kinematic_model = module.Wuji2HandKinematics(
        official_urdf, device=torch.device("cpu"), dtype=torch.float32
    )
    cfg = load_config(CONFIG_PATH)["grasp_label_generation"]
    tip_normals = cfg["squeeze_fingertip_normals"]
    tip_names = list(tip_normals)

    display = trimesh.Scene()
    scene_manifest = json.loads(
        (CASE_ROOT / "01_input" / f"{case['scene_id']}_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for record in scene_manifest["objects"]:
        mesh = load_object_mesh(record["asset"])
        mesh.apply_transform(record["T_world_centered_object"])
        if int(record["segmentation_id"]) == int(case["target_segmentation_id"]):
            mesh.visual.face_colors = [255, 188, 25, 145]
        else:
            mesh.visual.face_colors = [145, 150, 160, 28]
        display.add_geometry(
            mesh, geom_name=f"scene_object_{int(record['segmentation_id']):03d}"
        )
    table = trimesh.creation.box(extents=scene_manifest["table"]["size_m"])
    table.apply_translation(
        [
            0.0,
            0.0,
            float(scene_manifest["table"]["top_z_m"])
            - 0.5 * float(scene_manifest["table"]["size_m"][2]),
        ]
    )
    table.visual.face_colors = [175, 175, 175, 25]
    display.add_geometry(table, geom_name="table")

    for stage_name, stage_index, color in (
        ("grasp", grasp_i, COLORS["grasp"]),
        ("squeeze", squeeze_i, COLORS["squeeze"]),
    ):
        q_dict = dict(zip(joint_names, q[stage_index].astype(float).tolist()))
        for name, mesh in visual_model.scene_geometry(
            wrist_pose, q_dict, color, stage_name
        ):
            display.add_geometry(mesh, geom_name=name)

    q_grasp = torch.as_tensor(q[grasp_i : grasp_i + 1])
    q_squeeze = torch.as_tensor(q[squeeze_i : squeeze_i + 1])
    fk_grasp = kinematic_model.forward_kinematics_base(q_grasp)
    fk_squeeze = kinematic_model.forward_kinematics_base(q_squeeze)
    squeeze_distance = float(np.asarray(job["squeeze_width_m"]).item())
    records = []
    for tip_index, tip_name in enumerate(tip_names):
        tip_grasp_wrist = fk_grasp[tip_name][0].detach().cpu().numpy()
        tip_squeeze_wrist = fk_squeeze[tip_name][0].detach().cpu().numpy()
        tip_grasp_world = wrist_pose @ tip_grasp_wrist
        tip_squeeze_world = wrist_pose @ tip_squeeze_wrist
        origin = tip_grasp_world[:3, 3]
        actual = tip_squeeze_world[:3, 3]
        rotation = tip_grasp_world[:3, :3]
        normal_local = np.asarray(tip_normals[tip_name], dtype=np.float64)
        normal_world = rotation @ normal_local
        normal_world /= np.linalg.norm(normal_world)
        target = origin + squeeze_distance * normal_world

        add_sphere(display, origin, 0.0030, COLORS["origin"], f"{tip_name}_grasp_origin")
        add_sphere(display, target, 0.0032, COLORS["target"], f"{tip_name}_requested_target")
        add_sphere(display, actual, 0.0027, COLORS["actual"], f"{tip_name}_achieved_tip")
        for axis_index, (axis_name, axis_color) in enumerate(
            (("x", COLORS["x"]), ("y", COLORS["y"]), ("z", COLORS["z"]))
        ):
            add_arrow(
                display,
                origin,
                rotation[:, axis_index],
                0.012,
                0.00055,
                axis_color,
                f"{tip_name}_local_plus_{axis_name}",
            )
        add_arrow(
            display,
            origin,
            normal_world,
            squeeze_distance,
            0.00125,
            COLORS["normal"],
            f"{tip_name}_configured_inward_30mm",
        )
        add_tube(
            display,
            target,
            actual,
            0.0008,
            COLORS["actual"],
            f"{tip_name}_ik_target_error",
        )
        actual_delta = actual - origin
        records.append(
            {
                "tip": tip_name,
                "local_inward_xyz": normal_local.tolist(),
                "grasp_origin_world_m": origin.tolist(),
                "local_x_world": rotation[:, 0].tolist(),
                "local_y_world": rotation[:, 1].tolist(),
                "local_z_world": rotation[:, 2].tolist(),
                "configured_inward_world": normal_world.tolist(),
                "requested_target_world_m": target.tolist(),
                "achieved_squeeze_tip_world_m": actual.tolist(),
                "actual_displacement_mm": float(1000.0 * np.linalg.norm(actual_delta)),
                "along_configured_normal_mm": float(1000.0 * np.dot(actual_delta, normal_world)),
            }
        )

    output = (
        args.output.resolve()
        if args.output is not None
        else CASE_ROOT
        / "04_visualization"
        / f"{case['case_id']}_tip_frames_squeeze_direction_audit.glb"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    display.export(output)
    audit_path = output.with_suffix(".json")
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": case["case_id"],
                "squeeze_width_m": squeeze_distance,
                "keep_z": bool(case["frozen_native_action_contract"]["squeeze_keep_z"]),
                "legend": __doc__.split("Colors\n------\n", 1)[1].strip(),
                "tips": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"case={case['case_id']}")
    print(f"squeeze={1000.0 * squeeze_distance:.1f} mm keep_z=False")
    for record in records:
        print(
            f"{record['tip']:25s} local={record['local_inward_xyz']} "
            f"world={np.round(record['configured_inward_world'], 5).tolist()} "
            f"actual={record['actual_displacement_mm']:.2f} mm"
        )
    print(f"output={output}")
    print(f"audit={audit_path}")
    if args.show:
        display.show(caption=f"{case['case_id']} fingertip direction audit")


if __name__ == "__main__":
    main()
