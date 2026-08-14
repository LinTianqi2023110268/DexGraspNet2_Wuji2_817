#!/usr/bin/env python3
"""Show LEAP GRASP/SQUEEZE and fixed-root Wuji2 GRASP/SQUEEZE together."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from case_paths import PROJECT_ROOT, SHARED_ROOT, active_case_root  # noqa: E402

CASE_ROOT = active_case_root()
RESULT = CASE_ROOT / "04_squeeze/squeeze_official.npz"
OUTPUT = CASE_ROOT / "05_visualization/four_hand_final.glb"
MANIFEST = CASE_ROOT / "05_visualization/visualization_manifest.json"
LEAP_URDF = (
    SHARED_ROOT / "models/robot_models/urdf/leap_hand.urdf"
)
WUJI_URDF = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/hand2/hand2_beta1/body/urdf/right.urdf"
)
sys.path.insert(0, str(SHARED_ROOT / "lib"))
from common_kinematics import UrdfKinematicModel  # noqa: E402


FINGER_COLORS = {
    "thumb": [245, 70, 70, 255],
    "index": [55, 220, 95, 255],
    "middle": [70, 135, 245, 255],
    "ring": [245, 170, 45, 255],
}


def color(mesh, rgba):
    mesh.visual.face_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.faces), 1))


def add_hand(scene, model, names, q, pose, prefix, rgba):
    fk = model.forward_kinematics(dict(zip(names, q)))
    for mesh_name, source, visual_origin in model.visual_meshes():
        link = mesh_name.rsplit("_", 1)[0]
        mesh = source.copy()
        mesh.apply_transform(pose @ fk[link] @ visual_origin)
        color(mesh, rgba)
        scene.add_geometry(mesh, geom_name=f"{prefix}_{mesh_name}")


def add_sphere(scene, point, name, rgba, radius):
    mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
    mesh.apply_translation(np.asarray(point, dtype=float))
    color(mesh, rgba)
    scene.add_geometry(mesh, geom_name=name)


def add_segment(scene, start, end, name, rgba):
    if np.linalg.norm(np.asarray(end) - np.asarray(start)) < 1.0e-9:
        return
    mesh = trimesh.creation.cylinder(radius=0.0008, segment=np.stack([start, end]))
    color(mesh, rgba)
    scene.add_geometry(mesh, geom_name=name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    for path in (RESULT, LEAP_URDF, WUJI_URDF):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(RESULT, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    leap_names = [str(value) for value in data["leap_joint_names"].tolist()]
    wuji_names = [str(value) for value in data["wuji2_joint_names"].tolist()]
    leap_pose = np.asarray(data["source_leap_pose_world"], dtype=float)
    wuji_pose = np.asarray(data["fixed_wuji2_root_pose_world"], dtype=float)
    leap_model = UrdfKinematicModel(LEAP_URDF, "hand_base_link", SHARED_ROOT / "models")
    wuji_model = UrdfKinematicModel(WUJI_URDF, "r_wrist")

    scene = trimesh.Scene()
    add_hand(scene, leap_model, leap_names, data["leap_q16_grasp"], leap_pose, "BLUE_LEAP_GRASP", [45, 125, 245, 58])
    add_hand(scene, leap_model, leap_names, data["leap_q16_squeeze"], leap_pose, "CYAN_LEAP_SQUEEZE", [35, 220, 235, 78])
    add_hand(scene, wuji_model, wuji_names, data["wuji2_q20_grasp"], wuji_pose, "PURPLE_WUJI_GRASP", [185, 75, 235, 70])
    add_hand(scene, wuji_model, wuji_names, data["wuji2_q20_squeeze"], wuji_pose, "GREEN_WUJI_RETRY_SQUEEZE", [45, 225, 100, 100])
    leap_tip = np.asarray(data["leap_four_tip_world_m"][-1], dtype=float)
    wuji_tip = np.asarray(data["wuji2_four_tip_world_m"][-1], dtype=float)
    fingers = [str(value) for value in data["finger_names"].tolist()]
    for index, finger in enumerate(fingers):
        rgba = FINGER_COLORS[finger]
        add_sphere(scene, leap_tip[index], f"{finger}_LEAP_SQUEEZE_TARGET", rgba, 0.0035)
        add_sphere(scene, wuji_tip[index], f"{finger}_WUJI_SQUEEZE", [255, 255, 255, 255], 0.0025)
        add_segment(scene, wuji_tip[index], leap_tip[index], f"{finger}_SQUEEZE_ERROR", rgba)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scene.export(OUTPUT)
    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "file": str(OUTPUT),
                "colors": {
                    "blue": "LEAP network GRASP",
                    "cyan": "LEAP official SQUEEZE",
                    "purple": "accepted Wuji2 GRASP with fixed aligned root",
                    "green": "new official-retargeting Wuji2 SQUEEZE",
                    "colored_spheres": "LEAP SQUEEZE fingertip targets",
                    "white_spheres": "Wuji2 SQUEEZE fingertips",
                    "colored_lines": "remaining fingertip errors",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[OK] visualization={OUTPUT}")
    if args.show:
        trimesh.load(OUTPUT, force="scene", process=False).show()


if __name__ == "__main__":
    main()
