#!/usr/bin/env python3
"""Visualize Stage-01 Wuji2 world-frame grasps with Trimesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import transforms3d
from urdf_parser_py.urdf import Mesh as UrdfMesh
from urdf_parser_py.urdf import Robot

from .project import project_path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ADAPTER_ROOT = PROJECT_ROOT
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "02_training_dataset/data/scene_datasets/"
    "wuji2_train60_100seminal_256view_v1"
)
DEFAULT_URDF = (
    PROJECT_ROOT
    / "02_training_dataset/assets/wuji2_factory/02_wuji2_hand/"
    "original_wuji2_right/body/urdf/right.urdf"
)
STAGE_NAME = "01_transformed_object_grasps"
PALETTE = np.asarray(
    [
        [240, 130, 40, 120],
        [45, 135, 225, 120],
        [180, 75, 210, 120],
        [75, 190, 105, 120],
        [220, 190, 45, 120],
        [20, 190, 200, 120],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--grasp-index", type=int, default=0)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--hide-points", action="store_true")
    return parser.parse_args()


def frame_transform(origin) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    xyz = np.asarray(origin.xyz if origin.xyz is not None else [0, 0, 0], dtype=np.float64)
    rpy = np.asarray(origin.rpy if origin.rpy is not None else [0, 0, 0], dtype=np.float64)
    transform[:3, :3] = transforms3d.euler.euler2mat(*rpy, axes="sxyz")
    transform[:3, 3] = xyz
    return transform


def axis_rotation(axis, angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = transforms3d.axangles.axangle2mat(
        np.asarray(axis, dtype=np.float64), float(angle)
    )
    return transform


class Wuji2VisualModel:
    def __init__(self, urdf_path: Path):
        self.urdf_path = urdf_path.resolve()
        self.robot = Robot.from_xml_file(str(self.urdf_path))
        link_names = {link.name for link in self.robot.links}
        child_names = {joint.child for joint in self.robot.joints}
        roots = sorted(link_names - child_names)
        if len(roots) != 1:
            raise RuntimeError(f"Expected exactly one URDF root link, got {roots}")
        self.root = roots[0]
        self.children: dict[str, list] = {}
        for joint in self.robot.joints:
            self.children.setdefault(joint.parent, []).append(joint)
        self.links = {link.name: link for link in self.robot.links}
        self.meshes: dict[str, list[tuple[trimesh.Trimesh, np.ndarray]]] = {}
        for link in self.robot.links:
            visuals = list(getattr(link, "visuals", []) or [])
            if not visuals and link.visual is not None:
                visuals = [link.visual]
            entries = []
            for visual in visuals:
                if not isinstance(visual.geometry, UrdfMesh):
                    continue
                mesh_path = (self.urdf_path.parent / visual.geometry.filename).resolve()
                loaded = trimesh.load(mesh_path, force="mesh", process=False)
                if isinstance(loaded, trimesh.Scene):
                    loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
                mesh = loaded.copy()
                scale = visual.geometry.scale
                if scale is not None:
                    mesh.apply_scale(np.asarray(scale, dtype=np.float64))
                entries.append((mesh, frame_transform(visual.origin)))
            if entries:
                self.meshes[link.name] = entries

    def forward_kinematics(self, qpos: dict[str, float]) -> dict[str, np.ndarray]:
        transforms = {self.root: np.eye(4, dtype=np.float64)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                transform = transforms[parent] @ frame_transform(joint.origin)
                if joint.type in ("revolute", "continuous"):
                    if joint.name not in qpos:
                        raise KeyError(f"Missing qpos for {joint.name}")
                    transform = transform @ axis_rotation(joint.axis, qpos[joint.name])
                elif joint.type != "fixed":
                    raise ValueError(f"Unsupported joint type {joint.type}: {joint.name}")
                transforms[joint.child] = transform
                stack.append(joint.child)
        return transforms

    def scene_geometry(
        self,
        world_from_base: np.ndarray,
        qpos: dict[str, float],
        color: np.ndarray,
        prefix: str,
    ) -> list[tuple[str, trimesh.Trimesh]]:
        link_transforms = self.forward_kinematics(qpos)
        result = []
        hand_color = np.asarray(color, dtype=np.uint8).copy()
        hand_color[3] = 220
        for link_name, entries in self.meshes.items():
            for visual_index, (source, visual_origin) in enumerate(entries):
                mesh = source.copy()
                mesh.apply_transform(
                    world_from_base @ link_transforms[link_name] @ visual_origin
                )
                mesh.visual.face_colors = hand_color
                result.append((f"{prefix}_{link_name}_{visual_index}", mesh))
        return result


def load_object_mesh(asset: dict) -> trimesh.Trimesh:
    loaded = trimesh.load(asset["source_obj"], process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = loaded.copy()
    center = np.asarray(asset["native_aabb_center"], dtype=np.float64)
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) * float(
        asset["scale"]
    )
    return mesh


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    scene_dir = output_root / "scenes" / f"scene_{args.scene:04d}"
    stage_dir = (
        output_root
        / "grasp_label_stages"
        / STAGE_NAME
        / f"scene_{args.scene:04d}"
    )
    scene_manifest = json.loads(
        (scene_dir / "scene_manifest.json").read_text(encoding="utf-8")
    )
    stage_manifest = json.loads(
        (stage_dir / "stage_manifest.json").read_text(encoding="utf-8")
    )
    joint_order = stage_manifest["label_contract"]["joint_order"]
    model = Wuji2VisualModel(args.urdf)
    display = trimesh.Scene()
    table = trimesh.creation.box(extents=scene_manifest["table"]["size_m"])
    table.apply_translation(
        [
            0.0,
            0.0,
            float(scene_manifest["table"]["top_z_m"])
            - 0.5 * float(scene_manifest["table"]["size_m"][2]),
        ]
    )
    table.visual.face_colors = np.asarray([145, 145, 145, 80], dtype=np.uint8)
    display.add_geometry(table, geom_name="table")
    display.add_geometry(
        trimesh.creation.axis(origin_size=0.006, axis_length=0.10),
        geom_name="world_frame",
    )
    if not args.hide_points:
        with np.load(scene_dir / "network_input.npz") as payload:
            points_camera = np.asarray(payload["pc"][0], dtype=np.float64)
            segmentation = np.asarray(payload["seg"][0], dtype=np.int64)
            world_from_camera = np.asarray(payload["extrinsics"][0], dtype=np.float64)
        points_world = (
            points_camera @ world_from_camera[:3, :3].T
            + world_from_camera[:3, 3]
        )
        colors = np.full((len(points_world), 4), [120, 120, 120, 90], dtype=np.uint8)
        for segment_id in range(1, 7):
            colors[segmentation == segment_id] = PALETTE[segment_id - 1]
        display.add_geometry(
            trimesh.points.PointCloud(points_world, colors=colors),
            geom_name="view_0000_points",
        )
    for object_index, scene_object in enumerate(scene_manifest["objects"]):
        object_mesh = load_object_mesh(scene_object["asset"])
        object_mesh.apply_transform(
            np.asarray(scene_object["T_world_centered_object"], dtype=np.float64)
        )
        object_mesh.visual.face_colors = PALETTE[object_index]
        display.add_geometry(
            object_mesh,
            geom_name=f"object_{object_index + 1:03d}_{scene_object['object_code']}",
        )
        record = stage_manifest["object_records"][object_index]
        with np.load(project_path(record["output_npz"])) as grasps:
            count = len(grasps["qpos"])
            grasp_index = args.grasp_index % count
            qpos_values = np.asarray(grasps["qpos"][grasp_index], dtype=np.float64)
            world_from_base = np.asarray(
                grasps["T_world_r_base_link"][grasp_index], dtype=np.float64
            )
        qpos = dict(zip(joint_order, qpos_values.tolist()))
        for name, mesh in model.scene_geometry(
            world_from_base,
            qpos,
            PALETTE[object_index],
            f"hand_{object_index + 1:03d}",
        ):
            display.add_geometry(mesh, geom_name=name)
        display.add_geometry(
            trimesh.creation.axis(
                origin_size=0.003,
                axis_length=0.045,
                transform=world_from_base,
            ),
            geom_name=f"r_base_link_frame_{object_index + 1:03d}",
        )
        print(
            f"object {object_index + 1}: {scene_object['object_code']} "
            f"grasp={grasp_index}/{count} "
            f"r_base_link={np.round(world_from_base[:3, 3], 4).tolist()}"
        )
    print("axes on each hand mark r_base_link; X=red, Y=green, Z=blue")
    print("Stage 01 only: scene/table collision filtering has not been applied")
    if args.export is None:
        display.show(caption="Wuji2 Stage-01 transformed scene grasps")
    else:
        destination = args.export.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        display.export(destination)
        print(f"exported: {destination}")


if __name__ == "__main__":
    main()
