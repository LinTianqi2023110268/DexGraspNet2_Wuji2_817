"""Shared Trimesh helpers for the DexGraspNet2 teaching visualizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
LEAP_URDF = REPO_ROOT / "robot_models" / "urdf" / "leap_hand_simplified.urdf"
LEAP_META = REPO_ROOT / "robot_models" / "meta" / "leap_hand" / "meta.yaml"

SEGMENT_COLORS = {
    0: np.asarray([145, 145, 145, 150], dtype=np.uint8),
    1: np.asarray([242, 133, 45, 220], dtype=np.uint8),
    2: np.asarray([45, 145, 230, 220], dtype=np.uint8),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def point_colors(segmentation: np.ndarray, alpha: int = 220) -> np.ndarray:
    segmentation = np.asarray(segmentation).reshape(-1)
    colors = np.empty((len(segmentation), 4), dtype=np.uint8)
    for segment_id in np.unique(segmentation):
        color = SEGMENT_COLORS.get(
            int(segment_id), np.asarray([180, 80, 200, alpha], dtype=np.uint8)
        ).copy()
        color[3] = alpha
        colors[segmentation == segment_id] = color
    return colors


def add_point_cloud(
    scene: trimesh.Scene,
    points: np.ndarray,
    segmentation: np.ndarray,
    name: str = "network_point_cloud",
    max_points: int | None = None,
) -> None:
    points = np.asarray(points)
    segmentation = np.asarray(segmentation)
    if max_points is not None and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
        segmentation = segmentation[indices]
    scene.add_geometry(
        trimesh.points.PointCloud(points, colors=point_colors(segmentation)),
        geom_name=name,
    )


def add_axis(
    scene: trimesh.Scene,
    transform: np.ndarray | None = None,
    name: str = "coordinate_frame",
    axis_length: float = 0.12,
) -> None:
    scene.add_geometry(
        trimesh.creation.axis(
            origin_size=axis_length * 0.055,
            axis_length=axis_length,
            transform=np.eye(4) if transform is None else transform,
        ),
        geom_name=name,
    )


def add_table(scene: trimesh.Scene, frame_transform: np.ndarray | None = None) -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = -0.003
    if frame_transform is not None:
        transform = frame_transform @ transform
    table = trimesh.creation.box(extents=[0.56, 0.48, 0.006], transform=transform)
    table.visual.face_colors = [155, 155, 155, 80]
    scene.add_geometry(table, geom_name="table_z_0")


def add_scene_objects(
    scene: trimesh.Scene,
    manifest: Mapping,
    world_to_display: np.ndarray | None = None,
    alpha: int = 100,
) -> None:
    palette = ([242, 133, 45, alpha], [45, 145, 230, alpha])
    display = np.eye(4) if world_to_display is None else world_to_display
    for index, obj in enumerate(manifest["objects"]):
        mesh = trimesh.load(obj["visual_mesh"], force="mesh", process=False)
        mesh = mesh.copy()
        mesh.apply_transform(display @ np.asarray(obj["pose_world_object"], dtype=np.float64))
        mesh.visual.face_colors = palette[index % len(palette)]
        scene.add_geometry(mesh, geom_name="object_{}".format(obj["code"]))


def load_leap_robot_model():
    # Import lazily so the simple input viewer does not require PyTorch3D.
    from src.utils.robot_model import RobotModel

    return RobotModel(str(LEAP_URDF), str(LEAP_META))


def add_leap_hand(
    scene: trimesh.Scene,
    robot_model,
    pose_world_hand: np.ndarray,
    qpos: Mapping[str, float],
    name_prefix: str,
    color: Sequence[int] = (60, 220, 110, 210),
) -> None:
    """Add the articulated LEAP visual meshes using the repository's FK code."""

    missing = [name for name in robot_model.joint_names if name not in qpos]
    if missing:
        raise KeyError("Missing LEAP joint values: {}".format(missing))
    qpos_tensors = {
        name: torch.tensor([float(qpos[name])], dtype=torch.float32)
        for name in robot_model.joint_names
    }
    translations, rotations = robot_model.forward_kinematics(qpos_tensors)
    rgba = np.asarray(color, dtype=np.uint8)
    for link_name in robot_model.link_names:
        vertices, faces = robot_model.get_link_mesh(link_name, "visual")
        if vertices.numel() == 0 or faces.numel() == 0:
            continue
        pose_hand_link = as_transform(
            rotations[link_name][0].detach().cpu().numpy(),
            translations[link_name][0].detach().cpu().numpy(),
        )
        mesh = trimesh.Trimesh(
            vertices=vertices.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
            process=False,
        )
        mesh.apply_transform(np.asarray(pose_world_hand) @ pose_hand_link)
        mesh.visual.face_colors = rgba
        scene.add_geometry(mesh, geom_name="{}_{}".format(name_prefix, link_name))


def show_or_export(scene: trimesh.Scene, export: Path | None, title: str) -> None:
    if export is not None:
        export = export.resolve()
        export.parent.mkdir(parents=True, exist_ok=True)
        scene.export(export)
        print("exported {}".format(export))
        return
    try:
        scene.show(caption=title)
    except ModuleNotFoundError as exc:
        if exc.name == "pyglet":
            raise RuntimeError(
                "Trimesh GUI needs pyglet. Install it in graspnet2.0, or rerun "
                "with --export output.glb."
            ) from exc
        raise
