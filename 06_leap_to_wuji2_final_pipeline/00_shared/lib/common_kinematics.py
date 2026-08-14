#!/usr/bin/env python3
"""Shared, read-only kinematic utilities for the LEAP-to-Wuji2 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import transforms3d
from urdf_parser_py.urdf import Mesh as UrdfMesh
from urdf_parser_py.urdf import Robot


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
CONFIG_PATH = EXPERIMENT_ROOT / "config/semantic_mapping.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def origin_transform(origin) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if origin is None:
        return result
    xyz = origin.xyz if origin.xyz is not None else [0.0, 0.0, 0.0]
    rpy = origin.rpy if origin.rpy is not None else [0.0, 0.0, 0.0]
    result[:3, :3] = transforms3d.euler.euler2mat(*rpy, axes="sxyz")
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def axis_rotation(axis, angle: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transforms3d.axangles.axangle2mat(
        np.asarray(axis, dtype=np.float64), float(angle)
    )
    return result


class UrdfKinematicModel:
    """Minimal URDF FK model; mesh loading is optional and handled lazily."""

    def __init__(self, urdf_path: Path, expected_root: str, package_root: Path | None = None):
        self.urdf_path = urdf_path.resolve()
        self.package_root = package_root.resolve() if package_root is not None else None
        self.robot = Robot.from_xml_file(str(self.urdf_path))
        self.joints = {joint.name: joint for joint in self.robot.joints}
        self.links = {link.name: link for link in self.robot.links}
        child_links = {joint.child for joint in self.robot.joints}
        roots = sorted(set(self.links) - child_links)
        if roots != [expected_root]:
            raise RuntimeError(
                f"{self.urdf_path}: expected root {expected_root!r}, got {roots}"
            )
        self.root = expected_root
        self.children: Dict[str, list] = {}
        for joint in self.robot.joints:
            self.children.setdefault(joint.parent, []).append(joint)

    @property
    def movable_joint_names(self) -> list[str]:
        return [
            joint.name
            for joint in self.robot.joints
            if joint.type in ("revolute", "continuous")
        ]

    def joint_limits(self, joint_name: str) -> tuple[float, float]:
        joint = self.joints[joint_name]
        if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
            raise RuntimeError(f"Joint {joint_name} has no finite lower/upper limits")
        return float(joint.limit.lower), float(joint.limit.upper)

    def forward_kinematics(self, qpos: Dict[str, float]) -> Dict[str, np.ndarray]:
        transforms = {self.root: np.eye(4, dtype=np.float64)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                transform = transforms[parent] @ origin_transform(joint.origin)
                if joint.type in ("revolute", "continuous"):
                    if joint.name not in qpos:
                        raise KeyError(f"Missing qpos for movable joint {joint.name}")
                    transform = transform @ axis_rotation(joint.axis, qpos[joint.name])
                elif joint.type != "fixed":
                    raise ValueError(f"Unsupported joint type {joint.type}: {joint.name}")
                transforms[joint.child] = transform
                stack.append(joint.child)
        return transforms

    def tip_positions(self, qpos: Dict[str, float], tip_links: dict) -> Dict[str, np.ndarray]:
        transforms = self.forward_kinematics(qpos)
        return {
            finger: np.asarray(transforms[link][:3, 3], dtype=np.float64)
            for finger, link in tip_links.items()
        }

    def resolve_mesh_path(self, filename: str) -> Path:
        if filename.startswith("package://"):
            if self.package_root is None:
                raise RuntimeError(f"package_root required for {filename}")
            relative = filename[len("package://") :]
            return (self.package_root / relative).resolve()
        return (self.urdf_path.parent / filename).resolve()

    def visual_meshes(self) -> Iterable[tuple[str, object, np.ndarray]]:
        import trimesh

        for link in self.robot.links:
            visuals = list(getattr(link, "visuals", []) or [])
            if not visuals and link.visual is not None:
                visuals = [link.visual]
            for visual_index, visual in enumerate(visuals):
                if not isinstance(visual.geometry, UrdfMesh):
                    continue
                path = self.resolve_mesh_path(visual.geometry.filename)
                if not path.is_file():
                    raise FileNotFoundError(path)
                loaded = trimesh.load(path, force="mesh", process=False)
                if isinstance(loaded, trimesh.Scene):
                    loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
                mesh = loaded.copy()
                if visual.geometry.scale is not None:
                    mesh.apply_scale(np.asarray(visual.geometry.scale, dtype=np.float64))
                yield f"{link.name}_{visual_index}", mesh, origin_transform(visual.origin)


def build_models(config: dict) -> tuple[UrdfKinematicModel, UrdfKinematicModel]:
    leap = config["leap"]
    wuji2 = config["wuji2"]
    leap_model = UrdfKinematicModel(
        resolve_project_path(leap["urdf"]),
        leap["root_link"],
        resolve_project_path(leap["package_root"]),
    )
    wuji2_model = UrdfKinematicModel(
        resolve_project_path(wuji2["urdf"]), wuji2["root_link"]
    )
    return leap_model, wuji2_model


def load_leap_pose(path: Path, index: int, joint_order: list[str]) -> tuple[str, dict]:
    path = path.resolve()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "poses" in payload:
            poses = payload["poses"]
            if not 0 <= index < len(poses):
                raise IndexError(f"index {index} outside JSON pose count {len(poses)}")
            record = poses[index]
            qpos = record["qpos"]
            name = str(record.get("name", f"pose_{index:03d}"))
        else:
            qpos = payload["qpos"]
            name = str(payload.get("name", path.stem))
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            missing = [name for name in joint_order if name not in payload.files]
            if missing:
                raise KeyError(f"NPZ lacks LEAP joint arrays: {missing}")
            counts = {len(np.atleast_1d(payload[name])) for name in joint_order}
            if len(counts) != 1:
                raise RuntimeError(f"Inconsistent joint array lengths: {counts}")
            count = counts.pop()
            if not 0 <= index < count:
                raise IndexError(f"index {index} outside NPZ pose count {count}")
            qpos = {name: float(np.atleast_1d(payload[name])[index]) for name in joint_order}
            name = f"{path.stem}_{index:03d}"
    else:
        raise ValueError("Input must be .json or .npz")
    missing = [name for name in joint_order if name not in qpos]
    if missing:
        raise KeyError(f"Pose lacks LEAP joints: {missing}")
    return name, {name: float(qpos[name]) for name in joint_order}


def normalized_coordinate(value: float, lower: float, upper: float) -> float:
    if not upper > lower:
        raise ValueError(f"Invalid joint limits [{lower}, {upper}]")
    return (float(value) - lower) / (upper - lower)


def target_from_normalized(value: float, lower: float, upper: float) -> float:
    return lower + float(value) * (upper - lower)

