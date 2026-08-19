#!/usr/bin/env python3
"""Export LEAP -> Wuji2 retargeted waypoint examples as a Trimesh GLB scene.

The script is intentionally read-only with respect to retargeting: it consumes
existing ``final_waypoints.npz`` files and visualizes the source LEAP hand next
to the finalized Wuji2 hand for each waypoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import transforms3d
from urdf_parser_py.urdf import Mesh as UrdfMesh
from urdf_parser_py.urdf import Robot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "trimesh/outputs/leap_to_wuji2_retarget_examples"
DEFAULT_LEAP_URDF = (
    PROJECT_ROOT
    / "06_leap_to_wuji2_final_pipeline/00_shared/models/robot_models/urdf/leap_hand.urdf"
)
DEFAULT_WUJI2_URDF = (
    PROJECT_ROOT
    / "02_training_dataset/assets/wuji2_factory/02_wuji2_hand/"
    "original_wuji2_right/body/urdf/right.urdf"
)
DEFAULT_CASES = [
    PROJECT_ROOT
    / "06_leap_to_wuji2_final_pipeline/04_verified_baseline/"
    "scene0001_view0001_official_rank0/task/final_waypoints.npz",
    PROJECT_ROOT
    / "06_leap_to_wuji2_final_pipeline/01_cases/active/"
    "scene0009_view0004_official_rank0_pinky_ring/06_isaacsim/final_waypoints.npz",
]
# Official DGN2/LEAP q16 serialization order. This differs from raw URDF joint
# traversal for the abd/flex pair of index, middle and ring fingers.
DEFAULT_LEAP_Q16_ORDER = [
    "j12",
    "j13",
    "j14",
    "j15",
    "j1",
    "j0",
    "j2",
    "j3",
    "j9",
    "j8",
    "j10",
    "j11",
    "j5",
    "j4",
    "j6",
    "j7",
]
LEAP_COLOR = np.asarray([245, 145, 35, 225], dtype=np.uint8)
WUJI2_COLOR = np.asarray([45, 170, 90, 225], dtype=np.uint8)
AXIS_LENGTH_M = 0.055


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
    parser.add_argument("--leap-urdf", type=Path, default=DEFAULT_LEAP_URDF)
    parser.add_argument("--wuji2-urdf", type=Path, default=DEFAULT_WUJI2_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--export-name", default="leap_to_wuji2_retarget_examples.glb")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["pregrasp", "cover", "grasp", "squeeze", "lift"],
        help="Waypoint names to include.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a Trimesh viewer after exporting. Requires a working GUI.",
    )
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


def resolve_mesh_filename(urdf_path: Path, filename: str) -> Path:
    package_prefix = "package://robot_models/"
    if filename.startswith(package_prefix):
        package_root = urdf_path.parents[1]
        return (package_root / filename[len(package_prefix) :]).resolve()
    if filename.startswith("package://"):
        raise RuntimeError(f"Unsupported package URI in {urdf_path}: {filename}")
    return (urdf_path.parent / filename).resolve()


class UrdfVisualModel:
    def __init__(self, urdf_path: Path):
        self.urdf_path = urdf_path.resolve()
        self.robot = Robot.from_xml_file(str(self.urdf_path))
        link_names = {link.name for link in self.robot.links}
        child_names = {joint.child for joint in self.robot.joints}
        roots = sorted(link_names - child_names)
        if len(roots) != 1:
            raise RuntimeError(f"Expected one root link for {self.urdf_path}, got {roots}")
        self.root = roots[0]
        self.children: dict[str, list] = {}
        for joint in self.robot.joints:
            self.children.setdefault(joint.parent, []).append(joint)
        self.meshes: dict[str, list[tuple[trimesh.Trimesh, np.ndarray]]] = {}
        for link in self.robot.links:
            visuals = list(getattr(link, "visuals", []) or [])
            if not visuals and link.visual is not None:
                visuals = [link.visual]
            entries = []
            for visual in visuals:
                if not isinstance(visual.geometry, UrdfMesh):
                    continue
                mesh_path = resolve_mesh_filename(self.urdf_path, visual.geometry.filename)
                loaded = trimesh.load(mesh_path, force="mesh", process=False)
                if isinstance(loaded, trimesh.Scene):
                    loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
                mesh = loaded.copy()
                if visual.geometry.scale is not None:
                    mesh.apply_scale(np.asarray(visual.geometry.scale, dtype=np.float64))
                entries.append((mesh, frame_transform(visual.origin)))
            if entries:
                self.meshes[link.name] = entries

    @property
    def actuated_joint_names(self) -> list[str]:
        return [joint.name for joint in self.robot.joints if joint.type != "fixed"]

    def forward_kinematics(self, qpos: dict[str, float]) -> dict[str, np.ndarray]:
        transforms = {self.root: np.eye(4, dtype=np.float64)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                transform = transforms[parent] @ frame_transform(joint.origin)
                if joint.type in ("revolute", "continuous"):
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
        result: list[tuple[str, trimesh.Trimesh]] = []
        for link_name, entries in self.meshes.items():
            for visual_index, (source, visual_origin) in enumerate(entries):
                mesh = source.copy()
                mesh.apply_transform(world_from_base @ link_transforms[link_name] @ visual_origin)
                mesh.visual.face_colors = color
                result.append((f"{prefix}_{link_name}_{visual_index}", mesh))
        return result


@dataclass
class CasePayload:
    path: Path
    waypoint_names: list[str]
    leap_joint_names: list[str]
    wuji2_joint_names: list[str]
    wuji2_q: np.ndarray
    wuji2_pose: np.ndarray
    leap_q: np.ndarray
    leap_pose: np.ndarray
    score: float | None
    source_candidate_index: int | None


def _unwrap_first_candidate(array: np.ndarray) -> np.ndarray:
    if array.ndim >= 3 and array.shape[0] == 1:
        return np.asarray(array[0])
    return np.asarray(array)


def resolve_existing_npz(path_value) -> Path | None:
    raw = np.asarray(path_value).item() if isinstance(path_value, np.ndarray) else path_value
    path = Path(str(raw))
    if path.is_file():
        return path
    text = str(path)
    marker = "/08_leap_to_wuji2_final_pipeline/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        candidate = PROJECT_ROOT / "06_leap_to_wuji2_final_pipeline" / suffix
        if candidate.is_file():
            return candidate
        if "/01_cases/" in text:
            active_candidate = (
                PROJECT_ROOT
                / "06_leap_to_wuji2_final_pipeline/01_cases/active"
                / text.split("/01_cases/", 1)[1]
            )
            if active_candidate.is_file():
                return active_candidate
    return None


def load_leap_joint_names(final_path: Path, data) -> list[str]:
    if "leap_joint_names" in data.files:
        return [str(x) for x in data["leap_joint_names"].tolist()]
    if "retarget_source_npz" in data.files:
        source_path = resolve_existing_npz(data["retarget_source_npz"])
        if source_path is not None:
            with np.load(source_path, allow_pickle=True) as source:
                if "leap_joint_names" in source.files:
                    return [str(x) for x in source["leap_joint_names"].tolist()]
    for parent in final_path.resolve().parents:
        for relative in (
            Path("02_retargeting/grasp_official.npz"),
            Path("04_squeeze/squeeze_official.npz"),
        ):
            source_path = parent / relative
            if source_path.is_file():
                with np.load(source_path, allow_pickle=True) as source:
                    if "leap_joint_names" in source.files:
                        return [str(x) for x in source["leap_joint_names"].tolist()]
    return list(DEFAULT_LEAP_Q16_ORDER)


def load_case(path: Path) -> CasePayload:
    with np.load(path, allow_pickle=True) as data:
        score = float(data["score"]) if "score" in data.files else None
        source_candidate_index = (
            int(data["source_candidate_index"]) if "source_candidate_index" in data.files else None
        )
        return CasePayload(
            path=path.resolve(),
            waypoint_names=[str(x) for x in data["waypoint_names"].tolist()],
            leap_joint_names=load_leap_joint_names(path, data),
            wuji2_joint_names=[str(x) for x in data["finger_joint_names"].tolist()],
            wuji2_q=_unwrap_first_candidate(np.asarray(data["waypoint_joint_positions"])),
            wuji2_pose=_unwrap_first_candidate(np.asarray(data["waypoint_pose_world"])),
            leap_q=_unwrap_first_candidate(np.asarray(data["source_leap_waypoint_joint_positions"])),
            leap_pose=_unwrap_first_candidate(np.asarray(data["source_leap_waypoint_pose_world"])),
            score=score,
            source_candidate_index=source_candidate_index,
        )


def add_root_marker(scene: trimesh.Scene, transform: np.ndarray, name: str) -> None:
    scene.add_geometry(
        trimesh.creation.axis(
            origin_size=0.004,
            axis_radius=0.0012,
            axis_length=AXIS_LENGTH_M,
            transform=transform,
        ),
        geom_name=name,
    )


def add_floor_tile(scene: trimesh.Scene, center: np.ndarray, name: str) -> None:
    tile = trimesh.creation.box(extents=[0.36, 0.24, 0.003])
    tile.apply_translation(center + np.asarray([0.0, 0.0, -0.085]))
    tile.visual.face_colors = np.asarray([90, 90, 90, 65], dtype=np.uint8)
    scene.add_geometry(tile, geom_name=name)


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
        "output_contract": "LEAP source hand shown left/orange; Wuji2 retargeted hand shown right/green.",
        "leap_urdf": str(args.leap_urdf.resolve()),
        "wuji2_urdf": str(args.wuji2_urdf.resolve()),
        "cases": [],
    }

    x_spacing = 0.62
    y_spacing = 0.42
    hand_offset = 0.12
    for case_index, case_path in enumerate(case_paths):
        payload = load_case(case_path)
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
        for stage_index, stage_name in enumerate(wanted):
            waypoint_index = payload.waypoint_names.index(stage_name)
            cell_origin = np.asarray([stage_index * x_spacing, -case_index * y_spacing, 0.0])
            add_floor_tile(scene, cell_origin, f"tile_case{case_index:02d}_{stage_name}")

            leap_base = np.array(payload.leap_pose[waypoint_index], dtype=np.float64)
            leap_base[:3, 3] = cell_origin + np.asarray([-hand_offset, 0.0, 0.0])
            leap_q = dict(zip(payload.leap_joint_names, payload.leap_q[waypoint_index].tolist()))
            for name, mesh in leap_model.scene_geometry(
                leap_base, leap_q, LEAP_COLOR, f"case{case_index:02d}_{stage_name}_leap"
            ):
                scene.add_geometry(mesh, geom_name=name)
            add_root_marker(scene, leap_base, f"case{case_index:02d}_{stage_name}_leap_root")

            wuji2_base = np.array(payload.wuji2_pose[waypoint_index], dtype=np.float64)
            wuji2_base[:3, 3] = cell_origin + np.asarray([hand_offset, 0.0, 0.0])
            wuji2_q = dict(zip(payload.wuji2_joint_names, payload.wuji2_q[waypoint_index].tolist()))
            for name, mesh in wuji2_model.scene_geometry(
                wuji2_base, wuji2_q, WUJI2_COLOR, f"case{case_index:02d}_{stage_name}_wuji2"
            ):
                scene.add_geometry(mesh, geom_name=name)
            add_root_marker(scene, wuji2_base, f"case{case_index:02d}_{stage_name}_wuji2_root")

            connector = trimesh.load_path(
                np.asarray([[leap_base[:3, 3], wuji2_base[:3, 3]]], dtype=np.float64)
            )
            scene.add_geometry(connector, geom_name=f"case{case_index:02d}_{stage_name}_root_link")
            case_record["stages"].append(
                {
                    "name": stage_name,
                    "waypoint_index": waypoint_index,
                    "leap_display_xyz": leap_base[:3, 3].round(6).tolist(),
                    "wuji2_display_xyz": wuji2_base[:3, 3].round(6).tolist(),
                }
            )
        manifest["cases"].append(case_record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_path = args.output_dir / args.export_name
    manifest_path = args.output_dir / "leap_to_wuji2_retarget_examples_manifest.json"
    scene.export(export_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EXPORT] {export_path}")
    print(f"[MANIFEST] {manifest_path}")
    print(f"[CASES] {len(case_paths)}")
    if args.show:
        scene.show()


if __name__ == "__main__":
    main()
