"""Pure-Python geometry and I/O shared by the Isaac and Trimesh stages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .project import PROJECT_ROOT, project_path


ADAPTER_ROOT = PROJECT_ROOT


def _merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    resolved = path.resolve()
    if resolved in stack:
        raise RuntimeError(f"Config inheritance cycle: {stack + (resolved,)}")
    cfg = json.loads(resolved.read_text(encoding="utf-8"))
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _merge_config(
        _load_config_mapping(parent_path, stack + (resolved,)), cfg
    )


def load_config(path: Path) -> dict:
    cfg = _load_config_mapping(path)
    if int(cfg.get("schema_version", 0)) != 1:
        raise ValueError("Only adapter config schema_version=1 is supported")
    path_keys = {
        "output_root",
        "source_mesh_root",
        "single_object_output_root",
        "official_dexgraspnet2_root",
        "official_camera_reference_dir",
    }
    for key in path_keys & set(cfg.get("paths", {})):
        cfg["paths"][key] = str(project_path(cfg["paths"][key]))
    return cfg


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_manifest_ancestor_with_key(manifest: dict, key: str) -> dict:
    """Follow input_stage_manifest links until a manifest owns ``key``."""
    current = manifest
    visited: set[Path] = set()
    while key not in current:
        source = current.get("input_stage_manifest")
        if source is None:
            raise KeyError(f"No manifest in the input chain defines {key!r}")
        path = project_path(source)
        if path in visited:
            raise RuntimeError(f"Manifest input cycle detected at {path}")
        visited.add(path)
        current = json.loads(path.read_text(encoding="utf-8"))
    return current


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_obj_groups(path: Path) -> tuple[list[tuple[float, float, float]], list[list[tuple[int, int, int]]]]:
    vertices: list[tuple[float, float, float]] = []
    groups: list[list[tuple[int, int, int]]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("v "):
            values = stripped.split()
            vertices.append(tuple(float(value) for value in values[1:4]))
        elif stripped.startswith(("o ", "g ")):
            if faces:
                groups.append(faces)
                faces = []
        elif stripped.startswith("f "):
            indices = [int(token.split("/")[0]) - 1 for token in stripped.split()[1:]]
            for index in range(1, len(indices) - 1):
                faces.append((indices[0], indices[index], indices[index + 1]))
    if faces:
        groups.append(faces)
    if not vertices or not groups:
        raise ValueError(f"OBJ has no usable vertices/faces: {path}")
    return vertices, groups


def prepare_centered_object_asset(source_obj: Path, output_dir: Path, scale: float, code: str) -> dict:
    """Create a centered, scaled, multi-link convex URDF without touching source assets."""

    vertices, groups = parse_obj_groups(source_obj)
    array = np.asarray(vertices, dtype=np.float64)
    native_min = array.min(axis=0)
    native_max = array.max(axis=0)
    native_center = 0.5 * (native_min + native_max)
    centered = (array - native_center) * float(scale)
    aabb_min = centered.min(axis=0)
    aabb_max = centered.max(axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    pieces_dir = output_dir / "pieces"
    pieces_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "centered_combined.obj"
    combined_lines = ["o centered_combined"]
    combined_lines.extend("v {:.9g} {:.9g} {:.9g}".format(*vertex) for vertex in centered)
    combined_lines.extend(
        "f {} {} {}".format(*(index + 1 for index in face))
        for piece_faces in groups
        for face in piece_faces
    )
    combined_text = "\n".join(combined_lines) + "\n"
    if (
        not combined_path.is_file()
        or combined_path.read_text(encoding="utf-8") != combined_text
    ):
        with combined_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(combined_text)
    piece_paths: list[Path] = []
    for piece_index, piece_faces in enumerate(groups):
        used = sorted({index for face in piece_faces for index in face})
        remap = {old: new + 1 for new, old in enumerate(used)}
        piece_path = pieces_dir / f"piece_{piece_index:03d}.obj"
        lines = [f"o piece_{piece_index:03d}"]
        lines.extend("v {:.9g} {:.9g} {:.9g}".format(*centered[index]) for index in used)
        lines.extend(
            "f {} {} {}".format(*(remap[index] for index in face))
            for face in piece_faces
        )
        text = "\n".join(lines) + "\n"
        if not piece_path.is_file() or piece_path.read_text(encoding="utf-8") != text:
            with piece_path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        piece_paths.append(piece_path)
    safe_code = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in code)
    urdf_path = output_dir / f"{safe_code}_centered_coacd.urdf"
    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="{safe_code}">',
        '  <link name="object_root"/>',
    ]
    for piece_index, piece_path in enumerate(piece_paths):
        relative = piece_path.relative_to(output_dir).as_posix()
        link = f"convex_piece_{piece_index:03d}"
        lines.extend(
            [
                f'  <link name="{link}">',
                f'    <visual><geometry><mesh filename="{relative}"/></geometry></visual>',
                f'    <collision><geometry><mesh filename="{relative}"/></geometry></collision>',
                "  </link>",
                f'  <joint name="piece_joint_{piece_index:03d}" type="fixed">',
                '    <parent link="object_root"/>',
                f'    <child link="{link}"/>',
                '    <origin xyz="0 0 0" rpy="0 0 0"/>',
                "  </joint>",
            ]
        )
    lines.append("</robot>")
    urdf_text = "\n".join(lines) + "\n"
    if not urdf_path.is_file() or urdf_path.read_text(encoding="utf-8") != urdf_text:
        with urdf_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(urdf_text)
    metadata = {
        "object_code": code,
        "source_obj": str(source_obj.resolve()),
        "source_obj_sha256": sha256(source_obj),
        "scale": float(scale),
        "native_aabb_center": native_center.tolist(),
        "aabb_min_m": aabb_min.tolist(),
        "aabb_max_m": aabb_max.tolist(),
        "extent_m": (aabb_max - aabb_min).tolist(),
        "piece_count": len(piece_paths),
        "centered_combined_obj": str(combined_path.resolve()),
        "urdf": str(urdf_path.resolve()),
        "root_frame": "centered source-mesh AABB; source axes retained",
    }
    write_json_atomic(output_dir / "asset_manifest.json", metadata)
    return metadata


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = [(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s]
        elif axis == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = [(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = [(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s]
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def matrix_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transformed_aabb_corners(aabb_min: np.ndarray, aabb_max: np.ndarray, position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    corners = np.asarray(
        [[x, y, z] for x in (aabb_min[0], aabb_max[0]) for y in (aabb_min[1], aabb_max[1]) for z in (aabb_min[2], aabb_max[2])],
        dtype=np.float64,
    )
    return corners @ matrix_from_quat_wxyz(quat_wxyz).T + np.asarray(position, dtype=np.float64)


def quaternion_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) / np.linalg.norm(left)
    right = np.asarray(right, dtype=np.float64) / np.linalg.norm(right)
    dot = min(1.0, abs(float(np.dot(left, right))))
    return math.degrees(2.0 * math.acos(dot))


def backproject_opencv(depth_m: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Return dense HxWx3 points in OpenCV optical camera coordinates."""

    depth = np.asarray(depth_m, dtype=np.float32)
    # Replicator encodes rays that miss the scene as +inf.  Keep these pixels
    # invalid without emitting avoidable ``inf * 0`` NumPy warnings.
    depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, np.nan)
    height, width = depth.shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    with np.errstate(invalid="ignore"):
        x = (columns - cx) * depth / fx
        y = (rows - cy) * depth / fy
    return np.stack((x, y, depth), axis=-1)


def semantic_edges(segmentation: np.ndarray) -> np.ndarray:
    seg = np.asarray(segmentation)
    edge = np.zeros(seg.shape, dtype=np.uint8)
    horizontal = seg[:, 1:] != seg[:, :-1]
    vertical = seg[1:, :] != seg[:-1, :]
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return edge


def sample_network_view(
    depth_m: np.ndarray,
    segmentation: np.ndarray,
    intrinsic: np.ndarray,
    world_from_camera: np.ndarray,
    workspace: dict,
    point_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    dense_camera = backproject_opencv(depth_m, intrinsic)
    flat_camera = dense_camera.reshape(-1, 3)
    rotation = world_from_camera[:3, :3]
    translation = world_from_camera[:3, 3]
    flat_world = flat_camera @ rotation.T + translation
    valid = np.isfinite(flat_camera).all(axis=1) & (flat_camera[:, 2] > 0.0)
    mode = workspace.get("mode", "fixed_bounds")
    if mode == "official_visible_foreground_bbox":
        flat_segmentation = np.asarray(segmentation).reshape(-1)
        foreground = valid & (flat_segmentation > 0)
        if not np.any(foreground):
            raise RuntimeError("Camera view has no finite visible object pixels")
        lower = flat_world[foreground].min(axis=0) - float(workspace["outlier_m"])
        upper = flat_world[foreground].max(axis=0) + float(workspace["outlier_m"])
        valid &= np.all(flat_world > lower, axis=1) & np.all(flat_world < upper, axis=1)
    elif mode == "fixed_bounds":
        lower = np.empty(3, dtype=np.float64)
        upper = np.empty(3, dtype=np.float64)
        for axis, name in enumerate(("x_m", "y_m", "z_m")):
            lower[axis], upper[axis] = (float(value) for value in workspace[name])
            valid &= (flat_world[:, axis] >= lower[axis]) & (flat_world[:, axis] <= upper[axis])
    else:
        raise ValueError(f"Unknown workspace mode: {mode}")
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        raise RuntimeError("Camera produced no finite points inside the configured workspace")
    selected = rng.choice(candidates, size=int(point_count), replace=True)
    edge_image = semantic_edges(segmentation)
    return {
        "pc": flat_camera[selected].astype(np.float32),
        "seg": segmentation.reshape(-1)[selected].astype(np.int64),
        "edge": edge_image.reshape(-1)[selected].astype(np.int64),
        "pixel_indices": selected.astype(np.int64),
        "valid_pixel_count": np.asarray(len(candidates), dtype=np.int64),
        "workspace_bounds_world_m": np.stack((lower, upper)).astype(np.float64),
    }


def validate_network_input(path: Path, views: int, points: int) -> dict:
    with np.load(path) as payload:
        expected = {
            "pc": (views, points, 3),
            "seg": (views, points),
            "edge": (views, points),
            "extrinsics": (views, 4, 4),
        }
        actual = {name: tuple(payload[name].shape) for name in expected}
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name}: expected {shape}, got {actual[name]}")
        if not np.isfinite(payload["pc"]).all() or not np.isfinite(payload["extrinsics"]).all():
            raise ValueError("network input contains NaN/Inf")
        if not np.array_equal(payload["seg"] == 0, payload["seg"] < 1):
            raise ValueError("segmentation IDs are malformed")
    return actual
