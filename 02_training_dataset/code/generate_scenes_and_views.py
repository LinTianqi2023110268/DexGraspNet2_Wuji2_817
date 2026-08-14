"""Generate stable tabletop scenes and DexGraspNet2-compatible camera inputs.

This is adapter code, not an edit of the official DexGraspNet2 repository.
It must run in the ``wuji2_factory`` environment with Isaac Sim 5.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ADAPTER_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import (  # noqa: E402
    load_config,
    matrix_from_quat_wxyz,
    prepare_centered_object_asset,
    quat_wxyz_from_matrix,
    quaternion_angle_deg,
    sample_network_view,
    transformed_aabb_corners,
    validate_network_input,
    write_json_atomic,
)

from isaaclab.app import AppLauncher  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ADAPTER_ROOT
        / "config"
        / "wuji2_train60_100seminal_256view_v1.json",
    )
    parser.add_argument("--scene-count", type=int, default=None)
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument(
        "--repair-scene",
        type=int,
        action="append",
        help=(
            "Recapture only the requested completed scene(s) from their audited "
            "scene_manifest.json poses. This never reruns settling or changes poses."
        ),
    )
    parser.add_argument(
        "--repair-view",
        type=int,
        action="append",
        help="View index to recapture in repair mode; may be repeated.",
    )
    parser.add_argument(
        "--candidate-start",
        type=int,
        default=0,
        help="Skip this many candidates after applying the configured order.",
    )
    parser.add_argument(
        "--candidate-order",
        choices=("ascending", "descending"),
        default=None,
        help=(
            "Candidate evaluation order; default reads "
            "stable_pose_scene_generation.candidate_evaluation_order."
        ),
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Inspect at most this many candidates after --candidate-start.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate one scene/one view at 320x180, while preserving the physics acceptance test.",
    )
    parser.add_argument("--hold", action="store_true", help="Keep the GUI open after generation.")
    AppLauncher.add_app_launcher_args(parser)
    isaac_root = Path(os.environ.get("ISAAC_PATH", "/home/lin/isaacsim"))
    experience = isaac_root / "apps" / "isaacsim.exp.base.python.kit"
    if not experience.is_file():
        raise FileNotFoundError(f"Isaac Sim base experience not found: {experience}")
    parser.set_defaults(experience=str(experience), enable_cameras=True)
    return parser


ARGS = build_parser().parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import carb  # noqa: E402

import isaacsim.core.utils.prims as prim_utils  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

# We intentionally launch Isaac Sim's minimal base Python experience instead
# of Lab's task/RL experience.  The latter normally authors this setting in
# its .kit file; reproduce that one camera-specific setting explicitly.
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)


SCRIPT_REVISION = "wuji2-dgn2-candidate-to-sim50-camera-v7"
OBJECT_COLORS = (
    (0.95, 0.45, 0.10),
    (0.20, 0.55, 0.95),
    (0.25, 0.75, 0.35),
    (0.75, 0.25, 0.75),
    (0.90, 0.78, 0.12),
    (0.10, 0.78, 0.72),
)


class AdapterSimulationContext(SimulationContext):
    """Avoid the Isaac-Lab render preset that is incompatible with the base Kit file."""

    def _apply_render_settings_from_cfg(self) -> None:
        return


def prepare_assets(config: dict) -> list[dict]:
    source_root = Path(config["paths"]["source_mesh_root"])
    single_object_root = Path(config["paths"]["single_object_output_root"]) / "objects"
    output_root = Path(config["paths"]["output_root"]) / "prepared_assets"
    assets = []
    for item in config["objects"]:
        source_obj = source_root / item["code"] / "coacd" / "decomposed.obj"
        if not source_obj.is_file():
            raise FileNotFoundError(f"Missing COACD object: {source_obj}")
        single_manifest_path = single_object_root / item["code"] / "manifest.json"
        if not single_manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing Wuji2 1.0 output manifest: {single_manifest_path}"
            )
        single_manifest = json.loads(single_manifest_path.read_text(encoding="utf-8"))
        scale_1p0 = float(single_manifest["object_scale"])
        configured_scale = float(item["scale"])
        if not math.isclose(scale_1p0, configured_scale, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(
                f"Scale mismatch for {item['code']}: DexGraspNet1.0 output="
                f"{scale_1p0}, adapter config={configured_scale}"
            )
        asset = prepare_centered_object_asset(
            source_obj,
            output_root / f"object_{int(item['id']):03d}",
            scale_1p0,
            str(item["code"]),
        )
        if asset["source_obj_sha256"] != single_manifest["source_mesh_sha256"]:
            raise RuntimeError(
                f"Source mesh hash mismatch for {item['code']}: the scene mesh is not "
                "the mesh used by the Wuji2 DexGraspNet1.0 output"
            )
        asset["scale_contract"] = {
            "verified": True,
            "dexgraspnet1_output_manifest": str(single_manifest_path.resolve()),
            "object_scale": scale_1p0,
            "source_mesh_sha256": asset["source_obj_sha256"],
        }
        assets.append(asset)
    return assets


def make_simulation_config(config: dict) -> sim_utils.SimulationCfg:
    physics = config["physics"]
    return sim_utils.SimulationCfg(
        device=ARGS.device,
        dt=float(physics["dt_s"]),
        render_interval=1,
        gravity=tuple(float(value) for value in physics["gravity_world_m_s2"]),
        use_fabric=True,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=int(physics["position_iterations"]),
            max_position_iteration_count=int(physics["position_iterations"]),
            min_velocity_iteration_count=int(physics["velocity_iterations"]),
            max_velocity_iteration_count=int(physics["velocity_iterations"]),
            enable_ccd=False,
            enable_stabilization=False,
            enable_enhanced_determinism=bool(physics["enhanced_determinism"]),
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=float(physics["static_friction"]),
            dynamic_friction=float(physics["dynamic_friction"]),
            restitution=float(physics["restitution"]),
            friction_combine_mode="average",
        ),
    )


def _framing_box_corners(config: dict) -> np.ndarray:
    size = np.asarray(config["table"]["size_m"], dtype=np.float64)
    top = float(config["table"]["top_z_m"])
    height = float(config["camera"]["framing_box_height_m"])
    return np.asarray(
        [
            [x, y, z]
            for x in (-0.5 * size[0], 0.5 * size[0])
            for y in (-0.5 * size[1], 0.5 * size[1])
            for z in (top, top + height)
        ],
        dtype=np.float64,
    )


def _project_world_points(
    points_world: np.ndarray, world_from_camera: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = (
        np.asarray(points_world) - world_from_camera[:3, 3]
    ) @ world_from_camera[:3, :3]
    pixels = np.column_stack(
        (
            intrinsic[0, 0] * points_camera[:, 0] / points_camera[:, 2] + intrinsic[0, 2],
            intrinsic[1, 1] * points_camera[:, 1] / points_camera[:, 2] + intrinsic[1, 2],
        )
    )
    return pixels, points_camera[:, 2]


def _fit_camera_translation(
    rotation: np.ndarray,
    intrinsic: np.ndarray,
    config: dict,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict]:
    camera_cfg = config["camera"]
    target = np.asarray(camera_cfg["target_world_m"], dtype=np.float64)
    corners = _framing_box_corners(config)
    margin = float(camera_cfg["framing_margin_fraction"])
    lower_pixel = np.asarray((margin * width, margin * height), dtype=np.float64)
    upper_pixel = np.asarray(((1.0 - margin) * width, (1.0 - margin) * height))

    def candidate(distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = target - distance * rotation[:, 2]
        pixels, depths = _project_world_points(corners, pose, intrinsic)
        fits = bool(
            np.all(depths > float(camera_cfg["near_m"]))
            and np.all(depths < float(camera_cfg["far_m"]))
            and np.all(pixels >= lower_pixel)
            and np.all(pixels <= upper_pixel)
        )
        return pose, pixels, depths, fits

    low = float(camera_cfg["near_m"])
    high = 0.25
    while high < float(camera_cfg["far_m"]) and not candidate(high)[3]:
        high *= 1.5
    if high >= float(camera_cfg["far_m"]):
        high = float(camera_cfg["far_m"]) * 0.999
    if not candidate(high)[3]:
        raise RuntimeError("The official camera orientation cannot frame the configured table box")
    for _ in range(64):
        middle = 0.5 * (low + high)
        if candidate(middle)[3]:
            high = middle
        else:
            low = middle
    pose, pixels, depths, _ = candidate(high)
    return pose, {
        "camera_distance_to_target_m": float(high),
        "pixel_bounds_uv": [pixels.min(axis=0).tolist(), pixels.max(axis=0).tolist()],
        "depth_bounds_m": [float(depths.min()), float(depths.max())],
        "framing_margin_fraction": margin,
        "framing_box_world_m": [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()],
    }


def load_reference_cameras(
    config: dict, smoke_test: bool
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict]]:
    camera_cfg = config["camera"]
    reference_dir = Path(config["paths"]["official_camera_reference_dir"])
    intrinsic = np.load(reference_dir / "camK.npy").astype(np.float64)
    camera_poses = np.load(reference_dir / "camera_poses.npy").astype(np.float64)
    cam0_wrt_table = np.load(reference_dir / "cam0_wrt_table.npy").astype(np.float64)
    indices = [int(value) for value in camera_cfg["reference_view_indices"]]
    requested = 1 if smoke_test else (len(indices) if ARGS.views is None else int(ARGS.views))
    if requested <= 0 or requested > len(indices):
        raise ValueError(f"--views must be within 1..{len(indices)}")
    indices = indices[:requested]
    if smoke_test:
        scale_x = 320.0 / float(camera_cfg["width"])
        scale_y = 180.0 / float(camera_cfg["height"])
        intrinsic[0, :] *= scale_x
        intrinsic[1, :] *= scale_y
        intrinsic[2, 2] = 1.0
    width = 320 if smoke_test else int(camera_cfg["width"])
    height = 180 if smoke_test else int(camera_cfg["height"])
    # Omniverse's pinhole camera supports neither non-square pixels nor
    # principal-point offsets.  Isaac Lab therefore averages fx/fy and uses
    # the image centre internally.  Apply that documented conversion before
    # fitting camera distance so the requested framing margin is true for the
    # rendered camera, not only for the unavailable raw RealSense model.
    focal = 0.5 * (intrinsic[0, 0] + intrinsic[1, 1])
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[0, 2] = 0.5 * width
    intrinsic[1, 2] = 0.5 * height
    extrinsics = []
    framing = []
    for index in indices:
        official_pose = cam0_wrt_table @ camera_poses[index]
        fitted_pose, record = _fit_camera_translation(
            official_pose[:3, :3], intrinsic, config, width, height
        )
        record["reference_view_index"] = index
        record["official_position_table_m"] = official_pose[:3, 3].tolist()
        record["fitted_position_world_m"] = fitted_pose[:3, 3].tolist()
        extrinsics.append(fitted_pose)
        framing.append(record)
    return intrinsic, np.stack(extrinsics), indices, framing


def flatten_editable_usd(source: Path, destination: Path) -> Path:
    """Remove converter instances so contact/rest offsets can be authored."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return destination
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Could not open converted USD: {source}")
    for _ in range(8):
        instances = [prim for prim in stage.TraverseAll() if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    remaining = [str(prim.GetPath()) for prim in stage.TraverseAll() if prim.IsInstance()]
    if remaining:
        raise RuntimeError(f"Could not remove USD instances: {remaining[:8]}")
    flattened = stage.Flatten()
    if not flattened.Export(str(destination)):
        raise RuntimeError(f"Could not export flattened USD: {destination}")
    return destination


def make_object_spawn_cfg(config: dict, asset: dict, object_index: int) -> sim_utils.UsdFileCfg:
    physics = config["physics"]
    cache = Path(config["paths"]["output_root"]) / "usd_cache" / f"object_{object_index:03d}"
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        solver_position_iteration_count=int(physics["position_iterations"]),
        solver_velocity_iteration_count=int(physics["velocity_iterations"]),
        max_depenetration_velocity=5.0,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=float(physics["contact_offset_m"]),
        rest_offset=float(physics["rest_offset_m"]),
    )
    converter_cfg = sim_utils.UrdfFileCfg(
        asset_path=asset["urdf"],
        usd_dir=str(cache),
        usd_file_name=f"object_{object_index:03d}.usd",
        force_usd_conversion=False,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        collider_type="convex_hull",
        joint_drive=None,
        rigid_props=rigid_props,
        collision_props=collision_props,
        mass_props=sim_utils.MassPropertiesCfg(mass=float(physics["object_mass_kg"])),
    )
    converter = sim_utils.UrdfConverter(converter_cfg)
    editable_usd = flatten_editable_usd(
        Path(converter.usd_path), cache / "flat" / f"object_{object_index:03d}_editable.usd"
    )
    return sim_utils.UsdFileCfg(
        usd_path=str(editable_usd),
        rigid_props=rigid_props,
        collision_props=collision_props,
        mass_props=sim_utils.MassPropertiesCfg(mass=float(physics["object_mass_kg"])),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=OBJECT_COLORS[object_index % len(OBJECT_COLORS)],
            roughness=0.65,
        ),
        semantic_tags=[("class", f"object_{object_index + 1:03d}")],
    )


def create_scene(
    sim: AdapterSimulationContext,
    config: dict,
    assets: list[dict],
    intrinsic: np.ndarray,
) -> tuple[RigidPrim, Camera]:
    table = config["table"]
    physics = config["physics"]
    table_size = tuple(float(value) for value in table["size_m"])
    table_cfg = sim_utils.CuboidCfg(
        size=table_size,
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=float(physics["contact_offset_m"]),
            rest_offset=float(physics["rest_offset_m"]),
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.57, 0.60), roughness=0.9
        ),
        semantic_tags=[("class", "table")],
    )
    table_center_z = float(table["top_z_m"]) - 0.5 * table_size[2]
    table_cfg.func("/World/Table", table_cfg, translation=(0.0, 0.0, table_center_z))
    sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.88, 0.92)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.88, 0.92))
    )
    prim_utils.create_prim("/World/Objects", "Xform")
    for object_index, asset in enumerate(assets):
        spawn_cfg = make_object_spawn_cfg(config, asset, object_index)
        spawn_cfg.func(
            f"/World/Objects/Object_{object_index:03d}",
            spawn_cfg,
            translation=(2.0 + 0.3 * object_index, 0.0, 1.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
    rigid_paths = sorted(
        str(prim.GetPath())
        for prim in sim.stage.Traverse()
        if str(prim.GetPath()).startswith("/World/Objects/Object_")
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    if len(rigid_paths) != len(assets):
        raise RuntimeError(
            f"Expected one merged rigid body per object ({len(assets)}), found {rigid_paths}"
        )
    object_view = RigidPrim(
        prim_paths_expr="/World/Objects/Object_.*/object_root",
        name="scene_objects",
        reset_xform_properties=False,
    )

    camera_config = config["camera"]
    width = 320 if ARGS.smoke_test else int(camera_config["width"])
    height = 180 if ARGS.smoke_test else int(camera_config["height"])
    prim_utils.create_prim("/World/CameraRig", "Xform")
    color_mapping = {"class:table": (0, 0, 0, 255)}
    for index in range(len(assets)):
        color_mapping[f"class:object_{index + 1:03d}"] = (index + 1, 0, 0, 255)
    sensor_cfg = CameraCfg(
        prim_path="/World/CameraRig/Camera",
        update_period=0.0,
        width=width,
        height=height,
        data_types=["distance_to_image_plane", "semantic_segmentation"],
        semantic_filter=["class"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping=color_mapping,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic.reshape(-1).tolist(),
            width=width,
            height=height,
            clipping_range=(float(camera_config["near_m"]), float(camera_config["far_m"])),
        ),
    )
    return object_view, Camera(cfg=sensor_cfg)


def tensor_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def set_all_object_states(
    sim: AdapterSimulationContext,
    object_view: RigidPrim,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    object_view.set_world_poses(
        positions=torch.as_tensor(positions, dtype=torch.float32, device=sim.device),
        orientations=torch.as_tensor(quaternions, dtype=torch.float32, device=sim.device),
        usd=False,
    )
    object_view.set_velocities(
        torch.zeros((len(positions), 6), dtype=torch.float32, device=sim.device)
    )
    sim.forward()


def read_object_states(object_view: RigidPrim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, quaternions = object_view.get_world_poses()
    velocities = object_view.get_velocities()
    return tensor_numpy(positions), tensor_numpy(quaternions), tensor_numpy(velocities)


def wait_for_quiet(
    sim: AdapterSimulationContext,
    object_view: RigidPrim,
    active_indices: list[int],
    stability: dict,
) -> tuple[bool, float, dict]:
    dt = float(sim.get_physics_dt())
    required_steps = max(1, int(math.ceil(float(stability["continuous_quiet_time_s"]) / dt)))
    maximum_steps = max(
        1, int(math.ceil(float(stability["maximum_settling_time_s"]) / dt))
    )
    quiet_steps = 0
    peak_linear = 0.0
    peak_angular = 0.0
    best_max_linear = math.inf
    best_max_angular = math.inf
    last_linear = np.full(len(active_indices), math.inf, dtype=np.float64)
    last_angular = np.full(len(active_indices), math.inf, dtype=np.float64)
    for step in range(maximum_steps):
        sim.step(render=False)
        _, _, velocities = read_object_states(object_view)
        selected = velocities[active_indices]
        linear = np.linalg.norm(selected[:, :3], axis=1)
        angular = np.linalg.norm(selected[:, 3:], axis=1)
        last_linear = linear
        last_angular = angular
        peak_linear = max(peak_linear, float(linear.max()))
        peak_angular = max(peak_angular, float(angular.max()))
        best_max_linear = min(best_max_linear, float(linear.max()))
        best_max_angular = min(best_max_angular, float(angular.max()))
        if (
            np.all(linear <= float(stability["linear_speed_threshold_m_s"]))
            and np.all(angular <= float(stability["angular_speed_threshold_rad_s"]))
        ):
            quiet_steps += 1
            if quiet_steps >= required_steps:
                return True, (step + 1) * dt, {
                    "peak_linear_speed_m_s": peak_linear,
                    "peak_angular_speed_rad_s": peak_angular,
                    "best_max_linear_speed_m_s": best_max_linear,
                    "best_max_angular_speed_rad_s": best_max_angular,
                    "last_linear_speeds_m_s": last_linear.tolist(),
                    "last_angular_speeds_rad_s": last_angular.tolist(),
                }
        else:
            quiet_steps = 0
    return False, maximum_steps * dt, {
        "peak_linear_speed_m_s": peak_linear,
        "peak_angular_speed_rad_s": peak_angular,
        "best_max_linear_speed_m_s": best_max_linear,
        "best_max_angular_speed_rad_s": best_max_angular,
        "last_linear_speeds_m_s": last_linear.tolist(),
        "last_angular_speeds_rad_s": last_angular.tolist(),
    }


def scene_geometry_check(
    positions: np.ndarray,
    quaternions: np.ndarray,
    assets: list[dict],
    config: dict,
) -> tuple[bool, list[dict]]:
    table_size = np.asarray(config["table"]["size_m"], dtype=np.float64)
    stability = config["stability"]
    tolerance = float(stability["table_xy_tolerance_m"])
    table_top = float(config["table"]["top_z_m"])
    support_tolerance = float(stability["support_height_tolerance_m"])
    penetration_tolerance = float(
        stability.get("maximum_mesh_table_penetration_m", support_tolerance)
    )
    results = []
    valid = True
    for index, asset in enumerate(assets):
        corners = transformed_aabb_corners(
            np.asarray(asset["aabb_min_m"]),
            np.asarray(asset["aabb_max_m"]),
            positions[index],
            quaternions[index],
        )
        vertices = []
        with Path(asset["centered_combined_obj"]).open(
            "r", encoding="utf-8", errors="replace"
        ) as stream:
            for line in stream:
                if line.startswith("v "):
                    fields = line.split()
                    vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        if not vertices:
            raise RuntimeError(f"No vertices in {asset['centered_combined_obj']}")
        exact_vertices = (
            np.asarray(vertices, dtype=np.float64)
            @ matrix_from_quat_wxyz(quaternions[index]).T
            + np.asarray(positions[index], dtype=np.float64)
        )
        exact_min = exact_vertices.min(axis=0)
        exact_max = exact_vertices.max(axis=0)
        within_table_xy = bool(
            exact_min[0] >= -0.5 * table_size[0] - tolerance
            and exact_max[0] <= 0.5 * table_size[0] + tolerance
            and exact_min[1] >= -0.5 * table_size[1] - tolerance
            and exact_max[1] <= 0.5 * table_size[1] + tolerance
        )
        # Use every prepared mesh vertex for the hard support check.  The
        # transformed local AABB is retained only as a conservative diagnostic:
        # after rotation its empty corners can extend well beyond the mesh.
        table_supported = bool(
            exact_min[2] <= table_top + support_tolerance
            and exact_min[2] >= table_top - penetration_tolerance
            and exact_max[2] >= table_top
            and positions[index, 2] > table_top
        )
        object_valid = within_table_xy and table_supported
        valid &= object_valid
        results.append(
            {
                "object_index": index,
                "aabb_world_min_m": corners.min(axis=0).tolist(),
                "aabb_world_max_m": corners.max(axis=0).tolist(),
                "mesh_world_min_m": exact_min.tolist(),
                "mesh_world_max_m": exact_max.tolist(),
                "within_table_xy": within_table_xy,
                "table_supported": table_supported,
                "on_table": object_valid,
            }
        )
    return bool(valid), results


def evaluate_candidate(
    sim: AdapterSimulationContext,
    object_view: RigidPrim,
    assets: list[dict],
    config: dict,
    scene_index: int,
    candidate_path: Path,
) -> dict | None:
    stability = config["stability"]
    dt = float(sim.get_physics_dt())
    pool_count = len(assets)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    active_indices = [int(value) for value in candidate["active_object_indices"]]
    if len(active_indices) != int(config["scope"]["objects_per_scene"]):
        raise RuntimeError(f"Wrong active-object count in {candidate_path}")
    if active_indices != sorted(set(active_indices)) or any(
        index < 0 or index >= pool_count for index in active_indices
    ):
        raise RuntimeError(f"Invalid active_object_indices in {candidate_path}")
    candidate_codes = [item["code"] for item in candidate["objects"]]
    configured_codes = [config["objects"][index]["code"] for index in active_indices]
    if candidate_codes != configured_codes:
        raise RuntimeError(f"Object order mismatch in {candidate_path}")
    active_initial_positions = np.asarray(
        candidate["positions_world_m"], dtype=np.float32
    )
    active_initial_quaternions = np.asarray(
        candidate["quaternions_world_wxyz"], dtype=np.float32
    )
    active_count = len(active_indices)
    if active_initial_positions.shape != (active_count, 3) or active_initial_quaternions.shape != (
        active_count,
        4,
    ):
        raise RuntimeError(f"Invalid pose tensor shapes in {candidate_path}")
    # Isaac keeps one rigid actor for every member of the 11-object pool.  Only
    # the six selected actors are placed on the table; inactive actors start far
    # outside the camera/table workspace and are excluded from every metric.
    initial_positions = np.asarray(
        [[3.0 + 0.30 * index, 0.0, 1.0] for index in range(pool_count)],
        dtype=np.float32,
    )
    initial_quaternions = np.zeros((pool_count, 4), dtype=np.float32)
    initial_quaternions[:, 0] = 1.0
    initial_positions[active_indices] = active_initial_positions
    initial_quaternions[active_indices] = active_initial_quaternions
    object_view.enable_gravities()
    set_all_object_states(sim, object_view, initial_positions, initial_quaternions)
    quiet, elapsed, speed_info = wait_for_quiet(
        sim,
        object_view,
        active_indices,
        stability,
    )
    require_velocity_quiet = bool(
        stability.get("require_continuous_velocity_quiet", True)
    )
    if not quiet and require_velocity_quiet:
        print(
            f"[CANDIDATE {candidate['candidate_index']:04d}] rejected: "
            "did not settle in Isaac Sim; "
            f"best_max_v={speed_info['best_max_linear_speed_m_s']:.5f}m/s "
            f"best_max_w={speed_info['best_max_angular_speed_rad_s']:.5f}rad/s "
            f"last_v={np.round(speed_info['last_linear_speeds_m_s'], 5).tolist()} "
            f"last_w={np.round(speed_info['last_angular_speeds_rad_s'], 5).tolist()}",
            flush=True,
        )
        return None
    before_pos_all, before_quat_all, _ = read_object_states(object_view)
    before_pos = before_pos_all[active_indices]
    before_quat = before_quat_all[active_indices]
    observation_steps = max(
        1,
        int(math.ceil(float(stability["final_observation_time_s"]) / dt)),
    )
    for _ in range(observation_steps):
        sim.step(render=False)
    after_pos_all, after_quat_all, after_vel_all = read_object_states(object_view)
    after_pos = after_pos_all[active_indices]
    after_quat = after_quat_all[active_indices]
    after_vel = after_vel_all[active_indices]
    drift_m = np.linalg.norm(after_pos - before_pos, axis=1)
    drift_deg = np.asarray(
        [
            quaternion_angle_deg(before_quat[i], after_quat[i])
            for i in range(active_count)
        ]
    )
    adjustment_m = np.linalg.norm(after_pos - active_initial_positions, axis=1)
    adjustment_deg = np.asarray(
        [
            quaternion_angle_deg(active_initial_quaternions[i], after_quat[i])
            for i in range(active_count)
        ]
    )
    geometry_valid, geometry = scene_geometry_check(
        after_pos,
        after_quat,
        [assets[index] for index in active_indices],
        config,
    )
    stable_valid = bool(
        np.all(drift_m <= float(stability["maximum_final_translation_drift_m"]))
        and np.all(
            drift_deg <= float(stability["maximum_final_rotation_drift_deg"])
        )
    )
    if not geometry_valid or not stable_valid:
        print(
            f"[CANDIDATE {candidate['candidate_index']:04d}] rejected: "
            f"geometry={geometry_valid} stable={stable_valid} "
            f"max_drift_mm={1000.0 * drift_m.max():.2f}",
            flush=True,
        )
        return None
    object_view.set_velocities(
        torch.zeros((pool_count, 6), dtype=torch.float32, device=sim.device)
    )
    sim.forward()
    candidate_type = str(candidate["candidate_type"])
    print(
        f"[SCENE {scene_index:04d}] accepted {candidate_type} candidate "
        f"{candidate['candidate_index']:04d}",
        flush=True,
    )
    return {
        "pipeline": f"{candidate_type} candidate -> Isaac Sim 5.0 stability filter",
        "candidate_type": candidate_type,
        "candidate_source": str(candidate_path.resolve()),
        "candidate_index": int(candidate["candidate_index"]),
        "candidate_global_attempt": int(candidate["global_attempt"]),
        "candidate_generation_config": candidate["generation_config"],
        "candidate_layout_sampling": candidate.get("layout_sampling", {}),
        "active_object_indices": active_indices,
        "active_object_ids": [int(config["objects"][i]["id"]) for i in active_indices],
        "active_object_codes": configured_codes,
        "candidate_initial_records": candidate["initial_records"],
        "candidate_positions_world_m": active_initial_positions.tolist(),
        "candidate_quaternions_world_wxyz": active_initial_quaternions.tolist(),
        "isaac_settling_elapsed_s": elapsed,
        "velocity_quiet_reached": bool(quiet),
        "velocity_quiet_required": require_velocity_quiet,
        **speed_info,
        "positions_world_m": after_pos.tolist(),
        "quaternions_world_wxyz": after_quat.tolist(),
        "final_velocities_world": after_vel.tolist(),
        "post_quiet_translation_drift_m": drift_m.tolist(),
        "post_quiet_rotation_drift_deg": drift_deg.tolist(),
        "candidate_to_isaac_translation_adjustment_m": adjustment_m.tolist(),
        "candidate_to_isaac_rotation_adjustment_deg": adjustment_deg.tolist(),
        "geometry": geometry,
    }


def decode_semantic_rgba(rgba: np.ndarray, object_count: int, finite_depth: np.ndarray) -> np.ndarray:
    image = np.asarray(rgba, dtype=np.uint8)
    segmentation = np.zeros(image.shape[:2], dtype=np.int16)
    recognized = np.all(image == np.asarray((0, 0, 0, 255), dtype=np.uint8), axis=-1)
    for object_index in range(object_count):
        color = np.asarray((object_index + 1, 0, 0, 255), dtype=np.uint8)
        mask = np.all(image == color, axis=-1)
        segmentation[mask] = object_index + 1
        recognized |= mask
    unknown_valid = finite_depth & ~recognized
    if np.any(unknown_valid):
        colors, counts = np.unique(image[unknown_valid].reshape(-1, 4), axis=0, return_counts=True)
        summary = sorted(zip(counts.tolist(), colors.tolist()), reverse=True)[:8]
        raise RuntimeError(f"Unknown semantic colors on finite-depth pixels: {summary}")
    return segmentation


def capture_scene_views(
    sim: AdapterSimulationContext,
    camera: Camera,
    config: dict,
    scene_index: int,
    extrinsics: np.ndarray,
    reference_indices: list[int],
    framing_records: list[dict],
) -> dict:
    scene_dir = Path(config["paths"]["output_root"]) / "scenes" / f"scene_{scene_index:04d}"
    camera_dir = scene_dir / "camera"
    camera_dir.mkdir(parents=True, exist_ok=True)
    point_count = int(config["scope"]["points_per_view"])
    pc_all, seg_all, edge_all, pixels_all = [], [], [], []
    valid_counts = []
    actual_intrinsics = []
    workspace_bounds = []
    for local_view, world_from_camera in enumerate(extrinsics):
        position = torch.as_tensor(
            world_from_camera[:3, 3][None], dtype=torch.float32, device=sim.device
        )
        quaternion = torch.as_tensor(
            quat_wxyz_from_matrix(world_from_camera[:3, :3])[None],
            dtype=torch.float32,
            device=sim.device,
        )
        camera.set_world_poses(position, quaternion, convention="ros")
        # A newly-created Replicator annotator allocates its output buffers on
        # its first update.  After moving the camera, give RTX rendering several
        # complete render frames before reading depth and semantics.  Do not use
        # ``sim.step(render=True)`` here: on the minimal Isaac Sim 5.0 experience
        # it can block while synchronizing the stopped physics timeline.
        for _ in range(4):
            sim.render()
            camera.update(float(sim.get_physics_dt()), force_recompute=True)

        actual_position = tensor_numpy(camera.data.pos_w)[0].astype(np.float64)
        actual_quaternion = tensor_numpy(camera.data.quat_w_ros)[0].astype(np.float64)
        actual_rotation = matrix_from_quat_wxyz(actual_quaternion)
        position_error_m = float(
            np.linalg.norm(actual_position - world_from_camera[:3, 3])
        )
        rotation_error_deg = quaternion_angle_deg(
            actual_quaternion,
            quat_wxyz_from_matrix(world_from_camera[:3, :3]),
        )
        if position_error_m > 1.0e-4 or rotation_error_deg > 0.05:
            raise RuntimeError(
                "Isaac camera pose does not match requested T_world_camera: "
                f"translation_error={position_error_m:.6g} m, "
                f"rotation_error={rotation_error_deg:.6g} deg"
            )
        depth = tensor_numpy(camera.data.output["distance_to_image_plane"])[0, ..., 0].astype(np.float32)
        semantic_rgba = tensor_numpy(camera.data.output["semantic_segmentation"])[0].astype(np.uint8)
        intrinsic = tensor_numpy(camera.data.intrinsic_matrices)[0].astype(np.float64)
        finite_depth = np.isfinite(depth) & (depth > 0.0)
        finite_count = int(finite_depth.sum())
        if finite_count:
            finite_values = depth[finite_depth]
            depth_summary = (
                f"finite={finite_count}/{depth.size} "
                f"range=[{float(finite_values.min()):.4f}, "
                f"{float(finite_values.max()):.4f}]m"
            )
        else:
            depth_summary = f"finite=0/{depth.size}"
        print(
            f"[CAMERA-DIAG] scene={scene_index:04d} view={local_view:04d} "
            f"pose_error={position_error_m * 1000.0:.3f}mm/"
            f"{rotation_error_deg:.4f}deg {depth_summary} "
            f"eye={actual_position.round(4).tolist()} "
            f"forward={actual_rotation[:, 2].round(4).tolist()}",
            flush=True,
        )
        if finite_count == 0:
            raise RuntimeError(
                "Camera rendered no finite depth after four warm-up frames; "
                "see [CAMERA-DIAG] for the verified pose"
            )
        segmentation = decode_semantic_rgba(
            semantic_rgba, len(config["objects"]), finite_depth
        )
        rng = np.random.default_rng(
            int(config["random_seed"]) + scene_index * 1000003 + local_view * 7919
        )
        sampled = sample_network_view(
            depth,
            segmentation,
            intrinsic,
            world_from_camera,
            config["workspace"],
            point_count,
            rng,
        )
        view_dir = camera_dir / f"view_{local_view:04d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        np.save(view_dir / "depth_m.npy", depth)
        np.save(view_dir / "segmentation.npy", segmentation)
        np.save(view_dir / "semantic_rgba.npy", semantic_rgba)
        np.save(view_dir / "sample_pixel_indices.npy", sampled["pixel_indices"])
        pc_all.append(sampled["pc"])
        seg_all.append(sampled["seg"])
        edge_all.append(sampled["edge"])
        pixels_all.append(sampled["pixel_indices"])
        valid_counts.append(int(sampled["valid_pixel_count"]))
        workspace_bounds.append(sampled["workspace_bounds_world_m"].tolist())
        actual_intrinsics.append(intrinsic)
        print(
            f"[CAMERA] scene={scene_index:04d} view={local_view:04d} "
            f"reference={reference_indices[local_view]:04d} valid={valid_counts[-1]} sampled={point_count}",
            flush=True,
        )
    np.save(camera_dir / "intrinsics.npy", np.stack(actual_intrinsics))
    np.save(camera_dir / "extrinsics_world_from_camera.npy", extrinsics)
    network_path = scene_dir / "network_input.npz"
    np.savez_compressed(
        network_path,
        pc=np.stack(pc_all).astype(np.float32),
        seg=np.stack(seg_all).astype(np.int64),
        edge=np.stack(edge_all).astype(np.int64),
        extrinsics=extrinsics.astype(np.float32),
        pixel_indices=np.stack(pixels_all).astype(np.int64),
    )
    shapes = validate_network_input(network_path, len(extrinsics), point_count)
    return {
        "network_input": str(network_path.resolve()),
        "tensor_shapes": {name: list(shape) for name, shape in shapes.items()},
        "valid_pixel_count_per_view": valid_counts,
        "reference_view_indices": reference_indices,
        "framing_records": framing_records,
        "workspace_bounds_world_m_per_view": workspace_bounds,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def repair_scene_views(
    sim: AdapterSimulationContext,
    object_view: RigidPrim,
    camera: Camera,
    config: dict,
    extrinsics: np.ndarray,
    scene_indices: list[int],
    view_indices: list[int],
) -> None:
    """Atomically replace selected camera rows without changing accepted poses."""

    output_root = Path(config["paths"]["output_root"])
    object_count = len(config["objects"])
    point_count = int(config["scope"]["points_per_view"])
    expected_view_count = len(extrinsics)
    for view_index in view_indices:
        if view_index < 0 or view_index >= expected_view_count:
            raise ValueError(
                f"repair view {view_index} is outside 0..{expected_view_count - 1}"
            )

    for scene_index in scene_indices:
        scene_dir = output_root / "scenes" / f"scene_{scene_index:04d}"
        manifest_path = scene_dir / "scene_manifest.json"
        network_path = scene_dir / "network_input.npz"
        if not manifest_path.is_file() or not network_path.is_file():
            raise FileNotFoundError(f"Completed scene is missing: {scene_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["scene_index"]) != scene_index:
            raise RuntimeError(f"Scene manifest index mismatch: {manifest_path}")

        positions = np.asarray(
            [[3.0 + 0.30 * index, 0.0, 1.0] for index in range(object_count)],
            dtype=np.float32,
        )
        quaternions = np.zeros((object_count, 4), dtype=np.float32)
        quaternions[:, 0] = 1.0
        active_ids = set()
        for record in manifest["objects"]:
            pool_index = int(record["object_pool_index"])
            segmentation_id = int(record["segmentation_id"])
            if segmentation_id != pool_index + 1:
                raise RuntimeError(
                    f"Semantic/pool index mismatch in {manifest_path}: {record}"
                )
            transform = np.asarray(record["T_world_centered_object"], dtype=np.float64)
            positions[pool_index] = transform[:3, 3]
            quaternions[pool_index] = quat_wxyz_from_matrix(transform[:3, :3])
            active_ids.add(segmentation_id)

        object_view.disable_gravities()
        set_all_object_states(sim, object_view, positions, quaternions)
        with np.load(network_path) as archive:
            network = {name: archive[name].copy() for name in archive.files}
        if network["pc"].shape != (expected_view_count, point_count, 3):
            raise RuntimeError(f"Unexpected network input shape: {network_path}")

        repair_records = []
        for view_index in view_indices:
            world_from_camera = np.asarray(extrinsics[view_index], dtype=np.float64)
            position = torch.as_tensor(
                world_from_camera[:3, 3][None], dtype=torch.float32, device=sim.device
            )
            quaternion = torch.as_tensor(
                quat_wxyz_from_matrix(world_from_camera[:3, :3])[None],
                dtype=torch.float32,
                device=sim.device,
            )
            camera.set_world_poses(position, quaternion, convention="ros")

            # Scene changes need a longer RTX/Replicator flush than ordinary
            # camera motion. Validate semantics and keep rendering if necessary.
            segmentation = None
            for render_count in (16, 32, 64):
                for _ in range(render_count if segmentation is None else render_count // 2):
                    sim.render()
                    camera.update(float(sim.get_physics_dt()), force_recompute=True)
                depth = tensor_numpy(
                    camera.data.output["distance_to_image_plane"]
                )[0, ..., 0].astype(np.float32)
                semantic_rgba = tensor_numpy(
                    camera.data.output["semantic_segmentation"]
                )[0].astype(np.uint8)
                finite_depth = np.isfinite(depth) & (depth > 0.0)
                segmentation = decode_semantic_rgba(
                    semantic_rgba, object_count, finite_depth
                )
                observed_ids = set(int(value) for value in np.unique(segmentation)) - {0}
                if observed_ids and observed_ids <= active_ids:
                    break
            else:
                raise RuntimeError(
                    f"Semantic buffers remained stale for scene={scene_index:04d} "
                    f"view={view_index:04d}: observed={sorted(observed_ids)} "
                    f"expected subset of {sorted(active_ids)}"
                )

            intrinsic = tensor_numpy(camera.data.intrinsic_matrices)[0].astype(np.float64)
            rng = np.random.default_rng(
                int(config["random_seed"]) + scene_index * 1000003 + view_index * 7919
            )
            sampled = sample_network_view(
                depth,
                segmentation,
                intrinsic,
                world_from_camera,
                config["workspace"],
                point_count,
                rng,
            )
            sampled_ids = set(int(value) for value in np.unique(sampled["seg"])) - {0}
            if not sampled_ids or not sampled_ids <= active_ids:
                raise RuntimeError(
                    f"Repaired sample has invalid IDs for scene={scene_index:04d} "
                    f"view={view_index:04d}: {sorted(sampled_ids)}"
                )

            view_dir = scene_dir / "camera" / f"view_{view_index:04d}"
            old_hashes = {
                name: _sha256(view_dir / name)
                for name in (
                    "depth_m.npy",
                    "segmentation.npy",
                    "semantic_rgba.npy",
                    "sample_pixel_indices.npy",
                )
            }
            _atomic_save_npy(view_dir / "depth_m.npy", depth)
            _atomic_save_npy(view_dir / "segmentation.npy", segmentation)
            _atomic_save_npy(view_dir / "semantic_rgba.npy", semantic_rgba)
            _atomic_save_npy(
                view_dir / "sample_pixel_indices.npy", sampled["pixel_indices"]
            )
            network["pc"][view_index] = sampled["pc"]
            network["seg"][view_index] = sampled["seg"]
            network["edge"][view_index] = sampled["edge"]
            network["extrinsics"][view_index] = world_from_camera.astype(np.float32)
            network["pixel_indices"][view_index] = sampled["pixel_indices"]
            new_hashes = {name: _sha256(view_dir / name) for name in old_hashes}
            repair_records.append(
                {
                    "view_index": view_index,
                    "observed_segmentation_ids": sorted(sampled_ids),
                    "valid_pixel_count": int(sampled["valid_pixel_count"]),
                    "old_sha256": old_hashes,
                    "new_sha256": new_hashes,
                }
            )

        _atomic_savez_compressed(network_path, **network)
        validate_network_input(network_path, expected_view_count, point_count)
        manifest.setdefault("camera_repairs", []).append(
            {
                "reason": "stale semantic/render buffers detected in completed dataset audit",
                "policy": "exact accepted poses; no physics settling; atomic selected-view replacement",
                "script_revision": SCRIPT_REVISION,
                "views": repair_records,
            }
        )
        write_json_atomic(manifest_path, manifest)
        print(
            f"[REPAIRED] scene={scene_index:04d} views={view_indices} "
            f"active_ids={sorted(active_ids)}",
            flush=True,
        )


def build_scene_manifest(config: dict, assets: list[dict], scene_index: int, physics_result: dict, camera_result: dict) -> dict:
    object_rows = []
    active_indices = [int(value) for value in physics_result["active_object_indices"]]
    for local_index, pool_index in enumerate(active_indices):
        item = config["objects"][pool_index]
        pose = np.eye(4, dtype=np.float64)
        position = np.asarray(physics_result["positions_world_m"][local_index])
        quaternion = np.asarray(
            physics_result["quaternions_world_wxyz"][local_index]
        )
        pose[:3, :3] = matrix_from_quat_wxyz(quaternion)
        pose[:3, 3] = position
        object_rows.append(
            {
                "segmentation_id": int(item["id"]),
                "object_pool_index": pool_index,
                "object_code": item["code"],
                "object_frame": "centered source-mesh AABB; same centered object frame used by Wuji2 single-object output",
                "T_world_centered_object": pose.tolist(),
                "asset": assets[pool_index],
            }
        )
    return {
        "schema_version": 1,
        "script_revision": SCRIPT_REVISION,
        "scene_index": scene_index,
        "coordinate_contract": {
            "world": "tabletop center origin, +z upward",
            "camera": config["camera"]["coordinate_contract"],
            "object": "centered source-mesh AABB; source mesh axes retained",
        },
        "objects": object_rows,
        "table": config["table"],
        "camera": config["camera"],
        "workspace": config["workspace"],
        "physics_acceptance": physics_result,
        "camera_output": camera_result,
        "source_ledger": {
            "table": config["table"]["source"],
            "candidate_sampling": config["candidate_sampling"]["source"],
            "scene_generation": config["stable_pose_scene_generation"]["source"],
            "physics": config["physics"]["source"],
            "stability": config["stability"]["source"],
            "camera_pose": config["camera"]["pose_source"],
            "camera_intrinsics": config["camera"]["intrinsics_source"],
        },
    }


def main() -> None:
    config = load_config(ARGS.config)
    assets = prepare_assets(config)
    intrinsic, all_extrinsics, reference_indices, framing_records = load_reference_cameras(
        config, ARGS.smoke_test
    )
    scene_count = (
        1
        if ARGS.smoke_test
        else int(config["scope"]["scene_count"] if ARGS.scene_count is None else ARGS.scene_count)
    )
    if scene_count <= 0:
        raise ValueError("scene count must be positive")
    sim = AdapterSimulationContext(make_simulation_config(config))
    sim.set_camera_view([0.42, 0.48, 0.42], [0.0, 0.0, 0.04])
    object_view, camera = create_scene(sim, config, assets, intrinsic)
    sim.reset()
    object_view.initialize()
    if int(object_view.count) != len(assets):
        raise RuntimeError(f"Rigid view count={object_view.count}, expected {len(assets)}")
    if ARGS.repair_scene is not None:
        repair_views = [0] if ARGS.repair_view is None else ARGS.repair_view
        repair_scene_views(
            sim,
            object_view,
            camera,
            config,
            all_extrinsics,
            sorted(set(ARGS.repair_scene)),
            sorted(set(repair_views)),
        )
        print("[COMPLETE] selected camera views repaired", flush=True)
        return
    run_manifest = {
        "schema_version": 1,
        "script_revision": SCRIPT_REVISION,
        "config": str(ARGS.config.resolve()),
        "device": str(ARGS.device),
        "headless": bool(ARGS.headless),
        "smoke_test": bool(ARGS.smoke_test),
        "scene_count": scene_count,
        "view_count": len(all_extrinsics),
        "scene_generation_pipeline": config["scene_generation_pipeline"],
        "rejected_candidates": [],
        "scene_manifests": [],
    }
    output_root = Path(config["paths"]["output_root"])
    candidate_directory = output_root / config["paths"]["candidate_directory_name"]
    candidate_paths = sorted(candidate_directory.glob("candidate_*.json"))
    candidate_paths = [
        path for path in candidate_paths if path.name != "candidate_manifest.json"
    ]
    candidate_order = (
        ARGS.candidate_order
        if ARGS.candidate_order is not None
        else str(
            config["stable_pose_scene_generation"].get(
                "candidate_evaluation_order", "ascending"
            )
        )
    )
    if candidate_order not in {"ascending", "descending"}:
        raise ValueError(
            "stable_pose_scene_generation.candidate_evaluation_order must be "
            f"'ascending' or 'descending', got {candidate_order!r}"
        )
    if candidate_order == "descending":
        candidate_paths.reverse()
    run_manifest["candidate_evaluation_order"] = candidate_order
    if ARGS.candidate_start < 0:
        raise ValueError("--candidate-start must be non-negative")
    candidate_paths = candidate_paths[ARGS.candidate_start :]
    if ARGS.candidate_limit is not None:
        if ARGS.candidate_limit <= 0:
            raise ValueError("--candidate-limit must be positive")
        candidate_paths = candidate_paths[: ARGS.candidate_limit]
    run_manifest["first_candidate"] = (
        str(candidate_paths[0].resolve()) if candidate_paths else None
    )
    run_manifest["last_candidate"] = (
        str(candidate_paths[-1].resolve()) if candidate_paths else None
    )
    if len(candidate_paths) < scene_count:
        raise RuntimeError(
            f"Need at least {scene_count} candidates under "
            f"{candidate_directory}, found {len(candidate_paths)}. "
            "Run the candidate generator configured for this experiment first."
        )
    write_json_atomic(output_root / "run_progress.json", {**run_manifest, "status": "running"})
    scene_index = 0
    for candidate_path in candidate_paths:
        if scene_index >= scene_count:
            break
        physics_result = evaluate_candidate(
            sim,
            object_view,
            assets,
            config,
            scene_index,
            candidate_path,
        )
        if physics_result is None:
            run_manifest["rejected_candidates"].append(str(candidate_path.resolve()))
            write_json_atomic(
                output_root / "run_progress.json",
                {**run_manifest, "status": "running"},
            )
            continue
        camera_result = capture_scene_views(
            sim,
            camera,
            config,
            scene_index,
            all_extrinsics,
            reference_indices,
            framing_records,
        )
        manifest = build_scene_manifest(
            config, assets, scene_index, physics_result, camera_result
        )
        manifest_path = output_root / "scenes" / f"scene_{scene_index:04d}" / "scene_manifest.json"
        write_json_atomic(manifest_path, manifest)
        run_manifest["scene_manifests"].append(str(manifest_path.resolve()))
        write_json_atomic(output_root / "run_progress.json", {**run_manifest, "status": "running"})
        scene_index += 1
    if scene_index != scene_count:
        raise RuntimeError(
            f"Isaac Sim accepted only {scene_index}/{scene_count} scenes from "
            f"{len(candidate_paths)} candidates"
        )
    write_json_atomic(output_root / "run_manifest.json", {**run_manifest, "status": "complete"})
    write_json_atomic(output_root / "run_progress.json", {**run_manifest, "status": "complete"})
    print(
        f"[COMPLETE] {scene_count} scenes saved below {output_root / 'scenes'}",
        flush=True,
    )
    if ARGS.hold and not ARGS.headless:
        while SIMULATION_APP.is_running():
            sim.render()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    if ARGS.headless:
        # Isaac Sim 5.0's base experience can block indefinitely in
        # ``close_stage()`` after an annotator-only Replicator run.  At this
        # point all NumPy/JSON files have been synchronously closed and the
        # completion manifest has been atomically renamed, so use Kit's batch
        # fast-exit semantics and let the OS release CUDA/Vulkan resources.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    else:
        # All arrays are read and written synchronously above; no Replicator
        # writer remains to drain.  Isaac Sim 5.0 can otherwise wait forever
        # for the annotator-only orchestrator during shutdown.
        SIMULATION_APP.close(wait_for_replicator=False)
        if exit_code:
            raise SystemExit(exit_code)
