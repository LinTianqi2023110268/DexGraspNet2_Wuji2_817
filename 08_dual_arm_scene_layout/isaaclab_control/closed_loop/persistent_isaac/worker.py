#!/usr/bin/env python3
"""Persistent Isaac Lab/Isaac Sim worker for the interactive grasp loop.

One process owns one AppLauncher, one SimulationContext, one robot, one camera
and one set of rigid objects for the *whole* session.  Commands arrive as
line-delimited JSON on stdin:

    ping
    capture   -> hold HOME briefly and save aligned RGB-D + current scene state
    execute   -> execute an already-planned q7/q20 route; NO second IK/FK gate
    snapshot  -> save current object poses for audit/recovery
    shutdown

This removes the old capture-process -> close -> execution-process -> close
cycle.  The physical world seen by the camera is the same world that executes
the planned route.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import selectors
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher


_PROTOCOL = "__ISAAC_SESSION__"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stdio", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_arguments()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import torch  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.physx.scripts import physicsUtils  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


PROJECT_ROOT = ARGS.project_root.expanduser().resolve()
CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
sys.path.insert(0, str(CONTROL_ROOT))
from core.config import DEFAULT_INITIAL_RIGHT_ARM_DEG  # noqa: E402


ROBOT_PRIM = "/World/Layout/DualArmMount/DualArm"
CAMERA_PRIM = "/World/Sensors/TopD435iVirtual/Camera"
TASK_ROOT = "/World/Layout/TableAssembly/TestScene0000"
SOURCE_ZONE_PRIM = "/World/Layout/TableAssembly/SourceZone"
TABLE_PRIM = "/World/Layout/TableAssembly/Table"
CAPTURE_HIDE_PRIMS = (
    "/World/Layout/TableAssembly/SourceZone",
    "/World/Layout/TableAssembly/PlacementZone",
    "/World/Sensors/TopD435iVirtual/Frustum",
    "/World/Markers",
)
RIGHT_ARM_NAMES = [f"arm_r_joint_{index}" for index in range(1, 8)]
ROUTE_STAGES = [
    "pregrasp", "cover", "grasp", "squeeze", "lift",
    "transfer", "place", "release", "retreat",
]
TRACE_FIELDS = [
    "time_s", "state", "progress", "flange_error_mm", "flange_error_deg",
    "target_object_x_m", "target_object_y_m", "target_object_z_m",
    "target_object_qw", "target_object_qx", "target_object_qy", "target_object_qz",
    "target_object_vx_m_s", "target_object_vy_m_s", "target_object_vz_m_s",
    "target_object_wx_rad_s", "target_object_wy_rad_s", "target_object_wz_rad_s",
    "object_lift_mm", "max_arm_qdot_rad_s", "max_arm_joint_goal_error_deg",
    "max_wuji2_joint_target_error_deg",
    "target_contact_finger_count", "target_contact_force_max_n",
    "target_contact_force_sum_n",
    "target_contact_normal_force_estimated_n",
    "target_contact_friction_force_estimated_n",
    "thumb_target_force_n", "index_target_force_n", "middle_target_force_n",
    "ring_target_force_n", "pinky_target_force_n", "palm_target_force_n",
]
CONTACT_GROUPS = ("thumb", "index", "middle", "ring", "pinky", "palm")


def contact_group_for_body(body_name: str) -> str | None:
    if body_name.startswith("r_thumb"):
        return "thumb"
    if body_name.startswith("r_index_finger"):
        return "index"
    if body_name.startswith("r_middle_finger"):
        return "middle"
    if body_name.startswith("r_ring_finger"):
        return "ring"
    if body_name.startswith("r_pinky"):
        return "pinky"
    if body_name in {"r_wrist", "r_base_link"}:
        return "palm"
    return None


def create_target_contact_sensors(
    stage: Usd.Stage,
    robot_root: str,
    target_rigid_path: str,
) -> dict[str, tuple[str, ContactSensor]]:
    """Create grouped Wuji2 contact sensors filtered to the selected target object.

    Isaac Lab's ContactSensor reports force vectors but not per-contact PhysX
    normals.  Downstream diagnostics therefore report total force exactly and
    normal/friction as explicitly-labelled estimates.
    """

    sensors: dict[str, tuple[str, ContactSensor]] = {}
    grouped_paths = {group: [] for group in CONTACT_GROUPS}
    for prim in stage.Traverse():
        body_path = str(prim.GetPath())
        if not body_path.startswith(robot_root + "/") or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        group = contact_group_for_body(prim.GetName())
        if group is None:
            continue
        grouped_paths[group].append(body_path)
        sim_utils.activate_contact_sensors(body_path, threshold=0.0, stage=stage)
        sensors[f"{group}:{prim.GetName()}"] = (group, ContactSensor(ContactSensorCfg(
            prim_path=body_path,
            update_period=0.0,
            history_length=0,
            track_pose=True,
            track_air_time=False,
            filter_prim_paths_expr=[target_rigid_path],
        )))
    missing = [group for group, paths in grouped_paths.items() if not paths]
    if missing:
        raise RuntimeError(f"Missing Wuji2 contact body groups: {missing}")
    print(f"[CONTACT AUDIT] whole Wuji2 hand -> {target_rigid_path}", flush=True)
    for group, paths in grouped_paths.items():
        print(f"  {group}: {[Path(path).name for path in paths]}", flush=True)
    return sensors


def emit(payload: dict) -> None:
    print(_PROTOCOL + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale]
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale]
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = [(matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale]
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = [(matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def pose_from_position_quaternion_wxyz(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    pose[:3, :3] = quaternion_wxyz_to_matrix(quaternion)
    return pose


def gf_matrix_to_numpy(matrix: Gf.Matrix4d) -> np.ndarray:
    return np.asarray([[float(matrix[row][column]) for column in range(4)] for row in range(4)])


def rigid_world_transform(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    row_matrix = gf_matrix_to_numpy(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    raw = row_matrix[:3, :3]
    scales = np.linalg.norm(raw, axis=1)
    normalized = raw / scales[:, None]
    u, _, vt = np.linalg.svd(normalized)
    rotation_row = u @ vt
    if np.linalg.det(rotation_row) < 0.0:
        u[:, -1] *= -1.0
        rotation_row = u @ vt
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_row.T
    result[:3, 3] = row_matrix[3, :3]
    return result


def set_reference_transform(stage: Usd.Stage, root_path: str, pose: np.ndarray) -> None:
    prim = stage.GetPrimAtPath(root_path)
    quaternion = matrix_to_quaternion_wxyz(pose[:3, :3])
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*map(float, quaternion[1:]))))
    transform.SetTranslate(Gf.Vec3d(*map(float, pose[:3, 3])))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform)


def find_one_rigid_prim(stage: Usd.Stage, prefix: str) -> Usd.Prim:
    matches = [
        prim for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one rigid body below {prefix}, got {[str(x.GetPath()) for x in matches]}")
    return matches[0]


def configure_demo_view(config: dict) -> None:
    if bool(getattr(ARGS, "headless", False)):
        return
    view = config.get("viewer_camera", {})
    if not bool(view.get("enabled", True)):
        return
    stage = get_current_stage()
    for path in view.get("hide_prims", ["/World/Sensors/TopD435iVirtual/Frustum", "/World/Markers"]):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    target = np.asarray(view.get("target_world_m", [0.0, -0.145, 0.50]), dtype=np.float64)
    yaw = math.radians(float(view.get("yaw_about_world_z_deg", -90.0)))
    distance = float(view.get("horizontal_distance_m", 1.45))
    eye = target + np.asarray([
        distance * math.cos(yaw), distance * math.sin(yaw), float(view.get("height_above_target_m", 0.75))
    ])
    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")


def camera_calibration(stage: Usd.Stage, width: int, height: int) -> tuple[np.ndarray, np.ndarray, dict]:
    camera = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA_PRIM))
    focal_mm = float(camera.GetFocalLengthAttr().Get())
    horizontal_mm = float(camera.GetHorizontalApertureAttr().Get())
    vertical_mm = float(camera.GetVerticalApertureAttr().Get())
    clipping = camera.GetClippingRangeAttr().Get()
    intrinsic = np.asarray([
        [focal_mm * width / horizontal_mm, 0.0, width / 2.0],
        [0.0, focal_mm * height / vertical_mm, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    row = gf_matrix_to_numpy(
        UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = np.column_stack((row[0, :3], -row[1, :3], -row[2, :3]))
    world_from_camera[:3, 3] = row[3, :3]
    return intrinsic, world_from_camera, {
        "focal_length_mm": focal_mm,
        "horizontal_aperture_mm": horizontal_mm,
        "vertical_aperture_mm": vertical_mm,
        "near_far_m": [float(clipping[0]), float(clipping[1])],
    }


def depth_preview(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("Camera returned no valid depth")
    values = depth[valid]
    near, far = np.percentile(values, [1.0, 99.0])
    if far <= near:
        far = near + 1.0e-6
    image = np.zeros(depth.shape, dtype=np.float32)
    image[valid] = np.clip((far - depth[valid]) / (far - near), 0.0, 1.0)
    return np.round(image * 255).astype(np.uint8), {
        "valid_pixel_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "minimum_depth_m": float(values.min()),
        "maximum_depth_m": float(values.max()),
    }


def create_robot(config: dict) -> Articulation:
    from isaaclab.actuators import ImplicitActuatorCfg
    actuators = {
        "native_left_and_wuji2": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_.*", "r_.*"], stiffness=None, damping=None,
            effort_limit_sim=None, velocity_limit_sim=None,
        )
    }
    for group in config["right_arm_force_natural_frequency_groups"]:
        actuators[group["name"]] = ImplicitActuatorCfg(
            joint_names_expr=group["joint_names_expr"], stiffness=None, damping=None,
            effort_limit_sim=None, velocity_limit_sim=None,
        )
    return Articulation(ArticulationCfg(prim_path=config["robot_prim"], spawn=None, actuators=actuators))


def set_force_drive_type(config: dict) -> None:
    stage = get_current_stage()
    requested = set(config["right_arm_joints"])
    found = set()
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(config["robot_prim"] + "/") or prim.GetName() not in requested:
            continue
        drive = UsdPhysics.DriveAPI(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        found.add(prim.GetName())
    if found != requested:
        raise RuntimeError(f"Force Drive mapping failed: missing={sorted(requested - found)}")


def apply_ft04_gains(robot: Articulation, arm_ids: list[int], config: dict) -> list[dict]:
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[0].to(robot.device)
    stiffness = robot.data.joint_stiffness.clone()
    damping = robot.data.joint_damping.clone()
    audit = []
    for group in config["right_arm_force_natural_frequency_groups"]:
        ids, names = robot.find_joints(group["joint_names_expr"], preserve_order=True)
        if any(int(index) not in arm_ids for index in ids):
            raise RuntimeError(f"ft04 group matched non-arm DOF: {names}")
        frequency = float(group["natural_frequency_rad_s"])
        zeta = float(group.get("damping_ratio", 1.0))
        equivalent_mass = torch.clamp(torch.diag(mass_matrix)[ids], min=1.0e-6)
        stiffness[0, ids] = equivalent_mass * frequency * frequency
        damping[0, ids] = 2.0 * zeta * frequency * equivalent_mass
        audit.append({"names": names, "nf": frequency})
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.reset()
    return audit


def quintic(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def quat_error_deg(actual: torch.Tensor, target: torch.Tensor) -> float:
    dot = torch.abs(torch.sum(actual * target)).clamp(0.0, 1.0)
    return float(torch.rad2deg(2.0 * torch.acos(dot)))


def pose_errors(body_pose: torch.Tensor, target_matrix: np.ndarray | None) -> tuple[float, float]:
    if target_matrix is None:
        return 0.0, 0.0
    target_position = torch.as_tensor(target_matrix[:3, 3], device=body_pose.device, dtype=body_pose.dtype)
    target_quaternion = torch.as_tensor(
        matrix_to_quaternion_wxyz(target_matrix[:3, :3]), device=body_pose.device, dtype=body_pose.dtype
    )
    position_error_mm = 1000.0 * float(torch.linalg.vector_norm(body_pose[:3] - target_position))
    return position_error_mm, quat_error_deg(body_pose[3:7], target_quaternion)


class PersistentScene:
    def __init__(self, scene_manifest: Path, config: dict):
        self.scene_manifest_path = Path(scene_manifest).resolve()
        self.scene_source = load_json(self.scene_manifest_path)
        self.config = config
        stage_path = PROJECT_ROOT / config["stage"] if not Path(config["stage"]).is_absolute() else Path(config["stage"])
        print("[Isaac] 初始化持续场景：只加载一次 USD / 机械臂 / 相机", flush=True)
        if not stage_utils.open_stage(str(stage_path.resolve())):
            raise RuntimeError(f"Cannot open stage: {stage_path}")
        configure_demo_view(config)
        self.stage = get_current_stage()
        self.world_from_source = rigid_world_transform(self.stage, SOURCE_ZONE_PRIM)
        self.source_from_world = np.linalg.inv(self.world_from_source)
        self.objects, self.object_records = self._spawn_objects()
        self.objects_by_seg = {
            int(record["segmentation_id"]): wrapper
            for wrapper, record in zip(self.objects, self.object_records)
        }
        set_force_drive_type(config)
        self.simulation = SimulationContext(sim_utils.SimulationCfg(
            dt=float(config["physics_dt_s"]),
            render_interval=int(config.get("render_interval", 1)),
            device=ARGS.device,
        ))
        self.robot = create_robot(config)
        width, height = [int(x) for x in config.get("camera_resolution_wh", [1280, 720])]
        self.camera_width = width
        self.camera_height = height
        self.camera = Camera(CameraCfg(
            prim_path=CAMERA_PRIM,
            update_period=0.0,
            width=width,
            height=height,
            data_types=["rgb", "distance_to_image_plane"],
            update_latest_camera_pose=False,
            spawn=None,
        ))
        self.simulation.reset()
        self.dt = float(self.simulation.get_physics_dt())
        self.robot.update(self.dt)
        for obj in self.objects:
            obj.update(self.dt)
        # Read-only startup audit. Never silently repairs assets.
        self.object_physics_audit = self._audit_task_objects(emit_terminal=True)
        self.arm_ids, self.arm_names = self.robot.find_joints(config["right_arm_joints"], preserve_order=True)
        if self.arm_names != config["right_arm_joints"] or self.robot.num_joints != int(config["expected_total_actuated_joints"]) or not self.robot.is_fixed_base:
            raise RuntimeError("persistent robot audit failed")
        self.flange_ids, flange_names = self.robot.find_bodies([config["flange_body"]], preserve_order=True)
        if flange_names != [config["flange_body"]]:
            raise RuntimeError("flange body mapping changed")
        self.flange_id = int(self.flange_ids[0])
        self.gain_audit = apply_ft04_gains(self.robot, [int(x) for x in self.arm_ids], config)
        self.home_q = np.deg2rad(np.asarray(config.get("home_q_deg", DEFAULT_INITIAL_RIGHT_ARM_DEG), dtype=np.float32))
        authored = self.robot.data.joint_pos[:, self.arm_ids]
        authored_error = float(torch.max(torch.abs(torch.rad2deg(
            authored - torch.as_tensor(self.home_q, device=self.robot.device).reshape(1, 7)
        ))))
        if authored_error > float(config.get("initial_home_audit_tolerance_deg", 0.2)):
            raise RuntimeError(f"authored HOME differs by {authored_error:.3f} deg")
        self.command = self.robot.data.joint_pos.clone()
        self.robot.set_joint_position_target(self.command)
        self.current_desired_arm = self.robot.data.joint_pos[:, self.arm_ids].clone()
        self.capture_count = 0
        self.execute_count = 0
        # The CAPTURE command owns the single 1 s settle, including cycle 1.
        # Do not add a hidden second settle during worker construction.
        initial_hold = float(config.get("initial_hold_s", 0.0))
        if initial_hold > 0.0:
            self.hold(initial_hold, render=False)
        self.pause()
        print("[Isaac] ✓ 持续场景已就绪（规划期间物理暂停）", flush=True)


    def play(self) -> None:
        # Use Isaac Lab's SimulationContext lifecycle API rather than mutating
        # omni.timeline behind its back.  This keeps SimulationContext's own
        # playing state synchronized with the Kit/PhysX timeline.
        if not self.simulation.is_playing():
            self.simulation.play()

    def pause(self) -> None:
        if self.simulation.is_playing():
            self.simulation.pause()

    def _spawn_objects(self) -> tuple[list[RigidObject], list[dict]]:
        old = self.stage.GetPrimAtPath(TASK_ROOT)
        if old.IsValid():
            self.stage.RemovePrim(TASK_ROOT)
        root = UsdGeom.Xform.Define(self.stage, TASK_ROOT).GetPrim()
        root.CreateAttribute("dgn2:dynamicScene", Sdf.ValueTypeNames.Bool).Set(True)
        material_path = "/World/PhysicsMaterials/TaskObjectsPersistent"
        sim_utils.spawn_rigid_body_material(
            material_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        )
        wrappers: list[RigidObject] = []
        records: list[dict] = []
        root_paths: list[str] = []
        for record in self.scene_source["objects"]:
            seg_id = int(record["segmentation_id"])
            code = str(record.get("object_code", record.get("code", f"object_{seg_id}")))
            root_path = f"{TASK_ROOT}/Object_{seg_id:03d}"
            simulation_usd = record.get("simulation_usd")
            if simulation_usd is None:
                pool_index = int(record["object_pool_index"])
                simulation_usd = (
                    PROJECT_ROOT
                    / "02_training_dataset/data/scene_datasets/wuji2_test60_10upright_10view_v1"
                    / f"usd_cache/object_{pool_index:03d}/flat/object_{pool_index:03d}_editable.usd"
                )
            simulation_usd = Path(simulation_usd).expanduser().resolve()
            if not simulation_usd.is_file():
                raise FileNotFoundError(simulation_usd)
            add_reference_to_stage(str(simulation_usd), root_path)
            source_pose_value = record.get("T_world_centered_object", record.get("pose_world_object"))
            if source_pose_value is None:
                raise KeyError(f"object {seg_id} has no source-zone pose")
            world_pose = self.world_from_source @ np.asarray(source_pose_value, dtype=np.float64)
            set_reference_transform(self.stage, root_path, world_pose)
            prim = self.stage.GetPrimAtPath(root_path)
            prim.CreateAttribute("dgn2:segmentationId", Sdf.ValueTypeNames.Int).Set(seg_id)
            prim.CreateAttribute("dgn2:objectCode", Sdf.ValueTypeNames.String).Set(code)
            rigid = find_one_rigid_prim(self.stage, root_path)
            PhysxSchema.PhysxRigidBodyAPI.Apply(rigid).CreateDisableGravityAttr().Set(False)
            UsdPhysics.MassAPI.Apply(rigid).CreateMassAttr().Set(float(self.config.get("task_object_mass_kg", 0.1)))
            sim_utils.bind_physics_material(root_path, material_path)
            wrappers.append(RigidObject(RigidObjectCfg(prim_path=str(rigid.GetPath()), spawn=None)))
            root_paths.append(root_path)
            records.append({
                **record,
                "segmentation_id": seg_id,
                "object_code": code,
                "simulation_usd": str(simulation_usd),
                "root_path": root_path,
                "rigid_path": str(rigid.GetPath()),
            })
        # Preserve the validated execution behavior: task objects do not collide
        # with each other, while robot/table/object PhysX collisions stay active.
        group_path = "/World/CollisionGroups/PersistentTaskObjects"
        group = UsdPhysics.CollisionGroup.Define(self.stage, group_path)
        group.CreateFilteredGroupsRel().AddTarget(group.GetPath())
        for root_path in root_paths:
            physicsUtils.add_collision_to_collision_group(self.stage, root_path, group_path)
        print(f"[Isaac] ✓ 动态物体={len(wrappers)}，对象对过滤={len(list(itertools.combinations(root_paths, 2)))}", flush=True)
        return wrappers, records


    def _collision_aabb_world(self, root_path: str) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        # Union world AABB of prims carrying UsdPhysics.CollisionAPI.
        root = self.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return None, None, 0
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
            useExtentsHint=True,
        )
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        collision_count = 0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision_count += 1
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            lower = np.asarray(aligned.GetMin(), dtype=np.float64)
            upper = np.asarray(aligned.GetMax(), dtype=np.float64)
            if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
                mins.append(lower)
                maxs.append(upper)
        if not mins:
            return None, None, collision_count
        return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0), collision_count

    def _audit_task_objects(self, *, emit_terminal: bool) -> dict:
        table_min, table_max, table_collision_count = self._collision_aabb_world(TABLE_PRIM)
        table_top = None if table_max is None else float(table_max[2])
        warn_penetration_mm = float(self.config.get("object_table_penetration_warn_mm", 2.0))
        rows = []
        for wrapper, record in zip(self.objects, self.object_records):
            rigid = self.stage.GetPrimAtPath(str(record["rigid_path"]))
            rb = UsdPhysics.RigidBodyAPI(rigid)
            rigid_enabled_value = rb.GetRigidBodyEnabledAttr().Get() if rb else None
            kinematic_value = rb.GetKinematicEnabledAttr().Get() if rb else None
            rigid_enabled = True if rigid_enabled_value is None else bool(rigid_enabled_value)
            kinematic_enabled = False if kinematic_value is None else bool(kinematic_value)

            mass_api = UsdPhysics.MassAPI(rigid)
            mass_value = mass_api.GetMassAttr().Get() if mass_api else None
            mass_kg = None if mass_value is None else float(mass_value)

            lower, upper, collision_count = self._collision_aabb_world(str(record["root_path"]))
            bottom_minus_table_mm = None
            penetration_mm = None
            if lower is not None and table_top is not None:
                bottom_minus_table_mm = 1000.0 * float(lower[2] - table_top)
                penetration_mm = max(0.0, -bottom_minus_table_mm)

            warnings = []
            if not rigid_enabled:
                warnings.append("rigidBodyEnabled=false")
            if kinematic_enabled:
                warnings.append("kinematicEnabled=true")
            if collision_count <= 0:
                warnings.append("no CollisionAPI prim")
            if penetration_mm is not None and penetration_mm > warn_penetration_mm:
                warnings.append(
                    f"collision AABB penetrates table by {penetration_mm:.1f}mm "
                    f"> {warn_penetration_mm:.1f}mm"
                )

            row = {
                "segmentation_id": int(record["segmentation_id"]),
                "object_code": str(record["object_code"]),
                "rigid_path": str(record["rigid_path"]),
                "simulation_usd": str(record["simulation_usd"]),
                "rigid_body_enabled": bool(rigid_enabled),
                "kinematic_enabled": bool(kinematic_enabled),
                "mass_kg": mass_kg,
                "collision_prim_count": int(collision_count),
                "collision_aabb_world_min_m": None if lower is None else lower.tolist(),
                "collision_aabb_world_max_m": None if upper is None else upper.tolist(),
                "table_top_world_z_m": table_top,
                "collision_bottom_minus_table_top_mm": bottom_minus_table_mm,
                "table_penetration_mm": penetration_mm,
                "warnings": warnings,
            }
            rows.append(row)

            if emit_terminal:
                mass_text = "NA" if mass_kg is None else f"{mass_kg:.3f}kg"
                bottom_text = (
                    "NA" if bottom_minus_table_mm is None
                    else f"{bottom_minus_table_mm:+.1f}mm"
                )
                print(
                    f"[PHYS-AUDIT] seg={row['segmentation_id']:>3d} "
                    f"rigid={int(rigid_enabled)} kin={int(kinematic_enabled)} "
                    f"mass={mass_text} collision={collision_count} "
                    f"bottom-table={bottom_text} | {row['object_code']}",
                    flush=True,
                )
                if warnings:
                    print(
                        f"[PHYS-AUDIT WARNING] seg={row['segmentation_id']}: "
                        + "; ".join(warnings),
                        flush=True,
                    )

        return {
            "schema_version": 1,
            "status": "WARN" if any(row["warnings"] for row in rows) else "PASS",
            "table_prim": TABLE_PRIM,
            "table_collision_prim_count": int(table_collision_count),
            "table_collision_aabb_world_min_m": None if table_min is None else table_min.tolist(),
            "table_collision_aabb_world_max_m": None if table_max is None else table_max.tolist(),
            "table_top_world_z_m": table_top,
            "penetration_warning_threshold_mm": warn_penetration_mm,
            "objects": rows,
        }

    def _write_object_physics_audit(self, output_dir: Path) -> Path:
        audit = self._audit_task_objects(emit_terminal=False)
        self.object_physics_audit = audit
        path = Path(output_dir).resolve() / "object_physics_audit.json"
        path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def step(self, *, render: bool = False) -> None:
        self.robot.set_joint_position_target(self.command)
        self.robot.write_data_to_sim()
        for obj in self.objects:
            obj.write_data_to_sim()
        self.simulation.step(render=render)
        self.robot.update(self.dt)
        for obj in self.objects:
            obj.update(self.dt)

    def hold(self, seconds: float, *, render: bool = False) -> None:
        count = max(1, round(float(seconds) / self.dt))
        for _ in range(count):
            self.step(render=render)

    def _set_capture_debug_visibility(self, *, visible: bool, previous: dict[str, object] | None = None) -> dict[str, object]:
        """Hide visual debug helpers only while writing production RGB-D frames."""
        saved: dict[str, object] = {}
        for path in CAPTURE_HIDE_PRIMS:
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
                continue
            imageable = UsdGeom.Imageable(prim)
            attr = imageable.GetVisibilityAttr()
            if previous is None:
                saved[path] = attr.Get() if attr.IsValid() else None
            old_value = None if previous is None else previous.get(path)
            if visible:
                if old_value:
                    attr.Set(old_value)
                else:
                    imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        return saved

    def _current_scene_manifest(self) -> dict:
        rows = []
        for wrapper, record in zip(self.objects, self.object_records):
            pose = wrapper.data.root_pose_w[0].detach().cpu().numpy()
            world_from_object = pose_from_position_quaternion_wxyz(pose[:3], pose[3:7])
            source_from_object = self.source_from_world @ world_from_object
            row = dict(record)
            row["pose_world_object"] = source_from_object.tolist()
            row["T_world_centered_object"] = source_from_object.tolist()
            row["settled_pose_layout_world"] = world_from_object.tolist()
            row["settled_linear_velocity_world_m_s"] = wrapper.data.root_lin_vel_w[0].detach().cpu().tolist()
            row["settled_angular_velocity_world_rad_s"] = wrapper.data.root_ang_vel_w[0].detach().cpu().tolist()
            rows.append(row)
        return {
            "schema_version": 3,
            "status": "persistent_isaac_current_scene",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_scene_manifest": str(self.scene_manifest_path),
            "coordinate_contract": {
                "object_pose": "pose_world_object means T_SourceZone_centeredObject",
                "layout_bridge": "T_layoutWorld_object = T_layoutWorld_SourceZone @ pose_world_object",
            },
            "table": self.scene_source["table"],
            "world_from_source_zone": self.world_from_source.tolist(),
            "objects": rows,
            "persistent_session": True,
            "capture_count": self.capture_count,
            "execute_count": self.execute_count,
        }

    def write_snapshot(self, output: Path) -> Path:
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self._current_scene_manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output

    def _write_robot_state(self, output: Path) -> Path:
        joint_positions = self.robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
        joint_names = [str(name) for name in self.robot.joint_names]
        path = output / "robot_state.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "status": "persistent_measured_robot_state",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "robot_prim": self.config["robot_prim"],
            "joint_count": int(self.robot.num_joints),
            "joint_names": joint_names,
            "joint_positions_by_name": {
                name: float(value) for name, value in zip(joint_names, joint_positions)
            },
            "right_arm_joint_names": list(self.arm_names),
            "right_arm_q_current_rad": joint_positions[np.asarray(self.arm_ids, dtype=np.int64)].tolist(),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def capture(self, output_dir: Path, hold_s: float | None = None) -> dict:
        """Hold the current HOME scene, capture RGB-D, then freeze physics.

        The camera render frames are *inside* the requested hold interval.  This
        avoids the old one-shot behavior where a 1 s settle was followed by
        extra physics steps just to warm the camera.  The scene manifest and
        robot_state are written only after the final rendered physics step, so
        RGB-D, object poses and q_current refer to the same frozen instant.
        """
        self.play()
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        requested_hold = float(self.config.get("post_home_hold_s", 1.0) if hold_s is None else hold_s)
        if requested_hold < 0.0:
            raise ValueError("capture hold_s must be non-negative")

        camera_warmup_frames = max(1, int(self.config.get("camera_warmup_frames", 12)))
        # With the configured 120 Hz physics and 1.0 s HOME hold this runs 120
        # physics steps total, rendering/updating the camera only on the final
        # 12 steps.  No hidden +0.1 s settle is added.
        hold_steps = max(1, round(requested_hold / self.dt))
        render_start = max(0, hold_steps - camera_warmup_frames)
        visibility_before_capture = self._set_capture_debug_visibility(visible=False)
        try:
            for step_index in range(hold_steps):
                render = step_index >= render_start
                self.step(render=render)
                if render:
                    self.camera.update(dt=self.dt, force_recompute=True)

            # Freeze immediately at the image/state instant.  Planning can now take
            # arbitrarily long without q_current or the objects drifting.
            self.pause()
            self.capture_count += 1
            physics_audit_path = self._write_object_physics_audit(output)

            intrinsic, world_from_camera, camera_model = camera_calibration(
                self.stage, self.camera_width, self.camera_height
            )
            rgb = self.camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
            depth = np.squeeze(
                self.camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy().astype(np.float32)
            )
            preview, depth_stats = depth_preview(depth)
        finally:
            self._set_capture_debug_visibility(visible=True, previous=visibility_before_capture)

        # State is serialized AFTER the final camera/physics step and AFTER the
        # timeline is paused, keeping all downstream coordinate contracts aligned.
        settled_path = output / "settled_scene_manifest.json"
        settled_path.write_text(
            json.dumps(self._current_scene_manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        robot_state = self._write_robot_state(output)

        Image.fromarray(rgb, mode="RGB").save(output / "rgb.png")
        Image.fromarray(preview, mode="L").save(output / "depth_preview.png")
        np.save(output / "depth_m.npy", depth)
        np.save(output / "intrinsics.npy", intrinsic)
        np.save(output / "T_world_camera.npy", world_from_camera)
        capture_manifest = {
            "schema_version": 3,
            "status": "persistent_rgbd_capture_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "capture_backend": "persistent Isaac Lab 2.2 / Isaac Sim session",
            "camera_prim": CAMERA_PRIM,
            "resolution_wh": [self.camera_width, self.camera_height],
            "rgb": {"file": "rgb.png", "shape": list(rgb.shape)},
            "depth": {"file": "depth_m.npy", "shape": list(depth.shape), **depth_stats},
            "intrinsics": {"file": "intrinsics.npy", "K": intrinsic.tolist()},
            "extrinsics": {"file": "T_world_camera.npy", "matrix": world_from_camera.tolist()},
            "camera_model": camera_model,
            "settled_scene_manifest": str(settled_path),
            "robot_state": str(robot_state),
            "object_physics_audit": str(physics_audit_path),
            "post_home_hold_s": requested_hold,
            "persistent_session": True,
        }
        capture_path = output / "capture_manifest.json"
        capture_path.write_text(json.dumps(capture_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "status": "PASS",
            "capture_manifest": str(capture_path),
            "rgb": str(output / "rgb.png"),
            "depth_m": str(output / "depth_m.npy"),
            "intrinsics": str(output / "intrinsics.npy"),
            "T_world_camera": str(output / "T_world_camera.npy"),
            "settled_scene_manifest": str(settled_path),
            "robot_state": str(robot_state),
            "object_physics_audit": str(physics_audit_path),
            "hold_s": requested_hold,
            "valid_depth_fraction": depth_stats["valid_fraction"],
        }

    def execute(self, *, case_root: Path, plan_npz: Path, output_dir: Path, target_segmentation_id: int) -> dict:
        self.play()
        case_root = Path(case_root).resolve()
        plan_npz = Path(plan_npz).resolve()
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        if int(target_segmentation_id) not in self.objects_by_seg:
            raise KeyError(f"target segmentation id {target_segmentation_id} is absent from persistent scene")
        target_object = self.objects_by_seg[int(target_segmentation_id)]
        target_record = next(
            record for record in self.object_records
            if int(record["segmentation_id"]) == int(target_segmentation_id)
        )
        contact_sensors = create_target_contact_sensors(
            self.stage,
            ROBOT_PRIM,
            str(target_record["rigid_path"]),
        )
        with np.load(plan_npz, allow_pickle=False) as z:
            stage_names = [str(x) for x in z["waypoint_names"].tolist()]
            arm_q = np.asarray(z["arm_q_rad"], dtype=np.float64)
            flange_targets = np.asarray(z["world_from_right_flange"], dtype=np.float64)
            source_candidate_index = int(np.asarray(z["source_candidate_index"]).item())
        if stage_names != ROUTE_STAGES or arm_q.shape != (len(ROUTE_STAGES), 7):
            raise RuntimeError(f"invalid flexible plan: stages={stage_names}, q={arm_q.shape}")
        if flange_targets.shape != (len(ROUTE_STAGES), 4, 4):
            raise RuntimeError(f"invalid flange target shape {flange_targets.shape}")
        index_of = {name: i for i, name in enumerate(stage_names)}

        hand_path = case_root / "06_isaacsim/final_waypoints.npz"
        with np.load(hand_path, allow_pickle=False) as z:
            hand_names = [str(x) for x in z["finger_joint_names"].tolist()]
            hand_stage_names = [str(x) for x in z["waypoint_names"].tolist()]
            hand_q5 = np.asarray(z["waypoint_joint_positions"][0], dtype=np.float64)
            squeeze_dense = np.asarray(z["squeeze_dense_q20_path"], dtype=np.float64)
        hand_index = {name: i for i, name in enumerate(hand_stage_names)}
        hand_ids, matched_names = self.robot.find_joints(hand_names, preserve_order=True)
        if matched_names != hand_names or len(hand_ids) != 20:
            raise RuntimeError("Wuji2 20-DOF mapping changed")

        durations = self.config["durations_s"]
        telemetry_hz = float(self.config.get("telemetry_hz", 10.0))
        telemetry_stride = max(1, round(1.0 / (telemetry_hz * self.dt)))
        trace: list[dict] = []
        replay_fps = float(self.config.get("replay_record_fps", 30.0))
        replay_period = 1.0 / replay_fps
        replay_next = 0.0
        replay_time: list[float] = []
        replay_state: list[str] = []
        replay_joint: list[np.ndarray] = []
        replay_objects: list[np.ndarray] = []
        sim_time = 0.0
        action_started = time.perf_counter()
        initial_object_position = target_object.data.root_pos_w[0].clone()
        max_object_lift_mm = 0.0
        actual_arm_start = self.robot.data.joint_pos[:, self.arm_ids].clone()
        self.current_desired_arm = actual_arm_start.clone()
        current_stage = "INIT"
        verified_target_lift_mm: float | None = None

        def update_contact_sensors() -> None:
            for _, sensor in contact_sensors.values():
                sensor.update(self.dt, force_recompute=True)

        def contact_snapshot() -> dict:
            update_contact_sensors()
            object_position = target_object.data.root_pos_w[0]
            group_force_vectors = {
                group: torch.zeros(3, device=self.robot.device, dtype=self.command.dtype)
                for group in CONTACT_GROUPS
            }
            group_force_norms = {group: 0.0 for group in CONTACT_GROUPS}
            normal_estimated = 0.0
            friction_estimated = 0.0
            force_sum = torch.zeros(3, device=self.robot.device, dtype=self.command.dtype)
            for group, sensor in contact_sensors.values():
                target_matrix = sensor.data.force_matrix_w
                if target_matrix.numel() == 0:
                    continue
                vectors = target_matrix.reshape(-1, 3)
                norms = torch.linalg.vector_norm(vectors, dim=-1)
                if norms.numel() == 0:
                    continue
                max_index = int(torch.argmax(norms).item())
                vector = vectors[max_index]
                norm = float(norms[max_index])
                if norm <= group_force_norms[group]:
                    continue
                group_force_norms[group] = norm
                group_force_vectors[group] = vector
            for vector in group_force_vectors.values():
                force_sum = force_sum + vector
            total_force_norm = float(torch.linalg.vector_norm(force_sum))
            max_force = max(group_force_norms.values(), default=0.0)
            finger_count = sum(group_force_norms[group] > 1.0e-3 for group in CONTACT_GROUPS[:-1])
            # Estimate normal/tangential components using a radial axis from the
            # target-object COM to each hand body.  ContactSensor does not expose
            # PhysX contact normals, so these are diagnostics, not gate values.
            for group, sensor in contact_sensors.values():
                vector = group_force_vectors[group]
                norm = float(torch.linalg.vector_norm(vector))
                if norm <= 1.0e-9:
                    continue
                try:
                    sensor_pos = sensor.data.pos_w[0]
                    axis = sensor_pos - object_position
                    axis_norm = float(torch.linalg.vector_norm(axis))
                    if axis_norm <= 1.0e-9:
                        continue
                    axis = axis / axis_norm
                    normal_component = torch.dot(vector, axis) * axis
                    friction_component = vector - normal_component
                    normal_estimated += abs(float(torch.dot(vector, axis)))
                    friction_estimated += float(torch.linalg.vector_norm(friction_component))
                except Exception:
                    continue
            return {
                "finger_count": int(finger_count),
                "max_force_n": float(max_force),
                "sum_force_n": float(total_force_norm),
                "normal_force_estimated_n": float(normal_estimated),
                "friction_force_estimated_n": float(friction_estimated),
                "group_force_norms": group_force_norms,
                "sum_force_vector_world_n": force_sum.detach().cpu().tolist(),
            }

        def object_state_snapshot() -> dict:
            return {
                "position_world_m": target_object.data.root_pos_w[0].detach().cpu().tolist(),
                "quaternion_world_wxyz": target_object.data.root_quat_w[0].detach().cpu().tolist(),
                "linear_velocity_world_m_s": target_object.data.root_lin_vel_w[0].detach().cpu().tolist(),
                "angular_velocity_world_rad_s": target_object.data.root_ang_vel_w[0].detach().cpu().tolist(),
                "lift_mm": float(target_lift_mm_now()),
            }

        def print_grasp_contact_audit(stage_label: str) -> None:
            obj = object_state_snapshot()
            contact = contact_snapshot()
            print(f"\n[GRASP CONTACT AUDIT] {stage_label}", flush=True)
            print(
                "  object pose: "
                f"p={np.round(obj['position_world_m'], 6).tolist()} "
                f"q_wxyz={np.round(obj['quaternion_world_wxyz'], 6).tolist()}",
                flush=True,
            )
            print(
                "  object velocity: "
                f"v={np.round(obj['linear_velocity_world_m_s'], 6).tolist()} m/s "
                f"w={np.round(obj['angular_velocity_world_rad_s'], 6).tolist()} rad/s",
                flush=True,
            )
            print(
                "  contact: "
                f"fingers={contact['finger_count']}/5 "
                f"max={contact['max_force_n']:.3f} N "
                f"sum={contact['sum_force_n']:.3f} N "
                f"normal_est={contact['normal_force_estimated_n']:.3f} N "
                f"friction_est={contact['friction_force_estimated_n']:.3f} N",
                flush=True,
            )
            print(
                "  per group force N: "
                + ", ".join(
                    f"{group}={contact['group_force_norms'][group]:.3f}"
                    for group in CONTACT_GROUPS
                ),
                flush=True,
            )
            print(f"  object z delta: {obj['lift_mm']:.3f} mm", flush=True)

        def capture_replay(state: str, force: bool = False) -> None:
            nonlocal replay_next
            if not force and sim_time + 0.5 * self.dt < replay_next:
                return
            replay_time.append(float(sim_time))
            replay_state.append(state)
            replay_joint.append(self.robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32).copy())
            replay_objects.append(np.stack([
                torch.cat((obj.data.root_pos_w[0], obj.data.root_quat_w[0])).detach().cpu().numpy().astype(np.float32)
                for obj in self.objects
            ]))
            replay_next += replay_period

        def monitor(state: str, progress: float, target_matrix: np.ndarray | None, step_index: int) -> tuple[float, float]:
            nonlocal max_object_lift_mm
            position_error, orientation_error = pose_errors(
                self.robot.data.body_pose_w[0, self.flange_id], target_matrix
            )
            object_position = target_object.data.root_pos_w[0]
            object_quat = target_object.data.root_quat_w[0]
            object_lin_vel = target_object.data.root_lin_vel_w[0]
            object_ang_vel = target_object.data.root_ang_vel_w[0]
            object_lift = 1000.0 * float(object_position[2] - initial_object_position[2])
            max_object_lift_mm = max(max_object_lift_mm, object_lift)
            contact = contact_snapshot()
            row = {
                "time_s": float(sim_time),
                "state": state,
                "progress": float(progress),
                "flange_error_mm": float(position_error),
                "flange_error_deg": float(orientation_error),
                "target_object_x_m": float(object_position[0]),
                "target_object_y_m": float(object_position[1]),
                "target_object_z_m": float(object_position[2]),
                "target_object_qw": float(object_quat[0]),
                "target_object_qx": float(object_quat[1]),
                "target_object_qy": float(object_quat[2]),
                "target_object_qz": float(object_quat[3]),
                "target_object_vx_m_s": float(object_lin_vel[0]),
                "target_object_vy_m_s": float(object_lin_vel[1]),
                "target_object_vz_m_s": float(object_lin_vel[2]),
                "target_object_wx_rad_s": float(object_ang_vel[0]),
                "target_object_wy_rad_s": float(object_ang_vel[1]),
                "target_object_wz_rad_s": float(object_ang_vel[2]),
                "object_lift_mm": float(object_lift),
                "max_arm_qdot_rad_s": float(torch.max(torch.abs(self.robot.data.joint_vel[:, self.arm_ids]))),
                "max_arm_joint_goal_error_deg": float(torch.max(torch.abs(torch.rad2deg(
                    self.current_desired_arm - self.robot.data.joint_pos[:, self.arm_ids]
                )))),
                "max_wuji2_joint_target_error_deg": float(torch.max(torch.abs(torch.rad2deg(
                    self.command[:, hand_ids] - self.robot.data.joint_pos[:, hand_ids]
                )))),
                "target_contact_finger_count": contact["finger_count"],
                "target_contact_force_max_n": contact["max_force_n"],
                "target_contact_force_sum_n": contact["sum_force_n"],
                "target_contact_normal_force_estimated_n": contact["normal_force_estimated_n"],
                "target_contact_friction_force_estimated_n": contact["friction_force_estimated_n"],
                **{f"{group}_target_force_n": contact["group_force_norms"][group] for group in CONTACT_GROUPS},
            }
            trace.append(row)
            capture_replay(state)
            if step_index % telemetry_stride == 0:
                print(
                    f"\r[执行] {state:<14} {100.0*progress:5.1f}% | "
                    f"末端={position_error:5.1f}mm/{orientation_error:4.1f}° | "
                    f"抬升={object_lift:6.1f}mm | "
                    f"接触={contact['finger_count']}/5 max={contact['max_force_n']:6.2f}N",
                    end="", flush=True,
                )
            return position_error, orientation_error

        def execute_segment(state: str, arm_goal: np.ndarray | None, hand_goal: np.ndarray | None, duration_s: float, target_matrix: np.ndarray | None) -> None:
            nonlocal sim_time, current_stage
            current_stage = state
            start_arm = self.command[:, self.arm_ids].clone()
            start_hand = self.command[:, hand_ids].clone()
            if arm_goal is None:
                goal_arm = start_arm
                desired = self.current_desired_arm.clone()
            else:
                desired = torch.as_tensor(arm_goal, device=self.robot.device, dtype=self.command.dtype).reshape(1, 7)
                current_bias = self.command[:, self.arm_ids] - self.current_desired_arm
                goal_arm = desired + current_bias
            goal_hand = start_hand if hand_goal is None else torch.as_tensor(
                hand_goal, device=self.robot.device, dtype=self.command.dtype
            ).reshape(1, 20)
            count = max(2, round(float(duration_s) / self.dt))
            for i in range(count + 1):
                alpha = quintic(i / count)
                self.command[:, self.arm_ids] = start_arm + alpha * (goal_arm - start_arm)
                self.command[:, hand_ids] = start_hand + alpha * (goal_hand - start_hand)
                self.step(render=not bool(getattr(ARGS, "headless", False)))
                sim_time += self.dt
                self.current_desired_arm = desired
                monitor(state, i / count, target_matrix, i)
            print()

        def refine_exact_stage(state: str, stage_name: str, desired_arm: np.ndarray) -> tuple[bool, str | None]:
            nonlocal sim_time, current_stage
            current_stage = state
            if stage_name not in set(self.config.get("endpoint_refinement_stages", ["cover", "grasp"])):
                return True, None
            target_matrix = flange_targets[index_of[stage_name]]
            desired = torch.as_tensor(desired_arm, device=self.robot.device, dtype=self.command.dtype).reshape(1, 7)
            settings = self.config["endpoint_refinement"]
            bias = self.command[:, self.arm_ids].clone() - desired
            gain = float(settings["integral_gain_per_s"])
            max_bias = math.radians(float(settings["max_command_bias_deg"]))
            max_steps = max(1, round(float(settings["max_duration_s"]) / self.dt))
            stable_required = max(1, round(float(settings["stable_duration_s"]) / self.dt))
            stable = 0
            lower = self.robot.data.soft_joint_pos_limits[:, self.arm_ids, 0]
            upper = self.robot.data.soft_joint_pos_limits[:, self.arm_ids, 1]
            pos_limit = float(self.config["stage_tolerances"]["contact_position_mm"])
            rot_limit = float(self.config["stage_tolerances"]["contact_orientation_deg"])
            for i in range(max_steps):
                actual = self.robot.data.joint_pos[:, self.arm_ids]
                bias = torch.clamp(bias + gain * (desired - actual) * self.dt, -max_bias, max_bias)
                self.command[:, self.arm_ids] = torch.maximum(torch.minimum(desired + bias, upper), lower)
                self.step(render=not bool(getattr(ARGS, "headless", False)))
                sim_time += self.dt
                self.current_desired_arm = desired
                pos_err, rot_err = monitor(state, (i + 1) / max_steps, target_matrix, i)
                if pos_err <= pos_limit and rot_err <= rot_limit:
                    stable += 1
                    if stable >= stable_required:
                        print(f"\n[执行] ✓ {stage_name.upper()} 精确端点 {pos_err:.2f}mm/{rot_err:.2f}°")
                        return True, None
                else:
                    stable = 0
            pos_err, rot_err = pose_errors(self.robot.data.body_pose_w[0, self.flange_id], target_matrix)
            reason = f"{stage_name} exact endpoint failed: {pos_err:.2f}mm/{rot_err:.2f}deg"
            return False, reason

        def target_lift_mm_now() -> float:
            object_position = target_object.data.root_pos_w[0]
            return 1000.0 * float(object_position[2] - initial_object_position[2])

        def write_recovered_report(
            *,
            failure_stage: str,
            failure_type: str,
            failure_reason: str,
            recovery_status: str,
            verify_lift_mm: float | None,
        ) -> dict:
            capture_replay("RECOVERY_END", force=True)
            trace_path = output / "trace.csv"
            with trace_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
                writer.writeheader()
                writer.writerows(trace)

            replay_path = output / "physical_replay_30fps.npz"
            np.savez_compressed(
                replay_path,
                time_s=np.asarray(replay_time, dtype=np.float32),
                state=np.asarray(replay_state),
                joint_position_rad=np.stack(replay_joint),
                object_pose_world_wxyz=np.stack(replay_objects),
                metadata_json=np.asarray(json.dumps({
                    "schema_version": 2,
                    "persistent_session": True,
                    "joint_names": list(self.robot.joint_names),
                    "objects": [
                        {"segmentation_id": int(row["segmentation_id"])}
                        for row in self.object_records
                    ],
                    "candidate_index": source_candidate_index,
                    "recovered_failure": True,
                })),
            )
            self.execute_count += 1
            snapshot_path = output / "scene_after_execution.json"
            self.write_snapshot(snapshot_path)
            report = {
                "schema_version": 3,
                "status": "RECOVERED_FAIL",
                "persistent_session": True,
                "candidate_index": source_candidate_index,
                "target_segmentation_id": int(target_segmentation_id),
                "failure_stage": str(failure_stage),
                "failure_type": str(failure_type),
                "failure_reason": str(failure_reason),
                "recovery_attempted": True,
                "recovery_status": str(recovery_status),
                "current_target_lift_mm": float(target_lift_mm_now()),
                "verify_lift_mm": None if verify_lift_mm is None else float(verify_lift_mm),
                "max_object_lift_mm": float(max_object_lift_mm),
                "action_simulation_time_s": float(sim_time),
                "action_wall_time_s": float(time.perf_counter() - action_started),
                "post_home_hold_s_inside_execute": 0.0,
                "trace_csv": str(trace_path),
                "physical_replay_30fps": str(replay_path),
                "scene_after_execution": str(snapshot_path),
            }
            report_path = output / "report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.pause()
            return {**report, "report": str(report_path)}

        def recover_execution(
            *,
            failure_stage: str,
            failure_type: str,
            failure_reason: str,
            retreat_to_pregrasp: bool,
            verify_lift_mm: float | None = None,
        ) -> dict:
            nonlocal current_stage
            print(f"\n[EXECUTION ERROR] {failure_stage}: {failure_reason}", flush=True)
            print("[RECOVERY] 本轮停止后续动作，开始安全恢复。", flush=True)

            try:
                actual_arm = self.robot.data.joint_pos[:, self.arm_ids].clone()
                self.command[:, self.arm_ids] = actual_arm
                self.current_desired_arm = actual_arm.clone()

                execute_segment(
                    "RECOVERY_OPEN",
                    None,
                    hand_q5[hand_index["pregrasp"]],
                    float(durations.get("release", 1.2)),
                    None,
                )
                print("[RECOVERY] ✓ 手已张开", flush=True)

                if retreat_to_pregrasp:
                    execute_segment(
                        "RECOVERY_PREGRASP",
                        arm_q[index_of["pregrasp"]],
                        None,
                        float(durations.get("to_pregrasp", 2.5)),
                        flange_targets[index_of["pregrasp"]],
                    )
                    print("[RECOVERY] ✓ 已撤回 PREGRASP", flush=True)

                execute_segment(
                    "RECOVERY_HOME",
                    self.home_q,
                    None,
                    float(durations.get("return_home", 3.0)),
                    None,
                )

                home_actual = self.robot.data.joint_pos[:, self.arm_ids]
                home_target = torch.as_tensor(
                    self.home_q, device=self.robot.device, dtype=self.command.dtype
                ).reshape(1, 7)
                home_error_deg = float(
                    torch.max(torch.abs(torch.rad2deg(home_actual - home_target)))
                )
                home_tolerance_deg = float(
                    self.config.get("recovery_home_joint_tolerance_deg", 8.0)
                )
                if home_error_deg > home_tolerance_deg:
                    raise RuntimeError(
                        f"HOME recovery joint error {home_error_deg:.2f}deg "
                        f"> {home_tolerance_deg:.2f}deg"
                    )

                print(
                    f"[RECOVERY] ✓ 已返回 HOME | max joint error={home_error_deg:.2f}°",
                    flush=True,
                )
            except Exception as recovery_exc:
                print(
                    f"[RECOVERY ERROR] {type(recovery_exc).__name__}: {recovery_exc}",
                    flush=True,
                )
                raise RuntimeError(
                    f"recovery failed after {failure_type} at {failure_stage}: "
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                ) from recovery_exc

            current_stage = "RECOVERED_HOME"
            print("[RECOVERY] 本轮标记 RECOVERED_FAIL，Isaac 会话继续保留。", flush=True)
            return write_recovered_report(
                failure_stage=failure_stage,
                failure_type=failure_type,
                failure_reason=failure_reason,
                recovery_status="HOME",
                verify_lift_mm=verify_lift_mm,
            )

        capture_replay("ACTION_START", force=True)
        print(f"[执行] candidate={source_candidate_index}，直接使用规划阶段 q7（不再二次 IK）", flush=True)
        execute_segment("FORM_PREGRASP", None, hand_q5[hand_index["pregrasp"]], durations["form_pregrasp"], None)
        execute_segment("PREGRASP", arm_q[index_of["pregrasp"]], None, durations["to_pregrasp"], flange_targets[index_of["pregrasp"]])
        execute_segment("COVER", arm_q[index_of["cover"]], hand_q5[hand_index["cover"]], durations["cover"], flange_targets[index_of["cover"]])
        endpoint_ok, endpoint_reason = refine_exact_stage(
            "COVER_REFINE", "cover", arm_q[index_of["cover"]]
        )
        if not endpoint_ok:
            return recover_execution(
                failure_stage="COVER_REFINE",
                failure_type="ENDPOINT_ERROR",
                failure_reason=endpoint_reason or "COVER exact endpoint failed",
                retreat_to_pregrasp=True,
            )

        execute_segment("GRASP", arm_q[index_of["grasp"]], hand_q5[hand_index["grasp"]], durations["grasp"], flange_targets[index_of["grasp"]])
        endpoint_ok, endpoint_reason = refine_exact_stage(
            "GRASP_REFINE", "grasp", arm_q[index_of["grasp"]]
        )
        if not endpoint_ok:
            return recover_execution(
                failure_stage="GRASP_REFINE",
                failure_type="ENDPOINT_ERROR",
                failure_reason=endpoint_reason or "GRASP exact endpoint failed",
                retreat_to_pregrasp=True,
            )
        print_grasp_contact_audit("GRASP_END")

        print("[执行] SQUEEZE：41点 Wuji2 稠密收紧轨迹", flush=True)
        squeeze_duration = float(durations["squeeze"])
        steps_each = max(1, round(squeeze_duration / max(1, len(squeeze_dense) - 1) / self.dt))
        for path_index in range(1, len(squeeze_dense)):
            start_hand = self.command[:, hand_ids].clone()
            goal_hand = torch.as_tensor(
                squeeze_dense[path_index], device=self.robot.device, dtype=self.command.dtype
            ).reshape(1, 20)
            for local in range(steps_each):
                alpha = quintic((local + 1) / steps_each)
                self.command[:, hand_ids] = start_hand + alpha * (goal_hand - start_hand)
                desired = torch.as_tensor(
                    arm_q[index_of["squeeze"]], device=self.robot.device, dtype=self.command.dtype
                ).reshape(1, 7)
                current_bias = self.command[:, self.arm_ids] - self.current_desired_arm
                self.command[:, self.arm_ids] = desired + current_bias
                self.current_desired_arm = desired
                self.step(render=not bool(getattr(ARGS, "headless", False)))
                sim_time += self.dt
                monitor(
                    "SQUEEZE",
                    (path_index - 1 + (local + 1) / steps_each) / max(1, len(squeeze_dense) - 1),
                    flange_targets[index_of["squeeze"]],
                    (path_index - 1) * steps_each + local,
                )
        print()

        # SQUEEZE is a persistent closing command. Finger target error is
        # expected when the object blocks closure; VERIFY_LIFT decides success.
        squeeze_hold_s = float(durations.get("squeeze_hold", 0.4))
        squeeze_hold_steps = max(1, round(squeeze_hold_s / self.dt))
        squeeze_target = torch.as_tensor(
            hand_q5[hand_index["squeeze"]],
            device=self.robot.device,
            dtype=self.command.dtype,
        ).reshape(1, 20)
        self.command[:, hand_ids] = squeeze_target
        current_stage = "SQUEEZE_HOLD"
        for hold_index in range(squeeze_hold_steps):
            self.command[:, hand_ids] = squeeze_target
            self.step(render=not bool(getattr(ARGS, "headless", False)))
            sim_time += self.dt
            monitor(
                "SQUEEZE_HOLD",
                (hold_index + 1) / squeeze_hold_steps,
                flange_targets[index_of["squeeze"]],
                hold_index,
            )
        print()
        print_grasp_contact_audit("SQUEEZE_HOLD_END")

        execute_segment("LIFT", arm_q[index_of["lift"]], hand_q5[hand_index["squeeze"]], durations["lift"], flange_targets[index_of["lift"]])
        self.hold(float(durations.get("lift_hold", 0.4)), render=not bool(getattr(ARGS, "headless", False)))
        sim_time += float(durations.get("lift_hold", 0.4))
        print_grasp_contact_audit("LIFT_END")

        verified_target_lift_mm = target_lift_mm_now()
        lift_threshold_mm = float(self.config.get("object_lift_pass_mm", 30.0))
        if verified_target_lift_mm < lift_threshold_mm:
            reason = (
                f"target lift={verified_target_lift_mm:.1f}mm "
                f"< {lift_threshold_mm:.1f}mm"
            )
            print(f"[VERIFY] ✗ EMPTY_GRASP | {reason}", flush=True)
            return recover_execution(
                failure_stage="VERIFY_LIFT",
                failure_type="EMPTY_GRASP",
                failure_reason=reason,
                retreat_to_pregrasp=False,
                verify_lift_mm=verified_target_lift_mm,
            )

        print(
            f"[VERIFY] ✓ target follows lift | "
            f"{verified_target_lift_mm:.1f}mm >= {lift_threshold_mm:.1f}mm",
            flush=True,
        )
        execute_segment("TRANSFER", arm_q[index_of["transfer"]], None, durations["transfer"], flange_targets[index_of["transfer"]])
        execute_segment("PLACE", arm_q[index_of["place"]], None, durations["place"], flange_targets[index_of["place"]])
        self.hold(float(durations.get("place_hold", 0.4)), render=not bool(getattr(ARGS, "headless", False)))
        sim_time += float(durations.get("place_hold", 0.4))
        execute_segment("RELEASE", arm_q[index_of["release"]], hand_q5[hand_index["pregrasp"]], durations["release"], flange_targets[index_of["release"]])
        self.hold(float(durations.get("release_hold", 0.4)), render=not bool(getattr(ARGS, "headless", False)))
        sim_time += float(durations.get("release_hold", 0.4))
        execute_segment("RETREAT", arm_q[index_of["retreat"]], None, durations["retreat"], flange_targets[index_of["retreat"]])
        execute_segment("RETURN_HOME", self.home_q, None, durations["return_home"], None)
        # The next CAPTURE command owns the single configured HOME settle.
        # Do not wait here, otherwise every cycle would silently hold twice.
        post_home_hold = 0.0
        capture_replay("ACTION_END", force=True)

        final_position = target_object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        zone = load_json(PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
        centre = np.asarray(zone["transforms"]["placement_zone"]["position_world_m"], dtype=np.float64)
        size = np.asarray(zone["geometry"]["placement_zone_size_m"], dtype=np.float64)
        zone_min = centre[:2] - 0.5 * size[:2]
        zone_max = centre[:2] + 0.5 * size[:2]
        edge = float(self.config.get("placement_center_edge_margin_m", 0.01))
        final_in_green = bool(np.all(final_position[:2] >= zone_min + edge) and np.all(final_position[:2] <= zone_max - edge))
        lift_threshold_mm = float(self.config.get("object_lift_pass_mm", 30.0))
        lift_pass = bool(
            verified_target_lift_mm is not None
            and verified_target_lift_mm >= lift_threshold_mm
        )
        passed = bool(lift_pass and final_in_green)
        wall_s = time.perf_counter() - action_started

        trace_path = output / "trace.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            writer.writerows(trace)
        replay_path = output / "physical_replay_30fps.npz"
        np.savez_compressed(
            replay_path,
            time_s=np.asarray(replay_time, dtype=np.float32),
            state=np.asarray(replay_state),
            joint_position_rad=np.stack(replay_joint),
            object_pose_world_wxyz=np.stack(replay_objects),
            metadata_json=np.asarray(json.dumps({
                "schema_version": 2,
                "persistent_session": True,
                "joint_names": list(self.robot.joint_names),
                "objects": [{"segmentation_id": int(row["segmentation_id"])} for row in self.object_records],
                "candidate_index": source_candidate_index,
            })),
        )
        self.execute_count += 1
        snapshot_path = output / "scene_after_execution.json"
        self.write_snapshot(snapshot_path)
        report = {
            "schema_version": 2,
            "status": "PASS" if passed else "FAIL",
            "persistent_session": True,
            "second_ik_performed": False,
            "fk_precheck_performed": False,
            "candidate_index": source_candidate_index,
            "target_segmentation_id": int(target_segmentation_id),
            "max_object_lift_mm": float(max_object_lift_mm),
            "verify_lift_mm": None if verified_target_lift_mm is None else float(verified_target_lift_mm),
            "current_target_lift_mm": float(target_lift_mm_now()),
            "object_lift_pass": lift_pass,
            "final_object_position_world_m": final_position.tolist(),
            "final_object_center_inside_green_zone": final_in_green,
            "action_simulation_time_s": float(sim_time),
            "action_wall_time_s": float(wall_s),
            "post_home_hold_s_inside_execute": post_home_hold,
            "trace_csv": str(trace_path),
            "physical_replay_30fps": str(replay_path),
            "scene_after_execution": str(snapshot_path),
        }
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"[执行] {'✓ PASS' if passed else '✗ FAIL'} | 抬升={max_object_lift_mm:.1f}mm | "
            f"绿色区域={final_in_green} | 下一次拍照前静置由 CAPTURE 统一执行",
            flush=True,
        )
        self.pause()
        return {**report, "report": str(report_path)}


def main() -> int:
    config = load_json(ARGS.config)
    scene = PersistentScene(ARGS.scene_manifest, config)
    if not ARGS.stdio:
        print("Persistent Isaac worker ready")
        return 0
    selector = selectors.DefaultSelector()
    selector.register(sys.stdin, selectors.EVENT_READ)
    while True:
        events = selector.select(timeout=0.05)
        if not events:
            # Keep the GUI responsive while planning runs in other processes.
            # SimulationContext is deliberately NOT stepped here, so the camera
            # scene/q_current used for planning cannot drift before execution.
            if not bool(getattr(ARGS, "headless", False)):
                SIMULATION_APP.update()
            continue
        line = sys.stdin.readline()
        if line == "":
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "ping":
                emit({
                    "ok": True,
                    "op": "pong",
                    "persistent_session": True,
                    "physics_frozen_while_planning": True,
                    "joint_count": int(scene.robot.num_joints),
                    "object_count": len(scene.objects),
                    "home_q_deg": np.degrees(scene.home_q).tolist(),
                })
            elif op == "capture":
                result = scene.capture(
                    Path(req["output_dir"]),
                    None if "hold_s" not in req else float(req["hold_s"]),
                )
                emit({"ok": True, "op": "capture", **result})
            elif op == "execute":
                result = scene.execute(
                    case_root=Path(req["case_root"]),
                    plan_npz=Path(req["plan_npz"]),
                    output_dir=Path(req["output_dir"]),
                    target_segmentation_id=int(req["target_segmentation_id"]),
                )
                emit({"ok": True, "op": "execute", **result})
            elif op == "snapshot":
                path = scene.write_snapshot(Path(req["output"]))
                emit({"ok": True, "op": "snapshot", "output": str(path)})
            elif op == "shutdown":
                emit({"ok": True, "op": "shutdown"})
                return 0
            else:
                raise ValueError(f"unknown op: {op}")
        except Exception as exc:
            try:
                scene.pause()
            except Exception:
                pass
            traceback.print_exc()
            emit({"ok": False, "op": req.get("op") if "req" in locals() else None, "error": f"{type(exc).__name__}: {exc}"})


try:
    raise SystemExit(main())
finally:
    SIMULATION_APP.close()
